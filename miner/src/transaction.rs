//! Transactions.
//!
//! An output pays to a [`KeyTree`]: the Merkle root of [`KEY_LEAVES`]
//! leaves, each a way to spend it. A leaf is either a one-time WOTS key's
//! hash (the standard output: any key spends it, and a spend that needs
//! replacing, for a higher fee or a cancellation, is signed again with
//! another leaf, so no key ever signs twice) or a [`Condition`]'s hash.
//! Nothing on chain tells the two apart: a spend reveals its leaf only to
//! the block's proof.
//!
//! An [`Input`] names the commitment it spends and its position, the amount
//! and nonce behind it, and how it's spent ([`Spend`]): a key and its path,
//! or a condition, its path and its signers. A key spend signs the digest
//! of the whole [`TxBody`] ([`TxBody::digest`]); a condition's signers
//! sign the message its [`SigMode`] gives. The fee is implicit:
//! `sum(inputs) - sum(outputs)`.
//!
//! A condition needs `k` of up to [`MAX_SIGNERS`] keys to sign (a
//! threshold), each key a key tree of its owner's chosen depth (up to
//! [`MAX_KEY_DEPTH`]); it can
//! require each signing key's index to exceed a state number, which, with a
//! key per state, lets a later state's transaction replace an earlier one
//! (payment channels). It can also lock the output until a height
//! (absolute, `after_height`) or for a number of blocks after it was
//! created (relative, `after_blocks`), and require a hash's preimage,
//! which the spending transaction must publish as the start of one of its
//! outputs' nonces (an output per hashlock input), so whoever watches the
//! chain learns it.
//!
//! A condition can also protect its members: each key comes with a payout
//! (an output's commitment, which its member made), and when fewer than
//! all of them sign, the spending transaction must create the payout of
//! each member that doesn't. With a state number, any `k` members can then
//! move a shared output on without the others, who keep what the last
//! state they signed gave them. Protection needs `after_blocks`, so a
//! published old state can be replaced before a quorum spends it.
//!
//! [`Transaction::validate`] performs every check that needs no chain state
//! and returns the [`Totals`]. It does not require inputs to cover outputs, so
//! a block reward (outputs, no inputs) validates too. The caller must check
//! that each input is unspent and that its timelocks have passed
//! ([`Transaction::check_locks`]), and the block must bound how much its
//! transactions mint.
//!
//! Transactions travel between nodes (and stay out of blocks, which publish
//! only their lists) in the [`Transaction::to_bytes`] encoding; a
//! transaction's id ([`Transaction::id_of`]) is the chunk hash of that
//! encoding, so its chunks can be checked as they arrive.

use std::fmt;

use crate::babybear::{BabyBear, P};
use crate::chunk;
use crate::hash::{self, DOMAIN_SLOT, HASH_BYTES, HASH_LEN, Hash, domain};
use crate::output::{
    self, Amount, COMMITMENT_LEN, Commitment, NONCE_LEN, NONCE_LIMB_BYTES, Nonce, OUTPUT_HASH_LEN,
    Output, OutputHash,
};
use crate::poseidon2::{self, WIDTH};
use crate::wots::{self, NUM_CHAINS, PARAM_LEN, PublicKey, RAND_LEN, SecretKey, Signature};

/// Levels of an output's key tree.
pub const KEY_LEVELS: usize = 8;
/// One-time keys an output can be spent with: every leaf byte is a leaf.
pub const KEY_LEAVES: usize = 1 << KEY_LEVELS;
const _: () = assert!(KEY_LEAVES == 1 << u8::BITS);
/// A leaf's siblings, bottom up.
pub type KeyPath = [Hash; KEY_LEVELS];

/// The deepest key tree a condition's signer may use: `2^24` one-time
/// keys, so up to `2^24` states.
pub const MAX_KEY_DEPTH: usize = 24;
/// The most keys a condition may have, and the levels of their set.
pub const MAX_SIGNERS: usize = 1 << KEY_SET_LEVELS;
pub const KEY_SET_LEVELS: usize = 4;
/// The largest state number a condition may name (a signer's key index
/// must exceed it).
pub const MAX_STATE: u32 = (1 << MAX_KEY_DEPTH) - 1;
/// Timelocks are heights and block counts below `2^26`
/// ([`crate::header::MAX_HEIGHT_BITS`]).
pub const MAX_LOCK: u32 = (1 << crate::header::MAX_HEIGHT_BITS) - 1;
/// Bytes in a hashlock's preimage: the first five nonce limbs.
pub const PREIMAGE_LEN: usize = 5 * NONCE_LIMB_BYTES;
pub type Preimage = [u8; PREIMAGE_LEN];

/// Bytes in a transaction's header: input, output and signed-output
/// counts (`u16`s).
pub const TX_HEADER_BYTES: usize = 6;
/// Bytes in an encoded key spend input: commitment, position, amount,
/// nonce, kind, public key (parameter, root), leaf, path.
pub const INPUT_BYTES: usize =
    COMMITMENT_LEN + 8 + 8 + NONCE_LEN + 1 + PUBKEY_BYTES + 1 + KEY_LEVELS * HASH_BYTES;
const PUBKEY_BYTES: usize = PARAM_LEN * 4 + HASH_BYTES;
/// Bytes in an encoded output: hash, amount, nonce.
pub const OUTPUT_BYTES: usize = OUTPUT_HASH_LEN + 8 + NONCE_LEN;
/// Bytes in an encoded signature: randomizer, chains.
pub const SIGNATURE_BYTES: usize = RAND_LEN * 4 + NUM_CHAINS * HASH_BYTES;
/// The largest transaction nodes relay.
pub const MAX_TX_BYTES: usize = 100_000;

/// An output's key tree: its hash is the Merkle root ([`hash::merge`]; the
/// depth is fixed) of these leaves, each a one-time key's hash
/// (`H(pubkey)`) or a [`Condition`]'s ([`Condition::leaf`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyTree(pub [Hash; KEY_LEAVES]);

impl KeyTree {
    /// A tree whose only usable key is `pubkey`, at leaf 0 (for tests).
    pub fn single(pubkey: &PublicKey) -> Self {
        let mut leaves = [[BabyBear::ZERO; HASH_LEN]; KEY_LEAVES];
        leaves[0] = pubkey.hash();
        Self(leaves)
    }

    /// A tree of conditions, in leaf order (the rest left empty).
    ///
    /// # Panics
    /// If there are more than [`KEY_LEAVES`].
    pub fn of_conditions(conditions: &[Condition]) -> Self {
        assert!(conditions.len() <= KEY_LEAVES, "too many conditions");
        let mut leaves = [[BabyBear::ZERO; HASH_LEN]; KEY_LEAVES];
        for (l, c) in leaves.iter_mut().zip(conditions) {
            *l = c.leaf();
        }
        Self(leaves)
    }

    pub fn root(&self) -> Hash {
        let mut level = self.0.to_vec();
        while level.len() > 1 {
            level = level
                .chunks_exact(2)
                .map(|p| hash::merge(&p[0], &p[1]))
                .collect();
        }
        level[0]
    }

    /// The hash an output paying to these keys has.
    pub fn output_hash(&self) -> OutputHash {
        hash::to_bytes(&self.root())
    }

    /// Leaf `leaf`'s siblings, bottom up.
    pub fn path(&self, leaf: usize) -> KeyPath {
        let mut level = self.0.to_vec();
        let mut i = leaf;
        core::array::from_fn(|_| {
            let sibling = level[i ^ 1];
            level = level
                .chunks_exact(2)
                .map(|p| hash::merge(&p[0], &p[1]))
                .collect();
            i >>= 1;
            sibling
        })
    }
}

/// The root reached from key hash `pkh` at `leaf` along `path`.
pub fn key_root(pkh: &Hash, leaf: u8, path: &KeyPath) -> Hash {
    tree_root(pkh, leaf as u32, path)
}

/// The root reached from `node` at `index` along `path` (bottom up), of
/// any depth.
pub fn tree_root(node: &Hash, index: u32, path: &[Hash]) -> Hash {
    path.iter().enumerate().fold(*node, |h, (d, s)| {
        if index >> d & 1 == 0 {
            hash::merge(&h, s)
        } else {
            hash::merge(s, &h)
        }
    })
}

/// A signer's key tree: `2^depth` one-time keys' hashes, whose root a
/// [`Condition`] names. The index of the key that signs can be a state
/// number ([`Condition::min_state`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignerTree {
    /// Every level, leaves first, root last.
    levels: Vec<Vec<Hash>>,
}

impl SignerTree {
    /// The tree over `leaves` (a power of two of them, at most
    /// `2^MAX_KEY_DEPTH`).
    ///
    /// # Panics
    /// If the leaf count isn't a power of two or is too large.
    pub fn new(leaves: Vec<Hash>) -> Self {
        assert!(
            leaves.len().is_power_of_two() && leaves.len() <= 1 << MAX_KEY_DEPTH,
            "a signer tree has a power of two of keys, at most 2^{MAX_KEY_DEPTH}"
        );
        let mut levels = vec![leaves];
        while levels.last().expect("a level").len() > 1 {
            let next = levels
                .last()
                .expect("a level")
                .chunks_exact(2)
                .map(|p| hash::merge(&p[0], &p[1]))
                .collect();
            levels.push(next);
        }
        Self { levels }
    }

    pub fn depth(&self) -> usize {
        self.levels.len() - 1
    }

    /// The keys' hashes, by index.
    pub fn leaves(&self) -> &[Hash] {
        &self.levels[0]
    }

    pub fn root(&self) -> Hash {
        self.levels.last().expect("a level")[0]
    }

    /// Key `index`'s siblings, bottom up.
    pub fn path(&self, index: u32) -> Vec<Hash> {
        self.levels[..self.depth()]
            .iter()
            .enumerate()
            .map(|(d, level)| level[(index >> d) as usize ^ 1])
            .collect()
    }
}

/// What a condition's signatures cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigMode {
    /// The whole transaction ([`TxBody::digest`]), as key spends sign.
    All = 0,
    /// This input (its commitment and position) and the transaction's
    /// signed outputs ([`TxBody::signed_outputs`]): others can add inputs
    /// and outputs, say to pay a fee.
    AnyoneCanPay = 1,
    /// Only this input's amount and the signed outputs: the signature fits
    /// any output with this condition and amount, so a later state can
    /// spend whichever earlier one was published.
    AnyPrevout = 2,
}

impl SigMode {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::All),
            1 => Some(Self::AnyoneCanPay),
            2 => Some(Self::AnyPrevout),
            _ => None,
        }
    }
}

/// A spending condition: a key-tree leaf that releases the output when
/// all of its rules are met.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Condition {
    /// The signers' key-tree roots ([`SignerTree::root`]; a single key's
    /// hash is a tree of depth 0): one to [`MAX_SIGNERS`], committed as a
    /// set ([`Condition::key_set`]).
    pub keys: Vec<Hash>,
    /// How many of them must sign (from 1 to their number).
    pub threshold: usize,
    /// If set, each signing key's index must be above this.
    pub min_state: Option<u32>,
    pub mode: SigMode,
    /// The spend's block must be at least this high (0: no lock).
    pub after_height: u32,
    /// And at least this many blocks above the output's (0: none).
    pub after_blocks: u32,
    /// If set, the spending transaction must publish a preimage of this
    /// ([`hashlock`]).
    pub hashlock: Option<Hash>,
    /// If set (protection), a payout commitment per key: at least the
    /// threshold must sign, and the spending transaction must create the
    /// payout of each key that doesn't ([`TxBody::payouts`]), so members
    /// who don't sign keep what the state gave them. Needs `after_blocks`.
    pub payouts: Option<Vec<Hash>>,
}

impl Condition {
    /// A condition of one key and nothing else.
    pub fn key(root: Hash) -> Self {
        Self::all_of(vec![root])
    }

    /// A condition all of `keys` must sign, and nothing else.
    pub fn all_of(keys: Vec<Hash>) -> Self {
        Self {
            threshold: keys.len(),
            keys,
            min_state: None,
            mode: SigMode::All,
            after_height: 0,
            after_blocks: 0,
            hashlock: None,
            payouts: None,
        }
    }

    /// The flags its leaf hashes: a state rule (2), a hashlock (4),
    /// protection (8).
    fn flags(&self) -> u32 {
        (self.min_state.is_some() as u32) << 1
            | (self.hashlock.is_some() as u32) << 2
            | (self.payouts.is_some() as u32) << 3
    }

    /// Member `m`'s payout commitment (zero without protection).
    pub fn payout(&self, m: usize) -> Hash {
        self.payouts
            .as_ref()
            .and_then(|p| p.get(m).copied())
            .unwrap_or([BabyBear::ZERO; HASH_LEN])
    }

    /// The key set's leaves: `merge(key, payout)` per key, then zeros up to
    /// [`MAX_SIGNERS`] (no key tree has a zero root).
    fn key_set_leaves(&self) -> [Hash; MAX_SIGNERS] {
        core::array::from_fn(|i| match self.keys.get(i) {
            Some(k) => hash::merge(k, &self.payout(i)),
            None => [BabyBear::ZERO; HASH_LEN],
        })
    }

    /// The key set: the Merkle root ([`hash::merge`], [`KEY_SET_LEVELS`]
    /// levels) of each key merged with its payout, padded with zeros.
    pub fn key_set(&self) -> Hash {
        let mut level = self.key_set_leaves().to_vec();
        while level.len() > 1 {
            level = level
                .chunks_exact(2)
                .map(|p| hash::merge(&p[0], &p[1]))
                .collect();
        }
        level[0]
    }

    /// Member `member`'s siblings in the key set, bottom up.
    pub fn key_set_path(&self, member: usize) -> [Hash; KEY_SET_LEVELS] {
        let mut level = self.key_set_leaves().to_vec();
        let mut i = member;
        core::array::from_fn(|_| {
            let sibling = level[i ^ 1];
            level = level
                .chunks_exact(2)
                .map(|p| hash::merge(&p[0], &p[1]))
                .collect();
            i >>= 1;
            sibling
        })
    }

    /// Its key-tree leaf: `sponge(CONDITION, [min state, after height,
    /// after blocks, mode, flags], [key set, hashlock, threshold, key
    /// count])`, with zeros for what's absent.
    pub fn leaf(&self) -> Hash {
        let zero = [BabyBear::ZERO; HASH_LEN];
        hash::sponge(
            domain::CONDITION,
            &[
                BabyBear::new(self.min_state.unwrap_or(0)),
                BabyBear::new(self.after_height),
                BabyBear::new(self.after_blocks),
                BabyBear::new(self.mode as u32),
                BabyBear::new(self.flags()),
            ],
            &[
                &self.key_set(),
                self.hashlock.as_ref().unwrap_or(&zero),
                &[
                    BabyBear::new(self.threshold as u32),
                    BabyBear::new(self.keys.len() as u32),
                ],
            ],
        )
    }

    /// Its encoding: key count and keys, threshold, mode, flags (bit 0:
    /// state rule, bit 1: hashlock, bit 2: protection), state number, after
    /// height, after blocks (`u32`s, little-endian), the hashlock if any,
    /// the payouts if protected.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    /// Decodes exactly one [`Condition::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader(bytes);
        let c = Self::read(&mut r)?;
        r.0.is_empty().then_some(c)
    }

    fn write(&self, out: &mut Vec<u8>) {
        out.push(self.keys.len() as u8);
        for k in &self.keys {
            put_elems(out, k);
        }
        out.push(self.threshold as u8);
        out.push(self.mode as u8);
        out.push((self.flags() >> 1) as u8);
        out.extend(self.min_state.unwrap_or(0).to_le_bytes());
        out.extend(self.after_height.to_le_bytes());
        out.extend(self.after_blocks.to_le_bytes());
        if let Some(h) = &self.hashlock {
            put_elems(out, h);
        }
        for p in self.payouts.iter().flatten() {
            put_elems(out, p);
        }
    }

    fn read(r: &mut Reader<'_>) -> Option<Self> {
        let n_keys = r.u8()? as usize;
        if !(1..=MAX_SIGNERS).contains(&n_keys) {
            return None;
        }
        let keys = (0..n_keys).map(|_| r.hash()).collect::<Option<Vec<_>>>()?;
        let threshold = r.u8()? as usize;
        let mode = SigMode::from_u8(r.u8()?)?;
        let flags = r.u8()?;
        if flags >> 3 != 0 {
            return None;
        }
        let state = r.u32()?;
        let min_state = (flags & 1 == 1).then_some(state);
        if min_state.is_none() && state != 0 {
            return None;
        }
        let after_height = r.u32()?;
        let after_blocks = r.u32()?;
        let hashlock = if flags & 2 == 2 {
            Some(r.hash()?)
        } else {
            None
        };
        let payouts = if flags & 4 == 4 {
            Some((0..n_keys).map(|_| r.hash()).collect::<Option<Vec<_>>>()?)
        } else {
            None
        };
        Some(Self {
            keys,
            threshold,
            min_state,
            mode,
            after_height,
            after_blocks,
            hashlock,
            payouts,
        })
    }

    /// Whether its values are in range: one to [`MAX_SIGNERS`] keys (none
    /// zero), a threshold from 1 to their number, a state number up to
    /// [`MAX_STATE`], locks up to [`MAX_LOCK`], and with protection a
    /// payout per key and a block count of at least one (else a quorum
    /// could spend an old state before the others can replace it).
    pub fn is_well_formed(&self) -> bool {
        (1..=MAX_SIGNERS).contains(&self.keys.len())
            && (1..=self.keys.len()).contains(&self.threshold)
            && self.keys.iter().all(|k| *k != [BabyBear::ZERO; HASH_LEN])
            && self.min_state.is_none_or(|m| m <= MAX_STATE)
            && self.after_height <= MAX_LOCK
            && self.after_blocks <= MAX_LOCK
            && self
                .payouts
                .as_ref()
                .is_none_or(|p| p.len() == self.keys.len() && self.after_blocks >= 1)
    }

    /// Whether `signers` (members, in order) are enough: exactly the
    /// threshold, or with protection at least it.
    pub fn enough_signers(&self, signers: usize) -> bool {
        if self.payouts.is_some() {
            (self.threshold..=self.keys.len()).contains(&signers)
        } else {
            signers == self.threshold
        }
    }
}

/// The hash a hashlock names: `sponge(HASHLOCK, [], [preimage limbs])`.
pub fn hashlock(preimage: &Preimage) -> Hash {
    hash::sponge(domain::HASHLOCK, &[], &[&preimage_limbs(preimage)])
}

/// A preimage as the nonce limbs it fills.
pub fn preimage_limbs(preimage: &Preimage) -> [BabyBear; PREIMAGE_LEN / NONCE_LIMB_BYTES] {
    let mut nonce = [0; NONCE_LEN];
    nonce[..PREIMAGE_LEN].copy_from_slice(preimage);
    core::array::from_fn(|j| output::nonce_limbs(&nonce)[j])
}

/// One of a condition's signers: which of its keys (`member`), the key
/// that signs, its index (state) and its path in that key's tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signer {
    pub member: u8,
    pub pubkey: PublicKey,
    pub index: u32,
    /// Siblings, bottom up: the tree's depth.
    pub path: Vec<Hash>,
}

impl Signer {
    /// The key tree's root.
    pub fn root(&self) -> Hash {
        tree_root(&self.pubkey.hash(), self.index, &self.path)
    }

    /// Whether the index fits the path's depth (at most
    /// [`MAX_KEY_DEPTH`]).
    fn is_well_formed(&self) -> bool {
        self.path.len() <= MAX_KEY_DEPTH && (self.index as u64) < 1 << self.path.len()
    }
}

/// How an input is spent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Spend {
    /// By a one-time key at `leaf` of the output's key tree.
    Key {
        pubkey: PublicKey,
        leaf: u8,
        path: KeyPath,
    },
    /// By meeting the condition at `leaf`: as many signers as its
    /// threshold (with protection, at least as many), of distinct keys in
    /// key order, and the preimage of its hashlock, if it has one (which an
    /// output of the transaction must publish: [`TxBody::publishers`]).
    Condition {
        condition: Condition,
        leaf: u8,
        path: KeyPath,
        signers: Vec<Signer>,
        preimage: Option<Preimage>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// Commitment of the output being spent.
    pub commitment: Commitment,
    /// Its slot in the state tree: what makes the spent output unique, so
    /// a signature can't be replayed against an identical output created
    /// later (which lands in a new slot).
    pub position: u64,
    pub amount: Amount,
    pub nonce: Nonce,
    pub spend: Spend,
}

impl Input {
    /// The output hash the key (or condition) and path lead to.
    pub fn key_root(&self) -> Hash {
        match &self.spend {
            Spend::Key { pubkey, leaf, path } => key_root(&pubkey.hash(), *leaf, path),
            Spend::Condition {
                condition,
                leaf,
                path,
                ..
            } => key_root(&condition.leaf(), *leaf, path),
        }
    }

    /// Returns `true` if the revealed spend, amount and nonce open
    /// `commitment`.
    pub fn opens_commitment(&self) -> bool {
        output::commit_field(&self.key_root(), self.amount, &self.nonce) == self.commitment
    }

    /// The signatures it needs.
    pub fn signature_count(&self) -> usize {
        match &self.spend {
            Spend::Key { .. } => 1,
            Spend::Condition { signers, .. } => signers.len(),
        }
    }

    /// The leaf of its key, if it's a key spend.
    pub fn key_leaf(&self) -> Option<u8> {
        match self.spend {
            Spend::Key { leaf, .. } => Some(leaf),
            Spend::Condition { .. } => None,
        }
    }

    /// Its condition, if it spends one.
    pub fn condition(&self) -> Option<&Condition> {
        match &self.spend {
            Spend::Condition { condition, .. } => Some(condition),
            Spend::Key { .. } => None,
        }
    }

    /// The public keys that sign it, in order.
    pub fn pubkeys(&self) -> Vec<&PublicKey> {
        match &self.spend {
            Spend::Key { pubkey, .. } => vec![pubkey],
            Spend::Condition { signers, .. } => signers.iter().map(|s| &s.pubkey).collect(),
        }
    }
}

/// Everything the signatures commit to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TxBody {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    /// How many of the outputs, from the first, anyone-can-pay and
    /// any-prevout signatures cover. Only one such input may sign them
    /// (and it must come before any other condition input), so no two
    /// share the same outputs.
    pub signed_outputs: usize,
}

impl TxBody {
    /// A body of `inputs` and `outputs`, with no signed outputs.
    pub fn new(inputs: Vec<Input>, outputs: Vec<Output>) -> Self {
        Self {
            inputs,
            outputs,
            ..Self::default()
        }
    }

    /// The message key spends (and [`SigMode::All`] conditions) sign: a
    /// sponge (domain `TX_BODY`, with `n_in, n_out` and the absorbed length
    /// in the capacity) over each input's commitment, then its position
    /// (two limbs, zero-padded to a block), then each output's commitment.
    /// Each commitment binds its amount, nonce and hash (for an input, the
    /// key tree's root), the position binds which output instance is
    /// spent, and the signature checks the pubkey itself, so this covers
    /// every input and output field. `None` if a commitment or output hash
    /// is not a canonical encoding.
    pub fn digest(&self) -> Option<Hash> {
        let inputs = self
            .inputs
            .iter()
            .map(|i| Some((i.commitment.to_field()?, i.position)))
            .collect::<Option<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(|o| o.commitment()?.to_field())
            .collect::<Option<Vec<_>>>()?;
        Some(digest_of(&inputs, &outputs))
    }

    /// The signed outputs' digest: `sponge(SIGNED_OUTPUTS, [n], [their
    /// commitments])`, or zero if there are none. `None` if a commitment
    /// isn't canonical or there aren't that many outputs.
    pub fn signed_digest(&self) -> Option<Hash> {
        let signed = self.outputs.get(..self.signed_outputs)?;
        let commitments = signed
            .iter()
            .map(|o| o.commitment()?.to_field())
            .collect::<Option<Vec<_>>>()?;
        Some(signed_digest_of(&commitments))
    }

    /// The message input `i`'s signatures sign. `None` as for
    /// [`TxBody::digest`] and [`TxBody::signed_digest`].
    pub fn message(&self, i: usize) -> Option<Hash> {
        let input = &self.inputs[i];
        match input.condition().map(|c| c.mode) {
            None | Some(SigMode::All) => self.digest(),
            Some(SigMode::AnyoneCanPay) => Some(acp_message(
                &self.signed_digest()?,
                &input.commitment.to_field()?,
                input.position,
            )),
            Some(SigMode::AnyPrevout) => Some(apo_message(&self.signed_digest()?, input.amount)),
        }
    }

    /// The outputs publishing the hashlock inputs' preimages, by input:
    /// each takes the first output not taken yet whose nonce starts with
    /// its preimage. `None` if one has none.
    pub fn publishers(&self) -> Option<Vec<Option<usize>>> {
        let mut taken = vec![false; self.outputs.len()];
        let mut out = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let Spend::Condition {
                preimage: Some(p), ..
            } = &input.spend
            else {
                out.push(None);
                continue;
            };
            let o = (0..self.outputs.len())
                .find(|&o| !taken[o] && self.outputs[o].nonce[..PREIMAGE_LEN] == p[..])?;
            taken[o] = true;
            out.push(Some(o));
        }
        Some(out)
    }

    /// The commitments protected inputs must pay, in input order: each
    /// non-signing member's payout.
    pub fn required_payouts(&self) -> Vec<Hash> {
        let mut out = Vec::new();
        for input in &self.inputs {
            let Spend::Condition {
                condition: c,
                signers,
                ..
            } = &input.spend
            else {
                continue;
            };
            let Some(payouts) = &c.payouts else {
                continue;
            };
            for (m, p) in payouts.iter().enumerate() {
                if !signers.iter().any(|s| s.member as usize == m) {
                    out.push(*p);
                }
            }
        }
        out
    }

    /// The outputs paying [`TxBody::required_payouts`], in that order: each
    /// the first output with that commitment not taken yet, by another
    /// payout or by a preimage ([`TxBody::publishers`]). `None` if one has
    /// none.
    pub fn payouts(&self) -> Option<Vec<usize>> {
        let required = self.required_payouts();
        if required.is_empty() {
            return Some(Vec::new());
        }
        let mut taken = vec![false; self.outputs.len()];
        for o in self.publishers()?.into_iter().flatten() {
            taken[o] = true;
        }
        let commitments = self
            .outputs
            .iter()
            .map(|o| o.commitment()?.to_field())
            .collect::<Option<Vec<_>>>()?;
        let mut out = Vec::new();
        for p in required {
            let o = (0..self.outputs.len()).find(|&o| !taken[o] && commitments[o] == p)?;
            taken[o] = true;
            out.push(o);
        }
        Some(out)
    }

    /// Every signature's message, in signature order.
    pub fn messages(&self) -> Option<Vec<Hash>> {
        let mut out = Vec::new();
        for (i, input) in self.inputs.iter().enumerate() {
            let m = self.message(i)?;
            out.extend(std::iter::repeat_n(m, input.signature_count()));
        }
        Some(out)
    }
}

/// [`TxBody::digest`] from the inputs' commitments and positions and the
/// outputs' commitments, as field elements.
pub(crate) fn digest_of(inputs: &[(Hash, u64)], outputs: &[Hash]) -> Hash {
    {
        let (nin, nout) = (count(inputs.len()), count(outputs.len()));
        let mut state = [BabyBear::ZERO; WIDTH];
        state[HASH_LEN] = BabyBear::new(nin);
        state[HASH_LEN + 1] = BabyBear::new(nout);
        state[DOMAIN_SLOT] = BabyBear::new(domain::TX_BODY);
        state[DOMAIN_SLOT + 1] =
            BabyBear::new((HASH_LEN as u32 + 2) * nin + HASH_LEN as u32 * nout);
        let mut absorb = |block: &[BabyBear]| {
            for (x, &v) in state.iter_mut().zip(block) {
                *x += v;
            }
            poseidon2::permute(&mut state);
        };
        for (c, position) in inputs {
            absorb(c);
            let (lo, hi) = crate::state::split(*position);
            absorb(&[lo, hi]);
        }
        for c in outputs {
            absorb(c);
        }
        if nin + nout == 0 {
            poseidon2::permute(&mut state);
        }
        core::array::from_fn(|j| state[j])
    }
}

/// [`TxBody::signed_digest`] from the signed outputs' commitments.
pub(crate) fn signed_digest_of(commitments: &[Hash]) -> Hash {
    if commitments.is_empty() {
        return [BabyBear::ZERO; HASH_LEN];
    }
    let n = BabyBear::new(count(commitments.len()));
    let blocks: Vec<&[BabyBear]> = commitments.iter().map(|c| &c[..]).collect();
    hash::sponge(domain::SIGNED_OUTPUTS, &[n], &blocks)
}

/// What an anyone-can-pay signature on an input signs: `sponge(SIG_ACP,
/// [position lo, hi], [signed outputs' digest, commitment])`.
pub fn acp_message(signed: &Hash, commitment: &Hash, position: u64) -> Hash {
    let (lo, hi) = crate::state::split(position);
    hash::sponge(domain::SIG_ACP, &[lo, hi], &[signed, commitment])
}

/// What an any-prevout signature signs: `sponge(SIG_APO, [amount limbs],
/// [signed outputs' digest])`.
pub fn apo_message(signed: &Hash, amount: Amount) -> Hash {
    hash::sponge(domain::SIG_APO, &output::amount_limbs(amount), &[signed])
}

/// Counts must stay below `p` to be one field element each.
fn count(n: usize) -> u32 {
    assert!(
        n < crate::babybear::P as usize,
        "too many inputs or outputs"
    );
    n as u32
}

fn total(mut amounts: impl Iterator<Item = Amount>) -> Result<Amount, TxError> {
    amounts
        .try_fold(0, Amount::checked_add)
        .ok_or(TxError::AmountOverflow)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transaction {
    pub body: TxBody,
    /// The inputs' signatures, in input order (a condition's in its
    /// signers' order).
    pub signatures: Vec<Signature>,
}

/// Sums of a transaction's input and output amounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Totals {
    pub inputs: Amount,
    pub outputs: Amount,
}

impl Totals {
    /// `inputs - outputs`. Negative when the transaction mints, as a block
    /// reward does.
    pub fn fee(&self) -> i128 {
        self.inputs as i128 - self.outputs as i128
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TxError {
    SignatureCount {
        inputs: usize,
        signatures: usize,
    },
    /// Input at this index spends the same commitment as an earlier one.
    DuplicateInput(usize),
    /// Input's spend, amount and nonce do not open its commitment.
    CommitmentMismatch(usize),
    /// Input's condition is malformed, or its signers don't match it (a
    /// key, or a state number not above the condition's).
    BadCondition(usize),
    /// Input's preimage doesn't meet its hashlock (or it has one without
    /// a hashlock).
    Hashlock(usize),
    /// An input's preimage isn't the start of an output's nonce (one per
    /// hashlock input).
    PreimageNotPublished,
    /// A protected input's non-signing member isn't paid its payout (an
    /// output per payout, none also publishing a preimage).
    PayoutMissing,
    /// Input signs anyone-can-pay or any-prevout after another condition
    /// input: only the first condition input may (so no two share the
    /// signed outputs).
    PartialSigner(usize),
    /// More signed outputs than outputs.
    SignedOutputs,
    /// Input's timelock hasn't passed.
    Locked(usize),
    BadSignature(usize),
    AmountOverflow,
    /// A commitment or output hash is not a canonical field encoding.
    NonCanonical,
    /// No inputs and no outputs.
    Empty,
}

impl fmt::Display for TxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureCount { inputs, signatures } => {
                write!(f, "{inputs} inputs need other than {signatures} signatures")
            }
            Self::DuplicateInput(i) => write!(f, "input {i} spends a commitment twice"),
            Self::CommitmentMismatch(i) => {
                write!(f, "input {i}: revealed fields do not open its commitment")
            }
            Self::BadCondition(i) => {
                write!(
                    f,
                    "input {i}: malformed condition or signers that don't meet it"
                )
            }
            Self::Hashlock(i) => write!(f, "input {i}: its preimage doesn't meet its hashlock"),
            Self::PreimageNotPublished => {
                write!(f, "an input's preimage isn't published by an output")
            }
            Self::PayoutMissing => {
                write!(f, "a member who doesn't sign isn't paid its payout")
            }
            Self::PartialSigner(i) => write!(
                f,
                "input {i}: only the first condition input may sign anyone-can-pay or any-prevout"
            ),
            Self::SignedOutputs => write!(f, "more signed outputs than outputs"),
            Self::Locked(i) => write!(f, "input {i}: still timelocked"),
            Self::BadSignature(i) => write!(f, "input {i}: invalid signature"),
            Self::AmountOverflow => write!(f, "amounts overflow"),
            Self::NonCanonical => write!(f, "non-canonical commitment or output hash"),
            Self::Empty => write!(f, "transaction has no inputs or outputs"),
        }
    }
}

impl std::error::Error for TxError {}

/// Reads an encoded transaction.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (a, b) = self.0.split_at_checked(n)?;
        self.0 = b;
        Some(a)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn hash(&mut self) -> Option<Hash> {
        get_elems(self.take(HASH_BYTES)?)
    }

    fn pubkey(&mut self) -> Option<PublicKey> {
        Some(PublicKey {
            param: get_elems(self.take(PARAM_LEN * 4)?)?,
            root: self.hash()?,
        })
    }

    fn key_path(&mut self) -> Option<KeyPath> {
        let mut path = [[BabyBear::ZERO; HASH_LEN]; KEY_LEVELS];
        for h in &mut path {
            *h = self.hash()?;
        }
        Some(path)
    }
}

fn put_pubkey(out: &mut Vec<u8>, pk: &PublicKey) {
    put_elems(out, &pk.param);
    put_elems(out, &pk.root);
}

impl Transaction {
    /// The encoding: input, output and signed-output counts (`u16`s), the
    /// inputs, the outputs, then the signatures. An input is its
    /// commitment, position, amount, nonce and a kind byte: 0, a key spend
    /// (public key, leaf, path); 1, a condition ([`Condition::to_bytes`],
    /// leaf, path, per signer its public key, index, depth and path, then a
    /// preimage flag and the preimage). Integers are little-endian, field
    /// elements `u32`s.
    pub fn to_bytes(&self) -> Vec<u8> {
        let TxBody {
            inputs,
            outputs,
            signed_outputs,
        } = &self.body;
        let mut out = Vec::with_capacity(
            TX_HEADER_BYTES
                + inputs.len() * (INPUT_BYTES + SIGNATURE_BYTES)
                + outputs.len() * OUTPUT_BYTES,
        );
        out.extend((inputs.len() as u16).to_le_bytes());
        out.extend((outputs.len() as u16).to_le_bytes());
        out.extend((*signed_outputs as u16).to_le_bytes());
        for i in inputs {
            out.extend(i.commitment.0);
            out.extend(i.position.to_le_bytes());
            out.extend(i.amount.to_le_bytes());
            out.extend(i.nonce);
            match &i.spend {
                Spend::Key { pubkey, leaf, path } => {
                    out.push(0);
                    put_pubkey(&mut out, pubkey);
                    out.push(*leaf);
                    for h in path {
                        put_elems(&mut out, h);
                    }
                }
                Spend::Condition {
                    condition: c,
                    leaf,
                    path,
                    signers,
                    preimage,
                } => {
                    out.push(1);
                    c.write(&mut out);
                    out.push(*leaf);
                    for h in path {
                        put_elems(&mut out, h);
                    }
                    out.push(signers.len() as u8);
                    for s in signers {
                        out.push(s.member);
                        put_pubkey(&mut out, &s.pubkey);
                        out.extend(s.index.to_le_bytes());
                        out.push(s.path.len() as u8);
                        for h in &s.path {
                            put_elems(&mut out, h);
                        }
                    }
                    match preimage {
                        Some(p) => {
                            out.push(1);
                            out.extend(p);
                        }
                        None => out.push(0),
                    }
                }
            }
        }
        for o in outputs {
            out.extend(o.hash);
            out.extend(o.amount.to_le_bytes());
            out.extend(o.nonce);
        }
        for sig in &self.signatures {
            put_elems(&mut out, &sig.rand);
            for c in &sig.chains {
                put_elems(&mut out, c);
            }
        }
        out
    }

    /// Decodes [`Transaction::to_bytes`] output: `None` unless it parses
    /// exactly, every field element is canonical, and each condition has
    /// one or two keys and a signer for each (the rest is checked by
    /// [`Transaction::validate`]).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader(bytes);
        let n_in = r.u16()? as usize;
        let n_out = r.u16()? as usize;
        let signed_outputs = r.u16()? as usize;
        // Every input takes more than this, so a count can't make us
        // allocate much more than the bytes we have.
        if n_in * (COMMITMENT_LEN + 8 + 8 + NONCE_LEN + 1) > r.0.len() {
            return None;
        }
        let mut inputs = Vec::with_capacity(n_in);
        for _ in 0..n_in {
            let commitment = Commitment(r.array()?);
            let position = r.u64()?;
            let amount = r.u64()?;
            let nonce = r.array()?;
            let spend = match r.u8()? {
                0 => Spend::Key {
                    pubkey: r.pubkey()?,
                    leaf: r.u8()?,
                    path: r.key_path()?,
                },
                1 => {
                    let condition = Condition::read(&mut r)?;
                    let leaf = r.u8()?;
                    let path = r.key_path()?;
                    let n_signers = r.u8()? as usize;
                    if n_signers > condition.keys.len() {
                        return None;
                    }
                    let mut signers = Vec::with_capacity(n_signers);
                    for _ in 0..n_signers {
                        let member = r.u8()?;
                        let pubkey = r.pubkey()?;
                        let index = r.u32()?;
                        let depth = r.u8()? as usize;
                        if depth > MAX_KEY_DEPTH {
                            return None;
                        }
                        let path = (0..depth).map(|_| r.hash()).collect::<Option<Vec<_>>>()?;
                        signers.push(Signer {
                            member,
                            pubkey,
                            index,
                            path,
                        });
                    }
                    let preimage = match r.u8()? {
                        0 => None,
                        1 => Some(r.array::<PREIMAGE_LEN>()?),
                        _ => return None,
                    };
                    Spend::Condition {
                        condition,
                        leaf,
                        path,
                        signers,
                        preimage,
                    }
                }
                _ => return None,
            };
            inputs.push(Input {
                commitment,
                position,
                amount,
                nonce,
                spend,
            });
        }
        if n_out * OUTPUT_BYTES > r.0.len() {
            return None;
        }
        let mut outputs = Vec::with_capacity(n_out);
        for _ in 0..n_out {
            outputs.push(Output {
                hash: r.array()?,
                amount: r.u64()?,
                nonce: r.array()?,
            });
        }
        let n_sigs: usize = inputs.iter().map(Input::signature_count).sum();
        if r.0.len() != n_sigs * SIGNATURE_BYTES {
            return None;
        }
        let mut signatures = Vec::with_capacity(n_sigs);
        for _ in 0..n_sigs {
            let rand = get_elems(r.take(RAND_LEN * 4)?)?;
            let mut chains = [[BabyBear::ZERO; HASH_LEN]; NUM_CHAINS];
            for c in &mut chains {
                *c = r.hash()?;
            }
            signatures.push(Signature { rand, chains });
        }
        Some(Self {
            body: TxBody {
                inputs,
                outputs,
                signed_outputs,
            },
            signatures,
        })
    }

    /// The id of the transaction encoded as `bytes`.
    pub fn id_of(bytes: &[u8]) -> Hash {
        chunk::hash_of(domain::TX_ID, bytes)
    }

    /// Signs `body` with one key per signature ([`TxBody::messages`]), in
    /// order.
    ///
    /// # Panics
    /// If the number of keys differs from the number of signatures needed.
    pub fn sign(body: TxBody, keys: Vec<SecretKey>) -> Result<Self, TxError> {
        let messages = body.messages().ok_or(TxError::NonCanonical)?;
        assert_eq!(keys.len(), messages.len(), "one key per signature");
        let signatures = keys
            .into_iter()
            .zip(&messages)
            .map(|(k, m)| k.sign(m))
            .collect();
        Ok(Self { body, signatures })
    }

    /// Checks everything that needs no chain state and returns the totals.
    pub fn validate(&self) -> Result<Totals, TxError> {
        let TxBody {
            inputs, outputs, ..
        } = &self.body;
        if inputs.is_empty() && outputs.is_empty() {
            return Err(TxError::Empty);
        }
        let needed: usize = inputs.iter().map(Input::signature_count).sum();
        if self.signatures.len() != needed {
            return Err(TxError::SignatureCount {
                inputs: inputs.len(),
                signatures: self.signatures.len(),
            });
        }
        if self.body.signed_outputs > outputs.len() {
            return Err(TxError::SignedOutputs);
        }
        for (i, input) in inputs.iter().enumerate() {
            if inputs[..i].iter().any(|p| p.commitment == input.commitment) {
                return Err(TxError::DuplicateInput(i));
            }
            if let Spend::Condition {
                condition: c,
                signers,
                preimage,
                ..
            } = &input.spend
            {
                let fits = c.is_well_formed()
                    && c.enough_signers(signers.len())
                    && signers.windows(2).all(|w| w[0].member < w[1].member)
                    && signers.iter().all(|s| {
                        s.is_well_formed()
                            && c.keys.get(s.member as usize) == Some(&s.root())
                            && c.min_state.is_none_or(|m| s.index > m)
                    });
                if !fits {
                    return Err(TxError::BadCondition(i));
                }
                if c.hashlock != preimage.as_ref().map(hashlock) {
                    return Err(TxError::Hashlock(i));
                }
                if c.mode != SigMode::All && inputs[..i].iter().any(|p| p.condition().is_some()) {
                    return Err(TxError::PartialSigner(i));
                }
            }
            if !input.opens_commitment() {
                return Err(TxError::CommitmentMismatch(i));
            }
        }
        if self.body.publishers().is_none() {
            return Err(TxError::PreimageNotPublished);
        }
        if self.body.payouts().is_none() {
            return Err(TxError::PayoutMissing);
        }
        let totals = Totals {
            inputs: total(inputs.iter().map(|i| i.amount))?,
            outputs: total(outputs.iter().map(|o| o.amount))?,
        };

        // Signatures last: they are by far the most expensive check.
        let messages = self.body.messages().ok_or(TxError::NonCanonical)?;
        let mut sigs = self.signatures.iter().zip(&messages);
        for (i, input) in inputs.iter().enumerate() {
            for pubkey in input.pubkeys() {
                let (sig, msg) = sigs.next().expect("counted");
                if !wots::verify(pubkey, msg, sig) {
                    return Err(TxError::BadSignature(i));
                }
            }
        }
        Ok(totals)
    }

    /// Checks the inputs' timelocks for a block at `height`, given each
    /// spent output's creation height (`created`, by input).
    pub fn check_locks(
        &self,
        height: u64,
        created: impl Fn(&Input) -> Option<u32>,
    ) -> Result<(), TxError> {
        for (i, input) in self.body.inputs.iter().enumerate() {
            let Some(c) = input.condition() else {
                continue;
            };
            let born = created(input).ok_or(TxError::Locked(i))?;
            if height < c.after_height as u64 || height < born as u64 + c.after_blocks as u64 {
                return Err(TxError::Locked(i));
            }
        }
        Ok(())
    }
}

fn put_elems(out: &mut Vec<u8>, elems: &[BabyBear]) {
    for x in elems {
        out.extend(x.as_u32().to_le_bytes());
    }
}

/// `N` canonical field elements from `4 * N` bytes.
fn get_elems<const N: usize>(bytes: &[u8]) -> Option<[BabyBear; N]> {
    let mut out = [BabyBear::ZERO; N];
    for (x, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        let v = u32::from_le_bytes(chunk.try_into().ok()?);
        if v >= P {
            return None;
        }
        *x = BabyBear::new(v);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;
    use crate::output::NONCE_LEN;

    fn key(n: u32) -> (SecretKey, PublicKey) {
        wots::keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 100 + i as u32
        })))
    }

    /// An input spending an output of `amount` paid to key `n`.
    fn input(n: u32, amount: Amount) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let nonce = [n as u8; NONCE_LEN];
        // Leaf 1 of a tree of four keys (the rest unused here).
        let mut tree = KeyTree::single(&key(n + 50).1);
        tree.0[1] = pubkey.hash();
        let commitment = output::commit(&tree.output_hash(), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                position: n as u64,
                amount,
                nonce,
                spend: Spend::Key {
                    pubkey,
                    leaf: 1,
                    path: tree.path(1),
                },
            },
        )
    }

    fn out(amount: Amount, tag: u8) -> Output {
        Output {
            hash: [tag; 32],
            amount,
            nonce: [tag; NONCE_LEN],
        }
    }

    /// Two inputs (100 + 50) paying 120 + 20, so the fee is 10.
    fn tx() -> Transaction {
        let (sk1, in1) = input(1, 100);
        let (sk2, in2) = input(2, 50);
        let body = TxBody::new(vec![in1, in2], vec![out(120, 1), out(20, 2)]);
        Transaction::sign(body, vec![sk1, sk2]).unwrap()
    }

    /// Re-signs a modified body with the keys matching its inputs.
    fn resign(body: TxBody) -> Transaction {
        let keys = body
            .inputs
            .iter()
            .map(|i| {
                let pk = i.pubkeys()[0].clone();
                (1..=4).map(key).find(|(_, k)| *k == pk).unwrap().0
            })
            .collect();
        Transaction::sign(body, keys).unwrap()
    }

    fn totals(inputs: Amount, outputs: Amount) -> Result<Totals, TxError> {
        Ok(Totals { inputs, outputs })
    }

    #[test]
    fn transactions_round_trip_through_bytes() {
        let t = tx();
        let bytes = t.to_bytes();
        assert_eq!(
            bytes.len(),
            TX_HEADER_BYTES + 2 * (INPUT_BYTES + SIGNATURE_BYTES) + 2 * OUTPUT_BYTES
        );
        assert_eq!(Transaction::from_bytes(&bytes), Some(t));
        assert_eq!(Transaction::from_bytes(&bytes[..bytes.len() - 1]), None);
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(Transaction::from_bytes(&longer), None);
        // A non-canonical field element (the first key's parameter).
        let mut bad = bytes.clone();
        let at = TX_HEADER_BYTES + COMMITMENT_LEN + 8 + 8 + NONCE_LEN + 1;
        bad[at..at + 4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(Transaction::from_bytes(&bad), None);
        assert_ne!(Transaction::id_of(&bytes), Transaction::id_of(&longer));
    }

    #[test]
    fn valid_transaction_returns_totals() {
        let t = tx().validate().unwrap();
        assert_eq!(
            t,
            Totals {
                inputs: 150,
                outputs: 140
            }
        );
        assert_eq!(t.fee(), 10);
    }

    #[test]
    fn block_reward_has_outputs_and_no_inputs() {
        let reward = Transaction::sign(TxBody::new(vec![], vec![out(50, 1)]), vec![]).unwrap();
        let t = reward.validate().unwrap();
        assert_eq!(
            t,
            Totals {
                inputs: 0,
                outputs: 50
            }
        );
        assert_eq!(t.fee(), -50);
    }

    #[test]
    fn outputs_may_exceed_inputs() {
        // The balance rule belongs to the block, not the transaction.
        let mut body = tx().body;
        body.outputs[0].amount = 131;
        let t = resign(body).validate().unwrap();
        assert_eq!(t.fee(), -1);
    }

    #[test]
    fn exact_spend_and_all_fee() {
        let mut body = tx().body;
        body.outputs[1].amount = 30;
        assert_eq!(resign(body.clone()).validate(), totals(150, 150));
        body.outputs.clear();
        assert_eq!(resign(body).validate(), totals(150, 0));
    }

    #[test]
    fn rejects_amount_overflow() {
        let mut body = tx().body;
        body.outputs = vec![out(u64::MAX, 1), out(1, 2)];
        assert_eq!(resign(body).validate(), Err(TxError::AmountOverflow));
    }

    #[test]
    fn rejects_structural_errors() {
        let t = tx();
        let mut missing_sig = t.clone();
        missing_sig.signatures.pop();
        assert_eq!(
            missing_sig.validate(),
            Err(TxError::SignatureCount {
                inputs: 2,
                signatures: 1
            })
        );

        let mut dup = t.body.clone();
        dup.inputs[1] = dup.inputs[0].clone();
        assert_eq!(resign(dup).validate(), Err(TxError::DuplicateInput(1)));
        let mut signed = t.body.clone();
        signed.signed_outputs = 3;
        assert_eq!(resign(signed).validate(), Err(TxError::SignedOutputs));
    }

    #[test]
    fn rejects_inflated_input_amount() {
        // Claiming more than the committed amount must fail, even when re-signed.
        let mut body = tx().body;
        body.inputs[0].amount = 1000;
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn rejects_wrong_nonce_for_commitment() {
        let mut body = tx().body;
        body.inputs[0].nonce[0] ^= 1;
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn rejects_wrong_key_for_commitment() {
        let with = |f: fn(&mut PublicKey, &mut u8, &mut KeyPath)| {
            let mut body = tx().body;
            if let Spend::Key { pubkey, leaf, path } = &mut body.inputs[0].spend {
                f(pubkey, leaf, path);
            }
            body
        };
        let body = with(|pk, _, _| *pk = key(3).1);
        let k3 = key(3).0;
        let t = Transaction::sign(body, vec![k3, key(2).0]).unwrap();
        assert_eq!(t.validate(), Err(TxError::CommitmentMismatch(0)));
        // The key at another leaf, or a wrong path.
        let body = with(|_, leaf, _| *leaf = 0);
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
        let body = with(|_, _, path| path[2][0] += BabyBear::ONE);
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn key_trees_have_a_path_per_leaf() {
        let tree = KeyTree(core::array::from_fn(|i| key(i as u32).1.hash()));
        for leaf in 0..KEY_LEAVES {
            assert_eq!(
                key_root(&tree.0[leaf], leaf as u8, &tree.path(leaf)),
                tree.root()
            );
        }
        assert_ne!(key_root(&tree.0[3], 2, &tree.path(3)), tree.root());
        let signer = SignerTree::new((0..16).map(|i| key(i).1.hash()).collect());
        assert_eq!(signer.depth(), 4);
        for i in 0..16 {
            let pkh = key(i).1.hash();
            assert_eq!(tree_root(&pkh, i, &signer.path(i)), signer.root());
        }
    }

    #[test]
    fn signatures_cover_every_body_field() {
        let t = tx();
        let tampered: [fn(&mut TxBody); 4] = [
            |b| b.outputs[0].hash[0] ^= 1,
            |b| b.outputs[0].nonce[0] ^= 1,
            |b| {
                b.outputs[0].amount -= 1;
                b.outputs[1].amount += 1;
            },
            |b| b.outputs.swap(0, 1),
        ];
        for f in tampered {
            let mut bad = t.clone();
            f(&mut bad.body);
            assert_eq!(bad.validate(), Err(TxError::BadSignature(0)));
        }
    }

    #[test]
    fn rejects_swapped_signatures() {
        let mut t = tx();
        t.signatures.swap(0, 1);
        assert_eq!(t.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn digest_covers_counts_and_commitments() {
        let body = tx().body;
        let d = body.digest().unwrap();
        // Moving an output into a fresh slot changes the counts.
        let mut fewer = body.clone();
        fewer.outputs.pop();
        assert_ne!(fewer.digest().unwrap(), d);
        // Same commitments, swapped order: a different digest.
        let mut swapped = body.clone();
        swapped.outputs.swap(0, 1);
        assert_ne!(swapped.digest().unwrap(), d);
        // The spent position is signed: the same commitment elsewhere in
        // the tree is a different message.
        let mut moved = body.clone();
        moved.inputs[0].position += 1 << 30;
        assert_ne!(moved.digest().unwrap(), d);
        let mut moved = body.clone();
        moved.inputs[1].position += 1;
        assert_ne!(moved.digest().unwrap(), d);
        let t = tx();
        let mut replayed = t.clone();
        replayed.body.inputs[0].position = 999;
        assert_eq!(replayed.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn rejects_empty_transaction() {
        let empty = Transaction::sign(TxBody::default(), vec![]).unwrap();
        assert_eq!(empty.validate(), Err(TxError::Empty));
    }

    #[test]
    fn rejects_non_canonical_output_hash() {
        let mut body = tx().body;
        body.outputs[0].hash[..4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(body.digest(), None);
        let (sk1, _) = input(1, 100);
        let (sk2, _) = input(2, 50);
        assert_eq!(
            Transaction::sign(body.clone(), vec![sk1, sk2]),
            Err(TxError::NonCanonical)
        );
        let mut t = tx();
        t.body = body;
        assert_eq!(t.validate(), Err(TxError::NonCanonical));
    }

    /// A signer with a tree of `n` keys (seeds from `seed`), signing with
    /// key `index`.
    struct Party {
        seed: u32,
        keys: Vec<PublicKey>,
        tree: SignerTree,
    }

    impl Party {
        fn new(seed: u32, n: u32) -> Self {
            let keys: Vec<_> = (0..n).map(|i| key(seed + i).1).collect();
            let tree = SignerTree::new(keys.iter().map(PublicKey::hash).collect());
            Self { seed, keys, tree }
        }

        /// Key `index`, signing as member 0 of a condition's key set.
        fn signer(&self, index: u32) -> Signer {
            self.member(0, index)
        }

        /// Key `index`, signing as member `member`.
        fn member(&self, member: u8, index: u32) -> Signer {
            Signer {
                member,
                pubkey: self.keys[index as usize].clone(),
                index,
                path: self.tree.path(index),
            }
        }

        fn sk(&self, index: u32) -> SecretKey {
            key(self.seed + index).0
        }
    }

    /// An input spending an output of `amount`, at `position`, paying to
    /// `conditions` (spent by the one at `leaf`, with `signers`).
    fn cond_input(
        conditions: &[Condition],
        leaf: u8,
        signers: Vec<Signer>,
        amount: Amount,
        position: u64,
        tag: u8,
    ) -> Input {
        let tree = KeyTree::of_conditions(conditions);
        let nonce = [tag; NONCE_LEN];
        Input {
            commitment: output::commit(&tree.output_hash(), amount, &nonce).unwrap(),
            position,
            amount,
            nonce,
            spend: Spend::Condition {
                condition: conditions[leaf as usize].clone(),
                leaf,
                path: tree.path(leaf as usize),
                signers,
                preimage: None,
            },
        }
    }

    /// `input` with `preimage` for its hashlock.
    fn with_preimage(mut input: Input, preimage: Preimage) -> Input {
        if let Spend::Condition { preimage: p, .. } = &mut input.spend {
            *p = Some(preimage);
        }
        input
    }

    fn channel(a: &Party, b: &Party, state: u32) -> Condition {
        Condition {
            keys: vec![a.tree.root(), b.tree.root()],
            threshold: 2,
            min_state: Some(state),
            mode: SigMode::AnyPrevout,
            after_height: 0,
            after_blocks: 0,
            hashlock: None,
            payouts: None,
        }
    }

    #[test]
    fn a_later_state_spends_any_earlier_one() {
        let (a, b) = (Party::new(1000, 16), Party::new(2000, 16));
        // Update 9 pays to state 9's condition; it's signed once, and the
        // same signatures spend state 3's output or state 5's.
        let outputs = vec![Output {
            hash: KeyTree::of_conditions(&[channel(&a, &b, 9)]).output_hash(),
            amount: 1000,
            nonce: [9; NONCE_LEN],
        }];
        let update = |state: u32, position: u64| TxBody {
            inputs: vec![cond_input(
                &[channel(&a, &b, state)],
                0,
                vec![a.signer(9), b.member(1, 9)],
                1000,
                position,
                state as u8,
            )],
            outputs: outputs.clone(),
            signed_outputs: 1,
        };
        let signed = Transaction::sign(update(3, 40), vec![a.sk(9), b.sk(9)]).unwrap();
        assert_eq!(signed.validate(), totals(1000, 1000));
        let rebound = Transaction {
            body: update(5, 77),
            signatures: signed.signatures.clone(),
        };
        assert_eq!(rebound.validate(), totals(1000, 1000));
        // But not an output at state 9 or later, and not with keys at
        // indices that aren't above the state.
        let late = Transaction {
            body: update(9, 80),
            signatures: signed.signatures.clone(),
        };
        assert_eq!(late.validate(), Err(TxError::BadCondition(0)));
        // A signature on another output set doesn't carry over.
        let mut other = update(3, 40);
        other.outputs[0].amount = 999;
        let moved = Transaction {
            body: other,
            signatures: signed.signatures.clone(),
        };
        assert_eq!(moved.validate(), Err(TxError::BadSignature(0)));
        // Both parties must sign.
        let mut one = signed.clone();
        one.signatures[1] = one.signatures[0].clone();
        assert_eq!(one.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn anyone_can_pay_lets_others_add_inputs_and_outputs() {
        let a = Party::new(3000, 1);
        let cond = Condition {
            mode: SigMode::AnyoneCanPay,
            ..Condition::key(a.tree.root())
        };
        let mut body = TxBody {
            inputs: vec![cond_input(&[cond], 0, vec![a.signer(0)], 500, 12, 1)],
            outputs: vec![out(500, 1)],
            signed_outputs: 1,
        };
        let sig = Transaction::sign(body.clone(), vec![a.sk(0)])
            .unwrap()
            .signatures;
        // A fee payer adds an input and change, and signs everything.
        let (fee_sk, fee_input) = input(2, 50);
        body.inputs.push(fee_input);
        body.outputs.push(out(40, 2));
        let messages = body.messages().unwrap();
        let t = Transaction {
            signatures: vec![sig[0].clone(), fee_sk.sign(&messages[1])],
            body: body.clone(),
        };
        assert_eq!(t.validate(), totals(550, 540));
        // The signed output can't change, and the signature names its input.
        let mut bad = t.clone();
        bad.body.outputs[0].amount = 499;
        assert_eq!(bad.validate(), Err(TxError::BadSignature(0)));
        let mut bad = t.clone();
        if let Spend::Condition { .. } = bad.body.inputs[0].spend {
            bad.body.inputs[0].position = 13;
        }
        assert_eq!(bad.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn a_hashlock_needs_a_published_preimage() {
        let a = Party::new(4000, 1);
        let pre = [7u8; PREIMAGE_LEN];
        let cond = Condition {
            hashlock: Some(hashlock(&pre)),
            ..Condition::key(a.tree.root())
        };
        let body = |preimage: Option<Preimage>, nonce_start: u8| {
            let mut claim = out(90, 3);
            claim.nonce[..PREIMAGE_LEN].fill(nonce_start);
            let input = cond_input(std::slice::from_ref(&cond), 0, vec![a.signer(0)], 100, 5, 1);
            let input = match preimage {
                Some(p) => with_preimage(input, p),
                None => input,
            };
            TxBody::new(vec![input], vec![claim])
        };
        let sign = |b: TxBody| Transaction::sign(b, vec![a.sk(0)]).unwrap();
        assert_eq!(sign(body(Some(pre), 7)).validate(), totals(100, 90));
        assert_eq!(sign(body(None, 7)).validate(), Err(TxError::Hashlock(0)));
        assert_eq!(
            sign(body(Some([8; PREIMAGE_LEN]), 8)).validate(),
            Err(TxError::Hashlock(0))
        );
        assert_eq!(
            sign(body(Some(pre), 6)).validate(),
            Err(TxError::PreimageNotPublished)
        );
        // A preimage without a hashlock.
        let plain = with_preimage(
            cond_input(
                &[Condition::key(a.tree.root())],
                0,
                vec![a.signer(0)],
                100,
                5,
                1,
            ),
            pre,
        );
        let mut claim = out(90, 3);
        claim.nonce[..PREIMAGE_LEN].copy_from_slice(&pre);
        assert_eq!(
            sign(TxBody::new(vec![plain], vec![claim])).validate(),
            Err(TxError::Hashlock(0))
        );
    }

    /// One transaction claims two hashlocks with different preimages,
    /// each published by its own output; two claims of the same hashlock
    /// need two outputs.
    #[test]
    fn hashlocks_each_need_their_own_published_preimage() {
        let a = Party::new(4100, 4);
        let (p1, p2) = ([1u8; PREIMAGE_LEN], [2u8; PREIMAGE_LEN]);
        let lock = |p: &Preimage| Condition {
            hashlock: Some(hashlock(p)),
            ..Condition::key(a.tree.root())
        };
        let claim = |p: &Preimage, tag: u8| {
            let mut o = out(10, tag);
            o.nonce[..PREIMAGE_LEN].copy_from_slice(p);
            o
        };
        let inputs = |second: &Preimage| {
            vec![
                with_preimage(cond_input(&[lock(&p1)], 0, vec![a.signer(0)], 20, 5, 1), p1),
                with_preimage(
                    cond_input(&[lock(second)], 0, vec![a.signer(1)], 20, 6, 2),
                    *second,
                ),
            ]
        };
        let sign = |b: TxBody| Transaction::sign(b, vec![a.sk(0), a.sk(1)]).unwrap();
        let both = TxBody::new(inputs(&p2), vec![claim(&p2, 3), claim(&p1, 4)]);
        assert_eq!(both.publishers(), Some(vec![Some(1), Some(0)]));
        assert_eq!(sign(both).validate(), totals(40, 20));
        let one_output = TxBody::new(inputs(&p1), vec![claim(&p1, 3), out(10, 4)]);
        assert_eq!(
            sign(one_output).validate(),
            Err(TxError::PreimageNotPublished)
        );
        let two_outputs = TxBody::new(inputs(&p1), vec![claim(&p1, 3), claim(&p1, 4)]);
        assert_eq!(sign(two_outputs).validate(), totals(40, 20));
    }

    /// Only the first condition input may sign anyone-can-pay or
    /// any-prevout, so no two share the outputs they sign.
    #[test]
    fn only_the_first_condition_input_signs_partially() {
        let a = Party::new(4200, 4);
        let partial = |mode| Condition {
            mode,
            ..Condition::key(a.tree.root())
        };
        let body = |first: SigMode, second: SigMode| TxBody {
            inputs: vec![
                cond_input(&[partial(first)], 0, vec![a.signer(0)], 20, 5, 1),
                cond_input(&[partial(second)], 0, vec![a.signer(1)], 20, 6, 2),
            ],
            outputs: vec![out(30, 3)],
            signed_outputs: 1,
        };
        let sign = |b: TxBody| Transaction::sign(b, vec![a.sk(0), a.sk(1)]).unwrap();
        assert_eq!(
            sign(body(SigMode::AnyoneCanPay, SigMode::All)).validate(),
            totals(40, 30)
        );
        assert_eq!(
            sign(body(SigMode::AnyoneCanPay, SigMode::AnyPrevout)).validate(),
            Err(TxError::PartialSigner(1))
        );
        assert_eq!(
            sign(body(SigMode::All, SigMode::AnyoneCanPay)).validate(),
            Err(TxError::PartialSigner(1))
        );
        // A key spend before it doesn't count.
        let (sk, key_input) = input(2, 50);
        let mut b = body(SigMode::AnyoneCanPay, SigMode::All);
        b.inputs.insert(0, key_input);
        let t = Transaction::sign(b, vec![sk, a.sk(0), a.sk(1)]).unwrap();
        assert_eq!(t.validate(), totals(90, 30));
    }

    #[test]
    fn timelocks_count_from_the_height_and_the_outputs_creation() {
        let a = Party::new(5000, 1);
        let cond = Condition {
            after_height: 100,
            after_blocks: 10,
            ..Condition::key(a.tree.root())
        };
        let t = Transaction::sign(
            TxBody::new(
                vec![cond_input(&[cond], 0, vec![a.signer(0)], 100, 5, 1)],
                vec![out(90, 3)],
            ),
            vec![a.sk(0)],
        )
        .unwrap();
        assert_eq!(t.validate(), totals(100, 90));
        let at = |height, created| t.check_locks(height, |_| Some(created));
        assert_eq!(at(99, 0), Err(TxError::Locked(0)));
        assert_eq!(at(100, 90), Ok(()));
        assert_eq!(at(100, 91), Err(TxError::Locked(0)));
        assert_eq!(at(200, 190), Ok(()));
        assert_eq!(t.check_locks(200, |_| None), Err(TxError::Locked(0)));
        // Key spends have no locks.
        assert_eq!(tx().check_locks(0, |_| None), Ok(()));
    }

    #[test]
    fn conditions_round_trip_and_reject_bad_signers() {
        let (a, b) = (Party::new(6000, 4), Party::new(7000, 4));
        let pre = [3u8; PREIMAGE_LEN];
        let htlc = Condition {
            hashlock: Some(hashlock(&pre)),
            after_height: 7,
            ..Condition::key(a.tree.root())
        };
        let conditions = [channel(&a, &b, 1), htlc];
        let mut claim = out(10, 4);
        claim.nonce[..PREIMAGE_LEN].copy_from_slice(&pre);
        let body = TxBody {
            inputs: vec![
                cond_input(&conditions, 0, vec![a.signer(2), b.member(1, 3)], 60, 8, 1),
                with_preimage(cond_input(&conditions, 1, vec![a.signer(0)], 40, 9, 2), pre),
            ],
            outputs: vec![claim, out(80, 5)],
            signed_outputs: 2,
        };
        let t = Transaction::sign(body, vec![a.sk(2), b.sk(3), a.sk(0)]).unwrap();
        assert_eq!(t.validate(), totals(100, 90));
        let bytes = t.to_bytes();
        assert_eq!(Transaction::from_bytes(&bytes), Some(t.clone()));
        assert_eq!(Transaction::from_bytes(&bytes[..bytes.len() - 1]), None);
        // A signer whose key isn't under the condition's root, or whose
        // index doesn't fit its path.
        let mut bad = t.clone();
        if let Spend::Condition { signers, .. } = &mut bad.body.inputs[0].spend {
            signers[1].index = 2;
        }
        assert_eq!(bad.validate(), Err(TxError::BadCondition(0)));
        let mut bad = t.clone();
        if let Spend::Condition { signers, .. } = &mut bad.body.inputs[0].spend {
            signers[0].index = 6;
        }
        assert_eq!(bad.validate(), Err(TxError::BadCondition(0)));
        // A condition whose locks don't fit.
        let mut bad = t.clone();
        if let Spend::Condition { condition, .. } = &mut bad.body.inputs[1].spend {
            condition.after_blocks = MAX_LOCK + 1;
        }
        assert_eq!(bad.validate(), Err(TxError::BadCondition(1)));
    }

    /// Any 2 of 3 keys spend a 2-of-3 condition; one, a repeated member,
    /// members out of order, or a key at the wrong member don't.
    #[test]
    fn a_threshold_of_the_keys_signs() {
        let parties = [
            Party::new(8000, 2),
            Party::new(8100, 2),
            Party::new(8200, 2),
        ];
        let cond = Condition {
            threshold: 2,
            ..Condition::all_of(parties.iter().map(|p| p.tree.root()).collect())
        };
        let spend = |signers: Vec<Signer>, keys: Vec<SecretKey>| {
            let input = cond_input(std::slice::from_ref(&cond), 0, signers, 70, 3, 1);
            Transaction::sign(TxBody::new(vec![input], vec![out(60, 2)]), keys).unwrap()
        };
        let m = |p: usize, i: u32| parties[p].member(p as u8, i);
        let sk = |p: usize, i: u32| parties[p].sk(i);
        for (x, y) in [(0, 1), (0, 2), (1, 2)] {
            let t = spend(vec![m(x, 0), m(y, 0)], vec![sk(x, 0), sk(y, 0)]);
            assert_eq!(t.validate(), totals(70, 60), "{x} and {y}");
            assert_eq!(Transaction::from_bytes(&t.to_bytes()), Some(t));
        }
        let bad = [
            spend(vec![m(0, 0)], vec![sk(0, 0)]),
            spend(vec![m(1, 0), m(1, 1)], vec![sk(1, 0), sk(1, 1)]),
            spend(vec![m(2, 0), m(0, 0)], vec![sk(2, 0), sk(0, 0)]),
            spend(
                vec![m(0, 0), parties[1].member(2, 0)],
                vec![sk(0, 0), sk(1, 0)],
            ),
        ];
        for t in bad {
            assert_eq!(t.validate(), Err(TxError::BadCondition(0)));
        }
        // Thresholds must be from 1 to the number of keys.
        for threshold in [0, 4] {
            assert!(
                !Condition {
                    threshold,
                    ..cond.clone()
                }
                .is_well_formed()
            );
        }
    }

    /// A protected 2-of-3 condition: two members spend it, paying the
    /// third its payout; a payout missing, wrong or taken twice, too few
    /// signers, or protection without a delay don't.
    #[test]
    fn protection_pays_the_members_who_dont_sign() {
        let parties = [
            Party::new(8300, 2),
            Party::new(8400, 2),
            Party::new(8500, 2),
        ];
        let payouts = [out(20, 10), out(30, 11), out(15, 12)];
        let field = |o: &Output| o.commitment().unwrap().to_field().unwrap();
        let cond = Condition {
            threshold: 2,
            after_blocks: 5,
            payouts: Some(payouts.iter().map(field).collect()),
            ..Condition::all_of(parties.iter().map(|p| p.tree.root()).collect())
        };
        assert!(cond.is_well_formed());
        assert_eq!(Condition::from_bytes(&cond.to_bytes()), Some(cond.clone()));
        let m = |p: usize| parties[p].member(p as u8, 0);
        let sk = |p: usize| parties[p].sk(0);
        let spend = |cond: &Condition, signers: Vec<usize>, outputs: Vec<Output>| {
            let input = cond_input(
                std::slice::from_ref(cond),
                0,
                signers.iter().map(|&p| m(p)).collect(),
                65,
                3,
                1,
            );
            let keys = signers.iter().map(|&p| sk(p)).collect();
            Transaction::sign(TxBody::new(vec![input], outputs), keys).unwrap()
        };

        // 0 and 2 sign; 1 is paid.
        let t = spend(&cond, vec![0, 2], vec![out(30, 2), payouts[1]]);
        assert_eq!(t.validate(), totals(65, 60));
        assert_eq!(t.body.required_payouts(), vec![field(&payouts[1])]);
        assert_eq!(t.body.payouts(), Some(vec![1]));
        assert_eq!(Transaction::from_bytes(&t.to_bytes()), Some(t));
        // All sign: nobody needs paying.
        let t = spend(&cond, vec![0, 1, 2], vec![out(60, 2)]);
        assert_eq!(t.validate(), totals(65, 60));
        assert_eq!(t.body.payouts(), Some(vec![]));

        // Unpaid, or paid the wrong amount.
        let t = spend(&cond, vec![0, 2], vec![out(60, 2)]);
        assert_eq!(t.validate(), Err(TxError::PayoutMissing));
        let mut short = payouts[1];
        short.amount -= 1;
        let t = spend(&cond, vec![0, 2], vec![out(30, 2), short]);
        assert_eq!(t.validate(), Err(TxError::PayoutMissing));
        // Two members unpaid need two outputs, even with equal payouts.
        let same = Condition {
            threshold: 1,
            payouts: Some(vec![field(&payouts[1]); 3]),
            ..cond.clone()
        };
        let t = spend(&same, vec![0], vec![out(15, 2), payouts[1]]);
        assert_eq!(t.validate(), Err(TxError::PayoutMissing));
        // Below the threshold.
        let t = spend(&cond, vec![2], vec![payouts[0], payouts[1]]);
        assert_eq!(t.validate(), Err(TxError::BadCondition(0)));
        // Protection needs a delay, and a payout per key.
        let no_delay = Condition {
            after_blocks: 0,
            ..cond.clone()
        };
        assert!(!no_delay.is_well_formed());
        let t = spend(&no_delay, vec![0, 2], vec![out(30, 2), payouts[1]]);
        assert_eq!(t.validate(), Err(TxError::BadCondition(0)));
        let mut few = cond.clone();
        few.payouts.as_mut().unwrap().pop();
        assert!(!few.is_well_formed());
    }
}
