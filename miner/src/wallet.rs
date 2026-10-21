//! Wallets: one-time keys, output nonces that carry the amount, and
//! recovery from a seed.
//!
//! Every output a wallet makes (received, change or reward) gets its own
//! random 16-byte key seed `r`. Its keys are a [`KeyTree`] of
//! [`KEY_LEAVES`] one-time WOTS keys derived from the wallet's seed, `r`
//! and the leaf; the output's hash is the tree's root. A spend signs with
//! one leaf; replacing it (a higher fee, or a cancellation) signs with
//! another, and the [`WalletStore`] records which leaves each output has
//! used, so no key signs twice (outputs found on restore, whose history
//! is unknown, sign with random leaves of the upper half: [`Leaves`]).
//! Because `r` is random, no two outputs share
//! keys, whatever the wallet remembers: a wallet restored with history
//! missing (a fast-synced node), or two copies of one wallet, never reuse
//! a key. Nothing is ever published for others to pay to; the receiver
//! makes its own output ([`Response`]).
//!
//! An output's 32-byte nonce is:
//!
//! - bytes `0..16`: `r`, in the clear;
//! - bytes `16..24`: a tag, `H(view, r)`, from the wallet's view secret;
//! - bytes `24..32`: the amount (little-endian) XORed with another hash of
//!   the view secret and `r`.
//!
//! Without the view secret the tag and amount look random. The nonce is
//! part of the output's commitment and stored in the state tree, so a
//! wallet restored from its seed recovers every unspent output it owns,
//! with its amount and position, from the tree alone ([`Wallet::scan`]):
//! one hash per output checks the tag, and a match is confirmed by
//! recomputing the commitment from its key tree. There's no index to keep
//! and no gap to guess.
//!
//! **Paying.** Payments are interactive, through files (slates), as in
//! Grin. The sender picks coins and writes a [`Slate`] (an id, the amount
//! and fee); the receiver makes an output for the amount with new keys of
//! its own and writes it back in a [`Response`]; the sender then signs the
//! transaction (its coins in, the receiver's output and its change out)
//! and submits it. The response repeats the slate (id, amount and fee), and
//! the sender finalizes only a response to a slate it sent, unchanged. The
//! [`WalletStore`] keeps the seed and the wallet's records in LMDB.

use std::collections::HashMap;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::babybear::BabyBear;
use crate::hash::{self, Hash, domain};
use crate::output::{
    self, Amount, COMMITMENT_LEN, Commitment, NONCE_LEN, OUTPUT_HASH_LEN, Output, OutputHash,
    OutputRecord,
};
use crate::transaction::{Input, KEY_LEAVES, KeyTree};
use crate::wots::{self, PublicKey, SecretKey};

/// Bytes of an output's key seed `r`.
pub const KEY_SEED_LEN: usize = 16;
/// Each output's own random value: its keys derive from it (with the
/// wallet's spend secret), so no two outputs ever share a key.
pub type KeySeed = [u8; KEY_SEED_LEN];
/// Bytes of the tag, and of the masked amount, in a nonce.
const TAG_LEN: usize = 8;
const _: () = assert!(KEY_SEED_LEN + TAG_LEN + 8 == NONCE_LEN);

/// The key seed in `record`'s nonce.
pub fn key_seed_of(record: &OutputRecord) -> KeySeed {
    record.nonce[..KEY_SEED_LEN].try_into().expect("seed bytes")
}

/// `r` as eight 16-bit limbs (injective, so distinct seeds hash apart).
fn seed_limbs(r: &KeySeed) -> [BabyBear; KEY_SEED_LEN / 2] {
    core::array::from_fn(|i| BabyBear::new(u16::from_le_bytes([r[2 * i], r[2 * i + 1]]) as u32))
}

/// Eight bytes from a hash: the low 16 bits of four elements (each
/// uniform mod `p`, so these bits are uniform up to a negligible bias).
fn bytes8(h: &Hash) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (chunk, x) in out.chunks_exact_mut(2).zip(h) {
        chunk.copy_from_slice(&(x.as_u32() as u16).to_le_bytes());
    }
    out
}

/// What an output is made from: its key seed, the hash of its key tree,
/// and the tag and mask its nonce needs (from the view secret).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputKeys {
    pub r: KeySeed,
    pub hash: OutputHash,
    tag: [u8; TAG_LEN],
    mask: u64,
}

impl OutputKeys {
    /// An output paying `amount` to these keys: its nonce is `r`, the tag
    /// and the masked amount.
    pub fn pay(&self, amount: Amount) -> Output {
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..KEY_SEED_LEN].copy_from_slice(&self.r);
        nonce[KEY_SEED_LEN..KEY_SEED_LEN + TAG_LEN].copy_from_slice(&self.tag);
        nonce[KEY_SEED_LEN + TAG_LEN..].copy_from_slice(&(amount ^ self.mask).to_le_bytes());
        Output {
            hash: self.hash,
            amount,
            nonce,
        }
    }
}

/// An unspent output the wallet owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coin {
    /// The key seed in its nonce.
    pub r: KeySeed,
    /// Its slot in the state tree.
    pub position: u64,
    pub amount: Amount,
    pub record: OutputRecord,
}

/// Keys derived from a seed: one-time keys from the spend side, tags and
/// amount masks from the view side.
#[derive(Clone, Copy, Debug)]
pub struct Wallet {
    seed: Hash,
    view: Hash,
}

impl Wallet {
    pub fn from_seed(seed: Hash) -> Self {
        Self {
            seed,
            view: hash::sponge(domain::WALLET_VIEW, &[], &[&seed]),
        }
    }

    /// The one-time key pair at `leaf` of the key tree for `r`.
    pub fn key(&self, r: &KeySeed, leaf: u8) -> (SecretKey, PublicKey) {
        wots::keygen(hash::sponge(
            domain::WALLET_KEY,
            &[],
            &[&self.seed, &seed_limbs(r), &[BabyBear::new(leaf as u32)]],
        ))
    }

    /// The key tree for `r`: [`KEY_LEAVES`] key generations (in parallel).
    pub fn key_tree(&self, r: &KeySeed) -> KeyTree {
        let hashes = crate::par::par_map(KEY_LEAVES, 16, |leaf| self.key(r, leaf as u8).1.hash());
        KeyTree(hashes.try_into().expect("a hash per leaf"))
    }

    /// The tag and amount mask for `r`.
    fn tag_and_mask(&self, r: &KeySeed) -> ([u8; TAG_LEN], u64) {
        let limbs = seed_limbs(r);
        let tag = bytes8(&hash::sponge(domain::NONCE_TAG, &[], &[&self.view, &limbs]));
        let mask = bytes8(&hash::sponge(
            domain::NONCE_MASK,
            &[],
            &[&self.view, &limbs],
        ));
        (tag, u64::from_le_bytes(mask))
    }

    /// What an output with key seed `r` is made from.
    pub fn keys_for(&self, r: &KeySeed) -> OutputKeys {
        let (tag, mask) = self.tag_and_mask(r);
        OutputKeys {
            r: *r,
            hash: self.key_tree(r).output_hash(),
            tag,
            mask,
        }
    }

    /// Keys for a new output, with a fresh random seed.
    pub fn new_keys(&self) -> std::io::Result<OutputKeys> {
        let mut r = [0u8; KEY_SEED_LEN];
        std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom")?, &mut r)?;
        Ok(self.keys_for(&r))
    }

    /// The amount of `record` if its tag is ours (one hash: cheap enough
    /// to try on every output). Doesn't check the commitment.
    pub fn recognize(&self, record: &OutputRecord) -> Option<Amount> {
        let r = key_seed_of(record);
        let (tag, mask) = self.tag_and_mask(&r);
        let n = &record.nonce;
        (n[KEY_SEED_LEN..KEY_SEED_LEN + TAG_LEN] == tag).then(|| {
            u64::from_le_bytes(n[KEY_SEED_LEN + TAG_LEN..].try_into().expect("8 bytes")) ^ mask
        })
    }

    /// The key seed and amount of `record` if it's ours: its tag matches
    /// and it opens with our keys (a copied tag doesn't).
    pub fn identify(&self, record: &OutputRecord) -> Option<(KeySeed, Amount)> {
        let amount = self.recognize(record)?;
        let r = key_seed_of(record);
        let hash = self.key_tree(&r).output_hash();
        (output::commit(&hash, amount, &record.nonce) == Some(record.commitment))
            .then_some((r, amount))
    }

    /// The wallet's coins among `outputs` (slot and record, e.g. from
    /// [`crate::state::StateTree::outputs`]).
    pub fn scan(&self, outputs: &[(u64, OutputRecord)]) -> Vec<Coin> {
        crate::par::par_map(outputs.len(), 1024, |i| {
            let (position, record) = outputs[i];
            self.recognize(&record).map(|_| (position, record))
        })
        .into_iter()
        .flatten()
        .filter_map(|(position, record)| {
            let (r, amount) = self.identify(&record)?;
            Some(Coin {
                r,
                position,
                amount,
                record,
            })
        })
        .collect()
    }

    /// The input spending `coin` with the key at `leaf`, and that key.
    pub fn input(&self, coin: &Coin, leaf: u8) -> (SecretKey, Input) {
        let (sk, pubkey) = self.key(&coin.r, leaf);
        let input = Input {
            commitment: coin.record.commitment,
            position: coin.position,
            pubkey,
            leaf,
            path: self.key_tree(&coin.r).path(leaf as usize),
            amount: coin.amount,
            nonce: coin.record.nonce,
        };
        (sk, input)
    }
}

/// A payment's id: 16 random bytes.
pub type SlateId = [u8; 16];

/// What the sender gives the receiver: the payment's id, amount and fee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slate {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
}

/// What the receiver gives back: the slate as it got it, and its output for
/// the amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
    pub output: Output,
}

const SLATE_HEADER: &str = "r slate 2";
const RESPONSE_HEADER: &str = "r slate response 2";

impl Slate {
    /// The slate file: a header line, then `key value` lines.
    pub fn to_text(&self) -> String {
        format!(
            "{SLATE_HEADER}\nid {}\namount {}\nfee {}\n",
            hex(&self.id),
            self.amount,
            self.fee,
        )
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let f = Fields::parse(text, SLATE_HEADER)?;
        Ok(Self {
            id: f.id()?,
            amount: f.number("amount")?,
            fee: f.number("fee")?,
        })
    }
}

impl Response {
    pub fn to_text(&self) -> String {
        let mut out = self.output.hash.to_vec();
        out.extend(self.output.amount.to_le_bytes());
        out.extend(self.output.nonce);
        format!(
            "{RESPONSE_HEADER}\nid {}\namount {}\nfee {}\noutput {}\n",
            hex(&self.id),
            self.amount,
            self.fee,
            hex(&out),
        )
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let f = Fields::parse(text, RESPONSE_HEADER)?;
        let bytes = unhex(f.get("output")?).ok_or("bad output")?;
        if bytes.len() != OUTPUT_HASH_LEN + 8 + NONCE_LEN {
            return Err("bad output".into());
        }
        let (hash, rest) = bytes.split_at(OUTPUT_HASH_LEN);
        let (amount, nonce) = rest.split_at(8);
        let output = Output {
            hash: hash.try_into().expect("length"),
            amount: u64::from_le_bytes(amount.try_into().expect("length")),
            nonce: nonce.try_into().expect("length"),
        };
        if output.record().is_none() {
            return Err("bad output".into());
        }
        Ok(Self {
            id: f.id()?,
            amount: f.number("amount")?,
            fee: f.number("fee")?,
            output,
        })
    }
}

/// A slate file's `key value` lines.
struct Fields<'a>(HashMap<&'a str, &'a str>);

impl<'a> Fields<'a> {
    fn parse(text: &'a str, header: &str) -> Result<Self, String> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        if lines.next() != Some(header) {
            return Err(format!("not a slate file (expected \"{header}\" first)"));
        }
        let mut fields = HashMap::new();
        for line in lines {
            let (k, v) = line.split_once(' ').ok_or(format!("bad line \"{line}\""))?;
            if fields.insert(k, v.trim()).is_some() {
                return Err(format!("\"{k}\" appears twice"));
            }
        }
        Ok(Self(fields))
    }

    fn get(&self, key: &str) -> Result<&'a str, String> {
        self.0.get(key).copied().ok_or(format!("no \"{key}\""))
    }

    fn number(&self, key: &str) -> Result<Amount, String> {
        self.get(key)?.parse().map_err(|_| format!("bad \"{key}\""))
    }

    fn id(&self) -> Result<SlateId, String> {
        unhex(self.get("id")?)
            .and_then(|b| b.try_into().ok())
            .ok_or("bad id".into())
    }
}

/// Lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parses [`hex`] output (either case).
pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// The sender's side of a payment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sent {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
    /// The coins it spends (locked until it's cancelled or confirmed).
    pub coins: Vec<Coin>,
    /// The change output's key seed and amount, if any.
    pub change: Option<(KeySeed, Amount)>,
    /// When it was made (Unix seconds).
    pub created: u64,
    /// Replaced, after it was signed, by a transaction paying the coins
    /// back to the wallet ([`Sent::tx`] is then that one).
    pub cancelled: bool,
    /// The signed transaction, once finalized (and its latest replacement,
    /// after a bump or cancellation).
    pub tx: Option<Vec<u8>>,
}

/// The receiver's side of a payment: the output it made, until that shows
/// up in the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Received {
    pub id: SlateId,
    /// The output's key seed, amount and commitment.
    pub r: KeySeed,
    pub amount: Amount,
    pub commitment: Commitment,
    /// When it was made (Unix seconds).
    pub created: u64,
}

impl Received {
    fn to_bytes(self) -> Vec<u8> {
        let mut out = self.id.to_vec();
        out.extend(self.r);
        out.extend(self.amount.to_le_bytes());
        out.extend(self.commitment.0);
        out.extend(self.created.to_le_bytes());
        out
    }

    fn from_bytes(b: &[u8]) -> Option<Self> {
        const R: usize = 16 + KEY_SEED_LEN;
        if b.len() != R + 8 + COMMITMENT_LEN + 8 {
            return None;
        }
        let at = R + 8 + COMMITMENT_LEN;
        Some(Self {
            id: b[..16].try_into().ok()?,
            r: b[16..R].try_into().ok()?,
            amount: u64::from_le_bytes(b[R..R + 8].try_into().ok()?),
            commitment: Commitment(b[R + 8..at].try_into().ok()?),
            created: u64::from_le_bytes(b[at..].try_into().ok()?),
        })
    }
}

/// Bytes in an encoded coin: key seed, position, amount, commitment, nonce.
const COIN_BYTES: usize = KEY_SEED_LEN + 8 + 8 + COMMITMENT_LEN + NONCE_LEN;

impl Sent {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.id.to_vec();
        out.extend(self.amount.to_le_bytes());
        out.extend(self.fee.to_le_bytes());
        out.extend(self.created.to_le_bytes());
        out.extend((self.coins.len() as u16).to_le_bytes());
        for c in &self.coins {
            out.extend(c.r);
            out.extend(c.position.to_le_bytes());
            out.extend(c.amount.to_le_bytes());
            out.extend(c.record.commitment.0);
            out.extend(c.record.nonce);
        }
        match self.change {
            Some((r, amount)) => {
                out.push(1);
                out.extend(r);
                out.extend(amount.to_le_bytes());
            }
            None => out.push(0),
        }
        out.push(self.cancelled as u8);
        if let Some(tx) = &self.tx {
            out.extend(tx);
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = bytes;
        let mut take = |n: usize| -> Option<&[u8]> {
            let (a, b) = r.split_at_checked(n)?;
            r = b;
            Some(a)
        };
        let u64_of = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
        let id = take(16)?.try_into().ok()?;
        let amount = u64_of(take(8)?);
        let fee = u64_of(take(8)?);
        let created = u64_of(take(8)?);
        let n = u16::from_le_bytes(take(2)?.try_into().ok()?) as usize;
        let mut coins = Vec::with_capacity(n);
        for _ in 0..n {
            let c = take(COIN_BYTES)?;
            const K: usize = KEY_SEED_LEN;
            coins.push(Coin {
                r: c[..K].try_into().ok()?,
                position: u64_of(&c[K..K + 8]),
                amount: u64_of(&c[K + 8..K + 16]),
                record: OutputRecord {
                    commitment: Commitment(c[K + 16..K + 16 + COMMITMENT_LEN].try_into().ok()?),
                    nonce: c[K + 16 + COMMITMENT_LEN..].try_into().ok()?,
                },
            });
        }
        let change = match take(1)?[0] {
            0 => None,
            1 => Some((take(KEY_SEED_LEN)?.try_into().ok()?, u64_of(take(8)?))),
            _ => return None,
        };
        let cancelled = match take(1)?[0] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let tx = (!r.is_empty()).then(|| r.to_vec());
        Some(Self {
            id,
            amount,
            fee,
            coins,
            change,
            created,
            cancelled,
            tx,
        })
    }
}

/// One of the wallet's outputs, as found in a block (or the state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owned {
    /// The key seed in its nonce.
    pub r: KeySeed,
    pub amount: Amount,
    pub record: OutputRecord,
    /// The block that made it (for one found in the state, the tip then).
    pub height: u64,
    /// A mining reward of ours (which matures before it's spent).
    pub reward: bool,
    /// The height at which it was first seen gone from the state (spent),
    /// or 0.
    pub gone_since: u64,
}

const OWNED_BYTES: usize = 8 + NONCE_LEN + 8 + 1 + 8;

impl Owned {
    fn to_bytes(self) -> Vec<u8> {
        let mut out = self.amount.to_le_bytes().to_vec();
        out.extend(self.record.nonce);
        out.extend(self.height.to_le_bytes());
        out.push(self.reward as u8);
        out.extend(self.gone_since.to_le_bytes());
        out
    }

    fn from_bytes(commitment: Commitment, b: &[u8]) -> Option<Self> {
        if b.len() != OWNED_BYTES {
            return None;
        }
        let at = 8 + NONCE_LEN;
        let record = OutputRecord {
            commitment,
            nonce: b[8..at].try_into().ok()?,
        };
        Some(Self {
            r: key_seed_of(&record),
            amount: u64::from_le_bytes(b[..8].try_into().ok()?),
            record,
            height: u64::from_le_bytes(b[at..at + 8].try_into().ok()?),
            reward: b[at + 8] == 1,
            gone_since: u64::from_le_bytes(b[at + 9..].try_into().ok()?),
        })
    }
}

/// Which leaves of an output's key tree have signed, as far as the wallet
/// knows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Leaves {
    /// One bit per leaf.
    pub used: [u8; KEY_LEAVES / 8],
    /// Found on restore: earlier signatures may exist that the wallet
    /// doesn't know of, so it signs with a random leaf of the upper half,
    /// not the lowest.
    pub unknown: bool,
}

impl Leaves {
    pub fn is_used(&self, leaf: u8) -> bool {
        self.used[leaf as usize / 8] >> (leaf % 8) & 1 == 1
    }

    /// The leaf to sign with next, or `None` if every leaf has signed.
    ///
    /// A wallet signs from leaf 0 up, so an output whose history is
    /// unknown (found on restore) signs with a random unused leaf of the
    /// upper half (`random` picks): the earlier wallet's signatures are
    /// almost surely in the lower half, and an earlier restore's (whose
    /// spend never confirmed) match with odds of 1 in 128. When the upper
    /// half is used up, it picks among the rest.
    pub fn next(&self, random: u32) -> Option<u8> {
        let unused = |from: usize| -> Vec<u8> {
            (from..KEY_LEAVES)
                .map(|l| l as u8)
                .filter(|&l| !self.is_used(l))
                .collect()
        };
        if !self.unknown {
            return unused(0).first().copied();
        }
        let upper = unused(KEY_LEAVES / 2);
        let pool = if upper.is_empty() { unused(0) } else { upper };
        (!pool.is_empty()).then(|| pool[random as usize % pool.len()])
    }

    pub fn with(mut self, leaf: u8) -> Self {
        self.used[leaf as usize / 8] |= 1 << (leaf % 8);
        self
    }
}

/// The node's wallet in LMDB (database `wallet`):
/// - `entropy`: the 32 bytes behind the recovery words ([`crate::mnemonic`]);
/// - `sent/<id>`, `recv/<id>`: payments being sent and received;
/// - `out/<commitment>`: the wallet's outputs ([`Owned`]);
/// - `synced`: the height its outputs are found up to, and `blk/<height>`
///   the hash of each block scanned (to notice a reorg);
/// - `reward/<r>`: key seeds our mining rewards went to;
/// - `leaf/<commitment>`: the leaves of an output's key tree that have
///   signed ([`Leaves`]);
/// - `restoring`: set while a restored wallet looks for its outputs.
#[derive(Clone, Copy, Debug)]
pub struct WalletStore {
    db: Database<Bytes, Bytes>,
}

const ENTROPY_KEY: &[u8] = b"entropy";
const SYNCED_KEY: &[u8] = b"synced";
const RESTORING_KEY: &[u8] = b"restoring";
const OUT_PREFIX: &[u8] = b"out/";
const BLK_PREFIX: &[u8] = b"blk/";
const REWARD_PREFIX: &[u8] = b"reward/";
const LEAF_PREFIX: &[u8] = b"leaf/";
const SENT_PREFIX: &[u8] = b"sent/";
const RECV_PREFIX: &[u8] = b"recv/";

impl WalletStore {
    pub fn open(env: &Env) -> heed::Result<Self> {
        let mut txn = env.write_txn()?;
        let db = env.create_database(&mut txn, Some("wallet"))?;
        txn.commit()?;
        Ok(Self { db })
    }

    /// The entropy behind the recovery words, if there's a wallet yet.
    pub fn entropy(&self, txn: &RoTxn) -> heed::Result<Option<[u8; crate::mnemonic::ENTROPY_LEN]>> {
        Ok(self
            .db
            .get(txn, ENTROPY_KEY)?
            .and_then(|b| b.try_into().ok()))
    }

    /// Makes the wallet (there must be none yet).
    pub fn set_entropy(
        &self,
        txn: &mut RwTxn,
        entropy: &[u8; crate::mnemonic::ENTROPY_LEN],
    ) -> heed::Result<()> {
        self.db.put(txn, ENTROPY_KEY, entropy)
    }

    pub fn owned(&self, txn: &RoTxn) -> heed::Result<Vec<Owned>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, OUT_PREFIX)? {
            let (k, v) = entry?;
            let c = Commitment(
                k[OUT_PREFIX.len()..]
                    .try_into()
                    .map_err(|_| heed::Error::Encoding("bad key".into()))?,
            );
            if let Some(o) = Owned::from_bytes(c, v) {
                out.push(o);
            }
        }
        Ok(out)
    }

    pub fn put_owned(&self, txn: &mut RwTxn, o: &Owned) -> heed::Result<()> {
        self.db.put(
            txn,
            &[OUT_PREFIX, &o.record.commitment.0].concat(),
            &o.to_bytes(),
        )
    }

    pub fn remove_owned(&self, txn: &mut RwTxn, c: &Commitment) -> heed::Result<()> {
        self.db.delete(txn, &[OUT_PREFIX, &c.0].concat())?;
        Ok(())
    }

    /// The height the outputs are found up to, if any scan has run.
    pub fn synced(&self, txn: &RoTxn) -> heed::Result<Option<u64>> {
        Ok(self
            .db
            .get(txn, SYNCED_KEY)?
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes))
    }

    pub fn set_synced(&self, txn: &mut RwTxn, height: u64) -> heed::Result<()> {
        self.db.put(txn, SYNCED_KEY, &height.to_le_bytes())
    }

    /// The hash recorded for the block scanned at `height`.
    pub fn block_hash(&self, txn: &RoTxn, height: u64) -> heed::Result<Option<Vec<u8>>> {
        Ok(self
            .db
            .get(txn, &[BLK_PREFIX, &height.to_be_bytes()].concat())?
            .map(<[u8]>::to_vec))
    }

    pub fn set_block_hash(&self, txn: &mut RwTxn, height: u64, hash: &[u8]) -> heed::Result<()> {
        self.db
            .put(txn, &[BLK_PREFIX, &height.to_be_bytes()].concat(), hash)
    }

    /// Records that the output with key seed `r` is a mining reward of
    /// ours.
    pub fn mark_reward(&self, txn: &mut RwTxn, r: &KeySeed) -> heed::Result<()> {
        self.db.put(txn, &[REWARD_PREFIX, r].concat(), &[])
    }

    pub fn is_reward(&self, txn: &RoTxn, r: &KeySeed) -> heed::Result<bool> {
        Ok(self.db.get(txn, &[REWARD_PREFIX, r].concat())?.is_some())
    }

    /// What's known of which leaves of `c`'s key tree have signed.
    pub fn leaves(&self, txn: &RoTxn, c: &Commitment) -> heed::Result<Leaves> {
        Ok(self
            .db
            .get(txn, &[LEAF_PREFIX, &c.0].concat())?
            .and_then(|b| b.try_into().ok())
            .map_or(Leaves::default(), |b: [u8; KEY_LEAVES / 8 + 1]| Leaves {
                used: b[..KEY_LEAVES / 8].try_into().expect("the leaf bits"),
                unknown: b[KEY_LEAVES / 8] == 1,
            }))
    }

    pub fn set_leaves(&self, txn: &mut RwTxn, c: &Commitment, l: Leaves) -> heed::Result<()> {
        self.db.put(
            txn,
            &[LEAF_PREFIX, &c.0].concat(),
            &[&l.used[..], &[l.unknown as u8]].concat(),
        )
    }

    pub fn remove_leaves(&self, txn: &mut RwTxn, c: &Commitment) -> heed::Result<()> {
        self.db.delete(txn, &[LEAF_PREFIX, &c.0].concat())?;
        Ok(())
    }

    /// Whether a restored wallet is still looking for its outputs.
    pub fn restoring(&self, txn: &RoTxn) -> heed::Result<bool> {
        Ok(self.db.get(txn, RESTORING_KEY)?.is_some())
    }

    pub fn set_restoring(&self, txn: &mut RwTxn, on: bool) -> heed::Result<()> {
        if on {
            self.db.put(txn, RESTORING_KEY, &[])
        } else {
            self.db.delete(txn, RESTORING_KEY).map(|_| ())
        }
    }

    fn sent_key(id: &SlateId) -> Vec<u8> {
        [SENT_PREFIX, id].concat()
    }

    pub fn put_sent(&self, txn: &mut RwTxn, sent: &Sent) -> heed::Result<()> {
        self.db
            .put(txn, &Self::sent_key(&sent.id), &sent.to_bytes())
    }

    pub fn remove_sent(&self, txn: &mut RwTxn, id: &SlateId) -> heed::Result<()> {
        self.db.delete(txn, &Self::sent_key(id))?;
        Ok(())
    }

    pub fn put_received(&self, txn: &mut RwTxn, r: &Received) -> heed::Result<()> {
        self.db
            .put(txn, &[RECV_PREFIX, &r.id].concat(), &r.to_bytes())
    }

    pub fn remove_received(&self, txn: &mut RwTxn, id: &SlateId) -> heed::Result<()> {
        self.db.delete(txn, &[RECV_PREFIX, id].concat())?;
        Ok(())
    }

    /// Every payment being received, oldest first.
    pub fn receiveds(&self, txn: &RoTxn) -> heed::Result<Vec<Received>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, RECV_PREFIX)? {
            if let Some(r) = Received::from_bytes(entry?.1) {
                out.push(r);
            }
        }
        out.sort_by_key(|r| r.created);
        Ok(out)
    }

    /// Every payment being sent, oldest first.
    pub fn sents(&self, txn: &RoTxn) -> heed::Result<Vec<Sent>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, SENT_PREFIX)? {
            let (_, v) = entry?;
            if let Some(s) = Sent::from_bytes(v) {
                out.push(s);
            }
        }
        out.sort_by_key(|s| s.created);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wallet(n: u32) -> Wallet {
        Wallet::from_seed(BabyBear::new_array(core::array::from_fn(|i| {
            n * 100 + i as u32
        })))
    }

    fn record(o: &Output) -> OutputRecord {
        o.record().unwrap()
    }

    fn r(n: u8) -> KeySeed {
        [n; KEY_SEED_LEN]
    }

    #[test]
    fn nonces_carry_amounts() {
        let (alice, bob) = (wallet(1), wallet(2));
        for amount in [0, 1, 50, u64::MAX] {
            let o = alice.keys_for(&r(3)).pay(amount);
            assert_eq!(key_seed_of(&record(&o)), r(3));
            assert_eq!(alice.recognize(&record(&o)), Some(amount));
            assert_eq!(alice.identify(&record(&o)), Some((r(3), amount)));
            // Another wallet doesn't see it.
            assert_eq!(bob.recognize(&record(&o)), None);
        }
        // Different seeds get different tags and keys.
        let (a, b) = (alice.keys_for(&r(1)).pay(5), alice.keys_for(&r(2)).pay(5));
        assert_ne!(a.nonce[KEY_SEED_LEN..], b.nonce[KEY_SEED_LEN..]);
        assert_ne!(a.hash, b.hash);
        assert_ne!(alice.keys_for(&r(1)).hash, bob.keys_for(&r(1)).hash);
        assert_ne!(alice.new_keys().unwrap().r, alice.new_keys().unwrap().r);
    }

    #[test]
    fn scan_finds_own_outputs_with_amounts() {
        let (alice, bob) = (wallet(1), wallet(2));
        let outputs: Vec<(u64, OutputRecord)> = [
            alice.keys_for(&r(1)).pay(10),
            bob.keys_for(&r(1)).pay(20),
            alice.keys_for(&r(200)).pay(30),
            bob.keys_for(&r(2)).pay(40),
        ]
        .iter()
        .enumerate()
        .map(|(i, o)| (i as u64 + 1, record(o)))
        .collect();
        let found: Vec<(KeySeed, u64, Amount)> = alice
            .scan(&outputs)
            .iter()
            .map(|c| (c.r, c.position, c.amount))
            .collect();
        assert_eq!(found, [(r(1), 1, 10), (r(200), 3, 30)]);
        let bobs: Vec<Amount> = bob.scan(&outputs).iter().map(|c| c.amount).collect();
        assert_eq!(bobs, [20, 40]);
    }

    #[test]
    fn scan_rejects_a_matching_tag_with_the_wrong_commitment() {
        // An output copying one of ours's nonce (its seed, tag and masked
        // amount) to another key looks ours, but doesn't open with our keys.
        let alice = wallet(1);
        let mut fake = wallet(9).keys_for(&r(4)).pay(10);
        fake.nonce = alice.keys_for(&r(4)).pay(10).nonce;
        assert_eq!(alice.recognize(&record(&fake)), Some(10));
        assert!(alice.scan(&[(1, record(&fake))]).is_empty());
    }

    #[test]
    fn slates_and_responses_round_trip_through_text() {
        let w = wallet(2);
        let slate = Slate {
            id: *b"0123456789abcdef",
            amount: 150_000_000,
            fee: 100_000,
        };
        assert_eq!(Slate::from_text(&slate.to_text()), Ok(slate));
        let response = Response {
            id: slate.id,
            amount: slate.amount,
            fee: slate.fee,
            output: w.keys_for(&r(2)).pay(slate.amount),
        };
        assert_eq!(Response::from_text(&response.to_text()), Ok(response));
        // Each kind only reads its own.
        assert!(Slate::from_text(&response.to_text()).is_err());
        assert!(Response::from_text(&slate.to_text()).is_err());
        assert!(Slate::from_text("r slate 2\nid 00\namount 1\nfee 1\n").is_err());
        assert!(Slate::from_text("r slate 2\namount 1\n").is_err());
        assert_eq!(
            unhex(&hex(&[0, 1, 0xab, 0xff])),
            Some(vec![0, 1, 0xab, 0xff])
        );
        assert_eq!(unhex("abc"), None);
    }

    #[test]
    fn the_store_keeps_payments_and_outputs() {
        let env = crate::db::test::TempEnv::new();
        let store = WalletStore::open(env.env()).unwrap();
        let w = Wallet::from_seed(crate::mnemonic::seed_of(&[5; 32]));
        let mut txn = env.env().write_txn().unwrap();
        assert_eq!(store.entropy(&txn).unwrap(), None);
        store.set_entropy(&mut txn, &[5; 32]).unwrap();
        assert_eq!(store.entropy(&txn).unwrap(), Some([5; 32]));
        let coin = Coin {
            r: r(0),
            position: 5,
            amount: 70,
            record: record(&w.keys_for(&r(0)).pay(70)),
        };
        let mut sent = Sent {
            id: [7; 16],
            amount: 50,
            fee: 3,
            coins: vec![coin],
            change: Some((r(1), 17)),
            created: 1000,
            cancelled: false,
            tx: None,
        };
        store.put_sent(&mut txn, &sent).unwrap();
        let older = Sent {
            id: [1; 16],
            created: 10,
            change: None,
            cancelled: true,
            tx: Some(vec![1, 2, 3]),
            ..sent.clone()
        };
        store.put_sent(&mut txn, &older).unwrap();
        assert_eq!(
            store.sents(&txn).unwrap(),
            vec![older.clone(), sent.clone()]
        );
        sent.tx = Some(vec![9; 40]);
        store.put_sent(&mut txn, &sent).unwrap();
        store.remove_sent(&mut txn, &older.id).unwrap();
        assert_eq!(store.sents(&txn).unwrap(), vec![sent]);
        let got = Received {
            id: [3; 16],
            r: r(9),
            amount: 12,
            commitment: coin.record.commitment,
            created: 5,
        };
        store.put_received(&mut txn, &got).unwrap();
        assert_eq!(store.receiveds(&txn).unwrap(), vec![got]);
        // Kept apart from the payments being sent.
        assert_eq!(store.sents(&txn).unwrap().len(), 1);
        store.remove_received(&mut txn, &got.id).unwrap();
        assert!(store.receiveds(&txn).unwrap().is_empty());
        let owned = Owned {
            r: r(0),
            amount: 70,
            record: coin.record,
            height: 4,
            reward: true,
            gone_since: 9,
        };
        store.put_owned(&mut txn, &owned).unwrap();
        assert_eq!(store.owned(&txn).unwrap(), vec![owned]);
        store
            .remove_owned(&mut txn, &owned.record.commitment)
            .unwrap();
        assert!(store.owned(&txn).unwrap().is_empty());
        assert_eq!(store.synced(&txn).unwrap(), None);
        store.set_synced(&mut txn, 7).unwrap();
        store.set_block_hash(&mut txn, 7, &[1; 32]).unwrap();
        assert_eq!(store.synced(&txn).unwrap(), Some(7));
        assert_eq!(store.block_hash(&txn, 7).unwrap(), Some(vec![1; 32]));
        store.mark_reward(&mut txn, &r(3)).unwrap();
        assert!(store.is_reward(&txn, &r(3)).unwrap() && !store.is_reward(&txn, &r(4)).unwrap());
        let c = coin.record.commitment;
        assert_eq!(store.leaves(&txn, &c).unwrap(), Leaves::default());
        let l = Leaves {
            unknown: true,
            ..Leaves::default().with(0).with(2).with(255)
        };
        store.set_leaves(&mut txn, &c, l).unwrap();
        assert_eq!(store.leaves(&txn, &c).unwrap(), l);
        store.remove_leaves(&mut txn, &c).unwrap();
        assert_eq!(store.leaves(&txn, &c).unwrap(), Leaves::default());
        store.set_restoring(&mut txn, true).unwrap();
        assert!(store.restoring(&txn).unwrap());
        store.set_restoring(&mut txn, false).unwrap();
        assert!(!store.restoring(&txn).unwrap());
    }

    #[test]
    fn coins_spend_with_any_leaf_and_their_position() {
        let w = wallet(3);
        let coin = Coin {
            r: r(2),
            position: 77,
            amount: 9,
            record: record(&w.keys_for(&r(2)).pay(9)),
        };
        for leaf in [0, 5, 128, 255] {
            let (_, input) = w.input(&coin, leaf);
            assert!(input.opens_commitment());
            assert_eq!((input.position, input.leaf), (77, leaf));
        }
    }

    #[test]
    fn leaves_are_taken_in_order_or_at_random_from_the_upper_half_when_history_is_unknown() {
        let mut l = Leaves::default();
        assert_eq!(l.next(7), Some(0));
        l = l.with(0).with(1);
        assert_eq!(l.next(7), Some(2));
        l.unknown = true;
        // Seven unused leaves along the upper half, skipping the used ones.
        assert_eq!(l.next(7), Some(135));
        l = l.with(130);
        assert_eq!(l.next(7), Some(136));
        assert_eq!(l.next(127), Some(128));
        // The upper half used up, the rest.
        for leaf in 128..=255 {
            l = l.with(leaf);
        }
        assert_eq!(l.next(0), Some(2));
        let full = Leaves {
            used: [u8::MAX; KEY_LEAVES / 8],
            unknown: true,
        };
        assert_eq!(full.next(0), None);
        assert_eq!(
            Leaves {
                unknown: false,
                ..full
            }
            .next(0),
            None
        );
    }
}
