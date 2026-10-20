//! The chain state: an indexed Merkle tree of every output commitment ever
//! created, each marked spent or unspent.
//!
//! Leaves form a linked list sorted by key, stored in slots filled in order
//! from 0. Slot 0 holds a sentinel with the minimum key (zero), so every key
//! has a predecessor, the "low leaf". The low leaf of a key both proves the
//! key is absent (it falls strictly between the low leaf's key and its
//! successor's) and is the leaf that gets relinked when the key is inserted.
//!
//! Spent leaves are never removed, so a commitment can appear at most once in
//! the chain's lifetime and old transactions can never be replayed.
//!
//! [`verify_insert`] and [`verify_spend`] define the state transition from
//! proofs alone; [`StateTree`] maintains the full tree in LMDB and produces
//! the proofs.

use std::fmt;
use std::ops::Bound;
use std::sync::OnceLock;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::babybear::BabyBear;
use crate::hash::{self, DOMAIN_SLOT, HASH_BYTES, HASH_LEN, Hash, compress, domain, merge, sponge};
use crate::output::Commitment;
use crate::poseidon2::WIDTH;

pub const DEPTH: usize = 40;
pub const CAPACITY: u64 = 1 << DEPTH;

/// Hash stored in a slot that has never been filled.
const EMPTY_LEAF: Hash = [BabyBear::ZERO; HASH_LEN];

/// `next_index` value marking the end of the list. Slot 0 is the sentinel,
/// which has the smallest key and so is never anyone's successor.
const END: u64 = 0;

/// 40-bit values are hashed as two limbs: low 30 bits, then high bits.
const LIMB_BITS: u32 = 30;

pub type Siblings = [Hash; DEPTH];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub key: Hash,
    /// Key of the next leaf in sorted order; zero at the end of the list.
    pub next_key: Hash,
    /// Slot of the next leaf in sorted order, or [`END`].
    pub next_index: u64,
    pub spent: bool,
}

impl Leaf {
    const SENTINEL: Self = Self {
        key: [BabyBear::ZERO; HASH_LEN],
        next_key: [BabyBear::ZERO; HASH_LEN],
        next_index: END,
        // The sentinel is not an output and can never be spent.
        spent: true,
    };

    pub fn hash(&self) -> Hash {
        let (lo, hi) = split(self.next_index);
        let meta = [lo, hi, BabyBear::new(self.spent as u32)];
        sponge(domain::STATE_LEAF, &[], &[&self.key, &self.next_key, &meta])
    }

    /// Returns `true` if `key` falls strictly between this leaf and its successor.
    fn is_low_leaf_of(&self, key: &Hash) -> bool {
        less(&self.key, key) && (self.next_index == END || less(key, &self.next_key))
    }
}

/// What the block header commits to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct State {
    pub root: Hash,
    /// Number of filled slots, including the sentinel; the next insert goes
    /// into slot `size`.
    pub size: u64,
}

impl State {
    /// `H(root || size)`.
    pub fn commitment(&self) -> Hash {
        let (lo, hi) = split(self.size);
        let mut input = [BabyBear::ZERO; WIDTH];
        input[..HASH_LEN].copy_from_slice(&self.root);
        input[HASH_LEN] = lo;
        input[HASH_LEN + 1] = hi;
        input[DOMAIN_SLOT] = BabyBear::new(domain::STATE);
        compress(input)
    }
}

/// A leaf and the Merkle path proving it sits in slot `index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafWitness {
    pub index: u64,
    pub leaf: Leaf,
    pub siblings: Box<Siblings>,
}

/// Proof for inserting a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InsertProof {
    /// The low leaf before it is relinked.
    pub low: LeafWitness,
    /// Path to slot `size`, taken after the low leaf is relinked.
    pub new_siblings: Box<Siblings>,
}

/// Proof for spending a key: its leaf before being marked spent.
pub type SpendProof = LeafWitness;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    /// Commitment bytes are not a canonical hash, or are the sentinel key.
    InvalidKey,
    Duplicate,
    NotFound,
    AlreadySpent,
    Full,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidKey => "invalid commitment",
            Self::Duplicate => "commitment already exists",
            Self::NotFound => "commitment not found",
            Self::AlreadySpent => "commitment already spent",
            Self::Full => "state tree is full",
        })
    }
}

impl std::error::Error for StateError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofError {
    /// A Merkle path does not lead to the expected root.
    BadPath,
    /// The low leaf does not bracket the key, so the key may already exist.
    NotLowLeaf,
    KeyMismatch,
    AlreadySpent,
    Full,
    /// A commitment is not a valid tree key.
    InvalidKey,
    /// Proof counts do not match the commitment lists.
    ProofCount,
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BadPath => "Merkle path does not match the state root",
            Self::NotLowLeaf => "low leaf does not bracket the key",
            Self::KeyMismatch => "leaf key does not match",
            Self::AlreadySpent => "leaf is already spent",
            Self::Full => "state tree is full",
            Self::InvalidKey => "invalid commitment",
            Self::ProofCount => "proof count does not match the commitments",
        })
    }
}

impl std::error::Error for ProofError {}

/// Checks an insert of `key` against `state` and returns the new state.
pub fn verify_insert(state: &State, key: &Hash, proof: &InsertProof) -> Result<State, ProofError> {
    let low = &proof.low;
    if root_of(&low.leaf.hash(), low.index, &low.siblings) != state.root {
        return Err(ProofError::BadPath);
    }
    if !low.leaf.is_low_leaf_of(key) {
        return Err(ProofError::NotLowLeaf);
    }
    let index = state.size;
    if index >= CAPACITY {
        return Err(ProofError::Full);
    }

    let relinked = Leaf {
        next_key: *key,
        next_index: index,
        ..low.leaf
    };
    let mid_root = root_of(&relinked.hash(), low.index, &low.siblings);
    if root_of(&EMPTY_LEAF, index, &proof.new_siblings) != mid_root {
        return Err(ProofError::BadPath);
    }
    let leaf = Leaf {
        key: *key,
        next_key: low.leaf.next_key,
        next_index: low.leaf.next_index,
        spent: false,
    };
    Ok(State {
        root: root_of(&leaf.hash(), index, &proof.new_siblings),
        size: index + 1,
    })
}

/// Checks a spend of `key` against `state` and returns the new state.
pub fn verify_spend(state: &State, key: &Hash, proof: &SpendProof) -> Result<State, ProofError> {
    if root_of(&proof.leaf.hash(), proof.index, &proof.siblings) != state.root {
        return Err(ProofError::BadPath);
    }
    if proof.leaf.key != *key {
        return Err(ProofError::KeyMismatch);
    }
    if proof.leaf.spent {
        return Err(ProofError::AlreadySpent);
    }
    let spent = Leaf {
        spent: true,
        ..proof.leaf
    };
    Ok(State {
        root: root_of(&spent.hash(), proof.index, &proof.siblings),
        size: state.size,
    })
}

/// Proofs for applying a block's changes, in the order they were applied:
/// an insert per created commitment, then a spend per spent commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChangeProofs {
    pub inserts: Vec<InsertProof>,
    pub spends: Vec<SpendProof>,
}

/// Checks a block's state transition from proofs alone: inserts `created`,
/// then spends `spent`, in order. Inserting first lets a block spend an
/// output it creates.
pub fn verify_changes(
    state: &State,
    created: &[Commitment],
    spent: &[Commitment],
    proofs: &ChangeProofs,
) -> Result<State, ProofError> {
    let ChangeProofs { inserts, spends } = proofs;
    if inserts.len() != created.len() || spends.len() != spent.len() {
        return Err(ProofError::ProofCount);
    }
    let key = |c: &Commitment| key_of(c).map_err(|_| ProofError::InvalidKey);
    let mut state = *state;
    for (c, proof) in created.iter().zip(inserts) {
        state = verify_insert(&state, &key(c)?, proof)?;
    }
    for (c, proof) in spent.iter().zip(spends) {
        state = verify_spend(&state, &key(c)?, proof)?;
    }
    Ok(state)
}

/// The tree key for a commitment.
pub fn key_of(commitment: &Commitment) -> Result<Hash, StateError> {
    match hash::from_bytes(&commitment.0) {
        Some(key) if key != Leaf::SENTINEL.key => Ok(key),
        _ => Err(StateError::InvalidKey),
    }
}

/// Errors from the on-disk tree.
#[derive(Debug)]
pub enum Error {
    State(StateError),
    Db(heed::Error),
    /// Stored data failed to decode.
    Corrupt,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(e) => e.fmt(f),
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Corrupt => f.write_str("state database is corrupt"),
        }
    }
}

impl std::error::Error for Error {}

impl From<StateError> for Error {
    fn from(e: StateError) -> Self {
        Self::State(e)
    }
}

impl From<heed::Error> for Error {
    fn from(e: heed::Error) -> Self {
        Self::Db(e)
    }
}

/// The full tree, stored in LMDB. Every method works inside a caller-supplied
/// transaction, so a block's inserts and spends commit or abort together.
///
/// Databases, all integers big-endian so LMDB's byte order matches:
/// - `state_leaves`: slot (`u64`) to encoded [`Leaf`].
/// - `state_nodes`: level (`u8`) and index (`u64`) to node hash. Level 0
///   holds leaf hashes and level `DEPTH` the root; missing nodes are empty.
/// - `state_keys`: key as big-endian limbs (sorts in key order) to slot.
#[derive(Clone, Copy, Debug)]
pub struct StateTree {
    leaves: Database<Bytes, Bytes>,
    nodes: Database<Bytes, Bytes>,
    keys: Database<Bytes, Bytes>,
}

impl StateTree {
    /// Opens the tree in `env`, creating it with just the sentinel if new.
    pub fn open(env: &Env) -> Result<Self, Error> {
        let mut txn = env.write_txn()?;
        let tree = Self {
            leaves: env.create_database(&mut txn, Some("state_leaves"))?,
            nodes: env.create_database(&mut txn, Some("state_nodes"))?,
            keys: env.create_database(&mut txn, Some("state_keys"))?,
        };
        if tree.leaves.is_empty(&txn)? {
            tree.set_leaf(&mut txn, 0, &Leaf::SENTINEL)?;
            tree.keys.put(
                &mut txn,
                &sort_key(&Leaf::SENTINEL.key),
                &0u64.to_be_bytes(),
            )?;
        }
        txn.commit()?;
        Ok(tree)
    }

    pub fn state(&self, txn: &RoTxn) -> Result<State, Error> {
        Ok(State {
            root: self.node(txn, DEPTH, 0)?,
            size: self.leaves.len(txn)?,
        })
    }

    /// Returns the slot and leaf for `commitment`, spent or not.
    pub fn get(&self, txn: &RoTxn, commitment: &Commitment) -> Result<Option<(u64, Leaf)>, Error> {
        let Ok(key) = key_of(commitment) else {
            return Ok(None);
        };
        match self.keys.get(txn, &sort_key(&key))? {
            Some(bytes) => {
                let index = decode_u64(bytes)?;
                Ok(Some((index, self.leaf(txn, index)?)))
            }
            None => Ok(None),
        }
    }

    /// Adds `commitment` as an unspent leaf.
    pub fn insert(&self, txn: &mut RwTxn, commitment: &Commitment) -> Result<InsertProof, Error> {
        let key = key_of(commitment)?;
        let k = sort_key(&key);
        if self.keys.get(txn, &k)?.is_some() {
            return Err(StateError::Duplicate.into());
        }
        let index = self.leaves.len(txn)?;
        if index >= CAPACITY {
            return Err(StateError::Full.into());
        }
        // The sentinel has the smallest key, so a predecessor always exists.
        let range = (Bound::Unbounded, Bound::Excluded(&k[..]));
        let low_index = match self.keys.rev_range(txn, &range)?.next() {
            Some(entry) => decode_u64(entry?.1)?,
            None => return Err(Error::Corrupt),
        };
        let low = self.witness(txn, low_index)?;

        let relinked = Leaf {
            next_key: key,
            next_index: index,
            ..low.leaf
        };
        self.set_leaf(txn, low_index, &relinked)?;
        let new_siblings = self.siblings(txn, index)?;
        let leaf = Leaf {
            key,
            next_key: low.leaf.next_key,
            next_index: low.leaf.next_index,
            spent: false,
        };
        self.set_leaf(txn, index, &leaf)?;
        self.keys.put(txn, &k, &index.to_be_bytes())?;
        Ok(InsertProof { low, new_siblings })
    }

    /// Marks `commitment` spent.
    pub fn spend(&self, txn: &mut RwTxn, commitment: &Commitment) -> Result<SpendProof, Error> {
        let (index, leaf) = self.get(txn, commitment)?.ok_or(StateError::NotFound)?;
        if leaf.spent {
            return Err(StateError::AlreadySpent.into());
        }
        let proof = self.witness(txn, index)?;
        self.set_leaf(
            txn,
            index,
            &Leaf {
                spent: true,
                ..leaf
            },
        )?;
        Ok(proof)
    }

    /// Applies a block's changes: inserts `created`, then spends `spent`.
    ///
    /// On error `txn` is left partly updated; the caller should abort it.
    pub fn apply_changes(
        &self,
        txn: &mut RwTxn,
        created: &[Commitment],
        spent: &[Commitment],
    ) -> Result<ChangeProofs, Error> {
        let inserts = created
            .iter()
            .map(|c| self.insert(txn, c))
            .collect::<Result<_, _>>()?;
        let spends = spent
            .iter()
            .map(|c| self.spend(txn, c))
            .collect::<Result<_, _>>()?;
        Ok(ChangeProofs { inserts, spends })
    }

    fn leaf(&self, txn: &RoTxn, index: u64) -> Result<Leaf, Error> {
        let bytes = self.leaves.get(txn, &index.to_be_bytes())?;
        decode_leaf(bytes.ok_or(Error::Corrupt)?)
    }

    pub(crate) fn witness(&self, txn: &RoTxn, index: u64) -> Result<LeafWitness, Error> {
        Ok(LeafWitness {
            index,
            leaf: self.leaf(txn, index)?,
            siblings: self.siblings(txn, index)?,
        })
    }

    fn siblings(&self, txn: &RoTxn, index: u64) -> Result<Box<Siblings>, Error> {
        let mut siblings = Box::new([EMPTY_LEAF; DEPTH]);
        for (d, s) in siblings.iter_mut().enumerate() {
            *s = self.node(txn, d, (index >> d) ^ 1)?;
        }
        Ok(siblings)
    }

    fn node(&self, txn: &RoTxn, level: usize, index: u64) -> Result<Hash, Error> {
        match self.nodes.get(txn, &node_key(level, index))? {
            Some(bytes) => decode_hash(bytes),
            None => Ok(empty_hashes()[level]),
        }
    }

    /// Writes slot `index` (existing, or the next free one) and rehashes its path.
    fn set_leaf(&self, txn: &mut RwTxn, index: u64, leaf: &Leaf) -> Result<(), Error> {
        self.leaves
            .put(txn, &index.to_be_bytes(), &encode_leaf(leaf))?;
        let mut hash = leaf.hash();
        for d in 0..=DEPTH {
            let j = index >> d;
            self.nodes
                .put(txn, &node_key(d, j), &hash::to_bytes(&hash))?;
            if d < DEPTH {
                hash = merge(&self.node(txn, d, j & !1)?, &self.node(txn, d, j | 1)?);
            }
        }
        Ok(())
    }
}

/// Bytes in an encoded leaf: key, next key, next index, spent flag.
const LEAF_BYTES: usize = 2 * HASH_BYTES + 8 + 1;

fn encode_leaf(leaf: &Leaf) -> [u8; LEAF_BYTES] {
    let mut out = [0u8; LEAF_BYTES];
    out[..HASH_BYTES].copy_from_slice(&hash::to_bytes(&leaf.key));
    out[HASH_BYTES..2 * HASH_BYTES].copy_from_slice(&hash::to_bytes(&leaf.next_key));
    out[2 * HASH_BYTES..LEAF_BYTES - 1].copy_from_slice(&leaf.next_index.to_be_bytes());
    out[LEAF_BYTES - 1] = leaf.spent as u8;
    out
}

fn decode_leaf(bytes: &[u8]) -> Result<Leaf, Error> {
    if bytes.len() != LEAF_BYTES || bytes[LEAF_BYTES - 1] > 1 {
        return Err(Error::Corrupt);
    }
    Ok(Leaf {
        key: decode_hash(&bytes[..HASH_BYTES])?,
        next_key: decode_hash(&bytes[HASH_BYTES..2 * HASH_BYTES])?,
        next_index: decode_u64(&bytes[2 * HASH_BYTES..LEAF_BYTES - 1])?,
        spent: bytes[LEAF_BYTES - 1] == 1,
    })
}

fn decode_hash(bytes: &[u8]) -> Result<Hash, Error> {
    let bytes = bytes.try_into().map_err(|_| Error::Corrupt)?;
    hash::from_bytes(bytes).ok_or(Error::Corrupt)
}

fn decode_u64(bytes: &[u8]) -> Result<u64, Error> {
    Ok(u64::from_be_bytes(
        bytes.try_into().map_err(|_| Error::Corrupt)?,
    ))
}

fn node_key(level: usize, index: u64) -> [u8; 9] {
    let mut k = [0u8; 9];
    k[0] = level as u8;
    k[1..].copy_from_slice(&index.to_be_bytes());
    k
}

/// Key as big-endian limbs, so byte order is key order.
fn sort_key(key: &Hash) -> [u8; HASH_BYTES] {
    let mut out = [0u8; HASH_BYTES];
    for (chunk, x) in out.chunks_exact_mut(4).zip(limbs(key)) {
        chunk.copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// Root reached from `leaf_hash` in slot `index` via `siblings`.
fn root_of(leaf_hash: &Hash, index: u64, siblings: &Siblings) -> Hash {
    let mut hash = *leaf_hash;
    for (d, sibling) in siblings.iter().enumerate() {
        hash = if (index >> d) & 1 == 0 {
            merge(&hash, sibling)
        } else {
            merge(sibling, &hash)
        };
    }
    hash
}

/// `empty_hashes()[d]` is the root of an empty subtree of height `d`.
fn empty_hashes() -> &'static [Hash; DEPTH + 1] {
    static EMPTY: OnceLock<[Hash; DEPTH + 1]> = OnceLock::new();
    EMPTY.get_or_init(|| {
        let mut e = [EMPTY_LEAF; DEPTH + 1];
        for d in 1..=DEPTH {
            e[d] = merge(&e[d - 1], &e[d - 1]);
        }
        e
    })
}

/// Key order: lexicographic on canonical limbs, element 0 most significant.
fn less(a: &Hash, b: &Hash) -> bool {
    limbs(a) < limbs(b)
}

fn limbs(key: &Hash) -> [u32; HASH_LEN] {
    key.map(BabyBear::as_u32)
}

pub(crate) fn split(x: u64) -> (BabyBear, BabyBear) {
    debug_assert!(x <= CAPACITY);
    let mask = (1 << LIMB_BITS) - 1;
    (
        BabyBear::new((x & mask) as u32),
        BabyBear::new((x >> LIMB_BITS) as u32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::output::{self, NONCE_LEN};

    fn commitment(n: u32) -> Commitment {
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..4].copy_from_slice(&n.to_le_bytes());
        output::commit(&[7; 32], 1, &nonce).unwrap()
    }

    fn key(n: u32) -> Hash {
        key_of(&commitment(n)).unwrap()
    }

    fn state_err<T: fmt::Debug>(r: Result<T, Error>) -> StateError {
        match r {
            Err(Error::State(e)) => e,
            other => panic!("expected a state error, got {other:?}"),
        }
    }

    struct Fixture {
        tmp: TempEnv,
        tree: StateTree,
    }

    impl Fixture {
        /// A tree with `n` commitments inserted (in one committed transaction),
        /// checking every proof.
        fn new(n: u32) -> Self {
            let tmp = TempEnv::new();
            let tree = StateTree::open(tmp.env()).unwrap();
            let mut txn = tmp.env().write_txn().unwrap();
            for i in 0..n {
                let before = tree.state(&txn).unwrap();
                let proof = tree.insert(&mut txn, &commitment(i)).unwrap();
                assert_eq!(
                    verify_insert(&before, &key(i), &proof),
                    Ok(tree.state(&txn).unwrap())
                );
            }
            txn.commit().unwrap();
            Self { tmp, tree }
        }

        fn all_leaves(&self, txn: &RoTxn) -> Vec<Leaf> {
            let size = self.tree.state(txn).unwrap().size;
            (0..size).map(|i| self.tree.leaf(txn, i).unwrap()).collect()
        }
    }

    /// Root computed from scratch over the leaves, for cross-checking.
    fn naive_root(leaves: &[Leaf]) -> Hash {
        fn node(leaves: &[Leaf], d: usize, i: u64) -> Hash {
            if (i << d) >= leaves.len() as u64 {
                return empty_hashes()[d];
            }
            if d == 0 {
                return leaves[i as usize].hash();
            }
            merge(&node(leaves, d - 1, 2 * i), &node(leaves, d - 1, 2 * i + 1))
        }
        node(leaves, DEPTH, 0)
    }

    #[test]
    fn empty_tree_has_only_sentinel() {
        let f = Fixture::new(0);
        let txn = f.tmp.env().read_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 1);
        assert_eq!(state.root, naive_root(&[Leaf::SENTINEL]));
    }

    #[test]
    fn inserts_verify_and_match_naive_root() {
        let f = Fixture::new(40);
        let txn = f.tmp.env().read_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 41);
        assert_eq!(state.root, naive_root(&f.all_leaves(&txn)));
    }

    #[test]
    fn list_stays_sorted() {
        let f = Fixture::new(40);
        let txn = f.tmp.env().read_txn().unwrap();
        let leaves = f.all_leaves(&txn);
        let (mut i, mut count) = (0, 1);
        while leaves[i as usize].next_index != END {
            let leaf = leaves[i as usize];
            let next = leaves[leaf.next_index as usize];
            assert_eq!(leaf.next_key, next.key);
            assert!(less(&leaf.key, &next.key));
            i = leaf.next_index;
            count += 1;
        }
        assert_eq!(count, leaves.len());
    }

    #[test]
    fn spends_verify() {
        let f = Fixture::new(10);
        let mut txn = f.tmp.env().write_txn().unwrap();
        for i in [3, 0, 9] {
            let before = f.tree.state(&txn).unwrap();
            let proof = f.tree.spend(&mut txn, &commitment(i)).unwrap();
            assert_eq!(
                verify_spend(&before, &key(i), &proof),
                Ok(f.tree.state(&txn).unwrap())
            );
            assert!(f.tree.get(&txn, &commitment(i)).unwrap().unwrap().1.spent);
        }
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 11);
        assert_eq!(state.root, naive_root(&f.all_leaves(&txn)));
    }

    #[test]
    fn insert_then_spend_in_same_block() {
        let f = Fixture::new(3);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let s0 = f.tree.state(&txn).unwrap();
        let ins = f.tree.insert(&mut txn, &commitment(100)).unwrap();
        let s1 = verify_insert(&s0, &key(100), &ins).unwrap();
        let sp = f.tree.spend(&mut txn, &commitment(100)).unwrap();
        assert_eq!(
            verify_spend(&s1, &key(100), &sp),
            Ok(f.tree.state(&txn).unwrap())
        );
    }

    #[test]
    fn commitments_are_unique_forever() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let t = &f.tree;
        assert_eq!(
            state_err(t.insert(&mut txn, &commitment(2))),
            StateError::Duplicate
        );
        t.spend(&mut txn, &commitment(2)).unwrap();
        assert_eq!(
            state_err(t.spend(&mut txn, &commitment(2))),
            StateError::AlreadySpent
        );
        // Still present after being spent, so it can't be created again.
        assert_eq!(
            state_err(t.insert(&mut txn, &commitment(2))),
            StateError::Duplicate
        );
        assert_eq!(
            state_err(t.spend(&mut txn, &commitment(99))),
            StateError::NotFound
        );
    }

    #[test]
    fn rejects_invalid_keys() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        for bytes in [[0; 32], [0xff; 32]] {
            assert_eq!(
                state_err(f.tree.insert(&mut txn, &Commitment(bytes))),
                StateError::InvalidKey
            );
        }
    }

    #[test]
    fn aborted_transaction_leaves_state_unchanged() {
        let f = Fixture::new(5);
        let before = f.tree.state(&f.tmp.env().read_txn().unwrap()).unwrap();
        {
            let mut txn = f.tmp.env().write_txn().unwrap();
            f.tree.insert(&mut txn, &commitment(50)).unwrap();
            f.tree.spend(&mut txn, &commitment(1)).unwrap();
            // Dropped without commit.
        }
        let txn = f.tmp.env().read_txn().unwrap();
        assert_eq!(f.tree.state(&txn).unwrap(), before);
        assert!(f.tree.get(&txn, &commitment(50)).unwrap().is_none());
        assert!(!f.tree.get(&txn, &commitment(1)).unwrap().unwrap().1.spent);
    }

    #[test]
    fn state_persists_across_reopen() {
        let mut f = Fixture::new(8);
        let before = f.tree.state(&f.tmp.env().read_txn().unwrap()).unwrap();
        f.tmp.reopen();
        let tree = StateTree::open(f.tmp.env()).unwrap();
        let mut txn = f.tmp.env().write_txn().unwrap();
        assert_eq!(tree.state(&txn).unwrap(), before);
        // And it keeps working after the restart.
        let proof = tree.insert(&mut txn, &commitment(77)).unwrap();
        assert_eq!(
            verify_insert(&before, &key(77), &proof),
            Ok(tree.state(&txn).unwrap())
        );
    }

    #[test]
    fn verify_insert_rejects_existing_key() {
        // Reusing a valid proof for another key can't insert a duplicate:
        // an existing key is never strictly inside its own low leaf's range.
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        let proof = f.tree.insert(&mut txn, &commitment(50)).unwrap();
        for i in 0..5 {
            assert!(verify_insert(&state, &key(i), &proof).is_err());
        }
    }

    #[test]
    fn verify_insert_rejects_tampering() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        let proof = f.tree.insert(&mut txn, &commitment(50)).unwrap();

        let mut bad = proof.clone();
        bad.low.siblings[7][0] += BabyBear::ONE;
        assert_eq!(
            verify_insert(&state, &key(50), &bad),
            Err(ProofError::BadPath)
        );

        let mut bad = proof.clone();
        bad.new_siblings[0][0] += BabyBear::ONE;
        assert_eq!(
            verify_insert(&state, &key(50), &bad),
            Err(ProofError::BadPath)
        );

        // A wrong size would put the new leaf in a non-canonical slot.
        let wrong_size = State {
            size: state.size + 1,
            ..state
        };
        assert_eq!(
            verify_insert(&wrong_size, &key(50), &proof),
            Err(ProofError::BadPath)
        );
    }

    #[test]
    fn verify_spend_rejects_bad_proofs() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        let (index, _) = f.tree.get(&txn, &commitment(1)).unwrap().unwrap();
        let proof = f.tree.witness(&txn, index).unwrap();
        assert_eq!(
            verify_spend(&state, &key(2), &proof),
            Err(ProofError::KeyMismatch)
        );

        let mut bad = proof.clone();
        bad.leaf.spent = true;
        assert_eq!(
            verify_spend(&state, &key(1), &bad),
            Err(ProofError::BadPath)
        );

        // A proof taken after the spend shows the leaf already spent.
        f.tree.spend(&mut txn, &commitment(1)).unwrap();
        let after = f.tree.state(&txn).unwrap();
        let witness = f.tree.witness(&txn, index).unwrap();
        assert_eq!(
            verify_spend(&after, &key(1), &witness),
            Err(ProofError::AlreadySpent)
        );

        // The sentinel can never be spent.
        let sentinel = f.tree.witness(&txn, 0).unwrap();
        assert_eq!(
            verify_spend(&after, &Leaf::SENTINEL.key, &sentinel),
            Err(ProofError::AlreadySpent)
        );
    }

    #[test]
    fn leaf_encoding_roundtrips() {
        let leaf = Leaf {
            key: key(1),
            next_key: key(2),
            next_index: CAPACITY - 1,
            spent: true,
        };
        assert_eq!(decode_leaf(&encode_leaf(&leaf)).unwrap(), leaf);
        let mut bad = encode_leaf(&leaf);
        bad[LEAF_BYTES - 1] = 2;
        assert!(matches!(decode_leaf(&bad), Err(Error::Corrupt)));
    }

    /// Applies a block's changes and checks the proofs reproduce the new state.
    fn apply(
        f: &Fixture,
        txn: &mut RwTxn,
        created: &[u32],
        spent: &[u32],
    ) -> Result<ChangeProofs, Error> {
        let created: Vec<_> = created.iter().map(|&n| commitment(n)).collect();
        let spent: Vec<_> = spent.iter().map(|&n| commitment(n)).collect();
        let before = f.tree.state(txn).unwrap();
        let proofs = f.tree.apply_changes(txn, &created, &spent)?;
        assert_eq!(
            verify_changes(&before, &created, &spent, &proofs),
            Ok(f.tree.state(txn).unwrap())
        );
        Ok(proofs)
    }

    #[test]
    fn blocks_apply_and_verify() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        apply(&f, &mut txn, &[1, 2, 3], &[]).unwrap();
        apply(&f, &mut txn, &[4], &[1, 3]).unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 5);
        assert_eq!(state.root, naive_root(&f.all_leaves(&txn)));
        for (n, spent) in [(1, true), (2, false), (3, true), (4, false)] {
            assert_eq!(
                f.tree.get(&txn, &commitment(n)).unwrap().unwrap().1.spent,
                spent
            );
        }
    }

    #[test]
    fn block_can_spend_what_it_creates() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        apply(&f, &mut txn, &[1], &[]).unwrap();
        // Parent spends 1 and creates 2; child spends 2 and creates 3.
        apply(&f, &mut txn, &[2, 3], &[1, 2]).unwrap();
        assert!(f.tree.get(&txn, &commitment(2)).unwrap().unwrap().1.spent);
    }

    #[test]
    fn invalid_block_is_rolled_back_by_aborting() {
        let f = Fixture::new(3);
        let before = f.tree.state(&f.tmp.env().read_txn().unwrap()).unwrap();
        let cases: [(&[u32], &[u32], StateError); 4] = [
            (&[10], &[20], StateError::NotFound),
            (&[10], &[0, 0], StateError::AlreadySpent),
            (&[10, 1], &[], StateError::Duplicate),
            (&[10, 10], &[], StateError::Duplicate),
        ];
        for (created, spent, expected) in cases {
            let mut txn = f.tmp.env().write_txn().unwrap();
            assert_eq!(state_err(apply(&f, &mut txn, created, spent)), expected);
            txn.abort();
            let txn = f.tmp.env().read_txn().unwrap();
            assert_eq!(f.tree.state(&txn).unwrap(), before);
        }
    }

    #[test]
    fn verify_changes_rejects_mismatched_proofs() {
        let f = Fixture::new(2);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let s0 = f.tree.state(&txn).unwrap();
        let (created, spent) = ([commitment(5), commitment(6)], [commitment(0)]);
        let proofs = f.tree.apply_changes(&mut txn, &created, &spent).unwrap();

        let mut short = proofs.clone();
        short.inserts.pop();
        assert_eq!(
            verify_changes(&s0, &created, &spent, &short),
            Err(ProofError::ProofCount)
        );
        let mut swapped = proofs.clone();
        swapped.inserts.swap(0, 1);
        assert!(verify_changes(&s0, &created, &spent, &swapped).is_err());
        assert!(verify_changes(&s0, &created, &[commitment(1)], &proofs).is_err());
    }

    #[test]
    fn state_commitment_binds_size() {
        let s = State {
            root: key(1),
            size: 3,
        };
        let t = State { size: 4, ..s };
        assert_ne!(s.commitment(), t.commitment());
    }
}
