//! The chain state: an indexed Merkle tree of the unspent outputs, each leaf
//! holding an output's commitment and its nonce.
//!
//! Leaves form a linked list sorted by key, stored in slots. Slot 0 holds a
//! sentinel with the minimum key (zero), so every key has a predecessor, the
//! "low leaf". The low leaf of a key both proves the key is absent (it falls
//! strictly between the low leaf's key and its successor's) and is the leaf
//! that gets relinked when the key is inserted.
//!
//! New outputs take the next slot; the slot counter (the state's `size`)
//! only grows, so a slot number identifies one output instance for the
//! chain's lifetime. Spending removes the leaf: its predecessor is relinked
//! past it and its slot is emptied. A commitment can therefore appear again
//! once spent (in a new slot); transactions sign the slot of each output
//! they spend, so an old transaction can't be replayed against it. The tree
//! stores only unspent outputs, which is all a syncing node downloads.
//!
//! [`verify_insert`] and [`verify_remove`] define the state transition from
//! proofs alone; [`StateTree`] maintains the full tree in LMDB and produces
//! the proofs.
//!
//! **History and state sync.** The tree records each node's value before
//! every block that changes it ([`StateTree::apply_block`]), for a window of
//! recent blocks ([`StateTree::prune_history`]), so it can serve the tree
//! as it was after any of them ([`StateTree::sync_reply`]). A syncing node
//! downloads a tree from the root down ([`StateTree::sync_put`]): each
//! reply holds a node's descendants a few levels down (its leaves, at the
//! bottom), checked against the node's hash before it's stored, so any
//! peer can serve any part and a wrong one is caught at once. Progress is
//! kept in LMDB, so the download resumes after a restart, and the finished
//! leaves become the tree ([`StateTree::sync_finish`]).

use std::collections::HashSet;
use std::fmt;
use std::ops::Bound;
use std::sync::OnceLock;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::babybear::BabyBear;
use crate::hash::{self, DOMAIN_SLOT, HASH_BYTES, HASH_LEN, Hash, compress, domain, merge, sponge};
use crate::output::{self, Commitment, NONCE_LEN, NONCE_LIMBS, Nonce, OutputRecord};
use crate::poseidon2::WIDTH;

pub const DEPTH: usize = 40;
pub const CAPACITY: u64 = 1 << DEPTH;

/// Hash stored in an empty slot.
const EMPTY_LEAF: Hash = [BabyBear::ZERO; HASH_LEN];

/// `next_index` value marking the end of the list. Slot 0 is the sentinel,
/// which has the smallest key and so is never anyone's successor.
const END: u64 = 0;

/// 40-bit values are hashed as two limbs: low 30 bits, then high bits.
const LIMB_BITS: u32 = 30;

/// Elements in a leaf hash: key, nonce limbs, next key, next index lo/hi,
/// height.
pub const LEAF_LEN: usize = 2 * HASH_LEN + NONCE_LIMBS + 3;
/// Nonce limbs after the first block's worth, hashed last (so each part of
/// a leaf starts a sponge block).
pub const NONCE_TAIL: usize = NONCE_LIMBS - HASH_LEN;
const _: () = assert!(NONCE_LIMBS >= HASH_LEN && 3 + NONCE_TAIL <= HASH_LEN);

pub type Siblings = [Hash; DEPTH];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Leaf {
    pub key: Hash,
    pub nonce: Nonce,
    /// Key of the next leaf in sorted order; zero at the end of the list.
    pub next_key: Hash,
    /// Slot of the next leaf in sorted order, or [`END`].
    pub next_index: u64,
    /// Height of the block that created the output (0 for the sentinel):
    /// what relative timelocks count from.
    pub height: u32,
}

impl Leaf {
    pub const SENTINEL: Self = Self {
        key: [BabyBear::ZERO; HASH_LEN],
        nonce: [0; NONCE_LEN],
        next_key: [BabyBear::ZERO; HASH_LEN],
        next_index: END,
        height: 0,
    };

    /// `sponge(STATE_LEAF, [key, nonce limbs 0..8, next key, next index
    /// lo, hi, nonce limbs 8.., height])`.
    pub fn hash(&self) -> Hash {
        let (lo, hi) = split(self.next_index);
        let nonce = output::nonce_limbs(&self.nonce);
        sponge(
            domain::STATE_LEAF,
            &[],
            &[
                &self.key,
                &nonce[..HASH_LEN],
                &self.next_key,
                &[lo, hi],
                &nonce[HASH_LEN..],
                &[BabyBear::new(self.height)],
            ],
        )
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
    /// Slots ever used, including the sentinel's; the next insert goes into
    /// slot `size`.
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

/// Proof for removing (spending) a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoveProof {
    /// The key's predecessor before it is relinked past the key.
    pub pred: LeafWitness,
    /// The removed leaf, with its path taken after the predecessor is
    /// relinked. Its index is the spent output's position.
    pub leaf: LeafWitness,
}

impl RemoveProof {
    /// The slot of the removed output.
    pub fn position(&self) -> u64 {
        self.leaf.index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    /// Commitment bytes are not a canonical hash, or are the sentinel key.
    InvalidKey,
    /// An unspent output already has this commitment.
    Duplicate,
    NotFound,
    Full,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidKey => "invalid commitment",
            Self::Duplicate => "an unspent output already has this commitment",
            Self::NotFound => "commitment not found",
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
    /// The removed leaf's key isn't the spent key.
    KeyMismatch,
    /// The predecessor doesn't link to the removed leaf.
    NotLinked,
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
            Self::NotLinked => "predecessor does not link to the leaf",
            Self::Full => "state tree is full",
            Self::InvalidKey => "invalid commitment",
            Self::ProofCount => "proof count does not match the commitments",
        })
    }
}

impl std::error::Error for ProofError {}

/// Checks an insert of `key` with `nonce`, created at `height`, against
/// `state` and returns the new state.
pub fn verify_insert(
    state: &State,
    key: &Hash,
    nonce: &Nonce,
    height: u32,
    proof: &InsertProof,
) -> Result<State, ProofError> {
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
        nonce: *nonce,
        next_key: low.leaf.next_key,
        next_index: low.leaf.next_index,
        height,
    };
    Ok(State {
        root: root_of(&leaf.hash(), index, &proof.new_siblings),
        size: index + 1,
    })
}

/// Checks a removal of `key` against `state` and returns the new state.
/// The removed position is `proof.position()`.
pub fn verify_remove(state: &State, key: &Hash, proof: &RemoveProof) -> Result<State, ProofError> {
    let pred = &proof.pred;
    if root_of(&pred.leaf.hash(), pred.index, &pred.siblings) != state.root {
        return Err(ProofError::BadPath);
    }
    let removed = &proof.leaf;
    if removed.leaf.key != *key {
        return Err(ProofError::KeyMismatch);
    }
    if pred.leaf.next_key != *key || pred.leaf.next_index != removed.index {
        return Err(ProofError::NotLinked);
    }
    let relinked = Leaf {
        next_key: removed.leaf.next_key,
        next_index: removed.leaf.next_index,
        ..pred.leaf
    };
    let mid_root = root_of(&relinked.hash(), pred.index, &pred.siblings);
    if root_of(&removed.leaf.hash(), removed.index, &removed.siblings) != mid_root {
        return Err(ProofError::BadPath);
    }
    Ok(State {
        root: root_of(&EMPTY_LEAF, removed.index, &removed.siblings),
        size: state.size,
    })
}

/// Proofs for applying a block's changes, in the order they were applied:
/// an insert per created output, then a removal per spent commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChangeProofs {
    pub inserts: Vec<InsertProof>,
    pub removals: Vec<RemoveProof>,
}

impl ChangeProofs {
    /// What undoes these changes on a tree whose slot counter was `size`
    /// before them: each touched slot's value before the first change to it.
    pub fn undo(&self, size: u64) -> Undo {
        let mut slots: Vec<(u64, Option<Leaf>)> = Vec::new();
        let mut note = |index: u64, old: Option<Leaf>| {
            if !slots.iter().any(|(i, _)| *i == index) {
                slots.push((index, old));
            }
        };
        for (next, p) in (size..).zip(&self.inserts) {
            note(p.low.index, Some(p.low.leaf));
            note(next, None);
        }
        for p in &self.removals {
            note(p.pred.index, Some(p.pred.leaf));
            note(p.leaf.index, Some(p.leaf.leaf));
        }
        Undo { size, slots }
    }
}

/// Leaf changes staged for one [`StateTree::flush`].
#[derive(Default)]
struct Batch {
    /// Each touched slot's leaf before its first change, in order.
    undo: Vec<(u64, Option<Leaf>)>,
    seen: HashSet<u64>,
}

/// How to undo one block's changes to the tree ([`StateTree::revert`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Undo {
    /// The slot counter before.
    pub size: u64,
    /// Each touched slot's leaf before (`None`: empty).
    pub slots: Vec<(u64, Option<Leaf>)>,
}

impl Undo {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.size.to_be_bytes().to_vec();
        out.extend((self.slots.len() as u32).to_be_bytes());
        for (index, leaf) in &self.slots {
            out.extend(index.to_be_bytes());
            match leaf {
                Some(l) => {
                    out.push(1);
                    out.extend(encode_leaf(l));
                }
                None => out.push(0),
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let (size, mut rest) = bytes.split_at_checked(8)?;
        let size = u64::from_be_bytes(size.try_into().ok()?);
        let (n, r) = rest.split_at_checked(4)?;
        rest = r;
        let n = u32::from_be_bytes(n.try_into().ok()?) as usize;
        let mut slots = Vec::with_capacity(n.min(rest.len() / 9));
        for _ in 0..n {
            let (index, r) = rest.split_at_checked(8)?;
            let (&flag, r) = r.split_first()?;
            let index = u64::from_be_bytes(index.try_into().ok()?);
            rest = r;
            let leaf = match flag {
                0 => None,
                1 => {
                    let (l, r) = rest.split_at_checked(LEAF_BYTES)?;
                    rest = r;
                    Some(decode_leaf(l).ok()?)
                }
                _ => return None,
            };
            slots.push((index, leaf));
        }
        rest.is_empty().then_some(Self { size, slots })
    }
}

/// Checks the state transition of the block at `height` from proofs alone:
/// inserts `created`, then removes `spent`, in order.
pub fn verify_changes(
    state: &State,
    height: u64,
    created: &[OutputRecord],
    spent: &[Commitment],
    proofs: &ChangeProofs,
) -> Result<State, ProofError> {
    let ChangeProofs { inserts, removals } = proofs;
    if inserts.len() != created.len() || removals.len() != spent.len() {
        return Err(ProofError::ProofCount);
    }
    let key = |c: &Commitment| key_of(c).map_err(|_| ProofError::InvalidKey);
    let mut state = *state;
    for (o, proof) in created.iter().zip(inserts) {
        state = verify_insert(
            &state,
            &key(&o.commitment)?,
            &o.nonce,
            leaf_height(height),
            proof,
        )?;
    }
    for (c, proof) in spent.iter().zip(removals) {
        state = verify_remove(&state, &key(c)?, proof)?;
    }
    Ok(state)
}

/// The state of a new tree: just the sentinel, in slot 0.
pub fn empty_state() -> State {
    State {
        root: root_of(
            &Leaf::SENTINEL.hash(),
            0,
            &core::array::from_fn(|d| empty_hashes()[d]),
        ),
        size: 1,
    }
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
    /// [`StateTree::sync_finish`] needs a new tree.
    NotEmpty,
    /// The database is from an earlier, incompatible version.
    OldLayout,
    /// The state download still has nodes to fetch.
    Incomplete,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(e) => e.fmt(f),
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Corrupt => f.write_str("state database is corrupt"),
            Self::NotEmpty => f.write_str("the state tree isn't empty"),
            Self::OldLayout => f.write_str(
                "the database is from an earlier, incompatible version: delete it and sync again",
            ),
            Self::Incomplete => f.write_str("the state download isn't complete"),
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
/// transaction, so a block's inserts and removals commit or abort together.
///
/// Databases, all integers big-endian so LMDB's byte order matches:
/// - `state_leaves`: slot (`u64`) to stored leaf, for occupied slots: key,
///   nonce and next index. The next key is the next slot's key, so it isn't
///   stored ([`StateTree::leaf`]).
/// - `state_nodes`: level (`u8`) and index (`u64`) to node hash, for every
///   [`STRIDE`]th level (up to the root, at `DEPTH`); the levels between
///   are hashed from the stored level below when needed. Nodes of empty
///   subtrees aren't stored.
/// - `state_keys`: key as big-endian limbs (sorts in key order) to slot.
/// - `state_meta`: `size` to the slot counter; `history` to the lowest
///   height the tree can be served at; `sync` to the tag of a state
///   download in progress; `layout` to [`LAYOUT`].
/// - `state_history`: level (`u8`), index and height (`u64`s) to the node's
///   value before the block at that height changed it: at level 0 the
///   stored leaf, at stored levels the hash; empty if it was empty.
/// - `state_changes`: height, level and index (no value): the same entries
///   by height, for pruning and reverting.
/// - `sync_todo`: level and index to hash: nodes of a state download still
///   to fetch, each checked against its hash when it arrives.
/// - `sync_leaves`: slot to encoded leaf: the download's leaves so far.
#[derive(Clone, Copy, Debug)]
pub struct StateTree {
    leaves: Database<Bytes, Bytes>,
    nodes: Database<Bytes, Bytes>,
    keys: Database<Bytes, Bytes>,
    meta: Database<Bytes, Bytes>,
    history: Database<Bytes, Bytes>,
    changes: Database<Bytes, Bytes>,
    sync_todo: Database<Bytes, Bytes>,
    sync_leaves: Database<Bytes, Bytes>,
}

const SIZE_KEY: &[u8] = b"size";
const HISTORY_KEY: &[u8] = b"history";
const SYNC_KEY: &[u8] = b"sync";
const LAYOUT_KEY: &[u8] = b"layout";

/// The on-disk layout: 3 stores every [`STRIDE`]th level and leaves with
/// their creation height but without their next key. Earlier layouts
/// (2, and 1 with no `layout` entry) belong to chains whose leaves lacked
/// heights, which this version can't continue, so they're refused.
const LAYOUT: u8 = 3;

/// Levels between stored node levels. Hashing a missing level takes at
/// most `2^STRIDE` stored nodes (or leaves) from the stored level below.
pub const STRIDE: usize = 4;
const _: () = assert!(DEPTH.is_multiple_of(STRIDE));

/// Whether node level `level` is stored (the leaves' hashes aren't).
fn is_stored(level: usize) -> bool {
    level > 0 && level.is_multiple_of(STRIDE)
}

/// Slots (or nodes) under one node at a stored level, one stored level
/// down.
const GROUP: usize = 1 << STRIDE;

/// Levels below an interior node whose hashes a state-sync reply carries
/// (`2^5 = 32` hashes).
pub const SYNC_HASH_LEVELS: usize = 5;
/// The level whose nodes are fetched as their leaves (`2^3 = 8`).
pub const SYNC_LEAF_LEVEL: usize = 3;
/// The longest state-sync reply: a bitmap and 32 hashes.
pub const MAX_SYNC_REPLY: usize = 4 + (1 << SYNC_HASH_LEVELS) * HASH_BYTES;

/// The level of the nodes in a state-sync reply for a node at `level`.
fn reply_level(level: usize) -> usize {
    level.saturating_sub(SYNC_HASH_LEVELS).max(SYNC_LEAF_LEVEL)
}

/// What became of a state-sync reply ([`StateTree::sync_put`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncPut {
    /// Checked against its node's hash and stored.
    Stored,
    /// Not for a node still wanted (say, a duplicate).
    Unwanted,
    /// Malformed, or doesn't hash to the node: its sender misbehaved.
    Bad,
}

impl StateTree {
    /// Opens the tree in `env`, creating it with just the sentinel if new.
    pub fn open(env: &Env) -> Result<Self, Error> {
        let mut txn = env.write_txn()?;
        let tree = Self {
            leaves: env.create_database(&mut txn, Some("state_leaves"))?,
            nodes: env.create_database(&mut txn, Some("state_nodes"))?,
            keys: env.create_database(&mut txn, Some("state_keys"))?,
            meta: env.create_database(&mut txn, Some("state_meta"))?,
            history: env.create_database(&mut txn, Some("state_history"))?,
            changes: env.create_database(&mut txn, Some("state_changes"))?,
            sync_todo: env.create_database(&mut txn, Some("sync_todo"))?,
            sync_leaves: env.create_database(&mut txn, Some("sync_leaves"))?,
        };
        if tree.meta.get(&txn, SIZE_KEY)?.is_none() {
            tree.meta.put(&mut txn, LAYOUT_KEY, &[LAYOUT])?;
            tree.set_leaf(&mut txn, 0, &Leaf::SENTINEL, None)?;
            tree.keys.put(
                &mut txn,
                &sort_key(&Leaf::SENTINEL.key),
                &0u64.to_be_bytes(),
            )?;
            tree.meta.put(&mut txn, SIZE_KEY, &1u64.to_be_bytes())?;
        }
        if tree.meta.get(&txn, LAYOUT_KEY)? != Some(&[LAYOUT][..]) {
            return Err(Error::OldLayout);
        }
        txn.commit()?;
        Ok(tree)
    }

    pub fn state(&self, txn: &RoTxn) -> Result<State, Error> {
        Ok(State {
            root: self.node(txn, DEPTH, 0)?,
            size: self.size(txn)?,
        })
    }

    fn size(&self, txn: &RoTxn) -> Result<u64, Error> {
        decode_u64(self.meta.get(txn, SIZE_KEY)?.ok_or(Error::Corrupt)?)
    }

    /// Returns the slot and leaf of the unspent output with `commitment`.
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

    /// The commitment of the unspent output in slot `index`, if any.
    pub fn commitment_at(&self, txn: &RoTxn, index: u64) -> Result<Option<Commitment>, Error> {
        if index == 0 {
            return Ok(None);
        }
        match self.leaves.get(txn, &index.to_be_bytes())? {
            Some(bytes) => Ok(Some(Commitment(hash::to_bytes(&stored_key(bytes)?)))),
            None => Ok(None),
        }
    }

    /// Every unspent output with its slot, in slot order: what a wallet
    /// scans and a syncing node downloads.
    pub fn outputs(&self, txn: &RoTxn) -> Result<Vec<(u64, OutputRecord)>, Error> {
        let mut out = Vec::new();
        for entry in self.leaves.iter(txn)? {
            let (k, v) = entry?;
            let index = decode_u64(k)?;
            if index == 0 {
                continue;
            }
            let (key, nonce, ..) = decode_stored(v)?;
            out.push((
                index,
                OutputRecord {
                    commitment: Commitment(hash::to_bytes(&key)),
                    nonce,
                },
            ));
        }
        Ok(out)
    }

    /// Adds `record`, created by the block at `height`, as an unspent
    /// output in the next slot.
    pub fn insert(
        &self,
        txn: &mut RwTxn,
        record: &OutputRecord,
        height: u64,
    ) -> Result<InsertProof, Error> {
        self.insert_at(txn, record, height, None)
    }

    /// [`StateTree::insert`], recording history for the block at `at`.
    fn insert_at(
        &self,
        txn: &mut RwTxn,
        record: &OutputRecord,
        height: u64,
        at: Option<u64>,
    ) -> Result<InsertProof, Error> {
        let key = key_of(&record.commitment)?;
        let k = sort_key(&key);
        if self.keys.get(txn, &k)?.is_some() {
            return Err(StateError::Duplicate.into());
        }
        let index = self.size(txn)?;
        if index >= CAPACITY {
            return Err(StateError::Full.into());
        }
        let low_index = self.predecessor(txn, &k)?;
        let low = self.witness(txn, low_index)?;

        let relinked = Leaf {
            next_key: key,
            next_index: index,
            ..low.leaf
        };
        let leaf = Leaf {
            key,
            nonce: record.nonce,
            next_key: low.leaf.next_key,
            next_index: low.leaf.next_index,
            height: leaf_height(height),
        };
        // A stored leaf's next key is read from the next slot, so the
        // relinked low leaf needs the new one in place. Slot `index`'s
        // siblings don't depend on what it holds: they're as they were
        // between the two changes.
        self.put_leaf(txn, index, Some(&leaf), at)?;
        self.put_leaf(txn, low_index, Some(&relinked), at)?;
        self.flush(txn, &[low_index, index], at)?;
        let new_siblings = self.siblings(txn, index)?;
        self.keys.put(txn, &k, &index.to_be_bytes())?;
        self.meta.put(txn, SIZE_KEY, &(index + 1).to_be_bytes())?;
        Ok(InsertProof { low, new_siblings })
    }

    /// Removes the unspent output with `commitment`.
    pub fn remove(&self, txn: &mut RwTxn, commitment: &Commitment) -> Result<RemoveProof, Error> {
        self.remove_at(txn, commitment, None)
    }

    /// [`StateTree::remove`], recording history for the block at `at`.
    fn remove_at(
        &self,
        txn: &mut RwTxn,
        commitment: &Commitment,
        at: Option<u64>,
    ) -> Result<RemoveProof, Error> {
        let key = key_of(commitment)?;
        let k = sort_key(&key);
        let index = match self.keys.get(txn, &k)? {
            Some(bytes) => decode_u64(bytes)?,
            None => return Err(StateError::NotFound.into()),
        };
        let pred_index = self.predecessor(txn, &k)?;
        let pred = self.witness(txn, pred_index)?;
        let removed = self.leaf(txn, index)?;
        let relinked = Leaf {
            next_key: removed.next_key,
            next_index: removed.next_index,
            ..pred.leaf
        };
        self.set_leaf(txn, pred_index, &relinked, at)?;
        let leaf = self.witness(txn, index)?;
        self.put_leaf(txn, index, None, at)?;
        self.flush(txn, &[index], at)?;
        self.keys.delete(txn, &k)?;
        Ok(RemoveProof { pred, leaf })
    }

    /// Applies the changes of the block at `height`: inserts `created`,
    /// then removes `spent`.
    ///
    /// On error `txn` is left partly updated; the caller should abort it.
    pub fn apply_changes(
        &self,
        txn: &mut RwTxn,
        height: u64,
        created: &[OutputRecord],
        spent: &[Commitment],
    ) -> Result<ChangeProofs, Error> {
        self.apply_at(txn, height, None, created, spent)
    }

    /// The changes of [`StateTree::apply_changes`] for the block at
    /// `height`, recording each changed node's value before it, so the tree
    /// as it was before the block can still be served. Returns what undoes
    /// them.
    ///
    /// Checking a block needs only the result, not each step's witnesses,
    /// so the leaves change first and the nodes above them are rehashed
    /// once each at the end ([`StateTree::flush`]), not once per change:
    /// a block's new outputs sit side by side and share their paths.
    pub fn apply_block(
        &self,
        txn: &mut RwTxn,
        height: u64,
        created: &[OutputRecord],
        spent: &[Commitment],
    ) -> Result<Undo, Error> {
        let size = self.size(txn)?;
        let at = Some(height);
        let mut b = Batch::default();
        for o in created {
            self.insert_staged(txn, o, height, &mut b)?;
        }
        for c in spent {
            self.remove_staged(txn, c, at, &mut b)?;
        }
        let slots: Vec<u64> = b.undo.iter().map(|&(i, _)| i).collect();
        self.flush(txn, &slots, at)?;
        Ok(Undo {
            size,
            slots: b.undo,
        })
    }

    /// [`StateTree::insert_at`] for the block at `height`, recording
    /// history, with the leaves staged in `b`.
    fn insert_staged(
        &self,
        txn: &mut RwTxn,
        record: &OutputRecord,
        height: u64,
        b: &mut Batch,
    ) -> Result<(), Error> {
        let at = Some(height);
        let key = key_of(&record.commitment)?;
        let k = sort_key(&key);
        if self.keys.get(txn, &k)?.is_some() {
            return Err(StateError::Duplicate.into());
        }
        let index = self.size(txn)?;
        if index >= CAPACITY {
            return Err(StateError::Full.into());
        }
        let low_index = self.predecessor(txn, &k)?;
        let low = self.leaf(txn, low_index)?;
        let relinked = Leaf {
            next_key: key,
            next_index: index,
            ..low
        };
        self.stage(txn, low_index, Some(&relinked), at, b)?;
        let leaf = Leaf {
            key,
            nonce: record.nonce,
            next_key: low.next_key,
            next_index: low.next_index,
            height: leaf_height(height),
        };
        self.stage(txn, index, Some(&leaf), at, b)?;
        self.keys.put(txn, &k, &index.to_be_bytes())?;
        self.meta.put(txn, SIZE_KEY, &(index + 1).to_be_bytes())?;
        Ok(())
    }

    /// [`StateTree::remove_at`] with the leaves staged in `b`.
    fn remove_staged(
        &self,
        txn: &mut RwTxn,
        commitment: &Commitment,
        at: Option<u64>,
        b: &mut Batch,
    ) -> Result<(), Error> {
        let key = key_of(commitment)?;
        let k = sort_key(&key);
        let index = match self.keys.get(txn, &k)? {
            Some(bytes) => decode_u64(bytes)?,
            None => return Err(StateError::NotFound.into()),
        };
        let pred_index = self.predecessor(txn, &k)?;
        let pred = self.leaf(txn, pred_index)?;
        let removed = self.leaf(txn, index)?;
        let relinked = Leaf {
            next_key: removed.next_key,
            next_index: removed.next_index,
            ..pred
        };
        self.stage(txn, pred_index, Some(&relinked), at, b)?;
        self.stage(txn, index, None, at, b)?;
        self.keys.delete(txn, &k)?;
        Ok(())
    }

    /// Writes (or empties) slot `index`'s leaf without rehashing above it,
    /// noting in `b` what it held first and recording history for the block
    /// at `at`, if given.
    fn stage(
        &self,
        txn: &mut RwTxn,
        index: u64,
        leaf: Option<&Leaf>,
        at: Option<u64>,
        b: &mut Batch,
    ) -> Result<(), Error> {
        if b.seen.insert(index) {
            let old = match self.leaves.get(txn, &index.to_be_bytes())? {
                Some(_) => Some(self.leaf(txn, index)?),
                None => None,
            };
            b.undo.push((index, old));
        }
        self.put_leaf(txn, index, leaf, at)
    }

    /// Writes (or empties) slot `index`'s leaf alone, recording history
    /// for the block at `at`, if given; [`StateTree::flush`] rehashes.
    fn put_leaf(
        &self,
        txn: &mut RwTxn,
        index: u64,
        leaf: Option<&Leaf>,
        at: Option<u64>,
    ) -> Result<(), Error> {
        self.journal_leaf(txn, index, at)?;
        match leaf {
            Some(leaf) => self
                .leaves
                .put(txn, &index.to_be_bytes(), &encode_stored(leaf))?,
            None => {
                self.leaves.delete(txn, &index.to_be_bytes())?;
            }
        }
        Ok(())
    }

    /// Rehashes the stored nodes above `slots` (whose leaves changed),
    /// a stored level at a time, each node once (hashing in parallel),
    /// recording history for the block at `at`, if given. Each comes from
    /// the [`GROUP`] nodes (or leaves) under it one stored level down.
    fn flush(&self, txn: &mut RwTxn, slots: &[u64], at: Option<u64>) -> Result<(), Error> {
        let mut level = slots.to_vec();
        level.sort_unstable();
        level.dedup();
        for top in (STRIDE..=DEPTH).step_by(STRIDE) {
            let base = top - STRIDE;
            level = level.iter().map(|j| j >> STRIDE).collect();
            level.dedup();
            let hashes = if base == 0 {
                let leaves = level
                    .iter()
                    .map(|&t| self.group_leaves(txn, t))
                    .collect::<Result<Vec<_>, Error>>()?;
                crate::par::par_map(level.len(), 16, |n| {
                    fold(&leaves[n].map(|l| l.map_or(EMPTY_LEAF, |l| l.hash())), 0)
                })
            } else {
                let nodes = level
                    .iter()
                    .map(|&t| self.group_nodes(txn, base, t))
                    .collect::<Result<Vec<_>, Error>>()?;
                crate::par::par_map(level.len(), 64, |n| fold(&nodes[n], base))
            };
            for (&j, hash) in level.iter().zip(&hashes) {
                let key = node_key(top, j);
                if let Some(height) = at {
                    let old = self
                        .nodes
                        .get(txn, &key)?
                        .map(<[u8]>::to_vec)
                        .unwrap_or_default();
                    self.journal(txn, top, j, height, &old)?;
                }
                if *hash == empty_hashes()[top] {
                    self.nodes.delete(txn, &key)?;
                } else {
                    self.nodes.put(txn, &key, &hash::to_bytes(hash))?;
                }
            }
        }
        Ok(())
    }

    /// The leaves in the [`GROUP`] slots under stored node (`STRIDE`, `t`).
    fn group_leaves(&self, txn: &RoTxn, t: u64) -> Result<[Option<Leaf>; GROUP], Error> {
        let mut out = [None; GROUP];
        let (lo, hi) = (
            (t << STRIDE).to_be_bytes(),
            ((t + 1) << STRIDE).to_be_bytes(),
        );
        let range = (Bound::Included(&lo[..]), Bound::Excluded(&hi[..]));
        for entry in self.leaves.range(txn, &range)? {
            let (k, v) = entry?;
            let index = decode_u64(k)?;
            out[(index - (t << STRIDE)) as usize] = Some(self.with_next_key(txn, v)?);
        }
        Ok(out)
    }

    /// The hashes of the [`GROUP`] stored nodes at level `base` under node
    /// (`base + STRIDE`, `t`).
    fn group_nodes(&self, txn: &RoTxn, base: usize, t: u64) -> Result<[Hash; GROUP], Error> {
        let mut out = [empty_hashes()[base]; GROUP];
        let (lo, hi) = (
            node_key(base, t << STRIDE),
            node_key(base, (t + 1) << STRIDE),
        );
        let range = (Bound::Included(&lo[..]), Bound::Excluded(&hi[..]));
        for entry in self.nodes.range(txn, &range)? {
            let (k, v) = entry?;
            let index = decode_u64(&k[1..])?;
            out[(index - (t << STRIDE)) as usize] = decode_hash(v)?;
        }
        Ok(out)
    }

    fn apply_at(
        &self,
        txn: &mut RwTxn,
        height: u64,
        at: Option<u64>,
        created: &[OutputRecord],
        spent: &[Commitment],
    ) -> Result<ChangeProofs, Error> {
        let inserts = created
            .iter()
            .map(|o| self.insert_at(txn, o, height, at))
            .collect::<Result<_, _>>()?;
        let removals = spent
            .iter()
            .map(|c| self.remove_at(txn, c, at))
            .collect::<Result<_, _>>()?;
        Ok(ChangeProofs { inserts, removals })
    }

    /// Undoes a block's changes ([`ChangeProofs::undo`]), which must be the
    /// last ones applied: puts every touched slot back as it was, with its
    /// key and path, and the slot counter.
    pub fn revert(&self, txn: &mut RwTxn, undo: &Undo) -> Result<(), Error> {
        for &(index, _) in &undo.slots {
            if let Some(bytes) = self.leaves.get(txn, &index.to_be_bytes())? {
                let current = stored_key(bytes)?;
                self.keys.delete(txn, &sort_key(&current))?;
            }
        }
        for (index, old) in &undo.slots {
            self.put_leaf(txn, *index, old.as_ref(), None)?;
            if let Some(leaf) = old {
                self.keys
                    .put(txn, &sort_key(&leaf.key), &index.to_be_bytes())?;
            }
        }
        let slots: Vec<u64> = undo.slots.iter().map(|&(i, _)| i).collect();
        self.flush(txn, &slots, None)?;
        self.meta.put(txn, SIZE_KEY, &undo.size.to_be_bytes())?;
        Ok(())
    }

    /// [`StateTree::revert`] for the block at `height` (the last applied
    /// with [`StateTree::apply_block`]), dropping its history.
    pub fn revert_block(&self, txn: &mut RwTxn, height: u64, undo: &Undo) -> Result<(), Error> {
        self.revert(txn, undo)?;
        let lo = change_key(height, 0, 0);
        let hi = change_key(height, DEPTH, u64::MAX);
        self.drop_changes(txn, (Bound::Included(&lo[..]), Bound::Included(&hi[..])))
    }

    /// The lowest height the tree can be served at, once set: history
    /// below it was pruned, or never kept.
    pub fn history_from(&self, txn: &RoTxn) -> Result<Option<u64>, Error> {
        self.meta.get(txn, HISTORY_KEY)?.map(decode_u64).transpose()
    }

    /// Starts the history at `height` (the current tree's) unless it has
    /// started already.
    pub fn init_history(&self, txn: &mut RwTxn, height: u64) -> Result<(), Error> {
        if self.history_from(txn)?.is_none() {
            self.meta.put(txn, HISTORY_KEY, &height.to_be_bytes())?;
        }
        Ok(())
    }

    /// Forgets what blocks at heights up to `through` changed: the tree can
    /// then be served at heights from `through` up.
    pub fn prune_history(&self, txn: &mut RwTxn, through: u64) -> Result<(), Error> {
        let hi = change_key(through, DEPTH, u64::MAX);
        self.drop_changes(txn, (Bound::Unbounded, Bound::Included(&hi[..])))?;
        if self.history_from(txn)?.is_none_or(|h| h < through) {
            self.meta.put(txn, HISTORY_KEY, &through.to_be_bytes())?;
        }
        Ok(())
    }

    /// Deletes the history entries whose `state_changes` keys are in
    /// `range`.
    fn drop_changes(
        &self,
        txn: &mut RwTxn,
        range: (Bound<&[u8]>, Bound<&[u8]>),
    ) -> Result<(), Error> {
        let keys = self
            .changes
            .range(txn, &range)?
            .map(|e| e.map(|(k, _)| k.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        for k in keys {
            // Height, then level and index; the history key puts the
            // height last.
            let (height, node) = k.split_at_checked(8).ok_or(Error::Corrupt)?;
            self.history.delete(txn, &[node, height].concat())?;
            self.changes.delete(txn, &k)?;
        }
        Ok(())
    }

    /// Records `old` as node (`level`, `index`)'s value before the block at
    /// `height`, unless that block changed it already.
    fn journal(
        &self,
        txn: &mut RwTxn,
        level: usize,
        index: u64,
        height: u64,
        old: &[u8],
    ) -> Result<(), Error> {
        let key = hist_key(level, index, height);
        if self.history.get(txn, &key)?.is_none() {
            self.history.put(txn, &key, old)?;
            self.changes
                .put(txn, &change_key(height, level, index), &[])?;
        }
        Ok(())
    }

    /// Node (`level`, `index`)'s recorded value as of height `height`, if a
    /// later block changed it: its value before the first such change.
    fn past(
        &self,
        txn: &RoTxn,
        level: usize,
        index: u64,
        height: u64,
    ) -> Result<Option<Vec<u8>>, Error> {
        let lo = hist_key(level, index, height.saturating_add(1));
        let hi = hist_key(level, index, u64::MAX);
        let range = (Bound::Included(&lo[..]), Bound::Included(&hi[..]));
        Ok(self
            .history
            .range(txn, &range)?
            .next()
            .transpose()?
            .map(|(_, v)| v.to_vec()))
    }

    /// Slot `index`'s stored leaf as of height `height` (within the
    /// history).
    fn stored_at(&self, txn: &RoTxn, index: u64, height: u64) -> Result<Option<Vec<u8>>, Error> {
        Ok(match self.past(txn, 0, index, height)? {
            Some(v) if v.is_empty() => None,
            Some(v) => Some(v),
            None => self
                .leaves
                .get(txn, &index.to_be_bytes())?
                .map(<[u8]>::to_vec),
        })
    }

    /// Slot `index`'s leaf as of height `height` (within the history).
    fn leaf_at(&self, txn: &RoTxn, index: u64, height: u64) -> Result<Option<Leaf>, Error> {
        let Some(v) = self.stored_at(txn, index, height)? else {
            return Ok(None);
        };
        let (key, nonce, next_index, created) = decode_stored(&v)?;
        let next = self
            .stored_at(txn, next_index, height)?
            .ok_or(Error::Corrupt)?;
        Ok(Some(Leaf {
            key,
            nonce,
            next_key: stored_key(&next)?,
            next_index,
            height: created,
        }))
    }

    /// Node (`level`, `index`)'s hash as of height `height` (within the
    /// history).
    fn node_at(&self, txn: &RoTxn, level: usize, index: u64, height: u64) -> Result<Hash, Error> {
        if level == 0 {
            return Ok(self
                .leaf_at(txn, index, height)?
                .map_or(EMPTY_LEAF, |l| l.hash()));
        }
        if !is_stored(level) {
            return Ok(merge(
                &self.node_at(txn, level - 1, 2 * index, height)?,
                &self.node_at(txn, level - 1, 2 * index + 1, height)?,
            ));
        }
        match self.past(txn, level, index, height)? {
            Some(v) if v.is_empty() => Ok(empty_hashes()[level]),
            Some(v) => decode_hash(&v),
            None => self.node(txn, level, index),
        }
    }

    /// The state-sync reply for node (`level`, `index`) of the tree as it
    /// was after the block at `height`, which must be within the history
    /// ([`StateTree::history_from`]). At [`SYNC_LEAF_LEVEL`], a bitmap
    /// (`u8`) of the node's occupied slots, then their leaves; above it, a
    /// bitmap (`u32`, little-endian) of its non-empty descendants
    /// [`SYNC_HASH_LEVELS`] down (or at the leaf level, if nearer), then
    /// their hashes. `None` below the leaf level or for an index out of
    /// range.
    pub fn sync_reply(
        &self,
        txn: &RoTxn,
        height: u64,
        level: usize,
        index: u64,
    ) -> Result<Option<Vec<u8>>, Error> {
        if !(SYNC_LEAF_LEVEL..=DEPTH).contains(&level) || index >> (DEPTH - level) != 0 {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(MAX_SYNC_REPLY);
        if level == SYNC_LEAF_LEVEL {
            out.push(0u8);
            for k in 0..1 << level {
                if let Some(leaf) = self.leaf_at(txn, (index << level) + k, height)? {
                    out[0] |= 1 << k;
                    out.extend(encode_leaf(&leaf));
                }
            }
        } else {
            let t = reply_level(level);
            let span = level - t;
            let mut bitmap = 0u32;
            let mut hashes = Vec::new();
            if self.node_at(txn, level, index, height)? != empty_hashes()[level] {
                for k in 0..1u64 << span {
                    let h = self.node_at(txn, t, (index << span) + k, height)?;
                    if h != empty_hashes()[t] {
                        bitmap |= 1 << k;
                        hashes.extend(hash::to_bytes(&h));
                    }
                }
            }
            out.extend(bitmap.to_le_bytes());
            out.extend(hashes);
        }
        Ok(Some(out))
    }

    /// Starts downloading a tree with root `root`, tagged `tag` (which
    /// names what it's for), or resumes the download if it has that tag
    /// already. Returns whether it resumed.
    pub fn sync_begin(&self, txn: &mut RwTxn, tag: &[u8], root: &Hash) -> Result<bool, Error> {
        if self.meta.get(txn, SYNC_KEY)? == Some(tag) {
            return Ok(true);
        }
        self.sync_todo.clear(txn)?;
        self.sync_leaves.clear(txn)?;
        self.sync_todo
            .put(txn, &node_key(DEPTH, 0), &hash::to_bytes(root))?;
        self.meta.put(txn, SYNC_KEY, tag)?;
        Ok(false)
    }

    /// The tag of the download in progress, if any.
    pub fn sync_tag(&self, txn: &RoTxn) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.meta.get(txn, SYNC_KEY)?.map(<[u8]>::to_vec))
    }

    /// Drops the download in progress.
    pub fn sync_clear(&self, txn: &mut RwTxn) -> Result<(), Error> {
        self.sync_todo.clear(txn)?;
        self.sync_leaves.clear(txn)?;
        self.meta.delete(txn, SYNC_KEY)?;
        Ok(())
    }

    /// Up to `n` nodes still to fetch (as level and index), leaving out
    /// `skip` (the ones requested already): leaf groups first.
    pub fn sync_wanted(
        &self,
        txn: &RoTxn,
        n: usize,
        skip: &HashSet<(u8, u64)>,
    ) -> Result<Vec<(u8, u64)>, Error> {
        let mut out = Vec::new();
        for entry in self.sync_todo.iter(txn)? {
            if out.len() == n {
                break;
            }
            let (k, _) = entry?;
            let id = (k[0], decode_u64(&k[1..])?);
            if !skip.contains(&id) {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// Whether a download is in progress and has nothing left to fetch.
    pub fn sync_done(&self, txn: &RoTxn) -> Result<bool, Error> {
        Ok(self.meta.get(txn, SYNC_KEY)?.is_some() && self.sync_todo.is_empty(txn)?)
    }

    /// Takes `data`, a peer's [`StateTree::sync_reply`] for node (`level`,
    /// `index`), if that node is still wanted: checks that it hashes to the
    /// node, then stores its leaves, or queues its non-empty descendants.
    pub fn sync_put(
        &self,
        txn: &mut RwTxn,
        level: usize,
        index: u64,
        data: &[u8],
    ) -> Result<SyncPut, Error> {
        if level > DEPTH {
            return Ok(SyncPut::Unwanted);
        }
        let key = node_key(level, index);
        let Some(expected) = self.sync_todo.get(txn, &key)? else {
            return Ok(SyncPut::Unwanted);
        };
        let expected = decode_hash(expected)?;
        if level == SYNC_LEAF_LEVEL {
            let Some((&bitmap, rest)) = data.split_first() else {
                return Ok(SyncPut::Bad);
            };
            if rest.len() != bitmap.count_ones() as usize * LEAF_BYTES {
                return Ok(SyncPut::Bad);
            }
            let mut hashes = [EMPTY_LEAF; 1 << SYNC_LEAF_LEVEL];
            let mut entries = rest.chunks_exact(LEAF_BYTES);
            let mut leaves = Vec::new();
            for (k, h) in hashes.iter_mut().enumerate() {
                if bitmap >> k & 1 == 1 {
                    let bytes = entries.next().expect("counted");
                    let Ok(leaf) = decode_leaf(bytes) else {
                        return Ok(SyncPut::Bad);
                    };
                    *h = leaf.hash();
                    leaves.push(((index << level) + k as u64, bytes));
                }
            }
            if fold(&hashes, 0) != expected {
                return Ok(SyncPut::Bad);
            }
            for (slot, bytes) in leaves {
                self.sync_leaves.put(txn, &slot.to_be_bytes(), bytes)?;
            }
        } else {
            let t = reply_level(level);
            let span = level - t;
            let Some((bitmap, rest)) = data.split_first_chunk::<4>() else {
                return Ok(SyncPut::Bad);
            };
            let bitmap = u32::from_le_bytes(*bitmap);
            let n = 1usize << span;
            if (n < 32 && bitmap >> n != 0)
                || rest.len() != bitmap.count_ones() as usize * HASH_BYTES
            {
                return Ok(SyncPut::Bad);
            }
            let mut hashes = vec![empty_hashes()[t]; n];
            let mut entries = rest.chunks_exact(HASH_BYTES);
            for (k, h) in hashes.iter_mut().enumerate() {
                if bitmap >> k & 1 == 1 {
                    let Ok(hash) = decode_hash(entries.next().expect("counted")) else {
                        return Ok(SyncPut::Bad);
                    };
                    *h = hash;
                }
            }
            if fold(&hashes, t) != expected {
                return Ok(SyncPut::Bad);
            }
            for (k, h) in hashes.iter().enumerate() {
                if *h != empty_hashes()[t] {
                    self.sync_todo.put(
                        txn,
                        &node_key(t, (index << span) + k as u64),
                        &hash::to_bytes(h),
                    )?;
                }
            }
        }
        self.sync_todo.delete(txn, &key)?;
        Ok(SyncPut::Stored)
    }

    /// Makes a finished download the tree (which must be new: just the
    /// sentinel), with slot counter `size`, as of height `height` (where
    /// its history starts). Builds the key index and hashes the tree bottom
    /// up as it goes, a batch of leaves at a time, and drops the download.
    /// Returns the new state, for the caller to check; on error `txn` is
    /// left partly updated, and the caller should abort it.
    pub fn sync_finish(&self, txn: &mut RwTxn, size: u64, height: u64) -> Result<State, Error> {
        if self.size(txn)? != 1 {
            return Err(Error::NotEmpty);
        }
        if !self.sync_done(txn)? {
            return Err(Error::Incomplete);
        }
        if size > CAPACITY {
            return Err(Error::Corrupt);
        }
        self.leaves.clear(txn)?;
        self.keys.clear(txn)?;
        self.nodes.clear(txn)?;
        let mut pending = [None; DEPTH + 1];
        let mut from = 0u64;
        loop {
            let start = from.to_be_bytes();
            let range = (Bound::Included(&start[..]), Bound::Unbounded);
            let batch = self
                .sync_leaves
                .range(txn, &range)?
                .take(1024)
                .map(|e| e.map(|(k, v)| (k.to_vec(), v.to_vec())))
                .collect::<Result<Vec<_>, _>>()?;
            let Some((last, _)) = batch.last() else {
                break;
            };
            from = decode_u64(last)? + 1;
            for (k, v) in &batch {
                let index = decode_u64(k)?;
                if index >= size {
                    return Err(Error::Corrupt);
                }
                let leaf = decode_leaf(v)?;
                self.leaves.put(txn, k, &encode_stored(&leaf))?;
                self.keys.put(txn, &sort_key(&leaf.key), k)?;
                self.build(txn, &mut pending, 0, index, leaf.hash())?;
            }
        }
        for d in 0..DEPTH {
            if let Some((p, h)) = pending[d].take() {
                self.build(
                    txn,
                    &mut pending,
                    d + 1,
                    p >> 1,
                    merge(&h, &empty_hashes()[d]),
                )?;
            }
        }
        self.meta.put(txn, SIZE_KEY, &size.to_be_bytes())?;
        self.meta.put(txn, HISTORY_KEY, &height.to_be_bytes())?;
        self.sync_clear(txn)?;
        self.state(txn)
    }

    /// Adds node (`d`, `j`) with hash `h` to a bottom-up build fed nodes in
    /// index order: stores it, and passes its parent up once its sibling
    /// is known (or known to be empty). `pending` holds, per level, a left
    /// child waiting for its sibling.
    fn build(
        &self,
        txn: &mut RwTxn,
        pending: &mut [Option<(u64, Hash)>; DEPTH + 1],
        d: usize,
        j: u64,
        h: Hash,
    ) -> Result<(), Error> {
        if is_stored(d) && h != empty_hashes()[d] {
            self.nodes.put(txn, &node_key(d, j), &hash::to_bytes(&h))?;
        }
        if d == DEPTH {
            return Ok(());
        }
        let empty = empty_hashes()[d];
        if let Some((p, ph)) = pending[d].take() {
            if p == j ^ 1 {
                return self.build(txn, pending, d + 1, j >> 1, merge(&ph, &h));
            }
            self.build(txn, pending, d + 1, p >> 1, merge(&ph, &empty))?;
        }
        if j & 1 == 0 {
            pending[d] = Some((j, h));
            Ok(())
        } else {
            self.build(txn, pending, d + 1, j >> 1, merge(&empty, &h))
        }
    }

    /// The slot of the largest key below sort key `k`. The sentinel has the
    /// smallest key, so one always exists for a valid key.
    fn predecessor(&self, txn: &RoTxn, k: &[u8]) -> Result<u64, Error> {
        let range = (Bound::Unbounded, Bound::Excluded(k));
        match self.keys.rev_range(txn, &range)?.next() {
            Some(entry) => decode_u64(entry?.1),
            None => Err(Error::Corrupt),
        }
    }

    /// Slot `index`'s leaf, which must be occupied.
    fn leaf(&self, txn: &RoTxn, index: u64) -> Result<Leaf, Error> {
        let bytes = self.leaves.get(txn, &index.to_be_bytes())?;
        self.with_next_key(txn, bytes.ok_or(Error::Corrupt)?)
    }

    /// The leaf stored as `bytes`, with its next key: the key in the slot
    /// it links to (the sentinel's, zero, at the end of the list).
    fn with_next_key(&self, txn: &RoTxn, bytes: &[u8]) -> Result<Leaf, Error> {
        let (key, nonce, next_index, height) = decode_stored(bytes)?;
        let next = self.leaves.get(txn, &next_index.to_be_bytes())?;
        Ok(Leaf {
            key,
            nonce,
            next_key: stored_key(next.ok_or(Error::Corrupt)?)?,
            next_index,
            height,
        })
    }

    pub(crate) fn witness(&self, txn: &RoTxn, index: u64) -> Result<LeafWitness, Error> {
        Ok(LeafWitness {
            index,
            leaf: self.leaf(txn, index)?,
            siblings: self.siblings(txn, index)?,
        })
    }

    /// Slot `index`'s Merkle path: a stored level at a time, from the
    /// [`GROUP`] nodes (or leaves) under the stored node above.
    fn siblings(&self, txn: &RoTxn, index: u64) -> Result<Box<Siblings>, Error> {
        let mut siblings = Box::new([EMPTY_LEAF; DEPTH]);
        for top in (STRIDE..=DEPTH).step_by(STRIDE) {
            let base = top - STRIDE;
            let t = index >> top;
            let mut level = if base == 0 {
                self.group_leaves(txn, t)?
                    .map(|l| l.map_or(EMPTY_LEAF, |l| l.hash()))
            } else {
                self.group_nodes(txn, base, t)?
            }
            .to_vec();
            let mut at = (index >> base) as usize % GROUP;
            for s in &mut siblings[base..top] {
                *s = level[at ^ 1];
                level = level.chunks_exact(2).map(|p| merge(&p[0], &p[1])).collect();
                at >>= 1;
            }
        }
        Ok(siblings)
    }

    /// Node (`level`, `index`)'s hash: stored, or hashed from the stored
    /// level below.
    fn node(&self, txn: &RoTxn, level: usize, index: u64) -> Result<Hash, Error> {
        if level == 0 {
            return Ok(match self.leaves.get(txn, &index.to_be_bytes())? {
                Some(bytes) => self.with_next_key(txn, bytes)?.hash(),
                None => EMPTY_LEAF,
            });
        }
        if !is_stored(level) {
            return Ok(merge(
                &self.node(txn, level - 1, 2 * index)?,
                &self.node(txn, level - 1, 2 * index + 1)?,
            ));
        }
        match self.nodes.get(txn, &node_key(level, index))? {
            Some(bytes) => decode_hash(bytes),
            None => Ok(empty_hashes()[level]),
        }
    }

    /// Writes slot `index` and rehashes its path, recording history for
    /// the block at `at`, if given.
    fn set_leaf(
        &self,
        txn: &mut RwTxn,
        index: u64,
        leaf: &Leaf,
        at: Option<u64>,
    ) -> Result<(), Error> {
        self.put_leaf(txn, index, Some(leaf), at)?;
        self.flush(txn, &[index], at)
    }

    fn journal_leaf(&self, txn: &mut RwTxn, index: u64, at: Option<u64>) -> Result<(), Error> {
        if let Some(height) = at {
            let old = self
                .leaves
                .get(txn, &index.to_be_bytes())?
                .map(<[u8]>::to_vec)
                .unwrap_or_default();
            self.journal(txn, 0, index, height, &old)?;
        }
        Ok(())
    }
}

/// Bytes in an encoded leaf: key, nonce, next key, next index, height.
const LEAF_BYTES: usize = 2 * HASH_BYTES + NONCE_LEN + 8 + 4;

fn encode_leaf(leaf: &Leaf) -> [u8; LEAF_BYTES] {
    let mut out = [0u8; LEAF_BYTES];
    out[..HASH_BYTES].copy_from_slice(&hash::to_bytes(&leaf.key));
    out[HASH_BYTES..HASH_BYTES + NONCE_LEN].copy_from_slice(&leaf.nonce);
    let at = HASH_BYTES + NONCE_LEN;
    out[at..at + HASH_BYTES].copy_from_slice(&hash::to_bytes(&leaf.next_key));
    out[at + HASH_BYTES..at + HASH_BYTES + 8].copy_from_slice(&leaf.next_index.to_be_bytes());
    out[at + HASH_BYTES + 8..].copy_from_slice(&leaf.height.to_be_bytes());
    out
}

fn decode_leaf(bytes: &[u8]) -> Result<Leaf, Error> {
    if bytes.len() != LEAF_BYTES {
        return Err(Error::Corrupt);
    }
    let at = HASH_BYTES + NONCE_LEN;
    Ok(Leaf {
        key: decode_hash(&bytes[..HASH_BYTES])?,
        nonce: bytes[HASH_BYTES..at].try_into().expect("nonce bytes"),
        next_key: decode_hash(&bytes[at..at + HASH_BYTES])?,
        next_index: decode_u64(&bytes[at + HASH_BYTES..at + HASH_BYTES + 8])?,
        height: decode_u32(&bytes[at + HASH_BYTES + 8..])?,
    })
}

/// Bytes in a stored leaf: key, nonce, next index, height (no next key).
const STORED_LEAF_BYTES: usize = HASH_BYTES + NONCE_LEN + 8 + 4;

fn encode_stored(leaf: &Leaf) -> [u8; STORED_LEAF_BYTES] {
    let mut out = [0u8; STORED_LEAF_BYTES];
    let at = HASH_BYTES + NONCE_LEN;
    out[..HASH_BYTES].copy_from_slice(&hash::to_bytes(&leaf.key));
    out[HASH_BYTES..at].copy_from_slice(&leaf.nonce);
    out[at..at + 8].copy_from_slice(&leaf.next_index.to_be_bytes());
    out[at + 8..].copy_from_slice(&leaf.height.to_be_bytes());
    out
}

/// A stored leaf's key, nonce, next index and height.
fn decode_stored(bytes: &[u8]) -> Result<(Hash, Nonce, u64, u32), Error> {
    if bytes.len() != STORED_LEAF_BYTES {
        return Err(Error::Corrupt);
    }
    let at = HASH_BYTES + NONCE_LEN;
    Ok((
        stored_key(bytes)?,
        bytes[HASH_BYTES..at].try_into().expect("nonce bytes"),
        decode_u64(&bytes[at..at + 8])?,
        decode_u32(&bytes[at + 8..])?,
    ))
}

/// A stored leaf's key.
fn stored_key(bytes: &[u8]) -> Result<Hash, Error> {
    decode_hash(bytes.get(..HASH_BYTES).ok_or(Error::Corrupt)?)
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

fn decode_u32(bytes: &[u8]) -> Result<u32, Error> {
    Ok(u32::from_be_bytes(
        bytes.try_into().map_err(|_| Error::Corrupt)?,
    ))
}

/// A block height as a leaf's creation height. Chain heights stay far
/// below `2^26` ([`crate::header::MAX_HEIGHT_BITS`]).
fn leaf_height(height: u64) -> u32 {
    u32::try_from(height).expect("chain heights fit in 32 bits")
}

/// Hashes a level of `2^k` nodes at level `d` up to their root.
fn fold(nodes: &[Hash], d: usize) -> Hash {
    let mut level = nodes.to_vec();
    let mut d = d;
    while level.len() > 1 {
        level = level.chunks_exact(2).map(|p| merge(&p[0], &p[1])).collect();
        d += 1;
    }
    debug_assert!(d <= DEPTH);
    level[0]
}

/// `state_history` key: level, index, height.
fn hist_key(level: usize, index: u64, height: u64) -> [u8; 17] {
    let mut k = [0u8; 17];
    k[..9].copy_from_slice(&node_key(level, index));
    k[9..].copy_from_slice(&height.to_be_bytes());
    k
}

/// `state_changes` key: height, level, index.
fn change_key(height: u64, level: usize, index: u64) -> [u8; 17] {
    let mut k = [0u8; 17];
    k[..8].copy_from_slice(&height.to_be_bytes());
    k[8..].copy_from_slice(&node_key(level, index));
    k
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

    /// A block applied in one batch gives the same tree, and the same undo,
    /// as applied change by change; and reverting it gives back the tree.
    #[test]
    fn batched_blocks_match_change_by_change() {
        let (a, b) = (TempEnv::new(), TempEnv::new());
        let (ta, tb) = (
            StateTree::open(a.env()).unwrap(),
            StateTree::open(b.env()).unwrap(),
        );
        let (mut xa, mut xb) = (a.env().write_txn().unwrap(), b.env().write_txn().unwrap());
        // Outputs land in order and are spent out of order, so the chain
        // links run back and forth across the tree.
        let blocks: [(Vec<u32>, Vec<u32>); 4] = [
            ((0..300).map(|n| n * 7919 % 1000).collect(), vec![]),
            (
                (1000..1200).collect(),
                (0..300).step_by(3).map(|n| n * 7919 % 1000).collect(),
            ),
            (vec![5000], vec![1000, 1199, 1100]),
            ((2000..2600).collect(), (1001..1099).collect()),
        ];
        for (h, (created, spent)) in (1..).zip(&blocks) {
            let created: Vec<_> = created.iter().map(|&n| record(n)).collect();
            let spent: Vec<_> = spent.iter().map(|&n| commitment(n)).collect();
            let before = ta.state(&xa).unwrap();
            let undo = ta.apply_block(&mut xa, h, &created, &spent).unwrap();
            let proofs = tb.apply_changes(&mut xb, h, &created, &spent).unwrap();
            assert_eq!(ta.state(&xa).unwrap(), tb.state(&xb).unwrap(), "block {h}");
            assert_eq!(undo, proofs.undo(before.size), "block {h}");
            if h == 4 {
                ta.revert_block(&mut xa, h, &undo).unwrap();
                assert_eq!(ta.state(&xa).unwrap(), before);
                ta.apply_block(&mut xa, h, &created, &spent).unwrap();
                assert_eq!(ta.state(&xa).unwrap(), tb.state(&xb).unwrap());
            }
        }
    }

    /// A full block's worth of outputs (31,000) into a tree of 70,000,
    /// with history, as a node applies a 2 MB block. Prints the time.
    #[test]
    #[ignore = "benchmark"]
    fn bench_full_block_apply() {
        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        let old: Vec<_> = (0..70_000).map(record).collect();
        tree.apply_block(&mut txn, 1, &old, &[]).unwrap();
        txn.commit().unwrap();
        let new: Vec<_> = (70_000..101_000).map(record).collect();
        let t = std::time::Instant::now();
        let mut txn = env.env().write_txn().unwrap();
        tree.apply_block(&mut txn, 2, &new, &[]).unwrap();
        let applied = t.elapsed();
        txn.commit().unwrap();
        eprintln!("apply {applied:.2?}, with commit {:.2?}", t.elapsed());
        // Change by change, as the prover applies them for its witnesses.
        let more: Vec<_> = (101_000..132_000).map(record).collect();
        let t = std::time::Instant::now();
        let mut txn = env.env().write_txn().unwrap();
        tree.apply_changes(&mut txn, 3, &more, &[]).unwrap();
        eprintln!("change by change {:.2?}", t.elapsed());
        txn.abort();
    }

    /// Bytes on disk per live output, dense (nothing spent) and after
    /// churn (most outputs spent, the live ones scattered). Prints them.
    #[test]
    #[ignore = "benchmark"]
    fn bench_storage_per_output() {
        fn bytes(tree: &StateTree, txn: &RoTxn) -> [(&'static str, usize); 5] {
            let size = |db: &Database<Bytes, Bytes>| {
                let s = db.stat(txn).unwrap();
                (s.branch_pages + s.leaf_pages + s.overflow_pages) * s.page_size as usize
            };
            [
                ("leaves", size(&tree.leaves)),
                ("nodes", size(&tree.nodes)),
                ("keys", size(&tree.keys)),
                ("history", size(&tree.history)),
                ("changes", size(&tree.changes)),
            ]
        }
        fn report(label: &str, live: usize, b: &[(&str, usize)]) {
            let parts: Vec<String> = b
                .iter()
                .map(|(n, v)| format!("{n} {:.1}", *v as f64 / live as f64))
                .collect();
            let tree: usize = b[..3].iter().map(|x| x.1).sum();
            eprintln!(
                "{label}: live {live}, tree {:.1} B/output ({}), {}",
                tree as f64 / live as f64,
                tree,
                parts.join(", ")
            );
        }
        const BLOCK: u32 = 27_500;
        const BLOCKS: u32 = 100;
        let mut env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let mut live: Vec<u32> = Vec::new();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rand = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % n as u64) as usize
        };
        for h in 0..BLOCKS {
            let created: Vec<_> = (h * BLOCK..(h + 1) * BLOCK).map(record).collect();
            // Spend about as much as is created, once there's a backlog, so
            // the live set stays near 3 blocks' worth and spreads out.
            let spend = if live.len() > 3 * BLOCK as usize {
                BLOCK as usize
            } else {
                0
            };
            let spent: Vec<_> = (0..spend)
                .map(|_| commitment(live.swap_remove(rand(live.len()))))
                .collect();
            let mut txn = env.env().write_txn().unwrap();
            tree.apply_block(&mut txn, h as u64 + 1, &created, &spent)
                .unwrap();
            if h == 0 {
                let b = bytes(&tree, &txn);
                report("one full block", BLOCK as usize, &b);
            }
            txn.commit().unwrap();
            live.extend(h * BLOCK..(h + 1) * BLOCK);
            if h + 1 == 4 {
                let txn = env.env().read_txn().unwrap();
                report(
                    "dense, 4 blocks, history kept",
                    live.len(),
                    &bytes(&tree, &txn),
                );
            }
        }
        let txn = env.env().read_txn().unwrap();
        let b = bytes(&tree, &txn);
        report("churned, history of all blocks", live.len(), &b);
        eprintln!(
            "  history per block {:.1} MB, slots {}",
            (b[3].1 + b[4].1) as f64 / BLOCKS as f64 / 1e6,
            BLOCK * BLOCKS
        );
        drop(txn);
        let mut txn = env.env().write_txn().unwrap();
        tree.prune_history(&mut txn, BLOCKS as u64).unwrap();
        txn.commit().unwrap();
        let txn = env.env().read_txn().unwrap();
        report("churned, pruned", live.len(), &bytes(&tree, &txn));
        let state = tree.state(&txn).unwrap();
        drop(txn);
        env.env.take().unwrap().prepare_for_closing().wait();
        let (before, after) = crate::db::compact(&env.path).unwrap();
        eprintln!(
            "churned, pruned, compacted: file {:.1} -> {:.1} B/output",
            before as f64 / live.len() as f64,
            after as f64 / live.len() as f64
        );
        env.reopen();
        let tree = StateTree::open(env.env()).unwrap();
        assert_eq!(tree.state(&env.env().read_txn().unwrap()).unwrap(), state);
    }

    fn nonce(n: u32) -> Nonce {
        let mut nonce = [0u8; NONCE_LEN];
        nonce[..4].copy_from_slice(&n.to_le_bytes());
        nonce[15] = 0xaa;
        nonce
    }

    fn record(n: u32) -> OutputRecord {
        OutputRecord {
            commitment: output::commit(&[7; 32], 1, &nonce(n)).unwrap(),
            nonce: nonce(n),
        }
    }

    fn commitment(n: u32) -> Commitment {
        record(n).commitment
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
        /// A tree with outputs `0..n` inserted (in one committed
        /// transaction), checking every proof.
        fn new(n: u32) -> Self {
            let tmp = TempEnv::new();
            let tree = StateTree::open(tmp.env()).unwrap();
            let mut txn = tmp.env().write_txn().unwrap();
            for i in 0..n {
                let before = tree.state(&txn).unwrap();
                let proof = tree.insert(&mut txn, &record(i), 1).unwrap();
                assert_eq!(
                    verify_insert(&before, &key(i), &nonce(i), 1, &proof),
                    Ok(tree.state(&txn).unwrap())
                );
            }
            txn.commit().unwrap();
            Self { tmp, tree }
        }

        /// Every slot's leaf, `None` where empty.
        fn slots(&self, txn: &RoTxn) -> Vec<Option<Leaf>> {
            let size = self.tree.state(txn).unwrap().size;
            (0..size)
                .map(|i| {
                    self.tree
                        .leaves
                        .get(txn, &i.to_be_bytes())
                        .unwrap()
                        .map(|_| self.tree.leaf(txn, i).unwrap())
                })
                .collect()
        }
    }

    /// Root computed from scratch over the slots, for cross-checking.
    fn naive_root(slots: &[Option<Leaf>]) -> Hash {
        fn node(slots: &[Option<Leaf>], d: usize, i: u64) -> Hash {
            if (i << d) >= slots.len() as u64 {
                return empty_hashes()[d];
            }
            if d == 0 {
                return slots[i as usize].map_or(EMPTY_LEAF, |l| l.hash());
            }
            merge(&node(slots, d - 1, 2 * i), &node(slots, d - 1, 2 * i + 1))
        }
        node(slots, DEPTH, 0)
    }

    /// Walks the list from the sentinel: keys must ascend and every
    /// occupied slot must be on it.
    fn check_list(slots: &[Option<Leaf>]) {
        let (mut i, mut count) = (0u64, 1);
        loop {
            let leaf = slots[i as usize].unwrap();
            if leaf.next_index == END {
                break;
            }
            let next = slots[leaf.next_index as usize].unwrap();
            assert_eq!(leaf.next_key, next.key);
            assert!(less(&leaf.key, &next.key));
            i = leaf.next_index;
            count += 1;
        }
        assert_eq!(count, slots.iter().flatten().count());
    }

    #[test]
    fn empty_tree_has_only_sentinel() {
        let f = Fixture::new(0);
        let txn = f.tmp.env().read_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 1);
        assert_eq!(state.root, naive_root(&[Some(Leaf::SENTINEL)]));
        assert_eq!(state, empty_state());
        assert!(f.tree.outputs(&txn).unwrap().is_empty());
    }

    #[test]
    fn inserts_verify_and_match_naive_root() {
        let f = Fixture::new(40);
        let txn = f.tmp.env().read_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 41);
        let slots = f.slots(&txn);
        assert_eq!(state.root, naive_root(&slots));
        check_list(&slots);
    }

    #[test]
    fn removals_verify_and_empty_their_slots() {
        let f = Fixture::new(10);
        let mut txn = f.tmp.env().write_txn().unwrap();
        for i in [3, 0, 9, 4] {
            let before = f.tree.state(&txn).unwrap();
            let (slot, _) = f.tree.get(&txn, &commitment(i)).unwrap().unwrap();
            let proof = f.tree.remove(&mut txn, &commitment(i)).unwrap();
            assert_eq!(proof.position(), slot);
            assert_eq!(
                verify_remove(&before, &key(i), &proof),
                Ok(f.tree.state(&txn).unwrap())
            );
            assert!(f.tree.get(&txn, &commitment(i)).unwrap().is_none());
        }
        let state = f.tree.state(&txn).unwrap();
        // The slot counter never goes back.
        assert_eq!(state.size, 11);
        let slots = f.slots(&txn);
        assert_eq!(state.root, naive_root(&slots));
        check_list(&slots);
        assert_eq!(slots.iter().flatten().count(), 7);
        // Only unspent outputs are listed, with their nonces.
        let outputs = f.tree.outputs(&txn).unwrap();
        assert_eq!(outputs.len(), 6);
        for (slot, o) in outputs {
            assert_eq!(f.tree.get(&txn, &o.commitment).unwrap().unwrap().0, slot);
            assert!((0..10).any(|n| record(n) == o));
        }
    }

    #[test]
    fn removing_everything_restores_the_empty_root() {
        let f = Fixture::new(6);
        let empty = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        for i in 0..6 {
            f.tree.remove(&mut txn, &commitment(i)).unwrap();
        }
        let state = f.tree.state(&txn).unwrap();
        let empty_state = empty
            .tree
            .state(&empty.tmp.env().read_txn().unwrap())
            .unwrap();
        assert_eq!(state.root, empty_state.root);
        assert_eq!(state.size, 7);
    }

    #[test]
    fn a_spent_commitment_can_return_in_a_new_slot() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let t = &f.tree;
        assert_eq!(
            state_err(t.insert(&mut txn, &record(2), 1)),
            StateError::Duplicate
        );
        let first = t.remove(&mut txn, &commitment(2)).unwrap().position();
        assert_eq!(
            state_err(t.remove(&mut txn, &commitment(2))),
            StateError::NotFound
        );
        let before = t.state(&txn).unwrap();
        let proof = t.insert(&mut txn, &record(2), 1).unwrap();
        assert_eq!(
            verify_insert(&before, &key(2), &nonce(2), 1, &proof),
            Ok(t.state(&txn).unwrap())
        );
        let second = t.get(&txn, &commitment(2)).unwrap().unwrap().0;
        assert_ne!(first, second);
        assert_eq!(second, before.size);
        assert_eq!(
            state_err(t.remove(&mut txn, &commitment(99))),
            StateError::NotFound
        );
    }

    #[test]
    fn insert_then_remove_in_same_block() {
        let f = Fixture::new(3);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let s0 = f.tree.state(&txn).unwrap();
        let ins = f.tree.insert(&mut txn, &record(100), 1).unwrap();
        let s1 = verify_insert(&s0, &key(100), &nonce(100), 1, &ins).unwrap();
        let rm = f.tree.remove(&mut txn, &commitment(100)).unwrap();
        assert_eq!(
            verify_remove(&s1, &key(100), &rm),
            Ok(f.tree.state(&txn).unwrap())
        );
    }

    #[test]
    fn rejects_invalid_keys() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        for bytes in [[0; 32], [0xff; 32]] {
            let r = OutputRecord {
                commitment: Commitment(bytes),
                nonce: [0; NONCE_LEN],
            };
            assert_eq!(
                state_err(f.tree.insert(&mut txn, &r, 1)),
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
            f.tree.insert(&mut txn, &record(50), 1).unwrap();
            f.tree.remove(&mut txn, &commitment(1)).unwrap();
            // Dropped without commit.
        }
        let txn = f.tmp.env().read_txn().unwrap();
        assert_eq!(f.tree.state(&txn).unwrap(), before);
        assert!(f.tree.get(&txn, &commitment(50)).unwrap().is_none());
        assert!(f.tree.get(&txn, &commitment(1)).unwrap().is_some());
    }

    #[test]
    fn state_persists_across_reopen() {
        let mut f = Fixture::new(8);
        {
            let mut txn = f.tmp.env().write_txn().unwrap();
            f.tree.remove(&mut txn, &commitment(7)).unwrap();
            txn.commit().unwrap();
        }
        let before = f.tree.state(&f.tmp.env().read_txn().unwrap()).unwrap();
        f.tmp.reopen();
        let tree = StateTree::open(f.tmp.env()).unwrap();
        let mut txn = f.tmp.env().write_txn().unwrap();
        assert_eq!(tree.state(&txn).unwrap(), before);
        // And it keeps working after the restart, without reusing slot 8.
        let proof = tree.insert(&mut txn, &record(77), 1).unwrap();
        assert_eq!(
            verify_insert(&before, &key(77), &nonce(77), 1, &proof),
            Ok(tree.state(&txn).unwrap())
        );
        assert_eq!(tree.get(&txn, &commitment(77)).unwrap().unwrap().0, 9);
    }

    #[test]
    fn verify_insert_rejects_existing_key_and_tampering() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        let proof = f.tree.insert(&mut txn, &record(50), 1).unwrap();
        // An existing key is never strictly inside its own low leaf's range.
        for i in 0..5 {
            assert!(verify_insert(&state, &key(i), &nonce(i), 1, &proof).is_err());
        }
        let mut bad = proof.clone();
        bad.low.siblings[7][0] += BabyBear::ONE;
        assert_eq!(
            verify_insert(&state, &key(50), &nonce(50), 1, &bad),
            Err(ProofError::BadPath)
        );
        let mut bad = proof.clone();
        bad.new_siblings[0][0] += BabyBear::ONE;
        assert_eq!(
            verify_insert(&state, &key(50), &nonce(50), 1, &bad),
            Err(ProofError::BadPath)
        );
        let wrong_size = State {
            size: state.size + 1,
            ..state
        };
        assert_eq!(
            verify_insert(&wrong_size, &key(50), &nonce(50), 1, &proof),
            Err(ProofError::BadPath)
        );
    }

    #[test]
    fn verify_remove_rejects_bad_proofs() {
        let f = Fixture::new(5);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let state = f.tree.state(&txn).unwrap();
        let proof = f.tree.remove(&mut txn, &commitment(1)).unwrap();
        assert_eq!(
            verify_remove(&state, &key(2), &proof),
            Err(ProofError::KeyMismatch)
        );
        // Claiming another position.
        let mut bad = proof.clone();
        bad.leaf.index += 1;
        assert!(verify_remove(&state, &key(1), &bad).is_err());
        // A changed nonce doesn't hash to the stored leaf.
        let mut bad = proof.clone();
        bad.leaf.leaf.nonce[0] ^= 1;
        assert_eq!(
            verify_remove(&state, &key(1), &bad),
            Err(ProofError::BadPath)
        );
        // A predecessor that doesn't link to it.
        let mut bad = proof.clone();
        bad.pred.leaf.next_index += 1;
        assert!(verify_remove(&state, &key(1), &bad).is_err());
        // Removed already: against the new state, the proof's paths fail.
        let after = f.tree.state(&txn).unwrap();
        assert_eq!(
            verify_remove(&after, &key(1), &proof),
            Err(ProofError::BadPath)
        );
    }

    #[test]
    fn leaf_encoding_roundtrips() {
        let leaf = Leaf {
            key: key(1),
            nonce: nonce(9),
            next_key: key(2),
            next_index: CAPACITY - 1,
            height: 77,
        };
        assert_eq!(decode_leaf(&encode_leaf(&leaf)).unwrap(), leaf);
        assert!(matches!(
            decode_leaf(&encode_leaf(&leaf)[1..]),
            Err(Error::Corrupt)
        ));
    }

    #[test]
    fn leaf_hash_covers_the_nonce() {
        let leaf = Leaf {
            key: key(1),
            nonce: nonce(9),
            next_key: key(2),
            next_index: 3,
            height: 5,
        };
        let mut other = leaf;
        other.nonce[15] ^= 1;
        assert_ne!(leaf.hash(), other.hash());
        let mut other = leaf;
        other.height += 1;
        assert_ne!(leaf.hash(), other.hash());
    }

    /// Outputs keep the height of the block that created them, through
    /// storage and a revert.
    #[test]
    fn outputs_keep_their_creation_height() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.apply_block(&mut txn, 4, &[record(1)], &[]).unwrap();
        let undo = f
            .tree
            .apply_block(&mut txn, 9, &[record(2)], &[commitment(1)])
            .unwrap();
        let height = |c| f.tree.get(&txn, &c).unwrap().map(|(_, l)| l.height);
        assert_eq!(height(commitment(2)), Some(9));
        assert_eq!(height(commitment(1)), None);
        f.tree.revert(&mut txn, &undo).unwrap();
        let height = |c| f.tree.get(&txn, &c).unwrap().map(|(_, l)| l.height);
        assert_eq!(height(commitment(1)), Some(4));
        let proof = f.tree.remove(&mut txn, &commitment(1)).unwrap();
        assert_eq!(proof.leaf.leaf.height, 4);
    }

    /// Applies a block's changes and checks the proofs reproduce the new state.
    fn apply(
        f: &Fixture,
        txn: &mut RwTxn,
        created: &[u32],
        spent: &[u32],
    ) -> Result<ChangeProofs, Error> {
        let created: Vec<_> = created.iter().map(|&n| record(n)).collect();
        let spent: Vec<_> = spent.iter().map(|&n| commitment(n)).collect();
        let before = f.tree.state(txn).unwrap();
        let proofs = f.tree.apply_changes(txn, 1, &created, &spent)?;
        assert_eq!(
            verify_changes(&before, 1, &created, &spent, &proofs),
            Ok(f.tree.state(txn).unwrap())
        );
        Ok(proofs)
    }

    /// Downloads the tree as of `height` from `src` into a new tree,
    /// replying in the order asked, and returns it with its state.
    fn download(src: &Fixture, height: u64, root: &Hash, size: u64) -> (TempEnv, StateTree, State) {
        let env = TempEnv::new();
        let copy = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        assert!(!copy.sync_begin(&mut txn, b"tag", root).unwrap());
        let from = src.tmp.env().read_txn().unwrap();
        loop {
            let wanted = copy.sync_wanted(&txn, 7, &HashSet::new()).unwrap();
            if wanted.is_empty() {
                break;
            }
            for (level, index) in wanted {
                let data = src
                    .tree
                    .sync_reply(&from, height, level as usize, index)
                    .unwrap()
                    .unwrap();
                assert!(data.len() <= MAX_SYNC_REPLY);
                assert_eq!(
                    copy.sync_put(&mut txn, level as usize, index, &data)
                        .unwrap(),
                    SyncPut::Stored
                );
            }
        }
        assert!(copy.sync_done(&txn).unwrap());
        let state = copy.sync_finish(&mut txn, size, height).unwrap();
        assert_eq!(copy.sync_tag(&txn).unwrap(), None);
        txn.commit().unwrap();
        (env, copy, state)
    }

    /// A tree from an earlier layout (no leaf heights) is refused, not
    /// misread.
    #[test]
    fn an_old_layout_is_refused() {
        let f = Fixture::new(3);
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.meta.put(&mut txn, LAYOUT_KEY, &[2]).unwrap();
        txn.commit().unwrap();
        assert!(matches!(
            StateTree::open(f.tmp.env()),
            Err(Error::OldLayout)
        ));
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.meta.delete(&mut txn, LAYOUT_KEY).unwrap();
        txn.commit().unwrap();
        assert!(matches!(
            StateTree::open(f.tmp.env()),
            Err(Error::OldLayout)
        ));
    }

    #[test]
    fn past_trees_are_served_and_download_into_copies() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.init_history(&mut txn, 0).unwrap();
        let blocks: [(Vec<u32>, Vec<u32>); 4] = [
            ((1..=40).collect(), vec![]),
            (vec![41, 42], vec![3, 17, 18]),
            (vec![3], vec![40, 41, 1]),
            ((50..60).collect(), vec![42, 20]),
        ];
        let mut states = vec![f.tree.state(&txn).unwrap()];
        let mut undos = vec![];
        for (h, (created, spent)) in (1..).zip(&blocks) {
            let created: Vec<_> = created.iter().map(|&n| record(n)).collect();
            let spent: Vec<_> = spent.iter().map(|&n| commitment(n)).collect();
            let undo = f.tree.apply_block(&mut txn, h, &created, &spent).unwrap();
            undos.push(undo);
            states.push(f.tree.state(&txn).unwrap());
        }
        txn.commit().unwrap();

        // The tree at every height downloads exactly, and the copy works
        // like the original from there.
        for (h, state) in states.iter().enumerate() {
            let (env, copy, got) = download(&f, h as u64, &state.root, state.size);
            assert_eq!(got, *state, "height {h}");
            let mut txn = env.env().write_txn().unwrap();
            assert_eq!(copy.history_from(&txn).unwrap(), Some(h as u64));
            if let Some((created, spent)) = blocks.get(h) {
                let created: Vec<_> = created.iter().map(|&n| record(n)).collect();
                let spent: Vec<_> = spent.iter().map(|&n| commitment(n)).collect();
                copy.apply_changes(&mut txn, h as u64 + 1, &created, &spent)
                    .unwrap();
                assert_eq!(copy.state(&txn).unwrap(), states[h + 1]);
            }
        }

        // Replies are checked against the node's hash.
        let env = TempEnv::new();
        let copy = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        copy.sync_begin(&mut txn, b"x", &states[2].root).unwrap();
        let from = f.tmp.env().read_txn().unwrap();
        let root = f.tree.sync_reply(&from, 2, DEPTH, 0).unwrap().unwrap();
        let wrong_height = f.tree.sync_reply(&from, 1, DEPTH, 0).unwrap().unwrap();
        assert_eq!(
            copy.sync_put(&mut txn, DEPTH, 0, &wrong_height).unwrap(),
            SyncPut::Bad
        );
        let mut flipped = root.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert_eq!(
            copy.sync_put(&mut txn, DEPTH, 0, &flipped).unwrap(),
            SyncPut::Bad
        );
        assert_eq!(
            copy.sync_put(&mut txn, DEPTH, 0, &root[..root.len() - 1])
                .unwrap(),
            SyncPut::Bad
        );
        assert_eq!(
            copy.sync_put(&mut txn, DEPTH, 0, &root).unwrap(),
            SyncPut::Stored
        );
        assert_eq!(
            copy.sync_put(&mut txn, DEPTH, 0, &root).unwrap(),
            SyncPut::Unwanted
        );
        // Leaves too: a group with a leaf missing doesn't hash right.
        let leaves = f
            .tree
            .sync_reply(&from, 1, SYNC_LEAF_LEVEL, 0)
            .unwrap()
            .unwrap();
        assert_eq!(leaves[0], 0xff);
        let mut missing = vec![leaves[0] & !1];
        missing.extend(&leaves[1 + LEAF_BYTES..]);
        copy.sync_todo
            .put(
                &mut txn,
                &node_key(SYNC_LEAF_LEVEL, 0),
                &hash::to_bytes(&f.tree.node_at(&from, SYNC_LEAF_LEVEL, 0, 1).unwrap()),
            )
            .unwrap();
        assert_eq!(
            copy.sync_put(&mut txn, SYNC_LEAF_LEVEL, 0, &missing)
                .unwrap(),
            SyncPut::Bad
        );
        assert_eq!(
            copy.sync_put(&mut txn, SYNC_LEAF_LEVEL, 0, &leaves)
                .unwrap(),
            SyncPut::Stored
        );
        // An unfinished download can't become the tree, and resumes.
        assert!(matches!(
            copy.sync_finish(&mut txn, 1, 2),
            Err(Error::Incomplete)
        ));
        assert!(copy.sync_begin(&mut txn, b"x", &states[2].root).unwrap());
        assert!(
            !copy
                .sync_wanted(&txn, 100, &HashSet::new())
                .unwrap()
                .is_empty()
        );
        assert!(
            f.tree
                .sync_reply(&from, 2, SYNC_LEAF_LEVEL - 1, 0)
                .unwrap()
                .is_none()
        );
        assert!(f.tree.sync_reply(&from, 2, DEPTH, 1).unwrap().is_none());
        drop(from);
        txn.abort();

        // Pruning keeps what later heights need; reverting drops a block's
        // history.
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.prune_history(&mut txn, 2).unwrap();
        assert_eq!(f.tree.history_from(&txn).unwrap(), Some(2));
        let heights: HashSet<u64> = f
            .tree
            .changes
            .iter(&txn)
            .unwrap()
            .map(|e| decode_u64(&e.unwrap().0[..8]).unwrap())
            .collect();
        assert_eq!(heights, HashSet::from([3, 4]));
        f.tree.revert_block(&mut txn, 4, &undos[3]).unwrap();
        assert_eq!(f.tree.state(&txn).unwrap(), states[3]);
        assert_eq!(
            f.tree.history.len(&txn).unwrap(),
            f.tree.changes.len(&txn).unwrap()
        );
        txn.commit().unwrap();
        for h in 2..=3 {
            let (_env, _copy, got) =
                download(&f, h, &states[h as usize].root, states[h as usize].size);
            assert_eq!(got, states[h as usize]);
        }
        // Prune never moves the start back.
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.prune_history(&mut txn, 1).unwrap();
        assert_eq!(f.tree.history_from(&txn).unwrap(), Some(2));
    }

    #[test]
    fn blocks_revert_exactly() {
        let f = Fixture::new(6);
        let mut txn = f.tmp.env().write_txn().unwrap();
        // Spend some, create some, spend what was just created, bring a
        // spent commitment back in a new slot.
        let blocks: [(&[u32], &[u32]); 4] = [
            (&[10, 11, 12], &[2, 4]),
            (&[13], &[10, 0, 13]),
            (&[2, 14, 15], &[5, 1]),
            (&[], &[14]),
        ];
        let mut history = vec![(f.tree.state(&txn).unwrap(), f.slots(&txn))];
        let mut undos = Vec::new();
        for (created, spent) in blocks {
            let size = f.tree.state(&txn).unwrap().size;
            let proofs = apply(&f, &mut txn, created, spent).unwrap();
            let undo = proofs.undo(size);
            assert_eq!(Undo::from_bytes(&undo.to_bytes()), Some(undo.clone()));
            undos.push(undo);
            history.push((f.tree.state(&txn).unwrap(), f.slots(&txn)));
        }
        while let Some(undo) = undos.pop() {
            f.tree.revert(&mut txn, &undo).unwrap();
            history.pop();
            let (state, slots) = history.last().unwrap();
            assert_eq!(f.tree.state(&txn).unwrap(), *state);
            assert_eq!(f.slots(&txn), *slots);
            assert_eq!(naive_root(&f.slots(&txn)), state.root);
            check_list(&f.slots(&txn));
            // The key index matches the slots.
            for (i, leaf) in slots.iter().enumerate().skip(1) {
                if let Some(l) = leaf {
                    let c = Commitment(hash::to_bytes(&l.key));
                    assert_eq!(
                        f.tree.get(&txn, &c).unwrap().map(|(j, _)| j),
                        Some(i as u64)
                    );
                }
            }
        }
        // Back at the start, the original outputs are spendable again.
        assert!(f.tree.get(&txn, &commitment(2)).unwrap().is_some());
        assert!(f.tree.get(&txn, &commitment(10)).unwrap().is_none());
        assert_eq!(Undo::from_bytes(&[1, 2, 3]), None);
    }

    #[test]
    fn blocks_apply_and_verify() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        apply(&f, &mut txn, &[1, 2, 3], &[]).unwrap();
        apply(&f, &mut txn, &[4], &[1, 3]).unwrap();
        // A spent commitment created again.
        apply(&f, &mut txn, &[1], &[]).unwrap();
        let state = f.tree.state(&txn).unwrap();
        assert_eq!(state.size, 6);
        let slots = f.slots(&txn);
        assert_eq!(state.root, naive_root(&slots));
        check_list(&slots);
        for (n, live) in [(1, true), (2, true), (3, false), (4, true)] {
            assert_eq!(f.tree.get(&txn, &commitment(n)).unwrap().is_some(), live);
        }
    }

    #[test]
    fn block_can_spend_what_it_creates() {
        let f = Fixture::new(0);
        let mut txn = f.tmp.env().write_txn().unwrap();
        apply(&f, &mut txn, &[1], &[]).unwrap();
        apply(&f, &mut txn, &[2, 3], &[1, 2]).unwrap();
        assert!(f.tree.get(&txn, &commitment(2)).unwrap().is_none());
        assert!(f.tree.get(&txn, &commitment(3)).unwrap().is_some());
    }

    #[test]
    fn invalid_block_is_rolled_back_by_aborting() {
        let f = Fixture::new(3);
        let before = f.tree.state(&f.tmp.env().read_txn().unwrap()).unwrap();
        let cases: [(&[u32], &[u32], StateError); 4] = [
            (&[10], &[20], StateError::NotFound),
            (&[10], &[0, 0], StateError::NotFound),
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
        let (created, spent) = ([record(5), record(6)], [commitment(0)]);
        let proofs = f.tree.apply_changes(&mut txn, 1, &created, &spent).unwrap();

        let mut short = proofs.clone();
        short.inserts.pop();
        assert_eq!(
            verify_changes(&s0, 1, &created, &spent, &short),
            Err(ProofError::ProofCount)
        );
        let mut swapped = proofs.clone();
        swapped.inserts.swap(0, 1);
        assert!(verify_changes(&s0, 1, &created, &spent, &swapped).is_err());
        assert!(verify_changes(&s0, 1, &created, &[commitment(1)], &proofs).is_err());
        // The nonce is part of the state.
        let after = f.tree.state(&txn).unwrap();
        let mut other = created;
        other[0].nonce[0] ^= 1;
        assert_ne!(verify_changes(&s0, 1, &other, &spent, &proofs), Ok(after));
        // So is the height the outputs were created at.
        assert_ne!(verify_changes(&s0, 2, &created, &spent, &proofs), Ok(after));
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
