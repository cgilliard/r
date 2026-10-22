//! Blocks: the published input and output lists, how a block's state
//! transition is proved, and how its lists are applied.
//!
//! A block publishes the positions of the outputs it spends (ascending
//! state-tree slots) and its new output records (ascending state-tree key,
//! no duplicates), and one chain proof ([`crate::chain`]) covering its own
//! transition and the whole history before it. Transactions stay private.
//! A block can't spend an output it creates (every spent position is
//! below the slot counter before the block), so every input was confirmed
//! in an earlier block.
//!
//! Fees are implicit and there is no separate reward rule: transactions
//! need not balance, and the only value rule is the block's, that
//! `sum(outputs) - sum(inputs)` equals the subsidy. A block reward is just a
//! transaction with outputs and no inputs, claiming the subsidy plus
//! whatever the other transactions leave unspent. The proofs carry only that
//! net value, so nothing about fees is published.
//!
//! [`BlockProver`] proves a block's transition as one aggregated proof
//! ([`crate::aggregate`]). The transactions become one stream of items
//! (each transaction's start, inputs, outputs and end), cut into item
//! chunks of a few fixed sizes ([`ITEM_SHAPES`]) wherever the stream
//! happens to cross a chunk boundary, so a transaction of any size fits and
//! only the last chunk has padding. Each input's signature is checked in a
//! signature chunk ([`SIG_SHAPES`]), and the tree updates in segments. The
//! three are tied together by multiset fingerprints, under challenges
//! `lambda, r` that must be fixed only after everything is
//! ([`challenges`]): they hash the previous and new state, whose roots
//! commit to every key inserted and spent, and the transaction hash, which
//! commits (blinded) to every chunk's contents. The chain proof checks
//! that derivation, the fingerprint ratios, the net value, that the item
//! stream starts and ends between transactions, and the state link,
//! in-circuit.
//!
//! The published lists are bound through the state: applying them to the
//! previous tree must give the header's new state ([`apply`]). Output
//! nonces are in the tree's leaves, so the state binds them too. Spent
//! outputs are listed by their position (state-tree slot, 5 bytes), not
//! their commitment: every node has the tree, so it looks the commitment
//! up, and slots are never reused, so a position names one output.
//!
//! On the wire a block is its header and body ([`encode`]), at most
//! [`MAX_BLOCK_SIZE`] bytes. A [`Header`] commits to the block through
//! [`Block::body_hash`], which covers the proof too.

use std::fmt;

use crate::aggregate::{AggError, Aggregator, NodeProof};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::chunk;
use crate::fri::FriParams;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::header::Header;
use crate::item_chunk_air::{self, Boundary, Item, ItemChunkAir};
use crate::output::{Amount, COMMITMENT_LEN, Commitment, NONCE_LEN, OutputRecord};
use crate::par;
use crate::rng::Rng;
use crate::sig_chunk_air::{self, SigChunkAir, SigItem};
use crate::stark::{self, KeyCache, LazyKey, StarkProof};
use crate::state::{self, State, StateTree, Undo, key_of};
use crate::state_air::{InsertOp, SegmentStart, SpendOp, StateSegmentAir};
use crate::transaction::{Transaction, TxError};

/// What a block publishes.
#[derive(Clone, Debug)]
pub struct Block {
    /// Spent outputs' positions (state-tree slots), ascending.
    pub inputs: Vec<u64>,
    /// New outputs, in canonical order.
    pub outputs: Vec<OutputRecord>,
    /// The chain proof ([`crate::chain`]).
    pub proof: StarkProof,
}

#[derive(Debug)]
pub enum BlockError {
    /// A transaction is invalid.
    Tx(usize, TxError),
    /// A transaction's input names a position other than the spent
    /// output's slot.
    Position(usize),
    /// The lists aren't in canonical order, repeat a commitment or
    /// position, or hold a commitment that isn't a valid key or a position
    /// that can't hold an output.
    NotCanonical,
    /// The block spends an output it creates, which isn't allowed (every
    /// input must be confirmed in an earlier block).
    SpendsItsOwnOutput,
    /// An input's position holds no unspent output.
    NotAnOutput(u64),
    /// The state tree rejected an operation (a missing or spent input, a
    /// duplicate output).
    State(state::Error),
    /// Aggregation or chain proving failed (prover), or the chain proof
    /// doesn't verify (verifier).
    Proof(AggError),
    /// The transactions don't create exactly the subsidy: their
    /// `sum(outputs) - sum(inputs)` is this.
    Net(i128),
    /// The header doesn't commit to this body.
    Body,
    /// The serialized block would exceed [`MAX_BLOCK_SIZE`]: its size.
    Oversize(usize),
    /// The tree isn't at the parent's state, or applying the block doesn't
    /// give the header's.
    StateMismatch,
    /// The chain proof doesn't verify.
    Chain(crate::stark::StarkError),
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tx(i, e) => write!(f, "transaction {i}: {e:?}"),
            Self::Position(i) => write!(f, "transaction {i} names the wrong position for an input"),
            Self::NotCanonical => write!(f, "lists not in canonical order"),
            Self::SpendsItsOwnOutput => write!(f, "spends an output it creates"),
            Self::NotAnOutput(p) => write!(f, "spends position {p}, which holds no unspent output"),
            Self::State(e) => write!(f, "state: {e:?}"),
            Self::Proof(e) => write!(f, "proof: {e}"),
            Self::Net(n) => write!(f, "transactions create {n}, not the subsidy"),
            Self::Body => write!(f, "header doesn't commit to the body"),
            Self::Oversize(n) => write!(f, "block is {n} bytes, over {MAX_BLOCK_SIZE}"),
            Self::StateMismatch => write!(f, "applying the block doesn't give the header's state"),
            Self::Chain(e) => write!(f, "chain proof: {e}"),
        }
    }
}

impl std::error::Error for BlockError {}

impl Block {
    /// The hash a header commits to: the chunk hash ([`crate::chunk`]) of
    /// the encoded body ([`encode_body`]), which covers both lists (with
    /// output nonces) and the proof, and lets a peer check each chunk of a
    /// body it downloads.
    pub fn body_hash(&self) -> Hash {
        chunk::hash_of(domain::BLOCK_BODY, &encode_body(self))
    }
}

/// Whether a commitment is in both lists (a block spending an output it
/// creates).
pub(crate) fn spends_own_output(inputs: &[Commitment], outputs: &[OutputRecord]) -> bool {
    let created: std::collections::HashSet<&Commitment> =
        outputs.iter().map(|o| &o.commitment).collect();
    inputs.iter().any(|c| created.contains(c))
}

/// Whether `positions` are strictly ascending slots that can hold an
/// output (slot 0 is the tree's sentinel).
pub(crate) fn positions_canonical(positions: &[u64]) -> bool {
    positions.first().is_none_or(|&p| p > 0)
        && positions.last().is_none_or(|&p| p < state::CAPACITY)
        && positions.windows(2).all(|w| w[0] < w[1])
}

/// Whether ascending `positions` reach a slot filled by the block itself:
/// one at or above `prev_size`, the slot counter before the block.
pub(crate) fn spends_new_slot(positions: &[u64], prev_size: u64) -> bool {
    positions.last().is_some_and(|&p| p >= prev_size)
}

/// The canonical order: ascending key, compared on canonical limbs.
fn order_key(c: &Commitment) -> Option<[u32; HASH_LEN]> {
    key_of(c).ok().map(|k| k.map(BabyBear::as_u32))
}

/// Whether `commitments` are valid keys in strictly ascending order.
pub(crate) fn is_canonical<'a>(commitments: impl IntoIterator<Item = &'a Commitment>) -> bool {
    let mut last: Option<[u32; HASH_LEN]> = None;
    for c in commitments {
        match order_key(c) {
            Some(k) if last.is_none_or(|l| l < k) => last = Some(k),
            _ => return false,
        }
    }
    true
}

fn sort_canonical<T>(
    items: &mut [T],
    commitment: impl Fn(&T) -> &Commitment,
) -> Result<(), BlockError> {
    let mut keys = Vec::with_capacity(items.len());
    for it in items.iter() {
        keys.push(order_key(commitment(it)).ok_or(BlockError::NotCanonical)?);
    }
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by_key(|&i| keys[i]);
    if idx.windows(2).any(|w| keys[w[0]] == keys[w[1]]) {
        return Err(BlockError::NotCanonical);
    }
    // Apply the permutation.
    let mut pos: Vec<usize> = vec![0; items.len()];
    for (to, &from) in idx.iter().enumerate() {
        pos[from] = to;
    }
    for i in 0..items.len() {
        while pos[i] != i {
            let j = pos[i];
            items.swap(i, j);
            pos.swap(i, j);
        }
    }
    Ok(())
}

/// The multiset challenges for a block: a hash of the previous and new
/// state and the transaction hash.
pub fn challenges(prev: &State, new: &State, tx_hash: &Hash) -> (BabyBear4, BabyBear4) {
    let (a, b) = state::split(prev.size);
    let (c, d) = state::split(new.size);
    let h = hash::sponge(
        domain::BLOCK_CHALLENGES,
        &[],
        &[&prev.root, &[a, b], &new.root, &[c, d], tx_hash],
    );
    (
        BabyBear4(core::array::from_fn(|i| h[i])),
        BabyBear4(core::array::from_fn(|i| h[4 + i])),
    )
}

/// The transaction hash of an aggregation tree with these leaf hashes
/// (zero for segments), merged pairwise level by level as
/// [`BlockProver::prove`] merges the nodes.
fn merge_tx_hashes(mut level: Vec<Hash>) -> Hash {
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|p| match p {
                [a, b] => hash::merge(a, b),
                [a] => *a,
                _ => unreachable!(),
            })
            .collect();
    }
    level[0]
}

/// The most bytes a serialized block ([`encode`]) may take.
pub const MAX_BLOCK_SIZE: usize = 2_000_000;

/// Bytes per serialized input: its position (a slot below
/// [`state::CAPACITY`]).
pub const INPUT_BYTES: usize = state::DEPTH / 8;
const _: () = assert!(state::DEPTH.is_multiple_of(8));
/// Bytes per serialized output: its commitment and nonce.
pub const OUTPUT_BYTES: usize = COMMITMENT_LEN + NONCE_LEN;

/// The serialized size of a block with these list lengths and proof size.
pub fn encoded_size(inputs: usize, outputs: usize, proof: usize) -> usize {
    crate::header::HEADER_BYTES + 12 + inputs * INPUT_BYTES + outputs * OUTPUT_BYTES + proof
}

/// Serializes a block with its header: `header | n_in (u32) | inputs
/// (positions, 5 bytes each) | n_out (u32) | outputs (commitment, nonce) | proof length (u32) | proof`,
/// integers little-endian.
pub fn encode(header: &Header, block: &Block) -> Vec<u8> {
    let mut out = header.to_bytes().to_vec();
    out.extend(encode_body(block));
    out
}

/// A block's body as on the wire, without its header: the input count and
/// inputs, the output count and outputs, the proof length and proof.
pub fn encode_body(block: &Block) -> Vec<u8> {
    let proof = block.proof.to_bytes();
    let mut out = Vec::with_capacity(
        encoded_size(block.inputs.len(), block.outputs.len(), proof.len())
            - crate::header::HEADER_BYTES,
    );
    out.extend((block.inputs.len() as u32).to_le_bytes());
    for p in &block.inputs {
        out.extend(&p.to_le_bytes()[..INPUT_BYTES]);
    }
    out.extend((block.outputs.len() as u32).to_le_bytes());
    for o in &block.outputs {
        out.extend(o.commitment.0);
        out.extend(o.nonce);
    }
    out.extend((proof.len() as u32).to_le_bytes());
    out.extend(proof);
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Over [`MAX_BLOCK_SIZE`].
    Oversize(usize),
    /// Ends early, has bytes left over, or a part doesn't parse.
    Malformed,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oversize(n) => write!(f, "block is {n} bytes, over {MAX_BLOCK_SIZE}"),
            Self::Malformed => write!(f, "malformed block"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Parses a block written by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<(Header, Block), DecodeError> {
    if bytes.len() > MAX_BLOCK_SIZE {
        return Err(DecodeError::Oversize(bytes.len()));
    }
    let (h, body) = bytes
        .split_at_checked(crate::header::HEADER_BYTES)
        .ok_or(DecodeError::Malformed)?;
    let header = Header::from_bytes(h).ok_or(DecodeError::Malformed)?;
    Ok((header, decode_body(body)?))
}

/// The inverse of [`encode_body`].
pub fn decode_body(bytes: &[u8]) -> Result<Block, DecodeError> {
    if bytes.len() + crate::header::HEADER_BYTES > MAX_BLOCK_SIZE {
        return Err(DecodeError::Oversize(
            bytes.len() + crate::header::HEADER_BYTES,
        ));
    }
    let mut rest = bytes;
    let mut take = |n: usize| -> Result<&[u8], DecodeError> {
        if rest.len() < n {
            return Err(DecodeError::Malformed);
        }
        let (a, b) = rest.split_at(n);
        rest = b;
        Ok(a)
    };
    let count = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize;
    let n_in = count(take(4)?);
    let mut inputs = Vec::with_capacity(n_in.min(MAX_BLOCK_SIZE / INPUT_BYTES));
    for _ in 0..n_in {
        let mut p = [0; 8];
        p[..INPUT_BYTES].copy_from_slice(take(INPUT_BYTES)?);
        inputs.push(u64::from_le_bytes(p));
    }
    let n_out = count(take(4)?);
    let mut outputs = Vec::with_capacity(n_out.min(MAX_BLOCK_SIZE / OUTPUT_BYTES));
    for _ in 0..n_out {
        let commitment = Commitment(take(COMMITMENT_LEN)?.try_into().expect("length"));
        let nonce = take(NONCE_LEN)?.try_into().expect("length");
        outputs.push(OutputRecord { commitment, nonce });
    }
    let proof_len = count(take(4)?);
    let proof = StarkProof::from_bytes(take(proof_len)?).ok_or(DecodeError::Malformed)?;
    if !rest.is_empty() {
        return Err(DecodeError::Malformed);
    }
    Ok(Block {
        inputs,
        outputs,
        proof,
    })
}

/// Applies `block`'s lists to `tree` (whose state in `txn` must be
/// `parent`'s) and checks the result is `header`'s state. Returns what
/// undoes the changes ([`StateTree::revert`]). On error `txn` may be partly
/// updated; abort it.
pub fn apply(
    tree: &StateTree,
    txn: &mut heed::RwTxn,
    parent: &Header,
    header: &Header,
    block: &Block,
) -> Result<Undo, BlockError> {
    if tree.state(txn).map_err(BlockError::State)? != parent.state {
        return Err(BlockError::StateMismatch);
    }
    if !positions_canonical(&block.inputs) {
        return Err(BlockError::NotCanonical);
    }
    if spends_new_slot(&block.inputs, parent.state.size) {
        return Err(BlockError::SpendsItsOwnOutput);
    }
    // What the positions hold, before the block's outputs go in.
    let mut spent = Vec::with_capacity(block.inputs.len());
    for &p in &block.inputs {
        spent.push(
            tree.commitment_at(txn, p)
                .map_err(BlockError::State)?
                .ok_or(BlockError::NotAnOutput(p))?,
        );
    }
    let undo = tree
        .apply_block(txn, header.height, &block.outputs, &spent)
        .map_err(BlockError::State)?;
    if tree.state(txn).map_err(BlockError::State)? != header.state {
        return Err(BlockError::StateMismatch);
    }
    Ok(undo)
}

/// A block's transition, proved: its lists and the final aggregation
/// proof, which [`crate::chain`] turns into the block's chain proof, with
/// what its statement's state hashes open to.
#[derive(Clone, Debug)]
pub struct Aggregated {
    /// Spent positions, ascending.
    pub inputs: Vec<u64>,
    pub outputs: Vec<OutputRecord>,
    pub node: NodeProof,
    /// The state before and after.
    pub prev: State,
    pub new: State,
    /// The last keys inserted and spent (zero if none).
    pub last_insert: Hash,
    pub last_spend: Hash,
}

/// A block ready to prove: its transactions, lists and state-tree
/// witnesses ([`BlockProver::prepare`]).
#[derive(Clone, Debug)]
pub struct Prepared {
    txs: Vec<Transaction>,
    /// Spent positions, ascending.
    pub inputs: Vec<u64>,
    pub outputs: Vec<OutputRecord>,
    /// The state before and after.
    pub prev: State,
    pub new: State,
    segments: Vec<(SegmentStart, Vec<InsertOp>, Vec<SpendOp>)>,
    /// Where the last segment left off.
    last: SegmentStart,
}

/// The quick, stateful part of proving a block of `txs` on top of
/// `tree`'s state in `txn`: checks the transactions (they must create
/// exactly `subsidy` in total, and the block must fit
/// [`MAX_BLOCK_SIZE`] with a proof of `proof_size` bytes) and applies the changes in `txn`, recording the
/// witnesses of segments shaped `seg`. The transaction can then be aborted and the
/// proving done without it.
pub fn prepare(
    tree: &StateTree,
    txn: &mut heed::RwTxn,
    txs: &[Transaction],
    subsidy: Amount,
    seg: &StateSegmentAir,
    proof_size: usize,
) -> Result<Prepared, BlockError> {
    let mut net: i128 = 0;
    for (i, tx) in txs.iter().enumerate() {
        let totals = tx.validate().map_err(|e| BlockError::Tx(i, e))?;
        net -= totals.fee();
    }
    if net != subsidy as i128 {
        return Err(BlockError::Net(net));
    }
    let n_in: usize = txs.iter().map(|t| t.body.inputs.len()).sum();
    let n_out: usize = txs.iter().map(|t| t.body.outputs.len()).sum();
    let size = encoded_size(n_in, n_out, proof_size);
    if size > MAX_BLOCK_SIZE {
        return Err(BlockError::Oversize(size));
    }
    let mut inputs: Vec<Commitment> = txs
        .iter()
        .flat_map(|t| t.body.inputs.iter().map(|i| i.commitment))
        .collect();
    let mut outputs: Vec<OutputRecord> = txs
        .iter()
        .flat_map(|t| t.body.outputs.iter().map(|o| o.record()))
        .collect::<Option<_>>()
        .ok_or(BlockError::NotCanonical)?;
    sort_canonical(&mut inputs, |c| c)?;
    sort_canonical(&mut outputs, |o| &o.commitment)?;
    if spends_own_output(&inputs, &outputs) {
        return Err(BlockError::SpendsItsOwnOutput);
    }
    let prev = tree.state(txn).map_err(BlockError::State)?;

    // Apply the state changes natively, in segments: all inserts, then
    // all spends.
    let mut segments = Vec::new();
    let mut start = SegmentStart {
        state: prev,
        prev_insert: [BabyBear::ZERO; HASH_LEN],
        prev_spend: [BabyBear::ZERO; HASH_LEN],
    };
    let (mut ins, mut sps) = (&outputs[..], &inputs[..]);
    while !ins.is_empty() || !sps.is_empty() || segments.is_empty() {
        let (take_ins, rest_ins) = ins.split_at(ins.len().min(seg.inserts));
        ins = rest_ins;
        let take_sp = if ins.is_empty() {
            let (t, rest) = sps.split_at(sps.len().min(seg.spends));
            sps = rest;
            t
        } else {
            &[]
        };
        let inserts: Vec<InsertOp> = take_ins
            .iter()
            .map(|record| {
                Ok(InsertOp {
                    proof: tree.insert(txn, record)?,
                    record: *record,
                })
            })
            .collect::<Result<_, state::Error>>()
            .map_err(BlockError::State)?;
        let spends: Vec<SpendOp> = take_sp
            .iter()
            .map(|c| {
                Ok(SpendOp {
                    proof: tree.remove(txn, c)?,
                    commitment: *c,
                })
            })
            .collect::<Result<_, state::Error>>()
            .map_err(BlockError::State)?;
        let next = SegmentStart {
            state: tree.state(txn).map_err(BlockError::State)?,
            prev_insert: take_ins.last().map_or(start.prev_insert, |o| {
                key_of(&o.commitment).expect("checked")
            }),
            prev_spend: take_sp
                .last()
                .map_or(start.prev_spend, |c| key_of(c).expect("checked")),
        };
        segments.push((start, inserts, spends));
        start = next;
    }
    let new = start.state;

    // Each input must sign the position the tree removed it from.
    let positions: std::collections::HashMap<Commitment, u64> = segments
        .iter()
        .flat_map(|(_, _, spends)| spends.iter().map(|s| (s.commitment, s.proof.position())))
        .collect();
    for (i, tx) in txs.iter().enumerate() {
        if tx
            .body
            .inputs
            .iter()
            .any(|input| positions.get(&input.commitment) != Some(&input.position))
        {
            return Err(BlockError::Position(i));
        }
    }
    // Published by position.
    let mut spent: Vec<u64> = positions.into_values().collect();
    spent.sort_unstable();
    Ok(Prepared {
        txs: txs.to_vec(),
        inputs: spent,
        outputs,
        prev,
        new,
        segments,
        last: start,
    })
}

/// The smallest well-formed proof: what blocks carry when nodes run with
/// `--debug-no-proof-validation` (they neither make nor check proofs).
pub fn no_proof() -> StarkProof {
    let mut e = vec![BabyBear::new(3)];
    e.extend([BabyBear::ONE; HASH_LEN]);
    e.push(BabyBear::ZERO);
    e.extend([BabyBear::ONE; HASH_LEN]);
    e.extend([BabyBear::ZERO; 7]);
    e.extend([
        BabyBear::ZERO,
        BabyBear::ZERO,
        BabyBear::ONE,
        BabyBear::ZERO,
        BabyBear::ZERO,
    ]);
    StarkProof::from_elements(&e).expect("well formed")
}

/// The state segments [`BlockProver::standard`] proves.
pub const STANDARD_SEGMENT: StateSegmentAir = StateSegmentAir {
    inserts: 16,
    spends: 16,
};

/// Item chunk shapes, smallest first: each fills a power-of-two trace
/// (`2^12`, `2^14` rows).
pub const ITEM_SHAPES: [ItemChunkAir; 2] =
    [ItemChunkAir { items: 682 }, ItemChunkAir { items: 2730 }];

/// Signature chunk shapes, smallest first (`2^12`, `2^14`, `2^16` rows).
pub const SIG_SHAPES: [SigChunkAir; 3] = [
    SigChunkAir { slots: 7 },
    SigChunkAir { slots: 31 },
    SigChunkAir { slots: 124 },
];

/// Cuts `n` things into chunks: full chunks of the largest capacity, then
/// the smallest that holds the rest. Returns (shape index, range).
fn cut(n: usize, capacities: &[usize]) -> Vec<(usize, std::ops::Range<usize>)> {
    let largest = capacities.len() - 1;
    let mut out = Vec::new();
    let mut at = 0;
    while at < n {
        let left = n - at;
        let shape = if left >= capacities[largest] {
            largest
        } else {
            capacities
                .iter()
                .position(|&cap| cap >= left)
                .expect("the largest holds it")
        };
        let take = left.min(capacities[shape]);
        out.push((shape, at..at + take));
        at += take;
    }
    out
}

/// Keys and shapes for proving blocks' transitions.
pub struct BlockProver {
    params: FriParams,
    /// Chunk shapes, smallest first, with their keys.
    items: Vec<(ItemChunkAir, LazyKey)>,
    sigs: Vec<(SigChunkAir, LazyKey)>,
    seg: StateSegmentAir,
    seg_key: LazyKey,
    agg: Aggregator,
}

impl BlockProver {
    /// Sets up proving with item and signature chunks of the shapes
    /// `items` and `sigs` (smallest first), segments of shape `seg`, and
    /// aggregation nodes of height `2^log_height`, with verifying keys from
    /// `keys` where it has them (proving keys are built on first use).
    pub fn new(
        params: FriParams,
        items: &[ItemChunkAir],
        sigs: &[SigChunkAir],
        seg: StateSegmentAir,
        log_height: u32,
        keys: &KeyCache,
    ) -> Self {
        let items: Vec<_> = items
            .iter()
            .map(|&air| (air, LazyKey::new(keys, &params, &air)))
            .collect();
        let sigs: Vec<_> = sigs
            .iter()
            .map(|&air| (air, LazyKey::new(keys, &params, &air)))
            .collect();
        let seg_key = LazyKey::new(keys, &params, &seg);
        let agg = Aggregator::new(
            params,
            items
                .iter()
                .map(|(air, k)| (*air, *k.verifying_key()))
                .collect(),
            sigs.iter()
                .map(|(air, k)| (*air, *k.verifying_key()))
                .collect(),
            seg,
            *seg_key.verifying_key(),
            log_height,
            keys,
        );
        Self {
            params,
            items,
            sigs,
            seg,
            seg_key,
            agg,
        }
    }

    /// Sets up the default shapes: [`ITEM_SHAPES`], [`SIG_SHAPES`], 16-op
    /// segments, nodes at `2^15`.
    pub fn standard(params: FriParams, keys: &KeyCache) -> Self {
        Self::new(
            params,
            &ITEM_SHAPES,
            &SIG_SHAPES,
            STANDARD_SEGMENT,
            15,
            keys,
        )
    }

    /// The aggregator, whose merge key verifiers need.
    pub fn aggregator(&self) -> &Aggregator {
        &self.agg
    }

    /// Proves the transition of a block of `txs` on top of `tree`'s state
    /// in `txn`, applying its changes there (commit `txn` to keep them):
    /// [`BlockProver::prepare`] then [`BlockProver::prove_prepared`].
    pub fn prove(
        &self,
        tree: &StateTree,
        txn: &mut heed::RwTxn,
        txs: &[Transaction],
        subsidy: Amount,
    ) -> Result<Aggregated, BlockError> {
        let prepared = self.prepare(tree, txn, txs, subsidy)?;
        self.prove_prepared(&prepared)
    }

    /// The quick, stateful part of proving a block of `txs` on top of
    /// `tree`'s state in `txn`: [`prepare`], with this prover's segments and
    /// proof size.
    pub fn prepare(
        &self,
        tree: &StateTree,
        txn: &mut heed::RwTxn,
        txs: &[Transaction],
        subsidy: Amount,
    ) -> Result<Prepared, BlockError> {
        prepare(tree, txn, txs, subsidy, &self.seg, self.agg.proof_size())
    }

    /// The slow part: proves a [`BlockProver::prepare`]d block.
    pub fn prove_prepared(&self, p: &Prepared) -> Result<Aggregated, BlockError> {
        let Prepared {
            txs,
            inputs,
            outputs,
            prev,
            new,
            segments,
            last: start,
        } = p;
        let (prev, new) = (*prev, *new);

        // The item stream and the signatures, cut into chunks.
        let items = item_chunk_air::stream(txs);
        let item_caps: Vec<usize> = self.items.iter().map(|(a, _)| a.items).collect();
        let item_chunks = cut(items.len(), &item_caps);
        let sig_items: Vec<SigItem> = txs
            .iter()
            .flat_map(|tx| {
                let msg = tx.body.digest().expect("validated");
                tx.body
                    .inputs
                    .iter()
                    .zip(&tx.signatures)
                    .map(move |(input, sig)| SigItem {
                        pubkey: input.pubkey.clone(),
                        leaf: input.leaf,
                        path: input.path,
                        sig: sig.clone(),
                        msg,
                    })
            })
            .collect();
        let sig_caps: Vec<usize> = self.sigs.iter().map(|(a, _)| a.slots).collect();
        let sig_chunks = cut(sig_items.len(), &sig_caps);

        // Fix each chunk's blinded hash, then derive the challenges from
        // both halves.
        let mut rng = Rng::from_os().expect("OS randomness");
        let mut blinding = || {
            let mut b = [BabyBear::ZERO; HASH_LEN];
            rng.fill(&mut b);
            b
        };
        let mut leaves = vec![[BabyBear::ZERO; HASH_LEN]; segments.len()];
        let mut item_jobs = Vec::with_capacity(item_chunks.len());
        let mut entry = Boundary::IDLE;
        let mut msg = [BabyBear::ZERO; HASH_LEN];
        for (shape, range) in &item_chunks {
            let mut digests = Vec::new();
            let mut exit = entry;
            for item in &items[range.clone()] {
                if let Item::Start { msg: m, .. } = item {
                    msg = *m;
                }
                if *item == Item::End {
                    digests.push(msg);
                }
                exit = exit.after(item);
            }
            let b = blinding();
            leaves.push(item_chunk_air::chunk_hash(&b, &digests));
            item_jobs.push((*shape, range.clone(), entry, b));
            entry = exit;
        }
        debug_assert_eq!(entry, Boundary::IDLE);
        let mut sig_jobs = Vec::with_capacity(sig_chunks.len());
        for (shape, range) in &sig_chunks {
            let pairs: Vec<(Hash, Hash)> = sig_items[range.clone()]
                .iter()
                .map(|s| (s.root(), s.msg))
                .collect();
            let b = blinding();
            leaves.push(sig_chunk_air::chunk_hash(&b, &pairs));
            sig_jobs.push((*shape, range.clone(), b));
        }
        let tx_hash = merge_tx_hashes(leaves);
        let (lambda, r) = challenges(&prev, &new, &tx_hash);

        // Prove the segments, item chunks and signature chunks, then merge
        // pairwise, keeping the segments' and item chunks' order. Leaves,
        // and merges on a level, are independent: a few run at once.
        type Task<'a> = Box<dyn FnOnce() -> Result<NodeProof, AggError> + Send + 'a>;
        let (params, items, sig_items) = (&self.params, &items, &sig_items);
        let mut tasks: Vec<Task<'_>> = Vec::new();
        for (start, inserts, spends) in segments {
            tasks.push(Box::new(move || {
                let (trace, public) = self.seg.trace(start, inserts, spends, lambda, r);
                let pk = self.seg_key.proving_key(params, &self.seg);
                let proof = stark::prove(params, pk, &self.seg, trace, &public);
                self.agg.prove_segment(&public, &proof)
            }));
        }
        for (shape, range, entry, b) in &item_jobs {
            tasks.push(Box::new(move || {
                let (air, key) = &self.items[*shape];
                let (trace, public) = air.trace(&items[range.clone()], entry, b, lambda, r);
                let proof = stark::prove(params, key.proving_key(params, air), air, trace, &public);
                self.agg.prove_items(*shape, &public, &proof)
            }));
        }
        for (shape, range, b) in &sig_jobs {
            tasks.push(Box::new(move || {
                let (air, key) = &self.sigs[*shape];
                let mut rng = Rng::from_os().expect("OS randomness");
                let (trace, public) = air.trace(&sig_items[range.clone()], b, lambda, r, &mut rng);
                let proof = stark::prove(params, key.proving_key(params, air), air, trace, &public);
                self.agg.prove_sigs(*shape, &public, &proof)
            }));
        }
        let mut nodes = par::run_jobs(tasks)
            .into_iter()
            .collect::<Result<Vec<NodeProof>, _>>()
            .map_err(BlockError::Proof)?;
        while nodes.len() > 1 {
            let odd = (nodes.len() % 2 == 1).then(|| nodes.pop().expect("odd"));
            let tasks: Vec<Task<'_>> = nodes
                .chunks(2)
                .map(|pair| Box::new(move || self.agg.merge(&pair[0], &pair[1])) as Task<'_>)
                .collect();
            let mut next = par::run_jobs(tasks)
                .into_iter()
                .collect::<Result<Vec<NodeProof>, _>>()
                .map_err(BlockError::Proof)?;
            next.extend(odd);
            nodes = next;
        }
        let top = nodes.pop().expect("at least one segment");
        if top.kind != crate::aggregate::Kind::Merge {
            // A lone segment: merge with nothing isn't possible; every block
            // has at least one transaction, so this doesn't happen.
            return Err(BlockError::Proof(AggError::Invalid));
        }
        debug_assert_eq!(top.statement.tx_hash(), tx_hash);
        Ok(Aggregated {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
            node: top,
            prev,
            new,
            last_insert: start.prev_insert,
            last_spend: start.prev_spend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::output::{self, NONCE_LEN, Nonce, Output};
    use crate::transaction::{Input, KeyTree, TxBody};
    use crate::wots::{SecretKey, keygen};

    const SUBSIDY: Amount = 50;

    #[test]
    fn shapes_fill_their_traces() {
        for (shape, log) in ITEM_SHAPES.iter().zip([12, 14]) {
            assert_eq!(shape.height(), 1 << log, "{shape:?}");
            let bigger = ItemChunkAir {
                items: shape.items + 1,
            };
            assert_eq!(bigger.height(), 2 << log, "{shape:?} isn't full");
        }
        for (shape, log) in SIG_SHAPES.iter().zip([12, 14, 16]) {
            assert_eq!(shape.height(), 1 << log, "{shape:?}");
            let bigger = SigChunkAir {
                slots: shape.slots + 1,
            };
            assert_eq!(bigger.height(), 2 << log, "{shape:?} isn't full");
        }
    }

    #[test]
    fn inputs_are_ascending_positions_of_five_bytes() {
        assert_eq!(INPUT_BYTES, 5);
        assert!(positions_canonical(&[]));
        assert!(positions_canonical(&[1, 7, state::CAPACITY - 1]));
        for bad in [&[0][..], &[3, 3], &[4, 2], &[state::CAPACITY]] {
            assert!(!positions_canonical(bad), "{bad:?}");
        }
        assert!(spends_new_slot(&[2, 9], 9) && !spends_new_slot(&[2, 9], 10));
        assert!(!spends_new_slot(&[], 0));
        // The largest position survives encoding.
        let block = Block {
            inputs: vec![1, state::CAPACITY - 1],
            outputs: vec![],
            proof: crate::ledger::tests::tiny_proof(),
        };
        let bytes = encode_body(&block);
        assert_eq!(
            &bytes[..4 + 2 * INPUT_BYTES],
            &[2, 0, 0, 0, 1, 0, 0, 0, 0, 255, 255, 255, 255, 255]
        );
        assert_eq!(decode_body(&bytes).unwrap().inputs, block.inputs);
    }

    #[test]
    fn cuts_into_full_chunks_and_a_fitting_rest() {
        let caps = [7, 31, 124];
        assert_eq!(cut(0, &caps), vec![]);
        assert_eq!(cut(1, &caps), vec![(0, 0..1)]);
        assert_eq!(cut(7, &caps), vec![(0, 0..7)]);
        assert_eq!(cut(8, &caps), vec![(1, 0..8)]);
        assert_eq!(cut(124, &caps), vec![(2, 0..124)]);
        assert_eq!(cut(128, &caps), vec![(2, 0..124), (0, 124..128)]);
        assert_eq!(
            cut(300, &caps),
            vec![(2, 0..124), (2, 124..248), (2, 248..300)]
        );
    }

    fn key(n: u32) -> (SecretKey, crate::wots::PublicKey) {
        keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 31 + i as u32
        })))
    }

    /// An output paying `amount` to key `n`.
    fn pay(n: u32, amount: Amount) -> Output {
        Output {
            hash: KeyTree::single(&key(n).1).output_hash(),
            amount,
            nonce: [n as u8; NONCE_LEN],
        }
    }

    fn spend(n: u32, amount: Amount, nonce: Nonce, position: u64) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let tree = KeyTree::single(&pubkey);
        let commitment = output::commit(&tree.output_hash(), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                position,
                pubkey,
                leaf: 0,
                path: tree.path(0),
                amount,
                nonce,
            },
        )
    }

    /// A block with one transaction (1 input, 2 outputs, fee 10) and a
    /// reward claiming the subsidy plus the fee: its transition proved, its
    /// encoding, and applying it.
    #[test]
    #[ignore = "slow: run in release"]
    fn proves_and_verifies_a_block() {
        let t = std::time::Instant::now();
        let prover = BlockProver::standard(FriParams::default(), &stark::KeyCache::none());
        eprintln!("setup {:.1} s", t.elapsed().as_secs_f64());

        // The chain so far holds one output of 100 to key 1.
        let funding = pay(1, 100);
        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        tree.insert(&mut txn, &funding.record().unwrap()).unwrap();
        txn.commit().unwrap();

        // It's in slot 1 (the sentinel has slot 0). Signing another slot is
        // refused before proving.
        let (sk, input) = spend(1, 100, funding.nonce, 2);
        let wrong = Transaction::sign(
            TxBody {
                inputs: vec![input],
                outputs: vec![pay(2, 60), pay(3, 30)],
            },
            vec![sk],
        )
        .unwrap();
        let reward = Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs: vec![pay(4, SUBSIDY + 10)],
            },
            vec![],
        )
        .unwrap();
        let mut txn = env.env().write_txn().unwrap();
        assert!(matches!(
            prover.prove(&tree, &mut txn, &[wrong, reward], SUBSIDY),
            Err(BlockError::Position(0))
        ));
        txn.abort();

        let (sk, input) = spend(1, 100, funding.nonce, 1);
        let tx = Transaction::sign(
            TxBody {
                inputs: vec![input],
                outputs: vec![pay(2, 60), pay(3, 30)],
            },
            vec![sk],
        )
        .unwrap();
        let reward = Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs: vec![pay(4, SUBSIDY + 10)],
            },
            vec![],
        )
        .unwrap();

        // Transactions that don't create exactly the subsidy are refused
        // before proving: here the reward claims one unit too many.
        let greedy = Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs: vec![pay(5, SUBSIDY + 11)],
            },
            vec![],
        )
        .unwrap();
        let mut txn = env.env().write_txn().unwrap();
        assert!(matches!(
            prover.prove(&tree, &mut txn, &[tx.clone(), greedy], SUBSIDY),
            Err(BlockError::Net(51))
        ));
        txn.abort();

        // The miner proves the transition (and drops its changes).
        let t = std::time::Instant::now();
        let mut txn = env.env().write_txn().unwrap();
        let prev = tree.state(&txn).unwrap();
        let agg = prover
            .prove(&tree, &mut txn, &[tx, reward], SUBSIDY)
            .unwrap();
        let claimed = tree.state(&txn).unwrap();
        txn.abort();
        eprintln!("prove {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!((agg.inputs.len(), agg.outputs.len()), (1, 3));

        // The final statement: what the chain proof checks in-circuit.
        let aggregator = prover.aggregator();
        let s = &agg.node.statement;
        assert_eq!(aggregator.verify(s, &agg.node.proof), Ok(()));
        let zero = [BabyBear::ZERO; HASH_LEN];
        assert_eq!((agg.prev, agg.new), (prev, claimed));
        assert_eq!(
            s.state_in(),
            crate::aggregate::state_hash(&prev, &zero, &zero)
        );
        assert_eq!(
            s.state_out(),
            crate::aggregate::state_hash(&claimed, &agg.last_insert, &agg.last_spend)
        );
        let idle = crate::aggregate::stream_hash(&Boundary::IDLE);
        assert_eq!((s.stream_in(), s.stream_out()), (idle, idle));
        assert_eq!(
            (s.lambda(), s.r()),
            challenges(&prev, &claimed, &s.tx_hash())
        );
        assert_eq!(
            (s.q_in(), s.q_out(), s.q_sig()),
            (BabyBear4::ONE, BabyBear4::ONE, BabyBear4::ONE)
        );
        assert_eq!(s.net(), SUBSIDY as i128);

        // Over the wire, with the aggregation proof standing in for the
        // chain proof: the encoding round-trips, and its size is as
        // computed.
        let block = Block {
            inputs: agg.inputs.clone(),
            outputs: agg.outputs.clone(),
            proof: agg.node.proof.clone(),
        };
        let parent = Header {
            height: 7,
            timestamp: 1000,
            state: prev,
            ..Header::default()
        };
        let header = Header {
            height: 8,
            prev: parent.hash(),
            timestamp: 1600,
            state: claimed,
            body: block.body_hash(),
            nonce: 0,
        };
        let bytes = encode(&header, &block);
        assert_eq!(bytes.len(), encoded_size(1, 3, aggregator.proof_size()));
        eprintln!("serialized {} bytes", bytes.len());
        let (h2, b2) = decode(&bytes).unwrap();
        assert_eq!(h2, header);
        assert_eq!((&b2.inputs, &b2.outputs), (&block.inputs, &block.outputs));
        assert_eq!(b2.proof.elements(), block.proof.elements());
        assert_eq!(b2.body_hash(), block.body_hash());
        assert_eq!(
            decode(&bytes[..bytes.len() - 1]).err(),
            Some(DecodeError::Malformed)
        );
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(decode(&long).err(), Some(DecodeError::Malformed));
        let mut huge = bytes.clone();
        huge.resize(MAX_BLOCK_SIZE + 1, 0);
        assert_eq!(
            decode(&huge).err(),
            Some(DecodeError::Oversize(MAX_BLOCK_SIZE + 1))
        );
        // The body hash covers the proof and the nonces.
        let mut b = block.clone();
        b.proof.fri.pow_witness += BabyBear::ONE;
        assert_ne!(b.body_hash(), block.body_hash());
        let mut b = block.clone();
        b.outputs[1].nonce[3] ^= 1;
        assert_ne!(b.body_hash(), block.body_hash());

        // The lists are bound through the state: a body missing an output
        // doesn't reach the header's state.
        let mut b = block.clone();
        b.outputs.remove(0);
        let mut txn = env.env().write_txn().unwrap();
        assert!(matches!(
            apply(&tree, &mut txn, &parent, &header, &b),
            Err(BlockError::StateMismatch)
        ));
        txn.abort();
        let mut txn = env.env().write_txn().unwrap();
        apply(&tree, &mut txn, &parent, &header, &block).unwrap();
        assert_eq!(tree.state(&txn).unwrap(), claimed);
        txn.commit().unwrap();
    }
}
