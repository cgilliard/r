//! Blocks: the published input and output lists, how a block's state
//! transition is proved, and how its lists are applied.
//!
//! A block publishes its spent input commitments and its new output
//! records, each in canonical order (ascending state-tree key, no
//! duplicates), and one chain proof ([`crate::chain`]) covering its own
//! transition and the whole history before it. Transactions stay private.
//!
//! Fees are implicit and there is no separate reward rule: transactions
//! need not balance, and the only value rule is the block's, that
//! `sum(outputs) - sum(inputs)` equals the subsidy. A block reward is just a
//! transaction with outputs and no inputs, claiming the subsidy plus
//! whatever the other transactions leave unspent. The proofs carry only that
//! net value, so nothing about fees is published.
//!
//! [`BlockProver`] proves a block's transition as one aggregated proof
//! ([`crate::aggregate`]). Its two private halves, the transactions
//! (sub-blocks) and the tree updates (segments), are tied together by
//! multiset fingerprints over commitments, under challenges `lambda, r`
//! that must be fixed only after both halves are ([`challenges`]): they
//! hash the previous and new state, whose roots commit to every key
//! inserted and spent, and the transaction hash, which commits (blinded) to
//! every commitment the transactions spend and create. The chain proof
//! checks that derivation, the fingerprint ratios, the net value and the
//! state link in-circuit.
//!
//! The published lists are bound through the state: applying them to the
//! previous tree must give the header's new state ([`apply`]). Output
//! nonces are published for wallets but aren't fingerprinted; only the body
//! hash commits to them.
//!
//! On the wire a block is its header and body ([`encode`]), at most
//! [`MAX_BLOCK_SIZE`] bytes. A [`Header`] commits to the block through
//! [`Block::body_hash`], which covers the proof too.

use std::fmt;

use crate::aggregate::{AggError, Aggregator, NodeProof};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::fri::FriParams;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::header::Header;
use crate::output::{self, Amount, COMMITMENT_LEN, Commitment, NONCE_LEN, OutputRecord};
use crate::rng::Rng;
use crate::stark::{self, ProvingKey, StarkProof};
use crate::state::{self, State, StateTree, key_of};
use crate::state_air::{InsertOp, SegmentStart, SpendOp, StateSegmentAir};
use crate::subblock_air::{self, SubBlockAir};
use crate::transaction::{Transaction, TxError};

/// What a block publishes.
#[derive(Clone, Debug)]
pub struct Block {
    /// Spent commitments, in canonical order.
    pub inputs: Vec<Commitment>,
    /// New outputs, in canonical order.
    pub outputs: Vec<OutputRecord>,
    /// The chain proof ([`crate::chain`]).
    pub proof: StarkProof,
}

#[derive(Debug)]
pub enum BlockError {
    /// A transaction is invalid.
    Tx(usize, TxError),
    /// A transaction doesn't fit the sub-block shape.
    TooLarge(usize),
    /// The lists aren't in canonical order, repeat a commitment, or hold a
    /// commitment that isn't a valid key.
    NotCanonical,
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
            Self::TooLarge(i) => write!(f, "transaction {i} exceeds the sub-block shape"),
            Self::NotCanonical => write!(f, "lists not in canonical order"),
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
    /// The hash a header commits to: both lists (with output nonces) and
    /// the proof's digest. `None` if a commitment isn't a field encoding.
    pub fn body_hash(&self) -> Option<Hash> {
        let mut data = vec![
            BabyBear::new(self.inputs.len() as u32),
            BabyBear::new(self.outputs.len() as u32),
        ];
        for c in &self.inputs {
            data.extend(c.to_field()?);
        }
        for o in &self.outputs {
            data.extend(o.commitment.to_field()?);
            data.extend(output::nonce_limbs(&o.nonce));
        }
        data.extend(self.proof.digest());
        Some(hash::sponge(domain::BLOCK_BODY, &[], &[&data]))
    }
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

/// Bytes per serialized input: its commitment.
pub const INPUT_BYTES: usize = COMMITMENT_LEN;
/// Bytes per serialized output: its commitment and nonce.
pub const OUTPUT_BYTES: usize = COMMITMENT_LEN + NONCE_LEN;

/// The serialized size of a block with these list lengths and proof size.
pub fn encoded_size(inputs: usize, outputs: usize, proof: usize) -> usize {
    crate::header::HEADER_BYTES
        + 12
        + inputs * INPUT_BYTES
        + outputs * OUTPUT_BYTES
        + proof
}

/// Serializes a block with its header: `header | n_in (u32) | inputs |
/// n_out (u32) | outputs (commitment, nonce) | proof length (u32) | proof`,
/// integers little-endian.
pub fn encode(header: &Header, block: &Block) -> Vec<u8> {
    let proof = block.proof.to_bytes();
    let mut out = Vec::with_capacity(encoded_size(
        block.inputs.len(),
        block.outputs.len(),
        proof.len(),
    ));
    out.extend(header.to_bytes());
    out.extend((block.inputs.len() as u32).to_le_bytes());
    for c in &block.inputs {
        out.extend(c.0);
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
    let header =
        Header::from_bytes(take(crate::header::HEADER_BYTES)?).ok_or(DecodeError::Malformed)?;
    let n_in = count(take(4)?);
    let mut inputs = Vec::with_capacity(n_in.min(MAX_BLOCK_SIZE / INPUT_BYTES));
    for _ in 0..n_in {
        inputs.push(Commitment(take(COMMITMENT_LEN)?.try_into().expect("length")));
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
    Ok((
        header,
        Block {
            inputs,
            outputs,
            proof,
        },
    ))
}

/// Applies `block`'s lists to `tree` (whose state in `txn` must be
/// `parent`'s) and checks the result is `header`'s state. On error `txn`
/// may be partly updated; abort it.
pub fn apply(
    tree: &StateTree,
    txn: &mut heed::RwTxn,
    parent: &Header,
    header: &Header,
    block: &Block,
) -> Result<(), BlockError> {
    if tree.state(txn).map_err(BlockError::State)? != parent.state {
        return Err(BlockError::StateMismatch);
    }
    tree.apply_changes(txn, &created(block), &block.inputs)
        .map_err(BlockError::State)?;
    if tree.state(txn).map_err(BlockError::State)? != header.state {
        return Err(BlockError::StateMismatch);
    }
    Ok(())
}

/// A block's transition, proved: its lists and the final aggregation
/// proof, which [`crate::chain`] turns into the block's chain proof.
#[derive(Clone, Debug)]
pub struct Aggregated {
    pub inputs: Vec<Commitment>,
    pub outputs: Vec<OutputRecord>,
    pub node: NodeProof,
}

/// Keys and shapes for proving blocks' transitions.
pub struct BlockProver {
    params: FriParams,
    sub: SubBlockAir,
    sub_pk: ProvingKey,
    seg: StateSegmentAir,
    seg_pk: ProvingKey,
    agg: Aggregator,
}

impl BlockProver {
    /// Sets up proving with sub-blocks of shape `sub`, segments of shape
    /// `seg`, and aggregation nodes of height `2^log_height`.
    pub fn new(params: FriParams, sub: SubBlockAir, seg: StateSegmentAir, log_height: u32) -> Self {
        let (sub_pk, sub_vk) = stark::setup(&params, &sub);
        let (seg_pk, seg_vk) = stark::setup(&params, &seg);
        let agg = Aggregator::new(params, sub, sub_vk, seg, seg_vk, log_height);
        Self {
            params,
            sub,
            sub_pk,
            seg,
            seg_pk,
            agg,
        }
    }

    /// The aggregator, whose merge key verifiers need.
    pub fn aggregator(&self) -> &Aggregator {
        &self.agg
    }

    /// Proves the transition of a block of `txs` on top of `tree`'s state
    /// in `txn`, applying its changes there (commit `txn` to keep them).
    /// The transactions must create exactly `subsidy` in total and the
    /// block must fit [`MAX_BLOCK_SIZE`]; both are checked before proving.
    pub fn prove(
        &self,
        tree: &StateTree,
        txn: &mut heed::RwTxn,
        txs: &[Transaction],
        subsidy: Amount,
    ) -> Result<Aggregated, BlockError> {
        let mut net: i128 = 0;
        for (i, tx) in txs.iter().enumerate() {
            let totals = tx.validate().map_err(|e| BlockError::Tx(i, e))?;
            net -= totals.fee();
            if tx.body.inputs.len() > self.sub.inputs || tx.body.outputs.len() > self.sub.outputs {
                return Err(BlockError::TooLarge(i));
            }
        }
        if net != subsidy as i128 {
            return Err(BlockError::Net(net));
        }
        let n_in: usize = txs.iter().map(|t| t.body.inputs.len()).sum();
        let n_out: usize = txs.iter().map(|t| t.body.outputs.len()).sum();
        let size = encoded_size(n_in, n_out, self.agg.proof_size());
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
            let (take_ins, rest_ins) = ins.split_at(ins.len().min(self.seg.inserts));
            ins = rest_ins;
            let take_sp = if ins.is_empty() {
                let (t, rest) = sps.split_at(sps.len().min(self.seg.spends));
                sps = rest;
                t
            } else {
                &[]
            };
            let inserts: Vec<InsertOp> = take_ins
                .iter()
                .map(|record| {
                    Ok(InsertOp {
                        proof: tree.insert(txn, &record.commitment)?,
                        commitment: record.commitment,
                    })
                })
                .collect::<Result<_, state::Error>>()
                .map_err(BlockError::State)?;
            let spends: Vec<SpendOp> = take_sp
                .iter()
                .map(|c| {
                    Ok(SpendOp {
                        proof: tree.spend(txn, c)?,
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

        // Fix the transactions' blinded hashes, then derive the challenges
        // from both halves.
        let mut rng = Rng::from_os().expect("OS randomness");
        let mut blindings = Vec::with_capacity(txs.len());
        let mut leaves = vec![[BabyBear::ZERO; HASH_LEN]; segments.len()];
        for tx in txs {
            let mut blinding = [BabyBear::ZERO; HASH_LEN];
            rng.fill(&mut blinding);
            let digest = tx.body.digest().ok_or(BlockError::NotCanonical)?;
            leaves.push(subblock_air::tx_hash(&digest, &blinding));
            blindings.push(blinding);
        }
        let tx_hash = merge_tx_hashes(leaves);
        let (lambda, r) = challenges(&prev, &new, &tx_hash);

        // Prove the segments, then the sub-blocks, then merge pairwise,
        // keeping the segments' order.
        let mut nodes: Vec<NodeProof> = Vec::with_capacity(segments.len() + txs.len());
        for (start, inserts, spends) in &segments {
            let (trace, public) = self.seg.trace(start, inserts, spends, lambda, r);
            let proof = stark::prove(&self.params, &self.seg_pk, &self.seg, trace, &public);
            nodes.push(
                self.agg
                    .prove_segment(&public, &proof)
                    .map_err(BlockError::Proof)?,
            );
        }
        for (tx, blinding) in txs.iter().zip(&blindings) {
            let (trace, public) = self.sub.trace(tx, blinding, lambda, r, &mut rng);
            let proof = stark::prove(&self.params, &self.sub_pk, &self.sub, trace, &public);
            nodes.push(
                self.agg
                    .prove_sub_block(&public, &proof)
                    .map_err(BlockError::Proof)?,
            );
        }
        while nodes.len() > 1 {
            let mut next = Vec::with_capacity(nodes.len().div_ceil(2));
            let mut it = nodes.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => next.push(self.agg.merge(&a, &b).map_err(BlockError::Proof)?),
                    None => next.push(a),
                }
            }
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
            inputs,
            outputs,
            node: top,
        })
    }
}

/// The published commitments of a block's outputs, for applying it.
pub fn created(block: &Block) -> Vec<Commitment> {
    block.outputs.iter().map(|o| o.commitment).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::output::{NONCE_LEN, Nonce, Output};
    use crate::transaction::{Input, TxBody, pubkey_hash};
    use crate::wots::{SecretKey, keygen};

    const SUBSIDY: Amount = 50;

    fn key(n: u32) -> (SecretKey, crate::wots::PublicKey) {
        keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 31 + i as u32
        })))
    }

    /// An output paying `amount` to key `n`.
    fn pay(n: u32, amount: Amount) -> Output {
        Output {
            hash: pubkey_hash(&key(n).1),
            amount,
            nonce: [n as u8; NONCE_LEN],
        }
    }

    fn spend(n: u32, amount: Amount, nonce: Nonce) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let commitment = output::commit(&pubkey_hash(&pubkey), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                pubkey,
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
        let prover = BlockProver::new(
            FriParams::default(),
            SubBlockAir {
                inputs: 16,
                outputs: 16,
            },
            StateSegmentAir {
                inserts: 16,
                spends: 16,
            },
            15,
        );
        eprintln!("setup {:.1} s", t.elapsed().as_secs_f64());

        // The chain so far holds one output of 100 to key 1.
        let funding = pay(1, 100);
        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        tree.insert(&mut txn, &funding.commitment().unwrap())
            .unwrap();
        txn.commit().unwrap();

        let (sk, input) = spend(1, 100, funding.nonce);
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
        let agg = prover.prove(&tree, &mut txn, &[tx, reward], SUBSIDY).unwrap();
        let claimed = tree.state(&txn).unwrap();
        txn.abort();
        eprintln!("prove {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!((agg.inputs.len(), agg.outputs.len()), (1, 3));

        // The final statement: what the chain proof checks in-circuit.
        let aggregator = prover.aggregator();
        let s = &agg.node.statement;
        assert_eq!(aggregator.verify(s, &agg.node.proof), Ok(()));
        assert_eq!((s.old_state(), s.new_state()), (prev, claimed));
        assert_eq!((s.lambda(), s.r()), challenges(&prev, &claimed, &s.tx_hash()));
        assert_eq!((s.q_in(), s.q_out()), (BabyBear4::ONE, BabyBear4::ONE));
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
            body: block.body_hash().unwrap(),
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
        assert_eq!(decode(&bytes[..bytes.len() - 1]).err(), Some(DecodeError::Malformed));
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
