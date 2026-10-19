//! Blocks: the published input and output lists, the proof of the state
//! transition, and how full nodes check them.
//!
//! A block publishes its spent input commitments and its new output
//! records, each in canonical order (ascending state-tree key, no
//! duplicates), plus its total fee and mint and one aggregated proof
//! ([`crate::aggregate`]). Transactions stay private.
//!
//! The multiset challenges `lambda, r` hash the previous state and both
//! lists, so they are fixed once the block's contents are. A full node
//! rebuilds the proof's statement from public data alone (the merge key,
//! `lambda, r`, the four products over the lists, the fee and mint, the
//! previous and new state, and the first and last keys) and checks the
//! proof against it. The statement then says: every published output was
//! created by a valid transaction and inserted, every published input was
//! spent by a valid transaction and marked spent, the tree went from the
//! previous state to the new one, and the transactions' fees and mint sum to
//! the published totals. The node also checks `mint <= subsidy + fee`.
//!
//! The header (proof-of-work, links) isn't defined yet; [`verify`] takes the
//! previous and new state directly.

use std::fmt;

use crate::aggregate::{AggError, Aggregator, NodeProof, Statement, statement as st};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::fri::FriParams;
use crate::hash::{self, HASH_LEN, domain};
use crate::output::{self, Amount, Commitment, OutputRecord};
use crate::rng::Rng;
use crate::stark::{self, ProvingKey, StarkProof};
use crate::state::{self, State, StateTree, key_of};
use crate::state_air::{InsertOp, SegmentStart, SpendOp, StateSegmentAir};
use crate::subblock_air::{SubBlockAir, input_element, output_element, product};
use crate::transaction::{Transaction, TxError};

/// What a block publishes.
#[derive(Clone, Debug)]
pub struct Block {
    /// Spent commitments, in canonical order.
    pub inputs: Vec<Commitment>,
    /// New outputs, in canonical order.
    pub outputs: Vec<OutputRecord>,
    /// Total fees paid by the block's transactions.
    pub fee: Amount,
    /// Total minted by its reward transactions.
    pub mint: Amount,
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
    /// Aggregation failed (prover) or the proof doesn't verify (verifier).
    Proof(AggError),
    /// Minted more than the subsidy plus fees.
    Overmint,
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tx(i, e) => write!(f, "transaction {i}: {e:?}"),
            Self::TooLarge(i) => write!(f, "transaction {i} exceeds the sub-block shape"),
            Self::NotCanonical => write!(f, "lists not in canonical order"),
            Self::State(e) => write!(f, "state: {e:?}"),
            Self::Proof(e) => write!(f, "proof: {e}"),
            Self::Overmint => write!(f, "mint exceeds subsidy plus fees"),
        }
    }
}

impl std::error::Error for BlockError {}

/// The canonical order: ascending key, compared on canonical limbs.
fn order_key(c: &Commitment) -> Option<[u32; HASH_LEN]> {
    key_of(c).ok().map(|k| k.map(BabyBear::as_u32))
}

/// Whether `commitments` are valid keys in strictly ascending order.
fn is_canonical<'a>(commitments: impl IntoIterator<Item = &'a Commitment>) -> bool {
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

/// The multiset challenges for a block: a hash of the previous state and
/// both lists (commitments, and outputs' nonces).
pub fn challenges(
    prev: &State,
    inputs: &[Commitment],
    outputs: &[OutputRecord],
) -> Option<(BabyBear4, BabyBear4)> {
    let (lo, hi) = state::split(prev.size);
    let mut data = prev.root.to_vec();
    data.extend([lo, hi]);
    data.extend([inputs.len(), outputs.len()].map(|n| BabyBear::new(n as u32)));
    for c in inputs {
        data.extend(c.to_field()?);
    }
    for o in outputs {
        data.extend(o.commitment.to_field()?);
        data.extend(output::nonce_limbs(&o.nonce));
    }
    let h = hash::sponge(domain::BLOCK_CHALLENGES, &[], &[&data]);
    Some((
        BabyBear4(core::array::from_fn(|b| h[b])),
        BabyBear4(core::array::from_fn(|b| h[4 + b])),
    ))
}

/// The statement a valid proof of this block must prove.
fn expected_statement(
    agg: &Aggregator,
    prev: &State,
    new: &State,
    block: &Block,
) -> Option<Statement> {
    let (lambda, r) = challenges(prev, &block.inputs, &block.outputs)?;
    let p_in = product(
        r,
        block
            .inputs
            .iter()
            .map(|c| input_element(lambda, c))
            .collect::<Option<Vec<_>>>()?,
    );
    let p_out = product(
        r,
        block
            .outputs
            .iter()
            .map(|o| output_element(lambda, o))
            .collect::<Option<Vec<_>>>()?,
    );
    let zero = [BabyBear::ZERO; HASH_LEN];
    let last = |c: Option<&Commitment>| c.map_or(Some(zero), |c| key_of(c).ok());
    let last_insert = last(block.outputs.last().map(|o| &o.commitment))?;
    let last_spend = last(block.inputs.last())?;

    let mut s = vec![BabyBear::ZERO; st::LEN];
    let mut put = |at: usize, xs: &[BabyBear]| s[at..at + xs.len()].copy_from_slice(xs);
    put(st::VK, &agg.merge_key());
    put(st::LAMBDA, &lambda.0);
    put(st::R, &r.0);
    put(st::P_SUB_IN, &p_in.0);
    put(st::P_SP, &p_in.0);
    put(st::P_SUB_OUT, &p_out.0);
    put(st::P_INS, &p_out.0);
    put(st::FEE, &output::amount_limbs(block.fee));
    put(st::MINT, &output::amount_limbs(block.mint));
    put(st::HAS_STATE, &[BabyBear::ONE]);
    for (state, root, size) in [
        (prev, st::OLD_ROOT, st::OLD_SIZE),
        (new, st::NEW_ROOT, st::NEW_SIZE),
    ] {
        let (lo, hi) = state::split(state.size);
        put(root, &state.root);
        put(size, &[lo, hi]);
    }
    put(st::PREV_INS_IN, &zero);
    put(st::PREV_INS_OUT, &last_insert);
    put(st::PREV_SP_IN, &zero);
    put(st::PREV_SP_OUT, &last_spend);
    Some(Statement(s))
}

/// Checks `block` as the transition from `prev` to `new`, with block
/// subsidy `subsidy`.
pub fn verify(
    agg: &Aggregator,
    prev: &State,
    new: &State,
    block: &Block,
    subsidy: Amount,
) -> Result<(), BlockError> {
    if !is_canonical(&block.inputs) || !is_canonical(block.outputs.iter().map(|o| &o.commitment)) {
        return Err(BlockError::NotCanonical);
    }
    if block.mint as u128 > subsidy as u128 + block.fee as u128 {
        return Err(BlockError::Overmint);
    }
    let statement = expected_statement(agg, prev, new, block).ok_or(BlockError::NotCanonical)?;
    agg.verify(&statement, &block.proof)
        .map_err(BlockError::Proof)
}

/// Keys and shapes for proving blocks.
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

    /// Proves a block of `txs` on top of `tree`'s state in `txn`, applying
    /// its changes there (commit `txn` to keep them).
    pub fn prove(
        &self,
        tree: &StateTree,
        txn: &mut heed::RwTxn,
        txs: &[Transaction],
    ) -> Result<Block, BlockError> {
        for (i, tx) in txs.iter().enumerate() {
            tx.validate().map_err(|e| BlockError::Tx(i, e))?;
            if tx.body.inputs.len() > self.sub.inputs || tx.body.outputs.len() > self.sub.outputs {
                return Err(BlockError::TooLarge(i));
            }
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
        let (lambda, r) = challenges(&prev, &inputs, &outputs).ok_or(BlockError::NotCanonical)?;

        // State segments, in order: all inserts, then all spends.
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
                        record: *record,
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
            let (trace, public) = self.seg.trace(&start, &inserts, &spends, lambda, r);
            let proof = stark::prove(&self.params, &self.seg_pk, &self.seg, trace, &public);
            segments.push(
                self.agg
                    .prove_segment(&public, &proof)
                    .map_err(BlockError::Proof)?,
            );
            start = SegmentStart {
                state: tree.state(txn).map_err(BlockError::State)?,
                prev_insert: take_ins.last().map_or(start.prev_insert, |o| {
                    key_of(&o.commitment).expect("checked")
                }),
                prev_spend: take_sp
                    .last()
                    .map_or(start.prev_spend, |c| key_of(c).expect("checked")),
            };
        }

        // Sub-blocks, then merge pairwise, keeping the segments' order.
        let mut rng = Rng::from_os().expect("OS randomness");
        let mut nodes: Vec<NodeProof> = segments;
        for tx in txs {
            let (trace, public) = self.sub.trace(tx, lambda, r, &mut rng);
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
        Ok(Block {
            inputs,
            outputs,
            fee: top.statement.fee(),
            mint: top.statement.mint(),
            proof: top.proof,
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
    /// reward claiming the subsidy plus the fee, proved, checked as a full
    /// node would, and applied natively.
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

        // The miner proves the block (and drops its changes).
        let t = std::time::Instant::now();
        let mut txn = env.env().write_txn().unwrap();
        let prev = tree.state(&txn).unwrap();
        let block = prover.prove(&tree, &mut txn, &[tx, reward]).unwrap();
        let claimed = tree.state(&txn).unwrap();
        txn.abort();
        eprintln!("prove {:.1} s", t.elapsed().as_secs_f64());
        assert_eq!((block.inputs.len(), block.outputs.len()), (1, 3));
        assert_eq!((block.fee, block.mint), (10, SUBSIDY + 10));

        // A full node checks it against its own state, then applies it.
        let agg = prover.aggregator();
        let t = std::time::Instant::now();
        assert!(verify(agg, &prev, &claimed, &block, SUBSIDY).is_ok());
        eprintln!("verify {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let mut txn = env.env().write_txn().unwrap();
        tree.apply_changes(&mut txn, &created(&block), &block.inputs)
            .unwrap();
        assert_eq!(tree.state(&txn).unwrap(), claimed);
        txn.commit().unwrap();

        // Rejections.
        let rejects = |b: &Block, prev: &State, new: &State, subsidy| {
            verify(agg, prev, new, b, subsidy).is_err()
        };
        assert!(rejects(&block, &prev, &claimed, SUBSIDY - 1), "overmint");
        assert!(
            rejects(&block, &claimed, &claimed, SUBSIDY),
            "wrong previous state"
        );
        assert!(rejects(&block, &prev, &prev, SUBSIDY), "wrong new state");
        let mut b = block.clone();
        b.outputs.pop();
        assert!(rejects(&b, &prev, &claimed, SUBSIDY), "dropped output");
        let mut b = block.clone();
        b.inputs.clear();
        assert!(rejects(&b, &prev, &claimed, SUBSIDY), "dropped input");
        let mut b = block.clone();
        b.outputs[0].nonce[0] ^= 1;
        assert!(rejects(&b, &prev, &claimed, SUBSIDY), "changed nonce");
        let mut b = block.clone();
        b.outputs.swap(0, 1);
        assert!(rejects(&b, &prev, &claimed, SUBSIDY), "not canonical");
        let mut b = block.clone();
        (b.fee, b.mint) = (20, SUBSIDY + 20);
        assert!(
            rejects(&b, &prev, &claimed, SUBSIDY),
            "inflated fee and mint"
        );
    }
}
