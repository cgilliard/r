//! Chain proofs: one proof per block that covers the whole chain up to it.
//!
//! Block `N` carries a chain proof whose statement ([`statement`]) says:
//! there is a valid chain from genesis whose newest header has hash `TIP`
//! (header `N - 1`, the block's parent), with total work `WORK` over headers
//! `1 ..= N - 1`, and a valid transition from header `N - 1`'s state to the
//! state `ROOT, SIZE` (block `N`'s). Header `N` commits to the proof through
//! its body hash, so a header's PoW is proven one block later, by its
//! child's chain proof.
//!
//! Three recursion circuits:
//!
//! - **first** (block 1): verifies block 1's aggregation proof and checks
//!   its transition starts at genesis, which is built in; tip = genesis,
//!   work 0.
//! - **chain** (block `N >= 2`): verifies block `N - 1`'s published (wrap)
//!   proof; checks header `N - 1` against header `N - 2`
//!   ([`pow_circuit::check_header`]: link, time, ASERT, PoW), that it
//!   extends the previous proof's tip and has its state; verifies block
//!   `N`'s aggregation proof as a transition from that state; adds header
//!   `N - 1`'s work. Its own key is part of the statement, since it can't
//!   be a constant of its own circuit.
//! - **wrap**: verifies a first or chain proof (keyed by the first key or
//!   the statement's key) and restates its statement, in a smaller trace
//!   with a larger blowup ([`wrap_params`]), so it needs fewer queries:
//!   the published proof is about two thirds the size. Every block publishes a
//!   wrap proof, and the next block's chain proof verifies that, so anyone
//!   with the chain can extend it.
//!
//! The first and chain circuits run at the aggregation height with
//! its parameters.
//!
//! For each aggregation proof the circuits check what a full node can't see
//! without the transactions: it names the merge key; its state hashes open
//! (with the new root, size and last keys as witness) to a change starting
//! at the old state with an empty key chain; its item stream starts and
//! ends between transactions; its challenges are derived from its old and
//! new state and its transaction hash ([`block::challenges`]); its three
//! fingerprint ratios are 1; and its net value is the subsidy for the
//! block's height ([`Consensus::subsidy`], which also ends the chain at its
//! final height).
//!
//! A node fast-syncing from block `N` needs only headers `N - 1` and `N` and
//! block `N`'s proof: [`Chain::verify_proof`] with header `N`'s `prev` and
//! state and the claimed work, plus header `N`'s own PoW natively; then the
//! state tree from peers, checked against header `N`'s root.

use crate::aggregate::{self, AggError, Aggregator, Node, inputs, publish, statement as ast};
use crate::babybear::BabyBear;
use crate::block::{self, Aggregated, Block, BlockError, BlockProver, MAX_BLOCK_SIZE};
use crate::circuit::{Cell, Circuit};
use crate::fri::FriParams;
use crate::hash::{HASH_LEN, Hash, domain};
use crate::header::{self, Consensus, Header, WORK_LIMBS};
use crate::item_chunk_air::Boundary;
use crate::pow::DatasetTree;
use crate::output;
use crate::pow_circuit::{self, HeaderCells};
use crate::recursion_air::RecursionAir;
use crate::stark::{self, KeyCache, StarkError, StarkProof};
use crate::state::{self, StateTree, Undo};
use crate::symbolic::{Program, compile};
use crate::verifier_circuit::{self, Inner, ProofCells};

/// Offsets in a chain proof's statement.
pub mod statement {
    use crate::hash::HASH_LEN;
    use crate::header::WORK_LIMBS;

    /// The chain circuit's key.
    pub const VK: usize = 0;
    /// The hash of the newest header covered (the block's parent).
    pub const TIP: usize = VK + HASH_LEN;
    /// The state after the block: root, then size as two limbs.
    pub const ROOT: usize = TIP + HASH_LEN;
    pub const SIZE: usize = ROOT + HASH_LEN;
    /// Total work of headers `1 ..= tip`, as [`crate::header::work_limbs`].
    pub const WORK: usize = SIZE + 2;
    pub const LEN: usize = WORK + WORK_LIMBS;
}

use statement as st;

/// The wrap circuit's height.
pub const WRAP_LOG_HEIGHT: u32 = 14;

/// The wrap proof's parameters: blowup 32 with 16 queries and 20 grinding
/// bits: the default's ~100 bits of conjectured security with 16 queries
/// instead of 28, so a published proof of about 163 KB. (Blowup 64 with 14
/// queries gave 150 KB but took twice as long to prove.)
pub fn wrap_params() -> FriParams {
    FriParams {
        log_blowup: 5,
        num_queries: 16,
        grinding_bits: 20,
        ..FriParams::default()
    }
}

/// A block's published (wrap) proof and its statement.
#[derive(Clone, Debug)]
pub struct ChainProof {
    pub statement: Vec<BabyBear>,
    pub proof: StarkProof,
}

/// Keys and circuits for proving and verifying chain proofs.
pub struct Chain {
    consensus: Consensus,
    genesis: Header,
    params: FriParams,
    log_height: u32,
    blocks: BlockProver,
    /// The PoW dataset's tree, for opening items in header checks (blank
    /// on a node that only verifies).
    tree: DatasetTree,
    first: Node,
    chain: Node,
    wrap: Node,
    /// Constraints of first and chain proofs (they share a shape) and of
    /// wrap proofs.
    chain_program: Program,
    wrap_program: Program,
}

fn consts(c: &mut Circuit, values: &[BabyBear]) -> Vec<Cell> {
    values.iter().map(|&v| c.constant(v)).collect()
}

fn assert_eq_all(c: &mut Circuit, a: &[Cell], b: &[Cell]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        c.assert_eq(x, y);
    }
}

/// The net value block `parent_height + 1` must create
/// ([`Consensus::subsidy`]), as [`output::net_limbs`] cells; fails if that
/// height is past the final one.
///
/// # Panics
/// If the final height isn't below `2^30` (the comparisons assume so).
fn subsidy_net(c: &mut Circuit, consensus: &Consensus, parent_height: &[Cell; 4]) -> Vec<Cell> {
    let last = consensus.final_height();
    assert!(last < 1 << 30, "final height too large");
    // Heights stay below 2^30, so the top limbs are zero and the height is
    // one field element.
    c.assert_zero(parent_height[2]);
    c.assert_zero(parent_height[3]);
    let hi = c.mul_const(parent_height[1], BabyBear::new(1 << 16));
    let h = c.add(parent_height[0], hi);
    let n = c.add_const(h, BabyBear::ONE);
    let n_value = c.value(n).as_u32() as u64;
    // n <= last.
    let k = c.constant(BabyBear::new(last as u32));
    let left = c.sub(k, n);
    c.bits(left, 30);
    // first = [n <= first_era], with n on the claimed side of the boundary.
    let first = c.hint(BabyBear::from(n_value <= consensus.first_era));
    let ff = c.mul(first, first);
    c.assert_eq(ff, first);
    let era = c.constant(BabyBear::new(consensus.first_era as u32));
    let below = c.sub(era, n);
    let above = c.sub(n, era);
    let above = c.add_const(above, -BabyBear::ONE);
    let d = c.sub(below, above);
    let picked = c.mul(first, d);
    let gap = c.add(above, picked);
    c.bits(gap, 30);
    // The era's net value.
    let a = output::net_limbs(consensus.first_subsidy as i128);
    let b = output::net_limbs(consensus.second_subsidy as i128);
    a.iter()
        .zip(&b)
        .map(|(&x, &y)| {
            let base = c.constant(y);
            let t = c.mul_const(first, x - y);
            c.add(base, t)
        })
        .collect()
}

/// Elements of a transition's witness: the new root and size and the
/// last keys inserted and spent, which with the old state open the final
/// aggregation statement's state hashes.
const WITNESS_LEN: usize = 3 * HASH_LEN + 2;

/// A transition's witness ([`WITNESS_LEN`]).
fn witness(agg: &Aggregated) -> Vec<BabyBear> {
    let (lo, hi) = state::split(agg.new.size);
    let mut w = agg.new.root.to_vec();
    w.extend([lo, hi]);
    w.extend(agg.last_insert);
    w.extend(agg.last_spend);
    w
}

/// The checks on a final aggregation statement `s` that its proof can't
/// make itself, for a transition starting at `old_root`, `old_size` and
/// creating `net` ([`output::net_limbs`] cells), with witness `w`
/// ([`WITNESS_LEN`] cells). Returns the new root and size cells.
fn check_transition(
    c: &mut Circuit,
    merge_key: &Hash,
    s: &[Cell],
    w: &[Cell],
    old_root: &[Cell],
    old_size: &[Cell],
    net: &[Cell],
) -> Vec<Cell> {
    let key = consts(c, merge_key);
    assert_eq_all(c, &s[ast::VK..ast::VK + HASH_LEN], &key);
    let new_state = &w[..HASH_LEN + 2];

    // The state goes from the old one, with segments starting from an
    // empty key chain, to the witness's.
    let zeros = consts(c, &[BabyBear::ZERO; 2 * HASH_LEN]);
    let mut before = old_root.to_vec();
    before.extend_from_slice(old_size);
    before.extend_from_slice(&zeros);
    let h = aggregate::state_hash_cells(c, &before);
    assert_eq_all(c, &s[ast::STATE_IN..ast::STATE_IN + HASH_LEN], &h);
    let h = aggregate::state_hash_cells(c, w);
    assert_eq_all(c, &s[ast::STATE_OUT..ast::STATE_OUT + HASH_LEN], &h);

    // The item stream starts and ends idle: every transaction is whole.
    let idle = consts(c, &aggregate::stream_hash(&Boundary::IDLE));
    assert_eq_all(c, &s[ast::STREAM_IN..ast::STREAM_IN + HASH_LEN], &idle);
    assert_eq_all(c, &s[ast::STREAM_OUT..ast::STREAM_OUT + HASH_LEN], &idle);
    let one = c.one();
    c.assert_eq(s[ast::HAS_STATE], one);
    c.assert_eq(s[ast::HAS_STREAM], one);

    // lambda, r = H(old state, new state, transaction hash).
    let mut elems = Vec::with_capacity(28);
    elems.extend_from_slice(old_root);
    elems.extend_from_slice(old_size);
    elems.extend_from_slice(new_state);
    elems.extend_from_slice(&s[ast::TX_HASH..ast::TX_HASH + HASH_LEN]);
    let h = verifier_circuit::sponge(c, domain::BLOCK_CHALLENGES, &elems, None);
    assert_eq_all(c, &s[ast::LAMBDA..ast::LAMBDA + 4], &h[..4]);
    assert_eq_all(c, &s[ast::R..ast::R + 4], &h[4..8]);

    let one_ext = consts(c, &crate::babybear4::BabyBear4::ONE.0);
    for at in [ast::Q_IN, ast::Q_OUT, ast::Q_SIG] {
        assert_eq_all(c, &s[at..at + 4], &one_ext);
    }
    assert_eq_all(c, &s[ast::NET..ast::NET + output::NET_LIMBS], net);
    new_state.to_vec()
}

/// Loads an aggregation statement and proof and verifies it.
fn load_aggregate(
    c: &mut Circuit,
    agg: &Aggregator,
    statement: &[BabyBear],
    proof: &StarkProof,
) -> Vec<Cell> {
    let s = inputs(c, statement);
    let cells = ProofCells::load(c, proof);
    verifier_circuit::verify(c, &agg.merge_inner(), &s, &cells);
    s
}

fn first_circuit(
    consensus: &Consensus,
    genesis: &Header,
    agg: &Aggregator,
    vk: &Hash,
    statement: &[BabyBear],
    proof: &StarkProof,
    witness: &[BabyBear],
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let vk = inputs(&mut c, vk);
    let s = load_aggregate(&mut c, agg, statement, proof);
    let w = inputs(&mut c, witness);
    let (lo, hi) = state::split(genesis.state.size);
    let old_root = consts(&mut c, &genesis.state.root);
    let old_size = consts(&mut c, &[lo, hi]);
    let subsidy = consensus.subsidy(1).expect("block 1 is valid");
    let net = consts(&mut c, &output::net_limbs(subsidy as i128));
    let new = check_transition(&mut c, &agg.merge_key(), &s, &w, &old_root, &old_size, &net);
    let mut out = vk;
    out.extend(consts(&mut c, &genesis.hash()));
    out.extend(new);
    out.extend(consts(&mut c, &header::work_limbs(0)));
    let public = publish(&mut c, &out);
    (c, public)
}

/// Restates a first or chain proof's statement, verifying the proof under
/// the first circuit's key (`is_first`) or the statement's own key.
fn wrap_circuit(
    inner: &Inner<'_, RecursionAir>,
    first_root: &Hash,
    statement: &[BabyBear],
    proof: &StarkProof,
    is_first: bool,
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let s = inputs(&mut c, statement);
    let first = c.hint(BabyBear::from(is_first));
    let ff = c.mul(first, first);
    c.assert_eq(ff, first);
    let root: [Cell; HASH_LEN] = core::array::from_fn(|j| {
        let k = c.constant(first_root[j]);
        let d = c.sub(k, s[st::VK + j]);
        let t = c.mul(first, d);
        c.add(s[st::VK + j], t)
    });
    let cells = ProofCells::load(&mut c, proof);
    verifier_circuit::verify_keyed(&mut c, inner, Some(root), &s, &cells);
    let public = publish(&mut c, &s);
    (c, public)
}

/// Extends block `N - 1`'s wrap proof (`prev`) to block `N`: checks header
/// `N - 1` (`header`) against header `N - 2` (`parent`) and that it's the
/// previous proof's tip state, verifies block `N`'s aggregation proof as a
/// transition from header `N - 1`'s state, and adds header `N - 1`'s work.
#[allow(clippy::too_many_arguments)]
fn chain_circuit(
    consensus: &Consensus,
    tree: &DatasetTree,
    agg: &Aggregator,
    wrap: &Inner<'_, RecursionAir>,
    vk: &Hash,
    prev: &ChainProof,
    parent: &Header,
    header: &Header,
    statement: &[BabyBear],
    proof: &StarkProof,
    witness: &[BabyBear],
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let vk = inputs(&mut c, vk);

    let ps = inputs(&mut c, &prev.statement);
    let cells = ProofCells::load(&mut c, &prev.proof);
    verifier_circuit::verify(&mut c, wrap, &ps, &cells);
    assert_eq_all(&mut c, &ps[st::VK..st::VK + HASH_LEN], &vk);

    // Header N - 1 extends the previous proof's tip and has its state.
    let p = HeaderCells::input(&mut c, parent);
    let h = HeaderCells::input(&mut c, header);
    let check = pow_circuit::check_header(&mut c, consensus, tree, &p, &h);
    assert_eq_all(&mut c, &check.parent_hash, &ps[st::TIP..st::TIP + HASH_LEN]);
    assert_eq_all(&mut c, &h.root, &ps[st::ROOT..st::ROOT + HASH_LEN]);
    assert_eq_all(&mut c, &h.size, &ps[st::SIZE..st::SIZE + 2]);

    // Block N's transition from there.
    let s = load_aggregate(&mut c, agg, statement, proof);
    let w = inputs(&mut c, witness);
    let net = subsidy_net(&mut c, consensus, &h.height);
    let new = check_transition(&mut c, &agg.merge_key(), &s, &w, &h.root, &h.size, &net);

    let work = pow_circuit::add_work(&mut c, &ps[st::WORK..st::WORK + WORK_LIMBS], &check.work);
    let mut out = vk;
    out.extend(check.hash);
    out.extend(new);
    out.extend(work);
    let public = publish(&mut c, &out);
    (c, public)
}

impl Chain {
    /// Sets up chain proofs for the chain starting at `genesis` under
    /// `consensus`, proving transitions with `blocks` and header checks
    /// with the dataset's `tree` ([`crate::pow::Dataset::tree`]; a
    /// [`DatasetTree::Blank`] chain verifies but can't prove).
    ///
    /// # Panics
    /// If a circuit doesn't fit the aggregation height.
    pub fn new(
        consensus: Consensus,
        genesis: Header,
        blocks: BlockProver,
        tree: DatasetTree,
        keys: &KeyCache,
    ) -> Self {
        let agg = blocks.aggregator();
        let params = *agg.params();
        let log_height = agg.log_height();
        let merge_air = agg.merge_inner().air.clone();
        let agg_statement = vec![BabyBear::ZERO; ast::LEN];
        let agg_proof = StarkProof::placeholder(&params, &merge_air, log_height);
        let zero_vk = [BabyBear::ZERO; HASH_LEN];
        let w = [BabyBear::ZERO; WITNESS_LEN];

        let (c, public) = first_circuit(
            &consensus,
            &genesis,
            agg,
            &zero_vk,
            &agg_statement,
            &agg_proof,
            &w,
        );
        let first = Node::new(&params, &c, &public, log_height, keys);
        let chain_program = compile(&first.air, st::LEN);

        // The wrap circuit verifies first and chain proofs, which share a
        // shape: lay it out with the first circuit standing in.
        let wrap_params = wrap_params();
        let inner = ChainProof {
            statement: vec![BabyBear::ZERO; st::LEN],
            proof: StarkProof::placeholder(&params, &first.air, log_height),
        };
        let (c, public) = wrap_circuit(
            &Inner {
                params: &params,
                vk: &first.vk,
                air: &first.air,
                program: &chain_program,
                log_n: log_height,
            },
            &first.root(),
            &inner.statement,
            &inner.proof,
            false,
        );
        let wrap = Node::new(&wrap_params, &c, &public, WRAP_LOG_HEIGHT, keys);
        let wrap_program = compile(&wrap.air, st::LEN);

        let prev = ChainProof {
            statement: vec![BabyBear::ZERO; st::LEN],
            proof: StarkProof::placeholder(&wrap_params, &wrap.air, WRAP_LOG_HEIGHT),
        };
        let default = Header::default();
        let (c, public) = chain_circuit(
            &consensus,
            &tree,
            agg,
            &Inner {
                params: &wrap_params,
                vk: &wrap.vk,
                air: &wrap.air,
                program: &wrap_program,
                log_n: WRAP_LOG_HEIGHT,
            },
            &zero_vk,
            &prev,
            &default,
            &default,
            &agg_statement,
            &agg_proof,
            &w,
        );
        let chain = Node::new(&params, &c, &public, log_height, keys);
        Self {
            consensus,
            genesis,
            params,
            log_height,
            blocks,
            tree,
            first,
            chain,
            wrap,
            chain_program,
            wrap_program,
        }
    }

    pub fn consensus(&self) -> &Consensus {
        &self.consensus
    }

    pub fn genesis(&self) -> &Header {
        &self.genesis
    }

    pub fn block_prover(&self) -> &BlockProver {
        &self.blocks
    }

    /// The key every chain statement names.
    pub fn key(&self) -> Hash {
        self.chain.root()
    }

    /// Rows used by the first and chain circuits (of `2^log_height`) and
    /// the wrap circuit (of `2^WRAP_LOG_HEIGHT`).
    pub fn used_rows(&self) -> ([usize; 3], u32) {
        (
            [&self.first, &self.chain, &self.wrap].map(|n| n.air.used_rows()),
            self.log_height,
        )
    }

    /// Wraps a first (`is_first`) or chain proof.
    fn wrap(
        &self,
        statement: Vec<BabyBear>,
        proof: StarkProof,
        is_first: bool,
    ) -> Result<ChainProof, AggError> {
        let inner = Inner {
            params: &self.params,
            vk: &self.chain.vk,
            air: &self.chain.air,
            program: &self.chain_program,
            log_n: self.log_height,
        };
        let (c, public) = wrap_circuit(&inner, &self.first.root(), &statement, &proof, is_first);
        let (statement, proof) = self
            .wrap
            .prove(&wrap_params(), &c, &public)
            .ok_or(AggError::Invalid)?;
        Ok(ChainProof { statement, proof })
    }

    /// Block 1's chain proof.
    pub fn prove_first(&self, agg: &Aggregated) -> Result<ChainProof, AggError> {
        let (c, public) = first_circuit(
            &self.consensus,
            &self.genesis,
            self.blocks.aggregator(),
            &self.key(),
            &agg.node.statement.0,
            &agg.node.proof,
            &witness(agg),
        );
        let (statement, proof) = self
            .first
            .prove(&self.params, &c, &public)
            .ok_or(AggError::Invalid)?;
        self.wrap(statement, proof, true)
    }

    /// Block `N`'s chain proof (`N >= 2`), given block `N - 1`'s published
    /// proof `prev`, headers `N - 2` (`grandparent`) and `N - 1`
    /// (`parent`), and block `N`'s transition.
    pub fn prove_next(
        &self,
        prev: &ChainProof,
        grandparent: &Header,
        parent: &Header,
        agg: &Aggregated,
    ) -> Result<ChainProof, AggError> {
        let wrap_params = wrap_params();
        let (c, public) = chain_circuit(
            &self.consensus,
            &self.tree,
            self.blocks.aggregator(),
            &Inner {
                params: &wrap_params,
                vk: &self.wrap.vk,
                air: &self.wrap.air,
                program: &self.wrap_program,
                log_n: WRAP_LOG_HEIGHT,
            },
            &self.key(),
            prev,
            grandparent,
            parent,
            &agg.node.statement.0,
            &agg.node.proof,
            &witness(agg),
        );
        let (statement, proof) = self
            .chain
            .prove(&self.params, &c, &public)
            .ok_or(AggError::Invalid)?;
        self.wrap(statement, proof, false)
    }

    /// The statement block `header`'s chain proof must prove, given the
    /// total work of headers `1 ..= header.height - 1`.
    pub fn statement(&self, header: &Header, work_before: u128) -> Vec<BabyBear> {
        let (lo, hi) = state::split(header.state.size);
        let mut s = self.key().to_vec();
        s.extend(header.prev);
        s.extend(header.state.root);
        s.extend([lo, hi]);
        s.extend(header::work_limbs(work_before));
        s
    }

    /// Checks a chain proof for block `header`, given the total work of
    /// headers `1 ..= header.height - 1`. Doesn't check `header` itself.
    pub fn verify_proof(
        &self,
        header: &Header,
        work_before: u128,
        proof: &StarkProof,
    ) -> Result<(), StarkError> {
        if header.height == 0 {
            return Err(StarkError::Shape);
        }
        let statement = self.statement(header, work_before);
        stark::verify(
            &wrap_params(),
            &self.wrap.vk,
            &self.wrap.air,
            &statement,
            proof,
        )
    }

    /// Checks `block` as the body of `header`: it fits [`MAX_BLOCK_SIZE`],
    /// its lists are canonical, no input is in a slot the block fills (no
    /// input is created in the same block), the header commits to it, and
    /// its chain
    /// proof holds. Header rules ([`header::check`]) and applying the lists
    /// ([`block::apply`]) are separate.
    pub fn verify_block(
        &self,
        header: &Header,
        block: &Block,
        work_before: u128,
    ) -> Result<(), BlockError> {
        let size = block::encoded_size(block.inputs.len(), block.outputs.len(), block.proof.size());
        if size > MAX_BLOCK_SIZE {
            return Err(BlockError::Oversize(size));
        }
        if !block::positions_canonical(&block.inputs)
            || !block::is_canonical(block.outputs.iter().map(|o| &o.commitment))
        {
            return Err(BlockError::NotCanonical);
        }
        // The slot counter before the block: each output fills one slot.
        let prev_size = header
            .state
            .size
            .checked_sub(block.outputs.len() as u64)
            .ok_or(BlockError::StateMismatch)?;
        if block::spends_new_slot(&block.inputs, prev_size) {
            return Err(BlockError::SpendsItsOwnOutput);
        }
        if block.body_hash() != header.body {
            return Err(BlockError::Body);
        }
        self.verify_proof(header, work_before, &block.proof)
            .map_err(BlockError::Chain)
    }

    /// A full node's check of a received block: [`Chain::verify_block`],
    /// then [`block::apply`] to `tree` (at `parent`'s state in `txn`).
    pub fn check_and_apply(
        &self,
        tree: &StateTree,
        txn: &mut heed::RwTxn,
        parent: &Header,
        header: &Header,
        block: &Block,
        work_before: u128,
    ) -> Result<Undo, BlockError> {
        self.verify_block(header, block, work_before)?;
        block::apply(tree, txn, parent, header, block)
    }
}

/// The total work in a chain statement.
pub fn statement_work(statement: &[BabyBear]) -> Option<u128> {
    header::work_value(&statement[st::WORK..st::WORK + WORK_LIMBS])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::header::FRAC_BITS;
    use crate::output::Output;
    use crate::pow::Dataset;
    use crate::transaction::{Transaction, TxBody};
    use crate::wallet::{Coin, Wallet};

    /// A transaction spending `coins` into `outputs`.
    fn spend(wallet: &Wallet, coins: &[Coin], outputs: Vec<Output>) -> Transaction {
        let (keys, inputs) = coins.iter().map(|c| wallet.input(c, 0)).unzip();
        Transaction::sign(TxBody { inputs, outputs }, keys).unwrap()
    }

    /// Three blocks with real spends, each with its chain proof, mined at an
    /// easy target, checked as a full node and as a fast-syncing node; then
    /// mismatched proofs and statements.
    #[test]
    #[ignore = "slow: run in release"]
    fn proves_and_verifies_a_chain() {
        let data = Dataset::get();
        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let consensus = Consensus {
            genesis_time: 1_000_000,
            // About 2^8 expected hashes.
            genesis_log_target: (100 << FRAC_BITS) + 1000,
            // Two blocks of 50, then one of 7: the chain ends at height 3.
            first_era: 2,
            first_subsidy: 50,
            second_era: 1,
            second_subsidy: 7,
            ..Consensus::DEFAULT
        };
        let genesis = Header {
            timestamp: consensus.genesis_time,
            state: tree.state(&env.env().read_txn().unwrap()).unwrap(),
            ..Header::default()
        };
        let t = std::time::Instant::now();
        let blocks = BlockProver::standard(FriParams::default(), &KeyCache::none());
        let chain = Chain::new(
            consensus,
            genesis,
            blocks,
            DatasetTree::Full(data.tree()),
            &KeyCache::none(),
        );
        // A node that doesn't mine sets up without the dataset, and gets the
        // same keys.
        let verifier = Chain::new(
            consensus,
            genesis,
            BlockProver::standard(FriParams::default(), &KeyCache::none()),
            DatasetTree::Blank,
            &KeyCache::none(),
        );
        eprintln!(
            "setup {:.1} s, rows {:?}",
            t.elapsed().as_secs_f64(),
            chain.used_rows()
        );

        let (early, late) = (consensus.first_subsidy, consensus.second_subsidy);
        let wallet = Wallet::from_seed(BabyBear::new_array([9, 8, 7, 6, 5, 4, 3, 2]));
        let pay = |i: u8, amount| wallet.keys_for(&[i; 16]).pay(amount);
        let coins = |tree: &StateTree| {
            let outputs = tree.outputs(&env.env().read_txn().unwrap()).unwrap();
            wallet.scan(&outputs)
        };

        let mut headers = vec![genesis];
        let mut work = vec![0u128];
        let mut proofs: Vec<ChainProof> = Vec::new();
        let mut bodies: Vec<Block> = Vec::new();
        for height in 1..=3u64 {
            // Spend everything the wallet holds (found in the state, with
            // positions) and claim the reward.
            let held = coins(&tree);
            let subsidy = consensus.subsidy(height).unwrap();
            let txs = match height {
                1 => vec![spend(&wallet, &[], vec![pay(0, early)])],
                2 => vec![
                    spend(&wallet, &held, vec![pay(1, 20), pay(2, 25)]),
                    spend(&wallet, &[], vec![pay(3, early + 5)]),
                ],
                // The second era.
                _ => vec![
                    spend(&wallet, &held, vec![pay(4, 100)]),
                    spend(&wallet, &[], vec![pay(5, late)]),
                ],
            };
            let txs = &txs;
            let parent = *headers.last().unwrap();
            let t = std::time::Instant::now();
            let mut txn = env.env().write_txn().unwrap();
            let agg = chain
                .block_prover()
                .prove(&tree, &mut txn, txs, subsidy)
                .unwrap();
            let state = tree.state(&txn).unwrap();
            txn.abort();
            let proof = match proofs.last() {
                None => chain.prove_first(&agg).unwrap(),
                Some(prev) => chain
                    .prove_next(prev, &headers[headers.len() - 2], &parent, &agg)
                    .unwrap(),
            };
            eprintln!(
                "block {}: proved in {:.1} s",
                parent.height + 1,
                t.elapsed().as_secs_f64()
            );
            let block = Block {
                inputs: agg.inputs,
                outputs: agg.outputs,
                proof: proof.proof.clone(),
            };
            let mut header = Header {
                height: parent.height + 1,
                prev: parent.hash(),
                timestamp: parent.timestamp + 600,
                state,
                body: block.body_hash(),
                nonce: 0,
            };
            assert!(header.mine(data, consensus.target(&parent)));

            // A full node.
            let work_before = *work.last().unwrap();
            assert_eq!(statement_work(&proof.statement), Some(work_before));
            assert_eq!(header::check(&consensus, &parent, &header), Ok(()));
            let mut txn = env.env().write_txn().unwrap();
            chain
                .check_and_apply(&tree, &mut txn, &parent, &header, &block, work_before)
                .unwrap();
            txn.commit().unwrap();
            verifier.verify_block(&header, &block, work_before).unwrap();

            work.push(work_before + header::work(consensus.log_target(&parent)));
            headers.push(header);
            proofs.push(proof);
            bodies.push(block);
        }

        // The wallet's coins are recovered from the state with amounts:
        // block 3's two outputs.
        let mut held: Vec<(u8, u64)> = coins(&tree).iter().map(|c| (c.r[0], c.amount)).collect();
        held.sort();
        assert_eq!(held, [(4, 100), (5, late)]);

        // Fast sync from the tip: two headers and one proof.
        let tip = headers[3];
        assert_eq!(chain.verify_proof(&tip, work[2], &proofs[2].proof), Ok(()));
        assert!(work[2] > 0);

        // Mismatches.
        let rejects = |h: &Header, w: u128, p: &StarkProof| chain.verify_proof(h, w, p).is_err();
        assert!(rejects(&tip, work[2] + 1, &proofs[2].proof), "wrong work");
        assert!(
            rejects(&tip, work[1], &proofs[2].proof),
            "work one block short"
        );
        let other_state = Header {
            state: headers[2].state,
            ..tip
        };
        assert!(
            rejects(&other_state, work[2], &proofs[2].proof),
            "wrong state"
        );
        let other_parent = Header {
            prev: headers[1].hash(),
            ..tip
        };
        assert!(
            rejects(&other_parent, work[2], &proofs[2].proof),
            "wrong parent"
        );
        assert!(
            rejects(&tip, work[2], &proofs[1].proof),
            "previous block's proof"
        );
        assert!(
            rejects(&headers[1], work[0], &proofs[2].proof),
            "tip's proof for block 1"
        );
        // Every published proof is a wrap proof, about two thirds the size
        // of the inner proofs.
        for p in &proofs {
            assert_eq!(p.proof.size(), proofs[0].proof.size());
        }
        eprintln!("published proof {} bytes", proofs[0].proof.size());
        // A full node rejects a body whose header doesn't commit to it.
        let mut b = bodies[2].clone();
        b.outputs[0].nonce[0] ^= 1;
        assert!(matches!(
            chain.verify_block(&tip, &b, work[2]),
            Err(BlockError::Body)
        ));

        // The prover can't extend from a header that doesn't match the
        // transition: a transition on top of header 3, claimed for header 2.
        let mut txn = env.env().write_txn().unwrap();
        let agg = chain
            .block_prover()
            .prove(
                &tree,
                &mut txn,
                &[spend(&wallet, &[], vec![pay(6, late)])],
                late,
            )
            .unwrap();
        txn.abort();
        assert_eq!(
            chain
                .prove_next(&proofs[2], &headers[1], &headers[2], &agg)
                .err(),
            Some(AggError::Invalid),
            "transition doesn't start at the parent's state"
        );
        // And the chain has ended: no block 4, in the proof or natively.
        assert_eq!(
            chain
                .prove_next(&proofs[2], &headers[2], &headers[3], &agg)
                .err(),
            Some(AggError::Invalid),
            "block past the final height"
        );
        let mut past = Header {
            height: 4,
            prev: headers[3].hash(),
            timestamp: headers[3].timestamp + 600,
            ..headers[3]
        };
        assert!(past.mine(data, consensus.target(&headers[3])));
        assert_eq!(
            header::check(&consensus, &headers[3], &past),
            Err(header::HeaderError::PastEnd)
        );
    }
}
