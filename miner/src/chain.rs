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
//! Four recursion circuits:
//!
//! - **first** (block 1): verifies block 1's aggregation proof and checks
//!   its transition starts at genesis, which is built in; tip = genesis,
//!   work 0.
//! - **block** (block `N >= 2`): checks header `N - 1` against header
//!   `N - 2` ([`pow_circuit::check_header`]: link, time, ASERT, PoW) and
//!   verifies block `N`'s aggregation proof, starting at header `N - 1`'s
//!   state. States the two header hashes, header `N - 1`'s state, the new
//!   state and header `N - 1`'s work ([`block_statement`]).
//! - **chain** (block `N >= 2`): verifies block `N - 1`'s published (wrap)
//!   proof and a block proof; checks the block proof extends the previous
//!   one (its parent hash is the previous tip, its tip's state the previous
//!   state); adds the work. Its own key is part of the statement, since it
//!   can't be a constant of its own circuit.
//! - **wrap**: verifies a first or chain proof (keyed by the first key or
//!   the statement's key) and restates its statement, in a smaller trace
//!   with a larger blowup ([`wrap_params`]), so it needs half the queries:
//!   the published proof is about half the size. Every block publishes a
//!   wrap proof, and the next block's chain proof verifies that, so anyone
//!   with the chain can extend it.
//!
//! The first, block and chain circuits run at the aggregation height with
//! its parameters.
//!
//! For each aggregation proof the circuits check what a full node can't see
//! without the transactions: it names the merge key, its challenges are
//! derived from its old and new state and its transaction hash
//! ([`block::challenges`]), its fingerprint ratios are 1, its net value is
//! the subsidy, and its segments start from an empty key chain.
//!
//! A node fast-syncing from block `N` needs only headers `N - 1` and `N` and
//! block `N`'s proof: [`Chain::verify_proof`] with header `N`'s `prev` and
//! state and the claimed work, plus header `N`'s own PoW natively; then the
//! state tree from peers, checked against header `N`'s root.

use crate::aggregate::{AggError, Aggregator, Node, inputs, publish, statement as ast};
use crate::babybear::BabyBear;
use crate::block::{self, Aggregated, Block, BlockError, BlockProver, MAX_BLOCK_SIZE};
use crate::circuit::{Cell, Circuit};
use crate::fri::FriParams;
use crate::hash::{HASH_LEN, Hash, domain};
use crate::header::{self, Consensus, Header, WORK_LIMBS};
use crate::merkle::MerkleTree;
use crate::output;
use crate::pow_circuit::{self, HeaderCells};
use crate::recursion_air::RecursionAir;
use crate::stark::{self, StarkError, StarkProof};
use crate::state::{self, StateTree};
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

/// Offsets in a block proof's statement.
pub mod block_statement {
    use crate::hash::HASH_LEN;
    use crate::header::WORK_LIMBS;

    /// Header `N - 2`'s hash.
    pub const PARENT: usize = 0;
    /// Header `N - 1`'s hash.
    pub const TIP: usize = PARENT + HASH_LEN;
    /// Header `N - 1`'s state.
    pub const TIP_ROOT: usize = TIP + HASH_LEN;
    pub const TIP_SIZE: usize = TIP_ROOT + HASH_LEN;
    /// Block `N`'s new state.
    pub const NEW_ROOT: usize = TIP_SIZE + 2;
    pub const NEW_SIZE: usize = NEW_ROOT + HASH_LEN;
    /// Header `N - 1`'s work.
    pub const WORK: usize = NEW_SIZE + 2;
    pub const LEN: usize = WORK + WORK_LIMBS;
}

use block_statement as bs;
use statement as st;

/// The wrap circuit's height.
pub const WRAP_LOG_HEIGHT: u32 = 14;

/// The wrap proof's parameters: blowup 64 with 14 queries and 16 grinding
/// bits, the same ~100 bits of conjectured security as the default with
/// half the queries.
pub fn wrap_params() -> FriParams {
    FriParams {
        log_blowup: 6,
        num_queries: 14,
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
    /// The PoW dataset's tree, for opening items in header checks.
    tree: MerkleTree,
    first: Node,
    block: Node,
    chain: Node,
    wrap: Node,
    /// Constraints of first and chain proofs (they share a shape), block
    /// proofs and wrap proofs.
    chain_program: Program,
    block_program: Program,
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

/// The checks on a final aggregation statement `s` that its proof can't
/// make itself, for a transition starting at `old_root`, `old_size`.
fn check_transition(
    c: &mut Circuit,
    consensus: &Consensus,
    merge_key: &Hash,
    s: &[Cell],
    old_root: &[Cell],
    old_size: &[Cell],
) {
    let key = consts(c, merge_key);
    assert_eq_all(c, &s[ast::VK..ast::VK + HASH_LEN], &key);
    assert_eq_all(c, &s[ast::OLD_ROOT..ast::OLD_ROOT + HASH_LEN], old_root);
    assert_eq_all(c, &s[ast::OLD_SIZE..ast::OLD_SIZE + 2], old_size);

    // lambda, r = H(old state, new state, transaction hash).
    let mut elems = Vec::with_capacity(28);
    for (at, len) in [
        (ast::OLD_ROOT, HASH_LEN),
        (ast::OLD_SIZE, 2),
        (ast::NEW_ROOT, HASH_LEN),
        (ast::NEW_SIZE, 2),
        (ast::TX_HASH, HASH_LEN),
    ] {
        elems.extend_from_slice(&s[at..at + len]);
    }
    let h = verifier_circuit::sponge(c, domain::BLOCK_CHALLENGES, &elems, None);
    assert_eq_all(c, &s[ast::LAMBDA..ast::LAMBDA + 4], &h[..4]);
    assert_eq_all(c, &s[ast::R..ast::R + 4], &h[4..8]);

    let one_ext = consts(c, &crate::babybear4::BabyBear4::ONE.0);
    assert_eq_all(c, &s[ast::Q_IN..ast::Q_IN + 4], &one_ext);
    assert_eq_all(c, &s[ast::Q_OUT..ast::Q_OUT + 4], &one_ext);
    let net = consts(c, &output::net_limbs(consensus.subsidy as i128));
    assert_eq_all(c, &s[ast::NET..ast::NET + output::NET_LIMBS], &net);
    let one = c.one();
    c.assert_eq(s[ast::HAS_STATE], one);
    let zeros = consts(c, &[BabyBear::ZERO; HASH_LEN]);
    assert_eq_all(c, &s[ast::PREV_INS_IN..ast::PREV_INS_IN + HASH_LEN], &zeros);
    assert_eq_all(c, &s[ast::PREV_SP_IN..ast::PREV_SP_IN + HASH_LEN], &zeros);
}

/// Loads an aggregation statement and proof and verifies it.
fn load_aggregate(c: &mut Circuit, agg: &Aggregator, statement: &[BabyBear], proof: &StarkProof) -> Vec<Cell> {
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
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let vk = inputs(&mut c, vk);
    let s = load_aggregate(&mut c, agg, statement, proof);
    let (lo, hi) = state::split(genesis.state.size);
    let old_root = consts(&mut c, &genesis.state.root);
    let old_size = consts(&mut c, &[lo, hi]);
    check_transition(&mut c, consensus, &agg.merge_key(), &s, &old_root, &old_size);
    let mut out = vk;
    out.extend(consts(&mut c, &genesis.hash()));
    out.extend_from_slice(&s[ast::NEW_ROOT..ast::NEW_ROOT + HASH_LEN]);
    out.extend_from_slice(&s[ast::NEW_SIZE..ast::NEW_SIZE + 2]);
    out.extend(consts(&mut c, &header::work_limbs(0)));
    let public = publish(&mut c, &out);
    (c, public)
}

#[allow(clippy::too_many_arguments)]
fn block_circuit(
    consensus: &Consensus,
    tree: &MerkleTree,
    agg: &Aggregator,
    parent: &Header,
    header: &Header,
    statement: &[BabyBear],
    proof: &StarkProof,
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let p = HeaderCells::input(&mut c, parent);
    let h = HeaderCells::input(&mut c, header);
    let check = pow_circuit::check_header(&mut c, consensus, tree, &p, &h);
    let s = load_aggregate(&mut c, agg, statement, proof);
    check_transition(&mut c, consensus, &agg.merge_key(), &s, &h.root, &h.size);
    let mut out = check.parent_hash.to_vec();
    out.extend(check.hash);
    out.extend(h.root);
    out.extend(h.size);
    out.extend_from_slice(&s[ast::NEW_ROOT..ast::NEW_ROOT + HASH_LEN]);
    out.extend_from_slice(&s[ast::NEW_SIZE..ast::NEW_SIZE + 2]);
    out.extend(check.work);
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

/// Extends block `N - 1`'s wrap proof (`prev`) with a block proof.
fn chain_circuit(
    wrap: &Inner<'_, RecursionAir>,
    block: &Inner<'_, RecursionAir>,
    vk: &Hash,
    prev: &ChainProof,
    block_statement: &[BabyBear],
    block_proof: &StarkProof,
) -> (Circuit, Vec<Cell>) {
    let mut c = Circuit::new();
    let vk = inputs(&mut c, vk);

    let ps = inputs(&mut c, &prev.statement);
    let cells = ProofCells::load(&mut c, &prev.proof);
    verifier_circuit::verify(&mut c, wrap, &ps, &cells);
    assert_eq_all(&mut c, &ps[st::VK..st::VK + HASH_LEN], &vk);

    // The block proof extends it.
    let b = inputs(&mut c, block_statement);
    let cells = ProofCells::load(&mut c, block_proof);
    verifier_circuit::verify(&mut c, block, &b, &cells);
    assert_eq_all(&mut c, &b[bs::PARENT..bs::PARENT + HASH_LEN], &ps[st::TIP..st::TIP + HASH_LEN]);
    assert_eq_all(
        &mut c,
        &b[bs::TIP_ROOT..bs::TIP_ROOT + HASH_LEN],
        &ps[st::ROOT..st::ROOT + HASH_LEN],
    );
    assert_eq_all(&mut c, &b[bs::TIP_SIZE..bs::TIP_SIZE + 2], &ps[st::SIZE..st::SIZE + 2]);
    let work = pow_circuit::add_work(
        &mut c,
        &ps[st::WORK..st::WORK + WORK_LIMBS],
        &b[bs::WORK..bs::WORK + WORK_LIMBS],
    );

    let mut out = vk;
    out.extend_from_slice(&b[bs::TIP..bs::TIP + HASH_LEN]);
    out.extend_from_slice(&b[bs::NEW_ROOT..bs::NEW_ROOT + HASH_LEN]);
    out.extend_from_slice(&b[bs::NEW_SIZE..bs::NEW_SIZE + 2]);
    out.extend(work);
    let public = publish(&mut c, &out);
    (c, public)
}

impl Chain {
    /// Sets up chain proofs for the chain starting at `genesis` under
    /// `consensus`, proving transitions with `blocks` and header checks
    /// with the dataset's `tree` ([`crate::pow::Dataset::tree`]).
    ///
    /// # Panics
    /// If a circuit doesn't fit the aggregation height.
    pub fn new(consensus: Consensus, genesis: Header, blocks: BlockProver, tree: MerkleTree) -> Self {
        let agg = blocks.aggregator();
        let params = *agg.params();
        let log_height = agg.log_height();
        let merge_air = agg.merge_inner().air.clone();
        let agg_statement = vec![BabyBear::ZERO; ast::LEN];
        let agg_proof = StarkProof::placeholder(&params, &merge_air, log_height);
        let zero_vk = [BabyBear::ZERO; HASH_LEN];

        let (c, public) = first_circuit(&consensus, &genesis, agg, &zero_vk, &agg_statement, &agg_proof);
        let first = Node::new(&params, &c, &public, log_height);
        let default = Header::default();
        let (c, public) =
            block_circuit(&consensus, &tree, agg, &default, &default, &agg_statement, &agg_proof);
        let block = Node::new(&params, &c, &public, log_height);
        let chain_program = compile(&first.air, st::LEN);
        let block_program = compile(&block.air, bs::LEN);

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
        let wrap = Node::new(&wrap_params, &c, &public, WRAP_LOG_HEIGHT);
        let wrap_program = compile(&wrap.air, st::LEN);

        let prev = ChainProof {
            statement: vec![BabyBear::ZERO; st::LEN],
            proof: StarkProof::placeholder(&wrap_params, &wrap.air, WRAP_LOG_HEIGHT),
        };
        let block_proof = StarkProof::placeholder(&params, &block.air, log_height);
        let (c, public) = chain_circuit(
            &Inner {
                params: &wrap_params,
                vk: &wrap.vk,
                air: &wrap.air,
                program: &wrap_program,
                log_n: WRAP_LOG_HEIGHT,
            },
            &Inner {
                params: &params,
                vk: &block.vk,
                air: &block.air,
                program: &block_program,
                log_n: log_height,
            },
            &zero_vk,
            &prev,
            &[BabyBear::ZERO; bs::LEN],
            &block_proof,
        );
        let chain = Node::new(&params, &c, &public, log_height);
        Self {
            consensus,
            genesis,
            params,
            log_height,
            blocks,
            tree,
            first,
            block,
            chain,
            wrap,
            chain_program,
            block_program,
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

    /// Rows used by the first, block and chain circuits (of
    /// `2^log_height`) and the wrap circuit (of `2^WRAP_LOG_HEIGHT`).
    pub fn used_rows(&self) -> ([usize; 4], u32) {
        (
            [&self.first, &self.block, &self.chain, &self.wrap].map(|n| n.air.used_rows()),
            self.log_height,
        )
    }

    /// Wraps a first (`is_first`) or chain proof.
    fn wrap(&self, statement: Vec<BabyBear>, proof: StarkProof, is_first: bool) -> Result<ChainProof, AggError> {
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
        let (c, public) = block_circuit(
            &self.consensus,
            &self.tree,
            self.blocks.aggregator(),
            grandparent,
            parent,
            &agg.node.statement.0,
            &agg.node.proof,
        );
        let (block_statement, block_proof) = self
            .block
            .prove(&self.params, &c, &public)
            .ok_or(AggError::Invalid)?;
        let wrap_params = wrap_params();
        let (c, public) = chain_circuit(
            &Inner {
                params: &wrap_params,
                vk: &self.wrap.vk,
                air: &self.wrap.air,
                program: &self.wrap_program,
                log_n: WRAP_LOG_HEIGHT,
            },
            &Inner {
                params: &self.params,
                vk: &self.block.vk,
                air: &self.block.air,
                program: &self.block_program,
                log_n: self.log_height,
            },
            &self.key(),
            prev,
            &block_statement,
            &block_proof,
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
        stark::verify(&wrap_params(), &self.wrap.vk, &self.wrap.air, &statement, proof)
    }

    /// Checks `block` as the body of `header`: it fits [`MAX_BLOCK_SIZE`],
    /// its lists are canonical, the header commits to it, and its chain
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
        if !block::is_canonical(&block.inputs)
            || !block::is_canonical(block.outputs.iter().map(|o| &o.commitment))
        {
            return Err(BlockError::NotCanonical);
        }
        if block.body_hash() != Some(header.body) {
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
    ) -> Result<(), BlockError> {
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
    use crate::output::{Amount, NONCE_LEN, Nonce, Output};
    use crate::pow::Dataset;
    use crate::state_air::StateSegmentAir;
    use crate::subblock_air::SubBlockAir;
    use crate::transaction::{Input, Transaction, TxBody, pubkey_hash};
    use crate::wots::{self, SecretKey};

    fn key(n: u32) -> (SecretKey, wots::PublicKey) {
        wots::keygen(BabyBear::new_array(core::array::from_fn(|i| n * 31 + i as u32)))
    }

    fn nonce(n: u32) -> Nonce {
        [n as u8; NONCE_LEN]
    }

    fn pay(n: u32, amount: Amount) -> Output {
        Output {
            hash: pubkey_hash(&key(n).1),
            amount,
            nonce: nonce(n),
        }
    }

    fn spend(coins: &[(u32, Amount)], outputs: Vec<Output>) -> Transaction {
        let (keys, inputs) = coins
            .iter()
            .map(|&(n, amount)| {
                let (sk, pubkey) = key(n);
                let commitment =
                    output::commit(&pubkey_hash(&pubkey), amount, &nonce(n)).unwrap();
                (
                    sk,
                    Input {
                        commitment,
                        pubkey,
                        amount,
                        nonce: nonce(n),
                    },
                )
            })
            .unzip();
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
            ..Consensus::DEFAULT
        };
        let genesis = Header {
            timestamp: consensus.genesis_time,
            state: tree.state(&env.env().read_txn().unwrap()).unwrap(),
            ..Header::default()
        };
        let t = std::time::Instant::now();
        let blocks = BlockProver::new(
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
        let chain = Chain::new(consensus, genesis, blocks, data.tree());
        eprintln!("setup {:.1} s, rows {:?}", t.elapsed().as_secs_f64(), chain.used_rows());

        let subsidy = consensus.subsidy;
        let block_txs: Vec<Vec<Transaction>> = vec![
            vec![spend(&[], vec![pay(1, subsidy)])],
            vec![
                spend(&[(1, subsidy)], vec![pay(2, 20), pay(3, 25)]),
                spend(&[], vec![pay(4, subsidy + 5)]),
            ],
            vec![
                spend(&[(2, 20), (3, 25), (4, subsidy + 5)], vec![pay(5, 100)]),
                spend(&[], vec![pay(6, subsidy)]),
            ],
        ];

        let mut headers = vec![genesis];
        let mut work = vec![0u128];
        let mut proofs: Vec<ChainProof> = Vec::new();
        let mut bodies: Vec<Block> = Vec::new();
        for txs in &block_txs {
            let parent = *headers.last().unwrap();
            let t = std::time::Instant::now();
            let mut txn = env.env().write_txn().unwrap();
            let agg = chain.block_prover().prove(&tree, &mut txn, txs, subsidy).unwrap();
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
                body: block.body_hash().unwrap(),
                nonce: 0,
            };
            assert!(header.mine(data, consensus.target(&parent)));

            // A full node.
            let work_before = *work.last().unwrap();
            assert_eq!(statement_work(&proof.statement), Some(work_before));
            assert_eq!(header::check(&consensus, data, &parent, &header), Ok(()));
            let mut txn = env.env().write_txn().unwrap();
            chain
                .check_and_apply(&tree, &mut txn, &parent, &header, &block, work_before)
                .unwrap();
            txn.commit().unwrap();

            work.push(work_before + header::work(consensus.log_target(&parent)));
            headers.push(header);
            proofs.push(proof);
            bodies.push(block);
        }

        // Fast sync from the tip: two headers and one proof.
        let tip = headers[3];
        assert_eq!(chain.verify_proof(&tip, work[2], &proofs[2].proof), Ok(()));
        assert!(work[2] > 0);

        // Mismatches.
        let rejects = |h: &Header, w: u128, p: &StarkProof| chain.verify_proof(h, w, p).is_err();
        assert!(rejects(&tip, work[2] + 1, &proofs[2].proof), "wrong work");
        assert!(rejects(&tip, work[1], &proofs[2].proof), "work one block short");
        let other_state = Header {
            state: headers[2].state,
            ..tip
        };
        assert!(rejects(&other_state, work[2], &proofs[2].proof), "wrong state");
        let other_parent = Header {
            prev: headers[1].hash(),
            ..tip
        };
        assert!(rejects(&other_parent, work[2], &proofs[2].proof), "wrong parent");
        assert!(rejects(&tip, work[2], &proofs[1].proof), "previous block's proof");
        assert!(rejects(&headers[1], work[0], &proofs[2].proof), "tip's proof for block 1");
        // Every published proof is a wrap proof, about half the size of the
        // inner proofs.
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
        // transition: block 3's transition on top of header 1.
        let mut txn = env.env().write_txn().unwrap();
        let agg = chain
            .block_prover()
            .prove(&tree, &mut txn, &[spend(&[], vec![pay(7, subsidy)])], subsidy)
            .unwrap();
        txn.abort();
        assert_eq!(
            chain
                .prove_next(&proofs[2], &headers[1], &headers[2], &agg)
                .err(),
            Some(AggError::Invalid),
            "transition doesn't start at the parent's state"
        );
    }
}

