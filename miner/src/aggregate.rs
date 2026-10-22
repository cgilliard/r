//! Aggregation: folds item chunk, signature chunk, condition chunk and
//! state segment proofs into one proof of a combined [`Statement`].
//!
//! Every node of the aggregation tree is a [`RecursionAir`] proof of the
//! same height whose public values are a statement:
//!
//! - an item chunk leaf verifies one [`ItemChunkAir`] proof and states its
//!   fingerprint ratios (as products), net value, transaction hash and the
//!   item stream's boundary hashes where it starts and stops;
//! - a signature chunk leaf verifies one [`SigChunkAir`] proof and states
//!   the inverse of its signature fingerprint and its triple hash;
//! - a condition chunk leaf verifies one [`CondChunkAir`] proof and states
//!   its signers' fingerprint (on the transaction side), the inverse of its
//!   conditions' and its hash;
//! - a segment leaf verifies one [`StateSegmentAir`] proof and states the
//!   inverses of its fingerprints and its state change (as hashes);
//! - a merge verifies two node proofs and states their combination:
//!   fingerprint ratios multiply, net values add (offset 16-bit limbs,
//!   renormalized), transaction hashes merge (`hash::merge(a, b)`), state
//!   changes and item streams chain (the first child's end must be the
//!   second child's start), and `lambda`, `r` and the block height must
//!   agree.
//!
//! The ratios are transaction side over the other side, separately for
//! spent commitments (`Q_IN`, inputs over segment removals), created
//! commitments (`Q_OUT`, outputs over segment inserts), signatures
//! (`Q_SIG`, key spends' `(pkh, digest, leaf)` and condition signers'
//! triples over signature chunks'), condition spends (`Q_COND`, item
//! chunks' claims over condition chunks') and preimages (`Q_PRE`, outputs
//! publishing one over hashlock spends): all are 1 exactly when the
//! multisets match (with overwhelming probability over `lambda, r`). The final transaction hash commits to every chunk's
//! contents and the final state to every key the segments touched, so the
//! block's challenges are derived from both ([`crate::block`]).
//!
//! All nodes share one constraint program, so a node's verifying key is its
//! preprocessed commitment alone, and a merge can verify any kind of leaf
//! or another merge with the key as data. The merge's own key can't be a
//! constant of its own circuit, so it is part of the statement: a merge
//! accepts a child whose key is a leaf key or the statement's key, and
//! requires the children's statements to name the same key.
//!
//! Grouping stays hidden: only the final proof is published, it is zero
//! knowledge, and its statement holds only totals and hashes. The chain
//! proof ([`crate::chain`]) checks the final statement.

use std::fmt;

use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::circuit::{Cell, Circuit, ExtCell};
use crate::cond_chunk_air::{self, CondChunkAir};
use crate::fri::FriParams;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::item_chunk_air::{self, Boundary, ItemChunkAir};
use crate::output::{self, NET_LIMBS, NET_OFFSET};
use crate::recursion_air::RecursionAir;
use crate::sig_chunk_air::{self, SigChunkAir};
use crate::stark::{self, KeyCache, LazyKey, StarkError, StarkProof, VerifyingKey};
use crate::state::{self, State};
use crate::state_air::{self, StateSegmentAir};
use crate::symbolic::{Program, compile};
use crate::verifier_circuit::{self, HashCells, Inner, ProofCells};

/// Offsets of a statement's elements.
pub mod statement {
    use crate::hash::HASH_LEN;

    /// The merge verifying key (its preprocessed commitment).
    pub const VK: usize = 0;
    pub const LAMBDA: usize = VK + HASH_LEN;
    pub const R: usize = LAMBDA + 4;
    /// The block's height (segments insert at it; chunks just carry it).
    pub const HEIGHT: usize = R + 4;
    /// Fingerprint ratios: spent commitments, created commitments,
    /// signatures, condition spends.
    pub const Q_IN: usize = HEIGHT + 1;
    pub const Q_OUT: usize = Q_IN + 4;
    pub const Q_SIG: usize = Q_OUT + 4;
    pub const Q_COND: usize = Q_SIG + 4;
    /// Published preimages: outputs over hashlock spends.
    pub const Q_PRE: usize = Q_COND + 4;
    /// `sum(outputs) - sum(inputs)` as [`crate::output::net_limbs`].
    pub const NET: usize = Q_PRE + 4;
    /// The leaves' transaction and pair hashes, merged in tree order.
    pub const TX_HASH: usize = NET + crate::output::NET_LIMBS;
    /// 1 if any segment is included; the state hashes are zero otherwise.
    pub const HAS_STATE: usize = TX_HASH + HASH_LEN;
    /// [`super::state_hash`] before and after.
    pub const STATE_IN: usize = HAS_STATE + 1;
    pub const STATE_OUT: usize = STATE_IN + HASH_LEN;
    /// 1 if any item chunk is included; the stream hashes are zero
    /// otherwise.
    pub const HAS_STREAM: usize = STATE_OUT + HASH_LEN;
    /// [`super::stream_hash`] of the item stream's boundary before and
    /// after.
    pub const STREAM_IN: usize = HAS_STREAM + 1;
    pub const STREAM_OUT: usize = STREAM_IN + HASH_LEN;
    pub const LEN: usize = STREAM_OUT + HASH_LEN;
}

use statement as st;

/// The hash a statement holds for a tree state and the last keys the
/// segments inserted and spent.
pub fn state_hash(state: &State, prev_insert: &Hash, prev_spend: &Hash) -> Hash {
    let (lo, hi) = state::split(state.size);
    hash::sponge(
        domain::AGG_STATE,
        &[],
        &[&state.root, &[lo, hi], prev_insert, prev_spend],
    )
}

/// The hash a statement holds for an item stream boundary.
pub fn stream_hash(b: &Boundary) -> Hash {
    hash::sponge(domain::AGG_STREAM, &[], &[&b.elements()])
}

/// A node's public values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement(pub Vec<BabyBear>);

impl Statement {
    fn ext(&self, at: usize) -> BabyBear4 {
        BabyBear4(core::array::from_fn(|b| self.0[at + b]))
    }

    fn hash(&self, at: usize) -> Hash {
        core::array::from_fn(|j| self.0[at + j])
    }

    pub fn vk(&self) -> Hash {
        self.hash(st::VK)
    }

    pub fn lambda(&self) -> BabyBear4 {
        self.ext(st::LAMBDA)
    }

    pub fn r(&self) -> BabyBear4 {
        self.ext(st::R)
    }

    pub fn height(&self) -> BabyBear {
        self.0[st::HEIGHT]
    }

    pub fn q_in(&self) -> BabyBear4 {
        self.ext(st::Q_IN)
    }

    pub fn q_out(&self) -> BabyBear4 {
        self.ext(st::Q_OUT)
    }

    pub fn q_cond(&self) -> BabyBear4 {
        self.ext(st::Q_COND)
    }

    pub fn q_pre(&self) -> BabyBear4 {
        self.ext(st::Q_PRE)
    }

    pub fn q_sig(&self) -> BabyBear4 {
        self.ext(st::Q_SIG)
    }

    /// `sum(outputs) - sum(inputs)`.
    pub fn net(&self) -> i128 {
        output::net_value(&self.0[st::NET..st::NET + NET_LIMBS])
    }

    pub fn tx_hash(&self) -> Hash {
        self.hash(st::TX_HASH)
    }

    pub fn has_state(&self) -> bool {
        self.0[st::HAS_STATE] == BabyBear::ONE
    }

    pub fn state_in(&self) -> Hash {
        self.hash(st::STATE_IN)
    }

    pub fn state_out(&self) -> Hash {
        self.hash(st::STATE_OUT)
    }

    pub fn has_stream(&self) -> bool {
        self.0[st::HAS_STREAM] == BabyBear::ONE
    }

    pub fn stream_in(&self) -> Hash {
        self.hash(st::STREAM_IN)
    }

    pub fn stream_out(&self) -> Hash {
        self.hash(st::STREAM_OUT)
    }
}

/// What a node proof proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// An item chunk leaf of the given shape (index into the aggregator's).
    Items(usize),
    /// A signature chunk leaf of the given shape.
    Sigs(usize),
    /// A condition chunk leaf of the given shape.
    Conds(usize),
    Segment,
    Merge,
}

/// A node of the aggregation tree. `kind` is the prover's bookkeeping; a
/// verifier only trusts the key it checks the proof against.
#[derive(Clone, Debug)]
pub struct NodeProof {
    pub kind: Kind,
    pub statement: Statement,
    pub proof: StarkProof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggError {
    /// An inner proof doesn't verify, or the children don't combine.
    Invalid,
    /// The statement names a key other than the merge key.
    WrongKey,
    Stark(StarkError),
}

impl fmt::Display for AggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => write!(f, "inner proofs don't verify or don't combine"),
            Self::WrongKey => write!(f, "statement names the wrong merge key"),
            Self::Stark(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AggError {}

/// One node circuit's layout and keys (the proving key built on first
/// use).
pub(crate) struct Node {
    pub(crate) air: RecursionAir,
    key: LazyKey,
    pub(crate) vk: VerifyingKey,
}

impl Node {
    pub(crate) fn new(
        params: &FriParams,
        circuit: &Circuit,
        public: &[Cell],
        log_height: u32,
        keys: &KeyCache,
    ) -> Self {
        let air = RecursionAir::with_height(circuit, public, log_height);
        let key = LazyKey::new(keys, params, &air);
        let vk = *key.verifying_key();
        Self { air, key, vk }
    }

    /// The node's key: its preprocessed commitment.
    pub(crate) fn root(&self) -> Hash {
        self.vk.preprocessed.expect("preprocessed columns").root
    }

    /// Proves `c` with `public` as this node. `None` if an assertion in `c`
    /// failed.
    ///
    /// # Panics
    /// If `c` doesn't have this node's shape.
    pub(crate) fn prove(
        &self,
        params: &FriParams,
        c: &Circuit,
        public: &[Cell],
    ) -> Option<(Vec<BabyBear>, StarkProof)> {
        if !c.failures().is_empty() {
            return None;
        }
        let log_height = self.air.height().ilog2();
        assert!(
            RecursionAir::with_height(c, public, log_height) == self.air,
            "circuit shape changed"
        );
        let pv = self.air.public_values(c);
        let pk = self.key.proving_key(params, &self.air);
        let proof = stark::prove(params, pk, &self.air, self.air.trace(c), &pv);
        Some((pv, proof))
    }
}

/// An inner AIR the leaves verify, with its key and compiled constraints.
struct Leaf<A> {
    air: A,
    vk: VerifyingKey,
    program: Program,
    log_n: u32,
}

impl<A: crate::air::Air> Leaf<A> {
    fn new(air: A, vk: VerifyingKey, public_len: usize) -> Self {
        let log_n = vk.preprocessed.expect("preprocessed columns").log_n;
        let program = compile(&air, public_len);
        Self {
            air,
            vk,
            program,
            log_n,
        }
    }

    fn inner<'a>(&'a self, params: &'a FriParams) -> Inner<'a, A> {
        Inner {
            params,
            vk: &self.vk,
            air: &self.air,
            program: &self.program,
            log_n: self.log_n,
        }
    }
}

/// Keys and layouts for aggregating item and signature chunks of lists of
/// shapes and one segment shape.
pub struct Aggregator {
    params: FriParams,
    log_height: u32,
    items: Vec<Leaf<ItemChunkAir>>,
    sigs: Vec<Leaf<SigChunkAir>>,
    conds: Vec<Leaf<CondChunkAir>>,
    seg: Leaf<StateSegmentAir>,
    /// The constraints shared by every node.
    node_program: Program,
    /// One leaf circuit per item, signature and condition chunk shape.
    leaf_items: Vec<Node>,
    leaf_sigs: Vec<Node>,
    leaf_conds: Vec<Node>,
    leaf_seg: Node,
    merge: Node,
    /// Bytes in a node proof (every node proof has the same shape).
    proof_size: usize,
}

/// Inputs `values` and returns their cells.
pub(crate) fn inputs(c: &mut Circuit, values: &[BabyBear]) -> Vec<Cell> {
    values.iter().map(|&v| c.input(v)).collect()
}

/// New input cells equal to `cells`: the node's public values.
pub(crate) fn publish(c: &mut Circuit, cells: &[Cell]) -> Vec<Cell> {
    cells
        .iter()
        .map(|&x| {
            let p = c.input(c.value(x));
            c.assert_eq(p, x);
            p
        })
        .collect()
}

pub(crate) fn ext_at(cells: &[Cell], at: usize) -> ExtCell {
    ExtCell(core::array::from_fn(|b| cells[at + b]))
}

/// [`state_hash`] in a circuit, of `root | size lo, hi | prev insert |
/// prev spend` cells.
pub(crate) fn state_hash_cells(c: &mut Circuit, fields: &[Cell]) -> HashCells {
    assert_eq!(fields.len(), 2 * HASH_LEN + 2 + HASH_LEN);
    verifier_circuit::sponge(c, domain::AGG_STATE, fields, None)
}

/// [`stream_hash`] in a circuit, of a boundary's cells.
pub(crate) fn stream_hash_cells(c: &mut Circuit, boundary: &[Cell]) -> HashCells {
    assert_eq!(boundary.len(), item_chunk_air::BOUNDARY_LEN);
    verifier_circuit::sponge(c, domain::AGG_STREAM, boundary, None)
}

/// The sum of two net values in offset limbs ([`output::net_limbs`]):
/// `a + b - 2^95`, renormalized. Fails unless the total is in range.
fn add_nets(c: &mut Circuit, a: &[Cell], b: &[Cell]) -> Vec<Cell> {
    let mut carry = c.zero();
    let mut out = Vec::with_capacity(NET_LIMBS);
    for i in 0..NET_LIMBS {
        let s = c.add(a[i], b[i]);
        let mut t = c.add(s, carry);
        let offset = (NET_OFFSET >> (16 * i)) as u16 as u32;
        if offset != 0 {
            t = c.add_const(t, -BabyBear::new(offset));
        }
        if i + 1 == NET_LIMBS {
            c.bits(t, 16);
            out.push(t);
            break;
        }
        let k = c.hint(BabyBear::from(c.value(t).as_u32() >> 16 != 0));
        let kk = c.mul(k, k);
        c.assert_eq(kk, k);
        let shifted = c.mul_const(k, BabyBear::new(1 << 16));
        let lo = c.sub(t, shifted);
        c.bits(lo, 16);
        out.push(lo);
        carry = k;
    }
    out
}

/// `a + bit (b - a)`.
fn select(c: &mut Circuit, bit: Cell, a: Cell, b: Cell) -> Cell {
    let d = c.sub(b, a);
    let m = c.mul(bit, d);
    c.add(a, m)
}

/// A statement span (state or stream) that's absent: flag and both hashes
/// zero.
fn absent(c: &mut Circuit) -> Vec<Cell> {
    vec![c.zero(); 1 + 2 * HASH_LEN]
}

impl Aggregator {
    /// Sets up aggregation of item, signature and condition chunks of the
    /// shapes in `items`, `sigs` and `conds` (with their verifying keys)
    /// and `seg` segments, every node at height `2^log_height`.
    ///
    /// # Panics
    /// If a shape list is empty or a node circuit doesn't fit the height.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: FriParams,
        items: Vec<(ItemChunkAir, VerifyingKey)>,
        sigs: Vec<(SigChunkAir, VerifyingKey)>,
        conds: Vec<(CondChunkAir, VerifyingKey)>,
        seg: StateSegmentAir,
        seg_vk: VerifyingKey,
        log_height: u32,
        keys: &KeyCache,
    ) -> Self {
        assert!(
            !items.is_empty() && !sigs.is_empty() && !conds.is_empty(),
            "no chunk shapes"
        );
        let items: Vec<Leaf<ItemChunkAir>> = items
            .into_iter()
            .map(|(air, vk)| Leaf::new(air, vk, item_chunk_air::public::LEN))
            .collect();
        let sigs: Vec<Leaf<SigChunkAir>> = sigs
            .into_iter()
            .map(|(air, vk)| Leaf::new(air, vk, sig_chunk_air::public::LEN))
            .collect();
        let conds: Vec<Leaf<CondChunkAir>> = conds
            .into_iter()
            .map(|(air, vk)| Leaf::new(air, vk, cond_chunk_air::public::LEN))
            .collect();
        let seg = Leaf::new(seg, seg_vk, state_air::public::LEN);
        let zero_vk = [BabyBear::ZERO; HASH_LEN];

        let leaf_items: Vec<Node> = items
            .iter()
            .map(|leaf| {
                let p = StarkProof::placeholder(&params, &leaf.air, leaf.log_n);
                let public = [BabyBear::ZERO; item_chunk_air::public::LEN];
                let (c, out) = Self::leaf_items_circuit(&params, leaf, zero_vk, &public, &p, 0);
                Node::new(&params, &c, &out, log_height, keys)
            })
            .collect();
        let leaf_sigs: Vec<Node> = sigs
            .iter()
            .map(|leaf| {
                let p = StarkProof::placeholder(&params, &leaf.air, leaf.log_n);
                let public = [BabyBear::ZERO; sig_chunk_air::public::LEN];
                let (c, out) = Self::leaf_sigs_circuit(&params, leaf, zero_vk, &public, &p, 0);
                Node::new(&params, &c, &out, log_height, keys)
            })
            .collect();
        let leaf_conds: Vec<Node> = conds
            .iter()
            .map(|leaf| {
                let p = StarkProof::placeholder(&params, &leaf.air, leaf.log_n);
                let public = [BabyBear::ZERO; cond_chunk_air::public::LEN];
                let (c, out) = Self::leaf_conds_circuit(&params, leaf, zero_vk, &public, &p);
                Node::new(&params, &c, &out, log_height, keys)
            })
            .collect();
        let p = StarkProof::placeholder(&params, &seg.air, seg.log_n);
        let (c, public) = Self::leaf_seg_circuit(
            &params,
            &seg,
            zero_vk,
            &[BabyBear::ZERO; state_air::public::LEN],
            &p,
        );
        let leaf_seg = Node::new(&params, &c, &public, log_height, keys);

        let first = &leaf_items[0];
        let node_program = compile(&first.air, st::LEN);
        let p = StarkProof::placeholder(&params, &first.air, log_height);
        let child = NodeProof {
            kind: Kind::Merge,
            statement: Statement(vec![BabyBear::ZERO; st::LEN]),
            proof: p,
        };
        let merge = Node {
            air: first.air.clone(),
            key: LazyKey::verifying_only(first.vk),
            vk: first.vk,
        };
        let mut agg = Self {
            params,
            log_height,
            items,
            sigs,
            conds,
            seg,
            node_program,
            leaf_items,
            leaf_sigs,
            leaf_conds,
            leaf_seg,
            merge,
            proof_size: child.proof.size(),
        };
        let (c, public) = agg.merge_circuit(&child, &child);
        agg.merge = Node::new(&agg.params, &c, &public, log_height, keys);
        agg
    }

    /// Bytes in a node proof ([`StarkProof::size`]).
    pub fn proof_size(&self) -> usize {
        self.proof_size
    }

    pub fn params(&self) -> &FriParams {
        &self.params
    }

    /// The verifier of final (merge) proofs, for circuits that verify one.
    pub(crate) fn merge_inner(&self) -> Inner<'_, RecursionAir> {
        Inner {
            params: &self.params,
            vk: &self.merge.vk,
            air: &self.merge.air,
            program: &self.node_program,
            log_n: self.log_height,
        }
    }

    /// The merge key every statement must name.
    pub fn merge_key(&self) -> Hash {
        self.merge.root()
    }

    pub fn log_height(&self) -> u32 {
        self.log_height
    }

    /// The item chunk shapes, as [`Aggregator::prove_items`] indexes them.
    pub fn item_shapes(&self) -> Vec<ItemChunkAir> {
        self.items.iter().map(|l| l.air).collect()
    }

    /// The signature chunk shapes, as [`Aggregator::prove_sigs`] indexes
    /// them.
    pub fn sig_shapes(&self) -> Vec<SigChunkAir> {
        self.sigs.iter().map(|l| l.air).collect()
    }

    /// The condition chunk shapes, as [`Aggregator::prove_conds`] indexes
    /// them.
    pub fn cond_shapes(&self) -> Vec<CondChunkAir> {
        self.conds.iter().map(|l| l.air).collect()
    }

    fn node(&self, kind: Kind) -> &Node {
        match kind {
            Kind::Items(i) => &self.leaf_items[i],
            Kind::Sigs(i) => &self.leaf_sigs[i],
            Kind::Conds(i) => &self.leaf_conds[i],
            Kind::Segment => &self.leaf_seg,
            Kind::Merge => &self.merge,
        }
    }

    fn leaf_items_circuit(
        params: &FriParams,
        leaf: &Leaf<ItemChunkAir>,
        vk: Hash,
        public: &[BabyBear],
        proof: &StarkProof,
        height: u32,
    ) -> (Circuit, Vec<Cell>) {
        use item_chunk_air::public as ip;
        let mut c = Circuit::new();
        let vk = inputs(&mut c, &vk);
        let pv = inputs(&mut c, public);
        let cells = ProofCells::load(&mut c, proof);
        verifier_circuit::verify(&mut c, &leaf.inner(params), &pv, &cells);
        let one = c.one();
        let mut s = vk;
        // lambda, r, the height (any: the chunk doesn't depend on it), Q_in
        // = P_in, Q_out = P_out, Q_sig = P_sig, Q_cond = P_cond, Q_pre =
        // P_pre, net, tx hash.
        s.extend_from_slice(&pv[ip::LAMBDA..ip::P_IN]);
        s.push(c.input(BabyBear::new(height)));
        s.extend_from_slice(&pv[ip::P_IN..ip::ENTRY]);
        s.extend(absent(&mut c));
        s.push(one); // has_stream
        let entry = stream_hash_cells(&mut c, &pv[ip::ENTRY..ip::EXIT]);
        let exit = stream_hash_cells(&mut c, &pv[ip::EXIT..ip::LEN]);
        s.extend(entry);
        s.extend(exit);
        let public = publish(&mut c, &s);
        (c, public)
    }

    fn leaf_sigs_circuit(
        params: &FriParams,
        leaf: &Leaf<SigChunkAir>,
        vk: Hash,
        public: &[BabyBear],
        proof: &StarkProof,
        height: u32,
    ) -> (Circuit, Vec<Cell>) {
        use sig_chunk_air::public as gp;
        let mut c = Circuit::new();
        let vk = inputs(&mut c, &vk);
        let pv = inputs(&mut c, public);
        let cells = ProofCells::load(&mut c, proof);
        verifier_circuit::verify(&mut c, &leaf.inner(params), &pv, &cells);
        let (one, zero) = (c.one(), c.zero());
        let mut s = vk;
        s.extend_from_slice(&pv[gp::LAMBDA..gp::P_SIG]);
        s.push(c.input(BabyBear::new(height)));
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // Q_in, Q_out
        let q_sig = c.ext_inverse(ext_at(&pv, gp::P_SIG));
        s.extend(q_sig.0);
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // Q_cond, Q_pre
        for limb in output::net_limbs(0) {
            s.push(c.constant(limb)); // net
        }
        s.extend_from_slice(&pv[gp::HASH..gp::LEN]); // tx hash: the pair hash
        s.extend(absent(&mut c));
        s.extend(absent(&mut c));
        let public = publish(&mut c, &s);
        (c, public)
    }

    fn leaf_conds_circuit(
        params: &FriParams,
        leaf: &Leaf<CondChunkAir>,
        vk: Hash,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> (Circuit, Vec<Cell>) {
        use cond_chunk_air::public as cp;
        let mut c = Circuit::new();
        let vk = inputs(&mut c, &vk);
        let pv = inputs(&mut c, public);
        let cells = ProofCells::load(&mut c, proof);
        verifier_circuit::verify(&mut c, &leaf.inner(params), &pv, &cells);
        let (one, zero) = (c.one(), c.zero());
        let mut s = vk;
        s.extend_from_slice(&pv[cp::LAMBDA..cp::P_COND]); // lambda, r, height
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // Q_in, Q_out
        // Its signers are on the transaction side; its conditions on the
        // other.
        s.extend_from_slice(&pv[cp::P_SIG..cp::P_SIG + 4]);
        let q_cond = c.ext_inverse(ext_at(&pv, cp::P_COND));
        s.extend(q_cond.0);
        let q_pre = c.ext_inverse(ext_at(&pv, cp::P_PRE));
        s.extend(q_pre.0);
        for limb in output::net_limbs(0) {
            s.push(c.constant(limb)); // net
        }
        s.extend_from_slice(&pv[cp::HASH..cp::LEN]); // tx hash
        s.extend(absent(&mut c));
        s.extend(absent(&mut c));
        let public = publish(&mut c, &s);
        (c, public)
    }

    fn leaf_seg_circuit(
        params: &FriParams,
        seg: &Leaf<StateSegmentAir>,
        vk: Hash,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> (Circuit, Vec<Cell>) {
        use state_air::public as sp;
        let mut c = Circuit::new();
        let vk = inputs(&mut c, &vk);
        let pv = inputs(&mut c, public);
        let cells = ProofCells::load(&mut c, proof);
        verifier_circuit::verify(&mut c, &seg.inner(params), &pv, &cells);
        let (one, zero) = (c.one(), c.zero());
        let mut s = vk;
        s.extend_from_slice(&pv[sp::LAMBDA..sp::P_INS]);
        s.push(pv[sp::HEIGHT]);
        let q_in = c.ext_inverse(ext_at(&pv, sp::P_SP));
        let q_out = c.ext_inverse(ext_at(&pv, sp::P_INS));
        s.extend(q_in.0);
        s.extend(q_out.0);
        s.extend([one, zero, zero, zero]); // Q_sig
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // Q_cond, Q_pre
        for limb in output::net_limbs(0) {
            s.push(c.constant(limb)); // net
        }
        s.extend([zero; HASH_LEN]); // tx hash
        s.push(one); // has_state
        let field = |at: usize, len: usize| pv[at..at + len].to_vec();
        let mut before = field(sp::OLD_ROOT, HASH_LEN + 2);
        before.extend(field(sp::PREV_INS_IN, HASH_LEN));
        before.extend(field(sp::PREV_SP_IN, HASH_LEN));
        let mut after = field(sp::NEW_ROOT, HASH_LEN + 2);
        after.extend(field(sp::PREV_INS_OUT, HASH_LEN));
        after.extend(field(sp::PREV_SP_OUT, HASH_LEN));
        let state_in = state_hash_cells(&mut c, &before);
        let state_out = state_hash_cells(&mut c, &after);
        s.extend(state_in);
        s.extend(state_out);
        s.extend(absent(&mut c));
        let public = publish(&mut c, &s);
        (c, public)
    }

    /// Verifies `child` in `c` against the key chosen by its kind hint: a
    /// leaf key, or `merge_vk` (the statement's own key).
    fn verify_child(&self, c: &mut Circuit, child: &NodeProof, merge_vk: &[Cell]) -> Vec<Cell> {
        let s = inputs(c, &child.statement.0);
        let kinds: Vec<Kind> = (0..self.leaf_items.len())
            .map(Kind::Items)
            .chain((0..self.leaf_sigs.len()).map(Kind::Sigs))
            .chain((0..self.leaf_conds.len()).map(Kind::Conds))
            .chain([Kind::Segment, Kind::Merge])
            .collect();
        let sel: Vec<Cell> = kinds
            .iter()
            .map(|&k| c.hint(BabyBear::from(child.kind == k)))
            .collect();
        let mut sum = c.zero();
        for &b in &sel {
            let bb = c.mul(b, b);
            c.assert_eq(bb, b);
            sum = c.add(sum, b);
        }
        let one = c.one();
        c.assert_eq(sum, one);
        // The child's key: a leaf key picked by the selectors, or the
        // statement's own (merge) key.
        let leaf_keys: Vec<Hash> = self
            .leaf_items
            .iter()
            .chain(&self.leaf_sigs)
            .chain(&self.leaf_conds)
            .chain([&self.leaf_seg])
            .map(Node::root)
            .collect();
        let merge_sel = *sel.last().expect("merge selector");
        let root: [Cell; HASH_LEN] = core::array::from_fn(|j| {
            let mut acc = c.mul(merge_sel, merge_vk[j]);
            for (&b, key) in sel.iter().zip(&leaf_keys) {
                let t = c.mul_const(b, key[j]);
                acc = c.add(acc, t);
            }
            acc
        });
        let cells = ProofCells::load(c, &child.proof);
        let inner = Inner {
            params: &self.params,
            vk: &self.merge.vk,
            air: &self.merge.air,
            program: &self.node_program,
            log_n: self.log_height,
        };
        verifier_circuit::verify_keyed(c, &inner, Some(root), &s, &cells);
        s
    }

    /// Chains an optional span (state or stream) of two children: with
    /// both, `a`'s end must be `b`'s start; the combination starts where
    /// the first present one starts and ends where the last ends. Returns
    /// `has | start | end`.
    fn chain(
        c: &mut Circuit,
        sa: &[Cell],
        sb: &[Cell],
        has: usize,
        start: usize,
        end: usize,
    ) -> Vec<Cell> {
        let (ha, hb) = (sa[has], sb[has]);
        let both = c.mul(ha, hb);
        let either = c.add(ha, hb);
        let h = c.sub(either, both);
        let mut out = vec![h];
        for j in 0..HASH_LEN {
            let d = c.sub(sa[end + j], sb[start + j]);
            let gated = c.mul(both, d);
            c.assert_zero(gated);
        }
        for j in 0..HASH_LEN {
            out.push(select(c, ha, sb[start + j], sa[start + j]));
        }
        for j in 0..HASH_LEN {
            out.push(select(c, hb, sa[end + j], sb[end + j]));
        }
        out
    }

    fn merge_circuit(&self, a: &NodeProof, b: &NodeProof) -> (Circuit, Vec<Cell>) {
        let mut circuit = Circuit::new();
        let c = &mut circuit;
        let vk = inputs(c, &a.statement.vk());
        let sa = self.verify_child(c, a, &vk);
        let sb = self.verify_child(c, b, &vk);
        for j in 0..HASH_LEN {
            c.assert_eq(sa[st::VK + j], vk[j]);
            c.assert_eq(sb[st::VK + j], vk[j]);
        }
        for j in st::LAMBDA..st::Q_IN {
            c.assert_eq(sa[j], sb[j]);
        }
        let mut s = vk.clone();
        s.extend_from_slice(&sa[st::LAMBDA..st::Q_IN]);
        for at in [st::Q_IN, st::Q_OUT, st::Q_SIG, st::Q_COND, st::Q_PRE] {
            let p = c.ext_mul(ext_at(&sa, at), ext_at(&sb, at));
            s.extend(p.0);
        }
        let net = add_nets(
            c,
            &sa[st::NET..st::NET + NET_LIMBS],
            &sb[st::NET..st::NET + NET_LIMBS],
        );
        s.extend(net);
        let a_tx: [Cell; HASH_LEN] = core::array::from_fn(|j| sa[st::TX_HASH + j]);
        let b_tx: [Cell; HASH_LEN] = core::array::from_fn(|j| sb[st::TX_HASH + j]);
        let left = c.zero();
        s.extend(c.merkle_step(a_tx, b_tx, left));
        s.extend(Self::chain(
            c,
            &sa,
            &sb,
            st::HAS_STATE,
            st::STATE_IN,
            st::STATE_OUT,
        ));
        s.extend(Self::chain(
            c,
            &sa,
            &sb,
            st::HAS_STREAM,
            st::STREAM_IN,
            st::STREAM_OUT,
        ));
        let public = publish(c, &s);
        (circuit, public)
    }

    fn prove_node(&self, kind: Kind, c: &Circuit, public: &[Cell]) -> Result<NodeProof, AggError> {
        assert_eq!(public.len(), st::LEN);
        let (pv, proof) = self
            .node(kind)
            .prove(&self.params, c, public)
            .ok_or(AggError::Invalid)?;
        Ok(NodeProof {
            kind,
            statement: Statement(pv),
            proof,
        })
    }

    /// Wraps a proof of item chunk shape `shape` with public values
    /// `public`, for the block at `height`.
    pub fn prove_items(
        &self,
        shape: usize,
        public: &[BabyBear],
        proof: &StarkProof,
        height: u32,
    ) -> Result<NodeProof, AggError> {
        let (c, cells) = Self::leaf_items_circuit(
            &self.params,
            &self.items[shape],
            self.merge_key(),
            public,
            proof,
            height,
        );
        self.prove_node(Kind::Items(shape), &c, &cells)
    }

    /// Wraps a proof of signature chunk shape `shape` with public values
    /// `public`, for the block at `height`.
    pub fn prove_sigs(
        &self,
        shape: usize,
        public: &[BabyBear],
        proof: &StarkProof,
        height: u32,
    ) -> Result<NodeProof, AggError> {
        let (c, cells) = Self::leaf_sigs_circuit(
            &self.params,
            &self.sigs[shape],
            self.merge_key(),
            public,
            proof,
            height,
        );
        self.prove_node(Kind::Sigs(shape), &c, &cells)
    }

    /// Wraps a proof of condition chunk shape `shape` with public values
    /// `public`.
    pub fn prove_conds(
        &self,
        shape: usize,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> Result<NodeProof, AggError> {
        let (c, cells) = Self::leaf_conds_circuit(
            &self.params,
            &self.conds[shape],
            self.merge_key(),
            public,
            proof,
        );
        self.prove_node(Kind::Conds(shape), &c, &cells)
    }

    /// Wraps a state-segment proof with public values `public`.
    pub fn prove_segment(
        &self,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> Result<NodeProof, AggError> {
        let (c, cells) =
            Self::leaf_seg_circuit(&self.params, &self.seg, self.merge_key(), public, proof);
        self.prove_node(Kind::Segment, &c, &cells)
    }

    /// Merges two nodes; `a`'s state change and item stream (if any) come
    /// first.
    pub fn merge(&self, a: &NodeProof, b: &NodeProof) -> Result<NodeProof, AggError> {
        let (c, cells) = self.merge_circuit(a, b);
        self.prove_node(Kind::Merge, &c, &cells)
    }

    /// Checks a final (merge) proof of `statement`: it must name the merge
    /// key and verify under it.
    pub fn verify(&self, statement: &Statement, proof: &StarkProof) -> Result<(), AggError> {
        if statement.0.len() != st::LEN || statement.vk() != self.merge_key() {
            return Err(AggError::WrongKey);
        }
        stark::verify(
            &self.params,
            &self.merge.vk,
            &self.merge.air,
            &statement.0,
            proof,
        )
        .map_err(AggError::Stark)
    }

    /// Checks any node proof against its kind's key.
    pub fn verify_node(&self, node: &NodeProof) -> Result<(), AggError> {
        let n = self.node(node.kind);
        stark::verify(&self.params, &n.vk, &n.air, &node.statement.0, &node.proof)
            .map_err(AggError::Stark)
    }

    /// Rows each node circuit uses: the item chunk leaves (by shape), the
    /// signature chunk leaves, the segment leaf, the merge.
    pub fn used_rows(&self) -> Vec<usize> {
        self.leaf_items
            .iter()
            .chain(&self.leaf_sigs)
            .chain(&self.leaf_conds)
            .chain([&self.leaf_seg, &self.merge])
            .map(|n| n.air.used_rows())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::item_chunk_air::{Item, output_element, product, stream};
    use crate::output::{Amount, NONCE_LEN, Output, OutputRecord};
    use crate::rng::Rng;
    use crate::stark::ProvingKey;
    use crate::state::StateTree;
    use crate::state_air::{InsertOp, SegmentStart};
    use crate::transaction::{Transaction, TxBody};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);
    const HEIGHT: u32 = 3;

    fn out(n: u8, amount: Amount) -> Output {
        Output {
            hash: hash::to_bytes(&BabyBear::new_array(core::array::from_fn(|i| {
                n as u32 * 100 + i as u32
            }))),
            amount,
            nonce: [n; NONCE_LEN],
        }
    }

    fn reward(outputs: Vec<Output>) -> Transaction {
        Transaction::sign(TxBody::new(vec![], outputs), vec![]).unwrap()
    }

    #[test]
    fn nets_add_with_carries() {
        let limbs = output::net_limbs;
        let big = 1i128 << 94;
        for (a, b) in [
            (0, 0),
            (0xffff, 1),
            (-1, 1),
            (-5, -7),
            (u64::MAX as i128, -(u64::MAX as i128) - 3),
            (big, big - 1),
            (-big, -big + 1),
        ] {
            let mut c = Circuit::new();
            let (ca, cb) = (inputs(&mut c, &limbs(a)), inputs(&mut c, &limbs(b)));
            let sum = add_nets(&mut c, &ca, &cb);
            assert!(c.failures().is_empty(), "{a} + {b}");
            let v: Vec<BabyBear> = sum.iter().map(|&x| c.value(x)).collect();
            assert_eq!(v, limbs(a + b));
        }
        for (a, b) in [(big, big), (-big, -big - 1)] {
            let mut c = Circuit::new();
            let (ca, cb) = (inputs(&mut c, &limbs(a)), inputs(&mut c, &limbs(b)));
            add_nets(&mut c, &ca, &cb);
            assert!(!c.failures().is_empty(), "{a} + {b} overflows");
        }
    }

    #[test]
    fn hashes_match_in_and_out_of_circuit() {
        let state = State {
            root: [BabyBear::new(4); HASH_LEN],
            size: (1 << 33) + 5,
        };
        let (ins, sp) = ([BabyBear::new(7); HASH_LEN], [BabyBear::new(9); HASH_LEN]);
        let mut c = Circuit::new();
        let (lo, hi) = state::split(state.size);
        let mut fields = state.root.to_vec();
        fields.extend([lo, hi]);
        fields.extend(ins);
        fields.extend(sp);
        let cells = inputs(&mut c, &fields);
        let h = state_hash_cells(&mut c, &cells);
        assert_eq!(h.map(|x| c.value(x)), state_hash(&state, &ins, &sp));
        let b = Boundary {
            cnt_out: 3,
            ..Boundary::IDLE
        };
        let cells = inputs(&mut c, &b.elements());
        let h = stream_hash_cells(&mut c, &cells);
        assert_eq!(h.map(|x| c.value(x)), stream_hash(&b));
    }

    struct Setup {
        params: FriParams,
        items: ItemChunkAir,
        items_pk: ProvingKey,
        sigs: SigChunkAir,
        sigs_pk: ProvingKey,
        seg: StateSegmentAir,
        seg_pk: ProvingKey,
        agg: Aggregator,
    }

    impl Setup {
        fn new() -> Self {
            let params = FriParams::default();
            let items = crate::block::ITEM_SHAPES[0];
            let sigs = SigChunkAir { slots: 1 };
            let seg = StateSegmentAir {
                inserts: 3,
                spends: 1,
            };
            let (items_pk, items_vk) = stark::setup(&params, &items);
            let (sigs_pk, sigs_vk) = stark::setup(&params, &sigs);
            let (seg_pk, seg_vk) = stark::setup(&params, &seg);
            let conds = CondChunkAir { slots: 1 };
            let (_, conds_vk) = stark::setup(&params, &conds);
            let t = std::time::Instant::now();
            let agg = Aggregator::new(
                params,
                vec![(items, items_vk)],
                vec![(sigs, sigs_vk)],
                vec![(conds, conds_vk)],
                seg,
                seg_vk,
                15,
                &KeyCache::none(),
            );
            eprintln!(
                "setup {:.1} s, rows used {:?}",
                t.elapsed().as_secs_f64(),
                agg.used_rows()
            );
            Self {
                params,
                items,
                items_pk,
                sigs,
                sigs_pk,
                seg,
                seg_pk,
                agg,
            }
        }

        fn items(&self, items: &[Item], entry: &Boundary) -> NodeProof {
            let blinding = [BabyBear::new(77); HASH_LEN];
            let (t, public) = self.items.trace(items, entry, &blinding, LAMBDA, R);
            let proof = stark::prove(&self.params, &self.items_pk, &self.items, t, &public);
            let time = std::time::Instant::now();
            let node = self.agg.prove_items(0, &public, &proof, HEIGHT).unwrap();
            eprintln!("item leaf {:.1} s", time.elapsed().as_secs_f64());
            node
        }

        fn no_sigs(&self) -> NodeProof {
            let blinding = [BabyBear::new(78); HASH_LEN];
            let (t, public) = self
                .sigs
                .trace(&[], &blinding, LAMBDA, R, &mut Rng::from_seed(2));
            let proof = stark::prove(&self.params, &self.sigs_pk, &self.sigs, t, &public);
            self.agg.prove_sigs(0, &public, &proof, HEIGHT).unwrap()
        }

        /// Inserts `records` (sorted) into `tree` as one segment.
        fn segment(
            &self,
            env: &TempEnv,
            tree: &StateTree,
            records: &[OutputRecord],
        ) -> (NodeProof, State, State, Hash) {
            let mut txn = env.env().write_txn().unwrap();
            let before = tree.state(&txn).unwrap();
            let mut records = records.to_vec();
            records.sort_by_key(|r| {
                crate::state::key_of(&r.commitment)
                    .unwrap()
                    .map(|x| x.as_u32())
            });
            let last = crate::state::key_of(&records.last().unwrap().commitment).unwrap();
            let ops: Vec<InsertOp> = records
                .into_iter()
                .map(|record| InsertOp {
                    proof: tree.insert(&mut txn, &record, HEIGHT as u64).unwrap(),
                    record,
                })
                .collect();
            let after = tree.state(&txn).unwrap();
            txn.commit().unwrap();
            let start = SegmentStart {
                state: before,
                prev_insert: [BabyBear::ZERO; HASH_LEN],
                prev_spend: [BabyBear::ZERO; HASH_LEN],
            };
            let (t, public) = self.seg.trace(&start, &ops, &[], HEIGHT, LAMBDA, R);
            let proof = stark::prove(&self.params, &self.seg_pk, &self.seg, t, &public);
            let node = self.agg.prove_segment(&public, &proof).unwrap();
            (node, before, after, last)
        }

        fn merge(&self, a: &NodeProof, b: &NodeProof) -> NodeProof {
            let time = std::time::Instant::now();
            let node = self.agg.merge(a, b).unwrap();
            eprintln!("merge {:.1} s", time.elapsed().as_secs_f64());
            node
        }
    }

    /// Two rewards streamed across two item chunks, a segment inserting
    /// their outputs and an empty signature chunk, merged into one proof.
    #[test]
    #[ignore = "slow: run in release"]
    fn aggregates_chunks_and_a_segment() {
        let s = Setup::new();
        let tx1 = reward(vec![out(1, 50), out(2, 0x1_0000_ffff)]);
        let tx2 = reward(vec![out(3, 0xffff)]);
        let txs = [tx1, tx2];
        let records: Vec<OutputRecord> = txs
            .iter()
            .flat_map(|t| t.body.outputs.iter().map(|o| o.record().unwrap()))
            .collect();
        let items = stream(&txs, |_| 0);
        assert_eq!(items.len(), 7);

        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let (seg, before, after, last) = s.segment(&env, &tree, &records);
        // Cut inside the first transaction, after its first output.
        let mid = items[..2].iter().fold(Boundary::IDLE, |b, i| b.after(i));
        let chunk1 = s.items(&items[..2], &Boundary::IDLE);
        let chunk2 = s.items(&items[2..], &mid);
        let sigs = s.no_sigs();
        for leaf in [&seg, &chunk1, &chunk2, &sigs] {
            assert_eq!(s.agg.verify_node(leaf), Ok(()));
        }

        let m1 = s.merge(&seg, &chunk1);
        let m2 = s.merge(&chunk2, &sigs);
        let top = s.merge(&m1, &m2);
        assert_eq!(s.agg.verify(&top.statement, &top.proof), Ok(()));

        let st = &top.statement;
        assert_eq!(st.vk(), s.agg.merge_key());
        assert_eq!((st.lambda(), st.r()), (LAMBDA, R));
        // The transactions create exactly what the segment inserts.
        assert_eq!(
            (st.q_in(), st.q_out(), st.q_sig()),
            (BabyBear4::ONE, BabyBear4::ONE, BabyBear4::ONE)
        );
        assert_eq!(st.net(), 50 + 0x1_0000_ffff + 0xffff);
        let zero = [BabyBear::ZERO; HASH_LEN];
        assert!(st.has_state() && st.has_stream());
        assert_eq!(st.state_in(), state_hash(&before, &zero, &zero));
        assert_eq!(st.state_out(), state_hash(&after, &last, &zero));
        assert_eq!(st.stream_in(), stream_hash(&Boundary::IDLE));
        assert_eq!(st.stream_out(), stream_hash(&Boundary::IDLE));
        let p_out = product(
            R,
            records.iter().map(|r| output_element(LAMBDA, r).unwrap()),
        );
        assert_eq!(chunk1.statement.q_out() * chunk2.statement.q_out(), p_out);

        // Tampered statements and keys are rejected.
        let mut bad = st.clone();
        bad.0[st::NET] += BabyBear::ONE;
        assert!(matches!(
            s.agg.verify(&bad, &top.proof),
            Err(AggError::Stark(_))
        ));
        let mut bad = st.clone();
        bad.0[st::VK] += BabyBear::ONE;
        assert_eq!(s.agg.verify(&bad, &top.proof), Err(AggError::WrongKey));
        // A leaf isn't a final proof.
        assert!(s.agg.verify(&chunk1.statement, &chunk1.proof).is_err());
        // Chunks must chain: the first chunk stops mid-transaction, where
        // it doesn't start, so it can't follow itself.
        assert_ne!(chunk1.statement.stream_out(), stream_hash(&Boundary::IDLE));
        assert_eq!(s.agg.merge(&chunk1, &chunk1).err(), Some(AggError::Invalid));
        // A child claimed under the wrong key doesn't verify.
        let mut wrong = chunk2.clone();
        wrong.kind = Kind::Segment;
        assert_eq!(s.agg.merge(&m1, &wrong).err(), Some(AggError::Invalid));
        // Two segments must chain.
        assert_eq!(s.agg.merge(&seg, &seg).err(), Some(AggError::Invalid));
    }
}
