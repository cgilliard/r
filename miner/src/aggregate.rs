//! Aggregation: folds validity sub-block proofs and state-segment proofs
//! into one proof of a combined [`Statement`].
//!
//! Every node of the aggregation tree is a [`RecursionAir`] proof of the
//! same height whose public values are a statement:
//!
//! - a sub-block leaf verifies one [`SubBlockAir`] proof and states its
//!   fingerprints, fee and mint, with no state change;
//! - a segment leaf verifies one [`StateSegmentAir`] proof and states its
//!   fingerprints and state change, with no fee or mint;
//! - a merge verifies two node proofs and states their combination:
//!   fingerprints multiply, fees and mints add (renormalized to 16-bit
//!   limbs), state changes chain (the first child's end must be the second
//!   child's start), and `lambda`, `r` must agree.
//!
//! All nodes share one constraint program, so a node's verifying key is its
//! preprocessed commitment alone, and a merge can verify either kind of
//! leaf or another merge with the key as data. The merge's own key can't be
//! a constant of its own circuit, so it is part of the statement: a merge
//! accepts a child whose key is a leaf key or the statement's key, and
//! requires the children's statements to name the same key. Whoever checks
//! the final proof checks that the key is the real merge key.
//!
//! Grouping stays hidden: only the final proof is published, it is zero
//! knowledge, and its statement holds only totals. The block-level checks
//! (products against the published lists, the state chain against the
//! headers, mint against subsidy plus fees) are made natively on the final
//! statement.

use std::fmt;

use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::circuit::{Cell, Circuit, ExtCell};
use crate::fri::FriParams;
use crate::hash::{HASH_LEN, Hash};
use crate::output::AMOUNT_LIMBS;
use crate::recursion_air::RecursionAir;
use crate::stark::{self, ProvingKey, StarkError, StarkProof, VerifyingKey};
use crate::state::State;
use crate::state_air::{self, StateSegmentAir};
use crate::subblock_air::{self, SubBlockAir};
use crate::symbolic::{Program, compile};
use crate::verifier_circuit::{self, Inner, ProofCells};

/// Offsets of a statement's elements.
pub mod statement {
    use crate::hash::HASH_LEN;

    /// The merge verifying key (its preprocessed commitment).
    pub const VK: usize = 0;
    pub const LAMBDA: usize = VK + HASH_LEN;
    pub const R: usize = LAMBDA + 4;
    /// Fingerprints of the sub-blocks' inputs and outputs.
    pub const P_SUB_IN: usize = R + 4;
    pub const P_SUB_OUT: usize = P_SUB_IN + 4;
    /// Fingerprints of the segments' inserts and spends.
    pub const P_INS: usize = P_SUB_OUT + 4;
    pub const P_SP: usize = P_INS + 4;
    /// 16-bit limbs.
    pub const FEE: usize = P_SP + 4;
    pub const MINT: usize = FEE + 4;
    /// 1 if any segment is included; the state fields are zero otherwise.
    pub const HAS_STATE: usize = MINT + 4;
    /// `root | size lo, hi` before and after, as a segment's public values.
    pub const OLD_ROOT: usize = HAS_STATE + 1;
    pub const OLD_SIZE: usize = OLD_ROOT + HASH_LEN;
    pub const NEW_ROOT: usize = OLD_SIZE + 2;
    pub const NEW_SIZE: usize = NEW_ROOT + HASH_LEN;
    /// The last inserted and spent keys before and after.
    pub const PREV_INS_IN: usize = NEW_SIZE + 2;
    pub const PREV_INS_OUT: usize = PREV_INS_IN + HASH_LEN;
    pub const PREV_SP_IN: usize = PREV_INS_OUT + HASH_LEN;
    pub const PREV_SP_OUT: usize = PREV_SP_IN + HASH_LEN;
    pub const LEN: usize = PREV_SP_OUT + HASH_LEN;

    /// The state fields: `OLD_ROOT .. LEN`.
    pub const STATE: usize = OLD_ROOT;
    /// Before (`OLD_*`, `PREV_*_IN`) and after (`NEW_*`, `PREV_*_OUT`) in
    /// chain order: `(old, new)` offset pairs and their lengths.
    pub const LINKS: [(usize, usize, usize); 4] = [
        (OLD_ROOT, NEW_ROOT, HASH_LEN),
        (OLD_SIZE, NEW_SIZE, 2),
        (PREV_INS_IN, PREV_INS_OUT, HASH_LEN),
        (PREV_SP_IN, PREV_SP_OUT, HASH_LEN),
    ];
}

use statement as st;

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

    fn amount(&self, at: usize) -> u64 {
        (0..AMOUNT_LIMBS).fold(0, |acc, i| {
            acc | (self.0[at + i].as_u32() as u64) << (16 * i)
        })
    }

    fn state(&self, root: usize, size: usize) -> State {
        let (lo, hi) = (
            self.0[size].as_u32() as u64,
            self.0[size + 1].as_u32() as u64,
        );
        State {
            root: self.hash(root),
            size: lo | hi << 30,
        }
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

    pub fn p_sub_in(&self) -> BabyBear4 {
        self.ext(st::P_SUB_IN)
    }

    pub fn p_sub_out(&self) -> BabyBear4 {
        self.ext(st::P_SUB_OUT)
    }

    pub fn p_ins(&self) -> BabyBear4 {
        self.ext(st::P_INS)
    }

    pub fn p_sp(&self) -> BabyBear4 {
        self.ext(st::P_SP)
    }

    pub fn fee(&self) -> u64 {
        self.amount(st::FEE)
    }

    pub fn mint(&self) -> u64 {
        self.amount(st::MINT)
    }

    pub fn has_state(&self) -> bool {
        self.0[st::HAS_STATE] == BabyBear::ONE
    }

    pub fn old_state(&self) -> State {
        self.state(st::OLD_ROOT, st::OLD_SIZE)
    }

    pub fn new_state(&self) -> State {
        self.state(st::NEW_ROOT, st::NEW_SIZE)
    }

    pub fn prev_insert_in(&self) -> Hash {
        self.hash(st::PREV_INS_IN)
    }

    pub fn prev_insert_out(&self) -> Hash {
        self.hash(st::PREV_INS_OUT)
    }

    pub fn prev_spend_in(&self) -> Hash {
        self.hash(st::PREV_SP_IN)
    }

    pub fn prev_spend_out(&self) -> Hash {
        self.hash(st::PREV_SP_OUT)
    }
}

/// What a node proof proves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    SubBlock,
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

/// One node circuit's layout and keys.
struct Node {
    air: RecursionAir,
    pk: ProvingKey,
    vk: VerifyingKey,
}

impl Node {
    fn new(params: &FriParams, circuit: &Circuit, public: &[Cell], log_height: u32) -> Self {
        let air = RecursionAir::with_height(circuit, public, log_height);
        let (pk, vk) = stark::setup(params, &air);
        Self { air, pk, vk }
    }

    fn root(&self) -> Hash {
        self.vk.preprocessed.expect("preprocessed columns").root
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

/// Keys and layouts for aggregating one sub-block shape and one segment
/// shape.
pub struct Aggregator {
    params: FriParams,
    log_height: u32,
    sub: Leaf<SubBlockAir>,
    seg: Leaf<StateSegmentAir>,
    /// The constraints shared by every node.
    node_program: Program,
    leaf_sub: Node,
    leaf_seg: Node,
    merge: Node,
}

/// Inputs `values` and returns their cells.
fn inputs(c: &mut Circuit, values: &[BabyBear]) -> Vec<Cell> {
    values.iter().map(|&v| c.input(v)).collect()
}

/// New input cells equal to `cells`: the node's public values.
fn publish(c: &mut Circuit, cells: &[Cell]) -> Vec<Cell> {
    assert_eq!(cells.len(), st::LEN);
    cells
        .iter()
        .map(|&x| {
            let p = c.input(c.value(x));
            c.assert_eq(p, x);
            p
        })
        .collect()
}

fn ext_at(cells: &[Cell], at: usize) -> ExtCell {
    ExtCell(core::array::from_fn(|b| cells[at + b]))
}

/// `a + b` on 16-bit limbs, renormalized; the total must fit in the limbs.
fn add_amounts(c: &mut Circuit, a: &[Cell], b: &[Cell]) -> Vec<Cell> {
    let mut carry = c.zero();
    let mut out = Vec::with_capacity(AMOUNT_LIMBS);
    for i in 0..AMOUNT_LIMBS {
        let s = c.add(a[i], b[i]);
        let t = c.add(s, carry);
        if i + 1 == AMOUNT_LIMBS {
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

impl Aggregator {
    /// Sets up aggregation of `sub` proofs (verifying key `sub_vk`) and
    /// `seg` proofs, every node at height `2^log_height`.
    ///
    /// # Panics
    /// If a node circuit doesn't fit the height.
    pub fn new(
        params: FriParams,
        sub: SubBlockAir,
        sub_vk: VerifyingKey,
        seg: StateSegmentAir,
        seg_vk: VerifyingKey,
        log_height: u32,
    ) -> Self {
        let sub = Leaf::new(sub, sub_vk, subblock_air::PUBLIC_LEN);
        let seg = Leaf::new(seg, seg_vk, state_air::public::LEN);
        let zero_vk = [BabyBear::ZERO; HASH_LEN];

        let p = StarkProof::placeholder(&params, &sub.air, sub.log_n);
        let (c, public) = Self::leaf_sub_circuit(
            &params,
            &sub,
            zero_vk,
            &[BabyBear::ZERO; subblock_air::PUBLIC_LEN],
            &p,
        );
        let leaf_sub = Node::new(&params, &c, &public, log_height);

        let p = StarkProof::placeholder(&params, &seg.air, seg.log_n);
        let (c, public) = Self::leaf_seg_circuit(
            &params,
            &seg,
            zero_vk,
            &[BabyBear::ZERO; state_air::public::LEN],
            &p,
        );
        let leaf_seg = Node::new(&params, &c, &public, log_height);

        let node_program = compile(&leaf_sub.air, st::LEN);
        let p = StarkProof::placeholder(&params, &leaf_sub.air, log_height);
        let child = NodeProof {
            kind: Kind::Merge,
            statement: Statement(vec![BabyBear::ZERO; st::LEN]),
            proof: p,
        };
        let mut agg = Self {
            params,
            log_height,
            sub,
            seg,
            node_program,
            merge: Node {
                air: leaf_sub.air.clone(),
                pk: leaf_sub.pk.clone(),
                vk: leaf_sub.vk,
            },
            leaf_sub,
            leaf_seg,
        };
        let (c, public) = agg.merge_circuit(&child, &child);
        agg.merge = Node::new(&agg.params, &c, &public, log_height);
        agg
    }

    /// The merge key every statement must name.
    pub fn merge_key(&self) -> Hash {
        self.merge.root()
    }

    pub fn log_height(&self) -> u32 {
        self.log_height
    }

    fn node(&self, kind: Kind) -> &Node {
        match kind {
            Kind::SubBlock => &self.leaf_sub,
            Kind::Segment => &self.leaf_seg,
            Kind::Merge => &self.merge,
        }
    }

    fn leaf_sub_circuit(
        params: &FriParams,
        sub: &Leaf<SubBlockAir>,
        vk: Hash,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> (Circuit, Vec<Cell>) {
        let mut c = Circuit::new();
        let vk = inputs(&mut c, &vk);
        let pv = inputs(&mut c, public);
        let cells = ProofCells::load(&mut c, proof);
        verifier_circuit::verify(&mut c, &sub.inner(params), &pv, &cells);
        let (one, zero) = (c.one(), c.zero());
        let mut s = vk;
        s.extend_from_slice(&pv[..16]); // lambda, r, P_in, P_out
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // P_ins, P_sp
        s.extend_from_slice(&pv[16..24]); // fee, mint
        s.push(zero); // has_state
        s.resize(st::LEN, zero);
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
        s.extend([one, zero, zero, zero, one, zero, zero, zero]); // P_sub_in, P_sub_out
        s.extend_from_slice(&pv[sp::P_INS..sp::PREV_INS_IN]);
        s.extend([zero; 2 * AMOUNT_LIMBS]); // fee, mint
        s.push(one); // has_state
        s.extend_from_slice(&pv[sp::OLD_ROOT..sp::LAMBDA]);
        s.extend_from_slice(&pv[sp::PREV_INS_IN..sp::LEN]);
        let public = publish(&mut c, &s);
        (c, public)
    }

    /// Verifies `child` in `c` against the key chosen by its kind hint: a
    /// leaf key, or `merge_vk` (the statement's own key).
    fn verify_child(&self, c: &mut Circuit, child: &NodeProof, merge_vk: &[Cell]) -> Vec<Cell> {
        let s = inputs(c, &child.statement.0);
        let sel: Vec<Cell> = [Kind::SubBlock, Kind::Segment, Kind::Merge]
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
        let (l_sub, l_seg) = (self.leaf_sub.root(), self.leaf_seg.root());
        let root: [Cell; HASH_LEN] = core::array::from_fn(|j| {
            let a = c.mul_const(sel[0], l_sub[j]);
            let b = c.mul_const(sel[1], l_seg[j]);
            let m = c.mul(sel[2], merge_vk[j]);
            let ab = c.add(a, b);
            c.add(ab, m)
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
        for j in st::LAMBDA..st::P_SUB_IN {
            c.assert_eq(sa[j], sb[j]);
        }
        let mut s = vk.clone();
        s.extend_from_slice(&sa[st::LAMBDA..st::P_SUB_IN]);
        for at in [st::P_SUB_IN, st::P_SUB_OUT, st::P_INS, st::P_SP] {
            let p = c.ext_mul(ext_at(&sa, at), ext_at(&sb, at));
            s.extend(p.0);
        }
        for at in [st::FEE, st::MINT] {
            let sum = add_amounts(c, &sa[at..at + AMOUNT_LIMBS], &sb[at..at + AMOUNT_LIMBS]);
            s.extend(sum);
        }
        // State: chain when both have one, else whichever has one.
        let (ha, hb) = (sa[st::HAS_STATE], sb[st::HAS_STATE]);
        let both = c.mul(ha, hb);
        let either = c.add(ha, hb);
        let h = c.sub(either, both);
        s.push(h);
        let mut fields = vec![c.zero(); st::LEN - st::STATE];
        for (old, new, len) in st::LINKS {
            for j in 0..len {
                let d = c.sub(sa[new + j], sb[old + j]);
                let gated = c.mul(both, d);
                c.assert_zero(gated);
                // Start: a's if it has a state; end: b's if it has one.
                fields[old - st::STATE + j] = select(c, ha, sb[old + j], sa[old + j]);
                fields[new - st::STATE + j] = select(c, hb, sa[new + j], sb[new + j]);
            }
        }
        s.extend(fields);
        let public = publish(c, &s);
        (circuit, public)
    }

    fn prove_node(&self, kind: Kind, c: &Circuit, public: &[Cell]) -> Result<NodeProof, AggError> {
        if !c.failures().is_empty() {
            return Err(AggError::Invalid);
        }
        let node = self.node(kind);
        assert!(
            RecursionAir::with_height(c, public, self.log_height) == node.air,
            "circuit shape changed"
        );
        let pv = node.air.public_values(c);
        let proof = stark::prove(&self.params, &node.pk, &node.air, node.air.trace(c), &pv);
        Ok(NodeProof {
            kind,
            statement: Statement(pv),
            proof,
        })
    }

    /// Wraps a sub-block proof with public values `public`.
    pub fn prove_sub_block(
        &self,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> Result<NodeProof, AggError> {
        let (c, cells) =
            Self::leaf_sub_circuit(&self.params, &self.sub, self.merge_key(), public, proof);
        self.prove_node(Kind::SubBlock, &c, &cells)
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

    /// Merges two nodes; `a`'s state change (if any) comes first.
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

    /// Rows each node circuit uses, out of `2^log_height`.
    pub fn used_rows(&self) -> [usize; 3] {
        [&self.leaf_sub, &self.leaf_seg, &self.merge].map(|n| n.air.used_rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;
    use crate::hash;
    use crate::output::{Amount, NONCE_LEN, Output, OutputRecord};
    use crate::rng::Rng;
    use crate::state::StateTree;
    use crate::state_air::{InsertOp, SegmentStart};
    use crate::subblock_air::{output_element, product};
    use crate::transaction::{Transaction, TxBody};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);

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
        Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs,
            },
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn amounts_add_with_carries() {
        let limbs = |x: u64| crate::output::amount_limbs(x);
        for (a, b) in [
            (0, 0),
            (0xffff, 1),
            (u64::MAX - 7, 7),
            (0x1234_ffff_ffff, 0x1),
        ] {
            let mut c = Circuit::new();
            let (ca, cb) = (inputs(&mut c, &limbs(a)), inputs(&mut c, &limbs(b)));
            let sum = add_amounts(&mut c, &ca, &cb);
            assert!(c.failures().is_empty());
            let v: Vec<BabyBear> = sum.iter().map(|&x| c.value(x)).collect();
            assert_eq!(v, limbs(a + b));
        }
        let mut c = Circuit::new();
        let (ca, cb) = (inputs(&mut c, &limbs(u64::MAX)), inputs(&mut c, &limbs(1)));
        add_amounts(&mut c, &ca, &cb);
        assert!(!c.failures().is_empty());
    }

    struct Setup {
        params: FriParams,
        sub: SubBlockAir,
        sub_pk: ProvingKey,
        seg: StateSegmentAir,
        seg_pk: ProvingKey,
        agg: Aggregator,
    }

    impl Setup {
        fn new() -> Self {
            let params = FriParams::default();
            let sub = SubBlockAir {
                inputs: 1,
                outputs: 2,
            };
            let seg = StateSegmentAir {
                inserts: 3,
                spends: 1,
            };
            let (sub_pk, sub_vk) = stark::setup(&params, &sub);
            let (seg_pk, seg_vk) = stark::setup(&params, &seg);
            let t = std::time::Instant::now();
            let agg = Aggregator::new(params, sub, sub_vk, seg, seg_vk, 15);
            eprintln!(
                "setup {:.1} s, rows used {:?}",
                t.elapsed().as_secs_f64(),
                agg.used_rows()
            );
            Self {
                params,
                sub,
                sub_pk,
                seg,
                seg_pk,
                agg,
            }
        }

        fn sub_block(&self, tx: &Transaction, lambda: BabyBear4) -> NodeProof {
            let (t, public) = self.sub.trace(tx, lambda, R, &mut Rng::from_seed(1));
            let proof = stark::prove(&self.params, &self.sub_pk, &self.sub, t, &public);
            let time = std::time::Instant::now();
            let node = self.agg.prove_sub_block(&public, &proof).unwrap();
            eprintln!("sub-block leaf {:.1} s", time.elapsed().as_secs_f64());
            node
        }

        /// Inserts `records` (sorted) into `tree` as one segment.
        fn segment(
            &self,
            env: &TempEnv,
            tree: &StateTree,
            records: &[OutputRecord],
        ) -> (NodeProof, State, State) {
            let mut txn = env.env().write_txn().unwrap();
            let before = tree.state(&txn).unwrap();
            let mut records = records.to_vec();
            records.sort_by_key(|r| {
                crate::state::key_of(&r.commitment)
                    .unwrap()
                    .map(|x| x.as_u32())
            });
            let ops: Vec<InsertOp> = records
                .into_iter()
                .map(|record| InsertOp {
                    proof: tree.insert(&mut txn, &record.commitment).unwrap(),
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
            let (t, public) = self.seg.trace(&start, &ops, &[], LAMBDA, R);
            let proof = stark::prove(&self.params, &self.seg_pk, &self.seg, t, &public);
            let time = std::time::Instant::now();
            let node = self.agg.prove_segment(&public, &proof).unwrap();
            eprintln!("segment leaf {:.1} s", time.elapsed().as_secs_f64());
            (node, before, after)
        }

        fn merge(&self, a: &NodeProof, b: &NodeProof) -> NodeProof {
            let time = std::time::Instant::now();
            let node = self.agg.merge(a, b).unwrap();
            eprintln!("merge {:.1} s", time.elapsed().as_secs_f64());
            node
        }
    }

    /// The full pipeline: two reward sub-blocks and the segment inserting
    /// their outputs, merged into one proof, and a merge of a merge.
    #[test]
    #[ignore = "slow: run in release"]
    fn aggregates_sub_blocks_and_a_segment() {
        let s = Setup::new();
        let tx1 = reward(vec![out(1, 50), out(2, 0x1_0000_ffff)]);
        let tx2 = reward(vec![out(3, 0xffff)]);
        let records: Vec<OutputRecord> = [&tx1, &tx2]
            .iter()
            .flat_map(|t| t.body.outputs.iter().map(|o| o.record().unwrap()))
            .collect();

        let env = TempEnv::new();
        let tree = StateTree::open(env.env()).unwrap();
        let (seg, before, after) = s.segment(&env, &tree, &records);
        let sub1 = s.sub_block(&tx1, LAMBDA);
        let sub2 = s.sub_block(&tx2, LAMBDA);
        for leaf in [&seg, &sub1, &sub2] {
            assert_eq!(s.agg.verify_node(leaf), Ok(()));
        }

        let m1 = s.merge(&sub1, &seg);
        let top = s.merge(&m1, &sub2);
        assert_eq!(s.agg.verify(&m1.statement, &m1.proof), Ok(()));
        assert_eq!(s.agg.verify(&top.statement, &top.proof), Ok(()));

        let st = &top.statement;
        assert_eq!(st.vk(), s.agg.merge_key());
        assert_eq!((st.lambda(), st.r()), (LAMBDA, R));
        let p_out = product(
            R,
            records.iter().map(|r| output_element(LAMBDA, r).unwrap()),
        );
        assert_eq!(st.p_sub_out(), p_out);
        assert_eq!(st.p_ins(), p_out);
        assert_eq!((st.p_sub_in(), st.p_sp()), (BabyBear4::ONE, BabyBear4::ONE));
        assert_eq!((st.fee(), st.mint()), (0, 50 + 0x1_0000_ffff + 0xffff));
        assert!(st.has_state());
        assert_eq!((st.old_state(), st.new_state()), (before, after));
        assert_eq!(st.prev_spend_out(), [BabyBear::ZERO; HASH_LEN]);

        // Tampered statements and keys are rejected.
        let mut bad = st.clone();
        bad.0[st::MINT] += BabyBear::ONE;
        assert!(matches!(
            s.agg.verify(&bad, &top.proof),
            Err(AggError::Stark(_))
        ));
        let mut bad = st.clone();
        bad.0[st::VK] += BabyBear::ONE;
        assert_eq!(s.agg.verify(&bad, &top.proof), Err(AggError::WrongKey));
        // A leaf isn't a final proof.
        assert!(s.agg.verify(&sub1.statement, &sub1.proof).is_err());

        // Children must agree on the challenges.
        let other = s.sub_block(&tx2, LAMBDA + BabyBear4::ONE);
        assert_eq!(s.agg.merge(&m1, &other).err(), Some(AggError::Invalid));
        // A child claimed under the wrong key doesn't verify.
        let mut wrong = sub2.clone();
        wrong.kind = Kind::Segment;
        assert_eq!(s.agg.merge(&m1, &wrong).err(), Some(AggError::Invalid));
        // Two segments must chain.
        assert_eq!(s.agg.merge(&seg, &seg).err(), Some(AggError::Invalid));
    }
}
