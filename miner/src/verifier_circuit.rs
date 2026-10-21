//! The STARK verifier as a [`Circuit`]: [`stark::verify`] and
//! [`fri::verify`] rebuilt from circuit operations, so a STARK can prove
//! that it checked another STARK's proof.
//!
//! The inner proof enters as input cells ([`ProofCells::load`]); its
//! verifying key, AIR, parameters and trace height are fixed, so the
//! circuit's shape is too. The inner AIR's constraints are evaluated from
//! their compiled [`Program`]. The verifier passes iff the circuit records no
//! failed assertion.
//!
//! One deliberate difference: [`stark::verify`] re-samples `zeta` if it falls
//! in the trace or LDE domain (probability ~2^-100); the circuit uses the
//! first sample.
//!
//! [`stark::verify`]: crate::stark::verify
//! [`fri::verify`]: crate::fri::verify

use crate::air::Air;
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::circuit::{Cell, Circuit, ExtCell};
use crate::fri::{FriParams, FriProof};
use crate::hash::{DOMAIN_SLOT, HASH_LEN, Hash, domain};
use crate::poseidon2::WIDTH;
use crate::stark::{Layout, SHIFT, StarkProof, VerifyingKey};
use crate::symbolic::{Input, Node, Program};

pub(crate) type HashCells = [Cell; HASH_LEN];
const RATE: usize = HASH_LEN;

/// One FRI query's openings, as cells.
pub struct LayerCells {
    pub values: Vec<ExtCell>,
    pub path: Vec<HashCells>,
}

/// The committed rows opened at one query, as cells.
pub struct OpeningCells {
    pub trace_row: Vec<Cell>,
    pub trace_path: Vec<HashCells>,
    pub preprocessed_row: Vec<Cell>,
    pub preprocessed_path: Vec<HashCells>,
    pub aux_row: Vec<Cell>,
    pub aux_path: Vec<HashCells>,
    pub quotient_row: Vec<Cell>,
    pub quotient_path: Vec<HashCells>,
}

/// A STARK proof loaded into input cells.
pub struct ProofCells {
    pub trace_commitment: HashCells,
    pub aux_commitment: Option<HashCells>,
    pub quotient_commitment: HashCells,
    pub trace_at_zeta: Vec<ExtCell>,
    pub trace_at_next: Vec<ExtCell>,
    pub preprocessed_at_zeta: Vec<ExtCell>,
    pub preprocessed_at_next: Vec<ExtCell>,
    pub aux_at_zeta: Vec<ExtCell>,
    pub aux_at_next: Vec<ExtCell>,
    pub quotient_at_zeta: Vec<ExtCell>,
    pub fri_commitments: Vec<HashCells>,
    pub final_poly: Vec<ExtCell>,
    pub pow_witness: Cell,
    pub fri_queries: Vec<Vec<LayerCells>>,
    pub openings: Vec<OpeningCells>,
}

fn hash_input(c: &mut Circuit, h: &Hash) -> HashCells {
    h.map(|x| c.input(x))
}

fn exts_input(c: &mut Circuit, v: &[BabyBear4]) -> Vec<ExtCell> {
    v.iter().map(|&x| c.ext_input(x)).collect()
}

fn row_input(c: &mut Circuit, row: &[BabyBear]) -> Vec<Cell> {
    row.iter().map(|&x| c.input(x)).collect()
}

fn path_input(c: &mut Circuit, path: &[Hash]) -> Vec<HashCells> {
    path.iter().map(|h| hash_input(c, h)).collect()
}

impl ProofCells {
    pub fn load(c: &mut Circuit, proof: &StarkProof) -> Self {
        let fri: &FriProof = &proof.fri;
        Self {
            trace_commitment: hash_input(c, &proof.trace_commitment),
            aux_commitment: proof.aux_commitment.map(|h| hash_input(c, &h)),
            quotient_commitment: hash_input(c, &proof.quotient_commitment),
            trace_at_zeta: exts_input(c, &proof.trace_at_zeta),
            trace_at_next: exts_input(c, &proof.trace_at_next),
            preprocessed_at_zeta: exts_input(c, &proof.preprocessed_at_zeta),
            preprocessed_at_next: exts_input(c, &proof.preprocessed_at_next),
            aux_at_zeta: exts_input(c, &proof.aux_at_zeta),
            aux_at_next: exts_input(c, &proof.aux_at_next),
            quotient_at_zeta: exts_input(c, &proof.quotient_at_zeta),
            fri_commitments: fri.commitments.iter().map(|h| hash_input(c, h)).collect(),
            final_poly: exts_input(c, &fri.final_poly),
            pow_witness: c.input(fri.pow_witness),
            fri_queries: fri
                .queries
                .iter()
                .map(|q| {
                    q.layers
                        .iter()
                        .map(|l| LayerCells {
                            values: exts_input(c, &l.values),
                            path: path_input(c, &l.path),
                        })
                        .collect()
                })
                .collect(),
            openings: proof
                .openings
                .iter()
                .map(|o| OpeningCells {
                    trace_row: row_input(c, &o.trace_row),
                    trace_path: path_input(c, &o.trace_path),
                    preprocessed_row: row_input(c, &o.preprocessed_row),
                    preprocessed_path: path_input(c, &o.preprocessed_path),
                    aux_row: row_input(c, &o.aux_row),
                    aux_path: path_input(c, &o.aux_path),
                    quotient_row: row_input(c, &o.quotient_row),
                    quotient_path: path_input(c, &o.quotient_path),
                })
                .collect(),
        }
    }
}

/// The duplex transcript of [`crate::challenger::Challenger`], in cells.
struct Transcript {
    state: [Cell; WIDTH],
    input: Vec<Cell>,
    output: Vec<Cell>,
}

impl Transcript {
    fn new(c: &mut Circuit) -> Self {
        let zero = c.zero();
        let mut state = [zero; WIDTH];
        state[DOMAIN_SLOT] = c.constant(BabyBear::new(domain::TRANSCRIPT));
        Self {
            state,
            input: Vec::with_capacity(RATE),
            output: Vec::with_capacity(RATE),
        }
    }

    fn observe(&mut self, c: &mut Circuit, x: Cell) {
        self.output.clear();
        self.input.push(x);
        if self.input.len() == RATE {
            self.duplex(c);
        }
    }

    fn observe_all(&mut self, c: &mut Circuit, xs: &[Cell]) {
        for &x in xs {
            self.observe(c, x);
        }
    }

    fn observe_ext(&mut self, c: &mut Circuit, x: ExtCell) {
        self.observe_all(c, &x.0);
    }

    fn duplex(&mut self, c: &mut Circuit) {
        self.state = c.duplex(self.state, &self.input);
        self.input.clear();
        self.output.clear();
        self.output.extend_from_slice(&self.state[..RATE]);
    }

    fn sample(&mut self, c: &mut Circuit) -> Cell {
        if !self.input.is_empty() || self.output.is_empty() {
            self.duplex(c);
        }
        self.output.pop().expect("duplex fills the output")
    }

    fn sample_ext(&mut self, c: &mut Circuit) -> ExtCell {
        ExtCell([
            self.sample(c),
            self.sample(c),
            self.sample(c),
            self.sample(c),
        ])
    }

    /// The low `bits` bits of a sample's canonical value.
    fn sample_bits(&mut self, c: &mut Circuit, bits: usize) -> Vec<Cell> {
        let x = self.sample(c);
        let mut all = canonical_bits(c, x);
        all.truncate(bits);
        all
    }
}

/// The 31 bits of `x`'s canonical value: `x = sum b_i 2^i` with the integer
/// below `p = 15 * 2^27 + 1`, i.e. top four bits `t <= 15` (automatic) and
/// `t = 15 => ` the low 27 bits are zero.
pub(crate) fn canonical_bits(c: &mut Circuit, x: Cell) -> Vec<Cell> {
    let bits = c.bits(x, 31);
    let weighted = |c: &mut Circuit, bits: &[Cell]| {
        let mut acc = c.zero();
        for (i, &b) in bits.iter().enumerate() {
            let t = c.mul_const(b, BabyBear::new(1 << i));
            acc = c.add(acc, t);
        }
        acc
    };
    let top = weighted(c, &bits[27..]);
    let fifteen = c.constant(BabyBear::new(15));
    let w = c.sub(fifteen, top);
    let inv = c.hint(c.value(w).inverse().unwrap_or(BabyBear::ZERO));
    let wi = c.mul(w, inv);
    let one = c.one();
    let z = c.sub(one, wi);
    let wz = c.mul(w, z);
    c.assert_zero(wz);
    let low = weighted(c, &bits[..27]);
    let zl = c.mul(z, low);
    c.assert_zero(zl);
    bits
}

/// A running dot product `acc = sum gamma^j v_j` over absorbed values, with
/// `pow` the next power.
pub(crate) struct DotState {
    acc: ExtCell,
    pow: ExtCell,
    gamma: ExtCell,
}

/// `sponge(dom, [], elems)` as [`crate::hash::sponge`]: one sponge block
/// per 8 elements, the first from the initial state. With `dot`, each
/// block's values are also added to the running dot product, right after
/// the block (so the two can share a row).
pub(crate) fn sponge(
    c: &mut Circuit,
    dom: u32,
    elems: &[Cell],
    mut dot: Option<&mut DotState>,
) -> HashCells {
    let zero = c.zero();
    let mut state = [zero; WIDTH];
    state[DOMAIN_SLOT] = c.constant(BabyBear::new(dom));
    state[DOMAIN_SLOT + 1] = c.constant(BabyBear::new(elems.len() as u32));
    if elems.is_empty() {
        state = c.perm(state);
    }
    for block in elems.chunks(RATE) {
        state = c.sponge(state, block);
        if let Some(d) = dot.as_deref_mut() {
            (d.acc, d.pow) = c.dot(d.acc, d.pow, d.gamma, block);
        }
    }
    core::array::from_fn(|i| state[i])
}

/// Checks `row` sits at the index given by `bits` under `root`.
pub(crate) fn verify_path(
    c: &mut Circuit,
    root: &HashCells,
    row: &[Cell],
    bits: &[Cell],
    path: &[HashCells],
    dot: Option<&mut DotState>,
) {
    let mut node = sponge(c, domain::MERKLE_LEAF, row, dot);
    for (d, sibling) in path.iter().enumerate() {
        node = c.merkle_step(node, *sibling, bits[d]);
    }
    for j in 0..HASH_LEN {
        c.assert_eq(node[j], root[j]);
    }
}

/// `shift * w^index` with `w` of order `2^log_order` and `index` given by
/// its bits.
fn domain_point(c: &mut Circuit, shift: BabyBear, log_order: u32, bits: &[Cell]) -> Cell {
    let w = BabyBear::two_adic_generator(log_order);
    let mut acc = c.constant(shift);
    let mut pw = w;
    for &b in bits {
        // 1 + b (w^(2^d) - 1)
        let t = c.mul_const(b, pw - BabyBear::ONE);
        let f = c.add_const(t, BabyBear::ONE);
        acc = c.mul(acc, f);
        pw *= pw;
    }
    acc
}

/// The element `x^k` of the extension's basis.
fn basis(k: usize) -> BabyBear4 {
    let mut b = BabyBear4::ZERO;
    b.0[k] = BabyBear::ONE;
    b
}

/// `sum_b x^b coords[b]` for each group of four coordinate values.
fn from_coordinates(c: &mut Circuit, coords: &[ExtCell]) -> Vec<ExtCell> {
    coords
        .chunks_exact(4)
        .map(|g| {
            let mut acc = g[0];
            for (b, &v) in g.iter().enumerate().skip(1) {
                let k = c.ext_const(basis(b));
                acc = c.ext_fma(v, k, acc);
            }
            acc
        })
        .collect()
}

/// Evaluates a compiled constraint program on extension cells.
fn eval_program(c: &mut Circuit, program: &Program, input: impl Fn(Input) -> ExtCell) -> ExtCell {
    let mut values: Vec<ExtCell> = Vec::with_capacity(program.nodes.len());
    let mut consts: Vec<Option<BabyBear4>> = Vec::with_capacity(program.nodes.len());
    for node in &program.nodes {
        let (v, k) = match *node {
            Node::Input(i) => (input(i), None),
            Node::Const(k) => (c.ext_const(k), Some(k)),
            Node::Add(a, b) => (c.ext_add(values[a as usize], values[b as usize]), None),
            Node::Sub(a, b) => (c.ext_sub(values[a as usize], values[b as usize]), None),
            Node::Neg(a) => {
                let z = c.ext_const(BabyBear4::ZERO);
                (c.ext_sub(z, values[a as usize]), None)
            }
            Node::Mul(a, b) => {
                let (va, vb) = (values[a as usize], values[b as usize]);
                let v = match (consts[a as usize], consts[b as usize]) {
                    (Some(k), _) => c.ext_mul_const(vb, k),
                    (_, Some(k)) => c.ext_mul_const(va, k),
                    _ => c.ext_mul(va, vb),
                };
                (v, None)
            }
        };
        values.push(v);
        consts.push(k);
    }
    *values.last().expect("a program has an output")
}

/// `fri::fold` in cells.
fn fold(c: &mut Circuit, values: &[ExtCell], x: Cell, beta: ExtCell) -> ExtCell {
    let log_a = values.len().trailing_zeros();
    let zeta_inv = BabyBear::two_adic_generator(log_a)
        .inverse()
        .expect("roots are nonzero");
    let x_inv = c.inverse(x);
    let r = c.ext_mul_base(beta, x_inv);
    let mut result = c.ext_const(BabyBear4::ZERO);
    let mut r_j = c.ext_const(BabyBear4::ONE);
    let mut zeta_neg_j = BabyBear::ONE;
    for j in 0..values.len() {
        let mut u = values[0];
        let mut z = zeta_neg_j;
        for &v in &values[1..] {
            let zc = c.ext_const(BabyBear4::from(z));
            u = c.ext_fma(v, zc, u);
            z *= zeta_neg_j;
        }
        result = if j == 0 {
            c.ext_add(result, u)
        } else {
            c.ext_fma(u, r_j, result)
        };
        if j + 1 < values.len() {
            r_j = if j == 0 { r } else { c.ext_mul(r_j, r) };
        }
        zeta_neg_j *= zeta_inv;
    }
    c.ext_mul_const(result, BabyBear4::from(BabyBear::inv_two_pow(log_a)))
}

/// Selects `values[pos]` with `pos` given by its bits (least significant
/// first).
fn select_by_bits(c: &mut Circuit, values: &[ExtCell], bits: &[Cell]) -> ExtCell {
    let mut level = values.to_vec();
    for &b in bits {
        level = level
            .chunks_exact(2)
            .map(|pair| c.ext_select(b, pair[0], pair[1]))
            .collect();
    }
    level[0]
}

/// Everything fixed about the proofs a verifier circuit checks.
pub struct Inner<'a, A> {
    pub params: &'a FriParams,
    pub vk: &'a VerifyingKey,
    pub air: &'a A,
    /// `air`'s compiled constraints.
    pub program: &'a Program,
    /// The trace height `2^log_n`.
    pub log_n: u32,
}

/// Verifies an inner proof in-circuit, with public values `public` (cells
/// the caller supplies).
///
/// # Panics
/// If the loaded proof's shape doesn't match the AIR and parameters.
pub fn verify<A: Air>(c: &mut Circuit, inner: &Inner<'_, A>, public: &[Cell], proof: &ProofCells) {
    let root = inner.vk.preprocessed.map(|p| p.root.map(|x| c.constant(x)));
    verify_keyed(c, inner, root, public, proof);
}

/// [`verify`] with the preprocessed commitment given as cells (`inner.vk`
/// only fixes whether there is one), so the verified key can vary.
///
/// # Panics
/// As [`verify`], or if `pre_root` doesn't match `inner.vk` in presence.
pub fn verify_keyed<A: Air>(
    c: &mut Circuit,
    inner: &Inner<'_, A>,
    pre_root: Option<HashCells>,
    public: &[Cell],
    proof: &ProofCells,
) {
    assert_eq!(pre_root.is_some(), inner.vk.preprocessed.is_some());
    let Inner {
        params,
        vk: _,
        air,
        program,
        log_n,
    } = *inner;
    let layout = Layout::new(air, params, log_n).expect("supported shape");
    assert_eq!(proof.quotient_at_zeta.len(), layout.quotient_width());
    assert_eq!(proof.openings.len(), params.num_queries);

    // Transcript up to gamma, as `stark::verify`.
    let mut t = Transcript::new(c);
    let consts =
        [air.width(), air.preprocessed_width()].map(|x| c.constant(BabyBear::new(x as u32)));
    t.observe_all(c, &consts);
    if let Some(root) = &pre_root {
        t.observe_all(c, root);
    }
    let meta = [log_n as usize, public.len()].map(|x| c.constant(BabyBear::new(x as u32)));
    t.observe_all(c, &meta);
    t.observe_all(c, public);
    t.observe_all(c, &proof.trace_commitment);
    let challenges: Vec<ExtCell> = (0..air.num_challenges()).map(|_| t.sample_ext(c)).collect();
    if let Some(root) = &proof.aux_commitment {
        t.observe_all(c, root);
    }
    let alpha = t.sample_ext(c);
    t.observe_all(c, &proof.quotient_commitment);
    let zeta = t.sample_ext(c);
    let g = BabyBear::two_adic_generator(log_n);
    let zeta_next = c.ext_mul_const(zeta, BabyBear4::from(g));
    for group in [
        &proof.trace_at_zeta,
        &proof.trace_at_next,
        &proof.preprocessed_at_zeta,
        &proof.preprocessed_at_next,
        &proof.aux_at_zeta,
        &proof.aux_at_next,
        &proof.quotient_at_zeta,
    ] {
        for &e in group.iter() {
            t.observe_ext(c, e);
        }
    }
    let gamma = t.sample_ext(c);

    // Constraints at zeta against the recombined quotient.
    let mut zeta_n = zeta;
    for _ in 0..log_n {
        zeta_n = c.ext_mul(zeta_n, zeta_n);
    }
    let one = c.ext_const(BabyBear4::ONE);
    let z_h = c.ext_sub(zeta_n, one);
    let g_inv = g.inverse().expect("generators are nonzero");
    let zeta_minus_one = c.ext_sub(zeta, one);
    let g_inv_c = c.ext_const(BabyBear4::from(g_inv));
    let zeta_minus_g_inv = c.ext_sub(zeta, g_inv_c);
    let inv_first = c.ext_inverse(zeta_minus_one);
    let inv_last = c.ext_inverse(zeta_minus_g_inv);
    let is_first = c.ext_mul(z_h, inv_first);
    let is_last = c.ext_mul(z_h, inv_last);
    let aux_local = from_coordinates(c, &proof.aux_at_zeta);
    let aux_next = from_coordinates(c, &proof.aux_at_next);
    let public_ext: Vec<ExtCell> = public.iter().map(|&p| c.ext_from_base(p)).collect();
    let folded = eval_program(c, program, |i| match i {
        Input::Local(j) => proof.trace_at_zeta[j as usize],
        Input::Next(j) => proof.trace_at_next[j as usize],
        Input::PreprocessedLocal(j) => proof.preprocessed_at_zeta[j as usize],
        Input::PreprocessedNext(j) => proof.preprocessed_at_next[j as usize],
        Input::AuxLocal(j) => aux_local[j as usize],
        Input::AuxNext(j) => aux_next[j as usize],
        Input::Challenge(j) => challenges[j as usize],
        Input::Public(j) => public_ext[j as usize],
        Input::IsFirstRow => is_first,
        Input::IsLastRow => is_last,
        Input::IsTransition => zeta_minus_g_inv,
        Input::Alpha => alpha,
    });
    let s_n = layout.coset_powers();
    let mut quotient = c.ext_const(BabyBear4::ZERO);
    for (i, chunk) in proof.quotient_at_zeta[..4 * layout.chunks]
        .chunks_exact(4)
        .enumerate()
    {
        let q_i = from_coordinates(c, chunk)[0];
        let mut num = one;
        let mut den = BabyBear::ONE;
        for (j, &s) in s_n.iter().enumerate() {
            if j != i {
                let sc = c.ext_const(BabyBear4::from(s));
                let d = c.ext_sub(zeta_n, sc);
                num = c.ext_mul(num, d);
                den *= s_n[i] - s;
            }
        }
        let l_i = c.ext_mul_const(
            num,
            BabyBear4::from(den.inverse().expect("distinct cosets")),
        );
        quotient = c.ext_fma(l_i, q_i, quotient);
    }
    let expected = c.ext_mul(quotient, z_h);
    c.ext_assert_eq(folded, expected);

    // DEEP: A = sum_j gamma^j p_j over the zeta group (trace, preprocessed,
    // aux, quotient); the next group's sum over (trace, preprocessed, aux)
    // with powers gamma^(K + j) is gamma^K times the matching prefix of A.
    let zeta_group: Vec<ExtCell> = [
        &proof.trace_at_zeta,
        &proof.preprocessed_at_zeta,
        &proof.aux_at_zeta,
        &proof.quotient_at_zeta,
    ]
    .iter()
    .flat_map(|g| g.iter().copied())
    .collect();
    let next_group: Vec<ExtCell> = [
        &proof.trace_at_next,
        &proof.preprocessed_at_next,
        &proof.aux_at_next,
    ]
    .iter()
    .flat_map(|g| g.iter().copied())
    .collect();
    let zero_e = c.ext_const(BabyBear4::ZERO);
    let (mut a_at_zeta, mut next_sum) = (zero_e, zero_e);
    let mut pw = one;
    for (j, &v) in zeta_group.iter().enumerate() {
        a_at_zeta = c.ext_fma(pw, v, a_at_zeta);
        if let Some(&w) = next_group.get(j) {
            next_sum = c.ext_fma(pw, w, next_sum);
        }
        pw = c.ext_mul(pw, gamma);
    }
    let gamma_k = pw;
    let b_at_next = c.ext_mul(gamma_k, next_sum);

    // FRI, as `fri::verify`.
    let log_lde = layout.log_lde;
    let (rounds, _) = params.schedule(log_lde);
    assert_eq!(proof.fri_commitments.len(), rounds.len());
    let betas: Vec<ExtCell> = proof
        .fri_commitments
        .iter()
        .map(|root| {
            t.observe_all(c, root);
            t.sample_ext(c)
        })
        .collect();
    for &coeff in &proof.final_poly {
        t.observe_ext(c, coeff);
    }
    t.observe(c, proof.pow_witness);
    for b in t.sample_bits(c, params.grinding_bits as usize) {
        c.assert_zero(b);
    }

    for (query, o) in proof.fri_queries.iter().zip(&proof.openings) {
        let bits = t.sample_bits(c, log_lde as usize);

        // The input value: the DEEP combination from the openings, its
        // dot product accumulated while hashing each opened row.
        let mut dot = DotState {
            acc: zero_e,
            pow: one,
            gamma,
        };
        let trace_root = proof.trace_commitment;
        verify_path(
            c,
            &trace_root,
            &o.trace_row,
            &bits,
            &o.trace_path,
            Some(&mut dot),
        );
        if let Some(root) = &pre_root {
            verify_path(
                c,
                root,
                &o.preprocessed_row,
                &bits,
                &o.preprocessed_path,
                Some(&mut dot),
            );
        }
        if let Some(root) = &proof.aux_commitment {
            verify_path(c, root, &o.aux_row, &bits, &o.aux_path, Some(&mut dot));
        }
        let a_prefix = dot.acc;
        let quotient_root = proof.quotient_commitment;
        verify_path(
            c,
            &quotient_root,
            &o.quotient_row,
            &bits,
            &o.quotient_path,
            Some(&mut dot),
        );
        let a = dot.acc;
        let x = domain_point(c, SHIFT, log_lde, &bits);
        let b = c.ext_mul(gamma_k, a_prefix);
        let x_ext = c.ext_from_base(x);
        let dz = c.ext_sub(x_ext, zeta);
        let dn = c.ext_sub(x_ext, zeta_next);
        let inv_z = c.ext_inverse(dz);
        let inv_n = c.ext_inverse(dn);
        let na = c.ext_sub(a, a_at_zeta);
        let nb = c.ext_sub(b, b_at_next);
        let tb = c.ext_mul(nb, inv_n);
        let mut value = c.ext_fma(na, inv_z, tb);

        // Folding layers.
        let mut index_bits = bits;
        let mut shift_k = SHIFT;
        for ((layer, &la), (root, &beta)) in query
            .iter()
            .zip(&rounds)
            .zip(proof.fri_commitments.iter().zip(&betas))
        {
            let log_m = index_bits.len() - la as usize;
            let (leaf_bits, pos_bits) = index_bits.split_at(log_m);
            let opened = select_by_bits(c, &layer.values, &pos_bits[..la as usize]);
            c.ext_assert_eq(opened, value);
            let row: Vec<Cell> = layer.values.iter().flat_map(|v| v.0).collect();
            verify_path(c, root, &row, leaf_bits, &layer.path, None);
            // The point in the current (unfolded) domain.
            let x = domain_point(c, shift_k, index_bits.len() as u32, leaf_bits);
            value = fold(c, &layer.values, x, beta);
            index_bits = leaf_bits.to_vec();
            shift_k = shift_k.pow(1 << la);
        }
        let x = domain_point(c, shift_k, index_bits.len() as u32, &index_bits);
        let mut acc = c.ext_const(BabyBear4::ZERO);
        let x_ext = c.ext_from_base(x);
        for &coeff in proof.final_poly.iter().rev() {
            acc = c.ext_fma(acc, x_ext, coeff);
        }
        c.ext_assert_eq(acc, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::{Constraints, Var};
    use crate::lookup::{self, Term};
    use crate::matrix::RowMajorMatrix;
    use crate::poseidon2_air::{self, Poseidon2Air};
    use crate::rng::Rng;
    use crate::stark;
    use crate::symbolic::compile;

    fn params() -> FriParams {
        FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        }
    }

    /// Verifies `proof` both natively and in-circuit, returning whether the
    /// circuit accepted and checking the two agree and the circuit re-checks.
    fn both<A: Air>(air: &A, public: &[BabyBear], proof: &StarkProof) -> bool {
        let (_, vk) = stark::setup(&params(), air);
        let native = stark::verify(&params(), &vk, air, public, proof).is_ok();
        let program = compile(air, public.len());
        let mut c = Circuit::new();
        let public_cells: Vec<Cell> = public.iter().map(|&x| c.input(x)).collect();
        let cells = ProofCells::load(&mut c, proof);
        let inner = Inner {
            params: &params(),
            vk: &vk,
            air,
            program: &program,
            log_n: proof.log_n,
        };
        verify(&mut c, &inner, &public_cells, &cells);
        let in_circuit = c.failures().is_empty();
        assert_eq!(
            native, in_circuit,
            "native and in-circuit verifiers disagree"
        );
        assert_eq!(c.check().is_ok(), in_circuit, "re-execution disagrees");
        in_circuit
    }

    fn prove<A: Air>(air: &A, trace: RowMajorMatrix, public: &[BabyBear]) -> StarkProof {
        let (pk, _) = stark::setup(&params(), air);
        stark::prove_with_rng(&params(), &pk, air, trace, public, &mut Rng::from_seed(3))
    }

    struct Fibonacci;

    impl Air for Fibonacci {
        fn width(&self) -> usize {
            2
        }
        fn max_degree(&self) -> usize {
            2
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let (a, b) = (c.local()[0], c.local()[1]);
            let (a2, b2) = (c.next()[0], c.next()[1]);
            let p = c.public();
            c.assert_zero_first_row(a - p[0]);
            c.assert_zero_first_row(b - p[1]);
            c.assert_zero_transition(a2 - b);
            c.assert_zero_transition(b2 - (a + b));
            c.assert_zero_last_row(b - p[2]);
        }
    }

    fn fibonacci(n: usize) -> (RowMajorMatrix, Vec<BabyBear>) {
        let (mut a, mut b) = (BabyBear::ZERO, BabyBear::ONE);
        let mut values = Vec::with_capacity(2 * n);
        for _ in 0..n {
            values.extend([a, b]);
            (a, b) = (b, a + b);
        }
        let last = values[2 * n - 1];
        (
            RowMajorMatrix::new(values, 2),
            vec![BabyBear::ZERO, BabyBear::ONE, last],
        )
    }

    /// Preprocessed selector plus a LogUp bus between two column pairs.
    struct Mixed;

    impl Mixed {
        const N: usize = 64;

        fn terms<V: Var>(l: &[V]) -> Vec<Term<V>> {
            vec![
                Term::new(V::from(BabyBear::ONE), vec![l[0], l[1]]),
                Term::new(-V::from(BabyBear::ONE), vec![l[2], l[3]]),
            ]
        }

        fn trace() -> RowMajorMatrix {
            let pairs: Vec<(u32, u32)> = (0..Self::N as u32).map(|i| (i, i * 3 + 1)).collect();
            let values = (0..Self::N)
                .flat_map(|i| {
                    let (a, v) = pairs[i];
                    let (b, w) = pairs[(i * 5 + 3) % Self::N];
                    [a, v, b, w]
                })
                .map(BabyBear::new)
                .collect();
            RowMajorMatrix::new(values, 4)
        }
    }

    impl Air for Mixed {
        fn width(&self) -> usize {
            4
        }
        fn max_degree(&self) -> usize {
            3
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let l = c.local();
            let sel = c.preprocessed_local()[0];
            // Writes on even rows have value = 3 address + 1.
            c.assert_zero(sel * (l[1] - l[0] * BabyBear::new(3) - BabyBear::ONE));
            let terms = Self::terms(l);
            lookup::eval(c, &terms, 0, 0);
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
        fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
            Some(RowMajorMatrix::new(
                (0..Self::N).map(|i| BabyBear::from(i % 2 == 0)).collect(),
                1,
            ))
        }
        fn num_challenges(&self) -> usize {
            lookup::CHALLENGES
        }
        fn aux_width(&self) -> usize {
            lookup::aux_width(2)
        }
        fn aux_trace(
            &self,
            main: &RowMajorMatrix,
            _pre: Option<&RowMajorMatrix>,
            ch: &[BabyBear4],
        ) -> Vec<BabyBear4> {
            let rows: Vec<_> = main.rows().map(Self::terms).collect();
            lookup::trace(&rows, ch[0], ch[1]).concat()
        }
    }

    #[test]
    fn accepts_valid_proofs() {
        let (trace, public) = fibonacci(64);
        assert!(both(
            &Fibonacci,
            &public,
            &prove(&Fibonacci, trace, &public)
        ));

        let inputs: Vec<[BabyBear; 16]> = (0..64u32)
            .map(|i| core::array::from_fn(|j| BabyBear::new(i * 16 + j as u32)))
            .collect();
        let t = poseidon2_air::trace(&inputs);
        assert!(both(&Poseidon2Air, &[], &prove(&Poseidon2Air, t, &[])));

        assert!(both(&Mixed, &[], &prove(&Mixed, Mixed::trace(), &[])));
    }

    #[test]
    fn rejects_what_the_native_verifier_rejects() {
        let proof = prove(&Mixed, Mixed::trace(), &[]);
        type Tamper = Box<dyn Fn(&mut StarkProof)>;
        let tampers: Vec<(&str, Tamper)> = vec![
            (
                "trace at zeta",
                Box::new(|p| p.trace_at_zeta[1] += BabyBear4::ONE),
            ),
            (
                "aux at next",
                Box::new(|p| p.aux_at_next[2] += BabyBear4::ONE),
            ),
            (
                "quotient at zeta",
                Box::new(|p| p.quotient_at_zeta[0] += BabyBear4::ONE),
            ),
            (
                "trace row",
                Box::new(|p| p.openings[0].trace_row[0] += BabyBear::ONE),
            ),
            (
                "preprocessed row",
                Box::new(|p| p.openings[1].preprocessed_row[0] += BabyBear::ONE),
            ),
            (
                "aux path",
                Box::new(|p| p.openings[2].aux_path[1][0] += BabyBear::ONE),
            ),
            (
                "quotient row",
                Box::new(|p| p.openings[3].quotient_row[2] += BabyBear::ONE),
            ),
            (
                "fri layer value",
                Box::new(|p| p.fri.queries[4].layers[0].values[1] += BabyBear4::ONE),
            ),
            (
                "fri layer path",
                Box::new(|p| p.fri.queries[5].layers[1].path[0][3] += BabyBear::ONE),
            ),
            (
                "final poly",
                Box::new(|p| p.fri.final_poly[0] += BabyBear4::ONE),
            ),
            (
                "pow witness",
                Box::new(|p| p.fri.pow_witness += BabyBear::ONE),
            ),
            (
                "trace commitment",
                Box::new(|p| p.trace_commitment[0] += BabyBear::ONE),
            ),
        ];
        for (name, tamper) in tampers {
            let mut bad = proof.clone();
            tamper(&mut bad);
            assert!(!both(&Mixed, &[], &bad), "{name}");
        }
        // Wrong public values for Fibonacci.
        let (trace, public) = fibonacci(64);
        let proof = prove(&Fibonacci, trace, &public);
        let mut wrong = public.clone();
        wrong[2] += BabyBear::ONE;
        assert!(!both(&Fibonacci, &wrong, &proof));
    }

    #[test]
    fn canonical_bits_of_a_small_value() {
        // x = 5 also has the 31-bit form 5 + p, which the zero test on the
        // top bits rules out. The builder always takes the canonical bits;
        // the recursion AIR's tests cover a prover supplying others.
        let mut c = Circuit::new();
        let x = c.input(BabyBear::new(5));
        let bits = canonical_bits(&mut c, x);
        assert!(c.failures().is_empty());
        assert_eq!(c.value(bits[0]), BabyBear::ONE);
        assert_eq!(c.check(), Ok(()));
    }
}
