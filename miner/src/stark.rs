//! Zero-knowledge STARK prover and verifier for an [`Air`].
//!
//! For a trace of `n = 2^log_n` rows on the subgroup `H_n` (generator `g`):
//!
//! 1. **Masked trace.** Each column becomes a polynomial `T` of degree
//!    below `N = 2n`: it equals the trace on `H_n` (the even points of
//!    `H_N`) and is random on the odd points. Its low-degree extension over
//!    `SHIFT * H_(N * blowup)` is committed. Draw `alpha`.
//! 2. **Quotient.** Combine all constraints with powers of `alpha` into
//!    `C(x)` and divide by `Z_H(x) = x^n - 1`. `Q` has degree below
//!    `M = k N`, `k` = the constraint degree rounded up to a power of two.
//!    Its values on `SHIFT * H_M` split into `c = M / n` cosets `S_i` of
//!    `H_n`; chunk `q_i` interpolates `Q` on `S_i` (degree below `n`), so
//!    `Q = sum_i L_i q_i` with `L_i` the coset selectors. Each chunk gets a
//!    random `v_i t_i` added, `v_i` vanishing on `S_i`, with the `t_i`
//!    chosen to cancel in that sum (eprint 2024/1037), giving degree below
//!    `N`. The chunks and a random polynomial `R` (four base columns, an
//!    extension-field polynomial) are committed together.
//! 3. **Out-of-domain point** `zeta`: the prover sends every committed column
//!    at `zeta` and the trace at `g zeta`; the verifier checks
//!    `C(zeta) = Q(zeta) Z_H(zeta)`.
//! 4. **DEEP + FRI.** Draw `gamma`; FRI proves
//!    `sum_j gamma^j (p_j(X) - p_j(z)) / (X - z)` over all those claims has
//!    degree below `N`. `R` masks every value FRI reveals.
//!
//! Preprocessed columns (fixed per AIR) are committed once by [`setup`]
//! and opened like the trace, without masking: they are public.
//!
//! Hiding holds while the revealed evaluations stay within the randomness:
//! at most `n` per masked column and `N` for `R`, with margin. [`prove`]
//! checks this. Merkle leaves are not salted: each unopened row is a
//! function of the trace and at least `HIDING_MARGIN` unrevealed random
//! field elements, so its hash leaks nothing computationally.

use std::fmt;

use crate::air::{Air, Constraints, Rows, selectors_at};
use crate::babybear::{BabyBear, TWO_ADICITY};
use crate::babybear4::BabyBear4;
use crate::challenger::Challenger;
use crate::fri::{self, FriError, FriParams, FriProof};
use crate::hash::Hash;
use crate::matrix::RowMajorMatrix;
use crate::merkle::{self, MerklePath, MerkleTree};
use crate::ntt;
use crate::par::par_map;
use crate::rng::Rng;

/// Shift of the LDE domain: a generator of the multiplicative group, so
/// `SHIFT * H'` is disjoint from every subgroup of order `2^k`.
pub(crate) const SHIFT: BabyBear = BabyBear::new(31);

/// Random field elements per masked polynomial left unrevealed beyond what
/// the proof discloses (~248 bits).
const HIDING_MARGIN: usize = 8;

/// Base columns of the random polynomial `R`.
const RANDOM_COLUMNS: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarkProof {
    pub log_n: u32,
    pub trace_commitment: Hash,
    /// Commits to the auxiliary trace's base coordinates, if the AIR has one.
    pub aux_commitment: Option<Hash>,
    /// Commits to the quotient chunks followed by the random polynomial.
    pub quotient_commitment: Hash,
    pub trace_at_zeta: Vec<BabyBear4>,
    pub trace_at_next: Vec<BabyBear4>,
    pub preprocessed_at_zeta: Vec<BabyBear4>,
    pub preprocessed_at_next: Vec<BabyBear4>,
    /// The auxiliary trace's base coordinate columns (4 per extension
    /// column) at `zeta` and `g zeta`.
    pub aux_at_zeta: Vec<BabyBear4>,
    pub aux_at_next: Vec<BabyBear4>,
    /// Every column of the quotient commitment at `zeta`: `4 c` chunk
    /// columns, then the random polynomial's columns.
    pub quotient_at_zeta: Vec<BabyBear4>,
    pub fri: FriProof,
    /// Committed rows at each FRI query index.
    pub openings: Vec<QueryOpening>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryOpening {
    pub trace_row: Vec<BabyBear>,
    pub trace_path: MerklePath,
    /// Empty when the AIR has no preprocessed columns.
    pub preprocessed_row: Vec<BabyBear>,
    pub preprocessed_path: MerklePath,
    /// Empty when the AIR has no auxiliary trace.
    pub aux_row: Vec<BabyBear>,
    pub aux_path: MerklePath,
    pub quotient_row: Vec<BabyBear>,
    pub quotient_path: MerklePath,
}

/// Prover data fixed per AIR: its committed preprocessed columns.
#[derive(Clone, Debug)]
pub struct ProvingKey {
    preprocessed: Option<Preprocessed>,
}

#[derive(Clone, Debug)]
struct Preprocessed {
    log_n: u32,
    coeffs: RowMajorMatrix,
    tree: MerkleTree,
}

/// What a verifier needs to know about an AIR beyond its constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyingKey {
    pub preprocessed: Option<PreprocessedCommitment>,
}

/// Commitment to an AIR's preprocessed columns, and the trace height they fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreprocessedCommitment {
    pub root: Hash,
    pub log_n: u32,
}

impl StarkProof {
    /// An all-zero proof with the shape of a proof for `air` at height
    /// `2^log_n`: enough to lay out a verifier circuit, which depends only
    /// on the shape. It does not verify.
    ///
    /// # Panics
    /// If the shape is unsupported.
    pub fn placeholder<A: Air>(params: &FriParams, air: &A, log_n: u32) -> Self {
        let layout = Layout::new(air, params, log_n).expect("supported shape");
        let zero_h = [BabyBear::ZERO; crate::hash::HASH_LEN];
        let path = |len: u32| vec![zero_h; len as usize];
        let exts = |n: usize| vec![BabyBear4::ZERO; n];
        let base = |n: usize| vec![BabyBear::ZERO; n];
        let (pre, aux) = (air.preprocessed_width(), 4 * air.aux_width());
        let path_if = |width: usize| path(if width > 0 { layout.log_lde } else { 0 });
        let (rounds, log_final) = params.schedule(layout.log_lde);
        let query = || fri::QueryProof {
            layers: rounds
                .iter()
                .scan(layout.log_lde, |log_k, &la| {
                    *log_k -= la;
                    Some(fri::LayerOpening {
                        values: exts(1 << la),
                        path: path(*log_k),
                    })
                })
                .collect(),
        };
        Self {
            log_n,
            trace_commitment: zero_h,
            aux_commitment: (aux > 0).then_some(zero_h),
            quotient_commitment: zero_h,
            trace_at_zeta: exts(air.width()),
            trace_at_next: exts(air.width()),
            preprocessed_at_zeta: exts(pre),
            preprocessed_at_next: exts(pre),
            aux_at_zeta: exts(aux),
            aux_at_next: exts(aux),
            quotient_at_zeta: exts(layout.quotient_width()),
            fri: FriProof {
                commitments: vec![zero_h; rounds.len()],
                final_poly: exts(1 << log_final),
                pow_witness: BabyBear::ZERO,
                queries: (0..params.num_queries).map(|_| query()).collect(),
            },
            openings: (0..params.num_queries)
                .map(|_| QueryOpening {
                    trace_row: base(air.width()),
                    trace_path: path(layout.log_lde),
                    preprocessed_row: base(pre),
                    preprocessed_path: path_if(pre),
                    aux_row: base(aux),
                    aux_path: path_if(aux),
                    quotient_row: base(layout.quotient_width()),
                    quotient_path: path(layout.log_lde),
                })
                .collect(),
        }
    }
}

impl ProvingKey {
    pub fn verifying_key(&self) -> VerifyingKey {
        VerifyingKey {
            preprocessed: self.preprocessed.as_ref().map(|p| PreprocessedCommitment {
                root: p.tree.root(),
                log_n: p.log_n,
            }),
        }
    }
}

/// Commits to `air`'s preprocessed columns, if any.
///
/// # Panics
/// If the preprocessed columns' width is not `air.preprocessed_width()` or
/// their height is not a power of two of at least 2.
pub fn setup<A: Air>(params: &FriParams, air: &A) -> (ProvingKey, VerifyingKey) {
    let preprocessed = air.preprocessed_trace().map(|m| {
        assert_eq!(
            m.width,
            air.preprocessed_width(),
            "preprocessed width mismatch"
        );
        let n = m.height();
        assert!(
            n >= 2 && n.is_power_of_two(),
            "preprocessed height must be a power of two >= 2"
        );
        let mut coeffs = m;
        ntt::intt(&mut coeffs);
        // Same LDE domain as the masked trace: N * blowup = n * 2 * blowup.
        let tree = MerkleTree::new(ntt::coset_extend(
            coeffs.clone(),
            params.log_blowup + 1,
            SHIFT,
        ));
        Preprocessed {
            log_n: n.trailing_zeros(),
            coeffs,
            tree,
        }
    });
    assert!(
        preprocessed.is_some() || air.preprocessed_width() == 0,
        "AIR declares preprocessed columns but provides none"
    );
    let pk = ProvingKey { preprocessed };
    let vk = pk.verifying_key();
    (pk, vk)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarkError {
    /// The proof's shape does not match the AIR and parameters.
    Shape,
    /// The constraints at `zeta` disagree with the quotient at `zeta`.
    ConstraintMismatch,
    BadOpening {
        query: usize,
    },
    Fri(FriError),
}

impl fmt::Display for StarkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "proof has the wrong shape"),
            Self::ConstraintMismatch => write!(f, "constraints do not match the quotient"),
            Self::BadOpening { query } => write!(f, "query {query}: invalid trace opening"),
            Self::Fri(e) => write!(f, "FRI: {e}"),
        }
    }
}

impl std::error::Error for StarkError {}

/// Sizes derived from the trace height, AIR and parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub(crate) log_n: u32,
    /// Masked degree bound `N = 2n`.
    pub(crate) log_masked: u32,
    /// LDE domain size, `N * blowup`.
    pub(crate) log_lde: u32,
    /// Quotient domain size `M = k N`.
    pub(crate) log_quotient: u32,
    /// Number of quotient chunks `c = M / n`.
    pub(crate) chunks: usize,
}

impl Layout {
    pub(crate) fn new<A: Air>(air: &A, params: &FriParams, log_n: u32) -> Option<Self> {
        let log_k = air.max_degree().max(1).next_power_of_two().trailing_zeros();
        let log_masked = log_n + 1;
        let log_lde = log_masked + params.log_blowup;
        if log_n == 0 || log_k > params.log_blowup || log_lde > TWO_ADICITY {
            return None;
        }
        Some(Self {
            log_n,
            log_masked,
            log_lde,
            log_quotient: log_masked + log_k,
            chunks: 1 << (log_k + 1),
        })
    }

    pub(crate) fn quotient_width(&self) -> usize {
        4 * self.chunks + RANDOM_COLUMNS
    }

    /// LDE row offset from `x` to `g x`.
    fn next_offset(&self) -> usize {
        1 << (self.log_lde - self.log_n)
    }

    /// `s_i^n` for each chunk coset `S_i = SHIFT w_M^i H_n`.
    pub(crate) fn coset_powers(&self) -> Vec<BabyBear> {
        let w_c = BabyBear::two_adic_generator(self.log_quotient - self.log_n);
        let base = SHIFT.pow(1 << self.log_n);
        (0..self.chunks)
            .scan(base, |s, _| {
                let v = *s;
                *s *= w_c;
                Some(v)
            })
            .collect()
    }
}

/// Whether the proof's disclosures fit within the masking randomness: each
/// masked trace column and quotient chunk reveals `zeta`, `g zeta` and one
/// value per query; FRI reveals one value per query per folded point, all
/// masked by `R`.
fn hiding_budget_ok(params: &FriParams, layout: &Layout) -> bool {
    let n = 1usize << layout.log_n;
    let (rounds, _) = params.schedule(layout.log_lde);
    let fri_revealed: usize = rounds.iter().map(|&la| params.num_queries << la).sum();
    params.num_queries + 2 + HIDING_MARGIN <= n
        && fri_revealed + 1 + HIDING_MARGIN <= 1 << layout.log_masked
}

/// Proves that `trace` satisfies `air` with the given public values, masking
/// with randomness from the OS.
///
/// # Panics
/// If the trace shape is unsupported (height not a power of two, at least
/// 2, or not the preprocessed height; width not the AIR's), the AIR's degree
/// exceeds the blowup, the trace is too short to hide what the proof reveals,
/// or the OS has no randomness. An invalid trace does not panic; its proof
/// fails to verify.
pub fn prove<A: Air>(
    params: &FriParams,
    pk: &ProvingKey,
    air: &A,
    trace: RowMajorMatrix,
    public: &[BabyBear],
) -> StarkProof {
    let mut rng = Rng::from_os().expect("OS randomness");
    prove_with_rng(params, pk, air, trace, public, &mut rng)
}

/// [`prove`] with a caller-supplied source of masking randomness.
pub fn prove_with_rng<A: Air>(
    params: &FriParams,
    pk: &ProvingKey,
    air: &A,
    trace: RowMajorMatrix,
    public: &[BabyBear],
    rng: &mut Rng,
) -> StarkProof {
    let n = trace.height();
    assert_eq!(
        trace.width,
        air.width(),
        "trace width does not match the AIR"
    );
    assert!(
        n >= 2 && n.is_power_of_two(),
        "trace height must be a power of two >= 2"
    );
    let layout = Layout::new(air, params, n.trailing_zeros())
        .expect("constraint degree too high for the blowup, or trace too long");
    if let Some(p) = &pk.preprocessed {
        assert_eq!(
            layout.log_n, p.log_n,
            "trace height must match the preprocessed columns"
        );
    }
    assert!(
        hiding_budget_ok(params, &layout),
        "trace too short to hide what the proof reveals"
    );
    let log_blowup = params.log_blowup;
    let pre = pk.preprocessed.as_ref();
    let pre_row = |i: usize| pre.map_or(&[][..], |p| p.tree.matrix().row(i));

    let mut ch = Challenger::new();
    observe_instance(&mut ch, air, &pk.verifying_key(), layout.log_n, public);
    let mut timer = Timer::new();

    // 1. Masked trace: trace rows at even positions of H_N, random at odd.
    let trace_coeffs = masked_coeffs(&trace, rng);
    timer.lap("trace interpolation");
    let trace_lde = ntt::coset_extend(trace_coeffs.clone(), log_blowup, SHIFT);
    timer.lap("trace LDE");
    let trace_tree = MerkleTree::new(trace_lde);
    ch.observe_hash(&trace_tree.root());
    timer.lap("trace Merkle tree");

    // 1b. Auxiliary trace, from challenges drawn after the main commitment.
    let challenges: Vec<BabyBear4> = (0..air.num_challenges()).map(|_| ch.sample_ext()).collect();
    let aux_width = air.aux_width();
    let (aux_coeffs, aux_tree) = if aux_width == 0 {
        (None, None)
    } else {
        let pre_trace = air.preprocessed_trace();
        let values = air.aux_trace(&trace, pre_trace.as_ref(), &challenges);
        timer.lap("aux trace");
        assert_eq!(values.len(), n * aux_width, "aux trace shape");
        let flat = RowMajorMatrix::new(values.iter().flat_map(|e| e.0).collect(), 4 * aux_width);
        let coeffs = masked_coeffs(&flat, rng);
        let tree = MerkleTree::new(ntt::coset_extend(coeffs.clone(), log_blowup, SHIFT));
        ch.observe_hash(&tree.root());
        (Some(coeffs), Some(tree))
    };
    drop(trace);
    timer.lap("aux trace and commit");
    let alpha = ch.sample_ext();
    let aux_row = |i: usize| -> Vec<BabyBear4> {
        aux_tree
            .as_ref()
            .map_or_else(Vec::new, |t| to_ext(t.matrix().row(i)))
    };

    // 2. Quotient on SHIFT * H_M, a subset of the LDE rows.
    let lde = trace_tree.matrix();
    let lde_height = lde.height();
    let m = 1usize << layout.log_quotient;
    let step = lde_height / m;
    let w_m = BabyBear::two_adic_generator(layout.log_quotient);
    let eval_at = |j: usize, powers: Option<&[BabyBear4]>| {
        let x = SHIFT * w_m.pow(j as u64);
        let row = j * step;
        let next = (row + layout.next_offset()) % lde_height;
        let (selectors, z_h) = selectors_at(x, layout.log_n, BabyBear::inverse)
            .expect("LDE coset avoids the trace domain");
        let (aux_local, aux_next) = (aux_row(row), aux_row(next));
        let rows = Rows {
            local: lde.row(row),
            next: lde.row(next),
            preprocessed_local: pre_row(row),
            preprocessed_next: pre_row(next),
            aux_local: &aux_local,
            aux_next: &aux_next,
        };
        let mut c = Constraints::new(rows, &challenges, public, selectors, alpha);
        if let Some(p) = powers {
            c = c.with_powers(p);
        }
        air.eval(&mut c);
        let count = c.num_constraints();
        (
            c.finish() * z_h.inverse().expect("LDE coset avoids the trace domain"),
            count,
        )
    };
    // Count the constraints once, then combine with precomputed powers.
    let (_, count) = eval_at(0, None);
    let mut powers: Vec<BabyBear4> = (0..count)
        .scan(BabyBear4::ONE, |p, _| {
            let v = *p;
            *p *= alpha;
            Some(v)
        })
        .collect();
    powers.reverse();
    let quotient = par_map(m, 64, |j| eval_at(j, Some(&powers)).0);

    timer.lap("quotient evaluation");
    let chunk_coeffs = randomized_chunks(&layout, &quotient, rng);
    let quotient_tree = MerkleTree::new(ntt::coset_extend(chunk_coeffs.clone(), log_blowup, SHIFT));
    ch.observe_hash(&quotient_tree.root());
    timer.lap("quotient commit");

    // 3. Out-of-domain evaluations.
    let zeta = sample_zeta(&mut ch, layout.log_n, layout.log_lde);
    let zeta_next = zeta * BabyBear::two_adic_generator(layout.log_n);
    let pre_eval = |z| pre.map_or_else(Vec::new, |p| eval_columns(&p.coeffs, z));
    let aux_eval = |z| {
        aux_coeffs
            .as_ref()
            .map_or_else(Vec::new, |c| eval_columns(c, z))
    };
    let opened = OodValues {
        trace_at_zeta: eval_columns(&trace_coeffs, zeta),
        trace_at_next: eval_columns(&trace_coeffs, zeta_next),
        preprocessed_at_zeta: pre_eval(zeta),
        preprocessed_at_next: pre_eval(zeta_next),
        aux_at_zeta: aux_eval(zeta),
        aux_at_next: aux_eval(zeta_next),
        quotient_at_zeta: eval_columns(&chunk_coeffs, zeta),
    };
    opened.observe(&mut ch);
    timer.lap("out-of-domain evaluations");

    // 4. DEEP combination, then FRI.
    let gamma = ch.sample_ext();
    let deep = opened.deep(gamma);
    let w_lde = BabyBear::two_adic_generator(layout.log_lde);
    let point = |i: usize| BabyBear4::from(SHIFT * w_lde.pow(i as u64));
    let inv_zeta = batch_inverse(lde_height, |i| point(i) - zeta);
    let inv_next = batch_inverse(lde_height, |i| point(i) - zeta_next);
    let deep_evals = par_map(lde_height, 256, |i| {
        let (t, p, q) = (
            trace_tree.matrix().row(i),
            pre_row(i),
            quotient_tree.matrix().row(i),
        );
        let a = aux_tree.as_ref().map_or(&[][..], |t| t.matrix().row(i));
        deep.combine(&[t, p, a], q, inv_zeta[i], inv_next[i])
    });
    timer.lap("DEEP combination");
    let (fri, indices) = fri::prove(params, deep_evals, SHIFT, &mut ch);
    timer.lap("FRI");

    let openings = indices
        .iter()
        .map(|&i| {
            let (trace_row, trace_path) = trace_tree.open(i);
            let (preprocessed_row, preprocessed_path) =
                pre.map_or_else(Default::default, |p| p.tree.open(i));
            let (aux_row, aux_path) = aux_tree
                .as_ref()
                .map_or_else(Default::default, |t| t.open(i));
            let (quotient_row, quotient_path) = quotient_tree.open(i);
            QueryOpening {
                trace_row,
                trace_path,
                preprocessed_row,
                preprocessed_path,
                aux_row,
                aux_path,
                quotient_row,
                quotient_path,
            }
        })
        .collect();

    let OodValues {
        trace_at_zeta,
        trace_at_next,
        preprocessed_at_zeta,
        preprocessed_at_next,
        aux_at_zeta,
        aux_at_next,
        quotient_at_zeta,
    } = opened;
    StarkProof {
        log_n: layout.log_n,
        trace_commitment: trace_tree.root(),
        aux_commitment: aux_tree.as_ref().map(MerkleTree::root),
        quotient_commitment: quotient_tree.root(),
        trace_at_zeta,
        trace_at_next,
        preprocessed_at_zeta,
        preprocessed_at_next,
        aux_at_zeta,
        aux_at_next,
        quotient_at_zeta,
        fri,
        openings,
    }
}

/// Prints phase times when `STARK_PROFILE` is set.
struct Timer(Option<std::time::Instant>);

impl Timer {
    fn new() -> Self {
        Self(std::env::var_os("STARK_PROFILE").map(|_| std::time::Instant::now()))
    }

    fn lap(&mut self, phase: &str) {
        if let Some(t) = &mut self.0 {
            eprintln!("  {phase}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
            *t = std::time::Instant::now();
        }
    }
}

/// Coefficients of the masked columns: `m`'s rows on the even points of
/// `H_N` (`N = 2 * height`), random values on the odd points.
fn masked_coeffs(m: &RowMajorMatrix, rng: &mut Rng) -> RowMajorMatrix {
    let n = m.height();
    let mut random = vec![BabyBear::ZERO; n * m.width];
    rng.fill(&mut random);
    let mut coeffs = RowMajorMatrix::zeros(2 * n, m.width);
    for i in 0..n {
        coeffs.row_mut(2 * i).copy_from_slice(m.row(i));
        coeffs
            .row_mut(2 * i + 1)
            .copy_from_slice(&random[i * m.width..(i + 1) * m.width]);
    }
    ntt::intt(&mut coeffs);
    coeffs
}

/// Extension elements from consecutive groups of 4 base coordinates.
fn to_ext(row: &[BabyBear]) -> Vec<BabyBear4> {
    row.chunks_exact(4)
        .map(|c| BabyBear4(c.try_into().expect("4 coordinates")))
        .collect()
}

/// Extension elements from their base coordinates' values at an extension
/// point: `sum_b x^b c_b` per group of 4.
fn from_coordinates(coords: &[BabyBear4]) -> Vec<BabyBear4> {
    coords
        .chunks_exact(4)
        .map(|c| (0..4).fold(BabyBear4::ZERO, |acc, b| acc + basis(b) * c[b]))
        .collect()
}

/// Splits `Q` (its values on `SHIFT * H_M`, in order) into randomized chunks
/// and appends the random polynomial. Returns an `N`-row coefficient matrix:
/// columns `4i..4i+4` hold chunk `i`, the last `RANDOM_COLUMNS` hold `R`.
fn randomized_chunks(layout: &Layout, quotient: &[BabyBear4], rng: &mut Rng) -> RowMajorMatrix {
    let n = 1usize << layout.log_n;
    let c = layout.chunks;
    let s_n = layout.coset_powers();
    // d_i = prod_{j != i} (s_i^n - s_j^n), so that L_i v_i = V / d_i.
    let d: Vec<BabyBear> = (0..c)
        .map(|i| {
            (0..c)
                .filter(|&j| j != i)
                .fold(BabyBear::ONE, |acc, j| acc * (s_n[i] - s_n[j]))
        })
        .collect();

    // t_i random for i < c - 1; t_last = -d_last sum_i t_i / d_i.
    let mut t = vec![RowMajorMatrix::zeros(n, 4); c];
    let mut sum = RowMajorMatrix::zeros(n, 4);
    for i in 0..c - 1 {
        rng.fill(&mut t[i].values);
        let d_inv = d[i].inverse().expect("coset powers are distinct");
        for (s, &v) in sum.values.iter_mut().zip(&t[i].values) {
            *s += v * d_inv;
        }
    }
    for (v, &s) in t[c - 1].values.iter_mut().zip(&sum.values) {
        *v = -(d[c - 1] * s);
    }

    let w_m = BabyBear::two_adic_generator(layout.log_quotient);
    let mut out = RowMajorMatrix::zeros(2 * n, layout.quotient_width());
    for i in 0..c {
        // Q on S_i = SHIFT w_M^i H_n sits at quotient indices i + c t.
        let mut chunk =
            RowMajorMatrix::new((0..n).flat_map(|r| quotient[i + c * r].0).collect(), 4);
        ntt::coset_intt(&mut chunk, SHIFT * w_m.pow(i as u64));
        // q_i + (X^n - s_i^n) t_i.
        for r in 0..n {
            for b in 0..4 {
                let tv = t[i].row(r)[b];
                out.row_mut(r)[4 * i + b] = chunk.row(r)[b] - s_n[i] * tv;
                out.row_mut(n + r)[4 * i + b] = tv;
            }
        }
    }
    for r in 0..2 * n {
        rng.fill(&mut out.row_mut(r)[4 * c..]);
    }
    out
}

/// Verifies `proof` for `air` with the given public values.
pub fn verify<A: Air>(
    params: &FriParams,
    vk: &VerifyingKey,
    air: &A,
    public: &[BabyBear],
    proof: &StarkProof,
) -> Result<(), StarkError> {
    let layout = Layout::new(air, params, proof.log_n).ok_or(StarkError::Shape)?;
    let (width, q_width) = (air.width(), layout.quotient_width());
    let pre_width = vk.preprocessed.map_or(0, |_| air.preprocessed_width());
    let aux_width = 4 * air.aux_width();
    if vk.preprocessed.is_some_and(|p| p.log_n != proof.log_n)
        || pre_width != air.preprocessed_width()
        || proof.aux_commitment.is_some() != (aux_width > 0)
        || proof.aux_at_zeta.len() != aux_width
        || proof.aux_at_next.len() != aux_width
        || proof.trace_at_zeta.len() != width
        || proof.trace_at_next.len() != width
        || proof.preprocessed_at_zeta.len() != pre_width
        || proof.preprocessed_at_next.len() != pre_width
        || proof.quotient_at_zeta.len() != q_width
        || proof.openings.len() != params.num_queries
        || proof.openings.iter().any(|o| {
            o.trace_row.len() != width
                || o.preprocessed_row.len() != pre_width
                || o.aux_row.len() != aux_width
                || o.quotient_row.len() != q_width
        })
    {
        return Err(StarkError::Shape);
    }

    let mut ch = Challenger::new();
    observe_instance(&mut ch, air, vk, layout.log_n, public);
    ch.observe_hash(&proof.trace_commitment);
    let challenges: Vec<BabyBear4> = (0..air.num_challenges()).map(|_| ch.sample_ext()).collect();
    if let Some(root) = &proof.aux_commitment {
        ch.observe_hash(root);
    }
    let alpha = ch.sample_ext();
    ch.observe_hash(&proof.quotient_commitment);
    let zeta = sample_zeta(&mut ch, layout.log_n, layout.log_lde);
    let zeta_next = zeta * BabyBear::two_adic_generator(layout.log_n);
    let opened = OodValues {
        trace_at_zeta: proof.trace_at_zeta.clone(),
        trace_at_next: proof.trace_at_next.clone(),
        preprocessed_at_zeta: proof.preprocessed_at_zeta.clone(),
        preprocessed_at_next: proof.preprocessed_at_next.clone(),
        aux_at_zeta: proof.aux_at_zeta.clone(),
        aux_at_next: proof.aux_at_next.clone(),
        quotient_at_zeta: proof.quotient_at_zeta.clone(),
    };
    opened.observe(&mut ch);
    let gamma = ch.sample_ext();

    // Constraints at zeta against Q(zeta) = sum_i L_i(zeta) q_i(zeta).
    let (selectors, z_h) =
        selectors_at(zeta, layout.log_n, BabyBear4::inverse).expect("zeta avoids the trace domain");
    let (aux_local, aux_next) = (
        from_coordinates(&opened.aux_at_zeta),
        from_coordinates(&opened.aux_at_next),
    );
    let rows = Rows {
        local: &opened.trace_at_zeta,
        next: &opened.trace_at_next,
        preprocessed_local: &opened.preprocessed_at_zeta,
        preprocessed_next: &opened.preprocessed_at_next,
        aux_local: &aux_local,
        aux_next: &aux_next,
    };
    let public_ext: Vec<BabyBear4> = public.iter().map(|&x| x.into()).collect();
    let mut c = Constraints::new(rows, &challenges, &public_ext, selectors, alpha);
    air.eval(&mut c);
    let s_n = layout.coset_powers();
    let zeta_n = zeta.pow(1 << layout.log_n);
    let mut quotient = BabyBear4::ZERO;
    for (i, chunk) in opened.quotient_at_zeta[..4 * layout.chunks]
        .chunks_exact(4)
        .enumerate()
    {
        let q_i = (0..4).fold(BabyBear4::ZERO, |acc, b| acc + basis(b) * chunk[b]);
        let (mut num, mut den) = (BabyBear4::ONE, BabyBear::ONE);
        for (j, &s) in s_n.iter().enumerate() {
            if j != i {
                num *= zeta_n - s;
                den *= s_n[i] - s;
            }
        }
        quotient += num * den.inverse().expect("coset powers are distinct") * q_i;
    }
    if c.finish() != quotient * z_h {
        return Err(StarkError::ConstraintMismatch);
    }

    // FRI on the DEEP combination, computed from the openings.
    let deep = opened.deep(gamma);
    let w_lde = BabyBear::two_adic_generator(layout.log_lde);
    let log = layout.log_lde;
    let mut bad_opening = None;
    let result = fri::verify(params, log, SHIFT, &proof.fri, &mut ch, |q, index| {
        let o = &proof.openings[q];
        let pre_ok = vk.preprocessed.is_none_or(|p| {
            merkle::verify(
                &p.root,
                log,
                index,
                &o.preprocessed_row,
                &o.preprocessed_path,
            )
        });
        let aux_ok = proof
            .aux_commitment
            .is_none_or(|root| merkle::verify(&root, log, index, &o.aux_row, &o.aux_path));
        if !pre_ok
            || !aux_ok
            || !merkle::verify(
                &proof.trace_commitment,
                log,
                index,
                &o.trace_row,
                &o.trace_path,
            )
            || !merkle::verify(
                &proof.quotient_commitment,
                log,
                index,
                &o.quotient_row,
                &o.quotient_path,
            )
        {
            bad_opening = Some(q);
            return None;
        }
        let x = BabyBear4::from(SHIFT * w_lde.pow(index as u64));
        let inv_zeta = (x - zeta).inverse()?;
        let inv_next = (x - zeta_next).inverse()?;
        let (t, p, a, qr) = (
            &o.trace_row[..],
            &o.preprocessed_row[..],
            &o.aux_row[..],
            &o.quotient_row[..],
        );
        Some(deep.combine(&[t, p, a], qr, inv_zeta, inv_next))
    });
    match (result, bad_opening) {
        (Ok(()), _) => Ok(()),
        (Err(FriError::MissingInput { query }), Some(_)) => Err(StarkError::BadOpening { query }),
        (Err(e), _) => Err(StarkError::Fri(e)),
    }
}

/// Binds the statement into the transcript before any commitment: the AIR's
/// shape, its preprocessed commitment, the trace height and public values.
fn observe_instance<A: Air>(
    ch: &mut Challenger,
    air: &A,
    vk: &VerifyingKey,
    log_n: u32,
    public: &[BabyBear],
) {
    ch.observe(BabyBear::new(air.width() as u32));
    ch.observe(BabyBear::new(air.preprocessed_width() as u32));
    if let Some(p) = &vk.preprocessed {
        ch.observe_hash(&p.root);
    }
    ch.observe(BabyBear::new(log_n));
    ch.observe(BabyBear::new(public.len() as u32));
    ch.observe_slice(public);
}

/// Everything the prover sends at `zeta` and `g zeta`.
struct OodValues {
    trace_at_zeta: Vec<BabyBear4>,
    trace_at_next: Vec<BabyBear4>,
    preprocessed_at_zeta: Vec<BabyBear4>,
    preprocessed_at_next: Vec<BabyBear4>,
    aux_at_zeta: Vec<BabyBear4>,
    aux_at_next: Vec<BabyBear4>,
    quotient_at_zeta: Vec<BabyBear4>,
}

impl OodValues {
    fn observe(&self, ch: &mut Challenger) {
        for v in [
            &self.trace_at_zeta,
            &self.trace_at_next,
            &self.preprocessed_at_zeta,
            &self.preprocessed_at_next,
            &self.aux_at_zeta,
            &self.aux_at_next,
            &self.quotient_at_zeta,
        ] {
            v.iter().for_each(|&e| ch.observe_ext(e));
        }
    }

    fn deep(&self, gamma: BabyBear4) -> Deep {
        Deep::new(
            gamma,
            &[
                &self.trace_at_zeta,
                &self.preprocessed_at_zeta,
                &self.aux_at_zeta,
                &self.quotient_at_zeta,
            ],
            &[
                &self.trace_at_next,
                &self.preprocessed_at_next,
                &self.aux_at_next,
            ],
        )
    }
}

/// Samples `zeta` outside the trace domain and the LDE domain, where the
/// DEEP quotients would divide by zero. Re-sampling happens with
/// probability ~2^-100, identically for prover and verifier.
fn sample_zeta(ch: &mut Challenger, log_n: u32, log_big: u32) -> BabyBear4 {
    let shift_inv = BabyBear4::from(SHIFT.inverse().expect("shift is nonzero"));
    loop {
        let zeta = ch.sample_ext();
        let in_trace_domain = zeta.pow(1 << log_n) == BabyBear4::ONE;
        let in_lde_domain = (zeta * shift_inv).pow(1 << log_big) == BabyBear4::ONE;
        if !in_trace_domain && !in_lde_domain {
            return zeta;
        }
    }
}

/// The element `x^k` of the extension's basis.
fn basis(k: usize) -> BabyBear4 {
    let mut b = BabyBear4::ZERO;
    b.0[k] = BabyBear::ONE;
    b
}

/// Evaluates every column of a coefficient matrix at `z`.
/// Every column of `coeffs`, read as a polynomial, at `z`: a dot product
/// of each column with the powers of `z`, over blocks of columns in
/// parallel.
fn eval_columns(coeffs: &RowMajorMatrix, z: BabyBear4) -> Vec<BabyBear4> {
    const BLOCK: usize = 16;
    let (w, n) = (coeffs.width, coeffs.height());
    let powers: Vec<BabyBear4> = (0..n)
        .scan(BabyBear4::ONE, |p, _| {
            let v = *p;
            *p *= z;
            Some(v)
        })
        .collect();
    par_map(w.div_ceil(BLOCK), 1, |b| {
        let cols = b * BLOCK..((b + 1) * BLOCK).min(w);
        let mut acc = vec![BabyBear4::ZERO; cols.len()];
        for (row, &p) in coeffs.values.chunks_exact(w).zip(&powers) {
            for (a, &c) in acc.iter_mut().zip(&row[cols.clone()]) {
                *a += p * c;
            }
        }
        acc
    })
    .concat()
}

/// `1 / x(i)` for `i < n`, all nonzero: Montgomery's trick on a block of
/// indices per task.
fn batch_inverse(n: usize, x: impl Fn(usize) -> BabyBear4 + Sync) -> Vec<BabyBear4> {
    const BLOCK: usize = 1 << 12;
    par_map(n.div_ceil(BLOCK), 1, |b| {
        let xs: Vec<BabyBear4> = (b * BLOCK..((b + 1) * BLOCK).min(n)).map(&x).collect();
        let mut prefix = Vec::with_capacity(xs.len());
        let mut acc = BabyBear4::ONE;
        for &x in &xs {
            prefix.push(acc);
            acc *= x;
        }
        let mut inv = acc.inverse().expect("all elements are nonzero");
        let mut out = vec![BabyBear4::ZERO; xs.len()];
        for i in (0..xs.len()).rev() {
            out[i] = inv * prefix[i];
            inv *= xs[i];
        }
        out
    })
    .concat()
}

/// The DEEP combination
/// `(A(x) - A(zeta)) / (x - zeta) + (B(x) - B(g zeta)) / (x - g zeta)`,
/// with `A = sum gamma^j p_j` over the columns opened at `zeta` (trace,
/// preprocessed, auxiliary, quotient) and `B` over those opened at `g zeta`
/// (trace, preprocessed, auxiliary), the powers of `gamma` continuing from
/// `A` into `B`.
struct Deep {
    zeta_powers: Vec<BabyBear4>,
    /// `gamma^K` with `K` the number of values at `zeta`: the powers at
    /// `g zeta` are the first powers at `zeta` times this.
    gamma_k: BabyBear4,
    a_at_zeta: BabyBear4,
    b_at_next: BabyBear4,
}

impl Deep {
    fn new(gamma: BabyBear4, at_zeta: &[&[BabyBear4]], at_next: &[&[BabyBear4]]) -> Self {
        let mut g = BabyBear4::ONE;
        let mut powers = |len: usize| -> Vec<BabyBear4> {
            (0..len)
                .map(|_| {
                    let p = g;
                    g *= gamma;
                    p
                })
                .collect()
        };
        let zeta_powers = powers(at_zeta.iter().map(|v| v.len()).sum());
        let next_powers = powers(at_next.iter().map(|v| v.len()).sum());
        let dot = |ps: &[BabyBear4], vs: &[&[BabyBear4]]| {
            ps.iter()
                .zip(vs.iter().flat_map(|v| v.iter()))
                .fold(BabyBear4::ZERO, |acc, (&p, &v)| acc + p * v)
        };
        Self {
            a_at_zeta: dot(&zeta_powers, at_zeta),
            b_at_next: dot(&next_powers, at_next),
            gamma_k: next_powers.first().copied().unwrap_or(g),
            zeta_powers,
        }
    }

    /// The combination at a point, from the rows there: `prefix` (the
    /// groups also opened at `g zeta`) then `rest`, and the inverses of
    /// `x - zeta` and `x - g zeta`.
    fn combine(
        &self,
        prefix: &[&[BabyBear]],
        rest: &[BabyBear],
        inv_zeta: BabyBear4,
        inv_next: BabyBear4,
    ) -> BabyBear4 {
        let mut powers = self.zeta_powers.iter();
        let mut dot = |vs: &[BabyBear]| {
            vs.iter()
                .zip(&mut powers)
                .fold(BabyBear4::ZERO, |acc, (&v, &p)| acc + p * v)
        };
        let a_prefix = prefix.iter().fold(BabyBear4::ZERO, |acc, v| acc + dot(v));
        let a = a_prefix + dot(rest);
        let b = self.gamma_k * a_prefix;
        (a - self.a_at_zeta) * inv_zeta + (b - self.b_at_next) * inv_next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::{Var, check_constraints};

    fn params() -> FriParams {
        FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        }
    }

    /// Runs `setup` first; seeded masks.
    fn prove<A: Air>(
        params: &FriParams,
        air: &A,
        trace: RowMajorMatrix,
        public: &[BabyBear],
    ) -> StarkProof {
        prove_seeded(params, air, trace, public, 1)
    }

    fn prove_seeded<A: Air>(
        params: &FriParams,
        air: &A,
        trace: RowMajorMatrix,
        public: &[BabyBear],
        seed: u64,
    ) -> StarkProof {
        let (pk, _) = setup(params, air);
        super::prove_with_rng(params, &pk, air, trace, public, &mut Rng::from_seed(seed))
    }

    /// Runs `setup` first.
    fn verify<A: Air>(
        params: &FriParams,
        air: &A,
        public: &[BabyBear],
        proof: &StarkProof,
    ) -> Result<(), StarkError> {
        let (_, vk) = setup(params, air);
        super::verify(params, &vk, air, public, proof)
    }

    /// Rows `(a, b)` with `a' = b`, `b' = a + b`; public `[a_0, b_0, b_last]`.
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

    /// One column with `x' = x^E + 1` on transitions; degree `E + 1`.
    struct PowerChain<const E: u64>;

    impl<const E: u64> Air for PowerChain<E> {
        fn width(&self) -> usize {
            1
        }
        fn max_degree(&self) -> usize {
            E as usize + 1
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let (x, x2) = (c.local()[0], c.next()[0]);
            let mut p = x;
            for _ in 1..E {
                p = p * x;
            }
            c.assert_zero_first_row(x - c.public()[0]);
            c.assert_zero_transition(x2 - (p + BabyBear::ONE));
        }
    }

    fn power_chain<const E: u64>(n: usize) -> RowMajorMatrix {
        let mut x = BabyBear::new(3);
        let values = (0..n)
            .map(|_| {
                let v = x;
                x = x.pow(E) + BabyBear::ONE;
                v
            })
            .collect();
        RowMajorMatrix::new(values, 1)
    }

    #[test]
    fn fibonacci_proves_and_verifies() {
        for log_n in [6, 8, 10] {
            let (trace, public) = fibonacci(1 << log_n);
            assert_eq!(check_constraints(&Fibonacci, &trace, &public), Ok(()));
            let proof = prove(&params(), &Fibonacci, trace, &public);
            assert_eq!(
                verify(&params(), &Fibonacci, &public, &proof),
                Ok(()),
                "n = 2^{log_n}"
            );
        }
    }

    #[test]
    fn higher_degree_constraints() {
        // Degree 4 and degree 8 (k = 8 = the blowup).
        let trace = power_chain::<3>(64);
        let public = [BabyBear::new(3)];
        let proof = prove(&params(), &PowerChain::<3>, trace, &public);
        // Degree 4: k = 4, so 8 chunks of 4 columns, plus R.
        assert_eq!(proof.quotient_at_zeta.len(), 4 * 8 + RANDOM_COLUMNS);
        assert_eq!(verify(&params(), &PowerChain::<3>, &public, &proof), Ok(()));

        let trace = power_chain::<7>(64);
        let proof = prove(&params(), &PowerChain::<7>, trace, &public);
        assert_eq!(proof.quotient_at_zeta.len(), 4 * 16 + RANDOM_COLUMNS);
        assert_eq!(verify(&params(), &PowerChain::<7>, &public, &proof), Ok(()));
    }

    #[test]
    fn rejects_wrong_public_values() {
        let (trace, public) = fibonacci(64);
        let proof = prove(&params(), &Fibonacci, trace, &public);
        let mut wrong = public.clone();
        wrong[2] += BabyBear::ONE;
        assert!(verify(&params(), &Fibonacci, &wrong, &proof).is_err());
    }

    #[test]
    fn rejects_invalid_traces() {
        // A wrong cell in the middle, the first row, and the last row.
        for row in [17, 0, 63] {
            let (mut trace, public) = fibonacci(64);
            trace.row_mut(row)[1] += BabyBear::ONE;
            assert!(check_constraints(&Fibonacci, &trace, &public).is_err());
            let proof = prove(&params(), &Fibonacci, trace, &public);
            assert!(
                verify(&params(), &Fibonacci, &public, &proof).is_err(),
                "row {row}"
            );
        }
    }

    #[test]
    fn rejects_understated_degree() {
        // Claiming degree 2 for a degree-4 AIR: the quotient is too big for
        // its chunks, so they don't recombine to it.
        struct Liar;
        impl Air for Liar {
            fn width(&self) -> usize {
                1
            }
            fn max_degree(&self) -> usize {
                2
            }
            fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
                PowerChain::<3>.eval(c);
            }
        }
        let trace = power_chain::<3>(64);
        let public = [BabyBear::new(3)];
        let proof = prove(&params(), &Liar, trace, &public);
        assert!(verify(&params(), &Liar, &public, &proof).is_err());
    }

    #[test]
    fn rejects_tampered_proofs() {
        let (trace, public) = fibonacci(256);
        let proof = prove(&params(), &Fibonacci, trace, &public);
        let check = |p: &StarkProof| verify(&params(), &Fibonacci, &public, p);
        assert_eq!(check(&proof), Ok(()));

        let mut bad = proof.clone();
        bad.trace_at_zeta[0] += BabyBear4::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.trace_at_next[1] += BabyBear4::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.quotient_at_zeta[3] += BabyBear4::ONE;
        assert_eq!(check(&bad), Err(StarkError::ConstraintMismatch));

        let mut bad = proof.clone();
        bad.openings[2].trace_row[0] += BabyBear::ONE;
        assert_eq!(check(&bad), Err(StarkError::BadOpening { query: 2 }));

        let mut bad = proof.clone();
        bad.openings[0].quotient_path[3][0] += BabyBear::ONE;
        assert_eq!(check(&bad), Err(StarkError::BadOpening { query: 0 }));

        let mut bad = proof.clone();
        bad.trace_commitment[0] += BabyBear::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.log_n += 1;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.openings.pop();
        assert_eq!(check(&bad), Err(StarkError::Shape));
    }

    #[test]
    fn proofs_are_randomized() {
        // Same statement, different masks: everything the proof reveals about
        // the trace changes, and both verify.
        let (trace, public) = fibonacci(256);
        let a = prove_seeded(&params(), &Fibonacci, trace.clone(), &public, 1);
        let b = prove_seeded(&params(), &Fibonacci, trace.clone(), &public, 2);
        assert_ne!(a.trace_commitment, b.trace_commitment);
        assert_ne!(a.quotient_commitment, b.quotient_commitment);
        assert_ne!(a.trace_at_zeta, b.trace_at_zeta);
        assert_ne!(a.quotient_at_zeta, b.quotient_at_zeta);
        assert_ne!(a.openings[0].trace_row, b.openings[0].trace_row);
        assert_eq!(verify(&params(), &Fibonacci, &public, &a), Ok(()));
        assert_eq!(verify(&params(), &Fibonacci, &public, &b), Ok(()));
        // The OS-seeded prover also verifies.
        let (pk, _) = setup(&params(), &Fibonacci);
        let c = super::prove(&params(), &pk, &Fibonacci, trace, &public);
        assert_eq!(verify(&params(), &Fibonacci, &public, &c), Ok(()));
    }

    #[test]
    fn masked_trace_matches_trace_on_trace_domain() {
        // Interpolating the masked rows and evaluating at g^i gives row i.
        let (trace, _) = fibonacci(16);
        let mut rng = Rng::from_seed(3);
        let mut masked = RowMajorMatrix::zeros(32, 2);
        for i in 0..16 {
            masked.row_mut(2 * i).copy_from_slice(trace.row(i));
            rng.fill(masked.row_mut(2 * i + 1));
        }
        ntt::intt(&mut masked);
        let g = BabyBear::two_adic_generator(4);
        for i in 0..16 {
            let at = eval_columns(&masked, BabyBear4::from(g.pow(i as u64)));
            for (v, &t) in at.iter().zip(trace.row(i)) {
                assert_eq!(v.as_base(), Some(t));
            }
        }
    }

    #[test]
    fn quotient_chunks_recombine_after_randomization() {
        // sum_i L_i(z) q_i'(z) = Q(z) for random-looking z, i.e. the
        // randomizers cancel.
        let layout = Layout::new(&Fibonacci, &params(), 6).unwrap();
        let m = 1usize << layout.log_quotient;
        // Q = an arbitrary polynomial of degree < M, through its values.
        let coeffs: Vec<BabyBear4> = (0..m as u32)
            .map(|i| BabyBear4::new([i, 2 * i + 1, 7, i * i]))
            .collect();
        let w_m = BabyBear::two_adic_generator(layout.log_quotient);
        let eval = |x: BabyBear4| {
            coeffs
                .iter()
                .rev()
                .fold(BabyBear4::ZERO, |acc, &c| acc * x + c)
        };
        let values: Vec<BabyBear4> = (0..m)
            .map(|j| eval(BabyBear4::from(SHIFT * w_m.pow(j as u64))))
            .collect();
        let out = randomized_chunks(&layout, &values, &mut Rng::from_seed(9));

        let z = BabyBear4::new([5, 6, 7, 8]);
        let at = eval_columns(&out, z);
        let s_n = layout.coset_powers();
        let z_n = z.pow(1 << layout.log_n);
        let mut q = BabyBear4::ZERO;
        for i in 0..layout.chunks {
            let q_i = (0..4).fold(BabyBear4::ZERO, |acc, b| acc + basis(b) * at[4 * i + b]);
            let (mut num, mut den) = (BabyBear4::ONE, BabyBear::ONE);
            for j in 0..layout.chunks {
                if j != i {
                    num *= z_n - s_n[j];
                    den *= s_n[i] - s_n[j];
                }
            }
            q += num * den.inverse().unwrap() * q_i;
        }
        assert_eq!(q, eval(z));
    }

    #[test]
    #[should_panic(expected = "too short to hide")]
    fn refuses_traces_too_short_to_hide() {
        let (trace, public) = fibonacci(8);
        prove(&params(), &Fibonacci, trace, &public);
    }

    /// `x' = 2x` on rows where the preprocessed selector is 1, `x' = x + 1`
    /// where it is 0; public `[x_0]`.
    struct Selected {
        pattern: fn(usize) -> bool,
        n: usize,
    }

    impl Air for Selected {
        fn width(&self) -> usize {
            1
        }
        fn max_degree(&self) -> usize {
            3
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let (x, x2) = (c.local()[0], c.next()[0]);
            let sel = c.preprocessed_local()[0];
            let doubled = x2 - (x + x);
            let incremented = x2 - (x + BabyBear::ONE);
            c.assert_zero_first_row(x - c.public()[0]);
            c.assert_zero_transition(sel * doubled + (V::from(BabyBear::ONE) - sel) * incremented);
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
        fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
            let values = (0..self.n)
                .map(|i| BabyBear::from((self.pattern)(i)))
                .collect();
            Some(RowMajorMatrix::new(values, 1))
        }
    }

    fn selected_trace(air: &Selected) -> RowMajorMatrix {
        let mut x = BabyBear::new(5);
        let values = (0..air.n)
            .map(|i| {
                let v = x;
                x = if (air.pattern)(i) {
                    x + x
                } else {
                    x + BabyBear::ONE
                };
                v
            })
            .collect();
        RowMajorMatrix::new(values, 1)
    }

    const ALTERNATING: fn(usize) -> bool = |i| i % 2 == 0;
    const EVERY_THIRD: fn(usize) -> bool = |i| i % 3 == 0;

    #[test]
    fn preprocessed_columns_prove_and_verify() {
        let air = Selected {
            pattern: ALTERNATING,
            n: 64,
        };
        let trace = selected_trace(&air);
        let public = [BabyBear::new(5)];
        assert_eq!(check_constraints(&air, &trace, &public), Ok(()));
        let proof = prove(&params(), &air, trace, &public);
        assert_eq!(proof.preprocessed_at_zeta.len(), 1);
        assert_eq!(verify(&params(), &air, &public, &proof), Ok(()));
    }

    #[test]
    fn rejects_proof_against_different_preprocessed_columns() {
        // A trace valid for one selector pattern, verified against another
        // pattern's key.
        let alt = Selected {
            pattern: ALTERNATING,
            n: 64,
        };
        let third = Selected {
            pattern: EVERY_THIRD,
            n: 64,
        };
        let public = [BabyBear::new(5)];
        let proof = prove(&params(), &alt, selected_trace(&alt), &public);
        let (_, third_vk) = setup(&params(), &third);
        assert!(super::verify(&params(), &third_vk, &third, &public, &proof).is_err());
        // And a trace following the wrong pattern fails under the right key.
        assert!(check_constraints(&alt, &selected_trace(&third), &public).is_err());
        let proof = prove(&params(), &alt, selected_trace(&third), &public);
        assert!(verify(&params(), &alt, &public, &proof).is_err());
    }

    #[test]
    fn rejects_tampered_preprocessed_values() {
        let air = Selected {
            pattern: ALTERNATING,
            n: 64,
        };
        let public = [BabyBear::new(5)];
        let proof = prove(&params(), &air, selected_trace(&air), &public);

        let mut bad = proof.clone();
        bad.openings[1].preprocessed_row[0] += BabyBear::ONE;
        assert_eq!(
            verify(&params(), &air, &public, &bad),
            Err(StarkError::BadOpening { query: 1 })
        );

        let mut bad = proof.clone();
        bad.preprocessed_at_zeta[0] += BabyBear4::ONE;
        assert!(verify(&params(), &air, &public, &bad).is_err());

        let mut bad = proof.clone();
        bad.preprocessed_at_next[0] += BabyBear4::ONE;
        assert!(verify(&params(), &air, &public, &bad).is_err());

        // The preprocessed columns fix the trace height.
        let mut bad = proof.clone();
        bad.log_n += 1;
        assert_eq!(
            verify(&params(), &air, &public, &bad),
            Err(StarkError::Shape)
        );
    }

    #[test]
    #[should_panic(expected = "preprocessed columns")]
    fn trace_height_must_match_preprocessed() {
        let air = Selected {
            pattern: ALTERNATING,
            n: 64,
        };
        let long = Selected {
            pattern: ALTERNATING,
            n: 128,
        };
        prove(&params(), &air, selected_trace(&long), &[BabyBear::new(5)]);
    }

    #[test]
    fn batch_inverse_matches_inverse() {
        // Spans several blocks.
        let xs: Vec<_> = (1..10_000u32)
            .map(|i| BabyBear4::new([i, i * 3, 7, i + 11]))
            .collect();
        let inv = batch_inverse(xs.len(), |i| xs[i]);
        assert_eq!(inv.len(), xs.len());
        for (x, y) in xs.iter().zip(inv) {
            assert_eq!(*x * y, BabyBear4::ONE);
        }
    }
}
