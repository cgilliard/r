//! Algebraic intermediate representations: a trace (one row per step) and
//! polynomial constraints between each row and the next.
//!
//! An [`Air`] writes its constraints once, generically over [`Var`]. The
//! prover evaluates them over base-field trace values on its quotient domain;
//! the verifier evaluates them over extension-field values at the
//! out-of-domain point. Constraints are combined with powers of a random
//! `alpha` into one value, which must vanish on every trace row.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::matrix::RowMajorMatrix;

/// A value constraints can be evaluated over: base field (prover),
/// extension field (verifier), or a symbolic wire (circuit compilation).
pub trait Var:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Add<BabyBear, Output = Self>
    + Sub<BabyBear, Output = Self>
    + Mul<BabyBear, Output = Self>
    + From<BabyBear>
{
    /// Extension-field values alongside `Self`: auxiliary columns,
    /// challenges and the combined constraint value.
    type Ext: Ext<Self>;
}

/// Extension-field arithmetic over a [`Var`]'s values.
pub trait Ext<V>:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Mul<V, Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + From<V>
    + From<BabyBear4>
{
}

impl Var for BabyBear {
    type Ext = BabyBear4;
}

impl Var for BabyBear4 {
    type Ext = BabyBear4;
}

impl Ext<BabyBear> for BabyBear4 {}
impl Ext<BabyBear4> for BabyBear4 {}

pub trait Air: Sync {
    /// Number of trace columns.
    fn width(&self) -> usize;

    /// Highest degree of any constraint in the trace values, counting each
    /// row selector (`*_first_row`, `*_last_row`, `*_transition`) as one more.
    fn max_degree(&self) -> usize;

    /// Emits every constraint. The order must not depend on the values.
    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>);

    /// Number of preprocessed columns: fixed, public columns that are the
    /// same in every proof, such as selectors giving each row its role.
    fn preprocessed_width(&self) -> usize {
        0
    }

    /// The preprocessed columns, one row per trace row. Their height fixes
    /// the trace height. Committed once by `stark::setup`.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
        None
    }

    /// Extension-field challenges drawn after the main trace is committed,
    /// for the auxiliary trace (e.g. lookup arguments).
    fn num_challenges(&self) -> usize {
        0
    }

    /// Extension-field columns of the auxiliary trace, built from the main
    /// trace and the challenges, and committed after them.
    fn aux_width(&self) -> usize {
        0
    }

    /// The auxiliary trace, row-major: `aux_width()` values per trace row.
    fn aux_trace(
        &self,
        _main: &RowMajorMatrix,
        _preprocessed: Option<&RowMajorMatrix>,
        _challenges: &[BabyBear4],
    ) -> Vec<BabyBear4> {
        Vec::new()
    }
}

/// Row selectors at one point. On trace rows they are nonzero exactly where
/// they apply; elsewhere they are the polynomials that interpolate that.
#[derive(Clone, Copy, Debug)]
pub struct Selectors<V> {
    pub is_first_row: V,
    pub is_last_row: V,
    pub is_transition: V,
}

/// The trace and preprocessed rows at a point and at its successor.
#[derive(Clone, Copy)]
pub struct Rows<'a, V: Var> {
    pub local: &'a [V],
    pub next: &'a [V],
    pub preprocessed_local: &'a [V],
    pub preprocessed_next: &'a [V],
    pub aux_local: &'a [V::Ext],
    pub aux_next: &'a [V::Ext],
}

/// Collects constraint values at one point, as `sum alpha^k c_k`.
pub struct Constraints<'a, V: Var> {
    local: &'a [V],
    next: &'a [V],
    preprocessed_local: &'a [V],
    preprocessed_next: &'a [V],
    aux_local: &'a [V::Ext],
    aux_next: &'a [V::Ext],
    challenges: &'a [V::Ext],
    public: &'a [V],
    selectors: Selectors<V>,
    alpha: V::Ext,
    acc: V::Ext,
    /// `alpha^(K - 1 - i)` for constraint `i` of `K`, if precomputed: then
    /// each constraint costs a multiplication by a base value instead of a
    /// full extension multiplication.
    powers: Option<&'a [V::Ext]>,
    count: usize,
    /// Every constraint value in order, when debugging.
    log: Option<Vec<V::Ext>>,
}

impl<'a, V: Var> Constraints<'a, V> {
    pub fn new(
        rows: Rows<'a, V>,
        challenges: &'a [V::Ext],
        public: &'a [V],
        selectors: Selectors<V>,
        alpha: V::Ext,
    ) -> Self {
        Self {
            local: rows.local,
            next: rows.next,
            preprocessed_local: rows.preprocessed_local,
            preprocessed_next: rows.preprocessed_next,
            aux_local: rows.aux_local,
            aux_next: rows.aux_next,
            challenges,
            public,
            selectors,
            alpha,
            acc: V::Ext::from(BabyBear4::ZERO),
            powers: None,
            count: 0,
            log: None,
        }
    }

    /// Combines with precomputed powers of `alpha` (see
    /// [`Constraints::num_constraints`]); same result as without.
    pub fn with_powers(mut self, powers: &'a [V::Ext]) -> Self {
        self.powers = Some(powers);
        self
    }

    /// Constraints asserted so far.
    pub fn num_constraints(&self) -> usize {
        self.count
    }

    /// The current row.
    pub fn local(&self) -> &'a [V] {
        self.local
    }

    /// The next row (wrapping from the last row to the first).
    pub fn next(&self) -> &'a [V] {
        self.next
    }

    /// The current row of the preprocessed columns.
    pub fn preprocessed_local(&self) -> &'a [V] {
        self.preprocessed_local
    }

    /// The next row of the preprocessed columns.
    pub fn preprocessed_next(&self) -> &'a [V] {
        self.preprocessed_next
    }

    pub fn public(&self) -> &'a [V] {
        self.public
    }

    /// The current row of the auxiliary trace.
    pub fn aux_local(&self) -> &'a [V::Ext] {
        self.aux_local
    }

    /// The next row of the auxiliary trace.
    pub fn aux_next(&self) -> &'a [V::Ext] {
        self.aux_next
    }

    /// The challenges the auxiliary trace was built with.
    pub fn challenges(&self) -> &'a [V::Ext] {
        self.challenges
    }

    /// Requires the extension-field value `x = 0` on every row.
    pub fn assert_zero_ext(&mut self, x: V::Ext) {
        if let Some(log) = &mut self.log {
            log.push(x);
        }
        self.acc = match self.powers {
            Some(p) => self.acc + p[self.count] * x,
            None => self.acc * self.alpha + x,
        };
        self.count += 1;
    }

    pub fn assert_zero_ext_first_row(&mut self, x: V::Ext) {
        let s = V::Ext::from(self.selectors.is_first_row);
        self.assert_zero_ext(s * x);
    }

    pub fn assert_zero_ext_last_row(&mut self, x: V::Ext) {
        let s = V::Ext::from(self.selectors.is_last_row);
        self.assert_zero_ext(s * x);
    }

    pub fn assert_zero_ext_transition(&mut self, x: V::Ext) {
        let s = V::Ext::from(self.selectors.is_transition);
        self.assert_zero_ext(s * x);
    }

    /// Requires `x = 0` on every row.
    pub fn assert_zero(&mut self, x: V) {
        match self.powers {
            Some(p) if self.log.is_none() => {
                self.acc += p[self.count] * x;
                self.count += 1;
            }
            _ => self.assert_zero_ext(V::Ext::from(x)),
        }
    }

    pub fn assert_eq(&mut self, a: V, b: V) {
        self.assert_zero(a - b);
    }

    /// Requires `x = 0` on the first row.
    pub fn assert_zero_first_row(&mut self, x: V) {
        self.assert_zero(self.selectors.is_first_row * x);
    }

    /// Requires `x = 0` on the last row.
    pub fn assert_zero_last_row(&mut self, x: V) {
        self.assert_zero(self.selectors.is_last_row * x);
    }

    /// Requires `x = 0` on every row except the last, i.e. between each row
    /// and its successor.
    pub fn assert_zero_transition(&mut self, x: V) {
        self.assert_zero(self.selectors.is_transition * x);
    }

    /// The combined value of all constraints so far.
    pub fn finish(self) -> V::Ext {
        self.acc
    }
}

/// Selectors at a point `x` for a trace domain of size `n` with generator
/// `g`, using the unnormalized forms `Z_H(x) / (x - 1)`,
/// `Z_H(x) / (x - g^-1)` and `x - g^-1`, where `Z_H(x) = x^n - 1`.
/// Returns `None` if `x` is in the trace domain, where these are undefined.
pub fn selectors_at<V: Var + PartialEq>(
    x: V,
    log_n: u32,
    inverse: impl Fn(V) -> Option<V>,
) -> Option<(Selectors<V>, V)> {
    let mut x_n = x;
    for _ in 0..log_n {
        x_n = x_n * x_n;
    }
    let z_h = x_n - BabyBear::ONE;
    let g_inv = BabyBear::two_adic_generator(log_n)
        .inverse()
        .expect("generators are nonzero");
    let x_minus_one = x - BabyBear::ONE;
    let x_minus_g_inv = x - g_inv;
    Some((
        Selectors {
            is_first_row: z_h * inverse(x_minus_one)?,
            is_last_row: z_h * inverse(x_minus_g_inv)?,
            is_transition: x_minus_g_inv,
        },
        z_h,
    ))
}

/// Fixed challenges used when checking constraints directly. Cancellation
/// at them is as unlikely as at random ones.
fn check_challenges<A: Air>(air: &A) -> Vec<BabyBear4> {
    (0..air.num_challenges() as u32)
        .map(|i| BabyBear4::new([0x0bad_cafe + i, 0x1357_9bdf, 0x2468_ace0 + i, 0x0f1e_2d3c]))
        .collect()
}

/// Everything needed to evaluate an AIR's constraints on its trace rows.
struct Checked<'a> {
    trace: &'a RowMajorMatrix,
    pre: Option<&'a RowMajorMatrix>,
    aux: &'a [BabyBear4],
    challenges: &'a [BabyBear4],
    public: &'a [BabyBear],
}

impl Checked<'_> {
    /// Evaluates every constraint on trace row `i`, logging each value.
    fn eval_row<A: Air>(&self, air: &A, alpha: BabyBear4, i: usize) -> (BabyBear4, Vec<BabyBear4>) {
        let n = self.trace.height();
        let w = air.aux_width();
        let pre_row = |i: usize| self.pre.map_or(&[][..], |p| p.row(i % n));
        let aux_row = |i: usize| &self.aux[(i % n) * w..(i % n) * w + w];
        let selectors = Selectors {
            is_first_row: BabyBear::from(i == 0),
            is_last_row: BabyBear::from(i == n - 1),
            is_transition: BabyBear::from(i != n - 1),
        };
        let rows = Rows {
            local: self.trace.row(i),
            next: self.trace.row((i + 1) % n),
            preprocessed_local: pre_row(i),
            preprocessed_next: pre_row(i + 1),
            aux_local: aux_row(i),
            aux_next: aux_row(i + 1),
        };
        let mut c = Constraints::new(rows, self.challenges, self.public, selectors, alpha);
        c.log = Some(Vec::new());
        air.eval(&mut c);
        let log = c.log.take().unwrap_or_default();
        (c.finish(), log)
    }
}

/// The indices (in emission order) of the constraints that are nonzero on
/// trace row `i`. For debugging circuits.
pub fn failing_constraints<A: Air>(
    air: &A,
    trace: &RowMajorMatrix,
    public: &[BabyBear],
    i: usize,
) -> Vec<usize> {
    let pre = air.preprocessed_trace();
    let challenges = check_challenges(air);
    let aux = air.aux_trace(trace, pre.as_ref(), &challenges);
    let checked = Checked {
        trace,
        pre: pre.as_ref(),
        aux: &aux,
        challenges: &challenges,
        public,
    };
    let (_, log) = checked.eval_row(air, BabyBear4::ONE, i);
    log.iter()
        .enumerate()
        .filter(|(_, v)| **v != BabyBear4::ZERO)
        .map(|(k, _)| k)
        .collect()
}

/// Checks every constraint directly on the trace rows. For tests and
/// debugging: a prover given a bad trace produces a proof that fails to
/// verify, and this says which row is wrong.
pub fn check_constraints<A: Air>(
    air: &A,
    trace: &RowMajorMatrix,
    public: &[BabyBear],
) -> Result<(), usize> {
    let pre = air.preprocessed_trace();
    let challenges = check_challenges(air);
    let aux = air.aux_trace(trace, pre.as_ref(), &challenges);
    assert_eq!(
        aux.len(),
        trace.height() * air.aux_width(),
        "aux trace shape"
    );
    // A fixed random-looking alpha: nonzero constraints cancelling out at it
    // is as unlikely as for a random one.
    let alpha = BabyBear4::new([0x1234_5678, 0x2345_6789, 0x3456_789a, 0x4567_89ab]);
    let checked = Checked {
        trace,
        pre: pre.as_ref(),
        aux: &aux,
        challenges: &challenges,
        public,
    };
    for i in 0..trace.height() {
        let (acc, _) = checked.eval_row(air, alpha, i);
        if acc != BabyBear4::ZERO {
            return Err(i);
        }
    }
    Ok(())
}
