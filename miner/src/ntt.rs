//! Number-theoretic transforms over BabyBear, applied to every column of a
//! row-major matrix at once.
//!
//! For a matrix of height `n = 2^k` and `w = BabyBear::two_adic_generator(k)`:
//! [`ntt`] maps coefficients (row `i` = coefficient of `X^i`) to evaluations
//! (row `i` = value at `w^i`), and [`intt`] maps back. [`coset_lde`]
//! re-evaluates over a larger shifted domain, as STARKs need for their
//! low-degree extensions.

use crate::babybear::{BabyBear, TWO_ADICITY};
use crate::matrix::RowMajorMatrix;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::par::threads;

/// Eight adjacent columns' values in one row: transforms work on blocks of
/// columns, eight lanes at a time.
type Block = [BabyBear; 8];

/// Coefficients to evaluations over the subgroup of order `height`.
pub fn ntt(m: &mut RowMajorMatrix) {
    let log_n = log2_height(m);
    let tw = Twiddles::new(log_n, BabyBear::two_adic_generator(log_n));
    *m = map_columns(m, m.height(), |col, out| {
        tw.transform(col);
        out.rows(col, 0, 1);
    });
}

/// Evaluations over the subgroup of order `height` to coefficients.
pub fn intt(m: &mut RowMajorMatrix) {
    coset_intt(m, BabyBear::ONE);
}

/// Low-degree extension: given evaluations over the subgroup `H` of order
/// `n`, returns the same polynomials evaluated over `shift * H'`, where `H'`
/// has order `n << log_blowup`. Row `i` of the result is the value at
/// `shift * w'^i`.
pub fn coset_lde(evals: &RowMajorMatrix, log_blowup: u32, shift: BabyBear) -> RowMajorMatrix {
    let mut coeffs = evals.clone();
    intt(&mut coeffs);
    coset_extend(coeffs, log_blowup, shift)
}

/// Coefficients to evaluations over `shift * H'`, where `H'` has order
/// `height << log_blowup` (the coefficients are zero-padded).
///
/// With `B = 2^log_blowup` and `w'` of order `nB`, row `jB + k` is
/// `p(shift w'^k w_n^j)`: for each `k`, an order-`n` transform of the
/// coefficients scaled by `(shift w'^k)^t`.
pub fn coset_extend(coeffs: RowMajorMatrix, log_blowup: u32, shift: BabyBear) -> RowMajorMatrix {
    let log_n = log2_height(&coeffs);
    let (n, blowup) = (coeffs.height(), 1usize << log_blowup);
    assert!(
        log_n + log_blowup <= TWO_ADICITY,
        "LDE exceeds the field's 2-adicity"
    );
    let tw = Twiddles::new(log_n, BabyBear::two_adic_generator(log_n));
    let w_big = BabyBear::two_adic_generator(log_n + log_blowup);
    // Row-scaling powers (shift w'^k)^t for each coset k.
    let powers: Vec<Vec<BabyBear>> = (0..blowup)
        .map(|k| powers(shift * w_big.pow(k as u64), n))
        .collect();
    let (tw, powers) = (&tw, &powers);
    // Each worker clones the closure, and so has its own buffer.
    let mut buf = Vec::new();
    map_columns(&coeffs, n * blowup, move |col, out| {
        buf.resize(n, Block::default());
        for (k, pw) in powers.iter().enumerate() {
            buf.copy_from_slice(col);
            scale_rows(&mut buf, pw);
            tw.transform(&mut buf);
            out.rows(&buf, k, blowup);
        }
    })
}

/// Evaluations over `shift * H` to coefficients.
pub fn coset_intt(m: &mut RowMajorMatrix, shift: BabyBear) {
    let log_n = log2_height(m);
    let root = BabyBear::two_adic_generator(log_n);
    let tw = Twiddles::new(log_n, root.inverse().expect("roots are nonzero"));
    // These are coefficients of p(shift * X), scaled by n; undo both.
    let n_inv = BabyBear::inv_two_pow(log_n);
    let mut scale = powers(shift.inverse().expect("shift is nonzero"), m.height());
    for s in &mut scale {
        *s *= n_inv;
    }
    *m = map_columns(m, m.height(), |col, out| {
        tw.transform(col);
        scale_rows(col, &scale);
        out.rows(col, 0, 1);
    });
}

/// `1, base, base^2, ...` (`n` of them).
fn powers(base: BabyBear, n: usize) -> Vec<BabyBear> {
    let mut v = Vec::with_capacity(n);
    let mut x = BabyBear::ONE;
    for _ in 0..n {
        v.push(x);
        x *= base;
    }
    v
}

fn log2_height(m: &RowMajorMatrix) -> u32 {
    let n = m.height();
    assert!(n.is_power_of_two(), "height must be a power of two");
    let log_n = n.trailing_zeros();
    assert!(log_n <= TWO_ADICITY, "height exceeds the field's 2-adicity");
    log_n
}

/// A new matrix of height `out_height` whose columns come from `m`'s,
/// eight at a time: `f` gets a block of columns (zero-padded past the last
/// column) to work in, and writes the block's output rows through
/// [`BlockOut`]. Workers take blocks in turn and write straight into the
/// output, each to its own columns.
fn map_columns<F>(m: &RowMajorMatrix, out_height: usize, f: F) -> RowMajorMatrix
where
    F: FnMut(&mut [Block], &mut BlockOut<'_>) + Clone + Send,
{
    let w = m.width;
    let len = out_height * w;
    if len == 0 {
        return RowMajorMatrix::zeros(out_height, w);
    }
    let mut values: Vec<BabyBear> = Vec::with_capacity(len);
    let out = SharedOut(values.as_mut_ptr());
    let blocks = w.div_ceil(8);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads().min(blocks) {
            let mut f = f.clone();
            let (out, next) = (&out, &next);
            s.spawn(move || {
                let mut col = vec![Block::default(); m.height()];
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= blocks {
                        break;
                    }
                    let c0 = 8 * b;
                    let lanes = (w - c0).min(8);
                    for (x, row) in col.iter_mut().zip(m.values.chunks_exact(w)) {
                        x[..lanes].copy_from_slice(&row[c0..c0 + lanes]);
                    }
                    let mut o = BlockOut {
                        out,
                        width: w,
                        height: out_height,
                        c0,
                        lanes,
                        written: 0,
                    };
                    f(&mut col, &mut o);
                    assert_eq!(o.written, out_height, "every output row written");
                }
            });
        }
    });
    // SAFETY: the blocks cover every column of every row.
    unsafe { values.set_len(len) };
    RowMajorMatrix::new(values, w)
}

/// The output buffer, written by several workers at disjoint positions.
struct SharedOut(*mut BabyBear);

// SAFETY: workers write disjoint elements, and the buffer outlives them.
unsafe impl Sync for SharedOut {}

/// Where one block of columns goes in the output.
struct BlockOut<'a> {
    out: &'a SharedOut,
    width: usize,
    height: usize,
    c0: usize,
    lanes: usize,
    /// Rows written so far, to check every row is.
    written: usize,
}

impl BlockOut<'_> {
    /// Writes `rows[j]` to output row `first + j * step`.
    fn rows(&mut self, rows: &[Block], first: usize, step: usize) {
        assert!(rows.is_empty() || first + (rows.len() - 1) * step < self.height);
        for (j, x) in rows.iter().enumerate() {
            let at = (first + j * step) * self.width + self.c0;
            // SAFETY: the row is in bounds (checked above), and only this
            // block writes columns `c0 .. c0 + lanes`.
            unsafe { core::ptr::copy_nonoverlapping(x.as_ptr(), self.out.0.add(at), self.lanes) };
        }
        self.written += rows.len();
    }
}

/// Multiplies row `i` of `a` by `s[i]`.
fn scale_rows(a: &mut [Block], s: &[BabyBear]) {
    #[cfg(target_arch = "x86_64")]
    if crate::packed::avx2_available() {
        // SAFETY: AVX2 is available.
        unsafe { avx2::scale_rows(a, s) };
        return;
    }
    for (x, &k) in a.iter_mut().zip(s) {
        for v in x {
            *v *= k;
        }
    }
}

/// Powers of a root of order `2^log_n`, for transforms of that size.
struct Twiddles {
    log_n: u32,
    /// `root^j` for `j < n / 2`.
    powers: Vec<BabyBear>,
}

impl Twiddles {
    fn new(log_n: u32, root: BabyBear) -> Self {
        Self {
            log_n,
            powers: powers(root, (1usize << log_n) / 2),
        }
    }

    /// Iterative radix-2 Cooley-Tukey on a block of columns: bit-reverse
    /// the rows, then combine pairs of rows in stages of doubling size.
    fn transform(&self, a: &mut [Block]) {
        let n = a.len();
        debug_assert_eq!(n, 1 << self.log_n);
        if n <= 1 {
            return;
        }
        let shift = usize::BITS - self.log_n;
        for i in 0..n {
            let j = i.reverse_bits() >> shift;
            if i < j {
                a.swap(i, j);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if crate::packed::avx2_available() {
            // SAFETY: AVX2 is available.
            unsafe { avx2::stages(a, &self.powers) };
            return;
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for block in a.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(half);
                for (j, (x, y)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let w = self.powers[j * stride];
                    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
                        let t = *y * w;
                        *y = *x - t;
                        *x += t;
                    }
                }
            }
            len *= 2;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Block;
    use crate::babybear::BabyBear;
    use crate::packed::avx2::{add, load, mul, splat, store, sub};

    /// The butterfly stages of [`super::Twiddles::transform`].
    #[target_feature(enable = "avx2")]
    pub(super) fn stages(a: &mut [Block], powers: &[BabyBear]) {
        let n = a.len();
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for block in a.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(half);
                for (j, (x, y)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let t = mul(load(y), splat(powers[j * stride].raw()));
                    let u = load(x);
                    store(x, add(u, t));
                    store(y, sub(u, t));
                }
            }
            len *= 2;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn scale_rows(a: &mut [Block], s: &[BabyBear]) {
        for (x, k) in a.iter_mut().zip(s) {
            store(x, mul(load(x), splat(k.raw())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;

    fn random_matrix(height: usize, width: usize, seed: u64) -> RowMajorMatrix {
        let mut s = seed | 1;
        let values = (0..height * width)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                BabyBear::new((s % P as u64) as u32)
            })
            .collect();
        RowMajorMatrix::new(values, width)
    }

    /// Evaluates column `col` of `coeffs`, read as a polynomial, at `x`.
    fn eval(coeffs: &RowMajorMatrix, col: usize, x: BabyBear) -> BabyBear {
        coeffs
            .column(col)
            .iter()
            .rev()
            .fold(BabyBear::ZERO, |acc, &c| acc * x + c)
    }

    #[test]
    fn ntt_matches_naive_evaluation() {
        for log_n in 0..=6 {
            let n = 1 << log_n;
            let coeffs = random_matrix(n, 3, log_n as u64 + 1);
            let mut evals = coeffs.clone();
            ntt(&mut evals);
            let w = BabyBear::two_adic_generator(log_n);
            for i in 0..n {
                for col in 0..3 {
                    assert_eq!(evals.row(i)[col], eval(&coeffs, col, w.pow(i as u64)));
                }
            }
        }
    }

    #[test]
    fn intt_inverts_ntt() {
        for log_n in [0, 1, 2, 5, 10, 12] {
            let original = random_matrix(1 << log_n, 5, 99 + log_n as u64);
            let mut m = original.clone();
            ntt(&mut m);
            intt(&mut m);
            assert_eq!(m, original);
        }
    }

    #[test]
    fn coset_lde_matches_naive_evaluation() {
        let (log_n, log_blowup) = (4, 3);
        let evals = random_matrix(1 << log_n, 2, 7);
        let shift = BabyBear::new(31);
        let lde = coset_lde(&evals, log_blowup, shift);
        assert_eq!(lde.height(), 1 << (log_n + log_blowup));

        let mut coeffs = evals.clone();
        intt(&mut coeffs);
        let w = BabyBear::two_adic_generator(log_n + log_blowup);
        for i in 0..lde.height() {
            let x = shift * w.pow(i as u64);
            for col in 0..2 {
                assert_eq!(lde.row(i)[col], eval(&coeffs, col, x));
            }
        }
    }

    #[test]
    fn coset_intt_inverts_coset_extend() {
        let coeffs = random_matrix(16, 3, 21);
        let shift = BabyBear::new(31);
        let mut evals = coset_extend(coeffs.clone(), 0, shift);
        coset_intt(&mut evals, shift);
        assert_eq!(evals, coeffs);
    }

    #[test]
    fn unshifted_lde_contains_original_evaluations() {
        let log_blowup = 2;
        let evals = random_matrix(64, 4, 3);
        let lde = coset_lde(&evals, log_blowup, BabyBear::ONE);
        for i in 0..evals.height() {
            assert_eq!(lde.row(i << log_blowup), evals.row(i));
        }
    }

    #[test]
    fn low_degree_extension_has_low_degree() {
        // Interpolating the LDE recovers the original coefficients, with
        // zeros above the original degree.
        let evals = random_matrix(32, 1, 11);
        let mut coeffs = evals.clone();
        intt(&mut coeffs);
        let mut lde = coset_lde(&evals, 2, BabyBear::ONE);
        intt(&mut lde);
        assert_eq!(&lde.values[..32], coeffs.values.as_slice());
        assert!(lde.values[32..].iter().all(|&c| c == BabyBear::ZERO));
    }
}
