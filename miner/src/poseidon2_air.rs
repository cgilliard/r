//! Poseidon2 as AIR columns: one full permutation per row, every constraint
//! of degree 3. The layout follows Plonky3's Poseidon2 AIR:
//!
//! - 16 input columns;
//! - per full round, 16 S-box registers `x^3` and the 16-element state after
//!   the round's linear layer: `reg = (s + rc)^3`, `post = M (reg^2 (s + rc))`;
//! - per partial round, the register and the S-box output for element 0;
//!   the other 15 elements stay linear expressions, not columns.
//!
//! [`eval`] and [`generate`] work on a column slice, so larger circuits can
//! embed the permutation; [`Poseidon2Air`] is the standalone form.

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::matrix::RowMajorMatrix;
use crate::poseidon2::{
    HALF_FULL_ROUNDS, INTERNAL_DIAG, PARTIAL_ROUNDS, RC_EXTERNAL_FINAL, RC_EXTERNAL_INITIAL,
    RC_INTERNAL, WIDTH,
};

const FULL_ROUND_COLS: usize = 2 * WIDTH;
const PARTIAL_ROUND_COLS: usize = 2;
const BEGIN_OFFSET: usize = WIDTH;
const PARTIAL_OFFSET: usize = BEGIN_OFFSET + HALF_FULL_ROUNDS * FULL_ROUND_COLS;
const END_OFFSET: usize = PARTIAL_OFFSET + PARTIAL_ROUNDS * PARTIAL_ROUND_COLS;

/// Columns used by one permutation.
pub const NUM_COLS: usize = END_OFFSET + HALF_FULL_ROUNDS * FULL_ROUND_COLS;

/// Where the permutation's output sits within its columns.
pub const OUTPUT_OFFSET: usize = NUM_COLS - WIDTH;

/// Constrains `cols` (at least [`NUM_COLS`] long) to be one Poseidon2
/// permutation of its first 16 columns, and returns the output.
pub fn eval<V: Var>(c: &mut Constraints<'_, V>, cols: &[V]) -> [V; WIDTH] {
    eval_gated(c, cols, None)
}

/// [`eval`] with every constraint multiplied by `gate`, so the columns are
/// free wherever `gate` is 0. Degree 4 instead of 3.
pub fn eval_gated<V: Var>(c: &mut Constraints<'_, V>, cols: &[V], gate: Option<V>) -> [V; WIDTH] {
    let mut g = Gate { c, gate };
    let mut state: [V; WIDTH] = core::array::from_fn(|i| cols[i]);
    external_linear_layer(&mut state);
    for (r, rc) in RC_EXTERNAL_INITIAL.iter().enumerate() {
        state = eval_full_round(
            &mut g,
            &cols[BEGIN_OFFSET + r * FULL_ROUND_COLS..],
            &state,
            rc,
        );
    }
    for (r, &rc) in RC_INTERNAL.iter().enumerate() {
        let col = &cols[PARTIAL_OFFSET + r * PARTIAL_ROUND_COLS..];
        state[0] = eval_sbox(&mut g, col[0], col[1], state[0] + rc);
        internal_linear_layer(&mut state);
    }
    for (r, rc) in RC_EXTERNAL_FINAL.iter().enumerate() {
        state = eval_full_round(
            &mut g,
            &cols[END_OFFSET + r * FULL_ROUND_COLS..],
            &state,
            rc,
        );
    }
    state
}

/// Constraints, optionally multiplied by a gate.
struct Gate<'a, 'b, V: Var> {
    c: &'a mut Constraints<'b, V>,
    gate: Option<V>,
}

impl<V: Var> Gate<'_, '_, V> {
    fn assert_eq(&mut self, a: V, b: V) {
        match self.gate {
            Some(g) => self.c.assert_zero(g * (a - b)),
            None => self.c.assert_eq(a, b),
        }
    }
}

/// Constrains one full round and returns its committed output.
fn eval_full_round<V: Var>(
    c: &mut Gate<'_, '_, V>,
    cols: &[V],
    state: &[V; WIDTH],
    rc: &[BabyBear; WIDTH],
) -> [V; WIDTH] {
    let (regs, post) = (&cols[..WIDTH], &cols[WIDTH..FULL_ROUND_COLS]);
    let mut out: [V; WIDTH] = core::array::from_fn(|i| {
        let x = state[i] + rc[i];
        c.assert_eq(regs[i], x * x * x);
        regs[i] * regs[i] * x
    });
    external_linear_layer(&mut out);
    for (&p, &o) in post.iter().zip(&out) {
        c.assert_eq(p, o);
    }
    core::array::from_fn(|i| post[i])
}

/// Constrains `reg = x^3` and `out = x^7`, and returns `out`.
fn eval_sbox<V: Var>(c: &mut Gate<'_, '_, V>, reg: V, out: V, x: V) -> V {
    c.assert_eq(reg, x * x * x);
    c.assert_eq(out, reg * reg * x);
    out
}

/// Fills `cols` (at least [`NUM_COLS`] long) with the permutation of `input`
/// and returns the output.
pub fn generate(input: [BabyBear; WIDTH], cols: &mut [BabyBear]) -> [BabyBear; WIDTH] {
    cols[..WIDTH].copy_from_slice(&input);
    let mut state = input;
    external_linear_layer(&mut state);
    for (r, rc) in RC_EXTERNAL_INITIAL.iter().enumerate() {
        state = generate_full_round(&mut cols[BEGIN_OFFSET + r * FULL_ROUND_COLS..], &state, rc);
    }
    for (r, &rc) in RC_INTERNAL.iter().enumerate() {
        let col = &mut cols[PARTIAL_OFFSET + r * PARTIAL_ROUND_COLS..];
        let x = state[0] + rc;
        col[0] = x * x * x;
        col[1] = col[0] * col[0] * x;
        state[0] = col[1];
        internal_linear_layer(&mut state);
    }
    for (r, rc) in RC_EXTERNAL_FINAL.iter().enumerate() {
        state = generate_full_round(&mut cols[END_OFFSET + r * FULL_ROUND_COLS..], &state, rc);
    }
    state
}

fn generate_full_round(
    cols: &mut [BabyBear],
    state: &[BabyBear; WIDTH],
    rc: &[BabyBear; WIDTH],
) -> [BabyBear; WIDTH] {
    let mut out = [BabyBear::ZERO; WIDTH];
    for i in 0..WIDTH {
        let x = state[i] + rc[i];
        cols[i] = x * x * x;
        out[i] = cols[i] * cols[i] * x;
    }
    external_linear_layer(&mut out);
    cols[WIDTH..FULL_ROUND_COLS].copy_from_slice(&out);
    out
}

/// The external linear layer `circ(2 M4, M4, M4, M4)`, generic over the
/// value type (see `poseidon2::external_linear_layer`).
fn external_linear_layer<V: Var>(state: &mut [V; WIDTH]) {
    for chunk in state.chunks_exact_mut(4) {
        let x: [V; 4] = core::array::from_fn(|i| chunk[i]);
        let t01 = x[0] + x[1];
        let t23 = x[2] + x[3];
        let t0123 = t01 + t23;
        let t01123 = t0123 + x[1];
        let t01233 = t0123 + x[3];
        chunk[3] = t01233 + x[0] + x[0];
        chunk[1] = t01123 + x[2] + x[2];
        chunk[0] = t01123 + t01;
        chunk[2] = t01233 + t23;
    }
    let sums: [V; 4] =
        core::array::from_fn(|k| state[k] + state[k + 4] + state[k + 8] + state[k + 12]);
    for (i, s) in state.iter_mut().enumerate() {
        *s = *s + sums[i % 4];
    }
}

/// The internal linear layer `1 + Diag(V)`.
fn internal_linear_layer<V: Var>(state: &mut [V; WIDTH]) {
    let sum = state[1..].iter().fold(state[0], |acc, &x| acc + x);
    for (s, &d) in state.iter_mut().zip(&INTERNAL_DIAG) {
        *s = sum + *s * d;
    }
}

/// A trace of independent permutations, one per row.
pub struct Poseidon2Air;

impl Air for Poseidon2Air {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn max_degree(&self) -> usize {
        3
    }

    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
        let local = c.local();
        eval(c, local);
    }
}

/// The trace for [`Poseidon2Air`]: one row per input.
pub fn trace(inputs: &[[BabyBear; WIDTH]]) -> RowMajorMatrix {
    let mut m = RowMajorMatrix::zeros(inputs.len(), NUM_COLS);
    for (row, &input) in m.rows_mut().zip(inputs) {
        generate(input, row);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::fri::FriParams;
    use crate::poseidon2;
    use crate::rng::Rng;
    use crate::stark;

    fn inputs(n: usize) -> Vec<[BabyBear; WIDTH]> {
        let mut rng = Rng::from_seed(5);
        (0..n)
            .map(|_| core::array::from_fn(|_| rng.next_element()))
            .collect()
    }

    #[test]
    fn layout_has_expected_width() {
        assert_eq!(NUM_COLS, 298);
    }

    #[test]
    fn generated_output_matches_permutation() {
        let mut cols = vec![BabyBear::ZERO; NUM_COLS];
        for input in inputs(10) {
            let out = generate(input, &mut cols);
            let mut expected = input;
            poseidon2::permute(&mut expected);
            assert_eq!(out, expected);
            assert_eq!(&cols[OUTPUT_OFFSET..], &expected);
        }
    }

    #[test]
    fn plonky3_vector_through_air() {
        // The AIR's generic linear layers, run on base-field values,
        // reproduce Plonky3's known-answer vector.
        let input = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);
        let expected = BabyBear::new_array([
            516096821, 90309867, 1101817252, 1660784290, 360715097, 1789519026, 1788910906,
            563338433, 319524748, 1741414159, 1650859320, 894311162, 1121347488, 1692793758,
            1052633829, 1344246938,
        ]);
        let mut cols = vec![BabyBear::ZERO; NUM_COLS];
        assert_eq!(generate(input, &mut cols), expected);
    }

    #[test]
    fn constraints_hold_on_generated_trace() {
        let t = trace(&inputs(16));
        assert_eq!(check_constraints(&Poseidon2Air, &t, &[]), Ok(()));
    }

    #[test]
    fn constraints_catch_any_wrong_cell() {
        let t = trace(&inputs(4));
        for col in [
            0,
            5,
            BEGIN_OFFSET + 3,
            PARTIAL_OFFSET + 1,
            END_OFFSET,
            NUM_COLS - 1,
        ] {
            let mut bad = t.clone();
            bad.row_mut(2)[col] += BabyBear::ONE;
            assert_eq!(
                check_constraints(&Poseidon2Air, &bad, &[]),
                Err(2),
                "column {col}"
            );
        }
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let t = trace(&inputs(64));
        let (pk, vk) = stark::setup(&params, &Poseidon2Air);
        let proof = stark::prove(&params, &pk, &Poseidon2Air, t, &[]);
        assert_eq!(
            stark::verify(&params, &vk, &Poseidon2Air, &[], &proof),
            Ok(())
        );
    }
}
