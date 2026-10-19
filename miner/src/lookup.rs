//! LogUp: a running-sum argument that a set of fractions
//! `m_k / (beta - fp(t_k))` sums to zero over the whole trace, where
//! `fp(t) = sum_j gamma^j t_j` fingerprints a tuple of values.
//!
//! This covers lookups (each query counts `+1`, each table entry `-m` for its
//! multiplicity) and multiset equality, i.e. copy constraints (each value
//! counts `+1` where written and `-1` where read). An AIR lists its terms per
//! row with one generic function used by both [`eval`] and [`trace`], so the
//! constraints and the auxiliary trace can't drift apart.
//!
//! Auxiliary columns, starting at an offset the AIR chooses: one fraction
//! `u_k = m_k / (beta - fp(t_k))` per term, then the running sum `S`, which
//! starts at 0 and accumulates each row's fractions; the last row's sum
//! plus its fractions must be 0. Challenges: `beta` then `gamma`.

use crate::air::{Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;

/// Challenges a bus uses.
pub const CHALLENGES: usize = 2;

/// One fraction `multiplicity / (beta - fp(tuple))`.
#[derive(Clone, Debug)]
pub struct Term<V> {
    pub multiplicity: V,
    pub tuple: Vec<V>,
}

impl<V> Term<V> {
    pub fn new(multiplicity: V, tuple: Vec<V>) -> Self {
        Self {
            multiplicity,
            tuple,
        }
    }
}

/// Auxiliary columns a bus with `terms` terms per row uses.
pub const fn aux_width(terms: usize) -> usize {
    terms + 1
}

/// `beta - sum_j gamma^j t_j`.
fn denominator<V: Var>(beta: V::Ext, gamma: V::Ext, tuple: &[V]) -> V::Ext {
    let mut pow = V::Ext::from(BabyBear4::ONE);
    let mut acc = beta;
    for &t in tuple {
        acc -= pow * V::Ext::from(t);
        pow *= gamma;
    }
    acc
}

/// Constrains the bus: `terms` are this row's fractions (the same list, in
/// the same order, on every row), `aux` the auxiliary column offset and
/// `challenge` the offset of `beta, gamma` among the AIR's challenges.
pub fn eval<V: Var>(c: &mut Constraints<'_, V>, terms: &[Term<V>], aux: usize, challenge: usize) {
    let (beta, gamma) = (c.challenges()[challenge], c.challenges()[challenge + 1]);
    let (local, next) = (c.aux_local(), c.aux_next());
    let mut row_sum = V::Ext::from(BabyBear4::ZERO);
    for (k, term) in terms.iter().enumerate() {
        let u = local[aux + k];
        let d = denominator(beta, gamma, &term.tuple);
        c.assert_zero_ext(u * d - V::Ext::from(term.multiplicity));
        row_sum += u;
    }
    let s = local[aux + terms.len()];
    c.assert_zero_ext_first_row(s);
    c.assert_zero_ext_transition(next[aux + terms.len()] - s - row_sum);
    c.assert_zero_ext_last_row(s + row_sum);
}

/// The bus's auxiliary values for every row, `aux_width(terms)` per row.
/// `rows[i]` lists row `i`'s terms.
pub fn trace(
    rows: &[Vec<Term<BabyBear>>],
    beta: BabyBear4,
    gamma: BabyBear4,
) -> Vec<Vec<BabyBear4>> {
    let denominators: Vec<BabyBear4> = rows
        .iter()
        .flat_map(|terms| {
            terms
                .iter()
                .map(|t| denominator::<BabyBear>(beta, gamma, &t.tuple))
        })
        .collect();
    let inverses = batch_inverse(&denominators);
    let mut out = Vec::with_capacity(rows.len());
    let mut sum = BabyBear4::ZERO;
    let mut k = 0;
    for terms in rows {
        let mut row = Vec::with_capacity(terms.len() + 1);
        let mut row_sum = BabyBear4::ZERO;
        for t in terms {
            let u = inverses[k] * t.multiplicity;
            k += 1;
            row.push(u);
            row_sum += u;
        }
        row.push(sum);
        sum += row_sum;
        out.push(row);
    }
    out
}

/// Inverts every element; zero stays zero (it only arises with negligible
/// probability, and then the trace fails its constraints).
fn batch_inverse(xs: &[BabyBear4]) -> Vec<BabyBear4> {
    let mut prefix = Vec::with_capacity(xs.len());
    let mut acc = BabyBear4::ONE;
    for &x in xs {
        prefix.push(acc);
        if x != BabyBear4::ZERO {
            acc *= x;
        }
    }
    let mut inv = acc.inverse().unwrap_or(BabyBear4::ZERO);
    let mut out = vec![BabyBear4::ZERO; xs.len()];
    for i in (0..xs.len()).rev() {
        if xs[i] != BabyBear4::ZERO {
            out[i] = inv * prefix[i];
            inv *= xs[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::{Air, check_constraints};
    use crate::fri::FriParams;
    use crate::matrix::RowMajorMatrix;
    use crate::stark;

    fn params() -> FriParams {
        FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        }
    }

    fn prove_verify<A: Air>(air: &A, trace: RowMajorMatrix) -> Result<(), stark::StarkError> {
        let (pk, vk) = stark::setup(&params(), air);
        let proof = stark::prove(&params(), &pk, air, trace, &[]);
        stark::verify(&params(), &vk, air, &[], &proof)
    }

    /// Range check: every value in column 0 lies in the preprocessed table
    /// `0..256`; column 1 holds each table row's multiplicity.
    struct Range8;

    impl Range8 {
        const HEIGHT: usize = 256;

        fn terms<V: Var>(local: &[V], pre: &[V]) -> Vec<Term<V>> {
            vec![
                Term::new(V::from(BabyBear::ONE), vec![local[0]]),
                Term::new(-local[1], vec![pre[0]]),
            ]
        }

        fn trace(values: &[u32]) -> RowMajorMatrix {
            let mut m = RowMajorMatrix::zeros(Self::HEIGHT, 2);
            let mut counts = [0u32; 256];
            for (i, &v) in values.iter().enumerate() {
                m.row_mut(i)[0] = BabyBear::new(v);
                if (v as usize) < 256 {
                    counts[v as usize] += 1;
                }
            }
            // Unused value rows look up 0.
            counts[0] += (Self::HEIGHT - values.len()) as u32;
            for (t, &c) in counts.iter().enumerate() {
                m.row_mut(t)[1] = BabyBear::new(c);
            }
            m
        }
    }

    impl Air for Range8 {
        fn width(&self) -> usize {
            2
        }
        fn max_degree(&self) -> usize {
            3
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let terms = Self::terms(c.local(), c.preprocessed_local());
            eval(c, &terms, 0, 0);
        }
        fn preprocessed_width(&self) -> usize {
            1
        }
        fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
            Some(RowMajorMatrix::new(
                (0..Self::HEIGHT as u32).map(BabyBear::new).collect(),
                1,
            ))
        }
        fn num_challenges(&self) -> usize {
            CHALLENGES
        }
        fn aux_width(&self) -> usize {
            aux_width(2)
        }
        fn aux_trace(
            &self,
            main: &RowMajorMatrix,
            pre: Option<&RowMajorMatrix>,
            ch: &[BabyBear4],
        ) -> Vec<BabyBear4> {
            let pre = pre.expect("table");
            let rows: Vec<_> = (0..main.height())
                .map(|i| Self::terms(main.row(i), pre.row(i)))
                .collect();
            trace(&rows, ch[0], ch[1]).concat()
        }
    }

    /// Multiset equality of `(address, value)` pairs: columns 0-1 are writes,
    /// 2-3 reads, in any order.
    struct Memory;

    impl Memory {
        fn terms<V: Var>(local: &[V]) -> Vec<Term<V>> {
            vec![
                Term::new(V::from(BabyBear::ONE), vec![local[0], local[1]]),
                Term::new(-V::from(BabyBear::ONE), vec![local[2], local[3]]),
            ]
        }

        fn trace(writes: &[(u32, u32)], reads: &[(u32, u32)]) -> RowMajorMatrix {
            let values = writes
                .iter()
                .zip(reads)
                .flat_map(|(&(a, v), &(b, w))| [a, v, b, w])
                .map(BabyBear::new)
                .collect();
            RowMajorMatrix::new(values, 4)
        }
    }

    impl Air for Memory {
        fn width(&self) -> usize {
            4
        }
        fn max_degree(&self) -> usize {
            3
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let terms = Self::terms(c.local());
            eval(c, &terms, 0, 0);
        }
        fn num_challenges(&self) -> usize {
            CHALLENGES
        }
        fn aux_width(&self) -> usize {
            aux_width(2)
        }
        fn aux_trace(
            &self,
            main: &RowMajorMatrix,
            _pre: Option<&RowMajorMatrix>,
            ch: &[BabyBear4],
        ) -> Vec<BabyBear4> {
            let rows: Vec<_> = main.rows().map(Self::terms).collect();
            trace(&rows, ch[0], ch[1]).concat()
        }
    }

    type Ops = Vec<(u32, u32)>;

    fn memory_ops(n: u32) -> (Ops, Ops) {
        let writes: Vec<_> = (0..n).map(|i| (i, i * i + 7)).collect();
        // Reads in a different order.
        let reads = (0..n)
            .map(|i| writes[((i * 37 + 11) % n) as usize])
            .collect();
        (writes, reads)
    }

    #[test]
    fn range_check_accepts_values_in_range() {
        let values: Vec<u32> = (0..200).map(|i| (i * 13 + 5) % 256).collect();
        let t = Range8::trace(&values);
        assert_eq!(check_constraints(&Range8, &t, &[]), Ok(()));
        assert_eq!(prove_verify(&Range8, t), Ok(()));
    }

    #[test]
    fn range_check_rejects_out_of_range_value() {
        let mut values: Vec<u32> = (0..100).collect();
        values[40] = 256;
        let t = Range8::trace(&values);
        assert!(check_constraints(&Range8, &t, &[]).is_err());
        assert!(prove_verify(&Range8, t).is_err());
    }

    #[test]
    fn range_check_rejects_wrong_multiplicities() {
        let values: Vec<u32> = (0..100).collect();
        let mut t = Range8::trace(&values);
        t.row_mut(7)[1] += BabyBear::ONE;
        assert!(check_constraints(&Range8, &t, &[]).is_err());
    }

    #[test]
    fn memory_accepts_permuted_reads() {
        let (writes, reads) = memory_ops(64);
        let t = Memory::trace(&writes, &reads);
        assert_eq!(check_constraints(&Memory, &t, &[]), Ok(()));
        assert_eq!(prove_verify(&Memory, t), Ok(()));
    }

    #[test]
    fn memory_rejects_a_wrong_value_or_address() {
        let (writes, mut reads) = memory_ops(64);
        reads[10].1 += 1;
        assert!(check_constraints(&Memory, &Memory::trace(&writes, &reads), &[]).is_err());

        let (writes, mut reads) = memory_ops(64);
        // Same value at a different address: tuples are bound together.
        reads[3] = (reads[3].0 + 1, reads[3].1);
        assert!(check_constraints(&Memory, &Memory::trace(&writes, &reads), &[]).is_err());

        // A read duplicated in place of another.
        let (writes, mut reads) = memory_ops(64);
        reads[5] = reads[6];
        let t = Memory::trace(&writes, &reads);
        assert!(check_constraints(&Memory, &t, &[]).is_err());
        assert!(prove_verify(&Memory, t).is_err());
    }

    #[test]
    fn tampered_aux_commitment_fails() {
        let (writes, reads) = memory_ops(64);
        let t = Memory::trace(&writes, &reads);
        let (pk, vk) = stark::setup(&params(), &Memory);
        let proof = stark::prove(&params(), &pk, &Memory, t, &[]);
        let mut bad = proof.clone();
        bad.aux_at_zeta[0] += BabyBear4::ONE;
        assert!(stark::verify(&params(), &vk, &Memory, &[], &bad).is_err());
        let mut bad = proof.clone();
        bad.openings[0].aux_row[0] += BabyBear::ONE;
        assert_eq!(
            stark::verify(&params(), &vk, &Memory, &[], &bad),
            Err(stark::StarkError::BadOpening { query: 0 })
        );
        let mut bad = proof;
        bad.aux_commitment = None;
        assert_eq!(
            stark::verify(&params(), &vk, &Memory, &[], &bad),
            Err(stark::StarkError::Shape)
        );
    }
}
