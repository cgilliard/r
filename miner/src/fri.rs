//! FRI: proves that a committed function on a coset domain is (close to) a
//! polynomial of degree below `domain size / blowup`.
//!
//! The input is `f` evaluated over `shift * H`, `|H| = 2^log_n`, in natural
//! order (index `i` is the point `shift * w^i`), with values in the
//! extension field.
//!
//! Each round commits to the current evaluations, with the `a = 2^log_arity`
//! points `x * zeta^i` that fold together (`zeta` of order `a`) stored in one
//! Merkle leaf, then draws `beta` and folds: writing
//! `f(X) = sum_j X^j f_j(X^a)`, the next function is
//! `g(Y) = sum_j beta^j f_j(Y)` on the domain `shift^a * H^a`. Degree and
//! domain both shrink by `a`. Once the degree bound reaches
//! `2^log_final_len`, the prover sends the polynomial's coefficients.
//!
//! The verifier samples query indices, opens each layer's leaf on the path,
//! checks it contains the value carried down from the layer above, folds it,
//! and finally compares against the final polynomial.

use std::fmt;

use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::challenger::Challenger;
use crate::hash::Hash;
use crate::matrix::RowMajorMatrix;
use crate::merkle::{self, MerklePath, MerkleTree};
use crate::ntt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FriParams {
    /// `log2(domain size / degree bound)`.
    pub log_blowup: u32,
    /// Each round folds by `2^log_arity` (the last round may fold less).
    pub log_arity: u32,
    /// Folding stops at degree bound `2^log_final_len`.
    pub log_final_len: u32,
    pub num_queries: usize,
    pub grinding_bits: u32,
}

impl Default for FriParams {
    /// Blowup 8, arity 8, 28 queries and 16 grinding bits: ~100 bits of
    /// conjectured security.
    fn default() -> Self {
        Self {
            log_blowup: 3,
            log_arity: 3,
            log_final_len: 2,
            num_queries: 28,
            grinding_bits: 16,
        }
    }
}

impl FriParams {
    /// Conjectured security in bits: each query gains `log_blowup` bits,
    /// grinding adds its bits on top.
    pub fn conjectured_security_bits(&self) -> u32 {
        self.num_queries as u32 * self.log_blowup + self.grinding_bits
    }

    /// Log arities of the folding rounds for a domain of size `2^log_n`, and
    /// the log length of the final polynomial.
    pub(crate) fn schedule(&self, log_n: u32) -> (Vec<u32>, u32) {
        assert!(log_n >= self.log_blowup, "domain smaller than the blowup");
        let mut log_d = log_n - self.log_blowup;
        let log_final = self.log_final_len.min(log_d);
        let mut rounds = Vec::new();
        while log_d > log_final {
            let la = self.log_arity.min(log_d - log_final);
            rounds.push(la);
            log_d -= la;
        }
        (rounds, log_final)
    }
}

/// One query's openings: one leaf per folding round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryProof {
    pub layers: Vec<LayerOpening>,
}

/// The `a` values that fold together, and their Merkle path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerOpening {
    pub values: Vec<BabyBear4>,
    pub path: MerklePath,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriProof {
    /// One commitment per folding round.
    pub commitments: Vec<Hash>,
    /// Coefficients of the final polynomial, lowest first.
    pub final_poly: Vec<BabyBear4>,
    pub pow_witness: BabyBear,
    pub queries: Vec<QueryProof>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FriError {
    /// The proof's shape does not match the parameters.
    Shape,
    InvalidPow,
    /// The caller could not supply the input value for a query.
    MissingInput {
        query: usize,
    },
    /// A layer's opening fails Merkle verification.
    BadOpening {
        query: usize,
        layer: usize,
    },
    /// An opened value disagrees with the value folded from the layer above
    /// (layer 0: with the caller's input value).
    Inconsistent {
        query: usize,
        layer: usize,
    },
    /// The last folded value disagrees with the final polynomial.
    FinalPoly {
        query: usize,
    },
}

impl fmt::Display for FriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape => write!(f, "FRI proof has the wrong shape"),
            Self::InvalidPow => write!(f, "invalid proof-of-work witness"),
            Self::MissingInput { query } => write!(f, "query {query}: no input value"),
            Self::BadOpening { query, layer } => {
                write!(f, "query {query}, layer {layer}: invalid Merkle opening")
            }
            Self::Inconsistent { query, layer } => {
                write!(f, "query {query}, layer {layer}: inconsistent value")
            }
            Self::FinalPoly { query } => {
                write!(f, "query {query}: final polynomial mismatch")
            }
        }
    }
}

impl std::error::Error for FriError {}

/// Proves `evals` (over `shift * H`) has low degree. Returns the proof and
/// the sampled query indices, at which the caller must open the input.
pub fn prove(
    params: &FriParams,
    mut evals: Vec<BabyBear4>,
    shift: BabyBear,
    challenger: &mut Challenger,
) -> (FriProof, Vec<usize>) {
    assert!(
        evals.len().is_power_of_two(),
        "domain size must be a power of two"
    );
    let log_n0 = evals.len().trailing_zeros();
    let (rounds, log_final) = params.schedule(log_n0);

    let mut trees = Vec::with_capacity(rounds.len());
    let mut shift_k = shift;
    for &la in &rounds {
        let (a, m) = (1usize << la, evals.len() >> la);
        // Leaf t holds f(x_t * zeta^i) = evals[t + i m] for i < a.
        let mut leaves = Vec::with_capacity(evals.len() * 4);
        for t in 0..m {
            for i in 0..a {
                leaves.extend_from_slice(&evals[t + i * m].0);
            }
        }
        let tree = MerkleTree::new(RowMajorMatrix::new(leaves, 4 * a));
        challenger.observe_hash(&tree.root());
        let beta = challenger.sample_ext();

        let w = BabyBear::two_adic_generator(evals.len().trailing_zeros());
        let mut x = shift_k;
        let mut fiber = vec![BabyBear4::ZERO; a];
        evals = (0..m)
            .map(|t| {
                for (i, v) in fiber.iter_mut().enumerate() {
                    *v = evals[t + i * m];
                }
                let folded = fold(&fiber, x, beta);
                x *= w;
                folded
            })
            .collect();
        shift_k = shift_k.pow(a as u64);
        trees.push(tree);
    }

    let final_poly = interpolate(&evals, shift_k, 1 << log_final);
    for &c in &final_poly {
        challenger.observe_ext(c);
    }
    let pow_witness = challenger.grind(params.grinding_bits);

    let indices: Vec<usize> = (0..params.num_queries)
        .map(|_| challenger.sample_bits(log_n0))
        .collect();
    let queries = indices
        .iter()
        .map(|&index| {
            let mut i = index;
            let layers = trees
                .iter()
                .map(|tree| {
                    let leaf = i % tree.matrix().height();
                    let (row, path) = tree.open(leaf);
                    i = leaf;
                    LayerOpening {
                        values: row
                            .chunks_exact(4)
                            .map(|c| BabyBear4(c.try_into().unwrap()))
                            .collect(),
                        path,
                    }
                })
                .collect();
            QueryProof { layers }
        })
        .collect();

    let proof = FriProof {
        commitments: trees.iter().map(MerkleTree::root).collect(),
        final_poly,
        pow_witness,
        queries,
    };
    (proof, indices)
}

/// Verifies a proof for a domain of size `2^log_n` shifted by `shift`.
/// `input(query, index)` must return the claimed input value at `index`,
/// for the caller to check against its own commitments.
pub fn verify(
    params: &FriParams,
    log_n: u32,
    shift: BabyBear,
    proof: &FriProof,
    challenger: &mut Challenger,
    mut input: impl FnMut(usize, usize) -> Option<BabyBear4>,
) -> Result<(), FriError> {
    let (rounds, log_final) = params.schedule(log_n);
    if proof.commitments.len() != rounds.len()
        || proof.final_poly.len() != 1 << log_final
        || proof.queries.len() != params.num_queries
        || proof.queries.iter().any(|q| q.layers.len() != rounds.len())
    {
        return Err(FriError::Shape);
    }

    let betas: Vec<BabyBear4> = proof
        .commitments
        .iter()
        .map(|c| {
            challenger.observe_hash(c);
            challenger.sample_ext()
        })
        .collect();
    for &c in &proof.final_poly {
        challenger.observe_ext(c);
    }
    if !challenger.check_witness(params.grinding_bits, proof.pow_witness) {
        return Err(FriError::InvalidPow);
    }

    for (q, query) in proof.queries.iter().enumerate() {
        let mut index = challenger.sample_bits(log_n);
        let mut value = input(q, index).ok_or(FriError::MissingInput { query: q })?;
        let (mut log_n_k, mut shift_k) = (log_n, shift);

        for (k, ((opening, &la), (root, &beta))) in query
            .layers
            .iter()
            .zip(&rounds)
            .zip(proof.commitments.iter().zip(&betas))
            .enumerate()
        {
            let (a, log_m) = (1usize << la, log_n_k - la);
            if opening.values.len() != a {
                return Err(FriError::Shape);
            }
            let (leaf, pos) = (index & ((1 << log_m) - 1), index >> log_m);
            if opening.values[pos] != value {
                return Err(FriError::Inconsistent { query: q, layer: k });
            }
            let row: Vec<BabyBear> = opening.values.iter().flat_map(|v| v.0).collect();
            if !merkle::verify(root, log_m, leaf, &row, &opening.path) {
                return Err(FriError::BadOpening { query: q, layer: k });
            }
            let x = shift_k * BabyBear::two_adic_generator(log_n_k).pow(leaf as u64);
            value = fold(&opening.values, x, beta);
            index = leaf;
            log_n_k = log_m;
            shift_k = shift_k.pow(a as u64);
        }

        let x = shift_k * BabyBear::two_adic_generator(log_n_k).pow(index as u64);
        if evaluate(&proof.final_poly, x) != value {
            return Err(FriError::FinalPoly { query: q });
        }
    }
    Ok(())
}

/// Folds the values of `f` at `x * zeta^i` (`zeta` of order `a`) into
/// `g(x^a) = sum_j beta^j f_j(x^a)`.
///
/// The values are a DFT of `u_j = x^j f_j(x^a)`, so
/// `u_j = (1/a) sum_i v_i zeta^(-ij)` and `g = sum_j u_j (beta / x)^j`.
fn fold(values: &[BabyBear4], x: BabyBear, beta: BabyBear4) -> BabyBear4 {
    let log_a = values.len().trailing_zeros();
    let zeta_inv = BabyBear::two_adic_generator(log_a)
        .inverse()
        .expect("roots are nonzero");
    let r = beta * x.inverse().expect("domain points are nonzero");
    let (mut result, mut r_j, mut zeta_neg_j) = (BabyBear4::ZERO, BabyBear4::ONE, BabyBear::ONE);
    for _ in 0..values.len() {
        let mut u = BabyBear4::ZERO;
        let mut z = BabyBear::ONE;
        for &v in values {
            u += v * z;
            z *= zeta_neg_j;
        }
        result += u * r_j;
        r_j *= r;
        zeta_neg_j *= zeta_inv;
    }
    result * BabyBear::inv_two_pow(log_a)
}

/// Coefficients (lowest first) of the polynomial through `evals` on
/// `shift * H`, truncated to `len`. The truncated part is zero for an honest
/// prover; a dishonest one is caught by the final-polynomial check.
fn interpolate(evals: &[BabyBear4], shift: BabyBear, len: usize) -> Vec<BabyBear4> {
    let mut m = RowMajorMatrix::new(evals.iter().flat_map(|e| e.0).collect(), 4);
    ntt::coset_intt(&mut m, shift);
    m.rows()
        .take(len)
        .map(|row| BabyBear4(row.try_into().unwrap()))
        .collect()
}

fn evaluate(coeffs: &[BabyBear4], x: BabyBear) -> BabyBear4 {
    coeffs
        .iter()
        .rev()
        .fold(BabyBear4::ZERO, |acc, &c| acc * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;

    const SHIFT: BabyBear = BabyBear::new(31);

    fn params(log_arity: u32, log_final_len: u32) -> FriParams {
        FriParams {
            log_blowup: 3,
            log_arity,
            log_final_len,
            num_queries: 20,
            grinding_bits: 4,
        }
    }

    fn rng(seed: u64) -> impl FnMut() -> BabyBear {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            BabyBear::new((s % P as u64) as u32)
        }
    }

    /// Evaluations over `SHIFT * H` (size `2^log_n`) of a random polynomial of
    /// degree below `2^log_degree`.
    fn low_degree(log_n: u32, log_degree: u32, seed: u64) -> Vec<BabyBear4> {
        let mut next = rng(seed);
        let small = RowMajorMatrix::new((0..4 << log_degree).map(|_| next()).collect(), 4);
        let lde = ntt::coset_lde(&small, log_n - log_degree, SHIFT);
        lde.rows()
            .map(|r| BabyBear4(r.try_into().unwrap()))
            .collect()
    }

    fn prove_and_verify(p: &FriParams, evals: Vec<BabyBear4>) -> Result<(), FriError> {
        let log_n = evals.len().trailing_zeros();
        let (proof, indices) = prove(p, evals.clone(), SHIFT, &mut Challenger::new());
        verify(p, log_n, SHIFT, &proof, &mut Challenger::new(), |q, i| {
            assert_eq!(indices[q], i);
            Some(evals[i])
        })
    }

    #[test]
    fn schedule() {
        let p = FriParams::default();
        // Degree 2^10 -> 2^7 -> 2^4 -> 2^2.
        assert_eq!(p.schedule(13), (vec![3, 3, 2], 2));
        // Degree already below the final length: no rounds.
        assert_eq!(p.schedule(4), (vec![], 1));
        assert_eq!(p.conjectured_security_bits(), 100);
    }

    #[test]
    fn honest_proofs_verify() {
        for (log_arity, log_final) in [(1, 0), (2, 1), (3, 2), (3, 0), (4, 3)] {
            let p = params(log_arity, log_final);
            for log_n in [3, 6, 10, 12] {
                let evals = low_degree(log_n, log_n - 3, log_n as u64 * 7 + log_arity as u64);
                assert_eq!(
                    prove_and_verify(&p, evals),
                    Ok(()),
                    "arity 2^{log_arity}, final 2^{log_final}, n = 2^{log_n}"
                );
            }
        }
    }

    #[test]
    fn lower_degree_also_verifies() {
        let evals = low_degree(10, 3, 5);
        assert_eq!(prove_and_verify(&params(3, 2), evals), Ok(()));
    }

    #[test]
    fn rejects_degree_one_over_the_bound() {
        // Degree exactly 2^7 = the bound for n = 2^10 at blowup 8.
        let log_n = 10;
        let mut evals = low_degree(log_n, 7, 3);
        let w = BabyBear::two_adic_generator(log_n);
        let mut x = SHIFT;
        for e in &mut evals {
            *e += BabyBear4::from_base(x.pow(1 << 7));
            x *= w;
        }
        assert!(prove_and_verify(&params(3, 2), evals).is_err());
    }

    #[test]
    fn rejects_random_function() {
        let mut next = rng(42);
        let evals = (0..1 << 10)
            .map(|_| BabyBear4([next(), next(), next(), next()]))
            .collect();
        assert!(prove_and_verify(&params(3, 2), evals).is_err());
    }

    #[test]
    fn rejects_tampering() {
        let p = params(3, 1);
        let log_n = 10;
        let evals = low_degree(log_n, 7, 9);
        let (proof, _) = prove(&p, evals.clone(), SHIFT, &mut Challenger::new());
        let check = |proof: &FriProof| {
            verify(&p, log_n, SHIFT, proof, &mut Challenger::new(), |_, i| {
                Some(evals[i])
            })
        };
        assert_eq!(check(&proof), Ok(()));

        let mut bad = proof.clone();
        bad.queries[3].layers[1].values[0] += BabyBear4::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.final_poly[0] += BabyBear4::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.commitments[0][0] += BabyBear::ONE;
        assert!(check(&bad).is_err());

        let mut bad = proof.clone();
        bad.queries.pop();
        assert_eq!(check(&bad), Err(FriError::Shape));

        // Wrong input values are caught at layer 0.
        let result = verify(&p, log_n, SHIFT, &proof, &mut Challenger::new(), |_, i| {
            Some(evals[i] + BabyBear4::ONE)
        });
        assert_eq!(result, Err(FriError::Inconsistent { query: 0, layer: 0 }));

        // A different transcript gives different queries and challenges.
        let mut other = Challenger::new();
        other.observe(BabyBear::ONE);
        assert!(verify(&p, log_n, SHIFT, &proof, &mut other, |_, i| Some(evals[i])).is_err());
    }

    #[test]
    fn fold_matches_coefficient_definition() {
        // f(X) = sum_k c_k X^k with a = 4: f_j(Y) = sum_m c_{4m+j} Y^m.
        let mut next = rng(17);
        let coeffs: Vec<BabyBear4> = (0..16)
            .map(|_| BabyBear4([next(), next(), next(), next()]))
            .collect();
        let beta = BabyBear4([next(), next(), next(), next()]);
        let x = BabyBear::new(12345);
        let zeta = BabyBear::two_adic_generator(2);
        let f = |p: BabyBear| {
            coeffs
                .iter()
                .rev()
                .fold(BabyBear4::ZERO, |acc, &c| acc * p + c)
        };
        let values: Vec<_> = (0..4).map(|i| f(x * zeta.pow(i))).collect();

        let y = x.pow(4);
        let expected = (0..4).rev().fold(BabyBear4::ZERO, |acc, j| {
            let fj = (0..4)
                .rev()
                .fold(BabyBear4::ZERO, |acc, m| acc * y + coeffs[4 * m + j]);
            acc * beta + fj
        });
        assert_eq!(fold(&values, x, beta), expected);
    }
}
