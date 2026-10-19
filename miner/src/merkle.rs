//! Merkle commitments to the rows of a matrix, for STARK and FRI oracles.
//!
//! Leaf `i` is the hash of row `i`; the tree has height `log2(rows)`, which
//! the verifier always knows, so internal nodes need no domain tag (see
//! [`hash::merge`]).

use crate::babybear::BabyBear;
use crate::hash::{self, Hash, domain};
use crate::matrix::RowMajorMatrix;
use crate::par::par_map;
use crate::poseidon2_x8::LANES;

/// Hash of one row.
pub fn hash_row(row: &[BabyBear]) -> Hash {
    hash::sponge(domain::MERKLE_LEAF, &[], &[row])
}

/// A committed matrix together with every layer of its tree.
#[derive(Clone, Debug)]
pub struct MerkleTree {
    matrix: RowMajorMatrix,
    /// `layers[0]` holds the leaf hashes and the last layer the root.
    layers: Vec<Vec<Hash>>,
}

/// Siblings from the leaf up to (not including) the root.
pub type MerklePath = Vec<Hash>;

impl MerkleTree {
    /// # Panics
    /// If the matrix height is not a power of two.
    pub fn new(matrix: RowMajorMatrix) -> Self {
        assert!(
            matrix.height().is_power_of_two(),
            "matrix height must be a power of two"
        );
        let leaves = hash_rows(&matrix);
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let next = merge_layer(layers.last().unwrap());
            layers.push(next);
        }
        Self { matrix, layers }
    }

    pub fn root(&self) -> Hash {
        self.layers.last().unwrap()[0]
    }

    pub fn matrix(&self) -> &RowMajorMatrix {
        &self.matrix
    }

    pub fn log_height(&self) -> u32 {
        self.matrix.height().trailing_zeros()
    }

    /// Row `index` and its authentication path.
    pub fn open(&self, index: usize) -> (Vec<BabyBear>, MerklePath) {
        let path = self.layers[..self.layers.len() - 1]
            .iter()
            .enumerate()
            .map(|(d, layer)| layer[(index >> d) ^ 1])
            .collect();
        (self.matrix.row(index).to_vec(), path)
    }
}

/// Checks that `row` is row `index` of a matrix of height `2^log_height`
/// committed to by `root`.
pub fn verify(root: &Hash, log_height: u32, index: usize, row: &[BabyBear], path: &[Hash]) -> bool {
    if path.len() != log_height as usize || index >> log_height != 0 {
        return false;
    }
    let mut node = hash_row(row);
    for (d, sibling) in path.iter().enumerate() {
        node = if (index >> d) & 1 == 0 {
            hash::merge(&node, sibling)
        } else {
            hash::merge(sibling, &node)
        };
    }
    node == *root
}

/// `(0..n).map(f)` spread across threads, in order.
/// Leaf hashes, eight rows per batch.
fn hash_rows(m: &RowMajorMatrix) -> Vec<Hash> {
    let n = m.height();
    if n < LANES {
        return (0..n).map(|i| hash_row(m.row(i))).collect();
    }
    par_map(n / LANES, 8, |b| {
        hash::sponge_x8(
            domain::MERKLE_LEAF,
            &[],
            core::array::from_fn(|l| m.row(b * LANES + l)),
        )
    })
    .concat()
}

/// The layer above `prev` (even length), eight nodes per batch.
fn merge_layer(prev: &[Hash]) -> Vec<Hash> {
    let n = prev.len() / 2;
    if n < LANES {
        return (0..n)
            .map(|i| hash::merge(&prev[2 * i], &prev[2 * i + 1]))
            .collect();
    }
    par_map(n / LANES, 128, |b| {
        let at = 2 * b * LANES;
        hash::merge_x8(core::array::from_fn(|l| {
            (&prev[at + 2 * l], &prev[at + 2 * l + 1])
        }))
    })
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(height: usize, width: usize) -> RowMajorMatrix {
        RowMajorMatrix::new(
            (0..height * width)
                .map(|i| BabyBear::new(i as u32 * 7 + 1))
                .collect(),
            width,
        )
    }

    /// Root computed by straightforward recursion.
    fn naive_root(m: &RowMajorMatrix, lo: usize, hi: usize) -> Hash {
        if hi - lo == 1 {
            return hash_row(m.row(lo));
        }
        let mid = (lo + hi) / 2;
        hash::merge(&naive_root(m, lo, mid), &naive_root(m, mid, hi))
    }

    #[test]
    fn root_matches_naive() {
        for log_h in [0, 1, 3, 10] {
            let m = matrix(1 << log_h, 3);
            let tree = MerkleTree::new(m.clone());
            assert_eq!(tree.root(), naive_root(&m, 0, m.height()));
            assert_eq!(tree.log_height(), log_h);
        }
    }

    #[test]
    fn openings_verify() {
        let tree = MerkleTree::new(matrix(1 << 9, 5));
        for i in [0, 1, 2, 255, 256, 511] {
            let (row, path) = tree.open(i);
            assert_eq!(row, tree.matrix().row(i));
            assert!(verify(&tree.root(), 9, i, &row, &path));
        }
    }

    #[test]
    fn rejects_bad_openings() {
        let tree = MerkleTree::new(matrix(16, 2));
        let root = tree.root();
        let (row, path) = tree.open(5);
        assert!(!verify(&root, 4, 6, &row, &path));
        assert!(!verify(&root, 3, 5, &row, &path[..3]));
        assert!(!verify(&root, 4, 5 + 16, &row, &path));

        let mut bad_row = row.clone();
        bad_row[1] += BabyBear::ONE;
        assert!(!verify(&root, 4, 5, &bad_row, &path));

        let mut bad_path = path.clone();
        bad_path[2][0] += BabyBear::ONE;
        assert!(!verify(&root, 4, 5, &row, &bad_path));
    }
}
