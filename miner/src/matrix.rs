//! Row-major matrices of field elements, the shape of STARK traces and their
//! extensions: one row per point of the domain, one column per polynomial.

use crate::babybear::BabyBear;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowMajorMatrix {
    pub values: Vec<BabyBear>,
    pub width: usize,
}

impl RowMajorMatrix {
    /// # Panics
    /// If `width` is zero or does not divide `values.len()`.
    pub fn new(values: Vec<BabyBear>, width: usize) -> Self {
        assert!(width > 0, "matrix width must be nonzero");
        assert!(
            values.len().is_multiple_of(width),
            "values do not fill whole rows"
        );
        Self { values, width }
    }

    pub fn zeros(height: usize, width: usize) -> Self {
        Self::new(vec![BabyBear::ZERO; height * width], width)
    }

    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    pub fn row(&self, i: usize) -> &[BabyBear] {
        &self.values[i * self.width..(i + 1) * self.width]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [BabyBear] {
        &mut self.values[i * self.width..(i + 1) * self.width]
    }

    pub fn rows(&self) -> impl Iterator<Item = &[BabyBear]> {
        self.values.chunks_exact(self.width)
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [BabyBear]> {
        self.values.chunks_exact_mut(self.width)
    }

    /// Column `j` as a vector.
    pub fn column(&self, j: usize) -> Vec<BabyBear> {
        self.rows().map(|r| r[j]).collect()
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        let (lo, hi) = (i.min(j), i.max(j));
        let (a, b) = self.values.split_at_mut(hi * self.width);
        a[lo * self.width..(lo + 1) * self.width].swap_with_slice(&mut b[..self.width]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(values: &[u32], width: usize) -> RowMajorMatrix {
        RowMajorMatrix::new(values.iter().map(|&v| BabyBear::new(v)).collect(), width)
    }

    #[test]
    fn shape_and_access() {
        let mut a = m(&[1, 2, 3, 4, 5, 6], 2);
        assert_eq!(a.height(), 3);
        assert_eq!(a.row(1), m(&[3, 4], 2).values.as_slice());
        assert_eq!(a.column(1), m(&[2, 4, 6], 1).values);
        a.swap_rows(0, 2);
        assert_eq!(a, m(&[5, 6, 3, 4, 1, 2], 2));
        a.swap_rows(1, 1);
        assert_eq!(a, m(&[5, 6, 3, 4, 1, 2], 2));
    }

    #[test]
    #[should_panic(expected = "whole rows")]
    fn rejects_ragged() {
        m(&[1, 2, 3], 2);
    }
}
