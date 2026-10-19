//! Eight Poseidon2 permutations at once, one per AVX2 lane, for hashing
//! many independent inputs (Merkle leaves and nodes). Falls back to the
//! scalar permutation without AVX2. Same results as [`poseidon2::permute`].

use crate::babybear::BabyBear;
use crate::poseidon2::{self, WIDTH};

pub const LANES: usize = 8;

/// Permutes each of `states`.
pub fn permute_x8(states: &mut [[BabyBear; WIDTH]; LANES]) {
    #[cfg(target_arch = "x86_64")]
    if crate::packed::avx2_available() {
        // SAFETY: AVX2 is available.
        unsafe { avx2::permute(states) };
        return;
    }
    for s in states {
        poseidon2::permute(s);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use crate::babybear::BabyBear;
    use crate::poseidon2::{
        INTERNAL_DIAG, RC_EXTERNAL_FINAL, RC_EXTERNAL_INITIAL, RC_INTERNAL, WIDTH,
    };

    use super::LANES;
    use crate::packed::avx2::{V, add, load, mul, splat, store};

    #[target_feature(enable = "avx2")]
    #[inline]
    fn pow7(x: V) -> V {
        let x2 = mul(x, x);
        let x3 = mul(x2, x);
        let x6 = mul(x3, x3);
        mul(x6, x)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn mat4(x: &mut [V]) {
        let t01 = add(x[0], x[1]);
        let t23 = add(x[2], x[3]);
        let t0123 = add(t01, t23);
        let t01123 = add(t0123, x[1]);
        let t01233 = add(t0123, x[3]);
        x[3] = add(t01233, add(x[0], x[0]));
        x[1] = add(t01123, add(x[2], x[2]));
        x[0] = add(t01123, t01);
        x[2] = add(t01233, t23);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn external_linear_layer(s: &mut [V; WIDTH]) {
        for chunk in s.chunks_exact_mut(4) {
            mat4(chunk);
        }
        let sums: [V; 4] =
            core::array::from_fn(|k| add(add(s[k], s[4 + k]), add(s[8 + k], s[12 + k])));
        for (i, x) in s.iter_mut().enumerate() {
            *x = add(*x, sums[i % 4]);
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn full_round(s: &mut [V; WIDTH], rc: &[BabyBear; WIDTH]) {
        for (x, c) in s.iter_mut().zip(rc) {
            *x = pow7(add(*x, splat(c.raw())));
        }
        external_linear_layer(s);
    }

    #[target_feature(enable = "avx2")]
    pub(super) fn permute(states: &mut [[BabyBear; WIDTH]; LANES]) {
        let mut s: [V; WIDTH] =
            core::array::from_fn(|i| load(&core::array::from_fn(|l| states[l][i])));
        let diag: [V; WIDTH] = core::array::from_fn(|i| splat(INTERNAL_DIAG[i].raw()));
        external_linear_layer(&mut s);
        for rc in &RC_EXTERNAL_INITIAL {
            full_round(&mut s, rc);
        }
        for rc in &RC_INTERNAL {
            s[0] = pow7(add(s[0], splat(rc.raw())));
            let sum = s.iter().fold(splat(0), |acc, &x| add(acc, x));
            for (x, &d) in s.iter_mut().zip(&diag) {
                *x = add(sum, mul(d, *x));
            }
        }
        for rc in &RC_EXTERNAL_FINAL {
            full_round(&mut s, rc);
        }
        for (i, v) in s.iter().enumerate() {
            let mut lane = [BabyBear::ZERO; LANES];
            store(&mut lane, *v);
            for (l, &x) in lane.iter().enumerate() {
                states[l][i] = x;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scalar() {
        let mut x = 1u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            BabyBear::new(x)
        };
        for _ in 0..20 {
            let mut states: [[BabyBear; WIDTH]; LANES] =
                core::array::from_fn(|_| core::array::from_fn(|_| next()));
            states[0] = [BabyBear::ZERO; WIDTH];
            states[1] = [-BabyBear::ONE; WIDTH];
            let mut expected = states;
            for s in &mut expected {
                poseidon2::permute(s);
            }
            permute_x8(&mut states);
            assert_eq!(states, expected);
        }
    }
}
