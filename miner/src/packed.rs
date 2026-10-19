//! Eight BabyBear elements per AVX2 register, for vectorized hashing and
//! transforms. Elements are in Montgomery form, as [`BabyBear`] stores
//! them.
//!
//! Multiplication is Montgomery multiplication on 32-bit lanes: even and
//! odd lanes multiply separately into 64-bit products, the reduction is
//! `(x - (x mu mod 2^32) p) / 2^32`, and results stay canonical (`< p`) via
//! `min(t, t +- p)` on unsigned 32-bit lanes.
//!
//! [`BabyBear`]: crate::babybear::BabyBear

/// Whether the CPU supports AVX2 (checked once).
#[cfg(target_arch = "x86_64")]
pub fn avx2_available() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::*;

    use crate::babybear::{BabyBear, MONTY_MU, P};

    pub type V = __m256i;

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn splat(x: u32) -> V {
        _mm256_set1_epi32(x as i32)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn add(a: V, b: V) -> V {
        let t = _mm256_add_epi32(a, b);
        _mm256_min_epu32(t, _mm256_sub_epi32(t, splat(P)))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn sub(a: V, b: V) -> V {
        let t = _mm256_sub_epi32(a, b);
        _mm256_min_epu32(t, _mm256_add_epi32(t, splat(P)))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn mul(a: V, b: V) -> V {
        let (p, mu) = (splat(P), splat(MONTY_MU));
        let prod_even = _mm256_mul_epu32(a, b);
        let prod_odd = _mm256_mul_epu32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(b));
        let qp_even = _mm256_mul_epu32(_mm256_mul_epu32(prod_even, mu), p);
        let qp_odd = _mm256_mul_epu32(_mm256_mul_epu32(prod_odd, mu), p);
        // The low halves cancel; the high halves' difference is in (-p, p).
        let d_even = _mm256_sub_epi32(
            _mm256_srli_epi64::<32>(prod_even),
            _mm256_srli_epi64::<32>(qp_even),
        );
        let d_odd = _mm256_sub_epi32(prod_odd, qp_odd);
        let d = _mm256_blend_epi32::<0b1010_1010>(d_even, d_odd);
        _mm256_min_epu32(d, _mm256_add_epi32(d, p))
    }

    /// Loads 8 elements.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn load(x: &[BabyBear; 8]) -> V {
        // SAFETY: BabyBear is a u32 in Montgomery form (repr(transparent)),
        // and the array is 32 bytes.
        unsafe { _mm256_loadu_si256(x.as_ptr().cast()) }
    }

    /// Stores 8 elements.
    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn store(x: &mut [BabyBear; 8], v: V) {
        // SAFETY: as `load`; every lane holds a canonical value.
        unsafe { _mm256_storeu_si256(x.as_mut_ptr().cast(), v) }
    }
}
