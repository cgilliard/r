//! The BabyBear prime field, `p = 2^31 - 2^27 + 1`.
//!
//! Elements are stored in Montgomery form (`x * 2^32 mod p`) so multiplication
//! needs a single 64-bit product and one reduction, with no division.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The BabyBear prime.
pub const P: u32 = 0x7800_0001;

/// `p - 1 = 2^27 * 15`, so the multiplicative group has subgroups of every
/// order `2^k` for `k <= TWO_ADICITY`.
pub const TWO_ADICITY: u32 = 27;

/// `p^-1 mod 2^32`, used by `monty_reduce`.
pub(crate) const MONTY_MU: u32 = 0x8800_0001;

/// A BabyBear field element, internally in Montgomery form.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BabyBear(u32);

impl BabyBear {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self::new(1);

    /// The Montgomery representation, for vectorized arithmetic.
    #[inline(always)]
    pub(crate) const fn raw(self) -> u32 {
        self.0
    }

    /// Builds an element from a `u32`, reducing it mod `p`.
    #[inline]
    pub const fn new(x: u32) -> Self {
        Self((((x as u64) << 32) % P as u64) as u32)
    }

    /// Builds an array of elements from an array of `u32`s.
    pub const fn new_array<const N: usize>(xs: [u32; N]) -> [Self; N] {
        let mut out = [Self::ZERO; N];
        let mut i = 0;
        while i < N {
            out[i] = Self::new(xs[i]);
            i += 1;
        }
        out
    }

    /// Returns the canonical representative in `[0, p)`.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        monty_reduce(self.0 as u64)
    }

    #[inline]
    pub const fn add(self, rhs: Self) -> Self {
        // Both operands are < p < 2^31, so the sum fits in a u32.
        let sum = self.0 + rhs.0;
        let (corr, over) = sum.overflowing_sub(P);
        Self(if over { sum } else { corr })
    }

    #[inline]
    pub const fn sub(self, rhs: Self) -> Self {
        let (diff, over) = self.0.overflowing_sub(rhs.0);
        Self(if over { diff.wrapping_add(P) } else { diff })
    }

    #[inline]
    pub const fn mul(self, rhs: Self) -> Self {
        Self(monty_reduce(self.0 as u64 * rhs.0 as u64))
    }

    #[inline]
    pub const fn neg(self) -> Self {
        Self::ZERO.sub(self)
    }

    #[inline]
    pub const fn double(self) -> Self {
        self.add(self)
    }

    /// Multiplies by `1/2`. Works in Montgomery form because halving is linear.
    #[inline]
    pub const fn halve(self) -> Self {
        let x = self.0;
        Self(if x & 1 == 0 {
            x >> 1
        } else {
            (x >> 1) + P.div_ceil(2)
        })
    }

    /// Returns `1 / 2^k`.
    pub const fn inv_two_pow(k: u32) -> Self {
        let mut x = Self::ONE;
        let mut i = 0;
        while i < k {
            x = x.halve();
            i += 1;
        }
        x
    }

    pub const fn pow(self, mut exp: u64) -> Self {
        let (mut base, mut acc) = (self, Self::ONE);
        while exp > 0 {
            if exp & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        acc
    }

    /// Multiplicative inverse, or `None` for zero.
    pub const fn inverse(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(P as u64 - 2))
        }
    }

    /// A generator of the subgroup of order `2^bits`.
    ///
    /// # Panics
    /// If `bits > TWO_ADICITY`.
    pub const fn two_adic_generator(bits: u32) -> Self {
        assert!(bits <= TWO_ADICITY, "no subgroup of that order");
        // Same generator as Plonky3, so domains match.
        Self::new(0x1a42_7a41).pow(1 << (TWO_ADICITY - bits))
    }

    /// The Poseidon2 S-box, `x^7`.
    #[inline(always)]
    pub const fn pow7(self) -> Self {
        let x2 = self.mul(self);
        let x3 = x2.mul(self);
        let x4 = x2.mul(x2);
        x3.mul(x4)
    }
}

/// Montgomery reduction: given `x < p * 2^32`, returns `x * 2^-32 mod p` in `[0, p)`.
#[inline(always)]
const fn monty_reduce(x: u64) -> u32 {
    let t = (x as u32).wrapping_mul(MONTY_MU) as u64;
    let u = t * P as u64;
    let (x_sub_u, over) = x.overflowing_sub(u);
    let hi = (x_sub_u >> 32) as u32;
    if over { hi.wrapping_add(P) } else { hi }
}

impl From<bool> for BabyBear {
    fn from(b: bool) -> Self {
        if b { Self::ONE } else { Self::ZERO }
    }
}

impl fmt::Debug for BabyBear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.as_u32(), f)
    }
}

impl fmt::Display for BabyBear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_u32(), f)
    }
}

impl Add for BabyBear {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        BabyBear::add(self, rhs)
    }
}

impl Sub for BabyBear {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        BabyBear::sub(self, rhs)
    }
}

impl Mul for BabyBear {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        BabyBear::mul(self, rhs)
    }
}

impl Neg for BabyBear {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        BabyBear::neg(self)
    }
}

impl AddAssign for BabyBear {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for BabyBear {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for BabyBear {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for x in [0, 1, 2, P - 1, 12345, 0x7fff_ffff, u32::MAX] {
            assert_eq!(BabyBear::new(x).as_u32(), x % P);
        }
    }

    #[test]
    fn arithmetic_matches_naive() {
        let vals = [0u32, 1, 2, 3, P - 1, P - 2, 0x34167c58, 0x61f3207b, 1 << 30];
        for &a in &vals {
            for &b in &vals {
                let (fa, fb) = (BabyBear::new(a), BabyBear::new(b));
                let (a, b, p) = (a as u64, b as u64, P as u64);
                assert_eq!((fa + fb).as_u32() as u64, (a + b) % p);
                assert_eq!((fa - fb).as_u32() as u64, (a + p - b) % p);
                assert_eq!((fa * fb).as_u32() as u64, a * b % p);
            }
            assert_eq!((-BabyBear::new(a)).as_u32(), (P - a % P) % P);
        }
    }

    #[test]
    fn known_product() {
        // Vector from Plonky3's BabyBear tests.
        let m1 = BabyBear::new(0x34167c58);
        let m2 = BabyBear::new(0x61f3207b);
        assert_eq!((m1 * m2).as_u32(), 0x1b5c8046);
    }

    #[test]
    fn inverse_and_pow() {
        assert_eq!(BabyBear::ZERO.inverse(), None);
        for x in [1, 2, 3, 12345, P - 1] {
            let x = BabyBear::new(x);
            assert_eq!(x * x.inverse().unwrap(), BabyBear::ONE);
        }
        assert_eq!(BabyBear::new(3).pow(5), BabyBear::new(243));
        assert_eq!(BabyBear::new(5).pow(0), BabyBear::ONE);
    }

    #[test]
    fn two_adic_generators_have_exact_order() {
        for bits in 0..=TWO_ADICITY {
            let g = BabyBear::two_adic_generator(bits);
            assert_eq!(g.pow(1 << bits), BabyBear::ONE);
            if bits > 0 {
                assert_ne!(g.pow(1 << (bits - 1)), BabyBear::ONE);
            }
        }
        // Plonky3's 2^3 roots, from `ROOTS_8`.
        assert_eq!(BabyBear::two_adic_generator(3).as_u32(), 0x5ee99486);
    }

    #[test]
    fn halving() {
        let two = BabyBear::new(2);
        assert_eq!(BabyBear::new(7).halve() * two, BabyBear::new(7));
        assert_eq!(BabyBear::new(8).halve(), BabyBear::new(4));
        assert_eq!(
            BabyBear::inv_two_pow(27) * BabyBear::new(1 << 27),
            BabyBear::ONE
        );
    }
}
