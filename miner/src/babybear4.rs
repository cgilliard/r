//! The degree-4 extension `BabyBear[x] / (x^4 - 11)`, about 124 bits.
//!
//! The STARK draws its random challenges from here: challenges from the
//! 31-bit base field would cap soundness at ~31 bits. Same field as
//! Plonky3's `BinomialExtensionField<BabyBear, 4>`.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::babybear::{BabyBear, P};

/// `x^4 = W`. 11 is a non-square mod p and `p = 1 mod 4`, so `x^4 - 11` is
/// irreducible.
const W: BabyBear = BabyBear::new(11);

/// `W^((p - 1) / 4)`: the Frobenius map sends `x` to `x^p = Z * x`.
const Z: BabyBear = W.pow((P as u64 - 1) / 4);

/// `a0 + a1 x + a2 x^2 + a3 x^3`.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct BabyBear4(pub [BabyBear; 4]);

impl BabyBear4 {
    pub const ZERO: Self = Self([BabyBear::ZERO; 4]);
    pub const ONE: Self = Self::from_base(BabyBear::ONE);

    pub const fn from_base(x: BabyBear) -> Self {
        Self([x, BabyBear::ZERO, BabyBear::ZERO, BabyBear::ZERO])
    }

    pub const fn new(coeffs: [u32; 4]) -> Self {
        Self(BabyBear::new_array(coeffs))
    }

    /// The element `x`.
    pub const fn generator_x() -> Self {
        Self([
            BabyBear::ZERO,
            BabyBear::ONE,
            BabyBear::ZERO,
            BabyBear::ZERO,
        ])
    }

    /// Returns the base-field value if this element is in the base field.
    pub fn as_base(&self) -> Option<BabyBear> {
        self.0[1..]
            .iter()
            .all(|&c| c == BabyBear::ZERO)
            .then_some(self.0[0])
    }

    pub fn square(self) -> Self {
        self * self
    }

    pub fn pow(self, mut exp: u128) -> Self {
        let (mut base, mut acc) = (self, Self::ONE);
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            exp >>= 1;
        }
        acc
    }

    /// `self^(p^k)`. Coefficient `i` is multiplied by `Z^(i k)`.
    pub fn frobenius(self, k: u32) -> Self {
        let zk = Z.pow(k as u64 % 4);
        let mut zi = BabyBear::ONE;
        Self(core::array::from_fn(|i| {
            let c = self.0[i] * zi;
            zi *= zk;
            c
        }))
    }

    /// Multiplicative inverse, or `None` for zero.
    ///
    /// `self * frob(self) * frob^2(self) * frob^3(self)` is the norm, which
    /// lies in the base field, so `inverse = frob * frob^2 * frob^3 / norm`.
    pub fn inverse(self) -> Option<Self> {
        let rest = self.frobenius(1) * self.frobenius(2) * self.frobenius(3);
        let norm = (self * rest).as_base().expect("norm is in the base field");
        Some(rest * norm.inverse()?)
    }
}

impl From<BabyBear> for BabyBear4 {
    fn from(x: BabyBear) -> Self {
        Self::from_base(x)
    }
}

impl fmt::Debug for BabyBear4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl fmt::Display for BabyBear4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(x) = self.as_base() {
            return write!(f, "{x}");
        }
        let terms: Vec<String> = self
            .0
            .iter()
            .enumerate()
            .filter(|(_, c)| **c != BabyBear::ZERO)
            .map(|(i, c)| match i {
                0 => format!("{c}"),
                1 => format!("{c} x"),
                _ => format!("{c} x^{i}"),
            })
            .collect();
        f.write_str(&terms.join(" + "))
    }
}

impl Add for BabyBear4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl Sub for BabyBear4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl Neg for BabyBear4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.map(|c| -c))
    }
}

impl Mul for BabyBear4 {
    type Output = Self;
    /// Schoolbook product, then `x^4 = W` folds the high terms down.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.0, rhs.0);
        Self([
            a[0] * b[0] + W * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
            a[0] * b[1] + a[1] * b[0] + W * (a[2] * b[3] + a[3] * b[2]),
            a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + W * (a[3] * b[3]),
            a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
        ])
    }
}

impl Add<BabyBear> for BabyBear4 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: BabyBear) -> Self {
        self.0[0] += rhs;
        self
    }
}

impl Sub<BabyBear> for BabyBear4 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: BabyBear) -> Self {
        self.0[0] -= rhs;
        self
    }
}

impl Mul<BabyBear> for BabyBear4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: BabyBear) -> Self {
        Self(self.0.map(|c| c * rhs))
    }
}

macro_rules! assign_ops {
    ($($rhs:ty),*) => {$(
        impl AddAssign<$rhs> for BabyBear4 {
            #[inline]
            fn add_assign(&mut self, rhs: $rhs) {
                *self = *self + rhs;
            }
        }
        impl SubAssign<$rhs> for BabyBear4 {
            #[inline]
            fn sub_assign(&mut self, rhs: $rhs) {
                *self = *self - rhs;
            }
        }
        impl MulAssign<$rhs> for BabyBear4 {
            #[inline]
            fn mul_assign(&mut self, rhs: $rhs) {
                *self = *self * rhs;
            }
        }
    )*};
}
assign_ops!(BabyBear4, BabyBear);

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random elements (xorshift), for property tests.
    fn elems(n: usize) -> Vec<BabyBear4> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % P as u64) as u32
        };
        (0..n)
            .map(|_| BabyBear4::new([next(), next(), next(), next()]))
            .collect()
    }

    /// Product computed by full polynomial multiplication and reduction.
    fn naive_mul(a: BabyBear4, b: BabyBear4) -> BabyBear4 {
        let mut prod = [BabyBear::ZERO; 7];
        for i in 0..4 {
            for j in 0..4 {
                prod[i + j] += a.0[i] * b.0[j];
            }
        }
        for k in (4..7).rev() {
            let hi = prod[k];
            prod[k - 4] += W * hi;
        }
        BabyBear4(core::array::from_fn(|i| prod[i]))
    }

    #[test]
    fn modulus_is_irreducible() {
        // 11 is a non-square: 11^((p-1)/2) = -1.
        assert_eq!(W.pow((P as u64 - 1) / 2), -BabyBear::ONE);
        assert_eq!(P % 4, 1);
    }

    #[test]
    fn x_to_the_fourth_is_w() {
        let x = BabyBear4::generator_x();
        assert_eq!(x.pow(4), BabyBear4::from_base(W));
    }

    #[test]
    fn frobenius_constant_matches_plonky3() {
        assert_eq!(Z.as_u32(), 1728404513);
    }

    #[test]
    fn mul_matches_naive() {
        let v = elems(20);
        for &a in &v {
            for &b in &v {
                assert_eq!(a * b, naive_mul(a, b));
            }
        }
    }

    #[test]
    fn field_axioms() {
        let v = elems(12);
        for &a in &v {
            assert_eq!(a + BabyBear4::ZERO, a);
            assert_eq!(a * BabyBear4::ONE, a);
            assert_eq!(a - a, BabyBear4::ZERO);
            assert_eq!(a + (-a), BabyBear4::ZERO);
            for &b in &v {
                assert_eq!(a * b, b * a);
                for &c in &v[..4] {
                    assert_eq!((a * b) * c, a * (b * c));
                    assert_eq!(a * (b + c), a * b + a * c);
                }
            }
        }
    }

    #[test]
    fn mixed_ops_match_embedding() {
        let a = elems(1)[0];
        let b = BabyBear::new(123456);
        let eb = BabyBear4::from(b);
        assert_eq!(a * b, a * eb);
        assert_eq!(a + b, a + eb);
        assert_eq!(a - b, a - eb);
    }

    #[test]
    fn frobenius_is_pth_power() {
        for a in elems(4) {
            assert_eq!(a.frobenius(1), a.pow(P as u128));
            assert_eq!(a.frobenius(4), a);
            assert_eq!(a.frobenius(2), a.frobenius(1).frobenius(1));
        }
    }

    #[test]
    fn inverse() {
        assert_eq!(BabyBear4::ZERO.inverse(), None);
        for a in elems(30) {
            assert_eq!(a * a.inverse().unwrap(), BabyBear4::ONE);
        }
        let b = BabyBear4::from_base(BabyBear::new(7));
        assert_eq!(b.inverse().unwrap().as_base(), BabyBear::new(7).inverse());
    }

    #[test]
    fn group_order() {
        // Every nonzero element satisfies a^(p^4 - 1) = 1.
        let order = (P as u128).pow(4) - 1;
        for a in elems(3) {
            assert_eq!(a.pow(order), BabyBear4::ONE);
        }
    }

    #[test]
    fn display() {
        assert_eq!(BabyBear4::from_base(BabyBear::new(5)).to_string(), "5");
        assert_eq!(BabyBear4::new([1, 0, 2, 0]).to_string(), "1 + 2 x^2");
    }
}
