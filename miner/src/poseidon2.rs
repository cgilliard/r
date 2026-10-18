//! Poseidon2 permutation over BabyBear, width 16 (<https://eprint.iacr.org/2023/323>).
//!
//! Parameters match Plonky3's `default_babybear_poseidon2_16`:
//! S-box `x^7`, 8 full rounds (4 + 4), 13 partial rounds, round constants from
//! the Grain LFSR, the `[2 3 1 1]` circulant 4x4 block for the external layer,
//! and internal layer `1 + Diag(V)`.

use crate::babybear::BabyBear;

pub const WIDTH: usize = 16;
pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 13;

/// Applies the Poseidon2 permutation in place.
#[inline]
pub fn permute(state: &mut [BabyBear; WIDTH]) {
    external_linear_layer(state);
    for rc in &RC_EXTERNAL_INITIAL {
        full_round(state, rc);
    }
    for &rc in &RC_INTERNAL {
        state[0] = (state[0] + rc).pow7();
        internal_linear_layer(state);
    }
    for rc in &RC_EXTERNAL_FINAL {
        full_round(state, rc);
    }
}

#[inline(always)]
fn full_round(state: &mut [BabyBear; WIDTH], rc: &[BabyBear; WIDTH]) {
    for (s, &c) in state.iter_mut().zip(rc) {
        *s = (*s + c).pow7();
    }
    external_linear_layer(state);
}

/// Multiplies a 4-element chunk by the circulant matrix
/// `[[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]`.
#[inline(always)]
fn mat4(x: &mut [BabyBear; 4]) {
    let t01 = x[0] + x[1];
    let t23 = x[2] + x[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + x[1];
    let t01233 = t0123 + x[3];
    // x[1] and x[3] must be overwritten before x[0] and x[2].
    x[3] = t01233 + x[0].double();
    x[1] = t01123 + x[2].double();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

/// Multiplies by `circ(2*M4, M4, M4, M4)`: apply M4 to each chunk, then add to
/// every element the sum of the elements in its position across all chunks.
#[inline(always)]
fn external_linear_layer(state: &mut [BabyBear; WIDTH]) {
    let (chunks, _) = state.as_chunks_mut::<4>();
    for chunk in chunks.iter_mut() {
        mat4(chunk);
    }
    let mut sums = [BabyBear::ZERO; 4];
    for chunk in chunks.iter() {
        for k in 0..4 {
            sums[k] += chunk[k];
        }
    }
    for chunk in chunks.iter_mut() {
        for k in 0..4 {
            chunk[k] += sums[k];
        }
    }
}

/// Multiplies by `1 + Diag(V)`, i.e. `s_i <- sum(s) + V_i * s_i`.
#[inline(always)]
fn internal_linear_layer(state: &mut [BabyBear; WIDTH]) {
    let sum = state.iter().fold(BabyBear::ZERO, |acc, &x| acc + x);
    for (s, &d) in state.iter_mut().zip(&INTERNAL_DIAG) {
        *s = sum + d * *s;
    }
}

/// `V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]`.
const INTERNAL_DIAG: [BabyBear; WIDTH] = {
    let n = BabyBear::new;
    let inv = BabyBear::inv_two_pow;
    [
        n(2).neg(),
        n(1),
        n(2),
        inv(1),
        n(3),
        n(4),
        inv(1).neg(),
        n(3).neg(),
        n(4).neg(),
        inv(8),
        inv(2),
        inv(3),
        inv(27),
        inv(8).neg(),
        inv(4).neg(),
        inv(27).neg(),
    ]
};

// Round constants, copied from Plonky3 (`baby-bear/src/poseidon2.rs`). Generated
// by the Grain LFSR with field_type=1, alpha=7, n=31, t=16, R_F=8, R_P=13.

const RC_EXTERNAL_INITIAL: [[BabyBear; WIDTH]; HALF_FULL_ROUNDS] = [
    BabyBear::new_array([
        0x69cbb6af, 0x46ad93f9, 0x60a00f4e, 0x6b1297cd, 0x23189afe, 0x732e7bef, 0x72c246de,
        0x2c941900, 0x0557eede, 0x1580496f, 0x3a3ea77b, 0x54f3f271, 0x0f49b029, 0x47872fe1,
        0x221e2e36, 0x1ab7202e,
    ]),
    BabyBear::new_array([
        0x487779a6, 0x3851c9d8, 0x38dc17c0, 0x209f8849, 0x268dcee8, 0x350c48da, 0x5b9ad32e,
        0x0523272b, 0x3f89055b, 0x01e894b2, 0x13ddedde, 0x1b2ef334, 0x7507d8b4, 0x6ceeb94e,
        0x52eb6ba2, 0x50642905,
    ]),
    BabyBear::new_array([
        0x05453f3f, 0x06349efc, 0x6922787c, 0x04bfff9c, 0x768c714a, 0x3e9ff21a, 0x15737c9c,
        0x2229c807, 0x0d47f88c, 0x097e0ecc, 0x27eadba0, 0x2d7d29e4, 0x3502aaa0, 0x0f475fd7,
        0x29fbda49, 0x018afffd,
    ]),
    BabyBear::new_array([
        0x0315b618, 0x6d4497d1, 0x1b171d9e, 0x52861abd, 0x2e5d0501, 0x3ec8646c, 0x6e5f250a,
        0x148ae8e6, 0x17f5fa4a, 0x3e66d284, 0x0051aa3b, 0x483f7913, 0x2cfe5f15, 0x023427ca,
        0x2cc78315, 0x1e36ea47,
    ]),
];

const RC_EXTERNAL_FINAL: [[BabyBear; WIDTH]; HALF_FULL_ROUNDS] = [
    BabyBear::new_array([
        0x7290a80d, 0x6f7e5329, 0x598ec8a8, 0x76a859a0, 0x6559e868, 0x657b83af, 0x13271d3f,
        0x1f876063, 0x0aeeae37, 0x706e9ca6, 0x46400cee, 0x72a05c26, 0x2c589c9e, 0x20bd37a7,
        0x6a2d3d10, 0x20523767,
    ]),
    BabyBear::new_array([
        0x5b8fe9c4, 0x2aa501d6, 0x1e01ac3e, 0x1448bc54, 0x5ce5ad1c, 0x4918a14d, 0x2c46a83f,
        0x4fcf6876, 0x61d8d5c8, 0x6ddf4ff9, 0x11fda4d3, 0x02933a8f, 0x170eaf81, 0x5a9c314f,
        0x49a12590, 0x35ec52a1,
    ]),
    BabyBear::new_array([
        0x58eb1611, 0x5e481e65, 0x367125c9, 0x0eba33ba, 0x1fc28ded, 0x066399ad, 0x0cbec0ea,
        0x75fd1af0, 0x50f5bf4e, 0x643d5f41, 0x6f4fe718, 0x5b3cbbde, 0x1e3afb3e, 0x296fb027,
        0x45e1547b, 0x4a8db2ab,
    ]),
    BabyBear::new_array([
        0x59986d19, 0x30bcdfa3, 0x1db63932, 0x1d7c2824, 0x53b33681, 0x0673b747, 0x038a98a3,
        0x2c5bce60, 0x351979cd, 0x5008fb73, 0x547bca78, 0x711af481, 0x3f93bf64, 0x644d987b,
        0x3c8bcd87, 0x608758b8,
    ]),
];

const RC_INTERNAL: [BabyBear; PARTIAL_ROUNDS] = BabyBear::new_array([
    0x5a8053c0, 0x693be639, 0x3858867d, 0x19334f6b, 0x128f0fd8, 0x4e2b1ccb, 0x61210ce0, 0x3c318939,
    0x0b5b2f22, 0x2edb11d5, 0x213effdf, 0x0cac4606, 0x241af16d,
]);

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer test from Plonky3's `test_default_babybear_poseidon2_width_16`.
    #[test]
    fn plonky3_vector() {
        let mut state = BabyBear::new_array([
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]);
        let expected = BabyBear::new_array([
            516096821, 90309867, 1101817252, 1660784290, 360715097, 1789519026, 1788910906,
            563338433, 319524748, 1741414159, 1650859320, 894311162, 1121347488, 1692793758,
            1052633829, 1344246938,
        ]);
        permute(&mut state);
        assert_eq!(state, expected);
    }

    /// Checks the chunked external layer against an explicit 16x16 matrix.
    #[test]
    fn external_layer_matches_dense_matrix() {
        const M4: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
        let input: [u32; WIDTH] =
            core::array::from_fn(|i| (i as u32 + 1) * 0x0123_4567 % crate::babybear::P);
        let mut state = BabyBear::new_array(input);
        external_linear_layer(&mut state);
        for i in 0..WIDTH {
            let mut acc = 0u64;
            for j in 0..WIDTH {
                let block = if i / 4 == j / 4 { 2 } else { 1 };
                acc += block * M4[i % 4][j % 4] * input[j] as u64;
            }
            assert_eq!(state[i].as_u32() as u64, acc % crate::babybear::P as u64);
        }
    }
}
