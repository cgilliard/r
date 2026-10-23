//! ML-DSA-44 (FIPS 204, the standardized Dilithium): lattice signatures.
//! Keys are 1312 bytes public and 2560 secret; signatures are 2420 bytes.
//! A key pair comes from one 32-byte seed. Signing retries until the
//! signature leaks nothing about the key (about four tries on average), so
//! it isn't constant time, but what varies is independent of the key.
//!
//! This is the lowest security level (category 2): `k = 4`, `l = 4`,
//! `eta = 2`, `tau = 39`, `gamma1 = 2^17`, `gamma2 = (q - 1) / 88`,
//! `omega = 80`.
//!
//! Both interfaces are here: [`sign_internal`] / [`verify_internal`] over
//! a raw message, and the pure [`sign`] / [`verify`], which prefix the
//! message with a context string (up to 255 bytes).

use super::keccak::{Sponge, shake256};

const N: usize = 256;
const Q: u32 = 8380417;
/// Bits dropped from `t`.
const D: u32 = 13;
const K: usize = 4;
const L: usize = 4;
const ETA: u32 = 2;
/// Nonzero coefficients of the challenge.
const TAU: usize = 39;
const GAMMA1: u32 = 1 << 17;
const GAMMA2: u32 = (Q - 1) / 88;
const BETA: u32 = TAU as u32 * ETA;
/// Most hint bits in a signature.
const OMEGA: usize = 80;
/// Bytes in the commitment hash `c~`.
const CT_LEN: usize = 32;

/// Bits per coefficient of `t1`, of `s1`/`s2`, of `t0`, of `z` and of `w1`.
const T1_BITS: usize = 10;
const S_BITS: usize = 3;
const T0_BITS: usize = D as usize;
const Z_BITS: usize = 18;
const W1_BITS: usize = 6;

/// Bytes in a public key: `rho || t1`.
pub const PK_LEN: usize = 32 + 32 * T1_BITS * K;
/// Bytes in a secret key: `rho || K || tr || s1 || s2 || t0`.
pub const SK_LEN: usize = 128 + 32 * (S_BITS * (K + L) + T0_BITS * K);
/// Bytes in a signature: `c~ || z || h`.
pub const SIG_LEN: usize = CT_LEN + 32 * Z_BITS * L + OMEGA + K;
/// Bytes of randomness for a hedged signature.
pub const RND_LEN: usize = 32;

/// Coefficients in `[0, q)`.
type Poly = [u32; N];

/// Which coefficients of each `w1` polynomial the signer's hint corrects.
type Hint = [[bool; N]; K];

/// `1753^BitRev8(i) mod q`, for the NTT.
const ZETAS: [u32; N] = {
    let mut z = [0u32; N];
    let mut i = 0;
    while i < N {
        let mut rev = 0;
        let mut b = 0;
        while b < 8 {
            rev |= ((i >> b) & 1) << (7 - b);
            b += 1;
        }
        let mut r = 1u64;
        let mut e = 0;
        while e < rev {
            r = r * 1753 % Q as u64;
            e += 1;
        }
        z[i] = r as u32;
        i += 1;
    }
    z
};

fn add(a: u32, b: u32) -> u32 {
    (a + b) % Q
}

fn sub(a: u32, b: u32) -> u32 {
    (a + Q - b) % Q
}

fn mul(a: u32, b: u32) -> u32 {
    (a as u64 * b as u64 % Q as u64) as u32
}

/// `x` as an integer in `(-q/2, q/2]`.
fn centered(x: u32) -> i32 {
    if x > Q / 2 {
        x as i32 - Q as i32
    } else {
        x as i32
    }
}

/// The integer `x` mod q.
fn modq(x: i32) -> u32 {
    x.rem_euclid(Q as i32) as u32
}

/// The largest `|coefficient|` of `f`, centered.
fn norm(f: &Poly) -> u32 {
    f.iter()
        .map(|&x| centered(x).unsigned_abs())
        .max()
        .unwrap_or(0)
}

fn ntt(f: &mut Poly) {
    let mut m = 0;
    let mut len = 128;
    while len >= 1 {
        for start in (0..N).step_by(2 * len) {
            m += 1;
            let zeta = ZETAS[m];
            for j in start..start + len {
                let t = mul(zeta, f[j + len]);
                f[j + len] = sub(f[j], t);
                f[j] = add(f[j], t);
            }
        }
        len /= 2;
    }
}

fn inv_ntt(f: &mut Poly) {
    let mut m = N;
    let mut len = 1;
    while len < N {
        for start in (0..N).step_by(2 * len) {
            m -= 1;
            let zeta = Q - ZETAS[m];
            for j in start..start + len {
                let t = f[j];
                f[j] = add(t, f[j + len]);
                f[j + len] = mul(zeta, sub(t, f[j + len]));
            }
        }
        len *= 2;
    }
    // 256^-1 mod q.
    for x in f.iter_mut() {
        *x = mul(*x, 8347681);
    }
}

fn poly_add(f: &Poly, g: &Poly) -> Poly {
    core::array::from_fn(|i| add(f[i], g[i]))
}

fn poly_sub(f: &Poly, g: &Poly) -> Poly {
    core::array::from_fn(|i| sub(f[i], g[i]))
}

/// The product of two polynomials in NTT form.
fn mul_ntt(f: &Poly, g: &Poly) -> Poly {
    core::array::from_fn(|i| mul(f[i], g[i]))
}

/// `A * v` for `v` in NTT form, back out of NTT form.
fn mat_mul(a: &[[Poly; L]; K], v: &[Poly; L]) -> [Poly; K] {
    core::array::from_fn(|i| {
        let mut acc = [0u32; N];
        for j in 0..L {
            acc = poly_add(&acc, &mul_ntt(&a[i][j], &v[j]));
        }
        inv_ntt(&mut acc);
        acc
    })
}

/// `c * v` for `c` and `v` in NTT form, back out of NTT form.
fn scale<const M: usize>(c: &Poly, v: &[Poly; M]) -> [Poly; M] {
    core::array::from_fn(|i| {
        let mut p = mul_ntt(c, &v[i]);
        inv_ntt(&mut p);
        p
    })
}

fn ntt_vec<const M: usize>(v: &[Poly; M]) -> [Poly; M] {
    core::array::from_fn(|i| {
        let mut p = v[i];
        ntt(&mut p);
        p
    })
}

/// SHAKE256 of `parts`, `M` bytes of it.
fn h<const M: usize>(parts: &[&[u8]]) -> [u8; M] {
    let mut out = [0u8; M];
    shake256(parts, &mut out);
    out
}

/// A uniform polynomial in NTT form from SHAKE128(`seed`).
fn rej_ntt_poly(seed: &[u8]) -> Poly {
    let mut xof = Sponge::shake128();
    xof.absorb(seed);
    let mut f = [0u32; N];
    let mut n = 0;
    let mut c = [0u8; 3];
    while n < N {
        xof.squeeze(&mut c);
        let z = c[0] as u32 | (c[1] as u32) << 8 | ((c[2] & 0x7f) as u32) << 16;
        if z < Q {
            f[n] = z;
            n += 1;
        }
    }
    f
}

/// A polynomial with coefficients in `[-eta, eta]` from SHAKE256(`seed`).
fn rej_bounded_poly(seed: &[u8]) -> Poly {
    let mut xof = Sponge::shake256();
    xof.absorb(seed);
    let mut f = [0u32; N];
    let mut n = 0;
    let mut z = [0u8; 1];
    while n < N {
        xof.squeeze(&mut z);
        for b in [z[0] & 15, z[0] >> 4] {
            // eta = 2: half-bytes below 15, reduced mod 5.
            if b < 15 && n < N {
                f[n] = modq(2 - (b % 5) as i32);
                n += 1;
            }
        }
    }
    f
}

/// The `k x l` matrix from `rho`, in NTT form.
fn expand_a(rho: &[u8; 32]) -> [[Poly; L]; K] {
    core::array::from_fn(|r| {
        core::array::from_fn(|s| rej_ntt_poly(&[&rho[..], &[s as u8, r as u8]].concat()))
    })
}

/// The secret vectors `s1` and `s2` from `rho'`.
fn expand_s(rho: &[u8; 64]) -> ([Poly; L], [Poly; K]) {
    let poly = |r: usize| rej_bounded_poly(&[&rho[..], &(r as u16).to_le_bytes()].concat());
    (
        core::array::from_fn(poly),
        core::array::from_fn(|r| poly(r + L)),
    )
}

/// The mask `y` from `rho''` and the counter `kappa`.
fn expand_mask(rho: &[u8; 64], kappa: u16) -> [Poly; L] {
    core::array::from_fn(|r| {
        let mut v = [0u8; 32 * Z_BITS];
        shake256(&[rho, &(kappa + r as u16).to_le_bytes()], &mut v);
        let raw = unpack(Z_BITS, &v);
        core::array::from_fn(|i| sub(GAMMA1, raw[i]))
    })
}

/// The challenge from `c~`: `tau` coefficients of `+-1`, the rest 0.
fn sample_in_ball(ct: &[u8; CT_LEN]) -> Poly {
    let mut xof = Sponge::shake256();
    xof.absorb(ct);
    let mut s = [0u8; 8];
    xof.squeeze(&mut s);
    let signs = u64::from_le_bytes(s);
    let mut c = [0u32; N];
    let mut j = [0u8; 1];
    for i in N - TAU..N {
        loop {
            xof.squeeze(&mut j);
            if j[0] as usize <= i {
                break;
            }
        }
        let j = j[0] as usize;
        c[i] = c[j];
        c[j] = if (signs >> (i + TAU - N)) & 1 == 1 {
            Q - 1
        } else {
            1
        };
    }
    c
}

/// `r = r1 * 2^d + r0` with `r0` in `(-2^(d-1), 2^(d-1)]`: `(r1, r0)`.
fn power2_round(r: u32) -> (u32, i32) {
    let mut r0 = (r & ((1 << D) - 1)) as i32;
    if r0 > 1 << (D - 1) {
        r0 -= 1 << D;
    }
    (((r as i32 - r0) >> D) as u32, r0)
}

/// `r = r1 * 2 gamma2 + r0` with `r0` in `(-gamma2, gamma2]`, except that
/// the top `r1` wraps to 0: `(r1, r0)`.
fn decompose(r: u32) -> (u32, i32) {
    let mut r0 = (r % (2 * GAMMA2)) as i32;
    if r0 > GAMMA2 as i32 {
        r0 -= 2 * GAMMA2 as i32;
    }
    if r as i32 - r0 == Q as i32 - 1 {
        (0, r0 - 1)
    } else {
        ((r as i32 - r0) as u32 / (2 * GAMMA2), r0)
    }
}

fn high_bits(r: u32) -> u32 {
    decompose(r).0
}

/// Whether adding `z` to `r` changes its high bits.
fn make_hint(z: u32, r: u32) -> bool {
    high_bits(r) != high_bits(add(r, z))
}

/// The high bits of `r`, corrected by the hint.
fn use_hint(hint: bool, r: u32) -> u32 {
    let m = (Q - 1) / (2 * GAMMA2);
    let (r1, r0) = decompose(r);
    match (hint, r0 > 0) {
        (false, _) => r1,
        (true, true) => (r1 + 1) % m,
        (true, false) => (r1 + m - 1) % m,
    }
}

/// `bits` bits per coefficient, little-endian.
fn pack(bits: usize, f: &[u32; N], out: &mut [u8]) {
    let (mut acc, mut n, mut o) = (0u64, 0, 0);
    for &x in f {
        acc |= (x as u64) << n;
        n += bits;
        while n >= 8 {
            out[o] = acc as u8;
            acc >>= 8;
            n -= 8;
            o += 1;
        }
    }
}

/// Inverse of [`pack`].
fn unpack(bits: usize, bytes: &[u8]) -> [u32; N] {
    let (mut acc, mut n, mut b) = (0u64, 0, 0);
    core::array::from_fn(|_| {
        while n < bits {
            acc |= (bytes[b] as u64) << n;
            b += 1;
            n += 8;
        }
        let x = (acc & ((1 << bits) - 1)) as u32;
        acc >>= bits;
        n -= bits;
        x
    })
}

/// Packs `bound - f` (each in `[0, 2^bits)`).
fn pack_signed(bits: usize, bound: u32, f: &Poly, out: &mut [u8]) {
    pack(bits, &core::array::from_fn(|i| sub(bound, f[i])), out);
}

/// Inverse of [`pack_signed`].
fn unpack_signed(bits: usize, bound: u32, bytes: &[u8]) -> Poly {
    let raw = unpack(bits, bytes);
    core::array::from_fn(|i| sub(bound, raw[i]))
}

/// The `w1` encoding hashed into the challenge.
fn w1_bytes(w1: &[Poly; K]) -> [u8; 32 * W1_BITS * K] {
    let mut out = [0u8; 32 * W1_BITS * K];
    for (i, p) in w1.iter().enumerate() {
        pack(
            W1_BITS,
            p,
            &mut out[32 * W1_BITS * i..32 * W1_BITS * (i + 1)],
        );
    }
    out
}

struct SecretKey {
    rho: [u8; 32],
    key: [u8; 32],
    tr: [u8; 64],
    s1: [Poly; L],
    s2: [Poly; K],
    t0: [Poly; K],
}

const S_BYTES: usize = 32 * S_BITS;
const T0_BYTES: usize = 32 * T0_BITS;

fn sk_decode(sk: &[u8; SK_LEN]) -> SecretKey {
    let s = |i: usize| unpack_signed(S_BITS, ETA, &sk[128 + S_BYTES * i..128 + S_BYTES * (i + 1)]);
    let t0_at = 128 + S_BYTES * (K + L);
    SecretKey {
        rho: sk[..32].try_into().expect("32 bytes"),
        key: sk[32..64].try_into().expect("32 bytes"),
        tr: sk[64..128].try_into().expect("64 bytes"),
        s1: core::array::from_fn(s),
        s2: core::array::from_fn(|i| s(L + i)),
        t0: core::array::from_fn(|i| {
            unpack_signed(
                T0_BITS,
                1 << (D - 1),
                &sk[t0_at + T0_BYTES * i..t0_at + T0_BYTES * (i + 1)],
            )
        }),
    }
}

/// `t1` and `rho` from a public key.
fn pk_decode(pk: &[u8; PK_LEN]) -> ([u8; 32], [Poly; K]) {
    let t1 = core::array::from_fn(|i| {
        unpack(
            T1_BITS,
            &pk[32 + 32 * T1_BITS * i..32 + 32 * T1_BITS * (i + 1)],
        )
    });
    (pk[..32].try_into().expect("32 bytes"), t1)
}

/// `c~`, `z` and the hint from a signature; `None` if the hint is
/// malformed (counts out of order, or positions not increasing or not
/// zero-padded).
fn sig_decode(sig: &[u8; SIG_LEN]) -> Option<([u8; CT_LEN], [Poly; L], Hint)> {
    let z_bytes = 32 * Z_BITS;
    let z = core::array::from_fn(|i| {
        unpack_signed(
            Z_BITS,
            GAMMA1,
            &sig[CT_LEN + z_bytes * i..CT_LEN + z_bytes * (i + 1)],
        )
    });
    let y = &sig[CT_LEN + z_bytes * L..];
    let mut hint: Hint = [[false; N]; K];
    let mut index = 0;
    for (i, row) in hint.iter_mut().enumerate() {
        let end = y[OMEGA + i] as usize;
        if end < index || end > OMEGA {
            return None;
        }
        for at in index..end {
            if at > index && y[at - 1] >= y[at] {
                return None;
            }
            row[y[at] as usize] = true;
        }
        index = end;
    }
    if y[index..OMEGA].iter().any(|&b| b != 0) {
        return None;
    }
    Some((sig[..CT_LEN].try_into().expect("32 bytes"), z, hint))
}

/// A key pair from a 32-byte random seed: (public key, secret key).
pub fn keygen(seed: &[u8; 32]) -> ([u8; PK_LEN], [u8; SK_LEN]) {
    let expanded: [u8; 128] = h(&[seed, &[K as u8, L as u8]]);
    let rho: [u8; 32] = expanded[..32].try_into().expect("32 bytes");
    let rho_s: [u8; 64] = expanded[32..96].try_into().expect("64 bytes");
    let a = expand_a(&rho);
    let (s1, s2) = expand_s(&rho_s);
    let as1 = mat_mul(&a, &ntt_vec(&s1));
    let mut pk = [0u8; PK_LEN];
    let mut sk = [0u8; SK_LEN];
    pk[..32].copy_from_slice(&rho);
    let t0_at = 128 + S_BYTES * (K + L);
    for i in 0..K {
        let t = poly_add(&as1[i], &s2[i]);
        let mut t1 = [0u32; N];
        let mut t0 = [0u32; N];
        for j in 0..N {
            let (hi, lo) = power2_round(t[j]);
            t1[j] = hi;
            t0[j] = modq(lo);
        }
        pack(
            T1_BITS,
            &t1,
            &mut pk[32 + 32 * T1_BITS * i..32 + 32 * T1_BITS * (i + 1)],
        );
        pack_signed(
            T0_BITS,
            1 << (D - 1),
            &t0,
            &mut sk[t0_at + T0_BYTES * i..t0_at + T0_BYTES * (i + 1)],
        );
    }
    sk[..32].copy_from_slice(&rho);
    sk[32..64].copy_from_slice(&expanded[96..]);
    sk[64..128].copy_from_slice(&h::<64>(&[&pk]));
    for (i, s) in s1.iter().chain(&s2).enumerate() {
        pack_signed(
            S_BITS,
            ETA,
            s,
            &mut sk[128 + S_BYTES * i..128 + S_BYTES * (i + 1)],
        );
    }
    (pk, sk)
}

/// Signs `msg` as is (FIPS 204's internal interface). `rnd` is 32 fresh
/// random bytes (hedged); `None` signs deterministically.
pub fn sign_internal(msg: &[u8], sk: &[u8; SK_LEN], rnd: Option<&[u8; RND_LEN]>) -> [u8; SIG_LEN] {
    let key = sk_decode(sk);
    let (s1, s2, t0) = (ntt_vec(&key.s1), ntt_vec(&key.s2), ntt_vec(&key.t0));
    let a = expand_a(&key.rho);
    let mu: [u8; 64] = h(&[&key.tr, msg]);
    let rho: [u8; 64] = h(&[&key.key, rnd.unwrap_or(&[0; RND_LEN]), &mu]);
    let mut kappa = 0u16;
    loop {
        let y = expand_mask(&rho, kappa);
        kappa += L as u16;
        let w = mat_mul(&a, &ntt_vec(&y));
        let w1: [Poly; K] = core::array::from_fn(|i| core::array::from_fn(|j| high_bits(w[i][j])));
        let ct: [u8; CT_LEN] = h(&[&mu, &w1_bytes(&w1)]);
        let mut c = sample_in_ball(&ct);
        ntt(&mut c);
        let cs1 = scale(&c, &s1);
        let cs2 = scale(&c, &s2);
        let z: [Poly; L] = core::array::from_fn(|i| poly_add(&y[i], &cs1[i]));
        if z.iter().any(|p| norm(p) >= GAMMA1 - BETA) {
            continue;
        }
        let r: [Poly; K] = core::array::from_fn(|i| poly_sub(&w[i], &cs2[i]));
        let low_too_big = r
            .iter()
            .flatten()
            .any(|&x| decompose(x).1.unsigned_abs() >= GAMMA2 - BETA);
        if low_too_big {
            continue;
        }
        let ct0 = scale(&c, &t0);
        if ct0.iter().any(|p| norm(p) >= GAMMA2) {
            continue;
        }
        let mut hint: Hint = [[false; N]; K];
        let mut count = 0;
        for i in 0..K {
            for j in 0..N {
                hint[i][j] = make_hint(sub(0, ct0[i][j]), add(r[i][j], ct0[i][j]));
                count += hint[i][j] as usize;
            }
        }
        if count > OMEGA {
            continue;
        }
        let mut sig = [0u8; SIG_LEN];
        sig[..CT_LEN].copy_from_slice(&ct);
        let z_bytes = 32 * Z_BITS;
        for (i, p) in z.iter().enumerate() {
            pack_signed(
                Z_BITS,
                GAMMA1,
                p,
                &mut sig[CT_LEN + z_bytes * i..CT_LEN + z_bytes * (i + 1)],
            );
        }
        let y = &mut sig[CT_LEN + z_bytes * L..];
        let mut index = 0;
        for (i, row) in hint.iter().enumerate() {
            for (j, _) in row.iter().enumerate().filter(|(_, b)| **b) {
                y[index] = j as u8;
                index += 1;
            }
            y[OMEGA + i] = index as u8;
        }
        return sig;
    }
}

/// Checks `sig` on `msg` as is (FIPS 204's internal interface).
pub fn verify_internal(msg: &[u8], sig: &[u8], pk: &[u8; PK_LEN]) -> bool {
    let Ok(sig) = <&[u8; SIG_LEN]>::try_from(sig) else {
        return false;
    };
    let Some((ct, z, hint)) = sig_decode(sig) else {
        return false;
    };
    if z.iter().any(|p| norm(p) >= GAMMA1 - BETA) {
        return false;
    }
    let (rho, t1) = pk_decode(pk);
    let a = expand_a(&rho);
    let tr: [u8; 64] = h(&[pk]);
    let mu: [u8; 64] = h(&[&tr, msg]);
    let mut c = sample_in_ball(&ct);
    ntt(&mut c);
    let az = mat_mul(&a, &ntt_vec(&z));
    let t1: [Poly; K] = core::array::from_fn(|i| core::array::from_fn(|j| t1[i][j] << D));
    let ct1 = scale(&c, &ntt_vec(&t1));
    let w1: [Poly; K] = core::array::from_fn(|i| {
        core::array::from_fn(|j| use_hint(hint[i][j], sub(az[i][j], ct1[i][j])))
    });
    h::<CT_LEN>(&[&mu, &w1_bytes(&w1)]) == ct
}

/// The message the pure interface signs: `0 || len(ctx) || ctx || msg`.
/// `None` if `ctx` is over 255 bytes.
fn with_context(msg: &[u8], ctx: &[u8]) -> Option<Vec<u8>> {
    let len = u8::try_from(ctx.len()).ok()?;
    Some([&[0, len][..], ctx, msg].concat())
}

/// Signs `msg` with context `ctx` (FIPS 204's pure ML-DSA). `None` if
/// `ctx` is over 255 bytes.
pub fn sign(
    msg: &[u8],
    ctx: &[u8],
    sk: &[u8; SK_LEN],
    rnd: Option<&[u8; RND_LEN]>,
) -> Option<[u8; SIG_LEN]> {
    Some(sign_internal(&with_context(msg, ctx)?, sk, rnd))
}

/// Checks a [`sign`] signature.
pub fn verify(msg: &[u8], ctx: &[u8], sig: &[u8], pk: &[u8; PK_LEN]) -> bool {
    with_context(msg, ctx).is_some_and(|m| verify_internal(&m, sig, pk))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// NIST's ACVP vectors for ML-DSA-44 (testdata/mldsa44.txt: one per
    /// line, `kind field=value ...`).
    fn vectors() -> Vec<(String, std::collections::HashMap<String, String>)> {
        include_str!("testdata/mldsa44.txt")
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let mut parts = l.split(' ');
                let kind = parts.next().unwrap().to_string();
                let fields = parts
                    .map(|p| {
                        let (k, v) = p.split_once('=').unwrap();
                        (k.to_string(), v.to_string())
                    })
                    .collect();
                (kind, fields)
            })
            .collect()
    }

    #[test]
    fn matches_nist_vectors() {
        let mut counts = std::collections::HashMap::new();
        for (kind, f) in vectors() {
            let b = |k: &str| unhex(&f[k]);
            let pure = f.get("iface").map(String::as_str) == Some("pure");
            *counts
                .entry(format!("{kind} {}", f.get("iface").map_or("", |s| s)))
                .or_insert(0) += 1;
            match kind.as_str() {
                "keygen" => {
                    let (pk, sk) = keygen(b("seed").as_slice().try_into().unwrap());
                    assert_eq!(pk.to_vec(), b("pk"));
                    assert_eq!(sk.to_vec(), b("sk"));
                }
                "sign" => {
                    let sk: &[u8; SK_LEN] = &b("sk").try_into().unwrap();
                    let rnd: Option<[u8; RND_LEN]> =
                        f.get("rnd").map(|_| b("rnd").try_into().unwrap());
                    let sig = if pure {
                        sign(&b("msg"), &b("ctx"), sk, rnd.as_ref()).unwrap()
                    } else {
                        sign_internal(&b("msg"), sk, rnd.as_ref())
                    };
                    assert_eq!(sig.to_vec(), b("sig"));
                }
                "verify" => {
                    let pk: &[u8; PK_LEN] = &b("pk").try_into().unwrap();
                    let ok = if pure {
                        verify(&b("msg"), &b("ctx"), &b("sig"), pk)
                    } else {
                        verify_internal(&b("msg"), &b("sig"), pk)
                    };
                    assert_eq!(ok, b("pass") == [1], "{f:?}");
                }
                _ => panic!("unknown vector kind {kind}"),
            }
        }
        for kind in [
            "keygen ",
            "sign pure",
            "sign internal",
            "verify pure",
            "verify internal",
        ] {
            assert!(
                counts.get(kind).copied().unwrap_or(0) > 0,
                "no {kind} vectors"
            );
        }
    }

    #[test]
    fn signatures_verify_only_for_their_message_context_and_key() {
        let (pk, sk) = keygen(&[1; 32]);
        let sig = sign(b"hello", b"relay", &sk, Some(&[2; RND_LEN])).unwrap();
        assert!(verify(b"hello", b"relay", &sig, &pk));
        assert!(!verify(b"hellp", b"relay", &sig, &pk));
        assert!(!verify(b"hello", b"other", &sig, &pk));
        assert!(!verify(b"hello", b"relay", &sig[1..], &pk));
        let (other, _) = keygen(&[3; 32]);
        assert!(!verify(b"hello", b"relay", &sig, &other));
        let mut bad = sig;
        bad[100] ^= 1;
        assert!(!verify(b"hello", b"relay", &bad, &pk));
        assert!(sign(b"hello", &[0; 256], &sk, None).is_none());
        // Hedged signatures differ; deterministic ones repeat.
        assert_ne!(
            sign(b"hello", b"", &sk, Some(&[4; RND_LEN])),
            sign(b"hello", b"", &sk, Some(&[5; RND_LEN]))
        );
        assert_eq!(
            sign(b"hello", b"", &sk, None),
            sign(b"hello", b"", &sk, None)
        );
        assert_eq!((PK_LEN, SK_LEN, SIG_LEN), (1312, 2560, 2420));
    }
}
