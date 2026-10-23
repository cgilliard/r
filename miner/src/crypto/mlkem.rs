//! ML-KEM-512 (FIPS 203, the standardized Kyber) key encapsulation: a key
//! pair from 64 random bytes, then anyone holding the encapsulation key
//! makes a ciphertext and a 32-byte shared secret, which only the holder
//! of the decapsulation key recovers. Decapsulation never fails: a bad
//! ciphertext gives an unrelated, pseudorandom secret (implicit
//! rejection).
//!
//! This is the lowest security level (category 1, about AES-128): `k = 2`,
//! `eta1 = 3`, `eta2 = 2`, `du = 10`, `dv = 4`.

use super::keccak::{Sponge, sha3_256, sha3_512, shake256};

const N: usize = 256;
const Q: u32 = 3329;
const K: usize = 2;
const ETA1: usize = 3;
const ETA2: usize = 2;
const DU: u32 = 10;
const DV: u32 = 4;

/// Bytes in an encapsulation (public) key.
pub const EK_LEN: usize = 384 * K + 32;
/// Bytes in a decapsulation (secret) key.
pub const DK_LEN: usize = 768 * K + 96;
/// Bytes in a ciphertext.
pub const CT_LEN: usize = 32 * (DU as usize * K + DV as usize);
/// Bytes in the shared secret.
pub const SECRET_LEN: usize = 32;

type Poly = [u16; N];

/// `17^BitRev7(i) mod q`, for the NTT.
const ZETAS: [u16; 128] = {
    let mut z = [0u16; 128];
    let mut i = 0;
    while i < 128 {
        let mut rev = 0;
        let mut b = 0;
        while b < 7 {
            rev |= ((i >> b) & 1) << (6 - b);
            b += 1;
        }
        z[i] = pow17(rev as u32) as u16;
        i += 1;
    }
    z
};

/// `17^(2 BitRev7(i) + 1) mod q`, for base-case multiplication.
const GAMMAS: [u16; 128] = {
    let mut g = [0u16; 128];
    let mut i = 0;
    while i < 128 {
        let mut rev = 0;
        let mut b = 0;
        while b < 7 {
            rev |= ((i >> b) & 1) << (6 - b);
            b += 1;
        }
        g[i] = pow17(2 * rev as u32 + 1) as u16;
        i += 1;
    }
    g
};

const fn pow17(e: u32) -> u32 {
    let mut r = 1u32;
    let mut i = 0;
    while i < e {
        r = r * 17 % Q;
        i += 1;
    }
    r
}

fn add(a: u16, b: u16) -> u16 {
    ((a as u32 + b as u32) % Q) as u16
}

fn sub(a: u16, b: u16) -> u16 {
    ((a as u32 + Q - b as u32) % Q) as u16
}

fn mul(a: u16, b: u16) -> u16 {
    (a as u32 * b as u32 % Q) as u16
}

fn ntt(f: &mut Poly) {
    let mut i = 1;
    let mut len = 128;
    while len >= 2 {
        for start in (0..N).step_by(2 * len) {
            let zeta = ZETAS[i];
            i += 1;
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
    let mut i = 127;
    let mut len = 2;
    while len <= 128 {
        for start in (0..N).step_by(2 * len) {
            let zeta = ZETAS[i];
            i -= 1;
            for j in start..start + len {
                let t = f[j];
                f[j] = add(t, f[j + len]);
                f[j + len] = mul(zeta, sub(f[j + len], t));
            }
        }
        len *= 2;
    }
    // 128^-1 mod q.
    for x in f.iter_mut() {
        *x = mul(*x, 3303);
    }
}

/// The product of two polynomials in NTT form.
fn mul_ntt(f: &Poly, g: &Poly) -> Poly {
    let mut h = [0u16; N];
    for i in 0..128 {
        let (a0, a1, b0, b1) = (f[2 * i], f[2 * i + 1], g[2 * i], g[2 * i + 1]);
        h[2 * i] = add(mul(a0, b0), mul(mul(a1, b1), GAMMAS[i]));
        h[2 * i + 1] = add(mul(a0, b1), mul(a1, b0));
    }
    h
}

fn poly_add(f: &Poly, g: &Poly) -> Poly {
    core::array::from_fn(|i| add(f[i], g[i]))
}

/// A uniform polynomial in NTT form from SHAKE128(`rho || j || i`).
fn sample_ntt(rho: &[u8; 32], j: u8, i: u8) -> Poly {
    let mut xof = Sponge::shake128();
    xof.absorb(rho);
    xof.absorb(&[j, i]);
    let mut f = [0u16; N];
    let mut n = 0;
    let mut c = [0u8; 3];
    while n < N {
        xof.squeeze(&mut c);
        let d1 = c[0] as u16 | (c[1] as u16 & 15) << 8;
        let d2 = (c[1] >> 4) as u16 | (c[2] as u16) << 4;
        if (d1 as u32) < Q {
            f[n] = d1;
            n += 1;
        }
        if (d2 as u32) < Q && n < N {
            f[n] = d2;
            n += 1;
        }
    }
    f
}

/// A centered-binomial polynomial with parameter `eta` from
/// PRF(`s`, `b`) = SHAKE256(`s || b`).
fn sample_cbd(eta: usize, s: &[u8; 32], b: u8) -> Poly {
    let mut buf = [0u8; 64 * ETA1];
    let buf = &mut buf[..64 * eta];
    shake256(&[s, &[b]], buf);
    let bit = |k: usize| ((buf[k / 8] >> (k % 8)) & 1) as u16;
    core::array::from_fn(|i| {
        let x: u16 = (0..eta).map(|j| bit(2 * i * eta + j)).sum();
        let y: u16 = (0..eta).map(|j| bit(2 * i * eta + eta + j)).sum();
        sub(x, y)
    })
}

/// The `k x k` matrix from `rho` (row `i`, column `j`).
fn matrix(rho: &[u8; 32]) -> [[Poly; K]; K] {
    core::array::from_fn(|i| core::array::from_fn(|j| sample_ntt(rho, j as u8, i as u8)))
}

/// `d` bits per coefficient, little-endian.
fn encode(d: u32, f: &Poly, out: &mut [u8]) {
    out.fill(0);
    for (i, &x) in f.iter().enumerate() {
        for b in 0..d as usize {
            let k = i * d as usize + b;
            out[k / 8] |= (((x >> b) & 1) as u8) << (k % 8);
        }
    }
}

/// Inverse of [`encode`] (coefficients may reach `2^d - 1`).
fn decode(d: u32, bytes: &[u8]) -> Poly {
    core::array::from_fn(|i| {
        (0..d as usize).fold(0u16, |acc, b| {
            let k = i * d as usize + b;
            acc | (((bytes[k / 8] >> (k % 8)) & 1) as u16) << b
        })
    })
}

fn compress(d: u32, f: &Poly) -> Poly {
    core::array::from_fn(|i| (((((f[i] as u32) << d) + Q / 2) / Q) & ((1 << d) - 1)) as u16)
}

fn decompress(d: u32, f: &Poly) -> Poly {
    core::array::from_fn(|i| ((f[i] as u32 * Q + (1 << (d - 1))) >> d) as u16)
}

/// The vector of 12-bit polynomials in `bytes`; `None` if a coefficient
/// isn't below q.
fn decode_vec12(bytes: &[u8]) -> Option<[Poly; K]> {
    let mut out = [[0u16; N]; K];
    for (i, p) in out.iter_mut().enumerate() {
        *p = decode(12, &bytes[384 * i..384 * (i + 1)]);
        if p.iter().any(|&x| x as u32 >= Q) {
            return None;
        }
    }
    Some(out)
}

/// K-PKE key generation from `d`: (encryption key, decryption key).
fn pke_keygen(d: &[u8; 32]) -> ([u8; EK_LEN], [u8; 384 * K]) {
    let g = sha3_512(&[d, &[K as u8]]);
    let rho: [u8; 32] = g[..32].try_into().expect("32 bytes");
    let sigma: [u8; 32] = g[32..].try_into().expect("32 bytes");
    let a = matrix(&rho);
    let mut s: [Poly; K] = core::array::from_fn(|i| sample_cbd(ETA1, &sigma, i as u8));
    let mut e: [Poly; K] = core::array::from_fn(|i| sample_cbd(ETA1, &sigma, (K + i) as u8));
    s.iter_mut().for_each(ntt);
    e.iter_mut().for_each(ntt);
    let mut ek = [0u8; EK_LEN];
    let mut dk = [0u8; 384 * K];
    for i in 0..K {
        let mut t = e[i];
        for j in 0..K {
            t = poly_add(&t, &mul_ntt(&a[i][j], &s[j]));
        }
        encode(12, &t, &mut ek[384 * i..384 * (i + 1)]);
        encode(12, &s[i], &mut dk[384 * i..384 * (i + 1)]);
    }
    ek[384 * K..].copy_from_slice(&rho);
    (ek, dk)
}

/// K-PKE encryption of `m` under `ek` with randomness `r`; `None` if `ek`
/// isn't well formed.
fn pke_encrypt(ek: &[u8; EK_LEN], m: &[u8; 32], r: &[u8; 32]) -> Option<[u8; CT_LEN]> {
    let t = decode_vec12(&ek[..384 * K])?;
    let rho: [u8; 32] = ek[384 * K..].try_into().expect("32 bytes");
    let a = matrix(&rho);
    let mut y: [Poly; K] = core::array::from_fn(|i| sample_cbd(ETA1, r, i as u8));
    let e1: [Poly; K] = core::array::from_fn(|i| sample_cbd(ETA2, r, (K + i) as u8));
    let e2 = sample_cbd(ETA2, r, 2 * K as u8);
    y.iter_mut().for_each(ntt);
    let mut c = [0u8; CT_LEN];
    let du_bytes = 32 * DU as usize;
    for i in 0..K {
        let mut u = [0u16; N];
        for j in 0..K {
            u = poly_add(&u, &mul_ntt(&a[j][i], &y[j]));
        }
        inv_ntt(&mut u);
        let u = poly_add(&u, &e1[i]);
        encode(
            DU,
            &compress(DU, &u),
            &mut c[du_bytes * i..du_bytes * (i + 1)],
        );
    }
    let mut v = [0u16; N];
    for j in 0..K {
        v = poly_add(&v, &mul_ntt(&t[j], &y[j]));
    }
    inv_ntt(&mut v);
    let mu = decompress(1, &decode(1, m));
    let v = poly_add(&poly_add(&v, &e2), &mu);
    encode(DV, &compress(DV, &v), &mut c[du_bytes * K..]);
    Some(c)
}

/// K-PKE decryption of `c` with `dk`.
fn pke_decrypt(dk: &[u8], c: &[u8; CT_LEN]) -> [u8; 32] {
    let du_bytes = 32 * DU as usize;
    let mut w = decompress(DV, &decode(DV, &c[du_bytes * K..]));
    let mut acc = [0u16; N];
    for i in 0..K {
        let mut u = decompress(DU, &decode(DU, &c[du_bytes * i..du_bytes * (i + 1)]));
        ntt(&mut u);
        let s = decode(12, &dk[384 * i..384 * (i + 1)]);
        acc = poly_add(&acc, &mul_ntt(&s, &u));
    }
    inv_ntt(&mut acc);
    for (x, a) in w.iter_mut().zip(acc) {
        *x = sub(*x, a);
    }
    let mut m = [0u8; 32];
    encode(1, &compress(1, &w), &mut m);
    m
}

/// A key pair from seeds `d` and `z` (64 random bytes in all):
/// (encapsulation key, decapsulation key).
pub fn keygen(d: &[u8; 32], z: &[u8; 32]) -> ([u8; EK_LEN], [u8; DK_LEN]) {
    let (ek, dk_pke) = pke_keygen(d);
    let mut dk = [0u8; DK_LEN];
    dk[..384 * K].copy_from_slice(&dk_pke);
    dk[384 * K..768 * K + 32].copy_from_slice(&ek);
    dk[768 * K + 32..768 * K + 64].copy_from_slice(&sha3_256(&[&ek]));
    dk[768 * K + 64..].copy_from_slice(z);
    (ek, dk)
}

/// Encapsulates to `ek` with the random `m`: (ciphertext, shared secret).
/// `None` if `ek` isn't a valid key (a coefficient not below q).
pub fn encapsulate(ek: &[u8; EK_LEN], m: &[u8; 32]) -> Option<([u8; CT_LEN], [u8; SECRET_LEN])> {
    let g = sha3_512(&[m, &sha3_256(&[ek])]);
    let r: [u8; 32] = g[32..].try_into().expect("32 bytes");
    let c = pke_encrypt(ek, m, &r)?;
    Some((c, g[..32].try_into().expect("32 bytes")))
}

/// The shared secret in `c`, with `dk`: a pseudorandom one if `c` isn't
/// an encapsulation to `dk`'s key. `None` if `dk` fails its hash check.
pub fn decapsulate(dk: &[u8; DK_LEN], c: &[u8; CT_LEN]) -> Option<[u8; SECRET_LEN]> {
    if !check_dk(dk) {
        return None;
    }
    let dk_pke = &dk[..384 * K];
    let ek: &[u8; EK_LEN] = dk[384 * K..768 * K + 32].try_into().expect("ek");
    let h = &dk[768 * K + 32..768 * K + 64];
    let z = &dk[768 * K + 64..];
    let m = pke_decrypt(dk_pke, c);
    let g = sha3_512(&[&m, h]);
    let r: [u8; 32] = g[32..].try_into().expect("32 bytes");
    let mut reject = [0u8; 32];
    shake256(&[z, c], &mut reject);
    let again = pke_encrypt(ek, &m, &r)?;
    // Constant time: pick the secret by a mask.
    let differ = again.iter().zip(c).fold(0u8, |acc, (a, b)| acc | (a ^ b));
    let mask = ((differ as u16).wrapping_sub(1) >> 8) as u8; // 0xff if equal
    Some(core::array::from_fn(|i| {
        (g[i] & mask) | (reject[i] & !mask)
    }))
}

/// Whether `ek` is well formed (every coefficient below q).
pub fn check_ek(ek: &[u8; EK_LEN]) -> bool {
    decode_vec12(&ek[..384 * K]).is_some()
}

/// Whether `dk` holds the hash of the key it contains.
pub fn check_dk(dk: &[u8; DK_LEN]) -> bool {
    sha3_256(&[&dk[384 * K..768 * K + 32]]) == dk[768 * K + 32..768 * K + 64]
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

    /// NIST's ACVP vectors for ML-KEM-512 (testdata/mlkem512.txt: one per
    /// line, `kind field=hex ...`).
    fn vectors() -> Vec<(String, std::collections::HashMap<String, Vec<u8>>)> {
        include_str!("testdata/mlkem512.txt")
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let mut parts = l.split(' ');
                let kind = parts.next().unwrap().to_string();
                let fields = parts
                    .map(|p| {
                        let (k, v) = p.split_once('=').unwrap();
                        (k.to_string(), unhex(v))
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
            *counts.entry(kind.clone()).or_insert(0) += 1;
            match kind.as_str() {
                "keygen" => {
                    let (ek, dk) = keygen(
                        f["d"].as_slice().try_into().unwrap(),
                        f["z"].as_slice().try_into().unwrap(),
                    );
                    assert_eq!(ek.to_vec(), f["ek"]);
                    assert_eq!(dk.to_vec(), f["dk"]);
                }
                "encap" => {
                    let (c, k) = encapsulate(
                        f["ek"].as_slice().try_into().unwrap(),
                        f["m"].as_slice().try_into().unwrap(),
                    )
                    .unwrap();
                    assert_eq!(c.to_vec(), f["c"]);
                    assert_eq!(k.to_vec(), f["k"]);
                }
                "decap" => {
                    let k = decapsulate(
                        f["dk"].as_slice().try_into().unwrap(),
                        f["c"].as_slice().try_into().unwrap(),
                    )
                    .unwrap();
                    assert_eq!(k.to_vec(), f["k"]);
                }
                "ekcheck" => assert_eq!(
                    check_ek(f["ek"].as_slice().try_into().unwrap()),
                    f["pass"] == [1]
                ),
                "dkcheck" => assert_eq!(
                    check_dk(f["dk"].as_slice().try_into().unwrap()),
                    f["pass"] == [1]
                ),
                _ => panic!("unknown vector kind {kind}"),
            }
        }
        for kind in ["keygen", "encap", "decap", "ekcheck", "dkcheck"] {
            assert!(
                counts.get(kind).copied().unwrap_or(0) > 0,
                "no {kind} vectors"
            );
        }
    }

    #[test]
    fn both_sides_agree_and_a_bad_ciphertext_gets_another_secret() {
        let (ek, dk) = keygen(&[1; 32], &[2; 32]);
        let (c, k) = encapsulate(&ek, &[3; 32]).unwrap();
        assert_eq!(decapsulate(&dk, &c), Some(k));
        let mut bad = c;
        bad[100] ^= 1;
        let k2 = decapsulate(&dk, &bad).unwrap();
        assert_ne!(k2, k);
        // The rejection secret is J(z || c): deterministic.
        assert_eq!(decapsulate(&dk, &bad), Some(k2));
        assert_eq!((EK_LEN, DK_LEN, CT_LEN), (800, 1632, 768));
    }
}
