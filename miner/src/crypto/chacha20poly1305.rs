//! ChaCha20-Poly1305 authenticated encryption (RFC 8439): a 256-bit key, a
//! 96-bit nonce (never reused with one key), optional associated data
//! (authenticated, not encrypted), and a 16-byte tag appended to the
//! ciphertext.

pub const KEY_LEN: usize = 32;
pub const NONCE_LEN: usize = 12;
pub const TAG_LEN: usize = 16;

fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

/// The ChaCha20 block for `key`, `counter` and `nonce`.
pub fn block(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN]) -> [u8; 64] {
    let word = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]);
    for i in 0..8 {
        s[4 + i] = word(&key[4 * i..4 * i + 4]);
    }
    s[12] = counter;
    for i in 0..3 {
        s[13 + i] = word(&nonce[4 * i..4 * i + 4]);
    }
    let mut w = s;
    for _ in 0..10 {
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[4 * i..4 * i + 4].copy_from_slice(&w[i].wrapping_add(s[i]).to_le_bytes());
    }
    out
}

/// XORs `data` with the ChaCha20 keystream from block `counter` on.
pub fn chacha20(key: &[u8; KEY_LEN], counter: u32, nonce: &[u8; NONCE_LEN], data: &mut [u8]) {
    for (i, chunk) in data.chunks_mut(64).enumerate() {
        let ks = block(key, counter.wrapping_add(i as u32), nonce);
        for (d, k) in chunk.iter_mut().zip(ks) {
            *d ^= k;
        }
    }
}

/// Poly1305 over `msg` with the one-time key `key` (`r || s`), in 26-bit
/// limbs.
pub fn poly1305(key: &[u8; 32], msg: &[u8]) -> [u8; TAG_LEN] {
    let le32 = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
    // r, clamped.
    let r0 = le32(&key[0..4]) & 0x3ffffff;
    let r1 = (le32(&key[3..7]) >> 2) & 0x3ffff03;
    let r2 = (le32(&key[6..10]) >> 4) & 0x3ffc0ff;
    let r3 = (le32(&key[9..13]) >> 6) & 0x3f03fff;
    let r4 = (le32(&key[12..16]) >> 8) & 0x00fffff;
    let (s1, s2, s3, s4) = (r1 * 5, r2 * 5, r3 * 5, r4 * 5);
    let mut h = [0u32; 5];
    for chunk in msg.chunks(16) {
        let mut block = [0u8; 17];
        block[..chunk.len()].copy_from_slice(chunk);
        block[chunk.len()] = 1;
        let hibit = (block[16] as u32) << 24;
        h[0] += le32(&block[0..4]) & 0x3ffffff;
        h[1] += (le32(&block[3..7]) >> 2) & 0x3ffffff;
        h[2] += (le32(&block[6..10]) >> 4) & 0x3ffffff;
        h[3] += (le32(&block[9..13]) >> 6) & 0x3ffffff;
        h[4] += (le32(&block[12..16]) >> 8) | hibit;
        let m = |a: u32, b: u32| a as u64 * b as u64;
        let d0 = m(h[0], r0) + m(h[1], s4) + m(h[2], s3) + m(h[3], s2) + m(h[4], s1);
        let mut d1 = m(h[0], r1) + m(h[1], r0) + m(h[2], s4) + m(h[3], s3) + m(h[4], s2);
        let mut d2 = m(h[0], r2) + m(h[1], r1) + m(h[2], r0) + m(h[3], s4) + m(h[4], s3);
        let mut d3 = m(h[0], r3) + m(h[1], r2) + m(h[2], r1) + m(h[3], r0) + m(h[4], s4);
        let mut d4 = m(h[0], r4) + m(h[1], r3) + m(h[2], r2) + m(h[3], r1) + m(h[4], r0);
        let mut c = d0 >> 26;
        h[0] = d0 as u32 & 0x3ffffff;
        d1 += c;
        c = d1 >> 26;
        h[1] = d1 as u32 & 0x3ffffff;
        d2 += c;
        c = d2 >> 26;
        h[2] = d2 as u32 & 0x3ffffff;
        d3 += c;
        c = d3 >> 26;
        h[3] = d3 as u32 & 0x3ffffff;
        d4 += c;
        c = d4 >> 26;
        h[4] = d4 as u32 & 0x3ffffff;
        h[0] += c as u32 * 5;
        let c = h[0] >> 26;
        h[0] &= 0x3ffffff;
        h[1] += c;
    }
    // Full carry, then h mod 2^130 - 5.
    let mut c = h[1] >> 26;
    h[1] &= 0x3ffffff;
    for x in &mut h[2..] {
        *x += c;
        c = *x >> 26;
        *x &= 0x3ffffff;
    }
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x3ffffff;
    h[1] += c;
    // g = h + 5 - 2^130: if it doesn't borrow, h >= p and g is the result.
    let mut g = [0u32; 5];
    let mut c = 5u32;
    for i in 0..5 {
        g[i] = h[i] + c;
        c = g[i] >> 26;
        g[i] &= 0x3ffffff;
    }
    // c is the carry out of bit 130: 1 exactly when h + 5 >= 2^130.
    let mask = c.wrapping_neg();
    for i in 0..5 {
        h[i] = (h[i] & !mask) | (g[i] & mask);
    }
    // h as 128 bits, plus s.
    let h0 = (h[0] | (h[1] << 26)) as u64;
    let h1 = ((h[1] >> 6) | (h[2] << 20)) as u64;
    let h2 = ((h[2] >> 12) | (h[3] << 14)) as u64;
    let h3 = ((h[3] >> 18) | (h[4] << 8)) as u64;
    let mut out = [0u8; TAG_LEN];
    let mut f = 0u64;
    for (i, hi) in [h0, h1, h2, h3].into_iter().enumerate() {
        f += (hi & 0xffff_ffff) + le32(&key[16 + 4 * i..20 + 4 * i]) as u64;
        out[4 * i..4 * i + 4].copy_from_slice(&(f as u32).to_le_bytes());
        f >>= 32;
    }
    out
}

/// The Poly1305 input for `aad` and `ciphertext`: each padded to 16
/// bytes, then their lengths.
fn mac_data(aad: &[u8], ciphertext: &[u8]) -> Vec<u8> {
    let pad = |n: usize| (16 - n % 16) % 16;
    let mut m = Vec::with_capacity(aad.len() + ciphertext.len() + 48);
    m.extend_from_slice(aad);
    m.resize(m.len() + pad(aad.len()), 0);
    m.extend_from_slice(ciphertext);
    m.resize(m.len() + pad(ciphertext.len()), 0);
    m.extend((aad.len() as u64).to_le_bytes());
    m.extend((ciphertext.len() as u64).to_le_bytes());
    m
}

fn one_time_key(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN]) -> [u8; 32] {
    block(key, 0, nonce)[..32].try_into().expect("32 bytes")
}

/// Encrypts `plaintext`: the ciphertext with its tag appended.
pub fn seal(key: &[u8; KEY_LEN], nonce: &[u8; NONCE_LEN], aad: &[u8], plaintext: &[u8]) -> Vec<u8> {
    let mut out = plaintext.to_vec();
    chacha20(key, 1, nonce, &mut out);
    let tag = poly1305(&one_time_key(key, nonce), &mac_data(aad, &out));
    out.extend_from_slice(&tag);
    out
}

/// Decrypts `sealed` (ciphertext and tag); `None` if the tag doesn't
/// match.
pub fn open(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    sealed: &[u8],
) -> Option<Vec<u8>> {
    let split = sealed.len().checked_sub(TAG_LEN)?;
    let (ciphertext, tag) = sealed.split_at(split);
    let want = poly1305(&one_time_key(key, nonce), &mac_data(aad, ciphertext));
    // Constant time.
    if want.iter().zip(tag).fold(0u8, |acc, (a, b)| acc | (a ^ b)) != 0 {
        return None;
    }
    let mut out = ciphertext.to_vec();
    chacha20(key, 1, nonce, &mut out);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    const KEY: [u8; 32] = {
        let mut k = [0u8; 32];
        let mut i = 0;
        while i < 32 {
            k[i] = i as u8;
            i += 1;
        }
        k
    };
    const NONCE: [u8; 12] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111];

    fn bytes(n: usize, mul: usize, add: usize) -> Vec<u8> {
        (0..n).map(|i| ((i * mul + add) % 256) as u8).collect()
    }

    /// Values from Python's `cryptography` (OpenSSL).
    #[test]
    fn matches_openssl() {
        assert_eq!(
            hex(&block(&KEY, 7, &NONCE)),
            "55242666718e450195ce99146d343f75543fd32d377b900f53ff29b05eb73beb\
             a0dcfbba279a81a6af4b97081f52c966b1ab77a323e3ad69a18b5fcea1d7da4f"
        );
        let sealed = [
            (0, "a33126986fcb1312c2476c125059ffe4"),
            (1, "339ad5b40fe0fa62b7dda8dff67429f21a"),
            (
                15,
                "3337feae1b07b7b204c028238913611160cb5ef4b97377ffda347ed977f029",
            ),
            (
                16,
                "3337feae1b07b7b204c028238913614b954583751fe8b9ebb066f4a0db9684ff",
            ),
            (
                17,
                "3337feae1b07b7b204c028238913614b87f8d782047979cf7db9ed7198fe4dc1d2",
            ),
            (
                64,
                "3337feae1b07b7b204c028238913614b87d0a760ecc25a7d2931dbfe91b8653b\
                 2dbf97a4ccb8ebb34011d7d1b3d97170132d9a294c19a7599151e6ba4fdcd859\
                 f42cf1b7e458646742ab630d0f2d486d",
            ),
            (
                65,
                "3337feae1b07b7b204c028238913614b87d0a760ecc25a7d2931dbfe91b8653b\
                 2dbf97a4ccb8ebb34011d7d1b3d97170132d9a294c19a7599151e6ba4fdcd859\
                 d5658cdd0889703129df8dc223f4042c31",
            ),
        ];
        for (n, want) in sealed {
            let pt = bytes(n, 31, 7);
            let aad = bytes(n % 23, 5, 1);
            let s = seal(&KEY, &NONCE, &aad, &pt);
            assert_eq!(hex(&s), want, "{n}");
            assert_eq!(open(&KEY, &NONCE, &aad, &s), Some(pt), "{n}");
        }
        let pt = bytes(200, 31, 7);
        let s = seal(&KEY, &NONCE, &bytes(16, 5, 1), &pt);
        assert_eq!(hex(&s[200..]), "ad5c00e546f96788f9c7b79735e422e8");

        let pk: [u8; 32] = bytes(32, 17, 3).try_into().unwrap();
        for (n, want) in [
            (0, "132435465768798a9bacbdcedff00112"),
            (1, "194f9377e55ca408b66db599867ec609"),
            (16, "09f51f88e3374efa347f13964215ba58"),
            (33, "9e957615dc83660baeb5da7b03a7f36c"),
            (100, "109d9549a153f2a00458a26273830d13"),
        ] {
            assert_eq!(hex(&poly1305(&pk, &bytes(n, 7, 2))), want, "{n}");
        }
        assert_eq!(
            hex(&poly1305(&[0xff; 32], &[0xff; 64])),
            "900fe32bc15fa8d7bca8efe4c7e37eb1"
        );
    }

    #[test]
    fn tampering_is_refused() {
        let s = seal(&KEY, &NONCE, b"header", b"onion layer");
        assert_eq!(open(&KEY, &NONCE, b"header", &s).unwrap(), b"onion layer");
        for i in 0..s.len() {
            let mut bad = s.clone();
            bad[i] ^= 1;
            assert_eq!(open(&KEY, &NONCE, b"header", &bad), None, "byte {i}");
        }
        assert_eq!(open(&KEY, &NONCE, b"headex", &s), None);
        let mut other = NONCE;
        other[0] ^= 1;
        assert_eq!(open(&KEY, &other, b"header", &s), None);
        assert_eq!(open(&KEY, &NONCE, b"header", &s[..15]), None);
    }
}
