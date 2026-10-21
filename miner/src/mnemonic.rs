//! BIP39 recovery words: a wallet's 32 bytes of entropy as 24 words from the
//! standard English list, the last word carrying an 8-bit SHA-256 checksum,
//! so any BIP39 tool reads them and a mistyped word is caught. The wallet's
//! seed is derived from the entropy itself ([`seed_of`]), with no
//! passphrase.

use crate::babybear::BabyBear;
use crate::hash::{self, Hash, domain};

/// Bytes of entropy.
pub const ENTROPY_LEN: usize = 32;
/// Words for [`ENTROPY_LEN`] bytes: 256 bits and 8 of checksum, 11 a word.
pub const WORDS: usize = 24;

/// The standard English list (2048 words, SHA-256 `2f5eed53…dbda`).
const LIST: &str = include_str!("bip39_english.txt");

fn list() -> &'static [&'static str] {
    static WORDS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    WORDS.get_or_init(|| LIST.lines().collect())
}

/// The recovery words for `entropy`.
pub fn to_words(entropy: &[u8; ENTROPY_LEN]) -> Vec<&'static str> {
    let mut bits = entropy.to_vec();
    bits.push(sha256(entropy)[0]);
    (0..WORDS)
        .map(|w| {
            let index = (0..11).fold(0usize, |acc, b| {
                let bit = w * 11 + b;
                acc << 1 | (bits[bit / 8] >> (7 - bit % 8) & 1) as usize
            });
            list()[index]
        })
        .collect()
}

/// The entropy behind recovery words (separated by whitespace, in any case).
pub fn from_words(words: &str) -> Result<[u8; ENTROPY_LEN], String> {
    let words: Vec<String> = words.split_whitespace().map(str::to_lowercase).collect();
    if words.len() != WORDS {
        return Err(format!("expected {WORDS} words, got {}", words.len()));
    }
    let mut bits = [0u8; ENTROPY_LEN + 1];
    for (w, word) in words.iter().enumerate() {
        let index = list()
            .binary_search(&word.as_str())
            .map_err(|_| format!("\"{word}\" isn't a recovery word"))?;
        for b in 0..11 {
            if index >> (10 - b) & 1 == 1 {
                let bit = w * 11 + b;
                bits[bit / 8] |= 1 << (7 - bit % 8);
            }
        }
    }
    let entropy: [u8; ENTROPY_LEN] = bits[..ENTROPY_LEN].try_into().expect("length");
    if sha256(&entropy)[0] != bits[ENTROPY_LEN] {
        return Err("the words don't check out (one is wrong, or out of order)".into());
    }
    Ok(entropy)
}

/// The wallet seed from its entropy: a sponge over the entropy as 16-bit
/// limbs.
pub fn seed_of(entropy: &[u8; ENTROPY_LEN]) -> Hash {
    let limbs: Vec<BabyBear> = entropy
        .chunks_exact(2)
        .map(|c| BabyBear::new(u16::from_le_bytes([c[0], c[1]]) as u32))
        .collect();
    hash::sponge(domain::WALLET_SEED, &[], &[&limbs])
}

/// SHA-256 (FIPS 180-4), for the checksum.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend((data.len() as u64 * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().expect("4 bytes"));
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(v);
        }
    }
    let mut out = [0u8; 32];
    for (chunk, x) in out.chunks_exact_mut(4).zip(h) {
        chunk.copy_from_slice(&x.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha256_matches_known_digests() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // Two blocks.
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            hex(&sha256(LIST.as_bytes())),
            "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda"
        );
    }

    #[test]
    fn words_match_the_bip39_test_vectors_and_round_trip() {
        let abandon = format!("{} art", ["abandon"; 23].join(" "));
        let legal = "legal winner thank year wave sausage worth useful legal winner thank year \
                     wave sausage worth useful legal winner thank year wave sausage worth title";
        let zoo = format!("{} vote", ["zoo"; 23].join(" "));
        for (byte, words) in [(0u8, abandon.as_str()), (0x7f, legal), (0xff, zoo.as_str())] {
            let entropy = [byte; ENTROPY_LEN];
            assert_eq!(to_words(&entropy).join(" "), words);
            assert_eq!(from_words(words), Ok(entropy));
        }
        let e: [u8; 32] = core::array::from_fn(|i| (i * 37 + 11) as u8);
        let words = to_words(&e).join(" ");
        assert_eq!(from_words(&format!("  {}\n", words.to_uppercase())), Ok(e));
        // A swapped word fails the checksum (almost always; this one does).
        let mut w: Vec<&str> = words.split(' ').collect();
        w.swap(0, 1);
        assert!(from_words(&w.join(" ")).is_err());
        assert!(from_words(&abandon.replace("art", "zzz")).is_err());
        assert!(from_words("abandon").is_err());
        assert_ne!(seed_of(&e), seed_of(&[0; 32]));
    }
}
