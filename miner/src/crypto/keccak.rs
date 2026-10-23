//! Keccak-f[1600] and the SHA-3 functions built on it (FIPS 202): SHA3-256,
//! SHA3-512, and the extendable-output SHAKE128 and SHAKE256, which ML-KEM
//! and ML-DSA use.

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets, by lane `x + 5y`.
const ROTATIONS: [u32; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Keccak-f[1600] on 25 lanes (lane `x + 5y`).
pub fn keccak_f(a: &mut [u64; 25]) {
    for rc in ROUND_CONSTANTS {
        // θ
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // ρ and π: lane (x, y) moves to (y, 2x + 3y).
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(ROTATIONS[x + 5 * y]);
            }
        }
        // χ
        for y in 0..5 {
            for x in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        // ι
        a[0] ^= rc;
    }
}

/// A Keccak sponge with `rate` bytes and domain-separation byte `pad`
/// (`0x06` for SHA-3, `0x1f` for SHAKE): absorb, then squeeze.
#[derive(Clone)]
pub struct Sponge {
    state: [u64; 25],
    rate: usize,
    pad: u8,
    /// Bytes absorbed into (or squeezed from) the current block.
    pos: usize,
    squeezing: bool,
}

impl Sponge {
    fn new(rate: usize, pad: u8) -> Self {
        Self {
            state: [0; 25],
            rate,
            pad,
            pos: 0,
            squeezing: false,
        }
    }

    /// SHAKE128 (rate 168 bytes).
    pub fn shake128() -> Self {
        Self::new(168, 0x1f)
    }

    /// SHAKE256 (rate 136 bytes).
    pub fn shake256() -> Self {
        Self::new(136, 0x1f)
    }

    fn xor_byte(&mut self, i: usize, b: u8) {
        self.state[i / 8] ^= (b as u64) << (8 * (i % 8));
    }

    /// Absorbs `data`.
    ///
    /// # Panics
    /// After squeezing has started.
    pub fn absorb(&mut self, data: &[u8]) {
        assert!(!self.squeezing, "absorbing after squeezing");
        for &b in data {
            self.xor_byte(self.pos, b);
            self.pos += 1;
            if self.pos == self.rate {
                keccak_f(&mut self.state);
                self.pos = 0;
            }
        }
    }

    fn finish(&mut self) {
        self.xor_byte(self.pos, self.pad);
        self.xor_byte(self.rate - 1, 0x80);
        keccak_f(&mut self.state);
        self.pos = 0;
        self.squeezing = true;
    }

    /// Squeezes `out.len()` more bytes.
    pub fn squeeze(&mut self, out: &mut [u8]) {
        if !self.squeezing {
            self.finish();
        }
        for o in out {
            if self.pos == self.rate {
                keccak_f(&mut self.state);
                self.pos = 0;
            }
            *o = (self.state[self.pos / 8] >> (8 * (self.pos % 8))) as u8;
            self.pos += 1;
        }
    }
}

fn hash<const N: usize>(rate: usize, parts: &[&[u8]]) -> [u8; N] {
    let mut s = Sponge::new(rate, 0x06);
    for p in parts {
        s.absorb(p);
    }
    let mut out = [0u8; N];
    s.squeeze(&mut out);
    out
}

/// SHA3-256 of the concatenation of `parts`.
pub fn sha3_256(parts: &[&[u8]]) -> [u8; 32] {
    hash(136, parts)
}

/// SHA3-512 of the concatenation of `parts`.
pub fn sha3_512(parts: &[&[u8]]) -> [u8; 64] {
    hash(72, parts)
}

/// SHAKE256 of the concatenation of `parts`, `out.len()` bytes of it.
pub fn shake256(parts: &[&[u8]], out: &mut [u8]) {
    let mut s = Sponge::shake256();
    for p in parts {
        s.absorb(p);
    }
    s.squeeze(out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Digests from Python's hashlib.
    #[test]
    fn matches_known_digests() {
        assert_eq!(
            hex(&sha3_256(&[])),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
        assert_eq!(
            hex(&sha3_256(&[b"abc"])),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
        assert_eq!(
            hex(&sha3_512(&[b"abc"])),
            "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e\
             10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0"
        );
        let mut out = [0u8; 32];
        Sponge::shake128().squeeze(&mut out);
        assert_eq!(
            hex(&out),
            "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
        );
        shake256(&[], &mut out);
        assert_eq!(
            hex(&out),
            "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
        );
    }

    /// Absorbing in pieces and squeezing in pieces give the same bytes as
    /// all at once, across block boundaries.
    #[test]
    fn pieces_match_whole() {
        let msg: Vec<u8> = (0..500u32).map(|i| (i * 13 + 5) as u8).collect();
        let mut whole = vec![0u8; 400];
        shake256(&[&msg], &mut whole);
        let mut s = Sponge::shake256();
        for chunk in msg.chunks(37) {
            s.absorb(chunk);
        }
        let mut pieces = vec![0u8; 400];
        for chunk in pieces.chunks_mut(51) {
            s.squeeze(chunk);
        }
        assert_eq!(whole, pieces);
        // hashlib's, for the long message.
        assert_eq!(
            hex(&whole[368..]),
            "f8153ed6615a404d3d0684d6948406f8bf5558f88a11a13127a2e9ee3e6666e7"
        );
        let mut s128 = Sponge::shake128();
        s128.absorb(&msg);
        let mut out = [0u8; 200];
        s128.squeeze(&mut out);
        assert_eq!(
            hex(&out[168..]),
            "855162ef05e5036192ae31e509d2a59a5a331c5c695283b69fb35a102e341bef"
        );
        assert_eq!(
            hex(&sha3_512(&[&msg])),
            "abbe5e844a3bf900d787fd9eaf84b61d51837db2383fee9ee077919b5081099d\
             4f44b7e32cb68f81144254347403b178e3850ef0f6376508e0ecd35b03f98d2f"
        );
        assert_eq!(sha3_256(&[&msg[..200], &msg[200..]]), sha3_256(&[&msg]));
    }
}
