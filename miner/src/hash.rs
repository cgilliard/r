//! Hashing primitives shared across the crate, built on Poseidon2, plus the
//! registry of domain tags.

use crate::babybear::{BabyBear, P};
use crate::poseidon2::{self, WIDTH};

/// Field elements per hash (~248 bits).
pub const HASH_LEN: usize = 8;
pub type Hash = [BabyBear; HASH_LEN];
/// Bytes in a serialized hash: each element as a little-endian canonical `u32`.
pub const HASH_BYTES: usize = HASH_LEN * 4;

/// Sponge rate. The capacity holds a parameter, the domain tag and the input length.
const RATE: usize = HASH_LEN;
/// Longest parameter the sponge capacity can hold.
pub const MAX_PARAM_LEN: usize = 5;
/// Where the domain tag goes, both in [`compress`] inputs and in the sponge capacity.
pub const DOMAIN_SLOT: usize = WIDTH - 3;

const _: () = assert!(RATE + MAX_PARAM_LEN == DOMAIN_SLOT);

/// Every domain tag in the crate. Each kind of hashed data gets its own tag,
/// so hashes of different kinds can never collide. Tags must be unique.
pub mod domain {
    pub const WOTS_CHAIN: u32 = 1;
    pub const WOTS_PRF: u32 = 2;
    pub const WOTS_PUBLIC_KEY: u32 = 3;
    pub const WOTS_MESSAGE: u32 = 4;
    pub const WOTS_RANDOMIZER: u32 = 5;
    pub const WOTS_PARAMETER: u32 = 6;
    pub const PUBLIC_KEY_HASH: u32 = 7;
    pub const OUTPUT: u32 = 8;
    pub const TX_BODY: u32 = 9;

    #[cfg(test)]
    pub const ALL: &[u32] = &[
        WOTS_CHAIN,
        WOTS_PRF,
        WOTS_PUBLIC_KEY,
        WOTS_MESSAGE,
        WOTS_RANDOMIZER,
        WOTS_PARAMETER,
        PUBLIC_KEY_HASH,
        OUTPUT,
        TX_BODY,
    ];
}

/// Truncated permutation with feed-forward: `(P(x) + x)[..HASH_LEN]`.
/// Callers put a domain tag at `input[DOMAIN_SLOT]`.
#[inline]
pub fn compress(input: [BabyBear; WIDTH]) -> Hash {
    let mut state = input;
    poseidon2::permute(&mut state);
    core::array::from_fn(|i| state[i] + input[i])
}

/// Sponge over the concatenation of `inputs`. The capacity is initialised with
/// `param` (zero-padded), the domain and the total input length, so zero
/// padding is injective. Each domain must always use the same `param` length.
pub fn sponge(domain: u32, param: &[BabyBear], inputs: &[&[BabyBear]]) -> Hash {
    assert!(param.len() <= MAX_PARAM_LEN, "sponge parameter too long");
    let len: usize = inputs.iter().map(|x| x.len()).sum();
    assert!(len < P as usize, "sponge input too long");
    let mut state = [BabyBear::ZERO; WIDTH];
    state[RATE..RATE + param.len()].copy_from_slice(param);
    state[DOMAIN_SLOT] = BabyBear::new(domain);
    state[DOMAIN_SLOT + 1] = BabyBear::new(len as u32);

    let mut pos = 0;
    for &x in inputs.iter().flat_map(|s| s.iter()) {
        state[pos] += x;
        pos += 1;
        if pos == RATE {
            poseidon2::permute(&mut state);
            pos = 0;
        }
    }
    if pos != 0 || len == 0 {
        poseidon2::permute(&mut state);
    }
    core::array::from_fn(|i| state[i])
}

/// Serializes a hash as little-endian canonical `u32`s.
pub fn to_bytes(hash: &Hash) -> [u8; HASH_BYTES] {
    let mut bytes = [0u8; HASH_BYTES];
    for (chunk, x) in bytes.chunks_exact_mut(4).zip(hash) {
        chunk.copy_from_slice(&x.as_u32().to_le_bytes());
    }
    bytes
}

/// Hashes a byte string: its length, then 3 bytes per field element.
pub fn hash_bytes(domain: u32, bytes: &[u8]) -> Hash {
    assert!(bytes.len() < P as usize, "byte string too long");
    let mut elems = Vec::with_capacity(1 + bytes.len().div_ceil(3));
    elems.push(BabyBear::new(bytes.len() as u32));
    elems.extend(bytes.chunks(3).map(|c| {
        let mut word = [0u8; 4];
        word[..c.len()].copy_from_slice(c);
        BabyBear::new(u32::from_le_bytes(word))
    }));
    sponge(domain, &[], &[&elems])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_tags_are_unique() {
        let mut tags = domain::ALL.to_vec();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), domain::ALL.len());
    }

    #[test]
    fn sponge_separates_lengths_and_domains() {
        let x = BabyBear::new_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut padded = x.to_vec();
        padded.push(BabyBear::ZERO);
        assert_ne!(sponge(1, &[], &[&x]), sponge(1, &[], &[&padded]));
        assert_ne!(sponge(1, &[], &[&x]), sponge(2, &[], &[&x]));
        assert_ne!(sponge(1, &[], &[]), sponge(1, &[], &[&[BabyBear::ZERO]]));
        // How the input is split across slices doesn't matter.
        assert_eq!(sponge(1, &[], &[&x[..3], &x[3..]]), sponge(1, &[], &[&x]));
    }

    #[test]
    fn hash_bytes_separates_trailing_zeros() {
        assert_ne!(hash_bytes(1, &[]), hash_bytes(1, &[0]));
        assert_ne!(hash_bytes(1, &[1]), hash_bytes(1, &[1, 0]));
        assert_ne!(hash_bytes(1, &[1, 0]), hash_bytes(1, &[1, 0, 0]));
        assert_ne!(hash_bytes(1, &[1, 2, 3]), hash_bytes(2, &[1, 2, 3]));
    }
}
