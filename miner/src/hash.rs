//! Hashing primitives shared across the crate, built on Poseidon2, plus the
//! registry of domain tags.

use crate::babybear::{BabyBear, P};
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_x8::{LANES, permute_x8};

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
    pub const STATE_LEAF: u32 = 10;
    pub const STATE: u32 = 11;
    pub const MERKLE_LEAF: u32 = 12;
    pub const TRANSCRIPT: u32 = 13;
    pub const ZK_RNG: u32 = 14;
    pub const BLOCK_CHALLENGES: u32 = 15;
    pub const HEADER: u32 = 16;
    pub const POW_SEED: u32 = 17;
    pub const POW: u32 = 18;
    pub const POW_DATASET: u32 = 19;
    pub const BLOCK_BODY: u32 = 20;
    pub const PROOF: u32 = 21;
    pub const WALLET_KEY: u32 = 22;
    pub const WALLET_VIEW: u32 = 23;
    pub const NONCE_TAG: u32 = 24;
    pub const NONCE_MASK: u32 = 25;
    pub const GENESIS: u32 = 26;
    pub const AGG_STATE: u32 = 27;
    pub const AGG_STREAM: u32 = 28;
    /// Not consensus: the local cache of verifying keys.
    pub const KEY_CACHE: u32 = 29;
    /// A data chunk's leaf hash ([`crate::chunk`]).
    pub const BODY_CHUNK: u32 = 30;
    /// Not consensus: handshake cookies ([`crate::net`]).
    pub const NET_COOKIE: u32 = 31;
    /// Not consensus: a relayed transaction's id ([`crate::transaction`]).
    pub const TX_ID: u32 = 32;
    /// Not consensus: a wallet's seed from its recovery entropy
    /// ([`crate::mnemonic`]).
    pub const WALLET_SEED: u32 = 33;
    /// A spending condition's leaf in an output's key tree
    /// ([`crate::transaction::Condition`]).
    pub const CONDITION: u32 = 34;
    /// A hashlock's hash of its preimage.
    pub const HASHLOCK: u32 = 35;
    /// The message an anyone-can-pay signature signs.
    pub const SIG_ACP: u32 = 36;
    /// The message an any-prevout signature signs.
    pub const SIG_APO: u32 = 37;
    /// The outputs anyone-can-pay and any-prevout signatures cover.
    pub const SIGNED_OUTPUTS: u32 = 38;
    /// Not consensus: the chunk id of a relayed preimage
    /// ([`crate::net::wire::preimage_id`]).
    pub const PREIMAGE: u32 = 39;

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
        STATE_LEAF,
        STATE,
        MERKLE_LEAF,
        TRANSCRIPT,
        ZK_RNG,
        BLOCK_CHALLENGES,
        HEADER,
        POW_SEED,
        POW,
        POW_DATASET,
        BLOCK_BODY,
        PROOF,
        WALLET_KEY,
        WALLET_VIEW,
        NONCE_TAG,
        NONCE_MASK,
        GENESIS,
        AGG_STATE,
        AGG_STREAM,
        KEY_CACHE,
        BODY_CHUNK,
        NET_COOKIE,
        TX_ID,
        WALLET_SEED,
        CONDITION,
        HASHLOCK,
        SIG_ACP,
        SIG_APO,
        SIGNED_OUTPUTS,
        PREIMAGE,
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

/// Merkle internal node: `compress(left || right)`. No domain tag; trees using
/// it must give every path a fixed length known to the verifier, so a node
/// can never be presented as a leaf or vice versa.
#[inline]
pub fn merge(left: &Hash, right: &Hash) -> Hash {
    let mut input = [BabyBear::ZERO; WIDTH];
    input[..HASH_LEN].copy_from_slice(left);
    input[HASH_LEN..].copy_from_slice(right);
    compress(input)
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

/// [`sponge`] of eight inputs of equal length at once.
///
/// # Panics
/// If the inputs' lengths differ, or as [`sponge`].
pub fn sponge_x8(domain: u32, param: &[BabyBear], inputs: [&[BabyBear]; LANES]) -> [Hash; LANES] {
    assert!(param.len() <= MAX_PARAM_LEN, "sponge parameter too long");
    let len = inputs[0].len();
    assert!(inputs.iter().all(|x| x.len() == len), "lengths differ");
    assert!(len < P as usize, "sponge input too long");
    let mut init = [BabyBear::ZERO; WIDTH];
    init[RATE..RATE + param.len()].copy_from_slice(param);
    init[DOMAIN_SLOT] = BabyBear::new(domain);
    init[DOMAIN_SLOT + 1] = BabyBear::new(len as u32);
    let mut states = [init; LANES];
    if len == 0 {
        permute_x8(&mut states);
    }
    for start in (0..len).step_by(RATE) {
        let end = (start + RATE).min(len);
        for (state, input) in states.iter_mut().zip(&inputs) {
            for (s, &x) in state.iter_mut().zip(&input[start..end]) {
                *s += x;
            }
        }
        permute_x8(&mut states);
    }
    states.map(|s| core::array::from_fn(|i| s[i]))
}

/// [`merge`] of eight pairs at once.
pub fn merge_x8(pairs: [(&Hash, &Hash); LANES]) -> [Hash; LANES] {
    let inputs: [[BabyBear; WIDTH]; LANES] = pairs.map(|(l, r)| {
        let mut input = [BabyBear::ZERO; WIDTH];
        input[..HASH_LEN].copy_from_slice(l);
        input[HASH_LEN..].copy_from_slice(r);
        input
    });
    let mut states = inputs;
    permute_x8(&mut states);
    core::array::from_fn(|l| core::array::from_fn(|i| states[l][i] + inputs[l][i]))
}

/// Serializes a hash as little-endian canonical `u32`s.
pub fn to_bytes(hash: &Hash) -> [u8; HASH_BYTES] {
    let mut bytes = [0u8; HASH_BYTES];
    for (chunk, x) in bytes.chunks_exact_mut(4).zip(hash) {
        chunk.copy_from_slice(&x.as_u32().to_le_bytes());
    }
    bytes
}

/// Parses bytes written by [`to_bytes`]. Returns `None` if any element is not
/// canonical (`>= p`).
pub fn from_bytes(bytes: &[u8; HASH_BYTES]) -> Option<Hash> {
    let mut hash = [BabyBear::ZERO; HASH_LEN];
    for (x, chunk) in hash.iter_mut().zip(bytes.chunks_exact(4)) {
        let v = u32::from_le_bytes(chunk.try_into().unwrap());
        if v >= P {
            return None;
        }
        *x = BabyBear::new(v);
    }
    Some(hash)
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
    fn bytes_roundtrip() {
        let h = BabyBear::new_array([0, 1, 2, P - 1, 5, 6, 7, 8]);
        assert_eq!(from_bytes(&to_bytes(&h)), Some(h));
        let mut bad = to_bytes(&h);
        bad[..4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(from_bytes(&bad), None);
    }

    #[test]
    fn batched_hashes_match() {
        for len in [0, 1, 7, 8, 9, 30] {
            let rows: Vec<Vec<BabyBear>> = (0..LANES as u32)
                .map(|l| {
                    (0..len as u32)
                        .map(|i| BabyBear::new(l * 1000 + i))
                        .collect()
                })
                .collect();
            let refs: [&[BabyBear]; LANES] = core::array::from_fn(|l| &rows[l][..]);
            let param = [BabyBear::new(5), BabyBear::new(6)];
            let got = sponge_x8(domain::MERKLE_LEAF, &param, refs);
            for l in 0..LANES {
                assert_eq!(got[l], sponge(domain::MERKLE_LEAF, &param, &[refs[l]]));
            }
        }
        let hs: Vec<Hash> = (0..16u32)
            .map(|i| BabyBear::new_array([i; HASH_LEN]))
            .collect();
        let got = merge_x8(core::array::from_fn(|l| (&hs[2 * l], &hs[2 * l + 1])));
        for l in 0..LANES {
            assert_eq!(got[l], merge(&hs[2 * l], &hs[2 * l + 1]));
        }
    }
}
