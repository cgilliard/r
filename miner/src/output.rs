//! Transaction outputs.
//!
//! An [`Output`] is a `hash`, an `amount` and a `nonce`. The `hash` says how
//! the output can be spent (for example `H(pubkey)`), so new spend conditions
//! need no changes here. On chain an output is an [`OutputRecord`]: the 32-byte
//! [`Commitment`] `H(hash || amount || nonce)` and the nonce in the clear,
//! which wallets scan to find their own outputs.

use crate::babybear::BabyBear;
use crate::hash::{HASH_BYTES, domain, sponge, to_bytes};

pub type Amount = u64;

pub const OUTPUT_HASH_LEN: usize = 32;
/// Spend condition of an output, e.g. a public key hash.
pub type OutputHash = [u8; OUTPUT_HASH_LEN];

pub const NONCE_LEN: usize = 16;

/// Per-output value, committed to and also stored in the clear so a wallet can
/// recognise its own outputs (e.g. during recovery). Distinct nonces also keep
/// two outputs with the same hash and amount from sharing a commitment.
pub type Nonce = [u8; NONCE_LEN];

pub const COMMITMENT_LEN: usize = HASH_BYTES;

/// `hash || amount || nonce` is packed little-endian into 30-bit limbs.
/// `2^30 < p`, so each limb is a distinct field element and the packing is
/// injective.
const LIMB_BITS: usize = 30;
const AMOUNT_LEN: usize = Amount::BITS as usize / 8;
const PACKED_LEN: usize = OUTPUT_HASH_LEN + AMOUNT_LEN + NONCE_LEN;
const LIMBS: usize = (PACKED_LEN * 8).div_ceil(LIMB_BITS);

const _: () = {
    assert!(COMMITMENT_LEN == 32);
    assert!(1 << LIMB_BITS < crate::babybear::P);
};

/// `H(hash || amount || nonce)`, as stored on chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Commitment(pub [u8; COMMITMENT_LEN]);

/// What the chain stores for an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputRecord {
    pub commitment: Commitment,
    pub nonce: Nonce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Output {
    pub hash: OutputHash,
    pub amount: Amount,
    pub nonce: Nonce,
}

impl Output {
    pub fn commitment(&self) -> Commitment {
        commit(&self.hash, self.amount, &self.nonce)
    }

    pub fn record(&self) -> OutputRecord {
        OutputRecord {
            commitment: self.commitment(),
            nonce: self.nonce,
        }
    }

    /// Returns `true` if this output opens `commitment`.
    pub fn opens(&self, commitment: &Commitment) -> bool {
        self.commitment() == *commitment
    }
}

/// Computes `H(hash || amount || nonce)`: 15 limbs, so a two-permutation sponge.
pub fn commit(hash: &OutputHash, amount: Amount, nonce: &Nonce) -> Commitment {
    let mut packed = [0u8; PACKED_LEN];
    let (h, rest) = packed.split_at_mut(OUTPUT_HASH_LEN);
    let (a, n) = rest.split_at_mut(AMOUNT_LEN);
    h.copy_from_slice(hash);
    a.copy_from_slice(&amount.to_le_bytes());
    n.copy_from_slice(nonce);
    Commitment(to_bytes(&sponge(domain::OUTPUT, &[], &[&pack(&packed)])))
}

/// Splits `bytes`, read as a little-endian integer, into `LIMB_BITS`-bit limbs.
fn pack(bytes: &[u8; PACKED_LEN]) -> [BabyBear; LIMBS] {
    let mut limbs = [BabyBear::ZERO; LIMBS];
    let (mut acc, mut bits, mut i) = (0u64, 0, 0);
    for &b in bytes {
        acc |= (b as u64) << bits;
        bits += 8;
        if bits >= LIMB_BITS {
            limbs[i] = BabyBear::new((acc & ((1 << LIMB_BITS) - 1)) as u32);
            acc >>= LIMB_BITS;
            bits -= LIMB_BITS;
            i += 1;
        }
    }
    if bits > 0 {
        limbs[i] = BabyBear::new(acc as u32);
    }
    limbs
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONCE: Nonce = *b"0123456789abcdef";

    fn hash(n: u8) -> OutputHash {
        core::array::from_fn(|i| n.wrapping_mul(31).wrapping_add(i as u8))
    }

    #[test]
    fn record_holds_commitment_and_nonce() {
        let out = Output {
            hash: hash(1),
            amount: 42,
            nonce: NONCE,
        };
        let record = out.record();
        assert_eq!(record.commitment, commit(&hash(1), 42, &NONCE));
        assert_eq!(record.nonce, NONCE);
        assert!(out.opens(&record.commitment));
    }

    #[test]
    fn amounts_are_distinct() {
        let h = hash(1);
        let amounts = [0, 1, 0xffff, 0x1_0000, 0xffff_ffff, 1 << 48, u64::MAX];
        for (i, &a) in amounts.iter().enumerate() {
            for &b in &amounts[i + 1..] {
                assert_ne!(commit(&h, a, &NONCE), commit(&h, b, &NONCE));
            }
        }
    }

    #[test]
    fn rejects_wrong_hash_amount_or_nonce() {
        let c = commit(&hash(1), 7, &NONCE);
        let out = |h, amount, nonce| Output {
            hash: h,
            amount,
            nonce,
        };
        assert!(out(hash(1), 7, NONCE).opens(&c));
        assert!(!out(hash(2), 7, NONCE).opens(&c));
        assert!(!out(hash(1), 8, NONCE).opens(&c));
        assert!(!out(hash(1), 7, [0; NONCE_LEN]).opens(&c));
    }

    #[test]
    fn every_bit_is_committed() {
        let (h, amount) = (hash(5), 0x0123_4567_89ab_cdef);
        let c = commit(&h, amount, &NONCE);
        for bit in 0..OUTPUT_HASH_LEN * 8 {
            let mut h2 = h;
            h2[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(commit(&h2, amount, &NONCE), c);
        }
        for bit in 0..Amount::BITS {
            assert_ne!(commit(&h, amount ^ (1 << bit), &NONCE), c);
        }
        for bit in 0..NONCE_LEN * 8 {
            let mut n2 = NONCE;
            n2[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(commit(&h, amount, &n2), c);
        }
    }

    #[test]
    fn pack_splits_into_30_bit_limbs() {
        let limbs = pack(&[0xff; PACKED_LEN]);
        assert_eq!(LIMBS, 15);
        for x in &limbs[..LIMBS - 1] {
            assert_eq!(x.as_u32(), (1 << 30) - 1);
        }
        // 448 bits = 14 * 30 + 28.
        assert_eq!(limbs[LIMBS - 1].as_u32(), (1 << 28) - 1);

        let mut bytes = [0u8; PACKED_LEN];
        bytes[3] = 0x40; // bit 30: the first bit of the second limb
        let limbs = pack(&bytes);
        assert_eq!((limbs[0].as_u32(), limbs[1].as_u32()), (0, 1));
    }

    #[test]
    fn commitment_bytes_are_canonical_le() {
        let c = commit(&hash(3), 0, &[0; NONCE_LEN]);
        for chunk in c.0.chunks_exact(4) {
            let x = u32::from_le_bytes(chunk.try_into().unwrap());
            assert!(x < crate::babybear::P);
        }
    }
}
