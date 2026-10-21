//! Transaction outputs.
//!
//! An [`Output`] is a `hash`, an `amount` and a `nonce`. The `hash` says how
//! the output can be spent (for example `H(pubkey)`), so new spend conditions
//! need no changes here. On chain an output is an [`OutputRecord`]: the 32-byte
//! [`Commitment`] `H(hash || amount || nonce)` and the nonce in the clear,
//! which wallets scan to find their own outputs.

use crate::babybear::BabyBear;
use crate::hash::{self, HASH_BYTES, HASH_LEN, Hash, domain, sponge};

pub type Amount = u64;

/// Units per coin.
pub const COIN: Amount = 100_000_000;

pub const OUTPUT_HASH_LEN: usize = HASH_BYTES;
/// Spend condition of an output, e.g. a public key hash: 8 field elements as
/// little-endian `u32`s (see [`hash::to_bytes`]), each below `p`.
pub type OutputHash = [u8; OUTPUT_HASH_LEN];

pub const NONCE_LEN: usize = 32;

/// Per-output value, committed to and also stored in the clear so a wallet can
/// recognise its own outputs (e.g. during recovery). Distinct nonces also keep
/// two outputs with the same hash and amount from sharing a commitment.
pub type Nonce = [u8; NONCE_LEN];

/// Bytes per nonce limb: 24 bits, always below `p`, so the encoding is
/// injective without range checks.
pub const NONCE_LIMB_BYTES: usize = 3;

pub const COMMITMENT_LEN: usize = HASH_BYTES;

/// Amounts are hashed as 16-bit limbs, little-endian, which a circuit can
/// range-check and sum without overflowing the field.
pub const LIMB_BITS: u32 = 16;
pub const AMOUNT_LIMBS: usize = (Amount::BITS / LIMB_BITS) as usize;
/// Nonces are hashed as 3-byte limbs (the last one shorter).
pub const NONCE_LIMBS: usize = NONCE_LEN.div_ceil(NONCE_LIMB_BYTES);

/// `units` as a decimal number of coins ("1.5", "1000").
pub fn format_coins(units: u128) -> String {
    let (whole, frac) = (units / COIN as u128, units % COIN as u128);
    if frac == 0 {
        format!("{whole}")
    } else {
        format!("{whole}.{}", format!("{frac:08}").trim_end_matches('0'))
    }
}

/// Parses a decimal number of coins (at most 8 decimals) into units.
pub fn parse_coins(s: &str) -> Option<Amount> {
    let (whole, frac) = s.split_once('.').unwrap_or((s, ""));
    if (whole.is_empty() && frac.is_empty())
        || frac.len() > 8
        || !whole
            .bytes()
            .chain(frac.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let whole: Amount = if whole.is_empty() {
        0
    } else {
        whole.parse().ok()?
    };
    let frac: Amount = if frac.is_empty() {
        0
    } else {
        format!("{frac:0<8}").parse().ok()?
    };
    whole.checked_mul(COIN)?.checked_add(frac)
}

/// `H(hash || amount || nonce)`, as stored on chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Commitment(pub [u8; COMMITMENT_LEN]);

impl Commitment {
    /// The commitment as field elements, or `None` if its bytes are not a
    /// canonical encoding (no commitment computed by [`commit`] is).
    pub fn to_field(&self) -> Option<Hash> {
        hash::from_bytes(&self.0)
    }
}

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
    /// `None` if `hash` is not a canonical encoding of 8 field elements.
    pub fn commitment(&self) -> Option<Commitment> {
        commit(&self.hash, self.amount, &self.nonce)
    }

    pub fn record(&self) -> Option<OutputRecord> {
        Some(OutputRecord {
            commitment: self.commitment()?,
            nonce: self.nonce,
        })
    }

    /// Returns `true` if this output opens `commitment`.
    pub fn opens(&self, commitment: &Commitment) -> bool {
        self.commitment() == Some(*commitment)
    }
}

/// `H(hash || amount || nonce)`: a sponge over the hash's 8 elements, the
/// amount's 4 limbs and the nonce's 11 limbs, 23 elements in all (three
/// permutations). `None` if
/// `hash` is not a canonical encoding.
pub fn commit(hash: &OutputHash, amount: Amount, nonce: &Nonce) -> Option<Commitment> {
    let hash = hash::from_bytes(hash)?;
    Some(commit_field(&hash, amount, nonce))
}

/// [`commit`] with the hash already as field elements.
pub fn commit_field(hash: &Hash, amount: Amount, nonce: &Nonce) -> Commitment {
    let digest = sponge(
        domain::OUTPUT,
        &[],
        &[hash, &amount_limbs(amount), &nonce_limbs(nonce)],
    );
    Commitment(hash::to_bytes(&digest))
}

/// `amount` as 16-bit limbs, least significant first.
pub fn amount_limbs(amount: Amount) -> [BabyBear; AMOUNT_LIMBS] {
    core::array::from_fn(|i| BabyBear::new((amount >> (LIMB_BITS as usize * i)) as u16 as u32))
}

/// Limbs of a net value (see [`net_limbs`]).
pub const NET_LIMBS: usize = 6;
/// The offset making net values non-negative: `2^95`.
pub const NET_OFFSET: u128 = 1 << 95;

/// A net value `sum(outputs) - sum(inputs)`, which may be negative, as the
/// 16-bit limbs of `NET_OFFSET + net`, least significant first.
///
/// # Panics
/// If `|net| >= 2^95`.
pub fn net_limbs(net: i128) -> [BabyBear; NET_LIMBS] {
    assert!(net.unsigned_abs() < NET_OFFSET, "net value out of range");
    let v = (NET_OFFSET as i128 + net) as u128;
    core::array::from_fn(|i| BabyBear::new((v >> (LIMB_BITS as usize * i)) as u16 as u32))
}

/// Inverse of [`net_limbs`] for canonical limbs.
pub fn net_value(limbs: &[BabyBear]) -> i128 {
    let v = limbs.iter().enumerate().fold(0u128, |acc, (i, x)| {
        acc | (x.as_u32() as u128) << (LIMB_BITS as usize * i)
    });
    v as i128 - NET_OFFSET as i128
}

/// `nonce` as limbs: 3-byte groups, little-endian.
pub fn nonce_limbs(nonce: &Nonce) -> [BabyBear; NONCE_LIMBS] {
    core::array::from_fn(|i| {
        let group = &nonce[NONCE_LIMB_BYTES * i..(NONCE_LIMB_BYTES * (i + 1)).min(NONCE_LEN)];
        BabyBear::new(
            group
                .iter()
                .rev()
                .fold(0u32, |acc, &b| (acc << 8) | b as u32),
        )
    })
}

const _: () = assert!(HASH_LEN * 4 == OUTPUT_HASH_LEN);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;

    const NONCE: Nonce = *b"0123456789abcdef0123456789ABCDEF";

    #[test]
    fn coins_format_and_parse() {
        for (text, units) in [
            ("0", 0),
            ("1", COIN),
            ("1.5", COIN + COIN / 2),
            ("0.00000001", 1),
            ("1000", 1000 * COIN),
        ] {
            assert_eq!(parse_coins(text), Some(units), "{text}");
            assert_eq!(format_coins(units as u128), text);
        }
        assert_eq!(parse_coins(".5"), Some(COIN / 2));
        assert_eq!(parse_coins("2."), Some(2 * COIN));
        for bad in [
            "",
            ".",
            "1.000000001",
            "-1",
            "1e3",
            "x",
            "1.2.3",
            "999999999999",
        ] {
            assert_eq!(parse_coins(bad), None, "{bad}");
        }
    }

    /// A canonical hash.
    fn hash(n: u32) -> OutputHash {
        hash::to_bytes(&BabyBear::new_array(core::array::from_fn(|i| {
            n * 1000 + i as u32
        })))
    }

    fn out(h: OutputHash, amount: Amount, nonce: Nonce) -> Output {
        Output {
            hash: h,
            amount,
            nonce,
        }
    }

    #[test]
    fn record_holds_commitment_and_nonce() {
        let o = out(hash(1), 42, NONCE);
        let record = o.record().unwrap();
        assert_eq!(Some(record.commitment), commit(&hash(1), 42, &NONCE));
        assert_eq!(record.nonce, NONCE);
        assert!(o.opens(&record.commitment));
        assert!(record.commitment.to_field().is_some());
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
        let c = commit(&hash(1), 7, &NONCE).unwrap();
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
            // Either no longer canonical, or a different commitment.
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
    fn rejects_non_canonical_hashes() {
        let mut h = hash(1);
        h[..4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(commit(&h, 1, &NONCE), None);
        assert_eq!(out(h, 1, NONCE).record(), None);
        assert_eq!(Commitment(h).to_field(), None);
    }

    #[test]
    fn limbs_roundtrip() {
        let amount = 0xfedc_ba98_7654_3210u64;
        let limbs = amount_limbs(amount);
        let back = limbs
            .iter()
            .rev()
            .fold(0u64, |acc, l| (acc << 16) | l.as_u32() as u64);
        assert_eq!(back, amount);
        assert!(limbs.iter().all(|l| l.as_u32() < 1 << 16));
        let n = nonce_limbs(&NONCE);
        assert_eq!(n[0].as_u32(), u32::from_le_bytes(*b"012\0"));
        assert_eq!(n[NONCE_LIMBS - 1].as_u32(), u32::from_le_bytes(*b"EF\0\0"));
    }

    #[test]
    fn commitment_matches_field_form() {
        let h = hash(3);
        let field = hash::from_bytes(&h).unwrap();
        assert_eq!(commit(&h, 9, &NONCE), Some(commit_field(&field, 9, &NONCE)));
    }
}
