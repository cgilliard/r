//! Block headers, proof-of-work checks and difficulty.
//!
//! A header commits to its parent, the state after its block, and the block
//! body ([`crate::block::Block::body_hash`], which includes the proof). It
//! has no difficulty field: the target follows from the parent's height and
//! timestamp by ASERT (absolutely scheduled exponentially rising targets),
//! anchored at genesis. The target doubles for every half-life the chain
//! runs behind schedule and halves for every half-life ahead.
//!
//! Targets are handled as fixed-point base-2 logarithms with [`FRAC_BITS`]
//! fractional bits. To keep the rule cheap to prove, the half-life is a
//! power of two (so the exponent is a shift), and `2^x` is interpolated
//! linearly within each octave: `target = (2^16 + frac) * 2^(int - 16)`.
//! That is monotone and within 6.1% of `2^x`, which ASERT tolerates (it
//! only shifts the schedule slightly). Every node, and the chain proof
//! ([`crate::pow_circuit`]), computes exactly the same target.

use std::fmt;

use crate::hash::{self, Hash, domain};
use crate::output::{Amount, COIN};
use crate::pow::{self, Dataset, VALUE_BITS, u64_limbs};
use crate::state::{self, State};

/// Fractional bits of a log-target.
pub const FRAC_BITS: u32 = 16;
/// The smallest log-target: a target of `2^16`, `2^92` expected hashes.
pub const MIN_LOG_TARGET: i64 = 16 << FRAC_BITS;
/// The largest log-target: [`pow::MAX_TARGET`], which every hash meets.
pub const MAX_LOG_TARGET: i64 = (VALUE_BITS as i64) << FRAC_BITS;

/// Chain-wide difficulty and timing rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Consensus {
    /// The genesis header's timestamp, the schedule's anchor.
    pub genesis_time: u64,
    /// The log-target of the block after genesis.
    pub genesis_log_target: i64,
    /// Target seconds between blocks.
    pub spacing: u64,
    /// The half-life, `2^log_half_life` seconds: being that far behind
    /// (ahead of) schedule doubles (halves) the target. At least 2^16.
    pub log_half_life: u32,
    /// How far past a node's clock a timestamp may be. Not provable, so a
    /// node applies it only to blocks it receives.
    pub max_future: u64,
    /// Emission: blocks `1 ..= first_era` each create `first_subsidy`
    /// (what a block's transactions must create in total,
    /// `sum(outputs) - sum(inputs)`), the next `second_era` blocks
    /// `second_subsidy`. After that the chain ends: no block above
    /// [`Consensus::final_height`] is valid.
    pub first_era: u64,
    pub first_subsidy: Amount,
    pub second_era: u64,
    pub second_subsidy: Amount,
}

impl Consensus {
    pub const DEFAULT: Self = Self {
        genesis_time: crate::genesis::TIMESTAMP,
        // 2^84.72: about 2^23.3 expected hashes, under half a minute on one
        // desktop. To be raised before launch.
        genesis_log_target: (84 << FRAC_BITS) + 42406,
        spacing: 600,
        // About 36 hours.
        log_half_life: 17,
        max_future: 2 * 3600,
        // Seven years of 1000 coins, then a thousand years of 7, at 144
        // blocks a day.
        first_era: 7 * 144 * 365,
        first_subsidy: 1000 * COIN,
        second_era: 1000 * 144 * 365,
        second_subsidy: 7 * COIN,
    };

    /// The last valid height.
    pub fn final_height(&self) -> u64 {
        self.first_era + self.second_era
    }

    /// What block `height` must create; `None` for genesis and past the
    /// final height.
    pub fn subsidy(&self, height: u64) -> Option<Amount> {
        match height {
            0 => None,
            h if h <= self.first_era => Some(self.first_subsidy),
            h if h <= self.final_height() => Some(self.second_subsidy),
            _ => None,
        }
    }

    /// Every coin the chain creates.
    pub fn total_supply(&self) -> u128 {
        self.first_era as u128 * self.first_subsidy as u128
            + self.second_era as u128 * self.second_subsidy as u128
    }

    /// The log-target of the block after `parent`.
    pub fn log_target(&self, parent: &Header) -> i64 {
        let behind = parent.timestamp as i128
            - self.genesis_time as i128
            - self.spacing as i128 * parent.height as i128;
        // floor(behind * 2^16 / 2^log_half_life)
        let exponent = behind >> (self.log_half_life - FRAC_BITS);
        (self.genesis_log_target as i128 + exponent)
            .clamp(MIN_LOG_TARGET as i128, MAX_LOG_TARGET as i128) as i64
    }

    /// The target of the block after `parent`.
    pub fn target(&self, parent: &Header) -> u128 {
        target_of(self.log_target(parent))
    }
}

/// About `2^(log_target / 2^16)`: `(2^16 + frac) * 2^(int - 16)`, with the
/// log-target clamped to `[MIN_LOG_TARGET, MAX_LOG_TARGET]`.
pub fn target_of(log_target: i64) -> u128 {
    let l = log_target.clamp(MIN_LOG_TARGET, MAX_LOG_TARGET);
    let (int, frac) = ((l >> FRAC_BITS) as u32, (l & 0xffff) as u128);
    ((1 << FRAC_BITS) + frac) << (int - FRAC_BITS)
}

/// Bytes in a serialized header ([`Header::to_bytes`]).
pub const HEADER_BYTES: usize = 4 * 8 + 3 * hash::HASH_BYTES;

/// Bits per limb of a work value in a circuit or statement.
pub const WORK_LIMB_BITS: u32 = 12;
/// Limbs of a work value: 144 bits.
pub const WORK_LIMBS: usize = 12;

/// The work a block mined at `log_target` adds to its chain: its expected
/// hashes, `MAX_TARGET / target`, in units of `2^-17` hashes. With the
/// target `(2^16 + frac) * 2^(int - 16)` ([`target_of`]) that is
/// `floor(2^33 / (2^16 + frac)) * 2^(108 - int)`: exact at octave
/// boundaries, and below the exact value by less than `2^-16` in between
/// (the quotient's rounding). Never increasing in the target.
pub fn work(log_target: i64) -> u128 {
    let l = log_target.clamp(MIN_LOG_TARGET, MAX_LOG_TARGET);
    let (int, frac) = ((l >> FRAC_BITS) as u32, (l & 0xffff) as u128);
    ((1u128 << (2 * FRAC_BITS + 1)) / ((1 << FRAC_BITS) + frac)) << (VALUE_BITS - int)
}

/// `w` as [`WORK_LIMBS`] limbs of [`WORK_LIMB_BITS`] bits.
pub fn work_limbs(w: u128) -> [crate::babybear::BabyBear; WORK_LIMBS] {
    let mask = (1 << WORK_LIMB_BITS) - 1;
    core::array::from_fn(|k| {
        let shift = WORK_LIMB_BITS as usize * k;
        crate::babybear::BabyBear::new(if shift < 128 {
            (w >> shift) as u32 & mask
        } else {
            0
        })
    })
}

/// Inverse of [`work_limbs`]; `None` if a limb is out of range or the value
/// exceeds `u128`.
pub fn work_value(limbs: &[crate::babybear::BabyBear]) -> Option<u128> {
    let mut w: u128 = 0;
    for (k, x) in limbs.iter().enumerate() {
        let v = x.as_u32() as u128;
        if v >> WORK_LIMB_BITS != 0 {
            return None;
        }
        if v == 0 {
            continue;
        }
        let shift = WORK_LIMB_BITS as usize * k;
        if shift >= 128 || v.leading_zeros() < shift as u32 {
            return None;
        }
        w |= v << shift;
    }
    Some(w)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub height: u64,
    /// The parent's [`Header::hash`]; zero for genesis.
    pub prev: Hash,
    /// Seconds since the Unix epoch.
    pub timestamp: u64,
    /// The state after this block.
    pub state: State,
    /// The block's [`crate::block::Block::body_hash`].
    pub body: Hash,
    pub nonce: u64,
}

impl Header {
    /// The hash of every field but the nonce, which the PoW starts from.
    pub fn seed(&self) -> Hash {
        let (lo, hi) = state::split(self.state.size);
        hash::sponge(
            domain::POW_SEED,
            &[],
            &[
                &u64_limbs(self.height),
                &self.prev,
                &u64_limbs(self.timestamp),
                &self.state.root,
                &[lo, hi],
                &self.body,
            ],
        )
    }

    /// The header's identity: its seed and nonce.
    pub fn hash(&self) -> Hash {
        hash::sponge(domain::HEADER, &[], &[&self.seed(), &u64_limbs(self.nonce)])
    }

    /// `height | prev | timestamp | state root | state size | body |
    /// nonce`, integers as little-endian `u64`s and hashes as
    /// [`hash::to_bytes`].
    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        let mut at = 0;
        let mut put = |bytes: &[u8]| {
            out[at..at + bytes.len()].copy_from_slice(bytes);
            at += bytes.len();
        };
        put(&self.height.to_le_bytes());
        put(&hash::to_bytes(&self.prev));
        put(&self.timestamp.to_le_bytes());
        put(&hash::to_bytes(&self.state.root));
        put(&self.state.size.to_le_bytes());
        put(&hash::to_bytes(&self.body));
        put(&self.nonce.to_le_bytes());
        out
    }

    /// Parses [`Header::to_bytes`] output. `None` if a hash isn't canonical
    /// or the state size exceeds the tree's capacity.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != HEADER_BYTES {
            return None;
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let b = &bytes[at..at + n];
            at += n;
            b
        };
        let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
        let hash_at = |b: &[u8]| hash::from_bytes(b.try_into().expect("hash bytes"));
        let height = u64_at(take(8));
        let prev = hash_at(take(hash::HASH_BYTES))?;
        let timestamp = u64_at(take(8));
        let root = hash_at(take(hash::HASH_BYTES))?;
        let size = u64_at(take(8));
        let body = hash_at(take(hash::HASH_BYTES))?;
        let nonce = u64_at(take(8));
        if size > state::CAPACITY {
            return None;
        }
        Some(Self {
            height,
            prev,
            timestamp,
            state: State { root, size },
            body,
            nonce,
        })
    }

    pub fn pow(&self, data: &Dataset) -> Hash {
        data.hash(&self.seed(), self.nonce)
    }

    /// Searches nonces from the current one until one meets `target`.
    /// Returns whether it found one.
    pub fn mine(&mut self, data: &Dataset, target: u128) -> bool {
        match data.mine(&self.seed(), target, self.nonce..u64::MAX) {
            Some(n) => {
                self.nonce = n;
                true
            }
            None => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeaderError {
    /// Not one above the parent's height.
    Height,
    /// Doesn't link to the parent.
    Prev,
    /// Not after the parent's timestamp.
    TooOld,
    /// Too far ahead of the node's clock.
    TooNew,
    /// The PoW hash isn't below the target.
    Work,
    /// Above the final height: the chain has ended.
    PastEnd,
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Height => "wrong height",
            Self::Prev => "doesn't link to its parent",
            Self::TooOld => "timestamp not after its parent's",
            Self::TooNew => "timestamp too far in the future",
            Self::Work => "insufficient proof of work",
            Self::PastEnd => "the chain has ended",
        })
    }
}

impl std::error::Error for HeaderError {}

/// Checks `header` as the child of `parent`: the rules a chain proof can
/// also enforce (height, link, increasing time, PoW).
pub fn check(
    c: &Consensus,
    data: &Dataset,
    parent: &Header,
    header: &Header,
) -> Result<(), HeaderError> {
    if header.height != parent.height + 1 {
        return Err(HeaderError::Height);
    }
    if header.height > c.final_height() {
        return Err(HeaderError::PastEnd);
    }
    if header.prev != parent.hash() {
        return Err(HeaderError::Prev);
    }
    if header.timestamp <= parent.timestamp {
        return Err(HeaderError::TooOld);
    }
    if pow::value(&header.pow(data)) >= c.target(parent) {
        return Err(HeaderError::Work);
    }
    Ok(())
}

/// [`check`], plus the node-local rule that `header` isn't more than
/// `max_future` past `now`.
pub fn check_new(
    c: &Consensus,
    data: &Dataset,
    parent: &Header,
    header: &Header,
    now: u64,
) -> Result<(), HeaderError> {
    if header.timestamp > now.saturating_add(c.max_future) {
        return Err(HeaderError::TooNew);
    }
    check(c, data, parent, header)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::BabyBear;
    use crate::hash::HASH_LEN;
    use crate::pow::MAX_TARGET;

    fn header(height: u64, timestamp: u64) -> Header {
        Header {
            height,
            prev: [BabyBear::ZERO; HASH_LEN],
            timestamp,
            state: State {
                root: [BabyBear::ONE; HASH_LEN],
                size: 1,
            },
            body: [BabyBear::ZERO; HASH_LEN],
            nonce: 0,
        }
    }

    #[test]
    fn target_of_matches_powers_of_two() {
        for int in 16..=108 {
            assert_eq!(target_of(int << FRAC_BITS), 1u128 << int);
        }
        assert_eq!(target_of(MAX_LOG_TARGET), MAX_TARGET);
        assert_eq!(target_of(-5), 1 << 16);
        // Linear within an octave, within 6.1% of 2^x.
        let half = target_of((60 << FRAC_BITS) + (1 << (FRAC_BITS - 1)));
        assert_eq!(half, 3 << 59);
        for f in (0..1 << FRAC_BITS).step_by(97) {
            let t = target_of((60 << FRAC_BITS) + f) as f64;
            let exact = 2f64.powf(60.0 + f as f64 / 65536.0);
            assert!(t >= exact && t / exact < 1.062);
        }
        let genesis = target_of(Consensus::DEFAULT.genesis_log_target) as f64;
        assert!((genesis.log2() - 84.72).abs() < 0.001);
        // Monotone in the log-target.
        let mut last = 0;
        for l in (40 << FRAC_BITS)..(41 << FRAC_BITS) {
            let t = target_of(l);
            assert!(t >= last);
            last = t;
        }
    }

    #[test]
    fn asert_follows_the_schedule() {
        let c = Consensus {
            genesis_time: 1000,
            spacing: 512,
            ..Consensus::DEFAULT
        };
        let l0 = c.genesis_log_target;
        let half_life = 1 << c.log_half_life;
        // On schedule: unchanged.
        assert_eq!(c.log_target(&header(0, 1000)), l0);
        assert_eq!(c.log_target(&header(10, 1000 + 10 * c.spacing)), l0);
        // One half-life behind: target doubles. Ahead: halves.
        let behind = header(10, 1000 + 10 * c.spacing + half_life);
        assert_eq!(c.log_target(&behind), l0 + (1 << FRAC_BITS));
        let ahead = header(10 + half_life / c.spacing, 1000 + 10 * c.spacing);
        assert_eq!(c.log_target(&ahead), l0 - (1 << FRAC_BITS));
        // Rounds down, even when ahead.
        assert_eq!(c.log_target(&header(1, 1001)), l0 - 256);
        assert_eq!(c.log_target(&header(1, 1002)), l0 - 255);
        // Clamped.
        assert_eq!(c.log_target(&header(0, u64::MAX)), MAX_LOG_TARGET);
        assert_eq!(c.log_target(&header(u64::MAX / 1024, 1000)), MIN_LOG_TARGET);
    }

    #[test]
    fn bytes_round_trip() {
        let mut h = header(123_456, 1_700_000_000);
        h.prev[2] = BabyBear::new(99);
        h.body[7] = BabyBear::new(crate::babybear::P - 1);
        h.state.size = state::CAPACITY;
        h.nonce = u64::MAX;
        let bytes = h.to_bytes();
        assert_eq!(Header::from_bytes(&bytes), Some(h));
        assert_eq!(Header::from_bytes(&bytes[1..]), None);
        let mut bad = bytes;
        bad[8..12].copy_from_slice(&crate::babybear::P.to_le_bytes());
        assert_eq!(Header::from_bytes(&bad), None, "non-canonical hash");
        let mut bad = bytes;
        let at = 8 + 32 + 8 + 32;
        bad[at..at + 8].copy_from_slice(&(state::CAPACITY + 1).to_le_bytes());
        assert_eq!(Header::from_bytes(&bad), None, "size over capacity");
    }

    #[test]
    fn emission_schedule() {
        let c = Consensus::DEFAULT;
        assert_eq!(c.subsidy(0), None);
        assert_eq!(c.subsidy(1), Some(100_000_000_000));
        assert_eq!(c.subsidy(367_920), Some(100_000_000_000));
        assert_eq!(c.subsidy(367_921), Some(700_000_000));
        assert_eq!(c.final_height(), 52_927_920);
        assert_eq!(c.subsidy(52_927_920), Some(700_000_000));
        assert_eq!(c.subsidy(52_927_921), None);
        // 367,920 * 1000 + 52,560,000 * 7 coins: the two eras emit equally.
        assert_eq!(c.total_supply(), 735_840_000 * COIN as u128);
        assert!(c.total_supply() < u64::MAX as u128);
    }

    #[test]
    fn work_is_scaled_expected_hashes() {
        // At octave boundaries, exactly 2^17 * MAX_TARGET / target.
        for int in 16..=108 {
            let l = int << FRAC_BITS;
            assert_eq!(work(l), (MAX_TARGET / target_of(l)) << 17, "{int}");
        }
        // Within 2^-16 below the exact expected hashes, everywhere, and
        // never increasing in the target.
        let mut last = u128::MAX;
        for l in (40 << FRAC_BITS)..(42 << FRAC_BITS) {
            let w = work(l);
            let exact = (MAX_TARGET << 17) as f64 / target_of(l) as f64;
            assert!(
                w as f64 <= exact && w as f64 >= exact * (1.0 - 1.0 / 65536.0),
                "{l}"
            );
            assert!(w <= last);
            last = w;
        }
        for w in [0, 1, 4095, 4096, work(MIN_LOG_TARGET), u128::MAX] {
            assert_eq!(work_value(&work_limbs(w)), Some(w));
        }
        let mut limbs = work_limbs(5);
        limbs[11] = crate::babybear::BabyBear::new(1);
        assert_eq!(work_value(&limbs), None, "beyond u128");
    }

    #[test]
    fn hashes_cover_every_field() {
        let h = header(5, 1234);
        let mut changed = [h; 7];
        changed[0].height += 1;
        changed[1].prev[3] += BabyBear::ONE;
        changed[2].timestamp += 1;
        changed[3].state.root[7] += BabyBear::ONE;
        changed[4].state.size += 1;
        changed[5].body[0] += BabyBear::ONE;
        changed[6].nonce += 1;
        for (i, c) in changed.iter().enumerate() {
            assert_ne!(c.hash(), h.hash(), "field {i}");
            assert_eq!(c.seed() == h.seed(), i == 6, "field {i}");
        }
    }

    /// Mines a short chain on the real dataset at an easy target, then
    /// checks each header and the rejections.
    #[test]
    #[ignore]
    fn mines_and_checks_a_chain() {
        let data = Dataset::get();
        let c = Consensus {
            genesis_time: 1_000_000,
            // About 2^10 expected hashes.
            genesis_log_target: (108 - 10) << FRAC_BITS,
            ..Consensus::DEFAULT
        };
        let genesis = header(0, c.genesis_time);
        let mut chain = vec![genesis];
        for i in 1..=5 {
            let parent = chain.last().unwrap();
            let mut h = Header {
                height: parent.height + 1,
                prev: parent.hash(),
                timestamp: parent.timestamp + 500 + 100 * i,
                ..*parent
            };
            assert!(h.mine(data, c.target(parent)));
            assert_eq!(check(&c, data, parent, &h), Ok(()));
            chain.push(h);
        }

        let (parent, h) = (&chain[4], chain[5]);
        let reject = |f: &dyn Fn(&mut Header)| {
            let mut bad = h;
            f(&mut bad);
            check(&c, data, parent, &bad)
        };
        assert_eq!(reject(&|b| b.height += 1), Err(HeaderError::Height));
        assert_eq!(
            reject(&|b| b.prev = chain[3].hash()),
            Err(HeaderError::Prev)
        );
        assert_eq!(
            reject(&|b| b.timestamp = parent.timestamp),
            Err(HeaderError::TooOld)
        );
        // Another nonce almost surely misses an easy-but-not-trivial target.
        let target = c.target(parent);
        let bad_nonce = (h.nonce + 1..)
            .find(|&n| pow::value(&data.hash(&h.seed(), n)) >= target)
            .unwrap();
        assert_eq!(reject(&|b| b.nonce = bad_nonce), Err(HeaderError::Work));
        assert_eq!(
            check_new(&c, data, parent, &h, h.timestamp - c.max_future - 1),
            Err(HeaderError::TooNew)
        );
        assert_eq!(
            check_new(&c, data, parent, &h, h.timestamp - c.max_future),
            Ok(())
        );
    }
}
