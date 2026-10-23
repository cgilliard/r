//! The header rules as a circuit: what a chain proof checks about the
//! previous block's header ([`crate::header::check`] minus the node-local
//! future-time rule).
//!
//! - **Link:** the header's height is its parent's plus one and its `prev`
//!   is the parent's hash.
//! - **Time:** its timestamp is after its parent's.
//! - **Difficulty:** the parent's height and timestamp give the log-target
//!   by ASERT; the log-target gives the target.
//! - **Work:** the PoW hash, recomputed with one dataset Merkle opening
//!   (against [`pow::ROOT`]) per round, is below the target; the header's
//!   work ([`crate::header::work`]) is computed for the chain's total.
//!
//! Integers wider than a field element (heights, timestamps, the schedule
//! arithmetic) are 16-bit limbs, compared with hinted carries.

use crate::babybear::{BabyBear, P};
use crate::circuit::{Cell, Circuit};
use crate::hash::{DOMAIN_SLOT, HASH_LEN, domain};
use crate::header::{
    AUX_LIMBS, Consensus, FRAC_BITS, Header, MAX_LOG_TARGET, MIN_LOG_TARGET, WORK_LIMB_BITS,
    WORK_LIMBS, aux_limbs,
};
use crate::poseidon2::WIDTH;
use crate::pow::{self, DatasetTree, LOG_ITEMS, ROUNDS, VALUE_BITS, VALUE_LIMB_BITS, u64_limbs};
use crate::verifier_circuit::{HashCells, canonical_bits, sponge, verify_path};

/// Limbs of a `u64`.
const LIMBS: usize = 4;
/// Bits of a log-target (`MAX_LOG_TARGET < 2^23`).
const LOG_TARGET_BITS: usize = 23;
/// Limbs of the schedule equation: `spacing * height` needs 74 bits.
const SCHEDULE_LIMBS: usize = 5;

const _: () = assert!(MAX_LOG_TARGET < 1 << LOG_TARGET_BITS);

/// A header's fields as cells, integers as 16-bit limbs.
#[derive(Clone, Copy, Debug)]
pub struct HeaderCells {
    pub height: [Cell; LIMBS],
    pub prev: HashCells,
    pub timestamp: [Cell; LIMBS],
    pub root: HashCells,
    pub size: [Cell; 2],
    pub body: HashCells,
    /// [`crate::header::aux_limbs`] (only hashed: no range checks).
    pub aux_hash: [Cell; AUX_LIMBS],
    pub nonce: [Cell; LIMBS],
}

impl HeaderCells {
    /// Inputs `h`'s fields, range-checking the limbs.
    pub fn input(c: &mut Circuit, h: &Header) -> Self {
        let limbs = |c: &mut Circuit, x: u64| {
            u64_limbs(x).map(|v| {
                let cell = c.input(v);
                c.bits(cell, 16);
                cell
            })
        };
        let height = limbs(c, h.height);
        let timestamp = limbs(c, h.timestamp);
        let nonce = limbs(c, h.nonce);
        let (lo, hi) = crate::state::split(h.state.size);
        Self {
            height,
            prev: h.prev.map(|v| c.input(v)),
            timestamp,
            root: h.state.root.map(|v| c.input(v)),
            size: [c.input(lo), c.input(hi)],
            body: h.body.map(|v| c.input(v)),
            aux_hash: aux_limbs(&h.aux_hash).map(|v| c.input(v)),
            nonce,
        }
    }

    /// [`Header::seed`].
    pub fn seed(&self, c: &mut Circuit) -> HashCells {
        let mut elems = self.height.to_vec();
        elems.extend(self.prev);
        elems.extend(self.timestamp);
        elems.extend(self.root);
        elems.extend(self.size);
        elems.extend(self.body);
        elems.extend(self.aux_hash);
        sponge(c, domain::POW_SEED, &elems, None)
    }

    /// [`Header::hash`], given the seed.
    pub fn hash_with_seed(&self, c: &mut Circuit, seed: &HashCells) -> HashCells {
        let mut elems = seed.to_vec();
        elems.extend(self.nonce);
        sponge(c, domain::HEADER, &elems, None)
    }

    pub fn hash(&self, c: &mut Circuit) -> HashCells {
        let seed = self.seed(c);
        self.hash_with_seed(c, &seed)
    }
}

/// What [`check_header`] establishes about a header.
#[derive(Clone, Copy, Debug)]
pub struct HeaderCheck {
    pub hash: HashCells,
    pub parent_hash: HashCells,
    /// The header's work, as [`crate::header::work_limbs`].
    pub work: [Cell; WORK_LIMBS],
}

/// Checks `header` as the child of `parent` under `consensus`, opening
/// dataset items from `tree`.
///
/// # Panics
/// If `consensus` has a spacing of `2^12` or more or a half-life under
/// `2^16` (the limb arithmetic assumes otherwise), or `tree` isn't the
/// dataset's shape.
pub fn check_header(
    c: &mut Circuit,
    consensus: &Consensus,
    tree: &DatasetTree,
    parent: &HeaderCells,
    header: &HeaderCells,
) -> HeaderCheck {
    if let DatasetTree::Full(t) = tree {
        assert_eq!(t.log_height(), LOG_ITEMS, "not the dataset's tree");
    }
    // Height: parent + 1.
    let one = c.one();
    let mut lhs = parent.height;
    lhs[0] = c.add(lhs[0], one);
    int_eq(c, &lhs, &header.height, 2);

    // Link.
    let parent_hash = parent.hash(c);
    for (&a, &b) in header.prev.iter().zip(&parent_hash) {
        c.assert_eq(a, b);
    }

    // Time: header = parent + 1 + gap, gap >= 0.
    let gap = u64_of(c, &header.timestamp)
        .wrapping_sub(u64_of(c, &parent.timestamp))
        .wrapping_sub(1);
    let gap = u64_limbs(gap).map(|v| {
        let g = c.hint(v);
        c.bits(g, 16);
        g
    });
    let mut lhs = [one; LIMBS];
    for k in 0..LIMBS {
        lhs[k] = c.add(parent.timestamp[k], gap[k]);
    }
    lhs[0] = c.add(lhs[0], one);
    int_eq(c, &lhs, &header.timestamp, 3);

    // Work.
    let log_target = log_target(c, consensus, &parent.height, &parent.timestamp);
    let seed = header.seed(c);
    let h = pow_hash(c, tree, &seed, &header.nonce);
    let (frac, shifts) = assert_below_target(c, &h, log_target);
    HeaderCheck {
        hash: header.hash_with_seed(c, &seed),
        parent_hash,
        work: work(c, frac, &shifts),
    }
}

/// [`crate::header::work`] of the log-target with fraction `frac` and
/// one-hot `shifts` (`shifts[s]` set for `int = 16 + s`):
/// `floor(2^33 / (2^16 + frac))` placed at bit `108 - int`, as normalized
/// limbs.
fn work(c: &mut Circuit, frac: Cell, shifts: &[Cell]) -> [Cell; WORK_LIMBS] {
    let bits = WORK_LIMB_BITS as usize;
    let m = quotient(c, frac);
    let max_shift = shifts.len() - 1;
    let mut out = [c.zero(); WORK_LIMBS];
    let mut carry = c.zero();
    for (k, limb) in out.iter_mut().enumerate() {
        // g = sum of 2^(bit % 12) over the shifts placing m in this limb.
        let mut g = c.zero();
        for (s, &sel) in shifts.iter().enumerate() {
            let bit = max_shift - s;
            if bit / bits == k {
                let t = c.mul_const(sel, BabyBear::new(1 << (bit % bits)));
                g = c.add(g, t);
            }
        }
        let placed = c.mul(m, g);
        let t = c.add(placed, carry);
        let (lo, next) = split_limb(c, t, 18);
        *limb = lo;
        carry = next;
    }
    c.assert_zero(carry);
    out
}

/// `q = floor(2^33 / d)` for `d = 2^16 + frac` (`frac` 16 bits), so
/// `q` is in `(2^16, 2^17]`.
///
/// The prover supplies `q` as `q1 2^9 + q0` (9 bits each) and the remainder
/// `r = 2^33 - q d`, and the circuit checks `0 <= r < d`. The products stay
/// far below `p` by splitting the equation at bit 9: `q0 d + r = c 2^9` and
/// `q1 d + c = 2^24`, with `c` hinted.
fn quotient(c: &mut Circuit, frac: Cell) -> Cell {
    quotient_claiming(c, frac, 0)
}

/// [`quotient`], with the prover claiming the true quotient plus `error`.
fn quotient_claiming(c: &mut Circuit, frac: Cell, error: i64) -> Cell {
    let d = c.add_const(frac, BabyBear::new(1 << FRAC_BITS));
    let dv = c.value(d).as_u32() as u64;
    let num = 1u64 << (2 * FRAC_BITS + 1);
    let qv = (num / dv).wrapping_add_signed(error);
    let rv = num.wrapping_sub(qv * dv) & 0xffff_ffff;
    let q0 = c.hint(BabyBear::new((qv & 0x1ff) as u32));
    let q1 = c.hint(BabyBear::new((qv >> 9) as u32));
    let r = c.hint(BabyBear::new(rv as u32));
    let carry = c.hint(BabyBear::new(((qv & 0x1ff) * dv + rv) as u32 >> 9));
    c.bits(q0, 9);
    c.bits(q1, 9);
    c.bits(carry, 18);
    // 0 <= r < d.
    c.bits(r, 17);
    let room = c.sub(d, r);
    let room = c.add_const(room, -BabyBear::ONE);
    c.bits(room, 17);
    // q0 d + r = carry 2^9.
    let low = c.mul(q0, d);
    let low = c.add(low, r);
    let shifted = c.mul_const(carry, BabyBear::new(1 << 9));
    c.assert_eq(low, shifted);
    // q1 d + carry = 2^24.
    let high = c.mul(q1, d);
    let high = c.add(high, carry);
    let top = c.constant(BabyBear::new(1 << 24));
    c.assert_eq(high, top);
    let q1_shifted = c.mul_const(q1, BabyBear::new(1 << 9));
    c.add(q1_shifted, q0)
}

/// `t = lo + 2^12 next` with `lo` 12 bits and `next` `next_bits` bits.
fn split_limb(c: &mut Circuit, t: Cell, next_bits: usize) -> (Cell, Cell) {
    let bits = WORK_LIMB_BITS as usize;
    let v = c.value(t).as_u32();
    let lo = c.hint(BabyBear::new(v & ((1 << bits) - 1)));
    let next = c.hint(BabyBear::new(v >> bits));
    c.bits(lo, bits);
    c.bits(next, next_bits);
    let shifted = c.mul_const(next, BabyBear::new(1 << bits));
    let sum = c.add(lo, shifted);
    c.assert_eq(sum, t);
    (lo, next)
}

/// The sum of two work values in normalized limbs; fails on overflow.
pub fn add_work(c: &mut Circuit, a: &[Cell], b: &[Cell]) -> [Cell; WORK_LIMBS] {
    let mut out = [c.zero(); WORK_LIMBS];
    let mut carry = c.zero();
    for k in 0..WORK_LIMBS {
        let s = c.add(a[k], b[k]);
        let t = c.add(s, carry);
        let (lo, next) = split_limb(c, t, 1);
        out[k] = lo;
        carry = next;
    }
    c.assert_zero(carry);
    out
}

/// The PoW hash for `seed` and `nonce`, with each round's item opened
/// against [`pow::ROOT`].
fn pow_hash(
    c: &mut Circuit,
    tree: &DatasetTree,
    seed: &HashCells,
    nonce: &[Cell; LIMBS],
) -> HashCells {
    let zero = c.zero();
    let mut s = [zero; WIDTH];
    s[..HASH_LEN].copy_from_slice(seed);
    s[HASH_LEN..HASH_LEN + LIMBS].copy_from_slice(nonce);
    s[DOMAIN_SLOT] = c.constant(BabyBear::new(domain::POW));
    s = c.perm(s);
    let root = pow::ROOT.map(|v| c.constant(BabyBear::new(v)));
    for _ in 0..ROUNDS {
        let bits = canonical_bits(c, s[HASH_LEN]);
        let index = c.value(s[HASH_LEN]).as_u32() as usize & (pow::ITEMS - 1);
        let (row, path) = tree.open(index);
        let row: Vec<Cell> = row.iter().map(|&v| c.hint(v)).collect();
        let path: Vec<HashCells> = path.iter().map(|h| h.map(|v| c.hint(v))).collect();
        verify_path(c, &root, &row, &bits[..LOG_ITEMS as usize], &path, None);
        s = c.sponge(s, &row);
    }
    core::array::from_fn(|i| s[i])
}

/// The log-target after a parent at `height` and `timestamp`: ASERT as in
/// [`Consensus::log_target`].
///
/// With `D = timestamp - genesis_time - spacing * height`, `m` the
/// half-life's shift and `L` the clamped log-target, the prover supplies
/// `L`, `b < 2^m` and non-negative `pos`, `neg` with
///
/// `timestamp + 2^m L0 + neg = genesis_time + spacing * height + 2^m L + b + pos`
///
/// where `pos` may be nonzero only if `L` is the maximum and `neg` only if
/// it's the minimum. With both zero this says `L = L0 + floor(D / 2^m)`;
/// `pos > 0` says the unclamped value is at least the maximum, `neg > 0`
/// that it's at most the minimum.
fn log_target(
    c: &mut Circuit,
    consensus: &Consensus,
    height: &[Cell; LIMBS],
    timestamp: &[Cell; LIMBS],
) -> Cell {
    log_target_claiming(c, consensus, height, timestamp, 0)
}

/// [`log_target`], with the prover claiming the true value plus `error`.
fn log_target_claiming(
    c: &mut Circuit,
    consensus: &Consensus,
    height: &[Cell; LIMBS],
    timestamp: &[Cell; LIMBS],
    error: i64,
) -> Cell {
    assert!(consensus.spacing < 1 << 12, "spacing too large");
    assert!(consensus.log_half_life >= FRAC_BITS, "half-life too short");
    let m = consensus.log_half_life - FRAC_BITS;
    assert!(m <= 4, "half-life too long");
    let (h, t) = (u64_of(c, height), u64_of(c, timestamp));
    let parent = Header {
        height: h,
        timestamp: t,
        ..Header::default()
    };
    let l = consensus.log_target(&parent) + error;
    let d = t as i128 - consensus.genesis_time as i128 - consensus.spacing as i128 * h as i128;
    let b = d - ((d >> m) << m);
    let delta = d - ((l - consensus.genesis_log_target) as i128) * (1 << m) - b;
    let (pos, neg) = (delta.max(0) as u128, (-delta).max(0) as u128);

    let l_cell = c.hint(BabyBear::new(l as u32));
    let lo = c.add_const(l_cell, -BabyBear::new(MIN_LOG_TARGET as u32));
    c.bits(lo, LOG_TARGET_BITS);
    let hi_const = c.constant(BabyBear::new(MAX_LOG_TARGET as u32));
    let hi = c.sub(hi_const, l_cell);
    c.bits(hi, LOG_TARGET_BITS);
    let b_cell = c.hint(BabyBear::new(b as u32));
    if m > 0 {
        c.bits(b_cell, m as usize);
    } else {
        c.assert_zero(b_cell);
    }
    let limbs_of = |c: &mut Circuit, x: u128| -> [Cell; SCHEDULE_LIMBS] {
        core::array::from_fn(|k| {
            let cell = c.hint(BabyBear::new((x >> (16 * k)) as u16 as u32));
            c.bits(cell, 16);
            cell
        })
    };
    let (pos, neg) = (limbs_of(c, pos), limbs_of(c, neg));
    for k in 0..SCHEDULE_LIMBS {
        let p = c.mul(hi, pos[k]);
        c.assert_zero(p);
        let n = c.mul(lo, neg[k]);
        c.assert_zero(n);
    }

    let l0 = (consensus.genesis_log_target as u128) << m;
    let g = consensus.genesis_time as u128;
    let limb = |x: u128, k: usize| BabyBear::new((x >> (16 * k)) as u16 as u32);
    let scaled_l = c.mul_const(l_cell, BabyBear::new(1 << m));
    let mut lhs = Vec::with_capacity(SCHEDULE_LIMBS);
    let mut rhs = Vec::with_capacity(SCHEDULE_LIMBS);
    for k in 0..SCHEDULE_LIMBS {
        let t = if k < LIMBS { timestamp[k] } else { c.zero() };
        let x = c.add_const(t, limb(l0, k));
        lhs.push(c.add(x, neg[k]));
        let mut y = c.constant(limb(g, k));
        if k < LIMBS {
            let sh = c.mul_const(height[k], BabyBear::new(consensus.spacing as u32));
            y = c.add(y, sh);
        }
        if k == 0 {
            y = c.add(y, scaled_l);
            y = c.add(y, b_cell);
        }
        rhs.push(c.add(y, pos[k]));
    }
    int_eq(c, &lhs, &rhs, 12);
    l_cell
}

/// Asserts the value of PoW hash `h` ([`pow::value`]) is below
/// [`crate::header::target_of`] of `log_target`, which must be in range.
///
/// With `int, frac` the log-target's parts, `s = int - 16` and
/// `f = 2^16 + frac`, the target is `f * 2^s`, so `value < target` iff the
/// value has no bits from `s + 17` up and its bits from `s` up, read as an
/// integer, are below `f`. `s` is picked out by a one-hot hint. Returns
/// `frac` and the one-hot selectors.
fn assert_below_target(c: &mut Circuit, h: &HashCells, log_target: Cell) -> (Cell, Vec<Cell>) {
    let bits_per = VALUE_LIMB_BITS as usize;
    let n = VALUE_BITS as usize;
    // Value bits, least significant first: the last element is lowest.
    let mut v = Vec::with_capacity(n);
    for i in (0..4).rev() {
        let bits = canonical_bits(c, h[i]);
        v.extend_from_slice(&bits[..bits_per]);
    }
    // suffix[k] = number of set bits at k and above; window[k] = the bits
    // from k up as an integer (exact when suffix[k + 17] = 0).
    let zero = c.zero();
    let mut suffix = vec![zero; n + 2];
    let mut window = vec![zero; n + 1];
    for k in (0..n).rev() {
        suffix[k] = c.add(suffix[k + 1], v[k]);
        let w = c.add(window[k + 1], window[k + 1]);
        window[k] = c.add(w, v[k]);
    }

    let lt = c.bits(log_target, LOG_TARGET_BITS);
    let weighted = |c: &mut Circuit, bits: &[Cell]| {
        let mut acc = c.zero();
        for (i, &b) in bits.iter().enumerate() {
            let t = c.mul_const(b, BabyBear::new(1 << i));
            acc = c.add(acc, t);
        }
        acc
    };
    let frac = weighted(c, &lt[..FRAC_BITS as usize]);
    let int = weighted(c, &lt[FRAC_BITS as usize..]);
    let shift = c.add_const(int, -BabyBear::new(FRAC_BITS));

    let shifts = (MAX_LOG_TARGET >> FRAC_BITS) as usize - FRAC_BITS as usize;
    let s_value = c.value(shift).as_u32() as usize;
    let one = c.one();
    let (mut sum, mut index, mut high, mut w) = (c.zero(), c.zero(), c.zero(), c.zero());
    let mut sels = Vec::with_capacity(shifts + 1);
    for s in 0..=shifts {
        let sel = c.hint(BabyBear::from(s == s_value));
        sels.push(sel);
        let ss = c.mul(sel, sel);
        c.assert_eq(ss, sel);
        sum = c.add(sum, sel);
        let t = c.mul_const(sel, BabyBear::new(s as u32));
        index = c.add(index, t);
        let t = c.mul(sel, suffix[(s + 17).min(n + 1)]);
        high = c.add(high, t);
        let t = c.mul(sel, window[s.min(n)]);
        w = c.add(w, t);
    }
    c.assert_eq(sum, one);
    c.assert_eq(index, shift);
    c.assert_zero(high);
    // f - 1 - w in [0, 2^17).
    let f = c.add_const(frac, BabyBear::new((1 << FRAC_BITS) - 1));
    let d = c.sub(f, w);
    c.bits(d, 17);
    (frac, sels)
}

/// Asserts `sum lhs[k] 2^(16k) = sum rhs[k] 2^(16k)` as integers, given
/// per-position sums whose differences (plus carries) stay far below
/// `p / 2`, with carries below `2^(carry_bits - 1)` in magnitude.
fn int_eq(c: &mut Circuit, lhs: &[Cell], rhs: &[Cell], carry_bits: usize) {
    assert_eq!(lhs.len(), rhs.len());
    let mut carry = c.zero();
    for k in 0..lhs.len() {
        let d = c.sub(lhs[k], rhs[k]);
        let d = c.add(d, carry);
        if k + 1 == lhs.len() {
            c.assert_zero(d);
            break;
        }
        let next = c.hint(from_signed(signed(c.value(d)) >> 16));
        let shifted = c.mul_const(next, BabyBear::new(1 << 16));
        c.assert_eq(d, shifted);
        let off = c.add_const(next, BabyBear::new(1 << (carry_bits - 1)));
        c.bits(off, carry_bits);
        carry = next;
    }
}

fn signed(x: BabyBear) -> i64 {
    let v = x.as_u32() as i64;
    if v > (P / 2) as i64 { v - P as i64 } else { v }
}

fn from_signed(x: i64) -> BabyBear {
    BabyBear::new(x.rem_euclid(P as i64) as u32)
}

/// The `u64` in 16-bit limb cells (their current values).
fn u64_of(c: &Circuit, limbs: &[Cell; LIMBS]) -> u64 {
    limbs.iter().enumerate().fold(0, |acc, (k, &l)| {
        acc | (c.value(l).as_u32() as u64 & 0xffff) << (16 * k)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::target_of;

    fn header(height: u64, timestamp: u64) -> Header {
        Header {
            height,
            timestamp,
            aux_hash: [0xff; 32],
            ..Header::default()
        }
    }

    #[test]
    fn log_target_matches_native() {
        let c0 = Consensus {
            genesis_time: 1_000_000,
            ..Consensus::DEFAULT
        };
        let c1 = Consensus {
            log_half_life: 16,
            spacing: 1,
            ..c0
        };
        let c3 = Consensus {
            log_half_life: 19,
            ..c0
        };
        for consensus in [c0, c1, c3] {
            let g = consensus.genesis_time;
            let s = consensus.spacing;
            for (h, t) in [
                (0, g),
                (1, g + 1),
                (1, g + 2),
                (10, g + 10 * s),
                (10, g + 10 * s + 12345),
                (10, g + 10 * s - 12345),
                (5000, g + 1),     // far ahead: clamps low
                (1, u64::MAX),     // far behind: clamps high
                (u64::MAX / 2, g), // huge height
                (0, 0),            // before genesis
                (3, g + 3 * s + (1 << 30)),
            ] {
                let parent = header(h, t);
                let mut c = Circuit::new();
                let cells = HeaderCells::input(&mut c, &parent);
                let l = log_target(&mut c, &consensus, &cells.height, &cells.timestamp);
                assert!(c.failures().is_empty(), "{h} {t}");
                assert!(c.check().is_ok());
                assert_eq!(
                    c.value(l),
                    BabyBear::new(consensus.log_target(&parent) as u32)
                );
                for error in [-1, 1] {
                    let mut c = Circuit::new();
                    let cells = HeaderCells::input(&mut c, &parent);
                    log_target_claiming(&mut c, &consensus, &cells.height, &cells.timestamp, error);
                    assert!(!c.failures().is_empty(), "{h} {t} accepted {error}");
                }
            }
        }
    }

    /// A hash whose value is `v`.
    fn hash_of(v: u128) -> [BabyBear; HASH_LEN] {
        let mask = (1 << VALUE_LIMB_BITS) - 1;
        core::array::from_fn(|i| {
            if i < 4 {
                // Add a multiple of 2^27 above the value bits to vary the
                // ignored top bits.
                let x = (v >> (VALUE_LIMB_BITS * (3 - i as u32))) as u32 & mask;
                BabyBear::new(x + (((i as u32 * 5) % 15) << VALUE_LIMB_BITS))
            } else {
                BabyBear::new(7)
            }
        })
    }

    fn below(v: u128, log_target: i64) -> bool {
        let mut c = Circuit::new();
        let h = hash_of(v).map(|x| c.input(x));
        let l = c.input(BabyBear::new(log_target as u32));
        assert_below_target(&mut c, &h, l);
        assert!(c.check().is_ok() || !c.failures().is_empty());
        c.failures().is_empty()
    }

    #[test]
    fn target_comparison_matches_native() {
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let logs = [
            MIN_LOG_TARGET,
            MIN_LOG_TARGET + 1,
            (40 << FRAC_BITS) + 12345,
            (81 << FRAC_BITS) + 20938,
            (100 << FRAC_BITS) + 65535,
            MAX_LOG_TARGET,
        ];
        for &l in &logs {
            let t = target_of(l);
            for v in [0, 1, t - 2, t - 1, t, t + 1, 2 * t, pow::MAX_TARGET - 1]
                .into_iter()
                .filter(|&v| v < pow::MAX_TARGET)
            {
                assert_eq!(below(v, l), v < t, "value {v:#x} log-target {l}");
            }
            for _ in 0..20 {
                // Random values near the target's magnitude.
                let r = ((next() as u128) << 64 | next() as u128) % (2 * t);
                let r = r.min(pow::MAX_TARGET - 1);
                assert_eq!(below(r, l), r < t);
            }
        }
    }

    /// The full header check on the real dataset: a mined header passes,
    /// broken ones don't.
    #[test]
    #[ignore]
    fn checks_a_mined_header() {
        let data = pow::Dataset::get();
        let t = std::time::Instant::now();
        let tree = DatasetTree::Full(data.tree());
        eprintln!("tree {:.2?}", t.elapsed());
        let consensus = Consensus {
            genesis_time: 1_000_000,
            genesis_log_target: (108 - 12) << FRAC_BITS,
            ..Consensus::DEFAULT
        };
        let parent = Header {
            height: 41,
            timestamp: 1_000_000 + 41 * 600 + 77,
            aux_hash: [7; 32],
            ..Header::default()
        };
        let mut h = Header {
            height: 42,
            prev: parent.hash(),
            timestamp: parent.timestamp + 610,
            nonce: 0,
            ..parent
        };
        assert!(h.mine(data, consensus.target(&parent)));
        assert_eq!(crate::header::check(&consensus, &parent, &h), Ok(()));

        let build = |h: &Header| {
            let mut c = Circuit::new();
            let p = HeaderCells::input(&mut c, &parent);
            let hc = HeaderCells::input(&mut c, h);
            let hash = check_header(&mut c, &consensus, &tree, &p, &hc);
            (c, hash)
        };
        let (c, check) = build(&h);
        assert!(c.failures().is_empty());
        assert!(c.check().is_ok());
        assert_eq!(check.hash.map(|x| c.value(x)), h.hash());
        assert_eq!(check.parent_hash.map(|x| c.value(x)), parent.hash());
        let work = crate::header::work(consensus.log_target(&parent));
        assert_eq!(
            check.work.map(|x| c.value(x)),
            crate::header::work_limbs(work)
        );
        eprintln!("{:?}", c.stats());

        let target = consensus.target(&parent);
        let bad_nonce = (h.nonce + 1..)
            .find(|&n| pow::value(&data.hash(&h.seed(), n)) >= target)
            .unwrap();
        for (what, bad) in [
            (
                "nonce",
                Header {
                    nonce: bad_nonce,
                    ..h
                },
            ),
            ("height", Header { height: 43, ..h }),
            (
                "aux hash",
                Header {
                    aux_hash: [8; 32],
                    ..h
                },
            ),
            (
                "prev",
                Header {
                    prev: [BabyBear::ONE; HASH_LEN],
                    ..h
                },
            ),
            (
                "timestamp",
                Header {
                    timestamp: parent.timestamp,
                    ..h
                },
            ),
        ] {
            assert!(!build(&bad).0.failures().is_empty(), "{what}");
        }
    }

    #[test]
    fn quotient_is_exact() {
        for frac in (0..1u32 << FRAC_BITS).step_by(97).chain([1, 65534, 65535]) {
            let mut c = Circuit::new();
            let f = c.input(BabyBear::new(frac));
            let q = quotient(&mut c, f);
            assert!(c.failures().is_empty(), "{frac}");
            assert!(c.check().is_ok());
            let want = (1u64 << 33) / ((1 << 16) + frac as u64);
            assert_eq!(c.value(q).as_u32() as u64, want, "{frac}");
            for error in [-1, 1] {
                let mut c = Circuit::new();
                let f = c.input(BabyBear::new(frac));
                quotient_claiming(&mut c, f, error);
                assert!(!c.failures().is_empty(), "{frac} accepted {error}");
            }
        }
    }

    #[test]
    fn work_matches_native() {
        for l in [
            MIN_LOG_TARGET,
            MIN_LOG_TARGET + 1,
            (27 << FRAC_BITS) + 4095,
            (84 << FRAC_BITS) + 42406,
            (100 << FRAC_BITS) + 65535,
            MAX_LOG_TARGET,
        ] {
            let mut c = Circuit::new();
            let h = hash_of(0).map(|x| c.input(x));
            let lt = c.input(BabyBear::new(l as u32));
            let (frac, shifts) = assert_below_target(&mut c, &h, lt);
            let w = work(&mut c, frac, &shifts);
            assert!(c.failures().is_empty());
            let want = crate::header::work_limbs(crate::header::work(l));
            assert_eq!(w.map(|x| c.value(x)), want, "{l}");
            // And sums.
            let doubled = add_work(&mut c, &w, &w);
            let want = crate::header::work_limbs(2 * crate::header::work(l));
            assert_eq!(doubled.map(|x| c.value(x)), want);
            assert!(c.failures().is_empty());
        }
    }
}
