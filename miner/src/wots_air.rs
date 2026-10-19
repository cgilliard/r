//! WOTS signature verification as AIR rows: proves `wots::verify(pk, msg,
//! sig)` with the signature as private witness.
//!
//! Every row holds one Poseidon2 permutation (see [`poseidon2_air`]) plus
//! auxiliary columns; preprocessed selectors give each row its role:
//!
//! - `M0`, `M1`: the message sponge over `rand || msg`, giving the digest
//!   whose low 24 bits per element are the message digits.
//! - per chain `i`, step rows `S0..S6`: chain step `t` with input
//!   `[v, param, WOTS_CHAIN, i, t]`. A step is active iff `t >= d_i`; inactive
//!   steps pass `v` through. Activity is monotone and the inactive count must
//!   equal `d_i`.
//! - per chain, an absorb row `A`: adds the chain end into the public-key
//!   sponge, whose state is carried through the next chain's step rows.
//!
//! Digits: a remainder column consumes one 3-bit message digit per chain;
//! at each digest element's first chain `e = rem + 2^24 h` with `h <= 120`
//! and `h = 120 => rem = 0`, which forces the canonical value. A running
//! checksum is consumed into the 3 checksum digits the same way.
//!
//! Every constraint is a preprocessed selector times at most degree 3.

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::hash::{HASH_LEN, domain};
use crate::matrix::RowMajorMatrix;
use crate::poseidon2::WIDTH;
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::wots::{self, NUM_CHAINS, PARAM_LEN, PublicKey, RAND_LEN, Signature, W};

/// Chain steps per chain: positions `0..W - 1`.
const STEPS: usize = W - 1;
const ROWS_PER_CHAIN: usize = STEPS + 1;
/// Rows used by one verification.
pub const NUM_ROWS: usize = 2 + NUM_CHAINS * ROWS_PER_CHAIN;
/// Trace height of the standalone AIR.
pub const LOG_HEIGHT: u32 = NUM_ROWS.next_power_of_two().trailing_zeros();

const MSG_CHAINS: usize = wots::MSG_CHAINS;
const DIGITS_PER_ELEM: usize = 8;
/// Message-sponge input length: `rand` then `msg`.
const MSG_LEN: usize = RAND_LEN + HASH_LEN;
/// Public-key sponge input length: every chain end.
const PK_LEN: usize = NUM_CHAINS * HASH_LEN;

// Auxiliary columns, after the permutation's.
const P: usize = poseidon2_air::NUM_COLS;
const VAL: usize = P;
const CARRY: usize = VAL + HASH_LEN;
pub(crate) const PARAM: usize = CARRY + WIDTH;
pub(crate) const MSG: usize = PARAM + PARAM_LEN;
const E: usize = MSG + HASH_LEN;
const MBITS: usize = E + HASH_LEN;
const CBITS: usize = MBITS + 3;
const ACTIVE: usize = CBITS + 3;
const IA: usize = ACTIVE + 1;
const REM: usize = IA + 1;
const CS: usize = REM + 1;
const CR: usize = CS + 1;
const HBITS: usize = CR + 1;
const GBITS: usize = HBITS + 7;
const INV: usize = GBITS + 7;
/// Trace width.
pub const NUM_COLS: usize = INV + 1;

// Preprocessed columns.
const M0: usize = 0;
const M1: usize = 1;
const STEP: usize = 2;
const STEP_FIRST: usize = 3;
const STEP_NONLAST: usize = 4;
const ABSORB: usize = 5;
const ABSORB_NONLAST: usize = 6;
const CHAIN_IDX: usize = 7;
const POS: usize = 8;
/// Absorb rows of message chains: chain digit = message digit, checksum grows.
const MSG_ABSORB: usize = 9;
const ELEM_START: usize = 10;
const ELEM_END: usize = 11;
/// Absorb rows where the digest carry shifts to the next element.
const SHIFT_E: usize = 12;
/// Checksum digit weights `64, 8, 1` on the checksum chains' absorb rows.
const CWEIGHT: usize = 13;
const CS_START: usize = 14;
const FIRST_CHAIN: usize = 15;
pub(crate) const LAST_ABSORB: usize = 16;
const CS_ABSORB_NONLAST: usize = 17;
pub const PRE_WIDTH: usize = 18;

/// The highest digest element value above the low 24 bits: `p = 120 * 2^24 + 1`.
const H_MAX: u32 = 120;

/// The preprocessed rows of one verification (`NUM_ROWS` of them).
pub(crate) fn preprocessed_rows() -> Vec<[BabyBear; PRE_WIDTH]> {
    let one = BabyBear::ONE;
    let mut rows = vec![[BabyBear::ZERO; PRE_WIDTH]; NUM_ROWS];
    rows[0][M0] = one;
    rows[1][M1] = one;
    for i in 0..NUM_CHAINS {
        let base = 2 + i * ROWS_PER_CHAIN;
        for t in 0..STEPS {
            let r = &mut rows[base + t];
            r[STEP] = one;
            r[STEP_FIRST] = BabyBear::from(t == 0);
            r[STEP_NONLAST] = BabyBear::from(t + 1 < STEPS);
            r[CHAIN_IDX] = BabyBear::new(i as u32);
            r[POS] = BabyBear::new(t as u32);
            r[ELEM_START] = BabyBear::from(t == 0 && i % DIGITS_PER_ELEM == 0);
            r[CS_START] = BabyBear::from(t == 0 && i == MSG_CHAINS);
            r[FIRST_CHAIN] = BabyBear::from(t == 0 && i == 0);
        }
        let a = &mut rows[base + STEPS];
        let last = i == NUM_CHAINS - 1;
        let elem_end = i % DIGITS_PER_ELEM == DIGITS_PER_ELEM - 1;
        a[ABSORB] = one;
        a[ABSORB_NONLAST] = BabyBear::from(!last);
        a[MSG_ABSORB] = BabyBear::from(i < MSG_CHAINS);
        a[ELEM_END] = BabyBear::from(elem_end);
        a[SHIFT_E] = BabyBear::from(elem_end && !last);
        if i >= MSG_CHAINS {
            a[CWEIGHT] = BabyBear::new(1 << (3 * (NUM_CHAINS - 1 - i)));
        }
        a[LAST_ABSORB] = BabyBear::from(last);
        a[CS_ABSORB_NONLAST] = BabyBear::from(i >= MSG_CHAINS && !last);
    }
    rows
}

/// Constrains one verification's rows against the `MSG` and `PARAM`
/// columns. With `root`, the last absorb row's output must equal it;
/// without, the caller constrains that output (the public-key root).
/// Constancy of `MSG` and `PARAM` across the rows is also the caller's.
pub(crate) fn eval<V: Var>(c: &mut Constraints<'_, V>, root: Option<&[V]>) {
    let (l, n, p) = (c.local(), c.next(), c.preprocessed_local());
    let k = |x: u32| V::from(BabyBear::new(x));
    let input = |j: usize| l[j];
    let out = |j: usize| l[OUTPUT_OFFSET + j];
    let param = |j: usize| l[PARAM + j];
    let bits = |col: usize| l[col] + l[col + 1] * BabyBear::new(2) + l[col + 2] * BabyBear::new(4);

    poseidon2_air::eval(c, l);

    // M0: input [rand, msg_0, param, WOTS_MESSAGE, 15, 0]; M1 adds msg_1..7.
    let m0 = p[M0];
    c.assert_zero(m0 * (input(7) - l[MSG]));
    for j in 0..PARAM_LEN {
        c.assert_zero(m0 * (input(HASH_LEN + j) - param(j)));
    }
    c.assert_zero(m0 * (input(13) - k(domain::WOTS_MESSAGE)));
    c.assert_zero(m0 * (input(14) - k(MSG_LEN as u32)));
    c.assert_zero(m0 * input(15));
    for j in 0..WIDTH {
        let absorbed = if j < HASH_LEN - 1 {
            l[MSG + j + 1]
        } else {
            k(0)
        };
        c.assert_zero(m0 * (n[j] - out(j) - absorbed));
    }
    // M1's rate is the digest, carried into the chains.
    for j in 0..HASH_LEN {
        c.assert_zero(p[M1] * (n[E + j] - out(j)));
    }

    // First chain: public-key sponge initial state, empty checksum.
    let first = p[FIRST_CHAIN];
    for j in 0..HASH_LEN {
        c.assert_zero(first * l[CARRY + j]);
    }
    for j in 0..PARAM_LEN {
        c.assert_zero(first * (l[CARRY + HASH_LEN + j] - param(j)));
    }
    c.assert_zero(first * (l[CARRY + 13] - k(domain::WOTS_PUBLIC_KEY)));
    c.assert_zero(first * (l[CARRY + 14] - k(PK_LEN as u32)));
    c.assert_zero(first * l[CARRY + 15]);
    c.assert_zero(first * l[CS]);

    // Step rows.
    let step = p[STEP];
    let a = l[ACTIVE];
    for j in 0..HASH_LEN {
        c.assert_zero(step * (input(j) - l[VAL + j]));
        // Active: compress = output + input; inactive: pass through.
        c.assert_zero(step * (n[VAL + j] - l[VAL + j] - a * out(j)));
    }
    for j in 0..PARAM_LEN {
        c.assert_zero(step * (input(HASH_LEN + j) - param(j)));
    }
    c.assert_zero(step * (input(13) - k(domain::WOTS_CHAIN)));
    c.assert_zero(step * (input(14) - p[CHAIN_IDX]));
    c.assert_zero(step * (input(15) - p[POS]));
    c.assert_zero(step * a * (a - BabyBear::ONE));
    c.assert_zero(step * (n[IA] - l[IA] - k(1) + a));
    for col in (CARRY..CARRY + WIDTH)
        .chain(E..E + HASH_LEN)
        .chain([REM, CS, CR])
    {
        c.assert_zero(step * (n[col] - l[col]));
    }
    for col in MBITS..CBITS + 3 {
        c.assert_zero(step * (n[col] - l[col]));
    }
    c.assert_zero(p[STEP_FIRST] * l[IA]);
    c.assert_zero(p[STEP_NONLAST] * a * (k(1) - n[ACTIVE]));

    // Absorb rows.
    let absorb = p[ABSORB];
    for j in 0..HASH_LEN {
        c.assert_zero(absorb * (input(j) - l[CARRY + j] - l[VAL + j]));
        c.assert_zero(absorb * (input(HASH_LEN + j) - l[CARRY + HASH_LEN + j]));
    }
    for &bit in &l[MBITS..CBITS + 3] {
        c.assert_zero(absorb * bit * (bit - BabyBear::ONE));
    }
    let (d, m) = (bits(CBITS), bits(MBITS));
    c.assert_zero(absorb * (l[IA] - d));
    for b in 0..3 {
        c.assert_zero(p[MSG_ABSORB] * (l[CBITS + b] - l[MBITS + b]));
    }
    c.assert_zero(p[ELEM_END] * (l[REM] - m));
    c.assert_zero((absorb - p[ELEM_END]) * (l[REM] - m - n[REM] * BabyBear::new(8)));
    let nonlast = p[ABSORB_NONLAST];
    for j in 0..WIDTH {
        c.assert_zero(nonlast * (n[CARRY + j] - out(j)));
    }
    c.assert_zero(nonlast * (n[CS] - l[CS] - p[MSG_ABSORB] * (k(W as u32 - 1) - m)));
    for j in 0..HASH_LEN {
        if j + 1 < HASH_LEN {
            c.assert_zero(p[SHIFT_E] * (n[E + j] - l[E + j + 1]));
        }
        c.assert_zero((nonlast - p[SHIFT_E]) * (n[E + j] - l[E + j]));
    }
    c.assert_zero(p[CS_ABSORB_NONLAST] * (n[CR] - l[CR] + p[CWEIGHT] * d));
    c.assert_zero(p[LAST_ABSORB] * (l[CR] - p[CWEIGHT] * d));
    for (j, &r) in root.into_iter().flatten().enumerate() {
        c.assert_zero(p[LAST_ABSORB] * (out(j) - r));
    }
    c.assert_zero(p[CS_START] * (l[CR] - l[CS]));

    // Element start: e = rem + 2^24 h canonically.
    let start = p[ELEM_START];
    let mut h = k(0);
    let mut g = k(0);
    for b in 0..7 {
        let (hb, gb) = (l[HBITS + b], l[GBITS + b]);
        c.assert_zero(start * hb * (hb - BabyBear::ONE));
        c.assert_zero(start * gb * (gb - BabyBear::ONE));
        h = h + hb * BabyBear::new(1 << b);
        g = g + gb * BabyBear::new(1 << b);
    }
    c.assert_zero(start * (l[E] - l[REM] - h * BabyBear::new(1 << 24)));
    let w = k(H_MAX) - h;
    c.assert_zero(start * (w - g));
    let z = k(1) - w * l[INV];
    c.assert_zero(start * w * z);
    c.assert_zero(start * l[REM] * z);
}

/// Standalone verification of one signature. Public values:
/// `msg (8) | param (5) | root (8)`.
pub struct WotsAir;

/// Public values for [`WotsAir`].
pub fn public_values(pk: &PublicKey, msg: &[BabyBear; HASH_LEN]) -> Vec<BabyBear> {
    msg.iter()
        .chain(&pk.param)
        .chain(&pk.root)
        .copied()
        .collect()
}

impl Air for WotsAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn max_degree(&self) -> usize {
        4
    }

    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
        let pub_ = c.public();
        let (msg, rest) = pub_.split_at(HASH_LEN);
        let (param, root) = rest.split_at(PARAM_LEN);
        let l = c.local();
        let n = c.next();
        for j in 0..HASH_LEN {
            c.assert_zero_first_row(l[MSG + j] - msg[j]);
            c.assert_zero_transition(n[MSG + j] - l[MSG + j]);
        }
        for j in 0..PARAM_LEN {
            c.assert_zero_first_row(l[PARAM + j] - param[j]);
            c.assert_zero_transition(n[PARAM + j] - l[PARAM + j]);
        }
        eval(c, Some(root));
    }

    fn preprocessed_width(&self) -> usize {
        PRE_WIDTH
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
        let mut m = RowMajorMatrix::zeros(1 << LOG_HEIGHT, PRE_WIDTH);
        for (i, row) in preprocessed_rows().iter().enumerate() {
            m.row_mut(i).copy_from_slice(row);
        }
        Some(m)
    }
}

/// The trace for [`WotsAir`] verifying `sig` on `msg` under `pk`. An invalid
/// signature gives a trace that fails the constraints.
pub fn trace(pk: &PublicKey, msg: &[BabyBear; HASH_LEN], sig: &Signature) -> RowMajorMatrix {
    trace_from(pk, msg, &honest_witness(pk, msg, sig))
}

/// Writes one verification's rows starting at `start`, including the `MSG`
/// and `PARAM` columns on those rows.
pub(crate) fn fill(
    m: &mut RowMajorMatrix,
    start: usize,
    pk: &PublicKey,
    msg: &[BabyBear; HASH_LEN],
    sig: &Signature,
) {
    fill_region(m, start, pk, msg, &honest_witness(pk, msg, sig));
}

fn honest_witness(pk: &PublicKey, msg: &[BabyBear; HASH_LEN], sig: &Signature) -> Witness {
    let e = message_digest(pk, msg, &sig.rand);
    Witness {
        rand: sig.rand,
        mdigits: core::array::from_fn(|i| {
            (e[i / DIGITS_PER_ELEM].as_u32() >> (3 * (i % DIGITS_PER_ELEM))) & 7
        }),
        hs: e.map(|x| x.as_u32() >> 24),
        digits: wots::encode(&pk.param, &sig.rand, msg),
        starts: sig.chains,
    }
}

/// The digest whose elements' low 24 bits are the message digits.
fn message_digest(
    pk: &PublicKey,
    msg: &[BabyBear; HASH_LEN],
    rand: &[BabyBear; RAND_LEN],
) -> [BabyBear; HASH_LEN] {
    crate::hash::sponge(domain::WOTS_MESSAGE, &pk.param, &[rand, msg])
}

/// Everything the prover chooses. Honest values come from [`trace`]; tests
/// build dishonest ones to check the constraints reject them.
struct Witness {
    rand: [BabyBear; RAND_LEN],
    /// The digest digits as claimed: 8 per element, low bits first.
    mdigits: [u32; NUM_CHAINS],
    /// The claimed part of each digest element above the low 24 bits.
    hs: [u32; HASH_LEN],
    /// The digit each chain is walked from.
    digits: [u8; NUM_CHAINS],
    /// Chain values at those digits.
    starts: [[BabyBear; HASH_LEN]; NUM_CHAINS],
}

fn trace_from(pk: &PublicKey, msg: &[BabyBear; HASH_LEN], w: &Witness) -> RowMajorMatrix {
    let mut m = RowMajorMatrix::zeros(1 << LOG_HEIGHT, NUM_COLS);
    fill_region(&mut m, 0, pk, msg, w);
    // Padding rows: permutations of zero, same globals.
    for r in NUM_ROWS..m.height() {
        let row = m.row_mut(r);
        poseidon2_air::generate([BabyBear::ZERO; WIDTH], row);
        row[PARAM..PARAM + PARAM_LEN].copy_from_slice(&pk.param);
        row[MSG..MSG + HASH_LEN].copy_from_slice(msg);
    }
    m
}

fn fill_region(
    m: &mut RowMajorMatrix,
    start: usize,
    pk: &PublicKey,
    msg: &[BabyBear; HASH_LEN],
    w: &Witness,
) {
    // Message sponge.
    let mut input = [BabyBear::ZERO; WIDTH];
    input[..RAND_LEN].copy_from_slice(&w.rand);
    input[RAND_LEN] = msg[0];
    input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&pk.param);
    input[13] = BabyBear::new(domain::WOTS_MESSAGE);
    input[14] = BabyBear::new(MSG_LEN as u32);
    let out0 = poseidon2_air::generate(input, m.row_mut(start));
    let mut input = out0;
    for j in 0..HASH_LEN - 1 {
        input[j] += msg[j + 1];
    }
    let digest_state = poseidon2_air::generate(input, m.row_mut(start + 1));
    let e: [BabyBear; HASH_LEN] = core::array::from_fn(|j| digest_state[j]);

    let digits = w.digits;
    let mdigit = |i: usize| w.mdigits[i];
    // rem_i = sum of the element's digits from slot i up, base 8.
    let rem_of = |i: usize| {
        let end = (i / DIGITS_PER_ELEM + 1) * DIGITS_PER_ELEM;
        (i..end).rev().fold(0, |acc, j| acc * 8 + w.mdigits[j])
    };

    let mut carry = [BabyBear::ZERO; WIDTH];
    carry[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&pk.param);
    carry[13] = BabyBear::new(domain::WOTS_PUBLIC_KEY);
    carry[14] = BabyBear::new(PK_LEN as u32);
    let mut cs = 0u32;
    let mut cr = 0u32;

    for (i, &d) in digits.iter().enumerate() {
        let base = start + 2 + i * ROWS_PER_CHAIN;
        let d = d as usize;
        let k_elem = i / DIGITS_PER_ELEM;
        let e_rot: [BabyBear; HASH_LEN] =
            core::array::from_fn(|j| e.get(k_elem + j).copied().unwrap_or(BabyBear::ZERO));
        if i == MSG_CHAINS {
            cr = cs;
        }
        let common = |row: &mut [BabyBear], ia: usize| {
            row[CARRY..CARRY + WIDTH].copy_from_slice(&carry);
            row[E..E + HASH_LEN].copy_from_slice(&e_rot);
            for b in 0..3 {
                row[MBITS + b] = BabyBear::new((mdigit(i) >> b) & 1);
                row[CBITS + b] = BabyBear::new((d as u32 >> b) & 1);
            }
            row[IA] = BabyBear::new(ia as u32);
            row[REM] = BabyBear::new(rem_of(i));
            row[CS] = BabyBear::new(cs);
            row[CR] = BabyBear::new(cr);
        };

        let mut v = w.starts[i];
        let mut ia = 0;
        for t in 0..STEPS {
            let row = m.row_mut(base + t);
            let active = t >= d;
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(&v);
            input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&pk.param);
            input[13] = BabyBear::new(domain::WOTS_CHAIN);
            input[14] = BabyBear::new(i as u32);
            input[15] = BabyBear::new(t as u32);
            let out = poseidon2_air::generate(input, row);
            row[VAL..VAL + HASH_LEN].copy_from_slice(&v);
            row[ACTIVE] = BabyBear::from(active);
            common(row, ia);
            if t == 0 && i % DIGITS_PER_ELEM == 0 {
                let h = w.hs[k_elem];
                let g = H_MAX.wrapping_sub(h);
                for b in 0..7 {
                    row[HBITS + b] = BabyBear::new((h >> b) & 1);
                    row[GBITS + b] = BabyBear::new((g >> b) & 1);
                }
                row[INV] = BabyBear::new(g).inverse().unwrap_or(BabyBear::ZERO);
            }
            if active {
                for j in 0..HASH_LEN {
                    v[j] += out[j];
                }
            } else {
                ia += 1;
            }
        }

        let row = m.row_mut(base + STEPS);
        let mut input = carry;
        for j in 0..HASH_LEN {
            input[j] += v[j];
        }
        let out = poseidon2_air::generate(input, row);
        row[VAL..VAL + HASH_LEN].copy_from_slice(&v);
        common(row, ia);
        carry = out;
        if i < MSG_CHAINS {
            cs += W as u32 - 1 - mdigit(i);
        } else {
            cr = cr.wrapping_sub((d as u32) << (3 * (NUM_CHAINS - 1 - i)));
        }
    }

    for r in start..start + NUM_ROWS {
        let row = m.row_mut(r);
        row[PARAM..PARAM + PARAM_LEN].copy_from_slice(&pk.param);
        row[MSG..MSG + HASH_LEN].copy_from_slice(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::fri::FriParams;
    use crate::stark;
    use crate::wots::{SecretKey, keygen};

    fn key(n: u32) -> (SecretKey, PublicKey) {
        keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 77 + i as u32 + 1
        })))
    }

    fn msg(n: u32) -> [BabyBear; HASH_LEN] {
        BabyBear::new_array(core::array::from_fn(|i| n * 1000 + i as u32))
    }

    fn signed(n: u32) -> (PublicKey, [BabyBear; HASH_LEN], Signature) {
        let (sk, pk) = key(n);
        let m = msg(n);
        let sig = sk.sign(&m);
        assert!(wots::verify(&pk, &m, &sig));
        (pk, m, sig)
    }

    #[test]
    fn layout() {
        assert_eq!(NUM_ROWS, 514);
        assert_eq!(LOG_HEIGHT, 10);
        assert_eq!(NUM_COLS, poseidon2_air::NUM_COLS + 71);
    }

    #[test]
    fn valid_signatures_satisfy_constraints() {
        for n in 0..6 {
            let (pk, m, sig) = signed(n);
            let t = trace(&pk, &m, &sig);
            assert_eq!(
                check_constraints(&WotsAir, &t, &public_values(&pk, &m)),
                Ok(()),
                "key {n}"
            );
        }
    }

    #[test]
    fn invalid_signatures_fail() {
        let (pk, m, sig) = signed(1);
        let public = public_values(&pk, &m);

        let mut bad = sig.clone();
        bad.chains[10][0] += BabyBear::ONE;
        assert!(!wots::verify(&pk, &m, &bad));
        assert!(check_constraints(&WotsAir, &trace(&pk, &m, &bad), &public).is_err());

        let mut bad = sig.clone();
        bad.rand[0] += BabyBear::ONE;
        assert!(check_constraints(&WotsAir, &trace(&pk, &m, &bad), &public).is_err());

        // Valid signature, different message or key.
        let other = public_values(&pk, &msg(2));
        assert!(check_constraints(&WotsAir, &trace(&pk, &m, &sig), &other).is_err());
        let (_, pk2) = key(9);
        let other = public_values(&pk2, &m);
        assert!(check_constraints(&WotsAir, &trace(&pk, &m, &sig), &other).is_err());
    }

    #[test]
    fn tampered_witness_columns_fail() {
        let (pk, m, sig) = signed(3);
        let public = public_values(&pk, &m);
        let t = trace(&pk, &m, &sig);
        let row_of = |chain: usize, t: usize| 2 + chain * ROWS_PER_CHAIN + t;
        // Flip an activity flag, a digit bit, the checksum, a remainder and an
        // element-decomposition bit.
        let cases = [
            (row_of(5, 6), ACTIVE),
            (row_of(5, STEPS), CBITS),
            (row_of(62, 0), CS),
            (row_of(9, 0), REM),
            (row_of(16, 0), HBITS),
            (row_of(20, STEPS), MBITS + 1),
        ];
        for (row, col) in cases {
            let mut bad = t.clone();
            bad.row_mut(row)[col] += BabyBear::ONE;
            assert!(
                check_constraints(&WotsAir, &bad, &public).is_err(),
                "row {row}, col {col}"
            );
        }
    }

    /// A witness for `digest_digits` (and claimed high parts `hs`) that is
    /// otherwise fully consistent: checksum and chain digits follow from the
    /// digits, and chain values come from the secret key, so every chain end
    /// and the root are right.
    fn forged(
        sk: &SecretKey,
        rand: [BabyBear; RAND_LEN],
        mdigits: [u32; NUM_CHAINS],
        hs: [u32; HASH_LEN],
    ) -> Witness {
        let checksum: u32 = mdigits[..MSG_CHAINS].iter().map(|&d| 7 - d).sum();
        let digits: [u8; NUM_CHAINS] = core::array::from_fn(|i| {
            if i < MSG_CHAINS {
                mdigits[i] as u8
            } else {
                ((checksum >> (3 * (NUM_CHAINS - 1 - i))) & 7) as u8
            }
        });
        Witness {
            rand,
            mdigits,
            hs,
            digits,
            starts: core::array::from_fn(|i| sk.chain_value(i, digits[i] as usize)),
        }
    }

    fn honest_digits(e: &[BabyBear; HASH_LEN]) -> ([u32; NUM_CHAINS], [u32; HASH_LEN]) {
        (
            core::array::from_fn(|i| (e[i / 8].as_u32() >> (3 * (i % 8))) & 7),
            e.map(|x| x.as_u32() >> 24),
        )
    }

    #[test]
    fn forged_witness_helper_is_sound_when_honest() {
        // Positive control: the helper with honest digits satisfies the AIR.
        let (sk, pk) = key(7);
        let m = msg(7);
        let rand = [BabyBear::new(3); RAND_LEN];
        let e = message_digest(&pk, &m, &rand);
        let (md, hs) = honest_digits(&e);
        let t = trace_from(&pk, &m, &forged(&sk, rand, md, hs));
        assert_eq!(
            check_constraints(&WotsAir, &t, &public_values(&pk, &m)),
            Ok(())
        );
    }

    #[test]
    fn rejects_digits_that_differ_from_the_digest() {
        let (sk, pk) = key(7);
        let m = msg(7);
        let rand = [BabyBear::new(3); RAND_LEN];
        let (mut md, hs) = honest_digits(&message_digest(&pk, &m, &rand));
        md[0] ^= 1;
        let t = trace_from(&pk, &m, &forged(&sk, rand, md, hs));
        // Fails exactly at element 0's decomposition check, chain 0's first row.
        assert_eq!(
            check_constraints(&WotsAir, &t, &public_values(&pk, &m)),
            Err(2)
        );
    }

    #[test]
    fn rejects_non_canonical_digest_decomposition() {
        // Find rand with digest element 0 below 2^24, so e + p =
        // (rem + 1) + 2^24 * 120 is an alternative decomposition.
        let (sk, pk) = key(8);
        let m = msg(8);
        let (rand, e) = (0..10_000)
            .map(|x| {
                let rand = [BabyBear::new(x); RAND_LEN];
                (rand, message_digest(&pk, &m, &rand))
            })
            .find(|(_, e)| e[0].as_u32() < (1 << 24) - 1)
            .expect("about 1 in 120 rands works");
        let (mut md, mut hs) = honest_digits(&e);
        let alt = e[0].as_u32() + 1;
        for (j, d) in md[..8].iter_mut().enumerate() {
            *d = (alt >> (3 * j)) & 7;
        }
        hs[0] = H_MAX;
        let t = trace_from(&pk, &m, &forged(&sk, rand, md, hs));
        assert_eq!(
            check_constraints(&WotsAir, &t, &public_values(&pk, &m)),
            Err(2)
        );
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let (pk, m, sig) = signed(4);
        let public = public_values(&pk, &m);
        let (pkey, vk) = stark::setup(&params, &WotsAir);
        let proof = stark::prove(&params, &pkey, &WotsAir, trace(&pk, &m, &sig), &public);
        assert_eq!(
            stark::verify(&params, &vk, &WotsAir, &public, &proof),
            Ok(())
        );

        let wrong = public_values(&pk, &msg(5));
        assert!(stark::verify(&params, &vk, &WotsAir, &wrong, &proof).is_err());
    }
}
