//! The validity sub-block circuit: proves one transaction with up to
//! `inputs` inputs and `outputs` outputs, revealing only multiset
//! fingerprints of its input commitments and output records, plus its fee
//! and mint.
//!
//! Rows (each one Poseidon2 permutation plus auxiliary columns; preprocessed
//! selectors give each row its role):
//!
//! - input slot: a WOTS region verifying the input's signature on the
//!   transaction digest ([`wots_air`]), then `PKH` (`H(pubkey)` from the
//!   region's root and parameter), `C1..C3` (the commitment) and `D` (absorb
//!   the commitment into the digest sponge);
//! - output slot: `C1..C3` with range-checked amount limbs, and `D`;
//! - a final row checking the balance, digest, counts and fingerprints.
//!
//! Each slot has an active flag, constant within the slot; active slots are
//! a prefix, inputs then outputs, in transaction order. Inactive slots add
//! nothing to the sums, fingerprints or digest. An inactive input slot still
//! verifies a signature: the prover fills it with a dummy key signing the
//! same digest, which keeps every constraint at degree 4.
//!
//! The digest is `sponge(TX_BODY, [n_in, n_out], commitments)`, as
//! [`TxBody::digest`]; every WOTS region checks its signature against it.
//! A transaction with no inputs is a block reward: it mints `sum(outputs)`
//! and pays no fee. Otherwise `sum(inputs) - sum(outputs) = fee >= 0` and
//! nothing is minted.
//!
//! Public values: `lambda (4) | r (4) | P_in (4) | P_out (4) | fee (4) |
//! mint (4)`, with `P_in = prod (r - sum_j lambda^j C_j)` over active input
//! commitments and `P_out` likewise over output commitments followed by
//! their nonce limbs (see [`input_element`], [`output_element`]).
//!
//! [`TxBody::digest`]: crate::transaction::TxBody::digest

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::matrix::RowMajorMatrix;
use crate::output::{self, AMOUNT_LIMBS, Commitment, NONCE_LIMBS, Nonce, OutputRecord};
use crate::poseidon2::WIDTH;
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::rng::Rng;
use crate::transaction::Transaction;
use crate::wots::{self, PARAM_LEN, PublicKey, Signature};
use crate::wots_air;

const COMMIT_LEN: usize = HASH_LEN + AMOUNT_LIMBS + NONCE_LIMBS;
const LIMB_BITS: usize = output::LIMB_BITS as usize;
const CARRY_BITS: usize = 6;
const CARRY_OFFSET: u32 = 1 << (CARRY_BITS - 1);
/// `x^4 = W` in the extension field.
const EXT_W: BabyBear = BabyBear::new(11);

/// Rows per input slot: the WOTS region, `PKH`, `C1..C3`, `D`.
const IN_ROWS: usize = wots_air::NUM_ROWS + 5;
const OUT_ROWS: usize = 4;

// Auxiliary columns after the permutation's and the WOTS region's.
const T: usize = wots_air::NUM_COLS;
const ACT: usize = T;
const AMT: usize = ACT + 1;
const NONCE: usize = AMT + AMOUNT_LIMBS;
const SUM: usize = NONCE + NONCE_LIMBS;
const DIG: usize = SUM + AMOUNT_LIMBS;
const PIN: usize = DIG + WIDTH;
const POUT: usize = PIN + 4;
const CNT_IN: usize = POUT + 4;
const CNT_OUT: usize = CNT_IN + 1;
const NIN: usize = CNT_OUT + 1;
const NOUT: usize = NIN + 1;
/// Output amount bits on `C2_OUT` rows; fee-or-mint bits on the final row.
const BITS: usize = NOUT + 1;
const CB: usize = BITS + AMOUNT_LIMBS * LIMB_BITS;
const ZINV: usize = CB + (AMOUNT_LIMBS - 1) * CARRY_BITS;
const INV2: usize = ZINV + 1;
pub const NUM_COLS: usize = INV2 + 1;

// Preprocessed columns after the WOTS region's.
const PW: usize = wots_air::PRE_WIDTH;
const PKH: usize = PW;
const C1: usize = PW + 1;
const C2_IN: usize = PW + 2;
const C2_OUT: usize = PW + 3;
const C3_IN: usize = PW + 4;
const C3_OUT: usize = PW + 5;
const D_IN: usize = PW + 6;
const D_OUT: usize = PW + 7;
const FIRST: usize = PW + 8;
const FINAL: usize = PW + 9;
/// Rows whose next row is in the same slot.
const SAME_SLOT: usize = PW + 10;
/// `D` rows of input (output) slots followed by another input (output) slot.
const NEXT_IN_SLOT: usize = PW + 11;
const NEXT_OUT_SLOT: usize = PW + 12;
pub const PRE_WIDTH: usize = PW + 13;

/// Number of public values.
pub const PUBLIC_LEN: usize = 24;

/// A sub-block's capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubBlockAir {
    pub inputs: usize,
    pub outputs: usize,
}

impl SubBlockAir {
    fn final_row(&self) -> usize {
        self.inputs * IN_ROWS + self.outputs * OUT_ROWS
    }

    pub fn height(&self) -> usize {
        (self.final_row() + 1).next_power_of_two()
    }

    fn input_base(&self, i: usize) -> usize {
        i * IN_ROWS
    }

    fn output_base(&self, o: usize) -> usize {
        self.inputs * IN_ROWS + o * OUT_ROWS
    }
}

/// `sum_j lambda^j x_j`.
fn fingerprint(lambda: BabyBear4, xs: impl IntoIterator<Item = BabyBear>) -> BabyBear4 {
    let mut acc = BabyBear4::ZERO;
    let mut pow = BabyBear4::ONE;
    for x in xs {
        acc += pow * x;
        pow *= lambda;
    }
    acc
}

/// The multiset element of an input commitment.
pub fn input_element(lambda: BabyBear4, c: &Commitment) -> Option<BabyBear4> {
    Some(fingerprint(lambda, c.to_field()?))
}

/// The multiset element of an output record: commitment then nonce limbs.
pub fn output_element(lambda: BabyBear4, record: &OutputRecord) -> Option<BabyBear4> {
    let c = record.commitment.to_field()?;
    Some(fingerprint(
        lambda,
        c.into_iter().chain(output::nonce_limbs(&record.nonce)),
    ))
}

/// `prod (r - e)` over `elements`.
pub fn product(r: BabyBear4, elements: impl IntoIterator<Item = BabyBear4>) -> BabyBear4 {
    elements
        .into_iter()
        .fold(BabyBear4::ONE, |acc, e| acc * (r - e))
}

fn ext_mul<V: Var>(a: [V; 4], b: [V; 4]) -> [V; 4] {
    let w = |x: V| x * EXT_W;
    [
        a[0] * b[0] + w(a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
        a[0] * b[1] + a[1] * b[0] + w(a[2] * b[3] + a[3] * b[2]),
        a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w(a[3] * b[3]),
        a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
    ]
}

impl Air for SubBlockAir {
    fn width(&self) -> usize {
        NUM_COLS
    }

    fn max_degree(&self) -> usize {
        4
    }

    fn preprocessed_width(&self) -> usize {
        PRE_WIDTH
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix> {
        let one = BabyBear::ONE;
        let mut m = RowMajorMatrix::zeros(self.height(), PRE_WIDTH);
        let wots = wots_air::preprocessed_rows();
        for i in 0..self.inputs {
            let base = self.input_base(i);
            for (r, row) in wots.iter().enumerate() {
                m.row_mut(base + r)[..PW].copy_from_slice(row);
            }
            let tail = base + wots_air::NUM_ROWS;
            for (offset, col) in [(0, PKH), (1, C1), (2, C2_IN), (3, C3_IN), (4, D_IN)] {
                m.row_mut(tail + offset)[col] = one;
            }
            for r in base..base + IN_ROWS - 1 {
                m.row_mut(r)[SAME_SLOT] = one;
            }
            if i + 1 < self.inputs {
                m.row_mut(base + IN_ROWS - 1)[NEXT_IN_SLOT] = one;
            }
        }
        for o in 0..self.outputs {
            let base = self.output_base(o);
            for (offset, col) in [(0, C1), (1, C2_OUT), (2, C3_OUT), (3, D_OUT)] {
                m.row_mut(base + offset)[col] = one;
            }
            for r in base..base + OUT_ROWS - 1 {
                m.row_mut(r)[SAME_SLOT] = one;
            }
            if o + 1 < self.outputs {
                m.row_mut(base + OUT_ROWS - 1)[NEXT_OUT_SLOT] = one;
            }
        }
        m.row_mut(0)[FIRST] = one;
        m.row_mut(self.final_row())[FINAL] = one;
        Some(m)
    }

    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
        let (l, n, p) = (c.local(), c.next(), c.preprocessed_local());
        let public = c.public();
        let k = |x: u32| V::from(BabyBear::new(x));
        let zero = k(0);
        let out = |j: usize| l[OUTPUT_OFFSET + j];
        let ext = |col: usize, row: &[V]| -> [V; 4] { core::array::from_fn(|b| row[col + b]) };
        let pub4 = |at: usize| -> [V; 4] { core::array::from_fn(|b| public[at + b]) };
        let (lambda, r) = (pub4(0), pub4(4));
        let limb = |bits: usize, count: usize| {
            (0..count).fold(zero, |acc, b| acc + l[bits + b] * BabyBear::new(1 << b))
        };
        let act = l[ACT];

        poseidon2_air::eval(c, l);
        wots_air::eval(c, None);

        // Globals and per-slot constants.
        for col in (wots_air::MSG..wots_air::MSG + HASH_LEN).chain([NIN, NOUT]) {
            c.assert_zero_transition(n[col] - l[col]);
        }
        for j in 0..PARAM_LEN {
            let col = wots_air::PARAM + j;
            c.assert_zero(p[SAME_SLOT] * (n[col] - l[col]));
        }
        c.assert_zero(act * (act - BabyBear::ONE));
        c.assert_zero(p[SAME_SLOT] * (n[ACT] - act));
        c.assert_zero((p[NEXT_IN_SLOT] + p[NEXT_OUT_SLOT]) * n[ACT] * (k(1) - act));

        // The WOTS root feeds PKH: input root | param | PUBLIC_KEY_HASH | 0 | 0.
        let last_absorb = p[wots_air::LAST_ABSORB];
        for (j, &x) in n[..HASH_LEN].iter().enumerate() {
            c.assert_zero(last_absorb * (x - out(j)));
        }
        let pkh = p[PKH];
        for j in 0..PARAM_LEN {
            c.assert_zero(pkh * (l[HASH_LEN + j] - l[wots_air::PARAM + j]));
        }
        c.assert_zero(pkh * (l[13] - k(domain::PUBLIC_KEY_HASH)));
        c.assert_zero(pkh * l[14]);
        c.assert_zero(pkh * l[15]);
        for j in 0..HASH_LEN {
            c.assert_zero(pkh * (n[j] - out(j) - l[j]));
        }

        // Commitment sponge, as `output::commit`.
        let c1 = p[C1];
        for &x in &l[HASH_LEN..HASH_LEN + PARAM_LEN] {
            c.assert_zero(c1 * x);
        }
        c.assert_zero(c1 * (l[13] - k(domain::OUTPUT)));
        c.assert_zero(c1 * (l[14] - k(COMMIT_LEN as u32)));
        c.assert_zero(c1 * l[15]);
        for j in 0..WIDTH {
            let absorbed = match j {
                0..4 => n[AMT + j],
                4..8 => n[NONCE + j - 4],
                _ => zero,
            };
            c.assert_zero(c1 * (n[j] - out(j) - absorbed));
        }
        let c2 = p[C2_IN] + p[C2_OUT];
        for j in 0..NONCE_LIMBS {
            c.assert_zero(c2 * (n[NONCE + j] - l[NONCE + j]));
        }
        for j in 0..WIDTH {
            let absorbed = if j < 4 { l[NONCE + 4 + j] } else { zero };
            c.assert_zero(c2 * (n[j] - out(j) - absorbed));
        }

        // Output amounts are 16-bit limbs (and the final row's bits are bits).
        let bits_rows = p[C2_OUT] + p[FINAL];
        for b in 0..AMOUNT_LIMBS * LIMB_BITS {
            c.assert_zero(bits_rows * l[BITS + b] * (l[BITS + b] - BabyBear::ONE));
        }
        for j in 0..AMOUNT_LIMBS {
            c.assert_zero(p[C2_OUT] * (l[AMT + j] - limb(BITS + j * LIMB_BITS, LIMB_BITS)));
        }

        // Running sums: active inputs add, active outputs subtract.
        for j in 0..AMOUNT_LIMBS {
            let d = n[SUM + j] - l[SUM + j];
            c.assert_zero(p[C2_IN] * (d - act * l[AMT + j]));
            c.assert_zero(p[C2_OUT] * (d + act * l[AMT + j]));
            c.assert_zero_transition((k(1) - c2) * d);
        }

        // Fingerprints at C3 rows: next = current * (r - element), if active.
        let commitment: [V; HASH_LEN] = core::array::from_fn(out);
        let mut powers = [[zero; 4]; HASH_LEN + NONCE_LIMBS];
        powers[0][0] = k(1);
        for j in 1..powers.len() {
            powers[j] = ext_mul(powers[j - 1], lambda);
        }
        let element = |values: &mut dyn Iterator<Item = V>| -> [V; 4] {
            values.zip(&powers).fold([zero; 4], |acc, (x, pw)| {
                core::array::from_fn(|b| acc[b] + x * pw[b])
            })
        };
        let factor = |e: [V; 4]| -> [V; 4] {
            core::array::from_fn(|b| {
                let one = if b == 0 { k(1) } else { zero };
                one - act * one + act * (r[b] - e[b])
            })
        };
        let e_in = element(&mut commitment.iter().copied());
        let e_out = element(
            &mut commitment
                .iter()
                .copied()
                .chain(l[NONCE..NONCE + NONCE_LIMBS].iter().copied()),
        );
        for (sel, col, e) in [(p[C3_IN], PIN, e_in), (p[C3_OUT], POUT, e_out)] {
            let next = ext_mul(ext(col, l), factor(e));
            for b in 0..4 {
                c.assert_zero(sel * (n[col + b] - next[b]));
                c.assert_zero_transition((k(1) - sel) * (n[col + b] - l[col + b]));
            }
        }

        // Digest: C3 feeds D's absorption; D updates the carried state if active.
        let c3 = p[C3_IN] + p[C3_OUT];
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { out(j) } else { zero };
            c.assert_zero(c3 * (n[j] - n[DIG + j] - absorbed));
        }
        let d_row = p[D_IN] + p[D_OUT];
        for j in 0..WIDTH {
            let col = DIG + j;
            c.assert_zero(d_row * (n[col] - act * out(j) - (k(1) - act) * l[col]));
            c.assert_zero_transition((k(1) - d_row) * (n[col] - l[col]));
        }

        // Counts.
        for (sel, col) in [(p[D_IN], CNT_IN), (p[D_OUT], CNT_OUT)] {
            c.assert_zero(sel * (n[col] - l[col] - act));
            c.assert_zero_transition((k(1) - sel) * (n[col] - l[col]));
        }

        // First row: initial carried state.
        let first = p[FIRST];
        for col in (SUM..SUM + AMOUNT_LIMBS).chain([CNT_IN, CNT_OUT]) {
            c.assert_zero(first * l[col]);
        }
        for col in [PIN, POUT] {
            c.assert_zero(first * (l[col] - k(1)));
            for b in 1..4 {
                c.assert_zero(first * l[col + b]);
            }
        }
        for j in 0..WIDTH {
            let expected = match j {
                8 => l[NIN],
                9 => l[NOUT],
                13 => k(domain::TX_BODY),
                14 => (l[NIN] + l[NOUT]) * BabyBear::new(HASH_LEN as u32),
                _ => zero,
            };
            c.assert_zero(first * (l[DIG + j] - expected));
        }

        // Final row.
        let fin = p[FINAL];
        c.assert_zero(fin * (l[CNT_IN] - l[NIN]));
        c.assert_zero(fin * (l[CNT_OUT] - l[NOUT]));
        c.assert_zero(fin * ((l[NIN] + l[NOUT]) * l[INV2] - k(1)));
        for j in 0..HASH_LEN {
            c.assert_zero(fin * (l[DIG + j] - l[wots_air::MSG + j]));
        }
        for b in 0..4 {
            c.assert_zero(fin * (l[PIN + b] - public[8 + b]));
            c.assert_zero(fin * (l[POUT + b] - public[12 + b]));
        }
        // z = [n_in == 0]: a reward mints, anything else pays a fee.
        let z = k(1) - l[NIN] * l[ZINV];
        c.assert_zero(fin * l[NIN] * z);
        for b in 0..(AMOUNT_LIMBS - 1) * CARRY_BITS {
            c.assert_zero(fin * l[CB + b] * (l[CB + b] - BabyBear::ONE));
        }
        let carry = |j: usize| limb(CB + j * CARRY_BITS, CARRY_BITS) - k(CARRY_OFFSET);
        let sign = k(1) - z * BabyBear::new(2);
        for j in 0..AMOUNT_LIMBS {
            let x = limb(BITS + j * LIMB_BITS, LIMB_BITS);
            let carry_in = if j == 0 { zero } else { carry(j - 1) };
            let carry_out = if j + 1 < AMOUNT_LIMBS { carry(j) } else { zero };
            c.assert_zero(
                fin * (sign * l[SUM + j] + carry_in
                    - x
                    - carry_out * BabyBear::new(1 << LIMB_BITS)),
            );
            c.assert_zero(fin * (public[16 + j] - (k(1) - z) * x));
            c.assert_zero(fin * (public[20 + j] - z * x));
        }
    }
}

/// One input slot's witness.
struct InSlot {
    pubkey: PublicKey,
    sig: Signature,
    amount: [BabyBear; AMOUNT_LIMBS],
    nonce: [BabyBear; NONCE_LIMBS],
    active: bool,
}

/// One output slot's witness.
struct OutSlot {
    hash: Hash,
    amount: [BabyBear; AMOUNT_LIMBS],
    nonce: [BabyBear; NONCE_LIMBS],
    active: bool,
}

/// Values carried down the rows.
struct Carried {
    sum: [i64; AMOUNT_LIMBS],
    dig: [BabyBear; WIDTH],
    pin: BabyBear4,
    pout: BabyBear4,
    cnt_in: u32,
    cnt_out: u32,
}

impl Carried {
    fn write(&self, row: &mut [BabyBear]) {
        for (cell, &s) in row[SUM..SUM + AMOUNT_LIMBS].iter_mut().zip(&self.sum) {
            *cell = signed(s);
        }
        row[DIG..DIG + WIDTH].copy_from_slice(&self.dig);
        row[PIN..PIN + 4].copy_from_slice(&self.pin.0);
        row[POUT..POUT + 4].copy_from_slice(&self.pout.0);
        row[CNT_IN] = BabyBear::new(self.cnt_in);
        row[CNT_OUT] = BabyBear::new(self.cnt_out);
    }
}

fn signed(x: i64) -> BabyBear {
    let v = BabyBear::new(x.unsigned_abs() as u32);
    if x < 0 { -v } else { v }
}

/// A limb's value as a signed integer: limbs above 2^16 only appear in
/// adversarial tests, as small negative field elements.
fn limb_value(x: BabyBear) -> i64 {
    let v = x.as_u32() as i64;
    if v > i32::MAX as i64 {
        v - crate::babybear::P as i64
    } else {
        v
    }
}

impl SubBlockAir {
    /// The trace and public values proving `tx`, which must be validly
    /// signed. Inactive input slots get dummy keys drawn from `rng`.
    ///
    /// # Panics
    /// If `tx` has more inputs or outputs than the capacity, or a
    /// non-canonical hash.
    pub fn trace(
        &self,
        tx: &Transaction,
        lambda: BabyBear4,
        r: BabyBear4,
        rng: &mut Rng,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        let body = &tx.body;
        assert!(body.inputs.len() <= self.inputs && body.outputs.len() <= self.outputs);
        let msg = body.digest().expect("canonical hashes");
        let ins = (0..self.inputs)
            .map(|i| match body.inputs.get(i) {
                Some(input) => InSlot {
                    pubkey: input.pubkey.clone(),
                    sig: tx.signatures[i].clone(),
                    amount: output::amount_limbs(input.amount),
                    nonce: output::nonce_limbs(&input.nonce),
                    active: true,
                },
                None => dummy_input(&msg, rng),
            })
            .collect::<Vec<_>>();
        let outs = (0..self.outputs)
            .map(|o| match body.outputs.get(o) {
                Some(out) => OutSlot {
                    hash: hash::from_bytes(&out.hash).expect("canonical output hash"),
                    amount: output::amount_limbs(out.amount),
                    nonce: output::nonce_limbs(&out.nonce),
                    active: true,
                },
                None => OutSlot {
                    hash: [BabyBear::ZERO; HASH_LEN],
                    amount: [BabyBear::ZERO; AMOUNT_LIMBS],
                    nonce: [BabyBear::ZERO; NONCE_LIMBS],
                    active: false,
                },
            })
            .collect::<Vec<_>>();
        self.trace_from(&msg, &ins, &outs, lambda, r)
    }

    fn trace_from(
        &self,
        msg: &Hash,
        ins: &[InSlot],
        outs: &[OutSlot],
        lambda: BabyBear4,
        r: BabyBear4,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let nin = ins.iter().filter(|s| s.active).count() as u32;
        let nout = outs.iter().filter(|s| s.active).count() as u32;
        let mut dig = [BabyBear::ZERO; WIDTH];
        dig[8] = BabyBear::new(nin);
        dig[9] = BabyBear::new(nout);
        dig[13] = BabyBear::new(domain::TX_BODY);
        dig[14] = BabyBear::new(HASH_LEN as u32 * (nin + nout));
        let mut st = Carried {
            sum: [0; AMOUNT_LIMBS],
            dig,
            pin: BabyBear4::ONE,
            pout: BabyBear4::ONE,
            cnt_in: 0,
            cnt_out: 0,
        };
        let powers: Vec<BabyBear4> = (0..HASH_LEN + NONCE_LIMBS)
            .scan(BabyBear4::ONE, |pw, _| {
                let v = *pw;
                *pw *= lambda;
                Some(v)
            })
            .collect();

        // C1..C3 and D for one slot starting at `row`. Returns the commitment.
        let commit_rows = |m: &mut RowMajorMatrix,
                           st: &mut Carried,
                           row: usize,
                           hash_in: Hash,
                           amount: &[BabyBear; AMOUNT_LIMBS],
                           nonce: &[BabyBear; NONCE_LIMBS],
                           active: bool,
                           is_input: bool| {
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(&hash_in);
            input[13] = BabyBear::new(domain::OUTPUT);
            input[14] = BabyBear::new(COMMIT_LEN as u32);
            let out1 = poseidon2_air::generate(input, m.row_mut(row));
            st.write(m.row_mut(row));

            let mut input = out1;
            for j in 0..4 {
                input[j] += amount[j];
                input[4 + j] += nonce[j];
            }
            let out2 = poseidon2_air::generate(input, m.row_mut(row + 1));
            let c2 = m.row_mut(row + 1);
            c2[AMT..AMT + AMOUNT_LIMBS].copy_from_slice(amount);
            c2[NONCE..NONCE + NONCE_LIMBS].copy_from_slice(nonce);
            if !is_input {
                for (j, a) in amount.iter().enumerate() {
                    let v = a.as_u32();
                    for b in 0..LIMB_BITS {
                        c2[BITS + j * LIMB_BITS + b] = BabyBear::new((v >> b) & 1);
                    }
                }
            }
            st.write(c2);
            if active {
                let sign = if is_input { 1 } else { -1 };
                for (s, &a) in st.sum.iter_mut().zip(amount) {
                    *s += sign * limb_value(a);
                }
            }

            let mut input = out2;
            for (x, &nv) in input.iter_mut().zip(&nonce[4..]) {
                *x += nv;
            }
            let out3 = poseidon2_air::generate(input, m.row_mut(row + 2));
            m.row_mut(row + 2)[NONCE..NONCE + NONCE_LIMBS].copy_from_slice(nonce);
            st.write(m.row_mut(row + 2));
            let commitment: Hash = core::array::from_fn(|j| out3[j]);
            if active {
                let xs = commitment.iter().copied().chain(if is_input {
                    Vec::new()
                } else {
                    nonce.to_vec()
                });
                let e = xs
                    .zip(&powers)
                    .fold(BabyBear4::ZERO, |acc, (x, &pw)| acc + pw * x);
                let p = if is_input { &mut st.pin } else { &mut st.pout };
                *p *= r - e;
            }

            let mut input = st.dig;
            for (x, &cv) in input.iter_mut().zip(&commitment) {
                *x += cv;
            }
            let out_d = poseidon2_air::generate(input, m.row_mut(row + 3));
            st.write(m.row_mut(row + 3));
            if active {
                st.dig = out_d;
                if is_input {
                    st.cnt_in += 1;
                } else {
                    st.cnt_out += 1;
                }
            }
        };

        for (i, slot) in ins.iter().enumerate() {
            let base = self.input_base(i);
            wots_air::fill(&mut m, base, &slot.pubkey, msg, &slot.sig);
            for r in base..base + wots_air::NUM_ROWS {
                st.write(m.row_mut(r));
            }
            let pkh_row = base + wots_air::NUM_ROWS;
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(&slot.pubkey.root);
            input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&slot.pubkey.param);
            input[13] = BabyBear::new(domain::PUBLIC_KEY_HASH);
            let out = poseidon2_air::generate(input, m.row_mut(pkh_row));
            st.write(m.row_mut(pkh_row));
            let pkh: Hash = core::array::from_fn(|j| out[j] + input[j]);
            commit_rows(
                &mut m,
                &mut st,
                pkh_row + 1,
                pkh,
                &slot.amount,
                &slot.nonce,
                slot.active,
                true,
            );
            for r in base..base + IN_ROWS {
                let row = m.row_mut(r);
                row[ACT] = BabyBear::from(slot.active);
                row[wots_air::PARAM..wots_air::PARAM + PARAM_LEN]
                    .copy_from_slice(&slot.pubkey.param);
            }
        }
        for (o, slot) in outs.iter().enumerate() {
            let base = self.output_base(o);
            commit_rows(
                &mut m,
                &mut st,
                base,
                slot.hash,
                &slot.amount,
                &slot.nonce,
                slot.active,
                false,
            );
            for r in base..base + OUT_ROWS {
                m.row_mut(r)[ACT] = BabyBear::from(slot.active);
            }
        }

        // Final row: fee or mint, with carries.
        let fin = self.final_row();
        let reward = nin == 0;
        let value: i128 = st
            .sum
            .iter()
            .enumerate()
            .map(|(j, &s)| (s as i128) << (LIMB_BITS * j))
            .sum::<i128>()
            * if reward { -1 } else { 1 };
        let x = output::amount_limbs(value.clamp(0, u64::MAX as i128) as u64);
        {
            let row = m.row_mut(fin);
            let mut carry = 0i64;
            for j in 0..AMOUNT_LIMBS {
                let xv = x[j].as_u32();
                for b in 0..LIMB_BITS {
                    row[BITS + j * LIMB_BITS + b] = BabyBear::new((xv >> b) & 1);
                }
                if j + 1 < AMOUNT_LIMBS {
                    let s = if reward { -st.sum[j] } else { st.sum[j] };
                    carry = (s + carry - xv as i64).div_euclid(1 << LIMB_BITS);
                    let biased =
                        (carry + CARRY_OFFSET as i64).clamp(0, (1 << CARRY_BITS) - 1) as u32;
                    for b in 0..CARRY_BITS {
                        row[CB + j * CARRY_BITS + b] = BabyBear::new((biased >> b) & 1);
                    }
                }
            }
            row[ZINV] = BabyBear::new(nin).inverse().unwrap_or(BabyBear::ZERO);
            row[INV2] = BabyBear::new(nin + nout)
                .inverse()
                .unwrap_or(BabyBear::ZERO);
        }
        for r in fin..m.height() {
            st.write(m.row_mut(r));
            if r > fin {
                poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(r));
            }
        }
        poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(fin));
        for r in 0..m.height() {
            let row = m.row_mut(r);
            row[wots_air::MSG..wots_air::MSG + HASH_LEN].copy_from_slice(msg);
            row[NIN] = BabyBear::new(nin);
            row[NOUT] = BabyBear::new(nout);
        }

        let zero_limbs = [BabyBear::ZERO; AMOUNT_LIMBS];
        let (fee, mint) = if reward {
            (zero_limbs, x)
        } else {
            (x, zero_limbs)
        };
        let mut public = Vec::with_capacity(PUBLIC_LEN);
        for v in [lambda, r, st.pin, st.pout] {
            public.extend_from_slice(&v.0);
        }
        public.extend_from_slice(&fee);
        public.extend_from_slice(&mint);
        (m, public)
    }
}

/// An inactive input slot: a fresh key signing the transaction digest.
fn dummy_input(msg: &Hash, rng: &mut Rng) -> InSlot {
    let mut seed = [BabyBear::ZERO; wots::SEED_LEN];
    rng.fill(&mut seed);
    let (sk, pubkey) = wots::keygen(seed);
    InSlot {
        sig: sk.sign(msg),
        pubkey,
        amount: [BabyBear::ZERO; AMOUNT_LIMBS],
        nonce: output::nonce_limbs(&Nonce::default()),
        active: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::babybear::P;
    use crate::fri::FriParams;
    use crate::output::{Amount, NONCE_LEN, Output};
    use crate::stark;
    use crate::transaction::{Input, TxBody, pubkey_hash};
    use crate::wots::{SecretKey, keygen};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);

    fn key(n: u32) -> (SecretKey, PublicKey) {
        keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 31 + i as u32
        })))
    }

    fn input(n: u32, amount: Amount) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let nonce: Nonce = [n as u8 + 1; NONCE_LEN];
        let commitment = output::commit(&pubkey_hash(&pubkey), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                pubkey,
                amount,
                nonce,
            },
        )
    }

    fn out(n: u8, amount: Amount) -> Output {
        Output {
            hash: hash::to_bytes(&BabyBear::new_array(core::array::from_fn(|i| {
                n as u32 * 100 + i as u32
            }))),
            amount,
            nonce: [n; NONCE_LEN],
        }
    }

    fn tx(inputs: Vec<(SecretKey, Input)>, outputs: Vec<Output>) -> Transaction {
        let (keys, inputs) = inputs.into_iter().unzip();
        Transaction::sign(TxBody { inputs, outputs }, keys).unwrap()
    }

    /// Fingerprints computed natively from the transaction.
    fn expected(tx: &Transaction) -> (BabyBear4, BabyBear4) {
        let p_in = product(
            R,
            tx.body
                .inputs
                .iter()
                .map(|i| input_element(LAMBDA, &i.commitment).unwrap()),
        );
        let p_out = product(
            R,
            tx.body
                .outputs
                .iter()
                .map(|o| output_element(LAMBDA, &o.record().unwrap()).unwrap()),
        );
        (p_in, p_out)
    }

    fn check(air: &SubBlockAir, tx: &Transaction) -> (Result<(), usize>, Vec<BabyBear>) {
        let (t, public) = air.trace(tx, LAMBDA, R, &mut Rng::from_seed(1));
        (check_constraints(air, &t, &public), public)
    }

    fn ext_at(public: &[BabyBear], at: usize) -> BabyBear4 {
        BabyBear4(public[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn transfer_satisfies_constraints() {
        let t = tx(
            vec![input(1, 1 << 40), input(2, 70_000)],
            vec![out(1, (1 << 40) - 5), out(2, 65_536)],
        );
        assert_eq!(t.validate().unwrap().fee(), 4469);
        let air = SubBlockAir {
            inputs: 2,
            outputs: 3,
        };
        let (result, public) = check(&air, &t);
        assert_eq!(result, Ok(()));
        let (p_in, p_out) = expected(&t);
        assert_eq!(ext_at(&public, 8), p_in);
        assert_eq!(ext_at(&public, 12), p_out);
        assert_eq!(&public[16..20], &output::amount_limbs(4469));
        assert_eq!(&public[20..24], &output::amount_limbs(0));
    }

    #[test]
    fn partially_filled_slots() {
        // One of two input slots and one of three output slots active.
        let t = tx(vec![input(3, 500)], vec![out(1, 400)]);
        let air = SubBlockAir {
            inputs: 2,
            outputs: 3,
        };
        let (result, public) = check(&air, &t);
        assert_eq!(result, Ok(()));
        assert_eq!(ext_at(&public, 8), expected(&t).0);
        assert_eq!(&public[16..20], &output::amount_limbs(100));
    }

    #[test]
    fn reward_mints_and_pays_no_fee() {
        let t = tx(vec![], vec![out(1, 50), out(2, u64::MAX - 50)]);
        let air = SubBlockAir {
            inputs: 1,
            outputs: 2,
        };
        let (result, public) = check(&air, &t);
        assert_eq!(result, Ok(()));
        assert_eq!(ext_at(&public, 8), BabyBear4::ONE);
        assert_eq!(&public[16..20], &output::amount_limbs(0));
        assert_eq!(&public[20..24], &output::amount_limbs(u64::MAX));
    }

    #[test]
    fn rejects_overspend() {
        let t = tx(vec![input(1, 100)], vec![out(1, 101)]);
        let air = SubBlockAir {
            inputs: 1,
            outputs: 1,
        };
        assert!(check(&air, &t).0.is_err());
    }

    #[test]
    fn rejects_body_changed_after_signing() {
        let mut t = tx(vec![input(1, 100)], vec![out(1, 90)]);
        t.body.outputs[0].amount = 80;
        let air = SubBlockAir {
            inputs: 1,
            outputs: 1,
        };
        assert!(check(&air, &t).0.is_err());
    }

    #[test]
    fn rejects_wrong_public_values() {
        let t = tx(vec![input(1, 100)], vec![out(1, 90)]);
        let air = SubBlockAir {
            inputs: 1,
            outputs: 2,
        };
        let (trace, public) = air.trace(&t, LAMBDA, R, &mut Rng::from_seed(1));
        // P_in, P_out, fee, and a mint on a non-reward transaction.
        for at in [8, 13, 16, 20] {
            let mut bad = public.clone();
            bad[at] += BabyBear::ONE;
            assert!(
                check_constraints(&air, &trace, &bad).is_err(),
                "public {at}"
            );
        }
    }

    #[test]
    fn rejects_hiding_an_output() {
        // Marking a signed output inactive drops it from the digest, so the
        // signatures no longer verify.
        let t = tx(vec![input(1, 100)], vec![out(1, 40), out(2, 50)]);
        let air = SubBlockAir {
            inputs: 1,
            outputs: 2,
        };
        let (mut trace, public) = air.trace(&t, LAMBDA, R, &mut Rng::from_seed(1));
        let base = air.output_base(1);
        for r in base..base + OUT_ROWS {
            trace.row_mut(r)[ACT] = BabyBear::ZERO;
        }
        assert!(check_constraints(&air, &trace, &public).is_err());
    }

    #[test]
    fn rejects_negative_output_limb() {
        // An output limb of -5 (p - 5) would mint 5 units. Its commitment is
        // computed from those limbs and signed, so only the range check
        // stands in the way.
        let (sk, input) = input(1, 100);
        let hash = [BabyBear::ONE; HASH_LEN];
        let amount = [
            BabyBear::new(P - 5),
            BabyBear::ZERO,
            BabyBear::ZERO,
            BabyBear::ZERO,
        ];
        let nonce = [BabyBear::ZERO; NONCE_LIMBS];
        let c_out = hash::sponge(domain::OUTPUT, &[], &[&hash, &amount, &nonce]);
        let msg = hash::sponge(
            domain::TX_BODY,
            &[BabyBear::ONE, BabyBear::ONE],
            &[&input.commitment.to_field().unwrap(), &c_out],
        );
        let air = SubBlockAir {
            inputs: 1,
            outputs: 1,
        };
        let ins = [InSlot {
            sig: sk.sign(&msg),
            pubkey: input.pubkey.clone(),
            amount: output::amount_limbs(100),
            nonce: output::nonce_limbs(&input.nonce),
            active: true,
        }];
        let outs = [OutSlot {
            hash,
            amount,
            nonce,
            active: true,
        }];
        let (trace, public) = air.trace_from(&msg, &ins, &outs, LAMBDA, R);
        let err = check_constraints(&air, &trace, &public).unwrap_err();
        assert_eq!(err, air.output_base(0) + 1, "fails at the output's C2 row");
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let t = tx(vec![input(1, 100)], vec![out(1, 60), out(2, 30)]);
        let air = SubBlockAir {
            inputs: 1,
            outputs: 2,
        };
        let (trace, public) = air.trace(&t, LAMBDA, R, &mut Rng::from_seed(1));
        let (pk, vk) = stark::setup(&params, &air);
        let proof = stark::prove(&params, &pk, &air, trace, &public);
        assert_eq!(stark::verify(&params, &vk, &air, &public, &proof), Ok(()));
    }
}
