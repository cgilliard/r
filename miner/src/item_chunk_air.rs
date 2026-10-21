//! The item chunk circuit: proves a run of a block's transaction items,
//! revealing only multiset fingerprints, the net value and a blinded hash of
//! the transactions it finishes.
//!
//! A block's transactions form one stream of items: for each transaction a
//! start, its inputs, its outputs and an end. The stream is cut into chunks
//! of `items` slots, and a transaction may span several chunks: each chunk
//! publishes the open transaction's state where it starts and where it
//! stops ([`Boundary`]), and the aggregation chains them. Between
//! transactions the state is idle (all zero), which is how a block's stream
//! must start and end.
//!
//! Signatures are proved elsewhere ([`crate::sig_chunk_air`]): each input
//! here contributes `(pkh, its transaction's digest)` to a fingerprint
//! product `P_sig`, which the aggregation matches against the signature
//! chunks' (one signature per input).
//!
//! Rows: `items` slots of six rows (`R0..R5`), then a final row. Each item
//! is a start, input, output, end or padding (witness flags, constant in
//! the item):
//!
//! - start: `R0` checks the state is idle and starts the transaction's
//!   digest from its counts (the transaction's constants: claimed digest
//!   and counts, which may change only into a start or after an end);
//! - input: `R0..R2` hash its commitment from `pkh`, amount and nonce; `R3`
//!   absorbs the commitment and `R4` its position into the digest; `R0`
//!   also adds `(pkh, digest)` to `P_sig`;
//! - output: `R0..R2` hash its commitment (amount range-checked), `R3`
//!   absorbs it;
//! - end: `R0` checks the digest is the claimed one and the counts match,
//!   and absorbs it into the transaction-hash sponge; `R5` returns to idle.
//!
//! A transaction's digest is [`TxBody::digest`]. Transactions have no
//! balance rule of their own: the net value `sum(outputs) - sum(inputs)` is
//! summed, and only the block's total is constrained (to the subsidy). It
//! is published as [`output::net_limbs`].
//!
//! Public values: `lambda (4) | r (4) | P_in (4) | P_out (4) | P_sig (4) |
//! net (6) | tx hash (8) | entry (28) | exit (28)`, with `P_in` over inputs'
//! (commitment, position) ([`input_element`]), `P_out` over outputs'
//! (commitment, nonce) ([`output_element`]), `P_sig` over inputs' (pkh,
//! digest) ([`sig_element`]). The transaction hash ([`chunk_hash`]) is a
//! sponge over the digests of the transactions ending in the chunk, with a
//! random blinding as its initial capacity.
//!
//! [`TxBody::digest`]: crate::transaction::TxBody::digest

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::matrix::RowMajorMatrix;
use crate::output::{
    self, AMOUNT_LIMBS, Commitment, NET_LIMBS, NET_OFFSET, NONCE_LIMBS, OutputRecord,
};
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::transaction::Transaction;

const COMMIT_LEN: usize = HASH_LEN + AMOUNT_LIMBS + NONCE_LIMBS;
const LIMB_BITS: usize = output::LIMB_BITS as usize;
/// Net-value carries: bits per carry, enough for [`MAX_ITEMS`] items.
const CARRY_BITS: usize = 15;
const CARRY_OFFSET: u32 = 1 << (CARRY_BITS - 1);
/// The most items a chunk may have (the carries' range).
pub const MAX_ITEMS: usize = 1 << (CARRY_BITS - 2);
/// `x^4 = W` in the extension field.
const EXT_W: BabyBear = BabyBear::new(11);

/// Rows per item.
pub const ITEM_ROWS: usize = 6;
/// Elements of a boundary state: digest sponge, input and output counts,
/// claimed digest, declared input and output counts.
pub const BOUNDARY_LEN: usize = WIDTH + 2 + HASH_LEN + 2;

// Columns after the permutation's.
const P: usize = poseidon2_air::NUM_COLS;
/// Item kind flags, constant in an item.
const KSTART: usize = P;
const KIN: usize = KSTART + 1;
const KOUT: usize = KIN + 1;
const KEND: usize = KOUT + 1;
/// Item values, constant in an item.
const AMT: usize = KEND + 1;
const NONCE: usize = AMT + AMOUNT_LIMBS;
const POS: usize = NONCE + NONCE_LIMBS;
/// Per-transaction constants: claimed digest, input and output counts.
const MSG: usize = POS + 2;
const NIN: usize = MSG + HASH_LEN;
const NOUT: usize = NIN + 1;
/// Carried down the rows.
const SUM: usize = NOUT + 1;
const DIG: usize = SUM + AMOUNT_LIMBS;
const PIN: usize = DIG + WIDTH;
const POUT: usize = PIN + 4;
const PSIG: usize = POUT + 4;
const CNT_IN: usize = PSIG + 4;
const CNT_OUT: usize = CNT_IN + 1;
const TXS: usize = CNT_OUT + 1;
/// Output amount bits on `R1` rows; net value bits on the final row.
const BITS: usize = TXS + WIDTH;
const CB: usize = BITS + NET_LIMBS * LIMB_BITS;
pub const NUM_COLS: usize = CB + (NET_LIMBS - 1) * CARRY_BITS;

/// The boundary's columns, in [`Boundary`] order.
const BOUNDARY_COLS: [(usize, usize); 5] = [
    (DIG, WIDTH),
    (CNT_IN, 1),
    (CNT_OUT, 1),
    (MSG, HASH_LEN),
    (NIN, 2),
];

// Preprocessed columns: item rows `R0..R5`, rows followed by a row of the
// same item, every item row, the first item row, the final row.
const R0: usize = 0;
const R1: usize = 1;
const R2: usize = 2;
const R3: usize = 3;
const R4: usize = 4;
const R5: usize = 5;
const SAME_ITEM: usize = 6;
const ITEMS: usize = 7;
const ITEM_FIRST: usize = 8;
const FINAL: usize = 9;
pub const PRE_WIDTH: usize = 10;

/// Offsets of the public values.
pub mod public {
    use super::BOUNDARY_LEN;
    use crate::hash::HASH_LEN;
    use crate::output::NET_LIMBS;

    pub const LAMBDA: usize = 0;
    pub const R: usize = 4;
    pub const P_IN: usize = 8;
    pub const P_OUT: usize = 12;
    pub const P_SIG: usize = 16;
    pub const NET: usize = 20;
    pub const TX_HASH: usize = NET + NET_LIMBS;
    pub const ENTRY: usize = TX_HASH + HASH_LEN;
    pub const EXIT: usize = ENTRY + BOUNDARY_LEN;
    pub const LEN: usize = EXIT + BOUNDARY_LEN;
}

use public as pv;

/// A chunk's capacity in items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemChunkAir {
    pub items: usize,
}

impl ItemChunkAir {
    fn final_row(&self) -> usize {
        self.items * ITEM_ROWS
    }

    pub fn height(&self) -> usize {
        (self.final_row() + 1).next_power_of_two()
    }

    fn item_base(&self, i: usize) -> usize {
        i * ITEM_ROWS
    }
}

/// The transaction state where a chunk starts or stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boundary {
    /// The digest sponge.
    pub dig: [BabyBear; WIDTH],
    pub cnt_in: u32,
    pub cnt_out: u32,
    /// The transaction's constants: claimed digest and declared counts.
    pub msg: Hash,
    pub nin: u32,
    pub nout: u32,
}

impl Boundary {
    /// Between transactions: all zero.
    pub const IDLE: Self = Self {
        dig: [BabyBear::ZERO; WIDTH],
        cnt_in: 0,
        cnt_out: 0,
        msg: [BabyBear::ZERO; HASH_LEN],
        nin: 0,
        nout: 0,
    };

    pub fn elements(&self) -> [BabyBear; BOUNDARY_LEN] {
        let mut out = [BabyBear::ZERO; BOUNDARY_LEN];
        out[..WIDTH].copy_from_slice(&self.dig);
        out[WIDTH] = BabyBear::new(self.cnt_in);
        out[WIDTH + 1] = BabyBear::new(self.cnt_out);
        out[WIDTH + 2..WIDTH + 2 + HASH_LEN].copy_from_slice(&self.msg);
        out[BOUNDARY_LEN - 2] = BabyBear::new(self.nin);
        out[BOUNDARY_LEN - 1] = BabyBear::new(self.nout);
        out
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

/// The multiset element of a spent output: its commitment and position.
pub fn input_element(lambda: BabyBear4, c: &Commitment, position: u64) -> Option<BabyBear4> {
    let (lo, hi) = crate::state::split(position);
    Some(fingerprint(
        lambda,
        c.to_field()?.into_iter().chain([lo, hi]),
    ))
}

/// The multiset element of a created output: its commitment and nonce.
pub fn output_element(lambda: BabyBear4, record: &OutputRecord) -> Option<BabyBear4> {
    let c = record.commitment.to_field()?;
    Some(fingerprint(
        lambda,
        c.into_iter().chain(output::nonce_limbs(&record.nonce)),
    ))
}

/// The multiset element of a signature: the key's hash and the message.
pub fn sig_element(lambda: BabyBear4, pkh: &Hash, msg: &Hash) -> BabyBear4 {
    fingerprint(lambda, pkh.iter().chain(msg).copied())
}

/// A chunk's transaction hash: a sponge whose rate starts at zero and
/// capacity at `blinding`, absorbing each finished transaction's digest.
pub fn chunk_hash(blinding: &Hash, digests: &[Hash]) -> Hash {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[HASH_LEN..].copy_from_slice(blinding);
    for d in digests {
        for (x, &v) in s.iter_mut().zip(d) {
            *x += v;
        }
        poseidon2::permute(&mut s);
    }
    core::array::from_fn(|j| s[j])
}

/// `prod (r - e)` over `elements`.
pub fn product(r: BabyBear4, elements: impl IntoIterator<Item = BabyBear4>) -> BabyBear4 {
    elements
        .into_iter()
        .fold(BabyBear4::ONE, |acc, e| acc * (r - e))
}

/// A transaction digest's initial sponge state, as in [`TxBody::digest`].
///
/// [`TxBody::digest`]: crate::transaction::TxBody::digest
fn digest_init<V: Var>(nin: V, nout: V) -> [V; WIDTH] {
    let k = |x: u32| V::from(BabyBear::new(x));
    core::array::from_fn(|j| match j {
        8 => nin,
        9 => nout,
        13 => k(domain::TX_BODY),
        14 => nin * BabyBear::new(HASH_LEN as u32 + 2) + nout * BabyBear::new(HASH_LEN as u32),
        _ => k(0),
    })
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

impl Air for ItemChunkAir {
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
        for i in 0..self.items {
            let base = self.item_base(i);
            for k in 0..ITEM_ROWS {
                let row = m.row_mut(base + k);
                row[R0 + k] = one;
                row[ITEMS] = one;
                if k + 1 < ITEM_ROWS {
                    row[SAME_ITEM] = one;
                }
            }
        }
        if self.items > 0 {
            m.row_mut(0)[ITEM_FIRST] = one;
        }
        m.row_mut(self.final_row())[FINAL] = one;
        Some(m)
    }

    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
        let (l, n, p) = (c.local(), c.next(), c.preprocessed_local());
        let public = c.public();
        let k = |x: u32| V::from(BabyBear::new(x));
        let zero = k(0);
        let one = k(1);
        let out = |j: usize| l[OUTPUT_OFFSET + j];
        let ext = |col: usize, row: &[V]| -> [V; 4] { core::array::from_fn(|b| row[col + b]) };
        let pub4 = |at: usize| -> [V; 4] { core::array::from_fn(|b| public[at + b]) };
        let (lambda, r) = (pub4(pv::LAMBDA), pub4(pv::R));
        let limb = |bits: usize, count: usize| {
            (0..count).fold(zero, |acc, b| acc + l[bits + b] * BabyBear::new(1 << b))
        };
        let (kstart, kin, kout, kend) = (l[KSTART], l[KIN], l[KOUT], l[KEND]);
        let io = kin + kout;
        let items = p[ITEMS];

        poseidon2_air::eval(c, l);

        // Item kinds: at most one flag, constant in the item, as are its
        // values.
        for flag in [kstart, kin, kout, kend] {
            c.assert_zero(items * flag * (flag - one));
        }
        let kinds = kstart + kin + kout + kend;
        c.assert_zero(items * kinds * (kinds - one));
        for col in (KSTART..KEND + 1).chain(AMT..POS + 2) {
            c.assert_zero(p[SAME_ITEM] * (n[col] - l[col]));
        }
        // Transaction constants change only after an end. (A start needs
        // an idle digest, which only an end leaves, so this covers every
        // start but a chunk's first item, whose constants the entry check
        // leaves free.)
        let tx_end = p[R5] * kend;
        for col in (MSG..MSG + HASH_LEN).chain([NIN, NOUT]) {
            c.assert_zero_transition(items * (one - tx_end) * (n[col] - l[col]));
        }

        // Start item, on R0: idle state; start the digest.
        let start = p[R0] * kstart;
        c.assert_zero(start * l[CNT_IN]);
        c.assert_zero(start * l[CNT_OUT]);
        let init = digest_init(l[NIN], l[NOUT]);
        for (j, &v) in init.iter().enumerate() {
            c.assert_zero(start * l[DIG + j]);
            c.assert_zero(start * (n[DIG + j] - v));
        }

        // Commitment sponge (inputs and outputs), as `output::commit`.
        let r0 = p[R0] * io;
        for &x in &l[HASH_LEN..HASH_LEN + 5] {
            c.assert_zero(r0 * x);
        }
        c.assert_zero(r0 * (l[13] - k(domain::OUTPUT)));
        c.assert_zero(r0 * (l[14] - k(COMMIT_LEN as u32)));
        c.assert_zero(r0 * l[15]);
        for j in 0..WIDTH {
            let absorbed = match j {
                0..4 => l[AMT + j],
                4..8 => l[NONCE + j - 4],
                _ => zero,
            };
            c.assert_zero(r0 * (n[j] - out(j) - absorbed));
        }
        let r1 = p[R1] * io;
        for j in 0..WIDTH {
            let absorbed = if j < 4 { l[NONCE + 4 + j] } else { zero };
            c.assert_zero(r1 * (n[j] - out(j) - absorbed));
        }
        // Output amounts are 16-bit limbs (and the final row's bits are
        // bits).
        let bits_rows = p[R1] + p[FINAL];
        for b in 0..NET_LIMBS * LIMB_BITS {
            c.assert_zero(bits_rows * l[BITS + b] * (l[BITS + b] - one));
        }
        for j in 0..AMOUNT_LIMBS {
            c.assert_zero(p[R1] * kout * (l[AMT + j] - limb(BITS + j * LIMB_BITS, LIMB_BITS)));
        }

        // Running sums: inputs add, outputs subtract (on R1).
        for j in 0..AMOUNT_LIMBS {
            let d = n[SUM + j] - l[SUM + j];
            c.assert_zero(p[R1] * (d - (kin - kout) * l[AMT + j]));
            c.assert_zero_transition(items * (one - p[R1]) * d);
        }

        // Fingerprints: commitments on R2 (the row's output), signatures on
        // R0 (an input's pkh is the row's input).
        let mut powers = [[zero; 4]; 2 * HASH_LEN];
        powers[0][0] = one;
        for j in 1..powers.len() {
            powers[j] = ext_mul(powers[j - 1], lambda);
        }
        let element = |values: &mut dyn Iterator<Item = V>| -> [V; 4] {
            values.zip(&powers).fold([zero; 4], |acc, (x, pw)| {
                core::array::from_fn(|b| acc[b] + x * pw[b])
            })
        };
        let factor = |flag: V, e: [V; 4]| -> [V; 4] {
            core::array::from_fn(|b| {
                let unit = if b == 0 { one } else { zero };
                unit - flag * unit + flag * (r[b] - e[b])
            })
        };
        let commitment: [V; HASH_LEN] = core::array::from_fn(out);
        let e_in = element(&mut commitment.iter().copied().chain([l[POS], l[POS + 1]]));
        let e_out = element(
            &mut commitment
                .iter()
                .copied()
                .chain(l[NONCE..NONCE + NONCE_LIMBS].iter().copied()),
        );
        let e_sig = element(&mut l[..HASH_LEN].iter().chain(&l[MSG..MSG + HASH_LEN]).copied());
        for (row, col, flag, e) in [
            (R2, PIN, kin, e_in),
            (R2, POUT, kout, e_out),
            (R0, PSIG, kin, e_sig),
        ] {
            let next = ext_mul(ext(col, l), factor(flag, e));
            for b in 0..4 {
                c.assert_zero(p[row] * (n[col + b] - next[b]));
                c.assert_zero_transition(items * (one - p[row]) * (n[col + b] - l[col + b]));
            }
        }

        // Digest: R2's commitment feeds R3's absorption. An output's R3
        // updates the digest; an input's R3 continues into R4, absorbing its
        // position, and R4 updates it. A start sets it, an end clears it.
        let r2 = p[R2] * io;
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { out(j) } else { zero };
            c.assert_zero(r2 * (n[j] - n[DIG + j] - absorbed));
        }
        for (j, &x) in n[..WIDTH].iter().enumerate() {
            let absorbed = match j {
                0 => l[POS],
                1 => l[POS + 1],
                _ => zero,
            };
            c.assert_zero(p[R3] * kin * (x - out(j) - absorbed));
        }
        let update = p[R3] * kout + p[R4] * kin;
        for j in 0..WIDTH {
            let col = DIG + j;
            c.assert_zero(update * (n[col] - out(j)));
            c.assert_zero(tx_end * n[col]);
            c.assert_zero_transition(items * (one - update - tx_end - start) * (n[col] - l[col]));
        }

        // Counts: inputs and outputs so far in the transaction, cleared at
        // its end.
        for (col, flag) in [(CNT_IN, kin), (CNT_OUT, kout)] {
            c.assert_zero(p[R4] * (n[col] - l[col] - flag));
            c.assert_zero(p[R5] * (n[col] - (one - kend) * l[col]));
            c.assert_zero_transition(items * (one - p[R4] - p[R5]) * (n[col] - l[col]));
        }

        // End item, on R0: the digest is the claimed one, the counts match,
        // and the digest goes into the transaction-hash sponge.
        let end = p[R0] * kend;
        for j in 0..HASH_LEN {
            c.assert_zero(end * (l[DIG + j] - l[MSG + j]));
        }
        c.assert_zero(end * (l[CNT_IN] - l[NIN]));
        c.assert_zero(end * (l[CNT_OUT] - l[NOUT]));
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { l[MSG + j] } else { zero };
            c.assert_zero(end * (l[j] - l[TXS + j] - absorbed));
            c.assert_zero(end * (n[TXS + j] - out(j)));
            c.assert_zero_transition(items * (one - end) * (n[TXS + j] - l[TXS + j]));
        }

        // First row: the entry boundary (a start item brings its own
        // constants, so then the entry's must be idle) and the initial
        // totals. The transaction-hash capacity is left free: the blinding.
        let first = p[ITEM_FIRST];
        let mut at = pv::ENTRY;
        for (col, len) in BOUNDARY_COLS {
            for j in 0..len {
                let gate = if col == MSG || col == NIN {
                    first * (one - kstart)
                } else {
                    first
                };
                c.assert_zero(gate * (l[col + j] - public[at + j]));
                if col == MSG || col == NIN {
                    c.assert_zero(first * kstart * public[at + j]);
                }
            }
            at += len;
        }
        for col in (SUM..SUM + AMOUNT_LIMBS).chain(TXS..TXS + HASH_LEN) {
            c.assert_zero(first * l[col]);
        }
        for col in [PIN, POUT, PSIG] {
            c.assert_zero(first * (l[col] - one));
            for b in 1..4 {
                c.assert_zero(first * l[col + b]);
            }
        }

        // Final row: the exit boundary and the totals.
        let fin = p[FINAL];
        let mut at = pv::EXIT;
        for (col, len) in BOUNDARY_COLS {
            for j in 0..len {
                c.assert_zero(fin * (l[col + j] - public[at + j]));
            }
            at += len;
        }
        for (col, at) in [(PIN, pv::P_IN), (POUT, pv::P_OUT), (PSIG, pv::P_SIG)] {
            for b in 0..4 {
                c.assert_zero(fin * (l[col + b] - public[at + b]));
            }
        }
        for j in 0..HASH_LEN {
            c.assert_zero(fin * (public[pv::TX_HASH + j] - l[TXS + j]));
        }
        // Net value: limbs x of 2^95 - sum_j SUM_j 2^(16 j), with carries.
        for b in 0..(NET_LIMBS - 1) * CARRY_BITS {
            c.assert_zero(fin * l[CB + b] * (l[CB + b] - one));
        }
        let carry = |j: usize| limb(CB + j * CARRY_BITS, CARRY_BITS) - k(CARRY_OFFSET);
        for j in 0..NET_LIMBS {
            let x = limb(BITS + j * LIMB_BITS, LIMB_BITS);
            let sum = if j < AMOUNT_LIMBS { l[SUM + j] } else { zero };
            let offset = k((NET_OFFSET >> (LIMB_BITS * j)) as u16 as u32);
            let carry_in = if j == 0 { zero } else { carry(j - 1) };
            let carry_out = if j + 1 < NET_LIMBS { carry(j) } else { zero };
            c.assert_zero(
                fin * (offset - sum + carry_in - x - carry_out * BabyBear::new(1 << LIMB_BITS)),
            );
            c.assert_zero(fin * (public[pv::NET + j] - x));
        }
    }
}

/// One item of a block's stream, with what it needs from its transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    /// Starts a transaction with these constants.
    Start {
        msg: Hash,
        nin: u32,
        nout: u32,
    },
    Input {
        pkh: Hash,
        amount: [BabyBear; AMOUNT_LIMBS],
        nonce: [BabyBear; NONCE_LIMBS],
        position: u64,
    },
    Output {
        hash: Hash,
        amount: [BabyBear; AMOUNT_LIMBS],
        nonce: [BabyBear; NONCE_LIMBS],
    },
    End,
}

/// The item stream for `txs`, in order: each transaction's start, inputs,
/// outputs and end.
///
/// # Panics
/// If a hash isn't canonical.
pub fn stream(txs: &[Transaction]) -> Vec<Item> {
    let mut items = Vec::new();
    for tx in txs {
        let body = &tx.body;
        items.push(Item::Start {
            msg: body.digest().expect("canonical hashes"),
            nin: body.inputs.len() as u32,
            nout: body.outputs.len() as u32,
        });
        for input in &body.inputs {
            items.push(Item::Input {
                pkh: input.pubkey.hash(),
                amount: output::amount_limbs(input.amount),
                nonce: output::nonce_limbs(&input.nonce),
                position: input.position,
            });
        }
        for o in &body.outputs {
            items.push(Item::Output {
                hash: hash::from_bytes(&o.hash).expect("canonical output hash"),
                amount: output::amount_limbs(o.amount),
                nonce: output::nonce_limbs(&o.nonce),
            });
        }
        items.push(Item::End);
    }
    items
}

/// The commitment an input or output item hashes.
fn commitment_of(
    hash: &Hash,
    amount: &[BabyBear; AMOUNT_LIMBS],
    nonce: &[BabyBear; NONCE_LIMBS],
) -> Hash {
    hash::sponge(domain::OUTPUT, &[], &[hash, amount, nonce])
}

impl Boundary {
    /// The state after `item`, as the circuit computes it.
    pub fn after(mut self, item: &Item) -> Self {
        match *item {
            Item::Start { msg, nin, nout } => {
                self.msg = msg;
                self.nin = nin;
                self.nout = nout;
                self.dig = digest_init(BabyBear::new(nin), BabyBear::new(nout));
            }
            Item::Input {
                pkh,
                amount,
                nonce,
                position,
            } => {
                let c = commitment_of(&pkh, &amount, &nonce);
                for (x, &v) in self.dig.iter_mut().zip(&c) {
                    *x += v;
                }
                poseidon2::permute(&mut self.dig);
                let (lo, hi) = crate::state::split(position);
                self.dig[0] += lo;
                self.dig[1] += hi;
                poseidon2::permute(&mut self.dig);
                self.cnt_in += 1;
            }
            Item::Output {
                hash,
                amount,
                nonce,
            } => {
                let c = commitment_of(&hash, &amount, &nonce);
                for (x, &v) in self.dig.iter_mut().zip(&c) {
                    *x += v;
                }
                poseidon2::permute(&mut self.dig);
                self.cnt_out += 1;
            }
            Item::End => self = Self::IDLE,
        }
        self
    }
}

/// Values carried down the rows.
struct Carried {
    sum: [i64; AMOUNT_LIMBS],
    pin: BabyBear4,
    pout: BabyBear4,
    psig: BabyBear4,
    txs: [BabyBear; WIDTH],
    b: Boundary,
}

impl Carried {
    fn write(&self, row: &mut [BabyBear]) {
        for (cell, &s) in row[SUM..SUM + AMOUNT_LIMBS].iter_mut().zip(&self.sum) {
            *cell = signed(s);
        }
        row[PIN..PIN + 4].copy_from_slice(&self.pin.0);
        row[POUT..POUT + 4].copy_from_slice(&self.pout.0);
        row[PSIG..PSIG + 4].copy_from_slice(&self.psig.0);
        row[TXS..TXS + WIDTH].copy_from_slice(&self.txs);
        let mut at = 0;
        let e = self.b.elements();
        for (col, len) in BOUNDARY_COLS {
            row[col..col + len].copy_from_slice(&e[at..at + len]);
            at += len;
        }
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

impl ItemChunkAir {
    /// The trace and public values proving `items` (at most the capacity)
    /// from the `entry` boundary, with the transaction hash blinded by
    /// `blinding`. Also returns the exit boundary.
    ///
    /// # Panics
    /// If there are too many items.
    pub fn trace(
        &self,
        items: &[Item],
        entry: &Boundary,
        blinding: &Hash,
        lambda: BabyBear4,
        r: BabyBear4,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        assert!(
            self.items <= MAX_ITEMS,
            "too many items for the net carries"
        );
        assert!(items.len() <= self.items, "items don't fit the chunk");
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let mut txs = [BabyBear::ZERO; WIDTH];
        txs[HASH_LEN..].copy_from_slice(blinding);
        let mut st = Carried {
            sum: [0; AMOUNT_LIMBS],
            pin: BabyBear4::ONE,
            pout: BabyBear4::ONE,
            psig: BabyBear4::ONE,
            txs,
            b: *entry,
        };
        let powers: Vec<BabyBear4> = (0..2 * HASH_LEN)
            .scan(BabyBear4::ONE, |pw, _| {
                let v = *pw;
                *pw *= lambda;
                Some(v)
            })
            .collect();
        let fp = |xs: &[BabyBear]| {
            xs.iter()
                .zip(&powers)
                .fold(BabyBear4::ZERO, |acc, (&x, &pw)| acc + pw * x)
        };
        let zeros = [BabyBear::ZERO; WIDTH];
        for i in 0..self.items {
            let base = self.item_base(i);
            let item = items.get(i).copied();
            let (kstart, kin, kout, kend) = match item {
                Some(Item::Start { .. }) => (true, false, false, false),
                Some(Item::Input { .. }) => (false, true, false, false),
                Some(Item::Output { .. }) => (false, false, true, false),
                Some(Item::End) => (false, false, false, true),
                None => (false, false, false, false),
            };
            let (hash, amount, nonce, position) = match item {
                Some(Item::Input {
                    pkh,
                    amount,
                    nonce,
                    position,
                }) => (pkh, amount, nonce, position),
                Some(Item::Output {
                    hash,
                    amount,
                    nonce,
                }) => (hash, amount, nonce, 0),
                _ => (
                    [BabyBear::ZERO; HASH_LEN],
                    [BabyBear::ZERO; AMOUNT_LIMBS],
                    [BabyBear::ZERO; NONCE_LIMBS],
                    0,
                ),
            };
            // A start's constants are on its own rows.
            if let Some(Item::Start { msg, nin, nout }) = item {
                st.b.msg = msg;
                st.b.nin = nin;
                st.b.nout = nout;
            }
            let (lo, hi) = crate::state::split(position);
            let mut row_perm = |k: usize, input: [BabyBear; WIDTH], st: &Carried| {
                let out = poseidon2_air::generate(input, m.row_mut(base + k));
                st.write(m.row_mut(base + k));
                out
            };

            // R0: the commitment's first block, or an end's absorption.
            let mut input = zeros;
            if kin || kout {
                input[..HASH_LEN].copy_from_slice(&hash);
                input[13] = BabyBear::new(domain::OUTPUT);
                input[14] = BabyBear::new(COMMIT_LEN as u32);
            } else if kend {
                input = st.txs;
                for (x, &v) in input.iter_mut().zip(&st.b.msg) {
                    *x += v;
                }
            }
            let out0 = row_perm(0, input, &st);
            if kend {
                st.txs = out0;
            }
            if kin {
                let mut xs = hash.to_vec();
                xs.extend(st.b.msg);
                st.psig *= r - fp(&xs);
            }
            if kstart {
                st.b.dig = digest_init(BabyBear::new(st.b.nin), BabyBear::new(st.b.nout));
            }

            // R1: amount and nonce.
            let mut input = zeros;
            if kin || kout {
                input = out0;
                for j in 0..4 {
                    input[j] += amount[j];
                    input[4 + j] += nonce[j];
                }
            }
            let out1 = row_perm(1, input, &st);
            let sign = if kin {
                1
            } else if kout {
                -1
            } else {
                0
            };
            for (s, &a) in st.sum.iter_mut().zip(&amount) {
                *s += sign * limb_value(a);
            }

            // R2: the commitment.
            let mut input = zeros;
            if kin || kout {
                input = out1;
                for j in 0..4 {
                    input[j] += nonce[4 + j];
                }
            }
            let out2 = row_perm(2, input, &st);
            let commitment: Hash = core::array::from_fn(|j| out2[j]);
            if kin {
                let mut xs = commitment.to_vec();
                xs.extend([lo, hi]);
                st.pin *= r - fp(&xs);
            }
            if kout {
                let mut xs = commitment.to_vec();
                xs.extend(nonce);
                st.pout *= r - fp(&xs);
            }

            // R3: absorb the commitment.
            let mut input = zeros;
            if kin || kout {
                input = st.b.dig;
                for (x, &v) in input.iter_mut().zip(&commitment) {
                    *x += v;
                }
            }
            let out3 = row_perm(3, input, &st);
            if kout {
                st.b.dig = out3;
            }

            // R4: an input absorbs its position.
            let mut input = zeros;
            if kin {
                input = out3;
                input[0] += lo;
                input[1] += hi;
            }
            let out4 = row_perm(4, input, &st);
            if kin {
                st.b.dig = out4;
                st.b.cnt_in += 1;
            }
            if kout {
                st.b.cnt_out += 1;
            }

            // R5: an end returns to idle.
            row_perm(5, zeros, &st);
            if kend {
                st.b = Boundary::IDLE;
            }

            if kout {
                let row = m.row_mut(base + 1);
                for (j, a) in amount.iter().enumerate() {
                    let v = a.as_u32();
                    for b in 0..LIMB_BITS {
                        row[BITS + j * LIMB_BITS + b] = BabyBear::new((v >> b) & 1);
                    }
                }
            }
            for k in 0..ITEM_ROWS {
                let row = m.row_mut(base + k);
                row[KSTART] = BabyBear::from(kstart);
                row[KIN] = BabyBear::from(kin);
                row[KOUT] = BabyBear::from(kout);
                row[KEND] = BabyBear::from(kend);
                row[AMT..AMT + AMOUNT_LIMBS].copy_from_slice(&amount);
                row[NONCE..NONCE + NONCE_LIMBS].copy_from_slice(&nonce);
                row[POS] = lo;
                row[POS + 1] = hi;
            }
        }

        // Final row: the exit boundary and the net value, with carries.
        let fin = self.final_row();
        let net: i128 = -st
            .sum
            .iter()
            .enumerate()
            .map(|(j, &s)| (s as i128) << (LIMB_BITS * j))
            .sum::<i128>();
        let x = output::net_limbs(net);
        for row in fin..m.height() {
            poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(row));
        }
        st.write(m.row_mut(fin));
        {
            let row = m.row_mut(fin);
            let mut carry = 0i64;
            for j in 0..NET_LIMBS {
                let xv = x[j].as_u32();
                for b in 0..LIMB_BITS {
                    row[BITS + j * LIMB_BITS + b] = BabyBear::new((xv >> b) & 1);
                }
                if j + 1 < NET_LIMBS {
                    let sum = if j < AMOUNT_LIMBS { st.sum[j] } else { 0 };
                    let offset = (NET_OFFSET >> (LIMB_BITS * j)) as u16 as i64;
                    carry = (offset - sum + carry - xv as i64).div_euclid(1 << LIMB_BITS);
                    let biased =
                        (carry + CARRY_OFFSET as i64).clamp(0, (1 << CARRY_BITS) - 1) as u32;
                    for b in 0..CARRY_BITS {
                        row[CB + j * CARRY_BITS + b] = BabyBear::new((biased >> b) & 1);
                    }
                }
            }
        }

        let mut public = Vec::with_capacity(pv::LEN);
        for v in [lambda, r, st.pin, st.pout, st.psig] {
            public.extend_from_slice(&v.0);
        }
        public.extend_from_slice(&x);
        public.extend_from_slice(&st.txs[..HASH_LEN]);
        public.extend_from_slice(&entry.elements());
        public.extend_from_slice(&st.b.elements());
        (m, public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::babybear::P;
    use crate::fri::FriParams;
    use crate::output::{Amount, NONCE_LEN, Nonce, Output};
    use crate::stark;
    use crate::transaction::{Input, TxBody, pubkey_hash};
    use crate::wots::{PublicKey, SecretKey, keygen};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);
    const BLINDING: Hash = BabyBear::new_array([29, 31, 37, 41, 43, 47, 53, 59]);

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
                position: 1000 + n as u64,
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

    /// Products and transaction hash computed natively, for transactions
    /// all finishing in one chunk.
    fn expected(txs: &[Transaction]) -> [BabyBear4; 3] {
        let inputs = || txs.iter().flat_map(|t| t.body.inputs.iter());
        let p_in = product(
            R,
            inputs().map(|i| input_element(LAMBDA, &i.commitment, i.position).unwrap()),
        );
        let p_out = product(
            R,
            txs.iter().flat_map(|t| {
                t.body
                    .outputs
                    .iter()
                    .map(|o| output_element(LAMBDA, &o.record().unwrap()).unwrap())
            }),
        );
        let p_sig = product(
            R,
            txs.iter().flat_map(|t| {
                let msg = t.body.digest().unwrap();
                t.body
                    .inputs
                    .iter()
                    .map(move |i| sig_element(LAMBDA, &i.pubkey.hash(), &msg))
            }),
        );
        [p_in, p_out, p_sig]
    }

    fn ext_at(public: &[BabyBear], at: usize) -> BabyBear4 {
        BabyBear4(public[at..at + 4].try_into().unwrap())
    }

    fn run(
        air: &ItemChunkAir,
        items: &[Item],
        entry: &Boundary,
    ) -> (Result<(), usize>, Vec<BabyBear>) {
        let (t, public) = air.trace(items, entry, &BLINDING, LAMBDA, R);
        (check_constraints(air, &t, &public), public)
    }

    fn exit_of(public: &[BabyBear]) -> &[BabyBear] {
        &public[pv::EXIT..pv::LEN]
    }

    #[test]
    fn whole_transactions_and_the_reward() {
        let t1 = tx(vec![input(1, 100)], vec![out(1, 60), out(2, 30)]);
        let reward = tx(vec![], vec![out(4, 1000 + 15)]);
        let t2 = tx(vec![input(2, 50), input(3, 25)], vec![out(3, 70)]);
        let txs = [t1, reward, t2];
        let items = stream(&txs);
        assert_eq!(items.len(), 5 + 3 + 5);
        let air = ItemChunkAir { items: 14 };
        let (result, public) = run(&air, &items, &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        let [p_in, p_out, p_sig] = expected(&txs);
        assert_eq!(ext_at(&public, pv::P_IN), p_in);
        assert_eq!(ext_at(&public, pv::P_OUT), p_out);
        assert_eq!(ext_at(&public, pv::P_SIG), p_sig);
        assert_eq!(
            &public[pv::NET..pv::NET + NET_LIMBS],
            &output::net_limbs(1000)
        );
        let digests: Vec<Hash> = txs.iter().map(|t| t.body.digest().unwrap()).collect();
        assert_eq!(
            &public[pv::TX_HASH..pv::TX_HASH + HASH_LEN],
            &chunk_hash(&BLINDING, &digests)
        );
        assert_eq!(exit_of(&public), &Boundary::IDLE.elements());
    }

    #[test]
    fn a_transaction_spans_chunks() {
        // Cut a 3-input, 4-output transaction across three chunks of 4.
        let t = tx(
            vec![input(1, 10), input(2, 20), input(3, 30)],
            vec![out(1, 5), out(2, 15), out(3, 25), out(4, 10)],
        );
        let items = stream(std::slice::from_ref(&t));
        assert_eq!(items.len(), 9);
        let air = ItemChunkAir { items: 4 };
        let mut entry = Boundary::IDLE;
        let mut net = 0;
        let mut hashes = Vec::new();
        for chunk in items.chunks(4) {
            let (result, public) = run(&air, chunk, &entry);
            assert_eq!(result, Ok(()));
            let exit = chunk.iter().fold(entry, |b, item| b.after(item));
            assert_eq!(exit_of(&public), &exit.elements());
            net += output::net_value(&public[pv::NET..pv::NET + NET_LIMBS]);
            hashes.push(public[pv::TX_HASH..pv::TX_HASH + HASH_LEN].to_vec());
            entry = exit;
        }
        assert_eq!(entry, Boundary::IDLE);
        assert_eq!(net, -5);
        // Only the chunk with the end absorbs the digest.
        let empty = chunk_hash(&BLINDING, &[]);
        assert_eq!(hashes[0], empty);
        assert_eq!(
            hashes[2],
            chunk_hash(&BLINDING, &[t.body.digest().unwrap()])
        );
    }

    #[test]
    fn extreme_nets() {
        let max = u64::MAX;
        let air = ItemChunkAir { items: 40 };
        let t = tx((1..=16).map(|n| input(n, max)).collect(), vec![out(1, 0)]);
        let (result, public) = run(&air, &stream(std::slice::from_ref(&t)), &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        assert_eq!(
            &public[pv::NET..pv::NET + NET_LIMBS],
            &output::net_limbs(-16 * max as i128)
        );
        let t = tx(vec![], (1..=16).map(|n| out(n, max)).collect());
        let (result, public) = run(&air, &stream(std::slice::from_ref(&t)), &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        assert_eq!(
            &public[pv::NET..pv::NET + NET_LIMBS],
            &output::net_limbs(16 * max as i128)
        );
    }

    #[test]
    fn rejects_wrong_public_values() {
        let t = tx(vec![input(1, 100)], vec![out(1, 90)]);
        let air = ItemChunkAir { items: 4 };
        let (trace, public) = air.trace(
            &stream(std::slice::from_ref(&t)),
            &Boundary::IDLE,
            &BLINDING,
            LAMBDA,
            R,
        );
        for at in [
            pv::P_IN,
            pv::P_OUT + 1,
            pv::P_SIG + 2,
            pv::NET,
            pv::TX_HASH + 7,
            pv::ENTRY,
            pv::ENTRY + BOUNDARY_LEN - 1,
            pv::EXIT + 3,
        ] {
            let mut bad = public.clone();
            bad[at] += BabyBear::ONE;
            assert!(
                check_constraints(&air, &trace, &bad).is_err(),
                "public {at}"
            );
        }
    }

    #[test]
    fn rejects_an_output_changed_after_signing() {
        let t = tx(vec![input(1, 100)], vec![out(1, 90)]);
        let mut items = stream(std::slice::from_ref(&t));
        if let Item::Output { amount, .. } = &mut items[2] {
            *amount = output::amount_limbs(80);
        }
        let air = ItemChunkAir { items: 4 };
        // The end item's digest check fails.
        assert_eq!(run(&air, &items, &Boundary::IDLE).0, Err(air.item_base(3)));
    }

    #[test]
    fn rejects_a_wrong_count() {
        let t = tx(vec![], vec![out(1, 10), out(2, 20)]);
        let mut items = stream(std::slice::from_ref(&t));
        if let Item::Start { nout, .. } = &mut items[0] {
            *nout = 1;
        }
        let air = ItemChunkAir { items: 4 };
        assert!(run(&air, &items, &Boundary::IDLE).0.is_err());
    }

    #[test]
    fn rejects_a_start_mid_transaction() {
        // A second start before the first transaction ends.
        let t = tx(vec![], vec![out(1, 10)]);
        let mut items = stream(std::slice::from_ref(&t));
        items.insert(1, items[0]);
        let air = ItemChunkAir { items: 4 };
        assert_eq!(run(&air, &items, &Boundary::IDLE).0, Err(air.item_base(1)));
    }

    #[test]
    fn rejects_an_entry_mid_transaction_without_its_start() {
        // Items of a transaction whose start was never proved: the entry
        // must then be its open state, and a fake one fails at the end.
        let t = tx(vec![], vec![out(1, 10)]);
        let items = stream(std::slice::from_ref(&t));
        let air = ItemChunkAir { items: 3 };
        let fake = Boundary {
            msg: t.body.digest().unwrap(),
            nin: 0,
            nout: 1,
            ..Boundary::IDLE
        };
        assert!(run(&air, &items[1..], &fake).0.is_err());
    }

    #[test]
    fn rejects_negative_output_limb() {
        // An output limb of -5 (p - 5) would mint 5 units; only the range
        // check stands in the way.
        let hash = [BabyBear::ONE; HASH_LEN];
        let amount = [
            BabyBear::new(P - 5),
            BabyBear::ZERO,
            BabyBear::ZERO,
            BabyBear::ZERO,
        ];
        let nonce = [BabyBear::ZERO; NONCE_LIMBS];
        let c = commitment_of(&hash, &amount, &nonce);
        let msg = crate::transaction::digest_of(&[], &[c]);
        let items = [
            Item::Start {
                msg,
                nin: 0,
                nout: 1,
            },
            Item::Output {
                hash,
                amount,
                nonce,
            },
            Item::End,
        ];
        let air = ItemChunkAir { items: 3 };
        assert_eq!(
            run(&air, &items, &Boundary::IDLE).0,
            Err(air.item_base(1) + 1),
            "fails at the output's R1 row"
        );
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let txs = [
            tx(vec![input(1, 100)], vec![out(1, 60), out(2, 30)]),
            tx(vec![], vec![out(3, 60)]),
        ];
        let air = ItemChunkAir { items: 8 };
        let (trace, public) = air.trace(&stream(&txs), &Boundary::IDLE, &BLINDING, LAMBDA, R);
        if let Err(i) = check_constraints(&air, &trace, &public) {
            panic!(
                "row {i}: {:?}",
                crate::air::failing_constraints(&air, &trace, &public, i)
            );
        }
        let (pk, vk) = stark::setup(&params, &air);
        let proof = stark::prove(&params, &pk, &air, trace, &public);
        assert_eq!(stark::verify(&params, &vk, &air, &public, &proof), Ok(()));
    }
}
