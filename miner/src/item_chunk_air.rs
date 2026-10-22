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
//! Signatures are proved elsewhere ([`crate::sig_chunk_air`]): each key
//! spend here contributes `(pkh, its transaction's digest, its leaf)` to a
//! fingerprint product `P_sig`, which the aggregation matches against the
//! signature chunks' (one signature per key). A condition spend instead
//! contributes what its condition is checked against (its output's hash,
//! commitment, position, amount and creation height, the transaction's
//! digest and signed-outputs digest, and how many condition inputs come
//! before it: [`cond_element`]) to
//! `P_cond`, matched against the condition chunks'
//! ([`crate::cond_chunk_air`]), which check the condition and add its
//! signers to the signature side.
//!
//! Rows: `items` slots of six rows (`R0..R5`), then a final row. Each item
//! is a start, key-spend input, condition input, output, end or padding
//! (witness flags, constant in the item):
//!
//! - start: `R0` checks the state is idle and starts the transaction's
//!   digest from its counts and its signed-outputs digest from their count
//!   (the transaction's constants: claimed digest and counts, signed
//!   outputs' digest and count, which may change only into a start or
//!   after an end);
//! - input: `R0..R2` hash its commitment from `pkh`, amount and nonce; `R3`
//!   absorbs the commitment and `R4` its position into the digest; `R0`
//!   also adds its element to `P_sig` (a key spend) or `P_cond` (a
//!   condition); `R5` absorbs its leaf, creation height and kind into the
//!   transaction-hash sponge, binding them;
//! - output: `R0..R2` hash its commitment (amount range-checked), `R3`
//!   absorbs it into the digest, and `R4`, if it's signed, into the signed
//!   outputs' digest; an output flagged as publishing a preimage adds its
//!   nonce's first limbs to `P_pre`, which the aggregation matches against
//!   the condition chunks' hashlock preimages, and one flagged as a payout
//!   adds its transaction's digest and its commitment
//!   ([`payout_element`]), matched against the payouts the condition
//!   chunks require of protected inputs; `R5` binds the flags;
//! - end: `R0` checks both digests are the claimed ones and the counts
//!   match, and `R0..R2` absorb the digest, the signed outputs' digest and
//!   their count into the transaction-hash sponge; `R5` returns to idle.
//!
//! A transaction's digest is [`TxBody::digest`], its signed outputs'
//! [`TxBody::signed_digest`]. Transactions have no balance rule of their
//! own: the net value `sum(outputs) - sum(inputs)` is summed, and only the
//! block's total is constrained (to the subsidy). It is published as
//! [`output::net_limbs`].
//!
//! Public values: `lambda (4) | r (4) | P_in (4) | P_out (4) | P_sig (4) |
//! P_cond (4) | P_pre (4) | net (6) | tx hash (8) | entry | exit` ([`BOUNDARY_LEN`]
//! each), with `P_in` over inputs' (commitment, position, creation height)
//! ([`input_element`]), `P_out` over outputs' (commitment, nonce)
//! ([`output_element`]), `P_sig` over key spends' (pkh, digest, leaf)
//! ([`sig_element`]), `P_cond` over condition inputs' [`cond_element`],
//! `P_pre` over published preimages ([`preimage_element`]) and payouts
//! ([`payout_element`]). The
//! transaction hash ([`chunk_hash`]) is a sponge, with a random blinding
//! as its initial capacity.
//!
//! [`TxBody::digest`]: crate::transaction::TxBody::digest
//! [`TxBody::signed_digest`]: crate::transaction::TxBody::signed_digest

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
use crate::transaction::{PREIMAGE_LEN, Spend, Transaction};

const COMMIT_LEN: usize = HASH_LEN + AMOUNT_LIMBS + NONCE_LIMBS;
/// Nonce limbs absorbed with the amount (`R0`), and after it (`R1`).
const NONCE_A: usize = HASH_LEN - AMOUNT_LIMBS;
const NONCE_B: usize = NONCE_LIMBS - NONCE_A;
const _: () = assert!(COMMIT_LEN <= 3 * HASH_LEN && NONCE_B <= HASH_LEN);
/// A preimage's limbs: the first of a nonce's.
pub const PRE_LIMBS: usize = PREIMAGE_LEN / output::NONCE_LIMB_BYTES;
/// Elements of a condition input's element ([`cond_element`]).
pub const COND_LEN: usize = 4 * HASH_LEN + 2 + AMOUNT_LIMBS + 2;
/// Fingerprint powers: enough for the longest element.
const POWERS: usize = COND_LEN;
const _: () = assert!(COND_LEN >= HASH_LEN + NONCE_LIMBS && COND_LEN > 2 * HASH_LEN);
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
/// signed-outputs sponge and count, condition inputs so far, and the
/// transaction's constants (claimed digest, declared input and output
/// counts, claimed signed-outputs digest and count).
pub const BOUNDARY_LEN: usize = 2 * WIDTH + 4 + 2 * HASH_LEN + 3;

// Columns after the permutation's.
const P: usize = poseidon2_air::NUM_COLS;
/// Item kind flags, constant in an item: start, key-spend input, condition
/// input, output, end.
const KSTART: usize = P;
const KIN: usize = KSTART + 1;
const KCIN: usize = KIN + 1;
const KOUT: usize = KCIN + 1;
const KEND: usize = KOUT + 1;
/// Item values, constant in an item.
const AMT: usize = KEND + 1;
const NONCE: usize = AMT + AMOUNT_LIMBS;
const POS: usize = NONCE + NONCE_LIMBS;
/// An input's creation height (the spent output's).
const CRT: usize = POS + 2;
/// The commitment an input or output hashes to.
const CMT: usize = CRT + 1;
/// A key spend's leaf.
const NIDX: usize = CMT + HASH_LEN;
/// An output's flags: signed; publishes a preimage (its nonce's first
/// limbs); pays a protected input's non-signing member.
const SGN: usize = NIDX + 1;
const PREO: usize = SGN + 1;
const PAYO: usize = PREO + 1;
/// Per-transaction constants: claimed digest, input and output counts,
/// signed outputs' claimed digest and count.
const MSG: usize = PAYO + 1;
const NIN: usize = MSG + HASH_LEN;
const NOUT: usize = NIN + 1;
const OMSG: usize = NOUT + 1;
const NS: usize = OMSG + HASH_LEN;
/// Carried down the rows.
const SUM: usize = NS + 1;
const DIG: usize = SUM + AMOUNT_LIMBS;
const PIN: usize = DIG + WIDTH;
const POUT: usize = PIN + 4;
const PSIG: usize = POUT + 4;
const PCOND: usize = PSIG + 4;
const PPRE: usize = PCOND + 4;
const CNT_IN: usize = PPRE + 4;
const CNT_OUT: usize = CNT_IN + 1;
/// The signed outputs' sponge and count, and the condition inputs, so far.
const ODIG: usize = CNT_OUT + 1;
const CNT_SG: usize = ODIG + WIDTH;
const CNT_CIN: usize = CNT_SG + 1;
const TXS: usize = CNT_CIN + 1;
/// Output amount bits on `R1` rows; net value bits on the final row.
const BITS: usize = TXS + WIDTH;
const CB: usize = BITS + NET_LIMBS * LIMB_BITS;
pub const NUM_COLS: usize = CB + (NET_LIMBS - 1) * CARRY_BITS;

/// The boundary's columns, in [`Boundary`] order.
const BOUNDARY_COLS: [(usize, usize); 6] = [
    (DIG, WIDTH),
    (CNT_IN, 1),
    (CNT_OUT, 1),
    (ODIG, WIDTH),
    (CNT_SG, 2),
    (MSG, NS + 1 - MSG),
];
/// Where the transaction's constants start among them.
const TX_CONSTANTS: usize = 5;

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
    pub const P_COND: usize = 20;
    pub const P_PRE: usize = 24;
    pub const NET: usize = 28;
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
    /// The signed outputs' sponge and count so far.
    pub odig: [BabyBear; WIDTH],
    pub cnt_signed: u32,
    /// The condition inputs so far.
    pub cnt_cond: u32,
    /// The transaction's constants.
    pub tx: TxConstants,
}

/// A transaction's constants, which its start brings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxConstants {
    /// The claimed digest and declared counts.
    pub msg: Hash,
    pub nin: u32,
    pub nout: u32,
    /// The claimed signed-outputs digest and count.
    pub omsg: Hash,
    pub ns: u32,
}

impl TxConstants {
    pub const NONE: Self = Self {
        msg: [BabyBear::ZERO; HASH_LEN],
        nin: 0,
        nout: 0,
        omsg: [BabyBear::ZERO; HASH_LEN],
        ns: 0,
    };

    fn elements(&self) -> Vec<BabyBear> {
        let mut out = self.msg.to_vec();
        out.extend([BabyBear::new(self.nin), BabyBear::new(self.nout)]);
        out.extend(self.omsg);
        out.push(BabyBear::new(self.ns));
        out
    }
}

impl Boundary {
    /// Between transactions: all zero.
    pub const IDLE: Self = Self {
        dig: [BabyBear::ZERO; WIDTH],
        cnt_in: 0,
        cnt_out: 0,
        odig: [BabyBear::ZERO; WIDTH],
        cnt_signed: 0,
        cnt_cond: 0,
        tx: TxConstants::NONE,
    };

    pub fn elements(&self) -> [BabyBear; BOUNDARY_LEN] {
        let mut out = self.dig.to_vec();
        out.extend([BabyBear::new(self.cnt_in), BabyBear::new(self.cnt_out)]);
        out.extend(self.odig);
        out.extend([BabyBear::new(self.cnt_signed), BabyBear::new(self.cnt_cond)]);
        out.extend(self.tx.elements());
        out.try_into().expect("boundary length")
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

/// The multiset element of a spent output: its commitment, position and
/// creation height.
pub fn input_element(
    lambda: BabyBear4,
    c: &Commitment,
    position: u64,
    created: u32,
) -> Option<BabyBear4> {
    let (lo, hi) = crate::state::split(position);
    Some(fingerprint(
        lambda,
        c.to_field()?
            .into_iter()
            .chain([lo, hi, BabyBear::new(created)]),
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

/// The multiset element of a signature: the key tree's root, the message
/// and the key's index.
pub fn sig_element(lambda: BabyBear4, root: &Hash, msg: &Hash, index: u32) -> BabyBear4 {
    fingerprint(
        lambda,
        root.iter()
            .chain(msg)
            .copied()
            .chain([BabyBear::new(index)]),
    )
}

/// What a condition input is checked against: its output's hash,
/// commitment, position, amount and creation height, its transaction's
/// digest and signed-outputs digest, and how many condition inputs come
/// before it in the transaction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CondClaim {
    pub hash: Hash,
    pub commitment: Hash,
    pub position: u64,
    pub amount: [BabyBear; AMOUNT_LIMBS],
    pub created: u32,
    pub msg: Hash,
    pub omsg: Hash,
    pub index: u32,
}

impl CondClaim {
    /// Its elements, in fingerprint order.
    pub fn elements(&self) -> [BabyBear; COND_LEN] {
        let (lo, hi) = crate::state::split(self.position);
        let mut out = self.hash.to_vec();
        out.extend(self.commitment);
        out.extend([lo, hi]);
        out.extend(self.amount);
        out.push(BabyBear::new(self.created));
        out.extend(self.msg);
        out.extend(self.omsg);
        out.push(BabyBear::new(self.index));
        out.try_into().expect("claim length")
    }
}

/// The multiset element of a condition input.
pub fn cond_element(lambda: BabyBear4, claim: &CondClaim) -> BabyBear4 {
    fingerprint(lambda, claim.elements())
}

/// The multiset element of a published preimage (an output's nonce's
/// first limbs, or a hashlock spend's preimage).
pub fn preimage_element(lambda: BabyBear4, pre: &[BabyBear; PRE_LIMBS]) -> BabyBear4 {
    fingerprint(lambda, pre.iter().copied())
}

/// The multiset element of a payout: its transaction's digest and the
/// commitment, then a one (so it's never a preimage's).
pub fn payout_element(lambda: BabyBear4, msg: &Hash, commitment: &Hash) -> BabyBear4 {
    fingerprint(
        lambda,
        msg.iter().chain(commitment).copied().chain([BabyBear::ONE]),
    )
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

/// The signed outputs' sponge's initial state, as in
/// [`TxBody::signed_digest`] (whose rate stays zero with no outputs).
///
/// [`TxBody::signed_digest`]: crate::transaction::TxBody::signed_digest
fn signed_init<V: Var>(ns: V) -> [V; WIDTH] {
    let k = |x: u32| V::from(BabyBear::new(x));
    core::array::from_fn(|j| match j {
        8 => ns,
        13 => k(domain::SIGNED_OUTPUTS),
        14 => ns * BabyBear::new(HASH_LEN as u32),
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
        let (kstart, kin, kcin, kout, kend) = (l[KSTART], l[KIN], l[KCIN], l[KOUT], l[KEND]);
        let inp = kin + kcin;
        let io = inp + kout;
        let (sgn, preo, payo) = (l[SGN], l[PREO], l[PAYO]);
        let items = p[ITEMS];

        poseidon2_air::eval(c, l);

        // Item kinds: at most one flag, constant in the item, as are its
        // values. Output flags only on outputs.
        for flag in [kstart, kin, kcin, kout, kend, sgn, preo, payo] {
            c.assert_zero(items * flag * (flag - one));
        }
        let kinds = kstart + inp + kout + kend;
        c.assert_zero(items * kinds * (kinds - one));
        c.assert_zero(items * sgn * (one - kout));
        c.assert_zero(items * preo * (one - kout));
        c.assert_zero(items * payo * (one - kout));
        c.assert_zero(items * preo * payo);
        for col in (KSTART..KEND + 1).chain(AMT..PAYO + 1) {
            c.assert_zero(p[SAME_ITEM] * (n[col] - l[col]));
        }
        // Transaction constants change only after an end. (A start needs
        // an idle digest, which only an end leaves, so this covers every
        // start but a chunk's first item, whose constants the entry check
        // leaves free.)
        let tx_end = p[R5] * kend;
        for col in MSG..NS + 1 {
            c.assert_zero_transition(items * (one - tx_end) * (n[col] - l[col]));
        }

        // Start item, on R0: idle state; start both digests.
        let start = p[R0] * kstart;
        for col in [CNT_IN, CNT_OUT, CNT_SG, CNT_CIN] {
            c.assert_zero(start * l[col]);
        }
        let init = digest_init(l[NIN], l[NOUT]);
        let sinit = signed_init(l[NS]);
        for j in 0..WIDTH {
            c.assert_zero(start * l[DIG + j]);
            c.assert_zero(start * (n[DIG + j] - init[j]));
            c.assert_zero(start * l[ODIG + j]);
            c.assert_zero(start * (n[ODIG + j] - sinit[j]));
        }

        // Commitment sponge (inputs and outputs), as `output::commit`.
        let r0 = p[R0] * io;
        for &x in &l[HASH_LEN..HASH_LEN + 5] {
            c.assert_zero(r0 * x);
        }
        c.assert_zero(r0 * (l[13] - k(domain::OUTPUT)));
        c.assert_zero(r0 * (l[14] - k(COMMIT_LEN as u32)));
        c.assert_zero(r0 * l[15]);
        for (j, &x) in n[..WIDTH].iter().enumerate() {
            let absorbed = match j {
                j if j < AMOUNT_LIMBS => l[AMT + j],
                j if j < HASH_LEN => l[NONCE + j - AMOUNT_LIMBS],
                _ => zero,
            };
            c.assert_zero(r0 * (x - out(j) - absorbed));
        }
        let r1 = p[R1] * io;
        for j in 0..WIDTH {
            let absorbed = if j < NONCE_B {
                l[NONCE + NONCE_A + j]
            } else {
                zero
            };
            c.assert_zero(r1 * (n[j] - out(j) - absorbed));
        }
        for j in 0..HASH_LEN {
            c.assert_zero(p[R2] * io * (out(j) - l[CMT + j]));
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
            c.assert_zero(p[R1] * (d - (inp - kout) * l[AMT + j]));
            c.assert_zero_transition(items * (one - p[R1]) * d);
        }

        // Fingerprints: commitments on R2, signatures and conditions on R0
        // (an input's pkh is the row's input).
        let mut powers = [[zero; 4]; POWERS];
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
        let col = |at: usize, len: usize| (at..at + len).map(|j| l[j]);
        let e_in = element(&mut col(CMT, HASH_LEN).chain(col(POS, 2)).chain([l[CRT]]));
        let e_out = element(&mut col(CMT, HASH_LEN).chain(col(NONCE, NONCE_LIMBS)));
        let e_sig = element(&mut col(0, HASH_LEN).chain(col(MSG, HASH_LEN)).chain([l[NIDX]]));
        let e_cond = element(
            &mut col(0, HASH_LEN)
                .chain(col(CMT, HASH_LEN))
                .chain(col(POS, 2))
                .chain(col(AMT, AMOUNT_LIMBS))
                .chain([l[CRT]])
                .chain(col(MSG, HASH_LEN))
                .chain(col(OMSG, HASH_LEN))
                .chain([l[CNT_CIN]]),
        );
        let e_pre = element(&mut col(NONCE, PRE_LIMBS));
        let e_pay = element(&mut col(MSG, HASH_LEN).chain(col(CMT, HASH_LEN)).chain([one]));
        // P_pre takes a published preimage or a payout (never both).
        let f_pre: [V; 4] = core::array::from_fn(|b| {
            let unit = if b == 0 { one } else { zero };
            unit - (preo + payo) * unit + preo * (r[b] - e_pre[b]) + payo * (r[b] - e_pay[b])
        });
        for (row, col, f) in [
            (R2, PIN, factor(inp, e_in)),
            (R2, POUT, factor(kout, e_out)),
            (R0, PSIG, factor(kin, e_sig)),
            (R0, PCOND, factor(kcin, e_cond)),
            (R0, PPRE, f_pre),
        ] {
            let next = ext_mul(ext(col, l), f);
            for b in 0..4 {
                c.assert_zero(p[row] * (n[col + b] - next[b]));
                c.assert_zero_transition(items * (one - p[row]) * (n[col + b] - l[col + b]));
            }
        }

        // Digest: R3 absorbs the commitment. An output's R3 updates the
        // digest; an input's R3 continues into R4, absorbing its position,
        // and R4 updates it. A start sets it, an end clears it.
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { l[CMT + j] } else { zero };
            c.assert_zero(p[R3] * io * (l[j] - l[DIG + j] - absorbed));
        }
        for (j, &x) in n[..WIDTH].iter().enumerate() {
            let absorbed = match j {
                0 => l[POS],
                1 => l[POS + 1],
                _ => zero,
            };
            c.assert_zero(p[R3] * inp * (x - out(j) - absorbed));
        }
        let update = p[R3] * kout + p[R4] * inp;
        for j in 0..WIDTH {
            let col = DIG + j;
            c.assert_zero(update * (n[col] - out(j)));
            c.assert_zero(tx_end * n[col]);
            c.assert_zero_transition(items * (one - update - tx_end - start) * (n[col] - l[col]));
        }

        // Signed outputs: R4 absorbs a signed one's commitment.
        let signed = p[R4] * sgn;
        for j in 0..WIDTH {
            let col = ODIG + j;
            let absorbed = if j < HASH_LEN { l[CMT + j] } else { zero };
            c.assert_zero(signed * (l[j] - l[col] - absorbed));
            c.assert_zero(signed * (n[col] - out(j)));
            c.assert_zero(tx_end * n[col]);
            c.assert_zero_transition(items * (one - signed - tx_end - start) * (n[col] - l[col]));
        }

        // Counts: inputs, outputs, signed outputs and condition inputs so
        // far in the transaction, cleared at its end.
        for (col, flag) in [
            (CNT_IN, inp),
            (CNT_OUT, kout),
            (CNT_SG, sgn),
            (CNT_CIN, kcin),
        ] {
            c.assert_zero(p[R4] * (n[col] - l[col] - flag));
        }
        for col in [CNT_IN, CNT_OUT, CNT_SG, CNT_CIN] {
            c.assert_zero(p[R5] * (n[col] - (one - kend) * l[col]));
            c.assert_zero_transition(items * (one - p[R4] - p[R5]) * (n[col] - l[col]));
        }

        // End item, on R0: the digests are the claimed ones, the counts
        // match.
        let end = p[R0] * kend;
        for j in 0..HASH_LEN {
            c.assert_zero(end * (l[DIG + j] - l[MSG + j]));
            c.assert_zero(end * (l[ODIG + j] - l[OMSG + j]));
        }
        c.assert_zero(end * (l[CNT_IN] - l[NIN]));
        c.assert_zero(end * (l[CNT_OUT] - l[NOUT]));
        c.assert_zero(end * (l[CNT_SG] - l[NS]));
        // Transaction hash: an input's R5 absorbs its leaf, creation height
        // and kind, an output's whether it publishes a preimage or pays a
        // payout; an end's
        // R0, R1 and R2 absorb its digest, signed-outputs digest, then their
        // count.
        let absorbs: [(V, Vec<V>); 5] = [
            (p[R5] * inp, vec![l[NIDX], l[CRT], kcin]),
            (p[R5] * kout, vec![preo, payo]),
            (end, col(MSG, HASH_LEN).collect()),
            (p[R1] * kend, col(OMSG, HASH_LEN).collect()),
            (p[R2] * kend, vec![l[NS]]),
        ];
        let mut absorbing = zero;
        for (sel, block) in &absorbs {
            for j in 0..WIDTH {
                let absorbed = block.get(j).copied().unwrap_or(zero);
                c.assert_zero(*sel * (l[j] - l[TXS + j] - absorbed));
                c.assert_zero(*sel * (n[TXS + j] - out(j)));
            }
            absorbing = absorbing + *sel;
        }
        for j in 0..WIDTH {
            c.assert_zero_transition(items * (one - absorbing) * (n[TXS + j] - l[TXS + j]));
        }

        // First row: the entry boundary (a start item brings its own
        // constants, so then the entry's must be idle) and the initial
        // totals. The transaction-hash capacity is left free: the blinding.
        let first = p[ITEM_FIRST];
        let mut at = pv::ENTRY;
        for (i, (col, len)) in BOUNDARY_COLS.into_iter().enumerate() {
            for j in 0..len {
                if i == TX_CONSTANTS {
                    c.assert_zero(first * (one - kstart) * (l[col + j] - public[at + j]));
                    c.assert_zero(first * kstart * public[at + j]);
                } else {
                    c.assert_zero(first * (l[col + j] - public[at + j]));
                }
            }
            at += len;
        }
        for col in (SUM..SUM + AMOUNT_LIMBS).chain(TXS..TXS + HASH_LEN) {
            c.assert_zero(first * l[col]);
        }
        for col in [PIN, POUT, PSIG, PCOND, PPRE] {
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
        for (col, at) in [
            (PIN, pv::P_IN),
            (POUT, pv::P_OUT),
            (PSIG, pv::P_SIG),
            (PCOND, pv::P_COND),
            (PPRE, pv::P_PRE),
        ] {
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
    Start(TxConstants),
    Input {
        /// The output's hash (its key tree's root).
        pkh: Hash,
        amount: [BabyBear; AMOUNT_LIMBS],
        nonce: [BabyBear; NONCE_LIMBS],
        position: u64,
        /// The spent output's creation height.
        created: u32,
        /// A condition spend (else a key spend).
        condition: bool,
        /// A key spend's leaf (zero for a condition).
        leaf: u32,
    },
    Output {
        hash: Hash,
        amount: [BabyBear; AMOUNT_LIMBS],
        nonce: [BabyBear; NONCE_LIMBS],
        /// Covered by the transaction's signed-outputs digest.
        signed: bool,
        /// Publishes a hashlock input's preimage (its nonce's first limbs).
        publishes: bool,
        /// Pays a protected input's non-signing member.
        pays: bool,
    },
    End,
}

/// The item stream for `txs`, in order: each transaction's start, inputs,
/// outputs and end. `created` gives the creation height of the output at a
/// position.
///
/// # Panics
/// If a hash isn't canonical.
pub fn stream(txs: &[Transaction], created: impl Fn(u64) -> u32) -> Vec<Item> {
    let mut items = Vec::new();
    for tx in txs {
        let body = &tx.body;
        items.push(Item::Start(TxConstants {
            msg: body.digest().expect("canonical hashes"),
            nin: body.inputs.len() as u32,
            nout: body.outputs.len() as u32,
            omsg: body.signed_digest().expect("canonical hashes"),
            ns: body.signed_outputs as u32,
        }));
        for input in &body.inputs {
            let (condition, leaf) = match input.spend {
                Spend::Key { leaf, .. } => (false, leaf as u32),
                Spend::Condition { .. } => (true, 0),
            };
            items.push(Item::Input {
                pkh: input.key_root(),
                amount: output::amount_limbs(input.amount),
                nonce: output::nonce_limbs(&input.nonce),
                position: input.position,
                created: created(input.position),
                condition,
                leaf,
            });
        }

        // The outputs publishing the hashlock inputs' preimages, and those
        // paying protected inputs' payouts.
        let publishers = body.publishers().expect("validated");
        let payouts = body.payouts().expect("validated");
        for (i, o) in body.outputs.iter().enumerate() {
            items.push(Item::Output {
                hash: hash::from_bytes(&o.hash).expect("canonical output hash"),
                amount: output::amount_limbs(o.amount),
                nonce: output::nonce_limbs(&o.nonce),
                signed: i < body.signed_outputs,
                publishes: publishers.contains(&Some(i)),
                pays: payouts.contains(&i),
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

/// `state` after absorbing `block` (added into the rate) and permuting.
fn absorb(state: &mut [BabyBear; WIDTH], block: &[BabyBear]) {
    for (x, &v) in state.iter_mut().zip(block) {
        *x += v;
    }
    poseidon2::permute(state);
}

impl Boundary {
    /// The state after `item`, as the circuit computes it.
    pub fn after(mut self, item: &Item) -> Self {
        match *item {
            Item::Start(tx) => {
                self.tx = tx;
                self.dig = digest_init(BabyBear::new(tx.nin), BabyBear::new(tx.nout));
                self.odig = signed_init(BabyBear::new(tx.ns));
            }
            Item::Input {
                pkh,
                amount,
                nonce,
                position,
                condition,
                ..
            } => {
                let c = commitment_of(&pkh, &amount, &nonce);
                absorb(&mut self.dig, &c);
                let (lo, hi) = crate::state::split(position);
                absorb(&mut self.dig, &[lo, hi]);
                self.cnt_in += 1;
                self.cnt_cond += condition as u32;
            }
            Item::Output {
                hash,
                amount,
                nonce,
                signed,
                ..
            } => {
                let c = commitment_of(&hash, &amount, &nonce);
                absorb(&mut self.dig, &c);
                if signed {
                    absorb(&mut self.odig, &c);
                    self.cnt_signed += 1;
                }
                self.cnt_out += 1;
            }
            Item::End => self = Self::IDLE,
        }
        self
    }
}

/// What an item adds to the transaction-hash sponge (blocks absorbed in
/// turn), given the boundary before it.
fn tx_hash_blocks(b: &Boundary, item: &Item) -> Vec<Vec<BabyBear>> {
    match *item {
        Item::Input {
            created,
            condition,
            leaf,
            ..
        } => vec![vec![
            BabyBear::new(leaf),
            BabyBear::new(created),
            BabyBear::from(condition),
        ]],
        Item::Output {
            publishes, pays, ..
        } => vec![vec![BabyBear::from(publishes), BabyBear::from(pays)]],
        Item::End => vec![
            b.tx.msg.to_vec(),
            b.tx.omsg.to_vec(),
            vec![BabyBear::new(b.tx.ns)],
        ],
        Item::Start(_) => vec![],
    }
}

/// A chunk's transaction hash: a sponge whose rate starts at zero and
/// capacity at `blinding`, absorbing each input's leaf, creation height
/// and kind, each output's preimage flag and, at each transaction's end,
/// its digest, signed-outputs digest and their count, for `items` from
/// `entry`.
pub fn chunk_hash(blinding: &Hash, entry: &Boundary, items: &[Item]) -> Hash {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[HASH_LEN..].copy_from_slice(blinding);
    let mut b = *entry;
    for item in items {
        for block in tx_hash_blocks(&b, item) {
            absorb(&mut s, &block);
        }
        b = b.after(item);
    }
    core::array::from_fn(|j| s[j])
}

/// Values carried down the rows.
struct Carried {
    sum: [i64; AMOUNT_LIMBS],
    pin: BabyBear4,
    pout: BabyBear4,
    psig: BabyBear4,
    pcond: BabyBear4,
    ppre: BabyBear4,
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
        row[PCOND..PCOND + 4].copy_from_slice(&self.pcond.0);
        row[PPRE..PPRE + 4].copy_from_slice(&self.ppre.0);
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
            pcond: BabyBear4::ONE,
            ppre: BabyBear4::ONE,
            txs,
            b: *entry,
        };
        let powers: Vec<BabyBear4> = (0..POWERS)
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
            let kind = |f: fn(&Item) -> bool| item.as_ref().is_some_and(f);
            let kstart = kind(|i| matches!(i, Item::Start(_)));
            let kin = kind(|i| {
                matches!(
                    i,
                    Item::Input {
                        condition: false,
                        ..
                    }
                )
            });
            let kcin = kind(|i| {
                matches!(
                    i,
                    Item::Input {
                        condition: true,
                        ..
                    }
                )
            });
            let kout = kind(|i| matches!(i, Item::Output { .. }));
            let kend = kind(|i| matches!(i, Item::End));
            let inp = kin || kcin;
            let zero_hash = [BabyBear::ZERO; HASH_LEN];
            let zero_amount = [BabyBear::ZERO; AMOUNT_LIMBS];
            let zero_nonce = [BabyBear::ZERO; NONCE_LIMBS];
            let (hash, amount, nonce, position, created, leaf, sgn, preo, payo) = match item {
                Some(Item::Input {
                    pkh,
                    amount,
                    nonce,
                    position,
                    created,
                    leaf,
                    ..
                }) => (
                    pkh, amount, nonce, position, created, leaf, false, false, false,
                ),
                Some(Item::Output {
                    hash,
                    amount,
                    nonce,
                    signed,
                    publishes,
                    pays,
                }) => (hash, amount, nonce, 0, 0, 0, signed, publishes, pays),
                _ => (
                    zero_hash,
                    zero_amount,
                    zero_nonce,
                    0,
                    0,
                    0,
                    false,
                    false,
                    false,
                ),
            };
            let tx_blocks = item
                .as_ref()
                .map_or(vec![], |item| tx_hash_blocks(&st.b, item));
            // A start's constants are on its own rows.
            if let Some(Item::Start(tx)) = item {
                st.b.tx = tx;
            }
            let (lo, hi) = crate::state::split(position);
            let commitment = if inp || kout {
                commitment_of(&hash, &amount, &nonce)
            } else {
                zero_hash
            };
            let mut row_perm = |k: usize, input: [BabyBear; WIDTH], st: &Carried| {
                let out = poseidon2_air::generate(input, m.row_mut(base + k));
                st.write(m.row_mut(base + k));
                out
            };
            let absorb_txs = |st: &mut Carried, block: &[BabyBear]| {
                let mut input = st.txs;
                for (x, &v) in input.iter_mut().zip(block) {
                    *x += v;
                }
                input
            };

            // R0: the commitment's first block, or an end's absorption.
            let mut input = zeros;
            if inp || kout {
                input[..HASH_LEN].copy_from_slice(&hash);
                input[13] = BabyBear::new(domain::OUTPUT);
                input[14] = BabyBear::new(COMMIT_LEN as u32);
            } else if kend {
                input = absorb_txs(&mut st, &tx_blocks[0]);
            }
            let out0 = row_perm(0, input, &st);
            if kend {
                st.txs = out0;
            }
            if kin {
                let mut xs = hash.to_vec();
                xs.extend(st.b.tx.msg);
                xs.push(BabyBear::new(leaf));
                st.psig *= r - fp(&xs);
            }
            if kcin {
                let claim = CondClaim {
                    hash,
                    commitment,
                    position,
                    amount,
                    created,
                    msg: st.b.tx.msg,
                    omsg: st.b.tx.omsg,
                    index: st.b.cnt_cond,
                };
                st.pcond *= r - fp(&claim.elements());
            }
            if preo {
                st.ppre *= r - fp(&nonce[..PRE_LIMBS]);
            }
            if payo {
                let mut xs = st.b.tx.msg.to_vec();
                xs.extend(commitment);
                xs.push(BabyBear::ONE);
                st.ppre *= r - fp(&xs);
            }
            if kstart {
                st.b.dig = digest_init(BabyBear::new(st.b.tx.nin), BabyBear::new(st.b.tx.nout));
                st.b.odig = signed_init(BabyBear::new(st.b.tx.ns));
            }

            // R1: amount and nonce (or an end's second absorption).
            let mut input = zeros;
            if inp || kout {
                input = out0;
                for j in 0..AMOUNT_LIMBS {
                    input[j] += amount[j];
                }
                for j in 0..NONCE_A {
                    input[AMOUNT_LIMBS + j] += nonce[j];
                }
            } else if kend {
                input = absorb_txs(&mut st, &tx_blocks[1]);
            }
            let out1 = row_perm(1, input, &st);
            if kend {
                st.txs = out1;
            }
            let sign = if inp {
                1
            } else if kout {
                -1
            } else {
                0
            };
            for (s, &a) in st.sum.iter_mut().zip(&amount) {
                *s += sign * limb_value(a);
            }

            // R2: the commitment (or an end's third absorption).
            let mut input = zeros;
            if inp || kout {
                input = out1;
                for j in 0..NONCE_B {
                    input[j] += nonce[NONCE_A + j];
                }
            } else if kend {
                input = absorb_txs(&mut st, &tx_blocks[2]);
            }
            let out2 = row_perm(2, input, &st);
            if kend {
                st.txs = out2;
            }
            if inp || kout {
                debug_assert_eq!(&out2[..HASH_LEN], &commitment);
            }
            if inp {
                let mut xs = commitment.to_vec();
                xs.extend([lo, hi, BabyBear::new(created)]);
                st.pin *= r - fp(&xs);
            }
            if kout {
                let mut xs = commitment.to_vec();
                xs.extend(nonce);
                st.pout *= r - fp(&xs);
            }

            // R3: absorb the commitment.
            let mut input = zeros;
            if inp || kout {
                input = st.b.dig;
                for (x, &v) in input.iter_mut().zip(&commitment) {
                    *x += v;
                }
            }
            let out3 = row_perm(3, input, &st);
            if kout {
                st.b.dig = out3;
            }

            // R4: an input absorbs its position; a signed output its
            // commitment into the signed outputs' digest.
            let mut input = zeros;
            if inp {
                input = out3;
                input[0] += lo;
                input[1] += hi;
            } else if sgn {
                input = st.b.odig;
                for (x, &v) in input.iter_mut().zip(&commitment) {
                    *x += v;
                }
            }
            let out4 = row_perm(4, input, &st);
            if inp {
                st.b.dig = out4;
                st.b.cnt_in += 1;
            }
            if kout {
                st.b.cnt_out += 1;
            }
            if kcin {
                st.b.cnt_cond += 1;
            }
            if sgn {
                st.b.odig = out4;
                st.b.cnt_signed += 1;
            }

            // R5: an input binds its leaf, height and kind, an output its
            // preimage flag; an end returns to idle.
            let input = if inp || kout {
                absorb_txs(&mut st, &tx_blocks[0])
            } else {
                zeros
            };
            let out5 = row_perm(5, input, &st);
            if inp || kout {
                st.txs = out5;
            }
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
                row[KCIN] = BabyBear::from(kcin);
                row[KOUT] = BabyBear::from(kout);
                row[KEND] = BabyBear::from(kend);
                row[AMT..AMT + AMOUNT_LIMBS].copy_from_slice(&amount);
                row[NONCE..NONCE + NONCE_LIMBS].copy_from_slice(&nonce);
                row[POS] = lo;
                row[POS + 1] = hi;
                row[CRT] = BabyBear::new(created);
                row[CMT..CMT + HASH_LEN].copy_from_slice(&commitment);
                row[NIDX] = BabyBear::new(leaf);
                row[SGN] = BabyBear::from(sgn);
                row[PREO] = BabyBear::from(preo);
                row[PAYO] = BabyBear::from(payo);
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
        for v in [lambda, r, st.pin, st.pout, st.psig, st.pcond, st.ppre] {
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
    use crate::transaction::{
        Condition, Input, KeyTree, SigMode, Signer, SignerTree, Spend, TxBody, hashlock,
    };
    use crate::wots::{PublicKey, SecretKey, keygen};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);
    const BLINDING: Hash = BabyBear::new_array([29, 31, 37, 41, 43, 47, 53, 59]);

    /// The creation height of the output at `position`, made up.
    fn created(position: u64) -> u32 {
        position as u32 % 97
    }

    fn key(n: u32) -> (SecretKey, PublicKey) {
        keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 31 + i as u32
        })))
    }

    fn input(n: u32, amount: Amount) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let nonce: Nonce = [n as u8 + 1; NONCE_LEN];
        let tree = KeyTree::single(&pubkey);
        let commitment = output::commit(&tree.output_hash(), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                position: 1000 + n as u64,
                amount,
                nonce,
                spend: Spend::Key {
                    pubkey,
                    leaf: 0,
                    path: tree.path(0),
                },
            },
        )
    }

    /// An input spending a one-key condition (key `n`, mode `mode`).
    fn cond_input(
        n: u32,
        amount: Amount,
        mode: SigMode,
        pre: Option<[u8; 15]>,
    ) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let tree = SignerTree::new(vec![pubkey.hash()]);
        let condition = Condition {
            mode,
            hashlock: pre.map(|p| hashlock(&p)),
            ..Condition::key(tree.root())
        };
        let keys = KeyTree::of_conditions(std::slice::from_ref(&condition));
        let nonce: Nonce = [n as u8 + 1; NONCE_LEN];
        (
            sk,
            Input {
                commitment: output::commit(&keys.output_hash(), amount, &nonce).unwrap(),
                position: 2000 + n as u64,
                amount,
                nonce,
                spend: Spend::Condition {
                    condition,
                    leaf: 0,
                    path: keys.path(0),
                    signers: vec![Signer {
                        member: 0,
                        pubkey,
                        index: 0,
                        path: vec![],
                    }],
                    preimage: pre,
                },
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
        Transaction::sign(TxBody::new(inputs, outputs), keys).unwrap()
    }

    /// Products computed natively, for transactions all finishing in one
    /// chunk.
    fn expected(txs: &[Transaction]) -> [BabyBear4; 4] {
        let inputs = || txs.iter().flat_map(|t| t.body.inputs.iter());
        let p_in = product(
            R,
            inputs().map(|i| {
                input_element(LAMBDA, &i.commitment, i.position, created(i.position)).unwrap()
            }),
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
                t.body.inputs.iter().filter_map(move |i| {
                    let leaf = i.key_leaf()?;
                    Some(sig_element(LAMBDA, &i.key_root(), &msg, leaf as u32))
                })
            }),
        );
        let p_cond = product(
            R,
            txs.iter().flat_map(|t| {
                let b = &t.body;
                b.inputs
                    .iter()
                    .filter(|i| i.condition().is_some())
                    .enumerate()
                    .map(|(index, i)| {
                        cond_element(
                            LAMBDA,
                            &CondClaim {
                                hash: i.key_root(),
                                commitment: i.commitment.to_field().unwrap(),
                                position: i.position,
                                amount: output::amount_limbs(i.amount),
                                created: created(i.position),
                                msg: b.digest().unwrap(),
                                omsg: b.signed_digest().unwrap(),
                                index: index as u32,
                            },
                        )
                    })
            }),
        );
        [p_in, p_out, p_sig, p_cond]
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

    /// Checks `txs` in one chunk against the native products and hash.
    fn check_whole(txs: &[Transaction]) {
        let items = stream(txs, created);
        let air = ItemChunkAir {
            items: items.len() + 1,
        };
        let (result, public) = run(&air, &items, &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        let [p_in, p_out, p_sig, p_cond] = expected(txs);
        assert_eq!(ext_at(&public, pv::P_IN), p_in);
        assert_eq!(ext_at(&public, pv::P_OUT), p_out);
        assert_eq!(ext_at(&public, pv::P_SIG), p_sig);
        assert_eq!(ext_at(&public, pv::P_COND), p_cond);
        assert_eq!(
            &public[pv::TX_HASH..pv::TX_HASH + HASH_LEN],
            &chunk_hash(&BLINDING, &Boundary::IDLE, &items)
        );
        assert_eq!(exit_of(&public), &Boundary::IDLE.elements());
    }

    #[test]
    fn whole_transactions_and_the_reward() {
        let t1 = tx(vec![input(1, 100)], vec![out(1, 60), out(2, 30)]);
        let reward = tx(vec![], vec![out(4, 1000 + 15)]);
        let t2 = tx(vec![input(2, 50), input(3, 25)], vec![out(3, 70)]);
        let txs = [t1, reward, t2];
        let items = stream(&txs, created);
        assert_eq!(items.len(), 5 + 3 + 5);
        check_whole(&txs);
        let air = ItemChunkAir { items: 14 };
        let (_, public) = run(&air, &items, &Boundary::IDLE);
        assert_eq!(
            &public[pv::NET..pv::NET + NET_LIMBS],
            &output::net_limbs(1000)
        );
    }

    #[test]
    fn condition_spends_signed_outputs_and_preimages() {
        // (The ACP input comes first among the condition inputs.)
        let pre = [5u8; 15];
        let (sk, cin) = cond_input(7, 300, SigMode::AnyoneCanPay, Some(pre));
        let (sk2, kin) = input(8, 20);
        let mut claim = out(9, 250);
        claim.nonce[..15].copy_from_slice(&pre);
        let body = TxBody {
            inputs: vec![cin, kin],
            outputs: vec![out(1, 60), claim, out(2, 5)],
            signed_outputs: 2,
        };
        let t = Transaction::sign(body, vec![sk, sk2]).unwrap();
        assert!(t.validate().is_ok());
        let (sk3, apo) = cond_input(9, 40, SigMode::AnyPrevout, None);
        let t2 = Transaction::sign(
            TxBody {
                inputs: vec![apo],
                outputs: vec![out(3, 40)],
                signed_outputs: 1,
            },
            vec![sk3],
        )
        .unwrap();
        check_whole(&[t.clone(), t2]);

        // The publishing output's nonce goes into P_pre: the preimage, which
        // the condition chunks' hashlocks must match. Another output
        // instead gives another product (the aggregation's Q_pre fails).
        let air = ItemChunkAir { items: 8 };
        let items = stream(std::slice::from_ref(&t), created);
        let (result, public) = run(&air, &items, &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        let limbs = crate::transaction::preimage_limbs(&pre);
        assert_eq!(
            ext_at(&public, pv::P_PRE),
            product(R, [preimage_element(LAMBDA, &limbs)])
        );
        let mut moved = items.clone();
        if let Item::Output { publishes, .. } = &mut moved[3] {
            *publishes = true;
        }
        if let Item::Output { publishes, .. } = &mut moved[4] {
            *publishes = false;
        }
        let (result, other) = run(&air, &moved, &Boundary::IDLE);
        assert_eq!(result, Ok(()));
        assert_ne!(ext_at(&other, pv::P_PRE), ext_at(&public, pv::P_PRE));
        // Signing another set of outputs fails.
        let mut items = stream(std::slice::from_ref(&t), created);
        if let Item::Output { signed, .. } = &mut items[5] {
            *signed = true;
        }
        assert!(run(&air, &items, &Boundary::IDLE).0.is_err());
    }

    #[test]
    fn a_transaction_spans_chunks() {
        // Cut a 3-input, 4-output transaction across three chunks of 4.
        let t = tx(
            vec![input(1, 10), input(2, 20), input(3, 30)],
            vec![out(1, 5), out(2, 15), out(3, 25), out(4, 10)],
        );
        let items = stream(std::slice::from_ref(&t), created);
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
            let hash = &public[pv::TX_HASH..pv::TX_HASH + HASH_LEN];
            assert_eq!(hash, &chunk_hash(&BLINDING, &entry, chunk));
            hashes.push(hash.to_vec());
            entry = exit;
        }
        assert_eq!(entry, Boundary::IDLE);
        assert_eq!(net, -5);
    }

    #[test]
    fn extreme_nets() {
        let max = u64::MAX;
        let air = ItemChunkAir { items: 40 };
        let t = tx((1..=16).map(|n| input(n, max)).collect(), vec![out(1, 0)]);
        let (result, public) = run(
            &air,
            &stream(std::slice::from_ref(&t), created),
            &Boundary::IDLE,
        );
        assert_eq!(result, Ok(()));
        assert_eq!(
            &public[pv::NET..pv::NET + NET_LIMBS],
            &output::net_limbs(-16 * max as i128)
        );
        let t = tx(vec![], (1..=16).map(|n| out(n, max)).collect());
        let (result, public) = run(
            &air,
            &stream(std::slice::from_ref(&t), created),
            &Boundary::IDLE,
        );
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
            &stream(std::slice::from_ref(&t), created),
            &Boundary::IDLE,
            &BLINDING,
            LAMBDA,
            R,
        );
        for at in [
            pv::P_IN,
            pv::P_OUT + 1,
            pv::P_SIG + 2,
            pv::P_COND + 3,
            pv::NET,
            pv::TX_HASH + 7,
            pv::ENTRY,
            pv::ENTRY + BOUNDARY_LEN - 1,
            pv::EXIT + 3,
            pv::LEN - 1,
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
        let mut items = stream(std::slice::from_ref(&t), created);
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
        let mut items = stream(std::slice::from_ref(&t), created);
        if let Item::Start(tx) = &mut items[0] {
            tx.nout = 1;
        }
        let air = ItemChunkAir { items: 4 };
        assert!(run(&air, &items, &Boundary::IDLE).0.is_err());
    }

    #[test]
    fn rejects_a_start_mid_transaction() {
        // A second start before the first transaction ends.
        let t = tx(vec![], vec![out(1, 10)]);
        let mut items = stream(std::slice::from_ref(&t), created);
        items.insert(1, items[0]);
        let air = ItemChunkAir { items: 4 };
        assert_eq!(run(&air, &items, &Boundary::IDLE).0, Err(air.item_base(1)));
    }

    #[test]
    fn rejects_an_entry_mid_transaction_without_its_start() {
        // Items of a transaction whose start was never proved: the entry
        // must then be its open state, and a fake one fails at the end.
        let t = tx(vec![], vec![out(1, 10)]);
        let items = stream(std::slice::from_ref(&t), created);
        let air = ItemChunkAir { items: 3 };
        let fake = Boundary {
            tx: TxConstants {
                msg: t.body.digest().unwrap(),
                nin: 0,
                nout: 1,
                ..TxConstants::NONE
            },
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
            Item::Start(TxConstants {
                msg,
                nin: 0,
                nout: 1,
                ..TxConstants::NONE
            }),
            Item::Output {
                hash,
                amount,
                nonce,
                signed: false,
                publishes: false,
                pays: false,
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
        let (sk, cin) = cond_input(7, 300, SigMode::All, None);
        let txs = [
            tx(vec![input(1, 100)], vec![out(1, 60), out(2, 30)]),
            tx(vec![], vec![out(3, 60)]),
            tx(vec![(sk, cin)], vec![out(4, 290)]),
        ];
        let air = ItemChunkAir { items: 16 };
        let (trace, public) = air.trace(
            &stream(&txs, created),
            &Boundary::IDLE,
            &BLINDING,
            LAMBDA,
            R,
        );
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
