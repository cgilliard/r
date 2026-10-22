//! The condition chunk circuit: checks a run of condition spends
//! ([`crate::transaction::Condition`]), revealing only multiset
//! fingerprints of what each was checked against and of the signatures it
//! needs, and a blinded hash of them.
//!
//! Each slot takes a condition input's claim ([`CondClaim`]: its output's
//! hash, commitment, position, amount and creation height, its
//! transaction's digest and signed-outputs digest, and how many condition
//! inputs come before it), which the
//! aggregation matches against the item chunks' (`P_cond`), and the
//! condition it meets. The slot hashes the condition into its key-tree
//! leaf and walks the leaf's path to the output's hash, then checks:
//!
//! - the timelocks: the block's height (public) is at least the
//!   condition's height and at least the output's creation height plus its
//!   block count;
//! - a hashlock: the input's preimage hashes to it (and goes into `P_pre`,
//!   which the aggregation matches against the outputs publishing
//!   preimages);
//! - anyone-can-pay and any-prevout only on a transaction's first
//!   condition input, so no two inputs share the outputs they sign;
//! - the message the signers sign, by the condition's mode: the
//!   transaction's digest ([`SigMode::All`]), or a hash of the
//!   signed-outputs digest with the input's commitment and position
//!   ([`SigMode::AnyoneCanPay`]) or with its amount
//!   ([`SigMode::AnyPrevout`]);
//! - a state rule: each signer's key index is above the condition's state
//!   number.
//!
//! It then adds `(key root, message, index)` for each signer to `P_sig`,
//! on the transaction side, which the signature chunks
//! ([`crate::sig_chunk_air`]) must match.
//!
//! Rows per slot ([`SLOT_ROWS`]): three hashing the condition, eight
//! merges up the output's key tree, one hashing the preimage, three for
//! the messages, six absorbing the slot's claim and signer indices into
//! the chunk's hash (a sponge whose initial capacity is a random
//! blinding). Range checks (timelocks to [`MAX_HEIGHT_BITS`] bits, state
//! numbers to [`MAX_KEY_DEPTH`]) use bit columns on the first rows. Each
//! slot has an active flag; inactive slots contribute nothing.
//!
//! Public values: `lambda (4) | r (4) | height | P_cond (4) | P_sig (4) |
//! P_pre (4) | hash (8)`.
//!
//! [`SigMode::All`]: crate::transaction::SigMode::All
//! [`SigMode::AnyoneCanPay`]: crate::transaction::SigMode::AnyoneCanPay
//! [`SigMode::AnyPrevout`]: crate::transaction::SigMode::AnyPrevout

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::hash::{HASH_LEN, Hash, domain};
use crate::header::MAX_HEIGHT_BITS;
use crate::item_chunk_air::{
    COND_LEN, CondClaim, PRE_LIMBS, cond_element, preimage_element, sig_element,
};
use crate::matrix::RowMajorMatrix;
use crate::output::AMOUNT_LIMBS;
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::transaction::{Condition, KEY_LEVELS, KeyPath, MAX_KEY_DEPTH, SigMode};

const LOCK_BITS: usize = MAX_HEIGHT_BITS as usize;
const STATE_BITS: usize = MAX_KEY_DEPTH;
/// `x^4 = W` in the extension field.
const EXT_W: BabyBear = BabyBear::new(11);

// Row offsets in a slot.
const C1: usize = 0;
const C2: usize = 1;
const C3: usize = 2;
const M1: usize = 3;
const X: usize = M1 + KEY_LEVELS;
const G1: usize = X + 1;
const G2: usize = G1 + 1;
const G3: usize = G2 + 1;
const B1: usize = G3 + 1;
/// Blocks of the chunk hash per slot: the claim, the two indices and the
/// preimage.
const BLOCKS: usize = (COND_LEN + 2 + PRE_LIMBS).div_ceil(HASH_LEN);
/// Rows per slot.
pub const SLOT_ROWS: usize = B1 + BLOCKS;

// Columns after the permutation's.
const P: usize = poseidon2_air::NUM_COLS;
/// The slot's active flag, and active with two keys, and active with a
/// hashlock, constant in the slot.
const ACT: usize = P;
const ACT2: usize = ACT + 1;
const ACTH: usize = ACT2 + 1;
/// The claim, in [`CondClaim::elements`] order.
const CLAIM: usize = ACTH + 1;
const CH: usize = CLAIM;
const CC: usize = CH + HASH_LEN;
const CPOS: usize = CC + HASH_LEN;
const CAMT: usize = CPOS + 2;
const CCRT: usize = CAMT + AMOUNT_LIMBS;
const CMSG: usize = CCRT + 1;
const COMSG: usize = CMSG + HASH_LEN;
const CIDX: usize = COMSG + HASH_LEN;
const _: () = assert!(CIDX + 1 - CLAIM == COND_LEN);
/// The hashlock's preimage (zero without one).
const CPRE: usize = CIDX + 1;
/// The condition: keys, hashlock, state number, locks, mode and flags.
const K1: usize = CPRE + PRE_LIMBS;
const K2: usize = K1 + HASH_LEN;
const HL: usize = K2 + HASH_LEN;
const M: usize = HL + HASH_LEN;
const TL: usize = M + 1;
const DL: usize = TL + 1;
const MACP: usize = DL + 1;
const MAPO: usize = MACP + 1;
const TWO: usize = MAPO + 1;
const STATE: usize = TWO + 1;
const HLK: usize = STATE + 1;
/// The signers' key indices and the message they sign.
const N1: usize = HLK + 1;
const N2: usize = N1 + 1;
const SMSG: usize = N2 + 1;
/// Merge rows: the path's bit and sibling; the running node (from the
/// leaf up).
const BIT: usize = SMSG + HASH_LEN;
const SIB: usize = BIT + 1;
const RUN: usize = SIB + HASH_LEN;
/// Carried: the chunk-hash sponge and the two products.
const CHS: usize = RUN + HASH_LEN;
const PCOND: usize = CHS + WIDTH;
const PSIG: usize = PCOND + 4;
const PPRE: usize = PSIG + 4;
/// Range-check bits.
const RB: usize = PPRE + 4;
const RB_LEN: usize = 2 * LOCK_BITS;
pub const NUM_COLS: usize = RB + RB_LEN;

/// Columns constant in a slot.
const SLOT_CONSTANTS: std::ops::Range<usize> = ACT..SMSG + HASH_LEN;
const _: () = assert!(CPRE > ACT && CPRE + PRE_LIMBS <= SMSG);

// Preprocessed columns: one per row role, plus rows followed by a row of
// the same slot, every slot row, the first row, the final row.
const PC1: usize = 0;
const PC2: usize = 1;
const PC3: usize = 2;
const PMERGE: usize = 3;
const PX: usize = 4;
const PG1: usize = 5;
const PG2: usize = 6;
const PG3: usize = 7;
/// Hash rows `B1..`: one column each.
const PB: usize = 8;
const SAME_SLOT: usize = PB + BLOCKS;
const SLOTS: usize = SAME_SLOT + 1;
const FIRST: usize = SLOTS + 1;
const FINAL: usize = FIRST + 1;
pub const PRE_WIDTH: usize = FINAL + 1;

/// Offsets of the public values.
pub mod public {
    pub const LAMBDA: usize = 0;
    pub const R: usize = 4;
    pub const HEIGHT: usize = 8;
    pub const P_COND: usize = 9;
    pub const P_SIG: usize = 13;
    pub const P_PRE: usize = 17;
    pub const HASH: usize = 21;
    pub const LEN: usize = 29;
}

use public as pv;

/// A chunk's capacity in condition spends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CondChunkAir {
    pub slots: usize,
}

impl CondChunkAir {
    fn final_row(&self) -> usize {
        self.slots * SLOT_ROWS
    }

    pub fn height(&self) -> usize {
        (self.final_row() + 1).next_power_of_two()
    }

    fn base(&self, s: usize) -> usize {
        s * SLOT_ROWS
    }
}

/// A condition spend to check: the claim the item chunks make for it, the
/// condition it meets (at `leaf`, with `path`, of the output's key tree),
/// its signers' key indices (one per key) and its hashlock's preimage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CondItem {
    pub claim: CondClaim,
    pub condition: Condition,
    pub leaf: u8,
    pub path: KeyPath,
    pub indices: Vec<u32>,
    pub preimage: Option<[BabyBear; PRE_LIMBS]>,
}

impl CondItem {
    /// The message its signers sign.
    pub fn message(&self) -> Hash {
        let c = &self.claim;
        match self.condition.mode {
            SigMode::All => c.msg,
            SigMode::AnyoneCanPay => {
                crate::transaction::acp_message(&c.omsg, &c.commitment, c.position)
            }
            SigMode::AnyPrevout => apo_message(&c.omsg, &c.amount),
        }
    }

    /// The signature triples it needs: key root, message, index.
    pub fn signatures(&self) -> Vec<(Hash, Hash, u32)> {
        let msg = self.message();
        self.condition
            .keys
            .iter()
            .zip(&self.indices)
            .map(|(k, &i)| (*k, msg, i))
            .collect()
    }

    /// What the chunk hash absorbs for it: the claim, the indices, the
    /// preimage.
    fn bound(&self) -> Vec<BabyBear> {
        let mut out = self.claim.elements().to_vec();
        out.push(BabyBear::new(self.indices[0]));
        out.push(BabyBear::new(self.indices.get(1).copied().unwrap_or(0)));
        out.extend(self.pre());
        out.resize(BLOCKS * HASH_LEN, BabyBear::ZERO);
        out
    }
}

impl CondItem {
    /// The preimage's limbs (zero without one).
    fn pre(&self) -> [BabyBear; PRE_LIMBS] {
        self.preimage.unwrap_or([BabyBear::ZERO; PRE_LIMBS])
    }
}

/// [`crate::transaction::apo_message`] from amount limbs.
fn apo_message(omsg: &Hash, amount: &[BabyBear; AMOUNT_LIMBS]) -> Hash {
    crate::hash::sponge(domain::SIG_APO, amount, &[omsg])
}

/// A chunk's hash: a sponge with `blinding` as initial capacity, absorbing
/// each condition spend's claim, signer indices and preimage.
pub fn chunk_hash(blinding: &Hash, items: &[CondItem]) -> Hash {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[HASH_LEN..].copy_from_slice(blinding);
    for item in items {
        for block in item.bound().chunks_exact(HASH_LEN) {
            for (x, &v) in s.iter_mut().zip(block) {
                *x += v;
            }
            poseidon2::permute(&mut s);
        }
    }
    core::array::from_fn(|j| s[j])
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

impl Air for CondChunkAir {
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
        for s in 0..self.slots {
            let base = self.base(s);
            for (offset, col) in [
                (C1, PC1),
                (C2, PC2),
                (C3, PC3),
                (X, PX),
                (G1, PG1),
                (G2, PG2),
                (G3, PG3),
            ] {
                m.row_mut(base + offset)[col] = one;
            }
            for d in 0..KEY_LEVELS {
                m.row_mut(base + M1 + d)[PMERGE] = one;
            }
            for b in 0..BLOCKS {
                m.row_mut(base + B1 + b)[PB + b] = one;
            }
            for r in base..base + SLOT_ROWS {
                m.row_mut(r)[SLOTS] = one;
                if r + 1 < base + SLOT_ROWS {
                    m.row_mut(r)[SAME_SLOT] = one;
                }
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
        let one = k(1);
        let out = |j: usize| l[OUTPUT_OFFSET + j];
        let ext = |col: usize, row: &[V]| -> [V; 4] { core::array::from_fn(|b| row[col + b]) };
        let pub4 = |at: usize| -> [V; 4] { core::array::from_fn(|b| public[at + b]) };
        let (lambda, r) = (pub4(pv::LAMBDA), pub4(pv::R));
        let height = public[pv::HEIGHT];
        let slots = p[SLOTS];
        let act = l[ACT];

        poseidon2_air::eval(c, l);

        // Slot constants, and flags.
        for col in SLOT_CONSTANTS {
            c.assert_zero(p[SAME_SLOT] * (n[col] - l[col]));
        }
        for flag in [ACT, MACP, MAPO, TWO, STATE, HLK] {
            c.assert_zero(slots * l[flag] * (l[flag] - one));
        }
        c.assert_zero(slots * l[MACP] * l[MAPO]);
        c.assert_zero(slots * (l[ACT2] - act * l[TWO]));
        c.assert_zero(slots * (l[ACTH] - act * l[HLK]));
        // Only a transaction's first condition input may sign
        // anyone-can-pay or any-prevout, so no two share the signed
        // outputs.
        c.assert_zero(slots * (l[MACP] + l[MAPO]) * l[CIDX]);
        let mode = l[MACP] + l[MAPO] * BabyBear::new(2);
        let flags = l[TWO] + l[STATE] * BabyBear::new(2) + l[HLK] * BabyBear::new(4);

        // The condition's leaf: sponge(CONDITION, [state, after height,
        // after blocks, mode, flags], [key 1, key 2, hashlock]).
        let c1 = p[PC1];
        let param = [l[M], l[TL], l[DL], mode, flags];
        for j in 0..HASH_LEN {
            c.assert_zero(c1 * (l[j] - l[K1 + j]));
        }
        for (j, &v) in param.iter().enumerate() {
            c.assert_zero(c1 * (l[HASH_LEN + j] - v));
        }
        c.assert_zero(c1 * (l[13] - k(domain::CONDITION)));
        c.assert_zero(c1 * (l[14] - k(3 * HASH_LEN as u32)));
        c.assert_zero(c1 * l[15]);
        for (sel, block) in [(c1, K2), (p[PC2], HL)] {
            for j in 0..WIDTH {
                let absorbed = if j < HASH_LEN { l[block + j] } else { zero };
                c.assert_zero(sel * (n[j] - out(j) - absorbed));
            }
        }
        // C3's output is the leaf: the merges start from it.
        for j in 0..HASH_LEN {
            c.assert_zero(p[PC3] * (n[RUN + j] - out(j)));
        }

        // Merges up the output's key tree, in the order the bit gives, to
        // the claimed output hash.
        let merge = p[PMERGE];
        let bit = l[BIT];
        c.assert_zero(merge * bit * (bit - one));
        for j in 0..HASH_LEN {
            let (x, s) = (l[RUN + j], l[SIB + j]);
            c.assert_zero(merge * (l[j] - x - bit * (s - x)));
            c.assert_zero(merge * (l[HASH_LEN + j] - s - bit * (x - s)));
            c.assert_zero(merge * (n[RUN + j] - out(j) - l[j]));
        }
        let px = p[PX];
        for j in 0..HASH_LEN {
            c.assert_zero(px * (l[RUN + j] - l[CH + j]));
        }

        // The hashlock: sponge(HASHLOCK, [], [preimage]).
        for j in 0..WIDTH {
            let expected = match j {
                j if j < PRE_LIMBS => l[CPRE + j],
                13 => k(domain::HASHLOCK),
                14 => k(PRE_LIMBS as u32),
                _ => zero,
            };
            c.assert_zero(px * (l[j] - expected));
        }
        for j in 0..HASH_LEN {
            c.assert_zero(px * l[HLK] * (out(j) - l[HL + j]));
        }

        // The message: the digest, an anyone-can-pay message
        // (sponge(SIG_ACP, [position], [signed digest, commitment])) or an
        // any-prevout one (sponge(SIG_APO, [amount], [signed digest])).
        let g1 = p[PG1];
        for j in 0..WIDTH {
            let expected = match j {
                j if j < HASH_LEN => l[COMSG + j],
                8 => l[CPOS],
                9 => l[CPOS + 1],
                13 => k(domain::SIG_ACP),
                14 => k(2 * HASH_LEN as u32),
                _ => zero,
            };
            c.assert_zero(g1 * (l[j] - expected));
            let absorbed = if j < HASH_LEN { l[CC + j] } else { zero };
            c.assert_zero(g1 * (n[j] - out(j) - absorbed));
        }
        let g3 = p[PG3];
        for j in 0..WIDTH {
            let expected = match j {
                j if j < HASH_LEN => l[COMSG + j],
                j if j < HASH_LEN + AMOUNT_LIMBS => l[CAMT + j - HASH_LEN],
                13 => k(domain::SIG_APO),
                14 => k(HASH_LEN as u32),
                _ => zero,
            };
            c.assert_zero(g3 * (l[j] - expected));
        }
        for j in 0..HASH_LEN {
            c.assert_zero(p[PG2] * l[MACP] * (out(j) - l[SMSG + j]));
            c.assert_zero(g3 * l[MAPO] * (out(j) - l[SMSG + j]));
            c.assert_zero(px * (one - l[MACP] - l[MAPO]) * (l[SMSG + j] - l[CMSG + j]));
        }

        // Range checks: each row's bits make up two values.
        let bits = |at: usize, count: usize| {
            (0..count).fold(zero, |acc, b| acc + l[RB + at + b] * BabyBear::new(1 << b))
        };
        let range_rows = c1 + p[PC2] + p[PC3] + px;
        for b in 0..RB_LEN {
            c.assert_zero(range_rows * l[RB + b] * (l[RB + b] - one));
        }
        // After height: TL and height - TL fit; after blocks: DL and
        // height - created - DL do.
        c.assert_zero(c1 * (bits(0, LOCK_BITS) - l[TL]));
        c.assert_zero(c1 * (bits(LOCK_BITS, LOCK_BITS) - (height - l[TL])));
        c.assert_zero(p[PC2] * (bits(0, LOCK_BITS) - l[DL]));
        c.assert_zero(p[PC2] * (bits(LOCK_BITS, LOCK_BITS) - (height - l[CCRT] - l[DL])));
        // State rule: the state number fits, and each signer's index is
        // above it.
        c.assert_zero(p[PC3] * (bits(0, STATE_BITS) - l[M]));
        c.assert_zero(p[PC3] * (bits(STATE_BITS, STATE_BITS) - l[STATE] * (l[N1] - l[M] - one)));
        c.assert_zero(px * (bits(0, STATE_BITS) - l[STATE] * l[TWO] * (l[N2] - l[M] - one)));
        c.assert_zero(px * bits(STATE_BITS, STATE_BITS));

        // Products: the claim on X; the signatures on G1 (the first key)
        // and G2 (the second, if any).
        let mut powers = [[zero; 4]; COND_LEN];
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
        let e_claim = element(&mut col(CLAIM, COND_LEN));
        let e_sig1 = element(&mut col(K1, HASH_LEN).chain(col(SMSG, HASH_LEN)).chain([l[N1]]));
        let e_sig2 = element(&mut col(K2, HASH_LEN).chain(col(SMSG, HASH_LEN)).chain([l[N2]]));
        let e_pre = element(&mut col(CPRE, PRE_LIMBS));
        for (sel, col, flag, e) in [
            (px, PCOND, act, e_claim),
            (g1, PSIG, act, e_sig1),
            (p[PG2], PSIG, l[ACT2], e_sig2),
            (g3, PPRE, l[ACTH], e_pre),
        ] {
            let next = ext_mul(ext(col, l), factor(flag, e));
            for b in 0..4 {
                c.assert_zero(sel * (n[col + b] - next[b]));
            }
        }
        for b in 0..4 {
            c.assert_zero_transition(slots * (one - px) * (n[PCOND + b] - l[PCOND + b]));
            c.assert_zero_transition(slots * (one - g1 - p[PG2]) * (n[PSIG + b] - l[PSIG + b]));
            c.assert_zero_transition(slots * (one - g3) * (n[PPRE + b] - l[PPRE + b]));
        }

        // The chunk hash: an active slot's B rows absorb its claim, its
        // indices and its preimage.
        let mut bound: Vec<V> = col(CLAIM, COND_LEN)
            .chain([l[N1], l[N2]])
            .chain(col(CPRE, PRE_LIMBS))
            .collect();
        bound.resize(BLOCKS * HASH_LEN, zero);
        for (b, block) in bound.chunks_exact(HASH_LEN).enumerate() {
            let sel = p[PB + b];
            if b == 0 {
                for j in 0..WIDTH {
                    let absorbed = if j < HASH_LEN { block[j] } else { zero };
                    c.assert_zero(sel * act * (l[j] - l[CHS + j] - absorbed));
                }
            }
            if b + 1 < BLOCKS {
                let next_block = &bound[(b + 1) * HASH_LEN..(b + 2) * HASH_LEN];
                for j in 0..WIDTH {
                    let absorbed = if j < HASH_LEN { next_block[j] } else { zero };
                    c.assert_zero(sel * act * (n[j] - out(j) - absorbed));
                }
            }
        }
        let last = p[PB + BLOCKS - 1];
        for j in 0..WIDTH {
            let updated = act * out(j) + (one - act) * l[CHS + j];
            c.assert_zero(last * (n[CHS + j] - updated));
            c.assert_zero_transition(slots * (one - last) * (n[CHS + j] - l[CHS + j]));
        }

        // First row: empty products, zero rate (the capacity is the
        // blinding). Final row: the totals.
        let first = p[FIRST];
        for col in [PCOND, PSIG, PPRE] {
            c.assert_zero(first * (l[col] - one));
            for b in 1..4 {
                c.assert_zero(first * l[col + b]);
            }
        }
        for j in 0..HASH_LEN {
            c.assert_zero(first * l[CHS + j]);
        }
        let fin = p[FINAL];
        for b in 0..4 {
            c.assert_zero(fin * (l[PCOND + b] - public[pv::P_COND + b]));
            c.assert_zero(fin * (l[PSIG + b] - public[pv::P_SIG + b]));
            c.assert_zero(fin * (l[PPRE + b] - public[pv::P_PRE + b]));
        }
        for j in 0..HASH_LEN {
            c.assert_zero(fin * (l[CHS + j] - public[pv::HASH + j]));
        }
    }
}

/// A dummy condition spend for a spare slot: a one-key condition at leaf 0
/// of a tree of nothing else, created at height 0, claiming what it meets.
fn dummy() -> CondItem {
    let condition = Condition::key([BabyBear::ZERO; HASH_LEN]);
    let path = [[BabyBear::ZERO; HASH_LEN]; KEY_LEVELS];
    let hash = crate::transaction::key_root(&condition.leaf(), 0, &path);
    CondItem {
        claim: CondClaim {
            hash,
            commitment: [BabyBear::ZERO; HASH_LEN],
            position: 0,
            amount: [BabyBear::ZERO; AMOUNT_LIMBS],
            created: 0,
            msg: [BabyBear::ZERO; HASH_LEN],
            omsg: [BabyBear::ZERO; HASH_LEN],
            index: 0,
        },
        condition,
        leaf: 0,
        path,
        indices: vec![0],
        preimage: None,
    }
}

impl CondChunkAir {
    /// The trace and public values checking `items` (at most the capacity)
    /// for the block at `height`, with the chunk hash blinded by
    /// `blinding`. Items that don't meet their conditions give a trace
    /// that fails the constraints.
    ///
    /// # Panics
    /// If there are too many items, or an item has no index per key.
    pub fn trace(
        &self,
        items: &[CondItem],
        height: u32,
        blinding: &Hash,
        lambda: BabyBear4,
        r: BabyBear4,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        assert!(items.len() <= self.slots, "conditions don't fit the chunk");
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let mut chs = [BabyBear::ZERO; WIDTH];
        chs[HASH_LEN..].copy_from_slice(blinding);
        let (mut pcond, mut psig, mut ppre) = (BabyBear4::ONE, BabyBear4::ONE, BabyBear4::ONE);
        let zeros = [BabyBear::ZERO; WIDTH];
        let put_bits = |row: &mut [BabyBear], at: usize, v: u32, count: usize| {
            for b in 0..count {
                row[RB + at + b] = BabyBear::new((v >> b) & 1);
            }
        };
        for s in 0..self.slots {
            let base = self.base(s);
            let (item, active) = match items.get(s) {
                Some(item) => (item.clone(), true),
                None => (dummy(), false),
            };
            let cond = &item.condition;
            assert_eq!(item.indices.len(), cond.keys.len(), "an index per key");
            let two = cond.keys.len() == 2;
            let zero_hash = [BabyBear::ZERO; HASH_LEN];
            let k2 = *cond.keys.get(1).unwrap_or(&zero_hash);
            let hl = cond.hashlock.unwrap_or(zero_hash);
            let msg = item.message();
            let (n1, n2) = (item.indices[0], item.indices.get(1).copied().unwrap_or(0));

            // The slot's constants, on every row.
            for row in base..base + SLOT_ROWS {
                let row = m.row_mut(row);
                row[ACT] = BabyBear::from(active);
                row[ACT2] = BabyBear::from(active && two);
                row[ACTH] = BabyBear::from(active && cond.hashlock.is_some());
                row[CPRE..CPRE + PRE_LIMBS].copy_from_slice(&item.pre());
                row[CLAIM..CLAIM + COND_LEN].copy_from_slice(&item.claim.elements());
                row[K1..K1 + HASH_LEN].copy_from_slice(&cond.keys[0]);
                row[K2..K2 + HASH_LEN].copy_from_slice(&k2);
                row[HL..HL + HASH_LEN].copy_from_slice(&hl);
                row[M] = BabyBear::new(cond.min_state.unwrap_or(0));
                row[TL] = BabyBear::new(cond.after_height);
                row[DL] = BabyBear::new(cond.after_blocks);
                row[MACP] = BabyBear::from(cond.mode == SigMode::AnyoneCanPay);
                row[MAPO] = BabyBear::from(cond.mode == SigMode::AnyPrevout);
                row[TWO] = BabyBear::from(two);
                row[STATE] = BabyBear::from(cond.min_state.is_some());
                row[HLK] = BabyBear::from(cond.hashlock.is_some());
                row[N1] = BabyBear::new(n1);
                row[N2] = BabyBear::new(n2);
                row[SMSG..SMSG + HASH_LEN].copy_from_slice(&msg);
            }

            // The leaf.
            let mut input = zeros;
            input[..HASH_LEN].copy_from_slice(&cond.keys[0]);
            let param = [
                cond.min_state.unwrap_or(0),
                cond.after_height,
                cond.after_blocks,
                cond.mode as u32,
                two as u32
                    | (cond.min_state.is_some() as u32) << 1
                    | (cond.hashlock.is_some() as u32) << 2,
            ];
            for (j, &v) in param.iter().enumerate() {
                input[HASH_LEN + j] = BabyBear::new(v);
            }
            input[13] = BabyBear::new(domain::CONDITION);
            input[14] = BabyBear::new(3 * HASH_LEN as u32);
            let mut state = poseidon2_air::generate(input, m.row_mut(base + C1));
            for (offset, block) in [(C2, &k2), (C3, &hl)] {
                let mut input = state;
                for (x, &v) in input.iter_mut().zip(block) {
                    *x += v;
                }
                state = poseidon2_air::generate(input, m.row_mut(base + offset));
            }
            let mut run: Hash = core::array::from_fn(|j| state[j]);
            debug_assert_eq!(run, cond.leaf());

            // The merges.
            for (d, sib) in item.path.iter().enumerate() {
                let bit = item.leaf >> d & 1 == 1;
                let (left, right) = if bit { (sib, &run) } else { (&run, sib) };
                let mut input = zeros;
                input[..HASH_LEN].copy_from_slice(left);
                input[HASH_LEN..].copy_from_slice(right);
                let row = m.row_mut(base + M1 + d);
                row[BIT] = BabyBear::from(bit);
                row[SIB..SIB + HASH_LEN].copy_from_slice(sib);
                row[RUN..RUN + HASH_LEN].copy_from_slice(&run);
                let out = poseidon2_air::generate(input, row);
                run = core::array::from_fn(|j| out[j] + input[j]);
            }
            m.row_mut(base + X)[RUN..RUN + HASH_LEN].copy_from_slice(&run);

            // The preimage's hash.
            let mut input = zeros;
            input[..PRE_LIMBS].copy_from_slice(&item.pre());
            input[13] = BabyBear::new(domain::HASHLOCK);
            input[14] = BabyBear::new(PRE_LIMBS as u32);
            poseidon2_air::generate(input, m.row_mut(base + X));

            // The messages.
            let c = &item.claim;
            let (lo, hi) = crate::state::split(c.position);
            let mut input = zeros;
            input[..HASH_LEN].copy_from_slice(&c.omsg);
            input[8] = lo;
            input[9] = hi;
            input[13] = BabyBear::new(domain::SIG_ACP);
            input[14] = BabyBear::new(2 * HASH_LEN as u32);
            let mut state = poseidon2_air::generate(input, m.row_mut(base + G1));
            for (x, &v) in state.iter_mut().zip(&c.commitment) {
                *x += v;
            }
            poseidon2_air::generate(state, m.row_mut(base + G2));
            let mut input = zeros;
            input[..HASH_LEN].copy_from_slice(&c.omsg);
            input[HASH_LEN..HASH_LEN + AMOUNT_LIMBS].copy_from_slice(&c.amount);
            input[13] = BabyBear::new(domain::SIG_APO);
            input[14] = BabyBear::new(HASH_LEN as u32);
            poseidon2_air::generate(input, m.row_mut(base + G3));

            // The chunk hash.
            let bound = item.bound();
            let mut state = chs;
            for (b, block) in bound.chunks_exact(HASH_LEN).enumerate() {
                let mut input = if active { state } else { zeros };
                if active {
                    for (x, &v) in input.iter_mut().zip(block) {
                        *x += v;
                    }
                }
                state = poseidon2_air::generate(input, m.row_mut(base + B1 + b));
            }

            // Range checks.
            let sub = |a: u32, b: u32| a.wrapping_sub(b);
            {
                let row = m.row_mut(base + C1);
                put_bits(row, 0, cond.after_height, LOCK_BITS);
                put_bits(row, LOCK_BITS, sub(height, cond.after_height), LOCK_BITS);
            }
            {
                let row = m.row_mut(base + C2);
                put_bits(row, 0, cond.after_blocks, LOCK_BITS);
                let d = sub(sub(height, c.created), cond.after_blocks);
                put_bits(row, LOCK_BITS, d, LOCK_BITS);
            }
            let min = cond.min_state.unwrap_or(0);
            let above = |n: u32, on: bool| if on { sub(sub(n, min), 1) } else { 0 };
            {
                let row = m.row_mut(base + C3);
                put_bits(row, 0, min, STATE_BITS);
                put_bits(
                    row,
                    STATE_BITS,
                    above(n1, cond.min_state.is_some()),
                    STATE_BITS,
                );
            }
            put_bits(
                m.row_mut(base + X),
                0,
                above(n2, cond.min_state.is_some() && two),
                STATE_BITS,
            );

            // Carried values: the products change after X (claim), G1 and
            // G2 (signatures), the sponge after the last B row.
            let triples = item.signatures();
            let new_pcond = if active {
                pcond * (r - cond_element(lambda, &item.claim))
            } else {
                pcond
            };
            let sig = |t: &(Hash, Hash, u32)| r - sig_element(lambda, &t.0, &t.1, t.2);
            let psig1 = if active {
                psig * sig(&triples[0])
            } else {
                psig
            };
            let psig2 = if active && two {
                psig1 * sig(&triples[1])
            } else {
                psig1
            };
            let new_ppre = if active && cond.hashlock.is_some() {
                ppre * (r - preimage_element(lambda, &item.pre()))
            } else {
                ppre
            };
            for offset in 0..SLOT_ROWS {
                let row = m.row_mut(base + offset);
                row[CHS..CHS + WIDTH].copy_from_slice(&chs);
                row[PCOND..PCOND + 4]
                    .copy_from_slice(&if offset > X { new_pcond } else { pcond }.0);
                let p = match offset {
                    o if o <= G1 => psig,
                    G2 => psig1,
                    _ => psig2,
                };
                row[PSIG..PSIG + 4].copy_from_slice(&p.0);
                let q = if offset > G3 { new_ppre } else { ppre };
                row[PPRE..PPRE + 4].copy_from_slice(&q.0);
            }
            pcond = new_pcond;
            psig = psig2;
            ppre = new_ppre;
            if active {
                chs = state;
            }
        }
        let fin = self.final_row();
        for row in fin..m.height() {
            poseidon2_air::generate(zeros, m.row_mut(row));
        }
        {
            let row = m.row_mut(fin);
            row[CHS..CHS + WIDTH].copy_from_slice(&chs);
            row[PCOND..PCOND + 4].copy_from_slice(&pcond.0);
            row[PSIG..PSIG + 4].copy_from_slice(&psig.0);
            row[PPRE..PPRE + 4].copy_from_slice(&ppre.0);
        }
        let mut public = Vec::with_capacity(pv::LEN);
        public.extend_from_slice(&lambda.0);
        public.extend_from_slice(&r.0);
        public.push(BabyBear::new(height));
        public.extend_from_slice(&pcond.0);
        public.extend_from_slice(&psig.0);
        public.extend_from_slice(&ppre.0);
        public.extend_from_slice(&chs[..HASH_LEN]);
        (m, public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::fri::FriParams;
    use crate::item_chunk_air::product;
    use crate::stark;
    use crate::transaction::{KeyTree, SignerTree, hashlock};

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);
    const BLINDING: Hash = BabyBear::new_array([29, 31, 37, 41, 43, 47, 53, 59]);
    const HEIGHT: u32 = 1000;

    fn h(n: u32) -> Hash {
        BabyBear::new_array(core::array::from_fn(|i| n * 100 + i as u32))
    }

    /// A spend of `condition`, at leaf 3 of a tree of made-up hashes,
    /// created at height `created`, with indices `indices`.
    fn spend(
        condition: Condition,
        created: u32,
        indices: Vec<u32>,
        pre: Option<[u8; 15]>,
    ) -> CondItem {
        let mut tree = KeyTree(core::array::from_fn(|i| h(i as u32 + 7)));
        tree.0[3] = condition.leaf();
        let pre_limbs = pre.map(|p| crate::transaction::preimage_limbs(&p));
        CondItem {
            claim: CondClaim {
                hash: tree.root(),
                commitment: h(1),
                position: 77,
                amount: crate::output::amount_limbs(5000),
                created,
                msg: h(2),
                omsg: h(3),
                index: 0,
            },
            condition,
            leaf: 3,
            path: tree.path(3),
            indices,
            preimage: pre_limbs,
        }
    }

    fn channel() -> Condition {
        let a = SignerTree::new((0..8).map(h).collect());
        let b = SignerTree::new((8..16).map(h).collect());
        Condition {
            keys: vec![a.root(), b.root()],
            min_state: Some(4),
            mode: SigMode::AnyPrevout,
            after_height: 0,
            after_blocks: 0,
            hashlock: None,
        }
    }

    fn htlc(pre: &[u8; 15]) -> Condition {
        Condition {
            keys: vec![h(50)],
            min_state: None,
            mode: SigMode::AnyoneCanPay,
            after_height: 900,
            after_blocks: 10,
            hashlock: Some(hashlock(pre)),
        }
    }

    fn run(items: &[CondItem]) -> (Result<(), usize>, Vec<BabyBear>) {
        let air = CondChunkAir { slots: 3 };
        let (t, public) = air.trace(items, HEIGHT, &BLINDING, LAMBDA, R);
        (check_constraints(&air, &t, &public), public)
    }

    #[test]
    fn checks_conditions_and_lists_their_signatures() {
        let pre = [9u8; 15];
        let items = [
            spend(channel(), 5, vec![5, 7], None),
            spend(htlc(&pre), 990, vec![0], Some(pre)),
            spend(Condition::key(h(60)), 3, vec![2], None),
        ];
        let (result, public) = run(&items);
        assert_eq!(result, Ok(()));
        let p_cond = product(R, items.iter().map(|i| cond_element(LAMBDA, &i.claim)));
        let p_sig = product(
            R,
            items
                .iter()
                .flat_map(|i| i.signatures())
                .map(|(k, m, n)| sig_element(LAMBDA, &k, &m, n)),
        );
        assert_eq!(&public[pv::P_COND..pv::P_COND + 4], &p_cond.0);
        assert_eq!(&public[pv::P_SIG..pv::P_SIG + 4], &p_sig.0);
        let p_pre = product(
            R,
            [preimage_element(
                LAMBDA,
                &crate::transaction::preimage_limbs(&pre),
            )],
        );
        assert_eq!(&public[pv::P_PRE..pv::P_PRE + 4], &p_pre.0);
        assert_eq!(&public[pv::HASH..pv::LEN], &chunk_hash(&BLINDING, &items));
        // The messages are as the transaction defines them.
        let c = &items[1].claim;
        assert_eq!(
            items[1].message(),
            crate::transaction::acp_message(&c.omsg, &c.commitment, c.position)
        );
        assert_eq!(
            items[0].message(),
            crate::transaction::apo_message(&items[0].claim.omsg, 5000)
        );
        assert_eq!(items[2].message(), items[2].claim.msg);
    }

    #[test]
    fn rejects_unmet_conditions() {
        let pre = [9u8; 15];
        let fails = |item: CondItem| run(&[item]).0.is_err();
        // A state not above the condition's, for either signer.
        assert!(fails(spend(channel(), 5, vec![4, 7], None)));
        assert!(fails(spend(channel(), 5, vec![5, 3], None)));
        // Too early: the height lock, then the block count.
        let mut early = htlc(&pre);
        early.after_height = HEIGHT + 1;
        assert!(fails(spend(early, 0, vec![0], Some(pre))));
        assert!(fails(spend(htlc(&pre), 991, vec![0], Some(pre))));
        assert!(!fails(spend(htlc(&pre), 990, vec![0], Some(pre))));
        // A wrong or missing preimage.
        assert!(fails(spend(htlc(&pre), 0, vec![0], Some([8; 15]))));
        assert!(fails(spend(htlc(&pre), 0, vec![0], None)));
        // Anyone-can-pay or any-prevout on a later condition input.
        let mut item = spend(htlc(&pre), 0, vec![0], Some(pre));
        item.claim.index = 1;
        assert!(fails(item));
        let mut item = spend(Condition::key(h(60)), 3, vec![2], None);
        item.claim.index = 1;
        assert!(!fails(item));
        // A condition that isn't the output's.
        let mut item = spend(channel(), 5, vec![5, 7], None);
        item.condition.min_state = Some(3);
        assert!(fails(item));
        let mut item = spend(Condition::key(h(60)), 3, vec![2], None);
        item.claim.hash = h(61);
        assert!(fails(item));
    }

    #[test]
    fn rejects_wrong_public_values() {
        let items = [spend(channel(), 5, vec![5, 7], None)];
        let air = CondChunkAir { slots: 1 };
        let (t, public) = air.trace(&items, HEIGHT, &BLINDING, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        for at in [pv::P_COND, pv::P_SIG + 1, pv::HASH + 7] {
            let mut bad = public.clone();
            bad[at] += BabyBear::ONE;
            assert!(check_constraints(&air, &t, &bad).is_err(), "public {at}");
        }
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let air = CondChunkAir { slots: 2 };
        let (t, public) = air.trace(
            &[spend(channel(), 5, vec![5, 7], None)],
            HEIGHT,
            &BLINDING,
            LAMBDA,
            R,
        );
        let (pk, vk) = stark::setup(&params, &air);
        let proof = stark::prove(&params, &pk, &air, t, &public);
        assert_eq!(stark::verify(&params, &vk, &air, &public, &proof), Ok(()));
    }
}
