//! The condition chunk circuit: checks a run of condition spends
//! ([`crate::transaction::Condition`]), revealing only multiset
//! fingerprints of what each was checked against, of the signatures it
//! needs and of the preimages it reveals, and a blinded hash of them.
//!
//! Each slot takes a condition input's claim ([`CondClaim`]: its output's
//! hash, commitment, position, amount and creation height, its
//! transaction's digest and signed-outputs digest, and how many condition
//! inputs come before it), which the aggregation matches against the item
//! chunks' (`P_cond`), and the condition it meets. The slot hashes the
//! condition into its key-tree leaf and walks the leaf's path to the
//! output's hash, then checks:
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
//! - the signers: exactly the condition's threshold of them (with
//!   protection, at least it), each a distinct member of its key set
//!   (members in increasing order, each member's key and payout walked to
//!   the set's root), each key's index above the state number if the
//!   condition has one;
//! - with protection, every member: each signs or is paid its payout.
//!
//! It then adds `(key root, message, index)` for each signer to `P_sig`,
//! on the transaction side, which the signature chunks
//! ([`crate::sig_chunk_air`]) must match, and each unpaid member's
//! payout (the transaction's digest and the commitment:
//! [`payout_element`]) to `P_pre`, which the item chunks' payout outputs
//! must match.
//!
//! Rows per slot ([`SLOT_ROWS`]): three hashing the condition, eight
//! merges up the output's key tree, one hashing the preimage, three for
//! the messages, [`BLOCKS`] absorbing the slot's claim, preimage and
//! signers into the chunk's hash (a sponge whose initial capacity is a
//! random blinding), then [`MAX_SIGNERS`] member regions, used or not,
//! each a row merging the member's key with its payout and
//! [`KEY_SET_LEVELS`] merge rows up the set. Without protection the
//! regions are the signers; with it, every member in order. Range checks
//! (timelocks to [`MAX_HEIGHT_BITS`] bits, state numbers to
//! [`MAX_KEY_DEPTH`]) use bit columns. Each slot has an active flag;
//! inactive slots contribute nothing.
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
    COND_LEN, CondClaim, PRE_LIMBS, cond_element, payout_element, preimage_element, sig_element,
};
use crate::matrix::RowMajorMatrix;
use crate::output::AMOUNT_LIMBS;
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::transaction::{
    Condition, KEY_LEVELS, KEY_SET_LEVELS, KeyPath, MAX_KEY_DEPTH, MAX_SIGNERS, SigMode,
};

const LOCK_BITS: usize = MAX_HEIGHT_BITS as usize;
const STATE_BITS: usize = MAX_KEY_DEPTH;
/// Bits of a member's gap above the previous one's (`0..=16`).
const GAP_BITS: usize = KEY_SET_LEVELS + 1;
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
/// Blocks of the chunk hash per slot: the claim, the preimage, and each
/// signer's index and member.
const BLOCKS: usize = (COND_LEN + PRE_LIMBS + MAX_SIGNERS).div_ceil(HASH_LEN);
/// The members' regions: a leaf row, then the merges.
const S1: usize = B1 + BLOCKS;
const REGION_ROWS: usize = 1 + KEY_SET_LEVELS;
/// Rows per slot.
pub const SLOT_ROWS: usize = S1 + MAX_SIGNERS * REGION_ROWS;

// Columns after the permutation's.
const P: usize = poseidon2_air::NUM_COLS;
/// The slot's active flag, and active with a hashlock, constant in the
/// slot.
const ACT: usize = P;
const ACTH: usize = ACT + 1;
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
/// The condition: key set, hashlock, threshold, state number, locks, mode
/// and flags.
const KSET: usize = CPRE + PRE_LIMBS;
const HL: usize = KSET + HASH_LEN;
const K: usize = HL + HASH_LEN;
const M: usize = K + 1;
const TL: usize = M + 1;
const DL: usize = TL + 1;
const MACP: usize = DL + 1;
const MAPO: usize = MACP + 1;
const STATE: usize = MAPO + 1;
const HLK: usize = STATE + 1;
/// Protection, and the key count.
const PROT: usize = HLK + 1;
const N: usize = PROT + 1;
/// The message the signers sign.
const SMSG: usize = N + 1;
/// Each region's `(index * 2 + signs) * 16 + member` (zero if inactive),
/// for the chunk hash, which absorbs them before the regions.
const SIGNED: usize = SMSG + HASH_LEN;
/// Columns constant in a slot.
const SLOT_CONSTANTS: std::ops::Range<usize> = ACT..SIGNED + MAX_SIGNERS;
/// A member, constant in its region: active, key root, index, member,
/// signs, payout.
const SA: usize = SIGNED + MAX_SIGNERS;
const KR: usize = SA + 1;
const NJ: usize = KR + HASH_LEN;
const MEM: usize = NJ + 1;
const SG: usize = MEM + 1;
const PAY: usize = SG + 1;
const SIGNER_CONSTANTS: std::ops::Range<usize> = SA..PAY + HASH_LEN;
/// Merge rows: the path's bit and sibling; the running node; the member
/// position so far (signer regions).
const BIT: usize = PAY + HASH_LEN;
const SIB: usize = BIT + 1;
const RUN: usize = SIB + HASH_LEN;
const ACC: usize = RUN + HASH_LEN;
/// Carried through a slot's regions: signers so far, and the lowest
/// member the next may be.
const CNT: usize = ACC + 1;
const NEXTM: usize = CNT + 1;
/// Carried: the chunk-hash sponge and the three products.
const CHS: usize = NEXTM + 1;
const PCOND: usize = CHS + WIDTH;
const PSIG: usize = PCOND + 4;
const PPRE: usize = PSIG + 4;
/// Range-check bits.
const RB: usize = PPRE + 4;
const RB_LEN: usize = 2 * LOCK_BITS;
pub const NUM_COLS: usize = RB + RB_LEN;
/// Bits of a count (`0..=16`).
const COUNT_BITS: usize = KEY_SET_LEVELS + 1;
const _: () = assert!(STATE_BITS + GAP_BITS + 2 * COUNT_BITS <= RB_LEN);
const _: () = assert!(STATE_BITS + LOCK_BITS <= RB_LEN);

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
/// Regions: every merge row; the first; rows followed by one of the same
/// region (the leaf row too); the last; the last but in the slot's last
/// region; the slot's first region's first merge row; the slot's last
/// region's last row; a row's member-bit weight.
const PS: usize = PB + BLOCKS;
const PS_FIRST: usize = PS + 1;
const PS_NEXT: usize = PS + 2;
const PS_LAST: usize = PS + 3;
const PS_LAST_MORE: usize = PS + 4;
const PS_START: usize = PS + 5;
const PS_END: usize = PS + 6;
const PS_WEIGHT: usize = PS + 7;
/// Region `j`'s merge rows: one column each.
const PS_SIGNER: usize = PS + 8;
/// The regions' leaf rows.
const PSL: usize = PS_SIGNER + MAX_SIGNERS;
const SAME_SLOT: usize = PSL + 1;
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
/// its signers (the member of the key set each is, and its key's index;
/// members ascending) and its hashlock's preimage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CondItem {
    pub claim: CondClaim,
    pub condition: Condition,
    pub leaf: u8,
    pub path: KeyPath,
    pub signers: Vec<(u8, u32)>,
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
        self.signers
            .iter()
            .map(|&(m, i)| (self.key(m), msg, i))
            .collect()
    }

    /// Its regions' members, in order, each with its index if it signs:
    /// the signers, or with protection every member.
    pub fn members(&self) -> Members {
        if self.condition.payouts.is_none() {
            return self.signers.iter().map(|&(m, i)| (m, Some(i))).collect();
        }
        (0..self.condition.keys.len() as u8)
            .map(|m| (m, self.signers.iter().find(|s| s.0 == m).map(|s| s.1)))
            .collect()
    }

    /// The payouts it requires: each non-signing member's, with
    /// protection.
    pub fn payouts(&self) -> Vec<Hash> {
        self.members()
            .into_iter()
            .filter(|m| m.1.is_none())
            .map(|m| self.condition.payout(m.0 as usize))
            .collect()
    }

    /// Each region's `(index * 2 + signs) * 16 + member` (zero past the
    /// members).
    fn copies(&self) -> [BabyBear; MAX_SIGNERS] {
        copies_of(&self.members())
    }

    /// Member `m`'s key root (zero past the keys).
    fn key(&self, m: u8) -> Hash {
        self.condition
            .keys
            .get(m as usize)
            .copied()
            .unwrap_or([BabyBear::ZERO; HASH_LEN])
    }

    /// The preimage's limbs (zero without one).
    fn pre(&self) -> [BabyBear; PRE_LIMBS] {
        self.preimage.unwrap_or([BabyBear::ZERO; PRE_LIMBS])
    }

    /// What the chunk hash absorbs for it: the claim, the preimage, each
    /// region's member, whether it signs and its index ([`Self::copies`]).
    fn bound(&self) -> Vec<BabyBear> {
        let mut out = self.claim.elements().to_vec();
        out.extend(self.pre());
        out.extend(self.copies());
        out.resize(BLOCKS * HASH_LEN, BabyBear::ZERO);
        out
    }
}

/// A spend's regions: each member, with its index if it signs.
type Members = Vec<(u8, Option<u32>)>;

/// [`CondItem::copies`] for `members`.
fn copies_of(members: &[(u8, Option<u32>)]) -> [BabyBear; MAX_SIGNERS] {
    core::array::from_fn(|j| {
        members.get(j).map_or(BabyBear::ZERO, |&(m, i)| {
            let code = i.map_or(0, |i| i * 2 + 1);
            BabyBear::new(code * MAX_SIGNERS as u32 + m as u32)
        })
    })
}

/// [`crate::transaction::apo_message`] from amount limbs.
fn apo_message(omsg: &Hash, amount: &[BabyBear; AMOUNT_LIMBS]) -> Hash {
    crate::hash::sponge(domain::SIG_APO, amount, &[omsg])
}

/// A chunk's hash: a sponge with `blinding` as initial capacity, absorbing
/// each condition spend's claim, preimage and signers.
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
            for j in 0..MAX_SIGNERS {
                let leaf = base + S1 + j * REGION_ROWS;
                m.row_mut(leaf)[PSL] = one;
                m.row_mut(leaf)[PS_NEXT] = one;
                let first = leaf + 1;
                for d in 0..KEY_SET_LEVELS {
                    let row = m.row_mut(first + d);
                    row[PS] = one;
                    row[PS_SIGNER + j] = one;
                    row[PS_WEIGHT] = BabyBear::new(1 << d);
                    if d + 1 < KEY_SET_LEVELS {
                        row[PS_NEXT] = one;
                    }
                }
                m.row_mut(first)[PS_FIRST] = one;
                let last = first + KEY_SET_LEVELS - 1;
                m.row_mut(last)[PS_LAST] = one;
                if j + 1 < MAX_SIGNERS {
                    m.row_mut(last)[PS_LAST_MORE] = one;
                } else {
                    m.row_mut(last)[PS_END] = one;
                }
                if j == 0 {
                    m.row_mut(first)[PS_START] = one;
                }
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

        // Slot and signer constants, and flags.
        for col in SLOT_CONSTANTS {
            c.assert_zero(p[SAME_SLOT] * (n[col] - l[col]));
        }
        for col in SIGNER_CONSTANTS {
            c.assert_zero(p[PS_NEXT] * (n[col] - l[col]));
        }
        for flag in [ACT, MACP, MAPO, STATE, HLK, PROT] {
            c.assert_zero(slots * l[flag] * (l[flag] - one));
        }
        c.assert_zero(slots * l[MACP] * l[MAPO]);
        c.assert_zero(slots * (l[ACTH] - act * l[HLK]));
        // Only a transaction's first condition input may sign
        // anyone-can-pay or any-prevout, so no two share the signed
        // outputs.
        c.assert_zero(slots * (l[MACP] + l[MAPO]) * l[CIDX]);
        let mode = l[MACP] + l[MAPO] * BabyBear::new(2);
        let flags =
            l[STATE] * BabyBear::new(2) + l[HLK] * BabyBear::new(4) + l[PROT] * BabyBear::new(8);
        let prot = l[PROT];

        // The condition's leaf: sponge(CONDITION, [state, after height,
        // after blocks, mode, flags], [key set, hashlock, threshold, key
        // count]).
        let c1 = p[PC1];
        let param = [l[M], l[TL], l[DL], mode, flags];
        for j in 0..HASH_LEN {
            c.assert_zero(c1 * (l[j] - l[KSET + j]));
        }
        for (j, &v) in param.iter().enumerate() {
            c.assert_zero(c1 * (l[HASH_LEN + j] - v));
        }
        c.assert_zero(c1 * (l[13] - k(domain::CONDITION)));
        c.assert_zero(c1 * (l[14] - k(2 * HASH_LEN as u32 + 2)));
        c.assert_zero(c1 * l[15]);
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { l[HL + j] } else { zero };
            c.assert_zero(c1 * (n[j] - out(j) - absorbed));
            let absorbed = match j {
                0 => l[K],
                1 => l[N],
                _ => zero,
            };
            c.assert_zero(p[PC2] * (n[j] - out(j) - absorbed));
        }
        // C3's output is the leaf: the merges start from it.
        for j in 0..HASH_LEN {
            c.assert_zero(p[PC3] * (n[RUN + j] - out(j)));
        }

        // Merges, up the output's key tree (to the claimed output hash)
        // and up the key set (from each member's key and payout to the
        // set's root), in the order the bit gives. A region's leaf row
        // merges the key with the payout, and the merges start from it.
        let merging = p[PMERGE] + p[PS];
        let bit = l[BIT];
        c.assert_zero(merging * bit * (bit - one));
        let cur: [V; HASH_LEN] = core::array::from_fn(|j| out(j) + l[j]);
        for j in 0..HASH_LEN {
            let (x, s) = (l[RUN + j], l[SIB + j]);
            c.assert_zero(merging * (l[j] - x - bit * (s - x)));
            c.assert_zero(merging * (l[HASH_LEN + j] - s - bit * (x - s)));
            c.assert_zero((p[PMERGE] + p[PS_NEXT]) * (n[RUN + j] - cur[j]));
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

        // Range checks: each row's bits make up its values.
        let bits = |at: usize, count: usize| {
            (0..count).fold(zero, |acc, b| acc + l[RB + at + b] * BabyBear::new(1 << b))
        };
        let range_rows = c1 + p[PC2] + p[PC3] + p[PS_LAST];
        for b in 0..RB_LEN {
            c.assert_zero(range_rows * l[RB + b] * (l[RB + b] - one));
        }
        // After height: TL and height - TL fit; after blocks: DL and
        // height - created - DL do; the state number fits.
        c.assert_zero(c1 * (bits(0, LOCK_BITS) - l[TL]));
        c.assert_zero(c1 * (bits(LOCK_BITS, LOCK_BITS) - (height - l[TL])));
        c.assert_zero(p[PC2] * (bits(0, LOCK_BITS) - l[DL]));
        c.assert_zero(p[PC2] * (bits(LOCK_BITS, LOCK_BITS) - (height - l[CCRT] - l[DL])));
        // Protection needs a block count of at least one.
        c.assert_zero(p[PC3] * (bits(0, STATE_BITS) - l[M]));
        c.assert_zero(p[PC3] * (bits(STATE_BITS, LOCK_BITS) - prot * (l[DL] - one)));
        let rest = STATE_BITS + LOCK_BITS;
        c.assert_zero(p[PC3] * bits(rest, RB_LEN - rest));

        // Members: an active one's key and payout walk to the key set's
        // root, its member is its path's bits, above the previous one's,
        // and if it signs, its index is above the state number. Active
        // members come first, the first must be, and inactive slots have
        // none. Without protection every active member signs and they
        // number the threshold; with it they're every member, in order,
        // and those signing number at least the threshold (at least one).
        let (sa, sg) = (l[SA], l[SG]);
        c.assert_zero(p[PS] * sa * (sa - one));
        c.assert_zero(p[PS] * sa * (one - act));
        c.assert_zero(p[PS] * sg * (sg - one));
        c.assert_zero(p[PS] * sg * (one - sa));
        c.assert_zero(p[PS] * (one - prot) * (sa - sg));
        c.assert_zero(p[PS_START] * act * (one - sa));
        c.assert_zero(p[PS_LAST_MORE] * n[SA] * (one - sa));
        for j in 0..HASH_LEN {
            c.assert_zero(p[PSL] * (l[j] - l[KR + j]));
            c.assert_zero(p[PSL] * (l[HASH_LEN + j] - l[PAY + j]));
            c.assert_zero(p[PS_LAST] * sa * (cur[j] - l[KSET + j]));
        }
        c.assert_zero(p[PS_FIRST] * l[ACC]);
        c.assert_zero(p[PS_NEXT] * (n[ACC] - l[ACC] - bit * p[PS_WEIGHT]));
        let last_weight = BabyBear::new(1 << (KEY_SET_LEVELS - 1));
        c.assert_zero(p[PS_LAST] * (l[MEM] - l[ACC] - bit * last_weight));
        c.assert_zero(p[PS_START] * l[CNT]);
        c.assert_zero(p[PS_START] * l[NEXTM]);
        for col in [CNT, NEXTM] {
            c.assert_zero(p[PS_NEXT] * (n[col] - l[col]));
        }
        c.assert_zero(p[PS_LAST_MORE] * (n[CNT] - l[CNT] - sg));
        c.assert_zero(p[PS_LAST_MORE] * (n[NEXTM] - l[NEXTM] - sa * (l[MEM] + one - l[NEXTM])));
        let ps_last = p[PS_LAST];
        let ps_end = p[PS_END];
        let signing = l[CNT] + sg;
        c.assert_zero(ps_end * act * (one - prot) * (signing - l[K]));
        c.assert_zero(ps_last * prot * sa * (l[MEM] - l[NEXTM]));
        c.assert_zero(ps_end * act * prot * (l[NEXTM] + sa - l[N]));
        c.assert_zero(ps_last * (bits(0, STATE_BITS) - sg * l[STATE] * (l[NJ] - l[M] - one)));
        c.assert_zero(ps_last * (bits(STATE_BITS, GAP_BITS) - sa * (l[MEM] - l[NEXTM])));
        let counts = STATE_BITS + GAP_BITS;
        c.assert_zero(ps_end * (bits(counts, COUNT_BITS) - act * prot * (signing - l[K])));
        c.assert_zero(ps_end * (bits(counts + COUNT_BITS, COUNT_BITS) - act * prot * (l[K] - one)));
        c.assert_zero((ps_last - ps_end) * bits(counts, RB_LEN - counts));
        let rest = counts + 2 * COUNT_BITS;
        c.assert_zero(ps_end * bits(rest, RB_LEN - rest));

        // Products: the claim on X, the preimage on G3, each member's
        // signature or payout on its region's first merge row.
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
        let e_sig = element(&mut col(KR, HASH_LEN).chain(col(SMSG, HASH_LEN)).chain([l[NJ]]));
        let e_pre = element(&mut col(CPRE, PRE_LIMBS));
        let e_pay = element(&mut col(CMSG, HASH_LEN).chain(col(PAY, HASH_LEN)).chain([one]));
        for (sel, col, flag, e) in [
            (px, PCOND, act, e_claim),
            (g3, PPRE, l[ACTH], e_pre),
            (p[PS_FIRST], PSIG, sg, e_sig),
            (p[PS_FIRST], PPRE, sa - sg, e_pay),
        ] {
            let next = ext_mul(ext(col, l), factor(flag, e));
            for b in 0..4 {
                c.assert_zero(sel * (n[col + b] - next[b]));
            }
        }
        for (col, sel) in [(PCOND, px), (PPRE, g3 + p[PS_FIRST]), (PSIG, p[PS_FIRST])] {
            for b in 0..4 {
                c.assert_zero_transition(slots * (one - sel) * (n[col + b] - l[col + b]));
            }
        }

        // The chunk hash: an active slot's B rows absorb its claim, its
        // preimage and its signers.
        let mut bound: Vec<V> = col(CLAIM, COND_LEN).chain(col(CPRE, PRE_LIMBS)).collect();
        // Each signer's index and member: the slot's copies, which each
        // signer's region checks.
        bound.extend(col(SIGNED, MAX_SIGNERS));
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
        // The copies are each member's `(index * 2 + signs) * 16 + member`
        // (zero when inactive), checked on its region.
        let m16 = BabyBear::new(MAX_SIGNERS as u32);
        for j in 0..MAX_SIGNERS {
            let copy = sa * ((l[NJ] * BabyBear::new(2) + sg) * m16 + l[MEM]);
            c.assert_zero(p[PS_SIGNER + j] * (l[SIGNED + j] - copy));
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
    let condition = Condition::key([BabyBear::ONE; HASH_LEN]);
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
        signers: vec![(0, 0)],
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
    /// If there are too many items, or an item has more signers than
    /// [`MAX_SIGNERS`].
    pub fn trace(
        &self,
        items: &[CondItem],
        height: u32,
        blinding: &Hash,
        lambda: BabyBear4,
        r: BabyBear4,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        self.trace_with(items, height, blinding, lambda, r, &CondItem::members)
    }

    /// [`Self::trace`], with each item's regions from `members` (tests
    /// give a prover's lies).
    fn trace_with(
        &self,
        items: &[CondItem],
        height: u32,
        blinding: &Hash,
        lambda: BabyBear4,
        r: BabyBear4,
        members: &dyn Fn(&CondItem) -> Members,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        assert!(items.len() <= self.slots, "conditions don't fit the chunk");
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let mut chs = [BabyBear::ZERO; WIDTH];
        chs[HASH_LEN..].copy_from_slice(blinding);
        let (mut pcond, mut psig, mut ppre) = (BabyBear4::ONE, BabyBear4::ONE, BabyBear4::ONE);
        let zeros = [BabyBear::ZERO; WIDTH];
        let zero_hash = [BabyBear::ZERO; HASH_LEN];
        let put_bits = |row: &mut [BabyBear], at: usize, v: u32, count: usize| {
            for b in 0..count {
                row[RB + at + b] = BabyBear::new((v >> b) & 1);
            }
        };
        let merge_row = |row: &mut [BabyBear], run: &Hash, sib: &Hash, bit: bool| -> Hash {
            let (left, right) = if bit { (sib, run) } else { (run, sib) };
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(left);
            input[HASH_LEN..].copy_from_slice(right);
            row[BIT] = BabyBear::from(bit);
            row[SIB..SIB + HASH_LEN].copy_from_slice(sib);
            row[RUN..RUN + HASH_LEN].copy_from_slice(run);
            let out = poseidon2_air::generate(input, row);
            core::array::from_fn(|j| out[j] + input[j])
        };
        for s in 0..self.slots {
            let base = self.base(s);
            let (item, active) = match items.get(s) {
                Some(item) => (item.clone(), true),
                None => (dummy(), false),
            };
            let cond = &item.condition;
            assert!(item.signers.len() <= MAX_SIGNERS, "too many signers");
            let hl = cond.hashlock.unwrap_or(zero_hash);
            let msg = item.message();
            let kset = cond.key_set();
            let members = if active { members(&item) } else { vec![] };
            let signed = copies_of(&members);

            // The slot's constants, on every row.
            for row in base..base + SLOT_ROWS {
                let row = m.row_mut(row);
                row[ACT] = BabyBear::from(active);
                row[ACTH] = BabyBear::from(active && cond.hashlock.is_some());
                row[CLAIM..CLAIM + COND_LEN].copy_from_slice(&item.claim.elements());
                row[CPRE..CPRE + PRE_LIMBS].copy_from_slice(&item.pre());
                row[KSET..KSET + HASH_LEN].copy_from_slice(&kset);
                row[HL..HL + HASH_LEN].copy_from_slice(&hl);
                row[K] = BabyBear::new(cond.threshold as u32);
                row[M] = BabyBear::new(cond.min_state.unwrap_or(0));
                row[TL] = BabyBear::new(cond.after_height);
                row[DL] = BabyBear::new(cond.after_blocks);
                row[MACP] = BabyBear::from(cond.mode == SigMode::AnyoneCanPay);
                row[MAPO] = BabyBear::from(cond.mode == SigMode::AnyPrevout);
                row[STATE] = BabyBear::from(cond.min_state.is_some());
                row[HLK] = BabyBear::from(cond.hashlock.is_some());
                row[PROT] = BabyBear::from(cond.payouts.is_some());
                row[N] = BabyBear::new(cond.keys.len() as u32);
                row[SMSG..SMSG + HASH_LEN].copy_from_slice(&msg);
                row[SIGNED..SIGNED + MAX_SIGNERS].copy_from_slice(&signed);
            }

            // The leaf.
            let mut input = zeros;
            input[..HASH_LEN].copy_from_slice(&kset);
            let param = [
                cond.min_state.unwrap_or(0),
                cond.after_height,
                cond.after_blocks,
                cond.mode as u32,
                (cond.min_state.is_some() as u32) << 1
                    | (cond.hashlock.is_some() as u32) << 2
                    | (cond.payouts.is_some() as u32) << 3,
            ];
            for (j, &v) in param.iter().enumerate() {
                input[HASH_LEN + j] = BabyBear::new(v);
            }
            input[13] = BabyBear::new(domain::CONDITION);
            input[14] = BabyBear::new(2 * HASH_LEN as u32 + 2);
            let mut state = poseidon2_air::generate(input, m.row_mut(base + C1));
            let mut threshold = zero_hash;
            threshold[0] = BabyBear::new(cond.threshold as u32);
            threshold[1] = BabyBear::new(cond.keys.len() as u32);
            for (offset, block) in [(C2, &hl), (C3, &threshold)] {
                let mut input = state;
                for (x, &v) in input.iter_mut().zip(block) {
                    *x += v;
                }
                state = poseidon2_air::generate(input, m.row_mut(base + offset));
            }
            let mut run: Hash = core::array::from_fn(|j| state[j]);
            debug_assert_eq!(run, cond.leaf());

            // The merges up the output's key tree.
            for (d, sib) in item.path.iter().enumerate() {
                run = merge_row(m.row_mut(base + M1 + d), &run, sib, item.leaf >> d & 1 == 1);
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
            let mut bound: Vec<BabyBear> = item.claim.elements().to_vec();
            bound.extend(item.pre());
            bound.extend(signed);
            bound.resize(BLOCKS * HASH_LEN, BabyBear::ZERO);
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

            // Range checks on the condition's rows.
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
            {
                let row = m.row_mut(base + C3);
                put_bits(row, 0, min, STATE_BITS);
                if cond.payouts.is_some() {
                    put_bits(row, STATE_BITS, sub(cond.after_blocks, 1), LOCK_BITS);
                }
            }

            // Carried values: the products change after X (claim), G3
            // (preimage) and each region's first merge row (a signature or
            // a payout); the sponge after the last B row.
            let new_pcond = if active {
                pcond * (r - cond_element(lambda, &item.claim))
            } else {
                pcond
            };
            let pre_ppre = if active && cond.hashlock.is_some() {
                ppre * (r - preimage_element(lambda, &item.pre()))
            } else {
                ppre
            };
            let new_chs = if active { state } else { chs };
            for offset in 0..SLOT_ROWS {
                let row = m.row_mut(base + offset);
                let h = if offset >= S1 { &new_chs } else { &chs };
                row[CHS..CHS + WIDTH].copy_from_slice(h);
                let q = if offset > X { new_pcond } else { pcond };
                row[PCOND..PCOND + 4].copy_from_slice(&q.0);
                let q = if offset > G3 { pre_ppre } else { ppre };
                row[PPRE..PPRE + 4].copy_from_slice(&q.0);
                row[PSIG..PSIG + 4].copy_from_slice(&psig.0);
            }

            // The members' regions.
            let (mut new_psig, mut new_ppre) = (psig, pre_ppre);
            let (mut cnt, mut next_member) = (0u32, 0u32);
            let prot = cond.payouts.is_some();
            for j in 0..MAX_SIGNERS {
                let leaf = base + S1 + j * REGION_ROWS;
                let first = leaf + 1;
                let entry = members.get(j).copied();
                let on = entry.is_some();
                let (mem, idx) = entry.unwrap_or((0, None));
                let signs = idx.is_some();
                let idx = idx.unwrap_or(0);
                let (key, pay) = if on {
                    (item.key(mem), cond.payout(mem as usize))
                } else {
                    (zero_hash, zero_hash)
                };
                let (sig_before, pre_before) = (new_psig, new_ppre);
                if signs {
                    new_psig *= r - sig_element(lambda, &key, &msg, idx);
                } else if on {
                    new_ppre *= r - payout_element(lambda, &item.claim.msg, &pay);
                }
                // The leaf row: merge(key, payout).
                let mut input = zeros;
                input[..HASH_LEN].copy_from_slice(&key);
                input[HASH_LEN..].copy_from_slice(&pay);
                let out = poseidon2_air::generate(input, m.row_mut(leaf));
                let mut run: Hash = core::array::from_fn(|j| out[j] + key[j]);
                let path = cond.key_set_path(mem as usize);
                for r in leaf..first + KEY_SET_LEVELS {
                    let row = m.row_mut(r);
                    row[SA] = BabyBear::from(on);
                    row[SG] = BabyBear::from(signs);
                    row[KR..KR + HASH_LEN].copy_from_slice(&key);
                    row[PAY..PAY + HASH_LEN].copy_from_slice(&pay);
                    row[NJ] = BabyBear::new(idx);
                    row[MEM] = BabyBear::new(mem as u32);
                    row[CNT] = BabyBear::new(cnt);
                    row[NEXTM] = BabyBear::new(next_member);
                    let after = r > first;
                    let (ps, pp) = if after {
                        (new_psig, new_ppre)
                    } else {
                        (sig_before, pre_before)
                    };
                    row[PSIG..PSIG + 4].copy_from_slice(&ps.0);
                    row[PPRE..PPRE + 4].copy_from_slice(&pp.0);
                }
                for (d, sib) in path.iter().enumerate() {
                    let row = m.row_mut(first + d);
                    row[ACC] = BabyBear::new(mem as u32 & ((1 << d) - 1));
                    run = merge_row(row, &run, sib, mem >> d & 1 == 1);
                }
                let last = m.row_mut(first + KEY_SET_LEVELS - 1);
                let above = if signs && cond.min_state.is_some() {
                    sub(sub(idx, min), 1)
                } else {
                    0
                };
                put_bits(last, 0, above, STATE_BITS);
                let gap = if on { sub(mem as u32, next_member) } else { 0 };
                put_bits(last, STATE_BITS, gap, GAP_BITS);
                if j + 1 == MAX_SIGNERS && active && prot {
                    let signing = cnt + signs as u32;
                    let counts = STATE_BITS + GAP_BITS;
                    let k = cond.threshold as u32;
                    put_bits(last, counts, sub(signing, k), COUNT_BITS);
                    put_bits(last, counts + COUNT_BITS, sub(k, 1), COUNT_BITS);
                }
                if signs {
                    cnt += 1;
                }
                if on {
                    next_member = mem as u32 + 1;
                }
            }
            pcond = new_pcond;
            psig = new_psig;
            ppre = new_ppre;
            chs = new_chs;
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
            signers: indices
                .into_iter()
                .enumerate()
                .map(|(m, i)| (m as u8, i))
                .collect(),
            preimage: pre_limbs,
        }
    }

    fn channel() -> Condition {
        let a = SignerTree::new((0..8).map(h).collect());
        let b = SignerTree::new((8..16).map(h).collect());
        Condition {
            keys: vec![a.root(), b.root()],
            threshold: 2,
            min_state: Some(4),
            mode: SigMode::AnyPrevout,
            after_height: 0,
            after_blocks: 0,
            hashlock: None,
            payouts: None,
        }
    }

    fn htlc(pre: &[u8; 15]) -> Condition {
        Condition {
            keys: vec![h(50)],
            threshold: 1,
            min_state: None,
            mode: SigMode::AnyoneCanPay,
            after_height: 900,
            after_blocks: 10,
            hashlock: Some(hashlock(pre)),
            payouts: None,
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

    /// A 2-of-3 condition, signed by members 0 and 2; and the ways a
    /// threshold spend can be wrong.
    #[test]
    fn thresholds_need_distinct_members_in_order() {
        let cond = Condition {
            threshold: 2,
            min_state: Some(1),
            ..Condition::all_of(vec![h(70), h(71), h(72)])
        };
        let with = |signers: Vec<(u8, u32)>| CondItem {
            signers,
            ..spend(cond.clone(), 1, vec![], None)
        };
        let item = with(vec![(0, 3), (2, 5)]);
        let (result, public) = run(std::slice::from_ref(&item));
        assert_eq!(result, Ok(()));
        assert_eq!(item.signatures()[1].0, h(72));
        let p_sig = product(
            R,
            item.signatures()
                .iter()
                .map(|(k, m, n)| sig_element(LAMBDA, k, m, *n)),
        );
        assert_eq!(&public[pv::P_SIG..pv::P_SIG + 4], &p_sig.0);
        // (A member past the keys has a zero key, which no signature
        // chunk can match: no key tree's root is zero.)
        for bad in [
            vec![(0, 3)],
            vec![(0, 3), (1, 4), (2, 5)],
            vec![(2, 5), (0, 3)],
            vec![(1, 3), (1, 5)],
            vec![(0, 1), (2, 5)],
        ] {
            assert!(run(&[with(bad.clone())]).0.is_err(), "{bad:?}");
        }
    }

    /// A protected 2-of-3 condition (payouts `h(80..83)`, a block count
    /// of 10, state 1), created at height 1.
    fn protected() -> Condition {
        Condition {
            threshold: 2,
            min_state: Some(1),
            mode: SigMode::AnyPrevout,
            after_blocks: 10,
            payouts: Some(vec![h(80), h(81), h(82)]),
            ..Condition::all_of(vec![h(70), h(71), h(72)])
        }
    }

    fn protected_spend(cond: Condition, signers: Vec<(u8, u32)>) -> CondItem {
        CondItem {
            signers,
            ..spend(cond, 1, vec![], None)
        }
    }

    /// Members 0 and 2 sign; member 1 is paid its payout instead.
    #[test]
    fn protection_pays_members_who_dont_sign() {
        let item = protected_spend(protected(), vec![(0, 3), (2, 5)]);
        assert_eq!(item.payouts(), vec![h(81)]);
        let (result, public) = run(std::slice::from_ref(&item));
        assert_eq!(result, Ok(()));
        let p_sig = product(
            R,
            item.signatures()
                .iter()
                .map(|(k, m, n)| sig_element(LAMBDA, k, m, *n)),
        );
        assert_eq!(&public[pv::P_SIG..pv::P_SIG + 4], &p_sig.0);
        let p_pre = product(R, [payout_element(LAMBDA, &item.claim.msg, &h(81))]);
        assert_eq!(&public[pv::P_PRE..pv::P_PRE + 4], &p_pre.0);
        assert_eq!(&public[pv::HASH..pv::LEN], &chunk_hash(&BLINDING, &[item]));

        // All three sign: no payouts.
        let item = protected_spend(protected(), vec![(0, 3), (1, 4), (2, 5)]);
        let (result, public) = run(&[item]);
        assert_eq!(result, Ok(()));
        assert_eq!(&public[pv::P_PRE..pv::P_PRE + 4], &BabyBear4::ONE.0);
    }

    #[test]
    fn protection_needs_the_threshold_every_member_and_a_delay() {
        let fails = |item: CondItem| run(&[item]).0.is_err();
        // Below the threshold, or a signer's index not above the state.
        assert!(fails(protected_spend(protected(), vec![(1, 3)])));
        assert!(fails(protected_spend(protected(), vec![(0, 1), (2, 5)])));
        // Too early: created at 1, ten blocks needed.
        let mut early = protected_spend(protected(), vec![(0, 3), (2, 5)]);
        early.claim.created = HEIGHT - 9;
        assert!(fails(early));
        // Protection without a block count.
        let mut cond = protected();
        cond.after_blocks = 0;
        assert!(fails(protected_spend(cond, vec![(0, 3), (2, 5)])));
        // A threshold of zero.
        let mut cond = protected();
        cond.threshold = 0;
        assert!(fails(protected_spend(cond, vec![(0, 3), (2, 5)])));

        // A prover leaving a member out (so it isn't paid), or skipping
        // past it: the regions must be every member, in order.
        let item = protected_spend(protected(), vec![(0, 3), (2, 5)]);
        let lies: [Vec<(u8, Option<u32>)>; 4] = [
            vec![(0, Some(3)), (2, Some(5))],
            vec![(0, Some(3)), (1, None)],
            vec![(1, None), (2, Some(5))],
            vec![(0, Some(3)), (1, None), (1, None), (2, Some(5))],
        ];
        let air = CondChunkAir { slots: 1 };
        let honest = air.trace(std::slice::from_ref(&item), HEIGHT, &BLINDING, LAMBDA, R);
        assert_eq!(check_constraints(&air, &honest.0, &honest.1), Ok(()));
        for (n, lie) in lies.into_iter().enumerate() {
            let members = move |_: &CondItem| lie.clone();
            let (t, public) = air.trace_with(
                std::slice::from_ref(&item),
                HEIGHT,
                &BLINDING,
                LAMBDA,
                R,
                &members,
            );
            assert!(check_constraints(&air, &t, &public).is_err(), "lie {n}");
        }
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
