//! The state-update segment circuit: proves a run of state-tree inserts then
//! removals ([`verify_insert`], [`verify_remove`]), in strictly increasing
//! key order, moving the state from one `(root, size)` to another.
//!
//! Blocks apply their sorted output list (inserts) and then their sorted
//! input list (removals); segments cover consecutive slices, chained by
//! their public states and last keys. Each segment also outputs multiset
//! fingerprints of what it inserted (key and nonce) and removed (key and
//! position), which the aggregation matches against the item chunks'.
//!
//! Every row holds one Poseidon2 permutation plus auxiliary columns, with
//! preprocessed selectors giving each row its role. A leaf hash is four
//! sponge rows (key, nonce, next key, next index). Merkle paths are 40 merge
//! rows; two paths sharing siblings run interleaved (`A`, `B`, `A`, ...), so
//! each level's sibling is copied to the adjacent row. Per insert:
//!
//! - three comparisons: previous key < key, low key < key, key < low next
//!   key (or the end of the list);
//! - leaf hashes of the old low leaf, the relinked low leaf and the new leaf;
//! - path pair 1 (old low leaf / relinked) along the low leaf's index: `A`
//!   must reach the current root, `B` gives the intermediate root;
//! - path pair 2 (empty slot / new leaf) along `size`: `A` must reach the
//!   intermediate root, `B` is the new root.
//!
//! Per removal (spend) at position `pos`:
//!
//! - an order comparison: previous key < key;
//! - leaf hashes of the predecessor linking to `(key, pos)`, the predecessor
//!   relinked past it, and the removed leaf;
//! - path pair 1 (old / relinked predecessor) along the predecessor's
//!   index: `A` reaches the current root, `B` the intermediate root;
//! - path pair 2 (removed leaf / empty slot) along `pos`: `A` must reach
//!   the intermediate root, `B` is the new root.
//!
//! Comparisons are lexicographic on the keys' 8 canonical limbs: a one-hot
//! column marks the first differing limb, earlier limbs are equal, and that
//! limb pair is split into 15 + 16 bits (the right side checked canonical)
//! and compared with a 16-bit difference.
//!
//! Inactive operations leave the state alone; their rows are filled with
//! dummy values satisfying the ungated constraints.
//!
//! Public values (see [`public`]): state before (`root 8 | size lo, hi`),
//! state after, `lambda (4) | r (4)`, `P_ins (4) | P_sp (4)`, and the last
//! inserted and spent keys before and after (`8` each).
//!
//! [`verify_insert`]: crate::state::verify_insert
//! [`verify_remove`]: crate::state::verify_remove

use crate::air::{Air, Constraints, Var};
use crate::babybear::{BabyBear, P};
use crate::babybear4::BabyBear4;
use crate::hash::{HASH_LEN, Hash, domain};
use crate::matrix::RowMajorMatrix;
use crate::output::{self, Commitment, NONCE_LIMBS, OutputRecord};
use crate::poseidon2::WIDTH;
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::state::{self, DEPTH, InsertProof, LEAF_LEN, Leaf, RemoveProof, State, key_of};

/// `p - 1 = 30720 * 2^16`: the largest canonical limb's high part.
const HI_MAX: u32 = 30720;
const LO_LIMIT: u32 = (1 << 30) - 1;
/// The comparison bound for "end of list": every canonical limb is `<= p - 1`.
const MAX_LIMB: u32 = P - 1;

/// Rows of one leaf hash.
const LEAF_ROWS: usize = 4;
const INS_ROWS: usize = 3 + 2 * LEAF_ROWS + 2 * DEPTH + LEAF_ROWS + 2 * DEPTH + 1;
const SP_ROWS: usize = 1 + 2 * LEAF_ROWS + 2 * DEPTH + LEAF_ROWS + 2 * DEPTH + 1;

// Auxiliary columns after the permutation's.
const AUX: usize = poseidon2_air::NUM_COLS;
// Globals, carried down every row.
const ROOT: usize = AUX;
const SIZE_LO: usize = ROOT + HASH_LEN;
const SIZE_HI: usize = SIZE_LO + 1;
const PREV_INS: usize = SIZE_HI + 1;
const PREV_SP: usize = PREV_INS + HASH_LEN;
const PINS: usize = PREV_SP + HASH_LEN;
const PSP: usize = PINS + 4;
// Per operation, constant across its rows.
const ACT: usize = PSP + 4;
/// The inserted or removed key, and its nonce.
const KEY: usize = ACT + 1;
const NONCE: usize = KEY + HASH_LEN;
/// The low leaf (insert) or predecessor (removal): key, nonce, next key
/// and next index.
const LK: usize = NONCE + NONCE_LIMBS;
const LN: usize = LK + HASH_LEN;
const LNK: usize = LN + NONCE_LIMBS;
const LNI_LO: usize = LNK + HASH_LEN;
const LNI_HI: usize = LNI_LO + 1;
/// The removed leaf's next key and index, and its position.
const RNK: usize = LNI_HI + 1;
const RNI_LO: usize = RNK + HASH_LEN;
const RNI_HI: usize = RNI_LO + 1;
const POS_LO: usize = RNI_HI + 1;
const POS_HI: usize = POS_LO + 1;
const MID: usize = POS_HI + 1;
// Leaf hash rows.
const LEAF_K: usize = MID + HASH_LEN;
const LEAF_N: usize = LEAF_K + HASH_LEN;
const LEAF_NK: usize = LEAF_N + NONCE_LIMBS;
const LEAF_LO: usize = LEAF_NK + HASH_LEN;
const LEAF_HI: usize = LEAF_LO + 1;
// Merge rows.
const CUR_A: usize = LEAF_HI + 1;
const CUR_B: usize = CUR_A + HASH_LEN;
const SIB: usize = CUR_B + HASH_LEN;
const BIT: usize = SIB + HASH_LEN;
const IDX_LO: usize = BIT + 1;
const IDX_HI: usize = IDX_LO + 1;
// Comparison rows.
const CA: usize = IDX_HI + 1;
const CB: usize = CA + HASH_LEN;
const E: usize = CB + HASH_LEN;
/// `a = AH 2^16 + AL`: 16 low bits then 15 high bits.
const ABITS: usize = E + HASH_LEN;
const BBITS: usize = ABITS + 31;
const GBITS: usize = BBITS + 31;
const GINV: usize = GBITS + 15;
const F: usize = GINV + 1;
const DBITS: usize = F + 1;
const ISEND: usize = DBITS + 16;
const NINV: usize = ISEND + 1;
// End rows.
const CINV: usize = NINV + 1;
pub const NUM_COLS: usize = CINV + 1;

// Preprocessed columns.
const CMP_ORD: usize = 0;
const CMP_LOW: usize = 1;
const CMP_NEXT: usize = 2;
const CMP_ORD_SP: usize = 3;
const H1: usize = 4;
const H2: usize = 5;
const H3: usize = 6;
const H4: usize = 7;
const K_INS_OLD: usize = 8;
const K_INS_RELINK: usize = 9;
const K_INS_NEW: usize = 10;
const K_PRED_OLD: usize = 11;
const K_PRED_NEW: usize = 12;
const K_REMOVED: usize = 13;
const SET_A: usize = 14;
const SET_B: usize = 15;
const RESET_A: usize = 16;
const RESET_B: usize = 17;
const MA: usize = 18;
const MB: usize = 19;
const WLO: usize = 20;
const WHI: usize = 21;
const CHECK1: usize = 22;
const END_INS: usize = 23;
const END_SP: usize = 24;
const SAME_OP: usize = 25;
const FIRST: usize = 26;
const FINAL: usize = 27;
pub const PRE_WIDTH: usize = 28;

/// Offsets of the public values.
pub mod public {
    pub const OLD_ROOT: usize = 0;
    pub const OLD_SIZE: usize = 8;
    pub const NEW_ROOT: usize = 10;
    pub const NEW_SIZE: usize = 18;
    pub const LAMBDA: usize = 20;
    pub const R: usize = 24;
    pub const P_INS: usize = 28;
    pub const P_SP: usize = 32;
    pub const PREV_INS_IN: usize = 36;
    pub const PREV_INS_OUT: usize = 44;
    pub const PREV_SP_IN: usize = 52;
    pub const PREV_SP_OUT: usize = 60;
    pub const LEN: usize = 68;
}

/// A segment's capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateSegmentAir {
    pub inserts: usize,
    pub spends: usize,
}

/// Row offsets of one insert.
mod ins {
    use super::*;
    pub const ORD: usize = 0;
    pub const LOW: usize = 1;
    pub const NEXT: usize = 2;
    pub const H_OLD: usize = 3;
    pub const H_RELINK: usize = H_OLD + LEAF_ROWS;
    pub const P1: usize = H_RELINK + LEAF_ROWS;
    pub const H_NEW: usize = P1 + 2 * DEPTH;
    pub const P2: usize = H_NEW + LEAF_ROWS;
    pub const END: usize = P2 + 2 * DEPTH;
}

/// Row offsets of one removal.
mod sp {
    use super::*;
    pub const ORD: usize = 0;
    pub const H_PRED_OLD: usize = 1;
    pub const H_PRED_NEW: usize = H_PRED_OLD + LEAF_ROWS;
    pub const P1: usize = H_PRED_NEW + LEAF_ROWS;
    pub const H_REMOVED: usize = P1 + 2 * DEPTH;
    pub const P2: usize = H_REMOVED + LEAF_ROWS;
    pub const END: usize = P2 + 2 * DEPTH;
}

impl StateSegmentAir {
    fn insert_base(&self, i: usize) -> usize {
        i * INS_ROWS
    }

    fn spend_base(&self, s: usize) -> usize {
        self.inserts * INS_ROWS + s * SP_ROWS
    }

    fn final_row(&self) -> usize {
        self.inserts * INS_ROWS + self.spends * SP_ROWS
    }

    pub fn height(&self) -> usize {
        (self.final_row() + 1).next_power_of_two().max(64)
    }
}

fn ext_mul<V: Var>(a: [V; 4], b: [V; 4]) -> [V; 4] {
    let w = |x: V| x * BabyBear::new(11);
    [
        a[0] * b[0] + w(a[1] * b[3] + a[2] * b[2] + a[3] * b[1]),
        a[0] * b[1] + a[1] * b[0] + w(a[2] * b[3] + a[3] * b[2]),
        a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w(a[3] * b[3]),
        a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
    ]
}

impl Air for StateSegmentAir {
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
        let leaf = |m: &mut RowMajorMatrix, at: usize, kind: usize, set: &[usize]| {
            m.row_mut(at)[H1] = one;
            m.row_mut(at)[kind] = one;
            m.row_mut(at + 1)[H2] = one;
            m.row_mut(at + 2)[H3] = one;
            m.row_mut(at + 3)[H4] = one;
            for &col in set {
                m.row_mut(at + 3)[col] = one;
            }
        };
        let path = |m: &mut RowMajorMatrix, at: usize| {
            for d in 0..DEPTH {
                let a = m.row_mut(at + 2 * d);
                a[MA] = one;
                if d < 30 {
                    a[WLO] = BabyBear::new(1 << d);
                } else {
                    a[WHI] = BabyBear::new(1 << (d - 30));
                }
                m.row_mut(at + 2 * d + 1)[MB] = one;
            }
        };
        for i in 0..self.inserts {
            let b = self.insert_base(i);
            m.row_mut(b + ins::ORD)[CMP_ORD] = one;
            m.row_mut(b + ins::LOW)[CMP_LOW] = one;
            m.row_mut(b + ins::NEXT)[CMP_NEXT] = one;
            leaf(&mut m, b + ins::H_OLD, K_INS_OLD, &[SET_A]);
            leaf(&mut m, b + ins::H_RELINK, K_INS_RELINK, &[SET_B]);
            path(&mut m, b + ins::P1);
            leaf(&mut m, b + ins::H_NEW, K_INS_NEW, &[SET_B, RESET_A]);
            m.row_mut(b + ins::H_NEW)[CHECK1] = one;
            path(&mut m, b + ins::P2);
            m.row_mut(b + ins::END)[END_INS] = one;
            for r in b..b + ins::END {
                m.row_mut(r)[SAME_OP] = one;
            }
        }
        for s in 0..self.spends {
            let b = self.spend_base(s);
            m.row_mut(b + sp::ORD)[CMP_ORD_SP] = one;
            leaf(&mut m, b + sp::H_PRED_OLD, K_PRED_OLD, &[SET_A]);
            leaf(&mut m, b + sp::H_PRED_NEW, K_PRED_NEW, &[SET_B]);
            path(&mut m, b + sp::P1);
            leaf(&mut m, b + sp::H_REMOVED, K_REMOVED, &[SET_A, RESET_B]);
            m.row_mut(b + sp::H_REMOVED)[CHECK1] = one;
            path(&mut m, b + sp::P2);
            m.row_mut(b + sp::END)[END_SP] = one;
            for r in b..b + sp::END {
                m.row_mut(r)[SAME_OP] = one;
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
        let bits = |at: usize, count: usize| {
            (0..count).fold(zero, |acc, b| acc + l[at + b] * BabyBear::new(1 << b))
        };
        let ext = |col: usize, row: &[V]| -> [V; 4] { core::array::from_fn(|b| row[col + b]) };
        let pub4 = |at: usize| -> [V; 4] { core::array::from_fn(|b| public[at + b]) };
        let act = l[ACT];

        poseidon2_air::eval(c, l);

        // Per-operation constants.
        c.assert_zero(act * (act - one));
        for col in ACT..MID + HASH_LEN {
            c.assert_zero(p[SAME_OP] * (n[col] - l[col]));
        }

        // Leaf hash rows: sponge(STATE_LEAF, [], [key, nonce head, next
        // key, lo, hi, nonce tail]), one block per row.
        let h1 = p[H1];
        for j in 0..HASH_LEN {
            c.assert_zero(h1 * (l[j] - l[LEAF_K + j]));
        }
        for &x in &l[HASH_LEN..HASH_LEN + 5] {
            c.assert_zero(h1 * x);
        }
        c.assert_zero(h1 * (l[13] - k(domain::STATE_LEAF)));
        c.assert_zero(h1 * (l[14] - k(LEAF_LEN as u32)));
        c.assert_zero(h1 * l[15]);
        for col in LEAF_K..LEAF_HI + 1 {
            c.assert_zero((h1 + p[H2] + p[H3]) * (n[col] - l[col]));
        }
        for (sel, block) in [(h1, LEAF_N), (p[H2], LEAF_NK)] {
            for (j, &x) in n[..WIDTH].iter().enumerate() {
                let absorbed = if j < HASH_LEN { n[block + j] } else { zero };
                c.assert_zero(sel * (x - out(j) - absorbed));
            }
        }
        for (j, &x) in n[..WIDTH].iter().enumerate() {
            let absorbed = match j {
                0 => l[LEAF_LO],
                1 => l[LEAF_HI],
                j if j < 2 + state::NONCE_TAIL => l[LEAF_N + HASH_LEN + j - 2],
                _ => zero,
            };
            c.assert_zero(p[H3] * (x - out(j) - absorbed));
        }
        // Which leaf: (selector, key, nonce, next key, next index lo, hi).
        type Kind<V> = (usize, [V; 8], usize, [V; 8], V, V);
        let kinds: [Kind<V>; 6] = [
            (
                K_INS_OLD,
                ext8(l, LK),
                LN,
                ext8(l, LNK),
                l[LNI_LO],
                l[LNI_HI],
            ),
            (
                K_INS_RELINK,
                ext8(l, LK),
                LN,
                ext8(l, KEY),
                l[SIZE_LO],
                l[SIZE_HI],
            ),
            (
                K_INS_NEW,
                ext8(l, KEY),
                NONCE,
                ext8(l, LNK),
                l[LNI_LO],
                l[LNI_HI],
            ),
            (
                K_PRED_OLD,
                ext8(l, LK),
                LN,
                ext8(l, KEY),
                l[POS_LO],
                l[POS_HI],
            ),
            (
                K_PRED_NEW,
                ext8(l, LK),
                LN,
                ext8(l, RNK),
                l[RNI_LO],
                l[RNI_HI],
            ),
            (
                K_REMOVED,
                ext8(l, KEY),
                NONCE,
                ext8(l, RNK),
                l[RNI_LO],
                l[RNI_HI],
            ),
        ];
        for j in 0..HASH_LEN {
            let (mut dk, mut dnk) = (zero, zero);
            for (kind, key, _, nk, ..) in &kinds {
                dk = dk + p[*kind] * (l[LEAF_K + j] - key[j]);
                dnk = dnk + p[*kind] * (l[LEAF_NK + j] - nk[j]);
            }
            c.assert_zero(dk);
            c.assert_zero(dnk);
        }
        for j in 0..NONCE_LIMBS {
            let mut dn = zero;
            for (kind, _, nonce, ..) in &kinds {
                dn = dn + p[*kind] * (l[LEAF_N + j] - l[*nonce + j]);
            }
            c.assert_zero(dn);
        }
        let (mut dlo, mut dhi) = (zero, zero);
        for (kind, _, _, _, lo, hi) in &kinds {
            dlo = dlo + p[*kind] * (l[LEAF_LO] - *lo);
            dhi = dhi + p[*kind] * (l[LEAF_HI] - *hi);
        }
        c.assert_zero(dlo);
        c.assert_zero(dhi);

        // Running nodes: set from leaf hashes, reset to the empty leaf,
        // advanced by merges, else carried.
        let (ma, mb) = (p[MA], p[MB]);
        for j in 0..HASH_LEN {
            c.assert_zero(p[SET_A] * (n[CUR_A + j] - out(j)));
            c.assert_zero(p[RESET_A] * n[CUR_A + j]);
            c.assert_zero(p[SET_B] * (n[CUR_B + j] - out(j)));
            c.assert_zero(p[RESET_B] * n[CUR_B + j]);
            c.assert_zero_transition(
                (one - p[SET_A] - p[RESET_A] - ma) * (n[CUR_A + j] - l[CUR_A + j]),
            );
            c.assert_zero_transition(
                (one - p[SET_B] - p[RESET_B] - mb) * (n[CUR_B + j] - l[CUR_B + j]),
            );
        }
        for col in [IDX_LO, IDX_HI] {
            c.assert_zero((p[SET_B] + p[RESET_B]) * n[col]);
            c.assert_zero_transition((one - p[SET_B] - p[RESET_B] - ma) * (n[col] - l[col]));
        }
        let bit = l[BIT];
        for (sel, cur) in [(ma, CUR_A), (mb, CUR_B)] {
            for j in 0..HASH_LEN {
                let (x, s) = (l[cur + j], l[SIB + j]);
                c.assert_zero(sel * (l[j] - x - bit * (s - x)));
                c.assert_zero(sel * (l[HASH_LEN + j] - s - bit * (x - s)));
                c.assert_zero(sel * (n[cur + j] - out(j) - l[j]));
            }
        }
        c.assert_zero(ma * bit * (bit - one));
        for j in 0..HASH_LEN {
            c.assert_zero(ma * (n[SIB + j] - l[SIB + j]));
        }
        c.assert_zero(ma * (n[BIT] - bit));
        c.assert_zero(ma * (n[IDX_LO] - l[IDX_LO] - bit * p[WLO]));
        c.assert_zero(ma * (n[IDX_HI] - l[IDX_HI] - bit * p[WHI]));

        // End of pair 1 (on the second pair's first hash row): A is the
        // root, B the intermediate root.
        for j in 0..HASH_LEN {
            c.assert_zero(p[CHECK1] * act * (l[CUR_A + j] - l[ROOT + j]));
            c.assert_zero(p[CHECK1] * (l[MID + j] - l[CUR_B + j]));
        }

        // Comparisons.
        let max_key = [k(MAX_LIMB); HASH_LEN];
        let v = l[LNI_LO] + l[LNI_HI] * BabyBear::new(1 << 19);
        c.assert_zero(p[CMP_NEXT] * (l[ISEND] - one + v * l[NINV]));
        c.assert_zero(p[CMP_NEXT] * v * l[ISEND]);
        for j in 0..HASH_LEN {
            let next_bound = l[LNK + j] + l[ISEND] * (max_key[j] - l[LNK + j]);
            let pairs = [
                (CMP_ORD, l[PREV_INS + j], l[KEY + j]),
                (CMP_LOW, l[LK + j], l[KEY + j]),
                (CMP_NEXT, l[KEY + j], next_bound),
                (CMP_ORD_SP, l[PREV_SP + j], l[KEY + j]),
            ];
            for (sel, a, b) in pairs {
                c.assert_zero(p[sel] * (l[CA + j] - a));
                c.assert_zero(p[sel] * (l[CB + j] - b));
            }
        }
        let cmp = p[CMP_ORD] + p[CMP_LOW] + p[CMP_NEXT] + p[CMP_ORD_SP];
        let mut e_sum = zero;
        for j in 0..HASH_LEN {
            c.assert_zero(cmp * l[E + j] * (l[E + j] - one));
            e_sum = e_sum + l[E + j];
            let after = (j + 1..HASH_LEN).fold(zero, |acc, i| acc + l[E + i]);
            c.assert_zero(cmp * (l[CA + j] - l[CB + j]) * after);
        }
        c.assert_zero(cmp * (e_sum - one));
        let pick = |col: usize| (0..HASH_LEN).fold(zero, |acc, j| acc + l[E + j] * l[col + j]);
        for b in (ABITS..ABITS + 31)
            .chain(BBITS..BBITS + 31)
            .chain(GBITS..GBITS + 15)
            .chain(DBITS..DBITS + 16)
            .chain([F])
        {
            c.assert_zero(cmp * l[b] * (l[b] - one));
        }
        let (al, ah) = (bits(ABITS, 16), bits(ABITS + 16, 15));
        let (bl, bh) = (bits(BBITS, 16), bits(BBITS + 16, 15));
        c.assert_zero(cmp * (pick(CA) - ah * BabyBear::new(1 << 16) - al));
        c.assert_zero(cmp * (pick(CB) - bh * BabyBear::new(1 << 16) - bl));
        // b canonical: bh <= HI_MAX, and bh = HI_MAX => bl = 0.
        let g = bits(GBITS, 15);
        c.assert_zero(cmp * (bh + g - k(HI_MAX)));
        let z = one - g * l[GINV];
        c.assert_zero(cmp * g * z);
        c.assert_zero(cmp * bl * z);
        let f = l[F];
        c.assert_zero(cmp * (one - f) * (ah - bh));
        c.assert_zero(cmp * (bits(DBITS, 16) - f * (bh - ah - one) - (one - f) * (bl - al - one)));

        // End rows: pair 2's A reaches the intermediate root along the
        // operation's slot (size for inserts, the position for removals);
        // its B is the new root.
        let (ei, es) = (p[END_INS], p[END_SP]);
        for j in 0..HASH_LEN {
            c.assert_zero((ei + es) * act * (l[CUR_A + j] - l[MID + j]));
            let updated = act * l[CUR_B + j] + (one - act) * l[ROOT + j];
            c.assert_zero((ei + es) * (n[ROOT + j] - updated));
            c.assert_zero_transition((one - ei - es) * (n[ROOT + j] - l[ROOT + j]));
        }
        c.assert_zero(ei * act * (l[IDX_LO] - l[SIZE_LO]));
        c.assert_zero(ei * act * (l[IDX_HI] - l[SIZE_HI]));
        c.assert_zero(es * act * (l[IDX_LO] - l[POS_LO]));
        c.assert_zero(es * act * (l[IDX_HI] - l[POS_HI]));
        // size + 1, carrying from the low 30 bits.
        let w = l[SIZE_LO] - k(LO_LIMIT);
        let carry = one - w * l[CINV];
        c.assert_zero(ei * w * carry);
        c.assert_zero(
            ei * (n[SIZE_LO] - l[SIZE_LO] - act * (one - carry * BabyBear::new(1 << 30))),
        );
        c.assert_zero(ei * (n[SIZE_HI] - l[SIZE_HI] - act * carry));
        for col in [SIZE_LO, SIZE_HI] {
            c.assert_zero_transition((one - ei) * (n[col] - l[col]));
        }
        for (end, prev) in [(ei, PREV_INS), (es, PREV_SP)] {
            for j in 0..HASH_LEN {
                let updated = act * l[KEY + j] + (one - act) * l[prev + j];
                c.assert_zero(end * (n[prev + j] - updated));
                c.assert_zero_transition((one - end) * (n[prev + j] - l[prev + j]));
            }
        }
        // Fingerprints: inserted (key, nonce), removed (key, position).
        let (lambda, r) = (pub4(public::LAMBDA), pub4(public::R));
        let mut powers = [[zero; 4]; HASH_LEN + NONCE_LIMBS];
        powers[0][0] = one;
        for j in 1..powers.len() {
            powers[j] = ext_mul(powers[j - 1], lambda);
        }
        let element = |cols: &mut dyn Iterator<Item = V>| -> [V; 4] {
            cols.zip(&powers).fold([zero; 4], |acc, (x, pw)| {
                core::array::from_fn(|b| acc[b] + x * pw[b])
            })
        };
        let factor = |e: [V; 4]| -> [V; 4] {
            core::array::from_fn(|b| {
                let unit = if b == 0 { one } else { zero };
                unit - act * unit + act * (r[b] - e[b])
            })
        };
        let e_ins = element(&mut (KEY..NONCE + NONCE_LIMBS).map(|col| l[col]));
        let e_sp = element(
            &mut (KEY..KEY + HASH_LEN)
                .chain([POS_LO, POS_HI])
                .map(|col| l[col]),
        );
        for (end, col, e) in [(ei, PINS, e_ins), (es, PSP, e_sp)] {
            let next = ext_mul(ext(col, l), factor(e));
            for b in 0..4 {
                c.assert_zero(end * (n[col + b] - next[b]));
                c.assert_zero_transition((one - end) * (n[col + b] - l[col + b]));
            }
        }

        // First and final rows against the public values.
        let boundary = [
            (ROOT, public::OLD_ROOT, public::NEW_ROOT, HASH_LEN),
            (SIZE_LO, public::OLD_SIZE, public::NEW_SIZE, 2),
            (
                PREV_INS,
                public::PREV_INS_IN,
                public::PREV_INS_OUT,
                HASH_LEN,
            ),
            (PREV_SP, public::PREV_SP_IN, public::PREV_SP_OUT, HASH_LEN),
        ];
        for (col, before, after, len) in boundary {
            for j in 0..len {
                c.assert_zero(p[FIRST] * (l[col + j] - public[before + j]));
                c.assert_zero(p[FINAL] * (l[col + j] - public[after + j]));
            }
        }
        for (col, at) in [(PINS, public::P_INS), (PSP, public::P_SP)] {
            for b in 0..4 {
                let unit = if b == 0 { one } else { zero };
                c.assert_zero(p[FIRST] * (l[col + b] - unit));
                c.assert_zero(p[FINAL] * (l[col + b] - public[at + b]));
            }
        }
    }
}

fn ext8<V: Copy>(row: &[V], col: usize) -> [V; 8] {
    core::array::from_fn(|j| row[col + j])
}

/// One insert: the output being inserted and its tree proof.
#[derive(Clone, Debug)]
pub struct InsertOp {
    pub record: OutputRecord,
    pub proof: InsertProof,
}

/// One removal: the commitment being spent and its tree proof (which
/// holds its position).
#[derive(Clone, Debug)]
pub struct SpendOp {
    pub commitment: Commitment,
    pub proof: RemoveProof,
}

/// Where a segment starts: the state and the last keys inserted and spent
/// (zero before the first).
#[derive(Clone, Copy, Debug)]
pub struct SegmentStart {
    pub state: State,
    pub prev_insert: Hash,
    pub prev_spend: Hash,
}

impl StateSegmentAir {
    /// The trace and public values for applying `inserts` then `spends`,
    /// whose proofs must come from applying them in this order (e.g. with
    /// [`StateTree`](crate::state::StateTree)). Invalid operations give a
    /// trace that fails the constraints.
    ///
    /// # Panics
    /// If there are more operations than the capacity, or a key is not
    /// canonical.
    pub fn trace(
        &self,
        start: &SegmentStart,
        inserts: &[InsertOp],
        spends: &[SpendOp],
        lambda: BabyBear4,
        r: BabyBear4,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        assert!(inserts.len() <= self.inserts && spends.len() <= self.spends);
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let mut g = Globals {
            root: start.state.root,
            size: start.state.size,
            prev_ins: start.prev_insert,
            prev_sp: start.prev_spend,
            pins: BabyBear4::ONE,
            psp: BabyBear4::ONE,
        };
        let powers: Vec<BabyBear4> = (0..HASH_LEN + NONCE_LIMBS)
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

        for slot in 0..self.inserts {
            let b = self.insert_base(slot);
            let o = match inserts.get(slot) {
                Some(op) => OpFields {
                    active: true,
                    key: key_of(&op.record.commitment).expect("canonical key"),
                    nonce: op.record.nonce,
                    low: op.proof.low.leaf,
                    low_index: op.proof.low.index,
                    low_siblings: *op.proof.low.siblings,
                    new_siblings: *op.proof.new_siblings,
                    removed_next_key: [BabyBear::ZERO; HASH_LEN],
                    removed_next_index: 0,
                    position: 0,
                },
                None => OpFields::dummy_insert(),
            };
            let size = g.size;
            let relinked = Leaf {
                next_key: o.key,
                next_index: size,
                ..o.low
            };
            let new_leaf = Leaf {
                key: o.key,
                nonce: o.nonce,
                next_key: o.low.next_key,
                next_index: o.low.next_index,
            };
            o.write_constants(&mut m, b, INS_ROWS);
            compare(&mut m, b + ins::ORD, &g.prev_ins, &o.key);
            compare(&mut m, b + ins::LOW, &o.low.key, &o.key);
            let end = o.low.next_index == 0;
            let bound = if end {
                [BabyBear::new(MAX_LIMB); HASH_LEN]
            } else {
                o.low.next_key
            };
            compare(&mut m, b + ins::NEXT, &o.key, &bound);
            {
                let row = m.row_mut(b + ins::NEXT);
                row[ISEND] = BabyBear::from(end);
                let (lo, hi) = state::split(o.low.next_index);
                row[NINV] = (lo + hi * BabyBear::new(1 << 19))
                    .inverse()
                    .unwrap_or(BabyBear::ZERO);
            }
            let h_old = leaf_rows(&mut m, b + ins::H_OLD, &o.low);
            let h_relink = leaf_rows(&mut m, b + ins::H_RELINK, &relinked);
            let (_, b1) = path_rows(
                &mut m,
                b + ins::P1,
                h_old,
                h_relink,
                o.low_index,
                &o.low_siblings,
            );
            let h_new = leaf_rows(&mut m, b + ins::H_NEW, &new_leaf);
            let (_, b2) = path_rows(
                &mut m,
                b + ins::P2,
                [BabyBear::ZERO; HASH_LEN],
                h_new,
                size,
                &o.new_siblings,
            );
            for r in b..b + INS_ROWS {
                m.row_mut(r)[MID..MID + HASH_LEN].copy_from_slice(&b1);
            }
            g.write(&mut m, b, INS_ROWS);
            poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(b + ins::END));
            let row = m.row_mut(b + ins::END);
            let (lo, _) = state::split(size);
            row[CINV] = (lo - BabyBear::new(LO_LIMIT))
                .inverse()
                .unwrap_or(BabyBear::ZERO);
            if o.active {
                g.root = b2;
                g.size += 1;
                g.prev_ins = o.key;
                let mut xs = o.key.to_vec();
                xs.extend(output::nonce_limbs(&o.nonce));
                g.pins *= r - fp(&xs);
            }
        }

        for slot in 0..self.spends {
            let b = self.spend_base(slot);
            let o = match spends.get(slot) {
                Some(op) => OpFields {
                    active: true,
                    key: key_of(&op.commitment).expect("canonical key"),
                    nonce: op.proof.leaf.leaf.nonce,
                    low: op.proof.pred.leaf,
                    low_index: op.proof.pred.index,
                    low_siblings: *op.proof.pred.siblings,
                    new_siblings: *op.proof.leaf.siblings,
                    removed_next_key: op.proof.leaf.leaf.next_key,
                    removed_next_index: op.proof.leaf.leaf.next_index,
                    position: op.proof.position(),
                },
                None => OpFields::dummy_spend(),
            };
            o.write_constants(&mut m, b, SP_ROWS);
            compare(&mut m, b + sp::ORD, &g.prev_sp, &o.key);
            let pred_old = Leaf {
                next_key: o.key,
                next_index: o.position,
                ..o.low
            };
            let pred_new = Leaf {
                next_key: o.removed_next_key,
                next_index: o.removed_next_index,
                ..o.low
            };
            let removed = Leaf {
                key: o.key,
                nonce: o.nonce,
                next_key: o.removed_next_key,
                next_index: o.removed_next_index,
            };
            let h_old = leaf_rows(&mut m, b + sp::H_PRED_OLD, &pred_old);
            let h_new = leaf_rows(&mut m, b + sp::H_PRED_NEW, &pred_new);
            let (_, b1) = path_rows(
                &mut m,
                b + sp::P1,
                h_old,
                h_new,
                o.low_index,
                &o.low_siblings,
            );
            let h_removed = leaf_rows(&mut m, b + sp::H_REMOVED, &removed);
            let (_, b2) = path_rows(
                &mut m,
                b + sp::P2,
                h_removed,
                [BabyBear::ZERO; HASH_LEN],
                o.position,
                &o.new_siblings,
            );
            for r in b..b + SP_ROWS {
                m.row_mut(r)[MID..MID + HASH_LEN].copy_from_slice(&b1);
            }
            poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(b + sp::END));
            g.write(&mut m, b, SP_ROWS);
            if o.active {
                g.root = b2;
                g.prev_sp = o.key;
                let (lo, hi) = state::split(o.position);
                let mut xs = o.key.to_vec();
                xs.extend([lo, hi]);
                g.psp *= r - fp(&xs);
            }
        }

        let fin = self.final_row();
        let height = m.height();
        g.write(&mut m, fin, height - fin);
        for row in fin..height {
            poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(row));
        }
        self.propagate_nodes(&mut m);

        let old = &start.state;
        let mut public = vec![BabyBear::ZERO; public::LEN];
        let (olo, ohi) = state::split(old.size);
        let (nlo, nhi) = state::split(g.size);
        public[public::OLD_ROOT..public::OLD_ROOT + 8].copy_from_slice(&old.root);
        public[public::OLD_SIZE] = olo;
        public[public::OLD_SIZE + 1] = ohi;
        public[public::NEW_ROOT..public::NEW_ROOT + 8].copy_from_slice(&g.root);
        public[public::NEW_SIZE] = nlo;
        public[public::NEW_SIZE + 1] = nhi;
        public[public::LAMBDA..public::LAMBDA + 4].copy_from_slice(&lambda.0);
        public[public::R..public::R + 4].copy_from_slice(&r.0);
        public[public::P_INS..public::P_INS + 4].copy_from_slice(&g.pins.0);
        public[public::P_SP..public::P_SP + 4].copy_from_slice(&g.psp.0);
        public[public::PREV_INS_IN..public::PREV_INS_IN + 8].copy_from_slice(&start.prev_insert);
        public[public::PREV_INS_OUT..public::PREV_INS_OUT + 8].copy_from_slice(&g.prev_ins);
        public[public::PREV_SP_IN..public::PREV_SP_IN + 8].copy_from_slice(&start.prev_spend);
        public[public::PREV_SP_OUT..public::PREV_SP_OUT + 8].copy_from_slice(&g.prev_sp);
        (m, public)
    }
}

impl StateSegmentAir {
    /// Fills the running-node columns (`CUR_A`, `CUR_B`, `IDX`) down every
    /// row by the same rules the constraints enforce: set from leaf hashes,
    /// reset, advanced by merges, otherwise carried. Path rows were already
    /// filled from the same values; this checks them against the rules.
    fn propagate_nodes(&self, m: &mut RowMajorMatrix) {
        let pre = self.preprocessed_trace().expect("preprocessed columns");
        let one = BabyBear::ONE;
        for r in 0..m.height() - 1 {
            let (p, row) = (pre.row(r), m.row(r));
            let out = &row[OUTPUT_OFFSET..OUTPUT_OFFSET + HASH_LEN];
            let node: Hash = core::array::from_fn(|j| out[j] + row[j]);
            let mut a: Hash = core::array::from_fn(|j| row[CUR_A + j]);
            let mut b: Hash = core::array::from_fn(|j| row[CUR_B + j]);
            let (mut lo, mut hi) = (row[IDX_LO], row[IDX_HI]);
            if p[SET_A] == one {
                a.copy_from_slice(out);
            } else if p[RESET_A] == one {
                a = [BabyBear::ZERO; HASH_LEN];
            } else if p[MA] == one {
                a = node;
                lo += row[BIT] * p[WLO];
                hi += row[BIT] * p[WHI];
            }
            if p[SET_B] == one {
                b.copy_from_slice(out);
            } else if p[RESET_B] == one {
                b = [BabyBear::ZERO; HASH_LEN];
            } else if p[MB] == one {
                b = node;
            }
            if p[SET_B] == one || p[RESET_B] == one {
                lo = BabyBear::ZERO;
                hi = BabyBear::ZERO;
            }
            let path_row = pre.row(r + 1)[MA] == one || pre.row(r + 1)[MB] == one;
            let next = m.row_mut(r + 1);
            if path_row {
                debug_assert_eq!(&next[CUR_A..CUR_A + HASH_LEN], &a, "row {}", r + 1);
                debug_assert_eq!(&next[CUR_B..CUR_B + HASH_LEN], &b, "row {}", r + 1);
            }
            next[CUR_A..CUR_A + HASH_LEN].copy_from_slice(&a);
            next[CUR_B..CUR_B + HASH_LEN].copy_from_slice(&b);
            next[IDX_LO] = lo;
            next[IDX_HI] = hi;
        }
    }
}

/// The carried state between operations.
struct Globals {
    root: Hash,
    size: u64,
    prev_ins: Hash,
    prev_sp: Hash,
    pins: BabyBear4,
    psp: BabyBear4,
}

impl Globals {
    fn write(&self, m: &mut RowMajorMatrix, start: usize, rows: usize) {
        let (lo, hi) = state::split(self.size);
        for r in start..start + rows {
            let row = m.row_mut(r);
            row[ROOT..ROOT + HASH_LEN].copy_from_slice(&self.root);
            row[SIZE_LO] = lo;
            row[SIZE_HI] = hi;
            row[PREV_INS..PREV_INS + HASH_LEN].copy_from_slice(&self.prev_ins);
            row[PREV_SP..PREV_SP + HASH_LEN].copy_from_slice(&self.prev_sp);
            row[PINS..PINS + 4].copy_from_slice(&self.pins.0);
            row[PSP..PSP + 4].copy_from_slice(&self.psp.0);
        }
    }
}

/// The values an operation's rows need.
struct OpFields {
    active: bool,
    key: Hash,
    /// The inserted or removed output's nonce.
    nonce: output::Nonce,
    /// Insert: the low leaf. Removal: the predecessor.
    low: Leaf,
    low_index: u64,
    low_siblings: [Hash; DEPTH],
    /// Insert: the path to slot `size`. Removal: the removed leaf's path.
    new_siblings: [Hash; DEPTH],
    /// Removal: the removed leaf's link and its position.
    removed_next_key: Hash,
    removed_next_index: u64,
    position: u64,
}

impl OpFields {
    /// A dummy satisfying the comparisons: low key 0 < key < end of list,
    /// with key just below every real key's maximum.
    fn dummy_insert() -> Self {
        let mut key = [BabyBear::new(MAX_LIMB); HASH_LEN];
        key[HASH_LEN - 1] = BabyBear::new(MAX_LIMB - 1);
        Self {
            active: false,
            key,
            nonce: [0; output::NONCE_LEN],
            low: Leaf::SENTINEL,
            low_index: 0,
            low_siblings: [[BabyBear::ZERO; HASH_LEN]; DEPTH],
            new_siblings: [[BabyBear::ZERO; HASH_LEN]; DEPTH],
            removed_next_key: [BabyBear::ZERO; HASH_LEN],
            removed_next_index: 0,
            position: 0,
        }
    }

    fn dummy_spend() -> Self {
        Self {
            key: [BabyBear::new(MAX_LIMB); HASH_LEN],
            ..Self::dummy_insert()
        }
    }

    fn write_constants(&self, m: &mut RowMajorMatrix, start: usize, rows: usize) {
        let (lo, hi) = state::split(self.low.next_index);
        let (rlo, rhi) = state::split(self.removed_next_index);
        let (plo, phi) = state::split(self.position);
        for r in start..start + rows {
            let row = m.row_mut(r);
            row[ACT] = BabyBear::from(self.active);
            row[KEY..KEY + HASH_LEN].copy_from_slice(&self.key);
            row[NONCE..NONCE + NONCE_LIMBS].copy_from_slice(&output::nonce_limbs(&self.nonce));
            row[LK..LK + HASH_LEN].copy_from_slice(&self.low.key);
            row[LN..LN + NONCE_LIMBS].copy_from_slice(&output::nonce_limbs(&self.low.nonce));
            row[LNK..LNK + HASH_LEN].copy_from_slice(&self.low.next_key);
            row[LNI_LO] = lo;
            row[LNI_HI] = hi;
            row[RNK..RNK + HASH_LEN].copy_from_slice(&self.removed_next_key);
            row[RNI_LO] = rlo;
            row[RNI_HI] = rhi;
            row[POS_LO] = plo;
            row[POS_HI] = phi;
        }
    }
}

/// Fills a comparison row proving `a < b`; if it doesn't hold, the row fails.
fn compare(m: &mut RowMajorMatrix, at: usize, a: &Hash, b: &Hash) {
    // The row's permutation is unused but must still be valid.
    poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(at));
    let row = m.row_mut(at);
    row[CA..CA + HASH_LEN].copy_from_slice(a);
    row[CB..CB + HASH_LEN].copy_from_slice(b);
    let j = (0..HASH_LEN)
        .find(|&j| a[j] != b[j])
        .unwrap_or(HASH_LEN - 1);
    row[E + j] = BabyBear::ONE;
    let (av, bv) = (a[j].as_u32(), b[j].as_u32());
    let put = |row: &mut [BabyBear], at: usize, v: u32, count: usize| {
        for i in 0..count {
            row[at + i] = BabyBear::new((v >> i) & 1);
        }
    };
    let (ah, al, bh, bl) = (av >> 16, av & 0xffff, bv >> 16, bv & 0xffff);
    put(row, ABITS, al, 16);
    put(row, ABITS + 16, ah, 15);
    put(row, BBITS, bl, 16);
    put(row, BBITS + 16, bh, 15);
    let g = HI_MAX.wrapping_sub(bh);
    put(row, GBITS, g, 15);
    row[GINV] = BabyBear::new(g).inverse().unwrap_or(BabyBear::ZERO);
    let hi_differs = ah != bh;
    row[F] = BabyBear::from(hi_differs);
    let d = if hi_differs {
        bh.wrapping_sub(ah).wrapping_sub(1)
    } else {
        bl.wrapping_sub(al).wrapping_sub(1)
    };
    put(row, DBITS, d, 16);
}

/// Fills four leaf-hash rows and returns the leaf hash.
fn leaf_rows(m: &mut RowMajorMatrix, at: usize, leaf: &Leaf) -> Hash {
    let (lo, hi) = state::split(leaf.next_index);
    let nonce = output::nonce_limbs(&leaf.nonce);
    for r in at..at + LEAF_ROWS {
        let row = m.row_mut(r);
        row[LEAF_K..LEAF_K + HASH_LEN].copy_from_slice(&leaf.key);
        row[LEAF_N..LEAF_N + NONCE_LIMBS].copy_from_slice(&nonce);
        row[LEAF_NK..LEAF_NK + HASH_LEN].copy_from_slice(&leaf.next_key);
        row[LEAF_LO] = lo;
        row[LEAF_HI] = hi;
    }
    let mut input = [BabyBear::ZERO; WIDTH];
    input[..HASH_LEN].copy_from_slice(&leaf.key);
    input[13] = BabyBear::new(domain::STATE_LEAF);
    input[14] = BabyBear::new(LEAF_LEN as u32);
    let mut state = poseidon2_air::generate(input, m.row_mut(at));
    let mut last = vec![lo, hi];
    last.extend(&nonce[HASH_LEN..]);
    for (offset, block) in [
        (1, &nonce[..HASH_LEN]),
        (2, &leaf.next_key[..]),
        (3, &last[..]),
    ] {
        let mut input = state;
        for (x, &v) in input.iter_mut().zip(block) {
            *x += v;
        }
        state = poseidon2_air::generate(input, m.row_mut(at + offset));
    }
    let h = core::array::from_fn(|j| state[j]);
    debug_assert_eq!(h, leaf.hash());
    h
}

/// Fills an interleaved path pair from leaf hashes `a` and `b` at `index`
/// with shared `siblings`; returns both roots.
fn path_rows(
    m: &mut RowMajorMatrix,
    at: usize,
    mut a: Hash,
    mut b: Hash,
    index: u64,
    siblings: &[Hash; DEPTH],
) -> (Hash, Hash) {
    let mut idx = 0u64;
    for (d, &sib) in siblings.iter().enumerate() {
        let bit = (index >> d) & 1;
        for (offset, is_a) in [(0, true), (1, false)] {
            let r = at + 2 * d + offset;
            let cur = if is_a { a } else { b };
            let (left, right) = if bit == 1 { (sib, cur) } else { (cur, sib) };
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(&left);
            input[HASH_LEN..].copy_from_slice(&right);
            let out = poseidon2_air::generate(input, m.row_mut(r));
            let node: Hash = core::array::from_fn(|j| out[j] + left[j]);
            let row = m.row_mut(r);
            row[CUR_A..CUR_A + HASH_LEN].copy_from_slice(&a);
            row[CUR_B..CUR_B + HASH_LEN].copy_from_slice(&b);
            row[SIB..SIB + HASH_LEN].copy_from_slice(&sib);
            row[BIT] = BabyBear::new(bit as u32);
            let (lo, hi) = state::split(idx);
            row[IDX_LO] = lo;
            row[IDX_HI] = hi;
            if is_a {
                a = node;
                idx += bit << d;
            } else {
                b = node;
            }
        }
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::db::test::TempEnv;
    use crate::fri::FriParams;
    use crate::item_chunk_air::{input_element, output_element, product};
    use crate::output::NONCE_LEN;
    use crate::stark;
    use crate::state::StateTree;

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);

    fn record(n: u32) -> OutputRecord {
        let nonce = [n as u8; NONCE_LEN];
        let hash = crate::hash::to_bytes(&BabyBear::new_array(core::array::from_fn(|i| {
            n * 10 + i as u32
        })));
        OutputRecord {
            commitment: output::commit(&hash, n as u64, &nonce).unwrap(),
            nonce,
        }
    }

    fn sorted(mut records: Vec<OutputRecord>) -> Vec<OutputRecord> {
        records.sort_by_key(|r| key_of(&r.commitment).unwrap().map(|x| x.as_u32()));
        records
    }

    struct Fixture {
        tmp: TempEnv,
        tree: StateTree,
    }

    impl Fixture {
        /// A tree already holding `existing`, inserted in sorted order.
        fn new(existing: &[u32]) -> Self {
            let tmp = TempEnv::new();
            let tree = StateTree::open(tmp.env()).unwrap();
            let mut txn = tmp.env().write_txn().unwrap();
            for r in sorted(existing.iter().map(|&n| record(n)).collect()) {
                tree.insert(&mut txn, &r).unwrap();
            }
            txn.commit().unwrap();
            Self { tmp, tree }
        }

        /// Applies the operations natively and returns them with proofs.
        fn ops(
            &self,
            inserts: &[u32],
            spends: &[u32],
        ) -> (State, Vec<InsertOp>, Vec<SpendOp>, State) {
            let mut txn = self.tmp.env().write_txn().unwrap();
            let before = self.tree.state(&txn).unwrap();
            let ins = sorted(inserts.iter().map(|&n| record(n)).collect())
                .into_iter()
                .map(|record| InsertOp {
                    proof: self.tree.insert(&mut txn, &record).unwrap(),
                    record,
                })
                .collect();
            let sp = sorted(spends.iter().map(|&n| record(n)).collect())
                .into_iter()
                .map(|r| SpendOp {
                    proof: self.tree.remove(&mut txn, &r.commitment).unwrap(),
                    commitment: r.commitment,
                })
                .collect();
            let after = self.tree.state(&txn).unwrap();
            txn.abort();
            (before, ins, sp, after)
        }
    }

    fn start(state: State) -> SegmentStart {
        SegmentStart {
            state,
            prev_insert: [BabyBear::ZERO; HASH_LEN],
            prev_spend: [BabyBear::ZERO; HASH_LEN],
        }
    }

    fn state_at(public: &[BabyBear], root: usize, size: usize) -> (Hash, u64) {
        let h = core::array::from_fn(|j| public[root + j]);
        let s = public[size].as_u32() as u64 + ((public[size + 1].as_u32() as u64) << 30);
        (h, s)
    }

    #[test]
    fn layout() {
        assert_eq!(INS_ROWS, 176);
        assert_eq!(SP_ROWS, 174);
    }

    #[test]
    fn inserts_and_spends_match_native_state() {
        let f = Fixture::new(&[1, 2, 3]);
        let (before, ins, sp, after) = f.ops(&[10, 11, 12], &[2, 10]);
        let air = StateSegmentAir {
            inserts: 3,
            spends: 2,
        };
        let (t, public) = air.trace(&start(before), &ins, &sp, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        assert_eq!(
            state_at(&public, public::NEW_ROOT, public::NEW_SIZE),
            (after.root, after.size)
        );
        let p_ins = product(
            R,
            ins.iter()
                .map(|o| output_element(LAMBDA, &o.record).unwrap()),
        );
        let p_sp = product(
            R,
            sp.iter()
                .map(|o| input_element(LAMBDA, &o.commitment, o.proof.position()).unwrap()),
        );
        assert_eq!(&public[public::P_INS..public::P_INS + 4], &p_ins.0);
        assert_eq!(&public[public::P_SP..public::P_SP + 4], &p_sp.0);
    }

    #[test]
    fn partially_filled_segment() {
        let f = Fixture::new(&[1, 2]);
        let (before, ins, sp, after) = f.ops(&[7], &[1]);
        let air = StateSegmentAir {
            inserts: 3,
            spends: 2,
        };
        let (t, public) = air.trace(&start(before), &ins, &sp, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        assert_eq!(
            state_at(&public, public::NEW_ROOT, public::NEW_SIZE),
            (after.root, after.size)
        );
    }

    #[test]
    fn empty_segment_keeps_state() {
        let f = Fixture::new(&[1]);
        let (before, ..) = f.ops(&[], &[]);
        let air = StateSegmentAir {
            inserts: 1,
            spends: 1,
        };
        let (t, public) = air.trace(&start(before), &[], &[], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        assert_eq!(
            state_at(&public, public::NEW_ROOT, public::NEW_SIZE),
            (before.root, before.size)
        );
    }

    #[test]
    fn segments_chain() {
        // Two segments in sequence: the second starts where the first ends.
        let f = Fixture::new(&[1, 2, 3, 4]);
        let (before, ins, sp, after) = f.ops(&[20, 21, 22, 23], &[1, 3]);
        let air = StateSegmentAir {
            inserts: 2,
            spends: 1,
        };
        let (t1, p1) = air.trace(&start(before), &ins[..2], &[], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t1, &p1), Ok(()));
        let mid = SegmentStart {
            state: {
                let (root, size) = state_at(&p1, public::NEW_ROOT, public::NEW_SIZE);
                State { root, size }
            },
            prev_insert: core::array::from_fn(|j| p1[public::PREV_INS_OUT + j]),
            prev_spend: core::array::from_fn(|j| p1[public::PREV_SP_OUT + j]),
        };
        let (t2, p2) = air.trace(&mid, &ins[2..], &sp[..1], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t2, &p2), Ok(()));
        let mid2 = SegmentStart {
            state: {
                let (root, size) = state_at(&p2, public::NEW_ROOT, public::NEW_SIZE);
                State { root, size }
            },
            prev_insert: core::array::from_fn(|j| p2[public::PREV_INS_OUT + j]),
            prev_spend: core::array::from_fn(|j| p2[public::PREV_SP_OUT + j]),
        };
        let (t3, p3) = air.trace(&mid2, &[], &sp[1..], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t3, &p3), Ok(()));
        assert_eq!(
            state_at(&p3, public::NEW_ROOT, public::NEW_SIZE),
            (after.root, after.size)
        );
    }

    #[test]
    fn rejects_wrong_claimed_state() {
        let f = Fixture::new(&[1, 2]);
        let (before, ins, sp, _) = f.ops(&[5], &[2]);
        let air = StateSegmentAir {
            inserts: 1,
            spends: 1,
        };
        let (t, public) = air.trace(&start(before), &ins, &sp, LAMBDA, R);
        for at in [
            public::OLD_ROOT,
            public::NEW_ROOT + 3,
            public::NEW_SIZE,
            public::P_INS,
            public::P_SP + 2,
        ] {
            let mut bad = public.clone();
            bad[at] += BabyBear::ONE;
            assert!(check_constraints(&air, &t, &bad).is_err(), "public {at}");
        }
    }

    #[test]
    fn rejects_out_of_order_inserts() {
        // The tree accepts any order, but the segment must not.
        let f = Fixture::new(&[1]);
        let mut txn = f.tmp.env().write_txn().unwrap();
        let before = f.tree.state(&txn).unwrap();
        let mut records = sorted(vec![record(30), record(31)]);
        records.reverse();
        let ins: Vec<InsertOp> = records
            .into_iter()
            .map(|record| InsertOp {
                proof: f.tree.insert(&mut txn, &record).unwrap(),
                record,
            })
            .collect();
        let air = StateSegmentAir {
            inserts: 2,
            spends: 0,
        };
        let (t, public) = air.trace(&start(before), &ins, &[], LAMBDA, R);
        let err = check_constraints(&air, &t, &public).unwrap_err();
        assert_eq!(err, air.insert_base(1) + ins::ORD);
    }

    #[test]
    fn rejects_duplicate_insert() {
        // Re-insert an existing key using its true predecessor as the low
        // leaf: that leaf's next key is the key itself, so `key < next key`
        // fails.
        let f = Fixture::new(&[1, 2, 3]);
        let txn = f.tmp.env().read_txn().unwrap();
        let before = f.tree.state(&txn).unwrap();
        let target = record(2);
        let key = key_of(&target.commitment).unwrap();
        let sorted_keys: Vec<Hash> = sorted(vec![record(1), record(2), record(3)])
            .iter()
            .map(|r| key_of(&r.commitment).unwrap())
            .collect();
        let pos = sorted_keys.iter().position(|k| *k == key).unwrap();
        let low_index = if pos == 0 {
            0
        } else {
            let pred = sorted(vec![record(1), record(2), record(3)])[pos - 1];
            f.tree.get(&txn, &pred.commitment).unwrap().unwrap().0
        };
        let low = f.tree.witness(&txn, low_index).unwrap();
        assert_eq!(low.leaf.next_key, key);
        let ins = [InsertOp {
            record: target,
            proof: InsertProof {
                low,
                new_siblings: Box::new([[BabyBear::ZERO; HASH_LEN]; DEPTH]),
            },
        }];
        let air = StateSegmentAir {
            inserts: 1,
            spends: 0,
        };
        let (t, public) = air.trace(&start(before), &ins, &[], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Err(ins::NEXT));
    }

    #[test]
    fn removal_frees_the_commitment_for_a_new_slot() {
        // Spend 1, then create it again: it lands in a new slot.
        let f = Fixture::new(&[1, 2]);
        let (before, _, sp, after) = f.ops(&[], &[1]);
        let air = StateSegmentAir {
            inserts: 1,
            spends: 1,
        };
        let (t, public) = air.trace(&start(before), &[], &sp, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.remove(&mut txn, &record(1).commitment).unwrap();
        assert_eq!(f.tree.state(&txn).unwrap(), after);
        let ins = [InsertOp {
            proof: f.tree.insert(&mut txn, &record(1)).unwrap(),
            record: record(1),
        }];
        let (t, public) = air.trace(&start(after), &ins, &[], LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        assert_eq!(
            state_at(&public, public::NEW_ROOT, public::NEW_SIZE),
            (f.tree.state(&txn).unwrap().root, after.size + 1)
        );
    }

    #[test]
    fn rejects_double_spend() {
        // Remove 1 for real, then replay the earlier removal proof: the
        // predecessor no longer links to it, so its old hash doesn't reach
        // the root.
        let f = Fixture::new(&[1, 2]);
        let (_, _, sp, _) = f.ops(&[], &[1]);
        let mut txn = f.tmp.env().write_txn().unwrap();
        f.tree.remove(&mut txn, &record(1).commitment).unwrap();
        let spent_state = f.tree.state(&txn).unwrap();
        let air = StateSegmentAir {
            inserts: 0,
            spends: 1,
        };
        let (t, public) = air.trace(&start(spent_state), &[], &sp, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Err(sp::H_REMOVED));
    }

    #[test]
    fn rejects_spending_a_missing_key() {
        let f = Fixture::new(&[1, 2]);
        let (before, _, mut sp, _) = f.ops(&[], &[1]);
        sp[0].commitment = record(9).commitment;
        let air = StateSegmentAir {
            inserts: 0,
            spends: 1,
        };
        let (t, public) = air.trace(&start(before), &[], &sp, LAMBDA, R);
        // The predecessor hashed as linking to the wrong key doesn't reach
        // the root.
        assert_eq!(check_constraints(&air, &t, &public), Err(sp::H_REMOVED));
    }

    #[test]
    fn rejects_a_wrong_position_or_nonce() {
        let f = Fixture::new(&[1, 2, 3]);
        let (before, _, sp, _) = f.ops(&[], &[2]);
        let air = StateSegmentAir {
            inserts: 0,
            spends: 1,
        };
        // Claiming another slot: the predecessor's link no longer matches.
        let mut bad = sp.clone();
        bad[0].proof.leaf.index += 1;
        let (t, public) = air.trace(&start(before), &[], &bad, LAMBDA, R);
        assert!(check_constraints(&air, &t, &public).is_err());
        // Another nonce: the removed leaf doesn't reach the intermediate
        // root.
        let mut bad = sp.clone();
        bad[0].proof.leaf.leaf.nonce[0] ^= 1;
        let (t, public) = air.trace(&start(before), &[], &bad, LAMBDA, R);
        assert_eq!(check_constraints(&air, &t, &public), Err(sp::END));
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let f = Fixture::new(&[1, 2]);
        let (before, ins, sp, _) = f.ops(&[3], &[1]);
        let air = StateSegmentAir {
            inserts: 1,
            spends: 1,
        };
        let (t, public) = air.trace(&start(before), &ins, &sp, LAMBDA, R);
        let (pk, vk) = stark::setup(&params, &air);
        let proof = stark::prove(&params, &pk, &air, t, &public);
        assert_eq!(stark::verify(&params, &vk, &air, &public, &proof), Ok(()));
    }
}
