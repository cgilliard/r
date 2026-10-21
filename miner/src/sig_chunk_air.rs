//! The signature chunk circuit: proves a run of WOTS signatures, revealing
//! only a multiset fingerprint of the `(key root, message)` pairs they sign
//! and a blinded hash of them.
//!
//! A key root is the root of the output's key tree
//! ([`crate::transaction::KeyTree`]) that the signing key is a leaf of. The
//! aggregation matches the fingerprint against the item chunks' (each input
//! contributes `(its output's hash, its transaction's digest)`,
//! [`crate::item_chunk_air`]), so every input is signed by one of its
//! output's keys. The hash fixes the pairs before the block's challenges
//! are derived.
//!
//! Rows: `slots` slots, each a WOTS region verifying a signature on the
//! slot's message ([`wots_air`]), a `PKH` row hashing the public key,
//! [`KEY_LEVELS`] merge rows walking the key's path to the root (a bit and
//! a sibling per row), and two rows absorbing the root then the message
//! into the pair-hash sponge (whose initial capacity is a random blinding);
//! then a final row. Each slot has an active flag; inactive slots (dummy
//! signatures) contribute nothing.
//!
//! Public values: `lambda (4) | r (4) | P_sig (4) | hash (8)`.

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::hash::{HASH_LEN, Hash, domain};
use crate::item_chunk_air::sig_element;
use crate::matrix::RowMajorMatrix;
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_air::{self, OUTPUT_OFFSET};
use crate::rng::Rng;
use crate::transaction::{KEY_LEVELS, KeyPath, key_root};
use crate::wots::{self, PARAM_LEN, PublicKey, Signature};
use crate::wots_air;

/// Rows per slot: the WOTS region, `PKH`, the merges, two hash rows.
pub const SLOT_ROWS: usize = wots_air::NUM_ROWS + 1 + KEY_LEVELS + 2;
/// `x^4 = W` in the extension field.
const EXT_W: BabyBear = BabyBear::new(11);

// Columns after the WOTS region's (which uses `MSG` and `PARAM`).
const T: usize = wots_air::NUM_COLS;
/// The slot's active flag, constant in the slot.
const ACT: usize = T;
/// The pair-hash sponge, carried.
const SIGS: usize = ACT + 1;
/// The fingerprint product, carried.
const PSIG: usize = SIGS + WIDTH;
/// On merge rows: the path's bit (1: the running node is the right child)
/// and sibling.
const BIT: usize = PSIG + 4;
const SIB: usize = BIT + 1;
pub const NUM_COLS: usize = SIB + HASH_LEN;

// Preprocessed columns after the WOTS region's.
const PW: usize = wots_air::PRE_WIDTH;
const PKH: usize = PW;
const H1: usize = PW + 1;
const H2: usize = PW + 2;
/// Slot rows followed by a row of the same slot.
const SAME_SLOT: usize = PW + 3;
/// Every slot row.
const SLOTS: usize = PW + 4;
const FIRST: usize = PW + 5;
const FINAL: usize = PW + 6;
/// Merge rows; rows followed by one (`PKH` and all merges but the last);
/// the last (whose result is the root).
const MERGE: usize = PW + 7;
const STEP: usize = PW + 8;
const ROOT: usize = PW + 9;
pub const PRE_WIDTH: usize = PW + 10;

/// Offsets of the public values.
pub mod public {
    pub const LAMBDA: usize = 0;
    pub const R: usize = 4;
    pub const P_SIG: usize = 8;
    pub const HASH: usize = 12;
    pub const LEN: usize = 20;
}

use public as pv;

/// A chunk's capacity in signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigChunkAir {
    pub slots: usize,
}

impl SigChunkAir {
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

/// A signature to prove: `pubkey`, at `leaf` of a key tree (with `path`),
/// signed `msg`.
#[derive(Clone, Debug)]
pub struct SigItem {
    pub pubkey: PublicKey,
    pub leaf: u8,
    pub path: KeyPath,
    pub sig: Signature,
    pub msg: Hash,
}

impl SigItem {
    /// The root of the key tree.
    pub fn root(&self) -> Hash {
        key_root(&self.pubkey.hash(), self.leaf, &self.path)
    }
}

/// The pair hash of a chunk: a sponge with `blinding` as initial capacity,
/// absorbing each signature's key root then message.
pub fn chunk_hash(blinding: &Hash, pairs: &[(Hash, Hash)]) -> Hash {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[HASH_LEN..].copy_from_slice(blinding);
    for (pkh, msg) in pairs {
        for block in [pkh, msg] {
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

impl Air for SigChunkAir {
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
        for s in 0..self.slots {
            let base = self.base(s);
            for (r, row) in wots.iter().enumerate() {
                m.row_mut(base + r)[..PW].copy_from_slice(row);
            }
            let tail = base + wots_air::NUM_ROWS;
            m.row_mut(tail)[PKH] = one;
            m.row_mut(tail)[STEP] = one;
            for d in 1..=KEY_LEVELS {
                m.row_mut(tail + d)[MERGE] = one;
                if d < KEY_LEVELS {
                    m.row_mut(tail + d)[STEP] = one;
                }
            }
            m.row_mut(tail + KEY_LEVELS)[ROOT] = one;
            m.row_mut(tail + KEY_LEVELS + 1)[H1] = one;
            m.row_mut(tail + KEY_LEVELS + 2)[H2] = one;
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
        let pub4 = |at: usize| -> [V; 4] { core::array::from_fn(|b| public[at + b]) };
        let (lambda, r) = (pub4(pv::LAMBDA), pub4(pv::R));
        let act = l[ACT];
        let slots = p[SLOTS];

        poseidon2_air::eval(c, l);
        wots_air::eval(c, None);

        // Slot constants: message, parameter, active flag.
        for col in (wots_air::PARAM..wots_air::PARAM + PARAM_LEN)
            .chain(wots_air::MSG..wots_air::MSG + HASH_LEN)
            .chain([ACT])
        {
            c.assert_zero(p[SAME_SLOT] * (n[col] - l[col]));
        }
        c.assert_zero(slots * act * (act - one));

        // The last WOTS row's root feeds PKH: root | param | PUBLIC_KEY_HASH
        // | 0 | 0, and pkh = output + root (compression).
        let last_absorb = p[wots_air::LAST_ABSORB];
        for (j, &x) in n[..HASH_LEN].iter().enumerate() {
            c.assert_zero(last_absorb * (x - out(j)));
        }
        let pkh_row = p[PKH];
        for j in 0..PARAM_LEN {
            c.assert_zero(pkh_row * (l[HASH_LEN + j] - l[wots_air::PARAM + j]));
        }
        c.assert_zero(pkh_row * (l[13] - k(domain::PUBLIC_KEY_HASH)));
        c.assert_zero(pkh_row * l[14]);
        c.assert_zero(pkh_row * l[15]);
        // A compression's result (PKH and merge rows): output + input.
        let cur: [V; HASH_LEN] = core::array::from_fn(|j| out(j) + l[j]);

        // Merges: the next row hashes the running node with its sibling,
        // in the order the bit gives.
        let step = p[STEP];
        let (bit, sib) = (n[BIT], &n[SIB..SIB + HASH_LEN]);
        c.assert_zero(p[MERGE] * l[BIT] * (l[BIT] - one));
        for j in 0..HASH_LEN {
            c.assert_zero(step * (n[j] - cur[j] - bit * (sib[j] - cur[j])));
            c.assert_zero(step * (n[HASH_LEN + j] - sib[j] - bit * (cur[j] - sib[j])));
        }
        let root = cur;
        let root_row = p[ROOT];

        // The fingerprint, on the root.
        let mut powers = [[zero; 4]; 2 * HASH_LEN];
        powers[0][0] = one;
        for j in 1..powers.len() {
            powers[j] = ext_mul(powers[j - 1], lambda);
        }
        let e = root
            .iter()
            .chain(&l[wots_air::MSG..wots_air::MSG + HASH_LEN])
            .zip(&powers)
            .fold([zero; 4], |acc, (&x, pw)| {
                core::array::from_fn(|b| acc[b] + x * pw[b])
            });
        let factor: [V; 4] = core::array::from_fn(|b| {
            let unit = if b == 0 { one } else { zero };
            unit - act * unit + act * (r[b] - e[b])
        });
        let next = ext_mul(core::array::from_fn(|b| l[PSIG + b]), factor);
        for b in 0..4 {
            c.assert_zero(root_row * (n[PSIG + b] - next[b]));
            c.assert_zero_transition(slots * (one - root_row) * (n[PSIG + b] - l[PSIG + b]));
        }

        // Pair hash: an active slot's H1 absorbs the root into the sponge,
        // H2 its message; H2's output is the new sponge state.
        for j in 0..WIDTH {
            let absorbed = if j < HASH_LEN { root[j] } else { zero };
            c.assert_zero(root_row * act * (n[j] - l[SIGS + j] - absorbed));
            let absorbed = if j < HASH_LEN {
                l[wots_air::MSG + j]
            } else {
                zero
            };
            c.assert_zero(p[H1] * act * (n[j] - out(j) - absorbed));
            let updated = act * out(j) + (one - act) * l[SIGS + j];
            c.assert_zero(p[H2] * (n[SIGS + j] - updated));
            c.assert_zero_transition(slots * (one - p[H2]) * (n[SIGS + j] - l[SIGS + j]));
        }

        // First row: empty product, zero rate (the capacity is the
        // blinding). Final row: the totals.
        let first = p[FIRST];
        c.assert_zero(first * (l[PSIG] - one));
        for b in 1..4 {
            c.assert_zero(first * l[PSIG + b]);
        }
        for j in 0..HASH_LEN {
            c.assert_zero(first * l[SIGS + j]);
        }
        let fin = p[FINAL];
        for b in 0..4 {
            c.assert_zero(fin * (l[PSIG + b] - public[pv::P_SIG + b]));
        }
        for j in 0..HASH_LEN {
            c.assert_zero(fin * (l[SIGS + j] - public[pv::HASH + j]));
        }
    }
}

impl SigChunkAir {
    /// The trace and public values proving `sigs` (at most the capacity),
    /// with the pair hash blinded by `blinding`. Spare slots get dummy keys
    /// drawn from `rng`.
    ///
    /// # Panics
    /// If there are too many signatures.
    pub fn trace(
        &self,
        sigs: &[SigItem],
        blinding: &Hash,
        lambda: BabyBear4,
        r: BabyBear4,
        rng: &mut Rng,
    ) -> (RowMajorMatrix, Vec<BabyBear>) {
        assert!(sigs.len() <= self.slots, "signatures don't fit the chunk");
        let mut m = RowMajorMatrix::zeros(self.height(), NUM_COLS);
        let mut sponge = [BabyBear::ZERO; WIDTH];
        sponge[HASH_LEN..].copy_from_slice(blinding);
        let mut psig = BabyBear4::ONE;
        let write = |row: &mut [BabyBear], sponge: &[BabyBear; WIDTH], psig: BabyBear4| {
            row[SIGS..SIGS + WIDTH].copy_from_slice(sponge);
            row[PSIG..PSIG + 4].copy_from_slice(&psig.0);
        };
        for s in 0..self.slots {
            let base = self.base(s);
            let (item, active) = match sigs.get(s) {
                Some(item) => (item.clone(), true),
                None => (dummy(rng), false),
            };
            wots_air::fill(&mut m, base, &item.pubkey, &item.msg, &item.sig);
            let tail = base + wots_air::NUM_ROWS;
            let mut input = [BabyBear::ZERO; WIDTH];
            input[..HASH_LEN].copy_from_slice(&item.pubkey.root);
            input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&item.pubkey.param);
            input[13] = BabyBear::new(domain::PUBLIC_KEY_HASH);
            let out = poseidon2_air::generate(input, m.row_mut(tail));
            let mut cur: Hash = core::array::from_fn(|j| out[j] + input[j]);
            debug_assert_eq!(cur, item.pubkey.hash());
            for (d, sib) in item.path.iter().enumerate() {
                let bit = item.leaf >> d & 1 == 1;
                let (left, right) = if bit { (sib, &cur) } else { (&cur, sib) };
                let mut input = [BabyBear::ZERO; WIDTH];
                input[..HASH_LEN].copy_from_slice(left);
                input[HASH_LEN..].copy_from_slice(right);
                let row = m.row_mut(tail + 1 + d);
                row[BIT] = BabyBear::from(bit);
                row[SIB..SIB + HASH_LEN].copy_from_slice(sib);
                let out = poseidon2_air::generate(input, row);
                cur = core::array::from_fn(|j| out[j] + input[j]);
            }
            let root = cur;
            debug_assert_eq!(root, item.root());
            let (h1_row, h2_row) = (tail + KEY_LEVELS + 1, tail + KEY_LEVELS + 2);
            let mut h1 = [BabyBear::ZERO; WIDTH];
            let mut h2 = [BabyBear::ZERO; WIDTH];
            if active {
                h1 = sponge;
                for (x, &v) in h1.iter_mut().zip(&root) {
                    *x += v;
                }
            }
            let out1 = poseidon2_air::generate(h1, m.row_mut(h1_row));
            if active {
                h2 = out1;
                for (x, &v) in h2.iter_mut().zip(&item.msg) {
                    *x += v;
                }
            }
            let out2 = poseidon2_air::generate(h2, m.row_mut(h2_row));
            for r in base..base + SLOT_ROWS {
                let row = m.row_mut(r);
                row[ACT] = BabyBear::from(active);
                row[wots_air::MSG..wots_air::MSG + HASH_LEN].copy_from_slice(&item.msg);
                row[wots_air::PARAM..wots_air::PARAM + PARAM_LEN]
                    .copy_from_slice(&item.pubkey.param);
                // The carried values change after the root (product) and
                // H2 (sponge).
                if r <= tail + KEY_LEVELS {
                    write(row, &sponge, psig);
                }
            }
            let new_psig = if active {
                psig * (r - sig_element(lambda, &root, &item.msg))
            } else {
                psig
            };
            for row in [h1_row, h2_row] {
                write(m.row_mut(row), &sponge, new_psig);
            }
            psig = new_psig;
            if active {
                sponge = out2;
            }
        }
        let fin = self.final_row();
        for row in fin..m.height() {
            poseidon2_air::generate([BabyBear::ZERO; WIDTH], m.row_mut(row));
        }
        write(m.row_mut(fin), &sponge, psig);
        let mut public = Vec::with_capacity(pv::LEN);
        for v in [lambda, r, psig] {
            public.extend_from_slice(&v.0);
        }
        public.extend_from_slice(&sponge[..HASH_LEN]);
        (m, public)
    }
}

/// A dummy signature for a spare slot: a fresh key signing zero.
fn dummy(rng: &mut Rng) -> SigItem {
    let mut seed = [BabyBear::ZERO; wots::SEED_LEN];
    rng.fill(&mut seed);
    let (sk, pubkey) = wots::keygen(seed);
    let msg = [BabyBear::ZERO; HASH_LEN];
    SigItem {
        sig: sk.sign(&msg),
        pubkey,
        leaf: 0,
        path: [[BabyBear::ZERO; HASH_LEN]; KEY_LEVELS],
        msg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::fri::FriParams;
    use crate::item_chunk_air::product;
    use crate::stark;
    use crate::transaction::KeyTree;

    const LAMBDA: BabyBear4 = BabyBear4::new([3, 5, 7, 11]);
    const R: BabyBear4 = BabyBear4::new([13, 17, 19, 23]);
    const BLINDING: Hash = BabyBear::new_array([29, 31, 37, 41, 43, 47, 53, 59]);

    fn signed(n: u32, msg: u32) -> SigItem {
        let (sk, pubkey) = wots::keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 31 + i as u32
        })));
        let msg = [BabyBear::new(msg); HASH_LEN];
        // At leaf `n * 37 % 256` of a tree of made-up hashes.
        let leaf = (n * 37 % 256) as u8;
        let mut tree = KeyTree(core::array::from_fn(|i| {
            BabyBear::new_array([n * 1000 + i as u32; HASH_LEN])
        }));
        tree.0[leaf as usize] = pubkey.hash();
        SigItem {
            sig: sk.sign(&msg),
            pubkey,
            leaf,
            path: tree.path(leaf as usize),
            msg,
        }
    }

    fn expected(sigs: &[SigItem]) -> (BabyBear4, Hash) {
        let pairs: Vec<(Hash, Hash)> = sigs.iter().map(|s| (s.root(), s.msg)).collect();
        (
            product(R, pairs.iter().map(|(p, m)| sig_element(LAMBDA, p, m))),
            chunk_hash(&BLINDING, &pairs),
        )
    }

    #[test]
    fn proves_the_pairs() {
        let sigs = [signed(1, 10), signed(2, 20), signed(1, 30)];
        let air = SigChunkAir { slots: 4 };
        let (t, public) = air.trace(&sigs, &BLINDING, LAMBDA, R, &mut Rng::from_seed(1));
        assert_eq!(check_constraints(&air, &t, &public), Ok(()));
        let (p, h) = expected(&sigs);
        assert_eq!(&public[pv::P_SIG..pv::P_SIG + 4], &p.0);
        assert_eq!(&public[pv::HASH..pv::LEN], &h);
    }

    #[test]
    fn rejects_a_bad_signature_or_public_values() {
        let mut sigs = [signed(1, 10), signed(2, 20)];
        let air = SigChunkAir { slots: 2 };
        let (t, public) = air.trace(&sigs, &BLINDING, LAMBDA, R, &mut Rng::from_seed(1));
        for at in [pv::P_SIG, pv::HASH + 3] {
            let mut bad = public.clone();
            bad[at] += BabyBear::ONE;
            assert!(check_constraints(&air, &t, &bad).is_err(), "public {at}");
        }
        // A signature on another message.
        sigs[1].msg = [BabyBear::new(21); HASH_LEN];
        let (t, public) = air.trace(&sigs, &BLINDING, LAMBDA, R, &mut Rng::from_seed(1));
        assert!(check_constraints(&air, &t, &public).is_err());
        // A path bit or sibling changed in the trace (the root stays).
        let sigs = [signed(1, 10), signed(2, 20)];
        let (t, public) = air.trace(&sigs, &BLINDING, LAMBDA, R, &mut Rng::from_seed(1));
        let merge = wots_air::NUM_ROWS + 2;
        for col in [BIT, SIB + 3] {
            let mut bad = t.clone();
            let v = bad.row_mut(merge)[col];
            bad.row_mut(merge)[col] = v + BabyBear::ONE;
            assert!(check_constraints(&air, &bad, &public).is_err(), "col {col}");
        }
    }

    #[test]
    fn proves_and_verifies() {
        let params = FriParams {
            num_queries: 8,
            grinding_bits: 4,
            ..FriParams::default()
        };
        let air = SigChunkAir { slots: 2 };
        let (t, public) = air.trace(
            &[signed(1, 5)],
            &BLINDING,
            LAMBDA,
            R,
            &mut Rng::from_seed(1),
        );
        let (pk, vk) = stark::setup(&params, &air);
        let proof = stark::prove(&params, &pk, &air, t, &public);
        assert_eq!(stark::verify(&params, &vk, &air, &public, &proof), Ok(()));
    }
}
