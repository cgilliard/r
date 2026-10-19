//! Arithmetic circuits: cells holding field elements, and the operations
//! relating them. The recursive verifier is written against [`Circuit`];
//! the recursion AIR enforces each operation kind.
//!
//! Besides base-field arithmetic, the operations include fused forms of the
//! verifier's heavy patterns (sponge blocks, the transcript's duplex, Merkle
//! path steps, running dot products, extension multiply-adds), each of which
//! the recursion AIR handles in one row.
//!
//! Building is eager: every operation computes its result immediately, so
//! running a verifier "in-circuit" both records the program and fills in the
//! witness. Failed assertions are noted rather than panicking, so a bad
//! proof yields a complete program with [`Circuit::failures`]. The program
//! never branches on values, so its shape depends only on the parameters.
//! [`Circuit::check`] re-executes the recorded operations from scratch,
//! which is the semantics a recursion AIR must enforce.

use std::collections::HashMap;

use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::hash::HASH_LEN;
use crate::poseidon2::{self, WIDTH};

pub type Cell = u32;

/// An extension-field value: four cells holding its coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtCell(pub [Cell; 4]);

/// Values a [`Op::Dot`] adds per operation, at most.
pub const DOT_WIDTH: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// A value supplied with the program (proof data or public values).
    Input(Cell),
    /// A value supplied by the prover, constrained by later operations.
    Hint(Cell),
    Const(Cell, BabyBear),
    /// `out = a + b`, as `(a, b, out)`.
    Add(Cell, Cell, Cell),
    Sub(Cell, Cell, Cell),
    Mul(Cell, Cell, Cell),
    /// `out = a * b + c` in the extension field.
    ExtFma {
        a: ExtCell,
        b: ExtCell,
        c: ExtCell,
        out: ExtCell,
    },
    /// `out = Poseidon2(state + absorb)`, `absorb` added into the first
    /// rate positions (at most 8).
    Sponge {
        state: Box<[Cell; WIDTH]>,
        absorb: Vec<Cell>,
        out: Box<[Cell; WIDTH]>,
    },
    /// The transcript's duplex: overwrite the first `input.len()` positions,
    /// add `input.len()` to the last, then permute.
    Duplex {
        state: Box<[Cell; WIDTH]>,
        input: Vec<Cell>,
        out: Box<[Cell; WIDTH]>,
    },
    /// One Merkle path level: with `(l, r) = bit ? (sibling, node) : (node,
    /// sibling)`, `out = Poseidon2(l || r)[..8] + l`. `bit` must be 0 or 1.
    MerkleStep {
        node: [Cell; HASH_LEN],
        sibling: [Cell; HASH_LEN],
        bit: Cell,
        out: [Cell; HASH_LEN],
    },
    /// A running dot product with powers of `gamma`: with `p_0 = pow`,
    /// `acc_out = acc + sum_k p_k values[k]`, `p_(k+1) = p_k gamma`,
    /// `pow_out = p_(len)`. `values` are base cells (at most 8).
    Dot {
        acc: ExtCell,
        pow: ExtCell,
        gamma: ExtCell,
        values: Vec<Cell>,
        acc_out: ExtCell,
        pow_out: ExtCell,
    },
    AssertEq(Cell, Cell),
    /// `value = sum_i bits[i] 2^i`, every bit 0 or 1.
    Bits(Cell, Vec<Cell>),
}

/// Operation counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub cells: usize,
    pub inputs: usize,
    pub hints: usize,
    pub consts: usize,
    pub base_ops: usize,
    pub ext_fmas: usize,
    pub sponges: usize,
    pub duplexes: usize,
    pub merkle_steps: usize,
    pub dots: usize,
    pub asserts: usize,
    pub bits: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Circuit {
    values: Vec<BabyBear>,
    ops: Vec<Op>,
    consts: HashMap<u32, Cell>,
    failures: Vec<usize>,
}

/// The semantics of [`Op::MerkleStep`].
pub fn merkle_step(
    node: &[BabyBear; HASH_LEN],
    sibling: &[BabyBear; HASH_LEN],
    bit: bool,
) -> [BabyBear; HASH_LEN] {
    let (l, r) = if bit {
        (sibling, node)
    } else {
        (node, sibling)
    };
    let mut state = [BabyBear::ZERO; WIDTH];
    state[..HASH_LEN].copy_from_slice(l);
    state[HASH_LEN..].copy_from_slice(r);
    poseidon2::permute(&mut state);
    core::array::from_fn(|j| state[j] + l[j])
}

/// The semantics of [`Op::Sponge`].
pub fn sponge_block(state: &[BabyBear; WIDTH], absorb: &[BabyBear]) -> [BabyBear; WIDTH] {
    let mut s = *state;
    for (x, &a) in s.iter_mut().zip(absorb) {
        *x += a;
    }
    poseidon2::permute(&mut s);
    s
}

/// The semantics of [`Op::Duplex`].
pub fn duplex(state: &[BabyBear; WIDTH], input: &[BabyBear]) -> [BabyBear; WIDTH] {
    let mut s = *state;
    s[..input.len()].copy_from_slice(input);
    s[WIDTH - 1] += BabyBear::new(input.len() as u32);
    poseidon2::permute(&mut s);
    s
}

/// The semantics of [`Op::Dot`]: `(acc_out, pow_out)`.
pub fn dot(
    acc: BabyBear4,
    pow: BabyBear4,
    gamma: BabyBear4,
    values: &[BabyBear],
) -> (BabyBear4, BabyBear4) {
    let (mut acc, mut pow) = (acc, pow);
    for &v in values {
        acc += pow * v;
        pow *= gamma;
    }
    (acc, pow)
}

impl Circuit {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn value(&self, c: Cell) -> BabyBear {
        self.values[c as usize]
    }

    pub fn ext_value(&self, e: ExtCell) -> BabyBear4 {
        BabyBear4(e.0.map(|c| self.value(c)))
    }

    pub fn num_cells(&self) -> usize {
        self.values.len()
    }

    /// Indices of the operations whose assertions failed while building.
    pub fn failures(&self) -> &[usize] {
        &self.failures
    }

    fn cell(&mut self, v: BabyBear) -> Cell {
        self.values.push(v);
        (self.values.len() - 1) as Cell
    }

    fn cells<const N: usize>(&mut self, v: [BabyBear; N]) -> [Cell; N] {
        v.map(|x| self.cell(x))
    }

    fn fail_unless(&mut self, ok: bool) {
        if !ok {
            self.failures.push(self.ops.len() - 1);
        }
    }

    fn values_of(&self, cells: &[Cell]) -> Vec<BabyBear> {
        cells.iter().map(|&c| self.value(c)).collect()
    }

    pub fn input(&mut self, v: BabyBear) -> Cell {
        let c = self.cell(v);
        self.ops.push(Op::Input(c));
        c
    }

    pub fn hint(&mut self, v: BabyBear) -> Cell {
        let c = self.cell(v);
        self.ops.push(Op::Hint(c));
        c
    }

    pub fn constant(&mut self, v: BabyBear) -> Cell {
        if let Some(&c) = self.consts.get(&v.as_u32()) {
            return c;
        }
        let c = self.cell(v);
        self.ops.push(Op::Const(c, v));
        self.consts.insert(v.as_u32(), c);
        c
    }

    pub fn zero(&mut self) -> Cell {
        self.constant(BabyBear::ZERO)
    }

    pub fn one(&mut self) -> Cell {
        self.constant(BabyBear::ONE)
    }

    pub fn add(&mut self, a: Cell, b: Cell) -> Cell {
        let out = self.cell(self.value(a) + self.value(b));
        self.ops.push(Op::Add(a, b, out));
        out
    }

    pub fn sub(&mut self, a: Cell, b: Cell) -> Cell {
        let out = self.cell(self.value(a) - self.value(b));
        self.ops.push(Op::Sub(a, b, out));
        out
    }

    pub fn mul(&mut self, a: Cell, b: Cell) -> Cell {
        let out = self.cell(self.value(a) * self.value(b));
        self.ops.push(Op::Mul(a, b, out));
        out
    }

    pub fn mul_const(&mut self, a: Cell, k: BabyBear) -> Cell {
        let k = self.constant(k);
        self.mul(a, k)
    }

    pub fn add_const(&mut self, a: Cell, k: BabyBear) -> Cell {
        let k = self.constant(k);
        self.add(a, k)
    }

    pub fn assert_eq(&mut self, a: Cell, b: Cell) {
        self.ops.push(Op::AssertEq(a, b));
        let ok = self.value(a) == self.value(b);
        self.fail_unless(ok);
    }

    pub fn assert_zero(&mut self, a: Cell) {
        let z = self.zero();
        self.assert_eq(a, z);
    }

    /// The low `n` bits of `a`, least significant first. Fails if `a` (as a
    /// canonical integer) needs more than `n` bits.
    pub fn bits(&mut self, a: Cell, n: usize) -> Vec<Cell> {
        let v = self.value(a).as_u32();
        let bits: Vec<Cell> = (0..n)
            .map(|i| self.hint(BabyBear::new((v >> i) & 1)))
            .collect();
        self.ops.push(Op::Bits(a, bits.clone()));
        self.fail_unless(n >= 32 || v >> n == 0);
        bits
    }

    /// `Poseidon2(state + absorb)` with `absorb` added into the rate.
    pub fn sponge(&mut self, state: [Cell; WIDTH], absorb: &[Cell]) -> [Cell; WIDTH] {
        assert!(
            absorb.len() <= HASH_LEN,
            "a sponge block absorbs at most the rate"
        );
        let out = sponge_block(&state.map(|c| self.value(c)), &self.values_of(absorb));
        let out = self.cells(out);
        self.ops.push(Op::Sponge {
            state: Box::new(state),
            absorb: absorb.to_vec(),
            out: Box::new(out),
        });
        out
    }

    pub fn perm(&mut self, state: [Cell; WIDTH]) -> [Cell; WIDTH] {
        self.sponge(state, &[])
    }

    pub fn duplex(&mut self, state: [Cell; WIDTH], input: &[Cell]) -> [Cell; WIDTH] {
        assert!(input.len() <= HASH_LEN, "a duplex absorbs at most the rate");
        let out = duplex(&state.map(|c| self.value(c)), &self.values_of(input));
        let out = self.cells(out);
        self.ops.push(Op::Duplex {
            state: Box::new(state),
            input: input.to_vec(),
            out: Box::new(out),
        });
        out
    }

    pub fn merkle_step(
        &mut self,
        node: [Cell; HASH_LEN],
        sibling: [Cell; HASH_LEN],
        bit: Cell,
    ) -> [Cell; HASH_LEN] {
        let b = self.value(bit);
        let out = merkle_step(
            &node.map(|c| self.value(c)),
            &sibling.map(|c| self.value(c)),
            b == BabyBear::ONE,
        );
        let out = self.cells(out);
        self.ops.push(Op::MerkleStep {
            node,
            sibling,
            bit,
            out,
        });
        self.fail_unless(b == BabyBear::ZERO || b == BabyBear::ONE);
        out
    }

    /// One [`Op::Dot`] step: returns `(acc_out, pow_out)`.
    pub fn dot(
        &mut self,
        acc: ExtCell,
        pow: ExtCell,
        gamma: ExtCell,
        values: &[Cell],
    ) -> (ExtCell, ExtCell) {
        assert!(
            values.len() <= DOT_WIDTH,
            "a dot step takes at most 8 values"
        );
        let (a, p) = dot(
            self.ext_value(acc),
            self.ext_value(pow),
            self.ext_value(gamma),
            &self.values_of(values),
        );
        let acc_out = ExtCell(self.cells(a.0));
        let pow_out = ExtCell(self.cells(p.0));
        self.ops.push(Op::Dot {
            acc,
            pow,
            gamma,
            values: values.to_vec(),
            acc_out,
            pow_out,
        });
        (acc_out, pow_out)
    }

    /// `1 / a`, as a hint checked by `a * inv = 1`. Fails for zero.
    pub fn inverse(&mut self, a: Cell) -> Cell {
        let inv = self.hint(self.value(a).inverse().unwrap_or(BabyBear::ZERO));
        let prod = self.mul(a, inv);
        let one = self.one();
        self.assert_eq(prod, one);
        inv
    }

    pub fn ext_input(&mut self, v: BabyBear4) -> ExtCell {
        ExtCell(v.0.map(|x| self.input(x)))
    }

    pub fn ext_const(&mut self, v: BabyBear4) -> ExtCell {
        ExtCell(v.0.map(|x| self.constant(x)))
    }

    /// A base value as an extension element.
    pub fn ext_from_base(&mut self, a: Cell) -> ExtCell {
        let z = self.zero();
        ExtCell([a, z, z, z])
    }

    /// `a * b + c`.
    pub fn ext_fma(&mut self, a: ExtCell, b: ExtCell, c: ExtCell) -> ExtCell {
        let v = self.ext_value(a) * self.ext_value(b) + self.ext_value(c);
        let out = ExtCell(self.cells(v.0));
        self.ops.push(Op::ExtFma { a, b, c, out });
        out
    }

    pub fn ext_mul(&mut self, a: ExtCell, b: ExtCell) -> ExtCell {
        let z = self.ext_const(BabyBear4::ZERO);
        self.ext_fma(a, b, z)
    }

    pub fn ext_add(&mut self, a: ExtCell, b: ExtCell) -> ExtCell {
        let one = self.ext_const(BabyBear4::ONE);
        self.ext_fma(a, one, b)
    }

    /// `a - b`.
    pub fn ext_sub(&mut self, a: ExtCell, b: ExtCell) -> ExtCell {
        let minus_one = self.ext_const(-BabyBear4::ONE);
        self.ext_fma(b, minus_one, a)
    }

    /// An extension element times a base element.
    pub fn ext_mul_base(&mut self, a: ExtCell, b: Cell) -> ExtCell {
        let b = self.ext_from_base(b);
        self.ext_mul(a, b)
    }

    pub fn ext_mul_const(&mut self, a: ExtCell, k: BabyBear4) -> ExtCell {
        let k = self.ext_const(k);
        self.ext_mul(a, k)
    }

    /// `1 / a`, as a hint checked by `a * inv = 1`. Fails for zero.
    pub fn ext_inverse(&mut self, a: ExtCell) -> ExtCell {
        let v = self.ext_value(a).inverse().unwrap_or(BabyBear4::ZERO);
        let inv = ExtCell(v.0.map(|x| self.hint(x)));
        let prod = self.ext_mul(a, inv);
        let one = self.ext_const(BabyBear4::ONE);
        self.ext_assert_eq(prod, one);
        inv
    }

    pub fn ext_assert_eq(&mut self, a: ExtCell, b: ExtCell) {
        for i in 0..4 {
            self.assert_eq(a.0[i], b.0[i]);
        }
    }

    /// `a + bit (b - a)`: `a` if `bit` is 0, `b` if it is 1.
    pub fn ext_select(&mut self, bit: Cell, a: ExtCell, b: ExtCell) -> ExtCell {
        let d = self.ext_sub(b, a);
        let bit = self.ext_from_base(bit);
        self.ext_fma(d, bit, a)
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            cells: self.values.len(),
            ..Stats::default()
        };
        for op in &self.ops {
            match op {
                Op::Input(_) => s.inputs += 1,
                Op::Hint(_) => s.hints += 1,
                Op::Const(..) => s.consts += 1,
                Op::Add(..) | Op::Sub(..) | Op::Mul(..) => s.base_ops += 1,
                Op::ExtFma { .. } => s.ext_fmas += 1,
                Op::Sponge { .. } => s.sponges += 1,
                Op::Duplex { .. } => s.duplexes += 1,
                Op::MerkleStep { .. } => s.merkle_steps += 1,
                Op::Dot { .. } => s.dots += 1,
                Op::AssertEq(..) => s.asserts += 1,
                Op::Bits(..) => s.bits += 1,
            }
        }
        s
    }

    /// Re-executes every operation from the inputs and hints, checking each
    /// defines its outputs exactly once, before use, and that every
    /// assertion holds. Returns the first failing operation's index.
    pub fn check(&self) -> Result<(), usize> {
        let mut defined = vec![false; self.values.len()];
        let v = |c: Cell| self.values[c as usize];
        let ev = |e: ExtCell| BabyBear4(e.0.map(v));
        let bit_ok = |b: BabyBear| b == BabyBear::ZERO || b == BabyBear::ONE;
        let vals = |cells: &[Cell]| cells.iter().map(|&c| v(c)).collect::<Vec<_>>();
        for (i, op) in self.ops.iter().enumerate() {
            let used =
                |cells: &[Cell], defined: &[bool]| cells.iter().all(|&c| defined[c as usize]);
            let define = |cells: &[Cell], defined: &mut Vec<bool>| -> bool {
                let fresh = cells.iter().all(|&c| !defined[c as usize]);
                for &c in cells {
                    defined[c as usize] = true;
                }
                fresh
            };
            let ok = match op {
                Op::Input(c) | Op::Hint(c) => define(&[*c], &mut defined),
                Op::Const(c, k) => define(&[*c], &mut defined) && v(*c) == *k,
                Op::Add(a, b, o) => {
                    used(&[*a, *b], &defined)
                        && define(&[*o], &mut defined)
                        && v(*o) == v(*a) + v(*b)
                }
                Op::Sub(a, b, o) => {
                    used(&[*a, *b], &defined)
                        && define(&[*o], &mut defined)
                        && v(*o) == v(*a) - v(*b)
                }
                Op::Mul(a, b, o) => {
                    used(&[*a, *b], &defined)
                        && define(&[*o], &mut defined)
                        && v(*o) == v(*a) * v(*b)
                }
                Op::ExtFma { a, b, c, out } => {
                    used(&a.0, &defined)
                        && used(&b.0, &defined)
                        && used(&c.0, &defined)
                        && define(&out.0, &mut defined)
                        && ev(*out) == ev(*a) * ev(*b) + ev(*c)
                }
                Op::Sponge { state, absorb, out } => {
                    used(&state[..], &defined)
                        && used(absorb, &defined)
                        && absorb.len() <= HASH_LEN
                        && define(&out[..], &mut defined)
                        && out.map(v) == sponge_block(&state.map(v), &vals(absorb))
                }
                Op::Duplex { state, input, out } => {
                    used(&state[..], &defined)
                        && used(input, &defined)
                        && input.len() <= HASH_LEN
                        && define(&out[..], &mut defined)
                        && out.map(v) == duplex(&state.map(v), &vals(input))
                }
                Op::MerkleStep {
                    node,
                    sibling,
                    bit,
                    out,
                } => {
                    used(node, &defined)
                        && used(sibling, &defined)
                        && used(&[*bit], &defined)
                        && bit_ok(v(*bit))
                        && define(out, &mut defined)
                        && out.map(v)
                            == merkle_step(&node.map(v), &sibling.map(v), v(*bit) == BabyBear::ONE)
                }
                Op::Dot {
                    acc,
                    pow,
                    gamma,
                    values,
                    acc_out,
                    pow_out,
                } => {
                    let (a, p) = dot(ev(*acc), ev(*pow), ev(*gamma), &vals(values));
                    used(&acc.0, &defined)
                        && used(&pow.0, &defined)
                        && used(&gamma.0, &defined)
                        && used(values, &defined)
                        && values.len() <= DOT_WIDTH
                        && define(&acc_out.0, &mut defined)
                        && define(&pow_out.0, &mut defined)
                        && ev(*acc_out) == a
                        && ev(*pow_out) == p
                }
                Op::AssertEq(a, b) => used(&[*a, *b], &defined) && v(*a) == v(*b),
                Op::Bits(a, bits) => {
                    let boolean = bits.iter().all(|&b| bit_ok(v(b)));
                    let sum = bits
                        .iter()
                        .rev()
                        .fold(BabyBear::ZERO, |acc, &b| acc + acc + v(b));
                    used(&[*a], &defined) && used(bits, &defined) && boolean && sum == v(*a)
                }
            };
            if !ok {
                return Err(i);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_and_check() {
        let mut c = Circuit::new();
        let a = c.input(BabyBear::new(6));
        let b = c.input(BabyBear::new(7));
        let p = c.mul(a, b);
        let k = c.constant(BabyBear::new(42));
        c.assert_eq(p, k);
        assert!(c.failures().is_empty());
        assert_eq!(c.check(), Ok(()));
        assert_eq!(c.constant(BabyBear::new(42)), k);
    }

    #[test]
    fn failed_assertions_are_recorded_and_rechecked() {
        let mut c = Circuit::new();
        let a = c.input(BabyBear::new(6));
        let b = c.input(BabyBear::new(7));
        let s = c.add(a, b);
        let k = c.constant(BabyBear::new(14));
        c.assert_eq(s, k);
        assert_eq!(c.failures(), &[4]);
        assert_eq!(c.check(), Err(4));
    }

    #[test]
    fn hints_are_constrained() {
        let mut c = Circuit::new();
        let a = c.input(BabyBear::new(9));
        let inv = c.inverse(a);
        assert_eq!(c.value(inv) * BabyBear::new(9), BabyBear::ONE);
        let bits = c.bits(a, 4);
        assert_eq!(
            bits.iter()
                .map(|&b| c.value(b).as_u32())
                .collect::<Vec<_>>(),
            [1, 0, 0, 1]
        );
        assert_eq!(c.check(), Ok(()));
        let _ = c.bits(a, 3);
        assert_eq!(c.failures().len(), 1);
        let z = c.input(BabyBear::ZERO);
        let _ = c.inverse(z);
        assert_eq!(c.failures().len(), 2);
    }

    #[test]
    fn extension_ops() {
        let mut c = Circuit::new();
        let x = BabyBear4::new([1, 2, 3, 4]);
        let y = BabyBear4::new([5, 6, 7, 8]);
        let (a, b) = (c.ext_input(x), c.ext_input(y));
        let p = c.ext_mul(a, b);
        assert_eq!(c.ext_value(p), x * y);
        let s = c.ext_sub(p, a);
        assert_eq!(c.ext_value(s), x * y - x);
        let inv = c.ext_inverse(a);
        assert_eq!(c.ext_value(inv), x.inverse().unwrap());
        let one = c.one();
        let sel = c.ext_select(one, a, b);
        assert_eq!(c.ext_value(sel), y);
        assert_eq!(c.check(), Ok(()));
    }

    #[test]
    fn fused_ops_match_their_definitions() {
        let mut c = Circuit::new();
        let state: [Cell; WIDTH] = core::array::from_fn(|i| c.input(BabyBear::new(i as u32)));
        let absorb: Vec<Cell> = (0..3).map(|i| c.input(BabyBear::new(100 + i))).collect();
        let out = c.sponge(state, &absorb);
        let mut expected = BabyBear::new_array(core::array::from_fn(|i| i as u32));
        for (i, e) in expected.iter_mut().take(3).enumerate() {
            *e += BabyBear::new(100 + i as u32);
        }
        poseidon2::permute(&mut expected);
        assert_eq!(out.map(|o| c.value(o)), expected);

        let node: [Cell; HASH_LEN] = core::array::from_fn(|i| c.input(BabyBear::new(i as u32)));
        let sib: [Cell; HASH_LEN] = core::array::from_fn(|i| c.input(BabyBear::new(50 + i as u32)));
        let bit = c.input(BabyBear::ONE);
        let m = c.merkle_step(node, sib, bit);
        let expected = crate::hash::merge(
            &BabyBear::new_array(core::array::from_fn(|i| 50 + i as u32)),
            &BabyBear::new_array(core::array::from_fn(|i| i as u32)),
        );
        assert_eq!(m.map(|o| c.value(o)), expected);

        let gamma = c.ext_input(BabyBear4::new([2, 0, 0, 0]));
        let acc = c.ext_const(BabyBear4::ZERO);
        let pow = c.ext_const(BabyBear4::ONE);
        let vals: Vec<Cell> = (1..=3).map(|i| c.input(BabyBear::new(i))).collect();
        let (acc, pow) = c.dot(acc, pow, gamma, &vals);
        // 1 + 2*2 + 3*4 = 17, next power 8.
        assert_eq!(c.ext_value(acc), BabyBear4::new([17, 0, 0, 0]));
        assert_eq!(c.ext_value(pow), BabyBear4::new([8, 0, 0, 0]));
        assert_eq!(c.check(), Ok(()));
    }

    #[test]
    fn merkle_step_requires_a_boolean_bit() {
        let mut c = Circuit::new();
        let node = [c.zero(); HASH_LEN];
        let two = c.input(BabyBear::new(2));
        let _ = c.merkle_step(node, node, two);
        assert_eq!(c.failures().len(), 1);
    }

    #[test]
    fn check_catches_tampered_values() {
        let mut c = Circuit::new();
        let a = c.input(BabyBear::new(2));
        let b = c.mul(a, a);
        c.values[b as usize] = BabyBear::new(5);
        assert_eq!(c.check(), Err(1));
    }
}
