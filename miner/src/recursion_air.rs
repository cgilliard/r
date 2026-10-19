//! The recursion AIR: executes a [`Circuit`], so a STARK proves that the
//! circuit's operations all hold. Run on the verifier circuit
//! ([`crate::verifier_circuit`]), it proves an inner proof verifies.
//!
//! Every row has `SLOTS` bus value slots. Permutation rows (sponge, duplex,
//! Merkle step) hold one operation and use the Poseidon2 block; a sponge row
//! can also carry the dot-product step over the values it absorbs. Other
//! rows leave the Poseidon2 block unconstrained (it is gated by the row
//! kind) and pack operations into lanes: 3 extension multiply-adds, 16
//! additions, subtractions or multiplications, 24 assertions, or 48 inputs
//! per row. Dot steps that can't share a sponge row, and bit decompositions,
//! get rows of their own.
//!
//! A permutation's input state comes from the previous row's result
//! (chained: sponge blocks of one hash, a Merkle path, the transcript's
//! duplexes, which are laid out as one contiguous run), from preprocessed
//! constants (a fresh sponge), or from bus slots.
//!
//! Wiring is a LogUp bus over `(cell, value)`: each cell is written exactly
//! once with multiplicity equal to its read count, and every read must find
//! that write, so all uses of a cell see the same value. Row order is free
//! apart from chains. Proof data and hints are written by input rows, except
//! those used exactly once, which are free values in the row that uses them.
//! The outer public values are the cells written on the first row (and, past
//! `SLOTS` of them, the last row).

use std::collections::HashMap;

use crate::air::{Air, Constraints, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;
use crate::circuit::{Cell, Circuit, DOT_WIDTH, Op};
use crate::hash::HASH_LEN;
use crate::lookup::{self, Term};
use crate::matrix::RowMajorMatrix;
use crate::poseidon2::WIDTH;
use crate::poseidon2_air::{self, OUTPUT_OFFSET};

/// Value slots per row.
pub const SLOTS: usize = 48;
/// Bits a [`Op::Bits`] row decomposes, at most.
const MAX_BITS: usize = 31;
/// Lanes per packed row.
const FMA_LANES: usize = SLOTS / 16;
const BASE_LANES: usize = SLOTS / 3;
const ASSERT_LANES: usize = SLOTS / 2;

// Slot layout of permutation and dot rows.
/// Absorbed, overwritten or sibling values; dot values.
const X: usize = 0;
const BIT: usize = HASH_LEN;
/// The result: 16 for a sponge or duplex, 8 for a Merkle step.
const OUT: usize = BIT + 1;
/// State read from the bus, or the dot step's operands.
const EXTRA: usize = OUT + WIDTH;
const D_ACC: usize = EXTRA;
const D_POW: usize = D_ACC + 4;
const D_GAMMA: usize = D_POW + 4;
const D_ACC_OUT: usize = D_GAMMA + 4;
const D_POW_OUT: usize = D_ACC_OUT + 4;
const _: () = assert!(D_POW_OUT + 4 <= SLOTS && EXTRA + WIDTH <= SLOTS);

// Main columns.
const P: usize = poseidon2_air::NUM_COLS;
const BV: usize = P;
/// A permutation's input state (a Merkle step's node in the first 8).
const S: usize = BV + SLOTS;
/// Dot product powers `p_1 .. p_8`, four coordinates each.
const DOTP: usize = S + WIDTH;
pub const NUM_COLS: usize = DOTP + 4 * DOT_WIDTH;

// Preprocessed columns: row kinds, flags, per-slot metadata.
const K_INPUT: usize = 0;
const K_ADD: usize = 1;
const K_SUB: usize = 2;
const K_MUL: usize = 3;
const K_ASSERT: usize = 4;
const K_FMA: usize = 5;
const K_BITS: usize = 6;
const K_SPONGE: usize = 7;
const K_DUPLEX: usize = 8;
const K_MERKLE: usize = 9;
const K_DOT: usize = 10;
const SRC_CONST: usize = 11;
const SRC_CHAIN: usize = 12;
const SRC_BUS: usize = 13;
const DOT_ON: usize = 14;
const SC: usize = 15;
const ID: usize = SC + WIDTH;
const MULT: usize = ID + SLOTS;
const CF: usize = MULT + SLOTS;
const CV: usize = CF + SLOTS;
/// Per rate position: absorbed (sponge) or overwritten (duplex).
const MASK: usize = CV + SLOTS;
const DLEN: usize = MASK + HASH_LEN;
/// One-hot `0..=8`: how many values a dot step takes.
const LENSEL: usize = DLEN + 1;
pub const PRE_WIDTH: usize = LENSEL + DOT_WIDTH + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Empty,
    Read(Cell),
    Write(Cell),
    Const(BabyBear),
    /// A value used only here: no bus term.
    Free(Cell),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    None,
    Const([BabyBear; WIDTH]),
    Chain,
    Bus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    kind: usize,
    slots: Vec<Slot>,
    mask: usize,
    dlen: u32,
    lensel: usize,
    dot_on: bool,
    source: Source,
    /// The permutation's state operand (a Merkle node: 8 cells).
    state: Vec<Cell>,
    /// The result cells, for detecting chains.
    out: Vec<Cell>,
    /// The absorbed values (sponge rows), for fusing a following dot.
    absorbed: Vec<Cell>,
}

impl Row {
    fn new(kind: usize) -> Self {
        Self {
            kind,
            slots: vec![Slot::Empty; SLOTS],
            mask: 0,
            dlen: 0,
            lensel: 0,
            dot_on: false,
            source: Source::None,
            state: Vec::new(),
            out: Vec::new(),
            absorbed: Vec::new(),
        }
    }

    fn is_perm(&self) -> bool {
        matches!(self.kind, K_SPONGE | K_DUPLEX | K_MERKLE)
    }

    fn set(&mut self, at: usize, slots: &[Slot]) {
        self.slots[at..at + slots.len()].copy_from_slice(slots);
    }

    fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| *s == Slot::Empty)
    }
}

/// A circuit laid out as rows: the recursion AIR for that circuit's shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursionAir {
    rows: Vec<Row>,
    reads: Vec<u32>,
    public: Vec<Cell>,
    height: usize,
}

/// Lane operations waiting to be packed into rows.
#[derive(Default)]
struct Lanes {
    fma: Vec<Vec<Slot>>,
    add: Vec<Vec<Slot>>,
    sub: Vec<Vec<Slot>>,
    mul: Vec<Vec<Slot>>,
    assert: Vec<Vec<Slot>>,
}

/// Where a permutation's state comes from, given the previous row of its run.
fn state_source(prev: Option<&Row>, state: &[Cell], consts: &HashMap<Cell, BabyBear>) -> Source {
    if let Some(prev) = prev
        && prev.is_perm()
        && prev.out.len() >= state.len()
        && prev.out[..state.len()] == *state
    {
        return Source::Chain;
    }
    if state.iter().all(|c| consts.contains_key(c)) {
        let mut v = [BabyBear::ZERO; WIDTH];
        for (x, c) in v.iter_mut().zip(state) {
            *x = consts[c];
        }
        return Source::Const(v);
    }
    Source::Bus
}

impl RecursionAir {
    /// Lays out `circuit`. `public` lists the input cells that become the
    /// outer proof's public values, in order (at most `2 * SLOTS`).
    ///
    /// # Panics
    /// If there are too many public values.
    pub fn new(circuit: &Circuit, public: &[Cell]) -> Self {
        Self::with_min_height(circuit, public, 64)
    }

    /// [`new`](Self::new) with exactly `2^log_height` rows.
    ///
    /// # Panics
    /// If the circuit doesn't fit, or as [`new`](Self::new).
    pub fn with_height(circuit: &Circuit, public: &[Cell], log_height: u32) -> Self {
        let air = Self::with_min_height(circuit, public, 1 << log_height);
        assert_eq!(
            air.height,
            1 << log_height,
            "circuit needs {} rows",
            air.used_rows()
        );
        air
    }

    fn with_min_height(circuit: &Circuit, public: &[Cell], min_height: usize) -> Self {
        assert!(public.len() <= 2 * SLOTS, "too many public values");
        let consts: HashMap<Cell, BabyBear> = circuit
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::Const(c, k) => Some((*c, *k)),
                _ => None,
            })
            .collect();
        let read = |c: Cell| consts.get(&c).map_or(Slot::Read(c), |&k| Slot::Const(k));
        let reads = |cells: &[Cell]| cells.iter().map(|&c| read(c)).collect::<Vec<_>>();
        let writes = |cells: &[Cell]| cells.iter().map(|&c| Slot::Write(c)).collect::<Vec<_>>();
        let zero = Slot::Const(BabyBear::ZERO);
        let padded = |mut v: Vec<Slot>, n: usize| {
            v.resize(n, zero);
            v
        };
        let is_public: HashMap<Cell, usize> =
            public.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        let mut main: Vec<Row> = Vec::new();
        let mut duplexes: Vec<Row> = Vec::new();
        let mut lanes = Lanes::default();
        let mut inputs = Vec::new();
        for op in circuit.ops() {
            match op {
                Op::Input(c) | Op::Hint(c) => {
                    if !is_public.contains_key(c) {
                        inputs.push(*c);
                    }
                }
                Op::Const(..) => {}
                Op::Add(a, b, o) | Op::Sub(a, b, o) | Op::Mul(a, b, o) => {
                    let lane = [reads(&[*a, *b]), writes(&[*o])].concat();
                    match op {
                        Op::Add(..) => lanes.add.push(lane),
                        Op::Sub(..) => lanes.sub.push(lane),
                        _ => lanes.mul.push(lane),
                    }
                }
                Op::AssertEq(a, b) => lanes.assert.push(reads(&[*a, *b])),
                Op::ExtFma { a, b, c, out } => lanes
                    .fma
                    .push([reads(&a.0), reads(&b.0), reads(&c.0), writes(&out.0)].concat()),
                Op::Bits(a, bits) => {
                    // The bits are hints, written by input rows.
                    assert!(bits.len() <= MAX_BITS);
                    let mut r = Row::new(K_BITS);
                    r.set(
                        0,
                        &padded([reads(&[*a]), reads(bits)].concat(), 1 + MAX_BITS),
                    );
                    main.push(r);
                }
                Op::Sponge { state, absorb, out }
                | Op::Duplex {
                    state,
                    input: absorb,
                    out,
                } => {
                    let duplex = matches!(op, Op::Duplex { .. });
                    let run = if duplex { &mut duplexes } else { &mut main };
                    let mut r = Row::new(if duplex { K_DUPLEX } else { K_SPONGE });
                    r.set(X, &padded(reads(absorb), HASH_LEN));
                    r.set(OUT, &writes(&out[..]));
                    r.mask = absorb.len();
                    r.dlen = absorb.len() as u32;
                    r.source = state_source(run.last(), &state[..], &consts);
                    if r.source == Source::Bus {
                        r.set(EXTRA, &reads(&state[..]));
                    }
                    r.state = state.to_vec();
                    r.out = out.to_vec();
                    r.absorbed = absorb.clone();
                    run.push(r);
                }
                Op::MerkleStep {
                    node,
                    sibling,
                    bit,
                    out,
                } => {
                    let mut r = Row::new(K_MERKLE);
                    r.set(X, &reads(sibling));
                    r.set(BIT, &reads(&[*bit]));
                    r.set(OUT, &writes(out));
                    r.source = state_source(main.last(), node, &consts);
                    if r.source == Source::Bus {
                        r.set(EXTRA, &reads(node));
                    }
                    r.state = node.to_vec();
                    r.out = out.to_vec();
                    main.push(r);
                }
                Op::Dot {
                    acc,
                    pow,
                    gamma,
                    values,
                    acc_out,
                    pow_out,
                } => {
                    let fields = [
                        reads(&acc.0),
                        reads(&pow.0),
                        reads(&gamma.0),
                        writes(&acc_out.0),
                        writes(&pow_out.0),
                    ]
                    .concat();
                    let fuse = main.last().is_some_and(|r| {
                        r.kind == K_SPONGE
                            && !r.dot_on
                            && r.source != Source::Bus
                            && r.absorbed == *values
                    });
                    if !fuse {
                        let mut r = Row::new(K_DOT);
                        r.set(X, &padded(reads(values), DOT_WIDTH));
                        main.push(r);
                    }
                    let r = main.last_mut().expect("a row");
                    r.set(EXTRA, &fields);
                    r.dot_on = true;
                    r.lensel = values.len();
                }
            }
        }

        let mut rows = Vec::new();
        let mut first_row = Row::new(K_INPUT);
        let (first, last) = public.split_at(public.len().min(SLOTS));
        first_row.set(0, &writes(first));
        rows.push(first_row);
        rows.extend(main);
        rows.extend(duplexes);
        let pack = |kind: usize, per_row: usize, items: Vec<Vec<Slot>>, rows: &mut Vec<Row>| {
            for chunk in items.chunks(per_row) {
                let mut r = Row::new(kind);
                r.set(0, &chunk.concat());
                rows.push(r);
            }
        };
        pack(K_FMA, FMA_LANES, lanes.fma, &mut rows);
        pack(K_ADD, BASE_LANES, lanes.add, &mut rows);
        pack(K_SUB, BASE_LANES, lanes.sub, &mut rows);
        pack(K_MUL, BASE_LANES, lanes.mul, &mut rows);
        pack(K_ASSERT, ASSERT_LANES, lanes.assert, &mut rows);

        // Inputs and hints read exactly once become free values there.
        let mut read_counts = vec![0u32; circuit.num_cells()];
        for row in &rows {
            for slot in &row.slots {
                if let Slot::Read(c) = slot {
                    read_counts[*c as usize] += 1;
                }
            }
        }
        let is_input: Vec<bool> = {
            let mut v = vec![false; circuit.num_cells()];
            for &c in &inputs {
                v[c as usize] = true;
            }
            v
        };
        for row in &mut rows {
            for slot in &mut row.slots {
                if let Slot::Read(c) = *slot
                    && is_input[c as usize]
                    && read_counts[c as usize] == 1
                {
                    *slot = Slot::Free(c);
                    read_counts[c as usize] = 0;
                }
            }
        }
        let bus_inputs: Vec<Vec<Slot>> = inputs
            .iter()
            .filter(|&&c| read_counts[c as usize] > 0)
            .map(|&c| vec![Slot::Write(c)])
            .collect();
        pack(K_INPUT, SLOTS, bus_inputs, &mut rows);

        let mut last_row = Row::new(K_INPUT);
        last_row.set(0, &writes(last));
        let height = (rows.len() + 1).next_power_of_two().max(min_height);
        rows.resize(height - 1, Row::new(K_INPUT));
        rows.push(last_row);
        Self {
            rows,
            reads: read_counts,
            public: public.to_vec(),
            height,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Rows holding operations (the rest is padding).
    pub fn used_rows(&self) -> usize {
        self.rows.iter().filter(|r| !r.is_empty()).count()
    }

    /// The outer public values: the public cells' values in `circuit`.
    pub fn public_values(&self, circuit: &Circuit) -> Vec<BabyBear> {
        self.public.iter().map(|&c| circuit.value(c)).collect()
    }

    /// The trace for `circuit`, which must have this layout's shape.
    pub fn trace(&self, circuit: &Circuit) -> RowMajorMatrix {
        let mut m = RowMajorMatrix::zeros(self.height, NUM_COLS);
        for (i, row) in self.rows.iter().enumerate() {
            for (k, slot) in row.slots.iter().enumerate() {
                m.row_mut(i)[BV + k] = match *slot {
                    Slot::Empty => BabyBear::ZERO,
                    Slot::Read(c) | Slot::Write(c) | Slot::Free(c) => circuit.value(c),
                    Slot::Const(v) => v,
                };
            }
            let state: [BabyBear; WIDTH] = match row.source {
                Source::None => [BabyBear::ZERO; WIDTH],
                Source::Const(v) => v,
                Source::Chain => core::array::from_fn(|j| m.row(i - 1)[BV + OUT + j]),
                Source::Bus => core::array::from_fn(|j| m.row(i)[BV + EXTRA + j]),
            };
            let r = m.row_mut(i);
            r[S..S + WIDTH].copy_from_slice(&state);
            let bv: Vec<BabyBear> = r[BV..BV + SLOTS].to_vec();
            if row.is_perm() {
                let x = |j: usize| bv[X + j];
                let input: [BabyBear; WIDTH] = match row.kind {
                    K_SPONGE => core::array::from_fn(|j| {
                        if j < row.mask {
                            state[j] + x(j)
                        } else {
                            state[j]
                        }
                    }),
                    K_DUPLEX => core::array::from_fn(|j| {
                        if j < row.mask {
                            x(j)
                        } else if j == WIDTH - 1 {
                            state[j] + BabyBear::new(row.dlen)
                        } else {
                            state[j]
                        }
                    }),
                    _ => {
                        let bit = bv[BIT] == BabyBear::ONE;
                        core::array::from_fn(|j| {
                            let (node, sib) = if j < HASH_LEN {
                                (state[j], x(j))
                            } else {
                                (x(j - HASH_LEN), state[j - HASH_LEN])
                            };
                            if bit { sib } else { node }
                        })
                    }
                };
                poseidon2_air::generate(input, r);
            }
            if row.dot_on {
                let gamma = BabyBear4(core::array::from_fn(|b| bv[D_GAMMA + b]));
                let mut p = BabyBear4(core::array::from_fn(|b| bv[D_POW + b]));
                for k in 0..DOT_WIDTH {
                    p *= gamma;
                    r[DOTP + 4 * k..DOTP + 4 * k + 4].copy_from_slice(&p.0);
                }
            }
        }
        m
    }
}

fn ext<V: Copy>(row: &[V], at: usize) -> [V; 4] {
    core::array::from_fn(|b| row[at + b])
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

impl RecursionAir {
    fn terms<V: Var>(local: &[V], pre: &[V]) -> Vec<Term<V>> {
        (0..SLOTS)
            .map(|k| Term::new(pre[MULT + k], vec![pre[ID + k], local[BV + k]]))
            .collect()
    }
}

impl Air for RecursionAir {
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
        let mut m = RowMajorMatrix::zeros(self.height, PRE_WIDTH);
        for (i, row) in self.rows.iter().enumerate() {
            let r = m.row_mut(i);
            r[row.kind] = BabyBear::ONE;
            match row.source {
                Source::None => {}
                Source::Const(v) => {
                    r[SRC_CONST] = BabyBear::ONE;
                    r[SC..SC + WIDTH].copy_from_slice(&v);
                }
                Source::Chain => r[SRC_CHAIN] = BabyBear::ONE,
                Source::Bus => r[SRC_BUS] = BabyBear::ONE,
            }
            r[DOT_ON] = BabyBear::from(row.dot_on);
            for (k, slot) in row.slots.iter().enumerate() {
                match *slot {
                    Slot::Empty => r[CF + k] = BabyBear::ONE,
                    Slot::Const(v) => {
                        r[CF + k] = BabyBear::ONE;
                        r[CV + k] = v;
                    }
                    Slot::Free(_) => {}
                    Slot::Read(c) => {
                        r[ID + k] = BabyBear::new(c);
                        r[MULT + k] = -BabyBear::ONE;
                    }
                    Slot::Write(c) => {
                        r[ID + k] = BabyBear::new(c);
                        r[MULT + k] = BabyBear::new(self.reads[c as usize]);
                    }
                }
            }
            for j in 0..row.mask.min(HASH_LEN) {
                r[MASK + j] = BabyBear::ONE;
            }
            r[DLEN] = BabyBear::new(row.dlen);
            r[LENSEL + row.lensel] = BabyBear::ONE;
        }
        Some(m)
    }

    fn num_challenges(&self) -> usize {
        lookup::CHALLENGES
    }

    fn aux_width(&self) -> usize {
        lookup::aux_width(SLOTS)
    }

    fn aux_trace(
        &self,
        main: &RowMajorMatrix,
        pre: Option<&RowMajorMatrix>,
        ch: &[BabyBear4],
    ) -> Vec<BabyBear4> {
        let pre = pre.expect("preprocessed columns");
        let rows: Vec<_> = (0..main.height())
            .map(|i| Self::terms(main.row(i), pre.row(i)))
            .collect();
        lookup::trace(&rows, ch[0], ch[1]).concat()
    }

    fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
        let (l, n, p, pn) = (
            c.local(),
            c.next(),
            c.preprocessed_local(),
            c.preprocessed_next(),
        );
        let public = c.public();
        let one = V::from(BabyBear::ONE);
        let zero = V::from(BabyBear::ZERO);
        let bv = |k: usize| l[BV + k];
        let out = |j: usize| l[OUTPUT_OFFSET + j];
        let st = |j: usize| l[S + j];

        let perm = p[K_SPONGE] + p[K_DUPLEX] + p[K_MERKLE];
        poseidon2_air::eval_gated(c, l, Some(perm));

        // Inlined constants (empty slots are the constant 0).
        for k in 0..SLOTS {
            c.assert_zero(p[CF + k] * (bv(k) - p[CV + k]));
        }

        // Lanes.
        for j in 0..BASE_LANES {
            let (a, b, o) = (bv(3 * j), bv(3 * j + 1), bv(3 * j + 2));
            c.assert_zero(p[K_ADD] * (o - a - b));
            c.assert_zero(p[K_SUB] * (o - a + b));
            c.assert_zero(p[K_MUL] * (o - a * b));
        }
        for j in 0..ASSERT_LANES {
            c.assert_zero(p[K_ASSERT] * (bv(2 * j) - bv(2 * j + 1)));
        }
        for j in 0..FMA_LANES {
            let at = BV + 16 * j;
            let fma = ext_mul(ext(l, at), ext(l, at + 4));
            for (b, &f) in fma.iter().enumerate() {
                c.assert_zero(p[K_FMA] * (l[at + 12 + b] - f - l[at + 8 + b]));
            }
        }

        // Bit decomposition.
        let mut sum = zero;
        for i in 0..MAX_BITS {
            let b = bv(1 + i);
            c.assert_zero(p[K_BITS] * b * (b - one));
            sum = sum + b * BabyBear::new(1 << i);
        }
        c.assert_zero(p[K_BITS] * (bv(0) - sum));

        // A permutation's state: constants, the bus, or the previous result.
        for j in 0..WIDTH {
            c.assert_zero(p[SRC_CONST] * (st(j) - p[SC + j]));
            c.assert_zero(p[SRC_BUS] * (st(j) - bv(EXTRA + j)));
            c.assert_zero(pn[SRC_CHAIN] * (n[S + j] - bv(OUT + j)));
        }

        // Sponge and duplex rows.
        for j in 0..WIDTH {
            let s = st(j);
            let (x, m) = if j < HASH_LEN {
                (bv(X + j), p[MASK + j])
            } else {
                (zero, zero)
            };
            c.assert_zero(p[K_SPONGE] * (l[j] - s - m * x));
            let dup = if j == WIDTH - 1 {
                s + p[DLEN]
            } else {
                s + m * (x - s)
            };
            c.assert_zero(p[K_DUPLEX] * (l[j] - dup));
            c.assert_zero((p[K_SPONGE] + p[K_DUPLEX]) * (bv(OUT + j) - out(j)));
        }
        // Merkle rows: node from the state, sibling from the slots.
        let bit = bv(BIT);
        c.assert_zero(p[K_MERKLE] * bit * (bit - one));
        for j in 0..HASH_LEN {
            let (node, sib) = (st(j), bv(X + j));
            let left = node + bit * (sib - node);
            let right = sib + bit * (node - sib);
            c.assert_zero(p[K_MERKLE] * (l[j] - left));
            c.assert_zero(p[K_MERKLE] * (l[HASH_LEN + j] - right));
            c.assert_zero(p[K_MERKLE] * (bv(OUT + j) - out(j) - left));
        }

        // Dot steps: p_0 = pow, p_(k+1) = p_k gamma, over the X slots.
        let dot = p[DOT_ON];
        let gamma = ext(l, BV + D_GAMMA);
        let mut powers = vec![ext(l, BV + D_POW)];
        for k in 0..DOT_WIDTH {
            let next = ext(l, DOTP + 4 * k);
            let expected = ext_mul(powers[k], gamma);
            for b in 0..4 {
                c.assert_zero(dot * (next[b] - expected[b]));
            }
            powers.push(next);
        }
        let mut acc = ext(l, BV + D_ACC);
        for (k, pw) in powers.iter().take(DOT_WIDTH).enumerate() {
            let v = bv(X + k);
            acc = core::array::from_fn(|b| acc[b] + pw[b] * v);
        }
        for b in 0..4 {
            c.assert_zero(dot * (bv(D_ACC_OUT + b) - acc[b]));
            let pow_out = (0..=DOT_WIDTH).fold(zero, |s, k| s + p[LENSEL + k] * powers[k][b]);
            c.assert_zero(dot * (bv(D_POW_OUT + b) - pow_out));
        }

        // Public values.
        for (k, &pv) in public.iter().enumerate() {
            if k < SLOTS {
                c.assert_zero_first_row(bv(k) - pv);
            } else {
                c.assert_zero_last_row(bv(k - SLOTS) - pv);
            }
        }

        let terms = Self::terms(l, p);
        lookup::eval(c, &terms, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::check_constraints;
    use crate::circuit::ExtCell;

    /// A circuit using every operation kind.
    fn sample_circuit() -> (Circuit, Vec<Cell>) {
        let mut c = Circuit::new();
        let x = c.input(BabyBear::new(7));
        let y = c.input(BabyBear::new(9));
        let s = c.add(x, y);
        let d = c.sub(s, x);
        let p = c.mul(d, x);
        let k = c.constant(BabyBear::new(63));
        c.assert_eq(p, k);
        let bits = c.bits(p, 8);
        let state: [Cell; WIDTH] = core::array::from_fn(|i| if i < 8 { bits[i] } else { s });
        let st = c.sponge(state, &[x, y, p]);
        let st = c.duplex(st, &[x, y]);
        let node: [Cell; HASH_LEN] = core::array::from_fn(|i| st[i]);
        let sib: [Cell; HASH_LEN] = core::array::from_fn(|i| st[HASH_LEN + i]);
        let m = c.merkle_step(node, sib, bits[0]);
        let a = c.ext_input(BabyBear4::new([1, 2, 3, 4]));
        let b = ExtCell([m[0], m[1], m[2], m[3]]);
        let f = c.ext_fma(a, b, a);
        let gamma = c.ext_input(BabyBear4::new([5, 0, 6, 0]));
        let one = c.ext_const(BabyBear4::ONE);
        let (acc, pow) = c.dot(f, one, gamma, &m[..5]);
        let _ = c.ext_mul(acc, pow);
        assert!(c.failures().is_empty());
        assert_eq!(c.check(), Ok(()));
        (c, vec![x, y])
    }

    #[test]
    fn executes_a_valid_circuit() {
        let (c, public) = sample_circuit();
        let air = RecursionAir::new(&c, &public);
        let t = air.trace(&c);
        assert_eq!(
            air.public_values(&c),
            vec![BabyBear::new(7), BabyBear::new(9)]
        );
        assert_eq!(check_constraints(&air, &t, &air.public_values(&c)), Ok(()));
    }

    #[test]
    fn rejects_a_failed_assertion() {
        let mut c = Circuit::new();
        let x = c.input(BabyBear::new(7));
        let k = c.constant(BabyBear::new(8));
        c.assert_eq(x, k);
        let air = RecursionAir::new(&c, &[]);
        assert!(check_constraints(&air, &air.trace(&c), &[]).is_err());
    }

    #[test]
    fn rejects_inconsistent_values() {
        let (c, public) = sample_circuit();
        let air = RecursionAir::new(&c, &public);
        let pv = air.public_values(&c);
        let t = air.trace(&c);
        // Change one slot on each row kind in turn: the bus or the row's own
        // constraints break.
        for i in 1..air.rows.len() - 1 {
            if air.rows[i].slots.iter().all(|s| matches!(s, Slot::Empty)) {
                continue;
            }
            let k = air.rows[i]
                .slots
                .iter()
                .position(|s| matches!(s, Slot::Read(_) | Slot::Write(_)))
                .unwrap();
            let mut bad = t.clone();
            bad.row_mut(i)[BV + k] += BabyBear::ONE;
            assert!(check_constraints(&air, &bad, &pv).is_err(), "row {i}");
        }
        // Wrong public values.
        let mut wrong = pv.clone();
        wrong[1] += BabyBear::ONE;
        assert!(check_constraints(&air, &t, &wrong).is_err());
    }

    mod verifying {
        use super::*;
        use crate::fri::FriParams;
        use crate::rng::Rng;
        use crate::stark::{self, StarkProof};
        use crate::symbolic::compile;
        use crate::verifier_circuit::{self, Inner, ProofCells};

        fn inner_params() -> FriParams {
            FriParams {
                num_queries: 8,
                grinding_bits: 4,
                ..FriParams::default()
            }
        }

        struct Fibonacci;

        impl Air for Fibonacci {
            fn width(&self) -> usize {
                2
            }
            fn max_degree(&self) -> usize {
                2
            }
            fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
                let (a, b) = (c.local()[0], c.local()[1]);
                let (a2, b2) = (c.next()[0], c.next()[1]);
                let pv = c.public();
                c.assert_zero_first_row(a - pv[0]);
                c.assert_zero_first_row(b - pv[1]);
                c.assert_zero_transition(a2 - b);
                c.assert_zero_transition(b2 - (a + b));
                c.assert_zero_last_row(b - pv[2]);
            }
        }

        fn fibonacci(n: usize, b0: u32) -> (RowMajorMatrix, Vec<BabyBear>) {
            let (mut a, mut b) = (BabyBear::ZERO, BabyBear::new(b0));
            let mut values = Vec::with_capacity(2 * n);
            for _ in 0..n {
                values.extend([a, b]);
                (a, b) = (b, a + b);
            }
            let last = values[2 * n - 1];
            (
                RowMajorMatrix::new(values, 2),
                vec![BabyBear::ZERO, BabyBear::new(b0), last],
            )
        }

        fn inner_proof(b0: u32, seed: u64) -> (StarkProof, Vec<BabyBear>) {
            let (trace, public) = fibonacci(64, b0);
            let (pk, _) = stark::setup(&inner_params(), &Fibonacci);
            let proof = stark::prove_with_rng(
                &inner_params(),
                &pk,
                &Fibonacci,
                trace,
                &public,
                &mut Rng::from_seed(seed),
            );
            (proof, public)
        }

        /// The verifier circuit for `proof`, with the inner public values as
        /// the outer public values.
        fn verifier(proof: &StarkProof, public: &[BabyBear]) -> (Circuit, Vec<Cell>) {
            let (_, vk) = stark::setup(&inner_params(), &Fibonacci);
            let program = compile(&Fibonacci, public.len());
            let mut c = Circuit::new();
            let public_cells: Vec<Cell> = public.iter().map(|&x| c.input(x)).collect();
            let cells = ProofCells::load(&mut c, proof);
            let inner = Inner {
                params: &inner_params(),
                vk: &vk,
                air: &Fibonacci,
                program: &program,
                log_n: proof.log_n,
            };
            verifier_circuit::verify(&mut c, &inner, &public_cells, &cells);
            (c, public_cells)
        }

        #[test]
        fn executes_the_verifier_of_a_valid_proof() {
            let (proof, public) = inner_proof(1, 1);
            let (c, cells) = verifier(&proof, &public);
            assert!(c.failures().is_empty());
            let air = RecursionAir::new(&c, &cells);
            assert_eq!(air.public_values(&c), public);
            let t = air.trace(&c);
            assert_eq!(check_constraints(&air, &t, &public), Ok(()));
            eprintln!("verifier circuit: {:?}, rows {}", c.stats(), air.height());
        }

        #[test]
        fn layout_depends_only_on_the_shape() {
            // Different statements and masks, same verifying key.
            let (p1, pub1) = inner_proof(1, 1);
            let (p2, pub2) = inner_proof(5, 2);
            let (c1, cells1) = verifier(&p1, &pub1);
            let (c2, cells2) = verifier(&p2, &pub2);
            let (a1, a2) = (
                RecursionAir::new(&c1, &cells1),
                RecursionAir::new(&c2, &cells2),
            );
            assert_eq!(a1.preprocessed_trace(), a2.preprocessed_trace());
        }

        #[test]
        fn a_placeholder_proof_has_the_same_layout() {
            let (proof, public) = inner_proof(1, 1);
            let placeholder = StarkProof::placeholder(&inner_params(), &Fibonacci, proof.log_n);
            let (c1, cells1) = verifier(&proof, &public);
            let (c2, cells2) = verifier(&placeholder, &public);
            assert!(!c2.failures().is_empty());
            assert!(RecursionAir::new(&c1, &cells1) == RecursionAir::new(&c2, &cells2));
        }

        #[test]
        fn rejects_the_verifier_of_a_bad_proof() {
            let (proof, public) = inner_proof(1, 1);
            for tamper in 0..3 {
                let mut bad = proof.clone();
                match tamper {
                    0 => bad.trace_at_zeta[0] += BabyBear4::ONE,
                    1 => bad.openings[2].trace_row[1] += BabyBear::ONE,
                    _ => bad.fri.final_poly[0] += BabyBear4::ONE,
                }
                let (c, cells) = verifier(&bad, &public);
                assert!(!c.failures().is_empty());
                let air = RecursionAir::new(&c, &cells);
                assert!(
                    check_constraints(&air, &air.trace(&c), &public).is_err(),
                    "tamper {tamper}"
                );
            }
            // A valid proof against claimed public values it doesn't prove.
            let mut wrong = public.clone();
            wrong[2] += BabyBear::ONE;
            let (c, cells) = verifier(&proof, &wrong);
            let air = RecursionAir::new(&c, &cells);
            assert!(check_constraints(&air, &air.trace(&c), &wrong).is_err());
        }

        /// A full recursive proof. Slow in debug builds: run with
        /// `cargo test --release -- --ignored`.
        #[test]
        #[ignore]
        fn recursive_proof_verifies() {
            let (proof, public) = inner_proof(1, 1);
            let (c, cells) = verifier(&proof, &public);
            let air = RecursionAir::new(&c, &cells);
            let outer = FriParams::default();
            let (pk, vk) = stark::setup(&outer, &air);
            let start = std::time::Instant::now();
            let outer_proof = stark::prove(&outer, &pk, &air, air.trace(&c), &public);
            eprintln!(
                "recursive proof: {:.2} s, {} rows",
                start.elapsed().as_secs_f64(),
                air.height()
            );
            assert_eq!(
                stark::verify(&outer, &vk, &air, &public, &outer_proof),
                Ok(())
            );
            let mut wrong = public.clone();
            wrong[2] += BabyBear::ONE;
            assert!(stark::verify(&outer, &vk, &air, &wrong, &outer_proof).is_err());
        }
    }
}
