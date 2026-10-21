//! Symbolic constraint evaluation: records an [`Air`]'s `eval` as a
//! straight-line arithmetic [`Program`] over extension-field values.
//!
//! The recursive verifier must evaluate an inner AIR's constraints at the
//! out-of-domain point inside a circuit. Running `eval` with [`Sym`] values
//! instead of numbers yields the exact sequence of operations to lay out as
//! circuit rows. Every value at that point is in the extension field, so one
//! node type suffices.
//!
//! The builder folds constants, orders commutative operands and deduplicates
//! identical nodes, so repeated subexpressions are computed once.

use core::cell::RefCell;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::collections::HashMap;

use crate::air::{Air, Constraints, Ext, Rows, Selectors, Var};
use crate::babybear::BabyBear;
use crate::babybear4::BabyBear4;

/// A program input: a value the verifier knows at the out-of-domain point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    Local(u32),
    Next(u32),
    PreprocessedLocal(u32),
    PreprocessedNext(u32),
    AuxLocal(u32),
    AuxNext(u32),
    Challenge(u32),
    Public(u32),
    IsFirstRow,
    IsLastRow,
    IsTransition,
    Alpha,
}

/// One operation. Operands are earlier node indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    Input(Input),
    Const(BabyBear4),
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
}

/// A symbolic value: an index into the active builder's nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sym(u32);

#[derive(Default)]
struct Builder {
    nodes: Vec<Node>,
    index: HashMap<Node, u32>,
}

thread_local! {
    static BUILDER: RefCell<Option<Builder>> = const { RefCell::new(None) };
}

impl Builder {
    fn constant(&self, s: u32) -> Option<BabyBear4> {
        match self.nodes[s as usize] {
            Node::Const(c) => Some(c),
            _ => None,
        }
    }

    fn intern(&mut self, node: Node) -> u32 {
        if let Some(&i) = self.index.get(&node) {
            return i;
        }
        let i = self.nodes.len() as u32;
        self.nodes.push(node);
        self.index.insert(node, i);
        i
    }

    /// Adds `node`, folding constants and trivial identities.
    fn push(&mut self, node: Node) -> u32 {
        let zero = BabyBear4::ZERO;
        let one = BabyBear4::ONE;
        let folded = match node {
            Node::Add(a, b) => match (self.constant(a), self.constant(b)) {
                (Some(x), Some(y)) => Node::Const(x + y),
                (Some(x), _) if x == zero => return b,
                (_, Some(y)) if y == zero => return a,
                _ => Node::Add(a.min(b), a.max(b)),
            },
            Node::Sub(a, b) => match (self.constant(a), self.constant(b)) {
                (Some(x), Some(y)) => Node::Const(x - y),
                (_, Some(y)) if y == zero => return a,
                _ if a == b => Node::Const(zero),
                _ => node,
            },
            Node::Mul(a, b) => match (self.constant(a), self.constant(b)) {
                (Some(x), Some(y)) => Node::Const(x * y),
                (Some(x), _) | (_, Some(x)) if x == zero => Node::Const(zero),
                (Some(x), _) if x == one => return b,
                (_, Some(y)) if y == one => return a,
                _ => Node::Mul(a.min(b), a.max(b)),
            },
            Node::Neg(a) => match self.constant(a) {
                Some(x) => Node::Const(-x),
                None => node,
            },
            other => other,
        };
        self.intern(folded)
    }
}

fn push(node: Node) -> Sym {
    BUILDER.with(|b| {
        let mut b = b.borrow_mut();
        let b = b
            .as_mut()
            .expect("symbolic values exist only inside `compile`");
        Sym(b.push(node))
    })
}

impl Sym {
    fn input(i: Input) -> Self {
        push(Node::Input(i))
    }
}

impl From<BabyBear4> for Sym {
    fn from(c: BabyBear4) -> Self {
        push(Node::Const(c))
    }
}

impl From<BabyBear> for Sym {
    fn from(c: BabyBear) -> Self {
        Self::from(BabyBear4::from(c))
    }
}

impl Add for Sym {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        push(Node::Add(self.0, rhs.0))
    }
}

impl Sub for Sym {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        push(Node::Sub(self.0, rhs.0))
    }
}

impl Mul for Sym {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        push(Node::Mul(self.0, rhs.0))
    }
}

impl Neg for Sym {
    type Output = Self;
    fn neg(self) -> Self {
        push(Node::Neg(self.0))
    }
}

impl Add<BabyBear> for Sym {
    type Output = Self;
    fn add(self, rhs: BabyBear) -> Self {
        self + Sym::from(rhs)
    }
}

impl Sub<BabyBear> for Sym {
    type Output = Self;
    fn sub(self, rhs: BabyBear) -> Self {
        self - Sym::from(rhs)
    }
}

impl Mul<BabyBear> for Sym {
    type Output = Self;
    fn mul(self, rhs: BabyBear) -> Self {
        self * Sym::from(rhs)
    }
}

impl AddAssign for Sym {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Sym {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Sym {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Var for Sym {
    type Ext = Sym;
}

impl Ext<Sym> for Sym {}

/// A compiled constraint evaluation: `nodes` in dependency order, the last
/// being the combined constraint value `sum alpha^k c_k`.
#[derive(Clone, Debug)]
pub struct Program {
    pub nodes: Vec<Node>,
}

/// Operation counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub inputs: usize,
    pub consts: usize,
    pub adds: usize,
    pub muls: usize,
}

/// Values for a program's inputs, all at the out-of-domain point.
pub struct Inputs<'a> {
    pub local: &'a [BabyBear4],
    pub next: &'a [BabyBear4],
    pub preprocessed_local: &'a [BabyBear4],
    pub preprocessed_next: &'a [BabyBear4],
    pub aux_local: &'a [BabyBear4],
    pub aux_next: &'a [BabyBear4],
    pub challenges: &'a [BabyBear4],
    pub public: &'a [BabyBear4],
    pub selectors: Selectors<BabyBear4>,
    pub alpha: BabyBear4,
}

impl Inputs<'_> {
    fn get(&self, i: Input) -> BabyBear4 {
        match i {
            Input::Local(j) => self.local[j as usize],
            Input::Next(j) => self.next[j as usize],
            Input::PreprocessedLocal(j) => self.preprocessed_local[j as usize],
            Input::PreprocessedNext(j) => self.preprocessed_next[j as usize],
            Input::AuxLocal(j) => self.aux_local[j as usize],
            Input::AuxNext(j) => self.aux_next[j as usize],
            Input::Challenge(j) => self.challenges[j as usize],
            Input::Public(j) => self.public[j as usize],
            Input::IsFirstRow => self.selectors.is_first_row,
            Input::IsLastRow => self.selectors.is_last_row,
            Input::IsTransition => self.selectors.is_transition,
            Input::Alpha => self.alpha,
        }
    }
}

impl Program {
    /// Evaluates the program: the combined constraint value.
    pub fn eval(&self, inputs: &Inputs<'_>) -> BabyBear4 {
        let mut values: Vec<BabyBear4> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let v = |i: u32| values[i as usize];
            let x = match *node {
                Node::Input(i) => inputs.get(i),
                Node::Const(c) => c,
                Node::Add(a, b) => v(a) + v(b),
                Node::Sub(a, b) => v(a) - v(b),
                Node::Mul(a, b) => v(a) * v(b),
                Node::Neg(a) => -v(a),
            };
            values.push(x);
        }
        *values.last().expect("a program has an output")
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats::default();
        for node in &self.nodes {
            match node {
                Node::Input(_) => s.inputs += 1,
                Node::Const(_) => s.consts += 1,
                Node::Add(..) | Node::Sub(..) | Node::Neg(_) => s.adds += 1,
                Node::Mul(..) => s.muls += 1,
            }
        }
        s
    }

    /// Keeps only the nodes the output depends on, renumbered in order.
    fn pruned(nodes: Vec<Node>, output: u32) -> Self {
        let mut live = vec![false; nodes.len()];
        live[output as usize] = true;
        for i in (0..=output as usize).rev() {
            if !live[i] {
                continue;
            }
            match nodes[i] {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    live[a as usize] = true;
                    live[b as usize] = true;
                }
                Node::Neg(a) => live[a as usize] = true,
                Node::Input(_) | Node::Const(_) => {}
            }
        }
        let mut remap = vec![u32::MAX; nodes.len()];
        let mut out = Vec::new();
        for (i, node) in nodes.into_iter().enumerate().take(output as usize + 1) {
            if !live[i] {
                continue;
            }
            let r = |x: u32| remap[x as usize];
            let node = match node {
                Node::Add(a, b) => Node::Add(r(a), r(b)),
                Node::Sub(a, b) => Node::Sub(r(a), r(b)),
                Node::Mul(a, b) => Node::Mul(r(a), r(b)),
                Node::Neg(a) => Node::Neg(r(a)),
                other => other,
            };
            remap[i] = out.len() as u32;
            out.push(node);
        }
        Self { nodes: out }
    }
}

/// Resets the thread's builder even if `eval` panics.
struct BuilderGuard;

impl Drop for BuilderGuard {
    fn drop(&mut self) {
        BUILDER.with(|b| *b.borrow_mut() = None);
    }
}

/// Compiles `air`'s constraint evaluation for `public_len` public values.
pub fn compile<A: Air>(air: &A, public_len: usize) -> Program {
    BUILDER.with(|b| {
        let mut b = b.borrow_mut();
        assert!(b.is_none(), "compile is not reentrant");
        *b = Some(Builder::default());
    });
    let _guard = BuilderGuard;
    let inputs = |f: fn(u32) -> Input, n: usize| -> Vec<Sym> {
        (0..n as u32).map(|j| Sym::input(f(j))).collect()
    };
    let (w, pw, aw) = (air.width(), air.preprocessed_width(), air.aux_width());
    let local = inputs(Input::Local, w);
    let next = inputs(Input::Next, w);
    let pre_local = inputs(Input::PreprocessedLocal, pw);
    let pre_next = inputs(Input::PreprocessedNext, pw);
    let aux_local = inputs(Input::AuxLocal, aw);
    let aux_next = inputs(Input::AuxNext, aw);
    let challenges = inputs(Input::Challenge, air.num_challenges());
    let public = inputs(Input::Public, public_len);
    let selectors = Selectors {
        is_first_row: Sym::input(Input::IsFirstRow),
        is_last_row: Sym::input(Input::IsLastRow),
        is_transition: Sym::input(Input::IsTransition),
    };
    let rows = Rows {
        local: &local,
        next: &next,
        preprocessed_local: &pre_local,
        preprocessed_next: &pre_next,
        aux_local: &aux_local,
        aux_next: &aux_next,
    };
    let alpha = Sym::input(Input::Alpha);
    let mut c = Constraints::new(rows, &challenges, &public, selectors, alpha);
    air.eval(&mut c);
    // Ensure the output is the last node even if it was interned earlier.
    let output = c.finish();
    let nodes = BUILDER.with(|b| b.borrow_mut().take().expect("builder").nodes);
    Program::pruned(nodes, output.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;
    use crate::item_chunk_air::{self, ItemChunkAir};
    use crate::poseidon2_air::Poseidon2Air;
    use crate::sig_chunk_air::{self, SigChunkAir};
    use crate::state_air::{self, StateSegmentAir};
    use crate::wots_air::WotsAir;

    fn rng(seed: u64) -> impl FnMut() -> BabyBear4 {
        let mut s = seed | 1;
        move || {
            let mut next = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % P as u64) as u32
            };
            BabyBear4::new([next(), next(), next(), next()])
        }
    }

    /// Direct evaluation and the compiled program agree at random inputs.
    fn check<A: Air>(air: &A, public_len: usize) -> Stats {
        let program = compile(air, public_len);
        let mut next = rng(public_len as u64 + air.width() as u64);
        let mut vec = |n: usize| (0..n).map(|_| next()).collect::<Vec<_>>();
        let (w, pw, aw) = (air.width(), air.preprocessed_width(), air.aux_width());
        let local = vec(w);
        let nxt = vec(w);
        let pre_local = vec(pw);
        let pre_next = vec(pw);
        let aux_local = vec(aw);
        let aux_next = vec(aw);
        let challenges = vec(air.num_challenges());
        let public = vec(public_len);
        let sel = vec(4);
        let selectors = Selectors {
            is_first_row: sel[0],
            is_last_row: sel[1],
            is_transition: sel[2],
        };
        let alpha = sel[3];

        let rows = Rows {
            local: &local,
            next: &nxt,
            preprocessed_local: &pre_local,
            preprocessed_next: &pre_next,
            aux_local: &aux_local,
            aux_next: &aux_next,
        };
        let mut c = Constraints::new(rows, &challenges, &public, selectors, alpha);
        air.eval(&mut c);
        let direct = c.finish();

        let inputs = Inputs {
            local: &local,
            next: &nxt,
            preprocessed_local: &pre_local,
            preprocessed_next: &pre_next,
            aux_local: &aux_local,
            aux_next: &aux_next,
            challenges: &challenges,
            public: &public,
            selectors,
            alpha,
        };
        assert_eq!(program.eval(&inputs), direct);
        program.stats()
    }

    /// An AIR with an auxiliary trace and challenges, to cover those inputs.
    struct WithLookup;

    impl Air for WithLookup {
        fn width(&self) -> usize {
            3
        }
        fn max_degree(&self) -> usize {
            3
        }
        fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
            let l = c.local();
            let terms = [
                crate::lookup::Term::new(V::from(BabyBear::ONE), vec![l[0], l[1]]),
                crate::lookup::Term::new(-l[2], vec![l[1]]),
            ];
            crate::lookup::eval(c, &terms, 0, 0);
            c.assert_zero_transition(c.next()[0] - l[0] * l[1]);
        }
        fn num_challenges(&self) -> usize {
            crate::lookup::CHALLENGES
        }
        fn aux_width(&self) -> usize {
            crate::lookup::aux_width(2)
        }
    }

    #[test]
    fn compiled_programs_match_direct_evaluation() {
        let cases: Vec<(&str, Stats)> = vec![
            ("lookup", check(&WithLookup, 0)),
            ("poseidon2", check(&Poseidon2Air, 0)),
            ("wots", check(&WotsAir, 21)),
            (
                "item chunk",
                check(&ItemChunkAir { items: 4 }, item_chunk_air::public::LEN),
            ),
            (
                "sig chunk",
                check(&SigChunkAir { slots: 1 }, sig_chunk_air::public::LEN),
            ),
            (
                "state segment",
                check(
                    &StateSegmentAir {
                        inserts: 1,
                        spends: 1,
                    },
                    state_air::public::LEN,
                ),
            ),
        ];
        for (name, s) in cases {
            eprintln!("{name}: {s:?}");
            assert!(s.muls > 0);
        }
    }

    #[test]
    fn folds_constants_and_deduplicates() {
        struct Tiny;
        impl Air for Tiny {
            fn width(&self) -> usize {
                1
            }
            fn max_degree(&self) -> usize {
                2
            }
            fn eval<V: Var>(&self, c: &mut Constraints<'_, V>) {
                let x = c.local()[0];
                let two = V::from(BabyBear::new(2));
                // (x * 1 + 0) * (x + x) - x * (x + x): constant-folded and
                // the shared x + x computed once.
                let a = (x * V::from(BabyBear::ONE) + V::from(BabyBear::ZERO)) * (x + x);
                let b = x * (x + x);
                c.assert_zero(a - b + two * V::from(BabyBear::ZERO));
            }
        }
        let p = compile(&Tiny, 0);
        // a and b are the same node, so a - b folds to 0, and so does the
        // whole combination: no inputs, no arithmetic left.
        assert_eq!(p.nodes, vec![Node::Const(BabyBear4::ZERO)]);
        assert!(
            p.eval(&Inputs {
                local: &[BabyBear4::new([5, 0, 0, 0])],
                next: &[BabyBear4::ZERO],
                preprocessed_local: &[],
                preprocessed_next: &[],
                aux_local: &[],
                aux_next: &[],
                challenges: &[],
                public: &[],
                selectors: Selectors {
                    is_first_row: BabyBear4::ZERO,
                    is_last_row: BabyBear4::ZERO,
                    is_transition: BabyBear4::ONE,
                },
                alpha: BabyBear4::new([3, 0, 0, 0]),
            }) == BabyBear4::ZERO
        );
    }

    #[test]
    #[should_panic(expected = "only inside `compile`")]
    fn symbolic_values_need_a_builder() {
        let _ = Sym::from(BabyBear::ONE);
    }
}
