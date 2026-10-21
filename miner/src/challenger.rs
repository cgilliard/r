//! Fiat-Shamir transcript: a Poseidon2 duplex sponge.
//!
//! The prover and verifier feed it the same messages (`observe`) in the same
//! order and draw the same challenges (`sample`), so challenges depend on
//! everything the prover has committed to before them.
//!
//! Absorbed elements overwrite the rate; once the rate is full, or before
//! squeezing, the state is permuted. The number of elements absorbed is also
//! added into the capacity first, so trailing zeros can't go unnoticed. Squeezed elements come from the rate of
//! the latest permutation, and any new observation discards those left.

use crate::babybear::{BabyBear, P};
use crate::babybear4::BabyBear4;
use crate::hash::{DOMAIN_SLOT, Hash, domain};
use crate::poseidon2::{self, WIDTH};

const RATE: usize = 8;

#[derive(Clone, Debug)]
pub struct Challenger {
    state: [BabyBear; WIDTH],
    input: Vec<BabyBear>,
    output: Vec<BabyBear>,
}

impl Default for Challenger {
    fn default() -> Self {
        Self::new()
    }
}

impl Challenger {
    pub fn new() -> Self {
        let mut state = [BabyBear::ZERO; WIDTH];
        state[DOMAIN_SLOT] = BabyBear::new(domain::TRANSCRIPT);
        Self {
            state,
            input: Vec::with_capacity(RATE),
            output: Vec::with_capacity(RATE),
        }
    }

    pub fn observe(&mut self, x: BabyBear) {
        self.output.clear();
        self.input.push(x);
        if self.input.len() == RATE {
            self.duplex();
        }
    }

    pub fn observe_slice(&mut self, xs: &[BabyBear]) {
        for &x in xs {
            self.observe(x);
        }
    }

    pub fn observe_hash(&mut self, h: &Hash) {
        self.observe_slice(h);
    }

    pub fn observe_ext(&mut self, x: BabyBear4) {
        self.observe_slice(&x.0);
    }

    pub fn sample(&mut self) -> BabyBear {
        if !self.input.is_empty() || self.output.is_empty() {
            self.duplex();
        }
        self.output.pop().expect("duplex fills the output")
    }

    pub fn sample_ext(&mut self) -> BabyBear4 {
        BabyBear4([self.sample(), self.sample(), self.sample(), self.sample()])
    }

    /// A uniformly random integer below `2^bits` (`bits <= 27`). Samples are
    /// uniform on `[0, p)` with `p = 15 * 2^27 + 1`, so the low 27 bits are
    /// uniform except for a 2^-31 excess on zero.
    pub fn sample_bits(&mut self, bits: u32) -> usize {
        assert!(bits <= 27, "can only sample up to 27 bits");
        (self.sample().as_u32() & ((1 << bits) - 1)) as usize
    }

    /// Proof-of-work: finds a witness `w` such that observing it makes the
    /// next `sample_bits(bits)` zero, and leaves the transcript in that
    /// state. Each extra bit doubles the prover's work and adds a bit of
    /// soundness.
    pub fn grind(&mut self, bits: u32) -> BabyBear {
        use std::sync::atomic::{AtomicU32, Ordering};
        // Threads try interleaved candidates and stop once any succeeds;
        // the smallest success found wins (any valid witness would do).
        let threads = crate::par::threads() as u32;
        let found = AtomicU32::new(P);
        std::thread::scope(|s| {
            for t in 0..threads {
                let (found, this) = (&found, &*self);
                s.spawn(move || {
                    let mut w = t;
                    while w < found.load(Ordering::Relaxed) {
                        if this.clone().check_witness(bits, BabyBear::new(w)) {
                            found.fetch_min(w, Ordering::Relaxed);
                            return;
                        }
                        w += threads;
                    }
                });
            }
        });
        let w = BabyBear::new(found.into_inner());
        assert!(
            self.check_witness(bits, w),
            "some witness below p succeeds for bits <= 27"
        );
        w
    }

    /// Observes `w` and checks it is a valid [`grind`](Self::grind) witness.
    pub fn check_witness(&mut self, bits: u32, w: BabyBear) -> bool {
        self.observe(w);
        self.sample_bits(bits) == 0
    }

    fn duplex(&mut self) {
        for (s, &x) in self.state.iter_mut().zip(&self.input) {
            *s = x;
        }
        self.state[WIDTH - 1] += BabyBear::new(self.input.len() as u32);
        self.input.clear();
        poseidon2::permute(&mut self.state);
        self.output.clear();
        self.output.extend_from_slice(&self.state[..RATE]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elems(xs: &[u32]) -> Vec<BabyBear> {
        xs.iter().map(|&x| BabyBear::new(x)).collect()
    }

    #[test]
    fn same_transcript_same_challenges() {
        let mut a = Challenger::new();
        let mut b = Challenger::new();
        for c in [&mut a, &mut b] {
            c.observe_slice(&elems(&[1, 2, 3]));
        }
        assert_eq!(a.sample_ext(), b.sample_ext());
        assert_eq!(a.sample_bits(20), b.sample_bits(20));
    }

    #[test]
    fn challenges_depend_on_observations() {
        let sample_after = |xs: &[u32]| {
            let mut c = Challenger::new();
            c.observe_slice(&elems(xs));
            c.sample()
        };
        assert_ne!(sample_after(&[1, 2, 3]), sample_after(&[1, 2, 4]));
        assert_ne!(sample_after(&[1, 2, 3]), sample_after(&[1, 2, 3, 0]));
        // Spans a full rate's worth of input.
        let long: Vec<u32> = (0..20).collect();
        let mut longer = long.clone();
        longer[19] = 99;
        assert_ne!(sample_after(&long), sample_after(&longer));
    }

    #[test]
    fn consecutive_samples_differ() {
        let mut c = Challenger::new();
        c.observe(BabyBear::ONE);
        let xs: Vec<_> = (0..20).map(|_| c.sample()).collect();
        for i in 0..xs.len() {
            for j in i + 1..xs.len() {
                assert_ne!(xs[i], xs[j]);
            }
        }
    }

    #[test]
    fn observing_after_sampling_changes_later_samples() {
        let mut a = Challenger::new();
        let mut b = Challenger::new();
        a.sample();
        b.sample();
        a.observe(BabyBear::ONE);
        b.observe(BabyBear::new(2));
        assert_ne!(a.sample(), b.sample());
    }

    #[test]
    fn grinding_witness_verifies() {
        let mut prover = Challenger::new();
        prover.observe_slice(&elems(&[5, 6, 7]));
        let mut verifier = prover.clone();
        let w = prover.grind(8);

        assert!(verifier.check_witness(8, w));
        // Both continue in the same state.
        assert_eq!(prover.sample(), verifier.sample());

        // `grind` returns the smallest witness, so every smaller one fails.
        let mut base = Challenger::new();
        base.observe_slice(&elems(&[5, 6, 7]));
        for smaller in 0..w.as_u32() {
            assert!(!base.clone().check_witness(8, BabyBear::new(smaller)));
        }
    }
}
