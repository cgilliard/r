//! Cryptographic randomness for proof masking: a Poseidon2 keyed generator
//! (`compress(key || counter)`, see [`hash::compress`]) seeded from the OS.

use std::fs::File;
use std::io::{self, Read};

use crate::babybear::{BabyBear, P};
use crate::hash::{DOMAIN_SLOT, HASH_LEN, Hash, compress, domain};
use crate::par::par_chunks_mut;
use crate::poseidon2::WIDTH;
use crate::poseidon2_x8::{LANES, permute_x8};

pub struct Rng {
    key: Hash,
    counter: u64,
    buf: Vec<BabyBear>,
}

impl Rng {
    /// Seeds from `/dev/urandom`.
    pub fn from_os() -> io::Result<Self> {
        let mut bytes = [0u8; 8 * HASH_LEN];
        File::open("/dev/urandom")?.read_exact(&mut bytes)?;
        // u64 mod p: bias about p / 2^64 = 2^-33 per element.
        let key = core::array::from_fn(|i| {
            let v = u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap());
            BabyBear::new((v % P as u64) as u32)
        });
        Ok(Self::from_key(key))
    }

    /// Deterministic generator, for tests and reproducible runs only: masks
    /// from a known seed do not hide anything.
    pub fn from_seed(seed: u64) -> Self {
        let mut key = [BabyBear::ZERO; HASH_LEN];
        key[0] = BabyBear::new((seed % P as u64) as u32);
        key[1] = BabyBear::new((seed >> 32) as u32 % P);
        Self::from_key(key)
    }

    fn from_key(key: Hash) -> Self {
        Self {
            key,
            counter: 0,
            buf: Vec::with_capacity(HASH_LEN),
        }
    }

    /// A uniformly random field element.
    pub fn next_element(&mut self) -> BabyBear {
        if self.buf.is_empty() {
            self.buf.extend_from_slice(&block(&self.key, self.counter));
            self.counter += 1;
        }
        self.buf.pop().expect("just refilled")
    }

    /// Fills `out` with the next `out.len()` elements; whole blocks are
    /// generated in parallel.
    pub fn fill(&mut self, out: &mut [BabyBear]) {
        let mut i = 0;
        while i < out.len() && !self.buf.is_empty() {
            out[i] = self.buf.pop().expect("nonempty");
            i += 1;
        }
        let rest = &mut out[i..];
        let blocks = rest.len() / HASH_LEN;
        let (whole, tail) = rest.split_at_mut(blocks * HASH_LEN);
        let (key, start) = (self.key, self.counter);
        // Runs of 64 blocks per task, 8 blocks per batched permutation.
        par_chunks_mut(whole, 64 * HASH_LEN, |c, run| {
            for (g, group) in run.chunks_mut(LANES * HASH_LEN).enumerate() {
                let first = start + (c * 64 + g * LANES) as u64;
                let inputs: [[BabyBear; WIDTH]; LANES] =
                    core::array::from_fn(|l| block_input(&key, first + l as u64));
                let mut states = inputs;
                permute_x8(&mut states);
                for (l, chunk) in group.chunks_exact_mut(HASH_LEN).enumerate() {
                    // compress, elements last first as from `next_element`.
                    for (k, x) in chunk.iter_mut().enumerate() {
                        let i = HASH_LEN - 1 - k;
                        *x = states[l][i] + inputs[l][i];
                    }
                }
            }
        });
        self.counter += blocks as u64;
        for x in tail {
            *x = self.next_element();
        }
    }
}

/// Block `counter` of the stream: `compress(key || counter || ZK_RNG)`.
fn block(key: &Hash, counter: u64) -> Hash {
    compress(block_input(key, counter))
}

fn block_input(key: &Hash, counter: u64) -> [BabyBear; WIDTH] {
    let mut input = [BabyBear::ZERO; WIDTH];
    input[..HASH_LEN].copy_from_slice(key);
    input[HASH_LEN] = BabyBear::new((counter & ((1 << 30) - 1)) as u32);
    input[HASH_LEN + 1] = BabyBear::new((counter >> 30) as u32);
    input[DOMAIN_SLOT] = BabyBear::new(domain::ZK_RNG);
    input
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_matches_single_draws() {
        let mut a = Rng::from_seed(3);
        let mut b = Rng::from_seed(3);
        for len in [0, 3, 8, 13, 5000, 1, 64 * 8 * 13 + 5] {
            let mut v = vec![BabyBear::ZERO; len];
            a.fill(&mut v);
            let w: Vec<BabyBear> = (0..len).map(|_| b.next_element()).collect();
            assert_eq!(v, w, "len {len}");
        }
    }

    #[test]
    fn seeded_is_deterministic_and_os_is_not() {
        let take = |mut r: Rng| (0..20).map(|_| r.next_element()).collect::<Vec<_>>();
        assert_eq!(take(Rng::from_seed(7)), take(Rng::from_seed(7)));
        assert_ne!(take(Rng::from_seed(7)), take(Rng::from_seed(8)));

        let mut x = Rng::from_os().unwrap();
        let mut y = Rng::from_os().unwrap();
        assert_ne!(x.next_element(), y.next_element());
    }

    #[test]
    fn outputs_do_not_repeat() {
        let mut r = Rng::from_seed(1);
        let xs: Vec<_> = (0..64).map(|_| r.next_element()).collect();
        let mut sorted: Vec<_> = xs.iter().map(|x| x.as_u32()).collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), xs.len());
    }
}
