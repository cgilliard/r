//! Proof of work: Poseidon2 over a 64 MiB dataset expanded from the
//! compressed Bible text, in the manner of Ethash.
//!
//! The cache is the text, `resources/akjv.txt.gz`, as items of 8 field
//! elements: one per 32-byte chunk (each little-endian `u32` reduced mod
//! `p`). The dataset has [`ITEMS`] items, each derived from the cache
//! alone: it starts from its index, then absorbs [`PARENTS`] cache items,
//! each chosen by its own state, with one permutation per parent.
//!
//! Miners build the whole dataset once (`ITEMS * PARENTS` permutations,
//! cached in a file) and read one item per round. Verifiers keep only the
//! cache, which is in the binary, and derive the items a hash reads
//! ([`light_hash`]: about `ROUNDS * PARENTS` permutations, milliseconds).
//! A miner that skips storing the dataset pays the same `PARENTS`
//! permutations for every item it reads instead of one memory read; with
//! 256 parents that is meant to cost well over a random DRAM read even in
//! custom hardware, whatever on-chip memory it has for the cache.
//!
//! A header's PoW hash starts from its seed (a hash of every header field but
//! the nonce) and the nonce, then runs [`ROUNDS`] of: look up the item the
//! state points at, add it into the rate, permute. The hash's value (the low
//! 27 bits of its first four elements, most significant first) must be
//! below the target. The dataset is committed to by a Merkle tree whose root
//! is fixed ([`ROOT`]), so a circuit can check the PoW with one Merkle
//! opening per round.

use std::ops::Range;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use crate::babybear::BabyBear;
use crate::hash::{DOMAIN_SLOT, HASH_LEN, Hash, domain};
use crate::matrix::RowMajorMatrix;
use crate::merkle::{MerklePath, MerkleTree};
use crate::par;
use crate::poseidon2::{self, WIDTH};
use crate::poseidon2_x8::{LANES, permute_x8};

pub const LOG_ITEMS: u32 = 21;
pub const ITEMS: usize = 1 << LOG_ITEMS;
/// Cache items absorbed by each dataset item.
pub const PARENTS: usize = 256;
/// Dataset lookups per PoW hash.
pub const ROUNDS: usize = 64;
/// Bytes of text per seed item.
pub const CHUNK: usize = 32;

pub type Item = [BabyBear; HASH_LEN];

/// The dataset's Merkle root (leaves are items hashed as rows).
pub const ROOT: [u32; HASH_LEN] = [
    1796128910, 1758294722, 640279288, 1524090617, 1744323623, 1199373394, 1519935544, 936409918,
];

/// Bits taken from each of the first four hash elements for its value.
pub const VALUE_LIMB_BITS: u32 = 27;
/// Bits of a hash value.
pub const VALUE_BITS: u32 = 4 * VALUE_LIMB_BITS;
/// Hash values are below this.
pub const MAX_TARGET: u128 = 1 << VALUE_BITS;

static TEXT: &[u8] = include_bytes!("../../resources/akjv.txt.gz");

/// The cache: one item per (zero-padded) 32-byte chunk of the text.
pub fn cache() -> &'static [Item] {
    static CACHE: OnceLock<Vec<Item>> = OnceLock::new();
    CACHE.get_or_init(|| {
        TEXT.chunks(CHUNK)
            .map(|chunk| {
                let mut bytes = [0u8; CHUNK];
                bytes[..chunk.len()].copy_from_slice(chunk);
                core::array::from_fn(|k| {
                    BabyBear::new(u32::from_le_bytes(
                        bytes[4 * k..4 * k + 4].try_into().expect("4 bytes"),
                    ))
                })
            })
            .collect()
    })
}

/// Dataset item `i`, derived from the cache.
pub fn item(i: usize) -> Item {
    let cache = cache();
    let mut s = item_start(i);
    poseidon2::permute(&mut s);
    for _ in 0..PARENTS {
        let j = parent(&s, cache.len());
        add(&mut s, &cache[j]);
        poseidon2::permute(&mut s);
    }
    core::array::from_fn(|k| s[k])
}

/// The PoW hash for `seed` and `nonce`, deriving the items it reads from
/// the cache: what a node that doesn't mine checks headers with.
pub fn light_hash(seed: &Hash, nonce: u64) -> Hash {
    hash_with(seed, nonce, item)
}

/// The PoW hash for `seed` and `nonce`, reading items with `item`.
fn hash_with(seed: &Hash, nonce: u64, item: impl Fn(usize) -> Item) -> Hash {
    let mut s = start(seed, nonce);
    poseidon2::permute(&mut s);
    for _ in 0..ROUNDS {
        let j = lookup(&s);
        add(&mut s, &item(j));
        poseidon2::permute(&mut s);
    }
    core::array::from_fn(|i| s[i])
}

pub struct Dataset {
    items: Vec<Item>,
}

static DATASET: OnceLock<Dataset> = OnceLock::new();

/// Bytes of a dataset file: every item's elements, little-endian `u32`s.
const FILE_BYTES: usize = ITEMS * HASH_LEN * 4;

impl Dataset {
    /// The shared dataset, built on first use.
    pub fn get() -> &'static Dataset {
        DATASET.get_or_init(Dataset::build)
    }

    /// The shared dataset and its tree, loaded from the file `path` if it
    /// holds the dataset (its tree's root must be [`ROOT`], which checks
    /// every item), else built and written there. Also says whether it was
    /// loaded.
    pub fn load(path: &Path) -> (&'static Dataset, MerkleTree, bool) {
        if let Some(d) = DATASET.get() {
            return (d, d.tree(), false);
        }
        if let Some(d) = Self::read(path) {
            let tree = d.tree();
            if tree.root() == BabyBear::new_array(ROOT) {
                let d = DATASET.get_or_init(|| d);
                return (d, tree, true);
            }
        }
        let d = Self::get();
        // Best effort: a missing cache only costs time.
        let _ = d.write(path);
        (d, d.tree(), false)
    }

    fn read(path: &Path) -> Option<Dataset> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.len() != FILE_BYTES {
            return None;
        }
        let mut values = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")));
        let mut items = Vec::with_capacity(ITEMS);
        for _ in 0..ITEMS {
            let mut item = [BabyBear::ZERO; HASH_LEN];
            for x in &mut item {
                let v = values.next()?;
                if v >= crate::babybear::P {
                    return None;
                }
                *x = BabyBear::new(v);
            }
            items.push(item);
        }
        Some(Self { items })
    }

    fn write(&self, path: &Path) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(FILE_BYTES);
        for x in self.items.iter().flatten() {
            bytes.extend_from_slice(&x.as_u32().to_le_bytes());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(tmp, path)
    }

    /// Builds the dataset (`ITEMS * PARENTS` = 2^29 permutations, spread
    /// across threads).
    pub fn build() -> Self {
        let mut items = vec![[BabyBear::ZERO; HASH_LEN]; ITEMS];
        let cache = cache();
        par::par_chunks_mut(&mut items, 1024, |c, out| {
            for (g, group) in out.chunks_mut(LANES).enumerate() {
                derive_x8(cache, c * 1024 + g * LANES, group);
            }
        });
        Self { items }
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    /// The Merkle tree over the items (one row each), for proving lookups.
    pub fn tree(&self) -> MerkleTree {
        let values = self.items.iter().flatten().copied().collect();
        MerkleTree::new(RowMajorMatrix::new(values, HASH_LEN))
    }

    /// The PoW hash for `seed` and `nonce` (the same as [`light_hash`]).
    pub fn hash(&self, seed: &Hash, nonce: u64) -> Hash {
        hash_with(seed, nonce, |j| self.items[j])
    }

    /// [`Dataset::hash`] of eight nonces at once.
    pub fn hash_x8(&self, seed: &Hash, nonces: [u64; LANES]) -> [Hash; LANES] {
        let mut s = nonces.map(|n| start(seed, n));
        permute_x8(&mut s);
        for _ in 0..ROUNDS {
            for s in &mut s {
                add(s, &self.items[lookup(s)]);
            }
            permute_x8(&mut s);
        }
        s.map(|s| core::array::from_fn(|i| s[i]))
    }

    /// Searches `nonces` on every thread for one whose hash is below
    /// `target`. Returns some such nonce (not necessarily the first).
    pub fn mine(&self, seed: &Hash, target: u128, nonces: Range<u64>) -> Option<u64> {
        let found = AtomicBool::new(false);
        let nonce = AtomicU64::new(0);
        let next = AtomicU64::new(nonces.start);
        // Batches of nonces claimed at a time, a multiple of the lanes.
        const BATCH: u64 = 64 * LANES as u64;
        thread::scope(|s| {
            for _ in 0..par::threads() {
                s.spawn(|| {
                    while !found.load(Ordering::Relaxed) {
                        let start = next.fetch_add(BATCH, Ordering::Relaxed);
                        if start >= nonces.end {
                            return;
                        }
                        let end = (start + BATCH).min(nonces.end);
                        for first in (start..end).step_by(LANES) {
                            let lanes = core::array::from_fn(|l| first + l as u64);
                            let hashes = self.hash_x8(seed, lanes);
                            for (l, h) in hashes.iter().enumerate() {
                                if lanes[l] < end && value(h) < target {
                                    nonce.store(lanes[l], Ordering::Relaxed);
                                    found.store(true, Ordering::Release);
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        });
        found
            .load(Ordering::Acquire)
            .then(|| nonce.load(Ordering::Relaxed))
    }
}

/// What a chain proof opens dataset items with: the dataset's Merkle tree
/// (a miner's, to prove), or a blank stand-in for a node that only
/// verifies, which lays the circuits out the same (so they have the same
/// keys) but can't prove.
pub enum DatasetTree {
    Full(MerkleTree),
    Blank,
}

impl DatasetTree {
    /// Item `index` and its path (zeros when blank).
    pub fn open(&self, index: usize) -> (Vec<BabyBear>, MerklePath) {
        match self {
            Self::Full(tree) => tree.open(index),
            Self::Blank => (
                vec![BabyBear::ZERO; HASH_LEN],
                vec![[BabyBear::ZERO; HASH_LEN]; LOG_ITEMS as usize],
            ),
        }
    }
}

/// The value compared against the target: the low 27 bits of each of the
/// first four elements, most significant first. Below [`MAX_TARGET`].
/// (Each element is uniform mod `p = 15 * 2^27 + 1`, so its low 27 bits
/// are uniform but for a negligible excess at 0.)
pub fn value(h: &Hash) -> u128 {
    let mask = (1 << VALUE_LIMB_BITS) - 1;
    h[..4]
        .iter()
        .fold(0, |v, x| v << VALUE_LIMB_BITS | (x.as_u32() & mask) as u128)
}

/// `x` as 16-bit limbs, least significant first.
pub fn u64_limbs(x: u64) -> [BabyBear; 4] {
    core::array::from_fn(|i| BabyBear::new((x >> (16 * i)) as u16 as u32))
}

fn start(seed: &Hash, nonce: u64) -> [BabyBear; WIDTH] {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[..HASH_LEN].copy_from_slice(seed);
    s[HASH_LEN..HASH_LEN + 4].copy_from_slice(&u64_limbs(nonce));
    s[DOMAIN_SLOT] = BabyBear::new(domain::POW);
    s
}

/// The item a PoW round reads: the low bits of the first capacity element.
fn lookup(s: &[BabyBear; WIDTH]) -> usize {
    s[HASH_LEN].as_u32() as usize & (ITEMS - 1)
}

fn add(s: &mut [BabyBear; WIDTH], item: &Item) {
    for (x, &y) in s.iter_mut().zip(item) {
        *x += y;
    }
}

/// The cache item (of `b`) that a dataset item reads next.
fn parent(s: &[BabyBear; WIDTH], b: usize) -> usize {
    let x = (s[HASH_LEN].as_u32() as u64) << 31 | s[HASH_LEN + 1].as_u32() as u64;
    (x % b as u64) as usize
}

fn item_start(i: usize) -> [BabyBear; WIDTH] {
    let mut s = [BabyBear::ZERO; WIDTH];
    s[0] = BabyBear::new(i as u32);
    s[DOMAIN_SLOT] = BabyBear::new(domain::POW_DATASET);
    s
}

/// Derives the items from index `first` into `out` (at most 8), reading
/// parents from `cache`.
fn derive_x8(cache: &[Item], first: usize, out: &mut [Item]) {
    let b = cache.len();
    let mut s: [[BabyBear; WIDTH]; LANES] = core::array::from_fn(|l| item_start(first + l));
    permute_x8(&mut s);
    for _ in 0..PARENTS {
        for s in &mut s {
            add(s, &cache[parent(s, b)]);
        }
        permute_x8(&mut s);
    }
    for (item, s) in out.iter_mut().zip(&s) {
        *item = core::array::from_fn(|i| s[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;

    #[test]
    fn derive_x8_matches_scalar() {
        let mut out = [[BabyBear::ZERO; HASH_LEN]; 5];
        derive_x8(cache(), 1000, &mut out);
        for (l, x) in out.iter().enumerate() {
            assert_eq!(*x, item(1000 + l));
        }
    }

    #[test]
    fn the_cache_is_the_text() {
        let c = cache();
        assert_eq!(c.len(), TEXT.len().div_ceil(CHUNK));
        let first = u32::from_le_bytes(TEXT[..4].try_into().unwrap());
        assert_eq!(c[0][0], BabyBear::new(first));
    }

    #[test]
    fn value_orders_digits() {
        let mut h = [BabyBear::ZERO; HASH_LEN];
        h[3] = BabyBear::new(5);
        assert_eq!(value(&h), 5);
        h[0] = BabyBear::new(1);
        assert_eq!(value(&h), 1 << 81 | 5);
        h = [BabyBear::new((1 << 27) - 1); HASH_LEN];
        assert_eq!(value(&h), MAX_TARGET - 1);
        // Only the low 27 bits count.
        h = [BabyBear::new(P - 1); HASH_LEN];
        assert_eq!(value(&h), 0);
    }

    #[test]
    fn dataset_files_must_be_whole() {
        let dir = crate::db::test::TempEnv::new();
        let path = dir.path.join("pow-dataset");
        assert!(Dataset::read(&path).is_none());
        std::fs::write(&path, [0u8; 64]).unwrap();
        assert!(Dataset::read(&path).is_none());
    }

    /// Builds the real dataset and checks it against the pinned root, the
    /// x8 hash and the miner. Slow in debug builds.
    #[test]
    #[ignore]
    fn dataset_root_hash_and_mining() {
        let t = std::time::Instant::now();
        let data = Dataset::build();
        eprintln!("build {:.2?}", t.elapsed());
        let items = data.items();
        for i in [0, 1, 40_000, ITEMS / 2 + 3, ITEMS - 1] {
            assert_eq!(items[i], item(i), "item {i}");
        }

        let t = std::time::Instant::now();
        let root = data.tree().root();
        eprintln!("tree {:.2?}", t.elapsed());
        eprintln!("root {:?}", root.map(|x| x.as_u32()));
        assert_eq!(root, BabyBear::new_array(ROOT));

        // The file cache round-trips, and the root catches a damaged copy.
        let dir = crate::db::test::TempEnv::new();
        let path = dir.path.join("pow-dataset");
        data.write(&path).unwrap();
        assert!(Dataset::read(&path).unwrap().items == data.items);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[100] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        assert_ne!(Dataset::read(&path).unwrap().tree().root(), root);

        let seed = BabyBear::new_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let nonces = [0, 1, 2, 3, u64::MAX, 5, 6, 7];
        let x8 = data.hash_x8(&seed, nonces);
        for (n, h) in nonces.iter().zip(&x8) {
            assert_eq!(*h, data.hash(&seed, *n));
        }
        // Verifiers get the same hash without the dataset.
        let t = std::time::Instant::now();
        assert_eq!(light_hash(&seed, 7), x8[7]);
        eprintln!("light hash {:.2?}", t.elapsed());

        let count = 1 << 16;
        let t = std::time::Instant::now();
        assert_eq!(data.mine(&seed, 0, 0..count), None);
        let rate = count as f64 / t.elapsed().as_secs_f64();
        eprintln!("{rate:.0} hashes/s on {} threads", par::threads());

        let target = MAX_TARGET >> 10;
        let n = data.mine(&seed, target, 0..u64::MAX).unwrap();
        assert!(value(&data.hash(&seed, n)) < target);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// Splits PoW time into permutations and dataset reads.
    #[test]
    #[ignore]
    fn memory_vs_compute() {
        let data = Dataset::get();
        let seed = [BabyBear::ONE; HASH_LEN];
        let n = 1 << 14;
        let run = |f: &dyn Fn(&[BabyBear; WIDTH]) -> usize| {
            let t = std::time::Instant::now();
            let mut acc = BabyBear::ZERO;
            for b in 0..n / LANES as u64 {
                let mut s: [[BabyBear; WIDTH]; LANES] =
                    core::array::from_fn(|l| start(&seed, b * 8 + l as u64));
                permute_x8(&mut s);
                for _ in 0..ROUNDS {
                    for s in &mut s {
                        let j = f(s);
                        add(s, &data.items[j]);
                    }
                    permute_x8(&mut s);
                }
                acc += s[0][0];
            }
            std::hint::black_box(acc);
            t.elapsed().as_secs_f64() / n as f64 * 1e6
        };
        let random = run(&|s| lookup(s));
        let cached = run(&|s| lookup(s) & 63);
        let t = std::time::Instant::now();
        let mut s = [[BabyBear::ONE; WIDTH]; LANES];
        for _ in 0..n * (ROUNDS as u64 + 1) / LANES as u64 {
            permute_x8(&mut s);
        }
        std::hint::black_box(s);
        let perm = t.elapsed().as_secs_f64() / (n * (ROUNDS as u64 + 1)) as f64 * 1e9;
        eprintln!(
            "one thread: random {random:.2} us/hash, cached {cached:.2} us/hash, \
             {perm:.0} ns/perm (x8), {:.0} ns/round",
            random * 1e3 / ROUNDS as f64
        );
    }
}
