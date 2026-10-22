//! Large objects as chunks with a Merkle root, so they can be sent in
//! datagrams and every piece checked as it arrives.
//!
//! An object of `size` bytes is cut into [`CHUNK_BYTES`]-byte data chunks
//! (the last may be shorter). Each chunk's leaf hash packs its bytes three
//! to a field element; the leaves form a Merkle tree (padded with zero
//! hashes to a power of two), and the object hash is
//! `sponge(domain, size, root)`. A block's body hash is this hash of its
//! encoded body ([`crate::block::Block::body_hash`]).
//!
//! On the wire an object is [`Layout::chunks`] chunks: first the leaf
//! hashes, [`HASHES_PER_CHUNK`] to a chunk, then the data. A receiver that
//! knows the object hash gets the size and root from the sender, checks
//! them against it, and checks the leaf list against the root once it has
//! all of it ([`Assembly`]). Data chunks are then checked one by one against
//! their leaves, so a bad peer can waste at most the (small) hash list.

use crate::babybear::BabyBear;
use crate::hash::{self, HASH_BYTES, Hash, domain};

/// Bytes in a data chunk.
pub const CHUNK_BYTES: usize = 1024;
/// Leaf hashes in a hash chunk.
pub const HASHES_PER_CHUNK: usize = CHUNK_BYTES / HASH_BYTES;

/// Data chunks in an object of `size` bytes (at least one).
pub fn data_chunks(size: usize) -> usize {
    size.div_ceil(CHUNK_BYTES).max(1)
}

/// A data chunk's leaf hash.
pub fn leaf(data: &[u8]) -> Hash {
    let elems: Vec<BabyBear> = data
        .chunks(3)
        .map(|b| {
            let mut w = [0u8; 4];
            w[..b.len()].copy_from_slice(b);
            BabyBear::new(u32::from_le_bytes(w))
        })
        .collect();
    hash::sponge(domain::BODY_CHUNK, &[], &[&elems])
}

/// Every data chunk's leaf hash.
pub fn leaves(bytes: &[u8]) -> Vec<Hash> {
    if bytes.is_empty() {
        return vec![leaf(&[])];
    }
    crate::par::par_map(data_chunks(bytes.len()), 64, |i| {
        leaf(&bytes[i * CHUNK_BYTES..((i + 1) * CHUNK_BYTES).min(bytes.len())])
    })
}

/// The Merkle root of `leaves` (zero hashes pad the bottom layer to a power
/// of two).
pub fn root(leaves: &[Hash]) -> Hash {
    let mut layer = leaves.to_vec();
    layer.resize(leaves.len().next_power_of_two(), [BabyBear::ZERO; 8]);
    while layer.len() > 1 {
        layer = layer.chunks(2).map(|p| hash::merge(&p[0], &p[1])).collect();
    }
    layer[0]
}

/// The hash of an object of `size` bytes whose leaves have root `root`,
/// under `domain`.
pub fn object_hash(domain: u32, size: usize, root: &Hash) -> Hash {
    hash::sponge(domain, &[], &[&[BabyBear::new(size as u32)], root])
}

/// The hash of `bytes` as an object under `domain`.
pub fn hash_of(domain: u32, bytes: &[u8]) -> Hash {
    object_hash(domain, bytes.len(), &root(&leaves(bytes)))
}

/// How an object of `size` bytes is laid out on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size: usize,
}

impl Layout {
    pub fn data_chunks(&self) -> usize {
        data_chunks(self.size)
    }

    pub fn hash_chunks(&self) -> usize {
        self.data_chunks().div_ceil(HASHES_PER_CHUNK)
    }

    /// Chunks on the wire: hash chunks, then data chunks.
    pub fn chunks(&self) -> usize {
        self.hash_chunks() + self.data_chunks()
    }

    /// Bytes in wire chunk `i`.
    pub fn chunk_len(&self, i: usize) -> usize {
        let h = self.hash_chunks();
        if i < h {
            let n = self.data_chunks() - i * HASHES_PER_CHUNK;
            n.min(HASHES_PER_CHUNK) * HASH_BYTES
        } else {
            let d = i - h;
            (self.size - d * CHUNK_BYTES).min(CHUNK_BYTES)
        }
    }
}

/// An object ready to serve: its bytes and leaves.
#[derive(Clone, Debug)]
pub struct Served {
    pub bytes: Vec<u8>,
    pub leaves: Vec<Hash>,
    pub root: Hash,
}

impl Served {
    pub fn new(bytes: Vec<u8>) -> Self {
        let leaves = leaves(&bytes);
        let root = root(&leaves);
        Self {
            bytes,
            leaves,
            root,
        }
    }

    pub fn layout(&self) -> Layout {
        Layout {
            size: self.bytes.len(),
        }
    }

    /// Wire chunk `i`, if there is one.
    pub fn chunk(&self, i: usize) -> Option<Vec<u8>> {
        let layout = self.layout();
        let h = layout.hash_chunks();
        if i < h {
            let leaves = &self.leaves[i * HASHES_PER_CHUNK..];
            let n = leaves.len().min(HASHES_PER_CHUNK);
            Some(leaves[..n].iter().flat_map(hash::to_bytes).collect())
        } else if i < layout.chunks() {
            let at = (i - h) * CHUNK_BYTES;
            Some(self.bytes[at..at + layout.chunk_len(i)].to_vec())
        } else {
            None
        }
    }
}

/// Why a chunk was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkError {
    /// Out of range or the wrong length.
    Shape,
    /// The leaf list doesn't have the root (all hash chunks are dropped).
    BadHashes,
    /// A data chunk doesn't match its leaf.
    BadData,
}

/// An object being received: what's arrived so far, all of it checked.
#[derive(Clone, Debug)]
pub struct Assembly {
    layout: Layout,
    root: Hash,
    hash_parts: Vec<Option<Vec<u8>>>,
    /// Known (and checked against the root) once every hash chunk is in.
    leaves: Option<Vec<Hash>>,
    data: Vec<u8>,
    have: Vec<bool>,
    missing: usize,
}

impl Assembly {
    /// The object's size in bytes.
    pub fn size(&self) -> usize {
        self.layout.size
    }

    /// Expects an object of `size` bytes with leaf root `root`; the caller
    /// has checked these against the object hash.
    pub fn new(size: usize, root: Hash) -> Self {
        let layout = Layout { size };
        Self {
            layout,
            root,
            hash_parts: vec![None; layout.hash_chunks()],
            leaves: None,
            data: vec![0; size],
            have: vec![false; layout.data_chunks()],
            missing: layout.chunks(),
        }
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn is_complete(&self) -> bool {
        self.missing == 0
    }

    /// Whether the leaf hashes are in (and checked), so data chunks can be
    /// taken.
    pub fn has_leaves(&self) -> bool {
        self.leaves.is_some()
    }

    /// Chunks received so far.
    pub fn received(&self) -> usize {
        self.layout.chunks() - self.missing
    }

    /// Up to `limit` wire chunks still needed, in order: hash chunks until
    /// the leaves are known, then data chunks.
    pub fn wanted(&self, limit: usize) -> Vec<usize> {
        let h = self.layout.hash_chunks();
        if self.leaves.is_none() {
            return (0..h)
                .filter(|&i| self.hash_parts[i].is_none())
                .take(limit)
                .collect();
        }
        (0..self.have.len())
            .filter(|&d| !self.have[d])
            .take(limit)
            .map(|d| h + d)
            .collect()
    }

    /// Takes wire chunk `i`. Returns whether it was new.
    pub fn add(&mut self, i: usize, bytes: &[u8]) -> Result<bool, ChunkError> {
        if i >= self.layout.chunks() || bytes.len() != self.layout.chunk_len(i) {
            return Err(ChunkError::Shape);
        }
        let h = self.layout.hash_chunks();
        if i < h {
            if self.leaves.is_some() || self.hash_parts[i].is_some() {
                return Ok(false);
            }
            self.hash_parts[i] = Some(bytes.to_vec());
            self.missing -= 1;
            if self.hash_parts.iter().all(Option::is_some) {
                let leaves: Option<Vec<Hash>> = self
                    .hash_parts
                    .iter()
                    .flatten()
                    .flat_map(|p| p.chunks_exact(HASH_BYTES))
                    .map(|b| hash::from_bytes(b.try_into().expect("hash bytes")))
                    .collect();
                match leaves {
                    Some(l) if root(&l) == self.root => self.leaves = Some(l),
                    _ => {
                        self.hash_parts.iter_mut().for_each(|p| *p = None);
                        self.missing += h;
                        return Err(ChunkError::BadHashes);
                    }
                }
            }
            return Ok(true);
        }
        let Some(leaves) = &self.leaves else {
            // Data before the leaves can't be checked yet: ask again later.
            return Ok(false);
        };
        let d = i - h;
        if self.have[d] {
            return Ok(false);
        }
        if leaf(bytes) != leaves[d] {
            return Err(ChunkError::BadData);
        }
        self.data[d * CHUNK_BYTES..d * CHUNK_BYTES + bytes.len()].copy_from_slice(bytes);
        self.have[d] = true;
        self.missing -= 1;
        Ok(true)
    }

    /// The object, once complete.
    pub fn finish(self) -> Option<Vec<u8>> {
        self.is_complete().then_some(self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(size: usize) -> Vec<u8> {
        (0..size).map(|i| (i * 7 + i / 300) as u8).collect()
    }

    #[test]
    fn layouts() {
        let l = Layout { size: 1 };
        assert_eq!(
            (
                l.hash_chunks(),
                l.data_chunks(),
                l.chunk_len(0),
                l.chunk_len(1)
            ),
            (1, 1, 32, 1)
        );
        let l = Layout {
            size: 32 * CHUNK_BYTES + 5,
        };
        assert_eq!((l.hash_chunks(), l.data_chunks()), (2, 33));
        assert_eq!((l.chunk_len(0), l.chunk_len(1)), (CHUNK_BYTES, 32));
        assert_eq!((l.chunk_len(2), l.chunk_len(34)), (CHUNK_BYTES, 5));
        // Two megabytes: a hash list of about 3%.
        let l = Layout { size: 2_000_000 };
        assert_eq!((l.hash_chunks(), l.data_chunks()), (62, 1954));
    }

    #[test]
    fn served_objects_reassemble() {
        for size in [1, 1000, CHUNK_BYTES, 40 * CHUNK_BYTES + 17] {
            let bytes = object(size);
            let s = Served::new(bytes.clone());
            assert_eq!(object_hash(7, size, &s.root), hash_of(7, &bytes));
            let mut a = Assembly::new(size, s.root);
            // Data before the leaves is put off.
            let h = s.layout().hash_chunks();
            assert_eq!(a.add(h, &s.chunk(h).unwrap()), Ok(false));
            // Each round takes what's wanted, newest first.
            while !a.is_complete() {
                for i in a.wanted(5).into_iter().rev() {
                    assert_eq!(
                        a.add(i, &s.chunk(i).unwrap()),
                        Ok(true),
                        "size {size} chunk {i}"
                    );
                    assert_eq!(a.add(i, &s.chunk(i).unwrap()), Ok(false));
                }
            }
            assert_eq!(a.finish().unwrap(), bytes);
        }
    }

    #[test]
    fn bad_chunks_are_refused() {
        let bytes = object(5 * CHUNK_BYTES + 3);
        let s = Served::new(bytes);
        let mut a = Assembly::new(s.bytes.len(), s.root);
        assert_eq!(a.add(99, &[0]), Err(ChunkError::Shape));
        assert_eq!(a.add(0, &[0; 3]), Err(ChunkError::Shape));
        // A wrong leaf list fails the root and is dropped whole.
        let mut bad = s.chunk(0).unwrap();
        bad[0] ^= 1;
        assert_eq!(a.add(0, &bad), Err(ChunkError::BadHashes));
        assert_eq!(a.wanted(10), vec![0]);
        assert_eq!(a.add(0, &s.chunk(0).unwrap()), Ok(true));
        let mut bad = s.chunk(3).unwrap();
        bad[10] ^= 1;
        assert_eq!(a.add(3, &bad), Err(ChunkError::BadData));
        assert!(a.wanted(10).contains(&3));
    }
}
