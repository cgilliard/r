//! Preimages of SHA-256 hashes that blocks' aux roots lead to: the nodes of
//! aux trees (each the 64 bytes `left || right` of its two children) and
//! files that leaves commit to (such as the relay consensus,
//! [`crate::onion`]). A node with this store serves them to peers and
//! fetches the ones its recent blocks lead to ([`crate::net::node`]), so a
//! leaf can be followed down from any header, as many levels as it goes.
//!
//! Kept in LMDB: database `preimages` (hash to bytes) and
//! `preimage_heights` (hash to the highest block that led to it, as 8
//! little-endian bytes). Preimages no recent block leads to are pruned
//! ([`PreimageStore::prune`]), so a miner can't fill others' disks by
//! committing to a new file in every block. The files our own aux leaves
//! were set from count as stored too (and are never pruned), so they're
//! never fetched.

use heed::types::Bytes;
use heed::{Database, Env};

use crate::aux_leaves::{AUX_LEAVES, AuxStore, Leaf};
use crate::mnemonic::sha256;

/// The largest preimage stored or fetched.
pub const MAX_PREIMAGE: usize = 1 << 20;
/// Levels of 64-byte nodes from an aux root down to its leaves.
pub const AUX_DEPTH: u8 = AUX_LEAVES.trailing_zeros() as u8;

/// Joins two hashes into the preimage of their parent.
pub fn node_bytes(left: &Leaf, right: &Leaf) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(left);
    out[32..].copy_from_slice(right);
    out
}

/// The two children in a node's preimage, if it is one (64 bytes).
pub fn children(bytes: &[u8]) -> Option<(Leaf, Leaf)> {
    (bytes.len() == 64).then(|| {
        (
            bytes[..32].try_into().expect("32 bytes"),
            bytes[32..].try_into().expect("32 bytes"),
        )
    })
}

/// The preimages of every node of the aux tree over `leaves` (none if
/// they're all zero: the root is zero then), root first.
pub fn tree_nodes(leaves: &[Leaf; AUX_LEAVES]) -> Vec<[u8; 64]> {
    if leaves.iter().all(|l| *l == [0; 32]) {
        return vec![];
    }
    let mut out = Vec::new();
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let nodes: Vec<[u8; 64]> = level
            .chunks_exact(2)
            .map(|p| node_bytes(&p[0], &p[1]))
            .collect();
        level = nodes.iter().map(|n| sha256(n)).collect();
        out.splice(0..0, nodes);
    }
    out
}

#[derive(Clone, Debug)]
pub struct PreimageStore {
    env: Env,
    data: Database<Bytes, Bytes>,
    heights: Database<Bytes, Bytes>,
    aux: AuxStore,
}

impl PreimageStore {
    pub fn open(env: &Env) -> heed::Result<Self> {
        let mut txn = env.write_txn()?;
        let data = env.create_database(&mut txn, Some("preimages"))?;
        let heights = env.create_database(&mut txn, Some("preimage_heights"))?;
        txn.commit()?;
        Ok(Self {
            env: env.clone(),
            data,
            heights,
            aux: AuxStore::open(env)?,
        })
    }

    /// The preimage of `hash`, if stored (or one of our aux files).
    pub fn get(&self, hash: &Leaf) -> heed::Result<Option<Vec<u8>>> {
        let stored = {
            let txn = self.env.read_txn()?;
            self.data.get(&txn, hash)?.map(<[u8]>::to_vec)
        };
        match stored {
            Some(b) => Ok(Some(b)),
            None => self.aux.file_for(&self.env, hash),
        }
    }

    /// Stores `bytes` (if new) as led to by block `height`: its hash.
    pub fn put(&self, bytes: &[u8], height: u64) -> heed::Result<Leaf> {
        self.put_all(&[bytes], height).map(|h| h[0])
    }

    /// Stores each of `items` (the new ones) as led to by block `height`:
    /// their hashes.
    pub fn put_all(&self, items: &[&[u8]], height: u64) -> heed::Result<Vec<Leaf>> {
        let mut txn = self.env.write_txn()?;
        let mut hashes = Vec::with_capacity(items.len());
        for bytes in items {
            let hash = sha256(bytes);
            if self.data.get(&txn, &hash)?.is_none() {
                self.data.put(&mut txn, &hash, bytes)?;
            }
            self.raise(&mut txn, &hash, height)?;
            hashes.push(hash);
        }
        txn.commit()?;
        Ok(hashes)
    }

    fn raise(&self, txn: &mut heed::RwTxn, hash: &Leaf, height: u64) -> heed::Result<()> {
        let old = self
            .heights
            .get(txn, hash)?
            .and_then(|b| <[u8; 8]>::try_from(b).ok())
            .map(u64::from_le_bytes);
        if old.is_none_or(|h| h < height) {
            self.heights.put(txn, hash, &height.to_le_bytes())?;
        }
        Ok(())
    }

    /// Notes, for each `(hash, height)` whose preimage is stored, that
    /// block `height` leads to it, so it's kept.
    pub fn touch(&self, hashes: &[(Leaf, u64)]) -> heed::Result<()> {
        let mut txn = self.env.write_txn()?;
        for (hash, height) in hashes {
            if self.data.get(&txn, hash)?.is_some() {
                self.raise(&mut txn, hash, *height)?;
            }
        }
        txn.commit()
    }

    /// Stores the aux tree over `leaves` and `files` (each leaf's
    /// preimage, where there is one) as led to by block `height`: what a
    /// miner serves for the blocks it mines.
    pub fn put_tree(
        &self,
        leaves: &[Leaf; AUX_LEAVES],
        files: &[Vec<u8>],
        height: u64,
    ) -> heed::Result<()> {
        let nodes = tree_nodes(leaves);
        let items: Vec<&[u8]> = nodes
            .iter()
            .map(|n| &n[..])
            .chain(files.iter().map(Vec::as_slice))
            .collect();
        self.put_all(&items, height).map(|_| ())
    }

    /// Leaf `n` of the aux tree with root `root`, if the nodes down to it
    /// are stored.
    pub fn leaf(&self, root: &Leaf, n: usize) -> heed::Result<Option<Leaf>> {
        let mut hash = *root;
        for level in (0..AUX_DEPTH).rev() {
            let Some((left, right)) = self.get(&hash)?.as_deref().and_then(children) else {
                return Ok(None);
            };
            hash = if (n >> level) & 1 == 0 { left } else { right };
        }
        Ok(Some(hash))
    }

    /// Removes the preimages no block from `height` on leads to: how many.
    pub fn prune(&self, height: u64) -> heed::Result<usize> {
        let mut txn = self.env.write_txn()?;
        let old: Vec<Vec<u8>> = self
            .heights
            .iter(&txn)?
            .filter_map(|e| {
                let (hash, h) = e.ok()?;
                let h = u64::from_le_bytes(h.try_into().ok()?);
                (h < height).then(|| hash.to_vec())
            })
            .collect();
        for hash in &old {
            self.heights.delete(&mut txn, hash)?;
            self.data.delete(&mut txn, hash)?;
        }
        txn.commit()?;
        Ok(old.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_leaves::root;
    use crate::db::test::TempEnv;

    #[test]
    fn a_tree_is_stored_walked_and_pruned() {
        let env = TempEnv::new();
        let store = PreimageStore::open(env.env()).unwrap();
        let mut leaves = [[0u8; 32]; AUX_LEAVES];
        assert!(tree_nodes(&leaves).is_empty());
        let file = b"onion\n1.2.3.4:5 ab".to_vec();
        leaves[0] = sha256(&file);
        leaves[9] = [9; 32];
        let nodes = tree_nodes(&leaves);
        assert_eq!(nodes.len(), AUX_LEAVES - 1);
        assert_eq!(sha256(&nodes[0]), root(&leaves));
        store
            .put_tree(&leaves, std::slice::from_ref(&file), 10)
            .unwrap();
        let r = root(&leaves);
        assert_eq!(store.leaf(&r, 0).unwrap(), Some(leaves[0]));
        assert_eq!(store.leaf(&r, 9).unwrap(), Some([9; 32]));
        assert_eq!(store.leaf(&r, 3).unwrap(), Some([0; 32]));
        assert_eq!(store.get(&leaves[0]).unwrap(), Some(file.clone()));
        assert_eq!(store.leaf(&[1; 32], 0).unwrap(), None);

        // A later block leads to the file alone; the tree goes first (its
        // zero subtrees share nodes).
        store.touch(&[(leaves[0], 20), ([5; 32], 20)]).unwrap();
        let distinct: std::collections::HashSet<_> = nodes.iter().collect();
        assert_eq!(store.prune(15).unwrap(), distinct.len());
        assert_eq!(store.get(&leaves[0]).unwrap(), Some(file));
        assert_eq!(store.leaf(&r, 0).unwrap(), None);
        assert_eq!(store.get(&[5; 32]).unwrap(), None);
        assert_eq!(store.prune(21).unwrap(), 1);
        assert_eq!(store.get(&leaves[0]).unwrap(), None);

        // Our own aux files count as stored, and aren't pruned.
        let aux = AuxStore::open(env.env()).unwrap();
        let ours = aux.set_file(env.env(), 4, b"init").unwrap();
        assert_eq!(store.get(&ours).unwrap(), Some(b"init".to_vec()));
        assert_eq!(store.prune(1000).unwrap(), 0);
        assert_eq!(store.get(&ours).unwrap(), Some(b"init".to_vec()));
        aux.set(env.env(), 4, &[0; 32]).unwrap();
        assert_eq!(store.get(&ours).unwrap(), None);
    }
}
