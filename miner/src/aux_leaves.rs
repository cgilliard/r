//! The miner's aux leaves: 16 values of 32 bytes whose Merkle root our
//! mined headers carry as their [`crate::header::Header::aux_hash`].
//! Consensus ignores it; layer 2 gives the leaves meaning (such as a file's
//! SHA-256).
//!
//! The root is a SHA-256 Merkle tree (each node `SHA-256(left || right)`)
//! over the leaves in order, so anyone can check a leaf against a header
//! with standard tools; if every leaf is zero (none set), the root is zero.
//! The leaves are kept in LMDB (database `aux`, keyed by leaf number; a
//! zero leaf isn't stored), so they survive restarts, with the file a leaf
//! was set from (keyed by [`AUX_LEAVES`] plus its number), so the miner
//! can serve it to peers ([`crate::preimages`]).

use heed::types::{Bytes, U8};
use heed::{Database, Env};

use crate::mnemonic::sha256;

/// Leaves of the tree.
pub const AUX_LEAVES: usize = 16;

pub type Leaf = [u8; 32];

/// The aux hash for `leaves`: their SHA-256 Merkle root, or zero if they
/// are all zero.
pub fn root(leaves: &[Leaf; AUX_LEAVES]) -> Leaf {
    if leaves.iter().all(|l| *l == [0; 32]) {
        return [0; 32];
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        level = level
            .chunks_exact(2)
            .map(|p| {
                let mut both = [0u8; 64];
                both[..32].copy_from_slice(&p[0]);
                both[32..].copy_from_slice(&p[1]);
                sha256(&both)
            })
            .collect();
    }
    level[0]
}

#[derive(Clone, Copy, Debug)]
pub struct AuxStore {
    db: Database<U8, Bytes>,
}

impl AuxStore {
    pub fn open(env: &Env) -> heed::Result<Self> {
        let mut txn = env.write_txn()?;
        let db = env.create_database(&mut txn, Some("aux"))?;
        txn.commit()?;
        Ok(Self { db })
    }

    /// The leaves (zero where unset).
    pub fn leaves(&self, env: &Env) -> heed::Result<[Leaf; AUX_LEAVES]> {
        let txn = env.read_txn()?;
        let mut out = [[0u8; 32]; AUX_LEAVES];
        for entry in self.db.iter(&txn)? {
            let (n, v) = entry?;
            if let (Some(leaf), Ok(v)) = (out.get_mut(n as usize), <Leaf>::try_from(v)) {
                *leaf = v;
            }
        }
        Ok(out)
    }

    /// The aux hash for the leaves now: [`root`].
    pub fn root(&self, env: &Env) -> heed::Result<Leaf> {
        Ok(root(&self.leaves(env)?))
    }

    /// The files leaves were set from.
    pub fn files(&self, env: &Env) -> heed::Result<Vec<Vec<u8>>> {
        let txn = env.read_txn()?;
        let mut out = Vec::new();
        for n in 0..AUX_LEAVES {
            if let Some(f) = self.db.get(&txn, &((AUX_LEAVES + n) as u8))? {
                out.push(f.to_vec());
            }
        }
        Ok(out)
    }

    /// The file of ours whose SHA-256 is `hash` (a leaf set from it).
    pub fn file_for(&self, env: &Env, hash: &Leaf) -> heed::Result<Option<Vec<u8>>> {
        if *hash == [0; 32] {
            return Ok(None);
        }
        let txn = env.read_txn()?;
        for n in 0..AUX_LEAVES {
            if self.db.get(&txn, &(n as u8))? == Some(&hash[..])
                && let Some(f) = self.db.get(&txn, &((AUX_LEAVES + n) as u8))?
            {
                return Ok(Some(f.to_vec()));
            }
        }
        Ok(None)
    }

    /// Sets leaf `n` (below [`AUX_LEAVES`]) to `value`.
    ///
    /// # Panics
    /// If `n` is out of range.
    pub fn set(&self, env: &Env, n: usize, value: &Leaf) -> heed::Result<()> {
        self.put(env, n, value, None)
    }

    /// Sets leaf `n` (below [`AUX_LEAVES`]) to the SHA-256 of `file`,
    /// keeping the file: the leaf.
    ///
    /// # Panics
    /// If `n` is out of range.
    pub fn set_file(&self, env: &Env, n: usize, file: &[u8]) -> heed::Result<Leaf> {
        let leaf = sha256(file);
        self.put(env, n, &leaf, Some(file))?;
        Ok(leaf)
    }

    fn put(&self, env: &Env, n: usize, value: &Leaf, file: Option<&[u8]>) -> heed::Result<()> {
        assert!(n < AUX_LEAVES, "aux leaves are 0 to {}", AUX_LEAVES - 1);
        let mut txn = env.write_txn()?;
        if *value == [0; 32] {
            self.db.delete(&mut txn, &(n as u8))?;
        } else {
            self.db.put(&mut txn, &(n as u8), value)?;
        }
        let file_key = (AUX_LEAVES + n) as u8;
        match file {
            Some(f) => self.db.put(&mut txn, &file_key, f)?,
            None => {
                self.db.delete(&mut txn, &file_key)?;
            }
        }
        txn.commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;

    #[test]
    fn the_root_is_zero_until_a_leaf_is_set() {
        let mut leaves = [[0u8; 32]; AUX_LEAVES];
        assert_eq!(root(&leaves), [0; 32]);
        leaves[5] = [7; 32];
        // By hand: up the tree from leaf 5 (binary 0101: a right child, then
        // left, right, left).
        let h = |a: &Leaf, b: &Leaf| sha256(&[&a[..], &b[..]].concat());
        let zero = [0u8; 32];
        let z1 = h(&zero, &zero);
        let z2 = h(&z1, &z1);
        let z3 = h(&z2, &z2);
        let n1 = h(&zero, &[7; 32]);
        let n2 = h(&n1, &z1);
        let n3 = h(&z2, &n2);
        assert_eq!(root(&leaves), h(&n3, &z3));
    }

    #[test]
    fn leaves_are_kept_and_cleared() {
        let env = TempEnv::new();
        let store = AuxStore::open(env.env()).unwrap();
        assert_eq!(store.root(env.env()).unwrap(), [0; 32]);
        store.set(env.env(), 15, &[9; 32]).unwrap();
        store.set(env.env(), 0, &[1; 32]).unwrap();
        let again = AuxStore::open(env.env()).unwrap();
        let leaves = again.leaves(env.env()).unwrap();
        assert_eq!(
            (leaves[0], leaves[15], leaves[3]),
            ([1; 32], [9; 32], [0; 32])
        );
        assert_eq!(again.root(env.env()).unwrap(), root(&leaves));
        assert!(again.files(env.env()).unwrap().is_empty());
        let leaf = again.set_file(env.env(), 2, b"a file").unwrap();
        assert_eq!(leaf, sha256(b"a file"));
        assert_eq!(again.leaves(env.env()).unwrap()[2], leaf);
        assert_eq!(again.files(env.env()).unwrap(), vec![b"a file".to_vec()]);
        again.set(env.env(), 2, &[3; 32]).unwrap();
        assert!(again.files(env.env()).unwrap().is_empty());
        again.set(env.env(), 2, &[0; 32]).unwrap();
        again.set(env.env(), 0, &[0; 32]).unwrap();
        again.set(env.env(), 15, &[0; 32]).unwrap();
        assert_eq!(again.root(env.env()).unwrap(), [0; 32]);
    }
}
