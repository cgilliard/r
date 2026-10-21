//! Block storage: every valid block known, by header hash, and the main
//! chain (the branch with the most work) by height, in LMDB.
//!
//! Each block is kept serialized as on the wire ([`block::encode`]), with
//! its header also stored on its own for cheap header walks, and the
//! chain's total work through it ([`header::work`] units). Blocks on the
//! main chain also have undo data ([`Undo`]), so the state tree can be
//! rolled back to a fork point. Callers write blocks in the same LMDB
//! transaction as their state-tree changes, so the two always agree
//! ([`crate::ledger`]).
//!
//! Databases, integers big-endian:
//! - `blk_headers`: header hash to [`Header::to_bytes`].
//! - `blk_bodies`: header hash to the encoded block (absent for genesis).
//! - `blk_work`: header hash to total work (`u128`) of headers `1 ..=` it.
//! - `blk_main`: height (`u64`) to the main chain's header hash.
//! - `blk_undo`: header hash to its [`Undo`], for main-chain blocks.
//! - `blk_invalid`: header hash of blocks known to be invalid.
//! - `blk_meta`: free-form values by name.
//!
//! A node that fast-synced starts at a base block instead of genesis: it
//! has the base and its parent (header only) on the main chain, nothing
//! between them and genesis, and can't reorganize below the base
//! ([`ChainStore::base`]).

use std::fmt;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::block::{self, Block, DecodeError};
use crate::hash::{self, Hash};
use crate::header::Header;
use crate::state::Undo;

#[derive(Debug)]
pub enum Error {
    Db(heed::Error),
    /// Stored data failed to decode.
    Corrupt,
    /// A block doesn't extend the main chain, or genesis is set twice.
    NotNext,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Corrupt => f.write_str("block store is corrupt"),
            Self::NotNext => f.write_str("block doesn't extend the main chain"),
        }
    }
}

impl std::error::Error for Error {}

impl From<heed::Error> for Error {
    fn from(e: heed::Error) -> Self {
        Self::Db(e)
    }
}

impl From<DecodeError> for Error {
    fn from(_: DecodeError) -> Self {
        Self::Corrupt
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChainStore {
    headers: Database<Bytes, Bytes>,
    bodies: Database<Bytes, Bytes>,
    work: Database<Bytes, Bytes>,
    main: Database<Bytes, Bytes>,
    undo: Database<Bytes, Bytes>,
    invalid: Database<Bytes, Bytes>,
    meta: Database<Bytes, Bytes>,
}

fn key(hash: &Hash) -> [u8; hash::HASH_BYTES] {
    hash::to_bytes(hash)
}

impl ChainStore {
    /// Opens (creating if needed) the store in `env`.
    pub fn open(env: &Env) -> Result<Self, Error> {
        let mut txn = env.write_txn()?;
        let store = Self {
            headers: env.create_database(&mut txn, Some("blk_headers"))?,
            bodies: env.create_database(&mut txn, Some("blk_bodies"))?,
            work: env.create_database(&mut txn, Some("blk_work"))?,
            main: env.create_database(&mut txn, Some("blk_main"))?,
            undo: env.create_database(&mut txn, Some("blk_undo"))?,
            invalid: env.create_database(&mut txn, Some("blk_invalid"))?,
            meta: env.create_database(&mut txn, Some("blk_meta"))?,
        };
        txn.commit()?;
        Ok(store)
    }

    /// Starts the chain at `genesis`.
    pub fn init(&self, txn: &mut RwTxn, genesis: &Header) -> Result<(), Error> {
        if genesis.height != 0 || self.tip(txn)?.is_some() {
            return Err(Error::NotNext);
        }
        let h = genesis.hash();
        self.headers.put(txn, &key(&h), &genesis.to_bytes())?;
        self.work.put(txn, &key(&h), &0u128.to_be_bytes())?;
        self.main.put(txn, &0u64.to_be_bytes(), &key(&h))?;
        Ok(())
    }

    /// Stores a block (not yet on the main chain) whose chain has `work`.
    pub fn add(
        &self,
        txn: &mut RwTxn,
        header: &Header,
        block: &Block,
        work: u128,
    ) -> Result<(), Error> {
        let h = key(&header.hash());
        self.headers.put(txn, &h, &header.to_bytes())?;
        self.bodies.put(txn, &h, &block::encode(header, block))?;
        self.work.put(txn, &h, &work.to_be_bytes())?;
        Ok(())
    }

    /// Stores a header without its block: a fast-synced node's base parent.
    pub fn add_header(&self, txn: &mut RwTxn, header: &Header, work: u128) -> Result<(), Error> {
        let h = key(&header.hash());
        self.headers.put(txn, &h, &header.to_bytes())?;
        self.work.put(txn, &h, &work.to_be_bytes())?;
        Ok(())
    }

    /// Starts the main chain of a fresh store (genesis only) at `base`
    /// (stored with its block) on `parent` (stored, header only): the
    /// fast-sync starting point. Blocks can't be reverted below it.
    pub fn set_base(&self, txn: &mut RwTxn, parent: &Header, base: &Header) -> Result<(), Error> {
        let tip = self.tip(txn)?.ok_or(Error::NotNext)?;
        if tip.height != 0
            || base.prev != parent.hash()
            || base.height != parent.height + 1
            || parent.height == 0
        {
            return Err(Error::NotNext);
        }
        self.main
            .put(txn, &parent.height.to_be_bytes(), &key(&parent.hash()))?;
        self.main
            .put(txn, &base.height.to_be_bytes(), &key(&base.hash()))?;
        self.set_meta(txn, BASE_KEY, &base.height.to_be_bytes())
    }

    /// The lowest block the main chain can be reverted to: the fast-sync
    /// base, or 0 (genesis) for a node with the whole chain.
    pub fn base(&self, txn: &RoTxn) -> Result<u64, Error> {
        match self.meta(txn, BASE_KEY)? {
            Some(b) => Ok(u64::from_be_bytes(
                b.as_slice().try_into().map_err(|_| Error::Corrupt)?,
            )),
            None => Ok(0),
        }
    }

    /// Puts a stored block on the main chain, above its parent, the tip.
    pub fn push_main(&self, txn: &mut RwTxn, header: &Header) -> Result<(), Error> {
        let tip = self.tip(txn)?.ok_or(Error::NotNext)?;
        if header.height != tip.height + 1 || header.prev != tip.hash() {
            return Err(Error::NotNext);
        }
        self.main
            .put(txn, &header.height.to_be_bytes(), &key(&header.hash()))?;
        Ok(())
    }

    /// Takes the tip off the main chain (it stays stored), returning it.
    pub fn pop_main(&self, txn: &mut RwTxn) -> Result<Header, Error> {
        let tip = self.tip(txn)?.ok_or(Error::Corrupt)?;
        if tip.height == 0 {
            return Err(Error::NotNext);
        }
        self.main.delete(txn, &tip.height.to_be_bytes())?;
        Ok(tip)
    }

    /// [`ChainStore::add`] then [`ChainStore::push_main`]; doesn't validate
    /// the block or touch the state tree.
    pub fn push(
        &self,
        txn: &mut RwTxn,
        header: &Header,
        block: &Block,
        work: u128,
    ) -> Result<(), Error> {
        let tip = self.tip(txn)?.ok_or(Error::NotNext)?;
        if header.height != tip.height + 1 || header.prev != tip.hash() {
            return Err(Error::NotNext);
        }
        self.add(txn, header, block, work)?;
        self.push_main(txn, header)
    }

    /// The main chain's tip, or `None` before [`ChainStore::init`].
    pub fn tip(&self, txn: &RoTxn) -> Result<Option<Header>, Error> {
        match self.main.last(txn)? {
            Some((_, h)) => self.header_by_hash(txn, &decode_hash(h)?),
            None => Ok(None),
        }
    }

    pub fn genesis(&self, txn: &RoTxn) -> Result<Option<Header>, Error> {
        self.header(txn, 0)
    }

    /// The main chain's hash at `height`.
    pub fn hash_at(&self, txn: &RoTxn, height: u64) -> Result<Option<Hash>, Error> {
        match self.main.get(txn, &height.to_be_bytes())? {
            Some(h) => Ok(Some(decode_hash(h)?)),
            None => Ok(None),
        }
    }

    /// The main chain's header at `height`.
    pub fn header(&self, txn: &RoTxn, height: u64) -> Result<Option<Header>, Error> {
        match self.hash_at(txn, height)? {
            Some(h) => self.header_by_hash(txn, &h),
            None => Ok(None),
        }
    }

    /// Any stored header.
    pub fn header_by_hash(&self, txn: &RoTxn, hash: &Hash) -> Result<Option<Header>, Error> {
        match self.headers.get(txn, &key(hash))? {
            Some(bytes) => Ok(Some(Header::from_bytes(bytes).ok_or(Error::Corrupt)?)),
            None => Ok(None),
        }
    }

    /// The main chain's block at `height` (none at genesis).
    pub fn block(&self, txn: &RoTxn, height: u64) -> Result<Option<Block>, Error> {
        match self.hash_at(txn, height)? {
            Some(h) => self.block_by_hash(txn, &h),
            None => Ok(None),
        }
    }

    /// Any stored block (none for genesis).
    pub fn block_by_hash(&self, txn: &RoTxn, hash: &Hash) -> Result<Option<Block>, Error> {
        match self.bodies.get(txn, &key(hash))? {
            Some(bytes) => Ok(Some(block::decode(bytes)?.1)),
            None => Ok(None),
        }
    }

    /// Any stored block as stored: [`block::encode`]d, header first.
    pub fn encoded_block(&self, txn: &RoTxn, hash: &Hash) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.bodies.get(txn, &key(hash))?.map(<[u8]>::to_vec))
    }

    /// The height of the header hashing to `hash`, if it's on the main
    /// chain.
    pub fn height_of(&self, txn: &RoTxn, hash: &Hash) -> Result<Option<u64>, Error> {
        let Some(header) = self.header_by_hash(txn, hash)? else {
            return Ok(None);
        };
        Ok((self.hash_at(txn, header.height)? == Some(*hash)).then_some(header.height))
    }

    /// Total work of the main chain's headers `1 ..= height`.
    pub fn work(&self, txn: &RoTxn, height: u64) -> Result<Option<u128>, Error> {
        match self.hash_at(txn, height)? {
            Some(h) => self.work_of(txn, &h),
            None => Ok(None),
        }
    }

    /// Total work of the chain ending at the stored header `hash`.
    pub fn work_of(&self, txn: &RoTxn, hash: &Hash) -> Result<Option<u128>, Error> {
        match self.work.get(txn, &key(hash))? {
            Some(bytes) => Ok(Some(u128::from_be_bytes(
                bytes.try_into().map_err(|_| Error::Corrupt)?,
            ))),
            None => Ok(None),
        }
    }

    pub fn put_undo(&self, txn: &mut RwTxn, hash: &Hash, undo: &Undo) -> Result<(), Error> {
        Ok(self.undo.put(txn, &key(hash), &undo.to_bytes())?)
    }

    /// Takes the undo data of block `hash`.
    pub fn take_undo(&self, txn: &mut RwTxn, hash: &Hash) -> Result<Undo, Error> {
        let undo = self
            .undo
            .get(txn, &key(hash))?
            .and_then(Undo::from_bytes)
            .ok_or(Error::Corrupt)?;
        self.undo.delete(txn, &key(hash))?;
        Ok(undo)
    }

    pub fn mark_invalid(&self, txn: &mut RwTxn, hash: &Hash) -> Result<(), Error> {
        Ok(self.invalid.put(txn, &key(hash), &[])?)
    }

    pub fn is_invalid(&self, txn: &RoTxn, hash: &Hash) -> Result<bool, Error> {
        Ok(self.invalid.get(txn, &key(hash))?.is_some())
    }

    pub fn meta(&self, txn: &RoTxn, name: &str) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.meta.get(txn, name.as_bytes())?.map(<[u8]>::to_vec))
    }

    pub fn set_meta(&self, txn: &mut RwTxn, name: &str, value: &[u8]) -> Result<(), Error> {
        Ok(self.meta.put(txn, name.as_bytes(), value)?)
    }
}

const BASE_KEY: &str = "base";

fn decode_hash(bytes: &[u8]) -> Result<Hash, Error> {
    hash::from_bytes(bytes.try_into().map_err(|_| Error::Corrupt)?).ok_or(Error::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::BabyBear;
    use crate::db::test::TempEnv;
    use crate::hash::HASH_LEN;
    use crate::output::{Commitment, NONCE_LEN, OutputRecord};
    use crate::stark::StarkProof;

    fn header(height: u64, prev: &Header) -> Header {
        Header {
            height,
            prev: prev.hash(),
            timestamp: prev.timestamp + 600,
            nonce: height * 7,
            ..*prev
        }
    }

    fn block(n: u8) -> Block {
        let mut c = [BabyBear::new(n as u32 + 1); HASH_LEN];
        c[0] = BabyBear::new(9);
        Block {
            inputs: vec![Commitment(hash::to_bytes(&c))],
            outputs: vec![OutputRecord {
                commitment: Commitment(hash::to_bytes(&[BabyBear::new(n as u32 + 3); HASH_LEN])),
                nonce: [n; NONCE_LEN],
            }],
            proof: StarkProof::from_elements(&tiny_proof()).unwrap(),
        }
    }

    /// The smallest well-formed proof encoding.
    fn tiny_proof() -> Vec<BabyBear> {
        let mut e = vec![BabyBear::new(3)];
        e.extend([BabyBear::ONE; HASH_LEN]);
        e.push(BabyBear::ZERO); // no aux commitment
        e.extend([BabyBear::ONE; HASH_LEN]);
        e.extend([BabyBear::ZERO; 7]); // empty OOD vectors
        e.push(BabyBear::ZERO); // FRI commitments
        e.push(BabyBear::ZERO); // final poly
        e.push(BabyBear::ONE); // pow witness
        e.push(BabyBear::ZERO); // queries
        e.push(BabyBear::ZERO); // openings
        e
    }

    #[test]
    fn stores_and_reloads_a_chain() {
        let mut env = TempEnv::new();
        let genesis = Header {
            timestamp: 1000,
            ..Header::default()
        };
        let mut headers = vec![genesis];
        {
            let store = ChainStore::open(env.env()).unwrap();
            let mut txn = env.env().write_txn().unwrap();
            assert!(store.tip(&txn).unwrap().is_none());
            store.init(&mut txn, &genesis).unwrap();
            assert!(matches!(
                store.init(&mut txn, &genesis),
                Err(Error::NotNext)
            ));
            for h in 1..=3 {
                let next = header(h, headers.last().unwrap());
                store
                    .push(&mut txn, &next, &block(h as u8), h as u128 * 100)
                    .unwrap();
                headers.push(next);
            }
            // Not the tip's child.
            let stray = header(5, &headers[1]);
            assert!(matches!(
                store.push(&mut txn, &stray, &block(9), 1),
                Err(Error::NotNext)
            ));
            store.set_meta(&mut txn, "seed", b"abc").unwrap();
            txn.commit().unwrap();
        }
        env.reopen();
        let store = ChainStore::open(env.env()).unwrap();
        let txn = env.env().read_txn().unwrap();
        assert_eq!(store.tip(&txn).unwrap(), Some(headers[3]));
        assert_eq!(store.genesis(&txn).unwrap(), Some(genesis));
        for (h, header) in headers.iter().enumerate() {
            let h = h as u64;
            assert_eq!(store.header(&txn, h).unwrap(), Some(*header));
            assert_eq!(store.height_of(&txn, &header.hash()).unwrap(), Some(h));
            assert_eq!(store.work(&txn, h).unwrap(), Some(h as u128 * 100));
        }
        assert!(store.block(&txn, 0).unwrap().is_none());
        let b = store.block(&txn, 2).unwrap().unwrap();
        assert_eq!((b.inputs, b.outputs), (block(2).inputs, block(2).outputs));
        assert_eq!(store.meta(&txn, "seed").unwrap(), Some(b"abc".to_vec()));
        assert_eq!(store.meta(&txn, "other").unwrap(), None);
    }
}
