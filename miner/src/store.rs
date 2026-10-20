//! Block storage: the chain's headers and blocks, by height, in LMDB.
//!
//! The store holds one chain from genesis (height 0, a header only) to its
//! tip. Each block is kept serialized as on the wire ([`block::encode`]),
//! with its header also stored on its own for cheap header walks, an index
//! from header hash to height, and the chain's total work through each
//! height ([`header::work`] units). Callers write a block in the same LMDB
//! transaction as its state-tree changes, so the two always agree.
//!
//! Databases, integers big-endian:
//! - `chain_headers`: height (`u64`) to [`Header::to_bytes`].
//! - `chain_blocks`: height to the encoded block (absent for genesis).
//! - `chain_hashes`: header hash bytes to height.
//! - `chain_work`: height to total work (`u128`) of headers `1 ..= height`.
//! - `chain_meta`: free-form values by name.

use std::fmt;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::block::{self, Block, DecodeError};
use crate::hash;
use crate::header::Header;

#[derive(Debug)]
pub enum Error {
    Db(heed::Error),
    /// Stored data failed to decode.
    Corrupt,
    /// A block doesn't extend the tip, or genesis is set twice.
    NotNext,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Db(e) => write!(f, "database error: {e}"),
            Self::Corrupt => f.write_str("block store is corrupt"),
            Self::NotNext => f.write_str("block doesn't extend the stored tip"),
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
    blocks: Database<Bytes, Bytes>,
    hashes: Database<Bytes, Bytes>,
    work: Database<Bytes, Bytes>,
    meta: Database<Bytes, Bytes>,
}

impl ChainStore {
    /// Opens (creating if needed) the store in `env`.
    pub fn open(env: &Env) -> Result<Self, Error> {
        let mut txn = env.write_txn()?;
        let store = Self {
            headers: env.create_database(&mut txn, Some("chain_headers"))?,
            blocks: env.create_database(&mut txn, Some("chain_blocks"))?,
            hashes: env.create_database(&mut txn, Some("chain_hashes"))?,
            work: env.create_database(&mut txn, Some("chain_work"))?,
            meta: env.create_database(&mut txn, Some("chain_meta"))?,
        };
        txn.commit()?;
        Ok(store)
    }

    /// Starts the chain at `genesis`.
    pub fn init(&self, txn: &mut RwTxn, genesis: &Header) -> Result<(), Error> {
        if genesis.height != 0 || self.tip(txn)?.is_some() {
            return Err(Error::NotNext);
        }
        self.put_header(txn, genesis, 0)
    }

    fn put_header(&self, txn: &mut RwTxn, header: &Header, work: u128) -> Result<(), Error> {
        let h = header.height.to_be_bytes();
        self.headers.put(txn, &h, &header.to_bytes())?;
        self.hashes.put(txn, &hash::to_bytes(&header.hash()), &h)?;
        self.work.put(txn, &h, &work.to_be_bytes())?;
        Ok(())
    }

    /// Appends `block` under `header`, which must extend the tip; `work` is
    /// the total work through `header`. Doesn't validate the block.
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
        self.put_header(txn, header, work)?;
        self.blocks.put(
            txn,
            &header.height.to_be_bytes(),
            &block::encode(header, block),
        )?;
        Ok(())
    }

    /// The highest stored header, or `None` before [`ChainStore::init`].
    pub fn tip(&self, txn: &RoTxn) -> Result<Option<Header>, Error> {
        match self.headers.last(txn)? {
            Some((_, bytes)) => Ok(Some(Header::from_bytes(bytes).ok_or(Error::Corrupt)?)),
            None => Ok(None),
        }
    }

    pub fn genesis(&self, txn: &RoTxn) -> Result<Option<Header>, Error> {
        self.header(txn, 0)
    }

    pub fn header(&self, txn: &RoTxn, height: u64) -> Result<Option<Header>, Error> {
        match self.headers.get(txn, &height.to_be_bytes())? {
            Some(bytes) => Ok(Some(Header::from_bytes(bytes).ok_or(Error::Corrupt)?)),
            None => Ok(None),
        }
    }

    /// The block at `height` (none at genesis).
    pub fn block(&self, txn: &RoTxn, height: u64) -> Result<Option<Block>, Error> {
        match self.blocks.get(txn, &height.to_be_bytes())? {
            Some(bytes) => Ok(Some(block::decode(bytes)?.1)),
            None => Ok(None),
        }
    }

    /// The height of the header hashing to `hash`.
    pub fn height_of(&self, txn: &RoTxn, hash: &hash::Hash) -> Result<Option<u64>, Error> {
        match self.hashes.get(txn, &hash::to_bytes(hash))? {
            Some(bytes) => Ok(Some(u64::from_be_bytes(
                bytes.try_into().map_err(|_| Error::Corrupt)?,
            ))),
            None => Ok(None),
        }
    }

    /// Total work of headers `1 ..= height`.
    pub fn work(&self, txn: &RoTxn, height: u64) -> Result<Option<u128>, Error> {
        match self.work.get(txn, &height.to_be_bytes())? {
            Some(bytes) => Ok(Some(u128::from_be_bytes(
                bytes.try_into().map_err(|_| Error::Corrupt)?,
            ))),
            None => Ok(None),
        }
    }

    pub fn meta(&self, txn: &RoTxn, name: &str) -> Result<Option<Vec<u8>>, Error> {
        Ok(self.meta.get(txn, name.as_bytes())?.map(<[u8]>::to_vec))
    }

    pub fn set_meta(&self, txn: &mut RwTxn, name: &str, value: &[u8]) -> Result<(), Error> {
        Ok(self.meta.put(txn, name.as_bytes(), value)?)
    }
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
            assert!(matches!(store.init(&mut txn, &genesis), Err(Error::NotNext)));
            for h in 1..=3 {
                let next = header(h, headers.last().unwrap());
                store.push(&mut txn, &next, &block(h as u8), h as u128 * 100).unwrap();
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
