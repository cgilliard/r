//! Networking: the UDP wire format ([`wire`]), the node ([`node`]), and the
//! ledger and mempool it serves and follows ([`LedgerBackend`]).

pub mod node;
pub mod wire;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::block::{self, MAX_BLOCK_SIZE};
use crate::hash::Hash;
use crate::header::{self, HEADER_BYTES, Header};
use crate::ledger::{Added, Ledger, LedgerError, Snapshot};
use crate::mempool::{Mempool, Refused};
use crate::state::SyncPut;
use node::{Backend, Rejected};
use wire::Tip;

/// The stored chain and the mempool: serves headers, bodies, state and
/// transactions, and adds blocks through the ledger (which verifies them and
/// picks the main chain) and transactions through the mempool.
pub struct LedgerBackend {
    pub ledger: Arc<Ledger>,
    pub mempool: Arc<Mempool>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Backend for LedgerBackend {
    fn tip(&self) -> Tip {
        let (tip, work) = self.ledger.tip().expect("a stored chain");
        Tip {
            height: tip.height,
            hash: tip.hash(),
            work,
            state_from: if tip.height == 0 {
                0
            } else {
                self.ledger.servable_from().unwrap_or(0)
            },
        }
    }

    fn header_at(&self, height: u64) -> Option<Header> {
        let txn = self.ledger.env.read_txn().ok()?;
        self.ledger.store.header(&txn, height).ok().flatten()
    }

    fn height_of(&self, hash: &Hash) -> Option<u64> {
        let txn = self.ledger.env.read_txn().ok()?;
        self.ledger.store.height_of(&txn, hash).ok().flatten()
    }

    fn header_by_hash(&self, hash: &Hash) -> Option<Header> {
        let txn = self.ledger.env.read_txn().ok()?;
        self.ledger.store.header_by_hash(&txn, hash).ok().flatten()
    }

    fn body(&self, hash: &Hash) -> Option<Vec<u8>> {
        let txn = self.ledger.env.read_txn().ok()?;
        let bytes = self.ledger.store.encoded_block(&txn, hash).ok()??;
        Some(bytes[HEADER_BYTES..].to_vec())
    }

    fn check_header(&self, parent: &Header, header: &Header) -> Result<(), String> {
        let l = &self.ledger;
        let txn = l.env.read_txn().map_err(|e| e.to_string())?;
        for h in [header.hash(), header.prev] {
            if l.store.is_invalid(&txn, &h).map_err(|e| e.to_string())? {
                return Err("known to be invalid".into());
            }
        }
        // A parent still queued was checked when it was.
        if l.store
            .header_by_hash(&txn, &parent.hash())
            .map_err(|e| e.to_string())?
            .is_some()
        {
            l.check_fork(&txn, parent).map_err(|e| e.to_string())?;
        }
        header::check_new(&l.consensus, l.data, parent, header, now()).map_err(|e| e.to_string())
    }

    fn accept(&self, header: &Header, body: &[u8]) -> Result<Added, Rejected> {
        let block = decode(body)?;
        self.ledger.add(header, &block).map_err(rejected)
    }

    fn snapshot(&self, height: u64) -> Option<Snapshot> {
        self.ledger.snapshot(height).ok().flatten()
    }

    fn state_nodes(&self, base: &Hash, level: u8, index: u64) -> Option<Vec<u8>> {
        self.ledger.state_nodes(base, level, index).ok().flatten()
    }

    fn check_base(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected> {
        let block = decode(body)?;
        self.ledger
            .verify_base(parent, parent_work, base, &block)
            .map_err(rejected)
    }

    fn begin_state(&self, base: &Header) -> Result<(), String> {
        self.ledger.begin_state(base).map_err(|e| e.to_string())
    }

    fn state_progress(&self) -> Option<(u64, Hash)> {
        self.ledger.state_progress().ok().flatten()
    }

    fn state_wanted(&self, n: usize, skip: &HashSet<(u8, u64)>) -> Vec<(u8, u64)> {
        self.ledger.state_wanted(n, skip).unwrap_or_default()
    }

    fn put_state(&self, replies: &[node::StateReply]) -> Result<Vec<SyncPut>, String> {
        self.ledger.put_state(replies).map_err(|e| e.to_string())
    }

    fn state_done(&self) -> bool {
        self.ledger.state_done().unwrap_or(false)
    }

    fn tx(&self, id: &Hash) -> Option<Vec<u8>> {
        self.mempool.get(id)
    }

    fn knows_tx(&self, id: &Hash) -> bool {
        self.mempool.knows(id)
    }

    fn add_tx(&self, bytes: &[u8]) -> Result<bool, Rejected> {
        match self.mempool.add(&*self.ledger, bytes) {
            Ok((_, new)) => Ok(new),
            Err(Refused::Invalid(e)) => Err(Rejected::Invalid(e)),
            Err(Refused::Unfit(e)) => Err(Rejected::Failed(e)),
        }
    }

    fn tx_ids(&self) -> Vec<Hash> {
        self.mempool.ids()
    }

    fn install(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected> {
        let block = decode(body)?;
        self.ledger
            .install(parent, parent_work, base, &block)
            .map_err(rejected)
    }
}

/// A body from a peer, decoded.
fn decode(body: &[u8]) -> Result<block::Block, Rejected> {
    if body.len() + HEADER_BYTES > MAX_BLOCK_SIZE {
        return Err(Rejected::Invalid("oversize block".into()));
    }
    block::decode_body(body).map_err(|e| Rejected::Invalid(e.to_string()))
}

fn rejected(e: LedgerError) -> Rejected {
    if e.is_invalid() {
        Rejected::Invalid(e.to_string())
    } else {
        Rejected::Failed(e.to_string())
    }
}
