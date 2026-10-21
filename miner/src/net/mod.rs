//! Networking: the UDP wire format ([`wire`]), the node ([`node`]), and the
//! ledger it serves and follows ([`LedgerBackend`]).

pub mod node;
pub mod wire;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::block::{self, MAX_BLOCK_SIZE};
use crate::hash::Hash;
use crate::header::{self, HEADER_BYTES, Header};
use crate::ledger::{Added, Ledger, LedgerError, Snapshot};
use crate::state::SyncPut;
use node::{Backend, Rejected};
use wire::Tip;

/// The stored chain: serves its headers and bodies, and adds blocks through
/// the ledger, which verifies them and picks the main chain.
pub struct LedgerBackend(pub Arc<Ledger>);

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Backend for LedgerBackend {
    fn tip(&self) -> Tip {
        let (tip, work) = self.0.tip().expect("a stored chain");
        Tip {
            height: tip.height,
            hash: tip.hash(),
            work,
        }
    }

    fn header_at(&self, height: u64) -> Option<Header> {
        let txn = self.0.env.read_txn().ok()?;
        self.0.store.header(&txn, height).ok().flatten()
    }

    fn height_of(&self, hash: &Hash) -> Option<u64> {
        let txn = self.0.env.read_txn().ok()?;
        self.0.store.height_of(&txn, hash).ok().flatten()
    }

    fn header_by_hash(&self, hash: &Hash) -> Option<Header> {
        let txn = self.0.env.read_txn().ok()?;
        self.0.store.header_by_hash(&txn, hash).ok().flatten()
    }

    fn body(&self, hash: &Hash) -> Option<Vec<u8>> {
        let txn = self.0.env.read_txn().ok()?;
        let bytes = self.0.store.encoded_block(&txn, hash).ok()??;
        Some(bytes[HEADER_BYTES..].to_vec())
    }

    fn check_header(&self, parent: &Header, header: &Header) -> Result<(), String> {
        let l = &self.0;
        let txn = l.env.read_txn().map_err(|e| e.to_string())?;
        for h in [header.hash(), header.prev] {
            if l.store.is_invalid(&txn, &h).map_err(|e| e.to_string())? {
                return Err("known to be invalid".into());
            }
        }
        header::check_new(&l.consensus, l.data, parent, header, now()).map_err(|e| e.to_string())
    }

    fn accept(&self, header: &Header, body: &[u8]) -> Result<Added, Rejected> {
        let block = decode(body)?;
        self.0.add(header, &block).map_err(rejected)
    }

    fn snapshot(&self, height: u64) -> Option<Snapshot> {
        self.0.snapshot(height).ok().flatten()
    }

    fn state_nodes(&self, base: &Hash, level: u8, index: u64) -> Option<Vec<u8>> {
        self.0.state_nodes(base, level, index).ok().flatten()
    }

    fn check_base(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected> {
        let block = decode(body)?;
        self.0
            .verify_base(parent, parent_work, base, &block)
            .map_err(rejected)
    }

    fn begin_state(&self, base: &Header) -> Result<(), String> {
        self.0.begin_state(base).map_err(|e| e.to_string())
    }

    fn state_progress(&self) -> Option<(u64, Hash)> {
        self.0.state_progress().ok().flatten()
    }

    fn state_wanted(&self, n: usize, skip: &HashSet<(u8, u64)>) -> Vec<(u8, u64)> {
        self.0.state_wanted(n, skip).unwrap_or_default()
    }

    fn put_state(&self, replies: &[node::StateReply]) -> Result<Vec<SyncPut>, String> {
        self.0.put_state(replies).map_err(|e| e.to_string())
    }

    fn state_done(&self) -> bool {
        self.0.state_done().unwrap_or(false)
    }

    fn install(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected> {
        let block = decode(body)?;
        self.0
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
