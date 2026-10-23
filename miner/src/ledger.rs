//! The ledger: the stored blocks, the state tree, and the rule that picks
//! the main chain among them.
//!
//! A block is checked when it arrives, without the state: its header against
//! its parent (link, time, ASERT target, PoW) and its body and chain proof
//! against its header ([`Chain::verify_block`]); the proof covers the whole
//! history, so a branch's blocks can be checked before the branch is
//! applied. Valid blocks are stored whichever branch they're on, as long
//! as the branch leaves the main chain at most [`Ledger::snap_sync_limit`]
//! blocks below the tip ([`Ledger::check_fork`]): no deeper reorg is
//! followed (a node that needs one resyncs), and side blocks forking far
//! back, where the work was cheap, are never stored.
//!
//! The main chain is the branch with the most total work; on a tie the one
//! seen first stays. When a block makes another branch heaviest, the ledger
//! reorganizes in one LMDB transaction: it reverts main-chain blocks down
//! to the fork point (with their undo data) and applies the branch's blocks
//! up to the new one. Applying checks each block's lists against its
//! header's state; if one fails, nothing changes, and it and its
//! descendants are marked invalid.
//!
//! **Fast sync.** A new node can start from a recent block instead of
//! genesis ([`Ledger::install`]): it checks that block (the base) with its
//! header and chain proof, which covers all history before it, downloads
//! the state tree at the base node by node from peers, each part checked
//! against the base header's state root on arrival
//! ([`Ledger::begin_state`], [`Ledger::put_state`]), and then follows the
//! chain from the base; it can't reorganize below it. A node serves the
//! state at any of its last [`Ledger::snap_sync_limit`] blocks
//! ([`Ledger::snapshot`], [`Ledger::state_nodes`]), keeping that much of
//! the tree's history.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use heed::{Env, RwTxn};

use crate::block::{self, Block, BlockError};
use crate::chain::{Chain, ChainVerifier};
use crate::hash::{self, Hash};
use crate::header::{self, Consensus, Header, HeaderError};
use crate::state::{self, StateTree, SyncPut};
use crate::store::{self, ChainStore};

/// What adding a block did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Added {
    /// Already stored.
    Known,
    /// Extended the main chain.
    Extended,
    /// Made its branch the main chain, reverting `reverted` blocks.
    Reorg { reverted: u64, applied: u64 },
    /// Stored on a branch with no more work than the main chain.
    Side,
}

#[derive(Debug)]
pub enum LedgerError {
    /// The parent isn't stored (fetch it first).
    UnknownParent,
    /// The block, or an ancestor, is known to be invalid.
    KnownInvalid,
    /// The branch leaves the main chain below the fast-sync base.
    BeforeBase,
    /// The branch leaves the main chain more than
    /// [`Ledger::snap_sync_limit`] blocks below the tip.
    TooDeep,
    /// A downloaded state tree doesn't match its header.
    BadSnapshot,
    /// Fast sync needs a ledger with only genesis.
    NotFresh,
    Header(HeaderError),
    Block(BlockError),
    Store(store::Error),
    State(state::Error),
    Db(heed::Error),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownParent => f.write_str("unknown parent"),
            Self::KnownInvalid => f.write_str("known to be invalid"),
            Self::BeforeBase => f.write_str("forks below this node's fast-sync base"),
            Self::TooDeep => {
                f.write_str("forks too far below the tip (beyond the snap-sync limit)")
            }
            Self::BadSnapshot => f.write_str("the state snapshot doesn't match its header"),
            Self::NotFresh => f.write_str("the chain isn't new (only genesis)"),
            Self::Header(e) => write!(f, "header: {e}"),
            Self::Block(e) => write!(f, "block: {e}"),
            Self::Store(e) => write!(f, "{e}"),
            Self::State(e) => write!(f, "{e}"),
            Self::Db(e) => write!(f, "database error: {e}"),
        }
    }
}

impl std::error::Error for LedgerError {}

impl From<store::Error> for LedgerError {
    fn from(e: store::Error) -> Self {
        Self::Store(e)
    }
}

impl From<state::Error> for LedgerError {
    fn from(e: state::Error) -> Self {
        Self::State(e)
    }
}

impl From<heed::Error> for LedgerError {
    fn from(e: heed::Error) -> Self {
        Self::Db(e)
    }
}

impl LedgerError {
    /// Whether the block itself is at fault (rather than our state).
    pub fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::Header(_) | Self::Block(_) | Self::KnownInvalid | Self::BadSnapshot
        )
    }
}

/// Checks a block against its header without the state: size, canonical
/// lists, body hash and chain proof ([`Chain::verify_block`]).
pub trait Verifier: Send + Sync {
    fn verify_block(
        &self,
        header: &Header,
        block: &Block,
        work_before: u128,
    ) -> Result<(), BlockError>;
}

impl Verifier for Chain {
    fn verify_block(
        &self,
        header: &Header,
        block: &Block,
        work_before: u128,
    ) -> Result<(), BlockError> {
        Chain::verify_block(self, header, block, work_before)
    }
}

impl Verifier for crate::chain::NoProofs {
    fn verify_block(&self, header: &Header, block: &Block, _: u128) -> Result<(), BlockError> {
        crate::chain::check_body(header, block)
    }
}

impl Verifier for ChainVerifier {
    fn verify_block(
        &self,
        header: &Header,
        block: &Block,
        work_before: u128,
    ) -> Result<(), BlockError> {
        ChainVerifier::verify_block(self, header, block, work_before)
    }
}

/// A main-chain block a fast-syncing node can start from (the base), with
/// its parent and the parent's total work; its state is served node by
/// node ([`Ledger::state_nodes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub parent: Header,
    pub base: Header,
    pub parent_work: u128,
}

pub struct Ledger {
    pub env: Env,
    pub store: ChainStore,
    pub tree: StateTree,
    pub consensus: Consensus,
    pub verifier: Arc<dyn Verifier>,
    /// Blocks below the tip whose state is kept to serve fast-syncing
    /// nodes, and the deepest reorg followed (undo data is kept for them).
    pub snap_sync_limit: u64,
    /// Blocks below the tip whose bodies are kept (at least
    /// `snap_sync_limit`), for peers catching up and the wallet; 0 keeps
    /// every block.
    pub keep_blocks: u64,
}

/// Blocks of state history kept by default: a fast-syncing node downloads
/// the state [`crate::args::DEFAULT_SYNC_DEPTH`] (100) blocks down, and
/// the rest is its time to finish while the chain moves on (about 16 hours).
pub const DEFAULT_SNAP_SYNC_LIMIT: u64 = 200;

/// Blocks kept by default: about a week, so a node offline that long
/// catches up from peers' blocks.
pub const DEFAULT_KEEP_BLOCKS: u64 = 1000;

/// The tag of a state download: the base's height and hash.
fn sync_tag(base: &Header) -> Vec<u8> {
    let mut tag = base.height.to_be_bytes().to_vec();
    tag.extend(hash::to_bytes(&base.hash()));
    tag
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Ledger {
    /// The main chain's tip and its total work.
    pub fn tip(&self) -> Result<(Header, u128), LedgerError> {
        let txn = self.env.read_txn()?;
        let tip = self.store.tip(&txn)?.ok_or(store::Error::Corrupt)?;
        let work = self
            .store
            .work_of(&txn, &tip.hash())?
            .ok_or(store::Error::Corrupt)?;
        Ok((tip, work))
    }

    /// Checks `header` against its stored parent (as a new header: not too
    /// far in the future).
    pub fn check_header(&self, header: &Header) -> Result<Header, LedgerError> {
        let txn = self.env.read_txn()?;
        if self.store.is_invalid(&txn, &header.hash())?
            || self.store.is_invalid(&txn, &header.prev)?
        {
            return Err(LedgerError::KnownInvalid);
        }
        let parent = self
            .store
            .header_by_hash(&txn, &header.prev)?
            .ok_or(LedgerError::UnknownParent)?;
        self.check_fork(&txn, &parent)?;
        header::check_new(&self.consensus, &parent, header, now()).map_err(LedgerError::Header)?;
        Ok(parent)
    }

    /// Refuses a block on `parent` (stored) whose branch leaves the main
    /// chain more than [`Ledger::snap_sync_limit`] blocks below the tip, or
    /// below the fast-sync base. Side blocks cost their work at the
    /// difficulty where they fork, so this keeps cheap ones, forking far
    /// back where the work was low, out; a reorg that deep isn't followed
    /// anyway (a node that needs one resyncs).
    pub fn check_fork(&self, txn: &heed::RoTxn, parent: &Header) -> Result<(), LedgerError> {
        let tip = self.store.tip(txn)?.ok_or(store::Error::Corrupt)?;
        let base = self.store.base(txn)?;
        // Undo data below `undo_from` is gone (the tip may have come down
        // since it was pruned).
        let floor = tip
            .height
            .saturating_sub(self.snap_sync_limit)
            .max(base)
            .max(self.store.undo_from(txn)?);
        let mut h = *parent;
        loop {
            if self.store.height_of(txn, &h.hash())?.is_some() {
                // On the main chain: the fork point.
                return if h.height < floor {
                    Err(if h.height < base {
                        LedgerError::BeforeBase
                    } else {
                        LedgerError::TooDeep
                    })
                } else {
                    Ok(())
                };
            }
            if h.height <= floor {
                // The fork point is lower still.
                return Err(if h.height <= base {
                    LedgerError::BeforeBase
                } else {
                    LedgerError::TooDeep
                });
            }
            h = self
                .store
                .header_by_hash(txn, &h.prev)?
                .ok_or(store::Error::Corrupt)?;
        }
    }

    /// Checks and stores `block` under `header`, and makes it (and its
    /// branch) the main chain if that has the most work.
    pub fn add(&self, header: &Header, block: &Block) -> Result<Added, LedgerError> {
        let hash = header.hash();
        {
            let txn = self.env.read_txn()?;
            if self.store.header_by_hash(&txn, &hash)?.is_some() {
                return Ok(Added::Known);
            }
        }
        let parent = self.check_header(header)?;
        let mut txn = self.env.write_txn()?;
        let work_before = self
            .store
            .work_of(&txn, &parent.hash())?
            .ok_or(store::Error::Corrupt)?;
        if let Err(e) = self.verifier.verify_block(header, block, work_before) {
            self.store.mark_invalid(&mut txn, &hash)?;
            txn.commit()?;
            return Err(LedgerError::Block(e));
        }
        let work = work_before + header::work(self.consensus.log_target(&parent));
        self.store.add(&mut txn, header, block, work)?;

        let tip = self.store.tip(&txn)?.ok_or(store::Error::Corrupt)?;
        let tip_work = self
            .store
            .work_of(&txn, &tip.hash())?
            .ok_or(store::Error::Corrupt)?;
        if work <= tip_work {
            txn.commit()?;
            return Ok(Added::Side);
        }
        match self.switch_to(&mut txn, header) {
            Ok(added) => {
                txn.commit()?;
                Ok(added)
            }
            Err((bad, e)) => {
                // Nothing is kept but the verdict: the failing block and its
                // descendants on this branch are invalid.
                txn.abort();
                if e.is_invalid() {
                    let mut txn = self.env.write_txn()?;
                    for h in bad {
                        self.store.mark_invalid(&mut txn, &h)?;
                    }
                    txn.commit()?;
                }
                Err(e)
            }
        }
    }

    /// Starts the state tree's history at the tip if it has none yet (a
    /// ledger from before history was kept); call once after opening.
    pub fn start_history(&self) -> Result<(), LedgerError> {
        let mut txn = self.env.write_txn()?;
        let tip = self.store.tip(&txn)?.ok_or(store::Error::Corrupt)?;
        self.tree.init_history(&mut txn, tip.height)?;
        txn.commit()?;
        Ok(())
    }

    /// The lowest height whose state this ledger can serve.
    pub fn servable_from(&self) -> Result<u64, LedgerError> {
        let txn = self.env.read_txn()?;
        self.servable_from_in(&txn)
    }

    fn servable_from_in(&self, txn: &heed::RoTxn) -> Result<u64, LedgerError> {
        let tip = self.store.tip(txn)?.ok_or(store::Error::Corrupt)?;
        let history = self.tree.history_from(txn)?.unwrap_or(tip.height);
        Ok(history.max(self.store.base(txn)?).max(1))
    }

    /// Whether the state after main-chain block `height` can be served.
    fn servable(&self, txn: &heed::RoTxn, height: u64) -> Result<bool, LedgerError> {
        let tip = self.store.tip(txn)?.ok_or(store::Error::Corrupt)?;
        Ok(height >= self.servable_from_in(txn)? && height <= tip.height)
    }

    /// The base at main-chain height `height`, if this node can serve the
    /// state there: from its fast-sync base and its kept history up.
    pub fn snapshot(&self, height: u64) -> Result<Option<Snapshot>, LedgerError> {
        let txn = self.env.read_txn()?;
        if !self.servable(&txn, height)? {
            return Ok(None);
        }
        let corrupt = || LedgerError::Store(store::Error::Corrupt);
        Ok(Some(Snapshot {
            base: self.store.header(&txn, height)?.ok_or_else(corrupt)?,
            parent: self.store.header(&txn, height - 1)?.ok_or_else(corrupt)?,
            parent_work: self.store.work(&txn, height - 1)?.ok_or_else(corrupt)?,
        }))
    }

    /// A state-sync reply ([`StateTree::sync_reply`]): node (`level`,
    /// `index`) of the state tree after main-chain block `base`, if it can
    /// be served.
    pub fn state_nodes(
        &self,
        base: &Hash,
        level: u8,
        index: u64,
    ) -> Result<Option<Vec<u8>>, LedgerError> {
        let txn = self.env.read_txn()?;
        let Some(height) = self.store.height_of(&txn, base)? else {
            return Ok(None);
        };
        if !self.servable(&txn, height)? {
            return Ok(None);
        }
        Ok(self.tree.sync_reply(&txn, height, level as usize, index)?)
    }

    /// Starts (or resumes) downloading the state at fast-sync base `base`
    /// (checked already) into a new ledger.
    pub fn begin_state(&self, base: &Header) -> Result<(), LedgerError> {
        let mut txn = self.env.write_txn()?;
        if self.store.tip(&txn)?.ok_or(store::Error::Corrupt)?.height != 0 {
            return Err(LedgerError::NotFresh);
        }
        self.tree
            .sync_begin(&mut txn, &sync_tag(base), &base.state.root)?;
        txn.commit()?;
        Ok(())
    }

    /// The base (height and hash) of a state download in progress.
    pub fn state_progress(&self) -> Result<Option<(u64, Hash)>, LedgerError> {
        let txn = self.env.read_txn()?;
        Ok(self.tree.sync_tag(&txn)?.and_then(|tag| {
            let (height, hash) = tag.split_at_checked(8)?;
            Some((
                u64::from_be_bytes(height.try_into().ok()?),
                hash::from_bytes(hash.try_into().ok()?)?,
            ))
        }))
    }

    /// Up to `n` state-tree nodes still to download, not in `skip`.
    pub fn state_wanted(
        &self,
        n: usize,
        skip: &HashSet<(u8, u64)>,
    ) -> Result<Vec<(u8, u64)>, LedgerError> {
        let txn = self.env.read_txn()?;
        Ok(self.tree.sync_wanted(&txn, n, skip)?)
    }

    /// Takes downloaded state-tree replies (level, index, data), in one
    /// transaction: what became of each.
    pub fn put_state(&self, replies: &[(u8, u64, Vec<u8>)]) -> Result<Vec<SyncPut>, LedgerError> {
        let mut txn = self.env.write_txn()?;
        let out = replies
            .iter()
            .map(|(level, index, data)| self.tree.sync_put(&mut txn, *level as usize, *index, data))
            .collect::<Result<_, _>>()?;
        txn.commit()?;
        Ok(out)
    }

    /// Whether the state download has nothing left to fetch.
    pub fn state_done(&self) -> Result<bool, LedgerError> {
        let txn = self.env.read_txn()?;
        Ok(self.tree.sync_done(&txn)?)
    }

    /// Checks a fast-sync base: `base` against `parent` (link, time, target,
    /// PoW) and `block` with its chain proof, given the parent's total work.
    /// The proof covers the parent and everything before it.
    pub fn verify_base(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        block: &Block,
    ) -> Result<(), LedgerError> {
        if parent.height == 0 {
            return Err(LedgerError::Header(HeaderError::Height));
        }
        header::check(&self.consensus, parent, base).map_err(LedgerError::Header)?;
        self.verifier
            .verify_block(base, block, parent_work)
            .map_err(LedgerError::Block)
    }

    /// Starts a new ledger (only genesis) at a fast-sync base: checks it
    /// ([`Ledger::verify_base`]), makes the finished state download the
    /// tree and checks it against the base's state, and stores the base and
    /// its parent's header as the main chain.
    pub fn install(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        block: &Block,
    ) -> Result<(), LedgerError> {
        self.verify_base(parent, parent_work, base, block)?;
        let mut txn = self.env.write_txn()?;
        if self.store.tip(&txn)?.ok_or(store::Error::Corrupt)?.height != 0 {
            return Err(LedgerError::NotFresh);
        }
        if self.tree.sync_tag(&txn)? != Some(sync_tag(base)) {
            return Err(state::Error::Incomplete.into());
        }
        match self
            .tree
            .sync_finish(&mut txn, base.state.size, base.height)
        {
            Ok(s) if s == base.state => {}
            Ok(_) | Err(state::Error::Corrupt) => {
                // Every part was checked on arrival, so this is our fault:
                // start the download over.
                txn.abort();
                let mut txn = self.env.write_txn()?;
                self.tree.sync_clear(&mut txn)?;
                txn.commit()?;
                return Err(LedgerError::BadSnapshot);
            }
            Err(state::Error::NotEmpty) => return Err(LedgerError::NotFresh),
            Err(e) => return Err(e.into()),
        }
        let work = parent_work + header::work(self.consensus.log_target(parent));
        self.store.add_header(&mut txn, parent, parent_work)?;
        self.store.add(&mut txn, base, block, work)?;
        self.store.set_base(&mut txn, parent, base)?;
        txn.commit()?;
        Ok(())
    }

    /// Makes the branch ending at `new` (stored) the main chain. On a block
    /// that fails to apply, returns it and its descendants up to `new`.
    fn switch_to(&self, txn: &mut RwTxn, new: &Header) -> Result<Added, (Vec<Hash>, LedgerError)> {
        let fail = |e: LedgerError| (vec![], e);
        // The branch, newest first, down to a main-chain block.
        let mut branch = vec![*new];
        loop {
            let last = branch.last().expect("non-empty");
            let parent = self
                .store
                .header_by_hash(txn, &last.prev)
                .map_err(|e| fail(e.into()))?
                .ok_or_else(|| fail(store::Error::Corrupt.into()))?;
            if self
                .store
                .height_of(txn, &parent.hash())
                .map_err(|e| fail(e.into()))?
                .is_some()
            {
                break;
            }
            branch.push(parent);
        }
        branch.reverse();
        let fork = branch[0].height - 1;
        if fork < self.store.base(txn).map_err(|e| fail(e.into()))? {
            return Err(fail(LedgerError::BeforeBase));
        }

        // Revert the main chain down to the fork point.
        let mut reverted = 0;
        loop {
            let tip = self
                .store
                .tip(txn)
                .map_err(|e| fail(e.into()))?
                .ok_or_else(|| fail(store::Error::Corrupt.into()))?;
            if tip.height == fork {
                break;
            }
            let undo = self
                .store
                .take_undo(txn, &tip.hash())
                .map_err(|e| fail(e.into()))?;
            self.tree
                .revert_block(txn, tip.height, &undo)
                .map_err(|e| fail(e.into()))?;
            self.store.pop_main(txn).map_err(|e| fail(e.into()))?;
            reverted += 1;
        }

        // Apply the branch.
        for (i, h) in branch.iter().enumerate() {
            let bad = || branch[i..].iter().map(Header::hash).collect::<Vec<_>>();
            let parent = self
                .store
                .tip(txn)
                .map_err(|e| fail(e.into()))?
                .ok_or_else(|| fail(store::Error::Corrupt.into()))?;
            let block = self
                .store
                .block_by_hash(txn, &h.hash())
                .map_err(|e| fail(e.into()))?
                .ok_or_else(|| fail(store::Error::Corrupt.into()))?;
            let undo = block::apply(&self.tree, txn, &parent, h, &block)
                .map_err(|e| (bad(), LedgerError::Block(e)))?;
            self.store
                .put_undo(txn, &h.hash(), &undo)
                .map_err(|e| fail(e.into()))?;
            self.store.push_main(txn, h).map_err(|e| fail(e.into()))?;
            self.tree
                .prune_history(txn, h.height.saturating_sub(self.snap_sync_limit))
                .map_err(|e| fail(e.into()))?;
            self.store
                .prune_undo(txn, h.height.saturating_sub(self.snap_sync_limit))
                .map_err(|e| fail(e.into()))?;
            if self.keep_blocks != 0 {
                let keep = self.keep_blocks.max(self.snap_sync_limit);
                self.store
                    .prune_bodies(txn, h.height.saturating_sub(keep))
                    .map_err(|e| fail(e.into()))?;
            }
        }
        Ok(if reverted == 0 && branch.len() == 1 {
            Added::Extended
        } else {
            Added::Reorg {
                reverted,
                applied: branch.len() as u64,
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::babybear::BabyBear;
    use crate::db::test::TempEnv;
    use crate::hash::{self, HASH_LEN};
    use crate::header::MAX_LOG_TARGET;
    use crate::output::{Commitment, NONCE_LEN, OutputRecord};
    use crate::stark::StarkProof;
    use crate::state::DEPTH;
    use crate::state::State;

    /// Checks only the body hash: the chain proof is out of scope here.
    pub(crate) struct BodyOnly;

    impl Verifier for BodyOnly {
        fn verify_block(&self, header: &Header, block: &Block, _: u128) -> Result<(), BlockError> {
            if block.body_hash() == header.body {
                Ok(())
            } else {
                Err(BlockError::Body)
            }
        }
    }

    /// The smallest well-formed proof encoding.
    pub(crate) fn tiny_proof() -> StarkProof {
        crate::block::no_proof()
    }

    fn record(n: u32) -> OutputRecord {
        let mut c = [BabyBear::new(n + 1000); HASH_LEN];
        c[1] = BabyBear::new(n);
        OutputRecord {
            commitment: Commitment(hash::to_bytes(&c)),
            nonce: [n as u8; NONCE_LEN],
        }
    }

    /// One branch's view: a scratch tree that applies the branch's lists to
    /// learn each header's state.
    struct Branch {
        env: TempEnv,
        tree: StateTree,
        tip: Header,
    }

    impl Branch {
        fn new(genesis: Header) -> Self {
            let env = TempEnv::new();
            let tree = StateTree::open(env.env()).unwrap();
            Self {
                env,
                tree,
                tip: genesis,
            }
        }

        fn state(&self) -> State {
            self.tree
                .state(&self.env.env().read_txn().unwrap())
                .unwrap()
        }

        /// The next block on this branch creating `created` and spending
        /// `spent` (by record number).
        fn next(&mut self, created: &[u32], spent: &[u32]) -> (Header, Block) {
            let spent: Vec<_> = spent.iter().map(|&n| record(n).commitment).collect();
            let outputs: Vec<_> = created.iter().map(|&n| record(n)).collect();
            let mut txn = self.env.env().write_txn().unwrap();
            // Inserts first, as the tree applies a block, so a spend of a
            // new output finds its slot.
            let height = self.tip.height + 1;
            self.tree
                .apply_changes(&mut txn, height, &outputs, &[])
                .unwrap();
            let mut inputs: Vec<u64> = spent
                .iter()
                .map(|c| self.tree.get(&txn, c).unwrap().unwrap().0)
                .collect();
            inputs.sort_unstable();
            self.tree
                .apply_changes(&mut txn, height, &[], &spent)
                .unwrap();
            let block = Block {
                inputs,
                outputs,
                proof: tiny_proof(),
            };
            txn.commit().unwrap();
            let header = Header {
                height,
                prev: self.tip.hash(),
                timestamp: self.tip.timestamp + 600,
                state: self.state(),
                body: block.body_hash(),
                aux_hash: [0; 32],
                nonce: 0,
            };
            self.tip = header;
            (header, block)
        }
    }

    pub(crate) fn ledger(env: &TempEnv, genesis: &Header) -> Ledger {
        ledger_keeping(env, genesis, DEFAULT_SNAP_SYNC_LIMIT)
    }

    fn ledger_keeping(env: &TempEnv, genesis: &Header, snap_sync_limit: u64) -> Ledger {
        let store = ChainStore::open(env.env()).unwrap();
        let tree = StateTree::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        store.init(&mut txn, genesis).unwrap();
        txn.commit().unwrap();
        let l = Ledger {
            env: env.env().clone(),
            store,
            tree,
            // The easiest target: any hash passes.
            consensus: Consensus {
                genesis_time: genesis.timestamp,
                genesis_log_target: MAX_LOG_TARGET,
                ..Consensus::DEFAULT
            },
            verifier: Arc::new(BodyOnly),
            snap_sync_limit,
            keep_blocks: snap_sync_limit,
        };
        l.start_history().unwrap();
        l
    }

    fn outputs(l: &Ledger) -> Vec<OutputRecord> {
        let txn = l.env.read_txn().unwrap();
        l.tree
            .outputs(&txn)
            .unwrap()
            .into_iter()
            .map(|(_, o)| o)
            .collect()
    }

    fn sorted(mut v: Vec<OutputRecord>) -> Vec<OutputRecord> {
        v.sort_by_key(|o| o.commitment.0);
        v
    }

    #[test]
    fn follows_the_branch_with_the_most_work() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let l = ledger(&env, &genesis);
        let (mut a, mut b) = (Branch::new(genesis), Branch::new(genesis));

        // Branch A: two blocks, the second spending the first's output.
        let a1 = a.next(&[1, 2], &[]);
        let a2 = a.next(&[3], &[1]);
        assert_eq!(l.add(&a1.0, &a1.1).unwrap(), Added::Extended);
        assert_eq!(l.add(&a2.0, &a2.1).unwrap(), Added::Extended);
        assert_eq!(l.add(&a2.0, &a2.1).unwrap(), Added::Known);
        assert_eq!(l.tip().unwrap().0, a2.0);

        // Branch B: as heavy at two blocks (the first seen stays), heavier
        // at three: a reorg.
        let b1 = b.next(&[10], &[]);
        let b2 = b.next(&[11, 12], &[10]);
        let b3 = b.next(&[13], &[11]);
        assert_eq!(l.add(&b1.0, &b1.1).unwrap(), Added::Side);
        assert_eq!(l.add(&b2.0, &b2.1).unwrap(), Added::Side);
        assert_eq!(l.tip().unwrap().0, a2.0);
        assert_eq!(
            l.add(&b3.0, &b3.1).unwrap(),
            Added::Reorg {
                reverted: 2,
                applied: 3
            }
        );
        assert_eq!(l.tip().unwrap().0, b3.0);
        let txn = l.env.read_txn().unwrap();
        assert_eq!(l.tree.state(&txn).unwrap(), b.state());
        drop(txn);
        assert_eq!(sorted(outputs(&l)), sorted(vec![record(12), record(13)]));

        // And back: A3 ties, A4 wins.
        let a3 = a.next(&[4], &[2]);
        let a4 = a.next(&[5], &[]);
        assert_eq!(l.add(&a3.0, &a3.1).unwrap(), Added::Side);
        // The branches fork at genesis, so all of A is applied again.
        assert_eq!(
            l.add(&a4.0, &a4.1).unwrap(),
            Added::Reorg {
                reverted: 3,
                applied: 4
            }
        );
        assert_eq!(l.tip().unwrap().0, a4.0);
        let txn = l.env.read_txn().unwrap();
        assert_eq!(l.tree.state(&txn).unwrap(), a.state());
        assert_eq!(l.store.height_of(&txn, &b3.0.hash()).unwrap(), None);
        assert!(
            l.store
                .header_by_hash(&txn, &b3.0.hash())
                .unwrap()
                .is_some()
        );
        drop(txn);
        assert_eq!(
            sorted(outputs(&l)),
            sorted(vec![record(3), record(4), record(5)])
        );
    }

    #[test]
    fn branches_forking_beyond_the_limit_are_refused() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let l = ledger_keeping(&env, &genesis, 2);
        let mut a = Branch::new(genesis);
        for n in 1..=5 {
            let (h, b) = a.next(&[n], &[]);
            l.add(&h, &b).unwrap();
        }
        // The tip is at 5: branches may leave from block 3 up.
        let fork_at = |k: u32| {
            let mut b = Branch::new(genesis);
            for n in 1..=k {
                b.next(&[n], &[]);
            }
            b
        };
        let mut deep = fork_at(2);
        let d = deep.next(&[20], &[]);
        assert!(matches!(l.add(&d.0, &d.1), Err(LedgerError::TooDeep)));
        assert!(matches!(l.check_header(&d.0), Err(LedgerError::TooDeep)));
        let mut ok = fork_at(3);
        let o1 = ok.next(&[30], &[]);
        assert_eq!(l.add(&o1.0, &o1.1).unwrap(), Added::Side);
        let o2 = ok.next(&[31], &[]);
        assert_eq!(l.add(&o2.0, &o2.1).unwrap(), Added::Side);
        // The main chain moves on: the branch at 3 is now too deep to grow.
        let (h, b) = a.next(&[6], &[]);
        l.add(&h, &b).unwrap();
        let o3 = ok.next(&[32], &[]);
        assert!(matches!(l.add(&o3.0, &o3.1), Err(LedgerError::TooDeep)));
    }

    /// Bodies older than `keep_blocks` and undo data older than the
    /// reorg limit are deleted as the chain grows; headers stay, and
    /// reorgs within the limit still work.
    #[test]
    fn old_blocks_and_undo_data_are_pruned() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let mut l = ledger_keeping(&env, &genesis, 2);
        l.keep_blocks = 4;
        let mut a = Branch::new(genesis);
        for n in 1..=8 {
            let (h, b) = a.next(&[n], &[]);
            l.add(&h, &b).unwrap();
        }
        let mut txn = l.env.write_txn().unwrap();
        for h in 1..=8u64 {
            let hash = l.store.hash_at(&txn, h).unwrap().unwrap();
            assert!(l.store.header(&txn, h).unwrap().is_some());
            assert_eq!(
                l.store.block(&txn, h).unwrap().is_some(),
                h >= 4,
                "body {h}"
            );
            assert_eq!(
                l.store.take_undo(&mut txn, &hash).is_ok(),
                h >= 6,
                "undo {h}"
            );
        }
        assert_eq!(l.store.bodies_from(&txn).unwrap(), 4);
        assert_eq!(l.store.undo_from(&txn).unwrap(), 6);
        txn.abort();
        // A heavier branch from block 6 reverts 7 and 8 with their undo data.
        let mut b = Branch::new(genesis);
        for n in 1..=6 {
            b.next(&[n], &[]);
        }
        let mut last = None;
        for n in 60..63 {
            let (h, blk) = b.next(&[n], &[]);
            last = Some(l.add(&h, &blk).unwrap());
        }
        assert_eq!(
            last,
            Some(Added::Reorg {
                reverted: 2,
                applied: 3
            })
        );
    }

    #[test]
    fn a_block_cant_spend_what_it_creates() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let l = ledger(&env, &genesis);
        let mut a = Branch::new(genesis);
        let a1 = a.next(&[1, 2], &[1]);
        let e = l.add(&a1.0, &a1.1).unwrap_err();
        assert!(
            matches!(e, LedgerError::Block(BlockError::SpendsItsOwnOutput)),
            "{e}"
        );
        assert_eq!(l.tip().unwrap().0, genesis);
        // A block later, it can.
        let mut b = Branch::new(genesis);
        let b1 = b.next(&[1, 2], &[]);
        let b2 = b.next(&[3], &[1]);
        l.add(&b1.0, &b1.1).unwrap();
        assert_eq!(l.add(&b2.0, &b2.1).unwrap(), Added::Extended);
    }

    #[test]
    fn blocks_spend_by_the_position_of_an_unspent_output() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let l = ledger(&env, &genesis);
        let mut b = Branch::new(genesis);
        let b1 = b.next(&[1, 2], &[]);
        let b2 = b.next(&[3], &[1]);
        l.add(&b1.0, &b1.1).unwrap();
        l.add(&b2.0, &b2.1).unwrap();
        let spent = b2.1.inputs[0];
        // A block naming the slot just emptied, or the sentinel's.
        let (h3, block3) = b.next(&[4], &[]);
        fn refusal(l: &Ledger, h3: &Header, block3: &Block, inputs: Vec<u64>) -> BlockError {
            let block = Block {
                inputs,
                ..block3.clone()
            };
            let h = Header {
                body: block.body_hash(),
                ..*h3
            };
            match l.add(&h, &block).unwrap_err() {
                LedgerError::Block(e) => e,
                e => panic!("{e}"),
            }
        }
        let e = refusal(&l, &h3, &block3, vec![spent]);
        assert!(matches!(e, BlockError::NotAnOutput(p) if p == spent), "{e}");
        let e = refusal(&l, &h3, &block3, vec![0]);
        assert!(matches!(e, BlockError::NotCanonical), "{e}");
        assert_eq!(l.tip().unwrap().0, b2.0);
    }

    #[test]
    fn a_block_that_doesnt_apply_changes_nothing() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        let l = ledger(&env, &genesis);
        let mut a = Branch::new(genesis);
        let a1 = a.next(&[1], &[]);
        l.add(&a1.0, &a1.1).unwrap();

        // A heavier branch whose second block claims a state its lists don't
        // produce.
        let mut b = Branch::new(genesis);
        let b1 = b.next(&[7], &[]);
        let (mut b2, block2) = b.next(&[8], &[]);
        b2.state.size += 1;
        let b2 = Header {
            body: block2.body_hash(),
            ..b2
        };
        assert_eq!(l.add(&b1.0, &b1.1).unwrap(), Added::Side);
        let e = l.add(&b2, &block2).unwrap_err();
        assert!(
            matches!(e, LedgerError::Block(BlockError::StateMismatch)),
            "{e}"
        );
        // Nothing changed, and the bad block (not its valid parent) is
        // marked.
        assert_eq!(l.tip().unwrap().0, a1.0);
        assert_eq!(sorted(outputs(&l)), vec![record(1)]);
        let txn = l.env.read_txn().unwrap();
        assert!(l.store.is_invalid(&txn, &b2.hash()).unwrap());
        assert!(!l.store.is_invalid(&txn, &b1.0.hash()).unwrap());
        drop(txn);
        // Its children are refused outright; so are orphans and bad bodies.
        let child = Header {
            height: 3,
            prev: b2.hash(),
            timestamp: b2.timestamp + 600,
            ..b2
        };
        assert!(matches!(
            l.add(&child, &block2),
            Err(LedgerError::KnownInvalid)
        ));
        let orphan = Header {
            prev: [BabyBear::new(5); HASH_LEN],
            ..child
        };
        assert!(matches!(
            l.add(&orphan, &block2),
            Err(LedgerError::UnknownParent)
        ));
        let mut a2 = a.next(&[2], &[]);
        a2.0.body = b1.1.body_hash();
        assert!(matches!(
            l.add(&a2.0, &a2.1),
            Err(LedgerError::Block(BlockError::Body))
        ));
        assert_eq!(l.tip().unwrap().0, a1.0);
    }

    /// Downloads the state at `base` from `from` into `to`, as a
    /// fast-syncing node would.
    pub(crate) fn download_state(from: &Ledger, to: &Ledger, base: &Header) {
        to.begin_state(base).unwrap();
        loop {
            let wanted = to.state_wanted(5, &HashSet::new()).unwrap();
            if wanted.is_empty() {
                break;
            }
            let replies: Vec<_> = wanted
                .iter()
                .map(|&(l, i)| (l, i, from.state_nodes(&base.hash(), l, i).unwrap().unwrap()))
                .collect();
            assert!(
                to.put_state(&replies)
                    .unwrap()
                    .iter()
                    .all(|p| *p == SyncPut::Stored)
            );
        }
        assert!(to.state_done().unwrap());
    }

    #[test]
    fn a_new_ledger_starts_from_a_downloaded_state() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let env = TempEnv::new();
        // Keeping two blocks of history.
        let full = ledger_keeping(&env, &genesis, 2);
        let mut a = Branch::new(genesis);
        let blocks = [
            a.next(&[1, 2, 3], &[]),
            a.next(&[4], &[1]),
            a.next(&[5, 6], &[2, 4]),
            a.next(&[7], &[3]),
        ];
        for (h, b) in &blocks {
            assert_eq!(full.add(h, b).unwrap(), Added::Extended);
        }
        assert_eq!(full.servable_from().unwrap(), 2);
        assert_eq!(full.snapshot(1).unwrap(), None);
        assert_eq!(full.snapshot(5).unwrap(), None);
        assert_eq!(
            full.state_nodes(&blocks[0].0.hash(), DEPTH as u8, 0)
                .unwrap(),
            None
        );
        let snap = full.snapshot(2).unwrap().unwrap();
        assert_eq!((snap.parent, snap.base), (blocks[0].0, blocks[1].0));

        let env2 = TempEnv::new();
        let new = ledger(&env2, &genesis);
        let block2 = &blocks[1].1;
        // Not before the state is all there.
        new.begin_state(&snap.base).unwrap();
        let e = new
            .install(&snap.parent, snap.parent_work, &snap.base, block2)
            .unwrap_err();
        assert!(
            matches!(e, LedgerError::State(state::Error::Incomplete)),
            "{e}"
        );
        // A reply that doesn't hash to its node is refused.
        let mut bad = full
            .state_nodes(&snap.base.hash(), DEPTH as u8, 0)
            .unwrap()
            .unwrap();
        *bad.last_mut().unwrap() ^= 1;
        assert_eq!(
            new.put_state(&[(DEPTH as u8, 0, bad)]).unwrap(),
            vec![SyncPut::Bad]
        );
        download_state(&full, &new, &snap.base);
        let e = new
            .install(&snap.parent, snap.parent_work, &snap.base, &blocks[0].1)
            .unwrap_err();
        assert!(matches!(e, LedgerError::Block(BlockError::Body)), "{e}");
        new.install(&snap.parent, snap.parent_work, &snap.base, block2)
            .unwrap();
        let work2 = full
            .store
            .work(&env.env().read_txn().unwrap(), 2)
            .unwrap()
            .unwrap();
        assert_eq!(new.tip().unwrap(), (blocks[1].0, work2));
        assert_eq!(new.state_progress().unwrap(), None);
        // Only once.
        assert!(matches!(
            new.begin_state(&snap.base),
            Err(LedgerError::NotFresh)
        ));
        let e = new
            .install(&snap.parent, snap.parent_work, &snap.base, block2)
            .unwrap_err();
        assert!(matches!(e, LedgerError::NotFresh), "{e}");

        // It follows from there, and can serve the state from its base up.
        for (h, b) in &blocks[2..] {
            assert_eq!(new.add(h, b).unwrap(), Added::Extended);
        }
        assert_eq!(new.tip().unwrap(), full.tip().unwrap());
        assert_eq!(sorted(outputs(&new)), sorted(outputs(&full)));
        assert_eq!(new.snapshot(1).unwrap(), None);
        assert_eq!(new.snapshot(3).unwrap(), full.snapshot(3).unwrap());
        for (base, level) in [(1, DEPTH as u8), (2, DEPTH as u8), (3, 3)] {
            let hash = blocks[base].0.hash();
            assert_eq!(
                new.state_nodes(&hash, level, 0).unwrap(),
                full.state_nodes(&hash, level, 0).unwrap()
            );
        }
        // A third node syncs from the second's history.
        let env3 = TempEnv::new();
        let third = ledger(&env3, &genesis);
        let snap3 = new.snapshot(3).unwrap().unwrap();
        download_state(&new, &third, &snap3.base);
        third
            .install(&snap3.parent, snap3.parent_work, &snap3.base, &blocks[2].1)
            .unwrap();
        assert_eq!(
            third.add(&blocks[3].0, &blocks[3].1).unwrap(),
            Added::Extended
        );
        assert_eq!(sorted(outputs(&third)), sorted(outputs(&full)));

        // A branch from below the base isn't even stored.
        let mut b = Branch::new(genesis);
        let (h, blk) = b.next(&[100], &[]);
        let e = new.add(&h, &blk).unwrap_err();
        assert!(matches!(e, LedgerError::BeforeBase), "{e}");
        assert_eq!(new.tip().unwrap().0, blocks[3].0);
    }
}
