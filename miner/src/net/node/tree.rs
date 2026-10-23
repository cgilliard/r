//! Preimages for aux trees ([`crate::preimages`]), on a node with a
//! [`Config::preimages`] store (which advertises [`cap::TREE`]).
//!
//! **Serving.** `GET_PREIMAGE` asks for the bytes whose SHA-256 is a hash.
//! Short ones come back whole (`PREIMAGE`); longer ones as their size and
//! chunk root (`PREIMAGE_INFO`), whose chunks are then pulled like a
//! transaction's (`GET_BLOCK_CHUNKS` under [`wire::preimage_id`]). The
//! asker checks the bytes against the hash: a peer that sends wrong ones is
//! dropped. The protocol knows nothing of trees, so a leaf can be followed
//! down as far as it goes.
//!
//! **Fetching.** When the tip changes (and the chain isn't syncing), the
//! node walks the aux trees of the last [`SCAN`] main-chain blocks: each
//! root's node, its children, down to the 16 leaves, and the file of the
//! consensus leaf ([`onion::CONSENSUS_LEAF`]). What it has, it notes as
//! still in use; what it doesn't, it fetches from peers advertising
//! [`cap::TREE`], one after another until one has it. A hash no peer had
//! is left alone for [`RETRY_AFTER`]. A hash whose preimage we have
//! (fetched before, or one of our own aux files) is never fetched again.
//! Preimages no block in those [`SCAN`] leads to are pruned: nothing older
//! counts, so a miner committing to a new file in every block costs others
//! at most [`SCAN`] files of disk.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use super::{Backend, MAX_RANGES, MAX_SERVE, Message, Node, REQUEST_TIMEOUT};
use crate::chunk::{Assembly, ChunkError, Served};
use crate::hash::Hash;
use crate::mnemonic::sha256;
use crate::net::wire::{self, MAX_INLINE_PREIMAGE, Sha256, cap};
use crate::onion;
use crate::preimages::{AUX_DEPTH, MAX_PREIMAGE, children};

/// Blocks whose aux trees are fetched: those the relay consensus counts,
/// and the ones above them.
pub const SCAN: u64 = onion::WINDOW + onion::LAG;
/// Preimages fetched at once, at most.
const FETCHES: usize = 8;
/// Hashes waiting to be fetched, at most.
const MAX_QUEUE: usize = 4096;
/// Ask a peer this many times before trying another.
const TRIES: u32 = 3;
/// Leave a hash no peer had alone this long.
pub const RETRY_AFTER: Duration = Duration::from_secs(60);
/// Long preimages kept ready to serve.
const SERVED_CACHE: usize = 16;

/// Where a hash is in an aux tree: `level` 0 is the root, [`AUX_DEPTH`]
/// the leaves (whose preimages are files); `height` is the block.
#[derive(Clone, Copy, Debug)]
struct Want {
    level: u8,
    index: usize,
    height: u64,
}

struct Fetch {
    want: Want,
    session: u32,
    /// Peers asked so far.
    tried: HashSet<u32>,
    /// From `PREIMAGE_INFO`: the chunk id and what's arrived.
    asm: Option<(Hash, Assembly)>,
    asked: Instant,
    tries: u32,
}

#[derive(Default)]
pub(super) struct Trees {
    fetches: HashMap<Sha256, Fetch>,
    queue: VecDeque<(Sha256, Want)>,
    queued: HashSet<Sha256>,
    /// Chunk ids of fetches being pulled in chunks.
    ids: HashMap<Hash, Sha256>,
    /// Hashes no peer had, and when we gave up.
    failed: HashMap<Sha256, Instant>,
    /// Stored preimages blocks led to (and the blocks), to note at the
    /// tick.
    touched: Vec<(Sha256, u64)>,
    /// The tip whose blocks were last walked.
    scanned: Option<Hash>,
    pub(super) served: VecDeque<(Hash, Served)>,
}

impl Trees {
    /// Whether chunks under `id` are for a preimage.
    pub(super) fn has_id(&self, id: &Hash) -> bool {
        self.ids.contains_key(id)
    }
}

impl<B: Backend> Node<B> {
    pub(super) fn on_get_preimage(&mut self, session: u32, hash: Sha256) {
        let data = self
            .cfg
            .preimages
            .as_ref()
            .and_then(|s| s.get(&hash).ok().flatten());
        let msg = match data {
            None => Message::NoPreimage { hash },
            Some(data) if data.len() <= MAX_INLINE_PREIMAGE => Message::Preimage { hash, data },
            Some(data) => {
                let served = Served::new(data);
                let (size, root) = (served.bytes.len() as u32, served.root);
                let id = wire::preimage_id(size, &root);
                let served_now = &mut self.trees.served;
                if !served_now.iter().any(|(h, _)| *h == id) {
                    if served_now.len() == SERVED_CACHE {
                        served_now.pop_front();
                    }
                    served_now.push_back((id, served));
                }
                Message::PreimageInfo { hash, size, root }
            }
        };
        self.send_peer(session, &msg);
    }

    /// Notes a hash that block `want.height` leads to: walks on from it if
    /// we have its preimage, else queues it for fetching.
    fn want(&mut self, hash: Sha256, want: Want) {
        // Zero is an unset leaf; of the leaves, only the consensus file is
        // fetched (the rest mean nothing to this node).
        if hash == [0; 32] || (want.level == AUX_DEPTH && want.index != onion::CONSENSUS_LEAF) {
            return;
        }
        let Some(store) = &self.cfg.preimages else {
            return;
        };
        match store.get(&hash) {
            Ok(Some(bytes)) => {
                self.trees.touched.push((hash, want.height));
                self.walk(&bytes, want);
            }
            Ok(None) => {
                let t = &mut self.trees;
                if t.fetches.contains_key(&hash)
                    || t.failed.contains_key(&hash)
                    || t.queue.len() >= MAX_QUEUE
                    || !t.queued.insert(hash)
                {
                    return;
                }
                t.queue.push_back((hash, want));
            }
            Err(e) => (self.log)(format!("couldn't read a preimage: {e}")),
        }
    }

    /// Walks on from a node's preimage to its children.
    fn walk(&mut self, bytes: &[u8], want: Want) {
        if want.level >= AUX_DEPTH {
            return;
        }
        if let Some((left, right)) = children(bytes) {
            for (i, child) in [left, right].into_iter().enumerate() {
                let w = Want {
                    level: want.level + 1,
                    index: 2 * want.index + i,
                    height: want.height,
                };
                self.want(child, w);
            }
        }
    }

    /// A peer advertising [`cap::TREE`] not asked yet.
    fn tree_source(&mut self, tried: &HashSet<u32>) -> Option<u32> {
        let mut sources: Vec<u32> = self
            .peers
            .iter()
            .filter(|(s, p)| p.capabilities & cap::TREE != 0 && !tried.contains(s))
            .map(|(&s, _)| s)
            .collect();
        sources.sort_unstable();
        (!sources.is_empty())
            .then(|| sources[self.rng.next_element().as_u32() as usize % sources.len()])
    }

    fn start_tree_fetches(&mut self) {
        while self.trees.fetches.len() < FETCHES {
            let Some((hash, want)) = self.trees.queue.pop_front() else {
                return;
            };
            let Some(session) = self.tree_source(&HashSet::new()) else {
                // No one to ask yet.
                self.trees.queue.push_front((hash, want));
                return;
            };
            self.trees.queued.remove(&hash);
            self.trees.fetches.insert(
                hash,
                Fetch {
                    want,
                    session,
                    tried: HashSet::from([session]),
                    asm: None,
                    asked: Instant::now(),
                    tries: 1,
                },
            );
            self.send_peer(session, &Message::GetPreimage { hash });
        }
    }

    /// Asks the next peer for `hash`, or gives up on it for a while.
    fn next_tree_source(&mut self, hash: Sha256) {
        let Some(f) = self.trees.fetches.get_mut(&hash) else {
            return;
        };
        if let Some((id, _)) = f.asm.take() {
            self.trees.ids.remove(&id);
        }
        let tried = f.tried.clone();
        match self.tree_source(&tried) {
            Some(s) => {
                let f = self.trees.fetches.get_mut(&hash).expect("fetching");
                f.session = s;
                f.tried.insert(s);
                f.asked = Instant::now();
                f.tries = 1;
                self.send_peer(s, &Message::GetPreimage { hash });
            }
            None => {
                self.trees.fetches.remove(&hash);
                self.trees.failed.insert(hash, Instant::now());
            }
        }
    }

    /// The fetch of `hash` if `session` is the peer it's asking.
    fn fetch_from(&mut self, session: u32, hash: &Sha256) -> Option<&mut Fetch> {
        self.trees
            .fetches
            .get_mut(hash)
            .filter(|f| f.session == session)
    }

    pub(super) fn on_preimage(&mut self, session: u32, hash: Sha256, data: Vec<u8>) {
        if self.fetch_from(session, &hash).is_none() {
            return;
        }
        if sha256(&data) != hash {
            self.drop_peer(session, "sent a wrong preimage");
            return;
        }
        self.got_preimage(hash, &data);
    }

    pub(super) fn on_preimage_info(&mut self, session: u32, hash: Sha256, size: u32, root: Hash) {
        let Some(f) = self.fetch_from(session, &hash) else {
            return;
        };
        if f.asm.is_some() {
            return;
        }
        if size as usize > MAX_PREIMAGE {
            // Too long for us, whoever has it.
            self.trees.fetches.remove(&hash);
            self.trees.failed.insert(hash, Instant::now());
            return;
        }
        let id = wire::preimage_id(size, &root);
        f.asm = Some((id, Assembly::new(size as usize, root)));
        f.asked = Instant::now();
        self.trees.ids.insert(id, hash);
        self.request_tree_chunks(hash);
    }

    pub(super) fn on_no_preimage(&mut self, session: u32, hash: Sha256) {
        if self.fetch_from(session, &hash).is_some() {
            self.next_tree_source(hash);
        }
    }

    /// The peer no longer has the chunks under `id`.
    pub(super) fn tree_not_found(&mut self, session: u32, id: &Hash) {
        if let Some(&hash) = self.trees.ids.get(id)
            && self.fetch_from(session, &hash).is_some()
        {
            self.next_tree_source(hash);
        }
    }

    /// Asks for every chunk of `hash`'s preimage still missing.
    fn request_tree_chunks(&mut self, hash: Sha256) {
        let Some(Fetch {
            session,
            asm: Some((id, asm)),
            ..
        }) = self.trees.fetches.get(&hash)
        else {
            return;
        };
        let mut ranges: Vec<(u32, u16)> = Vec::new();
        for i in asm.wanted(MAX_SERVE) {
            match ranges.last_mut() {
                Some((first, count)) if *first as usize + *count as usize == i => *count += 1,
                _ => ranges.push((i as u32, 1)),
            }
        }
        let (session, id) = (*session, *id);
        for batch in ranges.chunks(MAX_RANGES) {
            self.send_peer(
                session,
                &Message::GetBlockChunks {
                    hash: id,
                    ranges: batch.to_vec(),
                },
            );
        }
    }

    pub(super) fn on_tree_chunk(&mut self, session: u32, id: Hash, index: u32, data: &[u8]) {
        let Some(&hash) = self.trees.ids.get(&id) else {
            return;
        };
        let Some(Fetch {
            asm: Some((_, asm)),
            ..
        }) = self.fetch_from(session, &hash)
        else {
            return;
        };
        let had_leaves = asm.has_leaves();
        match asm.add(index as usize, data) {
            Ok(_) if asm.is_complete() => {
                let f = self.trees.fetches.get_mut(&hash).expect("fetching");
                let (id, asm) = f.asm.take().expect("chunks");
                self.trees.ids.remove(&id);
                let bytes = asm.finish().expect("complete");
                if sha256(&bytes) != hash {
                    self.drop_peer(session, "sent a wrong preimage");
                    return;
                }
                self.got_preimage(hash, &bytes);
            }
            Ok(_) if !had_leaves && asm.has_leaves() => self.request_tree_chunks(hash),
            Ok(_) | Err(ChunkError::Shape) => {}
            Err(e) => self.drop_peer(session, &format!("bad preimage chunk ({e:?})")),
        }
    }

    /// Stores a fetched preimage and walks on from it.
    fn got_preimage(&mut self, hash: Sha256, bytes: &[u8]) {
        let Some(f) = self.trees.fetches.remove(&hash) else {
            return;
        };
        let Some(store) = &self.cfg.preimages else {
            return;
        };
        if let Err(e) = store.put(bytes, f.want.height) {
            (self.log)(format!("couldn't store a preimage: {e}"));
            return;
        }
        if f.want.level == AUX_DEPTH {
            let from = self
                .peers
                .get(&f.session)
                .map_or("?".into(), |p| p.addr.to_string());
            (self.log)(format!(
                "fetched relay consensus file {} ({} bytes, from block {}) from {from}: {}",
                &crate::wallet::hex(&hash)[..16],
                bytes.len(),
                f.want.height,
                match onion::parse(bytes) {
                    Ok(onion::Mode::Init) => "init mode".to_string(),
                    Ok(onion::Mode::Onion(r)) => format!("onion mode, {} relays", r.len()),
                    Err(e) => format!("not valid ({e})"),
                }
            ));
        }
        self.walk(bytes, f.want);
    }

    /// The fetches of a peer that's gone go to others.
    pub(super) fn tree_lost_session(&mut self, session: u32) {
        let lost: Vec<Sha256> = self
            .trees
            .fetches
            .iter()
            .filter(|(_, f)| f.session == session)
            .map(|(h, _)| *h)
            .collect();
        for hash in lost {
            self.next_tree_source(hash);
        }
    }

    pub(super) fn tree_tick(&mut self, now: Instant) {
        let Some(store) = self.cfg.preimages.clone() else {
            return;
        };
        // Late answers: ask again, then another peer.
        let late: Vec<Sha256> = self
            .trees
            .fetches
            .iter()
            .filter(|(_, f)| now - f.asked > REQUEST_TIMEOUT)
            .map(|(h, _)| *h)
            .collect();
        for hash in late {
            let f = self.trees.fetches.get_mut(&hash).expect("fetching");
            if f.tries >= TRIES {
                self.next_tree_source(hash);
                continue;
            }
            f.tries += 1;
            f.asked = now;
            if f.asm.is_some() {
                self.request_tree_chunks(hash);
            } else {
                let s = f.session;
                self.send_peer(s, &Message::GetPreimage { hash });
            }
        }
        self.trees.failed.retain(|_, at| now - *at < RETRY_AFTER);

        // A new tip: walk the recent blocks' trees, and prune.
        let tip = self.backend.tip();
        if self.trees.scanned != Some(tip.hash) && !self.chain_busy() {
            self.trees.scanned = Some(tip.hash);
            for height in tip.height.saturating_sub(SCAN - 1).max(1)..=tip.height {
                if let Some(h) = self.backend.header_at(height) {
                    let want = Want {
                        level: 0,
                        index: 0,
                        height,
                    };
                    self.want(h.aux_hash, want);
                }
            }
            // Keep what the scanned blocks lead to.
            if tip.height >= SCAN {
                let from = tip.height + 1 - SCAN;
                match store.prune(from) {
                    Ok(0) => {}
                    Ok(n) => (self.log)(format!(
                        "pruned {n} preimage{} no block since {from} leads to",
                        if n == 1 { "" } else { "s" },
                    )),
                    Err(e) => (self.log)(format!("couldn't prune preimages: {e}")),
                }
            }
        }
        if !self.trees.touched.is_empty() {
            let touched = std::mem::take(&mut self.trees.touched);
            if let Err(e) = store.touch(&touched) {
                (self.log)(format!("couldn't note preimages in use: {e}"));
            }
        }
        self.start_tree_fetches();
    }
}
