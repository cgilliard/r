//! The miner and node: proves, mines and checks blocks until stopped (or
//! until the chain ends), logging each step. The chain (blocks on every
//! branch, the state tree) lives in LMDB in `<data dir>/chain`, so each run
//! carries on from the last. With `--no-mine` it only follows and serves the
//! chain ([`follow`]).
//!
//! A wallet with a random seed pays itself: each block holds a reward and
//! one transaction spending every coin the wallet finds in the state into
//! two new outputs, paying a fee. Every output goes to a fresh address, with
//! its amount carried in its nonce ([`crate::wallet`]). For each block the
//! miner:
//!
//! 1. proves the block's state transition ([`BlockProver`]), holding the
//!    database only for the quick native part;
//! 2. proves its chain proof ([`Chain`]), covering the parent header's PoW
//!    and every earlier block;
//! 3. builds the header and mines it at the ASERT target;
//! 4. serializes the block and adds the decoded copy to the ledger as a
//!    peer's block would be ([`Ledger::add`]: header, chain proof, lists
//!    against the state).
//!
//! A network node ([`crate::net`]) runs alongside: it finds peers (from
//! `--peer`, the saved peer list in `<data dir>/peers`, and what peers tell
//! it), syncs the chain (a new node fast-syncs from `--sync-depth` blocks
//! below the best tip), serves it, announces each block mined here, and
//! adds peers' blocks to the same ledger, which follows the branch with the
//! most work. Mining starts once the node has caught up. Whenever the tip
//! moves (a peer's block or a reorg), the miner drops the block in progress
//! (checked after each proving step and between batches of nonces) and
//! starts on the new tip. Rewards follow the emission schedule
//! ([`Consensus::subsidy`]); at the final height the chain ends.
//!
//! When the chain ends it checks the tip the way a fast-syncing node would, and
//! restores the wallet from its seed using only the state tree.

use std::error::Error;
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::babybear::BabyBear;
use crate::block::{self, Block, BlockProver};
use crate::chain::{Chain, ChainProof};
use crate::db;
use crate::fri::FriParams;
use crate::hash::{self, Hash};
use crate::header::{self, Consensus, FRAC_BITS, Header};
use crate::ledger::{Added, Ledger};
use crate::merkle::MerkleTree;
use crate::net::LedgerBackend;
use crate::net::node::{self, Config, Event, Node};
use crate::net::wire::cap;
use crate::output::{Amount, COIN};
use crate::pow::{self, Dataset};
use crate::rng::Rng;
use crate::stark::KeyCache;
use crate::state::StateTree;
use crate::store::ChainStore;
use crate::transaction::{Transaction, TxBody};
use crate::wallet::Wallet;

/// How the node reaches the network.
#[derive(Clone, Debug)]
pub struct Net {
    /// The UDP address to accept peers on (any port if none, and then not
    /// advertised to other nodes).
    pub listen: Option<SocketAddr>,
    /// Peers to dial, and redial whenever they're lost.
    pub peers: Vec<SocketAddr>,
    /// A new node fast-syncs from this many blocks below the best peer's
    /// tip.
    pub sync_depth: u64,
    /// Most peers connected at once.
    pub max_peers: usize,
    /// Blocks below the tip whose state is kept, to serve fast-syncing
    /// nodes.
    pub snap_sync_limit: u64,
}

/// The implicit fee each transaction leaves for the reward: 0.001 coin.
pub const FEE: Amount = COIN / 1000;

/// An amount in coins.
fn coins(units: u128) -> String {
    let (whole, frac) = (units / COIN as u128, units % COIN as u128);
    if frac == 0 {
        format!("{whole}")
    } else {
        format!("{whole}.{}", format!("{frac:08}").trim_end_matches('0'))
    }
}

/// Coins created by blocks `1 ..= height`.
fn supply(c: &Consensus, height: u64) -> u128 {
    let first = height.min(c.first_era) as u128;
    let second = height.min(c.final_height()).saturating_sub(c.first_era) as u128;
    first * c.first_subsidy as u128 + second * c.second_subsidy as u128
}

fn seed_bytes(seed: &Hash) -> Vec<u8> {
    seed.iter().flat_map(|x| x.as_u32().to_le_bytes()).collect()
}

fn seed_from(bytes: &[u8]) -> Option<Hash> {
    if bytes.len() != 32 {
        return None;
    }
    let mut seed = [BabyBear::ZERO; 8];
    for (x, chunk) in seed.iter_mut().zip(bytes.chunks_exact(4)) {
        *x = BabyBear::new(u32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(seed)
}

fn provers(consensus: Consensus, genesis: Header, tree: MerkleTree, keys: &KeyCache) -> Chain {
    let blocks = BlockProver::standard(FriParams::default(), keys);
    Chain::new(consensus, genesis, blocks, tree, keys)
}

struct Log(Instant);

impl Log {
    fn line(&self, msg: impl AsRef<str>) {
        println!(
            "[{:>8.1}s] {}",
            self.0.elapsed().as_secs_f64(),
            msg.as_ref()
        );
    }
}

fn short(h: &Hash) -> String {
    hash::to_bytes(h)[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn log2(x: u128) -> f64 {
    (x as f64).log2()
}

/// Expected hashes for `work` ([`header::work`] units are `2^-17` hashes).
fn hashes_log2(work: u128) -> f64 {
    log2(work) - 17.0
}

/// A network node's log lines, on the same clock as `log`.
fn net_log(log: &Log) -> Box<dyn Fn(String) + Send> {
    let start = log.0;
    Box::new(move |line| println!("[{:>8.1}s] net: {line}", start.elapsed().as_secs_f64()))
}

/// The running network node: its events and thread.
type Server = (Sender<Event>, JoinHandle<()>);

/// The chain in `data_dir/chain`, open: its ledger and what's around it.
struct Opened {
    dir: PathBuf,
    env: heed::Env,
    store: ChainStore,
    tree: StateTree,
    data: &'static Dataset,
    genesis: Header,
    consensus: Consensus,
    ledger: Arc<Ledger>,
    chain: Arc<Chain>,
}

/// Opens (creating if needed) the chain in `data_dir/chain`, with the
/// PoW dataset and verifying keys cached in `data_dir/cache`, and sets up
/// the provers and verifier.
fn open(log: &Log, data_dir: &Path, snap_sync_limit: u64) -> Result<Opened, Box<dyn Error>> {
    let dir = data_dir.join("chain");
    let cache = data_dir.join("cache");
    let (data, pow_tree, loaded) = Dataset::load(&cache.join("pow-dataset"));
    log.line(if loaded {
        "PoW dataset (64 MiB) loaded and checked against its root"
    } else {
        "PoW dataset (64 MiB) built and cached"
    });
    let env = db::open_env(&dir)?;
    let tree = StateTree::open(&env)?;
    let store = ChainStore::open(&env)?;
    let genesis = crate::genesis::header();
    let stored = store.genesis(&env.read_txn()?)?;
    match stored {
        Some(stored) if stored != genesis => {
            return Err(format!(
                "{} holds a chain from another genesis; delete it to start over",
                dir.display()
            )
            .into());
        }
        Some(_) => {}
        None => {
            if tree.state(&env.read_txn()?)? != genesis.state {
                return Err("the state tree isn't empty".into());
            }
            let mut txn = env.write_txn()?;
            store.init(&mut txn, &genesis)?;
            txn.commit()?;
        }
    }
    let consensus = Consensus::DEFAULT;
    log.line(
        "setting up the provers and verifier (item chunks of 682/2730, signature chunks of \
         7/31/126, 16-op segments, nodes at 2^15)",
    );
    let keys = KeyCache::open(cache.join("keys"));
    let chain = provers(consensus, genesis, pow_tree, &keys);
    if let Err(e) = keys.save() {
        log.line(format!("couldn't save the key cache: {e}"));
    }
    let ([first_rows, chain_rows, wrap_rows], log_height) = chain.used_rows();
    log.line(format!(
        "ready (proving keys are built on first use); chain circuits use {first_rows} (first), \
         {chain_rows} (chain) rows of 2^{log_height}, wrap {wrap_rows} of 2^{}",
        crate::chain::WRAP_LOG_HEIGHT
    ));
    let chain = Arc::new(chain);
    let ledger = Arc::new(Ledger {
        env: env.clone(),
        store,
        tree,
        consensus,
        data,
        verifier: chain.clone(),
        snap_sync_limit,
    });
    ledger.start_history()?;
    Ok(Opened {
        dir,
        env,
        store,
        tree,
        data,
        genesis,
        consensus,
        ledger,
        chain,
    })
}

/// A node on `net.listen` (or any port), dialing `net.peers` and the saved
/// peer list, serving the ledger and adding peers' blocks to it.
fn node(
    log: &Log,
    o: &Opened,
    data_dir: &Path,
    net: &Net,
) -> Result<Node<LedgerBackend>, Box<dyn Error>> {
    let socket = UdpSocket::bind(
        net.listen
            .unwrap_or_else(|| "0.0.0.0:0".parse().expect("address")),
    )?;
    let cfg = Config {
        capabilities: if net.listen.is_some() {
            cap::LISTENING
        } else {
            0
        },
        peers: net.peers.clone(),
        peer_file: Some(data_dir.join("peers")),
        sync_depth: Some(net.sync_depth),
        max_peers: net.max_peers,
        ..Config::new(node::magic(&o.genesis.hash()))
    };
    let node = Node::new(socket, cfg, LedgerBackend(o.ledger.clone()), net_log(log));
    log.line(format!(
        "{} on {} (UDP); peers: {}; at most {} peers; a new node fast-syncs from {} blocks below the best tip; \
         serving the state at the last {} blocks",
        if net.listen.is_some() { "listening" } else { "not accepting peers; sending" },
        node.local_addr(),
        if net.peers.is_empty() {
            "from the saved list and other peers".to_string()
        } else {
            net.peers.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")
        },
        net.max_peers,
        net.sync_depth,
        net.snap_sync_limit,
    ));
    Ok(node)
}

/// Keeps the node running until the process is stopped.
fn keep_serving(log: &Log, server: Server) -> Result<(), Box<dyn Error>> {
    log.line("the chain has ended; still serving it (Ctrl-C to stop)");
    server.1.join().map_err(|_| "network thread panicked")?;
    Ok(())
}

fn describe(added: Added) -> String {
    match added {
        Added::Known => "already stored".into(),
        Added::Extended => "new tip".into(),
        Added::Side => "stored on a side branch (another block got there first)".into(),
        Added::Reorg { reverted, applied } => {
            format!("new tip after a reorg (reverted {reverted}, applied {applied})")
        }
    }
}

/// Mines onto the chain in `data_dir/chain`, creating it if there's none
/// yet, until the process is stopped or the chain ends, with a node that
/// syncs and serves the chain, announces each block mined here, and adds
/// peers' blocks, so several miners can share the chain. Mining starts once
/// the node has caught up. Whenever the tip moves (a peer's block, or a
/// reorg), the block in progress is dropped and the next one built on the
/// new tip.
pub fn run(data_dir: &Path, net: &Net) -> Result<(), Box<dyn Error>> {
    let log = Log(Instant::now());
    log.line(format!(
        "mining the chain in {} with {} threads (Ctrl-C to stop; --no-mine to only follow)",
        data_dir.join("chain").display(),
        crate::par::threads()
    ));
    let o = open(&log, data_dir, net.snap_sync_limit)?;
    let (env, store, tree, data, consensus) = (&o.env, o.store, o.tree, o.data, o.consensus);
    let ledger = &o.ledger;
    let chain = &o.chain;

    let stored_seed = store.meta(&env.read_txn()?, "wallet_seed")?;
    let seed = match stored_seed {
        Some(bytes) => seed_from(&bytes).ok_or("bad wallet seed")?,
        None => {
            let mut seed = [BabyBear::ZERO; 8];
            Rng::from_os()?.fill(&mut seed);
            let mut txn = env.write_txn()?;
            store.set_meta(&mut txn, "wallet_seed", &seed_bytes(&seed))?;
            txn.commit()?;
            seed
        }
    };
    let mut n = node(&log, &o, data_dir, net)?;
    let synced = n.synced();
    let (tx, rx) = channel();
    let handle = std::thread::spawn(move || n.run(&rx, &AtomicBool::new(false)));
    let server: Server = (tx, handle);
    log.line("catching up with peers before mining");
    while !synced.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    let (tip, _) = ledger.tip()?;
    log.line(format!(
        "tip {} at height {}; genesis {} (committing to Bitcoin block {} {}), spacing {} s, \
         half-life 2^{} s, first target 2^{:.2} (~2^{:.2} hashes)",
        short(&tip.hash()),
        tip.height,
        short(&o.genesis.hash()),
        crate::genesis::BITCOIN_HEIGHT,
        crate::genesis::BITCOIN_HASH,
        consensus.spacing,
        consensus.log_half_life,
        consensus.genesis_log_target as f64 / (1 << FRAC_BITS) as f64,
        log2(pow::MAX_TARGET / header::target_of(consensus.log_target(&o.genesis))),
    ));
    log.line(format!(
        "emission: {} coins a block to height {}, then {} to height {}; {} coins in all",
        coins(consensus.first_subsidy as u128),
        consensus.first_era,
        coins(consensus.second_subsidy as u128),
        consensus.final_height(),
        coins(consensus.total_supply()),
    ));

    let wallet = Wallet::from_seed(seed);
    let mut next_index = match store.meta(&env.read_txn()?, "wallet_next")? {
        Some(b) => u32::from_le_bytes(b.as_slice().try_into().map_err(|_| "bad wallet index")?),
        None => 0,
    };
    let moved = |parent: &Header| -> Result<Option<Header>, Box<dyn Error>> {
        let (tip, _) = ledger.tip()?;
        Ok((tip.hash() != parent.hash()).then_some(tip))
    };
    // Blocks this run mined, to tell them from peers' blocks.
    let mut ours: Vec<Hash> = Vec::new();
    let whose = |ours: &[Hash], h: &Header| {
        if ours.contains(&h.hash()) {
            "our"
        } else {
            "a peer's"
        }
    };
    let lost = |ours: &[Hash], tip: &Header, when: &str| {
        log.line(format!(
            "{} block {} took height {} {when}; dropping ours and starting over",
            whose(ours, tip),
            short(&tip.hash()),
            tip.height,
        ));
    };

    loop {
        let (parent, work_before) = ledger.tip()?;
        let height = parent.height + 1;
        let Some(subsidy) = consensus.subsidy(height) else {
            log.line(format!(
                "the chain has ended at its final height {}: all {} coins are mined; \
                 shutting down",
                parent.height,
                coins(consensus.total_supply()),
            ));
            break;
        };
        log.line(format!(
            "── block {height} (on {}) ──",
            short(&parent.hash())
        ));

        // The wallet's coins, found by scanning the state's unspent
        // outputs.
        let outputs = tree.outputs(&env.read_txn()?)?;
        let held = wallet.scan(&outputs, 0..next_index);
        let mut fresh = || {
            next_index += 1;
            next_index - 1
        };

        // Transactions: spend every coin into two outputs, plus the reward.
        let mut txs = Vec::new();
        let mut fee = 0;
        if !held.is_empty() {
            let (keys, inputs) = held.iter().map(|c| wallet.input(c)).unzip();
            let total: Amount = held.iter().map(|c| c.amount).sum();
            let half = (total - FEE) / 2;
            let (a, b) = (half, total - FEE - half);
            let outputs = vec![
                wallet.address(fresh()).pay(a),
                wallet.address(fresh()).pay(b),
            ];
            let tx = Transaction::sign(TxBody { inputs, outputs }, keys)
                .map_err(|e| format!("{e:?}"))?;
            let spent: Vec<String> = held
                .iter()
                .map(|c| format!("{} @ slot {}", coins(c.amount as u128), c.position))
                .collect();
            log.line(format!(
                "tx: spends {} = {} -> {} + {} (fee {})",
                spent.join(" + "),
                coins(total as u128),
                coins(a as u128),
                coins(b as u128),
                coins(FEE as u128),
            ));
            txs.push(tx);
            fee = FEE;
        }
        let reward = wallet.address(fresh()).pay(subsidy + fee);
        log.line(format!(
            "reward: {} (subsidy {} + implicit fees {})",
            coins(reward.amount as u128),
            coins(subsidy as u128),
            coins(fee as u128),
        ));
        txs.push(
            Transaction::sign(
                TxBody {
                    inputs: vec![],
                    outputs: vec![reward],
                },
                vec![],
            )
            .map_err(|e| format!("{e:?}"))?,
        );
        {
            let mut txn = env.write_txn()?;
            store.set_meta(&mut txn, "wallet_next", &next_index.to_le_bytes())?;
            txn.commit()?;
        }

        // 1. Prove the transition: apply it in a write transaction (quick),
        // drop that, then prove without holding the database.
        let t = Instant::now();
        let prepared = {
            let mut txn = env.write_txn()?;
            let p = chain.block_prover().prepare(&tree, &mut txn, &txs, subsidy);
            txn.abort();
            p?
        };
        let new_state = prepared.new;
        let agg = chain.block_prover().prove_prepared(&prepared)?;
        log.line(format!(
            "transition proved in {:.1} s: {} inputs, {} outputs",
            t.elapsed().as_secs_f64(),
            agg.inputs.len(),
            agg.outputs.len(),
        ));
        if let Some(tip) = moved(&parent)? {
            lost(&ours, &tip, "while proving");
            continue;
        }

        // 2. The chain proof: block 1 from genesis, later blocks extend the
        // parent's proof and prove the parent header's PoW.
        let t = Instant::now();
        let proof = if parent.height == 0 {
            chain.prove_first(&agg)?
        } else {
            let txn = env.read_txn()?;
            let grandparent = store
                .header_by_hash(&txn, &parent.prev)?
                .ok_or("missing header")?;
            let work_gp = store
                .work_of(&txn, &grandparent.hash())?
                .ok_or("missing work")?;
            let prev = ChainProof {
                statement: chain.statement(&parent, work_gp),
                proof: store
                    .block_by_hash(&txn, &parent.hash())?
                    .ok_or("missing block")?
                    .proof,
            };
            drop(txn);
            chain.prove_next(&prev, &grandparent, &parent, &agg)?
        };
        let secs = t.elapsed().as_secs_f64();
        let size = proof.proof.size() / 1024;
        if height == 1 {
            log.line(format!(
                "chain proof in {secs:.1} s (first block from genesis, wrapped): {size} KB"
            ));
        } else {
            log.line(format!(
                "chain proof in {secs:.1} s (chain node, wrapped): {size} KB, covers \
                 headers 1..={} with ~2^{:.2} hashes of work",
                height - 1,
                hashes_log2(work_before),
            ));
        }
        let block = Block {
            inputs: agg.inputs,
            outputs: agg.outputs,
            proof: proof.proof,
        };
        // Proving's buffers aren't needed while mining.
        crate::alloc::release();
        if let Some(tip) = moved(&parent)? {
            lost(&ours, &tip, "while proving");
            continue;
        }

        // 3. Mine the header, which commits to the block and its proof,
        // checking for a new tip between batches of nonces.
        let mut header = Header {
            height,
            prev: parent.hash(),
            timestamp: now().max(parent.timestamp + 1),
            state: new_state,
            body: block.body_hash(),
            nonce: 0,
        };
        let log_target = consensus.log_target(&parent);
        let target = header::target_of(log_target);
        log.line(format!(
            "mining: log-target {:.4}, ~2^{:.2} expected hashes",
            log_target as f64 / (1 << FRAC_BITS) as f64,
            log2(pow::MAX_TARGET / target),
        ));
        let t = Instant::now();
        const BATCH: u64 = 1 << 18;
        let seed = header.seed();
        let mut start = 0u64;
        let nonce = loop {
            if let Some(n) = data.mine(&seed, target, start..start + BATCH) {
                break Some(n);
            }
            start += BATCH;
            if moved(&parent)?.is_some() {
                break None;
            }
        };
        let Some(nonce) = nonce else {
            lost(&ours, &ledger.tip()?.0, "while mining");
            continue;
        };
        header.nonce = nonce;
        let secs = t.elapsed().as_secs_f64();
        log.line(format!(
            "mined in {secs:.1} s: nonce {} (~{:.0} kH/s), hash {}",
            header.nonce,
            (header.nonce + 1) as f64 / secs / 1e3,
            short(&header.hash()),
        ));

        // 4. Serialize, then add the decoded block to the ledger as a peer's
        // block would be: header, chain proof, lists against the state.
        let bytes = block::encode(&header, &block);
        let (header, block) = block::decode(&bytes)?;
        log.line(format!(
            "serialized: {} bytes ({:.1}% of the {} limit)",
            bytes.len(),
            100.0 * bytes.len() as f64 / block::MAX_BLOCK_SIZE as f64,
            block::MAX_BLOCK_SIZE,
        ));
        let t = Instant::now();
        let added = ledger.add(&header, &block)?;
        ours.push(header.hash());
        let unspent = tree.outputs(&env.read_txn()?)?.len();
        log.line(format!(
            "verified (header, chain proof, lists against the state) and stored in {:.1} ms: {}; \
             {unspent} unspent outputs",
            t.elapsed().as_secs_f64() * 1e3,
            describe(added),
        ));
        let on_main = {
            let txn = env.read_txn()?;
            let mut n = 0;
            for h in &ours {
                n += usize::from(store.height_of(&txn, h)?.is_some());
            }
            n
        };
        log.line(match added {
            Added::Extended | Added::Reorg { .. } => format!(
                "our block {} won height {} ({on_main} of ours on the main chain this run)",
                short(&header.hash()),
                header.height
            ),
            _ => format!(
                "our block {} lost height {}: a peer's block got there first",
                short(&header.hash()),
                header.height
            ),
        });
        let _ = server.0.send(Event::NewTip);
    }

    // The chain has ended: sum up.

    // Which main-chain blocks are ours (a reorg may have dropped some).
    {
        let txn = env.read_txn()?;
        let tip = ledger.tip()?.0.height;
        let mut heights = Vec::new();
        for h in 1..=tip {
            if store
                .hash_at(&txn, h)?
                .is_some_and(|hash| ours.contains(&hash))
            {
                heights.push(h.to_string());
            }
        }
        log.line(format!(
            "this run mined {} of the {tip} main-chain blocks{}{}",
            heights.len(),
            if heights.is_empty() { "" } else { ": heights " },
            heights.join(", "),
        ));
    }

    // Restore the wallet from its seed alone: scan the unspent outputs (what
    // a fast-syncing node downloads) for its tags.
    let txn = env.read_txn()?;
    let t = Instant::now();
    let outputs = tree.outputs(&txn)?;
    let restored = Wallet::from_seed(seed).scan(&outputs, 0..next_index + 100);
    let balance: u128 = restored.iter().map(|c| c.amount as u128).sum();
    log.line(format!(
        "wallet restored from its seed in {:.1} ms: {} of {} unspent outputs are ours, \
         balance {} coins",
        t.elapsed().as_secs_f64() * 1e3,
        restored.len(),
        outputs.len(),
        coins(balance),
    ));

    // A node fast-syncing from the tip needs only the last two headers and
    // the tip's proof (plus the state tree, checked against the tip's root).
    let tip = store.tip(&txn)?.ok_or("empty store")?;
    if tip.height >= 1 {
        let tip_parent = store
            .header(&txn, tip.height - 1)?
            .ok_or("missing header")?;
        let claimed = store.work(&txn, tip.height - 1)?.ok_or("missing work")?;
        let proof = store
            .block(&txn, tip.height)?
            .ok_or("missing tip block")?
            .proof;
        let t = Instant::now();
        if tip.height >= 2 {
            header::check(&consensus, data, &tip_parent, &tip)?;
        }
        chain
            .verify_proof(&tip, claimed, &proof)
            .map_err(|e| format!("tip proof: {e}"))?;
        log.line(format!(
            "fast sync: tip {} at height {} checked from headers {} and {} and one proof in \
             {:.1} ms; ~2^{:.2} hashes of total work",
            short(&tip.hash()),
            tip.height,
            tip.height - 1,
            tip.height,
            t.elapsed().as_secs_f64() * 1e3,
            hashes_log2(store.work(&txn, tip.height)?.ok_or("missing work")?),
        ));
    }
    drop(txn);
    keep_serving(&log, server)
}

/// With `--no-mine`: runs a node that follows the chain in `data_dir/chain`:
/// syncs it (fast-syncing if new), fully verifying each block and following
/// the branch with the most work, and serves it to others. Runs until the
/// process is stopped.
pub fn follow(data_dir: &Path, net: &Net) -> Result<(), Box<dyn Error>> {
    let log = Log(Instant::now());
    log.line(format!(
        "following the chain in {} without mining",
        data_dir.join("chain").display()
    ));
    let o = open(&log, data_dir, net.snap_sync_limit)?;
    let (tip, _) = o.ledger.tip()?;
    log.line(format!(
        "local tip {} at height {} in {}",
        short(&tip.hash()),
        tip.height,
        o.dir.display()
    ));
    let mut n = node(&log, &o, data_dir, net)?;
    let (_tx, rx) = channel();
    n.run(&rx, &AtomicBool::new(false));
    Ok(())
}

/// Prints the stored chain in `dir`.
pub fn status(dir: &Path) -> Result<(), Box<dyn Error>> {
    if !dir.exists() {
        println!(
            "no chain in {}; run without --status to create it",
            dir.display()
        );
        return Ok(());
    }
    let env = db::open_env(dir)?;
    let tree = StateTree::open(&env)?;
    let store = ChainStore::open(&env)?;
    let txn = env.read_txn()?;
    let Some(tip) = store.tip(&txn)? else {
        println!(
            "no chain in {}; run without --status to create it",
            dir.display()
        );
        return Ok(());
    };
    let genesis = store.genesis(&txn)?.ok_or("missing genesis")?;
    let consensus = Consensus::DEFAULT;
    let state = tree.state(&txn)?;
    let work = store.work(&txn, tip.height)?.ok_or("missing work")?;
    println!("chain in {}", dir.display());
    println!(
        "  genesis    {} at {}{}",
        short(&genesis.hash()),
        genesis.timestamp,
        if genesis == crate::genesis::header() {
            format!(
                ", committing to Bitcoin block {} {}",
                crate::genesis::BITCOIN_HEIGHT,
                crate::genesis::BITCOIN_HASH
            )
        } else {
            " (not this build's genesis)".to_string()
        }
    );
    println!(
        "  tip        {} at height {}, time {}",
        short(&tip.hash()),
        tip.height,
        tip.timestamp
    );
    if tip.height > 0 {
        println!("  work       ~2^{:.2} hashes", hashes_log2(work));
    }
    let base = store.base(&txn)?;
    if base > 0 {
        println!("  history    fast-synced: blocks from height {base} up");
    }
    if let Some(from) = tree.history_from(&txn)? {
        println!(
            "  serves     the state at heights {} to {} (to fast-syncing nodes)",
            from.max(base).max(1).min(tip.height),
            tip.height
        );
    }
    if let Some(tag) = tree.sync_tag(&txn)?
        && let Some(height) = tag.get(..8).and_then(|h| h.try_into().ok())
    {
        println!(
            "  download   an unfinished state download (height {}) resumes on the next run",
            u64::from_be_bytes(height)
        );
    }
    println!(
        "  state      {} unspent outputs in {} slots; {}",
        tree.outputs(&txn)?.len(),
        state.size - 1,
        if state == tip.state {
            "matches the tip header"
        } else {
            "DOES NOT match the tip header"
        }
    );
    println!(
        "  supply     {} of {} coins issued",
        coins(supply(&consensus, tip.height)),
        coins(consensus.total_supply())
    );
    match consensus.subsidy(tip.height + 1) {
        Some(s) => println!(
            "  next       block {} creates {} coins; the chain ends at height {}",
            tip.height + 1,
            coins(s as u128),
            consensus.final_height()
        ),
        None => println!("  the chain has ended"),
    }
    Ok(())
}
