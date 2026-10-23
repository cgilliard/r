//! The node: follows, serves and (unless `--no-mine`) mines the chain until
//! stopped, with a wallet at its prompt ([`crate::cli`]). The chain (blocks
//! on every branch, the state tree) and the wallet live in LMDB in
//! `<data dir>/chain`, so each run carries on from the last; the log goes to
//! `<data dir>/node.log`.
//!
//! Three threads share the ledger and the mempool:
//!
//! - the network node ([`crate::net`]) finds peers (from `--peer`, the saved
//!   peer list (in the chain database), and what peers tell it, or else the
//!   [`SEEDS`]), syncs the
//!   chain (a new node fast-syncs from `--sync-depth` blocks below the best
//!   tip), serves it, relays transactions, and adds peers' blocks to the
//!   ledger, which follows the branch with the most work;
//! - the miner, once the node has caught up, builds each block from the
//!   mempool's best-paying transactions plus a reward (the subsidy and their
//!   fees) to a fresh key of the wallet, and:
//!   1. proves the block's state transition ([`BlockProver`]), holding the
//!      database only for the quick native part;
//!   2. proves its chain proof ([`Chain`]), covering the parent header's PoW
//!      and every earlier block;
//!   3. builds the header and mines it at the ASERT target;
//!   4. serializes the block and adds the decoded copy to the ledger as a
//!      peer's block would be ([`Ledger::add`]).
//!
//!   Whenever the tip moves (a peer's block or a reorg), it drops the block
//!   in progress (checked after each proving step and between batches of
//!   nonces) and starts on the new tip. Rewards follow the emission schedule
//!   ([`Consensus::subsidy`]); at the final height the chain ends;
//! - the main thread reads wallet commands until `quit`.

use std::error::Error;
use std::fs::File;
use std::io::Write as _;
use std::net::{SocketAddr, UdpSocket};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::aux_leaves::AuxStore;
use crate::block::{self, Block, BlockProver};
use crate::chain::{self, Chain, ChainProof, ChainVerifier};
use crate::cli::Cli;
use crate::db;
use crate::fri::FriParams;
use crate::hash::{self, Hash};
use crate::header::{self, Consensus, FRAC_BITS, Header};
use crate::ledger::{Added, Ledger, Verifier};
use crate::mempool::Mempool;
use crate::net::LedgerBackend;
use crate::net::node::PostConfig;
use crate::net::node::{self, Config, Event, Node};
use crate::net::peers::PeerStore;
use crate::net::wire::cap;
use crate::output::{Amount, format_coins};
use crate::payments::PaymentKeys;
use crate::pow::{self, Dataset, DatasetTree};
use crate::preimages::PreimageStore;
use crate::stark::KeyCache;
use crate::state::StateTree;
use crate::store::ChainStore;
use crate::transaction::{Transaction, TxBody};
use crate::wallet::{Wallet, WalletStore};

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
    /// Blocks below the tip whose bodies are kept (0: all).
    pub keep_blocks: u64,
    /// Hosts that pass payments to and from our address (`host:port`; the
    /// [`SEEDS`] if none).
    pub slate_hosts: Vec<String>,
}

/// An amount in coins.
fn coins(units: u128) -> String {
    format_coins(units)
}

/// Coins created by blocks `1 ..= height`.
fn supply(c: &Consensus, height: u64) -> u128 {
    let first = height.min(c.first_era) as u128;
    let second = height.min(c.final_height()).saturating_sub(c.first_era) as u128;
    first * c.first_subsidy as u128 + second * c.second_subsidy as u128
}

fn provers(consensus: Consensus, genesis: Header, tree: DatasetTree, keys: &KeyCache) -> Chain {
    let blocks = BlockProver::standard(FriParams::default(), keys);
    Chain::new(consensus, genesis, blocks, tree, keys)
}

/// The log file, each line stamped with the time since the start.
#[derive(Clone)]
struct Log {
    start: Instant,
    file: Arc<Mutex<File>>,
}

impl Log {
    /// Appends to the log at `path`.
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            start: Instant::now(),
            file: Arc::new(Mutex::new(file)),
        })
    }

    fn line(&self, msg: impl AsRef<str>) {
        let mut f = self.file.lock().expect("log lock");
        let _ = writeln!(
            f,
            "[{:>8.1}s] {}",
            self.start.elapsed().as_secs_f64(),
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

/// The network node's log lines.
fn net_log(log: &Log) -> Box<dyn Fn(String) + Send> {
    let log = log.clone();
    Box::new(move |line| log.line(format!("net: {line}")))
}

/// The chain in `data_dir/chain`, open: its ledger and what's around it.
struct Opened {
    env: heed::Env,
    store: ChainStore,
    tree: StateTree,
    /// The PoW dataset, if mining.
    data: Option<&'static Dataset>,
    genesis: Header,
    consensus: Consensus,
    ledger: Arc<Ledger>,
    /// The provers, if mining (with proofs).
    chain: Option<Arc<Chain>>,
    /// `--debug-no-proof-validation`.
    debug_no_proofs: bool,
}

/// Opens (creating if needed) the chain in `data_dir/chain`, with the
/// PoW dataset and proving setup (if mining) cached in `data_dir/cache`,
/// and sets up the provers (if mining) and the verifier.
fn open(
    log: &Log,
    data_dir: &Path,
    snap_sync_limit: u64,
    keep_blocks: u64,
    mine: bool,
    debug_no_proofs: bool,
) -> Result<Opened, Box<dyn Error>> {
    let dir = data_dir.join("chain");
    let cache = data_dir.join("cache");
    std::fs::create_dir_all(&cache)?;
    // Only a miner needs the dataset: others check headers' PoW from the
    // cache in the binary.
    let (data, pow_tree) = if mine {
        let (data, tree, loaded) = Dataset::load(&cache.join("pow-dataset"));
        log.line(if loaded {
            "PoW dataset (64 MiB) loaded and checked against its root"
        } else {
            "PoW dataset (64 MiB) built and cached"
        });
        // Without proofs, nothing needs the tree.
        (
            Some(data),
            (!debug_no_proofs).then_some(DatasetTree::Full(tree)),
        )
    } else {
        (None, None)
    };
    let env = db::open_env(&dir)?;
    let tree = StateTree::open(&env)?;
    let store = ChainStore::open(&env)?;
    let genesis = crate::genesis::header();
    let stored = store.genesis(&env.read_txn()?)?;
    // A chain made without proofs is marked, so it's never taken for a real
    // one (or the other way round).
    let marker = dir.join(DEBUG_MARKER);
    match (stored.is_some(), marker.exists(), debug_no_proofs) {
        (true, false, true) => {
            return Err(format!(
                "{} holds a chain with proofs; use another --data-dir for \
                 --debug-no-proof-validation",
                dir.display()
            )
            .into());
        }
        (_, true, false) => {
            return Err(format!(
                "{} holds a chain made with --debug-no-proof-validation (its blocks have no \
                 proofs); use another --data-dir",
                dir.display()
            )
            .into());
        }
        (false, false, true) => std::fs::write(&marker, "blocks here have no proofs\n")?,
        _ => {}
    }
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
    // Only a miner needs the provers; others check chain proofs with the
    // pinned keys.
    let (chain, verifier): (Option<Arc<Chain>>, Arc<dyn Verifier>) = match pow_tree {
        _ if debug_no_proofs => {
            log.line(
                "--debug-no-proof-validation: blocks are made and checked without chain proofs \
                 (testing only; a separate network)",
            );
            (None, Arc::new(chain::NoProofs))
        }
        Some(pow_tree) => {
            log.line(
                "setting up the provers and verifier (item chunks of 682/2730, signature chunks \
                 of 7/30/120, condition chunks of 9/39, 16-op segments, nodes at 2^15)",
            );
            let keys = KeyCache::open(cache.join("keys"));
            let chain = provers(consensus, genesis, pow_tree, &keys);
            if let Err(e) = keys.save() {
                log.line(format!("couldn't save the key cache: {e}"));
            }
            if chain.verifier().keys() != chain::KEYS {
                return Err(
                    "the circuits' keys aren't the pinned ones (chain::KEYS): this build \
                            can't make proofs other nodes accept"
                        .into(),
                );
            }
            let ([first_rows, chain_rows, wrap_rows], log_height) = chain.used_rows();
            log.line(format!(
                "ready (proving keys are built on first use); chain circuits use {first_rows} \
                 (first), {chain_rows} (chain) rows of 2^{log_height}, wrap {wrap_rows} of 2^{}",
                chain::WRAP_LOG_HEIGHT
            ));
            let chain = Arc::new(chain);
            (Some(chain.clone()), chain)
        }
        None => {
            log.line("ready: checking chain proofs with the pinned keys (no provers: not mining)");
            (None, Arc::new(ChainVerifier::new(chain::KEYS)))
        }
    };
    let ledger = Arc::new(Ledger {
        env: env.clone(),
        store,
        tree,
        consensus,
        verifier,
        snap_sync_limit,
        keep_blocks,
    });
    ledger.start_history()?;
    Ok(Opened {
        env,
        store,
        tree,
        data,
        genesis,
        consensus,
        ledger,
        chain,
        debug_no_proofs,
    })
}

/// The file marking a chain directory made with `--debug-no-proof-validation`.
const DEBUG_MARKER: &str = "debug-no-proofs";

/// The seed nodes a node dials when it has no one else to: a new node with
/// no `--peer`, or one whose saved peers are all gone.
pub const SEEDS: [&str; 2] = ["tabernacle1.root.sx:3737", "tabernacle2.root.sx:3737"];

/// A node on `net.listen` (or any port), dialing `net.peers` and the saved
/// peer list (kept in the chain database; or the [`SEEDS`], when it has no
/// one else to dial), serving the ledger and mempool and adding peers'
/// blocks and transactions to them.
fn node(
    log: &Log,
    o: &Opened,
    net: &Net,
    mempool: Arc<Mempool>,
    post: PostConfig,
) -> Result<Node<LedgerBackend>, Box<dyn Error>> {
    let addr = net
        .listen
        .unwrap_or_else(|| "0.0.0.0:0".parse().expect("address"));
    let socket = UdpSocket::bind(addr).map_err(|e| {
        format!(
            "can't listen on {addr}: {e} (another node on this machine needs its own --listen \
             address, or --no-listen)"
        )
    })?;
    // Nodes without proofs get their own network: the real magic, inverted.
    let mut magic = node::magic(&o.genesis.hash());
    if o.debug_no_proofs {
        magic = magic.map(|b| !b);
    }
    let cfg = Config {
        capabilities: if net.listen.is_some() {
            cap::LISTENING
        } else {
            0
        },
        peers: net.peers.clone(),
        // The seeds are on the real network.
        seeds: if o.debug_no_proofs {
            vec![]
        } else {
            SEEDS.map(String::from).to_vec()
        },
        peer_store: Some(PeerStore::open(&o.ledger.env)?),
        // Serve and fetch aux trees' preimages (advertising cap::TREE).
        preimages: Some(PreimageStore::open(&o.ledger.env)?),
        post: Some(post),
        sync_depth: Some(net.sync_depth),
        max_peers: net.max_peers,
        ..Config::new(magic)
    };
    let backend = LedgerBackend {
        ledger: o.ledger.clone(),
        mempool,
    };
    let node = Node::new(socket, cfg, backend, net_log(log));
    log.line(format!(
        "{} on {} (UDP); peers: {}; at most {} peers; a new node fast-syncs from {} blocks below the best tip; \
         serving the state at the last {} blocks; keeping {}",
        if net.listen.is_some() { "listening" } else { "not accepting peers; sending" },
        node.local_addr(),
        if net.peers.is_empty() {
            format!(
                "from the saved list and other peers, or the seeds ({})",
                SEEDS.join(", ")
            )
        } else {
            net.peers.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")
        },
        net.max_peers,
        net.sync_depth,
        net.snap_sync_limit,
        match net.keep_blocks {
            0 => "every block".to_string(),
            n => format!("the last {} blocks", n.max(net.snap_sync_limit)),
        },
    ));
    Ok(node)
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

/// What the node does besides following the chain.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub mine: bool,
    /// Confirmations before a mining reward of ours can be spent.
    pub coin_maturity: u64,
    /// Restore the wallet from recovery words read from the terminal.
    pub restore: bool,
    /// Testing: make and check no chain proofs (`--debug-no-proof-validation`).
    pub debug_no_proofs: bool,
    /// Answer payments to our address as they come (else they wait for
    /// `accept`).
    pub auto_receive: bool,
}

/// Runs the node on the chain in `data_dir/chain` (creating it if there's
/// none yet), mining if asked, with the wallet prompt on the terminal, until
/// `quit`. If the input ends (no terminal), it runs until the process is
/// stopped.
pub fn run(data_dir: &Path, net: &Net, opts: &Options) -> Result<(), Box<dyn Error>> {
    let mine = opts.mine;
    std::fs::create_dir_all(data_dir)?;
    let log_path = data_dir.join("node.log");
    let log = Log::open(&log_path)?;
    println!(
        "r {}: {} the chain in {}; log in {} (tail -f to watch)",
        env!("CARGO_PKG_VERSION"),
        if mine {
            "mining"
        } else {
            "following (not mining)"
        },
        data_dir.join("chain").display(),
        log_path.display(),
    );
    println!(
        "starting up{}...",
        if mine {
            " (the PoW dataset and provers take a moment)"
        } else {
            ""
        }
    );
    log.line(format!(
        "starting: {} the chain in {}{}",
        if mine {
            "mining"
        } else {
            "following, without mining,"
        },
        data_dir.join("chain").display(),
        if mine {
            format!(" with {} threads", crate::par::threads())
        } else {
            String::new()
        },
    ));
    let o = Arc::new(open(
        &log,
        data_dir,
        net.snap_sync_limit,
        net.keep_blocks,
        mine,
        opts.debug_no_proofs,
    )?);
    let mempool = Arc::new(Mempool::new());
    let wallet_store = WalletStore::open(&o.env)?;
    let mut wallet = open_wallet(&o.env, &wallet_store, opts.restore, &log)?;
    if opts.debug_no_proofs {
        wallet = wallet.with_cheap_trees();
    }
    let stop = Arc::new(AtomicBool::new(false));

    // Payments to and from our address pass through hosts: the seeds
    // (which are on the real network), unless others are given.
    let entropy = {
        let txn = o.env.read_txn()?;
        wallet_store.entropy(&txn)?.ok_or("no wallet")?
    };
    let payments = PaymentKeys::from_entropy(&entropy);
    let hosts = if !net.slate_hosts.is_empty() {
        net.slate_hosts.clone()
    } else if opts.debug_no_proofs {
        vec![]
    } else {
        SEEDS.map(String::from).to_vec()
    };
    log.line(format!(
        "payment address {}; payments pass through {}; {}",
        payments.address,
        if hosts.is_empty() {
            "no hosts (give --slate-host)".to_string()
        } else {
            hosts.join(", ")
        },
        if opts.auto_receive {
            "answering payment requests as they come"
        } else {
            "holding payment requests until accepted"
        }
    ));
    let (post_events, post_rx) = channel();
    let post = PostConfig {
        keys: payments.clone(),
        hosts,
        events: post_events,
    };

    let mut n = node(&log, &o, net, mempool.clone(), post)?;
    let synced = n.synced();
    let (events, rx) = channel();
    let node_thread = {
        let stop = stop.clone();
        thread::spawn(move || n.run(&rx, &stop))
    };
    let miner = mine.then(|| {
        let (o, log, mempool, events, synced, stop) = (
            o.clone(),
            log.clone(),
            mempool.clone(),
            events.clone(),
            synced.clone(),
            stop.clone(),
        );
        thread::spawn(move || {
            let m = Miner {
                o: &o,
                log: &log,
                mempool: &mempool,
                wallet_store,
                wallet,
                events,
            };
            if let Err(e) = m.run(&synced, &stop) {
                log.line(format!("mining stopped: {e}"));
                eprintln!("mining stopped: {e} (see the log)");
            }
        })
    });

    let cli = Arc::new(Cli {
        ledger: o.ledger.clone(),
        mempool,
        events,
        store: wallet_store,
        wallet,
        synced,
        maturity: opts.coin_maturity,
        debug: opts.debug_no_proofs,
        aux: AuxStore::open(&o.ledger.env)?,
        preimages: PreimageStore::open(&o.ledger.env)?,
        payments,
        auto_receive: opts.auto_receive,
        incoming: Default::default(),
        log: {
            let log = log.clone();
            Box::new(move |line| log.line(line))
        },
    });
    // What the post brings: payment requests to answer, answers to
    // finalize, failures.
    {
        let (cli, log) = (cli.clone(), log.clone());
        thread::spawn(move || {
            for event in post_rx {
                let text = cli.on_post(event);
                if !text.is_empty() {
                    log.line(format!("wallet: {text}"));
                    println!("{text}");
                }
            }
        });
    }
    // Signed payments not in a block yet go back to the mempool whenever
    // they've dropped out of it (say, a peer's reorg).
    {
        let (cli, stop) = (cli.clone(), stop.clone());
        thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(30));
                let _ = cli.resubmit();
            }
        });
    }
    match cli.resubmit() {
        Ok(0) => {}
        Ok(n) => println!(
            "submitted {n} signed payment{} again",
            if n == 1 { "" } else { "s" }
        ),
        Err(e) => println!("couldn't resubmit signed payments: {e}"),
    }
    println!("ready; type help for commands");
    if !cli.run(std::io::stdin().lock(), std::io::stdout()) {
        log.line("the input ended; running until stopped");
        let _ = node_thread.join();
        return Ok(());
    }
    log.line("stopping");
    stop.store(true, Ordering::Relaxed);
    // The node stops within a tick (saving its peer list). The miner isn't
    // waited for: the block in progress is just dropped, and every database
    // write is a transaction, so stopping mid-step loses nothing else.
    let _ = node_thread.join();
    drop(miner);
    log.line("stopped");
    Ok(())
}

/// The wallet in `store`: restored from words read from the terminal if
/// `restore`, or made the first time (showing its recovery words once).
fn open_wallet(
    env: &heed::Env,
    store: &WalletStore,
    restore: bool,
    log: &Log,
) -> Result<Wallet, Box<dyn Error>> {
    let existing = store.entropy(&env.read_txn()?)?;
    let entropy = match (existing, restore) {
        (Some(_), true) => {
            return Err(
                "this data directory has a wallet already; restore into a new --data-dir".into(),
            );
        }
        (Some(e), false) => e,
        (None, true) => {
            println!("enter your {} recovery words:", crate::mnemonic::WORDS);
            // Pasted words may arrive over several lines.
            let mut words = String::new();
            while words.split_whitespace().count() < crate::mnemonic::WORDS {
                if std::io::stdin().read_line(&mut words)? == 0 {
                    break;
                }
            }
            let entropy = crate::mnemonic::from_words(&words)?;
            let mut txn = env.write_txn()?;
            store.set_entropy(&mut txn, &entropy)?;
            store.set_restoring(&mut txn, true)?;
            txn.commit()?;
            println!("wallet restored; it finds its coins once the node has caught up (see info)");
            log.line("wallet restored from recovery words");
            entropy
        }
        (None, false) => {
            let mut entropy = [0u8; crate::mnemonic::ENTROPY_LEN];
            std::io::Read::read_exact(&mut File::open("/dev/urandom")?, &mut entropy)?;
            let mut txn = env.write_txn()?;
            store.set_entropy(&mut txn, &entropy)?;
            txn.commit()?;
            println!(
                "a new wallet was made. Write down its recovery words and keep them offline: \
                 they restore it (r --restore), and anyone with them can spend its coins. \
                 (Show them again with: seed)\n{}",
                crate::cli::words_block(&entropy)
            );
            log.line("new wallet made");
            entropy
        }
    };
    let wallet = Wallet::from_seed(crate::mnemonic::seed_of(&entropy));
    Ok(wallet)
}

/// The miner's thread: what it shares with the rest of the node.
struct Miner<'a> {
    o: &'a Opened,
    log: &'a Log,
    mempool: &'a Mempool,
    wallet_store: WalletStore,
    wallet: Wallet,
    events: Sender<Event>,
}

impl Miner<'_> {
    /// Mines onto the tip once the node has caught up, until `stop` is set
    /// or the chain ends.
    fn run(&self, synced: &AtomicBool, stop: &AtomicBool) -> Result<(), Box<dyn Error>> {
        let Self {
            o,
            log,
            mempool,
            wallet_store,
            wallet,
            events,
        } = self;
        let (env, store, tree, consensus) = (&o.env, o.store, o.tree, o.consensus);
        let aux = AuxStore::open(env)?;
        let preimages = PreimageStore::open(env)?;
        let data = o.data.expect("a miner has the dataset");
        let ledger = &o.ledger;
        // No provers with --debug-no-proof-validation.
        let chain = o.chain.as_deref();
        log.line("catching up with peers before mining");
        while !synced.load(Ordering::Relaxed) {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(100));
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
        // Room for the transactions' lists, beside the reward and the proof.
        let proof_size = chain.map_or(block::no_proof().size(), |c| {
            c.block_prover().aggregator().proof_size()
        });
        let room = block::MAX_BLOCK_SIZE.saturating_sub(block::encoded_size(0, 1, proof_size));
        let moved = |parent: &Header| -> Result<Option<Header>, Box<dyn Error>> {
            let (tip, _) = ledger.tip()?;
            Ok((tip.hash() != parent.hash()).then_some(tip))
        };
        // Blocks this run mined, to tell them from peers' blocks.
        let mut ours: Vec<Hash> = Vec::new();
        let mut reward_keys: Option<crate::wallet::OutputKeys> = None;
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

        while !stop.load(Ordering::Relaxed) {
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

            // Transactions from the mempool, the best-paying first, and the
            // reward (subsidy and fees) to a fresh key of the wallet.
            // Without proving (--debug-no-proof-validation), as many as fit.
            let picked = match chain {
                Some(_) => mempool.select(&**ledger, room),
                None => mempool.select_up_to(&**ledger, room, usize::MAX),
            };
            let fees: Amount = picked.iter().map(|(_, _, fee)| fee).sum();
            let mut txs: Vec<Transaction> = picked.iter().map(|(_, tx, _)| tx.clone()).collect();
            // The same keys for the reward until a block of ours is
            // finished (an attempt that never finished published nothing,
            // so its keys are fine to use again).
            let keys = match reward_keys {
                Some(k) => k,
                None => {
                    let k = wallet.new_keys()?;
                    let mut txn = env.write_txn()?;
                    wallet_store.mark_reward(&mut txn, &k.r)?;
                    txn.commit()?;
                    reward_keys = Some(k);
                    k
                }
            };
            let reward = keys.pay(subsidy + fees);
            log.line(format!(
                "{} transaction{} from the mempool (fees {}); reward {} to our wallet",
                picked.len(),
                if picked.len() == 1 { "" } else { "s" },
                coins(fees as u128),
                coins(reward.amount as u128),
            ));
            txs.push(
                Transaction::sign(TxBody::new(vec![], vec![reward]), vec![])
                    .map_err(|e| format!("{e:?}"))?,
            );

            // 1. Prove the transition: apply it in a write transaction (quick),
            // drop that, then prove without holding the database.
            let t = Instant::now();
            let prepared = {
                let mut txn = env.write_txn()?;
                let p = match chain {
                    Some(c) => c
                        .block_prover()
                        .prepare(&tree, &mut txn, &txs, height, subsidy),
                    None => block::prepare(
                        &tree,
                        &mut txn,
                        &txs,
                        height,
                        subsidy,
                        &block::STANDARD_SEGMENT,
                        proof_size,
                    ),
                };
                txn.abort();
                match p {
                    Ok(p) => p,
                    // A transaction the state no longer takes (the pool checks
                    // them, so this is rare): drop the lot and start over.
                    Err(e) if !picked.is_empty() => {
                        log.line(format!(
                            "couldn't build the block with those transactions ({e}); dropping them"
                        ));
                        let ids: Vec<Hash> = picked.iter().map(|(id, _, _)| *id).collect();
                        mempool.remove(&ids);
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                }
            };
            let new_state = prepared.new;
            let block = match chain {
                None => {
                    log.line(format!(
                        "no proof (--debug-no-proof-validation): {} inputs, {} outputs",
                        prepared.inputs.len(),
                        prepared.outputs.len(),
                    ));
                    Block {
                        inputs: prepared.inputs,
                        outputs: prepared.outputs,
                        proof: block::no_proof(),
                    }
                }
                Some(chain) => {
                    let agg = chain.block_prover().prove_prepared(&prepared)?;
                    log.line(format!(
                        "transition proved in {:.1} s: {} inputs, {} outputs",
                        t.elapsed().as_secs_f64(),
                        agg.inputs.len(),
                        agg.outputs.len(),
                    ));
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
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
                    Block {
                        inputs: agg.inputs,
                        outputs: agg.outputs,
                        proof: proof.proof,
                    }
                }
            };
            // Proving's buffers aren't needed while mining.
            crate::alloc::release();
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if let Some(tip) = moved(&parent)? {
                lost(&ours, &tip, "while proving");
                continue;
            }

            // 3. Mine the header, which commits to the block and its proof,
            // checking for a new tip between batches of nonces. It commits
            // to the aux leaves as they are now, whose tree (and files) we
            // keep to serve to peers.
            let leaves = aux.leaves(env)?;
            preimages.put_tree(&leaves, &aux.files(env)?, height)?;
            let mut header = Header {
                height,
                prev: parent.hash(),
                timestamp: now().max(parent.timestamp + 1),
                state: new_state,
                body: block.body_hash(),
                aux_hash: crate::aux_leaves::root(&leaves),
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
                if stop.load(Ordering::Relaxed) || moved(&parent)?.is_some() {
                    break None;
                }
            };
            let Some(nonce) = nonce else {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
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
            // Finished and stored, won or lost: its output may get out, so
            // the next block takes new keys.
            reward_keys = None;
            log.line(match added {
                Added::Extended | Added::Reorg { .. } => format!(
                    "our block {} won height {} ({on_main} of ours on the main chain this \
                     run){}",
                    short(&header.hash()),
                    header.height,
                    if header.aux_hash == [0; 32] {
                        String::new()
                    } else {
                        format!(", aux root {}", crate::wallet::hex(&header.aux_hash))
                    }
                ),
                _ => format!(
                    "our block {} lost height {}: a peer's block got there first",
                    short(&header.hash()),
                    header.height
                ),
            });
            let _ = events.send(Event::NewTip);
        }

        if stop.load(Ordering::Relaxed) {
            return Ok(());
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

        // A node fast-syncing from the tip needs only the last two headers and
        // the tip's proof (plus the state tree, checked against the tip's root).
        let txn = env.read_txn()?;
        let tip = store.tip(&txn)?.ok_or("empty store")?;
        if let (Some(chain), true) = (chain, tip.height >= 1) {
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
                header::check(&consensus, &tip_parent, &tip)?;
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
        log.line("the chain has ended; still serving it");
        Ok(())
    }
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
