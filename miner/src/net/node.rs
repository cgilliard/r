//! A peer-to-peer node over UDP: handshakes, peer discovery, tips, header
//! sync, fast sync, header-first block relay and chunked downloads
//! ([`super::wire`], [`crate::chunk`]).
//!
//! **Handshake.** A dialing node sends a padded `HELLO`. The other side
//! answers with a small `COOKIE` (a keyed hash of the dialer's address and
//! the minute) and keeps no state, so spoofed `HELLO`s cost it a hash and
//! can't make it send more than it received. The dialer repeats its
//! `HELLO` with the cookie, proving it receives packets at its address; only
//! then does the other side create a session and answer `WELCOME`. Every
//! later packet carries the session and must come from the session's
//! address.
//!
//! **Peers.** A node dials its configured peers (redialing them whenever
//! they're lost) and keeps up to [`Config::outbound`] outbound connections
//! in all, filling the rest from an address book. The book holds nodes that
//! accept connections: ones it dialed, ones that connected advertising
//! [`cap::LISTENING`], and addresses learned from peers (`GET_PEERS` /
//! `PEERS`, asked of every new outbound peer and now and then of a random
//! one). Learned addresses are tried, and dropped after a few failures; the
//! ones that worked are saved to [`Config::peer_file`]. So that one hostile
//! peer can't fill the book with made-up addresses, a `PEERS` we didn't ask
//! for is ignored, one peer's IP can have at most `MAX_FROM_ONE`
//! untried addresses in it, and when it's full, new addresses push out
//! untried ones (never ones that worked). At most
//! [`Config::max_peers`] sessions are open. A node that's cut off (no
//! session and nothing to dial: a new node with an empty book, or one whose
//! saved addresses have all gone) looks up its [`Config::seeds`] by name and
//! dials them, at most every `SEED_EVERY`; once connected it learns other
//! peers from them, so the seeds aren't kept as connections.
//!
//! **Sync.** A following node asks a peer whose tip is ahead for headers
//! after its own last one (`GET_HEADERS`), checks each against its parent
//! ([`Backend::check_header`]: link, time, ASERT target, PoW), and queues
//! them. It then downloads queued bodies one at a time: `GET_BLOCK` gives
//! the body's size and chunk root, which must hash to the header's body
//! hash; the body's chunks are pulled with `GET_BLOCK_CHUNKS` in a window
//! that grows by one per chunk received and halves on a loss. Every chunk
//! is checked on arrival. A complete body goes to [`Backend::accept`] (full
//! verification), and the header is announced to the other peers
//! (`HEADER`), who fetch the body the same way. A peer that sends a bad
//! chunk, a body that doesn't verify, or a header that doesn't check is
//! dropped.
//!
//! **Fast sync.** A new node (genesis only) with a [`Config::sync_depth`]
//! of `d` doesn't download the whole chain when the best peer's tip `H` is
//! more than `d` blocks up: it asks that peer whether it can serve the state
//! at height `H - d` (`GET_SNAPSHOT`). The answer (`SNAPSHOT_INFO`) gives
//! that block (the base), its parent and the parent's total work. The node
//! checks the base's header against the parent, downloads the base block
//! and checks its chain proof, which covers everything before it
//! ([`Backend::check_base`]). It then downloads the state tree at the base
//! from the root down (`GET_STATE`), spreading requests over every peer
//! that's far enough along; each reply is checked against its node's hash
//! as it's stored, so a peer that sends a wrong one is caught and dropped,
//! and what's stored survives a restart (the download resumes). Once the
//! tree is complete it becomes the state ([`Backend::install`]), and the
//! node syncs block by block from the base. A peer serves the state at its
//! last few blocks only (as many as it keeps history for), answering
//! `NO_SNAPSHOT` or `NO_STATE` otherwise.
//!
//! **Transactions.** A node announces the ids of transactions it adds to
//! its mempool (`TX_INV`: on connecting, everything in it; then each new
//! one, to every peer but the one it came from). A peer that doesn't know
//! an id fetches it (`GET_TX`, then its chunks, checked against the id),
//! several at a time, and adds it through [`Backend::add_tx`]; a peer whose
//! transaction is invalid in itself is dropped.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::wire::{
    self, COOKIE_LEN, MAX_ADDRS, MAX_HEADERS, MAX_LOCATOR, MAX_PACKET, MAX_RANGES, MAX_STATE_DATA,
    MAX_TX_IDS, Message, Packet, Tip, cap,
};
use crate::babybear::BabyBear;
use crate::block::MAX_BLOCK_SIZE;
use crate::chunk::{self, Assembly, ChunkError, Served};
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::header::{HEADER_BYTES, Header};
use crate::ledger::{Added, Snapshot};
use crate::rng::Rng;
use crate::state::SyncPut;

/// What a node needs from the chain it serves and follows.
pub trait Backend: Sync {
    /// The main chain's tip.
    fn tip(&self) -> Tip;
    /// The main chain's header at `height`.
    fn header_at(&self, height: u64) -> Option<Header>;
    /// The height of `hash` if it's on the main chain.
    fn height_of(&self, hash: &Hash) -> Option<u64>;
    /// Any stored header.
    fn header_by_hash(&self, hash: &Hash) -> Option<Header>;
    /// The encoded body ([`crate::block::encode_body`]) of any stored block
    /// with this header hash.
    fn body(&self, hash: &Hash) -> Option<Vec<u8>>;
    /// The header rules against its parent (link, time, target, PoW), and
    /// that neither is known to be invalid.
    fn check_header(&self, parent: &Header, header: &Header) -> Result<(), String>;
    /// Fully checks and stores the block (`header` with `body`, whose chunks
    /// match its body hash; its parent is stored), choosing the main chain.
    fn accept(&self, header: &Header, body: &[u8]) -> Result<Added, Rejected>;
    /// The fast-sync base at main-chain height `height`, if this node can
    /// serve the state there.
    fn snapshot(&self, height: u64) -> Option<Snapshot>;
    /// A reply for node (`level`, `index`) of the state tree after
    /// main-chain block `base`, if this node can serve it.
    fn state_nodes(&self, base: &Hash, level: u8, index: u64) -> Option<Vec<u8>>;
    /// Checks a fast-sync base: `base` (whose body is `body`) against
    /// `parent`, and its chain proof given the parent's total work.
    fn check_base(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected>;
    /// Starts (or resumes) downloading the state at a checked base.
    fn begin_state(&self, base: &Header) -> Result<(), String>;
    /// The base (height and hash) of an unfinished state download.
    fn state_progress(&self) -> Option<(u64, Hash)>;
    /// Up to `n` state-tree nodes still to fetch, leaving out `skip`.
    fn state_wanted(&self, n: usize, skip: &HashSet<(u8, u64)>) -> Vec<(u8, u64)>;
    /// Takes replies (level, index, data), each checked against its node's
    /// hash: what became of each.
    fn put_state(&self, replies: &[StateReply]) -> Result<Vec<SyncPut>, String>;
    /// Whether the state download has nothing left to fetch.
    fn state_done(&self) -> bool;
    /// Starts a new chain at a checked base whose state is downloaded,
    /// checking the state against the base's header.
    fn install(
        &self,
        parent: &Header,
        parent_work: u128,
        base: &Header,
        body: &[u8],
    ) -> Result<(), Rejected>;
    /// The encoding of mempool transaction `id`.
    fn tx(&self, id: &Hash) -> Option<Vec<u8>>;
    /// Whether transaction `id` needn't be fetched: it's in the mempool, or
    /// was turned away lately.
    fn knows_tx(&self, id: &Hash) -> bool;
    /// Checks a peer's transaction and adds it to the mempool: whether it's
    /// new. `Invalid` if it's bad in itself; `Failed` if it just doesn't
    /// fit (our state may differ from the peer's).
    fn add_tx(&self, bytes: &[u8]) -> Result<bool, Rejected>;
    /// The mempool's transaction ids.
    fn tx_ids(&self) -> Vec<Hash>;
}

/// A state-tree node's level and index, and a peer's reply for it.
pub type StateReply = (u8, u64, Vec<u8>);

/// Why a block wasn't accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejected {
    /// The block is invalid: its sender misbehaved.
    Invalid(String),
    /// Something else went wrong (say, a database error).
    Failed(String),
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) | Self::Failed(e) => f.write_str(e),
        }
    }
}

/// Events from the rest of the program.
#[derive(Clone, Debug)]
pub enum Event {
    /// The local chain has a new tip (a block mined here): announce it.
    NewTip,
    /// A transaction was added to the mempool here: announce it.
    NewTx(Hash),
    /// Send a [`PeerReport`] to the given channel.
    Peers(std::sync::mpsc::Sender<PeerReport>),
}

/// The node's peers as of a moment: what the `peers` command shows.
#[derive(Clone, Debug, Default)]
pub struct PeerReport {
    pub connected: Vec<PeerInfo>,
    /// Addresses being dialed, not answering yet.
    pub dialing: Vec<SocketAddr>,
    /// The address book.
    pub book: Vec<BookEntry>,
}

#[derive(Clone, Debug)]
pub struct PeerInfo {
    pub addr: SocketAddr,
    /// Whether we dialed it.
    pub outbound: bool,
    /// Where it accepts connections, if it does.
    pub listen: Option<SocketAddr>,
    pub tip: Tip,
    /// Seconds since we last heard from it.
    pub idle: u64,
}

#[derive(Clone, Debug)]
pub struct BookEntry {
    pub addr: SocketAddr,
    /// We've had a session with it.
    pub verified: bool,
    /// Dials that went unanswered.
    pub failures: u32,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// The network: the first bytes of its genesis hash ([`magic`]).
    pub magic: [u8; 4],
    /// [`cap`] bits: [`cap::LISTENING`] if this node's socket is bound to
    /// an address others can reach.
    pub capabilities: u16,
    /// Whether to download and accept blocks from peers.
    pub follow: bool,
    /// Peers to dial, and redial whenever they're lost.
    pub peers: Vec<SocketAddr>,
    /// Seed nodes (`host:port` names, looked up when dialed), for when
    /// there's no one else to dial.
    pub seeds: Vec<String>,
    /// Outbound connections to keep, configured peers included.
    pub outbound: usize,
    /// Where to keep the address book, if anywhere.
    pub peer_file: Option<PathBuf>,
    /// Most open sessions, inbound and outbound.
    pub max_peers: usize,
    /// Fast sync: a new node starts this many blocks below the best peer's
    /// tip; `None` downloads every block.
    pub sync_depth: Option<u64>,
    /// Drop this many of every 1000 packets received (for tests).
    pub loss_per_mille: u32,
}

impl Config {
    /// A following node on network `magic` with no peers yet.
    pub fn new(magic: [u8; 4]) -> Self {
        Self {
            magic,
            capabilities: 0,
            follow: true,
            peers: vec![],
            seeds: vec![],
            outbound: 8,
            max_peers: 64,
            peer_file: None,
            sync_depth: None,
            loss_per_mille: 0,
        }
    }
}

/// The magic of the network whose genesis header hashes to `genesis`.
pub fn magic(genesis: &Hash) -> [u8; 4] {
    hash::to_bytes(genesis)[..4].try_into().expect("4 bytes")
}

/// Resend a `HELLO` this often until a dial completes.
const DIAL_RETRY: Duration = Duration::from_millis(1000);
/// `HELLO`s sent to a learned address before giving up on it this time.
const DIAL_TRIES: u32 = 5;
/// Failed dials after which a learned address is forgotten.
const MAX_FAILURES: u32 = 3;
/// Wait at least this long between starting dials to learned addresses.
const BOOK_DIAL_EVERY: Duration = Duration::from_millis(500);
/// The least time between dials of the seed nodes.
const SEED_EVERY: Duration = Duration::from_secs(60);
/// Most addresses kept in the book.
const MAX_BOOK: usize = 1000;
/// Most untried addresses one peer (by IP) can have in the book.
const MAX_FROM_ONE: usize = 64;
/// Ask a random peer for addresses this often.
const PEERS_EVERY: Duration = Duration::from_secs(300);
/// Save the book (when it changed) at most this often.
const SAVE_EVERY: Duration = Duration::from_secs(10);
const PING_EVERY: Duration = Duration::from_secs(5);
/// A peer silent this long is dropped.
const PEER_TIMEOUT: Duration = Duration::from_secs(30);
/// Re-request a chunk not received within this many round trips (bounded
/// below and above).
const RTO_RTTS: u32 = 3;
const MIN_RTO: Duration = Duration::from_millis(30);
const MAX_RTO: Duration = Duration::from_secs(1);
/// The timeout before any round trip is measured.
const INITIAL_RTO: Duration = Duration::from_millis(400);
/// Re-ask for headers, a block's info or a snapshot after this.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
/// Times to ask a peer for a snapshot before trying another.
const SNAPSHOT_TRIES: u32 = 3;
/// State-tree requests outstanding to one peer, at most.
const STATE_WINDOW: usize = 64;
/// Ask again (any peer) for a state-tree node not received within this.
const STATE_TIMEOUT: Duration = Duration::from_secs(1);
/// State-tree replies stored in one transaction, at most.
const STATE_BATCH: usize = 256;
/// Log a state download's progress this often.
const REPORT_EVERY: Duration = Duration::from_secs(10);
/// Give up a download that's made no progress for this long.
const STALL_TIMEOUT: Duration = Duration::from_secs(15);
/// With no peers this long after starting, a node counts as caught up
/// (the first node of a network has no one to sync from).
const ALONE_AFTER: Duration = Duration::from_secs(10);
const INITIAL_WINDOW: usize = 16;
const MAX_WINDOW: usize = 256;
/// Chunks served for one request, at most.
const MAX_SERVE: usize = 256;
/// Headers queued ahead of their bodies, at most.
const MAX_PENDING: usize = 4096;
/// Bodies kept ready to serve.
const SERVED_CACHE: usize = 4;
/// Transactions kept ready to serve (a peer may fetch `MAX_TX_FETCHES` at
/// once).
const SERVED_TX_CACHE: usize = 512;
/// Transactions fetched at once, at most.
const MAX_TX_FETCHES: usize = 256;
/// Announced transactions waiting for a fetch slot, at most (each costs an
/// id and its announcers).
const MAX_TX_QUEUE: usize = 100_000;
/// Bytes of transactions being fetched at once, at most (well under the
/// socket buffer, so replies aren't lost in bursts); a fetch whose size
/// isn't known yet counts as `TX_SIZE_GUESS`.
const TX_BYTES_IN_FLIGHT: usize = 1 << 20;
const TX_SIZE_GUESS: usize = 8 << 10;
/// Times to ask for a transaction before trying another peer that has it.
const TX_TRIES: u32 = 3;
/// Packets handled between timer runs, at most, so a flood can't starve
/// the timers.
const PACKETS_PER_TICK: usize = 1024;

struct Peer {
    addr: SocketAddr,
    capabilities: u16,
    tip: Tip,
    /// Whether we dialed it.
    outbound: bool,
    /// Where it accepts connections, if it does.
    listen: Option<SocketAddr>,
    /// We asked it for addresses and haven't had its answer.
    asked_peers: bool,
    last_seen: Instant,
    last_ping: Instant,
}

struct Dial {
    last_sent: Option<Instant>,
    cookie: Option<[u8; COOKIE_LEN]>,
    tries: u32,
    /// A configured peer: dialed until it answers.
    persistent: bool,
}

/// An address in the book.
struct Known {
    /// We've had a session with it.
    verified: bool,
    /// The peer that told us of it, while it's untried.
    source: Option<IpAddr>,
    failures: u32,
    next_try: Instant,
}

/// What a download is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A block to accept.
    Block,
    /// A fast-sync base block to check.
    Base,
}

struct Download {
    kind: Kind,
    /// The block's header hash.
    id: Hash,
    header: Header,
    session: u32,
    asm: Option<Assembly>,
    /// Requested chunks: when, and in which request (the sender answers
    /// requests in order, so a chunk of a later request arriving first
    /// means an earlier one was lost).
    inflight: HashMap<usize, (Instant, u64)>,
    requests: u64,
    /// Smoothed round-trip time of chunk requests.
    srtt: Option<Duration>,
    window: usize,
    asked: Instant,
    started: Instant,
    last_progress: Instant,
}

/// A transaction being fetched.
struct TxFetch {
    session: u32,
    /// Other peers that announced it, to try next.
    others: Vec<u32>,
    asm: Option<Assembly>,
    asked: Instant,
    tries: u32,
}

/// A fast-sync base, from `SNAPSHOT_INFO`.
#[derive(Clone, Copy, Debug)]
struct Info {
    parent: Header,
    base: Header,
    parent_work: u128,
}

/// Fast sync in progress.
enum FastSync {
    /// Asked `session` for the base at `height` (to resume a download if
    /// `resume`).
    Asking {
        session: u32,
        height: u64,
        asked: Instant,
        tries: u32,
        resume: bool,
    },
    /// Downloading the base block from `session`.
    Base { session: u32, info: Box<Info> },
    /// Downloading the state, the base block checked.
    State(Box<StateSync>),
}

impl FastSync {
    /// The one peer this step depends on, if any.
    fn session(&self) -> Option<u32> {
        match self {
            Self::Asking { session, .. } | Self::Base { session, .. } => Some(*session),
            Self::State(_) => None,
        }
    }
}

/// A state download: requests spread over every peer that may serve it.
struct StateSync {
    info: Info,
    /// The base's hash.
    base: Hash,
    body: Vec<u8>,
    /// Requested nodes: from whom, and when.
    inflight: HashMap<(u8, u64), (u32, Instant)>,
    /// Sessions that can't serve this state.
    refused: HashSet<u32>,
    /// Replies to store (in batches): session, level, index, data.
    replies: Vec<(u32, u8, u64, Vec<u8>)>,
    nodes: u64,
    bytes: u64,
    /// Sessions that sent nodes we stored.
    from: HashSet<u32>,
    started: Instant,
    last_progress: Instant,
    last_report: Instant,
}

pub struct Node<B: Backend> {
    socket: UdpSocket,
    cfg: Config,
    backend: B,
    log: Box<dyn Fn(String) + Send>,
    rng: Rng,
    secret: Hash,
    node_id: u64,
    seq: u32,
    peers: HashMap<u32, Peer>,
    dials: HashMap<SocketAddr, Dial>,
    book: HashMap<SocketAddr, Known>,
    /// Our own addresses (a dial that reached ourselves).
    ours: HashSet<SocketAddr>,
    book_dirty: bool,
    last_save: Instant,
    last_book_dial: Instant,
    /// When the seed nodes were last dialed.
    last_seed: Option<Instant>,
    last_peers_ask: Instant,
    served: VecDeque<(Hash, Served)>,
    served_txs: VecDeque<(Hash, Served)>,
    tx_fetches: HashMap<Hash, TxFetch>,
    /// Announced transactions waiting for a fetch slot, oldest first, and
    /// the peers that announced each.
    tx_queue: VecDeque<Hash>,
    tx_queued: HashMap<Hash, Vec<u32>>,
    /// Fetched transactions waiting to be checked, with the peer each came
    /// from: checked together, in parallel, at the tick.
    tx_done: Vec<(Hash, u32, Vec<u8>)>,
    /// Transactions added here, to announce at the next tick (together),
    /// with the peer each came from (not told).
    tx_to_announce: Vec<(Hash, Option<u32>)>,
    pending: VecDeque<Header>,
    download: Option<Download>,
    header_request: Option<(u32, Instant)>,
    fast: Option<FastSync>,
    /// Sessions that couldn't give us a snapshot.
    no_snapshot: HashSet<u32>,
    /// A peer couldn't serve the base of the unfinished state download:
    /// start a new one instead.
    resume_failed: bool,
    /// Per session, the highest block it said it doesn't have (a peer that
    /// fast-synced has none below its base).
    lacks: HashMap<u32, u64>,
    synced: Arc<AtomicBool>,
    started: Instant,
    drop_counter: u32,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn short(h: &Hash) -> String {
    hash::to_bytes(h)[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whether `addr` can be dialed at all.
/// Asks for 4 MiB socket buffers (the kernel caps them at its limit, such
/// as Linux's `net.core.rmem_max`): a burst of replies (a block's chunks,
/// or many transactions') mustn't overflow the default (~200 KB on Linux)
/// between two reads, or it's lost.
fn widen_buffers(socket: &UdpSocket) {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        unsafe extern "C" {
            fn setsockopt(
                fd: i32,
                level: i32,
                name: i32,
                value: *const std::ffi::c_void,
                len: u32,
            ) -> i32;
        }
        #[cfg(target_os = "linux")]
        const OPTS: (i32, [i32; 2]) = (1, [7, 8]);
        #[cfg(not(target_os = "linux"))]
        const OPTS: (i32, [i32; 2]) = (0xffff, [0x1001, 0x1002]);
        let size: i32 = 4 << 20;
        for name in OPTS.1 {
            // SAFETY: an open socket, and a pointer to an i32 of the length
            // given. A failure only leaves the default size.
            unsafe {
                setsockopt(
                    socket.as_raw_fd(),
                    OPTS.0,
                    name,
                    (&size as *const i32).cast(),
                    4,
                );
            }
        }
    }
}

fn dialable(addr: &SocketAddr) -> bool {
    addr.port() != 0 && !addr.ip().is_unspecified() && !addr.ip().is_multicast()
}

impl<B: Backend> Node<B> {
    /// A node on `socket` (already bound) serving `backend`, logging with
    /// `log`.
    pub fn new(
        socket: UdpSocket,
        cfg: Config,
        backend: B,
        log: Box<dyn Fn(String) + Send>,
    ) -> Self {
        let mut rng = Rng::from_os().expect("OS randomness");
        let mut secret = [BabyBear::ZERO; HASH_LEN];
        rng.fill(&mut secret);
        let node_id =
            (rng.next_element().as_u32() as u64) << 32 | rng.next_element().as_u32() as u64;
        socket
            .set_read_timeout(Some(Duration::from_millis(10)))
            .expect("socket timeout");
        widen_buffers(&socket);
        let now = Instant::now();
        let mut book = HashMap::new();
        if let Some(path) = &cfg.peer_file
            && let Ok(text) = std::fs::read_to_string(path)
        {
            for addr in text
                .lines()
                .filter_map(|l| l.trim().parse::<SocketAddr>().ok())
            {
                if dialable(&addr) && book.len() < MAX_BOOK {
                    book.insert(
                        addr,
                        Known {
                            verified: true,
                            source: None,
                            failures: 0,
                            next_try: now,
                        },
                    );
                }
            }
        }
        let long_ago = now.checked_sub(PEERS_EVERY).unwrap_or(now);
        Self {
            socket,
            cfg,
            backend,
            log,
            rng,
            secret,
            node_id,
            seq: 0,
            peers: HashMap::new(),
            dials: HashMap::new(),
            book,
            ours: HashSet::new(),
            book_dirty: false,
            last_save: now,
            last_book_dial: long_ago,
            last_seed: None,
            last_peers_ask: now,
            served: VecDeque::new(),
            served_txs: VecDeque::new(),
            tx_fetches: HashMap::new(),
            tx_queue: VecDeque::new(),
            tx_queued: HashMap::new(),
            tx_to_announce: Vec::new(),
            tx_done: Vec::new(),
            pending: VecDeque::new(),
            download: None,
            header_request: None,
            fast: None,
            no_snapshot: HashSet::new(),
            resume_failed: false,
            lacks: HashMap::new(),
            synced: Arc::new(AtomicBool::new(false)),
            started: now,
            drop_counter: 0,
        }
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.socket.local_addr().expect("bound socket")
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Set once the node has caught up with its peers (or found none for a
    /// while): the time to start mining.
    pub fn synced(&self) -> Arc<AtomicBool> {
        self.synced.clone()
    }

    /// Peers with an open session: address, capabilities and tip.
    pub fn peers(&self) -> Vec<(SocketAddr, u16, Tip)> {
        self.peers
            .values()
            .map(|p| (p.addr, p.capabilities, p.tip))
            .collect()
    }

    /// Runs until `stop` is set, taking events from `events`.
    pub fn run(&mut self, events: &Receiver<Event>, stop: &AtomicBool) {
        let mut buf = [0u8; MAX_PACKET + 1];
        while !stop.load(Ordering::Relaxed) {
            // Take what has arrived, then do timed work.
            for _ in 0..PACKETS_PER_TICK {
                let Ok((n, addr)) = self.socket.recv_from(&mut buf) else {
                    break;
                };
                self.receive(addr, &buf[..n]);
            }
            while let Ok(event) = events.try_recv() {
                match event {
                    Event::NewTip => self.announce_tip(None),
                    // Announced together at the tick.
                    Event::NewTx(id) => self.tx_to_announce.push((id, None)),
                    Event::Peers(reply) => {
                        let _ = reply.send(self.peer_report());
                    }
                }
            }
            self.tick();
        }
        self.save_book();
    }

    fn send(&mut self, addr: SocketAddr, session: u32, msg: &Message) {
        self.seq = self.seq.wrapping_add(1);
        let bytes = wire::encode(
            self.cfg.magic,
            Packet {
                session,
                seq: self.seq,
            },
            msg,
        );
        let _ = self.socket.send_to(&bytes, addr);
    }

    fn send_peer(&mut self, session: u32, msg: &Message) {
        if let Some(addr) = self.peers.get(&session).map(|p| p.addr) {
            self.send(addr, session, msg);
        }
    }

    fn cookie(&self, addr: &SocketAddr, minute: u64) -> [u8; COOKIE_LEN] {
        let ip: Vec<u8> = match addr.ip() {
            IpAddr::V4(a) => a.to_ipv6_mapped().octets().to_vec(),
            IpAddr::V6(a) => a.octets().to_vec(),
        };
        let ip: Vec<BabyBear> = ip
            .chunks(3)
            .map(|b| {
                let mut w = [0u8; 4];
                w[..b.len()].copy_from_slice(b);
                BabyBear::new(u32::from_le_bytes(w))
            })
            .collect();
        let extra = [
            BabyBear::new(addr.port() as u32),
            BabyBear::new((minute & 0xff_ffff) as u32),
        ];
        let h = hash::sponge(domain::NET_COOKIE, &[], &[&self.secret, &ip, &extra]);
        hash::to_bytes(&h)[..COOKIE_LEN]
            .try_into()
            .expect("cookie bytes")
    }

    fn cookie_ok(&self, addr: &SocketAddr, cookie: &[u8; COOKIE_LEN]) -> bool {
        let minute = now_secs() / 60;
        [minute, minute.wrapping_sub(1)]
            .iter()
            .any(|&m| self.cookie(addr, m) == *cookie)
    }

    fn hello(&self, cookie: Option<[u8; COOKIE_LEN]>) -> Message {
        Message::Hello {
            capabilities: self.cfg.capabilities,
            node_id: self.node_id,
            listen_port: self.local_addr().port(),
            tip: self.backend.tip(),
            cookie,
        }
    }

    fn receive(&mut self, addr: SocketAddr, bytes: &[u8]) {
        if let Some(k) = 1000u32.checked_div(self.cfg.loss_per_mille) {
            // Spread the drops: every k-th packet.
            self.drop_counter = (self.drop_counter + 1) % 1000;
            if self.drop_counter.is_multiple_of(k.max(1)) {
                return;
            }
        }
        let Ok((packet, msg)) = wire::decode(self.cfg.magic, bytes) else {
            return;
        };
        match msg {
            Message::Hello {
                capabilities,
                node_id,
                listen_port,
                tip,
                cookie,
            } if packet.session == 0 => {
                self.on_hello(addr, capabilities, node_id, listen_port, tip, cookie)
            }
            Message::Cookie { cookie } if packet.session == 0 => {
                if let Some(d) = self.dials.get_mut(&addr) {
                    d.cookie = Some(cookie);
                    d.last_sent = Some(Instant::now());
                    let hello = self.hello(Some(cookie));
                    self.send(addr, 0, &hello);
                }
            }
            Message::Welcome {
                capabilities, tip, ..
            } if packet.session != 0 => {
                if self.dials.remove(&addr).is_some() && !self.peers.contains_key(&packet.session) {
                    self.add_peer(packet.session, addr, capabilities, tip, true, Some(addr));
                    self.ask_peers(packet.session);
                }
            }
            msg => {
                let Some(peer) = self.peers.get_mut(&packet.session) else {
                    return;
                };
                if peer.addr != addr {
                    return;
                }
                peer.last_seen = Instant::now();
                self.on_message(packet.session, msg);
            }
        }
    }

    fn add_peer(
        &mut self,
        session: u32,
        addr: SocketAddr,
        capabilities: u16,
        tip: Tip,
        outbound: bool,
        listen: Option<SocketAddr>,
    ) {
        let now = Instant::now();
        (self.log)(format!(
            "peer {addr} connected ({}, session {session:08x}, capabilities {capabilities:04x}): tip {} at height {}",
            if outbound { "outbound" } else { "inbound" },
            short(&tip.hash),
            tip.height
        ));
        if let Some(l) = listen
            && dialable(&l)
            && (self.book.contains_key(&l) || self.make_room(None))
        {
            self.book.insert(
                l,
                Known {
                    verified: true,
                    source: None,
                    failures: 0,
                    next_try: now,
                },
            );
            self.book_dirty = true;
        }
        self.peers.insert(
            session,
            Peer {
                addr,
                capabilities,
                tip,
                outbound,
                listen,
                asked_peers: false,
                last_seen: now,
                last_ping: now,
            },
        );
        let ids = self.backend.tx_ids();
        for batch in ids.chunks(MAX_TX_IDS) {
            self.send_peer(
                session,
                &Message::TxInv {
                    ids: batch.to_vec(),
                },
            );
        }
        self.maybe_sync();
    }

    fn on_hello(
        &mut self,
        addr: SocketAddr,
        capabilities: u16,
        node_id: u64,
        listen_port: u16,
        tip: Tip,
        cookie: Option<[u8; COOKIE_LEN]>,
    ) {
        if node_id == self.node_id {
            // We dialed ourselves: never again.
            if self.ours.insert(addr) {
                (self.log)(format!("{addr} is this node; not dialing it"));
            }
            self.dials.remove(&addr);
            if self.book.remove(&addr).is_some() {
                self.book_dirty = true;
            }
            return;
        }
        match cookie {
            Some(c) if self.cookie_ok(&addr, &c) => {
                let reopening = self.peers.values().any(|p| p.addr == addr);
                if !reopening && self.peers.len() >= self.cfg.max_peers {
                    return;
                }
                // A validated address: (re)open a session.
                self.peers.retain(|_, p| p.addr != addr);
                self.dials.remove(&addr);
                let session = loop {
                    let s = self.rng.next_element().as_u32();
                    if s != 0 && !self.peers.contains_key(&s) {
                        break s;
                    }
                };
                let welcome = Message::Welcome {
                    capabilities: self.cfg.capabilities,
                    node_id: self.node_id,
                    tip: self.backend.tip(),
                };
                self.send(addr, session, &welcome);
                let listen = (capabilities & cap::LISTENING != 0)
                    .then(|| SocketAddr::new(addr.ip(), listen_port));
                self.add_peer(session, addr, capabilities, tip, false, listen);
            }
            _ => {
                let minute = now_secs() / 60;
                let cookie = self.cookie(&addr, minute);
                self.send(addr, 0, &Message::Cookie { cookie });
            }
        }
    }

    fn drop_peer(&mut self, session: u32, why: &str) {
        if let Some(p) = self.peers.remove(&session) {
            (self.log)(format!("dropping peer {}: {why}", p.addr));
        }
        if self.download.as_ref().is_some_and(|d| d.session == session) {
            self.download = None;
        }
        if self.header_request.is_some_and(|(s, _)| s == session) {
            self.header_request = None;
        }
        let lost = match &mut self.fast {
            Some(FastSync::State(s)) => {
                s.inflight.retain(|_, (sid, _)| *sid != session);
                s.refused.remove(&session);
                false
            }
            Some(f) => f.session() == Some(session),
            None => false,
        };
        if lost {
            self.fast = None;
        }
        self.lacks.remove(&session);
        self.no_snapshot.remove(&session);
        let fetches: Vec<Hash> = self
            .tx_fetches
            .iter()
            .filter(|(_, f)| f.session == session)
            .map(|(id, _)| *id)
            .collect();
        for id in fetches {
            self.next_tx_source(id);
        }
        for f in self.tx_fetches.values_mut() {
            f.others.retain(|&s| s != session);
        }
    }

    fn on_message(&mut self, session: u32, msg: Message) {
        match msg {
            Message::Ping { nonce } => self.send_peer(session, &Message::Pong { nonce }),
            Message::Pong { .. } => {}
            Message::Status { tip } => {
                if let Some(p) = self.peers.get_mut(&session) {
                    p.tip = tip;
                }
                self.maybe_sync();
            }
            Message::GetHeaders { locator, count } => {
                // Headers after the newest locator entry on our main chain
                // (genesis if none).
                let from = locator
                    .iter()
                    .find_map(|h| self.backend.height_of(h))
                    .map_or(1, |h| h + 1);
                let headers: Vec<Header> = (from..from
                    .saturating_add(count.min(MAX_HEADERS as u16) as u64))
                    .map_while(|h| self.backend.header_at(h))
                    .collect();
                self.send_peer(session, &Message::Headers { headers });
            }
            Message::Headers { headers } => self.on_headers(session, headers),
            Message::NewHeader { header } => self.on_new_header(session, header),
            Message::GetBlock { hash } => match self.serve(&hash) {
                Some((size, root)) => {
                    self.send_peer(session, &Message::BlockInfo { hash, size, root })
                }
                None => self.send_peer(session, &Message::NotFound { hash }),
            },
            Message::GetBlockChunks { hash, ranges } => self.serve_chunks(session, hash, &ranges),
            Message::BlockInfo { hash, size, root } => {
                self.on_block_info(session, hash, size, root)
            }
            Message::BlockChunk { hash, index, data } => self.on_chunk(session, hash, index, &data),
            Message::NotFound { hash } => {
                if self
                    .tx_fetches
                    .get(&hash)
                    .is_some_and(|f| f.session == session)
                {
                    self.next_tx_source(hash);
                }
                if self
                    .download
                    .as_ref()
                    .is_some_and(|d| d.session == session && d.id == hash)
                {
                    let height = self.download.take().map_or(0, |d| d.header.height);
                    if self
                        .fast
                        .as_ref()
                        .is_some_and(|f| f.session() == Some(session))
                    {
                        self.no_snapshot.insert(session);
                        self.fast = None;
                    } else {
                        let l = self.lacks.entry(session).or_insert(0);
                        *l = (*l).max(height);
                    }
                    self.maybe_download();
                    self.maybe_sync();
                }
            }
            Message::GetPeers => self.serve_peers(session),
            Message::Peers { addrs } => self.on_peers(session, &addrs),
            Message::GetSnapshot { height } => {
                let msg = match self.backend.snapshot(height) {
                    Some(s) => Message::SnapshotInfo {
                        parent: s.parent,
                        base: s.base,
                        parent_work: s.parent_work,
                    },
                    None => Message::NoSnapshot { height },
                };
                self.send_peer(session, &msg);
            }
            Message::SnapshotInfo {
                parent,
                base,
                parent_work,
            } => self.on_snapshot_info(
                session,
                Info {
                    parent,
                    base,
                    parent_work,
                },
            ),
            Message::GetState { base, level, index } => {
                let msg = match self.backend.state_nodes(&base, level, index) {
                    Some(data) if data.len() <= MAX_STATE_DATA => Message::State {
                        base,
                        level,
                        index,
                        data,
                    },
                    _ => Message::NoState { base },
                };
                self.send_peer(session, &msg);
            }
            Message::State {
                base,
                level,
                index,
                data,
            } => self.on_state(session, base, level, index, data),
            Message::NoState { base } => {
                if let Some(FastSync::State(s)) = &mut self.fast
                    && s.base == base
                {
                    if s.refused.insert(session) {
                        s.inflight.retain(|_, (sid, _)| *sid != session);
                        let addr = self.peers.get(&session).map(|p| p.addr);
                        (self.log)(format!(
                            "fast sync: {} can't serve the state at height {}",
                            addr.map_or("?".into(), |a| a.to_string()),
                            s.info.base.height
                        ));
                    }
                    self.fill_state();
                }
            }
            Message::NoSnapshot { height } => {
                if let Some(FastSync::Asking {
                    session: s,
                    height: h,
                    resume,
                    ..
                }) = &self.fast
                    && (*s, *h) == (session, height)
                {
                    let addr = self.peers.get(&session).map(|p| p.addr);
                    (self.log)(format!(
                        "fast sync: {} has no state at height {height}{}",
                        addr.map_or("?".into(), |a| a.to_string()),
                        if *resume {
                            "; starting a new download"
                        } else {
                            ""
                        }
                    ));
                    if *resume {
                        self.resume_failed = true;
                    } else {
                        self.no_snapshot.insert(session);
                    }
                    self.fast = None;
                    self.maybe_sync();
                }
            }
            Message::TxInv { ids } => self.on_tx_inv(session, &ids),
            Message::GetTx { id } => match self.serve(&id) {
                Some((size, root)) => self.send_peer(session, &Message::TxInfo { id, size, root }),
                None => self.send_peer(session, &Message::NotFound { hash: id }),
            },
            Message::TxInfo { id, size, root } => self.on_tx_info(session, id, size, root),
            Message::Hello { .. } | Message::Cookie { .. } | Message::Welcome { .. } => {}
        }
    }

    // Peers.

    /// Addresses to pass on: nodes that accept connections, ones we've had
    /// sessions with first, in random order.
    fn serve_peers(&mut self, session: u32) {
        let asker = self.peers.get(&session).map(|p| (p.addr, p.listen));
        let mut verified: Vec<SocketAddr> = self
            .peers
            .values()
            .filter_map(|p| p.listen)
            .chain(
                self.book
                    .iter()
                    .filter(|(_, k)| k.verified)
                    .map(|(a, _)| *a),
            )
            .filter(|a| asker.is_none_or(|(addr, listen)| *a != addr && Some(*a) != listen))
            .collect();
        verified.sort();
        verified.dedup();
        // Fisher-Yates.
        for i in (1..verified.len()).rev() {
            let j = self.rng.next_element().as_u32() as usize % (i + 1);
            verified.swap(i, j);
        }
        verified.truncate(MAX_ADDRS);
        self.send_peer(session, &Message::Peers { addrs: verified });
    }

    fn ask_peers(&mut self, session: u32) {
        if let Some(p) = self.peers.get_mut(&session) {
            p.asked_peers = true;
            self.send_peer(session, &Message::GetPeers);
        }
    }

    /// Whether the book has room for one more address, pushing out a
    /// random untried one (not one `adding` told us of) if it's full.
    fn make_room(&mut self, adding: Option<IpAddr>) -> bool {
        if self.book.len() < MAX_BOOK {
            return true;
        }
        let untried: Vec<SocketAddr> = self
            .book
            .iter()
            .filter(|(_, k)| !k.verified && (adding.is_none() || k.source != adding))
            .map(|(a, _)| *a)
            .collect();
        if untried.is_empty() {
            return false;
        }
        let i = self.rng.next_element().as_u32() as usize % untried.len();
        self.book.remove(&untried[i]);
        true
    }

    fn on_peers(&mut self, session: u32, addrs: &[SocketAddr]) {
        let Some(peer) = self.peers.get_mut(&session) else {
            return;
        };
        if !std::mem::take(&mut peer.asked_peers) {
            // Unasked: ignored.
            return;
        }
        let source = peer.addr.ip();
        let now = Instant::now();
        let mut from_it = self
            .book
            .values()
            .filter(|k| k.source == Some(source))
            .count();
        let mut learned = 0;
        for a in addrs {
            if from_it >= MAX_FROM_ONE {
                break;
            }
            if dialable(a)
                && !self.ours.contains(a)
                && !self.book.contains_key(a)
                && self.make_room(Some(source))
            {
                self.book.insert(
                    *a,
                    Known {
                        verified: false,
                        source: Some(source),
                        failures: 0,
                        next_try: now,
                    },
                );
                learned += 1;
                from_it += 1;
            }
        }
        if learned > 0 {
            let from = self.peers.get(&session).map(|p| p.addr);
            (self.log)(format!(
                "learned {learned} peer address{} from {}",
                if learned == 1 { "" } else { "es" },
                from.map_or("?".into(), |a| a.to_string())
            ));
        }
    }

    /// Writes the book's verified addresses to the peer file.
    fn save_book(&mut self) {
        let Some(path) = &self.cfg.peer_file else {
            return;
        };
        let mut addrs: Vec<String> = self
            .book
            .iter()
            .filter(|(_, k)| k.verified)
            .map(|(a, _)| a.to_string())
            .collect();
        addrs.sort();
        let mut text = addrs.join("\n");
        text.push('\n');
        if let Err(e) = std::fs::write(path, text) {
            (self.log)(format!(
                "couldn't save the peer list to {}: {e}",
                path.display()
            ));
        }
        self.book_dirty = false;
        self.last_save = Instant::now();
    }

    fn peer_report(&self) -> PeerReport {
        let now = Instant::now();
        let mut connected: Vec<PeerInfo> = self
            .peers
            .values()
            .map(|p| PeerInfo {
                addr: p.addr,
                outbound: p.outbound,
                listen: p.listen,
                tip: p.tip,
                idle: (now - p.last_seen).as_secs(),
            })
            .collect();
        connected.sort_by_key(|p| p.addr);
        let mut dialing: Vec<SocketAddr> = self.dials.keys().copied().collect();
        dialing.sort();
        let mut book: Vec<BookEntry> = self
            .book
            .iter()
            .map(|(a, k)| BookEntry {
                addr: *a,
                verified: k.verified,
                failures: k.failures,
            })
            .collect();
        book.sort_by_key(|b| (!b.verified, b.addr));
        PeerReport {
            connected,
            dialing,
            book,
        }
    }

    /// Looks up the seed nodes' addresses and dials them (once each, not
    /// redialing them when lost).
    fn dial_seeds(&mut self) {
        let mut dialed = Vec::new();
        for name in self.cfg.seeds.clone() {
            let addrs = match std::net::ToSocketAddrs::to_socket_addrs(name.as_str()) {
                Ok(addrs) => addrs,
                Err(e) => {
                    (self.log)(format!("seed {name}: can't look it up: {e}"));
                    continue;
                }
            };
            for addr in addrs.filter(|a| dialable(a) && !self.ours.contains(a)) {
                self.dials.entry(addr).or_insert(Dial {
                    last_sent: None,
                    cookie: None,
                    tries: 0,
                    persistent: false,
                });
                dialed.push(format!("{name} ({addr})"));
            }
        }
        if !dialed.is_empty() {
            (self.log)(format!("no peers: dialing seeds {}", dialed.join(", ")));
        }
    }

    /// Dials configured peers we aren't connected to, and learned ones up
    /// to the outbound target; resends `HELLO`s; gives up on learned
    /// addresses that don't answer.
    fn dial(&mut self, now: Instant) {
        let connected: HashSet<SocketAddr> = self.peers.values().map(|p| p.addr).collect();
        self.dials.retain(|a, _| !connected.contains(a));
        for addr in self.cfg.peers.clone() {
            if !connected.contains(&addr) && !self.ours.contains(&addr) {
                self.dials.entry(addr).or_insert(Dial {
                    last_sent: None,
                    cookie: None,
                    tries: 0,
                    persistent: true,
                });
            }
        }
        let outbound = self.peers.values().filter(|p| p.outbound).count() + self.dials.len();
        if outbound < self.cfg.outbound
            && self.peers.len() < self.cfg.max_peers
            && now - self.last_book_dial >= BOOK_DIAL_EVERY
        {
            let candidates: Vec<SocketAddr> = self
                .book
                .iter()
                .filter(|(a, k)| {
                    k.next_try <= now
                        && !connected.contains(a)
                        && !self.dials.contains_key(a)
                        && !self.ours.contains(a)
                })
                .map(|(a, _)| *a)
                .collect();
            if !candidates.is_empty() {
                let addr = candidates[self.rng.next_element().as_u32() as usize % candidates.len()];
                self.last_book_dial = now;
                self.dials.insert(
                    addr,
                    Dial {
                        last_sent: None,
                        cookie: None,
                        tries: 0,
                        persistent: false,
                    },
                );
            }
        }
        // Cut off: ask the seeds.
        if self.peers.is_empty()
            && self.dials.is_empty()
            && !self.cfg.seeds.is_empty()
            && self.last_seed.is_none_or(|t| now - t >= SEED_EVERY)
        {
            self.last_seed = Some(now);
            self.dial_seeds();
        }

        let mut gave_up = Vec::new();
        let addrs: Vec<SocketAddr> = self.dials.keys().copied().collect();
        for addr in addrs {
            let d = self.dials.get_mut(&addr).expect("dial");
            if d.last_sent.is_some_and(|t| now - t < DIAL_RETRY) {
                continue;
            }
            if !d.persistent && d.tries >= DIAL_TRIES {
                gave_up.push(addr);
                continue;
            }
            d.last_sent = Some(now);
            d.tries += 1;
            // A stale cookie: start over now and then.
            if d.tries.is_multiple_of(4) {
                d.cookie = None;
            }
            let cookie = d.cookie;
            let hello = self.hello(cookie);
            self.send(addr, 0, &hello);
        }
        for addr in gave_up {
            self.dials.remove(&addr);
            if let Some(k) = self.book.get_mut(&addr) {
                k.failures += 1;
                k.next_try = now + Duration::from_secs(60 << k.failures.min(6));
                if k.failures >= MAX_FAILURES {
                    self.book.remove(&addr);
                }
                self.book_dirty = true;
            }
        }
    }

    // Serving.

    /// The body with header hash `hash`, ready to serve: its size and root.
    /// (Or the mempool transaction with id `hash`.)
    fn serve(&mut self, hash: &Hash) -> Option<(u32, Hash)> {
        let cached = |c: &VecDeque<(Hash, Served)>| c.iter().any(|(h, _)| h == hash);
        if !cached(&self.served) && !cached(&self.served_txs) {
            if let Some(body) = self.backend.body(hash) {
                if self.served.len() == SERVED_CACHE {
                    self.served.pop_front();
                }
                self.served.push_back((*hash, Served::new(body)));
            } else {
                let tx = self.backend.tx(hash)?;
                if self.served_txs.len() == SERVED_TX_CACHE {
                    self.served_txs.pop_front();
                }
                self.served_txs.push_back((*hash, Served::new(tx)));
            }
        }
        let (_, s) = self
            .served
            .iter()
            .chain(&self.served_txs)
            .find(|(h, _)| h == hash)?;
        Some((s.bytes.len() as u32, s.root))
    }

    fn serve_chunks(&mut self, session: u32, hash: Hash, ranges: &[(u32, u16)]) {
        if self.serve(&hash).is_none() {
            self.send_peer(session, &Message::NotFound { hash });
            return;
        }
        let indices: Vec<usize> = ranges
            .iter()
            .flat_map(|&(first, count)| first as usize..first as usize + count as usize)
            .take(MAX_SERVE)
            .collect();
        let chunks: Vec<(usize, Vec<u8>)> = {
            let s = &self
                .served
                .iter()
                .chain(&self.served_txs)
                .find(|(h, _)| *h == hash)
                .expect("cached")
                .1;
            indices
                .into_iter()
                .filter_map(|i| s.chunk(i).map(|c| (i, c)))
                .collect()
        };
        for (index, data) in chunks {
            self.send_peer(
                session,
                &Message::BlockChunk {
                    hash,
                    index: index as u32,
                    data,
                },
            );
        }
    }

    // Following.

    /// Hashes that say where our chain is: the newest queued header, then
    /// the main chain at its tip and 1, 2, 4, ... blocks back, and genesis.
    fn locator(&self) -> Vec<Hash> {
        let mut out: Vec<Hash> = self.pending.back().map(Header::hash).into_iter().collect();
        let tip = self.backend.tip().height;
        let mut back = 0u64;
        loop {
            let h = tip.saturating_sub(back);
            if let Some(header) = self.backend.header_at(h) {
                out.push(header.hash());
            }
            if h == 0 || out.len() == MAX_LOCATOR - 1 {
                break;
            }
            back = if back == 0 { 1 } else { back * 2 };
        }
        if tip > 0
            && let Some(g) = self.backend.header_at(0)
        {
            out.push(g.hash());
        }
        out.dedup();
        out
    }

    /// Asks `session` for the headers after our chain.
    fn request_headers(&mut self, session: u32) {
        self.header_request = Some((session, Instant::now()));
        let locator = self.locator();
        self.send_peer(
            session,
            &Message::GetHeaders {
                locator,
                count: MAX_HEADERS as u16,
            },
        );
    }

    /// Sessions that said they don't have the block after our tip.
    fn unhelpful(&self) -> HashSet<u32> {
        let next = self.backend.tip().height + 1;
        self.lacks
            .iter()
            .filter(|&(_, &l)| l >= next)
            .map(|(&s, _)| s)
            .collect()
    }

    /// The peer with the most work, more than ours, whose tip we don't
    /// have.
    fn best_peer(&self, exclude: &HashSet<u32>) -> Option<(u32, SocketAddr, u64)> {
        let ours = self.backend.tip().work;
        self.peers
            .iter()
            .filter(|(s, p)| {
                !exclude.contains(s)
                    && p.tip.work > ours
                    && self.backend.header_by_hash(&p.tip.hash).is_none()
            })
            .max_by_key(|(_, p)| p.tip.work)
            .map(|(&s, p)| (s, p.addr, p.tip.height))
    }

    fn maybe_sync(&mut self) {
        if !self.cfg.follow || self.fast.is_some() || self.pending.len() >= MAX_PENDING {
            return;
        }
        if self.maybe_fast_sync() {
            return;
        }
        // With headers queued, more are asked for as each full batch
        // arrives ([`Node::on_headers`]).
        if !self.pending.is_empty()
            || self
                .header_request
                .is_some_and(|(_, at)| at.elapsed() < REQUEST_TIMEOUT)
        {
            return;
        }
        match self.best_peer(&self.unhelpful()) {
            Some((session, addr, height)) => {
                if self.header_request.is_none() && self.pending.is_empty() {
                    (self.log)(format!(
                        "syncing from {addr} (its tip is at height {height})"
                    ));
                }
                self.request_headers(session);
            }
            None => self.header_request = None,
        }
    }

    /// Starts fast sync if this is a new node and a peer is far enough
    /// ahead. Returns whether it started.
    fn maybe_fast_sync(&mut self) -> bool {
        let Some(depth) = self.cfg.sync_depth else {
            return false;
        };
        if self.backend.tip().height != 0 || self.download.is_some() || !self.pending.is_empty() {
            return false;
        }
        let exclude = self.no_snapshot.clone();
        let Some((session, addr, tip)) = self.best_peer(&exclude) else {
            return false;
        };
        // An unfinished download (from before a restart) carries on if
        // peers can still serve its base.
        let resume = (!self.resume_failed)
            .then(|| self.backend.state_progress())
            .flatten()
            .map(|(h, _)| h)
            .filter(|&h| h <= tip);
        let height = match resume {
            Some(h) => h,
            None if tip <= depth => return false,
            None => tip - depth,
        };
        // Ask a peer whose advertised history covers it (the best one if
        // it does).
        let serves = |p: &Peer| {
            p.tip.state_from != 0 && p.tip.state_from <= height && height <= p.tip.height
        };
        let session = if self.peers.get(&session).is_some_and(serves) {
            session
        } else {
            match self
                .peers
                .iter()
                .find(|(s, p)| !exclude.contains(s) && serves(p))
            {
                Some((&s, _)) => s,
                None => {
                    (self.log)(format!(
                        "fast sync: {addr} has no state at height {height}{}",
                        if resume.is_some() {
                            "; starting a new download"
                        } else {
                            ""
                        }
                    ));
                    if resume.is_some() {
                        self.resume_failed = true;
                    } else {
                        self.no_snapshot.insert(session);
                    }
                    return false;
                }
            }
        };
        let asked = self.peers[&session].addr;
        (self.log)(match resume {
            Some(_) => format!(
                "fast sync: asking {asked} for the state at height {height}, to resume its download"
            ),
            None => format!(
                "fast sync: asking {asked} for the state at height {height} ({depth} below the \
                 best tip, at {tip})"
            ),
        });
        self.fast = Some(FastSync::Asking {
            session,
            height,
            asked: Instant::now(),
            tries: 1,
            resume: resume.is_some(),
        });
        self.send_peer(session, &Message::GetSnapshot { height });
        true
    }

    fn on_snapshot_info(&mut self, session: u32, info: Info) {
        let Some(FastSync::Asking {
            session: s, height, ..
        }) = &self.fast
        else {
            return;
        };
        if *s != session || info.base.height != *height {
            return;
        }
        let Info { parent, base, .. } = info;
        if parent.height == 0 || base.prev != parent.hash() || base.height != parent.height + 1 {
            self.drop_peer(session, "snapshot info doesn't link");
            return;
        }
        if let Err(e) = self.backend.check_header(&parent, &base) {
            self.drop_peer(session, &format!("bad snapshot base header: {e}"));
            return;
        }
        self.fast = Some(FastSync::Base {
            session,
            info: Box::new(info),
        });
        self.start_download(Kind::Base, base, session);
    }

    /// Starts downloading `header`'s block (a block to accept or a base)
    /// from `session`.
    fn start_download(&mut self, kind: Kind, header: Header, session: u32) {
        let now = Instant::now();
        let id = header.hash();
        self.download = Some(Download {
            kind,
            id,
            header,
            session,
            asm: None,
            inflight: HashMap::new(),
            requests: 0,
            srtt: None,
            window: INITIAL_WINDOW,
            asked: now,
            started: now,
            last_progress: now,
        });
        self.send_peer(session, &Message::GetBlock { hash: id });
    }

    /// Queues `header` for download if it extends the queue (or, with an
    /// empty queue, a stored block) and checks. `Err` if it doesn't check
    /// (the sender misbehaved); `Ok(false)` if it doesn't connect.
    fn queue(&mut self, header: Header) -> Result<bool, String> {
        let parent = match self.pending.back() {
            Some(b) if b.hash() == header.prev => *b,
            Some(_) => return Ok(false),
            None => match self.backend.header_by_hash(&header.prev) {
                Some(p) => p,
                None => return Ok(false),
            },
        };
        self.backend.check_header(&parent, &header)?;
        self.pending.push_back(header);
        Ok(true)
    }

    fn on_headers(&mut self, session: u32, headers: Vec<Header>) {
        if !self.cfg.follow || self.fast.is_some() {
            return;
        }
        if self.header_request.is_some_and(|(s, _)| s == session) {
            self.header_request = None;
        }
        let full = headers.len() == MAX_HEADERS;
        let ours = self.backend.tip();
        if headers.is_empty()
            && self
                .peers
                .get(&session)
                .is_some_and(|p| p.tip.work > ours.work)
        {
            // Ahead, but nothing after our chain: it fast-synced from a
            // block above ours.
            let l = self.lacks.entry(session).or_insert(0);
            *l = (*l).max(ours.height + 1);
        }
        for h in headers {
            let hash = h.hash();
            if self.backend.header_by_hash(&hash).is_some()
                || self.pending.iter().any(|p| p.hash() == hash)
            {
                continue;
            }
            match self.queue(h) {
                Ok(true) => {}
                // Doesn't connect to what we're fetching: finish that first.
                Ok(false) => break,
                Err(e) => {
                    self.drop_peer(session, &format!("bad header {}: {e}", h.height));
                    return;
                }
            }
        }
        self.maybe_download();
        if full && self.pending.len() < MAX_PENDING {
            self.request_headers(session);
        }
    }

    fn on_new_header(&mut self, session: u32, header: Header) {
        if !self.cfg.follow || self.fast.is_some() {
            return;
        }
        let hash = header.hash();
        if self.backend.header_by_hash(&hash).is_some()
            || self.pending.iter().any(|p| p.hash() == hash)
        {
            return;
        }
        match self.queue(header) {
            Ok(true) => self.maybe_download(),
            // Its parent is unknown: ask this peer where its chain leaves ours
            // (or, for a new node, fast sync).
            Ok(false) => {
                if self.pending.is_empty() && !self.maybe_fast_sync() {
                    self.request_headers(session);
                }
            }
            Err(e) => self.drop_peer(session, &format!("bad header {}: {e}", header.height)),
        }
    }

    fn maybe_download(&mut self) {
        if self.download.is_some() || self.fast.is_some() {
            return;
        }
        let Some(&header) = self.pending.front() else {
            return;
        };
        // Prefer a peer known to have the block; skip ones that said they
        // don't.
        let may_have = |s: &u32| self.lacks.get(s).is_none_or(|&l| header.height > l);
        let Some(session) = self
            .peers
            .iter()
            .filter(|(s, p)| p.tip.height >= header.height && may_have(s))
            .map(|(&s, _)| s)
            .next()
            .or_else(|| self.peers.keys().copied().find(may_have))
        else {
            return;
        };
        self.start_download(Kind::Block, header, session);
    }

    fn on_block_info(&mut self, session: u32, hash: Hash, size: u32, root: Hash) {
        let Some(d) = &mut self.download else {
            return;
        };
        if d.session != session || d.id != hash || d.asm.is_some() {
            return;
        }
        let size = size as usize;
        if size + HEADER_BYTES > MAX_BLOCK_SIZE
            || chunk::object_hash(domain::BLOCK_BODY, size, &root) != d.header.body
        {
            self.drop_peer(session, "block info doesn't match the header's body hash");
            return;
        }
        d.asm = Some(Assembly::new(size, root));
        d.last_progress = Instant::now();
        self.request_chunks();
    }

    fn request_chunks(&mut self) {
        let Some(d) = &mut self.download else {
            return;
        };
        let Some(asm) = &d.asm else {
            return;
        };
        let room = d.window.saturating_sub(d.inflight.len());
        if room == 0 {
            return;
        }
        let want: Vec<usize> = asm
            .wanted(d.window + d.inflight.len())
            .into_iter()
            .filter(|i| !d.inflight.contains_key(i))
            .take(room)
            .collect();
        let now = Instant::now();
        d.requests += 1;
        let mut ranges: Vec<(u32, u16)> = Vec::new();
        for &i in &want {
            d.inflight.insert(i, (now, d.requests));
            match ranges.last_mut() {
                Some((first, count))
                    if *first as usize + *count as usize == i && *count < u16::MAX =>
                {
                    *count += 1
                }
                _ => ranges.push((i as u32, 1)),
            }
        }
        let (session, hash) = (d.session, d.id);
        for batch in ranges.chunks(MAX_RANGES) {
            self.send_peer(
                session,
                &Message::GetBlockChunks {
                    hash,
                    ranges: batch.to_vec(),
                },
            );
        }
    }

    fn on_chunk(&mut self, session: u32, hash: Hash, index: u32, data: &[u8]) {
        if self.tx_fetches.contains_key(&hash) {
            self.on_tx_chunk(session, hash, index, data);
            return;
        }
        let Some(d) = &mut self.download else {
            return;
        };
        if d.session != session || d.id != hash {
            return;
        }
        let Some(asm) = &mut d.asm else {
            return;
        };
        let index = index as usize;
        match asm.add(index, data) {
            Ok(new) => {
                let now = Instant::now();
                if let Some((at, request)) = d.inflight.remove(&index) {
                    let sample = now - at;
                    d.srtt = Some(d.srtt.map_or(sample, |s| (s * 7 + sample) / 8));
                    // Chunks of earlier requests still missing were lost:
                    // ask again now, and slow down.
                    let before = d.inflight.len();
                    d.inflight.retain(|_, (_, r)| *r >= request);
                    if d.inflight.len() < before {
                        d.window = (d.window / 2).max(4);
                    }
                }
                if new {
                    d.last_progress = now;
                    d.window = (d.window + 1).min(MAX_WINDOW);
                }
                if asm.is_complete() {
                    self.finish_download();
                } else {
                    self.request_chunks();
                }
            }
            Err(ChunkError::Shape) => {}
            Err(e) => self.drop_peer(session, &format!("bad chunk ({e:?})")),
        }
    }

    fn finish_download(&mut self) {
        let mut d = self.download.take().expect("a download");
        let asm = d.asm.take().expect("info");
        let chunks = asm.layout().chunks();
        let bytes = asm.finish().expect("complete");
        let secs = d.started.elapsed().as_secs_f64();
        let from = self
            .peers
            .get(&d.session)
            .map_or("?".into(), |p| p.addr.to_string());
        let t = Instant::now();
        match d.kind {
            Kind::Block => self.finish_block(&d, &bytes, chunks, secs, &from),
            Kind::Base => {
                let Some(FastSync::Base { session, info }) = self.fast.take() else {
                    return;
                };
                match self
                    .backend
                    .check_base(&info.parent, info.parent_work, &info.base, &bytes)
                {
                    Ok(()) => {
                        (self.log)(format!(
                            "fast sync: base block {} {} from {from} in {secs:.2} s; its chain proof (covering \
                             the {} block{} before it) checked in {:.1} ms; downloading the state from peers",
                            info.base.height,
                            short(&info.base.hash()),
                            info.parent.height,
                            if info.parent.height == 1 { "" } else { "s" },
                            t.elapsed().as_secs_f64() * 1e3,
                        ));
                        if let Err(e) = self.backend.begin_state(&info.base) {
                            (self.log)(format!(
                                "fast sync: couldn't start the state download: {e}"
                            ));
                            return;
                        }
                        let now = Instant::now();
                        self.fast = Some(FastSync::State(Box::new(StateSync {
                            info: *info,
                            base: info.base.hash(),
                            body: bytes,
                            inflight: HashMap::new(),
                            refused: HashSet::new(),
                            replies: Vec::new(),
                            nodes: 0,
                            bytes: 0,
                            from: HashSet::new(),
                            started: now,
                            last_progress: now,
                            last_report: now,
                        })));
                        self.fill_state();
                    }
                    Err(Rejected::Invalid(e)) => {
                        self.drop_peer(session, &format!("bad fast-sync base: {e}"))
                    }
                    Err(Rejected::Failed(e)) => {
                        (self.log)(format!("fast sync: couldn't check the base: {e}"))
                    }
                }
            }
        }
    }

    // State download.

    /// Sends state-tree requests to the peers that may serve the base's
    /// state, up to [`STATE_WINDOW`] outstanding each (the least loaded
    /// first), and installs the state once it's all stored.
    fn fill_state(&mut self) {
        let Some(FastSync::State(s)) = &self.fast else {
            return;
        };
        let mut load: HashMap<u32, usize> = self
            .peers
            .iter()
            .filter(|(id, p)| {
                let h = s.info.base.height;
                p.tip.state_from != 0
                    && p.tip.state_from <= h
                    && p.tip.height >= h
                    && !s.refused.contains(id)
            })
            .map(|(&id, _)| (id, 0))
            .collect();
        for (sid, _) in s.inflight.values() {
            if let Some(n) = load.get_mut(sid) {
                *n += 1;
            }
        }
        let room: usize = load.values().map(|n| STATE_WINDOW.saturating_sub(*n)).sum();
        let wanted = if room == 0 {
            vec![]
        } else {
            let skip: HashSet<(u8, u64)> = s.inflight.keys().copied().collect();
            self.backend.state_wanted(room, &skip)
        };
        if wanted.is_empty() {
            if s.inflight.is_empty() && s.replies.is_empty() && self.backend.state_done() {
                self.finish_state();
            }
            return;
        }
        let base = s.base;
        let mut sends = Vec::with_capacity(wanted.len());
        for id in wanted {
            let Some((&sid, n)) = load
                .iter_mut()
                .filter(|(_, n)| **n < STATE_WINDOW)
                .min_by_key(|(_, n)| **n)
            else {
                break;
            };
            *n += 1;
            sends.push((sid, id));
        }
        let now = Instant::now();
        if let Some(FastSync::State(s)) = &mut self.fast {
            for &(sid, id) in &sends {
                s.inflight.insert(id, (sid, now));
            }
        }
        for (sid, (level, index)) in sends {
            self.send_peer(sid, &Message::GetState { base, level, index });
        }
    }

    fn on_state(&mut self, session: u32, base: Hash, level: u8, index: u64, data: Vec<u8>) {
        let Some(FastSync::State(s)) = &mut self.fast else {
            return;
        };
        if s.base != base {
            return;
        }
        // From whichever peer: it's checked against the node's hash.
        s.inflight.remove(&(level, index));
        s.replies.push((session, level, index, data));
        if s.replies.len() >= STATE_BATCH {
            self.flush_state();
            self.fill_state();
        }
    }

    /// Stores the replies received so far, dropping peers that sent bad
    /// ones.
    fn flush_state(&mut self) {
        let Some(FastSync::State(s)) = &mut self.fast else {
            return;
        };
        if s.replies.is_empty() {
            return;
        }
        let (sessions, items): (Vec<u32>, Vec<StateReply>) = std::mem::take(&mut s.replies)
            .into_iter()
            .map(|(sid, l, i, d)| (sid, (l, i, d)))
            .unzip();
        let results = match self.backend.put_state(&items) {
            Ok(r) => r,
            Err(e) => {
                (self.log)(format!("fast sync: couldn't store the state: {e}"));
                self.fast = None;
                return;
            }
        };
        let now = Instant::now();
        let mut bad = Vec::new();
        if let Some(FastSync::State(s)) = &mut self.fast {
            for ((sid, (_, _, data)), r) in sessions.iter().zip(&items).zip(results) {
                match r {
                    SyncPut::Stored => {
                        s.nodes += 1;
                        s.bytes += data.len() as u64;
                        s.last_progress = now;
                        s.from.insert(*sid);
                    }
                    SyncPut::Unwanted => {}
                    SyncPut::Bad => bad.push(*sid),
                }
            }
        }
        bad.sort_unstable();
        bad.dedup();
        for sid in bad {
            self.drop_peer(sid, "sent a state-tree node that doesn't match its hash");
        }
    }

    /// Makes the complete download the state and starts following the
    /// chain from the base.
    fn finish_state(&mut self) {
        let Some(FastSync::State(s)) = self.fast.take() else {
            return;
        };
        let t = Instant::now();
        match self
            .backend
            .install(&s.info.parent, s.info.parent_work, &s.info.base, &s.body)
        {
            Ok(()) => {
                (self.log)(format!(
                    "fast sync: state at height {} downloaded ({} tree nodes, {} bytes, from {} peer{} in \
                     {:.2} s), built and checked against the header in {:.1} ms; following from there",
                    s.info.base.height,
                    s.nodes,
                    s.bytes,
                    s.from.len(),
                    if s.from.len() == 1 { "" } else { "s" },
                    s.started.elapsed().as_secs_f64(),
                    t.elapsed().as_secs_f64() * 1e3,
                ));
                self.announce_tip(None);
                self.maybe_sync();
            }
            Err(e) => (self.log)(format!("fast sync: couldn't install the state: {e}")),
        }
    }

    fn finish_block(&mut self, d: &Download, body: &[u8], chunks: usize, secs: f64, from: &str) {
        let t = Instant::now();
        match self.backend.accept(&d.header, body) {
            Ok(added) => {
                self.pending.pop_front();
                let what = match added {
                    Added::Known => "already had it".to_string(),
                    Added::Extended => "new tip".to_string(),
                    Added::Side => "stored on a side branch".to_string(),
                    Added::Reorg { reverted, applied } => {
                        format!("reorg: reverted {reverted}, applied {applied}; new tip")
                    }
                };
                (self.log)(format!(
                    "a peer's block {} {}: {} bytes in {chunks} chunks from {from} in {secs:.2} s; verified in {:.1} ms; {what}",
                    d.header.height,
                    short(&d.header.hash()),
                    body.len() + HEADER_BYTES,
                    t.elapsed().as_secs_f64() * 1e3,
                ));
                self.announce(&d.header, Some(d.session));
                self.maybe_download();
                self.maybe_sync();
            }
            Err(Rejected::Invalid(e)) => {
                // The header checked, so its sender vouched for a block
                // that doesn't verify.
                self.pending.clear();
                self.drop_peer(
                    d.session,
                    &format!("block {} is invalid: {e}", d.header.height),
                );
            }
            Err(Rejected::Failed(e)) => {
                self.pending.clear();
                (self.log)(format!("couldn't store block {}: {e}", d.header.height));
            }
        }
    }

    /// Tells every peer (but `except`) about a new block, and our tip.
    fn announce(&mut self, header: &Header, except: Option<u32>) {
        let tip = self.backend.tip();
        let sessions: Vec<u32> = self
            .peers
            .keys()
            .copied()
            .filter(|&s| Some(s) != except)
            .collect();
        for s in sessions {
            self.send_peer(s, &Message::NewHeader { header: *header });
            self.send_peer(s, &Message::Status { tip });
        }
    }

    /// Tells every peer (but `except`) about our tip.
    fn announce_tip(&mut self, except: Option<u32>) {
        let tip = self.backend.tip();
        if let Some(header) = self.backend.header_at(tip.height) {
            self.announce(&header, except);
        }
    }

    // Transactions.

    /// Notes who has transactions `ids`, queueing the new ones for
    /// fetching.
    fn on_tx_inv(&mut self, session: u32, ids: &[Hash]) {
        for id in ids {
            if let Some(f) = self.tx_fetches.get_mut(id) {
                if f.session != session && !f.others.contains(&session) {
                    f.others.push(session);
                }
                continue;
            }
            if let Some(from) = self.tx_queued.get_mut(id) {
                if !from.contains(&session) {
                    from.push(session);
                }
                continue;
            }
            if self.tx_queue.len() >= MAX_TX_QUEUE || self.backend.knows_tx(id) {
                continue;
            }
            self.tx_queue.push_back(*id);
            self.tx_queued.insert(*id, vec![session]);
        }
        self.start_tx_fetches();
    }

    /// Fetches queued transactions while there are free slots, each from
    /// the first peer still connected that announced it, unless the chain
    /// is being synced.
    fn start_tx_fetches(&mut self) {
        if self.chain_busy() {
            return;
        }
        let mut in_flight: usize = self
            .tx_fetches
            .values()
            .map(|f| f.asm.as_ref().map_or(TX_SIZE_GUESS, Assembly::size))
            .sum();
        while self.tx_fetches.len() < MAX_TX_FETCHES && in_flight < TX_BYTES_IN_FLIGHT {
            in_flight += TX_SIZE_GUESS;
            let Some(id) = self.tx_queue.pop_front() else {
                return;
            };
            let mut from = self.tx_queued.remove(&id).unwrap_or_default();
            from.retain(|s| self.peers.contains_key(s));
            if from.is_empty() || self.backend.knows_tx(&id) {
                continue;
            }
            let session = from.remove(0);
            from.reverse();
            self.tx_fetches.insert(
                id,
                TxFetch {
                    session,
                    others: from,
                    asm: None,
                    asked: Instant::now(),
                    tries: 1,
                },
            );
            self.send_peer(session, &Message::GetTx { id });
        }
    }

    fn on_tx_info(&mut self, session: u32, id: Hash, size: u32, root: Hash) {
        let Some(f) = self.tx_fetches.get_mut(&id) else {
            return;
        };
        if f.session != session || f.asm.is_some() {
            return;
        }
        let size = size as usize;
        if size > crate::transaction::MAX_TX_BYTES
            || chunk::object_hash(domain::TX_ID, size, &root) != id
        {
            self.drop_peer(session, "transaction info doesn't match its id");
            return;
        }
        f.asm = Some(Assembly::new(size, root));
        f.asked = Instant::now();
        self.request_tx_chunks(id);
    }

    /// Asks for every chunk of transaction `id` still missing.
    fn request_tx_chunks(&mut self, id: Hash) {
        let Some(f) = self.tx_fetches.get(&id) else {
            return;
        };
        let Some(asm) = &f.asm else {
            return;
        };
        let mut ranges: Vec<(u32, u16)> = Vec::new();
        for i in asm.wanted(MAX_SERVE) {
            match ranges.last_mut() {
                Some((first, count)) if *first as usize + *count as usize == i => *count += 1,
                _ => ranges.push((i as u32, 1)),
            }
        }
        let session = f.session;
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

    fn on_tx_chunk(&mut self, session: u32, id: Hash, index: u32, data: &[u8]) {
        let Some(f) = self.tx_fetches.get_mut(&id) else {
            return;
        };
        let Some(asm) = &mut f.asm else {
            return;
        };
        if f.session != session {
            return;
        }
        let had_leaves = asm.has_leaves();
        match asm.add(index as usize, data) {
            Ok(_) if asm.is_complete() => {
                let f = self.tx_fetches.remove(&id).expect("fetching");
                let bytes = f.asm.and_then(Assembly::finish).expect("complete");
                self.tx_done.push((id, session, bytes));
            }
            // The leaf hashes are in: ask for the data now, not at the
            // retry timer.
            Ok(_) if !had_leaves && asm.has_leaves() => self.request_tx_chunks(id),
            Ok(_) | Err(ChunkError::Shape) => {}
            Err(e) => self.drop_peer(session, &format!("bad transaction chunk ({e:?})")),
        }
    }

    /// Checks the fetched transactions (signatures and all, in parallel) and
    /// adds them to the mempool.
    fn add_fetched_txs(&mut self) {
        if self.tx_done.is_empty() {
            return;
        }
        let done = std::mem::take(&mut self.tx_done);
        let backend = &self.backend;
        let results = crate::par::par_map(done.len(), 4, |i| backend.add_tx(&done[i].2));
        for ((id, session, bytes), result) in done.into_iter().zip(results) {
            match result {
                Ok(true) => {
                    let from = self
                        .peers
                        .get(&session)
                        .map_or("?".into(), |p| p.addr.to_string());
                    (self.log)(format!(
                        "transaction {} ({} bytes) from {from} added to the mempool",
                        short(&id),
                        bytes.len()
                    ));
                    self.tx_to_announce.push((id, Some(session)));
                }
                Ok(false) => {}
                Err(Rejected::Invalid(e)) => {
                    self.drop_peer(session, &format!("invalid transaction {}: {e}", short(&id)))
                }
                Err(Rejected::Failed(e)) => {
                    (self.log)(format!("transaction {} not added: {e}", short(&id)))
                }
            }
        }
    }

    /// Asks the next peer that announced transaction `id`, or gives up on it.
    fn next_tx_source(&mut self, id: Hash) {
        let Some(f) = self.tx_fetches.get_mut(&id) else {
            return;
        };
        match f.others.pop() {
            Some(s) if self.peers.contains_key(&s) => {
                f.session = s;
                f.asm = None;
                f.asked = Instant::now();
                f.tries = 1;
                self.send_peer(s, &Message::GetTx { id });
            }
            Some(_) => self.next_tx_source(id),
            None => {
                self.tx_fetches.remove(&id);
            }
        }
    }

    // Timers.

    fn tick(&mut self) {
        let now = Instant::now();
        self.dial(now);

        // Keepalive.
        let silent: Vec<u32> = self
            .peers
            .iter()
            .filter(|(_, p)| now - p.last_seen > PEER_TIMEOUT)
            .map(|(&s, _)| s)
            .collect();
        for s in silent {
            self.drop_peer(s, "timed out");
        }
        let due: Vec<u32> = self
            .peers
            .iter()
            .filter(|(_, p)| now - p.last_ping >= PING_EVERY)
            .map(|(&s, _)| s)
            .collect();
        for s in due {
            if let Some(p) = self.peers.get_mut(&s) {
                p.last_ping = now;
            }
            let nonce = self.rng.next_element().as_u32() as u64;
            self.send_peer(s, &Message::Ping { nonce });
        }

        // More addresses now and then.
        if now - self.last_peers_ask >= PEERS_EVERY && !self.peers.is_empty() {
            self.last_peers_ask = now;
            let sessions: Vec<u32> = self.peers.keys().copied().collect();
            let s = sessions[self.rng.next_element().as_u32() as usize % sessions.len()];
            self.ask_peers(s);
        }
        if self.book_dirty && now - self.last_save >= SAVE_EVERY {
            self.save_book();
        }

        // Fast sync: re-ask, or try another peer.
        match &mut self.fast {
            Some(FastSync::Asking {
                session,
                height,
                asked,
                tries,
                ..
            }) if now - *asked > REQUEST_TIMEOUT => {
                let (s, h) = (*session, *height);
                if *tries >= SNAPSHOT_TRIES {
                    self.no_snapshot.insert(s);
                    self.fast = None;
                } else {
                    *tries += 1;
                    *asked = now;
                    self.send_peer(s, &Message::GetSnapshot { height: h });
                }
            }
            // Its download stalled and was dropped: start over.
            Some(FastSync::Base { .. }) if self.download.is_none() => self.fast = None,
            Some(FastSync::State(s)) => {
                // Lost requests go out again, to whoever has room.
                s.inflight.retain(|_, (_, at)| now - *at < STATE_TIMEOUT);
                if now - s.last_progress > STALL_TIMEOUT {
                    (self.log)(
                        "fast sync: the state download stalled; starting over (what's stored is kept)"
                            .into(),
                    );
                    self.fast = None;
                } else {
                    if now - s.last_report >= REPORT_EVERY {
                        s.last_report = now;
                        (self.log)(format!(
                            "fast sync: state at height {}: {} tree nodes ({} bytes) so far, from {} peer{}",
                            s.info.base.height,
                            s.nodes,
                            s.bytes,
                            s.from.len(),
                            if s.from.len() == 1 { "" } else { "s" },
                        ));
                    }
                    self.flush_state();
                    self.fill_state();
                }
            }
            _ => {}
        }

        // The download: re-ask, re-request lost chunks, or give up.
        if let Some(d) = &mut self.download {
            if now - d.last_progress > STALL_TIMEOUT {
                let s = d.session;
                self.download = None;
                self.drop_peer(s, "download stalled");
            } else if d.asm.is_none() {
                if now - d.asked > REQUEST_TIMEOUT {
                    d.asked = now;
                    let (s, hash) = (d.session, d.id);
                    self.send_peer(s, &Message::GetBlock { hash });
                }
            } else {
                let rto = d
                    .srtt
                    .map_or(INITIAL_RTO, |s| (s * RTO_RTTS).clamp(MIN_RTO, MAX_RTO));
                let before = d.inflight.len();
                d.inflight.retain(|_, (at, _)| now - *at < rto);
                if d.inflight.len() < before {
                    d.window = (d.window / 2).max(4);
                }
                self.request_chunks();
            }
        }
        // Transactions: check the fetched ones, announce the new ones, ask
        // again for late ones (or try another peer), and start queued
        // fetches.
        self.add_fetched_txs();
        if !self.tx_to_announce.is_empty() {
            let new = std::mem::take(&mut self.tx_to_announce);
            let sessions: Vec<u32> = self.peers.keys().copied().collect();
            for s in sessions {
                let ids: Vec<Hash> = new
                    .iter()
                    .filter(|(_, from)| *from != Some(s))
                    .map(|(id, _)| *id)
                    .collect();
                for batch in ids.chunks(MAX_TX_IDS) {
                    self.send_peer(
                        s,
                        &Message::TxInv {
                            ids: batch.to_vec(),
                        },
                    );
                }
            }
        }
        let late: Vec<Hash> = self
            .tx_fetches
            .iter()
            .filter(|(_, f)| now - f.asked > REQUEST_TIMEOUT)
            .map(|(id, _)| *id)
            .collect();
        for id in late {
            let f = self.tx_fetches.get_mut(&id).expect("fetching");
            if f.tries >= TX_TRIES {
                self.next_tx_source(id);
                continue;
            }
            f.tries += 1;
            f.asked = now;
            if f.asm.is_some() {
                self.request_tx_chunks(id);
            } else {
                let s = f.session;
                self.send_peer(s, &Message::GetTx { id });
            }
        }
        self.start_tx_fetches();
        self.maybe_download();
        self.maybe_sync();
        self.check_synced(now);
    }

    /// Whether the chain is being synced or a block fetched: transactions
    /// wait (they're checked against our state, which isn't the peers' yet).
    fn chain_busy(&self) -> bool {
        self.fast.is_some()
            || self.download.is_some()
            || !self.pending.is_empty()
            || self.header_request.is_some()
            || self.best_peer(&self.unhelpful()).is_some()
    }

    /// Sets the synced flag once nothing is left to fetch from any peer (or
    /// no peer has turned up for a while).
    fn check_synced(&mut self, now: Instant) {
        if self.synced.load(Ordering::Relaxed) {
            return;
        }
        let busy = self.chain_busy()
            // Peers' pending transactions too, so a miner starts with them.
            || !self.tx_queue.is_empty()
            || !self.tx_fetches.is_empty()
            || !self.tx_done.is_empty();
        let alone = self.peers.is_empty() && now - self.started >= ALONE_AFTER;
        if (!self.peers.is_empty() && !busy) || alone {
            let tip = self.backend.tip();
            (self.log)(if alone {
                format!(
                    "no peers after {} s; carrying on alone at height {}",
                    ALONE_AFTER.as_secs(),
                    tip.height
                )
            } else {
                format!(
                    "caught up with peers: tip {} at height {}",
                    short(&tip.hash),
                    tip.height
                )
            });
            self.synced.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{Sender, channel};
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Blocks checked only for links and body hashes, with fork choice by
    /// work (one unit a block; the first seen wins ties). Its "state" at a
    /// block is the list of hashes before it, in groups of [`GROUP`]: the
    /// root node (level 1) lists the groups' digests, and the header's
    /// state root is the root node's digest.
    struct MockChain {
        /// Header, body (none for genesis and a fast-sync base's parent),
        /// work.
        blocks: HashMap<Hash, (Header, Option<Vec<u8>>, u128)>,
        /// The main chain by height; below a fast-sync base, hashes without
        /// blocks.
        main: Vec<Hash>,
        base: u64,
        /// Serves the state at this many blocks below the tip, at most.
        limit: u64,
        /// Serves wrong state nodes.
        corrupt: bool,
        /// State nodes served.
        served: u64,
        sync: Option<MockSync>,
        /// The mempool: any bytes are a transaction, unless they start with
        /// 0xff (invalid).
        txs: HashMap<Hash, Vec<u8>>,
    }

    /// A state download: the base, the nodes still wanted (with their
    /// digests) and the groups so far.
    struct MockSync {
        height: u64,
        base: Hash,
        todo: std::collections::BTreeMap<(u8, u64), Hash>,
        groups: std::collections::BTreeMap<u64, Vec<Hash>>,
    }

    const GROUP: usize = 8;

    fn digest(bytes: &[u8]) -> Hash {
        chunk::hash_of(domain::BLOCK_BODY, bytes)
    }

    fn group_bytes(hashes: &[Hash]) -> Vec<u8> {
        hashes.iter().flat_map(hash::to_bytes).collect()
    }

    /// The root node of the state that is `hashes`: its groups' digests.
    fn mock_root_node(hashes: &[Hash]) -> Vec<u8> {
        hashes
            .chunks(GROUP)
            .flat_map(|g| hash::to_bytes(&digest(&group_bytes(g))))
            .collect()
    }

    fn parse_hashes(bytes: &[u8]) -> Option<Vec<Hash>> {
        if !bytes.len().is_multiple_of(hash::HASH_BYTES) {
            return None;
        }
        bytes
            .chunks(hash::HASH_BYTES)
            .map(|b| hash::from_bytes(b.try_into().ok()?))
            .collect()
    }

    #[derive(Clone)]
    struct Mock(Arc<Mutex<MockChain>>);

    impl Mock {
        fn new() -> Self {
            let g = Header::default();
            let blocks = HashMap::from([(g.hash(), (g, None, 0))]);
            Self(Arc::new(Mutex::new(MockChain {
                blocks,
                main: vec![g.hash()],
                base: 0,
                limit: u64::MAX,
                corrupt: false,
                served: 0,
                sync: None,
                txs: HashMap::new(),
            })))
        }

        fn has_tx(&self, bytes: &[u8]) -> bool {
            let id = crate::transaction::Transaction::id_of(bytes);
            self.0.lock().unwrap().txs.contains_key(&id)
        }

        fn base(&self) -> u64 {
            self.0.lock().unwrap().base
        }

        fn served(&self) -> u64 {
            self.0.lock().unwrap().served
        }

        fn height(&self) -> u64 {
            self.0.lock().unwrap().main.len() as u64 - 1
        }

        /// Adds a block on `parent` (the tip if `None`) whose body is `size`
        /// bytes derived from `seed`.
        fn mine_on(&self, parent: Option<Hash>, size: usize, seed: u8) -> Header {
            let (parent, before) = {
                let c = self.0.lock().unwrap();
                let parent = c.blocks[&parent.unwrap_or(*c.main.last().unwrap())].0;
                // The hashes up to the parent, by walking back.
                let mut before = vec![parent.hash()];
                let mut h = parent;
                while h.height > 0 {
                    h = c.blocks[&h.prev].0;
                    before.push(h.hash());
                }
                before.reverse();
                (parent, before)
            };
            let body: Vec<u8> = (0..size)
                .map(|i| (i as u8).wrapping_mul(seed) ^ (i >> 8) as u8)
                .collect();
            let header = Header {
                height: parent.height + 1,
                prev: parent.hash(),
                timestamp: parent.timestamp + 1,
                body: chunk::hash_of(domain::BLOCK_BODY, &body),
                nonce: seed as u64,
                state: crate::state::State {
                    root: digest(&mock_root_node(&before)),
                    size: before.len() as u64,
                },
            };
            self.accept(&header, &body).unwrap();
            header
        }

        fn mine(&self, size: usize, seed: u8) -> Header {
            self.mine_on(None, size, seed)
        }
    }

    impl Backend for Mock {
        fn tip(&self) -> Tip {
            let c = self.0.lock().unwrap();
            let (h, _, work) = &c.blocks[c.main.last().unwrap()];
            Tip {
                height: h.height,
                hash: h.hash(),
                work: *work,
                state_from: if h.height == 0 {
                    0
                } else {
                    h.height.saturating_sub(c.limit).max(c.base).max(1)
                },
            }
        }

        fn header_at(&self, height: u64) -> Option<Header> {
            let c = self.0.lock().unwrap();
            c.main
                .get(height as usize)
                .and_then(|h| c.blocks.get(h))
                .map(|b| b.0)
        }

        fn height_of(&self, hash: &Hash) -> Option<u64> {
            let c = self.0.lock().unwrap();
            c.main
                .iter()
                .position(|h| h == hash && c.blocks.contains_key(h))
                .map(|i| i as u64)
        }

        fn header_by_hash(&self, hash: &Hash) -> Option<Header> {
            self.0.lock().unwrap().blocks.get(hash).map(|b| b.0)
        }

        fn body(&self, hash: &Hash) -> Option<Vec<u8>> {
            self.0
                .lock()
                .unwrap()
                .blocks
                .get(hash)
                .and_then(|b| b.1.clone())
        }

        fn check_header(&self, parent: &Header, header: &Header) -> Result<(), String> {
            if header.prev == parent.hash() && header.timestamp > parent.timestamp {
                Ok(())
            } else {
                Err("bad link".into())
            }
        }

        fn accept(&self, header: &Header, body: &[u8]) -> Result<Added, Rejected> {
            let mut c = self.0.lock().unwrap();
            let hash = header.hash();
            if c.blocks.contains_key(&hash) {
                return Ok(Added::Known);
            }
            let Some(&(_, _, parent_work)) = c.blocks.get(&header.prev) else {
                return Err(Rejected::Failed("unknown parent".into()));
            };
            if chunk::hash_of(domain::BLOCK_BODY, body) != header.body {
                return Err(Rejected::Invalid("bad body".into()));
            }
            let work = parent_work + 1;
            c.blocks.insert(hash, (*header, Some(body.to_vec()), work));
            let tip_work = c.blocks[c.main.last().unwrap()].2;
            if work <= tip_work {
                return Ok(Added::Side);
            }
            let old_tip = *c.main.last().unwrap();
            let mut branch = vec![hash];
            while !c.main.contains(&c.blocks[branch.last().unwrap()].0.prev) {
                let prev = c.blocks[branch.last().unwrap()].0.prev;
                branch.push(prev);
            }
            let fork = c.blocks[&c.blocks[branch.last().unwrap()].0.prev].0.height as usize;
            let reverted = (c.main.len() - 1 - fork) as u64;
            c.main.truncate(fork + 1);
            let applied = branch.len() as u64;
            branch.reverse();
            c.main.extend(branch);
            Ok(if header.prev == old_tip {
                Added::Extended
            } else {
                Added::Reorg { reverted, applied }
            })
        }

        fn snapshot(&self, height: u64) -> Option<Snapshot> {
            let c = self.0.lock().unwrap();
            let tip = c.main.len() as u64 - 1;
            if height == 0 || height < c.base || height > tip || tip - height > c.limit {
                return None;
            }
            let base = c.blocks[&c.main[height as usize]].0;
            let (parent, _, parent_work) = c.blocks[&c.main[height as usize - 1]].clone();
            Some(Snapshot {
                parent,
                base,
                parent_work,
            })
        }

        fn state_nodes(&self, base: &Hash, level: u8, index: u64) -> Option<Vec<u8>> {
            let height = self.height_of(base)?;
            self.snapshot(height)?;
            let mut c = self.0.lock().unwrap();
            let before = &c.main[..height as usize];
            let mut data = match (level, index) {
                (1, 0) => mock_root_node(before),
                (0, g) => group_bytes(before.chunks(GROUP).nth(g as usize)?),
                _ => return None,
            };
            if c.corrupt {
                data[0] ^= 1;
            }
            c.served += 1;
            Some(data)
        }

        fn check_base(
            &self,
            parent: &Header,
            _: u128,
            base: &Header,
            body: &[u8],
        ) -> Result<(), Rejected> {
            if base.prev == parent.hash() && chunk::hash_of(domain::BLOCK_BODY, body) == base.body {
                Ok(())
            } else {
                Err(Rejected::Invalid("bad base".into()))
            }
        }

        fn begin_state(&self, base: &Header) -> Result<(), String> {
            let mut c = self.0.lock().unwrap();
            if c.sync.as_ref().is_none_or(|s| s.base != base.hash()) {
                c.sync = Some(MockSync {
                    height: base.height,
                    base: base.hash(),
                    todo: [((1, 0), base.state.root)].into(),
                    groups: Default::default(),
                });
            }
            Ok(())
        }

        fn state_progress(&self) -> Option<(u64, Hash)> {
            let c = self.0.lock().unwrap();
            c.sync.as_ref().map(|s| (s.height, s.base))
        }

        fn state_wanted(&self, n: usize, skip: &HashSet<(u8, u64)>) -> Vec<(u8, u64)> {
            let c = self.0.lock().unwrap();
            let Some(s) = &c.sync else {
                return vec![];
            };
            s.todo
                .keys()
                .filter(|id| !skip.contains(id))
                .take(n)
                .copied()
                .collect()
        }

        fn put_state(&self, replies: &[StateReply]) -> Result<Vec<SyncPut>, String> {
            let mut c = self.0.lock().unwrap();
            let s = c.sync.as_mut().ok_or("no download")?;
            Ok(replies
                .iter()
                .map(|(level, index, data)| {
                    let Some(&want) = s.todo.get(&(*level, *index)) else {
                        return SyncPut::Unwanted;
                    };
                    let Some(hashes) = parse_hashes(data).filter(|_| digest(data) == want) else {
                        return SyncPut::Bad;
                    };
                    s.todo.remove(&(*level, *index));
                    if *level == 1 {
                        for (g, d) in hashes.into_iter().enumerate() {
                            s.todo.insert((0, g as u64), d);
                        }
                    } else {
                        s.groups.insert(*index, hashes);
                    }
                    SyncPut::Stored
                })
                .collect())
        }

        fn state_done(&self) -> bool {
            let c = self.0.lock().unwrap();
            c.sync.as_ref().is_some_and(|s| s.todo.is_empty())
        }

        fn install(
            &self,
            parent: &Header,
            parent_work: u128,
            base: &Header,
            body: &[u8],
        ) -> Result<(), Rejected> {
            self.check_base(parent, parent_work, base, body)?;
            let mut c = self.0.lock().unwrap();
            let s = c
                .sync
                .take()
                .ok_or(Rejected::Failed("no download".into()))?;
            let mut hashes: Vec<Hash> = s.groups.into_values().flatten().collect();
            if s.base != base.hash()
                || hashes.len() != base.height as usize
                || hashes.last() != Some(&parent.hash())
            {
                return Err(Rejected::Invalid("bad state".into()));
            }
            assert_eq!(c.main.len(), 1, "installing on a new chain");
            hashes.push(base.hash());
            c.blocks.insert(parent.hash(), (*parent, None, parent_work));
            c.blocks
                .insert(base.hash(), (*base, Some(body.to_vec()), parent_work + 1));
            c.main = hashes;
            c.base = base.height;
            Ok(())
        }

        fn tx(&self, id: &Hash) -> Option<Vec<u8>> {
            self.0.lock().unwrap().txs.get(id).cloned()
        }

        fn knows_tx(&self, id: &Hash) -> bool {
            self.0.lock().unwrap().txs.contains_key(id)
        }

        fn add_tx(&self, bytes: &[u8]) -> Result<bool, Rejected> {
            if bytes.first() == Some(&0xff) {
                return Err(Rejected::Invalid("bad tx".into()));
            }
            let id = crate::transaction::Transaction::id_of(bytes);
            Ok(self
                .0
                .lock()
                .unwrap()
                .txs
                .insert(id, bytes.to_vec())
                .is_none())
        }

        fn tx_ids(&self) -> Vec<Hash> {
            self.0.lock().unwrap().txs.keys().copied().collect()
        }
    }

    struct Running {
        addr: SocketAddr,
        events: Sender<Event>,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl Drop for Running {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn start(backend: Mock, follow: bool, peers: Vec<SocketAddr>, loss: u32) -> Running {
        start_with(
            backend,
            Config {
                follow,
                peers,
                loss_per_mille: loss,
                ..Config::new(*b"test")
            },
            Arc::new(Mutex::new(vec![])),
        )
    }

    /// A node with `cfg` (listening), its log lines also kept in `lines`.
    fn start_with(backend: Mock, cfg: Config, lines: Arc<Mutex<Vec<String>>>) -> Running {
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let cfg = Config {
            capabilities: cap::LISTENING,
            ..cfg
        };
        let log = Box::new(move |line: String| {
            eprintln!("{line}");
            lines.lock().unwrap().push(line);
        });
        let mut node = Node::new(socket, cfg, backend, log);
        let addr = node.local_addr();
        let (events, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let handle = thread::spawn(move || node.run(&rx, &s));
        Running {
            addr,
            events,
            stop,
            handle: Some(handle),
        }
    }

    fn wait_for(what: &str, secs: u64, mut done: impl FnMut() -> bool) {
        let t = Instant::now();
        while !done() {
            assert!(
                t.elapsed() < Duration::from_secs(secs),
                "timed out waiting for {what}"
            );
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_follower_syncs_and_then_follows_announcements() {
        let a = Mock::new();
        for i in 0..12 {
            // Bodies from 1 byte to ~200 KB.
            a.mine([1, 500, 3000, 50_000, 200_000][i % 5], i as u8 + 1);
        }
        let b = Mock::new();
        let na = start(a.clone(), false, vec![], 0);
        let _nb = start(b.clone(), true, vec![na.addr], 0);
        wait_for("sync", 20, || b.height() == 12);
        assert_eq!(b.tip(), a.tip());
        // A new block is announced and fetched.
        a.mine(120_000, 99);
        na.events.send(Event::NewTip).unwrap();
        wait_for("relay", 10, || b.height() == 13);
        assert_eq!(b.tip(), a.tip());
    }

    #[test]
    fn sync_survives_packet_loss_and_relays_down_a_line() {
        let a = Mock::new();
        for i in 0..4 {
            a.mine(150_000, i + 1);
        }
        let (b, c) = (Mock::new(), Mock::new());
        let na = start(a.clone(), false, vec![], 0);
        // B loses 5% of what it receives; C syncs through B, and only B (one
        // session at most, so it can't dial A once B passes A's address on).
        let nb = start(b.clone(), true, vec![na.addr], 50);
        let _nc = start_with(
            c.clone(),
            Config {
                peers: vec![nb.addr],
                max_peers: 1,
                ..Config::new(*b"test")
            },
            Arc::new(Mutex::new(vec![])),
        );
        wait_for("sync through a lossy hop", 30, || c.height() == 4);
        assert_eq!((b.tip(), c.tip()), (a.tip(), a.tip()));
    }

    #[test]
    fn nodes_converge_on_the_branch_with_the_most_work() {
        // A and B mine different chains from genesis: 3 blocks and 5.
        let (a, b) = (Mock::new(), Mock::new());
        for i in 0..3 {
            a.mine(20_000, 10 + i);
        }
        for i in 0..5 {
            b.mine(20_000, 50 + i);
        }
        let na = start(a.clone(), true, vec![], 0);
        let nb = start(b.clone(), true, vec![na.addr], 0);
        // A reorganizes onto B's heavier branch; B keeps its own.
        wait_for("A to reorg", 20, || a.tip().hash == b.tip().hash);
        assert_eq!(a.height(), 5);
        // A's old blocks are still stored, on a side branch.
        assert_eq!(a.0.lock().unwrap().blocks.len(), 1 + 3 + 5);

        // Both extend the tip at once: equal work, each keeps its own.
        let tip = a.tip().hash;
        let ha = a.mine_on(Some(tip), 1000, 1);
        let hb = b.mine_on(Some(tip), 1000, 2);
        na.events.send(Event::NewTip).unwrap();
        nb.events.send(Event::NewTip).unwrap();
        wait_for("each to store the other's block", 10, || {
            a.header_by_hash(&hb.hash()).is_some() && b.header_by_hash(&ha.hash()).is_some()
        });
        assert_eq!((a.tip().hash, b.tip().hash), (ha.hash(), hb.hash()));
        // The next block on B's side settles it.
        b.mine(1000, 3);
        nb.events.send(Event::NewTip).unwrap();
        wait_for("A to follow B's branch", 10, || {
            a.tip().hash == b.tip().hash
        });
        assert_eq!(a.height(), 7);
    }

    #[test]
    fn cookies_depend_on_the_address() {
        let n = Node::new(
            UdpSocket::bind("127.0.0.1:0").unwrap(),
            Config::new(*b"test"),
            Mock::new(),
            Box::new(|_| {}),
        );
        let a: SocketAddr = "10.0.0.1:5000".parse().unwrap();
        let b: SocketAddr = "10.0.0.1:5001".parse().unwrap();
        let c = n.cookie(&a, now_secs() / 60);
        assert!(n.cookie_ok(&a, &c));
        assert!(!n.cookie_ok(&b, &c));
        // Last minute's still works, older ones don't.
        assert!(n.cookie_ok(&a, &n.cookie(&a, now_secs() / 60 - 1)));
        assert!(!n.cookie_ok(&a, &n.cookie(&a, now_secs() / 60 - 2)));
    }

    #[test]
    fn a_new_node_fast_syncs_and_serves_from_its_base() {
        let a = Mock::new();
        for i in 0..30 {
            a.mine(3000 + i * 100, i as u8 + 1);
        }
        let na = start(a.clone(), false, vec![], 0);
        let b = Mock::new();
        let cfg = |peers, depth| Config {
            peers,
            sync_depth: Some(depth),
            ..Config::new(*b"test")
        };
        let nb = start_with(
            b.clone(),
            cfg(vec![na.addr], 10),
            Arc::new(Mutex::new(vec![])),
        );
        wait_for("B to fast sync", 20, || b.tip().hash == a.tip().hash);
        // B has the blocks from its base (height 20) up, none before.
        let hash_at = |m: &Mock, h: u64| m.0.lock().unwrap().main[h as usize];
        assert_eq!(b.body(&hash_at(&a, 19)), None);
        assert!(b.body(&hash_at(&a, 20)).is_some() && b.body(&hash_at(&a, 30)).is_some());
        assert_eq!(b.0.lock().unwrap().blocks.len(), 1 + 1 + 11);

        // It follows new blocks, and a newer node can fast sync from it.
        a.mine(5000, 77);
        na.events.send(Event::NewTip).unwrap();
        wait_for("B to follow", 10, || b.tip().hash == a.tip().hash);
        let c = Mock::new();
        let _nc = start_with(
            c.clone(),
            cfg(vec![nb.addr], 5),
            Arc::new(Mutex::new(vec![])),
        );
        wait_for("C to fast sync from B", 20, || c.tip().hash == a.tip().hash);
        assert_eq!(c.0.lock().unwrap().base, 26);

        // Asked for a state below its base, B has none: D syncs every
        // block from A instead.
        let d = Mock::new();
        let _nd = start_with(
            d.clone(),
            cfg(vec![nb.addr, na.addr], 25),
            Arc::new(Mutex::new(vec![])),
        );
        wait_for("D to sync", 30, || d.tip().hash == a.tip().hash);
        assert!(d.0.lock().unwrap().base >= 6);
    }

    fn has_line(lines: &Arc<Mutex<Vec<String>>>, needle: &str) -> bool {
        lines.lock().unwrap().iter().any(|l| l.contains(needle))
    }

    #[test]
    fn the_state_comes_from_several_peers_and_a_bad_one_is_dropped() {
        let a = Mock::new();
        for i in 0..200 {
            a.mine(100 + i, i as u8 + 1);
        }
        let na = start(a.clone(), false, vec![], 0);
        // G and E sync every block from A; E then serves bad state nodes.
        let (g, e) = (Mock::new(), Mock::new());
        let ng = start(g.clone(), true, vec![na.addr], 0);
        let ne = start(e.clone(), true, vec![na.addr], 0);
        wait_for("G and E to sync", 30, || {
            g.tip().hash == a.tip().hash && e.tip().hash == a.tip().hash
        });
        e.0.lock().unwrap().corrupt = true;

        let f = Mock::new();
        let lines = Arc::new(Mutex::new(vec![]));
        let _nf = start_with(
            f.clone(),
            Config {
                peers: vec![na.addr, ng.addr, ne.addr],
                sync_depth: Some(10),
                ..Config::new(*b"test")
            },
            lines.clone(),
        );
        wait_for("F to fast sync", 30, || f.tip().hash == a.tip().hash);
        assert_eq!(f.base(), 190);
        // 190 hashes: a root node and 24 groups, spread over the peers.
        assert!(
            a.served() > 0 && g.served() > 0,
            "{} {}",
            a.served(),
            g.served()
        );
        assert!(
            has_line(
                &lines,
                &format!("dropping peer {}: sent a state-tree node", ne.addr)
            ),
            "{:#?}",
            lines.lock().unwrap()
        );
    }

    #[test]
    fn a_state_download_resumes_and_peers_serve_only_recent_states() {
        let a = Mock::new();
        for i in 0..60 {
            a.mine(100, i as u8 + 1);
        }
        // A new node with part of the state at height 50 already stored
        // (as after a restart).
        let f = Mock::new();
        let base = a.header_at(50).unwrap();
        f.begin_state(&base).unwrap();
        let root = a.state_nodes(&base.hash(), 1, 0).unwrap();
        let group = a.state_nodes(&base.hash(), 0, 3).unwrap();
        assert_eq!(
            f.put_state(&[(1, 0, root), (0, 3, group)]).unwrap(),
            vec![SyncPut::Stored; 2]
        );
        let served = a.served();
        let na = start(a.clone(), false, vec![], 0);
        let lines = Arc::new(Mutex::new(vec![]));
        let _nf = start_with(
            f.clone(),
            Config {
                peers: vec![na.addr],
                sync_depth: Some(5),
                ..Config::new(*b"test")
            },
            lines.clone(),
        );
        wait_for("F to resume", 20, || f.tip().hash == a.tip().hash);
        assert!(has_line(&lines, "at height 50, to resume its download"));
        assert_eq!(f.base(), 50);
        // Only the 6 groups it didn't have were fetched.
        assert_eq!(a.served() - served, 6);

        // A serves the state at its last 5 blocks only: asked for one 10
        // down, it has none, and H downloads every block.
        a.0.lock().unwrap().limit = 5;
        let h = Mock::new();
        let lines = Arc::new(Mutex::new(vec![]));
        let _nh = start_with(
            h.clone(),
            Config {
                peers: vec![na.addr],
                sync_depth: Some(10),
                ..Config::new(*b"test")
            },
            lines.clone(),
        );
        wait_for("H to sync", 20, || h.tip().hash == a.tip().hash);
        assert_eq!(h.base(), 0);
        assert!(has_line(&lines, "has no state at height 50"));
    }

    #[test]
    fn transactions_spread_and_bad_ones_get_their_sender_dropped() {
        // A has one transaction when B connects; C hears of B's through B.
        let tx = |n: u8, len: usize| -> Vec<u8> { (0..len).map(|i| n ^ i as u8).collect() };
        let (a, b, c) = (Mock::new(), Mock::new(), Mock::new());
        let first = tx(1, 2500);
        a.add_tx(&first).unwrap();
        let na = start(a.clone(), true, vec![], 0);
        let nb = start(b.clone(), true, vec![na.addr], 0);
        let c_lines = Arc::new(Mutex::new(vec![]));
        let _nc = start_with(
            c.clone(),
            Config {
                peers: vec![nb.addr],
                max_peers: 1,
                ..Config::new(*b"test")
            },
            c_lines.clone(),
        );
        wait_for("the first transaction to reach C", 10, || c.has_tx(&first));
        // A new one from B reaches A and C.
        let second = tx(2, 40);
        b.add_tx(&second).unwrap();
        nb.events
            .send(Event::NewTx(crate::transaction::Transaction::id_of(
                &second,
            )))
            .unwrap();
        wait_for("the second transaction to spread", 10, || {
            a.has_tx(&second) && c.has_tx(&second)
        });
        // An invalid one from B gets B dropped by C.
        let bad = [vec![0xff], tx(3, 100)].concat();
        b.0.lock()
            .unwrap()
            .txs
            .insert(crate::transaction::Transaction::id_of(&bad), bad.clone());
        nb.events
            .send(Event::NewTx(crate::transaction::Transaction::id_of(&bad)))
            .unwrap();
        wait_for("C to drop B", 10, || {
            has_line(
                &c_lines,
                &format!("dropping peer {}: invalid transaction", nb.addr),
            )
        });
        assert!(!c.has_tx(&bad));
    }

    #[test]
    fn nodes_find_each_other_through_a_peer() {
        // B and C know only A; they learn of each other from it.
        let log = |_: ()| Arc::new(Mutex::new(Vec::<String>::new()));
        let na = start(Mock::new(), true, vec![], 0);
        let nb = start(Mock::new(), true, vec![na.addr], 0);
        let c_log = log(());
        let _nc = start_with(
            Mock::new(),
            Config {
                peers: vec![na.addr],
                ..Config::new(*b"test")
            },
            c_log.clone(),
        );
        let connected = |lines: &Arc<Mutex<Vec<String>>>, addr: SocketAddr| {
            let needle = format!("peer {addr} connected");
            lines.lock().unwrap().iter().any(|l| l.starts_with(&needle))
        };
        wait_for("C to connect to B", 15, || connected(&c_log, nb.addr));
    }

    /// A session with a peer at `addr` that we've asked for addresses.
    fn asked(n: &mut Node<Mock>, session: u32, addr: &str) {
        let now = Instant::now();
        n.peers.insert(
            session,
            Peer {
                addr: addr.parse().unwrap(),
                capabilities: 0,
                tip: n.backend.tip(),
                outbound: true,
                listen: None,
                asked_peers: true,
                last_seen: now,
                last_ping: now,
            },
        );
    }

    #[test]
    fn a_node_with_no_one_to_dial_finds_the_network_through_its_seeds() {
        let a = Mock::new();
        for i in 0..3 {
            a.mine(1000, i + 1);
        }
        let na = start(a.clone(), false, vec![], 0);
        let seeds = vec![format!("localhost:{}", na.addr.port())];
        // A new node, alone: it looks the seed up by name and syncs from it.
        let b = Mock::new();
        let lines = Arc::new(Mutex::new(vec![]));
        let _nb = start_with(
            b.clone(),
            Config {
                seeds: seeds.clone(),
                ..Config::new(*b"test")
            },
            lines.clone(),
        );
        wait_for("sync through the seed", 20, || b.height() == 3);
        assert!(
            lines
                .lock()
                .unwrap()
                .iter()
                .any(|l| l.contains("dialing seeds localhost:"))
        );
        // One with a peer to dial leaves the seeds alone.
        let c = Mock::new();
        let lines = Arc::new(Mutex::new(vec![]));
        let _nc = start_with(
            c.clone(),
            Config {
                peers: vec![na.addr],
                seeds,
                ..Config::new(*b"test")
            },
            lines.clone(),
        );
        wait_for("sync through the peer", 20, || c.height() == 3);
        assert!(
            !lines
                .lock()
                .unwrap()
                .iter()
                .any(|l| l.contains("dialing seeds"))
        );
    }

    #[test]
    fn a_hostile_peer_cant_fill_the_address_book() {
        let mut n = Node::new(
            UdpSocket::bind("127.0.0.1:0").unwrap(),
            Config::new(*b"test"),
            Mock::new(),
            Box::new(|_| {}),
        );
        let addrs = |first: u32, count: u32| -> Vec<SocketAddr> {
            (first..first + count)
                .map(|i| SocketAddr::from(([10, (i >> 16) as u8, (i >> 8) as u8, i as u8], 7333)))
                .collect()
        };
        // Unasked: ignored. Asked: taken, once.
        asked(&mut n, 1, "10.200.0.1:1");
        n.peers.get_mut(&1).unwrap().asked_peers = false;
        n.on_peers(1, &addrs(1, 5));
        assert!(n.book.is_empty());
        n.peers.get_mut(&1).unwrap().asked_peers = true;
        n.on_peers(1, &addrs(1, MAX_ADDRS as u32));
        n.on_peers(1, &addrs(1000, 5));
        assert_eq!(n.book.len(), MAX_ADDRS);
        // One peer adds at most MAX_FROM_ONE untried addresses.
        for k in 0..10 {
            n.peers.get_mut(&1).unwrap().asked_peers = true;
            n.on_peers(1, &addrs(2000 + k * 100, MAX_ADDRS as u32));
        }
        assert_eq!(n.book.len(), MAX_FROM_ONE);
        // A full book: new addresses push out untried ones, never ones
        // that worked.
        let worked = n.book.keys().next().copied().unwrap();
        n.book.get_mut(&worked).unwrap().verified = true;
        n.book.get_mut(&worked).unwrap().source = None;
        let mut session = 100;
        while n.book.len() < MAX_BOOK {
            session += 1;
            asked(
                &mut n,
                session,
                &format!("10.201.{}.{}:1", session >> 8, session & 255),
            );
            n.on_peers(session, &addrs(100_000 + session * 100, MAX_ADDRS as u32));
        }
        asked(&mut n, 1, "10.202.0.1:1");
        let fresh = addrs(900_000, MAX_ADDRS as u32);
        n.on_peers(1, &fresh);
        assert_eq!(n.book.len(), MAX_BOOK);
        assert!(fresh.iter().all(|a| n.book.contains_key(a)));
        assert!(n.book[&worked].verified);
    }

    #[test]
    fn the_address_book_is_saved_and_loaded() {
        let dir = std::env::temp_dir().join(format!("r-peers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("peers");
        let a: SocketAddr = "10.1.2.3:7333".parse().unwrap();
        std::fs::write(&file, format!("{a}\nnonsense\n0.0.0.0:1\n")).unwrap();
        let mut n = Node::new(
            UdpSocket::bind("127.0.0.1:0").unwrap(),
            Config {
                peer_file: Some(file.clone()),
                ..Config::new(*b"test")
            },
            Mock::new(),
            Box::new(|_| {}),
        );
        assert_eq!(n.book.keys().copied().collect::<Vec<_>>(), vec![a]);
        let b: SocketAddr = "[2001:db8::1]:9".parse().unwrap();
        asked(&mut n, 1, "10.9.9.9:1");
        n.on_peers(1, &[b, a]);
        // Only addresses that have worked are saved.
        n.save_book();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), format!("{a}\n"));
        n.book.get_mut(&b).unwrap().verified = true;
        n.save_book();
        let mut want = [a.to_string(), b.to_string()];
        want.sort();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            want.join("\n") + "\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
