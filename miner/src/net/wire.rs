//! The wire format: one message per UDP datagram.
//!
//! Every datagram is at most [`MAX_PACKET`] bytes (small enough to cross
//! any path without IP fragmentation) and starts with a 16-byte header,
//! integers little-endian:
//!
//! | bytes | field   |                                                    |
//! |-------|---------|----------------------------------------------------|
//! | 4     | magic   | the first bytes of the genesis hash: one network   |
//! | 1     | version | [`VERSION`]                                        |
//! | 1     | type    | the message type                                   |
//! | 2     | flags   | reserved, zero                                     |
//! | 4     | session | from the handshake; 0 before it                    |
//! | 4     | seq     | the sender's packet counter                        |
//!
//! then the message body ([`Message`]). Hashes are 32 bytes
//! ([`hash::to_bytes`]), headers 128 ([`Header::to_bytes`]). Decoding is
//! strict: a body must be exactly as long as its fields.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use crate::crypto::mlkem::{CT_LEN, EK_LEN};
use crate::hash::{self, Hash};
use crate::header::{HEADER_BYTES, Header};
use crate::payments::Address;

/// Stays 0 until launch: during development nothing older needs to talk
/// to anything newer.
pub const VERSION: u8 = 0;
/// Largest datagram sent or accepted.
pub const MAX_PACKET: usize = 1200;
pub const HEADER_LEN: usize = 16;
/// A `HELLO` is padded to this, so the answer to an unvalidated address is
/// always smaller than what it sent.
pub const HELLO_LEN: usize = MAX_PACKET;
/// Headers in one `HEADERS` message.
pub const MAX_HEADERS: usize = (MAX_PACKET - HEADER_LEN - 1) / HEADER_BYTES;
/// Hashes in one `GET_HEADERS` locator.
pub const MAX_LOCATOR: usize = 32;
/// Ranges in one `GET_BLOCK_CHUNKS` message.
pub const MAX_RANGES: usize = 64;
pub const COOKIE_LEN: usize = 8;
/// Addresses in one `PEERS` message.
pub const MAX_ADDRS: usize = 32;
/// Ids in one `TX_INV` message.
pub const MAX_TX_IDS: usize = 32;
/// Longest `STATE` data: what's left of a datagram after its fields.
pub const MAX_STATE_DATA: usize = MAX_PACKET - HEADER_LEN - 32 - 1 - 8 - 2;
const _: () = assert!(crate::state::MAX_SYNC_REPLY <= MAX_STATE_DATA);
/// Longest preimage sent whole in a `PREIMAGE`; longer ones are sent as
/// chunks.
pub const MAX_INLINE_PREIMAGE: usize = MAX_PACKET - HEADER_LEN - 32 - 2;

/// A SHA-256 hash, as aux trees use ([`crate::preimages`]).
pub type Sha256 = [u8; 32];
/// Longest sealed slate in a `DELIVER`: what's left of a datagram.
/// Bytes in an onion cell ([`crate::onion_circuit`]).
pub const CELL_LEN: usize = MAX_PACKET - HEADER_LEN - 4;
pub const MAX_DELIVER_DATA: usize = MAX_PACKET - HEADER_LEN - 32 - 4 - CT_LEN - 2;
const _: () = assert!(MAX_DELIVER_DATA >= 160);

/// Capability bits in `HELLO` and `WELCOME`.
pub mod cap {
    /// Accepts connections on its address (the `HELLO`'s source IP and
    /// listen port), so it may be passed on to other nodes in `PEERS`.
    pub const LISTENING: u16 = 1;
    /// Serves the preimages of hashes that blocks' aux roots lead to
    /// (`GET_PREIMAGE`), and fetches those of recent blocks
    /// ([`crate::preimages`]).
    pub const TREE: u16 = 2;
    /// Relays onion-routed traffic ([`crate::onion`]).
    pub const ONION: u16 = 4;
}

/// A node's best chain, as it advertises it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tip {
    pub height: u64,
    pub hash: Hash,
    /// Total work through the tip ([`crate::header::work`] units).
    pub work: u128,
    /// The lowest height whose state it serves for fast sync (its history
    /// window, or its own fast-sync base if that's higher); 0 if none.
    pub state_from: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// Opens a handshake (padded to [`HELLO_LEN`]); sent again with the
    /// cookie from the answering [`Message::Cookie`].
    Hello {
        capabilities: u16,
        node_id: u64,
        listen_port: u16,
        tip: Tip,
        cookie: Option<[u8; COOKIE_LEN]>,
    },
    /// Proof the sender's address is real: echo it in the next `Hello`.
    Cookie {
        cookie: [u8; COOKIE_LEN],
    },
    /// Completes a handshake; the packet's session field is the session.
    Welcome {
        capabilities: u16,
        node_id: u64,
        tip: Tip,
    },
    Ping {
        nonce: u64,
    },
    Pong {
        nonce: u64,
    },
    /// The sender's tip, sent when it changes.
    Status {
        tip: Tip,
    },
    /// Up to `count` main-chain headers after the newest `locator` hash on
    /// the receiver's main chain (after genesis if none is): the locator
    /// lists the sender's chain from its tip back, at growing distances.
    GetHeaders {
        locator: Vec<Hash>,
        count: u16,
    },
    /// Consecutive headers (possibly none).
    Headers {
        headers: Vec<Header>,
    },
    /// A new block, announced by its header alone.
    NewHeader {
        header: Header,
    },
    /// The body of the block with this header hash.
    GetBlock {
        hash: Hash,
    },
    /// The body's size and chunk root ([`crate::chunk`]): the receiver
    /// checks them against the header's body hash.
    BlockInfo {
        hash: Hash,
        size: u32,
        root: Hash,
    },
    /// Wire chunks of a body, as `(first, count)` ranges.
    GetBlockChunks {
        hash: Hash,
        ranges: Vec<(u32, u16)>,
    },
    BlockChunk {
        hash: Hash,
        index: u32,
        data: Vec<u8>,
    },
    /// The sender doesn't have this object.
    NotFound {
        hash: Hash,
    },
    /// Addresses of nodes that accept connections.
    GetPeers,
    /// Up to [`MAX_ADDRS`] addresses.
    Peers {
        addrs: Vec<SocketAddr>,
    },
    /// Whether the sender can serve the state after the main-chain block
    /// at `height`, for fast sync.
    GetSnapshot {
        height: u64,
    },
    /// It can: that block (the base), its parent and the parent's total
    /// work. The state is fetched with `GET_STATE`.
    SnapshotInfo {
        parent: Header,
        base: Header,
        parent_work: u128,
    },
    /// The sender can't serve the state at `height`.
    NoSnapshot {
        height: u64,
    },
    /// Node (`level`, `index`) of the state tree after block `base`.
    GetState {
        base: Hash,
        level: u8,
        index: u64,
    },
    /// The node as its descendants a few levels down, or its leaves
    /// ([`crate::state::StateTree::sync_reply`]).
    State {
        base: Hash,
        level: u8,
        index: u64,
        data: Vec<u8>,
    },
    /// The sender can't serve the state after block `base`.
    NoState {
        base: Hash,
    },
    /// Ids of transactions the sender has in its mempool (up to
    /// [`MAX_TX_IDS`]).
    TxInv {
        ids: Vec<Hash>,
    },
    /// The transaction with this id.
    GetTx {
        id: Hash,
    },
    /// Its encoding's size and chunk root, which hash to the id; its chunks
    /// are fetched with `GET_BLOCK_CHUNKS` under the id.
    TxInfo {
        id: Hash,
        size: u32,
        root: Hash,
    },
    /// The bytes whose SHA-256 is `hash`.
    GetPreimage {
        hash: Sha256,
    },
    /// They are `data` (up to [`MAX_INLINE_PREIMAGE`] bytes).
    Preimage {
        hash: Sha256,
        data: Vec<u8>,
    },
    /// They are `size` bytes with this chunk root: fetch their chunks with
    /// `GET_BLOCK_CHUNKS` under [`preimage_id`]`(size, root)`.
    PreimageInfo {
        hash: Sha256,
        size: u32,
        root: Hash,
    },
    /// The sender doesn't have them.
    NoPreimage {
        hash: Sha256,
    },
    /// Init mode ([`crate::payments`]): register (or keep registered) the
    /// payment address of this key with the receiving host.
    Register {
        ek: Box<[u8; EK_LEN]>,
    },
    /// Prove you hold the key: a secret sealed to it.
    Challenge {
        address: Address,
        ct: Box<[u8; CT_LEN]>,
    },
    /// The proof ([`crate::payments::register_proof`]).
    Prove {
        address: Address,
        proof: [u8; 32],
    },
    /// The address is registered with the sender for a while: refresh it.
    Registered {
        address: Address,
    },
    /// The key of a payment address registered with the receiver.
    Lookup {
        address: Address,
    },
    AddressKey {
        address: Address,
        ek: Box<[u8; EK_LEN]>,
    },
    /// No one with this address is registered with the sender (or it went
    /// quiet): it's offline.
    Offline {
        address: Address,
    },
    /// A sealed slate for `to`: to its host, which passes it on (or
    /// answers `OFFLINE`); then to the recipient. `post` is the sender's
    /// number for it.
    Deliver {
        to: Address,
        post: u32,
        ct: Box<[u8; CT_LEN]>,
        data: Vec<u8>,
    },
    /// The host passed delivery `post` on.
    Forwarded {
        to: Address,
        post: u32,
    },
    /// Onion mode ([`crate::onion_circuit`]): open circuit `circ` on this
    /// link to the receiving relay: a challenge and a fresh key.
    Create {
        circ: u32,
        challenge: [u8; 32],
        ek: Box<[u8; EK_LEN]>,
    },
    /// Piece `index` of the relay's answer.
    Created {
        circ: u32,
        index: u8,
        data: Vec<u8>,
    },
    /// A cell going out on circuit `circ` (away from its client).
    Forward {
        circ: u32,
        cell: Box<[u8; CELL_LEN]>,
    },
    /// A cell coming back on circuit `circ` (toward its client).
    Backward {
        circ: u32,
        cell: Box<[u8; CELL_LEN]>,
    },
}

/// What a preimage's chunks are requested under.
pub fn preimage_id(size: u32, root: &Hash) -> Hash {
    crate::chunk::object_hash(hash::domain::PREIMAGE, size as usize, root)
}

impl Message {
    fn kind(&self) -> u8 {
        match self {
            Self::Hello { .. } => 1,
            Self::Cookie { .. } => 2,
            Self::Welcome { .. } => 3,
            Self::Ping { .. } => 4,
            Self::Pong { .. } => 5,
            Self::Status { .. } => 6,
            Self::GetHeaders { .. } => 7,
            Self::Headers { .. } => 8,
            Self::NewHeader { .. } => 9,
            Self::GetBlock { .. } => 10,
            Self::BlockInfo { .. } => 11,
            Self::GetBlockChunks { .. } => 12,
            Self::BlockChunk { .. } => 13,
            Self::NotFound { .. } => 14,
            Self::GetPeers => 15,
            Self::Peers { .. } => 16,
            Self::GetSnapshot { .. } => 17,
            Self::SnapshotInfo { .. } => 18,
            Self::NoSnapshot { .. } => 19,
            Self::GetState { .. } => 20,
            Self::State { .. } => 21,
            Self::NoState { .. } => 22,
            Self::TxInv { .. } => 23,
            Self::GetTx { .. } => 24,
            Self::TxInfo { .. } => 25,
            Self::GetPreimage { .. } => 26,
            Self::Preimage { .. } => 27,
            Self::PreimageInfo { .. } => 28,
            Self::NoPreimage { .. } => 29,
            Self::Register { .. } => 30,
            Self::Challenge { .. } => 31,
            Self::Prove { .. } => 32,
            Self::Registered { .. } => 33,
            Self::Lookup { .. } => 34,
            Self::AddressKey { .. } => 35,
            Self::Offline { .. } => 36,
            Self::Deliver { .. } => 37,
            Self::Forwarded { .. } => 38,
            Self::Create { .. } => 39,
            Self::Created { .. } => 40,
            Self::Forward { .. } => 41,
            Self::Backward { .. } => 42,
        }
    }

    /// The message's name, for logs.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "HELLO",
            Self::Cookie { .. } => "COOKIE",
            Self::Welcome { .. } => "WELCOME",
            Self::Ping { .. } => "PING",
            Self::Pong { .. } => "PONG",
            Self::Status { .. } => "STATUS",
            Self::GetHeaders { .. } => "GET_HEADERS",
            Self::Headers { .. } => "HEADERS",
            Self::NewHeader { .. } => "HEADER",
            Self::GetBlock { .. } => "GET_BLOCK",
            Self::BlockInfo { .. } => "BLOCK_INFO",
            Self::GetBlockChunks { .. } => "GET_BLOCK_CHUNKS",
            Self::BlockChunk { .. } => "BLOCK_CHUNK",
            Self::NotFound { .. } => "NOT_FOUND",
            Self::GetPeers => "GET_PEERS",
            Self::Peers { .. } => "PEERS",
            Self::GetSnapshot { .. } => "GET_SNAPSHOT",
            Self::SnapshotInfo { .. } => "SNAPSHOT_INFO",
            Self::NoSnapshot { .. } => "NO_SNAPSHOT",
            Self::GetState { .. } => "GET_STATE",
            Self::State { .. } => "STATE",
            Self::NoState { .. } => "NO_STATE",
            Self::TxInv { .. } => "TX_INV",
            Self::GetTx { .. } => "GET_TX",
            Self::TxInfo { .. } => "TX_INFO",
            Self::GetPreimage { .. } => "GET_PREIMAGE",
            Self::Preimage { .. } => "PREIMAGE",
            Self::PreimageInfo { .. } => "PREIMAGE_INFO",
            Self::NoPreimage { .. } => "NO_PREIMAGE",
            Self::Register { .. } => "REGISTER",
            Self::Challenge { .. } => "CHALLENGE",
            Self::Prove { .. } => "PROVE",
            Self::Registered { .. } => "REGISTERED",
            Self::Lookup { .. } => "LOOKUP",
            Self::AddressKey { .. } => "ADDRESS_KEY",
            Self::Offline { .. } => "OFFLINE",
            Self::Deliver { .. } => "DELIVER",
            Self::Forwarded { .. } => "FORWARDED",
            Self::Create { .. } => "CREATE",
            Self::Created { .. } => "CREATED",
            Self::Forward { .. } => "FORWARD",
            Self::Backward { .. } => "BACKWARD",
        }
    }
}

/// A datagram's header fields other than magic, version and type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packet {
    pub session: u32,
    pub seq: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WireError {
    /// Another network's magic.
    Magic,
    Version(u8),
    UnknownType(u8),
    /// Too long, too short, or a field doesn't parse.
    Malformed,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Magic => f.write_str("another network's packet"),
            Self::Version(v) => write!(f, "unsupported version {v}"),
            Self::UnknownType(t) => write!(f, "unknown message type {t}"),
            Self::Malformed => f.write_str("malformed packet"),
        }
    }
}

impl std::error::Error for WireError {}

fn put_tip(out: &mut Vec<u8>, tip: &Tip) {
    out.extend(tip.height.to_le_bytes());
    out.extend(hash::to_bytes(&tip.hash));
    out.extend(tip.work.to_le_bytes());
    out.extend(tip.state_from.to_le_bytes());
}

fn put_hash(out: &mut Vec<u8>, h: &Hash) {
    out.extend(hash::to_bytes(h));
}

/// Encodes `msg` as one datagram.
///
/// # Panics
/// If it doesn't fit [`MAX_PACKET`] (more than [`MAX_HEADERS`] headers,
/// [`MAX_RANGES`] ranges or a chunk over [`crate::chunk::CHUNK_BYTES`]).
pub fn encode(magic: [u8; 4], packet: Packet, msg: &Message) -> Vec<u8> {
    let mut out = Vec::with_capacity(MAX_PACKET);
    out.extend(magic);
    out.push(VERSION);
    out.push(msg.kind());
    out.extend(0u16.to_le_bytes());
    out.extend(packet.session.to_le_bytes());
    out.extend(packet.seq.to_le_bytes());
    match msg {
        Message::Hello {
            capabilities,
            node_id,
            listen_port,
            tip,
            cookie,
        } => {
            out.extend(capabilities.to_le_bytes());
            out.extend(node_id.to_le_bytes());
            out.extend(listen_port.to_le_bytes());
            put_tip(&mut out, tip);
            out.push(cookie.is_some() as u8);
            out.extend(cookie.unwrap_or_default());
            out.resize(HELLO_LEN, 0);
        }
        Message::Cookie { cookie } => out.extend(cookie),
        Message::Welcome {
            capabilities,
            node_id,
            tip,
        } => {
            out.extend(capabilities.to_le_bytes());
            out.extend(node_id.to_le_bytes());
            put_tip(&mut out, tip);
        }
        Message::Ping { nonce } | Message::Pong { nonce } => out.extend(nonce.to_le_bytes()),
        Message::Status { tip } => put_tip(&mut out, tip),
        Message::GetHeaders { locator, count } => {
            assert!(locator.len() <= MAX_LOCATOR, "locator too long");
            out.push(locator.len() as u8);
            for h in locator {
                put_hash(&mut out, h);
            }
            out.extend(count.to_le_bytes());
        }
        Message::Headers { headers } => {
            assert!(headers.len() <= MAX_HEADERS, "too many headers");
            out.push(headers.len() as u8);
            for h in headers {
                out.extend(h.to_bytes());
            }
        }
        Message::NewHeader { header } => out.extend(header.to_bytes()),
        Message::GetBlock { hash } | Message::NotFound { hash } => put_hash(&mut out, hash),
        Message::BlockInfo { hash, size, root } => {
            put_hash(&mut out, hash);
            out.extend(size.to_le_bytes());
            put_hash(&mut out, root);
        }
        Message::GetBlockChunks { hash, ranges } => {
            assert!(ranges.len() <= MAX_RANGES, "too many ranges");
            put_hash(&mut out, hash);
            out.push(ranges.len() as u8);
            for (first, count) in ranges {
                out.extend(first.to_le_bytes());
                out.extend(count.to_le_bytes());
            }
        }
        Message::BlockChunk { hash, index, data } => {
            assert!(data.len() <= crate::chunk::CHUNK_BYTES, "chunk too long");
            put_hash(&mut out, hash);
            out.extend(index.to_le_bytes());
            out.extend((data.len() as u16).to_le_bytes());
            out.extend(data);
        }
        Message::GetPeers => {}
        Message::Peers { addrs } => {
            assert!(addrs.len() <= MAX_ADDRS, "too many addresses");
            out.push(addrs.len() as u8);
            for a in addrs {
                match a.ip() {
                    IpAddr::V4(ip) => {
                        out.push(4);
                        out.extend(ip.octets());
                    }
                    IpAddr::V6(ip) => {
                        out.push(6);
                        out.extend(ip.octets());
                    }
                }
                out.extend(a.port().to_le_bytes());
            }
        }
        Message::GetSnapshot { height } | Message::NoSnapshot { height } => {
            out.extend(height.to_le_bytes())
        }
        Message::SnapshotInfo {
            parent,
            base,
            parent_work,
        } => {
            out.extend(parent.to_bytes());
            out.extend(base.to_bytes());
            out.extend(parent_work.to_le_bytes());
        }
        Message::GetState { base, level, index } => {
            put_hash(&mut out, base);
            out.push(*level);
            out.extend(index.to_le_bytes());
        }
        Message::State {
            base,
            level,
            index,
            data,
        } => {
            assert!(data.len() <= MAX_STATE_DATA, "state data too long");
            put_hash(&mut out, base);
            out.push(*level);
            out.extend(index.to_le_bytes());
            out.extend((data.len() as u16).to_le_bytes());
            out.extend(data);
        }
        Message::NoState { base } => put_hash(&mut out, base),
        Message::TxInv { ids } => {
            assert!(ids.len() <= MAX_TX_IDS, "too many ids");
            out.push(ids.len() as u8);
            for id in ids {
                put_hash(&mut out, id);
            }
        }
        Message::GetTx { id } => put_hash(&mut out, id),
        Message::TxInfo { id, size, root } => {
            put_hash(&mut out, id);
            out.extend(size.to_le_bytes());
            put_hash(&mut out, root);
        }
        Message::GetPreimage { hash } | Message::NoPreimage { hash } => out.extend(hash),
        Message::Preimage { hash, data } => {
            assert!(data.len() <= MAX_INLINE_PREIMAGE, "preimage too long");
            out.extend(hash);
            out.extend((data.len() as u16).to_le_bytes());
            out.extend(data);
        }
        Message::PreimageInfo { hash, size, root } => {
            out.extend(hash);
            out.extend(size.to_le_bytes());
            put_hash(&mut out, root);
        }
        Message::Register { ek } => out.extend(&ek[..]),
        Message::Challenge { address, ct } => {
            out.extend(address.0);
            out.extend(&ct[..]);
        }
        Message::Prove { address, proof } => {
            out.extend(address.0);
            out.extend(proof);
        }
        Message::Registered { address }
        | Message::Lookup { address }
        | Message::Offline { address } => out.extend(address.0),
        Message::AddressKey { address, ek } => {
            out.extend(address.0);
            out.extend(&ek[..]);
        }
        Message::Deliver { to, post, ct, data } => {
            assert!(data.len() <= MAX_DELIVER_DATA, "sealed slate too long");
            out.extend(to.0);
            out.extend(post.to_le_bytes());
            out.extend(&ct[..]);
            out.extend((data.len() as u16).to_le_bytes());
            out.extend(data);
        }
        Message::Forwarded { to, post } => {
            out.extend(to.0);
            out.extend(post.to_le_bytes());
        }
        Message::Create {
            circ,
            challenge,
            ek,
        } => {
            out.extend(circ.to_le_bytes());
            out.extend(challenge);
            out.extend(&ek[..]);
        }
        Message::Created { circ, index, data } => {
            assert!(
                data.len() <= crate::onion_circuit::FRAGMENT,
                "piece too long"
            );
            out.extend(circ.to_le_bytes());
            out.push(*index);
            out.extend((data.len() as u16).to_le_bytes());
            out.extend(data);
        }
        Message::Forward { circ, cell } | Message::Backward { circ, cell } => {
            out.extend(circ.to_le_bytes());
            out.extend(&cell[..]);
        }
    }
    assert!(
        out.len() <= MAX_PACKET,
        "{} doesn't fit a datagram",
        msg.name()
    );
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let (a, b) = self.0.split_at_checked(n).ok_or(WireError::Malformed)?;
        self.0 = b;
        Ok(a)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        Ok(self.take(N)?.try_into().expect("length"))
    }

    fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, WireError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn hash(&mut self) -> Result<Hash, WireError> {
        hash::from_bytes(&self.array()?).ok_or(WireError::Malformed)
    }

    fn header(&mut self) -> Result<Header, WireError> {
        Header::from_bytes(self.take(HEADER_BYTES)?).ok_or(WireError::Malformed)
    }

    fn tip(&mut self) -> Result<Tip, WireError> {
        Ok(Tip {
            height: self.u64()?,
            hash: self.hash()?,
            work: u128::from_le_bytes(self.array()?),
            state_from: self.u64()?,
        })
    }

    fn end(&self) -> Result<(), WireError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(WireError::Malformed)
        }
    }
}

/// Decodes a datagram from network `magic`.
pub fn decode(magic: [u8; 4], bytes: &[u8]) -> Result<(Packet, Message), WireError> {
    if bytes.len() > MAX_PACKET {
        return Err(WireError::Malformed);
    }
    let mut r = Reader(bytes);
    if r.array::<4>()? != magic {
        return Err(WireError::Magic);
    }
    let version = r.u8()?;
    if version != VERSION {
        return Err(WireError::Version(version));
    }
    let kind = r.u8()?;
    if r.u16()? != 0 {
        return Err(WireError::Malformed);
    }
    let packet = Packet {
        session: r.u32()?,
        seq: r.u32()?,
    };
    let msg = match kind {
        1 => {
            if bytes.len() != HELLO_LEN {
                return Err(WireError::Malformed);
            }
            let capabilities = r.u16()?;
            let node_id = r.u64()?;
            let listen_port = r.u16()?;
            let tip = r.tip()?;
            let has_cookie = r.u8()?;
            let cookie: [u8; COOKIE_LEN] = r.array()?;
            if has_cookie > 1 || (has_cookie == 0 && cookie != [0; COOKIE_LEN]) {
                return Err(WireError::Malformed);
            }
            // The padding must be zeros.
            if r.0.iter().any(|&b| b != 0) {
                return Err(WireError::Malformed);
            }
            r.0 = &[];
            Message::Hello {
                capabilities,
                node_id,
                listen_port,
                tip,
                cookie: (has_cookie == 1).then_some(cookie),
            }
        }
        2 => Message::Cookie { cookie: r.array()? },
        3 => Message::Welcome {
            capabilities: r.u16()?,
            node_id: r.u64()?,
            tip: r.tip()?,
        },
        4 => Message::Ping { nonce: r.u64()? },
        5 => Message::Pong { nonce: r.u64()? },
        6 => Message::Status { tip: r.tip()? },
        7 => {
            let n = r.u8()? as usize;
            if n > MAX_LOCATOR {
                return Err(WireError::Malformed);
            }
            let locator = (0..n).map(|_| r.hash()).collect::<Result<_, _>>()?;
            Message::GetHeaders {
                locator,
                count: r.u16()?,
            }
        }
        8 => {
            let n = r.u8()? as usize;
            if n > MAX_HEADERS {
                return Err(WireError::Malformed);
            }
            let headers = (0..n).map(|_| r.header()).collect::<Result<_, _>>()?;
            Message::Headers { headers }
        }
        9 => Message::NewHeader {
            header: r.header()?,
        },
        10 => Message::GetBlock { hash: r.hash()? },
        11 => Message::BlockInfo {
            hash: r.hash()?,
            size: r.u32()?,
            root: r.hash()?,
        },
        12 => {
            let hash = r.hash()?;
            let n = r.u8()? as usize;
            if n > MAX_RANGES {
                return Err(WireError::Malformed);
            }
            let ranges = (0..n)
                .map(|_| Ok((r.u32()?, r.u16()?)))
                .collect::<Result<_, _>>()?;
            Message::GetBlockChunks { hash, ranges }
        }
        13 => {
            let hash = r.hash()?;
            let index = r.u32()?;
            let len = r.u16()? as usize;
            if len > crate::chunk::CHUNK_BYTES {
                return Err(WireError::Malformed);
            }
            Message::BlockChunk {
                hash,
                index,
                data: r.take(len)?.to_vec(),
            }
        }
        14 => Message::NotFound { hash: r.hash()? },
        15 => Message::GetPeers,
        16 => {
            let n = r.u8()? as usize;
            if n > MAX_ADDRS {
                return Err(WireError::Malformed);
            }
            let addrs = (0..n)
                .map(|_| {
                    let ip = match r.u8()? {
                        4 => IpAddr::V4(Ipv4Addr::from(r.array::<4>()?)),
                        6 => IpAddr::V6(Ipv6Addr::from(r.array::<16>()?)),
                        _ => return Err(WireError::Malformed),
                    };
                    Ok(SocketAddr::new(ip, r.u16()?))
                })
                .collect::<Result<_, _>>()?;
            Message::Peers { addrs }
        }
        17 => Message::GetSnapshot { height: r.u64()? },
        18 => Message::SnapshotInfo {
            parent: r.header()?,
            base: r.header()?,
            parent_work: u128::from_le_bytes(r.array()?),
        },
        19 => Message::NoSnapshot { height: r.u64()? },
        20 => Message::GetState {
            base: r.hash()?,
            level: r.u8()?,
            index: r.u64()?,
        },
        21 => {
            let base = r.hash()?;
            let level = r.u8()?;
            let index = r.u64()?;
            let len = r.u16()? as usize;
            if len > MAX_STATE_DATA {
                return Err(WireError::Malformed);
            }
            Message::State {
                base,
                level,
                index,
                data: r.take(len)?.to_vec(),
            }
        }
        22 => Message::NoState { base: r.hash()? },
        23 => {
            let n = r.u8()? as usize;
            if n > MAX_TX_IDS {
                return Err(WireError::Malformed);
            }
            let ids = (0..n).map(|_| r.hash()).collect::<Result<_, _>>()?;
            Message::TxInv { ids }
        }
        24 => Message::GetTx { id: r.hash()? },
        25 => Message::TxInfo {
            id: r.hash()?,
            size: r.u32()?,
            root: r.hash()?,
        },
        26 => Message::GetPreimage { hash: r.array()? },
        27 => {
            let hash = r.array()?;
            let len = r.u16()? as usize;
            if len > MAX_INLINE_PREIMAGE {
                return Err(WireError::Malformed);
            }
            Message::Preimage {
                hash,
                data: r.take(len)?.to_vec(),
            }
        }
        28 => Message::PreimageInfo {
            hash: r.array()?,
            size: r.u32()?,
            root: r.hash()?,
        },
        29 => Message::NoPreimage { hash: r.array()? },
        30 => Message::Register {
            ek: Box::new(r.array()?),
        },
        31 => Message::Challenge {
            address: Address(r.array()?),
            ct: Box::new(r.array()?),
        },
        32 => Message::Prove {
            address: Address(r.array()?),
            proof: r.array()?,
        },
        33 => Message::Registered {
            address: Address(r.array()?),
        },
        34 => Message::Lookup {
            address: Address(r.array()?),
        },
        35 => Message::AddressKey {
            address: Address(r.array()?),
            ek: Box::new(r.array()?),
        },
        36 => Message::Offline {
            address: Address(r.array()?),
        },
        37 => {
            let to = Address(r.array()?);
            let post = r.u32()?;
            let ct = Box::new(r.array()?);
            let len = r.u16()? as usize;
            if len > MAX_DELIVER_DATA {
                return Err(WireError::Malformed);
            }
            Message::Deliver {
                to,
                post,
                ct,
                data: r.take(len)?.to_vec(),
            }
        }
        38 => Message::Forwarded {
            to: Address(r.array()?),
            post: r.u32()?,
        },
        39 => Message::Create {
            circ: r.u32()?,
            challenge: r.array()?,
            ek: Box::new(r.array()?),
        },
        40 => {
            let circ = r.u32()?;
            let index = r.u8()?;
            let len = r.u16()? as usize;
            if len > crate::onion_circuit::FRAGMENT {
                return Err(WireError::Malformed);
            }
            Message::Created {
                circ,
                index,
                data: r.take(len)?.to_vec(),
            }
        }
        41 => Message::Forward {
            circ: r.u32()?,
            cell: Box::new(r.array()?),
        },
        42 => Message::Backward {
            circ: r.u32()?,
            cell: Box::new(r.array()?),
        },
        t => return Err(WireError::UnknownType(t)),
    };
    r.end()?;
    Ok((packet, msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::BabyBear;

    const MAGIC: [u8; 4] = *b"test";

    fn h(n: u32) -> Hash {
        BabyBear::new_array(core::array::from_fn(|i| n * 10 + i as u32))
    }

    fn header(n: u64) -> Header {
        Header {
            height: n,
            prev: h(n as u32),
            timestamp: 1000 + n,
            nonce: n * 3,
            ..Header::default()
        }
    }

    fn tip() -> Tip {
        Tip {
            height: 5,
            hash: h(5),
            work: u128::MAX - 3,
            state_from: 2,
        }
    }

    #[test]
    fn every_message_round_trips_and_fits() {
        let msgs = vec![
            Message::Hello {
                capabilities: 0xbeef,
                node_id: 42,
                listen_port: 7333,
                tip: tip(),
                cookie: None,
            },
            Message::Hello {
                capabilities: 1,
                node_id: 42,
                listen_port: 0,
                tip: tip(),
                cookie: Some(*b"cookie!!"),
            },
            Message::Cookie { cookie: [9; 8] },
            Message::Welcome {
                capabilities: 3,
                node_id: 7,
                tip: tip(),
            },
            Message::Ping { nonce: 1 },
            Message::Pong { nonce: u64::MAX },
            Message::Status { tip: tip() },
            Message::GetHeaders {
                locator: vec![],
                count: 9,
            },
            Message::GetHeaders {
                locator: (0..MAX_LOCATOR as u32).map(h).collect(),
                count: 9,
            },
            Message::Headers { headers: vec![] },
            Message::Headers {
                headers: (1..=MAX_HEADERS as u64).map(header).collect(),
            },
            Message::NewHeader { header: header(3) },
            Message::GetBlock { hash: h(1) },
            Message::BlockInfo {
                hash: h(1),
                size: 163_000,
                root: h(2),
            },
            Message::GetBlockChunks {
                hash: h(1),
                ranges: (0..MAX_RANGES as u32).map(|i| (i * 100, 7)).collect(),
            },
            Message::BlockChunk {
                hash: h(1),
                index: 77,
                data: vec![5; crate::chunk::CHUNK_BYTES],
            },
            Message::NotFound { hash: h(3) },
            Message::GetPeers,
            Message::Peers { addrs: vec![] },
            Message::Peers {
                addrs: (0..MAX_ADDRS as u16)
                    .map(|i| {
                        if i % 2 == 0 {
                            SocketAddr::from(([10, 0, 0, i as u8], 7000 + i))
                        } else {
                            SocketAddr::from(([0x2001, 0xdb8, 0, 0, 0, 0, 0, i], 7000 + i))
                        }
                    })
                    .collect(),
            },
            Message::GetSnapshot { height: 990 },
            Message::SnapshotInfo {
                parent: header(989),
                base: header(990),
                parent_work: u128::MAX / 3,
            },
            Message::NoSnapshot { height: 7 },
            Message::GetState {
                base: h(4),
                level: 40,
                index: u64::MAX >> 24,
            },
            Message::State {
                base: h(4),
                level: 3,
                index: 9,
                data: vec![],
            },
            Message::State {
                base: h(4),
                level: 35,
                index: 1,
                data: vec![7; MAX_STATE_DATA],
            },
            Message::NoState { base: h(4) },
            Message::TxInv { ids: vec![] },
            Message::TxInv {
                ids: (0..MAX_TX_IDS as u32).map(h).collect(),
            },
            Message::GetTx { id: h(6) },
            Message::TxInfo {
                id: h(6),
                size: 2400,
                root: h(7),
            },
            Message::GetPreimage { hash: [3; 32] },
            Message::Preimage {
                hash: [4; 32],
                data: vec![],
            },
            Message::Preimage {
                hash: [4; 32],
                data: vec![8; MAX_INLINE_PREIMAGE],
            },
            Message::PreimageInfo {
                hash: [5; 32],
                size: 1 << 20,
                root: h(8),
            },
            Message::NoPreimage { hash: [6; 32] },
            Message::Register {
                ek: Box::new([7; EK_LEN]),
            },
            Message::Challenge {
                address: Address([8; 32]),
                ct: Box::new([9; CT_LEN]),
            },
            Message::Prove {
                address: Address([8; 32]),
                proof: [10; 32],
            },
            Message::Registered {
                address: Address([8; 32]),
            },
            Message::Lookup {
                address: Address([11; 32]),
            },
            Message::AddressKey {
                address: Address([11; 32]),
                ek: Box::new([12; EK_LEN]),
            },
            Message::Offline {
                address: Address([11; 32]),
            },
            Message::Deliver {
                to: Address([13; 32]),
                post: 77,
                ct: Box::new([14; CT_LEN]),
                data: vec![15; MAX_DELIVER_DATA],
            },
            Message::Forwarded {
                to: Address([13; 32]),
                post: 77,
            },
            Message::Create {
                circ: 5,
                challenge: [16; 32],
                ek: Box::new([17; EK_LEN]),
            },
            Message::Created {
                circ: 5,
                index: 4,
                data: vec![18; crate::onion_circuit::FRAGMENT],
            },
            Message::Forward {
                circ: u32::MAX,
                cell: Box::new([19; CELL_LEN]),
            },
            Message::Backward {
                circ: 6,
                cell: Box::new([20; CELL_LEN]),
            },
        ];
        let packet = Packet {
            session: 0xdead,
            seq: 12,
        };
        for m in msgs {
            let bytes = encode(MAGIC, packet, &m);
            assert!(bytes.len() <= MAX_PACKET);
            assert_eq!(
                decode(MAGIC, &bytes),
                Ok((packet, m.clone())),
                "{}",
                m.name()
            );
            // Trailing or missing bytes are refused.
            let mut longer = bytes.clone();
            longer.push(0);
            assert!(decode(MAGIC, &longer).is_err(), "{}", m.name());
            assert!(
                decode(MAGIC, &bytes[..bytes.len() - 1]).is_err(),
                "{}",
                m.name()
            );
        }
        assert_eq!(MAX_HEADERS, 7);
    }

    #[test]
    fn foreign_and_bad_packets_are_refused() {
        let packet = Packet { session: 0, seq: 0 };
        let bytes = encode(MAGIC, packet, &Message::Ping { nonce: 1 });
        assert_eq!(decode(*b"main", &bytes), Err(WireError::Magic));
        let mut v = bytes.clone();
        v[4] = 2;
        assert_eq!(decode(MAGIC, &v), Err(WireError::Version(2)));
        let mut v = bytes.clone();
        v[5] = 99;
        assert_eq!(decode(MAGIC, &v), Err(WireError::UnknownType(99)));
        let mut v = bytes;
        v[6] = 1;
        assert_eq!(decode(MAGIC, &v), Err(WireError::Malformed));
        // A non-canonical hash.
        let mut v = encode(MAGIC, packet, &Message::GetBlock { hash: h(1) });
        v[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(MAGIC, &v), Err(WireError::Malformed));
        // A HELLO with junk in its padding.
        let mut v = encode(
            MAGIC,
            packet,
            &Message::Hello {
                capabilities: 0,
                node_id: 1,
                listen_port: 1,
                tip: tip(),
                cookie: None,
            },
        );
        *v.last_mut().unwrap() = 1;
        assert_eq!(decode(MAGIC, &v), Err(WireError::Malformed));
    }
}
