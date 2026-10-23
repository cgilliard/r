//! Onion circuits on the node ([`crate::onion_circuit`] has their
//! cryptography): relaying them (with [`Config::relay`], advertising
//! [`cap::ONION`]), and building our own.
//!
//! **Links.** A circuit crosses existing peer sessions. Each link has its
//! own circuit id, chosen by the side that opened it: `CREATE` and
//! `FORWARD` go from the opener, `CREATED` and `BACKWARD` come back, so a
//! node tells the two directions apart by message type.
//!
//! **Relaying.** A `CREATE` on a new link id gets the handshake answer
//! (`CREATED`, in pieces; sent again if the `CREATE` comes again). A
//! `FORWARD` is opened with the hop's key: `RELAY` passes it to the next
//! link, `EXTEND` opens the next link (dialing the relay if need be) and
//! turns its `CREATED` pieces into `EXTENDED` messages back, and `DATA` is a
//! post message for this hop as a rendezvous point ([`super::post`]). A
//! `BACKWARD` from the next link gets this hop's layer and goes back. A
//! circuit idle for [`IDLE`] (or whose link is gone) is dropped.
//!
//! **Building.** We pick the hops, open the first link, and extend hop by
//! hop through the circuit so far, checking each relay's answer against its
//! key hash in the consensus. A step not answered in [`STEP_TIMEOUT`] is
//! asked again, [`STEP_TRIES`] times; then the circuit fails, and so does
//! what it was for (which tries elsewhere).

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::{Duration, Instant};

use super::{Backend, Dial, Message, Node, Packet, wire};
use crate::crypto::{mldsa, mlkem};
use crate::onion::Relay;
use crate::onion_circuit::{self as oc, Keys, back, cmd};

/// A relay circuit idle this long is dropped.
pub const IDLE: Duration = Duration::from_secs(180);
/// Ask again for a handshake step not answered within this.
const STEP_TIMEOUT: Duration = Duration::from_secs(3);
const STEP_TRIES: u32 = 3;
/// Give up opening the next link if its relay doesn't connect in this.
const EXTEND_TIMEOUT: Duration = Duration::from_secs(10);

/// A relay's identity keys ([`crate::onion::identity`]).
#[derive(Clone)]
pub struct RelayKeys {
    pub pk: Box<[u8; mldsa::PK_LEN]>,
    pub sk: Box<[u8; mldsa::SK_LEN]>,
}

impl std::fmt::Debug for RelayKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RelayKeys({})",
            crate::wallet::hex(&crate::onion::key_hash(&self.pk))
        )
    }
}

/// Where a post message goes, or came from ([`super::post`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Via {
    /// A peer session (init mode).
    Session(u32),
    /// Our circuit's far end.
    Out(u32),
    /// The client of a circuit we're the end of.
    In(u32),
}

/// What a circuit of ours is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// To register our address with this rendezvous point.
    Register(Relay),
    /// To deliver post `n`.
    Send(u32),
}

struct Extending {
    addr: SocketAddr,
    challenge: [u8; 32],
    ek: Box<[u8; mlkem::EK_LEN]>,
    since: Instant,
}

struct Relayed {
    /// The link toward the client: session and its circuit id.
    prev: (u32, u32),
    keys: Keys,
    back_n: u64,
    /// Our handshake answer, to send again.
    created: Vec<u8>,
    next: Option<(u32, u32)>,
    extending: Option<Extending>,
    last: Instant,
}

struct Pending {
    challenge: [u8; 32],
    ek: Box<[u8; mlkem::EK_LEN]>,
    dk: Box<[u8; mlkem::DK_LEN]>,
    pieces: Vec<Option<Vec<u8>>>,
    asked: Option<Instant>,
    tries: u32,
}

struct Ours {
    /// The hops, the last the end.
    path: Vec<Relay>,
    /// The first link, once its session is up.
    session: Option<u32>,
    circ: u32,
    hops: Vec<(Keys, u64)>,
    pending: Option<Pending>,
    purpose: Purpose,
    last: Instant,
}

#[derive(Default)]
pub(super) struct Circuits {
    relayed: HashMap<u32, Relayed>,
    by_prev: HashMap<(u32, u32), u32>,
    by_next: HashMap<(u32, u32), u32>,
    ours: HashMap<u32, Ours>,
    by_link: HashMap<(u32, u32), u32>,
    next_id: u32,
}

fn addr_bytes(a: &SocketAddr) -> Vec<u8> {
    let mut out = match a.ip() {
        IpAddr::V4(ip) => [&[4u8][..], &ip.octets()].concat(),
        IpAddr::V6(ip) => [&[6u8][..], &ip.octets()].concat(),
    };
    out.extend(a.port().to_le_bytes());
    out
}

fn addr_from(b: &[u8]) -> Option<(SocketAddr, &[u8])> {
    let (&kind, rest) = b.split_first()?;
    let (ip, rest) = match kind {
        4 => {
            let (ip, rest) = rest.split_at_checked(4)?;
            (
                IpAddr::V4(Ipv4Addr::from(<[u8; 4]>::try_from(ip).ok()?)),
                rest,
            )
        }
        6 => {
            let (ip, rest) = rest.split_at_checked(16)?;
            (
                IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(ip).ok()?)),
                rest,
            )
        }
        _ => return None,
    };
    let (port, rest) = rest.split_at_checked(2)?;
    Some((
        SocketAddr::new(ip, u16::from_le_bytes(port.try_into().ok()?)),
        rest,
    ))
}

impl<B: Backend> Node<B> {
    fn random_bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for c in out.chunks_mut(4) {
            let w = self.rng.next_element().as_u32().to_le_bytes();
            c.copy_from_slice(&w[..c.len()]);
        }
        out
    }

    /// A new local circuit number.
    fn circuit_id(&mut self) -> u32 {
        self.circuits.next_id = self.circuits.next_id.wrapping_add(1);
        self.circuits.next_id
    }

    /// A link circuit id on `session` not in `next` (else `link`) yet.
    fn link_id(&mut self, session: u32, next: bool) -> u32 {
        loop {
            let c = self.rng.next_element().as_u32();
            let used = if next {
                &self.circuits.by_next
            } else {
                &self.circuits.by_link
            };
            if !used.contains_key(&(session, c)) {
                return c;
            }
        }
    }

    fn session_at(&self, addr: &SocketAddr) -> Option<u32> {
        self.peers
            .iter()
            .find(|(_, p)| p.addr == *addr)
            .map(|(&s, _)| s)
    }

    fn dial_relay(&mut self, addr: SocketAddr) {
        if !self.ours.contains(&addr) {
            self.dials.entry(addr).or_insert(Dial {
                last_sent: None,
                cookie: None,
                tries: 0,
                persistent: false,
            });
        }
    }

    /// A post message as bytes for a circuit.
    fn post_bytes(&self, msg: &Message) -> Vec<u8> {
        wire::encode(self.cfg.magic, Packet { session: 0, seq: 0 }, msg)
    }

    // Relaying.

    pub(super) fn on_create(
        &mut self,
        session: u32,
        circ: u32,
        challenge: [u8; 32],
        ek: Box<[u8; mlkem::EK_LEN]>,
    ) {
        let Some(keys) = self.cfg.relay.clone() else {
            return;
        };
        if let Some(&id) = self.circuits.by_prev.get(&(session, circ)) {
            // Asked again: the same answer.
            let created = self.circuits.relayed[&id].created.clone();
            self.send_created(session, circ, &created);
            return;
        }
        let (m, rnd) = (self.random_bytes::<32>(), self.random_bytes::<32>());
        let Some((created, hop_keys)) = oc::answer(&keys.pk, &keys.sk, &challenge, &ek, &m, &rnd)
        else {
            return;
        };
        let id = self.circuit_id();
        self.circuits.by_prev.insert((session, circ), id);
        self.circuits.relayed.insert(
            id,
            Relayed {
                prev: (session, circ),
                keys: hop_keys,
                back_n: 0,
                created: created.clone(),
                next: None,
                extending: None,
                last: Instant::now(),
            },
        );
        self.send_created(session, circ, &created);
    }

    fn send_created(&mut self, session: u32, circ: u32, created: &[u8]) {
        for (index, data) in created.chunks(oc::FRAGMENT).enumerate() {
            let msg = Message::Created {
                circ,
                index: index as u8,
                data: data.to_vec(),
            };
            self.send_peer(session, &msg);
        }
    }

    pub(super) fn on_forward(&mut self, session: u32, circ: u32, cell: Box<[u8; oc::CELL_LEN]>) {
        let Some(&id) = self.circuits.by_prev.get(&(session, circ)) else {
            return;
        };
        let r = self.circuits.relayed.get_mut(&id).expect("relayed");
        let Some(pt) = oc::open(&r.keys.out, &cell[..]) else {
            return;
        };
        r.last = Instant::now();
        let Some((command, body)) = oc::unframe(&pt) else {
            return;
        };
        match command {
            cmd::RELAY => {
                if let (Some((s, c)), Some(cell)) = (r.next, oc::cell(body)) {
                    self.send_peer(s, &Message::Forward { circ: c, cell });
                }
            }
            cmd::EXTEND => {
                let Some((addr, rest)) = addr_from(body) else {
                    return;
                };
                let (Ok(challenge), Ok(ek)) = (
                    <[u8; 32]>::try_from(rest.get(..32).unwrap_or_default()),
                    <[u8; mlkem::EK_LEN]>::try_from(rest.get(32..).unwrap_or_default()),
                ) else {
                    return;
                };
                r.extending = Some(Extending {
                    addr,
                    challenge,
                    ek: Box::new(ek),
                    since: Instant::now(),
                });
                self.try_extend(id);
            }
            cmd::DATA => {
                if let Ok((_, msg)) = wire::decode(self.cfg.magic, body) {
                    self.on_post_message(Via::In(id), msg);
                }
            }
            _ => {}
        }
    }

    /// Opens (or asks again on) the next link of relayed circuit `id`, once
    /// we have a session with its relay.
    fn try_extend(&mut self, id: u32) {
        let Some(e) = self
            .circuits
            .relayed
            .get(&id)
            .and_then(|r| r.extending.as_ref())
        else {
            return;
        };
        let addr = e.addr;
        let Some(session) = self.session_at(&addr) else {
            self.dial_relay(addr);
            return;
        };
        let next = match self.circuits.relayed[&id].next {
            Some((s, c)) if s == session => c,
            _ => {
                let c = self.link_id(session, true);
                self.circuits.by_next.insert((session, c), id);
                self.circuits.relayed.get_mut(&id).expect("relayed").next = Some((session, c));
                c
            }
        };
        let r = self.circuits.relayed.get_mut(&id).expect("relayed");
        let e = r.extending.take().expect("extending");
        let msg = Message::Create {
            circ: next,
            challenge: e.challenge,
            ek: e.ek,
        };
        self.send_peer(session, &msg);
    }

    /// Sends `kind` and `body` back from relayed circuit `id`, in its layer.
    fn send_back(&mut self, id: u32, kind: u8, body: &[u8], pad: usize) {
        let Some(r) = self.circuits.relayed.get_mut(&id) else {
            return;
        };
        let layer = oc::seal(&r.keys.back, r.back_n, &oc::frame(kind, body, pad));
        r.back_n += 1;
        let (s, c) = r.prev;
        if let Some(cell) = oc::cell(&layer) {
            self.send_peer(s, &Message::Backward { circ: c, cell });
        }
    }

    /// A post message to the client of relayed circuit `id`.
    pub(super) fn relay_post(&mut self, id: u32, msg: &Message) {
        let bytes = self.post_bytes(msg);
        if bytes.len() <= oc::MAX_DATA {
            self.send_back(id, back::DATA, &bytes, oc::BACK_PLAINTEXT);
        }
    }

    pub(super) fn on_created(&mut self, session: u32, circ: u32, index: u8, data: Vec<u8>) {
        if let Some(&id) = self.circuits.by_next.get(&(session, circ)) {
            // The next hop's answer: back to the client, a piece at a time.
            let body = [&[index][..], &data].concat();
            self.send_back(id, back::EXTENDED, &body, oc::BACK_PLAINTEXT);
        } else if let Some(&id) = self.circuits.by_link.get(&(session, circ)) {
            self.on_piece(id, 0, index, data);
        }
    }

    pub(super) fn on_backward(&mut self, session: u32, circ: u32, cell: Box<[u8; oc::CELL_LEN]>) {
        if let Some(&id) = self.circuits.by_next.get(&(session, circ)) {
            let len = u16::from_le_bytes([cell[0], cell[1]]) as usize;
            if let Some(inner) = cell.get(..2 + len) {
                let inner = inner.to_vec();
                if let Some(r) = self.circuits.relayed.get_mut(&id) {
                    r.last = Instant::now();
                }
                self.send_back(id, back::RELAYED, &inner, 0);
            }
            return;
        }
        let Some(&id) = self.circuits.by_link.get(&(session, circ)) else {
            return;
        };
        let Some(o) = self.circuits.ours.get(&id) else {
            return;
        };
        let keys: Vec<Keys> = o.hops.iter().map(|(k, _)| *k).collect();
        let Some((from, kind, body)) = oc::unwrap_back(&keys, &cell[..]) else {
            return;
        };
        match kind {
            back::EXTENDED if from + 1 == keys.len() => {
                if let Some((&index, data)) = body.split_first() {
                    self.on_piece(id, from + 1, index, data.to_vec());
                }
            }
            back::DATA if from + 1 == o.path.len() => {
                if let Some(o) = self.circuits.ours.get_mut(&id) {
                    o.last = Instant::now();
                }
                if let Ok((_, msg)) = wire::decode(self.cfg.magic, &body) {
                    self.on_post_message(Via::Out(id), msg);
                }
            }
            _ => {}
        }
    }

    // Building ours.

    /// Starts a circuit through `path` (the last hop its end): its number.
    pub(super) fn build_circuit(&mut self, path: Vec<Relay>, purpose: Purpose) -> u32 {
        let id = self.circuit_id();
        let pending = self.new_pending();
        self.circuits.ours.insert(
            id,
            Ours {
                path,
                session: None,
                circ: 0,
                hops: Vec::new(),
                pending: Some(pending),
                purpose,
                last: Instant::now(),
            },
        );
        self.step(id);
        id
    }

    fn new_pending(&mut self) -> Pending {
        let (d, z) = (self.random_bytes::<32>(), self.random_bytes::<32>());
        let (ek, dk) = mlkem::keygen(&d, &z);
        Pending {
            challenge: self.random_bytes(),
            ek: Box::new(ek),
            dk: Box::new(dk),
            pieces: vec![None; oc::FRAGMENTS],
            asked: None,
            tries: 0,
        }
    }

    /// Sends (or sends again) circuit `id`'s pending handshake step.
    fn step(&mut self, id: u32) {
        let Some(o) = self.circuits.ours.get(&id) else {
            return;
        };
        let Some(p) = &o.pending else {
            return;
        };
        let depth = o.hops.len();
        if depth == 0 {
            let first = o.path[0].addr;
            let (old_session, old_circ) = (o.session, o.circ);
            let Some(session) = self.session_at(&first) else {
                self.dial_relay(first);
                return;
            };
            let circ = if old_session == Some(session) {
                old_circ
            } else {
                if let Some(s) = old_session {
                    self.circuits.by_link.remove(&(s, old_circ));
                }
                let c = self.link_id(session, false);
                let o = self.circuits.ours.get_mut(&id).expect("ours");
                o.session = Some(session);
                o.circ = c;
                self.circuits.by_link.insert((session, c), id);
                c
            };
            let o = self.circuits.ours.get_mut(&id).expect("ours");
            let p = o.pending.as_mut().expect("pending");
            p.asked = Some(Instant::now());
            p.tries += 1;
            let msg = Message::Create {
                circ,
                challenge: p.challenge,
                ek: p.ek.clone(),
            };
            self.send_peer(session, &msg);
        } else {
            let body = [
                &addr_bytes(&o.path[depth].addr)[..],
                &p.challenge,
                &p.ek[..],
            ]
            .concat();
            let (session, circ) = (o.session, o.circ);
            let o = self.circuits.ours.get_mut(&id).expect("ours");
            let p = o.pending.as_mut().expect("pending");
            p.asked = Some(Instant::now());
            p.tries += 1;
            let cell = oc::wrap_out(&mut o.hops, depth - 1, cmd::EXTEND, &body);
            if let (Some(s), Some(cell)) = (session, cell) {
                self.send_peer(s, &Message::Forward { circ, cell });
            }
        }
    }

    /// Piece `index` of hop `depth`'s handshake answer on circuit `id`.
    fn on_piece(&mut self, id: u32, depth: usize, index: u8, data: Vec<u8>) {
        let Some(o) = self.circuits.ours.get_mut(&id) else {
            return;
        };
        if depth != o.hops.len() {
            return;
        }
        let Some(p) = o.pending.as_mut() else {
            return;
        };
        let Some(slot) = p.pieces.get_mut(index as usize) else {
            return;
        };
        *slot = Some(data);
        if p.pieces.iter().any(Option::is_none) {
            return;
        }
        let created: Vec<u8> = p.pieces.iter().flatten().flatten().copied().collect();
        let relay = o.path[depth];
        match oc::finish(&relay.key_hash, &p.challenge, &p.ek, &p.dk, &created) {
            Some(keys) => {
                o.hops.push((keys, 0));
                o.last = Instant::now();
                if o.hops.len() == o.path.len() {
                    o.pending = None;
                    let purpose = o.purpose;
                    self.circuit_ready(id, purpose);
                } else {
                    let pending = self.new_pending();
                    self.circuits.ours.get_mut(&id).expect("ours").pending = Some(pending);
                    self.step(id);
                }
            }
            None => self.fail_circuit(id, &format!("relay {} failed its handshake", relay.addr)),
        }
    }

    /// Whether our circuit `id` exists (built or building).
    pub(super) fn circuits_has(&self, id: u32) -> bool {
        self.circuits.ours.contains_key(&id)
    }

    /// Whether circuit `id` is built.
    pub(super) fn circuit_ready_now(&self, id: u32) -> bool {
        self.circuits
            .ours
            .get(&id)
            .is_some_and(|o| o.pending.is_none())
    }

    /// A post message to the end of our circuit `id`.
    pub(super) fn circuit_post(&mut self, id: u32, msg: &Message) {
        let bytes = self.post_bytes(msg);
        let Some(o) = self.circuits.ours.get_mut(&id) else {
            return;
        };
        if o.pending.is_some() || bytes.len() > oc::MAX_DATA {
            return;
        }
        let depth = o.hops.len() - 1;
        let (session, circ) = (o.session, o.circ);
        if let (Some(s), Some(cell)) =
            (session, oc::wrap_out(&mut o.hops, depth, cmd::DATA, &bytes))
        {
            self.send_peer(s, &Message::Forward { circ, cell });
        }
    }

    /// Drops our circuit `id` (its relays let it time out).
    pub(super) fn close_circuit(&mut self, id: u32) {
        if let Some(o) = self.circuits.ours.remove(&id)
            && let Some(s) = o.session
        {
            self.circuits.by_link.remove(&(s, o.circ));
        }
    }

    fn fail_circuit(&mut self, id: u32, why: &str) {
        let Some(o) = self.circuits.ours.get(&id) else {
            return;
        };
        let purpose = o.purpose;
        self.close_circuit(id);
        self.circuit_failed(id, purpose, why);
    }

    /// Our circuits and relayed ones that used `session` are gone.
    pub(super) fn circuits_lost_session(&mut self, session: u32) {
        let gone: Vec<u32> = self
            .circuits
            .relayed
            .iter()
            .filter(|(_, r)| r.prev.0 == session)
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            self.drop_relayed(id);
        }
        for r in self.circuits.relayed.values_mut() {
            if r.next.is_some_and(|(s, _)| s == session) {
                r.next = None;
            }
        }
        self.circuits.by_next.retain(|(s, _), _| *s != session);
        let ours: Vec<u32> = self
            .circuits
            .ours
            .iter()
            .filter(|(_, o)| o.session == Some(session))
            .map(|(&id, _)| id)
            .collect();
        for id in ours {
            self.fail_circuit(id, "its first relay went away");
        }
    }

    fn drop_relayed(&mut self, id: u32) {
        if let Some(r) = self.circuits.relayed.remove(&id) {
            self.circuits.by_prev.remove(&r.prev);
            if let Some(n) = r.next {
                self.circuits.by_next.remove(&n);
            }
            self.post_lost_via(Via::In(id));
        }
    }

    pub(super) fn circuits_tick(&mut self, now: Instant) {
        // Relayed: drop idle ones, open waiting links.
        let idle: Vec<u32> = self
            .circuits
            .relayed
            .iter()
            .filter(|(_, r)| now - r.last > IDLE)
            .map(|(&id, _)| id)
            .collect();
        for id in idle {
            self.drop_relayed(id);
        }
        let waiting: Vec<(u32, Instant)> = self
            .circuits
            .relayed
            .iter()
            .filter_map(|(&id, r)| r.extending.as_ref().map(|e| (id, e.since)))
            .collect();
        for (id, since) in waiting {
            if now - since > EXTEND_TIMEOUT {
                self.circuits
                    .relayed
                    .get_mut(&id)
                    .expect("relayed")
                    .extending = None;
            } else {
                self.try_extend(id);
            }
        }
        // Ours: steps not answered.
        let late: Vec<(u32, u32, bool)> = self
            .circuits
            .ours
            .iter()
            .filter_map(|(&id, o)| {
                let p = o.pending.as_ref()?;
                p.asked.is_none_or(|t| now - t > STEP_TIMEOUT).then_some((
                    id,
                    p.tries,
                    p.asked.is_none(),
                ))
            })
            .collect();
        for (id, tries, never) in late {
            if tries >= STEP_TRIES || (never && now - self.circuits.ours[&id].last > EXTEND_TIMEOUT)
            {
                let o = &self.circuits.ours[&id];
                let at = o.path[o.hops.len()].addr;
                self.fail_circuit(id, &format!("no answer from relay {at}"));
            } else {
                self.step(id);
            }
        }
    }
}
