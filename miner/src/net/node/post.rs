//! The slate post ([`crate::payments`]): wallets that are online pass
//! encrypted slates through a host. In init mode the hosts are the seed
//! nodes, reached directly; in onion mode ([`crate::onion`]) they're the
//! recipient's rendezvous points, reached through circuits ([`super::relay`]).
//! The same messages run over both ([`Via`]). Nothing is stored for anyone
//! offline.
//!
//! **Hosts.** Every node can host. A wallet registers its payment address
//! (`REGISTER`, with its public key); the host seals a secret to the key
//! (`CHALLENGE`) and registers the address to where it came from once the
//! wallet proves it opened it (`PROVE`, `REGISTERED`). The wallet refreshes
//! it every [`REGISTER_EVERY`]; a registration not refreshed for [`GRACE`]
//! (or whose session or circuit ends) is gone, and the address counts as
//! offline. A `LOOKUP` gets the key of a registered address
//! (`ADDRESS_KEY`) or `OFFLINE`; a `DELIVER` for one is passed on (answering
//! the sender `FORWARDED`), or answered `OFFLINE`.
//!
//! **Init mode.** A wallet with [`Config::post`] stays connected to its
//! hosts and registers with each. To send ([`Event::Post`]) it looks the
//! address up with all of them (its own registrations first, if it hosts
//! too), checks the key against the address, seals the payload and delivers
//! it through the host that knew it.
//!
//! **Onion mode.** A wallet keeps a circuit open to each of its
//! [`onion::RENDEZVOUS`] rendezvous points (2 hops, then the point) and
//! registers through it. To send, it tries the recipient's rendezvous
//! points one at a time, in random order, each through a new circuit:
//! look up, deliver. A point that says offline, or can't be reached, moves
//! it on to the next.
//!
//! Either way it retries until the host forwards the delivery (and, for a
//! payment request, until the wallet has its answer: [`Event::PostDone`]).
//! When every host says offline, or time runs out, the post fails
//! ([`PostEvent::Failed`]). A delivery for our address is opened and
//! handed to the wallet ([`PostEvent::Received`]).

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use super::relay::{Purpose, Via};
use super::{Backend, Message, Node};
use crate::aux_leaves::Leaf;
use crate::crypto::mlkem;
use crate::onion::{self, Relay};
use crate::payments::{self, Address, Payload, PaymentKeys};
use crate::wallet::{SlateId, hex};

/// Refresh registrations this often.
pub const REGISTER_EVERY: Duration = Duration::from_secs(60);
/// A registration lasts this long after its last refresh.
pub const GRACE: Duration = Duration::from_secs(120);
/// Look up the hosts' addresses again this often.
const RESOLVE_EVERY: Duration = Duration::from_secs(600);
/// Re-send a lookup or delivery not answered within this.
const RETRY: Duration = Duration::from_secs(2);
/// Give up on a post that needs an answer (a payment request) after this,
/// and on others after half of it (init mode).
const POST_TIMEOUT: Duration = Duration::from_secs(30);
/// Onion mode: the same, with circuits to build to up to 3 points.
const ONION_POST_TIMEOUT: Duration = Duration::from_secs(60);
/// Onion mode: move on from a rendezvous point that hasn't forwarded in
/// this.
const POINT_TIMEOUT: Duration = Duration::from_secs(15);
/// A registration circuit whose point hasn't confirmed in this is rebuilt.
const CONFIRM_TIMEOUT: Duration = Duration::from_secs(15);
/// Wait this long before rebuilding a failed registration circuit.
const REBUILD_AFTER: Duration = Duration::from_secs(5);
/// A challenge must be proved within this.
const CHALLENGE_TIMEOUT: Duration = Duration::from_secs(10);
/// Most registrations and open challenges a host keeps.
const MAX_REGISTRATIONS: usize = 100_000;
const MAX_CHALLENGES: usize = 10_000;

/// A wallet's side of the post.
#[derive(Clone, Debug)]
pub struct PostConfig {
    pub keys: PaymentKeys,
    /// Init mode's hosts (`host:port`).
    pub hosts: Vec<String>,
    /// Where opened payloads and failures go.
    pub events: Sender<PostEvent>,
}

/// Something to send to a payment address.
#[derive(Clone, Debug)]
pub struct Post {
    pub to: Address,
    /// The payment it's about (for [`Event::PostDone`] and failures).
    pub id: SlateId,
    pub payload: Payload,
    /// Keep delivering until the wallet has its answer (a payment
    /// request), not just until a host forwards it.
    pub wait_for_answer: bool,
}

/// What the post tells the wallet.
#[derive(Clone, Debug)]
pub enum PostEvent {
    Received(Payload),
    Failed {
        id: SlateId,
        to: Address,
        why: String,
    },
}

struct Registration {
    via: Via,
    ek: Box<[u8; mlkem::EK_LEN]>,
    last: Instant,
}

/// An open challenge: the proof it wants, the key, and when it was sent.
type Challenge = ([u8; 32], Box<[u8; mlkem::EK_LEN]>, Instant);

enum Phase {
    /// Asking for the key; these said offline.
    Lookup { offline: HashSet<Via> },
    /// Delivering through `via`.
    Deliver {
        via: Via,
        ct: Box<[u8; mlkem::CT_LEN]>,
        data: Vec<u8>,
        forwarded: bool,
    },
}

struct Outgoing {
    post: Post,
    phase: Phase,
    started: Instant,
    last_try: Option<Instant>,
    /// Onion mode: the rendezvous points still to try, the circuit to the
    /// current one, and when we moved to it.
    points: Vec<Relay>,
    circuit: Option<u32>,
    point_started: Instant,
    /// Why the last point didn't work.
    why: Option<String>,
}

/// Where we register: when we last did, whether it was answered since,
/// and whether it ever was.
type Registered = (Instant, bool, bool);

#[derive(Default)]
pub(super) struct PostState {
    // Hosting.
    registrations: HashMap<Address, Registration>,
    challenges: HashMap<(Via, Address), Challenge>,
    // Init mode's hosts.
    hosts: Vec<SocketAddr>,
    resolved: Option<Instant>,
    registered: HashMap<Via, Registered>,
    outgoing: HashMap<u32, Outgoing>,
    next_post: u32,
    /// Onion mode: the relays, and the consensus file.
    relays: Option<(Vec<Relay>, Leaf)>,
    /// Our registration circuits, by rendezvous point; and when a failed
    /// one may be rebuilt.
    point_circuits: HashMap<Relay, u32>,
    rebuild: HashMap<Relay, Instant>,
}

impl PostState {
    /// Whether posts are being sent.
    pub(super) fn busy(&self) -> bool {
        !self.outgoing.is_empty()
    }
}

impl<B: Backend> Node<B> {
    fn random_32(&mut self) -> [u8; 32] {
        let mut m = [0u8; 32];
        for c in m.chunks_exact_mut(4) {
            c.copy_from_slice(&self.rng.next_element().as_u32().to_le_bytes());
        }
        m
    }

    fn our_address(&self) -> Option<Address> {
        self.cfg.post.as_ref().map(|p| p.keys.address)
    }

    /// Our relay identity's key hash, if we relay.
    fn our_relay(&self) -> Option<Leaf> {
        self.cfg.relay.as_ref().map(|k| onion::key_hash(&k.pk))
    }

    /// A registration still in its grace period.
    fn registration(&self, address: &Address) -> Option<&Registration> {
        self.post
            .registrations
            .get(address)
            .filter(|r| r.last.elapsed() < GRACE)
    }

    fn via_name(&self, via: Via) -> String {
        match via {
            Via::Session(s) => self
                .peers
                .get(&s)
                .map_or("?".into(), |p| p.addr.to_string()),
            Via::Out(c) => format!("our circuit {c}"),
            Via::In(c) => format!("relayed circuit {c}"),
        }
    }

    fn send_via(&mut self, via: Via, msg: &Message) {
        match via {
            Via::Session(s) => self.send_peer(s, msg),
            Via::Out(c) => self.circuit_post(c, msg),
            Via::In(c) => self.relay_post(c, msg),
        }
    }

    /// Takes a post message from a session or a circuit.
    pub(super) fn on_post_message(&mut self, via: Via, msg: Message) {
        match msg {
            Message::Register { ek } => self.on_register(via, ek),
            Message::Challenge { address, ct } => self.on_challenge(via, address, ct),
            Message::Prove { address, proof } => self.on_prove(via, address, proof),
            Message::Registered { address } => self.on_registered(via, address),
            Message::Lookup { address } => self.on_lookup(via, address),
            Message::AddressKey { address, ek } => self.on_address_key(via, address, ek),
            Message::Offline { address } => self.on_offline(via, address),
            Message::Deliver { to, post, ct, data } => self.on_deliver(via, to, post, ct, data),
            Message::Forwarded { to, post } => self.on_forwarded(via, to, post),
            _ => {}
        }
    }

    /// A peer misbehaved: drop it, if it's a session.
    fn misbehaved(&mut self, via: Via, why: &str) {
        if let Via::Session(s) = via {
            self.drop_peer(s, why);
        }
    }

    /// The relay consensus changed ([`Event::Relays`]): onion mode over
    /// these relays and file, or init mode.
    pub(super) fn set_relays(&mut self, relays: Option<(Vec<Relay>, Leaf)>) {
        if self.post.relays == relays {
            return;
        }
        (self.log)(match &relays {
            Some((r, file)) => format!(
                "post: onion mode, {} relays (consensus {})",
                r.len(),
                &hex(file)[..16]
            ),
            None => "post: init mode".into(),
        });
        let circuits: Vec<u32> = self.post.point_circuits.drain().map(|(_, c)| c).collect();
        for c in circuits {
            self.close_circuit(c);
        }
        self.post.registered.clear();
        self.post.rebuild.clear();
        self.post.relays = relays;
    }

    // Hosting.

    fn on_register(&mut self, via: Via, ek: Box<[u8; mlkem::EK_LEN]>) {
        if !mlkem::check_ek(&ek) {
            return;
        }
        let address = Address::of(&ek);
        if let Some(r) = self.post.registrations.get_mut(&address)
            && r.via == via
        {
            r.last = Instant::now();
            self.send_via(via, &Message::Registered { address });
            return;
        }
        if self.post.challenges.len() >= MAX_CHALLENGES {
            return;
        }
        let m = self.random_32();
        let (ct, secret) = mlkem::encapsulate(&ek, &m).expect("checked key");
        let proof = payments::register_proof(&address, &secret);
        self.post
            .challenges
            .insert((via, address), (proof, ek, Instant::now()));
        self.send_via(
            via,
            &Message::Challenge {
                address,
                ct: Box::new(ct),
            },
        );
    }

    fn on_prove(&mut self, via: Via, address: Address, proof: [u8; 32]) {
        let Some((want, ek, _)) = self.post.challenges.remove(&(via, address)) else {
            return;
        };
        if want != proof {
            self.misbehaved(via, "failed to prove it holds a payment address");
            return;
        }
        if !self.post.registrations.contains_key(&address)
            && self.post.registrations.len() >= MAX_REGISTRATIONS
        {
            return;
        }
        // The latest wins (a wallet that reconnected).
        self.post.registrations.insert(
            address,
            Registration {
                via,
                ek,
                last: Instant::now(),
            },
        );
        self.send_via(via, &Message::Registered { address });
    }

    fn on_lookup(&mut self, via: Via, address: Address) {
        let ek = match &self.cfg.post {
            // Our own wallet is online.
            Some(p) if p.keys.address == address => Some(Box::new(p.keys.ek)),
            _ => self.registration(&address).map(|r| r.ek.clone()),
        };
        let msg = match ek {
            Some(ek) => Message::AddressKey { address, ek },
            None => Message::Offline { address },
        };
        self.send_via(via, &msg);
    }

    fn on_deliver(
        &mut self,
        via: Via,
        to: Address,
        post: u32,
        ct: Box<[u8; mlkem::CT_LEN]>,
        data: Vec<u8>,
    ) {
        if self.our_address() == Some(to) {
            // For us: passed on by a host we registered with, or from a
            // sender who found us as a host (we answer for our own
            // address), who wants to know it arrived.
            if !self.post.registered.contains_key(&via) {
                self.send_via(via, &Message::Forwarded { to, post });
            }
            self.open_delivery(via, &ct, &data);
            return;
        }
        match self.registration(&to).map(|r| r.via) {
            Some(v) if v != via => {
                self.send_via(v, &Message::Deliver { to, post, ct, data });
                self.send_via(via, &Message::Forwarded { to, post });
            }
            _ => self.send_via(via, &Message::Offline { address: to }),
        }
    }

    fn open_delivery(&mut self, via: Via, ct: &[u8; mlkem::CT_LEN], data: &[u8]) {
        let Some(p) = &self.cfg.post else {
            return;
        };
        let payload = p.keys.open(ct, data).and_then(|b| Payload::from_bytes(&b));
        match payload {
            Some(payload) => {
                let _ = p.events.send(PostEvent::Received(payload));
            }
            None => (self.log)(format!(
                "post: a delivery from {} didn't open (not sealed to our key)",
                self.via_name(via)
            )),
        }
    }

    // Registering.

    fn on_challenge(&mut self, via: Via, address: Address, ct: Box<[u8; mlkem::CT_LEN]>) {
        let Some(p) = &self.cfg.post else {
            return;
        };
        if p.keys.address != address || !self.post.registered.contains_key(&via) {
            return;
        }
        let proof = payments::register_proof(&address, &p.keys.decapsulate(&ct));
        self.send_via(via, &Message::Prove { address, proof });
    }

    fn on_registered(&mut self, via: Via, address: Address) {
        if self.our_address() != Some(address) {
            return;
        }
        let Some((_, answered, ever)) = self.post.registered.get_mut(&via) else {
            return;
        };
        *answered = true;
        if !*ever {
            *ever = true;
            let name = match via {
                Via::Out(c) => self
                    .post
                    .point_circuits
                    .iter()
                    .find(|(_, id)| **id == c)
                    .map_or("?".into(), |(r, _)| format!("rendezvous point {}", r.addr)),
                _ => format!("host {}", self.via_name(via)),
            };
            (self.log)(format!("post: payment address registered with {name}"));
        }
    }

    fn register(&mut self, via: Via, ek: &[u8; mlkem::EK_LEN], now: Instant) {
        let ever = self.post.registered.get(&via).is_some_and(|r| r.2);
        self.post.registered.insert(via, (now, false, ever));
        self.send_via(via, &Message::Register { ek: Box::new(*ek) });
    }

    /// Init mode: sessions with our hosts.
    fn host_sessions(&self) -> Vec<Via> {
        let mut s: Vec<u32> = self
            .peers
            .iter()
            .filter(|(_, p)| self.post.hosts.contains(&p.addr))
            .map(|(&s, _)| s)
            .collect();
        s.sort_unstable();
        s.into_iter().map(Via::Session).collect()
    }

    /// Two relays to reach `point` through (fewer if there aren't enough),
    /// at random, then the point.
    fn path_to(&mut self, point: Relay) -> Vec<Relay> {
        let ours = self.our_relay();
        let mut others: Vec<Relay> = self
            .post
            .relays
            .as_ref()
            .map(|(r, _)| r.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|r| *r != point && Some(r.key_hash) != ours)
            .collect();
        let mut path = Vec::new();
        while path.len() < 2 && !others.is_empty() {
            let i = self.rng.next_element().as_u32() as usize % others.len();
            path.push(others.swap_remove(i));
        }
        path.push(point);
        path
    }

    /// Onion mode: the rendezvous points of `address`.
    fn points_of(&self, address: &Address) -> Vec<Relay> {
        match &self.post.relays {
            Some((relays, file)) => onion::rendezvous(address, relays, file),
            None => vec![],
        }
    }

    /// Our circuit `id` (for `purpose`) is built.
    pub(super) fn circuit_ready(&mut self, id: u32, purpose: Purpose) {
        match purpose {
            Purpose::Register(point) if self.post.point_circuits.get(&point) == Some(&id) => {
                if let Some(p) = self.cfg.post.clone() {
                    self.register(Via::Out(id), &p.keys.ek, Instant::now());
                }
            }
            Purpose::Send(n)
                if self
                    .post
                    .outgoing
                    .get(&n)
                    .is_some_and(|o| o.circuit == Some(id)) =>
            {
                self.try_post(n)
            }
            _ => self.close_circuit(id),
        }
    }

    /// Our circuit `id` (for `purpose`) couldn't be built, or broke.
    pub(super) fn circuit_failed(&mut self, id: u32, purpose: Purpose, why: &str) {
        match purpose {
            Purpose::Register(point) => {
                if self.post.point_circuits.get(&point) == Some(&id) {
                    self.post.point_circuits.remove(&point);
                    self.post
                        .rebuild
                        .insert(point, Instant::now() + REBUILD_AFTER);
                    (self.log)(format!(
                        "post: circuit to rendezvous point {} failed: {why}",
                        point.addr
                    ));
                }
                self.post_lost_via(Via::Out(id));
            }
            Purpose::Send(n) => {
                if self
                    .post
                    .outgoing
                    .get(&n)
                    .is_some_and(|o| o.circuit == Some(id))
                {
                    self.post.outgoing.get_mut(&n).expect("post").circuit = None;
                    self.next_point(n, why.to_string());
                }
            }
        }
    }

    /// Whatever went through `via` (a session or circuit that's gone).
    pub(super) fn post_lost_via(&mut self, via: Via) {
        self.post.registrations.retain(|_, r| r.via != via);
        self.post.challenges.retain(|(v, _), _| *v != via);
        self.post.registered.remove(&via);
        for o in self.post.outgoing.values_mut() {
            if matches!(o.phase, Phase::Deliver { via: v, .. } if v == via) && o.circuit.is_none() {
                o.phase = Phase::Lookup {
                    offline: HashSet::new(),
                };
                o.last_try = None;
            }
        }
    }

    // Sending.

    pub(super) fn start_post(&mut self, post: Post) {
        let n = self.post.next_post;
        self.post.next_post = n.wrapping_add(1);
        // Onion mode: the recipient's rendezvous points, in random order.
        let mut points = self.points_of(&post.to);
        for i in (1..points.len()).rev() {
            let j = self.rng.next_element().as_u32() as usize % (i + 1);
            points.swap(i, j);
        }
        let onion = self.post.relays.is_some();
        self.post.outgoing.insert(
            n,
            Outgoing {
                post,
                phase: Phase::Lookup {
                    offline: HashSet::new(),
                },
                started: Instant::now(),
                last_try: None,
                points,
                circuit: None,
                point_started: Instant::now(),
                why: None,
            },
        );
        if onion {
            self.next_point(n, String::new());
        } else {
            self.try_post(n);
        }
    }

    /// Onion mode: moves post `n` on to its next rendezvous point (the last
    /// one didn't work, for `why`), or fails it.
    fn next_point(&mut self, n: u32, why: String) {
        let Some(o) = self.post.outgoing.get_mut(&n) else {
            return;
        };
        if !why.is_empty() {
            o.why = Some(why);
        }
        if let Some(c) = o.circuit.take() {
            self.close_circuit(c);
        }
        let o = self.post.outgoing.get_mut(&n).expect("post");
        let Some(point) = o.points.pop() else {
            let why = o.why.clone().unwrap_or_default();
            let to = o.post.to;
            self.fail_post(
                n,
                format!(
                    "{to} is offline or unreachable at all its rendezvous points (last: {why})"
                ),
            );
            return;
        };
        o.phase = Phase::Lookup {
            offline: HashSet::new(),
        };
        o.point_started = Instant::now();
        o.last_try = None;
        let to = o.post.to;
        if Some(point.key_hash) == self.our_relay() {
            // We're the point: deliver straight to its registration.
            match self.registration(&to).map(|r| (r.via, r.ek.clone())) {
                Some((via, ek)) => self.deliver_post(n, via, &ek),
                None => self.next_point(n, format!("{to} isn't registered with us")),
            }
            return;
        }
        let path = self.path_to(point);
        let c = self.build_circuit(path, Purpose::Send(n));
        let building = self.circuits_has(c);
        match self.post.outgoing.get_mut(&n) {
            Some(o) if building => o.circuit = Some(c),
            _ => self.close_circuit(c),
        }
    }

    pub(super) fn post_done(&mut self, id: &SlateId) {
        let done: Vec<u32> = self
            .post
            .outgoing
            .iter()
            .filter(|(_, o)| o.post.id == *id)
            .map(|(&n, _)| n)
            .collect();
        for n in done {
            self.end_post(n);
        }
    }

    fn end_post(&mut self, n: u32) -> Option<Outgoing> {
        let o = self.post.outgoing.remove(&n)?;
        if let Some(c) = o.circuit {
            self.close_circuit(c);
        }
        Some(o)
    }

    fn fail_post(&mut self, n: u32, why: String) {
        let Some(o) = self.end_post(n) else {
            return;
        };
        (self.log)(format!(
            "post: payment {} to {}: {why}",
            &hex(&o.post.id)[..8],
            o.post.to
        ));
        if let Some(p) = &self.cfg.post {
            let _ = p.events.send(PostEvent::Failed {
                id: o.post.id,
                to: o.post.to,
                why,
            });
        }
    }

    /// Seals post `n` to `ek` and delivers it through `via`.
    fn deliver_post(&mut self, n: u32, via: Via, ek: &[u8; mlkem::EK_LEN]) {
        let m = self.random_32();
        let Some(o) = self.post.outgoing.get_mut(&n) else {
            return;
        };
        let Some((ct, data)) = payments::seal(ek, &o.post.payload.to_bytes(), &m) else {
            return;
        };
        o.phase = Phase::Deliver {
            via,
            ct: Box::new(ct),
            data,
            forwarded: false,
        };
        o.last_try = None;
        self.try_post(n);
    }

    /// (Re)sends post `n`'s current step.
    fn try_post(&mut self, n: u32) {
        let hosts = self.host_sessions();
        let Some(o) = self.post.outgoing.get_mut(&n) else {
            return;
        };
        o.last_try = Some(Instant::now());
        let (to, circuit) = (o.post.to, o.circuit);
        match &o.phase {
            Phase::Lookup { offline } => {
                let offline = offline.clone();
                if let Some(c) = circuit {
                    // Onion mode: ask the point, once the circuit is up.
                    if self.circuit_ready_now(c) {
                        self.send_via(Via::Out(c), &Message::Lookup { address: to });
                    }
                    return;
                }
                if self.post.relays.is_some() {
                    return;
                }
                // A wallet we host: deliver straight to it.
                if let Some(r) = self.registration(&to) {
                    let (via, ek) = (r.via, r.ek.clone());
                    self.deliver_post(n, via, &ek);
                    return;
                }
                let ask: Vec<Via> = hosts.into_iter().filter(|s| !offline.contains(s)).collect();
                for via in ask {
                    self.send_via(via, &Message::Lookup { address: to });
                }
            }
            Phase::Deliver { via, ct, data, .. } => {
                let msg = Message::Deliver {
                    to,
                    post: n,
                    ct: ct.clone(),
                    data: data.clone(),
                };
                let via = *via;
                // Straight to a wallet we host: no one to say forwarded.
                let direct = self.registration(&to).is_some_and(|r| r.via == via);
                self.send_via(via, &msg);
                if direct {
                    self.on_forwarded(via, to, n);
                }
            }
        }
    }

    fn on_address_key(&mut self, via: Via, address: Address, ek: Box<[u8; mlkem::EK_LEN]>) {
        if Address::of(&ek) != address || !mlkem::check_ek(&ek) {
            self.misbehaved(via, "sent a key that isn't the address's");
            return;
        }
        let waiting: Vec<u32> = self
            .post
            .outgoing
            .iter()
            .filter(|(_, o)| {
                o.post.to == address
                    && matches!(o.phase, Phase::Lookup { .. })
                    && o.circuit.is_none_or(|c| via == Via::Out(c))
            })
            .map(|(&n, _)| n)
            .collect();
        for n in waiting {
            self.deliver_post(n, via, &ek);
        }
    }

    fn on_offline(&mut self, via: Via, address: Address) {
        let hosts = self.host_sessions();
        let (mut failed, mut moved) = (Vec::new(), Vec::new());
        for (&n, o) in self.post.outgoing.iter_mut() {
            if o.post.to != address {
                continue;
            }
            if let Some(c) = o.circuit {
                if via == Via::Out(c) {
                    moved.push(n);
                }
                continue;
            }
            match &mut o.phase {
                Phase::Lookup { offline } => {
                    offline.insert(via);
                    if hosts.iter().all(|s| offline.contains(s)) {
                        failed.push(n);
                    }
                }
                // Its host lost it: look again.
                Phase::Deliver { via: v, .. } if *v == via => {
                    o.phase = Phase::Lookup {
                        offline: HashSet::from([via]),
                    };
                    o.last_try = None;
                }
                Phase::Deliver { .. } => {}
            }
        }
        for n in failed {
            self.fail_post(n, format!("{address} is offline"));
        }
        for n in moved {
            self.next_point(n, "offline there".into());
        }
    }

    fn on_forwarded(&mut self, via: Via, to: Address, post: u32) {
        let Some(o) = self.post.outgoing.get_mut(&post) else {
            return;
        };
        if o.post.to != to {
            return;
        }
        if let Phase::Deliver {
            via: v, forwarded, ..
        } = &mut o.phase
            && *v == via
        {
            *forwarded = true;
            if !o.post.wait_for_answer {
                self.end_post(post);
            }
        }
    }

    pub(super) fn post_tick(&mut self, now: Instant) {
        // Hosting: forget what's expired.
        self.post.registrations.retain(|_, r| now - r.last < GRACE);
        self.post
            .challenges
            .retain(|_, (_, _, at)| now - *at < CHALLENGE_TIMEOUT);
        let Some(cfg) = self.cfg.post.clone() else {
            return;
        };
        if self.post.relays.is_some() {
            self.keep_points(&cfg, now);
        } else {
            self.keep_hosts(&cfg, now);
        }

        // Posts: give up, move on, or try again.
        let onion = self.post.relays.is_some();
        let posts: Vec<(u32, bool, Instant, Option<Instant>, bool, Instant)> = self
            .post
            .outgoing
            .iter()
            .map(|(&n, o)| {
                let forwarded = matches!(
                    o.phase,
                    Phase::Deliver {
                        forwarded: true,
                        ..
                    }
                );
                (
                    n,
                    o.post.wait_for_answer,
                    o.started,
                    o.last_try,
                    forwarded,
                    o.point_started,
                )
            })
            .collect();
        for (n, wait, started, last_try, forwarded, point_started) in posts {
            let limit = match (onion, wait) {
                (true, _) => ONION_POST_TIMEOUT,
                (false, true) => POST_TIMEOUT,
                (false, false) => POST_TIMEOUT / 2,
            };
            if now - started > limit {
                let lookup = matches!(
                    self.post.outgoing.get(&n).map(|o| &o.phase),
                    Some(Phase::Lookup { .. })
                );
                let why = if forwarded {
                    "no answer from the receiver"
                } else if lookup && onion {
                    "its rendezvous points didn't answer"
                } else if lookup && self.host_sessions().is_empty() {
                    "not connected to any host"
                } else if lookup {
                    "no host knows the address"
                } else {
                    "couldn't deliver it"
                };
                self.fail_post(n, why.into());
            } else if onion && !forwarded && now - point_started > POINT_TIMEOUT {
                self.next_point(n, "it didn't answer".into());
            } else if last_try.is_none_or(|t| now - t >= RETRY) {
                self.try_post(n);
            }
        }
    }

    /// Init mode: stay connected to our hosts, and registered with each.
    fn keep_hosts(&mut self, cfg: &PostConfig, now: Instant) {
        if self.post.resolved.is_none_or(|t| now - t >= RESOLVE_EVERY) {
            self.post.resolved = Some(now);
            let mut hosts = Vec::new();
            for name in &cfg.hosts {
                match std::net::ToSocketAddrs::to_socket_addrs(name.as_str()) {
                    Ok(addrs) => hosts.extend(addrs.filter(|a| !self.ours.contains(a))),
                    Err(e) => (self.log)(format!("post: host {name}: can't look it up: {e}")),
                }
            }
            hosts.sort();
            hosts.dedup();
            self.post.hosts = hosts;
        }
        let connected: HashSet<SocketAddr> = self.peers.values().map(|p| p.addr).collect();
        for &addr in &self.post.hosts {
            if !connected.contains(&addr) && !self.ours.contains(&addr) {
                self.dials.entry(addr).or_insert(super::Dial {
                    last_sent: None,
                    cookie: None,
                    tries: 0,
                    persistent: true,
                });
            }
        }
        for via in self.host_sessions() {
            if self
                .post
                .registered
                .get(&via)
                .is_none_or(|(at, _, _)| now - *at >= REGISTER_EVERY)
            {
                self.register(via, &cfg.keys.ek, now);
            }
        }
    }

    /// Onion mode: keep a circuit to each of our rendezvous points, and
    /// stay registered through each.
    fn keep_points(&mut self, cfg: &PostConfig, now: Instant) {
        let ours = self.our_relay();
        let points: Vec<Relay> = self
            .points_of(&cfg.keys.address)
            .into_iter()
            // As a point ourselves, we're online there already.
            .filter(|p| Some(p.key_hash) != ours)
            .collect();
        let stale: Vec<(Relay, u32)> = self
            .post
            .point_circuits
            .iter()
            .filter(|(r, _)| !points.contains(r))
            .map(|(r, c)| (*r, *c))
            .collect();
        for (r, c) in stale {
            self.post.point_circuits.remove(&r);
            self.close_circuit(c);
            self.post.registered.remove(&Via::Out(c));
        }
        for point in points {
            match self.post.point_circuits.get(&point).copied() {
                None => {
                    if self.post.rebuild.get(&point).is_some_and(|t| now < *t) {
                        continue;
                    }
                    let path = self.path_to(point);
                    let c = self.build_circuit(path, Purpose::Register(point));
                    // (It may have failed at once.)
                    if self.circuits_has(c) {
                        self.post.point_circuits.insert(point, c);
                    }
                }
                Some(c) => {
                    let via = Via::Out(c);
                    match self.post.registered.get(&via).copied() {
                        // Not confirmed: the circuit is broken somewhere.
                        Some((at, false, _)) if now - at > CONFIRM_TIMEOUT => {
                            self.post.point_circuits.remove(&point);
                            self.post.registered.remove(&via);
                            self.close_circuit(c);
                            (self.log)(format!(
                                "post: rendezvous point {} stopped answering; rebuilding",
                                point.addr
                            ));
                        }
                        Some((at, _, _)) if now - at >= REGISTER_EVERY => {
                            self.register(via, &cfg.keys.ek, now)
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
