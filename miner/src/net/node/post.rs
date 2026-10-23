//! Init mode's slate post ([`crate::payments`]): wallets that are online
//! pass encrypted slates through hosts (the seed nodes, or later the first
//! onion relay). Nothing is stored for anyone offline.
//!
//! **Hosts.** Every node can host. A wallet registers its payment address
//! (`REGISTER`, with its public key); the host seals a secret to the key
//! (`CHALLENGE`) and registers the address to the session once the wallet
//! proves it opened it (`PROVE`, `REGISTERED`). The wallet refreshes it
//! every [`REGISTER_EVERY`]; a registration not refreshed for [`GRACE`]
//! (or whose session ends) is gone, and the address counts as offline. A
//! `LOOKUP` gets the key of a registered address (`ADDRESS_KEY`) or
//! `OFFLINE`; a `DELIVER` for one is passed on to its session (answering
//! the sender `FORWARDED`), or answered `OFFLINE`.
//!
//! **Wallets.** A node with [`Config::post`] stays connected to its hosts
//! and registers with each. To send ([`Event::Post`]) it looks the address
//! up with every host it's connected to (its own registrations first, if
//! it hosts too), checks the key against the address, seals the payload
//! and delivers it through the host that knew it, retrying until the host
//! forwards it (and, for a payment request, until the wallet has its
//! answer: [`Event::PostDone`]). If every host says the address is offline,
//! or time runs out, the post fails ([`PostEvent::Failed`]). A delivery
//! for our address is opened and handed to the wallet
//! ([`PostEvent::Received`]).

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use super::{Backend, Message, Node};
use crate::crypto::mlkem;
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
/// and on others after half of it.
const POST_TIMEOUT: Duration = Duration::from_secs(30);
/// A challenge must be proved within this.
const CHALLENGE_TIMEOUT: Duration = Duration::from_secs(10);
/// Most registrations and open challenges a host keeps.
const MAX_REGISTRATIONS: usize = 100_000;
const MAX_CHALLENGES: usize = 10_000;

/// A wallet's side of the post.
#[derive(Clone, Debug)]
pub struct PostConfig {
    pub keys: PaymentKeys,
    /// Hosts to register with and send through (`host:port`).
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

/// An open challenge: the proof it wants, the key, and when it was sent.
type Challenge = ([u8; 32], Box<[u8; mlkem::EK_LEN]>, Instant);

struct Registration {
    session: u32,
    ek: Box<[u8; mlkem::EK_LEN]>,
    last: Instant,
}

enum Phase {
    /// Asking the hosts for the key; these said offline.
    Lookup { offline: HashSet<u32> },
    /// Delivering through `session`.
    Deliver {
        session: u32,
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
}

#[derive(Default)]
pub(super) struct PostState {
    // Hosting.
    registrations: HashMap<Address, Registration>,
    challenges: HashMap<(u32, Address), Challenge>,
    // Sending and receiving.
    hosts: Vec<SocketAddr>,
    resolved: Option<Instant>,
    /// Host sessions: when we last registered, and whether it confirmed.
    registered: HashMap<u32, (Instant, bool)>,
    outgoing: HashMap<u32, Outgoing>,
    next_post: u32,
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

    /// A registration still in its grace period.
    fn registration(&self, address: &Address) -> Option<&Registration> {
        self.post
            .registrations
            .get(address)
            .filter(|r| r.last.elapsed() < GRACE)
    }

    fn peer_name(&self, session: u32) -> String {
        self.peers
            .get(&session)
            .map_or("?".into(), |p| p.addr.to_string())
    }

    // Hosting.

    pub(super) fn on_register(&mut self, session: u32, ek: Box<[u8; mlkem::EK_LEN]>) {
        if !mlkem::check_ek(&ek) {
            return;
        }
        let address = Address::of(&ek);
        if let Some(r) = self.post.registrations.get_mut(&address)
            && r.session == session
        {
            r.last = Instant::now();
            self.send_peer(session, &Message::Registered { address });
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
            .insert((session, address), (proof, ek, Instant::now()));
        self.send_peer(
            session,
            &Message::Challenge {
                address,
                ct: Box::new(ct),
            },
        );
    }

    pub(super) fn on_prove(&mut self, session: u32, address: Address, proof: [u8; 32]) {
        let Some((want, ek, _)) = self.post.challenges.remove(&(session, address)) else {
            return;
        };
        if want != proof {
            self.drop_peer(session, "failed to prove it holds a payment address");
            return;
        }
        if !self.post.registrations.contains_key(&address)
            && self.post.registrations.len() >= MAX_REGISTRATIONS
        {
            return;
        }
        // The latest session wins (a wallet that reconnected).
        self.post.registrations.insert(
            address,
            Registration {
                session,
                ek,
                last: Instant::now(),
            },
        );
        self.send_peer(session, &Message::Registered { address });
    }

    pub(super) fn on_lookup(&mut self, session: u32, address: Address) {
        let ek = match &self.cfg.post {
            // Our own wallet is online.
            Some(p) if p.keys.address == address => Some(Box::new(p.keys.ek)),
            _ => self.registration(&address).map(|r| r.ek.clone()),
        };
        let msg = match ek {
            Some(ek) => Message::AddressKey { address, ek },
            None => Message::Offline { address },
        };
        self.send_peer(session, &msg);
    }

    pub(super) fn on_deliver(
        &mut self,
        session: u32,
        to: Address,
        post: u32,
        ct: Box<[u8; mlkem::CT_LEN]>,
        data: Vec<u8>,
    ) {
        if self.our_address() == Some(to) {
            // For us: passed on by our host, or from a sender who found us
            // as a host (we answer for our own address), who wants to know
            // it arrived.
            if !self.host_sessions().contains(&session) {
                self.send_peer(session, &Message::Forwarded { to, post });
            }
            self.open_delivery(session, &ct, &data);
            return;
        }
        match self.registration(&to).map(|r| r.session) {
            Some(s) if s != session => {
                let msg = Message::Deliver { to, post, ct, data };
                self.send_peer(s, &msg);
                self.send_peer(session, &Message::Forwarded { to, post });
            }
            _ => self.send_peer(session, &Message::Offline { address: to }),
        }
    }

    fn open_delivery(&mut self, session: u32, ct: &[u8; mlkem::CT_LEN], data: &[u8]) {
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
                self.peer_name(session)
            )),
        }
    }

    // Registering with hosts.

    pub(super) fn on_challenge(
        &mut self,
        session: u32,
        address: Address,
        ct: Box<[u8; mlkem::CT_LEN]>,
    ) {
        let Some(p) = &self.cfg.post else {
            return;
        };
        if p.keys.address != address || !self.post.registered.contains_key(&session) {
            return;
        }
        let proof = payments::register_proof(&address, &p.keys.decapsulate(&ct));
        self.send_peer(session, &Message::Prove { address, proof });
    }

    pub(super) fn on_registered(&mut self, session: u32, address: Address) {
        if self.our_address() != Some(address) {
            return;
        }
        if let Some((_, confirmed)) = self.post.registered.get_mut(&session)
            && !*confirmed
        {
            *confirmed = true;
            let name = self.peer_name(session);
            (self.log)(format!("post: payment address registered with host {name}"));
        }
    }

    /// Sessions with our hosts.
    fn host_sessions(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self
            .peers
            .iter()
            .filter(|(_, p)| self.post.hosts.contains(&p.addr))
            .map(|(&s, _)| s)
            .collect();
        s.sort_unstable();
        s
    }

    // Sending.

    pub(super) fn start_post(&mut self, post: Post) {
        let n = self.post.next_post;
        self.post.next_post = n.wrapping_add(1);
        self.post.outgoing.insert(
            n,
            Outgoing {
                post,
                phase: Phase::Lookup {
                    offline: HashSet::new(),
                },
                started: Instant::now(),
                last_try: None,
            },
        );
        self.try_post(n);
    }

    pub(super) fn post_done(&mut self, id: &SlateId) {
        self.post.outgoing.retain(|_, o| o.post.id != *id);
    }

    fn fail_post(&mut self, n: u32, why: String) {
        let Some(o) = self.post.outgoing.remove(&n) else {
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

    /// Seals post `n` to `ek` and delivers it through `session`.
    fn deliver_post(&mut self, n: u32, session: u32, ek: &[u8; mlkem::EK_LEN]) {
        let m = self.random_32();
        let Some(o) = self.post.outgoing.get_mut(&n) else {
            return;
        };
        let Some((ct, data)) = payments::seal(ek, &o.post.payload.to_bytes(), &m) else {
            return;
        };
        o.phase = Phase::Deliver {
            session,
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
        let to = o.post.to;
        match &o.phase {
            Phase::Lookup { offline } => {
                // A wallet we host: deliver straight to its session.
                if let Some(r) = self
                    .post
                    .registrations
                    .get(&to)
                    .filter(|r| r.last.elapsed() < GRACE)
                {
                    let (s, ek) = (r.session, r.ek.clone());
                    self.deliver_post(n, s, &ek);
                    return;
                }
                let ask: Vec<u32> = hosts.into_iter().filter(|s| !offline.contains(s)).collect();
                for s in ask {
                    self.send_peer(s, &Message::Lookup { address: to });
                }
            }
            Phase::Deliver {
                session, ct, data, ..
            } => {
                let msg = Message::Deliver {
                    to,
                    post: n,
                    ct: ct.clone(),
                    data: data.clone(),
                };
                let s = *session;
                // Straight to a wallet we host: no one to say forwarded.
                let direct = self.registration(&to).is_some_and(|r| r.session == s);
                self.send_peer(s, &msg);
                if direct {
                    self.on_forwarded(s, to, n);
                }
            }
        }
    }

    pub(super) fn on_address_key(
        &mut self,
        session: u32,
        address: Address,
        ek: Box<[u8; mlkem::EK_LEN]>,
    ) {
        if Address::of(&ek) != address || !mlkem::check_ek(&ek) {
            self.drop_peer(session, "sent a key that isn't the address's");
            return;
        }
        let waiting: Vec<u32> = self
            .post
            .outgoing
            .iter()
            .filter(|(_, o)| o.post.to == address && matches!(o.phase, Phase::Lookup { .. }))
            .map(|(&n, _)| n)
            .collect();
        for n in waiting {
            self.deliver_post(n, session, &ek);
        }
    }

    pub(super) fn on_offline(&mut self, session: u32, address: Address) {
        let hosts = self.host_sessions();
        let mut failed = Vec::new();
        for (&n, o) in self.post.outgoing.iter_mut() {
            if o.post.to != address {
                continue;
            }
            match &mut o.phase {
                Phase::Lookup { offline } => {
                    offline.insert(session);
                    if hosts.iter().all(|s| offline.contains(s)) {
                        failed.push(n);
                    }
                }
                // Its host lost it: look again.
                Phase::Deliver { session: s, .. } if *s == session => {
                    o.phase = Phase::Lookup {
                        offline: HashSet::from([session]),
                    };
                    o.last_try = None;
                }
                Phase::Deliver { .. } => {}
            }
        }
        for n in failed {
            self.fail_post(n, format!("{address} is offline"));
        }
    }

    pub(super) fn on_forwarded(&mut self, session: u32, to: Address, post: u32) {
        let Some(o) = self.post.outgoing.get_mut(&post) else {
            return;
        };
        if o.post.to != to {
            return;
        }
        if let Phase::Deliver {
            session: s,
            forwarded,
            ..
        } = &mut o.phase
            && *s == session
        {
            *forwarded = true;
            if !o.post.wait_for_answer {
                self.post.outgoing.remove(&post);
            }
        }
    }

    /// A host's session ended: its registrations are gone, and so is ours
    /// with it.
    pub(super) fn post_lost_session(&mut self, session: u32) {
        self.post.registrations.retain(|_, r| r.session != session);
        self.post.challenges.retain(|(s, _), _| *s != session);
        self.post.registered.remove(&session);
        for o in self.post.outgoing.values_mut() {
            if matches!(o.phase, Phase::Deliver { session: s, .. } if s == session) {
                o.phase = Phase::Lookup {
                    offline: HashSet::new(),
                };
                o.last_try = None;
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

        // Our hosts: look them up now and then, stay connected, register.
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
        for s in self.host_sessions() {
            let due = self
                .post
                .registered
                .get(&s)
                .is_none_or(|(at, _)| now - *at >= REGISTER_EVERY);
            if due {
                let confirmed = self.post.registered.get(&s).is_some_and(|r| r.1);
                self.post.registered.insert(s, (now, confirmed));
                self.send_peer(
                    s,
                    &Message::Register {
                        ek: Box::new(cfg.keys.ek),
                    },
                );
            }
        }

        // Posts: give up, or try again.
        let posts: Vec<(u32, bool, Instant, Option<Instant>)> = self
            .post
            .outgoing
            .iter()
            .map(|(&n, o)| (n, o.post.wait_for_answer, o.started, o.last_try))
            .collect();
        for (n, wait, started, last_try) in posts {
            let limit = if wait { POST_TIMEOUT } else { POST_TIMEOUT / 2 };
            if now - started > limit {
                let why = match self.post.outgoing.get(&n).map(|o| &o.phase) {
                    Some(Phase::Lookup { .. }) if self.host_sessions().is_empty() => {
                        "not connected to any host"
                    }
                    Some(Phase::Lookup { .. }) => "no host knows the address",
                    _ if wait => "no answer from the receiver",
                    _ => "couldn't deliver it",
                };
                self.fail_post(n, why.into());
            } else if last_try.is_none_or(|t| now - t >= RETRY) {
                self.try_post(n);
            }
        }
    }
}
