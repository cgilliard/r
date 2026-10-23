//! Layer 2's relay network: a node's relay identity, and the relay
//! consensus that says how wallets reach each other.
//!
//! **Identity.** A node's relay key is an ML-DSA-44 key pair
//! ([`crate::crypto::mldsa`]) derived from its wallet's recovery entropy,
//! so restoring the wallet restores the key. Relays are named by the
//! SHA-256 of the public key ([`key_hash`]).
//!
//! **The consensus file.** A team (like Tor's directory authorities)
//! publishes a text file and gives it to miners, who commit to its SHA-256
//! in aux leaf [`CONSENSUS_LEAF`] of the blocks they mine. Its first word
//! is the mode: `init` (no relays yet: wallets pass slates through the seed
//! nodes) or `onion`, followed by relays, each an `ip:port` and a key hash
//! (64 hex digits), in either order. Words are separated by any whitespace,
//! so it's easy to edit by hand; relays may come in any order.
//!
//! **Which file.** Nodes fetch the files that recent blocks commit to
//! ([`crate::preimages`]). The one in force is the one most of the
//! [`WINDOW`] blocks ending [`LAG`] below the tip commit to (a valid file
//! that a node could fetch; ties go to the most recent), so the mode
//! doesn't flip with each block or a short reorg. With none, it's `init`.
//! One block is enough: if a miner commits to a bad file, the others are
//! expected to commit to a better one.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

use crate::aux_leaves::Leaf;
use crate::crypto::{keccak, mldsa};
use crate::ledger::Ledger;
use crate::mnemonic::{ENTROPY_LEN, sha256};
use crate::preimages::PreimageStore;

/// The aux leaf that holds the consensus file's hash.
pub const CONSENSUS_LEAF: usize = 0;
/// Blocks whose votes count.
pub const WINDOW: u64 = 100;
/// The newest blocks, left out of the count.
pub const LAG: u64 = 10;

/// The seed of a wallet's relay key.
fn identity_seed(entropy: &[u8; ENTROPY_LEN]) -> [u8; 32] {
    let mut seed = [0u8; 32];
    keccak::shake256(&[b"r relay identity (ML-DSA-44)", entropy], &mut seed);
    seed
}

/// The relay key pair of the wallet with recovery `entropy`: (public
/// key, secret key).
pub fn identity(entropy: &[u8; ENTROPY_LEN]) -> ([u8; mldsa::PK_LEN], [u8; mldsa::SK_LEN]) {
    mldsa::keygen(&identity_seed(entropy))
}

/// What names a relay: the SHA-256 of its public key.
pub fn key_hash(pk: &[u8; mldsa::PK_LEN]) -> Leaf {
    sha256(pk)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Relay {
    pub addr: SocketAddr,
    pub key_hash: Leaf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// No relays: slates go through the seed nodes.
    Init,
    /// Onion routing over these relays (sorted, at least one).
    Onion(Vec<Relay>),
}

fn parse_hash(word: &str) -> Option<Leaf> {
    if word.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(word.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Reads a consensus file.
pub fn parse(bytes: &[u8]) -> Result<Mode, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "not UTF-8 text")?;
    let mut words = text.split_whitespace();
    let mode = words.next().ok_or("empty")?;
    let words: Vec<&str> = words.collect();
    if mode.eq_ignore_ascii_case("init") {
        return match words.first() {
            None => Ok(Mode::Init),
            Some(w) => Err(format!("\"{w}\" after init (relays are for onion mode)")),
        };
    }
    if !mode.eq_ignore_ascii_case("onion") {
        return Err(format!("the first word is \"{mode}\", not init or onion"));
    }
    if words.len() % 2 == 1 {
        return Err(format!(
            "\"{}\" has no partner: each relay is an ip:port and a key hash",
            words[words.len() - 1]
        ));
    }
    let mut relays = Vec::new();
    let (mut addrs, mut keys) = (HashSet::new(), HashSet::new());
    for pair in words.chunks_exact(2) {
        let relay = match (pair[0].parse(), parse_hash(pair[1])) {
            (Ok(addr), Some(key_hash)) => Relay { addr, key_hash },
            _ => match (parse_hash(pair[0]), pair[1].parse()) {
                (Some(key_hash), Ok(addr)) => Relay { addr, key_hash },
                _ => {
                    return Err(format!(
                        "\"{} {}\" isn't an ip:port and a key hash (64 hex digits)",
                        pair[0], pair[1]
                    ));
                }
            },
        };
        if !addrs.insert(relay.addr) {
            return Err(format!("{} is listed twice", relay.addr));
        }
        if !keys.insert(relay.key_hash) {
            return Err(format!(
                "key {} is listed twice",
                crate::wallet::hex(&relay.key_hash)
            ));
        }
        relays.push(relay);
    }
    if relays.is_empty() {
        return Err("onion mode with no relays".into());
    }
    relays.sort();
    Ok(Mode::Onion(relays))
}

/// The file with the most `votes` (oldest first), ties to the most recent
/// vote: its hash and how many.
pub fn choose(votes: &[Leaf]) -> Option<(Leaf, u64)> {
    let mut counts: HashMap<Leaf, (u64, usize)> = HashMap::new();
    for (i, v) in votes.iter().enumerate() {
        let c = counts.entry(*v).or_default();
        *c = (c.0 + 1, i);
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(h, (n, _))| (h, n))
}

/// The relay consensus in force.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Consensus {
    pub mode: Mode,
    /// The file's hash and its votes (`None`: no block voted; `init`).
    pub file: Option<(Leaf, u64)>,
    /// The heights whose votes counted (empty near genesis).
    pub heights: std::ops::RangeInclusive<u64>,
}

/// The heights whose votes count when the tip is at `tip`.
pub fn window(tip: u64) -> std::ops::RangeInclusive<u64> {
    let last = tip.saturating_sub(LAG);
    last.saturating_sub(WINDOW - 1).max(1)..=last
}

/// The relay consensus on `ledger`'s main chain, from the files in
/// `store`.
pub fn current(ledger: &Ledger, store: &PreimageStore) -> Result<Consensus, String> {
    let (tip, _) = ledger.tip().map_err(|e| e.to_string())?;
    let heights = window(tip.height);
    // The roots first: the store reads in its own transactions.
    let roots: Vec<Leaf> = {
        let txn = ledger.env.read_txn().map_err(|e| e.to_string())?;
        let mut roots = Vec::new();
        for h in heights.clone() {
            if let Some(header) = ledger.store.header(&txn, h).map_err(|e| e.to_string())? {
                roots.push(header.aux_hash);
            }
        }
        roots
    };
    let mut votes = Vec::new();
    let mut modes = HashMap::new();
    for root in roots.iter().filter(|r| **r != [0; 32]) {
        let Some(file) = store
            .leaf(root, CONSENSUS_LEAF)
            .map_err(|e| e.to_string())?
            .filter(|f| *f != [0; 32])
        else {
            continue;
        };
        let mode = match modes.entry(file) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(
                store
                    .get(&file)
                    .map_err(|e| e.to_string())?
                    .and_then(|b| parse(&b).ok()),
            ),
        };
        if mode.is_some() {
            votes.push(file);
        }
    }
    let file = choose(&votes);
    let mode = match file {
        Some((h, _)) => modes[&h].clone().expect("a valid file"),
        None => Mode::Init,
    };
    Ok(Consensus {
        mode,
        file,
        heights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    #[test]
    fn files_are_read_loosely_but_checked() {
        assert_eq!(parse(b"init"), Ok(Mode::Init));
        assert_eq!(parse(b"  INIT \r\n"), Ok(Mode::Init));
        let text = format!(
            "onion\n  1.2.3.4:5000   {KEY}\n{}\t[::1]:9\n",
            KEY.replace('0', "f")
        );
        let Ok(Mode::Onion(relays)) = parse(text.as_bytes()) else {
            panic!("{:?}", parse(text.as_bytes()));
        };
        assert_eq!(relays.len(), 2);
        assert_eq!(relays[0].addr, "1.2.3.4:5000".parse().unwrap());
        assert_eq!(relays[0].key_hash, parse_hash(KEY).unwrap());
        assert_eq!(relays[1].addr, "[::1]:9".parse().unwrap());
        // Order doesn't matter: the same relays.
        let swapped = format!("onion {} [::1]:9 1.2.3.4:5000 {KEY}", KEY.replace('0', "f"));
        assert_eq!(parse(swapped.as_bytes()), Ok(Mode::Onion(relays)));
        for bad in [
            String::new(),
            "tor".into(),
            format!("init 1.2.3.4:5 {KEY}"),
            "onion".into(),
            "onion 1.2.3.4:5".into(),
            format!("onion example.com:5 {KEY}"),
            format!("onion 1.2.3.4:5 {}", &KEY[2..]),
            format!("onion 1.2.3.4:5 {KEY} 1.2.3.4:5 {}", KEY.replace('0', "f")),
            format!("onion 1.2.3.4:5 {KEY} 1.2.3.4:6 {KEY}"),
        ] {
            assert!(parse(bad.as_bytes()).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_most_voted_file_wins_and_ties_go_to_the_latest() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        assert_eq!(choose(&[]), None);
        assert_eq!(choose(&[a]), Some((a, 1)));
        assert_eq!(choose(&[a, b, a]), Some((a, 2)));
        assert_eq!(choose(&[a, b]), Some((b, 1)));
        assert_eq!(choose(&[b, a, a, b]), Some((b, 2)));
        assert!(window(5).is_empty());
        assert_eq!(window(11), 1..=1);
        assert_eq!(window(500), 391..=490);
    }

    #[test]
    fn the_relay_key_comes_from_the_recovery_entropy() {
        let (pk, _) = identity(&[1; ENTROPY_LEN]);
        assert_eq!(identity(&[1; ENTROPY_LEN]).0, pk);
        assert_ne!(identity(&[2; ENTROPY_LEN]).0, pk);
        assert_eq!(key_hash(&pk), sha256(&pk));
    }
}
