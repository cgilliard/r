//! The genesis block: the fixed starting point of the chain.
//!
//! Genesis is a header only, at height 0, with no block body: it creates no
//! coins ([`crate::header::Consensus::subsidy`] is `None` at height 0) and
//! its state is the empty tree. Its `body` field commits instead to a
//! recent Bitcoin block (hash and height), which shows the chain can't have
//! been started before that block existed.
//!
//! The values come from `src/genesis_data.rs`, written by the `genesis` tool
//! (`tools/genesis.rs`), which fetches the latest Bitcoin block.

use crate::babybear::BabyBear;
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::header::Header;
use crate::state;

mod data {
    include!("genesis_data.rs");
}

pub use data::{BITCOIN_HASH, BITCOIN_HEIGHT, HASH, TIMESTAMP};

/// Parses a Bitcoin block hash as displayed (64 hex digits).
pub fn parse_hash(hex: &str) -> Option<[u8; 32]> {
    let hex = hex.trim();
    if hex.len() != 64 || !hex.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// The genesis `body`: `sponge(GENESIS, [hash as 16-bit limbs, height])`,
/// over the hash's bytes in display order.
pub fn body(bitcoin_hash: &[u8; 32], bitcoin_height: u64) -> Hash {
    let mut elems: Vec<BabyBear> = bitcoin_hash
        .chunks_exact(2)
        .map(|b| BabyBear::new(u16::from_be_bytes([b[0], b[1]]) as u32))
        .collect();
    let (lo, hi) = state::split(bitcoin_height.min(state::CAPACITY));
    elems.extend([lo, hi]);
    hash::sponge(domain::GENESIS, &[], &[&elems])
}

/// The genesis header committing to `bitcoin_hash` at `bitcoin_height`,
/// made at `timestamp`.
pub fn build(bitcoin_hash: &[u8; 32], bitcoin_height: u64, timestamp: u64) -> Header {
    Header {
        height: 0,
        prev: [BabyBear::ZERO; HASH_LEN],
        timestamp,
        state: state::empty_state(),
        body: body(bitcoin_hash, bitcoin_height),
        aux_hash: [0; crate::header::AUX_HASH_LEN],
        nonce: 0,
    }
}

/// The chain's genesis header.
pub fn header() -> Header {
    let bitcoin_hash = parse_hash(BITCOIN_HASH).expect("genesis data: bad Bitcoin hash");
    build(&bitcoin_hash, BITCOIN_HEIGHT, TIMESTAMP)
}

/// Hex of a header hash's bytes.
pub fn hex(h: &Hash) -> String {
    hash::to_bytes(h)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Consensus;

    #[test]
    fn genesis_matches_its_data() {
        let g = header();
        assert_eq!(g.height, 0);
        assert_eq!(g.state, state::empty_state());
        assert_eq!(g.timestamp, Consensus::DEFAULT.genesis_time);
        if !HASH.is_empty() {
            assert_eq!(hex(&g.hash()), HASH);
        }
    }

    #[test]
    fn genesis_creates_nothing() {
        let c = Consensus::DEFAULT;
        assert_eq!(c.subsidy(0), None);
        assert_eq!(header().state.size, 1, "only the sentinel");
    }

    #[test]
    fn body_commits_to_the_bitcoin_block() {
        let a = parse_hash(&"ab".repeat(32)).unwrap();
        let mut b = a;
        b[31] ^= 1;
        assert_ne!(body(&a, 5), body(&b, 5));
        assert_ne!(body(&a, 5), body(&a, 6));
        assert_eq!(parse_hash("zz"), None);
        assert_eq!(parse_hash(&"g".repeat(64)), None);
    }
}
