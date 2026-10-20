//! Wallets: one-time addresses, output nonces that carry the amount, and
//! recovery from a seed.
//!
//! A wallet derives everything from a seed and an address index: the WOTS
//! key (one-time, so each address should receive once) and a 16-byte view
//! secret. An [`Address`] is the key's hash plus the view secret; whoever
//! pays it learns both, and nobody else learns either.
//!
//! The payer builds the output's 16-byte nonce from the view secret:
//!
//! - bytes `0..8`: a tag, `H(view, NONCE_TAG)`, which lets the wallet spot
//!   its outputs with one table lookup each;
//! - bytes `8..16`: the amount (little-endian) XORed with
//!   `H(view, NONCE_MASK)`.
//!
//! Without the view secret both halves look random. The nonce is part of
//! the output's commitment and stored in the state tree, so a wallet
//! restored from its seed recovers every unspent output it owns, with its
//! amount and position, from the tree alone ([`Wallet::scan`]): it derives
//! the tags of its address range, looks each output's tag up, decrypts the
//! amount, and confirms it by recomputing the commitment.

use std::collections::HashMap;
use std::ops::Range;

use crate::babybear::{BabyBear, P};
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::output::{self, Amount, NONCE_LEN, Nonce, Output, OutputHash, OutputRecord};
use crate::transaction::{Input, pubkey_hash};
use crate::wots::{self, PublicKey, SecretKey};

/// Bytes of a view secret: four field elements.
pub const VIEW_LEN: usize = 16;
const VIEW_ELEMS: usize = VIEW_LEN / 4;
/// Bytes of the tag, and of the encrypted amount, in a nonce.
const TAG_LEN: usize = 8;

/// Where to send coins: a one-time key hash and its view secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address {
    pub hash: OutputHash,
    pub view: [u8; VIEW_LEN],
}

/// Bytes in a serialized address.
pub const ADDRESS_LEN: usize = output::OUTPUT_HASH_LEN + VIEW_LEN;

impl Address {
    pub fn to_bytes(&self) -> [u8; ADDRESS_LEN] {
        let mut out = [0u8; ADDRESS_LEN];
        out[..output::OUTPUT_HASH_LEN].copy_from_slice(&self.hash);
        out[output::OUTPUT_HASH_LEN..].copy_from_slice(&self.view);
        out
    }

    /// Parses [`Address::to_bytes`] output; `None` unless every field is
    /// canonical.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != ADDRESS_LEN {
            return None;
        }
        let (h, v) = bytes.split_at(output::OUTPUT_HASH_LEN);
        hash::from_bytes(h.try_into().ok()?)?;
        view_elems(v.try_into().ok()?)?;
        Some(Self {
            hash: h.try_into().ok()?,
            view: v.try_into().ok()?,
        })
    }

    /// An output paying `amount` to this address, with its nonce built from
    /// the view secret.
    ///
    /// # Panics
    /// If the view secret isn't canonical (an address from
    /// [`Address::from_bytes`] or [`Wallet::address`] always is).
    pub fn pay(&self, amount: Amount) -> Output {
        Output {
            hash: self.hash,
            amount,
            nonce: nonce_for(&self.view, amount),
        }
    }
}

fn view_elems(view: &[u8; VIEW_LEN]) -> Option<[BabyBear; VIEW_ELEMS]> {
    let mut out = [BabyBear::ZERO; VIEW_ELEMS];
    for (x, chunk) in out.iter_mut().zip(view.chunks_exact(4)) {
        let v = u32::from_le_bytes(chunk.try_into().expect("4 bytes"));
        if v >= P {
            return None;
        }
        *x = BabyBear::new(v);
    }
    Some(out)
}

/// Eight bytes derived from a view secret under `dom`: the low 16 bits of
/// four hash elements (each element is uniform mod `p`, so these bits are
/// uniform up to a negligible bias).
fn derive8(view: &[u8; VIEW_LEN], dom: u32) -> [u8; TAG_LEN] {
    let v = view_elems(view).expect("canonical view secret");
    let h = hash::sponge(dom, &[], &[&v]);
    let mut out = [0u8; TAG_LEN];
    for (chunk, x) in out.chunks_exact_mut(2).zip(&h) {
        chunk.copy_from_slice(&(x.as_u32() as u16).to_le_bytes());
    }
    out
}

/// The nonce for paying `amount` to an address with view secret `view`.
pub fn nonce_for(view: &[u8; VIEW_LEN], amount: Amount) -> Nonce {
    let mut nonce = [0u8; NONCE_LEN];
    nonce[..TAG_LEN].copy_from_slice(&derive8(view, domain::NONCE_TAG));
    let mask = u64::from_le_bytes(derive8(view, domain::NONCE_MASK));
    nonce[TAG_LEN..].copy_from_slice(&(amount ^ mask).to_le_bytes());
    nonce
}

/// The amount in a nonce built for view secret `view`.
pub fn amount_in(view: &[u8; VIEW_LEN], nonce: &Nonce) -> Amount {
    let mask = u64::from_le_bytes(derive8(view, domain::NONCE_MASK));
    u64::from_le_bytes(nonce[TAG_LEN..].try_into().expect("8 bytes")) ^ mask
}

/// An unspent output the wallet owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coin {
    /// The address index it was paid to.
    pub index: u32,
    /// Its slot in the state tree.
    pub position: u64,
    pub amount: Amount,
    pub record: OutputRecord,
}

/// Keys and addresses derived from a seed.
#[derive(Clone, Copy, Debug)]
pub struct Wallet {
    seed: Hash,
}

impl Wallet {
    pub fn from_seed(seed: Hash) -> Self {
        Self { seed }
    }

    fn derive(&self, dom: u32, index: u32) -> Hash {
        hash::sponge(dom, &[], &[&self.seed, &[BabyBear::new(index)]])
    }

    /// The one-time key of address `index`.
    pub fn key(&self, index: u32) -> (SecretKey, PublicKey) {
        wots::keygen(self.derive(domain::WALLET_KEY, index))
    }

    /// The view secret of address `index`.
    pub fn view(&self, index: u32) -> [u8; VIEW_LEN] {
        let h = self.derive(domain::WALLET_VIEW, index);
        let mut view = [0u8; VIEW_LEN];
        for (chunk, x) in view.chunks_exact_mut(4).zip(&h[..VIEW_ELEMS]) {
            chunk.copy_from_slice(&x.as_u32().to_le_bytes());
        }
        view
    }

    pub fn address(&self, index: u32) -> Address {
        Address {
            hash: pubkey_hash(&self.key(index).1),
            view: self.view(index),
        }
    }

    /// The wallet's coins among `outputs` (slot and record, e.g. from
    /// [`crate::state::StateTree::outputs`]), looking at addresses
    /// `indices`.
    pub fn scan(&self, outputs: &[(u64, OutputRecord)], indices: Range<u32>) -> Vec<Coin> {
        let views: HashMap<[u8; TAG_LEN], (u32, [u8; VIEW_LEN])> = indices
            .map(|i| {
                let view = self.view(i);
                (derive8(&view, domain::NONCE_TAG), (i, view))
            })
            .collect();
        let mut hashes: HashMap<u32, OutputHash> = HashMap::new();
        let mut coins = Vec::new();
        for &(position, record) in outputs {
            let tag: [u8; TAG_LEN] = record.nonce[..TAG_LEN].try_into().expect("tag bytes");
            let Some(&(index, view)) = views.get(&tag) else {
                continue;
            };
            let amount = amount_in(&view, &record.nonce);
            let hash = *hashes
                .entry(index)
                .or_insert_with(|| pubkey_hash(&self.key(index).1));
            if output::commit(&hash, amount, &record.nonce) == Some(record.commitment) {
                coins.push(Coin {
                    index,
                    position,
                    amount,
                    record,
                });
            }
        }
        coins
    }

    /// The input spending `coin`, with the key that signs it.
    pub fn input(&self, coin: &Coin) -> (SecretKey, Input) {
        let (sk, pubkey) = self.key(coin.index);
        let input = Input {
            commitment: coin.record.commitment,
            position: coin.position,
            pubkey,
            amount: coin.amount,
            nonce: coin.record.nonce,
        };
        (sk, input)
    }
}

const _: () = assert!(HASH_LEN >= VIEW_ELEMS);

#[cfg(test)]
mod tests {
    use super::*;

    fn wallet(n: u32) -> Wallet {
        Wallet::from_seed(BabyBear::new_array(core::array::from_fn(|i| n * 100 + i as u32)))
    }

    fn record(o: &Output) -> OutputRecord {
        o.record().unwrap()
    }

    #[test]
    fn nonces_carry_amounts() {
        let w = wallet(1);
        for amount in [0, 1, 50, u64::MAX] {
            let view = w.view(3);
            let nonce = nonce_for(&view, amount);
            assert_eq!(amount_in(&view, &nonce), amount);
            // Another view secret reads something else.
            assert_ne!(amount_in(&w.view(4), &nonce), amount);
        }
        // Different addresses get different tags.
        assert_ne!(
            nonce_for(&w.view(1), 5)[..TAG_LEN],
            nonce_for(&w.view(2), 5)[..TAG_LEN]
        );
    }

    #[test]
    fn address_bytes_round_trip() {
        let a = wallet(1).address(7);
        assert_eq!(Address::from_bytes(&a.to_bytes()), Some(a));
        let mut bad = a.to_bytes();
        bad[output::OUTPUT_HASH_LEN..output::OUTPUT_HASH_LEN + 4]
            .copy_from_slice(&P.to_le_bytes());
        assert_eq!(Address::from_bytes(&bad), None);
        assert_eq!(Address::from_bytes(&bad[1..]), None);
    }

    #[test]
    fn scan_finds_own_outputs_with_amounts() {
        let (alice, bob) = (wallet(1), wallet(2));
        let outputs: Vec<(u64, OutputRecord)> = [
            alice.address(0).pay(10),
            bob.address(0).pay(20),
            alice.address(5).pay(30),
            bob.address(1).pay(40),
            alice.address(50).pay(60),
        ]
        .iter()
        .enumerate()
        .map(|(i, o)| (i as u64 + 1, record(o)))
        .collect();
        let coins = alice.scan(&outputs, 0..10);
        let found: Vec<(u32, u64, Amount)> =
            coins.iter().map(|c| (c.index, c.position, c.amount)).collect();
        assert_eq!(found, [(0, 1, 10), (5, 3, 30)]);
        // A wider range finds the far address too.
        assert_eq!(alice.scan(&outputs, 0..100).len(), 3);
        // Bob sees only his.
        let bobs: Vec<Amount> = bob.scan(&outputs, 0..10).iter().map(|c| c.amount).collect();
        assert_eq!(bobs, [20, 40]);
    }

    #[test]
    fn scan_rejects_a_matching_tag_with_the_wrong_commitment() {
        // Someone who knows an address can copy its tag, but can't make the
        // commitment check pass for another key.
        let alice = wallet(1);
        let mut fake = wallet(9).address(0).pay(10);
        fake.nonce = nonce_for(&alice.view(0), 10);
        let outputs = [(1, record(&fake))];
        assert!(alice.scan(&outputs, 0..5).is_empty());
    }

    #[test]
    fn coins_spend_with_their_position() {
        let w = wallet(3);
        let coin = Coin {
            index: 2,
            position: 77,
            amount: 9,
            record: record(&w.address(2).pay(9)),
        };
        let (_, input) = w.input(&coin);
        assert!(input.opens_commitment());
        assert_eq!(input.position, 77);
    }
}
