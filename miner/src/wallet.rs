//! Wallets: one-time keys, output nonces that carry the amount, and
//! recovery from a seed.
//!
//! A wallet derives everything from a seed and a key index: the WOTS key
//! (one-time: each index takes one output, and its key signs once) and a
//! 16-byte view secret. A [`OneTimeKey`] is the key's hash plus the view
//! secret: what an output is made from. Nothing is ever published for others
//! to pay to; the receiver makes its own output ([`Response`]).
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
//! the tags of its key indices, looks each output's tag up, decrypts the
//! amount, and confirms it by recomputing the commitment.
//!
//! **Paying.** Payments are interactive, through files (slates), as in
//! Grin. The sender picks coins and writes a [`Slate`] (an id, the amount
//! and fee); the receiver makes an output for the amount with a fresh key
//! of its own and writes it back in a [`Response`]; the sender then signs
//! the transaction (its coins in, the receiver's output and its change out)
//! and submits it. Every output, received, change or reward, gets a fresh
//! key index, so each one-time key signs exactly once.
//!
//! The response repeats the slate (id, amount and fee), and the sender
//! finalizes only a response to a slate it sent, unchanged. The
//! [`WalletStore`] keeps the seed, the next index
//! and the sender's side of each payment ([`Sent`]) in LMDB.

use std::collections::HashMap;
use std::ops::Range;

use heed::types::Bytes;
use heed::{Database, Env, RoTxn, RwTxn};

use crate::babybear::{BabyBear, P};
use crate::hash::{self, HASH_LEN, Hash, domain};
use crate::output::{
    self, Amount, COMMITMENT_LEN, Commitment, NONCE_LEN, Nonce, OUTPUT_HASH_LEN, Output,
    OutputHash, OutputRecord,
};
use crate::transaction::{Input, pubkey_hash};
use crate::wots::{self, PublicKey, SecretKey};

/// Bytes of a view secret: four field elements.
pub const VIEW_LEN: usize = 16;
const VIEW_ELEMS: usize = VIEW_LEN / 4;
/// Bytes of the tag, and of the encrypted amount, in a nonce.
const TAG_LEN: usize = 8;

/// The tag in an output's nonce, which names the key index it's for.
pub type Tag = [u8; TAG_LEN];

/// The tag of `record`'s nonce.
pub fn tag_of(record: &OutputRecord) -> Tag {
    record.nonce[..TAG_LEN].try_into().expect("tag bytes")
}

/// What an output is made from: a one-time key's hash and its view secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneTimeKey {
    pub hash: OutputHash,
    pub view: [u8; VIEW_LEN],
}

impl OneTimeKey {
    /// An output paying `amount` to this key, with its nonce built from
    /// the view secret.
    ///
    /// # Panics
    /// If the view secret isn't canonical (one from [`Wallet::key_for`]
    /// always is).
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

/// The nonce for paying `amount` to a key with view secret `view`.
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
    /// The key index it was paid to.
    pub index: u32,
    /// Its slot in the state tree.
    pub position: u64,
    pub amount: Amount,
    pub record: OutputRecord,
}

/// Keys derived from a seed.
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

    /// The one-time key pair at `index`.
    pub fn key(&self, index: u32) -> (SecretKey, PublicKey) {
        wots::keygen(self.derive(domain::WALLET_KEY, index))
    }

    /// The view secret at `index`.
    pub fn view(&self, index: u32) -> [u8; VIEW_LEN] {
        let h = self.derive(domain::WALLET_VIEW, index);
        let mut view = [0u8; VIEW_LEN];
        for (chunk, x) in view.chunks_exact_mut(4).zip(&h[..VIEW_ELEMS]) {
            chunk.copy_from_slice(&x.as_u32().to_le_bytes());
        }
        view
    }

    /// What an output for key `index` is made from.
    pub fn key_for(&self, index: u32) -> OneTimeKey {
        OneTimeKey {
            hash: pubkey_hash(&self.key(index).1),
            view: self.view(index),
        }
    }

    /// The wallet's coins among `outputs` (slot and record, e.g. from
    /// [`crate::state::StateTree::outputs`]), looking at key indices
    /// `indices`.
    pub fn scan(&self, outputs: &[(u64, OutputRecord)], indices: Range<u32>) -> Vec<Coin> {
        let tags = self.tags(indices);
        outputs
            .iter()
            .filter_map(|&(position, record)| {
                let index = *tags.get(&tag_of(&record))?;
                Some(Coin {
                    index,
                    position,
                    amount: self.identify(index, &record)?,
                    record,
                })
            })
            .collect()
    }

    /// The nonce tag of key `index`.
    pub fn tag(&self, index: u32) -> Tag {
        derive8(&self.view(index), domain::NONCE_TAG)
    }

    /// The nonce tags of key indices `indices`, to look outputs up by.
    pub fn tags(&self, indices: Range<u32>) -> HashMap<Tag, u32> {
        indices
            .map(|i| (derive8(&self.view(i), domain::NONCE_TAG), i))
            .collect()
    }

    /// The amount of `record` if it was paid to key `index` (its tag
    /// matched): decrypts the amount and checks the commitment.
    pub fn identify(&self, index: u32, record: &OutputRecord) -> Option<Amount> {
        let amount = amount_in(&self.view(index), &record.nonce);
        let hash = pubkey_hash(&self.key(index).1);
        (output::commit(&hash, amount, &record.nonce) == Some(record.commitment)).then_some(amount)
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

/// A payment's id: 16 random bytes.
pub type SlateId = [u8; 16];

/// What the sender gives the receiver: the payment's id, amount and fee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slate {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
}

/// What the receiver gives back: the slate as it got it, and its output for
/// the amount.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Response {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
    pub output: Output,
}

const SLATE_HEADER: &str = "r slate 2";
const RESPONSE_HEADER: &str = "r slate response 2";

impl Slate {
    /// The slate file: a header line, then `key value` lines.
    pub fn to_text(&self) -> String {
        format!(
            "{SLATE_HEADER}\nid {}\namount {}\nfee {}\n",
            hex(&self.id),
            self.amount,
            self.fee,
        )
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let f = Fields::parse(text, SLATE_HEADER)?;
        Ok(Self {
            id: f.id()?,
            amount: f.number("amount")?,
            fee: f.number("fee")?,
        })
    }
}

impl Response {
    pub fn to_text(&self) -> String {
        let mut out = self.output.hash.to_vec();
        out.extend(self.output.amount.to_le_bytes());
        out.extend(self.output.nonce);
        format!(
            "{RESPONSE_HEADER}\nid {}\namount {}\nfee {}\noutput {}\n",
            hex(&self.id),
            self.amount,
            self.fee,
            hex(&out),
        )
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let f = Fields::parse(text, RESPONSE_HEADER)?;
        let bytes = unhex(f.get("output")?).ok_or("bad output")?;
        if bytes.len() != OUTPUT_HASH_LEN + 8 + NONCE_LEN {
            return Err("bad output".into());
        }
        let (hash, rest) = bytes.split_at(OUTPUT_HASH_LEN);
        let (amount, nonce) = rest.split_at(8);
        let output = Output {
            hash: hash.try_into().expect("length"),
            amount: u64::from_le_bytes(amount.try_into().expect("length")),
            nonce: nonce.try_into().expect("length"),
        };
        if output.record().is_none() {
            return Err("bad output".into());
        }
        Ok(Self {
            id: f.id()?,
            amount: f.number("amount")?,
            fee: f.number("fee")?,
            output,
        })
    }
}

/// A slate file's `key value` lines.
struct Fields<'a>(HashMap<&'a str, &'a str>);

impl<'a> Fields<'a> {
    fn parse(text: &'a str, header: &str) -> Result<Self, String> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        if lines.next() != Some(header) {
            return Err(format!("not a slate file (expected \"{header}\" first)"));
        }
        let mut fields = HashMap::new();
        for line in lines {
            let (k, v) = line.split_once(' ').ok_or(format!("bad line \"{line}\""))?;
            if fields.insert(k, v.trim()).is_some() {
                return Err(format!("\"{k}\" appears twice"));
            }
        }
        Ok(Self(fields))
    }

    fn get(&self, key: &str) -> Result<&'a str, String> {
        self.0.get(key).copied().ok_or(format!("no \"{key}\""))
    }

    fn number(&self, key: &str) -> Result<Amount, String> {
        self.get(key)?.parse().map_err(|_| format!("bad \"{key}\""))
    }

    fn id(&self) -> Result<SlateId, String> {
        unhex(self.get("id")?)
            .and_then(|b| b.try_into().ok())
            .ok_or("bad id".into())
    }
}

/// Lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parses [`hex`] output (either case).
pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// The sender's side of a payment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sent {
    pub id: SlateId,
    pub amount: Amount,
    pub fee: Amount,
    /// The coins it spends (locked until it's cancelled or confirmed).
    pub coins: Vec<Coin>,
    /// The change output's key index and amount, if any.
    pub change: Option<(u32, Amount)>,
    /// When it was made (Unix seconds).
    pub created: u64,
    /// The signed transaction, once finalized: its keys are then used, so it
    /// can only be resubmitted, never cancelled.
    pub tx: Option<Vec<u8>>,
}

/// The receiver's side of a payment: the output it made, until that shows
/// up in the state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Received {
    pub id: SlateId,
    /// The output's key index, amount and commitment.
    pub index: u32,
    pub amount: Amount,
    pub commitment: Commitment,
    /// When it was made (Unix seconds).
    pub created: u64,
}

impl Received {
    fn to_bytes(self) -> Vec<u8> {
        let mut out = self.id.to_vec();
        out.extend(self.index.to_le_bytes());
        out.extend(self.amount.to_le_bytes());
        out.extend(self.commitment.0);
        out.extend(self.created.to_le_bytes());
        out
    }

    fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != 16 + 4 + 8 + COMMITMENT_LEN + 8 {
            return None;
        }
        let at = 28 + COMMITMENT_LEN;
        Some(Self {
            id: b[..16].try_into().ok()?,
            index: u32::from_le_bytes(b[16..20].try_into().ok()?),
            amount: u64::from_le_bytes(b[20..28].try_into().ok()?),
            commitment: Commitment(b[28..at].try_into().ok()?),
            created: u64::from_le_bytes(b[at..].try_into().ok()?),
        })
    }
}

/// Bytes in an encoded coin: index, position, amount, commitment, nonce.
const COIN_BYTES: usize = 4 + 8 + 8 + COMMITMENT_LEN + NONCE_LEN;

impl Sent {
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = self.id.to_vec();
        out.extend(self.amount.to_le_bytes());
        out.extend(self.fee.to_le_bytes());
        out.extend(self.created.to_le_bytes());
        out.extend((self.coins.len() as u16).to_le_bytes());
        for c in &self.coins {
            out.extend(c.index.to_le_bytes());
            out.extend(c.position.to_le_bytes());
            out.extend(c.amount.to_le_bytes());
            out.extend(c.record.commitment.0);
            out.extend(c.record.nonce);
        }
        match self.change {
            Some((index, amount)) => {
                out.push(1);
                out.extend(index.to_le_bytes());
                out.extend(amount.to_le_bytes());
            }
            None => out.push(0),
        }
        if let Some(tx) = &self.tx {
            out.extend(tx);
        }
        out
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut r = bytes;
        let mut take = |n: usize| -> Option<&[u8]> {
            let (a, b) = r.split_at_checked(n)?;
            r = b;
            Some(a)
        };
        let u64_of = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
        let id = take(16)?.try_into().ok()?;
        let amount = u64_of(take(8)?);
        let fee = u64_of(take(8)?);
        let created = u64_of(take(8)?);
        let n = u16::from_le_bytes(take(2)?.try_into().ok()?) as usize;
        let mut coins = Vec::with_capacity(n);
        for _ in 0..n {
            let c = take(COIN_BYTES)?;
            coins.push(Coin {
                index: u32::from_le_bytes(c[..4].try_into().ok()?),
                position: u64_of(&c[4..12]),
                amount: u64_of(&c[12..20]),
                record: OutputRecord {
                    commitment: Commitment(c[20..20 + COMMITMENT_LEN].try_into().ok()?),
                    nonce: c[20 + COMMITMENT_LEN..].try_into().ok()?,
                },
            });
        }
        let change = match take(1)?[0] {
            0 => None,
            1 => Some((
                u32::from_le_bytes(take(4)?.try_into().ok()?),
                u64_of(take(8)?),
            )),
            _ => return None,
        };
        let tx = (!r.is_empty()).then(|| r.to_vec());
        Some(Self {
            id,
            amount,
            fee,
            coins,
            change,
            created,
            tx,
        })
    }
}

/// One of the wallet's outputs, as found in a block (or the state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owned {
    pub index: u32,
    pub amount: Amount,
    pub record: OutputRecord,
    /// The block that made it (for one found in the state, the tip then).
    pub height: u64,
    /// A mining reward of ours (which matures before it's spent).
    pub reward: bool,
    /// The height at which it was first seen gone from the state (spent),
    /// or 0.
    pub gone_since: u64,
}

const OWNED_BYTES: usize = 4 + 8 + NONCE_LEN + 8 + 1 + 8;

impl Owned {
    fn to_bytes(self) -> Vec<u8> {
        let mut out = self.index.to_le_bytes().to_vec();
        out.extend(self.amount.to_le_bytes());
        out.extend(self.record.nonce);
        out.extend(self.height.to_le_bytes());
        out.push(self.reward as u8);
        out.extend(self.gone_since.to_le_bytes());
        out
    }

    fn from_bytes(commitment: Commitment, b: &[u8]) -> Option<Self> {
        if b.len() != OWNED_BYTES {
            return None;
        }
        let at = 12 + NONCE_LEN;
        Some(Self {
            index: u32::from_le_bytes(b[..4].try_into().ok()?),
            amount: u64::from_le_bytes(b[4..12].try_into().ok()?),
            record: OutputRecord {
                commitment,
                nonce: b[12..at].try_into().ok()?,
            },
            height: u64::from_le_bytes(b[at..at + 8].try_into().ok()?),
            reward: b[at + 8] == 1,
            gone_since: u64::from_le_bytes(b[at + 9..].try_into().ok()?),
        })
    }
}

/// The node's wallet in LMDB (database `wallet`):
/// - `entropy`: the 32 bytes behind the recovery words ([`crate::mnemonic`]);
/// - `next`: the next unused key index;
/// - `sent/<id>`, `recv/<id>`: payments being sent and received;
/// - `out/<commitment>`: the wallet's outputs ([`Owned`]);
/// - `synced`: the height its outputs are found up to, and `blk/<height>`
///   the hash of each block scanned (to notice a reorg);
/// - `reward/<index>`: key indices our mining rewards went to;
/// - `restoring`: set while a restored wallet looks for its outputs.
#[derive(Clone, Copy, Debug)]
pub struct WalletStore {
    db: Database<Bytes, Bytes>,
}

const ENTROPY_KEY: &[u8] = b"entropy";
const SYNCED_KEY: &[u8] = b"synced";
const RESTORING_KEY: &[u8] = b"restoring";
const OUT_PREFIX: &[u8] = b"out/";
const BLK_PREFIX: &[u8] = b"blk/";
const REWARD_PREFIX: &[u8] = b"reward/";
const NEXT_KEY: &[u8] = b"next";
const SENT_PREFIX: &[u8] = b"sent/";
const RECV_PREFIX: &[u8] = b"recv/";

impl WalletStore {
    pub fn open(env: &Env) -> heed::Result<Self> {
        let mut txn = env.write_txn()?;
        let db = env.create_database(&mut txn, Some("wallet"))?;
        txn.commit()?;
        Ok(Self { db })
    }

    /// The entropy behind the recovery words, if there's a wallet yet.
    pub fn entropy(&self, txn: &RoTxn) -> heed::Result<Option<[u8; crate::mnemonic::ENTROPY_LEN]>> {
        Ok(self
            .db
            .get(txn, ENTROPY_KEY)?
            .and_then(|b| b.try_into().ok()))
    }

    /// Makes the wallet (there must be none yet).
    pub fn set_entropy(
        &self,
        txn: &mut RwTxn,
        entropy: &[u8; crate::mnemonic::ENTROPY_LEN],
    ) -> heed::Result<()> {
        self.db.put(txn, ENTROPY_KEY, entropy)
    }

    pub fn owned(&self, txn: &RoTxn) -> heed::Result<Vec<Owned>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, OUT_PREFIX)? {
            let (k, v) = entry?;
            let c = Commitment(
                k[OUT_PREFIX.len()..]
                    .try_into()
                    .map_err(|_| heed::Error::Encoding("bad key".into()))?,
            );
            if let Some(o) = Owned::from_bytes(c, v) {
                out.push(o);
            }
        }
        Ok(out)
    }

    pub fn put_owned(&self, txn: &mut RwTxn, o: &Owned) -> heed::Result<()> {
        self.db.put(
            txn,
            &[OUT_PREFIX, &o.record.commitment.0].concat(),
            &o.to_bytes(),
        )
    }

    pub fn remove_owned(&self, txn: &mut RwTxn, c: &Commitment) -> heed::Result<()> {
        self.db.delete(txn, &[OUT_PREFIX, &c.0].concat())?;
        Ok(())
    }

    /// The height the outputs are found up to, if any scan has run.
    pub fn synced(&self, txn: &RoTxn) -> heed::Result<Option<u64>> {
        Ok(self
            .db
            .get(txn, SYNCED_KEY)?
            .and_then(|b| b.try_into().ok())
            .map(u64::from_le_bytes))
    }

    pub fn set_synced(&self, txn: &mut RwTxn, height: u64) -> heed::Result<()> {
        self.db.put(txn, SYNCED_KEY, &height.to_le_bytes())
    }

    /// The hash recorded for the block scanned at `height`.
    pub fn block_hash(&self, txn: &RoTxn, height: u64) -> heed::Result<Option<Vec<u8>>> {
        Ok(self
            .db
            .get(txn, &[BLK_PREFIX, &height.to_be_bytes()].concat())?
            .map(<[u8]>::to_vec))
    }

    pub fn set_block_hash(&self, txn: &mut RwTxn, height: u64, hash: &[u8]) -> heed::Result<()> {
        self.db
            .put(txn, &[BLK_PREFIX, &height.to_be_bytes()].concat(), hash)
    }

    /// Records that key `index` holds a mining reward of ours.
    pub fn mark_reward(&self, txn: &mut RwTxn, index: u32) -> heed::Result<()> {
        self.db
            .put(txn, &[REWARD_PREFIX, &index.to_be_bytes()].concat(), &[])
    }

    pub fn is_reward(&self, txn: &RoTxn, index: u32) -> heed::Result<bool> {
        Ok(self
            .db
            .get(txn, &[REWARD_PREFIX, &index.to_be_bytes()].concat())?
            .is_some())
    }

    /// Whether a restored wallet is still looking for its outputs.
    pub fn restoring(&self, txn: &RoTxn) -> heed::Result<bool> {
        Ok(self.db.get(txn, RESTORING_KEY)?.is_some())
    }

    pub fn set_restoring(&self, txn: &mut RwTxn, on: bool) -> heed::Result<()> {
        if on {
            self.db.put(txn, RESTORING_KEY, &[])
        } else {
            self.db.delete(txn, RESTORING_KEY).map(|_| ())
        }
    }

    /// Sets the next unused key index.
    pub fn set_next_index(&self, txn: &mut RwTxn, next: u32) -> heed::Result<()> {
        self.db.put(txn, NEXT_KEY, &next.to_le_bytes())
    }

    /// The next unused key index.
    pub fn next_index(&self, txn: &RoTxn) -> heed::Result<u32> {
        Ok(self
            .db
            .get(txn, NEXT_KEY)?
            .and_then(|b| b.try_into().ok())
            .map_or(0, u32::from_le_bytes))
    }

    /// Takes a fresh key index.
    pub fn take_index(&self, txn: &mut RwTxn) -> heed::Result<u32> {
        let i = self.next_index(txn)?;
        self.db.put(txn, NEXT_KEY, &(i + 1).to_le_bytes())?;
        Ok(i)
    }

    fn sent_key(id: &SlateId) -> Vec<u8> {
        [SENT_PREFIX, id].concat()
    }

    pub fn put_sent(&self, txn: &mut RwTxn, sent: &Sent) -> heed::Result<()> {
        self.db
            .put(txn, &Self::sent_key(&sent.id), &sent.to_bytes())
    }

    pub fn remove_sent(&self, txn: &mut RwTxn, id: &SlateId) -> heed::Result<()> {
        self.db.delete(txn, &Self::sent_key(id))?;
        Ok(())
    }

    pub fn put_received(&self, txn: &mut RwTxn, r: &Received) -> heed::Result<()> {
        self.db
            .put(txn, &[RECV_PREFIX, &r.id].concat(), &r.to_bytes())
    }

    pub fn remove_received(&self, txn: &mut RwTxn, id: &SlateId) -> heed::Result<()> {
        self.db.delete(txn, &[RECV_PREFIX, id].concat())?;
        Ok(())
    }

    /// Every payment being received, oldest first.
    pub fn receiveds(&self, txn: &RoTxn) -> heed::Result<Vec<Received>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, RECV_PREFIX)? {
            if let Some(r) = Received::from_bytes(entry?.1) {
                out.push(r);
            }
        }
        out.sort_by_key(|r| r.created);
        Ok(out)
    }

    /// Every payment being sent, oldest first.
    pub fn sents(&self, txn: &RoTxn) -> heed::Result<Vec<Sent>> {
        let mut out = Vec::new();
        for entry in self.db.prefix_iter(txn, SENT_PREFIX)? {
            let (_, v) = entry?;
            if let Some(s) = Sent::from_bytes(v) {
                out.push(s);
            }
        }
        out.sort_by_key(|s| s.created);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wallet(n: u32) -> Wallet {
        Wallet::from_seed(BabyBear::new_array(core::array::from_fn(|i| {
            n * 100 + i as u32
        })))
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
        // Different key indices get different tags.
        assert_ne!(
            nonce_for(&w.view(1), 5)[..TAG_LEN],
            nonce_for(&w.view(2), 5)[..TAG_LEN]
        );
    }

    #[test]
    fn scan_finds_own_outputs_with_amounts() {
        let (alice, bob) = (wallet(1), wallet(2));
        let outputs: Vec<(u64, OutputRecord)> = [
            alice.key_for(0).pay(10),
            bob.key_for(0).pay(20),
            alice.key_for(5).pay(30),
            bob.key_for(1).pay(40),
            alice.key_for(50).pay(60),
        ]
        .iter()
        .enumerate()
        .map(|(i, o)| (i as u64 + 1, record(o)))
        .collect();
        let coins = alice.scan(&outputs, 0..10);
        let found: Vec<(u32, u64, Amount)> = coins
            .iter()
            .map(|c| (c.index, c.position, c.amount))
            .collect();
        assert_eq!(found, [(0, 1, 10), (5, 3, 30)]);
        // A wider range finds the far index too.
        assert_eq!(alice.scan(&outputs, 0..100).len(), 3);
        // Bob sees only his.
        let bobs: Vec<Amount> = bob.scan(&outputs, 0..10).iter().map(|c| c.amount).collect();
        assert_eq!(bobs, [20, 40]);
    }

    #[test]
    fn scan_rejects_a_matching_tag_with_the_wrong_commitment() {
        // Someone who knows a key's view secret can copy its tag, but can't make the
        // commitment check pass for another key.
        let alice = wallet(1);
        let mut fake = wallet(9).key_for(0).pay(10);
        fake.nonce = nonce_for(&alice.view(0), 10);
        let outputs = [(1, record(&fake))];
        assert!(alice.scan(&outputs, 0..5).is_empty());
    }

    #[test]
    fn slates_and_responses_round_trip_through_text() {
        let w = wallet(2);
        let slate = Slate {
            id: *b"0123456789abcdef",
            amount: 150_000_000,
            fee: 100_000,
        };
        assert_eq!(Slate::from_text(&slate.to_text()), Ok(slate));
        let response = Response {
            id: slate.id,
            amount: slate.amount,
            fee: slate.fee,
            output: w.key_for(2).pay(slate.amount),
        };
        assert_eq!(Response::from_text(&response.to_text()), Ok(response));
        // Each kind only reads its own.
        assert!(Slate::from_text(&response.to_text()).is_err());
        assert!(Response::from_text(&slate.to_text()).is_err());
        assert!(Slate::from_text("r slate 2\nid 00\namount 1\nfee 1\n").is_err());
        assert!(Slate::from_text("r slate 2\namount 1\n").is_err());
        assert_eq!(
            unhex(&hex(&[0, 1, 0xab, 0xff])),
            Some(vec![0, 1, 0xab, 0xff])
        );
        assert_eq!(unhex("abc"), None);
    }

    #[test]
    fn the_store_keeps_payments_and_hands_out_fresh_indices() {
        let env = crate::db::test::TempEnv::new();
        let store = WalletStore::open(env.env()).unwrap();
        let w = Wallet::from_seed(crate::mnemonic::seed_of(&[5; 32]));
        let mut txn = env.env().write_txn().unwrap();
        assert_eq!(store.entropy(&txn).unwrap(), None);
        store.set_entropy(&mut txn, &[5; 32]).unwrap();
        assert_eq!(store.entropy(&txn).unwrap(), Some([5; 32]));
        assert_eq!(store.take_index(&mut txn).unwrap(), 0);
        assert_eq!(store.take_index(&mut txn).unwrap(), 1);
        assert_eq!(store.next_index(&txn).unwrap(), 2);
        let coin = Coin {
            index: 0,
            position: 5,
            amount: 70,
            record: record(&w.key_for(0).pay(70)),
        };
        let mut sent = Sent {
            id: [7; 16],
            amount: 50,
            fee: 3,
            coins: vec![coin],
            change: Some((1, 17)),
            created: 1000,
            tx: None,
        };
        store.put_sent(&mut txn, &sent).unwrap();
        let older = Sent {
            id: [1; 16],
            created: 10,
            change: None,
            tx: Some(vec![1, 2, 3]),
            ..sent.clone()
        };
        store.put_sent(&mut txn, &older).unwrap();
        assert_eq!(
            store.sents(&txn).unwrap(),
            vec![older.clone(), sent.clone()]
        );
        sent.tx = Some(vec![9; 40]);
        store.put_sent(&mut txn, &sent).unwrap();
        store.remove_sent(&mut txn, &older.id).unwrap();
        assert_eq!(store.sents(&txn).unwrap(), vec![sent]);
        let got = Received {
            id: [3; 16],
            index: 9,
            amount: 12,
            commitment: coin.record.commitment,
            created: 5,
        };
        store.put_received(&mut txn, &got).unwrap();
        assert_eq!(store.receiveds(&txn).unwrap(), vec![got]);
        // Kept apart from the payments being sent.
        assert_eq!(store.sents(&txn).unwrap().len(), 1);
        store.remove_received(&mut txn, &got.id).unwrap();
        assert!(store.receiveds(&txn).unwrap().is_empty());
        let owned = Owned {
            index: 0,
            amount: 70,
            record: coin.record,
            height: 4,
            reward: true,
            gone_since: 9,
        };
        store.put_owned(&mut txn, &owned).unwrap();
        assert_eq!(store.owned(&txn).unwrap(), vec![owned]);
        store
            .remove_owned(&mut txn, &owned.record.commitment)
            .unwrap();
        assert!(store.owned(&txn).unwrap().is_empty());
        assert_eq!(store.synced(&txn).unwrap(), None);
        store.set_synced(&mut txn, 7).unwrap();
        store.set_block_hash(&mut txn, 7, &[1; 32]).unwrap();
        assert_eq!(store.synced(&txn).unwrap(), Some(7));
        assert_eq!(store.block_hash(&txn, 7).unwrap(), Some(vec![1; 32]));
        store.mark_reward(&mut txn, 3).unwrap();
        assert!(store.is_reward(&txn, 3).unwrap() && !store.is_reward(&txn, 4).unwrap());
        store.set_restoring(&mut txn, true).unwrap();
        assert!(store.restoring(&txn).unwrap());
        store.set_restoring(&mut txn, false).unwrap();
        assert!(!store.restoring(&txn).unwrap());
    }

    #[test]
    fn coins_spend_with_their_position() {
        let w = wallet(3);
        let coin = Coin {
            index: 2,
            position: 77,
            amount: 9,
            record: record(&w.key_for(2).pay(9)),
        };
        let (_, input) = w.input(&coin);
        assert!(input.opens_commitment());
        assert_eq!(input.position, 77);
    }
}
