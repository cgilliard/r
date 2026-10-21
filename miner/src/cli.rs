//! The node's command line: the wallet, and a look at the node, typed at its
//! prompt while it runs (its log goes to a file).
//!
//! Payments go through files, as in Grin ([`crate::wallet`]):
//!
//! - `send <amount> <fee> <file>`: picks coins, locks them, and writes the
//!   slate to `<file>` for the receiver;
//! - `receive <file>`: makes an output for the slate's amount at a fresh
//!   key and writes it to `<file>.response` for the sender;
//! - `finalize <file>`: signs the transaction with the receiver's output and
//!   submits it to the mempool, which relays it.
//!
//! `info` shows the balance, with the coins received but not yet in a block
//! as pending; `txs` lists the payments in progress, sent and received (and
//! drops the confirmed ones); `cancel <id>` unlocks the coins of a payment
//! not yet finalized, or stops tracking one being received.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ledger::Ledger;
use crate::mempool::{Chainstate, Mempool, Refused};
use crate::net::node::Event;
use crate::output::{Amount, format_coins, parse_coins};
use crate::transaction::{Transaction, TxBody};
use crate::wallet::{
    Coin, Owned, Received, Response, Sent, Slate, SlateId, Tag, Wallet, WalletStore, hex, tag_of,
};

/// Inputs in one payment, at most (each adds about 2.2 KB).
const MAX_INPUTS: usize = 32;
/// Key indices looked at past the last one used, when scanning blocks.
const LOOKAHEAD: u32 = 100;
/// Key indices scanned past the last one found when looking for a restored
/// wallet's outputs
/// (a restored wallet), and skipped after restoring.
const GAP: u32 = 1000;
/// Blocks a spent output is remembered for, in case a reorg unspends it.
const KEEP_GONE: u64 = 100;
/// Mining rewards' maturity by default, in confirmations.
pub const DEFAULT_MATURITY: u64 = 6;

/// Tags of key indices below `end`.
#[derive(Default)]
pub struct TagCache {
    end: u32,
    map: HashMap<Tag, u32>,
}

/// What an output of the wallet's can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    Spendable,
    /// A mining reward this many confirmations short of maturity.
    Immature(u64),
    /// Spent by a payment in progress.
    Locked(SlateId),
    /// Spent by a transaction in the mempool.
    Spending,
}

/// An unspent output of the wallet's.
struct Holding {
    coin: Coin,
    owned: Owned,
    /// Recorded here as a mining reward of ours (not just found on
    /// restore).
    known_reward: bool,
    confirmations: u64,
    state: Use,
}

/// Recovery words, on one line, ready to paste into a restore.
pub fn words_block(entropy: &[u8; crate::mnemonic::ENTROPY_LEN]) -> String {
    crate::mnemonic::to_words(entropy).join(" ")
}

pub const HELP: &str = "\
commands:
  info                         balance and chain height
  outputs                      the wallet's outputs, one per line
  seed                         the recovery words
  send <amount> <fee> <file>   start a payment: writes the slate to <file>
  receive <file>               answer a slate: writes <file>.response
  finalize <file>              sign and submit a payment from its response
  txs                          payments in progress, sent and received
  cancel <id>                  drop a payment not yet finalized (or stop
                               tracking one being received)
  status                       the node: tip, sync, mempool
  help                         this list
  quit                         stop the node";

/// What a command did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Text(String),
    Quit,
}

/// The wallet and the node it runs in.
pub struct Cli {
    pub ledger: Arc<Ledger>,
    pub mempool: Arc<Mempool>,
    pub events: Sender<Event>,
    pub store: WalletStore,
    pub wallet: Wallet,
    pub synced: Arc<AtomicBool>,
    /// Confirmations before a mining reward of ours can be spent.
    pub maturity: u64,
    /// Key index tags computed so far.
    pub tag_cache: Mutex<TagCache>,
    /// Logs to the node's log.
    pub log: Box<dyn Fn(String) + Send + Sync>,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn coins(units: Amount) -> String {
    format_coins(units as u128)
}

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Reads a file, saying which one failed.
fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("couldn't read {path}: {e}"))
}

/// Writes a new file (never one that exists).
fn write_new(path: &str, text: &str) -> Result<(), String> {
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("couldn't create {path}: {e}"))?;
    f.write_all(text.as_bytes())
        .map_err(|e| format!("couldn't write {path}: {e}"))
}

impl Cli {
    /// Runs commands from `input` until `quit` (returns `true`) or the end
    /// of input (`false`), prompting on `out`.
    pub fn run(&self, input: impl BufRead, mut out: impl Write) -> bool {
        let _ = write!(out, "> ");
        let _ = out.flush();
        for line in input.lines() {
            let Ok(line) = line else {
                break;
            };
            match self.exec(&line) {
                Ok(Outcome::Quit) => return true,
                Ok(Outcome::Text(t)) if t.is_empty() => {}
                Ok(Outcome::Text(t)) => {
                    let _ = writeln!(out, "{t}");
                }
                Err(e) => {
                    let _ = writeln!(out, "error: {e}");
                }
            }
            let _ = write!(out, "> ");
            let _ = out.flush();
        }
        false
    }

    /// Runs one command.
    pub fn exec(&self, line: &str) -> Result<Outcome, String> {
        let words: Vec<&str> = line.split_whitespace().collect();
        let text = match words.as_slice() {
            [] => String::new(),
            ["help"] => HELP.into(),
            ["quit" | "exit"] => return Ok(Outcome::Quit),
            ["info"] => self.info()?,
            ["outputs" | "output"] => self.outputs()?,
            ["seed"] => self.seed()?,
            ["status"] => self.status()?,
            ["send", amount, fee, file] => {
                let amount = parse_coins(amount).ok_or("bad amount")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.send(amount, fee, file)?
            }
            ["receive", file] => self.receive(file)?,
            ["finalize", file] => self.finalize(file)?,
            ["txs"] => self.txs()?,
            ["cancel", id] => self.cancel(id)?,
            [cmd, ..] => return Err(format!("unknown or incomplete command \"{cmd}\"; try help")),
        };
        Ok(Outcome::Text(text))
    }

    /// Brings the wallet's outputs up to the tip. After a reorg it first
    /// drops what came from blocks no longer on the main chain. Then it
    /// looks each new main-chain block's outputs up by their tags, or, when
    /// blocks are missing (below a fast-sync base) or a restored wallet is
    /// still looking, scans the whole state. Outputs gone from the state
    /// (spent) are kept for [`KEEP_GONE`] blocks, in case a reorg brings
    /// them back.
    fn sync(&self) -> Result<(), String> {
        let l = &self.ledger;
        let txn = l.env.read_txn().map_err(err)?;
        let tip = l.store.tip(&txn).map_err(err)?.ok_or("no chain")?;
        let restoring = self.store.restoring(&txn).map_err(err)?;
        let synced = self.store.synced(&txn).map_err(err)?;
        // Back to the last scanned block still on the main chain.
        let mut fork = synced.map(|h| h.min(tip.height));
        while let Some(h) = fork.filter(|&h| h > 0) {
            let recorded = self.store.block_hash(&txn, h).map_err(err)?;
            let current = l.store.hash_at(&txn, h).map_err(err)?;
            match (recorded, current) {
                (Some(a), Some(b)) if a == crate::hash::to_bytes(&b) => break,
                // Not recorded (found by a state scan): take it as is.
                (None, _) => break,
                _ => fork = Some(h - 1),
            }
        }
        let base = l.store.base(&txn).map_err(err)?;
        // A restored wallet looks from the start (it doesn't know how far
        // its key indices go, so the range widens while outputs turn up near
        // its end). Outputs it finds may be mining rewards: they mature.
        let start = if restoring {
            1
        } else {
            fork.map_or(1, |h| h + 1)
        };
        let whole_state = base > 0 && start < base;
        let next = self.store.next_index(&txn).map_err(err)?;
        let mut end = next.saturating_add(if restoring { GAP } else { LOOKAHEAD });
        let mut found: Vec<Owned>;
        let mut hashes;
        loop {
            let tags = self.tags(end);
            found = Vec::new();
            hashes = Vec::new();
            // The blocks there are: from the start, or (when older ones are
            // missing, below a fast-sync base) from the base.
            let first = if whole_state { base.max(1) } else { start };
            for h in first..=tip.height {
                let block = l
                    .store
                    .block(&txn, h)
                    .map_err(err)?
                    .ok_or("missing block")?;
                let hash = l
                    .store
                    .hash_at(&txn, h)
                    .map_err(err)?
                    .ok_or("missing block")?;
                for record in &block.outputs {
                    let Some(&index) = tags.get(&tag_of(record)) else {
                        continue;
                    };
                    if let Some(amount) = self.wallet.identify(index, record) {
                        found.push(Owned {
                            index,
                            amount,
                            record: *record,
                            height: h,
                            reward: restoring || self.store.is_reward(&txn, index).map_err(err)?,
                            gone_since: 0,
                        });
                    }
                }
                hashes.push((h, crate::hash::to_bytes(&hash)));
            }
            if whole_state {
                // The state has every unspent output; one no block above
                // the base made was made at or below it.
                let outputs = l.tree.outputs(&txn).map_err(err)?;
                for (_, record) in outputs {
                    if found
                        .iter()
                        .any(|o| o.record.commitment == record.commitment)
                    {
                        continue;
                    }
                    let Some(&index) = tags.get(&tag_of(&record)) else {
                        continue;
                    };
                    if let Some(amount) = self.wallet.identify(index, &record) {
                        found.push(Owned {
                            index,
                            amount,
                            record,
                            height: base,
                            reward: restoring || self.store.is_reward(&txn, index).map_err(err)?,
                            gone_since: 0,
                        });
                    }
                }
            }
            let top = found.iter().map(|o| o.index).max().unwrap_or(0);
            if !restoring || top.saturating_add(GAP) < end {
                break;
            }
            end = top.saturating_add(1).saturating_add(GAP);
        }
        let known = self.store.owned(&txn).map_err(err)?;
        drop(txn);

        let mut txn = l.env.write_txn().map_err(err)?;
        if let Some(f) = fork.filter(|f| synced.is_some_and(|s| *f < s)) {
            for o in known.iter().filter(|o| o.height > f) {
                self.store
                    .remove_owned(&mut txn, &o.record.commitment)
                    .map_err(err)?;
            }
        }
        let top = found.iter().map(|o| o.index).max();
        for o in &found {
            // A state scan doesn't know heights below the base: keep the
            // one we know, if it's earlier.
            let old = known
                .iter()
                .find(|k| k.record.commitment == o.record.commitment);
            let o = match old {
                Some(k) if k.height < o.height => Owned {
                    height: k.height,
                    ..*o
                },
                _ => *o,
            };
            self.store.put_owned(&mut txn, &o).map_err(err)?;
        }
        for (h, hash) in &hashes {
            self.store.set_block_hash(&mut txn, *h, hash).map_err(err)?;
        }
        self.store.set_synced(&mut txn, tip.height).map_err(err)?;
        if let Some(top) = top
            && top >= next
        {
            self.store.set_next_index(&mut txn, top + 1).map_err(err)?;
        }
        if restoring && self.synced.load(Ordering::Relaxed) {
            // Skip past key indices that may have been used and spent: their
            // keys may have signed already.
            let next = self.store.next_index(&txn).map_err(err)?;
            self.store
                .set_next_index(&mut txn, next.saturating_add(GAP))
                .map_err(err)?;
            self.store.set_restoring(&mut txn, false).map_err(err)?;
            (self.log)(format!(
                "wallet: restored; {} output{} found",
                found.len(),
                if found.len() == 1 { "" } else { "s" }
            ));
        }
        // Mark the spent ones, and forget the long-spent.
        for o in self.store.owned(&txn).map_err(err)? {
            let present = l
                .tree
                .get(&txn, &o.record.commitment)
                .map_err(err)?
                .is_some();
            let updated = match (present, o.gone_since) {
                (true, g) if g != 0 => Some(Owned { gone_since: 0, ..o }),
                (false, 0) => Some(Owned {
                    gone_since: tip.height.max(1),
                    ..o
                }),
                (false, g) if tip.height.saturating_sub(g) >= KEEP_GONE => {
                    self.store
                        .remove_owned(&mut txn, &o.record.commitment)
                        .map_err(err)?;
                    None
                }
                _ => None,
            };
            if let Some(u) = updated {
                self.store.put_owned(&mut txn, &u).map_err(err)?;
            }
        }
        // Payments being received whose outputs have arrived are done.
        let owned = self.store.owned(&txn).map_err(err)?;
        for r in self.store.receiveds(&txn).map_err(err)? {
            if owned.iter().any(|o| o.record.commitment == r.commitment) {
                self.store.remove_received(&mut txn, &r.id).map_err(err)?;
            }
        }
        txn.commit().map_err(err)
    }

    /// The tags of key indices below `end` (cached: computing each takes a
    /// hash).
    fn tags(&self, end: u32) -> HashMap<Tag, u32> {
        let mut cache = self.tag_cache.lock().expect("tag cache");
        if cache.end < end {
            let more = self.wallet.tags(cache.end..end);
            cache.map.extend(more);
            cache.end = end;
        }
        cache.map.clone()
    }

    /// The wallet's unspent outputs (brought up to date first), each with
    /// its slot, confirmations and what it can do.
    fn holdings(&self) -> Result<Vec<Holding>, String> {
        self.sync()?;
        let l = &self.ledger;
        let txn = l.env.read_txn().map_err(err)?;
        let tip = l.store.tip(&txn).map_err(err)?.ok_or("no chain")?;
        let sents = self.store.sents(&txn).map_err(err)?;
        let mut out = Vec::new();
        for o in self.store.owned(&txn).map_err(err)? {
            let Some((position, _)) = l.tree.get(&txn, &o.record.commitment).map_err(err)? else {
                continue;
            };
            let confirmations = tip.height.saturating_sub(o.height) + 1;
            let locked_by = sents
                .iter()
                .find(|s| {
                    s.coins
                        .iter()
                        .any(|c| c.record.commitment == o.record.commitment)
                })
                .map(|s| s.id);
            let state = if let Some(id) = locked_by {
                Use::Locked(id)
            } else if self.mempool.spends(&o.record.commitment) {
                Use::Spending
            } else if o.reward && confirmations < self.maturity {
                Use::Immature(self.maturity - confirmations)
            } else {
                Use::Spendable
            };
            out.push(Holding {
                known_reward: self.store.is_reward(&txn, o.index).map_err(err)?,
                coin: Coin {
                    index: o.index,
                    position,
                    amount: o.amount,
                    record: o.record,
                },
                owned: o,
                confirmations,
                state,
            });
        }
        out.sort_by_key(|h| (h.owned.height, h.coin.index));
        Ok(out)
    }

    /// Payments being received whose outputs aren't in a block yet (as of
    /// the last [`Cli::sync`]).
    fn pending(&self) -> Result<Vec<Received>, String> {
        let txn = self.ledger.env.read_txn().map_err(err)?;
        self.store.receiveds(&txn).map_err(err)
    }

    fn info(&self) -> Result<String, String> {
        let (tip, _) = self.ledger.tip().map_err(err)?;
        let held = self.holdings()?;
        let line = |what: &str, pick: &dyn Fn(&Use) -> bool, note: &str| {
            let (n, sum) = held
                .iter()
                .filter(|h| pick(&h.state))
                .fold((0, 0), |(n, s), h| (n + 1, s + h.coin.amount));
            format!(
                "{what:<11} {} coins in {n} output{}{note}",
                coins(sum),
                if n == 1 { "" } else { "s" }
            )
        };
        let pending = self.pending()?;
        let pending_sum: Amount = pending.iter().map(|r| r.amount).sum();
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let sents = self.store.sents(&txn).map_err(err)?;
        // (Checking the coins opens its own transaction.)
        drop(txn);
        let incoming: Amount = sents
            .iter()
            .filter(|s| s.tx.is_some() && self.unconfirmed(s))
            .filter_map(|s| s.change.map(|(_, a)| a))
            .sum();
        Ok([
            format!(
                "chain height {}{}",
                tip.height,
                if self.synced.load(Ordering::Relaxed) {
                    ""
                } else {
                    " (still syncing)"
                }
            ),
            line("spendable", &|u| *u == Use::Spendable, ""),
            line(
                "immature",
                &|u| matches!(u, Use::Immature(_)),
                &format!(
                    " (mining rewards: spendable after {} confirmations)",
                    self.maturity
                ),
            ),
            format!(
                "pending     {} coins in {} output{} (received; spendable once the sender \
                 finalizes and a block includes it)",
                coins(pending_sum),
                pending.len(),
                if pending.len() == 1 { "" } else { "s" },
            ),
            line(
                "locked",
                &|u| matches!(u, Use::Locked(_) | Use::Spending),
                " (payments in progress)",
            ),
            format!(
                "change      {} coins coming back once sent payments confirm",
                coins(incoming)
            ),
        ]
        .join("\n"))
    }

    fn outputs(&self) -> Result<String, String> {
        let held = self.holdings()?;
        if held.is_empty() {
            return Ok("no outputs".into());
        }
        let mut lines = vec![format!(
            "{:>7}  {:>16}  {:>7}  {:>6}  {}",
            "key", "coins", "height", "confs", "status"
        )];
        for h in &held {
            let status = match h.state {
                Use::Spendable => "spendable".to_string(),
                Use::Immature(n) => {
                    format!("immature: {n} more block{}", if n == 1 { "" } else { "s" })
                }
                Use::Locked(id) => format!("locked by payment {}", &hex(&id)[..8]),
                Use::Spending => "being spent (in the mempool)".to_string(),
            };
            lines.push(format!(
                "{:>7}  {:>16}  {:>7}  {:>6}  {status}{}",
                h.coin.index,
                coins(h.coin.amount),
                h.owned.height,
                h.confirmations,
                match (h.owned.reward, h.known_reward) {
                    (true, true) => " (mining reward)",
                    (true, false) => " (found on restore: may be a mining reward)",
                    _ => "",
                },
            ));
        }
        let total: Amount = held.iter().map(|h| h.coin.amount).sum();
        lines.push(format!(
            "{} output{}, {} coins",
            held.len(),
            if held.len() == 1 { "" } else { "s" },
            coins(total)
        ));
        Ok(lines.join("\n"))
    }

    fn seed(&self) -> Result<String, String> {
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let entropy = self.store.entropy(&txn).map_err(err)?.ok_or("no wallet")?;
        Ok(format!(
            "recovery words (anyone who has them can spend your coins; keep them offline):\n{}\n\
             restore with: r --restore --data-dir <a new directory>",
            words_block(&entropy)
        ))
    }

    fn status(&self) -> Result<String, String> {
        let (tip, _) = self.ledger.tip().map_err(err)?;
        let (n, bytes) = self.mempool.size(&*self.ledger);
        Ok(format!(
            "tip {} at height {}; {}; mempool: {n} transaction{} ({bytes} bytes)",
            &hex(&crate::hash::to_bytes(&tip.hash()))[..16],
            tip.height,
            if self.synced.load(Ordering::Relaxed) {
                "caught up with peers"
            } else {
                "syncing"
            },
            if n == 1 { "" } else { "s" },
        ))
    }

    fn send(
        &self,
        amount: Amount,
        fee: Amount,
        file: &str,
    ) -> Result<String, String> {
        if amount == 0 {
            return Err("the amount must be more than 0".into());
        }
        if Path::new(file).exists() {
            return Err(format!("{file} exists already"));
        }
        let need = amount.checked_add(fee).ok_or("too much")?;
        // Coins are spent by key: every output at a key index together, so
        // the one-time key signs once. A key with any output that
        // can't be spent yet is left alone.
        let held = self.holdings()?;
        let mut groups: Vec<Vec<Coin>> = Vec::new();
        for h in &held {
            if held
                .iter()
                .any(|o| o.coin.index == h.coin.index && o.state != Use::Spendable)
            {
                continue;
            }
            match groups.iter_mut().find(|g| g[0].index == h.coin.index) {
                Some(g) => g.push(h.coin),
                None => groups.push(vec![h.coin]),
            }
        }
        groups.sort_by_key(|g| std::cmp::Reverse(g.iter().map(|c| c.amount).sum::<Amount>()));
        let (mut picked, mut total) = (Vec::new(), 0u64);
        for g in groups {
            if total >= need {
                break;
            }
            if picked.len() + g.len() > MAX_INPUTS {
                continue;
            }
            total += g.iter().map(|c| c.amount).sum::<Amount>();
            picked.extend(g);
        }
        if total < need {
            return Err(format!(
                "not enough coins: {} needed, {} spendable{}",
                coins(need),
                coins(total),
                if picked.len() == MAX_INPUTS {
                    format!(" in the {MAX_INPUTS} largest outputs")
                } else {
                    String::new()
                }
            ));
        }
        let mut id: SlateId = [0; 16];
        let mut rng = crate::rng::Rng::from_os().map_err(err)?;
        for chunk in id.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_element().as_u32().to_le_bytes());
        }
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let change = total - need;
        let change = if change > 0 {
            Some((self.store.take_index(&mut txn).map_err(err)?, change))
        } else {
            None
        };
        let sent = Sent {
            id,
            amount,
            fee,
            coins: picked,
            change,
            created: now(),
            tx: None,
        };
        self.store.put_sent(&mut txn, &sent).map_err(err)?;
        txn.commit().map_err(err)?;
        if let Err(e) = write_new(
            file,
            &Slate { id, amount, fee }.to_text(),
        ) {
            let _ = self.forget(&id);
            return Err(e);
        }
        (self.log)(format!(
            "wallet: payment {} started: {} coins, fee {}, from {} output{}",
            hex(&id),
            coins(amount),
            coins(fee),
            sent.coins.len(),
            if sent.coins.len() == 1 { "" } else { "s" },
        ));
        Ok(format!(
            "payment {} of {} coins (fee {}): slate written to {file}\n\
             send it to the receiver, then run: finalize {file}.response",
            hex(&id),
            coins(amount),
            coins(fee),
        ))
    }

    fn receive(&self, file: &str) -> Result<String, String> {
        let slate = Slate::from_text(&read(file)?)?;
        if slate.amount == 0 {
            return Err("the slate is for 0 coins".into());
        }
        let out_file = format!("{file}.response");
        if Path::new(&out_file).exists() {
            return Err(format!("{out_file} exists already"));
        }
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let index = self.store.take_index(&mut txn).map_err(err)?;
        let output = self.wallet.key_for(index).pay(slate.amount);
        let commitment = output.commitment().ok_or("couldn't make the output")?;
        self.store
            .put_received(
                &mut txn,
                &Received {
                    id: slate.id,
                    index,
                    amount: slate.amount,
                    commitment,
                    created: now(),
                },
            )
            .map_err(err)?;
        txn.commit().map_err(err)?;
        let response = Response {
            id: slate.id,
            amount: slate.amount,
            fee: slate.fee,
            output,
        };
        if let Err(e) = write_new(&out_file, &response.to_text()) {
            let mut txn = self.ledger.env.write_txn().map_err(err)?;
            self.store
                .remove_received(&mut txn, &slate.id)
                .map_err(err)?;
            txn.commit().map_err(err)?;
            return Err(e);
        }
        (self.log)(format!(
            "wallet: answered payment {} of {} coins",
            hex(&slate.id),
            coins(slate.amount)
        ));
        Ok(format!(
            "receiving {} coins (fee {} paid by the sender): response written to {out_file}\n\
             send it back to the sender to finalize",
            coins(slate.amount),
            coins(slate.fee),
        ))
    }

    fn finalize(&self, file: &str) -> Result<String, String> {
        let response = Response::from_text(&read(file)?)?;
        // Only a slate we sent, as we sent it.
        let mut sent = self
            .find(&hex(&response.id))?
            .ok_or(
                "unknown slate: we sent no payment with its id (it was modified, or isn't \
                 ours); not finalized",
            )?;
        if sent.tx.is_some() {
            return Err("that payment is finalized already (see txs)".into());
        }
        if (response.amount, response.fee) != (sent.amount, sent.fee) {
            return Err(format!(
                "modified slate: it says {} coins with fee {}, but we sent {} coins with \
                 fee {}; not finalized",
                coins(response.amount),
                coins(response.fee),
                coins(sent.amount),
                coins(sent.fee)
            ));
        }
        if response.output.amount != sent.amount {
            return Err(format!(
                "modified slate: its output is for {} coins, but we sent {}; not finalized",
                coins(response.output.amount),
                coins(sent.amount)
            ));
        }
        if sent
            .coins
            .iter()
            .any(|c| self.ledger.unspent(&c.record.commitment) != Some(c.position))
        {
            return Err("some of its coins are spent or moved; cancel it and send again".into());
        }
        let (keys, inputs) = sent.coins.iter().map(|c| self.wallet.input(c)).unzip();
        let mut outputs = vec![response.output];
        if let Some((index, amount)) = sent.change {
            outputs.push(self.wallet.key_for(index).pay(amount));
        }
        let tx = Transaction::sign(TxBody { inputs, outputs }, keys).map_err(err)?;
        let bytes = tx.to_bytes();
        // Keep the signed transaction before it goes anywhere: its keys
        // must never sign anything else.
        sent.tx = Some(bytes.clone());
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store.put_sent(&mut txn, &sent).map_err(err)?;
        txn.commit().map_err(err)?;
        Ok(match self.submit(&bytes) {
            Ok(id) => format!(
                "payment {} signed and submitted: transaction {}",
                hex(&sent.id),
                &hex(&crate::hash::to_bytes(&id))[..16]
            ),
            Err(e) => format!(
                "payment {} signed, but the mempool turned it away: {e}\n\
                 it stays on record (txs) and is submitted again",
                hex(&sent.id)
            ),
        })
    }

    /// Adds a transaction to the mempool and has the node announce it.
    fn submit(&self, bytes: &[u8]) -> Result<crate::hash::Hash, Refused> {
        let (id, new) = self.mempool.add(&*self.ledger, bytes)?;
        if new {
            let _ = self.events.send(Event::NewTx(id));
            (self.log)(format!(
                "wallet: transaction {} ({} bytes) added to the mempool",
                &hex(&crate::hash::to_bytes(&id))[..16],
                bytes.len()
            ));
        }
        Ok(id)
    }

    /// Submits every finalized payment not yet confirmed (as at a start).
    pub fn resubmit(&self) -> Result<usize, String> {
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let sents = self.store.sents(&txn).map_err(err)?;
        drop(txn);
        let mut n = 0;
        for s in sents {
            if let Some(tx) = &s.tx
                && self.unconfirmed(&s)
                && self.submit(tx).is_ok()
            {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Whether some of a payment's coins are still unspent.
    fn unconfirmed(&self, s: &Sent) -> bool {
        s.coins
            .iter()
            .any(|c| self.ledger.unspent(&c.record.commitment).is_some())
    }

    fn txs(&self) -> Result<String, String> {
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let sents = self.store.sents(&txn).map_err(err)?;
        drop(txn);
        let mut lines = Vec::new();
        for s in &sents {
            let what = match &s.tx {
                None => "waiting for the receiver's response".to_string(),
                Some(_) if !self.unconfirmed(s) => {
                    self.forget(&s.id)?;
                    "confirmed (now dropped from this list)".to_string()
                }
                Some(tx) => {
                    let id = Transaction::id_of(tx);
                    if self.mempool.knows(&id) {
                        "submitted, waiting for a block".to_string()
                    } else {
                        "signed, not in the mempool (submitted again at the next start)".to_string()
                    }
                }
            };
            lines.push(format!(
                "{}  {} coins, fee {}: {what}",
                hex(&s.id),
                coins(s.amount),
                coins(s.fee)
            ));
        }
        // (Drops the received payments that have arrived.)
        self.sync()?;
        for r in self.pending()? {
            lines.push(format!(
                "{}  receiving {} coins: {}",
                hex(&r.id),
                coins(r.amount),
                if self.mempool.creates(&r.commitment) {
                    "submitted, waiting for a block"
                } else {
                    "waiting for the sender to finalize"
                }
            ));
        }
        Ok(if lines.is_empty() {
            "no payments in progress".into()
        } else {
            lines.join("\n")
        })
    }

    fn cancel(&self, prefix: &str) -> Result<String, String> {
        let Some(sent) = self.find(prefix)? else {
            let got = self
                .pending()?
                .into_iter()
                .find(|r| hex(&r.id).starts_with(&prefix.to_lowercase()))
                .ok_or("no payment has that id")?;
            let mut txn = self.ledger.env.write_txn().map_err(err)?;
            self.store.remove_received(&mut txn, &got.id).map_err(err)?;
            txn.commit().map_err(err)?;
            return Ok(format!(
                "stopped tracking payment {} (if it arrives anyway, it's spendable as usual)",
                hex(&got.id)
            ));
        };
        if sent.tx.is_some() {
            return Err("that payment is signed already, so it can't be cancelled: \
                        its coins' keys are used, and only that transaction may spend them"
                .into());
        }
        self.forget(&sent.id)?;
        (self.log)(format!("wallet: payment {} cancelled", hex(&sent.id)));
        Ok(format!(
            "payment {} cancelled; its coins are spendable again",
            hex(&sent.id)
        ))
    }

    /// The payment whose id (in hex) starts with `prefix` (at least 4
    /// characters, and only one may match).
    fn find(&self, prefix: &str) -> Result<Option<Sent>, String> {
        if prefix.len() < 4 {
            return Err("give at least 4 characters of the id".into());
        }
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let mut matches: Vec<Sent> = self
            .store
            .sents(&txn)
            .map_err(err)?
            .into_iter()
            .filter(|s| hex(&s.id).starts_with(&prefix.to_lowercase()))
            .collect();
        match matches.len() {
            0 | 1 => Ok(matches.pop()),
            _ => Err("more than one payment matches; give more of the id".into()),
        }
    }

    fn forget(&self, id: &SlateId) -> Result<(), String> {
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store.remove_sent(&mut txn, id).map_err(err)?;
        txn.commit().map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::db::test::TempEnv;
    use crate::header::Header;
    use crate::ledger::tests::{ledger, tiny_proof};
    use crate::output::{COIN, OutputRecord};
    use std::sync::mpsc::{Receiver, channel};

    /// (The wallet goes first: dropping the environment waits for every
    /// handle to it to close.)
    struct Node {
        cli: Cli,
        events: Receiver<Event>,
        _env: TempEnv,
    }

    fn node(genesis: &Header) -> Node {
        static N: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);
        node_with(genesis, [N.fetch_add(1, Ordering::Relaxed); 32], false)
    }

    /// A node whose wallet has `entropy` (restored from it if `restore`),
    /// maturing rewards after 3 confirmations.
    fn node_with(genesis: &Header, entropy: [u8; 32], restore: bool) -> Node {
        let env = TempEnv::new();
        let ledger = Arc::new(ledger(&env, genesis));
        let store = WalletStore::open(env.env()).unwrap();
        let mut txn = env.env().write_txn().unwrap();
        store.set_entropy(&mut txn, &entropy).unwrap();
        store.set_restoring(&mut txn, restore).unwrap();
        txn.commit().unwrap();
        let wallet = Wallet::from_seed(crate::mnemonic::seed_of(&entropy));
        let (tx, events) = channel();
        let cli = Cli {
            ledger,
            mempool: Arc::new(Mempool::new()),
            events: tx,
            store,
            wallet,
            synced: Arc::new(AtomicBool::new(true)),
            maturity: 3,
            tag_cache: Mutex::default(),
            log: Box::new(|_| {}),
        };
        Node {
            _env: env,
            cli,
            events,
        }
    }

    /// Adds a block creating `created` and spending `spent` to every node,
    /// on the first one's tip; returns it.
    fn block(
        nodes: &[&Node],
        created: &[OutputRecord],
        spent: &[crate::output::Commitment],
    ) -> (Header, Block) {
        let l = &nodes[0].cli.ledger;
        let (tip, _) = l.tip().unwrap();
        let state = {
            let mut txn = l.env.write_txn().unwrap();
            l.tree.apply_changes(&mut txn, created, spent).unwrap();
            let s = l.tree.state(&txn).unwrap();
            txn.abort();
            s
        };
        let block = Block {
            inputs: spent.to_vec(),
            outputs: created.to_vec(),
            proof: tiny_proof(),
        };
        let header = Header {
            height: tip.height + 1,
            prev: tip.hash(),
            timestamp: tip.timestamp + 600,
            state,
            body: block.body_hash(),
            nonce: 0,
        };
        for n in nodes {
            n.cli.ledger.add(&header, &block).unwrap();
        }
        (header, block)
    }

    /// An output of `amount` for key `index` of `n`'s wallet.
    fn pay_to(n: &Node, index: u32, amount: Amount) -> OutputRecord {
        n.cli.wallet.key_for(index).pay(amount).record().unwrap()
    }

    fn outputs(n: &Node) -> Vec<String> {
        run(n, "outputs")
            .unwrap()
            .lines()
            .skip(1)
            .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect()
    }

    /// An output of `amount` for a fresh key of `n`'s wallet.
    fn pay(n: &Node, amount: Amount) -> OutputRecord {
        let mut txn = n.cli.ledger.env.write_txn().unwrap();
        let i = n.cli.store.take_index(&mut txn).unwrap();
        txn.commit().unwrap();
        n.cli.wallet.key_for(i).pay(amount).record().unwrap()
    }

    fn run(n: &Node, line: &str) -> Result<String, String> {
        match n.cli.exec(line)? {
            Outcome::Text(t) => Ok(t),
            Outcome::Quit => Ok("quit".into()),
        }
    }

    fn spendable(n: &Node) -> String {
        let info = run(n, "info").unwrap();
        info.lines()
            .find(|l| l.starts_with("spendable"))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .to_string()
    }

    fn genesis() -> Header {
        Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        }
    }

    #[test]
    fn rewards_mature_and_outputs_are_listed() {
        let g = genesis();
        let (alice, other) = (node(&g), node(&g));
        // A reward of ours for key 0, and a payment for key 1.
        let mut txn = alice.cli.ledger.env.write_txn().unwrap();
        let r = alice.cli.store.take_index(&mut txn).unwrap();
        alice.cli.store.mark_reward(&mut txn, r).unwrap();
        txn.commit().unwrap();
        block(
            &[&alice, &other],
            &[pay_to(&alice, r, 50 * COIN), pay(&alice, 2 * COIN)],
            &[],
        );
        assert_eq!(
            outputs(&alice),
            [
                "0 50 1 1 immature: 2 more blocks (mining reward)",
                "1 2 1 1 spendable",
                "2 outputs, 52 coins"
            ]
        );
        assert!(
            run(&alice, "info")
                .unwrap()
                .contains("immature    50 coins in 1 output")
        );
        let dir = std::env::temp_dir().join(format!("r-cli-m-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("x").display().to_string();
        assert!(
            run(&alice, &format!("send 10 0.01 {file}"))
                .unwrap_err()
                .contains("not enough")
        );
        block(&[&alice, &other], &[pay_to(&other, 0, COIN)], &[]);
        block(&[&alice, &other], &[pay_to(&other, 1, COIN)], &[]);
        assert_eq!(spendable(&alice), "52");
        run(&alice, &format!("send 10 0.01 {file}")).unwrap();
        assert!(outputs(&alice)[0].contains("locked by payment"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reorg_takes_away_what_it_undoes() {
        let g = genesis();
        let (alice, side) = (node(&g), node(&g));
        block(&[&alice], &[pay(&alice, 5 * COIN)], &[]);
        assert_eq!(spendable(&alice), "5");
        // A heavier branch without it.
        let b1 = block(&[&side], &[pay_to(&side, 0, COIN)], &[]);
        let b2 = block(&[&side], &[pay_to(&side, 1, COIN)], &[]);
        alice.cli.ledger.add(&b1.0, &b1.1).unwrap();
        assert_eq!(spendable(&alice), "5");
        alice.cli.ledger.add(&b2.0, &b2.1).unwrap();
        assert_eq!(spendable(&alice), "0");
        assert_eq!(outputs(&alice), Vec::<String>::new());
        // New blocks are still scanned.
        block(&[&alice, &side], &[pay(&alice, 7 * COIN)], &[]);
        assert_eq!(spendable(&alice), "7");
    }

    #[test]
    fn a_restored_wallet_finds_its_outputs_and_skips_used_keys() {
        let g = genesis();
        let entropy = [42; 32];
        let alice = node_with(&g, entropy, false);
        // Far-apart key indices, each within the scan gap of the last.
        let created = [
            pay_to(&alice, 3, COIN),
            pay_to(&alice, 900, 2 * COIN),
            pay_to(&alice, 1800, 4 * COIN),
        ];
        block(&[&alice], &created, &[]);
        let copy = node_with(&g, entropy, true);
        let (h, b) = {
            let txn = alice.cli.ledger.env.read_txn().unwrap();
            let s = &alice.cli.ledger.store;
            (
                s.header(&txn, 1).unwrap().unwrap(),
                s.block(&txn, 1).unwrap().unwrap(),
            )
        };
        copy.cli.ledger.add(&h, &b).unwrap();
        // Not caught up yet: it looks, but stays in restore mode. What it
        // finds may be rewards, so they mature from their blocks.
        copy.cli.synced.store(false, Ordering::Relaxed);
        assert_eq!(spendable(&copy), "0");
        assert!(
            run(&copy, "info")
                .unwrap()
                .contains("immature    7 coins in 3 outputs")
        );
        let txn = copy.cli.ledger.env.read_txn().unwrap();
        assert!(copy.cli.store.restoring(&txn).unwrap());
        drop(txn);
        copy.cli.synced.store(true, Ordering::Relaxed);
        assert_eq!(spendable(&copy), "0");
        assert!(
            outputs(&copy)[0]
                .ends_with("immature: 2 more blocks (found on restore: may be a mining reward)")
        );
        let txn = copy.cli.ledger.env.read_txn().unwrap();
        assert!(!copy.cli.store.restoring(&txn).unwrap());
        assert_eq!(copy.cli.store.next_index(&txn).unwrap(), 1801 + GAP);
        drop(txn);
        // The words print bare, so they paste straight back into a restore.
        let seed = run(&copy, "seed").unwrap();
        let words = seed.lines().nth(1).unwrap();
        assert_eq!(words.split(' ').count(), crate::mnemonic::WORDS);
        assert!(crate::mnemonic::from_words(words).is_ok());
        // Two more blocks: three confirmations, mature.
        block(&[&alice, &copy], &[], &[]);
        block(&[&alice, &copy], &[], &[]);
        assert_eq!(spendable(&copy), "7");
        assert_eq!(
            outputs(&copy)[0],
            "3 1 1 3 spendable (found on restore: may be a mining reward)"
        );
    }

    #[test]
    fn only_a_slate_we_sent_unchanged_is_finalized() {
        let g = genesis();
        let (alice, bob) = (node(&g), node(&g));
        block(
            &[&alice, &bob],
            &[pay(&alice, 5 * COIN), pay(&alice, 5 * COIN)],
            &[],
        );
        let dir = std::env::temp_dir().join(format!("r-cli-m-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |n: &str| dir.join(n).display().to_string();
        let (slate, response) = (path("s"), path("s.response"));
        run(&alice, &format!("send 3 0.01 {slate}")).unwrap();
        run(&bob, &format!("receive {slate}")).unwrap();
        let good = Response::from_text(&read(&response).unwrap()).unwrap();
        let finalize = |r: Response, name: &str| {
            std::fs::write(path(name), r.to_text()).unwrap();
            run(&alice, &format!("finalize {}", path(name)))
        };
        // Changed on the way: the amount, the fee, the output's amount.
        for (r, name) in [
            (Response { amount: 4 * COIN, ..good }, "amount"),
            (Response { fee: 0, ..good }, "fee"),
        ] {
            let e = finalize(r, name).unwrap_err();
            assert!(e.starts_with("modified slate"), "{e}");
        }
        let mut output = good.output;
        output.amount = 2 * COIN;
        let e = finalize(Response { output, ..good }, "output").unwrap_err();
        assert!(e.starts_with("modified slate"), "{e}");
        // Not ours.
        let e = finalize(Response { id: [9; 16], ..good }, "unknown").unwrap_err();
        assert!(e.starts_with("unknown slate"), "{e}");
        // As sent: it goes through.
        assert!(
            run(&alice, &format!("finalize {response}"))
                .unwrap()
                .contains("signed and submitted")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_wallet_restored_on_a_fast_synced_node_knows_its_heights() {
        let g = genesis();
        let entropy = [77; 32];
        let alice = node_with(&g, entropy, false);
        let mut blocks = vec![block(&[&alice], &[pay_to(&alice, 0, COIN)], &[])];
        blocks.push(block(&[&alice], &[], &[]));
        blocks.push(block(&[&alice], &[pay_to(&alice, 1, 2 * COIN)], &[]));
        blocks.push(block(&[&alice], &[], &[]));
        // A new node fast-syncs from block 2, and restores the wallet.
        let copy = node_with(&g, entropy, true);
        let (from, to) = (&alice.cli.ledger, &copy.cli.ledger);
        let snap = from.snapshot(2).unwrap().unwrap();
        crate::ledger::tests::download_state(from, to, &snap.base);
        to.install(&snap.parent, snap.parent_work, &snap.base, &blocks[1].1)
            .unwrap();
        for (h, b) in &blocks[2..] {
            to.add(h, b).unwrap();
        }
        // The output from block 3 has its height; the one from below the
        // base gets the base's (it's at least that old).
        assert_eq!(
            outputs(&copy),
            [
                "0 1 2 3 spendable (found on restore: may be a mining reward)",
                "1 2 3 2 immature: 1 more block (found on restore: may be a mining reward)",
                "2 outputs, 3 coins"
            ]
        );
    }

    #[test]
    fn a_payment_goes_through_slate_files() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let (alice, bob) = (node(&genesis), node(&genesis));
        block(
            &[&alice, &bob],
            &[pay(&alice, 5 * COIN), pay(&alice, 3 * COIN)],
            &[],
        );
        assert_eq!(spendable(&alice), "8");
        assert_eq!(spendable(&bob), "0");

        let dir = std::env::temp_dir().join(format!("r-cli-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("pay").display().to_string();
        let response = format!("{file}.response");

        // Too much, bad numbers.
        assert!(
            run(&alice, &format!("send 9 0.01 {file}"))
                .unwrap_err()
                .contains("not enough")
        );
        assert!(run(&alice, &format!("send x 0.01 {file}")).is_err());
        assert!(!std::path::Path::new(&file).exists());

        run(&alice, &format!("send 6 0.01 {file}")).unwrap();
        assert_eq!(spendable(&alice), "0");
        assert!(
            run(&alice, &format!("send 1 0.01 {file}"))
                .unwrap_err()
                .contains("exists")
        );
        run(&bob, &format!("receive {file}")).unwrap();
        assert!(
            run(&bob, "info")
                .unwrap()
                .contains("pending     6 coins in 1 output ")
        );
        assert!(
            run(&bob, "txs")
                .unwrap()
                .contains("receiving 6 coins: waiting for the sender")
        );
        // A response for another amount is refused.
        let wrong = format!("{file}.wrong");
        let mut r = Response::from_text(&read(&response).unwrap()).unwrap();
        r.amount = 7 * COIN;
        std::fs::write(&wrong, r.to_text()).unwrap();
        assert!(run(&alice, &format!("finalize {wrong}")).is_err());

        let out = run(&alice, &format!("finalize {response}")).unwrap();
        assert!(out.contains("submitted"), "{out}");
        assert!(matches!(alice.events.try_recv(), Ok(Event::NewTx(_))));
        // (Bob's node relays it too, in practice.)
        let bytes = alice.cli.mempool.get(&alice.cli.mempool.ids()[0]).unwrap();
        bob.cli.mempool.add(&*bob.cli.ledger, &bytes).unwrap();
        assert!(
            run(&bob, "txs")
                .unwrap()
                .contains("receiving 6 coins: submitted")
        );
        assert_eq!(alice.cli.mempool.size(&*alice.cli.ledger).0, 1);
        assert!(
            run(&alice, &format!("finalize {response}"))
                .unwrap_err()
                .contains("already")
        );
        let id = hex(&r.id);
        assert!(
            run(&alice, &format!("cancel {id}"))
                .unwrap_err()
                .contains("signed")
        );
        assert!(run(&alice, "txs").unwrap().contains("waiting for a block"));
        assert!(
            run(&alice, "info")
                .unwrap()
                .contains("change      1.99 coins")
        );

        // Mined: Bob has 6, Alice her change.
        let (_, tx, _) = alice
            .cli
            .mempool
            .select(&*alice.cli.ledger, usize::MAX)
            .remove(0);
        let created: Vec<OutputRecord> = tx
            .body
            .outputs
            .iter()
            .map(|o| o.record().unwrap())
            .collect();
        let spent: Vec<_> = tx.body.inputs.iter().map(|i| i.commitment).collect();
        block(&[&alice, &bob], &created, &spent);
        assert_eq!(spendable(&bob), "6");
        assert!(
            run(&bob, "info")
                .unwrap()
                .contains("pending     0 coins in 0 outputs")
        );
        assert_eq!(spendable(&alice), "1.99");
        assert_eq!(alice.cli.mempool.size(&*alice.cli.ledger).0, 0);
        assert!(run(&alice, "txs").unwrap().contains("confirmed"));
        assert_eq!(run(&alice, "txs").unwrap(), "no payments in progress");

        // A payment cancelled before it's finalized frees its coins.
        let file2 = dir.join("pay2").display().to_string();
        let out = run(&bob, &format!("send 2 0.001 {file2}")).unwrap();
        assert_eq!(spendable(&bob), "0");
        let id = out.split_whitespace().nth(1).unwrap();
        run(&bob, &format!("cancel {}", &id[..6])).unwrap();
        assert_eq!(spendable(&bob), "6");
        assert_eq!(run(&bob, "quit"), Ok("quit".into()));
        assert!(run(&bob, "frobnicate").is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
