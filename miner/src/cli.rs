//! The node's command line: the wallet, and a look at the node, typed at its
//! prompt while it runs (its log goes to a file).
//!
//! Payments go through files, as in Grin ([`crate::wallet`]):
//!
//! - `send <amount> <fee> <file>`: picks coins, locks them, and writes the
//!   slate to `<file>` for the receiver;
//! - `receive <file>`: makes an output for the slate's amount with new
//!   keys and writes it to `<file>.response` for the sender;
//! - `finalize <file>`: signs the transaction with the receiver's output and
//!   submits it to the mempool, which relays it.
//!
//! A payment waiting for a block can be re-signed with other one-time keys
//! of its coins (each output has [`crate::transaction::KEY_LEAVES`]) to
//! replace it in the mempool: `bump <id> <fee>` raises its fee (out of the
//! change), and `cancel <id> <fee>` pays its coins back to the wallet
//! instead. Either replacement must pay a higher fee.
//!
//! `info` shows the balance, with the coins received but not yet in a block
//! as pending; `txs` lists the payments in progress, sent and received (and
//! drops the confirmed ones); `cancel <id>` unlocks the coins of a payment
//! not yet finalized, or stops tracking one being received.

use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::ledger::Ledger;
use crate::mempool::{Chainstate, Mempool, Refused};
use crate::net::node::Event;
use crate::output::{Amount, Output, format_coins, parse_coins};
use crate::transaction::{
    Condition, Input, MAX_KEY_DEPTH, PREIMAGE_LEN, SigMode, Signer, Spend, Transaction, TxBody,
    hashlock,
};
use crate::wallet::{
    Coin, KEY_SEED_LEN, Leaves, Locked, Owned, Received, Response, Sent, SignerRecord, Slate,
    SlateId, Wallet, WalletStore, conditions_from_text, hex, new_preimage, output_to_hex, unhex,
};

/// Inputs in one payment, at most (each adds about 2.3 KB).
const MAX_INPUTS: usize = 32;
/// Blocks a spent output is remembered for, in case a reorg unspends it.
const KEEP_GONE: u64 = 100;
/// New outputs of ours in each `debug-fill` transaction: the coins it
/// spends multiply for the next round.
const FILL_OURS: usize = 4;
/// Mining rewards' maturity by default, in confirmations.
pub const DEFAULT_MATURITY: u64 = 6;

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
  bump <id> <fee>              re-sign a payment waiting for a block with a
                               higher fee (taken from its change)
  cancel <id> [<fee>]          drop a payment not yet finalized (or stop
                               tracking one being received); a signed one
                               needs a higher fee, and pays its coins back
  status                       the node: tip, sync, mempool
  peers [all]                  connected peers, dials, and the address book
                               (all: every address in it)
  signer <depth>               a new signing key for conditions: a tree of
                               2^depth one-time keys (one per state); prints
                               its key, for a condition's \"key\" line
  preimage                     a new hashlock preimage: prints its hash, for
                               a condition's \"hashlock\" line
  payout <amount>              a new output of ours for <amount>: prints it,
                               for a \"payout\" line after our key in a
                               protected branch (one per state signed)
  lock <amount> <fee> <file>   pay <amount> to an output with the conditions
                               in <file> (one \"branch\" line each, then
                               key (up to 16, each optionally followed by
                               a payout line: given for all, they protect
                               the branch), threshold, state, mode,
                               after-height, after-blocks, hashlock lines);
                               writes <file>.locked for the others
  import <file>                watch an output locked by someone else
                               (a .locked file)
  locked                       the locked outputs we watch: their branches,
                               which keys are ours, their timelocks, and
                               preimages others' spends revealed
  unlock <id> <branch> <fee> [<file>] [without=<m>,..]
                               spend a locked output (<id>: its first hex
                               digits) by its branch (from 1) to a new
                               output of ours; if others must sign, writes
                               the transaction to <file> for them; in a
                               protected branch, members (from 1) given
                               with without= don't sign and get their
                               payouts
  cosign <file>                sign a transaction from unlock where our
                               keys are needed; submits it once complete,
                               else writes <file>.cosigned
  debug-fill <txs> <fee>       testing (--debug-no-proof-validation only):
                               fill blocks with up to <txs> transactions of
                               ~100 KB, each spending one coin to 4 new
                               outputs of ours and ~1300 empty ones
  debug-split <txs> <fee>      testing: split up to <txs> coins into
                               ~1350 coins of ours each
  debug-spend <txs> <fee>      testing: up to <txs> typical transactions to
                               ourselves, 2 or 3 inputs and as many outputs
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
    /// Logs to the node's log.
    pub log: Box<dyn Fn(String) + Send + Sync>,
    /// `--debug-no-proof-validation`: allows `debug-fill`.
    pub debug: bool,
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

/// How many more signatures `tx`'s condition inputs need.
fn missing_signatures(tx: &Transaction) -> usize {
    tx.body
        .inputs
        .iter()
        .filter_map(|i| match &i.spend {
            Spend::Condition {
                condition, signers, ..
            } => Some(signing_members(condition, tx).saturating_sub(signers.len())),
            Spend::Key { .. } => None,
        })
        .sum()
}

/// How many of `condition`'s members are to sign in `tx`: its threshold,
/// or if it's protected, every member whose payout `tx` doesn't pay.
fn signing_members(condition: &Condition, tx: &Transaction) -> usize {
    (0..condition.keys.len())
        .filter(|&m| is_signing(condition, tx, m))
        .count()
        .min(if condition.payouts.is_some() {
            usize::MAX
        } else {
            condition.threshold
        })
}

/// Whether member `m` of `condition` is to sign in `tx`: unless it's
/// protected and `tx` pays `m`'s payout.
fn is_signing(condition: &Condition, tx: &Transaction, m: usize) -> bool {
    if condition.payouts.is_none() {
        return true;
    }
    let payout = condition.payout(m);
    !tx.body
        .outputs
        .iter()
        .any(|o| o.commitment().and_then(|c| c.to_field()) == Some(payout))
}

const PARTIAL_HEADER: &str = "r partial transaction 1";

/// A transaction still missing signers, as a file.
fn partial_text(tx: &Transaction) -> String {
    format!("{PARTIAL_HEADER}\ntx {}\n", hex(&tx.to_bytes()))
}

fn partial_from_text(text: &str) -> Result<Transaction, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    if lines.next() != Some(PARTIAL_HEADER) {
        return Err(format!(
            "not a partial transaction (expected \"{PARTIAL_HEADER}\" first)"
        ));
    }
    lines
        .next()
        .and_then(|l| l.strip_prefix("tx "))
        .and_then(|h| unhex(h.trim()))
        .and_then(|b| Transaction::from_bytes(&b))
        .ok_or("bad transaction".into())
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
            ["peers"] => self.peers(false)?,
            ["peers", "all"] => self.peers(true)?,
            ["send", amount, fee, file] => {
                let amount = parse_coins(amount).ok_or("bad amount")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.send(amount, fee, file)?
            }
            ["receive", file] => self.receive(file)?,
            ["finalize", file] => self.finalize(file)?,
            ["txs"] => self.txs()?,
            ["cancel", id] => self.cancel(id, None)?,
            ["cancel", id, fee] => {
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.cancel(id, Some(fee))?
            }
            ["debug-split" | "debug-spend", txs, fee] => {
                let txs = txs.parse().map_err(|_| "bad transaction count")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                if words[0] == "debug-split" {
                    self.debug_split(txs, fee)?
                } else {
                    self.debug_spend(txs, fee)?
                }
            }
            ["debug-fill", txs, fee] => {
                let txs = txs.parse().map_err(|_| "bad transaction count")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.debug_fill(txs, fee)?
            }
            ["bump", id, fee] => {
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.bump(id, fee)?
            }
            ["signer", depth] => {
                let depth = depth.parse().map_err(|_| "bad depth")?;
                self.new_signer(depth)?
            }
            ["preimage"] => self.new_preimage()?,
            ["payout", amount] => {
                let amount = parse_coins(amount).filter(|&a| a > 0).ok_or("bad amount")?;
                self.new_payout(amount)?
            }
            ["lock", amount, fee, file] => {
                let amount = parse_coins(amount).ok_or("bad amount")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                self.lock(amount, fee, file)?
            }
            ["import", file] => self.import(file)?,
            ["locked"] => self.locked()?,
            ["unlock", id, branch, fee, rest @ ..] if rest.len() <= 2 => {
                let branch: usize = branch.parse().map_err(|_| "bad branch")?;
                let fee = parse_coins(fee).ok_or("bad fee")?;
                let (without, files): (Vec<&str>, Vec<&str>) =
                    rest.iter().partition(|a| a.starts_with("without="));
                if without.len() > 1 || files.len() > 1 {
                    return Err("unlock <id> <branch> <fee> [<file>] [without=<m>,..]".into());
                }
                let without = match without.first() {
                    Some(w) => w["without=".len()..]
                        .split(',')
                        .map(|m| m.parse::<usize>().ok().filter(|&m| m >= 1).map(|m| m - 1))
                        .collect::<Option<Vec<_>>>()
                        .ok_or("bad members (without=1,3: members from 1)")?,
                    None => vec![],
                };
                self.unlock(id, branch, fee, files.first().copied(), &without)?
            }
            ["cosign", file] => self.cosign(file)?,
            [cmd, ..] => return Err(format!("unknown or incomplete command \"{cmd}\"; try help")),
        };
        Ok(Outcome::Text(text))
    }

    /// Brings the wallet's outputs up to the tip. After a reorg it first
    /// drops what came from blocks no longer on the main chain. Then it
    /// checks each new main-chain block's outputs for its tag (one hash
    /// each), or, when blocks are missing (pruned, or below a fast-sync
    /// base), the whole state. Outputs gone from the state (spent) are kept for
    /// [`KEEP_GONE`] blocks, in case a reorg brings them back.
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
        // The lowest block still stored.
        let base = l.store.bodies_from(&txn).map_err(err)?;
        // A restored wallet looks from the start. Outputs it finds may be
        // mining rewards: they mature.
        let start = if restoring {
            1
        } else {
            fork.map_or(1, |h| h + 1)
        };
        let whole_state = base > 0 && start < base;
        let mut found: Vec<Owned> = Vec::new();
        let mut hashes = Vec::new();
        let owned = |r, amount, record, height, txn: &heed::RoTxn| -> Result<Owned, String> {
            Ok(Owned {
                r,
                amount,
                record,
                height,
                reward: restoring || self.store.is_reward(txn, &r).map_err(err)?,
                gone_since: 0,
            })
        };
        // The blocks there are: from the start, or (when older ones are
        // missing) from the lowest stored.
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
                if let Some((r, amount)) = self.wallet.identify(record) {
                    found.push(owned(r, amount, *record, h, &txn)?);
                }
            }
            hashes.push((h, crate::hash::to_bytes(&hash)));
        }
        if whole_state {
            // The state has every unspent output; one no block above the
            // base made was made at or below it.
            let outputs = l.tree.outputs(&txn).map_err(err)?;
            for coin in self.wallet.scan(&outputs) {
                if !found.iter().any(|o| o.record == coin.record) {
                    found.push(owned(coin.r, coin.amount, coin.record, base, &txn)?);
                }
            }
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
            // Found on restore: earlier signatures by its keys may exist
            // that we don't know of.
            let c = &o.record.commitment;
            if restoring && old.is_none() {
                let leaves = self.store.leaves(&txn, c).map_err(err)?;
                self.store
                    .set_leaves(
                        &mut txn,
                        c,
                        Leaves {
                            unknown: true,
                            ..leaves
                        },
                    )
                    .map_err(err)?;
            }
        }
        for (h, hash) in &hashes {
            self.store.set_block_hash(&mut txn, *h, hash).map_err(err)?;
        }
        self.store.set_synced(&mut txn, tip.height).map_err(err)?;
        if restoring && self.synced.load(Ordering::Relaxed) {
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
                    self.store
                        .remove_leaves(&mut txn, &o.record.commitment)
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
                known_reward: self.store.is_reward(&txn, &o.r).map_err(err)?,
                coin: Coin {
                    r: o.r,
                    position,
                    amount: o.amount,
                    record: o.record,
                },
                owned: o,
                confirmations,
                state,
            });
        }
        out.sort_by_key(|h| (h.owned.height, std::cmp::Reverse(h.coin.amount), h.coin.r));
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
            "{:<8}  {:>16}  {:>7}  {:>6}  {}",
            "output", "coins", "height", "confs", "status"
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
                "{:<8}  {:>16}  {:>7}  {:>6}  {status}{}",
                &hex(&h.coin.record.commitment.0)[..8],
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

    /// The network thread's peers: connected ones with their tips, dials
    /// in progress, and the address book (counts, or every entry).
    fn peers(&self, all: bool) -> Result<String, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.events
            .send(Event::Peers(tx))
            .map_err(|_| "the network isn't running")?;
        let r = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|_| "the network didn't answer")?;
        let mut out = format!("connected: {}\n", r.connected.len());
        if !r.connected.is_empty() {
            out += &format!(
                "  {:<22}  {:<3}  {:>8}  {:<22}  {}\n",
                "address", "dir", "height", "accepts peers on", "idle"
            );
        }
        for p in &r.connected {
            out += &format!(
                "  {:<22}  {:<3}  {:>8}  {:<22}  {}s\n",
                p.addr.to_string(),
                if p.outbound { "out" } else { "in" },
                p.tip.height,
                p.listen.map_or("-".into(), |a| a.to_string()),
                p.idle
            );
        }
        if !r.dialing.is_empty() {
            let list: Vec<String> = r.dialing.iter().map(|a| a.to_string()).collect();
            out += &format!("dialing: {}\n", list.join(", "));
        }
        let worked = r.book.iter().filter(|b| b.verified).count();
        out += &format!(
            "address book: {} ({worked} worked, {} untried)",
            r.book.len(),
            r.book.len() - worked
        );
        if all {
            for b in &r.book {
                out += &format!(
                    "\n  {:<22}  {}",
                    b.addr.to_string(),
                    match (b.verified, b.failures) {
                        (true, 0) => "worked".to_string(),
                        (true, f) => format!("worked; {f} failed dial(s) since"),
                        (false, 0) => "untried".to_string(),
                        (false, f) => format!("untried; {f} failed dial(s)"),
                    }
                );
            }
        }
        Ok(out)
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

    fn send(&self, amount: Amount, fee: Amount, file: &str) -> Result<String, String> {
        if amount == 0 {
            return Err("the amount must be more than 0".into());
        }
        if Path::new(file).exists() {
            return Err(format!("{file} exists already"));
        }
        let need = amount.checked_add(fee).ok_or("too much")?;
        // The largest spendable coins first.
        let mut spendable: Vec<Coin> = self
            .holdings()?
            .into_iter()
            .filter(|h| h.state == Use::Spendable)
            .map(|h| h.coin)
            .collect();
        spendable.sort_by_key(|c| std::cmp::Reverse(c.amount));
        let (mut picked, mut total) = (Vec::new(), 0u64);
        for c in spendable {
            if total >= need || picked.len() == MAX_INPUTS {
                break;
            }
            total += c.amount;
            picked.push(c);
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
        let change = total - need;
        let change = if change > 0 {
            Some((self.wallet.new_keys().map_err(err)?.r, change))
        } else {
            None
        };
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let sent = Sent {
            id,
            amount,
            fee,
            coins: picked,
            change,
            created: now(),
            cancelled: false,
            tx: None,
        };
        self.store.put_sent(&mut txn, &sent).map_err(err)?;
        txn.commit().map_err(err)?;
        if let Err(e) = write_new(file, &Slate { id, amount, fee }.to_text()) {
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
        let keys = self.wallet.new_keys().map_err(err)?;
        let output = keys.pay(slate.amount);
        let commitment = output.commitment().ok_or("couldn't make the output")?;
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store
            .put_received(
                &mut txn,
                &Received {
                    id: slate.id,
                    r: keys.r,
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
        let mut sent = self.find(&hex(&response.id))?.ok_or(
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
        let mut outputs = vec![response.output];
        if let Some((r, amount)) = sent.change {
            outputs.push(self.wallet.keys_for(&r).pay(amount));
        }
        let bytes = self.sign(&mut sent, outputs)?;
        Ok(self.submitted(&sent, &bytes, "signed"))
    }

    /// Signs a spend of `sent`'s coins to `outputs`, each coin with a leaf
    /// of its key tree that hasn't signed (as far as we know: we record
    /// ours, and look in the mempool for others), and keeps it as `sent`'s
    /// transaction. The leaves and the transaction are stored before the
    /// signatures go anywhere, so no key ever signs twice.
    fn sign(&self, sent: &mut Sent, outputs: Vec<Output>) -> Result<Vec<u8>, String> {
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let tx = self.sign_in(&mut txn, &sent.coins, outputs)?;
        let bytes = tx.to_bytes();
        sent.tx = Some(bytes.clone());
        self.store.put_sent(&mut txn, sent).map_err(err)?;
        txn.commit().map_err(err)?;
        Ok(bytes)
    }

    /// Testing only (`--debug-no-proof-validation`): fills blocks. Spends up
    /// to `txs` spendable coins (the largest first), each in its own
    /// transaction paying `fee`: [`FILL_OURS`] new outputs of ours, sharing
    /// the rest, and as many zero-amount outputs to random keys (which no
    /// one can spend) as fit in [`MAX_TX_BYTES`]. Those are cheap to make;
    /// ours each take a key tree.
    fn debug_fill(&self, txs: usize, fee: Amount) -> Result<String, String> {
        use crate::transaction::{INPUT_BYTES, MAX_TX_BYTES, OUTPUT_BYTES, SIGNATURE_BYTES};
        if !self.debug {
            return Err("debug-fill needs --debug-no-proof-validation".into());
        }
        let mut picked: Vec<Coin> = self
            .holdings()?
            .into_iter()
            .filter(|h| h.state == Use::Spendable && h.coin.amount > fee)
            .map(|h| h.coin)
            .collect();
        picked.sort_by_key(|c| std::cmp::Reverse(c.amount));
        picked.truncate(txs);
        if picked.is_empty() {
            return Err(format!(
                "no spendable coin is worth more than the fee {}",
                coins(fee)
            ));
        }
        let outputs_per_tx = (MAX_TX_BYTES - 4 - INPUT_BYTES - SIGNATURE_BYTES) / OUTPUT_BYTES;
        let mut rng = crate::rng::Rng::from_os().map_err(err)?;
        let (mut sent, mut outputs, mut bytes, mut refused) = (0, 0, 0, Vec::new());
        for coin in &picked {
            let value = coin.amount - fee;
            let ours = FILL_OURS.min(value as usize);
            let mut outs = Vec::with_capacity(outputs_per_tx);
            for k in 0..ours {
                let share = value / ours as u64 + u64::from(k == 0) * (value % ours as u64);
                outs.push(self.wallet.new_keys().map_err(err)?.pay(share));
            }
            while outs.len() < outputs_per_tx {
                let mut key = [crate::babybear::BabyBear::ZERO; crate::hash::HASH_LEN];
                rng.fill(&mut key);
                let mut nonce = [0u8; crate::output::NONCE_LEN];
                for chunk in nonce.chunks_exact_mut(4) {
                    chunk.copy_from_slice(&rng.next_element().as_u32().to_le_bytes());
                }
                outs.push(Output {
                    hash: crate::hash::to_bytes(&key),
                    amount: 0,
                    nonce,
                });
            }
            let n = outs.len();
            let mut txn = self.ledger.env.write_txn().map_err(err)?;
            let tx = self.sign_in(&mut txn, std::slice::from_ref(coin), outs)?;
            txn.commit().map_err(err)?;
            let tx = tx.to_bytes();
            match self.submit(&tx) {
                Ok(_) => {
                    sent += 1;
                    outputs += n;
                    bytes += tx.len();
                }
                Err(e) => refused.push(e.to_string()),
            }
        }
        let mut out = format!(
            "submitted {sent} transaction{}: {outputs} outputs ({} ours), {} KB",
            if sent == 1 { "" } else { "s" },
            sent * FILL_OURS,
            bytes / 1000
        );
        if !refused.is_empty() {
            out += &format!("; {} refused: {}", refused.len(), refused.join("; "));
        }
        Ok(out)
    }

    /// Testing: the spendable coins worth more than `min`, the largest
    /// first.
    fn debug_coins(&self, min: Amount) -> Result<Vec<Coin>, String> {
        if !self.debug {
            return Err("debug commands need --debug-no-proof-validation".into());
        }
        let mut coins: Vec<Coin> = self
            .holdings()?
            .into_iter()
            .filter(|h| h.state == Use::Spendable && h.coin.amount > min)
            .map(|h| h.coin)
            .collect();
        coins.sort_by_key(|c| std::cmp::Reverse(c.amount));
        Ok(coins)
    }

    /// Testing: signs one transaction per group of coins (leaves recorded
    /// first, then built in parallel), paying `fee` and the rest to `n(group)`
    /// new outputs of ours, and submits them. Says how long each part took.
    fn debug_send(
        &self,
        groups: Vec<Vec<Coin>>,
        fee: Amount,
        n: impl Fn(&[Coin]) -> usize + Sync,
    ) -> Result<String, String> {
        let t = std::time::Instant::now();
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let mut leaves = Vec::with_capacity(groups.len());
        for g in &groups {
            leaves.push(self.take_leaves(&mut txn, g)?);
        }
        txn.commit().map_err(err)?;
        let built = crate::par::par_map(groups.len(), 1, |i| {
            let g = &groups[i];
            let value: Amount = g.iter().map(|c| c.amount).sum::<Amount>() - fee;
            let k = n(g).min(value as usize).max(1);
            let mut outputs = Vec::with_capacity(k);
            for j in 0..k {
                let share = value / k as u64 + u64::from(j == 0) * (value % k as u64);
                outputs.push(self.wallet.new_keys().map_err(err)?.pay(share));
            }
            Ok::<_, String>(self.signed(g, &leaves[i], outputs)?.to_bytes())
        });
        let built: Vec<Vec<u8>> = built.into_iter().collect::<Result<_, _>>()?;
        let build_secs = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let (mut ok, mut refused) = (0, Vec::new());
        for tx in &built {
            match self.submit(tx) {
                Ok(_) => ok += 1,
                Err(e) => refused.push(e.to_string()),
            }
        }
        let inputs: usize = groups.iter().map(Vec::len).sum();
        let bytes: usize = built.iter().map(Vec::len).sum();
        let outputs: usize = built
            .iter()
            .filter_map(|b| Transaction::from_bytes(b))
            .map(|t| t.body.outputs.len())
            .sum();
        let mut out = format!(
            "built {} transactions ({inputs} inputs, {outputs} outputs, {:.1} MB) in {build_secs:.1} s; \
             the mempool took {ok} in {:.1} s",
            built.len(),
            bytes as f64 / 1e6,
            t.elapsed().as_secs_f64()
        );
        if let Some(e) = refused.first() {
            out += &format!(" ({} refused, as: {e})", refused.len());
        }
        Ok(out)
    }

    /// Testing: spends up to `txs` coins, each to as many new coins of ours
    /// as fit in a transaction (~1350), paying `fee`.
    fn debug_split(&self, txs: usize, fee: Amount) -> Result<String, String> {
        use crate::transaction::{INPUT_BYTES, MAX_TX_BYTES, OUTPUT_BYTES, SIGNATURE_BYTES};
        let mut picked = self.debug_coins(fee)?;
        picked.truncate(txs);
        if picked.is_empty() {
            return Err(format!(
                "no spendable coin is worth more than the fee {}",
                coins(fee)
            ));
        }
        let per_tx = (MAX_TX_BYTES - 4 - INPUT_BYTES - SIGNATURE_BYTES) / OUTPUT_BYTES;
        let groups = picked.into_iter().map(|c| vec![c]).collect();
        self.debug_send(groups, fee, |_| per_tx)
    }

    /// Testing: up to `txs` typical transactions to ourselves, alternately
    /// spending 2 and 3 coins to as many new coins, each paying `fee`.
    fn debug_spend(&self, txs: usize, fee: Amount) -> Result<String, String> {
        let mut coins = self.debug_coins(0)?.into_iter();
        let mut groups = Vec::new();
        while groups.len() < txs {
            let k = 2 + groups.len() % 2;
            let g: Vec<Coin> = coins.by_ref().take(k).collect();
            if g.len() < k || g.iter().map(|c| c.amount).sum::<Amount>() <= fee + k as u64 {
                break;
            }
            groups.push(g);
        }
        if groups.is_empty() {
            return Err("not enough spendable coins".into());
        }
        self.debug_send(groups, fee, |g| g.len())
    }

    /// Signs a transaction spending `coins` to `outputs`, each coin with a
    /// leaf of its key tree that hasn't signed (as far as we know: we record
    /// ours, and look in the mempool for others), recording the leaves in
    /// `txn`, which the caller commits before the signatures go anywhere.
    fn sign_in(
        &self,
        txn: &mut heed::RwTxn,
        coins: &[Coin],
        outputs: Vec<Output>,
    ) -> Result<Transaction, String> {
        let leaves = self.take_leaves(txn, coins)?;
        self.signed(coins, &leaves, outputs)
    }

    /// Picks and records (in `txn`) the leaf each of `coins` signs with
    /// next, for [`Cli::sign_in`].
    fn take_leaves(&self, txn: &mut heed::RwTxn, coins: &[Coin]) -> Result<Vec<u8>, String> {
        // Testing wallets' trees have one real key, which signs every time
        // (reusing a one-time key: fine on a test network).
        if self.wallet.cheap_trees() {
            return Ok(vec![0; coins.len()]);
        }
        let mut rng = crate::rng::Rng::from_os().map_err(err)?;
        let mut leaves = Vec::with_capacity(coins.len());
        for c in coins {
            let commitment = &c.record.commitment;
            let mut known = self.store.leaves(txn, commitment).map_err(err)?;
            if let Some(tx) = self.mempool.spender(commitment) {
                for i in tx
                    .body
                    .inputs
                    .iter()
                    .filter(|i| i.commitment == *commitment)
                {
                    if let Spend::Key { leaf, .. } = i.spend {
                        known = known.with(leaf);
                    }
                }
            }
            let leaf = known.next(rng.next_element().as_u32()).ok_or(format!(
                "every key of output {} has signed: it can't be re-signed",
                &hex(&commitment.0)[..8]
            ))?;
            self.store
                .set_leaves(txn, commitment, known.with(leaf))
                .map_err(err)?;
            leaves.push(leaf);
        }
        Ok(leaves)
    }

    /// The transaction spending `coins`, with `leaves` (recorded already),
    /// to `outputs`.
    fn signed(
        &self,
        coins: &[Coin],
        leaves: &[u8],
        outputs: Vec<Output>,
    ) -> Result<Transaction, String> {
        let (keys, inputs) = coins
            .iter()
            .zip(leaves)
            .map(|(c, &leaf)| self.wallet.input(c, leaf))
            .unzip();
        Transaction::sign(TxBody::new(inputs, outputs), keys).map_err(err)
    }

    /// Submits `sent`'s transaction `bytes`, saying what happened.
    fn submitted(&self, sent: &Sent, bytes: &[u8], what: &str) -> String {
        match self.submit(bytes) {
            Ok(id) => format!(
                "payment {} {what} and submitted: transaction {}",
                hex(&sent.id),
                &hex(&crate::hash::to_bytes(&id))[..16]
            ),
            Err(e) => format!(
                "payment {} {what}, but the mempool turned it away: {e}\n\
                 it stays on record (txs) and is submitted again",
                hex(&sent.id)
            ),
        }
    }

    /// A signed payment not yet in a block, for replacing.
    fn replaceable(&self, prefix: &str, fee: Amount) -> Result<Sent, String> {
        let sent = self.find(prefix)?.ok_or("no payment of ours has that id")?;
        if sent.tx.is_none() {
            return Err("that payment isn't signed yet: cancel it and send again".into());
        }
        if !self.unconfirmed(&sent) {
            return Err("that payment is in a block already".into());
        }
        if fee <= sent.fee {
            return Err(format!(
                "a replacement must pay a higher fee than its {}",
                coins(sent.fee)
            ));
        }
        // The mempool also wants more per byte: a replacement is at most
        // the same size (the same inputs, no more outputs), so a higher fee
        // pays that too.
        Ok(sent)
    }

    fn bump(&self, prefix: &str, fee: Amount) -> Result<String, String> {
        let mut sent = self.replaceable(prefix, fee)?;
        if sent.cancelled {
            return Err(
                "that payment is being cancelled: cancel it again with a higher fee \
                        to raise the cancellation's"
                    .into(),
            );
        }
        let more = fee - sent.fee;
        let change = sent.change.map_or(0, |(_, a)| a);
        if more > change {
            return Err(format!(
                "its change ({} coins) can't cover {} more in fees",
                coins(change),
                coins(more)
            ));
        }
        let old = Transaction::from_bytes(sent.tx.as_ref().expect("signed"))
            .ok_or("its stored transaction doesn't decode")?;
        // The receiver's output, unchanged; the change, smaller, with new
        // keys.
        let mut outputs = vec![old.body.outputs[0]];
        sent.change = if change > more {
            let keys = self.wallet.new_keys().map_err(err)?;
            outputs.push(keys.pay(change - more));
            Some((keys.r, change - more))
        } else {
            None
        };
        sent.fee = fee;
        let bytes = self.sign(&mut sent, outputs)?;
        (self.log)(format!(
            "wallet: payment {} re-signed with fee {}",
            hex(&sent.id),
            coins(fee)
        ));
        Ok(self.submitted(&sent, &bytes, &format!("re-signed with fee {}", coins(fee))))
    }

    /// Replaces signed payment `prefix` with a transaction paying its coins
    /// back to the wallet, with fee `fee`.
    fn cancel_signed(&self, prefix: &str, fee: Amount) -> Result<String, String> {
        let mut sent = self.replaceable(prefix, fee)?;
        let total: Amount = sent.coins.iter().map(|c| c.amount).sum();
        if fee >= total {
            return Err(format!(
                "the fee must be less than its coins' {}",
                coins(total)
            ));
        }
        let keys = self.wallet.new_keys().map_err(err)?;
        sent.change = Some((keys.r, total - fee));
        sent.fee = fee;
        sent.cancelled = true;
        let bytes = self.sign(&mut sent, vec![keys.pay(total - fee)])?;
        (self.log)(format!(
            "wallet: payment {} replaced by one paying its coins back (fee {})",
            hex(&sent.id),
            coins(fee)
        ));
        Ok(self.submitted(
            &sent,
            &bytes,
            &format!(
                "replaced by a transaction paying its coins back (fee {}); if the payment \
                 gets into a block first, it stands",
                coins(fee)
            ),
        ))
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
                    // Cancelled: which spend won?
                    let back = s
                        .change
                        .and_then(|(r, a)| self.wallet.keys_for(&r).pay(a).commitment());
                    let owned = {
                        let txn = self.ledger.env.read_txn().map_err(err)?;
                        self.store.owned(&txn).map_err(err)?
                    };
                    let returned = back.is_some_and(|c| {
                        self.ledger.unspent(&c).is_some()
                            || owned.iter().any(|o| o.record.commitment == c)
                    });
                    match (s.cancelled, returned) {
                        (true, true) => "cancelled: its coins came back (now dropped from this \
                                         list)"
                            .to_string(),
                        (true, false) => "went through before its cancellation could (now \
                                          dropped from this list)"
                            .to_string(),
                        _ => "confirmed (now dropped from this list)".to_string(),
                    }
                }
                Some(tx) => {
                    let id = Transaction::id_of(tx);
                    if self.mempool.knows(&id) && s.cancelled {
                        "being cancelled: its coins paid back, waiting for a block".to_string()
                    } else if self.mempool.knows(&id) {
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

    fn cancel(&self, prefix: &str, fee: Option<Amount>) -> Result<String, String> {
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
            return match fee {
                Some(fee) => self.cancel_signed(prefix, fee),
                None => Err(format!(
                    "that payment is signed already: cancelling it replaces it with a \
                     transaction paying its coins back, which must pay a higher fee than its \
                     {} (cancel <id> <fee>)",
                    coins(sent.fee)
                )),
            };
        }
        self.forget(&sent.id)?;
        (self.log)(format!("wallet: payment {} cancelled", hex(&sent.id)));
        Ok(format!(
            "payment {} cancelled; its coins are spendable again",
            hex(&sent.id)
        ))
    }

    /// Makes a signer key tree of `2^depth` keys and prints its key.
    fn new_signer(&self, depth: usize) -> Result<String, String> {
        if depth > MAX_KEY_DEPTH {
            return Err(format!("the depth is at most {MAX_KEY_DEPTH}"));
        }
        let mut id = [0u8; KEY_SEED_LEN];
        let mut rng = crate::rng::Rng::from_os().map_err(err)?;
        for chunk in id.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_element().as_u32().to_le_bytes());
        }
        let t = std::time::Instant::now();
        let tree = self.wallet.signer_tree(&id, depth);
        let record = SignerRecord {
            id,
            tree,
            used: vec![],
        };
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store.put_signer(&mut txn, &record).map_err(err)?;
        txn.commit().map_err(err)?;
        let root = crate::hash::to_bytes(&record.tree.root());
        (self.log)(format!(
            "wallet: signer key {} of {} keys made in {:.1} s",
            &hex(&root)[..16],
            1u64 << depth,
            t.elapsed().as_secs_f64()
        ));
        Ok(format!(
            "key {}\n(a tree of {} one-time keys: each state, or each spend, signs with \
             its own)",
            hex(&root),
            1u64 << depth
        ))
    }

    /// Makes and keeps a hashlock preimage, and prints its hash.
    fn new_preimage(&self) -> Result<String, String> {
        let p = new_preimage().map_err(err)?;
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store.put_preimage(&mut txn, &p).map_err(err)?;
        txn.commit().map_err(err)?;
        Ok(format!(
            "hashlock {}\n(the preimage stays in the wallet: unlocking a branch with this \
             hashlock publishes it)",
            hex(&crate::hash::to_bytes(&hashlock(&p)))
        ))
    }

    /// Makes an output of ours for `amount`, to name as our payout in a
    /// protected branch, and prints it. Nothing is kept: if a quorum pays
    /// it, the wallet finds it like any output of ours.
    fn new_payout(&self, amount: Amount) -> Result<String, String> {
        let o = self.wallet.new_keys().map_err(err)?.pay(amount);
        Ok(format!(
            "payout {}\n(an output of ours for {} coins: put it after our key in a protected \
             branch; make a new one for each state)",
            output_to_hex(&o),
            coins(amount)
        ))
    }

    /// Pays `amount` (plus `fee`) from our coins to an output with the
    /// conditions in `file`, as a payment of ours (see txs), and writes
    /// the output's description to `<file>.locked`.
    fn lock(&self, amount: Amount, fee: Amount, file: &str) -> Result<String, String> {
        let (conditions, payouts) = conditions_from_text(&read(file)?)?;
        let out_file = format!("{file}.locked");
        if Path::new(&out_file).exists() {
            return Err(format!("{out_file} exists already"));
        }
        let need = amount.checked_add(fee).ok_or("too much")?;
        let mut spendable: Vec<Coin> = self
            .holdings()?
            .into_iter()
            .filter(|h| h.state == Use::Spendable)
            .map(|h| h.coin)
            .collect();
        spendable.sort_by_key(|c| std::cmp::Reverse(c.amount));
        let (mut picked, mut total) = (Vec::new(), 0u64);
        for c in spendable {
            if total >= need || picked.len() == MAX_INPUTS {
                break;
            }
            total += c.amount;
            picked.push(c);
        }
        if total < need {
            return Err(format!(
                "not enough coins: {} needed, {} spendable",
                coins(need),
                coins(total)
            ));
        }
        // Our nonce (so the output is told apart from others), the
        // conditions' key tree.
        let locked = Locked {
            output: Output {
                hash: crate::transaction::KeyTree::of_conditions(&conditions).output_hash(),
                ..self.wallet.new_keys().map_err(err)?.pay(amount)
            },
            conditions,
            payouts,
            created: now(),
            seen: None,
        };
        let change = total - need;
        let change = if change > 0 {
            Some((self.wallet.new_keys().map_err(err)?.r, change))
        } else {
            None
        };
        let mut id: SlateId = [0; 16];
        let mut rng = crate::rng::Rng::from_os().map_err(err)?;
        for chunk in id.chunks_exact_mut(4) {
            chunk.copy_from_slice(&rng.next_element().as_u32().to_le_bytes());
        }
        let mut sent = Sent {
            id,
            amount,
            fee,
            coins: picked,
            change,
            created: now(),
            cancelled: false,
            tx: None,
        };
        let mut outputs = vec![locked.output];
        if let Some((r, amount)) = sent.change {
            outputs.push(self.wallet.keys_for(&r).pay(amount));
        }
        write_new(&out_file, &locked.to_text())?;
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        self.store.put_locked(&mut txn, &locked).map_err(err)?;
        txn.commit().map_err(err)?;
        let bytes = self.sign(&mut sent, outputs)?;
        let id_text = &hex(&locked.output.commitment().ok_or("bad output")?.0)[..8];
        Ok(format!(
            "{}\nlocked output {id_text}: {} coins, {} branch{}; described in {out_file}",
            self.submitted(&sent, &bytes, "signed"),
            coins(amount),
            locked.conditions.len(),
            if locked.conditions.len() == 1 {
                ""
            } else {
                "es"
            },
        ))
    }

    /// Watches the locked output described in `file`.
    fn import(&self, file: &str) -> Result<String, String> {
        let mut locked = Locked::from_text(&read(file)?)?;
        locked.created = now();
        let id = hex(&locked.output.commitment().ok_or("bad output")?.0);
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let known = self.store.locked(&txn).map_err(err)?;
        if known.iter().any(|l| l.output == locked.output) {
            return Err(format!("output {} is watched already", &id[..8]));
        }
        self.store.put_locked(&mut txn, &locked).map_err(err)?;
        txn.commit().map_err(err)?;
        Ok(format!(
            "watching locked output {}: {} coins",
            &id[..8],
            coins(locked.output.amount)
        ))
    }

    /// The locked outputs we watch, brought up to date: where they are,
    /// their branches, and preimages their spends revealed.
    fn locked(&self) -> Result<String, String> {
        let l = &self.ledger;
        let mut txn = l.env.write_txn().map_err(err)?;
        let tip = l.store.tip(&txn).map_err(err)?.ok_or("no chain")?;
        let signers = self.store.signers(&txn).map_err(err)?;
        let ours = |k: &crate::hash::Hash| signers.iter().any(|s| s.tree.root() == *k);
        let mut lines = Vec::new();
        for mut lk in self.store.locked(&txn).map_err(err)? {
            let c = lk.output.commitment().ok_or("bad output")?;
            let id = hex(&c.0)[..8].to_string();
            let now_in = l.tree.get(&txn, &c).map_err(err)?;
            let state = match (now_in, lk.seen) {
                (Some((slot, leaf)), _) => {
                    if lk.seen != Some((slot, leaf.height)) {
                        lk.seen = Some((slot, leaf.height));
                        self.store.put_locked(&mut txn, &lk).map_err(err)?;
                    }
                    format!("unspent in slot {slot}, from block {}", leaf.height)
                }
                (None, Some((slot, height))) => {
                    let revealed = self.revealed_preimages(&mut txn, &lk, slot, height)?;
                    let mut s = format!("spent (was in slot {slot})");
                    for p in revealed {
                        s.push_str(&format!("; its spend revealed preimage {}", hex(&p)));
                    }
                    s
                }
                (None, None) => match self.find_slot(&txn, &c)? {
                    // Made and spent since we last looked.
                    Some((slot, height)) => {
                        lk.seen = Some((slot, height));
                        self.store.put_locked(&mut txn, &lk).map_err(err)?;
                        let revealed = self.revealed_preimages(&mut txn, &lk, slot, height)?;
                        let mut s = format!("spent (was in slot {slot})");
                        for p in revealed {
                            s.push_str(&format!("; its spend revealed preimage {}", hex(&p)));
                        }
                        s
                    }
                    None => "not in a block yet".to_string(),
                },
            };
            lines.push(format!("{id} {} coins: {state}", coins(lk.output.amount)));
            for (b, cond) in lk.conditions.iter().enumerate() {
                let keys: Vec<&str> = cond
                    .keys
                    .iter()
                    .map(|k| if ours(k) { "ours" } else { "theirs" })
                    .collect();
                let mut rules = vec![if cond.threshold == keys.len() {
                    format!("keys {}", keys.join(" + "))
                } else {
                    format!("{} of keys {}", cond.threshold, keys.join(", "))
                }];
                if cond.payouts.is_some() {
                    let amounts: Vec<String> = (0..cond.keys.len())
                        .map(|m| {
                            lk.payout(&cond.payout(m))
                                .map_or("?".into(), |o| coins(o.amount))
                        })
                        .collect();
                    rules.push(format!("protected (payouts {})", amounts.join(", ")));
                }
                if let Some(m) = cond.min_state {
                    rules.push(format!("state above {m}"));
                }
                rules.push(
                    match cond.mode {
                        SigMode::All => "mode all",
                        SigMode::AnyoneCanPay => "mode anyone-can-pay",
                        SigMode::AnyPrevout => "mode any-prevout",
                    }
                    .into(),
                );
                let from = lk
                    .seen
                    .map_or(0, |(_, h)| h as u64 + cond.after_blocks as u64)
                    .max(cond.after_height as u64);
                if cond.after_height > 0 || cond.after_blocks > 0 {
                    rules.push(match lk.seen {
                        Some(_) if from <= tip.height + 1 => {
                            format!("unlocked (from block {from})")
                        }
                        Some(_) => format!("locked until block {from}"),
                        None => format!(
                            "locked until block {} and {} blocks after it's in one",
                            cond.after_height, cond.after_blocks
                        ),
                    });
                }
                if let Some(h) = &cond.hashlock {
                    let known = self.store.preimage(&txn, h).map_err(err)?.is_some();
                    rules.push(format!(
                        "hashlock (preimage {})",
                        if known { "known" } else { "unknown" }
                    ));
                }
                lines.push(format!("  branch {}: {}", b + 1, rules.join(", ")));
            }
        }
        txn.commit().map_err(err)?;
        if lines.is_empty() {
            return Ok("no locked outputs".into());
        }
        Ok(lines.join("\n"))
    }

    /// The slot and height of the stored block output with commitment `c`
    /// (a block's outputs fill the slots below its header's counter, in
    /// order).
    fn find_slot(
        &self,
        txn: &heed::RoTxn,
        c: &crate::output::Commitment,
    ) -> Result<Option<(u64, u32)>, String> {
        let l = &self.ledger;
        let tip = l.store.tip(txn).map_err(err)?.ok_or("no chain")?;
        let from = l.store.bodies_from(txn).map_err(err)?.max(1);
        for h in (from..=tip.height).rev() {
            let Some(block) = l.store.block(txn, h).map_err(err)? else {
                continue;
            };
            if let Some(i) = block.outputs.iter().position(|o| o.commitment == *c) {
                let hash = l
                    .store
                    .hash_at(txn, h)
                    .map_err(err)?
                    .ok_or("missing block")?;
                let header = l
                    .store
                    .header_by_hash(txn, &hash)
                    .map_err(err)?
                    .ok_or("missing header")?;
                let first = header.state.size - block.outputs.len() as u64;
                return Ok(Some((first + i as u64, h as u32)));
            }
        }
        Ok(None)
    }

    /// Preimages of `lk`'s hashlocks published by the block that spent
    /// it (from slot `slot`, made at `height`), kept in the wallet.
    fn revealed_preimages(
        &self,
        txn: &mut heed::RwTxn,
        lk: &Locked,
        slot: u64,
        height: u32,
    ) -> Result<Vec<[u8; PREIMAGE_LEN]>, String> {
        let l = &self.ledger;
        let locks: Vec<_> = lk.conditions.iter().filter_map(|c| c.hashlock).collect();
        let mut out = Vec::new();
        if locks.is_empty() {
            return Ok(out);
        }
        let tip = l.store.tip(txn).map_err(err)?.ok_or("no chain")?;
        let from = (height as u64 + 1).max(l.store.bodies_from(txn).map_err(err)?);
        for h in from..=tip.height {
            let Some(block) = l.store.block(txn, h).map_err(err)? else {
                continue;
            };
            if !block.inputs.contains(&slot) {
                continue;
            }
            for o in &block.outputs {
                let p: [u8; PREIMAGE_LEN] = o.nonce[..PREIMAGE_LEN].try_into().expect("length");
                if locks.contains(&hashlock(&p)) {
                    self.store.put_preimage(txn, &p).map_err(err)?;
                    out.push(p);
                }
            }
            break;
        }
        Ok(out)
    }

    /// The locked output whose commitment (in hex) starts with `prefix`.
    fn find_locked(&self, prefix: &str) -> Result<Locked, String> {
        if prefix.len() < 4 {
            return Err("give at least 4 characters of the id".into());
        }
        let txn = self.ledger.env.read_txn().map_err(err)?;
        let mut matches: Vec<Locked> = self
            .store
            .locked(&txn)
            .map_err(err)?
            .into_iter()
            .filter(|l| {
                l.output
                    .commitment()
                    .is_some_and(|c| hex(&c.0).starts_with(&prefix.to_lowercase()))
            })
            .collect();
        match matches.len() {
            0 => Err("no locked output we watch has that id (see locked)".into()),
            1 => Ok(matches.pop().expect("one")),
            _ => Err("more than one locked output matches; give more of the id".into()),
        }
    }

    /// Picks (and records, in `txn`) the key of ours for condition key
    /// `root` (member `member` of its key set) that signs next: the smallest unused index above the
    /// condition's state number. `None` if `root` isn't ours.
    fn take_signer(
        &self,
        txn: &mut heed::RwTxn,
        member: u8,
        root: &crate::hash::Hash,
        above: Option<u32>,
    ) -> Result<Option<(Signer, crate::wots::SecretKey)>, String> {
        let Some(mut record) = self
            .store
            .signers(txn)
            .map_err(err)?
            .into_iter()
            .find(|s| s.tree.root() == *root)
        else {
            return Ok(None);
        };
        let index = record.next_index(above).ok_or(
            "every key of our signer key that this condition allows has signed: make a \
             larger one (signer)",
        )?;
        record.used.push(index);
        self.store.put_signer(txn, &record).map_err(err)?;
        let (sk, pubkey) = self.wallet.signer_key(&record.id, index);
        Ok(Some((
            Signer {
                member,
                pubkey,
                index,
                path: record.tree.path(index),
            },
            sk,
        )))
    }

    /// Spends locked output `prefix` by branch `branch` (from 1) to a new
    /// output of ours, paying `fee`: signs with our keys, and submits the
    /// transaction if no others are needed, else writes it to `file`. In a
    /// protected branch, members `without` (from 0) don't sign, and are
    /// paid their payouts.
    fn unlock(
        &self,
        prefix: &str,
        branch: usize,
        fee: Amount,
        file: Option<&str>,
        without: &[usize],
    ) -> Result<String, String> {
        let lk = self.find_locked(prefix)?;
        let condition = branch
            .checked_sub(1)
            .and_then(|b| lk.conditions.get(b))
            .ok_or(format!("it has branches 1 to {}", lk.conditions.len()))?
            .clone();
        let leaf = (branch - 1) as u8;
        let commitment = lk.output.commitment().ok_or("bad output")?;
        let l = &self.ledger;
        let mut txn = l.env.write_txn().map_err(err)?;
        let tip = l.store.tip(&txn).map_err(err)?.ok_or("no chain")?;
        let (position, created) = match l.tree.get(&txn, &commitment).map_err(err)? {
            Some((slot, leaf)) => (slot, leaf.height),
            None => {
                return Err("it isn't unspent in the state (not in a block yet, or spent)".into());
            }
        };
        let from =
            (created as u64 + condition.after_blocks as u64).max(condition.after_height as u64);
        if from > tip.height + 1 {
            return Err(format!(
                "that branch is locked until block {from} (the next is {})",
                tip.height + 1
            ));
        }
        // The payouts of the members who won't sign.
        let mut payouts = Vec::new();
        if !without.is_empty() {
            if condition.payouts.is_none() {
                return Err("that branch isn't protected: every signer it needs signs".into());
            }
            let mut members = without.to_vec();
            members.sort_unstable();
            members.dedup();
            if members.last().is_some_and(|&m| m >= condition.keys.len()) {
                return Err(format!(
                    "that branch has members 1 to {}",
                    condition.keys.len()
                ));
            }
            if condition.keys.len() - members.len() < condition.threshold {
                return Err(format!(
                    "that branch needs {} of its {} members to sign",
                    condition.threshold,
                    condition.keys.len()
                ));
            }
            for m in members {
                payouts.push(
                    lk.payout(&condition.payout(m))
                        .ok_or(format!("we don't know member {}'s payout output", m + 1))?,
                );
            }
        }
        let paid: Amount = payouts.iter().map(|p| p.amount).sum();
        let amount = lk
            .output
            .amount
            .checked_sub(fee)
            .and_then(|a| a.checked_sub(paid))
            .filter(|&a| a > 0)
            .ok_or("the fee and payouts must be less than the output's amount")?;
        // The output: ours; starting its nonce with the preimage, if the
        // branch needs one.
        let preimage = match &condition.hashlock {
            Some(h) => Some(
                self.store
                    .preimage(&txn, h)
                    .map_err(err)?
                    .ok_or("that branch has a hashlock whose preimage we don't know")?,
            ),
            None => None,
        };
        let keys = match preimage {
            Some(p) => {
                let mut r = [0u8; KEY_SEED_LEN];
                r[..PREIMAGE_LEN].copy_from_slice(&p);
                r[PREIMAGE_LEN] = crate::rng::Rng::from_os()
                    .map_err(err)?
                    .next_element()
                    .as_u32() as u8;
                self.wallet.keys_for(&r)
            }
            None => self.wallet.new_keys().map_err(err)?,
        };
        let mut outputs = vec![keys.pay(amount)];
        outputs.extend(payouts);
        let tree = lk.key_tree();
        let body = TxBody {
            inputs: vec![Input {
                commitment,
                position,
                amount: lk.output.amount,
                nonce: lk.output.nonce,
                spend: Spend::Condition {
                    condition: condition.clone(),
                    leaf,
                    path: tree.path(leaf as usize),
                    signers: vec![],
                    preimage,
                },
            }],
            signed_outputs: if condition.mode == SigMode::All {
                0
            } else {
                outputs.len()
            },
            outputs,
        };
        let mut tx = Transaction {
            body,
            signatures: vec![],
        };
        let added = self.add_signatures(&mut txn, &mut tx)?;
        let missing = missing_signatures(&tx);
        if missing > 0 {
            let file = file.ok_or("others must sign too: give a file to write it to")?;
            // Our keys' use is recorded before any signature leaves us (a
            // file already there is refused first, so no key is spent).
            if Path::new(file).exists() {
                return Err(format!("{file} exists already"));
            }
            txn.commit().map_err(err)?;
            write_new(file, &partial_text(&tx))?;
            return Ok(format!(
                "signed with {added} key{} of ours; {missing} more signature{} needed: send \
                 {file} to the other signers (cosign {file})",
                if added == 1 { "" } else { "s" },
                if missing == 1 { "" } else { "s" },
            ));
        }
        txn.commit().map_err(err)?;
        tx.validate()
            .map_err(|e| format!("the transaction isn't valid: {e}"))?;
        match self.submit(&tx.to_bytes()) {
            Ok(id) => Ok(format!(
                "unlocked: transaction {} submitted",
                &hex(&crate::hash::to_bytes(&id))[..16]
            )),
            Err(e) => Err(format!("the mempool turned it away: {e}")),
        }
    }

    /// Adds our signatures to the partial transaction in `file`.
    fn cosign(&self, file: &str) -> Result<String, String> {
        let mut tx = partial_from_text(&read(file)?)?;
        let mut txn = self.ledger.env.write_txn().map_err(err)?;
        let added = self.add_signatures(&mut txn, &mut tx)?;
        if added == 0 {
            return Err("none of the signatures it still needs can be ours".into());
        }
        let coins_in: Amount = tx.body.inputs.iter().map(|i| i.amount).sum();
        let coins_out: Amount = tx.body.outputs.iter().map(|o| o.amount).sum();
        let summary = format!(
            "{} coins in, {} coins out",
            coins(coins_in),
            coins(coins_out)
        );
        let s = if added == 1 { "" } else { "s" };
        let missing = missing_signatures(&tx);
        if missing > 0 {
            let out_file = format!("{file}.cosigned");
            // Our keys' use is recorded before any signature leaves us.
            if Path::new(&out_file).exists() {
                return Err(format!("{out_file} exists already"));
            }
            txn.commit().map_err(err)?;
            write_new(&out_file, &partial_text(&tx))?;
            return Ok(format!(
                "added {added} signature{s} ({summary}); {missing} still needed: {out_file}"
            ));
        }
        txn.commit().map_err(err)?;
        tx.validate()
            .map_err(|e| format!("the transaction isn't valid: {e}"))?;
        match self.submit(&tx.to_bytes()) {
            Ok(id) => Ok(format!(
                "added {added} signature{s} ({summary}): transaction {} submitted",
                &hex(&crate::hash::to_bytes(&id))[..16]
            )),
            Err(e) => Err(format!("signed, but the mempool turned it away: {e}")),
        }
    }

    /// Adds signers of ours to `tx`'s condition inputs, each up to its
    /// threshold (protected: each member not paid out), for members of its
    /// key set whose keys are ours and that haven't signed (in member
    /// order), recording each key's use in `txn` before it signs. Returns
    /// how many signed.
    fn add_signatures(&self, txn: &mut heed::RwTxn, tx: &mut Transaction) -> Result<usize, String> {
        let messages: Vec<_> = (0..tx.body.inputs.len())
            .map(|i| tx.body.message(i))
            .collect::<Option<_>>()
            .ok_or("its transaction isn't well formed")?;
        let wanted: Vec<Vec<bool>> = tx
            .body
            .inputs
            .iter()
            .map(|i| match i.condition() {
                Some(c) => (0..c.keys.len()).map(|m| is_signing(c, tx, m)).collect(),
                None => vec![],
            })
            .collect();
        let limits: Vec<usize> = tx
            .body
            .inputs
            .iter()
            .map(|i| i.condition().map_or(0, |c| signing_members(c, tx)))
            .collect();
        let mut added = 0;
        let mut at = 0;
        for (i, input) in tx.body.inputs.iter_mut().enumerate() {
            if let Spend::Condition {
                condition, signers, ..
            } = &mut input.spend
            {
                for (member, &wanted) in wanted[i].iter().enumerate() {
                    if signers.len() >= limits[i] {
                        break;
                    }
                    if !wanted || signers.iter().any(|s| s.member as usize == member) {
                        continue;
                    }
                    let Some((signer, sk)) = self.take_signer(
                        txn,
                        member as u8,
                        &condition.keys[member],
                        condition.min_state,
                    )?
                    else {
                        continue;
                    };
                    let p = signers.partition_point(|s| (s.member as usize) < member);
                    signers.insert(p, signer);
                    tx.signatures.insert(at + p, sk.sign(&messages[i]));
                    added += 1;
                }
            }
            at += input.signature_count();
        }
        Ok(added)
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
            log: Box::new(|_| {}),
            debug: true,
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
        let (state, mut inputs) = {
            let mut txn = l.env.write_txn().unwrap();
            let positions: Vec<u64> = spent
                .iter()
                .map(|c| l.tree.get(&txn, c).unwrap().unwrap().0)
                .collect();
            l.tree
                .apply_changes(&mut txn, tip.height + 1, created, spent)
                .unwrap();
            let s = l.tree.state(&txn).unwrap();
            txn.abort();
            (s, positions)
        };
        inputs.sort_unstable();
        let block = Block {
            inputs,
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

    /// An output of `amount` for `n`'s wallet, with key seed `r`.
    fn pay_to(n: &Node, r: u8, amount: Amount) -> OutputRecord {
        n.cli
            .wallet
            .keys_for(&[r; 16])
            .pay(amount)
            .record()
            .unwrap()
    }

    /// The `outputs` lines, without the header or each output's id.
    fn outputs(n: &Node) -> Vec<String> {
        let text = run(n, "outputs").unwrap();
        let lines: Vec<&str> = text.lines().skip(1).collect();
        lines
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let words = l.split_whitespace();
                let skip = usize::from(i + 1 < lines.len());
                words.skip(skip).collect::<Vec<_>>().join(" ")
            })
            .collect()
    }

    /// An output of `amount` for new keys of `n`'s wallet.
    fn pay(n: &Node, amount: Amount) -> OutputRecord {
        n.cli
            .wallet
            .new_keys()
            .unwrap()
            .pay(amount)
            .record()
            .unwrap()
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
    fn peers_shows_the_networks_report() {
        use crate::babybear::BabyBear;
        use crate::net::node::{BookEntry, PeerInfo, PeerReport};
        let g = genesis();
        // Taking the receiver out of a node, not destructuring it: the node
        // must drop its CLI (holding the database) before its database.
        let mut lone = node(&g);
        // No network thread: the request goes unanswered.
        drop(std::mem::replace(&mut lone.events, channel().1));
        assert_eq!(
            run(&lone, "peers").unwrap_err(),
            "the network isn't running"
        );

        let mut n = node(&g);
        let events = std::mem::replace(&mut n.events, channel().1);
        let a: std::net::SocketAddr = "1.2.3.4:3737".parse().unwrap();
        let b: std::net::SocketAddr = "5.6.7.8:9000".parse().unwrap();
        let network = std::thread::spawn(move || {
            for _ in 0..2 {
                let Ok(Event::Peers(reply)) = events.recv() else {
                    panic!("not a peers request");
                };
                reply
                    .send(PeerReport {
                        connected: vec![PeerInfo {
                            addr: a,
                            outbound: true,
                            listen: Some(a),
                            tip: crate::net::wire::Tip {
                                height: 42,
                                hash: [BabyBear::ZERO; crate::hash::HASH_LEN],
                                work: 0,
                                state_from: 0,
                            },
                            idle: 3,
                        }],
                        dialing: vec![b],
                        book: vec![
                            BookEntry {
                                addr: a,
                                verified: true,
                                failures: 0,
                            },
                            BookEntry {
                                addr: b,
                                verified: false,
                                failures: 2,
                            },
                        ],
                    })
                    .unwrap();
            }
        });
        let out = run(&n, "peers").unwrap();
        assert!(out.starts_with("connected: 1\n"), "{out}");
        assert!(out.contains("1.2.3.4:3737") && out.contains("out") && out.contains("42"));
        assert!(out.contains("dialing: 5.6.7.8:9000"));
        assert!(
            out.ends_with("address book: 2 (1 worked, 1 untried)"),
            "{out}"
        );
        let all = run(&n, "peers all").unwrap();
        assert!(
            all.contains("5.6.7.8:9000            untried; 2 failed dial(s)"),
            "{all}"
        );
        network.join().unwrap();
    }

    #[test]
    fn rewards_mature_and_outputs_are_listed() {
        let g = genesis();
        let (alice, other) = (node(&g), node(&g));
        // A reward of ours, and a payment.
        let mut txn = alice.cli.ledger.env.write_txn().unwrap();
        alice.cli.store.mark_reward(&mut txn, &[7; 16]).unwrap();
        txn.commit().unwrap();
        block(
            &[&alice, &other],
            &[pay_to(&alice, 7, 50 * COIN), pay(&alice, 2 * COIN)],
            &[],
        );
        assert_eq!(
            outputs(&alice),
            [
                "50 1 1 immature: 2 more blocks (mining reward)",
                "2 1 1 spendable",
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
    fn a_restored_wallet_finds_its_outputs() {
        let g = genesis();
        let entropy = [42; 32];
        let alice = node_with(&g, entropy, false);
        // Outputs with random keys: nothing to count up to.
        let created = [
            pay(&alice, COIN),
            pay(&alice, 2 * COIN),
            pay(&alice, 4 * COIN),
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
        // Their keys' history is unknown: their first spends take random
        // leaves of the upper half.
        for c in &created {
            assert!(copy.cli.store.leaves(&txn, &c.commitment).unwrap().unknown);
            assert!(
                !alice
                    .cli
                    .store
                    .leaves(&alice.cli.ledger.env.read_txn().unwrap(), &c.commitment)
                    .unwrap()
                    .unknown
            );
        }
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
            "4 1 3 spendable (found on restore: may be a mining reward)"
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
        let dir = std::env::temp_dir().join(format!("r-cli-f-{}", std::process::id()));
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
            (
                Response {
                    amount: 4 * COIN,
                    ..good
                },
                "amount",
            ),
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
        let e = finalize(
            Response {
                id: [9; 16],
                ..good
            },
            "unknown",
        )
        .unwrap_err();
        assert!(e.starts_with("unknown slate"), "{e}");
        // As sent: it goes through.
        assert!(
            run(&alice, &format!("finalize {response}"))
                .unwrap()
                .contains("signed and submitted")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Mines the mempool's transactions into a block on every node.
    fn mine(nodes: &[&Node]) {
        let l = &nodes[0].cli.ledger;
        let (mut created, mut spent) = (vec![], vec![]);
        for (_, tx, _) in nodes[0].cli.mempool.select(&**l, usize::MAX) {
            created.extend(tx.body.outputs.iter().map(|o| o.record().unwrap()));
            spent.extend(tx.body.inputs.iter().map(|i| i.commitment));
        }
        block(nodes, &created, &spent);
    }

    /// The leaf the mempool's spend of `coin` signs with.
    fn leaf_spending(n: &Node, coin: &OutputRecord) -> u8 {
        let tx = n.cli.mempool.spender(&coin.commitment).unwrap();
        tx.body.inputs[0].key_leaf().unwrap()
    }

    #[test]
    fn debug_fill_makes_full_transactions_whose_coins_multiply() {
        use crate::transaction::{MAX_TX_BYTES, OUTPUT_BYTES};
        let g = genesis();
        let mut alice = node(&g);
        block(
            &[&alice],
            &[pay(&alice, 5 * COIN), pay(&alice, 3 * COIN)],
            &[],
        );
        alice.cli.debug = false;
        assert!(
            run(&alice, "debug-fill 5 0.01")
                .unwrap_err()
                .contains("--debug-no-proof")
        );
        alice.cli.debug = true;

        let out = run(&alice, "debug-fill 5 0.01").unwrap();
        assert!(out.starts_with("submitted 2 transactions"), "{out}");
        let picked = alice.cli.mempool.select(&*alice.cli.ledger, usize::MAX);
        assert_eq!(picked.len(), 2);
        for (_, tx, _) in &picked {
            // As full as a transaction gets, with 4 outputs of ours.
            let size = tx.to_bytes().len();
            assert!(
                size <= MAX_TX_BYTES && size > MAX_TX_BYTES - OUTPUT_BYTES,
                "{size}"
            );
            assert_eq!(
                tx.body.outputs.iter().filter(|o| o.amount > 0).count(),
                FILL_OURS
            );
        }
        mine(&[&alice]);
        // Two coins became eight, less the fees.
        assert_eq!(spendable(&alice), "7.98");
        let spendable_coins = alice
            .cli
            .holdings()
            .unwrap()
            .iter()
            .filter(|h| h.state == Use::Spendable)
            .count();
        assert_eq!(spendable_coins, 2 * FILL_OURS);
    }

    #[test]
    fn debug_split_and_spend_make_typical_transactions() {
        let g = genesis();
        let mut alice = node(&g);
        alice.cli.wallet = alice.cli.wallet.with_cheap_trees();
        block(&[&alice], &[pay(&alice, 5 * COIN)], &[]);
        let out = run(&alice, "debug-split 3 0.01").unwrap();
        assert!(
            out.starts_with("built 1 transactions (1 inputs, 1354 outputs"),
            "{out}"
        );
        mine(&[&alice]);
        let out = run(&alice, "debug-spend 10 0.001").unwrap();
        assert!(
            out.starts_with("built 10 transactions (25 inputs, 25 outputs"),
            "{out}"
        );
        assert!(out.contains("the mempool took 10"), "{out}");
        let mut shapes: Vec<_> = alice
            .cli
            .mempool
            .select_up_to(&*alice.cli.ledger, usize::MAX, usize::MAX)
            .iter()
            .map(|(_, tx, _)| (tx.body.inputs.len(), tx.body.outputs.len()))
            .collect();
        shapes.sort();
        assert_eq!(shapes, [[(2, 2); 5], [(3, 3); 5]].concat());
        mine(&[&alice]);
        assert_eq!(spendable(&alice), "4.98");
    }

    #[test]
    fn a_signed_payment_can_be_bumped_or_cancelled() {
        let g = genesis();
        let (alice, bob) = (node(&g), node(&g));
        let coin = pay(&alice, 5 * COIN);
        block(&[&alice, &bob], &[coin], &[]);
        let dir = std::env::temp_dir().join(format!("r-cli-b-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let slate = dir.join("s").display().to_string();
        let out = run(&alice, &format!("send 3 0.01 {slate}")).unwrap();
        let id = out.split_whitespace().nth(1).unwrap().to_string();
        assert!(
            run(&alice, &format!("bump {id} 0.02"))
                .unwrap_err()
                .contains("isn't signed")
        );
        run(&bob, &format!("receive {slate}")).unwrap();
        run(&alice, &format!("finalize {slate}.response")).unwrap();
        assert_eq!(leaf_spending(&alice, &coin), 0);

        // A higher fee, out of the change, signed with the next leaf: it
        // replaces the first in the mempool.
        assert!(
            run(&alice, &format!("bump {id} 0.005"))
                .unwrap_err()
                .contains("higher fee")
        );
        assert!(
            run(&alice, &format!("bump {id} 3"))
                .unwrap_err()
                .contains("can't cover")
        );
        assert!(
            run(&alice, &format!("bump {id} 0.02"))
                .unwrap()
                .contains("re-signed with fee 0.02 and submitted")
        );
        assert_eq!(leaf_spending(&alice, &coin), 1);
        assert_eq!(alice.cli.mempool.size(&*alice.cli.ledger).0, 1);
        assert!(
            run(&alice, "info")
                .unwrap()
                .contains("change      1.98 coins")
        );

        // Cancelling needs a fee above the bumped one.
        assert!(
            run(&alice, &format!("cancel {id}"))
                .unwrap_err()
                .contains("cancel <id> <fee>")
        );
        assert!(run(&alice, &format!("cancel {id} 0.02")).is_err());
        // With the wallet's record of its leaves lost, the mempool's spend
        // still keeps leaf 1 from signing again.
        let mut txn = alice.cli.ledger.env.write_txn().unwrap();
        alice
            .cli
            .store
            .set_leaves(&mut txn, &coin.commitment, Leaves::default().with(0))
            .unwrap();
        txn.commit().unwrap();
        run(&alice, &format!("cancel {id} 0.03")).unwrap();
        assert_eq!(leaf_spending(&alice, &coin), 2);
        assert!(run(&alice, "txs").unwrap().contains("being cancelled"));
        let txn = alice.cli.ledger.env.read_txn().unwrap();
        assert_eq!(
            alice.cli.store.leaves(&txn, &coin.commitment).unwrap(),
            Leaves::default().with(0).with(1).with(2)
        );
        drop(txn);

        // Mined: the coins came back, less the fee.
        mine(&[&alice, &bob]);
        assert_eq!(spendable(&alice), "4.97");
        assert_eq!(spendable(&bob), "0");
        assert!(
            run(&alice, "txs")
                .unwrap()
                .contains("cancelled: its coins came back")
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A wallet last synced before blocks its node has since pruned finds
    /// what they paid it in the state.
    #[test]
    fn a_wallet_behind_pruned_blocks_scans_the_state() {
        let g = genesis();
        let mut alice = node_with(&g, [78; 32], false);
        let l = Arc::get_mut(&mut alice.cli.ledger).unwrap();
        (l.snap_sync_limit, l.keep_blocks) = (2, 2);
        assert_eq!(run(&alice, "outputs").unwrap(), "no outputs");
        block(&[&alice], &[pay_to(&alice, 0, COIN)], &[]);
        for _ in 0..4 {
            block(&[&alice], &[], &[]);
        }
        block(&[&alice], &[pay_to(&alice, 1, 2 * COIN)], &[]);
        let txn = alice.cli.ledger.env.read_txn().unwrap();
        assert_eq!(alice.cli.ledger.store.bodies_from(&txn).unwrap(), 4);
        drop(txn);
        // Block 1's output is found in the state (taking the lowest stored
        // block's height), block 6's in its block.
        assert_eq!(
            outputs(&alice),
            ["1 4 3 spendable", "2 6 1 spendable", "2 outputs, 3 coins"]
        );
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
                "1 2 3 spendable (found on restore: may be a mining reward)",
                "2 3 2 immature: 1 more block (found on restore: may be a mining reward)",
                "2 outputs, 3 coins"
            ]
        );
    }

    /// The id (first hex digits) a command's output names after `what`.
    fn id_after(text: &str, what: &str) -> String {
        let at = text
            .find(what)
            .unwrap_or_else(|| panic!("no \"{what}\" in {text}"));
        text[at + what.len()..]
            .split(|c: char| !c.is_ascii_hexdigit())
            .next()
            .unwrap()
            .to_string()
    }

    /// The primitives from the wallet: two signer keys, a preimage, two
    /// outputs locked to the same three branches (2-of-2 by state, a
    /// hashlock for Bob, a refund to Alice after 3 blocks). One is spent
    /// 2-of-2 (Alice signs, Bob cosigns); the other Bob claims with the
    /// preimage, which Alice then learns from the chain; the refund waits
    /// for its timelock.
    #[test]
    fn conditions_lock_and_unlock() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let (alice, bob) = (node(&genesis), node(&genesis));
        block(&[&alice, &bob], &[pay(&alice, 10 * COIN)], &[]);
        let dir = std::env::temp_dir().join(format!("r-cli-cond-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).display().to_string();

        let key_line = |n: &Node| {
            let out = run(n, "signer 3").unwrap();
            out.lines().next().unwrap().to_string()
        };
        let (ka, kb) = (key_line(&alice), key_line(&bob));
        let hashlock = run(&bob, "preimage")
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .to_string();
        assert!(ka.starts_with("key ") && hashlock.starts_with("hashlock "));
        let spec = format!(
            "# a channel, an HTLC for Bob, a refund for Alice\n\
             branch\n{ka}\n{kb}\nstate 2\nmode any-prevout\n\
             branch\n{kb}\n{hashlock}\nmode anyone-can-pay\n\
             branch\n{ka}\nafter-blocks 3\n"
        );
        for name in ["one", "two"] {
            std::fs::write(path(name), &spec).unwrap();
        }
        assert!(run(&alice, &format!("lock 4 0.01 {}", path("bad"))).is_err());
        let one = run(&alice, &format!("lock 4 0.01 {}", path("one"))).unwrap();
        assert!(one.contains("submitted"), "{one}");
        let one = id_after(&one, "locked output ");
        mine(&[&alice, &bob]);
        let two = run(&alice, &format!("lock 3 0.01 {}", path("two"))).unwrap();
        let two = id_after(&two, "locked output ");
        mine(&[&alice, &bob]);
        assert_eq!(spendable(&alice), "2.98");
        for name in ["one", "two"] {
            run(&bob, &format!("import {}.locked", path(name))).unwrap();
        }
        assert!(run(&bob, &format!("import {}.locked", path("one"))).is_err());
        let listed = run(&bob, "locked").unwrap();
        assert!(
            listed.contains("keys theirs + ours, state above 2, mode any-prevout"),
            "{listed}"
        );
        assert!(listed.contains("hashlock (preimage known)"), "{listed}");
        let listed = run(&alice, "locked").unwrap();
        assert!(listed.contains("hashlock (preimage unknown)"), "{listed}");
        assert!(listed.contains("locked until block 6"), "{listed}");

        // The refund is still locked; Alice can't claim the hashlock.
        let early = run(&alice, &format!("unlock {two} 3 0.01")).unwrap_err();
        assert!(early.contains("locked until block 6"), "{early}");
        let no_pre = run(&alice, &format!("unlock {two} 2 0.01")).unwrap_err();
        assert!(no_pre.contains("preimage"), "{no_pre}");

        // 2-of-2: Alice signs, Bob cosigns and submits.
        let partial = path("partial");
        let out = run(&alice, &format!("unlock {one} 1 0.01")).unwrap_err();
        assert!(out.contains("give a file"), "{out}");
        let out = run(&alice, &format!("unlock {one} 1 0.01 {partial}")).unwrap();
        assert!(out.contains("1 more signature needed"), "{out}");
        assert!(run(&alice, &format!("cosign {partial}")).is_err());
        let out = run(&bob, &format!("cosign {partial}")).unwrap();
        assert!(out.contains("submitted"), "{out}");
        mine(&[&bob, &alice]);
        assert_eq!(spendable(&alice), "6.97");

        // Bob claims the HTLC, publishing the preimage, which Alice finds.
        let out = run(&bob, &format!("unlock {two} 2 0.01")).unwrap();
        assert!(out.contains("submitted"), "{out}");
        mine(&[&bob, &alice]);
        assert_eq!(spendable(&bob), "2.99");
        let listed = run(&alice, "locked").unwrap();
        assert!(listed.contains("its spend revealed preimage"), "{listed}");
        assert!(listed.contains("hashlock (preimage known)"), "{listed}");
        assert!(run(&alice, &format!("unlock {two} 3 0.01")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A 2-of-3 lock: Alice signs, Carol cosigns while Bob is offline,
    /// and the spend goes through; a third signature isn't added.
    #[test]
    fn two_of_three_unlock_without_the_third() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let (alice, bob, carol) = (node(&genesis), node(&genesis), node(&genesis));
        block(&[&alice, &carol], &[pay(&alice, 10 * COIN)], &[]);
        let dir = std::env::temp_dir().join(format!("r-cli-2of3-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).display().to_string();
        let key = |n: &Node| {
            run(n, "signer 2")
                .unwrap()
                .lines()
                .next()
                .unwrap()
                .to_string()
        };
        let keys = [key(&alice), key(&bob), key(&carol)].join("\n");
        std::fs::write(path("spec"), format!("branch\n{keys}\nthreshold 2\n")).unwrap();
        let out = run(&alice, &format!("lock 4 0.01 {}", path("spec"))).unwrap();
        let id = id_after(&out, "locked output ");
        mine(&[&alice, &carol]);
        run(&carol, &format!("import {}.locked", path("spec"))).unwrap();
        let listed = run(&carol, "locked").unwrap();
        assert!(
            listed.contains("2 of keys theirs, theirs, ours"),
            "{listed}"
        );
        let partial = path("partial");
        let out = run(&alice, &format!("unlock {id} 1 0.01 {partial}")).unwrap();
        assert!(out.contains("1 more signature needed"), "{out}");
        let out = run(&carol, &format!("cosign {partial}")).unwrap();
        assert!(
            out.contains("added 1 signature") && out.contains("submitted"),
            "{out}"
        );
        mine(&[&carol, &alice]);
        assert_eq!(spendable(&alice), "9.98");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A protected 2-of-3: Alice and Bob spend it while Carol is offline,
    /// and Carol's wallet finds her payout.
    #[test]
    fn a_quorum_pays_the_member_who_doesnt_sign() {
        let genesis = Header {
            timestamp: now() - 1_000_000,
            state: crate::state::empty_state(),
            ..Header::default()
        };
        let (alice, bob, carol) = (node(&genesis), node(&genesis), node(&genesis));
        let all = [&alice, &bob, &carol];
        block(&all, &[pay(&alice, 10 * COIN)], &[]);
        let dir = std::env::temp_dir().join(format!("r-cli-protect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).display().to_string();
        let first = |n: &Node, cmd: &str| run(n, cmd).unwrap().lines().next().unwrap().to_string();
        let mut spec = "branch\n".to_string();
        for (n, amount) in [(&alice, 1), (&bob, 1), (&carol, 2)] {
            spec += &format!("{}\n", first(n, "signer 2"));
            spec += &format!("{}\n", first(n, &format!("payout {amount}")));
        }
        spec += "threshold 2\nstate 0\nafter-blocks 1\n";
        // Protection needs a delay.
        std::fs::write(path("nodelay"), spec.replace("after-blocks 1\n", "")).unwrap();
        let e = run(&alice, &format!("lock 4 0.01 {}", path("nodelay"))).unwrap_err();
        assert!(e.contains("after-blocks"), "{e}");
        std::fs::write(path("spec"), &spec).unwrap();
        let out = run(&alice, &format!("lock 4 0.01 {}", path("spec"))).unwrap();
        let id = id_after(&out, "locked output ");
        mine(&all);
        run(&bob, &format!("import {}.locked", path("spec"))).unwrap();
        let listed = run(&bob, "locked").unwrap();
        assert!(
            listed.contains("2 of keys theirs, ours, theirs, protected (payouts 1, 1, 2)"),
            "{listed}"
        );

        // Carol is offline: without her, she's paid. Without two, too few
        // sign.
        let partial = path("partial");
        let e = run(&alice, &format!("unlock {id} 1 0.01 {partial} without=2,3")).unwrap_err();
        assert!(e.contains("needs 2 of its 3"), "{e}");
        let out = run(&alice, &format!("unlock {id} 1 0.01 {partial} without=3")).unwrap();
        assert!(out.contains("1 more signature needed"), "{out}");
        let out = run(&bob, &format!("cosign {partial}")).unwrap();
        assert!(
            out.contains("added 1 signature") && out.contains("submitted"),
            "{out}"
        );
        mine(&[&bob, &alice, &carol]);
        // Alice: 5.99 change and 4 - 0.01 - 2; Carol her 2.
        assert_eq!(spendable(&alice), "7.98");
        assert_eq!(spendable(&carol), "2");
        let _ = std::fs::remove_dir_all(&dir);
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
