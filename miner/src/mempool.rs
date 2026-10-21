//! The mempool: transactions waiting for a block.
//!
//! A transaction gets in if it checks on its own ([`Transaction::validate`]),
//! mints nothing (any fee, even none, is fine), spends only outputs that
//! are unspent in the main chain's state at the slots its inputs sign,
//! creates no output that exists already, and shares no input or output with
//! a transaction in the pool, unless it replaces those by fee: a
//! transaction spending inputs that pool transactions spend replaces them
//! if it pays more than all of them together and more per byte than each.
//! (Wallets re-sign a spend with another of the output's one-time keys to
//! raise its fee or cancel it.) Inputs must be confirmed: an input signs its
//! output's slot, which a block assigns only when it's applied.
//!
//! The pool re-checks every transaction against the state whenever the tip
//! moves. One whose inputs are all spent now was most likely mined: it
//! moves to a list of recently mined transactions, and if a reorg undoes
//! that block (its inputs unspent again), it comes back into the pool, so
//! nothing a reorg drops is lost. Others that no longer fit are dropped. When it's full, a transaction paying more per byte pushes out
//! the ones paying least. Miners take the best-paying first
//! ([`Mempool::select`]).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::Mutex;

use crate::babybear::BabyBear;
use crate::block;
use crate::hash::{self, HASH_LEN, Hash};
use crate::ledger::Ledger;
use crate::output::{Amount, Commitment};
use crate::state::key_of;
use crate::transaction::{MAX_TX_BYTES, Transaction};

/// The pool's size, in encoded bytes, at most.
pub const MAX_POOL_BYTES: usize = 20_000_000;
/// Transactions a miner puts in one block, at most (proving time grows with
/// each one).
pub const MAX_BLOCK_TXS: usize = 100;
/// Ids of transactions turned away, remembered until the tip moves.
const REMEMBER_REJECTED: usize = 1000;
/// Recently mined transactions kept, in case a reorg undoes their block.
const KEEP_MINED: usize = 5000;

/// What the pool needs from the chain.
pub trait Chainstate {
    /// The main chain's tip hash.
    fn tip_hash(&self) -> Hash;
    /// The slot of the unspent output with commitment `c`, if there is one.
    fn unspent(&self, c: &Commitment) -> Option<u64>;
}

impl Chainstate for Ledger {
    fn tip_hash(&self) -> Hash {
        self.tip()
            .map_or([BabyBear::ZERO; HASH_LEN], |(h, _)| h.hash())
    }

    fn unspent(&self, c: &Commitment) -> Option<u64> {
        let txn = self.env.read_txn().ok()?;
        self.tree.get(&txn, c).ok().flatten().map(|(slot, _)| slot)
    }
}

/// Why a transaction wasn't added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    /// Malformed or invalid in itself: whoever sent it misbehaved.
    Invalid(String),
    /// It doesn't fit the current state or the pool's rules.
    Unfit(String),
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(e) | Self::Unfit(e) => f.write_str(e),
        }
    }
}

struct Entry {
    tx: Transaction,
    bytes: Vec<u8>,
    fee: Amount,
}

impl Entry {
    /// Whether this pays less per byte than `fee` for `len` bytes.
    fn pays_less_than(&self, fee: Amount, len: usize) -> bool {
        (self.fee as u128 * len as u128) < fee as u128 * self.bytes.len() as u128
    }
}

#[derive(Default)]
struct Pool {
    txs: HashMap<Hash, Entry>,
    /// Inputs and outputs of the pool's transactions, to their ids.
    spends: HashMap<Commitment, Hash>,
    creates: HashMap<Commitment, Hash>,
    bytes: usize,
    /// The tip the pool was last checked against.
    tip: Option<Hash>,
    rejected: HashSet<Hash>,
    rejected_order: VecDeque<Hash>,
    /// Recently mined, oldest first.
    mined: VecDeque<(Hash, Entry)>,
}

pub struct Mempool {
    capacity: usize,
    pool: Mutex<Pool>,
}

impl Default for Mempool {
    fn default() -> Self {
        Self::new()
    }
}

impl Mempool {
    pub fn new() -> Self {
        Self::with_capacity(MAX_POOL_BYTES)
    }

    /// A pool holding at most `capacity` bytes of transactions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            pool: Mutex::new(Pool::default()),
        }
    }

    fn lock(&self, chain: &impl Chainstate) -> std::sync::MutexGuard<'_, Pool> {
        let mut p = self.pool.lock().expect("mempool lock");
        p.refresh(chain);
        p
    }

    /// Adds the transaction encoded as `bytes`: its id, and whether it's new
    /// (rather than in the pool already).
    pub fn add(&self, chain: &impl Chainstate, bytes: &[u8]) -> Result<(Hash, bool), Refused> {
        let id = Transaction::id_of(bytes);
        {
            let p = self.lock(chain);
            if p.txs.contains_key(&id) {
                return Ok((id, false));
            }
            if p.rejected.contains(&id) {
                return Err(Refused::Unfit("turned away already".into()));
            }
        }
        // The expensive checks (signatures) without holding the pool.
        let checked = check(bytes);
        let mut p = self.lock(chain);
        let result = checked.and_then(|(tx, fee)| {
            // Replace by fee: pool transactions spending its inputs give way
            // if it pays more than all of them, and more per byte than each.
            let replaced = p.conflicts(&tx);
            let (sum, len) = (
                replaced.iter().map(|r| p.txs[r].fee as u128).sum::<u128>(),
                bytes.len(),
            );
            if !replaced.is_empty()
                && (fee as u128 <= sum
                    || replaced.iter().any(|r| !p.txs[r].pays_less_than(fee, len)))
            {
                return Err(Refused::Unfit(
                    "spends an output a pool transaction spends, and doesn't pay enough more \
                     to replace it (more in all, and more per byte)"
                        .into(),
                ));
            }
            let old: Vec<(Hash, Entry)> = replaced
                .iter()
                .filter_map(|r| Some((*r, p.remove(r)?)))
                .collect();
            let added = p
                .fits(chain, &tx)
                .and_then(|()| p.make_room(self.capacity, fee, len));
            if let Err(e) = added {
                for (r, e) in old {
                    p.insert(r, e.tx, e.bytes, e.fee);
                }
                return Err(e);
            }
            p.insert(id, tx, bytes.to_vec(), fee);
            Ok((id, true))
        });
        if result.is_err() {
            p.reject(id);
        }
        result
    }

    /// The encoding of transaction `id`, if it's in the pool.
    pub fn get(&self, id: &Hash) -> Option<Vec<u8>> {
        let p = self.pool.lock().expect("mempool lock");
        p.txs.get(id).map(|e| e.bytes.clone())
    }

    /// Whether `id` is in the pool or was turned away since the tip last
    /// moved.
    pub fn knows(&self, id: &Hash) -> bool {
        let p = self.pool.lock().expect("mempool lock");
        p.txs.contains_key(id) || p.rejected.contains(id)
    }

    /// The pool's transaction ids.
    pub fn ids(&self) -> Vec<Hash> {
        let p = self.pool.lock().expect("mempool lock");
        p.txs.keys().copied().collect()
    }

    /// Whether a pool transaction spends commitment `c`.
    pub fn spends(&self, c: &Commitment) -> bool {
        let p = self.pool.lock().expect("mempool lock");
        p.spends.contains_key(c)
    }

    /// The pool transaction spending commitment `c`, if any.
    pub fn spender(&self, c: &Commitment) -> Option<Transaction> {
        let p = self.pool.lock().expect("mempool lock");
        p.spends.get(c).map(|id| p.txs[id].tx.clone())
    }

    /// Whether a pool transaction creates commitment `c`.
    pub fn creates(&self, c: &Commitment) -> bool {
        let p = self.pool.lock().expect("mempool lock");
        p.creates.contains_key(c)
    }

    /// Transactions and bytes in the pool.
    pub fn size(&self, chain: &impl Chainstate) -> (usize, usize) {
        let p = self.lock(chain);
        (p.txs.len(), p.bytes)
    }

    /// Transactions for a block, the best-paying per byte first, with their
    /// ids and fees: at most [`MAX_BLOCK_TXS`], whose published lists take
    /// at most `room` bytes.
    pub fn select(&self, chain: &impl Chainstate, room: usize) -> Vec<(Hash, Transaction, Amount)> {
        let p = self.lock(chain);
        let mut entries: Vec<(&Hash, &Entry)> = p.txs.iter().collect();
        entries.sort_by(|(ia, a), (ib, b)| {
            let (x, y) = (
                a.fee as u128 * b.bytes.len() as u128,
                b.fee as u128 * a.bytes.len() as u128,
            );
            y.cmp(&x)
                .then_with(|| ia.map(|v| v.as_u32()).cmp(&ib.map(|v| v.as_u32())))
        });
        let mut used = 0;
        let mut out = Vec::new();
        for (id, e) in entries {
            if out.len() == MAX_BLOCK_TXS {
                break;
            }
            let lists = e.tx.body.inputs.len() * block::INPUT_BYTES
                + e.tx.body.outputs.len() * block::OUTPUT_BYTES;
            if used + lists > room {
                continue;
            }
            used += lists;
            out.push((*id, e.tx.clone(), e.fee));
        }
        out
    }

    /// Drops transactions `ids` (say, ones a block couldn't take).
    pub fn remove(&self, ids: &[Hash]) {
        let mut p = self.pool.lock().expect("mempool lock");
        for id in ids {
            let _ = p.remove(id);
            p.reject(*id);
        }
    }
}

/// The checks that need no state: size, encoding, signatures, no minting,
/// canonical and distinct outputs. Returns the
/// transaction and its fee.
fn check(bytes: &[u8]) -> Result<(Transaction, Amount), Refused> {
    let invalid = |e: &str| Refused::Invalid(e.into());
    if bytes.len() > MAX_TX_BYTES {
        return Err(invalid("too large"));
    }
    let tx = Transaction::from_bytes(bytes).ok_or_else(|| invalid("malformed"))?;
    if tx.body.inputs.is_empty() {
        return Err(invalid("no inputs"));
    }
    let totals = tx.validate().map_err(|e| Refused::Invalid(e.to_string()))?;
    if totals.fee() < 0 {
        return Err(invalid("pays out more than it spends"));
    }
    let fee = totals.fee() as Amount;
    let mut seen = HashSet::new();
    for o in &tx.body.outputs {
        let c = o
            .commitment()
            .ok_or_else(|| invalid("non-canonical output"))?;
        key_of(&c).map_err(|_| invalid("unusable output commitment"))?;
        if !seen.insert(c) {
            return Err(invalid("an output repeats"));
        }
    }
    Ok((tx, fee))
}

impl Pool {
    /// Re-checks every transaction if the tip moved since the last check.
    fn refresh(&mut self, chain: &impl Chainstate) {
        let tip = chain.tip_hash();
        if self.tip == Some(tip) {
            return;
        }
        self.tip = Some(tip);
        self.rejected.clear();
        self.rejected_order.clear();
        let stale: Vec<Hash> = self
            .txs
            .iter()
            .filter(|(_, e)| !state_fits(chain, &e.tx))
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            let Some(e) = self.remove(&id) else {
                continue;
            };
            let spent =
                e.tx.body
                    .inputs
                    .iter()
                    .all(|i| chain.unspent(&i.commitment).is_none());
            if spent {
                self.mined.push_back((id, e));
                if self.mined.len() > KEEP_MINED {
                    self.mined.pop_front();
                }
            }
        }
        // Mined ones whose block a reorg undid come back.
        for (id, e) in std::mem::take(&mut self.mined) {
            if self.fits(chain, &e.tx).is_ok() {
                self.insert(id, e.tx, e.bytes, e.fee);
            } else {
                self.mined.push_back((id, e));
            }
        }
    }

    /// The pool transactions spending any of `tx`'s inputs.
    fn conflicts(&self, tx: &Transaction) -> Vec<Hash> {
        let mut ids: Vec<Hash> = tx
            .body
            .inputs
            .iter()
            .filter_map(|i| self.spends.get(&i.commitment).copied())
            .collect();
        ids.sort_by_key(hash::to_bytes);
        ids.dedup();
        ids
    }

    /// Whether `tx` fits the state and conflicts with nothing in the pool.
    fn fits(&self, chain: &impl Chainstate, tx: &Transaction) -> Result<(), Refused> {
        let unfit = |e: &str| Err(Refused::Unfit(e.into()));
        if !state_fits(chain, tx) {
            return unfit(
                "spends an output that isn't unspent at that slot, or creates one that exists",
            );
        }
        if tx
            .body
            .inputs
            .iter()
            .any(|i| self.spends.contains_key(&i.commitment))
        {
            return unfit("spends an output a pool transaction spends");
        }
        let creates_known = tx.body.outputs.iter().any(|o| {
            o.commitment()
                .is_some_and(|c| self.creates.contains_key(&c))
        });
        if creates_known {
            return unfit("creates an output a pool transaction creates");
        }
        Ok(())
    }

    /// Makes room for `len` bytes paying `fee`, pushing out transactions
    /// that pay less per byte.
    fn make_room(&mut self, capacity: usize, fee: Amount, len: usize) -> Result<(), Refused> {
        if len > capacity {
            return Err(Refused::Unfit("larger than the pool".into()));
        }
        let mut out = Vec::new();
        let mut freed = 0;
        let mut by_rate: Vec<(&Hash, &Entry)> = self.txs.iter().collect();
        by_rate.sort_by(|(_, a), (_, b)| {
            (a.fee as u128 * b.bytes.len() as u128).cmp(&(b.fee as u128 * a.bytes.len() as u128))
        });
        let mut cheapest = by_rate.into_iter();
        while self.bytes - freed + len > capacity {
            match cheapest.next() {
                Some((id, e)) if e.pays_less_than(fee, len) => {
                    freed += e.bytes.len();
                    out.push(*id);
                }
                _ => {
                    return Err(Refused::Unfit(
                        "the pool is full of better-paying transactions".into(),
                    ));
                }
            }
        }
        for id in out {
            let _ = self.remove(&id);
        }
        Ok(())
    }

    fn insert(&mut self, id: Hash, tx: Transaction, bytes: Vec<u8>, fee: Amount) {
        for i in &tx.body.inputs {
            self.spends.insert(i.commitment, id);
        }
        for o in &tx.body.outputs {
            if let Some(c) = o.commitment() {
                self.creates.insert(c, id);
            }
        }
        self.bytes += bytes.len();
        self.txs.insert(id, Entry { tx, bytes, fee });
    }

    fn remove(&mut self, id: &Hash) -> Option<Entry> {
        let e = self.txs.remove(id)?;
        for i in &e.tx.body.inputs {
            self.spends.remove(&i.commitment);
        }
        for o in &e.tx.body.outputs {
            if let Some(c) = o.commitment() {
                self.creates.remove(&c);
            }
        }
        self.bytes -= e.bytes.len();
        Some(e)
    }

    fn reject(&mut self, id: Hash) {
        if self.rejected.insert(id) {
            self.rejected_order.push_back(id);
            if self.rejected_order.len() > REMEMBER_REJECTED
                && let Some(old) = self.rejected_order.pop_front()
            {
                self.rejected.remove(&old);
            }
        }
    }
}

/// Whether each input of `tx` is unspent at the slot it signs, and none of
/// its outputs exists.
fn state_fits(chain: &impl Chainstate, tx: &Transaction) -> bool {
    tx.body
        .inputs
        .iter()
        .all(|i| chain.unspent(&i.commitment) == Some(i.position))
        && tx
            .body
            .outputs
            .iter()
            .all(|o| o.commitment().is_some_and(|c| chain.unspent(&c).is_none()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::output::COIN;

    const FEE: Amount = COIN / 1000;
    use crate::output::Output;
    use crate::transaction::TxBody;
    use crate::wallet::{Coin, Wallet};
    use std::cell::RefCell;

    /// A chain that's just a set of unspent outputs and a tip.
    struct Fake {
        tip: u32,
        unspent: RefCell<HashMap<Commitment, u64>>,
    }

    impl Chainstate for Fake {
        fn tip_hash(&self) -> Hash {
            BabyBear::new_array([self.tip; HASH_LEN])
        }

        fn unspent(&self, c: &Commitment) -> Option<u64> {
            self.unspent.borrow().get(c).copied()
        }
    }

    fn wallet() -> Wallet {
        Wallet::from_seed(BabyBear::new_array([1, 2, 3, 4, 5, 6, 7, 8]))
    }

    /// Coins for keys `0..n` of `amount` each, in slots `1..=n`.
    fn chain(n: u32, amount: Amount) -> (Fake, Vec<Coin>) {
        let w = wallet();
        let coins: Vec<Coin> = (0..n)
            .map(|i| Coin {
                r: [i as u8; 16],
                position: i as u64 + 1,
                amount,
                record: w.keys_for(&[i as u8; 16]).pay(amount).record().unwrap(),
            })
            .collect();
        let unspent = coins
            .iter()
            .map(|c| (c.record.commitment, c.position))
            .collect();
        (
            Fake {
                tip: 0,
                unspent: RefCell::new(unspent),
            },
            coins,
        )
    }

    /// Spends `coins` to new keys `to`, paying `fee`.
    fn spend(coins: &[Coin], to: u8, fee: Amount) -> Vec<u8> {
        spend_with(coins, to, fee, 0)
    }

    /// [`spend`] signing with the keys at `leaf`.
    fn spend_with(coins: &[Coin], to: u8, fee: Amount, leaf: u8) -> Vec<u8> {
        let w = wallet();
        let (keys, inputs) = coins.iter().map(|c| w.input(c, leaf)).unzip();
        let total: Amount = coins.iter().map(|c| c.amount).sum();
        let outputs: Vec<Output> = vec![w.keys_for(&[to; 16]).pay(total - fee)];
        Transaction::sign(TxBody { inputs, outputs }, keys)
            .unwrap()
            .to_bytes()
    }

    #[test]
    fn valid_transactions_get_in_once() {
        let (chain, coins) = chain(3, COIN);
        let pool = Mempool::new();
        let tx = spend(&coins[..1], 100, FEE);
        let (id, new) = pool.add(&chain, &tx).unwrap();
        assert!(new && pool.knows(&id));
        assert_eq!(pool.add(&chain, &tx), Ok((id, false)));
        assert_eq!(pool.get(&id), Some(tx));
        assert!(pool.spends(&coins[0].record.commitment));
        assert_eq!(pool.size(&chain).0, 1);
        // Any fee will do, even none.
        let free = spend(&coins[1..2], 101, 0);
        assert!(pool.add(&chain, &free).unwrap().1);
    }

    #[test]
    fn bad_and_unfit_transactions_are_refused() {
        let (chain, coins) = chain(3, COIN);
        let pool = Mempool::new();
        // A broken signature.
        let mut bad = spend(&coins[1..2], 101, FEE);
        let n = bad.len();
        bad[n - 1] ^= 1;
        assert!(matches!(pool.add(&chain, &bad), Err(Refused::Invalid(_))));
        assert!(matches!(
            pool.add(&chain, &[1, 2, 3]),
            Err(Refused::Invalid(_))
        ));
        // A spend of a coin at the wrong slot.
        let mut moved = coins[2];
        moved.position = 99;
        let wrong = spend(&[moved], 102, FEE);
        assert!(matches!(pool.add(&chain, &wrong), Err(Refused::Unfit(_))));
        // A double spend of a pool transaction's input, not paying more.
        let (first, _) = pool.add(&chain, &spend(&coins[..1], 103, FEE)).unwrap();
        let again = spend_with(&coins[..1], 104, FEE, 1);
        assert!(matches!(pool.add(&chain, &again), Err(Refused::Unfit(_))));
        assert_eq!(pool.size(&chain).0, 1);
        assert!(pool.knows(&first));
    }

    #[test]
    fn a_better_paying_spend_of_the_same_inputs_replaces_the_old() {
        let (chain, coins) = chain(3, COIN);
        let pool = Mempool::new();
        let (a, _) = pool.add(&chain, &spend(&coins[..1], 100, FEE)).unwrap();
        let (b, _) = pool.add(&chain, &spend(&coins[1..2], 101, FEE)).unwrap();
        // Re-signed with another leaf, paying more: it replaces the first.
        let bumped = spend_with(&coins[..1], 102, 2 * FEE, 1);
        let (c, new) = pool.add(&chain, &bumped).unwrap();
        assert!(new && pool.get(&a).is_none() && pool.get(&c).is_some());
        let spender = pool.spender(&coins[0].record.commitment).unwrap();
        assert_eq!(spender.body.inputs[0].leaf, 1);
        // One spending both pool transactions' inputs must pay more than
        // both together, and more per byte than each (it's about twice
        // their size).
        let both = spend_with(&coins[..2], 103, 3 * FEE, 2);
        assert!(matches!(pool.add(&chain, &both), Err(Refused::Unfit(_))));
        let both = spend_with(&coins[..2], 103, 3 * FEE + 1, 2);
        assert!(matches!(pool.add(&chain, &both), Err(Refused::Unfit(_))));
        assert!(pool.get(&b).is_some() && pool.get(&c).is_some());
        let both = spend_with(&coins[..2], 104, 5 * FEE, 2);
        pool.add(&chain, &both).unwrap();
        assert!(pool.get(&b).is_none() && pool.get(&c).is_none());
        assert_eq!(pool.size(&chain).0, 1);
    }

    #[test]
    fn a_new_tip_drops_what_it_spent_and_a_reorg_brings_it_back() {
        let (mut chain, coins) = chain(2, COIN);
        let pool = Mempool::new();
        let (a, _) = pool.add(&chain, &spend(&coins[..1], 100, FEE)).unwrap();
        let (b, _) = pool.add(&chain, &spend(&coins[1..], 101, FEE)).unwrap();
        // A block mines the first: its input is spent, its output made.
        let tx = Transaction::from_bytes(&pool.get(&a).unwrap()).unwrap();
        let made = tx.body.outputs[0].commitment().unwrap();
        chain
            .unspent
            .borrow_mut()
            .remove(&coins[0].record.commitment);
        chain.unspent.borrow_mut().insert(made, 50);
        chain.tip = 1;
        assert_eq!(pool.select(&chain, usize::MAX).len(), 1);
        assert!(!pool.knows(&a) && pool.knows(&b));
        // A reorg undoes that block: the transaction is back.
        chain.unspent.borrow_mut().remove(&made);
        chain
            .unspent
            .borrow_mut()
            .insert(coins[0].record.commitment, coins[0].position);
        chain.tip = 2;
        assert_eq!(pool.select(&chain, usize::MAX).len(), 2);
        assert!(pool.knows(&a));
        // Spent by another block instead (a double spend that won): it
        // stays out while its input stays spent.
        chain
            .unspent
            .borrow_mut()
            .remove(&coins[0].record.commitment);
        chain.tip = 3;
        assert_eq!(pool.select(&chain, usize::MAX).len(), 1);
        assert!(!pool.knows(&a));
    }

    #[test]
    fn miners_take_the_best_paying_first_and_a_full_pool_pushes_out_the_worst() {
        let (chain, coins) = chain(4, COIN);
        let txs: Vec<Vec<u8>> = (0..4)
            .map(|i| spend(&coins[i..=i], 100 + i as u8, FEE * (1 + i as u64)))
            .collect();
        // Room for three.
        let pool = Mempool::with_capacity(3 * txs[0].len());
        for t in &txs[..3] {
            pool.add(&chain, t).unwrap();
        }
        let ids: Vec<Hash> = txs.iter().map(|t| Transaction::id_of(t)).collect();
        let fees: Vec<Amount> = pool
            .select(&chain, usize::MAX)
            .iter()
            .map(|(_, _, f)| *f)
            .collect();
        assert_eq!(fees, [3 * FEE, 2 * FEE, FEE]);
        // The best-paying one pushes out the cheapest.
        pool.add(&chain, &txs[3]).unwrap();
        assert_eq!(pool.get(&ids[0]), None);
        assert_eq!(pool.size(&chain).0, 3);
        // Room for one transaction's lists only.
        let one = block::INPUT_BYTES + block::OUTPUT_BYTES;
        let best = pool.select(&chain, one);
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].0, ids[3]);
        // One paying less than everything there doesn't get in.
        let cheap = spend(&coins[..1], 200, FEE);
        assert!(matches!(pool.add(&chain, &cheap), Err(Refused::Unfit(_))));
    }
}
