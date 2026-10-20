//! A development chain: proves, mines and checks blocks locally, logging
//! each step, to watch the whole pipeline run. The chain (state tree and
//! blocks) lives in LMDB, so each run extends the previous one; delete the
//! directory to start over.
//!
//! A wallet with a random seed pays itself: each block holds a reward and
//! (from block 2) one transaction spending every coin the wallet finds in
//! the state (the previous block's outputs) into two new outputs, paying a
//! fee. Every output goes to a fresh address, with its amount carried in
//! its nonce ([`crate::wallet`]). For each block the node:
//!
//! 1. proves the block's state transition ([`BlockProver`]);
//! 2. proves its chain proof ([`Chain`]), covering the parent header's PoW
//!    and every earlier block;
//! 3. builds the header and mines it at the ASERT target;
//! 4. serializes the block, then checks the decoded copy as a full node
//!    would and applies it.
//!
//! Each accepted block is stored ([`ChainStore`]) in the same LMDB
//! transaction as its state changes. Rewards follow the emission schedule
//! ([`Consensus::subsidy`]); at the final height the chain ends and the
//! program exits.
//!
//! At the end it checks the tip the way a fast-syncing node would, and
//! restores the wallet from its seed using only the state tree.

use std::error::Error;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::babybear::BabyBear;
use crate::block::{self, Block, BlockProver};
use crate::chain::{Chain, ChainProof};
use crate::db;
use crate::fri::FriParams;
use crate::hash::{self, Hash};
use crate::header::{self, Consensus, FRAC_BITS, Header};
use crate::output::{Amount, COIN};
use crate::store::ChainStore;
use crate::pow::{self, Dataset};
use crate::state::StateTree;
use crate::state_air::StateSegmentAir;
use crate::subblock_air::SubBlockAir;
use crate::rng::Rng;
use crate::transaction::{Transaction, TxBody};
use crate::wallet::Wallet;

/// The implicit fee each transaction leaves for the reward: 0.001 coin.
pub const FEE: Amount = COIN / 1000;

/// An amount in coins.
fn coins(units: u128) -> String {
    let (whole, frac) = (units / COIN as u128, units % COIN as u128);
    if frac == 0 {
        format!("{whole}")
    } else {
        format!("{whole}.{}", format!("{frac:08}").trim_end_matches('0'))
    }
}

/// Coins created by blocks `1 ..= height`.
fn supply(c: &Consensus, height: u64) -> u128 {
    let first = height.min(c.first_era) as u128;
    let second = height.min(c.final_height()).saturating_sub(c.first_era) as u128;
    first * c.first_subsidy as u128 + second * c.second_subsidy as u128
}

fn seed_bytes(seed: &Hash) -> Vec<u8> {
    seed.iter().flat_map(|x| x.as_u32().to_le_bytes()).collect()
}

fn seed_from(bytes: &[u8]) -> Option<Hash> {
    if bytes.len() != 32 {
        return None;
    }
    let mut seed = [BabyBear::ZERO; 8];
    for (x, chunk) in seed.iter_mut().zip(bytes.chunks_exact(4)) {
        *x = BabyBear::new(u32::from_le_bytes(chunk.try_into().ok()?));
    }
    Some(seed)
}

fn provers(consensus: Consensus, genesis: Header, data: &Dataset) -> Chain {
    let blocks = BlockProver::new(
        FriParams::default(),
        SubBlockAir {
            inputs: 16,
            outputs: 16,
        },
        StateSegmentAir {
            inserts: 16,
            spends: 16,
        },
        15,
    );
    Chain::new(consensus, genesis, blocks, data.tree())
}

struct Log(Instant);

impl Log {
    fn line(&self, msg: impl AsRef<str>) {
        println!("[{:>8.1}s] {}", self.0.elapsed().as_secs_f64(), msg.as_ref());
    }
}

fn short(h: &Hash) -> String {
    hash::to_bytes(h)[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn log2(x: u128) -> f64 {
    (x as f64).log2()
}

/// Expected hashes for `work` ([`header::work`] units are `2^-17` hashes).
fn hashes_log2(work: u128) -> f64 {
    log2(work) - 17.0
}

/// Extends the chain in `dir` by `blocks` blocks, creating it (with a new
/// genesis and wallet seed) if there's none yet.
pub fn run(dir: &Path, blocks: u64) -> Result<(), Box<dyn Error>> {
    let log = Log(Instant::now());
    log.line(format!(
        "dev chain in {}: {blocks} more blocks, {} threads",
        dir.display(),
        crate::par::threads()
    ));

    log.line("building the PoW dataset (64 MiB)");
    let data = Dataset::get();
    log.line("dataset ready");

    let env = db::open_env(dir)?;
    let tree = StateTree::open(&env)?;
    let store = ChainStore::open(&env)?;
    let existing = store.genesis(&env.read_txn()?)?;
    let (genesis, seed) = match existing {
        Some(genesis) => {
            let bytes = store.meta(&env.read_txn()?, "wallet_seed")?.ok_or("no wallet seed")?;
            (genesis, seed_from(&bytes).ok_or("bad wallet seed")?)
        }
        None => {
            let genesis = Header {
                height: 0,
                timestamp: now(),
                state: tree.state(&env.read_txn()?)?,
                ..Header::default()
            };
            let mut seed = [BabyBear::ZERO; 8];
            Rng::from_os()?.fill(&mut seed);
            let mut txn = env.write_txn()?;
            store.init(&mut txn, &genesis)?;
            store.set_meta(&mut txn, "wallet_seed", &seed_bytes(&seed))?;
            txn.commit()?;
            (genesis, seed)
        }
    };
    let consensus = Consensus {
        genesis_time: genesis.timestamp,
        ..Consensus::DEFAULT
    };

    log.line("setting up the provers (16-input sub-blocks, 16-op segments, nodes at 2^15)");
    let chain = provers(consensus, genesis, data);
    let ([first_rows, block_rows, chain_rows, wrap_rows], log_height) = chain.used_rows();
    log.line(format!(
        "provers ready; chain circuits use {first_rows} (first), {block_rows} (block), \
         {chain_rows} (chain) rows of 2^{log_height}, wrap {wrap_rows} of 2^{}",
        crate::chain::WRAP_LOG_HEIGHT
    ));

    // Where the stored chain left off: the tip, and its proof to extend.
    let (tip, mut prev_proof) = {
        let txn = env.read_txn()?;
        let tip = store.tip(&txn)?.ok_or("empty store")?;
        let proof = if tip.height == 0 {
            None
        } else {
            let block = store.block(&txn, tip.height)?.ok_or("missing tip block")?;
            let work_before = store.work(&txn, tip.height - 1)?.ok_or("missing work")?;
            Some(ChainProof {
                statement: chain.statement(&tip, work_before),
                proof: block.proof,
            })
        };
        (tip, proof)
    };
    if existing.is_some() {
        log.line(format!(
            "resuming at height {} (tip {}, genesis {})",
            tip.height,
            short(&tip.hash()),
            short(&genesis.hash()),
        ));
    } else {
        log.line(format!(
            "new chain: genesis {}, spacing {} s, half-life 2^{} s, first target 2^{:.2} \
             (~2^{:.2} hashes)",
            short(&genesis.hash()),
            consensus.spacing,
            consensus.log_half_life,
            consensus.genesis_log_target as f64 / (1 << FRAC_BITS) as f64,
            log2(pow::MAX_TARGET / header::target_of(consensus.log_target(&genesis))),
        ));
    }
    log.line(format!(
        "emission: {} coins a block to height {}, then {} to height {}; {} coins in all",
        coins(consensus.first_subsidy as u128),
        consensus.first_era,
        coins(consensus.second_subsidy as u128),
        consensus.final_height(),
        coins(consensus.total_supply()),
    ));

    let wallet = Wallet::from_seed(seed);
    let mut next_index = match store.meta(&env.read_txn()?, "wallet_next")? {
        Some(b) => u32::from_le_bytes(b.as_slice().try_into().map_err(|_| "bad wallet index")?),
        None => 0,
    };

    for _ in 0..blocks {
        let (parent, grandparent, work_before) = {
            let txn = env.read_txn()?;
            let parent = store.tip(&txn)?.ok_or("empty store")?;
            let grandparent = match parent.height {
                0 => None,
                h => store.header(&txn, h - 1)?,
            };
            (parent, grandparent, store.work(&txn, parent.height)?.ok_or("missing work")?)
        };
        let height = parent.height + 1;
        let Some(subsidy) = consensus.subsidy(height) else {
            log.line(format!(
                "the chain has ended at its final height {}: all {} coins are mined; \
                 shutting down",
                parent.height,
                coins(consensus.total_supply()),
            ));
            return Ok(());
        };
        log.line(format!("── block {height} ──"));

        // The wallet's coins, found by scanning the state's unspent
        // outputs.
        let outputs = tree.outputs(&env.read_txn()?)?;
        let held = wallet.scan(&outputs, 0..next_index);
        let mut fresh = || {
            next_index += 1;
            next_index - 1
        };

        // Transactions: spend every coin into two outputs, plus the reward.
        let mut txs = Vec::new();
        let mut fee = 0;
        if !held.is_empty() {
            let (keys, inputs) = held.iter().map(|c| wallet.input(c)).unzip();
            let total: Amount = held.iter().map(|c| c.amount).sum();
            let half = (total - FEE) / 2;
            let (a, b) = (half, total - FEE - half);
            let outputs = vec![wallet.address(fresh()).pay(a), wallet.address(fresh()).pay(b)];
            let tx = Transaction::sign(TxBody { inputs, outputs }, keys)
                .map_err(|e| format!("{e:?}"))?;
            let spent: Vec<String> = held
                .iter()
                .map(|c| format!("{} @ slot {}", coins(c.amount as u128), c.position))
                .collect();
            log.line(format!(
                "tx: spends {} = {} -> {} + {} (fee {})",
                spent.join(" + "),
                coins(total as u128),
                coins(a as u128),
                coins(b as u128),
                coins(FEE as u128),
            ));
            txs.push(tx);
            fee = FEE;
        }
        let reward = wallet.address(fresh()).pay(subsidy + fee);
        log.line(format!(
            "reward: {} (subsidy {} + implicit fees {})",
            coins(reward.amount as u128),
            coins(subsidy as u128),
            coins(fee as u128),
        ));
        txs.push(
            Transaction::sign(
                TxBody {
                    inputs: vec![],
                    outputs: vec![reward],
                },
                vec![],
            )
            .map_err(|e| format!("{e:?}"))?,
        );

        // 1. Prove the transition.
        let t = Instant::now();
        let mut txn = env.write_txn()?;
        let agg = chain.block_prover().prove(&tree, &mut txn, &txs, subsidy)?;
        let new_state = tree.state(&txn)?;
        txn.abort();
        log.line(format!(
            "transition proved in {:.1} s: {} inputs, {} outputs",
            t.elapsed().as_secs_f64(),
            agg.inputs.len(),
            agg.outputs.len(),
        ));

        // 2. The chain proof: block 1 from genesis, later blocks extend the
        // previous proof and prove the parent header's PoW.
        let t = Instant::now();
        let proof = match (&prev_proof, &grandparent) {
            (Some(prev), Some(grandparent)) => chain.prove_next(prev, grandparent, &parent, &agg)?,
            _ => chain.prove_first(&agg)?,
        };
        let secs = t.elapsed().as_secs_f64();
        let size = proof.proof.size() / 1024;
        if height == 1 {
            log.line(format!(
                "chain proof in {secs:.1} s (first block from genesis, wrapped): {size} KB"
            ));
        } else {
            log.line(format!(
                "chain proof in {secs:.1} s (block + chain nodes, wrapped): {size} KB, covers \
                 headers 1..={} with ~2^{:.2} hashes of work",
                height - 1,
                hashes_log2(work_before),
            ));
        }
        let block = Block {
            inputs: agg.inputs,
            outputs: agg.outputs,
            proof: proof.proof.clone(),
        };

        // 3. Mine the header, which commits to the block and its proof.
        let mut header = Header {
            height,
            prev: parent.hash(),
            timestamp: now().max(parent.timestamp + 1),
            state: new_state,
            body: block.body_hash().ok_or("bad body")?,
            nonce: 0,
        };
        let log_target = consensus.log_target(&parent);
        let target = header::target_of(log_target);
        log.line(format!(
            "mining: log-target {:.4}, ~2^{:.2} expected hashes",
            log_target as f64 / (1 << FRAC_BITS) as f64,
            log2(pow::MAX_TARGET / target),
        ));
        let t = Instant::now();
        if !header.mine(data, target) {
            return Err("nonces exhausted".into());
        }
        let secs = t.elapsed().as_secs_f64();
        log.line(format!(
            "mined in {secs:.1} s: nonce {} (~{:.0} kH/s), hash {}",
            header.nonce,
            (header.nonce + 1) as f64 / secs / 1e3,
            short(&header.hash()),
        ));

        // 4. Serialize, then check, apply and store the decoded block as a
        // full node receiving it would.
        let bytes = block::encode(&header, &block);
        let (header, block) = block::decode(&bytes)?;
        log.line(format!(
            "serialized: {} bytes ({:.1}% of the {} limit)",
            bytes.len(),
            100.0 * bytes.len() as f64 / block::MAX_BLOCK_SIZE as f64,
            block::MAX_BLOCK_SIZE,
        ));
        let t = Instant::now();
        header::check(&consensus, data, &parent, &header)?;
        let header_time = t.elapsed();
        let t = Instant::now();
        chain.verify_block(&header, &block, work_before)?;
        let block_time = t.elapsed();
        let t = Instant::now();
        let work = work_before + header::work(log_target);
        let mut txn = env.write_txn()?;
        block::apply(&tree, &mut txn, &parent, &header, &block)?;
        store.push(&mut txn, &header, &block, work)?;
        store.set_meta(&mut txn, "wallet_next", &next_index.to_le_bytes())?;
        txn.commit()?;
        let apply_time = t.elapsed();
        let unspent = tree.outputs(&env.read_txn()?)?.len();
        log.line(format!(
            "verified: header (links, time, PoW) {:.2} ms, block (size, body hash, chain \
             proof) {:.2} ms; applied, matched the header's state and stored in {:.2} ms; \
             {unspent} unspent outputs, {} coins issued",
            header_time.as_secs_f64() * 1e3,
            block_time.as_secs_f64() * 1e3,
            apply_time.as_secs_f64() * 1e3,
            coins(supply(&consensus, height)),
        ));

        prev_proof = Some(proof);
    }

    // Restore the wallet from its seed alone: scan the unspent outputs (what
    // a fast-syncing node downloads) for its tags.
    let txn = env.read_txn()?;
    let t = Instant::now();
    let outputs = tree.outputs(&txn)?;
    let restored = Wallet::from_seed(seed).scan(&outputs, 0..next_index + 100);
    let balance: u128 = restored.iter().map(|c| c.amount as u128).sum();
    log.line(format!(
        "wallet restored from its seed in {:.1} ms: {} of {} unspent outputs are ours, \
         balance {} coins",
        t.elapsed().as_secs_f64() * 1e3,
        restored.len(),
        outputs.len(),
        coins(balance),
    ));

    // A node fast-syncing from the tip needs only the last two headers and
    // the tip's proof (plus the state tree, checked against the tip's root).
    let tip = store.tip(&txn)?.ok_or("empty store")?;
    if tip.height >= 1 {
        let tip_parent = store.header(&txn, tip.height - 1)?.ok_or("missing header")?;
        let claimed = store.work(&txn, tip.height - 1)?.ok_or("missing work")?;
        let proof = store.block(&txn, tip.height)?.ok_or("missing tip block")?.proof;
        let t = Instant::now();
        if tip.height >= 2 {
            header::check(&consensus, data, &tip_parent, &tip)?;
        }
        chain
            .verify_proof(&tip, claimed, &proof)
            .map_err(|e| format!("tip proof: {e}"))?;
        log.line(format!(
            "fast sync: tip {} at height {} checked from headers {} and {} and one proof in \
             {:.1} ms; ~2^{:.2} hashes of total work",
            short(&tip.hash()),
            tip.height,
            tip.height - 1,
            tip.height,
            t.elapsed().as_secs_f64() * 1e3,
            hashes_log2(store.work(&txn, tip.height)?.ok_or("missing work")?),
        ));
    }
    Ok(())
}

/// Prints the stored chain in `dir`.
pub fn status(dir: &Path) -> Result<(), Box<dyn Error>> {
    if !dir.exists() {
        println!("no chain in {}; create one with --dev-chain <BLOCKS>", dir.display());
        return Ok(());
    }
    let env = db::open_env(dir)?;
    let tree = StateTree::open(&env)?;
    let store = ChainStore::open(&env)?;
    let txn = env.read_txn()?;
    let Some(tip) = store.tip(&txn)? else {
        println!("no chain in {}; create one with --dev-chain <BLOCKS>", dir.display());
        return Ok(());
    };
    let genesis = store.genesis(&txn)?.ok_or("missing genesis")?;
    let consensus = Consensus {
        genesis_time: genesis.timestamp,
        ..Consensus::DEFAULT
    };
    let state = tree.state(&txn)?;
    let work = store.work(&txn, tip.height)?.ok_or("missing work")?;
    println!("chain in {}", dir.display());
    println!("  genesis    {} at {}", short(&genesis.hash()), genesis.timestamp);
    println!(
        "  tip        {} at height {}, time {}",
        short(&tip.hash()),
        tip.height,
        tip.timestamp
    );
    if tip.height > 0 {
        println!("  work       ~2^{:.2} hashes", hashes_log2(work));
    }
    println!(
        "  state      {} unspent outputs in {} slots; {}",
        tree.outputs(&txn)?.len(),
        state.size - 1,
        if state == tip.state {
            "matches the tip header"
        } else {
            "DOES NOT match the tip header"
        }
    );
    println!(
        "  supply     {} of {} coins issued",
        coins(supply(&consensus, tip.height)),
        coins(consensus.total_supply())
    );
    match consensus.subsidy(tip.height + 1) {
        Some(s) => println!(
            "  next       block {} creates {} coins; the chain ends at height {}",
            tip.height + 1,
            coins(s as u128),
            consensus.final_height()
        ),
        None => println!("  the chain has ended"),
    }
    Ok(())
}
