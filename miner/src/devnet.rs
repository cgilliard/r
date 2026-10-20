//! A development chain: proves, mines and checks blocks locally, logging
//! each step, to watch the whole pipeline run.
//!
//! Each block holds a reward and (from block 2) one transaction spending
//! every output of the previous block (its transaction's outputs and its
//! reward) into two new outputs, paying a fee. Keys
//! are derived from a counter, so they're for development only. For each
//! block the node:
//!
//! 1. proves the block's state transition ([`BlockProver`]);
//! 2. proves its chain proof ([`Chain`]), covering the parent header's PoW
//!    and every earlier block;
//! 3. builds the header and mines it at the ASERT target;
//! 4. serializes the block, then checks the decoded copy as a full node
//!    would and applies it.
//!
//! At the end it checks the tip the way a fast-syncing node would.

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
use crate::output::{self, Amount, NONCE_LEN, Nonce, Output};
use crate::pow::{self, Dataset};
use crate::state::StateTree;
use crate::state_air::StateSegmentAir;
use crate::subblock_air::SubBlockAir;
use crate::transaction::{Input, Transaction, TxBody, pubkey_hash};
use crate::wots::{self, SecretKey};

/// The implicit fee each transaction leaves for the reward.
pub const FEE: Amount = 1;

struct Log(Instant);

impl Log {
    fn line(&self, msg: impl AsRef<str>) {
        println!("[{:>8.1}s] {}", self.0.elapsed().as_secs_f64(), msg.as_ref());
    }
}

/// A coin the wallet can spend.
struct Coin {
    key: u32,
    amount: Amount,
    nonce: Nonce,
}

fn key(n: u32) -> (SecretKey, wots::PublicKey) {
    wots::keygen(core::array::from_fn(|i| BabyBear::new(n * 31 + i as u32 + 7)))
}

fn nonce(n: u32) -> Nonce {
    let mut nonce = [0u8; NONCE_LEN];
    nonce[..4].copy_from_slice(&n.to_le_bytes());
    nonce[4] = 0xde;
    nonce
}

fn pay(n: u32, amount: Amount) -> (Output, Coin) {
    let output = Output {
        hash: pubkey_hash(&key(n).1),
        amount,
        nonce: nonce(n),
    };
    let coin = Coin {
        key: n,
        amount,
        nonce: nonce(n),
    };
    (output, coin)
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

/// Builds a fresh chain of `blocks` blocks in `dir`, replacing anything
/// there.
pub fn run(dir: &Path, blocks: u64) -> Result<(), Box<dyn Error>> {
    let log = Log(Instant::now());
    log.line(format!(
        "dev chain: {blocks} blocks in {}, {} threads",
        dir.display(),
        crate::par::threads()
    ));

    log.line("building the PoW dataset (64 MiB)");
    let data = Dataset::get();
    log.line("dataset ready");

    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    let env = db::open_env(dir)?;
    let tree = StateTree::open(&env)?;
    let genesis_time = now();
    let consensus = Consensus {
        genesis_time,
        ..Consensus::DEFAULT
    };
    let genesis = Header {
        height: 0,
        timestamp: genesis_time,
        state: tree.state(&env.read_txn()?)?,
        ..Header::default()
    };

    log.line("setting up the provers (16-input sub-blocks, 16-op segments, nodes at 2^15)");
    let prover = BlockProver::new(
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
    let chain = Chain::new(consensus, genesis, prover, data.tree());
    let ([first_rows, block_rows, chain_rows, wrap_rows], log_height) = chain.used_rows();
    log.line(format!(
        "provers ready; chain circuits use {first_rows} (first), {block_rows} (block), \
         {chain_rows} (chain) rows of 2^{log_height}, wrap {wrap_rows} of 2^{}",
        crate::chain::WRAP_LOG_HEIGHT
    ));
    log.line(format!(
        "genesis {}: spacing {} s, half-life 2^{} s, first target 2^{:.2} (~2^{:.2} hashes)",
        short(&genesis.hash()),
        consensus.spacing,
        consensus.log_half_life,
        consensus.genesis_log_target as f64 / (1 << FRAC_BITS) as f64,
        log2(pow::MAX_TARGET / header::target_of(consensus.log_target(&genesis))),
    ));

    let mut headers = vec![genesis];
    // Work of headers 1 ..= i, per i.
    let mut work: Vec<u128> = vec![0];
    let mut proofs: Vec<ChainProof> = Vec::new();
    let mut wallet: Vec<Coin> = Vec::new();
    let mut next_key = 1u32;
    let mut fresh = || {
        next_key += 1;
        next_key
    };

    for height in 1..=blocks {
        let parent = *headers.last().expect("genesis");
        log.line(format!("── block {height} ──"));

        // Transactions: spend all of the previous block's outputs into two,
        // plus the reward.
        let mut txs = Vec::new();
        let mut new_coins = Vec::new();
        let mut fee = 0;
        if !wallet.is_empty() {
            let coins = std::mem::take(&mut wallet);
            let mut inputs = Vec::new();
            let mut keys = Vec::new();
            for coin in &coins {
                let (sk, pubkey) = key(coin.key);
                let commitment = output::commit(&pubkey_hash(&pubkey), coin.amount, &coin.nonce)
                    .ok_or("bad commitment")?;
                inputs.push(Input {
                    commitment,
                    pubkey,
                    amount: coin.amount,
                    nonce: coin.nonce,
                });
                keys.push(sk);
            }
            let total: Amount = coins.iter().map(|c| c.amount).sum();
            let half = (total - FEE) / 2;
            let (o1, c1) = pay(fresh(), half);
            let (o2, c2) = pay(fresh(), total - FEE - half);
            let tx = Transaction::sign(
                TxBody {
                    inputs,
                    outputs: vec![o1, o2],
                },
                keys,
            )
            .map_err(|e| format!("{e:?}"))?;
            let amounts: Vec<String> = coins.iter().map(|c| c.amount.to_string()).collect();
            log.line(format!(
                "tx: spends {} = {total} -> {} + {} (fee {FEE})",
                amounts.join(" + "),
                c1.amount,
                c2.amount
            ));
            txs.push(tx);
            new_coins.extend([c1, c2]);
            fee = FEE;
        }
        let subsidy = consensus.subsidy;
        let (o, c) = pay(fresh(), subsidy + fee);
        log.line(format!(
            "reward: {} (subsidy {subsidy} + implicit fees {fee})",
            c.amount
        ));
        txs.push(
            Transaction::sign(
                TxBody {
                    inputs: vec![],
                    outputs: vec![o],
                },
                vec![],
            )
            .map_err(|e| format!("{e:?}"))?,
        );
        new_coins.push(c);

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
        let proof = match proofs.last() {
            None => chain.prove_first(&agg)?,
            Some(prev) => {
                let grandparent = headers[headers.len() - 2];
                chain.prove_next(prev, &grandparent, &parent, &agg)?
            }
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
                hashes_log2(work[height as usize - 1]),
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

        // 4. Serialize, then check and apply the decoded block as a full
        // node receiving it would.
        let bytes = block::encode(&header, &block);
        let (header, block) = block::decode(&bytes)?;
        log.line(format!(
            "serialized: {} bytes ({:.1}% of the {} limit)",
            bytes.len(),
            100.0 * bytes.len() as f64 / block::MAX_BLOCK_SIZE as f64,
            block::MAX_BLOCK_SIZE,
        ));
        let work_before = work[height as usize - 1];
        let t = Instant::now();
        header::check(&consensus, data, &parent, &header)?;
        let header_time = t.elapsed();
        let t = Instant::now();
        chain.verify_block(&header, &block, work_before)?;
        let block_time = t.elapsed();
        let t = Instant::now();
        let mut txn = env.write_txn()?;
        block::apply(&tree, &mut txn, &parent, &header, &block)?;
        txn.commit()?;
        let apply_time = t.elapsed();
        log.line(format!(
            "verified: header (links, time, PoW) {:.2} ms, block (size, body hash, chain \
             proof) {:.2} ms, applied and matched the header's state {:.2} ms; state holds \
             {} outputs",
            header_time.as_secs_f64() * 1e3,
            block_time.as_secs_f64() * 1e3,
            apply_time.as_secs_f64() * 1e3,
            header.state.size - 1,
        ));

        work.push(work_before + header::work(log_target));
        wallet.extend(new_coins);
        proofs.push(proof);
        headers.push(header);
    }

    // A node fast-syncing from the tip needs only the last two headers and
    // the tip's proof (plus the state tree, checked against the tip's root).
    let tip = *headers.last().expect("tip");
    let tip_parent = headers[headers.len() - 2];
    let t = Instant::now();
    let claimed = work[tip.height as usize - 1];
    header::check(&consensus, data, &tip_parent, &tip)?;
    chain
        .verify_proof(&tip, claimed, &proofs.last().expect("tip proof").proof)
        .map_err(|e| format!("tip proof: {e}"))?;
    log.line(format!(
        "fast sync: tip {} at height {} checked from headers {} and {} and one proof in \
         {:.1} ms; ~2^{:.2} hashes of total work, {} outputs in the state",
        short(&tip.hash()),
        tip.height,
        tip.height - 1,
        tip.height,
        t.elapsed().as_secs_f64() * 1e3,
        hashes_log2(claimed + header::work(consensus.log_target(&tip_parent))),
        tip.state.size - 1,
    ));
    Ok(())
}
