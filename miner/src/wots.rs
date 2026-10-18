//! Winternitz one-time signatures (WOTS+ style) over Poseidon2/BabyBear.
//!
//! Every hash is a single Poseidon2 permutation (or a sponge of them) with a
//! public parameter and a domain/tweak in the input, so each chain step,
//! PRF call and digest is separated from every other one.
//!
//! A secret key must sign **at most one** message: two signatures under the
//! same key let anyone forge signatures for other messages. [`SecretKey::sign`]
//! consumes the key to make reuse harder, but regenerating the key from the
//! same seed bypasses that.

use crate::babybear::BabyBear;
use crate::hash::{DOMAIN_SLOT, MAX_PARAM_LEN, compress, domain, sponge};
pub use crate::hash::{HASH_LEN, Hash};
use crate::poseidon2::WIDTH;

/// Winternitz parameter: each chain has `W` positions, `0..W`.
pub const W: usize = 8;
const LOG_W: usize = 3;
/// Total chains (hashes) in a signature: message chains plus checksum chains.
pub const NUM_CHAINS: usize = 64;
pub const CHECKSUM_CHAINS: usize = 3;
pub const MSG_CHAINS: usize = NUM_CHAINS - CHECKSUM_CHAINS;
/// Field elements in the public parameter.
pub const PARAM_LEN: usize = 5;
/// Field elements in the per-signature message randomizer.
pub const RAND_LEN: usize = 7;
/// Field elements in the secret seed.
pub const SEED_LEN: usize = 8;

/// Low bits of each digest element used for message digits. `p = 120 * 2^24 + 1`,
/// so these bits are uniform up to a ~2^-7 relative bias on a single value.
const BITS_PER_ELEM: usize = 24;
const DIGITS_PER_ELEM: usize = BITS_PER_ELEM / LOG_W;

const _: () = {
    assert!(W == 1 << LOG_W);
    assert!(BITS_PER_ELEM.is_multiple_of(LOG_W));
    // Chain-step input layout: value | parameter | domain, chain, position.
    assert!(HASH_LEN + PARAM_LEN == DOMAIN_SLOT);
    assert!(SEED_LEN <= DOMAIN_SLOT);
    assert!(PARAM_LEN <= MAX_PARAM_LEN);
    assert!(RAND_LEN <= HASH_LEN && PARAM_LEN <= HASH_LEN);
    assert!(MSG_CHAINS <= HASH_LEN * DIGITS_PER_ELEM);
    // The largest checksum must fit in CHECKSUM_CHAINS base-W digits.
    assert!(MSG_CHAINS * (W - 1) < W.pow(CHECKSUM_CHAINS as u32));
};

pub type Param = [BabyBear; PARAM_LEN];
pub type Seed = [BabyBear; SEED_LEN];

/// Secret key. Deliberately not `Clone`: [`SecretKey::sign`] consumes it.
pub struct SecretKey {
    seed: Seed,
    param: Param,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicKey {
    pub param: Param,
    /// Hash of all chain ends.
    pub root: Hash,
}

impl PublicKey {
    /// `H(pubkey)`: one compression of `root | param`.
    pub fn hash(&self) -> Hash {
        let mut input = [BabyBear::ZERO; WIDTH];
        input[..HASH_LEN].copy_from_slice(&self.root);
        input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(&self.param);
        input[DOMAIN_SLOT] = BabyBear::new(domain::PUBLIC_KEY_HASH);
        compress(input)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub rand: [BabyBear; RAND_LEN],
    pub chains: [Hash; NUM_CHAINS],
}

/// Derives a key pair from a secret seed. The seed must be uniformly random
/// and kept secret.
pub fn keygen(seed: Seed) -> (SecretKey, PublicKey) {
    let digest = sponge(domain::WOTS_PARAMETER, &[], &[&seed]);
    let param: Param = core::array::from_fn(|i| digest[i]);
    let ends: [Hash; NUM_CHAINS] =
        core::array::from_fn(|i| chain(&param, i, 0, W - 1, prf(&seed, i)));
    let root = sponge(domain::WOTS_PUBLIC_KEY, &param, &[ends.as_flattened()]);
    (SecretKey { seed, param }, PublicKey { param, root })
}

impl SecretKey {
    /// Signs `msg`, consuming the key.
    pub fn sign(self, msg: &[BabyBear]) -> Signature {
        // Deterministic randomizer, so signing needs no RNG.
        let digest = sponge(domain::WOTS_RANDOMIZER, &self.param, &[&self.seed, msg]);
        let rand = core::array::from_fn(|i| digest[i]);
        let digits = encode(&self.param, &rand, msg);
        let chains = core::array::from_fn(|i| {
            chain(&self.param, i, 0, digits[i] as usize, prf(&self.seed, i))
        });
        Signature { rand, chains }
    }
}

/// Returns `true` if `sig` is a valid signature on `msg` under `pk`.
pub fn verify(pk: &PublicKey, msg: &[BabyBear], sig: &Signature) -> bool {
    let digits = encode(&pk.param, &sig.rand, msg);
    let ends: [Hash; NUM_CHAINS] = core::array::from_fn(|i| {
        let d = digits[i] as usize;
        chain(&pk.param, i, d, W - 1 - d, sig.chains[i])
    });
    sponge(domain::WOTS_PUBLIC_KEY, &pk.param, &[ends.as_flattened()]) == pk.root
}

/// Maps a message to `NUM_CHAINS` base-`W` digits: `MSG_CHAINS` digits of the
/// randomized message digest, then the checksum `sum(W - 1 - d)` big-endian.
fn encode(param: &Param, rand: &[BabyBear; RAND_LEN], msg: &[BabyBear]) -> [u8; NUM_CHAINS] {
    let digest = sponge(domain::WOTS_MESSAGE, param, &[rand, msg]);
    let mut digits = [0u8; NUM_CHAINS];
    for (i, d) in digits[..MSG_CHAINS].iter_mut().enumerate() {
        let elem = digest[i / DIGITS_PER_ELEM].as_u32();
        *d = ((elem >> (LOG_W * (i % DIGITS_PER_ELEM))) as usize % W) as u8;
    }
    let mut checksum: usize = digits[..MSG_CHAINS]
        .iter()
        .map(|&d| W - 1 - d as usize)
        .sum();
    for d in digits[MSG_CHAINS..].iter_mut().rev() {
        *d = (checksum % W) as u8;
        checksum /= W;
    }
    digits
}

/// Walks chain `chain_idx` `steps` times, starting at position `start`.
fn chain(param: &Param, chain_idx: usize, start: usize, steps: usize, mut value: Hash) -> Hash {
    debug_assert!(start + steps < W);
    for pos in start..start + steps {
        let mut input = [BabyBear::ZERO; WIDTH];
        input[..HASH_LEN].copy_from_slice(&value);
        input[HASH_LEN..HASH_LEN + PARAM_LEN].copy_from_slice(param);
        input[DOMAIN_SLOT] = BabyBear::new(domain::WOTS_CHAIN);
        input[DOMAIN_SLOT + 1] = BabyBear::new(chain_idx as u32);
        input[DOMAIN_SLOT + 2] = BabyBear::new(pos as u32);
        value = compress(input);
    }
    value
}

/// Secret start of chain `chain_idx`.
fn prf(seed: &Seed, chain_idx: usize) -> Hash {
    let mut input = [BabyBear::ZERO; WIDTH];
    input[..SEED_LEN].copy_from_slice(seed);
    input[DOMAIN_SLOT] = BabyBear::new(domain::WOTS_PRF);
    input[DOMAIN_SLOT + 1] = BabyBear::new(chain_idx as u32);
    compress(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(n: u32) -> Seed {
        BabyBear::new_array(core::array::from_fn(|i| n * 1000 + i as u32))
    }

    fn msg(n: u32) -> Vec<BabyBear> {
        (0..10).map(|i| BabyBear::new(n + i)).collect()
    }

    #[test]
    fn sign_verify_roundtrip() {
        for n in 0..4 {
            let (sk, pk) = keygen(seed(n));
            let m = msg(n);
            let sig = sk.sign(&m);
            assert!(verify(&pk, &m, &sig));
        }
    }

    #[test]
    fn empty_message() {
        let (sk, pk) = keygen(seed(1));
        let sig = sk.sign(&[]);
        assert!(verify(&pk, &[], &sig));
        assert!(!verify(&pk, &[BabyBear::ZERO], &sig));
    }

    #[test]
    fn rejects_wrong_message_key_or_tampering() {
        let (sk, pk) = keygen(seed(1));
        let m = msg(1);
        let sig = sk.sign(&m);

        assert!(!verify(&pk, &msg(2), &sig));
        // Zero padding must not make a trailing zero invisible.
        let mut padded = m.clone();
        padded.push(BabyBear::ZERO);
        assert!(!verify(&pk, &padded, &sig));

        let (_, other_pk) = keygen(seed(2));
        assert!(!verify(&other_pk, &m, &sig));

        let mut bad = sig.clone();
        bad.rand[0] += BabyBear::ONE;
        assert!(!verify(&pk, &m, &bad));

        for i in [0, MSG_CHAINS - 1, MSG_CHAINS, NUM_CHAINS - 1] {
            let mut bad = sig.clone();
            bad.chains[i][3] += BabyBear::ONE;
            assert!(!verify(&pk, &m, &bad));
        }
    }

    #[test]
    fn keygen_is_deterministic() {
        assert_eq!(keygen(seed(5)).1, keygen(seed(5)).1);
        assert_ne!(keygen(seed(5)).1, keygen(seed(6)).1);
    }

    #[test]
    fn checksum_balances_message_digits() {
        let param = keygen(seed(3)).1.param;
        for n in 0..20 {
            let digits = encode(&param, &[BabyBear::new(n); RAND_LEN], &msg(n));
            assert!(digits.iter().all(|&d| (d as usize) < W));
            let msg_sum: usize = digits[..MSG_CHAINS].iter().map(|&d| d as usize).sum();
            let checksum = digits[MSG_CHAINS..]
                .iter()
                .fold(0, |acc, &d| acc * W + d as usize);
            assert_eq!(msg_sum + checksum, MSG_CHAINS * (W - 1));
        }
    }

    #[test]
    fn chain_walks_compose() {
        let param = keygen(seed(4)).1.param;
        let start = prf(&seed(4), 7);
        for split in 0..W {
            let mid = chain(&param, 7, 0, split, start);
            assert_eq!(
                chain(&param, 7, split, W - 1 - split, mid),
                chain(&param, 7, 0, W - 1, start)
            );
        }
        // Different chains and positions must hash differently.
        assert_ne!(chain(&param, 7, 0, 1, start), chain(&param, 8, 0, 1, start));
        assert_ne!(chain(&param, 7, 0, 1, start), chain(&param, 7, 1, 1, start));
    }
}
