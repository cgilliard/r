//! Transactions. Only pay-to-pubkey-hash inputs are supported for now.
//!
//! An [`Input`] names the commitment it spends and reveals the WOTS public key,
//! amount and nonce behind it. Each input signs the digest of the whole
//! [`TxBody`] (every input and output, nonces included) with the key it
//! reveals. The fee is implicit: `sum(inputs) - sum(outputs)`.
//!
//! [`Transaction::validate`] performs every check that needs no chain state
//! and returns the [`Totals`]. It does not require inputs to cover outputs, so
//! a block reward (outputs, no inputs) validates too. The caller must check
//! that each input is unspent, and the block must bound how much its
//! transactions mint.

use std::fmt;

use crate::babybear::BabyBear;
use crate::hash::{self, DOMAIN_SLOT, HASH_LEN, Hash, domain};
use crate::poseidon2::{self, WIDTH};
use crate::output::{self, Amount, Commitment, Nonce, Output, OutputHash};
use crate::wots::{self, PublicKey, SecretKey, Signature};

/// The pubkey hash an output pays to: `H(pubkey)` as bytes.
pub fn pubkey_hash(pubkey: &PublicKey) -> OutputHash {
    hash::to_bytes(&pubkey.hash())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// Commitment of the output being spent.
    pub commitment: Commitment,
    /// Its slot in the state tree: what makes the spent output unique, so
    /// a signature can't be replayed against an identical output created
    /// later (which lands in a new slot).
    pub position: u64,
    pub pubkey: PublicKey,
    pub amount: Amount,
    pub nonce: Nonce,
}

impl Input {
    /// Returns `true` if the revealed pubkey, amount and nonce open `commitment`.
    pub fn opens_commitment(&self) -> bool {
        output::commit_field(&self.pubkey.hash(), self.amount, &self.nonce) == self.commitment
    }
}

/// Everything the signatures commit to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxBody {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

impl TxBody {
    /// The message every input signs: a sponge (domain `TX_BODY`, with
    /// `n_in, n_out` and the absorbed length in the capacity) over each
    /// input's commitment, then its position (two limbs, zero-padded to a
    /// block), then each output's commitment. Each commitment binds its
    /// amount, nonce and hash (for an input, `H(pubkey)`), the position
    /// binds which output instance is spent, and the signature checks the
    /// pubkey itself, so this covers every field. `None` if a commitment or
    /// output hash is not a canonical encoding.
    pub fn digest(&self) -> Option<Hash> {
        let inputs = self
            .inputs
            .iter()
            .map(|i| Some((i.commitment.to_field()?, i.position)))
            .collect::<Option<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(|o| o.commitment()?.to_field())
            .collect::<Option<Vec<_>>>()?;
        Some(digest_of(&inputs, &outputs))
    }
}

/// [`TxBody::digest`] from the inputs' commitments and positions and the
/// outputs' commitments, as field elements.
pub(crate) fn digest_of(inputs: &[(Hash, u64)], outputs: &[Hash]) -> Hash {
    {
        let (nin, nout) = (count(inputs.len()), count(outputs.len()));
        let mut state = [BabyBear::ZERO; WIDTH];
        state[HASH_LEN] = BabyBear::new(nin);
        state[HASH_LEN + 1] = BabyBear::new(nout);
        state[DOMAIN_SLOT] = BabyBear::new(domain::TX_BODY);
        state[DOMAIN_SLOT + 1] = BabyBear::new((HASH_LEN as u32 + 2) * nin + HASH_LEN as u32 * nout);
        let mut absorb = |block: &[BabyBear]| {
            for (x, &v) in state.iter_mut().zip(block) {
                *x += v;
            }
            poseidon2::permute(&mut state);
        };
        for (c, position) in inputs {
            absorb(c);
            let (lo, hi) = crate::state::split(*position);
            absorb(&[lo, hi]);
        }
        for c in outputs {
            absorb(c);
        }
        if nin + nout == 0 {
            poseidon2::permute(&mut state);
        }
        core::array::from_fn(|j| state[j])
    }
}

/// Counts must stay below `p` to be one field element each.
fn count(n: usize) -> u32 {
    assert!(
        n < crate::babybear::P as usize,
        "too many inputs or outputs"
    );
    n as u32
}

fn total(mut amounts: impl Iterator<Item = Amount>) -> Result<Amount, TxError> {
    amounts
        .try_fold(0, Amount::checked_add)
        .ok_or(TxError::AmountOverflow)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transaction {
    pub body: TxBody,
    /// One signature per input, in input order.
    pub signatures: Vec<Signature>,
}

/// Sums of a transaction's input and output amounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Totals {
    pub inputs: Amount,
    pub outputs: Amount,
}

impl Totals {
    /// `inputs - outputs`. Negative when the transaction mints, as a block
    /// reward does.
    pub fn fee(&self) -> i128 {
        self.inputs as i128 - self.outputs as i128
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TxError {
    SignatureCount {
        inputs: usize,
        signatures: usize,
    },
    /// Input at this index spends the same commitment as an earlier one.
    DuplicateInput(usize),
    /// Input's pubkey, amount and nonce do not open its commitment.
    CommitmentMismatch(usize),
    BadSignature(usize),
    AmountOverflow,
    /// A commitment or output hash is not a canonical field encoding.
    NonCanonical,
    /// No inputs and no outputs.
    Empty,
}

impl fmt::Display for TxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureCount { inputs, signatures } => {
                write!(f, "{inputs} inputs but {signatures} signatures")
            }
            Self::DuplicateInput(i) => write!(f, "input {i} spends a commitment twice"),
            Self::CommitmentMismatch(i) => {
                write!(f, "input {i}: revealed fields do not open its commitment")
            }
            Self::BadSignature(i) => write!(f, "input {i}: invalid signature"),
            Self::AmountOverflow => write!(f, "amounts overflow"),
            Self::NonCanonical => write!(f, "non-canonical commitment or output hash"),
            Self::Empty => write!(f, "transaction has no inputs or outputs"),
        }
    }
}

impl std::error::Error for TxError {}

impl Transaction {
    /// Signs `body` with one key per input, in input order.
    ///
    /// # Panics
    /// If the number of keys differs from the number of inputs.
    pub fn sign(body: TxBody, keys: Vec<SecretKey>) -> Result<Self, TxError> {
        assert_eq!(keys.len(), body.inputs.len(), "one key per input");
        let digest = body.digest().ok_or(TxError::NonCanonical)?;
        let signatures = keys.into_iter().map(|k| k.sign(&digest)).collect();
        Ok(Self { body, signatures })
    }

    /// Checks everything that needs no chain state and returns the totals.
    pub fn validate(&self) -> Result<Totals, TxError> {
        let TxBody { inputs, outputs } = &self.body;
        if inputs.is_empty() && outputs.is_empty() {
            return Err(TxError::Empty);
        }
        if self.signatures.len() != inputs.len() {
            return Err(TxError::SignatureCount {
                inputs: inputs.len(),
                signatures: self.signatures.len(),
            });
        }
        for (i, input) in inputs.iter().enumerate() {
            if inputs[..i].iter().any(|p| p.commitment == input.commitment) {
                return Err(TxError::DuplicateInput(i));
            }
            if !input.opens_commitment() {
                return Err(TxError::CommitmentMismatch(i));
            }
        }
        let totals = Totals {
            inputs: total(inputs.iter().map(|i| i.amount))?,
            outputs: total(outputs.iter().map(|o| o.amount))?,
        };

        // Signatures last: they are by far the most expensive check.
        let digest = self.body.digest().ok_or(TxError::NonCanonical)?;
        for (i, (input, sig)) in inputs.iter().zip(&self.signatures).enumerate() {
            if !wots::verify(&input.pubkey, &digest, sig) {
                return Err(TxError::BadSignature(i));
            }
        }
        Ok(totals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::babybear::P;
    use crate::output::NONCE_LEN;

    fn key(n: u32) -> (SecretKey, PublicKey) {
        wots::keygen(BabyBear::new_array(core::array::from_fn(|i| {
            n * 100 + i as u32
        })))
    }

    /// An input spending an output of `amount` paid to key `n`.
    fn input(n: u32, amount: Amount) -> (SecretKey, Input) {
        let (sk, pubkey) = key(n);
        let nonce = [n as u8; NONCE_LEN];
        let commitment = output::commit(&pubkey_hash(&pubkey), amount, &nonce).unwrap();
        (
            sk,
            Input {
                commitment,
                position: n as u64,
                pubkey,
                amount,
                nonce,
            },
        )
    }

    fn out(amount: Amount, tag: u8) -> Output {
        Output {
            hash: [tag; 32],
            amount,
            nonce: [tag; NONCE_LEN],
        }
    }

    /// Two inputs (100 + 50) paying 120 + 20, so the fee is 10.
    fn tx() -> Transaction {
        let (sk1, in1) = input(1, 100);
        let (sk2, in2) = input(2, 50);
        let body = TxBody {
            inputs: vec![in1, in2],
            outputs: vec![out(120, 1), out(20, 2)],
        };
        Transaction::sign(body, vec![sk1, sk2]).unwrap()
    }

    /// Re-signs a modified body with the keys matching its inputs.
    fn resign(body: TxBody) -> Transaction {
        let keys = body
            .inputs
            .iter()
            .map(|i| (1..=4).map(key).find(|(_, pk)| *pk == i.pubkey).unwrap().0)
            .collect();
        Transaction::sign(body, keys).unwrap()
    }

    fn totals(inputs: Amount, outputs: Amount) -> Result<Totals, TxError> {
        Ok(Totals { inputs, outputs })
    }

    #[test]
    fn valid_transaction_returns_totals() {
        let t = tx().validate().unwrap();
        assert_eq!(
            t,
            Totals {
                inputs: 150,
                outputs: 140
            }
        );
        assert_eq!(t.fee(), 10);
    }

    #[test]
    fn block_reward_has_outputs_and_no_inputs() {
        let reward = Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs: vec![out(50, 1)],
            },
            vec![],
        )
        .unwrap();
        let t = reward.validate().unwrap();
        assert_eq!(
            t,
            Totals {
                inputs: 0,
                outputs: 50
            }
        );
        assert_eq!(t.fee(), -50);
    }

    #[test]
    fn outputs_may_exceed_inputs() {
        // The balance rule belongs to the block, not the transaction.
        let mut body = tx().body;
        body.outputs[0].amount = 131;
        let t = resign(body).validate().unwrap();
        assert_eq!(t.fee(), -1);
    }

    #[test]
    fn exact_spend_and_all_fee() {
        let mut body = tx().body;
        body.outputs[1].amount = 30;
        assert_eq!(resign(body.clone()).validate(), totals(150, 150));
        body.outputs.clear();
        assert_eq!(resign(body).validate(), totals(150, 0));
    }

    #[test]
    fn rejects_amount_overflow() {
        let mut body = tx().body;
        body.outputs = vec![out(u64::MAX, 1), out(1, 2)];
        assert_eq!(resign(body).validate(), Err(TxError::AmountOverflow));
    }

    #[test]
    fn rejects_structural_errors() {
        let t = tx();
        let mut missing_sig = t.clone();
        missing_sig.signatures.pop();
        assert_eq!(
            missing_sig.validate(),
            Err(TxError::SignatureCount {
                inputs: 2,
                signatures: 1
            })
        );

        let mut dup = t.body.clone();
        dup.inputs[1] = dup.inputs[0].clone();
        assert_eq!(resign(dup).validate(), Err(TxError::DuplicateInput(1)));
    }

    #[test]
    fn rejects_inflated_input_amount() {
        // Claiming more than the committed amount must fail, even when re-signed.
        let mut body = tx().body;
        body.inputs[0].amount = 1000;
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn rejects_wrong_nonce_for_commitment() {
        let mut body = tx().body;
        body.inputs[0].nonce[0] ^= 1;
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn rejects_wrong_key_for_commitment() {
        let mut body = tx().body;
        body.inputs[0].pubkey = key(3).1;
        assert_eq!(resign(body).validate(), Err(TxError::CommitmentMismatch(0)));
    }

    #[test]
    fn signatures_cover_every_body_field() {
        let t = tx();
        let tampered: [fn(&mut TxBody); 4] = [
            |b| b.outputs[0].hash[0] ^= 1,
            |b| b.outputs[0].nonce[0] ^= 1,
            |b| {
                b.outputs[0].amount -= 1;
                b.outputs[1].amount += 1;
            },
            |b| b.outputs.swap(0, 1),
        ];
        for f in tampered {
            let mut bad = t.clone();
            f(&mut bad.body);
            assert_eq!(bad.validate(), Err(TxError::BadSignature(0)));
        }
    }

    #[test]
    fn rejects_swapped_signatures() {
        let mut t = tx();
        t.signatures.swap(0, 1);
        assert_eq!(t.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn digest_covers_counts_and_commitments() {
        let body = tx().body;
        let d = body.digest().unwrap();
        // Moving an output into a fresh slot changes the counts.
        let mut fewer = body.clone();
        fewer.outputs.pop();
        assert_ne!(fewer.digest().unwrap(), d);
        // Same commitments, swapped order: a different digest.
        let mut swapped = body.clone();
        swapped.outputs.swap(0, 1);
        assert_ne!(swapped.digest().unwrap(), d);
        // The spent position is signed: the same commitment elsewhere in
        // the tree is a different message.
        let mut moved = body.clone();
        moved.inputs[0].position += 1 << 30;
        assert_ne!(moved.digest().unwrap(), d);
        let mut moved = body.clone();
        moved.inputs[1].position += 1;
        assert_ne!(moved.digest().unwrap(), d);
        let t = tx();
        let mut replayed = t.clone();
        replayed.body.inputs[0].position = 999;
        assert_eq!(replayed.validate(), Err(TxError::BadSignature(0)));
    }

    #[test]
    fn rejects_empty_transaction() {
        let empty = Transaction::sign(
            TxBody {
                inputs: vec![],
                outputs: vec![],
            },
            vec![],
        )
        .unwrap();
        assert_eq!(empty.validate(), Err(TxError::Empty));
    }

    #[test]
    fn rejects_non_canonical_output_hash() {
        let mut body = tx().body;
        body.outputs[0].hash[..4].copy_from_slice(&P.to_le_bytes());
        assert_eq!(body.digest(), None);
        let (sk1, _) = input(1, 100);
        let (sk2, _) = input(2, 50);
        assert_eq!(
            Transaction::sign(body.clone(), vec![sk1, sk2]),
            Err(TxError::NonCanonical)
        );
        let mut t = tx();
        t.body = body;
        assert_eq!(t.validate(), Err(TxError::NonCanonical));
    }
}
