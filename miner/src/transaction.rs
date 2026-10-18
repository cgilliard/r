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

use crate::hash::{self, Hash, domain};
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
    pub pubkey: PublicKey,
    pub amount: Amount,
    pub nonce: Nonce,
}

impl Input {
    /// Returns `true` if the revealed pubkey, amount and nonce open `commitment`.
    pub fn opens_commitment(&self) -> bool {
        output::commit(&pubkey_hash(&self.pubkey), self.amount, &self.nonce) == self.commitment
    }
}

/// Everything the signatures commit to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxBody {
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
}

impl TxBody {
    /// Canonical encoding, all integers little-endian:
    /// `n_inputs: u32`, then per input `commitment | param | root | amount | nonce`;
    /// `n_outputs: u32`, then per output `hash | amount | nonce`.
    /// Field elements are written as canonical `u32`s.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&count(self.inputs.len()).to_le_bytes());
        for input in &self.inputs {
            out.extend_from_slice(&input.commitment.0);
            for x in input.pubkey.param.iter().chain(&input.pubkey.root) {
                out.extend_from_slice(&x.as_u32().to_le_bytes());
            }
            out.extend_from_slice(&input.amount.to_le_bytes());
            out.extend_from_slice(&input.nonce);
        }
        out.extend_from_slice(&count(self.outputs.len()).to_le_bytes());
        for o in &self.outputs {
            out.extend_from_slice(&o.hash);
            out.extend_from_slice(&o.amount.to_le_bytes());
            out.extend_from_slice(&o.nonce);
        }
        out
    }

    /// The message every input signs.
    pub fn digest(&self) -> Hash {
        hash::hash_bytes(domain::TX_BODY, &self.encode())
    }
}

fn count(n: usize) -> u32 {
    u32::try_from(n).expect("too many inputs or outputs")
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
        }
    }
}

impl std::error::Error for TxError {}

impl Transaction {
    /// Signs `body` with one key per input, in input order.
    ///
    /// # Panics
    /// If the number of keys differs from the number of inputs.
    pub fn sign(body: TxBody, keys: Vec<SecretKey>) -> Self {
        assert_eq!(keys.len(), body.inputs.len(), "one key per input");
        let digest = body.digest();
        let signatures = keys.into_iter().map(|k| k.sign(&digest)).collect();
        Self { body, signatures }
    }

    /// Checks everything that needs no chain state and returns the totals.
    pub fn validate(&self) -> Result<Totals, TxError> {
        let TxBody { inputs, outputs } = &self.body;
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
        let digest = self.body.digest();
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
    use crate::babybear::BabyBear;
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
        let commitment = output::commit(&pubkey_hash(&pubkey), amount, &nonce);
        (
            sk,
            Input {
                commitment,
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
        Transaction::sign(body, vec![sk1, sk2])
    }

    /// Re-signs a modified body with the keys matching its inputs.
    fn resign(body: TxBody) -> Transaction {
        let keys = body
            .inputs
            .iter()
            .map(|i| (1..=4).map(key).find(|(_, pk)| *pk == i.pubkey).unwrap().0)
            .collect();
        Transaction::sign(body, keys)
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
        );
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
    fn encoding_lengths() {
        let body = tx().body;
        let input_len = 32 + (wots::PARAM_LEN + hash::HASH_LEN) * 4 + 8 + NONCE_LEN;
        let output_len = 32 + 8 + NONCE_LEN;
        assert_eq!(body.encode().len(), 4 + 2 * input_len + 4 + 2 * output_len);
    }
}
