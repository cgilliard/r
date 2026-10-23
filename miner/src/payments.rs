//! Payments between online wallets over the network: a wallet's payment
//! address, and the encrypted slates sent to it.
//!
//! **Address.** A wallet's payment keys are an ML-KEM-512 key pair
//! ([`crate::crypto::mlkem`]) derived from its recovery entropy (apart from
//! its relay key, [`crate::onion::identity`]), so restoring the wallet
//! restores the address. The address is the SHA-256 of the public
//! (encapsulation) key, written in bech32m ([`Address`]: `tab1…`, with a
//! checksum that catches typos). A sender gets the key from the network,
//! checks it against the address, and encrypts to it, so no one in between
//! can read or substitute a slate. The same address works in init and in
//! onion mode ([`crate::onion::Mode`]).
//!
//! **Slates.** What travels is a [`Payload`]: a payment request (the
//! slate, and the sender's address for the answer), the receiver's response,
//! or a note that the receiver holds it for approval. Each is sealed to the
//! recipient's key ([`seal`]): a fresh ML-KEM ciphertext, and the payload
//! under ChaCha20-Poly1305 with the shared secret (the recipient's address
//! bound as associated data).

use std::fmt;
use std::str::FromStr;

use crate::crypto::{chacha20poly1305 as aead, keccak, mlkem};
use crate::mnemonic::{ENTROPY_LEN, sha256};
use crate::output::Output;
use crate::wallet::{Response, SlateId, output_bytes, output_from_bytes};

/// An address's text prefix.
pub const HRP: &str = "tab";
/// The address version, its first data symbol.
const VERSION: u8 = 0;
/// Bytes a sealed payload adds: the ML-KEM ciphertext, and the tag.
pub const SEALED_OVERHEAD: usize = mlkem::CT_LEN + aead::TAG_LEN;

/// A payment address: the SHA-256 of a wallet's payment public key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Address(pub [u8; 32]);

impl Address {
    /// The address of public key `ek`.
    pub fn of(ek: &[u8; mlkem::EK_LEN]) -> Self {
        Self(sha256(ek))
    }
}

const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";
/// bech32m's checksum constant (BIP 350).
const BECH32M: u32 = 0x2bc8_30a3;

fn polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [
        0x3b6a_57b2,
        0x2650_8e6d,
        0x1ea1_19fa,
        0x3d42_33dd,
        0x2a14_62b3,
    ];
    let mut chk = 1u32;
    for &v in values {
        let top = chk >> 25;
        chk = (chk & 0x1ff_ffff) << 5 ^ v as u32;
        for (i, g) in GEN.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

fn hrp_expand(hrp: &str) -> Vec<u8> {
    let b = hrp.as_bytes();
    b.iter()
        .map(|c| c >> 5)
        .chain([0])
        .chain(b.iter().map(|c| c & 31))
        .collect()
}

/// `data` (5-bit symbols) with `hrp` as bech32m text.
fn bech32m_encode(hrp: &str, data: &[u8]) -> String {
    let mut values = hrp_expand(hrp);
    values.extend(data);
    values.extend([0; 6]);
    let m = polymod(&values) ^ BECH32M;
    let checksum = (0..6).map(|i| ((m >> (5 * (5 - i))) & 31) as u8);
    let mut out = format!("{hrp}1");
    out.extend(
        data.iter()
            .copied()
            .chain(checksum)
            .map(|d| CHARSET[d as usize] as char),
    );
    out
}

/// The prefix and 5-bit symbols of bech32m text (one case only), its
/// checksum checked.
fn bech32m_decode(text: &str) -> Result<(String, Vec<u8>), String> {
    if text.len() > 90 {
        return Err("too long".into());
    }
    if text.chars().any(|c| c.is_ascii_lowercase()) && text.chars().any(|c| c.is_ascii_uppercase())
    {
        return Err("mixes upper and lower case".into());
    }
    let text = text.to_ascii_lowercase();
    let at = text.rfind('1').ok_or("no separator")?;
    let (hrp, data) = (&text[..at], &text[at + 1..]);
    if hrp.is_empty() || data.len() < 6 || hrp.bytes().any(|c| !(33..=126).contains(&c)) {
        return Err("malformed".into());
    }
    let data: Vec<u8> = data
        .bytes()
        .map(|c| CHARSET.iter().position(|&x| x == c).map(|p| p as u8))
        .collect::<Option<_>>()
        .ok_or("a character that isn't bech32")?;
    let mut values = hrp_expand(hrp);
    values.extend(&data);
    if polymod(&values) != BECH32M {
        return Err("bad checksum (a typo?)".into());
    }
    Ok((hrp.to_string(), data[..data.len() - 6].to_vec()))
}

/// Regroups bits: `from`-bit groups into `to`-bit ones (padding the last
/// with zeros if `pad`; otherwise leftover bits must be zero padding).
fn regroup(data: &[u8], from: u32, to: u32, pad: bool) -> Option<Vec<u8>> {
    let (mut acc, mut bits, mut out) = (0u32, 0u32, Vec::new());
    for &d in data {
        if (d as u32) >> from != 0 {
            return None;
        }
        acc = acc << from | d as u32;
        bits += from;
        while bits >= to {
            bits -= to;
            out.push(((acc >> bits) & ((1 << to) - 1)) as u8);
        }
    }
    if pad {
        if bits > 0 {
            out.push(((acc << (to - bits)) & ((1 << to) - 1)) as u8);
        }
    } else if bits >= from || (acc << (to - bits)) & ((1 << to) - 1) != 0 {
        return None;
    }
    Some(out)
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut data = vec![VERSION];
        data.extend(regroup(&self.0, 8, 5, true).expect("bytes"));
        f.write_str(&bech32m_encode(HRP, &data))
    }
}

impl FromStr for Address {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let (hrp, data) = bech32m_decode(s)?;
        if hrp != HRP {
            return Err(format!(
                "not a payment address (it starts {hrp}1, not {HRP}1)"
            ));
        }
        let (&version, rest) = data.split_first().ok_or("empty")?;
        if version != VERSION {
            return Err(format!("unknown address version {version}"));
        }
        let bytes = regroup(rest, 5, 8, false).ok_or("malformed")?;
        Ok(Self(bytes.try_into().map_err(|_| "the wrong length")?))
    }
}

/// A wallet's payment keys and address.
#[derive(Clone)]
pub struct PaymentKeys {
    pub ek: [u8; mlkem::EK_LEN],
    dk: [u8; mlkem::DK_LEN],
    pub address: Address,
}

impl fmt::Debug for PaymentKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PaymentKeys({})", self.address)
    }
}

impl PaymentKeys {
    /// The payment keys of the wallet with recovery `entropy`.
    pub fn from_entropy(entropy: &[u8; ENTROPY_LEN]) -> Self {
        let mut seed = [0u8; 64];
        keccak::shake256(&[b"r payment key (ML-KEM-512)", entropy], &mut seed);
        let (d, z) = seed.split_at(32);
        let (ek, dk) = mlkem::keygen(d.try_into().expect("32"), z.try_into().expect("32"));
        Self {
            ek,
            dk,
            address: Address::of(&ek),
        }
    }

    /// The secret in `ct`, sealed to our key.
    pub fn decapsulate(&self, ct: &[u8; mlkem::CT_LEN]) -> [u8; mlkem::SECRET_LEN] {
        mlkem::decapsulate(&self.dk, ct).expect("our own key checks")
    }

    /// Opens a payload [`seal`]ed to us.
    pub fn open(&self, ct: &[u8; mlkem::CT_LEN], data: &[u8]) -> Option<Vec<u8>> {
        let key = self.decapsulate(ct);
        aead::open(&key, &[0; aead::NONCE_LEN], &self.address.0, data)
    }
}

/// Seals `payload` to the holder of `ek` (whose address is `to`), with the
/// random `m`: the ML-KEM ciphertext, and the encrypted payload. `None` if
/// `ek` isn't a valid key.
pub fn seal(
    ek: &[u8; mlkem::EK_LEN],
    payload: &[u8],
    m: &[u8; 32],
) -> Option<([u8; mlkem::CT_LEN], Vec<u8>)> {
    let (ct, key) = mlkem::encapsulate(ek, m)?;
    // A fresh key every time, so a zero nonce is safe.
    let to = Address::of(ek);
    Some((ct, aead::seal(&key, &[0; aead::NONCE_LEN], &to.0, payload)))
}

/// What proves to a host that a wallet holds the key of `address`: a hash
/// of the secret the host sealed to it.
pub fn register_proof(address: &Address, secret: &[u8; mlkem::SECRET_LEN]) -> [u8; 32] {
    sha256(&[&b"r payment address registration"[..], &address.0, secret].concat())
}

/// What goes between two wallets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payload {
    /// A payment: the slate, and the sender's address for the answer.
    Request {
        id: SlateId,
        amount: u64,
        fee: u64,
        reply: Address,
    },
    /// The receiver's answer: its output.
    Response(Response),
    /// The receiver holds the request until its user accepts it.
    Held { id: SlateId },
}

impl Payload {
    pub fn id(&self) -> SlateId {
        match self {
            Self::Request { id, .. } | Self::Held { id } => *id,
            Self::Response(r) => r.id,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::Request {
                id,
                amount,
                fee,
                reply,
            } => [
                &[1][..],
                id,
                &amount.to_le_bytes(),
                &fee.to_le_bytes(),
                &reply.0,
            ]
            .concat(),
            Self::Response(r) => [
                &[2][..],
                &r.id,
                &r.amount.to_le_bytes(),
                &r.fee.to_le_bytes(),
                &output_bytes(&r.output),
            ]
            .concat(),
            Self::Held { id } => [&[3][..], id].concat(),
        }
    }

    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let (&kind, rest) = b.split_first()?;
        let id: SlateId = rest.get(..16)?.try_into().ok()?;
        let rest = &rest[16..];
        let number = |at: usize| -> Option<u64> {
            Some(u64::from_le_bytes(rest.get(at..at + 8)?.try_into().ok()?))
        };
        match kind {
            1 if rest.len() == 16 + 32 => Some(Self::Request {
                id,
                amount: number(0)?,
                fee: number(8)?,
                reply: Address(rest[16..].try_into().ok()?),
            }),
            2 => {
                let output: Output = output_from_bytes(rest.get(16..)?)?;
                output.record()?;
                Some(Self::Response(Response {
                    id,
                    amount: number(0)?,
                    fee: number(8)?,
                    output,
                }))
            }
            3 if rest.is_empty() => Some(Self::Held { id }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bech32m_matches_bip350() {
        // BIP 350's valid bech32m strings.
        for s in [
            "A1LQFN3A",
            "a1lqfn3a",
            "abcdef1l7aum6echk45nj3s0wdvt2fg8x9yrzpqzd3ryx",
            "split1checkupstagehandshakeupstreamerranterredcaperredlc445v",
            "?1v759aa",
        ] {
            let (hrp, data) = bech32m_decode(s).unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_eq!(bech32m_encode(&hrp, &data), s.to_ascii_lowercase());
        }
        // And invalid ones.
        for s in [
            "pzry9x0s0muk",
            "1pzry9x0s0muk",
            "a1lqfn3A",
            "abcdef1l7aum6echk45nj3s0wdvt2fg8x9yrzpqzd3ryy",
            // bech32 (not m): BIP 173's "a12uel5l".
            "a12uel5l",
        ] {
            assert!(bech32m_decode(s).is_err(), "{s}");
        }
    }

    #[test]
    fn addresses_round_trip_and_typos_are_caught() {
        let keys = PaymentKeys::from_entropy(&[7; ENTROPY_LEN]);
        let text = keys.address.to_string();
        assert!(text.starts_with("tab1q"), "{text}");
        assert_eq!(text.len(), 3 + 1 + 1 + 52 + 6);
        assert_eq!(text.parse::<Address>(), Ok(keys.address));
        assert_eq!(
            text.to_ascii_uppercase().parse::<Address>(),
            Ok(keys.address)
        );
        // Every single-character change is caught.
        for i in 4..text.len() {
            for c in CHARSET.iter().map(|&c| c as char) {
                if text.as_bytes()[i] as char != c {
                    let mut typo = text.clone();
                    typo.replace_range(i..i + 1, &c.to_string());
                    assert!(typo.parse::<Address>().is_err(), "{typo}");
                }
            }
        }
        assert!(text.replacen("tab", "tbb", 1).parse::<Address>().is_err());
        // The same words give the same address; others another.
        assert_eq!(PaymentKeys::from_entropy(&[7; ENTROPY_LEN]).ek, keys.ek);
        assert_ne!(
            PaymentKeys::from_entropy(&[8; ENTROPY_LEN]).address,
            keys.address
        );
    }

    #[test]
    fn only_the_recipient_opens_a_sealed_payload() {
        let bob = PaymentKeys::from_entropy(&[1; ENTROPY_LEN]);
        let eve = PaymentKeys::from_entropy(&[2; ENTROPY_LEN]);
        let payload = Payload::Request {
            id: [3; 16],
            amount: 5,
            fee: 1,
            reply: eve.address,
        };
        let bytes = payload.to_bytes();
        assert_eq!(Payload::from_bytes(&bytes), Some(payload));
        let (ct, data) = seal(&bob.ek, &bytes, &[9; 32]).unwrap();
        assert_eq!(data.len(), bytes.len() + aead::TAG_LEN);
        assert_eq!(bob.open(&ct, &data), Some(bytes.clone()));
        assert_eq!(eve.open(&ct, &data), None);
        let mut bad = data.clone();
        bad[0] ^= 1;
        assert_eq!(bob.open(&ct, &bad), None);
        // Registration: only the key holder knows the secret.
        let (ct, secret) = mlkem::encapsulate(&bob.ek, &[4; 32]).unwrap();
        assert_eq!(
            register_proof(&bob.address, &bob.decapsulate(&ct)),
            register_proof(&bob.address, &secret)
        );
        assert_ne!(eve.decapsulate(&ct), secret);
        assert_eq!(
            Payload::from_bytes(&Payload::Held { id: [1; 16] }.to_bytes()),
            Some(Payload::Held { id: [1; 16] })
        );
        assert_eq!(Payload::from_bytes(&bytes[..bytes.len() - 1]), None);
    }
}
