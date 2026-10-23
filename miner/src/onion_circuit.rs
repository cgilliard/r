//! Onion circuits' cryptography ([`crate::net::node`] runs them): the
//! handshake that adds a hop, and the layered cells that cross them.
//!
//! **Handshake.** To add a relay, the client sends it a random challenge
//! and a fresh ML-KEM-512 public key (`CREATE`; through the circuit so far
//! for every hop after the first). The relay seals a secret to that key and
//! signs the challenge, the key and the ciphertext with its ML-DSA-44
//! identity key ([`crate::onion::identity`]); it answers its public key,
//! the ciphertext and the signature ([`CREATED_LEN`] bytes, in
//! [`FRAGMENT`]-byte pieces). The client checks the public key's hash
//! against the consensus file and the signature, so it knows which relay
//! holds the other end; the secret, new for every hop of every circuit,
//! gives each direction its ChaCha20-Poly1305 key ([`Keys`]). The signature
//! covers the exchange, so no one in between can swap the key.
//!
//! **Cells.** Everything on a circuit is a [`CELL_LEN`]-byte cell. Going
//! out, the client wraps a message in one layer per hop, innermost for the
//! hop it's for: each layer is `len (2) || nonce (12) || ciphertext`, whose
//! plaintext is a command, a body length and the body ([`frame`]). A hop
//! opens its layer: `RELAY` passes the body (the next layer) on, padded
//! back to a whole cell; `EXTEND` adds the next hop; `DATA` is for it.
//! Coming back, the hop a message is from wraps it in its layer, and every
//! hop on the way adds its own (`RELAYED`); the client peels them in order,
//! which says which hop it came from. The innermost plaintext is padded so
//! outgoing cells are full at the first hop.

use crate::crypto::{chacha20poly1305 as aead, keccak, mldsa, mlkem};
use crate::mnemonic::sha256;

/// Bytes in a cell: what's left of a datagram after its header and the
/// circuit id.
pub const CELL_LEN: usize = crate::net::wire::CELL_LEN;
pub const CHALLENGE_LEN: usize = 32;
/// A relay's answer to `CREATE`: public key, ciphertext, signature.
pub const CREATED_LEN: usize = mldsa::PK_LEN + mlkem::CT_LEN + mldsa::SIG_LEN;
/// Bytes of the answer per piece.
pub const FRAGMENT: usize = 1000;
/// Pieces of an answer.
pub const FRAGMENTS: usize = CREATED_LEN.div_ceil(FRAGMENT);
/// Bytes a layer adds: length, nonce, tag.
pub const LAYER: usize = 2 + aead::NONCE_LEN + aead::TAG_LEN;
/// Bytes a frame adds: command, body length.
pub const FRAME: usize = 3;
/// Most hops a circuit has (2, then the rendezvous point).
pub const MAX_HOPS: usize = 3;
/// The plaintext a hop that answers pads to: room for every hop's layer
/// on the way back.
pub const BACK_PLAINTEXT: usize = CELL_LEN - LAYER - (MAX_HOPS - 1) * (LAYER + FRAME);
/// The longest body a `DATA` message can carry either way.
pub const MAX_DATA: usize = BACK_PLAINTEXT - FRAME;
// Room for a post message: a key (800 bytes) or a sealed slate.
const _: () = assert!(MAX_DATA >= 1000);

/// Commands going out.
pub mod cmd {
    pub const RELAY: u8 = 1;
    pub const EXTEND: u8 = 2;
    pub const DATA: u8 = 3;
}

/// What a message coming back is.
pub mod back {
    /// Another hop's, wrapped: peel on.
    pub const RELAYED: u8 = 1;
    /// A piece of the next hop's handshake answer.
    pub const EXTENDED: u8 = 2;
    pub const DATA: u8 = 3;
}

const CONTEXT: &[u8] = b"r onion circuit";

/// A hop's keys: out (to the relay) and back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keys {
    pub out: [u8; 32],
    pub back: [u8; 32],
}

fn transcript(challenge: &[u8; CHALLENGE_LEN], ek: &[u8; mlkem::EK_LEN], ct: &[u8]) -> [u8; 32] {
    keccak::sha3_256(&[b"r onion handshake", challenge, ek, ct])
}

fn derive(secret: &[u8; 32], transcript: &[u8; 32]) -> Keys {
    Keys {
        out: keccak::sha3_256(&[b"r onion out", secret, transcript]),
        back: keccak::sha3_256(&[b"r onion back", secret, transcript]),
    }
}

/// A relay's side of the handshake: its answer ([`CREATED_LEN`] bytes) and
/// the hop's keys, from the client's challenge and key, with randomness `m`
/// (the secret) and `rnd` (the signature's). `None` if the key isn't valid.
pub fn answer(
    pk: &[u8; mldsa::PK_LEN],
    sk: &[u8; mldsa::SK_LEN],
    challenge: &[u8; CHALLENGE_LEN],
    ek: &[u8; mlkem::EK_LEN],
    m: &[u8; 32],
    rnd: &[u8; 32],
) -> Option<(Vec<u8>, Keys)> {
    if !mlkem::check_ek(ek) {
        return None;
    }
    let (ct, secret) = mlkem::encapsulate(ek, m)?;
    let t = transcript(challenge, ek, &ct);
    let sig = mldsa::sign(&t, CONTEXT, sk, Some(rnd)).expect("short context");
    Some(([&pk[..], &ct, &sig].concat(), derive(&secret, &t)))
}

/// The client's side: checks a relay's answer (its key must hash to
/// `key_hash`) to our `challenge` and `ek` (whose secret key is `dk`), and
/// gives the hop's keys.
pub fn finish(
    key_hash: &[u8; 32],
    challenge: &[u8; CHALLENGE_LEN],
    ek: &[u8; mlkem::EK_LEN],
    dk: &[u8; mlkem::DK_LEN],
    created: &[u8],
) -> Option<Keys> {
    if created.len() != CREATED_LEN {
        return None;
    }
    let (pk, rest) = created.split_at(mldsa::PK_LEN);
    let (ct, sig) = rest.split_at(mlkem::CT_LEN);
    let pk: &[u8; mldsa::PK_LEN] = pk.try_into().ok()?;
    if sha256(pk) != *key_hash {
        return None;
    }
    let t = transcript(challenge, ek, ct);
    if !mldsa::verify(&t, CONTEXT, sig, pk) {
        return None;
    }
    let secret = mlkem::decapsulate(dk, ct.try_into().ok()?)?;
    Some(derive(&secret, &t))
}

/// `command || body length || body`, zero-padded to `pad_to` bytes.
pub fn frame(command: u8, body: &[u8], pad_to: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pad_to.max(FRAME + body.len()));
    out.push(command);
    out.extend((body.len() as u16).to_le_bytes());
    out.extend(body);
    out.resize(out.len().max(pad_to), 0);
    out
}

/// The command and body of a frame.
pub fn unframe(pt: &[u8]) -> Option<(u8, &[u8])> {
    let (&command, rest) = pt.split_first()?;
    let len = u16::from_le_bytes(rest.get(..2)?.try_into().ok()?) as usize;
    Some((command, rest.get(2..2 + len)?))
}

fn nonce(n: u64) -> [u8; aead::NONCE_LEN] {
    let mut out = [0u8; aead::NONCE_LEN];
    out[..8].copy_from_slice(&n.to_le_bytes());
    out
}

/// A layer around `pt` with `key`, message number `n` (each key's numbers
/// must differ): `len || nonce || ciphertext`.
pub fn seal(key: &[u8; 32], n: u64, pt: &[u8]) -> Vec<u8> {
    let ct = aead::seal(key, &nonce(n), &[], pt);
    let len = (aead::NONCE_LEN + ct.len()) as u16;
    [&len.to_le_bytes()[..], &nonce(n), &ct].concat()
}

/// The layer at the start of `bytes` (a cell or a layer), opened.
pub fn open(key: &[u8; 32], bytes: &[u8]) -> Option<Vec<u8>> {
    let len = u16::from_le_bytes(bytes.get(..2)?.try_into().ok()?) as usize;
    let layer = bytes.get(2..2 + len)?;
    let (n, ct) = layer.split_at_checked(aead::NONCE_LEN)?;
    aead::open(key, n.try_into().ok()?, &[], ct)
}

/// A layer as a cell (zero-padded). `None` if it's too long.
pub fn cell(layer: &[u8]) -> Option<Box<[u8; CELL_LEN]>> {
    let mut out = Box::new([0u8; CELL_LEN]);
    out.get_mut(..layer.len())?.copy_from_slice(layer);
    Some(out)
}

/// The plaintext a message for hop `depth` (from 0) pads to, so the cell
/// is full at the first hop.
pub fn out_plaintext(depth: usize) -> usize {
    CELL_LEN - LAYER - depth * (LAYER + FRAME)
}

/// A message for hop `depth` of a circuit with these `keys` (`n` per
/// hop, advanced): the cell for the first hop.
pub fn wrap_out(
    hops: &mut [(Keys, u64)],
    depth: usize,
    command: u8,
    body: &[u8],
) -> Option<Box<[u8; CELL_LEN]>> {
    let mut pt = frame(command, body, out_plaintext(depth));
    for i in (0..=depth).rev() {
        let (keys, n) = &mut hops[i];
        let layer = seal(&keys.out, *n, &pt);
        *n += 1;
        if i == 0 {
            return cell(&layer);
        }
        pt = frame(cmd::RELAY, &layer, 0);
    }
    None
}

/// Peels a cell coming back: the hop it's from (from 0), and its kind
/// and body.
pub fn unwrap_back(keys: &[Keys], cell: &[u8]) -> Option<(usize, u8, Vec<u8>)> {
    let mut layer = cell.to_vec();
    for (i, k) in keys.iter().enumerate() {
        let pt = open(&k.back, &layer)?;
        let (kind, body) = unframe(&pt)?;
        if kind != back::RELAYED {
            return Some((i, kind, body.to_vec()));
        }
        layer = body.to_vec();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relay(seed: u8) -> ([u8; mldsa::PK_LEN], [u8; mldsa::SK_LEN]) {
        mldsa::keygen(&[seed; 32])
    }

    /// The client and relay agree on keys; anything else fails.
    fn handshake(seed: u8) -> (Keys, Keys) {
        let (pk, sk) = relay(seed);
        let (ek, dk) = mlkem::keygen(&[seed; 32], &[seed ^ 1; 32]);
        let challenge = [seed; 32];
        let (created, relay_keys) = answer(&pk, &sk, &challenge, &ek, &[5; 32], &[6; 32]).unwrap();
        assert_eq!(created.len(), CREATED_LEN);
        let hash = sha256(&pk);
        let client = finish(&hash, &challenge, &ek, &dk, &created).unwrap();
        assert_eq!(client, relay_keys);
        assert_ne!(client.out, client.back);
        // Another relay's key, another challenge, a changed answer: refused.
        assert!(finish(&[0; 32], &challenge, &ek, &dk, &created).is_none());
        assert!(finish(&hash, &[0; 32], &ek, &dk, &created).is_none());
        for at in [10, mldsa::PK_LEN + 10, CREATED_LEN - 10] {
            let mut bad = created.clone();
            bad[at] ^= 1;
            assert!(finish(&hash, &challenge, &ek, &dk, &bad).is_none(), "{at}");
        }
        (client, relay_keys)
    }

    #[test]
    fn a_handshake_proves_the_relay_and_agrees_keys() {
        handshake(1);
        assert_eq!(FRAGMENTS, 5);
    }

    #[test]
    fn cells_cross_three_hops_each_way() {
        let keys: Vec<Keys> = (1..=3).map(|s| handshake(s).0).collect();
        let mut hops: Vec<(Keys, u64)> = keys.iter().map(|k| (*k, 0)).collect();
        let body = vec![7u8; MAX_DATA];
        // Out to the third hop: each opens its layer and passes the rest.
        let mut c = wrap_out(&mut hops, 2, cmd::DATA, &body).unwrap();
        for (i, k) in keys.iter().enumerate() {
            let pt = open(&k.out, &c[..]).unwrap();
            let (command, inner) = unframe(&pt).unwrap();
            if i < 2 {
                assert_eq!(command, cmd::RELAY);
                // The first hop's cell is full.
                if i == 0 {
                    assert_eq!(2 + u16::from_le_bytes([c[0], c[1]]) as usize, CELL_LEN);
                }
                c = cell(inner).unwrap();
            } else {
                assert_eq!((command, inner), (cmd::DATA, &body[..]));
            }
        }
        // A wrong hop can't open it.
        let c = wrap_out(&mut hops, 0, cmd::DATA, b"hi").unwrap();
        assert!(open(&keys[1].out, &c[..]).is_none());

        // Back from the third hop, wrapped by the others on the way.
        let mut layer = seal(&keys[2].back, 0, &frame(back::DATA, &body, BACK_PLAINTEXT));
        for k in keys[..2].iter().rev() {
            layer = seal(&k.back, 0, &frame(back::RELAYED, &layer, 0));
        }
        let c = cell(&layer).unwrap();
        assert_eq!(
            unwrap_back(&keys, &c[..]),
            Some((2, back::DATA, body.clone()))
        );
        // From the first hop.
        let c = cell(&seal(
            &keys[0].back,
            1,
            &frame(back::EXTENDED, b"x", BACK_PLAINTEXT),
        ))
        .unwrap();
        assert_eq!(
            unwrap_back(&keys, &c[..]),
            Some((0, back::EXTENDED, b"x".to_vec()))
        );
    }
}
