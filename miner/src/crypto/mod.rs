//! Cryptography for the onion-routing layer, written from scratch:
//! SHA-3 and SHAKE ([`keccak`]), ChaCha20-Poly1305 for symmetric
//! encryption ([`chacha20poly1305`]), ML-KEM-512 (Kyber) for key exchange
//! ([`mlkem`]), and ML-DSA-44 (Dilithium) for signatures ([`mldsa`]).

pub mod chacha20poly1305;
pub mod keccak;
pub mod mldsa;
pub mod mlkem;
