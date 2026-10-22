//! The saved address book: the addresses that have worked, in LMDB
//! (database `peers`, one key per address, as text), so a node can find
//! its peers again after a restart. Saving replaces the whole book in one
//! transaction.

use std::net::SocketAddr;

use heed::types::{Bytes, Unit};
use heed::{Database, Env};

#[derive(Clone, Debug)]
pub struct PeerStore {
    env: Env,
    db: Database<Bytes, Unit>,
}

impl PeerStore {
    pub fn open(env: &Env) -> heed::Result<Self> {
        let mut txn = env.write_txn()?;
        let db = env.create_database(&mut txn, Some("peers"))?;
        txn.commit()?;
        Ok(Self {
            env: env.clone(),
            db,
        })
    }

    /// The saved addresses (any that don't parse are skipped).
    pub fn load(&self) -> heed::Result<Vec<SocketAddr>> {
        let txn = self.env.read_txn()?;
        let mut out = Vec::new();
        for entry in self.db.iter(&txn)? {
            if let Some(a) = std::str::from_utf8(entry?.0)
                .ok()
                .and_then(|s| s.parse().ok())
            {
                out.push(a);
            }
        }
        Ok(out)
    }

    /// Replaces the saved addresses with `addrs`.
    pub fn save(&self, addrs: &[SocketAddr]) -> heed::Result<()> {
        let mut txn = self.env.write_txn()?;
        self.db.clear(&mut txn)?;
        for a in addrs {
            self.db.put(&mut txn, a.to_string().as_bytes(), &())?;
        }
        txn.commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test::TempEnv;

    #[test]
    fn the_book_is_replaced_and_survives_reopening() {
        let env = TempEnv::new();
        let a: SocketAddr = "10.1.2.3:7333".parse().unwrap();
        let b: SocketAddr = "[2001:db8::1]:9".parse().unwrap();
        let store = PeerStore::open(env.env()).unwrap();
        assert_eq!(store.load().unwrap(), vec![]);
        store.save(&[a, b]).unwrap();
        store.save(&[b]).unwrap();
        assert_eq!(PeerStore::open(env.env()).unwrap().load().unwrap(), vec![b]);
    }
}
