//! The LMDB environment holding all on-disk state.

use std::path::Path;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, PutFlags};

/// Maximum database size. LMDB only reserves address space for this; the
/// file grows as data is written.
pub const MAP_SIZE: usize = 1 << 40;
/// Named databases the environment can hold.
pub const MAX_DBS: u32 = 32;

/// Opens (creating if needed) the environment in `path`.
pub fn open_env(path: &Path) -> heed::Result<Env> {
    std::fs::create_dir_all(path)?;
    // SAFETY: LMDB files must not be modified other than through LMDB while
    // mapped. Only this process's environment touches files in the data
    // directory.
    unsafe {
        EnvOpenOptions::new()
            .map_size(MAP_SIZE)
            .max_dbs(MAX_DBS)
            .open(path)
    }
}

/// Rewrites the environment in `path` (which must not be open): copies
/// every database in key order into a new file, appending, so pages come
/// out full (LMDB's own compacting copy drops free pages but keeps
/// half-empty ones, which deletions leave), and replaces the old file.
/// Returns the file's size before and after.
pub fn compact(path: &Path) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    // Entries per write transaction, to bound its dirty pages.
    const BATCH: usize = 100_000;
    let file = path.join("data.mdb");
    let before = std::fs::metadata(&file)?.len();
    let dir = path.join("compacting");
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let old = open_env(path)?;
    let new = open_env(&dir)?;
    let rtxn = old.read_txn()?;
    let names: Vec<String> = old
        .open_database::<Bytes, Bytes>(&rtxn, None)?
        .ok_or("no main database")?
        .iter(&rtxn)?
        .map(|e| Ok(String::from_utf8(e?.0.to_vec())?))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let dbs = names
        .iter()
        .map(|n| {
            old.open_database::<Bytes, Bytes>(&rtxn, Some(n))?
                .ok_or("missing database".into())
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    rtxn.commit()?;
    let rtxn = old.read_txn()?;
    for (name, src) in names.iter().zip(dbs) {
        let mut wtxn = new.write_txn()?;
        let dst: Database<Bytes, Bytes> = new.create_database(&mut wtxn, Some(name))?;
        for (n, entry) in src.iter(&rtxn)?.enumerate() {
            let (k, v) = entry?;
            dst.put_with_flags(&mut wtxn, PutFlags::APPEND, k, v)?;
            if (n + 1) % BATCH == 0 {
                wtxn.commit()?;
                wtxn = new.write_txn()?;
            }
        }
        wtxn.commit()?;
    }
    drop(rtxn);
    old.prepare_for_closing().wait();
    new.prepare_for_closing().wait();
    std::fs::rename(dir.join("data.mdb"), &file)?;
    std::fs::remove_dir_all(&dir)?;
    Ok((before, std::fs::metadata(&file)?.len()))
}

#[cfg(test)]
pub(crate) mod test {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    use heed::Env;

    /// An environment in a fresh temporary directory, removed on drop.
    pub struct TempEnv {
        pub env: Option<Env>,
        pub path: PathBuf,
    }

    impl TempEnv {
        pub fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "r-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&path);
            let env = super::open_env(&path).unwrap();
            Self {
                env: Some(env),
                path,
            }
        }

        pub fn env(&self) -> &Env {
            self.env.as_ref().unwrap()
        }

        /// Closes (if open) and reopens the environment, as a restart
        /// would.
        pub fn reopen(&mut self) {
            if let Some(env) = self.env.take() {
                env.prepare_for_closing().wait();
            }
            self.env = Some(super::open_env(&self.path).unwrap());
        }
    }

    impl Drop for TempEnv {
        fn drop(&mut self) {
            if let Some(env) = self.env.take() {
                env.prepare_for_closing().wait();
            }
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Compaction keeps every database's entries and shrinks a file that
    /// deletions left half empty.
    #[test]
    fn compaction_keeps_the_data_in_a_smaller_file() {
        use heed::types::Bytes;
        let mut tmp = TempEnv::new();
        let mut txn = tmp.env().write_txn().unwrap();
        let a: heed::Database<Bytes, Bytes> =
            tmp.env().create_database(&mut txn, Some("a")).unwrap();
        let b: heed::Database<Bytes, Bytes> =
            tmp.env().create_database(&mut txn, Some("b")).unwrap();
        for i in 0u32..50_000 {
            // Scattered keys, as hashes are.
            let k = i.wrapping_mul(2_654_435_761).to_be_bytes();
            a.put(&mut txn, &k, &[7; 40]).unwrap();
            b.put(&mut txn, &i.to_be_bytes(), &[i as u8]).unwrap();
        }
        txn.commit().unwrap();
        let mut txn = tmp.env().write_txn().unwrap();
        for i in (0u32..50_000).filter(|i| i % 3 != 0) {
            a.delete(&mut txn, &i.wrapping_mul(2_654_435_761).to_be_bytes())
                .unwrap();
        }
        txn.commit().unwrap();
        tmp.env.take().unwrap().prepare_for_closing().wait();
        let (before, after) = super::compact(&tmp.path).unwrap();
        assert!(after * 2 < before, "{before} -> {after}");
        tmp.reopen();
        let txn = tmp.env().read_txn().unwrap();
        let a: heed::Database<Bytes, Bytes> =
            tmp.env().open_database(&txn, Some("a")).unwrap().unwrap();
        let b: heed::Database<Bytes, Bytes> =
            tmp.env().open_database(&txn, Some("b")).unwrap().unwrap();
        assert_eq!(a.len(&txn).unwrap(), 16_667);
        assert_eq!(b.len(&txn).unwrap(), 50_000);
        assert_eq!(
            a.get(&txn, &3u32.wrapping_mul(2_654_435_761).to_be_bytes())
                .unwrap(),
            Some(&[7; 40][..])
        );
    }
}
