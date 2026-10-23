//! The LMDB environment holding all on-disk state.

use std::path::Path;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions, PutFlags};

/// Maximum database size. LMDB only reserves address space for this (no
/// memory); the file grows as data is written. Machines whose processes
/// get less address space (RISC-V in Sv39 mode has 256 GiB in all) get
/// what they can: [`map_size`].
pub const MAP_SIZE: usize = 1 << 40;
/// The least address space worth running with.
pub const MIN_MAP_SIZE: usize = 16 << 30;
/// Named databases the environment can hold.
pub const MAX_DBS: u32 = 32;

/// Whether `len` bytes of address space can be reserved now (reserved
/// without memory or swap, and released at once).
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn can_reserve(len: usize) -> bool {
    use std::ffi::c_void;
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
    // PROT_NONE; MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE.
    #[cfg(target_os = "linux")]
    const FLAGS: i32 = 0x02 | 0x20 | 0x4000;
    #[cfg(target_os = "macos")]
    const FLAGS: i32 = 0x02 | 0x1000 | 0x40;
    // SAFETY: an anonymous, inaccessible mapping, unmapped before
    // returning; nothing else refers to it.
    unsafe {
        let p = mmap(std::ptr::null_mut(), len, 0, FLAGS, -1, 0);
        if p as isize == -1 {
            return false;
        }
        munmap(p, len);
    }
    true
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn can_reserve(_len: usize) -> bool {
    true
}

/// The address space LMDB reserves: [`MAP_SIZE`] where it can be had;
/// else three quarters of the largest range that can (the rest is left
/// for the heap, stacks and the PoW dataset), but not below
/// [`MIN_MAP_SIZE`]. Found once, at the first call.
pub fn map_size() -> usize {
    static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        if can_reserve(MAP_SIZE) {
            return MAP_SIZE;
        }
        // Halve to a size that works, then search up toward twice it.
        let mut good = MAP_SIZE / 2;
        while good > MIN_MAP_SIZE && !can_reserve(good) {
            good /= 2;
        }
        let mut bad = good * 2;
        while bad - good > MIN_MAP_SIZE / 4 {
            let mid = good + (bad - good) / 2;
            if can_reserve(mid) {
                good = mid;
            } else {
                bad = mid;
            }
        }
        // Whole pages.
        (good / 4 * 3).max(MIN_MAP_SIZE) & !0xfff_ffff
    })
}

/// Opens (creating if needed) the environment in `path`, reserving
/// [`map_size`].
pub fn open_env(path: &Path) -> heed::Result<Env> {
    std::fs::create_dir_all(path)?;
    let size = map_size();
    let file = std::fs::metadata(path.join("data.mdb")).map_or(0, |m| m.len());
    if file > size as u64 {
        return Err(heed::Error::Io(std::io::Error::other(format!(
            "the database is {} GiB, but this machine's processes can only reserve {} GiB of \
             address space for it",
            file >> 30,
            size >> 30
        ))));
    }
    // SAFETY: LMDB files must not be modified other than through LMDB while
    // mapped. Only this process's environment touches files in the data
    // directory.
    unsafe {
        EnvOpenOptions::new()
            .map_size(size)
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
mod tests {
    use super::*;

    #[test]
    fn the_reservation_fits_the_machine() {
        let size = map_size();
        assert!((MIN_MAP_SIZE..=MAP_SIZE).contains(&size), "{size}");
        assert!(can_reserve(size));
        assert_eq!(size % 4096, 0);
    }
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
