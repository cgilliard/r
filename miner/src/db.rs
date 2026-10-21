//! The LMDB environment holding all on-disk state.

use std::path::Path;

use heed::{Env, EnvOpenOptions};

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

        /// Closes and reopens the environment, as a restart would.
        pub fn reopen(&mut self) {
            self.env.take().unwrap().prepare_for_closing().wait();
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
}
