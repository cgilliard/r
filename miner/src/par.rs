//! Data parallelism on scoped threads, one per available core.

use std::sync::OnceLock;
use std::thread;

/// Threads to split work across: `R_THREADS` if set, else one per core.
pub fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("R_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .filter(|&t| t > 0)
            .unwrap_or_else(|| thread::available_parallelism().map_or(1, |t| t.get()))
    })
}

/// `(0..n).map(f).collect()`, split into contiguous ranges across threads,
/// each at least `min_per_thread` long (smaller jobs aren't worth a thread).
pub fn par_map<T: Send>(n: usize, min_per_thread: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let chunk = n.div_ceil(threads()).max(min_per_thread.max(1));
    if chunk >= n {
        return (0..n).map(f).collect();
    }
    thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|start| {
                let f = &f;
                s.spawn(move || (start..(start + chunk).min(n)).map(f).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker thread panicked"))
            .collect()
    })
}

/// Calls `f(i, chunk)` for each `chunk_len`-long chunk of `data` (the last
/// may be shorter), with contiguous runs of chunks on each thread.
pub fn par_chunks_mut<T: Send>(
    data: &mut [T],
    chunk_len: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let chunks = data.len().div_ceil(chunk_len);
    let per_thread = chunks.div_ceil(threads()).max(1);
    if per_thread >= chunks {
        data.chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
        return;
    }
    thread::scope(|s| {
        for (t, run) in data.chunks_mut(per_thread * chunk_len).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (i, c) in run.chunks_mut(chunk_len).enumerate() {
                    f(t * per_thread + i, c);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_preserves_order() {
        for n in [0, 1, 255, 256, 257, 10_000] {
            for min in [1, 256] {
                assert_eq!(
                    par_map(n, min, |i| i * 3),
                    (0..n).map(|i| i * 3).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn par_chunks_mut_visits_every_chunk_once() {
        for (len, chunk) in [(0, 3), (1, 3), (100, 7), (10_000, 8), (10_001, 64)] {
            let mut v = vec![0usize; len];
            par_chunks_mut(&mut v, chunk, |i, c| {
                for (k, x) in c.iter_mut().enumerate() {
                    *x += i * chunk + k + 1;
                }
            });
            assert_eq!(v, (1..=len).collect::<Vec<_>>());
        }
    }
}
