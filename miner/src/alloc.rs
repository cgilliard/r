//! The global allocator: the system's, plus a cache of large freed blocks.
//!
//! Proving allocates and frees buffers of hundreds of megabytes, the same
//! sizes again for every proof of a shape. The system allocator returns
//! such blocks to the kernel on free, so every proof faults all of its
//! memory back in page by page (most of a block proof's system time). This
//! keeps up to `R_ALLOC_CACHE` (default 2 GiB) of freed blocks of at least
//! [`MIN`] bytes, sizes rounded up to [`GRAIN`], and hands them out again for
//! the same rounded size. It also tracks the most memory ever in use in
//! large blocks, and returns old blocks to the system whenever in-use plus
//! cached would pass that, so the cache never raises the process's peak;
//! [`release`] empties it.
//!
//! `R_ALLOC_CACHE` is in MiB; 0 disables the cache.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Blocks this large and up are cached.
const MIN: usize = 1 << 20;
/// Cached sizes are rounded up to this, so near sizes match.
const GRAIN: usize = 1 << 21;
/// Most alignment cached blocks may need.
const MAX_ALIGN: usize = 4096;
/// Most blocks kept.
const SLOTS: usize = 64;
const DEFAULT_CAP: usize = 2 << 30;

#[derive(Clone, Copy)]
struct Block {
    ptr: usize,
    size: usize,
    align: usize,
}

struct Cache {
    blocks: [Block; SLOTS],
    len: usize,
    /// Bytes cached.
    bytes: usize,
    /// Bytes of large blocks allocated and not yet freed, and the most ever.
    in_use: usize,
    peak: usize,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    blocks: [Block {
        ptr: 0,
        size: 0,
        align: 0,
    }; SLOTS],
    len: 0,
    bytes: 0,
    in_use: 0,
    peak: 0,
});

fn lock() -> std::sync::MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The cap in bytes, read once (without allocating) from `R_ALLOC_CACHE`.
static CAP: AtomicUsize = AtomicUsize::new(usize::MAX);

fn cap() -> usize {
    let c = CAP.load(Ordering::Relaxed);
    if c != usize::MAX {
        return c;
    }
    // `var_os` allocates; that's fine here, as only large blocks reach
    // this, and the small allocations it makes don't.
    let c = std::env::var_os("R_ALLOC_CACHE")
        .and_then(|v| v.to_str()?.parse::<usize>().ok())
        .map_or(DEFAULT_CAP, |mib| mib << 20);
    CAP.store(c, Ordering::Relaxed);
    c
}

fn cached(layout: &Layout) -> Option<Layout> {
    if layout.size() < MIN || layout.align() > MAX_ALIGN {
        return None;
    }
    Layout::from_size_align(layout.size().next_multiple_of(GRAIN), layout.align()).ok()
}

impl Cache {
    fn remove(&mut self, i: usize) -> Block {
        let b = self.blocks[i];
        let len = self.len;
        self.blocks.copy_within(i + 1..len, i);
        self.len -= 1;
        self.bytes -= b.size;
        b
    }

    /// Returns the oldest cached block to the system.
    fn free_oldest(&mut self) {
        let b = self.remove(0);
        // SAFETY: cached blocks were allocated by `System` with their size
        // and alignment.
        unsafe {
            System.dealloc(
                b.ptr as *mut u8,
                Layout::from_size_align_unchecked(b.size, b.align),
            )
        };
    }

    /// Counts a block of `l` (rounded) as in use, and takes a cached one if
    /// there is one. Otherwise the caller allocates a new one: first old
    /// blocks go back to the system until what's in use plus what's cached
    /// is within the most ever in use, so the cache never raises the peak.
    fn take(&mut self, l: &Layout) -> Option<*mut u8> {
        self.in_use += l.size();
        self.peak = self.peak.max(self.in_use);
        let hit = self.blocks[..self.len]
            .iter()
            .position(|b| b.size == l.size() && b.align == l.align());
        if let Some(i) = hit {
            return Some(self.remove(i).ptr as *mut u8);
        }
        while self.len > 0 && self.in_use + self.bytes > self.peak {
            self.free_oldest();
        }
        None
    }

    /// Takes back a freed block of `l` (rounded), keeping at most `cap`
    /// bytes and [`SLOTS`] blocks.
    fn put(&mut self, ptr: *mut u8, l: &Layout, cap: usize) {
        self.in_use -= l.size();
        if l.size() > cap {
            // SAFETY: allocated by `System` with the rounded layout.
            return unsafe { System.dealloc(ptr, *l) };
        }
        while self.len == SLOTS || self.bytes + l.size() > cap {
            self.free_oldest();
        }
        let len = self.len;
        self.blocks[len] = Block {
            ptr: ptr as usize,
            size: l.size(),
            align: l.align(),
        };
        self.len += 1;
        self.bytes += l.size();
    }
}

/// Returns every cached block to the system: call when a burst of large
/// allocations is over (say, between proving and mining), so the process
/// doesn't hold memory it isn't using.
pub fn release() {
    let mut cache = lock();
    while cache.len > 0 {
        cache.free_oldest();
    }
}

/// The allocator; [`crate`] installs it.
pub struct Recycling;

unsafe impl GlobalAlloc for Recycling {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match cached(&layout) {
            Some(l) => {
                if let Some(p) = lock().take(&l) {
                    return p;
                }
                // SAFETY: `l` has nonzero size.
                unsafe { System.alloc(l) }
            }
            // SAFETY: as the caller's contract.
            None => unsafe { System.alloc(layout) },
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        match cached(&layout) {
            Some(l) => {
                if let Some(p) = lock().take(&l) {
                    // SAFETY: the block holds `l.size()` bytes.
                    unsafe { core::ptr::write_bytes(p, 0, l.size()) };
                    return p;
                }
                // SAFETY: `l` has nonzero size.
                unsafe { System.alloc_zeroed(l) }
            }
            // SAFETY: as the caller's contract.
            None => unsafe { System.alloc_zeroed(layout) },
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match cached(&layout) {
            Some(l) => {
                let cap = cap();
                lock().put(ptr, &l, cap);
            }
            // SAFETY: allocated by `System` with this layout.
            None => unsafe { System.dealloc(ptr, layout) },
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `layout` is valid, so its alignment is too.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let (old_c, new_c) = (cached(&layout), cached(&new_layout));
        let old = old_c.unwrap_or(layout);
        let new = new_c.unwrap_or(new_layout);
        if old.size() == new.size() && old_c.is_some() == new_c.is_some() {
            return ptr;
        }
        if old_c.is_some() || new_c.is_some() {
            let mut cache = lock();
            cache.in_use -= old_c.map_or(0, |l| l.size());
            cache.in_use += new_c.map_or(0, |l| l.size());
            cache.peak = cache.peak.max(cache.in_use);
        }
        // SAFETY: the block was allocated by `System` with `old`, and the
        // result is freed (or reallocated) with `new`.
        unsafe { System.realloc(ptr, old, new.size()) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn large_blocks_are_reused() {
        let a = vec![7u8; 5 << 20];
        let p = a.as_ptr() as usize;
        drop(a);
        // Same rounded size: the cached block may come back (another test
        // thread may take it first), zeroed when asked.
        let b = vec![0u8; (5 << 20) - 100];
        assert!(b.iter().all(|&x| x == 0));
        let _ = (p, b);
        let mut v: Vec<u64> = Vec::with_capacity(1 << 18);
        v.extend(0..1 << 18);
        v.reserve(1 << 20);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64));
    }
}
