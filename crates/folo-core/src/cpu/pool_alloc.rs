//! A cache of large `f32` buffers.
//!
//! Fresh allocations are slow on first touch (the OS maps and zeroes every page, often taking longer than the
//! kernel that fills them). On Windows that already holds at tens of kilobytes: the 230 KB padded input copy of
//! a 32-channel 40x40 conv took 80 µs fresh and 13 µs reused, and a SiLU over a 64-channel 20x20 activation
//! (100 KB) 85-110 µs against 47 µs, about as long as the conv that made it. Training and inference loops
//! allocate the same sizes every step, so storages of at least [`MIN_POOLED`] floats are returned here when
//! their last tensor drops and handed back out for the next request of exactly the same length.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

/// Buffers shorter than this (in floats, 16 KiB) bypass the pool.
pub(crate) const MIN_POOLED: usize = 1 << 12;

static LIMIT: AtomicUsize = AtomicUsize::new(1 << 30);

struct Pool {
    free: HashMap<usize, Vec<Vec<f32>>>,
    bytes: usize,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

static POISON: AtomicBool = AtomicBool::new(false);

/// A testing aid: while on (process-wide), every buffer the CPU kernels take without initialising it is filled
/// with NaN, so an output element a kernel never wrote, or an input it reads that nobody wrote (results that
/// would otherwise depend on whatever a recycled buffer held), shows up as NaN.
pub fn poison_uninit(on: bool) {
    POISON.store(on, Ordering::Relaxed);
}

/// Cap the bytes held by the CPU buffer cache (default 1 GiB; 0 disables caching). Shrinks immediately.
pub fn set_buffer_cache_limit(bytes: usize) {
    LIMIT.store(bytes, Ordering::Relaxed);
    let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = g.as_mut()
        && p.bytes > bytes
    {
        p.free.clear();
        p.bytes = 0;
    }
}

/// Release every cached buffer.
pub fn clear_buffer_cache() {
    let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
    *g = None;
}

/// Bytes currently held by the cache.
pub fn buffer_cache_bytes() -> usize {
    POOL.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |p| p.bytes)
}

fn take(n: usize) -> Option<Vec<f32>> {
    if n < MIN_POOLED {
        return None;
    }
    let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let p = g.as_mut()?;
    let v = p.free.get_mut(&n)?.pop()?;
    p.bytes -= n * 4;
    Some(v)
}

/// A buffer of `n` floats with unspecified contents (for kernels that overwrite every element).
pub(crate) fn alloc_uninit(n: usize) -> Vec<f32> {
    if POISON.load(Ordering::Relaxed) {
        return alloc_filled(n, f32::NAN);
    }
    take(n).unwrap_or_else(|| vec![0.0; n])
}

/// A zeroed buffer of `n` floats.
pub(crate) fn alloc_zeroed(n: usize) -> Vec<f32> {
    alloc_filled(n, 0.0)
}

/// A buffer of `n` copies of `v`.
pub(crate) fn alloc_filled(n: usize, v: f32) -> Vec<f32> {
    match take(n) {
        Some(mut b) => {
            b.par_chunks_mut(1 << 16).for_each(|c| c.fill(v));
            b
        }
        None => vec![v; n],
    }
}

/// Return a buffer to the cache (dropped if small or the cache is full).
pub(crate) fn recycle(v: Vec<f32>) {
    let n = v.len();
    if n < MIN_POOLED || v.capacity() != n {
        return;
    }
    let limit = LIMIT.load(Ordering::Relaxed);
    let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let p = g.get_or_insert_with(|| Pool { free: HashMap::new(), bytes: 0 });
    if p.bytes + n * 4 > limit {
        return;
    }
    p.bytes += n * 4;
    p.free.entry(n).or_default().push(v);
}
