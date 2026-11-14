//! A cache of large `f32` buffers.
//!
//! Fresh allocations are slow on first touch (the OS maps and zeroes every page, often taking longer than the
//! kernel that fills them). On Windows that already holds at tens of kilobytes: the 230 KB padded input copy of
//! a 32-channel 40x40 conv took 80 µs fresh and 13 µs reused, and a SiLU over a 64-channel 20x20 activation
//! (100 KB) 85-110 µs against 47 µs, about as long as the conv that made it. Training and inference loops
//! allocate the same sizes every step, so storages of at least [`MIN_POOLED`] floats are returned here when
//! their last tensor drops and handed back out for the next request of exactly the same length.
//!
//! When the cache is full, the lengths nobody has asked for while [`STALE_LIMITS`] times the limit went through
//! the cache are evicted to make room. Without that, a full cache of one workload's lengths (another model's
//! batch, another input shape) never admits the next workload's, whose every buffer is then a fresh allocation
//! for as long as the process lives: a folo-s 1280×720 frame on the CPU ran 36–38 ms instead of 20 after an
//! untrained folo-s model's `predict_batch` of 16 frames filled the cache (docs/notes/hill/r3-cpufix.md).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

/// Buffers shorter than this (in floats, 16 KiB) bypass the pool.
pub(crate) const MIN_POOLED: usize = 1 << 12;

/// A length is stale, and evicted when the cache is full, once this many times the cache limit (in bytes) has
/// been returned to the cache since it was last taken or returned. A workload that cycles through the same
/// lengths touches each of them far more often than that, so only lengths that went out of use are evicted.
const STALE_LIMITS: u64 = 2;

static LIMIT: AtomicUsize = AtomicUsize::new(1 << 30);

/// The buffers of one length, and when (in [`Pool::traffic`]) the length was last taken or returned.
struct Class {
    bufs: Vec<Vec<f32>>,
    used: u64,
}

struct Pool {
    free: HashMap<usize, Class>,
    bytes: usize,
    /// Bytes returned to the cache so far: the clock that ages the lengths.
    traffic: u64,
    /// No length is stale before `traffic` reaches this (the oldest length's `used` plus the stale age at the
    /// last scan, a lower bound since `used` only grows), so a full cache with nothing stale scans once per
    /// stretch of traffic, not on every return.
    next_stale: u64,
    /// Scans for stale lengths so far (tests observe the skip with it).
    #[cfg(test)]
    scans: usize,
}

impl Pool {
    fn new() -> Self {
        Pool {
            free: HashMap::new(),
            bytes: 0,
            traffic: 0,
            next_stale: 0,
            #[cfg(test)]
            scans: 0,
        }
    }

    fn take(&mut self, n: usize) -> Option<Vec<f32>> {
        let c = self.free.get_mut(&n)?;
        let v = c.bufs.pop()?;
        c.used = self.traffic;
        if c.bufs.is_empty() {
            self.free.remove(&n);
        }
        self.bytes -= n * 4;
        Some(v)
    }

    /// Keep `v` if it fits under `limit` bytes, after evicting stale lengths if needed. Returns the buffers that
    /// leave the cache (evicted, or `v` itself when it does not fit), for the caller to free outside the lock.
    fn put(&mut self, v: Vec<f32>, limit: usize) -> Vec<Vec<f32>> {
        let n = v.len();
        self.traffic += n as u64 * 4;
        let mut out = Vec::new();
        if self.bytes + n * 4 > limit {
            self.evict_stale(limit, &mut out);
        }
        if self.bytes + n * 4 > limit {
            // Still a use: a length the workload keeps returning is not stale.
            if let Some(c) = self.free.get_mut(&n) {
                c.used = self.traffic;
            }
            out.push(v);
            return out;
        }
        self.bytes += n * 4;
        let c = self.free.entry(n).or_insert_with(|| Class { bufs: Vec::new(), used: 0 });
        c.used = self.traffic;
        c.bufs.push(v);
        out
    }

    /// Evict every length unused for [`STALE_LIMITS`] times `limit` bytes of traffic.
    fn evict_stale(&mut self, limit: usize, out: &mut Vec<Vec<f32>>) {
        if self.traffic < self.next_stale {
            return;
        }
        #[cfg(test)]
        {
            self.scans += 1;
        }
        let age = STALE_LIMITS * limit as u64;
        // A length cached later starts at the current traffic, so `traffic` bounds an empty cache too.
        let (traffic, mut oldest) = (self.traffic, self.traffic);
        let mut freed = 0;
        self.free.retain(|&n, c| {
            if c.used.saturating_add(age) <= traffic {
                freed += n * 4 * c.bufs.len();
                out.append(&mut c.bufs);
                false
            } else {
                oldest = oldest.min(c.used);
                true
            }
        });
        self.bytes -= freed;
        self.next_stale = oldest.saturating_add(age);
    }
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
    g.as_mut()?.take(n)
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

/// Return a buffer to the cache (dropped if small, or if the cache is full and nothing in it is stale).
pub(crate) fn recycle(v: Vec<f32>) {
    let n = v.len();
    if n < MIN_POOLED || v.capacity() != n {
        return;
    }
    let limit = LIMIT.load(Ordering::Relaxed);
    let gone = {
        let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
        g.get_or_insert_with(Pool::new).put(v, limit)
    };
    // Freed outside the lock: returning large buffers to the OS takes a while.
    drop(gone);
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = MIN_POOLED;

    fn put(p: &mut Pool, n: usize, limit: usize) -> usize {
        p.put(vec![0.0; n], limit).len()
    }

    /// `calls` calls that each take `k` buffers of length `n` and return them: the misses of each call.
    fn workload(p: &mut Pool, n: usize, k: usize, calls: usize, limit: usize) -> Vec<usize> {
        (0..calls)
            .map(|_| {
                let bufs: Vec<Option<Vec<f32>>> = (0..k).map(|_| p.take(n)).collect();
                let misses = bufs.iter().filter(|b| b.is_none()).count();
                for b in bufs {
                    p.put(b.unwrap_or_else(|| vec![0.0; n]), limit);
                }
                misses
            })
            .collect()
    }

    /// A cache filled with one workload's lengths admits the next workload's once the old ones went unused for
    /// `STALE_LIMITS` limits of traffic, and keeps them from then on: the next workload's calls stop allocating.
    /// Three workloads in turn, each filling the cache: the second evicts every length the cache held.
    #[test]
    fn a_full_cache_admits_each_new_working_set() {
        let limit = 8 * N * 4;
        let mut p = Pool::new();
        assert_eq!(workload(&mut p, N, 8, 1, limit), [8]);
        assert_eq!(p.bytes, limit);
        for (n, k) in [(2 * N, 4), (4 * N, 2)] {
            let misses = workload(&mut p, n, k, 40, limit);
            assert_eq!(misses[0], k, "admitted at once into a full cache of recent lengths: {misses:?}");
            // Two limits of traffic: two calls of refused returns, then the old lengths are evicted.
            assert!(misses[..4].iter().sum::<usize>() < 4 * k, "never admitted: {misses:?}");
            assert!(misses[4..].iter().all(|&m| m == 0), "{misses:?}");
            assert_eq!(p.bytes, limit);
            assert_eq!(p.free.keys().collect::<Vec<_>>(), [&n], "a stale length is still cached");
        }
    }

    /// A workload whose lengths do not all fit keeps the ones it has: nothing it keeps using is evicted.
    #[test]
    fn a_working_set_larger_than_the_cache_keeps_its_cached_part() {
        let limit = (4 * N + 6) * 4;
        let mut p = Pool::new();
        let lens = [N, N + 1, N + 2, N + 3, N + 4, N + 5];
        let (mut hits, mut calls) = (0, 0);
        for round in 0..200 {
            for &n in &lens {
                let v = p.take(n);
                if round >= 100 {
                    calls += 1;
                    hits += v.is_some() as usize;
                }
                p.put(v.unwrap_or_else(|| vec![0.0; n]), limit);
            }
        }
        assert_eq!(p.bytes, limit, "the first four lengths stay cached");
        assert_eq!(hits, calls * 4 / 6);
    }

    /// The scan for stale lengths is skipped until the oldest kept length can be stale.
    #[test]
    fn a_full_cache_with_nothing_stale_does_not_rescan() {
        let limit = 2 * N * 4;
        let mut p = Pool::new();
        put(&mut p, N, limit);
        put(&mut p, N, limit);
        assert_eq!(put(&mut p, N + 1, limit), 1);
        assert_eq!(
            (p.scans, p.next_stale),
            (1, 2 * limit as u64 + 2 * N as u64 * 4),
            "N last returned at 8N bytes"
        );
        assert_eq!(put(&mut p, N + 1, limit), 1);
        assert_eq!(p.scans, 1, "rescanned before anything could be stale");
        // The return that brings the traffic to `next_stale` scans again, evicts both N buffers and is kept.
        while p.traffic + (N as u64 + 1) * 4 < p.next_stale {
            assert_eq!(put(&mut p, N + 1, limit), 1);
        }
        assert_eq!(p.scans, 1);
        assert_eq!(put(&mut p, N + 1, limit), 2, "the two N buffers leave");
        assert_eq!((p.scans, p.bytes), (2, (N + 1) * 4));
    }
}
