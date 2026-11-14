//! A cache of large `f32` buffers.
//!
//! Fresh allocations are slow on first touch (the OS maps and zeroes every page, often taking longer than the
//! kernel that fills them). On Windows that already holds at tens of kilobytes: the 230 KB padded input copy of
//! a 32-channel 40x40 conv took 80 µs fresh and 13 µs reused, and a SiLU over a 64-channel 20x20 activation
//! (100 KB) 85-110 µs against 47 µs, about as long as the conv that made it. Training and inference loops
//! allocate the same sizes every step, so storages of at least [`MIN_POOLED`] floats are returned here when
//! their last tensor drops and handed back out for the next request of exactly the same length.
//!
//! When the cache is full, lengths that went out of use are evicted to make room: a length is stale once it has
//! gone unused for [`STALE_LIMITS`] times the limit of returned bytes, or twice the longest it ever went unused
//! before being asked for again, whichever is longer. Without that, a full cache of one workload's lengths
//! (another model's batch, another input shape) never admits the next workload's, whose every buffer is then a
//! fresh allocation for as long as the process lives: a folo-s 1280×720 frame on the CPU ran 36–38 ms instead of
//! 20 after an untrained folo-s model's `predict_batch` of 16 frames filled the cache
//! (docs/notes/hill/r3-cpufix.md). The second bound is for workloads that cycle through more than the cache
//! holds (a training step, two models taking turns): a fixed age would evict each cached length just before it
//! comes back round and leave no hits at all, so the cache remembers, for a while after a length leaves it, how
//! long it went unused. A workload whose cycle is longer than [`FORGET_LIMITS`] limits of traffic can still churn.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::prelude::*;

/// Buffers shorter than this (in floats, 16 KiB) bypass the pool.
pub(crate) const MIN_POOLED: usize = 1 << 12;

/// A length is stale, and evicted when the cache is full, at the earliest once this many times the cache limit
/// (in bytes) has been returned to the cache since it was last taken or returned.
const STALE_LIMITS: u64 = 2;

/// A length with no cached buffer is forgotten (its record dropped) after going unused for this many limits of
/// traffic, or for as long as it would take to go stale, whichever is longer.
const FORGET_LIMITS: u64 = 16;

static LIMIT: AtomicUsize = AtomicUsize::new(1 << 30);

/// The buffers of one length (possibly none: the record outlives them), when (in [`Pool::traffic`]) the length was
/// last taken or returned, and the longest it went unused between two of those.
struct Class {
    bufs: Vec<Vec<f32>>,
    used: u64,
    idle: u64,
}

impl Class {
    fn touch(&mut self, traffic: u64) {
        self.idle = self.idle.max(traffic - self.used);
        self.used = traffic;
    }

    /// The traffic at which the length is stale.
    fn stale_at(&self, age: u64) -> u64 {
        self.used.saturating_add(age.max(self.idle.saturating_mul(2)))
    }
}

struct Pool {
    free: HashMap<usize, Class>,
    bytes: usize,
    /// Bytes returned to the cache so far: the clock that ages the lengths.
    traffic: u64,
    /// No length is stale before `traffic` reaches this (the soonest a kept length could be stale at the last
    /// scan, a lower bound since `used` and `idle` only grow), so a full cache with nothing stale scans once per
    /// stretch of traffic, not on every return.
    next_stale: u64,
    /// When to drop the records of lengths with no buffer that went unused long enough to forget.
    next_forget: u64,
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
            next_forget: 0,
            #[cfg(test)]
            scans: 0,
        }
    }

    fn take(&mut self, n: usize) -> Option<Vec<f32>> {
        let c = self.free.get_mut(&n)?;
        c.touch(self.traffic);
        let v = c.bufs.pop()?;
        self.bytes -= n * 4;
        Some(v)
    }

    /// Keep `v` if it fits under `limit` bytes, after evicting stale lengths if needed. Returns the buffers that
    /// leave the cache (evicted, or `v` itself when it does not fit), for the caller to free outside the lock.
    fn put(&mut self, v: Vec<f32>, limit: usize) -> Vec<Vec<f32>> {
        let n = v.len();
        if n * 4 > limit {
            // Never fits: not traffic, or a length that never fits would age the ones the workload reuses.
            return vec![v];
        }
        self.traffic += n as u64 * 4;
        let traffic = self.traffic;
        // A refused return is a use too: a length the workload keeps returning is not stale.
        self.free
            .entry(n)
            .or_insert_with(|| Class { bufs: Vec::new(), used: traffic, idle: 0 })
            .touch(traffic);
        if traffic >= self.next_forget {
            self.forget(limit);
        }
        let mut out = Vec::new();
        if self.bytes + n * 4 > limit {
            self.evict_stale(limit, &mut out);
        }
        if self.bytes + n * 4 > limit {
            out.push(v);
            return out;
        }
        self.bytes += n * 4;
        self.free.get_mut(&n).expect("touched above").bufs.push(v);
        out
    }

    /// Evict the buffers of every stale length.
    fn evict_stale(&mut self, limit: usize, out: &mut Vec<Vec<f32>>) {
        if self.traffic < self.next_stale {
            return;
        }
        #[cfg(test)]
        {
            self.scans += 1;
        }
        let age = STALE_LIMITS * limit as u64;
        // A length cached later starts at the current traffic, so this bounds an empty cache too.
        let (traffic, mut soonest) = (self.traffic, self.traffic.saturating_add(age));
        let mut freed = 0;
        for (&n, c) in self.free.iter_mut().filter(|(_, c)| !c.bufs.is_empty()) {
            let at = c.stale_at(age);
            if at <= traffic {
                freed += n * 4 * c.bufs.len();
                out.append(&mut c.bufs);
            } else {
                soonest = soonest.min(at);
            }
        }
        self.bytes -= freed;
        self.next_stale = soonest;
    }

    /// Drop the records of lengths with no buffer unused for [`FORGET_LIMITS`] limits (and stale), once per limit
    /// of traffic, so lengths that come and go (one input shape, then another) do not pile up.
    fn forget(&mut self, limit: usize) {
        let traffic = self.traffic;
        let age = STALE_LIMITS * limit as u64;
        let forget = FORGET_LIMITS * limit as u64;
        self.free.retain(|_, c| {
            !c.bufs.is_empty() || c.used.saturating_add(forget) > traffic || c.stale_at(age) > traffic
        });
        self.next_forget = traffic + limit as u64;
    }

    /// The limit changed to `bytes`: drop everything if it no longer fits, and rescan on the next full return,
    /// since `next_stale` was set from the old limit.
    fn set_limit(&mut self, bytes: usize) {
        if self.bytes > bytes {
            self.free.clear();
            self.bytes = 0;
        }
        self.next_stale = 0;
        self.next_forget = 0;
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
    if let Some(p) = g.as_mut() {
        p.set_limit(bytes);
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

    fn cached(p: &Pool) -> Vec<usize> {
        let mut v: Vec<usize> = p.free.iter().filter(|(_, c)| !c.bufs.is_empty()).map(|(&n, _)| n).collect();
        v.sort();
        v
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
            assert_eq!(cached(&p), [n], "a stale length is still cached");
        }
    }

    /// A workload cycling through more lengths than the cache holds keeps a cache-sized part of them, also when a
    /// cycle is far longer than the stale age (1.5 to 12 limits of traffic): a length that comes back after going
    /// unused for a cycle is not evicted again for going unused that long. No buffer is evicted once settled.
    #[test]
    fn a_working_set_larger_than_the_cache_keeps_its_cached_part() {
        for m in [6, 12, 24, 48] {
            // Any 4 of the lengths fit, never 5.
            let limit = 4 * (N + m) * 4;
            let mut p = Pool::new();
            let (mut hits, mut gone) = (0, 0);
            for round in 0..40 {
                for n in N..N + m {
                    let hit = p.take(n).is_some();
                    let out = put(&mut p, n, limit);
                    if round >= 30 {
                        hits += hit as usize;
                        gone += out;
                    }
                }
            }
            // Only the missed lengths' returns leave the cache: nothing is evicted.
            assert_eq!((hits, gone), (10 * 4, 10 * (m - 4)), "{m} lengths: the last 10 cycles");
            assert_eq!(cached(&p).len(), 4);
        }
    }

    /// Returns that can never fit do not age the cache: they would evict the lengths the workload keeps reusing.
    #[test]
    fn a_return_larger_than_the_limit_ages_nothing() {
        let limit = 4 * N * 4;
        let mut p = Pool::new();
        let mut hits = 0;
        for _ in 0..100 {
            let small = p.take(N);
            hits += small.is_some() as usize;
            put(&mut p, N, limit);
            assert!(p.take(9 * N).is_none());
            assert_eq!(put(&mut p, 9 * N, limit), 1);
        }
        assert_eq!((hits, p.traffic), (99, 100 * N as u64 * 4));
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

    /// A smaller limit ages the lengths by the new limit at once, not after a scan deadline set from the old one.
    #[test]
    fn a_smaller_limit_takes_effect_on_the_stale_age() {
        let mut p = Pool::new();
        let big = 8 * N * 4;
        for _ in 0..8 {
            put(&mut p, N, big);
        }
        assert_eq!(put(&mut p, N + 1, big), 1);
        let small = 2 * N * 4;
        p.set_limit(small);
        put(&mut p, N, small);
        put(&mut p, N, small);
        let last = p.traffic;
        while put(&mut p, 2 * N, small) != 2 {
            assert!(p.traffic < last + 2 * small as u64 + small as u64, "still refused at {}", p.traffic);
        }
        assert_eq!(cached(&p), [2 * N]);
    }

    /// Records of lengths that come and go (each new input shape) are dropped once forgotten.
    #[test]
    fn lengths_used_once_are_forgotten() {
        let limit = 4 * N * 4;
        let mut p = Pool::new();
        for i in 0..2000 {
            put(&mut p, N + i, limit);
        }
        // `FORGET_LIMITS` limits of traffic at 4 returns per limit, plus a sweep's slack.
        assert!(p.free.len() <= FORGET_LIMITS as usize * 4 + 8, "{} records", p.free.len());
    }
}
