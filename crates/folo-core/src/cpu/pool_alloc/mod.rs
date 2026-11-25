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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use rayon::prelude::*;

/// Buffers shorter than this (in floats, 16 KiB) bypass the pool.
pub(crate) const MIN_POOLED: usize = 1 << 12;

/// Maximum buffers kept in one thread's local magazine before spilling oldest entries to the shared pool.
const TLS_MAX_BUFS: usize = 8;

/// Largest single buffer (in bytes, 4 MiB) kept in a thread-local magazine; larger buffers go straight to the
/// shared pool so huge allocations do not stay pinned to one worker thread.
const TLS_MAX_BUF_BYTES: usize = 4 << 20;

/// Maximum bytes held by one thread's local magazine (16 MiB).
const TLS_MAX_BYTES: usize = 16 << 20;

/// A length is stale, and evicted when the cache is full, at the earliest once this many times the cache limit
/// (in bytes) has been returned to the cache since it was last taken or returned.
const STALE_LIMITS: u64 = 2;

/// A length with no cached buffer is forgotten (its record dropped) after going unused for this many limits of
/// traffic, or for as long as it would take to go stale, whichever is longer.
const FORGET_LIMITS: u64 = 16;

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
    #[cfg(test)]
    fn put(&mut self, v: Vec<f32>, limit: usize) -> Vec<Vec<f32>> {
        self.put_with_budget(v, limit, None)
    }

    /// Like [`Pool::put`], optionally reserving admitted bytes against a shared `total_bytes` atomic budget
    /// shared with the per-thread magazines so concurrent returns across both levels can never exceed `limit`.
    fn put_with_budget(
        &mut self,
        v: Vec<f32>,
        limit: usize,
        total_bytes: Option<&AtomicUsize>,
    ) -> Vec<Vec<f32>> {
        let n = v.len();
        let bytes = n * 4;
        if bytes > limit {
            // Never fits: not traffic, or a length that never fits would age the ones the workload reuses.
            return vec![v];
        }
        self.traffic += bytes as u64;
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
        let reserved = match total_bytes {
            Some(tb) => {
                if try_reserve_atomic(tb, bytes, limit) {
                    true
                } else {
                    let before = self.bytes;
                    self.evict_stale(limit, &mut out);
                    let evicted = before - self.bytes;
                    if evicted > 0 {
                        tb.fetch_sub(evicted, Ordering::AcqRel);
                    }
                    try_reserve_atomic(tb, bytes, limit)
                }
            }
            None => {
                if self.bytes + bytes > limit {
                    self.evict_stale(limit, &mut out);
                }
                self.bytes + bytes <= limit
            }
        };
        if !reserved {
            out.push(v);
            return out;
        }
        self.bytes += bytes;
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
        let age = limits(STALE_LIMITS, limit);
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
        let age = limits(STALE_LIMITS, limit);
        let forget = limits(FORGET_LIMITS, limit);
        self.free.retain(|_, c| {
            !c.bufs.is_empty() || c.used.saturating_add(forget) > traffic || c.stale_at(age) > traffic
        });
        self.next_forget = traffic.saturating_add(limit as u64);
    }

    /// The limit changed to `bytes`: drop everything if it no longer fits, and rescan on the next full return,
    /// since `next_stale` was set from the old limit.
    #[cfg(test)]
    fn set_limit(&mut self, bytes: usize) {
        drop(self.set_limit_evict(bytes, self.bytes));
    }

    /// Like [`Pool::set_limit`], checking both this pool's bytes and the combined cache `total_bytes` against
    /// `bytes`, and returning any dropped map entries so the caller can free them outside the lock.
    fn set_limit_evict(&mut self, bytes: usize, total_bytes: usize) -> HashMap<usize, Class> {
        let evicted = if self.bytes > bytes || total_bytes > bytes {
            self.bytes = 0;
            std::mem::take(&mut self.free)
        } else {
            HashMap::new()
        };
        self.next_stale = 0;
        self.next_forget = 0;
        evicted
    }
}

/// `k` limits of traffic, saturating: a limit of `usize::MAX` means no limit.
fn limits(k: u64, limit: usize) -> u64 {
    k.saturating_mul(limit as u64)
}

/// One thread's L1 magazine of recently recycled buffers (oldest at front, newest at back).
///
/// Stored behind an `Arc<Mutex<Slot>>` owned by the thread's `thread_local!` and registered weakly in a
/// [`Cache`]'s slot list: the fast path locks only the calling thread's uncontended slot, while
/// [`clear_buffer_cache`] and [`set_buffer_cache_limit`] can still drain every live thread's magazine immediately.
struct Slot {
    bufs: Vec<Vec<f32>>,
    bytes: usize,
    /// Clear-epoch this slot last synchronized with (`Cache::epoch`).
    epoch: u64,
    /// Limit under which buffers currently in `bufs` were admitted (so `set_limit` drains on any reduction).
    last_limit: usize,
    tls_bytes: &'static AtomicUsize,
    total_bytes: &'static AtomicUsize,
}

impl Slot {
    fn new(tls_bytes: &'static AtomicUsize, total_bytes: &'static AtomicUsize) -> Self {
        Slot { bufs: Vec::new(), bytes: 0, epoch: 0, last_limit: 0, tls_bytes, total_bytes }
    }

    fn release_accounting(&self, bytes: usize) {
        if bytes > 0 {
            self.tls_bytes.fetch_sub(bytes, Ordering::AcqRel);
            self.total_bytes.fetch_sub(bytes, Ordering::AcqRel);
        }
    }

    fn sync_epoch(&mut self, cur_epoch: u64, out: &mut Vec<Vec<f32>>) {
        if self.epoch < cur_epoch {
            self.drain_into(out);
            self.epoch = cur_epoch;
        }
    }

    fn take(&mut self, n: usize) -> Option<Vec<f32>> {
        let idx = self.bufs.iter().rposition(|b| b.len() == n)?;
        let v = self.bufs.remove(idx);
        let bytes = n * 4;
        self.bytes -= bytes;
        self.release_accounting(bytes);
        Some(v)
    }

    fn drain_into(&mut self, out: &mut Vec<Vec<f32>>) {
        let freed = self.bytes;
        self.bytes = 0;
        self.release_accounting(freed);
        out.append(&mut self.bufs);
    }

    /// Try to keep `v` in this slot without exceeding per-thread, global TLS, or total cache caps; returns any
    /// spilled oldest buffers (plus `v` itself if it could not be kept in TLS) for the caller to hand to the
    /// shared pool outside this lock.
    fn put(&mut self, v: Vec<f32>, limit: usize, spill: &mut Vec<Vec<f32>>) -> Option<Vec<f32>> {
        let bytes = v.len() * 4;
        let tls_global_cap = limit / 2;
        let per_thread_cap = TLS_MAX_BYTES.min(tls_global_cap);
        if bytes > TLS_MAX_BUF_BYTES || bytes > per_thread_cap {
            return Some(v);
        }
        // Evict oldest buffers first so the active working set stays in L1 and stale entries drain to L2 `Pool`.
        let mut freed = 0usize;
        while self.bufs.len() >= TLS_MAX_BUFS || self.bytes + bytes > per_thread_cap {
            let Some(old) = (!self.bufs.is_empty()).then(|| self.bufs.remove(0)) else {
                break;
            };
            let old_bytes = old.len() * 4;
            self.bytes -= old_bytes;
            freed += old_bytes;
            spill.push(old);
        }
        self.release_accounting(freed);
        // Reserve from the TLS sub-budget first, then the shared total budget, so `total_bytes` never spikes on
        // a rejected TLS admission and concurrent TLS + Pool returns can never exceed `limit`.
        if !try_reserve_atomic(self.tls_bytes, bytes, tls_global_cap) {
            return Some(v);
        }
        if !try_reserve_atomic(self.total_bytes, bytes, limit) {
            self.tls_bytes.fetch_sub(bytes, Ordering::AcqRel);
            return Some(v);
        }
        self.bytes += bytes;
        self.last_limit = self.last_limit.max(limit);
        self.bufs.push(v);
        None
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let freed = self.bytes;
        self.bytes = 0;
        self.release_accounting(freed);
    }
}

fn try_reserve_atomic(counter: &AtomicUsize, add: usize, cap: usize) -> bool {
    let mut cur = counter.load(Ordering::Relaxed);
    loop {
        let Some(next) = cur.checked_add(add) else {
            return false;
        };
        if next > cap {
            return false;
        }
        match counter.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
}

struct PoolState {
    pool: Option<Pool>,
    epoch: u64,
}

impl PoolState {
    const fn new() -> Self {
        PoolState { pool: None, epoch: 0 }
    }

    fn sync_epoch(&mut self, cur_epoch: u64, total_bytes: &AtomicUsize) -> Option<Pool> {
        if self.epoch < cur_epoch {
            self.epoch = cur_epoch;
            if let Some(p) = self.pool.take() {
                if p.bytes > 0 {
                    total_bytes.fetch_sub(p.bytes, Ordering::AcqRel);
                }
                return Some(p);
            }
        }
        None
    }
}

/// Two-level buffer cache: per-thread [`Slot`] magazines in front of a shared [`Pool`].
///
/// Both levels reserve bytes against the same `total_bytes` atomic budget and honor the same `epoch` clear epoch,
/// so concurrent returns can never exceed `limit` and `clear()` / `set_limit(0)` invalidates both levels at one
/// atomic point. `set_limit_lock` serializes concurrent limit changes without touching any hot path; inside
/// `set_limit`, `slots`, each `Slot`, and `pool` are always acquired and released separately (and freed buffers
/// always drop outside all locks), so deadlock is impossible.
struct Cache {
    limit: AtomicUsize,
    epoch: AtomicU64,
    set_limit_lock: Mutex<()>,
    pool: Mutex<PoolState>,
    slots: Mutex<Vec<Weak<Mutex<Slot>>>>,
    tls_bytes: AtomicUsize,
    total_bytes: AtomicUsize,
}

impl Cache {
    const fn new(limit: usize) -> Self {
        Cache {
            limit: AtomicUsize::new(limit),
            epoch: AtomicU64::new(1),
            set_limit_lock: Mutex::new(()),
            pool: Mutex::new(PoolState::new()),
            slots: Mutex::new(Vec::new()),
            tls_bytes: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
        }
    }

    fn register_slot(&'static self) -> Arc<Mutex<Slot>> {
        let slot = Arc::new(Mutex::new(Slot::new(&self.tls_bytes, &self.total_bytes)));
        let mut g = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        g.retain(|w| w.strong_count() > 0);
        g.push(Arc::downgrade(&slot));
        slot
    }

    fn live_slots(&self) -> Vec<Arc<Mutex<Slot>>> {
        let mut g = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::with_capacity(g.len());
        g.retain(|w| match w.upgrade() {
            Some(arc) => {
                out.push(arc);
                true
            }
            None => false,
        });
        out
    }

    /// Drain live TLS slots only while `self.epoch` still matches `expected_epoch`, so a stale pre-clear return
    /// can never evict buffers admitted into TLS after `clear()`.
    fn drain_all_slots_if_epoch(&self, expected_epoch: u64) -> Vec<Vec<f32>> {
        let mut gone = Vec::new();
        for slot in self.live_slots() {
            if self.epoch.load(Ordering::Acquire) != expected_epoch {
                break;
            }
            let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
            if s.epoch > expected_epoch || self.epoch.load(Ordering::Acquire) != expected_epoch {
                continue;
            }
            s.drain_into(&mut gone);
        }
        gone
    }

    fn drain_all_slots_before(&self, new_epoch: u64) -> Vec<Vec<f32>> {
        let mut gone = Vec::new();
        for slot in self.live_slots() {
            let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
            s.sync_epoch(new_epoch, &mut gone);
        }
        gone
    }

    fn set_limit(&self, bytes: usize) {
        let (gone, stale_pool, evicted_map) = {
            let _limit_guard = self.set_limit_lock.lock().unwrap_or_else(|e| e.into_inner());
            self.limit.store(bytes, Ordering::Release);
            let clear_epoch =
                if bytes == 0 { Some(self.epoch.fetch_add(1, Ordering::AcqRel) + 1) } else { None };
            // Acquire every registered slot lock so any in-flight `Slot::put` that read an older limit either
            // finishes before we inspect/drain its slot or observes the updated limit after we unlock.
            let mut gone = Vec::new();
            for slot in self.live_slots() {
                let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(ne) = clear_epoch {
                    s.sync_epoch(ne, &mut gone);
                } else if s.bytes > 0
                    && (bytes < s.last_limit
                        || self.total_bytes.load(Ordering::Acquire) > bytes
                        || self.tls_bytes.load(Ordering::Acquire) > bytes / 2)
                {
                    s.drain_into(&mut gone);
                    s.last_limit = bytes;
                }
            }
            let (stale_pool, evicted_map) = {
                let mut g = self.pool.lock().unwrap_or_else(|e| e.into_inner());
                let stale = g.sync_epoch(self.epoch.load(Ordering::Acquire), &self.total_bytes);
                let evicted = if let Some(p) = g.pool.as_mut() {
                    let before = p.bytes;
                    // Check combined `total_bytes` too: an in-flight `put_global` that held `pool` with an old
                    // limit can coexist with a post-sweep TLS admission under `bytes` and push `total_bytes > bytes`.
                    let map = p.set_limit_evict(bytes, self.total_bytes.load(Ordering::Acquire));
                    let freed = before - p.bytes;
                    if freed > 0 {
                        self.total_bytes.fetch_sub(freed, Ordering::AcqRel);
                    }
                    map
                } else {
                    HashMap::new()
                };
                (stale, evicted)
            };
            (gone, stale_pool, evicted_map)
        };
        drop(gone);
        drop(stale_pool);
        drop(evicted_map);
    }

    fn clear(&self) {
        // Single atomic linearization point: bumping `epoch` immediately invalidates both levels on every lookup.
        let new_epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        let gone = self.drain_all_slots_before(new_epoch);
        let old_pool = {
            let mut g = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            g.sync_epoch(new_epoch, &self.total_bytes)
        };
        drop(gone);
        drop(old_pool);
    }

    fn bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Acquire)
    }

    fn take_from_pool(&self, n: usize) -> Option<Vec<f32>> {
        let (found, stale_pool) = {
            let mut g = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let stale = g.sync_epoch(self.epoch.load(Ordering::Acquire), &self.total_bytes);
            let found = g.pool.as_mut().and_then(|p| p.take(n));
            if found.is_some() {
                self.total_bytes.fetch_sub(n * 4, Ordering::AcqRel);
            }
            (found, stale)
        };
        drop(stale_pool);
        found
    }

    fn take_with_slot(&self, slot: Option<&Mutex<Slot>>, n: usize) -> Option<Vec<f32>> {
        if n < MIN_POOLED {
            return None;
        }
        if n.checked_mul(4).is_some_and(|b| b <= TLS_MAX_BUF_BYTES)
            && let Some(slot) = slot
        {
            let mut stale = Vec::new();
            let found = {
                let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
                s.sync_epoch(self.epoch.load(Ordering::Acquire), &mut stale);
                s.take(n)
            };
            drop(stale);
            if let Some(v) = found {
                return Some(v);
            }
        }
        self.take_from_pool(n)
    }

    fn put_global(&self, bufs: impl IntoIterator<Item = Vec<f32>>, expected_epoch: u64) {
        let mut pending: Vec<Vec<f32>> = bufs.into_iter().collect();
        if pending.is_empty() || self.epoch.load(Ordering::Acquire) != expected_epoch {
            return;
        }
        let limit = self.limit.load(Ordering::Acquire);
        if limit == 0 {
            return;
        }
        // If a single buffer fits under `limit` only when TLS gives up space (e.g. a small custom limit and a
        // buffer larger than `limit / 2`), drain same-epoch TLS once so the shared pool sees the full limit.
        let max_bytes = pending.iter().map(|v| v.len() * 4).max().unwrap_or(0);
        let mut drained = Vec::new();
        if max_bytes <= limit && self.tls_bytes.load(Ordering::Acquire) > limit - max_bytes {
            drained = self.drain_all_slots_if_epoch(expected_epoch);
        }
        let (stale_pool, gone) = {
            let mut g = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let cur_epoch = self.epoch.load(Ordering::Acquire);
            let stale = g.sync_epoch(cur_epoch, &self.total_bytes);
            let cur_limit = self.limit.load(Ordering::Acquire);
            // Re-check both epoch and limit inside the pool lock so a concurrent `clear()` or `set_limit()`
            // cannot be bypassed by pre-clear spilled buffers or a stale pre-lock limit.
            if cur_limit == 0 || expected_epoch != cur_epoch {
                (stale, pending)
            } else {
                let p = g.pool.get_or_insert_with(Pool::new);
                let mut evicted = Vec::new();
                for v in pending.drain(..) {
                    evicted.extend(p.put_with_budget(v, cur_limit, Some(&self.total_bytes)));
                }
                (stale, evicted)
            }
        };
        // Freed outside the lock: returning large buffers to the OS takes a while.
        drop(drained);
        drop(stale_pool);
        drop(gone);
    }

    fn recycle_with_slot(&self, slot: Option<&Mutex<Slot>>, v: Vec<f32>) {
        let n = v.len();
        if n < MIN_POOLED || v.capacity() != n {
            return;
        }
        if self.limit.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut spill = Vec::new();
        let mut stale = Vec::new();
        let (leftover, expected_epoch) = if let Some(slot) = slot {
            let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
            let cur_epoch = self.epoch.load(Ordering::Acquire);
            s.sync_epoch(cur_epoch, &mut stale);
            // Re-read `limit` under the slot lock so `set_limit` is serialized with every TLS admission.
            let cur_limit = self.limit.load(Ordering::Acquire);
            if cur_limit == 0 { (None, cur_epoch) } else { (s.put(v, cur_limit, &mut spill), cur_epoch) }
        } else {
            (Some(v), self.epoch.load(Ordering::Acquire))
        };
        drop(stale);
        if let Some(v) = leftover {
            spill.push(v);
        }
        if !spill.is_empty() {
            self.put_global(spill, expected_epoch);
        }
    }
}

static CACHE: Cache = Cache::new(1 << 30);

thread_local! {
    static TLS_SLOT: Arc<Mutex<Slot>> = CACHE.register_slot();
}

static POISON: AtomicBool = AtomicBool::new(false);

/// A testing aid: while on (process-wide), every buffer the CPU kernels take without initialising it is filled
/// with NaN, so an output element a kernel never wrote, or an input it reads that nobody wrote (results that
/// would otherwise depend on whatever a recycled buffer held), shows up as NaN.
pub fn poison_uninit(on: bool) {
    POISON.store(on, Ordering::Relaxed);
}

/// Cap the bytes held by the CPU buffer cache (default 1 GiB; 0 disables caching). Shrinks immediately.
pub fn set_buffer_cache_limit(bytes: usize) {
    CACHE.set_limit(bytes);
}

/// Release every cached buffer.
pub fn clear_buffer_cache() {
    CACHE.clear();
}

/// Bytes currently held by the cache.
pub fn buffer_cache_bytes() -> usize {
    CACHE.bytes()
}

fn take(n: usize) -> Option<Vec<f32>> {
    if n < MIN_POOLED {
        return None;
    }
    match TLS_SLOT.try_with(|slot| CACHE.take_with_slot(Some(slot), n)) {
        Ok(res) => res,
        Err(_) => CACHE.take_with_slot(None, n),
    }
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
    let mut v = Some(v);
    let handled = TLS_SLOT.try_with(|slot| {
        if let Some(buf) = v.take() {
            CACHE.recycle_with_slot(Some(slot), buf);
        }
    });
    if handled.is_err()
        && let Some(buf) = v.take()
    {
        CACHE.recycle_with_slot(None, buf);
    }
}

#[cfg(test)]
mod tests;
