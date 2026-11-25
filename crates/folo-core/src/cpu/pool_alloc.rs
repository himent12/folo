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
        drop(self.set_limit_evict(bytes));
    }

    /// Like [`Pool::set_limit`], returning any dropped map entries so the caller can free them outside the lock.
    fn set_limit_evict(&mut self, bytes: usize) -> HashMap<usize, Class> {
        let evicted = if self.bytes > bytes {
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
/// atomic point. No method holds more than one lock at a time (`slots`, an individual `Slot`, and `pool` are
/// always acquired and released separately), so deadlock is impossible and freed buffers drop outside all locks.
struct Cache {
    limit: AtomicUsize,
    epoch: AtomicU64,
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

    fn drain_all_slots(&self) -> Vec<Vec<f32>> {
        let mut gone = Vec::new();
        for slot in self.live_slots() {
            let mut s = slot.lock().unwrap_or_else(|e| e.into_inner());
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
        self.limit.store(bytes, Ordering::Release);
        let clear_epoch = if bytes == 0 { Some(self.epoch.fetch_add(1, Ordering::AcqRel) + 1) } else { None };
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
            let effective = self.limit.load(Ordering::Acquire);
            let evicted = if let Some(p) = g.pool.as_mut() {
                let before = p.bytes;
                let map = p.set_limit_evict(effective);
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

    fn put_global(&self, bufs: impl IntoIterator<Item = Vec<f32>>, expected_epoch: Option<u64>) {
        let mut pending: Vec<Vec<f32>> = bufs.into_iter().collect();
        if pending.is_empty() {
            return;
        }
        let limit = self.limit.load(Ordering::Acquire);
        if limit == 0 {
            return;
        }
        // If a single buffer fits under `limit` only when TLS gives up space (e.g. a small custom limit and a
        // buffer larger than `limit / 2`), drain TLS once so the shared pool sees the full limit.
        let max_bytes = pending.iter().map(|v| v.len() * 4).max().unwrap_or(0);
        let mut drained = Vec::new();
        if max_bytes <= limit && self.tls_bytes.load(Ordering::Acquire) > limit - max_bytes {
            drained = self.drain_all_slots();
        }
        let (stale_pool, gone) = {
            let mut g = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            let cur_epoch = self.epoch.load(Ordering::Acquire);
            let stale = g.sync_epoch(cur_epoch, &self.total_bytes);
            let cur_limit = self.limit.load(Ordering::Acquire);
            // Re-check both epoch and limit inside the pool lock so a concurrent `clear()` or `set_limit()`
            // cannot be bypassed by pre-clear spilled buffers or a stale pre-lock limit.
            if cur_limit == 0 || expected_epoch.is_some_and(|ee| ee != cur_epoch) {
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
            if cur_limit == 0 {
                (None, Some(cur_epoch))
            } else {
                (s.put(v, cur_limit, &mut spill), Some(cur_epoch))
            }
        } else {
            (Some(v), None)
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

    /// A limit of `usize::MAX` (no limit) caches everything and never overflows the traffic arithmetic.
    #[test]
    fn an_unbounded_limit_caches_everything() {
        let mut p = Pool::new();
        for i in 0..64 {
            assert_eq!(put(&mut p, N + i, usize::MAX), 0);
        }
        assert!(p.take(N).is_some());
        p.forget(usize::MAX);
        let mut out = Vec::new();
        p.evict_stale(usize::MAX, &mut out);
        assert!(out.is_empty());
        assert_eq!((p.bytes, p.next_forget, p.next_stale), ((63 * N + 64 * 63 / 2) * 4, u64::MAX, u64::MAX));
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

    /// Thread-local magazine hits without touching the shared pool, spills oldest buffers to the shared pool
    /// once full, and `clear()` / `set_limit(0)` immediately releases buffers held on other threads.
    #[test]
    fn tls_magazine_hits_spills_clears_and_reclaims_on_thread_exit() {
        static TEST_CACHE: Cache = Cache::new(1 << 24);
        let slot = TEST_CACHE.register_slot();

        // 1. Recycle & take on the same slot hits TLS without populating the shared pool.
        let mut v = vec![1.25f32; N];
        let ptr = v.as_ptr();
        TEST_CACHE.recycle_with_slot(Some(&slot), v);
        assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), N * 4);
        assert_eq!(TEST_CACHE.bytes(), N * 4);
        v = TEST_CACHE.take_with_slot(Some(&slot), N).expect("TLS hit");
        assert_eq!(v.as_ptr(), ptr, "reused exact buffer from TLS magazine");
        assert_eq!(TEST_CACHE.bytes(), 0);

        // 2. Filling past `TLS_MAX_BUFS` spills the oldest buffers into the shared pool, so another thread (or
        //    slot-less take) can still reuse them while the newest stay in TLS.
        for i in 0..TLS_MAX_BUFS + 3 {
            TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N + i]);
        }
        let expected_tls: usize = (3..TLS_MAX_BUFS + 3).map(|i| (N + i) * 4).sum();
        let expected_pool: usize = (0..3).map(|i| (N + i) * 4).sum();
        assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), expected_tls);
        assert_eq!(TEST_CACHE.bytes() - expected_tls, expected_pool);
        assert!(TEST_CACHE.take_with_slot(None, N).is_some(), "spilled oldest length found in shared pool");
        assert!(
            TEST_CACHE.take_with_slot(Some(&slot), N + TLS_MAX_BUFS + 2).is_some(),
            "newest length in TLS"
        );

        // 3. `clear()` releases buffers cached on a separate live worker thread immediately.
        let (tx_ready, rx_ready) = std::sync::mpsc::channel();
        let (tx_exit, rx_exit) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let wslot = TEST_CACHE.register_slot();
            TEST_CACHE.recycle_with_slot(Some(&wslot), vec![0.0; N * 2]);
            tx_ready.send(()).unwrap();
            rx_exit.recv().unwrap();
            // Recycle one more buffer, then exit without clearing so `Drop for Slot` must reclaim it.
            TEST_CACHE.recycle_with_slot(Some(&wslot), vec![0.0; N * 3]);
        });
        rx_ready.recv().unwrap();
        assert!(TEST_CACHE.bytes() >= N * 2 * 4);
        TEST_CACHE.clear();
        assert_eq!(TEST_CACHE.bytes(), 0, "clear() must drain live worker TLS slots immediately");
        tx_exit.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(TEST_CACHE.bytes(), 0, "exiting thread must release its TLS bytes on Slot::drop");

        // 4. `set_limit(0)` shrinks immediately and refuses subsequent recycles; restoring limit re-enables caching.
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
        assert_eq!(TEST_CACHE.bytes(), N * 4);
        TEST_CACHE.set_limit(0);
        assert_eq!(TEST_CACHE.bytes(), 0);
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
        assert_eq!(TEST_CACHE.bytes(), 0);

        // 5. Small custom limit: both TLS (`N`) and pool (`3 * N`) can coexist up to `4 * N`, and when a `3 * N`
        //    buffer arrives while TLS holds `2 * N` (leaving only `2 * N` for the pool), TLS is drained once so
        //    the `3 * N` buffer still fits in the shared pool.
        let small_limit = 4 * N * 4;
        TEST_CACHE.set_limit(small_limit);
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
        assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), N * 4);
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 3 * N]);
        assert_eq!(TEST_CACHE.bytes(), 4 * N * 4);
        assert!(TEST_CACHE.take_with_slot(Some(&slot), 3 * N).is_some());
        assert!(TEST_CACHE.take_with_slot(Some(&slot), N).is_some());
        assert_eq!(TEST_CACHE.bytes(), 0);

        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 2 * N]);
        assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), 2 * N * 4);
        TEST_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 3 * N]);
        assert_eq!(TEST_CACHE.tls_bytes.load(Ordering::Acquire), 0);
        assert_eq!(TEST_CACHE.bytes(), 3 * N * 4);
        assert!(TEST_CACHE.take_with_slot(Some(&slot), 3 * N).is_some());
        TEST_CACHE.clear();
    }

    /// Concurrent TLS + pool returns never exceed `limit`, `set_limit` reductions drain in-flight admissions,
    /// and `clear()`'s generation bump invalidates both levels atomically.
    #[test]
    fn cache_budget_limit_reduction_and_clear_are_atomic_across_levels() {
        // 1. Atomic clear across levels: bumping `epoch` before draining must miss BOTH TLS and Pool lookups.
        static CLEAR_CACHE: Cache = Cache::new(8 * N * 4);
        let slot = CLEAR_CACHE.register_slot();
        CLEAR_CACHE.recycle_with_slot(Some(&slot), vec![0.0; N]);
        CLEAR_CACHE.recycle_with_slot(Some(&slot), vec![0.0; 5 * N]);
        assert_eq!((CLEAR_CACHE.tls_bytes.load(Ordering::Acquire), CLEAR_CACHE.bytes()), (N * 4, 6 * N * 4));
        CLEAR_CACHE.epoch.fetch_add(1, Ordering::AcqRel);
        assert!(CLEAR_CACHE.take_with_slot(Some(&slot), N).is_none());
        assert!(CLEAR_CACHE.take_with_slot(Some(&slot), 5 * N).is_none());
        assert_eq!(CLEAR_CACHE.bytes(), 0);

        // 2. Concurrent TLS (`4 * N`) + Pool (`6 * N`) returns never exceed `limit = 8 * N`.
        const LIMIT: usize = 8 * N * 4;
        static RACE_CACHE: Cache = Cache::new(LIMIT);
        let threads: Vec<_> = (0..4)
            .map(|tid| {
                std::thread::spawn(move || {
                    let s = RACE_CACHE.register_slot();
                    for step in 0..200 {
                        let len = if (tid + step) % 2 == 0 { 4 * N } else { 6 * N };
                        RACE_CACHE.recycle_with_slot(Some(&s), vec![0.0; len]);
                        assert!(RACE_CACHE.bytes() <= LIMIT, "{} > {LIMIT}", RACE_CACHE.bytes());
                        let _ = RACE_CACHE.take_with_slot(Some(&s), len);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        RACE_CACHE.clear();
        assert_eq!(RACE_CACHE.bytes(), 0);

        // 3. `set_limit` reduction serializes on every slot lock and drains an in-flight old-limit admission.
        static LIMIT_CACHE: Cache = Cache::new(8 * N * 4);
        let wslot = LIMIT_CACHE.register_slot();
        let wslot_bg = wslot.clone();
        let (tx_locked, rx_locked) = std::sync::mpsc::channel();
        let (tx_go, rx_go) = std::sync::mpsc::channel();
        let t_in_flight = std::thread::spawn(move || {
            let mut s = wslot_bg.lock().unwrap();
            let (cur_epoch, old_limit) =
                (LIMIT_CACHE.epoch.load(Ordering::Acquire), LIMIT_CACHE.limit.load(Ordering::Acquire));
            tx_locked.send(()).unwrap();
            rx_go.recv().unwrap();
            let (mut spill, mut stale) = (Vec::new(), Vec::new());
            s.sync_epoch(cur_epoch, &mut stale);
            assert!(s.put(vec![0.0; 3 * N], old_limit, &mut spill).is_none());
        });
        rx_locked.recv().unwrap();
        let t_shrink = std::thread::spawn(|| LIMIT_CACHE.set_limit(2 * N * 4));
        while LIMIT_CACHE.limit.load(Ordering::Acquire) != 2 * N * 4 {
            std::thread::yield_now();
        }
        tx_go.send(()).unwrap();
        t_in_flight.join().unwrap();
        t_shrink.join().unwrap();
        assert!(LIMIT_CACHE.bytes() <= 2 * N * 4, "survived shrink: {}", LIMIT_CACHE.bytes());
        LIMIT_CACHE.clear();
        drop(wslot);
    }
}
