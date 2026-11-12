//! Buffer pool: size-class buckets so tensors that come and go every step reuse the same allocations.
//!
//! Sizes are rounded up to a *size class*: the next quarter step between powers of two (…, 1 MiB, 1.25 MiB,
//! 1.5 MiB, 1.75 MiB, 2 MiB, …) with a 256-byte floor, so at most 25% of a buffer is slack. Freed buffers go
//! back to a free list keyed by `(size class, usage)`; a later request for the same class takes the free buffer the
//! pool created first instead of asking the driver.
//!
//! Handing out the oldest free buffer, rather than the most recently freed one, makes the choice depend only on
//! *which* buffers are free, not on the order they came back in. A training step frees its tensors in a different
//! order than it allocated them (autograd's traversal, gradients in a hash map), so with last-in-first-out reuse
//! request *k* of a step got a different buffer every step and most of the step's bind groups missed the cache
//! (`context/binds.rs`). With the oldest-first rule the free set at the start of every step is the same (all of the
//! last step's temporaries), so by induction every request of the next step gets the same buffer as the step
//! before, as long as the program allocates and frees the same way; where it does not (a different batch size or
//! image size), the assignment differs for that stretch and falls back in step once the free sets agree again.
//!
//! The free lists hold at most the largest of [`BufferPool::max_cached_bytes`], the high-water mark of live bytes
//! and the *need* of the recent requests less the live bytes, so a training loop whose working set exceeds the
//! configured cap still reuses every buffer from one step to the next (like PyTorch's caching allocator). The need
//! is the sum over size classes of the class size times the most buffers of that class lent out at once in the
//! current or the previous window of [`NEED_WINDOW`] requests: what the pool must hold to serve those requests
//! without creating a buffer. It exceeds the live high-water mark when classes peak at different times (a folo-n
//! f32 training step needs 4.1 GiB of buffers for 3.4 GiB live at most), and without it the pool evicted and
//! recreated about 55 buffers every step. Classes no longer requested drop out of the need after two windows
//! (validation between epochs, the last partial batch). The cap is enforced on every free and on every allocation
//! miss, before the new buffer is created: the least recently freed buffers are evicted until the cache fits (the
//! driver frees them once in-flight work is done). With `M` the configured cap and `H` the high-water mark, the
//! memory the pool holds (live plus cached) is therefore at most `max(live + max(M, H), min(need, 2H))
//! <= 2 max(M, H)` at all times: never more than a cache bounded by `max(M, H)` alone could hold on top of the live
//! bytes. [`crate::GpuContext::trim_pool`] empties the cache.
//!
//! Reusing a buffer while earlier encoded work still reads it is safe: every dispatch and copy goes through
//! one queue in encoding order, and wgpu inserts the barriers between dispatches that touch the same buffer.
//! The one ordering hazard, `Queue::write_buffer` jumping ahead of not-yet-submitted work, is handled by
//! [`crate::GpuContext`], which flushes the open batch before every upload.
//!
//! The context caches bind groups by buffer, and a cached bind group keeps its buffers alive, so the pool records
//! the id of every buffer it lets go of ([`BufferPool::take_retired`]); the next dispatch or submission drops the
//! bind groups that use them (`context/binds.rs`). It also knows the ids of every buffer it created and has not let
//! go of ([`BufferPool::owns`]): only those are cached, since only their release is reported.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::context::binds::FxMap;

/// Smallest size class in bytes.
pub const MIN_CLASS: u64 = 256;
/// Retired buffer ids the pool keeps before it asks for the whole bind-group cache to be dropped instead.
const MAX_RETIRED: usize = 4096;
/// Requests per window of the need estimate (see the module docs): about three folo-n training steps, thirty
/// inference forwards.
pub(crate) const NEED_WINDOW: u64 = 4096;

/// Rounds a request up to its size class (see the module docs).
pub fn size_class(bytes: u64) -> u64 {
    let bytes = bytes.max(1);
    if bytes <= MIN_CLASS {
        return MIN_CLASS;
    }
    if bytes > 1 << 62 {
        // No GPU buffer is this large; return it unrounded (rounding would overflow) so the caller's limit
        // check rejects it.
        return bytes;
    }
    let p = bytes.next_power_of_two();
    let half = p / 2;
    let step = (half / 4).max(MIN_CLASS);
    let class = half + bytes.saturating_sub(half).div_ceil(step) * step;
    class.min(p).next_multiple_of(MIN_CLASS)
}

/// Counters for the pool, for `folo doctor` and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Requests served from a free list.
    pub hits: u64,
    /// Requests that created a new buffer.
    pub misses: u64,
    /// Bytes currently held in free lists.
    pub cached_bytes: u64,
    /// Bytes in buffers handed out and not yet returned.
    pub live_bytes: u64,
    /// Total bytes ever allocated from the driver.
    pub allocated_bytes: u64,
    /// The largest `live_bytes` seen since the pool was created or [`BufferPool::reset_peak`] ran (the
    /// equivalent of PyTorch's `max_memory_allocated`, in size-class bytes).
    pub peak_live_bytes: u64,
    /// The largest `live_bytes + cached_bytes` seen since the pool was created or [`BufferPool::reset_peak`] ran:
    /// the device memory the pool held at its fullest.
    pub peak_held_bytes: u64,
    /// The size-class bytes the recent requests needed (see the module docs), before the bound.
    pub need_bytes: u64,
}

type Key = (u64, wgpu::BufferUsages);

/// Free lists of buffers keyed by `(size class, usage)`.
pub struct BufferPool {
    /// Per key, the free buffers by `(creation number, free stamp)`; reuse takes the first (the oldest buffer, see
    /// the module docs). The stamp keeps keys unique even if two buffers' numbers collided.
    free: HashMap<Key, BTreeMap<(u64, u64), wgpu::Buffer>>,
    /// Every free event `(stamp, key, creation number)` in stamp order, for least-recently-freed eviction. Entries
    /// whose buffer was reused since are stale and skipped.
    order: VecDeque<(u64, Key, u64)>,
    stamp: u64,
    /// Creation number of the next new buffer.
    created: u64,
    cached_buffers: usize,
    high_water: u64,
    stats: PoolStats,
    /// Bytes the free lists may always hold (they may hold up to the live-bytes high-water mark).
    pub max_cached_bytes: u64,
    /// Ids of the buffers let go of since the last [`BufferPool::take_retired`].
    retired: Retired,
    /// Set when `retired` is not empty, so the dispatch path can check without taking the pool lock.
    retired_flag: Arc<AtomicBool>,
    /// Ids of the buffers this pool created and has not let go of (cached or lent out), with their creation numbers.
    owned: FxMap<u64>,
    /// Per key, the buffers lent out now and the most lent out at once in the current and the previous window.
    lent: HashMap<Key, Lent>,
    /// Requests in the current window.
    window_takes: u64,
}

/// Buffers of one key lent out now, and the most lent out at once in the current and the previous need window.
#[derive(Clone, Copy, Debug, Default)]
struct Lent {
    now: u32,
    cur: u32,
    prev: u32,
}

/// What [`BufferPool::take_retired`] returns: the ids of the buffers the pool let go of, or `All` when there
/// were more than [`MAX_RETIRED`].
#[derive(Debug, Default)]
pub(crate) enum Retired {
    #[default]
    None,
    Some(HashSet<u64>),
    All,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(2 << 30)
    }
}

impl BufferPool {
    /// Creates an empty pool that caches at most `max_cached_bytes`.
    pub fn new(max_cached_bytes: u64) -> Self {
        Self {
            free: HashMap::new(),
            order: VecDeque::new(),
            stamp: 0,
            created: 0,
            cached_buffers: 0,
            high_water: 0,
            stats: PoolStats::default(),
            max_cached_bytes,
            retired: Retired::None,
            retired_flag: Arc::default(),
            owned: FxMap::default(),
            lent: HashMap::new(),
            window_takes: 0,
        }
    }

    /// The flag set whenever a buffer is retired (cleared by [`BufferPool::take_retired`]).
    pub(crate) fn retired_flag(&self) -> Arc<AtomicBool> {
        self.retired_flag.clone()
    }

    /// The buffers let go of since the last call (see the module docs).
    pub(crate) fn take_retired(&mut self) -> Retired {
        self.retired_flag.store(false, Ordering::Relaxed);
        std::mem::take(&mut self.retired)
    }

    /// Whether the buffer with this id (`context::buffer_id`) is one the pool created and still holds or has lent
    /// out, so its release will be reported.
    pub(crate) fn owns(&self, id: u64) -> bool {
        self.owned.contains_key(&id)
    }

    /// Records that `buf` is no longer the pool's (it is dropped right after).
    fn retire(&mut self, buf: &wgpu::Buffer) {
        let id = crate::context::buffer_id(buf);
        self.owned.remove(&id);
        match &mut self.retired {
            Retired::All => {}
            Retired::Some(ids) if ids.len() >= MAX_RETIRED => self.retired = Retired::All,
            Retired::Some(ids) => {
                ids.insert(id);
            }
            Retired::None => self.retired = Retired::Some(HashSet::from([id])),
        }
        self.retired_flag.store(true, Ordering::Relaxed);
    }

    /// Current counters.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Restart [`PoolStats::peak_live_bytes`] from the current live bytes and [`PoolStats::peak_held_bytes`] from
    /// the current live plus cached bytes.
    pub fn reset_peak(&mut self) {
        self.stats.peak_live_bytes = self.stats.live_bytes;
        self.stats.peak_held_bytes = self.stats.live_bytes + self.stats.cached_bytes;
    }

    /// Starts a new need window; `forget` drops the previous one too (after a clear).
    fn roll_window(&mut self, forget: bool) {
        self.window_takes = 0;
        self.lent.retain(|_, l| {
            l.prev = if forget { l.now } else { l.cur };
            l.cur = l.now;
            l.prev > 0 || l.now > 0
        });
        self.stats.need_bytes = self.lent.iter().map(|(k, l)| k.0 * l.cur.max(l.prev) as u64).sum();
    }

    fn note_held(&mut self) {
        let held = self.stats.live_bytes + self.stats.cached_bytes;
        self.stats.peak_held_bytes = self.stats.peak_held_bytes.max(held);
    }

    /// The bytes the free lists may hold now: the largest of [`BufferPool::max_cached_bytes`], the live high-water
    /// mark and the need (at most twice the high-water mark) less the live bytes (see the module docs).
    fn cache_cap(&self) -> u64 {
        let need = self.stats.need_bytes.min(2 * self.high_water).saturating_sub(self.stats.live_bytes);
        self.max_cached_bytes.max(self.high_water).max(need)
    }

    /// Evicts the least recently freed buffers until the free lists hold at most `limit` bytes.
    fn evict_to(&mut self, limit: u64) {
        while self.stats.cached_bytes > limit {
            let Some((stamp, key, n)) = self.order.pop_front() else { break };
            if let Some(gone) = self.free.get_mut(&key).and_then(|l| l.remove(&(n, stamp))) {
                self.retire(&gone);
                self.stats.cached_bytes -= key.0;
                self.cached_buffers -= 1;
            }
        }
    }

    /// Drops every cached buffer.
    pub fn clear(&mut self) {
        let free = std::mem::take(&mut self.free);
        for (_, buf) in free.values().flatten() {
            self.retire(buf);
        }
        self.order.clear();
        self.cached_buffers = 0;
        self.stats.cached_bytes = 0;
        self.high_water = self.stats.live_bytes;
        self.roll_window(true);
    }

    /// Pops a cached buffer of class `class` and usage `usage`, or creates one with `device`.
    pub(crate) fn take(
        &mut self,
        device: &wgpu::Device,
        class: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.stats.live_bytes += class;
        self.high_water = self.high_water.max(self.stats.live_bytes);
        self.stats.peak_live_bytes = self.stats.peak_live_bytes.max(self.stats.live_bytes);
        let lent = self.lent.entry((class, usage)).or_default();
        lent.now += 1;
        if lent.now > lent.cur {
            if lent.now > lent.prev.max(lent.cur) {
                self.stats.need_bytes += class;
            }
            lent.cur = lent.now;
        }
        self.window_takes += 1;
        if self.window_takes == NEED_WINDOW {
            self.roll_window(false);
            // The need may have dropped: the bound holds after the rollover too, also when this request hits.
            self.evict_to(self.cache_cap());
        }
        if let Some((_, buf)) = self.free.get_mut(&(class, usage)).and_then(BTreeMap::pop_first) {
            self.stats.hits += 1;
            self.stats.cached_bytes -= class;
            self.cached_buffers -= 1;
            return buf;
        }
        // The bound holds on every allocation as well as every free, so a new buffer never lands on top of a cache
        // filled for a need the live buffers now take part of (see the module docs).
        self.evict_to(self.cache_cap());
        self.stats.misses += 1;
        self.stats.allocated_bytes += class;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu pooled"),
            size: class,
            usage,
            mapped_at_creation: false,
        });
        self.note_held();
        self.owned.insert(crate::context::buffer_id(&buf), self.created);
        self.created += 1;
        buf
    }

    /// Returns a buffer to its free list, evicting the least recently freed buffers if the cache is full.
    pub(crate) fn give(&mut self, buf: wgpu::Buffer, class: u64) {
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(class);
        if let Some(lent) = self.lent.get_mut(&(class, buf.usage())) {
            lent.now = lent.now.saturating_sub(1);
        }
        let cap = self.cache_cap();
        if class > cap {
            self.retire(&buf);
            return;
        }
        self.evict_to(cap - class);
        let key = (class, buf.usage());
        let n = self.owned.get(&crate::context::buffer_id(&buf)).copied().unwrap_or(u64::MAX);
        self.stamp += 1;
        self.free.entry(key).or_default().insert((n, self.stamp), buf);
        self.order.push_back((self.stamp, key, n));
        self.stats.cached_bytes += class;
        self.cached_buffers += 1;
        // Drop stale order entries (buffers reused since they were freed) once they dominate.
        if self.order.len() > 2 * self.cached_buffers + 1024 {
            let mut live: Vec<(u64, Key, u64)> =
                self.free.iter().flat_map(|(k, l)| l.keys().map(move |&(n, s)| (s, *k, n))).collect();
            live.sort_unstable_by_key(|e| e.0);
            self.order = live.into();
        }
    }
}

/// A buffer on loan from a [`BufferPool`]; it goes back to the pool when dropped.
pub struct GpuBuffer {
    buf: Option<wgpu::Buffer>,
    class: u64,
    pool: Arc<Mutex<BufferPool>>,
}

impl GpuBuffer {
    pub(crate) fn new(buf: wgpu::Buffer, class: u64, pool: Arc<Mutex<BufferPool>>) -> Self {
        Self { buf: Some(buf), class, pool }
    }

    /// The underlying wgpu buffer (its size is the size class, which can exceed what was asked for).
    pub fn raw(&self) -> &wgpu::Buffer {
        self.buf.as_ref().expect("buffer present until drop")
    }

    /// Allocated size in bytes (the size class).
    pub fn size(&self) -> u64 {
        self.class
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer").field("size", &self.class).finish()
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // A poisoned pool still takes the buffer back: giving it is what retires it (reports its release to the
        // bind-group cache), and a buffer dropped silently could stay pinned by a cached bind group.
        if let Some(buf) = self.buf.take() {
            self.pool.lock().unwrap_or_else(|p| p.into_inner()).give(buf, self.class);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_bound_slack() {
        assert_eq!(size_class(0), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(1 << 20), 1 << 20);
        assert_eq!(size_class((1 << 20) + 1), (1 << 20) + (1 << 18));
        assert_eq!(size_class(3 << 20), 3 << 20);
        assert_eq!(size_class(u64::MAX), u64::MAX);
        assert_eq!(size_class((1 << 62) + 1), (1 << 62) + 1);
        let mut prev = 0;
        for b in (1..5_000_000u64).step_by(9973) {
            let c = size_class(b);
            assert!(c >= b, "{b} -> {c}");
            assert!(c.is_multiple_of(256));
            assert!(c as f64 <= (b as f64 * 1.25).max(256.0) + 256.0, "{b} -> {c}");
            assert!(c >= prev);
            prev = c;
        }
    }
}
