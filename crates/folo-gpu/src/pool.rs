//! Buffer pool: size-class buckets so tensors that come and go every step reuse the same allocations.
//!
//! Sizes are rounded up to a *size class*: the next quarter step between powers of two (…, 1 MiB, 1.25 MiB,
//! 1.5 MiB, 1.75 MiB, 2 MiB, …) with a 256-byte floor, so at most 25% of a buffer is slack. Freed buffers go
//! back to a free list keyed by `(size class, usage)`; a later request for the same class pops the most recently
//! freed one instead of asking the driver. The free lists hold at most the larger of
//! [`BufferPool::max_cached_bytes`] and the high-water mark of live bytes, so a training loop whose working set
//! exceeds the configured cap still reuses every buffer from one step to the next (like PyTorch's caching
//! allocator). When a returned buffer does not fit, the least recently freed buffers are evicted first (the
//! driver frees them once in-flight work is done); [`crate::GpuContext::trim_pool`] empties the cache.
//!
//! Reusing a buffer while earlier encoded work still reads it is safe: every dispatch and copy goes through
//! one queue in encoding order, and wgpu inserts the barriers between dispatches that touch the same buffer.
//! The one ordering hazard, `Queue::write_buffer` jumping ahead of not-yet-submitted work, is handled by
//! [`crate::GpuContext`], which flushes the open batch before every upload.
//!
//! The context caches bind groups by buffer, and a cached bind group keeps its buffers alive, so the pool records
//! the id of every buffer it lets go of ([`BufferPool::take_retired`]); the next dispatch drops the bind groups
//! that use them (`context/binds.rs`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Smallest size class in bytes.
pub const MIN_CLASS: u64 = 256;
/// Retired buffer ids the pool keeps before it asks for the whole bind-group cache to be dropped instead.
const MAX_RETIRED: usize = 4096;

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
}

type Key = (u64, wgpu::BufferUsages);

/// Free lists of buffers keyed by `(size class, usage)`.
pub struct BufferPool {
    /// Per key, `(free stamp, buffer)` oldest first; reuse pops the newest.
    free: HashMap<Key, VecDeque<(u64, wgpu::Buffer)>>,
    /// Every free event in stamp order, for least-recently-freed eviction. Entries whose buffer was reused
    /// since are stale and skipped.
    order: VecDeque<(u64, Key)>,
    stamp: u64,
    cached_buffers: usize,
    high_water: u64,
    stats: PoolStats,
    /// Bytes the free lists may always hold (they may hold up to the live-bytes high-water mark).
    pub max_cached_bytes: u64,
    /// Ids of the buffers let go of since the last [`BufferPool::take_retired`].
    retired: Retired,
    /// Set when `retired` is not empty, so the dispatch path can check without taking the pool lock.
    retired_flag: Arc<AtomicBool>,
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
            cached_buffers: 0,
            high_water: 0,
            stats: PoolStats::default(),
            max_cached_bytes,
            retired: Retired::None,
            retired_flag: Arc::default(),
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

    /// Records that `buf` is no longer the pool's (it is dropped right after).
    fn retire(&mut self, buf: &wgpu::Buffer) {
        match &mut self.retired {
            Retired::All => {}
            Retired::Some(ids) if ids.len() >= MAX_RETIRED => self.retired = Retired::All,
            Retired::Some(ids) => {
                ids.insert(crate::context::buffer_id(buf));
            }
            Retired::None => self.retired = Retired::Some(HashSet::from([crate::context::buffer_id(buf)])),
        }
        self.retired_flag.store(true, Ordering::Relaxed);
    }

    /// Current counters.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Restart [`PoolStats::peak_live_bytes`] from the current live bytes.
    pub fn reset_peak(&mut self) {
        self.stats.peak_live_bytes = self.stats.live_bytes;
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
        if let Some((_, buf)) = self.free.get_mut(&(class, usage)).and_then(VecDeque::pop_back) {
            self.stats.hits += 1;
            self.stats.cached_bytes -= class;
            self.cached_buffers -= 1;
            return buf;
        }
        self.stats.misses += 1;
        self.stats.allocated_bytes += class;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu pooled"),
            size: class,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Returns a buffer to its free list, evicting the least recently freed buffers if the cache is full.
    pub(crate) fn give(&mut self, buf: wgpu::Buffer, class: u64) {
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(class);
        let cap = self.max_cached_bytes.max(self.high_water);
        if class > cap {
            self.retire(&buf);
            return;
        }
        while self.stats.cached_bytes + class > cap {
            let Some((stamp, key)) = self.order.pop_front() else { break };
            let Some(list) = self.free.get_mut(&key) else { continue };
            if list.front().is_some_and(|(s, _)| *s == stamp) {
                let (_, gone) = list.pop_front().expect("front checked");
                self.retire(&gone);
                self.stats.cached_bytes -= key.0;
                self.cached_buffers -= 1;
            }
        }
        let key = (class, buf.usage());
        self.stamp += 1;
        self.free.entry(key).or_default().push_back((self.stamp, buf));
        self.order.push_back((self.stamp, key));
        self.stats.cached_bytes += class;
        self.cached_buffers += 1;
        // Drop stale order entries (buffers reused since they were freed) once they dominate.
        if self.order.len() > 2 * self.cached_buffers + 1024 {
            let mut live: Vec<(u64, Key)> =
                self.free.iter().flat_map(|(k, l)| l.iter().map(move |(s, _)| (*s, *k))).collect();
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
        if let Some(buf) = self.buf.take()
            && let Ok(mut pool) = self.pool.lock()
        {
            pool.give(buf, self.class);
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
