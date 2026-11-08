//! Buffer pool: size-class buckets so tensors that come and go every step reuse the same allocations.
//!
//! Sizes are rounded up to a *size class*: the next quarter step between powers of two (…, 1 MiB, 1.25 MiB,
//! 1.5 MiB, 1.75 MiB, 2 MiB, …) with a 256-byte floor, so at most 25% of a buffer is slack. Freed buffers go
//! back to a free list keyed by `(size class, usage)`; a later request for the same class pops one instead of
//! asking the driver. The free lists are capped at [`BufferPool::max_cached_bytes`]; beyond that the buffer is
//! dropped (the driver frees it once in-flight work is done).
//!
//! Reusing a buffer while earlier encoded work still reads it is safe: every dispatch and copy goes through
//! one queue in encoding order, and wgpu inserts the barriers between dispatches that touch the same buffer.
//! The one ordering hazard, `Queue::write_buffer` jumping ahead of not-yet-submitted work, is handled by
//! [`crate::GpuContext`], which flushes the open batch before every upload.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Smallest size class in bytes.
pub const MIN_CLASS: u64 = 256;

/// Rounds a request up to its size class (see the module docs).
pub fn size_class(bytes: u64) -> u64 {
    let bytes = bytes.max(1);
    if bytes <= MIN_CLASS {
        return MIN_CLASS;
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
}

/// Free lists of buffers keyed by `(size class, usage)`.
pub struct BufferPool {
    free: HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>,
    stats: PoolStats,
    /// Upper bound on bytes kept in free lists.
    pub max_cached_bytes: u64,
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(2 << 30)
    }
}

impl BufferPool {
    /// Creates an empty pool that caches at most `max_cached_bytes`.
    pub fn new(max_cached_bytes: u64) -> Self {
        Self { free: HashMap::new(), stats: PoolStats::default(), max_cached_bytes }
    }

    /// Current counters.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }

    /// Drops every cached buffer.
    pub fn clear(&mut self) {
        self.free.clear();
        self.stats.cached_bytes = 0;
    }

    /// Pops a cached buffer of class `class` and usage `usage`, or creates one with `device`.
    pub(crate) fn take(
        &mut self,
        device: &wgpu::Device,
        class: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.stats.live_bytes += class;
        if let Some(buf) = self.free.get_mut(&(class, usage)).and_then(Vec::pop) {
            self.stats.hits += 1;
            self.stats.cached_bytes -= class;
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

    /// Returns a buffer to its free list (or drops it if the cache is full).
    pub(crate) fn give(&mut self, buf: wgpu::Buffer, class: u64) {
        self.stats.live_bytes = self.stats.live_bytes.saturating_sub(class);
        if self.stats.cached_bytes + class > self.max_cached_bytes {
            return;
        }
        self.stats.cached_bytes += class;
        self.free.entry((class, buf.usage())).or_default().push(buf);
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
