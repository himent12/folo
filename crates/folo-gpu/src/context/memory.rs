//! Device memory: allocation from the buffer pool, uploads (direct, staged on another thread, or written into
//! reused staging buffers) and giving cached memory back to the driver.

use std::hash::Hasher as _;

use super::binds::Fx;
use super::{AnchorEntry, GpuContext, MAX_ANCHOR_CACHE, STAGING_KEEP, Staged};
use crate::error::{GpuError, Result};
use crate::pool::{GpuBuffer, PoolStats, size_class};

impl GpuContext {
    /// Buffer pool counters.
    pub fn pool_stats(&self) -> PoolStats {
        self.inner.pool.lock().expect("pool lock").stats()
    }

    /// Device memory wgpu's allocator holds for this context, `(allocated, reserved)` bytes: the sum of the live
    /// allocations and of the memory blocks they are carved from, which is what the driver has given the process.
    /// `None` where wgpu cannot report it (the browser, Metal, GL).
    pub fn device_memory(&self) -> Option<(u64, u64)> {
        self.inner
            .device
            .generate_allocator_report()
            .map(|r| (r.total_allocated_bytes, r.total_reserved_bytes))
    }

    /// Restart the pool's peak-memory counters ([`PoolStats::peak_live_bytes`] from the current live bytes,
    /// [`PoolStats::peak_held_bytes`] from the live plus cached bytes).
    pub fn reset_peak_memory(&self) {
        self.inner.pool.lock().expect("pool lock").reset_peak();
    }

    /// Drops every cached (free) buffer in the pool and lowers its cap back to the live bytes (wgpu frees the memory
    /// at its next submission or poll; [`GpuContext::release_memory`] also makes it happen now). Cached bind groups
    /// that hold those buffers are dropped too.
    pub fn trim_pool(&self) {
        // The batch lock first, the order `alloc` inside an encode takes them in.
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.inner.pool.lock().expect("pool lock").clear();
        self.sweep_binds(&mut b);
    }

    /// Gives the pool's cached buffers back to the driver now: [`GpuContext::trim_pool`], then submits the open
    /// batch and waits for the GPU, so wgpu frees the dropped buffers instead of holding them until the next
    /// submission, and drops the idle staging buffers of [`GpuContext::upload_with`]. Returns the bytes released.
    ///
    /// For a long-lived process between jobs (`folo serve` after a training run, or once inference has been idle):
    /// the pool otherwise keeps up to the job's high-water mark of buffers, gigabytes after a training run, which
    /// other processes on the same GPU then have to page around. Buffers still in use stay; a dropped buffer still
    /// read by encoded work is freed only once that work is done (wgpu keeps it alive until then). Pipelines
    /// precompile workers finished that no thread has taken ([`GpuContext::precompile`]: a list of another shape
    /// names kernels this process never used) are dropped too; ones still queued or compiling stay.
    pub fn release_memory(&self) -> Result<u64> {
        self.drop_finished_precompiles();
        self.bump_plan_epoch();
        let planned = self.plans().clear();
        let anchors = self.clear_anchors();
        let released = {
            // The batch lock first (the order `alloc` inside an encode takes them in), so nothing encodes between
            // the trim and the submission below.
            let mut b = self.inner.batch.lock().expect("batch lock");
            let released = {
                let mut pool = self.inner.pool.lock().expect("pool lock");
                let cached = pool.stats().cached_bytes;
                pool.clear();
                cached
            };
            self.submit(&mut b);
            // The idle upload staging buffers (host memory the driver pins); those still being mapped again are
            // dropped when their map completes.
            let staged: u64 = {
                let mut free = self.inner.staging.lock().expect("staging lock");
                free.generation += 1;
                free.bufs.drain(..).map(|b| b.size()).sum()
            };
            // The uniform arena too (recreated on the next dispatch): its 64 KiB chunks are allocated mid-job, and a
            // live one keeps the driver allocator's whole memory block (up to 256 MiB) from being freed. Cached bind
            // groups hold chunks and pool buffers alive, so they go as well.
            b.chunks.clear();
            b.binds.clear();
            self.inner.pool.lock().expect("pool lock").take_retired();
            released + staged + planned + anchors
        };
        self.wait_idle()?;
        Ok(released)
    }

    /// Drops every cached anchor table and returns the device bytes freed.
    fn clear_anchors(&self) -> u64 {
        let mut cache = self.inner.anchors.lock().unwrap_or_else(|p| p.into_inner());
        cache.entries.drain(..).map(|e| e.buf.size()).sum()
    }

    /// Cached device-resident anchor tables currently held (for tests).
    #[cfg(test)]
    pub(crate) fn anchor_cache_len(&self) -> usize {
        self.inner.anchors.lock().unwrap_or_else(|p| p.into_inner()).entries.len()
    }

    /// Returns a device buffer holding the packed `(x, y, stride)` anchor words (`3 * points.len()` `u32`s),
    /// reusing a cached copy across training steps so static anchor tables are uploaded once per image size.
    pub(crate) fn cached_anchors(&self, points: &[[f32; 2]], strides: &[f32]) -> Result<wgpu::Buffer> {
        if points.len() != strides.len() {
            return Err(GpuError::Shape(format!(
                "cached_anchors: {} points vs {} strides",
                points.len(),
                strides.len()
            )));
        }
        let n_words = points
            .len()
            .checked_mul(3)
            .ok_or_else(|| GpuError::Limit("cached_anchors: anchor table size overflow".into()))?;
        let bytes = (n_words as u64)
            .checked_mul(4)
            .ok_or_else(|| GpuError::Limit("cached_anchors: anchor table byte overflow".into()))?
            .max(wgpu::COPY_BUFFER_ALIGNMENT);
        if bytes > self.inner.caps.max_buffer_size {
            return Err(GpuError::Limit(format!(
                "anchor table of {bytes} bytes exceeds the device limit ({} bytes)",
                self.inner.caps.max_buffer_size
            )));
        }
        let mut hasher = Fx::default();
        hasher.write_usize(points.len());
        for (p, s) in points.iter().zip(strides) {
            hasher.write_u32(p[0].to_bits());
            hasher.write_u32(p[1].to_bits());
            hasher.write_u32(s.to_bits());
        }
        let key = hasher.finish();
        let matches = |words: &[u32]| {
            words.len() == n_words
                && points.iter().zip(strides).enumerate().all(|(i, (p, s))| {
                    words[3 * i] == p[0].to_bits()
                        && words[3 * i + 1] == p[1].to_bits()
                        && words[3 * i + 2] == s.to_bits()
                })
        };
        {
            let mut cache = self.inner.anchors.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(pos) = cache.entries.iter().position(|e| e.key == key && matches(&e.words)) {
                cache.tick += 1;
                let tick = cache.tick;
                cache.entries[pos].used = tick;
                return Ok(cache.entries[pos].buf.clone());
            }
        }
        let mut words = Vec::with_capacity(n_words);
        for (p, s) in points.iter().zip(strides) {
            words.push(p[0].to_bits());
            words.push(p[1].to_bits());
            words.push(s.to_bits());
        }
        let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu cached anchors"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !words.is_empty() {
            self.inner.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&words));
        }
        let mut cache = self.inner.anchors.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(pos) = cache.entries.iter().position(|e| e.key == key && matches(&e.words)) {
            cache.tick += 1;
            let tick = cache.tick;
            cache.entries[pos].used = tick;
            return Ok(cache.entries[pos].buf.clone());
        }
        if cache.entries.len() >= MAX_ANCHOR_CACHE {
            let lru = (0..cache.entries.len()).min_by_key(|&i| cache.entries[i].used).expect("not empty");
            cache.entries.swap_remove(lru);
        }
        cache.tick += 1;
        let tick = cache.tick;
        cache.entries.push(AnchorEntry { key, words, buf: buf.clone(), used: tick });
        Ok(buf)
    }

    /// Sets the pool's [`crate::pool::BufferPool::max_cached_bytes`] (for tests).
    #[cfg(test)]
    pub(crate) fn set_pool_cap(&self, bytes: u64) {
        self.inner.pool.lock().expect("pool lock").max_cached_bytes = bytes;
    }

    /// Idle staging buffers of [`GpuContext::upload_with`] (for tests).
    #[cfg(test)]
    pub(crate) fn idle_staging(&self) -> usize {
        self.inner.staging.lock().expect("staging lock").bufs.len()
    }

    /// Allocates a storage buffer of at least `bytes` bytes from the pool. Its contents are unspecified.
    pub fn alloc(&self, bytes: u64) -> Result<GpuBuffer> {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        self.alloc_usage(bytes, usage)
    }

    pub(super) fn alloc_usage(&self, bytes: u64, usage: wgpu::BufferUsages) -> Result<GpuBuffer> {
        let caps = &self.inner.caps;
        let storage = usage.contains(wgpu::BufferUsages::STORAGE);
        let limit = if storage {
            caps.max_buffer_size.min(caps.max_storage_buffer_binding_size)
        } else {
            caps.max_buffer_size
        };
        if bytes > limit {
            return Err(GpuError::Limit(format!(
                "buffer of {bytes} bytes exceeds the device limit ({limit} bytes)"
            )));
        }
        let mut class = size_class(bytes);
        if class > limit {
            // Near the limit, allocate the exact size instead of rounding past it.
            class = bytes.max(4).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        }
        if class > limit {
            return Err(GpuError::Limit(format!(
                "buffer of {bytes} bytes exceeds the device limit ({limit} bytes)"
            )));
        }
        let buf = self.inner.pool.lock().expect("pool lock").take(&self.inner.device, class, usage);
        self.record_taken(&buf);
        Ok(GpuBuffer::new(buf, class, self.inner.pool.clone()))
    }

    /// Uploads `data` into a new pooled storage buffer.
    pub fn upload(&self, data: &[u8]) -> Result<GpuBuffer> {
        let buf = self.alloc(data.len() as u64)?;
        self.write(&buf, 0, data)?;
        Ok(buf)
    }

    /// Copies `data` into a new host-visible staging buffer (mapped at creation), ready for
    /// [`GpuContext::upload_staged`]. This touches neither the command batch nor the queue, so another thread can
    /// stage the next batch of inputs while this one encodes and the GPU runs: the host-side copy leaves the
    /// critical path, and the device copy is one transfer command in stream order.
    pub fn stage(&self, data: &[u8]) -> Result<Staged> {
        let len = data.len() as u64;
        let size = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(wgpu::COPY_BUFFER_ALIGNMENT);
        let buf = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("folo-gpu staging"),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let copied = buf
            .slice(..)
            .get_mapped_range_mut()
            .map(|mut view| view.slice(..data.len()).copy_from_slice(data))
            .map_err(|e| self.or_lost(GpuError::Map(e.to_string())));
        buf.unmap();
        copied?;
        Ok(Staged { buf, len })
    }

    /// Uploads `len` bytes that `fill` writes straight into host-visible staging memory, into a new pooled storage
    /// buffer. `fill` gets the mapped bytes as a [`wgpu::WriteOnly`] slice (it can be split and written from
    /// several threads; bytes it leaves unwritten are unspecified). The copy is encoded after all work so far
    /// and the open batch is submitted (no wait).
    ///
    /// Unlike [`GpuContext::upload`] (`Queue::write_buffer`, which copies into a fresh staging allocation of
    /// wgpu's) and [`GpuContext::stage`] (a new buffer per call), the staging buffers are reused: once the GPU
    /// has copied out of one it is mapped again and kept for the next upload of the same size class (the
    /// last few are kept), so a stream of predictions writes its pixels once, into memory that is already mapped and
    /// paged in.
    pub fn upload_with<R>(
        &self,
        len: u64,
        fill: impl FnOnce(wgpu::WriteOnly<'_, [u8]>) -> R,
    ) -> Result<(GpuBuffer, R)> {
        self.record_taint("an upload");
        let len4 = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(wgpu::COPY_BUFFER_ALIGNMENT);
        let class = size_class(len4);
        let reused = {
            let mut free = self.inner.staging.lock().expect("staging lock");
            free.bufs.iter().position(|b| b.size() == class).map(|i| free.bufs.swap_remove(i))
        };
        let buf = reused.unwrap_or_else(|| {
            self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("folo-gpu upload staging"),
                size: class,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        });
        let written = buf
            .slice(..len4)
            .get_mapped_range_mut()
            .map(|mut view| fill(view.slice(..len as usize)))
            .map_err(|e| self.or_lost(GpuError::Map(e.to_string())));
        buf.unmap();
        let r = written?;
        let out = self.alloc(len4)?;
        self.copy(&buf, 0, out.raw(), 0, len4);
        // A buffer may only be mapped again once the work that reads it is submitted.
        self.flush();
        let free = self.inner.staging.clone();
        let generation = free.lock().expect("staging lock").generation;
        let again = buf.clone();
        buf.slice(..).map_async(wgpu::MapMode::Write, move |mapped| {
            let mut free = free.lock().expect("staging lock");
            if mapped.is_ok() && free.generation == generation {
                // The most recently used sizes stay.
                if free.bufs.len() == STAGING_KEEP {
                    free.bufs.remove(0);
                }
                free.bufs.push(again);
            }
        });
        Ok((out, r))
    }

    /// A pooled storage buffer holding the bytes of `staged`, filled by a copy encoded after all work so far
    /// (no submission, no wait).
    pub fn upload_staged(&self, staged: &Staged) -> Result<GpuBuffer> {
        self.record_taint("an upload");
        let out = self.alloc(staged.len.max(1))?;
        let len = staged.len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        self.copy(&staged.buf, 0, out.raw(), 0, len.min(staged.buf.size()));
        Ok(out)
    }

    /// Writes `data` at byte `offset` of `buf`, ordered after all work encoded so far.
    pub fn write(&self, buf: &GpuBuffer, offset: u64, data: &[u8]) -> Result<()> {
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Shape(format!("write offset {offset} is not 4-byte aligned")));
        }
        if offset + data.len() as u64 > buf.size() {
            return Err(GpuError::Shape(format!(
                "write of {} bytes at {offset} overruns a {}-byte buffer",
                data.len(),
                buf.size()
            )));
        }
        if data.is_empty() {
            return Ok(());
        }
        self.record_taint("an upload");
        // `Queue::write_buffer` runs before the next submission's commands, so submit what is open first:
        // otherwise a pooled buffer still read by unsubmitted work would be overwritten early.
        self.flush();
        if data.len().is_multiple_of(4) {
            self.inner.queue.write_buffer(buf.raw(), offset, data);
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().next_multiple_of(4), 0);
            self.inner.queue.write_buffer(buf.raw(), offset, &padded);
        }
        Ok(())
    }
}
