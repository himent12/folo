//! The open command batch: kernels and copies encode into one command encoder (dispatches into one compute pass),
//! which is submitted when it grows large, before uploads and readbacks, and on [`GpuContext::flush`]. Also the
//! pipeline cache and the uniform arena that holds every dispatch's parameters.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{Batch, GpuContext, TracedKind, TracedOp, UNIFORM_CHUNK, UniformChunk};
use crate::error::{GpuError, Result};
use crate::pipeline::{Kernel, KernelDef, KernelKey, Params, Spec, UNIFORM_SLOT};
use crate::pool::Retired;

impl GpuContext {
    /// Number of compiled pipelines in the cache.
    pub fn pipeline_count(&self) -> usize {
        self.inner.kernels.lock().expect("kernel lock").len()
    }

    /// Copies `len` bytes between buffers, ordered with the other encoded work.
    pub fn copy(&self, src: &wgpu::Buffer, src_off: u64, dst: &wgpu::Buffer, dst_off: u64, len: u64) {
        if len == 0 {
            return;
        }
        self.record(|| TracedOp {
            name: "copy",
            spec: String::new(),
            groups: [(len / 4).min(u32::MAX as u64) as u32, 1, 1],
            kind: TracedKind::Copy { src: src.clone(), src_off, dst: dst.clone(), dst_off, len },
        });
        let mut b = self.inner.batch.lock().expect("batch lock");
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin("copy"));
        let enc = self.encoder(&mut b);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q);
        }
        enc.copy_buffer_to_buffer(src, src_off, dst, dst_off, len);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.bump(&mut b);
    }

    /// Zeroes the first `bytes` bytes of `buf` (rounded up to 4), ordered with the other encoded work.
    pub fn clear(&self, buf: &wgpu::Buffer, bytes: u64) {
        let bytes = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).min(buf.size());
        if bytes == 0 {
            return;
        }
        let mut b = self.inner.batch.lock().expect("batch lock");
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin("clear"));
        let enc = self.encoder(&mut b);
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q);
        }
        enc.clear_buffer(buf, 0, Some(bytes));
        if let (Some(p), Some(q)) = (&prof, q) {
            enc.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.bump(&mut b);
    }

    /// Submits the open batch without waiting.
    pub fn flush(&self) {
        let mut b = self.inner.batch.lock().expect("batch lock");
        self.submit(&mut b);
    }

    pub(super) fn encoder<'b>(&self, b: &'b mut Batch) -> &'b mut wgpu::CommandEncoder {
        b.pass = None;
        b.encoder.get_or_insert_with(|| {
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("folo-gpu batch") })
        })
    }

    fn bump(&self, b: &mut Batch) {
        b.dispatches += 1;
        let every = self.inner.flush_every.load(Ordering::Relaxed);
        if b.dispatches >= b.limit.min(every) {
            self.submit(b);
            b.limit = (b.limit * 2).min(every);
        }
    }

    pub(super) fn submit(&self, b: &mut Batch) {
        b.pass = None;
        let Some(mut enc) = b.encoder.take() else { return };
        let mut prof = self.profiler();
        let timed = prof.as_ref().and_then(|p| p.resolve(&mut enc));
        let used = if b.cursor == 0 { b.chunk } else { b.chunk + 1 };
        for (i, c) in b.chunks.iter().enumerate().take(used) {
            let len = if i == b.chunk { b.cursor } else { UNIFORM_CHUNK } as usize;
            if len > 0 {
                self.inner.queue.write_buffer(&c.buf, 0, &c.cpu[..len]);
            }
        }
        self.inner.queue.submit([enc.finish()]);
        self.inner.submissions.fetch_add(1, Ordering::Relaxed);
        if let (Some(p), Some(bytes)) = (prof.as_mut(), timed) {
            // Loss-aware: wgpu panics inside the poll of a lost device, which here would poison the batch lock.
            p.collect(bytes, || self.poll_checked(wgpu::PollType::wait_indefinitely()).is_ok());
        }
        b.dispatches = 0;
        b.chunk = 0;
        b.cursor = 0;
    }

    /// Compiles (or fetches from the cache) the pipeline for `def` specialised with `spec`.
    pub(crate) fn kernel(&self, def: &'static KernelDef, spec: Spec) -> Result<Arc<Kernel>> {
        let key = KernelKey { name: def.name, spec };
        if let Some(k) = self.inner.kernels.lock().expect("kernel lock").get(&key) {
            return Ok(k.clone());
        }
        // Compile outside the lock: pipeline creation can take tens of milliseconds.
        let kernel = Arc::new(Kernel::compile(&self.inner.device, def, &key.spec)?);
        Ok(self.inner.kernels.lock().expect("kernel lock").entry(key).or_insert(kernel).clone())
    }

    /// Encodes one dispatch of `kernel` with `buffers` bound at `@binding(1..)` and `params` at `@binding(0)`.
    pub(crate) fn dispatch(
        &self,
        kernel: &Kernel,
        buffers: &[&wgpu::Buffer],
        params: &Params,
        groups: [u32; 3],
    ) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        let max = self.inner.caps.max_workgroups_per_dim;
        if groups.iter().any(|&g| g > max) {
            return Err(GpuError::Limit(format!(
                "dispatch {groups:?} exceeds {max} workgroups per dimension"
            )));
        }
        if buffers.len() != kernel.storage.len() {
            return Err(GpuError::Shape(format!(
                "kernel {} takes {} buffers, got {}",
                kernel.name,
                kernel.storage.len(),
                buffers.len()
            )));
        }
        self.record(|| TracedOp {
            name: kernel.name,
            spec: kernel.spec.clone(),
            groups,
            kind: TracedKind::Dispatch {
                kernel: kernel.clone(),
                buffers: buffers.iter().map(|&b| b.clone()).collect(),
                params: Box::new(params.clone()),
            },
        });
        let mut guard = self.inner.batch.lock().expect("batch lock");
        let b = &mut *guard;
        if self.inner.retired.load(Ordering::Relaxed) {
            self.sweep_binds(b);
        }
        if b.cursor + UNIFORM_SLOT > UNIFORM_CHUNK {
            b.chunk += 1;
            b.cursor = 0;
        }
        if b.chunk == b.chunks.len() {
            b.chunks.push(UniformChunk {
                buf: self.inner.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("folo-gpu params"),
                    size: UNIFORM_CHUNK,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cpu: vec![0; UNIFORM_CHUNK as usize],
            });
        }
        let slot = b.cursor;
        let chunk = &mut b.chunks[b.chunk];
        chunk.cpu[slot as usize..(slot + UNIFORM_SLOT) as usize].copy_from_slice(params.bytes());
        let mut created = false;
        let bind = b.binds.get(&self.inner.device, &kernel.layout, &chunk.buf, buffers, &mut created);
        if created {
            self.inner.bind_groups.fetch_add(1, Ordering::Relaxed);
        }
        b.cursor += UNIFORM_SLOT.max(self.inner.caps.uniform_align as u64);
        if b.pass.is_none() {
            let enc = self.encoder(b);
            let pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None })
                .forget_lifetime();
            b.pass = Some(pass);
        }
        let pass = b.pass.as_mut().expect("pass");
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bind, &[slot as u32]);
        let mut prof = self.profiler();
        let q = prof.as_mut().map(|p| p.begin(kernel.name));
        if let (Some(p), Some(q)) = (&prof, q) {
            pass.write_timestamp(&p.set, q);
        }
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        if let (Some(p), Some(q)) = (&prof, q) {
            pass.write_timestamp(&p.set, q + 1);
        }
        drop(prof);
        self.inner.dispatches.fetch_add(1, Ordering::Relaxed);
        self.bump(b);
        Ok(())
    }

    /// Drops the cached bind groups that hold buffers the pool has let go of, so the driver can free them.
    pub(super) fn sweep_binds(&self, b: &mut Batch) {
        let retired = self.inner.pool.lock().expect("pool lock").take_retired();
        match retired {
            Retired::None => {}
            Retired::Some(ids) => b.binds.retire(&ids),
            Retired::All => b.binds.clear(),
        }
    }

    /// Bind groups in the cache (for tests and benchmarks).
    pub fn bind_cache_len(&self) -> usize {
        self.inner.batch.lock().expect("batch lock").binds.len()
    }

    /// A 1-D launch of `threads` invocations with `wg` threads per workgroup, folded into x/y when it exceeds
    /// the per-dimension limit. Shaders recover the linear id with `linear_id` from the prelude.
    pub(crate) fn grid_1d(&self, threads: u64, wg: u32) -> Result<[u32; 3]> {
        let groups = threads.div_ceil(wg as u64);
        let max = self.inner.caps.max_workgroups_per_dim as u64;
        if groups <= max {
            return Ok([groups as u32, 1, 1]);
        }
        let y = groups.div_ceil(max);
        if y > max {
            return Err(GpuError::Limit(format!("{threads} threads do not fit one dispatch")));
        }
        Ok([max as u32, y as u32, 1])
    }
}
