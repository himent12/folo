//! Recording the operations of a pass ([`TracedOp`]) to replay them one by one, and the per-kernel timestamp
//! profiler ([`crate::profile`]).

use std::sync::atomic::Ordering;

use super::{GpuContext, TracedKind, TracedOp};
use crate::error::Result;
use crate::profile::{ProfileRow, Profiler};

#[cfg(test)]
impl TracedOp {
    /// Whether the operation binds or copies `buf`.
    pub(crate) fn touches(&self, buf: &wgpu::Buffer) -> bool {
        match &self.kind {
            TracedKind::Dispatch { buffers, .. } => buffers.contains(buf),
            TracedKind::Copy { src, dst, .. } => src == buf || dst == buf,
            TracedKind::Clear { buf: b, .. } => b == buf,
        }
    }
}

impl GpuContext {
    /// Starts recording every dispatch and buffer copy encoded from now on (for profiling; see [`TracedOp`]).
    /// The recorded operations keep their buffers alive until the trace is dropped.
    pub fn trace_begin(&self) {
        self.inner.trace.lock().expect("trace lock").clear();
        self.inner.tracing.store(true, Ordering::Relaxed);
    }

    /// Stops recording and returns the operations since [`GpuContext::trace_begin`], in encoding order.
    pub fn trace_end(&self) -> Vec<TracedOp> {
        self.inner.tracing.store(false, Ordering::Relaxed);
        std::mem::take(&mut *self.inner.trace.lock().expect("trace lock"))
    }

    /// Encodes a recorded operation again, on the buffers it ran on (their contents are whatever they hold
    /// now: pooled buffers may have been reused, which does not change what the operation costs).
    pub fn replay(&self, op: &TracedOp) -> Result<()> {
        match &op.kind {
            TracedKind::Dispatch { kernel, buffers, params } => {
                let refs: Vec<&wgpu::Buffer> = buffers.iter().collect();
                self.dispatch(kernel, &refs, params, op.groups)
            }
            TracedKind::Copy { src, src_off, dst, dst_off, len } => {
                self.copy(src, *src_off, dst, *dst_off, *len);
                Ok(())
            }
            TracedKind::Clear { buf, bytes } => {
                self.clear(buf, *bytes);
                Ok(())
            }
        }
    }

    /// Whether [`GpuContext::trace_begin`] is recording.
    pub(crate) fn tracing(&self) -> bool {
        self.inner.tracing.load(Ordering::Relaxed)
    }

    pub(super) fn record(&self, op: impl FnOnce() -> TracedOp) {
        let tracing = self.inner.tracing.load(Ordering::Relaxed);
        let recording = self.recording();
        if !tracing && !recording {
            return;
        }
        let op = op();
        if recording {
            self.record_op(op.clone());
        }
        if tracing {
            self.inner.trace.lock().expect("trace lock").push(op);
        }
    }

    pub(super) fn profiler(&self) -> Option<std::sync::MutexGuard<'_, Profiler>> {
        self.inner.profiler.as_ref().map(|p| p.lock().expect("profiler lock"))
    }

    /// Whether this context times every kernel (`FOLO_GPU_PROFILE=1` on an adapter with timestamp queries inside
    /// passes; see [`crate::profile`]).
    pub fn profiling(&self) -> bool {
        self.inner.profiler.is_some()
    }

    /// The file of this context's persistent pipeline cache, when it has one ([`crate::pipeline_cache`]).
    pub fn pipeline_cache_file(&self) -> Option<&std::path::Path> {
        self.inner.pipeline_cache.as_deref().map(|d| d.path())
    }

    /// Whether this context's pipeline cache started from a valid file (false without a cache, or when the file
    /// was missing, corrupt or another adapter's and the cache started empty).
    pub fn pipeline_cache_loaded(&self) -> bool {
        self.inner.pipeline_cache.as_deref().is_some_and(|d| d.loaded)
    }

    /// The size of the data the pipeline cache holds now (tests: what wgpu was given, right after the open).
    #[cfg(test)]
    pub(crate) fn pipeline_cache_data_len(&self) -> Option<usize> {
        self.inner.pipeline_cache.as_deref().and_then(|d| d.cache.get_data()).map(|d| d.len())
    }

    /// Submits the open batch, waits for it, and returns the GPU time per label since the last call (largest
    /// first), resetting the totals. Empty unless [`GpuContext::profiling`].
    pub fn profile_take(&self) -> Vec<ProfileRow> {
        if !self.profiling() {
            return Vec::new();
        }
        self.flush();
        self.profiler().map(|mut p| p.take()).unwrap_or_default()
    }
}
