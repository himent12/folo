//! Reading buffers back to the host: single, deferred (non-blocking), and single-submission batched copies into
//! pooled staging buffers, with submission-targeted waits natively and event-loop futures in the browser.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

use super::GpuContext;
#[cfg(target_arch = "wasm32")]
use super::NO_BLOCKING;
use crate::error::{GpuError, Result};
use crate::pool::GpuBuffer;

impl GpuContext {
    /// Reads `len` bytes at `offset` of `buf` back to the host (flushes and waits). Not available in the
    /// browser, where [`GpuContext::read_async`] is the way to read.
    pub fn read(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with(buf, offset, len, <[u8]>::to_vec)
    }

    /// [`GpuContext::read`] that hands the mapped bytes to `f` instead of copying them into a `Vec` first,
    /// so a caller converting them (to typed host storage) copies once.
    pub fn read_with<R>(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        self.read_deferred(buf, offset, len)?.finish(f)
    }

    /// Reads `len` bytes at `offset` of `buf` back to the host without blocking: the copy is ordered after
    /// all work encoded so far, and the future resolves once the GPU has finished it. This is the browser's
    /// readback (WebGPU maps buffers asynchronously and the page's thread must not wait); natively it waits
    /// only for the readback's submission.
    pub async fn read_async(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.read_with_async(buf, offset, len, <[u8]>::to_vec).await
    }

    /// [`GpuContext::read_async`] that hands the mapped bytes to `f` (see [`GpuContext::read_with`]).
    pub async fn read_with_async<R>(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<R> {
        self.read_deferred(buf, offset, len)?.finish_async(f).await
    }

    /// Starts a non-blocking readback of `len` bytes at `offset` of `buf` and returns immediately after
    /// submitting the copy and requesting the staging map.
    ///
    /// Subsequent GPU work can be encoded and submitted before collecting the result with
    /// [`Readback::finish`] / [`Readback::finish_async`]: waiting targets only this readback's submission
    /// instead of draining later queued work. Dropping an unfinished [`Readback`] unmaps and retires its
    /// staging buffer from the pool cleanly.
    pub fn read_deferred(&self, buf: &wgpu::Buffer, offset: u64, len: u64) -> Result<Readback> {
        let mut batch = self.read_batch_deferred(&[(buf, offset, len)])?;
        Ok(batch.pop().expect("single readback"))
    }

    /// Starts a batch of readbacks in a single command buffer and a single queue submission.
    ///
    /// Every slice is validated and its staging buffer allocated before anything is encoded or submitted, so an
    /// error on one slice leaves the queue untouched. All non-empty copies are encoded into one command encoder,
    /// submitted together, and mapped in parallel; collecting them shares that one submission fence.
    pub fn read_batch_deferred(&self, reads: &[(&wgpu::Buffer, u64, u64)]) -> Result<Vec<Readback>> {
        if reads.is_empty() {
            return Ok(Vec::new());
        }
        let mut prepared: Vec<Option<(usize, GpuBuffer, u64, u64)>> = Vec::with_capacity(reads.len());
        let mut any_non_empty = false;
        for (idx, &(buf, offset, len)) in reads.iter().enumerate() {
            if len == 0 {
                prepared.push(None);
                continue;
            }
            if !buf.usage().contains(wgpu::BufferUsages::COPY_SRC) {
                return Err(GpuError::Shape("buffer cannot be read back (missing COPY_SRC usage)".into()));
            }
            if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                return Err(GpuError::Shape(format!("read offset {offset} is not 4-byte aligned")));
            }
            let Some(len4) = len.checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) else {
                return Err(GpuError::Shape(format!(
                    "read of {len} bytes at {offset} overruns a {}-byte buffer",
                    buf.size()
                )));
            };
            if offset.checked_add(len4).is_none_or(|end| end > buf.size()) {
                return Err(GpuError::Shape(format!(
                    "read of {len} bytes at {offset} overruns a {}-byte buffer",
                    buf.size()
                )));
            }
            let staging =
                self.alloc_usage(len4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
            prepared.push(Some((idx, staging, len, len4)));
            any_non_empty = true;
        }
        if !any_non_empty {
            return Ok((0..reads.len()).map(|_| Readback { inner: None }).collect());
        }
        self.record_taint("a readback");
        // Hold `batch` across both `submit` and `take_error` so a concurrent thread finishing an earlier valid
        // `Readback` cannot drain a validation error produced by this submission before we observe it.
        let (submission, submit_err) = {
            let mut b = self.inner.batch.lock().expect("batch lock");
            let enc = self.encoder(&mut b);
            for item in prepared.iter().flatten() {
                let (idx, staging, _, len4) = item;
                let (buf, offset, _) = reads[*idx];
                enc.copy_buffer_to_buffer(buf, offset, staging.raw(), 0, *len4);
            }
            let submission = self.submit(&mut b);
            (submission, self.take_error_locked())
        };
        if let Err(e) = submit_err {
            for (_, mut staging, _, _) in prepared.into_iter().flatten() {
                staging.discard();
            }
            return Err(self.or_lost(e));
        }
        let mut out = Vec::with_capacity(reads.len());
        for item in prepared {
            match item {
                None => out.push(Readback { inner: None }),
                Some((_, staging, len, len4)) => {
                    let state: Arc<MapSignal> = Arc::default();
                    let sink = state.clone();
                    staging.raw().slice(..len4).map_async(wgpu::MapMode::Read, move |r| {
                        let mut s = sink.state.lock().expect("map state lock");
                        s.result = Some(r.map_err(|e| e.to_string()));
                        if let Some(w) = s.waker.take() {
                            w.wake();
                        }
                        sink.ready.notify_all();
                    });
                    out.push(Readback {
                        inner: Some(PendingRead {
                            ctx: self.clone(),
                            staging,
                            len,
                            len4,
                            submission: submission.clone(),
                            state,
                            retired: false,
                        }),
                    });
                }
            }
        }
        Ok(out)
    }

    /// Reads several buffer ranges back to the host in a single submission and wait (blocking; native only).
    pub fn read_batch(&self, reads: &[(&wgpu::Buffer, u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let pending = self.read_batch_deferred(reads)?;
        pending.into_iter().map(Readback::into_vec).collect()
    }

    /// Reads several buffer ranges back to the host in a single submission without blocking.
    pub async fn read_batch_async(&self, reads: &[(&wgpu::Buffer, u64, u64)]) -> Result<Vec<Vec<u8>>> {
        let pending = self.read_batch_deferred(reads)?;
        let mut out = Vec::with_capacity(pending.len());
        for r in pending {
            out.push(r.into_vec_async().await?);
        }
        Ok(out)
    }
}

/// Handle to an in-flight host readback started by [`GpuContext::read_deferred`] or
/// [`GpuContext::read_batch_deferred`].
///
/// Later GPU work can be encoded and submitted while this readback transfers in the background; collecting it
/// waits only for its own submission. Dropping a `Readback` cancels or unmaps its staging buffer cleanly.
pub struct Readback {
    inner: Option<PendingRead>,
}

impl Readback {
    /// Bytes this readback will yield.
    pub fn len(&self) -> u64 {
        self.inner.as_ref().map_or(0, |p| p.len)
    }

    /// Whether this readback is empty (`0` bytes).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Polls the device non-blockingly and reports whether the mapped bytes are ready to collect without waiting.
    ///
    /// Returns `Err(GpuError::DeviceLost(_))` if the device was lost. In the browser, checks whether the
    /// event-loop map callback has already run.
    pub fn poll_ready(&self) -> Result<bool> {
        let Some(p) = &self.inner else { return Ok(true) };
        if p.is_signaled_ok() {
            if let Some(why) = p.ctx.lost_reason() {
                return Err(GpuError::DeviceLost(why));
            }
            return Ok(true);
        }
        #[cfg(not(target_arch = "wasm32"))]
        p.ctx.poll_checked(wgpu::PollType::Poll)?;
        #[cfg(target_arch = "wasm32")]
        if let Some(why) = p.ctx.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        Ok(p.is_signaled())
    }

    /// Waits for this readback's submission (blocking; native only), passes the mapped bytes to `f`, unmaps,
    /// and reports any captured device errors.
    pub fn finish<R>(self, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        let Some(mut p) = self.inner else { return Ok(f(&[])) };
        p.wait_blocking()?;
        p.finish(f)
    }

    /// [`Readback::finish`] returning the bytes in a new `Vec<u8>`.
    pub fn into_vec(self) -> Result<Vec<u8>> {
        self.finish(<[u8]>::to_vec)
    }

    /// Awaits this readback without blocking the browser, passes the mapped bytes to `f`, unmaps, and reports
    /// any captured device errors.
    pub async fn finish_async<R>(self, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        let Some(mut p) = self.inner else { return Ok(f(&[])) };
        p.wait().await?;
        p.finish(f)
    }

    /// [`Readback::finish_async`] returning the bytes in a new `Vec<u8>`.
    pub async fn into_vec_async(self) -> Result<Vec<u8>> {
        self.finish_async(<[u8]>::to_vec).await
    }
}

/// What a readback's map callback reported, and the task to wake.
#[derive(Default)]
struct MapState {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

/// [`MapState`] plus a condition variable for blocking waiters.
#[derive(Default)]
struct MapSignal {
    state: Mutex<MapState>,
    ready: Condvar,
}

/// A readback whose copy is submitted and whose staging buffer is being mapped.
pub(crate) struct PendingRead {
    ctx: GpuContext,
    staging: GpuBuffer,
    len: u64,
    len4: u64,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    submission: Option<wgpu::SubmissionIndex>,
    state: Arc<MapSignal>,
    /// True once the map callback has completed with `Ok(())`, guaranteeing the original submission retired
    /// before `staging` may be returned to the buffer pool on drop.
    retired: bool,
}

impl PendingRead {
    #[inline]
    fn is_signaled(&self) -> bool {
        self.state.state.lock().expect("map state lock").result.is_some()
    }

    #[inline]
    fn is_signaled_ok(&self) -> bool {
        matches!(self.state.state.lock().expect("map state lock").result, Some(Ok(())))
    }

    /// Waits for the map by blocking only on this readback's submission (native targets).
    ///
    /// The callback can run on another thread: when several threads poll the device, the one that collected
    /// the callbacks may still be running them after this thread's poll returned. So wait on the condition
    /// variable, polling again now and then in case no other thread is.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait_blocking(&mut self) -> Result<()> {
        if let Some(r) = self.state.state.lock().expect("map state lock").result.take() {
            self.ctx.map_outcome(r)?;
            self.retired = true;
            return Ok(());
        }
        self.ctx.wait_submission(self.submission.as_ref(), || self.is_signaled())?;
        let mut s = self.state.state.lock().expect("map state lock");
        loop {
            if let Some(r) = s.result.take() {
                drop(s);
                self.ctx.map_outcome(r)?;
                self.retired = true;
                return Ok(());
            }
            let (guard, timeout) = self
                .state
                .ready
                .wait_timeout(s, std::time::Duration::from_millis(10))
                .expect("map state lock");
            s = guard;
            if timeout.timed_out() && s.result.is_none() {
                drop(s);
                self.ctx.wait_submission(self.submission.as_ref(), || self.is_signaled())?;
                s = self.state.state.lock().expect("map state lock");
            }
        }
    }

    /// In the browser the map callback only runs from the page's event loop, which a blocking call never gives
    /// a turn.
    #[cfg(target_arch = "wasm32")]
    fn wait_blocking(&mut self) -> Result<()> {
        match self.state.state.lock().expect("map state lock").result.take() {
            Some(r) => {
                self.ctx.map_outcome(r)?;
                self.retired = true;
                Ok(())
            }
            None => Err(GpuError::Unsupported(NO_BLOCKING)),
        }
    }

    /// Waits for the map without blocking the browser. Natively waits only on this readback's submission first,
    /// so the future is ready at once; in the browser WebGPU resolves it from the event loop.
    pub(crate) async fn wait(&mut self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        if !self.is_signaled() {
            self.ctx.wait_submission(self.submission.as_ref(), || self.is_signaled())?;
        }
        let r = MapWait { state: &self.state.state }.await;
        self.ctx.map_outcome(r)?;
        self.retired = true;
        Ok(())
    }

    /// Hands the mapped bytes to `f`, unmaps, and reports the errors wgpu captured meanwhile.
    pub(crate) fn finish<R>(self, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        let out = self
            .staging
            .raw()
            .slice(..self.len4)
            .get_mapped_range()
            .map(|view| f(&view[..self.len as usize]))
            .map_err(|e| GpuError::Map(e.to_string()));
        let ctx = self.ctx.clone();
        drop(self);
        ctx.take_error()?;
        out
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        // Unmap on every path (also a failed or abandoned wait, which cancels a pending map).
        self.staging.raw().unmap();
        if !self.retired {
            // The original submission may still be in flight in wgpu: retire the staging buffer permanently
            // instead of returning it to the reuse pool so a later readback cannot observe this submission's
            // map completion before its own copy finishes.
            self.staging.discard();
        }
    }
}

/// Resolves when a readback's map callback has run.
struct MapWait<'a> {
    state: &'a Mutex<MapState>,
}

impl Future for MapWait<'_> {
    type Output = std::result::Result<(), String>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.state.lock().expect("map state lock");
        match s.result.take() {
            Some(r) => Poll::Ready(r),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
