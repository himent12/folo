//! Reading buffers back to the host: a copy into a staging buffer, then a map that is waited for by blocking on
//! the device (native) or awaited (the browser).

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
        let Some(pending) = self.begin_read(buf, offset, len)? else { return Ok(f(&[])) };
        pending.wait_blocking()?;
        pending.finish(f)
    }

    /// Reads `len` bytes at `offset` of `buf` back to the host without blocking: the copy is ordered after
    /// all work encoded so far, and the future resolves once the GPU has finished it. This is the browser's
    /// readback (WebGPU maps buffers asynchronously and the page's thread must not wait); natively it waits
    /// for the device like [`GpuContext::read`].
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
        let Some(pending) = self.begin_read(buf, offset, len)? else { return Ok(f(&[])) };
        pending.wait().await?;
        pending.finish(f)
    }

    /// Encodes the copy of a readback into a staging buffer, submits it and requests the map. `None` for an
    /// empty read.
    pub(crate) fn begin_read(
        &self,
        buf: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> Result<Option<PendingRead>> {
        if len == 0 {
            return Ok(None);
        }
        self.record_taint("a readback");
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Shape(format!("read offset {offset} is not 4-byte aligned")));
        }
        let len4 = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if offset + len4 > buf.size() {
            return Err(GpuError::Shape(format!(
                "read of {len} bytes at {offset} overruns a {}-byte buffer",
                buf.size()
            )));
        }
        let staging = self.alloc_usage(len4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
        {
            let mut b = self.inner.batch.lock().expect("batch lock");
            self.encoder(&mut b).copy_buffer_to_buffer(buf, offset, staging.raw(), 0, len4);
        }
        self.flush();
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
        Ok(Some(PendingRead { ctx: self.clone(), staging, len, len4, state }))
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

/// A readback whose copy is submitted and whose staging buffer is being mapped ([`GpuContext::begin_read`]).
pub(crate) struct PendingRead {
    ctx: GpuContext,
    staging: GpuBuffer,
    len: u64,
    len4: u64,
    state: Arc<MapSignal>,
}

impl PendingRead {
    /// Waits for the map by blocking on the device (native targets).
    ///
    /// The callback can run on another thread: when several threads poll the device, the one that collected
    /// the callbacks may still be running them after this thread's poll returned. So wait on the condition
    /// variable, polling again now and then in case no other thread is.
    #[cfg(not(target_arch = "wasm32"))]
    fn wait_blocking(&self) -> Result<()> {
        self.ctx.wait_idle()?;
        let mut s = self.state.state.lock().expect("map state lock");
        loop {
            if let Some(r) = s.result.take() {
                return self.ctx.map_outcome(r);
            }
            let (guard, timeout) = self
                .state
                .ready
                .wait_timeout(s, std::time::Duration::from_millis(10))
                .expect("map state lock");
            s = guard;
            if timeout.timed_out() && s.result.is_none() {
                drop(s);
                self.ctx.wait_idle()?;
                s = self.state.state.lock().expect("map state lock");
            }
        }
    }

    /// In the browser the map callback only runs from the page's event loop, which a blocking call never gives
    /// a turn.
    #[cfg(target_arch = "wasm32")]
    fn wait_blocking(&self) -> Result<()> {
        match self.state.state.lock().expect("map state lock").result.take() {
            Some(r) => self.ctx.map_outcome(r),
            None => Err(GpuError::Unsupported(NO_BLOCKING)),
        }
    }

    /// Waits for the map without blocking the browser. Natively the device is polled to completion first,
    /// so the future is ready at once; in the browser WebGPU resolves it from the event loop.
    pub(crate) async fn wait(&self) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        self.ctx.wait_idle()?;
        let r = MapWait { state: &self.state.state }.await;
        self.ctx.map_outcome(r)
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
        let out = out?;
        ctx.take_error()?;
        Ok(out)
    }
}

impl Drop for PendingRead {
    fn drop(&mut self) {
        // Unmap on every path (also a failed or abandoned wait, which cancels a pending map) so the staging
        // buffer goes back to the pool usable.
        self.staging.raw().unmap();
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
