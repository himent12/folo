//! Waiting for the device: [`GpuContext::sync`], loss-aware polls ([`GpuError::DeviceLost`]) and the validation /
//! out-of-memory errors wgpu reported.

use std::sync::atomic::Ordering;

use super::{FIRST_DISPATCHES_PER_BATCH, GpuContext};
use crate::error::{GpuError, Result};

impl GpuContext {
    /// Submits the open batch and blocks until the GPU has finished all submitted work.
    ///
    /// Returns any validation / out-of-memory error wgpu reported since the last check. In the browser it
    /// cannot wait: it submits and reports the errors seen so far (an awaited readback,
    /// [`GpuContext::read_async`], is what waits there).
    pub fn sync(&self) -> Result<()> {
        self.flush();
        self.wait_idle()?;
        self.take_error()
    }

    /// Blocks until the device is idle (returns at once in the browser, where WebGPU's `poll` cannot wait): polls for
    /// up to the wait spin first, then blocks in the driver; each wait counts in [`crate::WorkStats`] as a spin wait
    /// or a blocked one.
    ///
    /// A lost device is [`GpuError::DeviceLost`], not a panic: wgpu treats a failed poll on a lost device as fatal
    /// and panics inside `Device::poll` (its message still reaches stderr through the panic hook), so the poll
    /// runs under `catch_unwind` and the loss is remembered, failing every later wait at once.
    pub(super) fn wait_idle(&self) -> Result<()> {
        let polled = self.spin().and_then(|done| {
            let counter = if done { &self.inner.spin_waits } else { &self.inner.blocked_waits };
            counter.fetch_add(1, Ordering::Relaxed);
            if done { Ok(()) } else { self.poll_checked(wgpu::PollType::wait_indefinitely()) }
        });
        // The GPU has nothing queued now: submit the head of the next pass early.
        self.inner.batch.lock().expect("batch lock").limit = FIRST_DISPATCHES_PER_BATCH;
        polled
    }

    /// Polls the device without blocking until the queue is empty or the wait spin ([`GpuContext::set_wait_spin`])
    /// has passed; `true` when the queue emptied. A blocking wait sleeps in the driver and the thread wakes up
    /// later, often on another core; a batch-1 forward waits well under a millisecond, which polling catches as
    /// soon as it ends.
    fn spin(&self) -> Result<bool> {
        #[cfg(target_arch = "wasm32")]
        return Ok(false);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let budget = self.wait_spin();
            if budget.is_zero() {
                return Ok(false);
            }
            let start = std::time::Instant::now();
            loop {
                if self.poll_status(wgpu::PollType::Poll)?.is_queue_empty() {
                    return Ok(true);
                }
                if start.elapsed() >= budget {
                    return Ok(false);
                }
                std::hint::spin_loop();
            }
        }
    }

    /// One `Device::poll` that turns a lost device into [`GpuError::DeviceLost`]: before the poll if the loss is
    /// already known, if wgpu panics inside the poll (it treats a poll on a lost device as fatal), and after a
    /// successful poll if the device-lost callback fired during it. A loss found here is remembered.
    pub(super) fn poll_checked(&self, how: wgpu::PollType) -> Result<()> {
        self.poll_status(how).map(|_| ())
    }

    /// [`GpuContext::poll_checked`] that returns what the poll reported.
    fn poll_status(&self, how: wgpu::PollType) -> Result<wgpu::PollStatus> {
        if let Some(why) = self.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.device.poll(how)));
        let res = match polled {
            Ok(r) => r.map_err(|e| GpuError::Poll(e.to_string())),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "Device::poll panicked".into());
                let mut lost = self.inner.lost.lock().unwrap_or_else(|p| p.into_inner());
                return Err(GpuError::DeviceLost(lost.get_or_insert(msg).clone()));
            }
        };
        match self.lost_reason() {
            Some(why) => Err(GpuError::DeviceLost(why)),
            None => res,
        }
    }

    /// Whether the device was lost.
    pub(crate) fn is_lost(&self) -> bool {
        self.inner.lost.lock().map_or(true, |l| l.is_some())
    }

    /// Why the device was lost (wgpu's reason and message), if it was. Every wait then fails with
    /// [`GpuError::DeviceLost`].
    pub fn lost_reason(&self) -> Option<String> {
        self.inner.lost.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `e`, or [`GpuError::DeviceLost`] if the device turns out to be lost (checked with a non-blocking poll): wgpu
    /// reports work on a lost device as ordinary validation or map errors.
    pub(super) fn or_lost(&self, e: GpuError) -> GpuError {
        match self.poll_checked(wgpu::PollType::Poll) {
            Err(lost @ GpuError::DeviceLost(_)) => lost,
            _ => e,
        }
    }

    /// The result of a readback's buffer map, loss-aware: a recorded device loss wins over both a map failure and
    /// a map that succeeded meanwhile. This is the only loss check the browser gets, where waits cannot poll and
    /// the device-lost callback runs from the event loop.
    pub(crate) fn map_outcome(&self, r: std::result::Result<(), String>) -> Result<()> {
        if let Some(why) = self.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        match r {
            Ok(()) => Ok(()),
            Err(map_err) => {
                // Drain any device error captured while submitting/mapping (e.g. an OOM that caused the staging
                // map to fail) so the failing call sees the real cause instead of leaving it to poison the next
                // unrelated readback.
                self.take_error()?;
                Err(GpuError::Map(map_err))
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn record_error_for_test(&self, msg: &str) {
        self.inner.errors.lock().expect("error lock").push(msg.into());
    }

    pub(super) fn take_error(&self) -> Result<()> {
        let mut errs = self.inner.errors.lock().expect("error lock");
        if errs.is_empty() {
            return Ok(());
        }
        // The first error is the cause; once a buffer could not be created (out of memory), every later use of
        // it fails validation too, so the rest is only counted.
        let mut msg = errs[0].clone();
        if errs.len() > 1 {
            msg.push_str(&format!("\n({} more wgpu errors followed)", errs.len() - 1));
        }
        errs.clear();
        Err(GpuError::Wgpu(msg))
    }
}
