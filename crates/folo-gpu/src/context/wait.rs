//! Waiting for the device: [`GpuContext::sync`], loss-aware polls ([`GpuError::DeviceLost`]) and the validation /
//! out-of-memory errors wgpu reported.

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

    /// Blocks until the device is idle (returns at once in the browser, where WebGPU's `poll` cannot wait).
    ///
    /// A lost device is [`GpuError::DeviceLost`], not a panic: wgpu treats a failed poll on a lost device as fatal
    /// and panics inside `Device::poll` (its message still reaches stderr through the panic hook), so the poll
    /// runs under `catch_unwind` and the loss is remembered, failing every later wait at once.
    pub(super) fn wait_idle(&self) -> Result<()> {
        let polled = self.poll_checked(wgpu::PollType::wait_indefinitely());
        // The GPU has nothing queued now: submit the head of the next pass early.
        self.inner.batch.lock().expect("batch lock").limit = FIRST_DISPATCHES_PER_BATCH;
        polled
    }

    /// One `Device::poll` that turns a lost device into [`GpuError::DeviceLost`]: before the poll if the loss is
    /// already known, if wgpu panics inside the poll (it treats a poll on a lost device as fatal), and after a
    /// successful poll if the device-lost callback fired during it. A loss found here is remembered.
    pub(super) fn poll_checked(&self, how: wgpu::PollType) -> Result<()> {
        if let Some(why) = self.lost_reason() {
            return Err(GpuError::DeviceLost(why));
        }
        let polled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.inner.device.poll(how)));
        let res = match polled {
            Ok(r) => r.map(|_| ()).map_err(|e| GpuError::Poll(e.to_string())),
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
        r.map_err(GpuError::Map)
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
