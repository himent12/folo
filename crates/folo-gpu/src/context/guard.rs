//! The optimizer step guard (docs/GPU.md, "Step guard"): mixed-precision training takes an optimizer step only
//! when the window's gradients are finite. Instead of reading that back before the step, the trainer sets a guard
//! (a device scalar that is finite exactly when the step should happen) and encodes the step; the update kernels
//! that write optimizer state (`optim`, `optim_many`, `muon_apply`) bind it and return early when it is inf or
//! NaN, so a skipped step leaves every parameter and state buffer as it was. The host reads the scalar later.

use std::cell::RefCell;
use std::sync::Arc;

use super::GpuContext;
use crate::error::Result;
use crate::pool::GpuBuffer;

thread_local! {
    /// This thread's step guards, by context (the address of its shared state). Per thread, so that two trainings
    /// (or tests) stepping on one device at once do not guard each other's updates.
    static GUARDS: RefCell<Vec<(usize, GpuBuffer)>> = const { RefCell::new(Vec::new()) };
}

/// The guard's WGSL test, for a kernel that declares the guard binding as `SKIP: array<f32>`: whether the step is
/// to be skipped (the guard scalar is inf or NaN: all exponent bits set).
pub(crate) const SKIP_FN: &str =
    "fn skip_step() -> bool { return (bitcast<u32>(SKIP[0]) & 0x7f800000u) == 0x7f800000u; }\n";

impl GpuContext {
    /// Sets the step guard to the `f32` at byte `offset` of `flag` (copied, in order with the encoded work, into
    /// a buffer of the guard's own, so the caller may drop or reuse `flag`), or clears it (`None`). While it is
    /// set, every update of `kernels::optim` (single and multi-tensor) and Muon's parameter update leave their
    /// buffers unchanged when the scalar is not finite. The guard belongs to the calling thread: only the updates
    /// this thread encodes are guarded.
    pub fn set_step_guard(&self, flag: Option<(&wgpu::Buffer, u64)>) -> Result<()> {
        let guard = match flag {
            Some((buf, offset)) => {
                let g = self.alloc(4)?;
                self.copy(buf, offset, g.raw(), 0, 4);
                Some(g)
            }
            None => None,
        };
        let key = self.guard_key();
        GUARDS.with(|gs| {
            let mut gs = gs.borrow_mut();
            gs.retain(|(k, _)| *k != key);
            gs.extend(guard.map(|g| (key, g)));
        });
        Ok(())
    }

    /// The buffer of this thread's step guard on this context, while one is set ([`GpuContext::set_step_guard`]).
    pub(crate) fn step_guard(&self) -> Option<wgpu::Buffer> {
        let key = self.guard_key();
        GUARDS.with(|gs| gs.borrow().iter().find(|(k, _)| *k == key).map(|(_, g)| g.raw().clone()))
    }

    fn guard_key(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}
