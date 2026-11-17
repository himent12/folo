//! The optimizer step guard ([`StepGuard`], [`crate::backend::Backend::set_step_guard`]): mixed-precision training
//! takes an optimizer step only when the window's gradients are finite, and with a guard the device decides that
//! itself, so the trainer does not wait for a readback before encoding the step.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::backend::{Backend, View};
use crate::{Device, Error, Result, Tensor};

thread_local! {
    /// The snapshot of this thread's live guard's flag, which the device's guard also binds.
    static FLAG: RefCell<Option<Tensor>> = const { RefCell::new(None) };
}

/// A step guard while it lives (see [`crate::backend::Backend::set_step_guard`]): the device's optimizer updates
/// change nothing when the flag is not finite. Dropping it clears the guard. Updates that do not go through the
/// guarded kernels (a tensor-op fallback) must ask [`step_guard_allows`] first. The guard belongs to the thread that
/// set it (updates other threads encode are not guarded), so it cannot be sent to another thread.
pub struct StepGuard {
    /// The backend that set it, which its drop clears (not whichever backend the device maps to by then).
    backend: Arc<dyn Backend>,
    /// Not `Send`: dropping it on another thread would clear that thread's guard instead.
    _thread: PhantomData<*const ()>,
}

impl StepGuard {
    /// Guards the optimizer updates on `flag`'s device until the returned guard drops: they run only if `flag` (one
    /// `F32` element; its value as of now) is finite. `Ok(None)` where the backend has no guard (the CPU), and the
    /// caller then reads the flag before stepping. One guard at a time per thread.
    pub fn new(flag: &Tensor) -> Result<Option<StepGuard>> {
        if flag.numel() != 1 || flag.dtype() != crate::DType::F32 {
            return Err(Error::InvalidArgument {
                op: "StepGuard::new",
                msg: format!("the flag is one F32 element, got {:?} {}", flag.dtype(), flag.shape()),
            });
        }
        if FLAG.with(|f| f.borrow().is_some()) {
            return Err(Error::InvalidArgument {
                op: "StepGuard::new",
                msg: "a guard is already set on this thread".into(),
            });
        }
        // One snapshot of the value as of now, which both the device's guard and the fallbacks' check
        // ([`step_guard_allows`]) read: a later write to `flag` moves neither, and they cannot disagree.
        let copy = {
            let _g = crate::no_grad();
            let c = Tensor::zeros_with(flag.shape(), crate::DType::F32, flag.device())?;
            c.copy_from(flag)?;
            c
        };
        let b = copy.backend()?;
        let r = {
            let g = copy.cell().read();
            b.set_step_guard(Some(View::new(&g, copy.layout())))
        };
        match r {
            Ok(()) => {
                FLAG.with(|f| *f.borrow_mut() = Some(copy));
                Ok(Some(StepGuard { backend: b, _thread: PhantomData }))
            }
            Err(Error::NotSupported { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        let _ = self.backend.set_step_guard(None);
        FLAG.with(|f| *f.borrow_mut() = None);
    }
}

/// Whether a step guard is set on this thread (a caller that would hand updates to other threads keeps them on
/// this one instead: the guard does not reach them).
pub fn step_guard_active() -> bool {
    FLAG.with(|f| f.borrow().is_some())
}

/// Whether an optimizer update on `device` outside the guarded kernels may run: `true` without a guard on this
/// thread for that device (the guard covers its own device only, as the guarded kernels do), else whether the
/// guard's flag is finite (a host readback, so only for rare fallbacks). A fallback that gets `false` skips its
/// update, as the guarded kernels do.
pub fn step_guard_allows(device: Device) -> Result<bool> {
    let flag = FLAG.with(|f| f.borrow().clone());
    match flag {
        Some(f) if f.device() == device => Ok(f.item::<f32>()?.is_finite()),
        _ => Ok(true),
    }
}
