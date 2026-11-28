//! Automatic mixed precision for the forward pass (PyTorch's `torch.autocast`).
//!
//! While an [`autocast`] guard is alive on this thread, the matrix ops of tensors on a device whose backend
//! computes in `F16` ([`crate::backend::Backend::computes_f16`], folo-gpu with `shader-f16`) run in `F16`:
//!
//! - [`Tensor::conv2d_with`](crate::Tensor::conv2d_with) (so [`Tensor::conv2d`](crate::Tensor::conv2d) and the
//!   recording path of [`Tensor::conv2d_act`](crate::Tensor::conv2d_act)) casts an `F32` input to `F16` (a
//!   recorded cast) and uses `F16` copies of the weight and bias, but records the op against the `F32` weight
//!   and bias themselves: its backward computes the input gradient in `F16` and writes the weight gradient
//!   straight to `F32` ([`crate::backend::Backend::conv2d_backward_weight_as`]), so master weights never see an
//!   `F16` gradient;
//! - [`Tensor::matmul`](crate::Tensor::matmul) runs the forward product in `F16` on non-recording `F16` copies
//!   and records the op against the original operands: its backward computes gradients for any `F32` operand (such
//!   as `Linear` master weights) in `F32` and keeps `F16` operands in `F16`;
//! - [`Tensor::softmax`](crate::Tensor::softmax) and [`Tensor::log_softmax`](crate::Tensor::log_softmax) run an
//!   `F16` input in `F32` (a recorded cast; the result is `F32`), as `torch.autocast` does: their backward cancels.
//!
//! The matrix ops' results are `F16`; everything downstream runs in the dtype it receives (elementwise ops, pooling,
//! concatenation, the fused BatchNorm, which takes `F16` data with `F32` parameters and statistics). Other ops
//! are unchanged, and tensors on the CPU (which keeps `F16` for storage only) are never cast. Gradients flow back
//! through the recorded casts into the `F32` leaves. The guard affects recording only: backward closures keep the
//! dtypes chosen during the forward, wherever `backward()` runs.

use std::cell::Cell;
use std::marker::PhantomData;

use crate::DType;

thread_local! {
    static AUTOCAST: Cell<Option<DType>> = const { Cell::new(None) };
}

/// RAII guard returned by [`autocast`]; restores the previous state when dropped. Not `Send`: autocast is per
/// thread, like grad mode.
#[must_use = "autocast is active only while the guard is alive"]
pub struct AutocastGuard {
    prev: Option<DType>,
    _not_send: PhantomData<*const ()>,
}

/// Run the matrix ops of this thread in `F16` until the returned guard is dropped (see the module docs).
pub fn autocast() -> AutocastGuard {
    let prev = AUTOCAST.with(|a| a.replace(Some(DType::F16)));
    AutocastGuard { prev, _not_send: PhantomData }
}

/// Set this thread's autocast dtype to `dtype` until the returned guard is dropped (carries a caller's state
/// onto another thread, see [`crate::cpu::in_pool`]).
pub(crate) fn autocast_as(dtype: Option<DType>) -> AutocastGuard {
    let prev = AUTOCAST.with(|a| a.replace(dtype));
    AutocastGuard { prev, _not_send: PhantomData }
}

/// The dtype matrix ops run in under the active [`autocast`] guard, if any.
pub fn autocast_dtype() -> Option<DType> {
    AUTOCAST.with(|a| a.get())
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        AUTOCAST.with(|a| a.set(self.prev));
    }
}

/// Whether an op on `t` casts it under autocast: a guard is alive, `t` is `F32` and its backend computes in
/// `F16`.
pub(crate) fn casts(t: &crate::Tensor) -> bool {
    autocast_dtype().is_some()
        && t.dtype() == DType::F32
        && !t.device().is_cpu()
        && t.backend().is_ok_and(|b| b.computes_f16())
}
