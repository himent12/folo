//! folo-core's op descriptions in the kernels' terms: dtypes, layouts, pooling windows, fused activations and
//! optimizer steps, and folo-gpu errors as folo-core errors.

use folo_core::backend::{OptimStep, Pool2dParams, UnaryOp, View};
use folo_core::{DType as CoreDType, Error, Layout, Result, Shape};

use super::core_err;
use crate::error::GpuError;
use crate::kernels::conv::Conv2dDesc;
use crate::kernels::elementwise::UnaryOp as GpuUnary;
use crate::kernels::norm::BnAct;
use crate::kernels::optim::OptimUpdate;
use crate::kernels::pool_grad::Pool2d;
use crate::kernels::strided::{Elem, Strided};

pub(super) trait Core<T> {
    fn core(self) -> Result<T>;
}

impl<T> Core<T> for std::result::Result<T, GpuError> {
    fn core(self) -> Result<T> {
        self.map_err(core_err)
    }
}

pub(super) fn elem(d: CoreDType) -> Elem {
    match d {
        CoreDType::F32 => Elem::F32,
        CoreDType::F16 => Elem::F16,
        CoreDType::U8 => Elem::U8,
        CoreDType::U32 => Elem::U32,
        CoreDType::I64 => Elem::I64,
    }
}

/// The kernels' storage type of a float dtype (f16 stays f16, everything else is f32).
pub(super) fn kdt(d: CoreDType) -> crate::DType {
    if d == CoreDType::F16 { crate::DType::F16 } else { crate::DType::F32 }
}

/// Refuses f16 data in ops with f32 kernels only (the gradients mixed-precision training does not use).
pub(super) fn f32_only(v: &View<'_>, op: &'static str) -> Result<()> {
    match v.storage.dtype() {
        CoreDType::F16 => Err(Error::UnsupportedDType { op, dtype: CoreDType::F16 }),
        _ => Ok(()),
    }
}

pub(super) fn strided_of(l: &Layout) -> Strided {
    Strided { offset: l.offset(), dims: l.dims().to_vec(), strides: l.strides().to_vec() }
}

pub(super) fn split3(dims: &[usize], dim: usize) -> (usize, usize, usize) {
    (dims[..dim].iter().product(), dims[dim], dims[dim + 1..].iter().product())
}

pub(super) fn nchw(s: &Shape) -> (usize, usize, usize, usize) {
    let d = s.dims();
    (d[0], d[1], d[2], d[3])
}

pub(super) fn pool2d(p: &Pool2dParams) -> Pool2d {
    Pool2d {
        kernel: (p.kernel[0], p.kernel[1]),
        stride: (p.stride[0], p.stride[1]),
        padding: (p.padding[0], p.padding[1]),
    }
}

/// A conv problem as a profile label: channels, kernel, stride, groups, input size, dtype.
pub(super) fn conv_label(d: &Conv2dDesc) -> String {
    format!(
        "{}->{} k{} s{} g{} {}x{}x{} {:?}",
        d.c_in, d.c_out, d.win.kernel.0, d.win.stride.0, d.groups, d.n, d.h, d.w, d.dtype
    )
}

/// The fused update kernel's form of an optimizer step, if it has one.
pub(super) fn optim_update(step: &OptimStep) -> Option<OptimUpdate> {
    Some(match *step {
        OptimStep::Sgd { lr, momentum, weight_decay, nesterov } => {
            OptimUpdate::Sgd { lr, momentum, weight_decay, nesterov }
        }
        OptimStep::SgdDecoupled { lr, momentum, wd_step, nesterov, cautious } => {
            OptimUpdate::SgdDecoupled { lr, momentum, wd_step, nesterov, cautious }
        }
        OptimStep::AdamW {
            decay,
            beta1,
            one_minus_beta1,
            beta2,
            one_minus_beta2,
            step_size,
            bc2_sqrt,
            eps,
        } => OptimUpdate::AdamW {
            decay,
            beta1,
            one_minus_beta1,
            beta2,
            one_minus_beta2,
            step_size,
            bc2_sqrt,
            eps,
        },
        OptimStep::Axpby { a, b } => OptimUpdate::Axpby { a, b },
        _ => return None,
    })
}

/// The conv epilogue's form of a fused activation, if it has one.
pub(super) fn conv_act(act: Option<UnaryOp>) -> Option<GpuUnary> {
    Some(match act {
        None => GpuUnary::Identity,
        Some(UnaryOp::Silu) => GpuUnary::Silu,
        Some(UnaryOp::Relu) => GpuUnary::Relu,
        Some(UnaryOp::Sigmoid) => GpuUnary::Sigmoid,
        Some(UnaryOp::Tanh) => GpuUnary::Tanh,
        Some(UnaryOp::Hardswish) => GpuUnary::Hardswish,
        Some(UnaryOp::LeakyRelu(a)) => GpuUnary::LeakyRelu(a),
        Some(_) => return None,
    })
}

/// The BatchNorm kernels' form of a fused activation, if they have one.
pub(super) fn bn_act(act: Option<UnaryOp>) -> Option<BnAct> {
    Some(match act {
        None => BnAct::Identity,
        Some(UnaryOp::Silu) => BnAct::Silu,
        Some(UnaryOp::Relu) => BnAct::Relu,
        Some(_) => return None,
    })
}
