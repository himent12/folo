//! Tensor ops. Each op validates its arguments, calls the device backend, and records autograd.

mod copy;
mod elementwise;
mod fused;
mod half;
mod inplace;
mod muon;
mod nn;
mod overload;
mod reduce;
mod view;
mod vision;

use std::sync::Arc;

use smallvec::SmallVec;

pub use fused::BatchNormTrain;

use crate::backend::View;
use crate::storage::StorageCell;
use crate::{DType, Error, Layout, Result, Storage, Tensor};

/// Run `f` with read views of several tensors through the given layouts. Tensors sharing a storage share one
/// read lock (re-locking the same `RwLock` on one thread could deadlock).
pub(crate) fn with_views<R, const N: usize>(
    ts: [&Tensor; N],
    layouts: [&Layout; N],
    f: impl FnOnce([View<'_>; N]) -> R,
) -> R {
    let mut cells: SmallVec<[&Arc<StorageCell>; 4]> = SmallVec::new();
    let mut idx = [0usize; N];
    for (i, t) in ts.iter().enumerate() {
        idx[i] = match cells.iter().position(|c| Arc::ptr_eq(c, t.cell())) {
            Some(j) => j,
            None => {
                cells.push(t.cell());
                cells.len() - 1
            }
        };
    }
    let guards: SmallVec<[_; 4]> = cells.iter().map(|c| c.read()).collect();
    let views = std::array::from_fn(|i| View::new(&guards[idx[i]], layouts[i]));
    f(views)
}

/// [`with_views`] for a runtime number of tensors, each with its own layout.
pub(crate) fn with_views_dyn<R>(ts: &[&Tensor], f: impl FnOnce(&[View<'_>]) -> R) -> R {
    let mut cells: SmallVec<[&Arc<StorageCell>; 4]> = SmallVec::new();
    let mut idx: SmallVec<[usize; 8]> = SmallVec::new();
    for t in ts {
        idx.push(match cells.iter().position(|c| Arc::ptr_eq(c, t.cell())) {
            Some(j) => j,
            None => {
                cells.push(t.cell());
                cells.len() - 1
            }
        });
    }
    let guards: SmallVec<[_; 4]> = cells.iter().map(|c| c.read()).collect();
    let views: SmallVec<[View<'_>; 8]> =
        ts.iter().zip(&idx).map(|(t, &i)| View::new(&guards[i], t.layout())).collect();
    f(&views)
}

/// Build the gradient list returned by backward closures.
macro_rules! grads {
    ($($g:expr),* $(,)?) => {
        smallvec::smallvec![$($g),*]
    };
}
pub(crate) use grads;

impl Tensor {
    pub(crate) fn check_same_device(&self, other: &Tensor, op: &'static str) -> Result<()> {
        if self.device() != other.device() {
            return Err(Error::DeviceMismatch { op, lhs: self.device(), rhs: other.device() });
        }
        Ok(())
    }

    pub(crate) fn check_f32(&self, op: &'static str) -> Result<()> {
        if self.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op, expected: DType::F32, got: self.dtype() });
        }
        Ok(())
    }

    /// The dtype check of compute ops: `F32` on every backend, `F16` on backends that compute in it
    /// ([`crate::backend::Backend::computes_f16`], the GPU with `shader-f16`).
    pub(crate) fn check_float(&self, op: &'static str) -> Result<()> {
        match self.dtype() {
            DType::F32 => Ok(()),
            DType::F16 if self.backend()?.computes_f16() => Ok(()),
            got => Err(Error::DTypeMismatch { op, expected: DType::F32, got }),
        }
    }

    /// [`Tensor::check_float`] for a further operand of an op whose first operand is `first`: same dtype.
    pub(crate) fn check_float_like(&self, first: &Tensor, op: &'static str) -> Result<()> {
        if self.dtype() != first.dtype() {
            return Err(Error::DTypeMismatch { op, expected: first.dtype(), got: self.dtype() });
        }
        Ok(())
    }

    pub(crate) fn check_index_dtype(&self, op: &'static str) -> Result<()> {
        match self.dtype() {
            DType::I64 | DType::U32 => Ok(()),
            got => Err(Error::DTypeMismatch { op, expected: DType::I64, got }),
        }
    }

    /// A result tensor of `shape` from a backend storage.
    pub(crate) fn wrap(storage: Storage, shape: crate::Shape, op: &'static str) -> Result<Tensor> {
        if storage.len() != shape.numel() {
            return Err(Error::Backend {
                backend: "?",
                msg: format!("{op}: backend returned {} elements for shape {shape}", storage.len()),
            });
        }
        Ok(Tensor::make(storage, shape))
    }
}
