//! Error type shared by every fallible folo-core operation.

use crate::{DType, Device, Shape};

/// Result alias used throughout folo-core.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong in a tensor operation.
///
/// Messages always name the operation and, for shape problems, both shapes involved, so that an error deep in a
/// model points straight at the offending call.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Two operands have shapes the op cannot combine (including failed broadcasts).
    #[error("{op}: shape mismatch between {lhs} and {rhs}")]
    ShapeMismatch { op: &'static str, lhs: Shape, rhs: Shape },

    /// A shape argument is invalid for this op (wrong rank, element count, zero stride, ...).
    #[error("{op}: invalid shape {shape}: {msg}")]
    InvalidShape { op: &'static str, shape: Shape, msg: String },

    /// A dimension index is outside `-rank..rank`.
    #[error("{op}: dimension {dim} is out of range for a rank-{rank} tensor")]
    DimOutOfRange { op: &'static str, dim: isize, rank: usize },

    /// An index value (in an index tensor or an argument) is out of bounds.
    #[error("{op}: index {index} is out of range for a dimension of size {size}")]
    IndexOutOfRange { op: &'static str, index: i64, size: usize },

    /// Operands have different dtypes, or the op needs a specific dtype.
    #[error("{op}: expected dtype {expected:?}, got {got:?}")]
    DTypeMismatch { op: &'static str, expected: DType, got: DType },

    /// The op is not implemented for this dtype.
    #[error("{op}: dtype {dtype:?} is not supported")]
    UnsupportedDType { op: &'static str, dtype: DType },

    /// Operands live on different devices.
    #[error("{op}: device mismatch between {lhs:?} and {rhs:?}")]
    DeviceMismatch { op: &'static str, lhs: Device, rhs: Device },

    /// The backend for this device does not implement the op (yet).
    #[error("{op} is not supported by the {backend} backend")]
    NotSupported { op: &'static str, backend: &'static str },

    /// No backend has been registered for this device (for GPUs: call `folo_gpu::init()` or similar first).
    #[error("no backend registered for device {0:?}")]
    BackendNotRegistered(Device),

    /// An argument other than a shape is invalid (negative kernel size, bad step, ...).
    #[error("{op}: {msg}")]
    InvalidArgument { op: &'static str, msg: String },

    /// Autograd misuse: backward on a non-scalar, on a tensor that does not require grad, ...
    #[error("autograd: {0}")]
    Autograd(String),

    /// A tensor saved for backward was modified in place after it was recorded.
    #[error(
        "autograd: an input of `{op}` was modified in place after the op was recorded \
         (recorded version {recorded}, current version {current}); recompute the forward pass after in-place updates"
    )]
    ModifiedInPlace { op: &'static str, recorded: u64, current: u64 },

    /// Error raised by a device backend (driver error, out of memory, ...).
    #[error("{backend} backend: {msg}")]
    Backend { backend: &'static str, msg: String },
}

impl Error {
    pub(crate) fn invalid_arg(op: &'static str, msg: impl Into<String>) -> Self {
        Error::InvalidArgument { op, msg: msg.into() }
    }

    pub(crate) fn invalid_shape(op: &'static str, shape: &Shape, msg: impl Into<String>) -> Self {
        Error::InvalidShape { op, shape: shape.clone(), msg: msg.into() }
    }
}
