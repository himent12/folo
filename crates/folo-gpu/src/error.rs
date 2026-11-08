//! Error type for the GPU layer.

/// Errors returned by `folo-gpu`.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    /// No adapter matched the request (no GPU, no driver, or the `FOLO_GPU_ADAPTER` filter matched nothing).
    #[error("no GPU adapter available: {0}")]
    NoAdapter(String),
    /// The adapter refused to create a device with the features and limits FOLO asked for.
    #[error("failed to create GPU device: {0}")]
    RequestDevice(String),
    /// The adapter is below FOLO's minimum limits.
    #[error("GPU adapter does not meet FOLO's requirements: {0}")]
    Requirements(String),
    /// Mapping a staging buffer for readback failed.
    #[error("buffer map failed: {0}")]
    Map(String),
    /// Waiting on the device failed (device lost or timeout).
    #[error("device poll failed: {0}")]
    Poll(String),
    /// The device is gone (a driver reset / TDR, the GPU removed, or `destroy`); nothing on it can be used any
    /// more. Restart the process and resume from a checkpoint.
    #[error("GPU device lost: {0}")]
    DeviceLost(String),
    /// wgpu reported a validation or out-of-memory error for previously encoded work.
    #[error("wgpu error: {0}")]
    Wgpu(String),
    /// Shapes or sizes passed to a kernel are inconsistent.
    #[error("shape error: {0}")]
    Shape(String),
    /// The kernel has no variant for this dtype.
    #[error("unsupported dtype {dtype:?} for {op}")]
    DType {
        /// Operation that was asked for.
        op: &'static str,
        /// The offending dtype.
        dtype: crate::DType,
    },
    /// An optional adapter feature (for example `shader-f16`) is required but missing.
    #[error("adapter feature not available: {0}")]
    Unsupported(&'static str),
    /// A buffer or dispatch would exceed a device limit.
    #[error("device limit exceeded: {0}")]
    Limit(String),
}

impl GpuError {
    /// The message without the variant's prefix for [`GpuError::NoAdapter`], so re-wrapping a stored
    /// "no adapter" error (the shared context, `init`) does not repeat "no GPU adapter available".
    pub(crate) fn reason(&self) -> String {
        match self {
            GpuError::NoAdapter(m) => m.clone(),
            e => e.to_string(),
        }
    }
}

/// `Result` alias for `folo-gpu`.
pub type Result<T> = std::result::Result<T, GpuError>;

/// Shorthand for building a [`GpuError::Shape`].
pub(crate) fn shape_err(msg: impl Into<String>) -> GpuError {
    GpuError::Shape(msg.into())
}
