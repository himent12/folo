//! FOLO's GPU compute layer on [wgpu](https://wgpu.rs): Vulkan, DX12 and Metal natively, WebGPU in the browser.
//!
//! The crate has three layers:
//!
//! - **Device** — [`GpuContext`] picks an adapter (high-performance by default, overridable with
//!   `FOLO_GPU_ADAPTER` / `FOLO_GPU_BACKEND`), creates the device and queue with the features and limits FOLO
//!   needs (`shader-f16` when present, the adapter's full storage-buffer and workgroup-memory limits), and owns
//!   a size-class [buffer pool](pool), a pipeline cache keyed by kernel + specialisation constants and an open
//!   command batch: kernels encode dispatches into one compute pass and are submitted together on
//!   [`GpuContext::flush`] / [`GpuContext::sync`], before uploads and on readback.
//! - **Kernels** — [`kernels`]: WGSL compute kernels with Rust entry points that take raw `wgpu::Buffer`s and
//!   explicit sizes (elementwise, broadcasting binary ops, reductions, softmax, tiled GEMM, conv2d, pooling,
//!   upsample, concat, permute, casts).
//! - **Tensors** — [`GpuTensor`]: a buffer + shape + dtype handle with methods that validate shapes, allocate
//!   outputs from the pool and call the kernel layer.
//!
//! - **folo-core backend** — [`backend`]: [`GpuBackend`] implements `folo_core::backend::Backend` over the
//!   kernel layer; [`init`] registers it as `Device::Gpu(0)` so folo-core tensors (and autograd) run on the GPU.
//!
//! The kernel layer is deliberately independent of [`GpuTensor`] so the folo-core backend drives it with its
//! own storage type (see `docs/GPU.md`).
//!
//! ```no_run
//! use folo_gpu::{GpuContext, GpuTensor};
//! let ctx = GpuContext::new()?;
//! let a = GpuTensor::from_slice(&ctx, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
//! let b = a.matmul(&a)?;
//! assert_eq!(b.to_vec()?, vec![7.0, 10.0, 15.0, 22.0]);
//! # Ok::<(), folo_gpu::GpuError>(())
//! ```

pub mod backend;
pub mod context;
pub mod detect;
pub mod dtype;
pub mod error;
mod kernel_list;
pub mod kernels;
pub(crate) mod pipeline;
pub mod pipeline_cache;
pub mod pool;
pub mod profile;
pub mod tensor;

pub use backend::{
    GpuBackend, GpuStorage, context_for, dense_buffer, download_async, init, init_async, is_device_lost,
    register, tensor_from_buffer,
};
pub use context::{
    AdapterSummary, Capabilities, GpuContext, GpuOptions, Staged, TracedOp, WorkStats, request_tensor_cores,
};
pub use dtype::DType;
pub use error::{GpuError, Result};
pub use kernels::elementwise::{BinaryOp, UnaryOp};
pub use pipeline_cache::{pipeline_cache_dir, save_pipeline_caches, set_pipeline_cache_dir};
pub use pool::{GpuBuffer, PoolStats};
pub use tensor::GpuTensor;
/// The write-only view of mapped staging memory that [`GpuContext::upload_with`] hands its `fill` closure.
pub use wgpu::WriteOnly;

#[cfg(test)]
mod tests;
