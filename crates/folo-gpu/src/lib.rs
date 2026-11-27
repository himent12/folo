//! FOLO's GPU compute layer on [wgpu](https://wgpu.rs): Vulkan, DX12 and Metal natively, WebGPU in the browser.
//!
//! The crate has four layers:
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
//! - **folo-core backend** — [`backend`]: [`GpuBackend`] implements `folo_core::backend::Backend` over the
//!   kernel layer; [`init`] registers it as `Device::Gpu(0)` so folo-core tensors (and autograd) run on the GPU.
//!
//! The kernel layer is deliberately independent of [`GpuTensor`] so the folo-core backend drives it with its
//! own storage type (see `docs/GPU.md`).
//!
//! # Map
//! - [`context`]: [`GpuContext`] (adapter, device, queue, command batch, forward plans, precompile, readback).
//! - [`pool`]: the size-class buffer pool ([`GpuBuffer`], [`PoolStats`]).
//! - [`memory_budget`]: live & peak memory watermarks ([`PeakMemory`]), hardware [`MemoryLimits`],
//!   [`MemoryUsage`], [`MemoryBudget`], and [`HeadroomEstimate`].
//! - `pipeline.rs` (crate-private): kernel definitions, specialisation constants, the parameter block and the WGSL
//!   prelude every kernel is compiled with; [`pipeline_cache`]: compiled pipelines kept on disk.
//! - `kernel_list.rs` (crate-private): the kernel lists [`GpuContext::precompile`] records and compiles.
//! - [`kernels`]: the Rust entry points; their WGSL is in `src/shaders/` or generated in Rust.
//! - [`backend`]: [`GpuBackend`], folo-core's `Backend` over the kernel layer.
//! - [`tensor`]: the tensor-level API, [`GpuTensor`].
//! - [`detect`]: the task-aligned assigner and detection loss on folo-core GPU tensors.
//! - [`profile`]: per-kernel timings (`FOLO_GPU_PROFILE=1`); [`dtype`], [`error`]: element types and errors.
//!
//! `crates/folo-gpu/README.md` maps an op to its shader, entry point, backend method and test.
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
pub mod distill;
pub mod dtype;
pub mod error;
mod kernel_list;
pub mod kernels;
pub mod memory_budget;
pub(crate) mod pipeline;
pub mod pipeline_cache;
pub mod pool;
pub mod profile;
pub mod tensor;

pub use backend::{
    DeferredDownload, ForwardPlans, GpuBackend, GpuStorage, Tensors, context_for, dense_buffer,
    download_async, download_deferred, download_many, download_many_async, download_many_deferred, init,
    init_async, is_device_lost, poison_uninit, register, tensor_from_buffer,
};
pub use context::{
    AdapterSummary, Capabilities, Frozen, GpuContext, GpuOptions, MAX_PLAN_BYTES, MAX_PLAN_BYTES_EACH,
    MAX_PLANS, Readback, Staged, TracedOp, WorkStats, request_tensor_cores,
};
pub use dtype::DType;
pub use error::{GpuError, Result};
pub use kernels::elementwise::{BinaryOp, UnaryOp};
pub use memory_budget::{
    AllocationKind, HeadroomBlocker, HeadroomEstimate, MemoryBudget, MemoryLimits, MemoryUsage, PeakMemory,
};
pub use pipeline_cache::{pipeline_cache_dir, save_pipeline_caches, set_pipeline_cache_dir};
pub use pool::{GpuBuffer, PoolStats};
pub use tensor::GpuTensor;
/// The write-only view of mapped staging memory that [`GpuContext::upload_with`] hands its `fill` closure.
pub use wgpu::WriteOnly;

#[cfg(test)]
mod tests;
