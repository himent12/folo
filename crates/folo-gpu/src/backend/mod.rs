//! folo-core's [`Backend`] on a [`GpuContext`]: tensors on `Device::Gpu(i)` live in pooled wgpu buffers and
//! every op runs as WGSL kernels.
//!
//! [`init`] registers the shared context as `Device::Gpu(0)`; afterwards `tensor.to_device(Device::Gpu(0))`
//! uploads and every op on the result (including the gradient ops autograd records) is encoded into the
//! context's open command batch. Nothing waits for the GPU until data is read back (`to_vec`, `to_device`
//! to the CPU, `item`) or [`folo_core::backend::synchronize`] is called, so a forward + backward pass is a few
//! queue submissions with no host round trips.
//!
//! Storage is a [`GpuStorage`] (a pooled [`GpuBuffer`]) inside folo-core's `DeviceStorage`, for every dtype:
//! `u8` and `f16` elements are packed into 32-bit words, `i64` elements take two words. Views arrive as strided
//! layouts; elementwise ops, copies and matmul read them in place, other kernels get a dense copy first.
//!
//! **Half precision.** With `shader-f16` (and without `FOLO_GPU_NO_F16`) the backend computes in f16
//! ([`Backend::computes_f16`]): every forward op folo-core lets `F16` through takes f16 operands and returns
//! f16 results. Storage is f16, the arithmetic f32: GEMM and conv accumulate in f32, reductions and softmax
//! accumulate in f32, elementwise ops widen on load and round once on the store. Adaptive average pooling and the
//! arg-reductions run on an f32 copy. The gradients a mixed-precision training step needs take f16 too: conv2d
//! input and weight gradients (on tensor cores when the context has them on; the weight gradient can be written as
//! f32, [`Backend::conv2d_backward_weight_as`]), the fused BatchNorm on f16 data with f32 parameters and statistics,
//! and the max-pooling, average-pooling, and nearest-upsampling gradients. The other gradient kernels refuse f16 data.
//!
//! Differences from the CPU backend: index ops cannot report out-of-range indices without a host round trip,
//! so `index_select` / `gather` produce zeros and `index_add` / `scatter_add` skip such entries; the
//! scatter-adds use float atomics, so their summation order (and last-bit rounding) is unspecified.
//!
//! Files:
//! - `mod.rs`: [`GpuStorage`], [`GpuBackend`], [`init`] / [`register`], and helpers such as [`context_for`] and
//!   [`tensor_from_buffer`].
//! - `ops.rs`: the `Backend` trait implementation: one method per op, each calling a `crate::kernels` entry point.
//! - `encode.rs`: calls several ops share: the strided elementwise kernel, BatchNorm and conv descriptions, matmul
//!   on strided operands.
//! - `sites.rs`: the candidate-first detection hooks.
//! - `convert.rs`: folo-core descriptions and errors in the kernels' terms.
//! - `storage.rs`: a view's buffer, result storages, dense copies of strided views.
//! - `plan.rs`: [`ForwardPlans`], recorded and replayed inference calls.

mod convert;
mod encode;
mod norm;
mod ops;
mod plan;
mod sites;
mod storage;
pub use storage::poison_uninit;

use std::sync::{Arc, OnceLock};

#[cfg(doc)]
use folo_core::backend::Backend;
use folo_core::{
    CpuStorage, DType as CoreDType, Device, DeviceStorage, Error, Result, Shape, Storage, Tensor,
};

use self::convert::{Core, elem, strided_of};
#[cfg(test)]
pub(crate) use self::encode::conv_input;
pub use self::plan::{ForwardPlans, Tensors};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::kernels::strided;
use crate::pool::GpuBuffer;

/// Name used in errors.
pub const BACKEND_NAME: &str = "wgpu";

/// The buffer behind a GPU tensor's storage.
pub struct GpuStorage {
    buf: GpuBuffer,
}

impl GpuStorage {
    /// The pooled buffer. Elements are dense from byte 0 in folo-core's little-endian layout.
    pub fn buffer(&self) -> &GpuBuffer {
        &self.buf
    }
}

/// folo-core's [`Backend`] for one [`GpuContext`].
pub struct GpuBackend {
    ctx: GpuContext,
    device: Device,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend").field("device", &self.device).field("ctx", &self.ctx).finish()
    }
}

/// The outcome of the first [`init`] / [`init_async`].
static INIT: OnceLock<std::result::Result<Device, String>> = OnceLock::new();

fn register_shared(ctx: std::result::Result<GpuContext, GpuError>) -> std::result::Result<Device, String> {
    let ctx = ctx.map_err(|e| e.reason())?;
    let device = Device::Gpu(0);
    register(ctx, device).map_err(|e| e.to_string())?;
    Ok(device)
}

/// Registers [`GpuContext::shared`] as folo-core's `Device::Gpu(0)` (once per process) and returns the device.
///
/// Fails when there is no usable adapter; tensors then stay on the CPU. In the browser, call [`init_async`]
/// instead (afterwards `init` returns its result).
pub fn init() -> std::result::Result<Device, GpuError> {
    if let Some(r) = INIT.get() {
        return r.clone().map_err(GpuError::NoAdapter);
    }
    let ctx = GpuContext::shared();
    // In the browser `shared()` fails until `shared_async()` has run; that is not a result to keep, or a later
    // `init_async()` would return it too.
    #[cfg(target_arch = "wasm32")]
    if let Err(e) = ctx {
        return Err(e);
    }
    INIT.get_or_init(|| register_shared(ctx)).clone().map_err(GpuError::NoAdapter)
}

/// [`init`] without blocking, over [`GpuContext::shared_async`]: the way to get a GPU device in the browser
/// (WebGPU creates adapters and devices asynchronously).
pub async fn init_async() -> std::result::Result<Device, GpuError> {
    if INIT.get().is_none() {
        let ctx = GpuContext::shared_async().await;
        // A concurrent first call may have registered meanwhile; the first result stored wins.
        INIT.get_or_init(|| register_shared(ctx)).clone().map_err(GpuError::NoAdapter)?;
    }
    init()
}

/// Handle to an in-flight GPU-to-CPU tensor download started by [`download_deferred`] or
/// [`download_many_deferred`].
///
/// Subsequent GPU operations can be encoded and submitted before collecting the CPU tensor with
/// [`DeferredDownload::finish`] or [`DeferredDownload::finish_async`]; collecting waits only for this download's
/// submission rather than draining newer GPU work.
pub struct DeferredDownload {
    inner: DeferredDownloadInner,
}

enum DeferredDownloadInner {
    Ready(Tensor),
    Pending { readback: crate::context::Readback, shape: Shape, dtype: CoreDType },
}

impl DeferredDownload {
    /// Polls the device non-blockingly and reports whether the CPU tensor can be collected without waiting.
    pub fn poll_ready(&self) -> Result<bool> {
        match &self.inner {
            DeferredDownloadInner::Ready(_) => Ok(true),
            DeferredDownloadInner::Pending { readback, .. } => readback.poll_ready().core(),
        }
    }

    /// Waits for this download's submission (blocking; native only) and returns the owned CPU tensor.
    pub fn finish(self) -> Result<Tensor> {
        match self.inner {
            DeferredDownloadInner::Ready(t) => Ok(t),
            DeferredDownloadInner::Pending { readback, shape, dtype } => {
                let host = readback.finish(|bytes| CpuStorage::from_bytes(bytes, dtype)).core()??;
                Tensor::from_storage(Storage::Cpu(host), shape)
            }
        }
    }

    /// Awaits this download without blocking the browser and returns the owned CPU tensor.
    pub async fn finish_async(self) -> Result<Tensor> {
        match self.inner {
            DeferredDownloadInner::Ready(t) => Ok(t),
            DeferredDownloadInner::Pending { readback, shape, dtype } => {
                let host =
                    readback.finish_async(|bytes| CpuStorage::from_bytes(bytes, dtype)).await.core()??;
                Tensor::from_storage(Storage::Cpu(host), shape)
            }
        }
    }
}

/// Starts a non-blocking download of `t` into a new CPU tensor (its own storage, no gradient).
///
/// The copy (and any strided gather needed) is ordered after all work encoded so far and submitted once;
/// callers can immediately encode the next GPU step and later collect the result with
/// [`DeferredDownload::finish`] or [`DeferredDownload::finish_async`]. `ctx` must be the context of `t`'s
/// device ([`GpuContext::shared`] after [`init`] / [`init_async`]).
pub fn download_deferred(ctx: &GpuContext, t: &Tensor) -> Result<DeferredDownload> {
    let mut batch = download_many_deferred(ctx, &[t])?;
    Ok(batch.pop().expect("single download"))
}

/// Starts non-blocking downloads of all `tensors`, encoding all staging copies into a single command buffer and
/// queue submission.
///
/// Mixed CPU and GPU tensors (including strided or unaligned views) are supported: any strided or unaligned
/// views first encode gather dispatches into the open command batch (which follows
/// [`GpuContext::flush_interval`]), and then every staging copy is submitted together under one submission fence.
pub fn download_many_deferred(ctx: &GpuContext, tensors: &[&Tensor]) -> Result<Vec<DeferredDownload>> {
    if tensors.is_empty() {
        return Ok(Vec::new());
    }
    enum Slot {
        Ready(Tensor),
        Pending { read_idx: usize, shape: Shape, dtype: CoreDType },
    }
    // Validate every GPU view first so a malformed layout fails before encoding any gather kernels.
    for t in tensors {
        if t.device().is_cpu() {
            continue;
        }
        t.with_view(|v| -> Result<()> {
            match v.storage {
                Storage::Device(ds) if ds.downcast_ref::<GpuStorage>().is_some() => {}
                Storage::Device(_) => {
                    return Err(core_err(GpuError::Shape("device storage is not a wgpu buffer".into())));
                }
                Storage::Cpu(_) => unreachable!("checked above"),
            }
            if !v.layout.fits(v.storage.len()) {
                return Err(Error::InvalidShape {
                    op: "download_async",
                    shape: v.layout.shape().clone(),
                    msg: "layout reaches outside its storage".into(),
                });
            }
            Ok(())
        })?;
    }
    let mut slots = Vec::with_capacity(tensors.len());
    // Keep dense temporaries and cloned source buffers alive until `read_batch_deferred` encodes every copy.
    let mut held_dense: Vec<GpuBuffer> = Vec::new();
    let mut read_specs: Vec<(wgpu::Buffer, u64, u64)> = Vec::new();
    for t in tensors {
        if t.device().is_cpu() {
            let cpu = Tensor::from_storage(Storage::Cpu(t.to_cpu_storage()?), t.shape().clone())?;
            slots.push(Slot::Ready(cpu));
            continue;
        }
        let dtype = t.dtype();
        let n = t.numel();
        if n == 0 {
            let empty =
                Tensor::from_storage(Storage::Cpu(CpuStorage::from_bytes(&[], dtype)?), t.shape().clone())?;
            slots.push(Slot::Ready(empty));
            continue;
        }
        let eb = dtype.size_in_bytes();
        let (buf, off) = t.with_view(|v| -> Result<(wgpu::Buffer, u64)> {
            let src = match v.storage {
                Storage::Device(ds) => ds.downcast_ref::<GpuStorage>().expect("validated above").buf.raw(),
                Storage::Cpu(_) => unreachable!("checked above"),
            };
            let off = v.layout.offset() * eb;
            if v.layout.is_contiguous() && off.is_multiple_of(4) {
                return Ok((src.clone(), off as u64));
            }
            let dense = ctx.alloc(elem(dtype).buffer_bytes(n).core()?).core()?;
            strided::gather_strided(ctx, elem(dtype), src, &strided_of(v.layout), dense.raw()).core()?;
            let raw = dense.raw().clone();
            held_dense.push(dense);
            Ok((raw, 0))
        })?;
        let read_idx = read_specs.len();
        read_specs.push((buf, off, (n * eb) as u64));
        slots.push(Slot::Pending { read_idx, shape: t.shape().clone(), dtype });
    }
    let read_refs: Vec<(&wgpu::Buffer, u64, u64)> =
        read_specs.iter().map(|(b, off, len)| (b, *off, *len)).collect();
    let mut readbacks: Vec<Option<crate::context::Readback>> =
        ctx.read_batch_deferred(&read_refs).core()?.into_iter().map(Some).collect();
    drop(held_dense);
    let mut out = Vec::with_capacity(slots.len());
    for slot in slots {
        match slot {
            Slot::Ready(t) => out.push(DeferredDownload { inner: DeferredDownloadInner::Ready(t) }),
            Slot::Pending { read_idx, shape, dtype } => {
                let readback = readbacks[read_idx].take().expect("unique read_idx");
                out.push(DeferredDownload {
                    inner: DeferredDownloadInner::Pending { readback, shape, dtype },
                });
            }
        }
    }
    Ok(out)
}

/// Copies `t` to a new CPU tensor (its own storage, no gradient) without blocking.
///
/// The browser cannot wait for the GPU inside a call (WebGPU maps buffers from the event loop), so
/// [`Tensor::to_vec`] and `to_device(Device::Cpu)` fail there for GPU tensors; this is the readback to await
/// instead. The copy is ordered after all work encoded so far, which it submits. Natively it works too and
/// waits only for its own submission. `ctx` must be the context of `t`'s device ([`GpuContext::shared`] after
/// [`init`] / [`init_async`]).
pub async fn download_async(ctx: &GpuContext, t: &Tensor) -> Result<Tensor> {
    download_deferred(ctx, t)?.finish_async().await
}

/// Copies all `tensors` to new CPU tensors with all staging copies batched into one GPU submission and wait
/// (blocking; native only). See [`download_many_deferred`].
pub fn download_many(ctx: &GpuContext, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
    download_many_deferred(ctx, tensors)?.into_iter().map(DeferredDownload::finish).collect()
}

/// Copies all `tensors` to new CPU tensors with all staging copies batched into one GPU submission without
/// blocking. See [`download_many_deferred`].
pub async fn download_many_async(ctx: &GpuContext, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
    let pending = download_many_deferred(ctx, tensors)?;
    let mut out = Vec::with_capacity(pending.len());
    for d in pending {
        out.push(d.finish_async().await?);
    }
    Ok(out)
}

/// Registers a [`GpuBackend`] over `ctx` as folo-core's `device` and remembers the pair for [`context_for`], which
/// the tensor-level kernels outside the `Backend` trait (such as [`crate::detect`]) use to find the context.
pub fn register(ctx: GpuContext, device: Device) -> Result<()> {
    folo_core::backend::register_backend(device, Arc::new(GpuBackend::new(ctx.clone(), device)))?;
    let mut m = contexts().lock().expect("context registry lock");
    m.retain(|(d, _)| *d != device);
    m.push((device, ctx));
    Ok(())
}

fn contexts() -> &'static std::sync::Mutex<Vec<(Device, GpuContext)>> {
    static CONTEXTS: OnceLock<std::sync::Mutex<Vec<(Device, GpuContext)>>> = OnceLock::new();
    CONTEXTS.get_or_init(Default::default)
}

/// The context serving `device`, if it was registered through [`init`] or [`register`].
pub fn context_for(device: Device) -> Option<GpuContext> {
    contexts()
        .lock()
        .expect("context registry lock")
        .iter()
        .find(|(d, _)| *d == device)
        .map(|(_, c)| c.clone())
}

/// A folo-core tensor of `shape` over a pooled buffer holding its elements of `dtype` (dense from byte 0).
pub fn tensor_from_buffer(
    device: Device,
    dtype: CoreDType,
    shape: impl Into<Shape>,
    buf: GpuBuffer,
) -> Result<folo_core::Tensor> {
    let shape = shape.into();
    let len = shape.numel();
    let storage = Storage::Device(DeviceStorage::new(device, dtype, len, GpuStorage { buf }));
    folo_core::Tensor::from_storage(storage, shape)
}

/// The wgpu buffer behind a GPU tensor whose elements are dense from byte 0 (contiguous, offset 0), or `None`
/// for any other tensor. The handle does not keep the storage alive: hold on to the tensor while it is in use.
pub fn dense_buffer(t: &folo_core::Tensor) -> Option<wgpu::Buffer> {
    if !t.is_contiguous() || t.layout().offset() != 0 {
        return None;
    }
    t.with_view(|v| match v.storage {
        Storage::Device(ds) => ds.downcast_ref::<GpuStorage>().map(|s| s.buf.raw().clone()),
        Storage::Cpu(_) => None,
    })
}

/// The wgpu buffer behind any GPU tensor (whatever its layout), for tests that trace which operations touch it.
#[cfg(test)]
pub(crate) fn storage_buffer(t: &folo_core::Tensor) -> wgpu::Buffer {
    t.with_view(|v| match v.storage {
        Storage::Device(ds) => ds.downcast_ref::<GpuStorage>().map(|s| s.buf.raw().clone()),
        Storage::Cpu(_) => None,
    })
    .expect("a GPU tensor")
}

/// Whether a folo-core error is this backend reporting a lost device ([`GpuError::DeviceLost`]): the GPU was reset
/// or removed, so nothing on it can continue and a run should stop and resume from its last checkpoint.
pub fn is_device_lost(e: &Error) -> bool {
    let lost = GpuError::DeviceLost(String::new()).to_string();
    matches!(e, Error::Backend { backend, msg } if *backend == BACKEND_NAME && msg.starts_with(lost.as_str()))
}

/// A folo-gpu error as folo-core sees it from this backend (so [`is_device_lost`] recognises a lost device).
impl From<GpuError> for Error {
    fn from(e: GpuError) -> Self {
        core_err(e)
    }
}

fn core_err(e: GpuError) -> Error {
    Error::Backend { backend: BACKEND_NAME, msg: e.to_string() }
}

impl GpuBackend {
    /// A backend serving `device` with `ctx`. Register it with [`folo_core::backend::register_backend`].
    pub fn new(ctx: GpuContext, device: Device) -> Self {
        Self { ctx, device }
    }

    /// The context kernels run on.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }
}
