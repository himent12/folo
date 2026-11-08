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
//! Differences from the CPU backend: index ops cannot report out-of-range indices without a host round trip,
//! so `index_select` / `gather` produce zeros and `index_add` / `scatter_add` skip such entries; the
//! scatter-adds use float atomics, so their summation order (and last-bit rounding) is unspecified.

use std::sync::{Arc, OnceLock};

use folo_core::backend::{
    Backend, BinaryOp, Conv2dParams, OptimStep, Pool2dParams, ReduceOp, RunningUpdate, UnaryOp, View,
};
use folo_core::{
    CpuStorage, DType as CoreDType, Device, DeviceStorage, Error, Layout, Result, Shape, Storage, Tensor,
};

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::kernels::conv::{self, Conv2dDesc};
use crate::kernels::elementwise::UnaryOp as GpuUnary;
use crate::kernels::ew::{self, EwInput, EwOp};
use crate::kernels::gemm::{self, BatchStrides, GemmDesc};
use crate::kernels::layout;
use crate::kernels::norm::{self, BnAct, BnDesc, BnRunning};
use crate::kernels::optim::{self, OptimUpdate};
use crate::kernels::pool_grad::{self, Pool2d};
use crate::kernels::pooling::Window2d;
use crate::kernels::reduce::{self, ReduceOp as GpuReduce};
use crate::kernels::strided::{self, Elem, Indices, Strided};
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

/// Copies `t` to a new CPU tensor (its own storage, no gradient) without blocking.
///
/// The browser cannot wait for the GPU inside a call (WebGPU maps buffers from the event loop), so
/// [`Tensor::to_vec`] and `to_device(Device::Cpu)` fail there for GPU tensors; this is the readback to await
/// instead. The copy is ordered after all work encoded so far, which it submits. Natively it works too and
/// waits like a blocking download. `ctx` must be the context of `t`'s device ([`GpuContext::shared`] after
/// [`init`] / [`init_async`]).
pub async fn download_async(ctx: &GpuContext, t: &Tensor) -> Result<Tensor> {
    if t.device().is_cpu() {
        return Tensor::from_storage(Storage::Cpu(t.to_cpu_storage()?), t.shape().clone());
    }
    let dtype = t.dtype();
    let n = t.numel();
    let eb = dtype.size_in_bytes();
    let pending = t.with_view(|v| -> Result<_> {
        let buf = match v.storage {
            Storage::Device(ds) => ds
                .downcast_ref::<GpuStorage>()
                .map(|s| s.buf.raw())
                .ok_or_else(|| core_err(GpuError::Shape("device storage is not a wgpu buffer".into())))?,
            Storage::Cpu(_) => unreachable!("checked above"),
        };
        if !v.layout.fits(v.storage.len()) {
            return Err(Error::InvalidShape {
                op: "download_async",
                shape: v.layout.shape().clone(),
                msg: "layout reaches outside its storage".into(),
            });
        }
        let off = v.layout.offset() * eb;
        if v.layout.is_contiguous() && off.is_multiple_of(4) {
            return ctx.begin_read(buf, off as u64, (n * eb) as u64).core();
        }
        // Dense copy first; the buffer may go back to the pool at once, the readback is ordered after it.
        let dense = ctx.alloc(elem(dtype).buffer_bytes(n).core()?).core()?;
        strided::gather_strided(ctx, elem(dtype), buf, &strided_of(v.layout), dense.raw()).core()?;
        ctx.begin_read(dense.raw(), 0, (n * eb) as u64).core()
    })?;
    let host = match pending {
        None => CpuStorage::from_bytes(&[], dtype)?,
        Some(p) => {
            p.wait().await.core()?;
            p.finish(|bytes| CpuStorage::from_bytes(bytes, dtype)).core()??
        }
    };
    Tensor::from_storage(Storage::Cpu(host), t.shape().clone())
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

trait Core<T> {
    fn core(self) -> Result<T>;
}

impl<T> Core<T> for std::result::Result<T, GpuError> {
    fn core(self) -> Result<T> {
        self.map_err(core_err)
    }
}

fn elem(d: CoreDType) -> Elem {
    match d {
        CoreDType::F32 => Elem::F32,
        CoreDType::F16 => Elem::F16,
        CoreDType::U8 => Elem::U8,
        CoreDType::U32 => Elem::U32,
        CoreDType::I64 => Elem::I64,
    }
}

fn strided_of(l: &Layout) -> Strided {
    Strided { offset: l.offset(), dims: l.dims().to_vec(), strides: l.strides().to_vec() }
}

fn split3(dims: &[usize], dim: usize) -> (usize, usize, usize) {
    (dims[..dim].iter().product(), dims[dim], dims[dim + 1..].iter().product())
}

fn nchw(s: &Shape) -> (usize, usize, usize, usize) {
    let d = s.dims();
    (d[0], d[1], d[2], d[3])
}

fn pool2d(p: &Pool2dParams) -> Pool2d {
    Pool2d {
        kernel: (p.kernel[0], p.kernel[1]),
        stride: (p.stride[0], p.stride[1]),
        padding: (p.padding[0], p.padding[1]),
    }
}

/// The conv epilogue's form of a fused activation, if it has one.
fn conv_act(act: Option<UnaryOp>) -> Option<GpuUnary> {
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
fn bn_act(act: Option<UnaryOp>) -> Option<BnAct> {
    Some(match act {
        None => BnAct::Identity,
        Some(UnaryOp::Silu) => BnAct::Silu,
        Some(UnaryOp::Relu) => BnAct::Relu,
        Some(_) => return None,
    })
}

/// A dense (contiguous, offset 0) f32-or-other buffer for a view: borrowed when it already is, else a copy.
enum Dense<'a> {
    Borrowed(&'a wgpu::Buffer),
    Owned(GpuBuffer),
}

impl Dense<'_> {
    fn raw(&self) -> &wgpu::Buffer {
        match self {
            Dense::Borrowed(b) => b,
            Dense::Owned(b) => b.raw(),
        }
    }
}

/// `(transposed, leading dimension)` of a `rows × cols` matrix with strides `(rs, cs)` as the GEMM can
/// address it, if it can.
fn mat_operand(rows: usize, cols: usize, rs: usize, cs: usize) -> Option<(bool, usize)> {
    let (rs1, cs1) = (if rows == 1 { cols } else { rs }, if cols == 1 { 1 } else { cs });
    if cs1 == 1 && rs1 >= cols {
        return Some((false, rs1));
    }
    let (rs2, cs2) = (if rows == 1 { 1 } else { rs }, if cols == 1 { rows } else { cs });
    if rs2 == 1 && cs2 >= rows {
        return Some((true, cs2));
    }
    None
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

    fn buf<'a>(&self, v: &View<'a>) -> Result<&'a wgpu::Buffer> {
        match v.storage {
            Storage::Device(ds) if ds.device() == self.device => ds
                .downcast_ref::<GpuStorage>()
                .map(|s| s.buf.raw())
                .ok_or_else(|| core_err(GpuError::Shape("device storage is not a wgpu buffer".into()))),
            s => Err(Error::DeviceMismatch { op: "wgpu", lhs: self.device, rhs: s.device() }),
        }
    }

    /// The wgpu buffer of a whole storage on this device.
    fn storage_buf<'a>(&self, s: &'a Storage) -> Result<&'a wgpu::Buffer> {
        match s {
            Storage::Device(ds) if ds.device() == self.device => ds
                .downcast_ref::<GpuStorage>()
                .map(|s| s.buf.raw())
                .ok_or_else(|| core_err(GpuError::Shape("device storage is not a wgpu buffer".into()))),
            s => Err(Error::DeviceMismatch { op: "wgpu", lhs: self.device, rhs: s.device() }),
        }
    }

    fn bn_desc(&self, x: &Shape, act: Option<UnaryOp>, op: &'static str) -> Result<BnDesc> {
        let act = bn_act(act).ok_or(Error::NotSupported { op, backend: BACKEND_NAME })?;
        let (n, c, h, w) = nchw(x);
        Ok(BnDesc { n, c, hw: h * w, act })
    }

    fn alloc(&self, dtype: CoreDType, len: usize) -> Result<GpuBuffer> {
        self.ctx.alloc(elem(dtype).buffer_bytes(len).core()?).core()
    }

    fn wrap(&self, dtype: CoreDType, len: usize, buf: GpuBuffer) -> Storage {
        Storage::Device(DeviceStorage::new(self.device, dtype, len, GpuStorage { buf }))
    }

    fn dense<'a>(&self, v: &View<'a>) -> Result<Dense<'a>> {
        let b = self.buf(v)?;
        if v.layout.is_contiguous() && v.layout.offset() == 0 {
            return Ok(Dense::Borrowed(b));
        }
        let dtype = v.storage.dtype();
        let out = self.alloc(dtype, v.layout.numel())?;
        strided::gather_strided(&self.ctx, elem(dtype), b, &strided_of(v.layout), out.raw()).core()?;
        Ok(Dense::Owned(out))
    }

    fn check_fits(&self, v: &View<'_>, op: &'static str) -> Result<()> {
        if !v.layout.fits(v.storage.len()) {
            return Err(Error::InvalidShape {
                op,
                shape: v.layout.shape().clone(),
                msg: "layout reaches outside its storage".into(),
            });
        }
        Ok(())
    }

    fn ew(&self, op: EwOp, out_shape: &Shape, inputs: &[View<'_>]) -> Result<Storage> {
        let n = out_shape.numel();
        let out = self.alloc(CoreDType::F32, n)?;
        let views: Vec<Strided> = inputs.iter().map(|v| strided_of(v.layout)).collect();
        let bufs = inputs.iter().map(|v| self.buf(v)).collect::<Result<Vec<_>>>()?;
        let ins: Vec<EwInput<'_>> =
            bufs.iter().zip(&views).map(|(b, v)| EwInput { buf: b, view: v }).collect();
        ew::elementwise(&self.ctx, op, out_shape.dims(), &ins, out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, n, out))
    }

    fn indices<'a>(&self, v: &View<'a>, op: &'static str) -> Result<(Dense<'a>, bool)> {
        let i64 = match v.storage.dtype() {
            CoreDType::I64 => true,
            CoreDType::U32 => false,
            got => return Err(Error::DTypeMismatch { op, expected: CoreDType::I64, got }),
        };
        Ok((self.dense(v)?, i64))
    }

    fn conv_desc(&self, x: &Shape, w: &Shape, p: &Conv2dParams) -> Conv2dDesc {
        let (n, c, h, wi) = nchw(x);
        let (o, _, kh, kw) = nchw(w);
        Conv2dDesc {
            n,
            c_in: c,
            h,
            w: wi,
            c_out: o,
            groups: p.groups,
            win: Window2d {
                kernel: (kh, kw),
                stride: (p.stride[0], p.stride[1]),
                padding: (p.padding[0], p.padding[1]),
                dilation: (p.dilation[0], p.dilation[1]),
            },
            act: GpuUnary::Identity,
            dtype: crate::DType::F32,
        }
    }

    /// Encodes `a @ b` for strided views, or returns `None` when an operand's layout is not addressable
    /// (the caller densifies and retries).
    fn try_matmul(&self, a: &View<'_>, b: &View<'_>, out: &wgpu::Buffer) -> Result<Option<()>> {
        let (la, lb) = (a.layout, b.layout);
        let r = la.rank();
        let (m, k, n) = (la.dims()[r - 2], la.dims()[r - 1], lb.dims()[r - 1]);
        let Some((ta, lda)) = mat_operand(m, k, la.strides()[r - 2], la.strides()[r - 1]) else {
            return Ok(None);
        };
        let Some((tb, ldb)) = mat_operand(k, n, lb.strides()[r - 2], lb.strides()[r - 1]) else {
            return Ok(None);
        };
        let bdims = &la.dims()[..r - 2];
        let mut sc = vec![m * n; bdims.len()];
        for i in (0..bdims.len().saturating_sub(1)).rev() {
            sc[i] = sc[i + 1] * bdims[i + 1];
        }
        let (cd, cs) = strided::coalesce(bdims, &[&la.strides()[..r - 2], &lb.strides()[..r - 2], &sc]);
        if cd.len() > 2 {
            return Ok(None);
        }
        let get = |v: &Vec<usize>, i: usize| v.get(i).copied().unwrap_or(0);
        let (outer, inner) = match cd.len() {
            0 => (1, 1),
            1 => (cd[0], 1),
            _ => (cd[0], cd[1]),
        };
        let mut g = GemmDesc::new(m, n, k, ta, tb, outer);
        g.lda = lda;
        g.ldb = ldb;
        g.ldc = n;
        g.batch = (outer, inner);
        g.a = BatchStrides { offset: la.offset(), outer: get(&cs[0], 0), inner: get(&cs[0], 1) };
        g.b = BatchStrides { offset: lb.offset(), outer: get(&cs[1], 0), inner: get(&cs[1], 1) };
        g.c = BatchStrides { offset: 0, outer: get(&cs[2], 0), inner: get(&cs[2], 1) };
        // A shared B and a batch of A rows that form one strided matrix: one taller GEMM.
        if cd.len() == 1 && g.b.outer == 0 && !ta && g.a.outer == m * lda {
            g.m = outer * m;
            g.batch = (1, 1);
            g.a.outer = 0;
            g.c.outer = 0;
        }
        if g.batch.0 * g.batch.1 > 65535 {
            return Ok(None);
        }
        gemm::gemm(&self.ctx, &g, self.buf(a)?, self.buf(b)?, None, out).core()?;
        Ok(Some(()))
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.sync().core()
    }

    fn full(&self, _device: Device, dtype: CoreDType, len: usize, value: f64) -> Result<Storage> {
        let buf = self.alloc(dtype, len)?;
        let (p0, p1) = match dtype {
            CoreDType::F32 => ((value as f32).to_bits(), (value as f32).to_bits()),
            CoreDType::F16 => {
                let b = half::f16::from_f64(value).to_bits() as u32;
                (b | b << 16, b | b << 16)
            }
            CoreDType::U8 => {
                let b = value as u8 as u32;
                (b * 0x0101_0101, b * 0x0101_0101)
            }
            CoreDType::U32 => (value as u32, value as u32),
            CoreDType::I64 => {
                let v = value as i64 as u64;
                (v as u32, (v >> 32) as u32)
            }
        };
        let words = (elem(dtype).buffer_bytes(len).core()? / 4) as usize;
        strided::fill_words(&self.ctx, buf.raw(), words, p0, p1).core()?;
        Ok(self.wrap(dtype, len, buf))
    }

    fn upload(&self, _device: Device, data: &CpuStorage) -> Result<Storage> {
        let bytes = data.as_bytes();
        let buf = self.alloc(data.dtype(), data.len())?;
        self.ctx.write(&buf, 0, bytes).core()?;
        Ok(self.wrap(data.dtype(), data.len(), buf))
    }

    fn download(&self, src: View<'_>) -> Result<CpuStorage> {
        self.check_fits(&src, "download")?;
        let dtype = src.storage.dtype();
        let n = src.layout.numel();
        let eb = dtype.size_in_bytes();
        let b = self.buf(&src)?;
        let off = src.layout.offset() * eb;
        let decode = |bytes: &[u8]| CpuStorage::from_bytes(bytes, dtype);
        if src.layout.is_contiguous() && off.is_multiple_of(4) {
            self.ctx.read_with(b, off as u64, (n * eb) as u64, decode).core()?
        } else {
            let d = self.dense(&src)?;
            self.ctx.read_with(d.raw(), 0, (n * eb) as u64, decode).core()?
        }
    }

    fn copy_strided(&self, src: View<'_>) -> Result<Storage> {
        self.check_fits(&src, "copy_strided")?;
        let dtype = src.storage.dtype();
        let n = src.layout.numel();
        let out = self.alloc(dtype, n)?;
        strided::gather_strided(&self.ctx, elem(dtype), self.buf(&src)?, &strided_of(src.layout), out.raw())
            .core()?;
        Ok(self.wrap(dtype, n, out))
    }

    fn copy_into(&self, dst: &mut Storage, dst_layout: &Layout, src: View<'_>) -> Result<()> {
        if dst_layout.shape() != src.shape() {
            return Err(Error::ShapeMismatch {
                op: "copy_into",
                lhs: dst_layout.shape().clone(),
                rhs: src.shape().clone(),
            });
        }
        if dst.dtype() != src.storage.dtype() {
            return Err(Error::DTypeMismatch {
                op: "copy_into",
                expected: dst.dtype(),
                got: src.storage.dtype(),
            });
        }
        if !dst_layout.fits(dst.len()) || !src.layout.fits(src.storage.len()) {
            return Err(Error::InvalidShape {
                op: "copy_into",
                shape: dst_layout.shape().clone(),
                msg: "layout reaches outside its storage".into(),
            });
        }
        if !dst_layout.is_non_overlapping() {
            return Err(Error::InvalidShape {
                op: "copy_into",
                shape: dst_layout.shape().clone(),
                msg: "destination layout overlaps itself".into(),
            });
        }
        let dtype = dst.dtype();
        let db = self.buf(&View::new(dst, dst_layout))?;
        let sb = self.buf(&src)?;
        strided::scatter_strided(
            &self.ctx,
            elem(dtype),
            sb,
            &strided_of(src.layout),
            db,
            &strided_of(dst_layout),
        )
        .core()
    }

    fn cast(&self, src: View<'_>, dtype: CoreDType) -> Result<Storage> {
        self.check_fits(&src, "cast")?;
        let from = src.storage.dtype();
        if from == dtype {
            return self.copy_strided(src);
        }
        let n = src.layout.numel();
        let out = self.alloc(dtype, n)?;
        let (buf, offset) = if src.layout.is_contiguous() {
            (Dense::Borrowed(self.buf(&src)?), src.layout.offset())
        } else {
            (self.dense(&src)?, 0)
        };
        strided::cast(&self.ctx, buf.raw(), elem(from), offset, out.raw(), elem(dtype), n).core()?;
        Ok(self.wrap(dtype, n, out))
    }

    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        self.ew(EwOp::Unary(op), x.shape(), &[x])
    }

    fn unary_backward(&self, op: UnaryOp, x: View<'_>, grad: View<'_>) -> Result<Storage> {
        self.ew(EwOp::UnaryBackward(op), x.shape(), &[x, grad])
    }

    fn binary(&self, op: BinaryOp, a: View<'_>, b: View<'_>) -> Result<Storage> {
        self.ew(EwOp::Binary(op), a.shape(), &[a, b])
    }

    fn where_cond(&self, cond: View<'_>, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let cond_u8 = match cond.storage.dtype() {
            CoreDType::F32 => false,
            CoreDType::U8 => true,
            dtype => return Err(Error::UnsupportedDType { op: "where_cond", dtype }),
        };
        self.ew(EwOp::Where { cond_u8 }, a.shape(), &[cond, a, b])
    }

    fn reduce(&self, op: ReduceOp, x: View<'_>, dim: usize) -> Result<Storage> {
        let (outer, r, inner) = split3(x.shape().dims(), dim);
        let n = outer * inner;
        let arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
        let dtype = if arg { CoreDType::I64 } else { CoreDType::F32 };
        let out = self.alloc(dtype, n)?;
        if n > 0 {
            let xd = self.dense(&x)?;
            let gop = match op {
                ReduceOp::Sum => Some(GpuReduce::Sum),
                ReduceOp::Max => Some(GpuReduce::Max),
                ReduceOp::Min => Some(GpuReduce::Min),
                _ => None,
            };
            match gop {
                Some(g) => reduce::reduce_split(&self.ctx, g, xd.raw(), out.raw(), outer, r, inner).core()?,
                None => reduce::arg_reduce(
                    &self.ctx,
                    op == ReduceOp::ArgMax,
                    xd.raw(),
                    out.raw(),
                    outer,
                    r,
                    inner,
                )
                .core()?,
            }
        }
        Ok(self.wrap(dtype, n, out))
    }

    fn softmax(&self, x: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        let n = x.layout.numel();
        let out = self.alloc(CoreDType::F32, n)?;
        if n > 0 {
            let xd = self.dense(&x)?;
            reduce::softmax_dim(&self.ctx, xd.raw(), out.raw(), split3(x.shape().dims(), dim), log).core()?;
        }
        Ok(self.wrap(CoreDType::F32, n, out))
    }

    fn matmul(&self, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let r = a.layout.rank();
        let (m, k, n) = (a.layout.dims()[r - 2], a.layout.dims()[r - 1], b.layout.dims()[r - 1]);
        let batch: usize = a.layout.dims()[..r - 2].iter().product();
        let len = batch * m * n;
        let out = self.alloc(CoreDType::F32, len)?;
        if len == 0 {
            return Ok(self.wrap(CoreDType::F32, len, out));
        }
        if k == 0 {
            self.ctx.clear(out.raw(), crate::DType::F32.bytes(len));
            return Ok(self.wrap(CoreDType::F32, len, out));
        }
        self.check_fits(&a, "matmul")?;
        self.check_fits(&b, "matmul")?;
        if self.try_matmul(&a, &b, out.raw())?.is_none() {
            // Make both operands dense and try again (always addressable then).
            let (sa, sb) = (self.copy_strided(a)?, self.copy_strided(b)?);
            let (la, lb) = (Layout::contiguous(a.shape().clone()), Layout::contiguous(b.shape().clone()));
            self.try_matmul(&View::new(&sa, &la), &View::new(&sb, &lb), out.raw())?
                .ok_or_else(|| core_err(GpuError::Limit("matmul batch does not fit one dispatch".into())))?;
        }
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn conv2d(&self, x: View<'_>, w: View<'_>, bias: Option<View<'_>>, p: &Conv2dParams) -> Result<Storage> {
        let d = self.conv_desc(x.shape(), w.shape(), p);
        let (oh, ow) = d.output().core()?;
        let len = d.n * d.c_out * oh * ow;
        let out = self.alloc(CoreDType::F32, len)?;
        let (xd, wd) = (self.dense(&x)?, self.dense(&w)?);
        let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
        conv::conv2d(&self.ctx, &d, xd.raw(), wd.raw(), bd.as_ref().map(|b| b.raw()), out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn conv2d_backward_input(
        &self,
        grad: View<'_>,
        w: View<'_>,
        input_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let d = self.conv_desc(input_shape, w.shape(), p);
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let (gd, wd) = (self.dense(&grad)?, self.dense(&w)?);
        conv::conv2d_backward_input(&self.ctx, &d, gd.raw(), wd.raw(), out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn conv2d_backward_weight(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let d = self.conv_desc(x.shape(), weight_shape, p);
        let len = weight_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let (xd, gd) = (self.dense(&x)?, self.dense(&grad)?);
        conv::conv2d_backward_weight(&self.ctx, &d, xd.raw(), gd.raw(), out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn max_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<(Storage, Storage)> {
        let (n, c, h, w) = nchw(x.shape());
        let win = pool2d(p);
        let (oh, ow) = win.output(h, w).core()?;
        let len = n * c * oh * ow;
        let (vals, idx) = (self.alloc(CoreDType::F32, len)?, self.alloc(CoreDType::I64, len)?);
        let xd = self.dense(&x)?;
        pool_grad::max_pool2d_indices(&self.ctx, xd.raw(), vals.raw(), idx.raw(), (n * c, h, w), &win)
            .core()?;
        Ok((self.wrap(CoreDType::F32, len, vals), self.wrap(CoreDType::I64, len, idx)))
    }

    fn max_pool2d_backward(
        &self,
        grad: View<'_>,
        indices: View<'_>,
        input_shape: &Shape,
        p: &Pool2dParams,
    ) -> Result<Storage> {
        if indices.storage.dtype() != CoreDType::I64 {
            return Err(Error::DTypeMismatch {
                op: "max_pool2d_backward",
                expected: CoreDType::I64,
                got: indices.storage.dtype(),
            });
        }
        let (n, c, h, w) = nchw(input_shape);
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let (gd, id) = (self.dense(&grad)?, self.dense(&indices)?);
        pool_grad::max_pool2d_backward(&self.ctx, gd.raw(), id.raw(), out.raw(), (n * c, h, w), &pool2d(p))
            .core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn avg_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        let (n, c, h, w) = nchw(x.shape());
        let win = pool2d(p);
        let (oh, ow) = win.output(h, w).core()?;
        let len = n * c * oh * ow;
        let out = self.alloc(CoreDType::F32, len)?;
        let xd = self.dense(&x)?;
        pool_grad::avg_pool2d(&self.ctx, xd.raw(), out.raw(), (n * c, h, w), &win).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape, p: &Pool2dParams) -> Result<Storage> {
        let (n, c, h, w) = nchw(input_shape);
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let gd = self.dense(&grad)?;
        pool_grad::avg_pool2d_backward(&self.ctx, gd.raw(), out.raw(), (n * c, h, w), &pool2d(p)).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn adaptive_avg_pool2d(&self, x: View<'_>, out_hw: [usize; 2]) -> Result<Storage> {
        let (n, c, h, w) = nchw(x.shape());
        let len = n * c * out_hw[0] * out_hw[1];
        let out = self.alloc(CoreDType::F32, len)?;
        let xd = self.dense(&x)?;
        pool_grad::adaptive_avg_pool2d(&self.ctx, xd.raw(), out.raw(), (n * c, h, w), (out_hw[0], out_hw[1]))
            .core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn adaptive_avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape) -> Result<Storage> {
        let (n, c, h, w) = nchw(input_shape);
        let (_, _, oh, ow) = nchw(grad.shape());
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let gd = self.dense(&grad)?;
        pool_grad::adaptive_avg_pool2d_backward(&self.ctx, gd.raw(), out.raw(), (n * c, h, w), (oh, ow))
            .core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn upsample_nearest2d(&self, x: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        let (n, c, h, w) = nchw(x.shape());
        let (oh, ow) = (h * scale[0], w * scale[1]);
        let len = n * c * oh * ow;
        let out = self.alloc(CoreDType::F32, len)?;
        if len > 0 {
            let xd = self.dense(&x)?;
            layout::upsample_nearest2d(
                &self.ctx,
                xd.raw(),
                out.raw(),
                n * c,
                h,
                w,
                oh,
                ow,
                crate::DType::F32,
            )
            .core()?;
        }
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn upsample_nearest2d_backward(&self, grad: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        let (n, c, oh, ow) = nchw(grad.shape());
        let (h, w) = (oh / scale[0], ow / scale[1]);
        let len = n * c * h * w;
        let out = self.alloc(CoreDType::F32, len)?;
        let gd = self.dense(&grad)?;
        pool_grad::upsample_nearest2d_backward(
            &self.ctx,
            gd.raw(),
            out.raw(),
            (n * c, h, w),
            (scale[0], scale[1]),
        )
        .core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn index_select(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        let dtype = x.storage.dtype();
        let (outer, d, inner) = split3(x.shape().dims(), dim);
        let l = idx.layout.numel();
        let len = outer * l * inner;
        let out = self.alloc(dtype, len)?;
        let (id, i64) = self.indices(&idx, "index_select")?;
        let xd = self.dense(&x)?;
        let ix = Indices { buf: id.raw(), i64, len: l };
        strided::index_select(&self.ctx, elem(dtype), xd.raw(), (outer, d, inner), ix, out.raw()).core()?;
        Ok(self.wrap(dtype, len, out))
    }

    fn index_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        let (outer, d, inner) = split3(x.shape().dims(), dim);
        let out = self.copy_strided(x)?;
        let (id, i64) = self.indices(&idx, "index_add")?;
        let sd = self.dense(&src)?;
        let ix = Indices { buf: id.raw(), i64, len: idx.layout.numel() };
        let lo = Layout::contiguous(x.shape().clone());
        let ob = self.buf(&View::new(&out, &lo))?;
        strided::index_add(&self.ctx, ob, (outer, d, inner), ix, sd.raw()).core()?;
        Ok(out)
    }

    fn gather(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        let dtype = x.storage.dtype();
        let (outer, d, inner) = split3(x.shape().dims(), dim);
        let di = idx.shape().dims()[dim];
        let len = idx.layout.numel();
        let out = self.alloc(dtype, len)?;
        let (id, i64) = self.indices(&idx, "gather")?;
        let xd = self.dense(&x)?;
        let ix = Indices { buf: id.raw(), i64, len };
        strided::gather(&self.ctx, elem(dtype), xd.raw(), (outer, d, inner), ix, di, out.raw()).core()?;
        Ok(self.wrap(dtype, len, out))
    }

    fn scatter_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        let (outer, d, inner) = split3(x.shape().dims(), dim);
        let di = idx.shape().dims()[dim];
        let out = self.copy_strided(x)?;
        let (id, i64) = self.indices(&idx, "scatter_add")?;
        let sd = self.dense(&src)?;
        let ix = Indices { buf: id.raw(), i64, len: idx.layout.numel() };
        let lo = Layout::contiguous(x.shape().clone());
        let ob = self.buf(&View::new(&out, &lo))?;
        strided::scatter_add(&self.ctx, ob, (outer, d, inner), ix, di, sd.raw()).core()?;
        Ok(out)
    }

    fn topk(&self, x: View<'_>, k: usize, largest: bool) -> Result<(Storage, Storage)> {
        let d = *x.shape().dims().last().unwrap_or(&1);
        let rows = x.layout.numel().checked_div(d).unwrap_or(0);
        let len = rows * k;
        let (vals, idx) = (self.alloc(CoreDType::F32, len)?, self.alloc(CoreDType::I64, len)?);
        if len > 0 {
            let xd = self.dense(&x)?;
            reduce::topk(&self.ctx, xd.raw(), vals.raw(), idx.raw(), rows, d, k, largest).core()?;
        }
        Ok((self.wrap(CoreDType::F32, len, vals), self.wrap(CoreDType::I64, len, idx)))
    }

    fn conv2d_act(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        let mut d = self.conv_desc(x.shape(), w.shape(), p);
        d.act = conv_act(act).ok_or(Error::NotSupported { op: "conv2d_act", backend: BACKEND_NAME })?;
        let (oh, ow) = d.output().core()?;
        let len = d.n * d.c_out * oh * ow;
        let out = self.alloc(CoreDType::F32, len)?;
        let (xd, wd) = (self.dense(&x)?, self.dense(&w)?);
        let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
        conv::conv2d(&self.ctx, &d, xd.raw(), wd.raw(), bd.as_ref().map(|b| b.raw()), out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn batch_norm_act(
        &self,
        x: View<'_>,
        weight: View<'_>,
        bias: View<'_>,
        eps: f32,
        act: Option<UnaryOp>,
        running: Option<RunningUpdate<'_>>,
    ) -> Result<(Storage, Storage, Storage)> {
        let d = self.bn_desc(x.shape(), act, "batch_norm_act")?;
        let len = x.layout.numel();
        let (y, mean, invstd) = (
            self.alloc(CoreDType::F32, len)?,
            self.alloc(CoreDType::F32, d.c)?,
            self.alloc(CoreDType::F32, d.c)?,
        );
        let (xd, wd, bd) = (self.dense(&x)?, self.dense(&weight)?, self.dense(&bias)?);
        let run = match &running {
            Some(r) => Some(BnRunning {
                mean: self.storage_buf(r.mean)?,
                var: self.storage_buf(r.var)?,
                momentum: r.momentum,
            }),
            None => None,
        };
        norm::batch_norm_forward(
            &self.ctx,
            &d,
            xd.raw(),
            wd.raw(),
            bd.raw(),
            eps,
            run,
            y.raw(),
            mean.raw(),
            invstd.raw(),
        )
        .core()?;
        Ok((
            self.wrap(CoreDType::F32, len, y),
            self.wrap(CoreDType::F32, d.c, mean),
            self.wrap(CoreDType::F32, d.c, invstd),
        ))
    }

    fn batch_norm_act_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: View<'_>,
        bias: View<'_>,
        mean: View<'_>,
        invstd: View<'_>,
        act: Option<UnaryOp>,
    ) -> Result<(Storage, Storage, Storage)> {
        let d = self.bn_desc(x.shape(), act, "batch_norm_act_backward")?;
        let len = x.layout.numel();
        let (dx, dw, db) = (
            self.alloc(CoreDType::F32, len)?,
            self.alloc(CoreDType::F32, d.c)?,
            self.alloc(CoreDType::F32, d.c)?,
        );
        let (xd, gd) = (self.dense(&x)?, self.dense(&grad)?);
        let (wd, bd, md, id) =
            (self.dense(&weight)?, self.dense(&bias)?, self.dense(&mean)?, self.dense(&invstd)?);
        norm::batch_norm_backward(
            &self.ctx,
            &d,
            xd.raw(),
            gd.raw(),
            wd.raw(),
            bd.raw(),
            md.raw(),
            id.raw(),
            dx.raw(),
            dw.raw(),
            db.raw(),
        )
        .core()?;
        Ok((
            self.wrap(CoreDType::F32, len, dx),
            self.wrap(CoreDType::F32, d.c, dw),
            self.wrap(CoreDType::F32, d.c, db),
        ))
    }

    fn optimizer_step(
        &self,
        param: &mut Storage,
        grad: View<'_>,
        state: &mut [&mut Storage],
        step: &OptimStep,
    ) -> Result<()> {
        let n = grad.layout.numel();
        let u = match *step {
            OptimStep::Sgd { lr, momentum, weight_decay, nesterov } => {
                OptimUpdate::Sgd { lr, momentum, weight_decay, nesterov }
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
            _ => return Err(Error::NotSupported { op: "optimizer_step", backend: BACKEND_NAME }),
        };
        let gd = self.dense(&grad)?;
        let p = self.storage_buf(param)?;
        let s0 = state.first().map(|s| self.storage_buf(s)).transpose()?;
        let s1 = state.get(1).map(|s| self.storage_buf(s)).transpose()?;
        optim::optimizer_step(&self.ctx, &u, p, gd.raw(), s0, s1, n).core()
    }
}
