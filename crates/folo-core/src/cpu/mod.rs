//! The built-in CPU backend.
//!
//! Kernels work on contiguous `&[f32]` with rayon above a size threshold (32K elements). Strided inputs are either walked
//! row by row with merged dimensions (elementwise ops, copies) or made contiguous once (everything else).
//! The GEMM in [`gemm`] is public so other crates can call it on raw slices.
//!
//! # Files
//! - `mod.rs`: [`CpuBackend`], the [`Backend`] implementation that unpacks each call's views and hands them to the
//!   kernels below; [`in_pool`] and [`take_buffer`].
//! - `elementwise.rs`: strided copies, unary / binary / `where` kernels and the unary gradients, and the shared
//!   [`silu`], [`silu_backward`] and [`fast_exp`] formulas.
//! - `reduce/`: AVX2 reductions (`sum`, `min`, `max`, `var`), arg-reductions and top-k over one dimension
//!   (`[outer, d, inner]`).
//! - `norm/`: fused AVX2+FMA `layer_norm` and `rms_norm` forward and backward kernels.
//! - `index.rs`: `index_select`, `index_add`, `gather` and `scatter_add` over one dimension.
//! - `pool.rs`: max, average and adaptive average pooling and nearest upsampling, with their gradients.
//! - `pool_alloc/`: the two-level cache of large `f32` buffers ([`set_buffer_cache_limit`]).
//! - [`gemm`] (`gemm/`): the single-precision GEMM: blocked driver, micro-kernels, row-pointer operands.
//! - `conv/`: conv2d forward and gradients: implicit-im2col GEMM, Winograd, depthwise, few-input-channel and
//!   per-site paths.
//! - `sites_tests.rs`, `winograd_tests.rs`: unit tests of `conv2d_act_at` and of the Winograd dispatch.

use std::borrow::Cow;

use rayon::prelude::*;

use crate::backend::{Backend, BinaryOp, Conv2dParams, Pool2dParams, ReduceOp, UnaryOp, View};
use crate::{CpuStorage, DType, Device, Element, Error, Layout, Result, Shape, Storage, with_cpu_storage};

mod cast;
mod conv;
pub(crate) mod elementwise;
pub mod gemm;
mod index;
mod norm;
mod pool;
mod pool_alloc;
pub mod q8;
pub(crate) mod reduce;
#[cfg(test)]
mod sites_tests;
mod softmax;
#[cfg(test)]
mod winograd_tests;

pub use elementwise::{SILU_MAX_REL_ERR, fast_exp, silu, silu_backward};
pub(crate) use pool_alloc::{alloc_filled, alloc_uninit, alloc_zeroed, recycle};
pub use pool_alloc::{buffer_cache_bytes, clear_buffer_cache, poison_uninit, set_buffer_cache_limit};

/// A buffer of `n` floats from the CPU buffer cache (see [`set_buffer_cache_limit`]), with unspecified
/// contents: zeros or the stale data of a recycled buffer. For kernels outside folo-core (custom ops) that
/// overwrite every element; wrap the result with [`crate::Tensor::from_vec`] and it returns to the cache when the
/// tensor drops.
pub fn take_buffer(n: usize) -> Vec<f32> {
    alloc_uninit(n)
}

/// Run `f` on a worker of folo-core's CPU pool (the calling thread waits for it), under the calling thread's grad
/// mode and autocast state, and return its result.
///
/// A CPU forward pass makes a hundred or so parallel calls. Each started from a thread outside a rayon pool is
/// injected into the pool's queue, and the caller then sleeps until a worker wakes it; started from a worker,
/// it is pushed on that worker's own queue and the worker takes part in it. Running the whole pass on a worker
/// pays the injection once (a 64x400x64 GEMM, a 1x1 conv at 20x20, on 8 threads: 28 → 17 µs).
///
/// The pool is a dedicated one (as many threads as the global pool), not the global pool: grad mode and
/// autocast are per thread, and while `f` waits for its parallel work its worker runs other jobs of the same
/// pool. In the global pool those could be anybody's (a training step on rayon, say) and would see `f`'s state.
/// The dedicated pool only runs `in_pool` bodies, which set their state themselves, and the work they fan out,
/// which either never reads it (kernels) or sets it too (the detector head's branches). Already on a worker of any
/// rayon pool, or with a one-thread pool, `f` runs right here.
pub fn in_pool<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return f();
    }
    let Some(pool) = cpu_pool() else {
        return f();
    };
    let (grad, cast) = (crate::is_grad_enabled(), crate::autocast_dtype());
    pool.install(move || {
        let _g = crate::autograd::grad_mode(grad);
        let _c = crate::autocast::autocast_as(cast);
        f()
    })
}

/// The pool of [`in_pool`], built on first use with the global pool's thread count (`None` for one thread, or if
/// the threads cannot be spawned: `f` then runs on the caller).
fn cpu_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n = rayon::current_num_threads();
        (n > 1)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("folo-cpu-{i}"))
                    .build()
                    .ok()
            })
            .flatten()
    })
    .as_ref()
}

/// Element count above which kernels go parallel.
pub(crate) const PAR_MIN: usize = 1 << 15;
/// Elements per parallel task for streaming kernels.
pub(crate) const CHUNK: usize = 1 << 13;

/// `f(i, piece)` for every `chunk`-long piece `i` of `out` (the last may be shorter), in parallel.
///
/// Deliberately not generic over `f`: rayon's splitting and joining is compiled once per element type, not once
/// per op and kernel that fans out (the elementwise maps are instantiated for every unary and binary op), which kept
/// this crate's build several times longer. The price is one indirect call per task of thousands of elements;
/// the work inside `f` is still compiled for its op.
pub(crate) fn par_chunks<T: Send>(out: &mut [T], chunk: usize, f: &(dyn Fn(usize, &mut [T]) + Sync)) {
    out.par_chunks_mut(chunk).enumerate().for_each(|(i, piece)| f(i, piece));
}

/// `f(t)` for every `t < n`, in parallel; not generic over `f`, as [`par_chunks`].
pub(crate) fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    (0..n).into_par_iter().for_each(f);
}

/// The CPU backend. Stateless; obtain it through [`crate::backend::backend`]`(Device::Cpu)`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackend;

fn cpu<'a>(v: &View<'a>, op: &'static str) -> Result<&'a CpuStorage> {
    v.storage.as_cpu(op)
}

fn f32s<'a>(v: &View<'a>, op: &'static str) -> Result<&'a [f32]> {
    cpu(v, op)?.as_f32(op)
}

/// The elements of an `F32` view as a contiguous slice (borrowed when already contiguous).
pub(crate) fn contiguous_f32<'a>(v: &View<'a>, op: &'static str) -> Result<Cow<'a, [f32]>> {
    let data = f32s(v, op)?;
    if v.layout.numel() == 0 {
        return Ok(Cow::Borrowed(&[]));
    }
    Ok(match v.layout.contiguous_range() {
        Some(r) => Cow::Borrowed(&data[r]),
        None => Cow::Owned(elementwise::copy_strided_f32(data, v.layout)),
    })
}

/// An `F32` NCHW view whose images each hold their planes contiguously (strides `[s_n, H·W, W, 1]`, ignoring
/// size-1 dims): a contiguous tensor or a channel slice of one, as a conv input part read in place.
fn conv_part<'a>(v: &View<'a>) -> Result<Option<conv::ConvPart<'a>>> {
    let data = f32s(v, "conv2d")?;
    let (l, d) = (v.layout, v.shape().dims());
    let s = l.strides();
    let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
    let ok = (w <= 1 || s[3] == 1) && (h <= 1 || s[2] == w) && (c <= 1 || s[1] == h * w);
    if !ok || n * c * h * w == 0 {
        return Ok(None);
    }
    let img_stride = if n <= 1 { c * h * w } else { s[0] };
    Ok(Some(conv::ConvPart { data, offset: l.offset(), img_stride, c }))
}

/// The elements of a view of any dtype as a contiguous slice (borrowed when already contiguous).
pub(crate) fn contiguous<'a, T: Copy + Send + Sync + Default>(
    data: &'a [T],
    layout: &Layout,
) -> Cow<'a, [T]> {
    if layout.numel() == 0 {
        return Cow::Borrowed(&[]);
    }
    match layout.contiguous_range() {
        Some(r) => Cow::Borrowed(&data[r]),
        None => Cow::Owned(elementwise::copy_strided(data, layout)),
    }
}

/// Index values of an `I64`/`U32` view, validated against `size`.
fn indices(v: &View<'_>, op: &'static str, size: usize) -> Result<Vec<usize>> {
    match cpu(v, op)? {
        CpuStorage::I64(d) => index::to_usize_i64(op, &contiguous(d, v.layout), size),
        CpuStorage::U32(d) => index::to_usize_u32(op, &contiguous(d, v.layout), size),
        s => Err(Error::DTypeMismatch { op, expected: DType::I64, got: s.dtype() }),
    }
}

fn f32_storage(v: Vec<f32>) -> Storage {
    Storage::Cpu(CpuStorage::F32(v))
}

fn nchw(shape: &Shape) -> (usize, usize, usize, usize) {
    let d = shape.dims();
    (d[0], d[1], d[2], d[3])
}

/// Validate view bounds and cross-operand shapes for direct `CpuBackend` norm entry points.
fn check_norm_views(
    op: &'static str,
    x: &View<'_>,
    grad: Option<&View<'_>>,
    weight: Option<&View<'_>>,
    bias: Option<&View<'_>>,
    mean: Option<&View<'_>>,
    rstd: Option<&View<'_>>,
) -> Result<(usize, usize)> {
    let check_fits = |v: &View<'_>| -> Result<()> {
        let data = f32s(v, op)?;
        if !v.layout.fits(data.len()) {
            return Err(Error::invalid_shape(op, v.shape(), "layout reaches outside its storage"));
        }
        Ok(())
    };
    check_fits(x)?;
    if x.shape().rank() == 0 {
        return Err(Error::invalid_shape(op, x.shape(), "needs rank >= 1"));
    }
    let d = *x.shape().dims().last().expect("rank >= 1");
    if d == 0 {
        return Err(Error::invalid_shape(op, x.shape(), "last dimension must be non-zero"));
    }
    let outer = x.layout.numel() / d;
    if let Some(g) = grad {
        check_fits(g)?;
        if g.shape() != x.shape() {
            return Err(Error::ShapeMismatch { op, lhs: x.shape().clone(), rhs: g.shape().clone() });
        }
    }
    for param in [weight, bias].into_iter().flatten() {
        check_fits(param)?;
        if param.shape().dims() != [d] {
            return Err(Error::ShapeMismatch { op, lhs: Shape::new(&[d]), rhs: param.shape().clone() });
        }
    }
    for stat in [mean, rstd].into_iter().flatten() {
        check_fits(stat)?;
        if stat.layout.numel() != outer {
            return Err(Error::ShapeMismatch { op, lhs: Shape::new(&[outer]), rhs: stat.shape().clone() });
        }
    }
    Ok((outer, d))
}

/// Element offset of each batch matrix for a layout `[B.., r, c]`.
fn batch_offsets(layout: &Layout) -> smallvec::SmallVec<[usize; 8]> {
    let r = layout.rank();
    let dims = &layout.dims()[..r - 2];
    let strides = &layout.strides()[..r - 2];
    let count: usize = dims.iter().product();
    let mut out = smallvec::SmallVec::with_capacity(count);
    let mut idx = crate::shape::DimVec::from_elem(0, dims.len());
    for _ in 0..count {
        out.push(layout.offset() + idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>());
        for d in (0..dims.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Fast-path concatenation of contiguous CPU views along `dim` into contiguous `dst`.
/// Returns `false` (leaving `dst` untouched) if any view is not contiguous or storage types do not match.
pub(crate) fn cat_contiguous(dst: &mut Storage, views: &[View<'_>], dim: usize) -> bool {
    let Ok(dst_cpu) = dst.as_cpu_mut("cat") else {
        return false;
    };
    let Some(first) = views.first() else {
        return false;
    };
    let dims = first.shape().dims();
    if dim >= dims.len() {
        return false;
    }
    let outer: usize = dims[..dim].iter().product();
    let inner: usize = dims[dim + 1..].iter().product();
    let mut piece_lens: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::with_capacity(views.len());
    for v in views {
        let Some(len) = v.shape().dims()[dim].checked_mul(inner) else {
            return false;
        };
        piece_lens.push(len);
    }
    macro_rules! go {
        ($($var:ident),*) => {
            match dst_cpu {
                $(CpuStorage::$var(d) => {
                    let mut srcs: smallvec::SmallVec<[&[_]; 8]> = smallvec::SmallVec::with_capacity(views.len());
                    for v in views {
                        let Ok(CpuStorage::$var(s)) = v.storage.as_cpu("cat") else {
                            return false;
                        };
                        if !v.layout.fits(s.len()) {
                            return false;
                        }
                        if v.layout.numel() == 0 {
                            srcs.push(&[]);
                        } else {
                            let Some(r) = v.layout.contiguous_range() else {
                                return false;
                            };
                            srcs.push(&s[r]);
                        }
                    }
                    elementwise::cat_contiguous_slices(d, &srcs, &piece_lens, outer);
                    true
                })*
            }
        };
    }
    go!(F32, F16, U8, U32, I64)
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn full(&self, _device: Device, dtype: DType, len: usize, value: f64) -> Result<Storage> {
        if dtype == DType::F32 {
            return Ok(f32_storage(alloc_filled(len, value as f32)));
        }
        Ok(Storage::Cpu(CpuStorage::full(dtype, len, value)))
    }

    fn upload(&self, _device: Device, data: &CpuStorage) -> Result<Storage> {
        Ok(Storage::Cpu(data.clone()))
    }

    fn download(&self, src: View<'_>) -> Result<CpuStorage> {
        let s = cpu(&src, "download")?;
        if let CpuStorage::F32(v) = s {
            let out = alloc_uninit(src.layout.numel());
            return Ok(CpuStorage::F32(elementwise::copy_strided_f32_into(v, src.layout, out)));
        }
        Ok(with_cpu_storage!(s, v => Element::into_storage(elementwise::copy_strided(v, src.layout))))
    }

    fn copy_strided(&self, src: View<'_>) -> Result<Storage> {
        Ok(Storage::Cpu(self.download(src)?))
    }

    fn copy_into(&self, dst: &mut Storage, dst_layout: &Layout, src: View<'_>) -> Result<()> {
        let s = cpu(&src, "copy_into")?;
        let d = dst.as_cpu_mut("copy_into")?;
        if dst_layout.shape() != src.shape() {
            return Err(Error::ShapeMismatch {
                op: "copy_into",
                lhs: dst_layout.shape().clone(),
                rhs: src.shape().clone(),
            });
        }
        if !dst_layout.fits(d.len()) || !src.layout.fits(s.len()) {
            return Err(Error::invalid_shape(
                "copy_into",
                dst_layout.shape(),
                "layout reaches outside its storage",
            ));
        }
        if !dst_layout.is_non_overlapping() {
            return Err(Error::invalid_shape(
                "copy_into",
                dst_layout.shape(),
                "destination layout overlaps itself",
            ));
        }
        if let (CpuStorage::F32(d), CpuStorage::F32(s)) = (&mut *d, s)
            && dst_layout.numel() > 0
            && let Some(rd) = dst_layout.contiguous_range()
        {
            elementwise::copy_strided_f32_into_slice(s, src.layout, &mut d[rd]);
            return Ok(());
        }
        macro_rules! go {
            ($($v:ident),*) => {
                match (d, s) {
                    $((CpuStorage::$v(d), CpuStorage::$v(s)) => elementwise::copy_into(d, dst_layout, s, src.layout),)*
                    (d, s) => {
                        return Err(Error::DTypeMismatch { op: "copy_into", expected: d.dtype(), got: s.dtype() })
                    }
                }
            };
        }
        go!(F32, F16, U8, U32, I64);
        Ok(())
    }

    fn cast(&self, src: View<'_>, dtype: DType) -> Result<Storage> {
        let s = cpu(&src, "cast")?;
        if s.dtype() == dtype {
            return self.copy_strided(src);
        }
        Ok(Storage::Cpu(cast::cast(s, src.layout, dtype)))
    }

    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        Ok(f32_storage(elementwise::unary(op, f32s(&x, op.name())?, x.layout)))
    }

    fn unary_backward(&self, op: UnaryOp, x: View<'_>, grad: View<'_>) -> Result<Storage> {
        let (xs, gs) = (f32s(&x, op.name())?, f32s(&grad, op.name())?);
        Ok(f32_storage(elementwise::unary_backward(op, xs, x.layout, gs, grad.layout)))
    }

    fn binary(&self, op: BinaryOp, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let (av, bv) = (f32s(&a, op.name())?, f32s(&b, op.name())?);
        Ok(f32_storage(elementwise::binary(op, av, a.layout, bv, b.layout)))
    }

    fn where_cond(&self, cond: View<'_>, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let (av, bv) = (f32s(&a, "where_cond")?, f32s(&b, "where_cond")?);
        let out = match cpu(&cond, "where_cond")? {
            CpuStorage::F32(c) => elementwise::where3_f32(c, cond.layout, av, a.layout, bv, b.layout),
            CpuStorage::U8(c) => elementwise::where3_u8(c, cond.layout, av, a.layout, bv, b.layout),
            s => return Err(Error::UnsupportedDType { op: "where_cond", dtype: s.dtype() }),
        };
        Ok(f32_storage(out))
    }

    fn reduce(&self, op: ReduceOp, x: View<'_>, dim: usize) -> Result<Storage> {
        let data = contiguous_f32(&x, "reduce")?;
        let s3 = reduce::split3(x.shape().dims(), dim);
        Ok(match op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                Storage::Cpu(CpuStorage::I64(reduce::reduce_arg(op == ReduceOp::ArgMax, &data, s3)))
            }
            _ => f32_storage(reduce::reduce_f32(op, &data, s3)),
        })
    }

    fn reduce_var(&self, x: View<'_>, dim: usize, unbiased: bool) -> Result<Storage> {
        let data = contiguous_f32(&x, "reduce_var")?;
        let s3 = reduce::split3(x.shape().dims(), dim);
        Ok(f32_storage(reduce::reduce_var_f32(&data, s3, unbiased)))
    }

    fn softmax(&self, x: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        let data = contiguous_f32(&x, "softmax")?;
        Ok(f32_storage(softmax::softmax(&data, reduce::split3(x.shape().dims(), dim), log)))
    }

    fn softmax_backward(&self, y: View<'_>, grad: View<'_>, dim: usize, log: bool) -> Result<Storage> {
        let (yd, gd) = (contiguous_f32(&y, "softmax_backward")?, contiguous_f32(&grad, "softmax_backward")?);
        Ok(f32_storage(softmax::softmax_backward(&yd, &gd, reduce::split3(y.shape().dims(), dim), log)))
    }

    fn layer_norm(
        &self,
        x: View<'_>,
        weight: Option<View<'_>>,
        bias: Option<View<'_>>,
        eps: f32,
    ) -> Result<(Storage, Storage, Storage)> {
        const OP: &str = "layer_norm";
        let (outer, d) = check_norm_views(OP, &x, None, weight.as_ref(), bias.as_ref(), None, None)?;
        if !eps.is_finite() || eps < 0.0 {
            return Err(Error::invalid_arg(OP, format!("eps must be finite and >= 0 (got {eps})")));
        }
        let xd = contiguous_f32(&x, OP)?;
        let wd = weight.map(|w| contiguous_f32(&w, OP)).transpose()?;
        let bd = bias.map(|b| contiguous_f32(&b, OP)).transpose()?;
        let (y, m, r) = norm::layer_norm(&xd, wd.as_deref(), bd.as_deref(), outer, d, eps);
        Ok((f32_storage(y), f32_storage(m), f32_storage(r)))
    }

    fn layer_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        mean: View<'_>,
        rstd: View<'_>,
        need_weight: bool,
        need_bias: bool,
    ) -> Result<(Storage, Option<Storage>, Option<Storage>)> {
        const OP: &str = "layer_norm_backward";
        let (outer, d) =
            check_norm_views(OP, &x, Some(&grad), weight.as_ref(), None, Some(&mean), Some(&rstd))?;
        let xd = contiguous_f32(&x, OP)?;
        let gd = contiguous_f32(&grad, OP)?;
        let wd = weight.map(|w| contiguous_f32(&w, OP)).transpose()?;
        let md = contiguous_f32(&mean, OP)?;
        let rd = contiguous_f32(&rstd, OP)?;
        let (dx, dw, db) =
            norm::layer_norm_backward(&xd, &gd, wd.as_deref(), &md, &rd, outer, d, need_weight, need_bias);
        Ok((f32_storage(dx), dw.map(f32_storage), db.map(f32_storage)))
    }

    fn rms_norm(&self, x: View<'_>, weight: Option<View<'_>>, eps: f32) -> Result<(Storage, Storage)> {
        const OP: &str = "rms_norm";
        let (outer, d) = check_norm_views(OP, &x, None, weight.as_ref(), None, None, None)?;
        if !eps.is_finite() || eps < 0.0 {
            return Err(Error::invalid_arg(OP, format!("eps must be finite and >= 0 (got {eps})")));
        }
        let xd = contiguous_f32(&x, OP)?;
        let wd = weight.map(|w| contiguous_f32(&w, OP)).transpose()?;
        let (y, r) = norm::rms_norm(&xd, wd.as_deref(), outer, d, eps);
        Ok((f32_storage(y), f32_storage(r)))
    }

    fn rms_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        rstd: View<'_>,
        need_weight: bool,
    ) -> Result<(Storage, Option<Storage>)> {
        const OP: &str = "rms_norm_backward";
        let (outer, d) = check_norm_views(OP, &x, Some(&grad), weight.as_ref(), None, None, Some(&rstd))?;
        let xd = contiguous_f32(&x, OP)?;
        let gd = contiguous_f32(&grad, OP)?;
        let wd = weight.map(|w| contiguous_f32(&w, OP)).transpose()?;
        let rd = contiguous_f32(&rstd, OP)?;
        let (dx, dw) = norm::rms_norm_backward(&xd, &gd, wd.as_deref(), &rd, outer, d, need_weight);
        Ok((f32_storage(dx), dw.map(f32_storage)))
    }

    fn matmul(&self, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let (ad, bd) = (f32s(&a, "matmul")?, f32s(&b, "matmul")?);
        let (la, lb) = (a.layout, b.layout);
        let r = la.rank();
        let (m, k, n) = (la.dims()[r - 2], la.dims()[r - 1], lb.dims()[r - 1]);
        let batch: usize = la.dims()[..r - 2].iter().product();
        if batch * m * n == 0 || k == 0 {
            return Ok(f32_storage(alloc_zeroed(batch * m * n)));
        }
        let mut out = alloc_uninit(batch * m * n);
        let (rsa, csa) = (la.strides()[r - 2], la.strides()[r - 1]);
        let (rsb, csb) = (lb.strides()[r - 2], lb.strides()[r - 1]);
        // Fold the batch into M when B is shared and A's batch rows form one strided matrix.
        if batch > 1 && lb.strides()[..r - 2].iter().all(|&s| s == 0) {
            let shape = Shape::new(&[batch * m, k]);
            if let Some(fl) = la.reshape(&shape) {
                let lhs = gemm::Strided::lhs(ad, fl.offset(), fl.strides()[0], fl.strides()[1]);
                let rhs = gemm::Strided::rhs(bd, lb.offset(), rsb, csb);
                gemm::gemm(batch * m, n, k, &lhs, &rhs, &mut out, n, false, true);
                return Ok(f32_storage(out));
            }
        }
        let (oa, ob) = (batch_offsets(la), batch_offsets(lb));
        let one = |(i, c): (usize, &mut [f32]), threads: usize| {
            let lhs = gemm::Strided::lhs(ad, oa[i], rsa, csa);
            let rhs = gemm::Strided::rhs(bd, ob[i], rsb, csb);
            gemm::gemm_ex(m, n, k, &lhs, &rhs, c, n, false, threads, None);
        };
        // The matrices side by side, the pool split between them (an attention block's two heads: two
        // single-thread GEMMs of 10 MFLOP otherwise). Results do not depend on the split.
        let threads = rayon::current_num_threads();
        if batch > 1 {
            let share = (threads / batch).max(1);
            par_chunks(&mut out, m * n, &|i, c| one((i, c), share));
        } else {
            one((0, &mut out), threads);
        }
        Ok(f32_storage(out))
    }

    fn conv2d(&self, x: View<'_>, w: View<'_>, bias: Option<View<'_>>, p: &Conv2dParams) -> Result<Storage> {
        conv_forward(&[x], w, bias, p, None, false, None, false)
    }

    fn conv2d_act(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        conv_forward(&[x], w, bias, p, act, true, None, true)
    }

    fn conv2d_act_at(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        at: View<'_>,
    ) -> Result<Storage> {
        let (n, _, h, wi) = nchw(x.shape());
        let (_, _, kh, kw) = nchw(w.shape());
        let [oh, ow] =
            p.output_hw(h, wi, kh, kw).expect("conv2d_act_at: output size validated by the caller");
        let at = indices(&at, "conv2d_act_at", n * oh * ow)?;
        // Never use Winograd F(4x4) here: `conv2d_act_at` preserves strict per-site receptive-field isolation.
        conv_forward(&[x], w, bias, p, act, true, Some(&at), false)
    }

    fn conv2d_act_at_tiles(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        at: View<'_>,
    ) -> Result<Storage> {
        let (n, _, h, wi) = nchw(x.shape());
        let (_, _, kh, kw) = nchw(w.shape());
        let [oh, ow] =
            p.output_hw(h, wi, kh, kw).expect("conv2d_act_at_tiles: output size validated by the caller");
        let at = indices(&at, "conv2d_act_at_tiles", n * oh * ow)?;
        conv_forward(&[x], w, bias, p, act, true, Some(&at), true)
    }

    fn conv2d_act_cat(
        &self,
        xs: &[View<'_>],
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        conv_forward(xs, w, bias, p, act, true, None, true)
    }
    fn conv2d_backward_input(
        &self,
        grad: View<'_>,
        w: View<'_>,
        input_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let (gd, wd) = (contiguous_f32(&grad, "conv2d_backward")?, contiguous_f32(&w, "conv2d_backward")?);
        let (n, c, h, wi) = nchw(input_shape);
        let (o, _, kh, kw) = nchw(w.shape());
        let (_, _, oh, ow) = nchw(grad.shape());
        let g = conv::ConvGeom { n, c, h, w: wi, o, kh, kw, oh, ow, p: *p };
        let mut gx = if conv::backward_input_overwrites(&g) && !gd.is_empty() {
            alloc_uninit(input_shape.numel())
        } else {
            alloc_zeroed(input_shape.numel())
        };
        if !gx.is_empty() && !gd.is_empty() {
            conv::conv2d_backward_input(&gd, &wd, &mut gx, &g);
        }
        Ok(f32_storage(gx))
    }

    fn conv2d_backward_weight(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let (xd, gd) = (contiguous_f32(&x, "conv2d_backward")?, contiguous_f32(&grad, "conv2d_backward")?);
        let (n, c, h, wi) = nchw(x.shape());
        let (o, _, kh, kw) = nchw(weight_shape);
        let (_, _, oh, ow) = nchw(grad.shape());
        let g = conv::ConvGeom { n, c, h, w: wi, o, kh, kw, oh, ow, p: *p };
        let mut gw = alloc_zeroed(weight_shape.numel());
        if !gw.is_empty() && !gd.is_empty() {
            conv::conv2d_backward_weight(&xd, &gd, &mut gw, &g);
        }
        Ok(f32_storage(gw))
    }

    fn max_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<(Storage, Storage)> {
        let xd = contiguous_f32(&x, "max_pool2d")?;
        let (n, c, h, w) = nchw(x.shape());
        let ohw = p.output_hw(h, w).expect("max_pool2d: output size validated by the caller");
        let (v, i) = pool::max_pool2d(&xd, (n * c, h, w), p, ohw);
        Ok((f32_storage(v), Storage::Cpu(CpuStorage::I64(i))))
    }

    fn max_pool2d_values(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        let (n, c, h, w) = nchw(x.shape());
        let xd = contiguous_f32(&x, "max_pool2d")?;
        let ohw = p.output_hw(h, w).expect("max_pool2d: output size validated by the caller");
        Ok(f32_storage(pool::max_pool2d_values(&xd, (n * c, h, w), p, ohw)))
    }

    fn max_pool2d_backward(
        &self,
        grad: View<'_>,
        indices: View<'_>,
        input_shape: &Shape,
        _p: &Pool2dParams,
    ) -> Result<Storage> {
        let gd = contiguous_f32(&grad, "max_pool2d_backward")?;
        let idx = match cpu(&indices, "max_pool2d_backward")? {
            CpuStorage::I64(v) => elementwise::copy_strided(v, indices.layout),
            s => {
                return Err(Error::DTypeMismatch {
                    op: "max_pool2d_backward",
                    expected: DType::I64,
                    got: s.dtype(),
                });
            }
        };
        let (n, c, h, w) = nchw(input_shape);
        let (_, _, oh, ow) = nchw(grad.shape());
        Ok(f32_storage(pool::max_pool2d_backward(&gd, &idx, (n * c, h, w), oh * ow)))
    }

    fn avg_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        let xd = contiguous_f32(&x, "avg_pool2d")?;
        let (n, c, h, w) = nchw(x.shape());
        let ohw = p.output_hw(h, w).expect("avg_pool2d: output size validated by the caller");
        Ok(f32_storage(pool::avg_pool2d(&xd, (n * c, h, w), p, ohw)))
    }

    fn avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape, p: &Pool2dParams) -> Result<Storage> {
        let gd = contiguous_f32(&grad, "avg_pool2d_backward")?;
        let (n, c, h, w) = nchw(input_shape);
        let (_, _, oh, ow) = nchw(grad.shape());
        Ok(f32_storage(pool::avg_pool2d_backward(&gd, (n * c, h, w), p, [oh, ow])))
    }

    fn adaptive_avg_pool2d(&self, x: View<'_>, out_hw: [usize; 2]) -> Result<Storage> {
        let xd = contiguous_f32(&x, "adaptive_avg_pool2d")?;
        let (n, c, h, w) = nchw(x.shape());
        Ok(f32_storage(pool::adaptive_avg_pool2d(&xd, (n * c, h, w), out_hw)))
    }

    fn adaptive_avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape) -> Result<Storage> {
        let gd = contiguous_f32(&grad, "adaptive_avg_pool2d_backward")?;
        let (n, c, h, w) = nchw(input_shape);
        let (_, _, oh, ow) = nchw(grad.shape());
        Ok(f32_storage(pool::adaptive_avg_pool2d_backward(&gd, (n * c, h, w), [oh, ow])))
    }

    fn upsample_nearest2d(&self, x: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        let xd = contiguous_f32(&x, "upsample_nearest2d")?;
        let (n, c, h, w) = nchw(x.shape());
        Ok(f32_storage(pool::upsample_nearest2d(&xd, (n * c, h, w), scale)))
    }

    fn upsample_nearest2d_backward(&self, grad: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        let gd = contiguous_f32(&grad, "upsample_nearest2d_backward")?;
        let (n, c, oh, ow) = nchw(grad.shape());
        Ok(f32_storage(pool::upsample_nearest2d_backward(&gd, (n * c, oh / scale[0], ow / scale[1]), scale)))
    }

    fn index_select(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        let s3 = reduce::split3(x.shape().dims(), dim);
        let ix = indices(&idx, "index_select", s3.1)?;
        let s = cpu(&x, "index_select")?;
        if let CpuStorage::F32(v) = s {
            let xd = match x.layout.contiguous_range() {
                Some(r) => Cow::Borrowed(&v[r]),
                None => Cow::Owned(elementwise::copy_strided_f32(v, x.layout)),
            };
            let out = index::index_select_f32(&xd, s3, &ix);
            if let Cow::Owned(buf) = xd {
                recycle(buf);
            }
            return Ok(f32_storage(out));
        }
        Ok(Storage::Cpu(with_cpu_storage!(s, v => {
            Element::into_storage(index::index_select(&contiguous(v, x.layout), s3, &ix))
        })))
    }

    fn index_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        let s3 = reduce::split3(x.shape().dims(), dim);
        let ix = indices(&idx, "index_add", s3.1)?;
        let (xd, sd) = (contiguous_f32(&x, "index_add")?, contiguous_f32(&src, "index_add")?);
        Ok(f32_storage(index::index_add(&xd, s3, &ix, &sd)))
    }

    fn gather(&self, x: View<'_>, dim: usize, idx: View<'_>) -> Result<Storage> {
        let s3 = reduce::split3(x.shape().dims(), dim);
        let ix = indices(&idx, "gather", s3.1)?;
        let di = idx.shape().dims()[dim];
        let s = cpu(&x, "gather")?;
        if let CpuStorage::F32(v) = s {
            let xd = match x.layout.contiguous_range() {
                Some(r) => Cow::Borrowed(&v[r]),
                None => Cow::Owned(elementwise::copy_strided_f32(v, x.layout)),
            };
            let out = index::gather_f32(&xd, s3, &ix, di);
            if let Cow::Owned(buf) = xd {
                recycle(buf);
            }
            return Ok(f32_storage(out));
        }
        Ok(Storage::Cpu(with_cpu_storage!(s, v => {
            Element::into_storage(index::gather(&contiguous(v, x.layout), s3, &ix, di))
        })))
    }

    fn scatter_add(&self, x: View<'_>, dim: usize, idx: View<'_>, src: View<'_>) -> Result<Storage> {
        let s3 = reduce::split3(x.shape().dims(), dim);
        let ix = indices(&idx, "scatter_add", s3.1)?;
        let di = idx.shape().dims()[dim];
        let (xd, sd) = (contiguous_f32(&x, "scatter_add")?, contiguous_f32(&src, "scatter_add")?);
        Ok(f32_storage(index::scatter_add(&xd, s3, &ix, &sd, di)))
    }

    fn index_select_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let s3 = reduce::split3(input_shape.dims(), dim);
        let ix = indices(&idx, "index_select_backward", s3.1)?;
        let gd = contiguous_f32(&grad, "index_select_backward")?;
        let mut out = alloc_zeroed(input_shape.numel());
        index::index_add_into(&mut out, s3, &ix, &gd);
        Ok(f32_storage(out))
    }

    fn gather_backward(
        &self,
        grad: View<'_>,
        dim: usize,
        idx: View<'_>,
        input_shape: &Shape,
    ) -> Result<Storage> {
        let s3 = reduce::split3(input_shape.dims(), dim);
        let ix = indices(&idx, "gather_backward", s3.1)?;
        let di = idx.shape().dims()[dim];
        let gd = contiguous_f32(&grad, "gather_backward")?;
        let mut out = alloc_zeroed(input_shape.numel());
        index::scatter_add_into(&mut out, s3, &ix, &gd, di);
        Ok(f32_storage(out))
    }

    fn topk(&self, x: View<'_>, k: usize, largest: bool) -> Result<(Storage, Storage)> {
        let xd = contiguous_f32(&x, "topk")?;
        let d = *x.shape().dims().last().unwrap_or(&1);
        let rows = x.layout.numel().checked_div(d).unwrap_or(0);
        let (v, i) = reduce::topk(&xd, rows, d, k, largest);
        Ok((f32_storage(v), Storage::Cpu(CpuStorage::I64(i))))
    }
}

/// conv2d forward over the channel concatenation of `xs` with bias and activation, for every conv entry point.
/// `inference` (the fused entry points, which never record a gradient) may take Winograd F(4x4, 3x3) or
/// F(2x2, 3x3), which are not bit for bit the direct conv; the autograd forward keeps the direct conv that its
/// backward matches.
///
/// With `sites` (flat output indices, `conv2d_act_at` / `conv2d_act_at_tiles`), only those outputs need their
/// values: `allow_wino4` enables 4x4 Winograd F(4x4) tiles (for callers populating full 6x6 tile patches, bit for
/// bit the dense call); when `false`, `conv2d_act_at` skips F(4x4) to preserve strict 3x3 receptive-field
/// isolation. Any other conv runs densely.
#[allow(clippy::too_many_arguments)]
fn conv_forward(
    xs: &[View<'_>],
    w: View<'_>,
    bias: Option<View<'_>>,
    p: &Conv2dParams,
    act: Option<UnaryOp>,
    inference: bool,
    sites: Option<&[usize]>,
    allow_wino4: bool,
) -> Result<Storage> {
    // Parts whose images are each contiguous (a channel slice, for example) are read in place.
    let owned = xs
        .iter()
        .map(|x| match conv_part(x)? {
            Some(_) => Ok(None),
            None => contiguous_f32(x, "conv2d").map(|d| Some(d.into_owned())),
        })
        .collect::<Result<smallvec::SmallVec<[_; 4]>>>()?;
    let (n, _, h, wi) = nchw(xs[0].shape());
    let mut parts: smallvec::SmallVec<[_; 4]> = smallvec::SmallVec::with_capacity(xs.len());
    for (x, o) in xs.iter().zip(&owned) {
        let c = x.shape().dims()[1];
        parts.push(match o {
            Some(d) => conv::ConvPart::contiguous(d, c, h * wi),
            None => conv_part(x)?.expect("checked above"),
        });
    }
    let c = parts.iter().map(|p| p.c).sum();
    let wd = contiguous_f32(&w, "conv2d")?;
    let (o, _, kh, kw) = nchw(w.shape());
    let [oh, ow] = p.output_hw(h, wi, kh, kw).expect("conv2d: output size validated by the caller");
    let g = conv::ConvGeom { n, c, h, w: wi, o, kh, kw, oh, ow, p: *p };
    let mut out = alloc_uninit(n * o * oh * ow);
    let bd = bias.map(|b| contiguous_f32(&b, "conv2d")).transpose()?;
    if !out.is_empty() {
        // Bias and activation are applied to each finished block of the output, while it is in cache.
        let epi = gemm::BiasAct { bias: bd.as_deref(), act };
        let fused = bd.is_some() || act.is_some();
        let input = conv::ConvInput { parts: &parts };
        if inference && allow_wino4 && conv::uses_winograd4(&g) {
            let key = derived_key("winograd f(4x4, 3x3)", w.layout);
            let u = w.derived(key, || conv::try_transform_weights4(&wd, o, c));
            if let Some(u) = u.as_deref() {
                match sites {
                    Some(at) => {
                        let tiles = conv::winograd4_tiles_at(&g, at);
                        conv::winograd4_forward_at(input, &wd, u, &mut out, &g, fused.then_some(epi), &tiles);
                    }
                    None => conv::winograd4_forward(input, &wd, u, &mut out, &g, fused.then_some(epi)),
                }
                return Ok(f32_storage(out));
            }
        } else if inference && conv::uses_winograd(&g) {
            // Transformed once per weight tensor, and kept with it until it is written.
            let key = derived_key("winograd f(2x2, 3x3)", w.layout);
            let u = w.derived(key, || conv::transform_weights(&wd, o, c));
            match sites {
                Some(at) => {
                    let tiles = conv::winograd_tiles_at(&g, at);
                    conv::winograd_forward_at(input, &u, &mut out, &g, fused.then_some(epi), &tiles);
                }
                None => conv::winograd_forward(input, &u, &mut out, &g, fused.then_some(epi)),
            }
            return Ok(f32_storage(out));
        }
        if let Some(at) = sites
            && conv::pointwise_forward_at(input, &wd, &mut out, &g, fused.then_some(epi), at)
        {
            return Ok(f32_storage(out));
        }
        conv::conv2d_forward_parts(
            conv::ConvInput { parts: &parts },
            &wd,
            &mut out,
            &g,
            false,
            fused.then_some(epi),
        );
    }
    Ok(f32_storage(out))
}

/// The key of a value derived from data read through `layout` ([`View::derived`]): `what` and the layout.
fn derived_key(what: &str, layout: &Layout) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    (what, layout.offset(), layout.shape().dims(), layout.strides()).hash(&mut h);
    h.finish()
}
