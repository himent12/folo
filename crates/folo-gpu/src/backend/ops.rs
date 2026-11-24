//! folo-core's [`Backend`] trait for [`GpuBackend`]: every op, mapped onto the kernel layer, and the f32 detour of
//! the ops without an f16 kernel.

use folo_core::backend::{
    Backend, BinaryOp, ClassSites, Conv2dParams, DetectDecode, LetterboxImage, MuonItem, MuonSettings,
    NormReduce, OptimItem, OptimStep, Pool2dParams, ReduceOp, RunningUpdate, SiteLevel, UnaryOp, View,
};
use folo_core::{CpuStorage, DType as CoreDType, Device, Error, Layout, Result, Shape, Storage};

use super::convert::{
    Core, conv_act, conv_label, elem, f32_only, kdt, nchw, optim_update, pool2d, split3, strided_of,
};
use super::encode::conv_input;
use super::storage::Dense;
use super::{BACKEND_NAME, GpuBackend, core_err};
use crate::error::GpuError;
use crate::kernels::conv;
use crate::kernels::ew::EwOp;
use crate::kernels::gemm_cat::{CAT_STORAGE_BUFFERS, MAX_CAT_PARTS};
use crate::kernels::layout;
use crate::kernels::muon;
use crate::kernels::norm::{self, BnRunning};
use crate::kernels::optim;
use crate::kernels::pool_grad;
use crate::kernels::reduce::{self, ReduceOp as GpuReduce};
use crate::kernels::strided::{self, Indices};
use crate::kernels::vision::{self, LetterboxDesc};
use crate::profile;

impl GpuBackend {
    /// `f` on an f32 copy of an f16 `x`, its result cast back to f16 (ops without an f16 kernel); `f(x)` for
    /// anything else.
    fn via_f32(&self, x: View<'_>, f: impl FnOnce(View<'_>) -> Result<Storage>) -> Result<Storage> {
        if x.storage.dtype() != CoreDType::F16 {
            return f(x);
        }
        let tmp = self.cast(x, CoreDType::F32)?;
        let out = f(View::new(&tmp, &Layout::contiguous(x.shape().clone())))?;
        let whole = Layout::contiguous(Shape::new(&[out.len()]));
        self.cast(View::new(&out, &whole), CoreDType::F16)
    }
}

impl Backend for GpuBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.sync().core()
    }

    fn computes_f16(&self) -> bool {
        self.ctx.caps().shader_f16
    }

    fn full(&self, _device: Device, dtype: CoreDType, len: usize, value: f64) -> Result<Storage> {
        let _p = profile::scope(|| format!("full {len}"));
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
        let _p = profile::scope(|| format!("copy_strided {:?}", src.shape().dims()));
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
        let _p = profile::scope(|| format!("copy_into {:?}", src.shape().dims()));
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
        let _p = profile::scope(|| format!("cast {from}->{dtype} {:?}", src.shape().dims()));
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
        let _p = profile::scope(|| format!("reduce {op:?} {:?} dim {dim}", x.shape().dims()));
        let n = outer * inner;
        let arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
        let xdt = x.storage.dtype();
        let dtype = if arg { CoreDType::I64 } else { xdt };
        let out = self.alloc(dtype, n)?;
        if n > 0 {
            // The arg-reductions have an f32 kernel only.
            let wide = if arg && xdt == CoreDType::F16 { Some(self.cast(x, CoreDType::F32)?) } else { None };
            let wl = Layout::contiguous(x.shape().clone());
            let x = wide.as_ref().map_or(x, |s| View::new(s, &wl));
            let xd = self.dense(&x)?;
            let gop = match op {
                ReduceOp::Sum => Some(GpuReduce::Sum),
                ReduceOp::Max => Some(GpuReduce::Max),
                ReduceOp::Min => Some(GpuReduce::Min),
                _ => None,
            };
            match gop {
                Some(g) => reduce::reduce_split(&self.ctx, g, xd.raw(), out.raw(), outer, r, inner, kdt(xdt))
                    .core()?,
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
        let _p = profile::scope(|| format!("softmax {:?}", x.shape().dims()));
        let n = x.layout.numel();
        let dtype = x.storage.dtype();
        let out = self.alloc(dtype, n)?;
        if n > 0 {
            let xd = self.dense(&x)?;
            let dims = split3(x.shape().dims(), dim);
            reduce::softmax_dim(&self.ctx, xd.raw(), out.raw(), dims, log, kdt(dtype)).core()?;
        }
        Ok(self.wrap(dtype, n, out))
    }

    fn matmul(&self, a: View<'_>, b: View<'_>) -> Result<Storage> {
        let r = a.layout.rank();
        let (m, k, n) = (a.layout.dims()[r - 2], a.layout.dims()[r - 1], b.layout.dims()[r - 1]);
        let batch: usize = a.layout.dims()[..r - 2].iter().product();
        let _p = profile::scope(|| format!("matmul {batch}x{m}x{k}x{n}"));
        let len = batch * m * n;
        let dtype = a.storage.dtype();
        let out = self.alloc(dtype, len)?;
        if len == 0 {
            return Ok(self.wrap(dtype, len, out));
        }
        if k == 0 {
            self.ctx.clear(out.raw(), elem(dtype).buffer_bytes(len).core()?);
            return Ok(self.wrap(dtype, len, out));
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
        Ok(self.wrap(dtype, len, out))
    }

    fn conv2d(&self, x: View<'_>, w: View<'_>, bias: Option<View<'_>>, p: &Conv2dParams) -> Result<Storage> {
        self.conv2d_act(x, w, bias, p, None)
    }

    fn conv2d_backward_input(
        &self,
        grad: View<'_>,
        w: View<'_>,
        input_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let dtype = grad.storage.dtype();
        let d = self.conv_desc(input_shape, w.shape(), p, dtype);
        let _p = profile::scope(|| format!("conv_bwd_in {}", conv_label(&d)));
        let len = input_shape.numel();
        let out = self.alloc(dtype, len)?;
        let (gd, wd) = (self.dense(&grad)?, self.dense(&w)?);
        let per = (grad.shape().numel().checked_div(d.n).unwrap_or(0), d.c_in * d.h * d.w);
        self.conv_by_batch(&d, (gd.raw(), per.0), (out.raw(), per.1), |d, g, gx| {
            conv::conv2d_backward_input(&self.ctx, d, g, wd.raw(), gx)
        })?;
        Ok(self.wrap(dtype, len, out))
    }

    fn conv2d_backward_weight(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
    ) -> Result<Storage> {
        let dtype = x.storage.dtype();
        self.conv2d_backward_weight_as(x, grad, weight_shape, p, dtype)
    }

    fn conv2d_backward_weight_as(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight_shape: &Shape,
        p: &Conv2dParams,
        out_dtype: CoreDType,
    ) -> Result<Storage> {
        let d = self.conv_desc(x.shape(), weight_shape, p, x.storage.dtype());
        let _p = profile::scope(|| format!("conv_bwd_w {}", conv_label(&d)));
        let len = weight_shape.numel();
        let out = self.alloc(out_dtype, len)?;
        let (xd, gd) = (self.dense(&x)?, self.dense(&grad)?);
        conv::conv2d_backward_weight_as(&self.ctx, &d, xd.raw(), gd.raw(), out.raw(), kdt(out_dtype))
            .core()?;
        Ok(self.wrap(out_dtype, len, out))
    }

    fn max_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<(Storage, Storage)> {
        let (n, c, h, w) = nchw(x.shape());
        let _p = profile::scope(|| format!("max_pool {:?}", x.shape().dims()));
        let win = pool2d(p);
        let (oh, ow) = win.output(h, w).core()?;
        let len = n * c * oh * ow;
        let dtype = x.storage.dtype();
        let (vals, idx) = (self.alloc(dtype, len)?, self.alloc(CoreDType::I64, len)?);
        let xd = self.dense(&x)?;
        let planes = (n * c, h, w);
        pool_grad::max_pool2d_indices(&self.ctx, xd.raw(), vals.raw(), idx.raw(), planes, &win, kdt(dtype))
            .core()?;
        Ok((self.wrap(dtype, len, vals), self.wrap(CoreDType::I64, len, idx)))
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
        let dtype = grad.storage.dtype();
        let _p = profile::scope(|| format!("max_pool_bwd {:?}", input_shape.dims()));
        let (n, c, h, w) = nchw(input_shape);
        let len = input_shape.numel();
        let out = self.alloc(dtype, len)?;
        let (gd, id) = (self.dense(&grad)?, self.dense(&indices)?);
        let planes = (n * c, h, w);
        pool_grad::max_pool2d_backward(
            &self.ctx,
            gd.raw(),
            id.raw(),
            out.raw(),
            planes,
            &pool2d(p),
            kdt(dtype),
        )
        .core()?;
        Ok(self.wrap(dtype, len, out))
    }

    fn avg_pool2d(&self, x: View<'_>, p: &Pool2dParams) -> Result<Storage> {
        self.via_f32(x, |x| {
            let (n, c, h, w) = nchw(x.shape());
            let win = pool2d(p);
            let (oh, ow) = win.output(h, w).core()?;
            let len = n * c * oh * ow;
            let out = self.alloc(CoreDType::F32, len)?;
            let xd = self.dense(&x)?;
            pool_grad::avg_pool2d(&self.ctx, xd.raw(), out.raw(), (n * c, h, w), &win).core()?;
            Ok(self.wrap(CoreDType::F32, len, out))
        })
    }

    fn avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape, p: &Pool2dParams) -> Result<Storage> {
        f32_only(&grad, "avg_pool2d_backward")?;
        let (n, c, h, w) = nchw(input_shape);
        let len = input_shape.numel();
        let out = self.alloc(CoreDType::F32, len)?;
        let gd = self.dense(&grad)?;
        pool_grad::avg_pool2d_backward(&self.ctx, gd.raw(), out.raw(), (n * c, h, w), &pool2d(p)).core()?;
        Ok(self.wrap(CoreDType::F32, len, out))
    }

    fn adaptive_avg_pool2d(&self, x: View<'_>, out_hw: [usize; 2]) -> Result<Storage> {
        self.via_f32(x, |x| {
            let (n, c, h, w) = nchw(x.shape());
            let len = n * c * out_hw[0] * out_hw[1];
            let out = self.alloc(CoreDType::F32, len)?;
            let xd = self.dense(&x)?;
            let planes = (n * c, h, w);
            pool_grad::adaptive_avg_pool2d(&self.ctx, xd.raw(), out.raw(), planes, (out_hw[0], out_hw[1]))
                .core()?;
            Ok(self.wrap(CoreDType::F32, len, out))
        })
    }

    fn adaptive_avg_pool2d_backward(&self, grad: View<'_>, input_shape: &Shape) -> Result<Storage> {
        f32_only(&grad, "adaptive_avg_pool2d_backward")?;
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
        let _p = profile::scope(|| format!("upsample {:?}", x.shape().dims()));
        let (n, c, h, w) = nchw(x.shape());
        let (oh, ow) = (h * scale[0], w * scale[1]);
        let len = n * c * oh * ow;
        let dtype = x.storage.dtype();
        let out = self.alloc(dtype, len)?;
        if len > 0 {
            let xd = self.dense(&x)?;
            layout::upsample_nearest2d(&self.ctx, xd.raw(), out.raw(), n * c, h, w, oh, ow, kdt(dtype))
                .core()?;
        }
        Ok(self.wrap(dtype, len, out))
    }

    fn upsample_nearest2d_backward(&self, grad: View<'_>, scale: [usize; 2]) -> Result<Storage> {
        let _p = profile::scope(|| format!("upsample_bwd {:?}", grad.shape().dims()));
        let dtype = grad.storage.dtype();
        let (n, c, oh, ow) = nchw(grad.shape());
        let (h, w) = (oh / scale[0], ow / scale[1]);
        let len = n * c * h * w;
        let out = self.alloc(dtype, len)?;
        let gd = self.dense(&grad)?;
        pool_grad::upsample_nearest2d_backward(
            &self.ctx,
            gd.raw(),
            out.raw(),
            (n * c, h, w),
            (scale[0], scale[1]),
            kdt(dtype),
        )
        .core()?;
        Ok(self.wrap(dtype, len, out))
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
        let dtype = x.storage.dtype();
        let mut d = self.conv_desc(x.shape(), w.shape(), p, dtype);
        d.act = conv_act(act).ok_or(Error::NotSupported { op: "conv2d_act", backend: BACKEND_NAME })?;
        let _p = profile::scope(|| format!("conv {}", conv_label(&d)));
        let (oh, ow) = d.output().core()?;
        let len = d.n * d.c_out * oh * ow;
        let out = self.alloc(dtype, len)?;
        let wd = self.dense(&w)?;
        let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
        // A channel slice (or any view with contiguous images) is read in place, not gathered first.
        if !d.is_depthwise()
            && d.n * d.groups <= 65535
            && let Some(at) = conv_input(&x)
        {
            let bias = bd.as_ref().map(|b| b.raw());
            conv::conv2d_input(&self.ctx, &d, (self.buf(&x)?, at), wd.raw(), bias, out.raw(), None).core()?;
            return Ok(self.wrap(dtype, len, out));
        }
        let xd = self.dense(&x)?;
        let per = (d.c_in * d.h * d.w, d.c_out * oh * ow);
        self.conv_by_batch(&d, (xd.raw(), per.0), (out.raw(), per.1), |d, x, y| {
            conv::conv2d(&self.ctx, d, x, wd.raw(), bd.as_ref().map(|b| b.raw()), y).map(drop)
        })?;
        Ok(self.wrap(dtype, len, out))
    }

    fn conv2d_act_sites(
        &self,
        x: View<'_>,
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
        sites: View<'_>,
        to_map: bool,
    ) -> Result<Storage> {
        self.sites_conv(x, w, bias, p, act, sites, to_map)
    }

    fn conv2d_act_cat(
        &self,
        xs: &[View<'_>],
        w: View<'_>,
        bias: Option<View<'_>>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Storage> {
        // 1×1 convs over 2 to `MAX_CAT_PARTS` parts whose images are contiguous: one GEMM reads them in place.
        // Anything else is `NotSupported`, and the caller concatenates first.
        let unsupported = Error::NotSupported { op: "conv2d_act_cat", backend: BACKEND_NAME };
        if xs.is_empty()
            || xs.iter().any(|x| x.shape().rank() != 4)
            || self.ctx.caps().max_storage_buffers < CAT_STORAGE_BUFFERS
        {
            return Err(unsupported);
        }
        let (n, _, h, wi) = nchw(xs[0].shape());
        if xs.iter().any(|x| {
            let (pn, _, ph, pw) = nchw(x.shape());
            (pn, ph, pw) != (n, h, wi)
        }) {
            return Err(unsupported);
        }
        // Parts without channels add nothing; with one part left this is the plain conv on it.
        let xs: Vec<View<'_>> = xs.iter().copied().filter(|x| nchw(x.shape()).1 > 0).collect();
        match xs.len() {
            1 => return self.conv2d_act(xs[0], w, bias, p, act),
            2..=MAX_CAT_PARTS => {}
            _ => return Err(unsupported),
        }
        let dtype = xs[0].storage.dtype();
        let c: usize = xs.iter().map(|x| nchw(x.shape()).1).sum();
        let mut d = self.conv_desc(&Shape::new(&[n, c, h, wi]), w.shape(), p, dtype);
        d.act = conv_act(act).ok_or(unsupported.clone())?;
        if !d.is_pointwise() || d.groups != 1 || n > 65535 {
            return Err(unsupported);
        }
        let mut parts = Vec::with_capacity(xs.len());
        for x in &xs {
            let Some(at) = conv_input(x) else { return Err(unsupported) };
            parts.push(conv::ConvPart { buf: self.buf(x)?, channels: nchw(x.shape()).1, at });
        }
        let _p = profile::scope(|| format!("conv_cat {} {}", xs.len(), conv_label(&d)));
        let (oh, ow) = d.output().core()?;
        let len = d.n * d.c_out * oh * ow;
        let out = self.alloc(dtype, len)?;
        let wd = self.dense(&w)?;
        let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
        conv::conv2d_cat(&self.ctx, &d, &parts, wd.raw(), bd.as_ref().map(|b| b.raw()), out.raw()).core()?;
        Ok(self.wrap(dtype, len, out))
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
        let d = self.bn_desc(&x, act, "batch_norm_act")?;
        let _p = profile::scope(|| format!("bn_fwd {:?} {:?}", x.shape().dims(), d.dtype));
        let len = x.layout.numel();
        let dtype = x.storage.dtype();
        let (y, mean, invstd) =
            (self.alloc(dtype, len)?, self.alloc(CoreDType::F32, d.c)?, self.alloc(CoreDType::F32, d.c)?);
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
            self.wrap(dtype, len, y),
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
        let d = self.bn_desc(&x, act, "batch_norm_act_backward")?;
        let _p = profile::scope(|| format!("bn_bwd {:?} {:?}", x.shape().dims(), d.dtype));
        let len = x.layout.numel();
        let dtype = x.storage.dtype();
        let (dx, dw, db) =
            (self.alloc(dtype, len)?, self.alloc(CoreDType::F32, d.c)?, self.alloc(CoreDType::F32, d.c)?);
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
            self.wrap(dtype, len, dx),
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
        let _p = profile::scope(|| "optim".to_string());
        let u =
            optim_update(step).ok_or(Error::NotSupported { op: "optimizer_step", backend: BACKEND_NAME })?;
        let gd = self.dense(&grad)?;
        let p = self.storage_buf(param)?;
        let s0 = state.first().map(|s| self.storage_buf(s)).transpose()?;
        let s1 = state.get(1).map(|s| self.storage_buf(s)).transpose()?;
        optim::optimizer_step(&self.ctx, &u, p, gd.raw(), s0, s1, n).core()
    }

    fn optimizer_step_many(&self, items: &mut [OptimItem<'_>], step: &OptimStep) -> Result<()> {
        let _p = profile::scope(|| "optim_many".to_string());
        let u = optim_update(step)
            .ok_or(Error::NotSupported { op: "optimizer_step_many", backend: BACKEND_NAME })?;
        let grads = items.iter().map(|it| self.dense(&it.grad)).collect::<Result<Vec<_>>>()?;
        let mut many = Vec::with_capacity(items.len());
        for (it, g) in items.iter().zip(&grads) {
            many.push(optim::ManyItem {
                p: self.storage_buf(it.param)?,
                g: g.raw(),
                s0: it.state.first().map(|s| self.storage_buf(s)).transpose()?,
                s1: it.state.get(1).map(|s| self.storage_buf(s)).transpose()?,
                n: it.grad.layout.numel(),
            });
        }
        optim::optimizer_step_many(&self.ctx, &u, &many).core()
    }

    fn unscale_check_many(
        &self,
        xs: &mut [&mut Storage],
        ns: &[usize],
        factor: f32,
        flag: &mut Storage,
    ) -> Result<()> {
        let _p = profile::scope(|| "unscale_many".to_string());
        let bufs =
            xs.iter().zip(ns).map(|(x, &n)| Ok((self.storage_buf(x)?, n))).collect::<Result<Vec<_>>>()?;
        optim::unscale_check_many(&self.ctx, &bufs, factor, self.storage_buf(flag)?).core()
    }

    fn norm_partials_many(&self, xs: &[View<'_>], op: NormReduce, h: Option<View<'_>>) -> Result<Storage> {
        let _p = profile::scope(|| "norm_partials_many".to_string());
        let op = match op {
            NormReduce::AbsMax => optim::NormPartial::AbsMax,
            NormReduce::ScaledSquares => optim::NormPartial::ScaledSquares,
            _ => return Err(Error::NotSupported { op: "norm_partials_many", backend: BACKEND_NAME }),
        };
        let dense = xs.iter().map(|v| self.dense(v)).collect::<Result<Vec<_>>>()?;
        let bufs: Vec<(&wgpu::Buffer, usize)> =
            dense.iter().zip(xs).map(|(d, v)| (d.raw(), v.layout.numel())).collect();
        let hd = h.as_ref().map(|v| self.dense(v)).transpose()?;
        let ns: Vec<usize> = bufs.iter().map(|b| b.1).collect();
        let n = optim::norm_partials_len(&ns);
        let out = self.alloc(CoreDType::F32, n.max(1))?;
        optim::norm_partials_many(&self.ctx, op, &bufs, hd.as_ref().map(|d| d.raw()), out.raw()).core()?;
        Ok(self.wrap(CoreDType::F32, n, out))
    }

    fn scale_many(&self, xs: &mut [&mut Storage], ns: &[usize], f1: View<'_>, f2: View<'_>) -> Result<()> {
        let _p = profile::scope(|| "scale_many".to_string());
        let (d1, d2) = (self.dense(&f1)?, self.dense(&f2)?);
        let bufs =
            xs.iter().zip(ns).map(|(x, &n)| Ok((self.storage_buf(x)?, n))).collect::<Result<Vec<_>>>()?;
        optim::scale_many(&self.ctx, &bufs, d1.raw(), d2.raw()).core()
    }

    fn muon_step_many(&self, items: &mut [MuonItem<'_>], s: &MuonSettings) -> Result<()> {
        let _p = profile::scope(|| "muon_many".to_string());
        let dense = items
            .iter()
            .map(|it| Ok((self.dense(&it.momentum)?, self.dense(&it.grad)?)))
            .collect::<Result<Vec<_>>>()?;
        let mut mats = Vec::with_capacity(items.len());
        for (it, (m, g)) in items.iter().zip(&dense) {
            mats.push(muon::MuonMatrix {
                p: self.storage_buf(it.param)?,
                m: m.raw(),
                g: g.raw(),
                rows: it.rows,
                cols: it.cols,
                scale: it.scale,
                depthwise: it.depthwise,
            });
        }
        let settings = muon::MuonSettings {
            momentum: s.momentum,
            nesterov: s.nesterov,
            step_coeffs: s.step_coeffs,
            steps: s.steps,
            schatten4: s.schatten4,
            eps: s.eps,
            lr: s.lr,
        };
        // Declined (nothing written) where the device's bindings are too few or too small: tensor ops instead.
        if muon::muon_step_many(&self.ctx, &mats, &settings).core()? {
            Ok(())
        } else {
            Err(Error::NotSupported { op: "muon_step_many", backend: BACKEND_NAME })
        }
    }

    fn unscale_check(&self, x: &mut Storage, n: usize, factor: f32, flag: &mut Storage) -> Result<()> {
        let _p = profile::scope(|| "unscale".to_string());
        let (xb, fb) = (self.storage_buf(x)?, self.storage_buf(flag)?);
        optim::unscale_check(&self.ctx, xb, n, factor, fb).core()
    }

    fn letterbox_rgb8(
        &self,
        images: &[View<'_>],
        taps: View<'_>,
        geoms: &[LetterboxImage],
        out_hw: [usize; 2],
        pad_value: u8,
        scale: f32,
        dtype: CoreDType,
    ) -> Result<Storage> {
        let [oh, ow] = out_hw;
        let per = 3 * oh * ow;
        // Prefilter blocks the kernel cannot sum in u32: let the caller's host path handle them.
        if geoms.iter().any(|g| !vision::prefilter_fits(g.prefilter[0], g.prefilter[1])) {
            return Err(Error::NotSupported { op: "letterbox_rgb8", backend: BACKEND_NAME });
        }
        let len = images.len() * per;
        let out = self.alloc(dtype, len)?;
        let td = self.dense(&taps)?;
        for (i, (im, g)) in images.iter().zip(geoms).enumerate() {
            // A contiguous view (one image of a shared upload) is read in place at its byte offset.
            let (src, src_offset) = if im.layout.is_contiguous() {
                (Dense::Borrowed(self.buf(im)?), im.layout.offset() * im.storage.dtype().size_in_bytes())
            } else {
                (self.dense(im)?, 0)
            };
            let d = LetterboxDesc {
                out: [ow, oh],
                src: g.src,
                src_offset,
                pad: g.pad,
                resized: g.resized,
                prefilter: g.prefilter,
                pad_value,
                scale,
                out_offset: i * per,
                taps: g.taps,
            };
            vision::letterbox_rgb8(&self.ctx, src.raw(), td.raw(), out.raw(), &d, kdt(dtype)).core()?;
        }
        Ok(self.wrap(dtype, len, out))
    }

    fn detect_candidates(
        &self,
        box_raw: View<'_>,
        cls_raw: View<'_>,
        anchors: View<'_>,
        p: &DetectDecode,
    ) -> Result<Storage> {
        let d = box_raw.shape().dims();
        let (b, a) = (d[0], d[2]);
        let len = b * (1 + vision::RECORD_WORDS * a);
        let out = self.alloc(CoreDType::U32, len)?;
        let (bd, cd, ad) = (self.dense(&box_raw)?, self.dense(&cls_raw)?, self.dense(&anchors)?);
        vision::detect_candidates(
            &self.ctx,
            bd.raw(),
            cd.raw(),
            ad.raw(),
            out.raw(),
            b,
            a,
            p.num_classes,
            p.reg_max,
            p.conf,
            kdt(box_raw.storage.dtype()),
        )
        .core()?;
        Ok(self.wrap(CoreDType::U32, len, out))
    }

    fn detect_class_sites(&self, cls: View<'_>, p: &ClassSites) -> Result<(Storage, Storage, Storage)> {
        self.class_sites(cls, p)
    }

    fn detect_decode_sites(&self, levels: &[SiteLevel<'_>], reg_max: usize) -> Result<Storage> {
        self.decode_sites(levels, reg_max)
    }
}
