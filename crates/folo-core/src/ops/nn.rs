//! Matrix products, convolution, pooling and upsampling.

use super::grads;
use crate::autocast::casts;
use crate::backend::{Conv2dParams, Pool2dParams, View};
use crate::shape::DimVec;
use crate::{DType, Error, Result, Shape, Storage, Tensor, no_grad};

impl Tensor {
    /// Matrix product. Both operands need rank >= 2: `[.., m, k] @ [.., k, n] -> [.., m, n]`, with the batch
    /// dims broadcast (so `[B, m, k] @ [k, n]` works and runs as one large GEMM). Under [`crate::autocast`], `F32`
    /// operands are cast to `F16` first (recorded casts) and the product is `F16`.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        if casts(self) || casts(other) {
            return self.to_dtype(DType::F16)?.matmul(&other.to_dtype(DType::F16)?);
        }
        self.check_same_device(other, "matmul")?;
        self.check_float("matmul")?;
        other.check_float_like(self, "matmul")?;
        let mismatch =
            || Error::ShapeMismatch { op: "matmul", lhs: self.shape().clone(), rhs: other.shape().clone() };
        if self.rank() < 2 || other.rank() < 2 {
            return Err(mismatch());
        }
        let (ra, rb) = (self.rank(), other.rank());
        let (m, k) = (self.dims()[ra - 2], self.dims()[ra - 1]);
        let (k2, n) = (other.dims()[rb - 2], other.dims()[rb - 1]);
        if k != k2 {
            return Err(mismatch());
        }
        let ba = Shape::new(&self.dims()[..ra - 2]);
        let bb = Shape::new(&other.dims()[..rb - 2]);
        let batch = ba.broadcast(&bb, "matmul").map_err(|_| mismatch())?;
        let shape_of = |r: usize, c: usize| {
            let mut d: DimVec = batch.dims().iter().copied().collect();
            d.extend([r, c]);
            Shape::from(d)
        };
        let la = self.layout().broadcast_to(&shape_of(m, k), "matmul")?;
        let lb = other.layout().broadcast_to(&shape_of(k, n), "matmul")?;
        let b = self.backend()?;
        let s = super::with_views([self, other], [&la, &lb], |[a, bv]| b.matmul(a, bv))?;
        let out = Tensor::wrap(s, shape_of(m, n), "matmul")?;
        let (x, y) = (self.clone(), other.clone());
        Ok(out.track("matmul", &[self, other], move |g| {
            let ga = if x.requires_grad() { Some(g.matmul(&y.t()?)?.sum_to(x.shape())?) } else { None };
            let gb = if !y.requires_grad() {
                None
            } else if y.rank() == 2 && x.rank() > 2 {
                // Shared weight: fold the batch into one GEMM instead of summing B products.
                let rows: usize = x.dims()[..x.rank() - 1].iter().product();
                let xk = x.reshape([rows, k])?;
                let gn = g.reshape([rows, n])?;
                Some(xk.t()?.matmul(&gn)?)
            } else {
                Some(x.t()?.matmul(g)?.sum_to(y.shape())?)
            };
            Ok(grads![ga, gb])
        }))
    }

    /// 2-D convolution of an NCHW input with an OIHW weight (`[C_out, C_in / groups, KH, KW]`) and optional
    /// bias `[C_out]`. `stride`, `padding` and `dilation` apply to both spatial dims.
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let p = Conv2dParams { stride: [stride; 2], padding: [padding; 2], dilation: [dilation; 2], groups };
        self.conv2d_with(weight, bias, &p)
    }

    /// Validate a conv2d call; returns the output shape.
    fn check_conv2d(&self, weight: &Tensor, bias: Option<&Tensor>, p: &Conv2dParams) -> Result<Shape> {
        const OP: &str = "conv2d";
        self.check_float(OP)?;
        weight.check_float_like(self, OP)?;
        self.check_same_device(weight, OP)?;
        let (n, c, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "input must be NCHW"))?;
        let (o, cg, kh, kw) =
            weight.dims4().map_err(|_| Error::invalid_shape(OP, weight.shape(), "weight must be OIHW"))?;
        if p.groups == 0 || p.stride.contains(&0) || p.dilation.contains(&0) {
            return Err(Error::invalid_arg(
                OP,
                format!("stride, dilation and groups must be positive: {p:?}"),
            ));
        }
        if c % p.groups != 0 || o % p.groups != 0 || cg * p.groups != c || kh == 0 || kw == 0 {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: self.shape().clone(),
                rhs: weight.shape().clone(),
            });
        }
        if let Some(b) = bias {
            b.check_float_like(self, OP)?;
            self.check_same_device(b, OP)?;
            if b.dims() != [o] {
                return Err(Error::ShapeMismatch {
                    op: OP,
                    lhs: weight.shape().clone(),
                    rhs: b.shape().clone(),
                });
            }
        }
        let [oh, ow] = p.output_hw(h, w, kh, kw).ok_or_else(|| {
            Error::invalid_arg(OP, format!("kernel {kh}x{kw} with {p:?} does not fit input {}", self.shape()))
        })?;
        if [o, oh, ow].iter().try_fold(n, |a, &b| a.checked_mul(b)).is_none() {
            return Err(Error::invalid_arg(OP, format!("output [{n}, {o}, {oh}, {ow}] is too large")));
        }
        Ok(Shape::new(&[n, o, oh, ow]))
    }

    /// Run the backend's conv2d.
    fn conv2d_kernel(&self, weight: &Tensor, bias: Option<&Tensor>, p: &Conv2dParams) -> Result<Storage> {
        let backend = self.backend()?;
        match bias {
            Some(bt) => super::with_views(
                [self, weight, bt],
                [self.layout(), weight.layout(), bt.layout()],
                |[x, wv, bv]| backend.conv2d(x, wv, Some(bv), p),
            ),
            None => super::with_views([self, weight], [self.layout(), weight.layout()], |[x, wv]| {
                backend.conv2d(x, wv, None, p)
            }),
        }
    }

    /// [`Tensor::conv2d`] with per-axis parameters. Under [`crate::autocast`] (on a device that computes in
    /// `F16`) the conv runs in `F16` against the `F32` weight and bias (see the [`crate::autocast`] module docs).
    pub fn conv2d_with(&self, weight: &Tensor, bias: Option<&Tensor>, p: &Conv2dParams) -> Result<Tensor> {
        const OP: &str = "conv2d";
        if casts(self) || casts(weight) {
            return self.conv2d_autocast(weight, bias, p);
        }
        let shape = self.check_conv2d(weight, bias, p)?;
        let s = self.conv2d_kernel(weight, bias, p)?;
        let out = Tensor::wrap(s, shape, OP)?;
        let (x, wt, has_bias, p) = (self.clone(), weight.clone(), bias.is_some(), *p);
        let mut inputs: smallvec::SmallVec<[&Tensor; 3]> = smallvec::smallvec![self, weight];
        if let Some(b) = bias {
            inputs.push(b);
        }
        Ok(out.track(OP, &inputs, move |g| {
            // Both kernels read the gradient densely; materialise a broadcast/strided one once.
            let g = &g.contiguous()?;
            let backend = g.backend()?;
            let gx = if x.requires_grad() {
                let s = super::with_views([g, &wt], [g.layout(), wt.layout()], |[gv, wv]| {
                    backend.conv2d_backward_input(gv, wv, x.shape(), &p)
                })?;
                Some(Tensor::wrap(s, x.shape().clone(), "conv2d_backward_input")?)
            } else {
                None
            };
            let gw = if wt.requires_grad() {
                let s = super::with_views([&x, g], [x.layout(), g.layout()], |[xv, gv]| {
                    backend.conv2d_backward_weight(xv, gv, wt.shape(), &p)
                })?;
                Some(Tensor::wrap(s, wt.shape().clone(), "conv2d_backward_weight")?)
            } else {
                None
            };
            let mut gs = grads![gx, gw];
            if has_bias {
                gs.push(Some(g.sum([0, 2, 3], false)?));
            }
            Ok(gs)
        }))
    }

    /// The autocast conv: the input cast to `F16` (a recorded cast), `F16` copies of the weight and bias, an `F16`
    /// result, recorded against the weight and bias themselves. Backward: the input gradient in `F16`, the weight
    /// gradient written straight to the weight's dtype (`F32` master weights: f32 accumulation, no `F16` rounding
    /// or overflow in between), the bias gradient summed in `F32`.
    fn conv2d_autocast(&self, weight: &Tensor, bias: Option<&Tensor>, p: &Conv2dParams) -> Result<Tensor> {
        const OP: &str = "conv2d";
        let x = self.to_dtype(DType::F16)?;
        let (w16, b16) = {
            let _g = no_grad();
            (weight.to_dtype(DType::F16)?, bias.map(|b| b.to_dtype(DType::F16)).transpose()?)
        };
        let shape = x.check_conv2d(&w16, b16.as_ref(), p)?;
        let s = x.conv2d_kernel(&w16, b16.as_ref(), p)?;
        let out = Tensor::wrap(s, shape, OP)?;
        let (wt, bias_dtype, p) = (weight.clone(), bias.map(Tensor::dtype), *p);
        let mut inputs: smallvec::SmallVec<[&Tensor; 3]> = smallvec::smallvec![&x, weight];
        if let Some(b) = bias {
            inputs.push(b);
        }
        let xs = x.clone();
        Ok(out.track(OP, &inputs, move |g| {
            let g = &g.contiguous()?;
            let backend = g.backend()?;
            let gx = if xs.requires_grad() {
                let s = super::with_views([g, &w16], [g.layout(), w16.layout()], |[gv, wv]| {
                    backend.conv2d_backward_input(gv, wv, xs.shape(), &p)
                })?;
                Some(Tensor::wrap(s, xs.shape().clone(), "conv2d_backward_input")?)
            } else {
                None
            };
            let gw = if wt.requires_grad() {
                let s = super::with_views([&xs, g], [xs.layout(), g.layout()], |[xv, gv]| {
                    backend.conv2d_backward_weight_as(xv, gv, wt.shape(), &p, wt.dtype())
                })?;
                Some(Tensor::wrap(s, wt.shape().clone(), "conv2d_backward_weight")?)
            } else {
                None
            };
            let mut gs = grads![gx, gw];
            if let Some(bd) = bias_dtype {
                gs.push(Some(g.to_dtype(DType::F32)?.sum([0, 2, 3], false)?.to_dtype(bd)?));
            }
            Ok(gs)
        }))
    }

    fn check_pool(&self, op: &'static str, p: &Pool2dParams) -> Result<[usize; 2]> {
        self.check_float(op)?;
        let (_, _, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(op, self.shape(), "input must be NCHW"))?;
        if h == 0 || w == 0 {
            return Err(Error::invalid_shape(op, self.shape(), "empty spatial dimensions"));
        }
        if p.kernel.contains(&0) || p.stride.contains(&0) {
            return Err(Error::invalid_arg(op, format!("kernel and stride must be positive: {p:?}")));
        }
        if p.padding[0] > p.kernel[0] / 2 || p.padding[1] > p.kernel[1] / 2 {
            return Err(Error::invalid_arg(op, format!("padding must be at most half the kernel: {p:?}")));
        }
        if p.kernel[0].checked_mul(p.kernel[1]).is_none() {
            return Err(Error::invalid_arg(op, format!("window {p:?} is too large")));
        }
        p.output_hw(h, w).ok_or_else(|| {
            Error::invalid_arg(op, format!("window {p:?} does not fit input {}", self.shape()))
        })
    }

    /// Max pooling with a square `kernel`, `stride` and `padding` (padding never wins).
    pub fn max_pool2d(&self, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
        self.max_pool2d_with(&Pool2dParams {
            kernel: [kernel; 2],
            stride: [stride; 2],
            padding: [padding; 2],
        })
    }

    /// [`Tensor::max_pool2d`] with per-axis parameters.
    pub fn max_pool2d_with(&self, p: &Pool2dParams) -> Result<Tensor> {
        let [oh, ow] = self.check_pool("max_pool2d", p)?;
        let b = self.backend()?;
        let (n, c) = (self.dims()[0], self.dims()[1]);
        if !crate::is_grad_enabled() || !self.requires_grad() {
            // Nothing is recorded: the values alone (the same values, no index tensor).
            let vs = self.with_view(|v| b.max_pool2d_values(v, p))?;
            return Tensor::wrap(vs, Shape::new(&[n, c, oh, ow]), "max_pool2d");
        }
        let (vs, is) = self.with_view(|v| b.max_pool2d(v, p))?;
        let out = Tensor::wrap(vs, Shape::new(&[n, c, oh, ow]), "max_pool2d")?;
        let idx = Tensor::wrap(is, Shape::new(&[n, c, oh, ow]), "max_pool2d")?;
        let (in_shape, p) = (self.shape().clone(), *p);
        Ok(out.track_saved("max_pool2d", &[self], &[&idx.clone()], move |g| {
            let b = g.backend()?;
            let s = super::with_views([g, &idx], [g.layout(), idx.layout()], |[gv, iv]| {
                b.max_pool2d_backward(gv, iv, &in_shape, &p)
            })?;
            Ok(grads![Some(Tensor::wrap(s, in_shape.clone(), "max_pool2d_backward")?)])
        }))
    }

    /// Average pooling with a square `kernel`, `stride` and zero `padding` (padding counts in the divisor).
    pub fn avg_pool2d(&self, kernel: usize, stride: usize, padding: usize) -> Result<Tensor> {
        self.avg_pool2d_with(&Pool2dParams {
            kernel: [kernel; 2],
            stride: [stride; 2],
            padding: [padding; 2],
        })
    }

    /// [`Tensor::avg_pool2d`] with per-axis parameters.
    pub fn avg_pool2d_with(&self, p: &Pool2dParams) -> Result<Tensor> {
        let [oh, ow] = self.check_pool("avg_pool2d", p)?;
        let b = self.backend()?;
        let s = self.with_view(|v| b.avg_pool2d(v, p))?;
        let (n, c) = (self.dims()[0], self.dims()[1]);
        let out = Tensor::wrap(s, Shape::new(&[n, c, oh, ow]), "avg_pool2d")?;
        let (in_shape, p) = (self.shape().clone(), *p);
        Ok(out.track("avg_pool2d", &[self], move |g| {
            let b = g.backend()?;
            let s = g.with_view(|gv| b.avg_pool2d_backward(gv, &in_shape, &p))?;
            Ok(grads![Some(Tensor::wrap(s, in_shape.clone(), "avg_pool2d_backward")?)])
        }))
    }

    /// Adaptive average pooling to `[oh, ow]` (`[1, 1]` is global average pooling).
    pub fn adaptive_avg_pool2d(&self, out_hw: [usize; 2]) -> Result<Tensor> {
        const OP: &str = "adaptive_avg_pool2d";
        self.check_float(OP)?;
        let (n, c, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "input must be NCHW"))?;
        if out_hw.contains(&0) || h == 0 || w == 0 {
            return Err(Error::invalid_arg(OP, format!("output size {out_hw:?} for input {}", self.shape())));
        }
        let b = self.backend()?;
        let s = self.with_view(|v| b.adaptive_avg_pool2d(v, out_hw))?;
        let out = Tensor::wrap(s, Shape::new(&[n, c, out_hw[0], out_hw[1]]), OP)?;
        let in_shape = self.shape().clone();
        Ok(out.track(OP, &[self], move |g| {
            let b = g.backend()?;
            let s = g.with_view(|gv| b.adaptive_avg_pool2d_backward(gv, &in_shape))?;
            Ok(grads![Some(Tensor::wrap(s, in_shape.clone(), "adaptive_avg_pool2d_backward")?)])
        }))
    }

    /// Nearest-neighbour upsampling of NCHW by an integer factor on both axes.
    pub fn upsample_nearest2d(&self, scale: usize) -> Result<Tensor> {
        self.upsample_nearest2d_hw([scale, scale])
    }

    /// Nearest-neighbour upsampling by integer factors `[sh, sw]`.
    pub fn upsample_nearest2d_hw(&self, scale: [usize; 2]) -> Result<Tensor> {
        const OP: &str = "upsample_nearest2d";
        self.check_float(OP)?;
        let (n, c, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "input must be NCHW"))?;
        if scale.contains(&0) {
            return Err(Error::invalid_arg(OP, "scale must be positive"));
        }
        let (oh, ow) = (h.checked_mul(scale[0]), w.checked_mul(scale[1]));
        let Some((oh, ow)) =
            oh.zip(ow).filter(|&(oh, ow)| [c, oh, ow].iter().try_fold(n, |a, &b| a.checked_mul(b)).is_some())
        else {
            return Err(Error::invalid_arg(OP, format!("scale {scale:?} of {} is too large", self.shape())));
        };
        let b = self.backend()?;
        let s = self.with_view(|v| b.upsample_nearest2d(v, scale))?;
        let out = Tensor::wrap(s, Shape::new(&[n, c, oh, ow]), OP)?;
        let in_shape = self.shape().clone();
        Ok(out.track(OP, &[self], move |g| {
            let b = g.backend()?;
            let s = g.with_view(|gv: View<'_>| b.upsample_nearest2d_backward(gv, scale))?;
            Ok(grads![Some(Tensor::wrap(s, in_shape.clone(), "upsample_nearest2d_backward")?)])
        }))
    }
}
