//! Fused `LayerNorm`, `RMSNorm`, and `softmax_backward` hooks for [`GpuBackend`].

use folo_core::backend::View;
use folo_core::{DType as CoreDType, Error, Result, Storage};

use super::GpuBackend;
use super::convert::{Core, f32_only, kdt, split3};
use crate::kernels::{layer_norm, softmax_bwd};
use crate::profile;

/// Validates `[.., d]` norm input shape and optional `[d]` parameter shapes, returning `(outer, d)`.
fn check_norm_shape(x: &View<'_>, params: &[Option<&View<'_>>], op: &'static str) -> Result<(usize, usize)> {
    f32_only(x, op)?;
    let dims = x.shape().dims();
    let Some(&d) = dims.last() else {
        return Err(Error::InvalidShape { op, shape: x.shape().clone(), msg: "needs rank >= 1".into() });
    };
    if d == 0 {
        return Err(Error::InvalidShape {
            op,
            shape: x.shape().clone(),
            msg: "last dimension must be non-zero".into(),
        });
    }
    for p in params.iter().flatten() {
        f32_only(p, op)?;
        if p.shape().dims() != [d] {
            return Err(Error::ShapeMismatch {
                op,
                lhs: folo_core::Shape::new(&[d]),
                rhs: p.shape().clone(),
            });
        }
    }
    Ok((x.layout.numel() / d, d))
}

impl GpuBackend {
    pub(super) fn run_layer_norm(
        &self,
        x: View<'_>,
        weight: Option<View<'_>>,
        bias: Option<View<'_>>,
        eps: f32,
    ) -> Result<(Storage, Storage, Storage)> {
        let (outer, d) = check_norm_shape(&x, &[weight.as_ref(), bias.as_ref()], "layer_norm")?;
        let _p = profile::scope(|| format!("layer_norm {outer}x{d}"));
        let len = x.layout.numel();
        let (y, mean, rstd) = (
            self.alloc(CoreDType::F32, len)?,
            self.alloc(CoreDType::F32, outer)?,
            self.alloc(CoreDType::F32, outer)?,
        );
        if len > 0 {
            let xd = self.dense(&x)?;
            let wd = weight.as_ref().map(|w| self.dense(w)).transpose()?;
            let bd = bias.as_ref().map(|b| self.dense(b)).transpose()?;
            layer_norm::layer_norm_forward(
                &self.ctx,
                xd.raw(),
                wd.as_ref().map(|b| b.raw()),
                bd.as_ref().map(|b| b.raw()),
                y.raw(),
                mean.raw(),
                rstd.raw(),
                outer,
                d,
                eps,
            )
            .core()?;
        }
        Ok((
            self.wrap(CoreDType::F32, len, y),
            self.wrap(CoreDType::F32, outer, mean),
            self.wrap(CoreDType::F32, outer, rstd),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_layer_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        mean: View<'_>,
        rstd: View<'_>,
        need_weight: bool,
        need_bias: bool,
    ) -> Result<(Storage, Option<Storage>, Option<Storage>)> {
        let (outer, d) = check_norm_shape(&x, &[weight.as_ref()], "layer_norm_backward")?;
        f32_only(&grad, "layer_norm_backward")?;
        f32_only(&mean, "layer_norm_backward")?;
        f32_only(&rstd, "layer_norm_backward")?;
        if grad.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                op: "layer_norm_backward",
                lhs: x.shape().clone(),
                rhs: grad.shape().clone(),
            });
        }
        if mean.layout.numel() != outer || rstd.layout.numel() != outer {
            return Err(Error::InvalidShape {
                op: "layer_norm_backward",
                shape: mean.shape().clone(),
                msg: "stats length mismatch".into(),
            });
        }
        let _p = profile::scope(|| format!("layer_norm_bwd {outer}x{d}"));
        let len = x.layout.numel();
        let dx = self.alloc(CoreDType::F32, len)?;
        let dw = need_weight.then(|| self.alloc(CoreDType::F32, d)).transpose()?;
        let db = need_bias.then(|| self.alloc(CoreDType::F32, d)).transpose()?;
        let (xd, gd, md, rd) = (self.dense(&x)?, self.dense(&grad)?, self.dense(&mean)?, self.dense(&rstd)?);
        let wd = weight.as_ref().map(|w| self.dense(w)).transpose()?;
        layer_norm::layer_norm_backward(
            &self.ctx,
            xd.raw(),
            gd.raw(),
            wd.as_ref().map(|b| b.raw()),
            md.raw(),
            rd.raw(),
            dx.raw(),
            dw.as_ref().map(|b| b.raw()),
            db.as_ref().map(|b| b.raw()),
            outer,
            d,
        )
        .core()?;
        Ok((
            self.wrap(CoreDType::F32, len, dx),
            dw.map(|b| self.wrap(CoreDType::F32, d, b)),
            db.map(|b| self.wrap(CoreDType::F32, d, b)),
        ))
    }

    pub(super) fn run_rms_norm(
        &self,
        x: View<'_>,
        weight: Option<View<'_>>,
        eps: f32,
    ) -> Result<(Storage, Storage)> {
        let (outer, d) = check_norm_shape(&x, &[weight.as_ref()], "rms_norm")?;
        let _p = profile::scope(|| format!("rms_norm {outer}x{d}"));
        let len = x.layout.numel();
        let (y, rstd) = (self.alloc(CoreDType::F32, len)?, self.alloc(CoreDType::F32, outer)?);
        if len > 0 {
            let xd = self.dense(&x)?;
            let wd = weight.as_ref().map(|w| self.dense(w)).transpose()?;
            layer_norm::rms_norm_forward(
                &self.ctx,
                xd.raw(),
                wd.as_ref().map(|b| b.raw()),
                y.raw(),
                rstd.raw(),
                outer,
                d,
                eps,
            )
            .core()?;
        }
        Ok((self.wrap(CoreDType::F32, len, y), self.wrap(CoreDType::F32, outer, rstd)))
    }

    pub(super) fn run_rms_norm_backward(
        &self,
        x: View<'_>,
        grad: View<'_>,
        weight: Option<View<'_>>,
        rstd: View<'_>,
        need_weight: bool,
    ) -> Result<(Storage, Option<Storage>)> {
        let (outer, d) = check_norm_shape(&x, &[weight.as_ref()], "rms_norm_backward")?;
        f32_only(&grad, "rms_norm_backward")?;
        f32_only(&rstd, "rms_norm_backward")?;
        if grad.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                op: "rms_norm_backward",
                lhs: x.shape().clone(),
                rhs: grad.shape().clone(),
            });
        }
        if rstd.layout.numel() != outer {
            return Err(Error::InvalidShape {
                op: "rms_norm_backward",
                shape: rstd.shape().clone(),
                msg: "rstd length mismatch".into(),
            });
        }
        let _p = profile::scope(|| format!("rms_norm_bwd {outer}x{d}"));
        let len = x.layout.numel();
        let dx = self.alloc(CoreDType::F32, len)?;
        let dw = need_weight.then(|| self.alloc(CoreDType::F32, d)).transpose()?;
        let (xd, gd, rd) = (self.dense(&x)?, self.dense(&grad)?, self.dense(&rstd)?);
        let wd = weight.as_ref().map(|w| self.dense(w)).transpose()?;
        layer_norm::rms_norm_backward(
            &self.ctx,
            xd.raw(),
            gd.raw(),
            wd.as_ref().map(|b| b.raw()),
            rd.raw(),
            dx.raw(),
            dw.as_ref().map(|b| b.raw()),
            outer,
            d,
        )
        .core()?;
        Ok((self.wrap(CoreDType::F32, len, dx), dw.map(|b| self.wrap(CoreDType::F32, d, b))))
    }

    pub(super) fn run_softmax_backward(
        &self,
        y: View<'_>,
        grad: View<'_>,
        dim: usize,
        log: bool,
    ) -> Result<Storage> {
        let _p = profile::scope(|| format!("softmax_bwd {:?} dim {dim}", y.shape().dims()));
        let n = y.layout.numel();
        let dtype = y.storage.dtype();
        let out = self.alloc(dtype, n)?;
        if n > 0 {
            let (yd, gd) = (self.dense(&y)?, self.dense(&grad)?);
            let dims = split3(y.shape().dims(), dim);
            softmax_bwd::softmax_backward_dim(
                &self.ctx,
                yd.raw(),
                gd.raw(),
                out.raw(),
                dims,
                log,
                kdt(dtype),
            )
            .core()?;
        }
        Ok(self.wrap(dtype, n, out))
    }
}
