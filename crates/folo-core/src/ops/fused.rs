//! Fused kernels behind the optional `Backend` hooks: Conv2d + bias + activation for inference, train-mode
//! BatchNorm + activation (forward and backward), and in-place optimizer updates. A backend without them
//! returns `NotSupported`; [`Tensor::conv2d_act`] then composes ordinary ops, and the other entry points
//! return `Ok(None)` / `Ok(false)` so the caller (folo-nn) can use its own path.

use super::{with_views, with_views_dyn};
use crate::autograd::is_grad_enabled;
use crate::backend::{Conv2dParams, OptimStep, RunningUpdate, UnaryOp};
use crate::{DType, Error, Result, Shape, Tensor};

/// What [`Tensor::batch_norm_act_train`] returns: the output and the per-channel statistics its backward needs.
#[derive(Debug, Clone)]
pub struct BatchNormTrain {
    /// `act((x − mean) · invstd · weight + bias)`, `[N, C, H, W]`.
    pub y: Tensor,
    /// Batch mean, `[C]`.
    pub mean: Tensor,
    /// `1 / sqrt(biased batch variance + eps)`, `[C]`.
    pub invstd: Tensor,
}

fn unsupported(e: &Error) -> bool {
    matches!(e, Error::NotSupported { .. })
}

/// The output shape of a valid conv2d, or `None` (the composite path then reports the exact error).
fn conv_out_shape(x: &Tensor, w: &Tensor, bias: Option<&Tensor>, p: &Conv2dParams) -> Option<Shape> {
    conv_out_shape_dims(x.dims4().ok()?, w, bias, p)
}

fn conv_out_shape_dims(
    (n, c, h, wi): (usize, usize, usize, usize),
    w: &Tensor,
    bias: Option<&Tensor>,
    p: &Conv2dParams,
) -> Option<Shape> {
    let (o, cg, kh, kw) = w.dims4().ok()?;
    let ok = p.groups > 0
        && !p.stride.contains(&0)
        && !p.dilation.contains(&0)
        && cg * p.groups == c
        && o % p.groups == 0
        && kh > 0
        && kw > 0
        && bias.is_none_or(|b| b.dims() == [o]);
    let [oh, ow] = if ok { p.output_hw(h, wi, kh, kw)? } else { return None };
    Some(Shape::new(&[n, o, oh, ow]))
}

impl Tensor {
    /// `act(conv2d(self, weight) + bias)` (`act = None` is the identity), the inference form of a fused
    /// Conv + BN + activation block. When nothing is being recorded (grad mode off, or no operand requires
    /// grad) and the backend has a fused kernel ([`crate::backend::Backend::conv2d_act`]), this is one kernel
    /// with bias and activation in the epilogue; otherwise it is [`Tensor::conv2d_with`] then
    /// [`Tensor::unary`], with autograd.
    pub fn conv2d_act(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Tensor> {
        let records = is_grad_enabled()
            && (self.requires_grad() || weight.requires_grad() || bias.is_some_and(Tensor::requires_grad));
        let (dev, dt) = (self.device(), self.dtype());
        let uniform = weight.device() == dev
            && bias.is_none_or(|b| b.device() == dev && b.dtype() == dt)
            && weight.dtype() == dt
            && self.check_float("conv2d_act").is_ok();
        if !records
            && uniform
            && let Some(shape) = conv_out_shape(self, weight, bias, p)
        {
            let b = self.backend()?;
            let r = match bias {
                Some(bt) => {
                    with_views([self, weight, bt], [self.layout(), weight.layout(), bt.layout()], |v| {
                        b.conv2d_act(v[0], v[1], Some(v[2]), p, act)
                    })
                }
                None => with_views([self, weight], [self.layout(), weight.layout()], |v| {
                    b.conv2d_act(v[0], v[1], None, p, act)
                }),
            };
            match r {
                Ok(s) => return Tensor::wrap(s, shape, "conv2d_act"),
                Err(e) if unsupported(&e) => {}
                Err(e) => return Err(e),
            }
        }
        let y = self.conv2d_with(weight, bias, p)?;
        match act {
            Some(op) => y.unary(op),
            None => Ok(y),
        }
    }

    /// [`Tensor::conv2d_act`] of `Tensor::cat(xs, 1)`. When nothing is being recorded and the backend has the
    /// kernel ([`crate::backend::Backend::conv2d_act_cat`], the CPU does), the conv reads each part in place and
    /// the concatenation is never written; otherwise this is `cat` then [`Tensor::conv2d_act`], with autograd.
    pub fn conv2d_act_cat(
        xs: &[&Tensor],
        weight: &Tensor,
        bias: Option<&Tensor>,
        p: &Conv2dParams,
        act: Option<UnaryOp>,
    ) -> Result<Tensor> {
        let first = *xs.first().ok_or_else(|| Error::invalid_arg("conv2d_act_cat", "no inputs given"))?;
        if xs.len() == 1 {
            return first.conv2d_act(weight, bias, p, act);
        }
        let records = is_grad_enabled()
            && (xs.iter().any(|x| x.requires_grad())
                || weight.requires_grad()
                || bias.is_some_and(Tensor::requires_grad));
        let (dev, dt) = (first.device(), first.dtype());
        let uniform = xs.iter().all(|x| x.device() == dev && x.dtype() == dt)
            && weight.device() == dev
            && weight.dtype() == dt
            && bias.is_none_or(|b| b.device() == dev && b.dtype() == dt)
            && first.check_float("conv2d_act_cat").is_ok();
        // Every part NCHW with the first one's N, H, W; the channel counts add up.
        let dims = xs.iter().map(|x| x.dims4().ok()).collect::<Option<Vec<_>>>();
        let cat_dims = dims.and_then(|d| {
            let (n, _, h, w) = d[0];
            let same = d.iter().all(|&(n2, _, h2, w2)| (n2, h2, w2) == (n, h, w));
            same.then(|| (n, d.iter().map(|x| x.1).sum(), h, w))
        });
        if !records
            && uniform
            && let Some(shape) = cat_dims.and_then(|d| conv_out_shape_dims(d, weight, bias, p))
        {
            let b = first.backend()?;
            let mut ts: Vec<&Tensor> = xs.to_vec();
            ts.push(weight);
            ts.extend(bias);
            let r = with_views_dyn(&ts, |v| {
                let k = xs.len();
                b.conv2d_act_cat(&v[..k], v[k], v.get(k + 1).copied(), p, act)
            });
            match r {
                Ok(s) => return Tensor::wrap(s, shape, "conv2d_act_cat"),
                Err(e) if unsupported(&e) => {}
                Err(e) => return Err(e),
            }
        }
        Tensor::cat(xs, 1)?.conv2d_act(weight, bias, p, act)
    }

    /// Fused train-mode BatchNorm over the channels of `self` (`[N, C, H, W]`, statistics over `N·H·W`)
    /// followed by `act` (`None` is the identity), on backends that have the kernels
    /// ([`crate::backend::Backend::batch_norm_act`]); `Ok(None)` on the others.
    ///
    /// `y = act((x − mean) · invstd · weight + bias)` with the batch mean and the biased variance. `running`
    /// is `(running_mean, running_var, momentum)`: both `[C]`, contiguous, updated in place (their storage
    /// version is bumped) with PyTorch's rule and the unbiased variance. The result is **not** recorded by
    /// autograd: wrap it with [`Tensor::custom_op`] and [`Tensor::batch_norm_act_train_backward`]. `self` (and `y`)
    /// may be `F16` on a backend that computes in it (mixed precision); the parameters, the running statistics and
    /// the returned `mean` / `invstd` are `F32`.
    pub fn batch_norm_act_train(
        &self,
        weight: &Tensor,
        bias: &Tensor,
        eps: f64,
        act: Option<UnaryOp>,
        running: Option<(&Tensor, &Tensor, f64)>,
    ) -> Result<Option<BatchNormTrain>> {
        const OP: &str = "batch_norm_act_train";
        let (n, c, h, w) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "input must be NCHW"))?;
        if n * h * w == 0 {
            return Err(Error::invalid_shape(OP, self.shape(), "needs at least one value per channel"));
        }
        // The data may be `F16` where the backend computes in it (mixed precision); parameters and statistics are
        // always `F32`.
        self.check_float(OP)?;
        let mut params = vec![weight, bias];
        if let Some((rm, rv, _)) = running {
            params.extend([rm, rv]);
        }
        for t in &params {
            t.check_f32(OP)?;
            self.check_same_device(t, OP)?;
            if t.dims() != [c] {
                return Err(Error::ShapeMismatch {
                    op: OP,
                    lhs: self.shape().clone(),
                    rhs: t.shape().clone(),
                });
            }
        }
        let b = self.backend()?;
        let r = with_views([self, weight, bias], [self.layout(), weight.layout(), bias.layout()], |v| {
            let Some((rm, rv, momentum)) = running else {
                return b.batch_norm_act(v[0], v[1], v[2], eps as f32, act, None);
            };
            for t in [rm, rv] {
                if t.layout().contiguous_range() != Some(0..c) {
                    return Err(Error::invalid_shape(OP, t.shape(), "running statistics must be contiguous"));
                }
                if [self, weight, bias].iter().any(|o| o.same_storage(t)) {
                    return Err(Error::invalid_arg(OP, "running statistics share storage with an input"));
                }
            }
            if rm.same_storage(rv) {
                return Err(Error::invalid_arg(OP, "running mean and variance share a storage"));
            }
            // Versions are bumped only if the backend may have written (not on `NotSupported`), so a
            // fallback leaves graphs that saved the running statistics intact.
            let (mut gm, mut gv) = (rm.cell().write_unversioned(), rv.cell().write_unversioned());
            let up = RunningUpdate { mean: &mut gm, var: &mut gv, momentum: momentum as f32 };
            let r = b.batch_norm_act(v[0], v[1], v[2], eps as f32, act, Some(up));
            if !matches!(&r, Err(e) if unsupported(e)) {
                rm.cell().mark_modified();
                rv.cell().mark_modified();
            }
            r
        });
        let (y, mean, invstd) = match r {
            Ok(s) => s,
            Err(e) if unsupported(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(Some(BatchNormTrain {
            y: Tensor::wrap(y, self.shape().clone(), OP)?,
            mean: Tensor::wrap(mean, Shape::new(&[c]), OP)?,
            invstd: Tensor::wrap(invstd, Shape::new(&[c]), OP)?,
        }))
    }

    /// The backward of [`Tensor::batch_norm_act_train`] for input `self`: `(dx, dweight, dbias)` from the
    /// output gradient `grad` and the forward's `mean` / `invstd`, or `Ok(None)` without a fused kernel.
    /// Not recorded by autograd (call it inside a `custom_op` backward).
    #[allow(clippy::too_many_arguments)]
    pub fn batch_norm_act_train_backward(
        &self,
        grad: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        mean: &Tensor,
        invstd: &Tensor,
        act: Option<UnaryOp>,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        const OP: &str = "batch_norm_act_train_backward";
        let (_, c, _, _) =
            self.dims4().map_err(|_| Error::invalid_shape(OP, self.shape(), "input must be NCHW"))?;
        if grad.shape() != self.shape() {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: self.shape().clone(),
                rhs: grad.shape().clone(),
            });
        }
        self.check_float(OP)?;
        grad.check_float_like(self, OP)?;
        for t in [weight, bias, mean, invstd] {
            t.check_f32(OP)?;
        }
        for t in [grad, weight, bias, mean, invstd] {
            self.check_same_device(t, OP)?;
        }
        for t in [weight, bias, mean, invstd] {
            if t.dims() != [c] {
                return Err(Error::ShapeMismatch {
                    op: OP,
                    lhs: self.shape().clone(),
                    rhs: t.shape().clone(),
                });
            }
        }
        let b = self.backend()?;
        let ts = [self, grad, weight, bias, mean, invstd];
        let r = with_views(ts, ts.map(Tensor::layout), |v| {
            b.batch_norm_act_backward(v[0], v[1], v[2], v[3], v[4], v[5], act)
        });
        let (dx, dw, db) = match r {
            Ok(s) => s,
            Err(e) if unsupported(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        let cs = Shape::new(&[c]);
        Ok(Some((
            Tensor::wrap(dx, self.shape().clone(), OP)?,
            Tensor::wrap(dw, cs.clone(), OP)?,
            Tensor::wrap(db, cs, OP)?,
        )))
    }

    /// One in-place optimizer update of this parameter from `grad` with a fused kernel
    /// ([`crate::backend::Backend::optimizer_step`]); `state` is the momentum buffer (SGD with momentum),
    /// `[exp_avg, exp_avg_sq]` (AdamW) or empty (`Axpby`). Returns `Ok(false)`, touching nothing, when the
    /// backend has no such kernel or the tensors do not fit it (not all `F32` on one device, a tensor not
    /// contiguous from the start of its storage, or two of them sharing a storage); the caller then uses
    /// tensor ops. Like the other in-place updates it refuses a tensor that requires grad outside `no_grad`
    /// and bumps the storage versions it writes.
    pub fn optimizer_step_(&self, grad: &Tensor, state: &[&Tensor], step: &OptimStep) -> Result<bool> {
        const OP: &str = "optimizer_step_";
        if is_grad_enabled() && (self.requires_grad() || state.iter().any(|t| t.requires_grad())) {
            return Err(Error::Autograd(format!(
                "{OP}: in-place update of a tensor that requires grad; run it under no_grad()"
            )));
        }
        let want = match step {
            OptimStep::Sgd { .. } => state.len() <= 1,
            OptimStep::AdamW { .. } => state.len() == 2,
            OptimStep::Axpby { .. } => state.is_empty(),
        };
        if !want {
            return Err(Error::invalid_arg(OP, format!("{} state tensors do not fit {step:?}", state.len())));
        }
        let n = self.numel();
        let dev = self.device();
        let fits = |t: &Tensor| {
            t.dtype() == DType::F32
                && t.device() == dev
                && t.shape() == self.shape()
                && t.layout().contiguous_range() == Some(0..n)
        };
        if grad.shape() != self.shape()
            || grad.dtype() != DType::F32
            || grad.device() != dev
            || !fits(self)
            || !state.iter().all(|t| fits(t))
        {
            return Ok(false);
        }
        let mut all: Vec<&Tensor> = vec![self, grad];
        all.extend_from_slice(state);
        for (i, a) in all.iter().enumerate() {
            if all[..i].iter().any(|b| a.same_storage(b)) {
                return Ok(false);
            }
        }
        let b = self.backend()?;
        let r = {
            let gr = grad.cell().read();
            let mut pw = self.cell().write_unversioned();
            let mut sw: Vec<_> = state.iter().map(|t| t.cell().write_unversioned()).collect();
            let mut sm: Vec<&mut crate::Storage> = sw.iter_mut().map(|g| &mut **g).collect();
            let r = b.optimizer_step(&mut pw, crate::backend::View::new(&gr, grad.layout()), &mut sm, step);
            // Bump versions only if the backend may have written, so a fallback keeps existing graphs valid.
            if !matches!(&r, Err(e) if unsupported(e)) {
                for t in std::iter::once(self).chain(state.iter().copied()) {
                    t.cell().mark_modified();
                }
            }
            r
        };
        match r {
            Ok(()) => Ok(true),
            Err(e) if unsupported(&e) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Mixed precision: `self *= factor` in place with one fused kernel
    /// ([`crate::backend::Backend::unscale_check`]), setting `flag` (a one-element `U32` tensor on the same
    /// device) to a nonzero value if any result is not finite; `flag` is never cleared, so one flag can collect
    /// several tensors. Returns `Ok(false)`, touching nothing, when the backend has no such kernel or the tensors
    /// do not fit it (`self` not `F32` or not contiguous from the start of its storage, `flag` not a contiguous
    /// one-element `U32`, the two sharing a storage); the caller then uses tensor ops. Like the other in-place
    /// updates it refuses a tensor that requires grad outside `no_grad` and bumps the versions it writes.
    pub fn unscale_check_(&self, factor: f32, flag: &Tensor) -> Result<bool> {
        const OP: &str = "unscale_check_";
        if is_grad_enabled() && (self.requires_grad() || flag.requires_grad()) {
            return Err(Error::Autograd(format!(
                "{OP}: in-place update of a tensor that requires grad; run it under no_grad()"
            )));
        }
        let fits = self.dtype() == DType::F32
            && self.layout().contiguous_range() == Some(0..self.numel())
            && flag.dtype() == DType::U32
            && flag.numel() == 1
            && flag.layout().contiguous_range() == Some(0..1)
            && flag.device() == self.device()
            && !self.same_storage(flag);
        if !fits {
            return Ok(false);
        }
        let b = self.backend()?;
        let r = {
            let mut xw = self.cell().write_unversioned();
            let mut fw = flag.cell().write_unversioned();
            let r = b.unscale_check(&mut xw, self.numel(), factor, &mut fw);
            if !matches!(&r, Err(e) if unsupported(e)) {
                self.cell().mark_modified();
                flag.cell().mark_modified();
            }
            r
        };
        match r {
            Ok(()) => Ok(true),
            Err(e) if unsupported(&e) => Ok(false),
            Err(e) => Err(e),
        }
    }
}
