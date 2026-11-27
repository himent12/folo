//! 2-D batch normalisation, its fused CPU kernel, and the running-statistics guard for non-finite batches.
//!
//! - `mod.rs`: [`BatchNorm2d`], [`BN_EPS`], [`BN_MOMENTUM`], forward dispatch and eval affine helpers.
//! - `cpu.rs`: fused CPU training forward + backward kernel (one-read Chan moments, AVX2 + channel-parallel).
//! - `stats.rs`: [`RunningStats`], reusable buffer snapshots and device/host finiteness checks.

mod cpu;
mod stats;
#[cfg(test)]
mod tests;

pub use stats::{LiveFiniteCheck, RunningStats};

use folo_core::{DType, Tensor, UnaryOp, no_grad};

use super::simple::Activation;
use crate::module::{Forward, Mode, Module, TensorKind};
use crate::{Error, Result};
use cpu::{ACT_NONE, ACT_RELU, ACT_SILU, train_cpu_raw};

/// docs/MODEL.md §2.1: `eps = 1e-3`.
pub const BN_EPS: f64 = 1e-3;
/// docs/MODEL.md §2.1: `momentum = 0.03` (PyTorch convention, the weight of the new batch statistic).
pub const BN_MOMENTUM: f64 = 0.03;

/// Batch normalisation over the channels of `NCHW` input (docs/MODEL.md §2.1).
///
/// - **Training**: normalise with the batch mean and the *biased* batch variance over `N·H·W`, then update
///   the running statistics under `no_grad`: `running = (1 − momentum) · running + momentum · stat`, with the
///   *unbiased* variance (`· n / (n − 1)`). A batch with `N·H·W < 2` is an error. On the CPU this is one fused
///   kernel with a hand-written backward (statistics accumulated in `f64`). On a device whose backend has
///   the fused BatchNorm kernels (folo-gpu) it is those, with the running update inside the kernel and the
///   following activation fused in when called through [`BatchNorm2d::forward_act`]; other devices compose
///   tensor ops.
/// - **Evaluation**: normalise with the running statistics.
///
/// Tensor names: `weight` (γ), `bias` (β), `running_mean`, `running_var` (buffers). `num_batches_tracked` is
/// not kept: the momentum update does not need it, and `.folo` files do not store it.
pub struct BatchNorm2d {
    /// γ, `[C]`, initialised to 1.
    pub weight: Tensor,
    /// β, `[C]`, initialised to 0.
    pub bias: Tensor,
    /// Running mean, `[C]`, initialised to 0.
    pub running_mean: Tensor,
    /// Running (unbiased) variance, `[C]`, initialised to 1.
    pub running_var: Tensor,
    /// Added to the variance before the square root.
    pub eps: f64,
    /// Weight of the batch statistic in the running update.
    pub momentum: f64,
    mode: Mode,
}

impl BatchNorm2d {
    /// `channels` with docs/MODEL.md's `eps = 1e-3` and `momentum = 0.03`, in training mode.
    pub fn new(channels: usize) -> Self {
        Self::with_options(channels, BN_EPS, BN_MOMENTUM)
    }

    /// `channels` with an explicit `eps` and `momentum`.
    pub fn with_options(channels: usize, eps: f64, momentum: f64) -> Self {
        BatchNorm2d {
            weight: Tensor::ones([channels]).requires_grad_(),
            bias: Tensor::zeros([channels]).requires_grad_(),
            running_mean: Tensor::zeros([channels]),
            running_var: Tensor::ones([channels]),
            eps,
            momentum,
            mode: Mode::new(),
        }
    }

    /// A BatchNorm over existing tensors (all `[C]`, `F32`).
    pub fn from_tensors(
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let c = weight.numel();
        for (name, t) in [
            ("weight", &weight),
            ("bias", &bias),
            ("running_mean", &running_mean),
            ("running_var", &running_var),
        ] {
            if t.dims() != [c] || t.dtype() != DType::F32 {
                return Err(Error::Invalid(format!(
                    "BatchNorm2d: {name} is {:?} {}, expected [{c}] f32",
                    t.dims(),
                    t.dtype()
                )));
            }
        }
        Ok(BatchNorm2d { weight, bias, running_mean, running_var, eps, momentum, mode: Mode::new() })
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.weight.numel()
    }

    fn check_input(&self, x: &Tensor) -> Result<(usize, usize)> {
        let (n, c, h, w) = x
            .dims4()
            .map_err(|_| Error::Invalid(format!("BatchNorm2d: input must be NCHW, got {:?}", x.dims())))?;
        if c != self.channels() {
            return Err(Error::Invalid(format!(
                "BatchNorm2d: input has {c} channels, layer has {}",
                self.channels()
            )));
        }
        Ok((c, n * h * w))
    }

    /// Per-channel `(scale, shift)` for evaluation: `scale = γ / sqrt(running_var + eps)`,
    /// `shift = β − running_mean · scale`. Gradients flow to γ and β.
    pub fn eval_scale_shift(&self) -> Result<(Tensor, Tensor)> {
        let inv = self.running_var.add_scalar(self.eps as f32)?.rsqrt()?;
        let scale = self.weight.mul(&inv)?;
        let shift = self.bias.sub(&self.running_mean.mul(&scale)?)?;
        Ok((scale, shift))
    }

    /// `act(BatchNorm(x))`. In training mode on a device with fused kernels (see the type docs) the
    /// normalisation, the running-statistics update and `act` (SiLU, ReLU or none; others run separately)
    /// are one kernel set, recorded as one [`Tensor::custom_op`] whose hand-written backward also runs fused.
    /// Everywhere else it is the BatchNorm forward followed by `act`.
    pub fn forward_act(&self, x: &Tensor, act: Activation) -> Result<Tensor> {
        if self.mode.is_training() && !x.device().is_cpu() {
            let (_, n) = self.check_input(x)?;
            check_train_count(n, x)?;
            if let Some(y) = self.train_fused(x, act.unary_op())? {
                return Ok(y);
            }
            if act.unary_op().is_some()
                && let Some(y) = self.train_fused(x, None)?
            {
                return act.forward(&y);
            }
        }
        let (y, applied) = self.forward_plain(x, act)?;
        if applied { Ok(y) } else { act.forward(&y) }
    }

    /// The fused device kernels, or `None` when the backend lacks them (or this activation).
    fn train_fused(&self, x: &Tensor, act: Option<UnaryOp>) -> Result<Option<Tensor>> {
        let running = Some((&self.running_mean, &self.running_var, self.momentum));
        let Some(out) = x.batch_norm_act_train(&self.weight, &self.bias, self.eps, act, running)? else {
            return Ok(None);
        };
        let (xd, w, b) = (x.detach(), self.weight.detach(), self.bias.detach());
        let (mean, invstd) = (out.mean, out.invstd);
        let y = Tensor::custom_op("batch_norm2d_act", &[x, &self.weight, &self.bias], out.y, move |g| {
            let (dx, dw, db) = xd.batch_norm_act_train_backward(g, &w, &b, &mean, &invstd, act)?.ok_or(
                folo_core::Error::NotSupported { op: "batch_norm_act_train_backward", backend: "?" },
            )?;
            Ok(vec![Some(dx), Some(dw), Some(db)])
        })?;
        Ok(Some(y))
    }

    /// The BatchNorm forward, and whether `act` has been applied too (by the fused CPU training kernel, for
    /// the identity, SiLU and ReLU).
    fn forward_plain(&self, x: &Tensor, act: Activation) -> Result<(Tensor, bool)> {
        let (c, n) = self.check_input(x)?;
        let bshape = [1, c, 1, 1];
        if !self.mode.is_training() {
            let (scale, shift) = self.eval_scale_shift()?;
            return Ok((x.mul(&scale.reshape(bshape)?)?.add(&shift.reshape(bshape)?)?, false));
        }
        check_train_count(n, x)?;
        let fused =
            [x, &self.weight, &self.bias].iter().all(|t| t.device().is_cpu() && t.dtype() == DType::F32);
        let cpu_act = match act {
            Activation::Identity => Some(ACT_NONE),
            Activation::Silu => Some(ACT_SILU),
            Activation::Relu => Some(ACT_RELU),
            _ => None,
        };
        let applied = fused && cpu_act.is_some();
        let m = self.momentum as f32;
        let keep = 1.0 - m;
        let unbias = (n as f64 / (n as f64 - 1.0)) as f32;
        if fused {
            let (w, b) = (&self.weight, &self.bias);
            let (y, mean, var) = match cpu_act.unwrap_or(ACT_NONE) {
                ACT_SILU => train_cpu_raw::<ACT_SILU>(x, w, b, self.eps)?,
                ACT_RELU => train_cpu_raw::<ACT_RELU>(x, w, b, self.eps)?,
                _ => train_cpu_raw::<ACT_NONE>(x, w, b, self.eps)?,
            };
            let _g = no_grad();
            update_running_cpu_or_tensor(&self.running_mean, &mean, keep, m)?;
            update_running_cpu_or_tensor(&self.running_var, &var, keep, m * unbias)?;
            return Ok((y, applied));
        }

        let mean = x.mean([0, 2, 3], true)?;
        let diff = x.sub(&mean)?;
        let var = diff.square()?.mean([0, 2, 3], true)?;
        let inv = var.add_scalar(self.eps as f32)?.rsqrt()?;
        let scale = inv.mul(&self.weight.reshape(bshape)?)?;
        let y = diff.mul(&scale)?.add(&self.bias.reshape(bshape)?)?;
        let (mean, var) = (mean.detach().reshape([c])?, var.detach().reshape([c])?);

        let _g = no_grad();
        self.running_mean.mul_scalar_assign(keep)?;
        self.running_mean.add_scaled_assign(&mean, m)?;
        self.running_var.mul_scalar_assign(keep)?;
        self.running_var.add_scaled_assign(&var, m * unbias)?;
        Ok((y, applied))
    }
}

impl Module for BatchNorm2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("weight", &self.weight, TensorKind::Parameter);
        f("bias", &self.bias, TensorKind::Parameter);
        f("running_mean", &self.running_mean, TensorKind::Buffer);
        f("running_var", &self.running_var, TensorKind::Buffer);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl Forward for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_act(x, Activation::Identity)
    }
}

/// Update a running buffer in place: `dst[i] = dst[i] * keep + stat[i] * alpha`, matching the exact two-step
/// rounding of `mul_scalar_assign(keep)` followed by `add_scaled_assign(&stat, alpha)`. Avoids wrapping `stat`
/// in a temporary `Tensor` when `dst` is a contiguous CPU `F32` buffer.
fn update_running_cpu_or_tensor(dst: &Tensor, stat: &[f32], keep: f32, alpha: f32) -> Result<()> {
    if dst.device().is_cpu() && dst.dtype() == DType::F32 && dst.is_contiguous() && dst.dims() == [stat.len()]
    {
        dst.update_f32(|d| {
            for (r, &s) in d.iter_mut().zip(stat) {
                *r = *r * keep + alpha * s;
            }
        })?;
        return Ok(());
    }
    let t = Tensor::from_slice(stat, [stat.len()])?.to_device(dst.device())?;
    dst.mul_scalar_assign(keep)?;
    dst.add_scaled_assign(&t, alpha)?;
    Ok(())
}

fn check_train_count(n: usize, x: &Tensor) -> Result<()> {
    if n < 2 {
        return Err(Error::Invalid(format!(
            "BatchNorm2d: training needs at least 2 values per channel, got N·H·W = {n} (input {:?})",
            x.dims()
        )));
    }
    Ok(())
}
