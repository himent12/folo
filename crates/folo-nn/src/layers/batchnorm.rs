//! 2-D batch normalisation, and the running-statistics guard for non-finite batches.

use folo_core::{DType, Tensor, UnaryOp, no_grad};
use rayon::prelude::*;

use super::simple::Activation;
use crate::module::{Forward, Mode, Module, TensorKind};
use crate::{Error, Result};

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

impl BatchNorm2d {
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
        act.forward(&self.forward_plain(x)?)
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

    fn forward_plain(&self, x: &Tensor) -> Result<Tensor> {
        let (c, n) = self.check_input(x)?;
        let bshape = [1, c, 1, 1];
        if !self.mode.is_training() {
            let (scale, shift) = self.eval_scale_shift()?;
            return Ok(x.mul(&scale.reshape(bshape)?)?.add(&shift.reshape(bshape)?)?);
        }
        check_train_count(n, x)?;
        let fused =
            [x, &self.weight, &self.bias].iter().all(|t| t.device().is_cpu() && t.dtype() == DType::F32);
        let (y, mean, var) = if fused {
            train_cpu(x, &self.weight, &self.bias, self.eps)?
        } else {
            let mean = x.mean([0, 2, 3], true)?;
            let diff = x.sub(&mean)?;
            let var = diff.square()?.mean([0, 2, 3], true)?;
            let inv = var.add_scalar(self.eps as f32)?.rsqrt()?;
            let scale = inv.mul(&self.weight.reshape(bshape)?)?;
            let y = diff.mul(&scale)?.add(&self.bias.reshape(bshape)?)?;
            (y, mean.detach().reshape([c])?, var.detach().reshape([c])?)
        };

        let _g = no_grad();
        let m = self.momentum as f32;
        let unbias = (n as f64 / (n as f64 - 1.0)) as f32;
        self.running_mean.mul_scalar_assign(1.0 - m)?;
        self.running_mean.add_scaled_assign(&mean, m)?;
        self.running_var.mul_scalar_assign(1.0 - m)?;
        self.running_var.add_scaled_assign(&var, m * unbias)?;
        Ok(y)
    }
}

impl Forward for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_act(x, Activation::Identity)
    }
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

/// Train-mode BatchNorm on the CPU: per-channel statistics in `f64` (two passes), then the output, with a
/// hand-written backward: `dx = γ · inv / n · (n · g − Σg − x̂ · Σ(g · x̂))`, `dγ = Σ(g · x̂)`, `dβ = Σg`,
/// where `x̂ = (x − mean) · inv` and `inv = 1 / sqrt(var + eps)`. Returns the output and the batch mean and
/// biased variance, each `[C]`.
fn train_cpu(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
    let (nb, c, h, w) = x.dims4()?;
    let hw = h * w;
    let n = (nb * hw) as f64;
    // Per-channel sums of per-plane partial sums, planes added in image order (deterministic for any thread
    // count). Planes are the parallel unit, so a layer with few channels still uses every thread.
    let per_channel = move |plane: &[(f64, f64)]| -> Vec<(f64, f64)> {
        (0..c)
            .map(|ch| (0..nb).map(|b| plane[b * c + ch]).fold((0.0, 0.0), |a, v| (a.0 + v.0, a.1 + v.1)))
            .collect()
    };
    let (mean, inv, var) = x.with_f32_slice(|xs| {
        let sums: Vec<(f64, f64)> = if hw == 0 {
            vec![(0.0, 0.0); nb * c]
        } else {
            xs.par_chunks(hw).map(|p| (sum_f64(p, |v| v as f64), 0.0)).collect()
        };
        let mean: Vec<f64> = per_channel(&sums).iter().map(|s| s.0 / n).collect();
        let sq: Vec<(f64, f64)> = if hw == 0 {
            sums
        } else {
            xs.par_chunks(hw)
                .enumerate()
                .map(|(p, xp)| {
                    let m = mean[p % c];
                    (sum_f64(xp, |v| (v as f64 - m) * (v as f64 - m)), 0.0)
                })
                .collect()
        };
        let var: Vec<f64> = per_channel(&sq).iter().map(|s| s.0 / n).collect();
        let inv: Vec<f32> = var.iter().map(|v| (1.0 / (v + eps).sqrt()) as f32).collect();
        let mean: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        (mean, inv, var.iter().map(|&v| v as f32).collect::<Vec<f32>>())
    })?;
    let g = gamma.to_vec::<f32>()?;
    let scale: Vec<f32> = inv.iter().zip(&g).map(|(i, g)| i * g).collect();
    let bv = beta.to_vec::<f32>()?;
    let y = x.with_f32_slice(|xs| {
        // Every element is written below.
        let mut y = folo_core::cpu::take_buffer(xs.len());
        if hw > 0 {
            y.par_chunks_mut(hw).zip(xs.par_chunks(hw)).enumerate().for_each(|(p, (o, xp))| {
                let ch = p % c;
                let (m, s, b) = (mean[ch], scale[ch], bv[ch]);
                for (o, &v) in o.iter_mut().zip(xp) {
                    *o = (v - m) * s + b;
                }
            });
        }
        y
    })?;
    let y = Tensor::from_vec(y, x.shape().clone())?;
    let (mean_t, var_t) = (Tensor::from_slice(&mean, [c])?, Tensor::from_vec(var, [c])?);
    let xd = x.detach();
    let shape = x.shape().clone();
    let y = Tensor::custom_op("batch_norm2d", &[x, gamma, beta], y, move |gy| {
        let gy = gy.contiguous()?;
        let (dx, dg, db) = gy.with_f32_slice(|gs| {
            xd.with_f32_slice(|xs| {
                let planes: Vec<(f64, f64)> = if hw == 0 {
                    vec![(0.0, 0.0); nb * c]
                } else {
                    gs.par_chunks(hw)
                        .zip(xs.par_chunks(hw))
                        .enumerate()
                        .map(|(p, (gp, xp))| {
                            let (m, i) = (mean[p % c], inv[p % c]);
                            sum2_f64(gp, xp, |g, v| (g as f64, g as f64 * ((v - m) * i) as f64))
                        })
                        .collect()
                };
                let sums = per_channel(&planes);
                // Every element is written below.
                let mut dx = folo_core::cpu::take_buffer(xs.len());
                if hw > 0 {
                    dx.par_chunks_mut(hw).enumerate().for_each(|(p, o)| {
                        let ch = p % c;
                        let (m, i) = (mean[ch], inv[ch]);
                        let k = g[ch] * i / n as f32;
                        let (sg, sgx) = (sums[ch].0 as f32, sums[ch].1 as f32);
                        let r = p * hw..(p + 1) * hw;
                        for ((o, &gv), &v) in o.iter_mut().zip(&gs[r.clone()]).zip(&xs[r]) {
                            *o = k * (n as f32 * gv - sg - (v - m) * i * sgx);
                        }
                    });
                }
                let dg: Vec<f32> = sums.iter().map(|s| s.1 as f32).collect();
                let db: Vec<f32> = sums.iter().map(|s| s.0 as f32).collect();
                (dx, dg, db)
            })
        })??;
        Ok(vec![
            Some(Tensor::from_vec(dx, shape.clone())?),
            Some(Tensor::from_vec(dg, [c])?),
            Some(Tensor::from_vec(db, [c])?),
        ])
    })?;
    Ok((y, mean_t, var_t))
}

/// `sum(f(v))` in `f64` with 8 independent accumulators, so the loop vectorises.
#[inline]
fn sum_f64(xs: &[f32], f: impl Fn(f32) -> f64) -> f64 {
    let mut acc = [0f64; 8];
    let (chunks, rest) = xs.as_chunks::<8>();
    for ch in chunks {
        for i in 0..8 {
            acc[i] += f(ch[i]);
        }
    }
    let tail: f64 = rest.iter().map(|&v| f(v)).sum();
    acc.iter().sum::<f64>() + tail
}

/// The two sums `(sum f(a, b).0, sum f(a, b).1)` in `f64` with 4 independent accumulator pairs.
#[inline]
fn sum2_f64(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> (f64, f64)) -> (f64, f64) {
    let (mut s0, mut s1) = ([0f64; 4], [0f64; 4]);
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..4 {
            let (p, q) = f(x[i], y[i]);
            s0[i] += p;
            s1[i] += q;
        }
    }
    let (mut t0, mut t1) = (s0.iter().sum::<f64>(), s1.iter().sum::<f64>());
    for (&x, &y) in ra.iter().zip(rb) {
        let (p, q) = f(x, y);
        t0 += p;
        t1 += q;
    }
    (t0, t1)
}

/// A copy of every floating buffer of a model (the BatchNorm running statistics), taken before a train-mode
/// forward so a non-finite batch can be undone (docs/MODEL.md §8.3, "Non-finite batches").
///
/// ```
/// use folo_core::Tensor;
/// use folo_nn::{BatchNorm2d, Forward, RunningStats};
///
/// let bn = BatchNorm2d::new(2);
/// let snap = RunningStats::capture(&bn).unwrap();
/// bn.forward(&Tensor::full([2, 2, 2, 2], f32::NAN)).unwrap();
/// assert!(!RunningStats::all_finite(&bn).unwrap());
/// snap.restore().unwrap();
/// assert!(RunningStats::all_finite(&bn).unwrap());
/// ```
pub struct RunningStats {
    saved: Vec<(Tensor, Tensor)>,
}

impl RunningStats {
    /// Copy every floating buffer of `model`.
    pub fn capture<M: Module + ?Sized>(model: &M) -> Result<Self> {
        let _g = no_grad();
        let mut saved = Vec::new();
        for (_, b) in model.named_buffers() {
            if b.dtype().is_float() {
                let copy = Tensor::zeros_with(b.shape(), b.dtype(), b.device())?;
                copy.copy_from(&b)?;
                saved.push((b, copy));
            }
        }
        Ok(RunningStats { saved })
    }

    /// Write the copies back into the model's buffers.
    pub fn restore(&self) -> Result<()> {
        let _g = no_grad();
        for (live, copy) in &self.saved {
            live.copy_from(copy)?;
        }
        Ok(())
    }

    /// True if every floating buffer of `model` is finite.
    pub fn all_finite<M: Module + ?Sized>(model: &M) -> Result<bool> {
        for (_, b) in model.named_buffers() {
            if b.dtype() == DType::F32 && !all_finite(&b)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// True if every element of an `F32` tensor is finite.
fn all_finite(t: &Tensor) -> Result<bool> {
    if t.device().is_cpu() {
        return Ok(t.with_f32_slice(|s| s.iter().all(|v| v.is_finite()))?);
    }
    Ok(t.to_vec::<f32>()?.iter().all(|v| v.is_finite()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_core::rng::Rng;

    /// The fused CPU kernel and the composite path agree (forward, statistics and gradients).
    #[test]
    fn fused_matches_composite() {
        let mut rng = Rng::seed(9);
        let x = Tensor::randn([3, 4, 5, 6], 0.7, 2.0, &mut rng).requires_grad_();
        let g = Tensor::rand_uniform([4], 0.5, 1.5, &mut rng).requires_grad_();
        let b = Tensor::rand_uniform([4], -0.5, 0.5, &mut rng).requires_grad_();
        let w = Tensor::randn([3, 4, 5, 6], 0.0, 1.0, &mut rng);
        let (y1, m1, v1) = train_cpu(&x, &g, &b, BN_EPS).unwrap();
        let mean = x.mean([0, 2, 3], true).unwrap();
        let diff = x.sub(&mean).unwrap();
        let var = diff.square().unwrap().mean([0, 2, 3], true).unwrap();
        let inv = var.add_scalar(BN_EPS as f32).unwrap().rsqrt().unwrap();
        let scale = inv.mul(&g.reshape([1, 4, 1, 1]).unwrap()).unwrap();
        let y2 = diff.mul(&scale).unwrap().add(&b.reshape([1, 4, 1, 1]).unwrap()).unwrap();
        let near = |a: &Tensor, b: &Tensor, tol: f32| {
            let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
            let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(worst <= tol, "max diff {worst}");
        };
        near(&y1, &y2, 1e-5);
        near(&m1, &mean.reshape([4]).unwrap(), 1e-5);
        near(&v1, &var.reshape([4]).unwrap(), 1e-4);
        let g1 = y1.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        let g2 = y2.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
        for t in [&x, &g, &b] {
            near(g1.get(t).unwrap(), g2.get(t).unwrap(), 1e-4);
        }
    }
}
