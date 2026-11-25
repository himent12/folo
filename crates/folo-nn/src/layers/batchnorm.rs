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
        let (y, mean, var) = if fused {
            let (w, b) = (&self.weight, &self.bias);
            match cpu_act.unwrap_or(ACT_NONE) {
                ACT_SILU => train_cpu::<ACT_SILU>(x, w, b, self.eps)?,
                ACT_RELU => train_cpu::<ACT_RELU>(x, w, b, self.eps)?,
                _ => train_cpu::<ACT_NONE>(x, w, b, self.eps)?,
            }
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
        Ok((y, applied))
    }
}

impl Forward for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_act(x, Activation::Identity)
    }
}

/// The activations [`train_cpu`] applies itself.
const ACT_NONE: u8 = 0;
const ACT_SILU: u8 = 1;
const ACT_RELU: u8 = 2;

/// Activation `ACT` of `z`, exactly as the CPU's `Tensor::silu` / `Tensor::relu` compute it.
#[inline(always)]
fn act_fwd<const ACT: u8>(z: f32) -> f32 {
    match ACT {
        ACT_SILU => folo_core::cpu::silu(z),
        ACT_RELU => z.max(0.0),
        _ => z,
    }
}

/// `g · act'(z)`, exactly as the CPU backward of `Tensor::silu` / `Tensor::relu` computes it.
#[inline(always)]
fn act_bwd<const ACT: u8>(z: f32, g: f32) -> f32 {
    match ACT {
        ACT_SILU => folo_core::cpu::silu_backward(z, g),
        ACT_RELU => {
            if z > 0.0 {
                g
            } else {
                0.0
            }
        }
        _ => g,
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

/// Train-mode BatchNorm followed by activation `ACT` on the CPU: per-channel statistics in `f64` (one read of
/// x, [`plane_stats`]), then the output `act(z)`, `z = (x − mean) · inv · γ + β`, with a hand-written backward:
/// `dx = γ · inv / n · (n · g' − Σg' − x̂ · Σ(g' · x̂))`, `dγ = Σ(g' · x̂)`, `dβ = Σg'`, where
/// `g' = g · act'(z)` (`z` recomputed from `x`), `x̂ = (x − mean) · inv` and `inv = 1 / sqrt(var + eps)`.
/// The activation and its gradient run inside the BatchNorm passes, bit for bit the BatchNorm followed by the
/// separate activation op, without that op's passes over memory or a stored `z`. Returns the output and the
/// batch mean and biased variance, each `[C]`.
fn train_cpu<const ACT: u8>(
    x: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (nb, c, h, w) = x.dims4()?;
    let hw = h * w;
    let n = (nb * hw) as f64;
    let g = gamma.to_vec::<f32>()?;
    let bv = beta.to_vec::<f32>()?;
    // Channel-parallel fuses the two passes per channel while its `nb` planes stay hot in L1/L2; only use it
    // when there are enough channels to load-balance all threads and one channel fits comfortably in cache.
    const CHANNEL_CACHE_FLOATS: usize = 32_768;
    let channel_par =
        hw > 0 && c >= 4 * rayon::current_num_threads() && nb.saturating_mul(hw) <= CHANNEL_CACHE_FLOATS;
    // One read of x: each plane's sum and moments (`plane_stats`), combined per channel in image order: the
    // sums added (the mean is `Σ / n`), the moments merged with Chan's formula (the variance is `M2 / n`).
    // Deterministic for any thread count. When there are enough channels to saturate the thread pool, each
    // channel runs stats + normalise back-to-back so its `nb` planes stay hot in L2/L3; otherwise planes are
    // the parallel unit so a narrow layer still uses every thread.
    let (y, mean, inv, var, scale) = x.with_f32_slice(|xs| {
        // Every element is written below.
        let mut y = folo_core::cpu::take_buffer(xs.len());
        if channel_par {
            let ch_planes = split_planes_by_channel(&mut y, nb, c, hw);
            let ch_stats: Vec<(f32, f32, f32, f32)> = ch_planes
                .into_par_iter()
                .enumerate()
                .map(|(ch, mut planes)| {
                    let (g_ch, b_ch) = (g[ch], bv[ch]);
                    with_avx2(|| {
                        let mut sum = 0.0f64;
                        let mut mom = Moments::default();
                        for b in 0..nb {
                            let base = (b * c + ch) * hw;
                            let (ps, pm) = plane_stats(&xs[base..base + hw]);
                            sum += ps;
                            mom = Moments::merge(mom, pm);
                        }
                        let v = mom.m2 / n;
                        let inv_ch = (1.0 / (v + eps).sqrt()) as f32;
                        let mean_ch = (sum / n) as f32;
                        let scale_ch = inv_ch * g_ch;
                        for (b, o) in planes.iter_mut().enumerate() {
                            let base = (b * c + ch) * hw;
                            for (dst, &val) in o.iter_mut().zip(&xs[base..base + hw]) {
                                *dst = act_fwd::<ACT>((val - mean_ch) * scale_ch + b_ch);
                            }
                        }
                        (mean_ch, inv_ch, v as f32, scale_ch)
                    })
                })
                .collect();
            let mut mean = Vec::with_capacity(c);
            let mut inv = Vec::with_capacity(c);
            let mut var = Vec::with_capacity(c);
            let mut scale = Vec::with_capacity(c);
            for (m, i, v, s) in ch_stats {
                mean.push(m);
                inv.push(i);
                var.push(v);
                scale.push(s);
            }
            (y, mean, inv, var, scale)
        } else {
            let planes: Vec<(f64, Moments)> = if hw == 0 {
                vec![(0.0, Moments::default()); nb * c]
            } else {
                xs.par_chunks(hw).map(|p| with_avx2(|| plane_stats(p))).collect()
            };
            let stats: Vec<(f64, Moments)> = (0..c)
                .map(|ch| {
                    (0..nb)
                        .map(|b| planes[b * c + ch])
                        .fold((0.0, Moments::default()), |a, p| (a.0 + p.0, Moments::merge(a.1, p.1)))
                })
                .collect();
            let var_f64: Vec<f64> = stats.iter().map(|s| s.1.m2 / n).collect();
            let inv: Vec<f32> = var_f64.iter().map(|v| (1.0 / (v + eps).sqrt()) as f32).collect();
            let mean: Vec<f32> = stats.iter().map(|s| (s.0 / n) as f32).collect();
            let var: Vec<f32> = var_f64.iter().map(|&v| v as f32).collect();
            let scale: Vec<f32> = inv.iter().zip(&g).map(|(i, g)| i * g).collect();
            if hw > 0 {
                y.par_chunks_mut(hw).zip(xs.par_chunks(hw)).enumerate().for_each(|(p, (o, xp))| {
                    let ch = p % c;
                    let (m, s, b) = (mean[ch], scale[ch], bv[ch]);
                    with_avx2(|| {
                        for (o, &v) in o.iter_mut().zip(xp) {
                            *o = act_fwd::<ACT>((v - m) * s + b);
                        }
                    });
                });
            }
            (y, mean, inv, var, scale)
        }
    })?;
    let y = Tensor::from_vec(y, x.shape().clone())?;
    let (mean_t, var_t) = (Tensor::from_slice(&mean, [c])?, Tensor::from_vec(var, [c])?);
    let xd = x.detach();
    let shape = x.shape().clone();
    let name = match ACT {
        ACT_SILU => "batch_norm2d_silu",
        ACT_RELU => "batch_norm2d_relu",
        _ => "batch_norm2d",
    };
    // Per-channel sums of per-plane partial sums, planes added in image order.
    let per_channel = move |plane: &[(f64, f64)]| -> Vec<(f64, f64)> {
        (0..c)
            .map(|ch| (0..nb).map(|b| plane[b * c + ch]).fold((0.0, 0.0), |a, v| (a.0 + v.0, a.1 + v.1)))
            .collect()
    };
    let y = Tensor::custom_op(name, &[x, gamma, beta], y, move |gy| {
        let gy = gy.contiguous()?;
        let (dx, dg, db) = gy.with_f32_slice(|gs| {
            xd.with_f32_slice(|xs| {
                // Every element is written below. With an activation, the first pass computes each plane's
                // `g' = g · act'(z)` and reduces it in L1-resident tiles (`bwd_act_and_sum`); the second pass
                // turns it into dx in place. When `channel_par` holds, both passes run back-to-back per
                // channel so `xs` and `dx`/`gs` stay hot in L2/L3 across the two passes.
                let mut dx = folo_core::cpu::take_buffer(xs.len());
                let sums: Vec<(f64, f64)> = if channel_par {
                    let ch_planes = split_planes_by_channel(&mut dx, nb, c, hw);
                    ch_planes
                        .into_par_iter()
                        .enumerate()
                        .map(|(ch, mut planes)| {
                            let (m, i, s, b_val) = (mean[ch], inv[ch], scale[ch], bv[ch]);
                            let k = g[ch] * i / n as f32;
                            with_avx2(|| {
                                let (mut sum0, mut sum1) = (0.0f64, 0.0f64);
                                for (b, o) in planes.iter_mut().enumerate() {
                                    let base = (b * c + ch) * hw;
                                    let gp = &gs[base..base + hw];
                                    let xp = &xs[base..base + hw];
                                    let (p0, p1) = if ACT == ACT_NONE {
                                        sum2_f64(gp, xp, |gv, v| {
                                            (gv as f64, gv as f64 * ((v - m) * i) as f64)
                                        })
                                    } else {
                                        bwd_act_and_sum::<ACT>(o, gp, xp, m, i, s, b_val)
                                    };
                                    sum0 += p0;
                                    sum1 += p1;
                                }
                                let (sg, sgx) = (sum0 as f32, sum1 as f32);
                                for (b, o) in planes.iter_mut().enumerate() {
                                    let base = (b * c + ch) * hw;
                                    bwd_write_dx::<ACT>(
                                        o,
                                        &gs[base..base + hw],
                                        &xs[base..base + hw],
                                        m,
                                        i,
                                        k,
                                        n as f32,
                                        sg,
                                        sgx,
                                    );
                                }
                                (sum0, sum1)
                            })
                        })
                        .collect()
                } else {
                    let planes: Vec<(f64, f64)> = if hw == 0 {
                        vec![(0.0, 0.0); nb * c]
                    } else {
                        dx.par_chunks_mut(hw)
                            .zip(gs.par_chunks(hw))
                            .zip(xs.par_chunks(hw))
                            .enumerate()
                            .map(|(p, ((o, gp), xp))| {
                                let ch = p % c;
                                let (m, i, s, b_val) = (mean[ch], inv[ch], scale[ch], bv[ch]);
                                with_avx2(|| {
                                    if ACT == ACT_NONE {
                                        sum2_f64(gp, xp, |gv, v| {
                                            (gv as f64, gv as f64 * ((v - m) * i) as f64)
                                        })
                                    } else {
                                        bwd_act_and_sum::<ACT>(o, gp, xp, m, i, s, b_val)
                                    }
                                })
                            })
                            .collect()
                    };
                    let sums = per_channel(&planes);
                    if hw > 0 {
                        dx.par_chunks_mut(hw).enumerate().for_each(|(p, o)| {
                            let ch = p % c;
                            let (m, i) = (mean[ch], inv[ch]);
                            let k = g[ch] * i / n as f32;
                            let (sg, sgx) = (sums[ch].0 as f32, sums[ch].1 as f32);
                            let r = p * hw..(p + 1) * hw;
                            with_avx2(|| {
                                bwd_write_dx::<ACT>(o, &gs[r.clone()], &xs[r], m, i, k, n as f32, sg, sgx);
                            });
                        });
                    }
                    sums
                };
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

/// `f()` compiled for AVX2 when the CPU has it (detected at run time; the build targets baseline x86-64, whose
/// SSE2 loops are half as wide), else as is. Bit for bit the same result either way: the same IEEE operations
/// lane by lane, and Rust never fuses a multiply and an add on its own (FMA is not enabled here).
#[inline(always)]
fn with_avx2<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx2")]
        unsafe fn run<R>(f: impl FnOnce() -> R) -> R {
            f()
        }
        #[cfg(test)]
        let fallback = FORCE_FALLBACK.load(std::sync::atomic::Ordering::SeqCst);
        #[cfg(not(test))]
        let fallback = false;
        if !fallback && std::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (detected above).
            return unsafe { run(f) };
        }
    }
    f()
}

/// Test-only: [`with_avx2`] runs its closure as is, so a test can compare the two builds of the loops.
#[cfg(test)]
static FORCE_FALLBACK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Count, mean and sum of squared deviations (`M2`) of some values, in `f64`, as [`plane_stats`] makes them
/// and [`Moments::merge`] combines them. The mean here only serves the merge; the reported mean is a sum.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// Chan's parallel combination of two sets' moments: `M2 = M2a + M2b + d² · na · nb / n` with `d` the
    /// difference of the means, a sum of non-negative terms (no cancellation as in `Σv² − n·mean²`).
    fn merge(a: Moments, b: Moments) -> Moments {
        if b.n == 0.0 {
            return a;
        }
        if a.n == 0.0 {
            return b;
        }
        let n = a.n + b.n;
        let d = b.mean - a.mean;
        Moments { n, mean: a.mean + d * (b.n / n), m2: a.m2 + b.m2 + d * d * (a.n * b.n / n) }
    }
}

/// Floats per block of [`plane_stats`]: a multiple of 8 (the lanes of the plane sum), and 1 KiB, so a block
/// is still in L1 for its second loop.
const STAT_BLOCK: usize = 256;

/// One plane's sum and moments in one read, a block of [`STAT_BLOCK`] floats at a time, in `f64`.
///
/// - The **sum** is exactly the two-pass code's: 8 independent accumulators across the whole plane (each
///   block feeds them in turn, so every lane sees the same values in the same order), then the tail. Summed per
///   channel in image order and divided by `n`, it is bit for bit the old mean.
/// - **M2**: each block's own mean, then `Σ(v − mean_b)²` over the block while it is in L1, and the blocks
///   merged with Chan's formula. Every term is a square, so M2 cannot go negative, and a NaN or an infinity
///   in the plane makes it NaN (as the two-pass `Σ(v − mean)²` does).
#[inline]
fn plane_stats(p: &[f32]) -> (f64, Moments) {
    let mut acc = [0f64; 8];
    let mut m = Moments::default();
    let mut tail: &[f32] = &[];
    for blk in p.chunks(STAT_BLOCK) {
        let (chunks, rest) = blk.as_chunks::<8>();
        let mut bs = [0f64; 8];
        for ch in chunks {
            for i in 0..8 {
                acc[i] += ch[i] as f64;
                bs[i] += ch[i] as f64;
            }
        }
        let bn = blk.len() as f64;
        let mean = (bs.iter().sum::<f64>() + rest.iter().map(|&v| v as f64).sum::<f64>()) / bn;
        let mut sq = [0f64; 8];
        for ch in chunks {
            for i in 0..8 {
                let d = ch[i] as f64 - mean;
                sq[i] += d * d;
            }
        }
        let m2 =
            sq.iter().sum::<f64>() + rest.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>();
        m = Moments::merge(m, Moments { n: bn, mean, m2 });
        // Only the last block can have a rest (the blocks are whole lanes): the plane's tail.
        tail = rest;
    }
    let tail: f64 = tail.iter().map(|&v| v as f64).sum();
    (acc.iter().sum::<f64>() + tail, m)
}

/// Split a flat `[N, C, H, W]` buffer (`hw = H·W > 0`) into per-channel lists of `N` disjoint plane slices,
/// ordered by batch index `0..nb`.
fn split_planes_by_channel(buf: &mut [f32], nb: usize, c: usize, hw: usize) -> Vec<Vec<&mut [f32]>> {
    let mut by_ch: Vec<Vec<&mut [f32]>> = (0..c).map(|_| Vec::with_capacity(nb)).collect();
    for (p, plane) in buf.chunks_mut(hw).enumerate() {
        by_ch[p % c].push(plane);
    }
    by_ch
}

/// Floats per L1-resident backward tile: a multiple of 8 (AVX2 width) and 4 (`sum2_f64` lanes), 1 KiB per
/// slice so `o`, `gp` and `xp` (3 KiB total) stay in L1d between `act_bwd` and the reduction.
const BWD_BLOCK: usize = 256;
const _: () = assert!(BWD_BLOCK.is_multiple_of(4));

/// Compute `o = g · act'(z)` and accumulate `(Σ g', Σ g' · x̂)` in `f64` in [`BWD_BLOCK`]-float L1 tiles.
/// Because `BWD_BLOCK` is a multiple of 4, every 4-lane accumulator sees the same elements in the same order
/// as an unblocked pass followed by [`sum2_f64`], keeping the result bit-for-bit identical.
#[inline]
fn bwd_act_and_sum<const ACT: u8>(
    o: &mut [f32],
    gp: &[f32],
    xp: &[f32],
    m: f32,
    i: f32,
    s: f32,
    b: f32,
) -> (f64, f64) {
    let (mut s0, mut s1) = ([0f64; 4], [0f64; 4]);
    let mut tail_o = [0.0f32; 3];
    let mut tail_x = [0.0f32; 3];
    let mut tail_len = 0usize;
    for ((ob, gpb), xpb) in o.chunks_mut(BWD_BLOCK).zip(gp.chunks(BWD_BLOCK)).zip(xp.chunks(BWD_BLOCK)) {
        for ((dst, &gv), &v) in ob.iter_mut().zip(gpb).zip(xpb) {
            *dst = act_bwd::<ACT>((v - m) * s + b, gv);
        }
        let (co, ro) = ob.as_chunks::<4>();
        let (cx, rx) = xpb.as_chunks::<4>();
        for (g4, x4) in co.iter().zip(cx) {
            for k in 0..4 {
                let g = g4[k] as f64;
                s0[k] += g;
                s1[k] += g * ((x4[k] - m) * i) as f64;
            }
        }
        if !ro.is_empty() {
            tail_len = ro.len();
            tail_o[..tail_len].copy_from_slice(ro);
            tail_x[..tail_len].copy_from_slice(rx);
        }
    }
    let (mut t0, mut t1) = (s0.iter().sum::<f64>(), s1.iter().sum::<f64>());
    for k in 0..tail_len {
        let g = tail_o[k] as f64;
        t0 += g;
        t1 += g * ((tail_x[k] - m) * i) as f64;
    }
    (t0, t1)
}

/// Write `dx = k · (n · g' − Σg' − x̂ · Σ(g' · x̂))` for one plane (`g'` is `gp` when `ACT == ACT_NONE`,
/// else already stored in `o`).
#[inline]
#[allow(clippy::too_many_arguments)]
fn bwd_write_dx<const ACT: u8>(
    o: &mut [f32],
    gp: &[f32],
    xp: &[f32],
    m: f32,
    i: f32,
    k: f32,
    n: f32,
    sg: f32,
    sgx: f32,
) {
    if ACT == ACT_NONE {
        for ((dst, &gv), &v) in o.iter_mut().zip(gp).zip(xp) {
            *dst = k * (n * gv - sg - (v - m) * i * sgx);
        }
    } else {
        for (dst, &v) in o.iter_mut().zip(xp) {
            *dst = k * (n * *dst - sg - (v - m) * i * sgx);
        }
    }
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

    /// True if every floating buffer of `model` is finite. Buffers on a device are checked there with one
    /// readback in all (`Σ 0 · x` is 0 exactly when every `x` is finite).
    pub fn all_finite<M: Module + ?Sized>(model: &M) -> Result<bool> {
        let mut device = Vec::new();
        for (_, b) in model.named_buffers() {
            if b.dtype() != DType::F32 {
                continue;
            }
            if b.device().is_cpu() {
                if !all_finite(&b)? {
                    return Ok(false);
                }
            } else {
                device.push(b.flatten_all()?);
            }
        }
        if device.is_empty() {
            return Ok(true);
        }
        let _g = no_grad();
        let refs: Vec<&Tensor> = device.iter().collect();
        Ok(Tensor::cat(&refs, 0)?.mul_scalar(0.0)?.sum_all()?.item::<f32>()? == 0.0)
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
        let (y1, m1, v1) = train_cpu::<ACT_NONE>(&x, &g, &b, BN_EPS).unwrap();
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

    /// With SiLU or ReLU fused in, the CPU kernel is bit for bit the BatchNorm followed by the activation op:
    /// output, running statistics and the gradients of x, gamma and beta.
    #[test]
    fn fused_activation_is_bitwise_bn_then_act() {
        let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for act in [Activation::Silu, Activation::Relu] {
            for dims in [[3, 4, 5, 6], [2, 3, 1, 13], [4, 2, 8, 8], [2, 3, 33, 17], [2, 128, 19, 17]] {
                let mut rng = Rng::seed(11);
                let x = Tensor::randn(dims, 0.3, 1.5, &mut rng).requires_grad_();
                let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
                let run = |fused: bool| {
                    let bn = BatchNorm2d::new(dims[1]);
                    let mut rng = Rng::seed(5);
                    let _g = no_grad();
                    bn.weight.copy_from(&Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng)).unwrap();
                    bn.bias.copy_from(&Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng)).unwrap();
                    drop(_g);
                    let y = if fused {
                        bn.forward_act(&x, act).unwrap()
                    } else {
                        act.forward(&bn.forward(&x).unwrap()).unwrap()
                    };
                    let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                    let grads = [&x, &bn.weight, &bn.bias].map(|t| bits(gr.get(t).unwrap()));
                    (bits(&y), grads, bits(&bn.running_mean), bits(&bn.running_var))
                };
                assert_eq!(run(true), run(false), "{act:?} {dims:?}");
            }
        }
    }

    /// The one-read statistics: the mean is bit for bit the old two-pass code's (8-lane `f64` sums per plane,
    /// planes added in image order, `/ n`), and the variance matches a two-pass `f64` reference to f32
    /// precision, also where `Σv² − n·mean²` cancels: a mean 10⁴ or 10⁷ times the standard deviation, a
    /// constant channel, single-element planes, one outlier per plane, an extreme first value, a sum that
    /// needs every term. An infinity or a NaN gives a NaN variance, as before.
    #[test]
    fn one_pass_statistics_match_two_pass_reference() {
        type Case = ([usize; 4], fn(usize) -> f32);
        let cases: [Case; 10] = [
            ([3, 2, 7, 9], |i| 1.0e4 + ((i * 7919) % 101) as f32 * 1e-2),
            // f32 has 1 ulp at 10⁷: integers around 10⁷, spread under 1.
            ([2, 2, 16, 20], |i| 1.0e7 + ((i * 7919) % 3) as f32),
            ([2, 3, 4, 4], |i| if i % 48 < 16 { 3.25 } else { ((i * 31) % 17) as f32 - 8.0 }),
            ([5, 2, 1, 1], |i| (i as f32 * 0.37).sin() * 100.0),
            ([2, 2, 30, 30], |i| if i % 900 == 0 { 1.0e6 } else { (i % 13) as f32 * 0.1 }),
            ([1, 1, 200, 200], |i| if i == 0 { 1.0e6 } else { 0.1 }),
            ([1, 1, 2, 2], |i| [1.0e16, -1.0e16, 1.0, 1.0][i]),
            ([1, 1, 1, 2], |i| [0.0, f32::INFINITY][i]),
            ([1, 1, 1, 2], |i| [0.0, f32::NAN][i]),
            ([2, 1, 3, 100], |i| if i == 150 { f32::INFINITY } else { i as f32 }),
        ];
        for (dims, f) in cases {
            let [nb, c, h, w] = dims;
            let hw = h * w;
            let data: Vec<f32> = (0..nb * c * hw).map(f).collect();
            let x = Tensor::from_vec(data.clone(), dims).unwrap();
            let (g, b) = (Tensor::ones([c]), Tensor::zeros([c]));
            let (_, mean, var) = train_cpu::<ACT_NONE>(&x, &g, &b, BN_EPS).unwrap();
            let (mean, var) = (mean.to_vec::<f32>().unwrap(), var.to_vec::<f32>().unwrap());
            for ch in 0..c {
                let planes = || (0..nb).map(|i| &data[(i * c + ch) * hw..(i * c + ch + 1) * hw]);
                // The old mean, as origin/main computed it.
                let old_sum = |p: &[f32]| {
                    let mut acc = [0f64; 8];
                    let (chunks, rest) = p.as_chunks::<8>();
                    for chunk in chunks {
                        for i in 0..8 {
                            acc[i] += chunk[i] as f64;
                        }
                    }
                    let tail: f64 = rest.iter().map(|&v| v as f64).sum();
                    acc.iter().sum::<f64>() + tail
                };
                let n = (nb * hw) as f64;
                let old_mean = (planes().map(old_sum).fold(0.0, |a, s| a + s) / n) as f32;
                assert_eq!(
                    mean[ch].to_bits(),
                    old_mean.to_bits(),
                    "{dims:?} ch {ch}: mean {} vs {old_mean}",
                    mean[ch]
                );
                // Two-pass f64 reference of the variance.
                let vals: Vec<f64> = planes().flatten().map(|&v| v as f64).collect();
                let m = vals.iter().sum::<f64>() / n;
                let v = vals.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
                let close =
                    |got: f32, want: f64| (got as f64 - want).abs() <= 2.0 * f32::EPSILON as f64 * want.abs();
                assert!(
                    close(var[ch], v) || (v == 0.0 && var[ch] == 0.0) || (v.is_nan() && var[ch].is_nan()),
                    "{dims:?} ch {ch}: var {} vs {v}",
                    var[ch]
                );
                if dims == [2, 2, 16, 20] {
                    // The case bites: the naive formula is far off here.
                    let naive = vals.iter().map(|x| x * x).sum::<f64>() / n - m * m;
                    assert!(!close(naive as f32, v), "naive {naive} vs {v}");
                }
            }
        }
        // The values the review pinned: an extreme first value, and a mean that needs the small terms.
        let stats = |dims: [usize; 4], data: Vec<f32>| {
            let x = Tensor::from_vec(data, dims).unwrap();
            let (_, m, v) =
                train_cpu::<ACT_NONE>(&x, &Tensor::ones([1]), &Tensor::zeros([1]), BN_EPS).unwrap();
            (m.to_vec::<f32>().unwrap()[0], v.to_vec::<f32>().unwrap()[0])
        };
        // At this size the first draft (sums shifted by the plane's first value) was 132 f32 ulp off.
        let mut data = vec![0.1f32; 1000 * 1000];
        data[0] = 1.0e6;
        let (n, k, r) = (1.0e6f64, 1.0e6f64, 0.1f32 as f64);
        let want = (k - r) * (k - r) * (n - 1.0) / (n * n);
        let (_, v) = stats([1, 1, 1000, 1000], data);
        assert!((v as f64 - want).abs() <= f32::EPSILON as f64 * want, "var {v} vs {want}");
        assert_eq!(stats([1, 1, 2, 2], vec![1.0e16, -1.0e16, 1.0, 1.0]).0, 0.5);
        let (m, v) = stats([1, 1, 1, 2], vec![0.0, f32::INFINITY]);
        assert!(m == f32::INFINITY && v.is_nan(), "mean {m} var {v}");
        let (m, v) = stats([1, 1, 1, 2], vec![0.0, f32::NAN]);
        assert!(m.is_nan() && v.is_nan(), "mean {m} var {v}");
    }

    /// The AVX2 build of the per-plane loops and the plain one give the same bits: output, statistics and
    /// gradients of x, gamma and beta, with SiLU fused in. Planes of one value, of whole lanes, and of
    /// several statistics blocks with a ragged tail.
    #[test]
    fn avx2_and_fallback_are_bitwise_equal() {
        use std::sync::atomic::Ordering::SeqCst;
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                FORCE_FALLBACK.store(false, SeqCst);
            }
        }
        let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for dims in
            [[3, 2, 1, 1], [2, 3, 5, 7], [2, 2, 16, 16], [1, 2, 17, 19], [2, 3, 33, 17], [2, 128, 19, 17]]
        {
            let mut rng = Rng::seed(13);
            let x = Tensor::randn(dims, 0.3, 1.5, &mut rng).requires_grad_();
            let gamma = Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng).requires_grad_();
            let beta = Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng).requires_grad_();
            let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
            let run = |fallback: bool| {
                let _reset = Reset;
                FORCE_FALLBACK.store(fallback, SeqCst);
                let (y, m, v) = train_cpu::<ACT_SILU>(&x, &gamma, &beta, BN_EPS).unwrap();
                let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                let grads = [&x, &gamma, &beta].map(|t| bits(gr.get(t).unwrap()));
                (bits(&y), bits(&m), bits(&v), grads)
            };
            assert_eq!(run(false), run(true), "{dims:?}");
        }
    }

    /// Channel-parallel (`c >= 4 * threads`) and plane-parallel (`c < 4 * threads`) match bit for bit on
    /// forward output, batch stats, and all three gradients across all fused activations, `nb == 1`, and
    /// ragged multi-tile planes (`hw > BWD_BLOCK`, `hw % 4 != 0`).
    #[test]
    fn channel_par_matches_plane_par_bitwise() {
        let bits = |t: &Tensor| t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let p8 = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        for dims in [[2, 16, 19, 17], [1, 16, 17, 15]] {
            let mut rng = Rng::seed(29);
            let x = Tensor::randn(dims, 0.2, 1.4, &mut rng).requires_grad_();
            let gamma = Tensor::rand_uniform([dims[1]], 0.5, 1.5, &mut rng).requires_grad_();
            let beta = Tensor::rand_uniform([dims[1]], -0.5, 0.5, &mut rng).requires_grad_();
            let w = Tensor::randn(dims, 0.0, 1.0, &mut rng);
            let eval = |pool: &rayon::ThreadPool, act: u8| {
                pool.install(|| {
                    let (y, m, v) = match act {
                        ACT_SILU => train_cpu::<ACT_SILU>(&x, &gamma, &beta, BN_EPS).unwrap(),
                        ACT_RELU => train_cpu::<ACT_RELU>(&x, &gamma, &beta, BN_EPS).unwrap(),
                        _ => train_cpu::<ACT_NONE>(&x, &gamma, &beta, BN_EPS).unwrap(),
                    };
                    let gr = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                    let grads = [&x, &gamma, &beta].map(|t| bits(gr.get(t).unwrap()));
                    (bits(&y), bits(&m), bits(&v), grads)
                })
            };
            for act in [ACT_NONE, ACT_SILU, ACT_RELU] {
                assert_eq!(eval(&p1, act), eval(&p8, act), "{dims:?} act={act}");
            }
        }
    }
}
