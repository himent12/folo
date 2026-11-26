//! Spectral Normalization (Miyato et al., ICLR 2018): deterministic power-iteration estimation of the top
//! singular value `sigma(W)` and differentiable weight normalization `W_SN = W / (sigma(W) + eps)`.
//!
//! # Differentiable Forward / Backward Semantics
//!
//! Power iteration updates unit singular vectors `u` (`[out]`) and `v` (`[in]`) under `no_grad`, then forms
//! the Rayleigh quotient `sigma = u^T M v` as a differentiable `F32` tensor op with `u` and `v` detached and
//! returns `W_SN = W / (sigma + eps)`. By the Envelope Theorem (Miyato et al. 2018 Eq. 11), at the converged
//! top singular pair `(u*, v*)` the implicit derivatives `dsigma/du` and `dsigma/dv` vanish on the unit
//! spheres, so autograd yields the exact spectral-norm gradient:
//!
//! ```text
//! dL/dW = (1 / (sigma + eps)) * (dL/dW_SN - <dL/dW_SN, W_SN> * u v^T)
//! ```
//!
//! In training mode [`SpectralNorm`] advances `(u, v)` by `n_power_iterations` steps on each forward pass,
//! feeding fresh detached `(u_fresh, v_fresh)` tensors into the autograd graph while updating the persistent
//! buffers `(self.u, self.v)` separately so multiple training forwards can be summed before a single
//! `backward()` without triggering in-place version conflicts (`ModifiedInPlace`). In eval mode `(u, v)` are
//! frozen so inference never mutates buffers.

use folo_core::rng::Rng;
use folo_core::{DType, Tensor, no_grad};

use crate::layers::{Conv2d, Linear};
use crate::module::{Forward, Mode, Module, TensorKind};
use crate::{Error, Result};

/// Default numerical stability epsilon for spectral normalization (`1e-12`, matching Miyato et al. 2018).
pub const SPECTRAL_NORM_EPS: f32 = 1e-12;

/// Output of [`power_iteration`].
#[derive(Debug, Clone)]
pub struct PowerIterationResult {
    /// Estimated top singular value `sigma = u^T M v >= 0` (`NaN` if `weight` contains non-finite entries).
    pub sigma: f32,
    /// Updated detached left singular vector `u` (`[rows]`, unit L2 norm; preserved on zero matrices so
    /// subsequent non-zero updates recover immediately).
    pub u: Tensor,
    /// Updated detached right singular vector `v` (`[cols]`, unit L2 norm unless matrix is zero).
    pub v: Tensor,
}

/// Flatten a rank >= 2 `F32` weight tensor into `[rows, cols]` where `rows = dims[0]` and
/// `cols = numel / rows`.
fn weight_as_matrix(weight: &Tensor, op: &str) -> Result<(Tensor, usize, usize)> {
    if weight.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("{op}: weight must be f32, got {}", weight.dtype())));
    }
    let dims = weight.dims();
    if dims.len() < 2 {
        return Err(Error::Invalid(format!("{op}: weight must have rank >= 2, got shape {dims:?}")));
    }
    let rows = dims[0];
    let cols: usize = dims[1..].iter().product();
    if rows == 0 || cols == 0 {
        return Err(Error::Invalid(format!("{op}: weight must be non-empty, got shape {dims:?}")));
    }
    let mat = weight.reshape([rows, cols])?;
    Ok((mat, rows, cols))
}

/// Normalize a 1-D `F32` vector `x` to unit L2 norm under `no_grad`, returning `(normalized, is_zero)` where
/// `is_zero` is a scalar `0.0`/`1.0` tensor on `x.device()` (true iff every element is finite `0.0`).
///
/// Rescales by `max(|x_i|)` on both CPU (`f64` accumulation) and non-CPU devices so normalization is
/// scale-invariant across `1e-35..1e35` without squaring overflow/underflow, while preserving `NaN` / `±Inf`
/// poison propagation identically on CPU and GPU.
fn l2_normalize_vec(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let _g = no_grad();
    let dev = x.device();
    if dev.is_cpu() {
        let (out, zero_flag) = x.with_f32_slice(|s| {
            let mut max_abs = 0.0f64;
            for &v in s {
                if !v.is_finite() {
                    return (vec![f32::NAN; s.len()], 0.0f32);
                }
                max_abs = max_abs.max((v as f64).abs());
            }
            if max_abs == 0.0 {
                return (vec![0.0f32; s.len()], 1.0f32);
            }
            let inv_m = 1.0 / max_abs;
            let ss: f64 = s.iter().map(|&v| ((v as f64) * inv_m).powi(2)).sum();
            let norm = max_abs * ss.sqrt();
            let inv = 1.0 / norm;
            (s.iter().map(|&v| ((v as f64) * inv) as f32).collect(), 0.0f32)
        })?;
        return Ok((Tensor::from_vec(out, x.shape())?, Tensor::scalar(zero_flag)));
    }
    // Non-CPU device path: rescale by `max_abs` before squaring and detect exact zero via `norm == 0.0`
    // (which is `false` for `NaN`, guaranteeing `NaN` / `±Inf` poison propagation with zero host syncs).
    let max_abs = x.abs()?.max_all()?;
    let one = Tensor::scalar(1.0).to_device(dev)?;
    let max_is_zero = max_abs.eq_scalar(0.0)?;
    let safe_scale = max_is_zero.where_cond(&one, &max_abs)?;
    let scaled = x.div(&safe_scale)?;
    let norm = scaled.square()?.sum_all()?.sqrt()?;
    let norm_is_zero = norm.eq_scalar(0.0)?;
    let safe_norm = norm_is_zero.where_cond(&one, &norm)?;
    Ok((scaled.div(&safe_norm)?, norm_is_zero))
}

/// Deterministically initialize a unit vector `[len]` on `device` from `N(0, 1)` using `seed`.
fn init_unit_vec(len: usize, device: folo_core::Device, seed: u64) -> Result<Tensor> {
    let mut rng = Rng::seed(seed);
    let vals: Vec<f32> = (0..len).map(|_| rng.normal()).collect();
    let v = Tensor::from_vec(vals, [len])?.to_device(device)?;
    let (normed, is_zero) = l2_normalize_vec(&v)?;
    if is_zero.item::<f32>()? != 0.0 {
        let mut e0 = vec![0.0f32; len];
        e0[0] = 1.0;
        return Ok(Tensor::from_vec(e0, [len])?.to_device(device)?);
    }
    Ok(normed)
}

/// Compute `M^T u` (`[cols]`) in `F32` (never downcast to `F16` under GPU `autocast`).
fn mat_t_vec_f32(mat: &Tensor, mat_t: &Tensor, u: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    if mat.device().is_cpu() && folo_core::autocast_dtype().is_none() {
        Ok(mat_t.matmul(&u.reshape([rows, 1])?)?.reshape([cols])?)
    } else {
        Ok(mat.mul(&u.reshape([rows, 1])?)?.sum(0, false)?)
    }
}

/// Compute `M v` (`[rows]`) in `F32` (never downcast to `F16` under GPU `autocast`).
fn mat_vec_f32(mat: &Tensor, v: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    if mat.device().is_cpu() && folo_core::autocast_dtype().is_none() {
        Ok(mat.matmul(&v.reshape([cols, 1])?)?.reshape([rows])?)
    } else {
        Ok(mat.mul(&v.reshape([1, cols])?)?.sum(1, false)?)
    }
}

/// Internal power-iteration returning fresh detached `(u, v)` without forcing a host sync for `sigma`.
///
/// If `M v == 0` (e.g., exact zero matrix or null-space vector), preserves the previous unit `u` instead of
/// collapsing `u` to zero so subsequent non-zero weight updates recover immediately. If `u_init` itself is
/// all-zero and `fallback_u` is provided, substitutes `fallback_u` before iterating.
fn power_iteration_uv(
    weight: &Tensor,
    u_init: &Tensor,
    n_iters: usize,
    fallback_u: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    if n_iters == 0 {
        return Err(Error::Invalid("power_iteration: n_iters must be >= 1".into()));
    }
    let (mat, rows, cols) = weight_as_matrix(&weight.detach(), "power_iteration")?;
    if u_init.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("power_iteration: u_init must be f32, got {}", u_init.dtype())));
    }
    if u_init.dims() != [rows] {
        return Err(Error::Invalid(format!(
            "power_iteration: u_init shape {:?} does not match weight rows [{rows}]",
            u_init.dims()
        )));
    }
    if u_init.device() != weight.device() {
        return Err(Error::Invalid(format!(
            "power_iteration: u_init on {:?} != weight on {:?}",
            u_init.device(),
            weight.device()
        )));
    }

    let _g = no_grad();
    let mat_t = mat.t()?;
    let (mut u, u_zero) = l2_normalize_vec(&u_init.detach())?;
    if let Some(fb) = fallback_u {
        u = u_zero.where_cond(fb, &u)?;
    }
    let mut v = Tensor::zeros_with([cols], DType::F32, weight.device())?;
    for _ in 0..n_iters {
        let mv_t = mat_t_vec_f32(&mat, &mat_t, &u, rows, cols)?;
        let (v_next, _) = l2_normalize_vec(&mv_t)?;
        v = v_next;
        let mv = mat_vec_f32(&mat, &v, rows, cols)?;
        let (u_cand, mv_zero) = l2_normalize_vec(&mv)?;
        // Keep the previous unit `u` when `M v == 0` so zero-initialized layers retain a valid restart vector.
        u = mv_zero.where_cond(&u, &u_cand)?;
    }
    Ok((u, v))
}

/// Run `n_iters >= 1` deterministic power-iteration steps on `weight` (reshaped as `[rows, cols]`) starting
/// from left vector `u_init` (`[rows]`), returning `(sigma, u, v)` with `u` and `v` detached.
pub fn power_iteration(
    weight: &Tensor,
    u_init: &Tensor,
    n_iters: usize,
    eps: f32,
) -> Result<PowerIterationResult> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!("power_iteration: eps must be finite and > 0, got {eps}")));
    }
    let (u, v) = power_iteration_uv(weight, u_init, n_iters, None)?;
    let _g = no_grad();
    let (mat, rows, cols) = weight_as_matrix(&weight.detach(), "power_iteration")?;
    let mv = mat_vec_f32(&mat, &v, rows, cols)?;
    let sigma = u.mul(&mv)?.sum_all()?.item::<f32>()?;
    Ok(PowerIterationResult { sigma, u, v })
}

/// Estimate the top singular value `sigma_1(weight)` via `n_iters` deterministic power-iteration steps.
pub fn spectral_norm(weight: &Tensor, n_iters: usize, eps: f32, seed: u64) -> Result<f32> {
    let (_, rows, _) = weight_as_matrix(&weight.detach(), "spectral_norm")?;
    let u0 = init_unit_vec(rows, weight.device(), seed)?;
    Ok(power_iteration(weight, &u0, n_iters, eps)?.sigma)
}

/// Compute the spectrally normalized weight `W / (sigma + eps)` and scalar `sigma = u^T M v`,
/// differentiable with respect to `weight` in `F32` with `u` and `v` detached (Miyato et al. 2018).
///
/// Always keeps the Rayleigh quotient and division in `F32` even when GPU `autocast` is active.
pub fn spectral_normalize(weight: &Tensor, u: &Tensor, v: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!("spectral_normalize: eps must be finite and > 0, got {eps}")));
    }
    let (mat, rows, cols) = weight_as_matrix(weight, "spectral_normalize")?;
    if u.dims() != [rows] || u.dtype() != DType::F32 || u.device() != weight.device() {
        return Err(Error::Invalid(format!(
            "spectral_normalize: u ({:?} {} on {:?}) does not match rows [{rows}] f32 on {:?}",
            u.dims(),
            u.dtype(),
            u.device(),
            weight.device()
        )));
    }
    if v.dims() != [cols] || v.dtype() != DType::F32 || v.device() != weight.device() {
        return Err(Error::Invalid(format!(
            "spectral_normalize: v ({:?} {} on {:?}) does not match cols [{cols}] f32 on {:?}",
            v.dims(),
            v.dtype(),
            v.device(),
            weight.device()
        )));
    }
    let u_d = u.detach();
    let v_d = v.detach();
    let sigma = if weight.device().is_cpu() && folo_core::autocast_dtype().is_none() {
        u_d.reshape([1, rows])?.matmul(&mat)?.matmul(&v_d.reshape([cols, 1])?)?.reshape([] as [usize; 0])?
    } else {
        // Elementwise `mul` + `sum` stays in `F32` under GPU `autocast` (unlike `matmul`, which casts to `F16`).
        mat.mul(&u_d.reshape([rows, 1])?)?.sum(0, false)?.mul(&v_d)?.sum_all()?
    };
    let denom = sigma.add_scalar(eps)?;
    let w_sn = weight.div(&denom)?;
    Ok((w_sn, sigma))
}

/// Layer types that can be wrapped with [`SpectralNorm`].
pub trait SpectralNormTarget: Module {
    /// The primary weight tensor normalized by [`SpectralNorm`].
    fn weight(&self) -> &Tensor;
    /// Forward pass using an externally supplied normalized weight `w_sn` instead of `self.weight()`.
    fn forward_with_weight(&self, x: &Tensor, w_sn: &Tensor) -> Result<Tensor>;
}

impl SpectralNormTarget for Linear {
    fn weight(&self) -> &Tensor {
        &self.weight
    }

    fn forward_with_weight(&self, x: &Tensor, w_sn: &Tensor) -> Result<Tensor> {
        let (o, i) = w_sn.dims2()?;
        let last = *x.dims().last().ok_or_else(|| Error::Invalid("Linear: scalar input".into()))?;
        if last != i {
            return Err(Error::Invalid(format!(
                "Linear: input {:?} has {last} features, layer expects {i}",
                x.dims()
            )));
        }
        let rows = x.numel() / i.max(1);
        let mut y = x.reshape([rows, i])?.matmul(&w_sn.t()?)?;
        if let Some(b) = &self.bias {
            let b = if b.dtype() == DType::F32
                && y.dtype() == DType::F16
                && folo_core::autocast_dtype() == Some(DType::F16)
            {
                b.to_dtype(DType::F16)?
            } else {
                b.clone()
            };
            y = y.add(&b)?;
        }
        let mut out = x.dims().to_vec();
        *out.last_mut().expect("rank >= 1") = o;
        Ok(y.reshape(out)?)
    }
}

impl SpectralNormTarget for Conv2d {
    fn weight(&self) -> &Tensor {
        &self.weight
    }

    fn forward_with_weight(&self, x: &Tensor, w_sn: &Tensor) -> Result<Tensor> {
        let c = self.config();
        Ok(x.conv2d(w_sn, self.bias.as_ref(), c.stride, c.padding, c.dilation, c.groups)?)
    }
}

/// Spectral Normalization wrapper over any [`SpectralNormTarget`] ([`Linear`] or [`Conv2d`]).
///
/// Exposes the wrapped layer's parameters (`weight`, optional `bias`) alongside detached singular-vector
/// buffers (`u`, `v`), and honors [`Module::train`] / [`Module::eval`] (updating `u, v` via power iteration
/// during training and freezing them during evaluation).
pub struct SpectralNorm<M: SpectralNormTarget> {
    inner: M,
    /// Detached left singular vector buffer `[out]`.
    pub u: Tensor,
    /// Detached right singular vector buffer `[in_flat]`.
    pub v: Tensor,
    n_power_iterations: usize,
    eps: f32,
    seed: u64,
    mode: Mode,
}

impl<M: SpectralNormTarget> SpectralNorm<M> {
    /// Wrap `inner` with spectral normalization using `n_power_iterations >= 1` power-iteration steps per
    /// training forward pass, stability constant `eps > 0`, and deterministic `seed`.
    ///
    /// Runs `n_power_iterations` initial power-iteration steps so `u` and `v` are immediately aligned with
    /// `inner.weight()` before the first forward pass. If `inner.weight()` starts at zero, retains the seeded
    /// unit vector `u` so subsequent non-zero weight updates recover `sigma(W)` automatically.
    pub fn new(inner: M, n_power_iterations: usize, eps: f32, seed: u64) -> Result<Self> {
        if n_power_iterations == 0 {
            return Err(Error::Invalid("SpectralNorm: n_power_iterations must be >= 1".into()));
        }
        if !(eps.is_finite() && eps > 0.0) {
            return Err(Error::Invalid(format!("SpectralNorm: eps must be finite and > 0, got {eps}")));
        }
        let w = inner.weight();
        let (_, rows, _) = weight_as_matrix(&w.detach(), "SpectralNorm")?;
        let u0 = init_unit_vec(rows, w.device(), seed)?;
        let (u, v) = power_iteration_uv(w, &u0, n_power_iterations, Some(&u0))?;
        Ok(Self { inner, u, v, n_power_iterations, eps, seed, mode: Mode::new() })
    }

    /// Borrow the wrapped layer.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Power iterations executed per training forward pass.
    pub fn n_power_iterations(&self) -> usize {
        self.n_power_iterations
    }

    /// Numerical stability epsilon.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Current Rayleigh-quotient spectral norm estimate `u^T M v` (without advancing buffers).
    pub fn sigma(&self) -> Result<f32> {
        let _g = no_grad();
        let (_, sigma) = spectral_normalize(self.inner.weight(), &self.u, &self.v, self.eps)?;
        Ok(sigma.item::<f32>()?)
    }

    /// Compute the spectrally normalized weight `W / (sigma + eps)`.
    ///
    /// In training mode, runs power iteration into fresh `(u_fresh, v_fresh)` tensors, copies them into the
    /// persistent buffers `(self.u, self.v)` under `no_grad`, and feeds `(u_fresh, v_fresh)` into the
    /// autograd graph so consecutive training forwards can share one backward pass without in-place conflicts.
    /// In eval mode, freezes `(self.u, self.v)` and feeds cloned detached copies into the graph.
    pub fn normalized_weight(&self) -> Result<Tensor> {
        let w = self.inner.weight();
        if self.mode.is_training() {
            let rows = w.dims().first().copied().unwrap_or(0);
            let fallback_u = if rows > 0 { Some(init_unit_vec(rows, w.device(), self.seed)?) } else { None };
            let (u_fresh, v_fresh) =
                power_iteration_uv(w, &self.u, self.n_power_iterations, fallback_u.as_ref())?;
            {
                let _g = no_grad();
                self.u.copy_from(&u_fresh)?;
                self.v.copy_from(&v_fresh)?;
            }
            Ok(spectral_normalize(w, &u_fresh, &v_fresh, self.eps)?.0)
        } else {
            // Snapshot tiny 1-D buffers so an eval forward followed by a train forward in the same graph
            // never hits `ModifiedInPlace`.
            let (u_snap, v_snap) = {
                let _g = no_grad();
                (self.u.mul_scalar(1.0)?, self.v.mul_scalar(1.0)?)
            };
            Ok(spectral_normalize(w, &u_snap, &v_snap, self.eps)?.0)
        }
    }
}

impl<M: SpectralNormTarget> Module for SpectralNorm<M> {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        self.inner.visit_tensors(f);
        f("u", &self.u, TensorKind::Buffer);
        f("v", &self.v, TensorKind::Buffer);
    }

    fn visit_children(&self, f: &mut dyn FnMut(&str, &dyn Module)) {
        self.inner.visit_children(f);
    }

    fn mode(&self) -> Option<&Mode> {
        Some(&self.mode)
    }
}

impl<M: SpectralNormTarget> Forward for SpectralNorm<M> {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w_sn = self.normalized_weight()?;
        self.inner.forward_with_weight(x, &w_sn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conv2dConfig;
    use folo_core::testing::{GradCheck, grad_check_with};

    #[test]
    fn power_iteration_recovers_exact_top_singular_value_and_normalizes_to_one() {
        // 1. Diagonal matrix with known singular values {5.0, 2.5, 0.7}.
        let diag = Tensor::from_slice(&[2.5f32, 0.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.7], [3, 3]).unwrap();
        let s = spectral_norm(&diag, 30, SPECTRAL_NORM_EPS, 7).unwrap();
        assert!((s - 5.0).abs() < 1e-5, "diagonal sigma = {s}, want 5.0");

        // 2. Rotated rectangular matrix U * diag(7.25, 1.5) * V^T (exact sigma_1 = 7.25).
        let c1 = 0.6f32;
        let s1 = 0.8f32;
        let u_mat = Tensor::from_slice(&[c1, -s1, s1, c1], [2, 2]).unwrap();
        let diag_r = Tensor::from_slice(&[7.25f32, 0.0, 0.0, 0.0, 1.5, 0.0], [2, 3]).unwrap();
        let c2 = 0.8f32;
        let s2 = -0.6f32;
        let vt = Tensor::from_slice(&[c2, s2, 0.0, -s2, c2, 0.0, 0.0, 0.0, 1.0], [3, 3]).unwrap();
        let w = u_mat.matmul(&diag_r).unwrap().matmul(&vt).unwrap();
        let sigma = spectral_norm(&w, 40, SPECTRAL_NORM_EPS, 11).unwrap();
        assert!((sigma - 7.25).abs() < 1e-4, "rotated rect sigma = {sigma}, want 7.25");

        // 3. SpectralNorm<Linear> normalizes sigma_1(W_SN) == 1.0 and freezes u, v in eval mode.
        let lin = Linear::from_tensors(w.clone().requires_grad_(), None).unwrap();
        let sn = SpectralNorm::new(lin, 25, SPECTRAL_NORM_EPS, 13).unwrap();
        let w_sn = sn.normalized_weight().unwrap();
        let sn_after = spectral_norm(&w_sn, 30, SPECTRAL_NORM_EPS, 17).unwrap();
        assert!((sn_after - 1.0).abs() < 1e-4, "normalized sigma = {sn_after}, want 1.0");

        // Eval mode does not mutate u or v.
        sn.eval();
        let u_before = sn.u.to_vec::<f32>().unwrap();
        let v_before = sn.v.to_vec::<f32>().unwrap();
        let _ = sn.forward(&Tensor::ones([4, 3])).unwrap();
        assert_eq!(sn.u.to_vec::<f32>().unwrap(), u_before);
        assert_eq!(sn.v.to_vec::<f32>().unwrap(), v_before);

        // 4. 4-D Conv2d spectral norm.
        let mut rng = Rng::seed(21);
        let conv = Conv2d::new(Conv2dConfig::new(3, 5, 3).padding(1), &mut rng);
        let sn_conv = SpectralNorm::new(conv, 25, SPECTRAL_NORM_EPS, 23).unwrap();
        let w_conv_sn = sn_conv.normalized_weight().unwrap();
        let sigma_conv = spectral_norm(&w_conv_sn, 30, SPECTRAL_NORM_EPS, 29).unwrap();
        assert!((sigma_conv - 1.0).abs() < 1e-3, "Conv2d normalized sigma = {sigma_conv}, want 1.0");
    }

    #[test]
    fn spectral_norm_grad_check_linear_conv2d_and_envelope_theorem() {
        let mut rng = Rng::seed(101);
        let w = Tensor::randn([3, 4], 0.0, 1.0, &mut rng);
        let u0 = init_unit_vec(3, folo_core::Device::Cpu, 5).unwrap();
        let res = power_iteration(&w, &u0, 50, SPECTRAL_NORM_EPS).unwrap();
        let proj = Tensor::randn([3, 4], 0.0, 1.0, &mut rng);
        let gc = GradCheck { eps: 2e-3, atol: 3e-3, rtol: 2e-2, max_checks: 12, seed: 31 };

        // 1. Exact grad_check for spectral_normalize with fixed (u, v).
        grad_check_with(
            |inps| {
                spectral_normalize(&inps[0], &res.u, &res.v, SPECTRAL_NORM_EPS)
                    .unwrap()
                    .0
                    .mul(&proj)?
                    .sum_all()
            },
            std::slice::from_ref(&w),
            &gc,
        )
        .unwrap();

        // 2. Envelope-theorem verification: analytic gradient with detached converged (u*, v*) matches
        //    central finite differences where sigma(W + eps*E_ij) is RECOMPUTED via fresh power iteration.
        let w_leaf = w.detach().requires_grad_();
        let (w_sn, _) = spectral_normalize(&w_leaf, &res.u, &res.v, SPECTRAL_NORM_EPS).unwrap();
        let loss = w_sn.mul(&proj).unwrap().sum_all().unwrap();
        let g_ana = loss.backward().unwrap().get(&w_leaf).unwrap().to_vec::<f32>().unwrap();
        let wv = w.to_vec::<f32>().unwrap();
        let pv = proj.to_vec::<f32>().unwrap();
        let eps = 2e-3f32;
        for idx in 0..wv.len() {
            let eval_recomputed = |delta: f32| -> f32 {
                let mut tmp = wv.clone();
                tmp[idx] += delta;
                let wt = Tensor::from_vec(tmp, [3, 4]).unwrap();
                let sig = spectral_norm(&wt, 60, SPECTRAL_NORM_EPS, 5).unwrap();
                wt.to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .zip(&pv)
                    .map(|(&a, &b)| (a / (sig + SPECTRAL_NORM_EPS)) * b)
                    .sum()
            };
            let fd = (eval_recomputed(eps) - eval_recomputed(-eps)) / (2.0 * eps);
            let tol = 3e-3 + 2e-2 * fd.abs().max(g_ana[idx].abs());
            assert!((g_ana[idx] - fd).abs() <= tol, "envelope grad[{idx}]: ana={} fd={fd}", g_ana[idx]);
        }

        // 3. SpectralNorm<Linear> and SpectralNorm<Conv2d> forward/backward grad_check wrt input x.
        let lin = Linear::new(4, 3, true, &mut rng);
        let sn_lin = SpectralNorm::new(lin, 15, SPECTRAL_NORM_EPS, 37).unwrap();
        sn_lin.eval();
        let x_lin = Tensor::randn([2, 4], 0.0, 1.0, &mut rng);
        grad_check_with(
            |inps| sn_lin.forward(&inps[0]).unwrap().square()?.sum_all(),
            std::slice::from_ref(&x_lin),
            &gc,
        )
        .unwrap();

        let conv = Conv2d::new(Conv2dConfig::new(2, 3, 3).padding(1), &mut rng);
        let sn_conv = SpectralNorm::new(conv, 15, SPECTRAL_NORM_EPS, 41).unwrap();
        sn_conv.eval();
        let x_conv = Tensor::randn([1, 2, 4, 4], 0.0, 1.0, &mut rng);
        grad_check_with(
            |inps| sn_conv.forward(&inps[0]).unwrap().square()?.sum_all(),
            std::slice::from_ref(&x_conv),
            &gc,
        )
        .unwrap();

        // 4. Regression: summing two training forwards before a single backward must NOT fail with ModifiedInPlace.
        sn_lin.train();
        let x1 = Tensor::randn([2, 4], 0.0, 1.0, &mut rng).requires_grad_();
        let x2 = Tensor::randn([2, 4], 0.0, 1.0, &mut rng).requires_grad_();
        let y1 = sn_lin.forward(&x1).unwrap();
        let y2 = sn_lin.forward(&x2).unwrap();
        let total =
            y1.square().unwrap().sum_all().unwrap().add(&y2.square().unwrap().sum_all().unwrap()).unwrap();
        let grads = total.backward().unwrap();
        let gw = grads.get(&sn_lin.inner().weight).unwrap().to_vec::<f32>().unwrap();
        assert!(gw.iter().all(|v| v.is_finite()) && gw.iter().any(|v| v.abs() > 1e-6));
    }

    #[test]
    fn spectral_norm_edge_cases_zero_recovery_and_validation() {
        // 1. Exact zero matrix stays finite (sigma == 0, normalized weight == 0, finite grad) AND retains a
        //    valid unit `u` so transitioning zero -> non-zero immediately recovers sigma_1(W_SN) == 1.0.
        let zero_lin = Linear::from_tensors(Tensor::zeros([3, 4]).requires_grad_(), None).unwrap();
        let sn_zero = SpectralNorm::new(zero_lin, 15, SPECTRAL_NORM_EPS, 1).unwrap();
        let w0_sn = sn_zero.normalized_weight().unwrap();
        assert!(w0_sn.to_vec::<f32>().unwrap().iter().all(|&v| v == 0.0));
        assert_eq!(sn_zero.sigma().unwrap(), 0.0);
        let gz = w0_sn
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .get(&sn_zero.inner().weight)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert!(gz.iter().all(|v| v.is_finite()));

        // Transition weight from zero -> non-zero: must recover sigma_1 == 4.0 and normalized sigma == 1.0.
        {
            let _g = no_grad();
            let non_zero =
                Tensor::from_slice(&[4.0f32, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [3, 4])
                    .unwrap();
            sn_zero.inner().weight.copy_from(&non_zero).unwrap();
        }
        let w1_sn = sn_zero.normalized_weight().unwrap();
        assert!(
            (sn_zero.sigma().unwrap() - 4.0).abs() < 1e-4,
            "recovered sigma={}",
            sn_zero.sigma().unwrap()
        );
        let s1_after = spectral_norm(&w1_sn, 30, SPECTRAL_NORM_EPS, 3).unwrap();
        assert!((s1_after - 1.0).abs() < 1e-4, "recovered W_SN sigma={s1_after}");

        // Also recovers if `sn.u` was externally zeroed (e.g. from zero-initialized checkpoint buffer).
        {
            let _g = no_grad();
            sn_zero.u.copy_from(&Tensor::zeros([3])).unwrap();
        }
        let w2_sn = sn_zero.normalized_weight().unwrap();
        let s2_after = spectral_norm(&w2_sn, 30, SPECTRAL_NORM_EPS, 5).unwrap();
        assert!((s2_after - 1.0).abs() < 1e-4, "zeroed-u recovery sigma={s2_after}");

        // 2. Extreme scales (1e-18 and 1e18) avoid f32 squaring underflow/overflow.
        for scale in [1e-18f32, 1e18f32] {
            let big = Tensor::from_slice(&[scale, 0.0, 0.0, 0.5 * scale], [2, 2]).unwrap();
            let s = spectral_norm(&big, 20, SPECTRAL_NORM_EPS, 3).unwrap();
            assert!((s - scale).abs() <= 1e-4 * scale, "scale={scale:e}: got {s:e}");
        }

        // 3. Non-finite weight propagates NaN cleanly without panicking.
        let bad_w = Tensor::from_slice(&[f32::NAN, 1.0, 0.0, 2.0], [2, 2]).unwrap().requires_grad_();
        let res_bad = power_iteration(&bad_w, &Tensor::ones([2]), 3, SPECTRAL_NORM_EPS).unwrap();
        assert!(res_bad.sigma.is_nan());
        let (w_bad_sn, _) = spectral_normalize(&bad_w, &res_bad.u, &res_bad.v, SPECTRAL_NORM_EPS).unwrap();
        assert!(w_bad_sn.to_vec::<f32>().unwrap().iter().all(|v| v.is_nan()));

        // 4. Validation rejections.
        let ok = Tensor::ones([2, 3]);
        assert!(power_iteration(&ok, &Tensor::ones([2]), 0, SPECTRAL_NORM_EPS).is_err());
        assert!(power_iteration(&ok, &Tensor::ones([2]), 1, 0.0).is_err());
        assert!(power_iteration(&ok, &Tensor::ones([2]), 1, f32::NAN).is_err());
        assert!(power_iteration(&ok, &Tensor::ones([3]), 1, SPECTRAL_NORM_EPS).is_err());
        assert!(power_iteration(&Tensor::ones([3]), &Tensor::ones([3]), 1, SPECTRAL_NORM_EPS).is_err());
        let mut rng = Rng::seed(9);
        assert!(SpectralNorm::new(Linear::new(3, 2, false, &mut rng), 0, SPECTRAL_NORM_EPS, 1).is_err());
        assert!(SpectralNorm::new(Linear::new(3, 2, false, &mut rng), 1, -1e-6, 1).is_err());
    }

    #[test]
    fn spectral_norm_gpu_autocast_extreme_scale_and_nan_parity() {
        let Ok(ctx) = folo_gpu::GpuContext::shared() else {
            return;
        };
        if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
            return;
        }
        let Ok(dev) = folo_gpu::init() else {
            return;
        };
        // 1. Large scale diag(1e20, 5e19) on GPU must not overflow squaring to 0.
        let big = Tensor::from_slice(&[1e20f32, 0.0, 0.0, 5e19], [2, 2]).unwrap().to_device(dev).unwrap();
        let s_big = spectral_norm(&big, 25, SPECTRAL_NORM_EPS, 7).unwrap();
        assert!((s_big - 1e20).abs() <= 1e-3 * 1e20, "GPU big scale sigma={s_big:e}");

        // 2. Tiny scale diag(1e-20, 5e-21) on GPU must not underflow squaring.
        let tiny = Tensor::from_slice(&[1e-20f32, 0.0, 0.0, 5e-21], [2, 2]).unwrap().to_device(dev).unwrap();
        let s_tiny = spectral_norm(&tiny, 25, SPECTRAL_NORM_EPS, 7).unwrap();
        assert!((s_tiny - 1e-20).abs() <= 1e-3 * 1e-20, "GPU tiny scale sigma={s_tiny:e}");

        // 3. NaN weight on GPU must propagate NaN (never suppressed to finite/Inf).
        let nan_w = Tensor::from_slice(&[f32::NAN, 0.0, 0.0, 0.0], [2, 2]).unwrap().to_device(dev).unwrap();
        let u_gpu = Tensor::ones([2]).to_device(dev).unwrap();
        let res_nan = power_iteration(&nan_w, &u_gpu, 3, SPECTRAL_NORM_EPS).unwrap();
        assert!(res_nan.sigma.is_nan(), "GPU NaN sigma={}", res_nan.sigma);

        // 4. SpectralNorm<Linear> training forward + backward under GPU autocast (F16) succeeds cleanly.
        let mut rng = Rng::seed(55);
        let mut sn = SpectralNorm::new(Linear::new(4, 3, true, &mut rng), 5, SPECTRAL_NORM_EPS, 11).unwrap();
        sn.to_device(dev).unwrap();
        let x1 = Tensor::randn([2, 4], 0.0, 1.0, &mut rng).to_device(dev).unwrap().requires_grad_();
        let x2 = Tensor::randn([2, 4], 0.0, 1.0, &mut rng).to_device(dev).unwrap().requires_grad_();
        let loss = {
            let _ac = folo_core::autocast();
            let y1 = sn.forward(&x1).unwrap();
            let y2 = sn.forward(&x2).unwrap();
            y1.to_dtype(DType::F32)
                .unwrap()
                .square()
                .unwrap()
                .sum_all()
                .unwrap()
                .add(&y2.to_dtype(DType::F32).unwrap().square().unwrap().sum_all().unwrap())
                .unwrap()
        };
        let grads = loss.backward().unwrap();
        let gw = grads.get(&sn.inner().weight).unwrap().to_vec::<f32>().unwrap();
        assert!(gw.iter().all(|v| v.is_finite()), "GPU autocast SpectralNorm grad non-finite: {gw:?}");
    }
}
