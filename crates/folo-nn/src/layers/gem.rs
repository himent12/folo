//! Generalized Mean (GeM) Pooling ([`GeM2d`], [`gem_pool2d`]; Radenović, Tolias & Chum, TPAMI 2019 /
//! arXiv:1711.02512).
//!
//! For an `NCHW` feature map `x: [N, C, H, W]` with exponent `p` clamped to `[p_min, p_max]` (`p_min > 0`)
//! and positive floor `eps > 0`:
//! ```text
//! GeM(x)_{n,c,0,0} = ( (1 / (H * W)) * sum_{h,w} max(x_{n,c,h,w}, eps)^p )^(1 / p)
//! ```
//!
//! Evaluates via a shift-invariant `f64` log-sum-exp identity (`M = max(ln(max(x, eps)))`) so large
//! exponents (`p = 50`) and tiny/large activations never overflow or underflow, with exact analytical
//! gradients flowing into both `x` and the learnable exponent parameter `p`.

use folo_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::module::{Forward, Module, TensorKind};
use crate::{Error, Result};

/// Default GeM initial pooling exponent `p` (Radenović et al. 2019).
pub const GEM_DEFAULT_P: f32 = 3.0;
/// Default minimum clamp on the pooling exponent `p_min`.
pub const GEM_DEFAULT_P_MIN: f32 = 1.0;
/// Default maximum clamp on the pooling exponent `p_max`.
pub const GEM_DEFAULT_P_MAX: f32 = 10.0;
/// Default positive activation floor `eps` (`1e-6`).
pub const GEM_DEFAULT_EPS: f32 = 1e-6;

/// Configuration for [`GeM2d`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeM2dConfig {
    /// Initial pooling exponent `p` (default `3.0`).
    pub p_init: f32,
    /// Lower clamp `p_min > 0` for `p` during forward (default `1.0` = average pooling).
    pub p_min: f32,
    /// Upper clamp `p_max >= p_min` for `p` during forward (default `10.0`).
    pub p_max: f32,
    /// Positive floor `eps > 0` applied via `max(x, eps)` before exponentiation (default `1e-6`).
    pub eps: f32,
}

impl Default for GeM2dConfig {
    fn default() -> Self {
        Self {
            p_init: GEM_DEFAULT_P,
            p_min: GEM_DEFAULT_P_MIN,
            p_max: GEM_DEFAULT_P_MAX,
            eps: GEM_DEFAULT_EPS,
        }
    }
}

impl GeM2dConfig {
    /// Validate `p_min > 0`, `p_max >= p_min`, `p_init` finite, and `eps > 0`.
    pub fn validate(&self) -> Result<()> {
        validate_gem_params(self.p_min, self.p_max, self.eps)?;
        if !self.p_init.is_finite() {
            return Err(Error::Invalid(format!("GeM2d: p_init must be finite, got {}", self.p_init)));
        }
        Ok(())
    }
}

fn validate_gem_params(p_min: f32, p_max: f32, eps: f32) -> Result<()> {
    if !(p_min.is_finite() && p_min > 0.0) {
        return Err(Error::Invalid(format!("gem_pool2d: p_min must be finite and > 0, got {p_min}")));
    }
    if !(p_max.is_finite() && p_max >= p_min) {
        return Err(Error::Invalid(format!(
            "gem_pool2d: p_max must be finite and >= p_min ({p_min}), got {p_max}"
        )));
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::Invalid(format!("gem_pool2d: eps must be finite and > 0, got {eps}")));
    }
    Ok(())
}

/// Functional differentiable 2-D Generalized Mean pooling over an `NCHW` `F32` tensor `x: [N, C, H, W]`
/// with learnable exponent `p` of shape `[1]` (shared across channels) or `[C]` (per-channel),
/// returning `[N, C, 1, 1]`.
pub fn gem_pool2d(x: &Tensor, p: &Tensor, p_min: f32, p_max: f32, eps: f32) -> Result<Tensor> {
    validate_gem_params(p_min, p_max, eps)?;
    if x.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("gem_pool2d: x must be f32, got {}", x.dtype())));
    }
    if p.dtype() != DType::F32 {
        return Err(Error::Invalid(format!("gem_pool2d: p must be f32, got {}", p.dtype())));
    }
    if x.device() != p.device() {
        return Err(Error::Invalid(format!(
            "gem_pool2d: x ({:?}) and p ({:?}) must be on the same device",
            x.device(),
            p.device()
        )));
    }
    let (n, c, h, w) = x
        .dims4()
        .map_err(|_| Error::Invalid(format!("gem_pool2d: expected NCHW [N, C, H, W], got {:?}", x.dims())))?;
    if c == 0 || h == 0 || w == 0 {
        return Err(Error::Invalid(format!(
            "gem_pool2d: C, H, W must all be >= 1, got [{n}, {c}, {h}, {w}]"
        )));
    }
    let p_dims = p.dims();
    let per_channel = match *p_dims {
        [1] => false,
        [pc] if pc == c => true,
        _ => {
            return Err(Error::Invalid(format!(
                "gem_pool2d: p must have shape [1] or [{c}], got {p_dims:?}"
            )));
        }
    };

    let dev = x.device();
    let need_x = x.requires_grad();
    let need_p = p.requires_grad();
    let p_shape = p.shape().clone();

    if n == 0 {
        let out_t = Tensor::zeros_with([0, c, 1, 1], DType::F32, dev)?;
        return Ok(Tensor::custom_op("gem_pool2d", &[x, p], out_t, move |_g| {
            let gx = if need_x { Some(Tensor::zeros_with([0, c, h, w], DType::F32, dev)?) } else { None };
            let gp = if need_p { Some(Tensor::zeros_with(&p_shape, DType::F32, dev)?) } else { None };
            Ok(vec![gx, gp])
        })?);
    }

    let hw = h
        .checked_mul(w)
        .ok_or_else(|| Error::Invalid(format!("gem_pool2d: spatial size {h}*{w} overflows usize")))?;
    let nc = n
        .checked_mul(c)
        .ok_or_else(|| Error::Invalid(format!("gem_pool2d: batch*channels {n}*{c} overflows usize")))?;
    let total = nc
        .checked_mul(hw)
        .ok_or_else(|| Error::Invalid(format!("gem_pool2d: tensor size {nc}*{hw} overflows usize")))?;
    let xv = x.to_vec::<f32>()?;
    let pv = p.to_vec::<f32>()?;
    let ln_hw = (hw as f64).ln();
    let eps64 = eps as f64;
    let p_min64 = p_min as f64;
    let p_max64 = p_max as f64;

    let mut out = vec![0.0f32; nc];
    let mut saved_dx = if need_x { vec![0.0f64; total] } else { Vec::new() };
    let mut saved_dp = if need_p { vec![0.0f64; nc] } else { Vec::new() };

    let mut u_buf = vec![0.0f64; hw];
    let mut w_buf = vec![0.0f64; hw];

    for ni in 0..n {
        for ci in 0..c {
            let p_raw = pv[if per_channel { ci } else { 0 }] as f64;
            let nc_idx = ni * c + ci;
            let base = nc_idx * hw;
            if !p_raw.is_finite() {
                out[nc_idx] = f32::NAN;
                if need_x {
                    saved_dx[base..base + hw].fill(f64::NAN);
                }
                if need_p {
                    saved_dp[nc_idx] = f64::NAN;
                }
                continue;
            }
            let p_clamped = p_raw.clamp(p_min64, p_max64);
            let p_active = p_raw > p_min64 && p_raw < p_max64;

            let mut max_u = f64::NEG_INFINITY;
            let mut has_nan = false;
            for s in 0..hw {
                let raw = xv[base + s] as f64;
                if raw.is_nan() || raw == f64::INFINITY {
                    has_nan = true;
                    break;
                }
                let clamped = raw.max(eps64);
                let u = clamped.ln();
                u_buf[s] = u;
                if u > max_u {
                    max_u = u;
                }
            }
            if has_nan {
                out[nc_idx] = f32::NAN;
                if need_x {
                    saved_dx[base..base + hw].fill(f64::NAN);
                }
                if need_p {
                    saved_dp[nc_idx] = f64::NAN;
                }
                continue;
            }

            let mut sum_exp = 0.0f64;
            for s in 0..hw {
                let ew = (p_clamped * (u_buf[s] - max_u)).exp();
                w_buf[s] = ew;
                sum_exp += ew;
            }
            let lse_shifted = sum_exp.ln() - ln_hw;
            let log_y = max_u + lse_shifted / p_clamped;
            let y = log_y.exp();
            out[nc_idx] = y as f32;

            let inv_sum = 1.0 / sum_exp;
            if need_x {
                for s in 0..hw {
                    let raw = xv[base + s] as f64;
                    if raw > eps64 {
                        let weight = w_buf[s] * inv_sum;
                        saved_dx[base + s] = y * weight / raw;
                    }
                }
            }
            if need_p && p_active {
                let mut weighted_du = 0.0f64;
                for s in 0..hw {
                    weighted_du += (w_buf[s] * inv_sum) * (u_buf[s] - max_u);
                }
                saved_dp[nc_idx] = y * (weighted_du - lse_shifted / p_clamped) / p_clamped;
            }
        }
    }

    let out_t = Tensor::from_vec(out, [n, c, 1, 1])?.to_device(dev)?;
    Ok(Tensor::custom_op("gem_pool2d", &[x, p], out_t, move |g| {
        let gv = g.to_vec::<f32>()?;
        let gx = if need_x {
            let mut grad_x = vec![0.0f32; n * c * hw];
            for (nc_idx, &g_val) in gv.iter().enumerate() {
                let gi = g_val as f64;
                let base = nc_idx * hw;
                for s in 0..hw {
                    let d = saved_dx[base + s];
                    if d == 0.0 || (gi == 0.0 && d.is_finite()) {
                        continue;
                    }
                    grad_x[base + s] = (gi * d) as f32;
                }
            }
            Some(Tensor::from_vec(grad_x, [n, c, h, w])?.to_device(dev)?)
        } else {
            None
        };
        let gp = if need_p {
            let p_len = if per_channel { c } else { 1 };
            let mut acc = vec![0.0f64; p_len];
            for ni in 0..n {
                for ci in 0..c {
                    let nc_idx = ni * c + ci;
                    let gi = gv[nc_idx] as f64;
                    let d = saved_dp[nc_idx];
                    if d == 0.0 || (gi == 0.0 && d.is_finite()) {
                        continue;
                    }
                    let slot = if per_channel { ci } else { 0 };
                    acc[slot] += gi * d;
                }
            }
            let grad_p: Vec<f32> = acc.into_iter().map(|v| v as f32).collect();
            Some(Tensor::from_vec(grad_p, &p_shape)?.to_device(dev)?)
        } else {
            None
        };
        Ok(vec![gx, gp])
    })?)
}

/// Learnable 2-D Generalized Mean (GeM) Pooling module (Radenović et al., TPAMI 2019).
///
/// Parameter name: `p` (`[1]` shared across channels or `[C]` per-channel).
pub struct GeM2d {
    /// Learnable pooling exponent (`[1]` or `[C]`), clamped to `[p_min, p_max]` on forward.
    pub p: Tensor,
    p_min: f32,
    p_max: f32,
    eps: f32,
}

impl GeM2d {
    /// Construct a shared-exponent [`GeM2d`] (`p: [1]`) from [`GeM2dConfig`].
    pub fn new(cfg: GeM2dConfig) -> Result<Self> {
        cfg.validate()?;
        let p = Tensor::from_slice(&[cfg.p_init], [1])?.requires_grad_();
        Ok(Self { p, p_min: cfg.p_min, p_max: cfg.p_max, eps: cfg.eps })
    }

    /// Construct a per-channel [`GeM2d`] (`p: [channels]`) from [`GeM2dConfig`].
    pub fn per_channel(channels: usize, cfg: GeM2dConfig) -> Result<Self> {
        cfg.validate()?;
        if channels == 0 {
            return Err(Error::Invalid("GeM2d: channels must be >= 1, got 0".into()));
        }
        let p = Tensor::from_vec(vec![cfg.p_init; channels], [channels])?.requires_grad_();
        Ok(Self { p, p_min: cfg.p_min, p_max: cfg.p_max, eps: cfg.eps })
    }

    /// Minimum exponent clamp `p_min`.
    pub fn p_min(&self) -> f32 {
        self.p_min
    }

    /// Maximum exponent clamp `p_max`.
    pub fn p_max(&self) -> f32 {
        self.p_max
    }

    /// Activation floor `eps`.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for GeM2d {
    fn visit_tensors(&self, f: &mut dyn FnMut(&str, &Tensor, TensorKind)) {
        f("p", &self.p, TensorKind::Parameter);
    }
}

impl Forward for GeM2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        gem_pool2d(x, &self.p, self.p_min, self.p_max, self.eps)
    }
}
