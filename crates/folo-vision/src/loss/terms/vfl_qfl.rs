//! Standalone Varifocal Loss (VFL, Zhang et al. CVPR 2021) and Quality Focal Loss (QFL, Li et al.
//! NeurIPS 2020) for IoU-aware dense object detection classification with continuous quality targets
//! `q in [0, 1]`.
//!
//! Both kernels evaluate every term and its closed-form analytical derivative in `f64` using
//! cancellation-free log-sigmoid (`softplus(±x)`) and sigmoid (`p = sigmoid(x)`, `r = sigmoid(-x)`)
//! formulations so extreme logits (`±50`, `±500`, up to `±f32::MAX`) and boundary targets (`q = 0`,
//! `q = 1`) never suffer overflow, underflow-induced `0^(gamma - 1) = inf`, or catastrophic
//! cancellation:
//!
//! - **Varifocal Loss (VFL)** (`alpha > 0`, `gamma >= 0`):
//!   ```text
//!   L_VFL(x, q) = q * (q * softplus(-x) + (1 - q) * softplus(x))   if q > 0
//!               = alpha * p^gamma * softplus(x)                    if q == 0
//!   ```
//! - **Quality Focal Loss (QFL)** (`beta == 0` or `beta >= 1`):
//!   ```text
//!   L_QFL(x, q) = |q - p|^beta * (q * softplus(-x) + (1 - q) * softplus(x))
//!   ```
//!
//! Finite targets outside `[0, 1]` are rejected with [`Error::InvalidArgument`]; non-finite logits or
//! targets (`NaN`, `±Inf`) propagate `NaN` through both forward and backward.

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::loss::invalid;

/// Reduction mode for standalone classification losses ([`varifocal_loss`], [`quality_focal_loss`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossReduction {
    /// Return per-element losses with the same shape as `logits`.
    None,
    /// Return the scalar sum of all element losses (`0.0` for empty tensors).
    Sum,
    /// Return the scalar arithmetic mean of all element losses (`0.0` for empty tensors).
    #[default]
    Mean,
}

/// Hyper-parameters for standalone [`varifocal_loss`] (Zhang et al. CVPR 2021).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VflConfig {
    /// Negative-branch weighting factor `alpha > 0` (paper default `0.75`).
    pub alpha: f32,
    /// Negative-branch focusing exponent `gamma >= 0` (paper default `2.0`).
    pub gamma: f32,
    /// Output reduction mode (default [`LossReduction::Mean`]).
    pub reduction: LossReduction,
}

impl Default for VflConfig {
    fn default() -> Self {
        Self { alpha: 0.75, gamma: 2.0, reduction: LossReduction::Mean }
    }
}

impl VflConfig {
    /// Validate `alpha` (finite and `> 0`) and `gamma` (finite and `>= 0`).
    pub fn validate(&self) -> Result<()> {
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return Err(invalid(format!("varifocal_loss: alpha must be finite and > 0, got {}", self.alpha)));
        }
        if !(self.gamma.is_finite() && self.gamma >= 0.0) {
            return Err(invalid(format!(
                "varifocal_loss: gamma must be finite and >= 0, got {}",
                self.gamma
            )));
        }
        Ok(())
    }
}

/// Hyper-parameters for standalone [`quality_focal_loss`] (Li et al. NeurIPS 2020).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QflConfig {
    /// Focusing exponent `beta` (`0.0` for plain BCE or `>= 1.0`; paper default `2.0`).
    ///
    /// Values in `(0, 1)` are rejected because `|p - q|^(beta - 1)` diverges at `p = q` when `0 < q < 1`.
    pub beta: f32,
    /// Output reduction mode (default [`LossReduction::Mean`]).
    pub reduction: LossReduction,
}

impl Default for QflConfig {
    fn default() -> Self {
        Self { beta: 2.0, reduction: LossReduction::Mean }
    }
}

impl QflConfig {
    /// Validate `beta` (finite and either `0.0` or `>= 1.0`).
    pub fn validate(&self) -> Result<()> {
        if !(self.beta.is_finite() && (self.beta == 0.0 || self.beta >= 1.0)) {
            return Err(invalid(format!(
                "quality_focal_loss: beta must be finite and 0 or >= 1, got {}",
                self.beta
            )));
        }
        Ok(())
    }
}

/// Numerically stable `sigmoid(x)` in `f64`.
#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable `softplus(x) = ln(1 + exp(x))` in `f64`.
#[inline]
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Cancellation-free `BCE(x, q) = q * softplus(-x) + (1 - q) * softplus(x)` in `f64` for finite `x` and `q in [0, 1]`.
#[inline]
fn bce_f64(x: f64, q: f64) -> f64 {
    let pos = if q == 0.0 { 0.0 } else { q * softplus_f64(-x) };
    let neg = if q == 1.0 { 0.0 } else { (1.0 - q) * softplus_f64(x) };
    pos + neg
}

/// Cancellation-free `p - q` in `f64`:
/// - for `|x| <= 1`, evaluates `((1 - 2q) - q * expm1(-x)) / (1 + exp(-x))` so tiny perturbations around
///   `p == q` (such as `x = ±1e-20` at `q = 0.5`) retain full precision without rounding `p` to `0.5`;
/// - for `x > 1`, evaluates `(1 - q) - sigmoid(-x)` so `q = 1` at large positive `x` preserves `-sigmoid(-x)`;
/// - for `x < -1`, evaluates `sigmoid(x) - q` so `q = 0` at large negative `x` preserves `sigmoid(x)`.
#[inline]
fn prob_minus_target(x: f64, p: f64, r: f64, q: f64) -> f64 {
    if x.abs() <= 1.0 {
        ((1.0 - 2.0 * q) - q * (-x).exp_m1()) / (1.0 + (-x).exp())
    } else if x > 0.0 {
        (1.0 - q) - r
    } else {
        p - q
    }
}

/// Exact single-element Varifocal Loss `(loss, dL/dx)` in `f64` for logit `x`, quality target `q in [0, 1]`,
/// negative weight `alpha > 0`, and negative focusing exponent `gamma >= 0`.
///
/// Evaluates `p^gamma = exp(-gamma * softplus(-x))` in log space so large exponents (e.g. `gamma = 1e22` at
/// `x = 50`) stay accurate even when `sigmoid(x)` rounds to `1.0` in `f64`. Non-finite `x` or `q` return
/// `(NaN, NaN)`.
#[inline]
pub(crate) fn vfl_elem_f64(x: f64, q: f64, alpha: f64, gamma: f64) -> (f64, f64) {
    if !x.is_finite() || !q.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let p = sigmoid_f64(x);
    let r = sigmoid_f64(-x);
    if q == 0.0 {
        let sp = softplus_f64(x);
        if gamma == 0.0 {
            return (alpha * sp, alpha * p);
        }
        let pg = (-gamma * softplus_f64(-x)).exp();
        if pg == 0.0 {
            return (0.0, 0.0);
        }
        let gr_sp = if r == 0.0 { 0.0 } else { (gamma * r) * sp };
        return (alpha * pg * sp, alpha * pg * (gr_sp + p));
    }
    let bce = bce_f64(x, q);
    let diff = prob_minus_target(x, p, r, q);
    (q * bce, q * diff)
}

/// Exact single-element Quality Focal Loss `(loss, dL/dx)` in `f64` for logit `x`, quality target `q in [0, 1]`,
/// and focusing exponent `beta` (`0.0` or `>= 1.0`).
///
/// Special-cases `q = 0` and `q = 1` in log-space factored form (`exp(-beta * softplus(∓x))`) to prevent
/// `0^(beta - 1)` singularities, extreme-exponent rounding, and cancellation at extreme logits, and returns
/// `(0.0, 0.0)` at the exact `p == q` point (including the symmetric subgradient at the `beta == 1` cusp).
/// Non-finite `x` or `q` return `(NaN, NaN)`.
#[inline]
pub(crate) fn qfl_elem_f64(x: f64, q: f64, beta: f64) -> (f64, f64) {
    if !x.is_finite() || !q.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let p = sigmoid_f64(x);
    let r = sigmoid_f64(-x);
    if beta == 0.0 {
        return (bce_f64(x, q), prob_minus_target(x, p, r, q));
    }
    if q == 0.0 {
        let pb = (-beta * softplus_f64(-x)).exp();
        if pb == 0.0 {
            return (0.0, 0.0);
        }
        let sp = softplus_f64(x);
        let br_sp = if r == 0.0 { 0.0 } else { (beta * r) * sp };
        return (pb * sp, pb * (br_sp + p));
    }
    if q == 1.0 {
        let rb = (-beta * softplus_f64(x)).exp();
        if rb == 0.0 {
            return (0.0, 0.0);
        }
        let sp_neg = softplus_f64(-x);
        let bp_sp = if p == 0.0 { 0.0 } else { (beta * p) * sp_neg };
        return (rb * sp_neg, -rb * (bp_sp + r));
    }
    let diff = prob_minus_target(x, p, r, q);
    let d = diff.abs();
    if d == 0.0 {
        return (0.0, 0.0);
    }
    let bce = bce_f64(x, q);
    let pow_b = if beta == 2.0 {
        d * d
    } else if beta == 1.0 {
        d
    } else {
        d.powf(beta)
    };
    let d_pow = if beta == 2.0 {
        2.0 * diff
    } else if beta == 1.0 {
        diff.signum()
    } else {
        beta * d.powf(beta - 1.0) * diff.signum()
    };
    let pr_bce = if p == 0.0 || r == 0.0 { 0.0 } else { (p * r) * bce };
    (pow_b * bce, pow_b * diff + d_pow * pr_bce)
}

/// Validate `F32` dtype, matching device, and matching shape for `(logits, targets)`.
fn check_pair(logits: &Tensor, targets: &Tensor, op: &'static str) -> Result<()> {
    for t in [logits, targets] {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op, expected: DType::F32, got: t.dtype() });
        }
    }
    if logits.device() != targets.device() {
        return Err(Error::DeviceMismatch { op, lhs: logits.device(), rhs: targets.device() });
    }
    if logits.shape() != targets.shape() {
        return Err(Error::InvalidArgument {
            op,
            msg: format!("logits shape {} != targets shape {}", logits.shape(), targets.shape()),
        });
    }
    Ok(())
}

/// Host-evaluated custom op with exact `f64` forward, analytical backward w.r.t. `logits`, finite target
/// bounds validation (`q in [0, 1]`), and [`LossReduction`] support.
fn quality_cls_op<F>(
    logits: &Tensor,
    targets: &Tensor,
    reduction: LossReduction,
    op: &'static str,
    eval: F,
) -> Result<Tensor>
where
    F: Fn(f64, f64) -> (f64, f64) + Send + Sync + 'static,
{
    check_pair(logits, targets, op)?;
    let dev = logits.device();
    let shape = logits.shape().clone();
    let n = shape.numel();
    if n == 0 {
        let out_t = match reduction {
            LossReduction::None => Tensor::zeros_with(&shape, DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        return Tensor::custom_op(op, &[logits], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with(&shape, DType::F32, dev)?)])
        });
    }

    let xv = logits.to_vec::<f32>()?;
    let tv = targets.to_vec::<f32>()?;
    for (i, &q) in tv.iter().enumerate() {
        if q.is_finite() && !(0.0..=1.0).contains(&q) {
            return Err(Error::InvalidArgument {
                op,
                msg: format!("target[{i}] must be in [0, 1], got {q}"),
            });
        }
    }

    let mut dldx = vec![0.0f64; n];
    match reduction {
        LossReduction::None => {
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let (lv, gv) = eval(xv[i] as f64, tv[i] as f64);
                out[i] = lv as f32;
                dldx[i] = gv;
            }
            let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
            Tensor::custom_op(op, &[logits], out_t, move |g| {
                let gv = g.to_vec::<f32>()?;
                let mut grad = vec![0.0f32; n];
                for i in 0..n {
                    let gi = gv[i] as f64;
                    if gi == 0.0 && dldx[i].is_finite() {
                        continue;
                    }
                    grad[i] = (gi * dldx[i]) as f32;
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })
        }
        LossReduction::Sum | LossReduction::Mean => {
            let mut sum = 0.0f64;
            for i in 0..n {
                let (lv, gv) = eval(xv[i] as f64, tv[i] as f64);
                sum += lv;
                dldx[i] = gv;
            }
            let scale = if reduction == LossReduction::Mean { 1.0 / (n as f64) } else { 1.0 };
            let out_t = Tensor::scalar((sum * scale) as f32).to_device(dev)?;
            Tensor::custom_op(op, &[logits], out_t, move |g| {
                let g_scalar = g.item::<f32>()? as f64;
                let gs = g_scalar * scale;
                let mut grad = vec![0.0f32; n];
                for i in 0..n {
                    if gs == 0.0 && dldx[i].is_finite() {
                        continue;
                    }
                    grad[i] = (gs * dldx[i]) as f32;
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })
        }
    }
}

/// Standalone Varifocal Loss (VFL, Zhang et al. CVPR 2021) for raw classification `logits` and continuous
/// IoU-aware quality `targets` in `[0, 1]`, with exact `f64` analytical gradients w.r.t. `logits`.
///
/// ```text
/// L_VFL(x, q) = q * BCE(x, q)                        if q > 0
///             = alpha * sigmoid(x)^gamma * ln(1+e^x) if q == 0
/// ```
///
/// Supports [`LossReduction::None`], [`LossReduction::Sum`], and [`LossReduction::Mean`] via [`VflConfig`].
/// Errors on invalid config parameters, mismatched dtypes/devices/shapes, or finite targets outside `[0, 1]`.
pub fn varifocal_loss(logits: &Tensor, targets: &Tensor, cfg: &VflConfig) -> Result<Tensor> {
    cfg.validate()?;
    let (alpha, gamma) = (cfg.alpha as f64, cfg.gamma as f64);
    quality_cls_op(logits, targets, cfg.reduction, "varifocal_loss", move |x, q| {
        vfl_elem_f64(x, q, alpha, gamma)
    })
}

/// Standalone Quality Focal Loss (QFL, Li et al. NeurIPS 2020) for raw classification `logits` and
/// continuous IoU-aware quality `targets` in `[0, 1]`, with exact `f64` analytical gradients w.r.t. `logits`.
///
/// ```text
/// L_QFL(x, q) = |q - sigmoid(x)|^beta * BCE(x, q)
/// ```
///
/// Supports [`LossReduction::None`], [`LossReduction::Sum`], and [`LossReduction::Mean`] via [`QflConfig`].
/// Errors on invalid config parameters, mismatched dtypes/devices/shapes, or finite targets outside `[0, 1]`.
pub fn quality_focal_loss(logits: &Tensor, targets: &Tensor, cfg: &QflConfig) -> Result<Tensor> {
    cfg.validate()?;
    let beta = cfg.beta as f64;
    quality_cls_op(logits, targets, cfg.reduction, "quality_focal_loss", move |x, q| qfl_elem_f64(x, q, beta))
}
