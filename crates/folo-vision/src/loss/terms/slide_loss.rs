//! SlideLoss and EMA-SlideLoss (Yu et al., YOLO-FaceV2, Pattern Recognition 2024): adaptive sample
//! re-weighting around a dynamic or EMA-tracked IoU boundary `mu` to emphasize hard boundary
//! examples without manual hyperparameter tuning of per-class focal thresholds.
//!
//! For a sample with IoU quality `x in [0, 1]`, boundary threshold `mu in [0, 1]`, and transition
//! width `delta in (0, 1]` (default `0.1`):
//!
//! - [`SlideWeightKind::Standard`] (exact YOLO-FaceV2 piecewise formula):
//!   ```text
//!   w(x; mu) = 1.0              if x <= mu - delta
//!            = exp(1.0 - mu)    if mu - delta < x < mu
//!            = exp(1.0 - x)     if x >= mu
//!   ```
//! - [`SlideWeightKind::Smooth`] (default; continuous everywhere, C¹-smooth across `mu - delta` via
//!   a cubic Hermite ramp on `(mu - delta, mu)` meeting `exp(1 - x)` continuously at `x = mu` with
//!   left slope `0` and right slope `-exp(1 - mu)`):
//!   ```text
//!   w(x; mu) = 1.0                                         if x <= mu - delta
//!            = 1.0 + (exp(1.0 - mu) - 1.0) * (3t² - 2t³)   if mu - delta < x < mu, t = (x - (mu - delta)) / delta
//!            = exp(1.0 - x)                                if x >= mu
//!   ```
//!
//! Both weighting functions are bounded in `[1.0, e]`, peak at the hard boundary `x = mu` with
//! `w(mu) = exp(1 - mu)`, and decay smoothly to `1.0` at `x = 1.0`. Sample weights are detached
//! from the backward pass, so every classification logit receives a strictly descent-aligned
//! cancellation-free BCE gradient `w_i * (sigmoid(x_i) - q_i)` (or `w_i * dL_base/dz_i` when
//! wrapping an arbitrary differentiable loss via [`slide_reweight_loss`]).
//!
//! [`EmaSlideLoss`] maintains an exponential moving average of `mu` across batches with optional
//! warmup decay `d_eff(t) = ema_decay * (1 - exp(-t / ema_tau))`, matching YOLO EMA-SlideLoss.

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::vfl_qfl::LossReduction;
use crate::loss::invalid;

/// Default SlideLoss transition margin `delta` around `mu` (Yu et al., 2024).
pub const DEFAULT_SLIDE_DELTA: f32 = 0.1;

/// Shape of the SlideLoss weighting curve across `(mu - delta, mu)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlideWeightKind {
    /// Continuous everywhere with a cubic Hermite ramp from `1.0` at `mu - delta` (zero slope) to
    /// `exp(1 - mu)` at `mu` (zero left slope), meeting `exp(1 - x)` continuously for `x >= mu`.
    /// Eliminates the step discontinuity of [`SlideWeightKind::Standard`] at `mu - delta`.
    #[default]
    Smooth,
    /// Exact piecewise YOLO-FaceV2 SlideLoss: `1.0` for `x <= mu - delta`, `exp(1 - mu)` on
    /// `(mu - delta, mu)`, and `exp(1 - x)` for `x >= mu`.
    Standard,
}

/// Configuration for [`slide_bce_loss`], [`slide_reweight_loss`], and [`EmaSlideLoss`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SlideLossConfig {
    /// Width `delta in (0, 1]` of the hard-boundary transition band below `mu` (default `0.1`).
    pub delta: f32,
    /// Weighting curve shape (`Smooth` default or `Standard`).
    pub weight_kind: SlideWeightKind,
    /// Output reduction mode (default [`LossReduction::Mean`]).
    pub reduction: LossReduction,
}

impl Default for SlideLossConfig {
    fn default() -> Self {
        Self {
            delta: DEFAULT_SLIDE_DELTA,
            weight_kind: SlideWeightKind::Smooth,
            reduction: LossReduction::Mean,
        }
    }
}

impl SlideLossConfig {
    /// Validate `delta` (finite and in `(0, 1]`).
    pub fn validate(&self) -> Result<()> {
        if !(self.delta.is_finite() && self.delta > 0.0 && self.delta <= 1.0) {
            return Err(invalid(format!(
                "slide_loss: delta must be finite and in (0, 1], got {}",
                self.delta
            )));
        }
        Ok(())
    }
}

/// Evaluate the SlideLoss sample weight `w(iou; mu, delta)` in `f64`.
///
/// Returns `f64::NAN` if any input is non-finite, `iou` or `mu` is outside `[0, 1]`, or `delta`
/// is outside `(0, 1]`. For all valid inputs, returns a finite weight in `[1.0, e]`.
#[inline]
pub fn slide_weight_f64(iou: f64, mu: f64, delta: f64, kind: SlideWeightKind) -> f64 {
    if !(iou.is_finite()
        && mu.is_finite()
        && delta.is_finite()
        && (0.0..=1.0).contains(&iou)
        && (0.0..=1.0).contains(&mu)
        && delta > 0.0
        && delta <= 1.0)
    {
        return f64::NAN;
    }
    let low = mu - delta;
    if iou <= low {
        1.0
    } else if iou < mu {
        let peak = (1.0 - mu).exp();
        match kind {
            SlideWeightKind::Standard => peak,
            SlideWeightKind::Smooth => {
                let t = ((iou - low) / delta).clamp(0.0, 1.0);
                let s = t * t * (3.0 - 2.0 * t);
                1.0 + (peak - 1.0) * s
            }
        }
    } else {
        (1.0 - iou).exp()
    }
}

/// Evaluate the SlideLoss sample weight `w(iou; mu, delta)` in `f32` (delegates to [`slide_weight_f64`]).
#[inline]
pub fn slide_weight(iou: f32, mu: f32, delta: f32, kind: SlideWeightKind) -> f32 {
    slide_weight_f64(iou as f64, mu as f64, delta as f64, kind) as f32
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

/// Cancellation-free `BCE(x, q)` and `d(BCE)/dx = p - q` in `f64` for finite `x` and `q in [0, 1]`.
#[inline]
pub(crate) fn slide_bce_elem_f64(x: f64, q: f64, weight: f64) -> (f64, f64) {
    if !x.is_finite() || !q.is_finite() || !weight.is_finite() {
        return (f64::NAN, f64::NAN);
    }
    let pos = if q == 0.0 { 0.0 } else { q * softplus_f64(-x) };
    let neg = if q == 1.0 { 0.0 } else { (1.0 - q) * softplus_f64(x) };
    let bce = pos + neg;
    let p = sigmoid_f64(x);
    let r = sigmoid_f64(-x);
    let diff = if x.abs() <= 1.0 {
        ((1.0 - 2.0 * q) - q * (-x).exp_m1()) / (1.0 + (-x).exp())
    } else if x > 0.0 {
        (1.0 - q) - r
    } else {
        p - q
    };
    (weight * bce, weight * diff)
}

fn check_same_f32_shape(a: &Tensor, b: &Tensor, op: &'static str, b_name: &'static str) -> Result<()> {
    if a.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch { op, expected: DType::F32, got: a.dtype() });
    }
    if b.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch { op, expected: DType::F32, got: b.dtype() });
    }
    if a.device() != b.device() {
        return Err(Error::DeviceMismatch { op, lhs: a.device(), rhs: b.device() });
    }
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument {
            op,
            msg: format!("shape mismatch: {} vs {b_name} {}", a.shape(), b.shape()),
        });
    }
    Ok(())
}

/// Validate that every finite entry in `slice` lies within `[0, 1]`.
fn validate_unit_interval(slice: &[f32], op: &'static str, name: &'static str) -> Result<()> {
    for (i, &v) in slice.iter().enumerate() {
        if v.is_finite() && !(0.0..=1.0).contains(&v) {
            return Err(Error::InvalidArgument {
                op,
                msg: format!("{name}[{i}] must be in [0, 1], got {v}"),
            });
        }
    }
    Ok(())
}

/// Compute batch mean IoU over positive elements (`mask[i] == true`, or all elements when
/// `mask == None`), ignoring entries only when no positives exist. Returns `Ok(None)` when there
/// are no positive samples; returns `Ok(Some(f64::NAN))` if any positive sample has a non-finite
/// IoU.
fn positive_mean_iou(ious: &[f32], pos_targets: Option<&[f32]>, op: &'static str) -> Result<Option<f64>> {
    validate_unit_interval(ious, op, "iou")?;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    match pos_targets {
        Some(ts) => {
            for (&iou, &t) in ious.iter().zip(ts) {
                if t.is_finite() && t > 0.0 {
                    if !iou.is_finite() {
                        return Ok(Some(f64::NAN));
                    }
                    sum += iou as f64;
                    count += 1;
                }
            }
        }
        None => {
            for &iou in ious {
                if !iou.is_finite() {
                    return Ok(Some(f64::NAN));
                }
                sum += iou as f64;
                count += 1;
            }
        }
    }
    if count == 0 { Ok(None) } else { Ok(Some((sum / count as f64).clamp(0.0, 1.0))) }
}

/// Differentiable SlideLoss over classification `logits`, continuous targets `targets` in `[0, 1]`,
/// and per-element IoU qualities `ious` in `[0, 1]` (all matching shapes).
///
/// - When `mu == Some(m)`: uses explicit threshold `m in [0, 1]` (e.g. from [`EmaSlideLoss`]).
/// - When `mu == None`: dynamically sets `mu` to the batch mean of `ious` over positive entries
///   (`targets > 0`); if a batch has no positive targets, falls back to unweighted BCE (`weight = 1.0`)
///   for all finite entries.
///
/// Evaluates in `f64` with cancellation-free BCE and exact analytical gradients w.r.t. `logits`
/// (sample weights are detached). Non-finite logits, targets, or IoUs propagate `NaN` cleanly.
pub fn slide_bce_loss(
    logits: &Tensor,
    targets: &Tensor,
    ious: &Tensor,
    mu: Option<f32>,
    cfg: &SlideLossConfig,
) -> Result<Tensor> {
    const OP: &str = "slide_bce_loss";
    cfg.validate()?;
    if let Some(m) = mu
        && !(m.is_finite() && (0.0..=1.0).contains(&m))
    {
        return Err(invalid(format!("{OP}: mu must be finite and in [0, 1], got {m}")));
    }
    check_same_f32_shape(logits, targets, OP, "targets")?;
    check_same_f32_shape(logits, ious, OP, "ious")?;

    let dev = logits.device();
    let shape = logits.shape().clone();
    let n = shape.numel();
    if n == 0 {
        let out_t = match cfg.reduction {
            LossReduction::None => Tensor::zeros_with(&shape, DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        return Tensor::custom_op(OP, &[logits], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with(&shape, DType::F32, dev)?)])
        });
    }

    let xv = logits.to_vec::<f32>()?;
    let tv = targets.to_vec::<f32>()?;
    let iv = ious.to_vec::<f32>()?;
    validate_unit_interval(&tv, OP, "target")?;
    validate_unit_interval(&iv, OP, "iou")?;

    let eff_mu = match mu {
        Some(m) => Some(m as f64),
        None => positive_mean_iou(&iv, Some(&tv), OP)?,
    };
    let delta64 = cfg.delta as f64;
    let kind = cfg.weight_kind;

    let mut dldx = vec![0.0f64; n];
    match cfg.reduction {
        LossReduction::None => {
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let w = match eff_mu {
                    Some(m) => slide_weight_f64(iv[i] as f64, m, delta64, kind),
                    None => {
                        if iv[i].is_finite() {
                            1.0
                        } else {
                            f64::NAN
                        }
                    }
                };
                let (lv, gv) = slide_bce_elem_f64(xv[i] as f64, tv[i] as f64, w);
                out[i] = lv as f32;
                dldx[i] = gv;
            }
            let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
            Tensor::custom_op(OP, &[logits], out_t, move |g| {
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
                let w = match eff_mu {
                    Some(m) => slide_weight_f64(iv[i] as f64, m, delta64, kind),
                    None => {
                        if iv[i].is_finite() {
                            1.0
                        } else {
                            f64::NAN
                        }
                    }
                };
                let (lv, gv) = slide_bce_elem_f64(xv[i] as f64, tv[i] as f64, w);
                sum += lv;
                dldx[i] = gv;
            }
            let scale = if cfg.reduction == LossReduction::Mean { 1.0 / (n as f64) } else { 1.0 };
            let out_t = Tensor::scalar((sum * scale) as f32).to_device(dev)?;
            Tensor::custom_op(OP, &[logits], out_t, move |g| {
                let gs = (g.item::<f32>()? as f64) * scale;
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

/// Composable SlideLoss sample re-weighting wrapper for any differentiable per-element `base_loss`
/// tensor (e.g. BCE, VFL, QFL, ASL, or box regression losses) with matching `ious` in `[0, 1]`
/// and threshold `mu` in `[0, 1]`.
///
/// Computes `w(ious; mu) * base_loss` with detached Slide weights and applies `cfg.reduction`.
/// Gradients flow cleanly through `base_loss`.
pub fn slide_reweight_loss(
    base_loss: &Tensor,
    ious: &Tensor,
    mu: f32,
    cfg: &SlideLossConfig,
) -> Result<Tensor> {
    if !(mu.is_finite() && (0.0..=1.0).contains(&mu)) {
        return Err(invalid(format!("slide_reweight_loss: mu must be finite and in [0, 1], got {mu}")));
    }
    slide_reweight_loss_opt(base_loss, ious, Some(mu), cfg)
}

fn slide_reweight_loss_opt(
    base_loss: &Tensor,
    ious: &Tensor,
    mu: Option<f32>,
    cfg: &SlideLossConfig,
) -> Result<Tensor> {
    const OP: &str = "slide_reweight_loss";
    cfg.validate()?;
    check_same_f32_shape(base_loss, ious, OP, "ious")?;

    let dev = base_loss.device();
    let shape = base_loss.shape().clone();
    let n = shape.numel();
    if n == 0 {
        let out_t = match cfg.reduction {
            LossReduction::None => Tensor::zeros_with(&shape, DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        return Tensor::custom_op(OP, &[base_loss], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with(&shape, DType::F32, dev)?)])
        });
    }

    let lv = base_loss.to_vec::<f32>()?;
    let iv = ious.to_vec::<f32>()?;
    validate_unit_interval(&iv, OP, "iou")?;
    let delta64 = cfg.delta as f64;
    let kind = cfg.weight_kind;

    let weights: Vec<f64> = match mu {
        Some(m) => {
            let mu64 = m as f64;
            iv.iter().map(|&iou| slide_weight_f64(iou as f64, mu64, delta64, kind)).collect()
        }
        None => iv.iter().map(|&iou| if iou.is_finite() { 1.0 } else { f64::NAN }).collect(),
    };

    match cfg.reduction {
        LossReduction::None => {
            let out: Vec<f32> = lv.iter().zip(&weights).map(|(&l, &w)| (l as f64 * w) as f32).collect();
            let out_t = Tensor::from_vec(out, &shape)?.to_device(dev)?;
            Tensor::custom_op(OP, &[base_loss], out_t, move |g| {
                let gv = g.to_vec::<f32>()?;
                let mut grad = vec![0.0f32; n];
                for i in 0..n {
                    let gi = gv[i] as f64;
                    if gi == 0.0 && weights[i].is_finite() {
                        continue;
                    }
                    grad[i] = (gi * weights[i]) as f32;
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })
        }
        LossReduction::Sum | LossReduction::Mean => {
            let sum: f64 = lv.iter().zip(&weights).map(|(&l, &w)| l as f64 * w).sum();
            let scale = if cfg.reduction == LossReduction::Mean { 1.0 / (n as f64) } else { 1.0 };
            let out_t = Tensor::scalar((sum * scale) as f32).to_device(dev)?;
            Tensor::custom_op(OP, &[base_loss], out_t, move |g| {
                let gs = (g.item::<f32>()? as f64) * scale;
                let mut grad = vec![0.0f32; n];
                for i in 0..n {
                    if gs == 0.0 && weights[i].is_finite() {
                        continue;
                    }
                    grad[i] = (gs * weights[i]) as f32;
                }
                Ok(vec![Some(Tensor::from_vec(grad, &shape)?.to_device(dev)?)])
            })
        }
    }
}

/// Stateful EMA-SlideLoss tracker maintaining an exponential moving average of the positive-sample
/// IoU boundary threshold `mu` across training iterations.
///
/// On each update step `t = 1, 2, ...` with batch positive mean IoU `mu_batch`:
/// ```text
/// d_t  = ema_decay * (1 - exp(-t / ema_tau))   if ema_tau > 0
///      = ema_decay                             if ema_tau == 0
/// mu_t = d_t * mu_{t-1} + (1 - d_t) * mu_batch
/// ```
///
/// When created with [`EmaSlideLoss::new`], `mu` initializes directly to the first batch's
/// positive mean IoU (eliminating cold-start bias); alternatively [`EmaSlideLoss::with_initial_mu`]
/// seeds `mu` explicitly. Batches with zero positive samples leave EMA state unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmaSlideLoss {
    cfg: SlideLossConfig,
    ema_decay: f32,
    ema_tau: f32,
    mu: f64,
    initial_mu: Option<f64>,
    updates: u64,
    initialized: bool,
}

impl EmaSlideLoss {
    /// Create an EMA-SlideLoss tracker with lazy first-batch initialization of `mu`.
    ///
    /// - `ema_decay` must be finite and in `[0, 1)` (e.g. `0.99` or `0.9999`).
    /// - `ema_tau` must be finite and `>= 0.0` (`0.0` = constant decay; `> 0.0` = exponential
    ///   warmup `ema_decay * (1 - exp(-step / ema_tau))`).
    pub fn new(cfg: SlideLossConfig, ema_decay: f32, ema_tau: f32) -> Result<Self> {
        cfg.validate()?;
        if !(ema_decay.is_finite() && (0.0..1.0).contains(&ema_decay)) {
            return Err(invalid(format!(
                "EmaSlideLoss: ema_decay must be finite and in [0, 1), got {ema_decay}"
            )));
        }
        if !(ema_tau.is_finite() && ema_tau >= 0.0) {
            return Err(invalid(format!("EmaSlideLoss: ema_tau must be finite and >= 0, got {ema_tau}")));
        }
        Ok(Self { cfg, ema_decay, ema_tau, mu: 0.5, initial_mu: None, updates: 0, initialized: false })
    }

    /// Create an EMA-SlideLoss tracker seeded with an explicit `initial_mu` in `[0, 1]`.
    pub fn with_initial_mu(
        cfg: SlideLossConfig,
        ema_decay: f32,
        ema_tau: f32,
        initial_mu: f32,
    ) -> Result<Self> {
        if !(initial_mu.is_finite() && (0.0..=1.0).contains(&initial_mu)) {
            return Err(invalid(format!(
                "EmaSlideLoss: initial_mu must be finite and in [0, 1], got {initial_mu}"
            )));
        }
        let mut s = Self::new(cfg, ema_decay, ema_tau)?;
        s.mu = initial_mu as f64;
        s.initial_mu = Some(initial_mu as f64);
        s.initialized = true;
        Ok(s)
    }

    /// Underlying [`SlideLossConfig`].
    #[inline]
    pub fn config(&self) -> SlideLossConfig {
        self.cfg
    }

    /// Current EMA threshold `mu` in `[0, 1]`.
    #[inline]
    pub fn mu(&self) -> f32 {
        self.mu as f32
    }

    /// Current EMA threshold `mu` in full `f64` precision.
    #[inline]
    pub fn mu_f64(&self) -> f64 {
        self.mu
    }

    /// Number of batches that updated the EMA state.
    #[inline]
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Whether `mu` has been seeded (either via [`EmaSlideLoss::with_initial_mu`] or an observed batch).
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Base EMA decay factor.
    #[inline]
    pub fn ema_decay(&self) -> f32 {
        self.ema_decay
    }

    /// EMA warmup time constant `tau` (`0.0` means constant decay).
    #[inline]
    pub fn ema_tau(&self) -> f32 {
        self.ema_tau
    }

    /// Effective EMA decay for the next update (`updates + 1`).
    #[inline]
    pub fn effective_decay(&self) -> f64 {
        let base = self.ema_decay as f64;
        if self.ema_tau <= 0.0 {
            base
        } else {
            let step = (self.updates + 1) as f64;
            base * (-(-step / self.ema_tau as f64).exp_m1())
        }
    }

    /// Reset EMA update counter and threshold back to construction state.
    pub fn reset(&mut self) {
        self.updates = 0;
        if let Some(init) = self.initial_mu {
            self.mu = init;
            self.initialized = true;
        } else {
            self.mu = 0.5;
            self.initialized = false;
        }
    }

    /// Update EMA `mu` from a slice of IoU values (optionally restricted to `mask[i] == true`).
    ///
    /// - If `mask` is `Some(m)` with `m.len() != ious.len()`, returns an error.
    /// - Finite IoUs outside `[0, 1]` return an error.
    /// - If any IoU in the batch is non-finite, or if no entries are selected, EMA state is left
    ///   unchanged and the current `mu` is returned.
    pub fn update_mu(&mut self, ious: &[f32], mask: Option<&[bool]>) -> Result<f32> {
        const OP: &str = "EmaSlideLoss::update_mu";
        if let Some(m) = mask
            && m.len() != ious.len()
        {
            return Err(invalid(format!("{OP}: mask len {} != ious len {}", m.len(), ious.len())));
        }
        validate_unit_interval(ious, OP, "iou")?;
        if !ious.iter().all(|v| v.is_finite()) {
            return Ok(self.mu());
        }
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for (i, &iou) in ious.iter().enumerate() {
            if mask.is_none_or(|m| m[i]) {
                sum += iou as f64;
                count += 1;
            }
        }
        if count > 0 {
            let batch_mu = (sum / count as f64).clamp(0.0, 1.0);
            self.apply_batch_mu(batch_mu);
        }
        Ok(self.mu())
    }

    fn apply_batch_mu(&mut self, batch_mu: f64) {
        if !batch_mu.is_finite() {
            return;
        }
        let d = self.effective_decay();
        self.updates += 1;
        if !self.initialized {
            self.mu = batch_mu;
            self.initialized = true;
        } else {
            self.mu = (d * self.mu + (1.0 - d) * batch_mu).clamp(0.0, 1.0);
        }
    }

    /// Update EMA `mu` from positive entries (`targets > 0`) and compute differentiable Slide BCE
    /// loss with the updated `mu`.
    ///
    /// If the batch has no positive targets or contains any non-finite target/IoU entry, EMA state
    /// is preserved cleanly; non-finite tensor entries still propagate `NaN` through the returned loss.
    pub fn compute_bce(&mut self, logits: &Tensor, targets: &Tensor, ious: &Tensor) -> Result<Tensor> {
        const OP: &str = "EmaSlideLoss::compute_bce";
        check_same_f32_shape(logits, targets, OP, "targets")?;
        check_same_f32_shape(logits, ious, OP, "ious")?;
        if logits.numel() > 0 {
            let tv = targets.to_vec::<f32>()?;
            let iv = ious.to_vec::<f32>()?;
            validate_unit_interval(&tv, OP, "target")?;
            validate_unit_interval(&iv, OP, "iou")?;
            if iv.iter().all(|v| v.is_finite())
                && tv.iter().all(|v| v.is_finite())
                && let Some(batch_mu) = positive_mean_iou(&iv, Some(&tv), OP)?
                && batch_mu.is_finite()
            {
                self.apply_batch_mu(batch_mu);
            }
        }
        let mu_arg = if self.initialized { Some(self.mu()) } else { None };
        slide_bce_loss(logits, targets, ious, mu_arg, &self.cfg)
    }

    /// Evaluate Slide BCE loss using the current EMA `mu` without mutating EMA state (for eval/val).
    pub fn eval_bce(&self, logits: &Tensor, targets: &Tensor, ious: &Tensor) -> Result<Tensor> {
        let mu_arg = if self.initialized { Some(self.mu()) } else { None };
        slide_bce_loss(logits, targets, ious, mu_arg, &self.cfg)
    }

    /// Update EMA `mu` from `ious` (optionally masked by `pos_mask`) and re-weight `base_loss`.
    ///
    /// Always validates `pos_mask` length (including on empty tensors), leaves EMA state untouched
    /// when any IoU is non-finite or no positive entries are selected, and falls back to unweighted
    /// `1.0` weights (preserving `NaN` propagation) until EMA `mu` is initialized.
    pub fn reweight(
        &mut self,
        base_loss: &Tensor,
        ious: &Tensor,
        pos_mask: Option<&[bool]>,
    ) -> Result<Tensor> {
        const OP: &str = "EmaSlideLoss::reweight";
        check_same_f32_shape(base_loss, ious, OP, "ious")?;
        let n = ious.numel();
        if let Some(m) = pos_mask
            && m.len() != n
        {
            return Err(invalid(format!("{OP}: mask len {} != ious len {n}", m.len())));
        }
        if n > 0 {
            let iv = ious.to_vec::<f32>()?;
            self.update_mu(&iv, pos_mask)?;
        }
        let mu_arg = if self.initialized { Some(self.mu()) } else { None };
        slide_reweight_loss_opt(base_loss, ious, mu_arg, &self.cfg)
    }
}
