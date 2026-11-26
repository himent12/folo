//! Seesaw Loss for long-tailed object detection & classification (Wang et al. CVPR 2021, arXiv:2008.10032).
//!
//! For a sample with raw class logits `z in R^C`, target class `y in 0..C`, and cumulative per-class sample
//! counts `N_0, ..., N_{C-1}`:
//!
//! - **Mitigation factor `M_{y,j}`** (`j != y`): attenuates overwhelming negative gradients from frequent
//!   classes onto rare classes:
//!   ```text
//!   M_{y,j} = (N_j / N_y)^p    if N_j < N_y
//!           = 1                otherwise
//!   ```
//!   (floored by `eps` in `[0, 1]`; when `eps == 0` and `N_j == 0 < N_y`, class `j` is cleanly masked out
//!   of the softmax denominator).
//! - **Compensation factor `C_{y,j}`** (`j != y`): restores penalty whenever a false positive class `j`
//!   scores higher than the ground-truth class `y`:
//!   ```text
//!   C_{y,j} = (sigma_j / sigma_y)^q = exp(q * (z_j - z_y))    if z_j > z_y
//!           = 1                                                otherwise
//!   ```
//!   Evaluating `ln(C_{y,j}) = q * max(z_j - z_y, 0)` directly in log-space eliminates `0 / 0` softmax
//!   underflow for extreme logits.
//! - **Loss:**
//!   ```text
//!   L_seesaw(z, y) = -ln( exp(z_y) / (exp(z_y) + sum_{j != y} M_{y,j} * C_{y,j} * exp(z_j)) )
//!                  = logsumexp_k(a_k) - z_y
//!   ```
//!   where `a_y = z_y` and `a_j = z_j + ln(M_{y,j}) + ln(C_{y,j})`.
//!
//! By default (`SeesawConfig::detach_compensation = true`), `C_{y,j}` is treated as detached during
//! backpropagation (matching Wang et al. CVPR 2021 Eq. 5), yielding `dL/dz_j = p_hat_j - 1[j == y]` with
//! cancellation-free `dL/dz_y = -sum_{j != y} p_hat_j`. Setting `detach_compensation = false` differentiates
//! directly through `q * max(z_j - z_y, 0)`.

use folo_core::{DType, Error, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::vfl_qfl::LossReduction;
use crate::loss::invalid;

/// Hyper-parameters for [`seesaw_loss`] and [`SeesawLoss`] (Wang et al. CVPR 2021).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeesawConfig {
    /// Mitigation factor exponent `p >= 0` (paper default `0.8`; `0.0` disables mitigation).
    pub p: f32,
    /// Compensation factor exponent `q >= 0` (paper default `2.0`; `0.0` disables compensation).
    pub q: f32,
    /// Floor `eps in [0, 1]` applied to the mitigation factor `M_{y,j}` (default `1e-6`). When `eps == 0.0`
    /// and `N_j == 0 < N_y`, class `j` is excluded from the denominator cleanly with zero gradient.
    pub eps: f32,
    /// Detach the compensation factor `C_{y,j}` during backpropagation (default `true`, matching Wang et al.
    /// CVPR 2021 Eq. 5). Set `false` for end-to-end differentiation through `q * max(z_j - z_y, 0)`.
    pub detach_compensation: bool,
    /// Output reduction mode (default [`LossReduction::Mean`]).
    pub reduction: LossReduction,
}

impl Default for SeesawConfig {
    fn default() -> Self {
        Self { p: 0.8, q: 2.0, eps: 1e-6, detach_compensation: true, reduction: LossReduction::Mean }
    }
}

impl SeesawConfig {
    /// Validate all Seesaw Loss hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.p.is_finite() && self.p >= 0.0) {
            return Err(invalid(format!("seesaw_loss: p must be finite and >= 0, got {}", self.p)));
        }
        if !(self.q.is_finite() && self.q >= 0.0) {
            return Err(invalid(format!("seesaw_loss: q must be finite and >= 0, got {}", self.q)));
        }
        if !(self.eps.is_finite() && (0.0..=1.0).contains(&self.eps)) {
            return Err(invalid(format!("seesaw_loss: eps must be finite and in [0, 1], got {}", self.eps)));
        }
        Ok(())
    }
}

/// Evaluate one row of Seesaw Loss `(loss, dL/dz)` in `f64`.
///
/// Returns `(NaN, [NaN; C])` if any logit in `z` is non-finite.
pub(crate) fn seesaw_row_f64(
    z: &[f64],
    target: usize,
    counts: &[f64],
    p: f64,
    q: f64,
    eps: f64,
    detach_compensation: bool,
) -> (f64, Vec<f64>) {
    let c = z.len();
    if !z.iter().all(|v| v.is_finite()) {
        return (f64::NAN, vec![f64::NAN; c]);
    }
    let zy = z[target];
    let ny = counts[target];
    let log_eps = if eps > 0.0 { eps.ln() } else { f64::NEG_INFINITY };

    let mut adj_rel = vec![f64::NEG_INFINITY; c];
    let mut comp_active = vec![false; c];
    adj_rel[target] = 0.0;

    for j in 0..c {
        if j == target {
            continue;
        }
        let log_m = if p == 0.0 || counts[j] >= ny {
            0.0
        } else if counts[j] == 0.0 {
            if eps == 0.0 {
                // Masked out of softmax denominator.
                continue;
            }
            log_eps
        } else {
            let raw = p * (counts[j].ln() - ny.ln());
            if eps > 0.0 { raw.max(log_eps) } else { raw }
        };

        let dz = z[j] - zy;
        let log_c = if q > 0.0 && dz > 0.0 {
            comp_active[j] = true;
            q * dz
        } else {
            0.0
        };
        // Shift relative to z[target] before adding log_m / log_c so large shared offsets never swallow log_m.
        adj_rel[j] = dz + log_m + log_c;
    }

    let max_a = adj_rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Cancellation-free logsumexp(adj_rel): when target achieves max_a (0.0), compute ln_1p(sum_{j!=y} exp(a_j)).
    let mut sum_other = 0.0f64;
    let mut exp_vals = vec![0.0f64; c];
    for j in 0..c {
        if adj_rel[j].is_finite() {
            let e = (adj_rel[j] - max_a).exp();
            exp_vals[j] = e;
            if j != target {
                sum_other += e;
            }
        }
    }
    let sum_all = exp_vals[target] + sum_other;
    let loss = if max_a == 0.0 { sum_other.ln_1p() } else { max_a + sum_all.ln() };

    // Cancellation-free gradient: sum_j dL/dz_j == 0 identically, so dL/dz_y = -sum_{j != y} dL/dz_j.
    let mut grad = vec![0.0f64; c];
    let inv_sum = 1.0 / sum_all;
    let mut grad_other_sum = 0.0f64;
    for j in 0..c {
        if j == target {
            continue;
        }
        let phat = exp_vals[j] * inv_sum;
        let g = if !detach_compensation && comp_active[j] { phat * (1.0 + q) } else { phat };
        grad[j] = g;
        grad_other_sum += g;
    }
    grad[target] = -grad_other_sum;
    (loss, grad)
}

/// Stateless Seesaw Loss (Wang et al. CVPR 2021) for raw `[N, C]` `F32` classification `logits`, integer
/// class `targets` (`len == N`, each `< C`), and cumulative per-class counts `class_counts` (`len == C`,
/// finite and `>= 0`).
///
/// Supports [`LossReduction::None`] (`[N]`), [`LossReduction::Sum`] (`[]`), and [`LossReduction::Mean`]
/// (`[]`). Evaluates in `f64` with exact analytical gradients flowing into `logits`.
///
/// # Errors
/// Returns [`Error::DTypeMismatch`] if `logits` is not `F32`, or [`Error::InvalidArgument`] on invalid
/// config parameters, non-rank-2 `logits`, zero classes, length mismatches, out-of-range targets, or
/// non-finite/negative `class_counts`.
pub fn seesaw_loss(
    logits: &Tensor,
    targets: &[usize],
    class_counts: &[f64],
    cfg: &SeesawConfig,
) -> Result<Tensor> {
    cfg.validate()?;
    if logits.dtype() != DType::F32 {
        return Err(Error::DTypeMismatch { op: "seesaw_loss", expected: DType::F32, got: logits.dtype() });
    }
    let (n, c) = logits
        .dims2()
        .map_err(|_| invalid(format!("seesaw_loss: logits must be rank 2 [N, C], got {}", logits.shape())))?;
    if c == 0 {
        return Err(invalid("seesaw_loss: num_classes C must be >= 1, got 0"));
    }
    if targets.len() != n {
        return Err(invalid(format!("seesaw_loss: targets length {} != N ({n})", targets.len())));
    }
    if class_counts.len() != c {
        return Err(invalid(format!("seesaw_loss: class_counts length {} != C ({c})", class_counts.len())));
    }
    for (i, &y) in targets.iter().enumerate() {
        if y >= c {
            return Err(invalid(format!("seesaw_loss: targets[{i}] = {y} >= C ({c})")));
        }
    }
    for (j, &cnt) in class_counts.iter().enumerate() {
        if !(cnt.is_finite() && cnt >= 0.0) {
            return Err(invalid(format!(
                "seesaw_loss: class_counts[{j}] must be finite and >= 0, got {cnt}"
            )));
        }
    }

    let dev = logits.device();
    if n == 0 {
        let out_t = match cfg.reduction {
            LossReduction::None => Tensor::zeros_with([0], DType::F32, dev)?,
            LossReduction::Sum | LossReduction::Mean => Tensor::scalar(0.0).to_device(dev)?,
        };
        return Tensor::custom_op("seesaw_loss", &[logits], out_t, move |_g| {
            Ok(vec![Some(Tensor::zeros_with([0, c], DType::F32, dev)?)])
        });
    }

    let zv = logits.to_vec::<f32>()?;
    let (p, q, eps, detach) = (cfg.p as f64, cfg.q as f64, cfg.eps as f64, cfg.detach_compensation);
    let mut row_buf = vec![0.0f64; c];
    let mut row_grads = vec![0.0f64; n * c];

    match cfg.reduction {
        LossReduction::None => {
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..c {
                    row_buf[j] = zv[i * c + j] as f64;
                }
                let (lv, gv) = seesaw_row_f64(&row_buf, targets[i], class_counts, p, q, eps, detach);
                out[i] = lv as f32;
                row_grads[i * c..(i + 1) * c].copy_from_slice(&gv);
            }
            let out_t = Tensor::from_vec(out, [n])?.to_device(dev)?;
            Tensor::custom_op("seesaw_loss", &[logits], out_t, move |g| {
                let gv = g.to_vec::<f32>()?;
                let mut grad = vec![0.0f32; n * c];
                for i in 0..n {
                    let gi = gv[i] as f64;
                    let row = &row_grads[i * c..(i + 1) * c];
                    if gi == 0.0 && row.iter().all(|v| v.is_finite()) {
                        continue;
                    }
                    for j in 0..c {
                        grad[i * c + j] = (gi * row[j]) as f32;
                    }
                }
                Ok(vec![Some(Tensor::from_vec(grad, [n, c])?.to_device(dev)?)])
            })
        }
        LossReduction::Sum | LossReduction::Mean => {
            let mut sum = 0.0f64;
            for i in 0..n {
                for j in 0..c {
                    row_buf[j] = zv[i * c + j] as f64;
                }
                let (lv, gv) = seesaw_row_f64(&row_buf, targets[i], class_counts, p, q, eps, detach);
                sum += lv;
                row_grads[i * c..(i + 1) * c].copy_from_slice(&gv);
            }
            let scale = if cfg.reduction == LossReduction::Mean { 1.0 / n as f64 } else { 1.0 };
            let out_t = Tensor::scalar((sum * scale) as f32).to_device(dev)?;
            Tensor::custom_op("seesaw_loss", &[logits], out_t, move |g| {
                let gs = g.item::<f32>()? as f64 * scale;
                let mut grad = vec![0.0f32; n * c];
                for i in 0..n * c {
                    if gs == 0.0 && row_grads[i].is_finite() {
                        continue;
                    }
                    grad[i] = (gs * row_grads[i]) as f32;
                }
                Ok(vec![Some(Tensor::from_vec(grad, [n, c])?.to_device(dev)?)])
            })
        }
    }
}

/// Stateful Seesaw Loss tracker accumulating per-class cumulative sample counts across training steps.
///
/// By default ([`SeesawLoss::new`]), counts are Laplace-initialized to `1.0` per class (matching Wang et al.
/// CVPR 2021 / MMDetection) so ratios `N_j / N_y` stay strictly positive from step 0.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeesawLoss {
    cfg: SeesawConfig,
    counts: Vec<f64>,
}

impl SeesawLoss {
    /// Create a tracker for `num_classes >= 1` with all cumulative counts initialized to `1.0`.
    pub fn new(num_classes: usize, cfg: SeesawConfig) -> Result<Self> {
        cfg.validate()?;
        if num_classes == 0 {
            return Err(invalid("SeesawLoss::new: num_classes must be >= 1, got 0"));
        }
        Ok(Self { cfg, counts: vec![1.0; num_classes] })
    }

    /// Create a tracker with explicit initial per-class counts (finite and `>= 0`).
    pub fn with_counts(counts: Vec<f64>, cfg: SeesawConfig) -> Result<Self> {
        cfg.validate()?;
        if counts.is_empty() {
            return Err(invalid("SeesawLoss::with_counts: counts must be non-empty"));
        }
        for (j, &cnt) in counts.iter().enumerate() {
            if !(cnt.is_finite() && cnt >= 0.0) {
                return Err(invalid(format!(
                    "SeesawLoss::with_counts: counts[{j}] must be finite and >= 0, got {cnt}"
                )));
            }
        }
        Ok(Self { cfg, counts })
    }

    /// Current hyper-parameters.
    #[inline]
    pub fn config(&self) -> SeesawConfig {
        self.cfg
    }

    /// Current cumulative per-class sample counts.
    #[inline]
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Reset all cumulative per-class counts back to `1.0`.
    pub fn reset(&mut self) {
        self.counts.fill(1.0);
    }

    /// Accumulate a batch of integer class labels into the cumulative counts (validating all indices before
    /// mutating state).
    pub fn update_counts(&mut self, targets: &[usize]) -> Result<()> {
        let c = self.counts.len();
        for (i, &y) in targets.iter().enumerate() {
            if y >= c {
                return Err(invalid(format!("SeesawLoss::update_counts: targets[{i}] = {y} >= C ({c})")));
            }
        }
        for &y in targets {
            self.counts[y] += 1.0;
        }
        Ok(())
    }

    /// Compute Seesaw Loss using the current cumulative counts without modifying them (e.g. validation).
    pub fn loss(&self, logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
        seesaw_loss(logits, targets, &self.counts, &self.cfg)
    }

    /// Update cumulative class counts with `targets` (matching Wang et al. CVPR 2021) and compute Seesaw Loss.
    ///
    /// Validates `logits` and `targets` before mutating counts so an error leaves internal state unchanged.
    pub fn forward(&mut self, logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
        if logits.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch {
                op: "seesaw_loss",
                expected: DType::F32,
                got: logits.dtype(),
            });
        }
        let (n, c) = logits.dims2().map_err(|_| {
            invalid(format!("seesaw_loss: logits must be rank 2 [N, C], got {}", logits.shape()))
        })?;
        if c != self.counts.len() {
            return Err(invalid(format!(
                "SeesawLoss::forward: logits classes {c} != tracker num_classes {}",
                self.counts.len()
            )));
        }
        if targets.len() != n {
            return Err(invalid(format!("seesaw_loss: targets length {} != N ({n})", targets.len())));
        }
        for (i, &y) in targets.iter().enumerate() {
            if y >= c {
                return Err(invalid(format!("seesaw_loss: targets[{i}] = {y} >= C ({c})")));
            }
        }
        for &y in targets {
            self.counts[y] += 1.0;
        }
        seesaw_loss(logits, targets, &self.counts, &self.cfg)
    }
}
