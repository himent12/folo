//! Post-hoc confidence calibration via Temperature Scaling (Guo et al. 2017) and calibration quality metrics
//! (ECE, MCE, Brier score, and NLL).
//!
//! # Convex 1D Optimization in Inverse Temperature
//!
//! Temperature scaling divides raw logits `z` by a single positive scalar `T > 0` before applying sigmoid
//! (binary / object-detection scores) or softmax (multiclass classification). Because `z -> z / T` is strictly
//! order-preserving for `T > 0`, detection ranking (mAP) and top-1 accuracy are completely unchanged while
//! predicted probabilities become calibrated.
//!
//! Parameterizing by the **inverse temperature** `beta = 1 / T > 0` makes the negative log-likelihood
//! **globally convex** on `(0, +inf)` for both binary and multiclass models:
//!
//! - **Binary (sigmoid) NLL:**
//!   ```text
//!   L(beta)   = (1 / N) * sum_i [ softplus(beta * z_i) - y_i * beta * z_i ]
//!   L'(beta)  = (1 / N) * sum_i (p_i - y_i) * z_i,          p_i = sigmoid(beta * z_i)
//!   L''(beta) = (1 / N) * sum_i p_i * (1 - p_i) * z_i^2 >= 0
//!   ```
//! - **Multiclass (softmax) NLL:**
//!   ```text
//!   L(beta)   = (1 / N) * sum_i [ log(sum_c exp(beta * z_{i,c})) - beta * z_{i, y_i} ]
//!   L'(beta)  = (1 / N) * sum_i (E_{p_i}[z_i] - z_{i, y_i})
//!   L''(beta) = (1 / N) * sum_i Var_{p_i}(z_i) >= 0
//!   ```
//!
//! [`TemperatureScaler::fit_binary`], [`TemperatureScaler::fit_binary_targets`], and
//! [`TemperatureScaler::fit_multiclass`] minimize `L(beta)` over `beta in [1 / max_temp, 1 / min_temp]` in
//! `f64` using damped Newton-Raphson with exact 1st/2nd derivatives and Armijo backtracking line search.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Solver configuration for [`TemperatureScaler`] fitting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    /// Minimum allowed temperature `T_min > 0` (default `0.05`). Bounds `beta <= 1 / min_temp` on separable data.
    pub min_temp: f32,
    /// Maximum allowed temperature `T_max >= min_temp` (default `100.0`). Bounds `beta >= 1 / max_temp`.
    pub max_temp: f32,
    /// Maximum damped Newton-Raphson iterations (default `100`).
    pub max_iters: usize,
    /// Projected-gradient stopping tolerance `|L'(beta)| <= tol` in `f64` (default `1e-9`).
    pub tol: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self { min_temp: 0.05, max_temp: 100.0, max_iters: 100, tol: 1e-9 }
    }
}

impl CalibrationConfig {
    fn validate(&self) -> Result<()> {
        if !(self.min_temp.is_finite() && self.min_temp > 0.0) {
            return Err(Error::Config(format!(
                "calibration.min_temp must be finite and > 0, got {}",
                self.min_temp
            )));
        }
        if !(self.max_temp.is_finite() && self.max_temp >= self.min_temp) {
            return Err(Error::Config(format!(
                "calibration.max_temp ({}) must be finite and >= min_temp ({})",
                self.max_temp, self.min_temp
            )));
        }
        if self.max_iters == 0 {
            return Err(Error::Config("calibration.max_iters must be >= 1".into()));
        }
        if !(self.tol.is_finite() && self.tol > 0.0) {
            return Err(Error::Config(format!("calibration.tol must be finite and > 0, got {}", self.tol)));
        }
        Ok(())
    }
}

/// Result of fitting a [`TemperatureScaler`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemperatureFit {
    /// Fitted temperature scaler.
    pub scaler: TemperatureScaler,
    /// Uncalibrated NLL at `T = 1.0`.
    pub initial_nll: f64,
    /// Calibrated NLL at `scaler.temperature()`.
    pub final_nll: f64,
    /// Number of damped Newton-Raphson iterations executed.
    pub iterations: usize,
    /// Whether the KKT optimality condition (`|L'(beta)| <= tol` or active bound) was satisfied.
    pub converged: bool,
}

/// Post-hoc temperature scaling calibrator: maps raw logits `z` to `z / T` with `T > 0`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemperatureScaler {
    temperature: f32,
}

impl Default for TemperatureScaler {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl TemperatureScaler {
    /// Construct a scaler with an explicit positive finite temperature `T > 0`.
    pub fn new(temperature: f32) -> Result<Self> {
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(Error::Config(format!("temperature must be finite and > 0, got {temperature}")));
        }
        Ok(Self { temperature })
    }

    /// The scalar temperature `T > 0`.
    #[inline]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Fit temperature scaling on binary logits and boolean targets (`true` = positive, `false` = negative).
    pub fn fit_binary(logits: &[f32], labels: &[bool], cfg: &CalibrationConfig) -> Result<TemperatureFit> {
        if logits.len() != labels.len() {
            return Err(Error::Config(format!(
                "fit_binary: logits length ({}) != labels length ({})",
                logits.len(),
                labels.len()
            )));
        }
        let targets: Vec<f32> = labels.iter().map(|&y| if y { 1.0 } else { 0.0 }).collect();
        Self::fit_binary_targets(logits, &targets, cfg)
    }

    /// Fit temperature scaling on binary logits and continuous targets `y_i in [0, 1]` (e.g. IoU-quality scores).
    pub fn fit_binary_targets(
        logits: &[f32],
        targets: &[f32],
        cfg: &CalibrationConfig,
    ) -> Result<TemperatureFit> {
        cfg.validate()?;
        if logits.is_empty() {
            return Err(Error::Config("fit_binary_targets: logits must not be empty".into()));
        }
        if logits.len() != targets.len() {
            return Err(Error::Config(format!(
                "fit_binary_targets: logits length ({}) != targets length ({})",
                logits.len(),
                targets.len()
            )));
        }
        for (i, &z) in logits.iter().enumerate() {
            if !z.is_finite() {
                return Err(Error::Config(format!("fit_binary_targets: non-finite logit at index {i}: {z}")));
            }
        }
        for (i, &y) in targets.iter().enumerate() {
            if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
                return Err(Error::Config(format!(
                    "fit_binary_targets: target at index {i} must be in [0, 1], got {y}"
                )));
            }
        }
        solve_convex_1d(cfg, |beta| binary_nll_grad_hess(logits, targets, beta))
    }

    /// Fit temperature scaling on row-major multiclass logits `[N, num_classes]` and class indices `labels[i] < num_classes`.
    pub fn fit_multiclass(
        logits: &[f32],
        num_classes: usize,
        labels: &[usize],
        cfg: &CalibrationConfig,
    ) -> Result<TemperatureFit> {
        cfg.validate()?;
        if num_classes < 2 {
            return Err(Error::Config(format!(
                "fit_multiclass: num_classes must be >= 2, got {num_classes}"
            )));
        }
        if labels.is_empty() {
            return Err(Error::Config("fit_multiclass: labels must not be empty".into()));
        }
        let expected = labels
            .len()
            .checked_mul(num_classes)
            .ok_or_else(|| Error::Config("fit_multiclass: logits size overflow".into()))?;
        if logits.len() != expected {
            return Err(Error::Config(format!(
                "fit_multiclass: logits length ({}) != labels.len() * num_classes ({expected})",
                logits.len()
            )));
        }
        for (i, &z) in logits.iter().enumerate() {
            if !z.is_finite() {
                return Err(Error::Config(format!("fit_multiclass: non-finite logit at index {i}: {z}")));
            }
        }
        for (i, &y) in labels.iter().enumerate() {
            if y >= num_classes {
                return Err(Error::Config(format!(
                    "fit_multiclass: label[{i}] = {y} >= num_classes ({num_classes})"
                )));
            }
        }
        solve_convex_1d(cfg, |beta| multiclass_nll_grad_hess(logits, num_classes, labels, beta))
    }

    /// Scale a single raw logit by `1 / T`.
    #[inline]
    pub fn scale_logit(&self, logit: f32) -> f32 {
        (logit as f64 / self.temperature as f64) as f32
    }

    /// Calibrate a single binary logit to a probability `sigmoid(logit / T)`.
    #[inline]
    pub fn calibrate_binary(&self, logit: f32) -> f32 {
        if !logit.is_finite() {
            return if logit.is_nan() {
                f32::NAN
            } else if logit.is_sign_positive() {
                1.0
            } else {
                0.0
            };
        }
        sigmoid_f64(logit as f64 / self.temperature as f64) as f32
    }

    /// Calibrate an already-sigmoided probability `p in [0, 1]` via `sigmoid(logit(p) / T)`.
    ///
    /// Exact boundaries `0.0` and `1.0` are preserved; values outside `[0, 1]` are clamped first.
    #[inline]
    pub fn calibrate_prob(&self, prob: f32) -> f32 {
        if !prob.is_finite() {
            return f32::NAN;
        }
        if prob <= 0.0 {
            return 0.0;
        }
        if prob >= 1.0 {
            return 1.0;
        }
        let p = prob as f64;
        let logit = (p / (1.0 - p)).ln();
        sigmoid_f64(logit / self.temperature as f64) as f32
    }

    /// Calibrate a slice of binary logits to probabilities.
    pub fn calibrate_logits_binary(&self, logits: &[f32]) -> Vec<f32> {
        logits.iter().map(|&z| self.calibrate_binary(z)).collect()
    }

    /// Calibrate row-major multiclass logits `[N, num_classes]` to row-wise softmax probabilities.
    pub fn calibrate_logits_multiclass(&self, logits: &[f32], num_classes: usize) -> Result<Vec<f32>> {
        if num_classes == 0 || !logits.len().is_multiple_of(num_classes) {
            return Err(Error::Config(format!(
                "calibrate_logits_multiclass: logits length ({}) must be a multiple of num_classes ({num_classes} >= 1)",
                logits.len()
            )));
        }
        let inv_t = 1.0 / self.temperature as f64;
        let mut out = vec![0.0f32; logits.len()];
        for (row_in, row_out) in logits.chunks_exact(num_classes).zip(out.chunks_exact_mut(num_classes)) {
            if !row_in.iter().all(|v| v.is_finite()) {
                row_out.fill(f32::NAN);
                continue;
            }
            let max_x = row_in.iter().map(|&z| z as f64 * inv_t).fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for (&z, o) in row_in.iter().zip(row_out.iter_mut()) {
                let e = (z as f64 * inv_t - max_x).exp();
                *o = e as f32;
                denom += e;
            }
            let inv_denom = 1.0 / denom;
            for (&z, o) in row_in.iter().zip(row_out.iter_mut()) {
                *o = ((z as f64 * inv_t - max_x).exp() * inv_denom) as f32;
            }
        }
        Ok(out)
    }
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Exact binary NLL, 1st derivative `L'(beta)`, and 2nd derivative `L''(beta)` in `f64`.
pub(crate) fn binary_nll_grad_hess(logits: &[f32], targets: &[f32], beta: f64) -> (f64, f64, f64) {
    let mut nll = 0.0f64;
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    for (&zf, &yf) in logits.iter().zip(targets) {
        let z = zf as f64;
        let y = yf as f64;
        let x = beta * z;
        let ax = x.abs();
        let e_neg = (-ax).exp();
        // softplus(x) - y * x = max(x, 0) + ln(1 + exp(-|x|)) - y * x
        let sp = x.max(0.0) + e_neg.ln_1p();
        nll += sp - y * x;
        let p = if x >= 0.0 { 1.0 / (1.0 + e_neg) } else { e_neg / (1.0 + e_neg) };
        let one_plus = 1.0 + e_neg;
        let var = e_neg / (one_plus * one_plus);
        g += (p - y) * z;
        h += var * z * z;
    }
    let inv_n = 1.0 / logits.len() as f64;
    (nll * inv_n, g * inv_n, h * inv_n)
}

/// Exact multiclass softmax NLL, 1st derivative `L'(beta)`, and 2nd derivative `L''(beta)` in `f64`.
pub(crate) fn multiclass_nll_grad_hess(
    logits: &[f32],
    num_classes: usize,
    labels: &[usize],
    beta: f64,
) -> (f64, f64, f64) {
    let mut nll = 0.0f64;
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    let mut probs = vec![0.0f64; num_classes];
    for (row, &y) in logits.chunks_exact(num_classes).zip(labels) {
        let mut max_x = f64::NEG_INFINITY;
        for &zf in row {
            let x = beta * zf as f64;
            if x > max_x {
                max_x = x;
            }
        }
        let mut sum_e = 0.0f64;
        for (p, &zf) in probs.iter_mut().zip(row) {
            let e = (beta * zf as f64 - max_x).exp();
            *p = e;
            sum_e += e;
        }
        let inv_sum = 1.0 / sum_e;
        let mut mean_z = 0.0f64;
        for (p, &zf) in probs.iter_mut().zip(row) {
            *p *= inv_sum;
            mean_z += *p * zf as f64;
        }
        let mut var_z = 0.0f64;
        for (&p, &zf) in probs.iter().zip(row) {
            let dz = zf as f64 - mean_z;
            var_z += p * dz * dz;
        }
        let z_y = row[y] as f64;
        nll += (max_x + sum_e.ln()) - beta * z_y;
        g += mean_z - z_y;
        h += var_z;
    }
    let inv_n = 1.0 / labels.len() as f64;
    (nll * inv_n, g * inv_n, h * inv_n)
}

/// Minimize a 1D smooth convex objective `f(beta)` over `[1 / max_temp, 1 / min_temp]` using damped
/// Newton-Raphson with Armijo backtracking line search.
fn solve_convex_1d(cfg: &CalibrationConfig, eval: impl Fn(f64) -> (f64, f64, f64)) -> Result<TemperatureFit> {
    let beta_min = 1.0 / cfg.max_temp as f64;
    let beta_max = 1.0 / cfg.min_temp as f64;
    let (initial_nll, _, _) = eval(1.0);
    let mut beta = 1.0f64.clamp(beta_min, beta_max);
    let (mut val, mut g, mut h) = eval(beta);

    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..=cfg.max_iters {
        let at_lo = beta <= beta_min && g >= -cfg.tol;
        let at_hi = beta >= beta_max && g <= cfg.tol;
        if at_lo || at_hi || g.abs() <= cfg.tol {
            converged = true;
            iterations = iter;
            break;
        }
        if iter == cfg.max_iters {
            iterations = iter;
            break;
        }

        // Newton step if curvature is non-degenerate; otherwise move toward the active bound.
        let raw_step = if h > 1e-15 {
            -g / h
        } else if g < 0.0 {
            beta_max - beta
        } else {
            beta_min - beta
        };
        let target = (beta + raw_step).clamp(beta_min, beta_max);
        let step = target - beta;
        if step == 0.0 {
            converged = true;
            iterations = iter;
            break;
        }

        // Armijo backtracking line search on the convex objective.
        let mut alpha = 1.0f64;
        let mut next = None;
        for _ in 0..60 {
            let cand = (beta + alpha * step).clamp(beta_min, beta_max);
            if cand == beta {
                break;
            }
            let (cand_val, cand_g, cand_h) = eval(cand);
            if cand_val.is_finite() && cand_val <= val + 1e-4 * g * (cand - beta) {
                next = Some((cand, cand_val, cand_g, cand_h));
                break;
            }
            alpha *= 0.5;
        }
        match next {
            Some((nb, nv, ng, nh)) => {
                beta = nb;
                val = nv;
                g = ng;
                h = nh;
                iterations = iter + 1;
            }
            None => {
                // Reached f64 objective resolution floor.
                iterations = iter;
                converged = g.abs() <= cfg.tol * 1e3
                    || (beta <= beta_min && g >= 0.0)
                    || (beta >= beta_max && g <= 0.0);
                break;
            }
        }
    }

    let temperature = (1.0 / beta).clamp(cfg.min_temp as f64, cfg.max_temp as f64) as f32;
    let (final_nll, _, _) = eval(1.0 / temperature as f64);
    Ok(TemperatureFit {
        scaler: TemperatureScaler { temperature },
        initial_nll,
        final_nll,
        iterations,
        converged,
    })
}

/// One bin of a reliability diagram over `[lower, upper)`. The final bin is closed `[lower, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Lower edge of the bin in `[0, 1]`.
    pub lower: f32,
    /// Upper edge of the bin in `[0, 1]`.
    pub upper: f32,
    /// Number of samples falling in this bin.
    pub count: usize,
    /// Mean predicted confidence in this bin (`0.0` when empty).
    pub avg_confidence: f32,
    /// Mean empirical accuracy in this bin (`0.0` when empty).
    pub avg_accuracy: f32,
    /// Absolute calibration gap `|avg_confidence - avg_accuracy|` (`0.0` when empty).
    pub gap: f32,
}

/// Summary calibration diagnostics over binary probabilities and ground-truth labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error: `sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|`.
    pub ece: f32,
    /// Maximum Calibration Error: `max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|`.
    pub mce: f32,
    /// Brier score: `(1 / N) * sum_i (p_i - y_i)^2`.
    pub brier: f32,
    /// Binary negative log-likelihood (clamped away from 0/1 by `1e-12` in `f64` for finite scoring).
    pub nll: f32,
    /// Total number of evaluated samples `N`.
    pub count: usize,
    /// Equal-width reliability diagram bins over `[0, 1]`.
    pub bins: Vec<CalibrationBin>,
}

/// Compute reliability diagram bins, ECE, MCE, Brier score, and NLL for binary probabilities `probs`
/// (`p_i in [0, 1]`) and boolean correctness/ground-truth `labels`.
///
/// Uses `num_bins` equal-width bins `[m/M, (m+1)/M)` with the final bin closed `[(M-1)/M, 1.0]`.
pub fn calibration_metrics(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<CalibrationMetrics> {
    if num_bins == 0 {
        return Err(Error::Config("calibration_metrics: num_bins must be >= 1".into()));
    }
    if probs.len() != labels.len() {
        return Err(Error::Config(format!(
            "calibration_metrics: probs length ({}) != labels length ({})",
            probs.len(),
            labels.len()
        )));
    }
    for (i, &p) in probs.iter().enumerate() {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            return Err(Error::Config(format!(
                "calibration_metrics: prob[{i}] must be finite and in [0, 1], got {p}"
            )));
        }
    }

    let mut counts = vec![0usize; num_bins];
    let mut sum_conf = vec![0.0f64; num_bins];
    let mut sum_acc = vec![0.0f64; num_bins];
    let mut brier_sum = 0.0f64;
    let mut nll_sum = 0.0f64;

    const PROB_EPS: f64 = 1e-12;
    let m_f = num_bins as f64;
    for (&pf, &label) in probs.iter().zip(labels) {
        let p = pf as f64;
        let y = if label { 1.0 } else { 0.0 };
        let bin = ((p * m_f).floor() as usize).min(num_bins - 1);
        counts[bin] += 1;
        sum_conf[bin] += p;
        sum_acc[bin] += y;
        let diff = p - y;
        brier_sum += diff * diff;
        if label {
            nll_sum -= p.max(PROB_EPS).ln();
        } else {
            nll_sum -= (1.0 - p).max(PROB_EPS).ln();
        }
    }

    let n = probs.len();
    let mut bins = Vec::with_capacity(num_bins);
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    for b in 0..num_bins {
        let lower = (b as f64 / m_f) as f32;
        let upper = if b + 1 == num_bins { 1.0 } else { ((b + 1) as f64 / m_f) as f32 };
        let cnt = counts[b];
        if cnt == 0 {
            bins.push(CalibrationBin {
                lower,
                upper,
                count: 0,
                avg_confidence: 0.0,
                avg_accuracy: 0.0,
                gap: 0.0,
            });
        } else {
            let avg_conf = sum_conf[b] / cnt as f64;
            let avg_acc = sum_acc[b] / cnt as f64;
            let gap = (avg_conf - avg_acc).abs();
            ece += (cnt as f64 / n as f64) * gap;
            if gap > mce {
                mce = gap;
            }
            bins.push(CalibrationBin {
                lower,
                upper,
                count: cnt,
                avg_confidence: avg_conf as f32,
                avg_accuracy: avg_acc as f32,
                gap: gap as f32,
            });
        }
    }

    let (brier, nll) =
        if n == 0 { (0.0, 0.0) } else { ((brier_sum / n as f64) as f32, (nll_sum / n as f64) as f32) };
    Ok(CalibrationMetrics { ece: ece as f32, mce: mce as f32, brier, nll, count: n, bins })
}

/// Binary Brier score `(1 / N) * sum_i (p_i - y_i)^2` (returns `0.0` on empty input).
pub fn brier_score(probs: &[f32], labels: &[bool]) -> Result<f32> {
    if probs.len() != labels.len() {
        return Err(Error::Config(format!(
            "brier_score: probs length ({}) != labels length ({})",
            probs.len(),
            labels.len()
        )));
    }
    let mut sum = 0.0f64;
    for (i, (&pf, &label)) in probs.iter().zip(labels).enumerate() {
        if !(pf.is_finite() && (0.0..=1.0).contains(&pf)) {
            return Err(Error::Config(format!(
                "brier_score: prob[{i}] must be finite and in [0, 1], got {pf}"
            )));
        }
        let d = pf as f64 - if label { 1.0 } else { 0.0 };
        sum += d * d;
    }
    Ok(if probs.is_empty() { 0.0 } else { (sum / probs.len() as f64) as f32 })
}

/// Multiclass Brier score `(1 / N) * sum_i sum_c (p_{i,c} - 1[c == y_i])^2` over row-major `[N, num_classes]`
/// probabilities and integer class labels `labels[i] < num_classes` (returns `0.0` on empty input).
pub fn multiclass_brier_score(probs: &[f32], num_classes: usize, labels: &[usize]) -> Result<f32> {
    if num_classes < 2 {
        return Err(Error::Config(format!(
            "multiclass_brier_score: num_classes must be >= 2, got {num_classes}"
        )));
    }
    let expected = labels
        .len()
        .checked_mul(num_classes)
        .ok_or_else(|| Error::Config("multiclass_brier_score: size overflow".into()))?;
    if probs.len() != expected {
        return Err(Error::Config(format!(
            "multiclass_brier_score: probs length ({}) != labels.len() * num_classes ({expected})",
            probs.len()
        )));
    }
    let mut sum = 0.0f64;
    for (i, (row, &y)) in probs.chunks_exact(num_classes).zip(labels).enumerate() {
        if y >= num_classes {
            return Err(Error::Config(format!(
                "multiclass_brier_score: label[{i}] = {y} >= num_classes ({num_classes})"
            )));
        }
        for (c, &pf) in row.iter().enumerate() {
            if !(pf.is_finite() && (0.0..=1.0).contains(&pf)) {
                return Err(Error::Config(format!(
                    "multiclass_brier_score: prob[{i}, {c}] must be finite and in [0, 1], got {pf}"
                )));
            }
            let d = pf as f64 - if c == y { 1.0 } else { 0.0 };
            sum += d * d;
        }
    }
    Ok(if labels.is_empty() { 0.0 } else { (sum / labels.len() as f64) as f32 })
}

/// Expected Calibration Error (ECE) with `num_bins` equal-width bins.
pub fn expected_calibration_error(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<f32> {
    Ok(calibration_metrics(probs, labels, num_bins)?.ece)
}

/// Maximum Calibration Error (MCE) with `num_bins` equal-width bins.
pub fn max_calibration_error(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<f32> {
    Ok(calibration_metrics(probs, labels, num_bins)?.mce)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_and_multiclass_derivatives_match_finite_differences() {
        let logits = [-3.2f32, -1.1, 0.0, 0.7, 2.4, 5.1];
        let targets = [0.0f32, 0.25, 0.5, 1.0, 0.8, 1.0];
        let eps = 1e-6f64;
        for &beta in &[0.1f64, 0.5, 1.0, 2.3, 5.0] {
            let (_, g, h) = binary_nll_grad_hess(&logits, &targets, beta);
            assert!(h >= 0.0, "binary Hessian must be non-negative, got {h}");
            let (fp, gp, _) = binary_nll_grad_hess(&logits, &targets, beta + eps);
            let (fm, gm, _) = binary_nll_grad_hess(&logits, &targets, beta - eps);
            let fd_g = (fp - fm) / (2.0 * eps);
            let fd_h = (gp - gm) / (2.0 * eps);
            assert!((g - fd_g).abs() < 1e-8, "binary g at beta={beta}: {g} vs fd {fd_g}");
            assert!((h - fd_h).abs() < 1e-7, "binary h at beta={beta}: {h} vs fd {fd_h}");
        }

        let mc_logits = [
            2.1f32, -0.5, 0.3, -1.2, // sample 0
            -0.8, 1.7, 0.4, 0.9, // sample 1
            0.2, 0.2, 3.0, -2.1, // sample 2
            -1.5, -0.4, 0.1, 1.2, // sample 3
        ];
        let mc_labels = [0usize, 1, 2, 3];
        for &beta in &[0.15f64, 0.8, 1.0, 2.0, 4.0] {
            let (_, g, h) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta);
            assert!(h >= 0.0, "multiclass Hessian must be non-negative, got {h}");
            let (fp, gp, _) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta + eps);
            let (fm, gm, _) = multiclass_nll_grad_hess(&mc_logits, 4, &mc_labels, beta - eps);
            let fd_g = (fp - fm) / (2.0 * eps);
            let fd_h = (gp - gm) / (2.0 * eps);
            assert!((g - fd_g).abs() < 1e-8, "mc g at beta={beta}: {g} vs fd {fd_g}");
            assert!((h - fd_h).abs() < 1e-7, "mc h at beta={beta}: {h} vs fd {fd_h}");
        }
    }

    #[test]
    fn binary_temperature_scaling_recovers_overconfident_and_underconfident_scales() {
        // Deterministically generate true logits z_true and draw labels y ~ Bernoulli(sigmoid(z_true)).
        let mut rng = folo_core::rng::Rng::seed(2026);
        let n = 4000usize;
        let mut true_logits = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let z = rng.uniform(-3.0, 3.0);
            let p = sigmoid_f64(z as f64) as f32;
            let u = rng.uniform(0.0, 1.0);
            true_logits.push(z);
            labels.push(u < p);
        }

        let cfg = CalibrationConfig::default();

        // 1. Overconfident logits: z = 2.5 * z_true -> optimal T ≈ 2.5.
        let over_logits: Vec<f32> = true_logits.iter().map(|&z| 2.5 * z).collect();
        let fit_over = TemperatureScaler::fit_binary(&over_logits, &labels, &cfg).unwrap();
        assert!(fit_over.converged, "overconfident fit should converge: {fit_over:?}");
        assert!(
            (fit_over.scaler.temperature() - 2.5).abs() < 0.25,
            "expected T ≈ 2.5, got {}",
            fit_over.scaler.temperature()
        );
        assert!(fit_over.final_nll < fit_over.initial_nll - 0.05);

        let raw_probs: Vec<f32> = over_logits.iter().map(|&z| sigmoid_f64(z as f64) as f32).collect();
        let cal_probs = fit_over.scaler.calibrate_logits_binary(&over_logits);
        let m_raw = calibration_metrics(&raw_probs, &labels, 15).unwrap();
        let m_cal = calibration_metrics(&cal_probs, &labels, 15).unwrap();
        assert!(
            m_cal.ece < m_raw.ece * 0.4,
            "ECE should drop sharply: raw={} vs cal={}",
            m_raw.ece,
            m_cal.ece
        );
        assert!(
            m_cal.brier < m_raw.brier,
            "Brier should improve: raw={} vs cal={}",
            m_raw.brier,
            m_cal.brier
        );

        // 2. Underconfident logits: z = 0.4 * z_true -> optimal T ≈ 0.4.
        let under_logits: Vec<f32> = true_logits.iter().map(|&z| 0.4 * z).collect();
        let fit_under = TemperatureScaler::fit_binary(&under_logits, &labels, &cfg).unwrap();
        assert!(fit_under.converged);
        assert!(
            (fit_under.scaler.temperature() - 0.4).abs() < 0.08,
            "expected T ≈ 0.4, got {}",
            fit_under.scaler.temperature()
        );
        assert!(fit_under.final_nll < fit_under.initial_nll - 0.05);

        // 3. Exact soft-target identity: y_i = sigmoid(z_i / 2.0) -> exact T == 2.0 to 1e-5!
        let soft_targets: Vec<f32> =
            true_logits.iter().map(|&z| sigmoid_f64(z as f64 / 2.0) as f32).collect();
        let fit_exact = TemperatureScaler::fit_binary_targets(&true_logits, &soft_targets, &cfg).unwrap();
        assert!(fit_exact.converged);
        assert!(
            (fit_exact.scaler.temperature() - 2.0).abs() < 1e-5,
            "expected exact T = 2.0, got {}",
            fit_exact.scaler.temperature()
        );
    }

    #[test]
    fn multiclass_temperature_scaling_recovers_scale_and_improves_brier() {
        let mut rng = folo_core::rng::Rng::seed(4242);
        let n = 3000usize;
        let nc = 4usize;
        let true_temp = 3.0f32;
        let mut logits = Vec::with_capacity(n * nc);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let row = [
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ];
            let max_v = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: [f64; 4] = std::array::from_fn(|c| (row[c] - max_v).exp() as f64);
            let sum: f64 = exps.iter().sum();
            let u = rng.uniform(0.0, 1.0) as f64;
            let mut cdf = 0.0;
            let mut y = nc - 1;
            for (c, &e) in exps.iter().enumerate() {
                cdf += e / sum;
                if u < cdf {
                    y = c;
                    break;
                }
            }
            labels.push(y);
            // Overconfident by factor 3.0.
            for v in row {
                logits.push(v * true_temp);
            }
        }

        let cfg = CalibrationConfig::default();
        let fit = TemperatureScaler::fit_multiclass(&logits, nc, &labels, &cfg).unwrap();
        assert!(fit.converged);
        assert!(
            (fit.scaler.temperature() - true_temp).abs() < 0.25,
            "expected multiclass T ≈ {true_temp}, got {}",
            fit.scaler.temperature()
        );
        assert!(fit.final_nll < fit.initial_nll - 0.1);

        let raw_probs = TemperatureScaler::default().calibrate_logits_multiclass(&logits, nc).unwrap();
        let cal_probs = fit.scaler.calibrate_logits_multiclass(&logits, nc).unwrap();
        // Each row sums to 1.
        for row in cal_probs.chunks_exact(nc) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "softmax row sum = {s}");
        }
        let raw_brier = multiclass_brier_score(&raw_probs, nc, &labels).unwrap();
        let cal_brier = multiclass_brier_score(&cal_probs, nc, &labels).unwrap();
        assert!(cal_brier < raw_brier, "multiclass Brier should decrease: {raw_brier} -> {cal_brier}");
    }

    #[test]
    fn boundary_and_extreme_logit_edge_cases() {
        let cfg = CalibrationConfig::default();

        // 1. Perfectly separable data: mild logits hit min_temp bound; large logits reach near-zero NLL.
        let mild_sep = [-0.5f32, -0.2, 0.2, 0.5];
        let sep_labels = [false, false, true, true];
        let fit_mild_sep = TemperatureScaler::fit_binary(&mild_sep, &sep_labels, &cfg).unwrap();
        assert!(fit_mild_sep.converged);
        assert!((fit_mild_sep.scaler.temperature() - cfg.min_temp).abs() < 1e-5);

        let sep_logits = [-5.0f32, -2.0, 2.0, 5.0];
        let fit_sep = TemperatureScaler::fit_binary(&sep_logits, &sep_labels, &cfg).unwrap();
        assert!(fit_sep.converged);
        assert!(fit_sep.final_nll < 1e-8 && fit_sep.scaler.temperature() < 0.2);

        // 2. Completely inverted labels hit max_temp cleanly.
        let inv_labels = [true, true, false, false];
        let fit_inv = TemperatureScaler::fit_binary(&sep_logits, &inv_labels, &cfg).unwrap();
        assert!(fit_inv.converged);
        assert!((fit_inv.scaler.temperature() - cfg.max_temp).abs() < 1e-4);

        // 3. All-zero logits have flat objective -> stays at T = 1.0.
        let zero_logits = [0.0f32; 6];
        let zero_labels = [true, false, true, false, true, false];
        let fit_zero = TemperatureScaler::fit_binary(&zero_logits, &zero_labels, &cfg).unwrap();
        assert!(fit_zero.converged);
        assert_eq!(fit_zero.scaler.temperature(), 1.0);

        // 4. Huge logits (±1000) stay finite without overflow.
        let huge_logits = [-1000.0f32, -500.0, 500.0, 1000.0, 1.0, -1.0];
        let huge_labels = [false, false, true, true, false, true];
        let fit_huge = TemperatureScaler::fit_binary(&huge_logits, &huge_labels, &cfg).unwrap();
        assert!(fit_huge.final_nll.is_finite());
        assert!(fit_huge.scaler.temperature().is_finite() && fit_huge.scaler.temperature() > 0.0);

        // 5. calibrate_prob matches calibrate_binary on logit(p).
        let scaler = TemperatureScaler::new(2.25).unwrap();
        for &z in &[-4.0f32, -1.5, 0.0, 1.2, 3.8] {
            let p_raw = sigmoid_f64(z as f64) as f32;
            let via_logit = scaler.calibrate_binary(z);
            let via_prob = scaler.calibrate_prob(p_raw);
            assert!((via_logit - via_prob).abs() < 1e-6, "z={z}: {via_logit} vs {via_prob}");
        }
        assert_eq!(scaler.calibrate_prob(0.0), 0.0);
        assert_eq!(scaler.calibrate_prob(1.0), 1.0);
        assert!(scaler.calibrate_prob(f32::NAN).is_nan());

        // 6. Serde round-trip.
        let json = serde_json::to_string(&scaler).unwrap();
        assert_eq!(serde_json::from_str::<TemperatureScaler>(&json).unwrap(), scaler);
    }

    #[test]
    fn hand_computed_ece_mce_brier_and_validation() {
        // 4 samples across 2 bins ([0, 0.5), [0.5, 1.0]):
        // Bin 0: p = [0.0, 0.2], y = [false, false] -> avg_conf = 0.1, avg_acc = 0.0, gap = 0.1
        // Bin 1: p = [0.8, 1.0], y = [true,  false] -> avg_conf = 0.9, avg_acc = 0.5, gap = 0.4
        // ECE = 0.5 * 0.1 + 0.5 * 0.4 = 0.25, MCE = 0.4
        // Brier = (0.0^2 + 0.2^2 + 0.2^2 + 1.0^2) / 4 = 1.08 / 4 = 0.27
        let probs = [0.0f32, 0.2, 0.8, 1.0];
        let labels = [false, false, true, false];
        let m = calibration_metrics(&probs, &labels, 2).unwrap();
        assert_eq!(m.count, 4);
        assert_eq!(m.bins.len(), 2);
        assert_eq!(m.bins[0].count, 2);
        assert_eq!(m.bins[1].count, 2);
        assert!((m.ece - 0.25).abs() < 1e-6, "ece = {}", m.ece);
        assert!((m.mce - 0.40).abs() < 1e-6, "mce = {}", m.mce);
        assert!((m.brier - 0.27).abs() < 1e-6, "brier = {}", m.brier);
        assert!(m.nll.is_finite() && m.nll > 0.0);
        assert_eq!(brier_score(&probs, &labels).unwrap(), m.brier);
        assert_eq!(expected_calibration_error(&probs, &labels, 2).unwrap(), m.ece);
        assert_eq!(max_calibration_error(&probs, &labels, 2).unwrap(), m.mce);

        // Empty input returns clean zeros.
        let empty = calibration_metrics(&[], &[], 10).unwrap();
        assert_eq!((empty.ece, empty.mce, empty.brier, empty.nll, empty.count), (0.0, 0.0, 0.0, 0.0, 0));
        assert_eq!(empty.bins.len(), 10);
        assert_eq!(brier_score(&[], &[]).unwrap(), 0.0);
        assert_eq!(multiclass_brier_score(&[], 3, &[]).unwrap(), 0.0);

        // Validation rejections.
        assert!(TemperatureScaler::new(0.0).is_err());
        assert!(TemperatureScaler::new(-1.0).is_err());
        assert!(TemperatureScaler::new(f32::NAN).is_err());
        assert!(calibration_metrics(&probs, &labels, 0).is_err());
        assert!(calibration_metrics(&[1.1], &[true], 5).is_err());
        assert!(calibration_metrics(&[-0.1], &[false], 5).is_err());
        assert!(calibration_metrics(&[f32::NAN], &[true], 5).is_err());
        assert!(calibration_metrics(&[0.5], &[true, false], 5).is_err());
        let cfg = CalibrationConfig::default();
        assert!(TemperatureScaler::fit_binary(&[], &[], &cfg).is_err());
        assert!(TemperatureScaler::fit_binary(&[f32::NAN], &[true], &cfg).is_err());
        assert!(TemperatureScaler::fit_binary_targets(&[1.0], &[1.2], &cfg).is_err());
        assert!(TemperatureScaler::fit_multiclass(&[0.0, 1.0], 1, &[0], &cfg).is_err());
        assert!(TemperatureScaler::fit_multiclass(&[0.0, 1.0], 2, &[2], &cfg).is_err());
        let bad_cfg = CalibrationConfig { min_temp: 2.0, max_temp: 1.0, ..cfg };
        assert!(TemperatureScaler::fit_binary(&[1.0], &[true], &bad_cfg).is_err());
    }
}
