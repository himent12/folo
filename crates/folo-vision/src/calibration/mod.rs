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
//!   L(beta)   = (1 / N) * sum_i [ log(sum_c exp(beta * d_{i,c})) - beta * d_{i, y_i} ]
//!   L'(beta)  = (1 / N) * sum_i (E_{p_i}[d_i] - d_{i, y_i})
//!   L''(beta) = (1 / N) * sum_i Var_{p_i}(d_i) >= 0,        d_{i,c} = z_{i,c} - max_k z_{i,k}
//!   ```
//!
//! [`TemperatureScaler::fit_binary`], [`TemperatureScaler::fit_binary_targets`], and
//! [`TemperatureScaler::fit_multiclass`] minimize `L(beta)` over `beta in [1 / max_temp, 1 / min_temp]` in
//! `f64` using damped Newton-Raphson with exact 1st/2nd derivatives and Armijo backtracking line search.

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Solver configuration for [`TemperatureScaler`] fitting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    /// Minimum allowed temperature `T_min > 0` (default `0.05`). Bounds `beta <= 1 / min_temp` on separable data.
    pub min_temp: f64,
    /// Maximum allowed temperature `T_max >= min_temp` (default `100.0`). Bounds `beta >= 1 / max_temp`.
    pub max_temp: f64,
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
        if !(1.0 / self.min_temp).is_finite() || !(1.0 / self.max_temp).is_finite() {
            return Err(Error::Config(format!(
                "calibration temperature bounds [{}, {}] have non-finite inverse",
                self.min_temp, self.max_temp
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
    /// Whether the exact KKT optimality condition (`|L'(1/T)| <= tol` or active temperature bound) holds at
    /// the returned `scaler.temperature()`.
    pub converged: bool,
}

#[derive(Deserialize)]
struct TemperatureScalerRaw {
    temperature: f64,
}

/// Post-hoc temperature scaling calibrator: maps raw logits `z` to `z / T` with `T > 0`.
///
/// Deserialization validates the `T.is_finite() && T > 0.0` invariant through [`TemperatureScaler::new`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TemperatureScalerRaw")]
pub struct TemperatureScaler {
    temperature: f64,
}

impl TryFrom<TemperatureScalerRaw> for TemperatureScaler {
    type Error = Error;

    fn try_from(raw: TemperatureScalerRaw) -> Result<Self> {
        Self::new(raw.temperature)
    }
}

impl Default for TemperatureScaler {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl TemperatureScaler {
    /// Construct a scaler with an explicit positive finite temperature `T > 0`.
    pub fn new(temperature: f64) -> Result<Self> {
        if !(temperature.is_finite() && temperature > 0.0 && (1.0 / temperature).is_finite()) {
            return Err(Error::Config(format!("temperature must be finite and > 0, got {temperature}")));
        }
        Ok(Self { temperature })
    }

    /// The scalar temperature `T > 0`.
    #[inline]
    pub fn temperature(&self) -> f64 {
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
        (logit as f64 / self.temperature) as f32
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
        sigmoid_f64(logit as f64 / self.temperature) as f32
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
        sigmoid_f64(logit / self.temperature) as f32
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
        let inv_t = 1.0 / self.temperature;
        let mut out = vec![0.0f32; logits.len()];
        let mut exps = vec![0.0f64; num_classes];
        for (row_in, row_out) in logits.chunks_exact(num_classes).zip(out.chunks_exact_mut(num_classes)) {
            if !row_in.iter().all(|v| v.is_finite()) {
                row_out.fill(f32::NAN);
                continue;
            }
            let max_z = row_in.iter().map(|&z| z as f64).fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for (&z, e_slot) in row_in.iter().zip(exps.iter_mut()) {
                let e = ((z as f64 - max_z) * inv_t).exp();
                *e_slot = e;
                denom += e;
            }
            let inv_denom = 1.0 / denom;
            for (&e, o) in exps.iter().zip(row_out.iter_mut()) {
                *o = (e * inv_denom) as f32;
            }
        }
        Ok(out)
    }
}

#[inline]
pub(crate) fn sigmoid_f64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Exact binary NLL, 1st derivative `L'(beta)`, and 2nd derivative `L''(beta)` in `f64`, structured to avoid
/// catastrophic cancellation on confident samples (`x = beta * z >= 37` with `y = 1`, or `x <= -37` with `y = 0`).
pub(crate) fn binary_nll_grad_hess(logits: &[f32], targets: &[f32], beta: f64) -> (f64, f64, f64) {
    let mut nll = 0.0f64;
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    for (&zf, &yf) in logits.iter().zip(targets) {
        let z = zf as f64;
        let y = yf as f64;
        let x = beta * z;
        let e_neg = (-x.abs()).exp();
        // Cancel linear term before adding ln(1 + exp(-|x|)) so tiny residuals are preserved.
        let linear = if x >= 0.0 { (1.0 - y) * x } else { -y * x };
        nll += linear + e_neg.ln_1p();
        let one_plus = 1.0 + e_neg;
        let p_minus_y =
            if x >= 0.0 { ((1.0 - y) - y * e_neg) / one_plus } else { (e_neg * (1.0 - y) - y) / one_plus };
        let var = e_neg / (one_plus * one_plus);
        g += p_minus_y * z;
        h += var * z * z;
    }
    let inv_n = 1.0 / logits.len() as f64;
    (nll * inv_n, g * inv_n, h * inv_n)
}

/// Exact shift-invariant multiclass softmax NLL, 1st derivative `L'(beta)`, and 2nd derivative `L''(beta)`
/// in `f64`. Each row is centered on `max_z` before scaling so constant shifts (`z + C`) cancel identically
/// without precision loss.
pub(crate) fn multiclass_nll_grad_hess(
    logits: &[f32],
    num_classes: usize,
    labels: &[usize],
    beta: f64,
) -> (f64, f64, f64) {
    let mut nll = 0.0f64;
    let mut g = 0.0f64;
    let mut h = 0.0f64;
    let mut dz = vec![0.0f64; num_classes];
    let mut probs = vec![0.0f64; num_classes];
    for (row, &y) in logits.chunks_exact(num_classes).zip(labels) {
        let mut max_idx = 0usize;
        let mut max_z = row[0] as f64;
        for (c, &zf) in row.iter().enumerate().skip(1) {
            let z = zf as f64;
            if z > max_z {
                max_z = z;
                max_idx = c;
            }
        }
        let mut other_e = 0.0f64;
        for c in 0..num_classes {
            if c == max_idx {
                dz[c] = 0.0;
                probs[c] = 1.0;
            } else {
                let d = row[c] as f64 - max_z;
                dz[c] = d;
                let e = (beta * d).exp();
                probs[c] = e;
                other_e += e;
            }
        }
        let inv_sum = 1.0 / (1.0 + other_e);
        let mut mean_dz = 0.0f64;
        for c in 0..num_classes {
            probs[c] *= inv_sum;
            if c != max_idx {
                mean_dz += probs[c] * dz[c];
            }
        }
        let mut var_z = 0.0f64;
        for c in 0..num_classes {
            let diff = dz[c] - mean_dz;
            var_z += probs[c] * diff * diff;
        }
        let d_y = dz[y];
        nll += (-beta * d_y) + other_e.ln_1p();
        g += mean_dz - d_y;
        h += var_z;
    }
    let inv_n = 1.0 / labels.len() as f64;
    (nll * inv_n, g * inv_n, h * inv_n)
}

#[inline]
fn kkt_satisfied(beta: f64, g: f64, beta_min: f64, beta_max: f64, tol: f64) -> bool {
    if !g.is_finite() {
        return false;
    }
    let at_lo = beta <= beta_min && g >= -tol;
    let at_hi = beta >= beta_max && g <= tol;
    at_lo || at_hi || g.abs() <= tol
}

/// Minimize a 1D smooth convex objective `f(beta)` over `[1 / max_temp, 1 / min_temp]` using damped
/// Newton-Raphson with Armijo backtracking line search.
fn solve_convex_1d(cfg: &CalibrationConfig, eval: impl Fn(f64) -> (f64, f64, f64)) -> Result<TemperatureFit> {
    let beta_min = 1.0 / cfg.max_temp;
    let beta_max = 1.0 / cfg.min_temp;
    let (initial_nll, _, _) = eval(1.0);
    let mut beta = 1.0f64.clamp(beta_min, beta_max);
    let (mut val, mut g, mut h) = eval(beta);

    let mut iterations = 0usize;

    for iter in 0..cfg.max_iters {
        if kkt_satisfied(beta, g, beta_min, beta_max, cfg.tol) {
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
            iterations = iter;
            break;
        }

        // Armijo backtracking line search on the convex objective; terminates when cand == beta.
        let mut alpha = 1.0f64;
        let mut next = None;
        loop {
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
                iterations = iter;
                break;
            }
        }
    }

    let temperature = (1.0 / beta).clamp(cfg.min_temp, cfg.max_temp);
    let beta_ret = 1.0 / temperature;
    let (final_nll, final_g, _) = eval(beta_ret);
    let converged = final_g.is_finite()
        && (final_g.abs() <= cfg.tol
            || (temperature >= cfg.max_temp && final_g >= -cfg.tol)
            || (temperature <= cfg.min_temp && final_g <= cfg.tol));
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
/// Uses `num_bins` equal-width bins `[lower, upper)` with the final bin closed `[lower, 1.0]`; bin
/// membership is evaluated against the exact published `f32` bin edges.
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

    let m_f = num_bins as f64;
    let edges: Vec<f32> =
        (0..=num_bins).map(|b| if b == num_bins { 1.0 } else { (b as f64 / m_f) as f32 }).collect();

    let mut counts = vec![0usize; num_bins];
    let mut sum_conf = vec![0.0f64; num_bins];
    let mut sum_acc = vec![0.0f64; num_bins];
    let mut brier_sum = 0.0f64;
    let mut nll_sum = 0.0f64;

    const PROB_EPS: f64 = 1e-12;
    for (&pf, &label) in probs.iter().zip(labels) {
        let p = pf as f64;
        let y = if label { 1.0 } else { 0.0 };
        let mut bin = ((p * m_f).floor() as usize).min(num_bins - 1);
        while bin + 1 < num_bins && pf >= edges[bin + 1] {
            bin += 1;
        }
        while bin > 0 && pf < edges[bin] {
            bin -= 1;
        }
        counts[bin] += 1;
        sum_conf[bin] += p;
        sum_acc[bin] += y;
        let diff = p - y;
        brier_sum += diff * diff;
        if label {
            nll_sum -= p.max(PROB_EPS).ln();
        } else if p <= 0.5 {
            nll_sum -= (-p).ln_1p();
        } else {
            nll_sum -= (1.0 - p).max(PROB_EPS).ln();
        }
    }

    let n = probs.len();
    let mut bins = Vec::with_capacity(num_bins);
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    for b in 0..num_bins {
        let lower = edges[b];
        let upper = edges[b + 1];
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
