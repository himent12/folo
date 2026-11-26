//! Regularized Class-wise Temperature Scaling: per-class temperatures `T_c > 0` fitted via strictly convex
//! MAP optimization in inverse temperature `beta_c = 1 / T_c` with smooth shrinkage toward `T = 1` for
//! sparse or absent classes.
//!
//! For class `c` with `N_c > 0` binary calibration samples `(z_i, y_i)`, we minimize:
//! ```text
//! F_c(beta) = (1 / N_c) * sum_{i=1}^{N_c} BCE(beta * z_i, y_i) + (lambda / (2 * N_c)) * (beta - 1)^2
//! ```
//! over `beta in [1 / max_temp, 1 / min_temp]` where `lambda = prior_strength >= 0`.
//!
//! - When `lambda > 0`, `F_c''(beta) >= lambda / N_c > 0`, making each class's problem **strictly convex**
//!   with a unique global minimizer even on linearly separable rare classes.
//! - When `N_c = 0` (absent class), `T_c` falls back directly to `1.0` (clamped to `[min_temp, max_temp]`).
//! - When `N_c` is small (sparse class), `lambda / N_c` smoothly pulls `T_c` toward `1.0`, preventing
//!   overfitting.
//! - As `N_c -> infinity`, `lambda / N_c -> 0` and `T_c` converges to the exact per-class MLE temperature.

use serde::{Deserialize, Serialize};

use super::{CalibrationConfig, TemperatureScaler, binary_nll_grad_hess, solve_convex_1d};
use crate::{Error, Result};

/// Configuration for [`ClasswiseTemperatureScaler`] fitting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClasswiseCalibrationConfig {
    /// Base 1D convex Newton-Raphson solver bounds and stopping criteria.
    pub solver: CalibrationConfig,
    /// Quadratic MAP prior strength `lambda >= 0` pulling each class's inverse temperature `1 / T_c` toward
    /// `1.0` with effective weight `lambda / N_c` (default `1.0`; `0.0` = unregularized per-class MLE).
    pub prior_strength: f64,
}

impl Default for ClasswiseCalibrationConfig {
    fn default() -> Self {
        Self { solver: CalibrationConfig::default(), prior_strength: 1.0 }
    }
}

impl ClasswiseCalibrationConfig {
    fn validate(&self) -> Result<()> {
        self.solver.validate()?;
        if !(self.prior_strength.is_finite() && self.prior_strength >= 0.0) {
            return Err(Error::Config(format!(
                "calibration.prior_strength must be finite and >= 0, got {}",
                self.prior_strength
            )));
        }
        Ok(())
    }
}

/// Fit summary for a single class within [`ClasswiseTemperatureFit`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassTemperatureFit {
    /// Fitted temperature `T_c > 0` for this class.
    pub temperature: f64,
    /// Number of calibration samples observed for this class (`0` when absent -> fallback `T_c = 1.0`).
    pub sample_count: usize,
    /// Mean unregularized binary NLL for this class at `T_c = 1.0` (`0.0` when `sample_count == 0`).
    pub initial_nll: f64,
    /// Mean unregularized binary NLL for this class at `temperature` (`0.0` when `sample_count == 0`).
    pub final_nll: f64,
    /// Number of damped Newton-Raphson iterations executed for this class.
    pub iterations: usize,
    /// Whether the exact KKT condition holds for this class (always `true` for absent classes).
    pub converged: bool,
}

/// Result of fitting a [`ClasswiseTemperatureScaler`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClasswiseTemperatureFit {
    /// Fitted per-class temperature scaler.
    pub scaler: ClasswiseTemperatureScaler,
    /// Per-class fit diagnostics indexed by `class_id in 0..num_classes`.
    pub per_class: Vec<ClassTemperatureFit>,
    /// Overall sample-weighted mean unregularized binary NLL at `T_c = 1.0`.
    pub initial_nll: f64,
    /// Overall sample-weighted mean unregularized binary NLL at the fitted per-class temperatures.
    pub final_nll: f64,
    /// Whether every class converged.
    pub converged: bool,
}

#[derive(Deserialize)]
struct ClasswiseTemperatureScalerRaw {
    temperatures: Vec<f64>,
}

/// Post-hoc class-wise temperature scaling calibrator: maps class `c`'s logit `z` to `z / T_c` with `T_c > 0`.
///
/// Deserialization validates non-emptiness and `T_c.is_finite() && T_c > 0.0` through
/// [`ClasswiseTemperatureScaler::new`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ClasswiseTemperatureScalerRaw")]
pub struct ClasswiseTemperatureScaler {
    temperatures: Vec<f64>,
}

impl TryFrom<ClasswiseTemperatureScalerRaw> for ClasswiseTemperatureScaler {
    type Error = Error;

    fn try_from(raw: ClasswiseTemperatureScalerRaw) -> Result<Self> {
        Self::new(raw.temperatures)
    }
}

impl ClasswiseTemperatureScaler {
    /// Construct a class-wise scaler from explicit positive finite per-class temperatures.
    pub fn new(temperatures: Vec<f64>) -> Result<Self> {
        if temperatures.is_empty() {
            return Err(Error::Config("classwise temperatures must not be empty".into()));
        }
        for (c, &t) in temperatures.iter().enumerate() {
            if !(t.is_finite() && t > 0.0 && (1.0 / t).is_finite()) {
                return Err(Error::Config(format!(
                    "classwise temperature[{c}] must be finite and > 0, got {t}"
                )));
            }
        }
        Ok(Self { temperatures })
    }

    /// Construct an identity class-wise scaler (`T_c = 1.0` for all `num_classes >= 1`).
    pub fn identity(num_classes: usize) -> Result<Self> {
        if num_classes == 0 {
            return Err(Error::Config("classwise identity scaler requires num_classes >= 1".into()));
        }
        Ok(Self { temperatures: vec![1.0; num_classes] })
    }

    /// Number of classes.
    #[inline]
    pub fn num_classes(&self) -> usize {
        self.temperatures.len()
    }

    /// All per-class temperatures `T_c > 0`.
    #[inline]
    pub fn temperatures(&self) -> &[f64] {
        &self.temperatures
    }

    /// Temperature `T_c > 0` for `class_id`.
    pub fn temperature(&self, class_id: usize) -> Result<f64> {
        self.temperatures.get(class_id).copied().ok_or_else(|| {
            Error::Config(format!(
                "classwise temperature: class_id {class_id} >= num_classes ({})",
                self.temperatures.len()
            ))
        })
    }

    /// Scalar [`TemperatureScaler`] for `class_id`.
    pub fn scaler_for(&self, class_id: usize) -> Result<TemperatureScaler> {
        TemperatureScaler::new(self.temperature(class_id)?)
    }

    /// Fit regularized per-class temperatures on binary detection logits, boolean labels, and class ids.
    pub fn fit_binary_by_class(
        logits: &[f32],
        labels: &[bool],
        class_ids: &[usize],
        num_classes: usize,
        cfg: &ClasswiseCalibrationConfig,
    ) -> Result<ClasswiseTemperatureFit> {
        if logits.len() != labels.len() {
            return Err(Error::Config(format!(
                "fit_binary_by_class: logits length ({}) != labels length ({})",
                logits.len(),
                labels.len()
            )));
        }
        let targets: Vec<f32> = labels.iter().map(|&y| if y { 1.0 } else { 0.0 }).collect();
        Self::fit_binary_targets_by_class(logits, &targets, class_ids, num_classes, cfg)
    }

    /// Fit regularized per-class temperatures on binary logits, continuous targets `y_i in [0, 1]`, and
    /// class ids `class_ids[i] < num_classes`.
    ///
    /// Classes with zero samples smoothly fall back to `T_c = 1.0` (clamped to `[min_temp, max_temp]`);
    /// sparse classes are regularized toward `T_c = 1.0` via `prior_strength / N_c`.
    pub fn fit_binary_targets_by_class(
        logits: &[f32],
        targets: &[f32],
        class_ids: &[usize],
        num_classes: usize,
        cfg: &ClasswiseCalibrationConfig,
    ) -> Result<ClasswiseTemperatureFit> {
        cfg.validate()?;
        if num_classes == 0 {
            return Err(Error::Config("fit_binary_targets_by_class: num_classes must be >= 1".into()));
        }
        if logits.is_empty() {
            return Err(Error::Config("fit_binary_targets_by_class: logits must not be empty".into()));
        }
        if logits.len() != targets.len() || logits.len() != class_ids.len() {
            return Err(Error::Config(format!(
                "fit_binary_targets_by_class: length mismatch (logits {}, targets {}, class_ids {})",
                logits.len(),
                targets.len(),
                class_ids.len()
            )));
        }
        let mut cls_logits = vec![Vec::<f32>::new(); num_classes];
        let mut cls_targets = vec![Vec::<f32>::new(); num_classes];
        for (i, ((&z, &y), &c)) in logits.iter().zip(targets).zip(class_ids).enumerate() {
            if !z.is_finite() {
                return Err(Error::Config(format!(
                    "fit_binary_targets_by_class: non-finite logit at index {i}: {z}"
                )));
            }
            if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
                return Err(Error::Config(format!(
                    "fit_binary_targets_by_class: target at index {i} must be in [0, 1], got {y}"
                )));
            }
            if c >= num_classes {
                return Err(Error::Config(format!(
                    "fit_binary_targets_by_class: class_id[{i}] = {c} >= num_classes ({num_classes})"
                )));
            }
            cls_logits[c].push(z);
            cls_targets[c].push(y);
        }
        fit_buckets(&cls_logits, &cls_targets, cfg)
    }

    /// Fit regularized per-class temperatures on dense row-major multi-label logits `[N, num_classes]` and
    /// targets `[N, num_classes]` in `[0, 1]`.
    pub fn fit_multilabel(
        logits: &[f32],
        targets: &[f32],
        num_classes: usize,
        cfg: &ClasswiseCalibrationConfig,
    ) -> Result<ClasswiseTemperatureFit> {
        cfg.validate()?;
        if num_classes == 0 {
            return Err(Error::Config("fit_multilabel: num_classes must be >= 1".into()));
        }
        if logits.is_empty() {
            return Err(Error::Config("fit_multilabel: logits must not be empty".into()));
        }
        if logits.len() != targets.len() {
            return Err(Error::Config(format!(
                "fit_multilabel: logits length ({}) != targets length ({})",
                logits.len(),
                targets.len()
            )));
        }
        if !logits.len().is_multiple_of(num_classes) {
            return Err(Error::Config(format!(
                "fit_multilabel: logits length ({}) is not a multiple of num_classes ({num_classes})",
                logits.len()
            )));
        }
        let n = logits.len() / num_classes;
        let mut cls_logits = vec![Vec::<f32>::with_capacity(n); num_classes];
        let mut cls_targets = vec![Vec::<f32>::with_capacity(n); num_classes];
        for (i, (&z, &y)) in logits.iter().zip(targets).enumerate() {
            if !z.is_finite() {
                return Err(Error::Config(format!("fit_multilabel: non-finite logit at index {i}: {z}")));
            }
            if !(y.is_finite() && (0.0..=1.0).contains(&y)) {
                return Err(Error::Config(format!(
                    "fit_multilabel: target at index {i} must be in [0, 1], got {y}"
                )));
            }
            let c = i % num_classes;
            cls_logits[c].push(z);
            cls_targets[c].push(y);
        }
        fit_buckets(&cls_logits, &cls_targets, cfg)
    }

    /// Scale a single raw logit for `class_id` by `1 / T_c`.
    pub fn scale_logit(&self, class_id: usize, logit: f32) -> Result<f32> {
        Ok(self.scaler_for(class_id)?.scale_logit(logit))
    }

    /// Calibrate a single binary logit for `class_id` to `sigmoid(logit / T_c)`.
    pub fn calibrate_binary(&self, class_id: usize, logit: f32) -> Result<f32> {
        Ok(self.scaler_for(class_id)?.calibrate_binary(logit))
    }

    /// Calibrate a single probability `p in [0, 1]` for `class_id` via `sigmoid(logit(p) / T_c)`.
    pub fn calibrate_prob(&self, class_id: usize, prob: f32) -> Result<f32> {
        Ok(self.scaler_for(class_id)?.calibrate_prob(prob))
    }

    /// Calibrate binary detection logits grouped by `class_ids` to probabilities.
    pub fn calibrate_by_class(&self, logits: &[f32], class_ids: &[usize]) -> Result<Vec<f32>> {
        if logits.len() != class_ids.len() {
            return Err(Error::Config(format!(
                "calibrate_by_class: logits length ({}) != class_ids length ({})",
                logits.len(),
                class_ids.len()
            )));
        }
        let nc = self.temperatures.len();
        let mut out = Vec::with_capacity(logits.len());
        for (i, (&z, &c)) in logits.iter().zip(class_ids).enumerate() {
            if c >= nc {
                return Err(Error::Config(format!(
                    "calibrate_by_class: class_id[{i}] = {c} >= num_classes ({nc})"
                )));
            }
            out.push(TemperatureScaler::new(self.temperatures[c])?.calibrate_binary(z));
        }
        Ok(out)
    }

    /// Calibrate dense row-major multi-label logits `[N, num_classes]` using column-wise sigmoid `sigmoid(z_{i,c} / T_c)`.
    pub fn calibrate_multilabel(&self, logits: &[f32]) -> Result<Vec<f32>> {
        let nc = self.temperatures.len();
        if !logits.len().is_multiple_of(nc) {
            return Err(Error::Config(format!(
                "calibrate_multilabel: logits length ({}) must be a multiple of num_classes ({nc})",
                logits.len()
            )));
        }
        let scalers: Vec<TemperatureScaler> =
            self.temperatures.iter().map(|&t| TemperatureScaler::new(t)).collect::<Result<Vec<_>>>()?;
        let mut out = Vec::with_capacity(logits.len());
        for (i, &z) in logits.iter().enumerate() {
            out.push(scalers[i % nc].calibrate_binary(z));
        }
        Ok(out)
    }
}

/// Fit one regularized 1D convex temperature per class bucket and assemble [`ClasswiseTemperatureFit`].
fn fit_buckets(
    cls_logits: &[Vec<f32>],
    cls_targets: &[Vec<f32>],
    cfg: &ClasswiseCalibrationConfig,
) -> Result<ClasswiseTemperatureFit> {
    let num_classes = cls_logits.len();
    let fallback_temp = 1.0f64.clamp(cfg.solver.min_temp, cfg.solver.max_temp);

    let mut temperatures = Vec::with_capacity(num_classes);
    let mut per_class = Vec::with_capacity(num_classes);
    let mut total_count = 0usize;
    let mut total_init = 0.0f64;
    let mut total_final = 0.0f64;
    let mut all_converged = true;

    for c in 0..num_classes {
        let zl = &cls_logits[c];
        let yt = &cls_targets[c];
        let n_c = zl.len();
        if n_c == 0 {
            temperatures.push(fallback_temp);
            per_class.push(ClassTemperatureFit {
                temperature: fallback_temp,
                sample_count: 0,
                initial_nll: 0.0,
                final_nll: 0.0,
                iterations: 0,
                converged: true,
            });
            continue;
        }
        let reg = cfg.prior_strength / n_c as f64;
        let fit = solve_convex_1d(&cfg.solver, |beta| {
            let (nll, g, h) = binary_nll_grad_hess(zl, yt, beta);
            let diff = beta - 1.0;
            (nll + 0.5 * reg * diff * diff, g + reg * diff, h + reg)
        })?;
        let t_c = fit.scaler.temperature();
        let initial_nll = binary_nll_grad_hess(zl, yt, 1.0).0;
        let final_nll = binary_nll_grad_hess(zl, yt, 1.0 / t_c).0;
        temperatures.push(t_c);
        per_class.push(ClassTemperatureFit {
            temperature: t_c,
            sample_count: n_c,
            initial_nll,
            final_nll,
            iterations: fit.iterations,
            converged: fit.converged,
        });
        total_count += n_c;
        total_init += n_c as f64 * initial_nll;
        total_final += n_c as f64 * final_nll;
        all_converged &= fit.converged;
    }

    let inv_total = 1.0 / total_count.max(1) as f64;
    Ok(ClasswiseTemperatureFit {
        scaler: ClasswiseTemperatureScaler::new(temperatures)?,
        per_class,
        initial_nll: total_init * inv_total,
        final_nll: total_final * inv_total,
        converged: all_converged,
    })
}
