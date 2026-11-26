//! Power-Law EMA decay schedules:
//!
//! 1. [`PowerLawEmaSchedule`]: inverse-power decay ramp
//!    ```text
//!    decay(step) = min(1 - (1 + step / t0)^(-gamma), max_decay)
//!    ```
//!    evaluated cancellation-free via `-(-gamma * (step / t0).ln_1p()).exp_m1()`.
//! 2. [`Edm2PowerEmaSchedule`]: Karras et al. (CVPR 2024, EDM2 Appendix C / `phema.py`) exact
//!    power-function EMA recurrence
//!    ```text
//!    decay(step) = (1 - 1 / (step + t0))^(gamma + 1)    for step + t0 > 1, else 0
//!    ```
//!    whose discrete impulse-response profile on normalized training time `t / T in [0, 1]` has asymptotic
//!    relative standard deviation `sigma_rel = Std[t / T] = sqrt(gamma + 1) / ((gamma + 2) * sqrt(gamma + 3))`
//!    (Eq. 124), inverted overflow-safely for any `sigma_rel in (0, 1 / sqrt(12))`.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// Default inverse-power EMA ramp exponent (`gamma = 1.0`, harmonic incoming weight `1 / (1 + step / t0)`).
pub const POWER_LAW_EMA_DEFAULT_GAMMA: f64 = 1.0;
/// Default inverse-power EMA ramp scale `t0 = 10.0`.
pub const POWER_LAW_EMA_DEFAULT_T0: f64 = 10.0;
/// Default EDM2 power-function EMA exponent (`gamma = 6.94`, relative width `sigma_rel ≈ 0.10`, Karras et al. 2024).
pub const EDM2_EMA_DEFAULT_GAMMA: f64 = 6.94;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PowerLawEmaScheduleRaw {
    t0: f64,
    gamma: f64,
    #[serde(default = "default_max_decay")]
    max_decay: f64,
}

fn default_max_decay() -> f64 {
    1.0
}

/// Inverse-power EMA decay schedule: `decay(step) = min(1 - (1 + step / t0)^(-gamma), max_decay)`.
///
/// At `step == 0`, returns `0.0`; as `step -> infinity`, approaches `max_decay` (default `1.0`)
/// monotonically with incoming weight `(1 + step / t0)^(-gamma)`. For Karras et al. (2024) EDM2
/// relative-width parameterization (`sigma_rel`), see [`Edm2PowerEmaSchedule`].
///
/// Deserialization validates all invariants (`t0 > 0`, `gamma > 0`, `max_decay in (0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PowerLawEmaScheduleRaw")]
pub struct PowerLawEmaSchedule {
    t0: f64,
    gamma: f64,
    max_decay: f64,
}

impl TryFrom<PowerLawEmaScheduleRaw> for PowerLawEmaSchedule {
    type Error = Error;

    fn try_from(raw: PowerLawEmaScheduleRaw) -> Result<Self> {
        Self::with_max_decay(raw.t0, raw.gamma, raw.max_decay)
    }
}

impl Default for PowerLawEmaSchedule {
    fn default() -> Self {
        Self { t0: POWER_LAW_EMA_DEFAULT_T0, gamma: POWER_LAW_EMA_DEFAULT_GAMMA, max_decay: 1.0 }
    }
}

impl PowerLawEmaSchedule {
    /// Construct an uncapped inverse-power EMA schedule (`max_decay = 1.0`) with scale `t0 > 0` and
    /// exponent `gamma > 0`.
    pub fn new(t0: f64, gamma: f64) -> Result<Self> {
        Self::with_max_decay(t0, gamma, 1.0)
    }

    /// Construct an inverse-power EMA schedule capped at `max_decay in (0, 1]`.
    pub fn with_max_decay(t0: f64, gamma: f64, max_decay: f64) -> Result<Self> {
        if !(t0.is_finite() && t0 > 0.0) {
            return Err(Error::Invalid(format!("PowerLawEmaSchedule: t0 must be finite and > 0, got {t0}")));
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(Error::Invalid(format!(
                "PowerLawEmaSchedule: gamma must be finite and > 0, got {gamma}"
            )));
        }
        if !(max_decay.is_finite() && max_decay > 0.0 && max_decay <= 1.0) {
            return Err(Error::Invalid(format!(
                "PowerLawEmaSchedule: max_decay must be finite and in (0, 1], got {max_decay}"
            )));
        }
        Ok(Self { t0, gamma, max_decay })
    }

    /// Time-scale / warmup constant `t0 > 0`.
    #[inline]
    pub fn t0(&self) -> f64 {
        self.t0
    }

    /// Power-law exponent `gamma > 0`.
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Maximum decay cap in `(0, 1]`.
    #[inline]
    pub fn max_decay(&self) -> f64 {
        self.max_decay
    }

    /// Compute the EMA decay `min(1 - (1 + step / t0)^(-gamma), max_decay)` at `step`.
    #[inline]
    pub fn decay_at(&self, step: u64) -> f64 {
        if step == 0 {
            return 0.0;
        }
        let u = step as f64 / self.t0;
        let log_term = u.ln_1p();
        let d = -(-self.gamma * log_term).exp_m1();
        if d.is_finite() { d.clamp(0.0, self.max_decay) } else { self.max_decay }
    }
}

/// Evaluate inverse-power EMA decay `1 - (1 + step / t0)^(-gamma)` directly.
pub fn power_law_ema_decay(step: u64, t0: f64, gamma: f64) -> Result<f64> {
    Ok(PowerLawEmaSchedule::new(t0, gamma)?.decay_at(step))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Edm2PowerEmaScheduleRaw {
    gamma: f64,
    #[serde(default)]
    t0: f64,
}

/// Karras et al. (CVPR 2024, EDM2 Appendix C / `phema.py`) power-function EMA decay schedule:
///
/// ```text
/// decay(step) = (1 - 1 / (step + t0))^(gamma + 1)    for step + t0 > 1, else 0
/// ```
///
/// Its discrete impulse-response profile over `t in 1..=T` (with `t0 = 0`) has asymptotic relative standard
/// deviation `sigma_rel = Std[t / T] = sqrt(gamma + 1) / ((gamma + 2) * sqrt(gamma + 3))` (Eq. 124).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Edm2PowerEmaScheduleRaw")]
pub struct Edm2PowerEmaSchedule {
    gamma: f64,
    t0: f64,
}

impl TryFrom<Edm2PowerEmaScheduleRaw> for Edm2PowerEmaSchedule {
    type Error = Error;

    fn try_from(raw: Edm2PowerEmaScheduleRaw) -> Result<Self> {
        Self::new(raw.gamma, raw.t0)
    }
}

impl Default for Edm2PowerEmaSchedule {
    fn default() -> Self {
        Self { gamma: EDM2_EMA_DEFAULT_GAMMA, t0: 0.0 }
    }
}

impl Edm2PowerEmaSchedule {
    /// Construct an EDM2 power-function EMA schedule with exponent `gamma > 0` and prior shift `t0 >= 0`.
    pub fn new(gamma: f64, t0: f64) -> Result<Self> {
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(Error::Invalid(format!(
                "Edm2PowerEmaSchedule: gamma must be finite and > 0, got {gamma}"
            )));
        }
        if !(t0.is_finite() && t0 >= 0.0) {
            return Err(Error::Invalid(format!(
                "Edm2PowerEmaSchedule: t0 must be finite and >= 0, got {t0}"
            )));
        }
        Ok(Self { gamma, t0 })
    }

    /// Construct from Karras et al. (2024, Eq. 124) relative profile width
    /// `sigma_rel = Std[t / T] in (0, 1 / sqrt(12))`, solving
    /// `(gamma + 2) * sqrt((gamma + 3) / (gamma + 1)) = 1 / sigma_rel` overflow-safely without squaring
    /// `1 / sigma_rel`.
    pub fn from_relative_width(sigma_rel: f64, t0: f64) -> Result<Self> {
        let max_sigma = 1.0 / 12.0_f64.sqrt();
        if !(sigma_rel.is_finite() && sigma_rel > 0.0 && sigma_rel < max_sigma) {
            return Err(Error::Invalid(format!(
                "Edm2PowerEmaSchedule: sigma_rel must be finite and in (0, 1/sqrt(12)), got {sigma_rel}"
            )));
        }
        let inv_s = 1.0 / sigma_rel;
        // `F(g) = (g + 2) * sqrt(1 + 2 / (g + 1))` is strictly increasing and strictly convex on `[0, inf)`
        // with `F(0) = sqrt(12) < inv_s` and `F(inv_s - 2) > inv_s`, so Newton's method from the upper
        // bound `inv_s - 2` converges monotonically and quadratically without ever overflowing `inv_s^2`.
        let mut gamma = (inv_s - 2.0).max(0.0);
        for _ in 0..16 {
            let u = 1.0 / (gamma + 1.0);
            let root = (1.0 + 2.0 * u).sqrt();
            let f = (gamma + 2.0) * root - inv_s;
            let fp = (1.0 + u - u * u) / root;
            let step = f / fp;
            let next = (gamma - step).max(0.0);
            let done = (next - gamma).abs() <= 1e-15 * next.abs().max(1.0);
            gamma = next;
            if done {
                break;
            }
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(Error::Invalid(format!(
                "Edm2PowerEmaSchedule: sigma_rel {sigma_rel} yields non-positive/non-finite gamma {gamma}"
            )));
        }
        Self::new(gamma, t0)
    }

    /// Power-function exponent `gamma > 0`.
    #[inline]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Prior shift `t0 >= 0`.
    #[inline]
    pub fn t0(&self) -> f64 {
        self.t0
    }

    /// Asymptotic relative profile width `sigma_rel = Std[t / T] = sqrt(gamma + 1) / ((gamma + 2) * sqrt(gamma + 3))`
    /// (Karras et al. 2024, Eq. 124; overflow-safe).
    #[inline]
    pub fn relative_width(&self) -> f64 {
        let u = 1.0 / (self.gamma + 1.0);
        1.0 / ((self.gamma + 2.0) * (1.0 + 2.0 * u).sqrt())
    }

    /// Compute the EDM2 decay `(1 - 1 / (step + t0))^(gamma + 1)` for `step >= 1` (`0.0` if `step + t0 <= 1`).
    #[inline]
    pub fn decay_at(&self, step: u64) -> f64 {
        if step == 0 {
            return 0.0;
        }
        let denom = step as f64 + self.t0;
        if denom <= 1.0 {
            return 0.0;
        }
        let d = ((self.gamma + 1.0) * (-1.0 / denom).ln_1p()).exp();
        if d.is_finite() { d.clamp(0.0, 1.0) } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Conv2d, Conv2dConfig, ModelEma};
    use folo_core::rng::Rng;

    #[test]
    fn power_law_ema_exact_values_monotonicity_and_precision() {
        let sched = PowerLawEmaSchedule::new(10.0, 2.0).unwrap();
        assert_eq!(sched.decay_at(0), 0.0);
        // step = 10 -> 1 - (1 + 1)^(-2) = 1 - 1/4 = 0.75
        assert!((sched.decay_at(10) - 0.75).abs() < 1e-15);
        // step = 30 -> 1 - (1 + 3)^(-2) = 1 - 1/16 = 0.9375
        assert!((sched.decay_at(30) - 0.9375).abs() < 1e-15);
        assert_eq!(power_law_ema_decay(10, 10.0, 2.0).unwrap(), sched.decay_at(10));

        // Monotonically increasing across wide step range.
        let mut prev = 0.0;
        for step in [1u64, 2, 5, 10, 50, 100, 1_000, 10_000, 1_000_000, u64::MAX] {
            let d = sched.decay_at(step);
            assert!(d.is_finite() && d >= prev && d <= 1.0, "step={step}: d={d} prev={prev}");
            prev = d;
        }
        assert_eq!(sched.decay_at(u64::MAX), 1.0);

        // Cancellation-free precision for tiny u = step / t0 = 1e-12:
        // 1 - (1 + u)^(-gamma) = gamma * u - gamma*(gamma + 1)/2 * u^2 + O(u^3).
        let tiny = PowerLawEmaSchedule::new(1e12, 3.0).unwrap();
        let d_tiny = tiny.decay_at(1);
        let want_tiny = 3e-12 - 6e-24;
        assert!((d_tiny - want_tiny).abs() < 1e-24, "tiny u: got={d_tiny:e} want={want_tiny:e}");

        // Extreme t0 = 1e-300 at step = u64::MAX saturates cleanly to max_decay without NaN.
        let extreme = PowerLawEmaSchedule::with_max_decay(1e-300, 5.0, 0.9999).unwrap();
        assert_eq!(extreme.decay_at(u64::MAX), 0.9999);

        // Integration with ModelEma::update_with_power_law.
        let mut rng = Rng::seed(5);
        let src = Conv2d::new(Conv2dConfig::new(2, 3, 1), &mut rng);
        let shadow = Conv2d::new(Conv2dConfig::new(2, 3, 1), &mut rng);
        let mut ema = ModelEma::new(shadow, &src).unwrap();
        let w_init = src.weight.to_vec::<f32>().unwrap();
        {
            let _g = folo_core::no_grad();
            let shifted = src.weight.add_scalar(2.0).unwrap();
            src.weight.copy_from(&shifted).unwrap();
        }
        let w_new = src.weight.to_vec::<f32>().unwrap();
        let pl = PowerLawEmaSchedule::new(1.0, 1.0).unwrap(); // step 1: 1 - (1 + 1)^(-1) = 0.5
        ema.update_with_power_law(&src, &pl).unwrap();
        assert_eq!(ema.updates(), 1);
        let w_ema = ema.module().weight.to_vec::<f32>().unwrap();
        for i in 0..w_init.len() {
            let want = 0.5 * w_init[i] + 0.5 * w_new[i];
            assert!((w_ema[i] - want).abs() < 1e-6, "ema[{i}]: {} vs {want}", w_ema[i]);
        }

        // Serde round-trip & invalid rejections.
        let json = serde_json::to_string(&sched).unwrap();
        assert_eq!(serde_json::from_str::<PowerLawEmaSchedule>(&json).unwrap(), sched);
        assert!(serde_json::from_str::<PowerLawEmaSchedule>(r#"{"t0": 0.0, "gamma": 2.0}"#).is_err());
        assert!(serde_json::from_str::<PowerLawEmaSchedule>(r#"{"t0": 10.0, "gamma": -1.0}"#).is_err());
        assert!(
            serde_json::from_str::<PowerLawEmaSchedule>(r#"{"t0": 10.0, "gamma": 2.0, "max_decay": 1.1}"#)
                .is_err()
        );
        assert!(PowerLawEmaSchedule::new(f64::NAN, 2.0).is_err());
        assert!(PowerLawEmaSchedule::new(10.0, f64::INFINITY).is_err());
        assert!(ema.update_with_decay(&src, f64::NAN).is_err());
        assert!(ema.update_with_decay(&src, 1.1).is_err());
        assert!(ema.update_with_decay(&src, -0.01).is_err());
    }

    #[test]
    fn edm2_power_ema_impulse_response_matches_relative_width() {
        // Verify that the discrete impulse response of `Edm2PowerEmaSchedule` has empirical relative
        // standard deviation `Std[t / T]` matching `sigma_rel` (Karras et al. CVPR 2024, Appendix C Eq. 124).
        // Verify exact Eq. 124 cubic `(gamma + 2)^2 (gamma + 3) / (gamma + 1) = 1 / sigma_rel^2` and known
        // Karras et al. Table 1 values (`sigma_rel = 0.10 -> gamma ≈ 6.94`, `sigma_rel = 0.05 -> gamma ≈ 16.97`).
        let s10 = Edm2PowerEmaSchedule::from_relative_width(0.10, 0.0).unwrap();
        assert!((s10.gamma() - 6.937_203_937_6).abs() < 1e-8, "gamma(0.10)={}", s10.gamma());
        let s05 = Edm2PowerEmaSchedule::from_relative_width(0.05, 0.0).unwrap();
        assert!((s05.gamma() - 16.972_198_602_3).abs() < 1e-8, "gamma(0.05)={}", s05.gamma());
        for &sigma_rel in &[0.05f64, 0.10, 0.15, 0.20, 0.28] {
            let g = Edm2PowerEmaSchedule::from_relative_width(sigma_rel, 0.0).unwrap().gamma();
            let lhs = (g + 2.0).powi(2) * (g + 3.0) / (g + 1.0);
            let rhs = 1.0 / (sigma_rel * sigma_rel);
            assert!(
                (lhs - rhs).abs() <= 1e-12 * rhs,
                "Eq. 124 mismatch at sigma_rel={sigma_rel}: {lhs} vs {rhs}"
            );
        }

        let steps = 2000usize;
        for &sigma_rel in &[0.05f64, 0.10, 0.15, 0.20] {
            let sched = Edm2PowerEmaSchedule::from_relative_width(sigma_rel, 0.0).unwrap();
            assert!((sched.relative_width() - sigma_rel).abs() < 1e-13);
            // Roll out exact impulse-response weights w[k] for samples at t = 1..=steps.
            let mut weights = vec![0.0f64; steps];
            for t in 1..=steps {
                let b = sched.decay_at(t as u64);
                for w in &mut weights[..t - 1] {
                    *w *= b;
                }
                weights[t - 1] = 1.0 - b;
            }
            let sum_w: f64 = weights.iter().sum();
            assert!((sum_w - 1.0).abs() < 1e-12, "weights must sum to 1, got {sum_w}");
            let mean: f64 =
                weights.iter().enumerate().map(|(i, &w)| w * ((i + 1) as f64 / steps as f64)).sum();
            let var: f64 = weights
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let x = (i + 1) as f64 / steps as f64 - mean;
                    w * x * x
                })
                .sum();
            let empirical_sigma_rel = var.sqrt();
            assert!(
                (empirical_sigma_rel - sigma_rel).abs() < 1e-3,
                "EDM2 impulse relative width Std[t/T] for sigma_rel={sigma_rel}: empirical={empirical_sigma_rel}"
            );
        }

        // Extreme sigma_rel = 1e-200 does not overflow Newton inversion or `relative_width`.
        let ultra = Edm2PowerEmaSchedule::from_relative_width(1e-200, 0.0).unwrap();
        assert!(ultra.gamma().is_finite() && ultra.relative_width().is_finite());
        assert!((ultra.relative_width() - 1e-200).abs() <= 1e-212);

        // Serde round-trip & validation (`max_sigma = 1 / sqrt(12) ≈ 0.288675`).
        let s = Edm2PowerEmaSchedule::new(6.94, 10.0).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Edm2PowerEmaSchedule>(&json).unwrap(), s);
        assert!(Edm2PowerEmaSchedule::from_relative_width(0.0, 0.0).is_err());
        assert!(Edm2PowerEmaSchedule::from_relative_width(0.29, 0.0).is_err());
        assert!(Edm2PowerEmaSchedule::new(0.0, 0.0).is_err());
        assert!(Edm2PowerEmaSchedule::new(5.0, -1.0).is_err());
    }
}
