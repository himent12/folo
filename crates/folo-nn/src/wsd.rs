//! Standalone Warmup–Stable–Decay (WSD; Hu et al. 2024 / Hägele et al. 2024) learning-rate schedule.
//!
//! Step-indexed three-phase schedule with explicit step counts `(warmup_steps, stable_steps, decay_steps)`
//! and configurable cooldown tail shape ([`WsdTail::Linear`], [`WsdTail::Cosine`], or [`WsdTail::Sqrt`]
//! `1 - sqrt(u)`):
//!
//! ```text
//! step in [0, W):                 linear ramp from warmup_start_lr -> base_lr
//! step in [W, W + S):             constant plateau at base_lr
//! step in [W + S, W + S + D):     cooldown from base_lr -> min_lr using tail(u), u = (step - (W + S)) / D
//! step >= W + S + D:              clamped at min_lr (or base_lr when D == 0)
//! ```
//!
//! Guarantees exact continuity at phase boundaries (`lr(W) == base_lr`, `lr(W + S) == base_lr`,
//! `lr(W + S + D) == min_lr` when `D > 0`), strict monotonicity during warmup and decay, overflow-free
//! `u128` boundary arithmetic for arbitrary `u64` step durations, and validated serde round-tripping.

use serde::{Deserialize, Serialize};

use crate::optim::{Optimizer, ParamGroup};
use crate::sched::WsdTail;
use crate::{Error, Result};

/// Phase of a [`WsdSchedule`] at a given step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WsdPhase {
    /// Linear warmup (`step < warmup_steps`).
    Warmup,
    /// Constant learning-rate plateau (`warmup_steps <= step < warmup_steps + stable_steps`).
    Stable,
    /// Cooldown decay (`warmup_steps + stable_steps <= step < total_steps`).
    Decay,
    /// Post-schedule plateau (`step >= total_steps`).
    Finished,
}

/// Configuration for [`WsdSchedule`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WsdConfig {
    /// Peak / plateau learning rate `> 0` (default `1e-3`).
    pub base_lr: f64,
    /// Final cooldown learning rate in `[0, base_lr]` (default `0.0`).
    pub min_lr: f64,
    /// Initial warmup learning rate at `step = 0`, in `[0, base_lr]` (default `0.0`).
    pub warmup_start_lr: f64,
    /// Number of linear warmup steps (default `0`).
    pub warmup_steps: u64,
    /// Number of constant-LR plateau steps (default `0`).
    pub stable_steps: u64,
    /// Number of cooldown decay steps (default `0`).
    pub decay_steps: u64,
    /// Cooldown decay curve: [`WsdTail::Linear`], [`WsdTail::Cosine`], or [`WsdTail::Sqrt`] (`1 - sqrt(u)`).
    pub tail: WsdTail,
}

impl Default for WsdConfig {
    fn default() -> Self {
        Self {
            base_lr: 1e-3,
            min_lr: 0.0,
            warmup_start_lr: 0.0,
            warmup_steps: 0,
            stable_steps: 0,
            decay_steps: 0,
            tail: WsdTail::Linear,
        }
    }
}

impl WsdConfig {
    /// Validate all WSD configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.base_lr.is_finite() && self.base_lr > 0.0) {
            return Err(Error::Invalid(format!("wsd: base_lr must be finite and > 0, got {}", self.base_lr)));
        }
        if !(self.min_lr.is_finite() && self.min_lr >= 0.0 && self.min_lr <= self.base_lr) {
            return Err(Error::Invalid(format!(
                "wsd: min_lr must be finite and in [0, base_lr ({})], got {}",
                self.base_lr, self.min_lr
            )));
        }
        if !(self.warmup_start_lr.is_finite()
            && self.warmup_start_lr >= 0.0
            && self.warmup_start_lr <= self.base_lr)
        {
            return Err(Error::Invalid(format!(
                "wsd: warmup_start_lr must be finite and in [0, base_lr ({})], got {}",
                self.base_lr, self.warmup_start_lr
            )));
        }
        Ok(())
    }

    /// Total planned schedule length `warmup_steps + stable_steps + decay_steps` in `u128`.
    #[inline]
    pub fn total_steps(&self) -> u128 {
        self.warmup_steps as u128 + self.stable_steps as u128 + self.decay_steps as u128
    }
}

/// Step-indexed Warmup–Stable–Decay (WSD) learning-rate schedule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WsdSchedule {
    config: WsdConfig,
    step: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawWsdSchedule {
    config: WsdConfig,
    step: u64,
}

impl<'de> Deserialize<'de> for WsdSchedule {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawWsdSchedule::deserialize(deserializer)?;
        WsdSchedule::at_step(raw.config, raw.step).map_err(serde::de::Error::custom)
    }
}

impl WsdSchedule {
    /// Construct a validated [`WsdSchedule`] starting at `step = 0`.
    pub fn new(config: WsdConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, step: 0 })
    }

    /// Construct a validated [`WsdSchedule`] resumed at `step`.
    pub fn at_step(config: WsdConfig, step: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, step })
    }

    /// Underlying [`WsdConfig`].
    pub fn config(&self) -> &WsdConfig {
        &self.config
    }

    /// Current step counter.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Advance the step counter by 1 (saturating at `u64::MAX`).
    pub fn advance(&mut self) {
        self.step = self.step.saturating_add(1);
    }

    /// Set the step counter explicitly.
    pub fn set_step(&mut self, step: u64) {
        self.step = step;
    }

    /// Active [`WsdPhase`] at `step`.
    pub fn phase_at(&self, step: u64) -> WsdPhase {
        let s = step as u128;
        let w = self.config.warmup_steps as u128;
        let decay_start = w + self.config.stable_steps as u128;
        let total = decay_start + self.config.decay_steps as u128;
        if s < w {
            WsdPhase::Warmup
        } else if s < decay_start {
            WsdPhase::Stable
        } else if s < total {
            WsdPhase::Decay
        } else {
            WsdPhase::Finished
        }
    }

    /// Learning rate at the current [`Self::step`].
    pub fn lr(&self) -> f64 {
        self.lr_at(self.step)
    }

    /// Learning rate at an arbitrary `step: u64`.
    ///
    /// Uses endpoint-anchored interpolation and half-angle squared-sine identities so large
    /// `base_lr` (up to `f64::MAX`) and long decay tails never overflow or suffer cancellation.
    pub fn lr_at(&self, step: u64) -> f64 {
        let s = step as u128;
        let w = self.config.warmup_steps as u128;
        let decay_start = w + self.config.stable_steps as u128;
        let d = self.config.decay_steps as u128;
        let total = decay_start + d;

        if s < w {
            let span = self.config.base_lr - self.config.warmup_start_lr;
            return if s <= w / 2 {
                self.config.warmup_start_lr + span * (s as f64 / w as f64)
            } else {
                self.config.base_lr - span * ((w - s) as f64 / w as f64)
            };
        }
        if s < decay_start || d == 0 {
            return self.config.base_lr;
        }
        if s >= total {
            return self.config.min_lr;
        }

        let pos = s - decay_start;
        let rem = total - s;
        let span = self.config.base_lr - self.config.min_lr;
        match self.config.tail {
            WsdTail::Linear => {
                if pos <= d / 2 {
                    self.config.base_lr - span * (pos as f64 / d as f64)
                } else {
                    self.config.min_lr + span * (rem as f64 / d as f64)
                }
            }
            WsdTail::Cosine => {
                let half_pi = std::f64::consts::FRAC_PI_2;
                if pos <= d / 2 {
                    let sin_v = (half_pi * (pos as f64 / d as f64)).sin();
                    self.config.base_lr - span * (sin_v * sin_v)
                } else {
                    let sin_v = (half_pi * (rem as f64 / d as f64)).sin();
                    self.config.min_lr + span * (sin_v * sin_v)
                }
            }
            WsdTail::Sqrt => {
                let u = (pos as f64 / d as f64).clamp(0.0, 1.0);
                if pos <= d / 2 {
                    self.config.base_lr - span * u.sqrt()
                } else {
                    // 1 - sqrt(u) = (1 - u) / (1 + sqrt(u)) avoids cancellation near u -> 1.
                    let one_minus_u = rem as f64 / d as f64;
                    let keep = one_minus_u / (1.0 + u.sqrt());
                    self.config.min_lr + span * keep
                }
            }
        }
    }

    /// Relative multiplier `lr_at(step) / base_lr`.
    pub fn factor_at(&self, step: u64) -> f64 {
        self.lr_at(step) / self.config.base_lr
    }

    /// Apply the current step's relative LR multiplier to every [`ParamGroup`] (`g.lr = g.base_lr * factor`).
    pub fn apply_to_groups(&self, groups: &mut [ParamGroup]) {
        let f = self.factor_at(self.step);
        for g in groups {
            g.lr = g.base_lr * f;
        }
    }

    /// Apply the current step's relative LR multiplier to an [`Optimizer`].
    pub fn apply(&self, opt: &mut dyn Optimizer) {
        self.apply_to_groups(opt.groups_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_core::Tensor;

    #[test]
    fn wsd_phases_boundaries_and_monotonicity_across_all_tails() {
        for tail in [WsdTail::Linear, WsdTail::Cosine, WsdTail::Sqrt] {
            let cfg = WsdConfig {
                base_lr: 0.1,
                min_lr: 0.01,
                warmup_start_lr: 0.005,
                warmup_steps: 10,
                stable_steps: 20,
                decay_steps: 15,
                tail,
            };
            let sched = WsdSchedule::new(cfg).unwrap();
            assert_eq!(cfg.total_steps(), 45);

            // Warmup [0, 10): strictly increasing, continuous into step 10 == base_lr.
            assert_eq!(sched.phase_at(0), WsdPhase::Warmup);
            assert_eq!(sched.lr_at(0), 0.005);
            for s in 0..10 {
                assert_eq!(sched.phase_at(s), WsdPhase::Warmup);
                assert!(sched.lr_at(s) < sched.lr_at(s + 1), "{tail:?} warmup step {s}");
            }

            // Stable plateau [10, 30): exact constant base_lr.
            for s in 10..30 {
                assert_eq!(sched.phase_at(s), WsdPhase::Stable);
                assert_eq!(sched.lr_at(s), 0.1, "{tail:?} stable step {s}");
            }

            // Decay [30, 45): starts at base_lr, strictly monotonically decreasing, lands on min_lr at 45.
            assert_eq!(sched.phase_at(30), WsdPhase::Decay);
            assert_eq!(sched.lr_at(30), 0.1);
            for s in 30..45 {
                assert_eq!(sched.phase_at(s), WsdPhase::Decay);
                assert!(sched.lr_at(s) > sched.lr_at(s + 1), "{tail:?} decay step {s}");
            }

            // Finished [45, ..): clamped at min_lr.
            for s in [45u64, 46, 1000, u64::MAX] {
                assert_eq!(sched.phase_at(s), WsdPhase::Finished);
                assert_eq!(sched.lr_at(s), 0.01);
            }

            // Analytical midpoints.
            if tail == WsdTail::Linear {
                assert!((sched.lr_at(30 + 3) - (0.1 - 0.09 * 0.2)).abs() < 1e-15);
            } else if tail == WsdTail::Sqrt {
                // At u = 4/16 -> test exact u = 0.25 with decay_steps = 16.
                let s16 = WsdSchedule::new(WsdConfig { decay_steps: 16, ..cfg }).unwrap();
                let want = 0.01 + 0.09 * (1.0 - 0.5);
                assert!((s16.lr_at(34) - want).abs() < 1e-15);
            } else if tail == WsdTail::Cosine {
                let s16 = WsdSchedule::new(WsdConfig { decay_steps: 16, ..cfg }).unwrap();
                assert!((s16.lr_at(38) - 0.055).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn wsd_edge_cases_apply_serde_and_validation() {
        // Zero decay_steps holds base_lr after stable plateau.
        let no_decay = WsdSchedule::new(WsdConfig {
            base_lr: 0.05,
            min_lr: 0.001,
            warmup_start_lr: 0.0,
            warmup_steps: 4,
            stable_steps: 6,
            decay_steps: 0,
            tail: WsdTail::Cosine,
        })
        .unwrap();
        assert_eq!(no_decay.lr_at(4), 0.05);
        assert_eq!(no_decay.lr_at(100), 0.05);
        assert_eq!(no_decay.phase_at(10), WsdPhase::Finished);

        // Huge step durations (u64::MAX each) and f64::MAX base_lr stay overflow-free and finite.
        let huge = WsdSchedule::new(WsdConfig {
            base_lr: f64::MAX,
            min_lr: 1.5 * 2.0f64.powi(971),
            warmup_start_lr: 1.5 * 2.0f64.powi(971),
            warmup_steps: u64::MAX / 2,
            stable_steps: u64::MAX / 2,
            decay_steps: u64::MAX,
            tail: WsdTail::Sqrt,
        })
        .unwrap();
        for s in [0u64, u64::MAX / 4, u64::MAX / 2, u64::MAX - 1, u64::MAX] {
            let v = huge.lr_at(s);
            assert!(v.is_finite() && v > 0.0);
        }

        // apply_to_groups preserves per-group base_lr ratios.
        let mut sched = WsdSchedule::new(WsdConfig {
            base_lr: 0.1,
            min_lr: 0.0,
            warmup_start_lr: 0.0,
            warmup_steps: 0,
            stable_steps: 2,
            decay_steps: 4,
            tail: WsdTail::Linear,
        })
        .unwrap();
        let mut groups = vec![
            ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([2]))]).lr(0.02),
            ParamGroup::new("bias", vec![("b".into(), Tensor::zeros([2]))]).lr(0.06),
        ];
        sched.set_step(4); // midpoint of decay -> factor 0.5
        sched.apply_to_groups(&mut groups);
        assert!((groups[0].lr - 0.01).abs() < 1e-15);
        assert!((groups[1].lr - 0.03).abs() < 1e-15);
        sched.advance();
        assert_eq!(sched.step(), 5);

        // Serde round-trip and invalid JSON / config rejection.
        let json = serde_json::to_string(&sched).unwrap();
        assert_eq!(serde_json::from_str::<WsdSchedule>(&json).unwrap(), sched);
        for bad_json in [
            r#"{"config":{"base_lr":0.0},"step":0}"#,
            r#"{"config":{"base_lr":0.1,"min_lr":0.2},"step":0}"#,
            r#"{"config":{"base_lr":0.1,"warmup_start_lr":-0.1},"step":0}"#,
        ] {
            assert!(serde_json::from_str::<WsdSchedule>(bad_json).is_err());
        }
        for bad in [
            WsdConfig { base_lr: 0.0, ..Default::default() },
            WsdConfig { base_lr: f64::NAN, ..Default::default() },
            WsdConfig { base_lr: 0.1, min_lr: -0.01, ..Default::default() },
            WsdConfig { base_lr: 0.1, min_lr: 0.2, ..Default::default() },
            WsdConfig { base_lr: 0.1, warmup_start_lr: 0.2, ..Default::default() },
        ] {
            assert!(WsdSchedule::new(bad).is_err());
        }
    }
}
