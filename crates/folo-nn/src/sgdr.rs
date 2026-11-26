//! Standalone Cosine Annealing with Warm Restarts (SGDR; Loshchilov & Hutter, ICLR 2017).
//!
//! Unbounded step-indexed schedule with optional linear warmup followed by geometric cosine restart
//! cycles of lengths `T_c = T_0 * T_mult^c`:
//!
//! ```text
//! step < warmup_steps:
//!   lr(step) = warmup_start_lr + (base_lr - warmup_start_lr) * (step / warmup_steps)
//!
//! step >= warmup_steps (with t = step - warmup_steps, cycle c, offset t_cur in [0, T_c)):
//!   lr(step) = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t_cur / T_c))
//! ```
//!
//! Guarantees exact continuity at the warmup-to-cycle-0 handoff (`lr(warmup_steps) == base_lr`),
//! exact restart peaks (`lr == base_lr` whenever `t_cur == 0`), strict monotonicity within every
//! cycle (`base_lr > eta_min`), and overflow-free evaluation for every `step: u64`.

use serde::{Deserialize, Serialize};

use crate::optim::{Optimizer, ParamGroup};
use crate::{Error, Result};

/// Configuration for [`SgdrSchedule`] (Loshchilov & Hutter, ICLR 2017).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SgdrConfig {
    /// Peak learning rate `eta_max > 0` at every warm restart (default `1e-3`).
    pub base_lr: f64,
    /// Minimum learning rate `eta_min in [0, base_lr]` approached at the end of each cycle (default `0.0`).
    pub eta_min: f64,
    /// Length `T_0 >= 1` of the first post-warmup cosine cycle in steps (default `10`).
    pub t_0: u64,
    /// Integer geometric cycle-length multiplier `T_mult >= 1` (default `1`).
    pub t_mult: u64,
    /// Optional linear warmup duration in steps before cycle 0 starts (`0` disables warmup).
    pub warmup_steps: u64,
    /// Initial learning rate at `step = 0` during linear warmup (`in [0, base_lr]`, default `0.0`).
    pub warmup_start_lr: f64,
}

impl Default for SgdrConfig {
    fn default() -> Self {
        Self { base_lr: 1e-3, eta_min: 0.0, t_0: 10, t_mult: 1, warmup_steps: 0, warmup_start_lr: 0.0 }
    }
}

impl SgdrConfig {
    /// Validate all SGDR parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.base_lr.is_finite() && self.base_lr > 0.0) {
            return Err(Error::Invalid(format!(
                "sgdr: base_lr must be finite and > 0, got {}",
                self.base_lr
            )));
        }
        if !(self.eta_min.is_finite() && self.eta_min >= 0.0 && self.eta_min <= self.base_lr) {
            return Err(Error::Invalid(format!(
                "sgdr: eta_min must be finite and in [0, base_lr ({})], got {}",
                self.base_lr, self.eta_min
            )));
        }
        if self.t_0 == 0 {
            return Err(Error::Invalid("sgdr: t_0 must be >= 1, got 0".into()));
        }
        if self.t_mult == 0 {
            return Err(Error::Invalid("sgdr: t_mult must be >= 1, got 0".into()));
        }
        if !(self.warmup_start_lr.is_finite()
            && self.warmup_start_lr >= 0.0
            && self.warmup_start_lr <= self.base_lr)
        {
            return Err(Error::Invalid(format!(
                "sgdr: warmup_start_lr must be finite and in [0, base_lr ({})], got {}",
                self.base_lr, self.warmup_start_lr
            )));
        }
        Ok(())
    }
}

/// Phase / cycle breakdown returned by [`SgdrSchedule::cycle_info`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdrCycleInfo {
    /// `true` while `step < warmup_steps`.
    pub in_warmup: bool,
    /// Zero-based restart cycle index (`0` during warmup and the first cosine cycle).
    pub cycle: u64,
    /// Zero-based step offset within the current warmup phase or cosine cycle.
    pub step_in_cycle: u64,
    /// Exact length of the current cycle as `Some(len)` when it fits in `u128` (`None` only if
    /// `T_0 * T_mult^cycle` exceeds `u128::MAX`).
    pub cycle_len: Option<u128>,
    /// Length of the current warmup phase or cosine cycle in `f64` (exact for all practical runs and
    /// finite for every `u64` step).
    pub cycle_len_f64: f64,
}

/// Step-indexed Cosine Annealing with Warm Restarts (SGDR) learning-rate schedule.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SgdrSchedule {
    config: SgdrConfig,
    step: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSgdrSchedule {
    config: SgdrConfig,
    step: u64,
}

impl<'de> Deserialize<'de> for SgdrSchedule {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawSgdrSchedule::deserialize(deserializer)?;
        SgdrSchedule::at_step(raw.config, raw.step).map_err(serde::de::Error::custom)
    }
}

impl SgdrSchedule {
    /// Construct a validated [`SgdrSchedule`] starting at `step = 0`.
    pub fn new(config: SgdrConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, step: 0 })
    }

    /// Construct a validated [`SgdrSchedule`] resumed at `step`.
    pub fn at_step(config: SgdrConfig, step: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self { config, step })
    }

    /// The schedule configuration.
    pub fn config(&self) -> &SgdrConfig {
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

    /// Reset the step counter to `step`.
    pub fn set_step(&mut self, step: u64) {
        self.step = step;
    }

    /// Learning rate at the current [`Self::step`].
    pub fn lr(&self) -> f64 {
        self.lr_at(self.step)
    }

    /// Inspect warmup / cycle index, in-cycle offset, and cycle length at `step`.
    pub fn cycle_info(&self, step: u64) -> SgdrCycleInfo {
        let w = self.config.warmup_steps;
        if step < w {
            return SgdrCycleInfo {
                in_warmup: true,
                cycle: 0,
                step_in_cycle: step,
                cycle_len: Some(w as u128),
                cycle_len_f64: w as f64,
            };
        }
        let t = (step - w) as u128;
        let t_0 = self.config.t_0 as u128;
        let t_mult = self.config.t_mult as u128;
        if t_mult == 1 {
            let cycle = (t / t_0) as u64;
            let pos = (t % t_0) as u64;
            return SgdrCycleInfo {
                in_warmup: false,
                cycle,
                step_in_cycle: pos,
                cycle_len: Some(t_0),
                cycle_len_f64: t_0 as f64,
            };
        }
        let mut rem = t;
        let mut cur = t_0;
        let mut cycle = 0u64;
        let mut overflowed_f64: Option<f64> = None;
        while rem >= cur {
            rem -= cur;
            cycle += 1;
            match cur.checked_mul(t_mult) {
                Some(next) => cur = next,
                None => {
                    overflowed_f64 = Some((cur as f64) * (t_mult as f64));
                    break;
                }
            }
        }
        match overflowed_f64 {
            Some(len_f64) => SgdrCycleInfo {
                in_warmup: false,
                cycle,
                step_in_cycle: rem as u64,
                cycle_len: None,
                cycle_len_f64: len_f64,
            },
            None => SgdrCycleInfo {
                in_warmup: false,
                cycle,
                step_in_cycle: rem as u64,
                cycle_len: Some(cur),
                cycle_len_f64: cur as f64,
            },
        }
    }

    /// Learning rate at an arbitrary `step: u64`.
    ///
    /// Uses half-angle squared-sine identities interpolated from the nearer endpoint (`base_lr` in the
    /// first half of a cycle, `eta_min` using exact remaining steps `T_c - t_cur` in the second half) so
    /// long-cycle tails keep full precision and extreme valid configs (e.g. `base_lr = f64::MAX`) never
    /// overflow to `+Inf`.
    pub fn lr_at(&self, step: u64) -> f64 {
        let info = self.cycle_info(step);
        if info.in_warmup {
            let w = self.config.warmup_steps;
            let span = self.config.base_lr - self.config.warmup_start_lr;
            return if info.step_in_cycle <= w / 2 {
                let frac = (info.step_in_cycle as f64) / (w as f64);
                self.config.warmup_start_lr + span * frac
            } else {
                let rem_frac = ((w - info.step_in_cycle) as f64) / (w as f64);
                self.config.base_lr - span * rem_frac
            };
        }
        if info.step_in_cycle == 0 {
            return self.config.base_lr;
        }
        let span = self.config.base_lr - self.config.eta_min;
        let half_pi = std::f64::consts::FRAC_PI_2;
        let t_cur = info.step_in_cycle as u128;
        match info.cycle_len {
            Some(len) if t_cur <= len / 2 => {
                let s = (half_pi * (t_cur as f64 / len as f64)).sin();
                self.config.base_lr - span * (s * s)
            }
            Some(len) => {
                let rem = len - t_cur;
                let s = (half_pi * (rem as f64 / len as f64)).sin();
                self.config.eta_min + span * (s * s)
            }
            None => {
                // `len > u128::MAX > u64::MAX >= t_cur`, so `u = t_cur / len_f64 << 0.5` is strictly in the first half.
                let s = (half_pi * (info.step_in_cycle as f64 / info.cycle_len_f64)).sin();
                self.config.base_lr - span * (s * s)
            }
        }
    }

    /// Relative multiplier `lr_at(step) / base_lr` in `[eta_min / base_lr, 1.0]` (or `[warmup_start_lr / base_lr, 1.0]`
    /// during warmup).
    pub fn factor_at(&self, step: u64) -> f64 {
        self.lr_at(step) / self.config.base_lr
    }

    /// Apply the current step's relative LR factor to every [`ParamGroup`] (`g.lr = g.base_lr * factor`),
    /// preserving per-group LR ratios.
    pub fn apply_to_groups(&self, groups: &mut [ParamGroup]) {
        let f = self.factor_at(self.step);
        for g in groups {
            g.lr = g.base_lr * f;
        }
    }

    /// Apply the current step's relative LR factor to an [`Optimizer`].
    pub fn apply(&self, opt: &mut dyn Optimizer) {
        self.apply_to_groups(opt.groups_mut());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_core::Tensor;

    #[test]
    fn sgdr_warmup_continuity_restarts_and_monotonicity() {
        let cfg = SgdrConfig {
            base_lr: 0.1,
            eta_min: 0.01,
            t_0: 10,
            t_mult: 2,
            warmup_steps: 5,
            warmup_start_lr: 0.005,
        };
        let sched = SgdrSchedule::new(cfg).unwrap();

        // Warmup phase [0, 5): starts at warmup_start_lr, strictly increasing, continuous into step 5 == base_lr.
        assert_eq!(sched.lr_at(0), 0.005);
        for s in 0..5 {
            let info = sched.cycle_info(s);
            assert!(info.in_warmup && info.step_in_cycle == s && info.cycle_len == Some(5));
            assert!(sched.lr_at(s) < sched.lr_at(s + 1), "warmup must increase at step {s}");
        }
        assert!((sched.lr_at(5) - 0.1).abs() < 1e-15);

        // Cycle boundaries:
        // Cycle 0: steps [5, 15)  len 10 -> restart at 5,  midpoint at 10
        // Cycle 1: steps [15, 35) len 20 -> restart at 15, midpoint at 25
        // Cycle 2: steps [35, 75) len 40 -> restart at 35, midpoint at 55
        let cycles = [(0u64, 5u64, 10u64), (1, 15, 20), (2, 35, 40)];
        for &(c_idx, start, len) in &cycles {
            let info0 = sched.cycle_info(start);
            assert!(!info0.in_warmup && info0.cycle == c_idx && info0.step_in_cycle == 0);
            assert_eq!(info0.cycle_len, Some(len as u128));
            assert_eq!(sched.lr_at(start), cfg.base_lr);

            // Exact midpoint for even cycle length: 0.5 * (base_lr + eta_min) = 0.055.
            let mid = start + len / 2;
            assert!((sched.lr_at(mid) - 0.055).abs() < 1e-15, "cycle {c_idx} midpoint");

            // Strictly monotonically decreasing within the cycle and strictly above eta_min.
            let mut prev = sched.lr_at(start);
            for offset in 1..len {
                let cur = sched.lr_at(start + offset);
                assert!(
                    cur < prev,
                    "cycle {c_idx} offset {offset}: not strictly decreasing ({cur} >= {prev})"
                );
                assert!(cur > cfg.eta_min && cur < cfg.base_lr);
                prev = cur;
            }
            // Jump back up to exact base_lr at next restart boundary.
            assert_eq!(sched.lr_at(start + len), cfg.base_lr);
        }
    }

    #[test]
    fn sgdr_t_mult_1_edge_cases_huge_steps_and_apply_to_groups() {
        // Constant cycle length T_mult = 1.
        let cfg = SgdrConfig {
            base_lr: 0.02,
            eta_min: 0.002,
            t_0: 4,
            t_mult: 1,
            warmup_steps: 0,
            warmup_start_lr: 0.0,
        };
        let mut sched = SgdrSchedule::new(cfg).unwrap();
        for c in 0..5u64 {
            assert_eq!(sched.lr_at(c * 4), 0.02);
            assert!((sched.lr_at(c * 4 + 2) - 0.011).abs() < 1e-15);
            assert_eq!(sched.cycle_info(c * 4 + 3).cycle, c);
            assert_eq!(sched.cycle_info(c * 4 + 3).step_in_cycle, 3);
        }

        // Degenerate t_0 = 1, t_mult = 1 and eta_min == base_lr stay finite and exact.
        let p1 =
            SgdrSchedule::new(SgdrConfig { t_0: 1, t_mult: 1, base_lr: 0.05, eta_min: 0.01, ..cfg }).unwrap();
        for s in 0..5 {
            assert_eq!(p1.lr_at(s), 0.05);
        }
        let flat = SgdrSchedule::new(SgdrConfig { base_lr: 0.05, eta_min: 0.05, ..cfg }).unwrap();
        for s in 0..10 {
            assert_eq!(flat.lr_at(s), 0.05);
        }

        // Extreme u64::MAX step and u128 overflow in T_0 * T_mult^c stay finite and bounded.
        let huge = SgdrSchedule::new(SgdrConfig {
            base_lr: 1e-2,
            eta_min: 1e-5,
            t_0: 10,
            t_mult: u64::MAX / 2,
            warmup_steps: 3,
            warmup_start_lr: 0.0,
        })
        .unwrap();
        let lr_max = huge.lr_at(u64::MAX);
        assert!(lr_max.is_finite() && (1e-5..=1e-2).contains(&lr_max));

        // apply_to_groups preserves per-group base_lr ratios and advances step.
        let mut groups = vec![
            ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([2]))]).lr(0.02),
            ParamGroup::new("bias", vec![("b".into(), Tensor::zeros([2]))]).lr(0.06),
        ];
        sched.set_step(2);
        sched.apply_to_groups(&mut groups);
        assert!((groups[0].lr - 0.011).abs() < 1e-15);
        assert!((groups[1].lr - 0.033).abs() < 1e-15);
        sched.advance();
        assert_eq!(sched.step(), 3);

        // Serde round-trip & invalid config/JSON deserialization rejections.
        let json = serde_json::to_string(&sched).unwrap();
        assert_eq!(serde_json::from_str::<SgdrSchedule>(&json).unwrap(), sched);
        for bad_json in [
            r#"{"config":{"t_0":0},"step":0}"#,
            r#"{"config":{"t_mult":0},"step":5}"#,
            r#"{"config":{"base_lr":0.0},"step":0}"#,
            r#"{"config":{"base_lr":0.1,"eta_min":0.2},"step":0}"#,
        ] {
            assert!(
                serde_json::from_str::<SgdrSchedule>(bad_json).is_err(),
                "invalid SGDR JSON must fail deserialization: {bad_json}"
            );
        }
        for bad in [
            SgdrConfig { base_lr: 0.0, ..cfg },
            SgdrConfig { base_lr: -1.0, ..cfg },
            SgdrConfig { base_lr: f64::NAN, ..cfg },
            SgdrConfig { eta_min: -0.1, ..cfg },
            SgdrConfig { eta_min: 0.03, ..cfg }, // > base_lr
            SgdrConfig { t_0: 0, ..cfg },
            SgdrConfig { t_mult: 0, ..cfg },
            SgdrConfig { warmup_start_lr: -0.01, ..cfg },
            SgdrConfig { warmup_start_lr: 0.05, ..cfg }, // > base_lr
        ] {
            assert!(bad.validate().is_err(), "{bad:?} should be rejected");
            assert!(SgdrSchedule::new(bad).is_err());
        }

        // Long-cycle tail precision (t_0 = 1e9, base_lr = 1, eta_min = 0) at final in-cycle step:
        // sin(pi / (2 * 1e9))^2 ≈ (pi / 2e9)^2 ≈ 2.4674011002723397e-18 (must not cancel to 0.0).
        let long_cycle = SgdrSchedule::new(SgdrConfig {
            base_lr: 1.0,
            eta_min: 0.0,
            t_0: 1_000_000_000,
            t_mult: 1,
            warmup_steps: 0,
            warmup_start_lr: 0.0,
        })
        .unwrap();
        let tail = long_cycle.lr_at(999_999_999);
        let want_tail = (std::f64::consts::FRAC_PI_2 / 1e9).sin().powi(2);
        assert!((tail - want_tail).abs() <= 1e-12 * want_tail, "tail={tail:e} want={want_tail:e}");

        // Extreme base_lr = f64::MAX with non-zero eta_min never overflows to +Inf.
        let eta_min_ext = 1.5 * 2.0f64.powi(971);
        let extreme = SgdrSchedule::new(SgdrConfig {
            base_lr: f64::MAX,
            eta_min: eta_min_ext,
            t_0: 1_000_000_000,
            t_mult: 1,
            warmup_steps: 10,
            warmup_start_lr: eta_min_ext,
        })
        .unwrap();
        for s in [0u64, 1, 5, 9, 10, 11, 500_000_010, 1_000_000_009] {
            let v = extreme.lr_at(s);
            assert!(v.is_finite() && v >= eta_min_ext && v <= f64::MAX, "extreme step {s}: {v:e}");
        }
    }
}
