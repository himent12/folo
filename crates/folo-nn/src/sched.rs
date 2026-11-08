//! The LR / momentum schedule of docs/MODEL.md §8.4: linear warmup, then linear or cosine decay, counted in
//! optimizer steps (`folo.sched` in docs/FORMAT.md).

use serde::{Deserialize, Serialize};

use crate::optim::{BIAS_GROUP, Optimizer, ParamGroup};
use crate::{Error, Result};

/// The decay shape after warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DecayKind {
    /// `lf(t) = (1 − t / max(T − 1, 1)) · (1 − lrf) + lrf` (the default).
    Linear,
    /// `lf(t) = lrf + (1 − lrf) · (1 + cos(π · t / max(T − 1, 1))) / 2` (`cos_lr`).
    Cosine,
}

/// The unit the schedule advances in. v1 has one: an optimizer step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchedUnit {
    /// One `optimizer.step()`.
    Iteration,
}

/// `folo.sched` (docs/FORMAT.md). Given this and the optimizer groups, every group's LR and momentum at
/// every iteration is determined.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedContract {
    /// Decay shape.
    #[serde(rename = "type")]
    pub kind: DecayKind,
    /// Always `"iteration"`.
    pub unit: SchedUnit,
    /// Number of epochs.
    pub epochs: u64,
    /// Optimizer steps per epoch (`spe`).
    pub steps_per_epoch: u64,
    /// Final LR fraction.
    pub lrf: f64,
    /// Warmup length `W` in iterations.
    pub warmup_iters: u64,
    /// The bias group's LR at iteration 0.
    pub warmup_bias_lr: f64,
    /// Momentum / β1 at iteration 0.
    pub warmup_momentum: f64,
    /// Completed iterations; equals `sched.__step`.
    pub step: u64,
}

/// The schedule state: its parameters and the iteration counter `t`.
///
/// Each optimizer step: [`LrSchedule::apply`] (sets every group's `lr` and `momentum` for iteration `t`),
/// then `optimizer.step()`, then [`LrSchedule::advance`].
///
/// ```
/// use folo_nn::{DecayKind, LrSchedule};
/// // docs/MODEL.md §10.4: 10 epochs of 13 steps, 3 warmup epochs (floored to 100 iterations).
/// let s = LrSchedule::new(DecayKind::Linear, 10, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
/// assert_eq!((s.total_iters(), s.warmup_iters()), (130, 100));
/// assert_eq!(s.lf(0), 1.0);
/// assert!((s.lf(129) - 0.01).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LrSchedule {
    c: SchedContract,
}

/// docs/MODEL.md §8.4: `W = 0` if `warmup_epochs <= 0`, else
/// `min(max(round(warmup_epochs · spe), 100), max(epochs − 1, 0) · spe)`.
pub fn warmup_iters(warmup_epochs: f64, epochs: u64, steps_per_epoch: u64) -> u64 {
    if warmup_epochs.is_nan() || warmup_epochs <= 0.0 {
        return 0;
    }
    let w = (warmup_epochs * steps_per_epoch as f64).round().max(100.0) as u64;
    w.min(epochs.saturating_sub(1).saturating_mul(steps_per_epoch))
}

/// `interp(t, [0, w], [a, b])` for `t < w`.
fn interp(t: u64, w: u64, a: f64, b: f64) -> f64 {
    a + (b - a) * (t as f64 / w as f64)
}

impl LrSchedule {
    /// A schedule from the training config: `warmup_iters` is computed from `warmup_epochs` with
    /// [`warmup_iters`].
    pub fn new(
        kind: DecayKind,
        epochs: u64,
        steps_per_epoch: u64,
        lrf: f64,
        warmup_epochs: f64,
        warmup_bias_lr: f64,
        warmup_momentum: f64,
    ) -> Result<Self> {
        Self::from_contract(SchedContract {
            kind,
            unit: SchedUnit::Iteration,
            epochs,
            steps_per_epoch,
            lrf,
            warmup_iters: warmup_iters(warmup_epochs, epochs, steps_per_epoch),
            warmup_bias_lr,
            warmup_momentum,
            step: 0,
        })
    }

    /// A schedule from a `folo.sched` contract (resume). Rejects non-finite values, a length `T` that does not
    /// fit a `u64`, a warmup longer than the run, and a counter past the end.
    pub fn from_contract(c: SchedContract) -> Result<Self> {
        let finite = [c.lrf, c.warmup_bias_lr, c.warmup_momentum].iter().all(|v| v.is_finite());
        if !finite || c.steps_per_epoch == 0 || c.epochs == 0 {
            return Err(Error::Invalid(format!("schedule: invalid parameters {c:?}")));
        }
        let total = c.epochs.checked_mul(c.steps_per_epoch).ok_or_else(|| {
            Error::Invalid(format!("schedule: {} epochs × {} steps overflows", c.epochs, c.steps_per_epoch))
        })?;
        if c.warmup_iters > total {
            return Err(Error::Invalid(format!(
                "schedule: warmup {} is longer than the run ({total})",
                c.warmup_iters
            )));
        }
        if c.step > total {
            return Err(Error::Invalid(format!("schedule: step {} is past the end ({total})", c.step)));
        }
        Ok(LrSchedule { c })
    }

    /// The `folo.sched` contract (with the current step).
    pub fn contract(&self) -> &SchedContract {
        &self.c
    }

    /// `T = epochs · steps_per_epoch` (checked not to overflow when the schedule was built).
    pub fn total_iters(&self) -> u64 {
        self.c.epochs * self.c.steps_per_epoch
    }

    /// `W`.
    pub fn warmup_iters(&self) -> u64 {
        self.c.warmup_iters
    }

    /// The iteration counter `t` (completed optimizer steps).
    pub fn step(&self) -> u64 {
        self.c.step
    }

    /// Count one optimizer step.
    pub fn advance(&mut self) {
        self.c.step += 1;
    }

    /// The decay multiplier `lf(t)`.
    pub fn lf(&self, t: u64) -> f64 {
        let denom = (self.total_iters().saturating_sub(1)).max(1) as f64;
        let lrf = self.c.lrf;
        let x = t as f64 / denom;
        match self.c.kind {
            DecayKind::Linear => (1.0 - x) * (1.0 - lrf) + lrf,
            DecayKind::Cosine => lrf + (1.0 - lrf) * (1.0 + (std::f64::consts::PI * x).cos()) / 2.0,
        }
    }

    /// The LR of a group at iteration `t`: `lr0 · lf(t)`, ramped from `warmup_bias_lr` (the `bias` group) or 0
    /// (every other group) during warmup. `lr0` is the group's base LR.
    pub fn lr_at(&self, t: u64, group: &ParamGroup) -> f64 {
        let target = group.base_lr * self.lf(t);
        if t < self.c.warmup_iters {
            let start = if group.name == BIAS_GROUP { self.c.warmup_bias_lr } else { 0.0 };
            interp(t, self.c.warmup_iters, start, target)
        } else {
            target
        }
    }

    /// The momentum (SGD μ / AdamW β1) of a group at iteration `t`: ramped from `warmup_momentum` to the
    /// group's base value during warmup.
    pub fn momentum_at(&self, t: u64, group: &ParamGroup) -> f64 {
        if t < self.c.warmup_iters {
            interp(t, self.c.warmup_iters, self.c.warmup_momentum, group.base_momentum)
        } else {
            group.base_momentum
        }
    }

    /// Set every group's `lr` and `momentum` for the current iteration.
    pub fn apply(&self, opt: &mut dyn Optimizer) {
        let t = self.c.step;
        for g in opt.groups_mut() {
            g.lr = self.lr_at(t, g);
            g.momentum = self.momentum_at(t, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_core::Tensor;

    #[test]
    fn docs_model_10_4() {
        // N_train = 800, batch 16, E = 10: nb = 50, accumulate 4, spe = 13, T = 130, W = 100.
        let nb = 800usize.div_ceil(16);
        let acc = crate::optim::accumulate_steps(64, 16);
        let spe = nb.div_ceil(acc) as u64;
        assert_eq!((nb, acc, spe), (50, 4, 13));
        let s = LrSchedule::new(DecayKind::Linear, 10, spe, 0.01, 3.0, 0.1, 0.8).unwrap();
        assert_eq!((s.total_iters(), s.warmup_iters()), (130, 100));
        assert_eq!(warmup_iters(0.0, 10, 13), 0);
        assert_eq!(warmup_iters(3.0, 1, 13), 0); // epochs − 1 = 0
        let lr0 = 0.01;
        let mk = |name: &str| {
            ParamGroup::new(name, vec![("w".into(), Tensor::zeros([1]))]).lr(lr0).momentum(0.937)
        };
        let (decay, bias) = (mk("decay"), mk("bias"));
        let lf = |t: f64| (1.0 - t / 129.0) * (1.0 - 0.01) + 0.01;
        for t in [0u64, 50, 99, 100, 129] {
            let target = lr0 * lf(t as f64);
            let (want_d, want_b, want_m) = if t < 100 {
                let f = t as f64 / 100.0;
                (target * f, 0.1 + (target - 0.1) * f, 0.8 + (0.937 - 0.8) * f)
            } else {
                (target, target, 0.937)
            };
            assert!((s.lr_at(t, &decay) - want_d).abs() < 1e-15, "t={t}");
            assert!((s.lr_at(t, &bias) - want_b).abs() < 1e-15, "t={t}");
            assert!((s.momentum_at(t, &decay) - want_m).abs() < 1e-15, "t={t}");
        }
        assert_eq!(s.lr_at(0, &decay), 0.0);
        assert_eq!(s.lr_at(0, &bias), 0.1);
        assert_eq!(s.lf(0), 1.0);
        assert!((s.lf(129) - 0.01).abs() < 1e-15);

        let c = LrSchedule::new(DecayKind::Cosine, 10, 13, 0.01, 0.0, 0.0, 0.8).unwrap();
        assert_eq!(c.lf(0), 1.0);
        assert!((c.lf(129) - 0.01).abs() < 1e-15);
        assert!(
            (c.lf(64) - (0.01 + 0.99 * (1.0 + (std::f64::consts::PI * 64.0 / 129.0).cos()) / 2.0)).abs()
                < 1e-15
        );
        assert_eq!(c.lr_at(0, &decay), lr0); // no warmup
        // T = 1: the single step runs at lr0.
        let one = LrSchedule::new(DecayKind::Linear, 1, 1, 0.01, 3.0, 0.1, 0.8).unwrap();
        assert_eq!((one.warmup_iters(), one.lr_at(0, &decay)), (0, lr0));
    }

    #[test]
    fn malformed_contracts_are_errors() {
        let good =
            LrSchedule::new(DecayKind::Linear, 10, 13, 0.01, 3.0, 0.1, 0.8).unwrap().contract().clone();
        let bad = |f: &dyn Fn(&mut SchedContract)| {
            let mut c = good.clone();
            f(&mut c);
            LrSchedule::from_contract(c).unwrap_err().to_string()
        };
        assert!(bad(&|c| (c.epochs, c.steps_per_epoch) = (u64::MAX, 2)).contains("overflows"));
        assert!(bad(&|c| c.warmup_iters = 131).contains("warmup"));
        assert!(bad(&|c| c.step = 131).contains("past the end"));
        assert!(bad(&|c| c.steps_per_epoch = 0).contains("invalid"));
        assert!(bad(&|c| c.lrf = f64::NAN).contains("invalid"));
        // The config path saturates instead of overflowing; from_contract then reports the overflow.
        assert_eq!(warmup_iters(3.0, u64::MAX, u64::MAX), u64::MAX); // no overflow panic
        assert!(LrSchedule::new(DecayKind::Linear, u64::MAX, 2, 0.01, 3.0, 0.1, 0.8).is_err());
    }
}
