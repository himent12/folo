//! The LR / momentum schedule of docs/MODEL.md §8.4: linear warmup, then linear, cosine,
//! Warmup–Stable–Decay (WSD), cosine-with-restarts (SGDR), polynomial decay, or OneCycleLR,
//! counted in optimizer steps (`folo.sched` in docs/FORMAT.md).

use serde::{Deserialize, Serialize};

use crate::optim::{Optimizer, ParamGroup, is_bias_group};
use crate::{Error, Result};

/// Default fraction of the run spent in the decay phase of [`DecayKind::Wsd`] (last 20%).
pub const WSD_DEFAULT_DECAY_FRACTION: f64 = 0.2;
/// Default number of equal cosine cycles for [`DecayKind::CosineRestarts`] when `restart_period` is omitted.
pub const DEFAULT_RESTART_CYCLES: u64 = 3;
/// Default cycle-length multiplier (`T_mult`) for [`DecayKind::CosineRestarts`].
pub const DEFAULT_RESTART_MULT: u64 = 1;
/// Default exponent `p` for [`DecayKind::Polynomial`] (`1.0`, identical to linear decay).
pub const DEFAULT_POLY_POWER: f64 = 1.0;
/// Default rising-phase fraction for [`DecayKind::OneCycle`] (Smith & Topin 2019: `0.3`).
pub const DEFAULT_ONE_CYCLE_PCT_START: f64 = 0.3;
/// Default initial LR divisor for [`DecayKind::OneCycle`] (Smith & Topin 2019: `25.0`, starting at `lr / 25`).
pub const DEFAULT_ONE_CYCLE_DIV_FACTOR: f64 = 25.0;

/// The decay shape after warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecayKind {
    /// `lf(t) = (1 − t / max(T − 1, 1)) · (1 − lrf) + lrf` (the default).
    Linear,
    /// `lf(t) = lrf + (1 − lrf) · (1 + cos(π · t / max(T − 1, 1))) / 2` (`cos_lr`).
    Cosine,
    /// Warmup–Stable–Decay (Hu et al. 2024): holds `lf(t) = 1.0` through the stable plateau and decays over
    /// the final `decay_fraction` (default [`WSD_DEFAULT_DECAY_FRACTION`]) down to `lrf` with [`WsdTail`].
    Wsd,
    /// Cosine annealing with warm restarts (SGDR, Loshchilov & Hutter 2017): after warmup, repeats cosine
    /// cycles starting at `restart_period` iterations and multiplying cycle length by `restart_mult` each
    /// restart (defaults: [`DEFAULT_RESTART_CYCLES`] equal cycles, `restart_mult = 1`).
    CosineRestarts,
    /// Power-law polynomial decay: `lf(t) = lrf + (1 − lrf) · (1 − x)^p` with `p > 0`
    /// (default [`DEFAULT_POLY_POWER`] = `1.0`, matching [`DecayKind::Linear`]).
    Polynomial,
    /// OneCycleLR (Smith & Topin 2019): cosine rise from `1 / div_factor` to peak `1.0` over the first
    /// `pct_start` of the run, then cosine anneal down to `lrf` in `[0, 1]`, paired with inverse cosine
    /// momentum cycling between `base_momentum` (at endpoints) and `warmup_momentum` (at peak).
    ///
    /// Manages its own warmup via `pct_start`/`div_factor`, so `warmup_iters` must be `0`. Note: when
    /// cycling `momentum` with [`crate::optim::AdamW`], FOLO's `AdamW` (like PyTorch's) uses `1 − β1(t)^t`
    /// bias correction rather than `1 − ∏ β1(i)`; set `warmup_momentum == base_momentum` to keep `β1` fixed
    /// for exact AdamW bias correction, or pair momentum cycling with `Sgd` / `MuSgd`.
    OneCycle,
}

/// Decay tail shape for [`DecayKind::Wsd`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WsdTail {
    /// Linear decay from `1.0` to `lrf` (default).
    #[default]
    Linear,
    /// Half-cosine decay from `1.0` to `lrf`.
    Cosine,
    /// `1 − sqrt(u)` decay from `1.0` to `lrf` (Hägele et al. 2024).
    Sqrt,
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
    /// For [`DecayKind::Wsd`]: fraction of the run spent decaying, in `(0, 1]` (default `0.2`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decay_fraction: Option<f64>,
    /// For [`DecayKind::Wsd`]: shape of the decay tail (default [`WsdTail::Linear`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wsd_tail: Option<WsdTail>,
    /// For [`DecayKind::CosineRestarts`]: length `T_0 >= 1` of the first post-warmup cycle in iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart_period: Option<u64>,
    /// For [`DecayKind::CosineRestarts`]: integer cycle-length multiplier `T_mult >= 1` (default `1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart_mult: Option<u64>,
    /// For [`DecayKind::Polynomial`]: decay exponent `p > 0` (default [`DEFAULT_POLY_POWER`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub poly_power: Option<f64>,
    /// For [`DecayKind::OneCycle`]: rising-phase fraction in `(0, 1)` (default [`DEFAULT_ONE_CYCLE_PCT_START`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_cycle_pct_start: Option<f64>,
    /// For [`DecayKind::OneCycle`]: initial LR divisor `>= 1` (default [`DEFAULT_ONE_CYCLE_DIV_FACTOR`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_cycle_div_factor: Option<f64>,
}

/// The schedule state: its parameters and the iteration counter `t`.
///
/// Each optimizer step: [`LrSchedule::apply`] (sets every group's `lr` and `momentum` for iteration `t`),
/// then `optimizer.step()`, then [`LrSchedule::advance`].
///
/// ```
/// use folo_nn::{DecayKind, LrSchedule};
/// // docs/MODEL.md §10.4: 30 epochs of 13 steps, 3 warmup epochs (floored to 100 iterations).
/// let s = LrSchedule::new(DecayKind::Linear, 30, 13, 0.01, 3.0, 0.1, 0.8).unwrap();
/// assert_eq!((s.total_iters(), s.warmup_iters()), (390, 100));
/// assert_eq!(s.lf(0), 1.0);
/// assert!((s.lf(389) - 0.01).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LrSchedule {
    c: SchedContract,
}

/// The warmup floor of docs/MODEL.md §8.4 in iterations, before it is capped to
/// [`WARMUP_FLOOR_TENTHS`] of the run.
pub const WARMUP_FLOOR: u64 = 100;

/// The largest share of the run, in tenths, that [`WARMUP_FLOOR`] may stretch the warmup to (3: 30%).
const WARMUP_FLOOR_TENTHS: u64 = 3;

/// docs/MODEL.md §8.4: `W = 0` if `warmup_epochs <= 0`, else
/// `min(max(round(warmup_epochs · spe), min(100, floor(0.3 · T))), max(epochs − 1, 0) · spe)` with
/// `T = epochs · spe`. The floor applies in full from `T = 334` on, so a short run is not all warmup.
pub fn warmup_iters(warmup_epochs: f64, epochs: u64, steps_per_epoch: u64) -> u64 {
    warmup_iters_floor(warmup_epochs, epochs, steps_per_epoch, WARMUP_FLOOR)
}

/// [`warmup_iters`] with `floor` iterations in place of [`WARMUP_FLOOR`] (still capped to 30% of the run);
/// `floor = 0` is plain `round(warmup_epochs · spe)`, capped at `(epochs − 1) · spe`, as Ultralytics 8.4 counts it.
pub fn warmup_iters_floor(warmup_epochs: f64, epochs: u64, steps_per_epoch: u64, floor: u64) -> u64 {
    if warmup_epochs.is_nan() || warmup_epochs <= 0.0 {
        return 0;
    }
    let total = epochs.saturating_mul(steps_per_epoch);
    let floor = floor.min(total.saturating_mul(WARMUP_FLOOR_TENTHS) / 10);
    let w = ((warmup_epochs * steps_per_epoch as f64).round() as u64).max(floor);
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
        let w = warmup_iters(warmup_epochs, epochs, steps_per_epoch);
        Self::with_warmup_iters(kind, epochs, steps_per_epoch, lrf, w, warmup_bias_lr, warmup_momentum)
    }

    /// [`LrSchedule::new`] with a warmup of exactly `warmup_iters` iterations (e.g. from [`warmup_iters_floor`]).
    pub fn with_warmup_iters(
        kind: DecayKind,
        epochs: u64,
        steps_per_epoch: u64,
        lrf: f64,
        warmup_iters: u64,
        warmup_bias_lr: f64,
        warmup_momentum: f64,
    ) -> Result<Self> {
        Self::from_contract(SchedContract {
            kind,
            unit: SchedUnit::Iteration,
            epochs,
            steps_per_epoch,
            lrf,
            warmup_iters,
            warmup_bias_lr,
            warmup_momentum,
            step: 0,
            decay_fraction: None,
            wsd_tail: None,
            restart_period: None,
            restart_mult: None,
            poly_power: None,
            one_cycle_pct_start: None,
            one_cycle_div_factor: None,
        })
    }

    /// Configure the decay fraction `(0, 1]` and tail shape on a [`DecayKind::Wsd`] schedule.
    ///
    /// Rejects non-WSD schedules or out-of-range `decay_fraction`.
    pub fn with_wsd(mut self, decay_fraction: f64, tail: WsdTail) -> Result<Self> {
        self.c.decay_fraction = Some(decay_fraction);
        self.c.wsd_tail = Some(tail);
        Self::from_contract(self.c)
    }

    /// Configure the first post-warmup cycle length (`period >= 1` iterations) and integer cycle-length
    /// multiplier (`mult >= 1`) on a [`DecayKind::CosineRestarts`] schedule.
    ///
    /// Rejects non-`CosineRestarts` schedules or zero `period` / `mult`.
    pub fn with_restarts(mut self, period: u64, mult: u64) -> Result<Self> {
        self.c.restart_period = Some(period);
        self.c.restart_mult = Some(mult);
        Self::from_contract(self.c)
    }

    /// Configure the power exponent (`power > 0`, finite) on a [`DecayKind::Polynomial`] schedule.
    ///
    /// Rejects non-`Polynomial` schedules or invalid `power`.
    pub fn with_poly_power(mut self, power: f64) -> Result<Self> {
        self.c.poly_power = Some(power);
        Self::from_contract(self.c)
    }

    /// Configure the rising fraction (`pct_start` in `(0, 1)`) and initial divisor (`div_factor >= 1`)
    /// on a [`DecayKind::OneCycle`] schedule.
    ///
    /// Rejects non-`OneCycle` schedules or out-of-range parameters.
    pub fn with_one_cycle(mut self, pct_start: f64, div_factor: f64) -> Result<Self> {
        self.c.one_cycle_pct_start = Some(pct_start);
        self.c.one_cycle_div_factor = Some(div_factor);
        Self::from_contract(self.c)
    }

    /// A schedule from a `folo.sched` contract (resume). Rejects non-finite values, a length `T` that does not
    /// fit a `u64`, a warmup longer than the run (or non-zero warmup on [`DecayKind::OneCycle`]), a counter
    /// past the end, and invalid or mismatched optional WSD / restart / polynomial / OneCycle parameters.
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
        if (c.decay_fraction.is_some() || c.wsd_tail.is_some()) && c.kind != DecayKind::Wsd {
            return Err(Error::Invalid(format!(
                "schedule: decay_fraction / wsd_tail are only valid for wsd, got {:?}",
                c.kind
            )));
        }
        if let Some(df) = c.decay_fraction
            && (!df.is_finite() || df <= 0.0 || df > 1.0)
        {
            return Err(Error::Invalid(format!("schedule: decay_fraction must be in (0, 1], got {df}")));
        }
        if (c.restart_period.is_some() || c.restart_mult.is_some()) && c.kind != DecayKind::CosineRestarts {
            return Err(Error::Invalid(format!(
                "schedule: restart_period / restart_mult are only valid for cosine_restarts, got {:?}",
                c.kind
            )));
        }
        if let Some(p) = c.restart_period
            && p == 0
        {
            return Err(Error::Invalid("schedule: restart_period must be >= 1".into()));
        }
        if let Some(m) = c.restart_mult
            && m == 0
        {
            return Err(Error::Invalid("schedule: restart_mult must be >= 1".into()));
        }
        if c.poly_power.is_some() && c.kind != DecayKind::Polynomial {
            return Err(Error::Invalid(format!(
                "schedule: poly_power is only valid for polynomial, got {:?}",
                c.kind
            )));
        }
        if let Some(p) = c.poly_power
            && (!p.is_finite() || p <= 0.0)
        {
            return Err(Error::Invalid(format!("schedule: poly_power must be finite and > 0, got {p}")));
        }
        if c.kind == DecayKind::OneCycle {
            if c.warmup_iters > 0 {
                return Err(Error::Invalid(format!(
                    "schedule: one_cycle manages its own warmup via pct_start/div_factor; warmup_iters must be 0, got {}",
                    c.warmup_iters
                )));
            }
            if !(0.0..=1.0).contains(&c.lrf) {
                return Err(Error::Invalid(format!(
                    "schedule: one_cycle lrf must be in 0..=1, got {}",
                    c.lrf
                )));
            }
        }
        if (c.one_cycle_pct_start.is_some() || c.one_cycle_div_factor.is_some())
            && c.kind != DecayKind::OneCycle
        {
            return Err(Error::Invalid(format!(
                "schedule: one_cycle_* fields are only valid for one_cycle, got {:?}",
                c.kind
            )));
        }
        if let Some(pct) = c.one_cycle_pct_start
            && (!pct.is_finite() || pct <= 0.0 || pct >= 1.0)
        {
            return Err(Error::Invalid(format!(
                "schedule: one_cycle_pct_start must be finite and in (0, 1), got {pct}"
            )));
        }
        if let Some(div) = c.one_cycle_div_factor
            && (!div.is_finite() || div < 1.0)
        {
            return Err(Error::Invalid(format!(
                "schedule: one_cycle_div_factor must be finite and >= 1, got {div}"
            )));
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

    /// The decay multiplier `lf(t)`; from `t = T - 1` on it stays clamped at the final step (`lrf` for
    /// `Linear` / `Cosine` / `Wsd` / `Polynomial` / `OneCycle`, and the `T - 1` phase for `CosineRestarts`;
    /// `from_contract` accepts a finished run's `t = T`).
    pub fn lf(&self, t: u64) -> f64 {
        let total = self.total_iters();
        let denom = total.saturating_sub(1).max(1) as f64;
        let lrf = self.c.lrf;
        // `t ≤ T - 1` gives `x ≤ 1` exactly, so the clamp changes nothing a run steps through.
        let x = (t as f64 / denom).min(1.0);
        match self.c.kind {
            DecayKind::Linear => (1.0 - x) * (1.0 - lrf) + lrf,
            DecayKind::Cosine => lrf + (1.0 - lrf) * (1.0 + (std::f64::consts::PI * x).cos()) / 2.0,
            DecayKind::Wsd => {
                let w = self.c.warmup_iters;
                let last = total.saturating_sub(1);
                let tc = t.min(last);
                if tc < w || total <= 1 {
                    return if t >= total && total <= 1 { lrf } else { 1.0 };
                }
                if t >= last {
                    return lrf;
                }
                let df = self.c.decay_fraction.unwrap_or(WSD_DEFAULT_DECAY_FRACTION);
                let last_f = last as f64;
                let decay_start = (w as f64).max(last_f * (1.0 - df));
                let tc_f = tc as f64;
                if tc_f <= decay_start || last_f <= decay_start {
                    return 1.0;
                }
                let u = ((tc_f - decay_start) / (last_f - decay_start)).clamp(0.0, 1.0);
                let keep = match self.c.wsd_tail.unwrap_or_default() {
                    WsdTail::Linear => 1.0 - u,
                    WsdTail::Cosine => 0.5 * (1.0 + (std::f64::consts::PI * u).cos()),
                    WsdTail::Sqrt => 1.0 - u.sqrt(),
                };
                keep + (1.0 - keep) * lrf
            }
            DecayKind::CosineRestarts => {
                let w = self.c.warmup_iters;
                let tc = t.min(total.saturating_sub(1));
                if tc < w {
                    return 1.0;
                }
                let period = self.c.restart_period.unwrap_or_else(|| {
                    total.saturating_sub(w).max(1).div_ceil(DEFAULT_RESTART_CYCLES).max(1)
                });
                let mult = self.c.restart_mult.unwrap_or(DEFAULT_RESTART_MULT);
                let mut rem = (tc - w) as u128;
                let len: u128 = if mult == 1 {
                    rem %= period as u128;
                    period as u128
                } else {
                    let mut cur = period as u128;
                    while rem >= cur {
                        rem -= cur;
                        cur *= mult as u128;
                    }
                    cur
                };
                let u = if len <= 1 { 0.0 } else { rem as f64 / (len - 1) as f64 };
                let keep = 0.5 * (1.0 + (std::f64::consts::PI * u).cos());
                keep + (1.0 - keep) * lrf
            }
            DecayKind::Polynomial => {
                let p = self.c.poly_power.unwrap_or(DEFAULT_POLY_POWER);
                let keep = (1.0 - x).powf(p);
                keep + (1.0 - keep) * lrf
            }
            DecayKind::OneCycle => {
                if total <= 1 {
                    return 1.0;
                }
                let pct = self.c.one_cycle_pct_start.unwrap_or(DEFAULT_ONE_CYCLE_PCT_START);
                let div = self.c.one_cycle_div_factor.unwrap_or(DEFAULT_ONE_CYCLE_DIV_FACTOR);
                if x <= pct {
                    let u = x / pct;
                    let init_f = 1.0 / div;
                    let rise = 0.5 * (1.0 - (std::f64::consts::PI * u).cos());
                    rise + (1.0 - rise) * init_f
                } else {
                    let v = ((x - pct) / (1.0 - pct)).min(1.0);
                    let keep = 0.5 * (1.0 + (std::f64::consts::PI * v).cos());
                    keep + (1.0 - keep) * lrf
                }
            }
        }
    }

    /// Effective LR multiplier `lr(t) / base_lr` for non-bias parameter groups: matches [`Self::lf`] after
    /// warmup, and includes the linear `0 -> lf(t)` warmup ramp while `t < warmup_iters`. Used by
    /// LR-phase-coupled EMA (`PC-PEMA`).
    pub fn effective_lf(&self, t: u64) -> f64 {
        let f = self.lf(t);
        if t < self.c.warmup_iters { interp(t, self.c.warmup_iters, 0.0, f) } else { f }
    }

    /// The LR of a group at iteration `t`: `lr0 · lf(t)`, ramped from `warmup_bias_lr` (the `bias` group) or 0
    /// (every other group) during warmup. `lr0` is the group's base LR.
    pub fn lr_at(&self, t: u64, group: &ParamGroup) -> f64 {
        let target = group.base_lr * self.lf(t);
        if t < self.c.warmup_iters {
            let start = if is_bias_group(&group.name) { self.c.warmup_bias_lr } else { 0.0 };
            interp(t, self.c.warmup_iters, start, target)
        } else {
            target
        }
    }

    /// The momentum (SGD μ / AdamW β1) of a group at iteration `t`: ramped from `warmup_momentum` to the
    /// group's base value during warmup, or — for [`DecayKind::OneCycle`] — inversely cosine-cycled from
    /// `base_momentum` down to `warmup_momentum` at the LR peak and back to `base_momentum`.
    pub fn momentum_at(&self, t: u64, group: &ParamGroup) -> f64 {
        if self.c.kind == DecayKind::OneCycle {
            let total = self.total_iters();
            if total <= 1 {
                return group.base_momentum;
            }
            let x = (t as f64 / (total - 1) as f64).min(1.0);
            let pct = self.c.one_cycle_pct_start.unwrap_or(DEFAULT_ONE_CYCLE_PCT_START);
            let (m_max, m_min) = (group.base_momentum, self.c.warmup_momentum);
            return if x <= pct {
                let u = x / pct;
                let frac = 0.5 * (1.0 - (std::f64::consts::PI * u).cos());
                m_max + (m_min - m_max) * frac
            } else {
                let v = ((x - pct) / (1.0 - pct)).min(1.0);
                let frac = 0.5 * (1.0 - (std::f64::consts::PI * v).cos());
                m_min + (m_max - m_min) * frac
            };
        }
        if t < self.c.warmup_iters {
            interp(t, self.c.warmup_iters, self.c.warmup_momentum, group.base_momentum)
        } else {
            group.base_momentum
        }
    }

    /// Set every group's `lr` and `momentum` for the current iteration.
    pub fn apply(&self, opt: &mut dyn Optimizer) {
        self.apply_to_groups(opt.groups_mut());
    }

    /// Set every [`ParamGroup`]'s `lr` and `momentum` for the current iteration.
    pub fn apply_to_groups(&self, groups: &mut [ParamGroup]) {
        let t = self.c.step;
        for g in groups {
            g.lr = self.lr_at(t, g);
            g.momentum = self.momentum_at(t, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_core::Tensor;

    /// `from_contract` accepts `step == T` (a finished run's counter): the factor there is the schedule's end value
    /// `lrf`, never below it (linear with T = 2 went to -0.98) nor back up (cosine past π).
    #[test]
    fn the_factor_ends_at_lrf() {
        let lrf = 0.01;
        for kind in [DecayKind::Linear, DecayKind::Cosine, DecayKind::Wsd] {
            for (epochs, spe) in [(2, 1), (1, 3), (10, 13)] {
                let s = LrSchedule::new(kind, epochs, spe, lrf, 0.0, 0.0, 0.8).unwrap();
                let total = s.total_iters();
                assert!((s.lf(total - 1) - lrf).abs() < 1e-15, "{kind:?} T={total}");
                assert_eq!(s.lf(total), s.lf(total - 1), "{kind:?} T={total}: lf(T)");
                let mut c = s.contract().clone();
                c.step = total;
                let end = LrSchedule::from_contract(c).unwrap();
                let group =
                    ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([1]))]).lr(0.01).momentum(0.9);
                assert!(end.lr_at(total, &group) >= 0.01 * lrf - 1e-18, "{kind:?} T={total}");
            }
        }
    }

    #[test]
    fn docs_model_10_4() {
        // N_train = 800, batch 16, E = 30: nb = 50, accumulate 4, spe = 13, T = 390, W = 100.
        let nb = 800usize.div_ceil(16);
        let acc = crate::optim::accumulate_steps(64, 16);
        let spe = nb.div_ceil(acc) as u64;
        assert_eq!((nb, acc, spe), (50, 4, 13));
        let s = LrSchedule::new(DecayKind::Linear, 30, spe, 0.01, 3.0, 0.1, 0.8).unwrap();
        assert_eq!((s.total_iters(), s.warmup_iters()), (390, 100));
        assert_eq!(warmup_iters(0.0, 30, 13), 0);
        assert_eq!(warmup_iters(3.0, 1, 13), 0); // epochs − 1 = 0
        // Short runs: the floor shrinks to 30% of T, the epoch-based warmup stays.
        assert_eq!(warmup_iters(3.0, 10, 13), 39); // T = 130: floor 39, 3 epochs = 39
        assert_eq!(warmup_iters(3.0, 12, 3), 10); // T = 36: floor 10 (was 33 of 36)
        assert_eq!(warmup_iters(0.5, 37, 9), 99); // T = 333: floor 99
        assert_eq!(warmup_iters(0.5, 334, 1), 100); // T = 334: the full floor
        assert_eq!(warmup_iters(3.0, 4, 3), 9); // 3 epochs of 4, capped at epochs − 1
        assert_eq!(warmup_iters_floor(3.0, 100, 4, 0), 12); // a 205-image fine-tune: 3 epochs, no floor
        assert_eq!(warmup_iters_floor(3.0, 100, 4, 100), warmup_iters(3.0, 100, 4)); // 100 = the §8.4 floor
        assert_eq!(warmup_iters_floor(0.0, 100, 4, 0), 0);
        let lr0 = 0.01;
        let mk = |name: &str| {
            ParamGroup::new(name, vec![("w".into(), Tensor::zeros([1]))]).lr(lr0).momentum(0.937)
        };
        let (decay, bias) = (mk("decay"), mk("bias"));
        let lf = |t: f64| (1.0 - t / 389.0) * (1.0 - 0.01) + 0.01;
        for t in [0u64, 50, 99, 100, 389] {
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
        assert!((s.lf(389) - 0.01).abs() < 1e-15);

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

        // WSD / restart contract validation.
        assert!(bad(&|c| c.decay_fraction = Some(0.2)).contains("wsd"));
        assert!(bad(&|c| c.wsd_tail = Some(WsdTail::Cosine)).contains("wsd"));
        assert!(bad(&|c| c.restart_period = Some(10)).contains("cosine_restarts"));
        assert!(bad(&|c| c.restart_mult = Some(2)).contains("cosine_restarts"));
        let wsd = LrSchedule::new(DecayKind::Wsd, 10, 10, 0.01, 0.0, 0.0, 0.8).unwrap();
        assert!(wsd.clone().with_wsd(0.0, WsdTail::Linear).is_err());
        assert!(wsd.clone().with_wsd(-0.1, WsdTail::Linear).is_err());
        assert!(wsd.clone().with_wsd(1.1, WsdTail::Linear).is_err());
        assert!(wsd.clone().with_wsd(f64::NAN, WsdTail::Linear).is_err());
        assert!(wsd.clone().with_wsd(f64::INFINITY, WsdTail::Linear).is_err());
        assert!(wsd.with_restarts(10, 2).is_err());
        let sgdr = LrSchedule::new(DecayKind::CosineRestarts, 10, 10, 0.01, 0.0, 0.0, 0.8).unwrap();
        assert!(sgdr.clone().with_restarts(0, 1).is_err());
        assert!(sgdr.clone().with_restarts(10, 0).is_err());
        assert!(sgdr.with_wsd(0.2, WsdTail::Linear).is_err());
    }

    #[test]
    fn wsd_plateau_tails_and_contract_round_trip() {
        let lrf = 0.01;
        // T = 101 so x = t / 100 is exact; default decay_fraction = 0.2 -> stable for t in 0..=80, decay 80..=100.
        let s = LrSchedule::with_warmup_iters(DecayKind::Wsd, 101, 1, lrf, 10, 0.1, 0.8).unwrap();
        let decay_g = ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([1]))]).lr(0.01).momentum(0.9);
        assert_eq!(s.lr_at(0, &decay_g), 0.0);
        assert!((s.lr_at(5, &decay_g) - 0.005).abs() < 1e-15);
        for t in 10..=80 {
            assert_eq!(s.lf(t), 1.0, "stable plateau at t={t}");
            assert!((s.lr_at(t, &decay_g) - 0.01).abs() < 1e-15, "stable lr at t={t}");
        }
        // Linear midpoint t = 90 -> u = 0.5 -> 0.505.
        assert!((s.lf(90) - 0.505).abs() < 1e-14);
        assert!((s.lf(100) - lrf).abs() < 1e-15);
        assert_eq!(s.lf(101), lrf);

        // Cosine and Sqrt tails match analytical formulas and stay monotonic.
        for tail in [WsdTail::Linear, WsdTail::Cosine, WsdTail::Sqrt] {
            let st = s.clone().with_wsd(0.2, tail).unwrap();
            let mut prev = st.lf(0);
            for t in 1..=101 {
                let cur = st.lf(t);
                assert!(cur <= prev + 1e-15, "{tail:?} non-monotonic at t={t}: {cur} > {prev}");
                prev = cur;
            }
            assert_eq!(st.lf(80), 1.0);
            assert!((st.lf(100) - lrf).abs() < 1e-15);
            if tail == WsdTail::Cosine {
                assert!((st.lf(90) - (lrf + (1.0 - lrf) * 0.5)).abs() < 1e-14);
            } else if tail == WsdTail::Sqrt {
                let want = lrf + (1.0 - lrf) * (1.0 - 0.5f64.sqrt());
                assert!((st.lf(90) - want).abs() < 1e-14);
            }
        }

        // decay_fraction = 1.0 matches Linear / Cosine exactly.
        let lin = LrSchedule::new(DecayKind::Linear, 10, 13, lrf, 0.0, 0.0, 0.8).unwrap();
        let wsd_lin = LrSchedule::new(DecayKind::Wsd, 10, 13, lrf, 0.0, 0.0, 0.8)
            .unwrap()
            .with_wsd(1.0, WsdTail::Linear)
            .unwrap();
        let cos = LrSchedule::new(DecayKind::Cosine, 10, 13, lrf, 0.0, 0.0, 0.8).unwrap();
        let wsd_cos = LrSchedule::new(DecayKind::Wsd, 10, 13, lrf, 0.0, 0.0, 0.8)
            .unwrap()
            .with_wsd(1.0, WsdTail::Cosine)
            .unwrap();
        for t in 0..=130 {
            assert!((wsd_lin.lf(t) - lin.lf(t)).abs() < 1e-15, "t={t}");
            assert!((wsd_cos.lf(t) - cos.lf(t)).abs() < 1e-15, "t={t}");
        }

        // Tiny decay_fraction still lands cleanly on lrf at T - 1, and T = 1 stays at 1.0.
        let tiny = LrSchedule::with_warmup_iters(DecayKind::Wsd, 10, 1, lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_wsd(1e-17, WsdTail::Linear)
            .unwrap();
        assert_eq!(tiny.lf(8), 1.0);
        assert_eq!(tiny.lf(9), lrf);
        let one = LrSchedule::with_warmup_iters(DecayKind::Wsd, 1, 1, lrf, 0, 0.0, 0.8).unwrap();
        assert_eq!(one.lf(0), 1.0);

        // Warmup extending past 1 - decay_fraction never decays during warmup; all-warmup stays at 1.0.
        let long_w = LrSchedule::with_warmup_iters(DecayKind::Wsd, 101, 1, lrf, 90, 0.0, 0.8).unwrap();
        assert_eq!(long_w.lf(85), 1.0);
        assert_eq!(long_w.lf(90), 1.0);
        assert!((long_w.lf(95) - 0.505).abs() < 1e-14);
        assert!((long_w.lf(100) - lrf).abs() < 1e-15);
        let all_w = LrSchedule::with_warmup_iters(DecayKind::Wsd, 10, 1, lrf, 10, 0.0, 0.8).unwrap();
        assert_eq!(all_w.lf(9), 1.0);
        assert_eq!(all_w.lf(10), 1.0);

        // Literal legacy folo.sched JSON (without the 4 optional keys) deserializes cleanly.
        let legacy = r#"{"type":"linear","unit":"iteration","epochs":10,"steps_per_epoch":13,"lrf":0.01,"warmup_iters":0,"warmup_bias_lr":0.0,"warmup_momentum":0.8,"step":0}"#;
        let legacy_parsed: SchedContract = serde_json::from_str(legacy).unwrap();
        assert_eq!(LrSchedule::from_contract(legacy_parsed).unwrap(), lin);

        // Linear / Cosine omit new optional keys from JSON; Wsd round-trips them.
        let lin_json = serde_json::to_string(lin.contract()).unwrap();
        assert!(!lin_json.contains("decay_fraction") && !lin_json.contains("restart_period"));
        let wsd_json = serde_json::to_string(wsd_cos.contract()).unwrap();
        assert!(wsd_json.contains(r#""type":"wsd""#) && wsd_json.contains(r#""wsd_tail":"cosine""#));
        let parsed: SchedContract = serde_json::from_str(&wsd_json).unwrap();
        assert_eq!(LrSchedule::from_contract(parsed).unwrap(), wsd_cos);
    }

    #[test]
    fn cosine_restarts_cycles_mult_warmup_and_json() {
        let lrf = 0.01;
        // Single full cycle with no warmup matches plain Cosine identically.
        let cos = LrSchedule::with_warmup_iters(DecayKind::Cosine, 10, 10, lrf, 0, 0.0, 0.8).unwrap();
        let sgdr1 = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 10, 10, lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_restarts(100, 1)
            .unwrap();
        for t in 0..=100 {
            assert!((sgdr1.lf(t) - cos.lf(t)).abs() < 1e-15, "single cycle t={t}");
        }

        // Default restart_period (None) splits post-warmup into DEFAULT_RESTART_CYCLES (3) equal cycles.
        let def = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 9, 10, lrf, 0, 0.0, 0.8).unwrap();
        for start in [0u64, 30, 60] {
            assert_eq!(def.lf(start), 1.0);
            assert!((def.lf(start + 29) - lrf).abs() < 1e-15);
        }

        // period == 1 stays finite for mult == 1 and grows cleanly for mult == 2; lrf = 0, 1, > 1, f64::MAX stay finite.
        let p1 = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 10, 1, 2.0, 0, 0.0, 0.8)
            .unwrap()
            .with_restarts(1, 2)
            .unwrap();
        assert_eq!(p1.lf(0), 1.0);
        assert_eq!(p1.lf(1), 1.0);
        assert!((p1.lf(2) - 2.0).abs() < 1e-15);
        let max_lrf = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 10, 1, f64::MAX, 0, 0.0, 0.8)
            .unwrap()
            .with_restarts(5, 1)
            .unwrap();
        assert_eq!(max_lrf.lf(0), 1.0);
        assert_eq!(max_lrf.lf(4), f64::MAX);

        // Warmup W = 10, equal cycles of 30 across T = 100: restarts at t = 10, 40, 70; bottoms at 39, 69, 99.
        let eq = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 10, 10, lrf, 10, 0.1, 0.8)
            .unwrap()
            .with_restarts(30, 1)
            .unwrap();
        for t in 0..10 {
            assert_eq!(eq.lf(t), 1.0, "warmup t={t}");
        }
        for start in [10u64, 40, 70] {
            assert_eq!(eq.lf(start), 1.0, "cycle start t={start}");
            assert!((eq.lf(start + 29) - lrf).abs() < 1e-15, "cycle end t={}", start + 29);
        }
        assert_eq!(eq.lf(100), eq.lf(99));

        // Geometric cycles T_0 = 10, T_mult = 2 -> lengths 10, 20, 40 (total 70).
        let geo = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 7, 10, lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_restarts(10, 2)
            .unwrap();
        assert_eq!(geo.lf(0), 1.0);
        assert!((geo.lf(9) - lrf).abs() < 1e-15);
        assert_eq!(geo.lf(10), 1.0);
        assert!((geo.lf(29) - lrf).abs() < 1e-15);
        assert_eq!(geo.lf(30), 1.0);
        assert!((geo.lf(69) - lrf).abs() < 1e-15);
        assert_eq!(geo.lf(70), geo.lf(69));

        // Huge step count with T_mult = 2 finishes in <= 64 iterations without overflow.
        let huge = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 1 << 40, 1, lrf, 0, 0.0, 0.8)
            .unwrap()
            .with_restarts(10, 2)
            .unwrap();
        let v = huge.lf((1 << 40) - 1);
        assert!(v.is_finite() && (lrf..=1.0).contains(&v));

        // Edge case: warmup == total and T == 1 stay finite and at 1.0.
        let all_w = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 5, 1, lrf, 5, 0.0, 0.8).unwrap();
        assert_eq!(all_w.lf(0), 1.0);
        assert_eq!(all_w.lf(5), 1.0);
        let one = LrSchedule::with_warmup_iters(DecayKind::CosineRestarts, 1, 1, lrf, 0, 0.0, 0.8).unwrap();
        assert_eq!(one.lf(0), 1.0);
        assert_eq!(one.lf(1), 1.0);

        // JSON contract round-trip.
        let json = serde_json::to_string(geo.contract()).unwrap();
        assert!(
            json.contains(r#""type":"cosine_restarts""#)
                && json.contains(r#""restart_period":10"#)
                && json.contains(r#""restart_mult":2"#)
        );
        let parsed: SchedContract = serde_json::from_str(&json).unwrap();
        assert_eq!(LrSchedule::from_contract(parsed).unwrap(), geo);
    }

    #[test]
    fn effective_lf_ramps_during_warmup_and_matches_lf_after() {
        let lr0 = 0.01;
        let decay = ParamGroup::new("decay", vec![("w".into(), Tensor::zeros([1]))]).lr(lr0).momentum(0.937);
        for kind in [DecayKind::Linear, DecayKind::Cosine, DecayKind::Wsd, DecayKind::Polynomial] {
            let s = LrSchedule::with_warmup_iters(kind, 10, 10, 0.01, 20, 0.1, 0.8).unwrap();
            assert_eq!(s.effective_lf(0), 0.0, "{kind:?} t=0");
            for t in 0..100 {
                let want = s.lr_at(t, &decay) / lr0;
                assert!(
                    (s.effective_lf(t) - want).abs() < 1e-14,
                    "{kind:?} t={t}: effective_lf={} vs lr_at/lr0={want}",
                    s.effective_lf(t)
                );
            }
            for t in 20..100 {
                assert_eq!(s.effective_lf(t), s.lf(t), "{kind:?} post-warmup t={t}");
            }
        }
    }
}
