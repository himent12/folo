//! Dense distillation (docs/MODEL.md §8.9): the student's raw head outputs learn a frozen teacher's class
//! probabilities and DFL distributions, anchor by anchor.
//!
//! A teacher of the student's `nc`, `reg_max` and strides has the same anchors, so its raw outputs align with the
//! student's element by element. [`TeacherView::new`] turns one teacher forward into probabilities,
//! [`DistillTargets::new`] builds the constants of the loss, and [`distill_loss`] the differentiable term:
//!
//! - `L_kc = (1 / N) · Σ_{b,c,a} [softplus(x) − x · p~ − H(p~)]`, the Bernoulli KL from the teacher's `p~` to
//!   the student's `sigmoid(x)` (`H(p~)` is the teacher's entropy, a constant: the gradient is that of the BCE
//!   against `p~`);
//! - `L_kd = (T² / N) · Σ_{b,a} q_a · (1/4) Σ_side KL(s~ ‖ softmax(z / T))` with `q_a = max_c p~`;
//! - optionally (`rank_gain > 0`) the class-wise rank term `L_rank` of `rank.rs`: per class, one minus the weighted
//!   Pearson correlation of the student's and the teacher's class logits over the batch's candidates;
//! - the term is `B · (cls_gain · L_kc + dfl_gain · L_kd + rank_gain · L_rank)`, `N = max(Σ T, 1)` the supervised
//!   one-to-many loss's normaliser.
//!
//! Every element is formed before it is summed (no cancellation of large sums), and the sums are the fixed-order
//! sums of `terms.rs`, so the term is reproducible bit for bit on the CPU and composed only of tensor ops that every
//! backend has (softplus, log-softmax along a dim, elementwise arithmetic, row sums).

mod probe;
mod rank;
#[cfg(test)]
mod tests;

use folo_core::{DType, Error, Result, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::invalid;
use super::terms::sum_fixed_order;

pub use probe::class_gradient_norms;
pub use rank::{RankLoss, rank_loss};

/// `TrainConfig.distill` (docs/MODEL.md §8.9): the teacher and the weights of the two distillation terms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillConfig {
    /// The teacher: a `.folo` weights file or a published name (`folo weights pull`), with the student's `nc`,
    /// `reg_max` and strides.
    pub teacher: String,
    /// `lambda_c`, the weight of the class term `L_kc`.
    pub cls_gain: f32,
    /// `lambda_d`, the weight of the DFL term `L_kd`.
    pub dfl_gain: f32,
    /// `T`, the temperature of the DFL softmaxes (`> 0`); the class term has none. Starts at 4, which beat 1 on
    /// the proxy (docs/notes/hill/r4-distill.md).
    pub temperature: f32,
    /// `lambda_r`, the weight of the class-wise rank term `L_rank` (`rank.rs`); 0 (the default) turns it off and
    /// leaves the term exactly the two-term one.
    pub rank_gain: f32,
    /// `tau`, the candidate floor of the rank term, in `(0, 1)`: an element counts when the student's or the
    /// teacher's probability is above it, and logits below `logit(tau)` are raised to it.
    pub rank_floor: f32,
    /// `gamma`, the power of an element's weight `max(p, p~)^gamma` in the rank term (finite, `>= 0`).
    pub rank_power: f32,
    /// The fewest candidates a class needs in a batch to count in the rank term (`>= 2`).
    pub rank_min: usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        DistillConfig {
            teacher: String::new(),
            cls_gain: 0.5,
            dfl_gain: 1.5,
            temperature: 4.0,
            rank_gain: 0.0,
            rank_floor: 0.005,
            rank_power: 0.5,
            rank_min: 8,
        }
    }
}

impl DistillConfig {
    /// The starting values (`cls_gain` 0.5, `dfl_gain` 1.5, `T` 4) with `teacher`.
    pub fn new(teacher: impl Into<String>) -> Self {
        DistillConfig { teacher: teacher.into(), ..DistillConfig::default() }
    }

    /// Checks the settings: a teacher is named, the gains and `rank_power` are finite and `>= 0`, the temperature
    /// finite and `> 0`, `rank_floor` in `(0, 1)` and `rank_min >= 2`. The message names the `distill.*` key.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.teacher.trim().is_empty() {
            return Err("distill.teacher must name a .folo weights file or published weights".into());
        }
        for (name, v) in [
            ("distill.cls_gain", self.cls_gain),
            ("distill.dfl_gain", self.dfl_gain),
            ("distill.rank_gain", self.rank_gain),
            ("distill.rank_power", self.rank_power),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(format!("{name} must be a finite number >= 0, got {v}"));
            }
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(format!("distill.temperature must be a finite number > 0, got {}", self.temperature));
        }
        if !(self.rank_floor > 0.0 && self.rank_floor < 1.0) {
            return Err(format!("distill.rank_floor must be a number in (0, 1), got {}", self.rank_floor));
        }
        if self.rank_min < 2 {
            return Err(format!("distill.rank_min must be at least 2, got {}", self.rank_min));
        }
        Ok(())
    }

    /// Whether the rank term is on (`rank_gain > 0`).
    pub fn rank_on(&self) -> bool {
        self.rank_gain > 0.0
    }
}

/// The teacher's probabilities of one forward: constants (built under `no_grad`, `F32`).
#[derive(Clone, Debug)]
pub struct TeacherView {
    /// `cls_raw_T` in `F32`, `[B, nc, A]` (the rank term's teacher logits).
    pub cls_logit: Tensor,
    /// `sigmoid(cls_raw_T)`, `[B, nc, A]`.
    pub cls_prob: Tensor,
    /// `softmax(box_raw_T / T)` over the bins of each side, `[B, 4, reg_max, A]`.
    pub box_prob: Tensor,
}

impl TeacherView {
    /// The view of a teacher's raw outputs `box_raw: [B, 4 · reg_max, A]` and `cls_raw: [B, nc, A]` (`F32` or `F16`;
    /// converted to `F32` first) at `temperature`.
    pub fn new(box_raw: &Tensor, cls_raw: &Tensor, reg_max: usize, temperature: f32) -> Result<Self> {
        let _g = no_grad();
        let (b, ch, a) = box_raw.dims3()?;
        if reg_max == 0 || ch != 4 * reg_max {
            return Err(invalid(format!("teacher box_raw {} is not [B, 4 · {reg_max}, A]", box_raw.shape())));
        }
        let z = box_raw.to_dtype(DType::F32)?.reshape([b, 4, reg_max, a])?;
        let z = if temperature == 1.0 { z } else { z.div_scalar(temperature)? };
        let cls_logit = cls_raw.to_dtype(DType::F32)?;
        Ok(TeacherView { cls_prob: cls_logit.sigmoid()?, cls_logit, box_prob: z.softmax(2)? })
    }
}

/// The constants of [`distill_loss`] for one batch, built from a [`TeacherView`] (`F32`, on its device).
#[derive(Clone, Debug)]
pub struct DistillTargets {
    /// `p~`, `[B, nc, A]`.
    pub cls_prob: Tensor,
    /// The per-element Bernoulli entropy `H(p~) = −p~ ln p~ − (1 − p~) ln(1 − p~)`, `[B, nc, A]` (logarithms of
    /// values below `f32::MIN_POSITIVE` are taken at it, so a 0 or 1 contributes 0).
    pub cls_entropy: Tensor,
    /// `ln s~`, `[B, 4, reg_max, A]` (at `f32::MIN_POSITIVE` for a bin of probability 0).
    pub box_log_prob: Tensor,
    /// `q_a · s~`, `[B, 4, reg_max, A]`, with `q_a = max_c p~[b, c, a]`.
    pub box_weight: Tensor,
    /// The rank term's teacher logits `y' = max(cls_raw_T, logit(rank_floor))`, `[B, nc, A]`; `None` when the rank
    /// term is off.
    pub rank_logit: Option<Tensor>,
}

impl DistillTargets {
    /// The targets of `view` without the rank term.
    pub fn new(view: &TeacherView) -> Result<Self> {
        let _g = no_grad();
        let (p, s) = (&view.cls_prob, &view.box_prob);
        let (b, _, a) = p.dims3()?;
        let tiny = f32::MIN_POSITIVE;
        let xlogx = |t: &Tensor| t.mul(&t.clamp(tiny, f32::MAX)?.log()?);
        let cls_entropy = xlogx(p)?.add(&xlogx(&p.rsub_scalar(1.0)?)?)?.neg()?;
        let q = p.max(1, false)?.reshape([b, 1, 1, a])?;
        Ok(DistillTargets {
            cls_prob: p.clone(),
            cls_entropy,
            box_log_prob: s.clamp(tiny, f32::MAX)?.log()?,
            box_weight: s.mul(&q)?,
            rank_logit: None,
        })
    }

    /// The targets of `view` for `cfg`: [`DistillTargets::new`], plus the rank term's floored teacher logits when
    /// `cfg.rank_gain > 0`.
    pub fn for_config(view: &TeacherView, cfg: &DistillConfig) -> Result<Self> {
        let mut t = Self::new(view)?;
        if cfg.rank_on() {
            let _g = no_grad();
            t.rank_logit = Some(view.cls_logit.maximum_scalar(rank::floor_logit(cfg.rank_floor))?);
        }
        Ok(t)
    }
}

/// The result of [`distill_loss`].
#[derive(Clone, Debug)]
pub struct DistillLoss {
    /// `B · (cls_gain · L_kc + dfl_gain · L_kd + rank_gain · L_rank)`, rank 0, differentiable with respect to the
    /// student's `box_raw` and `cls_raw`.
    pub total: Tensor,
    /// `cls_gain · L_kc`, without `B` (as [`super::LossItems`] reports its terms).
    pub cls: f32,
    /// `dfl_gain · L_kd`, without `B`.
    pub dfl: f32,
    /// `rank_gain · L_rank`, without `B`; 0 when the rank term is off.
    pub rank: f32,
    /// The classes the rank term used in this batch (`rank.rs`); 0 when it is off.
    pub rank_classes: f32,
}

impl DistillLoss {
    /// Whether the reported terms are finite (the trainer skips the batch otherwise).
    pub fn is_finite(&self) -> bool {
        self.cls.is_finite() && self.dfl.is_finite() && self.rank.is_finite()
    }
}

/// The distillation term of docs/MODEL.md §8.9 for the student's raw one-to-many outputs `box_raw:
/// [B, 4 · reg_max, A]` and `cls_raw: [B, nc, A]` (`F32`), the teacher's `targets` and the supervised loss's
/// unfloored target-score sum (`N = max(target_score_sum, 1)`). Errors when the shapes do not match, or when
/// `cfg` turns the rank term on and `targets` has no rank logits ([`DistillTargets::for_config`]).
///
/// Gradients: `dL/dx = B · cls_gain · (sigmoid(x) − p~) / N` and `dL/dz = B · dfl_gain · T · q_a · (softmax(z / T)
/// − s~) / (4 N)`, plus the rank term's ([`rank_loss`]).
pub fn distill_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    targets: &DistillTargets,
    target_score_sum: f32,
    cfg: &DistillConfig,
) -> Result<DistillLoss> {
    const OP: &str = "distill_loss";
    for t in [box_raw, cls_raw] {
        if t.dtype() != DType::F32 {
            return Err(Error::DTypeMismatch { op: OP, expected: DType::F32, got: t.dtype() });
        }
    }
    let (b, nc, a) = cls_raw.dims3()?;
    let (b2, ch, a2) = box_raw.dims3()?;
    let r = ch / 4;
    if (b2, a2) != (b, a)
        || ch != 4 * r
        || targets.cls_prob.dims() != [b, nc, a]
        || targets.cls_entropy.dims() != [b, nc, a]
        || targets.box_weight.dims() != [b, 4, r, a]
        || targets.box_log_prob.dims() != [b, 4, r, a]
        || targets.rank_logit.as_ref().is_some_and(|t| t.dims() != [b, nc, a])
    {
        return Err(invalid(format!(
            "student box_raw {} and cls_raw {} do not match the teacher targets {} and {}",
            box_raw.shape(),
            cls_raw.shape(),
            targets.box_weight.shape(),
            targets.cls_prob.shape()
        )));
    }
    let rank_logit = match (&targets.rank_logit, cfg.rank_on()) {
        (Some(y), true) => Some(y),
        (None, true) => {
            return Err(invalid("distill.rank_gain > 0 needs DistillTargets::for_config's rank logits"));
        }
        (_, false) => None,
    };
    // `f32::max` ignores a NaN: a non-finite supervised loss (skipped anyway) still gives a finite normaliser.
    let n = target_score_sum.max(1.0);
    let temp = cfg.temperature;

    // Class: per element softplus(x) − x · p~ − H(p~) (formed before any sum), row sums per (image, class).
    let elem_c = cls_raw.softplus()?.sub(&cls_raw.mul(&targets.cls_prob)?)?.sub(&targets.cls_entropy)?;
    let l_kc = sum_fixed_order(&elem_c.sum(2, false)?)?.mul_scalar(cfg.cls_gain / n)?;

    // DFL: per element q_a · s~ · (ln s~ − log_softmax(z / T)), row sums per (image, side, bin).
    let z = box_raw.reshape([b, 4, r, a])?;
    let z = if temp == 1.0 { z } else { z.div_scalar(temp)? };
    let elem_d = targets.box_log_prob.sub(&z.log_softmax(2)?)?.mul(&targets.box_weight)?;
    let l_kd = sum_fixed_order(&elem_d.sum(3, false)?)?.mul_scalar(cfg.dfl_gain * temp * temp / (4.0 * n))?;

    let Some(y) = rank_logit else {
        // One readback for both reported values.
        let v = Tensor::stack(&[l_kc.detach(), l_kd.detach()], 0)?.to_vec::<f32>()?;
        let total = l_kc.add(&l_kd)?.mul_scalar(b as f32)?;
        return Ok(DistillLoss { total, cls: v[0], dfl: v[1], rank: 0.0, rank_classes: 0.0 });
    };
    let rank = rank_loss(cls_raw, y, &targets.cls_prob, cfg)?;
    let l_rank = rank.loss.mul_scalar(cfg.rank_gain)?;
    // One readback for the three reported values and the classes used.
    let v =
        Tensor::stack(&[l_kc.detach(), l_kd.detach(), l_rank.detach(), rank.classes], 0)?.to_vec::<f32>()?;
    let total = l_kc.add(&l_kd)?.add(&l_rank)?.mul_scalar(b as f32)?;
    Ok(DistillLoss { total, cls: v[0], dfl: v[1], rank: v[2], rank_classes: v[3] })
}
