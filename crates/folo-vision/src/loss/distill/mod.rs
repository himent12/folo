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
//! - the term is `B · (cls_gain · L_kc + dfl_gain · L_kd)`, `N = max(Σ T, 1)` the supervised one-to-many loss's
//!   normaliser.
//!
//! Every element is formed before it is summed (no cancellation of large sums), and the sums are the fixed-order
//! sums of `terms.rs`, so the term is reproducible bit for bit on the CPU and composed only of tensor ops that every
//! backend has (softplus, log-softmax along a dim, elementwise arithmetic, row sums).

mod fused;
#[cfg(test)]
mod tests;

use folo_core::{DType, Error, Result, Tensor, no_grad};
use serde::{Deserialize, Serialize};

use super::invalid;
use super::terms::sum_fixed_order;

pub use fused::{distill_loss_fused, distill_loss_fused_with_context, fused_supported};

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

fn is_false(v: &bool) -> bool {
    !*v
}

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
    /// BCKD focal disagreement exponent `γ` (`0.0` = legacy dense Bernoulli KL). When `> 0`, weights class KL by
    /// `sg(|p_S − p_T|^γ)` and clamps positive teacher targets upward to `max(p_T, t_gt)` when GT context is given.
    #[serde(skip_serializing_if = "is_zero_f32")]
    pub bckd_gamma: f32,
    /// GT-Verified DFL KD (`false` = legacy `max_c p_T` weighting). When `true`, distills DFL only on GT-matched
    /// positive anchors where `IoU(b_T, g) > IoU(b_S, g) + 0.02` and `IoU(b_T, g) > 0.50`, weighted by `IoU_T − IoU_S`.
    #[serde(skip_serializing_if = "is_false")]
    pub gt_verify: bool,
}

impl Default for DistillConfig {
    fn default() -> Self {
        DistillConfig {
            teacher: String::new(),
            cls_gain: 0.5,
            dfl_gain: 1.5,
            temperature: 4.0,
            bckd_gamma: 0.0,
            gt_verify: false,
        }
    }
}

/// The largest `cls_gain` and `dfl_gain` [`DistillConfig::validate`] accepts.
pub const MAX_GAIN: f32 = 1e6;

/// The temperatures [`DistillConfig::validate`] accepts.
pub const TEMPERATURE: std::ops::RangeInclusive<f32> = 0.05..=100.0;

impl DistillConfig {
    /// The starting values (`cls_gain` 0.5, `dfl_gain` 1.5, `T` 4) with `teacher`.
    pub fn new(teacher: impl Into<String>) -> Self {
        DistillConfig { teacher: teacher.into(), ..DistillConfig::default() }
    }

    /// Checks the settings: a teacher is named, the gains are finite and `>= 0`, the temperature finite and `> 0`.
    /// The message names the `distill.*` key.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.teacher.trim().is_empty() {
            return Err("distill.teacher must name a .folo weights file or published weights".into());
        }
        // The bounds keep the GPU kernel's scales (`B · λ · T / N` and the like) far inside f32's range.
        for (name, v) in [("distill.cls_gain", self.cls_gain), ("distill.dfl_gain", self.dfl_gain)] {
            if !(0.0..=MAX_GAIN).contains(&v) {
                return Err(format!("{name} must be a number from 0 to {MAX_GAIN}, got {v}"));
            }
        }
        if !TEMPERATURE.contains(&self.temperature) {
            let (lo, hi) = (TEMPERATURE.start(), TEMPERATURE.end());
            return Err(format!(
                "distill.temperature must be a number from {lo} to {hi}, got {}",
                self.temperature
            ));
        }
        if !(0.0..=5.0).contains(&self.bckd_gamma) {
            return Err(format!("distill.bckd_gamma must be in 0..=5, got {}", self.bckd_gamma));
        }
        Ok(())
    }
}

/// The teacher's probabilities of one forward: constants (built under `no_grad`, `F32`).
#[derive(Clone, Debug)]
pub struct TeacherView {
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
        Ok(TeacherView { cls_prob: cls_raw.to_dtype(DType::F32)?.sigmoid()?, box_prob: z.softmax(2)? })
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
    /// Unweighted `s~ = softmax(box_raw_T / T)`, `[B, 4, reg_max, A]` (used with GT-verified anchor weights).
    pub box_prob: Tensor,
}

impl DistillTargets {
    /// The targets of `view`.
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
            box_prob: s.clone(),
        })
    }
}

/// Positive-only GT-verified DFL KD anchor weights `[B, 1, 1, A]` on the host (matching GPU `kd_gt_verify.wgsl`).
/// Decodes only the `N_fg` positive anchors of `asg`, setting weight `iou_t − iou_s` where
/// `iou_t > iou_s + 0.02 && iou_t > 0.50` and `0.0` elsewhere.
pub fn gt_verify_weights_cpu(
    student_box: &Tensor,
    teacher_box: &Tensor,
    anchors: &crate::decode::Anchors,
    asg: &super::BatchAssignment,
    reg_max: usize,
) -> Result<Tensor> {
    let _g = no_grad();
    let (b, _, a) = student_box.dims3()?;
    let sb = student_box.to_dtype(DType::F32)?.to_vec::<f32>()?;
    let tb = teacher_box.to_dtype(DType::F32)?.to_vec::<f32>()?;
    let mut weights = vec![0.0f32; b * a];
    let decode_one = |raw: &[f32], img: usize, anc: usize| -> crate::boxes::Xyxy {
        let base = img * 4 * reg_max * a + anc;
        let mut d = [0.0f32; 4];
        for side in 0..4 {
            let mut m = f32::NEG_INFINITY;
            for k in 0..reg_max {
                m = m.max(raw[base + (side * reg_max + k) * a]);
            }
            let (mut z, mut e) = (0.0f32, 0.0f32);
            for k in 0..reg_max {
                let p = (raw[base + (side * reg_max + k) * a] - m).exp();
                z += p;
                e += p * k as f32;
            }
            d[side] = e / z;
        }
        crate::decode::dist2bbox(d, anchors.points[anc])
    };
    for k in 0..asg.num_fg() {
        if asg.score[k] <= 0.0 {
            continue;
        }
        let img = asg.image[k] as usize;
        let anc = asg.anchor[k] as usize;
        let inv = 1.0 / anchors.strides[anc];
        let gt = asg.target[k].scale(inv, inv);
        let pd_s = decode_one(&sb, img, anc);
        let pd_t = decode_one(&tb, img, anc);
        let iou_s = crate::assign::iou_plain(&pd_s, &gt);
        let iou_t = crate::assign::iou_plain(&pd_t, &gt);
        if iou_t > iou_s + 0.02 && iou_t > 0.50 {
            weights[img * a + anc] = iou_t - iou_s;
        }
    }
    Tensor::from_vec(weights, [b, 1, 1, a])?.to_device(student_box.device())
}

/// The result of [`distill_loss`].
#[derive(Clone, Debug)]
pub struct DistillLoss {
    /// `B · (cls_gain · L_kc + dfl_gain · L_kd)`, rank 0, differentiable with respect to the student's `box_raw`
    /// and `cls_raw`.
    pub total: Tensor,
    /// `cls_gain · L_kc`, without `B` (as [`super::LossItems`] reports its terms).
    pub cls: f32,
    /// `dfl_gain · L_kd`, without `B`.
    pub dfl: f32,
}

impl DistillLoss {
    /// Whether both reported terms are finite (the trainer skips the batch otherwise).
    pub fn is_finite(&self) -> bool {
        self.cls.is_finite() && self.dfl.is_finite()
    }
}

/// The distillation term of docs/MODEL.md §8.9 for the student's raw one-to-many outputs `box_raw:
/// [B, 4 · reg_max, A]` and `cls_raw: [B, nc, A]` (`F32`), the teacher's `targets` and the supervised loss's
/// unfloored target-score sum (`N = max(target_score_sum, 1)`). Errors when the shapes do not match.
///
/// Gradients: `dL/dx = B · cls_gain · (sigmoid(x) − p~) / N` and `dL/dz = B · dfl_gain · T · q_a · (softmax(z / T)
/// − s~) / (4 N)`.
pub fn distill_loss(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    targets: &DistillTargets,
    target_score_sum: f32,
    cfg: &DistillConfig,
) -> Result<DistillLoss> {
    distill_loss_with_context(box_raw, cls_raw, targets, target_score_sum, cfg, None, None)
}

/// [`distill_loss`] with optional supervised assignment context (`gt_asg` for conflict-free BCKD positive
/// clamping when `cfg.bckd_gamma > 0`, and `anchor_ver_w: [B, 1, 1, A]` for GT-verified DFL KD).
pub fn distill_loss_with_context(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    targets: &DistillTargets,
    target_score_sum: f32,
    cfg: &DistillConfig,
    gt_asg: Option<&super::BatchAssignment>,
    anchor_ver_w: Option<&Tensor>,
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
    {
        return Err(invalid(format!(
            "student box_raw {} and cls_raw {} do not match the teacher targets {} and {}",
            box_raw.shape(),
            cls_raw.shape(),
            targets.box_weight.shape(),
            targets.cls_prob.shape()
        )));
    }
    // `f32::max` ignores a NaN: a non-finite supervised loss (skipped anyway) still gives a finite normaliser.
    let n = target_score_sum.max(1.0);
    let temp = cfg.temperature;

    // Optional conflict-free positive target clamp: when GT target > teacher prob on a positive, clamp teacher up.
    let (cls_prob, cls_entropy) = if cfg.bckd_gamma > 0.0
        && let Some(asg) = gt_asg
        && asg.num_fg() > 0
    {
        let _g = no_grad();
        let mut p = targets.cls_prob.to_vec::<f32>()?;
        let mut h = targets.cls_entropy.to_vec::<f32>()?;
        for k in 0..asg.num_fg() {
            let idx = (asg.image[k] as usize * nc + asg.label[k] as usize) * a + asg.anchor[k] as usize;
            let t_gt = asg.score[k];
            if t_gt > p[idx] {
                p[idx] = t_gt;
                let q = 1.0 - t_gt;
                h[idx] =
                    if t_gt > 1e-7 && t_gt < 1.0 - 1e-7 { -(t_gt * t_gt.ln() + q * q.ln()) } else { 0.0 };
            }
        }
        let dev = cls_raw.device();
        (Tensor::from_vec(p, [b, nc, a])?.to_device(dev)?, Tensor::from_vec(h, [b, nc, a])?.to_device(dev)?)
    } else {
        (targets.cls_prob.clone(), targets.cls_entropy.clone())
    };

    // Class: per element softplus(x) − x · p~ − H(p~), optionally weighted by detached BCKD |p_S − p_T|^γ.
    let mut elem_c = cls_raw.softplus()?.sub(&cls_raw.mul(&cls_prob)?)?.sub(&cls_entropy)?;
    if cfg.bckd_gamma > 0.0 {
        let w = {
            let _g = no_grad();
            cls_raw.sigmoid()?.sub(&cls_prob)?.abs()?.powf(cfg.bckd_gamma)?
        };
        elem_c = elem_c.mul(&w)?;
    }
    let l_kc = sum_fixed_order(&elem_c.sum(2, false)?)?.mul_scalar(cfg.cls_gain / n)?;

    // DFL: per element weight · (ln s~ − log_softmax(z / T)), row sums per (image, side, bin).
    let z = box_raw.reshape([b, 4, r, a])?;
    let z = if temp == 1.0 { z } else { z.div_scalar(temp)? };
    let bw = if cfg.gt_verify {
        let vw = anchor_ver_w.ok_or_else(|| invalid("distill_loss: cfg.gt_verify requires anchor_ver_w"))?;
        targets.box_prob.mul(vw)?
    } else {
        match anchor_ver_w {
            Some(vw) => targets.box_prob.mul(vw)?,
            None => targets.box_weight.clone(),
        }
    };
    let elem_d = targets.box_log_prob.sub(&z.log_softmax(2)?)?.mul(&bw)?;
    let l_kd = sum_fixed_order(&elem_d.sum(3, false)?)?.mul_scalar(cfg.dfl_gain * temp * temp / (4.0 * n))?;

    // One readback for both reported values.
    let v = Tensor::stack(&[l_kc.detach(), l_kd.detach()], 0)?.to_vec::<f32>()?;
    let total = l_kc.add(&l_kd)?.mul_scalar(b as f32)?;
    Ok(DistillLoss { total, cls: v[0], dfl: v[1] })
}
