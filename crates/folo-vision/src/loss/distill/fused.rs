//! [`distill_loss`](super::distill_loss) on a folo-gpu device from the teacher's raw outputs, with the class and
//! DFL terms fused ([`folo_gpu::distill::distill_loss`]: no teacher probability tensors for them; the values
//! read each head once and the teacher's class head twice, for the class term and the DFL weight, and so do the
//! gradients). Values agree with the tensor path to float rounding (tests/gpu_distill.rs), not bit for bit.

use folo_core::{Result, Tensor};

use super::{DistillConfig, DistillLoss};

/// Whether [`distill_loss_fused`] runs for a class head `cls_raw: [B, nc, A]` with `reg_max` bins per side: on a
/// folo-gpu device and at sizes the kernels take.
pub fn fused_supported(cls_raw: &Tensor, reg_max: usize) -> bool {
    let Ok((b, nc, a)) = cls_raw.dims3() else { return false };
    folo_gpu::detect::supported(cls_raw.device()) && folo_gpu::distill::supports(b, nc, a, reg_max)
}

/// The distillation term of docs/MODEL.md §8.9 for the student's raw one-to-many outputs `box_raw: [B, 4 · reg_max,
/// A]` and `cls_raw: [B, nc, A]` (f32) and the teacher's raw outputs `teacher_box` and `teacher_cls` of the same
/// shapes (f32 or f16), on a folo-gpu device ([`fused_supported`]), with the supervised loss's unfloored
/// target-score sum. The same terms, gradients and reported values as [`super::distill_loss`] on the targets of
/// those teacher outputs, with one readback.
pub fn distill_loss_fused(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    teacher_box: &Tensor,
    teacher_cls: &Tensor,
    target_score_sum: f32,
    cfg: &DistillConfig,
) -> Result<DistillLoss> {
    distill_loss_fused_with_context(box_raw, cls_raw, teacher_box, teacher_cls, target_score_sum, cfg, None)
}

/// [`distill_loss_fused`] with optional GPU supervised assignment context for conflict-free BCKD and GT-verified
/// DFL KD.
pub fn distill_loss_fused_with_context(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    teacher_box: &Tensor,
    teacher_cls: &Tensor,
    target_score_sum: f32,
    cfg: &DistillConfig,
    gt_asg: Option<&folo_gpu::detect::Assignment>,
) -> Result<DistillLoss> {
    let (_, ch, _) = box_raw.dims3()?;
    let s = folo_gpu::distill::KdSettings {
        reg_max: ch / 4,
        cls_gain: cfg.cls_gain,
        dfl_gain: cfg.dfl_gain,
        temperature: cfg.temperature,
        // `f32::max` ignores a NaN: a non-finite supervised loss (skipped anyway) still gives a finite normaliser.
        normaliser: target_score_sum.max(1.0),
        bckd_gamma: cfg.bckd_gamma,
    };
    let kd = folo_gpu::distill::distill_loss_with_context(
        box_raw,
        cls_raw,
        teacher_box,
        teacher_cls,
        &s,
        gt_asg,
        cfg.gt_verify,
    )?;
    let v = kd.stats.to_vec::<f32>()?;
    Ok(DistillLoss { total: kd.total, cls: v[0], dfl: v[1] })
}
