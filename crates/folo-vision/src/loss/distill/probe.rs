//! A diagnostic of the rank term's weight (docs/MODEL.md §8.9, "Gradient probe"): the L2 norms of the gradients
//! that the supervised loss, the class distillation term and the rank term each put on the student's class logits
//! of one batch. The trainer logs them for the first `FOLO_DISTILL_PROBE` batches of each epoch; `rank_gain` is set
//! from their ratio.

use folo_core::{Result, Tensor};

use super::{DistillConfig, DistillTargets, rank_loss};
use crate::decode::Anchors;
use crate::loss::{ImageTargets, LossConfig, detection_loss};

/// The L2 norms `[supervised, class distillation, rank]` of `dL/dcls_raw` of each term of one batch, each with its
/// gain and `B` as the training step weights it (`o2m` student). The rank norm is 0 when the rank term is off.
/// Costs a second supervised loss (assignment included) and three backward passes over `[B, nc, A]`.
pub fn class_gradient_norms(
    box_raw: &Tensor,
    cls_raw: &Tensor,
    anchors: &Anchors,
    targets: &[ImageTargets],
    loss_cfg: &LossConfig,
    kd: &DistillTargets,
    cfg: &DistillConfig,
) -> Result<[f32; 3]> {
    let b = cls_raw.dims3()?.0 as f32;
    let norm = |loss: &Tensor, x: &Tensor| -> Result<f32> {
        let g = loss.backward()?.get(x).map(|g| g.square()?.sum_all()?.to_vec::<f32>()).transpose()?;
        Ok(g.map_or(0.0, |v| v[0].sqrt()))
    };
    let box_raw = box_raw.detach();
    let leaf = || cls_raw.detach().requires_grad_();
    let x = leaf();
    let sup = detection_loss(&box_raw, &x, anchors, targets, loss_cfg)?;
    let n = sup.target_score_sum.max(1.0);
    let sup_norm = if sup.is_finite() { norm(&sup.total, &x)? } else { f32::NAN };
    // The class term alone: B · cls_gain · (softplus(x) − x · p~) / N (its constant entropy left out).
    let x = leaf();
    let kc = x.softplus()?.sub(&x.mul(&kd.cls_prob)?)?.sum_all()?.mul_scalar(b * cfg.cls_gain / n)?;
    let kc_norm = norm(&kc, &x)?;
    let rank_norm = match (&kd.rank_logit, cfg.rank_on()) {
        (Some(y), true) => {
            let x = leaf();
            let r = rank_loss(&x, y, &kd.cls_prob, cfg)?;
            norm(&r.loss.mul_scalar(b * cfg.rank_gain)?, &x)?
        }
        _ => 0.0,
    };
    Ok([sup_norm, kc_norm, rank_norm])
}
