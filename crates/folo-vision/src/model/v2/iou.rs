//! Opt-in `LazyAlignHead` localization-quality (`iou_out`) training loss and inference score fusion.
//!
//! - [`iou_quality_loss`]: supervises `iou_raw [B, 1, A]` on foreground anchors with BCE-with-logits against the
//!   detached predicted-vs-ground-truth plain IoU, weighted by `asg.score` and normalized by `target_score_sum`
//!   (scaled by `B * gain` to match `detection_loss`).
//! - [`fuse_cls_iou_scores`]: fuses per-anchor class probabilities `p_cls` with `p_iou = sigmoid(iou_logit)` via
//!   `p_cls^(1 - alpha) * p_iou^alpha` (geometric mean at [`DEFAULT_IOU_FUSION_ALPHA`] = `0.5`).

use folo_core::{DType, Error, Result, Tensor, no_grad};

use crate::decode::Anchors;
use crate::loss::BatchAssignment;

const OP: &str = "iou_quality_loss";

/// Default localization-quality fusion exponent `alpha = 0.5` (geometric mean `sqrt(p_cls * p_iou)`).
pub const DEFAULT_IOU_FUSION_ALPHA: f32 = 0.5;

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidArgument { op: OP, msg: msg.into() }
}

/// Differentiable BCE-with-logits loss for `LazyAlignHead`'s opt-in `iou_out` branch (`[B, 1, A]`).
///
/// On every foreground anchor `k` in `asg`, decodes the predicted box from `box_raw` under `no_grad()`,
/// computes its detached plain IoU with `asg.target[k]` in `[0, 1]`, and forms:
///
/// ```text
/// L_iou = B * gain * sum_fg(t_k * BCEWithLogits(iou_raw_k, detach(IoU_k))) / max(sum_fg t_k, 1.0)
/// ```
///
/// When `asg.num_fg() == 0` or `gain == 0.0`, returns a scalar zero connected to `iou_raw` on the same device
/// so `backward()` runs cleanly with zero gradients.
pub fn iou_quality_loss(
    iou_raw: &Tensor,
    box_raw: &Tensor,
    anchors: &Anchors,
    asg: &BatchAssignment,
    reg_max: usize,
    gain: f32,
) -> Result<Tensor> {
    if !gain.is_finite() || gain < 0.0 {
        return Err(invalid(format!("gain must be finite and >= 0, got {gain}")));
    }
    if reg_max < 2 {
        return Err(invalid(format!("reg_max must be at least 2, got {reg_max}")));
    }
    let (b, c_iou, a) =
        iou_raw.dims3().map_err(|_| invalid(format!("iou_raw {:?} must be [B, 1, A]", iou_raw.dims())))?;
    if c_iou != 1 {
        return Err(invalid(format!("iou_raw {:?} must have 1 channel", iou_raw.dims())));
    }
    let (b2, c_box, a2) = box_raw
        .dims3()
        .map_err(|_| invalid(format!("box_raw {:?} must be [B, 4 * reg_max, A]", box_raw.dims())))?;
    if (b2, a2) != (b, a) || c_box != 4 * reg_max {
        return Err(invalid(format!(
            "box_raw {:?} does not match iou_raw [{b}, 1, {a}] with reg_max {reg_max}",
            box_raw.dims()
        )));
    }
    if anchors.len() != a {
        return Err(invalid(format!("anchors length {} does not match A = {a}", anchors.len())));
    }
    if asg.batch != b || asg.num_anchors != a {
        return Err(invalid(format!(
            "assignment ({}, {}) does not match batch {b}, anchors {a}",
            asg.batch, asg.num_anchors
        )));
    }
    if iou_raw.device() != box_raw.device() {
        return Err(invalid("iou_raw and box_raw must be on the same device"));
    }

    let dev = iou_raw.device();
    let iou_f32 =
        if iou_raw.dtype() == DType::F32 { iou_raw.clone() } else { iou_raw.to_dtype(DType::F32)? };
    let n_fg = asg.num_fg();
    if asg.anchor.len() != n_fg || asg.target.len() != n_fg || asg.score.len() != n_fg {
        return Err(invalid("assignment foreground arrays have inconsistent lengths"));
    }
    if n_fg == 0 || gain == 0.0 {
        return iou_f32.sum_all()?.mul_scalar(0.0);
    }

    let mut fg_idx = Vec::with_capacity(n_fg);
    let mut anc_xy = Vec::with_capacity(n_fg * 2);
    let mut gt_grid = Vec::with_capacity(n_fg * 4);
    for k in 0..n_fg {
        let img = asg.image[k] as usize;
        let an = asg.anchor[k] as usize;
        if img >= b || an >= a {
            return Err(invalid(format!(
                "assignment entry {k}: (image {img}, anchor {an}) out of bounds ({b}, {a})"
            )));
        }
        fg_idx.push((img * a + an) as u32);
        let [px, py] = anchors.points[an];
        let s = anchors.strides[an];
        let gt = asg.target[k];
        anc_xy.extend_from_slice(&[px, py]);
        gt_grid.extend_from_slice(&[gt.x1 / s, gt.y1 / s, gt.x2 / s, gt.y2 / s]);
    }

    let idx_t = Tensor::from_vec(fg_idx, [n_fg])?.to_device(dev)?;
    let iou_logits = iou_f32.reshape([b * a])?.index_select(0, &idx_t)?;

    // Detached predicted-vs-GT plain IoU in grid units (device-agnostic, zero host sync).
    let target_iou = {
        let _g = no_grad();
        let box_f32 =
            if box_raw.dtype() == DType::F32 { box_raw.clone() } else { box_raw.to_dtype(DType::F32)? };
        let fg_box = box_f32
            .transpose(1, 2)?
            .contiguous()?
            .reshape([b * a, 4 * reg_max])?
            .index_select(0, &idx_t)?
            .reshape([n_fg, 4, reg_max])?;
        let probs = fg_box.softmax(-1)?;
        let bins =
            Tensor::from_vec((0..reg_max).map(|i| i as f32).collect(), [1, 1, reg_max])?.to_device(dev)?;
        let ltrb = probs.mul(&bins)?.sum(2, false)?; // [N_fg, 4]
        let anc = Tensor::from_vec(anc_xy, [n_fg, 2])?.to_device(dev)?;
        let gt = Tensor::from_vec(gt_grid, [n_fg, 4])?.to_device(dev)?;
        let col = |t: &Tensor, k: usize| t.narrow(1, k, 1)?.squeeze(1);
        let (px, py) = (col(&anc, 0)?, col(&anc, 1)?);
        let (pd_x1, pd_y1) = (px.sub(&col(&ltrb, 0)?)?, py.sub(&col(&ltrb, 1)?)?);
        let (pd_x2, pd_y2) = (px.add(&col(&ltrb, 2)?)?, py.add(&col(&ltrb, 3)?)?);
        let (gt_x1, gt_y1, gt_x2, gt_y2) = (col(&gt, 0)?, col(&gt, 1)?, col(&gt, 2)?, col(&gt, 3)?);
        let iw = pd_x2.minimum(&gt_x2)?.sub(&pd_x1.maximum(&gt_x1)?)?.relu()?;
        let ih = pd_y2.minimum(&gt_y2)?.sub(&pd_y1.maximum(&gt_y1)?)?.relu()?;
        let inter = iw.mul(&ih)?;
        let area_pd = pd_x2.sub(&pd_x1)?.relu()?.mul(&pd_y2.sub(&pd_y1)?.relu()?)?;
        let area_gt = gt_x2.sub(&gt_x1)?.relu()?.mul(&gt_y2.sub(&gt_y1)?.relu()?)?;
        let union = area_pd.add(&area_gt)?.sub(&inter)?.relu()?.add_scalar(1e-7)?;
        inter.div(&union)?.clamp(0.0, 1.0)?
    };

    // Numerically stable BCEWithLogits(z, t) = softplus(z) - z * t.
    let bce = iou_logits.softplus()?.sub(&iou_logits.mul(&target_iou)?)?;
    let weights = Tensor::from_vec(asg.score.clone(), [n_fg])?.to_device(dev)?;
    let tss = asg.target_score_sum().max(1.0) as f32;
    bce.mul(&weights)?.sum_all()?.mul_scalar((b as f32) * gain / tss)
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Fuse per-anchor class probabilities `cls_scores` (`[A, num_classes]`, row-major) with `LazyAlignHead`'s
/// raw IoU quality logits `iou_logits` (`[A]`) in place:
///
/// ```text
/// s'_{a, c} = clamp(s_{a, c}, 0, 1)^(1 - alpha) * sigmoid(iou_logits[a])^alpha
/// ```
///
/// - `alpha == 0.0`: exact no-op.
/// - `alpha == 0.5` ([`DEFAULT_IOU_FUSION_ALPHA`]): geometric mean `sqrt(p_cls * p_iou)`.
/// - `alpha == 1.0`: pure localization quality `p_iou`.
///
/// # Panics
/// If `alpha` is not finite or not in `0..=1`, `num_classes == 0`, or
/// `cls_scores.len() != iou_logits.len() * num_classes`.
pub fn fuse_cls_iou_scores(cls_scores: &mut [f32], iou_logits: &[f32], num_classes: usize, alpha: f32) {
    assert!(
        alpha.is_finite() && (0.0..=1.0).contains(&alpha),
        "fuse_cls_iou_scores: alpha must be in 0..=1, got {alpha}"
    );
    assert!(num_classes > 0, "fuse_cls_iou_scores: num_classes must be positive");
    assert_eq!(
        cls_scores.len(),
        iou_logits.len() * num_classes,
        "fuse_cls_iou_scores: cls_scores length {} != {} * {}",
        cls_scores.len(),
        iou_logits.len(),
        num_classes
    );
    if alpha == 0.0 {
        return;
    }
    let cls_exp = 1.0 - alpha;
    for (row, &z) in cls_scores.chunks_exact_mut(num_classes).zip(iou_logits) {
        let p_iou = sigmoid_f32(z);
        if alpha == 1.0 {
            row.fill(p_iou);
        } else if alpha == 0.5 {
            for s in row {
                let c = if s.is_nan() { f32::NAN } else { s.clamp(0.0, 1.0) };
                *s = (c * p_iou).sqrt();
            }
        } else {
            let iou_fac = p_iou.powf(alpha);
            for s in row {
                let c = if s.is_nan() { f32::NAN } else { s.clamp(0.0, 1.0) };
                *s = c.powf(cls_exp) * iou_fac;
            }
        }
    }
}
