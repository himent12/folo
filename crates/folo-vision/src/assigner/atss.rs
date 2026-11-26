//! ATSS (Adaptive Training Sample Selection, Zhang et al. CVPR 2020) standalone sample assigner on plain slices.
//!
//! For one image with `A` anchor boxes partitioned across multi-scale pyramid levels (`level_sizes` summing
//! to `A`), `C` classes, and `G` ground-truth boxes (all in the same coordinate space, typically input pixels):
//!
//! 1. **Per-level L2 center-distance candidate selection.** For each valid (positive-area) ground-truth box
//!    `g` and each pyramid level `l`, select the `min(topk, level_sizes[l])` anchor boxes whose centers have
//!    the smallest squared Euclidean distance to `g`'s center (ties broken deterministically by smaller anchor
//!    index). The union across levels forms `g`'s candidate pool `C_g`.
//! 2. **Dynamic mean + std IoU threshold.** Compute `IoU(gt[g], anchor[a])` in `f64` for every candidate
//!    `a in C_g`, then set `t_g = max(mean(IoU) + std(IoU), min_iou_thresh)`.
//! 3. **Spatial & overlap filtering.** Keep candidates with `iou > 0.0 && iou >= t_g` and (when
//!    [`AtssConfig::in_gt_box`] is `true`, paper default) whose anchor center lies strictly inside `gt[g]`
//!    by more than [`AtssConfig::eps`].
//! 4. **Conflict resolution.** When an anchor is selected by multiple GTs, assign it to the GT with the
//!    strictly highest `IoU(gt[g], anchor[a])`, breaking ties deterministically by smaller GT index.
//!
//! Degenerate GT boxes (`width <= 0` or `height <= 0`) receive `0` candidates without disturbing valid GTs.
//! Non-finite anchor or GT coordinates set [`AtssAssignment::inputs_finite`] to `false` and return an empty,
//! all-finite assignment.

use serde::{Deserialize, Serialize};

use crate::assign::Assignment;
use crate::boxes::Xyxy;
use crate::error::{Error, Result};

/// Hyper-parameters for [`assign_atss`] (Zhang et al. CVPR 2020).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AtssConfig {
    /// Number of closest candidate anchors selected per FPN pyramid level (paper default `9`).
    pub topk: usize,
    /// Require positive anchor centers to lie inside the assigned GT box (paper default `true`).
    pub in_gt_box: bool,
    /// Use Bessel-corrected sample standard deviation (`N - 1` denominator when `N > 1`, matching PyTorch's
    /// default `std()`) instead of population standard deviation (`N` denominator; default `false`).
    pub unbiased_std: bool,
    /// Optional minimum floor in `[0, 1]` for the dynamic `mean + std` IoU threshold (default `0.0`).
    pub min_iou_thresh: f32,
    /// Strict boundary margin `>= 0` for the in-GT center containment check (default `0.0`).
    pub eps: f32,
}

impl Default for AtssConfig {
    fn default() -> Self {
        Self { topk: 9, in_gt_box: true, unbiased_std: false, min_iou_thresh: 0.0, eps: 0.0 }
    }
}

impl AtssConfig {
    /// Validate all ATSS hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if self.topk == 0 {
            return Err(Error::Config("atss.topk must be >= 1, got 0".into()));
        }
        if !(self.min_iou_thresh.is_finite() && (0.0..=1.0).contains(&self.min_iou_thresh)) {
            return Err(Error::Config(format!(
                "atss.min_iou_thresh must be finite and in [0, 1], got {}",
                self.min_iou_thresh
            )));
        }
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return Err(Error::Config(format!("atss.eps must be finite and >= 0, got {}", self.eps)));
        }
        Ok(())
    }
}

/// Per-anchor assignment produced by [`assign_atss`].
///
/// Background anchors (`fg_mask[a] == false`) have `target_gt[a] = 0`, `target_labels[a] = num_classes`,
/// `target_bboxes[a] = Xyxy::default()`, and `target_ious[a] = 0.0`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtssAssignment {
    /// Whether each anchor is foreground (`[A]`).
    pub fg_mask: Vec<bool>,
    /// Assigned GT index per anchor (`[A]`).
    pub target_gt: Vec<usize>,
    /// Assigned class id per anchor (`[A]`, `num_classes` for background).
    pub target_labels: Vec<u32>,
    /// Assigned GT box per anchor (`[A]`).
    pub target_bboxes: Vec<Xyxy>,
    /// Anchor-to-GT IoU for each foreground anchor (`[A]`, in `[0, 1]`).
    pub target_ious: Vec<f32>,
    /// Dynamic `max(mean + std, min_iou_thresh)` IoU threshold computed for each GT (`[G]`; `0.0` for
    /// degenerate or candidate-less GTs).
    pub iou_thresholds: Vec<f64>,
    /// Candidate pool size `|C_g|` considered per GT (`[G]`; `0` for degenerate GTs).
    pub candidate_counts: Vec<usize>,
    /// Total foreground anchors after conflict resolution.
    pub num_fg: usize,
    /// `true` when all anchor and GT coordinates were finite; `false` returns an empty assignment cleanly.
    pub inputs_finite: bool,
}

impl AtssAssignment {
    fn empty(num_anchors: usize, num_gts: usize, num_classes: usize, inputs_finite: bool) -> Self {
        Self {
            fg_mask: vec![false; num_anchors],
            target_gt: vec![0; num_anchors],
            target_labels: vec![num_classes as u32; num_anchors],
            target_bboxes: vec![Xyxy::default(); num_anchors],
            target_ious: vec![0.0; num_anchors],
            iou_thresholds: vec![0.0; num_gts],
            candidate_counts: vec![0; num_gts],
            num_fg: 0,
            inputs_finite,
        }
    }

    /// Convert to [`Assignment`], using [`Self::target_ious`] as `target_scores`.
    pub fn to_assignment(&self) -> Assignment {
        Assignment {
            fg_mask: self.fg_mask.clone(),
            target_gt: self.target_gt.clone(),
            target_labels: self.target_labels.clone(),
            target_bboxes: self.target_bboxes.clone(),
            target_scores: self.target_ious.clone(),
            num_fg: self.num_fg,
        }
    }

    /// Build dense `[A, num_classes]` row-major continuous quality targets (`target_ious[a]` at
    /// `(a, target_labels[a])`, `0.0` elsewhere).
    pub fn dense_quality_targets(&self, num_classes: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.fg_mask.len() * num_classes];
        for (a, &fg) in self.fg_mask.iter().enumerate() {
            if fg {
                out[a * num_classes + self.target_labels[a] as usize] = self.target_ious[a];
            }
        }
        out
    }
}

/// Build square anchor boxes `[cx - half, cy - half, cx + half, cy + half]` from anchor-point centers and
/// strides, where `half = 0.5 * scale * stride` (e.g. `scale = 8.0` for RetinaNet/ATSS-style anchors).
///
/// # Errors
/// Returns [`Error::Config`] on length mismatches, non-finite/non-positive `scale`, or non-finite/non-positive
/// strides or anchor points.
pub fn anchor_boxes_from_points(
    anchor_points: &[[f32; 2]],
    anchor_strides: &[f32],
    scale: f32,
) -> Result<Vec<Xyxy>> {
    if anchor_points.len() != anchor_strides.len() {
        return Err(Error::Config(format!(
            "anchor_boxes_from_points: anchor_points length {} != anchor_strides length {}",
            anchor_points.len(),
            anchor_strides.len()
        )));
    }
    if !(scale.is_finite() && scale > 0.0) {
        return Err(Error::Config(format!(
            "anchor_boxes_from_points: scale must be finite and > 0, got {scale}"
        )));
    }
    let mut out = Vec::with_capacity(anchor_points.len());
    for (i, (&pt, &stride)) in anchor_points.iter().zip(anchor_strides).enumerate() {
        if !(pt[0].is_finite() && pt[1].is_finite()) {
            return Err(Error::Config(format!(
                "anchor_boxes_from_points: anchor_points[{i}] must be finite, got {pt:?}"
            )));
        }
        if !(stride.is_finite() && stride > 0.0) {
            return Err(Error::Config(format!(
                "anchor_boxes_from_points: anchor_strides[{i}] must be finite and > 0, got {stride}"
            )));
        }
        let half = 0.5 * scale * stride;
        out.push(Xyxy::new(pt[0] - half, pt[1] - half, pt[0] + half, pt[1] + half));
    }
    Ok(out)
}

#[inline]
pub(super) fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Exact `f64` plain IoU in `[0, 1]` between two `Xyxy` boxes (`0.0` for empty/degenerate/disjoint boxes).
#[inline]
pub(super) fn box_iou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
    let w1 = (a.x2 as f64 - a.x1 as f64).max(0.0);
    let h1 = (a.y2 as f64 - a.y1 as f64).max(0.0);
    let w2 = (b.x2 as f64 - b.x1 as f64).max(0.0);
    let h2 = (b.y2 as f64 - b.y1 as f64).max(0.0);
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    if area1 <= 0.0 || area2 <= 0.0 {
        return 0.0;
    }
    let iw = ((a.x2 as f64).min(b.x2 as f64) - (a.x1 as f64).max(b.x1 as f64)).max(0.0);
    let ih = ((a.y2 as f64).min(b.y2 as f64) - (a.y1 as f64).max(b.y1 as f64)).max(0.0);
    let inter = iw * ih;
    let union = area1 + area2 - inter;
    if union <= 0.0 { 0.0 } else { (inter / union).clamp(0.0, 1.0) }
}

/// Validate that `level_sizes` sums to `num_anchors` without overflow.
pub(super) fn check_level_sizes(level_sizes: &[usize], num_anchors: usize, op: &str) -> Result<()> {
    let sum = level_sizes
        .iter()
        .try_fold(0usize, |acc, &len| acc.checked_add(len))
        .ok_or_else(|| Error::Config(format!("{op}: level_sizes sum overflows usize")))?;
    if sum != num_anchors {
        return Err(Error::Config(format!(
            "{op}: level_sizes sum ({sum}) != anchor_bboxes length ({num_anchors})"
        )));
    }
    Ok(())
}

/// Run ATSS (Adaptive Training Sample Selection) label assignment for one image.
///
/// - `anchor_bboxes`: `[A]` anchor boxes ordered contiguously by pyramid level.
/// - `level_sizes`: number of anchors in each pyramid level (must sum to `A`; pass `&[A]` for single-scale).
/// - `gt_bboxes`: `[G]` ground-truth boxes.
/// - `gt_labels`: `[G]` class ids (`< num_classes`).
/// - `num_classes`: number of foreground classes `C`.
/// - `cfg`: [`AtssConfig`] hyper-parameters.
///
/// # Errors
/// Returns [`Error::Config`] on invalid config parameters, `level_sizes` sum mismatch, `gt_labels` length
/// or range mismatch, or `num_classes == 0` with non-empty GTs.
pub fn assign_atss(
    anchor_bboxes: &[Xyxy],
    level_sizes: &[usize],
    gt_bboxes: &[Xyxy],
    gt_labels: &[u32],
    num_classes: usize,
    cfg: &AtssConfig,
) -> Result<AtssAssignment> {
    cfg.validate()?;

    let n_a = anchor_bboxes.len();
    let n_g = gt_bboxes.len();
    check_level_sizes(level_sizes, n_a, "atss")?;
    if gt_labels.len() != n_g {
        return Err(Error::Config(format!("atss: gt_labels length {} != G ({n_g})", gt_labels.len())));
    }
    if n_g > 0 && num_classes == 0 {
        return Err(Error::Config("atss: num_classes must be >= 1 when gt_bboxes is non-empty".into()));
    }
    for (g, &l) in gt_labels.iter().enumerate() {
        if (l as usize) >= num_classes {
            return Err(Error::Config(format!("atss: gt_labels[{g}] = {l} >= num_classes ({num_classes})")));
        }
    }

    let all_finite = anchor_bboxes.iter().all(box_is_finite) && gt_bboxes.iter().all(box_is_finite);
    if !all_finite {
        return Ok(AtssAssignment::empty(n_a, n_g, num_classes, false));
    }
    if n_g == 0 || n_a == 0 {
        return Ok(AtssAssignment::empty(n_a, n_g, num_classes, true));
    }

    let anchor_centers: Vec<[f64; 2]> = anchor_bboxes
        .iter()
        .map(|b| [0.5 * (b.x1 as f64 + b.x2 as f64), 0.5 * (b.y1 as f64 + b.y2 as f64)])
        .collect();

    let min_thr = cfg.min_iou_thresh as f64;
    let eps = cfg.eps as f64;

    let mut out = AtssAssignment::empty(n_a, n_g, num_classes, true);
    // For each anchor, the winning `(gt_idx, iou)` among GTs that selected it.
    let mut owner: Vec<Option<(usize, f64)>> = vec![None; n_a];
    let mut level_dists: Vec<(f64, usize)> = Vec::new();
    let mut cands: Vec<(usize, f64)> = Vec::new();

    for g in 0..n_g {
        let gt = &gt_bboxes[g];
        let gw = gt.x2 as f64 - gt.x1 as f64;
        let gh = gt.y2 as f64 - gt.y1 as f64;
        if gw <= 0.0 || gh <= 0.0 {
            continue;
        }
        let gt_cx = 0.5 * (gt.x1 as f64 + gt.x2 as f64);
        let gt_cy = 0.5 * (gt.y1 as f64 + gt.y2 as f64);

        cands.clear();
        let mut offset = 0usize;
        for &lvl_len in level_sizes {
            let end = offset + lvl_len;
            if lvl_len > 0 {
                level_dists.clear();
                level_dists.reserve(lvl_len);
                for (a, center) in anchor_centers.iter().enumerate().take(end).skip(offset) {
                    let dx = center[0] - gt_cx;
                    let dy = center[1] - gt_cy;
                    level_dists.push((dx * dx + dy * dy, a));
                }
                let k_l = cfg.topk.min(lvl_len);
                if lvl_len > k_l {
                    level_dists.select_nth_unstable_by(k_l - 1, |x, y| {
                        x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1))
                    });
                    level_dists.truncate(k_l);
                }
                // Deterministic anchor ordering within each level so mean/std summation order is fixed.
                level_dists.sort_unstable_by_key(|&(_, a)| a);
                for &(_, a) in &level_dists {
                    let iou = box_iou_f64(gt, &anchor_bboxes[a]);
                    cands.push((a, iou));
                }
            }
            offset = end;
        }

        let m = cands.len();
        out.candidate_counts[g] = m;
        if m == 0 {
            continue;
        }

        let mean = cands.iter().map(|&(_, iou)| iou).sum::<f64>() / m as f64;
        let var = if m <= 1 {
            0.0
        } else {
            let denom = if cfg.unbiased_std { (m - 1) as f64 } else { m as f64 };
            cands
                .iter()
                .map(|&(_, iou)| {
                    let d = iou - mean;
                    d * d
                })
                .sum::<f64>()
                / denom
        };
        let thresh = (mean + var.sqrt()).max(min_thr);
        out.iou_thresholds[g] = thresh;

        for &(a, iou) in &cands {
            if iou <= 0.0 || iou < thresh {
                continue;
            }
            if cfg.in_gt_box {
                let [ax, ay] = anchor_centers[a];
                let margin =
                    (ax - gt.x1 as f64).min(ay - gt.y1 as f64).min(gt.x2 as f64 - ax).min(gt.y2 as f64 - ay);
                if margin <= eps {
                    continue;
                }
            }
            // Conflict resolution: strictly higher IoU wins; ties stay with smaller GT index.
            let slot = &mut owner[a];
            if slot.is_none_or(|(_, best_iou)| iou > best_iou) {
                *slot = Some((g, iou));
            }
        }
    }

    for (a, slot) in owner.into_iter().enumerate() {
        if let Some((g, iou)) = slot {
            out.fg_mask[a] = true;
            out.target_gt[a] = g;
            out.target_labels[a] = gt_labels[g];
            out.target_bboxes[a] = gt_bboxes[g];
            out.target_ious[a] = iou as f32;
            out.num_fg += 1;
        }
    }
    Ok(out)
}
