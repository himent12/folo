//! SimOTA (Simplified Optimal Transport Assignment, Ge et al. YOLOX 2021) on plain slices.
//!
//! For one image with `A` anchors, `C` classes, and `G` ground truths (all boxes, points, and strides in
//! **input pixels**):
//!
//! 1. **Spatial candidate selection.** For each valid (non-degenerate, positive-area) GT box `g` with centre
//!    `(cx, cy)`:
//!    - `is_in_box(g, a)` holds when [`SimOtaConfig::in_box`] is enabled and `anchor_points[a]` lies
//!      strictly inside `gt_bboxes[g]` by more than [`SimOtaConfig::eps`].
//!    - `is_in_center(g, a)` holds when [`SimOtaConfig::center_radius`] is `Some(r)` and
//!      `|ax - cx| <= r * stride` and `|ay - cy| <= r * stride`.
//!    - When both priors are enabled (YOLOX default), candidates are `is_in_box || is_in_center` and
//!      anchors not in the intersection `is_in_box && is_in_center` receive [`SimOtaConfig::non_prior_cost`].
//!      When only one prior is enabled, candidates are anchors satisfying that prior with zero penalty.
//!      When neither prior is enabled, all `A` anchors are candidates with zero penalty.
//! 2. **Pairwise matching cost** (evaluated in `f64` on candidates only):
//!    ```text
//!    cls_cost(g, a) = softplus(-x[a, y_g]) + sum_{c != y_g} softplus(x[a, c])
//!    iou_cost(g, a) = -ln(iou(gt[g], pred[a]) + iou_eps)
//!    cost(g, a)     = cls_weight * cls_cost(g, a) + iou_weight * iou_cost(g, a) + prior_penalty(g, a)
//!    ```
//! 3. **Dynamic `k` estimation.** For each GT `g` with `M_g = |C_g| > 0` candidates, take the
//!    `q = min(top_q, M_g)` highest IoUs among its candidates, sum them in deterministic order, and set
//!    `k_g = floor(sum_q IoU).clamp(1, M_g)` (further capped by [`SimOtaConfig::max_k`] if set). Each GT
//!    selects its `k_g` lowest-cost candidates, ties broken by smaller anchor index.
//! 4. **Conflict resolution.** Any anchor selected by multiple GTs is assigned to the GT with the strictly
//!    lowest `cost(g, a)` among the GTs that selected it, ties broken by smaller GT index.
//!
//! Degenerate GT boxes (`width <= 0` or `height <= 0`) receive `0` candidates (`dynamic_k[g] = 0`) without
//! disturbing valid GTs. Non-finite model predictions, anchor coordinates, or GT coordinates set
//! [`SimOtaAssignment::inputs_finite`] to `false` and return an empty, all-finite assignment.

use serde::{Deserialize, Serialize};

use crate::assign::Assignment;
use crate::boxes::Xyxy;
use crate::error::{Error, Result};

/// Hyper-parameters for [`assign_simota`] (Ge et al. YOLOX 2021).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SimOtaConfig {
    /// Weight of the pairwise classification BCE-with-logits cost (default `1.0`).
    pub cls_weight: f32,
    /// Weight of the pairwise `-ln(iou + iou_eps)` regression cost (default `3.0`).
    pub iou_weight: f32,
    /// Number of top-IoU candidate anchors per GT summed to compute dynamic `k` (default `10`).
    pub top_q: usize,
    /// Optional center-prior half-width in stride units (default `Some(2.5)` for a `5x5` stride window).
    pub center_radius: Option<f32>,
    /// Whether anchors inside the GT box are included in the spatial prior (default `true`).
    pub in_box: bool,
    /// Additive penalty for candidate anchors outside the strict `in_box && in_center` intersection when
    /// both priors are active (default `100_000.0`).
    pub non_prior_cost: f32,
    /// Strict boundary margin `>= 0` for the in-box containment test (default `0.0`).
    pub eps: f32,
    /// Positive floor inside `-ln(iou + iou_eps)` so zero-IoU pairs remain finite (default `1e-8`).
    pub iou_eps: f32,
    /// Optional upper cap on dynamic `k` per GT (`None` = bounded only by candidate count).
    pub max_k: Option<usize>,
}

impl Default for SimOtaConfig {
    fn default() -> Self {
        Self {
            cls_weight: 1.0,
            iou_weight: 3.0,
            top_q: 10,
            center_radius: Some(2.5),
            in_box: true,
            non_prior_cost: 100_000.0,
            eps: 0.0,
            iou_eps: 1e-8,
            max_k: None,
        }
    }
}

impl SimOtaConfig {
    /// Validate all SimOTA hyper-parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.cls_weight.is_finite() && self.cls_weight >= 0.0) {
            return Err(Error::Config(format!(
                "simota.cls_weight must be finite and >= 0, got {}",
                self.cls_weight
            )));
        }
        if !(self.iou_weight.is_finite() && self.iou_weight >= 0.0) {
            return Err(Error::Config(format!(
                "simota.iou_weight must be finite and >= 0, got {}",
                self.iou_weight
            )));
        }
        if self.top_q == 0 {
            return Err(Error::Config("simota.top_q must be >= 1, got 0".into()));
        }
        if let Some(r) = self.center_radius
            && !(r.is_finite() && r > 0.0)
        {
            return Err(Error::Config(format!("simota.center_radius must be finite and > 0, got {r}")));
        }
        if !(self.non_prior_cost.is_finite() && self.non_prior_cost >= 0.0) {
            return Err(Error::Config(format!(
                "simota.non_prior_cost must be finite and >= 0, got {}",
                self.non_prior_cost
            )));
        }
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return Err(Error::Config(format!("simota.eps must be finite and >= 0, got {}", self.eps)));
        }
        if !(self.iou_eps.is_finite() && self.iou_eps > 0.0) {
            return Err(Error::Config(format!(
                "simota.iou_eps must be finite and > 0, got {}",
                self.iou_eps
            )));
        }
        if let Some(mk) = self.max_k
            && mk == 0
        {
            return Err(Error::Config("simota.max_k must be >= 1 when set, got 0".into()));
        }
        Ok(())
    }
}

/// Per-anchor assignment produced by [`assign_simota`].
///
/// Entries where `fg_mask[a]` is `false` are background (`target_gt[a] = 0`, `target_labels[a] = num_classes`,
/// `target_bboxes[a] = Xyxy::default()`, `target_ious[a] = 0.0`, `matched_costs[a] = 0.0`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SimOtaAssignment {
    /// Whether each anchor is foreground (`[A]`).
    pub fg_mask: Vec<bool>,
    /// Assigned GT index per anchor (`[A]`).
    pub target_gt: Vec<usize>,
    /// Assigned class id per anchor (`[A]`, `num_classes` for background).
    pub target_labels: Vec<u32>,
    /// Assigned GT box per anchor in input pixels (`[A]`).
    pub target_bboxes: Vec<Xyxy>,
    /// IoU between each foreground anchor's predicted box and its assigned GT (`[A]`, in `[0, 1]`).
    /// Directly usable as the continuous quality target for Varifocal Loss or Quality Focal Loss.
    pub target_ious: Vec<f32>,
    /// Pairwise SimOTA matching cost for each foreground anchor (`[A]`).
    pub matched_costs: Vec<f32>,
    /// Dynamic `k` chosen per GT prior to conflict resolution (`[G]`; `0` for degenerate or unmatched GTs).
    pub dynamic_k: Vec<usize>,
    /// Total foreground anchors after conflict resolution.
    pub num_fg: usize,
    /// `true` when all prediction, anchor, and GT inputs were finite; `false` skips assignment cleanly.
    pub inputs_finite: bool,
}

impl SimOtaAssignment {
    fn empty(num_anchors: usize, num_gts: usize, num_classes: usize, inputs_finite: bool) -> Self {
        Self {
            fg_mask: vec![false; num_anchors],
            target_gt: vec![0; num_anchors],
            target_labels: vec![num_classes as u32; num_anchors],
            target_bboxes: vec![Xyxy::default(); num_anchors],
            target_ious: vec![0.0; num_anchors],
            matched_costs: vec![0.0; num_anchors],
            dynamic_k: vec![0; num_gts],
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
    /// `(a, target_labels[a])`, `0.0` elsewhere) ready for [`crate::loss::varifocal_loss`] or
    /// [`crate::loss::quality_focal_loss`].
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

/// Numerically stable `softplus(x) = max(x, 0) + ln(1 + exp(-|x|))` in `f64`.
#[inline]
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Exact `f64` plain IoU in `[0, 1]` between two `Xyxy` boxes (`0.0` for empty/degenerate/disjoint boxes).
#[inline]
fn box_iou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
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

/// Cancellation-free multi-label BCE-with-logits cost in `f64` for one-hot target class `gt_class`:
/// `softplus(-x[gt_class]) + sum_{c != gt_class} softplus(x[c])`.
#[inline]
fn one_hot_bce_cost_f64(logits: &[f32], gt_class: usize) -> f64 {
    let mut sum = 0.0f64;
    for (c, &x) in logits.iter().enumerate() {
        if c == gt_class {
            sum += softplus_f64(-(x as f64));
        } else {
            sum += softplus_f64(x as f64);
        }
    }
    sum
}

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Candidate record for a single `(gt, anchor)` pair.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    anchor: usize,
    iou: f64,
    cost: f64,
}

/// Run SimOTA dynamic-`k` label assignment for one image.
///
/// - `pd_cls_logits`: `[A * num_classes]` row-major raw classification logits (detached).
/// - `num_classes`: number of foreground classes `C`.
/// - `pd_bboxes`: `[A]` predicted boxes in input pixels (`Xyxy`).
/// - `anchor_points`: `[A]` anchor centres `(x, y)` in input pixels.
/// - `anchor_strides`: `[A]` anchor strides in input pixels (must be finite and `> 0`).
/// - `gt_bboxes`: `[G]` ground-truth boxes in input pixels.
/// - `gt_labels`: `[G]` class ids (`< num_classes`).
/// - `cfg`: [`SimOtaConfig`] hyper-parameters.
///
/// # Errors
/// Returns [`Error::Config`] on invalid config parameters, slice length mismatches, out-of-range
/// `gt_labels`, `num_classes == 0` with non-empty GTs, or non-finite / non-positive `anchor_strides`.
#[allow(clippy::too_many_arguments)]
pub fn assign_simota(
    pd_cls_logits: &[f32],
    num_classes: usize,
    pd_bboxes: &[Xyxy],
    anchor_points: &[[f32; 2]],
    anchor_strides: &[f32],
    gt_bboxes: &[Xyxy],
    gt_labels: &[u32],
    cfg: &SimOtaConfig,
) -> Result<SimOtaAssignment> {
    cfg.validate()?;

    let n_a = pd_bboxes.len();
    let n_g = gt_bboxes.len();
    let expected_cls_len = n_a
        .checked_mul(num_classes)
        .ok_or_else(|| Error::Config(format!("simota: A ({n_a}) * num_classes ({num_classes}) overflows")))?;
    if pd_cls_logits.len() != expected_cls_len {
        return Err(Error::Config(format!(
            "simota: pd_cls_logits length {} != A * num_classes ({n_a} * {num_classes})",
            pd_cls_logits.len()
        )));
    }
    if anchor_points.len() != n_a {
        return Err(Error::Config(format!(
            "simota: anchor_points length {} != A ({n_a})",
            anchor_points.len()
        )));
    }
    if anchor_strides.len() != n_a {
        return Err(Error::Config(format!(
            "simota: anchor_strides length {} != A ({n_a})",
            anchor_strides.len()
        )));
    }
    if gt_labels.len() != n_g {
        return Err(Error::Config(format!("simota: gt_labels length {} != G ({n_g})", gt_labels.len())));
    }
    for (a, &s) in anchor_strides.iter().enumerate() {
        if !(s.is_finite() && s > 0.0) {
            return Err(Error::Config(format!(
                "simota: anchor_strides[{a}] must be finite and > 0, got {s}"
            )));
        }
    }
    if n_g > 0 && num_classes == 0 {
        return Err(Error::Config("simota: num_classes must be >= 1 when gt_bboxes is non-empty".into()));
    }
    for (g, &l) in gt_labels.iter().enumerate() {
        if (l as usize) >= num_classes {
            return Err(Error::Config(format!(
                "simota: gt_labels[{g}] = {l} >= num_classes ({num_classes})"
            )));
        }
    }

    // Check finiteness of all runtime numeric inputs before touching empty/non-empty paths so a
    // corrupted prediction batch always reports `inputs_finite = false` cleanly.
    let all_finite = pd_cls_logits.iter().all(|v| v.is_finite())
        && pd_bboxes.iter().all(box_is_finite)
        && anchor_points.iter().all(|p| p[0].is_finite() && p[1].is_finite())
        && gt_bboxes.iter().all(box_is_finite);
    if !all_finite {
        return Ok(SimOtaAssignment::empty(n_a, n_g, num_classes, false));
    }
    if n_g == 0 || n_a == 0 {
        return Ok(SimOtaAssignment::empty(n_a, n_g, num_classes, true));
    }

    let cls_w = cfg.cls_weight as f64;
    let iou_w = cfg.iou_weight as f64;
    let iou_eps = cfg.iou_eps as f64;
    let non_prior = cfg.non_prior_cost as f64;
    let eps = cfg.eps as f64;
    let both_priors = cfg.in_box && cfg.center_radius.is_some();

    let mut out = SimOtaAssignment::empty(n_a, n_g, num_classes, true);
    // Conflict resolution table: for each anchor, the winning `(gt_idx, cost, iou)` among GTs that selected it.
    let mut owner: Vec<Option<(usize, f64, f64)>> = vec![None; n_a];
    let mut cands: Vec<Candidate> = Vec::new();
    let mut ious_buf: Vec<(f64, usize)> = Vec::new();

    for g in 0..n_g {
        let gt = &gt_bboxes[g];
        // Skip degenerate (zero-width, zero-height, or inverted) GT boxes cleanly.
        let gw = gt.x2 as f64 - gt.x1 as f64;
        let gh = gt.y2 as f64 - gt.y1 as f64;
        if gw <= 0.0 || gh <= 0.0 {
            out.dynamic_k[g] = 0;
            continue;
        }
        let gt_cx = 0.5 * (gt.x1 as f64 + gt.x2 as f64);
        let gt_cy = 0.5 * (gt.y1 as f64 + gt.y2 as f64);
        let gt_class = gt_labels[g] as usize;

        cands.clear();
        for a in 0..n_a {
            let ax = anchor_points[a][0] as f64;
            let ay = anchor_points[a][1] as f64;
            let in_box = cfg.in_box
                && (ax - gt.x1 as f64).min(ay - gt.y1 as f64).min(gt.x2 as f64 - ax).min(gt.y2 as f64 - ay)
                    > eps;
            let in_center = if let Some(r) = cfg.center_radius {
                let half = (r as f64) * (anchor_strides[a] as f64);
                (ax - gt_cx).abs() <= half && (ay - gt_cy).abs() <= half
            } else {
                false
            };

            let is_candidate = match (cfg.in_box, cfg.center_radius.is_some()) {
                (true, true) => in_box || in_center,
                (true, false) => in_box,
                (false, true) => in_center,
                (false, false) => true,
            };
            if !is_candidate {
                continue;
            }

            let iou = box_iou_f64(gt, &pd_bboxes[a]);
            let iou_cost = -(iou + iou_eps).ln();
            let cls_cost = if cls_w == 0.0 {
                0.0
            } else {
                let row = &pd_cls_logits[a * num_classes..(a + 1) * num_classes];
                one_hot_bce_cost_f64(row, gt_class)
            };
            let penalty = if both_priors && !(in_box && in_center) { non_prior } else { 0.0 };
            let cost = cls_w * cls_cost + iou_w * iou_cost + penalty;
            cands.push(Candidate { anchor: a, iou, cost });
        }

        let n_cands = cands.len();
        if n_cands == 0 {
            out.dynamic_k[g] = 0;
            continue;
        }

        // Dynamic-k: sum the top-q highest candidate IoUs in deterministic order and clamp to [1, n_cands].
        ious_buf.clear();
        ious_buf.extend(cands.iter().map(|c| (c.iou, c.anchor)));
        let q = cfg.top_q.min(n_cands);
        ious_buf.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
        let iou_sum: f64 = ious_buf[..q].iter().map(|&(iou, _)| iou).sum();
        let mut k = (iou_sum.floor() as usize).clamp(1, n_cands);
        if let Some(mk) = cfg.max_k {
            k = k.min(mk.min(n_cands));
        }
        out.dynamic_k[g] = k;

        // Pick the k lowest-cost candidates for GT g (ties broken deterministically by smaller anchor index).
        if n_cands > k {
            cands.select_nth_unstable_by(k - 1, |x, y| {
                x.cost.total_cmp(&y.cost).then_with(|| x.anchor.cmp(&y.anchor))
            });
            cands.truncate(k);
        }

        // Conflict resolution: because g iterates in ascending order 0..n_g, replacing only on strictly
        // smaller cost breaks equal-cost ties deterministically in favor of the smaller GT index.
        for &c in &cands {
            let slot = &mut owner[c.anchor];
            if slot.is_none_or(|(_, best_cost, _)| c.cost < best_cost) {
                *slot = Some((g, c.cost, c.iou));
            }
        }
    }

    for (a, slot) in owner.into_iter().enumerate() {
        if let Some((g, cost, iou)) = slot {
            out.fg_mask[a] = true;
            out.target_gt[a] = g;
            out.target_labels[a] = gt_labels[g];
            out.target_bboxes[a] = gt_bboxes[g];
            out.target_ious[a] = iou as f32;
            out.matched_costs[a] = cost as f32;
            out.num_fg += 1;
        }
    }
    Ok(out)
}
