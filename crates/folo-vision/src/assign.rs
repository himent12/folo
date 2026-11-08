//! The task-aligned assigner (TAL, from TOOD; as used by YOLOv8) and the DFL target encoding, on plain arrays.
//!
//! For one image with `A` anchors, `C` classes and `G` ground truths:
//!
//! 1. **Candidates.** Anchor `a` is a candidate for gt `g` when its centre lies inside the gt box by more than
//!    [`TalConfig::eps`] on every side.
//! 2. **Alignment.** For candidates, `overlap = max(CIoU(gt, pred_box), 0)` (CIoU with Ultralytics' `eps = 1e-7`
//!    terms, so tiny boxes get the same overlaps as upstream) and
//!    `align = score[a, class(g)]^alpha * overlap^beta` (`alpha = 0.5`, `beta = 6`). Non-candidates have both 0.
//! 3. **Top-k.** Each gt keeps its `topk = 10` candidates with the largest `align` (ties: lower anchor index).
//!    *Parity exception:* Ultralytics runs `torch.topk` over **all** anchors (non-candidates scored 0) and only then
//!    intersects with the candidate mask, so when a gt has fewer than `topk` candidates with non-zero alignment,
//!    which zero-alignment anchors fill the remaining slots depends on `torch.topk`'s unspecified tie order (and
//!    differs between CPU and CUDA). FOLO ranks only the candidates, deterministically, so such a gt keeps its
//!    zero-alignment candidates. Those anchors get a target score of exactly 0, so with Ultralytics' loss (box and
//!    DFL terms weighted by the target score, BCE towards 0) they contribute the same as background; only
//!    `fg_mask` / `num_fg` can differ.
//! 4. **Conflicts.** An anchor selected by several gts goes to the gt with the largest `overlap` over *all* gts
//!    (first on ties), exactly like Ultralytics' `select_highest_overlaps`.
//! 5. **Targets.** Each foreground anchor gets its gt's class and box, and a soft score
//!    `align[g, a] * max_overlap[g] / (max_align[g] + eps)`, where the maxima run over gt `g`'s final positive
//!    anchors (Ultralytics' normalisation, so the best-aligned anchor of each gt gets that gt's best IoU).
//!
//! All boxes and points must be in the same units (Ultralytics uses input-image pixels: decoded predictions and
//! anchor points multiplied by their stride). Scores are post-sigmoid.

use crate::boxes::Xyxy;
use crate::decode::Anchors;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// `G * A` above which candidate scoring runs in parallel over gts.
const PAR_WORK: usize = 1 << 17;

/// Assigner hyper-parameters. Defaults are YOLOv8's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TalConfig {
    /// Candidates kept per gt.
    pub topk: usize,
    /// Exponent of the classification score in the alignment metric.
    pub alpha: f32,
    /// Exponent of the IoU in the alignment metric.
    pub beta: f32,
    /// Margin for the anchor-inside-gt test and the normalisation denominator.
    pub eps: f32,
}

impl Default for TalConfig {
    fn default() -> Self {
        Self { topk: 10, alpha: 0.5, beta: 6.0, eps: 1e-9 }
    }
}

/// Per-anchor assignment targets. Entries where `fg_mask` is false are background: `target_gt = 0`,
/// `target_labels = num_classes`, `target_bboxes` the zero box and `target_scores = 0`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Assignment {
    /// Whether the anchor is foreground.
    pub fg_mask: Vec<bool>,
    /// Index of the assigned gt.
    pub target_gt: Vec<usize>,
    /// Class of the assigned gt.
    pub target_labels: Vec<u32>,
    /// Box of the assigned gt.
    pub target_bboxes: Vec<Xyxy>,
    /// Normalised alignment score for the target class (the soft classification target).
    pub target_scores: Vec<f32>,
    /// Number of foreground anchors.
    pub num_fg: usize,
}

impl Assignment {
    /// The dense classification target `[A, num_classes]`: `target_scores[a]` at `target_labels[a]`, 0 elsewhere.
    pub fn dense_scores(&self, num_classes: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.fg_mask.len() * num_classes];
        for (a, &fg) in self.fg_mask.iter().enumerate() {
            if fg {
                out[a * num_classes + self.target_labels[a] as usize] = self.target_scores[a];
            }
        }
        out
    }
}

/// Run the task-aligned assigner for one image.
///
/// - `pd_scores`: `[A, num_classes]` row-major, post-sigmoid.
/// - `pd_bboxes`: `[A]` predicted boxes, `xyxy`.
/// - `anchor_points`: `[A]` anchor centres `(x, y)`.
/// - `gt_bboxes`, `gt_labels`: `[G]`.
///
/// # Panics
/// On shape mismatches or a gt label `>= num_classes`.
pub fn assign(
    pd_scores: &[f32],
    num_classes: usize,
    pd_bboxes: &[Xyxy],
    anchor_points: &[[f32; 2]],
    gt_bboxes: &[Xyxy],
    gt_labels: &[u32],
    cfg: &TalConfig,
) -> Assignment {
    let n_a = pd_bboxes.len();
    let n_g = gt_bboxes.len();
    assert_eq!(pd_scores.len(), n_a * num_classes, "assign: pd_scores must be [A, num_classes]");
    assert_eq!(anchor_points.len(), n_a, "assign: anchor_points must be [A, 2]");
    assert_eq!(gt_labels.len(), n_g, "assign: gt_labels must be [G]");
    for &l in gt_labels {
        assert!((l as usize) < num_classes, "assign: gt label {l} >= num_classes {num_classes}");
    }

    let mut out = Assignment {
        fg_mask: vec![false; n_a],
        target_gt: vec![0; n_a],
        target_labels: vec![num_classes as u32; n_a],
        target_bboxes: vec![Xyxy::default(); n_a],
        target_scores: vec![0.0; n_a],
        num_fg: 0,
    };
    if n_g == 0 || n_a == 0 {
        return out;
    }

    // (align, overlap) of gt g and a candidate anchor a.
    let metrics = |g: usize, a: usize| -> (f32, f32) {
        let ov = ciou_ultralytics(&gt_bboxes[g], &pd_bboxes[a]).max(0.0);
        let s = pd_scores[a * num_classes + gt_labels[g] as usize];
        (s.powf(cfg.alpha) * ov.powf(cfg.beta), ov)
    };
    // The same, zero for non-candidates.
    let pair = |g: usize, a: usize| -> (f32, f32) {
        let p = anchor_points[a];
        if gt_bboxes[g].contains_point(p[0], p[1], cfg.eps) { metrics(g, a) } else { (0.0, 0.0) }
    };

    // Steps 1-3: per gt, the top-k candidates by alignment.
    let topk_for = |g: usize| -> Vec<u32> {
        let mut cands: Vec<(f32, u32)> = Vec::new();
        let gb = gt_bboxes[g];
        for (a, p) in anchor_points.iter().enumerate() {
            if gb.contains_point(p[0], p[1], cfg.eps) {
                cands.push((metrics(g, a).0, a as u32));
            }
        }
        let by_rank = |x: &(f32, u32), y: &(f32, u32)| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1));
        if cands.len() > cfg.topk {
            if cfg.topk == 0 {
                return Vec::new();
            }
            cands.select_nth_unstable_by(cfg.topk - 1, by_rank);
            cands.truncate(cfg.topk);
        }
        cands.into_iter().map(|c| c.1).collect()
    };
    let selected: Vec<Vec<u32>> = if n_g * n_a >= PAR_WORK {
        (0..n_g).into_par_iter().map(topk_for).collect()
    } else {
        (0..n_g).map(topk_for).collect()
    };

    // Step 4: resolve anchors selected by several gts by highest overlap over all gts.
    let mut count = vec![0u32; n_a];
    for (g, sel) in selected.iter().enumerate() {
        for &a in sel {
            count[a as usize] += 1;
            out.target_gt[a as usize] = g;
        }
    }
    // (anchor, gt, align) of the final positives.
    let mut fg: Vec<(usize, usize, f32)> = Vec::new();
    let mut max_align = vec![0.0f32; n_g];
    let mut max_overlap = vec![0.0f32; n_g];
    for (a, &n) in count.iter().enumerate() {
        if n == 0 {
            continue;
        }
        if n > 1 {
            let (mut best_g, mut best) = (0, f32::NEG_INFINITY);
            for g in 0..n_g {
                let ov = pair(g, a).1;
                if ov > best {
                    best = ov;
                    best_g = g;
                }
            }
            out.target_gt[a] = best_g;
        }
        let g = out.target_gt[a];
        let (al, ov) = pair(g, a);
        max_align[g] = max_align[g].max(al);
        max_overlap[g] = max_overlap[g].max(ov);
        fg.push((a, g, al));
    }

    // Step 5: per-gt normalisation over the final positives.
    out.num_fg = fg.len();
    for (a, g, al) in fg {
        out.fg_mask[a] = true;
        out.target_labels[a] = gt_labels[g];
        out.target_bboxes[a] = gt_bboxes[g];
        out.target_scores[a] = al * max_overlap[g] / (max_align[g] + cfg.eps);
    }
    out
}

/// CIoU exactly as Ultralytics' `bbox_iou(box1, box2, xywh=False, CIoU=True)` computes it, including its
/// `eps = 1e-7` terms (added to both heights, the union and the squared enclosing diagonal). Unlike
/// [`crate::boxes::iou`], identical boxes score slightly below 1, noticeably so for boxes of a few 1e-3 pixels.
fn ciou_ultralytics(b1: &Xyxy, b2: &Xyxy) -> f32 {
    const EPS: f32 = 1e-7;
    let (w1, h1) = (b1.x2 - b1.x1, b1.y2 - b1.y1 + EPS);
    let (w2, h2) = (b2.x2 - b2.x1, b2.y2 - b2.y1 + EPS);
    let inter =
        (b1.x2.min(b2.x2) - b1.x1.max(b2.x1)).max(0.0) * (b1.y2.min(b2.y2) - b1.y1.max(b2.y1)).max(0.0);
    let union = w1 * h1 + w2 * h2 - inter + EPS;
    let iou = inter / union;
    let cw = b1.x2.max(b2.x2) - b1.x1.min(b2.x1);
    let ch = b1.y2.max(b2.y2) - b1.y1.min(b2.y1);
    let c2 = cw * cw + ch * ch + EPS;
    let dx = b2.x1 + b2.x2 - b1.x1 - b1.x2;
    let dy = b2.y1 + b2.y2 - b1.y1 - b1.y2;
    let rho2 = (dx * dx + dy * dy) / 4.0;
    // atan(w / h), kept finite if a malformed box makes h exactly 0.
    let at = |w: f32, h: f32| if h != 0.0 { (w / h).atan() } else { w.atan2(h) };
    let d = at(w2, h2) - at(w1, h1);
    let v = (4.0 / (std::f32::consts::PI * std::f32::consts::PI)) * d * d;
    let alpha = v / (v - iou + (1.0 + EPS));
    iou - (rho2 / c2 + v * alpha)
}

/// Box → `ltrb` distances from `anchor` for the DFL target, clamped to `[0, reg_max - 1 - 0.01]`. Both inputs are in
/// stride units (grid cells), as in Ultralytics' `bbox2dist(anchor_points, target_bboxes / stride, reg_max - 1)`.
#[inline]
pub fn bbox2dist_clamped(anchor: [f32; 2], b: &Xyxy, reg_max: usize) -> [f32; 4] {
    let hi = reg_max as f32 - 1.0 - 0.01;
    crate::decode::bbox2dist(anchor, b).map(|d| d.clamp(0.0, hi))
}

/// DFL targets `[A, 4]` for an assignment: for foreground anchors, the clamped `ltrb` distances (stride units) of the
/// target box from the anchor point; zeros for background.
///
/// `anchors` gives the points in grid units and the strides; `assignment.target_bboxes` are in input pixels.
pub fn dfl_targets(assignment: &Assignment, anchors: &Anchors, reg_max: usize) -> Vec<[f32; 4]> {
    assert_eq!(assignment.fg_mask.len(), anchors.len(), "dfl_targets: anchor count mismatch");
    (0..anchors.len())
        .map(|a| {
            if !assignment.fg_mask[a] {
                return [0.0; 4];
            }
            let inv = 1.0 / anchors.strides[a];
            bbox2dist_clamped(anchors.points[a], &assignment.target_bboxes[a].scale(inv, inv), reg_max)
        })
        .collect()
}

/// Split a continuous DFL target `t` into its two neighbouring bins and weights: `(left, w_left, w_right)` with
/// `left = floor(t)`, `w_left = left + 1 - t`, `w_right = 1 - w_left` (bin `left + 1` gets `w_right`).
#[inline]
pub fn dfl_bins(t: f32) -> (usize, f32, f32) {
    let left = t.max(0.0).floor();
    let wl = left + 1.0 - t;
    (left as usize, wl, 1.0 - wl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ciou_matches_ultralytics_eps() {
        // Identical 0.001-pixel boxes: Ultralytics' eps terms give IoU = 1e-6 / (2 * 1e-3 * (1e-3 + 1e-7) - 1e-6 + 1e-7).
        let b = Xyxy::new(0.0, 0.0, 0.001, 0.001);
        let want = 1e-6 / (2.0 * 1e-3 * (1e-3 + 1e-7) - 1e-6 + 1e-7);
        assert!((ciou_ultralytics(&b, &b) as f64 - want).abs() < 1e-4, "{}", ciou_ultralytics(&b, &b));
        // So the TAL target is that IoU, not 1 (score^alpha cancels in the normalisation).
        let a = assign(&[0.5], 1, &[b], &[[0.0005, 0.0005]], &[b], &[0], &TalConfig::default());
        assert!((a.target_scores[0] as f64 - want).abs() < 1e-4, "{}", a.target_scores[0]);
        // Normal-sized boxes agree with the eps-free CIoU.
        let (p, q) = (Xyxy::new(0.0, 0.0, 20.0, 10.0), Xyxy::new(5.0, 0.0, 25.0, 12.0));
        let exact = crate::boxes::iou(&p, &q, crate::boxes::IouKind::Ciou);
        assert!((ciou_ultralytics(&p, &q) - exact).abs() < 1e-5);
    }

    #[test]
    fn dfl_encoding() {
        let a = [5.5, 5.5];
        let b = Xyxy::new(2.0, 5.0, 30.0, 6.0);
        // Raw: l 3.5, t 0.5, r 24.5, b 0.5; r clamps to 16 - 1 - 0.01.
        assert_eq!(bbox2dist_clamped(a, &b, 16), [3.5, 0.5, 14.99, 0.5]);
        // Box not containing the anchor clamps at 0.
        assert_eq!(bbox2dist_clamped(a, &Xyxy::new(6.0, 6.0, 7.0, 7.0), 16)[..2], [0.0, 0.0]);
        let (l, wl, wr) = dfl_bins(3.25);
        assert_eq!((l, wl, wr), (3, 0.75, 0.25));
        let (l, wl, wr) = dfl_bins(14.99);
        assert_eq!(l, 14);
        assert!((wl - 0.01).abs() < 1e-5 && (wr - 0.99).abs() < 1e-5);
    }
}
