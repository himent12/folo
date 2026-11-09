//! The task-aligned assigner (TAL, from TOOD; as used by YOLOv8) and the DFL target encoding, on plain arrays.
//!
//! This is `docs/MODEL.md` §7.3 (assigner) and the DFL target rule of §7.4. For one image with `A` anchors, `C`
//! classes and `G` ground truths, with every box and point in **input pixels**:
//!
//! 1. **Candidates.** Each gt box is copied and every side shorter than [`TalConfig::min_box_size`] (16 px) is
//!    widened to it about the box centre ([`candidate_box`]). Anchor `a` is a candidate of gt `g` iff its point is
//!    inside that enlarged box by more than [`TalConfig::eps`] on all four sides. The enlarged box is used for this
//!    step only.
//! 2. **Metric** (candidates only). `u = max(CIoU(gt, pred_box), 0)` in the exact §7.2 form ([`ciou_eps`]) and
//!    `m = score[a, class(g)]^alpha * u^beta` (`alpha = 0.5`, `beta = 6`). Non-candidates have `u = m = 0`.
//! 3. **Top-k.** Each gt selects its `min(topk, #candidates)` candidates with the largest `m`, ties broken by the
//!    smaller anchor index. Zero-metric candidates are eligible.
//! 4. **Conflicts.** An anchor selected by several gts goes to the one with the largest `u` **among the gts that
//!    selected it**, ties broken by the smaller gt index. With [`TalConfig::topk2`] (the one-to-one head's
//!    assignment, MODEL.md §6.6: `topk = 7`, `topk2 = 1`), each gt then keeps only its `topk2` highest-metric
//!    positives (ties: smaller anchor); the anchors it drops become background.
//! 5. **Soft targets.** Over each gt's final positives `P_g`: `t[a] = m(g, a) * u_max / (m_max + eps)` with
//!    `m_max`, `u_max` the maxima of `m` and `u` over `P_g`, so the best-aligned anchor of each gt gets (about) that
//!    gt's best overlap.
//! 6. **Box targets.** The original (not enlarged) gt box; [`dfl_targets`] converts it to grid units.
//!
//! **Differences from Ultralytics' `TaskAlignedAssigner`** (all deliberate, MODEL.md §1): it has no small-box
//! enlargement; it runs `torch.topk` over *all* anchors and only then intersects with the candidate mask, so which
//! zero-metric anchors fill a gt's slots depends on `torch.topk`'s unspecified tie order (FOLO ranks only the
//! candidates, deterministically); and it resolves conflicts by the largest overlap over *all* gts, so an anchor can
//! end up with a gt that never selected it. Zero-metric positives get a target score of exactly 0, so they contribute
//! like background to the classification and box terms; only `fg_mask` / `num_fg` see them.

use crate::boxes::{Xyxy, ciou_eps};
use crate::decode::Anchors;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// `G * A` above which candidate scoring runs in parallel over gts.
const PAR_WORK: usize = 1 << 17;

/// Assigner hyper-parameters. Defaults are `docs/MODEL.md` §7.3's.
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
    /// Candidate selection widens every gt side shorter than this (pixels) to this length about the box centre
    /// (16, the P4 stride). `0` disables the enlargement.
    pub min_box_size: f32,
    /// After conflict resolution each gt keeps only this many of its positives, the highest-metric ones
    /// (`Some(1)` for the one-to-one head, MODEL.md §6.6). `None` (the default) keeps them all.
    pub topk2: Option<usize>,
}

impl Default for TalConfig {
    fn default() -> Self {
        Self { topk: 10, alpha: 0.5, beta: 6.0, eps: 1e-9, min_box_size: 16.0, topk2: None }
    }
}

impl TalConfig {
    /// The one-to-one head's assignment (MODEL.md §6.6): §7.3 with `topk = 7`, then `topk2 = 1`.
    pub fn o2o() -> Self {
        Self { topk: 7, topk2: Some(1), ..Self::default() }
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
    /// Box of the assigned gt (the original box, not the enlarged candidate box), in input pixels.
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

/// The box used for candidate selection (§7.3 step 1): `b` with every side shorter than `min_size` set to
/// `min_size` about the box centre. Sides at least `min_size` long are unchanged.
#[inline]
pub fn candidate_box(b: &Xyxy, min_size: f32) -> Xyxy {
    let widen = |lo: f32, hi: f32| {
        if hi - lo < min_size {
            let c = (lo + hi) * 0.5;
            (c - min_size * 0.5, c + min_size * 0.5)
        } else {
            (lo, hi)
        }
    };
    let ((x1, x2), (y1, y2)) = (widen(b.x1, b.x2), widen(b.y1, b.y2));
    Xyxy::new(x1, y1, x2, y2)
}

/// One anchor selected by a gt, with its metric `m` and overlap `u` for that gt.
#[derive(Clone, Copy, Debug)]
struct Pick {
    anchor: u32,
    m: f32,
    u: f32,
}

/// Run the task-aligned assigner for one image.
///
/// - `pd_scores`: `[A, num_classes]` row-major, post-sigmoid, detached.
/// - `pd_bboxes`: `[A]` predicted boxes, `xyxy`, in input pixels (the decode times the stride).
/// - `anchor_points`: `[A]` anchor centres `(x, y)` in input pixels ([`Anchors::points_px`]).
/// - `gt_bboxes`, `gt_labels`: `[G]`, boxes in input pixels.
///
/// Everything must be in pixels because [`TalConfig::min_box_size`] is.
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
    if n_g == 0 || n_a == 0 || cfg.topk == 0 {
        return out;
    }

    // Steps 1-3: per gt, its candidates (against the enlarged box), their metrics, and the top-k by `m`.
    let topk_for = |g: usize| -> Vec<Pick> {
        let cb = candidate_box(&gt_bboxes[g], cfg.min_box_size);
        let class = gt_labels[g] as usize;
        let mut cands: Vec<Pick> = Vec::new();
        for (a, p) in anchor_points.iter().enumerate() {
            if cb.contains_point(p[0], p[1], cfg.eps) {
                let u = ciou_eps(&gt_bboxes[g], &pd_bboxes[a]).max(0.0);
                let m = pd_scores[a * num_classes + class].powf(cfg.alpha) * u.powf(cfg.beta);
                cands.push(Pick { anchor: a as u32, m, u });
            }
        }
        if cands.len() > cfg.topk {
            let by_rank = |x: &Pick, y: &Pick| y.m.total_cmp(&x.m).then(x.anchor.cmp(&y.anchor));
            cands.select_nth_unstable_by(cfg.topk - 1, by_rank);
            cands.truncate(cfg.topk);
        }
        cands
    };
    let selected: Vec<Vec<Pick>> = if n_g * n_a >= PAR_WORK {
        (0..n_g).into_par_iter().map(topk_for).collect()
    } else {
        (0..n_g).map(topk_for).collect()
    };

    // Step 4: an anchor selected by several gts goes to the largest `u` among them. Gts are visited in index order
    // and only a strictly larger `u` takes over, so ties keep the smaller gt index.
    let mut owner: Vec<Option<(u32, Pick)>> = vec![None; n_a];
    for (g, picks) in selected.iter().enumerate() {
        for &p in picks {
            let slot = &mut owner[p.anchor as usize];
            if slot.is_none_or(|(_, q)| p.u > q.u) {
                *slot = Some((g as u32, p));
            }
        }
    }

    // The second filter (one-to-one assignment): each gt keeps its `topk2` best positives by (m descending, anchor
    // ascending), the order the picks were ranked in.
    if let Some(k2) = cfg.topk2 {
        let mut mine: Vec<Vec<(f32, u32)>> = vec![Vec::new(); n_g];
        for (a, o) in owner.iter().enumerate() {
            if let Some((g, p)) = o {
                mine[*g as usize].push((p.m, a as u32));
            }
        }
        for list in &mut mine {
            if list.len() > k2 {
                list.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
                for &(_, a) in &list[k2..] {
                    owner[a as usize] = None;
                }
            }
        }
    }

    // Step 5: per-gt maxima over the final positives, then the normalised soft target.
    let mut m_max = vec![0.0f32; n_g];
    let mut u_max = vec![0.0f32; n_g];
    for &(g, p) in owner.iter().flatten() {
        m_max[g as usize] = m_max[g as usize].max(p.m);
        u_max[g as usize] = u_max[g as usize].max(p.u);
    }
    for (a, o) in owner.iter().enumerate() {
        let Some((g, p)) = *o else { continue };
        let g = g as usize;
        out.fg_mask[a] = true;
        out.target_gt[a] = g;
        out.target_labels[a] = gt_labels[g];
        out.target_bboxes[a] = gt_bboxes[g];
        out.target_scores[a] = p.m * u_max[g] / (m_max[g] + cfg.eps);
        out.num_fg += 1;
    }
    out
}

/// Box → `ltrb` distances from `anchor` for the DFL target (MODEL.md §7.4), clamped to `[0, reg_max - 1 - 0.01]`
/// (`[0, 14.99]` for `reg_max = 16`), so the right-hand bin `floor(d) + 1` is at most `reg_max - 1`. Both inputs are
/// in stride units (grid cells), as in Ultralytics' `bbox2dist(anchor_points, target_bboxes / stride, reg_max - 1)`.
///
/// # Panics
/// If `reg_max < 2` (DFL needs at least two bins).
#[inline]
pub fn bbox2dist_clamped(anchor: [f32; 2], b: &Xyxy, reg_max: usize) -> [f32; 4] {
    assert!(reg_max >= 2, "bbox2dist_clamped: reg_max must be >= 2, got {reg_max}");
    let hi = reg_max as f32 - 1.0 - 0.01;
    crate::decode::bbox2dist(anchor, b).map(|d| d.clamp(0.0, hi))
}

/// DFL targets `[A, 4]` for an assignment: for foreground anchors, the clamped `ltrb` distances (stride units) of the
/// target box from the anchor point; zeros for background.
///
/// `anchors` gives the points in grid units and the strides; `assignment.target_bboxes` are in input pixels and are
/// divided by each anchor's stride.
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

/// Split a clamped DFL target `t` ([`bbox2dist_clamped`]) into its two neighbouring bins and weights:
/// `(left, w_left, w_right)` with `left = floor(t)`, `w_left = left + 1 - t`, `w_right = 1 - w_left` (bin `left + 1`
/// gets `w_right`).
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
        assert!((ciou_eps(&b, &b) as f64 - want).abs() < 1e-4, "{}", ciou_eps(&b, &b));
        // So the TAL target is that IoU, not 1 (score^alpha cancels in the normalisation).
        let a = assign(&[0.5], 1, &[b], &[[0.0005, 0.0005]], &[b], &[0], &TalConfig::default());
        assert!((a.target_scores[0] as f64 - want).abs() < 1e-4, "{}", a.target_scores[0]);
        // Normal-sized boxes agree with the eps-free CIoU.
        let (p, q) = (Xyxy::new(0.0, 0.0, 20.0, 10.0), Xyxy::new(5.0, 0.0, 25.0, 12.0));
        let exact = crate::boxes::iou(&p, &q, crate::boxes::IouKind::Ciou);
        assert!((ciou_eps(&p, &q) - exact).abs() < 1e-5);
    }

    #[test]
    fn candidate_box_enlarges_short_sides_about_the_centre() {
        // Both sides short: 6x6 centred at (8, 8) → 16x16 centred at (8, 8).
        assert_eq!(candidate_box(&Xyxy::new(5.0, 5.0, 11.0, 11.0), 16.0), Xyxy::new(0.0, 0.0, 16.0, 16.0));
        // Only the short side changes.
        assert_eq!(candidate_box(&Xyxy::new(0.0, 0.0, 40.0, 6.0), 16.0), Xyxy::new(0.0, -5.0, 40.0, 11.0));
        assert_eq!(candidate_box(&Xyxy::new(2.0, 1.0, 4.0, 41.0), 16.0), Xyxy::new(-5.0, 1.0, 11.0, 41.0));
        // Exactly 16 and larger: unchanged. min_size 0: unchanged.
        let b = Xyxy::new(3.0, 4.0, 19.0, 50.0);
        assert_eq!(candidate_box(&b, 16.0), b);
        assert_eq!(candidate_box(&Xyxy::new(5.0, 5.0, 6.0, 6.0), 0.0), Xyxy::new(5.0, 5.0, 6.0, 6.0));
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

    #[test]
    fn dfl_spec_cases() {
        // MODEL.md §10.3: δ = 3.3 puts 0.7 on bin 3 and 0.3 on bin 4.
        let (l, wl, wr) = dfl_bins(3.3);
        assert_eq!(l, 3);
        assert!((wl - 0.7).abs() < 1e-6 && (wr - 0.3).abs() < 1e-6, "{wl} {wr}");
        // δ = 15.2 clamps to 14.99: bins 14 and 15, the last bin of reg_max = 16.
        let d = bbox2dist_clamped([0.0, 0.0], &Xyxy::new(-15.2, -1.0, 15.2, 1.0), 16);
        assert_eq!(d, [14.99, 1.0, 14.99, 1.0]);
        let (l, _, wr) = dfl_bins(d[0]);
        assert_eq!(l + 1, 15);
        assert!(wr > 0.98 && wr < 1.0);
        // Another reg_max clamps at its own last bin.
        assert_eq!(bbox2dist_clamped([0.0, 0.0], &Xyxy::new(-9.0, 0.0, 0.0, 0.0), 8)[0], 6.99);
    }

    #[test]
    #[should_panic(expected = "reg_max must be >= 2")]
    fn dfl_rejects_reg_max_1() {
        bbox2dist_clamped([0.0, 0.0], &Xyxy::new(0.0, 0.0, 1.0, 1.0), 1);
    }
}
