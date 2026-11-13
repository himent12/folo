//! Cluster box fusion after NMS (docs/MODEL.md §9.4), opt-in through [`crate::nms::NmsConfig::fusion`].
//!
//! NMS keeps one candidate per object and drops the other members of its cluster: several more trained estimates
//! of the same four edges. Fusion moves each side of a kept box to a weighted mean of that side over its cluster:
//!
//! - [`BoxFusion::Vote`]: score-weighted box voting (each member weighs its class score);
//! - [`BoxFusion::Dfl`]: DFL evidence fusion, the score times the member's **per-side precision**, read from its
//!   DFL distribution ([`crate::decode::side_precision`]): an anchor near an object's left edge knows that edge
//!   sharply and the far edge vaguely, so its left edge counts more than its right.
//!
//! Scores, classes, the kept set and its order are NMS's; only the boxes move. The cluster of a kept candidate is
//! every candidate that entered NMS (after the confidence filter and the candidate cap) with its class (any class
//! under `agnostic`) whose IoU with the kept box is `>= iou`, the kept candidate included. Lane `fusion` of
//! hill-climb 3 chose the rule and the weights on 5000 train2017 images (docs/notes/hill/r3-fusion.md).

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;

/// `ε` of [`crate::decode::side_precision`], in bins²: the floor of a side's excess variance.
pub const DFL_FUSION_EPS: f32 = 0.1;

/// Default cluster IoU of [`crate::nms::NmsConfig::fusion_iou`].
pub const FUSION_IOU: f32 = 0.75;

/// How kept boxes are refined from their NMS cluster.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoxFusion {
    /// The kept boxes as NMS keeps them (the default; every published number).
    #[default]
    None,
    /// Score-weighted box voting.
    Vote,
    /// DFL evidence fusion: score times per-side precision. A caller without the DFL logits (boxes-only inputs)
    /// gives every side the same precision, which is [`BoxFusion::Vote`].
    Dfl,
}

impl BoxFusion {
    /// The name used by the CLI and in JSON (`none`, `vote`, `dfl`).
    pub fn as_str(self) -> &'static str {
        match self {
            BoxFusion::None => "none",
            BoxFusion::Vote => "vote",
            BoxFusion::Dfl => "dfl",
        }
    }
}

/// The fused boxes of the `kept` candidates (indices into `boxes` / `scores` / `classes`, the candidates that
/// entered NMS), in `kept`'s order. Member `j` weighs `scores[j] · precision(j)[side]` on each side, with
/// `precision` `None` meaning 1 (voting); the sums run in `f64` over the cluster in candidate order. A member whose
/// weight is not `> 0` (a zero score, a NaN) is skipped, and a side with no weight keeps the kept box's edge.
pub fn fuse_boxes(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    agnostic: bool,
    iou: f32,
    precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<Xyxy> {
    // Candidates grouped by class (stable: candidate order inside a class), so a cluster scans one bucket.
    let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
    if !agnostic {
        order.sort_by_key(|&i| classes[i as usize]);
    }
    let bucket = |c: u32| -> &[u32] {
        if agnostic {
            return &order;
        }
        let lo = order.partition_point(|&i| classes[i as usize] < c);
        let hi = order.partition_point(|&i| classes[i as usize] <= c);
        &order[lo..hi]
    };
    kept.iter()
        .map(|&k| {
            let b = boxes[k];
            let ba = b.area();
            let mut acc = [0f64; 4];
            let mut sum = [0f64; 4];
            for &j in bucket(classes[k]) {
                let j = j as usize;
                let c = &boxes[j];
                // The arithmetic of `nms::nms_sorted` (and `boxes::box_iou`).
                let w = (b.x2.min(c.x2) - b.x1.max(c.x1)).max(0.0);
                let h = (b.y2.min(c.y2) - b.y1.max(c.y1)).max(0.0);
                let inter = w * h;
                let union = ba + c.area() - inter;
                let o = if union > 0.0 { inter / union } else { 0.0 };
                if o.is_nan() || o < iou {
                    continue;
                }
                let s = scores[j] as f64;
                let p = precision.map_or([1.0; 4], |f| f(j));
                let x = c.to_array();
                for e in 0..4 {
                    let wt = s * p[e] as f64;
                    if wt > 0.0 {
                        acc[e] += wt * x[e] as f64;
                        sum[e] += wt;
                    }
                }
            }
            let old = b.to_array();
            let e = |i: usize| if sum[i] > 0.0 { (acc[i] / sum[i]) as f32 } else { old[i] };
            Xyxy::new(e(0), e(1), e(2), e(3))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::{side_precision, side_precisions, softmax_moments};

    fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> Xyxy {
        Xyxy::new(x1, y1, x2, y2)
    }

    /// A lone candidate (nothing else of its class reaches the cluster IoU) keeps its box exactly, voting or not.
    #[test]
    fn lone_candidate_is_unchanged() {
        let boxes = [b(10.0, 10.0, 50.0, 60.0), b(12.0, 10.0, 50.0, 60.0), b(200.0, 0.0, 240.0, 40.0)];
        // Candidate 1 overlaps 0 well but is another class; candidate 2 is far away.
        let classes = [0, 1, 0];
        let scores = [0.9, 0.8, 0.7];
        let prec = |j: usize| [1.0 + j as f32, 2.0, 3.0, 4.0];
        for p in [None, Some(&prec as &dyn Fn(usize) -> [f32; 4])] {
            assert_eq!(fuse_boxes(&[0, 2], &boxes, &scores, &classes, false, 0.75, p), [boxes[0], boxes[2]]);
        }
        // Agnostic: candidate 1 joins candidate 0's cluster (IoU 0.95).
        let got = fuse_boxes(&[0], &boxes, &scores, &classes, true, 0.75, None);
        assert!((got[0].x1 - (0.9 * 10.0 + 0.8 * 12.0) / 1.7).abs() < 1e-5, "{got:?}");
    }

    /// Two members, one sharp and one vague on each side: each side moves to its precision-and-score-weighted
    /// mean, so the result follows the sharp member side by side; voting takes the score-weighted mean.
    #[test]
    fn two_members_weigh_their_certainty_per_side() {
        let boxes = [b(10.0, 20.0, 110.0, 120.0), b(12.0, 22.0, 108.0, 118.0)];
        let (scores, classes) = ([0.8f32, 0.4], [3u32, 3]);
        // Member 0 is sharp on the left and top, member 1 on the right and bottom.
        let prec = |j: usize| if j == 0 { [9.0, 9.0, 1.0, 1.0] } else { [1.0, 1.0, 9.0, 9.0] };
        let got = fuse_boxes(&[0], &boxes, &scores, &classes, false, 0.75, Some(&prec))[0];
        let mean = |a: f32, wa: f32, c: f32, wc: f32| (a * wa + c * wc) / (wa + wc);
        let want = [
            mean(10.0, 0.8 * 9.0, 12.0, 0.4),
            mean(20.0, 0.8 * 9.0, 22.0, 0.4),
            mean(110.0, 0.8, 108.0, 0.4 * 9.0),
            mean(120.0, 0.8, 118.0, 0.4 * 9.0),
        ];
        for (g, w) in got.to_array().iter().zip(want) {
            assert!((g - w).abs() < 1e-4, "{got:?} vs {want:?}");
        }
        let vote = fuse_boxes(&[0], &boxes, &scores, &classes, false, 0.75, None)[0];
        assert!((vote.x2 - mean(110.0, 0.8, 108.0, 0.4)).abs() < 1e-4, "{vote:?}");
    }

    /// The cluster IoU is inclusive, members below it are left out, and a side with no weight (zero scores)
    /// keeps the kept box's edge. The result does not depend on where other classes' candidates sit.
    #[test]
    fn cluster_rule_and_zero_weights() {
        // IoU of [0, 0, 10, 10] and [0, 0, 10, 7.5] is 0.75 exactly.
        let boxes = [b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 7.5), b(0.0, 0.0, 10.0, 7.0)];
        let classes = [0, 0, 0];
        let got = fuse_boxes(&[0], &boxes, &[0.5, 0.5, 0.5], &classes, false, 0.75, None)[0];
        assert_eq!(got.y2, 8.75, "member 1 (IoU 0.75) is in, member 2 (0.7) is out");
        let got = fuse_boxes(&[0], &boxes, &[0.0, 0.0, 0.0], &classes, false, 0.75, None)[0];
        assert_eq!(got, boxes[0]);
        // Interleaving another class's candidates changes nothing.
        let boxes2 = [boxes[0], b(0.0, 0.0, 10.0, 9.0), boxes[1], boxes[2]];
        let got2 = fuse_boxes(&[0], &boxes2, &[0.5, 0.9, 0.5, 0.5], &[0, 1, 0, 0], false, 0.75, None)[0];
        assert_eq!(got2.y2, 8.75);
    }

    /// The side precision: a perfectly certain edge between two bins (DFL's {3: 0.7, 4: 0.3} for 3.3) has no
    /// excess variance, like one on a bin; a spread side has less precision; at stride 32 a side with the same
    /// distribution has 1/16 of its precision at stride 8 (a cluster spanning two levels trusts the finer one).
    #[test]
    fn side_precision_values() {
        let ln = |p: f32| p.ln();
        let mut row = vec![-30.0f32; 16];
        (row[3], row[4]) = (ln(0.7), ln(0.3));
        let (m, v) = softmax_moments(&row);
        assert!((m - 3.3).abs() < 1e-5 && (v - 0.21).abs() < 1e-5, "{m} {v}");
        let certain = 1.0 / (64.0 * DFL_FUSION_EPS);
        assert!((side_precision(m, v, 8.0) / certain - 1.0).abs() < 1e-3);
        assert!((side_precision(3.0, 0.0, 8.0) / certain - 1.0).abs() < 1e-6);
        // Uniform over 16 bins: variance 21.25, mean 7.5 (f = 0.5): excess 21.0.
        let (m, v) = softmax_moments(&[0.0; 16]);
        assert!((m - 7.5).abs() < 1e-5 && (v - 21.25).abs() < 1e-4);
        assert!((side_precision(m, v, 8.0) - 1.0 / (64.0 * (21.0 + DFL_FUSION_EPS))).abs() < 1e-9);
        // Two levels, one distribution.
        let mut four = row.clone();
        four.extend(row.iter().chain(&row).chain(&row));
        let (p8, p32) = (side_precisions(&four, 16, 8.0), side_precisions(&four, 16, 32.0));
        for s in 0..4 {
            assert!((p8[s] / p32[s] - 16.0).abs() < 1e-3);
        }
    }

    /// A cluster spanning two levels: two members with the same distributions and scores, one at stride 8 and one at
    /// stride 32, fuse 16 : 1 towards the stride-8 member on every side.
    #[test]
    fn cluster_spanning_two_levels() {
        let mut row = vec![0f32; 64];
        for (s, v) in row.iter_mut().enumerate() {
            *v = -((s % 16) as f32 - 4.0).powi(2) / 2.0;
        }
        let (p8, p32) = (side_precisions(&row, 16, 8.0), side_precisions(&row, 16, 32.0));
        let boxes = [b(100.0, 100.0, 200.0, 200.0), b(104.0, 96.0, 204.0, 196.0)];
        let prec = |j: usize| if j == 0 { p8 } else { p32 };
        let got = fuse_boxes(&[1], &boxes, &[0.5, 0.5], &[0, 0], false, 0.75, Some(&prec))[0];
        let want = [(16.0 * 100.0 + 104.0) / 17.0, (16.0 * 100.0 + 96.0) / 17.0];
        assert!((got.x1 - want[0]).abs() < 1e-3 && (got.y1 - want[1]).abs() < 1e-3, "{got:?}");
        assert!((got.x2 - (16.0 * 200.0 + 204.0) / 17.0).abs() < 1e-3, "{got:?}");
    }

    /// A censored side (all of its mass on the last bin: the edge may lie beyond `reg_max − 1` bins) has variance 0
    /// and so the highest precision, `1 / (s² · ε)`: no guard down-weights it. Lane `fusion` measured such a guard
    /// and it changed nothing (it fires on fewer than 0.02% of candidate sides on COCO val2017), so none is applied.
    #[test]
    fn censored_side_is_trusted_like_a_certain_one() {
        let mut censored = vec![-30.0f32; 16];
        censored[15] = 10.0;
        let (m, v) = softmax_moments(&censored);
        assert!((m - 15.0).abs() < 1e-4 && v < 1e-4, "{m} {v}");
        let p = side_precision(m, v, 8.0);
        assert!((p * 64.0 * DFL_FUSION_EPS - 1.0).abs() < 1e-3, "{p}");
    }

    /// The same inputs give the same boxes, bit for bit.
    #[test]
    fn deterministic() {
        let boxes: Vec<Xyxy> =
            (0..40).map(|i| b(i as f32 * 0.37, 5.0 - i as f32 * 0.11, 60.0 + i as f32 * 0.2, 70.0)).collect();
        let scores: Vec<f32> = (0..40).map(|i| 0.9 - i as f32 * 0.01).collect();
        let classes = vec![0u32; 40];
        let prec = |j: usize| [1.0 / (1.0 + j as f32), 2.0, 0.5, 1.0 + (j % 3) as f32];
        let kept = [0usize, 7, 21];
        let a = fuse_boxes(&kept, &boxes, &scores, &classes, false, 0.75, Some(&prec));
        assert_eq!(a, fuse_boxes(&kept, &boxes, &scores, &classes, false, 0.75, Some(&prec)));
        assert_ne!(a[0], boxes[0], "the cluster moved the box");
    }
}
