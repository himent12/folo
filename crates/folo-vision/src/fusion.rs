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
//!
//! [`ScoreRule::Consensus`], opt-in through [`crate::nms::NmsConfig::score`], uses such a cluster for the other half
//! of a detection, always within the box's own class: it scales each kept box's score by the score mass of its
//! cluster and re-sorts ([`consensus_scores`], docs/MODEL.md §9.4, docs/notes/hill/r6-consensus.md). Boxes, classes
//! and the kept set do not change.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;

/// `ε` of [`crate::decode::side_precision`], in bins²: the floor of a side's excess variance.
pub const DFL_FUSION_EPS: f32 = 0.1;

/// Default cluster IoU of [`crate::nms::NmsConfig::fusion_iou`].
pub const FUSION_IOU: f32 = 0.75;

/// The cluster IoU `τ` of [`ScoreRule::Consensus`]. Fixed: it does not follow `fusion_iou`.
pub const CONSENSUS_IOU: f32 = 0.75;

/// The cap `R` of [`ScoreRule::Consensus`]: a cluster carrying this many times the kept box's score keeps the score.
pub const CONSENSUS_CAP: f64 = 10.0;

/// The cluster IoU `τ` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_IOU: f32 = 0.75;

/// The cluster support cap `R` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_CAP: f64 = 10.0;

/// The log-support slope `α` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_ALPHA: f64 = 0.65;

/// The scale-preserving support reference `R_ref` of [`ScoreRule::Elc`] and [`ScoreRule::DflConsensus`].
pub const ELC_R_REF: f64 = 4.0;

/// The DFL excess-std slope `β` of [`ScoreRule::DflConsensus`].
pub const DFL_CONSENSUS_BETA: f64 = 0.60;

/// The scale-preserving DFL excess-std reference `σ_ref` (bins) of [`ScoreRule::DflConsensus`].
pub const DFL_CONSENSUS_SIG_REF: f64 = 0.50;

/// Lower clamp on the 4-side mean fused DFL excess std (bins) in [`ScoreRule::DflConsensus`].
const DFL_CONSENSUS_SIG_MIN: f64 = 0.1;

/// Probability clamp before taking log-odds in [`elc_scores`] and [`dfl_consensus_scores`].
const ELC_PROB_EPS: f64 = 1e-6;

/// How the boxes NMS keeps are scored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreRule {
    /// The class score (the default; every published number).
    #[default]
    None,
    /// The class score scaled by the score mass of the box's cluster ([`consensus_scores`]). The score scale
    /// changes: a box that fired alone is divided by up to `√10`, so a threshold on it means something else.
    Consensus,
    /// Exclusive Logit-Space Consensus ([`elc_scores`]): each candidate votes only for the kept box that
    /// suppressed it in NMS, updating log-odds so high-confidence scores are protected and the `[0, 1]` scale
    /// is preserved.
    Elc,
    /// Scale-Free Exclusive Logit-Consensus with fused DFL precision ([`dfl_consensus_scores`]): adds a
    /// scale-free fused DFL excess-std term from the DFL moments already computed for [`BoxFusion::Dfl`].
    /// Falls back to [`ScoreRule::Elc`] when DFL bin precisions are unavailable.
    #[serde(rename = "dfl-consensus")]
    DflConsensus,
}

impl ScoreRule {
    /// The name used by the CLI and in JSON (`none`, `consensus`, `elc`, `dfl-consensus`).
    pub fn as_str(self) -> &'static str {
        match self {
            ScoreRule::None => "none",
            ScoreRule::Consensus => "consensus",
            ScoreRule::Elc => "elc",
            ScoreRule::DflConsensus => "dfl-consensus",
        }
    }

    /// Whether this rule uses exclusive NMS suppressor ownership and centered logit-space calibration.
    #[inline]
    pub fn is_calibrated(self) -> bool {
        matches!(self, ScoreRule::Elc | ScoreRule::DflConsensus)
    }

    /// Whether this rule uses per-anchor scale-free DFL bin precisions ([`crate::decode::bin_precision`]).
    #[inline]
    pub fn wants_bin_precisions(self) -> bool {
        self == ScoreRule::DflConsensus
    }
}

/// The `n` candidates grouped by class (stable: candidate order inside a class), so a cluster scans one bucket; with
/// `agnostic`, one bucket of every candidate.
struct Buckets<'a> {
    order: Vec<u32>,
    classes: &'a [u32],
    agnostic: bool,
}

impl<'a> Buckets<'a> {
    fn new(n: usize, classes: &'a [u32], agnostic: bool) -> Self {
        let mut order: Vec<u32> = (0..n as u32).collect();
        if !agnostic {
            order.sort_by_key(|&i| classes[i as usize]);
        }
        Buckets { order, classes, agnostic }
    }

    fn of(&self, c: u32) -> &[u32] {
        if self.agnostic {
            return &self.order;
        }
        let lo = self.order.partition_point(|&i| self.classes[i as usize] < c);
        let hi = self.order.partition_point(|&i| self.classes[i as usize] <= c);
        &self.order[lo..hi]
    }
}

/// IoU of `b` (of area `ba`) and `c`, in the arithmetic of `nms::nms_sorted` (and `boxes::box_iou`).
#[inline]
fn overlap(b: &Xyxy, ba: f32, c: &Xyxy) -> f32 {
    let w = (b.x2.min(c.x2) - b.x1.max(c.x1)).max(0.0);
    let h = (b.y2.min(c.y2) - b.y1.max(c.y1)).max(0.0);
    let inter = w * h;
    let union = ba + c.area() - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// The consensus scores of the `kept` candidates (indices into `boxes` / `scores` / `classes`, the candidates that
/// entered NMS), in `kept`'s order:
///
/// ```text
/// M_k  = Σ s_j over the candidates j of k's class with IoU(b_k, b_j) >= τ      (j = k always included)
/// s'_k = s_k · min(M_k / s_k, R)^a / R^a                                       τ = 0.75, a = 0.5, R = 10
/// ```
///
/// The sum runs in `f64` in candidate order, so the result does not depend on threads or on other classes'
/// candidates: the cluster stays within `k`'s class even when NMS is `agnostic`. A member whose score is not `> 0` or
/// whose box is not finite adds nothing, a kept box that is not finite has no members but itself (`M = s`), and a
/// kept score that is not `> 0` stays as it is.
pub fn consensus_scores(kept: &[usize], boxes: &[Xyxy], scores: &[f32], classes: &[u32]) -> Vec<f32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, false);
    kept.iter()
        .map(|&k| {
            if scores[k].is_nan() || scores[k] <= 0.0 {
                return scores[k];
            }
            let b = boxes[k];
            let ba = b.area();
            let s = scores[k] as f64;
            let mut mass = 0f64;
            // A NaN box can reach any IoU from f32 min / max, so both boxes of a pair must be finite.
            if finite(&b) {
                for &j in buckets.of(classes[k]) {
                    let j = j as usize;
                    let c = &boxes[j];
                    let member = j == k || (overlap(&b, ba, c) >= CONSENSUS_IOU && finite(c));
                    if member && scores[j] > 0.0 {
                        mass += scores[j] as f64;
                    }
                }
            } else {
                mass = s;
            }
            // a = 0.5: a square root, which IEEE 754 rounds exactly on every platform.
            (s * (mass / s).min(CONSENSUS_CAP).sqrt() / CONSENSUS_CAP.sqrt()) as f32
        })
        .collect()
}

/// Overflow-safe logistic sigmoid in `f64`.
#[inline]
fn branch_safe_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Class-local suppressor ownership of every candidate `j` relative to `kept` (assumed in descending score order):
/// each kept box owns itself, and every other candidate of class `c` is owned by the first kept box of class `c`
/// whose IoU with it is `> nms_iou` (`u32::MAX` if none). Used when NMS ran `agnostic` or when a caller invokes
/// [`elc_scores`] / [`dfl_consensus_scores`] without precomputed owners.
fn class_local_owners(
    kept: &[usize],
    boxes: &[Xyxy],
    classes: &[u32],
    buckets: &Buckets<'_>,
    nms_iou: f32,
) -> Vec<u32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let mut own = vec![u32::MAX; boxes.len()];
    for (kp, &k) in kept.iter().enumerate() {
        own[k] = kp as u32;
    }
    for (kp, &k) in kept.iter().enumerate() {
        let b = boxes[k];
        if !finite(&b) {
            continue;
        }
        let ba = b.area();
        for &j in buckets.of(classes[k]) {
            let j = j as usize;
            if own[j] == u32::MAX && finite(&boxes[j]) && overlap(&b, ba, &boxes[j]) > nms_iou {
                own[j] = kp as u32;
            }
        }
    }
    own
}

/// Exclusive Logit-Space Consensus scores of `kept` (docs/MODEL.md §9.4), in `kept`'s order:
///
/// ```text
/// M_excl,k = Σ s_j over candidates j of k's class owned by k with IoU(b_k, b_j) >= 0.75   (j = k included)
/// logit(s'_k) = logit(s_k) + 0.65 · ln(min(M_excl,k / s_k, 10) / 4.0)
/// ```
///
/// `owner` maps each candidate index to the position in `kept` of the kept box of its class that suppressed it
/// (or itself, or `u32::MAX` if unowned); when `None` (e.g. under `agnostic` NMS), class-local ownership is derived
/// from `kept` and `nms_iou`. Sums run in `f64` in candidate order for same-platform bit determinism (`f64::ln` /
/// `exp` may differ by 1 ULP across OS libms).
pub fn elc_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
) -> Vec<f32> {
    dfl_consensus_scores(kept, boxes, scores, classes, owner, nms_iou, None)
}

/// Scale-Free Exclusive Logit-Consensus with fused DFL precision (docs/MODEL.md §9.4), in `kept`'s order:
///
/// ```text
/// σ̄_bin,k = (1/4) Σ_{e=0..4} ( (Σ_{j ∈ C_excl(k)} s_j · bin_prec(j)[e]) / M_excl,k )^{-1/2}
/// logit(s'_k) = logit(s_k) + 0.65 · ln(min(M_excl,k / s_k, 10) / 4.0) − 0.60 · ln(max(σ̄_bin,k, 0.1) / 0.50)
/// ```
///
/// When `bin_precision` is `None`, the DFL term is omitted and this reduces identically to [`elc_scores`].
pub fn dfl_consensus_scores(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    owner: Option<&[u32]>,
    nms_iou: f32,
    bin_precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<f32> {
    let finite = |x: &Xyxy| x.to_array().iter().all(|v| v.is_finite());
    let buckets = Buckets::new(boxes.len(), classes, false);
    let computed_owners;
    let owners: &[u32] = match owner {
        Some(o) => {
            debug_assert_eq!(o.len(), boxes.len(), "dfl_consensus_scores: owner length mismatch");
            o
        }
        None => {
            computed_owners = class_local_owners(kept, boxes, classes, &buckets, nms_iou);
            &computed_owners
        }
    };
    kept.iter()
        .enumerate()
        .map(|(kp, &k)| {
            if scores[k].is_nan() || scores[k] <= 0.0 {
                return scores[k];
            }
            let b = boxes[k];
            let ba = b.area();
            let s = scores[k] as f64;
            let kp_u32 = kp as u32;
            let mut mass = 0f64;
            let mut prec_sum = [0f64; 4];
            let mut prec_mass = [0f64; 4];
            let mut add_member = |j: usize, sj: f64| {
                mass += sj;
                if let Some(bp_fn) = bin_precision {
                    let bp = bp_fn(j);
                    for e in 0..4 {
                        let p = bp[e] as f64;
                        if p.is_finite() && p > 0.0 {
                            prec_sum[e] += sj * p;
                            prec_mass[e] += sj;
                        }
                    }
                }
            };
            if finite(&b) {
                for &j in buckets.of(classes[k]) {
                    let j = j as usize;
                    let c = &boxes[j];
                    let member =
                        j == k || (owners[j] == kp_u32 && finite(c) && overlap(&b, ba, c) >= ELC_IOU);
                    if member && scores[j] > 0.0 {
                        add_member(j, scores[j] as f64);
                    }
                }
            } else {
                add_member(k, s);
            }
            let sc = s.clamp(ELC_PROB_EPS, 1.0 - ELC_PROB_EPS);
            let mut z = (sc / (1.0 - sc)).ln();
            let ratio = (mass / s.max(1e-12)).clamp(1.0, ELC_CAP);
            z += ELC_ALPHA * (ratio / ELC_R_REF).ln();
            if bin_precision.is_some() && prec_mass.iter().all(|&m| m > 0.0) {
                let mut sum_std = 0f64;
                for e in 0..4 {
                    let mp = (prec_sum[e] / prec_mass[e]).max(1e-12);
                    sum_std += mp.sqrt().recip();
                }
                let sig_bin = (0.25 * sum_std).max(DFL_CONSENSUS_SIG_MIN);
                z -= DFL_CONSENSUS_BETA * (sig_bin / DFL_CONSENSUS_SIG_REF).ln();
            }
            branch_safe_sigmoid(z) as f32
        })
        .collect()
}

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
/// `precision` `None` meaning 1 (voting); the sums run in `f64` over the cluster in candidate order. A member side
/// whose weight is not `> 0` (a zero score, a NaN) or whose edge is not finite is skipped, and a side with no weight
/// keeps the kept box's edge.
pub fn fuse_boxes(
    kept: &[usize],
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    agnostic: bool,
    iou: f32,
    precision: Option<&dyn Fn(usize) -> [f32; 4]>,
) -> Vec<Xyxy> {
    let buckets = Buckets::new(boxes.len(), classes, agnostic);
    kept.iter()
        .map(|&k| {
            let b = boxes[k];
            let ba = b.area();
            let mut acc = [0f64; 4];
            let mut sum = [0f64; 4];
            for &j in buckets.of(classes[k]) {
                let j = j as usize;
                let c = &boxes[j];
                let o = overlap(&b, ba, c);
                if o.is_nan() || o < iou {
                    continue;
                }
                let s = scores[j] as f64;
                let p = precision.map_or([1.0; 4], |f| f(j));
                let x = c.to_array();
                for e in 0..4 {
                    let wt = s * p[e] as f64;
                    // A non-finite edge (a NaN box can join a cluster: its IoU comes out as 1) is left out.
                    if wt > 0.0 && x[e].is_finite() {
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
        // A NaN box overlaps everything (IoU 1 from f32 min / max) but must not poison the kept box.
        let boxes3 = [b(0.0, 0.0, 10.0, 10.0), b(f32::NAN, 0.0, 5.0, 10.0)];
        let got3 = fuse_boxes(&[0], &boxes3, &[0.9, 0.8], &[0, 0], false, 0.75, None)[0];
        assert_eq!(got3.x1, 0.0);
        assert!(got3.to_array().iter().all(|v| v.is_finite()), "{got3:?}");
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

    /// Hand-computed consensus scores: a lonely box (`M = s`, divided by `√10`), a cluster above the cap (the score
    /// stays, bit for bit), a cluster between (`M/s = 2.5`), a member just below the IoU, another class's box on top.
    #[test]
    fn consensus_scores_by_hand() {
        let boxes = [
            // Lonely: nothing else of class 0 near it; candidate 1 is the same box in class 1.
            b(0.0, 0.0, 10.0, 10.0),
            b(0.0, 0.0, 10.0, 10.0),
            // Between: kept 0.4 + a member at IoU 0.75 exactly (0.6); the third is at IoU 0.7, out.
            b(100.0, 0.0, 110.0, 10.0),
            b(100.0, 0.0, 110.0, 7.5),
            b(100.0, 0.0, 110.0, 7.0),
            // Above the cap: kept 0.1 with four members of 0.5 on the same box, M/s = 21.
            b(200.0, 0.0, 210.0, 10.0),
            b(200.0, 0.0, 210.0, 10.0),
            b(200.0, 0.0, 210.0, 10.0),
            b(200.0, 0.0, 210.0, 10.0),
            b(200.0, 0.0, 210.0, 10.0),
        ];
        let scores = [0.8f32, 0.9, 0.4, 0.6, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5];
        let classes = [0u32, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let got = consensus_scores(&[0, 2, 5], &boxes, &scores, &classes);
        assert_eq!(got[0], (0.8f32 as f64 / 10f64.sqrt()) as f32, "lonely: s / sqrt(10)");
        let mass = 0.4f32 as f64 + 0.6f32 as f64;
        let want = 0.4f32 as f64 * (mass / 0.4f32 as f64).sqrt() / 10f64.sqrt();
        assert_eq!(got[1], want as f32, "between: M/s = 2.5");
        assert!((got[1] - 0.2).abs() < 1e-6, "0.4 * sqrt(2.5 / 10) = 0.2, got {}", got[1]);
        assert_eq!(got[2].to_bits(), 0.1f32.to_bits(), "at the cap the score stays");
    }

    /// A kept box is in its own cluster even when its IoU with itself is not defined (zero area); a member with a
    /// NaN box or a score that is not positive adds nothing; a kept score that is not positive stays. A NaN box and
    /// a valid one reach IoU 1 through f32 min / max, and neither lends the other its score, in either direction.
    #[test]
    fn consensus_edge_cases() {
        let flat = [b(5.0, 5.0, 5.0, 9.0)];
        let got = consensus_scores(&[0], &flat, &[0.5], &[0])[0];
        assert_eq!(got, (0.5f64 / 10f64.sqrt()) as f32);
        let boxes = [b(0.0, 0.0, 10.0, 10.0), b(f32::NAN, 0.0, 5.0, 10.0), b(0.0, 0.0, 10.0, 10.0)];
        assert_eq!(overlap(&boxes[1], boxes[1].area(), &boxes[0]), 1.0, "the pair the guard is for");
        assert_eq!(overlap(&boxes[0], boxes[0].area(), &boxes[1]), 1.0, "the pair the guard is for");
        let got = consensus_scores(&[0], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0])[0];
        assert_eq!(got, (0.5f64 / 10f64.sqrt()) as f32, "the NaN box and the zero score add nothing");
        let got = consensus_scores(&[1], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0])[0];
        assert_eq!(got, (0.9f32 as f64 / 10f64.sqrt()) as f32, "a kept NaN box has no members but itself");
        assert_eq!(consensus_scores(&[2], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0]), [0.0]);
    }

    /// `fuse_boxes` reads a class only for a kept candidate under `agnostic`: a shorter `classes` still works.
    #[test]
    fn agnostic_fusion_reads_only_the_kept_class() {
        let boxes = [b(0.0, 0.0, 10.0, 10.0), b(1.0, 0.0, 10.0, 10.0)];
        let got = fuse_boxes(&[0], &boxes, &[0.5, 0.5], &[0], true, 0.75, None)[0];
        assert_eq!(got.x1, 0.5, "both boxes vote");
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
