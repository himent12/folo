//! Non-maximum suppression.
//!
//! - [`nms_sorted`]: greedy single-set NMS over boxes already sorted by descending score (the kernel).
//! - [`batched_nms`]: class-aware NMS over arbitrary boxes, scores and classes. Boxes of different classes never
//!   suppress each other (unless `agnostic`).
//! - [`non_max_suppression`]: the full detector post-process from decoded head outputs (boxes `[A]`, class scores
//!   `[A, C]`), mirroring Ultralytics' `non_max_suppression` (confidence filter, optional multi-label, class filter,
//!   candidate cap, batched NMS, `max_det`).
//! - [`nms_detections`]: the same filters applied to an existing list of [`Detection`]s.
//! - [`topk_detections`] / [`topk_candidates`]: the NMS-free selection of the one-to-one head (docs/MODEL.md
//!   §6.6), and [`postprocess`], which picks it or [`non_max_suppression`] by [`HeadMode`].
//!
//! **Semantics** (identical to torchvision / Ultralytics): candidates are those with `score > conf` (strict); they
//! are visited in descending score order (ties: lower input index first); a box is suppressed when its IoU with an
//! already kept box of the same class is `> iou` (strict). Output is sorted by descending score.
//!
//! **Fast path.** The candidates are sorted once; per-class buckets are then filled by a stable counting sort, so
//! each bucket is already in score order and no per-class sort is needed. Buckets run in parallel (rayon) when there
//! is enough work. Inside a bucket the boxes are laid out structure-of-arrays and every kept box suppresses all later
//! boxes in one branch-free pass the compiler vectorises (split across threads for very large buckets); a bucket stops
//! as soon as it has `max_det` survivors, since nothing after that can reach the output. With `max_det = 300` the
//! cost is therefore `O(n log n + max_det * n)` instead of the `O(n²)` worst case of textbook greedy NMS.

use crate::boxes::Xyxy;
use crate::fusion::{
    BoxFusion, FUSION_IOU, ScoreRule, consensus_scores, dfl_consensus_scores, dfl_repulse_scores, elc_scores,
    fuse_boxes,
};
use crate::model::HeadMode;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod topk;
pub use topk::{topk_candidates, topk_detections};

/// Pre-NMS extraction threshold for single-label calibrated scoring (`Elc` / `DflConsensus`) when `conf` is higher:
/// nearby supporting anchors down to `0.10` join the cluster before final scores are filtered by `score > conf`.
pub const CALIBRATED_PRE_CONF: f32 = 0.10;

/// Remaining-box count above which one suppression pass is split across threads.
const PAR_SUPPRESS: usize = 1 << 14;
/// Total candidate count above which class buckets are processed in parallel.
const PAR_BUCKETS: usize = 2048;
/// Anchor-row count above which candidate extraction runs in parallel.
const PAR_ROWS: usize = 4096;

/// One detection: a box, its confidence and its class index.
///
/// Serialises flat as `{ "x1", "y1", "x2", "y2", "score", "class" }` (the `ScoredBox` shape of `docs/API.md` minus
/// `name`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// The box, `xyxy`.
    #[serde(flatten)]
    pub bbox: Xyxy,
    /// Confidence in `[0, 1]`.
    pub score: f32,
    /// Class index.
    pub class: u32,
}

/// Post-processing settings. Defaults are Ultralytics' predict defaults; [`NmsConfig::val`] gives its validation
/// settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NmsConfig {
    /// Keep candidates with `score > conf`.
    pub conf: f32,
    /// Suppress a box when its IoU with a kept box is `> iou`.
    pub iou: f32,
    /// Maximum detections returned per image.
    pub max_det: usize,
    /// Only the `max_candidates` highest-scoring candidates enter NMS (Ultralytics' `max_nms`).
    pub max_candidates: usize,
    /// Keep only these classes (applied after the per-anchor best class is chosen, as Ultralytics does).
    pub classes: Option<Vec<u32>>,
    /// Class-agnostic NMS: boxes of different classes suppress each other.
    pub agnostic: bool,
    /// One candidate per (anchor, class) pair above `conf` instead of one per anchor (its best class).
    pub multi_label: bool,
    /// Refine each kept box from its NMS cluster ([`crate::fusion`], docs/MODEL.md §9.4). Off by default. Not used
    /// by the NMS-free `o2o` selection, which has no clusters.
    pub fusion: BoxFusion,
    /// The cluster of a kept box for [`NmsConfig::fusion`]: the candidates of its class (any class with
    /// `agnostic`) whose IoU with it is `>=` this.
    pub fusion_iou: f32,
    /// How the kept boxes are scored ([`ScoreRule`], docs/MODEL.md §9.4). Off by default: the consensus score
    /// changes the score scale. Not used by the NMS-free `o2o` selection, which has no clusters.
    pub score: ScoreRule,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
            max_candidates: 30_000,
            classes: None,
            agnostic: false,
            multi_label: false,
            fusion: BoxFusion::None,
            fusion_iou: FUSION_IOU,
            score: ScoreRule::None,
        }
    }
}

impl NmsConfig {
    /// Validation settings: `conf = 0.001`, `iou = 0.7`, `max_det = 300`, multi-label.
    pub fn val() -> Self {
        Self { conf: 0.001, multi_label: true, ..Self::default() }
    }

    /// Pre-NMS candidate threshold: `0.10` for single-label calibrated scoring (`Elc` / `DflConsensus`) when
    /// `conf > 0.10`, else `self.conf` unchanged.
    #[inline]
    pub fn candidate_conf(&self) -> f32 {
        if self.post_filter_calibrated() { CALIBRATED_PRE_CONF } else { self.conf }
    }

    /// Like [`NmsConfig::candidate_conf`], but returns `self.conf` unchanged for [`HeadMode::O2o`]
    /// (which skips NMS and score rules).
    #[inline]
    pub fn candidate_conf_for(&self, mode: HeadMode) -> f32 {
        if mode == HeadMode::O2m { self.candidate_conf() } else { self.conf }
    }

    /// Whether final detections are filtered by `score > self.conf` after calibrated re-scoring.
    #[inline]
    pub(crate) fn post_filter_calibrated(&self) -> bool {
        !self.multi_label && self.score.is_calibrated() && self.conf > CALIBRATED_PRE_CONF
    }
}

/// Greedy NMS over `boxes` sorted by descending score. Returns the kept positions in order, stopping after `limit`
/// survivors. A box is suppressed when its IoU with a kept box is `> iou_thr`.
pub fn nms_sorted(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> Vec<usize> {
    nms_sorted_inner(boxes, iou_thr, limit, false).0
}

/// [`nms_sorted`] also returning, for every input position `i`, the position in `keep` of the kept box that first
/// suppressed `i` (or `i`'s own position in `keep` when kept; `u32::MAX` if unsuppressed past `limit`).
pub fn nms_sorted_with_owner(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> (Vec<usize>, Vec<u32>) {
    nms_sorted_inner(boxes, iou_thr, limit, true)
}

fn nms_sorted_inner(boxes: &[Xyxy], iou_thr: f32, limit: usize, track_owner: bool) -> (Vec<usize>, Vec<u32>) {
    let n = boxes.len();
    let mut keep = Vec::new();
    let mut owner = if track_owner { vec![u32::MAX; n] } else { Vec::new() };
    if n == 0 || limit == 0 {
        return (keep, owner);
    }
    // Structure-of-arrays so the suppression pass vectorises.
    let x1: Vec<f32> = boxes.iter().map(|b| b.x1).collect();
    let y1: Vec<f32> = boxes.iter().map(|b| b.y1).collect();
    let x2: Vec<f32> = boxes.iter().map(|b| b.x2).collect();
    let y2: Vec<f32> = boxes.iter().map(|b| b.y2).collect();
    let area: Vec<f32> = boxes.iter().map(Xyxy::area).collect();
    let mut suppressed = vec![false; n];

    for i in 0..n {
        if suppressed[i] {
            continue;
        }
        let kp = keep.len() as u32;
        keep.push(i);
        if track_owner {
            owner[i] = kp;
        } else if keep.len() == limit {
            break;
        }
        let (bx1, by1, bx2, by2, ba) = (x1[i], y1[i], x2[i], y2[i], area[i]);
        let start = i + 1;
        let pass = |off: usize, sup: &mut [bool], own: Option<&mut [u32]>| {
            let lo = start + off;
            let hi = lo + sup.len();
            let (cx1, cy1, cx2, cy2, ca) =
                (&x1[lo..hi], &y1[lo..hi], &x2[lo..hi], &y2[lo..hi], &area[lo..hi]);
            if let Some(own) = own {
                for k in 0..sup.len() {
                    if sup[k] {
                        continue;
                    }
                    let w = (bx2.min(cx2[k]) - bx1.max(cx1[k])).max(0.0);
                    let h = (by2.min(cy2[k]) - by1.max(cy1[k])).max(0.0);
                    let inter = w * h;
                    let union = ba + ca[k] - inter;
                    let iou = if union > 0.0 { inter / union } else { 0.0 };
                    if iou > iou_thr {
                        sup[k] = true;
                        own[k] = kp;
                    }
                }
            } else {
                for k in 0..sup.len() {
                    // Same arithmetic as `box_iou`, so results match it bit for bit.
                    let w = (bx2.min(cx2[k]) - bx1.max(cx1[k])).max(0.0);
                    let h = (by2.min(cy2[k]) - by1.max(cy1[k])).max(0.0);
                    let inter = w * h;
                    let union = ba + ca[k] - inter;
                    let iou = if union > 0.0 { inter / union } else { 0.0 };
                    sup[k] |= iou > iou_thr;
                }
            }
        };
        let rest_sup = &mut suppressed[start..];
        if track_owner {
            let rest_own = &mut owner[start..];
            if rest_sup.len() >= PAR_SUPPRESS {
                const CHUNK: usize = 4096;
                rest_sup
                    .par_chunks_mut(CHUNK)
                    .zip(rest_own.par_chunks_mut(CHUNK))
                    .enumerate()
                    .for_each(|(c, (sup, own))| pass(c * CHUNK, sup, Some(own)));
            } else {
                pass(0, rest_sup, Some(rest_own));
            }
            if keep.len() == limit {
                break;
            }
        } else if rest_sup.len() >= PAR_SUPPRESS {
            const CHUNK: usize = 4096;
            rest_sup.par_chunks_mut(CHUNK).enumerate().for_each(|(c, sup)| pass(c * CHUNK, sup, None));
        } else {
            pass(0, rest_sup, None);
        }
    }
    (keep, owner)
}

/// Order of indices by descending score, ties by ascending index. NaN scores sort last.
fn score_order(scores: &[f32], idx: &mut [u32]) {
    let key = |i: &u32| {
        let s = scores[*i as usize];
        if s.is_nan() { f32::NEG_INFINITY } else { s }
    };
    idx.sort_unstable_by(|a, b| key(b).total_cmp(&key(a)).then(a.cmp(b)));
}

/// Class-aware NMS. Returns indices into the inputs of the kept boxes, sorted by descending score (ties: lower
/// index first), at most `max_det` of them. No confidence filter is applied here.
///
/// # Panics
/// If the three slices differ in length.
pub fn batched_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    max_det: usize,
    agnostic: bool,
) -> Vec<usize> {
    batched_nms_inner(boxes, scores, classes, iou_thr, max_det, agnostic, false).0
}

fn batched_nms_inner(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    max_det: usize,
    agnostic: bool,
    track_owner: bool,
) -> (Vec<usize>, Option<Vec<u32>>) {
    assert_eq!(boxes.len(), scores.len(), "batched_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_nms: boxes/classes length mismatch");
    let n = boxes.len();
    if n == 0 || max_det == 0 {
        return (Vec::new(), None);
    }
    assert!(n <= u32::MAX as usize, "batched_nms: too many boxes");
    let mut order: Vec<u32> = (0..n as u32).collect();
    score_order(scores, &mut order);

    if agnostic {
        let sorted: Vec<Xyxy> = order.iter().map(|&i| boxes[i as usize]).collect();
        let kept = nms_sorted(&sorted, iou_thr, max_det).into_iter().map(|k| order[k] as usize).collect();
        // Agnostic NMS mixes classes; calibrated score rules derive class-local ownership when `None`.
        return (kept, None);
    }

    // Group the score-ordered indices into per-class buckets, keeping score order inside each bucket: a stable
    // counting sort for dense class ids, a stable comparison sort when the ids are sparse (huge max id).
    let max_class = classes.iter().copied().max().unwrap_or(0) as usize;
    let (bucketed, ranges): (Vec<u32>, Vec<(usize, usize)>) = if max_class <= n.max(1024) {
        let mut starts = vec![0usize; max_class + 2];
        for &c in classes {
            starts[c as usize + 1] += 1;
        }
        for c in 0..=max_class {
            starts[c + 1] += starts[c];
        }
        let mut fill = starts.clone();
        let mut bucketed = vec![0u32; n];
        for &i in &order {
            let c = classes[i as usize] as usize;
            bucketed[fill[c]] = i;
            fill[c] += 1;
        }
        let ranges = starts.windows(2).map(|w| (w[0], w[1])).filter(|r| r.0 < r.1).collect();
        (bucketed, ranges)
    } else {
        let mut bucketed = order.clone();
        bucketed.sort_by_key(|&i| classes[i as usize]);
        let mut ranges = Vec::new();
        let mut lo = 0;
        for hi in 1..=n {
            if hi == n || classes[bucketed[hi] as usize] != classes[bucketed[lo] as usize] {
                ranges.push((lo, hi));
                lo = hi;
            }
        }
        (bucketed, ranges)
    };
    if !track_owner {
        let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<u32> {
            let ids = &bucketed[lo..hi];
            let sorted: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
            nms_sorted(&sorted, iou_thr, max_det).into_iter().map(|k| ids[k]).collect()
        };
        let mut kept: Vec<u32> = if n >= PAR_BUCKETS {
            ranges.par_iter().flat_map_iter(run_bucket).collect()
        } else {
            ranges.iter().flat_map(run_bucket).collect()
        };
        score_order(scores, &mut kept);
        kept.truncate(max_det);
        return (kept.into_iter().map(|i| i as usize).collect(), None);
    }
    let run_bucket = |&(lo, hi): &(usize, usize)| -> (Vec<u32>, Vec<u32>) {
        let ids = &bucketed[lo..hi];
        let sorted: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let (k, own) = nms_sorted_with_owner(&sorted, iou_thr, max_det);
        (k.into_iter().map(|m| ids[m]).collect(), own)
    };
    let parts: Vec<(Vec<u32>, Vec<u32>)> = if n >= PAR_BUCKETS {
        ranges.par_iter().map(run_bucket).collect()
    } else {
        ranges.iter().map(run_bucket).collect()
    };
    let mut kept: Vec<u32> = parts.iter().flat_map(|(k, _)| k.iter().copied()).collect();
    score_order(scores, &mut kept);
    kept.truncate(max_det);
    let mut global_pos = vec![u32::MAX; n];
    for (kp, &cand_id) in kept.iter().enumerate() {
        global_pos[cand_id as usize] = kp as u32;
    }
    let mut owner = vec![u32::MAX; n];
    for (&(lo, hi), (bk_kept, bk_own)) in ranges.iter().zip(&parts) {
        let ids = &bucketed[lo..hi];
        for (&cand_id, &loc_kp) in ids.iter().zip(bk_own) {
            if loc_kp != u32::MAX {
                let win = bk_kept[loc_kp as usize] as usize;
                owner[cand_id as usize] = global_pos[win];
            }
        }
    }
    (kept.into_iter().map(|i| i as usize).collect(), Some(owner))
}

/// A compact NMS candidate: score, source index (anchor or input detection) and class.
#[derive(Clone, Copy, Debug)]
struct Cand {
    score: f32,
    idx: u32,
    class: u32,
}

/// Per-side precisions ([`crate::decode::side_precision`]) of the candidates' sources, by source index (anchor
/// or input position): what [`BoxFusion::Dfl`] weighs a cluster member's sides by. `None`: equal precisions.
pub type Precisions<'a> = Option<&'a [[f32; 4]]>;

/// Optional per-source DFL and anchor metadata passed to post-processing ([`BoxFusion::Dfl`],
/// [`ScoreRule::DflConsensus`], [`ScoreRule::DflRepulse`]), indexed by source position (anchor or candidate).
#[derive(Clone, Copy, Debug, Default)]
pub struct DflInputs<'a> {
    /// Per-side DFL precisions ([`crate::decode::side_precision`]) for [`BoxFusion::Dfl`].
    pub precision: Precisions<'a>,
    /// Scale-free per-side DFL bin precisions ([`crate::decode::bin_precision`]) for [`ScoreRule::DflConsensus`].
    pub bin_precision: Precisions<'a>,
    /// Mean 4-side DFL Shannon entropy in nats ([`crate::decode::dfl_entropy`]) for [`ScoreRule::DflRepulse`].
    pub entropy: Option<&'a [f32]>,
    /// Anchor stride in input pixels for [`ScoreRule::DflRepulse`]'s cross-scale consensus term.
    pub strides: Option<&'a [f32]>,
}

/// Cap the candidates (given in `(idx, class)` order) at the `cap` best by score, ties to the earlier candidate,
/// then run batched NMS, fuse the kept boxes when [`NmsConfig::fusion`] asks for it (with `dfl.precision`, indexed
/// by source index, for [`BoxFusion::Dfl`]), re-score them when [`NmsConfig::score`] does (sorted by the new score,
/// ties in NMS's order), and build the detections. Every NMS path of every device ends here.
fn finish(
    mut cands: Vec<Cand>,
    cfg: &NmsConfig,
    bbox_of: impl Fn(u32) -> Xyxy,
    dfl: DflInputs<'_>,
) -> Vec<Detection> {
    let (cands, boxes): (Vec<Cand>, Vec<Xyxy>) = if cfg.score.is_calibrated() {
        // Filter non-finite scores (+Inf / NaN) and non-finite boxes before `max_candidates` truncation so
        // malformed high-scoring entries cannot evict valid candidates or survive calibrated NMS.
        let mut valid: Vec<(Cand, Xyxy)> = cands
            .into_iter()
            .filter_map(|c| {
                if !c.score.is_finite() {
                    return None;
                }
                let bx = bbox_of(c.idx);
                bx.to_array().iter().all(|v| v.is_finite()).then_some((c, bx))
            })
            .collect();
        if valid.len() > cfg.max_candidates {
            if cfg.max_candidates == 0 {
                return Vec::new();
            }
            let rank = |a: &(Cand, Xyxy), b: &(Cand, Xyxy)| {
                b.0.score.total_cmp(&a.0.score).then(a.0.idx.cmp(&b.0.idx)).then(a.0.class.cmp(&b.0.class))
            };
            valid.select_nth_unstable_by(cfg.max_candidates - 1, rank);
            valid.truncate(cfg.max_candidates);
            valid.sort_unstable_by_key(|(c, _)| (c.idx, c.class));
        }
        valid.into_iter().unzip()
    } else {
        if cands.len() > cfg.max_candidates {
            if cfg.max_candidates == 0 {
                return Vec::new();
            }
            let key = |c: &Cand| if c.score.is_nan() { f32::NEG_INFINITY } else { c.score };
            let rank = |a: &Cand, b: &Cand| {
                key(b).total_cmp(&key(a)).then(a.idx.cmp(&b.idx)).then(a.class.cmp(&b.class))
            };
            cands.select_nth_unstable_by(cfg.max_candidates - 1, rank);
            cands.truncate(cfg.max_candidates);
            // Back to candidate order so NMS tie-breaks by position.
            cands.sort_unstable_by_key(|c| (c.idx, c.class));
        }
        let boxes = cands.iter().map(|c| bbox_of(c.idx)).collect();
        (cands, boxes)
    };
    let scores: Vec<f32> = cands.iter().map(|c| c.score).collect();
    let classes: Vec<u32> = cands.iter().map(|c| c.class).collect();
    let (kept, owner) = batched_nms_inner(
        &boxes,
        &scores,
        &classes,
        cfg.iou,
        cfg.max_det,
        cfg.agnostic,
        cfg.score.is_calibrated(),
    );
    let cands_slice = cands.as_slice();
    let fused = match cfg.fusion {
        BoxFusion::None => None,
        BoxFusion::Vote => {
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, None))
        }
        BoxFusion::Dfl => {
            let of = dfl.precision.map(|p| move |k: usize| p[cands_slice[k].idx as usize]);
            let of = of.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, of))
        }
    };
    let mut dets: Vec<Detection> = kept
        .iter()
        .enumerate()
        .map(|(i, &k)| Detection {
            bbox: fused.as_ref().map_or(boxes[k], |f| f[i]),
            score: scores[k],
            class: classes[k],
        })
        .collect();
    let new_scores = match cfg.score {
        ScoreRule::None => None,
        ScoreRule::Consensus => Some(consensus_scores(&kept, &boxes, &scores, &classes)),
        ScoreRule::Elc => Some(elc_scores(&kept, &boxes, &scores, &classes, owner.as_deref(), cfg.iou)),
        ScoreRule::DflConsensus => {
            let bp = dfl.bin_precision.map(|p| move |k: usize| p[cands_slice[k].idx as usize]);
            let bp = bp.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
            Some(dfl_consensus_scores(&kept, &boxes, &scores, &classes, owner.as_deref(), cfg.iou, bp))
        }
        ScoreRule::DflRepulse => {
            let repulse_boxes: Vec<Xyxy> = dets.iter().map(|d| d.bbox).collect();
            let ent = dfl.entropy.map(|e| move |k: usize| e[cands_slice[k].idx as usize]);
            let ent = ent.as_ref().map(|f| f as &dyn Fn(usize) -> f32);
            let st = dfl.strides.map(|s| move |k: usize| s[cands_slice[k].idx as usize]);
            let st = st.as_ref().map(|f| f as &dyn Fn(usize) -> f32);
            Some(dfl_repulse_scores(
                &kept,
                &boxes,
                &repulse_boxes,
                &scores,
                &classes,
                owner.as_deref(),
                cfg.iou,
                ent,
                st,
            ))
        }
    };
    if let Some(new_scores) = new_scores {
        for (d, s) in dets.iter_mut().zip(new_scores) {
            d.score = s;
        }
        // Stable: equal new scores stay in NMS's order.
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        if cfg.post_filter_calibrated() {
            dets.retain(|d| d.score > cfg.conf);
        }
    }
    dets
}

pub(crate) fn class_allowed(cfg: &NmsConfig, c: u32) -> bool {
    cfg.classes.as_ref().is_none_or(|cs| cs.contains(&c))
}

/// Apply the confidence filter, class filter, candidate cap and batched NMS to a list of detections.
///
/// # Panics
/// If there are more than `u32::MAX` detections.
pub fn nms_detections(dets: &[Detection], cfg: &NmsConfig) -> Vec<Detection> {
    assert!(dets.len() <= u32::MAX as usize, "nms_detections: too many detections");
    let conf = cfg.candidate_conf();
    let cands: Vec<Cand> = dets
        .iter()
        .enumerate()
        .filter(|(_, d)| d.score > conf && class_allowed(cfg, d.class))
        .map(|(i, d)| Cand { score: d.score, idx: i as u32, class: d.class })
        .collect();
    finish(cands, cfg, |i| dets[i as usize].bbox, DflInputs::default())
}

/// The detector post-process for one image.
///
/// - `boxes`: `[A]` decoded boxes, `xyxy` (see [`crate::decode`]).
/// - `scores`: `[A, num_classes]` row-major class probabilities (post-sigmoid).
///
/// Candidates are one per anchor (its best class; first index on ties) or, with `multi_label`, one per
/// (anchor, class) pair, filtered by `score > conf` and the class filter, capped at `max_candidates`, then batched
/// NMS. Returns at most `max_det` detections sorted by descending score.
///
/// # Panics
/// If `scores.len() != boxes.len() * num_classes`, or there are more than `u32::MAX` anchors.
pub fn non_max_suppression(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_with_inputs(boxes, scores, DflInputs::default(), num_classes, cfg)
}

/// [`non_max_suppression`] with the anchors' side precisions (`[A]`, [`crate::decode::side_precisions`]) for
/// [`BoxFusion::Dfl`]; `None` weighs every side alike.
///
/// # Panics
/// As [`non_max_suppression`], or if `precision` is not `[A]`.
pub fn non_max_suppression_with(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_with_inputs(
        boxes,
        scores,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
    )
}

/// [`non_max_suppression_with`] also accepting scale-free per-anchor DFL bin precisions (`[A]`,
/// [`crate::decode::bin_precisions`]) for [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_with_dfl(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_with_inputs(boxes, scores, dfl, num_classes, cfg)
}

/// [`non_max_suppression`] accepting all optional per-anchor [`DflInputs`] (`[A]`).
pub fn non_max_suppression_with_inputs(
    boxes: &[Xyxy],
    scores: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let a = boxes.len();
    assert!(dfl.precision.is_none_or(|p| p.len() == a), "non_max_suppression: precision must be [A]");
    assert!(dfl.bin_precision.is_none_or(|p| p.len() == a), "non_max_suppression: bin_precision must be [A]");
    assert!(dfl.entropy.is_none_or(|e| e.len() == a), "non_max_suppression: entropy must be [A]");
    assert!(dfl.strides.is_none_or(|s| s.len() == a), "non_max_suppression: strides must be [A]");
    assert_eq!(scores.len(), a * num_classes, "non_max_suppression: scores must be [A, num_classes]");
    assert!(a <= u32::MAX as usize, "non_max_suppression: too many anchors");
    if num_classes == 0 || a == 0 {
        return Vec::new();
    }
    let cands = extract_candidates(scores, a, num_classes, num_classes, 0, cfg);
    finish(cands, cfg, |i| boxes[i as usize], dfl)
}

/// [`non_max_suppression`] of one image's decoded outputs as rows `[A, 4 + num_classes]` (`x1, y1, x2, y2`, then
/// the class scores: [`crate::model::HeadOutput::decoded_on_device`]'s layout), read in place. Same result.
///
/// # Panics
/// If `rows.len()` is not a multiple of `4 + num_classes`, or there are more than `u32::MAX` anchors.
pub fn non_max_suppression_rows(rows: &[f32], num_classes: usize, cfg: &NmsConfig) -> Vec<Detection> {
    non_max_suppression_rows_with_inputs(rows, DflInputs::default(), num_classes, cfg)
}

/// [`non_max_suppression_rows`] with the anchors' side precisions (`[A]`) for [`BoxFusion::Dfl`].
///
/// # Panics
/// As [`non_max_suppression_rows`], or if `precision` is not `[A]`.
pub fn non_max_suppression_rows_with(
    rows: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_rows_with_inputs(
        rows,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
    )
}

/// [`non_max_suppression_rows_with`] also accepting scale-free DFL bin precisions (`[A]`) for
/// [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_rows_with_dfl(
    rows: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_rows_with_inputs(rows, dfl, num_classes, cfg)
}

/// [`non_max_suppression_rows`] accepting all optional per-anchor [`DflInputs`] (`[A]`).
pub fn non_max_suppression_rows_with_inputs(
    rows: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let w = 4 + num_classes;
    assert!(rows.len().is_multiple_of(w), "non_max_suppression_rows: rows must be [A, 4 + num_classes]");
    let a = rows.len() / w;
    assert!(a <= u32::MAX as usize, "non_max_suppression_rows: too many anchors");
    if num_classes == 0 || a == 0 {
        return Vec::new();
    }
    assert!(dfl.precision.is_none_or(|p| p.len() == a), "non_max_suppression_rows: precision must be [A]");
    assert!(
        dfl.bin_precision.is_none_or(|p| p.len() == a),
        "non_max_suppression_rows: bin_precision must be [A]"
    );
    assert!(dfl.entropy.is_none_or(|e| e.len() == a), "non_max_suppression_rows: entropy must be [A]");
    assert!(dfl.strides.is_none_or(|s| s.len() == a), "non_max_suppression_rows: strides must be [A]");
    let cands = extract_candidates(rows, a, num_classes, w, 4, cfg);
    let bbox_of = |i: u32| {
        let r = &rows[i as usize * w..];
        Xyxy::new(r[0], r[1], r[2], r[3])
    };
    finish(cands, cfg, bbox_of, dfl)
}

/// Multi-label [`non_max_suppression_rows`] from the `(anchor, class, score)` pairs whose score is `> conf`, in any
/// order (what folo-gpu's `detect::multi_label_candidates` downloads instead of every anchor's scores), with the
/// box of anchor `i` from `bbox_of(i)`. Given exactly the pairs above `conf` of the same scores, the result is that
/// of [`non_max_suppression_rows`] (the pairs are put back in its `(anchor, class)` order). Needs `multi_label`.
pub fn non_max_suppression_pairs(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, DflInputs::default(), cfg)
}

/// [`non_max_suppression_pairs`] with the anchors' side precisions (indexed by anchor) for [`BoxFusion::Dfl`].
pub fn non_max_suppression_pairs_with(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, DflInputs { precision, ..Default::default() }, cfg)
}

/// [`non_max_suppression_pairs_with`] also accepting scale-free DFL bin precisions (indexed by anchor) for
/// [`ScoreRule::DflConsensus`].
pub fn non_max_suppression_pairs_with_dfl(
    pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
    bin_precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    non_max_suppression_pairs_with_inputs(pairs, bbox_of, dfl, cfg)
}

/// [`non_max_suppression_pairs`] accepting all optional per-anchor [`DflInputs`] (indexed by anchor).
pub fn non_max_suppression_pairs_with_inputs(
    mut pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    dfl: DflInputs<'_>,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    debug_assert!(cfg.multi_label, "non_max_suppression_pairs is the multi-label path");
    let conf = cfg.candidate_conf();
    pairs.sort_unstable_by_key(|&(a, c, _)| (a, c));
    let cands = pairs
        .into_iter()
        .filter(|&(_, c, s)| s > conf && class_allowed(cfg, c))
        .map(|(idx, class, score)| Cand { score, idx, class })
        .collect();
    finish(cands, cfg, bbox_of, dfl)
}

/// The candidates of `anchors` score rows (row `a` is `data[a * stride + offset..][..num_classes]`), in anchor
/// order: every class above `conf` with `multi_label`, else the anchor's best class (first on ties) if above.
fn extract_candidates(
    data: &[f32],
    anchors: usize,
    num_classes: usize,
    stride: usize,
    offset: usize,
    cfg: &NmsConfig,
) -> Vec<Cand> {
    let conf = cfg.candidate_conf();
    let extract = |a: usize, row: &[f32], out: &mut Vec<Cand>| {
        if cfg.multi_label {
            for (c, &s) in row.iter().enumerate() {
                if s > conf && class_allowed(cfg, c as u32) {
                    out.push(Cand { score: s, idx: a as u32, class: c as u32 });
                }
            }
        } else {
            let (mut best, mut bs) = (0usize, row[0]);
            for (c, &s) in row.iter().enumerate().skip(1) {
                if s > bs {
                    best = c;
                    bs = s;
                }
            }
            if bs > conf && class_allowed(cfg, best as u32) {
                out.push(Cand { score: bs, idx: a as u32, class: best as u32 });
            }
        }
    };
    let rows = |lo: usize, hi: usize, out: &mut Vec<Cand>| {
        for a in lo..hi {
            extract(a, &data[a * stride + offset..][..num_classes], out);
        }
    };
    if anchors >= PAR_ROWS {
        const ROWS: usize = 1024;
        let parts: Vec<Vec<Cand>> = (0..anchors.div_ceil(ROWS))
            .into_par_iter()
            .map(|ci| {
                let mut out = Vec::new();
                rows(ci * ROWS, ((ci + 1) * ROWS).min(anchors), &mut out);
                out
            })
            .collect();
        parts.concat()
    } else {
        let mut out = Vec::new();
        rows(0, anchors, &mut out);
        out
    }
}

/// An anchor's best class above the confidence threshold, with its decoded box: what a device decode
/// downloads instead of every anchor's scores (folo-core's `Tensor::detect_candidates`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Anchor index.
    pub anchor: u32,
    /// Its best class (first maximum).
    pub class: u32,
    /// That class's score.
    pub score: f32,
    /// The decoded box, `xyxy`.
    pub bbox: Xyxy,
}

/// [`non_max_suppression`] (single-label) from the candidates already extracted: one per anchor whose best
/// class scores `> conf`, **sorted by anchor**. Applies the class filter, the candidate cap and batched NMS;
/// for the candidates of the same decoded outputs the result is identical to [`non_max_suppression`] with
/// `multi_label` off.
///
/// # Panics
/// If there are more than `u32::MAX` candidates.
pub fn nms_candidates(cands: &[Candidate], cfg: &NmsConfig) -> Vec<Detection> {
    nms_candidates_with(cands, None, cfg)
}

/// [`nms_candidates`] with each candidate's side precisions (parallel to `cands`) for [`BoxFusion::Dfl`].
///
/// # Panics
/// As [`nms_candidates`], or if `precision` is not as long as `cands`.
pub fn nms_candidates_with(cands: &[Candidate], precision: Precisions, cfg: &NmsConfig) -> Vec<Detection> {
    nms_candidates_with_dfl(cands, precision, None, cfg)
}

/// [`nms_candidates_with`] also accepting scale-free DFL bin precisions (parallel to `cands`) for
/// [`ScoreRule::DflConsensus`].
pub fn nms_candidates_with_dfl(
    cands: &[Candidate],
    precision: Precisions,
    bin_precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    nms_candidates_with_inputs(cands, dfl, cfg)
}

/// [`nms_candidates`] accepting all optional per-candidate [`DflInputs`] (parallel to `cands`).
pub fn nms_candidates_with_inputs(
    cands: &[Candidate],
    dfl: DflInputs<'_>,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    let n = cands.len();
    assert!(n <= u32::MAX as usize, "nms_candidates: too many candidates");
    assert!(dfl.precision.is_none_or(|p| p.len() == n), "nms_candidates: precision must match cands");
    assert!(dfl.bin_precision.is_none_or(|p| p.len() == n), "nms_candidates: bin_precision must match cands");
    assert!(dfl.entropy.is_none_or(|e| e.len() == n), "nms_candidates: entropy must match cands");
    assert!(dfl.strides.is_none_or(|s| s.len() == n), "nms_candidates: strides must match cands");
    debug_assert!(cands.windows(2).all(|w| w[0].anchor < w[1].anchor), "nms_candidates: unsorted");
    let conf = cfg.candidate_conf();
    // Positions in anchor order tie-break exactly as anchor indices do.
    let list: Vec<Cand> = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score > conf && class_allowed(cfg, c.class))
        .map(|(i, c)| Cand { score: c.score, idx: i as u32, class: c.class })
        .collect();
    finish(list, cfg, |i| cands[i as usize].bbox, dfl)
}

/// The post-process of one image's decoded outputs for the branch that produced them: [`non_max_suppression`]
/// (`o2m`) or [`topk_detections`] (`o2o`, no NMS).
pub fn postprocess(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    postprocess_with_inputs(boxes, scores, DflInputs::default(), num_classes, cfg, mode)
}

/// [`postprocess`] with the anchors' side precisions (`[A]`) for [`BoxFusion::Dfl`] (unused by `o2o`).
pub fn postprocess_with(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    postprocess_with_inputs(
        boxes,
        scores,
        DflInputs { precision, ..Default::default() },
        num_classes,
        cfg,
        mode,
    )
}

/// [`postprocess_with`] also accepting scale-free DFL bin precisions (`[A]`) for [`ScoreRule::DflConsensus`].
pub fn postprocess_with_dfl(
    boxes: &[Xyxy],
    scores: &[f32],
    precision: Precisions,
    bin_precision: Precisions,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    let dfl = DflInputs { precision, bin_precision, ..Default::default() };
    postprocess_with_inputs(boxes, scores, dfl, num_classes, cfg, mode)
}

/// [`postprocess`] accepting all optional per-anchor [`DflInputs`] (`[A]`, unused by `o2o`).
pub fn postprocess_with_inputs(
    boxes: &[Xyxy],
    scores: &[f32],
    dfl: DflInputs<'_>,
    num_classes: usize,
    cfg: &NmsConfig,
    mode: HeadMode,
) -> Vec<Detection> {
    match mode {
        HeadMode::O2m => non_max_suppression_with_inputs(boxes, scores, dfl, num_classes, cfg),
        HeadMode::O2o => topk_detections(boxes, scores, num_classes, cfg),
    }
}

#[cfg(test)]
mod tests;
