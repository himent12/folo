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
use crate::fusion::{BoxFusion, FUSION_IOU, fuse_boxes};
use crate::model::HeadMode;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
        }
    }
}

impl NmsConfig {
    /// Validation settings: `conf = 0.001`, `iou = 0.7`, `max_det = 300`, multi-label.
    pub fn val() -> Self {
        Self { conf: 0.001, multi_label: true, ..Self::default() }
    }
}

/// Greedy NMS over `boxes` sorted by descending score. Returns the kept positions in order, stopping after `limit`
/// survivors. A box is suppressed when its IoU with a kept box is `> iou_thr`.
pub fn nms_sorted(boxes: &[Xyxy], iou_thr: f32, limit: usize) -> Vec<usize> {
    let n = boxes.len();
    let mut keep = Vec::new();
    if n == 0 || limit == 0 {
        return keep;
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
        keep.push(i);
        if keep.len() == limit {
            break;
        }
        let (bx1, by1, bx2, by2, ba) = (x1[i], y1[i], x2[i], y2[i], area[i]);
        let start = i + 1;
        let pass = |off: usize, sup: &mut [bool]| {
            let lo = start + off;
            let hi = lo + sup.len();
            let (cx1, cy1, cx2, cy2, ca) =
                (&x1[lo..hi], &y1[lo..hi], &x2[lo..hi], &y2[lo..hi], &area[lo..hi]);
            for k in 0..sup.len() {
                // Same arithmetic as `box_iou`, so results match it bit for bit.
                let w = (bx2.min(cx2[k]) - bx1.max(cx1[k])).max(0.0);
                let h = (by2.min(cy2[k]) - by1.max(cy1[k])).max(0.0);
                let inter = w * h;
                let union = ba + ca[k] - inter;
                let iou = if union > 0.0 { inter / union } else { 0.0 };
                sup[k] |= iou > iou_thr;
            }
        };
        let rest = &mut suppressed[start..];
        if rest.len() >= PAR_SUPPRESS {
            const CHUNK: usize = 4096;
            rest.par_chunks_mut(CHUNK).enumerate().for_each(|(c, sup)| pass(c * CHUNK, sup));
        } else {
            pass(0, rest);
        }
    }
    keep
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
    assert_eq!(boxes.len(), scores.len(), "batched_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_nms: boxes/classes length mismatch");
    let n = boxes.len();
    if n == 0 || max_det == 0 {
        return Vec::new();
    }
    assert!(n <= u32::MAX as usize, "batched_nms: too many boxes");
    let mut order: Vec<u32> = (0..n as u32).collect();
    score_order(scores, &mut order);

    if agnostic {
        let sorted: Vec<Xyxy> = order.iter().map(|&i| boxes[i as usize]).collect();
        return nms_sorted(&sorted, iou_thr, max_det).into_iter().map(|k| order[k] as usize).collect();
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
    kept.into_iter().map(|i| i as usize).collect()
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

/// Cap the candidates (given in `(idx, class)` order) at the `cap` best by score, ties to the earlier candidate,
/// then run batched NMS, fuse the kept boxes when [`NmsConfig::fusion`] asks for it (with `precision`, indexed by
/// source index, for [`BoxFusion::Dfl`]), and build the detections.
fn finish(
    mut cands: Vec<Cand>,
    cfg: &NmsConfig,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
) -> Vec<Detection> {
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
    let boxes: Vec<Xyxy> = cands.iter().map(|c| bbox_of(c.idx)).collect();
    let scores: Vec<f32> = cands.iter().map(|c| c.score).collect();
    let classes: Vec<u32> = cands.iter().map(|c| c.class).collect();
    let kept = batched_nms(&boxes, &scores, &classes, cfg.iou, cfg.max_det, cfg.agnostic);
    let fused = match cfg.fusion {
        BoxFusion::None => None,
        BoxFusion::Vote => {
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, None))
        }
        BoxFusion::Dfl => {
            let of = precision.map(|p| move |k: usize| p[cands[k].idx as usize]);
            let of = of.as_ref().map(|f| f as &dyn Fn(usize) -> [f32; 4]);
            Some(fuse_boxes(&kept, &boxes, &scores, &classes, cfg.agnostic, cfg.fusion_iou, of))
        }
    };
    kept.iter()
        .enumerate()
        .map(|(i, &k)| Detection {
            bbox: fused.as_ref().map_or(boxes[k], |f| f[i]),
            score: scores[k],
            class: classes[k],
        })
        .collect()
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
    let cands: Vec<Cand> = dets
        .iter()
        .enumerate()
        .filter(|(_, d)| d.score > cfg.conf && class_allowed(cfg, d.class))
        .map(|(i, d)| Cand { score: d.score, idx: i as u32, class: d.class })
        .collect();
    finish(cands, cfg, |i| dets[i as usize].bbox, None)
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
    non_max_suppression_with(boxes, scores, None, num_classes, cfg)
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
    assert!(precision.is_none_or(|p| p.len() == boxes.len()), "non_max_suppression: precision must be [A]");
    assert_eq!(
        scores.len(),
        boxes.len() * num_classes,
        "non_max_suppression: scores must be [A, num_classes]"
    );
    assert!(boxes.len() <= u32::MAX as usize, "non_max_suppression: too many anchors");
    if num_classes == 0 || boxes.is_empty() {
        return Vec::new();
    }
    let cands = extract_candidates(scores, boxes.len(), num_classes, num_classes, 0, cfg);
    finish(cands, cfg, |i| boxes[i as usize], precision)
}

/// [`non_max_suppression`] of one image's decoded outputs as rows `[A, 4 + num_classes]` (`x1, y1, x2, y2`, then
/// the class scores: [`crate::model::HeadOutput::decoded_on_device`]'s layout), read in place. Same result.
///
/// # Panics
/// If `rows.len()` is not a multiple of `4 + num_classes`, or there are more than `u32::MAX` anchors.
pub fn non_max_suppression_rows(rows: &[f32], num_classes: usize, cfg: &NmsConfig) -> Vec<Detection> {
    non_max_suppression_rows_with(rows, None, num_classes, cfg)
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
    let w = 4 + num_classes;
    assert!(rows.len().is_multiple_of(w), "non_max_suppression_rows: rows must be [A, 4 + num_classes]");
    let a = rows.len() / w;
    assert!(a <= u32::MAX as usize, "non_max_suppression_rows: too many anchors");
    if num_classes == 0 || a == 0 {
        return Vec::new();
    }
    assert!(precision.is_none_or(|p| p.len() == a), "non_max_suppression_rows: precision must be [A]");
    let cands = extract_candidates(rows, a, num_classes, w, 4, cfg);
    let bbox_of = |i: u32| {
        let r = &rows[i as usize * w..];
        Xyxy::new(r[0], r[1], r[2], r[3])
    };
    finish(cands, cfg, bbox_of, precision)
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
    non_max_suppression_pairs_with(pairs, bbox_of, None, cfg)
}

/// [`non_max_suppression_pairs`] with the anchors' side precisions (indexed by anchor) for [`BoxFusion::Dfl`].
pub fn non_max_suppression_pairs_with(
    mut pairs: Vec<(u32, u32, f32)>,
    bbox_of: impl Fn(u32) -> Xyxy,
    precision: Precisions,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    debug_assert!(cfg.multi_label, "non_max_suppression_pairs is the multi-label path");
    pairs.sort_unstable_by_key(|&(a, c, _)| (a, c));
    let cands = pairs
        .into_iter()
        .filter(|&(_, c, s)| s > cfg.conf && class_allowed(cfg, c))
        .map(|(idx, class, score)| Cand { score, idx, class })
        .collect();
    finish(cands, cfg, bbox_of, precision)
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
    let extract = |a: usize, row: &[f32], out: &mut Vec<Cand>| {
        if cfg.multi_label {
            for (c, &s) in row.iter().enumerate() {
                if s > cfg.conf && class_allowed(cfg, c as u32) {
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
            if bs > cfg.conf && class_allowed(cfg, best as u32) {
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
    assert!(cands.len() <= u32::MAX as usize, "nms_candidates: too many candidates");
    assert!(precision.is_none_or(|p| p.len() == cands.len()), "nms_candidates: precision must match cands");
    debug_assert!(cands.windows(2).all(|w| w[0].anchor < w[1].anchor), "nms_candidates: unsorted");
    // Positions in anchor order tie-break exactly as anchor indices do.
    let list: Vec<Cand> = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score > cfg.conf && class_allowed(cfg, c.class))
        .map(|(i, c)| Cand { score: c.score, idx: i as u32, class: c.class })
        .collect();
    finish(list, cfg, |i| cands[i as usize].bbox, precision)
}

/// Ranks `(score, key)` pairs by descending score, ties by ascending key (NaN never reaches here: it fails every
/// `> conf` test).
fn by_score<K: Ord>(a: &(f32, K), b: &(f32, K)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

/// The `k` best of `v` by [`by_score`], in that order.
fn top_sorted<K: Ord + Copy>(v: &mut Vec<(f32, K)>, k: usize) {
    if v.len() > k {
        if k == 0 {
            v.clear();
            return;
        }
        v.select_nth_unstable_by(k - 1, by_score);
        v.truncate(k);
    }
    v.sort_unstable_by(by_score);
}

/// The post-process of the one-to-one (NMS-free) head for one image (docs/MODEL.md §6.6): with `k =
/// min(max_det, A)`, the `k` anchors with the highest max-class score (ties: lower anchor), then
///
/// - `multi_label` (validation): the `k` best (anchor, class) pairs among those anchors (ties: the anchor's rank,
///   then the class), each kept if its score is `> conf`;
/// - single-label (prediction): each anchor once, with its best class (first maximum), kept if `> conf`.
///
/// Then the class filter. No NMS and no candidate cap (`iou`, `max_candidates` and `agnostic` are unused).
/// Returns at most `max_det` detections sorted by descending score. Anchors whose best score is not `> conf` can
/// contribute no pair above it, so they are dropped before ranking: the result is the spec's.
///
/// # Panics
/// If `scores.len() != boxes.len() * num_classes`, or there are more than `u32::MAX` anchors.
pub fn topk_detections(
    boxes: &[Xyxy],
    scores: &[f32],
    num_classes: usize,
    cfg: &NmsConfig,
) -> Vec<Detection> {
    assert_eq!(scores.len(), boxes.len() * num_classes, "topk_detections: scores must be [A, num_classes]");
    assert!(boxes.len() <= u32::MAX as usize, "topk_detections: too many anchors");
    if num_classes == 0 || boxes.is_empty() {
        return Vec::new();
    }
    let k = cfg.max_det.min(boxes.len());
    // (best score, (anchor, best class)): the class is a function of the anchor, so ties rank by anchor.
    let mut anchors: Vec<(f32, (u32, u32))> = Vec::new();
    for (a, row) in scores.chunks_exact(num_classes).enumerate() {
        let (mut bc, mut bs) = (0usize, row[0]);
        for (c, &s) in row.iter().enumerate().skip(1) {
            if s > bs {
                (bc, bs) = (c, s);
            }
        }
        if bs > cfg.conf {
            anchors.push((bs, (a as u32, bc as u32)));
        }
    }
    top_sorted(&mut anchors, k);
    let dets: Vec<Detection> = if cfg.multi_label {
        let mut pairs: Vec<(f32, (u32, u32))> = Vec::new();
        for (rank, &(_, (a, _))) in anchors.iter().enumerate() {
            let row = &scores[a as usize * num_classes..(a as usize + 1) * num_classes];
            for (c, &s) in row.iter().enumerate() {
                if s > cfg.conf {
                    pairs.push((s, (rank as u32, c as u32)));
                }
            }
        }
        top_sorted(&mut pairs, k);
        pairs
            .into_iter()
            .map(|(s, (r, c))| Detection {
                bbox: boxes[anchors[r as usize].1.0 as usize],
                score: s,
                class: c,
            })
            .collect()
    } else {
        anchors.iter().map(|&(s, (a, c))| Detection { bbox: boxes[a as usize], score: s, class: c }).collect()
    };
    dets.into_iter().filter(|d| class_allowed(cfg, d.class)).collect()
}

/// [`topk_detections`] single-label from the candidates already extracted (one per anchor whose best class scores
/// `> conf`, any order; folo-core's `Tensor::detect_candidates`): the same detections.
pub fn topk_candidates(cands: &[Candidate], cfg: &NmsConfig) -> Vec<Detection> {
    assert!(cands.len() <= u32::MAX as usize, "topk_candidates: too many candidates");
    // (score, (anchor, position)): ties rank by anchor, as the full selection does.
    let mut ranked: Vec<(f32, (u32, u32))> = cands
        .iter()
        .enumerate()
        .filter(|(_, c)| c.score > cfg.conf)
        .map(|(i, c)| (c.score, (c.anchor, i as u32)))
        .collect();
    top_sorted(&mut ranked, cfg.max_det);
    ranked
        .into_iter()
        .map(|(_, (_, i))| cands[i as usize])
        .map(|c| Detection { bbox: c.bbox, score: c.score, class: c.class })
        .filter(|d| class_allowed(cfg, d.class))
        .collect()
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
    postprocess_with(boxes, scores, None, num_classes, cfg, mode)
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
    match mode {
        HeadMode::O2m => non_max_suppression_with(boxes, scores, precision, num_classes, cfg),
        HeadMode::O2o => topk_detections(boxes, scores, num_classes, cfg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> Xyxy {
        Xyxy::new(x1, y1, x2, y2)
    }

    #[test]
    fn suppresses_overlapping_same_class_only() {
        let boxes = [b(0., 0., 10., 10.), b(1., 1., 11., 11.), b(0., 0., 10., 10.), b(50., 50., 60., 60.)];
        let scores = [0.9, 0.8, 0.7, 0.6];
        let classes = [0, 0, 1, 0];
        // Box 1 overlaps box 0 with IoU 81/119 = 0.68 > 0.5; box 2 is another class.
        assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 300, false), vec![0, 2, 3]);
        assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 300, true), vec![0, 3]);
        // Threshold is strict: IoU 0.68 is not > 0.7.
        assert_eq!(batched_nms(&boxes, &scores, &classes, 0.7, 300, false), vec![0, 1, 2, 3]);
        assert_eq!(batched_nms(&boxes, &scores, &classes, 0.5, 2, false), vec![0, 2]);
    }

    #[test]
    fn sparse_class_ids() {
        let boxes = [b(0., 0., 10., 10.), b(1., 1., 11., 11.), b(0., 0., 10., 10.)];
        let classes = [u32::MAX, u32::MAX, 7];
        assert_eq!(batched_nms(&boxes, &[0.9, 0.8, 0.7], &classes, 0.5, 300, false), vec![0, 2]);
    }

    /// NMS from `[A, 4 + nc]` rows is NMS from the split boxes and scores, single- and multi-label, with ties
    /// and enough anchors for the parallel extraction.
    #[test]
    fn rows_match_split_outputs() {
        let (a, nc) = (5000usize, 3usize);
        let mut state = 12345u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut rows = Vec::with_capacity(a * (4 + nc));
        let (mut boxes, mut scores) = (Vec::new(), Vec::new());
        for _ in 0..a {
            let (x, y) = (next() * 600.0, next() * 600.0);
            let bx = Xyxy::new(x, y, x + 5.0 + next() * 60.0, y + 5.0 + next() * 60.0);
            rows.extend_from_slice(&[bx.x1, bx.y1, bx.x2, bx.y2]);
            boxes.push(bx);
            for _ in 0..nc {
                let s = (next() * 16.0).floor() / 16.0; // many ties
                rows.push(s);
                scores.push(s);
            }
        }
        for multi_label in [false, true] {
            let cfg = NmsConfig { conf: 0.05, multi_label, max_candidates: 4000, ..NmsConfig::default() };
            let want = non_max_suppression(&boxes, &scores, nc, &cfg);
            assert!(want.len() > 50);
            assert_eq!(non_max_suppression_rows(&rows, nc, &cfg), want, "multi_label {multi_label}");
        }
        assert!(non_max_suppression_rows(&[], nc, &NmsConfig::val()).is_empty());
    }

    /// NMS from shuffled `(anchor, class, score)` pairs is NMS from the rows they come from: tied scores on
    /// overlapping boxes (the winner is the first in `(anchor, class)` order), scores below, at and above `conf`,
    /// NaNs, a class filter, agnostic NMS, the candidate cap, and no pairs at all.
    #[test]
    fn pairs_match_rows_nms() {
        let (a, nc) = (40usize, 3usize);
        let conf = 0.25f32;
        let levels = [0.5, 0.5, conf, 0.3, f32::NAN, 0.1, 0.5, 0.0, f32::from_bits(conf.to_bits() + 1)];
        let mut rows = Vec::with_capacity(a * (4 + nc));
        let mut pairs = Vec::new();
        for i in 0..a {
            // Groups of four boxes one pixel apart (IoU about 0.9), the groups far apart.
            let x = (i / 4 * 30 + i % 4) as f32;
            rows.extend_from_slice(&[x, 0.0, x + 20.0, 20.0]);
            for c in 0..nc {
                let s = levels[(i * 3 + c * 5) % levels.len()];
                rows.push(s);
                pairs.push((i as u32, c as u32, s));
            }
        }
        // Two lone boxes with nothing above conf: no detections, though nothing would suppress them.
        for (i, s) in [(a, [conf, f32::NAN, 0.1]), (a + 1, [0.0, conf, f32::NAN])] {
            let x = (i * 30) as f32;
            rows.extend_from_slice(&[x, 0.0, x + 20.0, 20.0]);
            rows.extend_from_slice(&s);
            pairs.extend((0..nc).map(|c| (i as u32, c as u32, s[c])));
        }
        // A fixed shuffle that puts later anchors before earlier ones.
        let mut state = 7u64;
        for k in (1..pairs.len()).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            pairs.swap(k, (state >> 33) as usize % (k + 1));
        }
        let bbox = |i: u32| {
            let r = &rows[i as usize * (4 + nc)..];
            Xyxy::new(r[0], r[1], r[2], r[3])
        };
        let val = NmsConfig { conf, ..NmsConfig::val() };
        for cfg in [
            val.clone(),
            NmsConfig { classes: Some(vec![0, 2]), ..val.clone() },
            NmsConfig { agnostic: true, ..val.clone() },
            NmsConfig { max_candidates: 9, max_det: 5, ..val.clone() },
        ] {
            let want = non_max_suppression_rows(&rows, nc, &cfg);
            assert!(want.len() >= 5, "{cfg:?}: {want:?}");
            assert_eq!(non_max_suppression_pairs(pairs.clone(), bbox, &cfg), want, "{cfg:?}");
        }
        // Nothing above conf, as rows or as (no) pairs.
        let none = NmsConfig { conf: 0.5, ..val };
        assert!(non_max_suppression_rows(&rows, nc, &none).is_empty());
        assert!(non_max_suppression_pairs(pairs, bbox, &none).is_empty());
        assert!(non_max_suppression_pairs(Vec::new(), bbox, &NmsConfig::val()).is_empty());
    }

    #[test]
    fn candidates_match_full_nms() {
        // 50 anchors, 4 classes, many overlapping boxes and tied scores.
        let (a, nc) = (50usize, 4usize);
        let boxes: Vec<Xyxy> = (0..a)
            .map(|i| b((i % 7) as f32, (i % 5) as f32, (i % 7) as f32 + 9.0, (i % 5) as f32 + 9.0))
            .collect();
        let scores: Vec<f32> = (0..a * nc).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
        for cfg in [
            NmsConfig { conf: 0.25, iou: 0.5, ..NmsConfig::default() },
            NmsConfig { conf: 0.05, iou: 0.3, max_det: 7, max_candidates: 20, ..NmsConfig::default() },
            NmsConfig { classes: Some(vec![1, 3]), agnostic: true, ..NmsConfig::default() },
        ] {
            let want = non_max_suppression(&boxes, &scores, nc, &cfg);
            let cands: Vec<Candidate> = (0..a)
                .filter_map(|i| {
                    let row = &scores[i * nc..(i + 1) * nc];
                    let (mut best, mut bs) = (0, row[0]);
                    for (c, &s) in row.iter().enumerate().skip(1) {
                        if s > bs {
                            (best, bs) = (c, s);
                        }
                    }
                    (bs > cfg.conf).then_some(Candidate {
                        anchor: i as u32,
                        class: best as u32,
                        score: bs,
                        bbox: boxes[i],
                    })
                })
                .collect();
            assert_eq!(nms_candidates(&cands, &cfg), want, "{cfg:?}");
        }
    }

    #[test]
    fn ties_keep_lower_index() {
        let boxes = [b(0., 0., 10., 10.), b(0., 0., 10., 10.)];
        assert_eq!(batched_nms(&boxes, &[0.5, 0.5], &[0, 0], 0.5, 10, false), vec![0]);
    }

    #[test]
    fn post_process_filters() {
        // Three anchors, two classes.
        let boxes = [b(0., 0., 10., 10.), b(0., 0., 10., 10.), b(20., 20., 30., 30.)];
        let scores = [0.9, 0.3, 0.2, 0.8, 0.1, 0.05];
        let cfg = NmsConfig { conf: 0.25, iou: 0.5, ..NmsConfig::default() };
        let out = non_max_suppression(&boxes, &scores, 2, &cfg);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].class, out[0].score), (0, 0.9));
        assert_eq!((out[1].class, out[1].score), (1, 0.8));
        // Multi-label adds anchor 0's class-1 candidate (0.3), which anchor 1's identical 0.8 box then suppresses.
        let ml = NmsConfig { multi_label: true, ..cfg.clone() };
        let out = non_max_suppression(&boxes, &scores, 2, &ml);
        assert_eq!(out.iter().map(|d| d.score).collect::<Vec<_>>(), vec![0.9, 0.8]);
        // Class filter after argmax: anchor 1's best class is 1, anchor 0's is 0.
        let only1 = NmsConfig { classes: Some(vec![1]), ..cfg.clone() };
        let out = non_max_suppression(&boxes, &scores, 2, &only1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, 1);
        // Candidate cap keeps the best.
        let capped = NmsConfig { max_candidates: 1, ..cfg };
        assert_eq!(non_max_suppression(&boxes, &scores, 2, &capped).len(), 1);
        assert!(non_max_suppression(&[], &[], 2, &NmsConfig::default()).is_empty());
    }

    #[test]
    fn detection_serialises_flat() {
        let d = Detection { bbox: b(1., 2., 3., 4.), score: 0.5, class: 7 };
        let v = serde_json::to_value(d).unwrap();
        assert_eq!(
            v,
            serde_json::json!({ "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "score": 0.5, "class": 7 })
        );
        assert_eq!(serde_json::from_value::<Detection>(v).unwrap(), d);
    }
}
