//! Fast-NMS (Bolya et al., YOLACT, ICCV 2019) and Cluster-NMS (Zheng et al., IEEE TCYB 2021)
//! with optional DIoU overlap, score decay (Linear / Gaussian SPM), and weighted-cluster box
//! refinement.
//!
//! Both algorithms operate over an upper-triangular pairwise overlap matrix `X` (`i < j`,
//! descending score order, deterministic lower-input-index tie-breaking):
//!
//! - **Fast-NMS** ([`fast_nms`], [`batched_fast_nms`], [`fast_nms_detections`]): single-pass
//!   vectorized column-wise maximum `max_{i < j} X[i, j] <= iou_thr`. Runs in one matrix sweep
//!   with no sequential dependency; may over-suppress chained overlaps where `A` suppresses `B`
//!   and `B` suppresses `C`.
//! - **Cluster-NMS** ([`cluster_nms`], [`batched_cluster_nms`], [`cluster_nms_detections`]):
//!   iterative masked upper-triangular fixed-point `b^{(t)}[j] = !any_{i < j} (b^{(t-1)}[i] &&
//!   X[i, j] > iou_thr)` converging in typically 2–4 iterations. At convergence, hard Cluster-NMS
//!   is **provably identical** to sequential greedy NMS (and DIoU-NMS when `use_diou` is enabled).
//!   Optional extensions:
//!   - [`ClusterScoreDecay::Linear`] / [`ClusterScoreDecay::Gaussian`]: decays overlapping boxes'
//!     scores against surviving cluster centers instead of hard-dropping.
//!   - `weighted_box`: refines each kept cluster center as the `score * IoU`-weighted average of
//!     its cluster members (convex combination in `f64`, ignoring non-finite member boxes).

use crate::boxes::Xyxy;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::Detection;
use super::kernel::bucket_by_class;
use super::matrix::exact_iou_f64;

/// Total candidate count above which class buckets run in parallel.
const PAR_BUCKETS: usize = 2048;

/// Optional score-decay rule applied on top of Cluster-NMS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClusterScoreDecay {
    /// Hard suppression: drop suppressed boxes completely (identical to greedy NMS at convergence).
    #[default]
    None,
    /// Linear score decay (Cluster-NMS_S): multiply each box `j`'s positive score by
    /// `prod_{i < j, kept[i], IoU(i,j) > iou_thr} (1 - IoU(i, j))` and retain scores `> score_thr`.
    Linear,
    /// Gaussian SPM score decay: multiply each box `j`'s positive score by
    /// `prod_{i < j, kept[i], IoU(i,j) > 0} exp(-IoU(i, j)² / sigma)` and retain scores `> score_thr`.
    Gaussian {
        /// Gaussian decay bandwidth `sigma > 0` (e.g. `0.5`).
        sigma: f32,
    },
}

/// Configuration for [`cluster_nms`], [`batched_cluster_nms`], and [`cluster_nms_detections`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusterNmsConfig {
    /// Overlap suppression threshold (`overlap > iou_thr`, default `0.7`).
    pub iou_thr: f32,
    /// Minimum confidence threshold (`score > score_thr`, default `0.0`).
    pub score_thr: f32,
    /// Maximum number of detections to return (default `300`).
    pub max_det: usize,
    /// Class-agnostic suppression when `true` (default `false`: strict per-class isolation).
    pub agnostic: bool,
    /// Use DIoU (`IoU - rho² / c²`) instead of plain IoU for upper-triangular suppression.
    pub use_diou: bool,
    /// Score-decay mode (default [`ClusterScoreDecay::None`]).
    pub score_decay: ClusterScoreDecay,
    /// Enable weighted-cluster box refinement (`b'_k = sum(w_j b_j) / sum(w_j)` with
    /// `w_j = score_j * IoU(b_k, b_j)` over cluster members) for hard Cluster-NMS.
    pub weighted_box: bool,
    /// Optional cap on fixed-point iterations (`None` = iterate until exact convergence, at most `m`;
    /// `Some(1)` recovers single-pass Fast-NMS).
    pub max_iters: Option<usize>,
}

impl Default for ClusterNmsConfig {
    fn default() -> Self {
        Self {
            iou_thr: 0.7,
            score_thr: 0.0,
            max_det: 300,
            agnostic: false,
            use_diou: false,
            score_decay: ClusterScoreDecay::None,
            weighted_box: false,
            max_iters: None,
        }
    }
}

/// One surviving entry from [`cluster_nms`] or [`batched_cluster_nms`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterNmsEntry {
    /// Index in the caller's input slice.
    pub index: usize,
    /// Final box (refined when `weighted_box` is enabled, else original input box).
    pub bbox: Xyxy,
    /// Final score (decayed when `score_decay != None`, else original input score).
    pub score: f32,
}

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Exact collapsed `f64` DIoU (`IoU - rho² / c²`), returning `0.0` on non-finite coordinates.
#[inline]
fn exact_diou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
    if !box_is_finite(a) || !box_is_finite(b) {
        return 0.0;
    }
    let iou = exact_iou_f64(a, b);
    let (ax1, ay1) = (a.x1 as f64, a.y1 as f64);
    let (ax2, ay2) = (a.x2.max(a.x1) as f64, a.y2.max(a.y1) as f64);
    let (bx1, by1) = (b.x1 as f64, b.y1 as f64);
    let (bx2, by2) = (b.x2.max(b.x1) as f64, b.y2.max(b.y1) as f64);
    let cw = (ax2.max(bx2) - ax1.min(bx1)).max(0.0);
    let ch = (ay2.max(by2) - ay1.min(by1)).max(0.0);
    let m_c = cw.max(ch);
    if m_c <= 0.0 {
        return iou;
    }
    let inv_c = 1.0 / m_c;
    let dx = ((bx1 - ax1) + (bx2 - ax2)) * 0.5 * inv_c;
    let dy = ((by1 - ay1) + (by2 - ay2)) * 0.5 * inv_c;
    let cws = cw * inv_c;
    let chs = ch * inv_c;
    let c2 = cws * cws + chs * chs;
    let dist = ((dx * dx + dy * dy) / c2).min(1.0);
    iou - dist
}

/// Filter finite scores `> score_thr` and sort indices by descending score (ties: smaller input index).
fn sorted_valid_indices(scores: &[f32], score_thr: f32) -> Vec<usize> {
    let mut order: Vec<usize> =
        (0..scores.len()).filter(|&i| scores[i].is_finite() && scores[i] > score_thr).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    order
}

/// Fast-NMS (Bolya et al., YOLACT 2019) over a single class's `boxes` and `scores`.
///
/// Drops non-finite scores (`NaN`, `±Inf`) and scores `<= score_thr`, sorts descending by score
/// (ties: lower input index), computes the upper-triangular column maximum
/// `max_{i < j} IoU(b_i, b_j)` in one vectorized sweep, and keeps entries with `max_iou <= iou_thr`
/// up to `limit`.
///
/// # Panics
/// If `boxes.len() != scores.len()`.
pub fn fast_nms(boxes: &[Xyxy], scores: &[f32], iou_thr: f32, score_thr: f32, limit: usize) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "fast_nms: boxes/scores length mismatch");
    if boxes.is_empty() || limit == 0 || iou_thr.is_nan() {
        return Vec::new();
    }
    let order = sorted_valid_indices(scores, score_thr);
    let m = order.len();
    if m == 0 {
        return Vec::new();
    }
    let sorted_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let thr = iou_thr as f64;
    let mut max_iou = vec![0.0f64; m];
    for i in 0..m {
        let bi = &sorted_boxes[i];
        if !box_is_finite(bi) {
            continue;
        }
        for j in (i + 1)..m {
            let ov = exact_iou_f64(bi, &sorted_boxes[j]);
            if ov > max_iou[j] {
                max_iou[j] = ov;
            }
        }
    }
    let mut kept = Vec::with_capacity(m.min(limit));
    for (pos, &orig_idx) in order.iter().enumerate() {
        if max_iou[pos] <= thr {
            kept.push(orig_idx);
            if kept.len() == limit {
                break;
            }
        }
    }
    kept
}

/// Class-aware (or class-agnostic) Fast-NMS returning surviving input indices sorted by descending
/// score (ties: smaller input index), capped at `max_det`.
///
/// # Panics
/// If `boxes`, `scores`, and `classes` differ in length.
#[allow(clippy::too_many_arguments)]
pub fn batched_fast_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    iou_thr: f32,
    score_thr: f32,
    max_det: usize,
    agnostic: bool,
) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "batched_fast_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_fast_nms: boxes/classes length mismatch");
    let n = boxes.len();
    if n == 0 || max_det == 0 || iou_thr.is_nan() {
        return Vec::new();
    }
    assert!(n <= u32::MAX as usize, "batched_fast_nms: too many boxes");
    if agnostic {
        return fast_nms(boxes, scores, iou_thr, score_thr, max_det);
    }
    let order: Vec<u32> = (0..n as u32).collect();
    let (bucketed, ranges) = bucket_by_class(&order, classes);
    let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<usize> {
        let ids = &bucketed[lo..hi];
        let b: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let s: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
        fast_nms(&b, &s, iou_thr, score_thr, max_det).into_iter().map(|k| ids[k] as usize).collect()
    };
    let mut kept: Vec<usize> = if n >= PAR_BUCKETS {
        ranges.par_iter().flat_map_iter(run_bucket).collect()
    } else {
        ranges.iter().flat_map(run_bucket).collect()
    };
    kept.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    kept.truncate(max_det);
    kept
}

/// Apply Fast-NMS directly to a slice of [`Detection`]s.
pub fn fast_nms_detections(
    dets: &[Detection],
    iou_thr: f32,
    score_thr: f32,
    max_det: usize,
    agnostic: bool,
) -> Vec<Detection> {
    let boxes: Vec<Xyxy> = dets.iter().map(|d| d.bbox).collect();
    let scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
    let classes: Vec<u32> = dets.iter().map(|d| d.class).collect();
    batched_fast_nms(&boxes, &scores, &classes, iou_thr, score_thr, max_det, agnostic)
        .into_iter()
        .map(|i| dets[i])
        .collect()
}

/// Cluster-NMS (Zheng et al., 2021) over a single class's `boxes` and `scores`.
///
/// Builds the strict upper-triangular overlap matrix `X[i, j]` (`i < j`), iterates the masked
/// upper-triangular fixed point until convergence (or `cfg.max_iters`), and optionally applies
/// [`ClusterScoreDecay`] and/or `weighted_box` refinement.
///
/// # Panics
/// If `boxes.len() != scores.len()`, `cfg.score_decay` has non-finite or `<= 0` Gaussian `sigma`,
/// or `cfg.max_iters == Some(0)`.
pub fn cluster_nms(boxes: &[Xyxy], scores: &[f32], cfg: &ClusterNmsConfig) -> Vec<ClusterNmsEntry> {
    assert_eq!(boxes.len(), scores.len(), "cluster_nms: boxes/scores length mismatch");
    if let ClusterScoreDecay::Gaussian { sigma } = cfg.score_decay {
        assert!(
            sigma.is_finite() && sigma > 0.0,
            "cluster_nms: Gaussian sigma must be finite and > 0, got {sigma}"
        );
    }
    if let Some(iters) = cfg.max_iters {
        assert!(iters > 0, "cluster_nms: max_iters must be >= 1");
    }
    if boxes.is_empty() || cfg.max_det == 0 || cfg.iou_thr.is_nan() {
        return Vec::new();
    }
    let order = sorted_valid_indices(scores, cfg.score_thr);
    let m = order.len();
    if m == 0 {
        return Vec::new();
    }
    let sorted_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let sorted_scores: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    let thr = cfg.iou_thr as f64;

    // Dense strict upper-triangular overlap matrix `X` (`m * m` row-major, only `i < j` populated)
    // and plain IoU cache (needed for score decay and weighted-cluster refinement when `use_diou`).
    let mut ov_mat = vec![0.0f64; m * m];
    let need_plain_iou =
        cfg.use_diou && (cfg.weighted_box || !matches!(cfg.score_decay, ClusterScoreDecay::None));
    let mut plain_iou_mat = if need_plain_iou { vec![0.0f64; m * m] } else { Vec::new() };
    let mut adj = vec![false; m * m];

    for i in 0..m {
        let bi = &sorted_boxes[i];
        let bi_fin = box_is_finite(bi);
        let row_off = i * m;
        for j in (i + 1)..m {
            if !bi_fin || !box_is_finite(&sorted_boxes[j]) {
                continue;
            }
            let ov = if cfg.use_diou {
                if need_plain_iou {
                    plain_iou_mat[row_off + j] = exact_iou_f64(bi, &sorted_boxes[j]);
                }
                exact_diou_f64(bi, &sorted_boxes[j])
            } else {
                exact_iou_f64(bi, &sorted_boxes[j])
            };
            ov_mat[row_off + j] = ov;
            adj[row_off + j] = ov > thr;
        }
    }

    // Iterative masked upper-triangular fixed-point:
    // b_next[j] = !any_{i < j} (b[i] && adj[i, j]).
    let max_iters = cfg.max_iters.unwrap_or(m).min(m);
    let mut alive = vec![true; m];
    let mut next_alive = vec![true; m];
    for _ in 0..max_iters {
        next_alive.fill(true);
        for i in 0..m {
            if !alive[i] {
                continue;
            }
            let row = &adj[i * m..(i + 1) * m];
            for j in (i + 1)..m {
                if row[j] {
                    next_alive[j] = false;
                }
            }
        }
        if next_alive == alive {
            break;
        }
        std::mem::swap(&mut alive, &mut next_alive);
    }

    let plain_iou = |i: usize, j: usize| -> f64 {
        if need_plain_iou { plain_iou_mat[i * m + j] } else { ov_mat[i * m + j] }
    };

    // Optional weighted-cluster box refinement for surviving centers `k`:
    // assign each suppressed box `j` to its primary surviving owner `k < j` (highest plain IoU > thr,
    // ties to earliest/highest-scoring `k`), then compute convex average weighted by `score_j * IoU(k, j)`.
    let refined_boxes: Option<Vec<Xyxy>> = if cfg.weighted_box {
        let mut owner = vec![usize::MAX; m];
        for j in 0..m {
            if alive[j] {
                owner[j] = j;
                continue;
            }
            let mut best_k = usize::MAX;
            let mut best_ov = thr;
            for (k, &is_alive) in alive.iter().enumerate().take(j) {
                if !is_alive {
                    continue;
                }
                let ov = plain_iou(k, j);
                if ov > best_ov {
                    best_ov = ov;
                    best_k = k;
                }
            }
            owner[j] = best_k;
        }
        let mut out_boxes = sorted_boxes.clone();
        for k in 0..m {
            if !alive[k] || !box_is_finite(&sorted_boxes[k]) || sorted_scores[k] <= 0.0 {
                continue;
            }
            let mut w_sum = 0.0f64;
            let mut sx1 = 0.0f64;
            let mut sy1 = 0.0f64;
            let mut sx2 = 0.0f64;
            let mut sy2 = 0.0f64;
            for j in k..m {
                if owner[j] != k || !box_is_finite(&sorted_boxes[j]) || sorted_scores[j] <= 0.0 {
                    continue;
                }
                let ov = if j == k { 1.0 } else { plain_iou(k, j) };
                let w = (sorted_scores[j] as f64) * ov;
                if w > 0.0 && w.is_finite() {
                    w_sum += w;
                    sx1 += w * sorted_boxes[j].x1 as f64;
                    sy1 += w * sorted_boxes[j].y1 as f64;
                    sx2 += w * sorted_boxes[j].x2 as f64;
                    sy2 += w * sorted_boxes[j].y2 as f64;
                }
            }
            if w_sum > 0.0 && w_sum.is_finite() {
                out_boxes[k] = Xyxy::new(
                    (sx1 / w_sum) as f32,
                    (sy1 / w_sum) as f32,
                    (sx2 / w_sum) as f32,
                    (sy2 / w_sum) as f32,
                );
            }
        }
        Some(out_boxes)
    } else {
        None
    };

    let final_boxes = refined_boxes.as_deref().unwrap_or(&sorted_boxes);
    let mut entries: Vec<ClusterNmsEntry> = Vec::with_capacity(m.min(cfg.max_det));

    match cfg.score_decay {
        ClusterScoreDecay::None => {
            for pos in 0..m {
                if alive[pos] {
                    entries.push(ClusterNmsEntry {
                        index: order[pos],
                        bbox: final_boxes[pos],
                        score: sorted_scores[pos],
                    });
                    if entries.len() == cfg.max_det {
                        break;
                    }
                }
            }
        }
        ClusterScoreDecay::Linear => {
            for j in 0..m {
                let s = sorted_scores[j];
                let decayed = if s > 0.0 {
                    let mut factor = 1.0f64;
                    for i in 0..j {
                        if alive[i] && adj[i * m + j] {
                            let iou = plain_iou(i, j);
                            factor *= (1.0 - iou).max(0.0);
                        }
                    }
                    (s as f64 * factor) as f32
                } else {
                    s
                };
                if decayed > cfg.score_thr {
                    entries.push(ClusterNmsEntry { index: order[j], bbox: final_boxes[j], score: decayed });
                }
            }
            entries.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.index.cmp(&b.index)));
            entries.truncate(cfg.max_det);
        }
        ClusterScoreDecay::Gaussian { sigma } => {
            let inv_sigma = 1.0 / (sigma as f64);
            for j in 0..m {
                let s = sorted_scores[j];
                let decayed = if s > 0.0 {
                    let mut sum_sq = 0.0f64;
                    for (i, &is_alive) in alive.iter().enumerate().take(j) {
                        if is_alive {
                            let iou = plain_iou(i, j);
                            if iou > 0.0 {
                                sum_sq += iou * iou;
                            }
                        }
                    }
                    (s as f64 * (-sum_sq * inv_sigma).exp()) as f32
                } else {
                    s
                };
                if decayed > cfg.score_thr {
                    entries.push(ClusterNmsEntry { index: order[j], bbox: final_boxes[j], score: decayed });
                }
            }
            entries.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.index.cmp(&b.index)));
            entries.truncate(cfg.max_det);
        }
    }
    entries
}

/// Class-aware (or class-agnostic when `cfg.agnostic` is `true`) Cluster-NMS returning
/// [`ClusterNmsEntry`] sorted by descending score (ties: smaller input index), capped at
/// `cfg.max_det`.
///
/// # Panics
/// If `boxes`, `scores`, and `classes` differ in length, or if `cfg` has invalid parameters.
pub fn batched_cluster_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    cfg: &ClusterNmsConfig,
) -> Vec<ClusterNmsEntry> {
    assert_eq!(boxes.len(), scores.len(), "batched_cluster_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_cluster_nms: boxes/classes length mismatch");
    let n = boxes.len();
    if n == 0 || cfg.max_det == 0 || cfg.iou_thr.is_nan() {
        return Vec::new();
    }
    assert!(n <= u32::MAX as usize, "batched_cluster_nms: too many boxes");
    if cfg.agnostic {
        return cluster_nms(boxes, scores, cfg);
    }
    let order: Vec<u32> = (0..n as u32).collect();
    let (bucketed, ranges) = bucket_by_class(&order, classes);
    let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<ClusterNmsEntry> {
        let ids = &bucketed[lo..hi];
        let b: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let s: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
        cluster_nms(&b, &s, cfg)
            .into_iter()
            .map(|e| ClusterNmsEntry { index: ids[e.index] as usize, bbox: e.bbox, score: e.score })
            .collect()
    };
    let mut kept: Vec<ClusterNmsEntry> = if n >= PAR_BUCKETS {
        ranges.par_iter().flat_map_iter(run_bucket).collect()
    } else {
        ranges.iter().flat_map(run_bucket).collect()
    };
    kept.sort_unstable_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.index.cmp(&b.index)));
    kept.truncate(cfg.max_det);
    kept
}

/// Apply Cluster-NMS directly to a slice of [`Detection`]s.
pub fn cluster_nms_detections(dets: &[Detection], cfg: &ClusterNmsConfig) -> Vec<Detection> {
    let boxes: Vec<Xyxy> = dets.iter().map(|d| d.bbox).collect();
    let scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
    let classes: Vec<u32> = dets.iter().map(|d| d.class).collect();
    batched_cluster_nms(&boxes, &scores, &classes, cfg)
        .into_iter()
        .map(|e| Detection { bbox: e.bbox, score: e.score, class: dets[e.index].class })
        .collect()
}
