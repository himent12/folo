//! Fast-NMS (Bolya et al., YOLACT, ICCV 2019) and Cluster-NMS (Zheng et al., IEEE TCYB 2021)
//! with optional DIoU overlap, score decay (Linear / Gaussian SPM), and weighted-cluster box
//! refinement.
//!
//! Both algorithms drop candidates with non-finite scores or non-finite box coordinates before
//! sorting descending by score (deterministic lower-input-index tie-breaking):
//!
//! - **Fast-NMS** ([`fast_nms`], [`batched_fast_nms`], [`fast_nms_detections`]): single-pass
//!   vectorized column-wise maximum `max_{i < j} X[i, j] <= iou_thr` in `O(m)` memory. Runs in
//!   one sweep with no sequential dependency; may over-suppress chained overlaps where `A`
//!   suppresses `B` and `B` suppresses `C`.
//! - **Cluster-NMS** ([`cluster_nms`], [`batched_cluster_nms`], [`cluster_nms_detections`]):
//!   iterative masked upper-triangular fixed-point `b^{(t)}[j] = !any_{i < j} (b^{(t-1)}[i] &&
//!   X[i, j] > iou_thr)` converging in typically 2–4 iterations. At convergence, hard Cluster-NMS
//!   is **provably identical** to sequential greedy NMS (and DIoU-NMS when `use_diou` is enabled);
//!   for uncapped hard suppression (`max_iters == None`, no decay, no weighted box), executes the
//!   equivalent `O(m)`-memory greedy fast path stopping early at `max_det`, while capped/optional
//!   modes use compact bit-packed upper-triangular adjacency (`1 bit / pair`, checked against
//!   overflow) and compute plain IoU on demand for surviving cluster centers (`O(m)` float memory).
//!   Optional extensions:
//!   - [`ClusterScoreDecay::Linear`] / [`ClusterScoreDecay::Gaussian`]: decays overlapping boxes'
//!     scores against surviving cluster centers instead of hard-dropping.
//!   - `weighted_box`: refines each kept cluster center as the `score * IoU`-weighted average of
//!     boxes it suppressed (convex combination in `f64`).

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
    /// `prod_{i < j, kept[i], adj[i,j]} (1 - IoU(i, j))` and retain scores `> score_thr`.
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
    /// `w_j = score_j * IoU(b_k, b_j)` over cluster members suppressed by `k`).
    pub weighted_box: bool,
    /// Optional cap on fixed-point iterations (`None` = exact convergence; `Some(1)` recovers
    /// single-pass Fast-NMS).
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

fn validate_cluster_cfg(cfg: &ClusterNmsConfig, op: &str) {
    if let ClusterScoreDecay::Gaussian { sigma } = cfg.score_decay {
        assert!(sigma.is_finite() && sigma > 0.0, "{op}: Gaussian sigma must be finite and > 0, got {sigma}");
    }
    if let Some(iters) = cfg.max_iters {
        assert!(iters > 0, "{op}: max_iters must be >= 1");
    }
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

#[inline]
fn overlap_f64(a: &Xyxy, b: &Xyxy, use_diou: bool) -> f64 {
    if use_diou { exact_diou_f64(a, b) } else { exact_iou_f64(a, b) }
}

/// Filter finite scores `> score_thr` and finite box coordinates, then sort descending by score
/// (ties: smaller input index).
fn sorted_valid_indices(boxes: &[Xyxy], scores: &[f32], score_thr: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len())
        .filter(|&i| scores[i].is_finite() && scores[i] > score_thr && box_is_finite(&boxes[i]))
        .collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    order
}

/// Compact bit-packed strict upper-triangular adjacency matrix (`1 bit` per `(i, j)` with `i < j`).
///
/// Row `i` stores `m - 1 - i` bits for `j in (i + 1)..m`, total pairs `m * (m - 1) / 2` (checked
/// against integer overflow on both 32-bit wasm32 and 64-bit targets).
struct UpperTriBits {
    row_offsets: Vec<usize>,
    words: Vec<u64>,
}

impl UpperTriBits {
    fn new(m: usize) -> Self {
        let mut row_offsets = Vec::with_capacity(m);
        let mut total_bits = 0usize;
        for i in 0..m {
            row_offsets.push(total_bits);
            total_bits = total_bits
                .checked_add(m - 1 - i)
                .expect("cluster_nms: upper-triangular adjacency size overflow");
        }
        let n_words = total_bits.div_ceil(64);
        Self { row_offsets, words: vec![0u64; n_words] }
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize) {
        debug_assert!(i < j);
        let bit = self.row_offsets[i] + (j - i - 1);
        self.words[bit >> 6] |= 1u64 << (bit & 63);
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> bool {
        debug_assert!(i < j);
        let bit = self.row_offsets[i] + (j - i - 1);
        ((self.words[bit >> 6] >> (bit & 63)) & 1) != 0
    }
}

/// Fast-NMS (Bolya et al., YOLACT 2019) over a single class's `boxes` and `scores`.
///
/// Drops non-finite scores (`NaN`, `±Inf`), non-finite box coordinates, and scores `<= score_thr`,
/// sorts descending by score (ties: lower input index), computes the upper-triangular column
/// maximum `max_{i < j} IoU(b_i, b_j)` in one `O(m)`-memory sweep, and keeps entries with
/// `max_iou <= iou_thr` (rank 0 always survives any finite `iou_thr`) up to `limit`.
///
/// # Panics
/// If `boxes.len() != scores.len()`.
pub fn fast_nms(boxes: &[Xyxy], scores: &[f32], iou_thr: f32, score_thr: f32, limit: usize) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "fast_nms: boxes/scores length mismatch");
    if boxes.is_empty() || limit == 0 || iou_thr.is_nan() {
        return Vec::new();
    }
    let order = sorted_valid_indices(boxes, scores, score_thr);
    let m = order.len();
    if m == 0 {
        return Vec::new();
    }
    let sorted_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let thr = iou_thr as f64;
    let mut max_iou = vec![f64::NEG_INFINITY; m];
    for i in 0..m {
        let bi = &sorted_boxes[i];
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
/// Drops candidates with non-finite scores, non-finite box coordinates, or scores `<= cfg.score_thr`,
/// sorts descending by score (ties: lower input index), solves the Cluster-NMS fixed point (using an
/// exact `O(m)`-memory fast path for uncapped hard Cluster-NMS, and a compact bit-packed upper
/// triangle for capped iterations / score-decay / weighted-box refinement), and returns at most
/// `cfg.max_det` [`ClusterNmsEntry`] items.
///
/// # Panics
/// If `boxes.len() != scores.len()`, `cfg.score_decay` has non-finite or `<= 0` Gaussian `sigma`,
/// or `cfg.max_iters == Some(0)`.
pub fn cluster_nms(boxes: &[Xyxy], scores: &[f32], cfg: &ClusterNmsConfig) -> Vec<ClusterNmsEntry> {
    assert_eq!(boxes.len(), scores.len(), "cluster_nms: boxes/scores length mismatch");
    validate_cluster_cfg(cfg, "cluster_nms");
    if boxes.is_empty() || cfg.max_det == 0 || cfg.iou_thr.is_nan() {
        return Vec::new();
    }
    let order = sorted_valid_indices(boxes, scores, cfg.score_thr);
    let m = order.len();
    if m == 0 {
        return Vec::new();
    }
    let sorted_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let sorted_scores: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    let thr = cfg.iou_thr as f64;

    // Fast path for exact uncapped hard Cluster-NMS (Zheng et al., Theorem 1): provably identical
    // to sequential greedy suppression in `O(m)` memory, stopping as soon as `max_det` survive.
    if cfg.max_iters.is_none() && matches!(cfg.score_decay, ClusterScoreDecay::None) && !cfg.weighted_box {
        let mut suppressed = vec![false; m];
        let mut entries = Vec::with_capacity(m.min(cfg.max_det));
        for i in 0..m {
            if suppressed[i] {
                continue;
            }
            entries.push(ClusterNmsEntry { index: order[i], bbox: sorted_boxes[i], score: sorted_scores[i] });
            if entries.len() == cfg.max_det {
                break;
            }
            let bi = &sorted_boxes[i];
            for j in (i + 1)..m {
                if !suppressed[j] && overlap_f64(bi, &sorted_boxes[j], cfg.use_diou) > thr {
                    suppressed[j] = true;
                }
            }
        }
        return entries;
    }

    // Compact bit-packed strict upper-triangular adjacency (`1 bit / pair`).
    let mut adj = UpperTriBits::new(m);
    for (i, bi) in sorted_boxes.iter().enumerate() {
        for (j, bj) in sorted_boxes.iter().enumerate().skip(i + 1) {
            if overlap_f64(bi, bj, cfg.use_diou) > thr {
                adj.set(i, j);
            }
        }
    }

    // Iterative masked upper-triangular fixed-point:
    // b_next[j] = !any_{i < j} (b[i] && adj[i, j]).
    let max_iters = cfg.max_iters.unwrap_or(m).min(m);
    let mut alive = vec![true; m];
    let mut next_alive = vec![true; m];
    for _ in 0..max_iters {
        next_alive.fill(true);
        for (i, &is_alive) in alive.iter().enumerate() {
            if !is_alive {
                continue;
            }
            for (j, slot) in next_alive.iter_mut().enumerate().skip(i + 1) {
                if adj.get(i, j) {
                    *slot = false;
                }
            }
        }
        if next_alive == alive {
            break;
        }
        std::mem::swap(&mut alive, &mut next_alive);
    }

    // Optional weighted-cluster box refinement for surviving centers `k`:
    // assign each suppressed box `j` to its primary surviving suppressor `k < j` (gated on
    // `adj.get(k, j)` so DIoU suppression owns only boxes it actually suppressed, ranking eligible
    // suppressors by highest plain IoU with ties broken toward the highest-scoring `k`).
    let refined_boxes: Option<Vec<Xyxy>> = if cfg.weighted_box {
        let mut owner = vec![usize::MAX; m];
        for j in 0..m {
            if alive[j] {
                owner[j] = j;
                continue;
            }
            let mut best_k = usize::MAX;
            let mut best_ov = -1.0f64;
            for (k, &is_alive) in alive.iter().enumerate().take(j) {
                if !is_alive || !adj.get(k, j) {
                    continue;
                }
                let ov = exact_iou_f64(&sorted_boxes[k], &sorted_boxes[j]);
                if ov > best_ov {
                    best_ov = ov;
                    best_k = k;
                }
            }
            owner[j] = best_k;
        }
        let mut out_boxes = sorted_boxes.clone();
        for k in 0..m {
            if !alive[k] || sorted_scores[k] <= 0.0 {
                continue;
            }
            let mut w_sum = 0.0f64;
            let mut sx1 = 0.0f64;
            let mut sy1 = 0.0f64;
            let mut sx2 = 0.0f64;
            let mut sy2 = 0.0f64;
            for j in k..m {
                if owner[j] != k || sorted_scores[j] <= 0.0 {
                    continue;
                }
                let ov = if j == k { 1.0 } else { exact_iou_f64(&sorted_boxes[k], &sorted_boxes[j]) };
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
                    for (i, &is_alive) in alive.iter().enumerate().take(j) {
                        if is_alive && adj.get(i, j) {
                            let iou = exact_iou_f64(&sorted_boxes[i], &sorted_boxes[j]);
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
                            let iou = exact_iou_f64(&sorted_boxes[i], &sorted_boxes[j]);
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
    validate_cluster_cfg(cfg, "batched_cluster_nms");
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
