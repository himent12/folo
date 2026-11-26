//! Rotated Non-Maximum Suppression (`rotated_nms`, `batched_rotated_nms`, `rotated_nms_detections`)
//! using exact Sutherland–Hodgman OBB polygon IoU with AABB pre-filtering.

use std::collections::BTreeMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{Obb, obb_iou_f64};

/// Candidate count above which per-class rotated NMS buckets execute in parallel.
const PAR_BUCKETS: usize = 512;

/// One oriented bounding-box detection: rotated box, confidence score, and class index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObbDetection {
    /// Oriented bounding box `(cx, cy, w, h, theta_rad)`.
    #[serde(flatten)]
    pub obb: Obb,
    /// Confidence score.
    pub score: f32,
    /// Class index.
    pub class: u32,
}

/// Configuration for [`batched_rotated_nms`] and [`rotated_nms_detections`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RotatedNmsConfig {
    /// Minimum confidence threshold (`score > conf`, default `0.25`).
    pub conf: f32,
    /// Suppression overlap threshold (`obb_iou > iou`, default `0.7`).
    pub iou: f32,
    /// Maximum detections returned per image (default `300`).
    pub max_det: usize,
    /// Class-agnostic suppression when `true` (default `false`: strict per-class isolation).
    pub agnostic: bool,
}

impl Default for RotatedNmsConfig {
    fn default() -> Self {
        Self { conf: 0.25, iou: 0.7, max_det: 300, agnostic: false }
    }
}

#[inline]
fn center_extents_may_overlap(a: &[f64; 4], b: &[f64; 4]) -> bool {
    (a[0] - b[0]).abs() <= a[2] + b[2] && (a[1] - b[1]).abs() <= a[3] + b[3]
}

/// Greedy single-set rotated NMS over pre-filtered candidate indices already sorted by descending
/// score (tie-break ascending input index).
fn rotated_nms_sorted_indices(boxes: &[Obb], order: &[usize], iou_thr: f64, max_det: usize) -> Vec<usize> {
    let m = order.len();
    if m == 0 || max_det == 0 {
        return Vec::new();
    }
    let ce: Vec<[f64; 4]> = order.iter().map(|&idx| boxes[idx].aabb_center_extent_f64()).collect();
    let allow_aabb_skip = iou_thr >= 0.0;
    let mut suppressed = vec![false; m];
    let mut kept = Vec::with_capacity(m.min(max_det));

    for i in 0..m {
        if suppressed[i] {
            continue;
        }
        let bi_idx = order[i];
        kept.push(bi_idx);
        if kept.len() >= max_det {
            break;
        }
        let bi = &boxes[bi_idx];
        let ce_i = &ce[i];
        for j in (i + 1)..m {
            if suppressed[j] {
                continue;
            }
            if allow_aabb_skip && !center_extents_may_overlap(ce_i, &ce[j]) {
                continue;
            }
            if obb_iou_f64(bi, &boxes[order[j]]) > iou_thr {
                suppressed[j] = true;
            }
        }
    }
    kept
}

/// Class-agnostic greedy rotated NMS returning indices into `boxes` sorted by descending score
/// (ties broken by ascending input index).
///
/// Candidates with non-finite scores, non-finite box parameters, or `score <= score_thr` are dropped.
/// A lower-ranked box is suppressed when its exact polygon IoU with an already kept box is strictly
/// greater than `iou_thr`.
///
/// # Panics
/// If `boxes.len() != scores.len()`.
pub fn rotated_nms(
    boxes: &[Obb],
    scores: &[f32],
    iou_thr: f32,
    score_thr: f32,
    max_det: usize,
) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "rotated_nms: boxes and scores length mismatch");
    if boxes.is_empty() || max_det == 0 || iou_thr.is_nan() || score_thr.is_nan() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..boxes.len())
        .filter(|&i| scores[i].is_finite() && scores[i] > score_thr && boxes[i].is_finite())
        .collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    rotated_nms_sorted_indices(boxes, &order, iou_thr as f64, max_det)
}

/// Class-aware (or class-agnostic when `cfg.agnostic`) rotated NMS returning kept indices sorted by
/// descending score (ties broken by ascending input index), capped at `cfg.max_det`.
///
/// # Panics
/// If `boxes.len() != scores.len()` or `boxes.len() != classes.len()`.
pub fn batched_rotated_nms(
    boxes: &[Obb],
    scores: &[f32],
    classes: &[u32],
    cfg: &RotatedNmsConfig,
) -> Vec<usize> {
    assert_eq!(boxes.len(), scores.len(), "batched_rotated_nms: boxes and scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_rotated_nms: boxes and classes length mismatch");
    if boxes.is_empty() || cfg.max_det == 0 || cfg.iou.is_nan() || cfg.conf.is_nan() {
        return Vec::new();
    }
    if cfg.agnostic {
        return rotated_nms(boxes, scores, cfg.iou, cfg.conf, cfg.max_det);
    }

    let mut order: Vec<usize> = (0..boxes.len())
        .filter(|&i| scores[i].is_finite() && scores[i] > cfg.conf && boxes[i].is_finite())
        .collect();
    if order.is_empty() {
        return Vec::new();
    }
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));

    // Preserve descending score order inside every class bucket.
    let mut buckets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &idx in &order {
        buckets.entry(classes[idx]).or_default().push(idx);
    }

    let iou_thr = cfg.iou as f64;
    let max_det = cfg.max_det;
    let bucket_vecs: Vec<Vec<usize>> = buckets.into_values().collect();
    let mut kept: Vec<usize> = if order.len() >= PAR_BUCKETS && bucket_vecs.len() > 1 {
        bucket_vecs.par_iter().flat_map(|b| rotated_nms_sorted_indices(boxes, b, iou_thr, max_det)).collect()
    } else {
        bucket_vecs.iter().flat_map(|b| rotated_nms_sorted_indices(boxes, b, iou_thr, max_det)).collect()
    };

    kept.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    kept.truncate(max_det);
    kept
}

/// Apply rotated NMS to a slice of [`ObbDetection`]s, returning kept detections in descending score order.
pub fn rotated_nms_detections(dets: &[ObbDetection], cfg: &RotatedNmsConfig) -> Vec<ObbDetection> {
    let boxes: Vec<Obb> = dets.iter().map(|d| d.obb).collect();
    let scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
    let classes: Vec<u32> = dets.iter().map(|d| d.class).collect();
    batched_rotated_nms(&boxes, &scores, &classes, cfg).into_iter().map(|i| dets[i]).collect()
}
