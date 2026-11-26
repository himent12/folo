//! Matrix-NMS (Wang et al. 2020, SOLOv2): non-sequential parallel score decay.
//!
//! For candidates sorted by descending score `s_0 >= s_1 >= ... >= s_{m-1}` with upper-triangular pairwise
//! IoU `M_{i,j} = IoU(b_i, b_j)` (`i < j`), let `M_max(i) = max_{k < i} M_{k, i}` be candidate `i`'s maximum
//! overlap with any higher-scoring box (`M_max(0) = 0`). The decay factor applied to candidate `j` is:
//!
//! ```text
//! Linear kernel:    decay_j = min(1, min_{i < j} (1 - M_{i,j}) / (1 - M_max(i)))
//! Gaussian kernel:  decay_j = min(1, min_{i < j} exp(-(M_{i,j}² - M_max(i)²) / sigma))
//! s'_j = s_j * decay_j    (for s_j > 0)
//! ```
//!
//! Because `M_max(i)` depends only on `k < i`, a single upper-triangular sweep computes exact Matrix-NMS in
//! `O(m)` memory instead of allocating an `m x m` matrix, with zero division-by-zero when `M_max(i) == 1`
//! (since `M_{i,j} <= M_max(i)` implies `decay >= 1` and cannot reduce `min_decay[j] <= 1`).

use crate::boxes::{IouKind, Xyxy, iou};
use rayon::prelude::*;

use super::kernel::bucket_by_class;

/// Total candidate count above which class buckets run in parallel.
const PAR_BUCKETS: usize = 2048;

#[inline]
fn safe_iou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
    if !(a.x1.is_finite()
        && a.y1.is_finite()
        && a.x2.is_finite()
        && a.y2.is_finite()
        && b.x1.is_finite()
        && b.y1.is_finite()
        && b.x2.is_finite()
        && b.y2.is_finite())
    {
        return 0.0;
    }
    (iou(a, b, IouKind::Iou) as f64).clamp(0.0, 1.0)
}

/// Matrix-NMS (Wang et al. 2020) over `boxes` and `scores` (not required to be pre-sorted).
///
/// - `sigma == None`: Linear Matrix-NMS (`decay = min_{i < j} (1 - M_{i,j}) / (1 - M_max(i))`).
/// - `sigma == Some(s)`: Gaussian Matrix-NMS (`decay = min_{i < j} exp(-(M_{i,j}² - M_max(i)²) / s)`).
///
/// Non-positive scores are never multiplied (so decay can never increase a score). Returns
/// `(input_index, decayed_score)` pairs sorted by descending decayed score (ties: lower input index),
/// keeping only boxes whose final score is `> score_thr` and stopping at `limit`.
///
/// # Panics
/// If `boxes.len() != scores.len()`, or `sigma` is `Some(s)` with `!s.is_finite() || s <= 0.0`.
pub fn matrix_nms_sorted(
    boxes: &[Xyxy],
    scores: &[f32],
    score_thr: f32,
    sigma: Option<f32>,
    limit: usize,
) -> Vec<(usize, f32)> {
    assert_eq!(boxes.len(), scores.len(), "matrix_nms_sorted: boxes/scores length mismatch");
    if let Some(s) = sigma {
        assert!(s.is_finite() && s > 0.0, "matrix_nms_sorted: sigma must be finite and > 0, got {s}");
    }
    let n = boxes.len();
    if n == 0 || limit == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..n).filter(|&i| scores[i] > score_thr).collect();
    if order.is_empty() {
        return Vec::new();
    }
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));

    let m = order.len();
    let sorted_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let mut max_iou_above = vec![0.0f64; m];
    let mut min_decay = vec![1.0f64; m];
    let sigma64 = sigma.map(|s| s as f64);

    for i in 0..m {
        let m_i = max_iou_above[i];
        let bi = &sorted_boxes[i];
        let m_i_sq = m_i * m_i;
        let inv_one_minus_mi = if sigma64.is_none() && m_i < 1.0 { 1.0 / (1.0 - m_i) } else { 0.0 };
        for j in (i + 1)..m {
            let ov = safe_iou_f64(bi, &sorted_boxes[j]);
            if ov > max_iou_above[j] {
                max_iou_above[j] = ov;
            }
            if ov > m_i {
                let d = match sigma64 {
                    Some(s) => (-(ov * ov - m_i_sq) / s).exp(),
                    None => (1.0 - ov) * inv_one_minus_mi,
                };
                if d < min_decay[j] {
                    min_decay[j] = d;
                }
            }
        }
    }

    let mut kept: Vec<(usize, f32)> = Vec::with_capacity(m.min(limit));
    for (pos, &orig_idx) in order.iter().enumerate() {
        let s = scores[orig_idx];
        let decayed = if s > 0.0 { (s as f64 * min_decay[pos]) as f32 } else { s };
        if decayed > score_thr {
            kept.push((orig_idx, decayed));
        }
    }
    kept.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    kept.truncate(limit);
    kept
}

/// Class-aware Matrix-NMS used by [`crate::nms::batched_nms_with_kind`] and [`crate::nms::finish`].
#[allow(clippy::too_many_arguments)]
pub(super) fn batched_matrix_nms(
    boxes: &[Xyxy],
    scores: &[f32],
    classes: &[u32],
    score_thr: f32,
    sigma: Option<f32>,
    max_det: usize,
    agnostic: bool,
) -> Vec<(usize, f32)> {
    assert_eq!(boxes.len(), scores.len(), "batched_matrix_nms: boxes/scores length mismatch");
    assert_eq!(boxes.len(), classes.len(), "batched_matrix_nms: boxes/classes length mismatch");
    if let Some(s) = sigma {
        assert!(s.is_finite() && s > 0.0, "batched_matrix_nms: sigma must be finite and > 0, got {s}");
    }
    let n = boxes.len();
    if n == 0 || max_det == 0 {
        return Vec::new();
    }
    assert!(n <= u32::MAX as usize, "batched_matrix_nms: too many boxes");
    if agnostic {
        return matrix_nms_sorted(boxes, scores, score_thr, sigma, max_det);
    }
    let order: Vec<u32> = (0..n as u32).collect();
    let (bucketed, ranges) = bucket_by_class(&order, classes);
    let run_bucket = |&(lo, hi): &(usize, usize)| -> Vec<(usize, f32)> {
        let ids = &bucketed[lo..hi];
        let b: Vec<Xyxy> = ids.iter().map(|&i| boxes[i as usize]).collect();
        let s: Vec<f32> = ids.iter().map(|&i| scores[i as usize]).collect();
        matrix_nms_sorted(&b, &s, score_thr, sigma, max_det)
            .into_iter()
            .map(|(k, sc)| (ids[k] as usize, sc))
            .collect()
    };
    let mut kept: Vec<(usize, f32)> = if n >= PAR_BUCKETS {
        ranges.par_iter().flat_map_iter(run_bucket).collect()
    } else {
        ranges.iter().flat_map(run_bucket).collect()
    };
    kept.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    kept.truncate(max_det);
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nms::{Detection, NmsConfig, NmsKind, batched_nms_with_kind, nms_detections};

    #[test]
    fn matrix_nms_matches_exact_formula_and_prevents_false_suppression() {
        // b0 (0.9) and b1 (0.8) overlap strongly (IoU = 1/3 -> let's make IoU = 0.8: [0,0,10,10] & [1,0,11,10] -> inter 90, union 110, IoU = 9/11).
        // b2 (0.7) overlaps b1 moderately (IoU = 4/16 = 0.25) and is disjoint from b0 (IoU = 0).
        // Because b1 already has M_max(1) = 9/11 > M_{1,2} = 0.25, Matrix-NMS does NOT suppress b2!
        let b0 = Xyxy::new(0.0, 0.0, 10.0, 10.0);
        let b1 = Xyxy::new(1.0, 0.0, 11.0, 10.0);
        let b2 = Xyxy::new(7.0, 0.0, 17.0, 10.0);
        let boxes = [b0, b1, b2];
        let scores = [0.9f32, 0.8, 0.7];

        let iou01 = iou(&b0, &b1, IouKind::Iou) as f64;
        let iou02 = iou(&b0, &b2, IouKind::Iou) as f64;
        let iou12 = iou(&b1, &b2, IouKind::Iou) as f64;
        assert!(iou01 > iou12 && iou02 > 0.0 && iou12 > 0.0);

        // Linear Matrix-NMS:
        // j=0: decay=1 -> 0.9
        // j=1: M_max(0)=0, M_{0,1}=iou01 -> decay = 1 - iou01
        // j=2: from i=0: decay = 1 - iou02; from i=1: M_{1,2} < M_max(1)=iou01 so decay = 1.0
        let lin = matrix_nms_sorted(&boxes, &scores, 0.0, None, 10);
        let want1 = (0.8 * (1.0 - iou01)) as f32;
        let want2 = (0.7 * (1.0 - iou02)) as f32;
        assert_eq!(lin.len(), 3);
        assert_eq!(lin[0].0, 0);
        assert!((lin[0].1 - 0.9).abs() < 1e-6);
        assert_eq!(lin[1].0, 2, "b2 should outrank heavily suppressed b1");
        assert!((lin[1].1 - want2).abs() < 1e-6);
        assert_eq!(lin[2].0, 1);
        assert!((lin[2].1 - want1).abs() < 1e-6);

        // Gaussian Matrix-NMS with sigma = 0.5:
        let sigma = 0.5f32;
        let gau = matrix_nms_sorted(&boxes, &scores, 0.0, Some(sigma), 10);
        let g1 = (0.8 * (-(iou01 * iou01) / 0.5).exp()) as f32;
        let g2 = (0.7 * (-(iou02 * iou02) / 0.5).exp()) as f32;
        let map: std::collections::HashMap<usize, f32> = gau.into_iter().collect();
        assert!((map[&0] - 0.9).abs() < 1e-6);
        assert!((map[&1] - g1).abs() < 1e-6);
        assert!((map[&2] - g2).abs() < 1e-6);
    }

    #[test]
    fn matrix_nms_exact_duplicates_and_degenerate_inputs_are_stable() {
        let b = Xyxy::new(10.0, 10.0, 30.0, 30.0);
        // Three identical boxes: M_max(1) == 1.0 must not divide by zero!
        // Box 0 keeps full score; box 1 and box 2 both get IoU=1.0 against box 0 -> linear decay 0.0.
        let lin = matrix_nms_sorted(&[b, b, b], &[0.9, 0.8, 0.7], 0.0, None, 10);
        assert_eq!(lin, vec![(0, 0.9)]);

        // Gaussian decay on identical boxes: decay = exp(-1/0.5) = exp(-2).
        let gau = matrix_nms_sorted(&[b, b, b], &[0.9, 0.8, 0.7], 0.0, Some(0.5), 10);
        assert_eq!(gau.len(), 3);
        let d = (-2.0f64).exp();
        assert!((gau[0].1 - 0.9).abs() < 1e-6);
        assert!((gau[1].1 - (0.8 * d) as f32).abs() < 1e-6);
        assert!((gau[2].1 - (0.7 * d) as f32).abs() < 1e-6);

        // NaN scores, non-finite boxes, inverted/zero-area boxes never produce NaN.
        let messy = [
            Xyxy::new(0.0, 0.0, 10.0, 10.0),
            Xyxy::new(f32::NAN, 0.0, 10.0, 10.0),
            Xyxy::new(5.0, 5.0, 5.0, 5.0),
            Xyxy::new(20.0, 20.0, 10.0, 10.0),
        ];
        let out = matrix_nms_sorted(&messy, &[0.9, 0.8, f32::NAN, 0.6], 0.0, None, 10);
        assert!(out.iter().all(|(_, s)| s.is_finite() && *s > 0.0));
        assert_eq!(out.len(), 3);

        // Class-aware vs agnostic via batched_nms_with_kind & nms_detections + serde round-trip.
        let dets =
            vec![Detection { bbox: b, score: 0.9, class: 0 }, Detection { bbox: b, score: 0.8, class: 1 }];
        let cfg = NmsConfig { conf: 0.1, nms_kind: NmsKind::MatrixLinear, ..NmsConfig::default() };
        assert_eq!(nms_detections(&dets, &cfg).len(), 2, "class-aware keeps both classes");
        let cfg_ag = NmsConfig { agnostic: true, ..cfg.clone() };
        assert_eq!(nms_detections(&dets, &cfg_ag).len(), 1, "agnostic suppresses identical cross-class box");

        let kinds = [NmsKind::MatrixLinear, NmsKind::MatrixGaussian { sigma: 0.35 }];
        for kind in kinds {
            let c = NmsConfig { nms_kind: kind, ..NmsConfig::default() };
            let j = serde_json::to_string(&c).unwrap();
            assert_eq!(serde_json::from_str::<NmsConfig>(&j).unwrap(), c);
            let res = batched_nms_with_kind(&[b, b], &[0.9, 0.8], &[0, 0], 0.5, 0.0, 10, false, kind);
            assert!(!res.is_empty() && res.iter().all(|(_, s)| s.is_finite()));
        }
    }
}
