//! Per-image object detection error diagnosis wired into [`crate::Evaluator`].
//!
//! While aggregate mAP summarises a dataset split, diagnosing *why* a model underperforms requires seeing the exact
//! images where it fails and whether its false positives come from poor box localization, class confusion,
//! duplicate predictions, or pure background hallucinations.
//!
//! [`diagnose_image`] partitions every evaluated prediction (`score >= conf_thresh`) and ground truth on one image
//! into exact, conservation-preserving counts ([`ImageDiagnostics`]):
//! - `num_truths == tp + fn_count`
//! - `num_preds == tp + fp`
//! - `fp == localization_errors + class_confusions + duplicate_errors + background_fp`
//!
//! [`worst_images`] ranks failing images (`fp + fn_count > 0`) worst-first so CLI, Studio, and Desktop can surface
//! failure cases immediately.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::metrics::Truth;
use crate::nms::Detection;

/// Upper bound of pycocotools' area range "all" (`1e5²`), matching [`crate::metrics::Evaluator`].
const COCO_MAX_AREA: f64 = 1e10;

/// Per-image detection error breakdown produced by [`crate::Evaluator`] when
/// [`crate::EvalConfig::record_diagnostics`] is enabled, or directly via [`diagnose_image`].
///
/// Conservation invariants:
/// - `num_truths == tp + fn_count`
/// - `num_preds == tp + fp` (counting finite-score predictions with `score >= conf_thresh`)
/// - `fp == localization_errors + class_confusions + duplicate_errors + background_fp`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageDiagnostics {
    /// Zero-based image index in the evaluator's insertion order.
    pub image_index: u64,
    /// Optional caller-supplied identifier (e.g. filename or dataset image ID).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    /// Ground-truth boxes evaluated in this image (excluding COCO-ignored area boxes when `coco_compatible` is on).
    pub num_truths: u32,
    /// Predictions evaluated in this image (`score.is_finite() && score >= conf_thresh`).
    pub num_preds: u32,
    /// True positives (same-class greedy match at `IoU >= tp_iou`).
    pub tp: u32,
    /// Total false positives (`localization_errors + class_confusions + duplicate_errors + background_fp`).
    pub fp: u32,
    /// False negatives (missed ground-truth boxes: `num_truths - tp`).
    pub fn_count: u32,
    /// False positives of the right class overlapping a same-class ground truth with `bg_iou <= IoU < tp_iou`.
    pub localization_errors: u32,
    /// False positives matching a different-class ground truth (`IoU >= tp_iou`, or `bg_iou <= IoU < tp_iou` when
    /// no same-class ground truth overlaps at `>= bg_iou`).
    pub class_confusions: u32,
    /// False positives overlapping an already-matched same-class ground truth at `IoU >= tp_iou`.
    pub duplicate_errors: u32,
    /// False positives overlapping no ground truth at `>= bg_iou` (pure background hallucinations).
    pub background_fp: u32,
    /// Per-image F1 score (`2 * tp / (2 * tp + fp + fn_count)`), or `1.0` when there are zero truths and zero FPs.
    pub f1: f32,
    /// Mean IoU across matched true positives in this image (`0.0` when `tp == 0`).
    pub mean_tp_iou: f32,
}

impl ImageDiagnostics {
    /// Total detection errors on this image (`fp + fn_count`).
    #[inline]
    pub fn total_errors(&self) -> u32 {
        self.fp.saturating_add(self.fn_count)
    }

    /// Per-image precision (`tp / (tp + fp)`), or `1.0` when `tp + fp == 0` and `fn_count == 0`, else `0.0`.
    #[inline]
    pub fn precision(&self) -> f32 {
        let denom = self.tp + self.fp;
        if denom > 0 {
            self.tp as f32 / denom as f32
        } else if self.fn_count == 0 {
            1.0
        } else {
            0.0
        }
    }

    /// Per-image recall (`tp / num_truths`), or `1.0` when `num_truths == 0 && fp == 0`, else `0.0`.
    #[inline]
    pub fn recall(&self) -> f32 {
        if self.num_truths > 0 {
            self.tp as f32 / self.num_truths as f32
        } else if self.fp == 0 {
            1.0
        } else {
            0.0
        }
    }
}

#[inline]
fn outside_coco_area(b: &Xyxy) -> bool {
    let a = (b.x2 as f64 - b.x1 as f64) * (b.y2 as f64 - b.y1 as f64);
    !(0.0..=COCO_MAX_AREA).contains(&a)
}

#[inline]
fn iou_f64(a: &Xyxy, b: &Xyxy) -> f64 {
    let (ax1, ay1, ax2, ay2) = (a.x1 as f64, a.y1 as f64, a.x2 as f64, a.y2 as f64);
    let (bx1, by1, bx2, by2) = (b.x1 as f64, b.y1 as f64, b.x2 as f64, b.y2 as f64);
    let w = ax2.min(bx2) - ax1.max(bx1);
    if w <= 0.0 {
        return 0.0;
    }
    let h = ay2.min(by2) - ay1.max(by1);
    if h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let union =
        (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0) + (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// Diagnose one image's predictions against its ground-truth boxes.
///
/// Predictions with finite `score >= conf_thresh` are evaluated in canonical descending-score order. Each prediction
/// either greedily matches the highest-IoU unmatched same-class ground truth at `IoU >= tp_iou` (a true positive)
/// or is categorized into exactly one false-positive error bucket:
/// 1. `duplicate_errors`: `best_same_iou >= tp_iou` (overlaps an already-matched same-class ground truth).
/// 2. `class_confusions`: `best_other_iou >= tp_iou` (tightly overlaps a different-class ground truth).
/// 3. `localization_errors`: `best_same_iou >= bg_iou` (same class, imprecise localization).
/// 4. `class_confusions`: `best_other_iou >= bg_iou` (different class, imprecise localization).
/// 5. `background_fp`: overlaps no ground truth at `>= bg_iou`.
#[allow(clippy::too_many_arguments)]
pub fn diagnose_image(
    image_index: u64,
    image_id: Option<String>,
    preds: &[Detection],
    truths: &[Truth],
    conf_thresh: f32,
    tp_iou: f32,
    bg_iou: f32,
    coco_compatible: bool,
) -> ImageDiagnostics {
    let tp_thr = if tp_iou.is_finite() { (tp_iou as f64).clamp(0.0, 1.0) } else { 0.5 };
    let bg_thr = if bg_iou.is_finite() { (bg_iou as f64).clamp(0.0, tp_thr) } else { 0.1f64.min(tp_thr) };
    let conf_thr = if conf_thresh.is_finite() { conf_thresh } else { 0.25 };

    let active_truths: Vec<&Truth> =
        truths.iter().filter(|t| !coco_compatible || !outside_coco_area(&t.bbox)).collect();

    let mut eval_preds: Vec<&Detection> =
        preds.iter().filter(|p| p.score.is_finite() && p.score >= conf_thr).collect();
    eval_preds.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.class.cmp(&b.class))
            .then_with(|| a.bbox.x1.total_cmp(&b.bbox.x1))
            .then_with(|| a.bbox.y1.total_cmp(&b.bbox.y1))
            .then_with(|| a.bbox.x2.total_cmp(&b.bbox.x2))
            .then_with(|| a.bbox.y2.total_cmp(&b.bbox.y2))
    });

    let mut gt_matched = vec![false; active_truths.len()];
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut localization_errors = 0u32;
    let mut class_confusions = 0u32;
    let mut duplicate_errors = 0u32;
    let mut background_fp = 0u32;
    let mut tp_iou_sum = 0.0f64;

    for p in &eval_preds {
        let mut best_unmatched_same = -1.0f64;
        let mut best_unmatched_idx = usize::MAX;
        let mut best_same = 0.0f64;
        let mut best_other = 0.0f64;

        for (gi, g) in active_truths.iter().enumerate() {
            let v = iou_f64(&p.bbox, &g.bbox);
            if g.class == p.class {
                if v > best_same {
                    best_same = v;
                }
                if !gt_matched[gi] && v >= tp_thr && v > best_unmatched_same {
                    best_unmatched_same = v;
                    best_unmatched_idx = gi;
                }
            } else if v > best_other {
                best_other = v;
            }
        }

        if best_unmatched_idx != usize::MAX {
            gt_matched[best_unmatched_idx] = true;
            tp += 1;
            tp_iou_sum += best_unmatched_same;
        } else {
            fp += 1;
            if best_same >= tp_thr {
                duplicate_errors += 1;
            } else if best_other >= tp_thr {
                class_confusions += 1;
            } else if best_same >= bg_thr {
                localization_errors += 1;
            } else if best_other >= bg_thr {
                class_confusions += 1;
            } else {
                background_fp += 1;
            }
        }
    }

    let num_truths = active_truths.len() as u32;
    let num_preds = eval_preds.len() as u32;
    let fn_count = num_truths.saturating_sub(tp);
    let denom = 2 * tp + fp + fn_count;
    let f1 = if denom > 0 { (2.0 * tp as f64 / denom as f64) as f32 } else { 1.0 };
    let mean_tp_iou = if tp > 0 { (tp_iou_sum / tp as f64) as f32 } else { 0.0 };

    ImageDiagnostics {
        image_index,
        image_id,
        num_truths,
        num_preds,
        tp,
        fp,
        fn_count,
        localization_errors,
        class_confusions,
        duplicate_errors,
        background_fp,
        f1,
        mean_tp_iou,
    }
}

/// Return up to `top_k` failing images (`fp + fn_count > 0`) ranked worst-first:
/// 1. Descending `total_errors()` (`fp + fn_count`)
/// 2. Ascending `f1`
/// 3. Descending `fn_count`
/// 4. Descending `fp`
/// 5. Ascending `mean_tp_iou`
/// 6. Ascending `image_index` (deterministic tie-breaker)
pub fn worst_images(diags: &[ImageDiagnostics], top_k: usize) -> Vec<&ImageDiagnostics> {
    if top_k == 0 {
        return Vec::new();
    }
    let mut failing: Vec<&ImageDiagnostics> = diags.iter().filter(|d| d.total_errors() > 0).collect();
    failing.sort_unstable_by(|a, b| {
        b.total_errors()
            .cmp(&a.total_errors())
            .then_with(|| a.f1.total_cmp(&b.f1))
            .then_with(|| b.fn_count.cmp(&a.fn_count))
            .then_with(|| b.fp.cmp(&a.fp))
            .then_with(|| a.mean_tp_iou.total_cmp(&b.mean_tp_iou))
            .then_with(|| a.image_index.cmp(&b.image_index))
    });
    failing.truncate(top_k);
    failing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
        Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
    }

    fn gt(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
        Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
    }

    #[test]
    fn diagnose_image_partitions_every_error_type_with_exact_conservation() {
        let truths = vec![
            gt(0.0, 0.0, 100.0, 100.0, 0),   // matched TP + duplicate FP
            gt(200.0, 0.0, 300.0, 100.0, 0), // localization error (IoU ~0.33) -> also FN
            gt(400.0, 0.0, 500.0, 100.0, 1), // class confusion (pred class 2 at IoU 1.0) -> also FN
            gt(600.0, 0.0, 700.0, 100.0, 2), // pure missed FN
        ];
        let preds = vec![
            det(0.0, 0.0, 100.0, 100.0, 0.95, 0),     // TP (IoU = 1.0)
            det(2.0, 2.0, 98.0, 98.0, 0.80, 0),       // duplicate_errors (same GT already matched, IoU > 0.5)
            det(250.0, 0.0, 350.0, 100.0, 0.70, 0), // localization_errors (same class, IoU = 1/3 in [0.1, 0.5))
            det(400.0, 0.0, 500.0, 100.0, 0.85, 2), // class_confusions (other class, IoU = 1.0)
            det(800.0, 800.0, 850.0, 850.0, 0.60, 1), // background_fp (no GT overlap)
            det(600.0, 0.0, 700.0, 100.0, 0.10, 2), // below conf_thresh (0.25) -> ignored
            det(0.0, 0.0, 10.0, 10.0, f32::NAN, 0), // NaN score -> ignored
        ];

        let d = diagnose_image(0, Some("frame_001.jpg".into()), &preds, &truths, 0.25, 0.5, 0.1, false);
        assert_eq!(d.image_index, 0);
        assert_eq!(d.image_id.as_deref(), Some("frame_001.jpg"));
        assert_eq!(d.num_truths, 4);
        assert_eq!(d.num_preds, 5);
        assert_eq!(d.tp, 1);
        assert_eq!(d.fp, 4);
        assert_eq!(d.fn_count, 3);
        assert_eq!(d.duplicate_errors, 1);
        assert_eq!(d.localization_errors, 1);
        assert_eq!(d.class_confusions, 1);
        assert_eq!(d.background_fp, 1);
        assert_eq!(d.tp + d.fn_count, d.num_truths);
        assert_eq!(d.tp + d.fp, d.num_preds);
        assert_eq!(d.localization_errors + d.class_confusions + d.duplicate_errors + d.background_fp, d.fp);
        assert!((d.mean_tp_iou - 1.0).abs() < 1e-6);
        assert!((d.f1 - (2.0 / 9.0)).abs() < 1e-6);
    }

    #[test]
    fn tight_other_class_box_touching_same_class_neighbor_is_class_confusion() {
        let truths = vec![
            gt(0.0, 0.0, 100.0, 100.0, 0), // same-class neighbor slightly touched (IoU ~ 0.14)
            gt(75.0, 0.0, 175.0, 100.0, 1), // wrong-class object tightly covered (IoU = 1.0)
        ];
        let preds = vec![det(75.0, 0.0, 175.0, 100.0, 0.9, 0)];
        let d = diagnose_image(0, None, &preds, &truths, 0.25, 0.5, 0.1, false);
        assert_eq!((d.tp, d.fp, d.class_confusions, d.localization_errors), (0, 1, 1, 0));
    }

    #[test]
    fn worst_images_ranks_failing_images_and_skips_perfect_images() {
        let perfect = diagnose_image(
            0,
            Some("perfect.jpg".into()),
            &[det(0.0, 0.0, 50.0, 50.0, 0.9, 0)],
            &[gt(0.0, 0.0, 50.0, 50.0, 0)],
            0.25,
            0.5,
            0.1,
            false,
        );
        let one_err_good_f1 = diagnose_image(
            1,
            Some("mostly_ok.jpg".into()),
            &[det(0.0, 0.0, 50.0, 50.0, 0.9, 0), det(100.0, 0.0, 150.0, 50.0, 0.9, 0)],
            &[gt(0.0, 0.0, 50.0, 50.0, 0), gt(100.0, 0.0, 150.0, 50.0, 0), gt(200.0, 0.0, 250.0, 50.0, 0)],
            0.25,
            0.5,
            0.1,
            false,
        );
        let one_err_zero_f1 = diagnose_image(
            2,
            Some("total_miss.jpg".into()),
            &[],
            &[gt(0.0, 0.0, 50.0, 50.0, 0)],
            0.25,
            0.5,
            0.1,
            false,
        );
        let three_errs = diagnose_image(
            3,
            Some("worst.jpg".into()),
            &[det(300.0, 0.0, 350.0, 50.0, 0.9, 0)],
            &[gt(0.0, 0.0, 50.0, 50.0, 0), gt(100.0, 0.0, 150.0, 50.0, 0)],
            0.25,
            0.5,
            0.1,
            false,
        );

        let all = vec![perfect, one_err_good_f1, one_err_zero_f1, three_errs];
        let ranked = worst_images(&all, 10);
        let ids: Vec<&str> = ranked.iter().map(|d| d.image_id.as_deref().unwrap()).collect();
        assert_eq!(ids, vec!["worst.jpg", "total_miss.jpg", "mostly_ok.jpg"]);
        assert_eq!(worst_images(&all, 2).len(), 2);
        assert!(worst_images(&all[..1], 5).is_empty());
        assert!(worst_images(&all, 0).is_empty());
    }
}
