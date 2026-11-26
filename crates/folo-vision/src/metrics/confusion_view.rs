//! Normalized precision/recall summary view for [`DetectionConfusionMatrix`].
//!
//! Builds directly on [`DetectionConfusionMatrix`] (`(C + 1) x (C + 1)` with background row/col at
//! index `C`) without duplicating matching logic, exposing:
//! - Row-normalized precision matrix (`precision_matrix[pred][truth]`): fraction of predictions for
//!   `pred` originating from `truth` (diagonal = per-class precision; `[c][bg]` = background FP share).
//! - Column-normalized recall matrix (`recall_matrix[pred][truth]`): fraction of ground truths of
//!   `truth` predicted as `pred` (diagonal = per-class recall; `[bg][c]` = background miss/FN share).
//! - Per-class TP / class-confusion FP / background FP / class-confusion FN / background FN breakdown.
//! - Macro (over classes with ground-truth support) and micro precision / recall / F1, plus ranked
//!   off-diagonal confusion pairs.

use serde::{Deserialize, Serialize};

use super::diagnostics::DetectionConfusionMatrix;

/// Per-class detection confusion breakdown separating class-confusion errors from background FP/FN.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassConfusionMetrics {
    /// Foreground class index.
    pub class: u32,
    /// True positives (`counts[c][c]`).
    pub tp: u64,
    /// Total false positives predicted as `class` (`class_fp + background_fp`).
    pub fp: u64,
    /// False positives matched to a ground truth of another foreground class.
    pub class_fp: u64,
    /// False positives matched to background (`counts[c][bg]`).
    pub background_fp: u64,
    /// Total false negatives for ground-truth `class` (`class_fn + background_fn`).
    pub fn_count: u64,
    /// Ground truths of `class` misclassified as another foreground class.
    pub class_fn: u64,
    /// Missed ground truths of `class` predicted as background (`counts[bg][c]`).
    pub background_fn: u64,
    /// Ground-truth support (`tp + fn_count`, saturating at `u64::MAX`).
    pub support: u64,
    /// Per-class precision `tp / (tp + fp)` (`0.0` when denominator is `0`).
    pub precision: f64,
    /// Per-class recall `tp / (tp + fn_count)` (`0.0` when denominator is `0`).
    pub recall: f64,
    /// Per-class F1 score `2 * P * R / (P + R)` (`0.0` when `P + R == 0`).
    pub f1: f64,
}

/// One non-zero off-diagonal confusion entry `(pred_class, truth_class)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfusionPair {
    /// Predicted class index (`num_classes` denotes missed / background prediction).
    pub pred_class: usize,
    /// Ground-truth class index (`num_classes` denotes background / spurious false positive).
    pub truth_class: usize,
    /// Raw error count `counts[pred_class][truth_class]`.
    pub count: u64,
    /// Fraction of all predictions in row `pred_class` (`precision_matrix[pred_class][truth_class]`).
    pub precision_share: f64,
    /// Fraction of all truths in column `truth_class` (`recall_matrix[pred_class][truth_class]`).
    pub recall_share: f64,
}

/// Normalized precision/recall view and per-class summary derived from a [`DetectionConfusionMatrix`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedConfusionView {
    /// Number of foreground classes `C`; matrices are `(C + 1) x (C + 1)` with background at index `C`.
    pub num_classes: usize,
    /// IoU threshold used during matching.
    pub iou_thresh: f32,
    /// Confidence threshold used during matching.
    pub conf_thresh: f32,
    /// Row-normalized `(C + 1) x (C + 1)` precision view (`precision_matrix[pred][truth]`).
    pub precision_matrix: Vec<Vec<f64>>,
    /// Column-normalized `(C + 1) x (C + 1)` recall view (`recall_matrix[pred][truth]`).
    pub recall_matrix: Vec<Vec<f64>>,
    /// Per-class error breakdown and P/R/F1 for each foreground class `0..C`.
    pub per_class: Vec<ClassConfusionMetrics>,
    /// Macro-averaged precision across foreground classes with `support > 0` (`0.0` if none).
    pub macro_precision: f64,
    /// Macro-averaged recall across foreground classes with `support > 0` (`0.0` if none).
    pub macro_recall: f64,
    /// Macro-averaged F1 across foreground classes with `support > 0` (`0.0` if none).
    pub macro_f1: f64,
    /// Micro-averaged precision `sum(TP) / total_predictions` (`0.0` if no predictions).
    pub micro_precision: f64,
    /// Micro-averaged recall `sum(TP) / total_truths` (`0.0` if no truths).
    pub micro_recall: f64,
    /// Micro-averaged F1 `2 * micro_P * micro_R / (micro_P + micro_R)` (`0.0` if both are `0`).
    pub micro_f1: f64,
    /// Non-zero off-diagonal confusion entries sorted by descending `count`, tie-broken by
    /// `(pred_class, truth_class)`.
    pub top_confusions: Vec<ConfusionPair>,
}

impl DetectionConfusionMatrix {
    /// Background false positives predicted as foreground `class` (`counts[class][background_index()]`).
    #[inline]
    pub fn background_fp(&self, class: usize) -> u64 {
        self.counts()[class][self.background_index()]
    }

    /// Background false negatives (missed ground truths) of foreground `class`
    /// (`counts[background_index()][class]`).
    #[inline]
    pub fn background_fn(&self, class: usize) -> u64 {
        self.counts()[self.background_index()][class]
    }

    /// Build the normalized precision/recall view and per-class diagnostic summary.
    pub fn normalized_view(&self) -> NormalizedConfusionView {
        let nc = self.num_classes();
        let bg = self.background_index();
        let counts = self.counts();
        let precision_matrix = self.row_normalized();
        let recall_matrix = self.col_normalized();

        let mut per_class = Vec::with_capacity(nc);
        let mut total_tp = 0u128;
        let mut total_preds = 0u128;
        let mut total_truths = 0u128;
        let mut active_classes = 0usize;
        let mut sum_p = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut sum_f1 = 0.0f64;

        for c in 0..nc {
            let tp = self.true_positives(c);
            let row_sum: u128 = counts[c].iter().map(|&v| v as u128).sum();
            let col_sum: u128 = counts.iter().map(|row| row[c] as u128).sum();
            let fp_u128 = row_sum - tp as u128;
            let fn_u128 = col_sum - tp as u128;
            let bg_fp = counts[c][bg];
            let bg_fn = counts[bg][c];
            let class_fp = (fp_u128 - bg_fp as u128).min(u64::MAX as u128) as u64;
            let class_fn = (fn_u128 - bg_fn as u128).min(u64::MAX as u128) as u64;
            let fp = fp_u128.min(u64::MAX as u128) as u64;
            let fn_count = fn_u128.min(u64::MAX as u128) as u64;
            let support = col_sum.min(u64::MAX as u128) as u64;
            let precision = self.precision(c);
            let recall = self.recall(c);
            let f1 = self.f1(c);

            total_tp += tp as u128;
            total_preds += row_sum;
            total_truths += col_sum;
            if col_sum > 0 {
                active_classes += 1;
                sum_p += precision;
                sum_r += recall;
                sum_f1 += f1;
            }

            per_class.push(ClassConfusionMetrics {
                class: c as u32,
                tp,
                fp,
                class_fp,
                background_fp: bg_fp,
                fn_count,
                class_fn,
                background_fn: bg_fn,
                support,
                precision,
                recall,
                f1,
            });
        }

        let (macro_precision, macro_recall, macro_f1) = if active_classes == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let inv = 1.0 / (active_classes as f64);
            (sum_p * inv, sum_r * inv, sum_f1 * inv)
        };

        let micro_precision = if total_preds == 0 { 0.0 } else { total_tp as f64 / total_preds as f64 };
        let micro_recall = if total_truths == 0 { 0.0 } else { total_tp as f64 / total_truths as f64 };
        let micro_f1 = if micro_precision + micro_recall == 0.0 {
            0.0
        } else {
            2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        };

        let dim = nc + 1;
        let mut top_confusions = Vec::new();
        for p in 0..dim {
            for t in 0..dim {
                if p == t {
                    continue;
                }
                let cnt = counts[p][t];
                if cnt > 0 {
                    top_confusions.push(ConfusionPair {
                        pred_class: p,
                        truth_class: t,
                        count: cnt,
                        precision_share: precision_matrix[p][t],
                        recall_share: recall_matrix[p][t],
                    });
                }
            }
        }
        top_confusions.sort_unstable_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| a.pred_class.cmp(&b.pred_class))
                .then_with(|| a.truth_class.cmp(&b.truth_class))
        });

        NormalizedConfusionView {
            num_classes: nc,
            iou_thresh: self.iou_thresh(),
            conf_thresh: self.conf_thresh(),
            precision_matrix,
            recall_matrix,
            per_class,
            macro_precision,
            macro_recall,
            macro_f1,
            micro_precision,
            micro_recall,
            micro_f1,
            top_confusions,
        }
    }
}
