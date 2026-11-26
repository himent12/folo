//! Standalone detection evaluation diagnostics operating directly on `(preds, truths)` image streams:
//!
//! - [`DetectionConfusionMatrix`]: `(C + 1) × (C + 1)` confusion matrix with a background row/column,
//!   configurable `iou_thresh` and `conf_thresh`, deterministic greedy IoU matching, per-class
//!   TP/FP/FN/precision/recall/F1, and row/column normalization.
//! - [`ClassPrCurve`] & [`OptimalF1Point`] ([`compute_pr_curves`]): exact detection-level per-class
//!   precision-recall curves with a monotonic precision envelope, exact continuous AP (piecewise-constant
//!   area under the envelope), 101-point COCO AP, and the exact global-optimum F1 confidence threshold
//!   evaluated over every unique score threshold.
//! - [`DetectionCalibrationDiagnostics`] & [`ReliabilityBin`] ([`compute_detection_calibration`],
//!   [`reliability_from_scores`]): Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
//!   Brier score, and per-bin reliability summaries directly on IoU-matched detections (pooled and
//!   per-class) or pre-matched `(confidence, is_tp)` arrays.

use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{NUM_RECALL_POINTS, Truth, iou_f64, recall_points};
use crate::nms::Detection;

#[inline]
fn box_is_finite(b: &Xyxy) -> bool {
    b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()
}

/// Wire format for validated [`DetectionConfusionMatrix`] deserialization.
#[derive(Deserialize)]
struct RawDetectionConfusionMatrix {
    num_classes: usize,
    conf_thresh: f32,
    iou_thresh: f32,
    counts: Vec<Vec<u64>>,
}

impl TryFrom<RawDetectionConfusionMatrix> for DetectionConfusionMatrix {
    type Error = String;

    fn try_from(raw: RawDetectionConfusionMatrix) -> std::result::Result<Self, Self::Error> {
        if !(raw.iou_thresh.is_finite() && (0.0..=1.0).contains(&raw.iou_thresh)) {
            return Err(format!(
                "DetectionConfusionMatrix: iou_thresh must be finite and in [0, 1], got {}",
                raw.iou_thresh
            ));
        }
        if !(raw.conf_thresh.is_finite() && (0.0..=1.0).contains(&raw.conf_thresh)) {
            return Err(format!(
                "DetectionConfusionMatrix: conf_thresh must be finite and in [0, 1], got {}",
                raw.conf_thresh
            ));
        }
        let dim = raw
            .num_classes
            .checked_add(1)
            .ok_or_else(|| "DetectionConfusionMatrix: num_classes + 1 overflows".to_string())?;
        if raw.counts.len() != dim || raw.counts.iter().any(|row| row.len() != dim) {
            return Err(format!("DetectionConfusionMatrix: counts must be a {dim} x {dim} square matrix"));
        }
        Ok(Self {
            num_classes: raw.num_classes,
            conf_thresh: raw.conf_thresh,
            iou_thresh: raw.iou_thresh,
            counts: raw.counts,
        })
    }
}

/// Detection confusion matrix with a background row and column at index `num_classes`.
///
/// Layout: `counts[pred_class][truth_class]`, where:
/// - `counts[c][c]` (`c < num_classes`) = true positives for class `c`,
/// - `counts[p][t]` (`p != t`, both `< num_classes`) = truth of class `t` predicted as class `p`,
/// - `counts[p][num_classes]` = false positive background detection predicted as class `p`,
/// - `counts[num_classes][t]` = missed ground truth of class `t` (predicted as background),
/// - `counts[num_classes][num_classes]` is always `0`.
///
/// Per image, predictions with `score >= conf_thresh` (ignoring non-finite scores or boxes) are greedily
/// matched one-to-one to ground truths with `IoU >= iou_thresh`, selecting pairs in descending IoU order
/// with deterministic tie-breaking by descending prediction score, ascending truth index, then ascending
/// prediction index.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawDetectionConfusionMatrix")]
pub struct DetectionConfusionMatrix {
    /// Number of foreground object classes `C`; matrix dimension is `(C + 1) × (C + 1)`.
    pub num_classes: usize,
    /// Minimum prediction confidence (`score >= conf_thresh`).
    pub conf_thresh: f32,
    /// Minimum IoU for a prediction-truth match (`iou >= iou_thresh`).
    pub iou_thresh: f32,
    /// Row-major `(C + 1) × (C + 1)` count table indexed as `counts[pred][truth]`.
    counts: Vec<Vec<u64>>,
}

impl DetectionConfusionMatrix {
    /// Create an empty confusion matrix with validated thresholds (`conf_thresh` in `[0, 1]`,
    /// `iou_thresh` in `[0, 1]`).
    pub fn new(num_classes: usize, iou_thresh: f32, conf_thresh: f32) -> Result<Self> {
        if !(iou_thresh.is_finite() && (0.0..=1.0).contains(&iou_thresh)) {
            return Err(Error::Config(format!(
                "DetectionConfusionMatrix: iou_thresh must be finite and in [0, 1], got {iou_thresh}"
            )));
        }
        if !(conf_thresh.is_finite() && (0.0..=1.0).contains(&conf_thresh)) {
            return Err(Error::Config(format!(
                "DetectionConfusionMatrix: conf_thresh must be finite and in [0, 1], got {conf_thresh}"
            )));
        }
        let dim = num_classes
            .checked_add(1)
            .ok_or_else(|| Error::Config("DetectionConfusionMatrix: num_classes + 1 overflows".into()))?;
        Ok(Self { num_classes, conf_thresh, iou_thresh, counts: vec![vec![0u64; dim]; dim] })
    }

    /// Read-only view of the `(num_classes + 1) × (num_classes + 1)` count matrix `counts[pred][truth]`.
    #[inline]
    pub fn counts(&self) -> &[Vec<u64>] {
        &self.counts
    }

    /// Build a confusion matrix directly from a slice of `(preds, truths)` image pairs.
    pub fn from_images<P, T>(
        num_classes: usize,
        iou_thresh: f32,
        conf_thresh: f32,
        images: &[(P, T)],
    ) -> Result<Self>
    where
        P: AsRef<[Detection]>,
        T: AsRef<[Truth]>,
    {
        let mut cm = Self::new(num_classes, iou_thresh, conf_thresh)?;
        for (preds, truths) in images {
            cm.add_image(preds.as_ref(), truths.as_ref())?;
        }
        Ok(cm)
    }

    /// Background row/column index (`== num_classes`).
    #[inline]
    pub fn background_index(&self) -> usize {
        self.num_classes
    }

    /// Accumulate one image's predictions and ground truths.
    ///
    /// # Errors
    /// Returns [`Error::Config`] if any prediction or truth class id is `>= num_classes`.
    pub fn add_image(&mut self, preds: &[Detection], truths: &[Truth]) -> Result<()> {
        let nc = self.num_classes;
        for (i, p) in preds.iter().enumerate() {
            if (p.class as usize) >= nc {
                return Err(Error::Config(format!(
                    "DetectionConfusionMatrix: pred[{i}].class = {} >= num_classes ({nc})",
                    p.class
                )));
            }
        }
        for (i, t) in truths.iter().enumerate() {
            if (t.class as usize) >= nc {
                return Err(Error::Config(format!(
                    "DetectionConfusionMatrix: truth[{i}].class = {} >= num_classes ({nc})",
                    t.class
                )));
            }
        }

        let conf = self.conf_thresh;
        let kept_preds: Vec<usize> = preds
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                (p.score.is_finite() && p.score >= conf && box_is_finite(&p.bbox)).then_some(i)
            })
            .collect();

        if truths.is_empty() {
            for pi in kept_preds {
                self.counts[preds[pi].class as usize][nc] += 1;
            }
            return Ok(());
        }
        if kept_preds.is_empty() {
            for t in truths {
                self.counts[nc][t.class as usize] += 1;
            }
            return Ok(());
        }

        let iou_thr = self.iou_thresh as f64;
        // Candidate matches: `(iou, pred_score, truth_idx, pred_idx)`.
        let mut pairs: Vec<(f64, f32, usize, usize)> = Vec::new();
        for &pi in &kept_preds {
            let p = &preds[pi];
            for (gi, g) in truths.iter().enumerate() {
                if !box_is_finite(&g.bbox) {
                    continue;
                }
                let v = iou_f64(&p.bbox, &g.bbox);
                if v >= iou_thr {
                    pairs.push((v, p.score, gi, pi));
                }
            }
        }
        // Highest IoU first; ties broken by higher pred score, smaller truth index, smaller pred index.
        pairs.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });

        let mut gt_used = vec![false; truths.len()];
        let mut pred_used = vec![false; preds.len()];
        for &(_, _, gi, pi) in &pairs {
            if gt_used[gi] || pred_used[pi] {
                continue;
            }
            gt_used[gi] = true;
            pred_used[pi] = true;
            self.counts[preds[pi].class as usize][truths[gi].class as usize] += 1;
        }
        for (gi, g) in truths.iter().enumerate() {
            if !gt_used[gi] {
                self.counts[nc][g.class as usize] += 1;
            }
        }
        for &pi in &kept_preds {
            if !pred_used[pi] {
                self.counts[preds[pi].class as usize][nc] += 1;
            }
        }
        Ok(())
    }

    /// Merge another confusion matrix with identical dimensions and thresholds into `self`.
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        if self.num_classes != other.num_classes
            || self.iou_thresh != other.iou_thresh
            || self.conf_thresh != other.conf_thresh
        {
            return Err(Error::Config(
                "DetectionConfusionMatrix::merge: class count or threshold mismatch".into(),
            ));
        }
        for (r_dst, r_src) in self.counts.iter_mut().zip(&other.counts) {
            for (d, &s) in r_dst.iter_mut().zip(r_src) {
                *d += s;
            }
        }
        Ok(())
    }

    /// Count at `(pred_class, truth_class)`. Use `background_index()` for background.
    #[inline]
    pub fn get(&self, pred: usize, truth: usize) -> u64 {
        self.counts[pred][truth]
    }

    /// True positives for class `c` (`counts[c][c]`).
    #[inline]
    pub fn true_positives(&self, class: usize) -> u64 {
        self.counts[class][class]
    }

    /// False positives predicted as class `c` (sum of row `c` excluding `counts[c][c]`).
    pub fn false_positives(&self, class: usize) -> u64 {
        self.counts[class].iter().sum::<u64>() - self.counts[class][class]
    }

    /// False negatives for ground-truth class `c` (sum of column `c` excluding `counts[c][c]`).
    pub fn false_negatives(&self, class: usize) -> u64 {
        self.counts.iter().map(|row| row[class]).sum::<u64>() - self.counts[class][class]
    }

    /// Per-class precision `TP / (TP + FP)` (`0.0` when `TP + FP == 0`).
    pub fn precision(&self, class: usize) -> f64 {
        let tp = self.true_positives(class);
        let denom = tp + self.false_positives(class);
        if denom == 0 { 0.0 } else { tp as f64 / denom as f64 }
    }

    /// Per-class recall `TP / (TP + FN)` (`0.0` when `TP + FN == 0`).
    pub fn recall(&self, class: usize) -> f64 {
        let tp = self.true_positives(class);
        let denom = tp + self.false_negatives(class);
        if denom == 0 { 0.0 } else { tp as f64 / denom as f64 }
    }

    /// Per-class F1 score `2 * P * R / (P + R)` (`0.0` when `P + R == 0`).
    pub fn f1(&self, class: usize) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    /// Total number of ground-truth instances accumulated across all object classes.
    pub fn total_truths(&self) -> u64 {
        (0..self.num_classes).map(|t| self.counts.iter().map(|row| row[t]).sum::<u64>()).sum()
    }

    /// Total number of above-threshold predictions accumulated across all object classes.
    pub fn total_predictions(&self) -> u64 {
        (0..self.num_classes).map(|p| self.counts[p].iter().sum::<u64>()).sum()
    }

    /// Row-normalized confusion matrix (each row sums to `1.0`, or all `0.0` if the row sum is `0`).
    pub fn row_normalized(&self) -> Vec<Vec<f64>> {
        self.counts
            .iter()
            .map(|row| {
                let s = row.iter().sum::<u64>();
                if s == 0 { vec![0.0; row.len()] } else { row.iter().map(|&v| v as f64 / s as f64).collect() }
            })
            .collect()
    }

    /// Column-normalized confusion matrix (each column sums to `1.0`, or all `0.0` if the column sum is `0`).
    pub fn col_normalized(&self) -> Vec<Vec<f64>> {
        let dim = self.num_classes + 1;
        let col_sums: Vec<u64> = (0..dim).map(|c| self.counts.iter().map(|row| row[c]).sum()).collect();
        self.counts
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(c, &v)| if col_sums[c] == 0 { 0.0 } else { v as f64 / col_sums[c] as f64 })
                    .collect()
            })
            .collect()
    }
}

/// One point on a detection-level precision-recall curve (in descending score order).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrCurvePoint {
    /// Detection confidence score at this operating point.
    pub confidence: f32,
    /// Cumulative recall `TP / num_truths` in `[0, 1]`.
    pub recall: f64,
    /// Raw precision `TP / (TP + FP)` in `[0, 1]`.
    pub precision: f64,
    /// Monotonically non-increasing precision envelope `max_{j >= i} precision[j]` in `[0, 1]`.
    pub precision_envelope: f64,
    /// F1 score `2 * precision * recall / (precision + recall)` in `[0, 1]`.
    pub f1: f64,
}

/// Optimal F1 operating point obtained by thresholding `score >= conf_thresh`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimalF1Point {
    /// Confidence threshold maximizing F1 (highest score on ties; `0.0` when no positive F1 exists).
    pub conf_thresh: f32,
    /// Maximum achievable F1 score in `[0, 1]`.
    pub f1: f64,
    /// Precision at `conf_thresh`.
    pub precision: f64,
    /// Recall at `conf_thresh`.
    pub recall: f64,
    /// True positives with `score >= conf_thresh`.
    pub tp: u64,
    /// False positives with `score >= conf_thresh`.
    pub fp: u64,
    /// Missed ground truths (`num_truths - tp`).
    pub fn_count: u64,
}

/// Per-class precision-recall curve, AP metrics, and exact optimal F1 threshold at one IoU threshold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassPrCurve {
    /// Class index.
    pub class: u32,
    /// Number of ground-truth boxes for this class.
    pub num_truths: u64,
    /// Number of finite predictions for this class.
    pub num_preds: u64,
    /// Exact piecewise-constant area under the monotonic precision envelope (all-point / VOC-2010+ AP).
    pub ap_continuous: f64,
    /// 101-point COCO-interpolated AP (`0.00:0.01:1.00`).
    pub ap_101: f64,
    /// Exact global-optimum F1 operating point across all unique detection score thresholds.
    pub best_f1: OptimalF1Point,
    /// Detection-level PR curve points in descending score order (non-decreasing recall, non-increasing
    /// `precision_envelope`).
    pub points: Vec<PrCurvePoint>,
}

/// Collection of per-class precision-recall curves and aggregate mean AP at one IoU threshold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrCurveDiagnostics {
    /// IoU threshold used for matching.
    pub iou_thresh: f32,
    /// Mean `ap_continuous` across classes with `num_truths > 0` (`0.0` if none).
    pub mean_ap_continuous: f64,
    /// Mean `ap_101` across classes with `num_truths > 0` (`0.0` if none).
    pub mean_ap_101: f64,
    /// One [`ClassPrCurve`] per class `0..num_classes`.
    pub per_class: Vec<ClassPrCurve>,
}

/// Matched detection record `(score, image_idx, det_idx, is_tp)`.
#[derive(Clone, Copy, Debug)]
struct MatchedDet {
    score: f32,
    image_idx: usize,
    det_idx: usize,
    is_tp: bool,
}

/// Per-image, per-class greedy IoU matching (descending score order, ties by input order) returning
/// per-class matched detections and ground-truth counts.
///
/// When `validate_unit_scores` is `true`, every finite detection score (even below `conf_thresh`) is
/// validated to lie in `[0, 1]` before applying the confidence filter.
fn match_images_by_class<P, T>(
    num_classes: usize,
    iou_thresh: f32,
    conf_thresh: Option<f32>,
    validate_unit_scores: bool,
    images: &[(P, T)],
) -> Result<(Vec<Vec<MatchedDet>>, Vec<u64>)>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    if !(iou_thresh.is_finite() && (0.0..=1.0).contains(&iou_thresh)) {
        return Err(Error::Config(format!(
            "diagnostics: iou_thresh must be finite and in [0, 1], got {iou_thresh}"
        )));
    }
    if let Some(ct) = conf_thresh
        && !(ct.is_finite() && (0.0..=1.0).contains(&ct))
    {
        return Err(Error::Config(format!(
            "diagnostics: conf_thresh must be finite and in [0, 1], got {ct}"
        )));
    }

    let mut per_class_dets = vec![Vec::<MatchedDet>::new(); num_classes];
    let mut gt_counts = vec![0u64; num_classes];
    let iou_thr = iou_thresh as f64;

    for (img_idx, (preds_ref, truths_ref)) in images.iter().enumerate() {
        let preds = preds_ref.as_ref();
        let truths = truths_ref.as_ref();
        for (i, p) in preds.iter().enumerate() {
            if (p.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "diagnostics: image {img_idx} pred[{i}].class = {} >= num_classes ({num_classes})",
                    p.class
                )));
            }
            if validate_unit_scores && (!p.score.is_finite() || !(0.0..=1.0).contains(&p.score)) {
                return Err(Error::Config(format!(
                    "diagnostics: image {img_idx} pred[{i}].score must be finite and in [0, 1], got {}",
                    p.score
                )));
            }
        }
        for (i, t) in truths.iter().enumerate() {
            if (t.class as usize) >= num_classes {
                return Err(Error::Config(format!(
                    "diagnostics: image {img_idx} truth[{i}].class = {} >= num_classes ({num_classes})",
                    t.class
                )));
            }
        }

        let mut img_preds: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        for (pi, p) in preds.iter().enumerate() {
            let passes_conf = conf_thresh.is_none_or(|ct| p.score >= ct);
            if p.score.is_finite() && passes_conf && box_is_finite(&p.bbox) {
                img_preds[p.class as usize].push(pi);
            }
        }
        let mut img_gts: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        for (gi, g) in truths.iter().enumerate() {
            gt_counts[g.class as usize] += 1;
            if box_is_finite(&g.bbox) {
                img_gts[g.class as usize].push(gi);
            }
        }

        for c in 0..num_classes {
            let p_list = &mut img_preds[c];
            if p_list.is_empty() {
                continue;
            }
            p_list
                .sort_unstable_by(|&a, &b| preds[b].score.total_cmp(&preds[a].score).then_with(|| a.cmp(&b)));
            let g_list = &img_gts[c];
            let mut gt_matched = vec![false; g_list.len()];
            let out = &mut per_class_dets[c];
            out.reserve(p_list.len());

            for &pi in p_list.iter() {
                let p = &preds[pi];
                let mut best_iou = -1.0f64;
                let mut best_g = None;
                for (local_g, &gi) in g_list.iter().enumerate() {
                    if gt_matched[local_g] {
                        continue;
                    }
                    let v = iou_f64(&p.bbox, &truths[gi].bbox);
                    if v >= iou_thr && v > best_iou {
                        best_iou = v;
                        best_g = Some(local_g);
                    }
                }
                let is_tp = if let Some(lg) = best_g {
                    gt_matched[lg] = true;
                    true
                } else {
                    false
                };
                out.push(MatchedDet { score: p.score, image_idx: img_idx, det_idx: pi, is_tp });
            }
        }
    }

    Ok((per_class_dets, gt_counts))
}

/// Compute exact per-class precision-recall curves (with monotonic precision envelope), continuous &
/// 101-point AP, and the exact global-optimum F1 confidence threshold at `iou_thresh`.
pub fn compute_pr_curves<P, T>(
    num_classes: usize,
    iou_thresh: f32,
    images: &[(P, T)],
) -> Result<PrCurveDiagnostics>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    let (mut per_class_dets, gt_counts) =
        match_images_by_class(num_classes, iou_thresh, None, false, images)?;
    let r_grid = recall_points();

    let mut per_class = Vec::with_capacity(num_classes);
    let mut sum_cont = 0.0f64;
    let mut sum_101 = 0.0f64;
    let mut active_classes = 0usize;

    for c in 0..num_classes {
        let dets = &mut per_class_dets[c];
        let n_gt = gt_counts[c];
        dets.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.image_idx.cmp(&b.image_idx))
                .then_with(|| a.det_idx.cmp(&b.det_idx))
        });

        let n_p = dets.len();
        if n_gt == 0 || n_p == 0 {
            let points: Vec<PrCurvePoint> = dets
                .iter()
                .map(|d| PrCurvePoint {
                    confidence: d.score,
                    recall: 0.0,
                    precision: 0.0,
                    precision_envelope: 0.0,
                    f1: 0.0,
                })
                .collect();
            let fp_at_zero = dets.iter().filter(|d| d.score >= 0.0).count() as u64;
            per_class.push(ClassPrCurve {
                class: c as u32,
                num_truths: n_gt,
                num_preds: n_p as u64,
                ap_continuous: 0.0,
                ap_101: 0.0,
                best_f1: OptimalF1Point {
                    conf_thresh: 0.0,
                    f1: 0.0,
                    precision: 0.0,
                    recall: 0.0,
                    tp: 0,
                    fp: fp_at_zero,
                    fn_count: n_gt,
                },
                points,
            });
            if n_gt > 0 {
                active_classes += 1;
            }
            continue;
        }

        let n_gt_f = n_gt as f64;
        let mut points = Vec::with_capacity(n_p);
        let mut tp = 0u64;
        let mut fp = 0u64;
        for d in dets.iter() {
            if d.is_tp {
                tp += 1;
            } else {
                fp += 1;
            }
            let prec = tp as f64 / (tp + fp) as f64;
            let rec = tp as f64 / n_gt_f;
            let f1 = if tp > 0 { (2.0 * tp as f64) / (tp + fp + n_gt) as f64 } else { 0.0 };
            points.push(PrCurvePoint {
                confidence: d.score,
                recall: rec,
                precision: prec,
                precision_envelope: prec,
                f1,
            });
        }

        // Backward suffix maximum builds the exact monotonically non-increasing precision envelope.
        let mut running_max = 0.0f64;
        for pt in points.iter_mut().rev() {
            if pt.precision > running_max {
                running_max = pt.precision;
            }
            pt.precision_envelope = running_max;
        }

        // Exact piecewise-constant area under the monotonic envelope: sum_i (r_i - r_{i-1}) * env_i.
        let mut ap_continuous = 0.0f64;
        let mut prev_r = 0.0f64;
        for pt in &points {
            let dr = pt.recall - prev_r;
            if dr > 0.0 {
                ap_continuous += dr * pt.precision_envelope;
                prev_r = pt.recall;
            }
        }

        // 101-point COCO interpolated AP: sample the envelope at r_grid[0..101] via left-sided search.
        let mut ap_101_sum = 0.0f64;
        let mut j = 0usize;
        for &rg in &r_grid {
            while j < n_p && points[j].recall < rg {
                j += 1;
            }
            if j < n_p {
                ap_101_sum += points[j].precision_envelope;
            }
        }
        let ap_101 = ap_101_sum / NUM_RECALL_POINTS as f64;

        // Exact optimal F1 over all unique score thresholds `score >= c` (evaluated at the end of each
        // equal-score run so tied detections are always thresholded together). Compare F1 via exact u128
        // cross-products of `tp / (tp + fp + n_gt)` so ties are broken strictly in favor of higher
        // `conf_thresh` without floating-point rounding drift.
        let fp_at_zero = dets.iter().filter(|d| !d.is_tp && d.score >= 0.0).count() as u64;
        let mut best_f1 = OptimalF1Point {
            conf_thresh: 0.0,
            f1: 0.0,
            precision: 0.0,
            recall: 0.0,
            tp: 0,
            fp: fp_at_zero,
            fn_count: n_gt,
        };
        let mut cur_tp = 0u64;
        let mut cur_fp = 0u64;
        for (idx, d) in dets.iter().enumerate() {
            if d.is_tp {
                cur_tp += 1;
            } else {
                cur_fp += 1;
            }
            let is_run_end = idx + 1 == n_p || dets[idx + 1].score != d.score;
            if is_run_end && cur_tp > 0 {
                let cur_num = cur_tp as u128;
                let cur_den = (cur_tp + cur_fp + n_gt) as u128;
                let best_num = best_f1.tp as u128;
                let best_den = (best_f1.tp + best_f1.fp + n_gt) as u128;
                if best_f1.tp == 0 || cur_num * best_den > best_num * cur_den {
                    let prec = cur_tp as f64 / (cur_tp + cur_fp) as f64;
                    let rec = cur_tp as f64 / n_gt_f;
                    let f1 = (2.0 * cur_tp as f64) / (cur_tp + cur_fp + n_gt) as f64;
                    best_f1 = OptimalF1Point {
                        conf_thresh: d.score,
                        f1,
                        precision: prec,
                        recall: rec,
                        tp: cur_tp,
                        fp: cur_fp,
                        fn_count: n_gt - cur_tp,
                    };
                }
            }
        }

        sum_cont += ap_continuous;
        sum_101 += ap_101;
        active_classes += 1;
        per_class.push(ClassPrCurve {
            class: c as u32,
            num_truths: n_gt,
            num_preds: n_p as u64,
            ap_continuous,
            ap_101,
            best_f1,
            points,
        });
    }

    let (mean_ap_continuous, mean_ap_101) = if active_classes == 0 {
        (0.0, 0.0)
    } else {
        (sum_cont / active_classes as f64, sum_101 / active_classes as f64)
    };

    Ok(PrCurveDiagnostics { iou_thresh, mean_ap_continuous, mean_ap_101, per_class })
}

/// One equal-width reliability diagram bin over `[lower, upper)` (the final bin is closed `[lower, 1.0]`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    /// Lower bin edge in `[0, 1]`.
    pub lower: f64,
    /// Upper bin edge in `[0, 1]`.
    pub upper: f64,
    /// Total detections in this bin (`tp + fp`).
    pub count: usize,
    /// True positives in this bin.
    pub tp: usize,
    /// False positives in this bin.
    pub fp: usize,
    /// Mean predicted confidence in this bin (`0.0` when empty).
    pub avg_confidence: f64,
    /// Empirical accuracy `tp / count` in this bin (`0.0` when empty).
    pub avg_accuracy: f64,
    /// Absolute calibration error `|avg_confidence - avg_accuracy|` (`0.0` when empty).
    pub gap: f64,
}

/// Calibration reliability summary over a set of scored detection outcomes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSummary {
    /// Expected Calibration Error: `sum_b (count_b / N) * |avg_accuracy_b - avg_confidence_b|`.
    pub ece: f64,
    /// Maximum Calibration Error: `max_{b: count_b > 0} |avg_accuracy_b - avg_confidence_b|`.
    pub mce: f64,
    /// Brier score: `(1 / N) * sum_i (p_i - y_i)²`.
    pub brier: f64,
    /// Total number of evaluated detections `N`.
    pub count: usize,
    /// Equal-width reliability bins over `[0, 1]`.
    pub bins: Vec<ReliabilityBin>,
}

/// Pooled and per-class detection calibration diagnostics evaluated directly from `(preds, truths)` images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionCalibrationDiagnostics {
    /// IoU threshold used to determine TP vs FP.
    pub iou_thresh: f32,
    /// Confidence floor applied before binning (`score >= conf_thresh`).
    pub conf_thresh: f32,
    /// Pooled calibration summary across all classes.
    pub overall: CalibrationSummary,
    /// Independent calibration summary for each class `0..num_classes`.
    pub per_class: Vec<CalibrationSummary>,
}

/// Compute ECE, MCE, Brier score, and per-bin reliability summaries from pre-matched confidence scores
/// (`scores[i]` in `[0, 1]`) and TP flags (`is_tp[i]`).
///
/// # Errors
/// Returns [`Error::Config`] if `num_bins == 0`, lengths differ, or any score is non-finite or outside `[0, 1]`.
pub fn reliability_from_scores(
    scores: &[f32],
    is_tp: &[bool],
    num_bins: usize,
) -> Result<CalibrationSummary> {
    if num_bins == 0 {
        return Err(Error::Config("reliability_from_scores: num_bins must be >= 1".into()));
    }
    if scores.len() != is_tp.len() {
        return Err(Error::Config(format!(
            "reliability_from_scores: scores length ({}) != is_tp length ({})",
            scores.len(),
            is_tp.len()
        )));
    }
    for (i, &s) in scores.iter().enumerate() {
        if !(s.is_finite() && (0.0..=1.0).contains(&s)) {
            return Err(Error::Config(format!(
                "reliability_from_scores: scores[{i}] must be finite and in [0, 1], got {s}"
            )));
        }
    }

    let m_f = num_bins as f64;
    let edges: Vec<f64> = (0..=num_bins).map(|b| if b == num_bins { 1.0 } else { b as f64 / m_f }).collect();

    let mut tp_counts = vec![0usize; num_bins];
    let mut fp_counts = vec![0usize; num_bins];
    let mut sum_conf = vec![0.0f64; num_bins];
    let mut brier_sum = 0.0f64;

    for (&sf, &tp) in scores.iter().zip(is_tp) {
        let p = sf as f64;
        let mut bin = ((p * m_f).floor() as usize).min(num_bins - 1);
        while bin + 1 < num_bins && p >= edges[bin + 1] {
            bin += 1;
        }
        while bin > 0 && p < edges[bin] {
            bin -= 1;
        }
        if tp {
            tp_counts[bin] += 1;
        } else {
            fp_counts[bin] += 1;
        }
        sum_conf[bin] += p;
        let d = p - if tp { 1.0 } else { 0.0 };
        brier_sum += d * d;
    }

    let n = scores.len();
    let mut bins = Vec::with_capacity(num_bins);
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;

    for b in 0..num_bins {
        let lower = edges[b];
        let upper = edges[b + 1];
        let tp = tp_counts[b];
        let fp = fp_counts[b];
        let cnt = tp + fp;
        if cnt == 0 {
            bins.push(ReliabilityBin {
                lower,
                upper,
                count: 0,
                tp: 0,
                fp: 0,
                avg_confidence: 0.0,
                avg_accuracy: 0.0,
                gap: 0.0,
            });
        } else {
            let avg_conf = sum_conf[b] / cnt as f64;
            let avg_acc = tp as f64 / cnt as f64;
            let gap = (avg_conf - avg_acc).abs();
            ece += (cnt as f64 / n as f64) * gap;
            if gap > mce {
                mce = gap;
            }
            bins.push(ReliabilityBin {
                lower,
                upper,
                count: cnt,
                tp,
                fp,
                avg_confidence: avg_conf,
                avg_accuracy: avg_acc,
                gap,
            });
        }
    }

    let brier = if n == 0 { 0.0 } else { brier_sum / n as f64 };
    Ok(CalibrationSummary { ece, mce, brier, count: n, bins })
}

/// Compute pooled and per-class detection calibration reliability diagnostics (ECE, MCE, Brier score, and
/// equal-width reliability bins) directly from `(preds, truths)` images at `iou_thresh` and `conf_thresh`.
///
/// # Errors
/// Returns [`Error::Config`] on invalid thresholds, `num_bins == 0`, out-of-range class ids, or detection
/// scores outside `[0, 1]`.
pub fn compute_detection_calibration<P, T>(
    num_classes: usize,
    iou_thresh: f32,
    conf_thresh: f32,
    num_bins: usize,
    images: &[(P, T)],
) -> Result<DetectionCalibrationDiagnostics>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    if num_bins == 0 {
        return Err(Error::Config("compute_detection_calibration: num_bins must be >= 1".into()));
    }
    let (per_class_dets, _) =
        match_images_by_class(num_classes, iou_thresh, Some(conf_thresh), true, images)?;

    let mut all_scores = Vec::new();
    let mut all_tp = Vec::new();
    let mut per_class = Vec::with_capacity(num_classes);

    for dets in &per_class_dets {
        let scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
        let flags: Vec<bool> = dets.iter().map(|d| d.is_tp).collect();
        all_scores.extend_from_slice(&scores);
        all_tp.extend_from_slice(&flags);
        per_class.push(reliability_from_scores(&scores, &flags, num_bins)?);
    }

    let overall = reliability_from_scores(&all_scores, &all_tp, num_bins)?;
    Ok(DetectionCalibrationDiagnostics { iou_thresh, conf_thresh, overall, per_class })
}
