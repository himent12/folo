//! Evaluation confidence-threshold sweeps (P / R / F1 vs confidence with exact optimal thresholds) and
//! extended reliability-bin calibration diagnostics (equal-width ECE/MCE, equal-mass adaptive ACE/MCE,
//! signed confidence bias, overconfidence error, and class-wise Static Calibration Error `SCE`).

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::metrics::Truth;
use crate::metrics::diagnostics::{OptimalF1Point, ReliabilityBin, match_images_by_class};
use crate::nms::Detection;

/// Configuration for [`compute_confidence_sweep`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfidenceSweepConfig {
    /// IoU threshold in `[0, 1]` used to match predictions to ground truths (default `0.5`).
    pub iou_thresh: f32,
    /// Number of linearly spaced confidence grid points in `[min_conf, max_conf]` (`>= 2`, default `101`).
    pub num_thresholds: usize,
    /// Lower end of the confidence sweep grid in `[0, 1]` (default `0.0`).
    pub min_conf: f32,
    /// Upper end of the confidence sweep grid in `[0, 1]` (`>= min_conf`, default `1.0`).
    pub max_conf: f32,
}

impl Default for ConfidenceSweepConfig {
    fn default() -> Self {
        Self { iou_thresh: 0.5, num_thresholds: 101, min_conf: 0.0, max_conf: 1.0 }
    }
}

impl ConfidenceSweepConfig {
    /// Validate sweep parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.iou_thresh.is_finite() && (0.0..=1.0).contains(&self.iou_thresh)) {
            return Err(Error::Config(format!(
                "ConfidenceSweepConfig: iou_thresh must be finite and in [0, 1], got {}",
                self.iou_thresh
            )));
        }
        if self.num_thresholds < 2 {
            return Err(Error::Config(format!(
                "ConfidenceSweepConfig: num_thresholds must be >= 2, got {}",
                self.num_thresholds
            )));
        }
        if !(self.min_conf.is_finite() && (0.0..=1.0).contains(&self.min_conf)) {
            return Err(Error::Config(format!(
                "ConfidenceSweepConfig: min_conf must be finite and in [0, 1], got {}",
                self.min_conf
            )));
        }
        if !(self.max_conf.is_finite()
            && (0.0..=1.0).contains(&self.max_conf)
            && self.max_conf >= self.min_conf)
        {
            return Err(Error::Config(format!(
                "ConfidenceSweepConfig: max_conf ({}) must be finite, in [0, 1], and >= min_conf ({})",
                self.max_conf, self.min_conf
            )));
        }
        Ok(())
    }
}

/// Precision, recall, F1, and detection counts at a single confidence threshold `score >= conf_thresh`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceSweepPoint {
    /// Confidence threshold `c` (detections with `score >= c` are active).
    pub conf_thresh: f32,
    /// Precision `TP / (TP + FP)` (or macro-averaged precision across active classes; `0.0` when undefined).
    pub precision: f64,
    /// Recall `TP / (TP + FN)` (or macro-averaged recall across active classes; `0.0` when undefined).
    pub recall: f64,
    /// F1 score `2 * P * R / (P + R)` (or macro-averaged F1 across active classes; `0.0` when undefined).
    pub f1: f64,
    /// True positives at this threshold.
    pub tp: u64,
    /// False positives at this threshold.
    pub fp: u64,
    /// Missed ground truths at this threshold.
    pub fn_count: u64,
}

/// Per-class confidence-threshold sweep curve and exact optimal F1 threshold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassConfidenceSweep {
    /// Class index.
    pub class: u32,
    /// Number of ground-truth boxes for this class.
    pub num_truths: u64,
    /// Number of finite predictions for this class.
    pub num_preds: u64,
    /// Exact global-optimum F1 operating point over all unique detection scores of this class.
    pub best_f1: OptimalF1Point,
    /// Operating points evaluated at every threshold of [`ConfidenceSweepDiagnostics::thresholds`].
    pub points: Vec<ConfidenceSweepPoint>,
}

/// Multi-class Precision / Recall / F1-Confidence sweep diagnostics at one IoU threshold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceSweepDiagnostics {
    /// IoU threshold used for matching.
    pub iou_thresh: f32,
    /// Ascending confidence threshold grid `[min_conf, ..., max_conf]`.
    pub thresholds: Vec<f32>,
    /// Micro-averaged (pooled across all classes) P/R/F1 curve across `thresholds`.
    pub micro_curve: Vec<ConfidenceSweepPoint>,
    /// Macro-averaged (mean over classes with `num_truths > 0`) P/R/F1 curve across `thresholds`.
    pub macro_curve: Vec<ConfidenceSweepPoint>,
    /// Exact global-optimum micro-F1 operating point over every unique detection score across all classes.
    pub best_micro_f1: OptimalF1Point,
    /// Optimal macro-F1 operating point over `thresholds` (highest confidence threshold on ties).
    pub best_macro_f1: OptimalF1Point,
    /// One [`ClassConfidenceSweep`] per class `0..num_classes`.
    pub per_class: Vec<ClassConfidenceSweep>,
}

/// Compute exact optimal F1 over all unique score thresholds of a descending-score-sorted `(score, is_tp)` slice.
fn exact_best_f1_sorted(sorted: &[(f32, bool)], n_gt: u64) -> OptimalF1Point {
    let fp_at_zero = sorted.iter().filter(|&&(s, tp)| !tp && s >= 0.0).count() as u64;
    let mut best = OptimalF1Point {
        conf_thresh: 0.0,
        f1: 0.0,
        precision: 0.0,
        recall: 0.0,
        tp: 0,
        fp: fp_at_zero,
        fn_count: n_gt,
    };
    if n_gt == 0 || sorted.is_empty() {
        return best;
    }
    let n_gt_f = n_gt as f64;
    let mut cur_tp = 0u64;
    let mut cur_fp = 0u64;
    let n = sorted.len();
    for (idx, &(score, is_tp)) in sorted.iter().enumerate() {
        if is_tp {
            cur_tp += 1;
        } else {
            cur_fp += 1;
        }
        let is_run_end = idx + 1 == n || sorted[idx + 1].0 != score;
        if is_run_end && cur_tp > 0 {
            let cur_num = cur_tp as u128;
            let cur_den = (cur_tp + cur_fp + n_gt) as u128;
            let best_num = best.tp as u128;
            let best_den = (best.tp + best.fp + n_gt) as u128;
            if best.tp == 0 || cur_num * best_den > best_num * cur_den {
                let precision = cur_tp as f64 / (cur_tp + cur_fp) as f64;
                let recall = cur_tp as f64 / n_gt_f;
                let f1 = (2.0 * cur_tp as f64) / (cur_tp + cur_fp + n_gt) as f64;
                best = OptimalF1Point {
                    conf_thresh: score,
                    f1,
                    precision,
                    recall,
                    tp: cur_tp,
                    fp: cur_fp,
                    fn_count: n_gt - cur_tp,
                };
            }
        }
    }
    best
}

/// Evaluate a confidence sweep along ascending `thresholds` for a descending-score-sorted `(score, is_tp)` slice.
fn sweep_curve_sorted(sorted: &[(f32, bool)], n_gt: u64, thresholds: &[f32]) -> Vec<ConfidenceSweepPoint> {
    let m = thresholds.len();
    let mut pts = vec![
        ConfidenceSweepPoint {
            conf_thresh: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            tp: 0,
            fp: 0,
            fn_count: n_gt,
        };
        m
    ];
    let mut j = 0usize;
    let mut tp = 0u64;
    let mut fp = 0u64;
    for i in (0..m).rev() {
        let thr = thresholds[i];
        while j < sorted.len() && sorted[j].0 >= thr {
            if sorted[j].1 {
                tp += 1;
            } else {
                fp += 1;
            }
            j += 1;
        }
        let prec = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let rec = if n_gt > 0 { tp as f64 / n_gt as f64 } else { 0.0 };
        let f1 = if tp > 0 && n_gt > 0 { (2.0 * tp as f64) / (tp + fp + n_gt) as f64 } else { 0.0 };
        pts[i] = ConfidenceSweepPoint {
            conf_thresh: thr,
            precision: prec,
            recall: rec,
            f1,
            tp,
            fp,
            fn_count: n_gt.saturating_sub(tp),
        };
    }
    pts
}

/// Compute per-class, micro-averaged, and macro-averaged Precision / Recall / F1-Confidence sweep curves
/// along with optimal confidence thresholds and peak F1 operating points.
pub fn compute_confidence_sweep<P, T>(
    num_classes: usize,
    cfg: &ConfidenceSweepConfig,
    images: &[(P, T)],
) -> Result<ConfidenceSweepDiagnostics>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    cfg.validate()?;
    let (per_class_dets, gt_counts) =
        match_images_by_class(num_classes, cfg.iou_thresh, None, false, images)?;

    let m = cfg.num_thresholds;
    let step = (cfg.max_conf as f64 - cfg.min_conf as f64) / (m - 1) as f64;
    let thresholds: Vec<f32> = (0..m)
        .map(|i| if i + 1 == m { cfg.max_conf } else { (cfg.min_conf as f64 + i as f64 * step) as f32 })
        .collect();

    let mut per_class = Vec::with_capacity(num_classes);
    let mut pooled: Vec<(f32, bool)> = Vec::new();
    let total_gt: u64 = gt_counts.iter().sum();

    let mut macro_prec = vec![0.0f64; m];
    let mut macro_rec = vec![0.0f64; m];
    let mut macro_f1 = vec![0.0f64; m];
    let mut active_classes = 0usize;

    for (c, dets) in per_class_dets.iter().enumerate() {
        let mut sorted: Vec<(f32, bool)> = dets.iter().map(|d| (d.score, d.is_tp)).collect();
        sorted.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        pooled.extend_from_slice(&sorted);

        let n_gt = gt_counts[c];
        let best_f1 = exact_best_f1_sorted(&sorted, n_gt);
        let points = sweep_curve_sorted(&sorted, n_gt, &thresholds);
        if n_gt > 0 {
            active_classes += 1;
            for i in 0..m {
                macro_prec[i] += points[i].precision;
                macro_rec[i] += points[i].recall;
                macro_f1[i] += points[i].f1;
            }
        }
        per_class.push(ClassConfidenceSweep {
            class: c as u32,
            num_truths: n_gt,
            num_preds: sorted.len() as u64,
            best_f1,
            points,
        });
    }

    pooled.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let best_micro_f1 = exact_best_f1_sorted(&pooled, total_gt);
    let micro_curve = sweep_curve_sorted(&pooled, total_gt, &thresholds);

    let mut macro_curve = Vec::with_capacity(m);
    let mut best_macro_f1 = OptimalF1Point {
        conf_thresh: thresholds[0],
        f1: 0.0,
        precision: 0.0,
        recall: 0.0,
        tp: micro_curve[0].tp,
        fp: micro_curve[0].fp,
        fn_count: micro_curve[0].fn_count,
    };
    let mut best_macro_score = -1.0f64;

    for i in 0..m {
        let (p, r, f) = if active_classes > 0 {
            let inv = 1.0 / active_classes as f64;
            (macro_prec[i] * inv, macro_rec[i] * inv, macro_f1[i] * inv)
        } else {
            (0.0, 0.0, 0.0)
        };
        let mc = micro_curve[i];
        macro_curve.push(ConfidenceSweepPoint {
            conf_thresh: thresholds[i],
            precision: p,
            recall: r,
            f1: f,
            tp: mc.tp,
            fp: mc.fp,
            fn_count: mc.fn_count,
        });
        // Pick highest threshold on exact ties across the actual grid (including zero-F1 ties).
        if f >= best_macro_score {
            best_macro_score = f;
            best_macro_f1 = OptimalF1Point {
                conf_thresh: thresholds[i],
                f1: f,
                precision: p,
                recall: r,
                tp: mc.tp,
                fp: mc.fp,
                fn_count: mc.fn_count,
            };
        }
    }

    Ok(ConfidenceSweepDiagnostics {
        iou_thresh: cfg.iou_thresh,
        thresholds,
        micro_curve,
        macro_curve,
        best_micro_f1,
        best_macro_f1,
        per_class,
    })
}

/// Binning scheme for reliability-bin calibration diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinningStrategy {
    /// Standard uniform equal-width bins over `[0, 1]` (ECE / MCE).
    #[default]
    EqualWidth,
    /// Adaptive equal-mass (quantile) bins partitioning sorted detections into balanced buckets (Adaptive
    /// Calibration Error `ACE` / MCE, Nixon et al. CVPRW 2019).
    EqualMass,
}

/// Configuration for [`compute_calibration_report`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationReportConfig {
    /// IoU threshold in `[0, 1]` for TP vs FP matching (default `0.5`).
    pub iou_thresh: f32,
    /// Minimum confidence threshold in `[0, 1]` applied before binning (default `0.0`).
    pub conf_thresh: f32,
    /// Number of reliability bins (`>= 1`, default `15`).
    pub num_bins: usize,
    /// Binning scheme (default [`BinningStrategy::EqualWidth`]).
    pub binning: BinningStrategy,
}

impl Default for CalibrationReportConfig {
    fn default() -> Self {
        Self { iou_thresh: 0.5, conf_thresh: 0.0, num_bins: 15, binning: BinningStrategy::EqualWidth }
    }
}

impl CalibrationReportConfig {
    /// Validate calibration report parameters.
    pub fn validate(&self) -> Result<()> {
        if !(self.iou_thresh.is_finite() && (0.0..=1.0).contains(&self.iou_thresh)) {
            return Err(Error::Config(format!(
                "CalibrationReportConfig: iou_thresh must be finite and in [0, 1], got {}",
                self.iou_thresh
            )));
        }
        if !(self.conf_thresh.is_finite() && (0.0..=1.0).contains(&self.conf_thresh)) {
            return Err(Error::Config(format!(
                "CalibrationReportConfig: conf_thresh must be finite and in [0, 1], got {}",
                self.conf_thresh
            )));
        }
        if self.num_bins == 0 {
            return Err(Error::Config("CalibrationReportConfig: num_bins must be >= 1, got 0".into()));
        }
        Ok(())
    }
}

/// Extended reliability-bin calibration diagnostics (count-weighted ECE, equal-bin-weight calibration error,
/// MCE, Brier score, signed bias, and overconfidence error).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    /// Binning strategy used (`EqualWidth` or `EqualMass`).
    pub binning: BinningStrategy,
    /// Sample-count-weighted Expected Calibration Error:
    /// `sum_b (count_b / N) * |avg_accuracy_b - avg_confidence_b|`.
    pub ece: f64,
    /// Equal-bin-weight mean calibration gap across non-empty bins:
    /// `(1 / B_nonempty) * sum_{b: count_b > 0} |avg_accuracy_b - avg_confidence_b|`.
    /// Under [`BinningStrategy::EqualMass`], this matches the equal-bin-weight Adaptive Calibration Error
    /// formulation of Nixon et al. (CVPRW 2019 §4).
    pub equal_bin_ece: f64,
    /// Maximum Calibration Error: `max_{b: count_b > 0} |avg_accuracy_b - avg_confidence_b|`.
    pub mce: f64,
    /// Brier score: `(1 / N) * sum_i (p_i - y_i)²`.
    pub brier: f64,
    /// Mean signed confidence bias `(1 / N) * sum_i (p_i - y_i)` (`> 0`: overconfident, `< 0`: underconfident).
    pub signed_bias: f64,
    /// Overconfidence Error `sum_b (count_b / N) * avg_confidence_b * max(avg_confidence_b - avg_accuracy_b, 0)`.
    pub overconfidence_error: f64,
    /// Total number of evaluated detections `N`.
    pub count: usize,
    /// Per-bin reliability breakdown.
    pub bins: Vec<ReliabilityBin>,
}

/// Pooled and per-class detection calibration report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionCalibrationReport {
    /// Configuration used to compute this report.
    pub config: CalibrationReportConfig,
    /// Pooled calibration report across all classes.
    pub overall: CalibrationReport,
    /// Per-class calibration reports for classes `0..num_classes`.
    pub per_class: Vec<CalibrationReport>,
    /// Macro average of per-class count-weighted `ece` across non-empty classes (`0.0` if none).
    pub mean_class_ece: f64,
    /// Static Calibration Error (macro average of per-class fixed-width `ece` across non-empty classes),
    /// populated (`Some`) under [`BinningStrategy::EqualWidth`] and `None` under [`BinningStrategy::EqualMass`].
    pub static_calibration_error: Option<f64>,
    /// Macro average of per-class equal-bin-weight `equal_bin_ece` across non-empty classes (matches
    /// class-averaged ACE under [`BinningStrategy::EqualMass`]; `0.0` if none).
    pub mean_class_equal_bin_ece: f64,
    /// Worst per-class `ece` across non-empty classes (`0.0` if none).
    pub max_class_ece: f64,
}

/// Compute extended reliability-bin calibration diagnostics (ECE/ACE, MCE, Brier, signed bias, and
/// overconfidence error) from pre-matched confidence `scores` in `[0, 1]` and TP flags `is_tp`.
pub fn reliability_diagnostics_from_scores(
    scores: &[f32],
    is_tp: &[bool],
    num_bins: usize,
    binning: BinningStrategy,
) -> Result<CalibrationReport> {
    if num_bins == 0 {
        return Err(Error::Config("reliability_diagnostics_from_scores: num_bins must be >= 1".into()));
    }
    if scores.len() != is_tp.len() {
        return Err(Error::Config(format!(
            "reliability_diagnostics_from_scores: scores length ({}) != is_tp length ({})",
            scores.len(),
            is_tp.len()
        )));
    }
    for (i, &s) in scores.iter().enumerate() {
        if !(s.is_finite() && (0.0..=1.0).contains(&s)) {
            return Err(Error::Config(format!(
                "reliability_diagnostics_from_scores: scores[{i}] must be finite and in [0, 1], got {s}"
            )));
        }
    }

    let n = scores.len();
    let mut brier_sum = 0.0f64;
    let mut bias_sum = 0.0f64;
    for (&sf, &tp) in scores.iter().zip(is_tp) {
        let p = sf as f64;
        let y = if tp { 1.0 } else { 0.0 };
        let d = p - y;
        brier_sum += d * d;
        bias_sum += d;
    }

    let bins = match binning {
        BinningStrategy::EqualWidth => {
            let m_f = num_bins as f64;
            let edges: Vec<f64> =
                (0..=num_bins).map(|b| if b == num_bins { 1.0 } else { b as f64 / m_f }).collect();
            let mut tp_c = vec![0usize; num_bins];
            let mut fp_c = vec![0usize; num_bins];
            let mut sum_conf = vec![0.0f64; num_bins];
            for (&sf, &tp) in scores.iter().zip(is_tp) {
                let p = sf as f64;
                let mut b = ((p * m_f).floor() as usize).min(num_bins - 1);
                while b + 1 < num_bins && p >= edges[b + 1] {
                    b += 1;
                }
                while b > 0 && p < edges[b] {
                    b -= 1;
                }
                if tp {
                    tp_c[b] += 1;
                } else {
                    fp_c[b] += 1;
                }
                sum_conf[b] += p;
            }
            (0..num_bins)
                .map(|b| make_bin(edges[b], edges[b + 1], tp_c[b], fp_c[b], sum_conf[b]))
                .collect::<Vec<_>>()
        }
        BinningStrategy::EqualMass => {
            if n == 0 {
                (0..num_bins)
                    .map(|b| {
                        let lo = b as f64 / num_bins as f64;
                        let hi = if b + 1 == num_bins { 1.0 } else { (b + 1) as f64 / num_bins as f64 };
                        make_bin(lo, hi, 0, 0, 0.0)
                    })
                    .collect()
            } else {
                // Group identical scores into atomic runs so tied confidence values are never split across
                // bins, making equal-mass metrics strictly permutation-invariant.
                let mut pairs: Vec<(f64, bool)> =
                    scores.iter().zip(is_tp).map(|(&s, &tp)| (s as f64, tp)).collect();
                pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

                struct ScoreRun {
                    score: f64,
                    count: usize,
                    tp: usize,
                }
                let mut runs: Vec<ScoreRun> = Vec::new();
                for (score, is_pos) in pairs {
                    if let Some(last) = runs.last_mut()
                        && last.score == score
                    {
                        last.count += 1;
                        if is_pos {
                            last.tp += 1;
                        }
                    } else {
                        runs.push(ScoreRun { score, count: 1, tp: usize::from(is_pos) });
                    }
                }

                let target_bins = num_bins.min(runs.len());
                let mut out = Vec::with_capacity(target_bins);
                let mut run_idx = 0usize;
                let mut emitted_samples = 0u128;
                let total_samples = n as u128;

                while run_idx < runs.len() {
                    let bins_left = (target_bins - out.len()).max(1);
                    let rem_samples = total_samples - emitted_samples;
                    let lower = runs[run_idx].score;
                    let mut upper = lower;
                    let mut tp_b = 0usize;
                    let mut count_b = 0usize;
                    let mut sum_c = 0.0f64;

                    if bins_left == 1 {
                        for r in &runs[run_idx..] {
                            upper = r.score;
                            tp_b += r.tp;
                            count_b += r.count;
                            sum_c += r.score * r.count as f64;
                        }
                        run_idx = runs.len();
                    } else {
                        let bins_left_u = bins_left as u128;
                        loop {
                            let r = &runs[run_idx];
                            upper = r.score;
                            tp_b += r.tp;
                            count_b += r.count;
                            sum_c += r.score * r.count as f64;
                            run_idx += 1;
                            let rem_runs = runs.len() - run_idx;
                            if rem_runs < bins_left {
                                break;
                            }
                            let cur_err = (count_b as u128 * bins_left_u).abs_diff(rem_samples);
                            let next_count = count_b as u128 + runs[run_idx].count as u128;
                            let next_err = (next_count * bins_left_u).abs_diff(rem_samples);
                            if next_err >= cur_err {
                                break;
                            }
                        }
                    }
                    emitted_samples += count_b as u128;
                    out.push(make_bin(lower, upper, tp_b, count_b - tp_b, sum_c));
                }
                out
            }
        }
    };

    let mut ece = 0.0f64;
    let mut gap_sum = 0.0f64;
    let mut non_empty_bins = 0usize;
    let mut mce = 0.0f64;
    let mut oe = 0.0f64;
    if n > 0 {
        let inv_n = 1.0 / n as f64;
        for bin in &bins {
            if bin.count == 0 {
                continue;
            }
            non_empty_bins += 1;
            gap_sum += bin.gap;
            let frac = bin.count as f64 * inv_n;
            ece += frac * bin.gap;
            if bin.gap > mce {
                mce = bin.gap;
            }
            let over = (bin.avg_confidence - bin.avg_accuracy).max(0.0);
            oe += frac * bin.avg_confidence * over;
        }
    }

    let equal_bin_ece = if non_empty_bins == 0 { 0.0 } else { gap_sum / non_empty_bins as f64 };
    let (brier, signed_bias) = if n == 0 { (0.0, 0.0) } else { (brier_sum / n as f64, bias_sum / n as f64) };
    Ok(CalibrationReport {
        binning,
        ece,
        equal_bin_ece,
        mce,
        brier,
        signed_bias,
        overconfidence_error: oe,
        count: n,
        bins,
    })
}

#[inline]
fn make_bin(lower: f64, upper: f64, tp: usize, fp: usize, sum_conf: f64) -> ReliabilityBin {
    let count = tp + fp;
    if count == 0 {
        ReliabilityBin {
            lower,
            upper,
            count: 0,
            tp: 0,
            fp: 0,
            avg_confidence: 0.0,
            avg_accuracy: 0.0,
            gap: 0.0,
        }
    } else {
        let avg_confidence = sum_conf / count as f64;
        let avg_accuracy = tp as f64 / count as f64;
        let gap = (avg_confidence - avg_accuracy).abs();
        ReliabilityBin { lower, upper, count, tp, fp, avg_confidence, avg_accuracy, gap }
    }
}

/// Compute pooled and per-class detection calibration diagnostics (equal-width ECE/MCE or equal-mass
/// ACE/MCE, Brier score, signed bias, overconfidence error, and Static Calibration Error `SCE`) directly
/// from `(preds, truths)` images.
pub fn compute_calibration_report<P, T>(
    num_classes: usize,
    cfg: &CalibrationReportConfig,
    images: &[(P, T)],
) -> Result<DetectionCalibrationReport>
where
    P: AsRef<[Detection]>,
    T: AsRef<[Truth]>,
{
    cfg.validate()?;
    let (per_class_dets, _) =
        match_images_by_class(num_classes, cfg.iou_thresh, Some(cfg.conf_thresh), true, images)?;

    let mut all_scores = Vec::new();
    let mut all_tp = Vec::new();
    let mut per_class = Vec::with_capacity(num_classes);
    let mut ece_sum = 0.0f64;
    let mut eq_ece_sum = 0.0f64;
    let mut non_empty = 0usize;
    let mut max_class_ece = 0.0f64;

    for dets in &per_class_dets {
        let scores: Vec<f32> = dets.iter().map(|d| d.score).collect();
        let flags: Vec<bool> = dets.iter().map(|d| d.is_tp).collect();
        all_scores.extend_from_slice(&scores);
        all_tp.extend_from_slice(&flags);
        let rep = reliability_diagnostics_from_scores(&scores, &flags, cfg.num_bins, cfg.binning)?;
        if rep.count > 0 {
            ece_sum += rep.ece;
            eq_ece_sum += rep.equal_bin_ece;
            non_empty += 1;
            if rep.ece > max_class_ece {
                max_class_ece = rep.ece;
            }
        }
        per_class.push(rep);
    }

    let overall = reliability_diagnostics_from_scores(&all_scores, &all_tp, cfg.num_bins, cfg.binning)?;
    let mean_class_ece = if non_empty == 0 { 0.0 } else { ece_sum / non_empty as f64 };
    let mean_class_equal_bin_ece = if non_empty == 0 { 0.0 } else { eq_ece_sum / non_empty as f64 };
    let static_calibration_error = (cfg.binning == BinningStrategy::EqualWidth).then_some(mean_class_ece);

    Ok(DetectionCalibrationReport {
        config: *cfg,
        overall,
        per_class,
        mean_class_ece,
        static_calibration_error,
        mean_class_equal_bin_ece,
        max_class_ece,
    })
}
