//! Calibration quality diagnostics: reliability diagram bins, Expected Calibration Error (ECE),
//! Maximum Calibration Error (MCE), binary and multiclass Brier scores, per-class calibration diagnostics,
//! and multiclass top-1 calibration metrics.

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One bin of a reliability diagram over `[lower, upper)`. The final bin is closed `[lower, 1.0]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Lower edge of the bin in `[0, 1]`.
    pub lower: f32,
    /// Upper edge of the bin in `[0, 1]`.
    pub upper: f32,
    /// Number of samples falling in this bin.
    pub count: usize,
    /// Mean predicted confidence in this bin (`0.0` when empty).
    pub avg_confidence: f32,
    /// Mean empirical accuracy in this bin (`0.0` when empty).
    pub avg_accuracy: f32,
    /// Absolute calibration gap `|avg_confidence - avg_accuracy|` (`0.0` when empty).
    pub gap: f32,
}

/// Summary calibration diagnostics over binary probabilities and ground-truth labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error: `sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|`.
    pub ece: f32,
    /// Maximum Calibration Error: `max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|`.
    pub mce: f32,
    /// Brier score: `(1 / N) * sum_i (p_i - y_i)^2`.
    pub brier: f32,
    /// Binary negative log-likelihood (clamped away from 0/1 by `1e-12` in `f64` for finite scoring).
    pub nll: f32,
    /// Total number of evaluated samples `N`.
    pub count: usize,
    /// Equal-width reliability diagram bins over `[0, 1]`.
    pub bins: Vec<CalibrationBin>,
}

/// Per-class and aggregate calibration diagnostics over binary detection probabilities grouped by class id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClasswiseCalibrationMetrics {
    /// Pooled calibration diagnostics across all samples regardless of class.
    pub overall: CalibrationMetrics,
    /// Independent calibration diagnostics for each class `0..num_classes`.
    pub per_class: Vec<CalibrationMetrics>,
    /// Sample-count weighted mean of per-class ECEs (`0.0` when empty).
    pub weighted_class_ece: f32,
    /// Macro-averaged per-class ECE across non-empty classes (`0.0` when empty).
    pub macro_class_ece: f32,
    /// Worst (maximum) per-class ECE across non-empty classes (`0.0` when empty).
    pub max_class_ece: f32,
}

/// Summary calibration diagnostics for multiclass probability distributions `[N, num_classes]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MulticlassCalibrationMetrics {
    /// Top-1 Expected Calibration Error over `max_c p_{i,c}` vs `argmax_c p_{i,c} == y_i`.
    pub ece: f32,
    /// Top-1 Maximum Calibration Error across non-empty confidence bins.
    pub mce: f32,
    /// Multiclass Brier score `(1 / N) * sum_i sum_c (p_{i,c} - 1[c == y_i])^2`.
    pub brier: f32,
    /// Multiclass negative log-likelihood `-(1 / N) * sum_i ln(max(p_{i, y_i}, 1e-12))`.
    pub nll: f32,
    /// Top-1 classification accuracy (`0.0` when empty).
    pub accuracy: f32,
    /// Total number of evaluated samples `N`.
    pub count: usize,
    /// Equal-width reliability diagram bins over top-1 confidence in `[0, 1]`.
    pub bins: Vec<CalibrationBin>,
}

/// Compute reliability diagram bins, ECE, MCE, Brier score, and NLL for binary probabilities `probs`
/// (`p_i in [0, 1]`) and boolean correctness/ground-truth `labels`.
///
/// Uses `num_bins` equal-width bins `[lower, upper)` with the final bin closed `[lower, 1.0]`; bin
/// membership is evaluated against the exact published `f32` bin edges.
pub fn calibration_metrics(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<CalibrationMetrics> {
    if num_bins == 0 {
        return Err(Error::Config("calibration_metrics: num_bins must be >= 1".into()));
    }
    if probs.len() != labels.len() {
        return Err(Error::Config(format!(
            "calibration_metrics: probs length ({}) != labels length ({})",
            probs.len(),
            labels.len()
        )));
    }
    for (i, &p) in probs.iter().enumerate() {
        if !(p.is_finite() && (0.0..=1.0).contains(&p)) {
            return Err(Error::Config(format!(
                "calibration_metrics: prob[{i}] must be finite and in [0, 1], got {p}"
            )));
        }
    }

    let m_f = num_bins as f64;
    let edges: Vec<f32> =
        (0..=num_bins).map(|b| if b == num_bins { 1.0 } else { (b as f64 / m_f) as f32 }).collect();

    let mut counts = vec![0usize; num_bins];
    let mut sum_conf = vec![0.0f64; num_bins];
    let mut sum_acc = vec![0.0f64; num_bins];
    let mut brier_sum = 0.0f64;
    let mut nll_sum = 0.0f64;

    const PROB_EPS: f64 = 1e-12;
    for (&pf, &label) in probs.iter().zip(labels) {
        let p = pf as f64;
        let y = if label { 1.0 } else { 0.0 };
        let mut bin = ((p * m_f).floor() as usize).min(num_bins - 1);
        while bin + 1 < num_bins && pf >= edges[bin + 1] {
            bin += 1;
        }
        while bin > 0 && pf < edges[bin] {
            bin -= 1;
        }
        counts[bin] += 1;
        sum_conf[bin] += p;
        sum_acc[bin] += y;
        let diff = p - y;
        brier_sum += diff * diff;
        if label {
            nll_sum -= p.max(PROB_EPS).ln();
        } else if p <= 0.5 {
            nll_sum -= (-p).ln_1p();
        } else {
            nll_sum -= (1.0 - p).max(PROB_EPS).ln();
        }
    }

    let n = probs.len();
    let mut bins = Vec::with_capacity(num_bins);
    let mut ece = 0.0f64;
    let mut mce = 0.0f64;
    for b in 0..num_bins {
        let lower = edges[b];
        let upper = edges[b + 1];
        let cnt = counts[b];
        if cnt == 0 {
            bins.push(CalibrationBin {
                lower,
                upper,
                count: 0,
                avg_confidence: 0.0,
                avg_accuracy: 0.0,
                gap: 0.0,
            });
        } else {
            let avg_conf = sum_conf[b] / cnt as f64;
            let avg_acc = sum_acc[b] / cnt as f64;
            let gap = (avg_conf - avg_acc).abs();
            ece += (cnt as f64 / n as f64) * gap;
            if gap > mce {
                mce = gap;
            }
            bins.push(CalibrationBin {
                lower,
                upper,
                count: cnt,
                avg_confidence: avg_conf as f32,
                avg_accuracy: avg_acc as f32,
                gap: gap as f32,
            });
        }
    }

    let (brier, nll) =
        if n == 0 { (0.0, 0.0) } else { ((brier_sum / n as f64) as f32, (nll_sum / n as f64) as f32) };
    Ok(CalibrationMetrics { ece: ece as f32, mce: mce as f32, brier, nll, count: n, bins })
}

/// Compute pooled and per-class calibration diagnostics (ECE, MCE, Brier score, NLL, and reliability bins)
/// for detection probabilities `probs`, boolean correctness `labels`, and class ids `class_ids[i] < num_classes`.
pub fn classwise_calibration_metrics(
    probs: &[f32],
    labels: &[bool],
    class_ids: &[usize],
    num_classes: usize,
    num_bins: usize,
) -> Result<ClasswiseCalibrationMetrics> {
    if num_classes == 0 {
        return Err(Error::Config("classwise_calibration_metrics: num_classes must be >= 1".into()));
    }
    if num_bins == 0 {
        return Err(Error::Config("classwise_calibration_metrics: num_bins must be >= 1".into()));
    }
    if probs.len() != labels.len() || probs.len() != class_ids.len() {
        return Err(Error::Config(format!(
            "classwise_calibration_metrics: length mismatch (probs {}, labels {}, class_ids {})",
            probs.len(),
            labels.len(),
            class_ids.len()
        )));
    }
    for (i, &c) in class_ids.iter().enumerate() {
        if c >= num_classes {
            return Err(Error::Config(format!(
                "classwise_calibration_metrics: class_id[{i}] = {c} >= num_classes ({num_classes})"
            )));
        }
    }

    let overall = calibration_metrics(probs, labels, num_bins)?;
    let mut cls_probs = vec![Vec::<f32>::new(); num_classes];
    let mut cls_labels = vec![Vec::<bool>::new(); num_classes];
    for ((&p, &y), &c) in probs.iter().zip(labels).zip(class_ids) {
        cls_probs[c].push(p);
        cls_labels[c].push(y);
    }

    let mut per_class = Vec::with_capacity(num_classes);
    let mut weighted_sum = 0.0f64;
    let mut macro_sum = 0.0f64;
    let mut non_empty = 0usize;
    let mut max_class_ece = 0.0f32;
    for c in 0..num_classes {
        let m = calibration_metrics(&cls_probs[c], &cls_labels[c], num_bins)?;
        if m.count > 0 {
            weighted_sum += m.count as f64 * m.ece as f64;
            macro_sum += m.ece as f64;
            non_empty += 1;
            if m.ece > max_class_ece {
                max_class_ece = m.ece;
            }
        }
        per_class.push(m);
    }

    let n = probs.len();
    let weighted_class_ece = if n == 0 { 0.0 } else { (weighted_sum / n as f64) as f32 };
    let macro_class_ece = if non_empty == 0 { 0.0 } else { (macro_sum / non_empty as f64) as f32 };

    Ok(ClasswiseCalibrationMetrics { overall, per_class, weighted_class_ece, macro_class_ece, max_class_ece })
}

/// Compute top-1 ECE, MCE, accuracy, multiclass Brier score, and multiclass NLL for row-major
/// `[N, num_classes]` probability rows and integer class labels `labels[i] < num_classes`.
pub fn multiclass_calibration_metrics(
    probs: &[f32],
    num_classes: usize,
    labels: &[usize],
    num_bins: usize,
) -> Result<MulticlassCalibrationMetrics> {
    if num_classes < 2 {
        return Err(Error::Config(format!(
            "multiclass_calibration_metrics: num_classes must be >= 2, got {num_classes}"
        )));
    }
    if num_bins == 0 {
        return Err(Error::Config("multiclass_calibration_metrics: num_bins must be >= 1".into()));
    }
    let expected = labels
        .len()
        .checked_mul(num_classes)
        .ok_or_else(|| Error::Config("multiclass_calibration_metrics: size overflow".into()))?;
    if probs.len() != expected {
        return Err(Error::Config(format!(
            "multiclass_calibration_metrics: probs length ({}) != labels.len() * num_classes ({expected})",
            probs.len()
        )));
    }

    let n = labels.len();
    let mut top_conf = Vec::with_capacity(n);
    let mut top_correct = Vec::with_capacity(n);
    let mut brier_sum = 0.0f64;
    let mut nll_sum = 0.0f64;
    let mut correct_count = 0usize;
    const PROB_EPS: f64 = 1e-12;

    for (i, (row, &y)) in probs.chunks_exact(num_classes).zip(labels).enumerate() {
        if y >= num_classes {
            return Err(Error::Config(format!(
                "multiclass_calibration_metrics: label[{i}] = {y} >= num_classes ({num_classes})"
            )));
        }
        let mut best_c = 0usize;
        let mut best_p = -1.0f32;
        for (c, &pf) in row.iter().enumerate() {
            if !(pf.is_finite() && (0.0..=1.0).contains(&pf)) {
                return Err(Error::Config(format!(
                    "multiclass_calibration_metrics: prob[{i}, {c}] must be finite and in [0, 1], got {pf}"
                )));
            }
            if pf > best_p {
                best_p = pf;
                best_c = c;
            }
            let d = pf as f64 - if c == y { 1.0 } else { 0.0 };
            brier_sum += d * d;
        }
        let ok = best_c == y;
        if ok {
            correct_count += 1;
        }
        top_conf.push(best_p);
        top_correct.push(ok);
        nll_sum -= (row[y] as f64).max(PROB_EPS).ln();
    }

    let top = calibration_metrics(&top_conf, &top_correct, num_bins)?;
    let (brier, nll, accuracy) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        ((brier_sum / n as f64) as f32, (nll_sum / n as f64) as f32, (correct_count as f64 / n as f64) as f32)
    };
    Ok(MulticlassCalibrationMetrics {
        ece: top.ece,
        mce: top.mce,
        brier,
        nll,
        accuracy,
        count: n,
        bins: top.bins,
    })
}

/// Binary Brier score `(1 / N) * sum_i (p_i - y_i)^2` (returns `0.0` on empty input).
pub fn brier_score(probs: &[f32], labels: &[bool]) -> Result<f32> {
    if probs.len() != labels.len() {
        return Err(Error::Config(format!(
            "brier_score: probs length ({}) != labels length ({})",
            probs.len(),
            labels.len()
        )));
    }
    let mut sum = 0.0f64;
    for (i, (&pf, &label)) in probs.iter().zip(labels).enumerate() {
        if !(pf.is_finite() && (0.0..=1.0).contains(&pf)) {
            return Err(Error::Config(format!(
                "brier_score: prob[{i}] must be finite and in [0, 1], got {pf}"
            )));
        }
        let d = pf as f64 - if label { 1.0 } else { 0.0 };
        sum += d * d;
    }
    Ok(if probs.is_empty() { 0.0 } else { (sum / probs.len() as f64) as f32 })
}

/// Multiclass Brier score `(1 / N) * sum_i sum_c (p_{i,c} - 1[c == y_i])^2` over row-major `[N, num_classes]`
/// probabilities and integer class labels `labels[i] < num_classes` (returns `0.0` on empty input).
pub fn multiclass_brier_score(probs: &[f32], num_classes: usize, labels: &[usize]) -> Result<f32> {
    if num_classes < 2 {
        return Err(Error::Config(format!(
            "multiclass_brier_score: num_classes must be >= 2, got {num_classes}"
        )));
    }
    let expected = labels
        .len()
        .checked_mul(num_classes)
        .ok_or_else(|| Error::Config("multiclass_brier_score: size overflow".into()))?;
    if probs.len() != expected {
        return Err(Error::Config(format!(
            "multiclass_brier_score: probs length ({}) != labels.len() * num_classes ({expected})",
            probs.len()
        )));
    }
    let mut sum = 0.0f64;
    for (i, (row, &y)) in probs.chunks_exact(num_classes).zip(labels).enumerate() {
        if y >= num_classes {
            return Err(Error::Config(format!(
                "multiclass_brier_score: label[{i}] = {y} >= num_classes ({num_classes})"
            )));
        }
        for (c, &pf) in row.iter().enumerate() {
            if !(pf.is_finite() && (0.0..=1.0).contains(&pf)) {
                return Err(Error::Config(format!(
                    "multiclass_brier_score: prob[{i}, {c}] must be finite and in [0, 1], got {pf}"
                )));
            }
            let d = pf as f64 - if c == y { 1.0 } else { 0.0 };
            sum += d * d;
        }
    }
    Ok(if labels.is_empty() { 0.0 } else { (sum / labels.len() as f64) as f32 })
}

/// Expected Calibration Error (ECE) with `num_bins` equal-width bins.
pub fn expected_calibration_error(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<f32> {
    Ok(calibration_metrics(probs, labels, num_bins)?.ece)
}

/// Maximum Calibration Error (MCE) with `num_bins` equal-width bins.
pub fn max_calibration_error(probs: &[f32], labels: &[bool], num_bins: usize) -> Result<f32> {
    Ok(calibration_metrics(probs, labels, num_bins)?.mce)
}
