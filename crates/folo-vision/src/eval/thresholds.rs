//! Global and per-class F1-optimal confidence threshold calibration.
//!
//! Computed directly from [`crate::Evaluator`]'s existing IoU 0.5 confidence sweep (`0.000..=1.000` in `0.001`
//! steps) with zero extra inference passes, and ready to feed straight into [`crate::NmsConfig::per_class_conf`]
//! or [`crate::Predictor::with_calibrated_thresholds`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Global F1-optimal confidence operating point at IoU 0.5 on the evaluator's `0.001` confidence grid.
///
/// Detections with `score >= best_conf` enter the IoU 0.5 sweep (matching [`crate::EvalReport::best_conf`]); on
/// equal F1 values, the highest confidence threshold wins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OperatingPoint {
    /// Confidence threshold maximising class-mean F1 at IoU 0.5 (highest threshold on ties; `0.0` if no class has
    /// ground truth).
    pub best_conf: f32,
    /// Class-mean F1 at `best_conf`.
    pub best_f1: f32,
    /// Class-mean precision at `best_conf`.
    pub precision: f32,
    /// Class-mean recall at `best_conf`.
    pub recall: f32,
}

/// F1-optimal confidence operating point for one evaluated class (a class with at least one ground truth).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClassThreshold {
    /// Class index.
    pub class: u32,
    /// Confidence threshold maximising this class's F1 at IoU 0.5 (highest threshold on ties).
    pub best_conf: f32,
    /// Maximum F1 achieved by this class at `best_conf`.
    pub best_f1: f32,
    /// Precision of this class at `best_conf`.
    pub precision: f32,
    /// Recall of this class at `best_conf`.
    pub recall: f32,
    /// Number of ground truths evaluated for this class.
    pub truths: u64,
    /// Number of predictions evaluated for this class.
    pub preds: u64,
}

/// Global and per-class F1-optimal confidence thresholds computed during evaluation
/// ([`crate::Evaluator::compute`] / [`crate::val::ValResult::thresholds`]) from the existing IoU 0.5 confidence
/// sweep with zero extra inference passes.
///
/// Note on boundary semantics: [`crate::Evaluator`] evaluates thresholds with `score >= best_conf`, whereas
/// [`crate::NmsConfig`] and [`crate::Predictor`] retain detections with strict `score > conf`. On continuous model
/// probabilities the difference is a measure-zero boundary event, while on discrete quantized scores callers may
/// keep or nudge thresholds as needed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThresholdCalibration {
    /// Global operating point maximising class-mean F1 across all evaluated classes.
    pub global: OperatingPoint,
    /// Per-class optimal operating points in ascending class-index order (only classes with `truths > 0`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_class: Vec<ClassThreshold>,
    /// Macro-average of each evaluated class's own [`ClassThreshold::best_f1`] (`>= global.best_f1` up to float
    /// rounding, since each class independently picks its own peak).
    pub macro_f1: f32,
}

impl ThresholdCalibration {
    /// Whether no classes with ground truth were calibrated.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.per_class.is_empty()
    }

    /// Build a `class_id -> best_conf` map for every calibrated class, ready for
    /// [`crate::NmsConfig::per_class_conf`] or [`crate::Predictor::with_per_class_conf`].
    pub fn per_class_map(&self) -> BTreeMap<u32, f32> {
        self.per_class.iter().map(|c| (c.class, c.best_conf)).collect()
    }

    /// Optimal threshold for `class`, falling back to [`OperatingPoint::best_conf`] when `class` had no ground
    /// truth during evaluation.
    pub fn threshold_for(&self, class: u32) -> f32 {
        self.per_class.iter().find(|c| c.class == class).map_or(self.global.best_conf, |c| c.best_conf)
    }

    /// Validate that the global threshold and every per-class threshold are finite and in `0.0..=1.0`.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.global.best_conf) {
            return Err(Error::Config(format!(
                "calibrated global best_conf must be finite and in 0..=1, got {}",
                self.global.best_conf
            )));
        }
        for c in &self.per_class {
            if !(0.0..=1.0).contains(&c.best_conf) {
                return Err(Error::Config(format!(
                    "calibrated per-class best_conf for class {} must be finite and in 0..=1, got {}",
                    c.class, c.best_conf
                )));
            }
        }
        Ok(())
    }
}

/// Validate a per-class confidence threshold map: every threshold must be finite and in `0.0..=1.0`.
pub fn validate_per_class_conf(map: &BTreeMap<u32, f32>) -> Result<()> {
    for (&class, &conf) in map {
        if !(0.0..=1.0).contains(&conf) {
            return Err(Error::Config(format!(
                "per_class_conf for class {class} must be finite and in 0..=1, got {conf}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_calibration_lookup_map_and_validation() {
        let cal = ThresholdCalibration {
            global: OperatingPoint { best_conf: 0.40, best_f1: 0.75, precision: 0.80, recall: 0.70 },
            per_class: vec![
                ClassThreshold {
                    class: 0,
                    best_conf: 0.25,
                    best_f1: 0.85,
                    precision: 0.85,
                    recall: 0.85,
                    truths: 10,
                    preds: 10,
                },
                ClassThreshold {
                    class: 3,
                    best_conf: 0.65,
                    best_f1: 0.90,
                    precision: 0.90,
                    recall: 0.90,
                    truths: 10,
                    preds: 10,
                },
            ],
            macro_f1: 0.875,
        };
        assert!(!cal.is_empty());
        assert!(cal.validate().is_ok());
        assert_eq!(cal.threshold_for(0), 0.25);
        assert_eq!(cal.threshold_for(3), 0.65);
        assert_eq!(cal.threshold_for(1), 0.40, "unseen class falls back to global best_conf");

        let map = cal.per_class_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&0], 0.25);
        assert_eq!(map[&3], 0.65);
        assert!(validate_per_class_conf(&map).is_ok());

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
            let mut m = BTreeMap::new();
            m.insert(2, bad);
            assert!(validate_per_class_conf(&m).is_err(), "expected rejection for {bad}");
            let mut bad_cal = cal.clone();
            bad_cal.per_class[0].best_conf = bad;
            assert!(bad_cal.validate().is_err(), "expected rejection for class conf {bad}");
            let mut bad_global = cal.clone();
            bad_global.global.best_conf = bad;
            assert!(bad_global.validate().is_err(), "expected rejection for global conf {bad}");
        }
    }
}
