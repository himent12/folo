//! Results: everything measured for a finished entry, and the per-image detections it made.

use serde::{Deserialize, Serialize};

use crate::{BenchDetection, BenchEntry, BenchImage};

/// `GET /api/bench/benchmarks/{id}/entries/{entry_id}/result`: everything measured, for a finished entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchResult {
    /// The entry.
    pub entry: BenchEntry,
    /// Accuracy on the unchanged images.
    pub accuracy: BenchAccuracy,
    /// Accuracy under each condition; `null` when the conditions were not run.
    pub robustness: Option<BenchRobustness>,
    /// How well confidence predicts being right; `null` without confidences, and when no detection reaches
    /// confidence 0.1 (the entry is then `partial`).
    pub calibration: Option<BenchCalibration>,
    /// How tight the boxes are.
    pub localization: BenchLocalization,
    /// What the contestant's own labels would be worth.
    pub self_labeling: BenchSelfLabeling,
}

/// Accuracy on the unchanged images (docs/METRICS.md for AP).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchAccuracy {
    /// mAP@0.5:0.95: the accuracy axis.
    pub map50_95: f32,
    /// Its 95% bootstrap interval over images, `[low, high]`.
    pub ci95: [f32; 2],
    /// mAP@0.5.
    pub map50: f32,
    /// mAP@0.75.
    pub map75: f32,
    /// Class-mean precision at `best_conf`.
    pub precision: f32,
    /// Class-mean recall at `best_conf`.
    pub recall: f32,
    /// Class-mean F1 at `best_conf`.
    pub f1: f32,
    /// The confidence with the best class-mean F1 (the highest on ties). With `f1` 0 there is no best F1: F1 is 0 at
    /// every confidence and this is 1, or 0 when no class has a true box.
    pub best_conf: f32,
    /// mAP@0.5:0.95 of objects under 32² px; `null` when the benchmark has none.
    pub map_small: Option<f32>,
    /// Of objects from 32² to 96² px.
    pub map_medium: Option<f32>,
    /// Of objects over 96² px.
    pub map_large: Option<f32>,
    /// Per class with at least one true box, in class order.
    pub per_class: Vec<BenchClassAccuracy>,
}

/// Accuracy on one class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchClassAccuracy {
    /// Class index.
    pub class: u32,
    /// True boxes of the class.
    pub truths: u64,
    /// AP@0.5.
    pub map50: f32,
    /// AP@0.5:0.95.
    pub map50_95: f32,
}

/// Accuracy under the plan's conditions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchRobustness {
    /// The robustness axis: the weighted mean of the conditions' `retention`.
    pub retention: f32,
    /// Per condition, in plan order.
    pub conditions: Vec<BenchConditionResult>,
}

/// One condition's result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchConditionResult {
    /// The condition's id.
    pub condition: String,
    /// Its weight in the plan.
    pub weight: f32,
    /// The mean of its severities' `retention`.
    pub retention: f32,
    /// Per severity run, ascending.
    pub points: Vec<BenchSeverityPoint>,
}

/// Accuracy at one severity of a condition.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchSeverityPoint {
    /// The severity, `1..=5`.
    pub severity: u8,
    /// mAP@0.5:0.95 under it.
    pub map50_95: f32,
    /// mAP@0.5 under it.
    pub map50: f32,
    /// `min(1, map50_95 / the unchanged images' map50_95)`; 0 when that is 0.
    pub retention: f32,
}

/// Reliability of the confidences on the unchanged images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchCalibration {
    /// Expected calibration error, `0..=1` (0 is perfect); the axis is `1 - ece`.
    pub ece: f32,
    /// Detections that entered it (confidence at least 0.1).
    pub detections: u64,
    /// Ten bins of confidence, `[0, 0.1)` to `[0.9, 1]`.
    pub bins: Vec<BenchCalibrationBin>,
}

/// One confidence bin of the reliability diagram.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchCalibrationBin {
    /// Lower edge.
    pub lo: f32,
    /// Upper edge.
    pub hi: f32,
    /// Detections in the bin.
    pub count: u64,
    /// Their mean confidence; `null` for an empty bin.
    pub confidence: Option<f32>,
    /// The share of them that are right (IoU at least 0.5 with a true box of the class); `null` for an empty bin.
    pub precision: Option<f32>,
}

/// Box quality of the correct detections on the unchanged images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchLocalization {
    /// Mean IoU of the correct detections with their true boxes, `0.5..=1`; `null` when nothing was correct. The
    /// axis is `(mean_iou - 0.5) / 0.5`.
    pub mean_iou: Option<f32>,
    /// Correct detections it is the mean of.
    pub matches: u64,
    /// How many of them fall in each tenth of IoU from 0.5 to 1: `[0.5, 0.55)`, ..., `[0.95, 1]` (ten counts).
    pub histogram: Vec<u64>,
}

/// Whether a contestant's own labels are good enough to train on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BenchReadiness {
    /// It labels at least half of all objects at the target precision.
    Ready,
    /// At least a quarter.
    Partial,
    /// Less, or it never reaches the target precision.
    NotReady,
}

/// What the contestant's own labels would be worth: the question behind a self-improvement loop, answered on
/// images whose truth is known.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchSelfLabeling {
    /// The precision labels must reach (0.9).
    pub target_precision: f32,
    /// The lowest confidence on the grid `0.00, 0.01, ..., 1.00` at which the detections kept are that precise
    /// and at least 10 of them are kept; `null` when no confidence is (and for a contestant without confidences,
    /// whose one operating point is all there is: it reaches the target with at least 10 detections).
    pub threshold: Option<f32>,
    /// Precision of the labels at `threshold` (of all detections when there is none).
    pub precision: f32,
    /// The share of all true boxes those labels find: the self-labeling axis, 0 when the target is not reached.
    pub recall: f32,
    /// Mean IoU of the right ones with their true boxes; `null` when there are none.
    pub mean_iou: Option<f32>,
    /// Labels per image at `threshold` (of all detections when there is none).
    pub labels_per_image: f32,
    /// Labels (detections) kept at `threshold`; all detections when there is none.
    #[serde(default)]
    pub labels: u64,
    /// The verdict.
    pub readiness: BenchReadiness,
    /// Precision and recall of the labels kept at each confidence `0.00, 0.01, ..., 1.00` (one point without
    /// confidences).
    pub curve: Vec<BenchLabelPoint>,
}

/// One confidence threshold of [`BenchSelfLabeling::curve`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchLabelPoint {
    /// Detections with at least this confidence are kept.
    pub confidence: f32,
    /// The share of them that are right; `null` when none is kept.
    pub precision: Option<f32>,
    /// The share of all true boxes they find.
    pub recall: f32,
    /// How many are kept.
    pub labels: u64,
}

/// One image of a finished entry: the truth and what the contestant found, unchanged image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchEntryImage {
    /// The image with its true boxes.
    pub image: BenchImage,
    /// The detections at or above the page's `conf`, highest confidence first.
    pub detections: Vec<BenchDetection>,
    /// True boxes none of them matches (IoU at least 0.5, same class).
    pub missed: u32,
    /// Detections that match no true box.
    pub wrong: u32,
}

/// `GET /api/bench/benchmarks/{id}/entries/{entry_id}/images`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchEntryImagePage {
    /// Images the entry ran (before `offset` / `limit`).
    pub total: u64,
    /// The confidence the detections were taken at: those at or above it, so 0 is every detection. The entry's
    /// `best_conf`, or 0 when it has no best F1 (`f1` 0). An entry graded by an older FOLO took `best_conf` in either
    /// case. `null` from a server older than this field (which took `best_conf`).
    #[serde(default)]
    pub conf: Option<f32>,
    /// This page.
    pub images: Vec<BenchEntryImage>,
}
