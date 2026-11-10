//! The catalog of `GET /api/bench/catalog`: the stress conditions and the axes of the score.

use serde::{Deserialize, Serialize};

use crate::BenchPlan;
#[cfg(doc)]
use crate::{BENCH_SEVERITIES, BenchPlanCondition};

/// `GET /api/bench/catalog`: what a benchmark can be made of and how it is scored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchCatalog {
    /// Every stress condition, in display order.
    pub conditions: Vec<BenchCondition>,
    /// The highest severity ([`BENCH_SEVERITIES`]).
    pub severities: u8,
    /// The plan a benchmark gets when it names none.
    pub default_plan: BenchPlan,
    /// The axes of the score, in display order, with their weights.
    pub axes: Vec<BenchAxisInfo>,
}

/// What kind of trouble a [`BenchCondition`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum BenchConditionGroup {
    /// Too little or too much light, flat contrast.
    Light,
    /// Defocus and motion.
    Blur,
    /// Sensor noise.
    Noise,
    /// What is between the camera and the scene: haze, fog, murky water, a colour cast.
    Medium,
    /// What the pipeline does to the image: compression, low resolution.
    Digital,
    /// Where the objects are: far away, partly hidden.
    Geometry,
}

/// One stress condition: a deterministic change to an image that keeps its size, at severities 1 to
/// [`BENCH_SEVERITIES`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchCondition {
    /// Stable id, e.g. `haze`; what a [`BenchPlanCondition`] names.
    pub id: String,
    /// Display name, e.g. `Haze`.
    pub name: String,
    /// Its group.
    pub group: BenchConditionGroup,
    /// What it does to an image and what it stands for, in a sentence.
    pub description: String,
}

/// An axis of the score. docs/BENCH.md defines each one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BenchAxis {
    /// mAP@0.5:0.95 on the unchanged images.
    Accuracy,
    /// The share of that accuracy kept under the plan's conditions.
    Robustness,
    /// The share of all objects the contestant labels by itself while at least 9 in 10 of its labels are right.
    SelfLabeling,
    /// How well confidence predicts being right (1 − expected calibration error).
    Calibration,
    /// How tight the boxes of correct detections are.
    Localization,
}

impl BenchAxis {
    /// Every axis, in display order.
    pub const ALL: [BenchAxis; 5] = [
        BenchAxis::Accuracy,
        BenchAxis::Robustness,
        BenchAxis::SelfLabeling,
        BenchAxis::Calibration,
        BenchAxis::Localization,
    ];

    /// The serialised name.
    pub fn as_str(self) -> &'static str {
        match self {
            BenchAxis::Accuracy => "accuracy",
            BenchAxis::Robustness => "robustness",
            BenchAxis::SelfLabeling => "self_labeling",
            BenchAxis::Calibration => "calibration",
            BenchAxis::Localization => "localization",
        }
    }

    /// The axis's share of the score when every axis is measured; the five sum to 1.
    pub fn weight(self) -> f32 {
        match self {
            BenchAxis::Accuracy => 0.35,
            BenchAxis::Robustness => 0.25,
            BenchAxis::SelfLabeling => 0.20,
            BenchAxis::Calibration => 0.10,
            BenchAxis::Localization => 0.10,
        }
    }
}

/// An axis as the catalog lists it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchAxisInfo {
    /// The axis.
    pub axis: BenchAxis,
    /// Display name, e.g. `Self-labeling`.
    pub name: String,
    /// Its share of the score ([`BenchAxis::weight`]).
    pub weight: f32,
    /// What it measures, in a sentence.
    pub description: String,
}
