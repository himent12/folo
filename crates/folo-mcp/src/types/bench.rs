//! Results of the vision-benchmark operations (docs/BENCH.md defines every number), which `folo bench --json`
//! prints and the MCP benchmark tools return.
//!
//! Small on purpose, as the labeling ones: a benchmark as its counts and plan, an entry as its state and, once it
//! is done, its score with the numbers behind each axis and one row per condition. The bulky parts of a result
//! (the reliability bins, the label curve, the IoU histogram, the per-image detections) stay behind the HTTP API
//! (`/api/bench/benchmarks/{id}/entries/{entry_id}/result`, docs/API.md).

use folo_api::{BenchAxis, BenchBest, BenchEntryStatus, BenchReadiness, BenchSource};
use schemars::JsonSchema;
use serde::Serialize;

/// The catalog: the conditions a plan can name, the default plan and the axes of the score.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchCatalogResult {
    /// Every condition, in display order.
    pub conditions: Vec<BenchConditionInfo>,
    /// Severities run from 1 to this.
    pub severities: u8,
    /// The plan a benchmark gets when it names none.
    pub default_plan: Vec<BenchPlanRow>,
    /// The axes of the score with their weights.
    pub axes: Vec<BenchAxisWeight>,
}

/// One condition of the catalog.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchConditionInfo {
    /// Its id, what a plan names.
    pub id: String,
    /// Its group: light, blur, noise, medium, digital or geometry.
    pub group: String,
    /// What it does to an image.
    pub description: String,
}

/// An axis and its share of the score.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchAxisWeight {
    /// The axis.
    pub axis: BenchAxis,
    /// Its share when every axis is measured.
    pub weight: f32,
}

/// One condition of a plan.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchPlanRow {
    /// The condition's id.
    pub condition: String,
    /// The severities run.
    pub severities: Vec<u8>,
    /// How much it counts in the robustness axis.
    pub weight: f32,
}

/// A benchmark in a list.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchSummary {
    /// Its id, e.g. `bn_3f9a0c1e5b7d`.
    pub id: String,
    /// Its name.
    pub name: String,
    /// The dataset's or labeling job's name it was made from.
    pub source_name: String,
    /// Images.
    pub images: u64,
    /// Classes.
    pub num_classes: usize,
    /// Conditions in its plan.
    pub conditions: usize,
    /// Entries, whatever their status.
    pub entries: u64,
    /// The leaderboard's first finished entry.
    pub best: Option<BenchBest>,
    /// An entry or loop of it is running.
    pub active: bool,
}

/// A benchmark with its plan and its leaderboard.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchDetail {
    /// Its id.
    pub id: String,
    /// Its name.
    pub name: String,
    /// What it stands for.
    pub description: String,
    /// Where its images came from.
    pub source: BenchSource,
    /// The dataset's or labeling job's name.
    pub source_name: String,
    /// Class names (first 100); a detection's class indexes them.
    pub classes: Vec<String>,
    /// Images.
    pub images: u64,
    /// True boxes.
    pub boxes: u64,
    /// The plan.
    pub plan: Vec<BenchPlanRow>,
    /// Its seed.
    pub seed: u64,
    /// The hash of its images, truth, plan, seed and grading engine.
    pub fingerprint: String,
    /// An entry or loop of it is running.
    pub active: bool,
    /// The entries, ranked as the leaderboard ranks them (docs/API.md, `BenchEntryList`).
    pub leaderboard: Vec<BenchEntryRow>,
}

/// One entry of a leaderboard.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchEntryRow {
    /// Its id, e.g. `be_91c2`.
    pub id: String,
    /// Its name.
    pub name: String,
    /// `model`, `agent` or `external`.
    pub kind: String,
    /// Where it stands.
    pub status: BenchEntryStatus,
    /// The score, `0..=100`, once done.
    pub score: Option<f32>,
    /// The score is over part of the axes or images.
    pub partial: bool,
    /// mAP@0.5:0.95 on the unchanged images, once done.
    pub map50_95: Option<f32>,
    /// Its 95% interval.
    pub ci95: Option<[f32; 2]>,
    /// Graded on another version of the benchmark (another fingerprint): it compares with nothing here.
    pub stale: bool,
}

/// An entry: where it stands and, once done, its score.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchEntryState {
    /// The benchmark.
    pub benchmark_id: String,
    /// The entry.
    pub entry_id: String,
    /// Its name.
    pub name: String,
    /// Where it stands.
    pub status: BenchEntryStatus,
    /// Images answered so far, over all passes.
    pub done: u64,
    /// Images to answer.
    pub total: u64,
    /// The condition being run (`null`: the unchanged images).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    /// Its severity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<u8>,
    /// Images it runs.
    pub images: u64,
    /// Whether it runs the plan's conditions.
    pub conditions: bool,
    /// Classes of the benchmark the model does not know (AP 0 on them).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_classes: Vec<String>,
    /// Why it failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// How to fix it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// The score and its numbers, once done.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<BenchScoreResult>,
}

/// A finished entry's score, the numbers behind each axis, and one row per condition.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchScoreResult {
    /// `0..=100`: the weighted mean of the measured axes, times 100.
    pub score: f32,
    /// An axis was not measured, or the entry ran part of the images.
    pub partial: bool,
    /// Every axis: its value `0..=1` (`null`: not measured) and its share of the score.
    pub axes: Vec<BenchAxisValue>,
    /// Accuracy on the unchanged images.
    pub accuracy: BenchAccuracySummary,
    /// Accuracy kept under each condition of the plan it ran, in plan order (empty without conditions).
    pub conditions: Vec<BenchConditionRow>,
    /// Expected calibration error (`null` without confidences).
    pub ece: Option<f32>,
    /// Mean IoU of the correct detections.
    pub mean_iou: Option<f32>,
    /// What its own labels would be worth.
    pub self_labeling: BenchSelfLabelingSummary,
    /// Milliseconds per image on the server's device (a model entry).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_per_image: Option<f32>,
}

/// One axis of a score.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchAxisValue {
    /// The axis.
    pub axis: BenchAxis,
    /// `0..=1`; `null` when not measured.
    pub value: Option<f32>,
    /// Its share of the score.
    pub weight: f32,
}

/// Accuracy on the unchanged images.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchAccuracySummary {
    /// mAP@0.5:0.95.
    pub map50_95: f32,
    /// Its 95% bootstrap interval over images.
    pub ci95: [f32; 2],
    /// mAP@0.5.
    pub map50: f32,
    /// mAP@0.75.
    pub map75: f32,
    /// Class-mean precision at `best_conf`.
    pub precision: f32,
    /// Class-mean recall at `best_conf`.
    pub recall: f32,
    /// The confidence of the best class-mean F1.
    pub best_conf: f32,
}

/// One condition of a result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchConditionRow {
    /// The condition.
    pub condition: String,
    /// Its weight in the plan.
    pub weight: f32,
    /// The mean retention over its severities.
    pub retention: f32,
    /// `[severity, map50_95, retention]` per severity, ascending.
    pub points: Vec<[f32; 3]>,
}

/// The self-labeling axis's numbers.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchSelfLabelingSummary {
    /// The lowest confidence whose labels are 90% right (`null`: none).
    pub threshold: Option<f32>,
    /// Precision of those labels.
    pub precision: f32,
    /// The share of all true boxes they find (the axis).
    pub recall: f32,
    /// The verdict.
    pub readiness: BenchReadiness,
}
