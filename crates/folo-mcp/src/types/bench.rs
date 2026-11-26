//! Results of the vision-benchmark operations (docs/BENCH.md defines every number), which `folo bench --json`
//! prints and the MCP benchmark tools return.
//!
//! Small on purpose, as the labeling ones: a benchmark as its counts and plan, an entry as its state and, once it
//! is done, its score with the numbers behind each axis and one row per condition. The bulky parts of a result
//! (the reliability bins, the label curve, the IoU histogram, the per-image detections) stay behind the HTTP API
//! (`/api/bench/benchmarks/{id}/entries/{entry_id}/result`, docs/API.md).

use folo_api::{
    BenchAxis, BenchBest, BenchEntryStatus, BenchPlan, BenchPlanCondition, BenchReadiness, BenchSource, Split,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    /// Where its images came from: `{kind: "dataset", dataset_id, split}` or `{kind: "label_job", job_id}` (an
    /// object in the schema, which stays small).
    #[schemars(with = "serde_json::Map<String, serde_json::Value>")]
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
    /// The score and its numbers, once done. Its schema is an object only: the tool listing every session carries
    /// stays small, and docs/MCP.md (Benchmarks) has its fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<serde_json::Map<String, serde_json::Value>>")]
    pub result: Option<BenchScoreResult>,
    /// Headline score returned directly on `BenchEntry` (used by [`BenchEntryBrief`] even if fetching the
    /// detailed `/result` breakdown failed after `/finish`).
    #[serde(skip)]
    pub(crate) summary_score: Option<f32>,
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

// The arguments and the small results of the MCP benchmark tools.

/// `start_bench_entry` and `stop_bench_entry` result: where the entry stands (`bench_entry` has the rest).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchEntryBrief {
    /// The benchmark.
    pub benchmark_id: String,
    /// The entry.
    pub entry_id: String,
    /// Its name.
    pub name: String,
    /// Where it stands.
    pub status: BenchEntryStatus,
    /// Images run (each pass counts its images).
    pub done: u64,
    /// Images to run.
    pub total: u64,
    /// Its score once done (a file of predictions is scored at once).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

impl From<BenchEntryState> for BenchEntryBrief {
    fn from(s: BenchEntryState) -> BenchEntryBrief {
        BenchEntryBrief {
            benchmark_id: s.benchmark_id,
            entry_id: s.entry_id,
            name: s.name,
            status: s.status,
            done: s.done,
            total: s.total,
            score: s.result.map(|r| r.score).or(s.summary_score),
        }
    }
}

/// `delete_benchmark` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BenchDeleted {
    /// The benchmark.
    pub benchmark_id: String,
    /// Always true.
    pub deleted: bool,
}

/// `get_benchmark` and `delete_benchmark`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BenchmarkIdArgs {
    /// Benchmark id, e.g. `bn_3f9a0c1e5b7d`.
    pub benchmark_id: String,
}

/// `stop_bench_entry`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BenchEntryIdArgs {
    /// Benchmark id.
    pub benchmark_id: String,
    /// Entry id, e.g. `be_91c2`.
    pub entry_id: String,
}

/// `bench_entry`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BenchEntryArgs {
    /// Benchmark id.
    pub benchmark_id: String,
    /// Entry id.
    pub entry_id: String,
    /// Seconds to wait for a running entry to end before answering (0 to 50).
    #[serde(default)]
    #[schemars(range(max = 50))]
    pub wait: u32,
}

/// `create_benchmark`: from a dataset split or a labeling job's labeled images.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateBenchmarkArgs {
    /// Name.
    pub name: String,
    /// A dataset (with split, default val).
    #[serde(default)]
    pub dataset_id: Option<String>,
    /// The dataset's split.
    #[serde(default)]
    pub split: Option<Split>,
    /// Or a labeling job's labeled images.
    #[serde(default)]
    pub label_job_id: Option<String>,
    /// What it stands for (camera, hard cases).
    #[serde(default)]
    pub description: String,
    /// At most this many images (1 to 5000, default 500).
    #[serde(default)]
    pub max_images: Option<u32>,
    /// Seed of the sample and the conditions.
    #[serde(default)]
    pub seed: u64,
    /// The plan (default: bench_catalog's default_plan).
    #[serde(default)]
    pub plan: Option<BenchPlanArgs>,
}

/// A plan: the conditions to run (bench_catalog lists them).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BenchPlanArgs {
    /// Each at most once; [] runs the unchanged images only.
    pub conditions: Vec<BenchPlanConditionArgs>,
}

/// One condition of a plan.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BenchPlanConditionArgs {
    /// Condition id.
    pub condition: String,
    /// 1 to 5 distinct severities in 1..=5, ascending.
    pub severities: Vec<u8>,
    /// Weight in the robustness axis (0 to 10, default 1).
    #[serde(default)]
    pub weight: Option<f32>,
}

impl From<BenchPlanArgs> for BenchPlan {
    fn from(p: BenchPlanArgs) -> BenchPlan {
        let conditions = p
            .conditions
            .into_iter()
            .map(|c| BenchPlanCondition {
                condition: c.condition,
                severities: c.severities,
                weight: c.weight.unwrap_or(1.0),
                why: None,
            })
            .collect();
        BenchPlan { conditions, rationale: None, designed_by: None }
    }
}

/// `start_bench_entry`: a model of the store, or a file of predictions made elsewhere.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StartBenchEntryArgs {
    /// Benchmark id.
    pub benchmark_id: String,
    /// A model (list_models).
    #[serde(default)]
    pub model_id: Option<String>,
    /// Or a JSON Lines file of {image, condition?, severity?, boxes: [{class, score, x1, y1, x2, y2}]}.
    #[serde(default)]
    pub predictions_file: Option<String>,
    /// Leaderboard name (required with predictions_file).
    #[serde(default)]
    pub name: Option<String>,
    /// model_id: the first this many images only.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// model_id: run the plan's conditions (default true).
    #[serde(default)]
    pub conditions: Option<bool>,
    /// predictions_file: false without confidences (default true).
    #[serde(default)]
    pub scored: Option<bool>,
}
