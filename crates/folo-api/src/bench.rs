//! Vision benchmarks (docs/API.md, *Vision benchmarks*; docs/BENCH.md defines every number): a frozen set of
//! labeled images plus a plan of stress conditions, the entries that ran it (FOLO models, coding agents, or
//! predictions made anywhere else) with their scores, and self-improvement loops. Everything under `/api/bench`.
//!
//! Not the speed probe of `/api/system/benchmark` ([`crate::BenchmarkRequest`]): that one times a model, this
//! one grades what a model sees.

use serde::{Deserialize, Serialize};

use crate::{BBox, LabelAgentId, LabelEffort, LabelUsage, Split};

/// [`SystemResponse::features`](crate::SystemResponse::features): the `/api/bench` endpoints exist.
pub const FEATURE_VISION_BENCH: &str = "vision_bench";

/// Severities of a condition run from 1 (barely visible) to this (severe, still solvable by a person).
pub const BENCH_SEVERITIES: u8 = 5;

// ---------------------------------------------------------------------------------------------------------------
// Catalog: conditions and axes
// ---------------------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------------------------------------------

/// Which conditions a benchmark runs, how hard and how much each counts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchPlan {
    /// The conditions, each at most once. An empty plan runs the unchanged images only (no robustness axis).
    pub conditions: Vec<BenchPlanCondition>,
    /// Why this plan fits the benchmark, when someone said (an agent's design does).
    #[serde(default)]
    pub rationale: Option<String>,
    /// The agent that proposed the plan; `null` for the default plan and for one written by hand.
    #[serde(default)]
    pub designed_by: Option<BenchDesigner>,
}

/// One condition of a [`BenchPlan`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchPlanCondition {
    /// A [`BenchCondition::id`].
    pub condition: String,
    /// The severities to run: 1 to 5 distinct values in `1..=5`, ascending.
    pub severities: Vec<u8>,
    /// How much the condition counts in the robustness axis, relative to the others (`0 < weight <= 10`).
    #[serde(default = "one")]
    pub weight: f32,
    /// Why it is in the plan, when someone said.
    #[serde(default)]
    pub why: Option<String>,
}

fn one() -> f32 {
    1.0
}

/// The agent behind a designed [`BenchPlan`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchDesigner {
    /// The agent.
    pub agent: LabelAgentId,
    /// The model it was asked for; `null` means the CLI's default.
    pub model: Option<String>,
}

/// Where a set of images comes from: a benchmark's images with their truth, or a self-improvement loop's pool
/// (whose labels, if it has any, are never read).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchSource {
    /// A split of a registered dataset.
    Dataset {
        /// The dataset.
        dataset_id: String,
        /// The split (a benchmark defaults to `val`, a pool to `train`).
        #[serde(default)]
        split: Option<Split>,
    },
    /// The images of a labeling job (docs/LABELING.md). A benchmark takes its labeled images, boxes included.
    LabelJob {
        /// The job.
        job_id: String,
    },
}

/// `GET /api/bench/benchmarks`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchList {
    /// Benchmarks, newest first.
    pub benchmarks: Vec<Bench>,
}

/// A benchmark: images with their true boxes, copied in and frozen, and the plan of conditions to run them
/// under. Its images and truth never change; its plan can change only while it has no `done` and no active entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Bench {
    /// Opaque id, e.g. `bn_3f9a0c1e5b7d` (random, never reused).
    pub id: String,
    /// Display name.
    pub name: String,
    /// What the benchmark stands for, in the user's words: where the camera is, what makes the job hard (may be
    /// empty). An agent that designs the plan reads it.
    pub description: String,
    /// Class names; a box's `class` indexes this list.
    pub classes: Vec<String>,
    /// RFC 3339 creation time.
    pub created: String,
    /// Where the images came from.
    pub source: BenchSource,
    /// The source's display name when the benchmark was made (the dataset's or the labeling job's).
    pub source_name: String,
    /// Images.
    pub images: u64,
    /// True boxes over all images.
    pub boxes: u64,
    /// True boxes per class, aligned with `classes`.
    pub class_boxes: Vec<u64>,
    /// The conditions to run.
    pub plan: BenchPlan,
    /// Seed of everything random in the benchmark (which images were sampled, where noise falls); at most
    /// 2^53 − 1.
    pub seed: u64,
    /// Identifies the benchmark's content: a hash of its images, truth, plan and seed and of the engine that grades
    /// (`folo_bench::ENGINE_VERSION`), as hex. Two entries are comparable when their fingerprints agree.
    pub fingerprint: String,
    /// Entries, whatever their status.
    pub entries: u64,
    /// The leaderboard's first entry when it is finished and has this fingerprint: the complete entry with the
    /// highest score (the older one of equal scores), else the partial one; `null` before the first.
    pub best: Option<BenchBest>,
    /// True while an entry of the benchmark is queued, running or stopping, or a loop of it is running or
    /// stopping (deleting it is then `409 bench_active`). A `bench` event is sent when it changes.
    #[serde(default)]
    pub active: bool,
}

/// The leading entry of a [`Bench`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchBest {
    /// The entry.
    pub entry_id: String,
    /// Its name.
    pub name: String,
    /// Its score, `0..=100`.
    pub score: f32,
    /// Its `score.partial`: the score compares only loosely with complete entries.
    #[serde(default)]
    pub partial: bool,
}

/// `POST /api/bench/benchmarks`: a new benchmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CreateBench {
    /// Display name.
    pub name: String,
    /// What the benchmark stands for (see [`Bench::description`]).
    #[serde(default)]
    pub description: String,
    /// The images and their truth.
    pub source: BenchSource,
    /// At most this many images (1 to 5000, default 500), sampled with `seed` when the source has more.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// Seed (default 0), at most 2^53 − 1 (the largest integer a browser holds exactly).
    #[serde(default)]
    pub seed: u64,
    /// The plan; absent uses [`BenchCatalog::default_plan`].
    #[serde(default)]
    pub plan: Option<BenchPlan>,
}

/// `PUT /api/bench/benchmarks/{id}`: the fields to change (absent ones stay).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UpdateBench {
    /// New display name.
    #[serde(default)]
    pub name: Option<String>,
    /// New description.
    #[serde(default)]
    pub description: Option<String>,
    /// New plan. Only while the benchmark has no `done` and no active entry (`409 bench_frozen` otherwise; failed
    /// and stopped entries do not count).
    #[serde(default)]
    pub plan: Option<BenchPlan>,
}

/// `POST /api/bench/benchmarks/{id}/design`: ask a coding agent to propose the plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DesignBench {
    /// The agent to ask.
    pub agent: LabelAgentId,
    /// Its model; absent or `null` uses the agent's default.
    #[serde(default)]
    pub model: Option<String>,
    /// Reasoning effort; absent or `null` uses the agent's default.
    #[serde(default)]
    pub effort: Option<LabelEffort>,
    /// Anything to add to the benchmark's description for this request.
    #[serde(default)]
    pub note: String,
}

/// Answer of `POST /api/bench/benchmarks/{id}/design`: a proposal, not saved until it is `PUT`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchDesign {
    /// The proposed plan, valid as it is (`designed_by` and `rationale` set).
    pub plan: BenchPlan,
    /// What the call used.
    pub usage: LabelUsage,
}

/// One image of a [`Bench`] with its truth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchImage {
    /// Opaque id, unique within the benchmark.
    pub id: String,
    /// File name, unique within the benchmark.
    pub file: String,
    /// Width in pixels (after EXIF orientation); the same under every condition.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The true boxes, pixel `xyxy`, under the condition asked for (a geometric condition moves them).
    pub boxes: Vec<BBox>,
}

/// `GET /api/bench/benchmarks/{id}/images`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchImagePage {
    /// All images of the benchmark (before `offset` / `limit`).
    pub total: u64,
    /// This page, in the benchmark's order.
    pub images: Vec<BenchImage>,
}

// ---------------------------------------------------------------------------------------------------------------
// Entries
// ---------------------------------------------------------------------------------------------------------------

/// A detection: a box, its class and how sure the contestant is.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchDetection {
    /// Class index into the benchmark's classes.
    pub class: u32,
    /// Confidence in `0..=1`; 1 when absent, and for a contestant that gives none.
    #[serde(default = "one")]
    pub score: f32,
    /// Left.
    pub x1: f32,
    /// Top.
    pub y1: f32,
    /// Right.
    pub x2: f32,
    /// Bottom.
    pub y2: f32,
}

/// The detections of one image under one condition, as an external contestant submits them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchPrediction {
    /// A [`BenchImage::id`].
    pub image: String,
    /// The condition the image was fetched under; absent or `null` for the unchanged image.
    #[serde(default)]
    pub condition: Option<String>,
    /// Its severity (required with `condition`).
    #[serde(default)]
    pub severity: Option<u8>,
    /// What the model found, in pixels of the image as it was served.
    pub boxes: Vec<BenchDetection>,
}

/// `POST /api/bench/benchmarks/{id}/entries/{entry_id}/predictions`: a part of an open external entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchPredictions {
    /// Records not sent before, in any order (see [`BenchContestantSpec::External`]).
    pub predictions: Vec<BenchPrediction>,
}

/// Who a new entry is ([`StartBenchEntry::contestant`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchContestantSpec {
    /// A model of this server (`GET /api/models`), run by FOLO.
    Model {
        /// The model.
        model_id: String,
    },
    /// A coding agent on the server's machine (`GET /api/label/agents`), asked for boxes as agent labeling asks.
    Agent {
        /// The agent.
        agent: LabelAgentId,
        /// Its model; absent or `null` uses the agent's default.
        #[serde(default)]
        model: Option<String>,
        /// Reasoning effort; absent or `null` uses the agent's default.
        #[serde(default)]
        effort: Option<LabelEffort>,
    },
    /// Any other vision model: its predictions on the benchmark's images, made wherever it runs.
    External {
        /// False when the model gives no confidences (every `score` is then taken as 1 and the axes that need
        /// confidences are not measured). Default true.
        #[serde(default = "yes")]
        scored: bool,
        /// False (the default): `predictions` is the whole entry, scored in the request. True: the entry is
        /// created `running` to be filled in parts (`POST …/entries/{entry_id}/predictions`) and scored by
        /// `POST …/entries/{entry_id}/finish`; `predictions` here, if any, is its first part.
        #[serde(default)]
        open: bool,
        /// One record per image and condition it ran (at most 1000 boxes each). Every unchanged image must be
        /// answered for (an image with nothing found has an empty `boxes`); a condition of the plan is either
        /// answered for every image at every planned severity or left out entirely.
        #[serde(default)]
        predictions: Vec<BenchPrediction>,
    },
}

fn yes() -> bool {
    true
}

/// `POST /api/bench/benchmarks/{id}/entries`: run a contestant on the benchmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StartBenchEntry {
    /// Who runs.
    pub contestant: BenchContestantSpec,
    /// The entry's name on the leaderboard; absent uses the model's name, or the agent's and its model's.
    /// Required for an external contestant.
    #[serde(default)]
    pub name: Option<String>,
    /// Run only the first this many images of the benchmark (its order is a seeded shuffle, so a prefix is a
    /// fair sample). Absent: every image for a model, 48 for an agent. Not for an external contestant.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// Whether to run the plan's conditions, not only the unchanged images. Absent: true for a model, false for
    /// an agent (every condition is another call per image). Not for an external contestant.
    #[serde(default)]
    pub conditions: Option<bool>,
}

/// Who an entry is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BenchContestant {
    /// A model of this server.
    Model {
        /// The model (it may have been deleted since).
        model_id: String,
        /// Its architecture, e.g. `folo-n`.
        arch: String,
        /// Its parameter count.
        params: u64,
        /// Its inference size in pixels.
        imgsz: u32,
    },
    /// A coding agent.
    Agent {
        /// The agent.
        agent: LabelAgentId,
        /// The model it was asked for; `null` means the CLI's default.
        model: Option<String>,
        /// Reasoning effort.
        effort: LabelEffort,
    },
    /// Predictions made elsewhere.
    External,
}

/// Where a [`BenchEntry`] stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum BenchEntryStatus {
    /// Waiting for the entries ahead of it.
    Queued,
    /// Running the benchmark.
    Running,
    /// A stop was requested.
    Stopping,
    /// Finished and scored.
    Done,
    /// Ended on an error; `error` says why.
    Failed,
    /// Stopped on request; nothing was scored.
    Stopped,
}

impl BenchEntryStatus {
    /// True while the entry waits or runs.
    pub fn is_active(self) -> bool {
        matches!(self, BenchEntryStatus::Queued | BenchEntryStatus::Running | BenchEntryStatus::Stopping)
    }
}

/// How far a running entry is.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchProgress {
    /// Images answered so far, over all conditions.
    pub done: u64,
    /// Images to answer: the entry's images times one plus its conditions' severities.
    pub total: u64,
    /// The condition being run; `null` for the unchanged images.
    pub condition: Option<String>,
    /// Its severity.
    pub severity: Option<u8>,
}

/// A contestant's run of a benchmark, on its leaderboard.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchEntry {
    /// Opaque id, e.g. `be_91c2`.
    pub id: String,
    /// The benchmark.
    pub bench_id: String,
    /// Name on the leaderboard.
    pub name: String,
    /// Who ran.
    pub contestant: BenchContestant,
    /// Where it stands.
    pub status: BenchEntryStatus,
    /// RFC 3339 time it was created.
    pub created: String,
    /// RFC 3339 time it finished, failed or was stopped; `null` until then.
    pub finished: Option<String>,
    /// How far it is.
    pub progress: BenchProgress,
    /// Images it runs: all of the benchmark's, or a prefix ([`StartBenchEntry::max_images`]).
    pub images: u64,
    /// Whether it runs the plan's conditions.
    pub conditions: bool,
    /// Whether the contestant gives confidences.
    pub scored: bool,
    /// The benchmark's fingerprint when the entry ran.
    pub fingerprint: String,
    /// Classes of the benchmark the contestant does not know (a model matched by class name); its AP on them is
    /// 0.
    pub missing_classes: Vec<String>,
    /// The score; `null` until `done`.
    pub score: Option<BenchScore>,
    /// mAP@0.5:0.95 on the unchanged images, repeated here for lists; `null` until `done`.
    pub map50_95: Option<f32>,
    /// The 95% interval of `map50_95` (`BenchResult.accuracy.ci95`), repeated here for lists; `null` until `done`.
    #[serde(default)]
    pub ci95: Option<[f32; 2]>,
    /// How many of the plan's conditions the entry ran completely (every planned severity, every image it runs;
    /// for an external entry, the conditions its predictions answered); 0 when none, and until `done`.
    #[serde(default)]
    pub conditions_run: u32,
    /// How fast the model ran (a model contestant, once `done`).
    #[serde(default)]
    pub speed: Option<BenchSpeed>,
    /// What the agent's calls used (an agent contestant).
    #[serde(default)]
    pub usage: Option<LabelUsage>,
    /// Why it failed (`status: failed`), else `null`.
    pub error: Option<String>,
    /// How to fix `error`, when there is something useful to say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// How fast a model contestant ran on the server. Not part of the score: it depends on the machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchSpeed {
    /// The device, e.g. `gpu:0`.
    pub device: String,
    /// Wall time per image on the unchanged images, decode to boxes, in milliseconds, in batches and without the
    /// first one (which pays for loading the model): a throughput, not the latency of a single image.
    pub ms_per_image: f32,
}

/// The score of an entry: one number and what it is made of.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchScore {
    /// `0..=100`: the weighted mean of the measured axes, times 100.
    pub total: f32,
    /// True when an axis was not measured (no confidences, conditions not run) or the entry ran a part of the
    /// images: the total is then over what was measured and compares loosely with complete entries.
    pub partial: bool,
    /// Every axis in [`BenchAxis::ALL`] order.
    pub axes: Vec<BenchAxisScore>,
}

/// One axis of a [`BenchScore`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchAxisScore {
    /// The axis.
    pub axis: BenchAxis,
    /// `0..=1`; `null` when it was not measured.
    pub value: Option<f32>,
    /// Its share of the score.
    pub weight: f32,
}

/// `GET /api/bench/benchmarks/{id}/entries`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchEntryList {
    /// The leaderboard: finished complete entries with the benchmark's fingerprint by score, highest first; then
    /// finished partial ones by score; then finished entries with another fingerprint by score; equal scores keep
    /// the older entry first; then the unfinished ones (and failed and stopped), newest first.
    pub entries: Vec<BenchEntry>,
}

// ---------------------------------------------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------------------------------------------

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
    /// The confidence with the best class-mean F1.
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
    /// The detections at or above the entry's `best_conf`, highest confidence first.
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
    /// This page.
    pub images: Vec<BenchEntryImage>,
}

// ---------------------------------------------------------------------------------------------------------------
// Self-improvement loops
// ---------------------------------------------------------------------------------------------------------------

/// `POST /api/bench/benchmarks/{id}/loops`: test whether a model can teach its successor. Each generation, the
/// current model labels the pool by itself, a student trains on those labels alone, and the benchmark grades the
/// student, which becomes the next generation's teacher. No true label enters training or model selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StartBenchLoop {
    /// Display name; absent names it after the teacher.
    #[serde(default)]
    pub name: Option<String>,
    /// The first teacher: a model of this server.
    pub teacher: String,
    /// The unlabeled images the teachers label. Its own labels are never read; images that are also in the
    /// benchmark are left out.
    pub pool: BenchSource,
    /// Generations to run, 1 to 5 (default 1).
    #[serde(default)]
    pub generations: Option<u32>,
    /// Keep a teacher's detections from this confidence up; absent uses the threshold its own benchmark entry
    /// found ([`BenchSelfLabeling::threshold`]). A teacher whose entry has none ends the loop before labeling: a
    /// loop never guesses a confidence.
    #[serde(default)]
    pub min_confidence: Option<f32>,
    /// How the students train.
    #[serde(default)]
    pub student: BenchStudent,
}

/// How the students of a loop train: the fields of `TrainConfig` a loop lets you choose.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchStudent {
    /// Architecture, e.g. `folo-s`; absent uses the first teacher's. A larger one is "the next bigger model".
    #[serde(default)]
    pub arch: Option<String>,
    /// Epochs per generation (default 30).
    #[serde(default)]
    pub epochs: Option<u32>,
    /// Image size (default: the first teacher's).
    #[serde(default)]
    pub imgsz: Option<u32>,
    /// Batch size (default: `TrainConfig`'s).
    #[serde(default)]
    pub batch: Option<u32>,
    /// Device (default `auto`).
    #[serde(default)]
    pub device: Option<String>,
    /// Starting weights as `TrainConfig.weights` takes them; absent trains from scratch.
    #[serde(default)]
    pub weights: Option<String>,
    /// Seed (default 0), at most 2^53 − 1.
    #[serde(default)]
    pub seed: u64,
}

/// Where a [`BenchLoop`] stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum BenchLoopStatus {
    /// Working through its generations.
    Running,
    /// A stop was requested.
    Stopping,
    /// Every generation ran.
    Done,
    /// Ended on an error; `error` says why.
    Failed,
    /// Stopped on request.
    Stopped,
}

impl BenchLoopStatus {
    /// True while the loop runs.
    pub fn is_active(self) -> bool {
        matches!(self, BenchLoopStatus::Running | BenchLoopStatus::Stopping)
    }
}

/// What a running loop is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum BenchLoopPhase {
    /// Running the benchmark on the generation's model.
    Benchmarking,
    /// The teacher is labeling the pool.
    Labeling,
    /// The student is training on those labels.
    Training,
}

/// How a loop came out: the last generation against the first teacher, on mAP@0.5:0.95 of the unchanged images.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum BenchLoopVerdict {
    /// Above the teacher's 95% interval: the model's own labels made a better model.
    Improved,
    /// Inside it: no measurable change.
    Held,
    /// Below it.
    Regressed,
}

/// A self-improvement loop of a benchmark.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchLoop {
    /// Opaque id, e.g. `bl_07d1`.
    pub id: String,
    /// The benchmark.
    pub bench_id: String,
    /// Display name.
    pub name: String,
    /// Where it stands.
    pub status: BenchLoopStatus,
    /// What it is doing; `null` once it has ended.
    pub phase: Option<BenchLoopPhase>,
    /// RFC 3339 time it was started.
    pub created: String,
    /// RFC 3339 time it ended; `null` while it runs.
    pub finished: Option<String>,
    /// The request, defaults filled in.
    pub config: StartBenchLoop,
    /// Generation 0 (the first teacher), then one per student, as far as the loop has come.
    pub generations: Vec<BenchGeneration>,
    /// How it came out; `null` until `done`.
    pub verdict: Option<BenchLoopVerdict>,
    /// Why it failed (`status: failed`), else `null`.
    pub error: Option<String>,
    /// How to fix `error`, when there is something useful to say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// One generation of a [`BenchLoop`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchGeneration {
    /// 0 for the first teacher, then 1, 2, ...
    pub generation: u32,
    /// The labels its teacher made for it; `null` for generation 0 and until they are made.
    pub labels: Option<BenchLabels>,
    /// The training run of this generation's model; `null` for generation 0 and until it starts.
    pub run_id: Option<String>,
    /// The generation's model; `null` until its training has finished.
    pub model_id: Option<String>,
    /// The model's name.
    pub model_name: Option<String>,
    /// Its benchmark entry; `null` until it is started.
    pub entry_id: Option<String>,
    /// Its score; `null` until the entry is done.
    pub score: Option<f32>,
    /// Its mAP@0.5:0.95 on the unchanged images.
    pub map50_95: Option<f32>,
    /// That number's 95% interval.
    pub ci95: Option<[f32; 2]>,
}

/// The labels a teacher made of a loop's pool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchLabels {
    /// The confidence from which detections were kept.
    pub threshold: f32,
    /// Images of the pool the teacher looked at.
    pub images: u64,
    /// Of those, images it kept at least one box on: the student's dataset. An image it kept nothing on is left
    /// out (its silence is not a label).
    pub labeled_images: u64,
    /// Boxes kept.
    pub boxes: u64,
    /// Pool images left out because the benchmark has them.
    pub excluded: u64,
    /// The dataset written from them (train and val both from the pool, both with the teacher's labels).
    pub dataset_id: String,
}

/// `GET /api/bench/benchmarks/{id}/loops`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchLoopList {
    /// Loops, newest first.
    pub loops: Vec<BenchLoop>,
}

// ---------------------------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------------------------

/// SSE `removed` payload of a benchmark's stream: an entry or a loop was deleted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BenchRemoved {
    /// The deleted entry's or loop's id.
    pub id: String,
}

/// One event of `GET /api/bench/benchmarks/{id}/events`. On the wire the SSE `event:` field is
/// [`BenchEvent::name`] and `data:` is the JSON of the payload alone.
#[derive(Clone, Debug, PartialEq)]
pub enum BenchEvent {
    /// `bench`: the benchmark after a change (name, plan, entry count, best entry).
    Bench(Box<Bench>),
    /// `entry`: an entry after a change (created, progress, done, failed).
    Entry(Box<BenchEntry>),
    /// `loop`: a loop after a change.
    Loop(Box<BenchLoop>),
    /// `removed`: an entry or a loop was deleted.
    Removed(BenchRemoved),
}

impl BenchEvent {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            BenchEvent::Bench(_) => "bench",
            BenchEvent::Entry(_) => "entry",
            BenchEvent::Loop(_) => "loop",
            BenchEvent::Removed(_) => "removed",
        }
    }

    /// The JSON `data:` payload.
    pub fn data_json(&self) -> String {
        let r = match self {
            BenchEvent::Bench(p) => serde_json::to_string(p),
            BenchEvent::Entry(p) => serde_json::to_string(p),
            BenchEvent::Loop(p) => serde_json::to_string(p),
            BenchEvent::Removed(p) => serde_json::to_string(p),
        };
        r.expect("event payloads always serialize")
    }

    /// Parses an SSE event from its `event:` name and `data:` JSON. Returns `None` for unknown event names (new
    /// event types may be added; clients should skip them) and an error for malformed data.
    pub fn from_sse(name: &str, data: &str) -> Option<Result<BenchEvent, serde_json::Error>> {
        Some(match name {
            "bench" => serde_json::from_str(data).map(BenchEvent::Bench),
            "entry" => serde_json::from_str(data).map(BenchEvent::Entry),
            "loop" => serde_json::from_str(data).map(BenchEvent::Loop),
            "removed" => serde_json::from_str(data).map(BenchEvent::Removed),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_weights_sum_to_one_and_names_round_trip() {
        let sum: f32 = BenchAxis::ALL.iter().map(|a| a.weight()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "{sum}");
        for a in BenchAxis::ALL {
            assert_eq!(serde_json::to_value(a).unwrap(), a.as_str());
        }
    }

    #[test]
    fn requests_take_their_defaults() {
        let c: CreateBench =
            serde_json::from_str(r#"{"name":"pond","source":{"kind":"dataset","dataset_id":"ds_1"}}"#)
                .unwrap();
        assert_eq!(c.source, BenchSource::Dataset { dataset_id: "ds_1".into(), split: None });
        assert!(c.plan.is_none() && c.max_images.is_none() && c.description.is_empty() && c.seed == 0);

        let p: BenchPlanCondition =
            serde_json::from_str(r#"{"condition":"haze","severities":[1,3]}"#).unwrap();
        assert_eq!((p.weight, p.why), (1.0, None));

        let s: StartBenchEntry = serde_json::from_str(
            r#"{"contestant":{"kind":"external","predictions":[{"image":"1","boxes":[{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}]}]},"name":"yolo"}"#,
        )
        .unwrap();
        let BenchContestantSpec::External { scored, open, predictions } = s.contestant else {
            panic!("external")
        };
        assert!(
            scored && !open && predictions[0].condition.is_none() && predictions[0].boxes[0].score == 1.0
        );
        let s: StartBenchEntry =
            serde_json::from_str(r#"{"contestant":{"kind":"external","open":true},"name":"yolo"}"#).unwrap();
        assert!(
            matches!(s.contestant, BenchContestantSpec::External { open: true, ref predictions, .. } if predictions.is_empty())
        );
        let s: StartBenchEntry =
            serde_json::from_str(r#"{"contestant":{"kind":"agent","agent":"codex"}}"#).unwrap();
        assert!(matches!(s.contestant, BenchContestantSpec::Agent { agent: LabelAgentId::Codex, .. }));
        assert!(serde_json::from_str::<StartBenchEntry>(r#"{"contestant":{"kind":"robot"}}"#).is_err());

        let l: StartBenchLoop =
            serde_json::from_str(r#"{"teacher":"m1","pool":{"kind":"label_job","job_id":"lb_1"}}"#).unwrap();
        assert_eq!(l.student, BenchStudent::default());
        assert!(l.generations.is_none() && l.min_confidence.is_none());
    }

    #[test]
    fn events_round_trip_by_name() {
        let ev = BenchEvent::Removed(BenchRemoved { id: "be_1".into() });
        assert_eq!(ev.name(), "removed");
        assert_eq!(BenchEvent::from_sse(ev.name(), &ev.data_json()).unwrap().unwrap(), ev);
        assert!(BenchEvent::from_sse("later", "{}").is_none());
        assert!(BenchEvent::from_sse("entry", "{").unwrap().is_err());
    }
}
