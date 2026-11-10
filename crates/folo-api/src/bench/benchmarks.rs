//! Benchmarks: their plans, sources and images, and the requests that make, change and design them.

use serde::{Deserialize, Serialize};

use crate::{BBox, LabelAgentId, LabelEffort, LabelUsage, Split};
#[cfg(doc)]
use crate::{BenchCatalog, BenchCondition};

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

pub(super) fn one() -> f32 {
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
