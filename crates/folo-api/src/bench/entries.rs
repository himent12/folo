//! Entries: who runs a benchmark (a model, a coding agent or external predictions), how far an entry is, and
//! its score.

use serde::{Deserialize, Serialize};

use super::benchmarks::one;
#[cfg(doc)]
use crate::BenchImage;
use crate::{BenchAxis, LabelAgentId, LabelEffort, LabelUsage};

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
    /// RFC 3339 time it was created, to the millisecond (`2026-11-10T07:22:50.123Z`) and later than every entry
    /// the benchmark already had: sorting by it is the order the entries were made in.
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
