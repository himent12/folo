//! Self-improvement loops: a model labels a pool, a student trains on those labels, the benchmark grades it.

use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::BenchSelfLabeling;
use crate::BenchSource;

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
