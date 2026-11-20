//! Training runs (docs/API.md, *Training runs*): the training configuration, a run, what the trainer resolved
//! it to, and the training queue.

use serde::{Deserialize, Serialize};

use crate::{Arch, Compute, RunCloud};

/// `GET /api/runs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunList {
    /// Runs, newest first.
    pub runs: Vec<Run>,
}

/// Training configuration (`POST /api/runs` body). Every field except `dataset_id` has a default.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainConfig {
    /// Run name; empty means `<dataset name>-<arch suffix>` (e.g. `shapes-1k-n`).
    #[serde(default)]
    pub name: String,
    /// Dataset to train on.
    pub dataset_id: String,
    /// Architecture.
    #[serde(default)]
    pub arch: Arch,
    /// Epochs.
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    /// Batch size.
    #[serde(default = "default_batch")]
    pub batch: u32,
    /// Square input size.
    #[serde(default = "default_imgsz")]
    pub imgsz: u32,
    /// `auto`, `cpu` or `gpu:<ordinal>`.
    #[serde(default = "default_device")]
    pub device: String,
    /// Optimizer.
    #[serde(default)]
    pub optimizer: Optimizer,
    /// Initial learning rate.
    #[serde(default = "default_lr0")]
    pub lr0: f32,
    /// Model id to start from (fine-tune), or `null` for random init.
    #[serde(default)]
    pub weights: Option<String>,
    /// Augmentation.
    #[serde(default)]
    pub augment: Augment,
    /// RNG seed.
    #[serde(default)]
    pub seed: u64,
    /// Mixed-precision training (f16 on GPU tensor cores, docs/MODEL.md §8.8). Needs a GPU whose context has tensor
    /// cores (`folo serve --tensor-cores`); otherwise the run fails with `not_supported`.
    #[serde(default)]
    pub amp: bool,
    /// Detection head (docs/MODEL.md §6.6): `o2m` (one-to-many, NMS at inference) or `o2o` (adds the NMS-free
    /// one-to-one branch, trained with dual assignment; the model predicts end to end without NMS).
    #[serde(default)]
    pub head: Head,
    /// Train on a cloud connection instead of this machine (docs/CLOUD.md); `null` or missing trains here, on
    /// `device`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<Compute>,
}

/// The detection head of a model (docs/MODEL.md §6.6), `TrainConfig.head`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum Head {
    /// One-to-many head; predictions go through NMS.
    #[default]
    O2m,
    /// One-to-many plus one-to-one (NMS-free) head, trained with dual assignment; predicts with the one-to-one
    /// branch and no NMS.
    O2o,
}

impl Head {
    /// `"o2m"` or `"o2o"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Head::O2m => "o2m",
            Head::O2o => "o2o",
        }
    }
}

fn default_epochs() -> u32 {
    100
}
fn default_batch() -> u32 {
    16
}
pub(crate) fn default_imgsz() -> u32 {
    640
}
fn default_device() -> String {
    "auto".into()
}
fn default_lr0() -> f32 {
    0.01
}

/// Optimizer choice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum Optimizer {
    /// MuSGD at lr0 0.01 for every run (`auto` rule 5, docs/MODEL.md §8.2).
    #[default]
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW.
    Adamw,
    /// SGD plus a Muon (orthogonalised momentum) update on conv weights (docs/MODEL.md §8.2).
    Musgd,
    /// A value this build does not know (from a newer server or client): decodes instead of failing the whole
    /// record, and is refused where a run would train with it (`invalid_config`). Never sent; not in the schema.
    #[serde(other)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    Unknown,
}

/// Augmentation settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default)]
pub struct Augment {
    /// Mosaic probability.
    pub mosaic: f32,
    /// Mixup probability.
    pub mixup: f32,
    /// HSV jitter gains `[h, s, v]`.
    pub hsv: [f32; 3],
    /// Horizontal flip probability.
    pub fliplr: f32,
    /// Scale jitter (`±` fraction).
    pub scale: f32,
    /// Translation jitter (fraction of the image).
    pub translate: f32,
    /// Disable mosaic for the last N epochs.
    pub close_mosaic: u32,
}

impl Default for Augment {
    fn default() -> Self {
        Augment {
            mosaic: 1.0,
            mixup: 0.0,
            hsv: [0.015, 0.7, 0.4],
            fliplr: 0.5,
            scale: 0.5,
            translate: 0.1,
            close_mosaic: 10,
        }
    }
}

/// Run lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    /// Waiting for a device.
    Queued,
    /// Training.
    Running,
    /// Stop requested; finishing the current step and saving a checkpoint.
    Stopping,
    /// Stopped by the user; the last checkpoint is kept.
    Stopped,
    /// Finished all epochs.
    Completed,
    /// Failed; see `error`.
    Failed,
}

impl RunStatus {
    /// True for `queued`, `running` and `stopping`: the run may still change.
    pub fn is_active(self) -> bool {
        matches!(self, RunStatus::Queued | RunStatus::Running | RunStatus::Stopping)
    }
}

/// A training run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Run {
    /// Opaque id, e.g. `run_7d1e`.
    pub id: String,
    /// Name (from the config).
    pub name: String,
    /// Lifecycle state.
    pub status: RunStatus,
    /// The configuration it was started with (defaults filled in).
    pub config: TrainConfig,
    /// RFC 3339 creation time.
    pub created: String,
    /// When training started, if it has.
    pub started: Option<String>,
    /// When it reached `stopped`, `completed` or `failed`; for a run whose server was killed under it (`stopped`,
    /// `interrupted`), when it last reported (a step, an epoch, a log line; recorded at least every 10 s while those
    /// come): a run killed during a long phase that reports nothing (a long validation pass) ends at its last
    /// report, before that phase.
    pub finished: Option<String>,
    /// Seconds a resumed run spent not training since `started`: from each end (stop, failure) until it trained
    /// again, summed over its resumes; while it waits in the queue after a resume, the wait so far is included (and
    /// a resumed run that ends there keeps it). Its training time is `finished` (or now, while it is active) minus
    /// `started` minus this, so it does not grow while the run waits. Absent for a run never resumed, and from servers
    /// that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_s: Option<f64>,
    /// Current epoch, 1-based; 0 before the first step. Once ended: the last epoch it finished (`completed`,
    /// `stopped`), or the epoch it failed in (`failed`).
    pub epoch: u32,
    /// Total epochs.
    pub epochs: u32,
    /// Global optimizer steps completed (over all epochs).
    pub step: u64,
    /// Optimizer steps per epoch.
    pub steps_per_epoch: u64,
    /// Estimated seconds remaining, `null` when not running.
    pub eta_s: Option<f64>,
    /// Best epoch so far by `map50_95`.
    pub best: Option<BestEpoch>,
    /// Device description, e.g. `NVIDIA GeForce RTX 4080 SUPER (vulkan)`.
    pub device: String,
    /// Model produced by this run (best weights), once one exists.
    pub model_id: Option<String>,
    /// Error message when `failed`.
    pub error: Option<String>,
    /// The stable error code (`folo_api::codes`) when `failed`, e.g. `not_supported` or `training_failed`; absent
    /// otherwise and from servers that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    /// What the trainer resolved `config` to (optimizer, effective LR, ...); `null` until the trainer is built.
    #[serde(default)]
    pub resolved: Option<ResolvedTrain>,
    /// 1-based place in the training queue while the run waits there (`1` starts next), else `null`.
    #[serde(default)]
    pub queue_position: Option<u32>,
    /// Where a cloud run (`config.compute`) executes and how far it is; absent for runs on this machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cloud: Option<RunCloud>,
    /// The dataset as it was when the run was created, so the run still names it (and its classes) once the dataset
    /// is removed; absent from servers that predate the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<RunDataset>,
}

/// `Run.dataset`: the name and class names of the dataset a run trains on, recorded when it was created.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunDataset {
    /// The dataset's name.
    pub name: String,
    /// Its class names, in class order; empty for runs created before they were recorded.
    #[serde(default)]
    pub classes: Vec<String>,
}

/// [`Run::resolved`]: the settings a run actually trains with, after `optimizer: auto` and the batch scaling of
/// docs/MODEL.md §8.2 and §8.4. The run folder's `config.json` holds the same numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ResolvedTrain {
    /// The optimizer used, never `auto`.
    pub optimizer: ResolvedOptimizer,
    /// Base learning rate of every parameter group: what `auto` picked, else `TrainConfig.lr0`.
    pub lr0: f64,
    /// Final LR fraction: the last optimizer step runs at `lr0 · lrf`.
    pub lrf: f64,
    /// Cosine (`true`) or linear (`false`) LR decay.
    pub cos_lr: bool,
    /// SGD momentum, or AdamW β1, after warmup.
    pub momentum: f64,
    /// AdamW's `[β1, β2]`; `null` for SGD.
    pub betas: Option<[f64; 2]>,
    /// The weight decay of the decay group after the batch scaling `weight_decay · batch · accumulate / nbs`.
    pub weight_decay: f64,
    /// Images per forward / backward pass (`TrainConfig.batch`).
    pub batch: u32,
    /// Batches whose gradients are summed per optimizer step: `max(round(nbs / batch), 1)`.
    pub accumulate: u32,
    /// The nominal batch `accumulate` aims at.
    pub nbs: u32,
    /// Optimizer steps per epoch: `ceil(batches per epoch / accumulate)`.
    pub optimizer_steps_per_epoch: u64,
    /// Warmup length in optimizer steps (from `warmup_epochs`, at most every epoch but the last; runs made before
    /// `warmup_floor` existed warm up for at least 100 steps, at most 30% of a short run).
    pub warmup_iters: u64,
    /// Warmup length as configured, in epochs.
    pub warmup_epochs: f64,
    /// Momentum / β1 at the start of warmup.
    pub warmup_momentum: f64,
    /// The bias group's LR at the start of warmup (`0.1`; `0` under `auto` rules 1 to 3).
    pub warmup_bias_lr: f64,
    /// Close-mosaic epochs the run trains with: `TrainConfig.augment.close_mosaic` capped at half the run
    /// (docs/MODEL.md §8.4), so mosaic is off for the last `close_mosaic` epochs. Left out by servers older than the
    /// field and for runs recorded before it whose `config.json` lacks the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub close_mosaic: Option<u32>,
    /// The EMA's asymptotic decay (docs/MODEL.md §8.5, `folo.ema`): 0.9999, or less for a fine-tune whose window is a
    /// share of its optimizer steps (`TrainConfig.ema_run_fraction`). Left out for a run without an EMA and by servers
    /// older than the field.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::json::exact_opt_f64"
    )]
    pub ema_decay: Option<f64>,
    /// The EMA's ramp time constant in optimizer steps (docs/MODEL.md §8.5, `folo.ema`): 2000, or a fifth of a
    /// fine-tune's window. Left out when `ema_decay` is.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::json::exact_opt_f64"
    )]
    pub ema_tau: Option<f64>,
}

/// [`ResolvedTrain::optimizer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum ResolvedOptimizer {
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW (decoupled weight decay).
    Adamw,
    /// MuSGD: SGD plus a Muon update on conv weights.
    Musgd,
    /// An optimizer this build does not know (a run of a newer server): decodes instead of failing the whole
    /// `Run`. Never sent; not in the schema.
    #[serde(other)]
    #[cfg_attr(feature = "schema", schemars(skip))]
    Unknown,
}

/// `POST /api/runs/{id}/queue`: moves a waiting run in the training queue. Exactly one of `action` and `position`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct QueueRequest {
    /// A relative move, or `start`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<QueueAction>,
    /// The 1-based place to move to (`1` starts next; larger than the queue means its end).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<u32>,
}

/// [`QueueRequest::action`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum QueueAction {
    /// To the front of the queue: it starts next.
    Top,
    /// One place forward.
    Up,
    /// One place back.
    Down,
    /// To the back of the queue.
    Bottom,
    /// Start it at once when nothing is training, else move it to the front; never preempts the active run.
    Start,
}

/// SSE `queue` payload: a waiting run's place in the training queue, sent to every waiting run after each change
/// of the queue (a run queued, moved, taken by the worker, stopped or deleted).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct QueueEvent {
    /// 1-based place in the queue (`Run.queue_position`).
    pub position: u32,
    /// Runs waiting in the queue, this one included.
    pub waiting: u32,
}

/// The best epoch of a run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BestEpoch {
    /// Epoch number.
    pub epoch: u32,
    /// Its validation mAP@0.5:0.95.
    pub map50_95: f32,
}
