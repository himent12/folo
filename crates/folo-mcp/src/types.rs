//! Tool arguments and results that are specific to MCP (paging, summaries, the extra fields of `run_status`, ...).
//! Everything else is a `folo-api` type, so the schemas match the HTTP API. See docs/MCP.md.

use std::collections::BTreeMap;

use folo_api::{
    Arch, BBox, BestEpoch, Capabilities, DatasetFormat, DatasetIssue, Epoch, ExportFormat, GpuInfo, Head,
    IssueCounts, IssueSeverity, LogLine, Loss, ModelMetrics, ModelPrecision, ModelSource, Run, RunStatus,
    Split, Splits, Step, TrainConfig, TrainedWith, ValMetrics,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default page size of list tools.
pub const DEFAULT_LIMIT: u32 = 50;
/// Largest page size of list tools.
pub const MAX_LIMIT: u32 = 200;

/// No arguments.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

/// Paging of a list tool.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct PageArgs {
    /// `next_cursor` of the previous page; omit for the first page.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size (the page may be shorter to stay under the result size limit).
    #[serde(default)]
    #[schemars(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

/// `get_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DatasetIdArgs {
    /// Dataset id, e.g. `ds_5a1e`.
    pub dataset_id: String,
}

/// `list_dataset_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListImagesArgs {
    /// Dataset id.
    pub dataset_id: String,
    /// Only this split (all splits when omitted).
    #[serde(default)]
    pub split: Option<Split>,
    /// Only images with at least one box of this class index.
    #[serde(default)]
    pub class: Option<u32>,
    /// Include the ground-truth boxes of each image.
    #[serde(default)]
    pub include_boxes: bool,
    /// `next_cursor` of the previous page; omit for the first page.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size (the page may be shorter to stay under the result size limit).
    #[serde(default)]
    #[schemars(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

/// `register_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegisterArgs {
    /// Display name; defaults to the folder name.
    #[serde(default)]
    pub name: Option<String>,
    /// Dataset folder (absolute, or relative to `--data-dir`).
    pub path: String,
    /// Format; `auto` detects it.
    #[serde(default)]
    pub format: DatasetFormat,
}

/// `create_synthetic_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SyntheticArgs {
    /// Display name.
    pub name: String,
    /// Number of images (split 80 / 20 into train / val).
    #[serde(default = "default_synth_images")]
    #[schemars(range(min = 1, max = 100_000))]
    pub images: u64,
    /// Class names; `circle`, `square`, `triangle` and `star` get those shapes.
    #[serde(default = "default_synth_classes")]
    pub classes: Vec<String>,
    /// Square image side in pixels.
    #[serde(default = "default_imgsz")]
    #[schemars(range(min = 32, max = 4096))]
    pub imgsz: u32,
    /// RNG seed.
    #[serde(default)]
    pub seed: u64,
}

fn default_synth_images() -> u64 {
    1000
}
fn default_synth_classes() -> Vec<String> {
    vec!["circle".into(), "square".into(), "triangle".into()]
}
fn default_imgsz() -> u32 {
    640
}

/// `train`: a `TrainConfig` (docs/API.md) plus `resume_run_id`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TrainArgs {
    /// The training configuration; only `dataset_id` is required (and not even that when resuming).
    #[serde(flatten)]
    pub config: TrainConfig,
    /// Resume this `stopped` or `failed` run (same id) from its last checkpoint instead of starting a new one. The
    /// other fields must then match the run's config or be omitted (`device` may change).
    #[serde(default)]
    pub resume_run_id: Option<String>,
}

/// `train` result.
#[derive(Debug, Serialize, JsonSchema)]
pub struct TrainResult {
    /// The new (or resumed) run.
    pub run_id: String,
    /// Its status (`queued` right after creation or resuming).
    pub status: RunStatus,
}

/// `reorder_run` result.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ReorderResult {
    /// The moved run.
    pub run_id: String,
    /// Its status (`queued`, or `running` when `start` found the slot free).
    pub status: RunStatus,
    /// Its new 1-based place in the queue; `null` once it started.
    pub queue_position: Option<u32>,
}

/// `run_status`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunStatusArgs {
    /// Run id.
    pub run_id: String,
    /// How many of the most recent log lines to include.
    #[serde(default = "default_log_lines")]
    #[schemars(range(min = 0, max = 200))]
    pub log_lines: u32,
}

fn default_log_lines() -> u32 {
    20
}

/// `run_status` result: the run plus its latest step, epoch and log lines.
#[derive(Debug, Serialize, JsonSchema)]
pub struct RunStatusResult {
    /// The run.
    #[serde(flatten)]
    pub run: Run,
    /// The latest optimizer step, if any.
    pub last_step: Option<Step>,
    /// The latest validated epoch, if any.
    pub last_epoch: Option<Epoch>,
    /// The run's distinct `warn` log lines, oldest first (at most 5): for example a run too short to learn
    /// from (docs/MODEL.md §8.4) or skipped non-finite batches. Left out when there are none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// The most recent log lines, oldest first (fewer than asked when the result would exceed the size limit).
    pub log: Vec<LogLine>,
}

/// `stop_run`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunIdArgs {
    /// Run id.
    pub run_id: String,
}

/// `reorder_run`: exactly one of `action` and `position` (docs/API.md, `POST /api/runs/{id}/queue`).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReorderRunArgs {
    /// Run id of a run waiting in the queue (`queue_position` set).
    pub run_id: String,
    /// `top` / `up` / `down` / `bottom` move it; `start` starts it now when nothing trains, else moves it to the
    /// front (the active run is never preempted).
    #[serde(default)]
    pub action: Option<folo_api::QueueAction>,
    /// The 1-based place to move it to (1 starts next; past the end means the end).
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub position: Option<u32>,
}

/// `list_runs`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListRunsArgs {
    /// Only runs with this status.
    #[serde(default)]
    pub status: Option<RunStatus>,
    /// `next_cursor` of the previous page; omit for the first page.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size (the page may be shorter to stay under the result size limit).
    #[serde(default)]
    #[schemars(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

/// `val`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValArgs {
    /// Model id.
    pub model_id: String,
    /// Dataset id.
    pub dataset_id: String,
    /// Split to evaluate.
    #[serde(default = "default_val_split")]
    pub split: Split,
    /// Square input size [default: the model's].
    #[serde(default)]
    pub imgsz: Option<u32>,
    /// Confidence threshold.
    #[serde(default = "folo_api::default_val_conf")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f32,
    /// NMS IoU threshold.
    #[serde(default = "folo_api::default_iou")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub iou: f32,
}

fn default_val_split() -> Split {
    Split::Val
}

/// `predict`. Give exactly one image source: `image_path`, `image_base64`, or `dataset_id` + `image_id`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PredictArgs {
    /// Model id.
    pub model_id: String,
    /// An image file (absolute, or relative to `--data-dir`) on the machine running `folo mcp`.
    #[serde(default)]
    pub image_path: Option<String>,
    /// The image bytes (JPEG, PNG, BMP or WebP), base64-encoded.
    #[serde(default)]
    pub image_base64: Option<String>,
    /// A dataset image: its dataset id (with `image_id`).
    #[serde(default)]
    pub dataset_id: Option<String>,
    /// A dataset image: its id (with `dataset_id`).
    #[serde(default)]
    pub image_id: Option<String>,
    /// Confidence threshold.
    #[serde(default = "folo_api::default_conf")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f32,
    /// NMS IoU threshold.
    #[serde(default = "folo_api::default_iou")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub iou: f32,
    /// Maximum detections.
    #[serde(default = "folo_api::default_max_det")]
    #[schemars(range(min = 1, max = 300))]
    pub max_det: u32,
}

/// `predict_batch`. Give `paths` or `glob`; at most 256 images after glob expansion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PredictBatchArgs {
    /// Model id.
    pub model_id: String,
    /// Image files (absolute, or relative to `--data-dir`).
    #[serde(default)]
    pub paths: Option<Vec<String>>,
    /// A glob such as `D:/photos/*.jpg` (relative patterns are relative to `--data-dir`).
    #[serde(default)]
    pub glob: Option<String>,
    /// Confidence threshold.
    #[serde(default = "folo_api::default_conf")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub conf: f32,
    /// NMS IoU threshold.
    #[serde(default = "folo_api::default_iou")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub iou: f32,
}

/// One image `predict_batch` could not process.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct BatchFailure {
    /// The input path.
    pub path: String,
    /// Error code, e.g. `image_decode_failed`.
    pub code: String,
    /// Human message.
    pub message: String,
}

/// Wall time of a batch.
#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
pub struct BatchTiming {
    /// Whole batch, milliseconds.
    pub total_ms: f64,
    /// Mean model time per processed image (`timing.total_ms` of `/api/predict`), milliseconds.
    pub mean_ms: f64,
}

/// `predict_batch` result: summaries only; the per-image boxes are at `details_uri`.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PredictBatchResult {
    /// Id of the batch (`folo://predictions/{batch_id}`).
    pub batch_id: String,
    /// Images processed.
    pub processed: u64,
    /// Images that failed (at most 50 listed; `failed_total` counts all).
    pub failed: Vec<BatchFailure>,
    /// All failures.
    pub failed_total: u64,
    /// Boxes over all images.
    pub boxes_total: u64,
    /// Boxes per class name.
    pub per_class_counts: BTreeMap<String, u64>,
    /// Timing.
    pub timing: BatchTiming,
    /// Resource with the per-image boxes.
    pub details_uri: String,
}

/// `export`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportArgs {
    /// Model id.
    pub model_id: String,
    /// Target format.
    pub format: ExportFormat,
    /// ONNX only: decode inside the graph (one `detections` output instead of the raw head).
    #[serde(default)]
    pub decode: bool,
    /// ONNX only: symbolic batch, height and width.
    #[serde(default)]
    pub dynamic: bool,
    /// ONNX only: f16 weights, input and outputs (the file name gets `-f16`).
    #[serde(default)]
    pub half: bool,
}

/// `compare_runs`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareArgs {
    /// 2 to 8 run ids.
    #[schemars(length(min = 2, max = 8))]
    pub run_ids: Vec<String>,
}

/// One run in `compare_runs`.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CompareEntry {
    /// Run id.
    pub run_id: String,
    /// Run name.
    pub name: String,
    /// Status.
    pub status: RunStatus,
    /// Architecture.
    pub arch: Arch,
    /// Validated epochs so far.
    pub epochs_done: u32,
    /// Configured epochs.
    pub epochs: u32,
    /// Best epoch by mAP@0.5:0.95.
    pub best: Option<BestEpoch>,
    /// Metrics of the best epoch (without per-class rows).
    pub best_metrics: Option<ModelMetrics>,
    /// Mean training loss of the last epoch.
    pub final_train_loss: Option<Loss>,
    /// Training wall time so far (sum of epoch durations), seconds.
    pub duration_s: f64,
}

/// `compare_runs` result.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CompareResult {
    /// The runs, in the order asked.
    pub runs: Vec<CompareEntry>,
    /// TrainConfig fields that differ between the runs (dotted paths such as `augment.mosaic`), each with one
    /// value per run in the same order.
    pub config_diff: BTreeMap<String, Vec<serde_json::Value>>,
}

/// A page of a list tool.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Page<T> {
    /// This page.
    pub items: Vec<T>,
    /// Pass as `cursor` to get the next page; `null` on the last page.
    pub next_cursor: Option<String>,
    /// Matching items over all pages.
    pub total: u64,
}

/// `list_datasets` item.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct DatasetSummary {
    /// Dataset id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Format.
    pub format: DatasetFormat,
    /// Number of classes.
    pub num_classes: usize,
    /// Class names (the first 100; `get_dataset` has all).
    pub classes: Vec<String>,
    /// Images per split.
    pub splits: Splits,
}

/// `get_dataset` result: the dataset with stats, plus its most serious issues.
#[derive(Debug, Serialize, JsonSchema)]
pub struct DatasetDetail {
    /// The dataset, with `stats`.
    #[serde(flatten)]
    pub dataset: folo_api::Dataset,
    /// Errors, then warnings, each in dataset order (at most 50; infos are only counted). `list_dataset_issues`
    /// pages through all of them.
    pub issues: Vec<DatasetIssue>,
    /// All issues, infos included.
    pub issues_total: u64,
    /// All issues by severity and by kind.
    pub issue_counts: IssueCounts,
}

/// `list_dataset_issues`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListIssuesArgs {
    /// Dataset id.
    pub dataset_id: String,
    /// Only this severity (`error`: the loader skips the image; `warning`; `info`: background images).
    #[serde(default)]
    pub severity: Option<IssueSeverity>,
    /// Only this kind, e.g. `class_overflow`.
    #[serde(default)]
    pub kind: Option<String>,
    /// `next_cursor` of the previous page; omit for the first page.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size (the page may be shorter to stay under the result size limit).
    #[serde(default)]
    #[schemars(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

/// `list_dataset_issues` result: a page of issues plus the counts of all of them.
#[derive(Debug, Serialize, JsonSchema)]
pub struct IssuesPage {
    /// This page, in dataset order.
    pub items: Vec<DatasetIssue>,
    /// Pass as `cursor` to get the next page; `null` on the last page.
    pub next_cursor: Option<String>,
    /// Issues matching the filter over all pages.
    pub total: u64,
    /// All issues of the dataset by severity and by kind, whatever the filter.
    pub counts: IssueCounts,
}

/// `list_dataset_images` item.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ImageSummary {
    /// Image id (`folo://datasets/{dataset_id}/images/{image_id}` is the image: a 512 px JPEG, `?size=original` for the file).
    pub image_id: String,
    /// Path relative to the dataset root.
    pub file: String,
    /// Split.
    pub split: Split,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Ground truth (only with `include_boxes: true`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub boxes: Option<Vec<BBox>>,
}

/// `list_models` item.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ModelSummary {
    /// Model id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Architecture.
    pub arch: Arch,
    /// Input size.
    pub imgsz: u32,
    /// Number of classes.
    pub num_classes: usize,
    /// Class names (the first 100).
    pub classes: Vec<String>,
    /// Validation metrics, if known.
    pub metrics: Option<ModelMetrics>,
    /// Detection head: `o2o` models predict with the NMS-free one-to-one branch.
    pub head: Head,
    /// Training settings from the file's `folo.train` (`null` for exports and imported weights).
    pub trained_with: Option<TrainedWith>,
    /// Stored weight dtype (`folo.dtype`): `f16` for half-precision exports. Left out when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<ModelPrecision>,
    /// Conv+BN fused (`folo.fused`, an inference-only export); run checkpoints are unfused. Left out when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fused: Option<bool>,
    /// `run`, `file` (the models folder) or `published` (weights from `folo weights pull`; the id is the published
    /// name, e.g. `folo-n-coco`). Left out by older servers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
    /// What published weights are; left out for other models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
}

/// CPU of the machine running `folo mcp`.
#[derive(Debug, Serialize, JsonSchema)]
pub struct CpuDevice {
    /// Marketing name.
    pub name: String,
    /// Logical threads.
    pub threads: u32,
    /// Total memory in bytes.
    pub memory_bytes: u64,
}

/// Compute devices.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Devices {
    /// The CPU.
    pub cpu: CpuDevice,
    /// GPUs FOLO can use (`memory_bytes` is 0 when the driver does not report it).
    pub gpus: Vec<GpuInfo>,
}

/// One model scale.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ScaleInfo {
    /// Architecture.
    pub arch: Arch,
    /// Learnable parameters with 80 classes.
    pub params: u64,
    /// GFLOPs per 640x640 image.
    pub gflops_640: f64,
    /// GFLOPs per 320x320 image.
    pub gflops_320: f64,
}

/// `folo_info` result.
#[derive(Debug, Serialize, JsonSchema)]
pub struct FoloInfo {
    /// FOLO version.
    pub version: String,
    /// `embedded`, `mock` or `proxy`.
    pub mode: String,
    /// The `folo serve` this server forwards to (proxy mode).
    pub server: Option<String>,
    /// Where FOLO keeps datasets, runs, models and exports, with `/` separators. In proxy mode the server's
    /// (`/api/system`'s `data_dir`; `null` for a server too old to report it).
    pub data_dir: Option<String>,
    /// Mutating tools are disabled.
    pub read_only: bool,
    /// Input paths must be inside `data_dir`.
    pub confine: bool,
    /// Devices of this machine (of the server's machine in proxy mode).
    pub devices: Devices,
    /// What runs with `device: auto` can use: f16 inference, tensor cores, mixed-precision training (`amp`, with
    /// the reason when it is not available). `/api/system`'s `capabilities` (`null` from a server too old to
    /// report it).
    pub capabilities: Option<Capabilities>,
    /// Model scales.
    pub scales: Vec<ScaleInfo>,
}

/// Rebuilds the summary of a val report (kept here so `compare_runs` and `val` describe metrics the same way).
pub fn metrics_of(v: &ValMetrics) -> ModelMetrics {
    ModelMetrics {
        map50: v.map50,
        map50_95: v.map50_95,
        precision: v.precision,
        recall: v.recall,
        best_f1_conf: v.best_f1_conf,
    }
}
