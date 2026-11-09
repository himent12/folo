//! FOLO HTTP API types: the serde request and response bodies of `docs/API.md`.
//!
//! This crate holds data types only (no IO, no async), so it can be shared by `folo-server`, the CLI and FOLO
//! Desktop (`apps/desktop`, a separate cargo workspace that depends on this crate by path). Field names and enum
//! spellings are the wire format; keep them in sync with `docs/API.md` and `web/src/lib/api/types.ts`.
//!
//! Conventions: times are RFC 3339 strings, durations are seconds (`*_s`) or milliseconds (`*_ms`), boxes are pixel
//! `xyxy` in the original image, ids are opaque strings.

use serde::{Deserialize, Serialize};

pub mod codes;
pub mod json;

/// API version string reported in [`Health::api`].
pub const API_VERSION: &str = "v1";

// ---------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------

/// The body of every non-2xx response: `{ "error": { "code", "message", "hint" } }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ErrorBody {
    /// The error itself.
    pub error: ApiError,
}

/// A machine-readable error: a stable `code`, a human `message` and an optional `hint` on how to fix it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ApiError {
    /// Stable snake_case code, e.g. `dataset_not_found`. See the table in `docs/API.md`.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// What to do about it, if there is something useful to say. Always serialized (`null` when absent).
    #[serde(default)]
    pub hint: Option<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------------------------------------------

/// `GET /api/health`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Health {
    /// Always `"ok"` when the server answers.
    pub status: String,
    /// Server (crate) version, e.g. `0.1.0`.
    pub version: String,
    /// API version, currently `v1`.
    #[serde(default = "default_api")]
    pub api: String,
    /// True when the server runs `--mock`.
    pub mock: bool,
}

fn default_api() -> String {
    API_VERSION.to_string()
}

/// `GET /api/system`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SystemInfo {
    /// Host CPU.
    pub cpu: CpuInfo,
    /// Host memory.
    pub memory: MemoryInfo,
    /// GPUs usable by FOLO (empty until GPU detection lands).
    pub gpus: Vec<GpuInfo>,
}

/// What `GET /api/system` sends: [`SystemInfo`] plus the server's data dir. (A separate type, so code that builds
/// a [`SystemInfo`] is unaffected.)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SystemResponse {
    /// CPU, memory and GPUs.
    #[serde(flatten)]
    pub system: SystemInfo,
    /// The server's data dir (`--data-dir`) with `/` separators, where its runs, models and exports live; absent
    /// in mock mode and from servers older than this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<String>,
    /// Optional API features this server implements (for example [`FEATURE_EXPORT_ONNX_SWITCHES`]), so a client
    /// can tell before it relies on one; empty from servers older than this field.
    #[serde(default)]
    pub features: Vec<String>,
}

/// [`SystemResponse::features`]: `POST /api/models/{id}/export` honours [`OnnxSwitches`] (older servers accept
/// and ignore them).
pub const FEATURE_EXPORT_ONNX_SWITCHES: &str = "export_onnx_switches";

/// CPU name, logical thread count and current utilisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CpuInfo {
    /// Marketing name, e.g. `AMD Ryzen 9 7950X 16-Core Processor`.
    pub name: String,
    /// Logical threads.
    pub threads: u32,
    /// Utilisation in `0..=1` averaged over all threads since the previous call.
    pub util: f32,
}

/// Host memory in bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryInfo {
    /// Bytes in use.
    pub used_bytes: u64,
    /// Total physical bytes.
    pub total_bytes: u64,
}

/// One GPU adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GpuInfo {
    /// Index used in device strings such as `gpu:0`.
    pub ordinal: u32,
    /// Adapter name.
    pub name: String,
    /// Graphics API the adapter is driven through.
    pub backend: GpuBackend,
    /// Dedicated memory in bytes.
    pub memory_bytes: u64,
    /// Utilisation in `0..=1`, or `null` when the platform does not report it.
    pub util: Option<f32>,
}

/// GPU graphics API.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    /// Vulkan.
    Vulkan,
    /// Direct3D 12.
    Dx12,
    /// Metal.
    Metal,
}

// ---------------------------------------------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------------------------------------------

/// `GET /api/datasets`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DatasetList {
    /// Registered datasets, oldest first.
    pub datasets: Vec<Dataset>,
}

/// A registered dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Dataset {
    /// Opaque id, e.g. `ds_8f2c`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Root folder on disk (`mock://<name>` in mock mode).
    pub path: String,
    /// On-disk format.
    pub format: DatasetFormat,
    /// Class names; a box's `class` indexes this list.
    pub classes: Vec<String>,
    /// RFC 3339 registration time.
    pub created: String,
    /// Image count per split.
    pub splits: Splits,
    /// Statistics; present on `GET /api/datasets/{id}`, omitted in lists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stats: Option<DatasetStats>,
}

/// Dataset format. `auto` is only valid in requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    /// Detect from the folder contents (requests only).
    #[default]
    Auto,
    /// YOLO txt labels + `data.yaml`.
    Yolo,
    /// COCO JSON.
    Coco,
}

/// Image counts per split.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Splits {
    /// Training images.
    pub train: u64,
    /// Validation images.
    pub val: u64,
    /// Test images.
    pub test: u64,
}

/// A dataset split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Training split.
    Train,
    /// Validation split.
    Val,
    /// Test split.
    Test,
}

impl Split {
    /// All splits in order.
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    /// Wire name (`train`, `val`, `test`).
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

/// Dataset statistics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DatasetStats {
    /// Total boxes over all splits.
    pub boxes: u64,
    /// Images with no boxes.
    pub empty_images: u64,
    /// Label lines that could not be parsed or are out of range.
    pub broken_labels: u64,
    /// Boxes per class, indexed like `classes`.
    pub class_counts: Vec<u64>,
    /// Histogram of relative box size `sqrt(box_area / image_area)`.
    pub box_size_hist: Histogram,
    /// Histogram of boxes per image.
    pub boxes_per_image_hist: Histogram,
}

/// A histogram: `counts[i]` counts values in `[edges[i], edges[i + 1])`; the last bin also takes values at or above
/// its upper edge. `edges.len() == counts.len() + 1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Histogram {
    /// Bin edges, ascending.
    pub edges: Vec<f64>,
    /// Counts per bin.
    pub counts: Vec<u64>,
}

/// `POST /api/datasets`: register a folder on disk.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RegisterDataset {
    /// Display name; defaults to the folder name.
    #[serde(default)]
    pub name: Option<String>,
    /// Root folder.
    pub path: String,
    /// Format; `auto` detects it.
    #[serde(default)]
    pub format: DatasetFormat,
}

/// `POST /api/datasets/synthetic`: generate a synthetic shapes dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SyntheticDataset {
    /// Display name.
    pub name: String,
    /// Number of images (split 80 / 20 into train / val).
    #[serde(default = "default_synth_images")]
    pub images: u64,
    /// Class names (1 to 8, distinct); `circle`, `square`, `triangle`, `star`, `ring`, `cross`, `hexagon` and `heart`
    /// get those shapes, other names the next shape no name has claimed.
    #[serde(default = "default_synth_classes")]
    pub classes: Vec<String>,
    /// Square image side in pixels.
    #[serde(default = "default_imgsz")]
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

/// `GET /api/datasets/{id}/images`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImagePage {
    /// Images matching the filter (before `offset` / `limit`).
    pub total: u64,
    /// This page.
    pub images: Vec<ImageInfo>,
}

/// `GET /api/datasets/{id}/issues`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IssuePage {
    /// Issues matching the filter (before `offset` / `limit`).
    pub total: u64,
    /// This page.
    pub issues: Vec<DatasetIssue>,
    /// Every issue of the dataset counted by severity and by kind, whatever the filter (servers before these
    /// counts existed omit them; they then read as zero).
    #[serde(default)]
    pub counts: IssueCounts,
}

/// Issue counts by severity and by kind (only kinds that occur), as `folo dataset check --json` reports them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct IssueCounts {
    /// Errors (the loader skips those images).
    pub error: u64,
    /// Warnings (fixed up on load, or suspicious).
    pub warning: u64,
    /// Infos (background images).
    pub info: u64,
    /// Issues per kind.
    pub by_kind: std::collections::BTreeMap<String, u64>,
}

impl IssueCounts {
    /// Counts `issues`.
    pub fn of<'a>(issues: impl IntoIterator<Item = &'a DatasetIssue>) -> IssueCounts {
        let mut c = IssueCounts::default();
        for i in issues {
            match i.severity {
                IssueSeverity::Error => c.error += 1,
                IssueSeverity::Warning => c.warning += 1,
                IssueSeverity::Info => c.info += 1,
            }
            *c.by_kind.entry(i.kind.clone()).or_default() += 1;
        }
        c
    }
}

/// Every issue kind, in the order `folo dataset check` reports counts. Errors: `broken_label`, `class_overflow`,
/// `out_of_range_box`, `corrupt_image`, `missing_image`; warnings: `clipped_box`, `zero_size_box`,
/// `duplicate_label`, `duplicate_image`, `other`; info: `empty_image`.
pub const ISSUE_KINDS: [&str; 11] = [
    "broken_label",
    "class_overflow",
    "out_of_range_box",
    "clipped_box",
    "zero_size_box",
    "duplicate_label",
    "corrupt_image",
    "missing_image",
    "duplicate_image",
    "empty_image",
    "other",
];

/// How serious a [`DatasetIssue`] is (ordered most serious first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum IssueSeverity {
    /// The loader skips the image.
    Error,
    /// Fixed up on load, or suspicious.
    Warning,
    /// Not a problem (a background image).
    Info,
}

impl IssueSeverity {
    /// Every severity, most serious first.
    pub const ALL: [IssueSeverity; 3] = [IssueSeverity::Error, IssueSeverity::Warning, IssueSeverity::Info];

    /// The serialised name (`error`, `warning`, `info`).
    pub fn as_str(self) -> &'static str {
        match self {
            IssueSeverity::Error => "error",
            IssueSeverity::Warning => "warning",
            IssueSeverity::Info => "info",
        }
    }

    /// Parses a serialised name.
    pub fn parse(s: &str) -> Option<IssueSeverity> {
        IssueSeverity::ALL.into_iter().find(|v| v.as_str() == s)
    }
}

/// One problem in a dataset; the same object as an entry of `folo dataset check --fix-dry-run --json`'s `issues`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DatasetIssue {
    /// One of [`ISSUE_KINDS`].
    pub kind: String,
    /// Severity of the kind.
    pub severity: IssueSeverity,
    /// The file it is in, relative to the dataset root with `/` separators (absolute outside it): a label file
    /// for label problems, else the image or the annotation JSON.
    pub file: String,
    /// 1-based line in `file` for label-line problems, else `null`.
    pub line: Option<u32>,
    /// Human-readable details.
    pub message: String,
    /// What a cleanup would do about it, e.g. `delete line 3` (absent for infos). FOLO never applies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

/// One dataset image with its ground-truth boxes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImageInfo {
    /// Opaque id, unique within the dataset.
    pub id: String,
    /// Path relative to the dataset root, e.g. `images/train/000042.jpg`.
    pub file: String,
    /// Split.
    pub split: Split,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Ground truth.
    pub boxes: Vec<BBox>,
}

/// A labelled box, pixel `xyxy`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BBox {
    /// Class index.
    pub class: u32,
    /// Left.
    pub x1: f32,
    /// Top.
    pub y1: f32,
    /// Right.
    pub x2: f32,
    /// Bottom.
    pub y2: f32,
}

/// A predicted box, pixel `xyxy`, with class name and confidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ScoredBox {
    /// Class index.
    pub class: u32,
    /// Class name.
    pub name: String,
    /// Confidence in `0..=1`.
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

// ---------------------------------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------------------------------

/// `GET /api/models`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelList {
    /// Models, newest first.
    pub models: Vec<Model>,
}

/// A trained model (`.folo` weights).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Model {
    /// Opaque id, e.g. `m_31aa`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Architecture.
    pub arch: Arch,
    /// Training / inference image size.
    pub imgsz: u32,
    /// Class names.
    pub classes: Vec<String>,
    /// Parameter count.
    pub params: u64,
    /// GFLOPs per image at `imgsz`.
    pub flops_g: f64,
    /// File size in bytes.
    pub bytes: u64,
    /// RFC 3339 creation time.
    pub created: String,
    /// The run that produced it, if any.
    pub run_id: Option<String>,
    /// Validation metrics of these weights, if known.
    pub metrics: Option<ModelMetrics>,
}

/// Detection metrics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelMetrics {
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision.
    pub precision: f32,
    /// Recall.
    pub recall: f32,
}

/// Detector architecture (scale).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Arch {
    /// Nano.
    #[default]
    #[serde(rename = "folo-n")]
    N,
    /// Small.
    #[serde(rename = "folo-s")]
    S,
    /// Medium.
    #[serde(rename = "folo-m")]
    M,
    /// Large.
    #[serde(rename = "folo-l")]
    L,
    /// Extra large.
    #[serde(rename = "folo-x")]
    X,
}

impl Arch {
    /// All architectures, smallest first.
    pub const ALL: [Arch; 5] = [Arch::N, Arch::S, Arch::M, Arch::L, Arch::X];

    /// Wire name, e.g. `folo-n`.
    pub fn as_str(self) -> &'static str {
        match self {
            Arch::N => "folo-n",
            Arch::S => "folo-s",
            Arch::M => "folo-m",
            Arch::L => "folo-l",
            Arch::X => "folo-x",
        }
    }

    /// Parses a wire name.
    pub fn parse(s: &str) -> Option<Arch> {
        Arch::ALL.into_iter().find(|a| a.as_str() == s)
    }
}

/// `GET /api/models/{id}/eval`: validation results of a model (PR curves, confusion matrix, per-class metrics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelEval {
    /// One precision-recall curve per class, in class order.
    pub pr: Vec<PrCurve>,
    /// `(classes + 1) x (classes + 1)` counts: rows are the true class, columns the predicted class; the last row and
    /// column are background (`confusion[c][n]` = missed objects of class `c`, `confusion[n][c]` = false positives
    /// predicted as `c`, `confusion[n][n]` = 0).
    pub confusion: Vec<Vec<u32>>,
    /// Per-class metrics (`map50` equals the curve's `ap50`).
    pub per_class: Vec<ClassMetrics>,
    /// Validation images evaluated.
    pub images: u32,
}

/// A precision-recall curve at IoU 0.5, sampled at 101 recall points (`0.00, 0.01, ..., 1.00`). Precision is the
/// COCO-interpolated envelope, so it never increases with recall.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PrCurve {
    /// Class index.
    pub class: u32,
    /// Recall points.
    pub recall: Vec<f32>,
    /// Precision at each recall point.
    pub precision: Vec<f32>,
    /// AP at IoU 0.5: the mean of `precision`.
    pub ap50: f32,
}

/// The ONNX switches of `POST /api/models/{id}/export` (docs/CLI.md, ONNX export), all off by default. Only
/// valid with `format: onnx`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OnnxSwitches {
    /// Decode inside the graph: one `detections: [B, A, 4 + nc]` output instead of the raw head.
    #[serde(default)]
    pub decode: bool,
    /// Symbolic batch, height and width.
    #[serde(default)]
    pub dynamic: bool,
    /// F16 weights, input, activations and outputs (the file name gets `-f16`).
    #[serde(default)]
    pub half: bool,
}

impl OnnxSwitches {
    /// Whether any switch is on.
    pub fn any(&self) -> bool {
        self.decode || self.dynamic || self.half
    }
}

/// The full body of `POST /api/models/{id}/export`: [`ExportRequest`]'s `format` plus the [`OnnxSwitches`]
/// (every one optional, so an [`ExportRequest`] body is one of these).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportSpec {
    /// Target format.
    pub format: ExportFormat,
    /// ONNX switches.
    #[serde(flatten)]
    pub onnx: OnnxSwitches,
}

impl From<ExportRequest> for ExportSpec {
    fn from(r: ExportRequest) -> Self {
        ExportSpec { format: r.format, onnx: OnnxSwitches::default() }
    }
}

/// `POST /api/models/{id}/export` with the format only ([`ExportSpec`] adds the ONNX switches).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportRequest {
    /// Target format.
    pub format: ExportFormat,
}

/// Export format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ExportFormat {
    /// `.folo` with f16 weights.
    #[serde(rename = "folo-f16")]
    FoloF16,
    /// ONNX.
    #[serde(rename = "onnx")]
    Onnx,
}

/// Result of an export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportResult {
    /// Where the file was written.
    pub path: String,
    /// Its size.
    pub bytes: u64,
}

// ---------------------------------------------------------------------------------------------------------------
// Training runs
// ---------------------------------------------------------------------------------------------------------------

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
}

fn default_epochs() -> u32 {
    100
}
fn default_batch() -> u32 {
    16
}
fn default_imgsz() -> u32 {
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
    /// SGD for long runs, AdamW for short ones.
    #[default]
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW.
    Adamw,
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
    /// When it reached `stopped`, `completed` or `failed`.
    pub finished: Option<String>,
    /// Current epoch, 1-based; 0 before the first step. Equals the last epoch once finished.
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

/// Loss components.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Loss {
    /// CIoU box loss.
    #[serde(rename = "box")]
    pub box_: f32,
    /// Classification BCE.
    pub cls: f32,
    /// Distribution focal loss.
    pub dfl: f32,
    /// Weighted sum.
    pub total: f32,
}

/// One optimizer step (SSE `step`, and `/metrics` history).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Step {
    /// Epoch the step belongs to, 1-based.
    pub epoch: u32,
    /// Global step number, 1-based.
    pub step: u64,
    /// Learning rate used for this step.
    pub lr: f32,
    /// Training loss for this batch.
    pub loss: Loss,
    /// Throughput.
    pub imgs_per_s: f32,
    /// Estimated seconds remaining.
    pub eta_s: f64,
    /// Optimizer steps per epoch (`Run.steps_per_epoch`); 0 from servers that predate the field.
    #[serde(default)]
    pub steps_per_epoch: u64,
    /// Total epochs of the run (`Run.epochs`); 0 from servers that predate the field.
    #[serde(default)]
    pub epochs: u32,
}

/// One finished epoch with validation (SSE `epoch`, and `/metrics` history).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Epoch {
    /// Epoch number, 1-based.
    pub epoch: u32,
    /// Mean training loss over the epoch.
    pub train_loss: Loss,
    /// Validation metrics.
    pub val: ValMetrics,
    /// Wall time of the epoch including validation.
    pub duration_s: f64,
}

/// Validation metrics, overall and per class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ValMetrics {
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision.
    pub precision: f32,
    /// Recall.
    pub recall: f32,
    /// Per-class metrics.
    pub per_class: Vec<ClassMetrics>,
}

/// Validation metrics of one class.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ClassMetrics {
    /// Class index.
    pub class: u32,
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision.
    pub precision: f32,
    /// Recall.
    pub recall: f32,
}

/// `GET /api/runs/{id}/metrics`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunMetrics {
    /// Step history, oldest first (every `every`-th step when that query parameter is given).
    pub steps: Vec<Step>,
    /// Epoch history, oldest first.
    pub epochs: Vec<Epoch>,
}

/// `GET /api/runs/{id}/samples`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunSamples {
    /// Epoch the predictions come from.
    pub epoch: u32,
    /// The fixed validation images with truth and predictions.
    pub samples: Vec<Sample>,
}

/// One validation image with ground truth and predictions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Sample {
    /// Image id (fetch it from `/api/datasets/{dataset_id}/images/{image_id}`).
    pub image_id: String,
    /// Dataset id.
    pub dataset_id: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Ground truth.
    pub truth: Vec<BBox>,
    /// Predictions after NMS.
    pub pred: Vec<ScoredBox>,
}

/// `GET /api/runs/{id}/logs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunLogs {
    /// The most recent log lines, oldest first.
    pub lines: Vec<LogLine>,
}

/// SSE `status` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StatusEvent {
    /// New state.
    pub status: RunStatus,
    /// Error message when `failed`.
    pub error: Option<String>,
    /// Optimizer steps per epoch (`Run.steps_per_epoch`): 0 while unknown (a run that has not been built yet) and
    /// from servers that predate the field.
    #[serde(default)]
    pub steps_per_epoch: u64,
    /// Total epochs of the run (`Run.epochs`); 0 from servers that predate the field.
    #[serde(default)]
    pub epochs: u32,
}

impl StatusEvent {
    /// A status event with unknown run length (`steps_per_epoch` and `epochs` 0).
    pub fn new(status: RunStatus, error: Option<String>) -> StatusEvent {
        StatusEvent { status, error, steps_per_epoch: 0, epochs: 0 }
    }

    /// Sets the run's length.
    pub fn with_length(mut self, epochs: u32, steps_per_epoch: u64) -> StatusEvent {
        self.epochs = epochs;
        self.steps_per_epoch = steps_per_epoch;
        self
    }
}

/// Log level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Debug.
    Debug,
    /// Info.
    Info,
    /// Warning.
    Warn,
    /// Error.
    Error,
}

/// SSE `log` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LogLine {
    /// RFC 3339 time.
    pub t: String,
    /// Level.
    pub level: LogLevel,
    /// Message.
    pub message: String,
}

/// One event of `GET /api/runs/{id}/events`. On the wire the SSE `event:` field is [`RunEvent::name`] and `data:`
/// is the JSON of the payload alone.
#[derive(Clone, Debug, PartialEq)]
pub enum RunEvent {
    /// `status`.
    Status(StatusEvent),
    /// `step`.
    Step(Step),
    /// `epoch`.
    Epoch(Epoch),
    /// `log`.
    Log(LogLine),
}

impl RunEvent {
    /// The SSE event name.
    pub fn name(&self) -> &'static str {
        match self {
            RunEvent::Status(_) => "status",
            RunEvent::Step(_) => "step",
            RunEvent::Epoch(_) => "epoch",
            RunEvent::Log(_) => "log",
        }
    }

    /// The JSON `data:` payload.
    pub fn data_json(&self) -> String {
        let r = match self {
            RunEvent::Status(p) => serde_json::to_string(p),
            RunEvent::Step(p) => serde_json::to_string(p),
            RunEvent::Epoch(p) => serde_json::to_string(p),
            RunEvent::Log(p) => serde_json::to_string(p),
        };
        r.expect("event payloads always serialize")
    }

    /// Parses an SSE event from its `event:` name and `data:` JSON. Returns `None` for unknown event names (new
    /// event types may be added; clients should skip them) and an error for malformed data.
    pub fn from_sse(name: &str, data: &str) -> Option<Result<RunEvent, serde_json::Error>> {
        Some(match name {
            "status" => serde_json::from_str(data).map(RunEvent::Status),
            "step" => serde_json::from_str(data).map(RunEvent::Step),
            "epoch" => serde_json::from_str(data).map(RunEvent::Epoch),
            "log" => serde_json::from_str(data).map(RunEvent::Log),
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------------------------------------------

/// JSON body of `POST /api/predict` for an image already in a dataset. (Uploads use `multipart/form-data` with the
/// same field names plus an `image` file part.)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PredictRequest {
    /// Model to run.
    pub model_id: String,
    /// Dataset holding the image.
    pub dataset_id: String,
    /// Image id within the dataset.
    pub image_id: String,
    /// Confidence threshold.
    #[serde(default = "default_conf")]
    pub conf: f32,
    /// NMS IoU threshold.
    #[serde(default = "default_iou")]
    pub iou: f32,
    /// Maximum detections.
    #[serde(default = "default_max_det")]
    pub max_det: u32,
}

/// Default confidence threshold.
pub fn default_conf() -> f32 {
    0.25
}
/// Default NMS IoU threshold.
pub fn default_iou() -> f32 {
    0.7
}
/// Default maximum detections.
pub fn default_max_det() -> u32 {
    300
}

/// Response of `POST /api/predict`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PredictResponse {
    /// Model that ran.
    pub model_id: String,
    /// Input image size.
    pub image: ImageSize,
    /// Detections, highest score first.
    pub boxes: Vec<ScoredBox>,
    /// Timings.
    pub timing: Timing,
    /// Device it ran on, e.g. `gpu:0`.
    pub device: String,
}

/// Image dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ImageSize {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// Inference timings in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Timing {
    /// Decode + letterbox.
    pub preprocess_ms: f64,
    /// Forward pass.
    pub inference_ms: f64,
    /// Decode + NMS + rescale.
    pub postprocess_ms: f64,
    /// Sum of the above.
    pub total_ms: f64,
}

// ---------------------------------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------------------------------

/// `POST /api/val`: evaluate a model on a split of a dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ValRequest {
    /// Model to evaluate.
    pub model_id: String,
    /// Dataset to evaluate on.
    pub dataset_id: String,
    /// Split to evaluate.
    #[serde(default = "default_val_split")]
    pub split: Split,
    /// Square input size; `null` uses the model's training size.
    #[serde(default)]
    pub imgsz: Option<u32>,
    /// Confidence threshold (low, for mAP).
    #[serde(default = "default_val_conf")]
    pub conf: f32,
    /// NMS IoU threshold.
    #[serde(default = "default_iou")]
    pub iou: f32,
}

fn default_val_split() -> Split {
    Split::Val
}

/// Default confidence threshold of validation.
pub fn default_val_conf() -> f32 {
    0.001
}

/// Response of `POST /api/val`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ValReport {
    /// Model that was evaluated.
    pub model_id: String,
    /// Dataset it was evaluated on.
    pub dataset_id: String,
    /// Split.
    pub split: Split,
    /// Input size used.
    pub imgsz: u32,
    /// Images evaluated.
    pub images: u64,
    /// Metrics, as in `Epoch.val`.
    pub val: ValMetrics,
    /// How long it took.
    pub timing: ValTiming,
}

/// Wall time of a validation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ValTiming {
    /// Whole validation, seconds.
    pub total_s: f64,
    /// Mean per image (preprocess + inference + postprocess), milliseconds.
    pub per_image_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_config_defaults() {
        let c: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
        assert_eq!(c.epochs, 100);
        assert_eq!(c.arch, Arch::N);
        assert_eq!(c.device, "auto");
        assert_eq!(c.augment, Augment::default());
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["arch"], "folo-n");
        assert_eq!(v["optimizer"], "auto");
        assert_eq!(v["weights"], serde_json::Value::Null);
    }

    #[test]
    fn loss_box_field_name() {
        let l = Loss { box_: 1.0, cls: 2.0, dfl: 3.0, total: 6.0 };
        let v = serde_json::to_value(l).unwrap();
        assert_eq!(v["box"], 1.0);
    }

    #[test]
    fn error_hint_is_always_present() {
        let e = ErrorBody { error: ApiError { code: "x".into(), message: "y".into(), hint: None } };
        assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"error":{"code":"x","message":"y","hint":null}}"#);
    }

    #[test]
    fn run_event_roundtrip() {
        let ev = RunEvent::Status(StatusEvent::new(RunStatus::Running, None).with_length(100, 50));
        let back = RunEvent::from_sse(ev.name(), &ev.data_json()).unwrap().unwrap();
        assert_eq!(back, ev);
        assert!(RunEvent::from_sse("future", "{}").is_none());
    }

    #[test]
    fn run_length_fields_default_for_older_servers() {
        let s: StatusEvent = serde_json::from_str(r#"{"status":"running","error":null}"#).unwrap();
        assert_eq!(s, StatusEvent::new(RunStatus::Running, None));
        let step: Step = serde_json::from_str(
            r#"{"epoch":1,"step":1,"lr":0.01,"loss":{"box":1,"cls":1,"dfl":1,"total":3},"imgs_per_s":1,"eta_s":5}"#,
        )
        .unwrap();
        assert_eq!((step.steps_per_epoch, step.epochs), (0, 0));
        let v = serde_json::to_value(StatusEvent::new(RunStatus::Queued, None).with_length(3, 7)).unwrap();
        assert_eq!((v["epochs"].as_u64(), v["steps_per_epoch"].as_u64()), (Some(3), Some(7)));
    }
}
