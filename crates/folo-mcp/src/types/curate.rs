//! MCP schemas for validation error diagnostics (`val`) and dataset curation tools (`audit_dataset`,
//! `merge_datasets`, `filter_dataset`).

use std::collections::BTreeMap;

use folo_api::{DatasetFormat, Split, Splits, ValReport};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Aggregated per-image detection error breakdown from `val` with `diagnostics: true` or `worst`.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ValDiagnosticsSummaryOut {
    pub num_truths: u64,
    pub num_preds: u64,
    pub tp: u64,
    pub fp: u64,
    pub fn_count: u64,
    pub localization_errors: u64,
    pub class_confusions: u64,
    pub duplicate_errors: u64,
    pub background_fp: u64,
    pub mean_f1: f32,
    pub mean_tp_iou: f32,
}

/// Per-image detection error breakdown for one failing validation image.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ImageDiagnosticsOut {
    pub image_index: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_id: Option<String>,
    pub num_truths: u32,
    pub num_preds: u32,
    pub tp: u32,
    pub fp: u32,
    pub fn_count: u32,
    pub localization_errors: u32,
    pub class_confusions: u32,
    pub duplicate_errors: u32,
    pub background_fp: u32,
    pub f1: f32,
    pub mean_tp_iou: f32,
}

/// Optional per-image error diagnosis payload attached to `val`.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ValDiagnosticsOut {
    pub conf: f32,
    pub tp_iou: f32,
    pub bg_iou: f32,
    pub images_diagnosed: u64,
    pub failing_images: u64,
    pub summary: ValDiagnosticsSummaryOut,
    pub worst_images: Vec<ImageDiagnosticsOut>,
}

/// `val` tool result: [`ValReport`] plus optional per-image `diagnostics`.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ValToolResult {
    #[serde(flatten)]
    pub report: ValReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<ValDiagnosticsOut>,
}

/// Class-name normalization mode for `merge_datasets` and `filter_dataset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ClassNormalizeMode {
    #[default]
    Exact,
    CaseAndWhitespace,
}

/// Split assignment mode for `merge_datasets`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum MergeSplitMode {
    #[default]
    Preserve,
    PoolToTrain,
    Stratified,
}

/// Filename collision handling for `merge_datasets`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum MergeCollisionMode {
    #[default]
    PrefixOnCollision,
    AlwaysPrefix,
}

/// `audit_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AuditDatasetArgs {
    /// Dataset id.
    pub dataset_id: String,
    /// Enable 64-bit dHash perceptual near-duplicate and cross-split leakage detection (default true).
    #[serde(default = "default_true")]
    pub perceptual: bool,
    /// Maximum Hamming distance (0..=64) for perceptual near-duplicates (requires `perceptual: true`; default 6).
    #[serde(default)]
    #[schemars(range(min = 0, max = 64))]
    pub dhash_threshold: Option<u32>,
    /// Maximum example groups returned per category (1..=50, default 20).
    #[serde(default)]
    #[schemars(range(min = 1, max = 50))]
    pub max_groups: Option<usize>,
}

fn default_true() -> bool {
    true
}

/// `audit_dataset` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct AuditDatasetResult {
    pub dataset_id: String,
    pub images: usize,
    pub exact_duplicate_groups: usize,
    pub exact_leakage_groups: usize,
    pub perceptual_enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dhash_threshold: Option<u32>,
    pub near_duplicate_groups: usize,
    pub near_leakage_groups: usize,
    pub has_split_leakage: bool,
    pub exact_leakage_examples: Vec<Vec<String>>,
    pub near_leakage_examples: Vec<Vec<String>>,
    pub near_duplicate_examples: Vec<Vec<String>>,
}

/// `merge_datasets`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MergeDatasetsArgs {
    /// 2 to 32 registered dataset ids to merge.
    #[schemars(length(min = 2, max = 32))]
    pub dataset_ids: Vec<String>,
    /// Display name for the merged dataset.
    pub name: String,
    /// Class-name normalization (`exact` or `case-and-whitespace`).
    #[serde(default)]
    pub normalize: ClassNormalizeMode,
    /// Optional class map (`source -> target`, or `source -> null` to drop that class).
    #[serde(default)]
    pub class_map: Option<BTreeMap<String, Option<String>>>,
    /// Drop boxes whose class is absent from `class_map`.
    #[serde(default)]
    pub drop_unmapped: bool,
    /// Drop images that become empty after dropping unmapped boxes.
    #[serde(default)]
    pub drop_empty_images: bool,
    /// Split strategy (`preserve`, `pool-to-train`, or `stratified`).
    #[serde(default)]
    pub splits: MergeSplitMode,
    /// Validation fraction for `splits: "stratified"` (0..=1, default 0.2).
    #[serde(default)]
    pub val: Option<f64>,
    /// Test fraction for `splits: "stratified"` (0..=1, default 0.0).
    #[serde(default)]
    pub test: Option<f64>,
    /// Deterministic RNG seed for `splits: "stratified"`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Filename collision handling (`prefix-on-collision` or `always-prefix`).
    #[serde(default)]
    pub collisions: MergeCollisionMode,
    /// Optional per-source filename prefixes (same length as `dataset_ids`).
    #[serde(default)]
    pub source_prefixes: Option<Vec<String>>,
}

/// `merge_datasets` result (bounded summary; call `get_dataset` for full statistics).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MergeDatasetsResult {
    pub dataset_id: String,
    pub name: String,
    pub path: String,
    pub format: DatasetFormat,
    pub num_classes: usize,
    pub classes: Vec<String>,
    pub splits: Splits,
    pub boxes: usize,
    pub renamed_collisions: usize,
}

/// `filter_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterDatasetArgs {
    /// Source dataset id.
    pub dataset_id: String,
    /// Display name for the filtered dataset.
    pub name: String,
    /// Output format (`yolo` or `coco`; default `yolo`).
    #[serde(default)]
    pub format: Option<DatasetFormat>,
    /// Keep only boxes of these class names.
    #[serde(default)]
    pub include_classes: Option<Vec<String>>,
    /// Drop boxes of these class names.
    #[serde(default)]
    pub exclude_classes: Option<Vec<String>>,
    /// Class-name normalization for matching (`exact` or `case-and-whitespace`).
    #[serde(default)]
    pub normalize: ClassNormalizeMode,
    /// Compact surviving class ids to `0..K` (default true).
    #[serde(default = "default_true")]
    pub compact_classes: bool,
    /// Keep only these splits (default: all).
    #[serde(default)]
    pub splits: Option<Vec<Split>>,
    /// Minimum normalized box area (`w * h` in `0..=1`).
    #[serde(default)]
    pub min_box_area: Option<f32>,
    /// Maximum normalized box area (`w * h` in `0..=1`).
    #[serde(default)]
    pub max_box_area: Option<f32>,
    /// Minimum box pixel width (`>= 0`).
    #[serde(default)]
    pub min_box_width_px: Option<f32>,
    /// Minimum box pixel height (`>= 0`).
    #[serde(default)]
    pub min_box_height_px: Option<f32>,
    /// Maximum box pixel width (`>= 0`).
    #[serde(default)]
    pub max_box_width_px: Option<f32>,
    /// Maximum box pixel height (`>= 0`).
    #[serde(default)]
    pub max_box_height_px: Option<f32>,
    /// Minimum image width in pixels (`>= 1`).
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub min_width: Option<u32>,
    /// Minimum image height in pixels (`>= 1`).
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub min_height: Option<u32>,
    /// Maximum image width in pixels (`>= 1`).
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub max_width: Option<u32>,
    /// Maximum image height in pixels (`>= 1`).
    #[serde(default)]
    #[schemars(range(min = 1))]
    pub max_height: Option<u32>,
    /// Drop images that have zero surviving boxes.
    #[serde(default)]
    pub drop_empty_images: bool,
}

/// `filter_dataset` result (bounded summary; call `get_dataset` for full statistics).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct FilterDatasetResult {
    pub dataset_id: String,
    pub name: String,
    pub path: String,
    pub format: DatasetFormat,
    pub num_classes: usize,
    pub classes: Vec<String>,
    pub splits: Splits,
    pub source_images: usize,
    pub source_boxes: usize,
    pub kept_images: usize,
    pub kept_boxes: usize,
    pub dropped_images: usize,
    pub dropped_boxes: usize,
    /// Class names from `include_classes` / `exclude_classes` that did not match any class in the source dataset.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmatched_classes: Vec<String>,
}
