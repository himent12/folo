//! Datasets (docs/API.md, *Datasets*): registered datasets, their statistics, images, boxes and issues.

use serde::{Deserialize, Serialize};

use crate::runs::default_imgsz;

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
    /// Labels left out because they cannot be read. For YOLO one per label file with a line that cannot be parsed or
    /// is out of range, however many such lines it has, and its image is left out. For COCO one per annotation that
    /// cannot be read (its image is left out), and one per category, image or annotation entry that cannot be read
    /// and per annotation file that is not COCO JSON (the entry or file is dropped, the other images are kept). Not
    /// a count of images. Each broken line or entry is an error in `/issues`.
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
