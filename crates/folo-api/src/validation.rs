//! Validation (docs/API.md, *Validation*): `POST /api/val`.

use serde::{Deserialize, Serialize};

use crate::{Split, ValMetrics, default_iou, json};

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
    /// That dataset's name. Additive: missing (older servers) reads as `null`.
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// Split (the requested one: `POST /api/val` never falls back, an empty split is `invalid_dataset`).
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
    #[serde(serialize_with = "json::serialize_ms")]
    pub per_image_ms: f64,
}
