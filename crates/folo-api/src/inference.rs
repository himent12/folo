//! Inference (docs/API.md, *Inference*): `POST /api/predict`.

use serde::{Deserialize, Serialize};

use crate::{ScoredBox, json};

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
    /// The model's first prediction since it was loaded: `timing` includes its warm-up (GPU kernels compiled,
    /// buffers allocated), so it is not what the next call takes. Left out when false (older servers never send it).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cold: bool,
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
    #[serde(serialize_with = "json::serialize_ms")]
    pub preprocess_ms: f64,
    /// Forward pass.
    #[serde(serialize_with = "json::serialize_ms")]
    pub inference_ms: f64,
    /// Decode + NMS + rescale.
    #[serde(serialize_with = "json::serialize_ms")]
    pub postprocess_ms: f64,
    /// Sum of the above.
    #[serde(serialize_with = "json::serialize_ms")]
    pub total_ms: f64,
}
