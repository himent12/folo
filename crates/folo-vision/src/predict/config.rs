//! Prediction configuration, timing breakdown, prediction output, and precision types.

use folo_core::{DType, Device};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::fusion::{BoxFusion, FUSION_IOU, ScoreRule};
use crate::model::HeadMode;
use crate::nms::{Detection, NmsConfig};

/// Prediction settings; the defaults are MODEL.md §9.3's `predict` column at `imgsz = 640`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictConfig {
    /// Target size `S` of the letterbox (a multiple of 32).
    pub imgsz: u32,
    /// Score threshold.
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Detections kept per image.
    pub max_det: usize,
    /// Pad only to the next multiple of 32 (`rect`, the default) instead of `S × S`: single images, and batches
    /// whose images all get the same such canvas ([`super::Predictor::predict_batch`]).
    pub rect: bool,
    /// The head branch to predict with (docs/MODEL.md §6.6): `None` (the default) uses the model's
    /// ([`crate::FoloModel::head_mode`]: `o2o`, no NMS, for a `head = "o2o"` model), `o2m` forces the one-to-many
    /// branch and NMS, `o2o` requires a model with the one-to-one branch.
    pub head: Option<HeadMode>,
    /// Refine each kept box from its NMS cluster ([`NmsConfig::fusion`]; off by default).
    pub fusion: BoxFusion,
    /// The cluster IoU of [`PredictConfig::fusion`] ([`NmsConfig::fusion_iou`]).
    pub fusion_iou: f32,
    /// How the kept boxes are scored ([`NmsConfig::score`]; off by default). `elc`, `dfl-consensus` and
    /// `dfl-repulse` preserve the `[0, 1]` score scale (pulling supporting anchors down to `0.10` before filtering
    /// final scores by `conf`); `consensus` rescales scores by up to `1 / √10`.
    pub score: ScoreRule,
}

impl Default for PredictConfig {
    fn default() -> Self {
        PredictConfig {
            imgsz: 640,
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
            rect: true,
            head: None,
            fusion: BoxFusion::None,
            fusion_iou: FUSION_IOU,
            score: ScoreRule::None,
        }
    }
}

impl PredictConfig {
    pub(super) fn validate(&self) -> Result<()> {
        if self.imgsz == 0 || !self.imgsz.is_multiple_of(32) || self.imgsz > 8192 {
            return Err(Error::Config(format!(
                "imgsz must be a multiple of 32 in 32..=8192, got {}",
                self.imgsz
            )));
        }
        if !(0.0..=1.0).contains(&self.conf) || !(0.0..=1.0).contains(&self.iou) {
            return Err(Error::Config(format!(
                "conf and iou must be in 0..=1, got {} and {}",
                self.conf, self.iou
            )));
        }
        if !(0.0..=1.0).contains(&self.fusion_iou) {
            return Err(Error::Config(format!("fusion_iou must be in 0..=1, got {}", self.fusion_iou)));
        }
        if self.max_det == 0 {
            return Err(Error::Config("max_det must be at least 1".into()));
        }
        Ok(())
    }

    pub(super) fn nms(&self) -> NmsConfig {
        NmsConfig {
            conf: self.conf,
            iou: self.iou,
            max_det: self.max_det,
            fusion: self.fusion,
            fusion_iou: self.fusion_iou,
            score: self.score,
            ..NmsConfig::default()
        }
    }
}

/// Wall time of a prediction in milliseconds (docs/API.md `timing`). On a GPU every stage includes the device
/// work it queued: the predictor waits for the device at the end of each stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    /// Letterbox and conversion to the input tensor (on a GPU: the upload and the letterbox kernel).
    pub preprocess_ms: f64,
    /// Forward pass, until the device has finished it.
    pub inference_ms: f64,
    /// Decode of the head outputs, readback, NMS and rescale.
    pub postprocess_ms: f64,
    /// Sum of the above.
    pub total_ms: f64,
}

impl Timing {
    /// A timing from its three parts (`total_ms` is their sum).
    pub fn new(pre: f64, inf: f64, post: f64) -> Self {
        Timing { preprocess_ms: pre, inference_ms: inf, postprocess_ms: post, total_ms: pre + inf + post }
    }

    /// docs/API.md's `Timing`.
    pub fn to_api(self) -> folo_api::Timing {
        folo_api::Timing {
            preprocess_ms: self.preprocess_ms,
            inference_ms: self.inference_ms,
            postprocess_ms: self.postprocess_ms,
            total_ms: self.total_ms,
        }
    }
}

/// The detections of one image.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prediction {
    /// Original width.
    pub width: u32,
    /// Original height.
    pub height: u32,
    /// Detections in original-image pixels, highest score first.
    pub detections: Vec<Detection>,
    /// Wall time (for a batch: the batch's time divided by its size).
    pub timing: Timing,
}

impl Prediction {
    /// The detections as docs/API.md `ScoredBox`es, naming each class from `classes`.
    pub fn scored_boxes(&self, classes: &[String]) -> Vec<folo_api::ScoredBox> {
        self.detections
            .iter()
            .map(|d| folo_api::ScoredBox {
                class: d.class,
                name: classes.get(d.class as usize).cloned().unwrap_or_else(|| format!("class{}", d.class)),
                score: d.score,
                x1: d.bbox.x1,
                y1: d.bbox.y1,
                x2: d.bbox.x2,
                y2: d.bbox.y2,
            })
            .collect()
    }
}

/// The numeric precision a [`super::Predictor`] runs in (docs/API.md `precision`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// f32 weights and activations (every device).
    #[default]
    F32,
    /// f16 weights and activations with f32 accumulation (a GPU with `shader-f16`).
    F16,
}

impl Precision {
    /// `"f32"` or `"f16"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
        }
    }

    /// The tensor dtype of the weights and activations.
    pub fn dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Precision {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" | "float32" => Ok(Precision::F32),
            "f16" | "fp16" | "half" | "float16" => Ok(Precision::F16),
            other => Err(Error::Config(format!("precision must be f32 or f16, got `{other}`"))),
        }
    }
}

/// Why f16 inference cannot run on `device` (so it runs in f32), or `None` when it can: it needs a GPU whose
/// backend computes in f16 (`shader-f16`, not turned off with `FOLO_GPU_NO_F16`).
pub fn f16_unavailable(device: Device) -> Option<String> {
    if device.is_cpu() {
        return Some("f16 inference needs a GPU; running in f32 on the CPU".into());
    }
    (!folo_core::backend::computes_f16(device)).then(|| {
        format!(
            "{device} has no f16 compute (the adapter lacks shader-f16, or FOLO_GPU_NO_F16 is set); running in f32"
        )
    })
}
