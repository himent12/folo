//! Configuration and result types for [`super::Detector`].

use folo_vision::fusion::BoxFusion;
use folo_vision::tta::DEFAULT_TTA_SCALES;
use folo_vision::{Precision, TileConfig, TileMergeMetric, TtaConfig};
use serde::{Deserialize, Serialize};

use super::err;

/// Which backend [`super::Detector::load`] should use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendChoice {
    /// WebGPU when it initialises and runs, otherwise the CPU.
    #[default]
    Auto,
    /// WebGPU or an error.
    Webgpu,
    /// The wasm CPU backend.
    Cpu,
}

/// The backend a [`super::Detector`] runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// folo-gpu on WebGPU (natively: Vulkan / DX12 / Metal).
    Webgpu,
    /// folo-core's CPU kernels (vectorised to wasm SIMD128 in browser builds).
    Cpu,
}

impl Backend {
    /// `"webgpu"` or `"cpu"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Webgpu => "webgpu",
            Backend::Cpu => "cpu",
        }
    }
}

/// [`super::Detector::load`] options (the JS `load` options object; unknown fields are ignored).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct LoadOptions {
    /// `auto` (default), `webgpu` or `cpu`.
    pub backend: BackendChoice,
    /// Echoed as `model_id` in every result (docs/API.md); empty by default.
    pub model_id: Option<String>,
    /// Letterbox size; defaults to the file's `folo.imgsz`.
    pub imgsz: Option<u32>,
    /// `f32` (default) or `f16`: half precision on WebGPU with `shader-f16`; f32 elsewhere (the CPU, an adapter
    /// without it), with the reason in [`DetectorInfo::precision_fallback`].
    pub precision: Precision,
}

/// Per-call options (the JS `detect` options object), with docs/API.md's `/api/predict` defaults.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DetectOptions {
    /// Score threshold in `0..=1`.
    pub conf: f32,
    /// NMS IoU threshold in `0..=1`.
    pub iou: f32,
    /// Detections kept, at least 1.
    pub max_det: usize,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions { conf: 0.25, iou: 0.7, max_det: 300 }
    }
}

/// Per-call options for SAHI tiled inference ([`super::Detector::detect_tiled`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TileDetectOptions {
    /// Score threshold in `0..=1` (default `0.25`).
    pub conf: f32,
    /// Per-tile NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per tile and after cross-tile merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Convenience square tile size in pixels (`tileWidth` / `tileHeight` override it when set; default `640`).
    pub tile_size: Option<u32>,
    /// Target tile width in pixels (defaults to `tileSize`, or `640`).
    pub tile_width: Option<u32>,
    /// Target tile height in pixels (defaults to `tileSize`, or `640`).
    pub tile_height: Option<u32>,
    /// Convenience symmetric overlap ratio in `[0, 1)` (`overlapX` / `overlapY` override it; default `0.2`).
    pub overlap: Option<f32>,
    /// Horizontal overlap ratio between adjacent tiles in `[0, 1)`.
    pub overlap_x: Option<f32>,
    /// Vertical overlap ratio between adjacent tiles in `[0, 1)`.
    pub overlap_y: Option<f32>,
    /// Also run a full-image pass alongside the tiles to preserve large boundary-spanning objects (default `true`).
    #[serde(alias = "fullImagePass")]
    pub full_image: bool,
    /// Maximum cropped tiles predicted per WebGPU batch, at least `1` (default `8`).
    pub tile_batch: usize,
    /// Cross-tile merge overlap metric: `"ios"` (default) or `"iou"`.
    pub merge_metric: TileMergeMetric,
    /// Cross-tile merge overlap threshold in `0..=1` (default `0.5`; also accepts JS alias `mergeIou`).
    #[serde(alias = "mergeIou")]
    pub merge_threshold: f32,
    /// Cross-tile box fusion rule: `"vote"` (default), `"none"`, `"dfl"`, or `"sebf"`.
    pub merge_fusion: BoxFusion,
    /// Merge overlapping detections across different classes (default `false`).
    pub class_agnostic: bool,
}

impl Default for TileDetectOptions {
    fn default() -> Self {
        let def = TileConfig::default();
        TileDetectOptions {
            conf: 0.25,
            iou: 0.7,
            max_det: def.max_det,
            tile_size: None,
            tile_width: None,
            tile_height: None,
            overlap: None,
            overlap_x: None,
            overlap_y: None,
            full_image: def.full_image,
            tile_batch: def.tile_batch,
            merge_metric: def.merge_metric,
            merge_threshold: def.merge_threshold,
            merge_fusion: def.merge_fusion,
            class_agnostic: def.class_agnostic,
        }
    }
}

impl TileDetectOptions {
    /// Validate and resolve into a [`TileConfig`].
    pub fn to_tile_config(&self) -> Result<TileConfig, String> {
        if let Some(0) = self.tile_size {
            return Err("tile_size must be positive, got 0".into());
        }
        if let Some(o) = self.overlap
            && (!o.is_finite() || !(0.0..1.0).contains(&o))
        {
            return Err(format!("tile overlap must be finite and in [0, 1), got {o}"));
        }
        let def = TileConfig::default();
        let tile_cfg = TileConfig {
            tile_width: self.tile_width.unwrap_or_else(|| self.tile_size.unwrap_or(def.tile_width)),
            tile_height: self.tile_height.unwrap_or_else(|| self.tile_size.unwrap_or(def.tile_height)),
            overlap_x: self.overlap_x.unwrap_or_else(|| self.overlap.unwrap_or(def.overlap_x)),
            overlap_y: self.overlap_y.unwrap_or_else(|| self.overlap.unwrap_or(def.overlap_y)),
            full_image: self.full_image,
            tile_batch: self.tile_batch,
            merge_metric: self.merge_metric,
            merge_threshold: self.merge_threshold,
            merge_fusion: self.merge_fusion,
            class_agnostic: self.class_agnostic,
            max_det: self.max_det,
        };
        tile_cfg.validate().map_err(err)?;
        Ok(tile_cfg)
    }
}

/// Per-call options for Test-Time Augmentation ([`super::Detector::detect_tta`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TtaDetectOptions {
    /// Score threshold in `0..=1` (default `0.25`).
    pub conf: f32,
    /// Per-view NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per view and after cross-view merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Also evaluate a horizontally flipped mirror at each scale (default `true`).
    pub hflip: bool,
    /// Multi-scale multipliers applied to the detector's `imgsz` (default `[0.83, 1.0, 1.17]`).
    pub scales: Vec<f32>,
    /// Cross-view merge IoU threshold in `0..=1` (defaults to `iou` when omitted).
    pub merge_iou: Option<f32>,
    /// Cross-view box fusion rule: `"vote"` (default), `"none"`, `"dfl"`, or `"sebf"`.
    pub merge_fusion: BoxFusion,
    /// Merge overlapping detections across different classes (default `false`).
    pub class_agnostic: bool,
}

impl Default for TtaDetectOptions {
    fn default() -> Self {
        let def = TtaConfig::default();
        TtaDetectOptions {
            conf: 0.25,
            iou: 0.7,
            max_det: 300,
            hflip: def.hflip,
            scales: DEFAULT_TTA_SCALES.to_vec(),
            merge_iou: def.merge_iou,
            merge_fusion: def.merge_fusion,
            class_agnostic: def.class_agnostic,
        }
    }
}

impl TtaDetectOptions {
    /// Validate and resolve into a [`TtaConfig`].
    pub fn to_tta_config(&self) -> Result<TtaConfig, String> {
        let tta_cfg = TtaConfig {
            hflip: self.hflip,
            scales: self.scales.clone(),
            merge_iou: self.merge_iou,
            merge_fusion: self.merge_fusion,
            class_agnostic: self.class_agnostic,
            max_det: Some(self.max_det),
        };
        tta_cfg.validate().map_err(err)?;
        Ok(tta_cfg)
    }
}

/// What [`super::Detector::info`] reports.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectorInfo {
    /// `webgpu` or `cpu`.
    pub backend: Backend,
    /// docs/API.md's device string: `gpu:0` or `cpu`.
    pub device: String,
    /// The GPU adapter's name (WebGPU only; `None` when the browser does not disclose it, as Chrome does).
    pub adapter: Option<String>,
    /// Why `auto` fell back to the CPU, if it did.
    pub fallback: Option<String>,
    /// The precision the model runs in: `f32` or `f16`.
    pub precision: Precision,
    /// Why a requested `f16` runs in `f32`, if it does.
    pub precision_fallback: Option<String>,
    /// Whether the CPU kernels were built with wasm SIMD128.
    pub simd128: bool,
    /// CPU threads (1: the wasm build is single-threaded).
    pub threads: usize,
    /// Model family: `"folo"` (v1) or `"folo2"` (v2-Iso).
    pub family: String,
    /// Model family and scale, e.g. `folo-n` or `folo2-n`.
    pub arch: String,
    /// Active evaluation head mode: `"o2m"` (NMS) or `"o2o"` (NMS-free top-k).
    pub head_mode: String,
    /// Whether the loaded model includes a one-to-one (`o2o`) head branch.
    pub has_o2o: bool,
    /// Whether the loaded FOLO v2 model includes the `iou_head` localization-quality branch.
    pub has_iou_head: bool,
    /// Letterbox size.
    pub imgsz: u32,
    /// Class names, indexed by class id.
    pub classes: Vec<String>,
    /// The `model_id` echoed in results.
    pub model_id: String,
}

/// A detection result: docs/API.md's `/api/predict` response plus the backend that produced it.
#[derive(Clone, Debug, Serialize)]
pub struct DetectResult {
    /// The `/api/predict` fields.
    #[serde(flatten)]
    pub response: folo_api::PredictResponse,
    /// `webgpu` or `cpu`.
    pub backend: Backend,
}
