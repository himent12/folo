//! Configuration and result types for [`super::Detector`].

use std::collections::BTreeMap;
use std::fmt;

use folo_vision::eval::validate_per_class_conf;
use folo_vision::fusion::BoxFusion;
use folo_vision::nms::NmsConfig;
use folo_vision::tta::DEFAULT_TTA_SCALES;
use folo_vision::{PredictConfig, TileConfig, TileMergeMetric, TtaConfig};
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use super::err;

/// Numeric precision for browser inference (`FoloLoadOptions.precision` and `FoloDetectorInfo.precision`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// `f32` weights and activations (every backend).
    #[default]
    F32,
    /// `f16` weights and activations with `f32` accumulation (WebGPU with `shader-f16`; falls back to `f32`
    /// elsewhere).
    F16,
    /// Post-training `int8` quantized backbone & neck on the CPU (WASM SIMD128 / native SIMD) using the `.folo`
    /// file's `folo.quant` calibration metadata (`folo-*` and `folo2-*`).
    Int8,
}

impl Precision {
    /// `"f32"`, `"f16"`, or `"int8"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Int8 => "int8",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parse a canonical base-10 non-negative `u32` class ID string (`"0"` or `[1-9][0-9]*` fitting in `u32`).
/// Rejects signs (`"+0"`, `"-1"`), leading zeros (`"00"`, `"01"`), whitespace, decimals, and overflows.
pub(crate) fn parse_class_id_key(key: &str) -> Option<u32> {
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if key.len() > 1 && key.starts_with('0') {
        return None;
    }
    key.parse::<u32>().ok()
}

/// Strict `serde` deserializer for `per_class_conf`: accepts `null` as `None`, validates canonical `u32` keys,
/// and validates finite `[0, 1]` values in `f64` before narrowing to `f32`.
fn deserialize_per_class_conf<'de, D>(deserializer: D) -> Result<Option<BTreeMap<u32, f32>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct OptVisitor;

    impl<'de> Visitor<'de> for OptVisitor {
        type Value = Option<BTreeMap<u32, f32>>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("null or an object mapping non-negative integer class IDs to thresholds in [0, 1]")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
            struct MapVisitor;

            impl<'de> Visitor<'de> for MapVisitor {
                type Value = BTreeMap<u32, f32>;

                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("an object mapping non-negative integer class IDs to thresholds in [0, 1]")
                }

                fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                    let mut map = BTreeMap::new();
                    while let Some((k, v)) = access.next_entry::<String, f64>()? {
                        let id = parse_class_id_key(&k).ok_or_else(|| {
                            de::Error::custom(format!(
                                "perClassConf key `{k}` must be a non-negative integer class ID"
                            ))
                        })?;
                        if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                            return Err(de::Error::custom(format!(
                                "perClassConf[{k}] must be finite and in [0, 1], got {v}"
                            )));
                        }
                        map.insert(id, v as f32);
                    }
                    Ok(map)
                }
            }

            deserializer.deserialize_map(MapVisitor).map(Some)
        }
    }

    deserializer.deserialize_option(OptVisitor)
}

/// Converts a [`PredictConfig`] plus optional per-class thresholds and inclusive-threshold toggle into a
/// validated [`NmsConfig`].
pub(crate) fn nms_config(
    cfg: &PredictConfig,
    per_class_conf: Option<BTreeMap<u32, f32>>,
    conf_inclusive: bool,
) -> Result<NmsConfig, String> {
    let per_class_conf = match per_class_conf {
        Some(map) => {
            validate_per_class_conf(&map).map_err(err)?;
            (!map.is_empty()).then_some(map)
        }
        None => None,
    };
    Ok(NmsConfig {
        conf: cfg.conf,
        iou: cfg.iou,
        max_det: cfg.max_det,
        fusion: cfg.fusion,
        fusion_iou: cfg.fusion_iou,
        score: cfg.score,
        per_class_conf,
        conf_inclusive,
        ..NmsConfig::default()
    })
}

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
    /// Optional precision override (`f32`, `f16`, or `int8`). When omitted (`None`): runs `f32` on WebGPU, or
    /// `int8` on the CPU when the `.folo` file carries `folo.quant` calibration metadata (`f32` otherwise).
    /// Requesting `f16` uses half precision on WebGPU with `shader-f16` and falls back to `f32` elsewhere;
    /// requesting `int8` runs the quantized CPU backend (or falls back to `f32` when `backend` is explicitly
    /// `webgpu`), with any fallback reason in [`DetectorInfo::precision_fallback`].
    pub precision: Option<Precision>,
}

/// Per-call options (the JS `detect` options object), with docs/API.md's `/api/predict` defaults.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct DetectOptions {
    /// Score threshold in `0..=1` (fallback for classes absent from `per_class_conf`).
    pub conf: f32,
    /// NMS IoU threshold in `0..=1`.
    pub iou: f32,
    /// Detections kept, at least 1.
    pub max_det: usize,
    /// Optional per-class confidence thresholds (`class_id -> conf`, each finite in `0..=1`).
    #[serde(default, deserialize_with = "deserialize_per_class_conf")]
    pub per_class_conf: Option<BTreeMap<u32, f32>>,
    /// Use inclusive (`score >= conf`) thresholding matching `folo val`'s calibrated F1 operating points
    /// (default `false`: strict `score > conf`).
    pub conf_inclusive: bool,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions { conf: 0.25, iou: 0.7, max_det: 300, per_class_conf: None, conf_inclusive: false }
    }
}

/// Per-call options for SAHI tiled inference ([`super::Detector::detect_tiled`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TileDetectOptions {
    /// Score threshold in `0..=1` (default `0.25`; fallback for classes absent from `per_class_conf`).
    pub conf: f32,
    /// Per-tile NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per tile and after cross-tile merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Optional per-class confidence thresholds (`class_id -> conf`, each finite in `0..=1`).
    #[serde(default, deserialize_with = "deserialize_per_class_conf")]
    pub per_class_conf: Option<BTreeMap<u32, f32>>,
    /// Use inclusive (`score >= conf`) thresholding matching `folo val`'s calibrated F1 operating points
    /// (default `false`: strict `score > conf`).
    pub conf_inclusive: bool,
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
            per_class_conf: None,
            conf_inclusive: false,
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
    /// Score threshold in `0..=1` (default `0.25`; fallback for classes absent from `per_class_conf`).
    pub conf: f32,
    /// Per-view NMS IoU threshold in `0..=1` (default `0.7`).
    pub iou: f32,
    /// Maximum detections kept per view and after cross-view merging, at least `1` (default `300`).
    pub max_det: usize,
    /// Optional per-class confidence thresholds (`class_id -> conf`, each finite in `0..=1`).
    #[serde(default, deserialize_with = "deserialize_per_class_conf")]
    pub per_class_conf: Option<BTreeMap<u32, f32>>,
    /// Use inclusive (`score >= conf`) thresholding matching `folo val`'s calibrated F1 operating points
    /// (default `false`: strict `score > conf`).
    pub conf_inclusive: bool,
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
            per_class_conf: None,
            conf_inclusive: false,
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
    /// The precision the model runs in: `f32`, `f16`, or `int8`.
    pub precision: Precision,
    /// Why a requested or embedded precision runs in `f32`, if it does.
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
