//! Validation error diagnostics, calibrated threshold helpers, prediction workload guards, and independent ONNX
//! dynamic-axis helpers shared across [`Store`](crate::Store) implementations, HTTP routes, and MCP proxy forwarding.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use folo_api::{ExportFormat, ExportSpec, ValReport};

use crate::error::{Error, Result};

pub use folo_vision::eval::{
    ClassThreshold, ImageDiagnostics, OperatingPoint, ThresholdCalibration, validate_per_class_conf,
};
pub use folo_vision::export::onnx::DynamicAxes;
pub use folo_vision::tta::DEFAULT_TTA_SCALES;
pub use folo_vision::{PredictOptions, TileConfig, TileMergeMetric, TtaConfig};

/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/predict` honours optional
/// `tile` (SAHI) and `tta` (Test-Time Augmentation) fields.
pub const FEATURE_PREDICT_OPTIONS: &str = "predict_options";
/// Server capability flag advertised on `/api/health` and `/api/system` when `/api/bench` honours client-chosen
/// `?entry_id=<id>` on `POST .../entries` and conditional `?only_if_open=true` on `POST .../entries/{id}/stop`.
pub const FEATURE_BENCH_UPLOAD_SAFE: &str = "bench_upload_safe";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/val` honours optional
/// per-image error diagnostics (`diagnostics`, `worst`, `diag_conf`, `diag_tp_iou`, `diag_bg_iou`).
pub const FEATURE_VAL_DIAGNOSTICS: &str = "val_diagnostics";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/models/{id}/export`
/// honours independent `dynamic_batch` and `dynamic_hw` ONNX axes.
pub const FEATURE_EXPORT_DYNAMIC_AXES: &str = "export_dynamic_axes";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/runs` honours GPU
/// VRAM-budget auto-batch sizing (`batch: "auto"` / `auto_batch: true` and `auto_batch_target`).
pub const FEATURE_TRAIN_AUTO_BATCH: &str = "train_auto_batch";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/bench/benchmarks`
/// honours deterministic generated-image sources (`source: { kind: "generated", ... }`).
pub const FEATURE_BENCH_GENERATED_SOURCE: &str = "bench_generated_source";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/runs` and
/// `POST /api/runs/estimate` accept FOLO v2-Iso architectures (`folo2-n`, `folo2-s`, `folo2-m`, `folo2-l`, `folo2-x`).
pub const FEATURE_FOLO2_ARCH: &str = "folo2_arch";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/val` returns calibrated
/// per-class F1 thresholds (`thresholds`) and `POST /api/predict` honours `per_class_conf` / `conf_inclusive`.
pub const FEATURE_THRESHOLD_CALIBRATION: &str = "threshold_calibration";
/// Server capability flag advertised on `/api/health` and `/api/system` when `POST /api/runs` honours per-run
/// cloud spot preemption retry limits (`max_preempt_retries`) and `GET /api/runs/{id}` reports `cloud_preemption`.
pub const FEATURE_CLOUD_SPOT_RESUME: &str = "cloud_spot_resume";

/// Default maximum automatic spot preemption recovery attempts per cloud run.
pub const DEFAULT_MAX_PREEMPT_RETRIES: u32 = 3;
/// Upper bound on per-run `max_preempt_retries` (`0..=20`, where `0` disables automatic preemption recovery).
pub const MAX_PREEMPT_RETRIES: u32 = 20;

/// Maximum SAHI tiles permitted in a single server/MCP `predict` call (prevents resource exhaustion on tiny tiles).
pub const MAX_PREDICT_TILES: usize = 256;
/// Maximum cropped tiles predicted in a single batch in server/MCP `predict`.
pub const MAX_PREDICT_TILE_BATCH: usize = 32;
/// Maximum peak RGB crop bytes allocated per tiled batch chunk in server/MCP `predict` (256 MiB).
pub const MAX_PREDICT_CROP_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum TTA views permitted in a single server/MCP `predict` call (`unique_scales * (hflip ? 2 : 1)`).
pub const MAX_PREDICT_TTA_VIEWS: usize = 16;

/// Validates `options` independent of a specific model `imgsz` or image dimensions (`400 invalid_body` on invalid
/// ranges, exceeding per-request batch/scale caps, or combining `tile` with `tta`).
pub fn check_predict_options(options: &PredictOptions) -> Result<()> {
    fn bad(field: &str, msg: impl std::fmt::Display) -> Error {
        Error::new(StatusCode::BAD_REQUEST, "invalid_body", format!("field `{field}`: {msg}"))
    }
    if options.tile.is_some() && options.tta.is_some() {
        return Err(bad("tile", "cannot be combined with tta"));
    }
    if let Some(tile) = &options.tile {
        tile.validate().map_err(|e| bad("tile", e))?;
        if tile.tile_batch > MAX_PREDICT_TILE_BATCH {
            return Err(bad(
                "tile",
                format!("tile_batch must be at most {MAX_PREDICT_TILE_BATCH}, got {}", tile.tile_batch),
            ));
        }
    }
    if let Some(tta) = &options.tta {
        tta.validate().map_err(|e| bad("tta", e))?;
        if tta.scales.len() > MAX_PREDICT_TTA_VIEWS {
            return Err(bad(
                "tta",
                format!("at most {MAX_PREDICT_TTA_VIEWS} scales are allowed, got {}", tta.scales.len()),
            ));
        }
    }
    Ok(())
}

/// Tile count along one image axis matching `folo_vision::tile::compute_tiles` without allocating tile rects.
fn axis_tile_count(img_len: u32, tile_len: u32, overlap: f32) -> usize {
    let tw = tile_len.min(img_len);
    if tw == img_len || tw == 0 {
        return 1;
    }
    let overlap_px = ((tw as f64) * (overlap as f64)).round() as u32;
    let stride = tw.saturating_sub(overlap_px.min(tw - 1)).max(1);
    let span = img_len - tw;
    (span as usize).div_ceil(stride as usize) + 1
}

/// Validates `options` against `model_imgsz` and (when known) image `(width, height)`, enforcing server-side SAHI
/// tile-count and TTA view-count budgets before locking the predictor or allocating tile grids.
pub fn check_predict_workload(
    options: &PredictOptions,
    model_imgsz: u32,
    image_size: Option<(u32, u32)>,
) -> Result<()> {
    let bad = |msg: String, hint: &'static str| {
        Error::new(StatusCode::BAD_REQUEST, "invalid_config", msg).hint(hint)
    };
    options.validate(model_imgsz).map_err(|e| {
        bad(e.to_string(), "check `tile` / `tta` settings against the model's input size (32..=8192)")
    })?;
    if let (Some(tile), Some((w, h))) = (&options.tile, image_size) {
        let nx = axis_tile_count(w, tile.tile_width, tile.overlap_x);
        let ny = axis_tile_count(h, tile.tile_height, tile.overlap_y);
        let tiles = nx.saturating_mul(ny);
        if tiles > MAX_PREDICT_TILES {
            return Err(bad(
                format!(
                    "tiled prediction on {w}x{h} with {}x{} tiles produces {tiles} tiles, exceeding the server limit of {MAX_PREDICT_TILES}",
                    tile.tile_width, tile.tile_height
                ),
                "use a larger tile size or lower overlap",
            ));
        }
        let crop_w = tile.tile_width.min(w) as u64;
        let crop_h = tile.tile_height.min(h) as u64;
        let chunk = tiles.min(tile.tile_batch) as u64;
        let crop_bytes = chunk.saturating_mul(crop_w).saturating_mul(crop_h).saturating_mul(3);
        if crop_bytes > MAX_PREDICT_CROP_BYTES {
            return Err(bad(
                format!(
                    "tiled prediction on {w}x{h} with {}x{} tiles and tile_batch {} needs {:.1} MiB of crop buffers per batch chunk, exceeding the server limit of {} MiB",
                    tile.tile_width,
                    tile.tile_height,
                    tile.tile_batch,
                    crop_bytes as f64 / (1024.0 * 1024.0),
                    MAX_PREDICT_CROP_BYTES / (1024 * 1024),
                ),
                "reduce `tile_batch` or use a smaller tile size",
            ));
        }
    }
    if let Some(tta) = &options.tta {
        let sizes = tta.resolve_imgszs(model_imgsz).map_err(|e| {
            bad(e.to_string(), "check `tta.scales` against the model's input size (32..=8192)")
        })?;
        let views = sizes.len().saturating_mul(if tta.hflip { 2 } else { 1 });
        if views > MAX_PREDICT_TTA_VIEWS {
            return Err(bad(
                format!("TTA produces {views} views, exceeding the server limit of {MAX_PREDICT_TTA_VIEWS}"),
                "pass fewer scales or set `hflip: false`",
            ));
        }
    }
    Ok(())
}

/// Validates optional per-class confidence thresholds (`400 invalid_body` on non-finite or out-of-range values;
/// `400 invalid_config` when `num_classes` is supplied and a class id is `>= num_classes`).
pub fn check_per_class_conf(map: &BTreeMap<u32, f32>, num_classes: Option<usize>) -> Result<()> {
    validate_per_class_conf(map).map_err(|e| {
        Error::new(StatusCode::BAD_REQUEST, "invalid_body", format!("field `per_class_conf`: {e}"))
    })?;
    if let Some(nc) = num_classes {
        for &class in map.keys() {
            if (class as usize) >= nc {
                return Err(Error::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_config",
                    format!(
                        "per_class_conf: class {class} is out of range for model with {nc} classes (0..{nc})"
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Extended validation output returned by [`crate::Store::val_extended`]: the core [`ValReport`], optional
/// per-image error diagnostics, and calibrated per-class F1 thresholds.
#[derive(Clone, Debug)]
pub struct ValExtended {
    /// Core validation report.
    pub report: ValReport,
    /// Optional per-image error diagnostics when requested.
    pub diagnostics: Option<ValDiagnostics>,
    /// Optional calibrated global and per-class F1 thresholds.
    pub thresholds: Option<ThresholdCalibration>,
}

/// Extra run options carried alongside [`folo_api::TrainConfig`] on `POST /api/runs`, `POST /api/runs/estimate`,
/// and the MCP `train` tool without altering `folo-api` wire types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunOptions {
    /// Opt-in GPU VRAM-budget auto-batch settings.
    pub auto_batch: AutoBatchOptions,
    /// Optional architecture override when training a FOLO v2-Iso model (`"folo2-n"`..`"folo2-x"`).
    pub arch: Option<String>,
    /// Optional per-run override for cloud spot/preemption auto-resume attempts (`0..=20`, cloud runs only).
    pub max_preempt_retries: Option<u32>,
}

impl RunOptions {
    /// True when no extra run options (auto-batch, `folo2-*` arch, or `max_preempt_retries`) are set.
    pub fn is_default(&self) -> bool {
        self.auto_batch.is_default() && self.arch.is_none() && self.max_preempt_retries.is_none()
    }
}

/// Spot preemption auto-resume status of a cloud training run, reported alongside [`folo_api::Run`] on
/// `GET /api/runs/{id}` and MCP `run_status` without altering `folo-api` wire types.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CloudPreemptionStatus {
    /// Automatic preemption recovery attempts already started for this run.
    pub retries: u32,
    /// Configured maximum automatic preemption recovery attempts for this run (`0` disables auto-resume).
    pub max_retries: u32,
    /// True while the run has released a preempted worker and is waiting to provision its replacement.
    pub recovering: bool,
    /// Ledger ids of earlier instances replaced after preemption, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preempted_instances: Vec<String>,
}

/// Validates `max_preempt_retries` (`0..=20`, cloud compute only; `400 invalid_config` otherwise).
pub fn check_max_preempt_retries(retries: Option<u32>, has_compute: bool) -> Result<()> {
    let Some(n) = retries else { return Ok(()) };
    if !has_compute {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            "max_preempt_retries is only valid for cloud runs (set `compute` alongside it)",
        )
        .hint("omit `max_preempt_retries` for local training, or pass `compute` to train on cloud"));
    }
    if n > MAX_PREEMPT_RETRIES {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            format!("max_preempt_retries must be in 0..={MAX_PREEMPT_RETRIES}, got {n}"),
        )
        .hint("pick a value in 0..=20 (0 disables automatic preemption recovery)"));
    }
    Ok(())
}

/// Parses an architecture name accepted by `POST /api/runs`, `POST /api/runs/estimate`, and MCP `train`:
/// `folo-n..x` maps to `(Arch, None)`, `folo2-n..x` maps to `(Arch, Some("folo2-<scale>"))`, and any other string
/// returns `422 invalid_body`.
pub fn parse_arch_override(raw: &str) -> Result<(folo_api::Arch, Option<String>)> {
    if let Some(arch) = folo_api::Arch::parse(raw) {
        return Ok((arch, None));
    }
    if let Some(scale) = raw.strip_prefix("folo2-")
        && let Some(arch) = folo_api::Arch::parse(&format!("folo-{scale}"))
    {
        return Ok((arch, Some(raw.to_string())));
    }
    Err(Error::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
        format!(
            "arch: expected one of folo-n, folo-s, folo-m, folo-l, folo-x, folo2-n, folo2-s, folo2-m, folo2-l, \
             folo2-x, got {raw:?}"
        ),
    ))
}

/// Validates an optional architecture override (`400 invalid_config` on unknown names) and returns the matching
/// scale [`folo_api::Arch`].
pub fn check_arch_override(arch: Option<&str>) -> Result<Option<folo_api::Arch>> {
    let Some(raw) = arch else { return Ok(None) };
    parse_arch_override(raw).map(|(a, _)| Some(a)).map_err(|_| {
        Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            format!("arch: unknown architecture {raw:?} (expected folo-n..folo-x or folo2-n..folo2-x)"),
        )
    })
}

/// The effective architecture string of a run: `arch_override` when set (e.g. `"folo2-n"`), else `arch.as_str()`.
pub fn effective_arch(arch: folo_api::Arch, arch_override: Option<&str>) -> &str {
    arch_override.unwrap_or_else(|| arch.as_str())
}

/// Default run name for an explicit architecture string (`shapes-n` for `folo-n`, `shapes-folo2-n` for `folo2-n`,
/// matching `folo train`), truncated to [`crate::store::MAX_RUN_NAME_CHARS`].
pub fn default_run_name_for_arch(dataset: &str, arch: &str) -> String {
    let size = arch.strip_prefix("folo-").unwrap_or(arch);
    let room = crate::store::MAX_RUN_NAME_CHARS.saturating_sub(size.chars().count() + 1);
    let base: String = dataset.chars().take(room).collect();
    format!("{}-{size}", base.trim_end())
}

/// Opt-in GPU VRAM-budget auto-batch settings carried alongside [`folo_api::TrainConfig`] on `POST /api/runs` and
/// the MCP `train` tool.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AutoBatchOptions {
    /// Whether GPU VRAM-budget auto-batch sizing (`folo_vision::TrainConfig::auto_batch`) is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Optional target VRAM utilization fraction in `0.25..=0.95` (`None` = engine default `0.80`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_utilization: Option<f64>,
}

impl AutoBatchOptions {
    /// True when auto-batch is disabled and no target utilization is set.
    pub fn is_default(&self) -> bool {
        !self.enabled && self.target_utilization.is_none()
    }
}

/// Validates [`AutoBatchOptions`] (`400 invalid_config` on out-of-range `target_utilization` or setting
/// `target_utilization` without enabling auto-batch).
pub fn check_auto_batch_options(opts: &AutoBatchOptions) -> Result<()> {
    let bad = |msg: String| Error::new(StatusCode::BAD_REQUEST, "invalid_config", msg);
    if let Some(target) = opts.target_utilization {
        let min = folo_vision::train::MIN_AUTO_BATCH_UTILIZATION;
        let max = folo_vision::train::MAX_AUTO_BATCH_UTILIZATION;
        if !(target.is_finite() && (min..=max).contains(&target)) {
            return Err(bad(format!("auto_batch_target must be finite in {min}..={max}, got {target}")));
        }
        if !opts.enabled {
            return Err(bad(
                "auto_batch_target requires auto-batch sizing (`batch: \"auto\"` or `auto_batch: true`)"
                    .into(),
            ));
        }
    }
    Ok(())
}

/// Default number of worst failing images returned when validation diagnostics are enabled.
pub const DEFAULT_VAL_DIAG_WORST: usize = 10;
/// Maximum `worst` images returned in a single server/MCP validation diagnostics response.
pub const MAX_VAL_DIAG_WORST: usize = 50;
/// Default confidence threshold for per-image error diagnosis.
pub const DEFAULT_VAL_DIAG_CONF: f32 = 0.25;
/// Default TP IoU threshold for per-image error diagnosis.
pub const DEFAULT_VAL_DIAG_TP_IOU: f32 = 0.50;
/// Default background IoU threshold for per-image error diagnosis.
pub const DEFAULT_VAL_DIAG_BG_IOU: f32 = 0.10;

/// Settings for optional per-image detection error diagnosis during validation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValDiagConfig {
    /// Maximum number of worst failing images to return (`1..=50`).
    pub worst: usize,
    /// Confidence threshold for per-image diagnosis (`0..=1`).
    pub conf: f32,
    /// IoU threshold for a true positive (`0..=1`).
    pub tp_iou: f32,
    /// IoU threshold below which a false positive is classified as background (`0..=tp_iou`).
    pub bg_iou: f32,
}

impl Default for ValDiagConfig {
    fn default() -> Self {
        Self {
            worst: DEFAULT_VAL_DIAG_WORST,
            conf: DEFAULT_VAL_DIAG_CONF,
            tp_iou: DEFAULT_VAL_DIAG_TP_IOU,
            bg_iou: DEFAULT_VAL_DIAG_BG_IOU,
        }
    }
}

/// Aggregated error breakdown across all diagnosed validation images.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ValDiagnosticsSummary {
    /// Total ground-truth boxes across diagnosed images.
    pub num_truths: u64,
    /// Total detections above `conf` across diagnosed images.
    pub num_preds: u64,
    /// Total true positives.
    pub tp: u64,
    /// Total false positives.
    pub fp: u64,
    /// Total missed ground-truth objects (false negatives).
    pub fn_count: u64,
    /// Correct-class detections with `bg_iou <= IoU < tp_iou`.
    pub localization_errors: u64,
    /// Wrong-class detections overlapping a ground-truth box at `IoU >= bg_iou`.
    pub class_confusions: u64,
    /// Duplicate correct-class detections overlapping an already-matched ground-truth box at `IoU >= tp_iou`.
    pub duplicate_errors: u64,
    /// False positives with `IoU < bg_iou` against every ground-truth box.
    pub background_fp: u64,
    /// Mean per-image F1 across diagnosed images.
    pub mean_f1: f32,
    /// Mean IoU over all true positives (`0.0` when `tp == 0`).
    pub mean_tp_iou: f32,
}

/// Optional per-image error diagnosis payload returned alongside [`folo_api::ValReport`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ValDiagnostics {
    /// Confidence threshold used for diagnosis.
    pub conf: f32,
    /// TP IoU threshold used for diagnosis.
    pub tp_iou: f32,
    /// Background IoU threshold used for diagnosis.
    pub bg_iou: f32,
    /// Number of images diagnosed.
    pub images_diagnosed: u64,
    /// Number of images with at least one FP or FN error.
    pub failing_images: u64,
    /// Aggregated counts and averages across all diagnosed images.
    pub summary: ValDiagnosticsSummary,
    /// Worst failing images ranked by total errors descending (up to `worst`).
    pub worst_images: Vec<ImageDiagnostics>,
}

/// Validates [`ValDiagConfig`] (`400 invalid_body` on out-of-range values).
pub fn check_val_diag_config(cfg: &ValDiagConfig) -> Result<()> {
    let bad = |msg: String| Error::new(StatusCode::BAD_REQUEST, "invalid_body", msg);
    if !(1..=MAX_VAL_DIAG_WORST).contains(&cfg.worst) {
        return Err(bad(format!("worst must be in 1..={MAX_VAL_DIAG_WORST}, got {}", cfg.worst)));
    }
    let unit = |x: f32| (0.0..=1.0).contains(&x);
    if !unit(cfg.conf) || !unit(cfg.tp_iou) || !unit(cfg.bg_iou) {
        return Err(bad(format!(
            "diag_conf, diag_tp_iou and diag_bg_iou must be in 0..=1, got conf {} tp_iou {} bg_iou {}",
            cfg.conf, cfg.tp_iou, cfg.bg_iou
        )));
    }
    if cfg.bg_iou > cfg.tp_iou {
        return Err(bad(format!(
            "diag_bg_iou ({}) must not exceed diag_tp_iou ({})",
            cfg.bg_iou, cfg.tp_iou
        )));
    }
    Ok(())
}

/// Validates an export request with independent ONNX dynamic axes (`400 invalid_body` if ONNX options or dynamic
/// axes are given for a non-ONNX format).
pub fn check_export_axes(spec: &ExportSpec, axes: DynamicAxes) -> Result<()> {
    crate::store::check_export(spec)?;
    if spec.format != ExportFormat::Onnx && axes.is_any() {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            "dynamic_batch and dynamic_hw only apply to format `onnx`",
        ));
    }
    Ok(())
}
