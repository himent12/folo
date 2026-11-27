//! Validation error diagnostics and independent ONNX dynamic-axis helpers shared across [`Store`](crate::Store)
//! implementations, HTTP routes, and MCP proxy forwarding.

use axum::http::StatusCode;
use folo_api::{ExportFormat, ExportSpec};

use crate::error::{Error, Result};

pub use folo_vision::eval::ImageDiagnostics;
pub use folo_vision::export::onnx::DynamicAxes;

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

/// Extra run options carried alongside [`folo_api::TrainConfig`] on `POST /api/runs`, `POST /api/runs/estimate`,
/// and the MCP `train` tool without altering `folo-api` wire types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunOptions {
    /// Opt-in GPU VRAM-budget auto-batch settings.
    pub auto_batch: AutoBatchOptions,
    /// Optional architecture override when training a FOLO v2-Iso model (`"folo2-n"`..`"folo2-x"`).
    pub arch: Option<String>,
}

impl RunOptions {
    /// True when neither auto-batch nor a `folo2-*` architecture override is set.
    pub fn is_default(&self) -> bool {
        self.auto_batch.is_default() && self.arch.is_none()
    }
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
