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
