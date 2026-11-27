//! The tools that use a model: list models, validate, predict (one image or a batch) and export.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use base64::Engine as _;
use folo_api::{ExportSpec, Model, OnnxSwitches, ValRequest};
use folo_server::store::{
    DEFAULT_TTA_SCALES, DEFAULT_VAL_DIAG_BG_IOU, DEFAULT_VAL_DIAG_CONF, DEFAULT_VAL_DIAG_TP_IOU,
    DEFAULT_VAL_DIAG_WORST, DynamicAxes, MAX_VAL_DIAG_WORST, PredictInput, PredictOptions, PredictSource,
    ThresholdCalibration, TileConfig, TtaConfig, ValDiagConfig, ValDiagnostics, check_export_axes,
    check_per_class_conf, check_predict_options, check_predict_workload, check_val_diag_config,
};
use rmcp::model::{ContentBlock, Resource};
use serde_json::json;

use super::Progress;
use super::arguments::check_unit;
use super::dispatch::join;
use super::results::{
    MAX_NAME_CHARS, MAX_RESULT_BYTES, Output, json_len, out, page_of_window, page_window, short,
    summary_classes,
};
use crate::batch_cache::MAX_BATCH_BYTES;
use crate::errors::{Error, invalid_arguments};
use crate::paths::{BatchPath, MAX_BATCH, too_many_inputs};
use crate::server::Inner;
use crate::types::*;

/// Largest image accepted by `predict` (as for `POST /api/predict`).
const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
/// Largest calibrated thresholds JSON file accepted by `predict` / `predict_batch`.
const MAX_THRESHOLDS_BYTES: u64 = 4 * 1024 * 1024;
/// At most this many `predict_batch` failures are listed inline.
const MAX_FAILURES: usize = 50;

impl Inner {
    pub(super) async fn list_models(&self, a: PageArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        let all = self.blocking(|s| s.list_models_with_arch()).await?;
        let items: Vec<ModelSummary> = all
            .into_iter()
            .map(|(m, arch_name): (Model, String)| ModelSummary {
                id: m.id,
                name: short(&m.name, MAX_NAME_CHARS),
                arch: m.arch,
                arch_name,
                imgsz: m.imgsz,
                num_classes: m.classes.len(),
                classes: summary_classes(&m.classes),
                metrics: m.metrics,
                head: m.head,
                trained_with: m.trained_with,
                precision: m.precision,
                fused: m.fused,
                source: m.source,
                about: m.about,
            })
            .collect();
        let page = page_of_window(items, offset, limit);
        let summary = format!("{} of {} models", page.items.len(), page.total);
        out(&page, summary)
    }

    pub(super) async fn val(&self, a: ValArgs, progress: &mut Progress) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        if let Some(s) = a.imgsz
            && (!s.is_multiple_of(32) || !(32..=4096).contains(&s))
        {
            return Err(invalid_arguments(format!("imgsz must be a multiple of 32 in 32..=4096, got {s}")));
        }
        let diag_enabled = a.diagnostics || a.worst.is_some();
        if !diag_enabled && (a.diag_conf.is_some() || a.diag_tp_iou.is_some() || a.diag_bg_iou.is_some()) {
            return Err(invalid_arguments(
                "diag_conf, diag_tp_iou and diag_bg_iou require `diagnostics: true` or `worst`",
            ));
        }
        let diag = if diag_enabled {
            let cfg = ValDiagConfig {
                worst: a.worst.unwrap_or(DEFAULT_VAL_DIAG_WORST),
                conf: a.diag_conf.unwrap_or(DEFAULT_VAL_DIAG_CONF),
                tp_iou: a.diag_tp_iou.unwrap_or(DEFAULT_VAL_DIAG_TP_IOU),
                bg_iou: a.diag_bg_iou.unwrap_or(DEFAULT_VAL_DIAG_BG_IOU),
            };
            check_val_diag_config(&cfg).map_err(|e| {
                invalid_arguments(e.body.message)
                    .hint(format!("worst must be in 1..={MAX_VAL_DIAG_WORST}, thresholds in 0..=1 with diag_bg_iou <= diag_tp_iou"))
            })?;
            Some(cfg)
        } else {
            None
        };
        self.ensure_dataset(&a.dataset_id).await?;
        progress.report(0.0, Some(1.0), false).await;
        let req = ValRequest {
            model_id: a.model_id,
            dataset_id: a.dataset_id,
            split: a.split,
            imgsz: a.imgsz,
            conf: a.conf,
            iou: a.iou,
        };
        let out_val = self.blocking(move |s| s.val_extended(req, diag)).await?;
        let r = out_val.report;
        let diagnostics = out_val.diagnostics;
        let thresholds = out_val.thresholds;
        progress.report(1.0, Some(1.0), true).await;
        // The dataset's name when the server reports it (older servers send only the id).
        let dataset = match &r.dataset_name {
            Some(name) => format!("{name} ({})", r.dataset_id),
            None => r.dataset_id.clone(),
        };
        let at = r.val.best_f1_conf.map_or(String::new(), |c| format!(" at conf {c:.3}"));
        let mut summary = format!(
            "{} on {dataset}/{} ({} images): mAP50 {:.3}, mAP50-95 {:.3}, P {:.3}, R {:.3}{at}",
            r.model_id,
            r.split.as_str(),
            r.images,
            r.val.map50,
            r.val.map50_95,
            r.val.precision,
            r.val.recall
        );
        let mut result = ValToolResult {
            report: r,
            diagnostics: diagnostics.map(val_diagnostics_out),
            thresholds: thresholds.map(threshold_calibration_out),
        };
        let thresholds_omitted = fit_val_result(&mut result);
        if thresholds_omitted {
            summary.push_str(
                "; calibrated thresholds omitted to stay under 32 KB (use `folo val --save-thresholds` to export full per-class thresholds)",
            );
        } else if let Some(t) = &result.thresholds
            && !t.per_class.is_empty()
        {
            summary.push_str(&format!(
                "; macro-F1 {:.3} ({} calibrated classes)",
                t.macro_f1,
                t.per_class.len()
            ));
        }
        if let Some(d) = &result.diagnostics {
            summary.push_str(&format!(
                "; diag: {}/{} failing (TP {}, FP {}, FN {})",
                d.failing_images, d.images_diagnosed, d.summary.tp, d.summary.fp, d.summary.fn_count
            ));
        }
        out(&result, summary)
    }

    async fn resolve_predict_thresholds(
        &self,
        explicit_conf: Option<f32>,
        thresholds_path: Option<String>,
        per_class_conf: Option<BTreeMap<u32, f32>>,
    ) -> Result<(f32, Option<BTreeMap<u32, f32>>, bool), Error> {
        if explicit_conf.is_some() && thresholds_path.is_some() {
            return Err(invalid_arguments("give either conf or thresholds_path, not both"));
        }
        let default_conf = explicit_conf.unwrap_or_else(folo_api::default_conf);
        match (thresholds_path, per_class_conf) {
            (Some(_), Some(_)) => {
                Err(invalid_arguments("give either thresholds_path or per_class_conf, not both"))
            }
            (Some(p), None) => {
                if p.trim().is_empty() {
                    return Err(invalid_arguments("thresholds_path must not be empty"));
                }
                let paths = self.paths.clone();
                let cal = tokio::task::spawn_blocking(move || match paths.resolve_for_batch(&p)? {
                    BatchPath::Ready(resolved) => read_thresholds(&resolved),
                    BatchPath::Missing(err) => Err(err),
                })
                .await
                .map_err(join)??;
                if cal.is_empty() {
                    Ok((default_conf, None, false))
                } else {
                    Ok((cal.global.best_conf, Some(cal.per_class_map()), true))
                }
            }
            (None, Some(map)) => {
                check_per_class_conf(&map, None).map_err(|e| invalid_arguments(e.body.message))?;
                if map.is_empty() {
                    Ok((default_conf, None, false))
                } else {
                    Ok((default_conf, Some(map), true))
                }
            }
            (None, None) => Ok((default_conf, None, false)),
        }
    }

    pub(super) async fn predict(&self, a: PredictArgs) -> Result<Output, Error> {
        if a.model_id.trim().is_empty() {
            return Err(invalid_arguments("model_id must not be empty"));
        }
        if let Some(conf) = a.conf {
            check_unit("conf", conf)?;
        }
        check_unit("iou", a.iou)?;
        if !(1..=300).contains(&a.max_det) {
            return Err(invalid_arguments(format!("max_det must be in 1..=300, got {}", a.max_det)));
        }
        let options = build_predict_options(
            a.tile,
            a.tile_overlap,
            a.tile_full_image,
            a.tta,
            a.tta_scales,
            a.tta_hflip,
            a.max_det as usize,
        )?;
        let (eff_conf, per_class_conf, conf_inclusive) =
            self.resolve_predict_thresholds(a.conf, a.thresholds_path, a.per_class_conf).await?;
        // The file an `image_path` names, for a decode error (the server only sees the uploaded bytes).
        let mut file = None;
        let source = match (a.image_path, a.image_base64, a.dataset_id, a.image_id) {
            (Some(p), None, None, None) => {
                file = Some(folo_api::json::path(&p));
                let paths = self.paths.clone();
                let read = move || read_image(&paths.resolve(&p)?);
                let bytes = tokio::task::spawn_blocking(read).await.map_err(join)??;
                if bytes.is_empty() {
                    return Err(invalid_arguments("image must not be empty"));
                }
                PredictSource::Upload(bytes)
            }
            (None, Some(b64), None, None) => {
                let decode = move || decode_image_base64(&b64, MAX_IMAGE_BYTES);
                let bytes = tokio::task::spawn_blocking(decode).await.map_err(join)??;
                if bytes.is_empty() {
                    return Err(invalid_arguments("image must not be empty"));
                }
                PredictSource::Upload(bytes)
            }
            (None, None, Some(dataset_id), Some(image_id)) => {
                if dataset_id.trim().is_empty() || image_id.trim().is_empty() {
                    return Err(invalid_arguments("dataset_id and image_id must not be empty"));
                }
                self.ensure_dataset(&dataset_id).await?;
                PredictSource::Dataset { dataset_id, image_id }
            }
            _ => {
                return Err(invalid_arguments(
                    "give exactly one image source: image_path, image_base64, or dataset_id + image_id",
                ));
            }
        };
        let input = PredictInput {
            model_id: a.model_id,
            source,
            conf: eff_conf,
            iou: a.iou,
            max_det: a.max_det,
            options,
            per_class_conf,
            conf_inclusive,
        };
        let mut r = self.blocking(move |s| s.predict(input)).await.map_err(|mut e| {
            if let Some(f) = &file
                && e.body.code == "image_decode_failed"
            {
                e.body.message = format!("{f}: {}", e.body.message);
            }
            e
        })?;
        r.boxes.truncate(a.max_det as usize);
        let mut dropped = 0;
        while json_len(&r) > MAX_RESULT_BYTES && r.boxes.pop().is_some() {
            dropped += 1;
        }
        let mut summary = format!(
            "{} boxes on a {}x{} image in {:.1} ms ({})",
            r.boxes.len(),
            r.image.width,
            r.image.height,
            r.timing.total_ms,
            r.device
        );
        if dropped > 0 {
            summary.push_str(&format!("; {dropped} lowest-scoring boxes left out to stay under 32 KB"));
        }
        out(&r, summary)
    }

    pub(super) async fn predict_batch(
        &self,
        a: PredictBatchArgs,
        progress: &mut Progress,
    ) -> Result<Output, Error> {
        if let Some(conf) = a.conf {
            check_unit("conf", conf)?;
        }
        check_unit("iou", a.iou)?;
        let options = build_predict_options(
            a.tile,
            a.tile_overlap,
            a.tile_full_image,
            a.tta,
            a.tta_scales,
            a.tta_hflip,
            folo_api::default_max_det() as usize,
        )?;
        let (eff_conf, per_class_conf, conf_inclusive) =
            self.resolve_predict_thresholds(a.conf, a.thresholds_path, a.per_class_conf).await?;
        // Validate input source shape and bounds before hitting the store or proxy server.
        let batch_input = match (a.paths, a.glob) {
            (Some(paths), None) => {
                if paths.len() > MAX_BATCH {
                    return Err(too_many_inputs(format!("{} paths given, at most {MAX_BATCH}", paths.len())));
                }
                if paths.is_empty() {
                    return Err(invalid_arguments("paths must not be empty"));
                }
                Ok(paths)
            }
            (None, Some(g)) => Err(g),
            _ => return Err(invalid_arguments("give exactly one of paths or glob")),
        };
        let model_id = a.model_id.clone();
        let model = self.blocking(move |s| s.get_model(&model_id)).await?;
        if let Some(map) = &per_class_conf {
            check_per_class_conf(map, Some(model.classes.len()))?;
        }
        check_predict_workload(&options, model.imgsz, None).map_err(|e| {
            let mut err = invalid_arguments(e.body.message);
            if let Some(h) = e.body.hint {
                err = err.hint(h);
            }
            err
        })?;
        let paths: Vec<(String, BatchPath)> = match batch_input {
            Ok(paths) => {
                let policy = self.paths.clone();
                let resolve = move || {
                    paths
                        .into_iter()
                        .map(|p| policy.resolve_for_batch(&p).map(|r| (folo_api::json::path(&p), r)))
                        .collect::<Result<Vec<_>, _>>()
                };
                tokio::task::spawn_blocking(resolve).await.map_err(join)??
            }
            Err(g) => {
                let paths = self.paths.clone();
                let found =
                    tokio::task::spawn_blocking(move || paths.expand_glob(&g)).await.map_err(join)??;
                if found.is_empty() {
                    return Err(invalid_arguments("the glob matches no files"));
                }
                found.into_iter().map(|p| (folo_api::json::path(&p), BatchPath::Ready(p))).collect()
            }
        };
        let started = Instant::now();
        let total = paths.len() as f64;
        progress.report(0.0, Some(total), false).await;
        let mut images = Vec::with_capacity(paths.len());
        let mut failed = Vec::new();
        let mut per_class: BTreeMap<String, u64> = BTreeMap::new();
        let (mut boxes_total, mut model_ms) = (0u64, 0f64);
        for (i, (shown, entry)) in paths.into_iter().enumerate() {
            let res = match entry {
                BatchPath::Ready(path) => {
                    let model_id = a.model_id.clone();
                    let (conf, iou) = (eff_conf, a.iou);
                    let options = options.clone();
                    let per_class_conf = per_class_conf.clone();
                    self.blocking(move |s| {
                        let bytes = read_image(&path)?;
                        s.predict(PredictInput {
                            model_id,
                            source: PredictSource::Upload(bytes),
                            conf,
                            iou,
                            max_det: folo_api::default_max_det(),
                            options,
                            per_class_conf,
                            conf_inclusive,
                        })
                    })
                    .await
                }
                BatchPath::Missing(err) => Err(err),
            };
            match res {
                Ok(r) => {
                    boxes_total += r.boxes.len() as u64;
                    model_ms += r.timing.total_ms;
                    for b in &r.boxes {
                        *per_class.entry(b.name.clone()).or_default() += 1;
                    }
                    images.push(
                        json!({ "path": shown, "image": r.image, "boxes": r.boxes, "timing": r.timing }),
                    );
                }
                // `not_implemented` (older server without `predict_options` or `threshold_calibration`) applies to
                // the whole batch; per-image `invalid_config` (e.g. an oversized image exceeding the tile grid
                // budget) is recorded in `failed` so valid images before and after it are preserved.
                Err(e) if e.body.code == "not_implemented" => {
                    return Err(e);
                }
                Err(e) => {
                    images.push(json!({ "path": shown, "error": e.body }));
                    failed.push(BatchFailure { path: shown, code: e.body.code, message: e.body.message });
                }
            }
            progress.report((i + 1) as f64, Some(total), i + 1 == total as usize).await;
        }
        let processed = (images.len() - failed.len()) as u64;
        let model_id = model.id;
        let Some(batch_id) = self
            .batches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .try_insert_next(|id| json!({ "batch_id": id, "model_id": model_id, "images": images }))
        else {
            return Err(Error::new(
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                format!(
                    "predict_batch result exceeds the {} MiB prediction cache budget",
                    MAX_BATCH_BYTES / (1024 * 1024)
                ),
            )
            .hint("split the batch into fewer images or raise `conf` to return fewer boxes"));
        };
        let details_uri = format!("folo://predictions/{batch_id}");
        let failed_total = failed.len() as u64;
        failed.truncate(MAX_FAILURES);
        let mut result = PredictBatchResult {
            batch_id: batch_id.clone(),
            processed,
            failed,
            failed_total,
            boxes_total,
            per_class_counts: per_class,
            timing: BatchTiming {
                total_ms: (started.elapsed().as_secs_f64() * 1e5).round() / 100.0,
                mean_ms: if processed > 0 {
                    (model_ms / processed as f64 * 100.0).round() / 100.0
                } else {
                    0.0
                },
            },
            details_uri: details_uri.clone(),
        };
        while json_len(&result) > MAX_RESULT_BYTES && result.failed.pop().is_some() {}
        let summary = format!(
            "{processed} images, {boxes_total} boxes, {failed_total} failed; per-image boxes at {details_uri}"
        );
        let mut o = out(&result, summary)?;
        o.links.push(ContentBlock::resource_link(
            Resource::new(details_uri, format!("predictions {batch_id}"))
                .with_description(format!("per-image boxes of predict_batch with {}", model.name))
                .with_mime_type("application/json"),
        ));
        Ok(o)
    }

    pub(super) async fn export(&self, a: ExportArgs) -> Result<Output, Error> {
        if a.format != folo_api::ExportFormat::Onnx && (a.dynamic_batch.is_some() || a.dynamic_hw.is_some()) {
            return Err(invalid_arguments("dynamic_batch and dynamic_hw only apply to format `onnx`"));
        }
        if a.dynamic && (a.dynamic_batch == Some(false) || a.dynamic_hw == Some(false)) {
            return Err(invalid_arguments(
                "`dynamic: true` cannot be combined with `dynamic_batch: false` or `dynamic_hw: false`",
            ));
        }
        let axes = DynamicAxes {
            batch: a.dynamic || a.dynamic_batch.unwrap_or(false),
            hw: a.dynamic || a.dynamic_hw.unwrap_or(false),
        };
        let spec = ExportSpec {
            format: a.format,
            onnx: OnnxSwitches { decode: a.decode, dynamic: axes.batch && axes.hw, half: a.half },
        };
        check_export_axes(&spec, axes).map_err(|e| invalid_arguments(e.body.message))?;
        let r = self.blocking(move |s| s.export_model_with_axes(&a.model_id, spec, axes)).await?;
        out(&r, format!("wrote {} ({} bytes)", r.path, r.bytes))
    }
}

fn threshold_calibration_out(c: ThresholdCalibration) -> ThresholdCalibrationOut {
    ThresholdCalibrationOut {
        global: OperatingPointOut {
            best_conf: c.global.best_conf,
            best_f1: c.global.best_f1,
            precision: c.global.precision,
            recall: c.global.recall,
        },
        per_class: c
            .per_class
            .into_iter()
            .map(|p| ClassThresholdOut {
                class: p.class,
                best_conf: p.best_conf,
                best_f1: p.best_f1,
                precision: p.precision,
                recall: p.recall,
                truths: p.truths,
                preds: p.preds,
            })
            .collect(),
        macro_f1: c.macro_f1,
    }
}

/// Fits `result` within [`MAX_RESULT_BYTES`]: first tries trimming `diagnostics.worst_images` while keeping the
/// full `thresholds` calibration; if even without `worst_images` the payload still exceeds the budget, omits
/// `thresholds` whole (never returning a silently truncated partial calibration), restores `worst_images`, and
/// trims only as many `worst_images` as needed for the remaining budget. Returns `true` when `thresholds` was
/// omitted.
fn fit_val_result(result: &mut ValToolResult) -> bool {
    if json_len(result) <= MAX_RESULT_BYTES {
        return false;
    }
    let saved_worst = result.diagnostics.as_ref().map(|d| d.worst_images.clone());
    while json_len(result) > MAX_RESULT_BYTES {
        if result.diagnostics.as_mut().and_then(|d| d.worst_images.pop()).is_none() {
            break;
        }
    }
    if json_len(result) <= MAX_RESULT_BYTES {
        return false;
    }
    let omitted = result.thresholds.take().is_some();
    if omitted && let Some(worst) = saved_worst {
        if let Some(d) = result.diagnostics.as_mut() {
            d.worst_images = worst;
        }
        while json_len(result) > MAX_RESULT_BYTES {
            if result.diagnostics.as_mut().and_then(|d| d.worst_images.pop()).is_none() {
                break;
            }
        }
    }
    omitted
}

/// Reads and validates a `ThresholdCalibration` JSON file saved by `folo val --save-thresholds`.
fn read_thresholds(path: &Path) -> Result<ThresholdCalibration, Error> {
    let shown = folo_api::json::path(path);
    let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::new(http::StatusCode::NOT_FOUND, "path_not_found", format!("{shown} does not exist"))
        }
        _ => Error::internal(format!("reading {shown}: {e}")),
    })?;
    if !meta.is_file() {
        return Err(invalid_arguments(format!("{shown} is not a file")));
    }
    if meta.len() > MAX_THRESHOLDS_BYTES {
        return Err(invalid_arguments(format!(
            "{shown} is too large ({} bytes, max {MAX_THRESHOLDS_BYTES})",
            meta.len()
        )));
    }
    let bytes = std::fs::read(path).map_err(|e| Error::internal(format!("reading {shown}: {e}")))?;
    let cal: ThresholdCalibration = serde_json::from_slice(&bytes)
        .map_err(|e| invalid_arguments(format!("{shown} is not valid ThresholdCalibration JSON: {e}")))?;
    cal.validate().map_err(|e| invalid_arguments(format!("{shown}: {e}")))?;
    Ok(cal)
}

fn val_diagnostics_out(d: ValDiagnostics) -> ValDiagnosticsOut {
    ValDiagnosticsOut {
        conf: d.conf,
        tp_iou: d.tp_iou,
        bg_iou: d.bg_iou,
        images_diagnosed: d.images_diagnosed,
        failing_images: d.failing_images,
        summary: ValDiagnosticsSummaryOut {
            num_truths: d.summary.num_truths,
            num_preds: d.summary.num_preds,
            tp: d.summary.tp,
            fp: d.summary.fp,
            fn_count: d.summary.fn_count,
            localization_errors: d.summary.localization_errors,
            class_confusions: d.summary.class_confusions,
            duplicate_errors: d.summary.duplicate_errors,
            background_fp: d.summary.background_fp,
            mean_f1: d.summary.mean_f1,
            mean_tp_iou: d.summary.mean_tp_iou,
        },
        worst_images: d
            .worst_images
            .into_iter()
            .map(|w| ImageDiagnosticsOut {
                image_index: w.image_index,
                image_id: w.image_id,
                num_truths: w.num_truths,
                num_preds: w.num_preds,
                tp: w.tp,
                fp: w.fp,
                fn_count: w.fn_count,
                localization_errors: w.localization_errors,
                class_confusions: w.class_confusions,
                duplicate_errors: w.duplicate_errors,
                background_fp: w.background_fp,
                f1: w.f1,
                mean_tp_iou: w.mean_tp_iou,
            })
            .collect(),
    }
}

/// Builds and validates optional SAHI tiling or TTA settings upfront before touching the store or proxy.
fn build_predict_options(
    tile: Option<u32>,
    tile_overlap: Option<f32>,
    tile_full_image: Option<bool>,
    tta: bool,
    tta_scales: Option<Vec<f32>>,
    tta_hflip: Option<bool>,
    max_det: usize,
) -> Result<PredictOptions, Error> {
    if tile.is_none() && (tile_overlap.is_some() || tile_full_image.is_some()) {
        return Err(invalid_arguments("tile_overlap and tile_full_image require `tile` to be set"));
    }
    if !tta && (tta_scales.is_some() || tta_hflip.is_some()) {
        return Err(invalid_arguments("tta_scales and tta_hflip require `tta: true`"));
    }
    if tile.is_some() && tta {
        return Err(invalid_arguments("`tile` and `tta` cannot be combined in the same prediction call"));
    }
    let tile_cfg = if let Some(size) = tile {
        let overlap = tile_overlap.unwrap_or(0.2);
        let cfg = TileConfig {
            tile_width: size,
            tile_height: size,
            overlap_x: overlap,
            overlap_y: overlap,
            full_image: tile_full_image.unwrap_or(true),
            max_det,
            ..TileConfig::default()
        };
        cfg.validate().map_err(|e| invalid_arguments(e.to_string()))?;
        Some(cfg)
    } else {
        None
    };
    let tta_cfg = if tta {
        let scales = tta_scales.unwrap_or_else(|| DEFAULT_TTA_SCALES.to_vec());
        let cfg = TtaConfig { hflip: tta_hflip.unwrap_or(true), scales, ..TtaConfig::default() };
        cfg.validate().map_err(|e| invalid_arguments(e.to_string()))?;
        Some(cfg)
    } else {
        None
    };
    let opts = PredictOptions { tta: tta_cfg, tile: tile_cfg };
    check_predict_options(&opts).map_err(|e| invalid_arguments(e.body.message))?;
    Ok(opts)
}

/// Reads an image file for inference (non-empty, at most 64 MiB).
fn read_image(path: &Path) -> Result<Vec<u8>, Error> {
    let meta = std::fs::metadata(path).map_err(|_| {
        Error::new(
            http::StatusCode::NOT_FOUND,
            "path_not_found",
            format!("{} does not exist", folo_api::json::path(path)),
        )
    })?;
    if !meta.is_file() {
        return Err(invalid_arguments(format!("{} is not a file", folo_api::json::path(path))));
    }
    if meta.len() > MAX_IMAGE_BYTES {
        return Err(too_large(&folo_api::json::path(path)));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| Error::internal(format!("reading {}: {e}", folo_api::json::path(path))))?;
    if bytes.is_empty() {
        return Err(invalid_arguments("image must not be empty"));
    }
    Ok(bytes)
}

/// `payload_too_large` for an image over [`MAX_IMAGE_BYTES`], `what` being a file or the argument.
fn too_large(what: &str) -> Error {
    Error::new(
        http::StatusCode::PAYLOAD_TOO_LARGE,
        "payload_too_large",
        format!("{what} is larger than 64 MiB"),
    )
}

/// The bytes of an `image_base64`, refused as too large over `max` of them. A string longer than `max` bytes can
/// encode is refused without decoding it; its length cannot tell the last two bytes apart (padding), so the decoded
/// length is checked too.
pub(super) fn decode_image_base64(b64: &str, max: u64) -> Result<Vec<u8>, Error> {
    let b64 = b64.trim();
    if b64.len() as u64 > max.div_ceil(3) * 4 {
        return Err(too_large("image_base64"));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| invalid_arguments(format!("image_base64 is not valid base64: {e}")))?;
    if bytes.len() as u64 > max {
        return Err(too_large("image_base64"));
    }
    Ok(bytes)
}

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
#[cfg(test)]
#[path = "models_thresholds_tests.rs"]
mod thresholds_tests;
