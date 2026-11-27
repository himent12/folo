//! Mock validation (`POST /api/val`, the MCP `val` tool): deterministic metrics derived from the model's own
//! metrics, per class jittered by class name, so the same model on the same split always scores the same.

use axum::http::StatusCode;
use folo_api::{ClassMetrics, Split, ValMetrics, ValReport, ValRequest, ValTiming};

use super::runs::{arch_spec, r3, r4};
use super::{MockStore, quality_of};
use crate::error::{Error, Result};
use crate::store::{
    ClassThreshold, ImageDiagnostics, OperatingPoint, Store, ThresholdCalibration, ValDiagConfig,
    ValDiagnostics, ValDiagnosticsSummary, ValExtended, check_val_diag_config,
};
use crate::util::{fnv1a, noise};

/// See the module docs.
pub(super) fn val(store: &MockStore, req: ValRequest) -> Result<ValReport> {
    val_extended(store, req, None).map(|v| v.report)
}

/// [`val`] with deterministic synthetic per-image error diagnostics when `diag` is `Some`.
pub(super) fn val_with_diagnostics(
    store: &MockStore,
    req: ValRequest,
    diag: Option<ValDiagConfig>,
) -> Result<(ValReport, Option<ValDiagnostics>)> {
    val_extended(store, req, diag).map(|v| (v.report, v.diagnostics))
}

/// [`val`] returning report, optional diagnostics, and calibrated per-class F1 thresholds.
pub(super) fn val_extended(
    store: &MockStore,
    req: ValRequest,
    diag: Option<ValDiagConfig>,
) -> Result<ValExtended> {
    if let Some(d) = &diag {
        check_val_diag_config(d)?;
    }
    let bad = |m: String| Err(Error::new(StatusCode::BAD_REQUEST, "invalid_body", m));
    if !(0.0..=1.0).contains(&req.conf) || !(0.0..=1.0).contains(&req.iou) {
        return bad("conf and iou must be in 0..=1".into());
    }
    if let Some(s) = req.imgsz
        && (s % 32 != 0 || !(32..=4096).contains(&s))
    {
        return bad(format!("imgsz must be a multiple of 32 in 32..=4096, got {s}"));
    }
    let model = store.get_model(&req.model_id)?;
    let ds = store.dataset(&req.dataset_id)?;
    let meta = ds.meta();
    let images = match req.split {
        Split::Train => meta.splits.train,
        Split::Val => meta.splits.val,
        Split::Test => meta.splits.test,
    };
    if images == 0 {
        let msg = format!("dataset '{}' has no {} images", meta.id, req.split.as_str());
        return Err(Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", msg));
    }
    let imgsz = req.imgsz.unwrap_or(model.imgsz);
    let base = model.metrics;
    let q = quality_of(&model);
    // Training images score a little higher, a size far from the training size a little lower, and a high
    // confidence threshold cuts recall.
    let split_gain = if req.split == Split::Train { 1.06 } else { 1.0 };
    let size_gain = 1.0 - 0.15 * ((imgsz as f32 / model.imgsz as f32).ln().abs()).min(1.0);
    let conf_gain = 1.0 - 0.4 * req.conf.clamp(0.0, 1.0);
    let gain = split_gain * size_gain;
    let seed = fnv1a(model.id.as_bytes()) ^ fnv1a(meta.id.as_bytes());
    let per_class: Vec<ClassMetrics> = meta
        .classes
        .iter()
        .enumerate()
        .filter(|(_, name)| model.classes.contains(name))
        .map(|(c, name)| {
            let j = 1.0 + 0.08 * noise(&[seed, fnv1a(name.as_bytes())]);
            let map50 = (q * gain * j).clamp(0.0, 0.995);
            ClassMetrics {
                class: c as u32,
                map50: r4(map50),
                map50_95: r4(map50 * base.map_or(0.75, |m| m.map50_95 / m.map50.max(1e-3))),
                precision: r4((base.map_or(q, |m| m.precision) * gain * j).clamp(0.0, 1.0)),
                recall: r4((base.map_or(q, |m| m.recall) * gain * j * conf_gain).clamp(0.0, 1.0)),
            }
        })
        .collect();
    let mean = |f: fn(&ClassMetrics) -> f32| {
        if per_class.is_empty() {
            0.0
        } else {
            r4(per_class.iter().map(f).sum::<f32>() / per_class.len() as f32)
        }
    };
    let val = ValMetrics {
        map50: mean(|c| c.map50),
        map50_95: mean(|c| c.map50_95),
        precision: mean(|c| c.precision),
        recall: mean(|c| c.recall),
        // Detections below the NMS `conf` never count, so the operating point is at least that.
        best_f1_conf: (!per_class.is_empty())
            .then(|| r3(base.and_then(|m| m.best_f1_conf).unwrap_or(0.3).max(req.conf).min(1.0))),
        per_class,
    };
    let (.., inf_ms) = arch_spec(model.arch);
    let per_image_ms = ((inf_ms * (imgsz as f64 / 640.0).powi(2) + 1.1) * 100.0).round() / 100.0;
    let total_s = ((per_image_ms * images as f64 / 1000.0) * 100.0).round() / 100.0;
    let report = ValReport {
        model_id: model.id,
        dataset_id: meta.id.clone(),
        dataset_name: Some(meta.name.clone()),
        split: req.split,
        imgsz,
        images,
        val,
        timing: ValTiming { total_s, per_image_ms },
    };
    let diagnostics = diag.map(|cfg| mock_diagnostics(&report, seed, cfg));
    let thresholds = Some(mock_thresholds(&report, seed, req.conf));
    Ok(ValExtended { report, diagnostics, thresholds })
}

fn mock_thresholds(report: &ValReport, seed: u64, min_conf: f32) -> ThresholdCalibration {
    let f1_of = |p: f32, r: f32| if p + r > 0.0 { r4(2.0 * p * r / (p + r)) } else { 0.0 };
    let floor = min_conf.clamp(0.0, 1.0);
    let global_conf = report.val.best_f1_conf.unwrap_or(0.0).clamp(floor, 1.0);
    let per_class: Vec<ClassThreshold> = report
        .val
        .per_class
        .iter()
        .map(|c| {
            let jitter = 0.08 * noise(&[seed, u64::from(c.class) ^ 0x9e37]);
            let best_conf = r3((global_conf + jitter).clamp(floor, 1.0));
            let precision = c.precision.clamp(0.0, 1.0);
            let recall = c.recall.clamp(0.0, 1.0);
            let truths = (report.images.max(1) * 2) / report.val.per_class.len().max(1) as u64;
            let truths = truths.max(1);
            let preds = ((truths as f32) * (recall / precision.max(0.05))).round().max(0.0) as u64;
            ClassThreshold {
                class: c.class,
                best_conf,
                best_f1: f1_of(precision, recall),
                precision,
                recall,
                truths,
                preds,
            }
        })
        .collect();
    let macro_f1 = if per_class.is_empty() {
        0.0
    } else {
        r4(per_class.iter().map(|c| c.best_f1).sum::<f32>() / per_class.len() as f32)
    };
    let global = if per_class.is_empty() {
        OperatingPoint::default()
    } else {
        let p = report.val.precision.clamp(0.0, 1.0);
        let r = report.val.recall.clamp(0.0, 1.0);
        OperatingPoint { best_conf: global_conf, best_f1: f1_of(p, r), precision: p, recall: r }
    };
    ThresholdCalibration { global, per_class, macro_f1 }
}

fn mock_diagnostics(report: &ValReport, seed: u64, cfg: ValDiagConfig) -> ValDiagnostics {
    let n = report.images.min(32) as usize;
    let split = report.split.as_str();
    let mut diags = Vec::with_capacity(n);
    let (mut num_truths, mut num_preds, mut tp_all, mut fp_all, mut fn_all) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut loc_all, mut cls_all, mut dup_all, mut bg_all, mut failing) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut f1_sum, mut iou_weighted) = (0.0f64, 0.0f64);
    for i in 0..n {
        let h = fnv1a(format!("{seed}:{split}:{i}").as_bytes());
        let tp = ((h & 3) + 1) as u32;
        let fn_count = ((h >> 2) % 3) as u32;
        let loc = ((h >> 5) % 2) as u32;
        let cls = ((h >> 7) % 2) as u32;
        let dup = ((h >> 9) % 2) as u32;
        let bg = ((h >> 11) % 2) as u32;
        let fp = loc + cls + dup + bg;
        let truths = tp + fn_count;
        let preds = tp + fp;
        let f1 = if 2 * tp + fp + fn_count == 0 {
            1.0
        } else {
            (2.0 * tp as f32) / (2.0 * tp as f32 + fp as f32 + fn_count as f32)
        };
        let mean_tp_iou = if tp > 0 {
            (cfg.tp_iou + (1.0 - cfg.tp_iou) * (0.4 + 0.4 * ((h >> 13) % 100) as f32 / 100.0)).min(0.99)
        } else {
            0.0
        };
        let d = ImageDiagnostics {
            image_index: i as u64,
            image_id: Some(format!("images/{split}/{split}_{i:04}.jpg")),
            num_truths: truths,
            num_preds: preds,
            tp,
            fp,
            fn_count,
            localization_errors: loc,
            class_confusions: cls,
            duplicate_errors: dup,
            background_fp: bg,
            f1,
            mean_tp_iou,
        };
        num_truths += u64::from(d.num_truths);
        num_preds += u64::from(d.num_preds);
        tp_all += u64::from(d.tp);
        fp_all += u64::from(d.fp);
        fn_all += u64::from(d.fn_count);
        loc_all += u64::from(d.localization_errors);
        cls_all += u64::from(d.class_confusions);
        dup_all += u64::from(d.duplicate_errors);
        bg_all += u64::from(d.background_fp);
        if d.total_errors() > 0 {
            failing += 1;
        }
        f1_sum += f64::from(d.f1);
        iou_weighted += f64::from(d.mean_tp_iou) * f64::from(d.tp);
        diags.push(d);
    }
    let mean_f1 = if diags.is_empty() { 1.0 } else { (f1_sum / diags.len() as f64) as f32 };
    let mean_tp_iou = if tp_all > 0 { (iou_weighted / tp_all as f64) as f32 } else { 0.0 };
    ValDiagnostics {
        conf: cfg.conf,
        tp_iou: cfg.tp_iou,
        bg_iou: cfg.bg_iou,
        images_diagnosed: diags.len() as u64,
        failing_images: failing,
        summary: ValDiagnosticsSummary {
            num_truths,
            num_preds,
            tp: tp_all,
            fp: fp_all,
            fn_count: fn_all,
            localization_errors: loc_all,
            class_confusions: cls_all,
            duplicate_errors: dup_all,
            background_fp: bg_all,
            mean_f1,
            mean_tp_iou,
        },
        worst_images: folo_vision::eval::worst_images(&diags, cfg.worst).into_iter().cloned().collect(),
    }
}
