//! Mock validation (`POST /api/val`, the MCP `val` tool): deterministic metrics derived from the model's own
//! metrics, per class jittered by class name, so the same model on the same split always scores the same.

use axum::http::StatusCode;
use folo_api::{ClassMetrics, Split, ValMetrics, ValReport, ValRequest, ValTiming};

use super::runs::{arch_spec, r4};
use super::{MockStore, quality_of};
use crate::error::{Error, Result};
use crate::store::Store;
use crate::util::{fnv1a, noise};

/// See the module docs.
pub(super) fn val(store: &MockStore, req: ValRequest) -> Result<ValReport> {
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
        return bad(format!("dataset '{}' has no {} images", meta.id, req.split.as_str()));
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
        per_class,
    };
    let (.., inf_ms) = arch_spec(model.arch);
    let per_image_ms = ((inf_ms * (imgsz as f64 / 640.0).powi(2) + 1.1) * 100.0).round() / 100.0;
    let total_s = ((per_image_ms * images as f64 / 1000.0) * 100.0).round() / 100.0;
    Ok(ValReport {
        model_id: model.id,
        dataset_id: meta.id.clone(),
        split: req.split,
        imgsz,
        images,
        val,
        timing: ValTiming { total_s, per_image_ms },
    })
}
