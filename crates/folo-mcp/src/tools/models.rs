//! The tools that use a model: list models, validate, predict (one image or a batch) and export.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine as _;
use folo_api::{ExportSpec, Model, OnnxSwitches, ValRequest};
use folo_server::store::{PredictInput, PredictSource};
use rmcp::model::{ContentBlock, Resource};
use serde_json::json;

use super::Progress;
use super::arguments::check_unit;
use super::dispatch::join;
use super::results::{
    MAX_NAME_CHARS, MAX_RESULT_BYTES, Output, json_len, out, page_of, short, summary_classes,
};
use crate::errors::{Error, invalid_arguments};
use crate::paths::{MAX_BATCH, too_many_inputs};
use crate::server::Inner;
use crate::types::*;

/// Largest image accepted by `predict` (as for `POST /api/predict`).
const MAX_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
/// At most this many `predict_batch` failures are listed inline.
const MAX_FAILURES: usize = 50;

impl Inner {
    pub(super) async fn list_models(&self, a: PageArgs) -> Result<Output, Error> {
        let all = self.blocking(|s| s.list_models()).await?;
        let items: Vec<ModelSummary> = all
            .into_iter()
            .map(|m: Model| ModelSummary {
                id: m.id,
                name: short(&m.name, MAX_NAME_CHARS),
                arch: m.arch,
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
        let page = page_of(items, a.cursor.as_deref(), a.limit)?;
        let summary = format!("{} of {} models", page.items.len(), page.total);
        out(&page, summary)
    }

    pub(super) async fn val(&self, a: ValArgs, progress: &mut Progress) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
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
        let r = self.blocking(move |s| s.val(req)).await?;
        progress.report(1.0, Some(1.0), true).await;
        // The dataset's name when the server reports it (older servers send only the id).
        let dataset = match &r.dataset_name {
            Some(name) => format!("{name} ({})", r.dataset_id),
            None => r.dataset_id.clone(),
        };
        let at = r.val.best_f1_conf.map_or(String::new(), |c| format!(" at conf {c:.3}"));
        let summary = format!(
            "{} on {dataset}/{} ({} images): mAP50 {:.3}, mAP50-95 {:.3}, P {:.3}, R {:.3}{at}",
            r.model_id,
            r.split.as_str(),
            r.images,
            r.val.map50,
            r.val.map50_95,
            r.val.precision,
            r.val.recall
        );
        out(&r, summary)
    }

    pub(super) async fn predict(&self, a: PredictArgs) -> Result<Output, Error> {
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        if !(1..=300).contains(&a.max_det) {
            return Err(invalid_arguments(format!("max_det must be in 1..=300, got {}", a.max_det)));
        }
        // The file an `image_path` names, for a decode error (the server only sees the uploaded bytes).
        let mut file = None;
        let source = match (a.image_path, a.image_base64, a.dataset_id, a.image_id) {
            (Some(p), None, None, None) => {
                file = Some(folo_api::json::path(&p));
                let paths = self.paths.clone();
                let read = move || read_image(&paths.resolve(&p)?);
                PredictSource::Upload(tokio::task::spawn_blocking(read).await.map_err(join)??)
            }
            (None, Some(b64), None, None) => {
                check_base64_len(b64.len())?;
                let decode = move || {
                    base64::engine::general_purpose::STANDARD
                        .decode(b64.trim())
                        .map_err(|e| invalid_arguments(format!("image_base64 is not valid base64: {e}")))
                };
                PredictSource::Upload(tokio::task::spawn_blocking(decode).await.map_err(join)??)
            }
            (None, None, Some(dataset_id), Some(image_id)) => {
                self.ensure_dataset(&dataset_id).await?;
                PredictSource::Dataset { dataset_id, image_id }
            }
            _ => {
                return Err(invalid_arguments(
                    "give exactly one image source: image_path, image_base64, or dataset_id + image_id",
                ));
            }
        };
        let input =
            PredictInput { model_id: a.model_id, source, conf: a.conf, iou: a.iou, max_det: a.max_det };
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
        check_unit("conf", a.conf)?;
        check_unit("iou", a.iou)?;
        let model_id = a.model_id.clone();
        let model = self.blocking(move |s| s.get_model(&model_id)).await?;
        let paths: Vec<(String, std::path::PathBuf)> = match (a.paths, a.glob) {
            (Some(paths), None) => {
                if paths.len() > MAX_BATCH {
                    return Err(too_many_inputs(format!("{} paths given, at most {MAX_BATCH}", paths.len())));
                }
                if paths.is_empty() {
                    return Err(invalid_arguments("paths must not be empty"));
                }
                let policy = self.paths.clone();
                let resolve = move || {
                    paths
                        .into_iter()
                        .map(|p| policy.resolve(&p).map(|r| (p, r)))
                        .collect::<Result<Vec<_>, _>>()
                };
                tokio::task::spawn_blocking(resolve).await.map_err(join)??
            }
            (None, Some(g)) => {
                let paths = self.paths.clone();
                let found =
                    tokio::task::spawn_blocking(move || paths.expand_glob(&g)).await.map_err(join)??;
                if found.is_empty() {
                    return Err(invalid_arguments("the glob matches no files"));
                }
                found.into_iter().map(|p| (p.to_string_lossy().into_owned(), p)).collect()
            }
            _ => return Err(invalid_arguments("give exactly one of paths or glob")),
        };
        let started = Instant::now();
        let total = paths.len() as f64;
        progress.report(0.0, Some(total), false).await;
        let mut images = Vec::with_capacity(paths.len());
        let mut failed = Vec::new();
        let mut per_class: BTreeMap<String, u64> = BTreeMap::new();
        let (mut boxes_total, mut model_ms) = (0u64, 0f64);
        for (i, (shown, path)) in paths.into_iter().enumerate() {
            let model_id = a.model_id.clone();
            let (conf, iou) = (a.conf, a.iou);
            let res = self
                .blocking(move |s| {
                    let bytes = read_image(&path)?;
                    s.predict(PredictInput {
                        model_id,
                        source: PredictSource::Upload(bytes),
                        conf,
                        iou,
                        max_det: folo_api::default_max_det(),
                    })
                })
                .await;
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
                Err(e) if e.body.code == "not_implemented" => return Err(e),
                Err(e) => {
                    images.push(json!({ "path": shown, "error": e.body }));
                    failed.push(BatchFailure { path: shown, code: e.body.code, message: e.body.message });
                }
            }
            progress.report((i + 1) as f64, Some(total), i + 1 == total as usize).await;
        }
        let processed = (images.len() - failed.len()) as u64;
        let batch_id = self.next_batch_id();
        let details_uri = format!("folo://predictions/{batch_id}");
        let details = json!({ "batch_id": batch_id, "model_id": model.id, "images": images });
        self.batches.lock().unwrap().insert(batch_id.clone(), Arc::new(details));
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
        let spec = ExportSpec {
            format: a.format,
            onnx: OnnxSwitches { decode: a.decode, dynamic: a.dynamic, half: a.half },
        };
        let r = self.blocking(move |s| s.export_model(&a.model_id, spec)).await?;
        out(&r, format!("wrote {} ({} bytes)", r.path, r.bytes))
    }
}

/// Reads an image file for inference (at most 64 MiB).
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
    std::fs::read(path).map_err(|e| Error::internal(format!("reading {}: {e}", folo_api::json::path(path))))
}

/// `payload_too_large` for an image over [`MAX_IMAGE_BYTES`], `what` being a file or the argument.
fn too_large(what: &str) -> Error {
    Error::new(
        http::StatusCode::PAYLOAD_TOO_LARGE,
        "payload_too_large",
        format!("{what} is larger than 64 MiB"),
    )
}

/// Refuses an `image_base64` of `len` characters whose image would be over [`MAX_IMAGE_BYTES`], before decoding it.
pub(super) fn check_base64_len(len: usize) -> Result<(), Error> {
    if len as u64 > MAX_IMAGE_BYTES * 4 / 3 + 4 {
        return Err(too_large("image_base64"));
    }
    Ok(())
}
