//! The tools that use a model: list models, validate, predict (one image or a batch) and export.

use std::collections::BTreeMap;
use std::path::Path;
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
/// At most this many `predict_batch` failures are listed inline.
const MAX_FAILURES: usize = 50;

impl Inner {
    pub(super) async fn list_models(&self, a: PageArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
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
        if a.model_id.trim().is_empty() {
            return Err(invalid_arguments("model_id must not be empty"));
        }
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
                    let (conf, iou) = (a.conf, a.iou);
                    self.blocking(move |s| {
                        let bytes = read_image(&path)?;
                        s.predict(PredictInput {
                            model_id,
                            source: PredictSource::Upload(bytes),
                            conf,
                            iou,
                            max_det: folo_api::default_max_det(),
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
                Err(e) if e.body.code == "not_implemented" => return Err(e),
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
        let spec = ExportSpec {
            format: a.format,
            onnx: OnnxSwitches { decode: a.decode, dynamic: a.dynamic, half: a.half },
        };
        let r = self.blocking(move |s| s.export_model(&a.model_id, spec)).await?;
        out(&r, format!("wrote {} ({} bytes)", r.path, r.bytes))
    }
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
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::Options;
    use crate::proxy::HttpStore;
    use crate::server::FoloMcp;
    use folo_server::Store;

    /// `list_models`, `list_runs`, `list_label_jobs`, `list_benchmarks`, `val`, and `predict_batch` validate
    /// paging and argument ranges before hitting the store or proxy server, keeping embedded and `--server`
    /// error codes identical even when the server is unreachable.
    #[tokio::test]
    async fn tools_validate_paging_and_args_before_proxy_access() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_bg = Arc::clone(&hits);
        let accept = tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                hits_bg.fetch_add(1, Ordering::SeqCst);
                drop(stream);
            }
        });
        let http = Arc::new(HttpStore::new(&url, None));
        let home = folo_test_support::TestDir::new("mcp-upfront-validation");
        let opts = Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        };
        let server = FoloMcp::new(http.clone(), Some(http), opts);

        let bad_page = || PageArgs { limit: Some(0), cursor: None };
        let bad_cursor = || PageArgs { limit: Some(10), cursor: Some("bogus".into()) };
        assert_eq!(server.inner.list_models(bad_page()).await.err().unwrap().body.code, "invalid_arguments");
        assert_eq!(
            server.inner.list_models(bad_cursor()).await.err().unwrap().body.code,
            "invalid_arguments"
        );
        assert_eq!(
            server
                .inner
                .list_runs(ListRunsArgs { status: None, limit: Some(0), cursor: None })
                .await
                .err()
                .unwrap()
                .body
                .code,
            "invalid_arguments"
        );
        assert_eq!(
            server.inner.list_label_jobs(bad_page()).await.err().unwrap().body.code,
            "invalid_arguments"
        );
        assert_eq!(
            server.inner.list_benchmarks(bad_page()).await.err().unwrap().body.code,
            "invalid_arguments"
        );

        // `val` rejects bad `imgsz` before touching the proxy.
        let val_err = server
            .inner
            .val(
                ValArgs {
                    model_id: "m_1".into(),
                    dataset_id: "ds_1".into(),
                    split: folo_api::Split::Val,
                    imgsz: Some(33),
                    conf: 0.001,
                    iou: 0.7,
                },
                &mut Progress::none(),
            )
            .await
            .err()
            .unwrap();
        assert_eq!(val_err.body.code, "invalid_arguments");

        // `predict_batch` rejects empty paths, both paths+glob, or neither before looking up `model_id`.
        for (paths, glob) in
            [(Some(vec![]), None), (None, None), (Some(vec!["a.jpg".into()]), Some("*.jpg".into()))]
        {
            let err = server
                .inner
                .predict_batch(
                    PredictBatchArgs { model_id: "m_missing".into(), paths, glob, conf: 0.25, iou: 0.7 },
                    &mut Progress::none(),
                )
                .await
                .err()
                .unwrap();
            assert_eq!(err.body.code, "invalid_arguments");
        }

        // `predict` rejects blank `model_id`, empty `image_base64`, and blank `dataset_id`/`image_id`
        // identically on both embedded and proxied backends before reading files or hitting the store.
        let embedded = FoloMcp::new(
            Arc::new(folo_server::MockStore::new()),
            None,
            Options {
                data_dir: home.to_path_buf(),
                read_only: false,
                confine: false,
                device: "cpu".into(),
                published_weights: None,
            },
        );
        for backend in [&server, &embedded] {
            for args in [
                PredictArgs {
                    model_id: "   ".into(),
                    image_path: None,
                    image_base64: Some("AAAA".into()),
                    dataset_id: None,
                    image_id: None,
                    conf: 0.25,
                    iou: 0.7,
                    max_det: 10,
                },
                PredictArgs {
                    model_id: "m_1".into(),
                    image_path: None,
                    image_base64: Some("".into()),
                    dataset_id: None,
                    image_id: None,
                    conf: 0.25,
                    iou: 0.7,
                    max_det: 10,
                },
                PredictArgs {
                    model_id: "m_1".into(),
                    image_path: None,
                    image_base64: None,
                    dataset_id: Some("  ".into()),
                    image_id: Some("img_1".into()),
                    conf: 0.25,
                    iou: 0.7,
                    max_det: 10,
                },
            ] {
                let err = backend.inner.predict(args).await.err().unwrap();
                assert_eq!(err.body.code, "invalid_arguments");
            }
        }

        assert_eq!(hits.load(Ordering::SeqCst), 0, "invalid arguments must never hit the proxy server");
        accept.abort();
        let _ = accept.await;

        // `predict_batch` records the same `invalid_arguments` failure code for a zero-byte file across both
        // embedded and proxied backends (never hitting the server with an empty multipart `image`).
        let empty_img = home.join("empty.jpg");
        std::fs::write(&empty_img, b"").unwrap();
        let model_id = folo_server::MockStore::new().list_models().unwrap()[0].id.clone();
        let mock_server =
            folo_server::Server::new(Arc::new(folo_server::MockStore::new()), Default::default());
        let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_url = format!("http://{}", http_listener.local_addr().unwrap());
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let http_task = tokio::spawn(mock_server.serve(http_listener, async move {
            let _ = stop_rx.await;
        }));
        let proxied_live = FoloMcp::new(
            Arc::new(HttpStore::new(&http_url, None)),
            Some(Arc::new(HttpStore::new(&http_url, None))),
            Options {
                data_dir: home.to_path_buf(),
                read_only: false,
                confine: false,
                device: "cpu".into(),
                published_weights: None,
            },
        );
        for backend in [&embedded, &proxied_live] {
            let res = backend
                .inner
                .predict_batch(
                    PredictBatchArgs {
                        model_id: model_id.clone(),
                        paths: Some(vec![empty_img.to_string_lossy().into_owned()]),
                        glob: None,
                        conf: 0.25,
                        iou: 0.7,
                    },
                    &mut Progress::none(),
                )
                .await
                .unwrap();
            let failed = res.value["failed"].as_array().unwrap();
            assert_eq!(failed.len(), 1);
            assert_eq!(failed[0]["code"], "invalid_arguments");
        }
        let _ = stop_tx.send(());
        let _ = http_task.await;
    }
}
