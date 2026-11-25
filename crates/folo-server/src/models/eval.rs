//! Single-flight cached model evaluation (`/api/models/{id}/eval`) and ad-hoc validation (`POST /api/val`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_api::{ModelEval, PrCurve, SplitFallback, ValReport, ValTiming};
use folo_core::Device;
use folo_data::{Dataset as _, Split};
use folo_vision::NmsConfig;
use folo_vision::val::{ValConfig, val_metrics, validate};
use serde::{Deserialize, Serialize};

use crate::engine::{self, FileKey, engine_error};
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::runs::{DatasetRef, data_format};
use crate::util::Flight;

use super::{BoundedCache, ModelRef, Models};

/// Validation batch of `/eval`.
pub const EVAL_BATCH: usize = 16;

/// Bumped when [`ModelEval`] gains a field the engine computes, so older `cache/eval/` files are recomputed.
const EVAL_CACHE_VERSION: &str = "v3";

#[derive(Serialize, Deserialize)]
struct CachedEval {
    key: String,
    eval: ModelEval,
}

type EvalFlight = Arc<Flight<Result<ModelEval>>>;

pub(super) struct EvalCache {
    pub(super) done: BoundedCache<String, (String, ModelEval)>,
    pub(super) in_flight: HashMap<(String, String), EvalFlight>,
}

impl EvalCache {
    pub(super) fn new(cap: usize) -> Self {
        Self { done: BoundedCache::new(cap), in_flight: HashMap::new() }
    }
}

impl Models {
    /// `/eval` of `m` on the validation split (training split when there is none) of `dataset`, whose index has
    /// fingerprint `ds_key`. Cached per (weights file version, dataset version); computed every time when the
    /// dataset has no fingerprint (it is not registered, so a change to its files could not be noticed). The
    /// dataset's id and name (also in a fallback's message) are always `dataset`'s, not the cached ones (a rename
    /// does not invalidate the cache).
    pub fn eval(
        &self,
        m: &ModelRef,
        dataset: &DatasetRef,
        ds_key: Option<&str>,
        device: Device,
    ) -> Result<ModelEval> {
        self.eval_cached(m, dataset, ds_key, device).map(|e| ModelEval {
            dataset_id: Some(dataset.id.clone()),
            dataset_name: Some(dataset.name.clone()),
            // Its message names the dataset too.
            split_fallback: e.split_fallback.as_ref().zip(e.split).map(|(f, used)| {
                SplitFallback::empty_split(f.requested, used, &dataset.name, u64::from(e.images))
            }),
            ..e
        })
    }

    fn eval_cached(
        &self,
        m: &ModelRef,
        dataset: &DatasetRef,
        ds_key: Option<&str>,
        device: Device,
    ) -> Result<ModelEval> {
        let Some(ds_key) = ds_key else { return evaluate(&m.path, dataset, device) };
        let fk = FileKey::of(&m.path).map_err(|e| Error::internal(format!("{}: {e}", m.path.display())))?;
        let mtime = fk
            .modified
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        // `EVAL_CACHE_VERSION` first: results cached before `best_f1_conf` or `split` existed are computed again.
        let key =
            format!("{EVAL_CACHE_VERSION}|{}|{}|{mtime}|{}|{ds_key}", fk.path.display(), fk.len, dataset.id);
        self.eval_cached_with(&m.model.id, &key, || evaluate(&m.path, dataset, device))
    }

    /// Single-flight cached evaluation lookup/compute for `(model_id, key)`: concurrent callers for the same key
    /// share one disk check / evaluation while distinct keys run in parallel.
    pub(super) fn eval_cached_with(
        &self,
        model_id: &str,
        key: &str,
        mut run: impl FnMut() -> Result<ModelEval>,
    ) -> Result<ModelEval> {
        let fkey = (model_id.to_string(), key.to_string());
        loop {
            let role = {
                let mut cache = self.evals.lock().unwrap_or_else(|p| p.into_inner());
                if let Some((k, e)) = cache.done.get(model_id)
                    && k == key
                {
                    return Ok(e.clone());
                }
                if let Some(existing) = cache.in_flight.get(&fkey).cloned() {
                    Err(existing)
                } else {
                    let created = Flight::new();
                    cache.in_flight.insert(fkey.clone(), created.clone());
                    Ok(created)
                }
            };
            match role {
                Err(existing) => {
                    if let Some(res) = existing.wait() {
                        return res;
                    }
                    // Leader panicked/aborted; retry.
                }
                Ok(mine) => {
                    struct Cleanup<'a> {
                        evals: &'a Mutex<EvalCache>,
                        key: (String, String),
                        flight: EvalFlight,
                        done: bool,
                    }
                    impl Drop for Cleanup<'_> {
                        fn drop(&mut self) {
                            if !self.done {
                                let mut cache = self.evals.lock().unwrap_or_else(|p| p.into_inner());
                                if cache
                                    .in_flight
                                    .get(&self.key)
                                    .is_some_and(|f| Arc::ptr_eq(f, &self.flight))
                                {
                                    cache.in_flight.remove(&self.key);
                                }
                                drop(cache);
                                self.flight.abort();
                            }
                        }
                    }
                    let mut guard =
                        Cleanup { evals: &self.evals, key: fkey.clone(), flight: mine.clone(), done: false };
                    let disk = self.data_dir.join("cache").join("eval").join(format!("{model_id}.json"));
                    let outcome = if let Some(c) =
                        std::fs::read(&disk).ok().and_then(|b| serde_json::from_slice::<CachedEval>(&b).ok())
                        && c.key == key
                    {
                        Ok(c.eval)
                    } else {
                        match run() {
                            Ok(eval) => {
                                let bytes = serde_json::to_vec(&CachedEval {
                                    key: key.to_string(),
                                    eval: eval.clone(),
                                })
                                .expect("evals serialize");
                                if let Err(e) = write_atomic(&disk, &bytes) {
                                    tracing::warn!("caching {}: {e}", disk.display());
                                }
                                Ok(eval)
                            }
                            Err(e) => Err(e),
                        }
                    };
                    {
                        let mut cache = self.evals.lock().unwrap_or_else(|p| p.into_inner());
                        if cache.in_flight.get(&fkey).is_some_and(|f| Arc::ptr_eq(f, &mine)) {
                            cache.in_flight.remove(&fkey);
                        }
                        if let Ok(eval) = &outcome {
                            cache.done.insert(model_id.to_string(), (key.to_string(), eval.clone()));
                        }
                    }
                    mine.finish(outcome.clone());
                    guard.done = true;
                    return outcome;
                }
            }
        }
    }
}

/// Opens `dataset` for validating `model`, whose classes it must have (in the same order).
fn open_for(model: &folo_vision::FoloModel, dataset: &DatasetRef) -> Result<folo_data::LoadedDataset> {
    let opts = folo_data::OpenOptions { format: data_format(dataset.format), verify_images: true };
    let ds = folo_data::open_with(Path::new(&dataset.path), &opts)
        .map_err(|e| crate::index::data_error(e, &dataset.path))?;
    if ds.classes != model.classes() {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
            format!(
                "the dataset's classes {:?} are not the model's {:?} (in the same order)",
                ds.classes,
                model.classes()
            ),
        ));
    }
    Ok(ds)
}

/// Validates the weights at `path` on `dataset` and shapes the report as `/eval`.
fn evaluate(path: &Path, dataset: &DatasetRef, device: Device) -> Result<ModelEval> {
    // The cached evaluation is the model's reference score: always f32.
    let model = engine::open_model(path, device, folo_vision::Precision::F32)?;
    let ds = open_for(&model, dataset)?;
    let split = if ds.val.is_empty() { Split::Train } else { Split::Val };
    let part = ds.split(split);
    if part.is_empty() {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
            "the dataset has no images to validate on",
        ));
    }
    let cfg = ValConfig { samples: 0, ..ValConfig::new(model.imgsz(), EVAL_BATCH) };
    let r = validate(&model, Arc::new(part.clone()), &cfg).map_err(engine_error)?;
    let used = if split == Split::Val { folo_api::Split::Val } else { folo_api::Split::Train };
    let fallback = (used != folo_api::Split::Val).then(|| {
        let f = SplitFallback::empty_split(folo_api::Split::Val, used, &dataset.name, part.len() as u64);
        tracing::warn!(dataset = %dataset.id, "/eval: {}", f.message);
        f
    });
    Ok(ModelEval {
        dataset_id: Some(dataset.id.clone()),
        dataset_name: Some(dataset.name.clone()),
        split: Some(used),
        split_fallback: fallback,
        ..eval_of(&r.report)
    })
}

/// `POST /api/val`: validates the weights of `m` on `split` of `dataset` at `imgsz` (the model's own size when
/// `None`) with NMS thresholds `conf` / `iou` (the rest of MODEL.md §8.7's validation settings), at `precision`
/// (f16 only where the device computes in it).
#[allow(clippy::too_many_arguments)]
pub fn val(
    m: &ModelRef,
    dataset: &DatasetRef,
    split: folo_api::Split,
    imgsz: Option<u32>,
    conf: f32,
    iou: f32,
    device: Device,
    precision: folo_vision::Precision,
) -> Result<ValReport> {
    let bad = |msg: String| Error::new(StatusCode::BAD_REQUEST, "invalid_body", msg);
    let unit = |x: f32| (0.0..=1.0).contains(&x); // false for NaN
    if !unit(conf) || !unit(iou) {
        return Err(bad(format!("conf and iou must be in 0..=1, got conf {conf} iou {iou}")));
    }
    if let Some(s) = imgsz
        && (!s.is_multiple_of(32) || !(32..=4096).contains(&s))
    {
        return Err(bad(format!("imgsz must be a multiple of 32 in 32..=4096, got {s}")));
    }
    let t0 = std::time::Instant::now();
    let model = engine::open_model(&m.path, device, precision)?;
    let ds = open_for(&model, dataset)?;
    let part = ds.split(match split {
        folo_api::Split::Train => Split::Train,
        folo_api::Split::Val => Split::Val,
        folo_api::Split::Test => Split::Test,
    });
    if part.is_empty() {
        // `invalid_dataset`, as `folo val` reports an empty split.
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
            format!("dataset '{}' has no {} images", dataset.id, split.as_str()),
        ));
    }
    let imgsz = imgsz.unwrap_or(model.imgsz());
    let cfg = ValConfig {
        samples: 0,
        nms: NmsConfig { conf, iou, ..NmsConfig::val() },
        ..ValConfig::new(imgsz, EVAL_BATCH)
    };
    let r = validate(&model, Arc::new(part.clone()), &cfg).map_err(engine_error)?;
    let images = r.report.num_images;
    let total_s = t0.elapsed().as_secs_f64();
    let per_image_ms = if images == 0 { 0.0 } else { r.seconds * 1e3 / images as f64 };
    Ok(ValReport {
        model_id: m.model.id.clone(),
        dataset_id: dataset.id.clone(),
        dataset_name: Some(dataset.name.clone()),
        split,
        imgsz,
        images,
        val: val_metrics(&r.report),
        timing: ValTiming {
            total_s: (total_s * 1000.0).round() / 1000.0,
            per_image_ms: (per_image_ms * 100.0).round() / 100.0,
        },
    })
}

/// docs/API.md's `/eval` shape of an evaluation report: PR curves (class order, 101 recall points), the
/// confusion matrix transposed to rows = truth (with its thresholds), and the per-class metrics. The dataset and
/// split are the caller's to fill.
pub fn eval_of(r: &folo_vision::EvalReport) -> ModelEval {
    let recall = r.curves.recall.clone();
    let pr = r
        .curves
        .per_class
        .iter()
        .filter_map(|c| {
            let class = c.class?;
            let ap50 = if c.pr.is_empty() { 0.0 } else { c.pr.iter().sum::<f32>() / c.pr.len() as f32 };
            Some(PrCurve { class, recall: recall.clone(), precision: c.pr.clone(), ap50 })
        })
        .collect();
    let m = &r.confusion.matrix;
    let n = m.len();
    let confusion = (0..n).map(|t| (0..n).map(|p| m[p][t].min(u32::MAX as u64) as u32).collect()).collect();
    ModelEval {
        pr,
        confusion,
        per_class: val_metrics(r).per_class,
        images: r.num_images.min(u32::MAX as u64) as u32,
        best_f1_conf: val_metrics(r).best_f1_conf,
        dataset_id: None,
        dataset_name: None,
        split: None,
        split_fallback: None,
        confusion_conf: Some(r.confusion.conf),
        confusion_iou: Some(r.confusion.iou),
    }
}
