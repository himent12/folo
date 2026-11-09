//! Models of the real store: the weights of finished runs and the `.folo` files under `<data dir>/models`, their
//! metadata (read once per file version), validation results (`/eval`, computed with folo-vision's [`validate`]
//! and cached in memory and under `cache/eval/`) and exports.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_api::{
    Arch, ExportFormat, ExportResult, ExportSpec, Model, ModelEval, ModelMetrics, PrCurve, ValReport,
    ValTiming,
};
use folo_core::Device;
use folo_data::{Dataset as _, Split};
use folo_nn::io::{FloatDType, Namespace};
use folo_vision::NmsConfig;
use folo_vision::export::onnx::{self, OnnxOptions};
use folo_vision::val::{ValConfig, val_metrics, validate};
use serde::{Deserialize, Serialize};

use crate::engine::{self, FileKey, engine_error, nn_error};
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::runs::{DatasetRef, data_format};
use crate::store::weights_io;
use crate::util::{fnv1a, rfc3339};

/// Validation batch of `/eval`.
pub const EVAL_BATCH: usize = 16;

/// What a `.folo` file says about itself (the parts of `Model` that come from the file).
#[derive(Clone, Debug)]
pub struct FileMeta {
    /// `folo.arch`.
    pub arch: Arch,
    /// `folo.imgsz`.
    pub imgsz: u32,
    /// `folo.classes`.
    pub classes: Vec<String>,
    /// Learnable parameters stored (BatchNorm statistics are not counted).
    pub params: u64,
    /// File size.
    pub bytes: u64,
    /// `folo.created`, else the modification time.
    pub created: String,
    /// `folo.metrics`.
    pub metrics: Option<ModelMetrics>,
}

/// GFLOPs per image at 640 px (docs/MODEL.md §3, 80 classes); the class count changes them by well under 1%.
fn gflops_640(arch: Arch) -> f64 {
    match arch {
        Arch::N => 6.54,
        Arch::S => 21.59,
        Arch::M => 68.10,
        Arch::L => 87.15,
        Arch::X => 195.27,
    }
}

/// GFLOPs per image at `imgsz` (the 640 px figure scaled by the pixel count).
pub fn flops_g(arch: Arch, imgsz: u32) -> f64 {
    let g = gflops_640(arch) * (imgsz as f64 / 640.0).powi(2);
    (g * 100.0).round() / 100.0
}

fn is_buffer(name: &str) -> bool {
    name.ends_with("running_mean") || name.ends_with("running_var") || name.ends_with("num_batches_tracked")
}

fn read_meta(path: &Path) -> Result<FileMeta> {
    let file = engine::read_model_file(path)?;
    let m = file.metadata();
    let params = file
        .tensor_infos()
        .iter()
        .filter(|t| Namespace::of(&t.name) == Namespace::Model && !is_buffer(&t.name))
        .map(|t| t.shape.iter().product::<usize>() as u64)
        .sum();
    let fs = std::fs::metadata(path).map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
    let mtime = fs
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0.0, |d| d.as_secs_f64());
    Ok(FileMeta {
        arch: Arch::parse(&m.arch)
            .unwrap_or_else(|| Arch::parse(&format!("folo-{}", m.config.scale)).unwrap_or_default()),
        imgsz: m.imgsz,
        classes: m.classes.clone(),
        params,
        bytes: fs.len(),
        created: m.created.clone().unwrap_or_else(|| rfc3339(mtime)),
        metrics: m.metrics.map(|x| ModelMetrics {
            map50: x.map50 as f32,
            map50_95: x.map50_95 as f32,
            precision: x.precision as f32,
            recall: x.recall as f32,
        }),
    })
}

/// A model and where it comes from.
#[derive(Clone, Debug)]
pub struct ModelRef {
    /// The API record.
    pub model: Model,
    /// The weights file.
    pub path: PathBuf,
    /// The dataset it was trained on (run models).
    pub dataset: Option<DatasetRef>,
}

/// Builds the API record of a weights file. `metrics` are used as given (`None` = unknown).
pub fn model_of(
    id: String,
    name: String,
    meta: &FileMeta,
    run_id: Option<String>,
    metrics: Option<ModelMetrics>,
) -> Model {
    Model {
        id,
        name,
        arch: meta.arch,
        imgsz: meta.imgsz,
        classes: meta.classes.clone(),
        params: meta.params,
        flops_g: flops_g(meta.arch, meta.imgsz),
        bytes: meta.bytes,
        created: meta.created.clone(),
        run_id,
        metrics,
    }
}

/// The id of a `.folo` file under the models folder: `m_` and 8 hex digits of its path's hash (stable across
/// restarts; run models use 4-digit ids, so the two never meet).
pub fn file_model_id(path: &Path) -> String {
    let h = fnv1a(path.to_string_lossy().as_bytes());
    format!("m_{:08x}", (h >> 32) as u32 ^ h as u32)
}

#[derive(Serialize, Deserialize)]
struct CachedEval {
    key: String,
    eval: ModelEval,
}

/// Metadata, eval and f16 weights caches.
pub struct Models {
    data_dir: PathBuf,
    meta: Mutex<HashMap<PathBuf, (FileKey, Arc<FileMeta>)>>,
    evals: Mutex<HashMap<String, (String, ModelEval)>>,
    /// Held while an f16 weights file is written, so concurrent first requests export once.
    f16: Mutex<()>,
}

impl Models {
    /// Caches rooted at `data_dir`.
    pub fn new(data_dir: PathBuf) -> Models {
        Models { data_dir, meta: Mutex::default(), evals: Mutex::default(), f16: Mutex::default() }
    }

    fn weights_cache(&self) -> PathBuf {
        self.data_dir.join("cache").join("weights")
    }

    /// The `folo-f16` export of `m` for `GET /api/models/{id}/weights?format=f16`, opened (so removing an older
    /// copy never pulls it from under a request): `cache/weights/<model id>.<hash of the weights file's path, size
    /// and mtime>.f16.folo`, written by the export code of `POST /api/models/{id}/export` the first time it is asked
    /// for (older versions of the model's file are then removed).
    pub fn f16_weights(&self, m: &ModelRef) -> Result<(PathBuf, std::fs::File)> {
        let id = m.model.id.as_str();
        let fk = FileKey::of(&m.path).map_err(|e| weights_io(e, &m.path, id))?;
        let mtime = fk
            .modified
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        let key = fnv1a(format!("{}|{}|{mtime}", fk.path.display(), fk.len).as_bytes());
        let dir = self.weights_cache();
        // Model ids have no dots, so `<id>.` never prefixes another model's files.
        let out = dir.join(format!("{id}.{key:016x}.f16.folo"));
        let open = || match std::fs::File::open(&out) {
            Ok(f) => Ok(Some(f)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::internal(format!("{}: {e}", out.display()))),
        };
        if let Some(f) = open()? {
            return Ok((out, f));
        }
        let _guard = self.f16.lock().unwrap();
        if let Some(f) = open()? {
            return Ok((out, f));
        }
        // Its run may have been deleted (and its cache cleared, under this lock) since the lookup.
        if !m.path.is_file() {
            return Err(Error::not_found("model", id));
        }
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        let file = engine::read_model_file(&m.path)
            .map_err(|e| if m.path.exists() { e } else { Error::not_found("model", id) })?;
        // Written to a temporary file and renamed into place, so a reader never sees a partial export.
        file.export(&out, FloatDType::F16).map_err(nn_error)?;
        // Opened before the lock is released: a newer export's cleanup may remove the file, not this handle.
        let f = open()?.ok_or_else(|| Error::internal(format!("{} vanished", out.display())))?;
        self.remove_weights(id, Some(&out));
        Ok((out, f))
    }

    /// Removes the cached f16 weights of model `id` (when its run is deleted). Waits for an export in progress, so
    /// none is left behind.
    pub fn forget_weights(&self, id: &str) {
        let _guard = self.f16.lock().unwrap();
        self.remove_weights(id, None);
    }

    /// Removes the cached f16 weights of model `id` except `keep`; the caller holds the export lock.
    fn remove_weights(&self, id: &str, keep: Option<&Path>) {
        let prefix = format!("{id}.");
        let Ok(entries) = std::fs::read_dir(self.weights_cache()) else { return };
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && name.ends_with(".f16.folo") && keep != Some(e.path().as_path()) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    /// The folder scanned for `.folo` files.
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// The metadata of `path`, read again only when the file changed.
    pub fn meta(&self, path: &Path) -> Result<Arc<FileMeta>> {
        let key = FileKey::of(path).map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
        if let Some((k, m)) = self.meta.lock().unwrap().get(path)
            && *k == key
        {
            return Ok(m.clone());
        }
        let m = Arc::new(read_meta(path)?);
        self.meta.lock().unwrap().insert(path.to_path_buf(), (key, m.clone()));
        Ok(m)
    }

    /// Every `.folo` file under the models folder (recursively), as models.
    pub fn scan(&self) -> Vec<ModelRef> {
        let dir = self.models_dir();
        let mut out = Vec::new();
        let mut files: Vec<PathBuf> = walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("folo")))
            .collect();
        files.sort();
        for path in files {
            match self.meta(&path) {
                Ok(meta) => {
                    let name =
                        path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
                    let model = model_of(file_model_id(&path), name, &meta, None, meta.metrics);
                    out.push(ModelRef { model, path, dataset: None });
                }
                Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
            }
        }
        out
    }

    /// `/eval` of `m` on the validation split (training split when there is none) of `dataset`, whose index has
    /// fingerprint `ds_key`. Cached per (weights file version, dataset version); computed every time when the
    /// dataset has no fingerprint (it is not registered, so a change to its files could not be noticed).
    pub fn eval(
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
        let key = format!("{}|{}|{mtime}|{}|{ds_key}", fk.path.display(), fk.len, dataset.id);
        if let Some((k, e)) = self.evals.lock().unwrap().get(&m.model.id)
            && *k == key
        {
            return Ok(e.clone());
        }
        let disk = self.data_dir.join("cache").join("eval").join(format!("{}.json", m.model.id));
        if let Some(c) = std::fs::read(&disk).ok().and_then(|b| serde_json::from_slice::<CachedEval>(&b).ok())
            && c.key == key
        {
            self.evals.lock().unwrap().insert(m.model.id.clone(), (key, c.eval.clone()));
            return Ok(c.eval);
        }
        let eval = evaluate(&m.path, dataset, device)?;
        let bytes = serde_json::to_vec(&CachedEval { key: key.clone(), eval: eval.clone() })
            .expect("evals serialize");
        if let Err(e) = write_atomic(&disk, &bytes) {
            tracing::warn!("caching {}: {e}", disk.display());
        }
        self.evals.lock().unwrap().insert(m.model.id.clone(), (key, eval.clone()));
        Ok(eval)
    }

    /// Exports `m` to `<data dir>/exports/`: `folo-f16` as `<name>-f16.folo`, `onnx` as `<name>.onnx` (the
    /// defaults of `folo export --format onnx`: f32, batch 1, the model's imgsz, raw head outputs). `-2`, `-3`,
    /// ... when the name is taken; never overwriting.
    pub fn export(&self, m: &ModelRef, spec: ExportSpec) -> Result<ExportResult> {
        let format = spec.format;
        let file = engine::read_model_file(&m.path)?;
        let dir = self.data_dir.join("exports");
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        let base = m.model.name.replace(['/', '\\', ':'], "-");
        let (suffix, ext) = match format {
            ExportFormat::FoloF16 => ("-f16", "folo"),
            ExportFormat::Onnx if spec.onnx.half => ("-f16", "onnx"),
            ExportFormat::Onnx => ("", "onnx"),
        };
        // create_new claims the name, so two exports never share a file; the export then replaces it atomically.
        let mut n = 1;
        let out = loop {
            let tag = if n == 1 { String::new() } else { format!("-{n}") };
            let p = dir.join(format!("{base}{suffix}{tag}.{ext}"));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
                Ok(_) => break p,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => return Err(Error::internal(format!("creating {}: {e}", p.display()))),
            }
        };
        let written = match format {
            ExportFormat::FoloF16 => file.export(&out, FloatDType::F16).map_err(nn_error),
            ExportFormat::Onnx => {
                let mut opts = OnnxOptions::new(file.metadata().imgsz);
                (opts.decode, opts.dynamic, opts.half) =
                    (spec.onnx.decode, spec.onnx.dynamic, spec.onnx.half);
                opts.metadata = file.metadata().for_weights().to_map();
                onnx::export_file(&file, &opts).and_then(|x| x.write_to(&out)).map_err(engine_error)
            }
        };
        if let Err(e) = written {
            let _ = std::fs::remove_file(&out);
            return Err(e);
        }
        let bytes = std::fs::metadata(&out).map(|x| x.len()).unwrap_or(0);
        Ok(ExportResult { path: folo_api::json::path(&out), bytes })
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
    Ok(eval_of(&r.report))
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
/// confusion matrix transposed to rows = truth, and the per-class metrics.
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
    }
}
