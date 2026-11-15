//! Models of the real store: the weights of finished runs, the `.folo` files under `<data dir>/models` and the
//! published weights downloaded with `folo weights pull` ([`folo_api::weights`]), their metadata (read once per file version), validation results (`/eval`, computed with folo-vision's [`validate`]
//! and cached in memory and under `cache/eval/`) and exports.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_api::{
    Arch, ExportFormat, ExportResult, ExportSpec, Head, Model, ModelEval, ModelMetrics, ModelPrecision,
    ModelSource, PrCurve, SplitFallback, TrainedWith, ValReport, ValTiming,
};
use folo_core::Device;
use folo_data::{Dataset as _, Split};
use folo_nn::io::safetensors::Container;
use folo_nn::io::{FloatDType, Metadata, Namespace};
use folo_vision::NmsConfig;
use folo_vision::export::onnx::{self, OnnxOptions};
use folo_vision::val::{ValConfig, val_metrics, validate};
use serde::{Deserialize, Serialize};

use crate::engine::{self, FileKey, engine_error, nn_error};
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::runs::{DatasetRef, data_format};
use crate::store::{WeightsFormat, weights_io};
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
    /// `folo.config`'s head.
    pub head: Head,
    /// From `folo.train`, when the file has it.
    pub trained_with: Option<TrainedWith>,
    /// `folo.dtype`, when the file has a known one.
    pub precision: Option<ModelPrecision>,
    /// `folo.fused`, unless its value is unknown.
    pub fused: Option<bool>,
}

/// [`Model::precision`] from a file's `folo.dtype`: `None` when it is missing (a file from another writer; FOLO's
/// always writes it) or not one this server knows.
fn precision(dtype: Option<&str>) -> Option<ModelPrecision> {
    match dtype? {
        "F32" => Some(ModelPrecision::F32),
        "F16" => Some(ModelPrecision::F16),
        _ => None,
    }
}

/// [`Model::fused`] from a file's `folo.fused`: writers leave the key out of unfused files (docs/FORMAT.md), so a
/// missing key is `false`; a value other than `"true"` / `"false"` is unknown.
fn fused(fused: Option<&str>) -> Option<bool> {
    match fused {
        None | Some("false") => Some(false),
        Some("true") => Some(true),
        Some(_) => None,
    }
}

/// The parts of `folo.train` (the trainer's resolved config) that [`TrainedWith`] shows. Files written before
/// mixed precision have no `amp`.
#[derive(Deserialize)]
struct TrainMeta {
    #[serde(default)]
    amp: bool,
    imgsz: u32,
    epochs: u32,
}

/// [`TrainedWith`] from a file's `folo.train`; `None` when it is missing or unreadable (the key is informational,
/// so a file with an odd one still lists).
fn trained_with(train: Option<&String>) -> Option<TrainedWith> {
    let t: TrainMeta = serde_json::from_str(train?).ok()?;
    Some(TrainedWith { amp: t.amp, imgsz: t.imgsz, epochs: t.epochs })
}

/// GFLOPs per image at 640 px (docs/MODEL.md §3, 80 classes); the class count changes them by well under 1%.
pub(crate) fn gflops_640(arch: Arch) -> f64 {
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

/// Reads only the file's header: the listing reads every file once per version, and a run checkpoint is several
/// times the size of its weights. The tensor data is checked when the file is loaded.
pub(crate) fn read_meta(path: &Path) -> Result<FileMeta> {
    let header = Container::read_header(path).map_err(|e| match e {
        folo_nn::Error::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => Error::new(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("{} no longer exists", path.display()),
        ),
        e => nn_error(e),
    })?;
    let mut map = header.metadata.unwrap_or_default();
    let precision = precision(map.get("folo.dtype").map(String::as_str));
    let fused = fused(map.get("folo.fused").map(String::as_str));
    // An unknown dtype or fused flag leaves that field unknown; the file still lists (loading it says what is
    // wrong), so the metadata is parsed without the key.
    if precision.is_none() {
        map.remove("folo.dtype");
    }
    if fused.is_none() {
        map.remove("folo.fused");
    }
    let m = &Metadata::from_map(&map).map_err(nn_error)?;
    let params = header
        .tensors
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
            best_f1_conf: x.best_f1_conf.map(|c| c as f32),
        }),
        head: if m.config.has_o2o() { Head::O2o } else { Head::O2m },
        trained_with: trained_with(m.extra.get("folo.train")),
        precision,
        fused,
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

impl ModelRef {
    /// A model and its file; `Model.path` is the file (`/` separators).
    pub fn new(mut model: Model, path: PathBuf, dataset: Option<DatasetRef>) -> ModelRef {
        model.path = Some(folo_api::json::path(&path));
        ModelRef { model, path, dataset }
    }
}

/// Builds the API record of a weights file. `metrics` are used as given (`None` = unknown). The source is `run`
/// with a `run_id`, else `file` (published weights set theirs after).
pub fn model_of(
    id: String,
    name: String,
    meta: &FileMeta,
    run_id: Option<String>,
    metrics: Option<ModelMetrics>,
) -> Model {
    Model {
        source: Some(if run_id.is_some() { ModelSource::Run } else { ModelSource::File }),
        about: None,
        // Set by `ModelRef::new`.
        path: None,
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
        head: meta.head,
        trained_with: meta.trained_with,
        precision: meta.precision,
        fused: meta.fused,
    }
}

/// The id of a `.folo` file under the models folder: `m_` and 8 hex digits of its path's hash (stable across
/// restarts; run models use 4-digit ids, so the two never meet).
pub fn file_model_id(path: &Path) -> String {
    let h = fnv1a(path.to_string_lossy().as_bytes());
    format!("m_{:08x}", (h >> 32) as u32 ^ h as u32)
}

/// Bumped when [`ModelEval`] gains a field the engine computes, so older `cache/eval/` files are recomputed.
const EVAL_CACHE_VERSION: &str = "v3";

#[derive(Serialize, Deserialize)]
struct CachedEval {
    key: String,
    eval: ModelEval,
}

/// Metadata, eval and f16 weights caches.
pub struct Models {
    data_dir: PathBuf,
    /// The `folo weights` cache whose downloaded files list as published models (`None`: none are listed).
    published_dir: Option<PathBuf>,
    meta: Mutex<HashMap<PathBuf, (FileKey, Arc<FileMeta>)>>,
    evals: Mutex<HashMap<String, (String, ModelEval)>>,
    /// The weights files listed as skipped (unreadable), each with the version that was: warned about once.
    skipped: Mutex<HashMap<PathBuf, FileKey>>,
    /// Held while a derived weights file (f16, ONNX) is written, so concurrent first requests export once.
    derived: Mutex<()>,
}

impl Models {
    /// Caches rooted at `data_dir`.
    pub fn new(data_dir: PathBuf) -> Models {
        Models {
            data_dir,
            published_dir: None,
            meta: Mutex::default(),
            evals: Mutex::default(),
            skipped: Mutex::default(),
            derived: Mutex::default(),
        }
    }

    /// Says in the log that the weights file at `path` is not listed, why, and only once per version of the file:
    /// the models are listed again on every `GET /api/models`, which both apps poll (x3-server-8). Returns whether
    /// it was said now.
    pub fn warn_skipped(&self, path: &Path, err: &Error) -> bool {
        let Ok(key) = FileKey::of(path) else { return false };
        if self.skipped.lock().unwrap().insert(path.to_path_buf(), key.clone()).as_ref() == Some(&key) {
            return false;
        }
        tracing::warn!("skipping {}: {err}", path.display());
        true
    }

    /// Lists the published weights downloaded into `dir` (the `folo weights` cache, [`folo_api::weights::cache_dir`])
    /// as models.
    pub fn set_published(&mut self, dir: Option<PathBuf>) {
        self.published_dir = dir;
    }

    /// The `folo weights` cache this store lists published models from.
    pub fn published_dir(&self) -> Option<&Path> {
        self.published_dir.as_deref()
    }

    /// The published weights downloaded into the cache, as models: id and name are the published name (what
    /// `folo weights pull` and `train --weights` take), `source: published`, `about` from the registry.
    pub fn published(&self) -> Vec<ModelRef> {
        let Some(dir) = &self.published_dir else { return Vec::new() };
        let mut out = Vec::new();
        for p in folo_api::weights::PUBLISHED {
            let path = dir.join(p.asset);
            if !path.is_file() {
                continue;
            }
            match self.meta(&path) {
                Ok(meta) => {
                    let mut model =
                        model_of(p.name.to_string(), p.name.to_string(), &meta, None, meta.metrics);
                    model.source = Some(ModelSource::Published);
                    model.about = Some(p.about.to_string());
                    out.push(ModelRef::new(model, path, None));
                }
                Err(e) => {
                    self.warn_skipped(&path, &e);
                }
            }
        }
        out
    }

    /// The published model `name`, when it is downloaded.
    pub fn published_model(&self, name: &str) -> Option<ModelRef> {
        folo_api::weights::find(name)?;
        self.published().into_iter().find(|m| m.model.id == name)
    }

    fn weights_cache(&self) -> PathBuf {
        self.data_dir.join("cache").join("weights")
    }

    /// The derived weights of `m` for `GET /api/models/{id}/weights?format=f16|onnx`, opened (so removing an
    /// older copy never pulls it from under a request): `cache/weights/<model id>.<hash of the weights file's
    /// path, size and mtime>.<variant>` (`f16.folo`, or `onnx` with the switches, e.g. `decode-half.onnx`), written
    /// by the export code of `POST /api/models/{id}/export` the first time it is asked for (copies made from
    /// older versions of the model's file are then removed). `format` is [`WeightsFormat::F16`] or
    /// [`WeightsFormat::Onnx`].
    pub fn derived_weights(&self, m: &ModelRef, format: WeightsFormat) -> Result<(PathBuf, std::fs::File)> {
        let id = m.model.id.as_str();
        let fk = FileKey::of(&m.path).map_err(|e| weights_io(e, &m.path, id))?;
        let mtime = fk
            .modified
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        let key = fnv1a(format!("{}|{}|{mtime}", fk.path.display(), fk.len).as_bytes());
        let dir = self.weights_cache();
        // Model ids have no dots, so `<id>.` never prefixes another model's files.
        let version = format!("{id}.{key:016x}.");
        let out = dir.join(format!("{version}{}", cache_variant(format)));
        let open = || match std::fs::File::open(&out) {
            Ok(f) => Ok(Some(f)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::internal(format!("{}: {e}", out.display()))),
        };
        if let Some(f) = open()? {
            return Ok((out, f));
        }
        let _guard = self.derived.lock().unwrap();
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
        match format {
            WeightsFormat::Onnx(switches) => {
                onnx_export(&file, switches)?.write_to(&out).map_err(engine_error)?
            }
            _ => file.export(&out, FloatDType::F16).map_err(nn_error)?,
        }
        // Opened before the lock is released: a newer export's cleanup may remove the file, not this handle.
        let f = open()?.ok_or_else(|| Error::internal(format!("{} vanished", out.display())))?;
        self.remove_weights(id, Some(&version));
        Ok((out, f))
    }

    /// Removes the cached derived weights of model `id` (when its run is deleted). Waits for an export in
    /// progress, so none is left behind.
    pub fn forget_weights(&self, id: &str) {
        let _guard = self.derived.lock().unwrap();
        self.remove_weights(id, None);
    }

    /// Removes the cached derived weights of model `id` except those named `keep…` (the current version of its
    /// file); the caller holds the export lock.
    fn remove_weights(&self, id: &str, keep: Option<&str>) {
        let prefix = format!("{id}.");
        let Ok(entries) = std::fs::read_dir(self.weights_cache()) else { return };
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && !keep.is_some_and(|k| name.starts_with(k)) {
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
                    out.push(ModelRef::new(model, path, None));
                }
                Err(e) => {
                    self.warn_skipped(&path, &e);
                }
            }
        }
        out
    }

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
    /// defaults of `folo export --format onnx`: f32, batch 1, the model's imgsz, raw head outputs; `-decoded` with
    /// `decode`, `-f16` with `half`, as the CLI names them). `-2`, `-3`, ... when the name is taken; never
    /// overwriting.
    pub fn export(&self, m: &ModelRef, spec: ExportSpec) -> Result<ExportResult> {
        let format = spec.format;
        let file = engine::read_model_file(&m.path)?;
        let dir = self.data_dir.join("exports");
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        let base = export_stem(&m.model.name);
        let (suffix, ext) = match format {
            ExportFormat::FoloF16 => ("-f16", "folo"),
            ExportFormat::Onnx => (spec.onnx.file_suffix(), "onnx"),
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
                onnx_export(&file, spec.onnx).and_then(|x| x.write_to(&out).map_err(engine_error))
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

/// The start of an export's file name for a model called `name` (a run's name is any text): the characters no file
/// name may hold, on Windows either, become `-`, and it is cut to [`MAX_STEM_BYTES`], so every model can be exported.
fn export_stem(name: &str) -> String {
    let mut stem = String::new();
    for c in name.chars() {
        let c = if c.is_control() || matches!(c, '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*') {
            '-'
        } else {
            c
        };
        if stem.len() + c.len_utf8() > MAX_STEM_BYTES {
            break;
        }
        stem.push(c);
    }
    stem
}

/// Longest [`export_stem`], in bytes: file names hold 255, and the export's suffix and `-2`, ... follow.
const MAX_STEM_BYTES: usize = 200;

/// The `onnx` export of `POST /api/models/{id}/export` (and `/weights?format=onnx`): `folo export --format onnx`'s
/// defaults (f32, batch 1, the model's training imgsz, raw head outputs) unless a switch says otherwise, with the
/// file's metadata. A model the exporter cannot map is `501 not_implemented`.
pub fn onnx_export(
    file: &folo_nn::io::ModelFile,
    switches: folo_api::OnnxSwitches,
) -> Result<onnx::OnnxExport> {
    let mut opts = OnnxOptions::new(file.metadata().imgsz);
    (opts.decode, opts.dynamic, opts.half) = (switches.decode, switches.dynamic, switches.half);
    opts.metadata = file.metadata().for_weights().to_map();
    onnx::export_file(file, &opts).map_err(engine_error)
}

/// The variant part of a derived weights file's name in `cache/weights/` (after `<model id>.<hash>.`).
fn cache_variant(format: WeightsFormat) -> String {
    match format {
        WeightsFormat::Onnx(s) => {
            let on: Vec<&str> = [("decode", s.decode), ("dynamic", s.dynamic), ("half", s.half)]
                .into_iter()
                .filter_map(|(n, on)| on.then_some(n))
                .collect();
            if on.is_empty() { "onnx".into() } else { format!("{}.onnx", on.join("-")) }
        }
        _ => "f16.folo".into(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_with_reads_folo_train_leniently() {
        let t = |s: &str| trained_with(Some(&s.to_string()));
        // The trainer's resolved config: the three fields, other keys ignored.
        let full = r#"{"name":"a","epochs":80,"imgsz":512,"amp":true,"head":"o2o","augment":{"mosaic":1.0}}"#;
        assert_eq!(t(full), Some(TrainedWith { amp: true, imgsz: 512, epochs: 80 }));
        // Written before mixed precision: no `amp`.
        assert_eq!(t(r#"{"epochs":3,"imgsz":64}"#), Some(TrainedWith { amp: false, imgsz: 64, epochs: 3 }));
        // Missing, unparsable or without the fields: none, never an error.
        assert_eq!(trained_with(None), None);
        for bad in [
            "",
            "not json",
            "[]",
            r#"{"epochs":3}"#,
            r#"{"epochs":-1,"imgsz":64}"#,
            r#"{"amp":"yes","epochs":3,"imgsz":64}"#,
        ] {
            assert_eq!(t(bad), None, "{bad}");
        }
    }

    #[test]
    fn precision_and_fused_read_their_keys_leniently() {
        assert_eq!(precision(Some("F32")), Some(ModelPrecision::F32));
        assert_eq!(precision(Some("F16")), Some(ModelPrecision::F16));
        for unknown in [None, Some("BF16"), Some("f16"), Some("")] {
            assert_eq!(precision(unknown), None, "{unknown:?}");
        }
        assert_eq!(
            (fused(None), fused(Some("false")), fused(Some("true"))),
            (Some(false), Some(false), Some(true))
        );
        assert_eq!((fused(Some("yes")), fused(Some("TRUE"))), (None, None));
    }

    /// A run model is named after its run, and a run's name is any text: the export still gets a file.
    #[test]
    fn models_with_any_name_export() {
        let dir = folo_test_support::TestDir::new("models-export-names");
        let path = dir.join("w.folo");
        let mut m = folo_vision::FoloModel::with_rng(
            folo_nn::io::ModelConfig::new("n", 1).unwrap(),
            &mut folo_core::rng::Rng::seed(1),
        )
        .unwrap();
        m.set_classes(vec!["cat".into()]).unwrap();
        m.set_imgsz(64);
        m.save(&path).unwrap();
        let models = Models::new(dir.join("home"));
        let meta = read_meta(&path).unwrap();
        let spec = ExportSpec { format: ExportFormat::FoloF16, onnx: Default::default() };
        for name in ["cats <v2>? \"best\" | *-best", "tab\there-best", &"long".repeat(80)] {
            let m = ModelRef::new(model_of("m_1".into(), name.into(), &meta, None, None), path.clone(), None);
            let out = models.export(&m, spec).unwrap_or_else(|e| panic!("{name:?}: {e}"));
            let file = Path::new(&out.path).file_name().unwrap().to_string_lossy().into_owned();
            assert!(file.ends_with("-f16.folo") && file.len() <= 255, "{file}");
            assert!(!file.contains(['<', '>', '?', '"', '|', '*', '\t']), "{file}");
        }
    }

    /// A damaged weights file is said once per version of the file, not on every listing (x3-server-8).
    #[test]
    fn a_damaged_weights_file_is_warned_about_once() {
        let dir = folo_test_support::TestDir::new("models-skipped");
        let models = Models::new(dir.join("home"));
        std::fs::create_dir_all(models.models_dir()).unwrap();
        let path = models.models_dir().join("broken.folo");
        std::fs::write(&path, b"not weights").unwrap();
        let err = models.meta(&path).unwrap_err();
        assert!(models.warn_skipped(&path, &err));
        assert!(!models.warn_skipped(&path, &err), "the same file again");
        assert!(models.scan().is_empty());
        assert!(!models.warn_skipped(&path, &err), "a listing does not say it again");
        // Another version of the file (another size) is said again.
        std::fs::write(&path, b"still not weights").unwrap();
        assert!(models.warn_skipped(&path, &err));
        assert!(!models.warn_skipped(dir.join("gone.folo").as_path(), &err), "a missing file is not said");
    }
}
