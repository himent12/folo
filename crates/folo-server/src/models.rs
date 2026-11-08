//! Models of the real store: the weights of finished runs and the `.folo` files under `<data dir>/models`, their
//! metadata (read once per file version), validation results (`/eval`, computed with folo-vision's [`validate`]
//! and cached in memory and under `cache/eval/`) and exports.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::http::StatusCode;
use folo_api::{Arch, ExportFormat, ExportResult, Model, ModelEval, ModelMetrics, PrCurve};
use folo_core::Device;
use folo_data::{Dataset as _, Split};
use folo_nn::io::{FloatDType, ModelFile, Namespace};
use folo_vision::val::{ValConfig, val_metrics, validate};
use serde::{Deserialize, Serialize};

use crate::engine::{self, FileKey, engine_error, nn_error};
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::runs::{DatasetRef, data_format};
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
    let file = ModelFile::open(path).map_err(nn_error)?;
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

/// Builds the API record of a weights file.
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
        metrics: metrics.or(meta.metrics),
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

/// Metadata and eval caches.
pub struct Models {
    data_dir: PathBuf,
    meta: Mutex<HashMap<PathBuf, (FileKey, Arc<FileMeta>)>>,
    evals: Mutex<HashMap<String, (String, ModelEval)>>,
}

impl Models {
    /// Caches rooted at `data_dir`.
    pub fn new(data_dir: PathBuf) -> Models {
        Models { data_dir, meta: Mutex::default(), evals: Mutex::default() }
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
                    let model = model_of(file_model_id(&path), name, &meta, None, None);
                    out.push(ModelRef { model, path, dataset: None });
                }
                Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
            }
        }
        out
    }

    /// `/eval` of `m` on the validation split (training split when there is none) of `dataset`, whose index has
    /// fingerprint `ds_key`. Cached per (weights file version, dataset version).
    pub fn eval(
        &self,
        m: &ModelRef,
        dataset: &DatasetRef,
        ds_key: &str,
        device: Device,
    ) -> Result<ModelEval> {
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

    /// Exports `m` to `<data dir>/exports/<name>-f16.folo` (`-2`, `-3`, ... when taken).
    pub fn export(&self, m: &ModelRef, format: ExportFormat) -> Result<ExportResult> {
        if format == ExportFormat::Onnx {
            return Err(
                Error::not_implemented("ONNX export").hint("export to folo-f16 for FOLO's own runtime")
            );
        }
        let file = ModelFile::open(&m.path).map_err(nn_error)?;
        let dir = self.data_dir.join("exports");
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        let base = m.model.name.replace(['/', '\\', ':'], "-");
        // create_new claims the name, so two exports never share a file; the export then replaces it atomically.
        let mut n = 1;
        let out = loop {
            let name = if n == 1 { format!("{base}-f16.folo") } else { format!("{base}-f16-{n}.folo") };
            let p = dir.join(name);
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
                Ok(_) => break p,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => return Err(Error::internal(format!("creating {}: {e}", p.display()))),
            }
        };
        if let Err(e) = file.export(&out, FloatDType::F16) {
            let _ = std::fs::remove_file(&out);
            return Err(nn_error(e));
        }
        let bytes = std::fs::metadata(&out).map(|x| x.len()).unwrap_or(0);
        Ok(ExportResult { path: out.to_string_lossy().into_owned(), bytes })
    }
}

/// Validates the weights at `path` on `dataset` and shapes the report as `/eval`.
fn evaluate(path: &Path, dataset: &DatasetRef, device: Device) -> Result<ModelEval> {
    let model = engine::open_model(path, device)?;
    let opts = folo_data::OpenOptions { format: data_format(dataset.format), verify_images: true };
    let ds = folo_data::open_with(Path::new(&dataset.path), &opts)
        .map_err(|e| crate::index::data_error(e, &dataset.path))?;
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
