//! `.folo` header metadata reading ([`read_meta`]), [`FileMeta`], [`ModelRef`], FLOPs estimates and model ID helpers.

use std::path::{Path, PathBuf};

use axum::http::StatusCode;
use folo_api::{Arch, Head, Model, ModelMetrics, ModelPrecision, ModelSource, TrainedWith};
use folo_nn::io::Metadata;
use folo_nn::io::Namespace;
use folo_nn::io::safetensors::Container;
use serde::Deserialize;

use crate::engine::nn_error;
use crate::error::{Error, Result};
use crate::runs::DatasetRef;
use crate::util::{fnv1a, rfc3339};

/// What a `.folo` file says about itself (the parts of `Model` that come from the file).
#[derive(Clone, Debug)]
pub struct FileMeta {
    /// `folo.arch` mapped to the API scale enum (`folo-n..x`).
    pub arch: Arch,
    /// Exact `folo.arch` string from the header (e.g. `"folo-n"` or `"folo2-n"`).
    pub arch_name: String,
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
pub(super) fn precision(dtype: Option<&str>) -> Option<ModelPrecision> {
    match dtype? {
        "F32" => Some(ModelPrecision::F32),
        "F16" => Some(ModelPrecision::F16),
        _ => None,
    }
}

/// [`Model::fused`] from a file's `folo.fused`: writers leave the key out of unfused files (docs/FORMAT.md), so a
/// missing key is `false`; a value other than `"true"` / `"false"` is unknown.
pub(super) fn fused(fused: Option<&str>) -> Option<bool> {
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
pub(super) fn trained_with(train: Option<&String>) -> Option<TrainedWith> {
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
        arch_name: m.arch.clone(),
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
    /// Exact architecture name from the weights header (e.g. `"folo-n"` or `"folo2-n"`).
    pub arch_name: String,
    /// The weights file.
    pub path: PathBuf,
    /// The dataset it was trained on (run models).
    pub dataset: Option<DatasetRef>,
}

impl ModelRef {
    /// A model and its file; `Model.path` is the file (`/` separators).
    pub fn new(mut model: Model, path: PathBuf, dataset: Option<DatasetRef>) -> ModelRef {
        model.path = Some(folo_api::json::path(&path));
        let arch_name = model.arch.as_str().to_string();
        ModelRef { model, arch_name, path, dataset }
    }

    /// Sets the exact architecture name from the weights header (e.g. `"folo2-n"`).
    pub fn with_arch_name(mut self, arch_name: String) -> ModelRef {
        self.arch_name = arch_name;
        self
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
