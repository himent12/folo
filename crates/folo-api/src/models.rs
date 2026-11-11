//! Models (docs/API.md, *Models*): trained weights, their validation results and exports.

use serde::{Deserialize, Serialize};

#[cfg(doc)]
use crate::weights;
use crate::{ClassMetrics, Head, Split};

/// `GET /api/models`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelList {
    /// Models, newest first.
    pub models: Vec<Model>,
}

/// A trained model (`.folo` weights).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Model {
    /// Opaque id, e.g. `m_31aa`.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Architecture.
    pub arch: Arch,
    /// Training / inference image size.
    pub imgsz: u32,
    /// Class names.
    pub classes: Vec<String>,
    /// Parameter count.
    pub params: u64,
    /// GFLOPs per image at `imgsz`.
    pub flops_g: f64,
    /// File size in bytes.
    pub bytes: u64,
    /// RFC 3339 creation time.
    pub created: String,
    /// The run that produced it, if any.
    pub run_id: Option<String>,
    /// Validation metrics of these weights, if known.
    pub metrics: Option<ModelMetrics>,
    /// Detection head (docs/MODEL.md §6.6), from the file's `folo.config`: `o2o` models predict with the NMS-free
    /// one-to-one branch. Additive: older servers leave it out, which reads as `o2m`.
    #[serde(default)]
    pub head: Head,
    /// How the weights were trained, from the file's `folo.train` (training checkpoints have it); `null` for files
    /// without it (exports, imported weights). Additive: missing reads as `null`.
    #[serde(default)]
    pub trained_with: Option<TrainedWith>,
    /// Stored weight dtype, from the file's `folo.dtype` (docs/FORMAT.md): `f16` for half-precision exports,
    /// `f32` otherwise. Left out when unknown: older servers, files without `folo.dtype`, and dtypes this API has no
    /// name for (a client reads any value it does not know as unknown too).
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "lenient_precision")]
    pub precision: Option<ModelPrecision>,
    /// Whether Conv+BN are fused (`folo.fused`): an inference-only tensor set, as `folo export` writes; run
    /// checkpoints are unfused. Left out when unknown (older servers, a `folo.fused` value other than
    /// `"true"` / `"false"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused: Option<bool>,
    /// Where the model comes from: a run, a file under the server's models folder, or published weights
    /// downloaded with `folo weights pull` (their id is the published name, e.g. `folo-n-coco`). Left out by older
    /// servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
    /// What published weights are (the line `folo weights list` prints); absent for other models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    /// The weights file on the server's machine (`/` separators): what `folo train --weights`, `val --model` and
    /// `export --model` take there. Left out by the mock server and older servers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// [`Model::source`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    /// The `best.folo` / `last.folo` of a run.
    Run,
    /// A `.folo` file under `<data dir>/models`.
    File,
    /// Published weights from the local `folo weights` cache ([`weights::cache_dir`]).
    Published,
}

/// [`Model::precision`]: the floating dtype a model's weights are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum ModelPrecision {
    /// 32-bit floats (training checkpoints, `folo export` without half precision).
    F32,
    /// 16-bit floats (`folo-f16` exports).
    F16,
}

impl ModelPrecision {
    /// `"f32"` or `"f16"`.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelPrecision::F32 => "f32",
            ModelPrecision::F16 => "f16",
        }
    }

    /// The API spelling (`"f32"`, `"f16"`); anything else is `None`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(ModelPrecision::F32),
            "f16" => Some(ModelPrecision::F16),
            _ => None,
        }
    }
}

/// Deserializes [`Model::precision`] leniently: `null` and values this version does not know (a newer server's)
/// read as `None` instead of failing the whole object.
fn lenient_precision<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<ModelPrecision>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.as_ref().and_then(serde_json::Value::as_str).and_then(ModelPrecision::parse))
}

/// [`Model::trained_with`]: the few training settings a front end shows next to a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TrainedWith {
    /// Mixed-precision training (`TrainConfig.amp`, docs/MODEL.md §8.8).
    pub amp: bool,
    /// Training image size.
    pub imgsz: u32,
    /// Planned epochs (`TrainConfig.epochs`; a stopped run trained fewer).
    pub epochs: u32,
}

/// Detection metrics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelMetrics {
    /// mAP at IoU 0.5.
    pub map50: f32,
    /// mAP averaged over IoU 0.5:0.95.
    pub map50_95: f32,
    /// Precision at `best_f1_conf`.
    pub precision: f32,
    /// Recall at `best_f1_conf`.
    pub recall: f32,
    /// The operating point of `precision` and `recall`: the confidence threshold (on a 0.001 grid) that maximises
    /// the class-mean F1 at IoU 0.5 (docs/METRICS.md). `null` when unknown (runs, `metrics.jsonl` and `.folo`
    /// files from before the field). Additive: missing reads as `null`.
    #[serde(default)]
    pub best_f1_conf: Option<f32>,
}

/// Detector architecture (scale).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum Arch {
    /// Nano.
    #[default]
    #[serde(rename = "folo-n")]
    N,
    /// Small.
    #[serde(rename = "folo-s")]
    S,
    /// Medium.
    #[serde(rename = "folo-m")]
    M,
    /// Large.
    #[serde(rename = "folo-l")]
    L,
    /// Extra large.
    #[serde(rename = "folo-x")]
    X,
}

impl Arch {
    /// All architectures, smallest first.
    pub const ALL: [Arch; 5] = [Arch::N, Arch::S, Arch::M, Arch::L, Arch::X];

    /// Wire name, e.g. `folo-n`.
    pub fn as_str(self) -> &'static str {
        match self {
            Arch::N => "folo-n",
            Arch::S => "folo-s",
            Arch::M => "folo-m",
            Arch::L => "folo-l",
            Arch::X => "folo-x",
        }
    }

    /// Parses a wire name.
    pub fn parse(s: &str) -> Option<Arch> {
        Arch::ALL.into_iter().find(|a| a.as_str() == s)
    }
}

/// `GET /api/models/{id}/eval`: validation results of a model (PR curves, confusion matrix, per-class metrics).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ModelEval {
    /// One precision-recall curve per class, in class order.
    pub pr: Vec<PrCurve>,
    /// `(classes + 1) x (classes + 1)` counts: rows are the true class, columns the predicted class; the last row and
    /// column are background (`confusion[c][n]` = missed objects of class `c`, `confusion[n][c]` = false positives
    /// predicted as `c`, `confusion[n][n]` = 0).
    pub confusion: Vec<Vec<u32>>,
    /// Per-class metrics (`map50` equals the curve's `ap50`).
    pub per_class: Vec<ClassMetrics>,
    /// Validation images evaluated.
    pub images: u32,
    /// The confidence at which the `per_class` precision and recall are taken ([`ModelMetrics::best_f1_conf`]).
    /// Additive: missing (older servers) reads as `null`.
    #[serde(default)]
    pub best_f1_conf: Option<f32>,
    /// The dataset evaluated: its registered id (for a run model whose dataset is no longer registered, the id it
    /// had when the run started). Additive: missing (older servers) reads as `null`.
    #[serde(default)]
    pub dataset_id: Option<String>,
    /// The evaluated dataset's name. Additive: missing reads as `null`.
    #[serde(default)]
    pub dataset_name: Option<String>,
    /// The split actually evaluated: `val`, or `train` when the dataset has no validation images (then
    /// `split_fallback` is set). Additive: missing (older servers) reads as `null`, which means unknown, not `val`.
    #[serde(default)]
    pub split: Option<Split>,
    /// Set when `split` is not the split `/eval` evaluates (`val`): which split was wanted and why it was not
    /// used. `null` when the validation split was evaluated. Additive: missing reads as `null`.
    #[serde(default)]
    pub split_fallback: Option<SplitFallback>,
    /// The confusion matrix counts predictions with `score > confusion_conf` (docs/METRICS.md §7: 0.25).
    /// Additive: missing reads as `null`.
    #[serde(default)]
    pub confusion_conf: Option<f32>,
    /// The confusion matrix matches a prediction to a ground truth when their IoU is `> confusion_iou`
    /// (docs/METRICS.md §7: 0.45). Additive: missing reads as `null`.
    #[serde(default)]
    pub confusion_iou: Option<f32>,
}

/// [`ModelEval::split_fallback`]: the evaluation used another split than the one it is defined on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SplitFallback {
    /// The split that was wanted (`/eval`: `val`).
    pub requested: Split,
    /// Why it was not used, a stable code: [`SplitFallback::EMPTY_SPLIT`] (it has no images). Clients treat a code
    /// they do not know like this one.
    pub reason: String,
    /// One sentence for people, naming the dataset and both splits.
    pub message: String,
}

impl SplitFallback {
    /// [`SplitFallback::reason`] when the requested split has no images.
    pub const EMPTY_SPLIT: &'static str = "empty_split";

    /// `requested` of dataset `dataset` (its name) is empty, so `used` (with `images` images) was evaluated.
    pub fn empty_split(requested: Split, used: Split, dataset: &str, images: u64) -> SplitFallback {
        let (r, u) = (requested.as_str(), used.as_str());
        let held_out = if used == Split::Train { ": training-set numbers, not held-out ones" } else { "" };
        SplitFallback {
            requested,
            reason: Self::EMPTY_SPLIT.into(),
            message: format!(
                "dataset '{dataset}' has no {r} images, so this is its {u} split ({images} images){held_out}"
            ),
        }
    }
}

/// A precision-recall curve at IoU 0.5, sampled at 101 recall points (`0.00, 0.01, ..., 1.00`). Precision is the
/// COCO-interpolated envelope, so it never increases with recall.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PrCurve {
    /// Class index.
    pub class: u32,
    /// Recall points.
    pub recall: Vec<f32>,
    /// Precision at each recall point.
    pub precision: Vec<f32>,
    /// AP at IoU 0.5: the mean of `precision`.
    pub ap50: f32,
}

/// The ONNX switches of `POST /api/models/{id}/export` (docs/CLI.md, ONNX export), all off by default. Only
/// valid with `format: onnx`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OnnxSwitches {
    /// Decode inside the graph: one `detections: [B, A, 4 + nc]` output instead of the raw head.
    #[serde(default)]
    pub decode: bool,
    /// Symbolic batch, height and width.
    #[serde(default)]
    pub dynamic: bool,
    /// F16 weights, input, activations and outputs (the file name gets `-f16`).
    #[serde(default)]
    pub half: bool,
}

impl OnnxSwitches {
    /// Whether any switch is on.
    pub fn any(&self) -> bool {
        self.decode || self.dynamic || self.half
    }

    /// What an ONNX file's name gets for these switches, before `.onnx`: `-decoded` with `decode`, `-f16` with
    /// `half`, both `-decoded-f16` (as `folo export` names its files). The export path and the download name share it,
    /// so two different graphs are never offered under one name.
    pub fn file_suffix(&self) -> &'static str {
        match (self.decode, self.half) {
            (true, true) => "-decoded-f16",
            (false, true) => "-f16",
            (true, false) => "-decoded",
            (false, false) => "",
        }
    }
}

/// The full body of `POST /api/models/{id}/export`: [`ExportRequest`]'s `format` plus the [`OnnxSwitches`]
/// (every one optional, so an [`ExportRequest`] body is one of these).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportSpec {
    /// Target format.
    pub format: ExportFormat,
    /// ONNX switches.
    #[serde(flatten)]
    pub onnx: OnnxSwitches,
}

impl From<ExportRequest> for ExportSpec {
    fn from(r: ExportRequest) -> Self {
        ExportSpec { format: r.format, onnx: OnnxSwitches::default() }
    }
}

/// `POST /api/models/{id}/export` with the format only ([`ExportSpec`] adds the ONNX switches).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportRequest {
    /// Target format.
    pub format: ExportFormat,
}

/// Export format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum ExportFormat {
    /// `.folo` with f16 weights.
    #[serde(rename = "folo-f16")]
    FoloF16,
    /// ONNX.
    #[serde(rename = "onnx")]
    Onnx,
}

/// Result of an export.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ExportResult {
    /// Where the file was written.
    pub path: String,
    /// Its size.
    pub bytes: u64,
}
