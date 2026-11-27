//! Inference, evaluation, quantization, and export arguments (`val`, `predict`, `export`, `quantize`).

use std::path::PathBuf;

use clap::{Args, ValueEnum};

/// Detection head (docs/MODEL.md §6.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum HeadArg {
    /// One-to-many head with NMS.
    O2m,
    /// One-to-one (NMS-free) head.
    #[value(alias = "end2end")]
    O2o,
}

impl From<HeadArg> for folo_vision::model::HeadMode {
    fn from(h: HeadArg) -> Self {
        match h {
            HeadArg::O2m => folo_vision::model::HeadMode::O2m,
            HeadArg::O2o => folo_vision::model::HeadMode::O2o,
        }
    }
}

/// Box fusion after NMS (docs/MODEL.md §9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FusionArg {
    /// The kept boxes as NMS keeps them.
    None,
    /// Score-weighted box voting.
    Vote,
    /// DFL evidence fusion: score times per-side DFL certainty.
    Dfl,
    /// Student-t Exclusive Box Fusion: exclusive NMS ownership + IoU^4 affinity + 1-step Student-t IRLS.
    Sebf,
}

impl From<FusionArg> for folo_vision::fusion::BoxFusion {
    fn from(f: FusionArg) -> Self {
        match f {
            FusionArg::None => folo_vision::fusion::BoxFusion::None,
            FusionArg::Vote => folo_vision::fusion::BoxFusion::Vote,
            FusionArg::Dfl => folo_vision::fusion::BoxFusion::Dfl,
            FusionArg::Sebf => folo_vision::fusion::BoxFusion::Sebf,
        }
    }
}

/// How the boxes NMS keeps are scored (docs/MODEL.md §9.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ScoreArg {
    /// The class score.
    None,
    /// The class score scaled by the score mass of the box's cluster.
    Consensus,
    /// Exclusive logit-space consensus: each candidate votes only for the kept box that suppressed it.
    Elc,
    /// Scale-free exclusive logit-space consensus combined with fused DFL excess-variance precision.
    #[value(name = "dfl-consensus")]
    DflConsensus,
    /// Power-weighted exclusive logit consensus with DFL Shannon entropy, cross-stride agreement and greedy
    /// log-survival soft repulsion on output boxes.
    #[value(name = "dfl-repulse")]
    DflRepulse,
    /// Confidence-gated DFL repulsion: `dfl-repulse` with `p_m^{0.25}` suppressor gating (`γ = 3.0`), best
    /// paired with `--fusion sebf --iou 0.78 --max-det 1000`.
    #[value(name = "dfl-cgr")]
    DflCgr,
}

impl From<ScoreArg> for folo_vision::fusion::ScoreRule {
    fn from(s: ScoreArg) -> Self {
        match s {
            ScoreArg::None => folo_vision::fusion::ScoreRule::None,
            ScoreArg::Consensus => folo_vision::fusion::ScoreRule::Consensus,
            ScoreArg::Elc => folo_vision::fusion::ScoreRule::Elc,
            ScoreArg::DflConsensus => folo_vision::fusion::ScoreRule::DflConsensus,
            ScoreArg::DflRepulse => folo_vision::fusion::ScoreRule::DflRepulse,
            ScoreArg::DflCgr => folo_vision::fusion::ScoreRule::DflCgr,
        }
    }
}

/// Dataset split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SplitArg {
    /// Training split.
    Train,
    /// Validation split.
    Val,
    /// Test split.
    Test,
}

impl From<SplitArg> for folo_data::Split {
    fn from(s: SplitArg) -> Self {
        match s {
            SplitArg::Train => folo_data::Split::Train,
            SplitArg::Val => folo_data::Split::Val,
            SplitArg::Test => folo_data::Split::Test,
        }
    }
}

/// Precision of `val` and `predict`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum PrecisionArg {
    /// f32 weights and activations.
    F32,
    /// f16 on a GPU with shader-f16 (--half).
    F16,
    /// int8 backbone and neck on the CPU (`folo quantize` first).
    Int8,
}

/// How `folo quantize` picks each range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum QuantMethodArg {
    /// The `--tail` and `1 - --tail` quantiles of the calibration values.
    Percentile,
    /// The minimum and maximum.
    Minmax,
    /// The range of least expected squared error (rounding inside, clipping outside).
    Mse,
}

/// `folo quantize`.
#[derive(Debug, Args)]
pub struct QuantizeArgs {
    /// Weights (.folo, or a published name) to calibrate.
    #[arg(long)]
    pub model: PathBuf,
    /// Dataset whose images calibrate the ranges: a folder (YOLO or COCO) or a data.yaml.
    #[arg(long)]
    pub data: String,
    /// Split the images come from.
    #[arg(long, value_enum, default_value_t = SplitArg::Train)]
    pub split: SplitArg,
    /// Calibration images [>= 1], drawn at random (by --seed) from the split; all of them when it has fewer.
    #[arg(long, default_value_t = 512)]
    pub images: usize,
    /// Seed of the draw.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// How each range is picked.
    #[arg(long, value_enum, default_value_t = QuantMethodArg::Percentile)]
    pub method: QuantMethodArg,
    /// With --method percentile: the mass cut from each tail in (0, 0.5).
    #[arg(long, default_value_t = 1e-4)]
    pub tail: f64,
    /// Square input size of the calibration images [32..=8192, multiple of 32] [default: the model's training size].
    #[arg(long)]
    pub imgsz: Option<u32>,
    /// Batch size [>= 1].
    #[arg(long, default_value_t = 8)]
    pub batch: u32,
    /// Data loader worker threads [default: min(cores, 16)].
    #[arg(long)]
    pub workers: Option<usize>,
    /// Output file: fused f32 weights with the calibration [default: `<name>-int8.folo` next to the input, with
    /// `-2`, `-3`, ... when taken].
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// `folo val`.
#[derive(Debug, Args)]
pub struct ValArgs {
    /// Weights (.folo, or a published name such as folo-n-coco) to evaluate (EMA weights when the file has them).
    #[arg(long)]
    pub model: PathBuf,
    /// Dataset: a folder (YOLO or COCO) or a data.yaml.
    #[arg(long)]
    pub data: String,
    /// Split to evaluate.
    #[arg(long, value_enum, default_value_t = SplitArg::Val)]
    pub split: SplitArg,
    /// Batch size [>= 1].
    #[arg(long, default_value_t = 32)]
    pub batch: u32,
    /// Square input size in pixels [>= 32, multiple of 32] [default: the model's training size].
    #[arg(long)]
    pub imgsz: Option<u32>,
    /// Confidence threshold [0..=1] (low for mAP).
    #[arg(long, default_value_t = 0.001)]
    pub conf: f32,
    /// NMS IoU threshold [0..=1].
    #[arg(long, default_value_t = 0.7)]
    pub iou: f32,
    /// Maximum detections per image [>= 1] (COCO's AP reads the best 100 of them).
    #[arg(long, default_value_t = 300)]
    pub max_det: u32,
    /// Refine each kept box from its NMS cluster (docs/MODEL.md §9.4): `vote` moves each side to the
    /// score-weighted mean of its cluster, `dfl` weighs each member's side by its score times the certainty of its
    /// DFL distribution, and `sebf` (Student-t Exclusive Box Fusion) fuses exclusively owned cluster members with
    /// `IoU⁴` affinity and 1-step Student-t (`ν = 3`) IRLS outlier rejection. Off by default.
    #[arg(long, value_enum, default_value_t = FusionArg::None)]
    pub fusion: FusionArg,
    /// With --fusion: a kept box's cluster is the candidates of its class whose IoU with it is at least this [0..=1].
    #[arg(long, default_value_t = folo_vision::fusion::FUSION_IOU)]
    pub fusion_iou: f32,
    /// Score of each kept box (docs/MODEL.md §9.4): `elc` updates log-odds from exclusive NMS suppressor mass,
    /// `dfl-consensus` adds scale-free fused DFL precision, `dfl-repulse` combines power-weighted exclusive
    /// consensus, DFL Shannon entropy, cross-stride agreement and greedy log-survival soft repulsion (best with
    /// `--fusion dfl --iou 0.75 --max-det 1000`), `dfl-cgr` gates greedy repulsion by suppressor confidence
    /// `p_m^{0.25}` (`γ = 3.0`, best with `--fusion sebf --iou 0.78 --max-det 1000`), and `consensus` scales
    /// probabilities by cluster mass (changes the score scale). Off by default.
    #[arg(long, value_enum, default_value_t = ScoreArg::None)]
    pub score: ScoreArg,
    /// Head to evaluate [default: the model's: o2o, without NMS, for a model trained with --head o2o]; `o2m`
    /// evaluates such a model's one-to-many branch with NMS.
    #[arg(long, value_enum)]
    pub head: Option<HeadArg>,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Half precision: f16 weights and activations (f32 accumulation) on a GPU with shader-f16; runs in f32 with a
    /// note anywhere else.
    #[arg(long)]
    pub half: bool,
    /// Precision: f32, f16 (the same as --half) or int8 (the backbone and neck on the CPU's int8 kernels, with the
    /// calibration `folo quantize` stored in the weights; CPU only) [default: f32].
    #[arg(long, value_enum, conflicts_with = "half")]
    pub precision: Option<PrecisionArg>,
    /// With --half, run the f16 convolutions and matrix products on the GPU's tensor cores (wgpu's experimental
    /// cooperative matrices; the same as FOLO_GPU_TENSOR_CORES=1). Falls back to the regular kernels, with a note,
    /// where they are not available.
    #[arg(long)]
    pub tensor_cores: bool,
    /// Data loader worker threads [default: min(cores, 16)].
    #[arg(long)]
    pub workers: Option<usize>,
    /// Also evaluate exactly as pycocotools' COCOeval does on a COCO dataset: original-image boxes, crowd regions,
    /// annotation areas, the small / medium / large buckets and AR@1/10/100 (adds a `coco` object).
    #[arg(long)]
    pub coco: bool,
    /// Write the detections as a COCO results file (the input of pycocotools' loadRes; COCO datasets only).
    #[arg(long, value_name = "FILE")]
    pub save_json: Option<PathBuf>,
    /// Compute per-image error diagnosis (TP, FP, FN, localization, class confusion, duplicates, background FP,
    /// per-image F1, mean TP IoU) and list the worst-performing failing images (adds a `diagnostics` object).
    #[arg(long)]
    pub diagnostics: bool,
    /// Number of worst failing images to report [>= 1] (implies `--diagnostics`; default 10 with `--diagnostics`).
    #[arg(long, value_name = "N")]
    pub worst: Option<usize>,
    /// With --diagnostics / --worst: minimum prediction confidence score included in error diagnosis [0..=1] [default: 0.25].
    #[arg(long, value_name = "SCORE")]
    pub diag_conf: Option<f32>,
    /// With --diagnostics / --worst: IoU threshold for true-positive matching in error diagnosis [0..=1] [default: 0.5].
    #[arg(long, value_name = "IOU")]
    pub diag_tp_iou: Option<f32>,
    /// With --diagnostics / --worst: background IoU threshold in error diagnosis [0..=1] (`<= --diag-tp-iou`) [default: 0.1].
    #[arg(long, value_name = "IOU")]
    pub diag_bg_iou: Option<f32>,
}

/// `folo predict`.
#[derive(Debug, Args)]
pub struct PredictArgs {
    /// Weights (.folo, or a published name such as folo-n-coco; EMA weights when the file has them).
    #[arg(long)]
    pub model: PathBuf,
    /// Images or folders of images (searched recursively).
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,
    /// Confidence threshold [0..=1].
    #[arg(long, default_value_t = 0.25)]
    pub conf: f32,
    /// NMS IoU threshold [0..=1].
    #[arg(long, default_value_t = 0.7)]
    pub iou: f32,
    /// Maximum detections per image [>= 1].
    #[arg(long, default_value_t = 300)]
    pub max_det: u32,
    /// Refine each kept box from its NMS cluster (docs/MODEL.md §9.4): `vote` moves each side to the
    /// score-weighted mean of its cluster, `dfl` weighs each member's side by its score times the certainty of its
    /// DFL distribution, and `sebf` (Student-t Exclusive Box Fusion) fuses exclusively owned cluster members with
    /// `IoU⁴` affinity and 1-step Student-t (`ν = 3`) IRLS outlier rejection. Off by default.
    #[arg(long, value_enum, default_value_t = FusionArg::None)]
    pub fusion: FusionArg,
    /// With --fusion: a kept box's cluster is the candidates of its class whose IoU with it is at least this [0..=1].
    #[arg(long, default_value_t = folo_vision::fusion::FUSION_IOU)]
    pub fusion_iou: f32,
    /// Score of each kept box (docs/MODEL.md §9.4): `elc`, `dfl-consensus`, `dfl-repulse` and `dfl-cgr` update
    /// log-odds from exclusive suppressor mass (plus DFL precision / entropy and soft repulsion) while preserving
    /// the `[0, 1]` scale; when `--conf > 0.10`, supporting anchors are gathered down to 0.10 before filtering
    /// final scores by `--conf` (for `--conf <= 0.10`, calibrated scores below `--conf` are kept in full).
    /// `consensus` scales probabilities by cluster mass (changes the score scale). Off by default.
    #[arg(long, value_enum, default_value_t = ScoreArg::None)]
    pub score: ScoreArg,
    /// Head to predict with [default: the model's: o2o, without NMS, for a model trained with --head o2o]; `o2m`
    /// uses such a model's one-to-many branch with NMS.
    #[arg(long, value_enum)]
    pub head: Option<HeadArg>,
    /// Letterbox size in pixels [32..=8192, multiple of 32] [default: the model's training size].
    #[arg(long)]
    pub imgsz: Option<u32>,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Half precision: f16 weights and activations (f32 accumulation) on a GPU with shader-f16; runs in f32 with a
    /// note anywhere else.
    #[arg(long)]
    pub half: bool,
    /// Precision: f32, f16 (the same as --half) or int8 (the backbone and neck on the CPU's int8 kernels, with the
    /// calibration `folo quantize` stored in the weights; CPU only) [default: f32].
    #[arg(long, value_enum, conflicts_with = "half")]
    pub precision: Option<PrecisionArg>,
    /// With --half, run the f16 convolutions and matrix products on the GPU's tensor cores (wgpu's experimental
    /// cooperative matrices; the same as FOLO_GPU_TENSOR_CORES=1). Falls back to the regular kernels, with a note,
    /// where they are not available.
    #[arg(long)]
    pub tensor_cores: bool,
    /// SAHI tiled inference: square tile size in pixels [>= 1] (clamped to the image size). Slices large images
    /// into overlapping windows and merges detections across tiles. Mutually exclusive with --tta.
    #[arg(long, value_name = "PX", conflicts_with_all = ["tta", "tta_scales", "no_tta_hflip"])]
    pub tile: Option<u32>,
    /// With --tile: overlap ratio between adjacent tiles in [0, 1) [default: 0.2].
    #[arg(
        long,
        value_name = "RATIO",
        requires = "tile",
        conflicts_with_all = ["tta", "tta_scales", "no_tta_hflip"]
    )]
    pub tile_overlap: Option<f32>,
    /// With --tile: skip the full-image pass alongside the cropped tiles (tiles only).
    #[arg(long, requires = "tile", conflicts_with_all = ["tta", "tta_scales", "no_tta_hflip"])]
    pub no_tile_full_image: bool,
    /// With --tile: maximum cropped tiles predicted per batch [>= 1] [default: 8].
    #[arg(
        long,
        value_name = "N",
        requires = "tile",
        conflicts_with_all = ["tta", "tta_scales", "no_tta_hflip"]
    )]
    pub tile_batch: Option<usize>,
    /// With --tile: cross-tile overlap metric (`ios` or `iou`) [default: ios].
    #[arg(
        long,
        value_enum,
        requires = "tile",
        conflicts_with_all = ["tta", "tta_scales", "no_tta_hflip"]
    )]
    pub tile_merge: Option<TileMergeArg>,
    /// Test-Time Augmentation: multi-scale + horizontal-flip ensembling (`0.83,1.0,1.17` with horizontal flip by
    /// default). Mutually exclusive with --tile.
    #[arg(
        long,
        conflicts_with_all = ["tile", "tile_overlap", "no_tile_full_image", "tile_batch", "tile_merge"]
    )]
    pub tta: bool,
    /// With --tta: comma-separated scale multipliers (> 0) applied to --imgsz [default: 0.83,1.0,1.17].
    #[arg(
        long,
        value_name = "SCALES",
        value_delimiter = ',',
        requires = "tta",
        conflicts_with_all = ["tile", "tile_overlap", "no_tile_full_image", "tile_batch", "tile_merge"]
    )]
    pub tta_scales: Option<Vec<f32>>,
    /// With --tta: disable the horizontally flipped view at each scale.
    #[arg(
        long,
        requires = "tta",
        conflicts_with_all = ["tile", "tile_overlap", "no_tile_full_image", "tile_batch", "tile_merge"]
    )]
    pub no_tta_hflip: bool,
    /// Enable ByteTrack multi-object tracking across ordered image / directory inputs; each emitted box includes
    /// persistent `track_id`, `age`, and `hits`.
    #[arg(long)]
    pub track: bool,
    /// With --track: stage-1 high-confidence detection threshold [0..=1] [default: 0.5].
    #[arg(long, value_name = "SCORE", requires = "track")]
    pub track_high_thresh: Option<f32>,
    /// With --track: stage-2 low-confidence occlusion recovery threshold [0..=1] (`<= --track-high-thresh`)
    /// [default: min(0.1, --track-high-thresh)].
    #[arg(long, value_name = "SCORE", requires = "track")]
    pub track_low_thresh: Option<f32>,
    /// With --track: minimum score for an unmatched high-confidence detection to spawn a new track [0..=1]
    /// (`>= --track-high-thresh`) [default: max(0.6, --track-high-thresh)].
    #[arg(long, value_name = "SCORE", requires = "track")]
    pub track_new_thresh: Option<f32>,
    /// With --track: stage-1 association minimum IoU [0..=1] [default: 0.2].
    #[arg(long, value_name = "IOU", requires = "track")]
    pub track_match_iou: Option<f32>,
    /// With --track: consecutive missed frames a lost track coasts before retiring [default: 30].
    #[arg(long, value_name = "FRAMES", requires = "track")]
    pub track_buffer: Option<u32>,
    /// With --track: consecutive matched detections required before confirming a new track [>= 1] [default: 1].
    #[arg(long, value_name = "N", requires = "track")]
    pub track_min_hits: Option<u32>,
    /// With --track: allow tracks to match detections across different classes (default is class-aware).
    #[arg(long, requires = "track")]
    pub track_class_agnostic: bool,
    /// Write annotated images (boxes coloured by class, with class id and score) to this folder.
    #[arg(long)]
    pub save: Option<PathBuf>,
}

/// Cross-tile overlap metric for SAHI tiled inference (`folo predict --tile`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum TileMergeArg {
    /// Intersection-over-Smaller: suppresses both cross-tile duplicates and truncated seam fragments.
    Ios,
    /// Standard Intersection-over-Union.
    Iou,
}

impl From<TileMergeArg> for folo_vision::TileMergeMetric {
    fn from(m: TileMergeArg) -> Self {
        match m {
            TileMergeArg::Ios => folo_vision::TileMergeMetric::Ios,
            TileMergeArg::Iou => folo_vision::TileMergeMetric::Iou,
        }
    }
}

/// Export format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExportFormatArg {
    /// Fused `.folo` with f16 weights.
    FoloF16,
    /// Fused `.folo` with f32 weights.
    Folo,
    /// Unfused `.folo` weights file in f32 (the EMA weights when present, no optimizer state): for fine-tuning
    /// with `folo train --weights` and for publishing.
    Weights,
    /// ONNX (opset 17) graph of the fused model (v1 `folo-*` and v2-Iso `folo2-*`), for onnxruntime, TensorRT,
    /// OpenVINO, ...
    Onnx,
}

/// `folo export`.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// Weights (.folo, or a published name) to export (supports v1 `folo-*` and v2-Iso `folo2-*` for ONNX).
    #[arg(long)]
    pub model: PathBuf,
    /// Target format.
    #[arg(long, value_enum, default_value_t = ExportFormatArg::FoloF16)]
    pub format: ExportFormatArg,
    /// Output file [default: next to the input, `<name>-f16.folo` / `<name>-fused.folo` / `<name>-weights.folo` /
    /// `<name>.onnx` / `<name>-f16.onnx` / `<name>-decoded.onnx` / `<name>-decoded-f16.onnx`, with `-2`, `-3`, ...
    /// when taken].
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Square input size baked into the graph [32..=8192, multiple of 32] (ONNX only; `.folo` files are
    /// size-independent) [default: the model's training imgsz].
    #[arg(long)]
    pub imgsz: Option<u32>,
    /// Fixed batch size of the graph [>= 1] (ONNX only) [default: 1].
    #[arg(long, conflicts_with_all = ["dynamic", "dynamic_batch"])]
    pub batch: Option<usize>,
    /// Symbolic batch, height and width (ONNX only): any batch, any H and W that are multiples of 32 (shorthand
    /// for `--dynamic-batch --dynamic-hw`).
    #[arg(long, conflicts_with = "batch")]
    pub dynamic: bool,
    /// Symbolic batch dimension only (ONNX only): any batch size with fixed height/width and constant anchors
    /// unless `--dynamic-hw` is also set.
    #[arg(long, conflicts_with = "batch")]
    pub dynamic_batch: bool,
    /// Symbolic height and width only (ONNX only): any H and W that are positive multiples of 32 (can be combined
    /// with `--batch` or `--dynamic-batch`).
    #[arg(long)]
    pub dynamic_hw: bool,
    /// F16 weights, input, activations and outputs (ONNX only; use `--format folo-f16` for `.folo`).
    #[arg(long)]
    pub half: bool,
    /// Decode in the graph (ONNX only): one `detections` output `[B, A, 4 + nc]` (x1 y1 x2 y2 in input pixels,
    /// then sigmoid class scores; NMS not included) instead of the raw `box_raw` / `cls_raw` head outputs. A model
    /// trained with --head o2o exports end-to-end detections instead: `[B, K, 6]` (x1 y1 x2 y2 score class, by
    /// score, no NMS needed), K = min(--max-det, anchors).
    #[arg(long)]
    pub decode: bool,
    /// With --decode on an o2o model (ONNX only): detections per image [>= 1].
    #[arg(long, default_value_t = 300)]
    pub max_det: usize,
    /// Head to export (ONNX only) [default: the model's]; `o2m` exports an o2o model's one-to-many branch
    /// (raw, or decoded for NMS) instead of its one-to-one branch.
    #[arg(long, value_enum)]
    pub head: Option<HeadArg>,
}
