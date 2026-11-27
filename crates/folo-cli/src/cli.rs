//! The clap command tree. `folo describe` serialises it, so help strings and defaults here are the documented
//! surface: keep them accurate. `folo label`'s arguments are in `cli/label.rs`, `folo bench`'s in
//! `cli/bench.rs`, `folo dataset`'s in `cli/dataset.rs`.

use std::path::PathBuf;

use clap::builder::NonEmptyStringValueParser;
use clap::{Args, Parser, Subcommand, ValueEnum};

mod bench;
pub mod bounds;
mod dataset;
mod label;
pub use bench::*;
pub use dataset::*;
pub use label::*;

/// FOLO: a from-scratch, very fast vision framework (YOLO-style detector, training, export, Studio).
#[derive(Debug, Parser)]
#[command(name = "folo", bin_name = "folo", version = crate::VERSION, propagate_version = true)]
pub struct Cli {
    /// Machine-readable output: stdout carries one JSON object per line, human text goes to stderr.
    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Command,
}

/// Top-level commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the HTTP API (docs/API.md) and serve FOLO Studio.
    Serve(ServeArgs),
    /// Print the CLI surface (commands, arguments, types, defaults, help) as JSON: all of it, or one command.
    Describe(DescribeArgs),
    /// Check the environment: version, OS, CPU, memory, GPU adapters and the one FOLO picks, web app build.
    Doctor(DoctorArgs),
    /// Train a detector.
    #[command(allow_negative_numbers = true)]
    Train(TrainArgs),
    /// Ask a running `folo train` to stop after its current batch (it keeps the last epoch's checkpoint; a request
    /// that comes after its last batch ends `completed`).
    ///
    /// A run is the CLI's to stop while a `folo train` process holds its folder, also a FOLO server's run resumed in
    /// place with `folo train --resume`; a server's run folder (`run.json`) that no `folo train` holds is stopped in
    /// Studio or with POST /api/runs/{id}/stop. A run of an older FOLO can be stopped once it is resumed with this
    /// version (it then gets its folo-train.json); a finished one needs no stop.
    Stop(StopArgs),
    /// Evaluate weights on a dataset split (mAP, precision, recall).
    Val(ValArgs),
    /// Run inference on images or folders.
    Predict(PredictArgs),
    /// Export weights to another format.
    Export(ExportArgs),
    /// Calibrate int8 inference on sample images and write the weights with the calibration (docs/MODEL.md §9.6).
    Quantize(QuantizeArgs),
    /// Vision benchmarks (docs/BENCH.md): frozen sets of labeled images with a plan of stress conditions, and the
    /// entries (FOLO models, predictions made elsewhere) that ran them with their scores.
    #[command(subcommand)]
    Bench(BenchCommand),
    /// Inspect and generate datasets.
    #[command(subcommand)]
    Dataset(DatasetCommand),
    /// Published pretrained weights: list them, download them into the local cache.
    #[command(subcommand)]
    Weights(WeightsCommand),
    /// Cloud compute (docs/CLOUD.md): providers, connections, offers and instances of the data dir, with or without
    /// a server running.
    #[command(subcommand)]
    Cloud(CloudCommand),
    /// Agent labeling (docs/LABELING.md): jobs of unlabeled images, boxes drawn by Claude Code, Codex or
    /// Antigravity, review, and the dataset made from them.
    #[command(subcommand)]
    Label(LabelCommand),
    /// Run a Model Context Protocol server over stdio so agents can drive FOLO.
    Mcp(McpArgs),
}

/// `folo serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Serve deterministic mock data instead of the real engine.
    #[arg(long)]
    pub mock: bool,
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to bind [0..=65535] (0 picks a free one).
    #[arg(long, default_value_t = 7878)]
    pub port: u16,
    /// Built web app folder to serve (with SPA fallback to index.html) [default: the Studio embedded in this binary
    /// if it was built with `embed-web`, else web/dist].
    #[arg(long)]
    pub web_dir: Option<PathBuf>,
    /// Where registered datasets, models and runs are kept [default: $FOLO_HOME or ~/.folo].
    #[arg(long, env = "FOLO_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
    /// Allow cross-origin requests from any website (default: only localhost / 127.0.0.1 / [::1] origins).
    #[arg(long)]
    pub cors_any: bool,
    /// Device of inference and model evaluation: auto (the GPU when one can hold the model, else the CPU), cpu or
    /// gpu:<n>. Training runs use their own `device` setting.
    #[arg(long, env = "FOLO_DEVICE", default_value = "auto")]
    pub device: String,
    /// Precision of inference and model evaluation: f32, or f16 (half precision on a GPU with shader-f16; f32
    /// with a warning where f16 is not available).
    #[arg(long, env = "FOLO_PRECISION", default_value = "f32")]
    pub precision: String,
    /// Run f16 inference and evaluation (`--precision f16`) on the GPU's tensor cores (wgpu's experimental
    /// cooperative matrices); the regular kernels are used where they are not available. Training runs with
    /// `"amp": true` need it (they fail with `not_supported` without tensor cores).
    #[arg(long, env = "FOLO_GPU_TENSOR_CORES", value_parser = clap::builder::BoolishValueParser::new())]
    pub tensor_cores: bool,
    /// Require `Authorization: Bearer <token>` on every /api request but GET /api/health (use it whenever the
    /// server listens beyond loopback; docs/CLOUD.md, Securing a server). Never logged or printed.
    #[arg(long, env = "FOLO_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    pub token: Option<String>,
    /// A Host name a server on loopback answers besides localhost, 127.x.y.z and [::1] (repeatable): a tunnel or
    /// reverse proxy that keeps its own Host. Other names are refused (forbidden_host) against DNS rebinding.
    #[arg(long = "allow-host", value_name = "NAME", value_parser = NonEmptyStringValueParser::new())]
    pub allow_host: Vec<String>,
    /// PEM certificate chain file to serve HTTPS natively (must be paired with --tls-key).
    #[arg(long = "tls-cert", env = "FOLO_TLS_CERT", value_name = "PATH", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,
    /// PEM private key file matching --tls-cert (PKCS#8, PKCS#1, or SEC1; must be paired with --tls-cert).
    #[arg(long = "tls-key", env = "FOLO_TLS_KEY", value_name = "PATH", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,
}

/// `folo describe`.
#[derive(Debug, Args)]
pub struct DescribeArgs {
    /// Only this command, e.g. `train` or `dataset check` (its arguments and subcommands, without `mcp_tools`).
    pub command: Vec<String>,
    /// Leave out `mcp_tools` (the `folo mcp` tool schemas), which most of the full output is.
    #[arg(long)]
    pub no_mcp_tools: bool,
}

/// `folo stop`.
#[derive(Debug, Args)]
pub struct StopArgs {
    /// The run folder `folo train` prints (`run_dir`), e.g. runs/shapes-n.
    pub run_dir: PathBuf,
}

/// `folo train --cache`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CacheArg {
    /// Decode every image every epoch.
    #[default]
    None,
    /// Decoded and resized images in memory.
    Ram,
}

impl From<CacheArg> for folo_data::CacheMode {
    fn from(c: CacheArg) -> Self {
        match c {
            CacheArg::None => folo_data::CacheMode::None,
            CacheArg::Ram => folo_data::CacheMode::Ram,
        }
    }
}

/// Which `folo train --json` records to print.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum JsonEvents {
    /// Every record: status, step (throttled to about 10 per second), epoch, log and done.
    #[default]
    All,
    /// No step records: status, epoch, log and done (about 1 KB per epoch).
    Epoch,
}

/// `folo doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Built web app folder to check [default: web/dist].
    #[arg(long)]
    pub web_dir: Option<PathBuf>,
    /// Ask for the tensor-core kernels, as `folo serve --tensor-cores` does, so `capabilities` shows what that
    /// server would report (the same as FOLO_GPU_TENSOR_CORES=1).
    #[arg(long, env = "FOLO_GPU_TENSOR_CORES", value_parser = clap::builder::BoolishValueParser::new())]
    pub tensor_cores: bool,
}

/// Optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OptimizerArg {
    /// MuSGD at lr0 0.01 for every run (`auto` rule 5, docs/MODEL.md §8.2).
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW.
    Adamw,
    /// MuSGD: SGD plus a Muon (orthogonalised momentum) update on conv weights (docs/MODEL.md §8.2).
    Musgd,
}

/// `folo train`. Mirrors `TrainConfig` in docs/API.md (plus the docs/MODEL.md §8.1 recipe keys).
#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("input").required(true).multiple(true).args(["data", "resume"])))]
pub struct TrainArgs {
    /// Dataset: a folder (YOLO or COCO) or a data.yaml.
    #[arg(long)]
    pub data: Option<String>,
    /// Format of --data: auto (detect from the files), or yolo / coco for a folder that has both.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub format: FormatArg,
    /// Architecture (folo-n, folo-s, folo-m, folo-l, folo-x), or .folo weights / a published name to fine-tune.
    #[arg(long, visible_alias = "arch", default_value = "folo-n")]
    pub model: String,
    /// Start from these weights (transfer learning; the architecture is theirs): a .folo file or a published name
    /// such as folo-n-coco (`folo weights list`; download with `folo weights pull`).
    #[arg(long, conflicts_with = "model")]
    pub weights: Option<String>,
    /// Distil from this frozen teacher (docs/MODEL.md §8.9): a .folo file or a published name such as folo-s-coco
    /// with the student's classes, reg_max and strides. Starts with distill.cls_gain 0.5, distill.dfl_gain 1.5,
    /// and distill.temperature 4; change them with --set distill.<key>=<value>.
    #[arg(long)]
    pub teacher: Option<String>,
    /// Run name [default: <dataset>-<arch suffix>]; an existing run folder gets a numeric suffix.
    #[arg(long)]
    pub name: Option<String>,
    /// Number of epochs [1..=10000].
    #[arg(long, default_value_t = 100)]
    pub epochs: u32,
    /// Batch size [1..=1024].
    #[arg(long, default_value_t = 16)]
    pub batch: u32,
    /// Square input size in pixels [64..=4096, multiple of 32].
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Data loader worker threads [default: min(cores, 16)].
    #[arg(long)]
    pub workers: Option<usize>,
    /// Keep the training images between epochs: `ram` decodes each once and keeps it resized in memory (faster
    /// epochs for large JPEGs; refused up front when it does not fit). The batches are the same either way.
    #[arg(long, value_enum, default_value_t = CacheArg::None)]
    pub cache: CacheArg,
    /// Optimizer.
    #[arg(long, value_enum, default_value_t = OptimizerArg::Auto)]
    pub optimizer: OptimizerArg,
    /// Initial learning rate in (0, 10) (ignored by --optimizer auto).
    #[arg(long, default_value_t = 0.01)]
    pub lr0: f32,
    /// Final learning rate as a fraction of lr0 in (0, 1].
    #[arg(long, default_value_t = 0.01)]
    pub lrf: f64,
    /// Cosine instead of linear learning-rate decay.
    #[arg(long)]
    pub cos_lr: bool,
    /// Mixed-precision training: f16 activations, convolutions and matrix products on the GPU's tensor cores, f32
    /// master weights, loss and BatchNorm statistics, dynamic loss scaling. Needs a GPU with tensor cores
    /// (cooperative matrices; asked for automatically); anything else is `not_supported`.
    #[arg(long)]
    pub amp: bool,
    /// Detection head: o2m (one-to-many, NMS at inference) or o2o (also the NMS-free one-to-one branch, trained
    /// with dual assignment: the model then predicts end to end without NMS; `end2end` is an alias)
    /// [default: the --weights file's head, else o2m].
    #[arg(long, value_enum)]
    pub head: Option<HeadArg>,
    /// Warmup length in epochs [>= 0].
    #[arg(long, default_value_t = 3.0)]
    pub warmup_epochs: f64,
    /// Stop after this many epochs without a mAP@0.5:0.95 gain.
    #[arg(long, default_value_t = 100)]
    pub patience: u32,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Mosaic augmentation probability [0..=1].
    #[arg(long, default_value_t = 1.0)]
    pub mosaic: f32,
    /// Mixup augmentation probability [0..=1].
    #[arg(long, default_value_t = 0.0)]
    pub mixup: f32,
    /// HSV jitter gains h,s,v [>= 0] (`--hsv 0.015,0.7,0.4` or `--hsv 0.015 0.7 0.4`).
    #[arg(long, value_delimiter = ',', num_args = 1..=3, default_values_t = [0.015, 0.7, 0.4])]
    pub hsv: Vec<f32>,
    /// Horizontal flip probability [0..=1].
    #[arg(long, default_value_t = 0.5)]
    pub fliplr: f32,
    /// Scale jitter (± fraction) in [0, 1).
    #[arg(long, default_value_t = 0.5)]
    pub scale: f32,
    /// Translation jitter (fraction of the image) [>= 0].
    #[arg(long, default_value_t = 0.1)]
    pub translate: f32,
    /// Disable mosaic for the last N epochs.
    #[arg(long, default_value_t = 10)]
    pub close_mosaic: u32,
    /// Set any other training key as KEY=VALUE (repeatable): the keys of a run's `config.json` `train` object
    /// (docs/MODEL.md §8.1), nested ones dotted, e.g. `--set momentum=0.9 --set weight_decay=0.001
    /// --set loss.tal.topk=13 --set augment.degrees=5`. Applied after the other flags; an unknown key or a value of
    /// the wrong type is `invalid_config`.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,
    /// Folder that run outputs (weights, metrics) go under.
    #[arg(long, default_value = "runs")]
    pub project: PathBuf,
    /// Continue an interrupted run from its checkpoint (runs/<name>/weights/last.folo). The run's stored settings
    /// are used: only --workers, --device, --data, --format, --cache, --project, --json-events, --teacher and
    /// --set distill.* may differ, other flags must match or be left out. A run that completed cannot be resumed;
    /// start a new run with --model <run>/weights/best.folo instead.
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Which records --json prints: all, or epoch (no per-batch step records).
    #[arg(long, value_enum, default_value_t = JsonEvents::All)]
    pub json_events: JsonEvents,
    /// The arguments given on the command line (set by `main`, so --resume can tell them from defaults).
    #[arg(skip)]
    pub explicit: Vec<String>,
}

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
        num_args = 1..,
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
    /// ONNX (opset 17) graph of the fused model, for onnxruntime, TensorRT, OpenVINO, ...
    Onnx,
}

/// `folo export`.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// Weights (.folo, or a published name) to export.
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
    #[arg(long, conflicts_with = "dynamic")]
    pub batch: Option<usize>,
    /// Symbolic batch, height and width (ONNX only): any batch, any H and W that are multiples of 32.
    #[arg(long, conflicts_with = "batch")]
    pub dynamic: bool,
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

/// `folo weights`.
#[derive(Debug, Subcommand)]
pub enum WeightsCommand {
    /// Published weights, whether each is downloaded, and its cache path ($FOLO_WEIGHTS_DIR or
    /// <FOLO_HOME or ~/.folo>/weights).
    List,
    /// Download published weights from the repository's GitHub release (over HTTPS; the GitHub CLI `gh`
    /// only as a fallback when that is refused) and verify their SHA-256 against the release's SHA256SUMS.
    Pull(WeightsPullArgs),
}

/// `folo weights pull`.
#[derive(Debug, Args)]
pub struct WeightsPullArgs {
    /// Published name (`folo weights list`), e.g. folo-n-coco.
    pub name: String,
    /// Download again even if the file is already in the cache.
    #[arg(long)]
    pub force: bool,
    /// GitHub repository (owner/name) whose release holds the weights.
    #[arg(long, default_value = crate::WEIGHTS_REPO)]
    pub repo: String,
}

/// `folo cloud`.
#[derive(Debug, Subcommand)]
pub enum CloudCommand {
    /// The providers FOLO can use and the values a connection to each takes.
    Providers(CloudDirArgs),
    /// Connect a provider: its values are checked with it before anything is stored.
    Connect(CloudConnectArgs),
    /// The connections, with what their last check found.
    Connections(CloudDirArgs),
    /// Check a connection's stored credentials again.
    Check(CloudIdArgs),
    /// Forget a connection (refused while it has live instances or active runs).
    Disconnect(CloudIdArgs),
    /// What a connection can run on now, cheapest first.
    Offers(CloudOffersArgs),
    /// The instances FOLO launched: alive first, then those that ended in the last 7 days.
    Instances(CloudDirArgs),
    /// Terminate an instance now (FOLO retries until the provider confirms); a run still using it fails.
    Terminate(CloudIdArgs),
}

/// The data dir of a `folo cloud` command.
#[derive(Debug, Args)]
pub struct CloudDirArgs {
    /// The data dir whose cloud state to use, as `folo serve` resolves it [default: $FOLO_HOME or ~/.folo].
    #[arg(long, env = "FOLO_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
}

/// `folo cloud check`, `disconnect` and `terminate`.
#[derive(Debug, Args)]
pub struct CloudIdArgs {
    /// The connection's id (`cc_…`), or for `terminate` the instance's (`ci_…`).
    pub id: String,
    #[command(flatten)]
    pub dir: CloudDirArgs,
}

/// `folo cloud offers`.
#[derive(Debug, Args)]
pub struct CloudOffersArgs {
    /// The connection's id (`cc_…`).
    pub connection: String,
    /// Ask the provider again instead of using the answer of the last minute.
    #[arg(long)]
    pub refresh: bool,
    #[command(flatten)]
    pub dir: CloudDirArgs,
}

/// `folo cloud connect`.
#[derive(Debug, Args)]
pub struct CloudConnectArgs {
    /// The provider's id (`folo cloud providers`), e.g. runpod or remote.
    pub provider: String,
    /// The connection's name [default: the provider's name, with 2, 3, ... when taken].
    #[arg(long, default_value = "", hide_default_value = true)]
    pub name: String,
    /// A value that is not secret, as KEY=VALUE (repeatable), e.g. --set url=https://gpu-box.example.com:7878.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    pub set: Vec<String>,
    /// A secret, from the environment (KEY=env:VAR) or from one line of stdin (KEY=-), repeatable, e.g. --secret
    /// api_key=env:RUNPOD_API_KEY. Never as a value on the command line, which other processes and the shell
    /// history can read.
    #[arg(long = "secret", value_name = "KEY=env:VAR|KEY=-")]
    pub secret: Vec<String>,
    #[command(flatten)]
    pub dir: CloudDirArgs,
}

/// `folo mcp`.
#[derive(Debug, Args)]
pub struct McpArgs {
    /// Where FOLO writes (datasets, runs, exports) and the base of relative input paths [default: $FOLO_HOME or
    /// ~/.folo].
    #[arg(long, env = "FOLO_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
    /// Serve deterministic mock data (like `folo serve --mock`), so the whole flow works without the engine.
    #[arg(long, conflicts_with = "server")]
    pub mock: bool,
    /// Only expose tools that do not change state; mutating tools are hidden and rejected.
    #[arg(long)]
    pub read_only: bool,
    /// Only read input paths inside --data-dir (canonical paths, symlinks and junctions resolved).
    #[arg(long)]
    pub confine: bool,
    /// Forward every tool to a running `folo serve` at this URL (e.g. http://127.0.0.1:7878) instead of running
    /// the engine in this process.
    #[arg(long, env = "FOLO_SERVER", value_name = "URL", value_parser = NonEmptyStringValueParser::new())]
    pub server: Option<String>,
    /// With --server: the token that `folo serve` was started with (`--token`), sent as `Authorization: Bearer`.
    #[arg(long, env = "FOLO_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    pub token: Option<String>,
    /// PEM CA certificate file to trust when connecting to an https:// `folo serve` (for self-signed or private-CA
    /// certificates).
    #[arg(long = "tls-ca", env = "FOLO_TLS_CA", value_name = "PATH", conflicts_with = "mock")]
    pub tls_ca: Option<PathBuf>,
    /// Embedded mode: device of `predict`, `predict_batch` and `val`: auto (the GPU when one can hold the model,
    /// else the CPU), cpu or gpu:<n>. Training runs use their own `device` setting.
    #[arg(long, env = "FOLO_DEVICE", default_value = "auto")]
    pub device: String,
    /// Embedded mode: ask for the GPU's tensor cores, as `folo serve --tensor-cores` does, so `train` with
    /// `"amp": true` (mixed precision) can run and f16 inference uses them (the same as FOLO_GPU_TENSOR_CORES=1).
    /// With --server, the `folo serve` decides.
    #[arg(long, env = "FOLO_GPU_TENSOR_CORES", value_parser = clap::builder::BoolishValueParser::new())]
    pub tensor_cores: bool,
}
