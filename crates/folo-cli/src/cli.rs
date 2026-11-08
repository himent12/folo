//! The clap command tree. `folo describe` serialises it, so help strings and defaults here are the documented
//! surface: keep them accurate.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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
    /// Print the whole CLI surface (commands, arguments, types, defaults, help) as JSON.
    Describe,
    /// Check the environment: version, OS, CPU, memory, web app build.
    Doctor(DoctorArgs),
    /// Train a detector.
    Train(TrainArgs),
    /// Evaluate weights on a dataset split (mAP, precision, recall).
    Val(ValArgs),
    /// Run inference on images or folders.
    Predict(PredictArgs),
    /// Export weights to another format.
    Export(ExportArgs),
    /// Benchmark inference or training speed.
    Bench(BenchArgs),
    /// Inspect and generate datasets.
    #[command(subcommand)]
    Dataset(DatasetCommand),
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
    /// Port to bind (0 picks a free one).
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
}

/// `folo doctor`.
#[derive(Debug, Args)]
pub struct DoctorArgs {
    /// Built web app folder to check [default: web/dist].
    #[arg(long)]
    pub web_dir: Option<PathBuf>,
}

/// Optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum OptimizerArg {
    /// SGD for long runs, AdamW for short ones.
    Auto,
    /// SGD with Nesterov momentum.
    Sgd,
    /// AdamW.
    Adamw,
}

/// `folo train`. Mirrors `TrainConfig` in docs/API.md.
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Dataset: a folder (YOLO or COCO), a data.yaml, or a registered dataset id.
    #[arg(long)]
    pub data: String,
    /// Architecture (folo-n, folo-s, folo-m, folo-l, folo-x) or .folo weights to fine-tune.
    #[arg(long, default_value = "folo-n")]
    pub model: String,
    /// Run name [default: <dataset>-<arch suffix>].
    #[arg(long)]
    pub name: Option<String>,
    /// Number of epochs.
    #[arg(long, default_value_t = 100)]
    pub epochs: u32,
    /// Batch size.
    #[arg(long, default_value_t = 16)]
    pub batch: u32,
    /// Square input size in pixels (multiple of 32).
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Optimizer.
    #[arg(long, value_enum, default_value_t = OptimizerArg::Auto)]
    pub optimizer: OptimizerArg,
    /// Initial learning rate.
    #[arg(long, default_value_t = 0.01)]
    pub lr0: f32,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Mosaic augmentation probability.
    #[arg(long, default_value_t = 1.0)]
    pub mosaic: f32,
    /// Mixup augmentation probability.
    #[arg(long, default_value_t = 0.0)]
    pub mixup: f32,
    /// HSV jitter gains h,s,v.
    #[arg(long, value_delimiter = ',', num_args = 3, default_values_t = [0.015, 0.7, 0.4])]
    pub hsv: Vec<f32>,
    /// Horizontal flip probability.
    #[arg(long, default_value_t = 0.5)]
    pub fliplr: f32,
    /// Scale jitter (± fraction).
    #[arg(long, default_value_t = 0.5)]
    pub scale: f32,
    /// Translation jitter (fraction of the image).
    #[arg(long, default_value_t = 0.1)]
    pub translate: f32,
    /// Disable mosaic for the last N epochs.
    #[arg(long, default_value_t = 10)]
    pub close_mosaic: u32,
    /// Folder that run outputs (weights, metrics) go under.
    #[arg(long, default_value = "runs")]
    pub project: PathBuf,
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

/// `folo val`.
#[derive(Debug, Args)]
pub struct ValArgs {
    /// Weights (.folo) to evaluate.
    #[arg(long)]
    pub model: PathBuf,
    /// Dataset: a folder, a data.yaml, or a registered dataset id.
    #[arg(long)]
    pub data: String,
    /// Split to evaluate.
    #[arg(long, value_enum, default_value_t = SplitArg::Val)]
    pub split: SplitArg,
    /// Batch size.
    #[arg(long, default_value_t = 16)]
    pub batch: u32,
    /// Square input size in pixels.
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// Confidence threshold (low for mAP).
    #[arg(long, default_value_t = 0.001)]
    pub conf: f32,
    /// NMS IoU threshold.
    #[arg(long, default_value_t = 0.7)]
    pub iou: f32,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
}

/// `folo predict`.
#[derive(Debug, Args)]
pub struct PredictArgs {
    /// Weights (.folo).
    #[arg(long)]
    pub model: PathBuf,
    /// Images or folders of images.
    #[arg(required = true)]
    pub sources: Vec<PathBuf>,
    /// Confidence threshold.
    #[arg(long, default_value_t = 0.25)]
    pub conf: f32,
    /// NMS IoU threshold.
    #[arg(long, default_value_t = 0.7)]
    pub iou: f32,
    /// Maximum detections per image.
    #[arg(long, default_value_t = 300)]
    pub max_det: u32,
    /// Square input size in pixels.
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Write annotated images to this folder.
    #[arg(long)]
    pub save: Option<PathBuf>,
}

/// Export format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ExportFormatArg {
    /// `.folo` with f16 weights.
    FoloF16,
    /// ONNX.
    Onnx,
}

/// `folo export`.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// Weights (.folo) to export.
    #[arg(long)]
    pub model: PathBuf,
    /// Target format.
    #[arg(long, value_enum, default_value_t = ExportFormatArg::FoloF16)]
    pub format: ExportFormatArg,
    /// Output file [default: next to the input].
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Square input size baked into the export.
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
}

/// What `folo bench` measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum BenchMode {
    /// Forward pass latency and throughput.
    Infer,
    /// Training step throughput.
    Train,
}

/// `folo bench`.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// Architecture (folo-n ... folo-x) or .folo weights.
    #[arg(long, default_value = "folo-n")]
    pub model: String,
    /// What to measure.
    #[arg(long, value_enum, default_value_t = BenchMode::Infer)]
    pub mode: BenchMode,
    /// Square input size in pixels.
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// Batch size.
    #[arg(long, default_value_t = 1)]
    pub batch: u32,
    /// Device: auto, cpu or gpu:<n>.
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Timed iterations.
    #[arg(long, default_value_t = 100)]
    pub iters: u32,
    /// Untimed warmup iterations.
    #[arg(long, default_value_t = 10)]
    pub warmup: u32,
}

/// `folo dataset`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Validate a dataset: missing or broken labels, out-of-range boxes, unreadable images.
    Check(DatasetPathArgs),
    /// Print dataset statistics (class counts, box sizes, boxes per image).
    Stats(DatasetPathArgs),
    /// Generate a synthetic shapes dataset (YOLO layout).
    Synth(SynthArgs),
}

/// A dataset location.
#[derive(Debug, Args)]
pub struct DatasetPathArgs {
    /// Dataset folder or data.yaml.
    pub path: PathBuf,
}

/// `folo dataset synth`.
#[derive(Debug, Args)]
pub struct SynthArgs {
    /// Dataset name.
    pub name: String,
    /// Number of images (split 80 / 20 into train / val).
    #[arg(long, default_value_t = 1000)]
    pub images: u64,
    /// Class names; circle, square, triangle and star get those shapes.
    #[arg(long, value_delimiter = ',', default_values_t = ["circle".to_string(), "square".to_string(), "triangle".to_string()])]
    pub classes: Vec<String>,
    /// Square image size in pixels.
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// RNG seed.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Output folder [default: datasets/<name>].
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// `folo mcp`.
#[derive(Debug, Args)]
pub struct McpArgs {
    /// Answer with mock data (like `folo serve --mock`).
    #[arg(long)]
    pub mock: bool,
}
