//! `folo dataset` subcommands and arguments (`check`, `stats`, `synth`, `convert`, `split`).

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};

/// `folo dataset`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Validate a dataset: broken labels, class-id overflow, out-of-range boxes, corrupt or missing images,
    /// duplicate images, train/val split leakage and empty images. Exits 1 when there are errors or warnings.
    Check(CheckArgs),
    /// Print dataset statistics (class counts, box sizes, boxes per image).
    Stats(StatsArgs),
    /// Generate a synthetic shapes dataset (YOLO layout).
    Synth(SynthArgs),
    /// Convert a dataset between YOLO, COCO and Pascal VOC (images are copied unchanged).
    Convert(ConvertArgs),
    /// Split a dataset into deterministic multi-label stratified train / val / optional test splits.
    Split(SplitArgs),
}

/// Dataset format on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FormatArg {
    /// Detect from the files present.
    Auto,
    /// YOLO txt labels (data.yaml, images/ + labels/).
    Yolo,
    /// COCO instances JSON.
    Coco,
    /// Pascal VOC XML (Annotations/*.xml, JPEGImages/, optional ImageSets/Main/*.txt).
    Voc,
}

/// Target format of `folo dataset convert` and `folo dataset split`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum TargetFormatArg {
    /// YOLO: images/<split>/, labels/<split>/*.txt and data.yaml.
    Yolo,
    /// COCO: images/<split>/ and annotations/instances_<split>.json.
    Coco,
    /// Pascal VOC: JPEGImages/, Annotations/*.xml and ImageSets/Main/*.txt.
    Voc,
}

/// `folo dataset check`.
#[derive(Debug, Args)]
pub struct CheckArgs {
    /// Dataset folder, data.yaml, COCO .json or Pascal VOC .xml.
    pub path: PathBuf,
    /// Format to read.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub format: FormatArg,
    /// Also show, for every issue, what a cleanup would change (nothing is modified).
    #[arg(long)]
    pub fix_dry_run: bool,
    /// Read only image headers instead of decoding every image (faster; misses truncated image data).
    #[arg(long)]
    pub headers_only: bool,
    /// List only issues at least this serious (`counts` always covers every issue) [default: warning in human
    /// output, info in --json].
    #[arg(long, value_enum)]
    pub min_severity: Option<SeverityArg>,
}

/// Issue severity, most serious first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum SeverityArg {
    /// The loader skips the image.
    Error,
    /// Fixed up on load, or suspicious.
    #[value(alias = "warn")]
    Warning,
    /// Background images.
    Info,
}

/// `folo dataset stats`.
#[derive(Debug, Args)]
pub struct StatsArgs {
    /// Dataset folder, data.yaml, COCO .json or Pascal VOC .xml.
    pub path: PathBuf,
    /// Format to read.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub format: FormatArg,
}

/// Image encoding of `folo dataset synth`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SynthImageFormat {
    /// JPEG, quality 90.
    Jpeg,
    /// Lossless PNG.
    Png,
}

/// `folo dataset synth`.
#[derive(Debug, Args)]
pub struct SynthArgs {
    /// Output folder (created; must be empty if it exists).
    pub out_dir: PathBuf,
    /// Number of images (train + val) [1..=1000000].
    #[arg(long, default_value_t = 1000)]
    pub images: u64,
    /// Class names (at most 8). circle, square, triangle, star, ring, cross, hexagon and heart get those shapes;
    /// other names take the next unused shape.
    #[arg(long, value_delimiter = ',', default_values_t = ["circle".to_string(), "square".to_string(), "triangle".to_string()])]
    pub classes: Vec<String>,
    /// Image side in pixels [32..=8192] (the long side with --vary-aspect).
    #[arg(long, default_value_t = 640)]
    pub imgsz: u32,
    /// RNG seed; the same seed gives the same dataset.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Fraction of images in the validation split [0..=1].
    #[arg(long, default_value_t = 0.2)]
    pub val_frac: f32,
    /// Image encoding.
    #[arg(long, value_enum, default_value_t = SynthImageFormat::Jpeg)]
    pub image_format: SynthImageFormat,
    /// Mix aspect ratios (1:1, 4:3, 3:4, 16:9, 3:2) instead of square images.
    #[arg(long)]
    pub vary_aspect: bool,
}

/// `folo dataset convert`.
#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Dataset to read: a folder, data.yaml, COCO .json or Pascal VOC .xml.
    pub input: PathBuf,
    /// Output folder (created; must be empty if it exists).
    pub output: PathBuf,
    /// Target format.
    #[arg(long, value_enum)]
    pub to: TargetFormatArg,
    /// Format of the input.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub from: FormatArg,
}

/// `folo dataset split`.
#[derive(Debug, Args)]
pub struct SplitArgs {
    /// Dataset to split: a folder (YOLO, COCO or Pascal VOC), data.yaml, COCO .json or Pascal VOC .xml.
    pub path: PathBuf,
    /// Output folder (created; must be empty if it exists).
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Fraction of images assigned to `val` [0..=1] (train receives `1 - val - test`).
    #[arg(long, value_name = "RATIO")]
    pub val: f64,
    /// Fraction of images assigned to `test` [0..=1].
    #[arg(long, value_name = "RATIO", default_value_t = 0.0)]
    pub test: f64,
    /// RNG seed for deterministic multi-label stratification.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Format of the input.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub from: FormatArg,
    /// Target format [default: the input dataset's format].
    #[arg(long, value_enum)]
    pub to: Option<TargetFormatArg>,
}
