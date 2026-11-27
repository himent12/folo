//! `folo dataset` subcommands and arguments (`check`, `stats`, `synth`, `convert`, `split`, `merge`, `filter`,
//! `export`).

use std::path::PathBuf;

use clap::{Args, Subcommand, ValueEnum};

use super::SplitArg;

/// `folo dataset`.
#[derive(Debug, Subcommand)]
pub enum DatasetCommand {
    /// Validate a dataset: broken labels, class-id overflow, out-of-range boxes, corrupt or missing images,
    /// duplicate images, train/val split leakage (plus optional perceptual `dHash` near-duplicates) and empty
    /// images. Exits 1 when there are errors or warnings.
    Check(CheckArgs),
    /// Print dataset statistics (class counts, box sizes, boxes per image).
    Stats(StatsArgs),
    /// Generate a synthetic shapes dataset (YOLO layout).
    Synth(SynthArgs),
    /// Convert a dataset between YOLO, COCO and Pascal VOC (images are copied unchanged).
    Convert(ConvertArgs),
    /// Split a dataset into deterministic multi-label stratified train / val / optional test splits.
    Split(SplitArgs),
    /// Merge one or more datasets (any mix of YOLO, COCO, Pascal VOC) into a unified YOLO dataset with class
    /// unification/remapping, split policies, and collision-safe filenames.
    Merge(MergeArgs),
    /// Filter a dataset by class whitelist/blacklist, split selection, box area/pixel size, image dimensions, and
    /// optional empty-image dropping.
    Filter(FilterArgs),
    /// Export a dataset (YOLO, COCO or Pascal VOC) to spec-compliant COCO instances JSON with selectable splits
    /// and copy / hard-link / reference image handling.
    Export(ExportDatasetArgs),
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

/// Target format of `folo dataset convert`, `folo dataset split`, and `folo dataset filter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum TargetFormatArg {
    /// YOLO: images/<split>/, labels/<split>/*.txt and data.yaml.
    Yolo,
    /// COCO: images/<split>/ and annotations/instances_<split>.json.
    Coco,
    /// Pascal VOC: JPEGImages/, Annotations/*.xml and ImageSets/Main/*.txt.
    Voc,
}

/// How class names are matched during `folo dataset merge` and `folo dataset filter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ClassNormalizeArg {
    /// Exact string equality (case- and whitespace-sensitive).
    #[default]
    Exact,
    /// Trim outer whitespace, collapse internal whitespace runs, and compare lowercase.
    #[value(name = "case-and-whitespace")]
    CaseAndWhitespace,
}

impl From<ClassNormalizeArg> for folo_data::ClassNormalize {
    fn from(n: ClassNormalizeArg) -> Self {
        match n {
            ClassNormalizeArg::Exact => folo_data::ClassNormalize::Exact,
            ClassNormalizeArg::CaseAndWhitespace => folo_data::ClassNormalize::CaseAndWhitespace,
        }
    }
}

/// How splits from input datasets are combined during `folo dataset merge`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum MergeSplitArg {
    /// Keep each image in its original split (`train` -> `train`, `val` -> `val`, `test` -> `test`).
    #[default]
    Preserve,
    /// Pool all images from every input split into `train`.
    #[value(name = "pool-to-train")]
    PoolToTrain,
    /// Pool all images and re-split them via deterministic multi-label stratification (`--val`, optional `--test`,
    /// `--seed`).
    Stratified,
}

/// How filename collisions are resolved during `folo dataset merge`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CollisionArg {
    /// Keep original filenames when unique within a split; prefix colliding stems with `<prefix>_`.
    #[default]
    #[value(name = "prefix-on-collision")]
    PrefixOnCollision,
    /// Prefix every output image filename with its source dataset prefix (`<prefix>_<stem>.<ext>`).
    #[value(name = "always-prefix")]
    AlwaysPrefix,
}

impl From<CollisionArg> for folo_data::CollisionPolicy {
    fn from(c: CollisionArg) -> Self {
        match c {
            CollisionArg::PrefixOnCollision => folo_data::CollisionPolicy::PrefixOnCollision,
            CollisionArg::AlwaysPrefix => folo_data::CollisionPolicy::AlwaysPrefix,
        }
    }
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
    #[arg(long, conflicts_with_all = ["perceptual", "dhash_threshold"])]
    pub headers_only: bool,
    /// Also run 64-bit perceptual difference-hash (`dHash`) near-duplicate and cross-split leakage detection
    /// (catches recompressed, resized, or lightly cropped copies). Incompatible with --headers-only.
    #[arg(long, conflicts_with = "headers_only")]
    pub perceptual: bool,
    /// Maximum Hamming distance [0..=64] for perceptual `dHash` near-duplicate detection (implies `--perceptual`;
    /// default 6 with `--perceptual`). Incompatible with --headers-only.
    #[arg(long, value_name = "BITS", conflicts_with = "headers_only")]
    pub dhash_threshold: Option<u32>,
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
    /// Comma-separated class names (at most 8). circle, square, triangle, star, ring, cross, hexagon and heart get
    /// those shapes; other names take the next unused shape.
    #[arg(long, value_name = "NAMES", value_delimiter = ',', default_value = "circle,square,triangle")]
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

/// `folo dataset merge`.
#[derive(Debug, Args)]
pub struct MergeArgs {
    /// Input datasets to merge (one or more folders, data.yaml, COCO .json, or Pascal VOC .xml).
    #[arg(required = true)]
    pub inputs: Vec<PathBuf>,
    /// Output YOLO dataset folder (created; must be empty if it exists).
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// How class names across datasets are matched and unified.
    #[arg(long, value_enum, default_value_t = ClassNormalizeArg::Exact)]
    pub normalize: ClassNormalizeArg,
    /// Explicit class remap as `SOURCE=TARGET` (repeatable), e.g. `--remap automobile=car`.
    #[arg(long = "remap", value_name = "SRC=DST")]
    pub remap: Vec<String>,
    /// Drop every box of this class name from all input datasets (repeatable).
    #[arg(long = "drop-class", value_name = "NAME")]
    pub drop_class: Vec<String>,
    /// Drop classes not listed in `--remap` instead of keeping them.
    #[arg(long)]
    pub drop_unmapped: bool,
    /// Drop images that end up with zero boxes after class dropping.
    #[arg(long)]
    pub drop_empty_images: bool,
    /// How splits from input datasets are combined.
    #[arg(long, value_enum, default_value_t = MergeSplitArg::Preserve)]
    pub splits: MergeSplitArg,
    /// With --splits stratified: fraction of images assigned to `val` [0..=1].
    #[arg(long, value_name = "RATIO")]
    pub val: Option<f64>,
    /// With --splits stratified: fraction of images assigned to `test` [0..=1] [default: 0.0].
    #[arg(long, value_name = "RATIO")]
    pub test: Option<f64>,
    /// With --splits stratified: RNG seed for deterministic stratification.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// How colliding image filenames are disambiguated.
    #[arg(long, value_enum, default_value_t = CollisionArg::PrefixOnCollision)]
    pub collisions: CollisionArg,
    /// Optional per-dataset filename prefixes (comma-separated or repeatable; count must match inputs).
    #[arg(long = "prefix", value_name = "TAG", value_delimiter = ',')]
    pub prefix: Vec<String>,
}

/// `folo dataset filter`.
#[derive(Debug, Args)]
pub struct FilterArgs {
    /// Dataset to filter: a folder (YOLO, COCO or Pascal VOC), data.yaml, COCO .json or Pascal VOC .xml.
    pub path: PathBuf,
    /// Output dataset folder (created; must be empty if it exists).
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Format of the input.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub from: FormatArg,
    /// Target format [default: the input dataset's format].
    #[arg(long, value_enum)]
    pub to: Option<TargetFormatArg>,
    /// Keep only boxes of these class names (whitelist; comma-separated or repeatable).
    #[arg(long = "include-class", value_name = "NAME", value_delimiter = ',')]
    pub include_class: Vec<String>,
    /// Drop boxes of these class names (blacklist, applied after --include-class; comma-separated or repeatable).
    #[arg(long = "exclude-class", value_name = "NAME", value_delimiter = ',')]
    pub exclude_class: Vec<String>,
    /// How class names in --include-class / --exclude-class are matched against the dataset's classes.
    #[arg(long, value_enum, default_value_t = ClassNormalizeArg::Exact)]
    pub normalize: ClassNormalizeArg,
    /// Keep the full original class list and IDs instead of compacting to retained classes.
    #[arg(long)]
    pub keep_all_classes: bool,
    /// Keep only these splits (`train`, `val`, `test`; comma-separated or repeatable).
    #[arg(long = "split", value_enum, value_delimiter = ',')]
    pub split: Vec<SplitArg>,
    /// Minimum normalised box area `w * h` in [0..=1] (inclusive).
    #[arg(long, value_name = "AREA")]
    pub min_box_area: Option<f32>,
    /// Maximum normalised box area `w * h` in [0..=1] (inclusive).
    #[arg(long, value_name = "AREA")]
    pub max_box_area: Option<f32>,
    /// Minimum box width in pixels [>= 0] (inclusive).
    #[arg(long, value_name = "PX")]
    pub min_box_width: Option<f32>,
    /// Maximum box width in pixels [>= 0] (inclusive).
    #[arg(long, value_name = "PX")]
    pub max_box_width: Option<f32>,
    /// Minimum box height in pixels [>= 0] (inclusive).
    #[arg(long, value_name = "PX")]
    pub min_box_height: Option<f32>,
    /// Maximum box height in pixels [>= 0] (inclusive).
    #[arg(long, value_name = "PX")]
    pub max_box_height: Option<f32>,
    /// Minimum image width in pixels [>= 1] (inclusive).
    #[arg(long, value_name = "PX")]
    pub min_image_width: Option<u32>,
    /// Maximum image width in pixels [>= 1] (inclusive).
    #[arg(long, value_name = "PX")]
    pub max_image_width: Option<u32>,
    /// Minimum image height in pixels [>= 1] (inclusive).
    #[arg(long, value_name = "PX")]
    pub min_image_height: Option<u32>,
    /// Maximum image height in pixels [>= 1] (inclusive).
    #[arg(long, value_name = "PX")]
    pub max_image_height: Option<u32>,
    /// Remove images that have zero surviving boxes after filtering.
    #[arg(long)]
    pub drop_empty_images: bool,
}

/// Export format of `folo dataset export`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ExportDatasetFormatArg {
    /// Spec-compliant COCO instances JSON (`annotations/instances_<split>.json`).
    #[default]
    Coco,
}

/// How `folo dataset export --format coco` handles source images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum CocoImageModeArg {
    /// Copy images byte for byte into `<out>/images/<split>/` (default; self-contained).
    #[default]
    Copy,
    /// Hard-link images into `<out>/images/<split>/`, falling back to a byte-for-byte copy across filesystems.
    Link,
    /// Do not copy or link images; reference source images in place via relative paths from `<out>/annotations/`.
    Reference,
}

impl From<CocoImageModeArg> for folo_data::convert::CocoImageMode {
    fn from(m: CocoImageModeArg) -> Self {
        match m {
            CocoImageModeArg::Copy => folo_data::convert::CocoImageMode::Copy,
            CocoImageModeArg::Link => folo_data::convert::CocoImageMode::Link,
            CocoImageModeArg::Reference => folo_data::convert::CocoImageMode::Reference,
        }
    }
}

/// `folo dataset export`.
#[derive(Debug, Args)]
pub struct ExportDatasetArgs {
    /// Dataset to export: a folder (YOLO, COCO or Pascal VOC), data.yaml, COCO .json or Pascal VOC .xml.
    pub path: PathBuf,
    /// Output folder (created; must not exist or be empty).
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Target export format (`coco`).
    #[arg(long, value_enum, default_value_t = ExportDatasetFormatArg::Coco)]
    pub format: ExportDatasetFormatArg,
    /// Format of the input dataset.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    pub from: FormatArg,
    /// Splits to export (`train`, `val`, `test`; comma-separated or repeatable; default: all non-empty splits).
    #[arg(long = "split", value_enum, value_delimiter = ',')]
    pub split: Vec<SplitArg>,
    /// How source images are materialized or referenced (`copy`, `link`, or `reference`).
    #[arg(long, value_enum, default_value_t = CocoImageModeArg::Copy)]
    pub images: CocoImageModeArg,
    /// Pretty-print the COCO instances JSON files instead of writing compact JSON.
    #[arg(long)]
    pub pretty: bool,
    /// Optional `info.description` written to each COCO JSON file.
    #[arg(long, value_name = "TEXT")]
    pub description: Option<String>,
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use crate::cli::{Cli, Command, DatasetCommand};

    #[test]
    fn dataset_synth_help_displays_comma_separated_classes_default() {
        let mut cmd = Cli::command();
        let synth = cmd.find_subcommand_mut("dataset").unwrap().find_subcommand_mut("synth").unwrap();
        let help = synth.render_long_help().to_string();
        assert!(help.contains("[default: circle,square,triangle]"), "{help}");
        assert!(help.contains("Comma-separated"), "{help}");

        let parsed = Cli::try_parse_from(["folo", "dataset", "synth", "out"]).unwrap();
        let Command::Dataset(DatasetCommand::Synth(args)) = parsed.command else { panic!("expected synth") };
        assert_eq!(args.classes, ["circle", "square", "triangle"]);

        let custom =
            Cli::try_parse_from(["folo", "dataset", "synth", "out", "--classes", "circle,square"]).unwrap();
        let Command::Dataset(DatasetCommand::Synth(args)) = custom.command else { panic!("expected synth") };
        assert_eq!(args.classes, ["circle", "square"]);
    }
}
