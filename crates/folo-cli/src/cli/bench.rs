//! `folo bench`: vision benchmarks (docs/BENCH.md) from the command line, the same operations as the MCP
//! benchmark tools.

use std::path::PathBuf;

use clap::builder::NonEmptyStringValueParser;
use clap::{Args, Subcommand, ValueEnum};

use super::SplitArg;

/// `folo bench`.
#[derive(Debug, Subcommand)]
pub enum BenchCommand {
    /// The stress conditions a plan can name, the default plan and the axes of the score.
    Catalog(BenchServerArgs),
    /// The benchmarks, newest first, with their best entry.
    List(BenchServerArgs),
    /// Make a benchmark from a dataset split, a labeling job, or deterministic generated scenes (frozen copies).
    Create(BenchCreateArgs),
    /// A benchmark: its plan and its leaderboard.
    Show(BenchIdArgs),
    /// Delete a benchmark with its entries (none may be running).
    Delete(BenchIdArgs),
    /// Add an entry: a model run by FOLO (--model), or predictions made elsewhere (--predictions). Without a
    /// `folo serve` owning the data dir, a model's entry runs in this process, which waits for it.
    Run(BenchRunArgs),
    /// An entry's progress, and once it is done its score with the numbers behind each axis and each condition.
    Status(BenchStatusArgs),
    /// Stop an entry (nothing of it is scored).
    Stop(BenchEntryArgs),
    /// Compare two finished entries of a benchmark head-to-head (deltas: candidate - baseline).
    #[command(allow_negative_numbers = true)]
    Compare(BenchCompareArgs),
}

/// Where a `folo bench` command finds the benchmarks.
#[derive(Debug, Args)]
pub struct BenchServerArgs {
    /// The data dir whose benchmarks to use [default: $FOLO_HOME or ~/.folo]. When a `folo serve` owns it, the
    /// command goes through that server.
    #[arg(long, env = "FOLO_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
    /// Use the `folo serve` at this URL (e.g. http://127.0.0.1:7878) instead of the data dir.
    #[arg(long, env = "FOLO_SERVER", value_name = "URL", value_parser = NonEmptyStringValueParser::new())]
    pub server: Option<String>,
    /// The token that `folo serve` was started with (`--token`).
    #[arg(long, env = "FOLO_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    pub token: Option<String>,
    /// PEM CA certificate file to trust when connecting to an https:// `folo serve` (for self-signed or private-CA
    /// certificates).
    #[arg(long = "tls-ca", env = "FOLO_TLS_CA", value_name = "PATH")]
    pub tls_ca: Option<PathBuf>,
}

/// `folo bench show` and `delete`.
#[derive(Debug, Args)]
pub struct BenchIdArgs {
    /// The benchmark's id (`bn_…`, `folo bench list`).
    pub bench: String,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}

/// `folo bench stop`.
#[derive(Debug, Args)]
pub struct BenchEntryArgs {
    /// The benchmark's id.
    pub bench: String,
    /// The entry's id (`be_…`, `folo bench show BENCH`).
    pub entry: String,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}

/// `folo bench status`.
#[derive(Debug, Args)]
pub struct BenchStatusArgs {
    /// The benchmark's id.
    pub bench: String,
    /// The entry's id (`be_…`, `folo bench show BENCH`).
    pub entry: String,
    /// Wait until the entry ends, printing its progress.
    #[arg(long)]
    pub wait: bool,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}

/// `folo bench create`.
#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("source").required(true).args(["dataset", "data", "label_job", "generated"])))]
pub struct BenchCreateArgs {
    /// The benchmark's name.
    pub name: String,
    /// A registered dataset's id.
    #[arg(long, value_name = "ID")]
    pub dataset: Option<String>,
    /// A YOLO or COCO dataset folder (on the server's machine), registered first (a folder registered before is
    /// the dataset it already is).
    #[arg(long, value_name = "PATH")]
    pub data: Option<String>,
    /// The split of the dataset to take [default: val].
    #[arg(long, value_enum, conflicts_with_all = ["label_job", "generated"])]
    pub split: Option<SplitArg>,
    /// A labeling job's id: its labeled images, with their boxes as truth.
    #[arg(long, value_name = "ID")]
    pub label_job: Option<String>,
    /// Deterministic generated synthetic scenes (procedural shapes via `folo-data`, no dataset required).
    #[arg(long, visible_alias = "synthetic")]
    pub generated: bool,
    /// With --generated: number of scenes to generate [1..=5000] [default: 64].
    #[arg(long, requires = "generated")]
    pub images: Option<u32>,
    /// With --generated: comma-separated class names (1 to 8) [default: circle,square,triangle].
    #[arg(long, value_delimiter = ',', requires = "generated")]
    pub classes: Option<Vec<String>>,
    /// With --generated: long side of each generated image in pixels [32..=4096] [default: 640].
    #[arg(long, requires = "generated")]
    pub imgsz: Option<u32>,
    /// With --generated: mix aspect ratios (1:1, 4:3, 3:4, 16:9, 3:2) instead of square images.
    #[arg(long, requires = "generated")]
    pub vary_aspect: bool,
    /// What the benchmark stands for: where the camera is, what makes the job hard.
    #[arg(long, default_value = "", hide_default_value = true)]
    pub description: String,
    /// At most this many images (1 to 5000), sampled with --seed when the source has more [default: 500; for
    /// --generated: 64].
    #[arg(long)]
    pub max_images: Option<u32>,
    /// Seed of the sample, generated scenes, and conditions' noise [0..=9007199254740991].
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// The plan as JSON (`{"conditions":[{"condition":"haze","severities":[1,3],"weight":1}]}`), or a file that
    /// holds it [default: the catalog's default plan].
    #[arg(long, value_name = "JSON|FILE")]
    pub plan: Option<String>,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}

/// `folo bench run`.
#[derive(Debug, Args)]
#[command(group(clap::ArgGroup::new("contestant").required(true).args(["model", "predictions"])))]
pub struct BenchRunArgs {
    /// The benchmark's id.
    pub bench: String,
    /// A model of the data dir, by id or name: a trained run's weights, a .folo file in its models/ folder, or
    /// published weights pulled with `folo weights pull` (named as published, e.g. folo-n-coco).
    #[arg(long, value_name = "ID")]
    pub model: Option<String>,
    /// A file of predictions made elsewhere: JSON Lines of {"image", "condition"?, "severity"?, "boxes": [{"class",
    /// "score", "x1", "y1", "x2", "y2"}]} records (docs/BENCH.md), read here and sent in parts.
    #[arg(long, value_name = "FILE")]
    pub predictions: Option<PathBuf>,
    /// The entry's name on the leaderboard [default: the model's]; required with --predictions.
    #[arg(long)]
    pub name: Option<String>,
    /// --model: run only the first this many images of the benchmark [>= 1] [default: all].
    #[arg(long)]
    pub max_images: Option<u32>,
    /// --model: run the unchanged images only, not the plan's conditions.
    #[arg(long)]
    pub no_conditions: bool,
    /// --predictions: the model gives no confidences (every score is taken as 1).
    #[arg(long)]
    pub unscored: bool,
    /// --model through a `folo serve`: wait until the entry ends, printing its progress (always so without one).
    #[arg(long)]
    pub wait: bool,
    /// --model without a `folo serve`: the device to run on, auto, cpu or gpu:<n> [default: $FOLO_DEVICE, else
    /// auto]. A server runs entries on its own --device.
    #[arg(long)]
    pub device: Option<String>,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}

/// Output format for `folo bench compare`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum BenchCompareFormat {
    /// Human-readable aligned tables.
    #[default]
    Table,
    /// GitHub-flavored Markdown report.
    #[value(alias = "md")]
    Markdown,
    /// CSV rows (overall, accuracy, axes, calibration/localization, speed, and conditions).
    Csv,
}

impl BenchCompareFormat {
    /// Canonical lowercase name of the format.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Markdown => "markdown",
            Self::Csv => "csv",
        }
    }
}

/// `folo bench compare`.
#[derive(Debug, Args)]
pub struct BenchCompareArgs {
    /// The benchmark's id (`bn_…`, `folo bench list`).
    pub bench: String,
    /// Baseline entry id (`be_…`, reference).
    pub baseline: String,
    /// Candidate entry id (`be_…`, compared against baseline).
    pub candidate: String,
    /// Output format for stdout or `--out`: table, markdown (alias: md), or csv.
    #[arg(long, value_enum, default_value_t = BenchCompareFormat::Table)]
    pub format: BenchCompareFormat,
    /// Write the formatted comparison report atomically to this new file (never overwrites).
    #[arg(long, visible_alias = "output", value_name = "PATH")]
    pub out: Option<PathBuf>,
    /// Exit non-zero if `candidate.score - baseline.score < --min-delta` (default threshold: 0.0).
    #[arg(long)]
    pub fail_on_regression: bool,
    /// With --fail-on-regression: minimum allowed score delta in points (negative allows small drops, e.g. -0.5)
    /// [default: 0.0].
    #[arg(long, requires = "fail_on_regression", value_name = "POINTS")]
    pub min_delta: Option<f64>,
    #[command(flatten)]
    pub conn: BenchServerArgs,
}
