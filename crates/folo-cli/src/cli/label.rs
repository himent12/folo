//! `folo label`: agent labeling (docs/LABELING.md) from the command line, the same operations as the MCP
//! labeling tools.

use std::path::PathBuf;

use clap::builder::NonEmptyStringValueParser;
use clap::{Args, Subcommand, ValueEnum};

/// `folo label`.
#[derive(Debug, Subcommand)]
pub enum LabelCommand {
    /// The agent CLIs that can label images (Claude Code, Codex, Antigravity): installed or not, models, defaults.
    Agents(LabelServerArgs),
    /// The labeling jobs, newest first, with their image counts.
    Jobs(LabelServerArgs),
    /// Create a labeling job with its classes and, optionally, its first images.
    Create(LabelCreateArgs),
    /// Add images (folders, searched recursively, or image files on the server's machine) to a job.
    Add(LabelAddArgs),
    /// Start an agent run on a job's unlabeled images (or --images). Without a `folo serve` owning the data dir, the
    /// run happens in this process, which waits for it.
    Start(LabelStartArgs),
    /// Stop a job's agent run (images already labeled stay).
    Stop(LabelJobArgs),
    /// A job's counts (labeled, failed, reviewed), run progress, tokens used and last log lines.
    Status(LabelStatusArgs),
    /// A job's images: status, box count, who drew them, reviewed.
    Images(LabelImagesArgs),
    /// Write an image with its boxes drawn and numbered to a JPEG, and print its boxes (default: the next image to
    /// review).
    View(LabelViewArgs),
    /// Change an image's boxes, all of them (--box, --clear) or by number (--remove, --relabel, --add). Saved as a
    /// hand edit and marked reviewed.
    Set(LabelSetArgs),
    /// Mark labeled images reviewed: their boxes are right (--undo takes the mark back).
    Review(LabelReviewArgs),
    /// Write the job's labeled images as a YOLO dataset (train / val), register it and print its id.
    Export(LabelExportArgs),
}

/// Where a `folo label` command finds the jobs.
#[derive(Debug, Args)]
pub struct LabelServerArgs {
    /// The data dir whose labeling jobs to use [default: $FOLO_HOME or ~/.folo]. When a `folo serve` owns it, the
    /// command goes through that server.
    #[arg(long, env = "FOLO_DATA_DIR")]
    pub data_dir: Option<PathBuf>,
    /// Use the `folo serve` at this URL (e.g. http://127.0.0.1:7878) instead of the data dir.
    #[arg(long, env = "FOLO_SERVER", value_name = "URL", value_parser = NonEmptyStringValueParser::new())]
    pub server: Option<String>,
    /// The token that `folo serve` was started with (`--token`).
    #[arg(long, env = "FOLO_TOKEN", hide_env_values = true, value_parser = NonEmptyStringValueParser::new())]
    pub token: Option<String>,
}

/// `folo label stop`.
#[derive(Debug, Args)]
pub struct LabelJobArgs {
    /// The job's id (`lb_…`, `folo label jobs`).
    pub job: String,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label create`.
#[derive(Debug, Args)]
pub struct LabelCreateArgs {
    /// The job's name (also the default name of its dataset).
    pub name: String,
    /// The classes to find, comma-separated or repeated (1 to 80).
    #[arg(long, value_delimiter = ',', required = true, num_args = 1..)]
    pub classes: Vec<String>,
    /// What the agent should know: what counts as each class, what to skip.
    #[arg(long, default_value = "", hide_default_value = true)]
    pub instructions: String,
    /// Folders (searched recursively) or image files on the server's machine to add; read in place.
    #[arg(long = "path", value_name = "PATH")]
    pub paths: Vec<String>,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label add`.
#[derive(Debug, Args)]
pub struct LabelAddArgs {
    /// The job's id.
    pub job: String,
    /// Folders (searched recursively) or image files on the server's machine; read in place.
    #[arg(required = true)]
    pub paths: Vec<String>,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// An agent CLI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum AgentArg {
    /// Claude Code (`claude`).
    Claude,
    /// Codex (`codex`).
    Codex,
    /// Antigravity (`agy`).
    Antigravity,
}

/// A reasoning effort.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EffortArg {
    /// Fewest reasoning tokens.
    Low,
    /// Medium.
    Medium,
    /// Most.
    High,
}

/// `folo label start`.
#[derive(Debug, Args)]
pub struct LabelStartArgs {
    /// The job's id.
    pub job: String,
    /// The agent to ask.
    #[arg(long, value_enum)]
    pub agent: AgentArg,
    /// Its model [default: the agent's, `folo label agents`].
    #[arg(long)]
    pub model: Option<String>,
    /// Reasoning effort [default: the agent's].
    #[arg(long, value_enum)]
    pub effort: Option<EffortArg>,
    /// Codex's Fast tier (priority processing, which uses more of the plan); only models that offer it.
    #[arg(long)]
    pub fast: bool,
    /// Exactly these images (comma-separated ids), e.g. to ask again with a --note.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    pub images: Vec<String>,
    /// Every image, replacing the labels they have (hand-drawn ones too); default: the unlabeled ones.
    #[arg(long, conflicts_with = "images")]
    pub all: bool,
    /// An extra instruction for this run only, e.g. what the last labels got wrong.
    #[arg(long, default_value = "", hide_default_value = true)]
    pub note: String,
    /// Images per agent call (1 to 16) [default: the agent's].
    #[arg(long)]
    pub batch: Option<u32>,
    /// Agent calls in flight (1 to 8) [default: 3].
    #[arg(long)]
    pub parallel: Option<u32>,
    /// Wait until the run ends, printing its progress (always so without a `folo serve`).
    #[arg(long)]
    pub wait: bool,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label status`.
#[derive(Debug, Args)]
pub struct LabelStatusArgs {
    /// The job's id.
    pub job: String,
    /// How many of the last log lines to show (0 to 50).
    #[arg(long, default_value_t = 5)]
    pub log: u32,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// An image status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ImageStatusArg {
    /// Not labeled yet.
    Pending,
    /// In an agent call now.
    Labeling,
    /// Labeled.
    Labeled,
    /// An agent call failed and it has no labels (a labeled image keeps its labels and status).
    Failed,
}

/// `folo label images`.
#[derive(Debug, Args)]
pub struct LabelImagesArgs {
    /// The job's id.
    pub job: String,
    /// Only images with this status.
    #[arg(long, value_enum)]
    pub status: Option<ImageStatusArg>,
    /// Only reviewed (true) or not reviewed (false) images.
    #[arg(long)]
    pub reviewed: Option<bool>,
    /// Include each image's boxes ([class, x1, y1, x2, y2] in pixels of its default 1024 px view, which `size`
    /// gives) and their revision.
    #[arg(long)]
    pub boxes: bool,
    /// Skip this many matching images.
    #[arg(long, default_value_t = 0)]
    pub offset: u64,
    /// List at most this many (1 to 1000).
    #[arg(long, default_value_t = 50)]
    pub limit: u64,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label view`.
#[derive(Debug, Args)]
pub struct LabelViewArgs {
    /// The job's id.
    pub job: String,
    /// The image's id [default: the next image to review].
    pub image: Option<String>,
    /// Where to write the JPEG [default: <job>-<image>.jpg in the current folder]; never one of the job's images.
    #[arg(long, short)]
    pub output: Option<PathBuf>,
    /// Long side of the view in pixels (256 to 1568; never more than the image's own).
    #[arg(long, default_value_t = 1024)]
    pub size: u32,
    /// The first box to list [>= 1] (a dense image lists 300 at a time).
    #[arg(long, default_value_t = 1)]
    pub from: u32,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label set`.
#[derive(Debug, Args)]
pub struct LabelSetArgs {
    /// The job's id.
    pub job: String,
    /// The image's id.
    pub image: String,
    /// A box as CLASS,X1,Y1,X2,Y2 (class index, name, `id:INDEX`, or `name:CLASS`, corners in pixels of the view,
    /// see --size), repeatable: the image's boxes become exactly these. A box given as `folo label view` lists it
    /// keeps its exact pixels.
    #[arg(long = "box", value_name = "C,X1,Y1,X2,Y2")]
    pub boxes: Vec<String>,
    /// Remove every box (a background image).
    #[arg(long, conflicts_with = "boxes")]
    pub clear: bool,
    /// Remove box N [>= 1] (as `folo label view` numbers them), repeatable.
    #[arg(long, value_name = "N")]
    pub remove: Vec<u32>,
    /// Give box N another class, as N=CLASS (index, name, `id:INDEX`, or `name:CLASS`), repeatable.
    #[arg(long, value_name = "N=CLASS")]
    pub relabel: Vec<String>,
    /// Add a box, as CLASS,X1,Y1,X2,Y2 (index, name, `id:INDEX`, or `name:CLASS`), repeatable.
    #[arg(long, value_name = "C,X1,Y1,X2,Y2")]
    pub add: Vec<String>,
    /// The revision of the boxes the edit was made on (`folo label view` prints it), required: box numbers are
    /// that revision's, and boxes the edit does not list are that revision's too. Another revision now: refused.
    #[arg(long, required = true, value_name = "REVISION", value_parser = NonEmptyStringValueParser::new())]
    pub expect: String,
    /// The view the coordinates are in, as WxH (`folo label view` prints it); required with --box and --add.
    #[arg(long, value_name = "WxH")]
    pub size: Option<String>,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label review`.
#[derive(Debug, Args)]
pub struct LabelReviewArgs {
    /// The job's id.
    pub job: String,
    /// The images whose boxes are right.
    #[arg(required = true)]
    pub images: Vec<String>,
    /// Mark them not reviewed instead.
    #[arg(long)]
    pub undo: bool,
    /// IMAGE=REVISION: mark IMAGE only if its boxes are still that revision (as checked), repeatable.
    #[arg(long, value_name = "IMAGE=REVISION")]
    pub expect: Vec<String>,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}

/// `folo label export`.
#[derive(Debug, Args)]
pub struct LabelExportArgs {
    /// The job's id.
    pub job: String,
    /// The dataset's name [default: the job's].
    #[arg(long)]
    pub name: Option<String>,
    /// Share of the images in the val split (0 to 0.5).
    #[arg(long, default_value_t = 0.2)]
    pub val_fraction: f32,
    /// Seed of the train / val shuffle.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    #[command(flatten)]
    pub conn: LabelServerArgs,
}
