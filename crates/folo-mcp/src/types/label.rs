//! Arguments and results of the labeling tools (docs/MCP.md, Labeling), which `folo label --json` prints too.
//!
//! They are small on purpose: an agent pays for every byte. Ids instead of records, counts instead of lists, and
//! boxes as [`RelBox`] arrays (`[class, x1, y1, x2, y2]` with the corners in 0..1000 of the image's width and
//! height, the units the labeling agents answer in), so a box means the same on the original image and on any
//! view of it.

use folo_api::{
    LabelAgentId, LabelCounts, LabelEffort, LabelImageStatus, LabelJobStatus, LabelOutcome, LabelRun,
    LabelScope, LabelSource, LabelUsage, SkippedImage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A box: `[class, x1, y1, x2, y2]`, the class index and the corners in 0..1000 of the image's width and height.
pub type RelBox = [u32; 5];

/// `list_label_agents` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelAgentsResult {
    /// Claude Code, then Codex.
    pub agents: Vec<LabelAgentSummary>,
}

/// One agent CLI on the server's machine.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelAgentSummary {
    /// `claude` or `codex`: `start_labeling`'s `agent`.
    pub id: LabelAgentId,
    /// Its CLI is installed (signed in is only known once a run calls it).
    pub available: bool,
    /// What `--version` printed.
    pub version: Option<String>,
    /// Model names to suggest (any name the CLI takes works).
    pub models: Vec<String>,
    /// The model a run uses when it names none; `null`: the CLI's default.
    pub default_model: Option<String>,
    /// The effort a run uses when it names none.
    pub default_effort: LabelEffort,
    /// How to make it available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

/// `list_label_jobs` item.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelJobSummary {
    /// Job id, e.g. `lb_3f9a`.
    pub id: String,
    /// Name.
    pub name: String,
    /// Whether an agent run is active.
    pub status: LabelJobStatus,
    /// Class names (first 100); a box's class indexes them.
    pub classes: Vec<String>,
    /// Image counts.
    pub counts: LabelCounts,
    /// The dataset last exported from the job.
    pub dataset_id: Option<String>,
}

/// `create_label_job`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateLabelJobArgs {
    /// Job (and default dataset) name.
    pub name: String,
    /// Class names (1 to 80, distinct).
    pub classes: Vec<String>,
    /// What the agent should know: what counts as each class, what to skip.
    #[serde(default)]
    pub instructions: String,
    /// Folders (searched recursively) or image files on the server's machine; read in place.
    #[serde(default)]
    #[schemars(length(max = 100))]
    pub paths: Vec<String>,
}

/// `add_label_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddLabelImagesArgs {
    /// Job id.
    pub job_id: String,
    /// Folders (searched recursively) or image files on the server's machine; read in place.
    #[schemars(length(min = 1, max = 100))]
    pub paths: Vec<String>,
}

/// `create_label_job` and `add_label_images` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelImagesAddedResult {
    /// Job id.
    pub job_id: String,
    /// Images the job has now.
    pub images: u64,
    /// Images added by this call.
    pub added: u64,
    /// Files left out (duplicates, not images, over the limit).
    pub skipped: u64,
    /// The first few left out, with the reason.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_first: Vec<SkippedImage>,
}

/// Tools that take only a job id.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobIdArgs {
    /// Job id, e.g. `lb_3f9a`.
    pub job_id: String,
}

/// `start_labeling`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StartLabelingArgs {
    /// Job id.
    pub job_id: String,
    /// The agent (list_label_agents).
    pub agent: LabelAgentId,
    /// Model; default: the agent's default_model.
    #[serde(default)]
    pub model: Option<String>,
    /// Reasoning effort; default: the agent's default_effort.
    #[serde(default)]
    pub effort: Option<LabelEffort>,
    /// `unlabeled` (pending and failed images) or `all` (relabel every image, hand edits too).
    #[serde(default)]
    pub scope: LabelScope,
    /// Exactly these images instead of `scope` (e.g. to ask again about some, with a `note`).
    #[serde(default)]
    pub image_ids: Option<Vec<String>>,
    /// An extra instruction for this run only, e.g. what the last labels got wrong.
    #[serde(default)]
    pub note: String,
    /// Images per agent call (1 to 16).
    #[serde(default)]
    pub batch: Option<u32>,
    /// Agent calls in flight (1 to 8, default 3).
    #[serde(default)]
    pub parallel: Option<u32>,
}

/// `start_labeling` and `stop_labeling` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelRunStarted {
    /// Job id.
    pub job_id: String,
    /// `running`, `stopping` or `idle`.
    pub status: LabelJobStatus,
    /// The run (`total`: images it labels).
    pub run: Option<LabelRunSummary>,
}

/// An agent run of a job, without its timestamps and tokens (`label_status`'s `usage` sums those of every run).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelRunSummary {
    /// The agent.
    pub agent: LabelAgentId,
    /// Model; `null`: the CLI's default.
    pub model: Option<String>,
    /// Reasoning effort.
    pub effort: LabelEffort,
    /// Images the run labels.
    pub total: u64,
    /// Images it has labeled.
    pub done: u64,
    /// Images whose agent call failed.
    pub failed: u64,
    /// `null` while it runs.
    pub outcome: Option<LabelOutcome>,
    /// Why it failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// What to do about it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

impl From<LabelRun> for LabelRunSummary {
    fn from(r: LabelRun) -> Self {
        Self {
            agent: r.agent,
            model: r.model,
            effort: r.effort,
            total: r.total,
            done: r.done,
            failed: r.failed,
            outcome: r.outcome,
            error: r.error,
            hint: r.hint,
        }
    }
}

/// `label_status`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LabelStatusArgs {
    /// Job id.
    pub job_id: String,
    /// How many of the last log lines to include.
    #[serde(default = "default_label_log")]
    #[schemars(range(min = 0, max = 50))]
    pub log_lines: u32,
}

fn default_label_log() -> u32 {
    5
}

/// `label_status` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelStatusResult {
    /// Job id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Whether an agent run is active.
    pub status: LabelJobStatus,
    /// Class names; a box's class indexes them.
    pub classes: Vec<String>,
    /// Image counts.
    pub counts: LabelCounts,
    /// Labeled images a person or an agent has checked (`review_label_images`, `set_label_boxes`).
    pub reviewed: u64,
    /// The active run, else the last one.
    pub run: Option<LabelRunSummary>,
    /// Tokens of every run of the job.
    pub usage: LabelUsage,
    /// The dataset last exported from the job.
    pub dataset_id: Option<String>,
    /// The last log lines, oldest first (`warn:` / `error:` marks those levels).
    pub log: Vec<String>,
}

/// `list_label_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListLabelImagesArgs {
    /// Job id.
    pub job_id: String,
    /// Only this status.
    #[serde(default)]
    pub status: Option<LabelImageStatus>,
    /// Only reviewed or not reviewed images (status `labeled` + `false`: what is left to review).
    #[serde(default)]
    pub reviewed: Option<bool>,
    /// Include each image's boxes.
    #[serde(default)]
    pub include_boxes: bool,
    /// `next_cursor` of the previous page.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Page size.
    #[serde(default)]
    #[schemars(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

/// `list_label_images` item.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelImageSummary {
    /// Image id within the job.
    pub id: String,
    /// File name.
    pub file: String,
    /// Where it stands.
    pub status: LabelImageStatus,
    /// Its boxes.
    pub num_boxes: u32,
    /// Who drew them: `agent` (a run) or `manual` (set_label_boxes, Studio).
    pub source: Option<LabelSource>,
    /// Checked by a person or an agent.
    pub reviewed: bool,
    /// Why its last agent call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Its boxes (with include_boxes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boxes: Option<Vec<RelBox>>,
}

/// `view_label_image`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ViewLabelImageArgs {
    /// Job id.
    pub job_id: String,
    /// The image; omit for the next one to review (labeled, not reviewed).
    #[serde(default)]
    pub image_id: Option<String>,
    /// Long side of the view in pixels.
    #[serde(default = "default_view_size")]
    #[schemars(range(min = 256, max = 1568))]
    pub size: u32,
}

/// Default long side of `view_label_image`.
pub fn default_view_size() -> u32 {
    1024
}

/// `view_label_image` result (the image itself is the result's `image` content).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelImageViewResult {
    /// Job id.
    pub job_id: String,
    /// Class names; a box's class indexes them.
    pub classes: Vec<String>,
    /// Labeled images not reviewed yet (this one included).
    pub to_review: u64,
    /// The image; `null` when nothing is left to review.
    pub image: Option<LabelImageView>,
}

/// One image with its boxes, numbered from 1 in this order on the view.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelImageView {
    /// Image id.
    pub id: String,
    /// File name.
    pub file: String,
    /// Width of the original, pixels.
    pub width: u32,
    /// Height of the original, pixels.
    pub height: u32,
    /// Where it stands.
    pub status: LabelImageStatus,
    /// Who drew the boxes.
    pub source: Option<LabelSource>,
    /// Checked by a person or an agent.
    pub reviewed: bool,
    /// Why its last agent call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// `[class, x1, y1, x2, y2]` in 0..1000; box n of the view is `boxes[n-1]`.
    pub boxes: Vec<RelBox>,
}

/// `set_label_boxes`: either `boxes`, or edits of the current boxes by their view number.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetLabelBoxesArgs {
    /// Job id.
    pub job_id: String,
    /// Image id.
    pub image_id: String,
    /// All the image's boxes, replacing what it has (`[]`: no objects, a background image).
    #[serde(default)]
    pub boxes: Option<Vec<RelBox>>,
    /// Box numbers (1-based, as the view shows them) to delete.
    #[serde(default)]
    pub remove: Vec<u32>,
    /// `[number, class]` pairs: give those boxes another class.
    #[serde(default)]
    pub relabel: Vec<[u32; 2]>,
    /// Boxes to append.
    #[serde(default)]
    pub add: Vec<RelBox>,
}

/// `review_label_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReviewLabelImagesArgs {
    /// Job id.
    pub job_id: String,
    /// Labeled images whose boxes are right.
    #[schemars(length(min = 1, max = 500))]
    pub image_ids: Vec<String>,
    /// False takes the mark back.
    #[serde(default = "yes")]
    pub reviewed: bool,
}

fn yes() -> bool {
    true
}

/// `review_label_images` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ReviewLabelImagesResult {
    /// Images marked.
    pub updated: u64,
    /// Images that could not be (first 20).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed: Vec<ReviewFailure>,
    /// Labeled images not reviewed yet.
    pub to_review: u64,
}

/// An image `review_label_images` could not mark.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ReviewFailure {
    /// Image id.
    pub id: String,
    /// Error code.
    pub code: String,
    /// What went wrong.
    pub message: String,
}

/// `export_label_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportLabelDatasetArgs {
    /// Job id.
    pub job_id: String,
    /// Dataset name; default: the job's.
    #[serde(default)]
    pub name: Option<String>,
    /// Share of the images in the val split (0 to 0.5).
    #[serde(default = "default_val_fraction")]
    pub val_fraction: f32,
    /// Seed of the train / val shuffle.
    #[serde(default)]
    pub seed: u64,
}

fn default_val_fraction() -> f32 {
    0.2
}

/// `export_label_dataset` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelDatasetResult {
    /// The registered dataset: `train`'s `dataset_id`.
    pub dataset_id: String,
    /// Its name.
    pub name: String,
    /// Its folder on the server's machine.
    pub path: String,
    /// Train images.
    pub train: u64,
    /// Val images.
    pub val: u64,
}
