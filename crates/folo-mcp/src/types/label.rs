//! Arguments and results of the labeling tools (docs/MCP.md, Labeling), which `folo label --json` prints too.
//!
//! They are small on purpose: an agent pays for every byte. Ids instead of records, counts instead of lists, and
//! boxes as [`PxBox`] arrays: `[class, x1, y1, x2, y2]` in pixels of the image as the view shows it (its `size`),
//! the frame current vision models answer in when they look at an image. The output schemas lose their field
//! descriptions in `tools/list` (docs/MCP.md documents the fields), so these doc comments are for Rust readers.

use folo_api::{
    LabelAgentId, LabelCounts, LabelEffort, LabelImageStatus, LabelJobStatus, LabelOutcome, LabelRun,
    LabelScope, LabelSource, LabelUsage, SkippedImage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A box: `[class, x1, y1, x2, y2]`, the class index and the corners in pixels of a view of the image.
pub type PxBox = [u32; 5];

/// The long side of a view when a call names none.
pub const DEFAULT_VIEW_SIZE: u32 = 1024;

/// `folo label agents` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelAgentsResult {
    /// Claude Code, Codex, then Antigravity.
    pub agents: Vec<LabelAgentSummary>,
}

/// One agent CLI on the server's machine.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelAgentSummary {
    /// `claude`, `codex` or `antigravity`: `start_labeling`'s `agent`.
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

/// `create_label_job`: a new job, or (with `job_id`) more images for one.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateLabelJobArgs {
    /// Add `paths` to this job instead of making one.
    #[serde(default)]
    pub job_id: Option<String>,
    /// New job: its name (also its dataset's).
    #[serde(default)]
    pub name: Option<String>,
    /// New job: class names (1 to 80).
    #[serde(default)]
    pub classes: Option<Vec<String>>,
    /// New job: what counts as each class, what to skip.
    #[serde(default)]
    pub instructions: String,
    /// Folders (recursive) or image files on the server's machine, read in place.
    #[serde(default)]
    #[schemars(length(max = 100))]
    pub paths: Vec<String>,
}

/// `create_label_job` result.
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
    /// The agent CLI.
    pub agent: LabelAgentId,
    /// Default: the agent's.
    #[serde(default)]
    pub model: Option<String>,
    /// Default: the agent's.
    #[serde(default)]
    pub effort: Option<LabelEffort>,
    /// Codex's Fast tier (uses more of the plan).
    #[serde(default)]
    pub fast: bool,
    /// `all` also relabels labeled images.
    #[serde(default)]
    pub scope: LabelScope,
    /// Only these images (e.g. a sample, or again with a `note`).
    #[serde(default)]
    pub image_ids: Option<Vec<String>>,
    /// An instruction for this run only.
    #[serde(default)]
    pub note: String,
    /// Images per agent call (1 to 16).
    #[serde(default)]
    pub batch: Option<u32>,
    /// Agent calls in flight (1 to 8).
    #[serde(default)]
    pub parallel: Option<u32>,
}

/// `start_labeling` result: the run is on its way (`label_status` follows it).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelRunStarted {
    /// Job id.
    pub job_id: String,
    /// Images the run labels.
    pub total: u64,
    /// The model it asks for; `null`: the CLI's default.
    pub model: Option<String>,
}

impl From<&folo_api::LabelJob> for LabelRunStarted {
    fn from(j: &folo_api::LabelJob) -> Self {
        let run = j.run.as_ref();
        Self {
            job_id: j.id.clone(),
            total: run.map_or(0, |r| r.total),
            model: run.and_then(|r| r.model.clone()),
        }
    }
}

/// `stop_labeling` result.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelJobState {
    /// Job id.
    pub job_id: String,
    /// `stopping`, or `idle` once the run has ended.
    pub status: LabelJobStatus,
}

/// An agent run of a job, without its timestamps, tokens and error (`label_status` has the job's tokens, its
/// log the error).
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
        }
    }
}

/// `label_status`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LabelStatusArgs {
    /// Job id.
    pub job_id: String,
    /// Last log lines to include.
    #[serde(default = "default_log_lines")]
    #[schemars(range(max = 50))]
    pub log_lines: u32,
}

fn default_log_lines() -> u32 {
    5
}

/// `label_status` result: the job, its last run, the tokens of all its runs (flat) and its last log lines.
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
    /// Labeled images a person or an agent has checked; left out only by `folo label start --wait` when the
    /// job could not be read again after its run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reviewed: Option<u64>,
    /// The active run, else the last one.
    pub run: Option<LabelRunSummary>,
    /// Tokens of every run of the job.
    #[serde(flatten)]
    pub usage: LabelUsage,
    /// The dataset last exported from the job.
    pub dataset_id: Option<String>,
    /// The last log lines, oldest first (`warn: ` / `error: ` before those levels); left out only by `folo label
    /// start --wait`, as `reviewed` is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<Vec<String>>,
}

/// `list_label_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListLabelImagesArgs {
    /// Job id.
    pub job_id: String,
    /// Only this status.
    #[serde(default)]
    pub status: Option<LabelImageStatus>,
    /// Only reviewed or not reviewed images.
    #[serde(default)]
    pub reviewed: Option<bool>,
    /// Add each image's `boxes` (pixels of its default view, `size`) and `revision`.
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
    /// Who drew them: `agent` (a run) or `manual` (by hand).
    pub source: Option<LabelSource>,
    /// Checked by a person or an agent.
    pub reviewed: bool,
    /// Why its last agent call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// With include_boxes: the default view's size, `[width, height]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<[u32; 2]>,
    /// With include_boxes: the revision of its boxes (`set_label_boxes`' `expect`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// With include_boxes: its boxes in pixels of that view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boxes: Option<Vec<PxBox>>,
}

/// `view_label_image`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ViewLabelImageArgs {
    /// Job id.
    pub job_id: String,
    /// Default: the next image to review.
    #[serde(default)]
    pub image_id: Option<String>,
    /// Long side of the view, pixels.
    #[serde(default = "default_view_size")]
    #[schemars(range(min = 256, max = 1568))]
    pub size: u32,
    /// First box (1-based) to list; a dense image lists 300 at a time.
    #[serde(default = "first")]
    #[schemars(range(min = 1))]
    pub from: u32,
}

/// [`DEFAULT_VIEW_SIZE`], for serde.
pub fn default_view_size() -> u32 {
    DEFAULT_VIEW_SIZE
}

fn first() -> u32 {
    1
}

fn is_first(n: &u32) -> bool {
    *n == 1
}

/// `view_label_image` result (the image itself is the result's `image` content).
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct LabelImageViewResult {
    /// Job id.
    pub job_id: String,
    /// Class names; a box's class indexes them.
    pub classes: Vec<String>,
    /// Labeled images not reviewed yet.
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
    /// The view's `[width, height]`: the frame of `boxes`.
    pub size: [u32; 2],
    /// Where it stands.
    pub status: LabelImageStatus,
    /// Who drew the boxes.
    pub source: Option<LabelSource>,
    /// Checked by a person or an agent.
    pub reviewed: bool,
    /// Why its last agent call failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// The revision of its boxes: `set_label_boxes`' and `review_label_images`' `expect`.
    pub revision: String,
    /// Its boxes.
    pub num_boxes: u32,
    /// The number of `boxes[0]` (1 unless the view was asked `from` another).
    #[serde(default = "first", skip_serializing_if = "is_first")]
    pub from: u32,
    /// `[class, x1, y1, x2, y2]` in pixels of `size`; box `from + i` is `boxes[i]`.
    pub boxes: Vec<PxBox>,
}

/// `set_label_boxes`: either `boxes`, or edits of the current boxes by their view number.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetLabelBoxesArgs {
    /// Job id.
    pub job_id: String,
    /// Image id.
    pub image_id: String,
    /// The `revision` the edit was made on (required).
    #[serde(default)]
    pub expect: Option<String>,
    /// The `size` of the view the coordinates were read on; required with boxes or add.
    #[serde(default)]
    pub size: Option<[u32; 2]>,
    /// All the image's boxes (`[]`: none).
    #[serde(default)]
    pub boxes: Option<Vec<PxBox>>,
    /// `[number, class]`: give box `number` another class.
    #[serde(default)]
    pub relabel: Vec<[u32; 2]>,
    /// Numbers of boxes to delete.
    #[serde(default)]
    pub remove: Vec<u32>,
    /// Boxes to append.
    #[serde(default)]
    pub add: Vec<PxBox>,
}

/// `review_label_images`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReviewLabelImagesArgs {
    /// Job id.
    pub job_id: String,
    /// Labeled images whose boxes are right.
    #[schemars(length(min = 1, max = 500))]
    pub image_ids: Vec<String>,
    /// `false` takes the mark back.
    #[serde(default = "yes")]
    pub reviewed: bool,
    /// Image id → the `revision` it was checked on.
    #[serde(default)]
    pub expect: std::collections::BTreeMap<String, String>,
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
    /// Every image that could not be marked.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub failed_total: u64,
    /// Labeled images not reviewed yet; `null` when it could not be counted after the change.
    pub to_review: Option<u64>,
}

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// An image `review_label_images` could not mark.
pub type ReviewFailure = folo_api::LabelReviewFailure;

/// `export_label_dataset`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportLabelDatasetArgs {
    /// Job id.
    pub job_id: String,
    /// Default: the job's name.
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
