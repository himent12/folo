//! State held by the Agent Labeling screens, per server (cleared on a source switch).

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use folo_ui::{Bounds, Entity, Pixels, ScrollHandle};

use crate::api::*;
use crate::input::TextInput;
use crate::store::Load;

/// Status filter on the Labeling jobs list.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LabelJobFilter {
    #[default]
    All,
    Active,
    Failed,
    Idle,
}

/// Image filter inside a labeling job's review workspace.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LabelImageFilter {
    #[default]
    All,
    ToReview,
    Reviewed,
    Pending,
    Failed,
}

/// Inline form to create a new labeling job.
pub struct NewLabelJobForm {
    pub seq: u64,
    pub name: Entity<TextInput>,
    pub classes: Entity<TextInput>,
    pub path: Entity<TextInput>,
    pub instructions: Entity<TextInput>,
    pub submitting: bool,
    pub touched: bool,
    pub error: Option<ApiError>,
}

/// Inline form to import a folder of images into an existing labeling job.
pub struct AddFolderForm {
    pub job: String,
    pub seq: u64,
    pub path: Entity<TextInput>,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// Inline form to export & register a YOLO dataset from a labeling job.
pub struct ExportDatasetForm {
    pub job: String,
    pub seq: u64,
    pub name: Entity<TextInput>,
    /// Index into `LABEL_VAL_FRACTIONS` (`0.1`, `0.2`, `0.3`; default `1` = 20%).
    pub val_idx: usize,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// A selected bounding box bound to the image and box-array revision it was picked from, so a live
/// SSE update or reload that shifts/replaces boxes can never cause Delete or `1`–`9` to mutate a
/// different box.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SelectedBox {
    pub image_id: String,
    pub index: usize,
    pub revision: String,
}

/// Live image SSE event buffered while a paginated image snapshot reload is in flight.
#[derive(Clone, Debug)]
pub enum PendingImageEvent {
    Updated(LabelImage),
    Removed(String),
}

/// Everything the Agent Labeling screens hold, per server.
pub struct LabelState {
    /// Generation bumped on `clear_label()` (backend switch or full Retry) so older in-flight
    /// callbacks cannot mutate reset state.
    pub epoch: u64,
    pub jobs: Load<Vec<LabelJob>>,
    pub agents: Load<Vec<LabelAgent>>,
    /// Per job: all images loaded for review.
    pub images: HashMap<String, Load<Vec<LabelImage>>>,
    /// Per job: log history + streamed log lines.
    pub logs: HashMap<String, Load<Vec<LogLine>>>,
    /// Active SSE stream for the currently open labeling job.
    pub events: Option<(String, LabelSubscription)>,
    pub subscribing: Option<String>,
    pub sub_for: Option<String>,
    pub list_filter: LabelJobFilter,
    pub list_stale: bool,

    // Active job review workspace
    pub img_filter: LabelImageFilter,
    /// Optional class filter (index into job.classes) on the image review workspace.
    pub class_filter: Option<u32>,
    /// Currently selected image id in the open labeling job.
    pub selected_image: Option<String>,
    /// Currently selected box on the active image, bound to its box-array revision.
    pub selected_box: Option<SelectedBox>,
    /// Currently hovered box index on the stage or box list.
    pub hover_box: Option<usize>,
    pub show_boxes: bool,
    pub show_labels: bool,
    pub zoom: f32,
    pub pan: (f32, f32),
    pub stage_view: Rc<Cell<Option<Bounds<Pixels>>>>,
    /// Whether a box/review mutation is currently in flight.
    pub mutating: bool,
    pub mutating_seq: u64,
    /// Inline error banner inside the review workspace (e.g. `409 label_boxes_changed`).
    pub review_error: Option<ApiError>,

    // Agent runner controls
    pub agent_id: Option<LabelAgentId>,
    /// Selected model index (`0` = agent default; `1..=models.len()` = explicit model).
    pub model_idx: usize,
    pub effort: LabelEffort,
    pub fast: bool,
    pub scope: LabelScope,
    pub starting: bool,
    pub starting_seq: u64,
    pub start_error: Option<ApiError>,

    // Forms & confirmations
    pub new_job: Option<NewLabelJobForm>,
    pub add_folder: Option<AddFolderForm>,
    pub export_ds: Option<ExportDatasetForm>,
    /// Two-click delete confirmation for a job id.
    pub confirm_delete_job: Option<String>,
    /// Two-click delete confirmation for `(job_id, image_id)`.
    pub confirm_delete_image: Option<(String, String)>,
    pub busy: HashSet<String>,
    pub removed: HashSet<String>,
    /// Per-job monotonic counter bumped on every live SSE `LabelEvent::Job` so stale HTTP
    /// responses (`/start`, `/stop`, list refresh) never regress a newer streamed job state.
    pub job_event_seq: HashMap<String, u64>,
    pub form_seq: u64,
    pub images_seq: u64,
    pub images_reqs: HashMap<String, u64>,
    pub pending_image_events: HashMap<String, Vec<PendingImageEvent>>,
    pub logs_seq: u64,
    pub logs_reqs: HashMap<String, u64>,
    pub pending_log_events: HashMap<String, Vec<LogLine>>,

    pub boxes_scroll: ScrollHandle,
    pub logs_scroll: ScrollHandle,
}

impl Default for LabelState {
    fn default() -> Self {
        Self {
            epoch: 0,
            jobs: Load::Idle,
            agents: Load::Idle,
            images: HashMap::new(),
            logs: HashMap::new(),
            events: None,
            subscribing: None,
            sub_for: None,
            list_filter: LabelJobFilter::All,
            list_stale: false,
            img_filter: LabelImageFilter::All,
            class_filter: None,
            selected_image: None,
            selected_box: None,
            hover_box: None,
            show_boxes: true,
            show_labels: true,
            zoom: 1.0,
            pan: (0., 0.),
            stage_view: Rc::new(Cell::new(None)),
            mutating: false,
            mutating_seq: 0,
            review_error: None,
            agent_id: None,
            model_idx: 0,
            effort: LabelEffort::Low,
            fast: false,
            scope: LabelScope::Unlabeled,
            starting: false,
            starting_seq: 0,
            start_error: None,
            new_job: None,
            add_folder: None,
            export_ds: None,
            confirm_delete_job: None,
            confirm_delete_image: None,
            busy: HashSet::new(),
            removed: HashSet::new(),
            job_event_seq: HashMap::new(),
            form_seq: 0,
            images_seq: 0,
            images_reqs: HashMap::new(),
            pending_image_events: HashMap::new(),
            logs_seq: 0,
            logs_reqs: HashMap::new(),
            pending_log_events: HashMap::new(),
            boxes_scroll: ScrollHandle::new(),
            logs_scroll: ScrollHandle::new(),
        }
    }
}
