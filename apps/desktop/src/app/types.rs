//! Small shared state types owned by `FoloApp` (filters, overlays, toasts, inspector, and drag state).

use std::sync::Arc;

use folo_ui::{Entity, Pixels, ScrollHandle, SharedString, Subscription};

use crate::api::{PredictInput, Sample};
use crate::input::TextInput;
use crate::ui::Tone;

use super::Action;

/// What the delete confirmation is about.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeleteTarget {
    /// A run that is not active: its folder, weights and models go.
    Run(String),
    /// A dataset: unregistered, its files stay (unless `delete_files`, for one the server wrote).
    Dataset(String),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunFilter {
    All,
    Active,
    Completed,
    Failed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DsTab {
    Images,
    Statistics,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleView {
    Predictions,
    Truth,
    Both,
}

/// The run log's level filter: every line, warnings, or errors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogFilter {
    All,
    Warn,
    Error,
}

pub struct Toast {
    pub id: u64,
    pub tone: Tone,
    pub title: SharedString,
    pub body: Option<SharedString>,
    /// An optional follow-up ("Show in folder").
    pub action: Option<(SharedString, Action)>,
}

/// Frozen filtered/sorted run-sample list and epoch captured when the inspector opens from a
/// run's Validation samples panel so stepping, counter, and overlays stay aligned with the gallery
/// even across live refreshes.
#[derive(Clone, Debug)]
pub struct SampleSnapshot {
    pub epoch: u32,
    pub conf: f32,
    pub samples: Vec<Sample>,
}

/// The image inspector over a dataset grid.
pub struct Inspector {
    pub dataset: String,
    /// Index into the filtered image list.
    pub index: usize,
    pub zoom: f32,
    pub pan: (f32, f32),
    pub drag_from: Option<(folo_ui::Point<Pixels>, (f32, f32))>,
    pub hover_box: Option<usize>,
    pub hover_from_stage: bool,
    /// Box zoomed onto via clicking a label/prediction row (clicking again resets to fit).
    pub focused_box: Option<usize>,
    /// Whether bounding boxes are drawn on the stage (`B` toggles).
    pub boxes: bool,
    /// Whether label chips are drawn on bounding boxes (`L` toggles).
    pub labels: bool,
    /// An image to land on once the dataset's image list is in (opened from a run's sample).
    pub want: Option<String>,
    /// The image last shown and the filter/sort/list state it was shown under (class, empty-only,
    /// filename query, image sort, loaded list length): when any of these change underneath (a page
    /// arrives under a non-default sort, a server switch resets filters, class list reloads), the
    /// inspector re-lands on the same image by id.
    pub shown: Option<(String, Option<u32>, bool, String, crate::screens::datasets::ImageSort, usize)>,
    /// Opened from a run's samples: that run and the view it was in. Images that are among its
    /// samples show the epoch's predictions over the (dashed) labels, as the run page draws them.
    pub sample: Option<(String, SampleView)>,
    /// Frozen filtered & sorted sample snapshot captured when opened from the samples panel.
    pub sample_snapshot: Option<SampleSnapshot>,
}

pub struct PlayImage {
    pub name: SharedString,
    pub input: PredictInput,
    /// A dataset image's file as fetched (none in the demo, whose images are pixels): local runs
    /// predict on it ([`PlayImage::predict_input`]).
    pub encoded: Option<Arc<Vec<u8>>>,
    pub image: Arc<folo_ui::RenderImage>,
    pub width: u32,
    pub height: u32,
    /// When loaded from a Playground batch, identifies `(batch_id, item_index)` so predictions and
    /// cached results never cross-contaminate same-basename files or failed stage loads.
    pub batch_ref: Option<(u64, usize)>,
}

pub struct Palette {
    pub input: Entity<TextInput>,
    pub query: String,
    pub selected: usize,
    /// The rows as last drawn (group, title): when the list changes under the highlight (a run or
    /// dataset deleted), the highlight stays on its item instead of on its old place.
    pub shown: Vec<(&'static str, String)>,
    pub scroll: ScrollHandle,
    pub(super) _sub: Subscription,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ServerMode {
    Mock,
    Local,
}

/// Slider being dragged: its id and value range, mapped through its last laid-out bounds.
pub struct Drag {
    pub id: &'static str,
}

/// Per-class box counts of one split, keyed by (dataset, split, image count).
pub type SplitCounts = ((String, crate::api::Split, usize, usize, usize), Vec<u64>);
