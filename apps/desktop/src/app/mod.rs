//! The root view: window chrome, sidebar navigation, routing, data fetching, the live event pump,
//! keyboard actions and the overlay layers (palette, dialogs, inspector, toasts).

mod actions;
#[cfg(test)]
mod bench_compare_tests;
#[cfg(test)]
mod class_filter_tests;
mod connection;
#[cfg(test)]
mod connection_tests;
mod data;
#[cfg(test)]
mod dataset_paging_tests;
#[cfg(test)]
mod dataset_tests;
mod deleted;
mod descriptions;
mod drag;
mod focus;
mod forms;
#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod label_editor_tests;
#[cfg(test)]
mod label_tests;
#[cfg(test)]
mod list_filter_tests;
#[cfg(test)]
mod list_tests;
mod lists;
mod navigation;
mod new_run;
#[cfg(test)]
mod overlay_press_tests;
mod overlay_state;
mod page;
mod playground;
#[cfg(test)]
mod playground_ab_tests;
mod playground_batch;
#[cfg(test)]
mod playground_batch_tests;
mod playground_compare;
#[cfg(test)]
mod playground_tests;
#[cfg(test)]
mod playtest_ux_tests;
mod preferences;
mod previews;
mod render;
#[cfg(test)]
mod revision_tests;
#[cfg(test)]
mod run_charts_tests;
#[cfg(test)]
mod run_compare_tests;
#[cfg(test)]
mod run_control_tests;
#[cfg(test)]
mod run_page_tests;
#[cfg(test)]
mod select_driving;
#[cfg(test)]
mod select_tests;
mod sidebar;
#[cfg(test)]
mod sprint14_tests;
#[cfg(test)]
mod sprint16_tests;
mod start_from;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
mod text_input_tests;
#[cfg(test)]
mod theme_tests;
mod titlebar;
#[cfg(test)]
mod unavailable_tests;

use std::{cell::RefCell, collections::HashMap, rc::Rc, sync::Arc};

use folo_ui::{
    App, AppContext, Bounds, Context, Entity, FocusHandle, KeyBinding, Pixels, ScrollHandle, SharedString,
    Subscription, Task, Window, actions, px,
};

use crate::api::*;
use crate::input::TextInput;
use crate::store::*;
use crate::theme::{Mode, theme};
use crate::ui::*;

pub(crate) use actions::picked as picked_path;
pub(crate) use connection::persistable_server_url;
pub use connection::{Connector, Reason};
pub use focus::focus_ring;
pub(crate) use focus::reveal_bounds_in_scroll;
pub use forms::{
    ARCHS, BATCHES, IMGSZ, NewRunForm, SYNTH_CLASSES, SYNTH_IMGSZ, SynthForm, device_label, format_lr,
    gpu_index, retry_batch, synth_errors,
};
pub use navigation::{Route, Section};
pub use playground::{IMAGE_FORMATS, PLAY_CONF_FLOOR};
pub(crate) use playground_compare::{export_default_dir, picked_save};
pub use preferences::{follow_system_appearance, is_dark};
pub use sidebar::logo;

actions!(
    folo,
    [
        OpenPalette,
        NewRunAction,
        GoOverview,
        GoDatasets,
        GoRuns,
        GoModels,
        GoBenchmarks,
        GoPlayground,
        GoLabeling,
        GoSettings,
        ToggleTheme,
        ToggleSidebar,
        ToggleFullscreen,
        Back,
        Forward,
        Dismiss,
        OpenImage,
        FocusNext,
        FocusPrev,
        Submit,
        NextRegion,
        PrevRegion,
    ]
);

pub fn bind_keys(cx: &mut App) {
    cx.bind_keys([
        KeyBinding::new("ctrl-k", OpenPalette, None),
        KeyBinding::new("ctrl-p", OpenPalette, None),
        KeyBinding::new("ctrl-n", NewRunAction, None),
        KeyBinding::new("ctrl-1", GoOverview, None),
        KeyBinding::new("ctrl-2", GoDatasets, None),
        KeyBinding::new("ctrl-3", GoRuns, None),
        KeyBinding::new("ctrl-4", GoModels, None),
        KeyBinding::new("ctrl-5", GoBenchmarks, None),
        KeyBinding::new("ctrl-6", GoPlayground, None),
        KeyBinding::new("ctrl-7", GoLabeling, None),
        KeyBinding::new("ctrl-,", GoSettings, None),
        KeyBinding::new("ctrl-b", ToggleSidebar, None),
        KeyBinding::new("ctrl-shift-l", ToggleTheme, None),
        KeyBinding::new("f11", ToggleFullscreen, None),
        KeyBinding::new("alt-left", Back, None),
        KeyBinding::new("alt-right", Forward, None),
        KeyBinding::new("escape", Dismiss, None),
        KeyBinding::new("ctrl-o", OpenImage, None),
        KeyBinding::new("tab", FocusNext, None),
        KeyBinding::new("shift-tab", FocusPrev, None),
        // Jump between the sidebar, the search field and the page without tabbing through each.
        KeyBinding::new("f6", NextRegion, None),
        KeyBinding::new("shift-f6", PrevRegion, None),
        KeyBinding::new("ctrl-enter", Submit, None),
    ]);
}

pub type Action = Rc<dyn Fn(&mut FoloApp, &mut Window, &mut Context<FoloApp>)>;

pub fn act(f: impl Fn(&mut FoloApp, &mut Window, &mut Context<FoloApp>) + 'static) -> Action {
    Rc::new(f)
}

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
    _sub: Subscription,
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

pub struct FoloApp {
    pub store: Store,
    pub route: Route,
    pub history: Vec<Route>,
    pub forward_history: Vec<Route>,
    /// The mode shown now (the theme's; with System, the OS's).
    pub mode: Mode,
    /// The user's theme choice: the mode (Dark, Light, System) and each mode's theme.
    pub theme_prefs: crate::theme::ThemePrefs,
    /// The themes to pick from: built-in, then custom (`<FOLO_HOME or ~/.folo>/themes`).
    pub themes: crate::theme::Catalog,
    /// Whether the OS appearance is dark now (System follows it).
    pub system_dark: bool,
    /// The custom themes folder is being read again (Reload).
    pub themes_loading: bool,
    /// The mode the window's system frame was last set to.
    frame_mode: Option<Mode>,
    pub frozen: bool,
    /// Keep every fetch pending (loading-state screenshots).
    pub hold_loading: bool,
    /// Where the window and theme are remembered on close (`window_state`); `None` for screenshots.
    pub state_path: Option<std::path::PathBuf>,
    /// How the window showed in the last frame drawn (none are drawn while it is minimized).
    pub shown_mode: crate::window_state::WindowMode,

    pub root_focus: FocusHandle,
    pub focus: RefCell<HashMap<SharedString, FocusHandle>>,
    /// Each switch's last drawn value and when it last changed: the knob animates only after a
    /// change, not when a switch first appears (opening Settings shows it settled).
    pub switch_seen: RefCell<HashMap<String, (bool, Option<std::time::Instant>)>>,
    /// The last input came from the keyboard, so focus rings are shown.
    pub kb: bool,
    /// The pointer moved since the last key: a keyboard focus tip (an image tile's) gives way to the
    /// pointer's own tooltips instead of standing beside them.
    pub pointer_moved: bool,
    pub page_scroll: ScrollHandle,
    /// A scroll thumb being dragged: its container, the pointer's start y and the start offset.
    pub scroll_drag: Option<(ScrollHandle, Pixels, Pixels)>,
    pub play_side_scroll: ScrollHandle,
    pub dialog_scroll: ScrollHandle,

    // Datasets
    /// Format filter on the Datasets index (`None` = All).
    pub ds_format_filter: Option<DatasetFormat>,
    /// Live search input on the Datasets index (matches name, path, format, or id).
    pub ds_search: Entity<TextInput>,
    /// Sort column and direction on the Datasets index.
    pub ds_sort: crate::screens::datasets::DatasetSort,
    pub ds_sort_desc: bool,
    pub ds_tab: DsTab,
    pub ds_split: Option<Split>,
    /// Live filename/path search input on a dataset's Images tab, plus its cached text.
    pub ds_img_search: Entity<TextInput>,
    pub ds_img_query: String,
    /// Sort order on a dataset's Images tab.
    pub ds_img_sort: crate::screens::datasets::ImageSort,
    /// Dataset id that `ds_img_search` currently belongs to (clears when switching datasets).
    pub ds_img_dataset: Option<String>,
    /// The class filter and the dataset it was set on: a class index means another class in
    /// another dataset, or none, so it applies only there (read it through `class_filter`).
    pub ds_class: Option<(String, u32)>,
    /// Show only images without labels (from the Empty images tile), on the dataset it was set on
    /// (read it through `empty_only`): it and the class filter never both apply to one dataset.
    /// Both go at a change of data source (`set_backend`).
    pub ds_empty_of: Option<String>,
    pub ds_boxes: bool,
    /// Grid tiles crop to fill the cell instead of showing the whole image.
    pub ds_fill: bool,
    /// Per-class box counts of one split, keyed by (dataset, split, image count).
    pub ds_split_counts: RefCell<Option<SplitCounts>>,
    /// Datasets whose image list is still arriving, a page at a time, with the generation of the
    /// load (a later load replaces it; its pages are dropped).
    pub images_more: HashMap<String, u64>,
    /// Datasets whose later-page image fetch stopped on a permanent error or exhausted retries,
    /// surfaced inline with a Retry action while keeping loaded pages visible.
    pub failed_pages: HashMap<String, data::FailedImagePage>,
    pub images_seq: u64,
    /// Each card preview's latest read (a number from `images_seq`): an older one is dropped.
    pub preview_reads: HashMap<String, u64>,
    /// Datasets read before, read again (in place) the next time a card or their page shows them:
    /// a folder moved away or put back since says so ([`FoloApp::ensure_dataset`]).
    pub ds_recheck: std::collections::HashSet<String>,
    /// Each dataset's latest read (a number from `ds_read_seq`): an older read answering after a
    /// newer one (two Retries, two visits) is dropped, so the newer answer stands.
    pub ds_reads: HashMap<String, u64>,
    pub ds_read_seq: u64,
    /// Each dataset's issue check on its way (a number from `issue_read_seq`): only its answer is
    /// kept, so a check of files since changed never comes back ([`FoloApp::ensure_issues`]).
    pub issue_reads: HashMap<String, u64>,
    pub issue_read_seq: u64,
    pub ds_tile: f32,
    /// The grid tile under the pointer (it reveals every label, as a focused one does).
    pub ds_hover_tile: Option<String>,
    /// Filtered and sorted image indices cached for the dataset grid.
    pub ds_filter_cache: RefCell<Option<crate::screens::datasets::DatasetFilterCache>>,
    /// Dataset Detail verified Curate / Convert CLI command builder state.
    pub ds_cli: crate::screens::datasets::DatasetCliState,
    pub grid_scroll: folo_ui::UniformListScrollHandle,
    pub ds_scroll_anchor: Option<data::DatasetScrollAnchor>,
    pub grid_width: Rc<std::cell::Cell<Pixels>>,
    pub inspector: Option<Inspector>,
    /// Focus target of the inspector so its keys (arrows, +/-, 0, Esc) work as soon as it opens.
    pub inspector_focus: FocusHandle,
    /// Focus to restore when the palette, a dialog or the stop confirmation closes.
    overlay_return: Option<overlay_state::OverlayReturn>,
    /// Restored focus handle to verify once the closing frame prepaints (falls back to `root_focus` if unmounted).
    pub(crate) verify_focus: Option<FocusHandle>,
    /// Where focus was before the inspector opened, restored on close.
    inspector_return: Option<FocusHandle>,
    pub hist_hover: HashMap<&'static str, f32>,

    // Runs
    pub run_filter: RunFilter,
    /// Architecture filter on the Runs list (`None` = All).
    pub run_arch_filter: Option<Arch>,
    /// Live search input on the Runs list (matches name, dataset, architecture, status, or id).
    pub run_search: Entity<TextInput>,
    pub chart_hover: Option<f32>,
    /// The run chart the cursor is on (its id, "run-map", "run-loss", ...): the pointer's, or the
    /// one ↑ / ↓ picked; only that chart draws `chart_hover`'s cursor and reads its legend there.
    /// None until either picks one (a cursor stepped with ← / → then shows on the first chart).
    pub chart_hover_on: Option<&'static str>,
    /// The run charts' x axis: Some(true) fits it to the epochs trained ("So far"), Some(false)
    /// shows the planned run ("All N"); None takes the default (planned while active, else trained).
    pub run_x_fit: Option<bool>,
    pub sample_view: SampleView,
    /// The validation samples' minimum confidence for a prediction to be drawn.
    pub sample_conf: f32,
    /// Error-explorer filter pill and sort order on a run's Validation samples panel.
    pub sample_error_filter: crate::screens::runs::SampleErrorFilter,
    pub sample_sort: crate::screens::runs::SampleSort,
    pub log_filter: LogFilter,
    /// Live search input for a run's Logs panel.
    pub log_search: Entity<TextInput>,
    /// Cached filtered log indices and level counts for the active run's Logs panel.
    pub(crate) log_cache: crate::screens::runs::LogFilterCache,
    pub loss_parts: [bool; 4],
    pub run_chart_opts: crate::screens::runs::RunChartOpts,
    /// Multi-run comparison state (pinned runs, chart toggles, hyperparameter diff).
    pub run_compare: crate::screens::runs::RunCompareState,
    pub log_scroll: ScrollHandle,
    /// The run page's Per class list, scrolled inside its panel.
    pub per_class_scroll: ScrollHandle,
    /// The model page's class table, scrolled inside the precision-recall panel.
    pub pc_scroll: ScrollHandle,
    /// Log lines on screen at the last render (new ones scroll into view while following).
    pub log_seen: u64,
    /// The log follows its tail. Only a scroll changes this (the offset moved); a reflow that
    /// grows the content without one keeps following.
    pub log_follow: bool,
    /// The log's scroll offset at the last render, to tell a scroll from a reflow.
    pub log_last_y: Pixels,
    /// The run whose log that state belongs to (another run's log starts following).
    pub log_run: Option<String>,
    pub sample_epoch: Option<u32>,

    // Models
    /// Architecture filter on the Models list (`None` = All).
    pub models_arch_filter: Option<Arch>,
    /// Live search input on the Models list (matches name, class, dataset, source run, arch, or id).
    pub model_search: Entity<TextInput>,
    /// Live search input on the Benchmarks list (matches name, source, leader, or id).
    pub bench_search: Entity<TextInput>,
    /// Live search input on the Agent Labeling job list (matches name, class, source, or id).
    pub label_search: Entity<TextInput>,
    /// Models whose full class list is expanded in the Details panel.
    pub model_classes_expanded: std::collections::HashSet<String>,
    pub export_open: bool,
    /// In-flight model exports `(model_id, format)` so triggers show progress and block duplicates.
    pub exporting: std::collections::HashSet<(String, &'static str)>,
    /// Model Detail interactive Export & Inference CLI command builder state.
    pub model_cli: crate::screens::models::ModelCliState,
    pub pr_hover: Option<f32>,
    /// The class isolated on the PR chart (hovered or focused in the per-class table).
    pub pr_focus: Option<u32>,
    /// The class pinned on the PR chart (Enter/Space or a click); hover and focus override it.
    pub pr_pin: Option<u32>,
    /// Active sort on a model's per-class table: `(column_key, descending)`; `None` keeps class order.
    pub class_table_sort: Option<(&'static str, bool)>,
    /// Highlighted option while the Export menu is open.
    pub export_hl: usize,
    /// The Models table's sort inside each evaluation group: column key and descending; None keeps
    /// the server's order.
    pub models_sort: Option<(&'static str, bool)>,
    /// The Runs table's sort: column key and descending.
    pub runs_sort: Option<(&'static str, bool)>,
    pub confusion_norm: bool,
    /// The confusion-matrix cell (row, column) under the pointer or the keyboard cursor.
    pub cm_cell: Option<(usize, usize)>,

    // Playground
    /// The model Playground runs; always one of the loaded list once it is in
    /// ([`FoloApp::reconcile_play_model`]).
    pub play_model: Option<String>,
    /// The run that made `play_model`: its delete drops the model even while the list is loading.
    play_model_run: Option<String>,
    /// The name of the model Playground lost (deleted), said once the next one is picked.
    play_model_gone: Option<String>,
    pub play_conf: f32,
    /// The confidence the current predictions were fetched down to: a threshold below it refetches.
    pub play_fetched_floor: f32,
    /// Show detection boxes on the Playground stage canvas.
    pub play_boxes: bool,
    /// Show class + score label chips on Playground stage boxes (effective while `play_boxes` is on).
    pub play_labels: bool,
    /// Optional class filter on the Playground detections (`None` = All classes).
    pub play_class: Option<u32>,
    /// Short windows fold the validation-image picker to one line; this opens it.
    pub play_picker_open: bool,
    pub play_iou: f32,
    /// Last committed NMS IoU value; used so a no-op slider click doesn't invalidate completed batch work.
    pub(crate) play_iou_committed: f32,
    /// Where Playground runs the model: on the server (`None`), or in this process.
    pub play_target: Option<crate::local::Target>,
    /// The Playground's line of predictions ([`crate::local::Ticket`]): a newer one skips the
    /// local runs still queued.
    play_line: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// This machine's GPU for local inference (its adapter name), probed once off the UI thread.
    pub local_gpu: Load<String>,
    pub play_image: Option<PlayImage>,
    pub play_pred: Load<PredictResponse>,
    pub play_hover: Option<usize>,
    pub(crate) play_hover_from_stage: bool,
    pub play_drop_hover: bool,
    /// Stage zoom (1 = fit) and pan in px, like the inspector.
    pub play_zoom: f32,
    pub play_pan: (f32, f32),
    pub play_drag: Option<(folo_ui::Point<Pixels>, (f32, f32))>,
    /// Generation of the latest image load request (older completions are dropped).
    play_load_seq: u64,
    /// The latest image load while it is on its way: `Some((dataset, image))` for a dataset's image,
    /// `None` for a file ([`FoloApp::forget_play_dataset`] asks again for one whose files changed).
    play_loading: Option<Option<(String, String)>>,
    /// The ticket of the person's latest action that sets the Playground's image or model (an Open
    /// image or drop accepted, a dataset's image, a model picked: [`playground::OpenRequest`]).
    play_open_seq: u64,
    /// Generation of the image on stage; predictions are tagged with it.
    play_image_gen: u64,
    /// Generation of the latest prediction request.
    play_pred_seq: u64,
    /// The latest prediction hasn't answered: shown results are the previous ones (an IoU change
    /// keeps them up meanwhile).
    pub play_pred_pending: bool,
    /// The shown result is another device's, kept (dimmed) while the one now picked runs.
    pub play_pred_dim: bool,
    /// Image generation and model the current `play_pred` belongs to.
    play_pred_for: Option<(u64, String)>,
    /// Playground Model A/B comparison state (Model B selection, prediction, hover, and tabs).
    pub play_compare: crate::screens::playground::PlayCompareState,
    /// Playground multi-image / folder batch queue, filmstrip navigation, and Run All state.
    pub play_batch: crate::screens::playground::PlayBatchState,

    // Settings
    pub server_mode: ServerMode,
    pub server_url: Entity<TextInput>,
    /// Explicitly configured Server URL remembered in `desktop.json` (never clobbered by CLI/env overrides or ephemeral auto-spawned ports).
    pub saved_server_url: Option<String>,
    pub spawn_server: bool,
    /// A local `folo serve` this app started (stopped when the app quits).
    server_proc: Option<crate::api::http::ServerProcess>,
    /// Startup / switch in progress: what the app is doing ("Starting folo serve…").
    pub connecting: Option<SharedString>,
    /// How servers are probed, started and talked to (fakes in tests).
    pub connector: Connector,
    /// The connected server is the local one (started here, or found through the data dir): if it
    /// stops answering, it is found or started again (`connection_lost`).
    pub local_url: Option<String>,
    /// The data dir the local server serves (kept across switches: a server found later that serves
    /// it is the local one again, `note_data_dir`).
    pub local_dir: Option<String>,
    /// The address the installed HTTP backend talks to.
    pub connected_url: Option<String>,
    /// Failed recoveries in a row, and the refreshes to let pass before the next attempt.
    pub recover_tries: u32,
    pub recover_wait: u32,
    /// Why the local server couldn't be started, while nothing answers.
    pub start_error: Option<SharedString>,
    /// The connected server has answered at least once (a lost connection, not one never made).
    pub reached: bool,
    /// A `/system` request is out: the 2 s refresh asks no more until it answers (`refresh_system`).
    pub system_out: bool,
    /// The last `/system` request's number: an answer to an earlier one (asked before a Retry) is
    /// dropped.
    pub system_seq: u64,
    pub reduce_motion: bool,
    /// User-toggled sidebar collapse (`Ctrl+B`), remembered in `desktop.json`.
    pub sidebar_collapsed: bool,

    // Overlays
    pub palette: Option<Palette>,
    pub new_run: Option<NewRunForm>,
    /// A New run form closed without starting (Esc, the scrim, Cancel): the next plain New run
    /// reopens it as it was, so a stray click doesn't lose a configured run.
    pub new_run_draft: Option<NewRunForm>,
    /// The architecture, image size and epochs of the last run started from the dialog: a fresh
    /// New run opens on them (this session), not on the defaults again.
    pub new_run_last: Option<(usize, usize, f32)>,
    /// `--open new-run`: open the dialog once the dataset list is in (it needs one to start from).
    pub open_new_run_pending: bool,
    /// The Generate synthetic dataset dialog.
    pub synth: Option<SynthForm>,
    pub synth_seq: u64,
    /// Restore focus to the opener on the next render (set where no window is at hand).
    pub restore_pending: bool,
    /// The window is below 1120 px wide (16 px page gutters); set each render.
    pub narrow: bool,
    /// Last rendered display scale factor (for device-pixel progress thresholds in `drain_streams`).
    pub(crate) last_scale_factor: f32,
    /// The window's activations seen by the last render: one more (each return to the window) reads
    /// the dataset shown, or inspected, again.
    pub window_activations: u64,
    pub confirm_stop: Option<String>,
    pub confirm_delete: Option<DeleteTarget>,
    /// Closing the window would stop these runs (on the server this app started): asked first. Empty:
    /// that server didn't say (it didn't answer in time).
    pub confirm_close: Option<Vec<String>>,
    /// The app's own server is being asked which runs closing would stop.
    pub close_checking: bool,
    /// Remove a dataset with its files (`?files=true`): asked in its confirmation, off each time
    /// it opens, offered only for a dataset the server wrote.
    pub delete_files: bool,
    /// Keeps the rest of the click that opened an overlay (a double-click's second press) from
    /// dismissing it or pressing what opened under the pointer.
    pub overlay_presses: crate::overlays::OverlayPresses,
    /// The server's estimate for the New run form's config, keyed by that config (servers with
    /// `run_estimate`); the last good answer stands while a changed config is out.
    pub nr_estimate: Option<(String, Load<crate::api::RunEstimate>)>,
    pub nr_estimate_last: Option<crate::api::RunEstimate>,
    /// Settings' device throughput table (`GET /api/system/throughput`).
    pub throughput: Load<crate::api::ThroughputTable>,
    /// Cloud connections, instances and offers, and their dialogs (servers with `cloud`).
    pub cloud: crate::cloud::CloudState,
    /// Vision benchmarks: lists, leaderboards, results, the open benchmark's stream (`vision_bench`).
    pub bench: crate::bench::BenchState,
    /// Agent labeling: jobs, images, review controls, and the open job's live stream (`agent_labeling`).
    pub label: crate::label::LabelState,
    /// A page to open on the next render (set where no window is at hand, e.g. after a request).
    pub route_pending: Option<Route>,
    /// Runs, models and datasets deleted lately, by id, with their names: a link to one left behind
    /// (an earlier toast's Open, the palette) opens its list instead, saying so.
    pub deleted: deleted::Deleted,
    /// Which half of it shows: training (0) or inference (1); reference rows too or measured only.
    pub tp_view: usize,
    pub tp_all: bool,
    /// A "Measure now" probe is out.
    pub bench_busy: bool,
    /// What the last probe said: its result, or why the server declined (409 `run_active`).
    pub bench_note: Option<(Tone, String)>,
    /// Bumped on every new estimate key (and on an invalid entry): only the latest settle timer asks.
    pub nr_estimate_gen: u64,
    /// Frames left to keep the page scrolled to its end (a review scene of a page's bottom): the
    /// page grows as its data lays out, so the end is pinned for a while, not once.
    pub scroll_end_pending: u8,
    /// Frames left in which the focused control scrolls into the page's view once laid out (a
    /// scenario that opens on keyboard focus below the fold).
    pub reveal_pending: u8,
    /// The keys of the control that has keyboard focus, set while the page renders and shown in a
    /// strip under the page (never over its content, and in view at any window size).
    pub key_hint: RefCell<Option<SharedString>>,
    pub toasts: Vec<Toast>,
    /// The toast under the pointer (its expiry waits).
    pub toast_hover: Option<u64>,
    /// Older toasts shown instead of folded.
    pub toasts_expanded: bool,
    /// The toast whose action or close button has keyboard focus (its expiry waits too).
    pub toast_focus: Option<u64>,
    /// Keyboard focus is on the stack's own controls (Show earlier, Dismiss all): no toast expires.
    pub toast_stack_focus: bool,
    toast_seq: u64,
    pub menu_open: Option<&'static str>,
    /// Highlighted option in the open select (committed with Enter).
    pub menu_hl: usize,
    pub menu_typeahead: crate::controls::SelectTypeahead,
    pub menu_scroll: ScrollHandle,
    /// Bounds of the open select's trigger, so a click on it toggles rather than counting as outside.
    pub menu_trigger: Rc<std::cell::Cell<Option<folo_ui::Bounds<Pixels>>>>,
    /// The confusion matrix's column area as last laid out (its headings fit names to it).
    pub cm_cols: Rc<std::cell::Cell<Option<Pixels>>>,
    /// The inspector's and the Playground's image views as last laid out (for pan clamping).
    pub insp_view: Rc<std::cell::Cell<Option<folo_ui::Bounds<Pixels>>>>,
    /// The inspector's label list (focus reveal scrolls it).
    pub insp_list_scroll: ScrollHandle,
    pub play_view: Rc<std::cell::Cell<Option<folo_ui::Bounds<Pixels>>>>,

    pub drag: Option<Drag>,
    pub slider_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    /// Each slider's value range as last rendered (a range can widen to hold a typed value), so
    /// drags and keys map a position back to the value the slider shows.
    pub slider_ranges: RefCell<HashMap<&'static str, (f32, f32)>>,

    subscriptions: HashMap<String, crate::api::Subscription>,
    /// Runs whose event stream is being opened (off the UI thread).
    subscribing: std::collections::HashSet<String>,
    /// Runs whose stream dropped mid-run, the events since not yet read back (a stream opened again
    /// reads them; a run that ended meanwhile gets its end handled once the list says so).
    stream_lost: std::collections::HashSet<String>,
    /// Ended runs whose final history could not be read over the partial one shown: read again at
    /// each run-list refresh until it is in.
    history_retry: std::collections::HashSet<String>,
    /// Bumped each time the server answers again after it was unreachable: a request sent before
    /// that which then fails as unreachable is a late failure, not a new loss.
    reconnects: u64,
    /// Per run: which subscription attempt is current (a resume supersedes one in flight).
    sub_attempt: HashMap<String, u64>,
    /// Bumped on every data-source switch; async results from an older source are discarded.
    backend_gen: u64,
    /// Bumped on every connection attempt; only the latest one may install its result.
    connect_seq: u64,
    /// The order of the dataset, model and run lists' answers: older ones are dropped.
    list_tickets: lists::ListTickets,
    /// The latest request per (run, "metrics" | "logs"); older responses are dropped.
    history_tickets: HashMap<(String, &'static str), u64>,
    history_seq: u64,
    _tasks: Vec<Task<()>>,
    _subs: Vec<Subscription>,
}

impl FoloApp {
    pub fn new(api: Arc<dyn Backend>, frozen: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let calm = crate::ui::os_prefers_reduced_motion();
        crate::ui::set_reduce_motion(calm);
        let root_focus = cx.focus_handle();
        window.focus(&root_focus);
        let server_url = cx.new(|cx| {
            let mut i = TextInput::new("http://127.0.0.1:7878", cx);
            i.set_text("http://127.0.0.1:7878", cx);
            i
        });
        let ds_search = cx.new(|cx| TextInput::new("Filter by name or path", cx));
        let ds_search_sub = cx.subscribe(&ds_search, |_, _, _: &crate::input::InputEvent, cx| cx.notify());
        let ds_img_search = cx.new(|cx| TextInput::new("Search files", cx));
        let ds_img_search_sub = cx.subscribe(&ds_img_search, |this, _, ev: &crate::input::InputEvent, cx| {
            let crate::input::InputEvent::Changed(text) = ev;
            if let Route::Dataset(id) = &this.route {
                this.ds_img_dataset = Some(id.clone());
            }
            this.ds_img_query = text.to_string();
            this.ds_hover_tile = None;
            this.ds_scroll_anchor = None;
            this.grid_scroll.scroll_to_item(0, folo_ui::ScrollStrategy::Top);
            cx.notify();
        });
        let run_search = cx.new(|cx| TextInput::new("Filter by name, dataset or arch", cx));
        let run_search_sub = cx.subscribe(&run_search, |_, _, _: &crate::input::InputEvent, cx| cx.notify());
        let model_search = cx.new(|cx| TextInput::new("Filter by name, class or arch", cx));
        let model_search_sub =
            cx.subscribe(&model_search, |_, _, _: &crate::input::InputEvent, cx| cx.notify());
        let bench_search = cx.new(|cx| TextInput::new("Search benchmarks", cx));
        let bench_search_sub =
            cx.subscribe(&bench_search, |_, _, _: &crate::input::InputEvent, cx| cx.notify());
        let label_search = cx.new(|cx| TextInput::new("Filter by name or class", cx));
        let label_search_sub =
            cx.subscribe(&label_search, |_, _, _: &crate::input::InputEvent, cx| cx.notify());
        let log_search = cx.new(|cx| TextInput::new("Search logs", cx));
        let log_search_sub = cx.subscribe(&log_search, |this, _, _: &crate::input::InputEvent, cx| {
            this.log_follow = true;
            this.log_scroll.scroll_to_bottom();
            cx.notify();
        });
        let mut this = Self {
            store: Store::new(api),
            route: Route::Overview,
            history: Vec::new(),
            forward_history: Vec::new(),
            mode: theme(cx).mode,
            theme_prefs: crate::theme::ThemePrefs::default(),
            themes: crate::theme::Catalog::builtin(),
            system_dark: preferences::is_dark(window.appearance()),
            themes_loading: false,
            frame_mode: None,
            frozen,
            hold_loading: false,
            state_path: None,
            shown_mode: crate::window_state::WindowMode::Windowed,
            root_focus,
            focus: RefCell::new(HashMap::new()),
            switch_seen: RefCell::new(HashMap::new()),
            kb: false,
            pointer_moved: false,
            page_scroll: ScrollHandle::new(),
            scroll_drag: None,
            play_side_scroll: ScrollHandle::new(),
            dialog_scroll: ScrollHandle::new(),
            ds_format_filter: None,
            ds_search,
            ds_sort: crate::screens::datasets::DatasetSort::Name,
            ds_sort_desc: false,
            ds_tab: DsTab::Images,
            ds_split: None,
            ds_img_search,
            ds_img_query: String::new(),
            ds_img_sort: crate::screens::datasets::ImageSort::Default,
            ds_img_dataset: None,
            ds_class: None,
            ds_empty_of: None,
            ds_boxes: true,
            // Whole images by default: a crop can hide the very boxes a dataset review is for.
            ds_fill: false,
            ds_split_counts: RefCell::new(None),
            images_more: HashMap::new(),
            failed_pages: HashMap::new(),
            images_seq: 0,
            preview_reads: HashMap::new(),
            ds_recheck: Default::default(),
            ds_reads: HashMap::new(),
            ds_read_seq: 0,
            issue_reads: HashMap::new(),
            issue_read_seq: 0,
            ds_tile: 168.,
            ds_hover_tile: None,
            ds_filter_cache: RefCell::new(None),
            ds_cli: Default::default(),
            grid_scroll: folo_ui::UniformListScrollHandle::new(),
            ds_scroll_anchor: None,
            grid_width: Rc::new(std::cell::Cell::new(px(900.))),
            inspector: None,
            inspector_focus: cx.focus_handle(),
            overlay_return: None,
            verify_focus: None,
            inspector_return: None,
            hist_hover: HashMap::new(),
            run_filter: RunFilter::All,
            run_arch_filter: None,
            run_search,
            chart_hover: None,
            chart_hover_on: None,
            run_x_fit: None,
            sample_view: SampleView::Both,
            // Every prediction the server keeps for a sample (docs/API.md: score >= 0.25), the slider's
            // floor, as Studio starts (it started at 0.3 here, so Desktop drew fewer boxes than Studio).
            sample_conf: crate::screens::runs::SAMPLE_CONF.0,
            sample_error_filter: crate::screens::runs::SampleErrorFilter::All,
            sample_sort: crate::screens::runs::SampleSort::Default,
            log_filter: LogFilter::All,
            log_search,
            log_cache: crate::screens::runs::LogFilterCache::default(),
            loss_parts: [true, true, true, true],
            run_chart_opts: crate::screens::runs::RunChartOpts::default(),
            run_compare: crate::screens::runs::RunCompareState::default(),
            log_scroll: ScrollHandle::new(),
            per_class_scroll: ScrollHandle::new(),
            pc_scroll: ScrollHandle::new(),
            log_seen: 0,
            log_follow: true,
            log_last_y: px(0.),
            log_run: None,
            sample_epoch: None,
            models_arch_filter: None,
            model_search,
            bench_search,
            label_search,
            model_classes_expanded: Default::default(),
            export_open: false,
            exporting: Default::default(),
            model_cli: Default::default(),
            pr_hover: None,
            pr_focus: None,
            pr_pin: None,
            class_table_sort: None,
            export_hl: 0,
            // Newest first, shown on the Created header, which is also how the server lists them.
            // Ranked within each group, best first (as Studio's list opens).
            models_sort: Some(("map", true)),
            runs_sort: Some(("created", true)),
            confusion_norm: true,
            cm_cell: None,
            play_model: None,
            play_model_run: None,
            play_model_gone: None,
            play_conf: 0.25,
            play_fetched_floor: PLAY_CONF_FLOOR,
            play_boxes: true,
            play_labels: true,
            play_class: None,
            play_picker_open: false,
            play_iou: 0.7,
            play_iou_committed: 0.7,
            play_target: None,
            play_line: Default::default(),
            local_gpu: Load::Idle,
            play_image: None,
            play_pred: Load::Idle,
            play_hover: None,
            play_hover_from_stage: false,
            play_drop_hover: false,
            play_zoom: 1.0,
            play_pan: (0., 0.),
            play_drag: None,
            play_load_seq: 0,
            play_loading: None,
            play_open_seq: 0,
            play_image_gen: 0,
            play_pred_seq: 0,
            play_pred_pending: false,
            play_pred_dim: false,
            play_pred_for: None,
            play_compare: Default::default(),
            play_batch: Default::default(),
            server_mode: ServerMode::Mock,
            server_url,
            saved_server_url: None,
            spawn_server: true,
            server_proc: None,
            connecting: None,
            connector: Connector::default(),
            local_url: None,
            local_dir: None,
            connected_url: None,
            recover_tries: 0,
            recover_wait: 0,
            start_error: None,
            reached: false,
            system_out: false,
            system_seq: 0,
            // Follow the OS animation setting until the user picks in Settings.
            reduce_motion: calm,
            sidebar_collapsed: false,
            palette: None,
            new_run: None,
            new_run_draft: None,
            new_run_last: None,
            open_new_run_pending: false,
            synth: None,
            synth_seq: 0,
            restore_pending: false,
            narrow: false,
            last_scale_factor: 1.5,
            window_activations: 0,
            confirm_stop: None,
            confirm_delete: None,
            confirm_close: None,
            close_checking: false,
            delete_files: false,
            overlay_presses: Default::default(),
            nr_estimate: None,
            nr_estimate_last: None,
            throughput: Load::Idle,
            cloud: Default::default(),
            bench: Default::default(),
            label: Default::default(),
            route_pending: None,
            deleted: Default::default(),
            tp_view: 0,
            tp_all: false,
            bench_busy: false,
            bench_note: None,
            nr_estimate_gen: 0,
            scroll_end_pending: 0,
            reveal_pending: 0,
            key_hint: RefCell::new(None),
            toasts: Vec::new(),
            toast_hover: None,
            toasts_expanded: false,
            toast_focus: None,
            toast_stack_focus: false,
            toast_seq: 0,
            menu_open: None,
            menu_hl: 0,
            menu_typeahead: Default::default(),
            menu_scroll: ScrollHandle::new(),
            menu_trigger: Rc::new(std::cell::Cell::new(None)),
            cm_cols: Rc::new(std::cell::Cell::new(None)),
            insp_view: Rc::new(std::cell::Cell::new(None)),
            insp_list_scroll: ScrollHandle::new(),
            play_view: Rc::new(std::cell::Cell::new(None)),
            drag: None,
            slider_bounds: Rc::new(RefCell::new(HashMap::new())),
            slider_ranges: RefCell::new(HashMap::new()),
            subscriptions: HashMap::new(),
            subscribing: Default::default(),
            stream_lost: Default::default(),
            history_retry: Default::default(),
            reconnects: 0,
            sub_attempt: HashMap::new(),
            backend_gen: 0,
            connect_seq: 0,
            list_tickets: Default::default(),
            history_tickets: HashMap::new(),
            history_seq: 0,
            _tasks: Vec::new(),
            _subs: vec![
                ds_search_sub,
                ds_img_search_sub,
                run_search_sub,
                model_search_sub,
                bench_search_sub,
                label_search_sub,
                log_search_sub,
            ],
        };
        if !frozen {
            this.start_pump(cx);
        }
        this._subs.push(cx.on_app_quit(|this: &mut Self, _| {
            this.server_proc = None;
            async {}
        }));
        this
    }
}
