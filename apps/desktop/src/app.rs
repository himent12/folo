//! The root view: window chrome, sidebar navigation, routing, data fetching, the live event pump,
//! keyboard actions and the overlay layers (palette, dialogs, inspector, toasts).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use gpui::point;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, App, AppContext, Bounds, Context, Entity, FocusHandle, Focusable, FontWeight,
    InteractiveElement, IntoElement, KeyBinding, MouseButton, MouseMoveEvent, MouseUpEvent, ParentElement,
    Pixels, Render, ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Subscription, Task,
    Window, WindowControlArea, actions, div, px,
};

use crate::api::*;
use crate::input::{InputEvent, TextInput};
use crate::store::*;
use crate::theme::{Mode, Theme, alpha, size as tsize, theme, ty};
use crate::ui::*;

actions!(
    folo,
    [
        OpenPalette,
        NewRunAction,
        GoOverview,
        GoDatasets,
        GoRuns,
        GoModels,
        GoPlayground,
        GoSettings,
        ToggleTheme,
        ToggleFullscreen,
        Back,
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
        KeyBinding::new("ctrl-5", GoPlayground, None),
        KeyBinding::new("ctrl-,", GoSettings, None),
        KeyBinding::new("ctrl-shift-l", ToggleTheme, None),
        KeyBinding::new("f11", ToggleFullscreen, None),
        KeyBinding::new("alt-left", Back, None),
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Route {
    Overview,
    Datasets,
    Dataset(String),
    Runs,
    Run(String),
    Models,
    Model(String),
    Playground,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Section {
    Overview,
    Datasets,
    Runs,
    Models,
    Playground,
    Settings,
}

impl Route {
    /// `overview`, `datasets`, `dataset:ID`, `runs`, `run:ID`, `models`, `model:ID`, `playground`, `settings`.
    pub fn parse(s: &str) -> Option<Route> {
        let (kind, id) = s.split_once(':').map_or((s, None), |(k, i)| (k, Some(i.to_string())));
        Some(match (kind, id) {
            ("overview", None) => Route::Overview,
            ("datasets", None) => Route::Datasets,
            ("dataset", Some(id)) => Route::Dataset(id),
            ("runs", None) => Route::Runs,
            ("run", Some(id)) => Route::Run(id),
            ("models", None) => Route::Models,
            ("model", Some(id)) => Route::Model(id),
            ("playground", None) => Route::Playground,
            ("settings", None) => Route::Settings,
            _ => return None,
        })
    }

    pub fn section(&self) -> Section {
        match self {
            Route::Overview => Section::Overview,
            Route::Datasets | Route::Dataset(_) => Section::Datasets,
            Route::Runs | Route::Run(_) => Section::Runs,
            Route::Models | Route::Model(_) => Section::Models,
            Route::Playground => Section::Playground,
            Route::Settings => Section::Settings,
        }
    }
}

pub type Action = Rc<dyn Fn(&mut FoloApp, &mut Window, &mut Context<FoloApp>)>;

/// Keyed by the image list's address too: a reload or a new source replaces the list.
pub type FilterCache = ((String, Option<Split>, Option<u32>, bool, usize, usize), Rc<Vec<usize>>);

pub fn act(f: impl Fn(&mut FoloApp, &mut Window, &mut Context<FoloApp>) + 'static) -> Action {
    Rc::new(f)
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

pub struct Toast {
    pub id: u64,
    pub tone: Tone,
    pub title: SharedString,
    pub body: Option<SharedString>,
    /// An optional follow-up ("Show in folder").
    pub action: Option<(SharedString, Action)>,
}

/// The image inspector over a dataset grid.
pub struct Inspector {
    pub dataset: String,
    /// Index into the filtered image list.
    pub index: usize,
    pub zoom: f32,
    pub pan: (f32, f32),
    pub drag_from: Option<(gpui::Point<Pixels>, (f32, f32))>,
    pub hover_box: Option<usize>,
    /// An image to land on once the dataset's image list is in (opened from a run's sample).
    pub want: Option<String>,
    /// Opened from a run's samples: that run and the view it was in. Images that are among its
    /// samples show the epoch's predictions over the (dashed) labels, as the run page draws them.
    pub sample: Option<(String, SampleView)>,
}

/// Close mosaic: a whole number of epochs, at most the run's length.
pub fn parse_close(s: &str, epochs: f32) -> Option<u32> {
    s.trim().parse::<u32>().ok().filter(|v| (*v as f32) <= epochs)
}

impl NewRunForm {
    /// Validate the close-mosaic text against the current epoch count.
    pub fn check_close(&mut self, s: &str) {
        match parse_close(s, self.epochs) {
            Some(v) => {
                self.close_mosaic = v as f32;
                self.close_invalid = false;
            }
            None => self.close_invalid = true,
        }
    }

    /// Picks optimizer `i` of [`OPTIMIZERS`] (also settles an unknown prefilled one).
    pub fn pick_optimizer(&mut self, i: usize) {
        self.optimizer = i;
        self.optimizer_unknown = false;
    }

    /// Prefills the optimizer from a run's config: an optimizer this build does not know (a newer server's)
    /// selects nothing and keeps the form from starting until one is picked, instead of becoming Auto.
    pub fn prefill_optimizer(&mut self, o: Optimizer) {
        match optimizer_index(o) {
            Some(i) => self.pick_optimizer(i),
            None => {
                self.optimizer = 0;
                self.optimizer_unknown = true;
            }
        }
    }

    /// Why the form cannot start for its optimizer, if it cannot.
    pub fn optimizer_blocks(&self) -> Option<&'static str> {
        self.optimizer_unknown
            .then_some("Pick an optimizer first: the run this was made from used an unknown one")
    }
}

/// The index of `o` in [`OPTIMIZERS`], `None` for one the picker does not offer (`Optimizer::Unknown`).
pub fn optimizer_index(o: Optimizer) -> Option<usize> {
    OPTIMIZERS.iter().position(|x| *x == o)
}

pub type ModelsOrderKey = (usize, usize, &'static str, bool);

/// Image sizes and class sets offered for synthetic datasets.
pub const SYNTH_IMGSZ: [u32; 3] = [256, 320, 640];
pub const SYNTH_CLASSES: [&[&str]; 3] = [
    &["circle", "square", "triangle"],
    &["circle", "square", "triangle", "star", "hexagon"],
    &["circle", "square", "triangle", "star", "ring", "cross", "hexagon", "heart"],
];

/// Images in a synthetic dataset: a whole number in 1..=100000 (the server's limits).
pub fn parse_images(s: &str) -> Option<u64> {
    s.trim().replace([',', '_', ' '], "").parse::<u64>().ok().filter(|v| (1..=100_000).contains(v))
}

pub struct SynthForm {
    /// Which opening of the dialog this is: a late answer only touches its own dialog.
    pub id: u64,
    pub name: Entity<TextInput>,
    pub images: Entity<TextInput>,
    pub imgsz: usize,
    pub classes: usize,
    pub submitting: bool,
    pub error: Option<String>,
}

pub struct NewRunForm {
    pub name: Entity<TextInput>,
    pub dataset: usize,
    pub arch: usize,
    pub epochs: f32,
    pub batch: usize,
    pub imgsz: usize,
    pub device: usize,
    pub optimizer: usize,
    /// Prefilled from a run whose optimizer this build does not know: no optimizer is selected, and the form
    /// does not start until one is picked ([`NewRunForm::optimizer_blocks`]).
    pub optimizer_unknown: bool,
    pub lr0: f32,
    pub mosaic: f32,
    pub mixup: f32,
    pub fliplr: f32,
    /// Epochs at the end trained without mosaic.
    pub close_mosaic: f32,
    /// Scale jitter (± fraction).
    pub scale: f32,
    /// Why some values differ from the run this form was prefilled from.
    pub note: Option<String>,
    /// The run this form was prefilled from: settings the form doesn't show (weights, seed, colour and
    /// translate jitter) carry over unchanged.
    pub base: Option<TrainConfig>,
    /// Exact entry next to the sliders; invalid text keeps the last good value and shows a hint.
    pub epochs_input: Entity<TextInput>,
    pub lr_input: Entity<TextInput>,
    pub epochs_invalid: bool,
    pub lr_invalid: bool,
    pub close_input: Entity<TextInput>,
    pub close_invalid: bool,
    /// Mixed-precision training (f16 on tensor cores). Sent only for a GPU device.
    pub amp: bool,
    /// The NMS-free end-to-end head (`head: o2o`); off is the default one-to-many head.
    pub e2e: bool,
    /// The Advanced group (head, optimizer, mixed precision, augmentation) is open.
    pub advanced: bool,
    /// Run on (servers with `cloud`): `None` trains on this machine, else on this cloud connection.
    pub run_on: Option<String>,
    /// The connection's offer to train on.
    pub offer: Option<String>,
    /// Stop after, hours (`compute.max_hours`): it follows the estimate until it is typed.
    pub max_hours: f32,
    pub hours_input: Entity<TextInput>,
    pub hours_invalid: bool,
    pub hours_edited: bool,
    /// What the server said about the compute choice, on the control it is about.
    pub compute_error: Option<(crate::cloud::ComputeField, String)>,
    /// New run from a cloud run's config: its offer, chosen once the connection's offers are in.
    pub want_offer: Option<String>,
    _subs: Vec<Subscription>,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

/// Epochs as typed: a whole number in 1..=10000 (the server's limit).
pub fn parse_epochs(s: &str) -> Option<u32> {
    s.trim().parse::<u32>().ok().filter(|v| (1..=10_000).contains(v))
}

/// Learning rate as typed, scientific notation allowed (`1e-3`): 0 < lr < 10.
/// A learning rate for its input field: three decimals in the slider's range (0.010), and as
/// many digits as it needs below it (0.0001 doesn't round to 0.000).
pub fn format_lr(v: f32) -> String {
    if v >= 0.001 { format!("{v:.3}") } else { format!("{v}") }
}

pub fn parse_lr(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0 && *v < 10.0)
}

pub const ARCHS: [Arch; 5] = Arch::ALL;
pub const BATCHES: [u32; 7] = [2, 4, 8, 16, 32, 64, 128];

/// The offered batch size closest to `v` without exceeding it (the smallest if `v` is below all).
pub fn batch_index(v: u32) -> usize {
    BATCHES.iter().rposition(|b| *b <= v).unwrap_or(0)
}

/// Batch for retrying an out-of-memory run: half, snapped to an offered size; None at the minimum.
pub fn retry_batch(v: u32) -> Option<u32> {
    (v > BATCHES[0]).then(|| BATCHES[batch_index(v / 2)])
}

/// Keep what the form can't edit from the run it was prefilled from.
pub fn carry_over(base: Option<&TrainConfig>, mut c: TrainConfig) -> TrainConfig {
    if let Some(b) = base {
        c.weights = b.weights.clone();
        c.seed = b.seed;
        c.augment.hsv = b.augment.hsv;
        c.augment.translate = b.augment.translate;
        // Not `amp` nor `head`: the form shows both (prefilled from the base).
    }
    c
}
/// The Playground's lowest default confidence: predictions are fetched down to it once, and the
/// slider filters above it locally.
pub const PLAY_CONF_FLOOR: f32 = 0.05;
pub const IMGSZ: [u32; 5] = [320, 480, 640, 800, 1024];
pub const DEVICES: [&str; 3] = ["auto", "gpu:0", "cpu"];
pub const OPTIMIZERS: [Optimizer; 4] = [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd];

pub struct PlayImage {
    pub name: SharedString,
    pub input: PredictInput,
    pub image: Arc<gpui::RenderImage>,
    pub width: u32,
    pub height: u32,
}

pub struct Palette {
    pub input: Entity<TextInput>,
    pub query: String,
    pub selected: usize,
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
    pub mode: Mode,
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
    pub page_scroll: ScrollHandle,
    /// A scroll thumb being dragged: its container, the pointer's start y and the start offset.
    pub scroll_drag: Option<(ScrollHandle, Pixels, Pixels)>,
    pub play_side_scroll: ScrollHandle,
    pub dialog_scroll: ScrollHandle,

    // Datasets
    pub ds_tab: DsTab,
    pub ds_split: Option<Split>,
    pub ds_class: Option<u32>,
    /// Show only images without labels (from the Empty images tile).
    pub ds_empty_only: bool,
    pub ds_boxes: bool,
    /// Grid tiles crop to fill the cell instead of showing the whole image.
    pub ds_fill: bool,
    /// Per-class box counts of one split, keyed by (dataset, split, image count).
    pub ds_split_counts: RefCell<Option<SplitCounts>>,
    pub ds_tile: f32,
    /// The grid tile under the pointer (it reveals every label, as a focused one does).
    pub ds_hover_tile: Option<String>,
    /// Filtered image indices keyed by (dataset, split, class, list length).
    pub ds_filter_cache: RefCell<Option<FilterCache>>,
    pub grid_scroll: gpui::UniformListScrollHandle,
    pub grid_width: Rc<std::cell::Cell<Pixels>>,
    pub inspector: Option<Inspector>,
    /// Focus target of the inspector so its keys (arrows, +/-, 0, Esc) work as soon as it opens.
    pub inspector_focus: FocusHandle,
    /// Focus to restore when the palette, a dialog or the stop confirmation closes.
    overlay_return: Option<FocusHandle>,
    /// Where focus was before the inspector opened, restored on close.
    inspector_return: Option<FocusHandle>,
    pub hist_hover: HashMap<&'static str, f32>,
    /// Grid view controls built by the image browser, placed on the tab row.
    pub ds_view_controls: Option<AnyElement>,

    // Runs
    pub run_filter: RunFilter,
    pub chart_hover: Option<f32>,
    pub sample_view: SampleView,
    pub loss_parts: [bool; 4],
    pub log_scroll: ScrollHandle,
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
    pub export_open: bool,
    pub pr_hover: Option<f32>,
    /// The class isolated on the PR chart (hovered or focused in the per-class table).
    pub pr_focus: Option<u32>,
    /// The class pinned on the PR chart (Enter/Space or a click); hover and focus override it.
    pub pr_pin: Option<u32>,
    /// Highlighted option while the Export menu is open.
    pub export_hl: usize,
    /// Models table order: column key and descending; None keeps the server's order.
    pub models_sort: Option<(&'static str, bool)>,
    /// The Runs table's sort: column key and descending.
    pub runs_sort: Option<(&'static str, bool)>,
    /// The sorted order of the models list, for (list allocation, length, column, descending).
    pub models_order: RefCell<Option<(ModelsOrderKey, Vec<usize>)>>,
    pub confusion_norm: bool,
    /// The confusion-matrix cell (row, column) under the pointer or the keyboard cursor.
    pub cm_cell: Option<(usize, usize)>,

    // Playground
    pub play_model: Option<String>,
    pub play_conf: f32,
    /// The confidence the current predictions were fetched down to: a threshold below it refetches.
    pub play_fetched_floor: f32,
    /// Short windows fold the validation-image picker to one line; this opens it.
    pub play_picker_open: bool,
    pub play_iou: f32,
    /// Where Playground runs the model: on the server (`None`), or in this process.
    pub play_target: Option<crate::local::Target>,
    /// This machine's GPU for local inference (its adapter name), probed once off the UI thread.
    pub local_gpu: Load<String>,
    pub play_image: Option<PlayImage>,
    pub play_pred: Load<PredictResponse>,
    pub play_hover: Option<usize>,
    pub play_drop_hover: bool,
    /// Stage zoom (1 = fit) and pan in px, like the inspector.
    pub play_zoom: f32,
    pub play_pan: (f32, f32),
    pub play_drag: Option<(gpui::Point<Pixels>, (f32, f32))>,
    /// Generation of the latest image load request (older completions are dropped).
    play_load_seq: u64,
    /// Generation of the image on stage; predictions are tagged with it.
    play_image_gen: u64,
    /// Generation of the latest prediction request.
    play_pred_seq: u64,
    /// The latest prediction hasn't answered: shown results are the previous ones (an IoU change
    /// keeps them up meanwhile).
    pub play_pred_pending: bool,
    /// Image generation and model the current `play_pred` belongs to.
    play_pred_for: Option<(u64, String)>,

    // Settings
    pub server_mode: ServerMode,
    pub server_url: Entity<TextInput>,
    pub spawn_server: bool,
    /// A `folo serve --mock` this app started (stopped when the app quits).
    server_proc: Option<crate::api::http::ServerProcess>,
    /// Startup / switch in progress: what the app is doing ("Starting folo serve…").
    pub connecting: Option<SharedString>,
    pub reduce_motion: bool,

    // Overlays
    pub palette: Option<Palette>,
    pub new_run: Option<NewRunForm>,
    /// A New run form closed without starting (Esc, the scrim, Cancel): the next plain New run
    /// reopens it as it was, so a stray click doesn't lose a configured run.
    pub new_run_draft: Option<NewRunForm>,
    /// The draft's dataset by id (the list can gain datasets at the front while the draft waits,
    /// so its index alone could point at another one).
    pub new_run_draft_ds: Option<String>,
    /// `--open new-run`: open the dialog once the dataset list is in (it needs one to start from).
    pub open_new_run_pending: bool,
    /// The Generate synthetic dataset dialog.
    pub synth: Option<SynthForm>,
    pub synth_seq: u64,
    /// Restore focus to the opener on the next render (set where no window is at hand).
    pub restore_pending: bool,
    /// The window is below 1120 px wide (16 px page gutters); set each render.
    pub narrow: bool,
    pub confirm_stop: Option<String>,
    /// The server's estimate for the New run form's config, keyed by that config (servers with
    /// `run_estimate`); the last good answer stands while a changed config is out.
    pub nr_estimate: Option<(String, Load<crate::api::RunEstimate>)>,
    pub nr_estimate_last: Option<crate::api::RunEstimate>,
    /// Settings' device throughput table (`GET /api/system/throughput`).
    pub throughput: Load<crate::api::ThroughputTable>,
    /// Cloud connections, instances and offers, and their dialogs (servers with `cloud`).
    pub cloud: crate::cloud::CloudState,
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
    pub menu_scroll: ScrollHandle,
    /// Bounds of the open select's trigger, so a click on it toggles rather than counting as outside.
    pub menu_trigger: Rc<std::cell::Cell<Option<gpui::Bounds<Pixels>>>>,
    /// The confusion matrix's column area as last laid out (its headings fit names to it).
    pub cm_cols: Rc<std::cell::Cell<Option<Pixels>>>,
    /// The inspector's and the Playground's image views as last laid out (for pan clamping).
    pub insp_view: Rc<std::cell::Cell<Option<gpui::Bounds<Pixels>>>>,
    /// The inspector's label list (focus reveal scrolls it).
    pub insp_list_scroll: ScrollHandle,
    pub play_view: Rc<std::cell::Cell<Option<gpui::Bounds<Pixels>>>>,

    pub drag: Option<Drag>,
    pub slider_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,
    /// Each slider's value range as last rendered (a range can widen to hold a typed value), so
    /// drags and keys map a position back to the value the slider shows.
    pub slider_ranges: RefCell<HashMap<&'static str, (f32, f32)>>,

    subscriptions: HashMap<String, crate::api::Subscription>,
    /// Runs whose event stream is being opened (off the UI thread).
    subscribing: std::collections::HashSet<String>,
    /// Per run: which subscription attempt is current (a resume supersedes one in flight).
    sub_attempt: HashMap<String, u64>,
    /// Bumped on every data-source switch; async results from an older source are discarded.
    backend_gen: u64,
    /// Bumped on every connection attempt; only the latest one may install its result.
    connect_seq: u64,
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
        let mut this = Self {
            store: Store::new(api),
            route: Route::Overview,
            history: Vec::new(),
            mode: theme(cx).mode,
            frozen,
            hold_loading: false,
            state_path: None,
            shown_mode: crate::window_state::WindowMode::Windowed,
            root_focus,
            focus: RefCell::new(HashMap::new()),
            switch_seen: RefCell::new(HashMap::new()),
            kb: false,
            page_scroll: ScrollHandle::new(),
            scroll_drag: None,
            play_side_scroll: ScrollHandle::new(),
            dialog_scroll: ScrollHandle::new(),
            ds_tab: DsTab::Images,
            ds_split: None,
            ds_class: None,
            ds_empty_only: false,
            ds_boxes: true,
            // Whole images by default: a crop can hide the very boxes a dataset review is for.
            ds_fill: false,
            ds_split_counts: RefCell::new(None),
            ds_tile: 168.,
            ds_hover_tile: None,
            ds_filter_cache: RefCell::new(None),
            grid_scroll: gpui::UniformListScrollHandle::new(),
            grid_width: Rc::new(std::cell::Cell::new(px(900.))),
            inspector: None,
            inspector_focus: cx.focus_handle(),
            overlay_return: None,
            inspector_return: None,
            hist_hover: HashMap::new(),
            ds_view_controls: None,
            run_filter: RunFilter::All,
            chart_hover: None,
            sample_view: SampleView::Predictions,
            loss_parts: [true, true, true, true],
            log_scroll: ScrollHandle::new(),
            log_seen: 0,
            log_follow: true,
            log_last_y: px(0.),
            log_run: None,
            sample_epoch: None,
            export_open: false,
            pr_hover: None,
            pr_focus: None,
            pr_pin: None,
            export_hl: 0,
            // Newest first, shown on the Created header, which is also how the server lists them.
            models_sort: Some(("created", true)),
            runs_sort: Some(("created", true)),
            models_order: RefCell::new(None),
            confusion_norm: true,
            cm_cell: None,
            play_model: None,
            play_conf: 0.25,
            play_fetched_floor: PLAY_CONF_FLOOR,
            play_picker_open: false,
            play_iou: 0.7,
            play_target: None,
            local_gpu: Load::Idle,
            play_image: None,
            play_pred: Load::Idle,
            play_hover: None,
            play_drop_hover: false,
            play_zoom: 1.0,
            play_pan: (0., 0.),
            play_drag: None,
            play_load_seq: 0,
            play_image_gen: 0,
            play_pred_seq: 0,
            play_pred_pending: false,
            play_pred_for: None,
            server_mode: ServerMode::Mock,
            server_url,
            spawn_server: true,
            server_proc: None,
            connecting: None,
            // Follow the OS animation setting until the user picks in Settings.
            reduce_motion: calm,
            palette: None,
            new_run: None,
            new_run_draft: None,
            new_run_draft_ds: None,
            open_new_run_pending: false,
            synth: None,
            synth_seq: 0,
            restore_pending: false,
            narrow: false,
            confirm_stop: None,
            nr_estimate: None,
            nr_estimate_last: None,
            throughput: Load::Idle,
            cloud: Default::default(),
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
            sub_attempt: HashMap::new(),
            backend_gen: 0,
            connect_seq: 0,
            history_tickets: HashMap::new(),
            history_seq: 0,
            _tasks: Vec::new(),
            _subs: Vec::new(),
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

    // ---------- focus ----------

    /// A stable focus handle for a control key; tab order follows paint order.
    pub fn fh(&self, key: impl Into<SharedString>, cx: &mut Context<Self>) -> FocusHandle {
        let key = key.into();
        self.focus.borrow_mut().entry(key).or_insert_with(|| cx.focus_handle().tab_stop(true)).clone()
    }

    /// Names the keys of the focused control in the strip under the page (see `key_hint`).
    pub fn hint_keys(&self, keys: impl Into<SharedString>) {
        *self.key_hint.borrow_mut() = Some(keys.into());
    }

    /// Keyboard focus is on this handle and should show a ring.
    pub fn ring(&self, fh: &FocusHandle, window: &Window) -> bool {
        self.kb && fh.is_focused(window)
    }

    pub fn focus_key(&mut self, key: &str, window: &mut Window, cx: &mut Context<Self>) {
        let fh = self.fh(key.to_string(), cx);
        window.focus(&fh);
    }

    // ---------- navigation ----------

    pub fn navigate(&mut self, route: Route, window: &mut Window, cx: &mut Context<Self>) {
        // The confusion-matrix cursor belongs to the model page it was set on.
        if matches!(route, Route::Model(_)) {
            self.cm_cell = None;
        }
        if route == self.route {
            return;
        }
        let prev = std::mem::replace(&mut self.route, route);
        self.history.push(prev);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
        self.after_navigate(window, cx);
    }

    pub fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(r) = self.history.pop() {
            self.route = r;
            self.after_navigate(window, cx);
        } else if let Some(parent) = self.parent_route() {
            self.route = parent;
            self.after_navigate(window, cx);
        }
    }

    pub fn parent_route(&self) -> Option<Route> {
        match self.route {
            Route::Dataset(_) => Some(Route::Datasets),
            Route::Run(_) => Some(Route::Runs),
            Route::Model(_) => Some(Route::Models),
            _ => None,
        }
    }

    fn after_navigate(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Settings shows the throughput table and the cloud as the server has them now (also when
        // going back).
        if self.route == Route::Settings && !self.throughput.is_loading() {
            self.throughput = Load::Idle;
        }
        if self.route == Route::Settings {
            self.reload_cloud();
        }
        self.page_scroll.set_offset(gpui::point(px(0.), px(0.)));
        self.chart_hover = None;
        self.export_open = false;
        self.pr_pin = None;
        self.pr_focus = None;
        self.menu_open = None;
        self.sample_epoch = None;
        if !self.focus_within_page(window) {
            window.focus(&self.root_focus);
        }
        cx.notify();
    }

    fn focus_within_page(&self, _window: &Window) -> bool {
        false
    }

    // ---------- data ----------

    /// Run a backend call on the background executor and apply its result on the main thread.
    pub fn fetch<T: Send + 'static>(
        &mut self,
        cx: &mut Context<Self>,
        call: impl FnOnce(&dyn Backend) -> ApiResult<T> + Send + 'static,
        apply: impl FnOnce(&mut Self, ApiResult<T>, &mut Context<Self>) + 'static,
    ) {
        if self.hold_loading {
            return;
        }
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async move { call(&*api) }).await;
            this.update(cx, |this, cx| {
                // Answered by a source we've since switched away from: not ours any more.
                if this.backend_gen != generation {
                    return;
                }
                apply(this, r, cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Fetch the model list again after it failed (nothing else retries a failed load).
    pub fn retry_models(&mut self, cx: &mut Context<Self>) {
        self.store.models = Load::Idle;
        self.ensure_core(cx);
        cx.notify();
    }

    pub fn ensure_core(&mut self, cx: &mut Context<Self>) {
        if self.store.runs.needs_fetch() {
            self.store.runs = Load::Loading;
            self.fetch(
                cx,
                |api| api.runs(),
                |this, r, cx| {
                    this.store.runs = r.map(newest_first_runs).into();
                    this.store.sync_run_list();
                    this.subscribe_active(cx);
                },
            );
        }
        if self.store.datasets.needs_fetch() {
            self.store.datasets = Load::Loading;
            self.fetch(cx, |api| api.datasets(), |this, r, _| this.store.datasets = r.into());
        }
        if self.store.models.needs_fetch() {
            self.store.models = Load::Loading;
            self.fetch(
                cx,
                |api| api.models(),
                |this, r, _| {
                    this.store.models = r
                        .map(|mut m| {
                            m.sort_by(|a, b| b.created.cmp(&a.created));
                            m
                        })
                        .into();
                    if this.play_model.is_none() {
                        this.play_model =
                            this.store.models.ready().and_then(|m| m.first()).map(|m| m.id.clone());
                    }
                },
            );
        }
        if self.store.system.needs_fetch() {
            self.store.system = Load::Loading;
            self.fetch(
                cx,
                |api| api.system(),
                |this, r, _| {
                    if let Ok(s) = &r
                        && let Some(g) = s.system.gpus.first().and_then(|g| g.util)
                    {
                        this.store.gpu_util.push_back(g);
                    }
                    let r = r.map(|s| {
                        this.store.capabilities = s.capabilities;
                        this.store.features = s.features;
                        s.system
                    });
                    this.store.system = r.into();
                },
            );
        }
        if self.store.health.needs_fetch() {
            self.store.health = Load::Loading;
            self.fetch(cx, |api| api.health(), |this, r, _| this.store.health = r.into());
        }
    }

    pub fn ensure_run_data(&mut self, id: &str, cx: &mut Context<Self>) {
        let spe = self.store.run(id).map_or(1, |r| r.steps_per_epoch);
        if self.store.run_data.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.run_data.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "metrics");
            let rid = id.to_string();
            let key = id.to_string();
            self.fetch(
                cx,
                move |api| api.metrics(&rid),
                move |this, r, _| {
                    // A reload started since: this older snapshot must not overwrite it.
                    if this.history_tickets.get(&(key.clone(), "metrics")) != Some(&ticket) {
                        return;
                    }
                    this.store.install_run_data(&key, r.map(|m| RunData::new(m, spe)));
                },
            );
        }
    }

    /// A fresh ticket for one history request of a run; earlier tickets stop counting.
    fn history_ticket(&mut self, id: &str, what: &'static str) -> u64 {
        self.history_seq += 1;
        self.history_tickets.insert((id.to_string(), what), self.history_seq);
        self.history_seq
    }

    pub fn ensure_samples(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.samples.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.samples.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "samples");
            let (rid, key, epoch) = (id.to_string(), id.to_string(), self.sample_epoch);
            self.fetch(
                cx,
                move |api| api.samples(&rid, epoch),
                move |this, r, _| {
                    if this.history_tickets.get(&(key.clone(), "samples")) != Some(&ticket) {
                        return;
                    }
                    this.store.samples.insert(key, r.into());
                },
            );
        }
    }

    pub fn ensure_logs(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.logs.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.logs.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "logs");
            let (rid, key) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.logs(&rid),
                move |this, r, _| {
                    if this.history_tickets.get(&(key.clone(), "logs")) != Some(&ticket) {
                        return;
                    }
                    this.store.install_logs(&key, r);
                    // Only the log on screen, and only while it follows (a late answer for another
                    // run mustn't move, or resume, the one being read).
                    if matches!(&this.route, Route::Run(id) if *id == key) && this.log_follow {
                        this.log_scroll.scroll_to_bottom();
                    }
                },
            );
        }
    }

    /// A dataset's issue counts, once (the Broken labels tile splits its count by severity).
    pub fn ensure_issues(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.issues.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.issues.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset_issues(&a).map(|p| p.counts),
                move |this, r, _| {
                    this.store.issues.insert(b, r.into());
                },
            );
        }
    }

    pub fn ensure_dataset(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.dataset.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.dataset.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset(&a),
                move |this, r, _| {
                    this.store.dataset.insert(b, r.into());
                },
            );
        }
        if self.store.images.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.images.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| {
                    api.images(&a, &ImageQuery { limit: 100_000, ..Default::default() }).map(|p| p.images)
                },
                move |this, r, _| {
                    this.store.images.insert(b, r.into());
                    // New list, new truth: nothing derived from the old one may survive.
                    *this.ds_filter_cache.borrow_mut() = None;
                    *this.ds_split_counts.borrow_mut() = None;
                },
            );
        }
    }

    pub fn ensure_eval(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.evals.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.evals.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.model_eval(&a),
                move |this, r, _| {
                    this.store.evals.insert(b, r.into());
                },
            );
        }
    }

    /// A decoded image, requesting it if needed.
    pub fn image(
        &mut self,
        dataset: &str,
        image: &str,
        kind: ImgKind,
        cx: &mut Context<Self>,
    ) -> Option<Arc<gpui::RenderImage>> {
        let key = ImgKey { dataset: dataset.to_string(), image: image.to_string(), kind };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => return None,
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        let thumb = (kind == ImgKind::Thumb).then_some(320);
        cx.spawn(async move |this, cx| {
            let k2 = key.clone();
            let img = cx
                .background_executor()
                .spawn(async move { api.image(&k2.dataset, &k2.image, thumb).ok().and_then(to_render_image) })
                .await;
            this.update(cx, |this, cx| {
                if this.backend_gen != generation {
                    if let Some(img) = img {
                        cx.drop_image(img, None);
                    }
                    return;
                }
                let evicted = this.store.images_px.finish(key, img);
                for e in evicted {
                    cx.drop_image(e, None);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
        None
    }

    /// Subscribe to live events of every active run.
    pub fn subscribe_active(&mut self, cx: &mut Context<Self>) {
        if self.frozen {
            return;
        }
        let ids: Vec<String> = self
            .store
            .runs
            .ready()
            .map(|runs| runs.iter().filter(|r| r.status.is_active()).map(|r| r.id.clone()).collect())
            .unwrap_or_default();
        for id in ids {
            if !self.subscriptions.contains_key(&id) {
                self.resubscribe(id, cx);
            }
        }
    }

    /// Open a run's live stream (off the UI thread), then reload whatever history of it is already
    /// loaded: events from the stream buffer while the history loads and merge into it, so nothing
    /// between an earlier snapshot (or a dropped stream) and this stream is lost. A failed attempt
    /// is retried by the periodic run-list refresh.
    fn resubscribe(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.subscribing.insert(id.clone()) {
            return;
        }
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        let attempt = self.sub_attempt.get(&id).copied().unwrap_or(0);
        cx.spawn(async move |this, cx| {
            let rid = id.clone();
            let sub = cx.background_executor().spawn(async move { api.events(&rid) }).await;
            this.update(cx, |this, cx| {
                // Superseded (a new source, or the run was resumed): the newer attempt owns it.
                if this.backend_gen != generation
                    || this.sub_attempt.get(&id).copied().unwrap_or(0) != attempt
                {
                    return;
                }
                this.subscribing.remove(&id);
                let Ok(sub) = sub else { return };
                this.subscriptions.insert(id.clone(), sub);
                let had_data = this.store.run_data.remove(&id).is_some();
                let had_logs = this.store.logs.remove(&id).is_some();
                if had_data {
                    this.ensure_run_data(&id, cx);
                }
                if had_logs {
                    this.ensure_logs(&id, cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Reload the run list in place (no loading state) and subscribe to whatever is active.
    pub fn refresh_runs(&mut self, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            |api| api.runs(),
            |this, r, cx| {
                if let Ok(runs) = r {
                    this.store.runs = Load::Ready(newest_first_runs(runs));
                    this.store.sync_run_list();
                    this.subscribe_active(cx);
                }
            },
        );
    }

    /// Drain live events ~10× a second and refresh slow-moving data every few seconds.
    fn start_pump(&mut self, cx: &mut Context<Self>) {
        let task = cx.spawn(async move |this, cx| {
            let mut tick: u64 = 0;
            loop {
                cx.background_executor().timer(Duration::from_millis(100)).await;
                tick += 1;
                let alive = this
                    .update(cx, |this, cx| {
                        let mut changed = false;
                        let mut ended = Vec::new();
                        let mut finished = Vec::new();
                        let ids: Vec<String> = this.subscriptions.keys().cloned().collect();
                        for id in ids {
                            loop {
                                let ev = match this.subscriptions.get(&id).map(|rx| rx.try_recv()) {
                                    Some(Ok(ev)) => ev,
                                    Some(Err(TryRecvError::Empty)) | None => break,
                                    Some(Err(TryRecvError::Disconnected)) => {
                                        ended.push(id.clone());
                                        break;
                                    }
                                };
                                if let RunEvent::Status(s) = &ev
                                    && matches!(s.status, RunStatus::Completed | RunStatus::Failed)
                                {
                                    finished.push((id.clone(), s.status));
                                }
                                this.store.apply(&id, ev);
                                changed = true;
                            }
                        }
                        // A stream that ended without a terminal status dropped out: the events in the
                        // gap are gone (the API replays only the status), so reload that run's history
                        // and the run list, which resubscribes if it's still training.
                        let dropped: Vec<String> = ended
                            .iter()
                            .filter(|id| !finished.iter().any(|(f, _)| f == *id))
                            .filter(|id| this.store.run(id).is_some_and(|r| r.status.is_active()))
                            .cloned()
                            .collect();
                        for id in ended {
                            this.subscriptions.remove(&id);
                        }
                        for id in dropped {
                            this.resubscribe(id, cx);
                        }
                        for (id, status) in finished {
                            let name = this.store.run(&id).map(|r| r.name.clone()).unwrap_or(id.clone());
                            if status == RunStatus::Completed {
                                this.toast(
                                    Tone::Success,
                                    format!("{name} finished"),
                                    Some("Best weights saved to Models.".into()),
                                    cx,
                                );
                                this.store.models = Load::Idle;
                            } else {
                                this.toast(Tone::Danger, format!("{name} failed"), None, cx);
                            }
                        }
                        if tick.is_multiple_of(20) {
                            this.store.system = Load::Idle;
                            this.ensure_core(cx);
                        }
                        if tick.is_multiple_of(50) && matches!(this.store.runs, Load::Ready(_)) {
                            this.refresh_runs(cx);
                        }
                        // A rented instance costs money: its row stays current while it lives.
                        if tick.is_multiple_of(50) && this.instance_alive() {
                            this.refresh_instances(cx);
                        }
                        if changed {
                            cx.notify();
                        }
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        });
        self._tasks.push(task);
    }

    // ---------- actions ----------

    pub fn toast(
        &mut self,
        tone: Tone,
        title: impl Into<SharedString>,
        body: Option<SharedString>,
        cx: &mut Context<Self>,
    ) {
        self.toast_with(tone, title, body, None, cx);
    }

    pub fn toast_with(
        &mut self,
        tone: Tone,
        title: impl Into<SharedString>,
        body: Option<SharedString>,
        action: Option<(SharedString, Action)>,
        cx: &mut Context<Self>,
    ) {
        self.toast_seq += 1;
        let id = self.toast_seq;
        // Toasts with an action stay until used or dismissed; others expire, but not while hovered.
        let expires = action.is_none();
        self.toasts.push(Toast { id, tone, title: title.into(), body, action });
        // Over three, the oldest passive toast goes first: one with an action waits for its action
        // or a dismissal (the stack folds them) unless five are waiting. The focused toast stays,
        // and so does the one just added (it would vanish unseen).
        while self.toasts.len() > 3 {
            let keep = |t: &Toast| t.id != id && Some(t.id) != self.toast_focus;
            let gone = self
                .toasts
                .iter()
                .position(|t| t.action.is_none() && keep(t))
                .or_else(|| (self.toasts.len() > 5).then(|| self.toasts.iter().position(keep)).flatten());
            let Some(i) = gone else { break };
            self.toasts.remove(i);
        }
        if !self.frozen && expires {
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(Duration::from_millis(4200)).await;
                loop {
                    let hovered = this
                        .update(cx, |this, _| {
                            this.toast_hover == Some(id)
                                || this.toast_focus == Some(id)
                                || this.toast_stack_focus
                        })
                        .unwrap_or(false);
                    if !hovered {
                        break;
                    }
                    cx.background_executor().timer(Duration::from_millis(800)).await;
                }
                this.update(cx, |this, cx| {
                    this.toasts.retain(|t| t.id != id);
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
        cx.notify();
    }

    pub fn dismiss_toast(&mut self, id: u64, cx: &mut Context<Self>) {
        self.toasts.retain(|t| t.id != id);
        cx.notify();
    }

    pub fn set_mode(&mut self, mode: Mode, cx: &mut Context<Self>) {
        self.mode = mode;
        cx.set_global(Theme::new(mode));
        cx.refresh_windows();
    }

    pub fn toggle_theme(&mut self, cx: &mut Context<Self>) {
        let next = if self.mode == Mode::Dark { Mode::Light } else { Mode::Dark };
        self.set_mode(next, cx);
    }

    /// Remember the window (restored bounds, maximized or full screen) and the theme for the next
    /// launch, when this one remembers (`window_state`).
    pub fn save_window_state(&self, window: &Window, _cx: &App) {
        use crate::window_state::{Placement, State, WindowMode};
        let Some(path) = &self.state_path else {
            return;
        };
        let (b, mode) = match window.window_bounds() {
            gpui::WindowBounds::Windowed(b) => (b, WindowMode::Windowed),
            gpui::WindowBounds::Maximized(b) => (b, WindowMode::Maximized),
            gpui::WindowBounds::Fullscreen(b) => (b, WindowMode::Fullscreen),
        };
        let mode = crate::window_state::closing_mode(mode, self.shown_mode);
        let state = State {
            window: Some(Placement {
                x: b.origin.x.into(),
                y: b.origin.y.into(),
                width: b.size.width.into(),
                height: b.size.height.into(),
                mode,
            }),
            theme: Some(if self.mode == Mode::Light { "light" } else { "dark" }.into()),
        };
        if let Err(e) = crate::window_state::save(path, &state) {
            eprintln!("folo-desktop: couldn't remember the window in {}: {e}", path.display());
        }
    }

    pub fn open_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.palette.is_some() {
            self.close_palette(window, cx);
            return;
        }
        // One overlay at a time: the palette doesn't open over a dialog or the inspector.
        if self.modal_open() {
            return;
        }
        self.remember_focus(window, cx);
        let input = cx.new(|cx| TextInput::new("Search runs, models, datasets and commands…", cx));
        let sub = cx.subscribe(&input, |this: &mut Self, _, ev: &InputEvent, cx| {
            let InputEvent::Changed(q) = ev;
            if let Some(p) = this.palette.as_mut() {
                p.query = q.to_string();
                p.selected = 0;
            }
            cx.notify();
        });
        let fh = input.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.palette = Some(Palette {
            input,
            query: String::new(),
            selected: 0,
            scroll: ScrollHandle::new(),
            _sub: sub,
        });
        cx.notify();
    }

    pub fn close_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.palette = None;
        self.restore_focus(window);
        cx.notify();
    }

    /// Remember the focused control before an overlay takes focus.
    pub fn remember_focus(&mut self, window: &Window, cx: &App) {
        self.overlay_return = window.focused(cx);
    }

    /// Give focus back to whatever had it before the overlay opened.
    pub fn restore_focus(&mut self, window: &mut Window) {
        match self.overlay_return.take() {
            Some(fh) => window.focus(&fh),
            None => window.focus(&self.root_focus),
        }
    }

    /// Generate synthetic: a small dialog against folo serve; the built-in demo explains itself.
    pub fn open_synth(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.server_mode == ServerMode::Mock {
            self.toast(
                Tone::Info,
                "Synthetic datasets need folo serve",
                Some("Run `folo dataset synth`, or connect to folo serve in Settings.".into()),
                cx,
            );
            return;
        }
        self.remember_focus(window, cx);
        let taken: Vec<String> = self
            .store
            .datasets
            .ready()
            .map(|d| d.iter().map(|d| d.name.clone()).collect())
            .unwrap_or_default();
        let default_name = (1..)
            .map(|n| if n == 1 { "shapes".to_string() } else { format!("shapes-{n}") })
            .find(|n| !taken.contains(n));
        let name = cx.new(|cx| {
            let mut i = TextInput::new("shapes", cx);
            i.set_text(default_name.unwrap_or_default(), cx);
            i
        });
        let images = cx.new(|cx| {
            let mut i = TextInput::new("1000", cx);
            i.set_text("1000", cx);
            i
        });
        window.focus(&name.read(cx).focus_handle(cx));
        self.synth_seq += 1;
        self.synth = Some(SynthForm {
            id: self.synth_seq,
            name,
            images,
            imgsz: 1,
            classes: 0,
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    pub fn close_synth(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.synth = None;
        self.restore_focus(window);
        cx.notify();
    }

    pub fn submit_synth(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(f) = self.synth.as_mut() else { return };
        if f.submitting {
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        let images = parse_images(&f.images.read(cx).text());
        let (Some(images), false) = (images, name.is_empty()) else {
            f.error = Some(if name.is_empty() {
                "Give the dataset a name.".into()
            } else {
                "Images: a whole number from 1 to 100,000.".into()
            });
            cx.notify();
            return;
        };
        f.submitting = true;
        f.error = None;
        let form = f.id;
        let req = SyntheticDataset {
            name,
            images,
            classes: SYNTH_CLASSES[f.classes].iter().map(|s| s.to_string()).collect(),
            imgsz: SYNTH_IMGSZ[f.imgsz],
            seed: 0,
        };
        cx.notify();
        self.fetch(
            cx,
            move |api| api.synth_dataset(&req),
            move |this, r, cx| {
                let mine = this.synth.as_ref().is_some_and(|f| f.id == form);
                match r {
                    Ok(ds) => {
                        if mine {
                            this.synth = None;
                            // Back to whatever opened the dialog, on the next frame (needs a window).
                            this.restore_pending = true;
                        }
                        this.dataset_added(ds, "Generated", cx);
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.synth.as_mut() {
                            f.submitting = false;
                            f.error = Some(e.message);
                        }
                    }
                    // Its dialog is gone (dismissed, or another one open): say so instead.
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    /// Add dataset: pick a folder and register it with folo serve.
    pub fn add_dataset_folder(&mut self, cx: &mut Context<Self>) {
        if self.server_mode == ServerMode::Mock {
            self.toast(
                Tone::Info,
                "Registering folders needs folo serve",
                Some("The built-in demo has a fixed set of datasets.".into()),
                cx,
            );
            return;
        }
        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Add dataset".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| {
                    let path = p.to_string_lossy().to_string();
                    let folder = p.file_name().map_or(path.clone(), |n| n.to_string_lossy().to_string());
                    this.toast(
                        Tone::Info,
                        format!("Indexing {folder}"),
                        Some("Reading every image header…".into()),
                        cx,
                    );
                    let req = RegisterDataset { name: None, path, format: Default::default() };
                    this.fetch(
                        cx,
                        move |api| api.register_dataset(&req),
                        |this, r, cx| match r {
                            Ok(ds) => this.dataset_added(ds, "Added", cx),
                            Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                        },
                    );
                })
                .ok();
            }
        })
        .detach();
    }

    /// A new dataset from the server: list it first and say so, with a way to open it.
    fn dataset_added(&mut self, ds: crate::api::Dataset, verb: &str, cx: &mut Context<Self>) {
        let (id, name) = (ds.id.clone(), ds.name.clone());
        let body = format!("{} images · {}", int(ds.splits.total()), ds.classes.join(", "));
        // Registering a folder twice answers with the dataset it already is: say so, and leave
        // the list's order alone.
        // Unknown while the list isn't loaded: neither "added" nor "already added" is claimed then.
        let existed = self.store.datasets.ready().map(|l| l.iter().any(|d| d.id == ds.id));
        if let Some(list) = self.store.datasets.ready_mut() {
            if let Some(slot) = list.iter_mut().find(|d| d.id == ds.id) {
                *slot = ds;
            } else {
                list.insert(0, ds);
            }
        }
        self.toast_with(
            if existed == Some(true) { Tone::Info } else { Tone::Accent },
            match existed {
                Some(true) => format!("{name} is already added"),
                Some(false) => format!("{verb} {name}"),
                None => format!("{name} is ready"),
            },
            Some(body.into()),
            Some((
                "Open".into(),
                Rc::new(move |this: &mut Self, w: &mut Window, cx: &mut Context<Self>| {
                    this.navigate(Route::Dataset(id.clone()), w, cx)
                }),
            )),
            cx,
        );
    }

    /// Resume a stopped or failed run from its last checkpoint.
    pub fn resume_run(&mut self, id: String, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            move |api| api.resume_run(&id),
            |this, r, cx| match r {
                Ok(run) => {
                    let (rid, name) = (run.id.clone(), run.name.clone());
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == rid)
                    {
                        *slot = run;
                    }
                    // Resumed steps replace the earlier ones. Retire everything tied to the old
                    // attempt (stream, in-flight history, buffered events), then open a fresh
                    // stream; the dashboard reloads history once it is open.
                    this.subscriptions.remove(&rid);
                    *this.sub_attempt.entry(rid.clone()).or_default() += 1;
                    this.subscribing.remove(&rid);
                    this.store.reset_for_resume(&rid);
                    this.history_ticket(&rid, "metrics");
                    this.history_ticket(&rid, "logs");
                    this.history_ticket(&rid, "samples");
                    this.resubscribe(rid.clone(), cx);
                    this.toast(
                        Tone::Accent,
                        format!("Resuming {name}"),
                        Some("From its last finished epoch, with the optimizer state.".into()),
                        cx,
                    );
                }
                Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
            },
        );
    }

    /// The form to keep as a draft when the dialog closes without starting a run: none while a
    /// submission is pending (that run is being created; keeping it would allow a duplicate).
    fn keep_draft(&mut self, mut f: NewRunForm) -> Option<NewRunForm> {
        if f.submitting {
            self.new_run_draft_ds = None;
            return None;
        }
        self.new_run_draft_ds =
            self.store.datasets.ready().and_then(|d| d.get(f.dataset)).map(|d| d.id.clone());
        f.error = None;
        Some(f)
    }

    pub fn close_new_run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.new_run_draft = self.new_run.take().and_then(|f| self.keep_draft(f));
        self.restore_focus(window);
        cx.notify();
    }

    /// The New run dialog prefilled from an earlier run; `note` explains any adjustment.
    pub fn open_new_run_from(
        &mut self,
        config: &TrainConfig,
        note: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_new_run(Some(&config.dataset_id), window, cx);
        let idx = |list: &[u32], v: u32| list.iter().position(|x| *x == v);
        let name = format!("{}-2", config.name);
        let cloud_on = self.cloud_on();
        if let Some(f) = self.new_run.as_mut() {
            f.arch = ARCHS.iter().position(|a| *a == config.arch).unwrap_or(f.arch);
            f.epochs = config.epochs as f32;
            f.batch = batch_index(config.batch);
            f.imgsz = idx(&IMGSZ, config.imgsz).unwrap_or(f.imgsz);
            f.device = DEVICES.iter().position(|d| *d == config.device).unwrap_or(0);
            f.prefill_optimizer(config.optimizer);
            f.lr0 = config.lr0;
            let (ei, li) = (f.epochs_input.clone(), f.lr_input.clone());
            let (ci, cv) = (f.close_input.clone(), config.augment.close_mosaic.to_string());
            ci.update(cx, |i, cx| i.set_text(cv, cx));
            let (ev, lv) = (config.epochs.to_string(), format!("{}", config.lr0));
            ei.update(cx, |i, cx| i.set_text(ev, cx));
            li.update(cx, |i, cx| i.set_text(lv, cx));
            f.mosaic = config.augment.mosaic;
            f.mixup = config.augment.mixup;
            f.fliplr = config.augment.fliplr;
            f.close_mosaic = config.augment.close_mosaic as f32;
            f.scale = config.augment.scale;
            f.amp = config.amp;
            f.e2e = config.head == Head::O2o;
            f.note = note;
            f.base = Some(config.clone());
            // A cloud run's connection and offer, when they still exist (checked once the
            // connections and offers are in), and its time limit as it was.
            if let Some(c) = config.compute.as_ref().filter(|_| cloud_on) {
                f.run_on = Some(c.connection_id.clone());
                f.want_offer = Some(c.offer_id.clone());
                f.max_hours = c.max_hours;
                f.hours_edited = true;
                let (hi, hv) = (f.hours_input.clone(), crate::ui::hours_value(c.max_hours));
                hi.update(cx, |i, cx| i.set_text(hv, cx));
            }
            let input = f.name.clone();
            input.update(cx, |i, cx| i.set_text(name, cx));
        }
        self.sync_compute(cx);
        cx.notify();
    }

    pub fn open_new_run(&mut self, dataset: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        // Nothing to train on until the dataset list is in (Ctrl+N included).
        if self.store.datasets.ready().is_none_or(|d| d.is_empty()) {
            return;
        }
        if self.new_run.is_none() {
            self.remember_focus(window, cx);
        }
        // A plain New run picks up a draft left by Esc or Cancel; one for a specific dataset (or
        // prefilled from a run) starts fresh.
        // Its dataset is found again by id; gone (or the list unloaded), the draft is dropped.
        let draft_ds = self
            .new_run_draft_ds
            .take()
            .and_then(|id| self.store.datasets.ready().and_then(|d| d.iter().position(|d| d.id == id)));
        if let Some(mut draft) = self.new_run_draft.take()
            && self.new_run.is_none()
            && dataset.is_none()
            && let Some(idx) = draft_ds
        {
            draft.dataset = idx;
            let fh = draft.name.read(cx).focus_handle(cx);
            self.new_run = Some(draft);
            window.focus(&fh);
            cx.notify();
            return;
        }
        let idx =
            dataset.and_then(|id| self.store.datasets.ready()?.iter().position(|d| d.id == id)).unwrap_or(0);
        let n = self.store.runs.ready().map_or(0, |r| r.len()) + 1;
        let ds_name = self
            .store
            .datasets
            .ready()
            .and_then(|d| d.get(idx))
            .map_or("shapes".to_string(), |d| d.name.clone());
        let name = cx.new(|cx| {
            let mut i = TextInput::new("run name", cx);
            i.set_text(format!("{ds_name}-s-{n}"), cx);
            i
        });
        let fh = name.read(cx).focus_handle(cx);
        let epochs_input = cx.new(|cx| {
            let mut i = TextInput::new("100", cx);
            i.set_text("100", cx);
            i
        });
        let lr_input = cx.new(|cx| {
            let mut i = TextInput::new("0.01", cx);
            i.set_text("0.010", cx);
            i
        });
        let close_input = cx.new(|cx| {
            let mut i = TextInput::new("10", cx);
            i.set_text("10", cx);
            i
        });
        let hours_input = cx.new(|cx| {
            let mut i = TextInput::new("12", cx);
            i.set_text(crate::ui::hours_value(folo_api::DEFAULT_MAX_HOURS), cx);
            i
        });
        let subs = vec![
            cx.subscribe(&hours_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match crate::cloud::parse_hours(s) {
                        Some(v) => {
                            // Any other value than the one the form put there is the user's.
                            if crate::ui::hours_value(v) != crate::ui::hours_value(f.max_hours) {
                                f.hours_edited = true;
                            }
                            f.max_hours = v;
                            f.hours_invalid = false;
                        }
                        None => f.hours_invalid = true,
                    }
                    f.compute_error.take_if(|(c, _)| *c == crate::cloud::ComputeField::StopAfter);
                    cx.notify();
                }
            }),
            cx.subscribe(&close_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    f.check_close(s);
                    cx.notify();
                }
            }),
            cx.subscribe(&epochs_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match parse_epochs(s) {
                        Some(v) => {
                            f.epochs = v as f32;
                            f.epochs_invalid = false;
                        }
                        None => f.epochs_invalid = true,
                    }
                    // Close mosaic is bounded by the epochs: recheck it against the new count.
                    let close = f.close_input.read(cx).text();
                    f.check_close(&close);
                    cx.notify();
                }
            }),
            cx.subscribe(&lr_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match parse_lr(s) {
                        Some(v) => {
                            f.lr0 = v;
                            f.lr_invalid = false;
                        }
                        None => {
                            f.lr_invalid = true;
                            // The rate lives under Advanced: keep the group open while it's being
                            // fixed, so the first valid keystroke doesn't fold the field away.
                            f.advanced = true;
                        }
                    }
                    cx.notify();
                }
            }),
        ];
        self.new_run = Some(NewRunForm {
            name,
            dataset: idx,
            arch: 1,
            epochs: 100.0,
            batch: batch_index(16),
            imgsz: 2,
            device: 0,
            optimizer: 0,
            optimizer_unknown: false,
            lr0: 0.01,
            mosaic: 1.0,
            mixup: 0.0,
            fliplr: 0.5,
            close_mosaic: 10.0,
            scale: 0.5,
            note: None,
            base: None,
            epochs_input,
            lr_input,
            close_input,
            close_invalid: false,
            amp: false,
            e2e: false,
            advanced: false,
            run_on: None,
            offer: None,
            max_hours: folo_api::DEFAULT_MAX_HOURS,
            hours_input,
            hours_invalid: false,
            hours_edited: false,
            compute_error: None,
            want_offer: None,
            epochs_invalid: false,
            lr_invalid: false,
            _subs: subs,
            submitting: false,
            error: None,
        });
        window.focus(&fh);
        cx.notify();
    }

    /// The New run form as the config it would submit (its name as typed), or None without a
    /// form or dataset. Submitting and the server's estimate both use it.
    pub fn form_config(&self, cx: &App) -> Option<TrainConfig> {
        let form = self.new_run.as_ref()?;
        let compute = crate::cloud::form_compute(
            self.cloud_on(),
            form.run_on.as_deref(),
            form.offer.as_deref(),
            form.max_hours,
        );
        // On a cloud instance mixed precision is checked when it boots, not against this machine.
        let amp_ok = compute.is_some() || self.amp_blocked(DEVICES[form.device]).is_none();
        let ds = self.store.datasets.ready().and_then(|d| d.get(form.dataset))?;
        Some(carry_over(
            form.base.as_ref(),
            TrainConfig {
                name: form.name.read(cx).text().trim().to_string(),
                dataset_id: ds.id.clone(),
                arch: ARCHS[form.arch],
                epochs: form.epochs.round() as u32,
                batch: BATCHES[form.batch],
                imgsz: IMGSZ[form.imgsz],
                // The worker picks its GPU.
                device: if compute.is_some() { "auto".into() } else { DEVICES[form.device].into() },
                optimizer: OPTIMIZERS[form.optimizer],
                lr0: form.lr0,
                weights: None,
                augment: Augment {
                    mosaic: form.mosaic,
                    mixup: form.mixup,
                    fliplr: form.fliplr,
                    close_mosaic: form.close_mosaic.round() as u32,
                    scale: form.scale,
                    ..Augment::default()
                },
                seed: 0,
                // Only where it can run: not the CPU, nor a server that says it can't.
                amp: form.amp && amp_ok,
                head: if form.e2e { Head::O2o } else { Head::O2m },
                compute,
            },
        ))
    }

    /// Why New run can't send its config yet because of its compute choice: a connection chosen
    /// whose offers are still out, no offer picked, or a Stop after that doesn't parse.
    pub fn compute_blocked(&self) -> Option<&'static str> {
        let f = self.new_run.as_ref()?;
        let conn = f.run_on.as_deref().filter(|_| self.cloud_on())?;
        if f.hours_invalid {
            Some("Fix the highlighted fields first")
        } else if f.offer.is_some() {
            None
        } else if self.cloud.offers.get(conn).is_none_or(|l| l.is_loading()) {
            Some("Loading offers…")
        } else {
            Some("Choose a GPU first")
        }
    }

    /// Asks the server how long the New run form's config would take (servers with
    /// `run_estimate`), once per distinct config.
    pub fn ensure_estimate(&mut self, cx: &mut Context<Self>) {
        if !self.store.features.iter().any(|f| f == "run_estimate") {
            return;
        }
        // A value being typed that doesn't parse yet: no estimate for the last good one.
        // Nor for a cloud run without its offer yet (the config would be this machine's).
        if self.new_run.as_ref().is_some_and(|f| {
            f.epochs_invalid
                || f.close_invalid
                || (f.lr_invalid && f.optimizer != 0)
                || f.optimizer_blocks().is_some()
        }) || self.compute_blocked().is_some()
        {
            // Cancel a pending ask; an unanswered key would otherwise stay "loading" for good.
            self.nr_estimate_gen += 1;
            if matches!(self.nr_estimate, Some((_, Load::Loading))) {
                self.nr_estimate = None;
            }
            return;
        }
        let Some(mut config) = self.form_config(cx) else { return };
        // The name doesn't change the estimate (and an empty one mustn't fail it).
        config.name = "estimate".into();
        let key = serde_json::to_string(&config).unwrap_or_default();
        if self.nr_estimate.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        self.nr_estimate = Some((key.clone(), Load::Loading));
        self.nr_estimate_gen += 1;
        let generation = self.nr_estimate_gen;
        // Settled first (250 ms; at once for the frozen scenes), so a slider drag asks once for
        // the value it stops on rather than for every step.
        if self.frozen {
            self.request_estimate(key, config, cx);
            return;
        }
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(250)).await;
            this.update(cx, |this, cx| {
                if this.nr_estimate_gen == generation {
                    this.request_estimate(key, config, cx);
                }
            })
            .ok();
        })
        .detach();
    }

    fn request_estimate(&mut self, key: String, config: TrainConfig, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            move |api| api.estimate_run(&config),
            move |this, r, cx| {
                if this.nr_estimate.as_ref().is_none_or(|(k, _)| *k != key) {
                    return;
                }
                if let Ok(e) = &r {
                    this.nr_estimate_last = Some(e.clone());
                }
                this.nr_estimate = Some((key, r.map_or_else(Load::Failed, Load::Ready)));
                // Stop after follows the estimate (until typed).
                this.sync_compute(cx);
                cx.notify();
            },
        );
    }

    pub fn submit_new_run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let config = self.form_config(cx);
        let compute_blocked = self.compute_blocked().is_some();
        let Some(form) = self.new_run.as_mut() else { return };
        // Auto sets its own rate, so an unparsed rate entry doesn't block it.
        let lr_invalid = form.lr_invalid && form.optimizer != 0;
        if form.submitting
            || form.epochs_invalid
            || lr_invalid
            || form.close_invalid
            || form.optimizer_blocks().is_some()
            || compute_blocked
        {
            return;
        }
        let Some(config) = config else { return };
        if config.name.is_empty() {
            form.error = Some(ApiError::new("invalid_name", "Give the run a name."));
            // The error shows at the top of the form, by the Name field: go there.
            let name_fh = form.name.read(cx).focus_handle(cx);
            window.focus(&name_fh);
            self.dialog_scroll.set_offset(point(px(0.), px(0.)));
            cx.notify();
            return;
        }
        form.submitting = true;
        // Which form this request belongs to (its name input, kept by a restored draft): the
        // dialog can be dismissed and another run submitted before this one answers.
        let form_id = form.name.entity_id();
        let _ = window;
        self.fetch(
            cx,
            move |api| api.create_run(config),
            move |this, r, cx| {
                let mine = this.new_run.as_ref().is_some_and(|f| f.name.entity_id() == form_id);
                match r {
                    Ok(run) => {
                        let id = run.id.clone();
                        let name = run.name.clone();
                        if let Some(runs) = this.store.runs.ready_mut() {
                            runs.insert(0, run);
                        }
                        this.subscribe_active(cx);
                        let run = this.store.run(&id).cloned();
                        let cloud = run.as_ref().and_then(|r| Some(this.cloud_device(r, r.cloud.as_ref()?)));
                        let queued = run.is_some_and(|r| r.status == crate::api::RunStatus::Queued);
                        if let Some(on) = cloud {
                            // It waits on its instance, not in the queue.
                            this.toast(
                                Tone::Accent,
                                format!("Launching {name}"),
                                Some(format!("On {on}: the instance starts, then the data uploads.").into()),
                                cx,
                            );
                            this.refresh_instances(cx);
                        } else if queued {
                            // The run's page opens next, so the action is the queue it waits in.
                            this.toast_with(
                                Tone::Info,
                                format!("Queued {name}"),
                                Some("It starts when the current run finishes.".into()),
                                Some((
                                    "View queue".into(),
                                    Rc::new(
                                        move |this: &mut Self, w: &mut Window, cx: &mut Context<Self>| {
                                            this.run_filter = RunFilter::Active;
                                            this.navigate(Route::Runs, w, cx)
                                        },
                                    ),
                                )),
                                cx,
                            );
                        } else {
                            this.toast(
                                Tone::Accent,
                                format!("Training {name}"),
                                Some("Live metrics stream into the dashboard.".into()),
                                cx,
                            );
                        }
                        // An earlier, dismissed submission leaves the form now open alone.
                        if mine {
                            this.new_run = None;
                            this.overlay_return = None;
                            let prev = std::mem::replace(&mut this.route, Route::Run(id));
                            this.history.push(prev);
                        }
                    }
                    Err(e) => {
                        if mine && let Some(f) = this.new_run.as_mut() {
                            f.submitting = false;
                            // A refused compute choice shows on its control (Run on, GPU, Stop
                            // after); a GPU that went away or is out of stock gets fresh offers.
                            match crate::cloud::compute_error_field(&e).filter(|_| f.run_on.is_some()) {
                                Some(field) => {
                                    let conn = f.run_on.clone();
                                    f.compute_error = Some((field, e.message));
                                    if field == crate::cloud::ComputeField::Gpu
                                        && let Some(conn) = conn
                                    {
                                        this.ensure_offers(&conn, true, cx);
                                    }
                                }
                                None => f.error = Some(e),
                            }
                            // The error leads the form: scroll it into view.
                            this.dialog_scroll.set_offset(point(px(0.), px(0.)));
                        } else {
                            // Its form is gone: say so where it can be seen.
                            this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx);
                        }
                    }
                }
            },
        );
        cx.notify();
    }

    /// Moves a waiting run in the training queue (servers with `queue_control`).
    pub fn move_queued(&mut self, id: String, action: crate::api::QueueAction, cx: &mut Context<Self>) {
        let req = crate::api::QueueRequest { action: Some(action), position: None };
        self.fetch(
            cx,
            move |api| api.queue_run(&id, req),
            move |this, r, cx| match r {
                Ok(run) => {
                    let name = run.name.clone();
                    let started = run.status != RunStatus::Queued;
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                    let title =
                        if started { format!("Started {name}") } else { format!("{name} starts next") };
                    this.toast(Tone::Neutral, title, None, cx);
                }
                Err(e) => {
                    let msg = if e.code == "run_not_queued" {
                        "It has already left the queue.".to_string()
                    } else {
                        e.message.clone()
                    };
                    this.toast(Tone::Danger, "Couldn't move the run", Some(msg.into()), cx);
                }
            },
        );
    }

    pub fn stop_run(&mut self, id: String, cx: &mut Context<Self>) {
        self.confirm_stop = None;
        let rid = id.clone();
        // Shown as stopping right away; the status it had comes back if the server refuses.
        let mut prev = None;
        if let Some(runs) = self.store.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == id)
        {
            prev = Some(std::mem::replace(&mut r.status, RunStatus::Stopping));
        }
        self.fetch(
            cx,
            move |api| api.stop_run(&rid),
            move |this, r, cx| match r {
                Ok(run) => {
                    // Keep the event subscription: the server may answer `stopping` and still send
                    // the last steps, a final validation and the terminal status.
                    let name = run.name.clone();
                    let epoch = run.epoch;
                    let stopping = run.status == RunStatus::Stopping;
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                    let (title, body) = if stopping {
                        (
                            format!("Stopping {name}"),
                            "Finishing the current step; the last checkpoint is kept.".to_string(),
                        )
                    } else {
                        (format!("Stopped {name}"), format!("Last checkpoint kept (epoch {epoch})."))
                    };
                    this.toast(Tone::Neutral, title, Some(body.into()), cx);
                }
                Err(e) => {
                    // Unless events moved it on meanwhile, so Stop is offered again.
                    if let Some(prev) = prev
                        && let Some(runs) = this.store.runs.ready_mut()
                        && let Some(r) =
                            runs.iter_mut().find(|r| r.id == id && r.status == RunStatus::Stopping)
                    {
                        r.status = prev;
                    }
                    this.toast(Tone::Danger, "Could not stop the run", Some(e.message.into()), cx)
                }
            },
        );
        cx.notify();
    }

    pub fn export_model(&mut self, id: String, format: &'static str, cx: &mut Context<Self>) {
        self.export_open = false;
        self.fetch(
            cx,
            move |api| api.export_model(&id, format),
            |this, r, cx| match r {
                Ok(res) => {
                    let (v, u) = bytes(res.bytes);
                    let path = res.path.clone();
                    this.toast_with(
                        Tone::Success,
                        "Export complete",
                        Some(format!("{} · {v} {u}", display_path(&res.path)).into()),
                        Some((
                            "Show in folder".into(),
                            act(move |_, _, cx| cx.reveal_path(std::path::Path::new(&path))),
                        )),
                        cx,
                    );
                }
                Err(e) => this.toast(Tone::Danger, "Export failed", Some(e.message.into()), cx),
            },
        );
    }

    // ---------- playground ----------

    pub fn play_dataset_image(&mut self, dataset: &str, image_id: &str, cx: &mut Context<Self>) {
        let input = PredictInput::Dataset { dataset_id: dataset.to_string(), image_id: image_id.to_string() };
        let (d, i) = (dataset.to_string(), image_id.to_string());
        let name: SharedString = format!(
            "{} / {}",
            self.store.dataset_summary(dataset).map_or(dataset, |d| d.name.as_str()),
            image_id
        )
        .into();
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        self.fetch(
            cx,
            move |api| {
                let data = api.image(&d, &i, None)?;
                let (w, h) = match &data {
                    ImageData::Rgba { width, height, .. } => (*width, *height),
                    ImageData::Encoded(b) => {
                        image::load_from_memory(b).map(|i| (i.width(), i.height())).unwrap_or((0, 0))
                    }
                };
                to_render_image(data)
                    .map(|img| (img, w, h))
                    .ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                match r {
                    Ok((image, width, height)) => {
                        this.set_play_image(PlayImage { name, input, image, width, height }, cx);
                    }
                    Err(e) => {
                        this.toast(Tone::Danger, "Could not open the image", Some(e.message.into()), cx)
                    }
                }
            },
        );
    }

    pub fn play_file(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        // Also after a picker that opened while models were there, if they've gone since.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let name: SharedString =
            path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default().into();
        let n2 = name.to_string();
        self.play_load_seq += 1;
        let seq = self.play_load_seq;
        self.fetch(
            cx,
            move |_| {
                let bytes = std::fs::read(&path)
                    .map_err(|e| ApiError::new("read_failed", format!("Could not read {n2}: {e}")))?;
                let img = image::load_from_memory(&bytes).map_err(|_| {
                    ApiError::new("unsupported_image", format!("{n2} is not an image FOLO can read"))
                        .with_hint("Use a JPEG or PNG.")
                })?;
                let (w, h) = (img.width(), img.height());
                let render = to_render_image(ImageData::Rgba {
                    width: w,
                    height: h,
                    pixels: img.to_rgba8().into_raw(),
                })
                .ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))?;
                Ok((Arc::new(bytes), render, w, h))
            },
            move |this, r, cx| {
                if seq != this.play_load_seq {
                    return;
                }
                match r {
                    Ok((bytes, image, width, height)) => {
                        let input = PredictInput::File { name: name.to_string(), bytes };
                        this.set_play_image(
                            PlayImage { name: name.clone(), input, image, width, height },
                            cx,
                        );
                    }
                    Err(e) => this.toast(Tone::Danger, e.message.clone(), e.hint.map(Into::into), cx),
                }
            },
        );
    }

    pub fn play_zoom_by(&mut self, f: f32, cx: &mut Context<Self>) {
        (self.play_zoom, self.play_pan) = crate::viz::zoom_step(self.play_zoom, self.play_pan, f);
        cx.notify();
    }

    /// Put a new image on the Playground stage; its old detections never paint over it.
    fn set_play_image(&mut self, img: PlayImage, cx: &mut Context<Self>) {
        self.play_image_gen += 1;
        // Images live in the GPU sprite atlas until dropped explicitly. Called from fetch
        // callbacks, where no window is mid-update, so every window's atlas is reached.
        if let Some(old) = self.play_image.replace(img) {
            cx.drop_image(old.image, None);
        }
        self.play_pred = Load::Idle;
        self.play_pred_pending = false;
        self.play_pred_for = None;
        self.play_hover = None;
        self.play_zoom = 1.0;
        self.play_pan = (0., 0.);
        self.run_predict(cx);
    }

    /// Choose where Playground runs the model, and rerun it there.
    pub fn set_play_target(&mut self, target: Option<crate::local::Target>, cx: &mut Context<Self>) {
        if self.play_target != target {
            self.play_target = target;
            // Another device's answer (and timing): "Detecting…" while it runs, not the previous
            // result, or an OOM error and its Retry still live during the CPU fallback.
            self.rerun_predict(cx);
        }
    }

    /// Look for a GPU for local inference (once; setting up folo-gpu blocks for a moment).
    pub fn probe_local_gpu(&mut self, cx: &mut Context<Self>) {
        if !matches!(self.local_gpu, Load::Idle) || !self.store.api.local_inference() {
            return;
        }
        self.local_gpu = Load::Loading;
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async { crate::local::gpu() }).await;
            this.update(cx, |this, cx| {
                this.local_gpu = match r {
                    Ok((_, name)) => Load::Ready(name),
                    Err(e) => Load::Failed(crate::api::ApiError::new("no_gpu", e)),
                };
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Predict again from scratch, showing "Detecting…" meanwhile (Retry, or another device). A plain
    /// rerun of the same image and model, an IoU change, keeps the old result up instead.
    pub fn rerun_predict(&mut self, cx: &mut Context<Self>) {
        self.play_pred_for = None;
        self.run_predict(cx);
    }

    /// Sets the Playground threshold, refetching when it drops below what was fetched (filtering
    /// is local only above that floor).
    pub fn set_play_conf(&mut self, conf: f32, cx: &mut Context<Self>) {
        self.play_conf = conf;
        if conf < self.play_fetched_floor - 1e-6 {
            self.run_predict(cx);
        }
    }

    pub fn run_predict(&mut self, cx: &mut Context<Self>) {
        let (Some(model), Some(img)) = (self.play_model.clone(), self.play_image.as_ref()) else { return };
        // Fetch everything above the lowest threshold once; the conf slider filters locally. The
        // floor is 0.05, or lower when the threshold is (a model's best-F1 confidence can be), so
        // local filtering never claims a threshold the fetch cut off.
        let floor = self.play_conf.min(PLAY_CONF_FLOOR);
        let req = PredictRequest {
            model_id: model.clone(),
            input: img.input.clone(),
            conf: floor,
            iou: self.play_iou,
            max_det: 300,
        };
        self.play_fetched_floor = floor;
        let key = (self.play_image_gen, model);
        // Same image and model (an IoU change): keep showing the old boxes until the new ones land.
        if self.play_pred_for.as_ref() != Some(&key) {
            self.play_pred = Load::Loading;
            self.play_hover = None;
        }
        self.play_pred_seq += 1;
        self.play_pred_pending = true;
        let seq = self.play_pred_seq;
        let target = self.play_target;
        // Taken with the backend handle: a switch after this makes the load's model throwaway.
        let generation = crate::local::generation();
        self.fetch(
            cx,
            move |api| match target {
                Some(t) => crate::local::predict(api, &req, t, generation),
                None => api.predict(&req),
            },
            move |this, r, _| {
                if seq == this.play_pred_seq {
                    // Every detection has a name to show: a class outside the model's list (a
                    // mismatched server, an old export) reads "class N", never a blank row.
                    let r = r.map(|mut p| {
                        for b in &mut p.boxes {
                            if b.name.trim().is_empty() {
                                b.name = format!("class {}", b.class);
                            }
                        }
                        p
                    });
                    this.play_pred = r.into();
                    this.play_pred_for = Some(key);
                    this.play_pred_pending = false;
                }
            },
        );
    }

    /// A run's identity line, "folo-m · shapes-hard": architecture, then dataset, the one order every
    /// surface uses (the hero, Recent runs, the run page, focus tips).
    pub fn run_meta(&self, r: &Run) -> String {
        let ds = self
            .store
            .dataset_summary(&r.config.dataset_id)
            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
        format!("{} · {ds}", r.config.arch.as_str())
    }

    /// The split a run's per-epoch scores come from: "val", or "train" when its dataset has no
    /// validation images (the trainer then evaluates on the training images), so those scores are
    /// never called validation.
    pub fn eval_split(&self, r: &Run) -> &'static str {
        self.store
            .dataset_summary(&r.config.dataset_id)
            .map_or("val", |d| if d.splits.val == 0 { "train" } else { "val" })
    }

    /// The mAP chart's title for that split.
    pub fn map_title(&self, r: &Run) -> &'static str {
        if self.eval_split(r) == "train" { "mAP on training images" } else { "Validation mAP" }
    }

    /// A run's subtitle, the same in the Overview hero and on its page: identity, device, start.
    pub fn run_subtitle(
        &self,
        t: &crate::theme::Theme,
        r: &Run,
        size: crate::theme::TypeStep,
        color: gpui::Hsla,
        id: &str,
    ) -> gpui::Div {
        let ds = self
            .store
            .dataset_summary(&r.config.dataset_id)
            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
        // A cloud run's device is where it rents it ("RunPod · RTX 4090").
        let device = match &r.cloud {
            Some(c) => self.cloud_device(r, c),
            None => crate::ui::device_label(&r.device),
        };
        let mut rest = vec![ds, device];
        // Precision is part of what trains (the e2e tag names the head).
        if r.config.amp {
            rest.push("mixed precision".into());
        }
        if let Some(s) = r.started.as_deref().and_then(crate::api::time::parse) {
            rest.push(format!("started {}", crate::api::time::ago(s, self.store.api.now())));
        }
        // The architecture leads, its e2e tag right after it, as in every list.
        let tag = (r.config.head == Head::O2o).then(|| crate::ui::e2e_tag(t, format!("e2e-{id}-{}", r.id)));
        crate::ui::meta_row(size, color, r.config.arch.as_str(), tag, &rest)
    }

    /// Whether `m` is end-to-end (the one-to-one head: it predicts without NMS). The model's own
    /// record says (`Model.head`, from its folo.config), so imported and orphaned models read right.
    pub fn model_e2e(&self, m: &Model) -> bool {
        m.head == Head::O2o
    }

    /// The help shown wherever a model or run is marked end-to-end.
    pub const E2E_HELP: &'static str =
        "End-to-end: predicts boxes directly, without non-maximum suppression (NMS)";

    /// Why Mixed precision can't be used for a run on `device` (an entry of `DEVICES`), if it can't:
    /// the CPU, or the server's own answer (`capabilities.amp`, the message such a run would fail
    /// with). An older server without capabilities says nothing; the run's failure tells then.
    pub fn amp_blocked(&self, device: &str) -> Option<String> {
        if device == "cpu" {
            return Some("Off on the CPU: it needs a GPU with tensor cores".into());
        }
        let caps = self.store.capabilities.as_ref()?;
        // The tensor cores' own short reason when there is one ("off: <GPU> has no cooperative
        // matrices", "off (not requested: …)"); else the trainer's full message.
        (!caps.amp.available).then(|| match (&caps.tensor_cores.reason, &caps.amp.reason) {
            (Some(tc), _) => format!("Not available: tensor cores {tc}"),
            (None, Some(amp)) => amp.clone(),
            (None, None) => "Mixed precision isn't available on this server".into(),
        })
    }

    /// Why Open image can't run now, if it can't: the model list failed (with the runs list too:
    /// not connected) or is still loading, or there's no model yet. Only the models decide: a
    /// failed runs list alone doesn't stop inference. The button, Ctrl+O and the palette share this.
    pub fn open_image_blocked(&self) -> Option<&'static str> {
        if self.store.models.error().is_some() {
            Some(if self.store.runs.error().is_some() {
                "Not connected"
            } else {
                "Models didn't load: retry under Best model per dataset"
            })
        } else if self.store.models.ready().is_none() {
            Some("Loading models…")
        } else if self.store.models.ready().is_some_and(|m| m.is_empty()) {
            Some("Train a model first")
        } else {
            None
        }
    }

    pub fn pick_image(&mut self, cx: &mut Context<Self>) {
        // The shortcut and palette obey the button's rule, and say why.
        if let Some(why) = self.open_image_blocked() {
            self.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
            return;
        }
        let rx = cx.prompt_for_paths(gpui::PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open image".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| {
                    this.route = Route::Playground;
                    this.play_file(p, cx);
                })
                .ok();
            }
        })
        .detach();
    }

    // ---------- sliders ----------

    pub fn on_drag_move(&mut self, ev: &MouseMoveEvent, _window: &mut Window, cx: &mut Context<Self>) {
        // A scroll thumb being dragged: move the content by the thumb's travel ratio.
        if let Some((handle, y0, off0)) = self.scroll_drag.as_ref() {
            if let Some(g) = crate::ui::ThumbGeo::of(handle) {
                let travel = (g.h - g.thumb).max(px(1.));
                let to = (*off0 + (ev.position.y - *y0) * (g.max / travel)).clamp(px(0.), g.max);
                handle.set_offset(point(handle.offset().x, -to));
                cx.notify();
            }
            return;
        }
        let Some(drag) = self.drag.as_ref() else { return };
        let Some(b) = self.slider_bounds.borrow().get(drag.id).copied() else { return };
        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
        let id = drag.id;
        self.set_slider(id, frac, cx);
    }

    fn on_drag_end(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if self.scroll_drag.take().is_some() {
            cx.notify();
        }
        // An image pan ends wherever the button comes up (also outside its canvas).
        if self.play_drag.take().is_some() {
            cx.notify();
        }
        if let Some(i) = self.inspector.as_mut()
            && i.drag_from.take().is_some()
        {
            cx.notify();
        }
        if let Some(d) = self.drag.take() {
            if d.id == "play-iou" {
                self.run_predict(cx);
            }
            cx.notify();
        }
    }

    // ---------- render ----------

    fn titlebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let crumbs = self.breadcrumbs();
        let maximized = window.is_maximized();
        // In full screen GPUI reports no caption areas, so Windows doesn't act on the buttons: they
        // take their own clicks then. Only then, since otherwise the caption press already acts
        // (on mouse up) and a click handler would act a second time.
        let fullscreen = window.is_fullscreen();
        let win_btn = |id: &'static str, icon_path: &'static str, area: WindowControlArea, close: bool| {
            div()
                .id(id)
                .group(id)
                .w(px(crate::theme::layout::DESKTOP_CAPTION_BUTTON))
                .h_full()
                .flex()
                .items_center()
                .justify_center()
                .window_control_area(area)
                // See `titlebar-drag`: the press must reach Windows, not focus the root.
                .occlude()
                .hover(move |s| if close { s.bg(t.window_close) } else { s.bg(t.bg_hover) })
                .active(|s| s.opacity(0.8))
                .child(
                    gpui::svg()
                        .path(icon_path)
                        .size(px(10.))
                        .flex_none()
                        .text_color(t.text_secondary)
                        .group_hover(id, move |s| {
                            s.text_color(if close { t.window_close_text } else { t.text_primary })
                        }),
                )
        };
        let search_fh = self.fh("titlebar-search", cx);
        let search_ring = self.ring(&search_fh, window);
        div()
            .h(px(tsize::TOPBAR - 8.))
            .flex_none()
            .flex()
            .items_center()
            .border_b_1()
            .border_color(t.line_subtle)
            .child(
                // Breadcrumbs (draggable area).
                div()
                    .id("titlebar-drag")
                    .flex_1()
                    .h_full()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    // The crumbs' 4 px pads put their text on the page gutter.
                    .pl(px(self.gutter() - 4.))
                    .window_control_area(WindowControlArea::Drag)
                    // A caption press goes to GPUI first and only reaches Windows (drag, double-click
                    // maximize, the caption buttons) if nothing prevents its default. The focus-tracked
                    // root would (focusing itself does), so the chrome hides the root from the hit
                    // test. The crumbs above it still take their own clicks. Hidden from the root, it
                    // leaves keyboard mode itself, as the root does for any other press.
                    .occlude()
                    .capture_any_mouse_down(self.leave_kb(cx))
                    .children(crumbs.into_iter().enumerate().map(|(i, (label, route))| {
                        let last = route.is_none();
                        let crumb = div().id(("crumb", i)).px(px(4.)).py(px(2.)).rounded(px(4.)).child(
                            text(ty::BASE, if last { t.text_primary } else { t.text_tertiary })
                                .when(last, |d| d.font_weight(FontWeight::MEDIUM))
                                .whitespace_nowrap()
                                .child(label),
                        );
                        let crumb = match route {
                            Some(r) => self
                                .bind(
                                    crumb
                                        .cursor_pointer()
                                        .hover(|s| s.bg(t.bg_hover))
                                        .active(|s| s.bg(t.bg_active)),
                                    &format!("crumb-{i}"),
                                    4.,
                                    window,
                                    cx,
                                    act(move |this, w, cx| this.navigate(r.clone(), w, cx)),
                                )
                                .into_any_element(),
                            None => crumb.into_any_element(),
                        };
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .when(i > 0, |d| d.child(icon("icons/chevron-right.svg", 12., t.text_disabled)))
                            .child(crumb)
                    })),
            )
            .child(
                div()
                    .id("titlebar-search")
                    .track_focus(&search_fh)
                    .relative()
                    .w(px(280.))
                    .h(px(28.))
                    .mr(px(12.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(6.))
                    .bg(t.bg_raised)
                    .border_1()
                    .border_color(t.line_default)
                    .cursor_pointer()
                    .hover(|s| s.border_color(t.line_strong))
                    .active(|s| s.bg(t.bg_active))
                    .on_click(cx.listener(|this, _, w, cx| this.open_palette(w, cx)))
                    .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, w, cx| {
                        if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                            this.open_palette(w, cx);
                            cx.stop_propagation();
                        }
                    }))
                    .when(search_ring, |d| d.child(focus_ring(&t, 6.)))
                    .child(icon("icons/search.svg", 14., t.text_tertiary))
                    .child(text(ty::SM, t.text_tertiary).flex_1().child("Search or jump to…"))
                    .child(shortcut(&t, &["Ctrl", "K"])),
            )
            .child(
                win_btn("win-min", "icons/win-min.svg", WindowControlArea::Min, false)
                    .capture_any_mouse_down(self.leave_kb(cx))
                    .when(fullscreen, |d| d.on_click(|_, w, _| w.minimize_window())),
            )
            .child(
                win_btn(
                    "win-max",
                    if maximized || fullscreen { "icons/win-restore.svg" } else { "icons/win-max.svg" },
                    WindowControlArea::Max,
                    false,
                )
                .capture_any_mouse_down(self.leave_kb(cx))
                // Full screen's "restore" leaves full screen.
                .when(fullscreen, |d| d.on_click(|_, w, _| w.toggle_fullscreen())),
            )
            .child(
                win_btn("win-close", "icons/win-close.svg", WindowControlArea::Close, true)
                    .capture_any_mouse_down(self.leave_kb(cx))
                    .when(fullscreen, |d| {
                        // Closing this way skips the window's should-close hook: remember here.
                        d.on_click(cx.listener(|this, _, w, cx| {
                            this.save_window_state(w, cx);
                            w.remove_window();
                        }))
                    }),
            )
    }

    fn breadcrumbs(&self) -> Vec<(SharedString, Option<Route>)> {
        match &self.route {
            // Top-level pages carry their own title; the titlebar stays quiet.
            Route::Overview
            | Route::Datasets
            | Route::Runs
            | Route::Models
            | Route::Playground
            | Route::Settings => {
                vec![]
            }
            Route::Dataset(id) => vec![
                ("Datasets".into(), Some(Route::Datasets)),
                (self.store.dataset_summary(id).map_or(id.clone(), |d| d.name.clone()).into(), None),
            ],
            Route::Run(id) => vec![
                ("Runs".into(), Some(Route::Runs)),
                (self.store.run(id).map_or(id.clone(), |r| r.name.clone()).into(), None),
            ],
            Route::Model(id) => vec![
                ("Models".into(), Some(Route::Models)),
                (self.store.model(id).map_or(id.clone(), |m| m.name.clone()).into(), None),
            ],
        }
    }

    fn sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        // Narrow windows collapse the sidebar to an icon rail.
        let compact = window.viewport_size().width < px(1120.);
        let section = self.route.section();
        let live = self.store.live_run().cloned();
        let active_count = self
            .store
            .runs
            .ready()
            .map_or(0, |r| r.iter().filter(|r| r.status == RunStatus::Running).count());
        let nav = [
            (Section::Overview, "Overview", "icons/layout-dashboard.svg", Route::Overview, "1"),
            (Section::Datasets, "Datasets", "icons/images.svg", Route::Datasets, "2"),
            (Section::Runs, "Runs", "icons/activity.svg", Route::Runs, "3"),
            (Section::Models, "Models", "icons/box.svg", Route::Models, "4"),
            (Section::Playground, "Playground", "icons/scan-eye.svg", Route::Playground, "5"),
        ];
        let items: Vec<AnyElement> = nav
            .into_iter()
            .map(|(sec, label, icon_path, route, key)| {
                let badge = (sec == Section::Runs && active_count > 0).then_some(active_count);
                self.nav_item(label, icon_path, section == sec, route, key, badge, compact, window, cx)
            })
            .collect();
        let settings = self.nav_item(
            "Settings",
            "icons/settings.svg",
            section == Section::Settings,
            Route::Settings,
            ",",
            None,
            compact,
            window,
            cx,
        );
        let mut live_card = None;
        // On Overview the live run is already the hero; don't repeat it in the sidebar.
        if let Some(r) =
            live.filter(|r| self.route != Route::Overview && self.route != Route::Run(r.id.clone()))
        {
            live_card =
                Some(if compact { self.live_rail(&r, window, cx) } else { self.live_mini(&r, window, cx) });
        }
        let brand_press = self.leave_kb(cx);
        let health = self.connection_line(compact, window, cx);
        div()
            .w(px(if compact { tsize::SIDEBAR_COLLAPSED } else { tsize::SIDEBAR }))
            .flex_none()
            .h_full()
            .flex()
            .flex_col()
            .bg(t.bg_panel)
            .border_r_1()
            .border_color(t.line_subtle)
            .child(
                // Brand row doubles as a drag area for the window.
                div()
                    .id("brand")
                    .h(px(tsize::TOPBAR - 8.))
                    .flex_none()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .px(px(if compact { 18. } else { 16. }))
                    .window_control_area(WindowControlArea::Drag)
                    // See `titlebar-drag`.
                    .occlude()
                    .capture_any_mouse_down(brand_press)
                    .child(logo(&t, 20.))
                    .when(!compact, |d| {
                        d.child(
                            div()
                                .flex()
                                .items_baseline()
                                .gap(px(6.))
                                .child(
                                    text(ty::MD, t.text_primary)
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .child("FOLO"),
                                )
                                .child(text(ty::MD, t.text_tertiary).child("Desktop")),
                        )
                    }),
            )
            .child(div().flex().flex_col().gap(px(2.)).px(px(8.)).pt(px(8.)).children(items))
            .child(div().flex_1())
            .children(live_card)
            .child(div().px(px(8.)).pb(px(4.)).child(settings))
            .child(health)
    }

    #[allow(clippy::too_many_arguments)]
    fn nav_item(
        &mut self,
        label: &'static str,
        icon_path: &'static str,
        active: bool,
        route: Route,
        key: &'static str,
        badge: Option<usize>,
        compact: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(format!("nav-{label}"), cx);
        let ring = self.ring(&fh, window);
        let r2 = route.clone();
        div()
            .id(SharedString::from(format!("nav-{label}")))
            .track_focus(&fh)
            .relative()
            .h(px(30.))
            .px(px(8.))
            .flex()
            .items_center()
            .gap(px(10.))
            .rounded(px(6.))
            .cursor_pointer()
            .when(active, |d| d.bg(t.bg_active))
            .when(!active, |d| d.hover(|s| s.bg(t.bg_hover)))
            .active(|s| s.bg(t.bg_active))
            .on_click(cx.listener(move |this, _, w, cx| this.navigate(route.clone(), w, cx)))
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(r2.clone(), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, 6.)))
            // The icon rail names the focused item, as its tooltip does for the pointer.
            .when(compact && ring, |d| {
                d.child(crate::controls::focus_tip(&t, format!("{label} · {}", shortcut_label(key)), 34.))
            })
            .when(compact, |d| {
                d.justify_center().px(px(0.)).tooltip(crate::controls::tip(label, Some(shortcut_label(key))))
            })
            .child(icon(icon_path, 16., if active { t.text_primary } else { t.text_tertiary }))
            .when(compact && badge.is_some(), |d| {
                d.child(div().absolute().top(px(6.)).right(px(8.)).child(dot(
                    t.accent_fill(),
                    6.,
                    true,
                    "nav-live-c",
                )))
            })
            .when(!compact, |d| {
                d.child(
                    text(ty::BASE, if active { t.text_primary } else { t.text_secondary })
                        .when(active, |d| d.font_weight(FontWeight::MEDIUM))
                        .flex_1()
                        .child(label),
                )
                .when_some(badge, |d, n| {
                    d.child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(5.))
                            .child(dot(t.accent_fill(), 6., true, "nav-live"))
                            .child(
                                text(ty::SM, t.accent_text)
                                    .tnum()
                                    .font_weight(FontWeight::MEDIUM)
                                    .child(format!("{n} live")),
                            ),
                    )
                })
                .tooltip(crate::controls::tip(label, Some(shortcut_label(key))))
            })
            .into_any_element()
    }

    fn live_mini(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let id = r.id.clone();
        let fh = self.fh("live-mini", cx);
        let ring = self.ring(&fh, window);
        let id2 = id.clone();
        // A single footer row: the nav badge already says a run is live, so this only names it.
        div()
            .id("live-mini")
            .track_focus(&fh)
            .relative()
            .mx(px(8.))
            .mb(px(6.))
            .h(px(36.))
            .px(px(10.))
            .flex()
            .flex_col()
            .justify_center()
            .gap(px(6.))
            .rounded(px(crate::theme::radius::MD))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(format!("{} · epoch {} / {}", r.name, r.epoch, r.epochs), None))
            .on_click(cx.listener(move |this, _, w, cx| this.navigate(Route::Run(id.clone()), w, cx)))
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(Route::Run(id2.clone()), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, crate::theme::radius::MD)))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(dot(t.accent_fill(), 6., true, "mini"))
                    .child(text(ty::SM, t.text_primary).w(px(140.)).ellipsis().child(r.name.clone()))
                    .child(div().flex_1())
                    .child(
                        text(ty::SM, t.text_secondary).tnum().child(format!("{:.0}%", r.progress() * 100.0)),
                    ),
            )
            .child(progress(&t, r.progress(), t.accent_fill(), 2.))
            .into_any_element()
    }

    /// The live run as a rail button: a progress ring around a pulsing dot.
    fn live_rail(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let id = r.id.clone();
        let el = div()
            .id("live-rail")
            .mx(px(10.))
            .mb(px(8.))
            .h(px(40.))
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap(px(5.))
            .rounded(px(8.))
            .bg(t.bg_raised)
            .border_1()
            .border_color(t.line_subtle)
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_default))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(format!("{} · epoch {} / {}", r.name, r.epoch, r.epochs), None))
            .child(dot(t.accent_fill(), 6., true, "rail"))
            .child(text(ty::SM, t.text_secondary).tnum().child(format!("{:.0}%", r.progress() * 100.)));
        self.bind(
            el,
            "live-rail",
            8.,
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
        )
        .into_any_element()
    }

    fn connection_line(&self, compact: bool, window: &Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let (color, glyph, label) = match &self.store.health {
            _ if self.connecting.is_some() => (
                t.warning_solid,
                "icons/refresh-cw.svg",
                self.connecting.clone().unwrap_or_default().to_string(),
            ),
            Load::Ready(h) => (
                t.success_solid,
                "icons/circle-check.svg",
                format!("{} · v{}", self.store.api.describe(), h.version),
            ),
            Load::Failed(_) => (t.danger_solid, "icons/circle-x.svg", "Server unreachable".to_string()),
            _ => (t.text_disabled, "icons/refresh-cw.svg", "Connecting…".to_string()),
        };
        // A control, by pointer and keyboard: it opens Connection settings, and the collapsed rail's
        // glyph (a shape, not colour alone) shows the full status on hover and on focus.
        let tip = format!("Connection: {label}");
        let ring = self.ring(&self.fh("connection", cx), window);
        let el = div()
            .id("connection")
            .relative()
            .h(px(36.))
            .flex_none()
            .flex()
            .items_center()
            .gap(px(8.))
            .px(px(16.))
            .border_t_1()
            .border_color(t.line_subtle)
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .tooltip(crate::controls::tip(tip.clone(), None))
            .when(compact, |d| {
                d.justify_center()
                    .px(px(0.))
                    .child(icon(glyph, 14., color))
                    .when(ring, |d| d.child(crate::controls::focus_tip(&t, tip.clone(), 30.)))
            })
            .when(!compact, |d| {
                d.child(div().size(px(6.)).rounded_full().bg(color))
                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(label))
            });
        self.bind(el, "connection", 0., window, cx, act(|this, w, cx| this.navigate(Route::Settings, w, cx)))
    }

    fn page(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        match self.route.clone() {
            Route::Overview => self.overview(window, cx),
            Route::Datasets => self.datasets(window, cx),
            Route::Dataset(id) => self.dataset(&id, window, cx),
            Route::Runs => self.runs(window, cx),
            Route::Run(id) => self.run_view(&id, window, cx),
            Route::Models => self.models(window, cx),
            Route::Model(id) => self.model_view(&id, window, cx),
            Route::Playground => self.playground(window, cx),
            Route::Settings => self.settings(window, cx),
        }
    }
}

fn shortcut_label(key: &str) -> &'static str {
    match key {
        "1" => "Ctrl 1",
        "2" => "Ctrl 2",
        "3" => "Ctrl 3",
        "4" => "Ctrl 4",
        "5" => "Ctrl 5",
        _ => "Ctrl ,",
    }
}

/// Runs newest first (the server already sends them so; the mock and refreshes may not).
fn newest_first_runs(mut runs: Vec<Run>) -> Vec<Run> {
    runs.sort_by(|a, b| b.created.cmp(&a.created));
    runs
}

/// The FOLO mark: a viewfinder (four corner brackets) around the signal dot.
pub fn logo(t: &Theme, size: f32) -> impl IntoElement {
    div()
        .relative()
        .size(px(size))
        .flex_none()
        .child(gpui::svg().path("icons/logo-frame.svg").size(px(size)).text_color(t.text_primary))
        .child(
            div()
                .absolute()
                .top(px(size * (12.0 - 3.25) / 24.0))
                .left(px(size * (12.0 - 3.25) / 24.0))
                .size(px(size * 6.5 / 24.0))
                .rounded_full()
                .bg(t.accent_solid),
        )
}

thread_local! {
    /// Scroll containers to search after a keyboard focus move, innermost first. The focus ring
    /// that paints next scrolls its control into view in the first container that holds it.
    static REVEAL: RefCell<Option<Vec<ScrollHandle>>> = const { RefCell::new(None) };
}

/// Scroll the first candidate container holding `b` (horizontally) so `b` is fully visible,
/// clear of the edge fades (24 px, so 32 at the top) and of the key-hint strip over the page's
/// bottom (32 px, so 40 there). Runs once per request.
fn reveal(b: gpui::Bounds<Pixels>, window: &mut Window) {
    let Some(handles) = REVEAL.with(|r| r.borrow_mut().take()) else { return };
    let margin = px(8.);
    let (top_m, bottom_m) = (px(32.), px(40.));
    // The innermost clip around the control: only a container that contains it owns the control
    // (a dialog's fixed header and footer sit outside the scrolling body).
    let mask = window.content_mask().bounds;
    let tol = px(1.);
    for h in handles {
        let vp = h.bounds();
        let inside = mask.left() >= vp.left() - tol
            && mask.right() <= vp.right() + tol
            && mask.top() >= vp.top() - tol
            && mask.bottom() <= vp.bottom() + tol;
        if vp.size.width <= px(0.) || !inside {
            continue;
        }
        let tall = b.size.height + top_m + bottom_m > vp.size.height;
        let dy = if b.top() - top_m < vp.top() && !tall {
            b.top() - top_m - vp.top()
        } else if tall {
            // Taller than the view (the run page's linked chart group at minimum size): its top
            // leads. Already in view, it stays; elsewhere, it comes to the top edge, never past.
            if b.top() < vp.top() || b.top() + margin > vp.bottom() - margin * 4. {
                b.top() - top_m - vp.top()
            } else {
                px(0.)
            }
        } else if b.bottom() + bottom_m > vp.bottom() {
            b.bottom() + bottom_m - vp.bottom()
        } else {
            px(0.)
        };
        if dy != px(0.) {
            let off = h.offset();
            let y = (off.y - dy).clamp(-h.max_offset().height, px(0.));
            h.set_offset(gpui::point(off.x, y));
            window.refresh();
        }
        return;
    }
}

/// The strip over the page's bottom edge that names the focused control's keys: 32 px, a
/// hairline above, a keyboard glyph and the keys. An overlay, so focusing a control never moves
/// the page (its last 32 px are the page's own bottom padding).
fn key_hint_strip(t: &Theme, keys: SharedString) -> gpui::Stateful<gpui::Div> {
    div()
        .id("key-hint")
        .flex_none()
        .h(px(32.))
        .px(px(16.))
        .flex()
        .items_center()
        .gap(px(8.))
        .border_t_1()
        .border_color(t.line_subtle)
        .bg(t.bg_panel)
        .child(crate::ui::icon("icons/keyboard.svg", 14., t.text_tertiary))
        .child(crate::ui::text(ty::SM, t.text_secondary).whitespace_nowrap().child(keys))
}

/// Keyboard focus ring: 2 px in the focus colour, 2 px outside the control.
pub fn focus_ring(t: &Theme, radius: f32) -> gpui::Div {
    let off = tsize::FOCUS_OFFSET + tsize::FOCUS_RING;
    div()
        .absolute()
        .top(px(-off))
        .left(px(-off))
        .right(px(-off))
        .bottom(px(-off))
        .rounded(px(radius + off))
        .border_2()
        .border_color(t.focus)
        // Keyboard focus moved here: make sure the control (and its ring) is on screen.
        .child(gpui::canvas(|b, window, _| reveal(b, window), |_, _, _, _| {}).size_full())
}

impl Render for FoloApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if std::mem::take(&mut self.restore_pending) {
            self.restore_focus(window);
        }
        if self.scroll_end_pending > 0 {
            self.scroll_end_pending -= 1;
            let max = self.page_scroll.max_offset().height;
            self.page_scroll.set_offset(gpui::point(px(0.), -max));
            window.request_animation_frame();
        }
        if self.reveal_pending > 0 {
            self.reveal_pending -= 1;
            REVEAL.with(|r| *r.borrow_mut() = Some(vec![self.page_scroll.clone()]));
            window.request_animation_frame();
        }
        if self.open_new_run_pending && self.store.datasets.ready().is_some_and(|d| !d.is_empty()) {
            self.open_new_run_pending = false;
            self.open_new_run(None, window, cx);
        }
        self.narrow = window.viewport_size().width < px(1120.);
        self.shown_mode = if window.is_fullscreen() {
            crate::window_state::WindowMode::Fullscreen
        } else if window.is_maximized() {
            crate::window_state::WindowMode::Maximized
        } else {
            crate::window_state::WindowMode::Windowed
        };
        let t = theme(cx);
        self.ensure_core(cx);
        self.ensure_cloud(cx);
        let titlebar = self.titlebar(window, cx).into_any_element();
        let sidebar = self.sidebar(window, cx).into_any_element();
        *self.key_hint.borrow_mut() = None;
        let page = self.page(window, cx);
        // A dialog or the inspector owns the keys: the page's hint stands down.
        let key_hint = self.key_hint.borrow_mut().take().filter(|_| self.kb && !self.modal_open());
        let overlays = self.overlays(window, cx);
        div()
            .id("root")
            .key_context("FoloApp")
            .track_focus(&self.root_focus)
            .size_full()
            .flex()
            .font_family(crate::theme::SANS)
            .text_size(px(ty::BASE.size))
            .line_height(px(ty::BASE.line))
            .text_color(t.text_primary)
            .bg(t.bg_app)
            .on_action(cx.listener(|this, _: &OpenPalette, w, cx| this.open_palette(w, cx)))
            .on_action(cx.listener(|this, _: &NewRunAction, w, cx| {
                if !this.modal_open() {
                    this.palette = None;
                    this.open_new_run(None, w, cx)
                }
            }))
            .on_action(cx.listener(|this, _: &GoOverview, w, cx| this.nav_shortcut(Route::Overview, w, cx)))
            .on_action(cx.listener(|this, _: &GoDatasets, w, cx| this.nav_shortcut(Route::Datasets, w, cx)))
            .on_action(cx.listener(|this, _: &GoRuns, w, cx| this.nav_shortcut(Route::Runs, w, cx)))
            .on_action(cx.listener(|this, _: &GoModels, w, cx| this.nav_shortcut(Route::Models, w, cx)))
            .on_action(
                cx.listener(|this, _: &GoPlayground, w, cx| this.nav_shortcut(Route::Playground, w, cx)),
            )
            .on_action(cx.listener(|this, _: &GoSettings, w, cx| this.nav_shortcut(Route::Settings, w, cx)))
            .on_action(cx.listener(|this, _: &ToggleTheme, _, cx| this.toggle_theme(cx)))
            .on_action(cx.listener(|_, _: &ToggleFullscreen, w, _| w.toggle_fullscreen()))
            .on_action(cx.listener(|this, _: &Back, w, cx| {
                if !this.modal_open() {
                    this.back(w, cx)
                }
            }))
            .on_action(cx.listener(|this, _: &OpenImage, _, cx| {
                if !this.modal_open() {
                    this.pick_image(cx)
                }
            }))
            .on_action(cx.listener(|this, _: &Submit, w, cx| {
                if this.cloud.connect.is_some() {
                    this.submit_connect(w, cx);
                } else if this.new_run.is_some() {
                    this.submit_new_run(w, cx);
                } else if this.synth.is_some() {
                    this.submit_synth(w, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &FocusNext, w, cx| {
                this.menu_open = None;
                this.export_open = false;
                this.kb = true;
                this.cycle_focus(true, w, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &FocusPrev, w, cx| {
                this.menu_open = None;
                this.export_open = false;
                this.kb = true;
                this.cycle_focus(false, w, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &NextRegion, w, cx| this.cycle_region(true, w, cx)))
            .on_action(cx.listener(|this, _: &PrevRegion, w, cx| this.cycle_region(false, w, cx)))
            .on_action(cx.listener(|this, _: &Dismiss, w, cx| this.dismiss(w, cx)))
            // Keyboard mode turns on in the capture phase, before a focused control (a slider, a
            // segmented group) consumes the key, so its first keyboard step already shows the ring.
            .capture_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, _, cx| {
                let nav = matches!(
                    ev.keystroke.key.as_str(),
                    "up" | "down" | "left" | "right" | "home" | "end" | "pageup" | "pagedown"
                );
                if nav && !this.kb {
                    this.kb = true;
                    cx.notify();
                }
            }))
            .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, _, cx| {
                // Enter and Space only when nothing took them (typed into a field they're text).
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") && !this.kb {
                    this.kb = true;
                    cx.notify();
                }
                let m = &ev.keystroke.modifiers;
                // Page Up / Down scroll the page about a screen (less 32 px, so a line of context
                // stays) whenever the focused control didn't use them: a chart group keeps its one
                // Tab stop and the page still pages.
                if matches!(ev.keystroke.key.as_str(), "pageup" | "pagedown")
                    && !this.modal_open()
                    && this.palette.is_none()
                    && !(m.control || m.alt || m.platform || m.shift)
                {
                    let s = &this.page_scroll;
                    let (y, max) = (-s.offset().y, s.max_offset().height);
                    let page = (s.bounds().size.height - px(32.)).max(px(32.));
                    let to = if ev.keystroke.key == "pageup" { y - page } else { y + page };
                    s.set_offset(gpui::point(s.offset().x, -to.clamp(px(0.), max)));
                    cx.stop_propagation();
                    cx.notify();
                    return;
                }
                if this.route == Route::Playground
                    && this.play_image.is_some()
                    && !this.modal_open()
                    && this.palette.is_none()
                    && !(m.control || m.alt || m.platform)
                {
                    let d = if m.shift { 128. } else { 32. };
                    let zoomed = this.play_zoom > 1.0;
                    match ev.keystroke.key.as_str() {
                        // Arrows reach here only when no focused control used them.
                        "left" if zoomed => this.play_pan.0 += d,
                        "right" if zoomed => this.play_pan.0 -= d,
                        "up" if zoomed => this.play_pan.1 += d,
                        "down" if zoomed => this.play_pan.1 -= d,
                        "+" | "=" => this.play_zoom_by(1.4, cx),
                        "-" => this.play_zoom_by(1.0 / 1.4, cx),
                        "0" => {
                            this.play_zoom = 1.0;
                            this.play_pan = (0., 0.);
                        }
                        _ => return,
                    }
                    cx.notify();
                }
            }))
            .capture_any_mouse_down(self.leave_kb(cx))
            .on_mouse_move(cx.listener(Self::on_drag_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_drop(cx.listener(|this, paths: &gpui::ExternalPaths, w, cx| {
                this.play_drop_hover = false;
                // A dialog or the inspector is up: don't navigate out from under it.
                if this.modal_open() {
                    this.toast(
                        Tone::Info,
                        "Close the dialog first",
                        Some("Then drop the image again.".into()),
                        cx,
                    );
                    return;
                }
                // A drop obeys Open image's rule (and says why) before it navigates anywhere.
                if let Some(why) = this.open_image_blocked() {
                    this.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
                    return;
                }
                if let Some(p) = paths.paths().first().cloned() {
                    this.navigate(Route::Playground, w, cx);
                    this.play_file(p, cx);
                }
            }))
            .child(sidebar)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(titlebar)
                    // The page's region anchor for F6 (not a Tab stop itself).
                    .child(
                        div()
                            .id("page-region")
                            .track_focus(&self.fh("page-region", cx).tab_stop(false))
                            .flex_1()
                            .min_h_0()
                            .relative()
                            .child(page)
                            .when_some(key_hint, |d, h| {
                                d.child(key_hint_strip(&t, h).absolute().left_0().right_0().bottom_0())
                            }),
                    ),
            )
            .children(overlays)
            // The top edge resizes the window, as the other three do. Windows only resizes where no
            // caption area is hit, and the title bar is one up to the top: this strip, on top of
            // it and hiding it from the hit test, is not. Maximized or full screen there is no edge,
            // and the top pixels are the title bar's.
            .when(!window.is_maximized() && !window.is_fullscreen(), |d| {
                d.child(
                    div()
                        .id("resize-top")
                        .absolute()
                        .top_0()
                        .left_0()
                        .right_0()
                        .h(px(crate::theme::layout::DESKTOP_RESIZE_EDGE))
                        .occlude()
                        .capture_any_mouse_down(self.leave_kb(cx)),
                )
            })
    }
}

impl FoloApp {
    /// A press leaves keyboard mode (focus rings off). The root listens for presses anywhere, but
    /// the window chrome hides the root from the hit test (`titlebar-drag`), so it listens too.
    /// Only a capture-phase listener: it never prevents the press's default.
    fn leave_kb(&self, cx: &mut Context<Self>) -> Box<dyn Fn(&gpui::MouseDownEvent, &mut Window, &mut App)> {
        Box::new(cx.listener(|this, _: &gpui::MouseDownEvent, _, cx| {
            if this.kb {
                this.kb = false;
                cx.notify();
            }
        }))
    }
    /// A dialog, the stop confirmation or the inspector is up: the page behind is inert.
    pub fn modal_open(&self) -> bool {
        self.new_run.is_some()
            || self.synth.is_some()
            || self.confirm_stop.is_some()
            || self.inspector.is_some()
            || self.cloud.connect.is_some()
            || self.cloud.disconnect.is_some()
            || self.cloud.terminate.is_some()
    }

    fn nav_shortcut(&mut self, route: Route, window: &mut Window, cx: &mut Context<Self>) {
        if !self.modal_open() {
            self.palette = None;
            self.navigate(route, window, cx);
        }
    }

    /// Whether the focused element belongs to the topmost overlay (always true without one).
    fn focus_in_scope(&self, window: &Window, cx: &App) -> bool {
        let prefixes: &[&str] = if self.confirm_stop.is_some() {
            &["stop-", "dialog-close"]
        } else if self.cloud.disconnect.is_some() {
            &["cd-", "dialog-close"]
        } else if self.cloud.terminate.is_some() {
            &["ct-", "dialog-close"]
        } else if self.cloud.connect.is_some() {
            &["cc-", "dialog-close"]
        } else if self.new_run.is_some() {
            &["nr-", "dialog-close"]
        } else if self.synth.is_some() {
            &["syn-", "dialog-close"]
        } else if self.inspector.is_some() {
            &["insp-"]
        } else if self.palette.is_some() {
            &[]
        } else {
            return true;
        };
        let focus = self.focus.borrow();
        if focus.iter().any(|(k, fh)| prefixes.iter().any(|p| k.starts_with(p)) && fh.is_focused(window)) {
            return true;
        }
        if let Some(f) = &self.new_run
            && self.confirm_stop.is_none()
            && [&f.name, &f.epochs_input, &f.lr_input, &f.close_input, &f.hours_input]
                .iter()
                .any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        {
            return true;
        }
        if let Some(f) = &self.cloud.connect
            && self.cloud.disconnect.is_none()
            && self.cloud.terminate.is_none()
            && f.fields.iter().chain([&f.name]).any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        {
            return true;
        }
        if let Some(f) = &self.synth
            && [&f.name, &f.images].iter().any(|i| i.read(cx).focus_handle(cx).is_focused(window))
        {
            return true;
        }
        if let Some(p) = &self.palette {
            return p.input.read(cx).focus_handle(cx).is_focused(window);
        }
        self.inspector.is_some() && self.inspector_focus.is_focused(window)
    }

    /// Tab / Shift-Tab, trapped inside the topmost overlay.
    /// F6 / Shift+F6: the sidebar (the current section's item), the search field, then the page
    /// (its first control). Dialogs and the palette keep focus to themselves.
    fn cycle_region(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.modal_open() || self.palette.is_some() {
            return;
        }
        let nav = format!(
            "nav-{}",
            match self.route.section() {
                Section::Overview => "Overview",
                Section::Datasets => "Datasets",
                Section::Runs => "Runs",
                Section::Models => "Models",
                Section::Playground => "Playground",
                Section::Settings => "Settings",
            }
        );
        // The sidebar region: its navigation and the connection line at its foot.
        let in_nav = self
            .focus
            .borrow()
            .iter()
            .any(|(k, f)| (k.starts_with("nav-") || k.as_ref() == "connection") && f.is_focused(window));
        let in_search = self.fh("titlebar-search", cx).is_focused(window);
        let at = if in_nav {
            0
        } else if in_search {
            1
        } else {
            2
        };
        let next = if forward { (at + 1) % 3 } else { (at + 2) % 3 };
        self.kb = true;
        match next {
            0 => window.focus(&self.fh(nav, cx)),
            1 => window.focus(&self.fh("titlebar-search", cx)),
            _ => {
                window.focus(&self.fh("page-region", cx));
                self.cycle_focus(true, window, cx);
            }
        }
        cx.notify();
    }

    fn cycle_focus(&mut self, forward: bool, window: &mut Window, cx: &mut Context<Self>) {
        // The newly focused control scrolls into view inside whichever of these holds it.
        let containers = if self.inspector.is_some() {
            vec![self.insp_list_scroll.clone()]
        } else if self.new_run.is_some() || self.synth.is_some() || self.cloud.connect.is_some() {
            vec![self.dialog_scroll.clone()]
        } else if matches!(self.route, Route::Playground) {
            vec![self.play_side_scroll.clone(), self.page_scroll.clone()]
        } else {
            vec![self.page_scroll.clone()]
        };
        REVEAL.with(|r| *r.borrow_mut() = Some(containers));
        cx.notify();
        for _ in 0..512 {
            if forward {
                window.focus_next()
            } else {
                window.focus_prev()
            }
            if self.focus_in_scope(window, cx) {
                return;
            }
        }
    }

    /// Ask before stopping a run; the safe choice has focus.
    pub fn ask_stop(&mut self, id: String, window: &mut Window, cx: &mut Context<Self>) {
        self.remember_focus(window, cx);
        self.confirm_stop = Some(id);
        let fh = self.fh("stop-cancel", cx);
        window.focus(&fh);
        cx.notify();
    }

    pub fn open_inspector(
        &mut self,
        dataset: String,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.inspector = Some(Inspector {
            dataset,
            index,
            zoom: 1.0,
            pan: (0., 0.),
            drag_from: None,
            hover_box: None,
            want: None,
            sample: None,
        });
        self.inspector_return = window.focused(cx);
        window.focus(&self.inspector_focus);
        cx.notify();
    }

    /// Opens the inspector on one image of a dataset by id (a run's sample tile): it lands there
    /// once the image list is in, and ←/→ then walk the dataset from it.
    pub fn open_sample(
        &mut self,
        dataset: String,
        image: String,
        run: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ensure_dataset(&dataset, cx);
        let view = self.sample_view;
        self.open_inspector(dataset, 0, window, cx);
        if let Some(i) = self.inspector.as_mut() {
            i.want = Some(image);
            i.sample = Some((run, view));
        }
    }

    pub fn close_inspector(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.inspector = None;
        match self.inspector_return.take() {
            Some(fh) => window.focus(&fh),
            None => window.focus(&self.root_focus),
        }
        cx.notify();
    }

    /// Swap the data source; everything reloads from the new backend.
    pub fn set_backend(&mut self, api: Arc<dyn Backend>, mode: ServerMode, cx: &mut Context<Self>) {
        self.backend_gen += 1;
        // Estimates and speeds are the old server's (and a pending answer will be dropped).
        self.throughput = Load::Idle;
        self.bench_busy = false;
        self.bench_note = None;
        self.nr_estimate = None;
        self.nr_estimate_last = None;
        self.nr_estimate_gen += 1;
        // A draft's dataset belongs to the old server.
        self.new_run_draft = None;
        self.new_run_draft_ds = None;
        // Model ids are per server: loaded local models belong to the old one.
        crate::local::forget_all();
        if !api.local_inference() {
            self.play_target = None;
        }
        self.store.api = api;
        self.server_mode = mode;
        self.subscriptions.clear();
        self.subscribing.clear();
        self.store.clear_pending();
        *self.ds_filter_cache.borrow_mut() = None;
        *self.ds_split_counts.borrow_mut() = None;
        // Ids are only unique per server: cached pixels (and failures) belong to the old one. The
        // textures are freed after this update, when no window is mid-draw, so every window's atlas
        // is reached (this often runs inside a click handler).
        let mut textures = self.store.images_px.clear();
        textures.extend(self.play_image.take().map(|p| p.image));
        cx.defer(move |cx| {
            for img in textures {
                cx.drop_image(img, None);
            }
        });
        self.connecting = None;
        self.hold_loading = false;
        self.store.gpu_util.clear();
        self.play_pred = Load::Idle;
        self.play_pred_pending = false;
        self.play_model = None;
        self.retry_all(cx);
    }

    /// Find a data source: a running `folo serve`, else one we start (`folo serve --mock`),
    /// else the built-in demo data. Runs off the UI thread; the app shows "Connecting…" meanwhile.
    pub fn autoconnect(&mut self, url: String, spawn: bool, cx: &mut Context<Self>) {
        use crate::api::http::{HttpBackend, spawn_mock_server};
        self.hold_loading = true;
        self.connecting = Some(format!("Connecting to {}", url.trim_start_matches("http://")).into());
        // Only the newest attempt may install its result; older ones bail out at each step, and a
        // server an obsolete attempt started is dropped (stopped) with it.
        self.connect_seq += 1;
        let seq = self.connect_seq;
        let live_step = 4180;
        cx.spawn(async move |this, cx| {
            let current = |this: &Self| this.connect_seq == seq;
            let probe = url.clone();
            let found = cx
                .background_executor()
                .spawn(async move { HttpBackend::new(&probe).health().is_ok() })
                .await;
            if found {
                this.update(cx, |this, cx| {
                    if current(this) {
                        this.set_backend(Arc::new(HttpBackend::new(&url)), ServerMode::Local, cx);
                    }
                })
                .ok();
                return;
            }
            if spawn {
                let go_on = this
                    .update(cx, |this, cx| {
                        if current(this) {
                            this.connecting = Some("Starting folo serve --mock…".into());
                            cx.notify();
                        }
                        current(this)
                    })
                    .unwrap_or(false);
                if !go_on {
                    return;
                }
                let started = cx
                    .background_executor()
                    .spawn(async move { spawn_mock_server(std::time::Duration::from_secs(10)) })
                    .await;
                if let Ok(proc_) = started {
                    this.update(cx, |this, cx| {
                        if !current(this) {
                            return; // proc_ drops here: the stray server stops.
                        }
                        let backend = HttpBackend::new(&proc_.url);
                        let url = proc_.url.clone();
                        this.server_proc = Some(proc_);
                        this.server_url.update(cx, |i, cx| i.set_text(url, cx));
                        this.set_backend(Arc::new(backend), ServerMode::Local, cx);
                    })
                    .ok();
                    return;
                }
            }
            this.update(cx, |this, cx| {
                if !current(this) {
                    return;
                }
                this.set_backend(
                    Arc::new(crate::api::mock::MockBackend::new(live_step, false)),
                    ServerMode::Mock,
                    cx,
                );
                this.toast(
                    Tone::Info,
                    "Using built-in demo data",
                    Some("No folo serve was found. Start one and pick it in Settings › Connection.".into()),
                    cx,
                );
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    /// Settings › Data source.
    pub fn switch_source(&mut self, mode: ServerMode, cx: &mut Context<Self>) {
        match mode {
            ServerMode::Mock => {
                // Cancels any connection attempt still in flight.
                self.connect_seq += 1;
                self.server_proc = None;
                self.set_backend(
                    Arc::new(crate::api::mock::MockBackend::new(4180, false)),
                    ServerMode::Mock,
                    cx,
                );
            }
            ServerMode::Local => {
                let url = self.server_url.read(cx).text().to_string();
                let spawn = self.spawn_server;
                self.autoconnect(url, spawn, cx);
            }
        }
    }

    pub fn dismiss(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.menu_open.is_some() || self.export_open {
            self.menu_open = None;
            self.export_open = false;
        } else if self.palette.is_some() {
            self.close_palette(window, cx);
        } else if self.confirm_stop.is_some() {
            self.confirm_stop = None;
            self.restore_focus(window);
        } else if self.cloud.disconnect.is_some() || self.cloud.terminate.is_some() {
            self.close_confirm(window, cx);
        } else if self.cloud.connect.is_some() {
            self.close_connect(window, cx);
        } else if self.new_run.is_some() {
            // Kept as a draft, like Cancel.
            self.new_run_draft = self.new_run.take().and_then(|f| self.keep_draft(f));
            self.restore_focus(window);
        } else if self.synth.is_some() {
            self.synth = None;
            self.restore_focus(window);
        } else if self.inspector.is_some() {
            self.close_inspector(window, cx);
        } else if let Some(p) = self.parent_route() {
            self.navigate(p, window, cx);
        }
        cx.notify();
    }

    /// A page frame: scrollable column with the standard gutter and an optional header.
    /// The page's side gutter: 16 px below 1120 px wide, else 24. Every page, and the breadcrumbs
    /// (which sit 4 px further in for their hover pads), use it.
    pub fn gutter(&self) -> f32 {
        if self.narrow { 16. } else { 24. }
    }

    /// The page's content width: the window less the sidebar (an icon rail below 1120), capped at
    /// `layout.desktop_content_max` (Desktop's own cap, apart from Studio's `size.content_max`;
    /// border-box, so it includes the gutters), less the gutters. Layouts that pick columns by the
    /// room they get measure it here, not by the window width.
    pub fn content_width(&self, window: &Window) -> f32 {
        let vw = f32::from(window.viewport_size().width);
        let side = if vw < 1120. { tsize::SIDEBAR_COLLAPSED } else { tsize::SIDEBAR };
        (vw - side).min(crate::theme::layout::DESKTOP_CONTENT_MAX) - 2. * self.gutter()
    }

    pub fn page_frame(
        &self,
        t: &Theme,
        header: impl IntoElement,
        body: impl IntoElement,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        div()
            .relative()
            .size_full()
            .child(
                div()
                    .id("page")
                    .size_full()
                    .overflow_y_scroll()
                    .track_scroll(&self.page_scroll)
                    .child(
                        div()
                            .w_full()
                            .max_w(px(crate::theme::layout::DESKTOP_CONTENT_MAX))
                            .mx_auto()
                            .flex()
                            .flex_col()
                            .gap(px(16.))
                            .px(px(self.gutter()))
                            .pt(px(20.))
                            .pb(px(32.))
                            .child(header)
                            .child(body),
                    )
                    .child(div().absolute().top_0().left_0().right_0().h(px(1.)).bg(alpha(t.bg_app, 0.0))),
            )
            // The page is a scroll area like the rest: edge fades where content lies past the
            // window (the at-rest cue, thumbs being hidden), and the thumb.
            .children(self.scroll_chrome(self.page_scroll.clone(), t.bg_app, 0., 0., cx))
            .into_any_element()
    }
}

#[cfg(test)]
mod retry_tests {
    use super::{BATCHES, batch_index, carry_over, retry_batch};
    use crate::api::{Augment, TrainConfig};

    #[test]
    fn small_learning_rates_keep_their_digits() {
        assert_eq!(crate::app::format_lr(0.01), "0.010");
        assert_eq!(crate::app::format_lr(0.0001), "0.0001");
        assert!(crate::app::parse_lr(&crate::app::format_lr(0.0001)).is_some());
    }

    #[test]
    fn oom_retries_halve_to_offered_sizes() {
        assert_eq!(retry_batch(16), Some(8));
        assert_eq!(retry_batch(8), Some(4));
        assert_eq!(retry_batch(4), Some(2));
        assert_eq!(retry_batch(2), None);
        assert_eq!(retry_batch(48), Some(16));
        // Whatever the retry picks, the form can show it exactly.
        for b in [2, 4, 8, 16, 32, 64, 128] {
            if let Some(to) = retry_batch(b) {
                assert_eq!(BATCHES[batch_index(to)], to);
            }
        }
        assert_eq!(BATCHES[batch_index(1)], 2);
        assert_eq!(BATCHES[batch_index(500)], 128);
    }

    #[test]
    fn hidden_settings_carry_over() {
        let base = TrainConfig {
            name: "a".into(),
            dataset_id: "ds".into(),
            arch: crate::api::Arch::S,
            epochs: 50,
            batch: 8,
            imgsz: 640,
            device: "auto".into(),
            optimizer: crate::api::Optimizer::Auto,
            lr0: 0.01,
            weights: Some("runs/a/best.folo".into()),
            augment: Augment { hsv: [0.1, 0.2, 0.3], translate: 0.25, ..Augment::default() },
            seed: 42,
            amp: true,
            head: Default::default(),
            compute: None,
        };
        let edited = TrainConfig {
            name: "a-2".into(),
            weights: None,
            seed: 0,
            augment: Augment::default(),
            // The form turned mixed precision off (the CPU, or the user): the base must not undo it.
            amp: false,
            ..base.clone()
        };
        let out = carry_over(Some(&base), edited.clone());
        assert_eq!(out.weights, base.weights);
        assert_eq!(out.seed, 42);
        assert_eq!(out.augment.hsv, [0.1, 0.2, 0.3]);
        assert_eq!(out.augment.translate, 0.25);
        assert_eq!(out.name, "a-2");
        assert!(!out.amp);
        // The form's head stands (it shows the base's and may change it).
        let o2o = TrainConfig { head: crate::api::Head::O2o, ..edited.clone() };
        assert_eq!(carry_over(Some(&base), o2o).head, crate::api::Head::O2o);
        // A fresh form keeps its own values.
        assert_eq!(carry_over(None, edited.clone()), edited);
    }
}

#[cfg(test)]
mod form_tests {
    use super::{parse_epochs, parse_lr};

    #[test]
    fn typed_numbers_validate() {
        assert_eq!(parse_epochs(" 250 "), Some(250));
        assert_eq!(parse_epochs("0"), None);
        assert_eq!(parse_epochs("10001"), None);
        assert_eq!(parse_epochs("12.5"), None);
        assert_eq!(parse_lr("1e-3"), Some(0.001));
        assert_eq!(parse_lr("0.01"), Some(0.01));
        assert_eq!(parse_lr("0"), None);
        assert_eq!(parse_lr("abc"), None);
        assert_eq!(parse_lr("NaN"), None);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_close;

    /// The New run optimizer picker offers every value a client can send, so a run prefilled from one that
    /// trained with MuSGD keeps it; an optimizer this build does not know has no index (the form then shows it
    /// as unknown and blocks Start until one is picked, rather than becoming Auto).
    #[test]
    fn optimizer_picker_offers_every_optimizer() {
        use super::{OPTIMIZERS, optimizer_index};
        use crate::api::Optimizer;
        for o in [Optimizer::Auto, Optimizer::Sgd, Optimizer::Adamw, Optimizer::Musgd] {
            assert_eq!(optimizer_index(o).map(|i| OPTIMIZERS[i]), Some(o), "{o:?}");
        }
        assert_eq!(optimizer_index(Optimizer::Unknown), None);
    }

    #[test]
    fn close_mosaic_capped_as_the_engine_runs_it() {
        // docs/DOGFOOD-3 S2: a 15-epoch run asked for 10 closes mosaic for the last 7.
        assert_eq!(crate::api::capped_close_mosaic(10, 15), 7);
        assert_eq!(crate::api::capped_close_mosaic(10, 100), 10);
        assert_eq!(crate::api::capped_close_mosaic(10, 1), 0);
    }

    #[test]
    fn close_mosaic_bounded_by_epochs() {
        // 200 is invalid at 100 epochs and valid once epochs rise to 300 ...
        assert_eq!(parse_close("200", 100.), None);
        assert_eq!(parse_close("200", 300.), Some(200));
        // ... and a valid value turns invalid when epochs drop below it.
        assert_eq!(parse_close("20", 100.), Some(20));
        assert_eq!(parse_close("20", 10.), None);
        assert_eq!(parse_close(" 7 ", 10.), Some(7));
        assert_eq!(parse_close("x", 10.), None);
    }

    #[test]
    fn synthetic_image_count() {
        use super::parse_images;
        assert_eq!(parse_images("1,000"), Some(1000));
        assert_eq!(parse_images(" 240 "), Some(240));
        assert_eq!(parse_images("0"), None);
        assert_eq!(parse_images("100001"), None);
        assert_eq!(parse_images("ten"), None);
    }
}
