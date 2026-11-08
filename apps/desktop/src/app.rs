//! The root view: window chrome, sidebar navigation, routing, data fetching, the live event pump,
//! keyboard actions and the overlay layers (palette, dialogs, inspector, toasts).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, App, AppContext, Bounds, Context, Entity, FocusHandle, Focusable, FontWeight, InteractiveElement,
    IntoElement, KeyBinding, MouseButton, MouseMoveEvent, MouseUpEvent, ParentElement, Pixels, Render,
    ScrollHandle, SharedString, StatefulInteractiveElement, Styled, Subscription, Task, Window,
    WindowControlArea, actions, div, px,
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
        Back,
        Dismiss,
        OpenImage,
        FocusNext,
        FocusPrev,
        Submit,
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
        KeyBinding::new("alt-left", Back, None),
        KeyBinding::new("escape", Dismiss, None),
        KeyBinding::new("ctrl-o", OpenImage, None),
        KeyBinding::new("tab", FocusNext, None),
        KeyBinding::new("shift-tab", FocusPrev, None),
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
    pub lr0: f32,
    pub mosaic: f32,
    pub mixup: f32,
    pub fliplr: f32,
    pub submitting: bool,
    pub error: Option<ApiError>,
}

pub const ARCHS: [&str; 5] = ["folo-n", "folo-s", "folo-m", "folo-l", "folo-x"];
pub const BATCHES: [u32; 5] = [8, 16, 32, 64, 128];
pub const IMGSZ: [u32; 5] = [320, 480, 640, 800, 1024];
pub const DEVICES: [&str; 3] = ["auto", "gpu:0", "cpu"];
pub const OPTIMIZERS: [&str; 3] = ["auto", "sgd", "adamw"];

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

pub struct FoloApp {
    pub store: Store,
    pub route: Route,
    pub history: Vec<Route>,
    pub mode: Mode,
    pub frozen: bool,
    /// Keep every fetch pending (loading-state screenshots).
    pub hold_loading: bool,

    pub root_focus: FocusHandle,
    pub focus: RefCell<HashMap<SharedString, FocusHandle>>,
    /// The last input came from the keyboard, so focus rings are shown.
    pub kb: bool,
    pub page_scroll: ScrollHandle,

    // Datasets
    pub ds_split: Option<Split>,
    pub ds_class: Option<u32>,
    pub ds_boxes: bool,
    pub ds_tile: f32,
    pub grid_scroll: gpui::UniformListScrollHandle,
    pub grid_width: Rc<std::cell::Cell<Pixels>>,
    pub inspector: Option<Inspector>,
    pub hist_hover: HashMap<&'static str, f32>,

    // Runs
    pub run_filter: RunFilter,
    pub chart_hover: Option<f32>,
    pub sample_view: SampleView,
    pub loss_parts: [bool; 4],
    pub log_scroll: ScrollHandle,
    pub sample_epoch: Option<u32>,

    // Models
    pub export_open: bool,
    pub pr_hover: Option<f32>,
    pub confusion_norm: bool,

    // Playground
    pub play_model: Option<String>,
    pub play_conf: f32,
    pub play_iou: f32,
    pub play_image: Option<PlayImage>,
    pub play_pred: Load<Prediction>,
    pub play_hover: Option<usize>,
    pub play_drop_hover: bool,
    play_seq: u64,

    // Settings
    pub server_mode: ServerMode,
    pub server_url: Entity<TextInput>,
    pub spawn_server: bool,
    pub reduce_motion: bool,

    // Overlays
    pub palette: Option<Palette>,
    pub new_run: Option<NewRunForm>,
    pub confirm_stop: Option<String>,
    pub toasts: Vec<Toast>,
    toast_seq: u64,
    pub menu_open: Option<&'static str>,

    pub drag: Option<Drag>,
    pub slider_bounds: Rc<RefCell<HashMap<&'static str, Bounds<Pixels>>>>,

    subscriptions: HashMap<String, Receiver<RunEvent>>,
    _tasks: Vec<Task<()>>,
    _subs: Vec<Subscription>,
}

impl FoloApp {
    pub fn new(api: Arc<dyn Backend>, frozen: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
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
            root_focus,
            focus: RefCell::new(HashMap::new()),
            kb: false,
            page_scroll: ScrollHandle::new(),
            ds_split: None,
            ds_class: None,
            ds_boxes: true,
            ds_tile: 168.,
            grid_scroll: gpui::UniformListScrollHandle::new(),
            grid_width: Rc::new(std::cell::Cell::new(px(900.))),
            inspector: None,
            hist_hover: HashMap::new(),
            run_filter: RunFilter::All,
            chart_hover: None,
            sample_view: SampleView::Predictions,
            loss_parts: [true, true, true, false],
            log_scroll: ScrollHandle::new(),
            sample_epoch: None,
            export_open: false,
            pr_hover: None,
            confusion_norm: true,
            play_model: None,
            play_conf: 0.25,
            play_iou: 0.7,
            play_image: None,
            play_pred: Load::Idle,
            play_hover: None,
            play_drop_hover: false,
            play_seq: 0,
            server_mode: ServerMode::Mock,
            server_url,
            spawn_server: true,
            reduce_motion: false,
            palette: None,
            new_run: None,
            confirm_stop: None,
            toasts: Vec::new(),
            toast_seq: 0,
            menu_open: None,
            drag: None,
            slider_bounds: Rc::new(RefCell::new(HashMap::new())),
            subscriptions: HashMap::new(),
            _tasks: Vec::new(),
            _subs: Vec::new(),
        };
        if !frozen {
            this.start_pump(cx);
        }
        this
    }

    // ---------- focus ----------

    /// A stable focus handle for a control key; tab order follows paint order.
    pub fn fh(&self, key: impl Into<SharedString>, cx: &mut Context<Self>) -> FocusHandle {
        let key = key.into();
        self.focus
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| cx.focus_handle().tab_stop(true))
            .clone()
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
        self.page_scroll.set_offset(gpui::point(px(0.), px(0.)));
        self.chart_hover = None;
        self.export_open = false;
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
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async move { call(&*api) }).await;
            this.update(cx, |this, cx| {
                apply(this, r, cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    pub fn ensure_core(&mut self, cx: &mut Context<Self>) {
        if self.store.runs.needs_fetch() {
            self.store.runs = Load::Loading;
            self.fetch(cx, |api| api.runs(), |this, r, cx| {
                this.store.runs = r.into();
                this.subscribe_active(cx);
            });
        }
        if self.store.datasets.needs_fetch() {
            self.store.datasets = Load::Loading;
            self.fetch(cx, |api| api.datasets(), |this, r, _| this.store.datasets = r.into());
        }
        if self.store.models.needs_fetch() {
            self.store.models = Load::Loading;
            self.fetch(cx, |api| api.models(), |this, r, _| {
                this.store.models = r.into();
                if this.play_model.is_none() {
                    this.play_model = this.store.models.ready().and_then(|m| m.first()).map(|m| m.id.clone());
                }
            });
        }
        if self.store.system.needs_fetch() {
            self.store.system = Load::Loading;
            self.fetch(cx, |api| api.system(), |this, r, _| {
                if let Ok(s) = &r
                    && let Some(g) = s.gpus.first().and_then(|g| g.util)
                {
                    this.store.gpu_util.push_back(g);
                }
                this.store.system = r.into();
            });
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
            let rid = id.to_string();
            let key = id.to_string();
            self.fetch(cx, move |api| api.metrics(&rid), move |this, r, _| {
                this.store.run_data.insert(key, r.map(|m| RunData::new(m, spe)).into());
            });
        }
    }

    pub fn ensure_samples(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.samples.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.samples.insert(id.to_string(), Load::Loading);
            let (rid, key, epoch) = (id.to_string(), id.to_string(), self.sample_epoch);
            self.fetch(cx, move |api| api.samples(&rid, epoch), move |this, r, _| {
                this.store.samples.insert(key, r.into());
            });
        }
    }

    pub fn ensure_logs(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.logs.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.logs.insert(id.to_string(), Load::Loading);
            let (rid, key) = (id.to_string(), id.to_string());
            self.fetch(cx, move |api| api.logs(&rid), move |this, r, _| {
                this.store.logs.insert(key, r.into());
                this.log_scroll.scroll_to_bottom();
            });
        }
    }

    pub fn ensure_dataset(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.dataset.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.dataset.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(cx, move |api| api.dataset(&a), move |this, r, _| {
                this.store.dataset.insert(b, r.into());
            });
        }
        if self.store.images.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.images.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.images(&a, &ImageQuery { limit: 100_000, ..Default::default() }).map(|p| p.images),
                move |this, r, _| {
                    this.store.images.insert(b, r.into());
                },
            );
        }
    }

    pub fn ensure_eval(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.evals.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.evals.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(cx, move |api| api.model_eval(&a), move |this, r, _| {
                this.store.evals.insert(b, r.into());
            });
        }
    }

    /// A decoded image, requesting it if needed.
    pub fn image(&mut self, dataset: &str, image: &str, kind: ImgKind, cx: &mut Context<Self>) -> Option<Arc<gpui::RenderImage>> {
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
        let thumb = (kind == ImgKind::Thumb).then_some(320);
        cx.spawn(async move |this, cx| {
            let k2 = key.clone();
            let img = cx
                .background_executor()
                .spawn(async move { api.image(&k2.dataset, &k2.image, thumb).ok().and_then(to_render_image) })
                .await;
            this.update(cx, |this, cx| {
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
    pub fn subscribe_active(&mut self, _cx: &mut Context<Self>) {
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
            if self.subscriptions.contains_key(&id) {
                continue;
            }
            if let Ok(rx) = self.store.api.events(&id) {
                self.subscriptions.insert(id, rx);
            }
        }
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
                        for id in ended {
                            this.subscriptions.remove(&id);
                        }
                        for (id, status) in finished {
                            let name = this.store.run(&id).map(|r| r.name.clone()).unwrap_or(id.clone());
                            if status == RunStatus::Completed {
                                this.toast(Tone::Success, format!("{name} finished"), Some("Best weights saved to Models.".into()), cx);
                                this.store.models = Load::Idle;
                            } else {
                                this.toast(Tone::Danger, format!("{name} failed"), None, cx);
                            }
                        }
                        if tick % 20 == 0 {
                            this.store.system = Load::Idle;
                            this.ensure_core(cx);
                        }
                        if tick % 50 == 0 && matches!(this.store.runs, Load::Ready(_)) {
                            this.fetch(cx, |api| api.runs(), |this, r, cx| {
                                if let Ok(runs) = r {
                                    this.store.runs = Load::Ready(runs);
                                    this.subscribe_active(cx);
                                }
                            });
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

    pub fn toast(&mut self, tone: Tone, title: impl Into<SharedString>, body: Option<SharedString>, cx: &mut Context<Self>) {
        self.toast_seq += 1;
        let id = self.toast_seq;
        self.toasts.push(Toast { id, tone, title: title.into(), body });
        if self.toasts.len() > 3 {
            self.toasts.remove(0);
        }
        if !self.frozen {
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(Duration::from_millis(4200)).await;
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

    pub fn open_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.palette.is_some() {
            self.close_palette(window, cx);
            return;
        }
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
        self.palette = Some(Palette { input, query: String::new(), selected: 0, _sub: sub });
        cx.notify();
    }

    pub fn close_palette(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.palette = None;
        window.focus(&self.root_focus);
        cx.notify();
    }

    pub fn open_new_run(&mut self, dataset: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        let idx = dataset
            .and_then(|id| self.store.datasets.ready()?.iter().position(|d| d.id == id))
            .unwrap_or(0);
        let n = self.store.runs.ready().map_or(0, |r| r.len()) + 1;
        let ds_name = self.store.datasets.ready().and_then(|d| d.get(idx)).map_or("shapes".to_string(), |d| d.name.clone());
        let name = cx.new(|cx| {
            let mut i = TextInput::new("run name", cx);
            i.set_text(format!("{ds_name}-s-{n}"), cx);
            i
        });
        let fh = name.read(cx).focus_handle(cx);
        self.new_run = Some(NewRunForm {
            name,
            dataset: idx,
            arch: 1,
            epochs: 100.0,
            batch: 1,
            imgsz: 2,
            device: 0,
            optimizer: 0,
            lr0: 0.01,
            mosaic: 1.0,
            mixup: 0.0,
            fliplr: 0.5,
            submitting: false,
            error: None,
        });
        window.focus(&fh);
        cx.notify();
    }

    pub fn submit_new_run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(form) = self.new_run.as_mut() else { return };
        if form.submitting {
            return;
        }
        let Some(ds) = self.store.datasets.ready().and_then(|d| d.get(form.dataset)).cloned() else { return };
        let name = form.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            form.error = Some(ApiError::new("invalid_name", "Give the run a name."));
            cx.notify();
            return;
        }
        form.submitting = true;
        let config = TrainConfig {
            name,
            dataset_id: ds.id.clone(),
            arch: ARCHS[form.arch].into(),
            epochs: form.epochs.round() as u32,
            batch: BATCHES[form.batch],
            imgsz: IMGSZ[form.imgsz],
            device: DEVICES[form.device].into(),
            optimizer: OPTIMIZERS[form.optimizer].into(),
            lr0: form.lr0,
            weights: None,
            augment: Augment { mosaic: form.mosaic, mixup: form.mixup, fliplr: form.fliplr, ..Augment::default() },
            seed: 0,
        };
        let _ = window;
        self.fetch(cx, move |api| api.create_run(config), |this, r, cx| match r {
            Ok(run) => {
                this.new_run = None;
                let id = run.id.clone();
                let name = run.name.clone();
                if let Some(runs) = this.store.runs.ready_mut() {
                    runs.insert(0, run);
                }
                this.subscribe_active(cx);
                this.toast(Tone::Accent, format!("Training {name}"), Some("Live metrics stream into the dashboard.".into()), cx);
                let prev = std::mem::replace(&mut this.route, Route::Run(id));
                this.history.push(prev);
            }
            Err(e) => {
                if let Some(f) = this.new_run.as_mut() {
                    f.submitting = false;
                    f.error = Some(e);
                }
            }
        });
        cx.notify();
    }

    pub fn stop_run(&mut self, id: String, cx: &mut Context<Self>) {
        self.confirm_stop = None;
        let rid = id.clone();
        if let Some(runs) = self.store.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == id)
        {
            r.status = RunStatus::Stopping;
        }
        self.fetch(cx, move |api| api.stop_run(&rid), move |this, r, cx| match r {
            Ok(run) => {
                let name = run.name.clone();
                let epoch = run.epoch;
                if let Some(runs) = this.store.runs.ready_mut()
                    && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                {
                    *slot = run;
                }
                this.subscriptions.remove(&id);
                this.store.logs.insert(id.clone(), Load::Idle);
                this.toast(Tone::Neutral, format!("Stopped {name}"), Some(format!("Last checkpoint kept (epoch {epoch}).").into()), cx);
            }
            Err(e) => this.toast(Tone::Danger, "Could not stop the run", Some(e.message.into()), cx),
        });
        cx.notify();
    }

    pub fn export_model(&mut self, id: String, format: &'static str, cx: &mut Context<Self>) {
        self.export_open = false;
        self.fetch(cx, move |api| api.export_model(&id, format), |this, r, cx| match r {
            Ok(res) => {
                let (v, u) = bytes(res.bytes);
                this.toast(Tone::Success, "Export complete", Some(format!("{} · {v} {u}", res.path).into()), cx);
            }
            Err(e) => this.toast(Tone::Danger, "Export failed", Some(e.message.into()), cx),
        });
    }

    // ---------- playground ----------

    pub fn play_dataset_image(&mut self, dataset: &str, image_id: &str, cx: &mut Context<Self>) {
        let input = PredictInput::Dataset { dataset_id: dataset.to_string(), image_id: image_id.to_string() };
        let (d, i) = (dataset.to_string(), image_id.to_string());
        let name: SharedString = format!("{} / {}", self.store.dataset_summary(dataset).map_or(dataset, |d| d.name.as_str()), image_id).into();
        self.play_seq += 1;
        let seq = self.play_seq;
        self.fetch(
            cx,
            move |api| {
                let data = api.image(&d, &i, None)?;
                let (w, h) = match &data {
                    ImageData::Rgba { width, height, .. } => (*width, *height),
                    ImageData::Encoded(b) => image::load_from_memory(b).map(|i| (i.width(), i.height())).unwrap_or((0, 0)),
                };
                to_render_image(data).map(|img| (img, w, h)).ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))
            },
            move |this, r, cx| {
                if seq != this.play_seq {
                    return;
                }
                match r {
                    Ok((image, width, height)) => {
                        this.play_image = Some(PlayImage { name, input, image, width, height });
                        this.run_predict(cx);
                    }
                    Err(e) => this.toast(Tone::Danger, "Could not open the image", Some(e.message.into()), cx),
                }
            },
        );
    }

    pub fn play_file(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        let name: SharedString = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default().into();
        let n2 = name.to_string();
        self.play_seq += 1;
        let seq = self.play_seq;
        self.fetch(
            cx,
            move |_| {
                let bytes = std::fs::read(&path).map_err(|e| ApiError::new("read_failed", format!("Could not read {n2}: {e}")))?;
                let img = image::load_from_memory(&bytes)
                    .map_err(|_| ApiError::new("unsupported_image", format!("{n2} is not an image FOLO can read")).with_hint("Use a JPEG or PNG."))?;
                let (w, h) = (img.width(), img.height());
                let render = to_render_image(ImageData::Rgba { width: w, height: h, pixels: img.to_rgba8().into_raw() })
                    .ok_or_else(|| ApiError::new("decode_failed", "Could not decode the image"))?;
                Ok((Arc::new(bytes), render, w, h))
            },
            move |this, r, cx| {
                if seq != this.play_seq {
                    return;
                }
                match r {
                    Ok((bytes, image, width, height)) => {
                        this.play_image = Some(PlayImage { name: name.clone(), input: PredictInput::File { name: name.to_string(), bytes }, image, width, height });
                        this.run_predict(cx);
                    }
                    Err(e) => this.toast(Tone::Danger, e.message.clone(), e.hint.map(Into::into), cx),
                }
            },
        );
    }

    pub fn run_predict(&mut self, cx: &mut Context<Self>) {
        let (Some(model), Some(img)) = (self.play_model.clone(), self.play_image.as_ref()) else { return };
        // Fetch everything above the lowest threshold once; the conf slider filters locally.
        let req = PredictRequest { model_id: model, input: img.input.clone(), conf: 0.05, iou: self.play_iou, max_det: 300 };
        if !matches!(self.play_pred, Load::Ready(_)) {
            self.play_pred = Load::Loading;
        }
        self.play_seq += 1;
        let seq = self.play_seq;
        self.fetch(cx, move |api| api.predict(&req), move |this, r, _| {
            if seq == this.play_seq {
                this.play_pred = r.into();
            }
        });
    }

    pub fn pick_image(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(gpui::PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Open image".into()) });
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
        let Some(drag) = self.drag.as_ref() else { return };
        let Some(b) = self.slider_bounds.borrow().get(drag.id).copied() else { return };
        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
        let id = drag.id;
        self.set_slider(id, frac, cx);
    }

    fn on_drag_end(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
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
        let win_btn = |id: &'static str, icon_path: &'static str, area: WindowControlArea, close: bool| {
            div()
                .id(id)
                .group(id)
                .w(px(46.))
                .h_full()
                .flex()
                .items_center()
                .justify_center()
                .window_control_area(area)
                .hover(move |s| if close { s.bg(gpui::rgb(0xC42B1C)) } else { s.bg(t.bg_hover) })
                .child(
                    gpui::svg()
                        .path(icon_path)
                        .size(px(10.))
                        .flex_none()
                        .text_color(t.text_secondary)
                        .group_hover(id, move |s| s.text_color(if close { gpui::white() } else { t.text_primary })),
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
                    .pl(px(20.))
                    .window_control_area(WindowControlArea::Drag)
                    .children(crumbs.into_iter().enumerate().map(|(i, (label, route))| {
                        let last = route.is_none();
                        div()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .when(i > 0, |d| d.child(icon("icons/chevron-right.svg", 12., t.text_disabled)))
                            .child(
                                div()
                                    .id(("crumb", i))
                                    .px(px(4.))
                                    .py(px(2.))
                                    .rounded(px(4.))
                                    .child(
                                        text(ty::BASE, if last { t.text_primary } else { t.text_tertiary })
                                            .when(last, |d| d.font_weight(FontWeight::MEDIUM))
                                            .whitespace_nowrap()
                                            .child(label),
                                    )
                                    .when_some(route, |d, r| {
                                        d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).on_click(cx.listener(move |this, _, w, cx| this.navigate(r.clone(), w, cx)))
                                    }),
                            )
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
            .child(win_btn("win-min", "icons/win-min.svg", WindowControlArea::Min, false))
            .child(win_btn(
                "win-max",
                if maximized { "icons/win-restore.svg" } else { "icons/win-max.svg" },
                WindowControlArea::Max,
                false,
            ))
            .child(win_btn("win-close", "icons/win-close.svg", WindowControlArea::Close, true))
    }

    fn breadcrumbs(&self) -> Vec<(SharedString, Option<Route>)> {
        match &self.route {
            Route::Overview => vec![("Overview".into(), None)],
            Route::Datasets => vec![("Datasets".into(), None)],
            Route::Dataset(id) => vec![
                ("Datasets".into(), Some(Route::Datasets)),
                (self.store.dataset_summary(id).map_or(id.clone(), |d| d.name.clone()).into(), None),
            ],
            Route::Runs => vec![("Runs".into(), None)],
            Route::Run(id) => vec![("Runs".into(), Some(Route::Runs)), (self.store.run(id).map_or(id.clone(), |r| r.name.clone()).into(), None)],
            Route::Models => vec![("Models".into(), None)],
            Route::Model(id) => vec![("Models".into(), Some(Route::Models)), (self.store.model(id).map_or(id.clone(), |m| m.name.clone()).into(), None)],
            Route::Playground => vec![("Playground".into(), None)],
            Route::Settings => vec![("Settings".into(), None)],
        }
    }

    fn sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let section = self.route.section();
        let live = self.store.live_run().cloned();
        let active_count = self.store.runs.ready().map_or(0, |r| r.iter().filter(|r| r.status.is_active()).count());
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
                self.nav_item(label, icon_path, section == sec, route, key, badge, window, cx)
            })
            .collect();
        let settings = self.nav_item("Settings", "icons/settings.svg", section == Section::Settings, Route::Settings, ",", None, window, cx);
        let mut live_card = None;
        if let Some(r) = live {
            live_card = Some(self.live_mini(&r, window, cx));
        }
        let health = self.connection_line(cx);
        div()
            .w(px(tsize::SIDEBAR))
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
                    .px(px(16.))
                    .window_control_area(WindowControlArea::Drag)
                    .child(logo(&t, 20.))
                    .child(
                        div()
                            .flex()
                            .items_baseline()
                            .gap(px(6.))
                            .child(text(ty::MD, t.text_primary).font_weight(FontWeight::SEMIBOLD).child("FOLO"))
                            .child(text(ty::MD, t.text_tertiary).child("Desktop")),
                    ),
            )
            .child(div().flex().flex_col().gap(px(2.)).px(px(8.)).pt(px(8.)).children(items))
            .child(div().flex_1())
            .children(live_card)
            .child(div().px(px(8.)).pb(px(4.)).child(settings))
            .child(health)
    }

    #[allow(clippy::too_many_arguments)]
    fn nav_item(&mut self, label: &'static str, icon_path: &'static str, active: bool, route: Route, key: &'static str, badge: Option<usize>, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
            .on_click(cx.listener(move |this, _, w, cx| this.navigate(route.clone(), w, cx)))
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(r2.clone(), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, 6.)))
            .child(icon(icon_path, 16., if active { t.text_primary } else { t.text_tertiary }))
            .child(
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
                        .child(dot(t.accent_solid, 6., true, "nav-live"))
                        .child(text(ty::XS, t.accent_text).tnum().font_weight(FontWeight::MEDIUM).child(format!("{n} live"))),
                )
            })
            .when(badge.is_none(), |d| d.child(div().opacity(if active { 0.0 } else { 0.0 }).child(shortcut(&t, &["Ctrl", key]))))
            .into_any_element()
    }

    fn live_mini(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let id = r.id.clone();
        let fh = self.fh("live-mini", cx);
        let ring = self.ring(&fh, window);
        let map = r.best.as_ref().map_or("–".to_string(), |b| f3(b.map50_95));
        let id2 = id.clone();
        div()
            .id("live-mini")
            .track_focus(&fh)
            .relative()
            .mx(px(8.))
            .mb(px(8.))
            .p(px(12.))
            .flex()
            .flex_col()
            .gap(px(10.))
            .rounded(px(8.))
            .bg(t.bg_raised)
            .border_1()
            .border_color(t.line_subtle)
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_default))
            .on_click(cx.listener(move |this, _, w, cx| this.navigate(Route::Run(id.clone()), w, cx)))
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, w, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    this.navigate(Route::Run(id2.clone()), w, cx);
                    cx.stop_propagation();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, 8.)))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(dot(t.accent_solid, 6., true, "mini"))
                    .child(text(ty::XS, t.accent_text).font_weight(FontWeight::MEDIUM).child("TRAINING"))
                    .child(div().flex_1())
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!("{:.0}%", r.progress() * 100.0))),
            )
            .child(text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).truncate().child(r.name.clone()))
            .child(progress(&t, r.progress(), t.accent_solid, 3.))
            .child(
                div()
                    .flex()
                    .justify_between()
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!("Epoch {} / {}", r.epoch, r.epochs)))
                    .child(text(ty::XS, t.text_secondary).tnum().child(format!("mAP {map}"))),
            )
            .into_any_element()
    }

    fn connection_line(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        let (color, label) = match &self.store.health {
            Load::Ready(h) if h.mock => (t.accent_solid, format!("{} · v{}", self.store.api.describe(), h.version)),
            Load::Ready(h) => (t.success_solid, format!("{} · v{}", self.store.api.describe(), h.version)),
            Load::Failed(_) => (t.danger_solid, "Server unreachable".to_string()),
            _ => (t.text_disabled, "Connecting…".to_string()),
        };
        div()
            .h(px(36.))
            .flex_none()
            .flex()
            .items_center()
            .gap(px(8.))
            .px(px(16.))
            .border_t_1()
            .border_color(t.line_subtle)
            .child(div().size(px(6.)).rounded_full().bg(color))
            .child(text(ty::XS, t.text_tertiary).truncate().child(label))
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
}

impl Render for FoloApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        self.ensure_core(cx);
        let titlebar = self.titlebar(window, cx).into_any_element();
        let sidebar = self.sidebar(window, cx).into_any_element();
        let page = self.page(window, cx);
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
                this.palette = None;
                this.open_new_run(None, w, cx)
            }))
            .on_action(cx.listener(|this, _: &GoOverview, w, cx| this.navigate(Route::Overview, w, cx)))
            .on_action(cx.listener(|this, _: &GoDatasets, w, cx| this.navigate(Route::Datasets, w, cx)))
            .on_action(cx.listener(|this, _: &GoRuns, w, cx| this.navigate(Route::Runs, w, cx)))
            .on_action(cx.listener(|this, _: &GoModels, w, cx| this.navigate(Route::Models, w, cx)))
            .on_action(cx.listener(|this, _: &GoPlayground, w, cx| this.navigate(Route::Playground, w, cx)))
            .on_action(cx.listener(|this, _: &GoSettings, w, cx| this.navigate(Route::Settings, w, cx)))
            .on_action(cx.listener(|this, _: &ToggleTheme, _, cx| this.toggle_theme(cx)))
            .on_action(cx.listener(|this, _: &Back, w, cx| this.back(w, cx)))
            .on_action(cx.listener(|this, _: &OpenImage, _, cx| this.pick_image(cx)))
            .on_action(cx.listener(|this, _: &Submit, w, cx| {
                if this.new_run.is_some() {
                    this.submit_new_run(w, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &FocusNext, w, cx| {
                this.kb = true;
                w.focus_next();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &FocusPrev, w, cx| {
                this.kb = true;
                w.focus_prev();
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &Dismiss, w, cx| this.dismiss(w, cx)))
            .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, _, cx| {
                if matches!(ev.keystroke.key.as_str(), "up" | "down" | "left" | "right" | "enter" | "space") && !this.kb {
                    this.kb = true;
                    cx.notify();
                }
            }))
            .capture_any_mouse_down(cx.listener(|this, _, _, cx| {
                if this.kb {
                    this.kb = false;
                    cx.notify();
                }
            }))
            .on_mouse_move(cx.listener(Self::on_drag_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_drop(cx.listener(|this, paths: &gpui::ExternalPaths, w, cx| {
                this.play_drop_hover = false;
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
                    .child(div().flex_1().min_h_0().relative().child(page)),
            )
            .children(overlays)
    }
}

impl FoloApp {
    pub fn dismiss(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.menu_open.is_some() || self.export_open {
            self.menu_open = None;
            self.export_open = false;
        } else if self.palette.is_some() {
            self.close_palette(window, cx);
        } else if self.confirm_stop.is_some() {
            self.confirm_stop = None;
        } else if self.new_run.is_some() {
            self.new_run = None;
            window.focus(&self.root_focus);
        } else if self.inspector.is_some() {
            self.inspector = None;
        } else if self.parent_route().is_some() {
            if let Some(p) = self.parent_route() {
                self.navigate(p, window, cx);
            }
        }
        cx.notify();
    }

    /// A page frame: scrollable column with the standard gutter and an optional header.
    pub fn page_frame(&self, t: &Theme, header: impl IntoElement, body: impl IntoElement) -> AnyElement {
        div()
            .id("page")
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&self.page_scroll)
            .child(
                div()
                    .w_full()
                    .max_w(px(tsize::CONTENT_MAX))
                    .mx_auto()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .px(px(24.))
                    .pt(px(20.))
                    .pb(px(32.))
                    .child(header)
                    .child(body),
            )
            .child(div().absolute().top_0().left_0().right_0().h(px(1.)).bg(alpha(t.bg_app, 0.0)))
            .into_any_element()
    }
}
