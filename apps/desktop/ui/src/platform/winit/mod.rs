//! The winit + softbuffer platform backend (Linux, macOS, and Windows with feature `winit`).
//!
//! `WinitPlatform::new(text_system)` builds the event loop on the calling (main) thread; `run` drives it
//! with `ControlFlow::Wait`, so an idle app sleeps. Windows present a `Framebuffer` through softbuffer;
//! the app's own caption is driven by its window control areas (`chrome.rs`). `FOLO_UI_OFFSCREEN=1` opens
//! windows outside every display and never activates them; `FOLO_UI_TRACE=1` prints `first-present <ms>`.

mod chrome;
mod dispatcher;
mod display;
mod events;
mod keys;
mod mouse;
mod present;
mod process;
mod system;
mod window;

pub use dispatcher::{MAX_BACKGROUND_THREADS, WinitDispatcher};
pub use display::WinitDisplay;
pub use window::WinitWindow;

use crate::{
    ClipboardItem, CursorStyle, PathPromptOptions, Platform, PlatformDispatcher, PlatformDisplay,
    PlatformTextSystem, PlatformWindow, WindowAppearance, WindowParams, oneshot,
};
use anyhow::{Context as _, Result, anyhow};
use parking_lot::Mutex;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

/// The event loop's user event: the dispatcher has main-thread work.
#[derive(Debug, Clone, Copy)]
pub(crate) enum UserEvent {
    Wake,
}

/// Whether `FOLO_UI_OFFSCREEN=1` is set.
pub fn offscreen_requested() -> bool {
    std::env::var_os("FOLO_UI_OFFSCREEN").is_some_and(|v| v == "1")
}

/// The main-thread state the platform, its windows and the loop share.
struct Shared {
    /// The running loop, while one of its callbacks runs (see `enter`).
    active_loop: Cell<*const ActiveEventLoop>,
    /// The loop before `run` takes it.
    pending_loop: RefCell<Option<EventLoop<UserEvent>>>,
    windows: RefCell<HashMap<WindowId, Weak<window::WindowInner>>>,
    dispatcher: Arc<WinitDispatcher>,
    cursor: Cell<CursorStyle>,
    clipboard: RefCell<system::Clipboard>,
    /// Jobs for the main thread outside any callback (macOS file dialogs).
    main_jobs: RefCell<Vec<Box<dyn FnOnce() + Send>>>,
    offscreen: bool,
    quit: Cell<bool>,
}

/// Restores the previous loop pointer when a callback ends.
struct LoopGuard<'a> {
    shared: &'a Shared,
    previous: *const ActiveEventLoop,
}

impl Drop for LoopGuard<'_> {
    fn drop(&mut self) {
        self.shared.active_loop.set(self.previous);
    }
}

impl Shared {
    fn enter<'a>(&'a self, event_loop: &'a ActiveEventLoop) -> LoopGuard<'a> {
        let previous = self.active_loop.replace(event_loop as *const ActiveEventLoop);
        LoopGuard { shared: self, previous }
    }

    /// Calls `f` with the running loop, if a loop callback is on the stack.
    fn with_loop<R>(&self, f: impl FnOnce(&ActiveEventLoop) -> R) -> Option<R> {
        let event_loop = self.active_loop.get();
        if event_loop.is_null() {
            return None;
        }
        // SAFETY: the pointer is set only by `enter`, from a `&ActiveEventLoop` that a winit callback
        // holds for as long as the returned guard lives, and the guard resets it before that callback
        // returns; so a non-null pointer always refers to a live loop on this (main) thread.
        Some(f(unsafe { &*event_loop }))
    }

    fn live_windows(&self) -> Vec<Rc<window::WindowInner>> {
        let mut windows = self.windows.borrow_mut();
        windows.retain(|_, w| w.strong_count() > 0);
        windows.values().filter_map(Weak::upgrade).collect()
    }

    fn window(&self, id: WindowId) -> Option<Rc<window::WindowInner>> {
        self.windows.borrow().get(&id).and_then(Weak::upgrade)
    }

    fn monitors(&self) -> Vec<winit::monitor::MonitorHandle> {
        if let Some(monitors) = self.with_loop(|el| el.available_monitors().collect()) {
            return monitors;
        }
        self.live_windows().first().map(|w| w.window.available_monitors().collect()).unwrap_or_default()
    }
}

/// The winit platform (see the module docs).
pub struct WinitPlatform {
    shared: Rc<Shared>,
    text_system: Arc<dyn PlatformTextSystem>,
}

impl WinitPlatform {
    /// Creates the event loop on this thread (which becomes the main thread) and the dispatcher.
    pub fn new(text_system: Arc<dyn PlatformTextSystem>) -> Result<Self> {
        process::mark_start();
        let event_loop =
            EventLoop::<UserEvent>::with_user_event().build().context("creating the winit event loop")?;
        let proxy = Mutex::new(event_loop.create_proxy());
        let dispatcher = Arc::new(WinitDispatcher::new(Box::new(move || {
            proxy.lock().send_event(UserEvent::Wake).is_ok()
        })));
        Ok(Self {
            shared: Rc::new(Shared {
                active_loop: Cell::new(std::ptr::null()),
                pending_loop: RefCell::new(Some(event_loop)),
                windows: RefCell::new(HashMap::new()),
                dispatcher,
                cursor: Cell::new(CursorStyle::Arrow),
                clipboard: RefCell::new(system::Clipboard::default()),
                main_jobs: RefCell::new(Vec::new()),
                offscreen: offscreen_requested(),
                quit: Cell::new(false),
            }),
            text_system,
        })
    }

    /// The concrete dispatcher (for `drain_main` in tests and tools).
    pub fn winit_dispatcher(&self) -> Arc<WinitDispatcher> {
        self.shared.dispatcher.clone()
    }

    fn run_dialog(&self) -> impl Fn(Box<dyn FnOnce() + Send>) + '_ {
        move |job| {
            if cfg!(target_os = "macos") {
                // AppKit panels must come from the main thread, outside the caller's borrows.
                self.shared.main_jobs.borrow_mut().push(job);
            } else {
                system::on_thread(job);
            }
        }
    }
}

impl Platform for WinitPlatform {
    fn dispatcher(&self) -> Arc<dyn PlatformDispatcher> {
        self.shared.dispatcher.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>) {
        let Some(event_loop) = self.shared.pending_loop.borrow_mut().take() else {
            log_error("WinitPlatform::run called twice");
            return;
        };
        event_loop.set_control_flow(ControlFlow::Wait);
        let mut app =
            LoopHandler { shared: self.shared.clone(), on_finish_launching: Some(on_finish_launching) };
        if let Err(error) = event_loop.run_app(&mut app) {
            log_error(&format!("winit event loop: {error}"));
        }
    }

    fn quit(&self) {
        self.shared.quit.set(true);
        self.shared.with_loop(|el| el.exit());
    }

    fn activate(&self, _ignoring_other_apps: bool) {
        if self.shared.offscreen {
            return;
        }
        if let Some(window) = self.shared.live_windows().first() {
            window.window.focus_window();
        }
    }

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        self.shared
            .monitors()
            .into_iter()
            .enumerate()
            .map(|(i, m)| Rc::new(WinitDisplay::new(i, m)) as Rc<dyn PlatformDisplay>)
            .collect()
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        let monitors = self.shared.monitors();
        let primary = self.shared.with_loop(|el| el.primary_monitor()).flatten();
        let index = primary.and_then(|p| monitors.iter().position(|m| *m == p)).unwrap_or(0);
        let monitor = monitors.into_iter().nth(index)?;
        Some(Rc::new(WinitDisplay::new(index, monitor)))
    }

    fn open_window(&self, params: WindowParams) -> Result<Box<dyn PlatformWindow>> {
        let monitors = self.shared.monitors();
        let params = place_on_display(params, &monitors);
        let offscreen = self.shared.offscreen;
        let origin = offscreen.then(|| window::offscreen_origin(monitors.into_iter()));
        let attributes = window::window_attributes(&params, origin);
        let created = match self.shared.with_loop(|el| el.create_window(attributes.clone())) {
            Some(window) => window,
            None => match self.shared.pending_loop.borrow().as_ref() {
                #[allow(deprecated)]
                Some(event_loop) => event_loop.create_window(attributes),
                None => return Err(anyhow!("open_window outside the event loop")),
            },
        };
        let window = created.context("creating a winit window")?;
        let id = window.id();
        let inner = window::WindowInner::new(window, &params, offscreen);
        inner.apply_cursor(self.shared.cursor.get());
        let shared = Rc::downgrade(&self.shared);
        inner.on_closed.set(Some(Box::new(move || {
            if let Some(shared) = shared.upgrade() {
                shared.windows.borrow_mut().remove(&id);
            }
        })));
        let platform_window = WinitWindow { inner };
        self.shared.windows.borrow_mut().insert(id, platform_window.downgrade());
        Ok(Box::new(platform_window))
    }

    fn window_appearance(&self) -> WindowAppearance {
        let theme = self
            .shared
            .with_loop(|el| el.system_theme())
            .flatten()
            .or_else(|| self.shared.live_windows().first().and_then(|w| w.window.theme()));
        window::appearance(theme)
    }

    fn open_url(&self, url: &str) {
        system::spawn_detached(system::open_url_command(url));
    }

    fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        system::prompt_for_paths(options, &self.run_dialog())
    }

    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        system::prompt_for_new_path(directory, suggested_name, &self.run_dialog())
    }

    fn reveal_path(&self, path: &Path) {
        system::spawn_detached(system::reveal_path_command(path));
    }

    fn set_cursor_style(&self, style: CursorStyle) {
        self.shared.cursor.set(style);
        for window in self.shared.live_windows() {
            window.apply_cursor(style);
        }
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        self.shared.clipboard.borrow_mut().write(item);
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.shared.clipboard.borrow_mut().read()
    }
}

/// Moves the window onto the display `params.display_id` names when its bounds are not on it (centred,
/// keeping its size).
fn place_on_display(mut params: WindowParams, monitors: &[winit::monitor::MonitorHandle]) -> WindowParams {
    let Some(id) = params.display_id else { return params };
    let Some(monitor) = monitors.get(u32::from(id) as usize) else { return params };
    let display = WinitDisplay::new(u32::from(id) as usize, monitor.clone());
    let area = display.bounds();
    if !area.contains(&params.bounds.center()) {
        let size = params.bounds.size.min(&area.size);
        params.bounds =
            crate::Bounds::new(area.center() - crate::point(size.width / 2., size.height / 2.), size);
    }
    params
}

fn log_error(message: &str) {
    eprintln!("folo-ui (winit): {message}");
}

/// The `ApplicationHandler`: routes winit's callbacks to the windows and the dispatcher.
struct LoopHandler {
    shared: Rc<Shared>,
    on_finish_launching: Option<Box<dyn FnOnce()>>,
}

impl ApplicationHandler<UserEvent> for LoopHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let _guard = self.shared.enter(event_loop);
        if let Some(on_finish_launching) = self.on_finish_launching.take() {
            on_finish_launching();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, UserEvent::Wake: UserEvent) {
        let _guard = self.shared.enter(event_loop);
        self.shared.dispatcher.drain_main();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let _guard = self.shared.enter(event_loop);
        if let Some(window) = self.shared.window(window_id) {
            window.handle_event(event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let _guard = self.shared.enter(event_loop);
        let jobs = std::mem::take(&mut *self.shared.main_jobs.borrow_mut());
        for job in jobs {
            job();
        }
        let now = Instant::now();
        let mut wake_at: Option<Instant> = None;
        for window in self.shared.live_windows() {
            window.end_batch();
            if let Some(due) = window.schedule_frame(now) {
                wake_at = Some(wake_at.map_or(due, |at| at.min(due)));
            }
        }
        if self.shared.quit.get() {
            event_loop.exit();
            return;
        }
        // No timer unless a frame is waiting for the next refresh: an idle app sleeps.
        event_loop.set_control_flow(wake_at.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
    }
}
