//! The headless platform: no OS window, fixed size and scale, synthetic input, renders to a `Framebuffer`.
//! Owner tag: `"elements"`. Driven by `HeadlessApp` (tests, `folo-desktop --shot`, CI).
//!
//! Determinism: the dispatcher never starts a thread. Main-thread, background and delayed runnables wait
//! in queues that `HeadlessDispatcher::run_until_idle` pumps on the calling thread, and its clock moves only
//! when the test calls `advance_clock`. A window's `request_frame` just records that a frame is wanted.
#![allow(dead_code)] // skeleton

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use async_task::Runnable;
use parking_lot::Mutex;

use crate::{
    Bounds, ClipboardItem, CursorStyle, DevicePixels, DispatchEventResult, DisplayId, Framebuffer, Modifiers,
    Pixels, Platform, PlatformDispatcher, PlatformDisplay, PlatformInput, PlatformInputHandler,
    PlatformTextSystem, PlatformWindow, Point, RequestFrameOptions, Size, WindowAppearance, WindowBounds,
    WindowControlArea, WindowParams, oneshot,
};

/// The queues and clock behind `HeadlessDispatcher`.
#[derive(Default)]
pub(crate) struct DispatcherState {
    /// The fake time (starts at the dispatcher's creation).
    pub(crate) now: Option<Instant>,
    /// Main-thread runnables.
    pub(crate) foreground: VecDeque<Runnable>,
    /// Background runnables (run on the pumping thread, in order).
    pub(crate) background: VecDeque<Runnable>,
    /// Delayed runnables and when they are due, in insertion order.
    pub(crate) delayed: Vec<(Instant, Runnable)>,
}

/// A dispatcher whose queues the test pumps.
#[derive(Default)]
pub struct HeadlessDispatcher {
    /// The queues.
    pub(crate) state: Mutex<DispatcherState>,
}

impl HeadlessDispatcher {
    /// Runs queued runnables (foreground and background) until none is left; delayed ones stay until due.
    /// Returns whether anything ran.
    pub fn run_until_idle(&self) -> bool {
        todo!("elements")
    }

    /// Moves the clock forward, making due delayed runnables ready.
    pub fn advance_clock(&self, _duration: Duration) {
        todo!("elements")
    }

    /// Whether nothing is queued or due.
    pub fn is_idle(&self) -> bool {
        todo!("elements")
    }
}

impl PlatformDispatcher for HeadlessDispatcher {
    fn is_main_thread(&self) -> bool {
        true
    }

    fn dispatch(&self, runnable: Runnable) {
        self.state.lock().background.push_back(runnable);
    }

    fn dispatch_on_main_thread(&self, runnable: Runnable) {
        self.state.lock().foreground.push_back(runnable);
    }

    fn dispatch_after(&self, _duration: Duration, _runnable: Runnable) {
        todo!("elements")
    }

    fn now(&self) -> Instant {
        todo!("elements")
    }
}

/// The one headless display (the window's size, at the origin).
#[derive(Debug)]
pub struct HeadlessDisplay {
    /// Its bounds.
    pub(crate) bounds: Bounds<Pixels>,
}

impl PlatformDisplay for HeadlessDisplay {
    fn id(&self) -> DisplayId {
        DisplayId::new(1)
    }

    fn uuid(&self) -> Result<uuid::Uuid> {
        Ok(uuid::Uuid::nil())
    }

    fn bounds(&self) -> Bounds<Pixels> {
        self.bounds
    }
}

/// The headless platform.
pub struct HeadlessPlatform {
    /// The dispatcher.
    pub(crate) dispatcher: Arc<HeadlessDispatcher>,
    /// The text backend.
    pub(crate) text_system: Arc<dyn PlatformTextSystem>,
    /// Every window's size (logical px).
    pub(crate) size: Size<Pixels>,
    /// Every window's scale factor.
    pub(crate) scale_factor: f32,
    /// The display.
    pub(crate) display: Rc<HeadlessDisplay>,
    /// The fake clipboard.
    pub(crate) clipboard: RefCell<Option<ClipboardItem>>,
    /// The cursor last set (tests can assert on it).
    pub(crate) cursor_style: Cell<CursorStyle>,
    /// The windows opened, for the harness to reach their state.
    pub(crate) windows: RefCell<Vec<Rc<HeadlessWindowState>>>,
    /// Set by `quit`.
    pub(crate) quit_requested: Cell<bool>,
}

impl HeadlessPlatform {
    /// A platform with windows of `size` at `scale_factor`, shaping text with `text_system`.
    pub fn new(_size: Size<Pixels>, _scale_factor: f32, _text_system: Arc<dyn PlatformTextSystem>) -> Self {
        todo!("elements")
    }

    /// The dispatcher (for the harness to pump).
    pub fn headless_dispatcher(&self) -> &Arc<HeadlessDispatcher> {
        &self.dispatcher
    }
}

impl Platform for HeadlessPlatform {
    fn dispatcher(&self) -> Arc<dyn PlatformDispatcher> {
        self.dispatcher.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>) {
        on_finish_launching()
    }

    fn quit(&self) {
        self.quit_requested.set(true)
    }

    fn activate(&self, _ignoring_other_apps: bool) {}

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        vec![self.display.clone()]
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn open_window(&self, _options: WindowParams) -> Result<Box<dyn PlatformWindow>> {
        todo!("elements")
    }

    fn window_appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn open_url(&self, _url: &str) {}

    fn prompt_for_paths(
        &self,
        _options: crate::PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        todo!("elements")
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        todo!("elements")
    }

    fn reveal_path(&self, _path: &Path) {}

    fn set_cursor_style(&self, style: CursorStyle) {
        self.cursor_style.set(style)
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        *self.clipboard.borrow_mut() = Some(item);
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        self.clipboard.borrow().clone()
    }
}

/// A headless window's callbacks (what the core registered).
#[derive(Default)]
pub(crate) struct HeadlessCallbacks {
    /// `on_request_frame`.
    pub(crate) request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    /// `on_input`.
    pub(crate) input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    /// `on_should_close`.
    pub(crate) should_close: Option<Box<dyn FnMut() -> bool>>,
    /// `on_close`.
    pub(crate) close: Option<Box<dyn FnOnce()>>,
    /// `on_hit_test_window_control`.
    pub(crate) hit_test_window_control: Option<Box<dyn FnMut() -> Option<WindowControlArea>>>,
}

/// State shared between a `HeadlessWindow` and the harness.
pub(crate) struct HeadlessWindowState {
    /// Size (logical px).
    pub(crate) size: Cell<Size<Pixels>>,
    /// Scale factor.
    pub(crate) scale_factor: f32,
    /// Mouse position (set by simulated input).
    pub(crate) mouse_position: Cell<Point<Pixels>>,
    /// Modifiers (set by simulated input).
    pub(crate) modifiers: Cell<Modifiers>,
    /// A frame was requested and not yet delivered.
    pub(crate) frame_requested: Cell<bool>,
    /// The callbacks.
    pub(crate) callbacks: RefCell<HeadlessCallbacks>,
    /// The text input handler the core gave the window.
    pub(crate) input_handler: RefCell<Option<Box<dyn PlatformInputHandler>>>,
    /// A copy of the last presented frame.
    pub(crate) presented: RefCell<Framebuffer>,
    /// The damage of the last present.
    pub(crate) last_damage: RefCell<Vec<Bounds<DevicePixels>>>,
    /// The title.
    pub(crate) title: RefCell<String>,
    /// Window mode flags.
    pub(crate) maximized: Cell<bool>,
    /// Full screen flag.
    pub(crate) fullscreen: Cell<bool>,
}

/// A headless window.
pub struct HeadlessWindow {
    /// Shared with the harness.
    pub(crate) state: Rc<HeadlessWindowState>,
    /// The display.
    pub(crate) display: Rc<HeadlessDisplay>,
}

impl PlatformWindow for HeadlessWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        Bounds::new(Point::default(), self.state.size.get())
    }

    fn is_maximized(&self) -> bool {
        self.state.maximized.get()
    }

    fn window_bounds(&self) -> WindowBounds {
        WindowBounds::Windowed(self.bounds())
    }

    fn content_size(&self) -> Size<Pixels> {
        self.state.size.get()
    }

    fn resize(&mut self, size: Size<Pixels>) {
        self.state.size.set(size)
    }

    fn scale_factor(&self) -> f32 {
        self.state.scale_factor
    }

    fn appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(self.display.clone())
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.state.mouse_position.get()
    }

    fn modifiers(&self) -> Modifiers {
        self.state.modifiers.get()
    }

    fn capslock(&self) -> crate::Capslock {
        crate::Capslock::default()
    }

    fn set_input_handler(&mut self, input_handler: Box<dyn PlatformInputHandler>) {
        *self.state.input_handler.borrow_mut() = Some(input_handler);
    }

    fn take_input_handler(&mut self) -> Option<Box<dyn PlatformInputHandler>> {
        self.state.input_handler.borrow_mut().take()
    }

    fn activate(&self) {}

    fn is_active(&self) -> bool {
        true
    }

    fn is_hovered(&self) -> bool {
        true
    }

    fn set_title(&mut self, title: &str) {
        *self.state.title.borrow_mut() = title.to_string();
    }

    fn minimize(&self) {}

    fn zoom(&self) {
        self.state.maximized.set(!self.state.maximized.get())
    }

    fn toggle_fullscreen(&self) {
        self.state.fullscreen.set(!self.state.fullscreen.get())
    }

    fn is_fullscreen(&self) -> bool {
        self.state.fullscreen.get()
    }

    fn request_frame(&self) {
        self.state.frame_requested.set(true)
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        self.state.callbacks.borrow_mut().request_frame = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        self.state.callbacks.borrow_mut().input = Some(callback);
    }

    fn on_active_status_change(&self, _callback: Box<dyn FnMut(bool)>) {}

    fn on_hover_status_change(&self, _callback: Box<dyn FnMut(bool)>) {}

    fn on_resize(&self, _callback: Box<dyn FnMut(Size<Pixels>, f32)>) {}

    fn on_moved(&self, _callback: Box<dyn FnMut()>) {}

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        self.state.callbacks.borrow_mut().should_close = Some(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        self.state.callbacks.borrow_mut().hit_test_window_control = Some(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        self.state.callbacks.borrow_mut().close = Some(callback);
    }

    fn on_appearance_changed(&self, _callback: Box<dyn FnMut()>) {}

    fn present(&mut self, _frame: &Framebuffer, _damage: &[Bounds<DevicePixels>]) {
        todo!("elements")
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}
}
