//! The headless platform: no OS window, fixed size and scale, synthetic input, renders to a `Framebuffer`.
//! Owner tag: `"elements"`. Driven by `HeadlessApp` (tests, `folo-desktop --shot`, CI).
//!
//! Determinism: the clock moves only when the harness advances it, main-thread runnables run when it pumps,
//! and background jobs run on real threads that `run_until_idle` waits for (`dispatcher.rs`). A window's
//! `request_frame` just records that a frame is wanted; the harness draws it.

mod dispatcher;

pub use dispatcher::*;

use std::{
    cell::{Cell, RefCell},
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use anyhow::Result;

use crate::{
    Bounds, ClipboardItem, CursorStyle, DevicePixels, DispatchEventResult, DisplayId, Framebuffer,
    HeadlessOptions, Modifiers, Pixels, Platform, PlatformDispatcher, PlatformDisplay, PlatformInput,
    PlatformInputHandler, PlatformTextSystem, PlatformWindow, Point, RequestFrameOptions, Size, TextBackend,
    WindowAppearance, WindowBounds, WindowControlArea, WindowParams, oneshot,
};

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
    /// What the next file prompts answer (`None`: cancelled). Tests set it.
    pub(crate) prompt_answer: RefCell<Option<Vec<PathBuf>>>,
    /// URLs passed to `open_url` and paths to `reveal_path`, for tests to assert on.
    pub(crate) opened: RefCell<Vec<String>>,
}

impl HeadlessPlatform {
    /// A platform with windows of `size` at `scale_factor`, shaping text with `text_system`.
    pub fn new(size: Size<Pixels>, scale_factor: f32, text_system: Arc<dyn PlatformTextSystem>) -> Self {
        HeadlessPlatform {
            dispatcher: Arc::new(HeadlessDispatcher::new()),
            text_system,
            size,
            scale_factor,
            display: Rc::new(HeadlessDisplay { bounds: Bounds::new(Point::default(), size) }),
            clipboard: RefCell::new(None),
            cursor_style: Cell::new(CursorStyle::default()),
            windows: RefCell::new(Vec::new()),
            quit_requested: Cell::new(false),
            prompt_answer: RefCell::new(None),
            opened: RefCell::new(Vec::new()),
        }
    }

    /// A platform for `options`, with the text backend it names. Panics when that backend is not built
    /// into this binary (`Fixed` needs feature `test-support`; `Portable` on Windows needs `portable-text`).
    pub fn from_options(options: HeadlessOptions) -> Self {
        Self::try_from_options(options).unwrap_or_else(|error| panic!("headless text backend: {error:#}"))
    }

    /// `from_options`, returning the error when the text backend is not built into this binary.
    pub fn try_from_options(options: HeadlessOptions) -> Result<Self> {
        Ok(Self::new(options.size, options.scale_factor, text_backend(options.text)?))
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
        // Every headless window has the platform's size, whatever bounds the app asked for.
        let state = Rc::new(HeadlessWindowState {
            size: Cell::new(self.size),
            scale_factor: self.scale_factor,
            // Outside the window until input moves it (as a real window the mouse is not over): nothing
            // starts hovered.
            mouse_position: Cell::new(crate::point(crate::px(-10_000.), crate::px(-10_000.))),
            modifiers: Cell::new(Modifiers::default()),
            frame_requested: Cell::new(false),
            callbacks: RefCell::new(HeadlessCallbacks::default()),
            input_handler: RefCell::new(None),
            presented: RefCell::new(Framebuffer::default()),
            last_damage: RefCell::new(Vec::new()),
            title: RefCell::new(String::new()),
            maximized: Cell::new(false),
            fullscreen: Cell::new(false),
        });
        self.windows.borrow_mut().push(state.clone());
        Ok(Box::new(HeadlessWindow { state, display: self.display.clone() }))
    }

    fn window_appearance(&self) -> WindowAppearance {
        WindowAppearance::Dark
    }

    fn open_url(&self, url: &str) {
        self.opened.borrow_mut().push(url.to_string());
    }

    fn prompt_for_paths(
        &self,
        _options: crate::PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (tx, rx) = oneshot::channel();
        tx.send(Ok(self.prompt_answer.borrow().clone())).ok();
        rx
    }

    fn prompt_for_new_path(
        &self,
        _directory: &Path,
        _suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let (tx, rx) = oneshot::channel();
        let answer = self.prompt_answer.borrow().as_ref().and_then(|paths| paths.first().cloned());
        tx.send(Ok(answer)).ok();
        rx
    }

    fn reveal_path(&self, path: &Path) {
        self.opened.borrow_mut().push(path.display().to_string());
    }

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

    fn present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]) {
        // Only the damage changed since the last present (as a real backend copies it); a new size copies all.
        let mut presented = self.state.presented.borrow_mut();
        if presented.width == frame.width && presented.height == frame.height {
            let width = frame.width as usize;
            for rect in damage {
                let (x0, x1) =
                    (rect.origin.x.0.max(0) as usize, (rect.origin.x.0 + rect.size.width.0) as usize);
                let (y0, y1) =
                    (rect.origin.y.0.max(0) as usize, (rect.origin.y.0 + rect.size.height.0) as usize);
                let x1 = x1.min(width);
                for y in y0..y1.min(frame.height as usize) {
                    let row = y * width;
                    presented.pixels[row + x0..row + x1].copy_from_slice(&frame.pixels[row + x0..row + x1]);
                }
            }
        } else {
            presented.clone_from(frame);
        }
        drop(presented);
        let mut last_damage = self.state.last_damage.borrow_mut();
        last_damage.clear();
        last_damage.extend_from_slice(damage);
    }

    fn update_ime_position(&self, _bounds: Bounds<Pixels>) {}

    fn inserts_key_text(&self) -> bool {
        // Synthetic keys are the only text input here.
        true
    }
}

/// The `PlatformTextSystem` for a headless text backend.
pub(crate) fn text_backend(backend: TextBackend) -> Result<Arc<dyn PlatformTextSystem>> {
    match backend {
        #[cfg(windows)]
        TextBackend::Native => Ok(Arc::new(crate::text::dwrite::DirectWriteTextSystem::new()?)),
        #[cfg(not(windows))]
        TextBackend::Native => Ok(Arc::new(crate::text::portable::CosmicTextSystem::new())),
        #[cfg(any(not(windows), feature = "portable-text"))]
        TextBackend::Portable => Ok(Arc::new(crate::text::portable::CosmicTextSystem::new())),
        #[cfg(not(any(not(windows), feature = "portable-text")))]
        TextBackend::Portable => {
            anyhow::bail!("`TextBackend::Portable` needs folo-ui's `portable-text` feature")
        }
        #[cfg(any(test, feature = "test-support"))]
        TextBackend::Fixed => Ok(Arc::new(crate::FixedTextSystem::new())),
        #[cfg(not(any(test, feature = "test-support")))]
        TextBackend::Fixed => anyhow::bail!("`TextBackend::Fixed` needs folo-ui's `test-support` feature"),
    }
}
