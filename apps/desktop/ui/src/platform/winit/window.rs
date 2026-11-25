//! A winit window as a `PlatformWindow`: its state, callbacks, attributes and presenting through a
//! softbuffer surface. Event handling is in `events.rs`.

use super::chrome::Chrome;
use super::display::WinitDisplay;
use super::mouse::{ClickState, FileDropTracker, cursor_icon};
use super::present::{clamp_damage, copy_rects, trace_first_present};
use crate::{
    Bounds, Capslock, CursorStyle, DevicePixels, DispatchEventResult, Framebuffer, Modifiers, MouseButton,
    Pixels, PlatformDisplay, PlatformInput, PlatformInputHandler, PlatformWindow, Point, RequestFrameOptions,
    Size, WindowAppearance, WindowBounds, WindowControlArea, WindowKind, WindowParams, point, px, size,
};
use std::cell::{Cell, RefCell};
use std::num::NonZeroU32;
use std::rc::{Rc, Weak};
use std::time::{Duration, Instant};
use winit::dpi::{LogicalPosition, LogicalSize, PhysicalPosition};
use winit::window::{Fullscreen, Theme, Window, WindowAttributes, WindowButtons, WindowLevel};

type Slot<T> = RefCell<Option<Box<T>>>;

/// The callbacks the core registers.
#[derive(Default)]
pub(crate) struct Callbacks {
    pub request_frame: Slot<dyn FnMut(RequestFrameOptions)>,
    pub input: Slot<dyn FnMut(PlatformInput) -> DispatchEventResult>,
    pub active_status_change: Slot<dyn FnMut(bool)>,
    pub hover_status_change: Slot<dyn FnMut(bool)>,
    pub resize: Slot<dyn FnMut(Size<Pixels>, f32)>,
    pub moved: Slot<dyn FnMut()>,
    pub should_close: Slot<dyn FnMut() -> bool>,
    pub hit_test_window_control: Slot<dyn FnMut() -> Option<WindowControlArea>>,
    pub close: Slot<dyn FnOnce()>,
    pub appearance_changed: Slot<dyn FnMut()>,
}

/// Calls the callback in `slot` with it taken out, so it may re-enter the window; `None` if none is set.
pub(crate) fn call<T: ?Sized, R>(slot: &Slot<T>, f: impl FnOnce(&mut T) -> R) -> Option<R> {
    let mut callback = slot.borrow_mut().take()?;
    let result = f(&mut callback);
    let mut current = slot.borrow_mut();
    if current.is_none() {
        *current = Some(callback);
    }
    Some(result)
}

/// The window's mutable state.
pub(crate) struct WindowState {
    pub scale_factor: f32,
    pub mouse_position: Point<Pixels>,
    pub modifiers: Modifiers,
    pub capslock: Capslock,
    pub pressed_buttons: Vec<MouseButton>,
    /// Buttons already synthetically released on focus loss; their late native release is ignored.
    pub cancelled_buttons: Vec<MouseButton>,
    pub clicks: ClickState,
    pub chrome: Chrome,
    pub file_drops: FileDropTracker,
    pub hovered: bool,
    pub active: bool,
    /// Bounds to restore to from maximized or full screen.
    pub restore_bounds: Bounds<Pixels>,
    /// The last known outer origin (Wayland does not tell).
    pub origin: Point<Pixels>,
    /// The core asked for a frame.
    pub frame_requested: bool,
    /// `request_redraw` was called and `RedrawRequested` has not come yet.
    pub redraw_pending: bool,
    pub last_frame: Option<Instant>,
    pub frame_interval: Duration,
    /// The next present must copy the whole frame (resize, expose, new surface).
    pub full_present: bool,
    pub surface_size: (u32, u32),
    /// An IME composition is in progress.
    pub composing: bool,
    /// What the last key press typed and the last IME commit, this batch (one keystroke reported both
    /// ways inserts once).
    pub key_text: Option<String>,
    pub commit_text: Option<String>,
    pub ime_allowed: bool,
    pub cursor: Option<CursorStyle>,
    pub closed: bool,
}

/// The shared part of a window: what the event loop and the `PlatformWindow` both hold.
pub(crate) struct WindowInner {
    pub window: Rc<Window>,
    pub state: RefCell<WindowState>,
    pub callbacks: Callbacks,
    pub input_handler: RefCell<Option<Box<dyn PlatformInputHandler>>>,
    surface: RefCell<Option<Surface>>,
    pub offscreen: bool,
    /// Removes the window from the platform's list when it closes.
    pub on_closed: Cell<Option<Box<dyn FnOnce()>>>,
}

struct Surface {
    _context: softbuffer::Context<Rc<Window>>,
    surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
}

/// The attributes `params` asks for. `offscreen_origin` places the window outside every display, never
/// activated (`FOLO_UI_OFFSCREEN=1`).
pub(crate) fn window_attributes(
    params: &WindowParams,
    offscreen_origin: Option<PhysicalPosition<i32>>,
) -> WindowAttributes {
    let titlebar = params.titlebar.as_ref();
    let title = titlebar.and_then(|t| t.title.as_ref()).map(|t| t.to_string()).unwrap_or_default();
    let bounds = params.bounds;
    let mut buttons = WindowButtons::CLOSE | WindowButtons::MAXIMIZE;
    if params.is_minimizable {
        buttons |= WindowButtons::MINIMIZE;
    }
    let mut attributes = Window::default_attributes()
        .with_title(title)
        .with_inner_size(LogicalSize::new(f32::from(bounds.size.width), f32::from(bounds.size.height)))
        .with_resizable(params.is_resizable)
        .with_enabled_buttons(buttons)
        .with_visible(params.show)
        .with_active(params.focus && offscreen_origin.is_none());
    attributes = match offscreen_origin {
        Some(origin) => attributes.with_position(origin),
        None => attributes
            .with_position(LogicalPosition::new(f32::from(bounds.origin.x), f32::from(bounds.origin.y))),
    };
    if let Some(min) = params.window_min_size {
        attributes =
            attributes.with_min_inner_size(LogicalSize::new(f32::from(min.width), f32::from(min.height)));
    }
    if matches!(params.kind, WindowKind::PopUp | WindowKind::Floating) {
        attributes = attributes.with_window_level(WindowLevel::AlwaysOnTop);
    }

    #[cfg(target_os = "macos")]
    if titlebar.is_some_and(|t| t.appears_transparent) {
        use winit::platform::macos::WindowAttributesExtMacOS;
        // GPUI's macOS window: transparent title bar over a full-size content view, native traffic lights.
        attributes = attributes
            .with_titlebar_transparent(true)
            .with_title_hidden(true)
            .with_fullsize_content_view(true);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    if let Some(app_id) = &params.app_id {
        use winit::platform::{wayland::WindowAttributesExtWayland, x11::WindowAttributesExtX11};
        attributes = WindowAttributesExtX11::with_name(attributes, app_id, app_id);
        attributes = WindowAttributesExtWayland::with_name(attributes, app_id, app_id);
    }
    #[cfg(windows)]
    if offscreen_origin.is_some() {
        use winit::platform::windows::WindowAttributesExtWindows;
        attributes = attributes.with_skip_taskbar(true);
    }
    attributes
}

/// A point to the right of and below every monitor (physical pixels), where an off-screen window goes.
pub(crate) fn offscreen_origin(
    monitors: impl Iterator<Item = winit::monitor::MonitorHandle>,
) -> PhysicalPosition<i32> {
    let (mut x, mut y) = (0i32, 0i32);
    for monitor in monitors {
        let (p, s) = (monitor.position(), monitor.size());
        x = x.max(p.x.saturating_add(s.width as i32));
        y = y.max(p.y.saturating_add(s.height as i32));
    }
    PhysicalPosition::new(x.saturating_add(200), y.saturating_add(200))
}

impl WindowInner {
    pub fn new(window: Window, params: &WindowParams, offscreen: bool) -> Rc<Self> {
        let window = Rc::new(window);
        let scale_factor = window.scale_factor() as f32;
        let inner = Rc::new(Self {
            state: RefCell::new(WindowState {
                scale_factor,
                mouse_position: Point::default(),
                modifiers: Modifiers::default(),
                capslock: Capslock::default(),
                pressed_buttons: Vec::new(),
                cancelled_buttons: Vec::new(),
                clicks: ClickState::default(),
                chrome: Chrome::default(),
                file_drops: FileDropTracker::default(),
                hovered: false,
                active: params.focus && !offscreen,
                restore_bounds: params.bounds,
                origin: params.bounds.origin,
                frame_requested: false,
                redraw_pending: false,
                last_frame: None,
                frame_interval: Duration::from_micros(16_667),
                full_present: true,
                surface_size: (0, 0),
                composing: false,
                key_text: None,
                commit_text: None,
                ime_allowed: false,
                cursor: None,
                closed: false,
            }),
            window,
            callbacks: Callbacks::default(),
            input_handler: RefCell::new(None),
            surface: RefCell::new(None),
            offscreen,
            on_closed: Cell::new(None),
        });
        inner.update_frame_interval();
        inner.state.borrow_mut().origin = inner.outer_bounds().origin;
        inner
    }

    /// The display refresh interval of the monitor the window is on.
    pub fn update_frame_interval(&self) {
        let millihertz = self
            .window
            .current_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .filter(|&mhz| mhz >= 1000)
            .unwrap_or(60_000);
        self.state.borrow_mut().frame_interval = Duration::from_secs_f64(1000.0 / millihertz as f64);
    }

    pub fn scale_factor(&self) -> f32 {
        self.state.borrow().scale_factor
    }

    /// The outer bounds in screen logical pixels.
    pub fn outer_bounds(&self) -> Bounds<Pixels> {
        let scale = self.window.scale_factor() as f32;
        let origin = match self.window.outer_position() {
            Ok(p) => point(px(p.x as f32 / scale), px(p.y as f32 / scale)),
            Err(_) => self.state.borrow().origin,
        };
        let outer = self.window.outer_size();
        Bounds::new(origin, size(px(outer.width as f32 / scale), px(outer.height as f32 / scale)))
    }

    pub fn content_size(&self) -> Size<Pixels> {
        let scale = self.window.scale_factor() as f32;
        let inner = self.window.inner_size();
        size(px(inner.width as f32 / scale), px(inner.height as f32 / scale))
    }

    /// Records the current bounds as the ones to restore to, while the window is neither maximized nor
    /// full screen.
    pub fn track_restore_bounds(&self) {
        let bounds = self.outer_bounds();
        let mut state = self.state.borrow_mut();
        state.origin = bounds.origin;
        if !self.window.is_maximized()
            && self.window.fullscreen().is_none()
            && !self.window.is_minimized().unwrap_or(false)
        {
            state.restore_bounds = bounds;
        }
    }

    /// Sends `input` to the core; with no input callback the event propagates.
    pub fn dispatch(&self, input: PlatformInput) -> DispatchEventResult {
        call(&self.callbacks.input, |callback| callback(input))
            .unwrap_or(DispatchEventResult { propagate: true, default_prevented: false })
    }

    /// Calls `f` with the input handler taken out (it may re-enter the window).
    pub fn with_input_handler<R>(&self, f: impl FnOnce(&mut dyn PlatformInputHandler) -> R) -> Option<R> {
        let mut handler = self.input_handler.borrow_mut().take()?;
        let result = f(handler.as_mut());
        let mut slot = self.input_handler.borrow_mut();
        if slot.is_none() {
            *slot = Some(handler);
        }
        Some(result)
    }

    /// Runs the core's frame callback now.
    pub fn draw(&self, options: RequestFrameOptions) {
        let size = self.window.inner_size();
        if self.state.borrow().closed || size.width == 0 || size.height == 0 {
            return;
        }
        self.state.borrow_mut().last_frame = Some(Instant::now());
        call(&self.callbacks.request_frame, |callback| callback(options));
    }

    /// Asks for `RedrawRequested` if a frame is wanted and the refresh interval has passed; otherwise
    /// returns when it will have.
    pub fn schedule_frame(&self, now: Instant) -> Option<Instant> {
        let mut state = self.state.borrow_mut();
        if !state.frame_requested || state.redraw_pending || state.closed {
            return None;
        }
        let due = state.last_frame.map_or(now, |last| last + state.frame_interval);
        if due <= now {
            state.redraw_pending = true;
            drop(state);
            self.window.request_redraw();
            None
        } else {
            Some(due)
        }
    }

    /// Applies the platform's cursor style (only when it changed).
    pub fn apply_cursor(&self, style: CursorStyle) {
        let mut state = self.state.borrow_mut();
        if state.cursor == Some(style) {
            return;
        }
        state.cursor = Some(style);
        drop(state);
        match cursor_icon(style) {
            Some(icon) => {
                self.window.set_cursor_visible(true);
                self.window.set_cursor(icon);
            }
            None => self.window.set_cursor_visible(false),
        }
    }

    /// Closes the window: hides it, tells the core once, and leaves the platform's list.
    pub fn close(&self) {
        {
            let mut state = self.state.borrow_mut();
            if state.closed {
                return;
            }
            state.closed = true;
        }
        self.window.set_visible(false);
        if let Some(on_close) = self.callbacks.close.borrow_mut().take() {
            on_close();
        }
        if let Some(on_closed) = self.on_closed.take() {
            on_closed();
        }
    }

    fn present(&self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]) {
        let (Some(width), Some(height)) = (NonZeroU32::new(frame.width), NonZeroU32::new(frame.height))
        else {
            return;
        };
        let mut surface = self.surface.borrow_mut();
        if surface.is_none() {
            let Ok(context) = softbuffer::Context::new(self.window.clone()) else { return };
            let Ok(created) = softbuffer::Surface::new(&context, self.window.clone()) else { return };
            *surface = Some(Surface { _context: context, surface: created });
            self.state.borrow_mut().full_present = true;
        }
        let surface = &mut surface.as_mut().expect("created above").surface;
        let mut state = self.state.borrow_mut();
        if state.surface_size != (frame.width, frame.height) {
            if surface.resize(width, height).is_err() {
                return;
            }
            state.surface_size = (frame.width, frame.height);
            state.full_present = true;
        }
        let Ok(mut buffer) = surface.buffer_mut() else { return };
        // A buffer that does not hold the previous frame (age 0: new, 2+: swapped) is copied whole.
        let full = std::mem::take(&mut state.full_present) || buffer.age() != 1;
        drop(state);
        self.window.pre_present_notify();
        let result = if full {
            buffer.copy_from_slice(&frame.pixels);
            buffer.present()
        } else {
            let rects = clamp_damage(damage, frame.width, frame.height);
            if rects.is_empty() {
                return;
            }
            copy_rects(&frame.pixels, &mut buffer, frame.width, &rects);
            let rects: Vec<softbuffer::Rect> = rects.into_iter().filter_map(|r| r.to_softbuffer()).collect();
            buffer.present_with_damage(&rects)
        };
        if result.is_ok() {
            trace_first_present();
        } else {
            self.state.borrow_mut().full_present = true;
        }
    }
}

/// The `PlatformWindow` the core owns. Dropping it closes the OS window.
pub struct WinitWindow {
    pub(crate) inner: Rc<WindowInner>,
}

impl WinitWindow {
    /// A weak handle the event loop keeps.
    pub(crate) fn downgrade(&self) -> Weak<WindowInner> {
        Rc::downgrade(&self.inner)
    }
}

impl Drop for WinitWindow {
    fn drop(&mut self) {
        self.inner.state.borrow_mut().closed = true;
        if let Some(on_closed) = self.inner.on_closed.take() {
            on_closed();
        }
    }
}

impl PlatformWindow for WinitWindow {
    fn bounds(&self) -> Bounds<Pixels> {
        self.inner.outer_bounds()
    }

    fn is_maximized(&self) -> bool {
        self.inner.window.is_maximized()
    }

    fn window_bounds(&self) -> WindowBounds {
        let restore = self.inner.state.borrow().restore_bounds;
        if self.inner.window.fullscreen().is_some() {
            WindowBounds::Fullscreen(restore)
        } else if self.inner.window.is_maximized() {
            WindowBounds::Maximized(restore)
        } else {
            WindowBounds::Windowed(self.inner.outer_bounds())
        }
    }

    fn content_size(&self) -> Size<Pixels> {
        self.inner.content_size()
    }

    fn resize(&mut self, size: Size<Pixels>) {
        let _ = self
            .inner
            .window
            .request_inner_size(LogicalSize::new(f32::from(size.width), f32::from(size.height)));
    }

    fn scale_factor(&self) -> f32 {
        self.inner.scale_factor()
    }

    fn appearance(&self) -> WindowAppearance {
        appearance(self.inner.window.theme())
    }

    fn display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        let current = self.inner.window.current_monitor()?;
        let index = self.inner.window.available_monitors().position(|m| m == current).unwrap_or(0);
        Some(Rc::new(WinitDisplay::new(index, current)))
    }

    fn mouse_position(&self) -> Point<Pixels> {
        self.inner.state.borrow().mouse_position
    }

    fn modifiers(&self) -> Modifiers {
        self.inner.state.borrow().modifiers
    }

    fn capslock(&self) -> Capslock {
        self.inner.state.borrow().capslock
    }

    fn set_input_handler(&mut self, input_handler: Box<dyn PlatformInputHandler>) {
        *self.inner.input_handler.borrow_mut() = Some(input_handler);
        // Allowed once and left on: toggling it per frame would reset the IME.
        if !std::mem::replace(&mut self.inner.state.borrow_mut().ime_allowed, true) {
            self.inner.window.set_ime_allowed(true);
        }
    }

    fn take_input_handler(&mut self) -> Option<Box<dyn PlatformInputHandler>> {
        self.inner.input_handler.borrow_mut().take()
    }

    fn activate(&self) {
        if !self.inner.offscreen {
            self.inner.window.focus_window();
        }
    }

    fn is_active(&self) -> bool {
        self.inner.state.borrow().active
    }

    fn is_hovered(&self) -> bool {
        self.inner.state.borrow().hovered
    }

    fn set_title(&mut self, title: &str) {
        self.inner.window.set_title(title);
    }

    fn minimize(&self) {
        self.inner.window.set_minimized(true);
    }

    fn zoom(&self) {
        self.inner.window.set_maximized(!self.inner.window.is_maximized());
    }

    fn toggle_fullscreen(&self) {
        let window = &self.inner.window;
        if window.fullscreen().is_some() {
            window.set_fullscreen(None);
        } else {
            self.inner.track_restore_bounds();
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
    }

    fn is_fullscreen(&self) -> bool {
        self.inner.window.fullscreen().is_some()
    }

    fn request_frame(&self) {
        self.inner.state.borrow_mut().frame_requested = true;
    }

    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>) {
        *self.inner.callbacks.request_frame.borrow_mut() = Some(callback);
    }

    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>) {
        *self.inner.callbacks.input.borrow_mut() = Some(callback);
    }

    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        *self.inner.callbacks.active_status_change.borrow_mut() = Some(callback);
    }

    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>) {
        *self.inner.callbacks.hover_status_change.borrow_mut() = Some(callback);
    }

    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>) {
        *self.inner.callbacks.resize.borrow_mut() = Some(callback);
    }

    fn on_moved(&self, callback: Box<dyn FnMut()>) {
        *self.inner.callbacks.moved.borrow_mut() = Some(callback);
    }

    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>) {
        *self.inner.callbacks.should_close.borrow_mut() = Some(callback);
    }

    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>) {
        *self.inner.callbacks.hit_test_window_control.borrow_mut() = Some(callback);
    }

    fn on_close(&self, callback: Box<dyn FnOnce()>) {
        *self.inner.callbacks.close.borrow_mut() = Some(callback);
    }

    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>) {
        *self.inner.callbacks.appearance_changed.borrow_mut() = Some(callback);
    }

    fn present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]) {
        self.inner.present(frame, damage);
    }

    fn update_ime_position(&self, bounds: Bounds<Pixels>) {
        self.inner.set_ime_area(bounds);
    }
}

impl WindowInner {
    /// Moves the IME candidate window next to `bounds` (window logical pixels).
    pub fn set_ime_area(&self, bounds: Bounds<Pixels>) {
        self.window.set_ime_cursor_area(
            LogicalPosition::new(f32::from(bounds.origin.x), f32::from(bounds.origin.y)),
            LogicalSize::new(f32::from(bounds.size.width), f32::from(bounds.size.height)),
        );
    }
}

/// GPUI's appearance for a winit theme (light when the system does not say).
pub(crate) fn appearance(theme: Option<Theme>) -> WindowAppearance {
    match theme {
        Some(Theme::Dark) => WindowAppearance::Dark,
        _ => WindowAppearance::Light,
    }
}
