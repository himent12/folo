// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/window.rs
//! `WindowsWindow`: our own window class, the window's state and callbacks, creation from
//! `WindowParams` (placement, maximized / full screen, the off-screen developer mode) and the
//! `PlatformWindow` methods. Message handling is in `events.rs`, `keyboard.rs` and `chrome.rs`.

use super::display::WindowsDisplay;
use super::pacer::{FrameSlot, Pacer};
use super::util::*;
use super::{Shared, drop_target, present};
use crate::*;
use ::windows::Win32::Foundation::*;
use ::windows::Win32::Graphics::Gdi::*;
use ::windows::Win32::System::LibraryLoader::GetModuleHandleW;
use ::windows::Win32::UI::Input::KeyboardAndMouse::*;
use ::windows::Win32::UI::WindowsAndMessaging::*;
use ::windows::core::{HSTRING, PCWSTR, w};
use anyhow::{Context as _, Result};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

/// Where an off-screen window's outer rectangle starts, in physical screen pixels.
pub(crate) const OFFSCREEN_ORIGIN: i32 = -30000;

/// What `on_resize` registers.
pub(crate) type ResizeCallback = Box<dyn FnMut(Size<Pixels>, f32)>;

/// The callbacks the core registers.
#[derive(Default)]
pub(crate) struct Callbacks {
    pub(crate) request_frame: Option<Box<dyn FnMut(RequestFrameOptions)>>,
    pub(crate) input: Option<Box<dyn FnMut(PlatformInput) -> DispatchEventResult>>,
    pub(crate) active_status_change: Option<Box<dyn FnMut(bool)>>,
    pub(crate) hovered_status_change: Option<Box<dyn FnMut(bool)>>,
    pub(crate) resize: Option<ResizeCallback>,
    pub(crate) moved: Option<Box<dyn FnMut()>>,
    pub(crate) should_close: Option<Box<dyn FnMut() -> bool>>,
    pub(crate) close: Option<Box<dyn FnOnce()>>,
    pub(crate) hit_test_window_control: Option<Box<dyn FnMut() -> Option<WindowControlArea>>>,
    pub(crate) appearance_changed: Option<Box<dyn FnMut()>>,
}

/// The window's mutable state (main thread only; never borrowed across a callback).
pub(crate) struct WindowState {
    pub(crate) origin: Point<Pixels>,
    pub(crate) logical_size: Size<Pixels>,
    pub(crate) min_size: Option<Size<Pixels>>,
    pub(crate) fullscreen_restore_bounds: Bounds<Pixels>,
    pub(crate) border_offset: WindowBorderOffset,
    pub(crate) appearance: WindowAppearance,
    pub(crate) scale_factor: f32,
    pub(crate) callbacks: Callbacks,
    pub(crate) input_handler: Option<Box<dyn PlatformInputHandler>>,
    pub(crate) pending_surrogate: Option<u16>,
    pub(crate) last_reported_modifiers: Option<Modifiers>,
    pub(crate) last_reported_capslock: Option<Capslock>,
    pub(crate) system_key_handled: bool,
    pub(crate) hovered: bool,
    pub(crate) click_state: ClickState,
    pub(crate) nc_button_pressed: Option<u32>,
    pub(crate) display: WindowsDisplay,
    pub(crate) fullscreen: Option<StyleAndBounds>,
    pub(crate) initial_placement: Option<WindowOpenStatus>,
    /// The next `present` copies the whole frame (after an expose, a resize, a DPI change).
    pub(crate) full_present: bool,
    /// WM_SIZE said minimized; frames wait for the restore.
    pub(crate) minimized: bool,
    /// Counts WM_SIZE messages (a DPI change checks whether one came).
    pub(crate) size_events: u64,
    /// Where the IME composition window goes (window logical pixels), from `update_ime_position`.
    pub(crate) ime_position: Option<Bounds<Pixels>>,
    pub(crate) wheel: WheelSettings,
    /// Off screen: the window rectangle to go back to from the emulated maximize.
    pub(crate) offscreen_restore: Option<RECT>,
    /// Off screen: the emulated maximize is on (the window has no WS_MAXIMIZE there).
    pub(crate) offscreen_maximized: bool,
    pub(crate) auto_hide_taskbar: Option<TaskbarEdge>,
}

impl WindowState {
    pub(crate) fn is_fullscreen(&self) -> bool {
        self.fullscreen.is_some()
    }
}

/// The window, shared by the window procedure, the drop target and the `PlatformWindow` box.
pub(crate) struct WindowInner {
    pub(crate) hwnd: HWND,
    /// The class DC (`CS_OWNDC`): presents need no GetDC / ReleaseDC.
    pub(crate) hdc: HDC,
    pub(crate) state: RefCell<WindowState>,
    /// While a `WM_PAINT` or `WM_PRINTCLIENT` asks the core to present, the DC it must draw on.
    pub(crate) paint_dc: Cell<Option<HDC>>,
    pub(crate) shared: Rc<Shared>,
    pub(crate) frame: Arc<FrameSlot>,
    pub(crate) pacer: Arc<Pacer>,
    pub(crate) hide_title_bar: bool,
    pub(crate) is_movable: bool,
    pub(crate) version: WindowsVersion,
    pub(crate) offscreen: bool,
    /// The hidden window that owns an off-screen window (so it gets no taskbar button).
    pub(crate) owner: Cell<Option<HWND>>,
    pub(crate) destroyed: Cell<bool>,
}

impl WindowInner {
    /// Takes a callback out, runs `f` on it without any borrow held, and puts it back (unless the
    /// call registered a new one).
    pub(crate) fn with_callback<C: ?Sized, R>(
        &self,
        slot: fn(&mut Callbacks) -> &mut Option<Box<C>>,
        f: impl FnOnce(&mut Box<C>) -> R,
    ) -> Option<R> {
        let mut callback = slot(&mut self.state.borrow_mut().callbacks).take()?;
        let result = f(&mut callback);
        let mut state = self.state.borrow_mut();
        let place = slot(&mut state.callbacks);
        if place.is_none() {
            *place = Some(callback);
        }
        Some(result)
    }

    /// Sends one input event to the core.
    pub(crate) fn dispatch_input(&self, input: PlatformInput) -> Option<DispatchEventResult> {
        self.with_callback(|c| &mut c.input, |f| f(input))
    }

    /// Runs `f` with the input handler (taken out, so `f` may call into the core).
    pub(crate) fn with_input_handler<R>(
        &self,
        f: impl FnOnce(&mut dyn PlatformInputHandler) -> R,
    ) -> Option<R> {
        let mut handler = self.state.borrow_mut().input_handler.take()?;
        let result = f(handler.as_mut());
        let mut state = self.state.borrow_mut();
        if state.input_handler.is_none() {
            state.input_handler = Some(handler);
        }
        Some(result)
    }

    /// Asks the core to draw now. `expose`: the window lost what it showed, so the next present
    /// copies all of the frame and the core must present. Returns whether the core was asked.
    pub(crate) fn draw(&self, expose: bool, force_render: bool) -> bool {
        // Nothing is drawn while minimized (GPUI takes the frame callback away there): a minimized
        // window still gets WM_PAINT where no shell hides it, and a frame drawn then would see the
        // window as neither maximized nor full screen. The restore's WM_SIZE draws.
        if self.state.borrow().minimized {
            return false;
        }
        if expose {
            self.state.borrow_mut().full_present = true;
        }
        self.with_callback(
            |c| &mut c.request_frame,
            |f| f(RequestFrameOptions { require_presentation: expose, force_render }),
        )
        .is_some()
    }

    pub(crate) fn is_maximized(&self) -> bool {
        let state = self.state.borrow();
        if self.offscreen {
            return !state.is_fullscreen() && state.offscreen_maximized;
        }
        // SAFETY: a plain query on our window.
        !state.is_fullscreen() && unsafe { IsZoomed(self.hwnd) }.as_bool()
    }

    /// The restored bounds (client area, logical) and whether the window is maximized: GPUI's
    /// `calculate_window_bounds` from the window placement.
    pub(crate) fn calculate_window_bounds(&self) -> (Bounds<Pixels>, bool) {
        let mut placement =
            WINDOWPLACEMENT { length: std::mem::size_of::<WINDOWPLACEMENT>() as u32, ..Default::default() };
        // SAFETY: a sized WINDOWPLACEMENT.
        let _ = unsafe { GetWindowPlacement(self.hwnd, &mut placement) };
        let state = self.state.borrow();
        if self.offscreen && state.offscreen_maximized {
            let normal = state.offscreen_restore.unwrap_or(placement.rcNormalPosition);
            return (calculate_client_rect(normal, state.border_offset, state.scale_factor), true);
        }
        (
            calculate_client_rect(placement.rcNormalPosition, state.border_offset, state.scale_factor),
            placement.showCmd == SW_SHOWMAXIMIZED.0 as u32,
        )
    }

    pub(crate) fn window_bounds(&self) -> WindowBounds {
        let (bounds, maximized) = self.calculate_window_bounds();
        let state = self.state.borrow();
        if state.is_fullscreen() {
            WindowBounds::Fullscreen(state.fullscreen_restore_bounds)
        } else if maximized {
            WindowBounds::Maximized(bounds)
        } else {
            WindowBounds::Windowed(bounds)
        }
    }

    /// Enters or leaves full screen: GPUI's style swap (no frame, caption or system menu) over the
    /// display's rectangle, and back to the saved style and window rectangle.
    pub(crate) fn toggle_fullscreen_now(&self) {
        let (style, x, y, cx, cy) = {
            let saved = self.state.borrow_mut().fullscreen.take();
            if let Some(s) = saved {
                (s.style, s.x, s.y, s.cx, s.cy)
            } else {
                let (restore, _) = self.calculate_window_bounds();
                let style = WINDOW_STYLE(get_window_long(self.hwnd, GWL_STYLE) as u32);
                let mut rc = RECT::default();
                // SAFETY: a valid out pointer.
                let _ = unsafe { GetWindowRect(self.hwnd, &mut rc) };
                let mut state = self.state.borrow_mut();
                state.fullscreen_restore_bounds = restore;
                state.fullscreen = Some(StyleAndBounds {
                    style,
                    x: rc.left,
                    y: rc.top,
                    cx: rc.right - rc.left,
                    cy: rc.bottom - rc.top,
                });
                let style =
                    style & !(WS_THICKFRAME | WS_SYSMENU | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_CAPTION);
                let b = state.display.physical_bounds();
                // Off screen, full screen is the display's size at the off-screen origin.
                let (x, y) = if self.offscreen {
                    (OFFSCREEN_ORIGIN, OFFSCREEN_ORIGIN)
                } else {
                    (b.origin.x.0, b.origin.y.0)
                };
                (style, x, y, b.size.width.0, b.size.height.0)
            }
        };
        set_window_long(self.hwnd, GWL_STYLE, style.0 as isize);
        // SAFETY: repositioning our own window.
        let _ = unsafe {
            SetWindowPos(self.hwnd, None, x, y, cx, cy, SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER)
        };
    }

    /// Applies a placement deferred because the window opened hidden (GPUI's `set_window_placement`).
    pub(crate) fn apply_initial_placement(&self) {
        let Some(status) = self.state.borrow_mut().initial_placement.take() else { return };
        if self.offscreen {
            let r = status.placement.rcNormalPosition;
            // SAFETY: showing our own window off screen without activating it.
            let _ = unsafe {
                SetWindowPos(
                    self.hwnd,
                    None,
                    OFFSCREEN_ORIGIN,
                    OFFSCREEN_ORIGIN,
                    r.right - r.left,
                    r.bottom - r.top,
                    SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                )
            };
            self.post_expose();
            match status.state {
                WindowOpenState::Maximized => self.offscreen_maximize(),
                WindowOpenState::Fullscreen => self.toggle_fullscreen_now(),
                WindowOpenState::Windowed => {}
            }
            return;
        }
        // SAFETY: a placement read from this window.
        let _ = unsafe { SetWindowPlacement(self.hwnd, &status.placement) };
        self.post_expose();
        match status.state {
            // SAFETY: showing our own window.
            WindowOpenState::Maximized => unsafe {
                let _ = ShowWindowAsync(self.hwnd, SW_MAXIMIZE);
            },
            WindowOpenState::Fullscreen => self.toggle_fullscreen_now(),
            WindowOpenState::Windowed => {}
        }
    }

    /// Takes the client area's size and position from the window. At creation the state holds what
    /// `CreateWindowExW` was given (`CW_USEDEFAULT`, which is `i32::MIN`), and a window opened hidden
    /// gets no `WM_SIZE` until it is shown, while the core sizes its first frame from this.
    pub(crate) fn read_client_area(&self) {
        let mut client = RECT::default();
        let mut origin = POINT::default();
        // SAFETY: valid out pointers for our own window.
        unsafe {
            let _ = GetClientRect(self.hwnd, &mut client);
            let _ = ClientToScreen(self.hwnd, &mut origin);
        }
        let mut state = self.state.borrow_mut();
        let scale_factor = state.scale_factor;
        state.logical_size = size(DevicePixels(client.right.max(1)), DevicePixels(client.bottom.max(1)))
            .to_pixels(scale_factor);
        state.origin = logical_point(origin.x as f32, origin.y as f32, scale_factor);
    }

    /// Asks for a full draw from the message loop (the window was just shown).
    pub(crate) fn post_expose(&self) {
        // SAFETY: posting to our own window.
        let _ = unsafe { PostMessageW(Some(self.hwnd), super::WM_FOLO_EXPOSE, WPARAM(0), LPARAM(0)) };
    }

    /// Shows the frame: the damage rectangles, or everything after an expose.
    pub(crate) fn present(&self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]) {
        let full = std::mem::take(&mut self.state.borrow_mut().full_present);
        let all = [Bounds::new(
            point(DevicePixels(0), DevicePixels(0)),
            size(DevicePixels(frame.width as i32), DevicePixels(frame.height as i32)),
        )];
        let dc = self.paint_dc.get().unwrap_or(self.hdc);
        present::blit(dc, frame, if full { &all } else { damage });
        if !self.shared.traced.replace(true) && trace() {
            eprintln!("first-present {:.1}", ms_since_process_start());
        }
    }
}

/// The frame (and the invisible resize border) around the client area, in device pixels.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WindowBorderOffset {
    pub(crate) width_offset: i32,
    pub(crate) height_offset: i32,
}

impl WindowBorderOffset {
    pub(crate) fn update(&mut self, hwnd: HWND) {
        let (mut window, mut client) = (RECT::default(), RECT::default());
        // SAFETY: valid out pointers.
        unsafe {
            let _ = GetWindowRect(hwnd, &mut window);
            let _ = GetClientRect(hwnd, &mut client);
        }
        self.width_offset = (window.right - window.left) - (client.right - client.left);
        self.height_offset = (window.bottom - window.top) - (client.bottom - client.top);
    }
}

/// The window's style and rectangle before it went full screen.
pub(crate) struct StyleAndBounds {
    style: WINDOW_STYLE,
    x: i32,
    y: i32,
    cx: i32,
    cy: i32,
}

/// A placement waiting for a hidden window's first `activate`.
pub(crate) struct WindowOpenStatus {
    placement: WINDOWPLACEMENT,
    pub(crate) state: WindowOpenState,
}

pub(crate) enum WindowOpenState {
    Maximized,
    Fullscreen,
    Windowed,
}

/// Mouse wheel settings (`SPI_GETWHEELSCROLLLINES` / `SPI_GETWHEELSCROLLCHARS`).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WheelSettings {
    pub(crate) scroll_lines: u32,
    pub(crate) scroll_chars: u32,
}

impl WheelSettings {
    pub(crate) fn read() -> Self {
        let read = |action| {
            let mut value = 3u32;
            // SAFETY: the parameter is a u32.
            let _ = unsafe {
                SystemParametersInfoW(
                    action,
                    0,
                    Some(&mut value as *mut u32 as *mut _),
                    SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
                )
            };
            value
        };
        Self { scroll_lines: read(SPI_GETWHEELSCROLLLINES), scroll_chars: read(SPI_GETWHEELSCROLLCHARS) }
    }
}

/// Which edge an auto-hide taskbar is on (the maximized window leaves it one pixel).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskbarEdge {
    Left,
    Top,
    Right,
    Bottom,
}

/// GPUI's double-click tracking: same button, within the system's time and distance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClickState {
    button: MouseButton,
    last_click: Instant,
    last_position: Point<DevicePixels>,
    tolerance_width: i32,
    tolerance_height: i32,
    interval: Duration,
    pub(crate) current_count: usize,
}

impl ClickState {
    pub(crate) fn new() -> Self {
        // SAFETY: plain system metric queries.
        let (w, h, ms) = unsafe {
            (GetSystemMetrics(SM_CXDOUBLECLK), GetSystemMetrics(SM_CYDOUBLECLK), GetDoubleClickTime())
        };
        Self {
            button: MouseButton::Left,
            last_click: Instant::now(),
            last_position: Point::default(),
            tolerance_width: w,
            tolerance_height: h,
            interval: Duration::from_millis(ms as u64),
            current_count: 0,
        }
    }

    /// Records a press and returns its click count.
    pub(crate) fn update(&mut self, button: MouseButton, position: Point<DevicePixels>) -> usize {
        let dx = (self.last_position.x.0 - position.x.0).abs();
        let dy = (self.last_position.y.0 - position.y.0).abs();
        let double = self.last_click.elapsed() < self.interval
            && dx <= self.tolerance_width
            && dy <= self.tolerance_height;
        self.current_count = if self.button == button && double { self.current_count + 1 } else { 1 };
        self.last_click = Instant::now();
        self.last_position = position;
        self.button = button;
        self.current_count
    }

    /// Re-reads one setting after `WM_SETTINGCHANGE` (`SPI_SETDOUBLECLKWIDTH` / `HEIGHT` / `TIME`).
    pub(crate) fn system_update(&mut self, wparam: usize) {
        // SAFETY: plain system metric queries.
        unsafe {
            match wparam {
                29 => self.tolerance_width = GetSystemMetrics(SM_CXDOUBLECLK),
                30 => self.tolerance_height = GetSystemMetrics(SM_CYDOUBLECLK),
                32 => self.interval = Duration::from_millis(GetDoubleClickTime() as u64),
                _ => {}
            }
        }
    }
}

/// The `PlatformWindow` the core owns. Dropping it destroys the OS window (later, from the queue).
pub struct WindowsWindow(pub(crate) Rc<WindowInner>);

impl WindowsWindow {
    /// The window handle (for tests and tools).
    pub fn hwnd(&self) -> HWND {
        self.0.hwnd
    }

    pub(crate) fn new(params: WindowParams, shared: Rc<Shared>) -> Result<Self> {
        register_window_class(shared.icon);
        let hide_title_bar = params.titlebar.as_ref().map(|t| t.appears_transparent).unwrap_or(true);
        let title = HSTRING::from(
            params.titlebar.as_ref().and_then(|t| t.title.as_ref()).map(|t| t.as_ref()).unwrap_or(""),
        );
        let offscreen = offscreen();
        let (mut ex_style, style) = if params.kind == WindowKind::PopUp {
            (WS_EX_TOOLWINDOW, WINDOW_STYLE(0))
        } else {
            let mut style = WS_SYSMENU;
            if params.is_resizable {
                style |= WS_THICKFRAME | WS_MAXIMIZEBOX;
            }
            if params.is_minimizable {
                style |= WS_MINIMIZEBOX;
            }
            (WS_EX_APPWINDOW, style)
        };
        // Off screen: no taskbar button or Alt-Tab entry (owned, without WS_EX_APPWINDOW), same style.
        let owner = if offscreen {
            ex_style &= !WS_EX_APPWINDOW;
            // SAFETY: a plain hidden STATIC window, destroyed with ours.
            unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("STATIC"),
                    w!(""),
                    WS_OVERLAPPED,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .ok()
        } else {
            None
        };
        let display = params
            .display_id
            .and_then(WindowsDisplay::new)
            .or_else(WindowsDisplay::primary)
            .context("no display")?;
        let appearance = system_appearance();
        let mut context = CreateContext {
            inner: None,
            shared: shared.clone(),
            hide_title_bar,
            display,
            is_movable: params.is_movable,
            min_size: params.window_min_size,
            appearance,
            offscreen,
        };
        // SAFETY: the class is registered; `context` outlives the call (WM_NCCREATE reads it).
        let created = unsafe {
            CreateWindowExW(
                ex_style,
                WINDOW_CLASS_NAME,
                &title,
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                owner,
                None,
                Some(module_handle().into()),
                Some(&context as *const CreateContext as *const _),
            )
        };
        let this = match context.inner.take() {
            Some(Ok(this)) => this,
            Some(Err(e)) => return Err(e),
            None => return Err(anyhow::anyhow!("window creation failed: {:?}", created.err())),
        };
        let hwnd = created?;
        this.owner.set(owner);
        drop_target::register(&this);
        configure_dwm_dark_mode(hwnd, appearance);
        this.state.borrow_mut().border_offset.update(hwnd);
        this.read_client_area();
        let placement = {
            let state = this.state.borrow();
            initial_placement(hwnd, &state.display, params.bounds, state.scale_factor, state.border_offset)
        };
        if params.show && offscreen {
            // SetWindowPlacement would pull a window that is wholly off screen back onto a display.
            let r = placement.rcNormalPosition;
            // SAFETY: positioning and showing our own window without activating it.
            unsafe {
                SetWindowPos(
                    hwnd,
                    None,
                    OFFSCREEN_ORIGIN,
                    OFFSCREEN_ORIGIN,
                    r.right - r.left,
                    r.bottom - r.top,
                    SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                )?
            };
            this.post_expose();
        } else if params.show {
            // SAFETY: a placement read from this window, with the requested normal rectangle.
            unsafe { SetWindowPlacement(hwnd, &placement)? };
            this.post_expose();
        } else {
            this.state.borrow_mut().initial_placement =
                Some(WindowOpenStatus { placement, state: WindowOpenState::Windowed });
        }
        Ok(Self(this))
    }
}

struct CreateContext {
    inner: Option<Result<Rc<WindowInner>>>,
    shared: Rc<Shared>,
    hide_title_bar: bool,
    display: WindowsDisplay,
    is_movable: bool,
    min_size: Option<Size<Pixels>>,
    appearance: WindowAppearance,
    offscreen: bool,
}

fn new_inner(context: &CreateContext, hwnd: HWND, cs: &CREATESTRUCTW) -> Result<Rc<WindowInner>> {
    // SAFETY: plain queries on the window being created.
    let scale_factor =
        unsafe { ::windows::Win32::UI::HiDpi::GetDpiForWindow(hwnd) } as f32 / USER_DEFAULT_SCREEN_DPI as f32;
    let origin = logical_point(cs.x as f32, cs.y as f32, scale_factor);
    let logical_size = size(DevicePixels(cs.cx), DevicePixels(cs.cy)).to_pixels(scale_factor);
    // SAFETY: the class has CS_OWNDC, so this DC stays valid for the window's life.
    let hdc = unsafe { GetDC(Some(hwnd)) };
    let state = WindowState {
        origin,
        logical_size,
        min_size: context.min_size,
        fullscreen_restore_bounds: Bounds { origin, size: logical_size },
        border_offset: WindowBorderOffset::default(),
        appearance: context.appearance,
        scale_factor,
        callbacks: Callbacks::default(),
        input_handler: None,
        pending_surrogate: None,
        last_reported_modifiers: None,
        last_reported_capslock: None,
        system_key_handled: false,
        hovered: false,
        click_state: ClickState::new(),
        nc_button_pressed: None,
        display: context.display,
        fullscreen: None,
        initial_placement: None,
        full_present: true,
        minimized: false,
        size_events: 0,
        ime_position: None,
        wheel: WheelSettings::read(),
        offscreen_restore: None,
        offscreen_maximized: false,
        auto_hide_taskbar: super::chrome::auto_hide_taskbar_edge(&context.display),
    };
    Ok(Rc::new(WindowInner {
        hwnd,
        hdc,
        state: RefCell::new(state),
        paint_dc: Cell::new(None),
        shared: context.shared.clone(),
        frame: FrameSlot::new(hwnd),
        pacer: context.shared.pacer.clone(),
        hide_title_bar: context.hide_title_bar,
        is_movable: context.is_movable,
        version: WindowsVersion::current(),
        offscreen: context.offscreen,
        owner: Cell::new(None),
        destroyed: Cell::new(false),
    }))
}

const WINDOW_CLASS_NAME: PCWSTR = w!("FOLO::Window");

fn register_window_class(icon: HICON) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let class = WNDCLASSW {
            lpfnWndProc: Some(window_procedure),
            hIcon: icon,
            lpszClassName: WINDOW_CLASS_NAME,
            // One DC per window (presents need no GetDC); no CS_HREDRAW / CS_VREDRAW and no background
            // brush: a resize repaints synchronously from WM_SIZE, nothing is erased first.
            style: CS_OWNDC,
            hInstance: module_handle().into(),
            ..Default::default()
        };
        // SAFETY: the class structure is complete and its strings are static.
        unsafe { RegisterClassW(&class) };
    });
}

pub(crate) fn module_handle() -> HMODULE {
    // SAFETY: the executable's own module handle.
    unsafe { GetModuleHandleW(None) }.unwrap_or_default()
}

unsafe extern "system" fn window_procedure(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_NCCREATE {
        // SAFETY: for WM_NCCREATE lparam is the CREATESTRUCTW whose lpCreateParams is our context,
        // alive for the whole CreateWindowExW call.
        let (cs, context) = unsafe {
            let cs = &*(lparam.0 as *const CREATESTRUCTW);
            (cs, &mut *(cs.lpCreateParams as *mut CreateContext))
        };
        return match new_inner(context, hwnd, cs) {
            Ok(inner) => {
                let weak = Box::new(Rc::downgrade(&inner));
                set_window_long(hwnd, GWLP_USERDATA, Box::into_raw(weak) as isize);
                context.inner = Some(Ok(inner));
                // SAFETY: default processing of WM_NCCREATE.
                unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
            }
            Err(e) => {
                context.inner = Some(Err(e));
                LRESULT(0)
            }
        };
    }
    let ptr = get_window_long(hwnd, GWLP_USERDATA) as *mut Weak<WindowInner>;
    if ptr.is_null() {
        // SAFETY: default processing.
        return unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) };
    }
    // SAFETY: GWLP_USERDATA holds the boxed Weak set at WM_NCCREATE until WM_NCDESTROY frees it.
    let inner = unsafe { &*ptr }.upgrade();
    let result = match inner {
        Some(inner) => inner.handle_msg(hwnd, msg, wparam, lparam),
        // SAFETY: default processing.
        None => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    };
    if msg == WM_NCDESTROY {
        set_window_long(hwnd, GWLP_USERDATA, 0);
        // SAFETY: the box was leaked at WM_NCCREATE and nothing reads it after this.
        drop(unsafe { Box::from_raw(ptr) });
    }
    result
}

/// GPUI's `calculate_window_rect`: the window rectangle around client `bounds`.
pub(crate) fn calculate_window_rect(bounds: Bounds<DevicePixels>, offset: WindowBorderOffset) -> RECT {
    let left_offset = offset.width_offset / 2;
    let top_offset = offset.height_offset / 2;
    RECT {
        left: bounds.origin.x.0 - left_offset,
        top: bounds.origin.y.0 - top_offset,
        right: bounds.origin.x.0 + bounds.size.width.0 + (offset.width_offset - left_offset),
        bottom: bounds.origin.y.0 + bounds.size.height.0 + (offset.height_offset - top_offset),
    }
}

/// GPUI's `calculate_client_rect`: the client bounds (logical) inside window rectangle `rect`.
pub(crate) fn calculate_client_rect(
    rect: RECT,
    offset: WindowBorderOffset,
    scale_factor: f32,
) -> Bounds<Pixels> {
    let left_offset = offset.width_offset / 2;
    let top_offset = offset.height_offset / 2;
    let left = rect.left + left_offset;
    let top = rect.top + top_offset;
    let right = rect.right - (offset.width_offset - left_offset);
    let bottom = rect.bottom - (offset.height_offset - top_offset);
    Bounds {
        origin: logical_point(left as f32, top as f32, scale_factor),
        size: size(DevicePixels(right - left), DevicePixels(bottom - top)).to_pixels(scale_factor),
    }
}

/// The placement a window opens with: the requested bounds if their centre is on the display, else
/// the display's default bounds (GPUI's `retrieve_window_placement`).
fn initial_placement(
    hwnd: HWND,
    display: &WindowsDisplay,
    bounds: Bounds<Pixels>,
    scale_factor: f32,
    offset: WindowBorderOffset,
) -> WINDOWPLACEMENT {
    let mut placement =
        WINDOWPLACEMENT { length: std::mem::size_of::<WINDOWPLACEMENT>() as u32, ..Default::default() };
    // SAFETY: a sized WINDOWPLACEMENT.
    let _ = unsafe { GetWindowPlacement(hwnd, &mut placement) };
    let bounds = if display.contains_center_of(bounds) { bounds } else { display.default_bounds() };
    placement.rcNormalPosition = calculate_window_rect(bounds.to_device_pixels(scale_factor), offset);
    placement
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_and_client_rects_round_trip() {
        // A Windows 11 window at 150%: 2 x 12 px of frame across, 2 + 12 down (GPUI's offsets).
        let offset = WindowBorderOffset { width_offset: 24, height_offset: 14 };
        let client = Bounds::new(
            point(DevicePixels(300), DevicePixels(150)),
            size(DevicePixels(1440), DevicePixels(930)),
        );
        let rect = calculate_window_rect(client, offset);
        assert_eq!((rect.left, rect.top, rect.right, rect.bottom), (288, 143, 1752, 1087));
        let back = calculate_client_rect(rect, offset, 1.5);
        assert_eq!(back, client.to_pixels(1.5));
    }

    #[test]
    fn clicks_count_up_within_time_and_distance() {
        let mut state = ClickState::new();
        let at = |x, y| point(DevicePixels(x), DevicePixels(y));
        assert_eq!(state.update(MouseButton::Left, at(0, 0)), 1);
        assert_eq!(state.update(MouseButton::Right, at(0, 0)), 1);
        assert_eq!(state.update(MouseButton::Left, at(0, 0)), 1);
        assert_eq!(state.update(MouseButton::Left, at(1, 1)), 2);
        assert_eq!(state.update(MouseButton::Left, at(1, 1)), 3);
        state.last_click -= Duration::from_millis(5000);
        assert_eq!(state.update(MouseButton::Left, at(1, 1)), 1);
        assert_eq!(state.update(MouseButton::Left, at(500, 1)), 1);
    }
}
