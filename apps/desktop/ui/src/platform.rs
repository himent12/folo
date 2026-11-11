// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform.rs
//! The platform contract (`ui/DESIGN.md`, Platform contract): `Platform`, `PlatformWindow`,
//! `PlatformDisplay`, `PlatformDispatcher`, `PlatformInputHandler`, and the values they exchange.
//! The backends live in the submodules (`windows`, `winit`, `headless`).

mod clipboard;
mod framebuffer;
pub mod oneshot;
mod options;

pub mod headless;
#[cfg(windows)]
pub mod windows;
#[cfg(any(not(windows), feature = "winit"))]
pub mod winit;

pub use clipboard::*;
pub use framebuffer::*;
pub use options::*;

use crate::{
    Bounds, Capslock, CursorStyle, DevicePixels, DispatchEventResult, Modifiers, Pixels, PlatformInput,
    PlatformTextSystem, Point, Size, WindowControlArea, point,
};
use anyhow::Result;
use async_task::Runnable;
use std::fmt::{self, Debug};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// The size a window gets when nothing else says (GPUI's `DEFAULT_WINDOW_SIZE`).
pub const DEFAULT_WINDOW_SIZE: Size<Pixels> = crate::size(crate::px(1536.), crate::px(864.));

/// An OS (or headless) platform: the run loop, displays, windows, clipboard, dialogs and the
/// dispatcher. One per process, used from the main thread.
pub trait Platform: 'static {
    /// The dispatcher executors run tasks on.
    fn dispatcher(&self) -> Arc<dyn PlatformDispatcher>;
    /// The text backend for this platform.
    fn text_system(&self) -> Arc<dyn PlatformTextSystem>;

    /// Runs the event loop; calls `on_finish_launching` once it is ready. Returns when the app quits.
    fn run(&self, on_finish_launching: Box<dyn 'static + FnOnce()>);
    /// Ends the event loop.
    fn quit(&self);
    /// Brings the app to the front.
    fn activate(&self, ignoring_other_apps: bool);

    /// Every connected display.
    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>>;
    /// The primary display, if there is one.
    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>>;

    /// Opens a window.
    fn open_window(&self, options: WindowParams) -> Result<Box<dyn PlatformWindow>>;
    /// The system's light / dark appearance.
    fn window_appearance(&self) -> WindowAppearance;

    /// Opens a URL in the default browser.
    fn open_url(&self, url: &str);
    /// Shows the file open dialog; the receiver gets the chosen paths (`None`: cancelled).
    fn prompt_for_paths(&self, options: PathPromptOptions)
    -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>>;
    /// Shows the file save dialog; the receiver gets the chosen path (`None`: cancelled).
    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>>;
    /// Shows `path` in the system file manager.
    fn reveal_path(&self, path: &Path);

    /// Sets the mouse cursor.
    fn set_cursor_style(&self, style: CursorStyle);
    /// Puts `item` on the clipboard.
    fn write_to_clipboard(&self, item: ClipboardItem);
    /// What the clipboard holds, if it is text.
    fn read_from_clipboard(&self) -> Option<ClipboardItem>;
}

/// A display (monitor) the platform can put windows on.
pub trait PlatformDisplay: Send + Sync + Debug {
    /// The display's id, for this run of the app.
    fn id(&self) -> DisplayId;

    /// A stable id across runs (GPUI's on Windows: a v5 UUID of the device name).
    fn uuid(&self) -> Result<Uuid>;

    /// The display's bounds in its own logical pixels.
    fn bounds(&self) -> Bounds<Pixels>;

    /// Where a window with no saved place opens: `DEFAULT_WINDOW_SIZE` (clipped) centred.
    fn default_bounds(&self) -> Bounds<Pixels> {
        let bounds = self.bounds();
        let center = bounds.center();
        let clipped_window_size = DEFAULT_WINDOW_SIZE.min(&bounds.size);

        let offset = clipped_window_size / 2.0;
        let origin = point(center.x - offset.width, center.y - offset.height);
        Bounds::new(origin, clipped_window_size)
    }
}

/// An opaque identifier for a hardware display.
#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct DisplayId(pub(crate) u32);

impl DisplayId {
    /// Wraps a backend's display number.
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

impl From<DisplayId> for u32 {
    fn from(id: DisplayId) -> Self {
        id.0
    }
}

impl Debug for DisplayId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisplayId({})", self.0)
    }
}

/// What a frame request asks of the window.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestFrameOptions {
    /// The platform needs a presented frame (for example after a resize).
    pub require_presentation: bool,
    /// Repaint everything, whatever is dirty.
    pub force_render: bool,
}

/// An OS window. The window (core) owns it and draws by `present`ing its framebuffer.
pub trait PlatformWindow: 'static {
    /// The window's outer bounds in screen logical pixels.
    fn bounds(&self) -> Bounds<Pixels>;
    /// Whether it is maximized.
    fn is_maximized(&self) -> bool;
    /// The restored bounds and the mode (windowed, maximized, full screen).
    fn window_bounds(&self) -> WindowBounds;
    /// The client area's size in logical pixels.
    fn content_size(&self) -> Size<Pixels>;
    /// Resizes the client area.
    fn resize(&mut self, size: Size<Pixels>);
    /// Device pixels per logical pixel.
    fn scale_factor(&self) -> f32;
    /// The window's light / dark appearance.
    fn appearance(&self) -> WindowAppearance;
    /// The display the window is on.
    fn display(&self) -> Option<Rc<dyn PlatformDisplay>>;
    /// The mouse position in window logical pixels.
    fn mouse_position(&self) -> Point<Pixels>;
    /// The modifier keys held now.
    fn modifiers(&self) -> Modifiers;
    /// The caps lock state.
    fn capslock(&self) -> Capslock;
    /// Sets the handler IME and text input go to.
    fn set_input_handler(&mut self, input_handler: Box<dyn PlatformInputHandler>);
    /// Takes the text input handler back.
    fn take_input_handler(&mut self) -> Option<Box<dyn PlatformInputHandler>>;
    /// Brings the window to the front and focuses it.
    fn activate(&self);
    /// Whether the window is the active one.
    fn is_active(&self) -> bool;
    /// Whether the mouse is over the window.
    fn is_hovered(&self) -> bool;
    /// Sets the title (taskbar, Alt-Tab).
    fn set_title(&mut self, title: &str);
    /// Minimizes the window.
    fn minimize(&self);
    /// Maximizes, or restores a maximized window.
    fn zoom(&self);
    /// Enters or leaves full screen.
    fn toggle_fullscreen(&self);
    /// Whether the window is full screen.
    fn is_fullscreen(&self) -> bool;

    /// Asks for one `on_request_frame` call at the next display refresh. Calls coalesce; a backend runs
    /// no timer while none is pending (`ui/DESIGN.md`, Frames).
    fn request_frame(&self);
    /// Called when the window should draw.
    fn on_request_frame(&self, callback: Box<dyn FnMut(RequestFrameOptions)>);
    /// Called for every input event; the result says whether the platform should handle it too.
    fn on_input(&self, callback: Box<dyn FnMut(PlatformInput) -> DispatchEventResult>);
    /// Called when the window becomes active or inactive.
    fn on_active_status_change(&self, callback: Box<dyn FnMut(bool)>);
    /// Called when the mouse enters or leaves the window.
    fn on_hover_status_change(&self, callback: Box<dyn FnMut(bool)>);
    /// Called with the new content size and scale factor.
    fn on_resize(&self, callback: Box<dyn FnMut(Size<Pixels>, f32)>);
    /// Called when the window moves.
    fn on_moved(&self, callback: Box<dyn FnMut()>);
    /// Called when the user asks to close; `false` keeps the window open.
    fn on_should_close(&self, callback: Box<dyn FnMut() -> bool>);
    /// Asks which window control area (caption, buttons) is under the mouse.
    fn on_hit_test_window_control(&self, callback: Box<dyn FnMut() -> Option<WindowControlArea>>);
    /// Called once when the window has closed.
    fn on_close(&self, callback: Box<dyn FnOnce()>);
    /// Called when the system appearance changes.
    fn on_appearance_changed(&self, callback: Box<dyn FnMut()>);

    /// Shows `frame` (the whole client area, device pixels). Only the `damage` rectangles changed since
    /// the previous call; the backend copies just those (all of `frame` after a resize or expose).
    fn present(&mut self, frame: &Framebuffer, damage: &[Bounds<DevicePixels>]);
    /// Called after a frame is complete.
    fn completed_frame(&self) {}

    /// Moves the IME candidate window next to `bounds` (window logical pixels).
    fn update_ime_position(&self, bounds: Bounds<Pixels>);

    /// Whether the window should type a key's text itself when nothing handled the key
    /// (`Window::insert_text_for_key`): only for a platform with no text input of its own (headless).
    /// Win32 (`WM_CHAR`, dead keys, the IME) and winit deliver typed text themselves, with the
    /// layout's Caps Lock, AltGr and dead-key handling, which a keystroke's `key_char` does not have.
    fn inserts_key_text(&self) -> bool {
        false
    }
}

/// Runs tasks: on a background pool, on the main thread, or after a delay.
pub trait PlatformDispatcher: Send + Sync {
    /// Whether the caller is on the main thread.
    fn is_main_thread(&self) -> bool;
    /// Runs `runnable` on a background thread (a pool that grows on demand and lets idle threads exit).
    fn dispatch(&self, runnable: Runnable);
    /// Runs `runnable` on the main thread.
    fn dispatch_on_main_thread(&self, runnable: Runnable);
    /// Runs `runnable` on a background thread after `duration`.
    fn dispatch_after(&self, duration: Duration, runnable: Runnable);
    /// The current time (a test dispatcher may fake it).
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// The text input side of the focused element, which IME and typed text go to. Ranges are in UTF-16
/// code units. The core adapts its `InputHandler` (which also takes the window and app) to this.
pub trait PlatformInputHandler: 'static {
    /// The selection, or `None` if there is no editable text (or input is disabled and not ignored).
    fn selected_text_range(&mut self, ignore_disabled_input: bool) -> Option<UTF16Selection>;
    /// The range of the IME composition, if any.
    fn marked_text_range(&mut self) -> Option<Range<usize>>;
    /// The text in `range_utf16`; `adjusted_range` gets the range actually returned.
    fn text_for_range(
        &mut self,
        range_utf16: Range<usize>,
        adjusted_range: &mut Option<Range<usize>>,
    ) -> Option<String>;
    /// Replaces `replacement_range` (or the marked text, or the selection) with `text`.
    fn replace_text_in_range(&mut self, replacement_range: Option<Range<usize>>, text: &str);
    /// Replaces a range with `new_text` and marks it as the IME composition.
    fn replace_and_mark_text_in_range(
        &mut self,
        range_utf16: Option<Range<usize>>,
        new_text: &str,
        new_selected_range: Option<Range<usize>>,
    );
    /// Ends the IME composition.
    fn unmark_text(&mut self);
    /// The window bounds of the text in `range_utf16`.
    fn bounds_for_range(&mut self, range_utf16: Range<usize>) -> Option<Bounds<Pixels>>;
    /// The UTF-16 index of the character at `point` (window logical pixels).
    fn character_index_for_point(&mut self, point: Point<Pixels>) -> Option<usize>;
}

/// A selection in UTF-16 code units; the head may be before the tail.
#[derive(Debug)]
pub struct UTF16Selection {
    /// The range of text in the document this selection corresponds to, in UTF-16 code units.
    pub range: Range<usize>,
    /// Whether the head of this selection is at the start (true), or end (false) of the range.
    pub reversed: bool,
}
