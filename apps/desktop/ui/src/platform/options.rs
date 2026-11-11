// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform.rs
//! Window options: `WindowOptions` (what the app passes), `WindowParams` (what a backend gets),
//! `TitlebarOptions`, `WindowBounds`, `WindowKind`, `WindowAppearance`, `PathPromptOptions`.

use crate::{Bounds, DisplayId, Pixels, Point, SharedString, Size};

/// The variables that can be configured when creating a new window.
#[derive(Debug)]
pub struct WindowOptions {
    /// The bounds and mode of the window (`None`: the display's default bounds, windowed).
    pub window_bounds: Option<WindowBounds>,

    /// The titlebar configuration of the window.
    pub titlebar: Option<TitlebarOptions>,

    /// Whether the window should be focused when created.
    pub focus: bool,

    /// Whether the window should be shown when created.
    pub show: bool,

    /// The kind of window to create.
    pub kind: WindowKind,

    /// Whether the window should be movable by the user.
    pub is_movable: bool,

    /// Whether the window should be resizable by the user.
    pub is_resizable: bool,

    /// Whether the window should be minimized by the user.
    pub is_minimizable: bool,

    /// The display to create the window on (`None`: the primary display).
    pub display_id: Option<DisplayId>,

    /// The appearance of the window background.
    pub window_background: WindowBackgroundAppearance,

    /// Application identifier of the window (Linux: the Wayland app id / X11 class).
    pub app_id: Option<String>,

    /// Window minimum size.
    pub window_min_size: Option<Size<Pixels>>,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self {
            window_bounds: None,
            titlebar: Some(TitlebarOptions {
                title: Default::default(),
                appears_transparent: Default::default(),
                traffic_light_position: Default::default(),
            }),
            focus: true,
            show: true,
            kind: WindowKind::Normal,
            is_movable: true,
            is_resizable: true,
            is_minimizable: true,
            display_id: None,
            window_background: WindowBackgroundAppearance::default(),
            app_id: None,
            window_min_size: None,
        }
    }
}

/// The variables a platform backend gets to open a window (resolved from `WindowOptions`).
#[derive(Debug)]
pub struct WindowParams {
    /// The window's bounds in screen logical pixels (the restored bounds when maximized).
    pub bounds: Bounds<Pixels>,

    /// The titlebar configuration of the window.
    pub titlebar: Option<TitlebarOptions>,

    /// The kind of window to create.
    pub kind: WindowKind,

    /// Whether the window should be movable by the user.
    pub is_movable: bool,

    /// Whether the window should be resizable by the user.
    pub is_resizable: bool,

    /// Whether the window should be minimized by the user.
    pub is_minimizable: bool,

    /// Whether the window should be focused when created.
    pub focus: bool,

    /// Whether the window should be shown when created.
    pub show: bool,

    /// The display to create the window on.
    pub display_id: Option<DisplayId>,

    /// Window minimum size.
    pub window_min_size: Option<Size<Pixels>>,

    /// Application identifier of the window (GPUI sets it after creation; winit needs it up front).
    pub app_id: Option<String>,
}

/// Represents the status of how a window should be opened.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum WindowBounds {
    /// Indicates that the window should open in a windowed state with the given bounds.
    Windowed(Bounds<Pixels>),
    /// Indicates that the window should open in a maximized state; the bounds are its restored bounds.
    Maximized(Bounds<Pixels>),
    /// Indicates that the window should open in fullscreen mode; the bounds are its restored bounds.
    Fullscreen(Bounds<Pixels>),
}

impl Default for WindowBounds {
    fn default() -> Self {
        WindowBounds::Windowed(Bounds::default())
    }
}

impl WindowBounds {
    /// Retrieve the inner bounds.
    pub fn get_bounds(&self) -> Bounds<Pixels> {
        match self {
            WindowBounds::Windowed(bounds) => *bounds,
            WindowBounds::Maximized(bounds) => *bounds,
            WindowBounds::Fullscreen(bounds) => *bounds,
        }
    }
}

/// The options that can be configured for a window's titlebar.
#[derive(Debug, Default)]
pub struct TitlebarOptions {
    /// The initial title of the window.
    pub title: Option<SharedString>,

    /// Should the default system titlebar be hidden to allow for a custom-drawn titlebar?
    pub appears_transparent: bool,

    /// The position of the macOS traffic light buttons.
    pub traffic_light_position: Option<Point<Pixels>>,
}

/// The kind of window to create.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WindowKind {
    /// A normal application window.
    Normal,

    /// A window that appears above all other windows, usually used for alerts or popups.
    PopUp,

    /// A floating window that appears on top of its parent window.
    Floating,
}

/// The appearance of the window, as defined by the operating system.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WindowAppearance {
    /// A light appearance.
    #[default]
    Light,

    /// A light appearance with vibrant colors (macOS).
    VibrantLight,

    /// A dark appearance.
    Dark,

    /// A dark appearance with vibrant colors (macOS).
    VibrantDark,
}

/// The appearance of the background of the window itself.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum WindowBackgroundAppearance {
    /// Opaque: the content fully covers the window.
    #[default]
    Opaque,
    /// Plain alpha transparency.
    Transparent,
    /// Transparency, with the background behind the window blurred.
    Blurred,
}

/// The options that can be configured for a file dialog prompt.
#[derive(Clone, Debug)]
pub struct PathPromptOptions {
    /// Should the prompt allow files to be selected?
    pub files: bool,
    /// Should the prompt allow directories to be selected?
    pub directories: bool,
    /// Should the prompt allow multiple files to be selected?
    pub multiple: bool,
    /// The prompt to show to a user when selecting a path.
    pub prompt: Option<SharedString>,
}
