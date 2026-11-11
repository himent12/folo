//! The colour theme, and the window state remembered between launches.

use gpui::{App, Context, Window};

use crate::theme::{Mode, Theme};

use super::FoloApp;

impl FoloApp {
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
    pub fn save_window_state(&self, window: &Window, cx: &App) {
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
                display: window.display(cx).and_then(|d| d.uuid().ok()).map(|u| u.to_string()),
            }),
            theme: Some(if self.mode == Mode::Light { "light" } else { "dark" }.into()),
            spawn_server: Some(self.spawn_server),
        };
        if let Err(e) = crate::window_state::save(path, &state) {
            eprintln!("folo-desktop: couldn't remember the window in {}: {e}", path.display());
        }
    }
}
