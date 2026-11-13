//! The colour theme, and the window state remembered between launches.

use folo_ui::{App, Context, Window};

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

    /// The window is asked to close (its Close button, Alt F4, full screen's Close). Closing stops the
    /// `folo serve` this app started, and every run of this machine training or waiting on it: that
    /// server is asked now (whatever server the window shows, and however old its list), and when a run
    /// would stop, or it doesn't answer in time, the window asks first (`false` keeps it open).
    pub fn should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.confirm_close.is_some() {
            return false;
        }
        let own = self.server_proc.as_mut().map(|p| (p.running(), p.url.clone()));
        let stops = match own {
            Some((true, url)) => {
                crate::api::http::runs_now(&url).map(|runs| crate::overlays::closing_stops(&runs))
            }
            // Nothing of ours runs: nothing stops.
            _ => Some(Vec::new()),
        };
        self.should_close_with(stops, window, cx)
    }

    /// [`Self::should_close`] with the runs closing would stop (`None`: the server couldn't say).
    pub(crate) fn should_close_with(
        &mut self,
        stops: Option<Vec<String>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        if stops.as_ref().is_some_and(|s| s.is_empty()) {
            self.save_window_state(window, cx);
            return true;
        }
        if self.confirm_close.is_none() {
            // Its own return place: an overlay under it (a Stop confirmation, New run) keeps its own.
            self.close_return = window.focused(cx);
            self.palette = None;
            self.close_unchecked = stops.is_none();
            self.confirm_close = Some(stops.unwrap_or_default());
            let fh = self.fh("close-cancel", cx);
            window.focus(&fh);
            cx.notify();
        }
        false
    }

    /// The window shows the server this app started.
    pub fn shows_own_server(&self) -> bool {
        self.server_proc.as_ref().is_some_and(|p| self.local_url.as_deref() == Some(p.url.as_str()))
    }

    /// Keep open: the question goes, focus goes back to where it was.
    pub fn keep_open(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.confirm_close = None;
        match self.close_return.take() {
            Some(fh) => window.focus(&fh),
            None => window.focus(&self.root_focus),
        }
        cx.notify();
    }

    /// Close the window after "Stop and close": remembered as any close is.
    pub fn close_now(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.save_window_state(window, cx);
        window.remove_window();
    }

    /// Remember the window (restored bounds, maximized or full screen), the theme and a Reduce motion
    /// that differs from the OS's for the next launch, when this one remembers (`window_state`).
    pub fn save_window_state(&self, window: &Window, cx: &App) {
        use crate::window_state::{Placement, State, WindowMode};
        let Some(path) = &self.state_path else {
            return;
        };
        let (b, mode) = match window.window_bounds() {
            folo_ui::WindowBounds::Windowed(b) => (b, WindowMode::Windowed),
            folo_ui::WindowBounds::Maximized(b) => (b, WindowMode::Maximized),
            folo_ui::WindowBounds::Fullscreen(b) => (b, WindowMode::Fullscreen),
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
            // Kept only when picked against the OS setting, so an untouched one keeps following it.
            reduce_motion: (self.reduce_motion != crate::ui::os_prefers_reduced_motion())
                .then_some(self.reduce_motion),
        };
        if let Err(e) = crate::window_state::save(path, &state) {
            eprintln!("folo-desktop: couldn't remember the window in {}: {e}", path.display());
        }
    }
}
