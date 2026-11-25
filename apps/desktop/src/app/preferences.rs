//! The colour theme (design/THEMES.md: the mode, each mode's theme, custom themes), and the window
//! state remembered between launches.

use folo_ui::{App, Context, WeakEntity, Window, WindowAppearance};

use crate::theme::{Catalog, Mode, ModePref};

use super::FoloApp;

/// Whether an OS appearance is dark.
pub fn is_dark(a: WindowAppearance) -> bool {
    matches!(a, WindowAppearance::Dark | WindowAppearance::VibrantDark)
}

/// Registers `window`'s handler for the OS appearance: from then on System follows it, through
/// folo-ui's delivery (Win32 `WM_SETTINGCHANGE`, winit's theme event, a headless test's change).
pub fn follow_system_appearance(window: &mut Window, this: WeakEntity<FoloApp>, cx: &App) {
    window.on_appearance_changed(cx, move |window, cx| {
        let dark = is_dark(window.appearance());
        this.update(cx, |app, cx| app.system_appearance_changed(dark, cx)).ok();
    });
}

impl FoloApp {
    /// Shows the theme the choice resolves to now (the OS appearance decides System's mode).
    pub fn apply_theme(&mut self, cx: &mut Context<Self>) {
        let t = self.themes.resolve(&self.theme_prefs, self.system_dark);
        self.mode = t.mode;
        cx.set_global(t);
        cx.refresh_windows();
    }

    /// Settings › Theme: Dark, Light or System.
    pub fn set_mode(&mut self, mode: ModePref, cx: &mut Context<Self>) {
        self.theme_prefs.mode = mode;
        self.apply_theme(cx);
    }

    /// Ctrl Shift L and the palette's switch: the other mode than the one shown (System becomes that
    /// explicit mode).
    pub fn toggle_theme(&mut self, cx: &mut Context<Self>) {
        self.theme_prefs.toggle(self.system_dark);
        self.apply_theme(cx);
    }

    /// Settings › Dark theme / Light theme and the palette's "Theme: …": theme `id` becomes its mode's
    /// theme, and the mode switches to it unless it is System.
    pub fn pick_theme(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(mode) = self.themes.get(id).map(|e| e.mode) {
            self.theme_prefs.pick(mode, id);
            self.apply_theme(cx);
        }
    }

    /// The OS switched between dark and light: System follows it at once.
    pub fn system_appearance_changed(&mut self, dark: bool, cx: &mut Context<Self>) {
        if std::mem::replace(&mut self.system_dark, dark) != dark && self.theme_prefs.mode == ModePref::System
        {
            self.apply_theme(cx);
        }
    }

    /// Settings › Custom themes › Reload: reads the themes folder again, off the UI thread, then shows
    /// the (maybe changed) theme.
    pub fn reload_themes(&mut self, cx: &mut Context<Self>) {
        if self.themes_loading {
            return;
        }
        let folder = self.themes.folder.clone().unwrap_or_else(crate::theme::catalog::custom_folder);
        self.themes_loading = true;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let catalog = cx.background_executor().spawn(async move { Catalog::load(&folder) }).await;
            this.update(cx, |this, cx| {
                this.themes = catalog;
                this.themes_loading = false;
                this.apply_theme(cx);
            })
            .ok();
        })
        .detach();
    }

    /// Draws the window's system frame (DWM's caption and border on Windows) in the mode shown, once
    /// per change (from render).
    pub(crate) fn sync_frame_appearance(&mut self, window: &Window) {
        if self.frame_mode != Some(self.mode) {
            self.frame_mode = Some(self.mode);
            window.set_frame_appearance(Some(match self.mode {
                Mode::Dark => WindowAppearance::Dark,
                Mode::Light => WindowAppearance::Light,
            }));
        }
    }

    /// The window is asked to close (its Close button, Alt F4, full screen's Close). Closing stops the
    /// `folo serve` this app started, and every run of this machine training or waiting on it: that
    /// server is asked (whatever server the window shows, and however old its list), off the UI thread,
    /// and the window stays until it answers. It closes then if nothing would stop; else, or when the
    /// server doesn't answer in time, it asks first. `false` keeps it open for now.
    pub fn should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.confirm_close.is_some() || self.close_checking {
            return false;
        }
        let Some(url) = self.server_proc.as_mut().and_then(|p| p.running().then(|| p.url.clone())) else {
            // Nothing of ours runs: nothing stops.
            self.save_window_state(window, cx);
            return true;
        };
        self.close_checking = true;
        let handle = window.window_handle();
        cx.spawn(async move |this, cx| {
            let stops =
                cx.background_executor()
                    .spawn(async move {
                        crate::api::http::runs_now(&url).map(|r| crate::overlays::closing_stops(&r))
                    })
                    .await;
            handle
                .update(cx, |_, window, cx| {
                    this.update(cx, |this, cx| {
                        this.close_checking = false;
                        if this.should_close_with(stops, window, cx) {
                            window.remove_window();
                        }
                    })
                })
                .ok();
        })
        .detach();
        false
    }

    /// [`Self::should_close`] with the runs closing would stop (`None`: the server couldn't say):
    /// `true` when nothing would. Else the question opens, alone: every other overlay closes first (New
    /// run is kept as a draft, as Esc keeps it), so nothing under it takes keys, focus or Esc.
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
            self.menu_open = None;
            self.export_open = false;
            self.palette = None;
            self.confirm_stop = None;
            self.confirm_delete = None;
            self.cloud.connect = None;
            self.cloud.disconnect = None;
            self.cloud.terminate = None;
            self.new_run_draft =
                self.new_run.take().and_then(|f| self.keep_draft(f)).or(self.new_run_draft.take());
            self.synth = None;
            self.take_inspector(cx);
            self.overlay_return = None;
            self.confirm_close = Some(stops.unwrap_or_default());
            let fh = self.fh("close-cancel", cx);
            window.focus(&fh);
            cx.notify();
        }
        false
    }

    /// Keep open: the question goes, focus goes back to the page.
    pub fn keep_open(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.confirm_close = None;
        window.focus(&self.root_focus);
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
        let mut state = State {
            window: Some(Placement {
                x: b.origin.x.into(),
                y: b.origin.y.into(),
                width: b.size.width.into(),
                height: b.size.height.into(),
                mode,
                display: window.display(cx).and_then(|d| d.uuid().ok()).map(|u| u.to_string()),
            }),
            theme: None,
            dark_theme: None,
            light_theme: None,
            spawn_server: Some(self.spawn_server),
            server_url: {
                let u = self.server_url.read(cx).text().trim().to_string();
                (!u.is_empty()).then_some(u)
            },
            // Kept only when picked against the OS setting, so an untouched one keeps following it.
            reduce_motion: (self.reduce_motion != crate::ui::os_prefers_reduced_motion())
                .then_some(self.reduce_motion),
        };
        state.set_theme_prefs(&self.theme_prefs);
        if let Err(e) = crate::window_state::save(path, &state) {
            eprintln!("folo-desktop: couldn't remember the window in {}: {e}", path.display());
        }
    }
}
