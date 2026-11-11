//! Moving between pages: navigate, back, the parent route, and the Ctrl+number shortcuts (inert
//! while a dialog is up).

use folo_ui::{Context, Window, px};

use crate::store::*;

use super::{FoloApp, Route};

impl FoloApp {
    // ---------- navigation ----------

    pub fn navigate(&mut self, route: Route, window: &mut Window, cx: &mut Context<Self>) {
        // The confusion-matrix cursor belongs to the model page it was set on.
        if matches!(route, Route::Model(_)) {
            self.cm_cell = None;
        }
        if route == self.route {
            return;
        }
        let prev = std::mem::replace(&mut self.route, route);
        // Runs opens on all runs, unless coming up from one of them: a filter (one the queue
        // toast set, say) doesn't wait there to make the list look empty later.
        if self.route == Route::Runs && !matches!(prev, Route::Run(_)) {
            self.run_filter = super::RunFilter::All;
        }
        self.history.push(prev);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
        self.after_navigate(window, cx);
    }

    pub fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(r) = self.history.pop() {
            self.route = r;
            self.after_navigate(window, cx);
        } else if let Some(parent) = self.parent_route() {
            self.route = parent;
            self.after_navigate(window, cx);
        }
    }

    pub fn parent_route(&self) -> Option<Route> {
        match &self.route {
            Route::Dataset(_) => Some(Route::Datasets),
            Route::Run(_) => Some(Route::Runs),
            Route::Model(_) => Some(Route::Models),
            Route::Bench(_) => Some(Route::Benches),
            Route::BenchEntry(b, _) => Some(Route::Bench(b.clone())),
            _ => None,
        }
    }

    fn after_navigate(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Settings shows the throughput table and the cloud as the server has them now (also when
        // going back).
        if self.route == Route::Settings && !self.throughput.is_loading() {
            self.throughput = Load::Idle;
        }
        if self.route == Route::Settings {
            self.reload_cloud();
        }
        self.after_bench_navigate();
        self.page_scroll.set_offset(folo_ui::point(px(0.), px(0.)));
        self.chart_hover = None;
        self.export_open = false;
        self.pr_pin = None;
        self.pr_focus = None;
        self.menu_open = None;
        self.sample_epoch = None;
        if !self.focus_within_page(window) {
            window.focus(&self.root_focus);
        }
        cx.notify();
    }

    fn focus_within_page(&self, _window: &Window) -> bool {
        false
    }

    /// A dialog, the stop confirmation or the inspector is up: the page behind is inert.
    pub fn modal_open(&self) -> bool {
        self.new_run.is_some()
            || self.synth.is_some()
            || self.confirm_stop.is_some()
            || self.inspector.is_some()
            || self.cloud.connect.is_some()
            || self.cloud.disconnect.is_some()
            || self.cloud.terminate.is_some()
    }

    pub(super) fn nav_shortcut(&mut self, route: Route, window: &mut Window, cx: &mut Context<Self>) {
        if !self.modal_open() {
            self.palette = None;
            self.navigate(route, window, cx);
        }
    }
}
