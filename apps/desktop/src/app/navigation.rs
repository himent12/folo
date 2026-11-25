//! Moving between pages: navigate, back, the parent route, and the Ctrl+number shortcuts (inert
//! while a dialog is up).

use folo_ui::{Context, Window, px};

use crate::store::*;
use crate::ui::Tone;

use super::{FoloApp, Route};

/// The list to open instead of `route` when it is the page of something in `deleted`, with that
/// thing's name.
pub(super) fn deleted_route(deleted: &super::deleted::Deleted, route: &Route) -> Option<(String, Route)> {
    let (id, list) = match route {
        Route::Run(id) => (id, Route::Runs),
        Route::Model(id) => (id, Route::Models),
        Route::Dataset(id) => (id, Route::Datasets),
        _ => return None,
    };
    deleted.get(id).map(|name| (name.clone(), list))
}

/// What a link to `name`, gone, says: a dataset "was removed" (unregistered; its files may well be
/// there), a run or a model "was deleted".
pub(super) fn gone_title(name: &str, route: &Route) -> String {
    match route {
        Route::Dataset(_) => format!("{name} was removed"),
        _ => format!("{name} was deleted"),
    }
}

impl FoloApp {
    // ---------- navigation ----------

    pub fn navigate(&mut self, route: Route, window: &mut Window, cx: &mut Context<Self>) {
        // A link to something deleted since it was made opens the list, and says why.
        let route = match deleted_route(&self.deleted, &route) {
            Some((name, list)) => {
                self.toast(Tone::Neutral, gone_title(&name, &route), None, cx);
                list
            }
            None => route,
        };
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
            self.run_arch_filter = None;
            self.run_search.update(cx, |i, cx| i.set_text("", cx));
        }
        self.history.push(prev);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
        self.after_navigate(window, cx);
    }

    pub fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Past the pages of what was deleted since, and the page already open.
        while let Some(r) = self.history.pop() {
            if r != self.route && deleted_route(&self.deleted, &r).is_none() {
                self.route = r;
                self.after_navigate(window, cx);
                return;
            }
        }
        if let Some(parent) = self.parent_route() {
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
        if let Route::Dataset(id) = &self.route
            && self.ds_img_dataset.as_deref() != Some(id.as_str())
        {
            self.ds_img_dataset = Some(id.clone());
            self.ds_img_query.clear();
            self.ds_img_search.update(cx, |i, cx| i.set_text("", cx));
            self.ds_img_sort = crate::screens::datasets::ImageSort::Default;
        }
        // The pages with dataset cards show each dataset's health as the server has it now: a
        // folder moved away (or put back) since it was read says so on the next visit.
        if matches!(self.route, Route::Overview | Route::Datasets | Route::Dataset(_)) {
            self.ds_recheck = self
                .store
                .dataset
                .iter()
                .filter(|(_, l)| !l.is_loading())
                .map(|(id, _)| id.clone())
                .collect();
        }
        self.page_scroll.set_offset(folo_ui::point(px(0.), px(0.)));
        self.chart_hover = None;
        self.chart_hover_on = None;
        self.export_open = false;
        self.pr_pin = None;
        self.pr_focus = None;
        self.menu_open = None;
        self.run_x_fit = None;
        self.log_filter = crate::app::LogFilter::All;
        self.log_search.update(cx, |i, cx| i.set_text("", cx));
        // An earlier epoch's samples were picked: the next run page starts on the latest again, and a
        // request for the picked epoch still on its way is dropped when it lands.
        if self.sample_epoch.take().is_some() {
            self.store.samples.clear();
            self.history_tickets.retain(|(_, what), _| *what != "samples");
        }
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
            || self.confirm_delete.is_some()
            || self.confirm_close.is_some()
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
