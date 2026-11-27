//! Moving between pages: navigate, back, the parent route, and the Ctrl+number shortcuts (inert
//! while a dialog is up).

use folo_ui::{Context, Window, px};

use crate::store::*;
use crate::ui::Tone;

use super::FoloApp;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Route {
    Overview,
    Datasets,
    Dataset(String),
    Runs,
    Run(String),
    /// Side-by-side comparison of 2–4 training runs.
    RunCompare,
    Models,
    Model(String),
    /// Vision benchmarks (servers with `vision_bench`).
    Benches,
    Bench(String),
    /// An entry of a benchmark: (benchmark, entry).
    BenchEntry(String, String),
    /// Side-by-side comparison of finished entries on a benchmark.
    BenchCompare(String),
    Playground,
    /// Agent labeling jobs (servers with `agent_labeling`).
    LabelJobs,
    LabelJob(String),
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Section {
    Overview,
    Datasets,
    Runs,
    Models,
    Benchmarks,
    Playground,
    Labeling,
    Settings,
}

impl Route {
    /// `overview`, `datasets`, `dataset:ID`, `runs`, `run:ID`, `runs-compare`, `models`, `model:ID`,
    /// `benchmarks`, `bench:ID`, `bench-entry:ID/ENTRY`, `bench-compare:ID`, `playground`, `labeling`,
    /// `label:ID`, `settings`.
    pub fn parse(s: &str) -> Option<Route> {
        let valid_id = |v: &str| !v.trim().is_empty();
        let (kind, id) = s.split_once(':').map_or((s, None), |(k, i)| (k, Some(i)));
        Some(match (kind, id) {
            ("overview", None) => Route::Overview,
            ("datasets", None) => Route::Datasets,
            ("dataset", Some(id)) if valid_id(id) => Route::Dataset(id.to_string()),
            ("runs", None) => Route::Runs,
            ("run", Some(id)) if valid_id(id) => Route::Run(id.to_string()),
            ("runs-compare" | "run-compare", None) => Route::RunCompare,
            ("models", None) => Route::Models,
            ("model", Some(id)) if valid_id(id) => Route::Model(id.to_string()),
            ("benchmarks" | "benches", None) => Route::Benches,
            ("bench", Some(id)) if valid_id(id) => Route::Bench(id.to_string()),
            ("bench-entry", Some(id)) => {
                let (b, e) = id.split_once('/')?;
                if !valid_id(b) || !valid_id(e) || e.contains('/') {
                    return None;
                }
                Route::BenchEntry(b.to_string(), e.to_string())
            }
            ("bench-compare", Some(id)) if valid_id(id) => Route::BenchCompare(id.to_string()),
            ("playground", None) => Route::Playground,
            ("labeling" | "label", None) => Route::LabelJobs,
            ("label", Some(id)) if valid_id(id) => Route::LabelJob(id.to_string()),
            ("settings", None) => Route::Settings,
            _ => return None,
        })
    }

    pub fn section(&self) -> Section {
        match self {
            Route::Overview => Section::Overview,
            Route::Datasets | Route::Dataset(_) => Section::Datasets,
            Route::Runs | Route::Run(_) | Route::RunCompare => Section::Runs,
            Route::Models | Route::Model(_) => Section::Models,
            Route::Benches | Route::Bench(_) | Route::BenchEntry(..) | Route::BenchCompare(_) => {
                Section::Benchmarks
            }
            Route::Playground => Section::Playground,
            Route::LabelJobs | Route::LabelJob(_) => Section::Labeling,
            Route::Settings => Section::Settings,
        }
    }
}

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
        // Guard unsaved bounding-box edits before leaving an active, renderable labeling job.
        if self.label_route_has_renderable_dirty_draft() {
            self.label.editor.pending_switch = Some(crate::label::PendingImageSwitch::Navigate(route));
            cx.notify();
            return;
        }
        let prev = std::mem::replace(&mut self.route, route);
        // Runs opens on all runs, unless coming up from one of them: a filter (one the queue
        // toast set, say) doesn't wait there to make the list look empty later.
        if self.route == Route::Runs && !matches!(prev, Route::Run(_) | Route::RunCompare) {
            self.run_filter = super::RunFilter::All;
            self.run_arch_filter = None;
            self.run_search.update(cx, |i, cx| i.set_text("", cx));
        }
        self.history.push(prev);
        if self.history.len() > 50 {
            self.history.remove(0);
        }
        self.forward_history.clear();
        self.after_navigate(window, cx);
    }

    /// Whether `route` points to a run, model, dataset, benchmark, or benchmark entry removed in this session.
    fn route_gone(&self, route: &Route) -> bool {
        if deleted_route(&self.deleted, route).is_some() {
            return true;
        }
        match route {
            Route::Bench(id) | Route::BenchCompare(id) => self.bench.removed.contains_key(id),
            Route::BenchEntry(b, e) => {
                self.bench.removed.contains_key(b) || self.bench.removed.contains_key(e)
            }
            Route::LabelJob(id) => self.label.removed.contains(id),
            _ => false,
        }
    }

    pub fn back(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.label_route_has_renderable_dirty_draft() {
            self.label.editor.pending_switch = Some(crate::label::PendingImageSwitch::Back);
            cx.notify();
            return;
        }
        // Past the pages of what was deleted since, and the page already open.
        while let Some(r) = self.history.pop() {
            if r != self.route && !self.route_gone(&r) {
                let prev = std::mem::replace(&mut self.route, r);
                self.forward_history.push(prev);
                if self.forward_history.len() > 50 {
                    self.forward_history.remove(0);
                }
                self.after_navigate(window, cx);
                return;
            }
        }
        if let Some(parent) = self.parent_route() {
            let target = if self.route_gone(&parent) { Route::Benches } else { parent };
            let prev = std::mem::replace(&mut self.route, target);
            self.forward_history.push(prev);
            if self.forward_history.len() > 50 {
                self.forward_history.remove(0);
            }
            self.after_navigate(window, cx);
        }
    }

    /// Re-opens the page most recently left with [`Self::back`] (`Alt+→`), skipping any route that
    /// was deleted since or matches the current page.
    pub fn forward(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.label_route_has_renderable_dirty_draft()
            && self.forward_history.iter().any(|r| r != &self.route && !self.route_gone(r))
        {
            self.label.editor.pending_switch = Some(crate::label::PendingImageSwitch::Forward);
            cx.notify();
            return;
        }
        while let Some(r) = self.forward_history.pop() {
            if r != self.route && !self.route_gone(&r) {
                let prev = std::mem::replace(&mut self.route, r);
                self.history.push(prev);
                if self.history.len() > 50 {
                    self.history.remove(0);
                }
                self.after_navigate(window, cx);
                return;
            }
        }
    }

    pub fn parent_route(&self) -> Option<Route> {
        match &self.route {
            Route::Dataset(_) => Some(Route::Datasets),
            Route::Run(_) | Route::RunCompare => Some(Route::Runs),
            Route::Model(_) => Some(Route::Models),
            Route::Bench(_) => Some(Route::Benches),
            Route::BenchEntry(b, _) | Route::BenchCompare(b) => Some(Route::Bench(b.clone())),
            Route::LabelJob(_) => Some(Route::LabelJobs),
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
        self.after_label_navigate();
        if self.route != Route::Playground {
            self.cancel_play_batch();
            self.play_batch.load_gen += 1;
        }
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
        self.ds_scroll_anchor = None;
        self.chart_hover = None;
        self.chart_hover_on = None;
        self.run_compare.hover = None;
        self.run_compare.hover_on = None;
        if self.route != Route::Runs {
            self.run_compare.select_mode = false;
        }
        self.export_open = false;
        self.model_cli.onnx_head_o2m = false;
        self.model_cli.predict_int8_target = None;
        self.pr_pin = None;
        self.pr_focus = None;
        self.class_table_sort = None;
        self.menu_open = None;
        self.menu_typeahead.clear();
        self.palette = None;
        self.overlay_return = None;
        self.verify_focus = None;
        self.run_x_fit = None;
        self.sample_error_filter = crate::screens::runs::SampleErrorFilter::All;
        self.sample_sort = crate::screens::runs::SampleSort::Default;
        self.log_filter = crate::app::LogFilter::All;
        self.log_search.update(cx, |i, cx| i.set_text("", cx));
        // An earlier epoch's samples were picked: the next run page starts on the latest again, and a
        // request for the picked epoch still on its way is dropped when it lands.
        if self.sample_epoch.take().is_some() {
            self.store.samples.clear();
            self.store.samples_stale.clear();
            self.store.samples_refreshing.clear();
            self.store.samples_refresh_error.clear();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::{drive, open};

    #[test]
    fn back_and_forward_round_trip_and_clear_on_new_navigation() {
        let (mut app, window) = open();
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Datasets, w, cx);
            this.navigate(Route::Runs, w, cx);
            this.navigate(Route::Models, w, cx);
            assert_eq!(this.route, Route::Models);
            assert!(this.forward_history.is_empty());

            // Back pushes Models onto forward_history.
            this.back(w, cx);
            assert_eq!(this.route, Route::Runs);
            this.back(w, cx);
            assert_eq!(this.route, Route::Datasets);
            assert_eq!(this.forward_history, vec![Route::Models, Route::Runs]);

            // Forward restores Runs, then Models.
            this.forward(w, cx);
            assert_eq!(this.route, Route::Runs);
            this.forward(w, cx);
            assert_eq!(this.route, Route::Models);
            assert!(this.forward_history.is_empty());

            // Back once, then navigating to a new page clears forward_history.
            this.back(w, cx);
            assert_eq!(this.route, Route::Runs);
            assert_eq!(this.forward_history, vec![Route::Models]);
            this.navigate(Route::Settings, w, cx);
            assert_eq!(this.route, Route::Settings);
            assert!(this.forward_history.is_empty());

            // Forward when empty is a no-op.
            this.forward(w, cx);
            assert_eq!(this.route, Route::Settings);

            // Forward skips deleted benchmarks and entries recorded in `bench.removed`.
            this.navigate(Route::Bench("bn_gone".into()), w, cx);
            this.navigate(Route::BenchEntry("bn_ok".into(), "ent_gone".into()), w, cx);
            this.back(w, cx);
            this.back(w, cx);
            assert_eq!(this.route, Route::Settings);
            this.bench.removed.insert("bn_gone".into(), 1);
            this.bench.removed.insert("ent_gone".into(), 2);
            this.forward(w, cx);
            assert_eq!(this.route, Route::Settings, "removed benchmark and entry are skipped");
            assert!(this.forward_history.is_empty());
        });
    }
}
