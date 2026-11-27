//! Loading from the backend: the background fetch helper, core list/system/run fetches, and
//! submodules for dataset paging (`datasets.rs`), decoded image queuing (`images.rs`), and live
//! run streams (`streams.rs`).

mod datasets;
mod images;
mod streams;

#[cfg(test)]
pub(crate) use datasets::IMAGE_PAGE;
pub(crate) use datasets::{DatasetScrollAnchor, FailedImagePage};

use folo_ui::Context;

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::lists::List;
use super::{FoloApp, Route};

impl FoloApp {
    // ---------- data ----------

    /// Run a backend call on the background executor and apply its result on the main thread.
    pub fn fetch<T: Send + 'static>(
        &mut self,
        cx: &mut Context<Self>,
        call: impl FnOnce(&dyn Backend) -> ApiResult<T> + Send + 'static,
        apply: impl FnOnce(&mut Self, ApiResult<T>, &mut Context<Self>) + 'static,
    ) {
        if self.hold_loading {
            return;
        }
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        let reconnects = self.reconnects;
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async move { call(&*api) }).await;
            this.update(cx, |this, cx| {
                // Answered by a source we've since switched away from: not ours any more.
                if this.backend_gen != generation {
                    return;
                }
                // Sent while the server was away and failed after it answered again: the reconnect
                // had nothing to clear yet, so the failure it installs is cleared now (the page asks
                // again when shown; it kept "Could not reach folo serve" with the server healthy).
                let late =
                    this.reconnects != reconnects && matches!(&r, Err(e) if e.code == "server_unreachable");
                apply(this, r, cx);
                this.trace_first_data();
                if late && !this.store.unreachable() {
                    this.clear_unreachable_dialogs();
                    this.forget_unreachable_loads(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// `FOLO_UI_TRACE`: says once when the lists every page starts from (runs, datasets and models) have
    /// all arrived (`first-lists`) and when the frame showing them is presented (`first-lists-frame`), and
    /// the same for `/system` (`system`, `system-frame`: Overview's hardware card). The image queue says when it
    /// first runs dry (`images-drained`, `images-drained-frame`: every image asked for so far was answered, which
    /// on a server that serves them all is the open grid's thumbnails on screen; a failed one asked again later
    /// is not waited for, so its placeholder may still show).
    fn trace_first_data(&self) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static LISTS: AtomicBool = AtomicBool::new(false);
        static SYSTEM: AtomicBool = AtomicBool::new(false);
        let s = &self.store;
        let lists = s.runs.ready().is_some() && s.datasets.ready().is_some() && s.models.ready().is_some();
        if lists && !LISTS.swap(true, Ordering::Relaxed) {
            folo_ui::trace_point("first-lists");
            folo_ui::trace_at_next_frame("first-lists-frame");
        }
        if s.system.ready().is_some() && !SYSTEM.swap(true, Ordering::Relaxed) {
            folo_ui::trace_point("system");
            folo_ui::trace_at_next_frame("system-frame");
        }
    }

    /// Fetch the model list again after it failed (nothing else retries a failed load).
    pub fn retry_models(&mut self, cx: &mut Context<Self>) {
        // An answer still out (the refresh's) would undo the request asked for now.
        self.list_tickets.restart(List::Models);
        self.store.models = Load::Idle;
        self.ensure_core(cx);
        cx.notify();
    }

    /// Fetch the dataset list again after it failed.
    pub fn retry_datasets(&mut self, cx: &mut Context<Self>) {
        self.list_tickets.restart(List::Datasets);
        self.store.datasets = Load::Idle;
        self.ensure_core(cx);
        cx.notify();
    }

    pub fn ensure_core(&mut self, cx: &mut Context<Self>) {
        if self.store.runs.needs_fetch() {
            self.store.runs = Load::Loading;
            self.fetch_list(List::Runs, cx);
        }
        if self.store.datasets.needs_fetch() {
            self.store.datasets = Load::Loading;
            self.fetch_list(List::Datasets, cx);
        }
        if self.store.models.needs_fetch() {
            self.store.models = Load::Loading;
            self.fetch_list(List::Models, cx);
        }
        if self.store.system.needs_fetch() && !self.system_out {
            self.store.system = Load::Loading;
            self.fetch_system(cx);
        }
        if self.store.health.needs_fetch() {
            self.store.health = Load::Loading;
            self.fetch(
                cx,
                |api| api.health(),
                |this, r, _| {
                    this.reached |= r.is_ok();
                    // The refresh may have said meanwhile that the server is gone or wants a token
                    // (`/api/health` is the one call a token server answers): that stands.
                    if this.store.health.is_loading() {
                        this.store.health = r.into();
                    }
                },
            );
        }
    }

    /// Asks `/system` again (the 2 s refresh; an image that couldn't reach the server): not while
    /// an answer is still out (a hung server was sent one every refresh), and the last answer, or
    /// why there is none, stays on screen until the new one lands (Overview's hardware strip
    /// showed a skeleton at every refresh).
    pub(super) fn refresh_system(&mut self, cx: &mut Context<Self>) {
        if self.system_out {
            return;
        }
        if self.store.system.needs_fetch() {
            self.store.system = Load::Loading;
        }
        self.fetch_system(cx);
    }

    fn fetch_system(&mut self, cx: &mut Context<Self>) {
        self.system_out = true;
        self.system_seq += 1;
        let seq = self.system_seq;
        self.fetch(
            cx,
            |api| api.system(),
            move |this, r, cx| {
                // Asked again since (Retry): this answer is older than what is on its way.
                if this.system_seq != seq {
                    return;
                }
                this.system_out = false;
                if let Ok(s) = &r
                    && let Some(g) = s.system.gpus.first().and_then(|g| g.util)
                {
                    this.store.gpu_util.push_back(g);
                }
                let was_down = this.store.unreachable();
                this.store.note_reachability(r.as_ref().err());
                if was_down && !this.store.unreachable() {
                    this.reconnected(cx);
                } else if this.store.unreachable() {
                    // Every time, not only when it goes: a recovery that failed tries again later.
                    this.connection_lost(cx);
                }
                // Back after being unreachable: check its health now, not in 2 s.
                if this.store.health.needs_fetch() {
                    this.ensure_core(cx);
                }
                if let Ok(s) = &r {
                    // It answers: a later loss starts over, and its data dir says whether it is ours.
                    this.recovered();
                    this.note_data_dir(s.data_dir.as_deref());
                    this.store.last_gpu_count = Some(s.system.gpus.len());
                }
                let r = r.map(|s| {
                    this.store.capabilities = s.capabilities;
                    this.store.data_dir = s.data_dir;
                    this.store.features = s.features;
                    s.system
                });
                this.store.system = r.into();
                this.probe_missing_model_archs(cx);
            },
        );
    }

    pub fn ensure_run_data(&mut self, id: &str, cx: &mut Context<Self>) {
        let spe = self.store.run(id).map_or(1, |r| r.steps_per_epoch);
        if self.store.run_data.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.run_data.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "metrics");
            let rid = id.to_string();
            let key = id.to_string();
            self.fetch(
                cx,
                move |api| api.metrics(&rid),
                move |this, r, _| {
                    // A reload started since: this older snapshot must not overwrite it.
                    if this.history_tickets.get(&(key.clone(), "metrics")) != Some(&ticket) {
                        return;
                    }
                    this.store.install_run_data(&key, r.map(|m| RunData::new(m, spe)));
                },
            );
        }
    }

    /// A run's history dropped (the run deleted, here or elsewhere): an answer still on its way
    /// (metrics, samples or logs asked for before) must not bring it back.
    pub(super) fn forget_history(&mut self, id: &str) {
        self.history_tickets.retain(|(run, _), _| run != id);
        self.store.samples_stale.remove(id);
        self.store.samples_refreshing.remove(id);
        self.store.samples_refresh_error.remove(id);
    }

    /// A fresh ticket for one history request of a run; earlier tickets stop counting.
    pub(super) fn history_ticket(&mut self, id: &str, what: &'static str) -> u64 {
        self.history_seq += 1;
        self.history_tickets.insert((id.to_string(), what), self.history_seq);
        self.history_seq
    }

    pub fn ensure_samples(&mut self, id: &str, cx: &mut Context<Self>) {
        // Recover if an in-flight initial fetch failed after a newer Epoch/Status marked it stale.
        let stale_failed = self.sample_epoch.is_none()
            && matches!(self.store.samples.get(id), Some(Load::Failed(_)))
            && self.store.samples_stale.remove(id);
        let needs = stale_failed || self.store.samples.get(id).is_none_or(|l| l.needs_fetch());
        let refresh_ready = !needs
            && self.sample_epoch.is_none()
            && matches!(self.store.samples.get(id), Some(Load::Ready(_)))
            && self.store.samples_stale.remove(id);
        if needs || refresh_ready {
            if needs {
                self.store.samples_stale.remove(id);
                self.store.samples_refreshing.remove(id);
                self.store.samples_refresh_error.remove(id);
                self.store.samples.insert(id.to_string(), Load::Loading);
            } else {
                self.store.samples_refresh_error.remove(id);
                self.store.samples_refreshing.insert(id.to_string());
            }
            let ticket = self.history_ticket(id, "samples");
            let (rid, key, epoch) = (id.to_string(), id.to_string(), self.sample_epoch);
            self.fetch(
                cx,
                move |api| api.samples(&rid, epoch),
                move |this, r, _| {
                    if this.history_tickets.get(&(key.clone(), "samples")) != Some(&ticket) {
                        return;
                    }
                    this.store.samples_refreshing.remove(&key);
                    match r {
                        Ok(v) => {
                            this.store.samples_refresh_error.remove(&key);
                            this.store.samples.insert(key, Load::Ready(v));
                        }
                        Err(e) if refresh_ready => {
                            // If a newer Epoch or terminal Status arrived while this refresh was in
                            // flight, `samples_stale` is already queued for an automatic retry.
                            if !this.store.samples_stale.contains(&key) {
                                // Keep existing tiles mounted so Logs don't jump, and record the error
                                // so the caption offers an explicit in-place Retry for the newer epoch.
                                this.store.samples_refresh_error.insert(key, e);
                            }
                        }
                        Err(e) => {
                            this.store.samples_refresh_error.remove(&key);
                            this.store.samples.insert(key, Load::Failed(e));
                        }
                    }
                },
            );
        }
    }

    /// Fetch a run's validation samples again after a failed load (or failed background refresh).
    pub fn retry_samples(&mut self, id: &str, cx: &mut Context<Self>) {
        self.store.samples_refresh_error.remove(id);
        if self.sample_epoch.is_none() && matches!(self.store.samples.get(id), Some(Load::Ready(_))) {
            self.store.samples_stale.insert(id.to_string());
        } else {
            self.store.samples.remove(id);
            self.store.samples_stale.remove(id);
        }
        self.ensure_samples(id, cx);
        cx.notify();
    }

    pub fn ensure_logs(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.logs.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.logs.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "logs");
            let (rid, key) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.logs(&rid),
                move |this, r, _| {
                    if this.history_tickets.get(&(key.clone(), "logs")) != Some(&ticket) {
                        return;
                    }
                    this.store.install_logs(&key, r);
                    // Only the log on screen, and only while it follows (a late answer for another
                    // run mustn't move, or resume, the one being read).
                    if matches!(&this.route, Route::Run(id) if *id == key) && this.log_follow {
                        this.log_scroll.scroll_to_bottom();
                    }
                },
            );
        }
    }

    /// Fetch a run's logs again after a failed load.
    pub fn retry_logs(&mut self, id: &str, cx: &mut Context<Self>) {
        self.store.logs.remove(id);
        self.ensure_logs(id, cx);
        cx.notify();
    }

    pub fn ensure_eval(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.evals.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.evals.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.model_eval(&a),
                move |this, r, _| {
                    this.store.evals.insert(b, r.into());
                },
            );
        }
    }

    /// Reload the dataset, model and run lists in place (no loading state): added or removed
    /// elsewhere (Studio, the CLI, another data dir behind the same server), they show here too, and a
    /// run's "(removed)" goes once its dataset is listed again. A list that failed is fetched again, its
    /// error kept on screen meanwhile (`retry_list`).
    pub fn refresh_lists(&mut self, cx: &mut Context<Self>) {
        if matches!(self.store.datasets, Load::Failed(_)) {
            self.retry_list(List::Datasets, cx);
        } else if matches!(self.store.datasets, Load::Ready(_)) {
            self.fetch_list(List::Datasets, cx);
        }
        if matches!(self.store.models, Load::Failed(_)) {
            self.retry_list(List::Models, cx);
        } else if matches!(self.store.models, Load::Ready(_)) {
            self.fetch_list(List::Models, cx);
        }
        // A failed run list too: nothing else asks for it again (Runs and Overview said "Could not
        // reach folo serve" for good when the app was opened before its server).
        if matches!(self.store.runs, Load::Failed(_)) {
            self.retry_list(List::Runs, cx);
        } else if matches!(self.store.runs, Load::Ready(_)) {
            self.refresh_runs(cx);
        }
        self.ensure_core(cx);
    }

    /// Periodic 5 s background poll of the lists: shares in-flight deduplication with window-
    /// activation refreshes so a slow background fetch is never duplicated while still pending.
    pub(super) fn poll_lists(&mut self, cx: &mut Context<Self>) {
        if matches!(self.store.datasets, Load::Failed(_)) {
            self.retry_list(List::Datasets, cx);
        } else if matches!(self.store.datasets, Load::Ready(_)) {
            self.refresh_list_if_idle(List::Datasets, cx);
        }
        if matches!(self.store.models, Load::Failed(_)) {
            self.retry_list(List::Models, cx);
        } else if matches!(self.store.models, Load::Ready(_)) {
            self.refresh_list_if_idle(List::Models, cx);
        }
        if matches!(self.store.runs, Load::Failed(_)) {
            self.retry_list(List::Runs, cx);
        } else if matches!(self.store.runs, Load::Ready(_)) {
            self.refresh_list_if_idle(List::Runs, cx);
        }
        self.ensure_core(cx);
    }

    /// A run's stream said it ended (completed, failed or stopped). The event carries only the
    /// status: its model, end time and final numbers are in the run list, read again now (not at
    /// the next 5 s refresh, which left a finished run with only Delete and a duration still
    /// counting), and a completed run's models too.
    pub(super) fn run_ended(&mut self, id: &str, status: RunStatus, cx: &mut Context<Self>) {
        let name = self.store.run(id).map_or(id.to_string(), |r| r.name.clone());
        match status {
            RunStatus::Completed => {
                self.toast(
                    Tone::Success,
                    format!("{name} finished"),
                    Some("Best weights saved to Models.".into()),
                    cx,
                );
                // Read again in place: the list (and which run made the Playground's model) stays
                // while it is out.
                if self.store.models.ready().is_some() {
                    self.fetch_list(List::Models, cx);
                } else {
                    self.ensure_core(cx);
                }
            }
            RunStatus::Failed => self.toast(Tone::Danger, format!("{name} failed"), None, cx),
            _ => {}
        }
        if matches!(self.store.runs, Load::Ready(_)) {
            // A list asked for before the end can still show the run active (stopping, running): the
            // end is newer, so those answers are dropped, as after a change here.
            self.list_tickets.changed(List::Runs);
            self.refresh_runs(cx);
        }
        // The stream sends at most ~10 steps a second (docs/API.md), so what a page watched live
        // holds is thinned: its loss line and smoothed loss were not the file's. Final now, the
        // whole history is read again (in place: the charts stay while it is out), as Studio does.
        // Still loading too: that request may have been answered before the end (a fresh ticket
        // drops it).
        if matches!(self.store.run_data.get(id), Some(Load::Ready(_) | Load::Loading)) {
            self.reload_run_history(id, cx);
        }
    }

    /// A run's metric history read again over the one shown (or still loading), which stays until
    /// the answer is in (and if it fails).
    pub(super) fn reload_run_history(&mut self, id: &str, cx: &mut Context<Self>) {
        let spe = self.store.run(id).map_or(1, |r| r.steps_per_epoch);
        let ticket = self.history_ticket(id, "metrics");
        let (rid, key) = (id.to_string(), id.to_string());
        self.fetch(
            cx,
            move |api| api.metrics(&rid),
            move |this, r, cx| {
                if this.history_tickets.get(&(key.clone(), "metrics")) != Some(&ticket) {
                    return;
                }
                // Dropped meanwhile (the run deleted): not brought back. A failure keeps a history
                // shown, and ends one still loading (its page says so and can ask again).
                // An ended run's final history that failed over a partial one shown is read again
                // at the next run-list refresh (nothing else would: its stream is gone).
                match (r, this.store.run_data.get(&key)) {
                    (Ok(m), Some(Load::Ready(_) | Load::Loading)) => {
                        this.store.install_run_data(&key, Ok(RunData::new(m, spe)));
                        cx.notify();
                    }
                    (Err(e), Some(Load::Loading)) => {
                        this.store.install_run_data(&key, Err(e));
                        cx.notify();
                    }
                    (Err(_), Some(Load::Ready(_)))
                        if this.store.run(&key).is_some_and(|r| !r.status.is_active()) =>
                    {
                        this.history_retry.insert(key);
                    }
                    _ => {}
                }
            },
        );
    }

    /// Reload the run list in place (no loading state) and subscribe to whatever is active.
    pub fn refresh_runs(&mut self, cx: &mut Context<Self>) {
        self.fetch_list(List::Runs, cx);
    }
}
