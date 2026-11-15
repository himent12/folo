//! Loading from the backend: the background fetch helper, what each page needs, run event
//! subscriptions and the live event pump.

use std::sync::Arc;
use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::lists::List;
use super::{FoloApp, Route};

/// Image records asked for at a time when a dataset page loads its list (ten of the server's
/// pages): the grid paints after the first.
pub(crate) const IMAGE_PAGE: u32 = 10_000;
/// How long a dataset's image list waits before asking again for a page that failed after the first.
const IMAGE_PAGE_RETRY: Duration = Duration::from_secs(3);

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
                if late && !this.store.unreachable() {
                    this.forget_unreachable_loads();
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Fetch the model list again after it failed (nothing else retries a failed load).
    pub fn retry_models(&mut self, cx: &mut Context<Self>) {
        self.store.models = Load::Idle;
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
        if self.store.system.needs_fetch() {
            self.store.system = Load::Loading;
            self.fetch(
                cx,
                |api| api.system(),
                |this, r, cx| {
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
                },
            );
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
    }

    /// A fresh ticket for one history request of a run; earlier tickets stop counting.
    pub(super) fn history_ticket(&mut self, id: &str, what: &'static str) -> u64 {
        self.history_seq += 1;
        self.history_tickets.insert((id.to_string(), what), self.history_seq);
        self.history_seq
    }

    pub fn ensure_samples(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.samples.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.samples.insert(id.to_string(), Load::Loading);
            let ticket = self.history_ticket(id, "samples");
            let (rid, key, epoch) = (id.to_string(), id.to_string(), self.sample_epoch);
            self.fetch(
                cx,
                move |api| api.samples(&rid, epoch),
                move |this, r, _| {
                    if this.history_tickets.get(&(key.clone(), "samples")) != Some(&ticket) {
                        return;
                    }
                    this.store.samples.insert(key, r.into());
                },
            );
        }
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

    /// A dataset's issue counts, once (the Broken labels tile splits its count by severity).
    pub fn ensure_issues(&mut self, id: &str, cx: &mut Context<Self>) {
        if self.store.issues.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.issues.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset_issues(&a).map(|p| p.counts),
                move |this, r, _| {
                    this.store.issues.insert(b, r.into());
                },
            );
        }
    }

    pub fn ensure_dataset(&mut self, id: &str, cx: &mut Context<Self>) {
        // "Not found" once, listed now (its folder registered again, here or elsewhere): ask again,
        // its images too.
        let back = matches!(self.store.dataset.get(id), Some(Load::Failed(e)) if e.code == "dataset_not_found")
            && self.store.dataset_summary(id).is_some();
        if back {
            self.store.dataset.remove(id);
            if self.store.images.get(id).is_some_and(|l| l.error().is_some()) {
                self.store.images.remove(id);
            }
        }
        if self.store.dataset.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.dataset.insert(id.to_string(), Load::Loading);
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| api.dataset(&a),
                move |this, r, _| {
                    this.store.dataset.insert(b, r.into());
                },
            );
        }
        if self.store.images.get(id).is_none_or(|l| l.needs_fetch()) {
            self.store.images.insert(id.to_string(), Load::Loading);
            self.images_seq += 1;
            self.images_more.insert(id.to_string(), self.images_seq);
            self.load_images_from(id.to_string(), self.images_seq, 0, cx);
        }
    }

    /// Every image of a dataset (the grid filters by split and class here, so a list cut short
    /// would lose the end of train and all of val: COCO has 123k), a page of
    /// [`IMAGE_PAGE`] at a time: the grid shows the first page while the rest arrives.
    /// `generation` is this load's: a page of a load since replaced (the list refetched or
    /// dropped) is not appended, and ends that load.
    fn load_images_from(&mut self, id: String, generation: u64, offset: u32, cx: &mut Context<Self>) {
        let a = id.clone();
        self.fetch(
            cx,
            move |api| api.images(&a, &ImageQuery { offset, limit: IMAGE_PAGE, ..Default::default() }),
            move |this, r, cx| {
                if this.images_more.get(&id) != Some(&generation) {
                    return;
                }
                let page = match r {
                    Ok(p) => p,
                    // A failed first page is the list's error.
                    Err(e) if offset == 0 => {
                        this.images_more.remove(&id);
                        this.store.images.insert(id, Load::Failed(e));
                        return;
                    }
                    // A later one keeps what came and asks again in a moment (the server
                    // restarting, a dropped connection), still "Loading…", never a partial list
                    // passed off as the whole; a reload or a dropped list ends it.
                    Err(_) => {
                        let backend = this.backend_gen;
                        cx.spawn(async move |this, cx| {
                            cx.background_executor().timer(IMAGE_PAGE_RETRY).await;
                            this.update(cx, |this, cx| {
                                // Only the same load, on the same server, of a list still held.
                                if this.images_more.get(&id) != Some(&generation) {
                                    return;
                                }
                                let held = this.store.images.get(&id).is_some_and(|l| l.ready().is_some());
                                if this.backend_gen == backend && held {
                                    this.load_images_from(id, generation, offset, cx);
                                } else {
                                    this.images_more.remove(&id);
                                }
                            })
                            .ok();
                        })
                        .detach();
                        return;
                    }
                };
                let list = match this.store.images.get_mut(&id) {
                    Some(Load::Ready(l)) if l.len() == offset as usize => l,
                    Some(Load::Loading) if offset == 0 => {
                        this.store.images.insert(id.clone(), Load::Ready(Vec::new()));
                        this.store.images.get_mut(&id).and_then(|l| l.ready_mut()).expect("just set")
                    }
                    _ => {
                        this.images_more.remove(&id);
                        return;
                    }
                };
                let got = page.images.len() as u32;
                list.extend(page.images);
                let next = offset + got;
                // New list, new truth: nothing derived from the old one may survive.
                *this.ds_filter_cache.borrow_mut() = None;
                *this.ds_split_counts.borrow_mut() = None;
                if got > 0 && u64::from(next) < page.total {
                    this.load_images_from(id, generation, next, cx);
                } else {
                    this.images_more.remove(&id);
                }
                cx.notify();
            },
        );
    }

    /// Whether more of a dataset's image list is still on its way (the grid shows what came).
    pub fn images_pending(&self, id: &str) -> bool {
        self.images_more.contains_key(id)
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

    /// A decoded image, requesting it if needed.
    pub fn image(
        &mut self,
        dataset: &str,
        image: &str,
        kind: ImgKind,
        cx: &mut Context<Self>,
    ) -> Option<Arc<folo_ui::RenderImage>> {
        let key = ImgKey { dataset: dataset.to_string(), image: image.to_string(), kind };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => return None,
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let thumb = (kind == ImgKind::Thumb).then_some(320);
        let (ds, im) = (key.dataset.clone(), key.image.clone());
        self.load_cached_image(key, move |api| api.image(&ds, &im, thumb), cx);
        None
    }

    /// Fetches and decodes an image off the UI thread into the slot `key` (already marked loading),
    /// dropping it if the data source changed meanwhile.
    pub fn load_cached_image(
        &mut self,
        key: ImgKey,
        call: impl FnOnce(&dyn Backend) -> ApiResult<ImageData> + Send + 'static,
        cx: &mut Context<Self>,
    ) {
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        // Full-size images are bounded too: a 12000 px photo would otherwise hold 400 MB.
        let max_side = (key.kind == ImgKind::Full).then_some(ImageCache::FULL_MAX_SIDE);
        cx.spawn(async move |this, cx| {
            let (img, err) = cx
                .background_executor()
                .spawn(async move {
                    match call(&*api) {
                        Ok(d) => (to_render_image_within(d, max_side), None),
                        Err(e) => (None, Some(e)),
                    }
                })
                .await;
            this.update(cx, |this, cx| {
                if this.backend_gen != generation {
                    if let Some(img) = img {
                        cx.drop_image(img, None);
                    }
                    return;
                }
                // An image that couldn't reach the server marks it down, so its return is noticed
                // (`reconnected` asks for the failed images again) even when only images saw the outage.
                if let Some(e) = err.filter(|e| e.code == "server_unreachable") {
                    this.store.note_reachability(Some(&e));
                }
                let evicted = this.store.images_px.finish(key, img);
                for e in evicted {
                    cx.drop_image(e, None);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Subscribe to live events of every active run.
    pub fn subscribe_active(&mut self, cx: &mut Context<Self>) {
        if self.frozen {
            return;
        }
        let ids: Vec<String> = self
            .store
            .runs
            .ready()
            .map(|runs| runs.iter().filter(|r| r.status.is_active()).map(|r| r.id.clone()).collect())
            .unwrap_or_default();
        for id in ids {
            if !self.subscriptions.contains_key(&id) {
                self.resubscribe(id, cx);
            }
        }
    }

    /// Open a run's live stream (off the UI thread), then reload whatever history of it is already
    /// loaded: events from the stream buffer while the history loads and merge into it, so nothing
    /// between an earlier snapshot (or a dropped stream) and this stream is lost. A failed attempt
    /// is retried by the periodic run-list refresh.
    pub(super) fn resubscribe(&mut self, id: String, cx: &mut Context<Self>) {
        if !self.subscribing.insert(id.clone()) {
            return;
        }
        let api = self.store.api.clone();
        let generation = self.backend_gen;
        let attempt = self.sub_attempt.get(&id).copied().unwrap_or(0);
        cx.spawn(async move |this, cx| {
            let rid = id.clone();
            let sub = cx.background_executor().spawn(async move { api.events(&rid) }).await;
            this.update(cx, |this, cx| {
                // Superseded (a new source, or the run was resumed): the newer attempt owns it.
                if this.backend_gen != generation
                    || this.sub_attempt.get(&id).copied().unwrap_or(0) != attempt
                {
                    return;
                }
                this.subscribing.remove(&id);
                // A failed attempt after a drop stays lost: the run list asks again while the run
                // trains, and handles its end if it ends meanwhile (`settle_lost_streams`).
                let Ok(sub) = sub else { return };
                let was_lost = this.stream_lost.remove(&id);
                this.subscriptions.insert(id.clone(), sub);
                if was_lost {
                    this.invalidate_latest_samples(&id);
                }
                let had_data = this.store.run_data.remove(&id).is_some();
                let had_logs = this.store.logs.remove(&id).is_some();
                if had_data {
                    this.ensure_run_data(&id, cx);
                }
                if had_logs {
                    this.ensure_logs(&id, cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Reload the dataset, model and run lists in place (no loading state): added or removed
    /// elsewhere (Studio, the CLI, another data dir behind the same server), they show here too, and a
    /// run's "(removed)" goes once its dataset is listed again. A list that failed is fetched again from
    /// scratch.
    pub fn refresh_lists(&mut self, cx: &mut Context<Self>) {
        if matches!(self.store.datasets, Load::Failed(_)) {
            self.store.datasets = Load::Idle;
        } else if matches!(self.store.datasets, Load::Ready(_)) {
            self.fetch_list(List::Datasets, cx);
        }
        if matches!(self.store.models, Load::Failed(_)) {
            self.store.models = Load::Idle;
        } else if matches!(self.store.models, Load::Ready(_)) {
            self.fetch_list(List::Models, cx);
        }
        // A failed run list too: nothing else asks for it again (Runs and Overview said "Could not
        // reach folo serve" for good when the app was opened before its server).
        if matches!(self.store.runs, Load::Failed(_)) {
            self.store.runs = Load::Idle;
        } else if matches!(self.store.runs, Load::Ready(_)) {
            self.refresh_runs(cx);
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
    fn reload_run_history(&mut self, id: &str, cx: &mut Context<Self>) {
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

    /// Apply what the run streams sent since the last tick; whether anything changed.
    pub(super) fn drain_streams(&mut self, cx: &mut Context<Self>) -> bool {
        let mut changed = false;
        let mut ended = Vec::new();
        let mut finished = Vec::new();
        let ids: Vec<String> = self.subscriptions.keys().cloned().collect();
        for id in ids {
            loop {
                let ev = match self.subscriptions.get(&id).map(|rx| rx.try_recv()) {
                    Some(Ok(ev)) => ev,
                    Some(Err(TryRecvError::Empty)) | None => break,
                    Some(Err(TryRecvError::Disconnected)) => {
                        ended.push(id.clone());
                        break;
                    }
                };
                if let RunEvent::Status(s) = &ev
                    && !s.status.is_active()
                {
                    finished.push((id.clone(), s.status));
                }
                self.store.apply(&id, ev);
                changed = true;
            }
        }
        // A stream that ended without a terminal status dropped out: the events in the gap are gone
        // (the API replays only the status), so reload that run's history and the run list, which
        // resubscribes if it's still training. One whose run the list already says has ended (it went
        // silent, and timed out after the run finished) never delivered that end: it is handled now
        // (the final history, the models), as a delivered end would be.
        let mut dropped = Vec::new();
        let mut silent = Vec::new();
        for id in ended.iter().filter(|id| !finished.iter().any(|(f, _)| f == *id)) {
            match self.store.run(id).map(|r| r.status) {
                Some(s) if s.is_active() => dropped.push(id.clone()),
                Some(s) => silent.push((id.clone(), s)),
                None => {}
            }
        }
        // A delivered end is the stream's last word (its reader stops): it goes now, so its
        // disconnect, seen on a later tick, is not taken for a silent end and handled twice.
        for id in ended.iter().chain(finished.iter().map(|(id, _)| id)) {
            self.subscriptions.remove(id);
        }
        for id in dropped {
            self.stream_lost.insert(id.clone());
            self.resubscribe(id, cx);
        }
        for (id, status) in finished {
            self.run_ended(&id, status, cx);
        }
        for (id, status) in silent {
            self.end_unseen(&id, status, cx);
        }
        changed
    }

    /// A run's end its stream never delivered (the stream went silent, or dropped and could not be
    /// opened again, while the run ended): handled as a delivered end, and what the stream would have
    /// added to a loaded log and to the latest predictions meanwhile is read again too.
    fn end_unseen(&mut self, id: &str, status: RunStatus, cx: &mut Context<Self>) {
        self.stream_lost.remove(id);
        self.run_ended(id, status, cx);
        if matches!(self.store.logs.get(id), Some(Load::Ready(_) | Load::Loading)) {
            self.store.logs.remove(id);
            self.ensure_logs(id, cx);
        }
        self.invalidate_latest_samples(id);
    }

    /// A run's latest predictions after its stream missed events (each epoch it missed made new
    /// ones): read again when shown. An earlier epoch picked on its page stays.
    fn invalidate_latest_samples(&mut self, id: &str) {
        let picked = matches!(&self.route, Route::Run(r) if r == id) && self.sample_epoch.is_some();
        if !picked && matches!(self.store.samples.get(id), Some(Load::Ready(_) | Load::Loading)) {
            self.store.samples.insert(id.to_string(), Load::Idle);
            // An answer still on its way predates the gap.
            self.history_tickets.remove(&(id.to_string(), "samples"));
        }
    }

    /// Runs whose stream dropped mid-run and could not be opened again. Nothing subscribes to a run
    /// that has ended, so once the run list says one did, its end is handled here (before, its page
    /// kept the history of the moment the stream dropped for good). An ended run's final history
    /// that failed to load is asked for again here too, once per list refresh, until it is in.
    pub(super) fn settle_lost_streams(&mut self, cx: &mut Context<Self>) {
        for id in std::mem::take(&mut self.history_retry) {
            if matches!(self.store.run_data.get(&id), Some(Load::Ready(_))) {
                self.reload_run_history(&id, cx);
            }
        }
        let lost: Vec<String> = self
            .stream_lost
            .iter()
            .filter(|id| !self.subscribing.contains(*id) && !self.subscriptions.contains_key(*id))
            .cloned()
            .collect();
        for id in lost {
            match self.store.run(&id).map(|r| r.status) {
                Some(s) if s.is_active() => {}
                Some(s) => self.end_unseen(&id, s, cx),
                None => {
                    self.stream_lost.remove(&id);
                }
            }
        }
    }

    /// Drain live events ~10× a second and refresh slow-moving data every few seconds.
    pub(super) fn start_pump(&mut self, cx: &mut Context<Self>) {
        let task = cx.spawn(async move |this, cx| {
            let mut tick: u64 = 0;
            loop {
                cx.background_executor().timer(Duration::from_millis(100)).await;
                tick += 1;
                let alive = this
                    .update(cx, |this, cx| {
                        let mut changed = this.drain_streams(cx);
                        if this.bench_pump(tick, cx) {
                            changed = true;
                        }
                        if tick.is_multiple_of(20) {
                            this.store.system = Load::Idle;
                            this.ensure_core(cx);
                        }
                        if tick.is_multiple_of(50) {
                            this.refresh_lists(cx);
                        }
                        // A rented instance costs money: its row stays current while it lives.
                        // So is one a cloud run has that the list hasn't got yet (a run started
                        // elsewhere after it was read), whose termination may later fail.
                        if tick.is_multiple_of(50) && (this.instance_alive() || this.instances_missing()) {
                            this.refresh_instances(cx);
                        }
                        // A ledger that couldn't be read is asked again every 30 s: a released
                        // run's page can't say whether its instance is gone until it is.
                        if tick.is_multiple_of(300) && this.cloud.instances.error().is_some() {
                            this.refresh_instances(cx);
                        }
                        if changed {
                            cx.notify();
                        }
                    })
                    .is_ok();
                if !alive {
                    break;
                }
            }
        });
        self._tasks.push(task);
    }
}
