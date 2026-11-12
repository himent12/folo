//! Loading from the backend: the background fetch helper, what each page needs, run event
//! subscriptions and the live event pump.

use std::sync::Arc;
use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;
use crate::ui::*;

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
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async move { call(&*api) }).await;
            this.update(cx, |this, cx| {
                // Answered by a source we've since switched away from: not ours any more.
                if this.backend_gen != generation {
                    return;
                }
                apply(this, r, cx);
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
            self.fetch(
                cx,
                |api| api.runs(),
                |this, r, cx| {
                    this.store.runs = r.map(newest_first_runs).into();
                    this.store.sync_run_list();
                    this.subscribe_active(cx);
                },
            );
        }
        if self.store.datasets.needs_fetch() {
            self.store.datasets = Load::Loading;
            self.fetch(cx, |api| api.datasets(), |this, r, _| this.store.datasets = r.into());
        }
        if self.store.models.needs_fetch() {
            self.store.models = Load::Loading;
            self.fetch(
                cx,
                |api| api.models(),
                |this, r, cx| {
                    this.store.models = r
                        .map(|mut m| {
                            m.sort_by(|a, b| b.created.cmp(&a.created));
                            m
                        })
                        .into();
                    // Playground starts on the best model (not the newest checkpoint, which may
                    // have learned nothing yet), and so does it again after its model was
                    // deleted: an image already on stage is predicted with it.
                    if this.play_model.is_none() {
                        this.play_model = this.store.best_model(None).map(|m| m.id.clone());
                        this.run_predict(cx);
                    }
                },
            );
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
                    }
                    // Back after being unreachable: check its health now, not in 2 s.
                    if this.store.health.needs_fetch() {
                        this.ensure_core(cx);
                    }
                    let r = r.map(|s| {
                        this.store.capabilities = s.capabilities;
                        this.store.features = s.features;
                        s.system
                    });
                    this.store.system = r.into();
                },
            );
        }
        if self.store.health.needs_fetch() {
            self.store.health = Load::Loading;
            self.fetch(cx, |api| api.health(), |this, r, _| this.store.health = r.into());
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
            let (a, b) = (id.to_string(), id.to_string());
            self.fetch(
                cx,
                move |api| {
                    api.images(&a, &ImageQuery { limit: 100_000, ..Default::default() }).map(|p| p.images)
                },
                move |this, r, _| {
                    this.store.images.insert(b, r.into());
                    // New list, new truth: nothing derived from the old one may survive.
                    *this.ds_filter_cache.borrow_mut() = None;
                    *this.ds_split_counts.borrow_mut() = None;
                },
            );
        }
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
        cx.spawn(async move |this, cx| {
            let img = cx
                .background_executor()
                .spawn(async move { call(&*api).ok().and_then(to_render_image) })
                .await;
            this.update(cx, |this, cx| {
                if this.backend_gen != generation {
                    if let Some(img) = img {
                        cx.drop_image(img, None);
                    }
                    return;
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
                let Ok(sub) = sub else { return };
                this.subscriptions.insert(id.clone(), sub);
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

    /// Reload the run list in place (no loading state) and subscribe to whatever is active.
    pub fn refresh_runs(&mut self, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            |api| api.runs(),
            |this, r, cx| {
                if let Ok(runs) = r {
                    this.store.runs = Load::Ready(newest_first_runs(runs));
                    this.store.sync_run_list();
                    this.subscribe_active(cx);
                }
            },
        );
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
                        let mut changed = false;
                        let mut ended = Vec::new();
                        let mut finished = Vec::new();
                        let ids: Vec<String> = this.subscriptions.keys().cloned().collect();
                        for id in ids {
                            loop {
                                let ev = match this.subscriptions.get(&id).map(|rx| rx.try_recv()) {
                                    Some(Ok(ev)) => ev,
                                    Some(Err(TryRecvError::Empty)) | None => break,
                                    Some(Err(TryRecvError::Disconnected)) => {
                                        ended.push(id.clone());
                                        break;
                                    }
                                };
                                if let RunEvent::Status(s) = &ev
                                    && matches!(s.status, RunStatus::Completed | RunStatus::Failed)
                                {
                                    finished.push((id.clone(), s.status));
                                }
                                this.store.apply(&id, ev);
                                changed = true;
                            }
                        }
                        // A stream that ended without a terminal status dropped out: the events in the
                        // gap are gone (the API replays only the status), so reload that run's history
                        // and the run list, which resubscribes if it's still training.
                        let dropped: Vec<String> = ended
                            .iter()
                            .filter(|id| !finished.iter().any(|(f, _)| f == *id))
                            .filter(|id| this.store.run(id).is_some_and(|r| r.status.is_active()))
                            .cloned()
                            .collect();
                        for id in ended {
                            this.subscriptions.remove(&id);
                        }
                        for id in dropped {
                            this.resubscribe(id, cx);
                        }
                        for (id, status) in finished {
                            let name = this.store.run(&id).map(|r| r.name.clone()).unwrap_or(id.clone());
                            if status == RunStatus::Completed {
                                this.toast(
                                    Tone::Success,
                                    format!("{name} finished"),
                                    Some("Best weights saved to Models.".into()),
                                    cx,
                                );
                                this.store.models = Load::Idle;
                            } else {
                                this.toast(Tone::Danger, format!("{name} failed"), None, cx);
                            }
                        }
                        if this.bench_pump(tick, cx) {
                            changed = true;
                        }
                        if tick.is_multiple_of(20) {
                            this.store.system = Load::Idle;
                            this.ensure_core(cx);
                        }
                        if tick.is_multiple_of(50) && matches!(this.store.runs, Load::Ready(_)) {
                            this.refresh_runs(cx);
                        }
                        // A rented instance costs money: its row stays current while it lives.
                        if tick.is_multiple_of(50) && this.instance_alive() {
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

/// Runs newest first (the server already sends them so; the mock and refreshes may not).
fn newest_first_runs(mut runs: Vec<Run>) -> Vec<Run> {
    runs.sort_by(|a, b| b.created.cmp(&a.created));
    runs
}
