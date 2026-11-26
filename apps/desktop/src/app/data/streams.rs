//! Live run event subscriptions, stream draining, lost-stream recovery, and the periodic pump.

use std::sync::mpsc::TryRecvError;
use std::time::Duration;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;

use super::super::{FoloApp, Route};

impl FoloApp {
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
    pub(crate) fn resubscribe(&mut self, id: String, cx: &mut Context<Self>) {
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

    /// Apply what the run streams sent since the last tick; whether anything visibly changed.
    ///
    /// Live step events arrive ~9–10×/s, so on screens that do not plot per-step metrics (e.g.
    /// Settings, Datasets, Models, Benchmarks, Playground, Runs) we apply every event to the store
    /// without forcing a full UI rebuild unless the displayed epoch, percentage, progress bar width
    /// (in device pixels), or an open stop dialog / queued ETA actually changed (#308).
    pub(crate) fn drain_streams(&mut self, cx: &mut Context<Self>) -> bool {
        let mut changed = false;
        let mut ended = Vec::new();
        let mut finished = Vec::new();
        let ids: Vec<String> = self.subscriptions.keys().cloned().collect();
        for id in ids {
            let before = self.step_visible_sig(&id);
            let mut saw_step = false;
            loop {
                let ev = match self.subscriptions.get(&id).map(|rx| rx.try_recv()) {
                    Some(Ok(ev)) => ev,
                    Some(Err(TryRecvError::Empty)) | None => break,
                    Some(Err(TryRecvError::Disconnected)) => {
                        ended.push(id.clone());
                        break;
                    }
                };
                let is_step = matches!(&ev, RunEvent::Step(_));
                if let RunEvent::Status(s) = &ev
                    && !s.status.is_active()
                {
                    finished.push((id.clone(), s.status));
                }
                self.store.apply(&id, ev);
                if is_step {
                    saw_step = true;
                } else {
                    changed = true;
                }
            }
            if saw_step && (self.step_needs_every_tick(&id) || self.step_visible_sig(&id) != before) {
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
        if !picked {
            self.store.samples_refresh_error.remove(id);
            match self.store.samples.get(id) {
                Some(Load::Ready(_)) => {
                    self.store.samples_stale.insert(id.to_string());
                    // An answer still on its way predates the gap.
                    self.history_tickets.remove(&(id.to_string(), "samples"));
                }
                Some(Load::Loading | Load::Failed(_)) => {
                    self.store.samples.insert(id.to_string(), Load::Idle);
                    self.store.samples_stale.remove(id);
                    // An answer still on its way predates the gap.
                    self.history_tickets.remove(&(id.to_string(), "samples"));
                }
                _ => {}
            }
        }
    }

    /// Runs whose stream dropped mid-run and could not be opened again. Nothing subscribes to a run
    /// that has ended, so once the run list says one did, its end is handled here (before, its page
    /// kept the history of the moment the stream dropped for good). An ended run's final history
    /// that failed to load is asked for again here too, once per list refresh, until it is in.
    pub(crate) fn settle_lost_streams(&mut self, cx: &mut Context<Self>) {
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
    pub(crate) fn start_pump(&mut self, cx: &mut Context<Self>) {
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
                            this.refresh_system(cx);
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

    /// Whether the currently visible screen plots per-step live data for `run_id` (its own run
    /// dashboard, or Overview's live hero card).
    fn step_needs_every_tick(&self, run_id: &str) -> bool {
        match &self.route {
            Route::Run(open) => open == run_id,
            Route::Overview => self.store.live_run().is_some_and(|r| r.id == run_id),
            _ => false,
        }
    }

    /// A compact signature of everything visibly rendered for `run_id` outside per-step views:
    /// epoch/length numbers, rounded percentage (`{:.0}%`), sidebar and table progress-bar
    /// device-pixel widths tracked independently (so a compact 24 px sidebar rail and a 56 px table
    /// rail never mask each other's pixel boundaries), any displayed ETA for a queued run waiting
    /// behind `run_id`, and the stop-dialog epoch boundary.
    fn step_visible_sig(&self, run_id: &str) -> Option<StepVisibleSig> {
        let r = self.store.run(run_id)?;
        let is_sidebar_live = self.store.live_run().is_some_and(|live| live.id == run_id);
        let sidebar_w = if is_sidebar_live {
            if self.narrow || self.sidebar_collapsed {
                24.0
            } else {
                // `SIDEBAR` (220 px) minus `live_mini`'s horizontal margins (2×8) and padding (2×10).
                crate::theme::size::SIDEBAR - 36.0
            }
        } else {
            0.0
        };
        // Both `Route::Runs` and `Route::Overview`'s Recent runs table render a 56 px progress rail.
        let table_w = if matches!(self.route, Route::Runs | Route::Overview) { 56.0 } else { 0.0 };
        let scale = self.last_scale_factor.max(1.0);
        let progress = r.progress().clamp(0.0, 1.0);
        let sidebar_bar_px = if sidebar_w > 0.0 { (progress * sidebar_w * scale).round() as u32 } else { 0 };
        let table_bar_px = if table_w > 0.0 { (progress * table_w * scale).round() as u32 } else { 0 };
        let pct = if is_sidebar_live { (progress * 100.0).round_ties_even() as u32 } else { 0 };
        // A queued run's detail page displays the formatted ETA of the run currently training ahead of it.
        let ahead_eta = if let Route::Run(open) = &self.route
            && open != run_id
            && let Some(runs) = self.store.runs.ready()
            && self.store.run(open).is_some_and(|o| o.status == RunStatus::Queued)
            && crate::api::training_now(runs, open).is_some_and(|a| a.id == run_id)
        {
            r.eta_s.map(crate::api::time::duration)
        } else {
            None
        };
        // `stop_dialog` switches copy once the current epoch's final batch is done (`kept_epoch`).
        let stop_epoch_done = self.confirm_stop.as_deref() == Some(run_id)
            && r.epoch > 0
            && r.steps_per_epoch > 0
            && r.step >= r.epoch as u64 * r.steps_per_epoch;
        Some(StepVisibleSig {
            epoch: r.epoch,
            epochs: r.epochs,
            spe: r.steps_per_epoch,
            pct,
            sidebar_bar_px,
            table_bar_px,
            ahead_eta,
            stop_epoch_done,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct StepVisibleSig {
    epoch: u32,
    epochs: u32,
    spe: u64,
    pct: u32,
    sidebar_bar_px: u32,
    table_bar_px: u32,
    ahead_eta: Option<String>,
    stop_epoch_done: bool,
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;

    use crate::api::{Loss, RunEvent, Step, Subscription};
    use crate::app::Route;
    use crate::app::tests::{drive, open};

    fn make_step(step: u64, spe: u64, epochs: u32) -> RunEvent {
        RunEvent::Step(Step {
            epoch: ((step.max(1) - 1) / spe + 1) as u32,
            step,
            lr: 0.01,
            loss: Loss { box_: 1.0, cls: 0.8, dfl: 0.5, total: 2.3 },
            imgs_per_s: 120.0,
            eta_s: ((epochs as u64 * spe).saturating_sub(step)) as f64 * 0.12,
            steps_per_epoch: spe,
            epochs,
        })
    }

    /// #308 item 7: on Settings (or Datasets/Models/Playground/Runs), live steps that move the
    /// sidebar progress bar by less than a device pixel still update the store without forcing a
    /// UI rebuild on every 10 Hz tick; once enough steps accumulate to move the bar or change the
    /// percentage/epoch, `drain_streams` notifies. On the live run's own page and Overview, every
    /// step still notifies so charts and live throughput stay smooth.
    #[test]
    fn idle_pages_skip_noop_step_rebuilds_while_live_views_notify_every_step() {
        let (mut app, window) = open();
        let (tx, rx) = channel();
        let (sub, _cancel) = Subscription::new(rx);
        let (live_id, start_step, spe, epochs) = drive(&mut app, window, |this, w, cx| {
            let live = this.store.live_run().expect("mock has a live run").clone();
            this.subscriptions.insert(live.id.clone(), sub);
            this.last_scale_factor = 1.5;
            this.navigate(Route::Settings, w, cx);
            (live.id, live.step, live.steps_per_epoch, live.epochs)
        });
        assert_eq!((spe, epochs), (100, 100), "mock live run has 10,000 steps");

        // Stream 40 consecutive steps (one per pump tick): all 40 update the store, but only the
        // 1–2 ticks that cross a device-pixel boundary on the 184×1.5 px sidebar bar trigger redraws.
        let mut settings_notifies = 0usize;
        for delta in 1..=40 {
            tx.send(make_step(start_step + delta, spe, epochs)).unwrap();
            let notified = drive(&mut app, window, |this, _, cx| this.drain_streams(cx));
            if notified {
                settings_notifies += 1;
            }
        }
        assert_eq!(
            drive(&mut app, window, |this, _, _| this.store.run(&live_id).unwrap().step),
            start_step + 40,
            "every step still lands in the store"
        );
        assert!(
            (1..=3).contains(&settings_notifies),
            "expected 1–3 visible bar updates over 40 steps on Settings, got {settings_notifies}"
        );

        // On the live run's dashboard and Overview, every step still notifies immediately.
        for route in [Route::Run(live_id.clone()), Route::Overview] {
            drive(&mut app, window, |this, w, cx| this.navigate(route.clone(), w, cx));
            for i in 1..=5 {
                let next = drive(&mut app, window, |this, _, _| this.store.run(&live_id).unwrap().step + 1);
                tx.send(make_step(next, spe, epochs)).unwrap();
                let notified = drive(&mut app, window, |this, _, cx| this.drain_streams(cx));
                assert!(notified, "{route:?} tick {i} must notify on every step");
            }
        }

        // On a short run (50 steps total), every single step moves the bar >1 device pixel and notifies.
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Settings, w, cx);
            let r = this.store.runs.ready_mut().unwrap().iter_mut().find(|r| r.id == live_id).unwrap();
            r.epochs = 5;
            r.steps_per_epoch = 10;
            r.step = 10;
            r.epoch = 1;
        });
        for s in 11..=15 {
            tx.send(make_step(s, 10, 5)).unwrap();
            let notified = drive(&mut app, window, |this, _, cx| this.drain_streams(cx));
            assert!(notified, "short run step {s} moves the bar visibly and must notify");
        }

        // Compact sidebar (24 px) on `Route::Runs` (56 px table rail) at 1x scale: 2.0% -> 2.2%
        // crosses the 24 px bar's rounded pixel boundary (0.48 -> 0.528) while the 56 px rail and
        // rounded % stay unchanged; tracking both widths independently catches it.
        drive(&mut app, window, |this, w, cx| {
            this.last_scale_factor = 1.0;
            this.sidebar_collapsed = true;
            this.navigate(Route::Runs, w, cx);
            let r = this.store.runs.ready_mut().unwrap().iter_mut().find(|r| r.id == live_id).unwrap();
            r.epochs = 10;
            r.steps_per_epoch = 100;
            r.step = 20;
            r.epoch = 1;
        });
        tx.send(make_step(22, 100, 10)).unwrap();
        assert!(
            drive(&mut app, window, |this, _, cx| this.drain_streams(cx)),
            "compact 24 px sidebar pixel crossing on Runs must notify even when 56 px table rail does not"
        );

        // On Overview, a non-hero running job in the Recent runs table still notifies when its 56 px
        // progress rail crosses a device pixel.
        let (tx2, rx2) = channel();
        let (sub2, _cancel2) = Subscription::new(rx2);
        let other_id = drive(&mut app, window, |this, w, cx| {
            let other = this
                .store
                .runs
                .ready_mut()
                .unwrap()
                .iter_mut()
                .find(|r| r.id != live_id)
                .expect("mock has a second run");
            other.status = crate::api::RunStatus::Running;
            other.epochs = 10;
            other.steps_per_epoch = 100;
            other.step = 20;
            other.epoch = 1;
            let id = other.id.clone();
            this.subscriptions.insert(id.clone(), sub2);
            this.navigate(Route::Overview, w, cx);
            id
        });
        tx2.send(make_step(30, 100, 10)).unwrap();
        assert!(
            drive(&mut app, window, |this, _, cx| this.drain_streams(cx)),
            "non-hero running job on Overview must notify when its Recent runs 56 px rail advances"
        );
        assert_eq!(drive(&mut app, window, |this, _, _| this.store.run(&other_id).unwrap().step), 30);
    }
}
