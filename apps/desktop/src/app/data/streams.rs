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

    /// Apply what the run streams sent since the last tick; whether anything changed.
    pub(crate) fn drain_streams(&mut self, cx: &mut Context<Self>) -> bool {
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
}
