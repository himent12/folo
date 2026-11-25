//! What a run says: its log lines, the events between two snapshots and the ticker that sends them.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use folo_api::{CloudPhase, Epoch, LogLevel, LogLine, QueueEvent, RunEvent, RunStatus};

use crate::mock::cloud::PHASE_S;
use crate::util::{rfc3339, unix_now};

use super::{SimRun, Snap, TICK, Timeline};

impl SimRun {
    fn log(&self, (level, message): (LogLevel, String)) -> RunEvent {
        RunEvent::Log(LogLine { t: rfc3339(unix_now()), level, message })
    }

    fn msg_start(&self) -> (LogLevel, String) {
        let c = &self.config;
        let message = format!(
            "training {} on {}: {} epochs, batch {}, imgsz {}, {} steps/epoch",
            c.arch.as_str(),
            self.device,
            c.epochs,
            c.batch,
            c.imgsz,
            self.spe
        );
        (LogLevel::Info, message)
    }

    fn msg_stopping() -> (LogLevel, String) {
        (LogLevel::Warn, "stop requested; finishing the current step".into())
    }

    fn msg_epoch(&self, ep: &Epoch) -> Vec<(LogLevel, String)> {
        let e = ep.epoch;
        let l = &ep.train_loss;
        let mut out = vec![(
            LogLevel::Info,
            format!(
                "epoch {e}/{}  box {:.3}  cls {:.3}  dfl {:.3}  mAP50 {:.3}  mAP50-95 {:.3}",
                self.config.epochs, l.box_, l.cls, l.dfl, ep.val.map50, ep.val.map50_95
            ),
        )];
        if self.curve.best(e).is_some_and(|b| b.epoch == e) {
            out.push((LogLevel::Debug, format!("saved best.folo (mAP50-95 {:.3})", ep.val.map50_95)));
        }
        if e > 0 && e == self.config.epochs.saturating_sub(self.config.augment.close_mosaic) {
            out.push((LogLevel::Info, "closing mosaic augmentation for the final epochs".into()));
        }
        out
    }

    fn msg_end(&self, snap: &Snap) -> Option<(LogLevel, String)> {
        match snap.status {
            RunStatus::Completed => {
                let best = self.curve.best(self.epochs_done(snap));
                let message = format!(
                    "training complete: best epoch {} (mAP50-95 {:.3})",
                    best.map_or(0, |b| b.epoch),
                    best.map_or(0.0, |b| b.map50_95)
                );
                Some((LogLevel::Info, message))
            }
            RunStatus::Stopped => {
                Some((LogLevel::Info, format!("stopped at step {}; last checkpoint kept", snap.step)))
            }
            RunStatus::Failed => Some((
                LogLevel::Error,
                format!(
                    "training failed at step {}: {}",
                    snap.step,
                    self.error(snap.status).unwrap_or_default()
                ),
            )),
            _ => None,
        }
    }

    /// Log history of the current cycle, oldest first: the same lines the event stream sends, stamped with the
    /// simulated times they happened at.
    pub fn logs(&self) -> Vec<LogLine> {
        let snap = self.snap();
        let per_epoch = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(c) => c.lock().unwrap().epoch_s,
        };
        let line = |t: f64, (level, message): (LogLevel, String)| LogLine { t: rfc3339(t), level, message };
        let mut out = Vec::new();
        // Training lines only exist once it started; a stop (even while queued) and the end are logged regardless.
        if let Some(started) = snap.started {
            out.push(line(started, self.msg_start()));
            for e in 1..=self.epochs_done(&snap) {
                let t = started + e as f64 * per_epoch;
                let ep = self.curve.epoch(e, self.epoch_duration(e));
                out.extend(self.msg_epoch(&ep).into_iter().map(|m| line(t, m)));
            }
        }
        if let Some(t) = snap.stop_requested {
            out.push(line(t, Self::msg_stopping()));
        }
        if let (Some(m), Some(t)) = (self.msg_end(&snap), snap.finished) {
            out.push(line(t, m));
        }
        out.extend(self.cloud_lines(&snap).into_iter().map(|(t, m)| line(t, (LogLevel::Info, m))));
        // Queue moves, merged in by time (the sort is stable, so lines of the same second keep their order).
        out.extend(self.notes.lock().unwrap().iter().cloned());
        out.sort_by(|a, b| a.t.cmp(&b.t));
        out
    }

    /// A cloud run's phase lines up to `snap`, with the times their phases began: before the training at the fixed
    /// steps of [`PHASE_S`], then when it starts, ends (or is stopped) and the run is released.
    fn cloud_lines(&self, snap: &Snap) -> Vec<(f64, String)> {
        let (Some(c), Some(cloud)) = (&self.cloud, self.cloud_at(snap)) else { return Vec::new() };
        let epoch_s = match &self.timeline {
            Timeline::Static { .. } => self.curve.real_epoch_s(),
            Timeline::Live(clock) => clock.lock().unwrap().epoch_s(),
        };
        let trained = snap.started.map(|s| s + self.config.epochs as f64 * epoch_s);
        let releasing = match snap.stop_requested {
            Some(stop) => Some(stop),
            None => trained.map(|t| t + PHASE_S),
        };
        let until = snap.finished.unwrap_or(snap.at);
        [
            (CloudPhase::Provisioning, Some(snap.created)),
            (CloudPhase::Booting, Some(snap.created + PHASE_S)),
            (CloudPhase::Syncing, Some(snap.created + 2.0 * PHASE_S)),
            (CloudPhase::Training, snap.started),
            (CloudPhase::Collecting, trained.filter(|_| snap.stop_requested.is_none())),
            (CloudPhase::Releasing, releasing),
            (CloudPhase::Released, snap.finished),
        ]
        .into_iter()
        .filter_map(|(p, t)| {
            let t = t.filter(|&t| p <= cloud.phase && t <= until)?;
            Some((t, c.phase_line(p, cloud.cost_usd)?))
        })
        .collect()
    }

    /// Events describing the change from `prev` to `now`.
    fn diff(&self, prev: &Snap, now: &Snap) -> Vec<RunEvent> {
        let mut out = Vec::new();
        let status = |s: &Snap| RunEvent::Status(self.status_event(s.status));
        let mut prev = *prev;
        if now.cycle != prev.cycle {
            out.push(
                self.log((LogLevel::Info, "mock run finished its loop; restarting from epoch 1".into())),
            );
            prev = Snap { status: RunStatus::Queued, step: 0, ..*now };
            out.push(status(&prev));
        }
        let entering =
            now.status != prev.status && !matches!(now.status, RunStatus::Completed | RunStatus::Stopped);
        if now.queue != prev.queue
            && let Some((position, waiting)) = now.queue
        {
            out.push(RunEvent::Queue(QueueEvent { position, waiting }));
        }
        // A cloud run's hand-over: a line when its phase changes, its state whenever it changed (progress moves in
        // tenths, the cost in cents).
        if let (Some(c), Some(before), Some(after)) = (&self.cloud, self.cloud_at(&prev), self.cloud_at(now))
            && before != after
        {
            if before.phase != after.phase
                && let Some(m) = c.phase_line(after.phase, after.cost_usd)
            {
                out.push(self.log((LogLevel::Info, m)));
            }
            out.push(RunEvent::Cloud(Box::new(after)));
        }
        if entering {
            out.push(status(now));
            match now.status {
                RunStatus::Running => out.push(self.log(self.msg_start())),
                RunStatus::Stopping => out.push(self.log(Self::msg_stopping())),
                _ => {}
            }
        }
        if now.step > prev.step {
            out.push(RunEvent::Step(self.curve.step(now.step, now.eta_s.unwrap_or(0.0).round())));
        }
        let (done_prev, done_now) = (self.epochs_done(&prev), self.epochs_done(now));
        for e in done_prev + 1..=done_now {
            let ep = self.curve.epoch(e, self.epoch_duration(e));
            out.extend(self.msg_epoch(&ep).into_iter().map(|m| self.log(m)));
            out.push(RunEvent::Epoch(ep));
        }
        if now.status != prev.status && !entering {
            out.extend(self.msg_end(now).map(|m| self.log(m)));
            out.push(status(now));
        }
        out
    }
}

/// Publishes a live run's events while it has subscribers; exits when the last one disconnects.
pub(super) async fn ticker(run: Arc<SimRun>, mut last: Snap) {
    let mut interval = tokio::time::interval(TICK);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        interval.tick().await;
        // Hold `hub` across `snap`, `diff`, and `broadcast` (matching `subscribe`'s `hub -> clock` order) so
        // lag-resync snapshots and fresh tail subscriptions never observe events out of order.
        let mut hub = run.hub.lock().unwrap();
        let alive = !run.deleted.load(Ordering::SeqCst);
        if alive && hub.receiver_count() > 0 {
            let now = run.snap();
            let events = run.diff(&last, &now);
            last = now;
            for ev in events {
                hub.broadcast(ev);
            }
        } else {
            hub.ticking = false;
            return;
        }
    }
}
