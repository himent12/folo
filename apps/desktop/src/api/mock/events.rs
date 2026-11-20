//! A run's live event stream: a local run's steps, epochs and log, and a cloud run's phases.

use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use crate::api::time;
use crate::api::*;

use super::training::{Curve, epoch_line};
use super::{MockBackend, STEP_DT};

impl MockBackend {
    /// [`Backend::events`]'s answer.
    pub(super) fn events_for(&self, id: &str) -> ApiResult<Subscription> {
        self.check()?;
        let (tx, rx) = channel();
        let (sub, cancel) = Subscription::new(rx);
        let (live, curve, spe, epochs, snap, log_mark, cloud) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (
                f.live,
                f.curve.clone(),
                f.run.steps_per_epoch,
                f.run.epochs,
                Self::snapshot(f),
                f.logs.len(),
                f.cloud.is_some(),
            )
        };
        let _ = tx.send(RunEvent::Status(StatusEvent::new(snap.status, None).with_length(epochs, spe)));
        if cloud && !self.frozen {
            self.stream_cloud(id, snap, curve, log_mark, tx, cancel);
            return Ok(sub);
        }
        let Some((base, at)) = live else { return Ok(sub) };
        let total = epochs as u64 * spe;
        let start_secs = self.sim_now();
        let runs = self.runs.clone();
        let run_id = id.to_string();
        // The stream starts where the history a client just fetched ends: capture it now, not
        // whenever the worker thread happens to start.
        let mut sent = (base + (at.elapsed().as_secs_f64() / STEP_DT) as u64).min(total);
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                // One lock for the whole decision, so a stop can't land between "is it stopped?"
                // and "how far has it got?".
                let (target, stop) = {
                    let runs = runs.lock().unwrap();
                    match runs.iter().find(|f| f.run.id == run_id) {
                        None => (sent, Some(Vec::new())),
                        Some(f) if f.live.is_none() => (
                            f.step.min(total),
                            Some(f.logs.get(log_mark..).map(<[LogLine]>::to_vec).unwrap_or_default()),
                        ),
                        Some(f) => (Self::current_step(f), None),
                    }
                };
                // Stopped through the API: flush the steps up to the stop point and the stop's log
                // lines, then report the terminal status and end the stream.
                if let Some(lines) = stop {
                    let stop_step = target;
                    while sent < stop_step {
                        sent += 1;
                        let _ = tx.send(RunEvent::Step(curve.step(sent)));
                        if sent.is_multiple_of(spe) {
                            let _ = tx.send(RunEvent::Epoch(curve.epoch((sent / spe) as u32)));
                        }
                    }
                    for l in lines {
                        let _ = tx.send(RunEvent::Log(l));
                    }
                    let _ = tx.send(RunEvent::Status(
                        StatusEvent::new(RunStatus::Stopped, None).with_length(epochs, spe),
                    ));
                    return;
                }
                let now = target;
                while sent < now {
                    sent += 1;
                    if tx.send(RunEvent::Step(curve.step(sent))).is_err() {
                        return;
                    }
                    if sent.is_multiple_of(spe) {
                        let e = (sent / spe) as u32;
                        let ep = curve.epoch(e);
                        let t = time::format(start_secs + at.elapsed().as_secs() as i64);
                        let msg = epoch_line(e, epochs, &ep);
                        if tx.send(RunEvent::Epoch(ep)).is_err()
                            || tx
                                .send(RunEvent::Log(LogLine { t, level: LogLevel::Info, message: msg }))
                                .is_err()
                        {
                            return;
                        }
                    }
                }
                if sent >= total {
                    let _ = tx.send(RunEvent::Status(
                        StatusEvent::new(RunStatus::Completed, None).with_length(epochs, spe),
                    ));
                    return;
                }
            }
        });
        Ok(sub)
    }

    /// A cloud run's live events, from `first` (the snapshot the stream started at) on: the steps
    /// and epochs it trains, a `status` when its status changes, and its `cloud` state on a new
    /// phase at once and otherwise at most once a second (transfer progress, cost). Ends once the
    /// run is released and finished.
    fn stream_cloud(
        &self,
        id: &str,
        first: Run,
        curve: Curve,
        log_mark: usize,
        tx: std::sync::mpsc::Sender<RunEvent>,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) {
        let runs = self.runs.clone();
        let run_id = id.to_string();
        let clock = self.clock;
        std::thread::spawn(move || {
            let (spe, epochs) = (first.steps_per_epoch, first.epochs);
            let (mut sent, mut status, mut logs) = (first.step, first.status, log_mark);
            let mut cloud = first.cloud;
            let mut cloud_at = Instant::now();
            loop {
                std::thread::sleep(Duration::from_millis(100));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    return;
                }
                let (now, lines) = {
                    let mut runs = runs.lock().unwrap();
                    let Some(f) = runs.iter_mut().find(|f| f.run.id == run_id) else { return };
                    super::training::log_cloud_end(f);
                    (Self::snapshot(f), f.logs.get(logs..).map(<[LogLine]>::to_vec).unwrap_or_default())
                };
                logs += lines.len();
                while sent < now.step {
                    sent += 1;
                    let _ = tx.send(RunEvent::Step(curve.step(sent)));
                    if sent.is_multiple_of(spe) {
                        let e = (sent / spe) as u32;
                        let ep = curve.epoch(e);
                        let line = LogLine {
                            t: time::format(clock.now() as i64),
                            level: LogLevel::Info,
                            message: epoch_line(e, epochs, &ep),
                        };
                        let _ = tx.send(RunEvent::Epoch(ep));
                        let _ = tx.send(RunEvent::Log(line));
                    }
                }
                for l in lines {
                    let _ = tx.send(RunEvent::Log(l));
                }
                let phase_changed = now.cloud.as_ref().map(|c| c.phase) != cloud.as_ref().map(|c| c.phase);
                if now.cloud != cloud && (phase_changed || cloud_at.elapsed() >= Duration::from_secs(1)) {
                    cloud.clone_from(&now.cloud);
                    cloud_at = Instant::now();
                    if let Some(c) = cloud.clone()
                        && tx.send(RunEvent::Cloud(Box::new(c))).is_err()
                    {
                        return;
                    }
                }
                if now.status != status {
                    status = now.status;
                    let ev = StatusEvent::new(status, now.error.clone())
                        .with_length(epochs, spe)
                        .with_error_code(now.error_code.clone());
                    let _ = tx.send(RunEvent::Status(ev));
                }
                if !status.is_active() && cloud.as_ref().is_some_and(|c| c.phase == CloudPhase::Released) {
                    return;
                }
            }
        });
    }
}
