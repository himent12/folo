//! Mirroring the worker's run into the local run folder (history, samples, status, checkpoint copies) and
//! collecting its weights at the end.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use folo_api::{
    CloudPhase, LogLevel, Run, RunEvent, RunLogs, RunMetrics, RunSamples, RunStatus, StatusEvent,
    TransferProgress,
};
use folo_client::transfer::{self, Control, TransferError};
use folo_client::{Client, encode};
use folo_cloud::InstanceState;

use crate::cloud::state::Outcome;
use crate::cloud::worker;
use crate::runs::run_model_id;
use crate::util::{rfc3339, unix_now};

use super::{
    COPY_RETRY_EVERY, Copying, Exec, Halt, LOG_PAGE, PROGRESS_EVERY, SLOW_STEP, failed, staged_copy,
};

impl Exec {
    /// One poll of the worker: its run's status, the steps and epochs after the cursors (each new epoch with its
    /// samples), the new log lines, and the checkpoint when its copy is due. Every step goes to `metrics.jsonl`;
    /// subscribers get every epoch and the newest step. The worker's run once it ended.
    pub(super) fn mirror(&mut self, client: &Client, run_id: &str) -> folo_client::Result<Option<Run>> {
        let id = encode(run_id);
        let mut started = Instant::now();
        let local = self.entry.id().to_string();
        let mut step = |what: &str| {
            let took = std::mem::replace(&mut started, Instant::now()).elapsed();
            if took >= SLOW_STEP {
                tracing::info!(run = %local, "mirroring the worker: {what} took {:.1} s", took.as_secs_f64());
            }
        };
        let run: Run = client.get(&format!("/runs/{id}"))?;
        step("its run");
        self.follow_status(&run);
        // Always with the cursors (0 before the first step): only then are the steps gap-free.
        let (after_step, after_epoch) = (self.private.after_step, self.private.after_epoch);
        let path = format!("/runs/{id}/metrics?after_step={after_step}&after_epoch={after_epoch}");
        let history: RunMetrics = client.get(&path)?;
        step("its new steps and epochs");
        let newest = history.steps.last().map(|s| s.step);
        let mut epoch_done = false;
        for ev in worker::in_order(history) {
            if let RunEvent::Epoch(e) = &ev {
                self.copy_samples(client, &id, e.epoch)?;
            }
            crate::runs::append_line(
                &self.entry.metrics_path(),
                &folo_vision::train::event_record(&ev).to_string(),
            );
            match &ev {
                RunEvent::Step(s) => {
                    self.private.after_step = s.step;
                    if Some(s.step) == newest {
                        self.entry.on_event(&ev);
                    }
                }
                RunEvent::Epoch(e) => {
                    self.private.after_epoch = e.epoch;
                    self.entry.on_event(&ev);
                    epoch_done = true;
                }
                _ => {}
            }
        }
        step("the new epochs' samples");
        loop {
            let logs: RunLogs =
                client.get(&format!("/runs/{id}/logs?after={}&limit={LOG_PAGE}", self.private.after_log))?;
            let n = logs.lines.len();
            for line in logs.lines {
                self.entry.on_event(&RunEvent::Log(line));
            }
            self.private.after_log += n as u64;
            if n < LOG_PAGE {
                break;
            }
        }
        self.save();
        step("its new log lines");
        self.reap_copy();
        let every = self.core.tuning.checkpoint_every;
        let due = self.private.synced.is_none_or(|t| unix_now() - t >= every.as_secs_f64())
            && self.copy_failed.is_none_or(|t| t.elapsed() >= COPY_RETRY_EVERY.min(every));
        if epoch_done && run.status.is_active() && due && self.copying.is_none() {
            self.start_copy(client, run_id);
        }
        Ok((!run.status.is_active()).then_some(run))
    }

    /// Starts a copy of the worker's `weights/last.folo` on a thread of its own ([`Copying`]): it fetches the file
    /// into a staging file, and [`Exec::reap_copy`] takes its result. It ends early when the process shuts down
    /// or a stop is asked for (the collection at the run's end copies the final file).
    fn start_copy(&mut self, client: &Client, run_id: &str) {
        let weights = self.dir.join("weights");
        let staged = staged_copy(&weights);
        let path = format!("/models/{}/weights", encode(&run_model_id(run_id, true)));
        let cancel = Arc::new(AtomicBool::new(false));
        let (client, core, entry) = (client.clone(), self.core.clone(), self.entry.clone());
        let (cancelled, dest) = (cancel.clone(), staged.clone());
        let copy = move || {
            std::fs::create_dir_all(&weights).map_err(TransferError::Io)?;
            let stop = || cancelled.load(Ordering::SeqCst) || core.halt.is_on() || entry.stop_requested();
            let mut ctl = Control { progress: &mut |_| {}, stop: &stop, backoff: transfer::FIRST_BACKOFF };
            let fetched = transfer::download(&client, &path, &dest, &mut ctl).map(drop);
            // Nobody takes the file any more (see `Copying`'s drop for the order).
            if cancelled.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&dest);
            }
            fetched
        };
        let name = format!("folo-cloud-copy-{}", self.entry.id());
        match std::thread::Builder::new().name(name).spawn(copy) {
            Ok(thread) => self.copying = Some(Copying { thread: Some(thread), cancel, staged }),
            Err(e) => {
                self.copy_failed = Some(Instant::now());
                tracing::warn!(run = %self.entry.id(), "could not start the checkpoint copy's thread: {e}");
            }
        }
    }

    /// Takes the result of the copy under way once it has ended: the staged file becomes `weights/last.folo`, with
    /// `Run.cloud.synced` and a log line, or a warning (and a later retry) says why not. A worker without the file
    /// yet, and a copy that was told to end, say nothing.
    pub(super) fn reap_copy(&mut self) {
        if !self.copying.as_ref().is_some_and(|c| c.thread.as_ref().is_none_or(|t| t.is_finished())) {
            return;
        }
        let Some(mut copying) = self.copying.take() else { return };
        let fetched = match copying.thread.take().map(|t| t.join()) {
            Some(Ok(fetched)) => fetched,
            Some(Err(_)) => {
                self.copy_failed = Some(Instant::now());
                tracing::error!(run = %self.entry.id(), "the checkpoint copy's thread panicked");
                return;
            }
            None => return,
        };
        // This thread, which holds the run, is the only one that writes `weights/last.folo`.
        let placed = fetched.and_then(|()| {
            std::fs::rename(&copying.staged, self.dir.join("weights").join("last.folo"))
                .map_err(TransferError::Io)
        });
        match placed {
            Ok(()) => {
                self.copy_failed = None;
                self.synced_now();
            }
            Err(TransferError::Stopped) => {}
            Err(TransferError::Failed(e)) if e.code() == "model_not_found" => {}
            Err(e) => {
                self.copy_failed = Some(Instant::now());
                self.entry.log(LogLevel::Warn, format!("copying weights/last.folo (retried later): {e}"));
            }
        }
    }

    /// `weights/last.folo` in the run folder is the worker's of now.
    fn synced_now(&mut self) {
        let now = unix_now();
        self.private.synced = Some(now);
        self.save();
        self.entry.update_cloud(|c| c.synced = Some(rfc3339(now)));
        self.entry.log(LogLevel::Info, "copied weights/last.folo");
    }

    /// The worker run's status, device and length, on the local run: `queued` while it waits in the worker's queue
    /// (logged when its place changes), then `running`.
    fn follow_status(&mut self, w: &Run) {
        if w.status == RunStatus::Queued {
            if w.queue_position != self.queued_at
                && let Some(p) = w.queue_position
            {
                self.entry
                    .log(LogLevel::Info, format!("waiting in the worker's training queue (position {p})"));
            }
            self.queued_at = w.queue_position;
            return;
        }
        let local = self.entry.run();
        let spe = if w.steps_per_epoch > 0 { w.steps_per_epoch } else { local.steps_per_epoch };
        if (local.device != w.device || local.steps_per_epoch != spe || local.resolved != w.resolved)
            && w.resolved.is_some()
        {
            self.entry.update(|r| {
                r.device = w.device.clone();
                r.steps_per_epoch = spe;
                r.resolved = w.resolved.clone();
            });
        }
        if local.status == RunStatus::Queued {
            let running = StatusEvent::new(RunStatus::Running, None).with_length(w.epochs, spe);
            self.entry.on_event(&RunEvent::Status(running));
        }
    }

    /// The worker's samples of `epoch`, as the local `samples/epoch_NNN.json`.
    fn copy_samples(&self, client: &Client, id: &str, epoch: u32) -> folo_client::Result<()> {
        let samples = match client.get::<RunSamples>(&format!("/runs/{id}/samples?epoch={epoch}")) {
            Ok(s) => s,
            Err(e) if e.code() == "epoch_not_available" => return Ok(()),
            Err(e) => return Err(e),
        };
        let file = worker::samples_file(&samples, &self.private.classes, self.entry.val_offset());
        let path = self.dir.join("samples").join(format!("epoch_{epoch:03}.json"));
        if let Err(e) = crate::index::write_atomic(&path, file.to_string().as_bytes()) {
            tracing::warn!("writing {}: {e}", path.display());
        }
        Ok(())
    }

    /// Copies the worker's `last.folo` (or `best.folo`) into the run folder at the run's end, with
    /// `Run.cloud.progress` following it; false when the worker has none.
    fn copy(
        &mut self,
        client: &Client,
        run_id: &str,
        last: bool,
    ) -> std::result::Result<bool, TransferError> {
        let file = if last { "last.folo" } else { "best.folo" };
        let weights = self.dir.join("weights");
        std::fs::create_dir_all(&weights).map_err(TransferError::Io)?;
        let path = format!("/models/{}/weights", encode(&run_model_id(run_id, last)));
        let entry = &self.entry;
        let mut shown = None::<Instant>;
        let mut progress = |p: TransferProgress| {
            if shown.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) || p.done_bytes >= p.total_bytes {
                entry.update_cloud(|c| c.progress = Some(p));
                shown = Some(Instant::now());
            }
        };
        let stop = || self.core.halt.is_on();
        let mut ctl = Control { progress: &mut progress, stop: &stop, backoff: transfer::FIRST_BACKOFF };
        match transfer::download(client, &path, &weights.join(file), &mut ctl) {
            Ok(_) => {}
            Err(TransferError::Failed(e)) if e.code() == "model_not_found" => return Ok(false),
            Err(e) => return Err(e),
        }
        if last {
            self.synced_now();
        } else {
            self.entry.log(LogLevel::Info, format!("copied weights/{file}"));
        }
        Ok(true)
    }

    /// The run ended on the worker (`end`; `None` when a stopped run did not end in time): copies both weights files
    /// and returns how it ended. A resumed run's worker writes `best.folo` only once an epoch beats the checkpoint's
    /// best, so the local `best.folo` stays when the worker has none.
    pub(super) fn collect(
        &mut self,
        client: &Client,
        run_id: &str,
        end: Option<Run>,
        time_up: bool,
    ) -> std::result::Result<Outcome, Halt> {
        // A copy still under way is told to end and not waited for: it only writes its staged file. One that
        // finished is kept, should the final copy fail.
        self.reap_copy();
        self.copying = None;
        self.phase(CloudPhase::Collecting, "the run ended on the worker; copying its weights");
        for last in [true, false] {
            match self.copy(client, run_id, last) {
                Ok(_) => {}
                Err(TransferError::Stopped) => return Err(Halt::LetGo),
                Err(e) => {
                    let file = if last { "last.folo" } else { "best.folo" };
                    // An instance its provider says is gone is lost, whichever request noticed it: the run resumes
                    // from the checkpoint copied before, as one lost while it trains does.
                    if matches!(self.instance_state(), Some(InstanceState::Gone | InstanceState::Stopped)) {
                        return Err(failed(
                            "cloud_instance_lost",
                            format!(
                                "copying weights/{file}: {e}; its instance is gone, so the weights copied so far \
                                 stay (resume the run to continue from them)"
                            ),
                        ));
                    }
                    return Err(failed("cloud_sync_failed", format!("copying weights/{file}: {e}")));
                }
            }
        }
        let outcome = match end {
            Some(r) => Outcome { status: r.status, error: r.error, error_code: r.error_code },
            None => Outcome::plain(RunStatus::Stopped),
        };
        // A run that completed (or failed) just as its time was up ended that way, not by the limit.
        if time_up && outcome.status == RunStatus::Stopped {
            self.entry.log(LogLevel::Info, format!("the time limit of {} ended the run", self.hours()));
        }
        Ok(outcome)
    }
}
