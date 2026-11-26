//! One run: its record, its log, its events (the trainer's sink) and how its execution ends.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use folo_api::{
    BestEpoch, LogLevel, LogLine, ModelMetrics, Run, RunCloud, RunDataset, RunEvent, RunStatus, StatusEvent,
};
use tokio::sync::broadcast;

use crate::index::write_atomic;
use crate::store::SseFrame;
use crate::util::{parse_rfc3339, unix_now};

use super::{
    ALIVE_EVERY, DatasetRef, EVENT_BUFFER, RunEntry, RunFile, RunState, TAIL_STEPS, add_paused, append_line,
    now,
};

impl RunEntry {
    pub(super) fn new(id: String, dir: PathBuf, mut file: RunFile) -> RunEntry {
        let (tx, _) = broadcast::channel(EVENT_BUFFER);
        let (sse_tx, _) = broadcast::channel(EVENT_BUFFER);
        // The dataset it was created on names it in the API, also once that dataset is removed (older run.json files
        // have its name but no classes).
        let d = &file.dataset;
        file.run
            .dataset
            .get_or_insert_with(|| RunDataset { name: d.name.clone(), classes: d.classes.clone() });
        RunEntry {
            id,
            dir,
            state: Mutex::new(RunState {
                file,
                tail: Vec::new(),
                attempt: 0,
                claimed: false,
                outcome: None,
                deleted: false,
                interrupted: false,
                alive_written: None,
            }),
            metrics_cache: Mutex::default(),
            tx,
            sse_tx,
            stop: Arc::new(AtomicBool::new(false)),
            measure: Mutex::new(None),
        }
    }

    /// The API record.
    pub fn run(&self) -> Run {
        self.state.lock().unwrap().file.shown()
    }

    /// The dataset it trains on.
    pub fn dataset(&self) -> DatasetRef {
        self.state.lock().unwrap().file.dataset.clone()
    }

    /// The weights file the run started from (`TrainConfig.weights` resolved when it was created).
    pub fn weights(&self) -> Option<PathBuf> {
        self.state.lock().unwrap().file.weights.clone()
    }

    /// The run folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Image id offset of validation samples.
    pub fn val_offset(&self) -> u64 {
        self.state.lock().unwrap().file.val_offset
    }

    /// Validation metrics of the last validated epoch.
    pub fn last_metrics(&self) -> Option<ModelMetrics> {
        self.state.lock().unwrap().file.last_metrics
    }

    pub(super) fn persist(&self, st: &RunState) {
        let bytes = serde_json::to_vec_pretty(&st.file).expect("runs serialize");
        let path = self.dir.join("run.json");
        if let Err(e) = write_atomic(&path, &bytes) {
            tracing::warn!("writing {}: {e}", path.display());
        }
    }

    pub(super) fn send(&self, ev: RunEvent) {
        // Pre-serialize once for all HTTP SSE subscribers when at least one is listening.
        if self.sse_tx.receiver_count() > 0 {
            let _ = self.sse_tx.send(SseFrame::run(&ev));
        }
        // Typed subscribers (e.g. MCP / Rust callers).
        if self.tx.receiver_count() > 0 {
            let _ = self.tx.send(ev);
        }
    }

    fn write_log(&self, line: &LogLine) {
        append_line(&self.dir.join("log.jsonl"), &serde_json::to_string(line).expect("logs serialize"));
    }

    /// A server-side log line (persisted and sent like the trainer's).
    pub(crate) fn log(&self, level: LogLevel, message: impl Into<String>) {
        let line = LogLine { t: now(), level, message: message.into() };
        self.write_log(&line);
        self.send(RunEvent::Log(line));
    }

    /// Moves to a final status outside the trainer (stopped while queued, failed to start, server restart).
    pub(super) fn finish(&self, st: &mut RunState, status: RunStatus, error: Option<String>) {
        self.finish_with_code(st, status, error, None);
    }

    /// [`RunEntry::finish`] with the failure's stable error code (`Run.error_code`; [`INTERRUPTED`] for a run the
    /// server's stop ended).
    pub(super) fn finish_with_code(
        &self,
        st: &mut RunState,
        status: RunStatus,
        error: Option<String>,
        code: Option<String>,
    ) {
        let r = &mut st.file.run;
        r.status = status;
        r.error = error;
        r.error_code = code;
        r.eta_s = None;
        // An active run has no end yet, except one a previous process left training: it ended when it last trained
        // (`history::reconcile`).
        let finished = r.finished.take().unwrap_or_else(now);
        // A resumed run that ends before it trained again: its wait is not training either, and a next resume counts
        // from this end.
        if let Some(at) = st.file.resumed_at.take() {
            add_paused(r, parse_rfc3339(&finished).unwrap_or(at) - at);
        }
        r.finished = Some(finished);
        // A failed run names the epoch it failed in (the one its steps were in); any other the last one it finished.
        r.epoch =
            if status == RunStatus::Failed { r.epoch.max(st.file.epochs_done) } else { st.file.epochs_done };
        r.model_id = model_id_of(&self.id, &self.dir, status).or(r.model_id.take());
        self.persist(st);
        self.send(RunEvent::Status(status_event(&st.file.run)));
    }

    /// The trainer's events (or a cloud worker's, mirrored): update the run, persist what matters, forward to
    /// subscribers.
    pub(crate) fn on_event(&self, ev: &RunEvent) {
        if let Some(m) = self.measure.lock().unwrap().as_mut() {
            match ev {
                RunEvent::Step(s) => m.last_step = Some(*s),
                RunEvent::Epoch(e) => m.epoch(e),
                _ => {}
            }
        }
        let mut st = self.state.lock().unwrap();
        let mut line = None;
        match ev {
            RunEvent::Status(s) => {
                let file = &mut st.file;
                let r = &mut file.run;
                if s.status == RunStatus::Running {
                    if s.epochs > 0 {
                        r.epochs = s.epochs;
                    }
                    if r.started.is_none() {
                        r.started = Some(now());
                    } else if let Some(at) = file.resumed_at.take() {
                        // Training again after a resume: its wait in the queue was not training either.
                        add_paused(r, unix_now().floor() - at);
                    }
                    // A stop requested before training began: the trainer stops right away.
                    if r.status == RunStatus::Stopping {
                        return;
                    }
                    r.status = RunStatus::Running;
                    self.alive(&mut st, true);
                } else {
                    st.outcome = Some((s.status, s.error.clone(), s.error_code.clone()));
                    self.alive(&mut st, false);
                    return;
                }
            }
            RunEvent::Step(s) => {
                let r = &mut st.file.run;
                if s.steps_per_epoch > 0 {
                    r.steps_per_epoch = s.steps_per_epoch;
                }
                r.epoch = s.epoch;
                r.step = s.step;
                r.eta_s = Some(s.eta_s);
                if st.tail.len() < TAIL_STEPS {
                    st.tail.push(*s);
                }
                self.alive(&mut st, false);
            }
            RunEvent::Epoch(e) => {
                st.tail.clear();
                st.file.epochs_done = e.epoch;
                let v = &e.val;
                st.file.last_metrics = Some(model_metrics(v));
                let r = &mut st.file.run;
                r.epoch = e.epoch;
                if r.best.is_none_or(|b| e.val.map50_95 > b.map50_95) {
                    r.best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
                }
                // One log line per epoch, so the log shows progress (a cloud run's worker writes its own, mirrored).
                if r.cloud.is_none() {
                    line = Some(epoch_line(e, r.epochs));
                }
                self.alive(&mut st, true);
            }
            RunEvent::Log(l) => {
                self.write_log(l);
                self.alive(&mut st, false);
            }
            // The trainer never sends these: the queue does (`Runs::notify_queue`), and only cloud runs have
            // `cloud` events.
            RunEvent::Queue(_) | RunEvent::Cloud(_) => {}
        }
        // Sent under the state lock, so subscribers see transitions in the order they happen.
        self.send(ev.clone());
        drop(st);
        if let Some(line) = line {
            self.log(LogLevel::Info, line);
        }
    }

    /// The run reported now: where it ended if the process dies before it reports again (`history::reconcile`).
    /// Writes `run.json` when `persist`, or when an event last wrote it [`ALIVE_EVERY`] ago or more.
    fn alive(&self, st: &mut RunState, persist: bool) {
        st.file.alive_at = Some(unix_now().floor());
        if persist || st.alive_written.is_none_or(|t| t.elapsed() >= ALIVE_EVERY) {
            self.persist(st);
            st.alive_written = Some(Instant::now());
        }
    }

    /// Takes a queued run for execution: false when it was stopped or deleted while it waited.
    pub(super) fn claim(&self) -> bool {
        let mut st = self.state.lock().unwrap();
        if st.file.run.status != RunStatus::Queued || st.deleted {
            return false;
        }
        st.claimed = true;
        true
    }

    /// Takes an active run a previous process left to its cloud worker, so this process follows it again: false
    /// when it was deleted meanwhile.
    pub(crate) fn follow(&self) -> bool {
        let mut st = self.state.lock().unwrap();
        st.claimed = !st.deleted;
        st.claimed
    }

    /// Runs `execute` on a claimed run, then publishes the final status it reported ([`RunEntry::set_outcome`],
    /// or the trainer's last `status` event; a panic is a failed run, `internal`) and lets go of the run, so a client
    /// that sees the status may resume or delete it at once.
    pub(crate) fn execute(&self, execute: impl FnOnce() -> Ended) {
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(execute));
        *self.measure.lock().unwrap() = None;
        let mut st = self.state.lock().unwrap();
        let internal = || Some("internal".to_string());
        let (status, error, code) = match (ran, st.outcome.take()) {
            (Ok(Ended::LetGo), _) => {
                st.claimed = false;
                return;
            }
            (Ok(Ended::Finished), Some(o)) => o,
            (Ok(Ended::Finished), None) => {
                (RunStatus::Failed, Some("the trainer ended without a final status".into()), internal())
            }
            (Err(_), _) => {
                let msg = "the trainer panicked (a bug)".to_string();
                drop(st);
                self.log(LogLevel::Error, msg.clone());
                st = self.state.lock().unwrap();
                (RunStatus::Failed, Some(msg), internal())
            }
        };
        st.claimed = false;
        let code =
            if status == RunStatus::Stopped && st.interrupted { Some(INTERRUPTED.to_string()) } else { code };
        self.finish_with_code(&mut st, status, error, code);
    }

    /// Records how the run ended, for [`RunEntry::execute`] to publish.
    pub(crate) fn set_outcome(&self, status: RunStatus, error: Option<String>, code: Option<String>) {
        self.state.lock().unwrap().outcome = Some((status, error, code));
    }

    /// A stop was requested ([`Runs::stop`](crate::runs::Runs::stop)).
    pub(crate) fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Changes the run's record outside the trainer (a cloud run's device and length from its worker) and persists
    /// it.
    pub(crate) fn update(&self, change: impl FnOnce(&mut Run)) {
        let mut st = self.state.lock().unwrap();
        change(&mut st.file.run);
        self.persist(&st);
    }

    /// Changes a cloud run's [`RunCloud`], persists it and sends it as a `cloud` event. Nothing for a run without
    /// one.
    pub(crate) fn update_cloud(&self, change: impl FnOnce(&mut RunCloud)) {
        let mut st = self.state.lock().unwrap();
        let Some(cloud) = st.file.run.cloud.as_mut() else { return };
        change(cloud);
        let event = RunEvent::Cloud(Box::new(cloud.clone()));
        self.persist(&st);
        self.send(event);
    }

    /// The run's id.
    pub(crate) fn id(&self) -> &str {
        &self.id
    }

    /// The run folder's `metrics.jsonl`, which a cloud run writes itself from its worker's history.
    pub(crate) fn metrics_path(&self) -> PathBuf {
        self.dir.join("metrics.jsonl")
    }
}

/// `Run.error_code` of a run that ended `stopped` because the server stopped (or restarted) while it trained or
/// waited, not because a client stopped it.
pub(crate) const INTERRUPTED: &str = "interrupted";

/// How an execution of a claimed run ended ([`RunEntry::execute`]).
pub(crate) enum Ended {
    /// It recorded its final status: publish it.
    Finished,
    /// This process let go of the run without ending it (a cloud run left to its worker when the server shuts
    /// down): it stays as persisted, for the next process to follow.
    LetGo,
}

/// A run's log line for a finished epoch: "epoch 3/5: loss 3.214, mAP50-95 0.073, 15.2 s".
pub(super) fn epoch_line(e: &folo_api::Epoch, epochs: u32) -> String {
    format!(
        "epoch {}/{epochs}: loss {:.3}, mAP50-95 {:.3}, {:.1} s",
        e.epoch, e.train_loss.total, e.val.map50_95, e.duration_s
    )
}

/// The model metrics of an epoch's validation.
pub(super) fn model_metrics(v: &folo_api::ValMetrics) -> ModelMetrics {
    ModelMetrics {
        map50: v.map50,
        map50_95: v.map50_95,
        precision: v.precision,
        recall: v.recall,
        best_f1_conf: v.best_f1_conf,
    }
}

/// The `status` event of a run's current state, with its length (`steps_per_epoch` is 0 until the trainer is built).
pub(super) fn status_event(r: &Run) -> StatusEvent {
    StatusEvent::new(r.status, r.error.clone())
        .with_length(r.epochs, r.steps_per_epoch)
        .with_error_code(r.error_code.clone())
}

/// The model id of a run's `best.folo` (or of `last.folo` when there is no best yet), for a run that ended
/// `completed` or `stopped` with weights.
fn model_id_of(run_id: &str, dir: &Path, status: RunStatus) -> Option<String> {
    if !matches!(status, RunStatus::Completed | RunStatus::Stopped) {
        return None;
    }
    let w = dir.join("weights");
    if w.join("best.folo").is_file() {
        Some(run_model_id(run_id, false))
    } else if w.join("last.folo").is_file() {
        Some(run_model_id(run_id, true))
    } else {
        None
    }
}

/// `m_<run suffix>` for a run's best weights, `m_<run suffix>-last` for its last.
pub fn run_model_id(run_id: &str, last: bool) -> String {
    let suffix = run_id.strip_prefix("run_").unwrap_or(run_id);
    if last { format!("m_{suffix}-last") } else { format!("m_{suffix}") }
}
