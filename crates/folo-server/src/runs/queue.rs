//! Creating runs, the FIFO queue and its worker thread, stops and shutdown.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    BestEpoch, CloudPhase, LogLevel, QueueEvent, Run, RunCloud, RunEvent, RunStatus, TrainConfig,
};

use crate::error::{Error, Result};
use crate::store::QueueMove;
use crate::util::{fnv1a, unique_id, unix_now};

use super::entry::{Ended, INTERRUPTED, status_event};
use super::train::execute;
use super::{DatasetRef, Queue, QueuedRun, RunEntry, RunFile, Runs, STOP_REQUESTED, now};

/// A validated checkpoint snapshot staged for a new continued run ([`Runs::create`], [`Runs::create_cloud`]).
pub struct CheckpointSnapshot<'a> {
    /// Private staged copy of the checkpoint file, validated before queueing so an active trainer overwriting
    /// `origin` cannot swap a different snapshot in underneath registration.
    pub staged: &'a Path,
    /// Original model path logged for the user.
    pub origin: &'a Path,
    /// Scheduled epoch count read from the checkpoint (overrides `Run.epochs` immediately while queued).
    pub epochs: Option<u32>,
    /// Best epoch already recorded in the checkpoint (`Run.best` starts here so a tying later epoch does not
    /// overwrite the checkpoint's best epoch).
    pub best: Option<BestEpoch>,
}

impl Runs {
    /// Sends every waiting run its place in the queue (the SSE `queue` event). Called with the queue locked, after
    /// every change to it, so a run's events arrive in the order of the changes, and before the `running` status
    /// of the worker taking it.
    pub(super) fn notify_queue(q: &Queue) {
        let waiting = q.runs.len() as u32;
        for (i, item) in q.runs.iter().enumerate() {
            item.entry.send(RunEvent::Queue(QueueEvent { position: i as u32 + 1, waiting }));
        }
    }

    /// `POST /api/runs/{id}/queue`: moves a waiting run within the queue (`start` moves it to the front and wakes
    /// the worker, which takes it as soon as no run is training; the active run is never preempted). A run that is
    /// not waiting (the worker took it and is starting it, or it is running or finished) is `409 run_not_queued`.
    pub fn move_queued(&self, id: &str, mv: QueueMove) -> Result<Run> {
        let e = self.get(id)?;
        // Lock order: a run's state, then the queue (as `stop`), so the run cannot start, stop or be deleted
        // while it moves.
        let st = e.state.lock().unwrap();
        if st.deleted {
            return Err(Error::not_found("run", id));
        }
        let mut q = self.queue.lock().unwrap();
        let Some(from) = q.runs.iter().position(|x| x.entry.id == id) else {
            return Err(not_queued(id, st.file.run.status));
        };
        let n = q.runs.len();
        let to = mv.target(from, n);
        if to != from {
            let item = q.runs.remove(from).expect("position is in range");
            q.runs.insert(to, item);
            Self::notify_queue(&q);
        }
        let training = self.active.lock().unwrap().is_some();
        drop(q);
        let mut run = st.file.shown();
        run.queue_position = Some(to as u32 + 1);
        drop(st);
        if mv == QueueMove::Start {
            self.wake.notify_all();
        }
        let place = format!("{} of {n}", to + 1);
        e.log(
            LogLevel::Info,
            match (mv, to == from) {
                (QueueMove::Start, _) if training => {
                    format!(
                        "start requested: moved to the front ({place}); it starts when the active run ends"
                    )
                }
                (QueueMove::Start, _) => "start requested: starting now".to_string(),
                (_, true) => format!("queue position unchanged ({place})"),
                (_, false) => format!("moved in the queue to position {place}"),
            },
        );
        Ok(run)
    }

    /// Registers a new run (status `queued`) and queues it. `config` is already validated, its name filled in. A
    /// `checkpoint` (a validated snapshot of a full training checkpoint) is copied to the run's `weights/last.folo`
    /// first, so the worker continues that exact snapshot as it resumes a run. Carries optional auto-batch and
    /// `folo2-*` architecture overrides.
    #[allow(clippy::too_many_arguments)]
    pub fn create_with_options(
        self: &Arc<Self>,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        val_offset: u64,
        device: String,
        options: crate::store::RunOptions,
    ) -> Result<Run> {
        let entry = self
            .register_with_options(config, dataset, weights, checkpoint, val_offset, device, None, options)?;
        self.enqueue(&entry, 0)?;
        let run = entry.run();
        Ok(Run { queue_position: self.position(&run.id), ..run })
    }

    /// Registers a new cloud run (status `queued`, with `cloud`) like [`Runs::create_with_options`], but outside
    /// the queue: it is claimed at once, for [`crate::cloud`] to execute.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_cloud(
        &self,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        val_offset: u64,
        device: String,
        cloud: RunCloud,
    ) -> Result<Arc<RunEntry>> {
        self.register(config, dataset, weights, checkpoint, val_offset, device, Some(cloud))
    }

    /// Creates a run's folder and record and adds it to the registry; a cloud run (`cloud`) is claimed before
    /// anything can see it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn register(
        &self,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        val_offset: u64,
        device: String,
        cloud: Option<RunCloud>,
    ) -> Result<Arc<RunEntry>> {
        self.register_with_options(
            config,
            dataset,
            weights,
            checkpoint,
            val_offset,
            device,
            cloud,
            crate::store::RunOptions::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn register_with_options(
        &self,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<CheckpointSnapshot<'_>>,
        val_offset: u64,
        device: String,
        cloud: Option<RunCloud>,
        options: crate::store::RunOptions,
    ) -> Result<Arc<RunEntry>> {
        self.check_open()?;
        let mut list = self.list.write().unwrap();
        let seed = fnv1a(config.name.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("run", seed, |id| list.iter().any(|e| e.id == id) || self.dir.join(id).exists());
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        if let Some(ckpt) = &checkpoint {
            let last = dir.join("weights").join("last.folo");
            let copied =
                std::fs::create_dir_all(dir.join("weights")).and_then(|()| std::fs::copy(ckpt.staged, &last));
            if let Err(e) = copied {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(Error::io(format!(
                    "copying {} to {}: {e}",
                    ckpt.origin.display(),
                    last.display()
                )));
            }
        }
        let run = Run {
            id: id.clone(),
            name: config.name.clone(),
            status: RunStatus::Queued,
            epochs: checkpoint.as_ref().and_then(|c| c.epochs).unwrap_or(config.epochs),
            config,
            created: now(),
            started: None,
            finished: None,
            paused_s: None,
            epoch: 0,
            step: 0,
            steps_per_epoch: 0,
            eta_s: None,
            best: checkpoint.as_ref().and_then(|c| c.best),
            device,
            model_id: None,
            error: None,
            error_code: None,
            resolved: None,
            queue_position: None,
            cloud,
            // Filled in from `dataset` by `RunEntry::new`.
            dataset: None,
        };
        let file = RunFile {
            run,
            dataset,
            weights,
            val_offset,
            epochs_done: 0,
            last_metrics: None,
            resumed_at: None,
            alive_at: None,
            auto_batch: options.auto_batch,
            arch_override: options.arch,
        };
        let entry = Arc::new(RunEntry::new(id, dir, file));
        {
            let mut st = entry.state.lock().unwrap();
            st.claimed = st.file.run.cloud.is_some();
            entry.persist(&st);
        }
        list.push(entry.clone());
        drop(list);
        if let Some(ckpt) = checkpoint {
            entry.log(
                LogLevel::Info,
                format!("continuing the training state of {}", folo_api::json::path(ckpt.origin)),
            );
        }
        Ok(entry)
    }

    /// `503 shutting_down` once [`Runs::shutdown`] has started.
    pub(super) fn check_open(&self) -> Result<()> {
        if self.queue.lock().unwrap().shutdown { Err(shutting_down()) } else { Ok(()) }
    }

    /// Queues `entry` for `attempt`. After [`Runs::shutdown`] has started (checked under the queue lock, so a run is
    /// either queued before the shutdown drains the queue or not at all) the run is marked `stopped` instead and the
    /// result is `503 shutting_down`. A run stopped, deleted, already claimed (e.g. resumed on cloud), or superseded
    /// by a newer attempt since it was registered or restarted is not queued.
    pub(super) fn enqueue(self: &Arc<Self>, entry: &Arc<RunEntry>, attempt: u64) -> Result<()> {
        let ahead = {
            // Lock order: the run's state, then the queue (as `stop`).
            let mut st = entry.state.lock().unwrap();
            if st.file.run.status != RunStatus::Queued || st.deleted || st.claimed || st.attempt != attempt {
                return Ok(());
            }
            let mut q = self.queue.lock().unwrap();
            if q.shutdown {
                drop(q);
                entry.finish(&mut st, RunStatus::Stopped, None);
                drop(st);
                entry.log(LogLevel::Warn, "the server is shutting down; the run was not queued");
                return Err(shutting_down());
            }
            q.runs.push_back(QueuedRun { entry: entry.clone(), attempt });
            Self::notify_queue(&q);
            q.runs.len() - 1 + usize::from(self.active.lock().unwrap().is_some())
        };
        entry.log(
            LogLevel::Info,
            if ahead == 0 { "queued".to_string() } else { format!("queued behind {ahead} other run(s)") },
        );
        self.wake.notify_all();
        self.ensure_worker();
        Ok(())
    }

    fn ensure_worker(self: &Arc<Self>) {
        let mut started = self.worker.lock().unwrap();
        if *started {
            return;
        }
        let me = self.clone();
        match std::thread::Builder::new().name("folo-runs".into()).spawn(move || me.work()) {
            Ok(_) => *started = true,
            Err(e) => tracing::error!("could not start the training worker: {e}"),
        }
    }

    fn work(self: Arc<Self>) {
        loop {
            let QueuedRun { entry, attempt } = {
                let mut q = self.queue.lock().unwrap();
                let item = loop {
                    if q.shutdown {
                        return;
                    }
                    if let Some(e) = q.runs.pop_front() {
                        Self::notify_queue(&q);
                        break e;
                    }
                    q = self.wake.wait(q).unwrap();
                };
                // Set before the queue lock is released, so `enqueue` counts the runs ahead exactly.
                *self.active.lock().unwrap() = Some(item.entry.clone());
                self.activity.fetch_add(1, Ordering::SeqCst);
                item
            };
            // A benchmark probe finishes first; the device is the run's until it ends.
            let _gate = self.gate.lock().unwrap_or_else(|p| p.into_inner());
            // Lock order: registry, then a run's state, then the queue (as `delete` and `stop` take them); the
            // queue lock is released first. A run stopped, deleted, or superseded by a newer attempt in between
            // is skipped.
            if entry.claim(attempt) {
                entry.execute(|| {
                    execute(&entry, &self.throughput, &self.hold);
                    Ended::Finished
                });
            }
            *self.active.lock().unwrap() = None;
            self.activity.fetch_add(1, Ordering::SeqCst);
            self.idle.notify_all();
        }
    }

    /// Test support: while `on`, every run waits after each epoch until this is turned off or the run is
    /// asked to stop ([`EpochHold`](crate::runs::EpochHold)), so a test can stop a run that would otherwise
    /// finish first.
    pub fn hold_after_epochs(&self, on: bool) {
        *self.hold.on.lock().unwrap_or_else(|p| p.into_inner()) = on;
        self.hold.lifted.notify_all();
    }

    /// Requests a stop: a queued run stops at once, a running one after its current batch (keeping the last
    /// epoch's checkpoint).
    pub fn stop(&self, id: &str) -> Result<Run> {
        self.stop_as(id, false)
    }

    /// [`Runs::stop`]; `interrupted`: the server's shutdown asks, and the run ends with error code `interrupted`.
    fn stop_as(&self, id: &str, interrupted: bool) -> Result<Run> {
        let e = self.get(id)?;
        let mut st = e.state.lock().unwrap();
        if st.deleted {
            return Err(Error::not_found("run", id));
        }
        let cloud = st.file.run.config.compute.is_some();
        if cloud && st.file.run.status.is_active() && !st.claimed {
            return Err(followed_elsewhere(id));
        }
        match st.file.run.status {
            RunStatus::Queued if !st.claimed => {
                {
                    let mut q = self.queue.lock().unwrap();
                    let before = q.runs.len();
                    q.runs.retain(|q| q.entry.id != id);
                    if q.runs.len() != before {
                        Self::notify_queue(&q);
                    }
                }
                e.finish_with_code(
                    &mut st,
                    RunStatus::Stopped,
                    None,
                    interrupted.then(|| INTERRUPTED.to_string()),
                );
                let run = st.file.shown();
                drop(st);
                if interrupted {
                    e.log(
                        LogLevel::Warn,
                        "the server stopped while the run waited in the queue; start it from its page in Studio",
                    );
                } else {
                    e.log(LogLevel::Info, "stopped before it started");
                }
                Ok(run)
            }
            RunStatus::Queued | RunStatus::Running => {
                st.interrupted |= interrupted;
                e.stop.store(true, Ordering::Relaxed);
                st.file.run.status = RunStatus::Stopping;
                e.persist(&st);
                e.send(RunEvent::Status(status_event(&st.file.run)));
                let run = st.file.shown();
                drop(st);
                let message = match run.cloud.as_ref().map(|c| c.phase) {
                    Some(p) if cloud && p < CloudPhase::Training => "stop requested; releasing the instance",
                    _ if interrupted && !cloud => {
                        "the server is shutting down: the run stops after the current batch and keeps its last \
                         checkpoint; resume it from its page in Studio"
                    }
                    _ => STOP_REQUESTED,
                };
                e.log(LogLevel::Info, message);
                Ok(run)
            }
            RunStatus::Stopping => Ok(st.file.shown()),
            s => Err(Error::run_not_active(id, serde_json::to_value(s).unwrap().as_str().unwrap_or(""))),
        }
    }

    /// Stops accepting work, stops the active run (it keeps its last checkpoint) and marks queued runs `stopped`.
    /// With `wait`, blocks until the active run has stopped (at most `wait`).
    pub fn shutdown(&self, wait: Option<Duration>) {
        let queued: Vec<Arc<RunEntry>> = {
            let mut q = self.queue.lock().unwrap();
            q.shutdown = true;
            q.runs.drain(..).map(|item| item.entry).collect()
        };
        self.wake.notify_all();
        for e in queued {
            let mut st = e.state.lock().unwrap();
            if st.file.run.status == RunStatus::Queued && !st.deleted {
                e.finish_with_code(&mut st, RunStatus::Stopped, None, Some(INTERRUPTED.to_string()));
                drop(st);
                e.log(
                    LogLevel::Warn,
                    "the server stopped while the run waited in the queue; start it from its page in Studio",
                );
            }
        }
        let active = self.active.lock().unwrap().clone();
        if let Some(e) = active {
            let _ = self.stop_as(&e.id, true);
        }
        if let Some(wait) = wait {
            let t0 = Instant::now();
            let mut a = self.active.lock().unwrap();
            while a.is_some() && t0.elapsed() < wait {
                a = self.idle.wait_timeout(a, Duration::from_millis(50)).unwrap().0;
            }
        }
    }
}

/// `409 run_not_queued`: moving a run that is not waiting in the queue.
fn not_queued(id: &str, status: RunStatus) -> Error {
    let what = match status {
        // Taken off the queue by the worker, which has not reported `running` yet.
        RunStatus::Queued => "is starting".to_string(),
        s => format!("is {}", serde_json::to_value(s).unwrap().as_str().unwrap_or("not queued")),
    };
    Error::conflict("run_not_queued", format!("run '{id}' {what}, not waiting in the queue"))
        .hint("only runs waiting in the queue (with a queue_position) can be moved")
}

/// `409 run_active` for a cloud run another FOLO process on this data dir follows (it holds `cloud.lock`): this
/// process shows the run as persisted and leaves it alone.
pub(crate) fn followed_elsewhere(id: &str) -> Error {
    Error::conflict("run_active", format!("run '{id}' is followed by another FOLO process on this data dir"))
        .hint("stop it there, or stop that process: the next `folo serve` on this data dir follows the run")
}

fn shutting_down() -> Error {
    Error::new(StatusCode::SERVICE_UNAVAILABLE, "shutting_down", "the server is shutting down")
        .hint("start the run again once the server has restarted")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::throughput::Throughput;

    fn test_runs(name: &str) -> (folo_test_support::TestDir, Arc<Runs>) {
        let dir = folo_test_support::TestDir::new(name);
        let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
        (dir, runs)
    }

    fn register_run(runs: &Runs, name: &str) -> Arc<RunEntry> {
        let config: TrainConfig =
            serde_json::from_str(&format!(r#"{{"dataset_id":"ds_1","name":"{name}"}}"#)).unwrap();
        let dataset = DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: "nowhere".into(),
            format: folo_api::DatasetFormat::Yolo,
            classes: vec![],
        };
        runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap()
    }

    #[test]
    fn shutdown_preserves_client_stop_and_marks_unclaimed_active_run_interrupted() {
        // 1. A run already stopped by a client (`Stopping`) must NOT become `interrupted` when shutdown arrives
        //    before its current batch completes.
        let (_dir1, runs1) = test_runs("runs-stop-client-vs-shutdown");
        let e1 = register_run(&runs1, "client-stop");
        {
            let mut st = e1.state.lock().unwrap();
            st.claimed = true;
            st.file.run.status = RunStatus::Running;
        }
        *runs1.active.lock().unwrap() = Some(e1.clone());

        let stopping = runs1.stop(e1.id()).unwrap();
        assert_eq!(stopping.status, RunStatus::Stopping);
        runs1.shutdown(None);
        e1.execute(|| {
            e1.set_outcome(RunStatus::Stopped, None, None);
            Ended::Finished
        });
        let done1 = e1.run();
        assert_eq!(done1.status, RunStatus::Stopped);
        assert_eq!(done1.error_code, None, "client-stopped run must keep error_code=None across shutdown");

        // 2. A run popped into `active` by the worker (waiting on `gate`, still `Queued` & unclaimed) must be
        //    marked `interrupted` when shutdown stops it.
        let (_dir2, runs2) = test_runs("runs-shutdown-unclaimed-active");
        let e2 = register_run(&runs2, "popped-active");
        *runs2.active.lock().unwrap() = Some(e2.clone());
        runs2.shutdown(None);
        let done2 = e2.run();
        assert_eq!(done2.status, RunStatus::Stopped);
        assert_eq!(done2.error_code.as_deref(), Some(INTERRUPTED));
    }
}
