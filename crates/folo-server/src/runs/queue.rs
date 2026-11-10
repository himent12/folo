//! Creating runs, the FIFO queue and its worker thread, stops and shutdown.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{CloudPhase, LogLevel, QueueEvent, Run, RunCloud, RunEvent, RunStatus, TrainConfig};

use crate::error::{Error, Result};
use crate::store::QueueMove;
use crate::util::{fnv1a, unique_id, unix_now};

use super::entry::{Ended, status_event};
use super::train::execute;
use super::{DatasetRef, Queue, RunEntry, RunFile, Runs, now};

impl Runs {
    /// Sends every waiting run its place in the queue (the SSE `queue` event). Called with the queue locked, after
    /// every change to it, so a run's events arrive in the order of the changes, and before the `running` status
    /// of the worker taking it.
    fn notify_queue(q: &Queue) {
        let waiting = q.runs.len() as u32;
        for (i, e) in q.runs.iter().enumerate() {
            e.send(RunEvent::Queue(QueueEvent { position: i as u32 + 1, waiting }));
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
        let Some(from) = q.runs.iter().position(|x| x.id == id) else {
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
        let mut run = st.file.run.clone();
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
    /// `checkpoint` (a full training checkpoint) is copied to the run's `weights/last.folo` first, so the worker
    /// continues it as it resumes a run.
    pub fn create(
        self: &Arc<Self>,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<&Path>,
        val_offset: u64,
        device: String,
    ) -> Result<Run> {
        let entry = self.register(config, dataset, weights, checkpoint, val_offset, device, None)?;
        self.enqueue(&entry)?;
        let run = entry.run();
        Ok(Run { queue_position: self.position(&run.id), ..run })
    }

    /// Registers a new cloud run (status `queued`, with `cloud`) like [`Runs::create`], but outside the queue: it
    /// is claimed at once, for [`crate::cloud`] to execute.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_cloud(
        &self,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<&Path>,
        val_offset: u64,
        device: String,
        cloud: RunCloud,
    ) -> Result<Arc<RunEntry>> {
        self.register(config, dataset, weights, checkpoint, val_offset, device, Some(cloud))
    }

    /// Creates a run's folder and record and adds it to the registry; a cloud run (`cloud`) is claimed before
    /// anything can see it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn register(
        &self,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        checkpoint: Option<&Path>,
        val_offset: u64,
        device: String,
        cloud: Option<RunCloud>,
    ) -> Result<Arc<RunEntry>> {
        self.check_open()?;
        let mut list = self.list.write().unwrap();
        let seed = fnv1a(config.name.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("run", seed, |id| list.iter().any(|e| e.id == id) || self.dir.join(id).exists());
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
        if let Some(src) = checkpoint {
            let last = dir.join("weights").join("last.folo");
            let copied =
                std::fs::create_dir_all(dir.join("weights")).and_then(|()| std::fs::copy(src, &last));
            if let Err(e) = copied {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(Error::io(format!("copying {} to {}: {e}", src.display(), last.display())));
            }
        }
        let run = Run {
            id: id.clone(),
            name: config.name.clone(),
            status: RunStatus::Queued,
            epochs: config.epochs,
            config,
            created: now(),
            started: None,
            finished: None,
            epoch: 0,
            step: 0,
            steps_per_epoch: 0,
            eta_s: None,
            best: None,
            device,
            model_id: None,
            error: None,
            error_code: None,
            resolved: None,
            queue_position: None,
            cloud,
        };
        let file = RunFile { run, dataset, weights, val_offset, epochs_done: 0, last_metrics: None };
        let entry = Arc::new(RunEntry::new(id, dir, file));
        {
            let mut st = entry.state.lock().unwrap();
            st.claimed = st.file.run.cloud.is_some();
            entry.persist(&st);
        }
        list.push(entry.clone());
        drop(list);
        if let Some(src) = checkpoint {
            entry.log(
                LogLevel::Info,
                format!("continuing the training state of {}", folo_api::json::path(src)),
            );
        }
        Ok(entry)
    }

    /// `503 shutting_down` once [`Runs::shutdown`] has started.
    pub(super) fn check_open(&self) -> Result<()> {
        if self.queue.lock().unwrap().shutdown { Err(shutting_down()) } else { Ok(()) }
    }

    /// Queues `entry`. After [`Runs::shutdown`] has started (checked under the queue lock, so a run is either
    /// queued before the shutdown drains the queue or not at all) the run is marked `stopped` instead and the
    /// result is `503 shutting_down`. A run stopped or deleted since it was registered or restarted (it was `queued`
    /// outside the queue, so it stopped at once) is not queued.
    pub(super) fn enqueue(self: &Arc<Self>, entry: &Arc<RunEntry>) -> Result<()> {
        let ahead = {
            // Lock order: the run's state, then the queue (as `stop`).
            let mut st = entry.state.lock().unwrap();
            if st.file.run.status != RunStatus::Queued || st.deleted {
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
            q.runs.push_back(entry.clone());
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
            let entry = {
                let mut q = self.queue.lock().unwrap();
                let entry = loop {
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
                *self.active.lock().unwrap() = Some(entry.clone());
                self.activity.fetch_add(1, Ordering::SeqCst);
                entry
            };
            // A benchmark probe finishes first; the device is the run's until it ends.
            let _gate = self.gate.lock().unwrap_or_else(|p| p.into_inner());
            // Lock order: registry, then a run's state, then the queue (as `delete` and `stop` take them); the
            // queue lock is released first. A run stopped or deleted in between is skipped.
            if entry.claim() {
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

    /// Test support: while `on`, every run waits after each epoch until this is turned off or the run is asked to
    /// stop ([`EpochHold`]), so a test can stop a run that would otherwise finish first.
    pub fn hold_after_epochs(&self, on: bool) {
        *self.hold.on.lock().unwrap_or_else(|p| p.into_inner()) = on;
        self.hold.lifted.notify_all();
    }

    /// Requests a stop: a queued run stops at once, a running one after its current batch (keeping the last
    /// epoch's checkpoint).
    pub fn stop(&self, id: &str) -> Result<Run> {
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
                    q.runs.retain(|q| q.id != id);
                    Self::notify_queue(&q);
                }
                e.finish(&mut st, RunStatus::Stopped, None);
                let run = st.file.run.clone();
                drop(st);
                e.log(LogLevel::Info, "stopped before it started");
                Ok(run)
            }
            RunStatus::Queued | RunStatus::Running => {
                e.stop.store(true, Ordering::Relaxed);
                st.file.run.status = RunStatus::Stopping;
                e.persist(&st);
                e.send(RunEvent::Status(status_event(&st.file.run)));
                let run = st.file.run.clone();
                drop(st);
                let message = match run.cloud.as_ref().map(|c| c.phase) {
                    Some(p) if cloud && p < CloudPhase::Training => "stop requested; releasing the instance",
                    _ if cloud => "stop requested; the worker stops after its current batch",
                    _ => "stop requested; stopping after the current batch",
                };
                e.log(LogLevel::Info, message);
                Ok(run)
            }
            RunStatus::Stopping => Ok(st.file.run.clone()),
            s => Err(Error::run_not_active(id, serde_json::to_value(s).unwrap().as_str().unwrap_or(""))),
        }
    }

    /// Stops accepting work, stops the active run (it keeps its last checkpoint) and marks queued runs `stopped`.
    /// With `wait`, blocks until the active run has stopped (at most `wait`).
    pub fn shutdown(&self, wait: Option<Duration>) {
        let queued: Vec<Arc<RunEntry>> = {
            let mut q = self.queue.lock().unwrap();
            q.shutdown = true;
            q.runs.drain(..).collect()
        };
        self.wake.notify_all();
        for e in queued {
            let mut st = e.state.lock().unwrap();
            if st.file.run.status == RunStatus::Queued && !st.deleted {
                e.finish(&mut st, RunStatus::Stopped, None);
            }
        }
        let active = self.active.lock().unwrap().clone();
        if let Some(e) = active {
            let _ = self.stop(&e.id);
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
