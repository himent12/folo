//! The run registry: loading it from disk, looking runs up, listing them with their queue places, deleting them
//! and gating the device for benchmark probes.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, TryLockError};

use folo_api::{LogLevel, Run, RunStatus};

use crate::error::{Error, Result};
use crate::throughput::Throughput;

use super::entry::INTERRUPTED;
use super::history::reconcile;
use super::queue::followed_elsewhere;
use super::{Queue, RunEntry, RunFile, Runs};

impl Runs {
    /// Loads the runs under `dir`. Runs a previous process left active become `stopped`, except cloud runs, which
    /// stay as they are for [`crate::cloud`]. Trained epochs are recorded in `throughput`.
    pub fn open(dir: PathBuf, throughput: Arc<Throughput>) -> Arc<Runs> {
        let mut list = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut dirs: Vec<PathBuf> =
                rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
            dirs.sort();
            let mut files = Vec::new();
            for d in dirs {
                let path = d.join("run.json");
                let Ok(bytes) = std::fs::read(&path) else { continue };
                match serde_json::from_slice::<RunFile>(&bytes) {
                    Ok(f) => files.push((d, f.run.id.clone(), f)),
                    Err(e) => tracing::warn!("skipping {}: {e}", path.display()),
                }
            }
            // A copy of a run folder (a backup made next to it) is not a second run of its id (x3-server-9).
            for (d, mut file) in crate::util::one_per_id("run", files) {
                // Runs from before `Run.resolved` (or its `close_mosaic`, or its EMA pair): the trainer's config.json
                // has the same numbers. (A run without an EMA has no pair: its config.json gives none either.)
                if file
                    .run
                    .resolved
                    .as_ref()
                    .is_none_or(|r| r.close_mosaic.is_none() || r.ema_decay.is_none())
                {
                    let from_config = std::fs::read(d.join("config.json"))
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok())
                        .and_then(|v| folo_vision::train::ResolvedOptim::from_run_config(&v));
                    match (&mut file.run.resolved, from_config) {
                        (Some(r), Some(c)) => {
                            r.close_mosaic = r.close_mosaic.or(c.close_mosaic);
                            if r.ema_decay.is_none() {
                                (r.ema_decay, r.ema_tau) = (c.ema_decay, c.ema_tau);
                            }
                        }
                        (r @ None, c) => *r = c,
                        _ => {}
                    }
                }
                file.run.queue_position = None;
                let id = file.run.id.clone();
                let e = RunEntry::new(id, d, file);
                {
                    let mut st = e.state.lock().unwrap();
                    // An active cloud run trains on its worker without this server: `crate::cloud` follows it again
                    // or ends it.
                    // One that was stopping had been asked to stop: it ends as a stop, the others as interrupted.
                    if st.file.run.status.is_active() && st.file.run.config.compute.is_none() {
                        let was = st.file.run.status;
                        reconcile(&mut st.file, &e.dir);
                        let code = (was != RunStatus::Stopping).then(|| INTERRUPTED.to_string());
                        e.finish_with_code(&mut st, RunStatus::Stopped, None, code);
                        drop(st);
                        e.log(
                            LogLevel::Warn,
                            match was {
                                RunStatus::Queued => {
                                    "the server stopped while the run waited in the queue; start it from its page \
                                     in Studio"
                                }
                                RunStatus::Stopping => "the server stopped while the run was stopping; it is stopped now",
                                _ => {
                                    "the server stopped during training; resume the run from its page in Studio (it \
                                     continues from its last checkpoint)"
                                }
                            },
                        );
                    }
                }
                list.push(Arc::new(e));
            }
        }
        Arc::new(Runs {
            dir,
            list: RwLock::new(list),
            queue: Mutex::new(Queue { runs: VecDeque::new(), shutdown: false }),
            wake: Condvar::new(),
            active: Mutex::new(None),
            idle: Condvar::new(),
            worker: Mutex::new(false),
            gate: Mutex::new(()),
            throughput,
            activity: AtomicU64::new(0),
            hold: Arc::default(),
        })
    }

    /// Changes whenever the worker takes up or lets go of a run: equal values around a prediction (with
    /// [`Runs::busy`] false at both ends) mean no run trained during it.
    pub fn activity(&self) -> u64 {
        self.activity.load(Ordering::SeqCst)
    }

    /// A run is queued, training or stopping.
    pub fn busy(&self) -> bool {
        !self.queue.lock().unwrap().runs.is_empty() || self.active.lock().unwrap().is_some()
    }

    /// The device to a benchmark probe: `409 run_active` while a run is queued or training (or another probe
    /// holds it). Runs queued meanwhile wait for the guard to drop.
    pub fn probe_gate(&self) -> Result<MutexGuard<'_, ()>> {
        let busy = || {
            Error::conflict(
                "run_active",
                "a training run is queued or running; a probe now would measure both slower",
            )
            .hint("wait until it finishes (or stop it), then probe again")
        };
        let guard = match self.gate.try_lock() {
            Ok(g) => g,
            Err(TryLockError::WouldBlock) => {
                return Err(if self.busy() {
                    busy()
                } else {
                    Error::conflict("run_active", "another benchmark probe is running")
                        .hint("try again in a few seconds")
                });
            }
            Err(TryLockError::Poisoned(p)) => p.into_inner(),
        };
        if self.busy() {
            return Err(busy());
        }
        Ok(guard)
    }

    /// One run.
    pub fn get(&self, id: &str) -> Result<Arc<RunEntry>> {
        self.list
            .read()
            .unwrap()
            .iter()
            .find(|e| e.id == id)
            .cloned()
            .ok_or_else(|| Error::not_found("run", id))
    }

    /// Every run, newest first.
    pub fn entries(&self) -> Vec<Arc<RunEntry>> {
        let list = self.list.read().unwrap();
        let mut v: Vec<(String, usize, Arc<RunEntry>)> = list
            .iter()
            .enumerate()
            .map(|(i, e)| (e.state.lock().unwrap().file.run.created.clone(), i, e.clone()))
            .collect();
        // Same-second runs: the later registered first.
        v.sort_by(|a, b| (&b.0, b.1).cmp(&(&a.0, a.1)));
        v.into_iter().map(|(_, _, e)| e).collect()
    }

    /// Every run, newest first, with its place in the queue. Every run's state is locked, then the queue once (the
    /// order `stop` and `move_queued` take them in), so the places are those of one queue: unique and gapless.
    pub fn list(&self) -> Vec<Run> {
        let entries = self.entries();
        let states: Vec<_> = entries.iter().map(|e| e.state.lock().unwrap()).collect();
        let q = self.queue.lock().unwrap();
        entries
            .iter()
            .zip(&states)
            .filter(|(_, st)| !st.deleted)
            .map(|(e, st)| {
                let mut r = st.file.shown();
                r.queue_position = q.runs.iter().position(|x| x.entry.id == e.id).map(|i| i as u32 + 1);
                r
            })
            .collect()
    }

    /// The API record of run `id` with its place in the queue ([`Run::queue_position`]).
    pub fn snapshot(&self, id: &str) -> Result<Run> {
        let e = self.get(id)?;
        self.record(&e)
    }

    /// `e`'s record and its place in the queue, read together (its state, then the queue: the order `stop` and
    /// `move_queued` lock them in), so a run that has a place is `queued`: the worker takes a run off the queue
    /// before it can report `running`, and a stop removes it from the queue under its state lock.
    pub(super) fn record(&self, e: &RunEntry) -> Result<Run> {
        let st = e.state.lock().unwrap();
        if st.deleted {
            return Err(Error::not_found("run", &e.id));
        }
        let mut r = st.file.shown();
        let q = self.queue.lock().unwrap();
        r.queue_position = q.runs.iter().position(|x| x.entry.id == e.id).map(|i| i as u32 + 1);
        Ok(r)
    }

    /// The 1-based place of run `id` among the waiting runs.
    pub(super) fn position(&self, id: &str) -> Option<u32> {
        let q = self.queue.lock().unwrap();
        q.runs.iter().position(|e| e.entry.id == id).map(|i| i as u32 + 1)
    }

    /// Deletes a finished run and its folder, or a local run waiting in this server's queue that never started (it
    /// leaves the queue). A run that trains, stops, is being started or waits to resume is `409 run_active`, and so
    /// is every active cloud run: its instance may be starting, here or for another process on this data dir.
    pub fn delete(&self, id: &str) -> Result<PathBuf> {
        // Lock order: registry, then the run's state, then the queue, then `active` (matching the worker, `stop`,
        // and `move_queued`).
        let mut list = self.list.write().unwrap();
        let e = list.iter().find(|e| e.id == id).cloned().ok_or_else(|| Error::not_found("run", id))?;
        let mut st = e.state.lock().unwrap();
        let r = &st.file.run;
        let cloud = r.config.compute.is_some();
        if cloud && r.status.is_active() && !st.claimed {
            return Err(followed_elsewhere(id));
        }
        let never_started = !cloud && r.status == RunStatus::Queued && r.started.is_none() && !st.claimed;
        if (r.status.is_active() && !never_started) || st.claimed {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                .hint(format!("stop it first (POST /api/runs/{id}/stop)")));
        }
        if never_started {
            let mut q = self.queue.lock().unwrap();
            if self.active.lock().unwrap().as_ref().is_some_and(|a| a.id == id) {
                return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                    .hint(format!("stop it first (POST /api/runs/{id}/stop)")));
            }
            let before = q.runs.len();
            q.runs.retain(|x| x.entry.id != id);
            if q.runs.len() != before {
                Self::notify_queue(&q);
            }
        }
        st.deleted = true;
        list.retain(|x| x.id != id);
        drop(st);
        drop(list);
        if let Err(err) = std::fs::remove_dir_all(&e.dir) {
            tracing::warn!("removing {}: {err}", e.dir.display());
        }
        Ok(e.dir.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::DatasetRef;
    use crate::store::{LogsQuery, MetricsQuery};
    use axum::http::StatusCode;
    use folo_api::{DatasetFormat, TrainConfig};
    use folo_test_support::TestDir;

    fn sample_run(runs: &Runs, name: &str) -> Arc<RunEntry> {
        let config: TrainConfig =
            serde_json::from_value(serde_json::json!({ "dataset_id": "ds_1", "name": name })).unwrap();
        let dataset = DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: "nowhere".into(),
            format: DatasetFormat::Yolo,
            classes: vec![],
        };
        runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap()
    }

    /// A queued run already popped into `active` by the worker (e.g. waiting on the device gate before `claim()`)
    /// is being started and must refuse `delete` with `409 run_active` instead of removing its folder underneath
    /// the worker.
    #[test]
    fn delete_refuses_queued_run_being_started_by_worker() {
        let dir = TestDir::new("runs-delete-active-starting");
        let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
        let e = sample_run(&runs, "starting");

        *runs.active.lock().unwrap() = Some(e.clone());
        let err = runs.delete(e.id()).unwrap_err();
        assert_eq!((err.status, err.body.code.as_str()), (StatusCode::CONFLICT, "run_active"));
        assert!(e.dir.is_dir(), "refused delete must leave run directory intact");

        *runs.active.lock().unwrap() = None;
        runs.delete(e.id()).unwrap();
        assert!(!e.dir.exists());
        runs.shutdown(None);
    }

    /// Once a run is marked `deleted`, concurrent readers (`record`, `subscribe`, `subscribe_sse`, `metrics`,
    /// `logs`, and `list`) never return stale `200 OK` records or empty logs/metrics for it.
    #[test]
    fn deleted_run_returns_not_found_across_reads_and_subscriptions() {
        let dir = TestDir::new("runs-deleted-reads");
        let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
        let e = sample_run(&runs, "race");
        let id = e.id().to_string();

        // Simulate the race window where `self.get(id)` returned `e` just before `delete` marked `st.deleted`.
        e.state.lock().unwrap().deleted = true;

        assert_eq!(runs.record(&e).unwrap_err().status, StatusCode::NOT_FOUND);
        assert_eq!(runs.snapshot(&id).unwrap_err().status, StatusCode::NOT_FOUND);
        assert_eq!(runs.subscribe(&id).err().unwrap().status, StatusCode::NOT_FOUND);
        assert_eq!(runs.subscribe_sse(&id).err().unwrap().status, StatusCode::NOT_FOUND);
        assert_eq!(runs.metrics(&id, &MetricsQuery::default()).unwrap_err().status, StatusCode::NOT_FOUND);
        assert_eq!(runs.logs(&id, &LogsQuery::last(10)).unwrap_err().status, StatusCode::NOT_FOUND);
        assert!(runs.list().is_empty(), "list() must exclude concurrently deleted runs");
        runs.shutdown(None);
    }

    /// A worker that popped attempt 0 before blocking on the device gate must not claim attempt 1 after
    /// `stop` + `restart` (whether resumed locally or claimed by a cloud executor).
    #[test]
    fn stale_worker_cannot_claim_newer_or_cloud_resumed_attempt() {
        let dir = TestDir::new("runs-stale-attempt-claim");
        let runs = Runs::open(dir.to_path_buf(), Arc::new(Throughput::in_memory(Default::default())));
        let e = sample_run(&runs, "race-attempt");
        let attempt0 = e.state.lock().unwrap().attempt;

        // Attempt 0 is stopped before the worker acquires the gate, then restarted into attempt 1.
        runs.stop(e.id()).unwrap();
        let cfg = e.run().config.clone();
        let (_, attempt1) = runs.restart(&e, cfg.clone(), None).unwrap();
        assert_eq!(attempt1, attempt0 + 1);

        // A delayed `enqueue` from attempt 0 must not duplicate attempt 1 in the queue.
        runs.enqueue(&e, attempt0).unwrap();
        assert!(runs.queue.lock().unwrap().runs.is_empty(), "stale enqueue must not push into the queue");
        assert!(!*runs.worker.lock().unwrap(), "stale enqueue must not start the worker");

        assert!(!e.claim(attempt0), "stale attempt 0 must not claim restarted attempt 1");
        assert!(e.claim(attempt1), "matching attempt 1 should claim once");
        assert!(!e.claim(attempt1), "already-claimed attempt must not be claimed twice");

        // Also verify a cloud-claimed attempt cannot be enqueued locally or claimed by the local worker.
        e.state.lock().unwrap().claimed = false;
        e.state.lock().unwrap().file.run.status = folo_api::RunStatus::Stopped;
        let cloud = folo_api::RunCloud {
            connection_id: "c_1".into(),
            provider: "runpod".into(),
            offer: folo_api::CloudOffer {
                id: "a100".into(),
                gpu: "A100".into(),
                gpu_count: 1,
                vram_bytes: 80 << 30,
                price_per_hour: Some(1.0),
                region: None,
                detail: None,
                availability: folo_api::CloudAvailability::High,
            },
            instance_id: None,
            phase: folo_api::CloudPhase::Provisioning,
            progress: None,
            max_hours: 1.0,
            deadline: None,
            cost_usd: None,
            synced: None,
            ended_in: None,
            unanswered_since: None,
        };
        let (_, attempt2) = runs.restart(&e, cfg, Some(cloud)).unwrap();
        runs.enqueue(&e, attempt1).unwrap();
        runs.enqueue(&e, attempt2).unwrap();
        assert!(runs.queue.lock().unwrap().runs.is_empty(), "cloud-claimed run must never enter local queue");
        assert!(!e.claim(attempt2), "local worker must not claim a cloud-claimed attempt");
        runs.shutdown(None);
    }
}
