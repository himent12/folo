//! The run registry: loading it from disk, looking runs up, listing them with their queue places, deleting them
//! and gating the device for benchmark probes.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, TryLockError};

use folo_api::{LogLevel, Run, RunStatus};

use crate::error::{Error, Result};
use crate::throughput::Throughput;

use super::history::reconcile;
use super::{Queue, RunEntry, RunFile, Runs};

impl Runs {
    /// Loads the runs under `dir`. Runs a previous process left active become `stopped`, except cloud runs, which
    /// stay as they are for [`crate::cloud`]. Trained epochs are recorded in `throughput`.
    pub fn open(dir: PathBuf, throughput: Arc<Throughput>) -> Arc<Runs> {
        let mut list = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for d in rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()) {
                let path = d.join("run.json");
                let Ok(bytes) = std::fs::read(&path) else { continue };
                let file: RunFile = match serde_json::from_slice(&bytes) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!("skipping {}: {e}", path.display());
                        continue;
                    }
                };
                let mut file = file;
                // Runs from before `Run.resolved` (or its `close_mosaic`): the trainer's config.json has the same
                // numbers.
                if file.run.resolved.as_ref().is_none_or(|r| r.close_mosaic.is_none()) {
                    let from_config = std::fs::read(d.join("config.json"))
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok())
                        .and_then(|v| folo_vision::train::ResolvedOptim::from_run_config(&v));
                    match (&mut file.run.resolved, from_config) {
                        (Some(r), Some(c)) => r.close_mosaic = c.close_mosaic,
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
                    if st.file.run.status.is_active() && st.file.run.config.compute.is_none() {
                        let was = st.file.run.status;
                        reconcile(&mut st.file, &e.dir);
                        e.finish(&mut st, RunStatus::Stopped, None);
                        drop(st);
                        e.log(
                            LogLevel::Warn,
                            format!(
                                "the server stopped while the run was {}; it is stopped now (resume it with \
                                 POST /api/runs and resume_run_id)",
                                serde_json::to_value(was).unwrap().as_str().unwrap_or("active")
                            ),
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
            .map(|(e, st)| {
                let mut r = st.file.run.clone();
                r.queue_position = q.runs.iter().position(|x| x.id == e.id).map(|i| i as u32 + 1);
                r
            })
            .collect()
    }

    /// The API record of run `id` with its place in the queue ([`Run::queue_position`]).
    pub fn snapshot(&self, id: &str) -> Result<Run> {
        let e = self.get(id)?;
        Ok(self.record(&e))
    }

    /// `e`'s record and its place in the queue, read together (its state, then the queue: the order `stop` and
    /// `move_queued` lock them in), so a run that has a place is `queued`: the worker takes a run off the queue
    /// before it can report `running`, and a stop removes it from the queue under its state lock.
    pub(super) fn record(&self, e: &RunEntry) -> Run {
        let st = e.state.lock().unwrap();
        let mut r = st.file.run.clone();
        let q = self.queue.lock().unwrap();
        r.queue_position = q.runs.iter().position(|x| x.id == e.id).map(|i| i as u32 + 1);
        r
    }

    /// The 1-based place of run `id` among the waiting runs.
    pub(super) fn position(&self, id: &str) -> Option<u32> {
        let q = self.queue.lock().unwrap();
        q.runs.iter().position(|e| e.id == id).map(|i| i as u32 + 1)
    }

    /// Deletes a finished run and its folder.
    pub fn delete(&self, id: &str) -> Result<PathBuf> {
        let mut list = self.list.write().unwrap();
        let e = list.iter().find(|e| e.id == id).cloned().ok_or_else(|| Error::not_found("run", id))?;
        let mut st = e.state.lock().unwrap();
        if st.file.run.status.is_active() || st.claimed {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                .hint("stop it first (POST /api/runs/{id}/stop)"));
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
