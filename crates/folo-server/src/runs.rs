//! Training runs of the real store: a registry persisted under `<data dir>/runs/<id>/`, a FIFO queue and one
//! worker thread that trains one run at a time with folo-vision's [`Trainer`].
//!
//! Each run folder holds `run.json` (the API `Run` plus what the server needs to restart it, rewritten atomically
//! on every status change and after every epoch), `log.jsonl` (every log line, flushed as it happens), and what the
//! trainer writes: `config.json` (the resolved trainer config), `metrics.jsonl` (every event, every step),
//! `weights/{last,best}.folo` and `samples/epoch_NNN.json`.
//!
//! Live events reach SSE subscribers through a per-run broadcast channel: the trainer's sink (on the worker
//! thread) updates the run, appends logs and forwards each event. On open, runs left `queued`, `running` or
//! `stopping` by a previous process become `stopped`; [`Runs::resume`] continues one from `weights/last.folo`
//! (same id, the full checkpoint state) or starts it over when it never finished an epoch.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    BestEpoch, DatasetFormat, Epoch, LogLevel, LogLine, ModelMetrics, Run, RunEvent, RunMetrics, RunStatus,
    StatusEvent, Step, TrainConfig,
};
use folo_data::{Dataset as _, Split};
use folo_vision::train::ResumeOverrides;
use folo_vision::{TrainData, TrainOptions, Trainer};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::engine;
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::store::Subscription;
use crate::util::{fnv1a, rfc3339, unique_id, unix_now};

/// Events buffered per SSE subscriber before it counts as lagging.
pub const EVENT_BUFFER: usize = 1024;
/// Steps kept in memory between epochs (the metrics file has all of them once an epoch ends).
const TAIL_STEPS: usize = 10_000;

/// The dataset a run trains on, as registered when it was created.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetRef {
    /// Registered id.
    pub id: String,
    /// Name.
    pub name: String,
    /// Absolute path (a folder, `data.yaml` or COCO `.json`).
    pub path: String,
    /// Format.
    pub format: DatasetFormat,
}

/// `run.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunFile {
    run: Run,
    dataset: DatasetRef,
    /// Weights to start from (`TrainConfig.weights` resolved to a file).
    #[serde(default)]
    weights: Option<PathBuf>,
    /// Training images in the dataset index when the run was created: validation image `i` is dataset image
    /// `val_offset + i` (0 when the run validates on the training split).
    #[serde(default)]
    val_offset: u64,
    /// Epochs validated so far.
    #[serde(default)]
    epochs_done: u32,
    /// Validation metrics of the last validated epoch (those of `weights/last.folo`).
    #[serde(default)]
    last_metrics: Option<ModelMetrics>,
}

struct RunState {
    file: RunFile,
    /// Steps since the last `epoch` event (`metrics.jsonl` is flushed at every epoch).
    tail: Vec<Step>,
    /// The worker took the run off the queue (a stop now goes through the trainer).
    claimed: bool,
    /// The trainer's final status, published once the worker has let go of the run (so a client that sees it may
    /// resume or delete the run right away).
    outcome: Option<(RunStatus, Option<String>)>,
    /// Removed from the registry: every later request treats the run as unknown.
    deleted: bool,
}

/// One run.
pub struct RunEntry {
    id: String,
    dir: PathBuf,
    state: Mutex<RunState>,
    tx: broadcast::Sender<RunEvent>,
    stop: Arc<AtomicBool>,
}

fn now() -> String {
    rfc3339(unix_now())
}

fn append_line(path: &Path, line: &str) {
    let r = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = r {
        tracing::warn!("writing {}: {e}", path.display());
    }
}

impl RunEntry {
    fn new(id: String, dir: PathBuf, file: RunFile) -> RunEntry {
        let (tx, _) = broadcast::channel(EVENT_BUFFER);
        RunEntry {
            id,
            dir,
            state: Mutex::new(RunState {
                file,
                tail: Vec::new(),
                claimed: false,
                outcome: None,
                deleted: false,
            }),
            tx,
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The API record.
    pub fn run(&self) -> Run {
        self.state.lock().unwrap().file.run.clone()
    }

    /// The dataset it trains on.
    pub fn dataset(&self) -> DatasetRef {
        self.state.lock().unwrap().file.dataset.clone()
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

    fn persist(&self, st: &RunState) {
        let bytes = serde_json::to_vec_pretty(&st.file).expect("runs serialize");
        let path = self.dir.join("run.json");
        if let Err(e) = write_atomic(&path, &bytes) {
            tracing::warn!("writing {}: {e}", path.display());
        }
    }

    fn send(&self, ev: RunEvent) {
        // No receivers is fine: nobody is watching.
        let _ = self.tx.send(ev);
    }

    fn write_log(&self, line: &LogLine) {
        append_line(&self.dir.join("log.jsonl"), &serde_json::to_string(line).expect("logs serialize"));
    }

    /// A server-side log line (persisted and sent like the trainer's).
    fn log(&self, level: LogLevel, message: impl Into<String>) {
        let line = LogLine { t: now(), level, message: message.into() };
        self.write_log(&line);
        self.send(RunEvent::Log(line));
    }

    /// Moves to a final status outside the trainer (stopped while queued, failed to start, server restart).
    fn finish(&self, st: &mut RunState, status: RunStatus, error: Option<String>) {
        let r = &mut st.file.run;
        r.status = status;
        r.error = error.clone();
        r.eta_s = None;
        r.finished = Some(now());
        r.epoch = st.file.epochs_done;
        r.model_id = model_id_of(&self.id, &self.dir, status).or(r.model_id.take());
        self.persist(st);
        self.send(RunEvent::Status(StatusEvent { status, error }));
    }

    /// The trainer's events: update the run, persist what matters, forward to subscribers.
    fn on_event(&self, ev: &RunEvent) {
        let mut st = self.state.lock().unwrap();
        match ev {
            RunEvent::Status(s) => {
                let r = &mut st.file.run;
                if s.status == RunStatus::Running {
                    if r.started.is_none() {
                        r.started = Some(now());
                    }
                    // A stop requested before training began: the trainer stops right away.
                    if r.status == RunStatus::Stopping {
                        return;
                    }
                    r.status = RunStatus::Running;
                    self.persist(&st);
                } else {
                    st.outcome = Some((s.status, s.error.clone()));
                    return;
                }
            }
            RunEvent::Step(s) => {
                let r = &mut st.file.run;
                r.epoch = s.epoch;
                r.step = s.step;
                r.eta_s = Some(s.eta_s);
                if st.tail.len() < TAIL_STEPS {
                    st.tail.push(*s);
                }
            }
            RunEvent::Epoch(e) => {
                st.tail.clear();
                st.file.epochs_done = e.epoch;
                let v = &e.val;
                st.file.last_metrics = Some(ModelMetrics {
                    map50: v.map50,
                    map50_95: v.map50_95,
                    precision: v.precision,
                    recall: v.recall,
                });
                let r = &mut st.file.run;
                r.epoch = e.epoch;
                if r.best.is_none_or(|b| e.val.map50_95 > b.map50_95) {
                    r.best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
                }
                self.persist(&st);
            }
            RunEvent::Log(l) => self.write_log(l),
        }
        // Sent under the state lock, so subscribers see transitions in the order they happen.
        self.send(ev.clone());
    }
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

struct Queue {
    ids: VecDeque<String>,
    shutdown: bool,
}

/// The registry, the queue and the worker. See the module docs.
pub struct Runs {
    dir: PathBuf,
    list: RwLock<Vec<Arc<RunEntry>>>,
    queue: Mutex<Queue>,
    wake: Condvar,
    /// The run the worker is training.
    active: Mutex<Option<Arc<RunEntry>>>,
    idle: Condvar,
    worker: Mutex<bool>,
}

impl Runs {
    /// Loads the runs under `dir`. Runs a previous process left active become `stopped`.
    pub fn open(dir: PathBuf) -> Arc<Runs> {
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
                let id = file.run.id.clone();
                let e = RunEntry::new(id, d, file);
                {
                    let mut st = e.state.lock().unwrap();
                    if st.file.run.status.is_active() {
                        let was = st.file.run.status;
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
            queue: Mutex::new(Queue { ids: VecDeque::new(), shutdown: false }),
            wake: Condvar::new(),
            active: Mutex::new(None),
            idle: Condvar::new(),
            worker: Mutex::new(false),
        })
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

    /// Every run, newest first.
    pub fn list(&self) -> Vec<Run> {
        self.entries().iter().map(|e| e.run()).collect()
    }

    /// Registers a new run (status `queued`) and queues it. `config` is already validated, its name filled in.
    pub fn create(
        self: &Arc<Self>,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        val_offset: u64,
        device: String,
    ) -> Result<Run> {
        let mut list = self.list.write().unwrap();
        let seed = fnv1a(config.name.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("run", seed, |id| list.iter().any(|e| e.id == id) || self.dir.join(id).exists());
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
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
        };
        let file =
            RunFile { run: run.clone(), dataset, weights, val_offset, epochs_done: 0, last_metrics: None };
        let entry = Arc::new(RunEntry::new(id.clone(), dir, file));
        entry.persist(&entry.state.lock().unwrap());
        list.push(entry.clone());
        drop(list);
        self.enqueue(&entry);
        Ok(run)
    }

    fn enqueue(self: &Arc<Self>, entry: &RunEntry) {
        let ahead = {
            let mut q = self.queue.lock().unwrap();
            q.ids.push_back(entry.id.clone());
            q.ids.len() - 1 + usize::from(self.active.lock().unwrap().is_some())
        };
        entry.log(
            LogLevel::Info,
            if ahead == 0 { "queued".to_string() } else { format!("queued behind {ahead} other run(s)") },
        );
        self.wake.notify_all();
        self.ensure_worker();
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
                let id = loop {
                    if q.shutdown {
                        return;
                    }
                    if let Some(id) = q.ids.pop_front() {
                        break id;
                    }
                    q = self.wake.wait(q).unwrap();
                };
                let Ok(entry) = self.get(&id) else { continue };
                // Set before the queue lock is released, so `enqueue` counts the runs ahead exactly.
                *self.active.lock().unwrap() = Some(entry.clone());
                entry
            };
            // Lock order: a run's state before the queue (as `stop` takes them), so the queue lock is released
            // first. A run stopped or deleted between the two is skipped.
            {
                let mut st = entry.state.lock().unwrap();
                if st.file.run.status != RunStatus::Queued || st.deleted {
                    drop(st);
                    *self.active.lock().unwrap() = None;
                    self.idle.notify_all();
                    continue;
                }
                st.claimed = true;
            }
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&entry)));
            let mut st = entry.state.lock().unwrap();
            let (status, error) = match (ran, st.outcome.take()) {
                (Ok(()), Some(o)) => o,
                (Ok(()), None) => {
                    (RunStatus::Failed, Some("the trainer ended without a final status".into()))
                }
                (Err(_), _) => {
                    let msg = "the trainer panicked (a bug)".to_string();
                    drop(st);
                    entry.log(LogLevel::Error, msg.clone());
                    st = entry.state.lock().unwrap();
                    (RunStatus::Failed, Some(msg))
                }
            };
            st.claimed = false;
            entry.finish(&mut st, status, error);
            drop(st);
            *self.active.lock().unwrap() = None;
            self.idle.notify_all();
        }
    }

    /// Requests a stop: a queued run stops at once, a running one after its current batch (keeping the last
    /// epoch's checkpoint).
    pub fn stop(&self, id: &str) -> Result<Run> {
        let e = self.get(id)?;
        let mut st = e.state.lock().unwrap();
        if st.deleted {
            return Err(Error::not_found("run", id));
        }
        match st.file.run.status {
            RunStatus::Queued if !st.claimed => {
                self.queue.lock().unwrap().ids.retain(|q| q != id);
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
                e.send(RunEvent::Status(StatusEvent { status: RunStatus::Stopping, error: None }));
                let run = st.file.run.clone();
                drop(st);
                e.log(LogLevel::Info, "stop requested; stopping after the current batch");
                Ok(run)
            }
            RunStatus::Stopping => Ok(st.file.run.clone()),
            s => Err(Error::conflict(
                "run_not_active",
                format!("run '{id}' is already {}", serde_json::to_value(s).unwrap().as_str().unwrap_or("")),
            )),
        }
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

    /// Resumes a `stopped` or `failed` run (same id): from `weights/last.folo` when it exists, from scratch
    /// otherwise. `fields` are the request body's other fields: each must equal the run's config (an empty `name`
    /// counts as omitted), except `device`, which may change.
    pub fn resume(
        self: &Arc<Self>,
        id: &str,
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Run> {
        let e = self.get(id)?;
        let mut st = e.state.lock().unwrap();
        if st.deleted {
            return Err(Error::not_found("run", id));
        }
        let status = st.file.run.status;
        if status.is_active() || st.claimed {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active")));
        }
        if status == RunStatus::Completed {
            return Err(Error::conflict("run_not_resumable", format!("run '{id}' already completed"))
                .hint("start a new run with `weights` set to its model to train further"));
        }
        let stored = st.file.run.config.clone();
        let mut merged = serde_json::to_value(&stored).expect("configs serialize");
        for (k, v) in fields {
            if k == "resume_run_id" || (k == "name" && v.as_str() == Some("")) {
                continue;
            }
            merged[k.as_str()] = v.clone();
        }
        let wanted: TrainConfig = serde_json::from_value(merged)
            .map_err(|err| Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_body", err.to_string()))?;
        let (a, b) = (serde_json::to_value(&stored).unwrap(), serde_json::to_value(&wanted).unwrap());
        let differ: Vec<&str> = a
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, v)| k.as_str() != "device" && b.get(k.as_str()) != Some(v))
            .map(|(k, _)| k.as_str())
            .collect();
        if !differ.is_empty() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("resuming keeps the run's config; {} differ(s) from it", differ.join(", ")),
            )
            .hint("omit the fields, or start a new run (with `weights` to start from this run's model)"));
        }
        engine::resolve(&wanted.device)?;
        if !Path::new(&st.file.dataset.path).exists() {
            return Err(Error::not_found("dataset", &st.file.dataset.id)
                .hint(format!("the run's dataset is gone from {}", st.file.dataset.path)));
        }
        let last = e.dir.join("weights").join("last.folo");
        let r = &mut st.file.run;
        r.config.device = wanted.device;
        r.status = RunStatus::Queued;
        r.finished = None;
        r.error = None;
        r.eta_s = None;
        st.tail.clear();
        e.stop.store(false, Ordering::Relaxed);
        e.persist(&st);
        e.send(RunEvent::Status(StatusEvent { status: RunStatus::Queued, error: None }));
        let run = st.file.run.clone();
        drop(st);
        e.log(
            LogLevel::Info,
            if last.is_file() {
                "resuming from weights/last.folo".to_string()
            } else {
                "no epoch finished before; starting over".to_string()
            },
        );
        self.enqueue(&e);
        Ok(run)
    }

    /// Subscribes to a run's events.
    pub fn subscribe(&self, id: &str) -> Result<Subscription> {
        let e = self.get(id)?;
        let events = e.tx.subscribe();
        let r = e.run();
        Ok(Subscription { initial: StatusEvent { status: r.status, error: r.error }, events })
    }

    /// The run's metric history: `metrics.jsonl` (resumed epochs replace the steps and epochs they repeat) plus the
    /// steps not flushed to it yet. `every > 1` keeps every `every`-th step and the latest.
    pub fn metrics(&self, id: &str, every: u64) -> Result<RunMetrics> {
        let e = self.get(id)?;
        let mut m = read_metrics(&e.dir.join("metrics.jsonl"));
        let tail = e.state.lock().unwrap().tail.clone();
        let last = m.steps.last().map_or(0, |s| s.step);
        m.steps.extend(tail.into_iter().filter(|s| s.step > last));
        if every > 1 {
            let latest = m.steps.last().map(|s| s.step);
            m.steps.retain(|s| s.step % every == 0 || Some(s.step) == latest);
        }
        Ok(m)
    }

    /// The last `limit` log lines, oldest first.
    pub fn logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>> {
        let e = self.get(id)?;
        let text = std::fs::read_to_string(e.dir.join("log.jsonl")).unwrap_or_default();
        let lines: Vec<LogLine> = text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        let skip = lines.len().saturating_sub(limit);
        Ok(lines.into_iter().skip(skip).collect())
    }

    /// Stops accepting work, stops the active run (it keeps its last checkpoint) and marks queued runs `stopped`.
    /// With `wait`, blocks until the active run has stopped (at most `wait`).
    pub fn shutdown(&self, wait: Option<Duration>) {
        let queued: Vec<String> = {
            let mut q = self.queue.lock().unwrap();
            q.shutdown = true;
            q.ids.drain(..).collect()
        };
        self.wake.notify_all();
        for id in queued {
            if let Ok(e) = self.get(&id) {
                let mut st = e.state.lock().unwrap();
                if st.file.run.status == RunStatus::Queued {
                    e.finish(&mut st, RunStatus::Stopped, None);
                }
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

/// Parses `metrics.jsonl`. A resumed run appends the steps of the epoch it repeats again: a step or epoch number
/// at or below the last one seen replaces everything from it on.
fn read_metrics(path: &Path) -> RunMetrics {
    let mut m = RunMetrics { steps: Vec::new(), epochs: Vec::new() };
    let Ok(text) = std::fs::read_to_string(path) else { return m };
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        match v.get("event").and_then(|e| e.as_str()) {
            Some("step") => {
                if let Ok(s) = serde_json::from_value::<Step>(v) {
                    while m.steps.last().is_some_and(|l| l.step >= s.step) {
                        m.steps.pop();
                    }
                    m.epochs.retain(|e| e.epoch < s.epoch);
                    m.steps.push(s);
                }
            }
            Some("epoch") => {
                if let Ok(e) = serde_json::from_value::<Epoch>(v) {
                    m.epochs.retain(|x| x.epoch < e.epoch);
                    m.epochs.push(e);
                }
            }
            _ => {}
        }
    }
    m
}

/// Trains one run on the worker thread.
fn execute(e: &Arc<RunEntry>) {
    let (config, dataset, weights) = {
        let st = e.state.lock().unwrap();
        (st.file.run.config.clone(), st.file.dataset.clone(), st.file.weights.clone())
    };
    let trainer = build(e, &config, &dataset, weights);
    let mut trainer = match trainer {
        Ok(t) => t,
        Err(msg) => {
            e.log(LogLevel::Error, msg.clone());
            e.state.lock().unwrap().outcome = Some((RunStatus::Failed, Some(msg)));
            return;
        }
    };
    {
        let mut st = e.state.lock().unwrap();
        st.file.run.steps_per_epoch = trainer.batches_per_epoch() as u64;
        st.file.run.device = engine::describe(trainer.device());
        e.persist(&st);
    }
    // The trainer reports the outcome itself (the final `status` event, the error as a log line); the worker
    // publishes it after dropping the trainer (and with it the run's open files).
    if let Err(err) = trainer.run() {
        tracing::warn!(run = %e.id, "training failed: {err}");
    }
}

fn build(
    e: &Arc<RunEntry>,
    config: &TrainConfig,
    dataset: &DatasetRef,
    weights: Option<PathBuf>,
) -> std::result::Result<Trainer, String> {
    let device = engine::resolve(&config.device).map_err(|err| err.body.message)?;
    let opts = folo_data::OpenOptions { format: data_format(dataset.format), verify_images: true };
    let ds = folo_data::open_with(Path::new(&dataset.path), &opts)
        .map_err(|err| format!("opening the dataset {}: {err}", dataset.path))?;
    if ds.train.is_empty() {
        return Err("the dataset has no training images".into());
    }
    let data = TrainData {
        train: Arc::new(ds.split(Split::Train).clone()),
        val: Arc::new(ds.split(Split::Val).clone()),
        classes: ds.classes.clone(),
    };
    let sink = e.clone();
    let opts = TrainOptions {
        run_dir: Some(e.dir.clone()),
        events: Some(Box::new(move |ev: &RunEvent| sink.on_event(ev))),
        stop: e.stop.clone(),
        weights,
        ..TrainOptions::default()
    };
    let last = e.dir.join("weights").join("last.folo");
    let t = if last.is_file() {
        let overrides = ResumeOverrides { workers: None, device: Some(device.to_string()) };
        Trainer::resume(&last, data, opts, overrides)
    } else {
        let mut cfg = folo_vision::TrainConfig::from_api(config);
        cfg.device = device.to_string();
        cfg.dataset = dataset.id.clone();
        Trainer::new(cfg, data, opts)
    };
    t.map_err(|err| err.to_string())
}

/// folo-data's format of an API format.
pub fn data_format(f: DatasetFormat) -> folo_data::Format {
    match f {
        DatasetFormat::Auto => folo_data::Format::Auto,
        DatasetFormat::Yolo => folo_data::Format::Yolo,
        DatasetFormat::Coco => folo_data::Format::Coco,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resumed_metrics_replace_repeated_steps() {
        let dir = std::env::temp_dir().join(format!("folo-runs-metrics-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("metrics.jsonl");
        let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"imgs_per_s":1,"eta_s":1"#;
        let val = r#""val":{"map50":0.1,"map50_95":0.05,"precision":0.1,"recall":0.1,"per_class":[]}"#;
        let step = |e: u32, s: u64| format!(r#"{{"event":"step","epoch":{e},"step":{s},{loss}}}"#);
        let epoch = |e: u32| {
            format!(
                r#"{{"event":"epoch","epoch":{e},"train_loss":{{"box":1,"cls":1,"dfl":1,"total":3}},{val},"duration_s":1}}"#
            )
        };
        // Epoch 1 (steps 1-2), epoch 2 interrupted after step 3, resumed: steps 3-4 again, epoch 2.
        let lines = [step(1, 1), step(1, 2), epoch(1), step(2, 3), step(2, 3), step(2, 4), epoch(2)];
        std::fs::write(&p, lines.join("\n")).unwrap();
        let m = read_metrics(&p);
        assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [1, 2, 3, 4]);
        assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_model_ids() {
        assert_eq!(run_model_id("run_7d1e", false), "m_7d1e");
        assert_eq!(run_model_id("run_7d1e", true), "m_7d1e-last");
    }
}
