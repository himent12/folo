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
//!
//! Every validated epoch after a session's first also feeds the throughput table ([`crate::throughput`]): its
//! training rate (the epoch's last `step` event), its validation rate (the rest of the epoch's duration) and the
//! GPU pool's peak memory.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, TryLockError};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    BestEpoch, DatasetFormat, Epoch, LogLevel, LogLine, ModelMetrics, PhaseKind, QueueEvent, Run, RunEvent,
    RunMetrics, RunStatus, StatusEvent, Step, TrainConfig,
};
use folo_data::{Dataset as _, Split};
use folo_vision::train::ResumeOverrides;
use folo_vision::{TrainData, TrainOptions, Trainer};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::engine;
use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::store::{QueueMove, Subscription};
use crate::throughput::{Throughput, TrainKey, TrainSample};
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
    /// The trainer's final status, error message and error code, published once the worker has let go of the run
    /// (so a client that sees it may resume or delete the run right away).
    outcome: Option<(RunStatus, Option<String>, Option<String>)>,
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
    /// What the throughput table learns while the run trains (set by the worker).
    measure: Mutex<Option<Measure>>,
}

/// A training session's contribution to the throughput table: each validated epoch after the session's first
/// (which includes pipeline compilation) is one sample.
struct Measure {
    throughput: Arc<Throughput>,
    key: TrainKey,
    train_images: u64,
    val_images: u64,
    epochs: u32,
    mosaic: bool,
    close_mosaic_epochs: u32,
    /// The training device (its folo-gpu context has the peak memory).
    device: folo_core::Device,
    /// The session's first epoch and its duration (the next epoch's shorter duration gives `startup_s`).
    first: Option<(u32, f64)>,
    /// The latest `step` event (the trainer always sends an epoch's last one).
    last_step: Option<Step>,
}

impl Measure {
    /// Epoch `e` (1-based) trains with mosaic.
    fn mosaic_in(&self, e: u32) -> bool {
        self.mosaic && e + self.close_mosaic_epochs <= self.epochs
    }

    fn epoch(&mut self, e: &Epoch) {
        // Only the epoch's final step covers all of it (steps of non-finite batches are not sent).
        let step = self.last_step.take().filter(|s| {
            s.epoch == e.epoch && s.step == e.epoch as u64 * s.steps_per_epoch && s.imgs_per_s > 0.0
        });
        let Some((first, first_s)) = self.first else {
            self.first = Some((e.epoch, e.duration_s));
            return;
        };
        // `imgs_per_s` of the epoch's last step is its training rate: images over the time since the epoch began.
        // Without it the epoch is not a sample at all.
        let Some(s) = step else { return };
        let ips = s.imgs_per_s as f64;
        let val_s = e.duration_s - self.train_images as f64 / ips;
        let val = (val_s > 0.01).then(|| self.val_images as f64 / val_s);
        let mosaic = self.mosaic_in(e.epoch);
        let startup =
            (first + 1 == e.epoch && mosaic == self.mosaic_in(first)).then_some(first_s - e.duration_s);
        let peak = folo_gpu::context_for(self.device).map(|c| c.pool_stats().peak_live_bytes);
        self.throughput.record_train(TrainSample {
            key: self.key.clone(),
            phase: if mosaic { PhaseKind::Mosaic } else { PhaseKind::CloseMosaic },
            train_imgs_per_s: Some(ips),
            val_imgs_per_s: val,
            startup_s: startup,
            peak_vram_bytes: peak,
            probe: false,
        });
    }
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
            measure: Mutex::new(None),
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
        self.finish_with_code(st, status, error, None);
    }

    /// [`RunEntry::finish`] with the failure's stable error code (`Run.error_code`).
    fn finish_with_code(
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
        r.finished = Some(now());
        r.epoch = st.file.epochs_done;
        r.model_id = model_id_of(&self.id, &self.dir, status).or(r.model_id.take());
        self.persist(st);
        self.send(RunEvent::Status(status_event(&st.file.run)));
    }

    /// The trainer's events: update the run, persist what matters, forward to subscribers.
    fn on_event(&self, ev: &RunEvent) {
        if let Some(m) = self.measure.lock().unwrap().as_mut() {
            match ev {
                RunEvent::Step(s) => m.last_step = Some(*s),
                RunEvent::Epoch(e) => m.epoch(e),
                _ => {}
            }
        }
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
                    st.outcome = Some((s.status, s.error.clone(), s.error_code.clone()));
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
                self.persist(&st);
            }
            RunEvent::Log(l) => self.write_log(l),
            // The trainer never sends these; the queue does (`Runs::notify_queue`).
            RunEvent::Queue(_) => {}
        }
        // Sent under the state lock, so subscribers see transitions in the order they happen.
        self.send(ev.clone());
    }
}

/// The model metrics of an epoch's validation.
fn model_metrics(v: &folo_api::ValMetrics) -> ModelMetrics {
    ModelMetrics {
        map50: v.map50,
        map50_95: v.map50_95,
        precision: v.precision,
        recall: v.recall,
        best_f1_conf: v.best_f1_conf,
    }
}

/// The `status` event of a run's current state, with its length (`steps_per_epoch` is 0 until the trainer is built).
fn status_event(r: &Run) -> StatusEvent {
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

struct Queue {
    /// Entries, not ids: the worker never looks the registry up while it holds the queue.
    runs: VecDeque<Arc<RunEntry>>,
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
    /// Counts the worker taking up and letting go of runs ([`Runs::activity`]).
    activity: AtomicU64,
    /// Held by the worker while it trains a run and by a benchmark probe while it measures ([`Runs::probe_gate`]),
    /// so the two never share the device.
    gate: Mutex<()>,
    throughput: Arc<Throughput>,
    /// Test support ([`Runs::hold_after_epochs`]).
    hold: Arc<EpochHold>,
}

/// Test support: while `on`, a run's trainer waits after each epoch (its checkpoint written and its `epoch` event
/// sent) until the hold is lifted or the run is asked to stop, so a test's stop always lands before the next epoch.
/// After the last epoch there is no next one: a run stopped there still completes, so tests stop runs that have
/// epochs left.
#[derive(Default)]
struct EpochHold {
    on: Mutex<bool>,
    lifted: Condvar,
}

impl EpochHold {
    /// Blocks while the hold is on and `stop` is unset (a stop does not signal the condvar, so it is polled).
    fn wait(&self, stop: &AtomicBool) {
        let mut on = self.on.lock().unwrap_or_else(|p| p.into_inner());
        while *on && !stop.load(Ordering::Relaxed) {
            on = self.lifted.wait_timeout(on, Duration::from_millis(20)).unwrap_or_else(|p| p.into_inner()).0;
        }
    }
}

impl Runs {
    /// Loads the runs under `dir`. Runs a previous process left active become `stopped`. Trained epochs are
    /// recorded in `throughput`.
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
                    if st.file.run.status.is_active() {
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
    fn record(&self, e: &RunEntry) -> Run {
        let st = e.state.lock().unwrap();
        let mut r = st.file.run.clone();
        let q = self.queue.lock().unwrap();
        r.queue_position = q.runs.iter().position(|x| x.id == e.id).map(|i| i as u32 + 1);
        r
    }

    /// The 1-based place of run `id` among the waiting runs.
    fn position(&self, id: &str) -> Option<u32> {
        let q = self.queue.lock().unwrap();
        q.runs.iter().position(|e| e.id == id).map(|i| i as u32 + 1)
    }

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

    /// Registers a new run (status `queued`) and queues it. `config` is already validated, its name filled in.
    pub fn create(
        self: &Arc<Self>,
        config: TrainConfig,
        dataset: DatasetRef,
        weights: Option<PathBuf>,
        val_offset: u64,
        device: String,
    ) -> Result<Run> {
        self.check_open()?;
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
            error_code: None,
            resolved: None,
            queue_position: None,
        };
        let file =
            RunFile { run: run.clone(), dataset, weights, val_offset, epochs_done: 0, last_metrics: None };
        let entry = Arc::new(RunEntry::new(id.clone(), dir, file));
        entry.persist(&entry.state.lock().unwrap());
        list.push(entry.clone());
        drop(list);
        self.enqueue(&entry)?;
        Ok(Run { queue_position: self.position(&run.id), ..run })
    }

    /// `503 shutting_down` once [`Runs::shutdown`] has started.
    fn check_open(&self) -> Result<()> {
        if self.queue.lock().unwrap().shutdown { Err(shutting_down()) } else { Ok(()) }
    }

    /// Queues `entry`. After [`Runs::shutdown`] has started (checked under the queue lock, so a run is either
    /// queued before the shutdown drains the queue or not at all) the run is marked `stopped` instead and the
    /// result is `503 shutting_down`.
    fn enqueue(self: &Arc<Self>, entry: &Arc<RunEntry>) -> Result<()> {
        let ahead = {
            let mut q = self.queue.lock().unwrap();
            if q.shutdown {
                drop(q);
                let mut st = entry.state.lock().unwrap();
                if !st.deleted {
                    entry.finish(&mut st, RunStatus::Stopped, None);
                }
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
            {
                let mut st = entry.state.lock().unwrap();
                if st.file.run.status != RunStatus::Queued || st.deleted {
                    drop(st);
                    *self.active.lock().unwrap() = None;
                    self.activity.fetch_add(1, Ordering::SeqCst);
                    self.idle.notify_all();
                    continue;
                }
                st.claimed = true;
            }
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                execute(&entry, &self.throughput, &self.hold)
            }));
            *entry.measure.lock().unwrap() = None;
            let mut st = entry.state.lock().unwrap();
            let internal = || Some("internal".to_string());
            let (status, error, code) = match (ran, st.outcome.take()) {
                (Ok(()), Some(o)) => o,
                (Ok(()), None) => {
                    (RunStatus::Failed, Some("the trainer ended without a final status".into()), internal())
                }
                (Err(_), _) => {
                    let msg = "the trainer panicked (a bug)".to_string();
                    drop(st);
                    entry.log(LogLevel::Error, msg.clone());
                    st = entry.state.lock().unwrap();
                    (RunStatus::Failed, Some(msg), internal())
                }
            };
            st.claimed = false;
            entry.finish_with_code(&mut st, status, error, code);
            drop(st);
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
                e.log(LogLevel::Info, "stop requested; stopping after the current batch");
                Ok(run)
            }
            RunStatus::Stopping => Ok(st.file.run.clone()),
            s => Err(Error::run_not_active(id, serde_json::to_value(s).unwrap().as_str().unwrap_or(""))),
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
        self.check_open()?;
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
        let device = engine::resolve(&wanted.device)?;
        // A mixed-precision run resumes in mixed precision (`amp` must match): refused before it is queued where this
        // server cannot train it, as a new run is.
        if wanted.amp
            && let Err(err) = folo_vision::train::amp_available(device)
        {
            let mut err = engine::engine_error(err);
            err.body.hint = Some(
                "the run trained in mixed precision and resumes in it: resume it on a server with tensor cores \
                 (`folo serve --tensor-cores` / `folo mcp --tensor-cores` on a GPU that has them), or start a new \
                 f32 run with `weights` set to its model"
                    .into(),
            );
            return Err(err);
        }
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
        r.error_code = None;
        r.eta_s = None;
        st.tail.clear();
        e.stop.store(false, Ordering::Relaxed);
        e.persist(&st);
        e.send(RunEvent::Status(status_event(&st.file.run)));
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
        self.enqueue(&e)?;
        Ok(Run { queue_position: self.position(id), ..run })
    }

    /// Subscribes to a run's events.
    pub fn subscribe(&self, id: &str) -> Result<Subscription> {
        let e = self.get(id)?;
        // Under the state lock (which every status change holds while it sends), so the initial status and the
        // first event received are in order.
        let st = e.state.lock().unwrap();
        let events = e.tx.subscribe();
        Ok(Subscription { initial: status_event(&st.file.run), events })
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

fn shutting_down() -> Error {
    Error::new(StatusCode::SERVICE_UNAVAILABLE, "shutting_down", "the server is shutting down")
        .hint("start the run again once the server has restarted")
}

/// Brings the progress in `run.json` of a run a previous process left active up to its files: the trainer writes
/// `weights/last.folo`, then the epoch's line in `metrics.jsonl`, then (through its event) `run.json`, so a crash in
/// between leaves `run.json` behind. Epochs done and the best epoch come from the newest of the three; the last
/// epoch's metrics only from `metrics.jsonl` (unknown when only the checkpoint has that epoch).
fn reconcile(file: &mut RunFile, dir: &Path) {
    let m = read_metrics(&dir.join("metrics.jsonl"));
    let ckpt = crate::engine::read_model_file(&dir.join("weights").join("last.folo"))
        .ok()
        .and_then(|f| f.checkpoint().map(|c| (c.epoch, c.best)));
    let from_metrics = m.epochs.last().map_or(0, |e| e.epoch);
    let from_ckpt = ckpt.map_or(0, |(e, _)| e.min(u32::MAX as u64) as u32);
    let done = file.epochs_done.max(from_metrics).max(from_ckpt);
    if done == file.epochs_done {
        return;
    }
    let mut best = file.run.best;
    for e in &m.epochs {
        if best.is_none_or(|b| e.val.map50_95 > b.map50_95) {
            best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
        }
    }
    if let Some((_, b)) = ckpt
        && let Some(f) = b.fitness
        && best.is_none_or(|x| (f as f32) > x.map50_95)
    {
        best = Some(BestEpoch { epoch: (b.epoch + 1).max(1) as u32, map50_95: f as f32 });
    }
    file.run.best = best;
    file.last_metrics = m.epochs.last().filter(|e| e.epoch == done).map(|e| model_metrics(&e.val));
    file.epochs_done = done;
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
fn execute(e: &Arc<RunEntry>, throughput: &Arc<Throughput>, hold: &Arc<EpochHold>) {
    let (config, dataset, weights) = {
        let st = e.state.lock().unwrap();
        (st.file.run.config.clone(), st.file.dataset.clone(), st.file.weights.clone())
    };
    let trainer = build(e, &config, &dataset, weights, hold);
    let mut trainer = match trainer {
        Ok(t) => t,
        Err((code, msg)) => {
            // A trainer that failed after placing its model (weights that do not load, a dataset error after the
            // model was built) left its buffers in the pool too.
            if let Ok(device) = engine::resolve(&config.device) {
                engine::release_gpu_memory(device);
            }
            e.log(LogLevel::Error, msg.clone());
            e.state.lock().unwrap().outcome = Some((RunStatus::Failed, Some(msg), Some(code)));
            return;
        }
    };
    {
        let mut st = e.state.lock().unwrap();
        st.file.run.steps_per_epoch = trainer.batches_per_epoch() as u64;
        st.file.run.device = engine::describe(trainer.device());
        st.file.run.resolved = Some(trainer.resolved_train());
        e.persist(&st);
    }
    let device = trainer.device();
    if let Some(c) = folo_gpu::context_for(device) {
        c.reset_peak_memory();
    }
    let rc = trainer.run_config();
    let count = |k: &str| rc[k].as_u64().unwrap_or(0);
    let (train_images, val_images) = (count("train_images"), count("val_images"));
    *e.measure.lock().unwrap() = Some(Measure {
        throughput: throughput.clone(),
        key: TrainKey {
            device: engine::device_id(device),
            arch: config.arch,
            imgsz: config.imgsz,
            batch: config.batch,
            amp: config.amp,
        },
        train_images,
        // An empty val split validates the training split.
        val_images: if val_images == 0 { train_images } else { val_images },
        epochs: config.epochs,
        mosaic: config.augment.mosaic > 0.0,
        close_mosaic_epochs: count("close_mosaic_epochs") as u32,
        device,
        first: None,
        last_step: None,
    });
    // The trainer reports the outcome itself (the final `status` event, the error as a log line); the worker
    // publishes it after dropping the trainer (and with it the run's open files).
    if let Err(err) = trainer.run() {
        tracing::warn!(run = %e.id, "training failed: {err}");
    }
    drop(trainer);
    // The run's model, optimizer state, EMA and activations are back in the pool now; hand them to the driver
    // before the run is reported finished, so the next job on this GPU (a CLI run, another process) has its memory.
    engine::release_gpu_memory(device);
}

fn build(
    e: &Arc<RunEntry>,
    config: &TrainConfig,
    dataset: &DatasetRef,
    weights: Option<PathBuf>,
    hold: &Arc<EpochHold>,
) -> std::result::Result<Trainer, (String, String)> {
    let device =
        engine::resolve(&config.device).map_err(|err| (err.body.code.to_string(), err.body.message))?;
    let opts = folo_data::OpenOptions { format: data_format(dataset.format), verify_images: true };
    let format = opts.format;
    let ds = folo_data::open_with(Path::new(&dataset.path), &opts)
        .map_err(|err| ("data_error".to_string(), format!("opening the dataset {}: {err}", dataset.path)))?;
    if ds.train.is_empty() {
        return Err(("invalid_dataset".into(), "the dataset has no training images".into()));
    }
    let data = TrainData {
        train: Arc::new(ds.split(Split::Train).clone()),
        val: Arc::new(ds.split(Split::Val).clone()),
        classes: ds.classes.clone(),
    };
    let (sink, hold, stop) = (e.clone(), hold.clone(), e.stop.clone());
    let opts = TrainOptions {
        run_dir: Some(e.dir.clone()),
        events: Some(Box::new(move |ev: &RunEvent| {
            sink.on_event(ev);
            if matches!(ev, RunEvent::Epoch(_)) {
                hold.wait(&stop);
            }
        })),
        stop: e.stop.clone(),
        weights,
        ..TrainOptions::default()
    };
    let last = e.dir.join("weights").join("last.folo");
    let t = if last.is_file() {
        let overrides =
            ResumeOverrides { workers: None, device: Some(device.to_string()), data_format: Some(format) };
        Trainer::resume(&last, data, opts, overrides)
    } else {
        let mut cfg = folo_vision::TrainConfig::from_api(config);
        cfg.device = device.to_string();
        cfg.dataset = dataset.id.clone();
        cfg.data_format = format;
        Trainer::new(cfg, data, opts)
    };
    t.map_err(|err| (err.code().to_string(), err.summary()))
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

    /// A create or resume that passed `check_open` before the shutdown and reaches `enqueue` after it: the run is
    /// marked stopped instead of queued, and no worker starts.
    #[test]
    fn enqueue_after_shutdown_refuses_and_stops_the_run() {
        let dir = std::env::temp_dir().join(format!("folo-runs-late-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let runs = Runs::open(dir.clone(), Arc::new(Throughput::in_memory(Default::default())));
        let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"late"}"#).unwrap();
        let run = Run {
            id: "run_late".into(),
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
            device: "cpu".into(),
            model_id: None,
            error: None,
            error_code: None,
            resolved: None,
            queue_position: None,
        };
        let dataset = DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: "nowhere".into(),
            format: DatasetFormat::Yolo,
        };
        let file = RunFile { run, dataset, weights: None, val_offset: 0, epochs_done: 0, last_metrics: None };
        std::fs::create_dir_all(dir.join("run_late")).unwrap();
        let entry = Arc::new(RunEntry::new("run_late".into(), dir.join("run_late"), file));
        runs.list.write().unwrap().push(entry.clone());

        runs.shutdown(None);
        let e = runs.enqueue(&entry).unwrap_err();
        assert_eq!((e.status, e.body.code.as_str()), (StatusCode::SERVICE_UNAVAILABLE, "shutting_down"));
        assert_eq!(entry.run().status, RunStatus::Stopped);
        assert!(runs.queue.lock().unwrap().runs.is_empty());
        assert!(!*runs.worker.lock().unwrap(), "no worker for a refused run");
        assert!(runs.logs("run_late", 10).unwrap().iter().any(|l| l.message.contains("not queued")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `list` reads one queue: positions stay unique and gapless while another thread keeps moving runs.
    #[test]
    fn listed_queue_positions_are_one_snapshot() {
        let dir = std::env::temp_dir().join(format!("folo-runs-list-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let runs = Runs::open(dir.clone(), Arc::new(Throughput::in_memory(Default::default())));
        let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"q"}"#).unwrap();
        let dataset = DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: "nowhere".into(),
            format: DatasetFormat::Yolo,
        };
        let ids: Vec<String> = (0..4).map(|i| format!("run_q{i}")).collect();
        for id in &ids {
            let run = Run {
                id: id.clone(),
                name: "q".into(),
                status: RunStatus::Queued,
                epochs: config.epochs,
                config: config.clone(),
                created: now(),
                started: None,
                finished: None,
                epoch: 0,
                step: 0,
                steps_per_epoch: 0,
                eta_s: None,
                best: None,
                device: "cpu".into(),
                model_id: None,
                error: None,
                error_code: None,
                resolved: None,
                queue_position: None,
            };
            let file = RunFile {
                run,
                dataset: dataset.clone(),
                weights: None,
                val_offset: 0,
                epochs_done: 0,
                last_metrics: None,
            };
            std::fs::create_dir_all(dir.join(id)).unwrap();
            let entry = Arc::new(RunEntry::new(id.clone(), dir.join(id), file));
            runs.list.write().unwrap().push(entry.clone());
            // Queued without a worker, so nothing leaves the queue.
            runs.queue.lock().unwrap().runs.push_back(entry);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mover = {
            let (runs, ids, stop) = (runs.clone(), ids.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut i = 0;
                while !stop.load(Ordering::Relaxed) {
                    let mv = if i % 2 == 0 { QueueMove::Top } else { QueueMove::Bottom };
                    runs.move_queued(&ids[i % ids.len()], mv).unwrap();
                    i += 1;
                }
            })
        };
        for _ in 0..300 {
            let mut places: Vec<u32> = runs.list().iter().filter_map(|r| r.queue_position).collect();
            places.sort_unstable();
            assert_eq!(places, [1, 2, 3, 4]);
        }
        stop.store(true, Ordering::Relaxed);
        mover.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The throughput samples of a run: the session's first epoch is warm-up, an epoch whose last step was not
    /// sent (a non-finite final batch) is no sample, and the rest go to their phase with a validation rate from the
    /// remainder of the epoch.
    #[test]
    fn only_complete_epochs_after_the_first_are_samples() {
        let tp = Arc::new(Throughput::in_memory(Default::default()));
        let device = crate::throughput::DeviceId { device: "cpu".into(), device_name: "a cpu".into() };
        let key = TrainKey { device, arch: folo_api::Arch::N, imgsz: 64, batch: 4, amp: false };
        let mut m = Measure {
            throughput: tp.clone(),
            key,
            train_images: 10,
            val_images: 2,
            epochs: 4,
            mosaic: true,
            close_mosaic_epochs: 1,
            device: folo_core::Device::Cpu,
            first: None,
            last_step: None,
        };
        let loss = r#""loss":{"box":1,"cls":1,"dfl":1,"total":3},"lr":0.01,"eta_s":1"#;
        let step = |e: u32, s: u64, ips: f32| -> Step {
            serde_json::from_str(&format!(
                r#"{{"epoch":{e},"step":{s},"imgs_per_s":{ips},"steps_per_epoch":3,"epochs":4,{loss}}}"#
            ))
            .unwrap()
        };
        let val = r#""val":{"map50":0.1,"map50_95":0.05,"precision":0.1,"recall":0.1,"per_class":[]}"#;
        let epoch = |e: u32, d: f64| -> Epoch {
            serde_json::from_str(&format!(
                r#"{{"epoch":{e},"train_loss":{{"box":1,"cls":1,"dfl":1,"total":3}},{val},"duration_s":{d}}}"#
            ))
            .unwrap()
        };
        for (e, s, ips, d) in [(1, 3, 5.0, 4.0), (2, 5, 7.0, 1.5), (3, 9, 10.0, 1.2), (4, 12, 20.0, 0.7)] {
            m.last_step = Some(step(e, s, ips));
            m.epoch(&epoch(e, d));
        }
        let rows = tp.file().training;
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        // Epoch 1: warm-up; epoch 2 ended on step 5 of 6: skipped; epoch 3 mosaic, epoch 4 close mosaic.
        assert_eq!(
            (r.mosaic.values.as_slice(), r.close_mosaic.values.as_slice()),
            (&[10.0][..], &[20.0][..])
        );
        // Validation: 2 images in 1.2 - 10 / 10 = 0.2 s and 0.7 - 10 / 20 = 0.2 s.
        assert_eq!(r.val.values.len(), 2);
        assert!(r.val.values.iter().all(|v| (v - 10.0).abs() < 1e-6), "{:?}", r.val.values);
        assert!(r.startup.is_empty(), "epoch 2 was no sample, so no startup");
    }

    #[test]
    fn run_model_ids() {
        assert_eq!(run_model_id("run_7d1e", false), "m_7d1e");
        assert_eq!(run_model_id("run_7d1e", true), "m_7d1e-last");
    }
}
