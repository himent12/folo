//! Training runs of the real store: a registry persisted under `<data dir>/runs/<id>/`, a FIFO queue and one
//! worker thread that trains one run at a time with folo-vision's [`Trainer`](folo_vision::Trainer).
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
//!
//! Cloud runs (`config.compute`, docs/CLOUD.md) share the registry, the run folder and the event sink, but not the
//! queue, the worker thread or the device: [`crate::cloud`] claims such a run when it registers it and executes it
//! on a thread of its own through the same [`RunEntry::execute`]. They do not feed the throughput table.
//!
//! Files:
//! - `queue`: creating runs, the FIFO queue and its worker thread (`folo-runs`), stops and shutdown;
//! - `train`: the worker's training of one run (building its [`Trainer`](folo_vision::Trainer), measuring its
//!   throughput);
//! - `entry`: one run: its record, its log, its events (the trainer's sink, [`RunEntry::on_event`]) and how its
//!   execution ends;
//! - `history`: event subscriptions, metrics and logs, and bringing `run.json` up to the files after a crash;
//! - `registry`: loading the registry from disk, looking runs up, listing and deleting them, the device gate of
//!   benchmark probes;
//! - `resume`: resuming a stopped or failed run, here or on a cloud connection;
//! - `tests`: unit tests of the above.

mod entry;
mod history;
mod queue;
pub use queue::CheckpointSnapshot;
mod registry;
mod resume;
mod train;

use std::collections::VecDeque;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use folo_api::{DatasetFormat, ModelMetrics, Run, RunEvent, RunStatus, Step};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::throughput::{Throughput, TrainKey};
use crate::util::{parse_rfc3339, rfc3339, unix_now};

pub(crate) use entry::Ended;
pub use entry::run_model_id;
pub(crate) use queue::followed_elsewhere;

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
    /// Class names (empty in runs created before they were recorded).
    #[serde(default)]
    pub classes: Vec<String>,
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
    /// When the run was last resumed (Unix seconds), while it has not trained since: the wait until it trains, or
    /// until it ends without training (stopped in the queue, the server's stop), is added to `Run.paused_s` then
    /// ([`add_paused`]); until then the API record counts the wait so far ([`RunFile::shown`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resumed_at: Option<f64>,
    /// When the run last reported (Unix seconds): every event of its trainer (step, epoch, log line, status) sets it,
    /// and `run.json` gets it with those that are written anyway and otherwise at most [`ALIVE_EVERY`] apart. A run a
    /// previous process left training ended then, not when the server started again ([`history::reconcile`]); one
    /// killed during a long phase that reports nothing (a long validation pass) ends at its last report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    alive_at: Option<f64>,
    /// Opt-in GPU VRAM-budget auto-batch settings requested when the run was created.
    #[serde(default, skip_serializing_if = "crate::store::AutoBatchOptions::is_default")]
    auto_batch: crate::store::AutoBatchOptions,
}

impl RunFile {
    /// The API record. While the run waits after a resume, `paused_s` includes the wait so far, so the training
    /// time it gives (`Run.paused_s`) stays where it was until the run trains again.
    fn shown(&self) -> Run {
        let mut r = self.run.clone();
        if let Some(at) = self.resumed_at {
            let end = r.finished.as_deref().and_then(parse_rfc3339).unwrap_or_else(|| unix_now().floor());
            add_paused(&mut r, end - at);
        }
        r
    }
}

/// How often a training run's `alive_at` is written to `run.json` while it reports, at most (by the monotonic clock,
/// so a wall clock set back does not hold the writes off).
const ALIVE_EVERY: Duration = Duration::from_secs(10);

/// Adds `secs` (whole seconds, as `started` and `finished` are) to the time run `r` spent not training.
fn add_paused(r: &mut Run, secs: f64) {
    r.paused_s = Some(r.paused_s.unwrap_or(0.0) + secs.max(0.0).round());
}

struct RunState {
    file: RunFile,
    /// Steps since the last `epoch` event (`metrics.jsonl` is flushed at every epoch).
    tail: Vec<Step>,
    /// Incremented on every resume ([`Runs::restart`]) so [`Runs::metrics`] never mixes two attempts.
    attempt: u64,
    /// The worker took the run off the queue (a stop now goes through the trainer).
    claimed: bool,
    /// The trainer's final status, error message and error code, published once the worker has let go of the run
    /// (so a client that sees it may resume or delete the run right away).
    outcome: Option<(RunStatus, Option<String>, Option<String>)>,
    /// Removed from the registry: every later request treats the run as unknown.
    deleted: bool,
    /// The server's shutdown stopped the run (not a client): it ends `stopped` with error code `interrupted`.
    interrupted: bool,
    /// When an event last wrote `run.json` with `alive_at` ([`RunEntry::alive`]).
    alive_written: Option<Instant>,
}

/// One run.
pub struct RunEntry {
    id: String,
    dir: PathBuf,
    state: Mutex<RunState>,
    /// Incremental parser cache of `metrics.jsonl` (never locked while holding `state`).
    metrics_cache: Mutex<history::MetricsCache>,
    tx: broadcast::Sender<RunEvent>,
    /// Pre-serialized SSE frames shared across concurrent HTTP subscribers.
    sse_tx: broadcast::Sender<crate::store::SseFrame>,
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

fn now() -> String {
    rfc3339(unix_now())
}

/// Appends `line` to the file at `path` (a run's `log.jsonl`, or the `metrics.jsonl` a cloud run mirrors); a failure
/// is logged. The line and its newline go in one write, so lines appended by several threads at once never mix.
pub(crate) fn append_line(path: &Path, line: &str) {
    let r = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(format!("{line}\n").as_bytes()));
    if let Err(e) = r {
        tracing::warn!("writing {}: {e}", path.display());
    }
}

/// Ensures an existing non-empty JSONL file ends with `\n`, appending one if a crash left a torn trailing line so
/// subsequent appends (e.g., recovered cloud mirroring or resumed training) never glue onto a partial record.
/// Called only when no writer is concurrently appending to `path`.
pub(crate) fn ensure_trailing_newline(path: &Path) {
    let mut f = match std::fs::OpenOptions::new().read(true).append(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!("opening {} to check trailing newline: {e}", path.display());
            return;
        }
    };
    let len = match f.seek(SeekFrom::End(0)) {
        Ok(0) => return,
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("seeking {} to check trailing newline: {e}", path.display());
            return;
        }
    };
    let mut last = [0u8; 1];
    match f.seek(SeekFrom::Start(len - 1)).and_then(|_| f.read_exact(&mut last)) {
        Ok(()) if last[0] != b'\n' => {
            if let Err(e) = f.write_all(b"\n") {
                tracing::warn!("writing newline to {}: {e}", path.display());
            }
        }
        Ok(()) => {}
        Err(e) => tracing::warn!("reading {} tail: {e}", path.display()),
    }
}

/// Appends a persisted resume boundary marker (`{"event":"resume","epoch":<epoch>}`) to an existing non-empty
/// `metrics.jsonl`, inserting a leading `\n` if the file ends with a torn line that lacks a trailing newline.
/// Called only when the run is inactive (under `RunEntry.state`), so no trainer or cloud mirror is concurrently
/// writing the file.
pub(super) fn mark_resume(path: &Path, epoch: u32) {
    let mut f = match std::fs::OpenOptions::new().read(true).append(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
        Err(e) => {
            tracing::warn!("opening {} for resume marker: {e}", path.display());
            return;
        }
    };
    let len = match f.seek(SeekFrom::End(0)) {
        Ok(0) => return,
        Ok(n) => n,
        Err(e) => {
            tracing::warn!("seeking {} for resume marker: {e}", path.display());
            return;
        }
    };
    let mut last = [0u8; 1];
    let needs_newline = match f.seek(SeekFrom::Start(len - 1)).and_then(|_| f.read_exact(&mut last)) {
        Ok(()) => last[0] != b'\n',
        Err(e) => {
            tracing::warn!("reading {} tail for resume marker: {e}", path.display());
            return;
        }
    };
    let prefix = if needs_newline { "\n" } else { "" };
    let record = format!("{prefix}{{\"event\":\"resume\",\"epoch\":{epoch}}}\n");
    if let Err(e) = f.write_all(record.as_bytes()) {
        tracing::warn!("writing resume marker to {}: {e}", path.display());
    }
}

/// A waiting queue entry tagged with the run's `attempt` counter when it was enqueued, so a worker woken
/// after a stop + resume cannot claim a newer attempt (or a cloud-resumed attempt) of the same run.
struct QueuedRun {
    entry: Arc<RunEntry>,
    attempt: u64,
}

struct Queue {
    /// Entries, not ids: the worker never looks the registry up while it holds the queue.
    runs: VecDeque<QueuedRun>,
    shutdown: bool,
}

/// The log line of a stop accepted while a run trains (`Runs::stop`). The stop is noticed after a batch, so one that
/// comes after the last batch (while the last epoch validates) changes nothing, and the line must hold then too.
pub const STOP_REQUESTED: &str = concat!(
    "stop requested: the run stops after the batch it is on; ",
    "if its last batch is already done, it finishes and ends completed"
);

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

/// The trainer's side of an [`EpochHold`]: a boundary pauses only while the hold is on, so a run nobody holds
/// keeps writing its checkpoint while the next epoch trains.
struct HoldHook(Arc<EpochHold>);

impl folo_vision::train::EpochHook for HoldHook {
    fn pauses(&mut self, _epoch: u32) -> bool {
        *self.0.on.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn pause(&mut self, _epoch: u32, stop: &AtomicBool) {
        self.0.wait(stop);
    }
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
mod tests;
