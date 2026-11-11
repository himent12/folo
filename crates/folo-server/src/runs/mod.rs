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

mod entry;
mod history;
mod queue;
mod registry;
mod resume;
mod train;

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use folo_api::{DatasetFormat, ModelMetrics, Run, RunEvent, RunStatus, Step};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::throughput::{Throughput, TrainKey};
use crate::util::{rfc3339, unix_now};

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
    /// The server's shutdown stopped the run (not a client): it ends `stopped` with error code `interrupted`.
    interrupted: bool,
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
