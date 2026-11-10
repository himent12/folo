//! Vision benchmarks (docs/API.md, *Vision benchmarks*; docs/BENCH.md defines every number): frozen sets of
//! labeled images with a plan of stress conditions, the entries that ran them and self-improvement loops.
//!
//! [`Benches`] is the service behind `/api/bench`: it keeps the benchmarks of a data dir (`bench/<id>/`), makes
//! new ones from a dataset split or a labeling job, serves their images and variants, runs entries and loops and
//! sends their events. The engine is `folo-bench` (conditions, plans, every number of a score); this module is
//! storage and scheduling:
//!
//! - `state`: a benchmark's files (`bench.json`, `images/`, `entries/<id>.json`, `loops/<id>.json`, all written
//!   atomically), its state in memory and its event channel;
//! - `source`: where images and truth come from; `images`: serving images and variants;
//! - `model`: the model-entry queue (one worker thread) and the predictor pipeline; `agent`: agent entries (a
//!   thread each, as labeling runs); `external`: predictions made elsewhere, in one request or in parts (open
//!   entries, which expire: a sweeper thread); `results`: grading, the same for every contestant;
//! - `design`: designing a plan with an agent; `loops`: self-improvement loops (a thread each, driving the store).

mod agent;
mod benchmarks;
mod design;
mod entries;
mod external;
pub mod http;
mod images;
mod loops;
mod model;
mod results;
mod source;
mod state;

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock, Weak};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{Bench, BenchCatalog, BenchEvent, BenchLoop, BenchLoopStatus, BenchRemoved, StartBenchLoop};
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;

pub use model::InferenceGuard;

use crate::error::{Error, Result};
use crate::store::Store;
use crate::util::{parse_rfc3339_millis, rfc3339, unix_now};
use entries::{add_entry, enqueue_model, new_entry, status_name, stop_entry_in};
use model::Engine;
use state::{Slot, bench_not_found};

/// Most images of a benchmark.
pub const MAX_IMAGES: u32 = 5000;
/// Images of a benchmark made without `max_images`.
pub const DEFAULT_IMAGES: u32 = 500;
/// Most classes of a benchmark.
pub const MAX_CLASSES: usize = 80;
/// Longest description, in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 4000;
/// Longest note of a design request, in characters.
pub const MAX_NOTE_CHARS: usize = 2000;
/// Images an agent entry runs without `max_images`.
pub const AGENT_IMAGES: u32 = 48;
/// The largest seed (of a benchmark, of a loop's students): 2^53 − 1, the largest integer a browser's numbers
/// hold exactly.
pub const MAX_SEED: u64 = (1 << 53) - 1;

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

fn io_error(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::internal(format!("{what} {}: {e}", path.display()))
}

/// The benchmarks of a data dir, their entries and loops. See the module docs.
pub struct Benches {
    /// `<data dir>/bench`.
    root: PathBuf,
    /// `<data dir>/cache/thumbs`.
    thumbs: PathBuf,
    /// `<data dir>/datasets`, where loops write their students' datasets.
    datasets: PathBuf,
    /// In creation order (shared with the sweeper of open entries).
    benches: Arc<RwLock<Vec<Arc<Slot>>>>,
    engine: Arc<Engine>,
    /// The model worker's thread and the sweeper's (open external entries that expire).
    worker: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Agent entries' and loops' threads, for shutdown to wait on.
    threads: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>>,
    renders: images::Renders,
}

/// SHA-256 of `bytes`, hex.
fn sha256(bytes: &[u8]) -> String {
    state::hex(&Sha256::digest(bytes))
}

impl Benches {
    /// Opens the benchmarks under `data_dir` (a benchmark whose file cannot be read is left out and logged, never
    /// overwritten; entries and loops a previous process left active become `failed`) and starts the model
    /// worker. `guard` marks inference for the store's GPU idle tracking.
    pub fn open(data_dir: &Path, guard: Option<InferenceGuard>) -> Benches {
        let root = data_dir.join("bench");
        // A folder without `bench.json` is a benchmark whose making a crash cut short: its copied images are
        // nobody's.
        for e in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            if e.path().is_dir() && !e.path().join("bench.json").exists() {
                tracing::warn!("removing {}: a benchmark that was never finished", e.path().display());
                let _ = std::fs::remove_dir_all(e.path());
            }
        }
        let mut benches: Vec<Arc<Slot>> = std::fs::read_dir(&root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("bench.json").is_file())
            .filter_map(|e| match Slot::load(e.path()) {
                Ok(slot) => {
                    let _ = std::fs::remove_dir_all(e.path().join("tmp"));
                    Some(Arc::new(slot))
                }
                Err(err) => {
                    tracing::warn!("skipping benchmark {}: {err}", e.path().display());
                    None
                }
            })
            .collect();
        // By time, not by text: one made before `created` had milliseconds has none.
        benches.sort_by_cached_key(|b| (parse_rfc3339_millis(&b.lock().file.created), b.id.clone()));
        let engine = Arc::new(Engine::new("auto".into(), guard));
        let benches = Arc::new(RwLock::new(benches));
        let worker = {
            let engine = engine.clone();
            std::thread::Builder::new().name("folo-bench-models".into()).spawn(move || model::work(engine))
        };
        let sweeper = {
            let (engine, benches) = (engine.clone(), benches.clone());
            std::thread::Builder::new()
                .name("folo-bench-expiry".into())
                .spawn(move || sweep(&engine, &benches))
        };
        let threads = [worker, sweeper]
            .into_iter()
            .filter_map(|t| {
                t.inspect_err(|e| tracing::error!("could not start a benchmark thread: {e}")).ok()
            })
            .collect();
        Benches {
            root,
            thumbs: data_dir.join("cache").join("thumbs"),
            datasets: data_dir.join("datasets"),
            benches,
            engine,
            worker: std::sync::Mutex::new(threads),
            threads: std::sync::Mutex::default(),
            renders: images::Renders::default(),
        }
    }

    /// The device model entries run on (the store's inference device).
    pub fn set_device(&self, device: &str) {
        self.engine.set_device(device.to_string());
    }

    /// How long an open external entry waits for its next part before it expires (default 30 minutes).
    pub fn set_open_timeout(&self, timeout: Duration) {
        self.engine.open_timeout_ms.store(timeout.as_millis() as u64, Ordering::SeqCst);
    }

    /// Test support: while `on`, a model entry waits after its last pass, before it is graded.
    #[doc(hidden)]
    pub fn hold_entries(&self, on: bool) {
        self.engine.hold.store(on, Ordering::SeqCst);
    }

    fn slot(&self, id: &str) -> Result<Arc<Slot>> {
        let list = self.benches.read().unwrap();
        list.iter().find(|b| b.id == id).cloned().ok_or_else(|| bench_not_found(id))
    }

    /// Keeps a thread for shutdown to wait on (finished ones are let go).
    fn keep(&self, thread: std::thread::JoinHandle<()>) {
        let mut threads = self.threads.lock().unwrap();
        threads.retain(|t| !t.is_finished());
        threads.push(thread);
    }

    /// `GET /api/bench/catalog`.
    pub fn catalog(&self) -> BenchCatalog {
        folo_bench::catalog()
    }

    /// `GET /api/bench/benchmarks`: newest first.
    pub fn list(&self) -> Vec<Bench> {
        self.benches.read().unwrap().iter().rev().map(|b| b.lock().bench()).collect()
    }

    /// `GET /api/bench/benchmarks/{id}`.
    pub fn get(&self, id: &str) -> Result<Bench> {
        Ok(self.slot(id)?.live()?.bench())
    }

    /// `GET /api/bench/benchmarks/{id}/loops`: newest first.
    pub fn loops(&self, id: &str) -> Result<Vec<BenchLoop>> {
        Ok(self.slot(id)?.live()?.loops.iter().rev().map(|l| l.lp.clone()).collect())
    }

    /// `GET …/loops/{loop_id}`.
    pub fn get_loop(&self, id: &str, loop_id: &str) -> Result<BenchLoop> {
        Ok(self.slot(id)?.live()?.lp(loop_id)?.lp.clone())
    }

    /// `POST /api/bench/benchmarks/{id}/loops`: checks the request ([`loops::check`]), adds the loop and starts its
    /// thread. `weak` is the store the loop drives, held weakly (see the `loops` module).
    pub fn start_loop(
        &self,
        store: &dyn Store,
        weak: Weak<dyn Store>,
        id: &str,
        req: StartBenchLoop,
    ) -> Result<BenchLoop> {
        let slot = self.slot(id)?;
        let checked = loops::check(store, &slot, req)?;
        let (lp, cancel) = loops::begin(&slot, &checked)?;
        let job = loops::LoopJob {
            slot: slot.clone(),
            id: lp.id.clone(),
            engine: self.engine.clone(),
            store: weak,
            cancel,
            config: checked.config,
            teacher: checked.teacher,
            pool: checked.pool,
            excluded: checked.excluded,
            datasets: self.datasets.clone(),
        };
        let thread = std::thread::Builder::new()
            .name(format!("folo-bench-loop-{}", lp.id))
            .spawn(move || loops::run(job));
        match thread {
            Ok(t) => self.keep(t),
            Err(e) => {
                let message = format!("could not start the loop's thread: {e}");
                slot.update_loop(&lp.id, |l| {
                    l.status = BenchLoopStatus::Failed;
                    l.phase = None;
                    l.finished = Some(rfc3339(unix_now()));
                    l.error = Some(message.clone());
                });
                return Err(Error::internal(message));
            }
        }
        Ok(lp)
    }

    /// `POST …/loops/{loop_id}/stop`: the loop stops what it is doing (its entry, its run) and ends `stopped`.
    pub fn stop_loop(&self, id: &str, loop_id: &str) -> Result<BenchLoop> {
        let slot = self.slot(id)?;
        let mut state = slot.live()?;
        let bench = state.file.id.clone();
        let l = state.lp_mut(loop_id).ok_or_else(|| state::loop_not_found(&bench, loop_id))?;
        match l.lp.status {
            BenchLoopStatus::Running => {
                l.lp.status = BenchLoopStatus::Stopping;
                l.cancel.store(true, Ordering::Relaxed);
                let lp = l.lp.clone();
                slot.save_loop(&state, &lp)?;
                slot.emit(BenchEvent::Loop(Box::new(lp.clone())));
                Ok(lp)
            }
            BenchLoopStatus::Stopping => Ok(l.lp.clone()),
            _ => Err(Error::conflict("bench_loop_not_active", format!("loop '{loop_id}' has already ended"))
                .hint("nothing to stop; GET …/loops/{loop_id} has how it came out")),
        }
    }

    /// `DELETE …/loops/{loop_id}` (its datasets, runs, models and entries stay).
    pub fn delete_loop(&self, id: &str, loop_id: &str) -> Result<()> {
        let slot = self.slot(id)?;
        let mut state = slot.live()?;
        if state.lp(loop_id)?.lp.status.is_active() {
            return Err(Error::conflict("bench_loop_active", format!("loop '{loop_id}' is still running"))
                .hint("stop it first: POST /api/bench/benchmarks/{id}/loops/{loop_id}/stop"));
        }
        slot.remove_loop(loop_id)?;
        state.loops.retain(|l| l.lp.id != loop_id);
        slot.emit(BenchEvent::Removed(BenchRemoved { id: loop_id.to_string() }));
        Ok(())
    }

    /// `GET /api/bench/benchmarks/{id}/events`: the benchmark now, its events from now on, and a weak handle that
    /// gives the benchmark again to a subscriber that fell behind (without keeping the stream open after the
    /// benchmark is deleted).
    pub(crate) fn subscribe(&self, id: &str) -> Result<(Bench, broadcast::Receiver<BenchEvent>, Weak<Slot>)> {
        let slot = self.slot(id)?;
        let state = slot.live()?;
        Ok((state.bench(), slot.subscribe(), Arc::downgrade(&slot)))
    }

    /// The benchmark behind a [`Benches::subscribe`] handle, while it exists.
    pub(crate) fn current(slot: &Weak<Slot>) -> Option<Bench> {
        let slot = slot.upgrade()?;
        let state = slot.lock();
        (!state.deleted).then(|| state.bench())
    }

    /// Ends every entry and loop (they become `failed`, "the server stopped"; the agent calls in flight are
    /// killed) and waits for their threads.
    pub fn shutdown(&self) {
        self.engine.closing.store(true, Ordering::SeqCst);
        self.engine.wake();
        let slots = self.benches.read().unwrap().clone();
        for slot in &slots {
            // Open external entries had their predictions in memory: they fail as the others do.
            let open: Vec<String> = {
                let mut state = slot.lock();
                let mut ids = Vec::new();
                for e in state.entries.iter_mut() {
                    if let Some(fill) = e.open.take() {
                        fill.lock().unwrap_or_else(std::sync::PoisonError::into_inner).closed = true;
                        ids.push(e.entry.id.clone());
                    }
                }
                ids
            };
            for id in open {
                model::finish_entry(slot, &id, true, Ok(None));
            }
            let state = slot.lock();
            state
                .entries
                .iter()
                .filter(|e| e.entry.status.is_active())
                .for_each(|e| e.cancel.store(true, Ordering::SeqCst));
            state
                .loops
                .iter()
                .filter(|l| l.lp.status.is_active())
                .for_each(|l| l.cancel.store(true, Ordering::SeqCst));
        }
        for job in self.engine.drain() {
            model::finish_entry(&job.slot, &job.entry, true, Ok(None));
        }
        // Not the current thread: the store can be dropped on a loop's thread, which then runs this.
        let me = std::thread::current().id();
        let threads = std::mem::take(&mut *self.threads.lock().unwrap());
        for t in threads.into_iter().filter(|t| t.thread().id() != me) {
            let _ = t.join();
        }
        let engine_threads = std::mem::take(&mut *self.worker.lock().unwrap());
        for t in engine_threads.into_iter().filter(|t| t.thread().id() != me) {
            // The sweeper parks between looks: woken, it sees `closing` at once.
            t.thread().unpark();
            let _ = t.join();
        }
    }
}

/// The sweeper's thread: every so often, open external entries that got no part for the engine's timeout are
/// failed ([`external::expire`]). Ends when the server shuts down (it looks every 50 ms, and [`Benches::shutdown`]
/// unparks it).
fn sweep(engine: &Engine, benches: &RwLock<Vec<Arc<Slot>>>) {
    let mut last = Instant::now();
    while !engine.closing.load(Ordering::SeqCst) {
        std::thread::park_timeout(Duration::from_millis(50));
        if engine.closing.load(Ordering::SeqCst) {
            break;
        }
        let timeout = Duration::from_millis(engine.open_timeout_ms.load(Ordering::SeqCst));
        // A tenth of the timeout between looks: an entry expires at most 10% late.
        if last.elapsed() < (timeout / 10).clamp(Duration::from_millis(50), Duration::from_secs(60)) {
            continue;
        }
        last = Instant::now();
        let slots = benches.read().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        for slot in slots {
            external::expire(&slot, timeout);
        }
    }
}
