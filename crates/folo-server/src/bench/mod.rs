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
mod design;
mod external;
pub mod http;
mod images;
mod loops;
mod model;
mod results;
mod source;
mod state;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    Bench, BenchCatalog, BenchContestant, BenchContestantSpec, BenchDesign, BenchEntry, BenchEntryImage,
    BenchEntryImagePage, BenchEntryStatus, BenchEvent, BenchImage, BenchImagePage, BenchLoop,
    BenchLoopStatus, BenchRemoved, BenchResult, BenchSource, CreateBench, DesignBench, LabelUsage, Split,
    StartBenchEntry, StartBenchLoop, UpdateBench,
};
use folo_bench::{Condition, conditions};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;

pub use model::InferenceGuard;

use crate::error::{Error, Result};
use crate::label::job::unique_file_name;
use crate::label::{check_model, check_name, check_text};
use crate::store::{ImageData, Store};
use crate::util::{fnv1a, parse_rfc3339_millis, rfc3339, rfc3339_millis, unique_id, unix_now};
use model::{Engine, Job, Resolved};
use state::{BenchFile, EntryState, ImageRecord, Slot, State, VERSION, bench_not_found};

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

/// Starts a model entry of `resolved` on `slot`: `queued` until the worker takes it. `name` defaults to the
/// model's, `max_images` to every image.
fn enqueue_model(
    engine: &Engine,
    slot: &Arc<Slot>,
    resolved: Resolved,
    name: Option<String>,
    max_images: Option<u32>,
    conditions: bool,
) -> Result<BenchEntry> {
    let mut state = slot.live()?;
    let images = entry_images(max_images, state.file.images.len(), state.file.images.len())?;
    let passes = model::passes(&state.file.plan, conditions).len();
    let entry = new_entry(
        &state,
        name.unwrap_or(resolved.name),
        resolved.contestant,
        BenchEntryStatus::Queued,
        images,
        conditions,
        true,
        model::total(images, passes),
        resolved.missing,
    );
    let cancel = Arc::new(AtomicBool::new(false));
    add_entry(slot, &mut state, entry.clone(), cancel.clone(), None, None)?;
    drop(state);
    engine.push(Job {
        slot: slot.clone(),
        entry: entry.id.clone(),
        model: resolved.spec,
        images,
        conditions,
        cancel,
    });
    Ok(entry)
}

/// Stops entry `id` of `slot` if it is active: one that waits is taken out of the queue and is `stopped` at once,
/// and so is an open external entry (what it received is dropped); one that runs is `stopping` until its work
/// ends. Returns the entry, or `409 bench_entry_not_active`. The entry's file is written before it changes in
/// memory.
fn stop_entry_in(engine: &Engine, slot: &Slot, id: &str) -> Result<BenchEntry> {
    let mut state = slot.live()?;
    let e = state.entry(id)?;
    let mut entry = e.entry.clone();
    let ends = if e.open.is_some() {
        entry.status = BenchEntryStatus::Stopped;
        true
    } else {
        match e.entry.status {
            BenchEntryStatus::Queued if engine.unqueue(&slot.id, id) => {
                entry.status = BenchEntryStatus::Stopped;
                true
            }
            BenchEntryStatus::Queued | BenchEntryStatus::Running => {
                entry.status = BenchEntryStatus::Stopping;
                false
            }
            BenchEntryStatus::Stopping => return Ok(entry),
            s => {
                return Err(Error::conflict(
                    "bench_entry_not_active",
                    format!("entry '{id}' has already ended ({})", status_name(s)),
                )
                .hint("nothing to stop; GET /api/bench/benchmarks/{id}/entries/{entry_id} has its result"));
            }
        }
    };
    if ends {
        entry.finished = Some(rfc3339(unix_now()));
    }
    slot.save_entry(&state, &entry, None)?;
    let e = state.entry_mut(id).expect("looked up above, under the same lock");
    e.cancel.store(true, Ordering::SeqCst);
    if let Some(fill) = e.open.take() {
        fill.lock().unwrap_or_else(std::sync::PoisonError::into_inner).closed = true;
    }
    e.entry = entry.clone();
    slot.emit(BenchEvent::Entry(Box::new(entry.clone())));
    if ends {
        slot.emit_bench(&state);
    }
    Ok(entry)
}

fn status_name(s: BenchEntryStatus) -> &'static str {
    match s {
        BenchEntryStatus::Queued => "queued",
        BenchEntryStatus::Running => "running",
        BenchEntryStatus::Stopping => "stopping",
        BenchEntryStatus::Done => "done",
        BenchEntryStatus::Failed => "failed",
        BenchEntryStatus::Stopped => "stopped",
    }
}

/// How many images an entry runs: `max_images` (at least 1, at most the benchmark's), else `default` (capped).
fn entry_images(max_images: Option<u32>, images: usize, default: usize) -> Result<usize> {
    match max_images {
        Some(0) => Err(invalid("max_images must be at least 1")),
        Some(n) => Ok((n as usize).min(images)),
        None => Ok(default.min(images)),
    }
}

/// A new entry of the benchmark in `state` with a fresh id.
#[allow(clippy::too_many_arguments)]
fn new_entry(
    state: &State,
    name: String,
    contestant: BenchContestant,
    status: BenchEntryStatus,
    images: usize,
    conditions: bool,
    scored: bool,
    total: u64,
    missing: Vec<String>,
) -> BenchEntry {
    let id = unique_id("be", fnv1a(name.as_bytes()) ^ unix_now().to_bits(), |id| {
        state.entries.iter().any(|e| e.entry.id == id)
    });
    // To the millisecond and later than every entry the benchmark has: `created` (then the id) is the order of
    // the leaderboard's ties, and it has to be the order the entries were made in, also after a restart.
    let last = state.entries.iter().filter_map(|e| parse_rfc3339_millis(&e.entry.created)).max();
    let now = (unix_now() * 1000.0) as i64;
    let created = rfc3339_millis(last.map_or(now, |l| now.max(l + 1)));
    BenchEntry {
        id,
        bench_id: state.file.id.clone(),
        name,
        contestant,
        status,
        created,
        finished: None,
        progress: model::progress(total),
        images: images as u64,
        conditions,
        scored,
        fingerprint: state.fingerprint.clone(),
        missing_classes: missing,
        score: None,
        map50_95: None,
        ci95: None,
        conditions_run: 0,
        speed: None,
        usage: None,
        error: None,
        hint: None,
    }
}

/// Adds `entry` to the benchmark (written, sent, the benchmark's count sent too). `parts` for an entry that is
/// done already (an external one); `open` for an open external entry.
fn add_entry(
    slot: &Slot,
    state: &mut State,
    entry: BenchEntry,
    cancel: Arc<AtomicBool>,
    parts: Option<(&state::ResultParts, &[state::ImageOutcome])>,
    open: Option<Arc<Mutex<external::Filling>>>,
) -> Result<()> {
    slot.save_entry(state, &entry, parts)?;
    state.entries.push(EntryState { entry: entry.clone(), cancel, open });
    slot.emit(BenchEvent::Entry(Box::new(entry)));
    slot.emit_bench(state);
    Ok(())
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
        benches.sort_by_cached_key(|b| (b.lock().file.created.clone(), b.id.clone()));
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

    /// `POST /api/bench/benchmarks`: copies the source's images and truth into a new benchmark.
    pub fn create(&self, store: &dyn Store, req: CreateBench) -> Result<Bench> {
        let name = check_name(&req.name)?;
        let description = check_text("description", &req.description, MAX_DESCRIPTION_CHARS)?;
        let max = req.max_images.unwrap_or(DEFAULT_IMAGES);
        if !(1..=MAX_IMAGES).contains(&max) {
            return Err(invalid(format!("max_images must be in 1..={MAX_IMAGES}")));
        }
        if req.seed > MAX_SEED {
            return Err(invalid(format!("seed must be at most {MAX_SEED} (2^53 - 1), got {}", req.seed)));
        }
        let plan = match &req.plan {
            Some(p) => folo_bench::check_plan(p).map_err(invalid)?,
            None => folo_bench::default_plan(),
        };
        let source = source::gather(store, &req.source, Split::Val, true)?;
        let what = match &req.source {
            BenchSource::Dataset { .. } => format!(
                "the {} split of dataset '{}'",
                source.split.map_or("val", |s| s.as_str()),
                source.name
            ),
            BenchSource::LabelJob { .. } => format!("labeling job '{}'", source.name),
        };
        if source.classes.is_empty() {
            return Err(source::empty(
                format!("{what} has no classes"),
                "a benchmark scores boxes of classes",
            ));
        }
        if source.classes.len() > MAX_CLASSES {
            return Err(source::empty(
                format!(
                    "{what} has {} classes; a benchmark takes at most {MAX_CLASSES}",
                    source.classes.len()
                ),
                "make the benchmark from a source with fewer classes",
            ));
        }
        if source.images.is_empty() {
            return Err(source::empty(
                format!("{what} has no labeled image"),
                "a benchmark is made of labeled images: pick a split that has images, or label the job's images",
            ));
        }
        let source_split = match req.source {
            BenchSource::Dataset { dataset_id, .. } => {
                BenchSource::Dataset { dataset_id, split: source.split }
            }
            other => other,
        };
        let order = source::sample_order(source.images.len(), max as usize, req.seed);
        // The id is claimed by its folder, so two requests never share one. It is random and long, so a deleted
        // benchmark's id never comes back: its image URLs are served as immutable.
        let (id, dir) = loop {
            let id = {
                let list = self.benches.read().unwrap();
                let taken = |id: &str| list.iter().any(|b| b.id == id) || self.root.join(id).exists();
                let mut bytes = [0u8; 6];
                match getrandom::fill(&mut bytes).map(|()| format!("bn_{}", state::hex(&bytes))) {
                    Ok(id) if !taken(&id) => id,
                    _ => unique_id("bn", fnv1a(name.as_bytes()) ^ unix_now().to_bits(), taken),
                }
            };
            std::fs::create_dir_all(&self.root).map_err(|e| io_error("creating", &self.root, e))?;
            match std::fs::create_dir(self.root.join(&id)) {
                Ok(()) => break (id.clone(), self.root.join(&id)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_error("creating", &self.root.join(&id), e)),
            }
        };
        let made = self.fill(&dir, &id, name, description, source_split, req.seed, plan, &source, &order);
        let file = match made {
            Ok(file) => file,
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(e);
            }
        };
        let fingerprint = state::fingerprint(&file);
        let slot = Arc::new(Slot::new(
            dir,
            State { file, fingerprint, entries: Vec::new(), loops: Vec::new(), deleted: false },
        ));
        let bench = slot.lock().bench();
        self.benches.write().unwrap().push(slot);
        Ok(bench)
    }

    /// Copies the sampled images into the new benchmark's folder and writes its `bench.json` (last: without it,
    /// the folder is no benchmark).
    #[allow(clippy::too_many_arguments)]
    fn fill(
        &self,
        dir: &Path,
        id: &str,
        name: String,
        description: String,
        source: BenchSource,
        seed: u64,
        plan: folo_api::BenchPlan,
        gathered: &source::Gathered,
        order: &[usize],
    ) -> Result<BenchFile> {
        let images_dir = dir.join("images");
        std::fs::create_dir_all(&images_dir).map_err(|e| io_error("creating", &images_dir, e))?;
        let mut taken = HashSet::new();
        let names: Vec<String> =
            order.iter().map(|&i| unique_file_name(&gathered.images[i].file, &mut taken)).collect();
        let classes = gathered.classes.len();
        let images = order
            .par_iter()
            .zip(names.par_iter())
            .enumerate()
            .map(|(k, (&i, file))| {
                let im = &gathered.images[i];
                let bytes = std::fs::read(&im.path).map_err(|e| io_error("reading", &im.path, e))?;
                let to = images_dir.join(file);
                std::fs::write(&to, &bytes).map_err(|e| io_error("writing", &to, e))?;
                Ok(ImageRecord {
                    id: (k + 1).to_string(),
                    file: file.clone(),
                    width: im.width,
                    height: im.height,
                    bytes: bytes.len() as u64,
                    sha256: sha256(&bytes),
                    boxes: source::clip_truth(&im.boxes, classes, (im.width, im.height)),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let file = BenchFile {
            version: VERSION,
            id: id.to_string(),
            name,
            description,
            classes: gathered.classes.clone(),
            created: rfc3339(unix_now()),
            source,
            source_name: gathered.name.clone(),
            seed,
            plan,
            images,
        };
        let bytes = serde_json::to_vec(&file).expect("benchmarks serialize");
        let path = dir.join("bench.json");
        crate::index::write_atomic(&path, &bytes).map_err(|e| io_error("writing", &path, e))?;
        Ok(file)
    }

    /// `PUT /api/bench/benchmarks/{id}`.
    pub fn update(&self, id: &str, req: UpdateBench) -> Result<Bench> {
        let name = req.name.as_deref().map(check_name).transpose()?;
        let description = req
            .description
            .as_deref()
            .map(|d| check_text("description", d, MAX_DESCRIPTION_CHARS))
            .transpose()?;
        let plan = req.plan.as_ref().map(folo_bench::check_plan).transpose().map_err(invalid)?;
        let slot = self.slot(id)?;
        let mut state = slot.live()?;
        if plan.is_some()
            && let Some(e) = state.frozen_by()
        {
            return Err(Error::conflict(
                "bench_frozen",
                format!(
                    "benchmark '{id}' has a {} entry ('{}'): its plan is fixed while it has done or active entries \
                     (entries compare only on one plan)",
                    status_name(e.status),
                    e.name
                ),
            )
            .hint("delete its done entries and stop its active ones first, or make a new benchmark with this plan"));
        }
        if let Some(name) = name {
            state.file.name = name;
        }
        if let Some(d) = description {
            state.file.description = d;
        }
        if let Some(plan) = plan {
            state.file.plan = plan;
            state.fingerprint = state::fingerprint(&state.file);
        }
        slot.save(&state)?;
        slot.emit_bench(&state);
        Ok(state.bench())
    }

    /// `DELETE /api/bench/benchmarks/{id}`: its folder (images, entries, loops) and thumbnails; never a model, a run
    /// or a dataset a loop made.
    pub fn delete(&self, id: &str) -> Result<()> {
        let slot = self.slot(id)?;
        {
            let mut state = slot.live()?;
            if state.busy() {
                return Err(Error::conflict(
                    "bench_active",
                    format!("benchmark '{id}' has an entry or a loop that is still active"),
                )
                .hint("stop them first (POST …/entries/{entry_id}/stop, …/loops/{loop_id}/stop)"));
            }
            let file = slot.dir.join("bench.json");
            if let Err(e) = std::fs::remove_file(&file)
                && e.kind() != std::io::ErrorKind::NotFound
            {
                return Err(io_error("removing", &file, e));
            }
            state.deleted = true;
        }
        self.benches.write().unwrap().retain(|b| b.id != id);
        self.renders.forget(id);
        let _ = std::fs::remove_dir_all(self.thumbs.join(id));
        if let Err(e) = std::fs::remove_dir_all(&slot.dir) {
            tracing::warn!("removing {}: {e}", slot.dir.display());
        }
        Ok(())
    }

    /// `POST /api/bench/benchmarks/{id}/design` ([`design::design`]).
    pub fn design(&self, store: &dyn Store, id: &str, req: DesignBench) -> Result<BenchDesign> {
        design::design(store, &*self.slot(id)?, &self.engine.closing, req)
    }

    /// `GET /api/bench/benchmarks/{id}/images`: a page, the truth under `variant` when one is given.
    pub fn images(
        &self,
        id: &str,
        offset: u64,
        limit: u64,
        variant: Option<(Condition, u8)>,
    ) -> Result<BenchImagePage> {
        let slot = self.slot(id)?;
        let state = slot.live()?;
        let images = state
            .file
            .images
            .iter()
            .skip(offset.min(usize::MAX as u64) as usize)
            .take(limit.min(usize::MAX as u64) as usize)
            .map(|im| bench_image(im, variant))
            .collect::<Result<Vec<_>>>()?;
        Ok(BenchImagePage { total: state.file.images.len() as u64, images })
    }

    /// `GET /api/bench/benchmarks/{id}/images/{image_id}`: the stored file, or the variant every contestant is
    /// graded on; `thumb` as for dataset images.
    pub fn image(
        &self,
        id: &str,
        image_id: &str,
        variant: Option<(Condition, u8)>,
        thumb: Option<u32>,
    ) -> Result<ImageData> {
        let slot = self.slot(id)?;
        let (im, seed) = {
            let state = slot.live()?;
            (
                state.file.image(image_id).cloned().ok_or_else(|| image_not_found(id, image_id))?,
                state.file.seed,
            )
        };
        images::image(&self.renders, &slot, &im, seed, variant, thumb, &self.thumbs.join(id))
    }

    /// `GET /api/bench/benchmarks/{id}/entries`: the leaderboard.
    pub fn entries(&self, id: &str) -> Result<Vec<BenchEntry>> {
        Ok(self.slot(id)?.live()?.leaderboard())
    }

    /// `GET /api/bench/benchmarks/{id}/entries/{entry_id}`.
    pub fn entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        Ok(self.slot(id)?.live()?.entry(entry_id)?.entry.clone())
    }

    /// `POST /api/bench/benchmarks/{id}/entries`.
    pub fn start_entry(&self, store: &dyn Store, id: &str, req: StartBenchEntry) -> Result<BenchEntry> {
        let slot = self.slot(id)?;
        let name =
            req.name.as_deref().map(str::trim).filter(|n| !n.is_empty()).map(check_name).transpose()?;
        match req.contestant {
            BenchContestantSpec::External { scored, open, predictions } => {
                if req.max_images.is_some() || req.conditions.is_some() {
                    return Err(invalid(
                        "max_images and conditions are not for an external contestant: its predictions say what \
                         it ran",
                    ));
                }
                let name = name.ok_or_else(|| invalid("name is required for an external contestant"))?;
                external::start(&slot, name, scored, open, predictions)
            }
            BenchContestantSpec::Model { model_id } => {
                let classes = slot.live()?.file.classes.clone();
                let m = store.get_model(&model_id)?;
                let resolved = model::resolve(&m, &classes)?;
                enqueue_model(
                    &self.engine,
                    &slot,
                    resolved,
                    name,
                    req.max_images,
                    req.conditions.unwrap_or(true),
                )
            }
            BenchContestantSpec::Agent { agent: agent_id, model, effort } => {
                let model = check_model(model.as_deref())?;
                let agent = store.labeler()?.agent(agent_id)?;
                let info = agent.info();
                let model = model.or(info.default_model);
                let effort = effort.unwrap_or(info.default_effort);
                let conditions = req.conditions.unwrap_or(false);
                let name = name.unwrap_or_else(|| match &model {
                    Some(m) => format!("{} ({m})", agent_id.display_name()),
                    None => agent_id.display_name().to_string(),
                });
                let mut state = slot.live()?;
                let images = entry_images(req.max_images, state.file.images.len(), AGENT_IMAGES as usize)?;
                let passes = model::passes(&state.file.plan, conditions).len();
                let mut entry = new_entry(
                    &state,
                    name,
                    BenchContestant::Agent { agent: agent_id, model: model.clone(), effort },
                    BenchEntryStatus::Running,
                    images,
                    conditions,
                    false,
                    model::total(images, passes),
                    Vec::new(),
                );
                entry.usage = Some(LabelUsage::default());
                let cancel = Arc::new(AtomicBool::new(false));
                add_entry(&slot, &mut state, entry.clone(), cancel.clone(), None, None)?;
                drop(state);
                let job = agent::AgentJob {
                    slot: slot.clone(),
                    entry: entry.id.clone(),
                    agent,
                    model,
                    effort,
                    images,
                    conditions,
                    cancel,
                };
                let engine = self.engine.clone();
                let thread = std::thread::Builder::new()
                    .name(format!("folo-bench-agent-{}", entry.id))
                    .spawn(move || agent::run(job, &engine.closing));
                match thread {
                    Ok(t) => self.keep(t),
                    Err(e) => {
                        let error = Error::internal(format!("could not start the entry's thread: {e}"));
                        model::finish_entry(&slot, &entry.id, false, Err(error.clone()));
                        return Err(error);
                    }
                }
                Ok(entry)
            }
        }
    }

    /// `POST …/entries/{entry_id}/predictions` ([`external::add`]).
    pub fn add_predictions(
        &self,
        id: &str,
        entry_id: &str,
        predictions: Vec<folo_api::BenchPrediction>,
    ) -> Result<BenchEntry> {
        external::add(&*self.slot(id)?, entry_id, predictions)
    }

    /// `POST …/entries/{entry_id}/finish` ([`external::finish`]).
    pub fn finish_open(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        external::finish(&*self.slot(id)?, entry_id)
    }

    /// `GET …/entries/{entry_id}/result`.
    pub fn result(&self, id: &str, entry_id: &str) -> Result<BenchResult> {
        let slot = self.slot(id)?;
        let entry = done_entry(&slot, entry_id)?;
        let file = slot.read_entry(entry_id)?;
        let r = file
            .result
            .ok_or_else(|| Error::internal(format!("entry '{entry_id}' has no result on disk")))?;
        Ok(BenchResult {
            entry,
            accuracy: r.accuracy,
            robustness: r.robustness,
            calibration: r.calibration,
            localization: r.localization,
            self_labeling: r.self_labeling,
        })
    }

    /// `GET …/entries/{entry_id}/images`: per image the truth, the detections and what was missed or wrong;
    /// `worst` puts the images with the most `missed + wrong` first.
    pub fn entry_images(
        &self,
        id: &str,
        entry_id: &str,
        worst: bool,
        offset: u64,
        limit: u64,
    ) -> Result<BenchEntryImagePage> {
        let slot = self.slot(id)?;
        done_entry(&slot, entry_id)?;
        let file = slot.read_entry(entry_id)?;
        let mut outcomes = file.images;
        if worst {
            outcomes.sort_by_key(|o| std::cmp::Reverse(o.missed + o.wrong));
        }
        let state = slot.live()?;
        let images = outcomes
            .into_iter()
            .skip(offset.min(usize::MAX as u64) as usize)
            .take(limit.min(usize::MAX as u64) as usize)
            .filter_map(|o| {
                let im = state.file.image(&o.image)?;
                Some(BenchEntryImage {
                    image: bench_image(im, None).ok()?,
                    detections: o.detections,
                    missed: o.missed,
                    wrong: o.wrong,
                })
            })
            .collect();
        Ok(BenchEntryImagePage { total: file.entry.images, images })
    }

    /// `POST …/entries/{entry_id}/stop`.
    pub fn stop_entry(&self, id: &str, entry_id: &str) -> Result<BenchEntry> {
        stop_entry_in(&self.engine, &*self.slot(id)?, entry_id)
    }

    /// `DELETE …/entries/{entry_id}`.
    pub fn delete_entry(&self, id: &str, entry_id: &str) -> Result<()> {
        let slot = self.slot(id)?;
        let mut state = slot.live()?;
        let e = state.entry(entry_id)?;
        if e.entry.status.is_active() {
            return Err(Error::conflict(
                "bench_entry_active",
                format!("entry '{entry_id}' is still {}", status_name(e.entry.status)),
            )
            .hint("stop it first: POST /api/bench/benchmarks/{id}/entries/{entry_id}/stop"));
        }
        if let Some(l) = state.loops.iter().find(|l| {
            l.lp.status.is_active()
                && l.lp.generations.iter().any(|g| g.entry_id.as_deref() == Some(entry_id))
        }) {
            return Err(Error::conflict(
                "bench_entry_in_loop",
                format!("loop '{}' is running and uses entry '{entry_id}'", l.lp.id),
            )
            .hint("delete it once the loop has ended, or stop the loop"));
        }
        slot.remove_entry(entry_id)?;
        state.entries.retain(|e| e.entry.id != entry_id);
        slot.emit(BenchEvent::Removed(BenchRemoved { id: entry_id.to_string() }));
        slot.emit_bench(&state);
        Ok(())
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
            let _ = t.join();
        }
    }
}

/// A finished entry, or `409 bench_entry_not_done`.
fn done_entry(slot: &Slot, entry_id: &str) -> Result<BenchEntry> {
    let state = slot.live()?;
    let e = state.entry(entry_id)?;
    if e.entry.status != BenchEntryStatus::Done {
        return Err(Error::conflict(
            "bench_entry_not_done",
            format!("entry '{entry_id}' is {}: it has a result once it is done", status_name(e.entry.status)),
        ));
    }
    Ok(e.entry.clone())
}

/// `404 image_not_found` for a benchmark's image.
fn image_not_found(bench: &str, id: &str) -> Error {
    Error::new(StatusCode::NOT_FOUND, "image_not_found", format!("benchmark '{bench}' has no image '{id}'"))
        .hint("GET /api/bench/benchmarks/{id}/images lists its images")
}

/// An image as the API shows it, its truth under `variant` when one is given.
fn bench_image(im: &ImageRecord, variant: Option<(Condition, u8)>) -> Result<BenchImage> {
    let boxes = match variant {
        None => im.boxes.clone(),
        Some((c, s)) => conditions::moved_boxes(c, s, im.width, im.height, &im.boxes)
            .map_err(|e| Error::new(StatusCode::BAD_REQUEST, "invalid_query", e.to_string()))?,
    };
    Ok(BenchImage { id: im.id.clone(), file: im.file.clone(), width: im.width, height: im.height, boxes })
}

/// The sweeper's thread: every so often, open external entries that got no part for the engine's timeout are
/// failed ([`external::expire`]). Ends when the server shuts down (it looks every 50 ms).
fn sweep(engine: &Engine, benches: &RwLock<Vec<Arc<Slot>>>) {
    let mut last = Instant::now();
    while !engine.closing.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(50));
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
