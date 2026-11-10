//! Model entries: a model of this server run on a benchmark by FOLO. One worker thread runs them one at a time
//! in the order they were started (`queued` until their turn), on the store's inference device with the
//! predictor settings of validation (`folo val`: a square letterbox at the model's size, NMS at confidence
//! 0.001 and IoU 0.7, multi-label, 300 detections), so a benchmark's unchanged-image mAP is the mAP `folo val`
//! gives on the same images.
//!
//! The expensive part of an entry is making the variants (decode, condition, the canonical JPEG and its decode):
//! a preparing thread does that for a batch of images at a time on the rayon pool, one batch ahead of the
//! inference, so memory stays at a few batches whatever the benchmark's size. Each variant is graded (its mAP
//! pair, all the result needs of it) as soon as its pass is done, so only the unchanged images' detections are
//! kept to the end. A self-improvement loop's teacher labels its pool through the same path ([`label_pool`]);
//! [`Engine::infer`] keeps the two from running at once.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    BBox, BenchContestant, BenchDetection, BenchEntryStatus, BenchEvent, BenchProgress, BenchSpeed, Model,
};
use folo_bench::score::{self, Sample};
use folo_bench::{Condition, conditions};
use folo_core::Device;
use folo_vision::{NmsConfig, PredictConfig, Predictor};
use rayon::prelude::*;

use super::results;
use super::state::{SERVER_STOPPED_ENTRY, Slot};
use crate::engine::{self, engine_error};
use crate::error::{Error, Result};
use crate::util::{rfc3339, unix_now};

/// Images per forward pass (as `/eval` validates).
pub const BATCH: usize = crate::models::EVAL_BATCH;
/// Progress events of an entry come at most this often.
pub const PROGRESS_EVERY: Duration = Duration::from_millis(500);

/// Marks inference on a device in flight until the returned guard drops (the real store's GPU idle tracking:
/// the pooled GPU memory is not released under a running entry).
pub type InferenceGuard = Arc<dyn Fn(Device) -> Box<dyn Send> + Send + Sync>;

/// What runs models for benchmarks: the queue of model entries, the device and the switch that ends it all.
pub struct Engine {
    /// The inference device (`auto`, `cpu`, `gpu:<n>`), as the store's.
    device: Mutex<String>,
    guard: Option<InferenceGuard>,
    /// Set when the server shuts down.
    pub closing: AtomicBool,
    /// Held while a model runs (an entry, or a loop's teacher on its pool): one model at a time. Taken with
    /// [`Engine::one_at_a_time`], which gives up on a stop.
    infer: Mutex<()>,
    queue: Mutex<VecDeque<Job>>,
    ready: Condvar,
    /// Test support: while set, a model entry waits after its last pass, before it is graded.
    pub hold: AtomicBool,
    /// How long an open external entry waits for its next part, in milliseconds.
    pub open_timeout_ms: std::sync::atomic::AtomicU64,
}

/// A model entry waiting for (or taken by) the worker.
pub struct Job {
    pub slot: Arc<Slot>,
    pub entry: String,
    pub model: ModelSpec,
    /// Images of the benchmark it runs (a prefix).
    pub images: usize,
    /// Whether it runs the plan's conditions.
    pub conditions: bool,
    pub cancel: Arc<AtomicBool>,
}

/// A model resolved for an entry.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    /// Its weights file.
    pub path: PathBuf,
    /// For each class of the model, the benchmark's class of the same name (without case), if any.
    pub map: Vec<Option<u32>>,
}

/// A model matched to a benchmark's classes ([`resolve`]).
pub struct Resolved {
    pub spec: ModelSpec,
    pub contestant: BenchContestant,
    pub missing: Vec<String>,
    pub name: String,
}

/// Model `m` for a benchmark with `classes`: its classes matched by name without case (`400
/// bench_class_mismatch` when none matches).
pub fn resolve(m: &Model, classes: &[String]) -> Result<Resolved> {
    let key = |s: &str| s.trim().to_lowercase();
    let map: Vec<Option<u32>> =
        m.classes.iter().map(|c| classes.iter().position(|b| key(b) == key(c)).map(|i| i as u32)).collect();
    let missing: Vec<String> = classes
        .iter()
        .enumerate()
        .filter(|(i, _)| !map.contains(&Some(*i as u32)))
        .map(|(_, c)| c.clone())
        .collect();
    if map.iter().all(Option::is_none) {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "bench_class_mismatch",
            format!(
                "model '{}' knows none of the benchmark's classes: it has {:?}, the benchmark {:?}",
                m.id, m.classes, classes
            ),
        )
        .hint("classes are matched by name without case; pick a model trained on these classes"));
    }
    let Some(path) = m.path.clone() else {
        return Err(Error::internal(format!("model '{}' has no weights file", m.id)));
    };
    Ok(Resolved {
        spec: ModelSpec { path: PathBuf::from(path), map },
        contestant: BenchContestant::Model {
            model_id: m.id.clone(),
            arch: m.arch.as_str().to_string(),
            params: m.params,
            imgsz: m.imgsz,
        },
        missing,
        name: m.name.clone(),
    })
}

impl Engine {
    /// An engine on `device` (the guard marks inference for the store's GPU idle tracking).
    pub fn new(device: String, guard: Option<InferenceGuard>) -> Engine {
        Engine {
            device: Mutex::new(device),
            guard,
            closing: AtomicBool::new(false),
            infer: Mutex::new(()),
            queue: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            hold: AtomicBool::new(false),
            open_timeout_ms: std::sync::atomic::AtomicU64::new(
                super::external::OPEN_TIMEOUT.as_millis() as u64
            ),
        }
    }

    /// The right to run a model (one at a time), waited for until `cancel` is set or the server shuts down
    /// (`None` then): a stopped entry or loop queued behind a long one ends at once.
    pub fn one_at_a_time(&self, cancel: &AtomicBool) -> Option<std::sync::MutexGuard<'_, ()>> {
        loop {
            if cancel.load(Ordering::Relaxed) || self.closing.load(Ordering::SeqCst) {
                return None;
            }
            match self.infer.try_lock() {
                Ok(guard) => return Some(guard),
                Err(std::sync::TryLockError::Poisoned(p)) => return Some(p.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }

    /// Changes the inference device (entries that start from now on use it).
    pub fn set_device(&self, device: String) {
        *self.device.lock().unwrap() = device;
    }

    /// Queues a model entry.
    pub fn push(&self, job: Job) {
        self.queue.lock().unwrap().push_back(job);
        self.ready.notify_all();
    }

    /// Takes entry `entry` of benchmark `bench` out of the queue, when it still waits there.
    pub fn unqueue(&self, bench: &str, entry: &str) -> bool {
        let mut q = self.queue.lock().unwrap();
        let before = q.len();
        q.retain(|j| !(j.slot.id == bench && j.entry == entry));
        q.len() != before
    }

    /// Every job still waiting (the server shuts down).
    pub fn drain(&self) -> Vec<Job> {
        self.queue.lock().unwrap().drain(..).collect()
    }

    /// Wakes the worker (to end it, once `closing` is set). Under the queue's lock: the worker checks `closing`
    /// and starts to wait under it, so the wake-up cannot fall between the two and be lost.
    pub fn wake(&self) {
        let _queue = self.queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        self.ready.notify_all();
    }

    fn next(&self) -> Option<Job> {
        let mut q = self.queue.lock().unwrap();
        loop {
            if self.closing.load(Ordering::SeqCst) {
                return None;
            }
            if let Some(job) = q.pop_front() {
                return Some(job);
            }
            q = self.ready.wait(q).unwrap();
        }
    }

    /// A predictor for the weights at `path` on the inference device, set up as validation is.
    fn predictor(&self, path: &std::path::Path) -> Result<Predictor> {
        let device = engine::resolve(&self.device.lock().unwrap().clone())?;
        let model = engine::open_model(path, device, folo_vision::Precision::F32)?;
        let cfg = PredictConfig {
            imgsz: model.imgsz(),
            conf: NmsConfig::val().conf,
            iou: NmsConfig::val().iou,
            max_det: NmsConfig::val().max_det,
            rect: false,
            head: None,
        };
        Predictor::new(model, cfg).map_err(engine_error)
    }

    fn guard(&self, device: Device) -> Option<Box<dyn Send>> {
        self.guard.as_ref().map(|g| g(device))
    }
}

/// The worker thread: runs queued model entries until the server shuts down.
pub fn work(engine: Arc<Engine>) {
    while let Some(job) = engine.next() {
        let (slot, id) = (job.slot.clone(), job.entry.clone());
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&engine, job)));
        if ran.is_err() {
            // The worker lives on for the entries behind this one.
            slot.update_entry(&id, true, |e| {
                e.status = BenchEntryStatus::Failed;
                e.finished = Some(rfc3339(unix_now()));
                e.error = Some("the entry stopped on an internal error (see the server log)".into());
            });
        }
    }
}

/// An image to predict: its id, its file and its true boxes (empty for a pool image).
pub struct Item {
    pub id: String,
    pub path: PathBuf,
    pub boxes: Vec<BBox>,
}

/// An image ready for the network: its pixels and the truth in its coordinates.
struct Prepared {
    image: folo_data::Image,
    truths: Vec<BBox>,
}

/// Reads image `item` and makes it what the contestant sees: the file decoded as datasets are, or under `variant`
/// the canonical JPEG of the variant decoded again.
fn prepare(item: &Item, variant: Option<(Condition, u8)>, seed: u64) -> Result<Prepared> {
    let bytes = std::fs::read(&item.path).map_err(|e| {
        Error::internal(format!("reading image '{}' ({}): {e}", item.id, item.path.display()))
    })?;
    let bad = |e: &dyn std::fmt::Display| Error::internal(format!("image '{}': {e}", item.id));
    match variant {
        None => Ok(Prepared {
            image: folo_data::decode(&bytes).map_err(|e| bad(&e))?,
            truths: item.boxes.clone(),
        }),
        Some((c, s)) => {
            let seed = conditions::variant_seed(seed, &item.id, c, s);
            let r = conditions::render(&bytes, c, s, &item.boxes, seed).map_err(|e| bad(&e))?;
            Ok(Prepared { image: folo_data::decode(&r.jpeg).map_err(|e| bad(&e))?, truths: r.boxes })
        }
    }
}

/// One forward pass over `images` with validation's post-processing, boxes in original-image pixels. Like
/// validation (which scores in network pixels), the boxes are not clipped to the image: an untrained model's
/// boxes reach past its edges, and clipping them would change their IoU, and so the mAP `folo val` reports.
fn predict(p: &Predictor, images: &[folo_data::Image]) -> Result<Vec<Vec<folo_vision::Detection>>> {
    let run = || -> folo_vision::Result<Vec<Vec<folo_vision::Detection>>> {
        let batch = p.prepare(images, false)?;
        let out = p.forward(&batch)?;
        Ok(out.detections(&NmsConfig::val())?)
    };
    let dets =
        if p.device().is_cpu() { folo_core::cpu::in_pool(run) } else { run() }.map_err(engine_error)?;
    // The square letterbox `prepare` used for a batch.
    let s = p.config().imgsz;
    let opts = folo_data::LetterboxOptions { stride: None, ..Default::default() };
    Ok(dets
        .into_iter()
        .zip(images)
        .map(|(dets, im)| {
            let lb = folo_data::Letterbox::compute(im.width, im.height, s, s, &opts);
            let ([rx, ry], [px, py]) = (lb.ratio, lb.pad);
            dets.into_iter()
                .map(|d| {
                    let b = d.bbox;
                    let bbox = folo_vision::Xyxy::new(
                        (b.x1 - px) / rx,
                        (b.y1 - py) / ry,
                        (b.x2 - px) / rx,
                        (b.y2 - py) / ry,
                    );
                    folo_vision::Detection { bbox, ..d }
                })
                .collect()
        })
        .collect())
}

/// What a pass gives per image: the truth in the variant's coordinates and the raw detections.
type Predicted = Vec<(Vec<BBox>, Vec<folo_vision::Detection>)>;

/// Predicts every item under `variant` (`None`: unchanged), the next batch prepared while one is predicted.
/// `each(n)` is called after every batch with the images done so far in this pass. `Ok(None)` when `cancel` was
/// set.
fn predict_pass(
    p: &Predictor,
    items: &[Item],
    variant: Option<(Condition, u8)>,
    seed: u64,
    cancel: &AtomicBool,
    mut each: impl FnMut(usize),
) -> Result<Option<Predicted>> {
    let mut out: Predicted = Vec::with_capacity(items.len());
    let ended = std::thread::scope(|s| -> Result<bool> {
        let (tx, rx) = sync_channel::<Result<Vec<Prepared>>>(1);
        s.spawn(move || {
            for chunk in items.chunks(BATCH) {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                let prepared =
                    chunk.par_iter().map(|it| prepare(it, variant, seed)).collect::<Result<Vec<_>>>();
                let failed = prepared.is_err();
                if tx.send(prepared).is_err() || failed {
                    return;
                }
            }
        });
        for prepared in rx {
            let prepared = prepared?;
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let (images, truths): (Vec<folo_data::Image>, Vec<Vec<BBox>>) =
                prepared.into_iter().map(|p| (p.image, p.truths)).unzip();
            let dets = predict(p, &images)?;
            out.extend(truths.into_iter().zip(dets));
            each(out.len());
        }
        Ok(!cancel.load(Ordering::Relaxed))
    })?;
    Ok((ended && out.len() == items.len()).then_some(out))
}

/// The detections as the benchmark's (`map`: model class → benchmark class; others are dropped).
fn to_bench(dets: Vec<folo_vision::Detection>, map: &[Option<u32>]) -> Vec<BenchDetection> {
    dets.into_iter()
        .filter_map(|d| {
            let class = (*map.get(d.class as usize)?)?;
            Some(BenchDetection {
                class,
                score: d.score,
                x1: d.bbox.x1,
                y1: d.bbox.y1,
                x2: d.bbox.x2,
                y2: d.bbox.y2,
            })
        })
        .collect()
}

/// Sends an entry's progress at most every [`PROGRESS_EVERY`] (the entry in memory always has it).
struct Progress<'a> {
    slot: &'a Slot,
    id: &'a str,
    last: Option<Instant>,
}

impl Progress<'_> {
    fn set(&mut self, done: u64, variant: Option<(Condition, u8)>) {
        let mut state = self.slot.lock();
        let Some(e) = state.entry_mut(self.id) else { return };
        e.entry.progress.done = done;
        e.entry.progress.condition = variant.map(|v| v.0.id().to_string());
        e.entry.progress.severity = variant.map(|v| v.1);
        if self.last.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) {
            self.last = Some(Instant::now());
            self.slot.emit(BenchEvent::Entry(Box::new(e.entry.clone())));
        }
    }
}

/// What a finished model entry has.
struct Done {
    graded: results::Graded,
    speed: BenchSpeed,
}

/// Runs one model entry to its end (or its stop) and closes it.
fn run(engine: &Engine, job: Job) {
    let Job { slot, entry: id, model, images, conditions, cancel } = job;
    // Taken off the queue: running now (unless it was stopped, or the benchmark deleted, meanwhile).
    let snapshot = {
        let mut state = slot.lock();
        if state.deleted {
            return;
        }
        let file = state.file.clone();
        let Some(e) = state.entry_mut(&id) else { return };
        match e.entry.status {
            BenchEntryStatus::Queued if !cancel.load(Ordering::SeqCst) => {}
            // Stopped while the worker was taking it.
            BenchEntryStatus::Queued | BenchEntryStatus::Stopping => {
                drop(state);
                finish(engine, &slot, &id, Ok(None));
                return;
            }
            _ => return,
        }
        e.entry.status = BenchEntryStatus::Running;
        let entry = e.entry.clone();
        if let Err(err) = slot.save_entry(&state, &entry, None) {
            tracing::warn!(bench = %slot.id, "{err}");
        }
        slot.emit(BenchEvent::Entry(Box::new(entry)));
        file
    };
    let outcome = run_model(engine, &slot, &id, &snapshot, &model, images, conditions, &cancel);
    finish(engine, &slot, &id, outcome);
}

/// Closes entry `id` with what its run came to. The entry's file is written before the entry in memory changes:
/// a result that cannot be written leaves a `failed` entry (with the IO error), never a `done` one without a
/// result on disk.
pub fn finish_entry(
    slot: &Slot,
    id: &str,
    closing: bool,
    outcome: Result<Option<(results::Graded, Option<BenchSpeed>)>>,
) {
    let now = rfc3339(unix_now());
    let mut state = slot.lock();
    let Some(e) = state.entry(id).ok() else { return };
    // A stop that was accepted (or the server's shutdown) wins over a grade that finished meanwhile: the last
    // pass is not checked for it again, and neither is the grading.
    let stopped = closing || e.entry.status == BenchEntryStatus::Stopping || e.cancel.load(Ordering::SeqCst);
    let outcome = match outcome {
        Ok(Some(_)) if stopped => Ok(None),
        other => other,
    };
    let mut entry = e.entry.clone();
    let mut parts = None;
    match outcome {
        Ok(Some((graded, speed))) => {
            entry.status = BenchEntryStatus::Done;
            entry.progress.done = entry.progress.total;
            entry.progress.condition = None;
            entry.progress.severity = None;
            entry.score = Some(graded.score.clone());
            entry.map50_95 = Some(graded.map50_95);
            entry.ci95 = Some(graded.parts.accuracy.ci95);
            entry.conditions_run = graded.conditions_run();
            if speed.is_some() {
                entry.speed = speed;
            }
            parts = Some((graded.parts, graded.images));
        }
        Ok(None) if closing => {
            entry.status = BenchEntryStatus::Failed;
            entry.error = Some(SERVER_STOPPED_ENTRY.into());
            entry.hint = Some("start the entry again once the server runs".into());
        }
        Ok(None) => entry.status = BenchEntryStatus::Stopped,
        Err(err) => {
            entry.status = BenchEntryStatus::Failed;
            entry.error = Some(err.body.message);
            entry.hint = err.body.hint;
        }
    }
    entry.finished = Some(now);
    let entry = write_closed(slot, &state, entry, parts.as_ref().map(|(r, i)| (r, i.as_slice())));
    if let Some(e) = state.entry_mut(id) {
        e.entry = entry.clone();
    }
    slot.emit(BenchEvent::Entry(Box::new(entry)));
    slot.emit_bench(&state);
}

/// Writes a closed entry's file and returns the entry to keep: as it is, or, when its file (with its result)
/// cannot be written, `failed` with the IO error.
pub fn write_closed(
    slot: &Slot,
    state: &super::state::State,
    mut entry: folo_api::BenchEntry,
    parts: Option<(&super::state::ResultParts, &[super::state::ImageOutcome])>,
) -> folo_api::BenchEntry {
    if let Err(err) = slot.save_entry(state, &entry, parts) {
        tracing::warn!(bench = %slot.id, "{err}");
        if entry.status == BenchEntryStatus::Done {
            entry.status = BenchEntryStatus::Failed;
            entry.score = None;
            entry.map50_95 = None;
            entry.ci95 = None;
            entry.conditions_run = 0;
            entry.error = Some(format!("the entry's result could not be written: {}", err.body.message));
            entry.hint =
                Some("free disk space or fix the data dir's permissions, then start it again".into());
            let _ = slot.save_entry(state, &entry, None);
        }
    }
    entry
}

fn finish(engine: &Engine, slot: &Slot, id: &str, outcome: Result<Option<Done>>) {
    let closing = engine.closing.load(Ordering::SeqCst);
    finish_entry(slot, id, closing, outcome.map(|o| o.map(|d| (d.graded, Some(d.speed)))));
}

/// The passes of an entry: the unchanged images, then the plan's variants when `conditions`.
pub fn passes(plan: &folo_api::BenchPlan, conditions: bool) -> Vec<Option<(Condition, u8)>> {
    let mut out = vec![None];
    if conditions {
        out.extend(folo_bench::variants(plan).into_iter().map(Some));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_model(
    engine: &Engine,
    slot: &Slot,
    id: &str,
    file: &super::state::BenchFile,
    model: &ModelSpec,
    images: usize,
    conditions: bool,
    cancel: &AtomicBool,
) -> Result<Option<Done>> {
    let items: Vec<Item> = file.images[..images]
        .iter()
        .map(|im| Item { id: im.id.clone(), path: slot.image_path(im), boxes: im.boxes.clone() })
        .collect();
    let passes = passes(&file.plan, conditions);
    // One model at a time: a loop's teacher labeling its pool waits for this entry, and the other way round.
    let Some(_one) = engine.one_at_a_time(cancel) else { return Ok(None) };
    let p = engine.predictor(&model.path)?;
    let device = p.device();
    let _gpu = engine.guard(device);
    let mut progress = Progress { slot, id, last: None };
    let classes = file.classes.len();
    // The unchanged images' samples, kept to the end; each variant only as its mAP pair.
    let (mut clean, mut runs) = (Vec::new(), Vec::new());
    let mut ms_per_image = 0.0f64;
    for (k, &variant) in passes.iter().enumerate() {
        let base = (k * items.len()) as u64;
        let t0 = Instant::now();
        // When the first batch was done and how many images it had: the pass is timed from there, so loading the
        // model and warming the device up (shader compilation on a GPU) is not counted as the model's speed.
        let mut warm: Option<(Instant, usize)> = None;
        let Some(pass) = predict_pass(&p, &items, variant, file.seed, cancel, |n| {
            warm.get_or_insert((Instant::now(), n));
            progress.set(base + n as u64, variant)
        })?
        else {
            return Ok(None);
        };
        let samples: Vec<Sample> = pass
            .into_iter()
            .map(|(truths, dets)| Sample { truths, detections: to_bench(dets, &model.map) })
            .collect();
        match variant {
            None => {
                ms_per_image = match warm {
                    Some((at, n)) if n < items.len() => {
                        at.elapsed().as_secs_f64() * 1e3 / (items.len() - n) as f64
                    }
                    _ => t0.elapsed().as_secs_f64() * 1e3 / items.len().max(1) as f64,
                };
                clean = samples;
            }
            Some((c, s)) => runs.push((c, s, score::map(&samples, classes, true))),
        }
    }
    // Test support: wait here, after the last pass and before the grade (a stop that lands now must still win).
    while engine.hold.load(Ordering::SeqCst) && !engine.closing.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(5));
    }
    let ids: Vec<String> = items.iter().map(|i| i.id.clone()).collect();
    let complete = images == file.images.len() && (conditions || file.plan.conditions.is_empty());
    let spec = results::Spec { classes, scored: true, plan: &file.plan, seed: file.seed, complete };
    let graded = results::grade_maps(&clean, &ids, &runs, &spec);
    Ok(Some(Done {
        graded,
        speed: BenchSpeed { device: device.to_string(), ms_per_image: ms_per_image as f32 },
    }))
}

/// A loop's teacher labeling its pool: the model at `path` (whose classes are the benchmark's, in order) on every
/// image, keeping its detections from `threshold` up ([`folo_bench::score::labels`]). `Ok(None)` when `cancel`
/// was set. `each(n)` gets the images done so far.
pub fn label_pool(
    engine: &Engine,
    path: &std::path::Path,
    classes: usize,
    items: &[Item],
    threshold: f32,
    cancel: &AtomicBool,
    each: impl FnMut(usize),
) -> Result<Option<Vec<Vec<BBox>>>> {
    let Some(_one) = engine.one_at_a_time(cancel) else { return Ok(None) };
    let p = engine.predictor(path)?;
    let _gpu = engine.guard(p.device());
    let map: Vec<Option<u32>> =
        (0..p.classes().len() as u32).map(|c| ((c as usize) < classes).then_some(c)).collect();
    let Some(pass) = predict_pass(&p, items, None, 0, cancel, each)? else { return Ok(None) };
    Ok(Some(
        pass.into_iter()
            .map(|(_, dets)| folo_bench::score::labels(&to_bench(dets, &map), classes, true, threshold))
            .collect(),
    ))
}

/// `progress.total` of an entry of `images` images with these passes.
pub fn total(images: usize, passes: usize) -> u64 {
    (images * passes) as u64
}

/// A fresh progress record.
pub fn progress(total: u64) -> BenchProgress {
    BenchProgress { done: 0, total, condition: None, severity: None }
}
