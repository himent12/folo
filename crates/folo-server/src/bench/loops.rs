//! Self-improvement loops (docs/BENCH.md, *Self-improvement loops*): per generation the benchmark grades the
//! teacher, the teacher labels a pool by itself, a student trains on those labels alone, and the student becomes
//! the next teacher.
//!
//! A loop is one thread that drives everything else through what already exists: model entries go through the
//! model queue like any other entry, and the student's dataset and training run go through the [`Store`]
//! (`register_dataset`, `create_run`, `get_run`, `stop_run`), so the run is an ordinary run of the server, queued,
//! listed and stoppable like the rest. The thread holds the store only weakly (upgraded for each call), so a
//! loop never keeps the store alive; [`super::Benches::shutdown`] ends it. Loops do not survive the server: one
//! that was running when it stopped is `failed` at the next start, with its finished generations.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use axum::http::StatusCode;
use folo_api::{
    BenchEntryStatus, BenchEvent, BenchGeneration, BenchLabels, BenchLoop, BenchLoopPhase, BenchLoopStatus,
    BenchLoopVerdict, BenchSource, BenchStudent, DatasetFormat, Model, RegisterDataset, RunStatus,
    StartBenchLoop, TrainConfig,
};
use rayon::prelude::*;

use super::model::{self, Engine, Item};
use super::source::{self, PoolImage};
use super::state::{LoopState, SERVER_STOPPED_LOOP, Slot};
use super::{MAX_SEED, invalid, sha256};
use crate::error::{Error, Result};
use crate::label::check_name;
use crate::label::dataset;
use crate::label::job::unique_file_name;
use crate::store::Store;
use crate::util::{claim_dir, fnv1a, rfc3339, slug, unique_id, unix_now};

/// The share of a student's dataset that goes to `val` (both splits carry the teacher's labels).
pub const VAL_FRACTION: f32 = 0.1;
/// Most generations of a loop.
pub const MAX_GENERATIONS: u32 = 5;
/// The provenance line of a student's dataset (`data.yaml`).
const PROVENANCE: &str =
    "# made by a FOLO self-improvement loop: the teacher's labels of the pool, no true label";
/// Fewest pool images a teacher must keep a box on for a student to train on them.
pub const MIN_LABELED_IMAGES: usize = 10;
/// How often a loop looks at the entry or the run it waits for.
const POLL: Duration = Duration::from_millis(100);

/// A loop to run, everything checked.
pub struct LoopJob {
    pub slot: Arc<Slot>,
    pub id: String,
    pub engine: Arc<Engine>,
    pub store: Weak<dyn Store>,
    pub cancel: Arc<AtomicBool>,
    /// The request with its defaults filled in.
    pub config: StartBenchLoop,
    /// The first teacher.
    pub teacher: Model,
    /// The pool, without the images the benchmark has (and without labels: a pool's are never read).
    pub pool: Vec<PoolImage>,
    /// Pool images left out because the benchmark has them.
    pub excluded: u64,
    /// `<data dir>/datasets`, where the students' datasets are written.
    pub datasets: PathBuf,
}

/// How a loop's work ended, short of an error.
enum Ended {
    Done(BenchLoopVerdict),
    Stopped,
}

/// A model with its benchmark entry, as a generation records it.
struct Graded {
    model: Model,
    map50_95: f32,
    ci95: [f32; 2],
    /// The entry's self-labeling threshold.
    threshold: Option<f32>,
}

/// Runs a loop to its end (the body of its thread) and closes it.
pub fn run(job: LoopJob) {
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(&job)));
    let closing = job.engine.closing.load(Ordering::SeqCst);
    job.slot.update_loop(&job.id, |lp| {
        lp.phase = None;
        lp.finished = Some(rfc3339(unix_now()));
        // A stop that was accepted wins over a last generation that finished meanwhile, as it does for an entry.
        let stopped = lp.status == BenchLoopStatus::Stopping || job.cancelled();
        match ran {
            _ if closing && lp.status == BenchLoopStatus::Stopping => lp.status = BenchLoopStatus::Stopped,
            _ if closing => {
                lp.status = BenchLoopStatus::Failed;
                lp.error = Some(SERVER_STOPPED_LOOP.into());
                lp.hint = Some("its finished generations are kept; start a new loop to run the rest".into());
            }
            Ok(Ok(Ended::Done(_))) if stopped => lp.status = BenchLoopStatus::Stopped,
            Ok(Ok(Ended::Done(verdict))) => {
                lp.status = BenchLoopStatus::Done;
                lp.verdict = Some(verdict);
            }
            Ok(Ok(Ended::Stopped)) => lp.status = BenchLoopStatus::Stopped,
            Ok(Err(e)) => {
                lp.status = BenchLoopStatus::Failed;
                lp.error = Some(e.body.message);
                lp.hint = e.body.hint;
            }
            Err(_) => {
                lp.status = BenchLoopStatus::Failed;
                lp.error = Some("the loop stopped on an internal error (see the server log)".into());
            }
        }
    });
}

fn failed(message: impl Into<String>) -> Error {
    Error::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
}

/// A checked `POST …/loops`: everything a loop needs but its id and its thread.
pub struct Checked {
    pub config: StartBenchLoop,
    pub teacher: Model,
    pub pool: Vec<PoolImage>,
    pub excluded: u64,
}

/// Checks `POST /api/bench/benchmarks/{id}/loops` and finds its pool: the numbers, the teacher (its classes must
/// be the benchmark's, in order: labels are written by class index), the students' training settings as `POST
/// /api/runs` checks them (all but the dataset, which does not exist yet), and the pool without the images the
/// benchmark has (`400 invalid_dataset` when nothing is left).
pub fn check(store: &dyn Store, slot: &Slot, req: StartBenchLoop) -> Result<Checked> {
    let generations = req.generations.unwrap_or(1);
    if !(1..=MAX_GENERATIONS).contains(&generations) {
        return Err(invalid(format!("generations must be in 1..={MAX_GENERATIONS}")));
    }
    if let Some(c) = req.min_confidence
        && !(0.0..=1.0).contains(&c)
    {
        return Err(invalid(format!("min_confidence must be in 0..=1, got {c}")));
    }
    let s = &req.student;
    if s.seed > MAX_SEED {
        return Err(invalid(format!("student.seed must be at most {MAX_SEED} (2^53 - 1), got {}", s.seed)));
    }
    if let Some(a) = s.arch.as_deref()
        && folo_api::Arch::parse(a).is_none()
    {
        return Err(invalid(format!("student.arch '{a}' is not an architecture (folo-n, folo-s, ...)")));
    }
    if s.epochs == Some(0) {
        return Err(invalid("student.epochs must be at least 1"));
    }
    if let Some(i) = s.imgsz
        && (!i.is_multiple_of(32) || !(32..=4096).contains(&i))
    {
        return Err(invalid(format!("student.imgsz must be a multiple of 32 in 32..=4096, got {i}")));
    }
    let teacher = store.get_model(&req.teacher)?;
    let (classes, sizes, hashes) = {
        let state = slot.live()?;
        let f = &state.file;
        let sizes: HashSet<u64> = f.images.iter().map(|i| i.bytes).collect();
        let hashes: HashSet<String> = f.images.iter().map(|i| i.sha256.clone()).collect();
        (f.classes.clone(), sizes, hashes)
    };
    let key = |c: &String| c.trim().to_lowercase();
    if teacher.classes.len() != classes.len()
        || !teacher.classes.iter().zip(&classes).all(|(a, b)| key(a) == key(b))
    {
        return Err(invalid(format!(
            "the teacher's classes {:?} are not the benchmark's {:?} (same names, same order): its labels are \
             written by class index",
            teacher.classes, classes
        ))
        .hint("a loop's teacher is a model trained on the benchmark's classes"));
    }
    let name = match req.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => check_name(n)?,
        None => format!("{} loop", teacher.name),
    };
    let student = BenchStudent {
        arch: Some(s.arch.clone().unwrap_or_else(|| teacher.arch.as_str().to_string())),
        epochs: Some(s.epochs.unwrap_or(30)),
        imgsz: Some(s.imgsz.unwrap_or(teacher.imgsz)),
        batch: s.batch,
        device: Some(s.device.clone().unwrap_or_else(|| "auto".into())),
        weights: s.weights.clone(),
        seed: s.seed,
    };
    // The students' runs are checked now, as `POST /api/runs` checks them (device, weights, batch, ...), not
    // after the teacher's entry and its labels.
    store.check_run(&student_config(&student, &teacher, "", String::new()))?;
    let pool = source::pool(store, &req.pool)?;
    // Pool images with a benchmark image's bytes are left out; only files of a benchmark image's size are read.
    let keep: Vec<bool> = pool
        .images
        .par_iter()
        .map(|im| {
            let len = std::fs::metadata(&im.path).map(|m| m.len()).unwrap_or(u64::MAX);
            !sizes.contains(&len)
                || std::fs::read(&im.path).map(|b| !hashes.contains(&sha256(&b))).unwrap_or(true)
        })
        .collect();
    let total = pool.images.len();
    let images: Vec<PoolImage> =
        pool.images.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(im, _)| im).collect();
    let excluded = (total - images.len()) as u64;
    if images.is_empty() {
        let why = if total == 0 {
            format!("the pool ('{}') has no images", pool.name)
        } else {
            format!("every image of the pool ('{}') is in the benchmark", pool.name)
        };
        return Err(source::empty(
            why,
            "a pool is the images the teacher labels for its student: pick images the benchmark does not have",
        ));
    }
    let pool_source = match req.pool {
        BenchSource::Dataset { dataset_id, .. } => BenchSource::Dataset { dataset_id, split: pool.split },
        other => other,
    };
    let config = StartBenchLoop {
        name: Some(name),
        teacher: teacher.id.clone(),
        pool: pool_source,
        generations: Some(generations),
        min_confidence: req.min_confidence,
        student,
    };
    Ok(Checked { config, teacher, pool: images, excluded })
}

/// Adds a new loop of `checked` to `slot` (`running`, written, sent) and returns it with its stop switch.
pub fn begin(slot: &Slot, checked: &Checked) -> Result<(BenchLoop, Arc<AtomicBool>)> {
    let cancel = Arc::new(AtomicBool::new(false));
    let mut state = slot.live()?;
    let name = checked.config.name.clone().unwrap_or_default();
    let lid = unique_id("bl", fnv1a(name.as_bytes()) ^ unix_now().to_bits(), |l| {
        state.loops.iter().any(|x| x.lp.id == l)
    });
    let lp = BenchLoop {
        id: lid,
        bench_id: slot.id.clone(),
        name,
        status: BenchLoopStatus::Running,
        phase: Some(BenchLoopPhase::Benchmarking),
        created: rfc3339(unix_now()),
        finished: None,
        config: checked.config.clone(),
        generations: Vec::new(),
        verdict: None,
        error: None,
        hint: None,
    };
    slot.save_loop(&state, &lp)?;
    state.loops.push(LoopState { lp: lp.clone(), cancel: cancel.clone() });
    slot.emit(BenchEvent::Loop(Box::new(lp.clone())));
    slot.emit_bench(&state);
    Ok((lp, cancel))
}

/// A student's training config: `student`'s settings (the defaults filled in from the first `teacher`) on
/// `dataset`, named `name`; what `POST /api/runs` gets.
fn student_config(student: &BenchStudent, teacher: &Model, dataset: &str, name: String) -> TrainConfig {
    let mut config: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": dataset }))
        .expect("a dataset id is a config");
    config.name = name;
    config.arch = student.arch.as_deref().and_then(folo_api::Arch::parse).unwrap_or(teacher.arch);
    config.epochs = student.epochs.unwrap_or(30);
    config.imgsz = student.imgsz.unwrap_or(teacher.imgsz);
    if let Some(b) = student.batch {
        config.batch = b;
    }
    config.device = student.device.clone().unwrap_or_else(|| "auto".into());
    config.weights = student.weights.clone();
    config.seed = student.seed;
    config
}

/// The verdict of a loop: the last generation's mAP@0.5:0.95 against the first teacher's 95% interval `ci`
/// (`improved` above it, `regressed` below, `held` inside, its bounds included).
pub fn verdict(last: f32, ci: [f32; 2]) -> BenchLoopVerdict {
    if last > ci[1] {
        BenchLoopVerdict::Improved
    } else if last < ci[0] {
        BenchLoopVerdict::Regressed
    } else {
        BenchLoopVerdict::Held
    }
}

/// The confidence a teacher labels the pool from: `min_confidence` when given, else the self-labeling threshold
/// of its entry (`threshold`). Without either there is no confidence the benchmark supports: the loop fails here,
/// before anything is labeled, rather than guess one.
pub fn label_threshold(min_confidence: Option<f32>, threshold: Option<f32>, teacher: &str) -> Result<f32> {
    min_confidence.or(threshold).ok_or_else(|| {
        Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            format!(
                "'{teacher}' never reaches 90% precision on at least {} labels on this benchmark, so there is no \
                 confidence to label the pool from",
                folo_bench::score::MIN_LABELS
            ),
        )
        .hint(
            "set min_confidence: its entry's self-labeling chart shows what each confidence costs in precision and \
             coverage",
        )
    })
}

impl LoopJob {
    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// The store, while the server runs.
    fn store(&self) -> Result<Arc<dyn Store>> {
        self.store.upgrade().ok_or_else(|| failed(SERVER_STOPPED_LOOP))
    }

    fn name(&self) -> &str {
        self.config.name.as_deref().unwrap_or("loop")
    }

    fn phase(&self, phase: BenchLoopPhase) {
        self.slot.update_loop(&self.id, |lp| lp.phase = Some(phase));
    }

    /// Changes generation `g` of the loop.
    fn generation(&self, g: u32, change: impl FnOnce(&mut BenchGeneration)) {
        self.slot.update_loop(&self.id, |lp| {
            if let Some(row) = lp.generations.iter_mut().find(|x| x.generation == g) {
                change(row);
            }
        });
    }

    fn start_generation(&self, g: u32, model: Option<&Model>) {
        self.slot.update_loop(&self.id, |lp| {
            lp.generations.push(BenchGeneration {
                generation: g,
                labels: None,
                run_id: None,
                model_id: model.map(|m| m.id.clone()),
                model_name: model.map(|m| m.name.clone()),
                entry_id: None,
                score: None,
                map50_95: None,
                ci95: None,
            })
        });
    }
}

/// The loop's work: generation 0, then each student.
fn drive(job: &LoopJob) -> Result<Ended> {
    job.start_generation(0, Some(&job.teacher));
    let Some(mut teacher) = graded(job, 0, &job.teacher)? else { return Ok(Ended::Stopped) };
    let first_ci = teacher.ci95;
    let generations = job.config.generations.unwrap_or(1);
    for g in 1..=generations {
        job.start_generation(g, None);
        let Some(dataset) = teach(job, g, &teacher)? else { return Ok(Ended::Stopped) };
        let Some(student) = train(job, g, &dataset)? else { return Ok(Ended::Stopped) };
        job.generation(g, |row| {
            row.model_id = Some(student.id.clone());
            row.model_name = Some(student.name.clone());
        });
        let Some(next) = graded(job, g, &student)? else { return Ok(Ended::Stopped) };
        teacher = next;
    }
    Ok(Ended::Done(verdict(teacher.map50_95, first_ci)))
}

/// Generation `g`'s model graded by the benchmark: a finished, complete entry of it on the current fingerprint,
/// else a new one. `None` when the loop was stopped.
fn graded(job: &LoopJob, g: u32, m: &Model) -> Result<Option<Graded>> {
    job.phase(BenchLoopPhase::Benchmarking);
    let reuse = {
        let state = job.slot.lock();
        // Complete: every image and every condition of the plan (not `partial`, which an unmeasured axis also
        // sets).
        let (images, planned) = (state.file.images.len() as u64, !state.file.plan.conditions.is_empty());
        state
            .entries
            .iter()
            .rev()
            .map(|e| &e.entry)
            .find(|e| {
                e.status == BenchEntryStatus::Done
                    && e.fingerprint == state.fingerprint
                    && e.images == images
                    && (e.conditions || !planned)
                    && matches!(&e.contestant, folo_api::BenchContestant::Model { model_id, .. } if *model_id == m.id)
            })
            .map(|e| e.id.clone())
    };
    let entry_id = match reuse {
        Some(id) => id,
        None => {
            let file_classes = job.slot.lock().file.classes.clone();
            let resolved = model::resolve(m, &file_classes)?;
            super::enqueue_model(&job.engine, &job.slot, resolved, None, None, true)?.id
        }
    };
    job.generation(g, |row| row.entry_id = Some(entry_id.clone()));
    // Waits for the entry; a stop of the loop stops it.
    loop {
        let status = job.slot.lock().entry(&entry_id).map(|e| e.entry.clone());
        let entry = status.map_err(|_| failed(format!("entry '{entry_id}' of the loop was deleted")))?;
        match entry.status {
            BenchEntryStatus::Done => {
                let file = job.slot.read_entry(&entry_id)?;
                let result =
                    file.result.ok_or_else(|| failed(format!("entry '{entry_id}' has no result")))?;
                let graded = Graded {
                    model: m.clone(),
                    map50_95: result.accuracy.map50_95,
                    ci95: result.accuracy.ci95,
                    threshold: result.self_labeling.threshold,
                };
                job.generation(g, |row| {
                    row.score = entry.score.as_ref().map(|s| s.total);
                    row.map50_95 = Some(graded.map50_95);
                    row.ci95 = Some(graded.ci95);
                });
                return Ok(Some(graded));
            }
            BenchEntryStatus::Failed => {
                return Err(failed(format!(
                    "the benchmark entry of model '{}' ({entry_id}) failed: {}",
                    m.id,
                    entry.error.unwrap_or_default()
                ))
                .hint("see the entry; start a new loop once it can run"));
            }
            BenchEntryStatus::Stopped if job.cancelled() => return Ok(None),
            BenchEntryStatus::Stopped => {
                return Err(failed(format!(
                    "the benchmark entry of model '{}' ({entry_id}) was stopped",
                    m.id
                )));
            }
            _ if job.cancelled() => {
                let _ = super::stop_entry_in(&job.engine, &job.slot, &entry_id);
            }
            _ => {}
        }
        std::thread::sleep(POLL);
    }
}

/// Generation `g`'s dataset: the teacher's labels of the pool, written as a YOLO dataset and registered. `None`
/// when the loop was stopped.
fn teach(job: &LoopJob, g: u32, teacher: &Graded) -> Result<Option<String>> {
    job.phase(BenchLoopPhase::Labeling);
    let threshold = label_threshold(job.config.min_confidence, teacher.threshold, &teacher.model.name)?;
    let classes = job.slot.lock().file.classes.clone();
    let path = teacher
        .model
        .path
        .clone()
        .ok_or_else(|| failed(format!("model '{}' has no file", teacher.model.id)))?;
    let items: Vec<Item> = job
        .pool
        .iter()
        .enumerate()
        .map(|(i, im)| Item { id: format!("pool-{i}"), path: im.path.clone(), boxes: Vec::new() })
        .collect();
    let labels = model::label_pool(
        &job.engine,
        std::path::Path::new(&path),
        classes.len(),
        &items,
        threshold,
        &job.cancel,
        |_| {},
    )?;
    let Some(labels) = labels else { return Ok(None) };
    // Clipped to each image as a dataset's labels must be (the model's boxes are scored unclipped).
    let labels: Vec<Vec<folo_api::BBox>> = labels
        .iter()
        .zip(&job.pool)
        .map(|(l, im)| super::source::clip_truth(l, classes.len(), (im.width, im.height)))
        .collect();
    let boxes: usize = labels.iter().map(Vec::len).sum();
    // An image the teacher kept nothing on is left out, not written as a background image: its silence is not a
    // label, and a student trained on such images would learn to see nothing in them.
    let kept: Vec<(&PoolImage, &Vec<folo_api::BBox>)> =
        job.pool.iter().zip(&labels).filter(|(_, boxes)| !boxes.is_empty()).collect();
    if kept.len() < MIN_LABELED_IMAGES {
        let message = format!(
            "the teacher '{}' kept {boxes} {} on {} of the pool's {} images at confidence {threshold}: too few to \
             train a student on (at least {MIN_LABELED_IMAGES} images)",
            teacher.model.name,
            if boxes == 1 { "box" } else { "boxes" },
            kept.len(),
            items.len()
        );
        return Err(Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", message).hint(
            "a lower min_confidence keeps more of its detections (the self-labeling chart of its benchmark entry \
             shows what each confidence costs in precision); a better teacher finds more",
        ));
    }
    // The teacher's labels and nothing else: the pool's own labels are never read.
    let mut taken = HashSet::new();
    let written: Vec<dataset::Item> = kept
        .iter()
        .map(|(im, boxes)| dataset::Item {
            source: im.path.clone(),
            file: unique_file_name(&im.file, &mut taken),
            width: im.width,
            height: im.height,
            boxes: (*boxes).clone(),
        })
        .collect();
    let name = format!("{} gen {g}", job.name());
    let dir = claim_dir(&job.datasets, &slug(&name, "loop"))
        .map_err(|e| failed(format!("creating a folder in {}: {e}", job.datasets.display())))?;
    let seed = job.config.student.seed;
    if let Err(e) = dataset::write(&dir, &classes, &written, VAL_FRACTION, seed, PROVENANCE) {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(failed(format!("writing the dataset {}: {e}", dir.display())));
    }
    let registered = job.store()?.register_dataset(RegisterDataset {
        name: Some(name),
        path: dir.to_string_lossy().into_owned(),
        format: DatasetFormat::Yolo,
    });
    let ds = registered.inspect_err(|_| {
        let _ = std::fs::remove_dir_all(&dir);
    })?;
    let stats = BenchLabels {
        threshold,
        images: labels.len() as u64,
        labeled_images: kept.len() as u64,
        boxes: boxes as u64,
        excluded: job.excluded,
        dataset_id: ds.id.clone(),
    };
    job.generation(g, |row| row.labels = Some(stats));
    Ok(Some(ds.id))
}

/// Generation `g`'s student: an ordinary training run on `dataset`, followed to its end; its best model. `None`
/// when the loop was stopped.
fn train(job: &LoopJob, g: u32, dataset: &str) -> Result<Option<Model>> {
    job.phase(BenchLoopPhase::Training);
    let config =
        student_config(&job.config.student, &job.teacher, dataset, format!("{} gen {g}", job.name()));
    let run = job.store()?.create_run(config)?;
    job.generation(g, |row| row.run_id = Some(run.id.clone()));
    let mut stopping = false;
    loop {
        let run = job.store()?.get_run(&run.id)?;
        if !run.status.is_active() {
            return match run.status {
                RunStatus::Completed => {
                    let model_id = run.model_id.ok_or_else(|| {
                        failed(format!("the training run '{}' finished without weights", run.id))
                    })?;
                    Ok(Some(job.store()?.get_model(&model_id)?))
                }
                _ if stopping => Ok(None),
                RunStatus::Stopped => Err(failed(format!(
                    "the training run '{}' of generation {g} was stopped before it finished",
                    run.id
                ))
                .hint("a loop's student must train to the end; start a new loop")),
                _ => Err(failed(format!(
                    "the training run '{}' of generation {g} failed: {}",
                    run.id,
                    run.error.unwrap_or_default()
                ))
                .hint("GET /api/runs/{id}/logs says why")),
            };
        }
        if job.cancelled() && !stopping {
            stopping = true;
            let _ = job.store()?.stop_run(&run.id);
            // The server is shutting down: its runs are stopped by it, nothing to wait for here.
            if job.engine.closing.load(Ordering::SeqCst) {
                return Ok(None);
            }
        }
        std::thread::sleep(POLL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_verdict_is_on_the_first_teachers_interval_bounds_included() {
        let ci = [0.30, 0.40];
        assert_eq!(verdict(0.41, ci), BenchLoopVerdict::Improved);
        assert_eq!(verdict(0.40, ci), BenchLoopVerdict::Held);
        assert_eq!(verdict(0.35, ci), BenchLoopVerdict::Held);
        assert_eq!(verdict(0.30, ci), BenchLoopVerdict::Held);
        assert_eq!(verdict(0.2999, ci), BenchLoopVerdict::Regressed);
        assert_eq!(verdict(0.0, [0.0, 0.0]), BenchLoopVerdict::Held);
    }

    #[test]
    fn a_loop_labels_from_min_confidence_else_the_teachers_threshold_never_a_guess() {
        assert_eq!(label_threshold(Some(0.2), Some(0.6), "t").unwrap(), 0.2);
        assert_eq!(label_threshold(None, Some(0.6), "t").unwrap(), 0.6);
        let e = label_threshold(None, None, "folo-n pond").unwrap_err();
        assert!(
            e.body.message.starts_with("'folo-n pond' never reaches 90% precision on at least 10 labels"),
            "{}",
            e.body.message
        );
        assert!(e.body.hint.unwrap().contains("min_confidence"));
    }
}
