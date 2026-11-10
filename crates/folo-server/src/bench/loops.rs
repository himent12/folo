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
    BenchEntryStatus, BenchGeneration, BenchLabels, BenchLoopPhase, BenchLoopStatus, BenchLoopVerdict,
    DatasetFormat, Model, RegisterDataset, RunStatus, StartBenchLoop, TrainConfig,
};

use super::model::{self, Engine, Item};
use super::source::SourceImage;
use super::state::{SERVER_STOPPED_LOOP, Slot};
use crate::error::{Error, Result};
use crate::label::dataset;
use crate::label::job::unique_file_name;
use crate::store::Store;
use crate::util::{claim_dir, rfc3339, slug, unix_now};

/// The share of a student's dataset that goes to `val` (both splits carry the teacher's labels).
pub const VAL_FRACTION: f32 = 0.1;
/// The confidence a teacher's labels are kept from when nothing else says.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
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
    /// The pool, without the images the benchmark has.
    pub pool: Vec<SourceImage>,
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
        match ran {
            _ if closing => {
                lp.status = BenchLoopStatus::Failed;
                lp.error = Some(SERVER_STOPPED_LOOP.into());
                lp.hint = Some("its finished generations are kept; start a new loop to run the rest".into());
            }
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
    let last = teacher.map50_95;
    Ok(Ended::Done(if last > first_ci[1] {
        BenchLoopVerdict::Improved
    } else if last < first_ci[0] {
        BenchLoopVerdict::Regressed
    } else {
        BenchLoopVerdict::Held
    }))
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
    let threshold = job.config.min_confidence.or(teacher.threshold).unwrap_or(DEFAULT_THRESHOLD);
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
    let kept: Vec<(&SourceImage, &Vec<folo_api::BBox>)> =
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
    if let Err(e) = dataset::write(&dir, &classes, &written, VAL_FRACTION, seed) {
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
    let s = &job.config.student;
    let mut config: TrainConfig = serde_json::from_value(serde_json::json!({ "dataset_id": dataset }))
        .expect("a dataset id is a config");
    config.name = format!("{} gen {g}", job.name());
    config.arch = s.arch.as_deref().and_then(folo_api::Arch::parse).unwrap_or(job.teacher.arch);
    config.epochs = s.epochs.unwrap_or(30);
    config.imgsz = s.imgsz.unwrap_or(job.teacher.imgsz);
    if let Some(b) = s.batch {
        config.batch = b;
    }
    config.device = s.device.clone().unwrap_or_else(|| "auto".into());
    config.weights = s.weights.clone();
    config.seed = s.seed;
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
