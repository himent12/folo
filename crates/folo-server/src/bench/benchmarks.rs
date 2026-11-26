//! Benchmarks themselves: making one from a dataset split or a labeling job, editing and deleting it, and
//! serving its images.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use axum::http::StatusCode;
use folo_api::{
    Bench, BenchDesign, BenchImage, BenchImagePage, BenchSource, CreateBench, DesignBench, Split, UpdateBench,
};
use folo_bench::{Condition, conditions};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::label::job::unique_file_name;
use crate::label::{check_name, check_text};
use crate::store::{ImageData, Store};
use crate::util::{fnv1a, parse_rfc3339_millis, rfc3339_millis, unique_id, unix_now};

use super::entries::status_name;
use super::state::{BenchFile, ImageRecord, Slot, State, VERSION};
use super::{
    Benches, DEFAULT_IMAGES, MAX_CLASSES, MAX_DESCRIPTION_CHARS, MAX_IMAGES, MAX_SEED, design, images,
    invalid, io_error, sha256, source, state,
};

/// Validates the non-source fields of [`CreateBench`] (`name`, `description`, `max_images`, `seed`, `plan`).
/// Shared by [`Benches::create`] and `folo bench create` so invalid CLI arguments fail before registering `--data`.
pub fn check_create(
    name: &str,
    description: &str,
    max_images: Option<u32>,
    seed: u64,
    plan: Option<&folo_api::BenchPlan>,
) -> Result<(String, String, u32, folo_api::BenchPlan)> {
    let name = check_name(name)?;
    let description = check_text("description", description, MAX_DESCRIPTION_CHARS)?;
    let max = max_images.unwrap_or(DEFAULT_IMAGES);
    if !(1..=MAX_IMAGES).contains(&max) {
        return Err(invalid(format!("max_images must be in 1..={MAX_IMAGES}")));
    }
    if seed > MAX_SEED {
        return Err(invalid(format!("seed must be at most {MAX_SEED} (2^53 - 1), got {seed}")));
    }
    let plan = match plan {
        Some(p) => folo_bench::check_plan(p).map_err(invalid)?,
        None => folo_bench::default_plan(),
    };
    Ok((name, description, max, plan))
}

/// Tracks an in-flight [`Benches::create`] referencing a dataset id so conditional dataset cleanup cannot
/// unregister the dataset while a benchmark is still gathering or writing it.
struct CreatingDatasetGuard<'a> {
    map: &'a std::sync::Mutex<std::collections::HashMap<String, usize>>,
    dataset_id: String,
}

impl<'a> CreatingDatasetGuard<'a> {
    fn new(map: &'a std::sync::Mutex<std::collections::HashMap<String, usize>>, dataset_id: String) -> Self {
        *map.lock().unwrap_or_else(|p| p.into_inner()).entry(dataset_id.clone()).or_insert(0) += 1;
        Self { map, dataset_id }
    }
}

impl Drop for CreatingDatasetGuard<'_> {
    fn drop(&mut self) {
        let mut guard = self.map.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(count) = guard.get_mut(&self.dataset_id) {
            if *count <= 1 {
                guard.remove(&self.dataset_id);
            } else {
                *count -= 1;
            }
        }
    }
}

impl Benches {
    /// Returns `true` when any existing benchmark or in-flight [`Benches::create`] references `dataset_id`.
    /// Checks in-flight creations before persisted benchmarks so a creation finishing concurrently cannot slip
    /// between the two checks.
    pub(crate) fn references_dataset(&self, dataset_id: &str) -> bool {
        if self.creating_datasets.lock().unwrap_or_else(|p| p.into_inner()).contains_key(dataset_id) {
            return true;
        }
        self.benches.read().unwrap().iter().any(|slot| {
            matches!(&slot.lock().file.source, BenchSource::Dataset { dataset_id: id, .. } if id == dataset_id)
        })
    }

    /// `POST /api/bench/benchmarks`: copies the source's images and truth into a new benchmark.
    pub fn create(&self, store: &dyn Store, req: CreateBench) -> Result<Bench> {
        let (name, description, max, plan) =
            check_create(&req.name, &req.description, req.max_images, req.seed, req.plan.as_ref())?;
        let _creating = match &req.source {
            BenchSource::Dataset { dataset_id, .. } => {
                Some(CreatingDatasetGuard::new(&self.creating_datasets, dataset_id.clone()))
            }
            BenchSource::LabelJob { .. } => None,
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
                let bytes = std::fs::read(&im.path).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        Error::new(
                            StatusCode::NOT_FOUND,
                            "image_not_found",
                            format!("image '{}' ({}) is no longer on disk", im.file, im.path.display()),
                        )
                        .hint("restore the missing image file, or remove it from the source")
                    } else {
                        io_error("reading", &im.path, e)
                    }
                })?;
                let digest = sha256(&bytes);
                if folo_data::jpeg_truncated(&bytes) {
                    return Err(Error::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "image_decode_failed",
                        format!(
                            "image '{}' ({}) cannot be decoded: truncated JPEG (missing end-of-image marker)",
                            im.file,
                            im.path.display()
                        ),
                    )
                    .hint("restore the original image file, or remove the corrupt image from the source"));
                }
                let decoded = folo_data::decode(&bytes).map_err(|err| {
                    let reason = match err {
                        folo_data::DataError::Image { message, .. } => message,
                        other => other.to_string(),
                    };
                    Error::new(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        "image_decode_failed",
                        format!("image '{}' ({}) cannot be decoded: {reason}", im.file, im.path.display()),
                    )
                    .hint("restore the original image file, or remove the corrupt image from the source")
                })?;
                if (decoded.width, decoded.height) != (im.width, im.height) {
                    return Err(Error::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_dataset",
                        format!(
                            "image '{}' ({}) changed size on disk: expected {}x{}, found {}x{}",
                            im.file,
                            im.path.display(),
                            im.width,
                            im.height,
                            decoded.width,
                            decoded.height
                        ),
                    )
                    .hint("restore the original image, or remove it from the source, add the new image and label it again"));
                }
                if let Some(expected) = &im.sha256
                    && digest != *expected
                {
                    return Err(Error::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_dataset",
                        format!(
                            "labeled image '{}' ({}) changed on disk since it was added to the job",
                            im.file,
                            im.path.display()
                        ),
                    )
                    .hint("restore the original image, or remove it from the job, add the new image and label it again"));
                }
                let to = images_dir.join(file);
                std::fs::write(&to, &bytes).map_err(|e| io_error("writing", &to, e))?;
                Ok(ImageRecord {
                    id: (k + 1).to_string(),
                    file: file.clone(),
                    width: im.width,
                    height: im.height,
                    bytes: bytes.len() as u64,
                    sha256: digest,
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
            // To the millisecond and later than every other benchmark's: the list is in the order they were made
            // in, also after a restart.
            created: {
                let list = self.benches.read().unwrap();
                let last = list.iter().filter_map(|b| parse_rfc3339_millis(&b.lock().file.created)).max();
                let now = (unix_now() * 1000.0) as i64;
                rfc3339_millis(last.map_or(now, |l| now.max(l + 1)))
            },
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
        if plan.is_some() {
            if let Some(e) = state.frozen_by() {
                return Err(Error::conflict(
                    "bench_frozen",
                    format!(
                        "benchmark '{id}' has a {} entry ('{}'): its plan is fixed while it has done or active \
                         entries (entries compare only on one plan)",
                        status_name(e.status),
                        e.name
                    ),
                )
                .hint(
                    "delete its done entries and stop its active ones first, or make a new benchmark with this \
                     plan",
                ));
            }
            if let Some(lp) = state.active_loop() {
                return Err(Error::conflict(
                    "bench_frozen",
                    format!(
                        "benchmark '{id}' has an active loop ('{}'): its plan is fixed while a loop is running",
                        lp.name
                    ),
                )
                .hint("stop the loop first (POST …/loops/{loop_id}/stop), or make a new benchmark with this plan"));
            }
        }
        let mut next = state.file.clone();
        if let Some(name) = name {
            next.name = name;
        }
        if let Some(d) = description {
            next.description = d;
        }
        let next_fp = if let Some(plan) = plan {
            next.plan = plan;
            Some(state::fingerprint(&next))
        } else {
            None
        };
        slot.save_file(&state, &next)?;
        state.file = next;
        if let Some(fp) = next_fp {
            state.fingerprint = fp;
        }
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
}

/// `404 image_not_found` for a benchmark's image.
fn image_not_found(bench: &str, id: &str) -> Error {
    Error::new(StatusCode::NOT_FOUND, "image_not_found", format!("benchmark '{bench}' has no image '{id}'"))
        .hint("GET /api/bench/benchmarks/{id}/images lists its images")
}

/// An image as the API shows it, its truth under `variant` when one is given.
pub(super) fn bench_image(im: &ImageRecord, variant: Option<(Condition, u8)>) -> Result<BenchImage> {
    let boxes = match variant {
        None => im.boxes.clone(),
        Some((c, s)) => conditions::moved_boxes(c, s, im.width, im.height, &im.boxes)
            .map_err(|e| Error::new(StatusCode::BAD_REQUEST, "invalid_query", e.to_string()))?,
    };
    Ok(BenchImage { id: im.id.clone(), file: im.file.clone(), width: im.width, height: im.height, boxes })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use folo_api::{
        BenchLoop, BenchLoopStatus, BenchPlan, BenchPlanCondition, BenchSource, Split, StartBenchLoop,
    };
    use folo_test_support::TestDir;

    use super::*;
    use crate::bench::state::{BenchFile, LoopState, VERSION};

    #[test]
    fn update_plan_is_frozen_by_active_loop_even_without_entries() {
        let dir = TestDir::new("bench-update-active-loop");
        let bdir = dir.join("bench").join("bn_loopfreeze");
        std::fs::create_dir_all(&bdir).unwrap();
        let file = BenchFile {
            version: VERSION,
            id: "bn_loopfreeze".into(),
            name: "b".into(),
            description: String::new(),
            classes: vec!["circle".into()],
            created: "2026-11-26T00:00:00Z".into(),
            source: BenchSource::Dataset { dataset_id: "ds_1".into(), split: Some(Split::Val) },
            source_name: "shapes".into(),
            plan: BenchPlan { conditions: Vec::new(), rationale: None, designed_by: None },
            seed: 0,
            images: Vec::new(),
        };
        std::fs::write(bdir.join("bench.json"), serde_json::to_vec(&file).unwrap()).unwrap();
        let benches = Benches::open(dir.path(), None);
        let slot = benches.slot("bn_loopfreeze").unwrap();
        let config: StartBenchLoop = serde_json::from_value(serde_json::json!({
            "teacher": "m_1",
            "pool": { "kind": "dataset", "dataset_id": "ds_1", "split": "train" },
            "student": { "epochs": 1 }
        }))
        .unwrap();
        slot.lock().loops.push(LoopState {
            lp: BenchLoop {
                id: "bl_1".into(),
                bench_id: "bn_loopfreeze".into(),
                name: "loop 1".into(),
                status: BenchLoopStatus::Running,
                phase: None,
                created: "2026-11-26T00:00:00Z".into(),
                finished: None,
                config,
                generations: Vec::new(),
                verdict: None,
                error: None,
                hint: None,
            },
            cancel: Arc::new(AtomicBool::new(false)),
        });
        let new_plan = BenchPlan {
            conditions: vec![BenchPlanCondition {
                condition: "haze".into(),
                severities: vec![1],
                weight: 1.0,
                why: None,
            }],
            rationale: None,
            designed_by: None,
        };
        let err = benches
            .update("bn_loopfreeze", UpdateBench { name: None, description: None, plan: Some(new_plan) })
            .unwrap_err();
        assert_eq!(err.body.code, "bench_frozen");
        assert!(err.body.message.contains("active loop"), "{}", err.body.message);
        benches.shutdown();
    }
}
