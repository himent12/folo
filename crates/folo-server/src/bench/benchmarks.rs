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

impl Benches {
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
}

/// `404 image_not_found` for a benchmark's image.
pub(super) fn image_not_found(bench: &str, id: &str) -> Error {
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
