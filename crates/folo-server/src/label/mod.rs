//! Agent labeling (docs/API.md, *Agent labeling*): unlabeled images, boxes drawn by a coding-agent CLI, and the
//! YOLO dataset made from them.
//!
//! [`Labeler`] is the service behind `/api/label`: it keeps the labeling jobs of a data dir (`label/<id>/`), adds
//! images to them, runs agents over them and writes datasets. The agents are behind the [`Agent`] trait:
//! [`ClaudeCode`] and [`Codex`] drive the two CLIs as child processes, one model request per call, and tests put
//! scripted agents in their place ([`Labeler::set_agents`]). How a call is kept cheap is in [`claude`] and
//! [`codex`]; what a run does when calls fail is in the `run` module.

pub mod agent;
pub mod claude;
pub mod codex;
mod dataset;
mod job;
pub mod prompt;
mod run;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use folo_api::{
    BBox, CreateLabelJob, Dataset, LabelAgent, LabelAgentId, LabelDatasetRequest, LabelEvent, LabelImage,
    LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelJob, LabelRun, LabelScope, LabelSource,
    LabelUsage, LogLevel, LogLine, RemovedImage, SkippedImage, StartLabeling, UpdateLabelImage,
    UpdateLabelJob,
};
use tokio::sync::broadcast;

pub use agent::{Agent, AgentError, AgentReply, AgentRequest};
pub use claude::ClaudeCode;
pub use codex::Codex;

use crate::error::{Error, Result};
use crate::index::{absolute, read_image_with, write_atomic};
use crate::store::ImageData;
use crate::util::{claim_dir, fnv1a, rfc3339, slug, unique_id, unix_now};
use job::{Active, Job, JobFile, Record, stem_key, unique_file_name};

/// Most images a job holds.
pub const MAX_IMAGES: usize = 10_000;
/// Most classes a job has.
pub const MAX_CLASSES: usize = 80;
/// Longest class name, in characters.
pub const MAX_CLASS_CHARS: usize = 64;
/// Longest job (and dataset) name, in characters.
pub const MAX_NAME_CHARS: usize = 100;
/// Longest instructions, in characters.
pub const MAX_INSTRUCTION_CHARS: usize = 4000;
/// Longest note of a run, in characters.
pub const MAX_NOTE_CHARS: usize = 2000;
/// Longest model name, in characters.
pub const MAX_MODEL_CHARS: usize = 100;
/// Most images in one agent call.
pub const MAX_BATCH: u32 = 16;
/// Most agent calls in flight at once.
pub const MAX_PARALLEL: u32 = 8;
/// Agent calls in flight at once when a run does not say.
pub const DEFAULT_PARALLEL: u32 = 3;
/// Images added at once beyond which subscribers get one `reload` event instead of an `image` event each.
const BULK_IMAGES: usize = 100;

/// An uploaded file: the name the client gave it and its bytes.
pub struct Upload {
    /// File name (folders in it are dropped).
    pub name: String,
    /// The file.
    pub bytes: Vec<u8>,
}

/// A dataset folder [`Labeler::write_dataset`] wrote, to be registered.
pub struct WrittenDataset {
    /// The dataset's name.
    pub name: String,
    /// Its folder.
    pub path: PathBuf,
}

/// The labeling jobs of a data dir and the agents that can work on them. See the module docs.
pub struct Labeler {
    /// `<data dir>/label`.
    root: PathBuf,
    /// `<data dir>/cache/thumbs`.
    thumbs: PathBuf,
    /// `<data dir>/datasets`.
    datasets: PathBuf,
    /// In creation order.
    jobs: RwLock<Vec<Arc<Job>>>,
    agents: Vec<Arc<dyn Agent>>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

fn io_error(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::internal(format!("{what} {}: {e}", path.display()))
}

fn path_not_found(path: &Path) -> Error {
    Error::new(StatusCode::NOT_FOUND, "path_not_found", format!("{} does not exist", path.display()))
        .hint("give a folder (or an image file) on the machine the server runs on")
}

/// A display name: trimmed, not blank, at most [`MAX_NAME_CHARS`].
fn check_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(invalid("name must not be empty"));
    }
    if name.chars().count() > MAX_NAME_CHARS {
        return Err(invalid(format!("name must be at most {MAX_NAME_CHARS} characters")));
    }
    Ok(name.to_string())
}

/// A class list: 1 to [`MAX_CLASSES`] trimmed names, distinct without case, each short and free of the characters
/// that separate classes in the agent's prompt.
fn check_classes(classes: &[String]) -> Result<Vec<String>> {
    if classes.is_empty() || classes.len() > MAX_CLASSES {
        return Err(invalid(format!("classes must have 1 to {MAX_CLASSES} names")));
    }
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(classes.len());
    for class in classes {
        let class = class.trim();
        if class.is_empty() {
            return Err(invalid("classes must not have an empty name"));
        }
        if class.chars().count() > MAX_CLASS_CHARS {
            return Err(invalid(format!("class '{class}' is longer than {MAX_CLASS_CHARS} characters")));
        }
        if class.contains([',', ':', '\n', '\r']) {
            return Err(invalid(format!(
                "class '{class}' must not contain a comma, a colon or a line break"
            )));
        }
        if !seen.insert(class.to_lowercase()) {
            return Err(invalid(format!(
                "class '{class}' is listed twice (names are compared without case)"
            )));
        }
        out.push(class.to_string());
    }
    Ok(out)
}

fn check_text(what: &str, text: &str, max: usize) -> Result<String> {
    if text.chars().count() > max {
        return Err(invalid(format!("{what} must be at most {max} characters")));
    }
    Ok(text.trim().to_string())
}

/// Hand-drawn boxes for a `width` x `height` image of a job with `classes` classes, clipped to the image.
fn check_boxes(boxes: &[BBox], classes: usize, (width, height): (u32, u32)) -> Result<Vec<BBox>> {
    if boxes.len() > prompt::MAX_BOXES {
        return Err(invalid(format!("an image takes at most {} boxes", prompt::MAX_BOXES)));
    }
    let (w, h) = (width as f32, height as f32);
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            if b.class as usize >= classes {
                return Err(invalid(format!(
                    "boxes[{i}]: class {} is not one of the job's {classes}",
                    b.class
                )));
            }
            if ![b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()) {
                return Err(invalid(format!("boxes[{i}]: coordinates must be finite numbers")));
            }
            let (x1, x2) = (b.x1.min(b.x2).clamp(0.0, w), b.x1.max(b.x2).clamp(0.0, w));
            let (y1, y2) = (b.y1.min(b.y2).clamp(0.0, h), b.y1.max(b.y2).clamp(0.0, h));
            if x2 - x1 < 1.0 || y2 - y1 < 1.0 {
                return Err(invalid(format!(
                    "boxes[{i}]: less than a pixel of it is inside the {width}x{height} image"
                )));
            }
            Ok(BBox { class: b.class, x1, y1, x2, y2 })
        })
        .collect()
}

impl Labeler {
    /// Opens the jobs under `data_dir` with the real agents (Claude Code and Codex). A job whose file cannot be
    /// read is left out (and logged), never overwritten.
    pub fn open(data_dir: &Path) -> Labeler {
        let root = data_dir.join("label");
        let mut jobs: Vec<Arc<Job>> = std::fs::read_dir(&root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("job.json").is_file())
            .filter_map(|e| match Job::load(e.path()) {
                Ok(job) => {
                    // Scratch folders of calls a previous process did not get to clean up.
                    let _ = std::fs::remove_dir_all(job.dir.join("tmp"));
                    Some(Arc::new(job))
                }
                Err(err) => {
                    tracing::warn!("skipping labeling job {}: {err}", e.path().display());
                    None
                }
            })
            .collect();
        jobs.sort_by_cached_key(|j| (j.lock().file.created.clone(), j.id.clone()));
        Labeler {
            root,
            thumbs: data_dir.join("cache").join("thumbs"),
            datasets: data_dir.join("datasets"),
            jobs: RwLock::new(jobs),
            agents: vec![Arc::new(ClaudeCode::new()), Arc::new(Codex::new())],
        }
    }

    /// Replaces the agents (tests script their own).
    pub fn set_agents(&mut self, agents: Vec<Arc<dyn Agent>>) {
        self.agents = agents;
    }

    fn job(&self, id: &str) -> Result<Arc<Job>> {
        let jobs = self.jobs.read().unwrap();
        jobs.iter().find(|j| j.id == id).cloned().ok_or_else(|| Error::not_found("label_job", id))
    }

    /// `GET /api/label/agents`.
    pub fn agents(&self) -> Vec<LabelAgent> {
        self.agents.iter().map(|a| a.info()).collect()
    }

    /// `GET /api/label/jobs`: newest first.
    pub fn list(&self) -> Vec<LabelJob> {
        self.jobs.read().unwrap().iter().rev().map(|j| j.lock().job()).collect()
    }

    /// `POST /api/label/jobs`.
    pub fn create(&self, req: CreateLabelJob) -> Result<LabelJob> {
        let name = check_name(&req.name)?;
        let classes = check_classes(&req.classes)?;
        let instructions = check_text("instructions", &req.instructions, MAX_INSTRUCTION_CHARS)?;
        // Checked before anything is created: a job is not left behind for a folder that is not there.
        let folder =
            req.path.as_deref().map(str::trim).filter(|p| !p.is_empty()).map(|p| absolute(Path::new(p)));
        if let Some(folder) = folder.as_ref().filter(|f| !f.exists()) {
            return Err(path_not_found(folder));
        }
        let job = {
            let mut jobs = self.jobs.write().unwrap();
            let id = unique_id("lb", fnv1a(name.as_bytes()) ^ unix_now().to_bits(), |id| {
                jobs.iter().any(|j| j.id == id) || self.root.join(id).exists()
            });
            let dir = self.root.join(&id);
            std::fs::create_dir_all(&dir).map_err(|e| io_error("creating", &dir, e))?;
            let job = Arc::new(Job::new(dir, JobFile::new(id, name, classes, instructions)));
            job.save(&job.lock())?;
            jobs.push(job.clone());
            job
        };
        match folder {
            Some(folder) => Ok(self.add_folder_to(&job, &folder)?.job),
            None => Ok(job.lock().job()),
        }
    }

    /// `GET /api/label/jobs/{id}`.
    pub fn get(&self, id: &str) -> Result<LabelJob> {
        Ok(self.job(id)?.lock().job())
    }

    /// `PUT /api/label/jobs/{id}`.
    pub fn update(&self, id: &str, req: UpdateLabelJob) -> Result<LabelJob> {
        let name = req.name.as_deref().map(check_name).transpose()?;
        let classes = req.classes.as_deref().map(check_classes).transpose()?;
        let instructions = req
            .instructions
            .as_deref()
            .map(|t| check_text("instructions", t, MAX_INSTRUCTION_CHARS))
            .transpose()?;
        let job = self.job(id)?;
        let mut state = job.lock();
        state.check_idle("change the job")?;
        if let Some(classes) = classes {
            let has_boxes = state.file.images.iter().any(|r| !r.boxes.is_empty());
            if has_boxes && classes.len() < state.file.classes.len() {
                return Err(invalid(format!(
                    "the job's boxes refer to its {} classes by position: classes can be renamed and appended, \
                     not removed",
                    state.file.classes.len()
                )));
            }
            state.file.classes = classes;
        }
        if let Some(name) = name {
            state.file.name = name;
        }
        if let Some(instructions) = instructions {
            state.file.instructions = instructions;
        }
        job.save(&state)?;
        job.emit_job(&state);
        Ok(state.job())
    }

    /// `DELETE /api/label/jobs/{id}`: the job's folder (uploads, labels, log) and thumbnails; never images that
    /// were added from a folder.
    pub fn delete(&self, id: &str) -> Result<()> {
        let mut jobs = self.jobs.write().unwrap();
        let job =
            jobs.iter().find(|j| j.id == id).cloned().ok_or_else(|| Error::not_found("label_job", id))?;
        {
            let mut state = job.lock();
            state.check_idle("delete the job")?;
            // Under the lock every writer takes: nothing in flight writes into the folder after this.
            state.deleted = true;
        }
        jobs.retain(|j| j.id != id);
        let _ = std::fs::remove_dir_all(self.thumbs.join(id));
        std::fs::remove_dir_all(&job.dir).map_err(|e| io_error("removing", &job.dir, e))
    }

    /// `POST /api/label/jobs/{id}/images` (multipart): stores the uploads in the job's folder.
    pub fn add_uploads(&self, id: &str, uploads: Vec<Upload>) -> Result<LabelImagesAdded> {
        static N: AtomicU64 = AtomicU64::new(0);
        let job = self.job(id)?;
        let dir = job.images_dir();
        std::fs::create_dir_all(&dir).map_err(|e| io_error("creating", &dir, e))?;
        // Written and probed before the lock is taken; named for good under it.
        let (mut staged, mut skipped) = (Vec::new(), Vec::new());
        for upload in uploads {
            let skip = |reason: &str| SkippedImage { file: upload.name.clone(), reason: reason.into() };
            if !folo_data::is_image_path(Path::new(&upload.name)) {
                skipped.push(skip("not a JPEG, PNG, BMP or WebP file"));
                continue;
            }
            let tmp =
                dir.join(format!(".upload-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            std::fs::write(&tmp, &upload.bytes).map_err(|e| io_error("writing", &tmp, e))?;
            match folo_data::probe_size(&tmp) {
                Ok(size) => staged.push(Candidate {
                    name: upload.name,
                    source: Source::Upload { tmp, hash: fnv1a(&upload.bytes) },
                    size,
                }),
                Err(_) => {
                    let _ = std::fs::remove_file(&tmp);
                    skipped.push(skip("not a readable image"));
                }
            }
        }
        self.insert(&job, staged, skipped)
    }

    /// `POST /api/label/jobs/{id}/images` (JSON): adds the images of a folder on this machine, in place.
    pub fn add_folder(&self, id: &str, path: &str) -> Result<LabelImagesAdded> {
        let job = self.job(id)?;
        let folder = absolute(Path::new(path.trim()));
        if path.trim().is_empty() || !folder.exists() {
            return Err(path_not_found(&folder));
        }
        self.add_folder_to(&job, &folder)
    }

    fn add_folder_to(&self, job: &Job, folder: &Path) -> Result<LabelImagesAdded> {
        let known: HashSet<PathBuf> =
            job.lock().file.images.iter().filter_map(|r| r.external.clone()).collect();
        let room = MAX_IMAGES.saturating_sub(job.lock().file.images.len());
        let files = walkdir::WalkDir::new(folder)
            .sort_by_file_name()
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && folo_data::is_image_path(e.path()))
            .map(|e| e.into_path());
        let (mut staged, mut skipped, mut over) = (Vec::new(), Vec::new(), 0usize);
        for path in files {
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if known.contains(&path) {
                skipped.push(SkippedImage { file: name, reason: "the job already has this file".into() });
            } else if staged.len() >= room {
                over += 1;
            } else {
                match folo_data::probe_size(&path) {
                    Ok(size) => staged.push(Candidate { name, source: Source::Folder(path), size }),
                    Err(_) => {
                        skipped.push(SkippedImage { file: name, reason: "not a readable image".into() })
                    }
                }
            }
        }
        if over > 0 {
            skipped.push(SkippedImage {
                file: folder.display().to_string(),
                reason: format!("{over} more images were left out: a job holds at most {MAX_IMAGES}"),
            });
        }
        self.insert(job, staged, skipped)
    }

    /// Adds probed images to the job under its lock: duplicates and images past the limit go to `skipped`.
    fn insert(
        &self,
        job: &Job,
        staged: Vec<Candidate>,
        mut skipped: Vec<SkippedImage>,
    ) -> Result<LabelImagesAdded> {
        let mut state = job.lock();
        let mut taken: HashSet<String> = state.file.images.iter().map(|r| stem_key(&r.file)).collect();
        let first_new = state.file.images.len();
        for candidate in staged {
            let Candidate { name, source, size: (width, height) } = candidate;
            let duplicate = match &source {
                Source::Upload { hash, .. } => state.file.images.iter().find(|r| r.hash == Some(*hash)),
                Source::Folder(path) => state.file.images.iter().find(|r| r.external.as_ref() == Some(path)),
            };
            let reason = if state.deleted {
                Some("the job was deleted".to_string())
            } else if let Some(same) = duplicate {
                Some(format!("the job already has this image ({})", same.file))
            } else if state.file.images.len() >= MAX_IMAGES {
                Some(format!("a job holds at most {MAX_IMAGES} images"))
            } else {
                None
            };
            if let Some(reason) = reason {
                source.discard();
                skipped.push(SkippedImage { file: name, reason });
                continue;
            }
            let file = unique_file_name(&name, &mut taken);
            let (external, hash) = match source {
                Source::Folder(path) => (Some(path), None),
                Source::Upload { tmp, hash } => {
                    let to = job.images_dir().join(&file);
                    std::fs::rename(&tmp, &to).map_err(|e| io_error("storing", &to, e))?;
                    (None, Some(hash))
                }
            };
            let id = state.file.next_image.to_string();
            state.file.next_image += 1;
            state.file.images.push(Record {
                id,
                file,
                external,
                hash,
                width,
                height,
                status: LabelImageStatus::Pending,
                boxes: Vec::new(),
                source: None,
                error: None,
            });
        }
        let added: Vec<LabelImage> = state.file.images[first_new..].iter().map(|r| state.image(r)).collect();
        if !added.is_empty() {
            job.save(&state)?;
            if added.len() > BULK_IMAGES {
                job.emit(LabelEvent::Reload);
            } else {
                added.iter().cloned().for_each(|image| job.emit(LabelEvent::Image(image)));
            }
            job.emit_job(&state);
        }
        Ok(LabelImagesAdded { added, skipped, job: state.job() })
    }

    /// `GET /api/label/jobs/{id}/images`.
    pub fn images(
        &self,
        id: &str,
        status: Option<LabelImageStatus>,
        offset: u64,
        limit: u64,
    ) -> Result<LabelImagePage> {
        let job = self.job(id)?;
        let state = job.lock();
        let matching =
            state.file.images.iter().map(|r| state.image(r)).filter(|i| status.is_none_or(|s| s == i.status));
        let (mut total, mut images) = (0u64, Vec::new());
        for image in matching {
            if total >= offset && (images.len() as u64) < limit {
                images.push(image);
            }
            total += 1;
        }
        Ok(LabelImagePage { total, images })
    }

    /// `GET /api/label/jobs/{id}/images/{image_id}`: the file's bytes, or a JPEG thumbnail whose long side is
    /// `thumb` (cached under the data dir's `cache/thumbs/<job id>`).
    pub fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        let job = self.job(id)?;
        let path = {
            let state = job.lock();
            job.image_path(state.record(image_id)?)
        };
        let cache = thumb.map(|_| self.thumbs.join(id));
        read_image_with(&path, thumb, cache.as_deref(), |file, bytes| {
            // Not after the job was deleted (its thumbnails went with it).
            if !job.lock().deleted
                && let Err(e) = write_atomic(file, bytes)
            {
                tracing::warn!("caching thumbnail {}: {e}", file.display());
            }
        })
    }

    /// `PUT /api/label/jobs/{id}/images/{image_id}`: boxes drawn by hand.
    pub fn update_image(&self, id: &str, image_id: &str, req: UpdateLabelImage) -> Result<LabelImage> {
        let job = self.job(id)?;
        let mut state = job.lock();
        state.check_free(image_id)?;
        let classes = state.file.classes.len();
        let record = state.record(image_id)?;
        let boxes = check_boxes(&req.boxes, classes, (record.width, record.height))?;
        let record = state.record_mut(image_id).expect("the record was just read");
        record.boxes = boxes;
        record.status = LabelImageStatus::Labeled;
        record.source = Some(LabelSource::Manual);
        record.error = None;
        job.save(&state)?;
        let image = state.image(state.record(image_id)?);
        job.emit(LabelEvent::Image(image.clone()));
        job.emit_job(&state);
        Ok(image)
    }

    /// `DELETE /api/label/jobs/{id}/images/{image_id}`: an upload's file goes too, a folder image's never.
    pub fn delete_image(&self, id: &str, image_id: &str) -> Result<()> {
        let job = self.job(id)?;
        let mut state = job.lock();
        state.check_free(image_id)?;
        let record = state.record(image_id)?.clone();
        state.file.images.retain(|r| r.id != image_id);
        job.save(&state)?;
        if record.external.is_none() {
            let _ = std::fs::remove_file(job.image_path(&record));
        }
        job.emit(LabelEvent::Removed(RemovedImage { id: record.id }));
        job.emit_job(&state);
        Ok(())
    }

    /// `POST /api/label/jobs/{id}/start`: starts an agent run on its own thread.
    pub fn start(&self, id: &str, req: StartLabeling) -> Result<LabelJob> {
        let note = check_text("note", &req.note, MAX_NOTE_CHARS)?;
        let model = req.model.as_deref().map(str::trim).filter(|m| !m.is_empty());
        if let Some(m) = model
            && (m.chars().count() > MAX_MODEL_CHARS
                || !m.chars().all(|c| c.is_ascii_alphanumeric() || "._:/-[]".contains(c)))
        {
            return Err(invalid(format!(
                "model must be at most {MAX_MODEL_CHARS} characters of A-Z a-z 0-9 . _ : / - [ ]"
            )));
        }
        if req.batch.is_some_and(|b| !(1..=MAX_BATCH).contains(&b)) {
            return Err(invalid(format!("batch must be in 1..={MAX_BATCH}")));
        }
        if req.parallel.is_some_and(|p| !(1..=MAX_PARALLEL).contains(&p)) {
            return Err(invalid(format!("parallel must be in 1..={MAX_PARALLEL}")));
        }
        let job = self.job(id)?;
        let agent = self.agent(req.agent)?;
        let model = model.map(str::to_string).or(agent.info().default_model);

        let mut state = job.lock();
        state.check_idle("start another run")?;
        let ids = match &req.image_ids {
            Some(ids) => {
                let mut seen = HashSet::new();
                for id in ids {
                    state
                        .record(id)
                        .map_err(|_| invalid(format!("image_ids: the job has no image '{id}'")))?;
                }
                ids.iter().filter(|id| seen.insert(id.as_str())).cloned().collect()
            }
            None => state
                .file
                .images
                .iter()
                .filter(|r| req.scope == LabelScope::All || r.status != LabelImageStatus::Labeled)
                .map(|r| r.id.clone())
                .collect::<Vec<_>>(),
        };
        if ids.is_empty() {
            let why =
                if state.file.images.is_empty() { "the job has no images" } else { "every image is labeled" };
            return Err(invalid(format!("nothing to label: {why}"))
                .hint("add images, or send \"scope\": \"all\" to label every image again"));
        }
        let batch = req.batch.map_or(agent.default_batch(), |b| b as usize);
        let plan = run::Plan {
            agent: agent.clone(),
            model: model.clone(),
            effort: req.effort,
            note,
            batches: ids.chunks(batch).map(<[String]>::to_vec).collect(),
            parallel: req.parallel.unwrap_or(DEFAULT_PARALLEL) as usize,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        state.file.run = Some(LabelRun {
            agent: req.agent,
            model: model.clone(),
            effort: req.effort,
            started: rfc3339(unix_now()),
            finished: None,
            total: ids.len() as u64,
            done: 0,
            failed: 0,
            outcome: None,
            error: None,
            hint: None,
            usage: LabelUsage::default(),
        });
        let calls = plan.batches.len();
        let message = format!(
            "{} ({}, {} effort) starts on {} {}: {calls} {} of up to {batch}",
            req.agent.display_name(),
            model.as_deref().unwrap_or("its default model"),
            req.effort.as_str(),
            ids.len(),
            if ids.len() == 1 { "image" } else { "images" },
            if calls == 1 { "call" } else { "calls" },
        );
        state.active = Some(Active {
            stopping: false,
            cancel: plan.cancel.clone(),
            waiting: ids.into_iter().collect(),
            in_call: HashSet::new(),
            thread: None,
        });
        job.log(&mut state, LogLevel::Info, message);
        job.save(&state)?;
        // The thread's first step takes the job's lock, so it starts working once this call has let go of it.
        let worker = job.clone();
        let thread = std::thread::Builder::new()
            .name(format!("folo-label-{id}"))
            .spawn(move || run::run(&worker, &plan))
            .map_err(|e| {
                state.active = None;
                state.file.run = None;
                Error::internal(format!("could not start the run's thread: {e}"))
            })?;
        if let Some(active) = state.active.as_mut() {
            active.thread = Some(thread);
        }
        job.emit_job(&state);
        Ok(state.job())
    }

    /// The agent `id`, installed: `501 not_supported` with how to get it otherwise.
    fn agent(&self, id: LabelAgentId) -> Result<Arc<dyn Agent>> {
        let unsupported = |message: String| Error::new(StatusCode::NOT_IMPLEMENTED, "not_supported", message);
        let Some(agent) = self.agents.iter().find(|a| a.id() == id) else {
            return Err(unsupported(format!("this server has no {} agent", id.display_name())));
        };
        let info = agent.info();
        if !info.available {
            let e = unsupported(format!("{} is not installed on the server's machine", info.name));
            return Err(match info.hint {
                Some(hint) => e.hint(hint),
                None => e,
            });
        }
        Ok(agent.clone())
    }

    /// `POST /api/label/jobs/{id}/stop`: ends the run's calls; the run's thread closes the run.
    pub fn stop(&self, id: &str) -> Result<LabelJob> {
        let job = self.job(id)?;
        let mut state = job.lock();
        let Some(active) = state.active.as_mut() else {
            return Err(Error::conflict(
                "label_job_not_active",
                format!("labeling job '{id}' has no agent run"),
            )
            .hint("nothing to stop; GET /api/label/jobs/{id} has the last run's result"));
        };
        if !active.stopping {
            active.stopping = true;
            active.cancel.store(true, Ordering::Relaxed);
            job.log(&mut state, LogLevel::Info, "stop requested");
            job.emit_job(&state);
        }
        Ok(state.job())
    }

    /// `GET /api/label/jobs/{id}/events`: the job now, and its events from now on.
    pub fn subscribe(&self, id: &str) -> Result<(LabelJob, broadcast::Receiver<LabelEvent>)> {
        let job = self.job(id)?;
        // Subscribed under the lock every change is made under, so no event falls between the two.
        let state = job.lock();
        Ok((state.job(), job.subscribe()))
    }

    /// `GET /api/label/jobs/{id}/logs`: the last `limit` lines, oldest first.
    pub fn logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>> {
        Ok(self.job(id)?.lock().logs(limit))
    }

    /// Writes the job's labeled images as a YOLO dataset into a new folder under `<data dir>/datasets`
    /// (the first half of `POST /api/label/jobs/{id}/dataset`; the store registers it).
    pub fn write_dataset(&self, id: &str, req: &LabelDatasetRequest) -> Result<WrittenDataset> {
        if !(0.0..=0.5).contains(&req.val_fraction) {
            return Err(invalid("val_fraction must be in 0..=0.5"));
        }
        let job = self.job(id)?;
        let (name, classes, items) = {
            let state = job.lock();
            state.check_idle("make the dataset")?;
            let name = match req.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
                Some(name) => check_name(name)?,
                None => state.file.name.clone(),
            };
            let items: Vec<dataset::Item> = state
                .file
                .images
                .iter()
                .filter(|r| r.status == LabelImageStatus::Labeled)
                .map(|r| dataset::Item {
                    source: job.image_path(r),
                    file: r.file.clone(),
                    width: r.width,
                    height: r.height,
                    boxes: r.boxes.clone(),
                })
                .collect();
            (name, state.file.classes.clone(), items)
        };
        if items.is_empty() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_dataset",
                format!("labeling job '{id}' has no labeled image yet"),
            )
            .hint("label images first: POST /api/label/jobs/{id}/start, or draw boxes by hand"));
        }
        let path = claim_dir(&self.datasets, &slug(&name, "labeled"))
            .map_err(|e| io_error("creating a folder in", &self.datasets, e))?;
        if let Err(e) = dataset::write(&path, &name, &classes, &items, req.val_fraction, req.seed) {
            let _ = std::fs::remove_dir_all(&path);
            return Err(io_error("writing the dataset", &path, e));
        }
        Ok(WrittenDataset { name, path })
    }

    /// Remembers the dataset made from job `id` (the second half of `POST /api/label/jobs/{id}/dataset`).
    pub fn set_dataset(&self, id: &str, dataset: &Dataset) -> Result<()> {
        let job = self.job(id)?;
        let mut state = job.lock();
        state.file.dataset_id = Some(dataset.id.clone());
        let message = format!(
            "dataset '{}' made: {} train and {} val images",
            dataset.name, dataset.splits.train, dataset.splits.val
        );
        job.log(&mut state, LogLevel::Info, message);
        job.save(&state)?;
        job.emit_job(&state);
        Ok(())
    }

    /// Ends every active run (their agent calls are killed) and waits for them to close.
    pub fn shutdown(&self) {
        let jobs = self.jobs.read().unwrap().clone();
        let threads: Vec<_> = jobs
            .iter()
            .filter_map(|job| {
                let mut state = job.lock();
                let active = state.active.as_mut()?;
                active.stopping = true;
                active.cancel.store(true, Ordering::Relaxed);
                active.thread.take()
            })
            .collect();
        for thread in threads {
            let _ = thread.join();
        }
    }
}

/// An image about to be added to a job.
struct Candidate {
    name: String,
    source: Source,
    size: (u32, u32),
}

enum Source {
    /// A file added in place.
    Folder(PathBuf),
    /// An upload, written to a temporary file in the job's `images/`.
    Upload { tmp: PathBuf, hash: u64 },
}

impl Source {
    /// Drops an upload that is not added after all.
    fn discard(&self) {
        if let Source::Upload { tmp, .. } = self {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_classes_and_boxes_are_checked() {
        assert_eq!(check_name("  street  ").unwrap(), "street");
        assert!(check_name(" ").is_err() && check_name(&"x".repeat(MAX_NAME_CHARS + 1)).is_err());

        let classes =
            |names: &[&str]| check_classes(&names.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(classes(&[" car ", "Traffic light"]).unwrap(), ["car", "Traffic light"]);
        for bad in [&[][..], &["car", "Car"], &["a,b"], &["a: b"], &[""], &["a\nb"]] {
            assert_eq!(classes(bad).unwrap_err().body.code, "invalid_body", "{bad:?}");
        }
        assert!(check_classes(&vec!["c".to_string(); 1]).is_ok());
        let many: Vec<String> = (0..=MAX_CLASSES).map(|i| format!("c{i}")).collect();
        assert!(check_classes(&many).is_err() && check_classes(&many[..MAX_CLASSES]).is_ok());

        let b = |class, x1, y1, x2, y2| BBox { class, x1, y1, x2, y2 };
        // Clipped to the image, corners put in order.
        let ok =
            check_boxes(&[b(1, 90.0, 40.0, -5.0, 10.0), b(0, 0.0, 0.0, 500.0, 500.0)], 2, (100, 50)).unwrap();
        assert_eq!(ok, [b(1, 0.0, 10.0, 90.0, 40.0), b(0, 0.0, 0.0, 100.0, 50.0)]);
        for bad in [b(2, 0.0, 0.0, 9.0, 9.0), b(0, f32::NAN, 0.0, 9.0, 9.0), b(0, 100.0, 0.0, 150.0, 9.0)] {
            let e = check_boxes(&[bad], 2, (100, 50)).unwrap_err();
            assert!(e.body.code == "invalid_body" && e.body.message.starts_with("boxes[0]"), "{e}");
        }
    }
}
