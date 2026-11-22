//! Agent labeling (docs/API.md, *Agent labeling*): unlabeled images, boxes drawn by a coding-agent CLI, and the
//! YOLO dataset made from them.
//!
//! [`Labeler`] is the service behind `/api/label`: it keeps the labeling jobs of a data dir (`label/<id>/`), adds
//! images to them, runs agents over them and writes datasets. The agents are behind the [`Agent`] trait:
//! [`ClaudeCode`], [`Codex`] and [`Antigravity`] drive the CLIs as child processes, one model request per call, and
//! tests put scripted agents in their place ([`Labeler::set_agents`]). How a call is kept cheap is in [`claude`],
//! [`codex`] and [`antigravity`]; what a run does when calls fail is in the `run` module.
//!
//! Files: this one holds [`Labeler`]; `job` (one job: `job.json`, its state and events), `images` (adding, listing,
//! editing and removing a job's images), `labeling` (starting and stopping a run), `run` (a run, batch by batch),
//! [`agent`] (the [`Agent`] trait and running a CLI with a deadline), [`claude`], [`codex`] and [`antigravity`] (the
//! agents), `ask` (one batch's calls: at most two, whatever made the first fail), `faults` (what a CLI's failure says:
//! a passing fault or a limit), `tree` (a child process with everything it starts), [`prompt`] (what an agent is asked
//! and how its reply is read), `validate` (checks of what clients send), `export` and `dataset` (the YOLO dataset a
//! job becomes).

pub mod agent;
pub mod antigravity;
mod ask;
pub mod claude;
pub mod codex;
pub(crate) mod dataset;
mod export;
mod faults;
mod images;
pub(crate) mod job;
mod labeling;
pub mod prompt;
mod run;
mod tree;
mod validate;

pub(crate) use run::MAX_SOFT_FAILURES;
pub use run::{AGENT_IMAGE_SIDE, AgentImage, TRANSIENT_BACKOFF, agent_jpeg, agent_jpeg_of};

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use axum::http::StatusCode;
use folo_api::{BBox, CreateLabelJob, LabelAgent, LabelEvent, LabelJob, LogLine, UpdateLabelJob};
use tokio::sync::broadcast;

pub use agent::{Agent, AgentError, AgentReply, AgentRequest};
pub use antigravity::Antigravity;
pub use claude::ClaudeCode;
pub use codex::Codex;
pub use images::LabelImageQuery;

use crate::error::{Error, Result};
use crate::index::absolute;
use crate::util::{fnv1a, unique_id, unix_now};
use job::{Exporting, Job, JobFile};
use validate::check_classes;
pub(crate) use validate::{check_model, check_name, check_text};

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

/// A dataset folder [`Labeler::write_dataset`] wrote, to be registered. While it lives, its job stays as it is
/// (no run, no change, no deletion), so [`Labeler::set_dataset`] finds the job the dataset was made from.
pub struct WrittenDataset {
    /// The dataset's name.
    pub name: String,
    /// Its folder.
    pub path: PathBuf,
    export: Exporting,
}

/// The images of a job as [`Labeler::job_images`] hands them out.
pub struct JobImages {
    /// The job's name.
    pub name: String,
    /// Its classes.
    pub classes: Vec<String>,
    /// The images, in the order they were added.
    pub images: Vec<JobImage>,
}

/// One image of [`JobImages`].
pub struct JobImage {
    /// Its file name in the job (stems unique).
    pub file: String,
    /// Where the file is.
    pub path: PathBuf,
    /// Width after EXIF orientation.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Its boxes when it is labeled, pixel `xyxy`; empty otherwise.
    pub boxes: Vec<BBox>,
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
    /// How often a run writes `job.json` at most ([`run::SAVE_EVERY`]).
    save_every: Duration,
    /// How long a call that failed on a passing fault waits before it is asked again ([`run::TRANSIENT_BACKOFF`]).
    backoff: Duration,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

fn io_error(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error::internal(format!("{what} {}: {e}", path.display()))
}

/// A folder (or image) path as people paste it: surrounding whitespace and one pair of matching quotes (Windows'
/// "Copy as path") are not part of it; None when nothing is left. A relative path is refused rather than read from
/// wherever the server was started, which would name a folder the user never typed.
fn folder_path(raw: &str) -> Result<Option<PathBuf>> {
    let p = raw.trim();
    let p = ['"', '\''].iter().find_map(|q| p.strip_prefix(*q)?.strip_suffix(*q)).unwrap_or(p).trim();
    if p.is_empty() {
        return Ok(None);
    }
    if !Path::new(p).is_absolute() {
        return Err(invalid(format!("path must be a full path: {p} is relative"))
            .hint("give the full path of a folder (or an image file) on the machine the server runs on, such as D:/photos"));
    }
    Ok(Some(absolute(Path::new(p))))
}

fn path_not_found(path: &Path) -> Error {
    Error::new(StatusCode::NOT_FOUND, "path_not_found", format!("{} does not exist", path.display()))
        .hint("give a folder (or an image file) on the machine the server runs on")
}

impl Labeler {
    /// Opens the jobs under `data_dir` with the real agents (Claude Code, Codex and Antigravity). A job whose file
    /// cannot be read is left out (and logged), never overwritten.
    pub fn open(data_dir: &Path) -> Labeler {
        let root = data_dir.join("label");
        // A copy of a job folder is not a second job of its id: it is skipped before loading (which closes an
        // interrupted run and writes the job back) and is not swept.
        // A copy that reads but does not load (its recovery cannot be written back) gives way to the next.
        let folders = crate::util::folders_by_id(&root, "job.json", Job::readable);
        let mut jobs: Vec<Arc<Job>> = crate::util::load_first("labeling job", folders, |dir| {
            Job::load(dir.to_path_buf()).map(|job| {
                job.sweep();
                Arc::new(job)
            })
        });
        jobs.sort_by_cached_key(|j| (j.lock().file.created.clone(), j.id.clone()));
        Labeler {
            root,
            thumbs: data_dir.join("cache").join("thumbs"),
            datasets: data_dir.join("datasets"),
            jobs: RwLock::new(jobs),
            agents: vec![Arc::new(ClaudeCode::new()), Arc::new(Codex::new()), Arc::new(Antigravity::new())],
            save_every: run::SAVE_EVERY,
            backoff: run::TRANSIENT_BACKOFF,
        }
    }

    /// Replaces the agents (tests script their own).
    pub fn set_agents(&mut self, agents: Vec<Arc<dyn Agent>>) {
        self.agents = agents;
    }

    /// Changes how often a run writes `job.json` at most (tests shorten it).
    pub fn set_save_every(&mut self, every: Duration) {
        self.save_every = every;
    }

    /// Changes how long a call that failed on a passing fault waits before it is asked again (tests shorten it).
    pub fn set_transient_backoff(&mut self, backoff: Duration) {
        self.backoff = backoff;
    }

    /// How long a call that failed on a passing fault waits before it is asked again: [`TRANSIENT_BACKOFF`],
    /// unless a test shortened it. Benchmark agent entries wait as long.
    pub fn transient_backoff(&self) -> Duration {
        self.backoff
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
        let folder = req.path.as_deref().map(folder_path).transpose()?.flatten();
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
        Ok(self.job(id)?.live()?.job())
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
        let mut state = job.live()?;
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
        let job = self.job(id)?;
        {
            let mut state = job.live()?;
            state.check_idle("delete the job")?;
            // The job's file goes first: without it the folder is not a job, whatever else of it is still there.
            let file = job.dir.join("job.json");
            match std::fs::remove_file(&file) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    return Err(io_error("removing", &file, e));
                }
                _ => {}
            }
            // Under the lock every writer takes: nothing in flight writes into the folder after this.
            state.deleted = true;
        }
        self.jobs.write().unwrap().retain(|j| j.id != id);
        // The folders go last and outside every lock: a job with thousands of uploads takes a while to remove.
        let _ = std::fs::remove_dir_all(self.thumbs.join(id));
        if let Err(e) = std::fs::remove_dir_all(&job.dir) {
            // A file of it is open somewhere (an image being served): what is left is not listed again.
            tracing::warn!("removing {}: {e}", job.dir.display());
        }
        Ok(())
    }

    /// `GET /api/label/jobs/{id}/events`: the job now, and its events from now on.
    pub fn subscribe(&self, id: &str) -> Result<(LabelJob, broadcast::Receiver<LabelEvent>)> {
        let job = self.job(id)?;
        // Subscribed under the lock every change is made under, so no event falls between the two.
        let state = job.live()?;
        Ok((state.job(), job.subscribe()))
    }

    /// `GET /api/label/jobs/{id}/logs`: the last `limit` lines, oldest first.
    pub fn logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>> {
        Ok(self.job(id)?.live()?.logs(limit))
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
