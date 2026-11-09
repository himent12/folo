//! One labeling job: its file on disk (`job.json`), its state in memory and the events it sends.

use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use folo_api::{
    BBox, LabelCounts, LabelEvent, LabelImage, LabelImageStatus, LabelJob, LabelJobStatus, LabelRun,
    LabelSource, LabelUsage, LogLevel, LogLine,
};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::util::{rfc3339, unix_now};

/// Version of `job.json`'s layout.
const VERSION: u32 = 1;
/// Events a slow subscriber may fall behind by before it is told to reload.
const EVENT_BUFFER: usize = 4096;
/// Log lines kept in memory for `GET /api/label/jobs/{id}/logs` (the file keeps them all).
const LOG_LINES: usize = 2000;

/// An image of a job as stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    /// File name, unique within the job (and so is its stem, which names the label file of a dataset).
    pub file: String,
    /// The file's absolute path when it was added from a folder; uploads live in the job's `images/` as `file`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external: Option<PathBuf>,
    /// FNV-1a of an upload's bytes, so the same upload is not added twice.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<u64>,
    pub width: u32,
    pub height: u32,
    /// `pending`, `labeled` or `failed` (`labeling` is a view of an image in a call, never stored).
    pub status: LabelImageStatus,
    pub boxes: Vec<BBox>,
    pub source: Option<LabelSource>,
    pub error: Option<String>,
}

/// `job.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobFile {
    pub version: u32,
    pub id: String,
    pub name: String,
    pub classes: Vec<String>,
    pub instructions: String,
    pub created: String,
    /// The id the next image gets (ids are never reused).
    pub next_image: u64,
    pub images: Vec<Record>,
    pub run: Option<LabelRun>,
    pub usage: LabelUsage,
    pub dataset_id: Option<String>,
}

impl JobFile {
    /// A new job without images.
    pub fn new(id: String, name: String, classes: Vec<String>, instructions: String) -> JobFile {
        JobFile {
            version: VERSION,
            id,
            name,
            classes,
            instructions,
            created: rfc3339(unix_now()),
            next_image: 1,
            images: Vec::new(),
            run: None,
            usage: LabelUsage::default(),
            dataset_id: None,
        }
    }
}

/// The agent run that is working on a job.
pub struct Active {
    /// A stop was requested.
    pub stopping: bool,
    /// Ends the calls in flight and keeps new ones from starting.
    pub cancel: Arc<AtomicBool>,
    /// Images of the run no call has taken yet (one changed by hand meanwhile leaves: [`State::claim`]).
    pub waiting: HashSet<String>,
    /// Images in a call right now.
    pub in_call: HashSet<String>,
    /// The run's thread, for shutdown to wait on.
    pub thread: Option<std::thread::JoinHandle<()>>,
}

/// A job's state; everything in it changes under [`Job::lock`].
pub struct State {
    pub file: JobFile,
    pub active: Option<Active>,
    logs: VecDeque<LogLine>,
    /// Set when the job is deleted, so work still in flight writes nothing back.
    pub deleted: bool,
}

impl State {
    /// The job as the API shows it.
    pub fn job(&self) -> LabelJob {
        let mut counts = LabelCounts { images: self.file.images.len() as u64, ..Default::default() };
        for r in &self.file.images {
            match r.status {
                // Only a labeled image's boxes count: one whose relabeling failed keeps its old boxes, but they
                // are not labels (and go into no dataset) until it is labeled again.
                LabelImageStatus::Labeled => {
                    counts.labeled += 1;
                    counts.boxes += r.boxes.len() as u64;
                }
                LabelImageStatus::Failed => counts.failed += 1,
                _ => counts.pending += 1,
            }
        }
        LabelJob {
            id: self.file.id.clone(),
            name: self.file.name.clone(),
            classes: self.file.classes.clone(),
            instructions: self.file.instructions.clone(),
            created: self.file.created.clone(),
            status: match &self.active {
                None => LabelJobStatus::Idle,
                Some(a) if a.stopping => LabelJobStatus::Stopping,
                Some(_) => LabelJobStatus::Running,
            },
            counts,
            run: self.file.run.clone(),
            usage: self.file.usage,
            dataset_id: self.file.dataset_id.clone(),
        }
    }

    /// An image's status as the API shows it: `labeling` while it is in a call.
    pub fn status(&self, r: &Record) -> LabelImageStatus {
        match &self.active {
            Some(a) if a.in_call.contains(&r.id) => LabelImageStatus::Labeling,
            _ => r.status,
        }
    }

    /// An image as the API shows it.
    pub fn image(&self, r: &Record) -> LabelImage {
        LabelImage {
            id: r.id.clone(),
            file: r.file.clone(),
            width: r.width,
            height: r.height,
            status: self.status(r),
            boxes: r.boxes.clone(),
            source: r.source,
            error: r.error.clone(),
        }
    }

    /// The stored image `image_id`.
    pub fn record(&self, image_id: &str) -> Result<&Record> {
        self.file.images.iter().find(|r| r.id == image_id).ok_or_else(|| Error::not_found("image", image_id))
    }

    /// The stored image `image_id`, to change.
    pub fn record_mut(&mut self, image_id: &str) -> Option<&mut Record> {
        self.file.images.iter_mut().find(|r| r.id == image_id)
    }

    /// `409 label_job_active` while a run is active.
    pub fn check_idle(&self, doing: &str) -> Result<()> {
        match &self.active {
            None => Ok(()),
            Some(_) => Err(Error::conflict(
                "label_job_active",
                format!("labeling job '{}' has an agent run in progress; {doing} once it ends", self.file.id),
            )
            .hint("POST /api/label/jobs/{id}/stop ends the run")),
        }
    }

    /// Takes image `image_id` for a change by hand. `409 label_job_active` while it is in an agent call; an image
    /// that only waits for one leaves the run instead (the run then has one image less to label), so work done by
    /// hand is never overwritten by a call that had not started.
    pub fn claim(&mut self, image_id: &str) -> Result<()> {
        let Some(active) = self.active.as_mut() else { return Ok(()) };
        if active.in_call.contains(image_id) {
            return Err(Error::conflict(
                "label_job_active",
                format!("image '{image_id}' is with the agent right now; change it once that call is done"),
            )
            .hint("the image is `labeling`; POST /api/label/jobs/{id}/stop ends the run"));
        }
        if active.waiting.remove(image_id)
            && let Some(run) = self.file.run.as_mut()
        {
            run.total = run.total.saturating_sub(1);
        }
        Ok(())
    }

    /// The last `limit` log lines, oldest first.
    pub fn logs(&self, limit: usize) -> Vec<LogLine> {
        self.logs.iter().skip(self.logs.len().saturating_sub(limit)).cloned().collect()
    }
}

/// A job in memory.
pub struct Job {
    pub id: String,
    /// `<data dir>/label/<id>`.
    pub dir: PathBuf,
    state: Mutex<State>,
    events: broadcast::Sender<LabelEvent>,
}

impl Job {
    /// A job around `file`, with the log `dir` already holds.
    pub fn new(dir: PathBuf, file: JobFile) -> Job {
        let logs = std::fs::read_to_string(dir.join("log.jsonl")).unwrap_or_default();
        let mut lines: VecDeque<LogLine> =
            logs.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
        lines.drain(..lines.len().saturating_sub(LOG_LINES));
        Job {
            id: file.id.clone(),
            dir,
            state: Mutex::new(State { file, active: None, logs: lines, deleted: false }),
            events: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    /// Reads the job in `dir`. A run its file still shows as active belonged to a process that is gone: it is
    /// closed as stopped.
    pub fn load(dir: PathBuf) -> Result<Job> {
        let path = dir.join("job.json");
        let bad = |e: &dyn std::fmt::Display| Error::internal(format!("reading {}: {e}", path.display()));
        let bytes = std::fs::read(&path).map_err(|e| bad(&e))?;
        let mut file: JobFile = serde_json::from_slice(&bytes).map_err(|e| bad(&e))?;
        if file.version != VERSION {
            return Err(bad(&format!("version {} is not one this server reads", file.version)));
        }
        let interrupted = file.run.as_mut().filter(|r| r.finished.is_none()).map(|run| {
            run.finished = Some(rfc3339(unix_now()));
            run.outcome = Some(folo_api::LabelOutcome::Stopped);
        });
        let job = Job::new(dir, file);
        if interrupted.is_some() {
            let mut state = job.lock();
            job.log(
                &mut state,
                LogLevel::Warn,
                "the server stopped during the agent run; start it again to continue",
            );
            job.save(&state)?;
        }
        Ok(job)
    }

    /// The job's state. Held briefly: never across an agent call or a decode. A panic under the lock does not
    /// take the job down with it: every change is a few plain assignments, so the state stays usable.
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Where an image's file is.
    pub fn image_path(&self, r: &Record) -> PathBuf {
        r.external.clone().unwrap_or_else(|| self.images_dir().join(&r.file))
    }

    /// The folder of uploaded images.
    pub fn images_dir(&self) -> PathBuf {
        self.dir.join("images")
    }

    /// Writes `job.json` (atomically), unless the job was deleted.
    pub fn save(&self, state: &State) -> Result<()> {
        if state.deleted {
            return Ok(());
        }
        let path = self.dir.join("job.json");
        let bytes = serde_json::to_vec(&state.file).expect("jobs serialize");
        write_atomic(&path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))
    }

    /// Sends `event` to the subscribers there are.
    pub fn emit(&self, event: LabelEvent) {
        let _ = self.events.send(event);
    }

    /// Sends the job as it is now.
    pub fn emit_job(&self, state: &State) {
        self.emit(LabelEvent::Job(Box::new(state.job())));
    }

    /// A subscription to the job's events.
    pub fn subscribe(&self) -> broadcast::Receiver<LabelEvent> {
        self.events.subscribe()
    }

    /// Adds a line to the job's log: memory, `log.jsonl` and the subscribers.
    pub fn log(&self, state: &mut State, level: LogLevel, message: impl Into<String>) {
        let line = LogLine { t: rfc3339(unix_now()), level, message: message.into() };
        if !state.deleted {
            let appended = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(self.dir.join("log.jsonl"))
                .and_then(|mut f| {
                    writeln!(f, "{}", serde_json::to_string(&line).expect("log lines serialize"))
                });
            if let Err(e) = appended {
                tracing::warn!(job = %self.id, "writing the job log: {e}");
            }
        }
        if state.logs.len() == LOG_LINES {
            state.logs.pop_front();
        }
        state.logs.push_back(line.clone());
        self.emit(LabelEvent::Log(line));
    }
}

/// A file name for an image called `name` that no image of the job has yet: `name` without folders, reduced to
/// letters, digits, `.`, `_` and `-`, with `-2`, `-3`, ... on its stem while that stem is in `taken` (lower-cased
/// stems; the new one is added). Stems are unique, not just names, because a dataset's label files are named by
/// stem (`a.jpg` and `a.png` would share `a.txt`), and without case, because Windows folders are.
pub fn unique_file_name(name: &str, taken: &mut HashSet<String>) -> String {
    let name = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let path = Path::new(name);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    let stem: String = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .chars()
        .map(|c| if c.is_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .take(100)
        .collect();
    let stem = match stem.trim_matches('.') {
        "" => "image",
        s => s,
    };
    let mut candidate = stem.to_string();
    let mut n = 1;
    while !taken.insert(candidate.to_lowercase()) {
        n += 1;
        candidate = format!("{stem}-{n}");
    }
    format!("{candidate}.{ext}")
}

/// The lower-cased stem of a stored file name (what [`unique_file_name`] keeps unique).
pub fn stem_key(file: &str) -> String {
    Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or(file).to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_are_safe_and_stems_unique_without_case() {
        let mut taken = HashSet::new();
        assert_eq!(unique_file_name("C:\\photos\\Cat 1.JPG", &mut taken), "Cat_1.jpg");
        assert_eq!(unique_file_name("../../cat_1.png", &mut taken), "cat_1-2.png");
        assert_eq!(unique_file_name("CAT_1.jpeg", &mut taken), "CAT_1-3.jpeg");
        assert_eq!(unique_file_name("..jpg", &mut taken), "image.jpg");
        assert_eq!(unique_file_name("…", &mut taken), "_.jpg");
        assert_eq!(unique_file_name("päron.webp", &mut taken), "päron.webp");
        assert_eq!(stem_key("Cat_1-2.png"), "cat_1-2");
        assert!(taken.contains("cat_1") && taken.contains("cat_1-2") && taken.contains("cat_1-3"));
    }

    #[test]
    fn counts_and_the_labeling_view() {
        let mut file = JobFile::new("lb_1".into(), "j".into(), vec!["a".into()], String::new());
        let record = |id: &str, status, boxes: usize| Record {
            id: id.into(),
            file: format!("{id}.jpg"),
            external: None,
            hash: None,
            width: 10,
            height: 10,
            status,
            boxes: vec![BBox { class: 0, x1: 0.0, y1: 0.0, x2: 5.0, y2: 5.0 }; boxes],
            source: None,
            error: None,
        };
        file.images = vec![
            record("1", LabelImageStatus::Pending, 0),
            record("2", LabelImageStatus::Labeled, 2),
            record("3", LabelImageStatus::Failed, 5),
            record("4", LabelImageStatus::Labeled, 1),
        ];
        let mut state = State { file, active: None, logs: VecDeque::new(), deleted: false };
        let job = state.job();
        assert_eq!(job.status, LabelJobStatus::Idle);
        assert_eq!(job.counts, LabelCounts { images: 4, pending: 1, labeled: 2, failed: 1, boxes: 3 });
        assert!(state.check_idle("x").is_ok() && state.claim("2").is_ok());

        state.active = Some(Active {
            stopping: false,
            cancel: Arc::new(AtomicBool::new(false)),
            waiting: HashSet::from(["1".to_string()]),
            in_call: HashSet::from(["2".to_string()]),
            thread: None,
        });
        assert_eq!(state.job().status, LabelJobStatus::Running);
        // An image in a call shows as `labeling` and keeps its boxes; counts go by the stored status.
        let shown = state.image(&state.file.images[1]);
        assert_eq!((shown.status, shown.boxes.len()), (LabelImageStatus::Labeling, 2));
        assert_eq!(state.job().counts.labeled, 2);
        assert_eq!(state.check_idle("x").unwrap_err().body.code, "label_job_active");
        // An image in a call cannot be taken; one that waits leaves the run; the others are free.
        assert_eq!(state.claim("2").unwrap_err().body.code, "label_job_active");
        assert!(state.claim("4").is_ok() && state.claim("1").is_ok());
        assert!(state.active.as_ref().unwrap().waiting.is_empty());
        state.active.as_mut().unwrap().stopping = true;
        assert_eq!(state.job().status, LabelJobStatus::Stopping);
    }
}
