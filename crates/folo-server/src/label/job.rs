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
use crate::store::SseFrame;
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
    /// A person checked its labels (see `LabelImage::reviewed`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub reviewed: bool,
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
    /// A dataset is being written from the job ([`Exporting`]): like a run, that keeps the job from being changed
    /// or deleted meanwhile.
    pub exporting: bool,
    /// Set when the job is deleted, so work still in flight writes nothing back.
    pub deleted: bool,
}

impl State {
    /// The job as the API shows it.
    pub fn job(&self) -> LabelJob {
        let mut counts = LabelCounts { images: self.file.images.len() as u64, ..Default::default() };
        for r in &self.file.images {
            match r.status {
                // Only a labeled image's boxes count (one whose relabeling failed stays labeled, with an `error`).
                // A failed image with boxes is one an older server left that way (#317): they go into no dataset
                // until it is labeled or saved again.
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
            reviewed: self.reviewed(r),
        }
    }

    /// Whether an image is reviewed as the API shows it: by the shown status, so an image back in an agent call is
    /// not reviewed while the call may replace its labels (the stored flag stays, for a call that leaves them
    /// untouched).
    pub fn reviewed(&self, r: &Record) -> bool {
        r.reviewed && self.status(r) == LabelImageStatus::Labeled
    }

    /// The stored image `image_id`.
    pub fn record(&self, image_id: &str) -> Result<&Record> {
        self.file.images.iter().find(|r| r.id == image_id).ok_or_else(|| Error::not_found("image", image_id))
    }

    /// The stored image `image_id`, to change.
    pub fn record_mut(&mut self, image_id: &str) -> Option<&mut Record> {
        self.file.images.iter_mut().find(|r| r.id == image_id)
    }

    /// `409 label_job_active` while a run is active, or a dataset is being written from the job.
    pub fn check_idle(&self, doing: &str) -> Result<()> {
        let id = &self.file.id;
        if self.active.is_some() {
            return Err(Error::conflict(
                "label_job_active",
                format!("labeling job '{id}' has an agent run in progress; {doing} once it ends"),
            )
            .hint("POST /api/label/jobs/{id}/stop ends the run"));
        }
        if self.exporting {
            return Err(Error::conflict(
                "label_job_active",
                format!("a dataset is being made from labeling job '{id}'; {doing} once it is written"),
            ));
        }
        Ok(())
    }

    /// `409 label_job_active` while `image_id` is in an agent call or a dataset is being written from the job,
    /// without removing a waiting image from the active run.
    pub fn check_editable(&self, image_id: &str) -> Result<()> {
        // A dataset being written has the images as they were when it began: changing one now would either be
        // missing from it without a word, or take a file away under the copy.
        if self.exporting {
            return Err(Error::conflict(
                "label_job_active",
                format!(
                    "a dataset is being made from labeling job '{}'; change its images once it is written",
                    self.file.id
                ),
            ));
        }
        if self.active.as_ref().is_some_and(|a| a.in_call.contains(image_id)) {
            return Err(Error::conflict(
                "label_job_active",
                format!("image '{image_id}' is with the agent right now; change it once that call is done"),
            )
            .hint("the image is `labeling`; POST /api/label/jobs/{id}/stop ends the run"));
        }
        Ok(())
    }

    /// Takes image `image_id` for a change by hand. `409 label_job_active` while it is in an agent call; an image
    /// that only waits for one leaves the run instead (the run then has one image less to label), so work done by
    /// hand is never overwritten by a call that had not started.
    pub fn claim(&mut self, image_id: &str) -> Result<()> {
        self.check_editable(image_id)?;
        let Some(active) = self.active.as_mut() else { return Ok(()) };
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
    /// Pre-serialized SSE frames shared across concurrent HTTP subscribers.
    sse: broadcast::Sender<SseFrame>,
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
            state: Mutex::new(State { file, active: None, logs: lines, exporting: false, deleted: false }),
            events: broadcast::channel(EVENT_BUFFER).0,
            sse: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    /// Whether `bytes` are a `job.json` [`Job::load`] takes (it parses and its version is one this server reads),
    /// found without writing anything: which copy of a job folder to load is picked by it.
    pub fn readable(bytes: &[u8]) -> bool {
        serde_json::from_slice::<JobFile>(bytes).is_ok_and(|f| f.version == VERSION)
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

    /// The state of a job that still exists, for a request: `404 label_job_not_found` once the job was deleted
    /// (a request that looked it up a moment before still holds it, and must not bring it back to life).
    pub fn live(&self) -> Result<MutexGuard<'_, State>> {
        let state = self.lock();
        if state.deleted {
            return Err(Error::not_found("label_job", &self.id));
        }
        Ok(state)
    }

    /// Removes what a previous process left in the job's folder and no longer belongs to the job: the scratch
    /// folders of agent calls, and files in `images/` that no image refers to (an upload that was being staged,
    /// or was stored after `job.json` was last written). Images added from other folders are never touched.
    pub fn sweep(&self) {
        let _ = std::fs::remove_dir_all(self.dir.join("tmp"));
        let state = self.lock();
        let uploads: HashSet<&str> =
            state.file.images.iter().filter(|r| r.external.is_none()).map(|r| r.file.as_str()).collect();
        for entry in std::fs::read_dir(self.images_dir()).into_iter().flatten().flatten() {
            if !uploads.contains(entry.file_name().to_string_lossy().as_ref()) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
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
        self.save_file(state, &state.file)
    }

    /// Writes `file` to `job.json` atomically unless the job was deleted, without requiring in-memory `State`
    /// to be mutated first.
    pub fn save_file(&self, state: &State, file: &JobFile) -> Result<()> {
        if state.deleted {
            return Ok(());
        }
        let path = self.dir.join("job.json");
        let bytes = serde_json::to_vec(file).expect("jobs serialize");
        write_atomic(&path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))
    }

    /// Sends `event` to the subscribers there are (pre-serializing once for all HTTP SSE subscribers).
    pub fn emit(&self, event: LabelEvent) {
        if self.sse.receiver_count() > 0 {
            let _ = self.sse.send(SseFrame::label(&event));
        }
        if self.events.receiver_count() > 0 {
            let _ = self.events.send(event);
        }
    }

    /// Sends the job as it is now when at least one subscriber is listening.
    pub fn emit_job(&self, state: &State) {
        if self.sse.receiver_count() > 0 || self.events.receiver_count() > 0 {
            self.emit(LabelEvent::Job(Box::new(state.job())));
        }
    }

    /// A subscription to the job's typed events.
    pub fn subscribe(&self) -> broadcast::Receiver<LabelEvent> {
        self.events.subscribe()
    }

    /// A subscription to the job's pre-serialized SSE frames.
    pub fn subscribe_sse(&self) -> broadcast::Receiver<SseFrame> {
        self.sse.subscribe()
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

/// A dataset being written from a job: while this lives, the job counts as busy ([`State::check_idle`]), so it
/// is still there, unchanged, when the dataset is registered.
pub struct Exporting(pub Arc<Job>);

impl Exporting {
    /// Marks `job` as being exported: `409 label_job_active` while a run is active or another export is.
    pub fn begin(job: Arc<Job>) -> Result<Exporting> {
        {
            let mut state = job.live()?;
            state.check_idle("make a dataset")?;
            state.exporting = true;
        }
        Ok(Exporting(job))
    }
}

impl Drop for Exporting {
    fn drop(&mut self) {
        self.0.lock().exporting = false;
    }
}

/// A file name for an image called `name` that no image of the job has yet: `name` without folders, reduced to
/// letters, digits, `.`, `_` and `-` (a stem of at most 100 bytes), with `-2`, `-3`, ... on its stem while that stem is in `taken` (lower-cased
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
        // At most 100 bytes: Linux and macOS file names hold 255 bytes, not characters.
        .scan(0, |bytes, c| {
            *bytes += c.len_utf8();
            (*bytes <= 100).then_some(c)
        })
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

    /// File names hold 255 bytes on Linux and macOS: a long name in a script of 3-byte characters still fits.
    #[test]
    fn long_names_stay_file_names() {
        let mut taken = HashSet::new();
        let name = unique_file_name(&format!("{}.jpg", "猫".repeat(300)), &mut taken);
        assert!(
            name.starts_with('猫') && name.ends_with(".jpg") && name.len() <= 255,
            "{} bytes",
            name.len()
        );
        let again = unique_file_name(&format!("{}.jpg", "猫".repeat(300)), &mut taken);
        assert!(again.ends_with("-2.jpg") && again.len() <= 255, "{again}");
    }

    fn state_of(file: JobFile) -> State {
        State { file, active: None, logs: VecDeque::new(), exporting: false, deleted: false }
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
            reviewed: false,
        };
        file.images = vec![
            record("1", LabelImageStatus::Pending, 0),
            record("2", LabelImageStatus::Labeled, 2),
            record("3", LabelImageStatus::Failed, 5),
            record("4", LabelImageStatus::Labeled, 1),
        ];
        file.images[1].reviewed = true;
        let mut state = State { file, active: None, logs: VecDeque::new(), exporting: false, deleted: false };
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
        // Nor is it reviewed while the call is open; its stored flag comes back if the call leaves it as it was.
        assert!(!shown.reviewed && state.file.images[1].reviewed);
        assert_eq!(state.job().counts.labeled, 2);
        assert_eq!(state.check_idle("x").unwrap_err().body.code, "label_job_active");
        let exporting = State {
            exporting: true,
            active: None,
            ..state_of(JobFile::new("lb_2".into(), "j".into(), vec![], String::new()))
        };
        assert_eq!(exporting.check_idle("x").unwrap_err().body.code, "label_job_active");
        let mut exporting = exporting;
        assert_eq!(exporting.claim("1").unwrap_err().body.code, "label_job_active");
        // An image in a call cannot be taken; one that waits leaves the run; the others are free.
        assert_eq!(state.claim("2").unwrap_err().body.code, "label_job_active");
        assert!(state.claim("4").is_ok() && state.claim("1").is_ok());
        assert!(state.active.as_ref().unwrap().waiting.is_empty());
        state.active.as_mut().unwrap().stopping = true;
        assert_eq!(state.job().status, LabelJobStatus::Stopping);
    }
}
