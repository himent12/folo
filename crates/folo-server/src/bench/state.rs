//! One benchmark: its files on disk (`bench.json`, `entries/<id>.json`, `loops/<id>.json`), its state in memory
//! and the events it sends.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, MutexGuard};

use axum::http::StatusCode;
use folo_api::{
    BBox, Bench, BenchAccuracy, BenchBest, BenchCalibration, BenchDetection, BenchEntry, BenchEntryStatus,
    BenchEvent, BenchLocalization, BenchLoop, BenchLoopStatus, BenchPlan, BenchRobustness, BenchSelfLabeling,
    BenchSource,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;

use crate::error::{Error, Result};
use crate::index::write_atomic;
use crate::util::{rfc3339, unix_now};

/// Version of the layout of a benchmark's files.
pub const VERSION: u32 = 1;
/// Events a slow subscriber may fall behind by before it is sent the benchmark again.
const EVENT_BUFFER: usize = 1024;

/// What an entry that a previous process left queued or running says after a restart.
pub const SERVER_STOPPED_ENTRY: &str = "the server stopped before the entry finished";
/// What a loop that a previous process left running says after a restart.
pub const SERVER_STOPPED_LOOP: &str = "the server stopped";

/// An image of a benchmark as stored (its file is `images/<file>`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRecord {
    /// `1`, `2`, ... in the benchmark's order.
    pub id: String,
    /// File name in `images/`, unique within the benchmark (and so is its stem).
    pub file: String,
    /// Width after EXIF orientation.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// The file's size.
    pub bytes: u64,
    /// SHA-256 of the file, hex.
    pub sha256: String,
    /// The true boxes, clipped to the image, each at least a pixel wide and high.
    pub boxes: Vec<BBox>,
}

/// `bench.json`: everything a benchmark is, frozen when it is made (except its name, description and, while it
/// has no entries, its plan).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchFile {
    pub version: u32,
    pub id: String,
    pub name: String,
    pub description: String,
    pub classes: Vec<String>,
    pub created: String,
    pub source: BenchSource,
    pub source_name: String,
    pub seed: u64,
    pub plan: BenchPlan,
    pub images: Vec<ImageRecord>,
}

impl BenchFile {
    /// True boxes per class.
    pub fn class_boxes(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.classes.len()];
        for b in self.images.iter().flat_map(|i| &i.boxes) {
            if let Some(c) = counts.get_mut(b.class as usize) {
                *c += 1;
            }
        }
        counts
    }

    /// The image `id`.
    pub fn image(&self, id: &str) -> Option<&ImageRecord> {
        self.images.iter().find(|i| i.id == id)
    }
}

/// The benchmark's fingerprint: SHA-256 (hex) over its seed, classes, every image's content hash, size and true
/// boxes in order, and the plan's conditions, severities and weights (not its free text: `rationale`, `why` and
/// `designed_by` change nothing an entry is graded on).
pub fn fingerprint(file: &BenchFile) -> String {
    let mut h = Sha256::new();
    h.update(b"folo-bench-v1\0");
    // The engine that grades: a change of a condition's pixels or of a definition of the score is another test.
    h.update(folo_bench::ENGINE_VERSION.to_le_bytes());
    h.update(file.seed.to_le_bytes());
    h.update((file.classes.len() as u64).to_le_bytes());
    for c in &file.classes {
        h.update((c.len() as u64).to_le_bytes());
        h.update(c.as_bytes());
    }
    h.update((file.images.len() as u64).to_le_bytes());
    for im in &file.images {
        h.update(im.sha256.as_bytes());
        h.update(im.width.to_le_bytes());
        h.update(im.height.to_le_bytes());
        h.update((im.boxes.len() as u64).to_le_bytes());
        for b in &im.boxes {
            h.update(b.class.to_le_bytes());
            for v in [b.x1, b.y1, b.x2, b.y2] {
                h.update(v.to_bits().to_le_bytes());
            }
        }
    }
    h.update((file.plan.conditions.len() as u64).to_le_bytes());
    for pc in &file.plan.conditions {
        h.update((pc.condition.len() as u64).to_le_bytes());
        h.update(pc.condition.as_bytes());
        h.update((pc.severities.len() as u64).to_le_bytes());
        h.update(&pc.severities);
        h.update(pc.weight.to_bits().to_le_bytes());
    }
    hex(&h.finalize())
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// [`folo_api::BenchResult`] without the entry: what a finished entry's file keeps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultParts {
    pub accuracy: BenchAccuracy,
    pub robustness: Option<BenchRobustness>,
    pub calibration: Option<BenchCalibration>,
    pub localization: BenchLocalization,
    pub self_labeling: BenchSelfLabeling,
}

/// One image of a finished entry (`GET …/entries/{id}/images` without the image itself).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageOutcome {
    /// The image's id.
    pub image: String,
    /// The detections at or above the entry's best-F1 confidence, highest first.
    pub detections: Vec<BenchDetection>,
    pub missed: u32,
    pub wrong: u32,
}

/// `entries/<id>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntryFile {
    pub version: u32,
    pub entry: BenchEntry,
    /// Set once the entry is `done`.
    #[serde(default)]
    pub result: Option<ResultParts>,
    /// Per image, in the benchmark's order; set once the entry is `done`.
    #[serde(default)]
    pub images: Vec<ImageOutcome>,
}

/// The head of an entry file: what is read for every entry at start-up (the rest is read when asked for).
#[derive(Deserialize)]
struct EntryHead {
    version: u32,
    entry: BenchEntry,
}

/// `loops/<id>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoopFile {
    pub version: u32,
    #[serde(rename = "loop")]
    pub lp: BenchLoop,
}

/// An entry in memory.
pub struct EntryState {
    pub entry: BenchEntry,
    /// Ends the entry's work (stop, shutdown).
    pub cancel: Arc<AtomicBool>,
    /// An open external entry: what has arrived so far (in memory only).
    pub open: Option<Arc<Mutex<super::external::Filling>>>,
}

/// A loop in memory.
pub struct LoopState {
    pub lp: BenchLoop,
    pub cancel: Arc<AtomicBool>,
}

/// A benchmark's state; everything in it changes under [`Slot::lock`].
pub struct State {
    pub file: BenchFile,
    pub fingerprint: String,
    /// In creation order.
    pub entries: Vec<EntryState>,
    /// In creation order.
    pub loops: Vec<LoopState>,
    /// Set when the benchmark is deleted, so work still in flight writes nothing back.
    pub deleted: bool,
}

impl State {
    /// The benchmark as the API shows it.
    pub fn bench(&self) -> Bench {
        let f = &self.file;
        let entries: Vec<&BenchEntry> = self.entries.iter().map(|e| &e.entry).collect();
        // The leaderboard's first entry, when it is finished and was graded on the benchmark as it is.
        let best = ranked(&entries, &self.fingerprint)
            .first()
            .filter(|e| e.status == BenchEntryStatus::Done && e.fingerprint == self.fingerprint)
            .map(|e| BenchBest {
                entry_id: e.id.clone(),
                name: e.name.clone(),
                score: e.score.as_ref().map_or(0.0, |s| s.total),
                partial: e.score.as_ref().is_some_and(|s| s.partial),
            });
        Bench {
            id: f.id.clone(),
            name: f.name.clone(),
            description: f.description.clone(),
            classes: f.classes.clone(),
            created: f.created.clone(),
            source: f.source.clone(),
            source_name: f.source_name.clone(),
            images: f.images.len() as u64,
            boxes: f.images.iter().map(|i| i.boxes.len() as u64).sum(),
            class_boxes: f.class_boxes(),
            plan: f.plan.clone(),
            seed: f.seed,
            fingerprint: self.fingerprint.clone(),
            entries: self.entries.len() as u64,
            best,
            active: self.busy(),
        }
    }

    /// The entry `id`.
    pub fn entry(&self, id: &str) -> Result<&EntryState> {
        self.entries.iter().find(|e| e.entry.id == id).ok_or_else(|| entry_not_found(&self.file.id, id))
    }

    /// The entry `id`, to change.
    pub fn entry_mut(&mut self, id: &str) -> Option<&mut EntryState> {
        self.entries.iter_mut().find(|e| e.entry.id == id)
    }

    /// The loop `id`.
    pub fn lp(&self, id: &str) -> Result<&LoopState> {
        self.loops.iter().find(|l| l.lp.id == id).ok_or_else(|| loop_not_found(&self.file.id, id))
    }

    /// The loop `id`, to change.
    pub fn lp_mut(&mut self, id: &str) -> Option<&mut LoopState> {
        self.loops.iter_mut().find(|l| l.lp.id == id)
    }

    /// The leaderboard ([`ranked`]).
    pub fn leaderboard(&self) -> Vec<BenchEntry> {
        let entries: Vec<&BenchEntry> = self.entries.iter().map(|e| &e.entry).collect();
        ranked(&entries, &self.fingerprint).into_iter().cloned().collect()
    }

    /// Whether an entry or a loop of the benchmark is active.
    pub fn busy(&self) -> bool {
        self.entries.iter().any(|e| e.entry.status.is_active())
            || self.loops.iter().any(|l| l.lp.status.is_active())
    }

    /// Why the plan cannot change: a `done` or an active entry (failed and stopped ones do not count, they
    /// compare with nothing).
    pub fn frozen_by(&self) -> Option<&BenchEntry> {
        self.entries
            .iter()
            .map(|e| &e.entry)
            .find(|e| e.status == BenchEntryStatus::Done || e.status.is_active())
    }
}

/// The leaderboard's order of `entries` (given in creation order) of a benchmark whose fingerprint is
/// `fingerprint`: finished entries with that fingerprint that are complete (`score.partial` false) by score,
/// highest first; then finished partial ones by score; then finished entries with another fingerprint (graded
/// before the engine changed: they do not compare with the others) by score; equal scores keep the older entry
/// first (by `created`, then by id: an order that does not depend on how the entries are held, so it is the same
/// after a restart and for a client that sorts them itself); then the unfinished ones, newest first. A benchmark's `best` is the first of this order when it is
/// finished and has the benchmark's fingerprint.
pub fn ranked<'a>(entries: &[&'a BenchEntry], fingerprint: &str) -> Vec<&'a BenchEntry> {
    let group = |e: &BenchEntry| match (&e.score, e.status) {
        (_, BenchEntryStatus::Done) if e.fingerprint != fingerprint => 2,
        (Some(s), BenchEntryStatus::Done) if !s.partial => 0,
        (_, BenchEntryStatus::Done) => 1,
        _ => 3,
    };
    let total = |e: &BenchEntry| e.score.as_ref().map_or(0.0, |s| s.total);
    let age = |a: &BenchEntry, b: &BenchEntry| a.created.cmp(&b.created).then_with(|| a.id.cmp(&b.id));
    let mut order: Vec<&BenchEntry> = entries.to_vec();
    order.sort_by(|a, b| {
        group(a).cmp(&group(b)).then_with(|| {
            if group(a) == 3 { age(b, a) } else { total(b).total_cmp(&total(a)).then_with(|| age(a, b)) }
        })
    });
    order
}

/// `404 bench_not_found`.
pub fn bench_not_found(id: &str) -> Error {
    Error::new(StatusCode::NOT_FOUND, "bench_not_found", format!("benchmark '{id}' does not exist"))
        .hint("GET /api/bench/benchmarks lists the benchmarks that exist")
}

/// `404 bench_entry_not_found`.
pub fn entry_not_found(bench: &str, id: &str) -> Error {
    Error::new(
        StatusCode::NOT_FOUND,
        "bench_entry_not_found",
        format!("benchmark '{bench}' has no entry '{id}'"),
    )
    .hint("GET /api/bench/benchmarks/{id}/entries lists its entries")
}

/// `404 bench_loop_not_found`.
pub fn loop_not_found(bench: &str, id: &str) -> Error {
    Error::new(
        StatusCode::NOT_FOUND,
        "bench_loop_not_found",
        format!("benchmark '{bench}' has no loop '{id}'"),
    )
    .hint("GET /api/bench/benchmarks/{id}/loops lists its loops")
}

/// A benchmark in memory.
pub struct Slot {
    pub id: String,
    /// `<data dir>/bench/<id>`.
    pub dir: PathBuf,
    state: Mutex<State>,
    events: broadcast::Sender<BenchEvent>,
}

impl Slot {
    /// A benchmark around `state`.
    pub fn new(dir: PathBuf, state: State) -> Slot {
        Slot {
            id: state.file.id.clone(),
            dir,
            state: Mutex::new(state),
            events: broadcast::channel(EVENT_BUFFER).0,
        }
    }

    /// Reads the benchmark in `dir` with its entries and loops. Entries and loops a previous process left active
    /// are closed as `failed` (and written back); an entry or loop file that cannot be read is skipped and logged.
    pub fn load(dir: PathBuf) -> Result<Slot> {
        let path = dir.join("bench.json");
        let bad = |e: &dyn std::fmt::Display| Error::internal(format!("reading {}: {e}", path.display()));
        let bytes = std::fs::read(&path).map_err(|e| bad(&e))?;
        let file: BenchFile = serde_json::from_slice(&bytes).map_err(|e| bad(&e))?;
        if file.version != VERSION {
            return Err(bad(&format!("version {} is not one this server reads", file.version)));
        }
        let now = rfc3339(unix_now());
        let mut entries = Vec::new();
        let mut closed_entries = Vec::new();
        for (path, head) in read_all::<EntryHead>(&dir.join("entries")) {
            if head.version != VERSION {
                tracing::warn!(
                    "skipping {}: version {} is not one this server reads",
                    path.display(),
                    head.version
                );
                continue;
            }
            let mut entry = head.entry;
            if entry.status.is_active() {
                entry.status = BenchEntryStatus::Failed;
                entry.finished = Some(now.clone());
                entry.error = Some(SERVER_STOPPED_ENTRY.into());
                entry.hint = Some("start the entry again: POST /api/bench/benchmarks/{id}/entries".into());
                closed_entries.push(entry.id.clone());
            }
            entries.push(EntryState { entry, cancel: Arc::new(AtomicBool::new(false)), open: None });
        }
        entries.sort_by(|a, b| (&a.entry.created, &a.entry.id).cmp(&(&b.entry.created, &b.entry.id)));
        let mut loops = Vec::new();
        let mut closed_loops = Vec::new();
        for (path, f) in read_all::<LoopFile>(&dir.join("loops")) {
            if f.version != VERSION {
                tracing::warn!(
                    "skipping {}: version {} is not one this server reads",
                    path.display(),
                    f.version
                );
                continue;
            }
            let mut lp = f.lp;
            if lp.status.is_active() {
                lp.status = BenchLoopStatus::Failed;
                lp.phase = None;
                lp.finished = Some(now.clone());
                lp.error = Some(SERVER_STOPPED_LOOP.into());
                lp.hint = Some(
                    "its finished generations are kept; start a new loop to run the rest: POST \
                     /api/bench/benchmarks/{id}/loops"
                        .into(),
                );
                closed_loops.push(lp.id.clone());
            }
            loops.push(LoopState { lp, cancel: Arc::new(AtomicBool::new(false)) });
        }
        loops.sort_by(|a, b| (&a.lp.created, &a.lp.id).cmp(&(&b.lp.created, &b.lp.id)));
        let fingerprint = fingerprint(&file);
        let slot = Slot::new(dir, State { file, fingerprint, entries, loops, deleted: false });
        {
            let state = slot.lock();
            for id in closed_entries {
                // Rewritten whole: what the previous process had of the entry is its head, and that is all a
                // failed entry has.
                let entry = state.entry(&id).expect("just loaded").entry.clone();
                if let Err(e) = slot.save_entry(&state, &entry, None) {
                    tracing::warn!("{e}");
                }
            }
            for id in closed_loops {
                let lp = state.lp(&id).expect("just loaded").lp.clone();
                if let Err(e) = slot.save_loop(&state, &lp) {
                    tracing::warn!("{e}");
                }
            }
        }
        Ok(slot)
    }

    /// The state. Held briefly: never across inference, an agent call, a file copy or a training run. A panic
    /// under the lock does not take the benchmark down with it: every change is a few plain assignments.
    pub fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The state of a benchmark that still exists, for a request: `404 bench_not_found` once it was deleted.
    pub fn live(&self) -> Result<MutexGuard<'_, State>> {
        let state = self.lock();
        if state.deleted {
            return Err(bench_not_found(&self.id));
        }
        Ok(state)
    }

    /// Where an image's file is.
    pub fn image_path(&self, im: &ImageRecord) -> PathBuf {
        self.dir.join("images").join(&im.file)
    }

    fn entry_path(&self, id: &str) -> PathBuf {
        self.dir.join("entries").join(format!("{id}.json"))
    }

    fn loop_path(&self, id: &str) -> PathBuf {
        self.dir.join("loops").join(format!("{id}.json"))
    }

    /// Writes `bench.json` (atomically), unless the benchmark was deleted.
    pub fn save(&self, state: &State) -> Result<()> {
        if state.deleted {
            return Ok(());
        }
        write_json(&self.dir.join("bench.json"), &state.file)
    }

    /// Writes an entry's file, with its result and per-image outcomes once it is done (`parts`).
    pub fn save_entry(
        &self,
        state: &State,
        entry: &BenchEntry,
        parts: Option<(&ResultParts, &[ImageOutcome])>,
    ) -> Result<()> {
        if state.deleted {
            return Ok(());
        }
        let file = EntryFile {
            version: VERSION,
            entry: entry.clone(),
            result: parts.map(|p| p.0.clone()),
            images: parts.map(|p| p.1.to_vec()).unwrap_or_default(),
        };
        write_json(&self.entry_path(&entry.id), &file)
    }

    /// Reads an entry's file (its result and outcomes).
    pub fn read_entry(&self, id: &str) -> Result<EntryFile> {
        let path = self.entry_path(id);
        let bytes =
            std::fs::read(&path).map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))
    }

    /// Removes an entry's file.
    pub fn remove_entry(&self, id: &str) -> Result<()> {
        remove(&self.entry_path(id))
    }

    /// Writes a loop's file.
    pub fn save_loop(&self, state: &State, lp: &BenchLoop) -> Result<()> {
        if state.deleted {
            return Ok(());
        }
        write_json(&self.loop_path(&lp.id), &LoopFile { version: VERSION, lp: lp.clone() })
    }

    /// Removes a loop's file.
    pub fn remove_loop(&self, id: &str) -> Result<()> {
        remove(&self.loop_path(id))
    }

    /// Sends `event` to the subscribers there are.
    pub fn emit(&self, event: BenchEvent) {
        let _ = self.events.send(event);
    }

    /// Sends the benchmark as it is now.
    pub fn emit_bench(&self, state: &State) {
        self.emit(BenchEvent::Bench(Box::new(state.bench())));
    }

    /// A subscription to the benchmark's events.
    pub fn subscribe(&self) -> broadcast::Receiver<BenchEvent> {
        self.events.subscribe()
    }

    /// Changes entry `id` (when it still exists), writes its file and sends it; `bench` too when `bench_changed` or
    /// when the benchmark's `active` changed.
    pub fn update_entry(&self, id: &str, bench_changed: bool, change: impl FnOnce(&mut BenchEntry)) {
        let mut state = self.lock();
        let was = state.busy();
        let Some(e) = state.entry_mut(id) else { return };
        change(&mut e.entry);
        let entry = e.entry.clone();
        if let Err(err) = self.save_entry(&state, &entry, None) {
            tracing::warn!(bench = %self.id, "{err}");
        }
        self.emit(BenchEvent::Entry(Box::new(entry)));
        if bench_changed || state.busy() != was {
            self.emit_bench(&state);
        }
    }

    /// Changes loop `id` (when it still exists), writes its file and sends it (and the benchmark, when its `active`
    /// changed).
    pub fn update_loop(&self, id: &str, change: impl FnOnce(&mut BenchLoop)) {
        let mut state = self.lock();
        let was = state.busy();
        let Some(l) = state.lp_mut(id) else { return };
        change(&mut l.lp);
        let lp = l.lp.clone();
        if let Err(err) = self.save_loop(&state, &lp) {
            tracing::warn!(bench = %self.id, "{err}");
        }
        self.emit(BenchEvent::Loop(Box::new(lp)));
        if state.busy() != was {
            self.emit_bench(&state);
        }
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value).expect("benchmark files serialize");
    write_atomic(path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))
}

fn remove(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            Err(Error::internal(format!("removing {}: {e}", path.display())))
        }
        _ => Ok(()),
    }
}

/// Every `*.json` file in `dir` that parses as `T` (the others are logged and skipped, never overwritten).
fn read_all<T: serde::de::DeserializeOwned>(dir: &Path) -> Vec<(PathBuf, T)> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = e.path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        match std::fs::read(&path)
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice::<T>(&b).map_err(|e| e.to_string()))
        {
            Ok(v) => out.push((path, v)),
            Err(err) => tracing::warn!("skipping {}: {err}", path.display()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use folo_api::BenchPlanCondition;

    use super::*;

    fn file() -> BenchFile {
        BenchFile {
            version: VERSION,
            id: "bn_1".into(),
            name: "pond".into(),
            description: String::new(),
            classes: vec!["fish".into(), "frog".into()],
            created: "2026-11-09T00:00:00Z".into(),
            source: BenchSource::LabelJob { job_id: "lb_1".into() },
            source_name: "pond".into(),
            seed: 0,
            plan: BenchPlan::default(),
            images: vec![ImageRecord {
                id: "1".into(),
                file: "a.jpg".into(),
                width: 10,
                height: 10,
                bytes: 3,
                sha256: "ab".into(),
                boxes: vec![BBox { class: 1, x1: 0.0, y1: 0.0, x2: 5.0, y2: 5.0 }],
            }],
        }
    }

    fn entry(id: &str, status: BenchEntryStatus, score: Option<(f32, bool)>) -> BenchEntry {
        BenchEntry {
            id: id.into(),
            bench_id: "bn_1".into(),
            name: id.into(),
            contestant: folo_api::BenchContestant::External,
            status,
            created: "2026-11-09T00:00:00Z".into(),
            finished: None,
            progress: folo_api::BenchProgress::default(),
            images: 1,
            conditions: false,
            scored: true,
            fingerprint: String::new(),
            missing_classes: vec![],
            score: score.map(|(total, partial)| folo_api::BenchScore { total, partial, axes: vec![] }),
            map50_95: None,
            ci95: None,
            conditions_run: 0,
            speed: None,
            usage: None,
            error: None,
            hint: None,
        }
    }

    #[test]
    fn the_leaderboard_ranks_complete_then_partial_then_unfinished() {
        use BenchEntryStatus as S;
        // In creation order, a second apart.
        let mut entries = [
            entry("partial-high", S::Done, Some((90.0, true))),
            entry("complete-a", S::Done, Some((50.0, false))),
            entry("running-old", S::Running, None),
            entry("complete-b", S::Done, Some((70.0, false))),
            entry("complete-tie", S::Done, Some((50.0, false))),
            entry("failed", S::Failed, None),
            entry("partial-low", S::Done, Some((10.0, true))),
            entry("queued-new", S::Queued, None),
            BenchEntry { fingerprint: "older".into(), ..entry("older-engine", S::Done, Some((99.0, false))) },
        ];
        for (i, e) in entries.iter_mut().enumerate() {
            e.created = format!("2026-11-09T00:00:{i:02}Z");
        }
        // Given in any order: the order of the result does not depend on how the entries are held.
        let refs: Vec<&BenchEntry> = entries.iter().rev().collect();
        let order: Vec<&str> = ranked(&refs, "").iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            order,
            [
                "complete-b",
                "complete-a",
                "complete-tie",
                "partial-high",
                "partial-low",
                "older-engine",
                "queued-new",
                "failed",
                "running-old"
            ]
        );
        // Equal scores made in the same second: by id, whichever way they are given.
        let twins =
            [entry("be_b", S::Done, Some((50.0, false))), entry("be_a", S::Done, Some((50.0, false)))];
        for given in [[&twins[0], &twins[1]], [&twins[1], &twins[0]]] {
            assert_eq!(
                ranked(&given, "").iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
                ["be_a", "be_b"]
            );
        }
        // `best` is the first of that order: a partial entry leads only when nothing complete is finished.
        let state = |entries: Vec<BenchEntry>| State {
            file: file(),
            fingerprint: String::new(),
            entries: entries
                .into_iter()
                .map(|entry| EntryState { entry, cancel: Arc::new(AtomicBool::new(false)), open: None })
                .collect(),
            loops: vec![],
            deleted: false,
        };
        let best = state(entries.to_vec()).bench().best.unwrap();
        assert_eq!((best.entry_id.as_str(), best.score, best.partial), ("complete-b", 70.0, false));
        let best = state(vec![entries[0].clone(), entries[2].clone()]).bench().best.unwrap();
        assert_eq!((best.entry_id.as_str(), best.partial), ("partial-high", true));
        assert!(state(vec![entries[2].clone(), entries[5].clone()]).bench().best.is_none());
        // An entry graded on another fingerprint never leads, whatever its score.
        assert!(state(vec![entries[8].clone(), entries[2].clone()]).bench().best.is_none());
        // The plan is frozen by a done or an active entry, not by a failed or stopped one.
        assert!(state(vec![entries[5].clone(), entry("s", S::Stopped, None)]).frozen_by().is_none());
        assert_eq!(
            state(vec![entries[5].clone(), entries[2].clone()]).frozen_by().unwrap().id,
            "running-old"
        );
        assert_eq!(state(vec![entries[1].clone()]).frozen_by().unwrap().id, "complete-a");
    }

    #[test]
    fn fingerprints_follow_content_plan_and_seed_not_names() {
        let a = file();
        let fp = fingerprint(&a);
        assert_eq!(fp.len(), 64);
        assert_eq!(fingerprint(&a.clone()), fp);
        let renamed = BenchFile { name: "other".into(), description: "x".into(), ..a.clone() };
        assert_eq!(fingerprint(&renamed), fp);
        let mut plan = a.clone();
        plan.plan.conditions.push(BenchPlanCondition {
            condition: "haze".into(),
            severities: vec![1],
            weight: 1.0,
            why: None,
        });
        assert_ne!(fingerprint(&plan), fp);
        let mut why = plan.clone();
        why.plan.conditions[0].why = Some("murky".into());
        why.plan.rationale = Some("a pond".into());
        assert_eq!(fingerprint(&why), fingerprint(&plan));
        assert_ne!(fingerprint(&BenchFile { seed: 1, ..a.clone() }), fp);
        let mut truth = a.clone();
        truth.images[0].boxes[0].x2 = 6.0;
        assert_ne!(fingerprint(&truth), fp);
        let mut bytes = a.clone();
        bytes.images[0].sha256 = "cd".into();
        assert_ne!(fingerprint(&bytes), fp);
        assert_eq!(a.class_boxes(), [0, 1]);
    }
}
