//! The benchmark operations the MCP tools and `folo bench` share, built on the calls of [`Benchmarks`]: each
//! answers with a result of [`crate::types::bench`]. Their errors carry no front end's hint ([`super::hints`] adds
//! one).

use std::io::{BufRead, BufReader, Read};
use std::iter::Peekable;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_api::{
    Bench, BenchCatalog, BenchContestant, BenchContestantSpec, BenchEntry, BenchEntryStatus, BenchPlan,
    BenchPrediction, BenchResult, CreateBench, StartBenchEntry,
};
use folo_server::bench::MAX_BOXES;
use http::StatusCode;
use serde_json::Value;

use super::Benchmarks;
use crate::errors::{Error, invalid_arguments};
use crate::types::bench::*;

/// Most boxes in one part of a file of predictions sent to the server (about 15 MB of JSON; a request takes 64 MiB).
pub const MAX_PART_BOXES: usize = 200_000;
/// Most records in one part.
const MAX_PART_RECORDS: usize = 20_000;
/// Most bytes of JSON in one part (a request takes 64 MiB), so a file of long strings cannot fill this process's
/// memory or a request.
const MAX_PART_BYTES: usize = 32 * 1024 * 1024;
/// Longest line of a file of predictions: far more than a record of [`MAX_BOXES`] boxes takes, and all of a line that
/// is read before it is judged.
const MAX_RECORD_BYTES: usize = 1024 * 1024;

/// How large a part of a file of predictions may be.
#[derive(Clone, Copy, Debug)]
struct PartLimits {
    records: usize,
    boxes: usize,
    bytes: usize,
}

const PARTS: PartLimits =
    PartLimits { records: MAX_PART_RECORDS, boxes: MAX_PART_BOXES, bytes: MAX_PART_BYTES };
/// Class names a [`BenchDetail`] lists.
const DETAIL_CLASSES: usize = 100;

/// Who runs a new entry.
#[derive(Clone, Debug)]
pub enum Contestant {
    /// A model of the store, run by FOLO (published weights pulled with `folo weights pull` are models too).
    Model {
        /// The model's id.
        model_id: String,
        /// Run only the first this many images (default: all).
        max_images: Option<u32>,
        /// Whether to run the plan's conditions (default: yes).
        conditions: Option<bool>,
    },
    /// Predictions made elsewhere, read from a file in this process and sent in parts ([`read_predictions`]).
    Predictions {
        /// The file.
        path: PathBuf,
        /// False when the model gives no confidences.
        scored: bool,
    },
}

impl Benchmarks {
    /// The catalog, compact.
    pub fn catalog_result(&self) -> Result<BenchCatalogResult, Error> {
        let c: BenchCatalog = self.catalog()?;
        Ok(BenchCatalogResult {
            conditions: c
                .conditions
                .into_iter()
                .map(|c| BenchConditionInfo {
                    id: c.id,
                    group: format!("{:?}", c.group).to_lowercase(),
                    description: c.description,
                })
                .collect(),
            severities: c.severities,
            default_plan: plan_rows(&c.default_plan),
            axes: c.axes.into_iter().map(|a| BenchAxisWeight { axis: a.axis, weight: a.weight }).collect(),
        })
    }

    /// Every benchmark, newest first, as summaries.
    pub fn summaries(&self) -> Result<Vec<BenchSummary>, Error> {
        Ok(self.list()?.into_iter().map(summary).collect())
    }

    /// A benchmark with its plan and leaderboard.
    pub fn detail(&self, id: &str) -> Result<BenchDetail, Error> {
        let bench = self.get(id)?;
        let entries = self.entries(id)?;
        Ok(detail(bench, &entries))
    }

    /// Makes a benchmark (the plan is checked by the store: `folo_bench::check_plan`).
    pub fn create_bench(&self, req: CreateBench) -> Result<BenchDetail, Error> {
        Ok(detail(self.create(req)?, &[]))
    }

    /// Starts an entry. A model's entry runs in the background (follow it with [`Benchmarks::follow`]); a file of
    /// predictions is read and sent here, and the entry comes back scored. `name` is required for predictions.
    pub fn start(&self, id: &str, who: Contestant, name: Option<String>) -> Result<BenchEntryState, Error> {
        self.start_unless(id, who, name, &AtomicBool::new(false))
    }

    /// [`Benchmarks::start`], with `stop` (Ctrl+C) ending the sending of a file of predictions between its parts:
    /// the entry is then removed and the error is `interrupted`.
    pub fn start_unless(
        &self,
        id: &str,
        who: Contestant,
        name: Option<String>,
        stop: &AtomicBool,
    ) -> Result<BenchEntryState, Error> {
        let entry = match who {
            Contestant::Model { model_id, max_images, conditions } => {
                let contestant = BenchContestantSpec::Model { model_id };
                self.start_entry(id, StartBenchEntry { contestant, name, max_images, conditions })?
            }
            Contestant::Predictions { path, scored } => {
                let name = name.filter(|n| !n.trim().is_empty()).ok_or_else(|| {
                    invalid_arguments("an entry of predictions needs a name for the leaderboard")
                })?;
                self.submit_in(id, name, scored, &path, PARTS, stop)?
            }
        };
        self.state_of(entry)
    }

    /// An entry's state, with its score once it is done.
    pub fn state(&self, id: &str, entry_id: &str) -> Result<BenchEntryState, Error> {
        self.state_of(self.entry(id, entry_id)?)
    }

    /// [`Benchmarks::state`] of an entry already read.
    pub fn state_of(&self, e: BenchEntry) -> Result<BenchEntryState, Error> {
        let result = match e.status {
            folo_api::BenchEntryStatus::Done => Some(score(&e, self.result(&e.bench_id, &e.id)?)),
            _ => None,
        };
        Ok(BenchEntryState {
            benchmark_id: e.bench_id,
            entry_id: e.id,
            name: e.name,
            status: e.status,
            done: e.progress.done,
            total: e.progress.total,
            condition: e.progress.condition,
            severity: e.progress.severity,
            images: e.images,
            conditions: e.conditions,
            missing_classes: e.missing_classes,
            error: e.error,
            hint: e.hint,
            result,
        })
    }

    /// Stops an entry.
    pub fn stop(&self, id: &str, entry_id: &str) -> Result<BenchEntryState, Error> {
        self.state_of(self.stop_entry(id, entry_id)?)
    }

    /// Reads the entry every `poll` until it has ended, or until `tick` (called with each read of it while it is
    /// active) answers false. Returns the last read.
    pub fn follow(
        &self,
        id: &str,
        entry_id: &str,
        poll: Duration,
        mut tick: impl FnMut(&BenchEntry) -> bool,
    ) -> Result<BenchEntry, Error> {
        loop {
            let e = self.entry(id, entry_id)?;
            if !e.status.is_active() || !tick(&e) {
                return Ok(e);
            }
            std::thread::sleep(poll);
        }
    }

    /// An external entry from the file at `path`, in parts of at most `limits`: opened with the file's first part,
    /// sent the rest part by part and finished (scored). A file that does not parse, a part the server refuses, or
    /// `stop` stops and deletes the entry, so none is left half-filled; the error says so. A finish that got no
    /// answer may have worked, so it never deletes what the server may have scored ([`Benchmarks::after_lost_finish`]).
    fn submit_in(
        &self,
        id: &str,
        name: String,
        scored: bool,
        path: &Path,
        limits: PartLimits,
        stop: &AtomicBool,
    ) -> Result<BenchEntry, Error> {
        let mut records = read_predictions(path)?.peekable();
        let first = next_part(&mut records, limits)?;
        if first.is_empty() {
            return Err(invalid_arguments(format!("{} holds no predictions", folo_api::json::path(path))));
        }
        if stop.load(Ordering::Relaxed) {
            return Err(interrupted());
        }
        let shown = name.clone();
        let contestant = BenchContestantSpec::External { scored, open: true, predictions: first };
        let req = StartBenchEntry { contestant, name: Some(name), max_images: None, conditions: None };
        let entry = self.start_entry(id, req).map_err(|e| lost_start(e, &shown))?;
        let sent = (|| {
            loop {
                let part = next_part(&mut records, limits)?;
                // After the read: reading a part can take long, and what comes next is a request or the finish.
                if stop.load(Ordering::Relaxed) {
                    return Err(interrupted());
                }
                if part.is_empty() {
                    return Ok(());
                }
                self.add_predictions(id, &entry.id, part)?;
            }
        })();
        if let Err(e) = sent {
            return Err(self.undo(id, &entry.id, e));
        }
        match self.finish(id, &entry.id) {
            // The answer was lost, or another client finished it: the entry's state says what happened.
            Err(e) if e.status.is_server_error() || e.body.code == "bench_entry_not_open" => {
                self.after_lost_finish(id, &entry.id, e)
            }
            Err(e) => Err(self.undo(id, &entry.id, e)),
            ok => ok,
        }
    }

    /// After finish answered `e`, which says nothing of what became of the entry (its answer was lost, or another
    /// client is finishing it): an entry that ended scored is the result; one stopped from elsewhere is removed;
    /// one still open, or that cannot be read, may yet be scored, so it is kept and the error names it.
    fn after_lost_finish(&self, id: &str, entry_id: &str, e: Error) -> Result<BenchEntry, Error> {
        match self.entry(id, entry_id) {
            Ok(entry) if matches!(entry.status, BenchEntryStatus::Done | BenchEntryStatus::Failed) => {
                Ok(entry)
            }
            Ok(entry) if !entry.status.is_active() => Err(self.undo(id, entry_id, e)),
            _ => {
                let what = format!(
                    "entry {entry_id} was kept, as it may still be scored: read it by its id (left open, it \
                     expires on its own)"
                );
                Err(with_hint(e, what))
            }
        }
    }

    /// Stops and deletes entry `entry_id` after `e` (a failed submit); `e`'s hint says how that went. Only an entry
    /// the server says is stopped is deleted: one that ended otherwise (scored by another client, expired) is kept.
    fn undo(&self, id: &str, entry_id: &str, e: Error) -> Error {
        let stopped = match self.stop_entry(id, entry_id) {
            Err(s) if s.body.code == "bench_entry_not_active" => self.entry(id, entry_id),
            other => other,
        };
        let what = match stopped {
            Ok(entry) if entry.status == BenchEntryStatus::Stopped => match self.delete_entry(id, entry_id) {
                Ok(()) => format!("entry {entry_id} was stopped and deleted"),
                Err(u) => {
                    format!("entry {entry_id} could not be removed ({}): delete it by hand", u.body.message)
                }
            },
            Ok(entry) => format!("entry {entry_id} was kept: it is {:?}", entry.status).to_lowercase(),
            Err(u) => {
                format!(
                    "entry {entry_id} could not be stopped ({}): stop and delete it by hand",
                    u.body.message
                )
            }
        };
        with_hint(e, what)
    }
}

/// `e` with `what` before the hint it has.
fn with_hint(e: Error, what: String) -> Error {
    let hint = match e.body.hint.as_deref() {
        Some(h) => format!("{what}; {h}"),
        None => what,
    };
    e.hint(hint)
}

/// The error of an upload stopped by Ctrl+C.
fn interrupted() -> Error {
    Error::new(StatusCode::SERVICE_UNAVAILABLE, "interrupted", "interrupted while sending the predictions")
}

/// The request that opens an entry failed: when the answer may be what was lost, the entry may exist, and its id is
/// not known here.
fn lost_start(e: Error, name: &str) -> Error {
    if !e.status.is_server_error() {
        return e;
    }
    let what = format!(
        "the server may have opened an entry named '{name}': if the benchmark lists it as running, stop it by its \
         id (left open, it expires on its own)"
    );
    with_hint(e, what)
}

/// The records of a file of predictions, in order: JSON Lines of `BenchPrediction` records (`{"image",
/// "condition"?, "severity"?, "boxes": [{"class", "score", "x1", "y1", "x2", "y2"}]}`, the records
/// `examples/bench_external.py` sends), one per line. The file is read as the records are taken, a line of at most
/// [`MAX_RECORD_BYTES`] at a time, so neither a large file nor one long line (an array, a document) is held whole.
pub fn read_predictions(
    path: &Path,
) -> Result<impl Iterator<Item = Result<BenchPrediction, Error>> + use<>, Error> {
    let shown = folo_api::json::path(path);
    let file = std::fs::File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::new(StatusCode::NOT_FOUND, "path_not_found", format!("{shown} does not exist"))
        }
        _ => Error::new(StatusCode::INTERNAL_SERVER_ERROR, "io_error", format!("reading {shown}: {e}")),
    })?;
    Ok(records_of(BufReader::new(file), shown))
}

/// The records of `reader`, one per line. At most a record and its line ending are read of any line.
fn records_of<R: BufRead>(
    mut reader: R,
    shown: String,
) -> impl Iterator<Item = Result<BenchPrediction, Error>> {
    let mut n = 0usize;
    std::iter::from_fn(move || {
        let bad = |what: String| invalid_arguments(format!("{shown}: {what}"));
        loop {
            n += 1;
            let mut line = Vec::new();
            match (&mut reader).take(MAX_RECORD_BYTES as u64 + 2).read_until(b'\n', &mut line) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(bad(format!("reading line {n}: {e}")))),
            }
            let body = line.strip_suffix(b"\n").unwrap_or(&line);
            let body = body.strip_suffix(b"\r").unwrap_or(body);
            // Judged before a blank line is skipped: the rest of a long line must not pass as the next line.
            let long = body.len() > MAX_RECORD_BYTES;
            let text = body.trim_ascii();
            if text.is_empty() && !long {
                continue;
            }
            return Some(parse_record(text, long, n, &bad));
        }
    })
}

/// The record on line `n` of a file of predictions (`text`, trimmed; `cut` when the line was longer than can be read).
fn parse_record(
    text: &[u8],
    cut: bool,
    n: usize,
    bad: &dyn Fn(String) -> Error,
) -> Result<BenchPrediction, Error> {
    let not_a_record = || {
        bad(format!(
            "line {n} is not a record: the file holds one {{\"image\", \"boxes\", ...}} record per line (JSON \
             Lines), not arrays or documents of them"
        ))
    };
    if text.first() != Some(&b'{') {
        return Err(not_a_record());
    }
    if cut {
        return Err(bad(format!("line {n} is longer than the {MAX_RECORD_BYTES} bytes a record may take")));
    }
    let v: Value = serde_json::from_slice(text)
        .map_err(|e| bad(format!("line {n} is not JSON (column {}): {e}", e.column())))?;
    if v.get("predictions").is_some() {
        return Err(not_a_record());
    }
    let r: BenchPrediction = serde_json::from_value(v).map_err(|e| bad(format!("line {n}: {e}")))?;
    if r.boxes.len() > MAX_BOXES {
        return Err(bad(format!(
            "line {n} has {} boxes; a record carries at most {MAX_BOXES}",
            r.boxes.len()
        )));
    }
    Ok(r)
}

/// The next part of `records`: whole records, at most `limits.records` of them, `limits.boxes` boxes and
/// `limits.bytes` of JSON; empty at the end. A record that could never fit is an error (an empty part would read as
/// the end of the file).
fn next_part(
    records: &mut Peekable<impl Iterator<Item = Result<BenchPrediction, Error>>>,
    limits: PartLimits,
) -> Result<Vec<BenchPrediction>, Error> {
    let (mut part, mut boxes, mut bytes) = (Vec::new(), 0usize, 0usize);
    while part.len() < limits.records {
        let size = match records.peek() {
            None => break,
            Some(Ok(r)) => {
                // With the comma between records.
                let size = serde_json::to_vec(r).map_or(0, |b| b.len()) + 1;
                if r.boxes.len() > limits.boxes || size > limits.bytes {
                    return Err(invalid_arguments(format!(
                        "a record of {} boxes and {size} bytes is over a part's {} boxes and {} bytes",
                        r.boxes.len(),
                        limits.boxes,
                        limits.bytes
                    )));
                }
                if boxes + r.boxes.len() > limits.boxes || bytes + size > limits.bytes {
                    break;
                }
                size
            }
            Some(Err(_)) => 0,
        };
        let r = records.next().expect("peeked")?;
        boxes += r.boxes.len();
        bytes += size;
        part.push(r);
    }
    Ok(part)
}

fn plan_rows(plan: &BenchPlan) -> Vec<BenchPlanRow> {
    plan.conditions
        .iter()
        .map(|c| BenchPlanRow {
            condition: c.condition.clone(),
            severities: c.severities.clone(),
            weight: c.weight,
        })
        .collect()
}

fn summary(b: Bench) -> BenchSummary {
    BenchSummary {
        id: b.id,
        name: b.name,
        source_name: b.source_name,
        images: b.images,
        num_classes: b.classes.len(),
        conditions: b.plan.conditions.len(),
        entries: b.entries,
        best: b.best,
        active: b.active,
    }
}

fn detail(b: Bench, entries: &[BenchEntry]) -> BenchDetail {
    let leaderboard = entries
        .iter()
        .map(|e| BenchEntryRow {
            id: e.id.clone(),
            name: e.name.clone(),
            kind: match e.contestant {
                BenchContestant::Model { .. } => "model",
                BenchContestant::Agent { .. } => "agent",
                BenchContestant::External => "external",
            }
            .into(),
            status: e.status,
            score: e.score.as_ref().map(|s| s.total),
            partial: e.score.as_ref().is_some_and(|s| s.partial),
            map50_95: e.map50_95,
            ci95: e.ci95,
            stale: e.fingerprint != b.fingerprint,
        })
        .collect();
    BenchDetail {
        plan: plan_rows(&b.plan),
        classes: b.classes.iter().take(DETAIL_CLASSES).cloned().collect(),
        id: b.id,
        name: b.name,
        description: b.description,
        source: b.source,
        source_name: b.source_name,
        images: b.images,
        boxes: b.boxes,
        seed: b.seed,
        fingerprint: b.fingerprint,
        active: b.active,
        leaderboard,
    }
}

/// A done entry's score with the numbers behind each axis.
fn score(e: &BenchEntry, r: BenchResult) -> BenchScoreResult {
    let s = e.score.clone().or(r.entry.score.clone());
    let a = r.accuracy;
    BenchScoreResult {
        score: s.as_ref().map_or(0.0, |s| s.total),
        partial: s.as_ref().is_some_and(|s| s.partial),
        axes: s
            .map(|s| s.axes)
            .unwrap_or_default()
            .into_iter()
            .map(|x| BenchAxisValue { axis: x.axis, value: x.value, weight: x.weight })
            .collect(),
        accuracy: BenchAccuracySummary {
            map50_95: a.map50_95,
            ci95: a.ci95,
            map50: a.map50,
            map75: a.map75,
            precision: a.precision,
            recall: a.recall,
            best_conf: a.best_conf,
        },
        conditions: r
            .robustness
            .map(|rb| {
                rb.conditions
                    .into_iter()
                    .map(|c| BenchConditionRow {
                        condition: c.condition,
                        weight: c.weight,
                        retention: c.retention,
                        points: c
                            .points
                            .iter()
                            .map(|p| [p.severity as f32, p.map50_95, p.retention])
                            .collect(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        ece: r.calibration.map(|c| c.ece),
        mean_iou: r.localization.mean_iou,
        self_labeling: BenchSelfLabelingSummary {
            threshold: r.self_labeling.threshold,
            precision: r.self_labeling.precision,
            recall: r.self_labeling.recall,
            readiness: r.self_labeling.readiness,
        },
        ms_per_image: e.speed.as_ref().map(|s| s.ms_per_image),
    }
}

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
