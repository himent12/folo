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
use folo_server::bench::{MAX_BOXES, RETRY_PART_HINT};
use http::StatusCode;
use serde_json::Value;

use super::Benchmarks;
use super::hints::ENTRY_COMMANDS;
use crate::errors::Error;
use crate::types::bench::*;

/// Front-end-neutral `invalid_arguments` error: [`super::hints::decorate`] supplies the MCP or CLI hint.
fn invalid_arguments(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_arguments", message)
}

/// Most boxes in one part of a file of predictions sent to the server (about 15 MB of JSON; a request takes 64 MiB).
pub const MAX_PART_BOXES: usize = 200_000;
/// Most records in one part.
const MAX_PART_RECORDS: usize = 20_000;
/// Most bytes of record JSON in one part, so a file of long strings cannot fill this process's memory or a request. The
/// commas between records and the request's wrapper come on top (a request takes 64 MiB).
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

    /// [`Benchmarks::start`], with `stop` (Ctrl+C) ending the sending of a file of predictions before the next record
    /// is read or the next request is sent: the error is `interrupted`, and [`Benchmarks::clean_up`] decides what
    /// becomes of the entry.
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
                let entry = self.submit_in(id, name, scored, &path, PARTS, stop)?;
                return Ok(self.state_after_submit(entry));
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
        Ok(state_with_result(e, result))
    }

    /// Like [`Benchmarks::state_of`], for a predictions upload that already finished on the server: if the entry is
    /// `done` and the separate `/result` request fails, keep the scored entry instead of failing the upload, with a
    /// hint noting the score and why the detailed breakdown is missing.
    fn state_after_submit(&self, mut e: BenchEntry) -> BenchEntryState {
        if e.status != BenchEntryStatus::Done {
            return state_with_result(e, None);
        }
        match self.result(&e.bench_id, &e.id) {
            Ok(r) => {
                let scored = score(&e, r);
                state_with_result(e, Some(scored))
            }
            Err(err) => {
                let who = named(&e);
                let score_note = e
                    .score
                    .as_ref()
                    .map(|s| format!(" (score {:.1}{})", s.total, if s.partial { ", partial" } else { "" }))
                    .unwrap_or_default();
                let note = format!(
                    "{who} was scored{score_note}, but its detailed result could not be read ({})",
                    err.body.message
                );
                e.hint = Some(match e.hint.take() {
                    Some(h) => format!("{note}; {h}"),
                    None => note,
                });
                state_with_result(e, None)
            }
        }
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
    /// sent the rest part by part and finished (scored). A failure before the finish, or a finish the server
    /// [`refused`], goes to [`Benchmarks::clean_up`]; any other failed finish to [`Benchmarks::after_finish`], which
    /// never stops or deletes.
    fn submit_in(
        &self,
        id: &str,
        name: String,
        scored: bool,
        path: &Path,
        limits: PartLimits,
        stop: &AtomicBool,
    ) -> Result<BenchEntry, Error> {
        let shown = folo_api::json::path(path);
        self.submit_numbered_records(
            id,
            name,
            scored,
            &shown,
            read_numbered_predictions_unless(path, scored, stop)?,
            limits,
            stop,
        )
    }

    /// [`Benchmarks::submit_in`] of `records` (`shown` names where they come from).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn submit_records(
        &self,
        id: &str,
        name: String,
        scored: bool,
        shown: &str,
        records: impl Iterator<Item = Result<BenchPrediction, Error>>,
        limits: PartLimits,
        stop: &AtomicBool,
    ) -> Result<BenchEntry, Error> {
        let numbered = records.enumerate().map(|(i, r)| r.map(|p| (i + 1, p)));
        self.submit_numbered_records(id, name, scored, shown, numbered, limits, stop)
    }

    /// [`Benchmarks::submit_in`] of `(line_number, record)` pairs (`shown` names where they come from).
    #[allow(clippy::too_many_arguments)]
    fn submit_numbered_records(
        &self,
        id: &str,
        name: String,
        scored: bool,
        shown: &str,
        records: impl Iterator<Item = Result<(usize, BenchPrediction), Error>>,
        limits: PartLimits,
        stop: &AtomicBool,
    ) -> Result<BenchEntry, Error> {
        let mut source = records;
        // Before each record is read, so a pipe that gave one record after Ctrl+C is not read again.
        let mut records = std::iter::from_fn(|| match check_interrupt(stop) {
            Ok(()) => source.next(),
            Err(e) => Some(Err(e)),
        })
        .peekable();
        // And after a part is read: a request or the finish comes next.
        let mut read = || {
            let part = next_numbered_part(&mut records, limits)?;
            check_interrupt(stop)?;
            Ok::<_, Error>(part)
        };
        let (first, first_lines) = read()?;
        if first.is_empty() {
            return Err(invalid_arguments(format!("{shown} holds no predictions")));
        }
        // As the server saves it.
        let listed = name.trim().to_string();
        let chosen_id = self.supports_upload_safe().then(|| super::fresh_entry_id(id, &listed));
        let contestant = BenchContestantSpec::External { scored, open: true, predictions: first };
        let req = StartBenchEntry { contestant, name: Some(name), max_images: None, conditions: None };
        let entry = match self.start_entry_with_id(id, req, chosen_id.as_deref()) {
            Ok(e) => e,
            Err(e) => {
                let e = rewrite_part_error(e, shown, &first_lines);
                // Definitive FOLO refusal (`4xx` parsed from FOLO or `501 not_implemented` before sending):
                // no entry was opened. Unparsed proxy `4xx` (`NOT_FOLO`) and other `5xx` are ambiguous.
                if start_definitely_refused(&e) {
                    return Err(e);
                }
                if let Some(eid) = chosen_id {
                    let who = format!("entry '{listed}' ({eid})");
                    let what = match self.clean_up_by_id(id, &eid, &who, false) {
                        CleanUp::Closed(w) | CleanUp::OpenOrUnknown(w) => w,
                    };
                    return Err(with_hint(e, what));
                }
                return Err(lost_start(e, &listed));
            }
        };
        let sent: Result<(), Error> = (|| {
            loop {
                let (part, part_lines) = read()?;
                if part.is_empty() {
                    return Ok(());
                }
                self.add_predictions(id, &entry.id, part)
                    .map_err(|e| rewrite_part_error(e, shown, &part_lines))?;
            }
        })();
        let failed = |mut e: Error| match self.clean_up_outcome(id, &entry) {
            CleanUp::Closed(what) => {
                // Once the open entry is closed or removed, the server's HTTP open-entry retry hint ("send only the
                // records not sent yet") is stale; tell the user to fix the file and re-run instead.
                if e.body.hint.as_deref() == Some(RETRY_PART_HINT) {
                    e.body.hint = Some("fix the predictions file and run again".into());
                }
                with_hint(e, what)
            }
            CleanUp::OpenOrUnknown(what) => with_hint(e, what),
        };
        if let Err(e) = sent {
            return Err(failed(e));
        }
        self.finish(id, &entry.id)
            .or_else(|e| if refused(&e) { Err(failed(e)) } else { self.after_finish(id, &entry, e) })
    }

    /// A failure while the server has not taken the entry for scoring. The entry is read: one still open is stopped
    /// and removed; one in any other state, or that cannot be read, is left and named. Returns what to tell the user.
    #[cfg(test)]
    fn clean_up(&self, id: &str, opened: &BenchEntry) -> String {
        match self.clean_up_outcome(id, opened) {
            CleanUp::Closed(s) | CleanUp::OpenOrUnknown(s) => s,
        }
    }

    fn clean_up_outcome(&self, id: &str, opened: &BenchEntry) -> CleanUp {
        let who = named(opened);
        self.clean_up_by_id(id, &opened.id, &who, true)
    }

    fn clean_up_by_id(&self, id: &str, entry_id: &str, who: &str, existed: bool) -> CleanUp {
        let classify = |e: &BenchEntry| {
            let msg = left(who, e);
            if e.status.is_active() { CleanUp::OpenOrUnknown(msg) } else { CleanUp::Closed(msg) }
        };
        let is_gone =
            |err: &Error| err.status == StatusCode::NOT_FOUND && err.body.code == "bench_entry_not_found";
        let already_removed = || CleanUp::Closed(format!("{who} was already removed"));
        match self.entry(id, entry_id) {
            Ok(e) if e.status == BenchEntryStatus::Running => {
                // On servers supporting `bench_upload_safe`, atomically stop + remove the open entry in one
                // request so there is never a window between stop and delete, and an ambiguous response never
                // deletes an entry stopped by another client.
                if self.supports_upload_safe() {
                    return match self.discard_open_entry(id, entry_id) {
                        Ok(_) => CleanUp::Closed(format!("{who} was removed")),
                        Err(ref err) if is_gone(err) => already_removed(),
                        Err(ref err)
                            if err.status.is_client_error() && err.body.code != folo_client::NOT_FOLO =>
                        {
                            match self.entry(id, entry_id) {
                                Ok(e) => classify(&e),
                                Err(ref g) if is_gone(g) => already_removed(),
                                Err(_) => CleanUp::OpenOrUnknown(unread(who)),
                            }
                        }
                        Err(_) => match self.entry(id, entry_id) {
                            Err(ref g) if is_gone(g) => CleanUp::Closed(format!("{who} was removed")),
                            Ok(e) => classify(&e),
                            Err(_) => CleanUp::OpenOrUnknown(unread(who)),
                        },
                    };
                }
                // Legacy server fallback (without `bench_upload_safe`): two-step stop then delete.
                let stopped = self.stop_entry(id, entry_id).or_else(|_| self.entry(id, entry_id));
                match stopped {
                    Ok(e) if e.status == BenchEntryStatus::Stopped => match self.delete_entry(id, entry_id) {
                        Ok(()) => CleanUp::Closed(format!("{who} was removed")),
                        Err(ref u) if is_gone(u) => already_removed(),
                        Err(u) => CleanUp::Closed(format!(
                            "{who} is stopped; removing it did not succeed ({})",
                            u.body.message
                        )),
                    },
                    Ok(e) => classify(&e),
                    Err(ref err) if is_gone(err) => already_removed(),
                    Err(_) => CleanUp::OpenOrUnknown(unread(who)),
                }
            }
            Ok(e) => classify(&e),
            Err(ref err) if is_gone(err) => {
                if existed {
                    already_removed()
                } else {
                    CleanUp::OpenOrUnknown(format!("{who} was not found on the server; {ENTRY_COMMANDS}"))
                }
            }
            Err(_) => CleanUp::OpenOrUnknown(unread(who)),
        }
    }

    /// A finish with no answer, a `5xx` or a `409`: the entry may be scoring, so nothing here stops or deletes it, on
    /// any server. The entry is read once: scored is the result; anything else is `e`, with a hint naming the entry.
    fn after_finish(&self, id: &str, sent: &BenchEntry, e: Error) -> Result<BenchEntry, Error> {
        let who = named(sent);
        match self.entry(id, &sent.id) {
            Ok(now) if now.status == BenchEntryStatus::Done => Ok(now),
            Ok(now) => Err(with_hint(e, left(&who, &now))),
            Err(_) => Err(with_hint(e, unread(&who))),
        }
    }
}

enum CleanUp {
    /// Entry was removed or is in a terminal (closed) state and can no longer receive parts.
    Closed(String),
    /// Entry may still be open/active or its state could not be read.
    OpenOrUnknown(String),
}

const REPEATS_PREDICTION: &str = " repeats prediction ";
const REPEATS_SUFFIX: &str = " with other boxes: answer each image and condition once";

/// Translates a server per-part record diagnostic (`prediction <k> ...`, and any trailing intra-part
/// `repeats prediction <j> with other boxes: answer each image and condition once`) into the exact 1-based file line
/// numbers of the predictions file (`shown`).
fn rewrite_part_error(mut e: Error, shown: &str, part_lines: &[usize]) -> Error {
    let Some(rest) = e.body.message.strip_prefix("prediction ") else {
        return e;
    };
    let digits_len = rest.bytes().take_while(u8::is_ascii_digit).count();
    let Some(k) = rest[..digits_len].parse::<usize>().ok().filter(|&k| k >= 1) else {
        return e;
    };
    let Some(&line) = part_lines.get(k - 1) else {
        return e;
    };
    let tail = &rest[digits_len..];
    if !tail.starts_with([' ', ':']) {
        return e;
    }
    let rewritten_tail = tail.strip_suffix(REPEATS_SUFFIX).and_then(|before_suffix| {
        let pos = before_suffix.rfind(REPEATS_PREDICTION)?;
        if !before_suffix[..pos].ends_with(')') {
            return None;
        }
        let digits = &before_suffix[pos + REPEATS_PREDICTION.len()..];
        let j = (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
            .then(|| digits.parse::<usize>().ok())
            .flatten()
            .filter(|&j| j >= 1)?;
        let &first_line = part_lines.get(j - 1)?;
        Some(format!("{} repeats line {first_line}{REPEATS_SUFFIX}", &before_suffix[..pos]))
    });
    e.body.message = match rewritten_tail {
        Some(t) => format!("{shown}: line {line}{t}"),
        None => format!("{shown}: line {line}{tail}"),
    };
    e
}

/// True when the FOLO server answered a finish with a refusal (did not take the entry for scoring).
///
/// Unparsed non-FOLO `4xx` responses (`bad_gateway`, e.g. from a proxy) are excluded because the finish may still have
/// reached the server. `409 bench_entry_not_open` is excluded too: it also answers a finish while another one scores
/// the entry, which still reads `running`, so a stop would cancel that score.
fn refused(e: &Error) -> bool {
    e.status.is_client_error()
        && e.body.code != folo_client::NOT_FOLO
        && !(e.status == StatusCode::CONFLICT && e.body.code == "bench_entry_not_open")
}

/// `e` with `what` before the hint it has, keeping any trailing [`ENTRY_COMMANDS`] placeholder at the very end so
/// [`super::hints::decorate`] only replaces that suffix and never text inside an entry's name.
fn with_hint(e: Error, what: String) -> Error {
    let (base, has_commands) = match what.strip_suffix(ENTRY_COMMANDS) {
        Some(prefix) => (prefix.strip_suffix("; ").unwrap_or(prefix), true),
        None => (what.as_str(), false),
    };
    let mut hint = match e.body.hint.as_deref() {
        Some(h) => format!("{base}; {h}"),
        None => base.to_string(),
    };
    if has_commands {
        hint.push_str("; ");
        hint.push_str(ENTRY_COMMANDS);
    }
    e.hint(hint)
}

/// Builds a [`BenchEntryState`] from `e` and an already-loaded (or omitted) score breakdown.
fn state_with_result(e: BenchEntry, result: Option<BenchScoreResult>) -> BenchEntryState {
    let summary_score = e.score.as_ref().map(|s| s.total);
    BenchEntryState {
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
        summary_score,
    }
}

/// An entry as the hints name it: as the server lists it, and its id.
fn named(e: &BenchEntry) -> String {
    format!("entry '{}' ({})", e.name, e.id)
}

/// The `interrupted` error once `stop` (Ctrl+C) is set.
fn check_interrupt(stop: &AtomicBool) -> Result<(), Error> {
    if !stop.load(Ordering::Relaxed) {
        return Ok(());
    }
    Err(Error::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "interrupted",
        "interrupted while sending the predictions",
    ))
}

/// What the hint says of an entry this upload leaves as it is.
fn left(who: &str, e: &BenchEntry) -> String {
    let state = match e.status {
        BenchEntryStatus::Running => "open or being scored",
        BenchEntryStatus::Queued => "queued",
        BenchEntryStatus::Stopping => "stopping",
        BenchEntryStatus::Done => "done",
        BenchEntryStatus::Failed => "failed",
        BenchEntryStatus::Stopped => "stopped",
    };
    format!("{who} was left: it is {state}; {ENTRY_COMMANDS}")
}

/// What the hint says of an entry whose state could not be read.
fn unread(who: &str) -> String {
    format!("{who} may be left on the server: its state could not be read; {ENTRY_COMMANDS}")
}

/// True when an opening `POST /entries` error came cleanly from FOLO and proves no entry was created
/// (any parsed FOLO `4xx` or `501 not_implemented`). Unparsed proxy `4xx` (`NOT_FOLO`) and other `5xx` are ambiguous.
fn start_definitely_refused(e: &Error) -> bool {
    e.body.code != folo_client::NOT_FOLO
        && (e.status.is_client_error() || e.status == StatusCode::NOT_IMPLEMENTED)
}

/// The request that opens an entry failed: when the answer may be what was lost, the entry may exist, and its id is
/// not known here.
fn lost_start(e: Error, name: &str) -> Error {
    if start_definitely_refused(&e) {
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
    let (file, shown) = open_predictions_file(path)?;
    Ok(records_of(BufReader::new(file), shown))
}

/// Like [`read_predictions`], yielding `(1-based file line number, record)` and checking `stop` before every line
/// read (including after skipping a blank line) so Ctrl+C during a blocked pipe read whose next line is blank never
/// starts another read.
fn read_numbered_predictions_unless<'a>(
    path: &Path,
    scored: bool,
    stop: &'a AtomicBool,
) -> Result<impl Iterator<Item = Result<(usize, BenchPrediction), Error>> + 'a, Error> {
    check_interrupt(stop)?;
    let (file, shown) = open_predictions_file(path)?;
    Ok(records_numbered_unless(BufReader::new(file), shown, scored, Some(stop)))
}

fn open_predictions_file(path: &Path) -> Result<(std::fs::File, String), Error> {
    let shown = folo_api::json::path(path);
    let file = std::fs::File::open(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            Error::new(StatusCode::NOT_FOUND, "path_not_found", format!("{shown} does not exist"))
        }
        _ => Error::new(StatusCode::INTERNAL_SERVER_ERROR, "io_error", format!("reading {shown}: {e}")),
    })?;
    Ok((file, shown))
}

/// The records of `reader`, one per line. At most a record and its line ending are read of any line.
fn records_of<'a, R: BufRead + 'a>(
    reader: R,
    shown: String,
) -> impl Iterator<Item = Result<BenchPrediction, Error>> + 'a {
    records_of_unless(reader, shown, true, None)
}

/// [`records_of`], checking `stop` before every line read (including after skipping a blank line).
fn records_of_unless<'a, R: BufRead + 'a>(
    reader: R,
    shown: String,
    scored: bool,
    stop: Option<&'a AtomicBool>,
) -> impl Iterator<Item = Result<BenchPrediction, Error>> + 'a {
    records_numbered_unless(reader, shown, scored, stop).map(|r| r.map(|(_, p)| p))
}

/// Like [`records_of_unless`], keeping each record's 1-based line number in the input file.
fn records_numbered_unless<'a, R: BufRead + 'a>(
    mut reader: R,
    shown: String,
    scored: bool,
    stop: Option<&'a AtomicBool>,
) -> impl Iterator<Item = Result<(usize, BenchPrediction), Error>> + 'a {
    let mut n = 0usize;
    std::iter::from_fn(move || {
        let bad = |what: String| invalid_arguments(format!("{shown}: {what}"));
        loop {
            if let Some(flag) = stop
                && let Err(e) = check_interrupt(flag)
            {
                return Some(Err(e));
            }
            n += 1;
            let mut line = Vec::new();
            match (&mut reader).take(MAX_RECORD_BYTES as u64 + 2).read_until(b'\n', &mut line) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => {
                    if let Some(flag) = stop
                        && let Err(ie) = check_interrupt(flag)
                    {
                        return Some(Err(ie));
                    }
                    return Some(Err(bad(format!("reading line {n}: {e}"))));
                }
            }
            // If line 1 starts with a 3-byte UTF-8 BOM and has not reached `\n` yet, budget up to 3 extra bytes so
            // a BOM-prefixed record at exactly `MAX_RECORD_BYTES` (with optional `\r\n`) is not truncated and an
            // overlong BOM-prefixed line cannot pass as `<= MAX_RECORD_BYTES` and split into two lines.
            if n == 1 && !line.ends_with(b"\n") && line.starts_with(b"\xEF\xBB\xBF") {
                if let Some(flag) = stop
                    && let Err(e) = check_interrupt(flag)
                {
                    return Some(Err(e));
                }
                if let Err(e) = (&mut reader).take(3).read_until(b'\n', &mut line) {
                    if let Some(flag) = stop
                        && let Err(ie) = check_interrupt(flag)
                    {
                        return Some(Err(ie));
                    }
                    return Some(Err(bad(format!("reading line {n}: {e}"))));
                }
            }
            // A `\r` is a line ending only before a `\n`: at the end of the file it is one more byte of the line.
            let body = match line.strip_suffix(b"\n") {
                Some(b) => b.strip_suffix(b"\r").unwrap_or(b),
                None => &line,
            };
            // Strip a UTF-8 BOM on line 1 (e.g. files written with UTF-8-SIG on Windows) before length/prefix checks.
            let body = if n == 1 { body.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(body) } else { body };
            // Judged before a blank line is skipped: the rest of a long line must not pass as the next line.
            let long = body.len() > MAX_RECORD_BYTES;
            let text = body.trim_ascii();
            if text.is_empty() && !long {
                continue;
            }
            return Some(parse_record(text, long, n, scored, &bad).map(|p| (n, p)));
        }
    })
}

/// Strict JSONL record input that rejects unknown/misspelled fields (such as `confidence` instead of `score`,
/// which would otherwise silently default to `1.0`).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictionInput {
    image: String,
    #[serde(default)]
    condition: Option<String>,
    #[serde(default)]
    severity: Option<u8>,
    boxes: Vec<DetectionInput>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DetectionInput {
    class: u32,
    #[serde(default = "default_score_one")]
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

fn default_score_one() -> f32 {
    1.0
}

/// The record on line `n` of a file of predictions (`text`, trimmed; `cut` when the line was longer than can be read).
fn parse_record(
    text: &[u8],
    cut: bool,
    n: usize,
    scored: bool,
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
    let raw: PredictionInput = serde_json::from_value(v).map_err(|e| bad(format!("line {n}: {e}")))?;
    if raw.boxes.len() > MAX_BOXES {
        return Err(bad(format!(
            "line {n} has {} boxes; a record carries at most {MAX_BOXES}",
            raw.boxes.len()
        )));
    }
    for (i, b) in raw.boxes.iter().enumerate() {
        if ![b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()) {
            return Err(bad(format!(
                "line {n} (image '{}'), box {}: a coordinate is not a finite number",
                raw.image,
                i + 1
            )));
        }
        if scored && !b.score.is_finite() {
            return Err(bad(format!(
                "line {n} (image '{}'), box {}: score is not a finite number",
                raw.image,
                i + 1
            )));
        }
        if b.x2 < b.x1 || b.y2 < b.y1 {
            return Err(bad(format!(
                "line {n} (image '{}'), box {}: x2 {} or y2 {} is less than x1 {} or y1 {}",
                raw.image,
                i + 1,
                b.x2,
                b.y2,
                b.x1,
                b.y1
            ))
            .hint("a box is its corners in pixels: top-left (x1, y1), bottom-right (x2, y2), not x, y, width, height"));
        }
    }
    Ok(BenchPrediction {
        image: raw.image,
        condition: raw.condition,
        severity: raw.severity,
        boxes: raw
            .boxes
            .into_iter()
            .map(|b| folo_api::BenchDetection {
                class: b.class,
                // Unscored contestants ignore confidences (graded as 1.0); normalize upfront so non-finite
                // ignored scores never serialize as JSON `null` over HTTP.
                score: if scored { b.score } else { 1.0 },
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
            })
            .collect(),
    })
}

/// The next part of `records`: whole records, at most `limits.records` of them, `limits.boxes` boxes and
/// `limits.bytes` of record JSON; empty at the end. A record that could never fit is an error (an empty part would read as
/// the end of the file).
#[cfg(test)]
fn next_part(
    records: &mut Peekable<impl Iterator<Item = Result<BenchPrediction, Error>>>,
    limits: PartLimits,
) -> Result<Vec<BenchPrediction>, Error> {
    let (mut part, mut boxes, mut bytes) = (Vec::new(), 0usize, 0usize);
    while part.len() < limits.records {
        let size = match records.peek() {
            None => break,
            Some(Ok(r)) => match fit_record(r, boxes, bytes, limits)? {
                Some(size) => size,
                None => break,
            },
            Some(Err(_)) => 0,
        };
        let r = records.next().expect("peeked")?;
        boxes += r.boxes.len();
        bytes += size;
        part.push(r);
    }
    Ok(part)
}

/// Like [`next_part`], keeping each record's 1-based file line number alongside the part.
fn next_numbered_part(
    records: &mut Peekable<impl Iterator<Item = Result<(usize, BenchPrediction), Error>>>,
    limits: PartLimits,
) -> Result<(Vec<BenchPrediction>, Vec<usize>), Error> {
    let (mut part, mut lines, mut boxes, mut bytes) = (Vec::new(), Vec::new(), 0usize, 0usize);
    while part.len() < limits.records {
        let size = match records.peek() {
            None => break,
            Some(Ok((_, r))) => match fit_record(r, boxes, bytes, limits)? {
                Some(size) => size,
                None => break,
            },
            Some(Err(_)) => 0,
        };
        let (line, r) = records.next().expect("peeked")?;
        boxes += r.boxes.len();
        bytes += size;
        part.push(r);
        lines.push(line);
    }
    Ok((part, lines))
}

fn fit_record(
    r: &BenchPrediction,
    boxes: usize,
    bytes: usize,
    limits: PartLimits,
) -> Result<Option<usize>, Error> {
    let size = serde_json::to_vec(r).map_or(0, |b| b.len());
    if r.boxes.len() > limits.boxes || size > limits.bytes {
        return Err(invalid_arguments(format!(
            "a record of {} boxes and {size} bytes is over a part's {} boxes and {} bytes",
            r.boxes.len(),
            limits.boxes,
            limits.bytes
        )));
    }
    if boxes + r.boxes.len() > limits.boxes || bytes + size > limits.bytes {
        return Ok(None);
    }
    Ok(Some(size))
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

pub(super) fn detail(b: Bench, entries: &[BenchEntry]) -> BenchDetail {
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
#[path = "ops_tests/mod.rs"]
mod tests;
