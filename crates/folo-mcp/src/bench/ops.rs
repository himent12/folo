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
/// Longest line of a file of predictions: far more than a record of [`MAX_BOXES`] boxes takes, and all of a line that
/// is read before it is judged.
const MAX_RECORD_BYTES: usize = 1024 * 1024;

/// How large a part of a file of predictions may be.
#[derive(Clone, Copy, Debug)]
struct PartLimits {
    records: usize,
    boxes: usize,
}

const PARTS: PartLimits = PartLimits { records: MAX_PART_RECORDS, boxes: MAX_PART_BOXES };
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
        let contestant = BenchContestantSpec::External { scored, open: true, predictions: first };
        let req = StartBenchEntry { contestant, name: Some(name), max_images: None, conditions: None };
        let entry = self.start_entry(id, req)?;
        let sent = (|| {
            loop {
                if stop.load(Ordering::Relaxed) {
                    let message = "interrupted while sending the predictions";
                    return Err(Error::new(StatusCode::SERVICE_UNAVAILABLE, "interrupted", message));
                }
                let part = next_part(&mut records, limits)?;
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
            Err(e) if e.status.is_server_error() => self.after_lost_finish(id, &entry.id, e),
            Err(e) => Err(self.undo(id, &entry.id, e)),
            ok => ok,
        }
    }

    /// After finish answered `e`, a failure that says nothing of what the server did (the answer may be what was
    /// lost): an entry that ended scored is the result; one stopped from elsewhere is removed; one still open, or
    /// that cannot be read, may yet be scored, so it is kept and the error names it.
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
                let hint = match e.body.hint.as_deref() {
                    Some(h) => format!("{what}; {h}"),
                    None => what,
                };
                Err(e.hint(hint))
            }
        }
    }

    /// Stops and deletes entry `entry_id` after `e` (a failed submit); `e`'s hint says how that went. An entry
    /// that is no longer open (stopped from elsewhere, which may be why `e` happened) is deleted as it is.
    fn undo(&self, id: &str, entry_id: &str, e: Error) -> Error {
        let stopped = match self.stop_entry(id, entry_id) {
            Err(s) if s.body.code == "bench_entry_not_active" => Ok(()),
            other => other.map(|_| ()),
        };
        let what = match stopped.and_then(|_| self.delete_entry(id, entry_id)) {
            Ok(()) => format!("entry {entry_id} was stopped and deleted"),
            Err(u) => {
                format!("entry {entry_id} could not be removed ({}): delete it by hand", u.body.message)
            }
        };
        let hint = match e.body.hint.as_deref() {
            Some(h) => format!("{what}; {h}"),
            None => what,
        };
        e.hint(hint)
    }
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
    let mut reader = BufReader::new(file);
    let mut n = 0usize;
    Ok(std::iter::from_fn(move || {
        let bad = |what: String| invalid_arguments(format!("{shown}: {what}"));
        loop {
            n += 1;
            let mut line = Vec::new();
            // One byte over the limit tells a line that is too long from one that just fits.
            let limit = MAX_RECORD_BYTES as u64 + 1;
            match (&mut reader).take(limit).read_until(b'\n', &mut line) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(bad(format!("reading line {n}: {e}")))),
            }
            let text = line.trim_ascii();
            if text.is_empty() {
                continue;
            }
            return Some(parse_record(
                text,
                line.len() > MAX_RECORD_BYTES && !line.ends_with(b"\n"),
                n,
                &bad,
            ));
        }
    }))
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
    if text[0] != b'{' {
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

/// The next part of `records`: whole records, at most `limits.records` of them and `limits.boxes` boxes; empty at
/// the end. A record that could never fit is an error (an empty part would read as the end of the file).
fn next_part(
    records: &mut Peekable<impl Iterator<Item = Result<BenchPrediction, Error>>>,
    limits: PartLimits,
) -> Result<Vec<BenchPrediction>, Error> {
    let (mut part, mut boxes) = (Vec::new(), 0usize);
    while part.len() < limits.records {
        let fits = match records.peek() {
            None => break,
            Some(Ok(r)) if r.boxes.len() > limits.boxes => {
                let n = r.boxes.len();
                return Err(invalid_arguments(format!(
                    "a record has {n} boxes, over a part's {}",
                    limits.boxes
                )));
            }
            Some(Ok(r)) => boxes + r.boxes.len() <= limits.boxes,
            Some(Err(_)) => true,
        };
        if !fits {
            break;
        }
        let r = records.next().expect("peeked")?;
        boxes += r.boxes.len();
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
mod tests {
    use super::*;
    use folo_test_support::TestFile;

    fn records(text: &str) -> Result<Vec<BenchPrediction>, Error> {
        let f = TestFile::new("bench-predictions", "p.jsonl");
        std::fs::write(&f, text).unwrap();
        read_predictions(&f)?.collect()
    }

    #[test]
    fn a_file_of_predictions_is_json_lines_of_records() {
        let r = |image: &str| {
            format!(r#"{{"image":"{image}","boxes":[{{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}}]}}"#)
        };
        let lines = format!("{}\n{}\n", r("1"), r("2"));
        let got: Vec<String> = records(&lines).unwrap().into_iter().map(|p| p.image).collect();
        assert_eq!(got, ["1", "2"]);
        assert_eq!(records(&r("9")).unwrap()[0].boxes[0].score, 1.0);
        assert!(records("").unwrap().is_empty());
        // Blank lines are skipped, and line numbers still count them.
        let e = records(&format!("{}\n\n{{\"image\":\"3\"}}\n", r("1"))).unwrap_err();
        assert!(e.body.message.contains("line 3") && e.body.message.contains("boxes"), "{e:?}");
        // Arrays and API bodies are not records.
        for doc in [format!("[{}]", r("1")), format!("{{\"predictions\": [{}]}}", r("1"))] {
            let e = records(&format!("{}\n{doc}\n", r("1"))).unwrap_err();
            assert!(e.body.message.contains("line 2 is not a record"), "{e:?}");
        }
        // Bad JSON says where; a bad record says which.
        let e = records(&format!("{}\n{{\"image\": ", r("1"))).unwrap_err();
        assert!(e.body.code == "invalid_arguments" && e.body.message.contains("line 2"), "{e:?}");
        let e = records(&format!("{}\n{{\"image\":\"2\"}}\n", r("1"))).unwrap_err();
        assert!(e.body.message.contains("line 2") && e.body.message.contains("boxes"), "{e:?}");
        let missing = read_predictions(Path::new("no/such/predictions.jsonl")).err().unwrap();
        assert_eq!(missing.body.code, "path_not_found");
    }

    /// A line is judged after at most [`MAX_RECORD_BYTES`]: one array or object on a single huge line is refused
    /// without being read whole, and a line of exactly the limit is still a record.
    #[test]
    fn a_long_line_is_refused_before_it_is_read_whole() {
        let array = format!("[{}]", "0,".repeat(MAX_RECORD_BYTES));
        assert!(records(&array).unwrap_err().body.message.contains("line 1 is not a record"));
        let object = format!("{{\"image\":\"{}\",\"boxes\":[]}}", "x".repeat(MAX_RECORD_BYTES));
        let e = records(&object).unwrap_err();
        assert!(e.body.message.contains("line 1 is longer"), "{e:?}");
        let head = "{\"image\":\"";
        let tail = "\",\"boxes\":[]}";
        let exact = format!("{head}{}{tail}", "x".repeat(MAX_RECORD_BYTES - head.len() - tail.len()));
        assert_eq!(exact.len(), MAX_RECORD_BYTES);
        assert_eq!(records(&format!("{exact}\n")).unwrap().len(), 1);
    }

    /// A record over the server's limit per record is refused where it is read, so no part can pass a part's limit.
    #[test]
    fn a_record_over_the_box_limit_is_refused() {
        let boxes = |n: usize| vec![r#"{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}"#; n].join(",");
        let line = |n: usize| format!("{{\"image\":\"1\",\"boxes\":[{}]}}\n", boxes(n));
        assert_eq!(records(&line(MAX_BOXES)).unwrap()[0].boxes.len(), MAX_BOXES);
        let e = records(&line(MAX_BOXES + 1)).unwrap_err();
        assert!(e.body.message.contains("has 1001 boxes"), "{e:?}");
    }

    fn record(image: &str, boxes: usize) -> BenchPrediction {
        BenchPrediction {
            image: image.into(),
            condition: None,
            severity: None,
            boxes: vec![
                folo_api::BenchDetection { class: 0, score: 1.0, x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
                boxes
            ],
        }
    }

    #[test]
    fn parts_hold_whole_records_within_the_limits() {
        let sizes = |p: Vec<BenchPrediction>| p.iter().map(|r| r.boxes.len()).collect::<Vec<_>>();
        // Boxes: a record that would cross the limit starts the next part.
        let limits = PartLimits { records: 100, boxes: 10 };
        let mut it = [4, 5, 2, 10, 0].map(|b| Ok(record("1", b))).into_iter().peekable();
        assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [4, 5]);
        assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [2]);
        assert_eq!(sizes(next_part(&mut it, limits).unwrap()), [10, 0]);
        assert!(next_part(&mut it, limits).unwrap().is_empty());
        // A record that fits no part is an error, not an empty part (which would end the file there).
        let mut it = [11].map(|b| Ok(record("1", b))).into_iter().peekable();
        assert!(next_part(&mut it, limits).unwrap_err().body.message.contains("11 boxes"));
        // Records, empty ones too.
        let limits = PartLimits { records: 2, boxes: 10 };
        let mut it = [0, 0, 0].map(|b| Ok(record("1", b))).into_iter().peekable();
        assert_eq!(next_part(&mut it, limits).unwrap().len(), 2);
        assert_eq!(next_part(&mut it, limits).unwrap().len(), 1);
        // The real limits: 199,999 boxes and then 1,000 make two parts.
        let mut it = [MAX_PART_BOXES - 1, 1000].map(|b| Ok(record("1", b))).into_iter().peekable();
        assert_eq!(sizes(next_part(&mut it, PARTS).unwrap()), [MAX_PART_BOXES - 1]);
        assert_eq!(sizes(next_part(&mut it, PARTS).unwrap()), [1000]);
    }

    /// A store with a benchmark of 4 synthetic images and haze 2 in its plan; its id.
    fn bench_store(home: &Path) -> (std::sync::Arc<folo_server::RealStore>, String) {
        use folo_server::Store;
        let store = std::sync::Arc::new(folo_server::RealStore::open(home.to_path_buf()).device("cpu"));
        let req = folo_api::SyntheticDataset {
            name: "shapes".into(),
            images: 20,
            classes: vec!["circle".into(), "square".into()],
            imgsz: 64,
            seed: 3,
        };
        let ds = store.create_synthetic(req).unwrap();
        let plan = BenchPlan {
            conditions: vec![folo_api::BenchPlanCondition {
                condition: "haze".into(),
                severities: vec![2],
                weight: 1.0,
                why: None,
            }],
            rationale: None,
            designed_by: None,
        };
        let req = CreateBench {
            name: "shapes".into(),
            description: String::new(),
            source: folo_api::BenchSource::Dataset { dataset_id: ds.id, split: None },
            max_images: None,
            seed: 0,
            plan: Some(plan),
        };
        let bench = Benchmarks::Store(store.clone()).create_bench(req).unwrap();
        assert_eq!(bench.images, 4);
        (store, bench.id)
    }

    /// A file of predictions in several parts: the parts after the first (the condition's records here) are sent
    /// and scored; a later part refused removes the entry.
    fn parts(b: &Benchmarks, id: &str, dir: &Path) {
        // Nothing found anywhere: 4 records of the unchanged images, then 4 of haze 2.
        let file = dir.join("p.jsonl");
        let write = |haze_image: &str| {
            let mut lines: Vec<String> =
                (1..=4).map(|i| format!(r#"{{"image":"{i}","boxes":[]}}"#)).collect();
            lines.extend(
                ["1", "2", "3", haze_image]
                    .map(|i| format!(r#"{{"image":"{i}","condition":"haze","severity":2,"boxes":[]}}"#)),
            );
            std::fs::write(&file, lines.join("\n")).unwrap();
        };
        let three = PartLimits { records: 3, boxes: 10 };
        write("4");
        let go = AtomicBool::new(false);
        let e = b.submit_in(id, "parts".into(), true, &file, three, &go).unwrap();
        let state = b.state_of(e).unwrap();
        assert_eq!((state.status, state.done), (folo_api::BenchEntryStatus::Done, 8), "{state:?}");
        let r = state.result.unwrap();
        assert_eq!((r.conditions.len(), r.conditions[0].points.len()), (1, 1), "{:?}", r.conditions);
        // The third part names an image the benchmark does not have: refused, and the entry is gone.
        write("99");
        let e = b.submit_in(id, "refused".into(), true, &file, three, &go).unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.message.contains("99") && hint.contains("stopped and deleted"), "{e:?}");
        assert_eq!(b.entries(id).unwrap().len(), 1);
        // Ctrl+C after the first part: the entry is removed, and the error says it was interrupted.
        write("4");
        let e = b.submit_in(id, "stopped".into(), true, &file, three, &AtomicBool::new(true)).unwrap_err();
        let hint = e.body.hint.clone().unwrap_or_default();
        assert!(e.body.code == "interrupted" && hint.contains("stopped and deleted"), "{e:?}");
        assert_eq!(b.entries(id).unwrap().len(), 1);
    }

    /// A finish whose answer was lost: what the server did decides, and a score it saved is never deleted.
    #[test]
    fn a_lost_finish_answer_keeps_what_the_server_did() {
        let home = folo_test_support::TestDir::new("mcp-bench-lost-finish");
        let (store, id) = bench_store(home.path());
        let b = Benchmarks::Store(store.clone());
        let lost = || Error::new(StatusCode::BAD_GATEWAY, "server_unreachable", "no answer");
        let open = |name: &str| {
            let predictions = (1..=4).map(|i| record(&i.to_string(), 0)).collect();
            let contestant = BenchContestantSpec::External { scored: true, open: true, predictions };
            let req =
                StartBenchEntry { contestant, name: Some(name.into()), max_images: None, conditions: None };
            b.start_entry(&id, req).unwrap().id
        };
        // Scored before the answer was lost: the result, and the entry stays.
        let scored = open("scored");
        b.finish(&id, &scored).unwrap();
        let got = b.after_lost_finish(&id, &scored, lost()).unwrap();
        assert_eq!(got.status, BenchEntryStatus::Done);
        assert_eq!(b.entry(&id, &scored).unwrap().status, BenchEntryStatus::Done);
        // Still open: kept, and the error names it.
        let kept = open("kept");
        let e = b.after_lost_finish(&id, &kept, lost()).unwrap_err();
        assert!(e.body.hint.as_deref().unwrap_or_default().contains(&kept), "{e:?}");
        assert!(b.entry(&id, &kept).unwrap().status.is_active());
        // Stopped from elsewhere: removed as any failed upload is.
        b.stop_entry(&id, &kept).unwrap();
        let e = b.after_lost_finish(&id, &kept, lost()).unwrap_err();
        assert!(e.body.hint.as_deref().unwrap_or_default().contains("stopped and deleted"), "{e:?}");
        assert!(b.entry(&id, &kept).is_err());
        folo_server::Store::shutdown(&*store);
    }

    #[test]
    fn predictions_go_in_parts_in_process() {
        let home = folo_test_support::TestDir::new("mcp-bench-parts");
        let (store, id) = bench_store(home.path());
        parts(&Benchmarks::Store(store.clone()), &id, home.path());
        folo_server::Store::shutdown(&*store);
    }

    #[test]
    fn predictions_go_in_parts_through_a_server() {
        let home = folo_test_support::TestDir::new("mcp-bench-parts-http");
        let (store, id) = bench_store(&home.join("server"));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
        let serving = rt.spawn(server.serve(listener, async move {
            let _ = stopped.await;
        }));
        let http = std::sync::Arc::new(crate::proxy::HttpStore::new(&url, None));
        parts(&Benchmarks::Http(http), &id, home.path());
        let _ = stop.send(());
        rt.block_on(serving).unwrap().unwrap();
        folo_server::Store::shutdown(&*store);
    }
}
