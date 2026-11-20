//! The benchmark operations the MCP tools and `folo bench` share, built on the calls of [`Benchmarks`]: each
//! answers with a result of [`crate::types::bench`]. Their errors carry no front end's hint ([`super::hints`] adds
//! one).

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Duration;

use folo_api::{
    Bench, BenchCatalog, BenchContestant, BenchContestantSpec, BenchEntry, BenchPlan, BenchPrediction,
    BenchPredictions, BenchResult, CreateBench, StartBenchEntry,
};
use http::StatusCode;
use serde_json::Value;

use super::Benchmarks;
use crate::errors::{Error, invalid_arguments};
use crate::types::bench::*;

/// Most boxes in one part of a file of predictions sent to the server (about 15 MB of JSON; a request takes 64 MiB).
pub const MAX_PART_BOXES: usize = 200_000;
/// Most records in one part.
const MAX_PART_RECORDS: usize = 20_000;
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
        let entry = match who {
            Contestant::Model { model_id, max_images, conditions } => {
                let contestant = BenchContestantSpec::Model { model_id };
                self.start_entry(id, StartBenchEntry { contestant, name, max_images, conditions })?
            }
            Contestant::Predictions { path, scored } => {
                let name = name.filter(|n| !n.trim().is_empty()).ok_or_else(|| {
                    invalid_arguments("an entry of predictions needs a name for the leaderboard")
                })?;
                self.submit(id, name, scored, &path)?
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

    /// An external entry from the file at `path`: opened with the file's first part, sent the rest part by part
    /// and finished (scored). A file that does not parse, or a part the server refuses, stops and deletes the
    /// entry, so none is left half-filled; the error says so.
    fn submit(&self, id: &str, name: String, scored: bool, path: &Path) -> Result<BenchEntry, Error> {
        let mut records = read_predictions(path)?;
        let first = next_part(&mut records)?;
        if first.is_empty() {
            return Err(invalid_arguments(format!("{} holds no predictions", folo_api::json::path(path))));
        }
        let contestant = BenchContestantSpec::External { scored, open: true, predictions: first };
        let req = StartBenchEntry { contestant, name: Some(name), max_images: None, conditions: None };
        let entry = self.start_entry(id, req)?;
        let sent = (|| {
            loop {
                let part = next_part(&mut records)?;
                if part.is_empty() {
                    return self.finish(id, &entry.id);
                }
                self.add_predictions(id, &entry.id, part)?;
            }
        })();
        sent.map_err(|e| self.undo(id, &entry.id, e))
    }

    /// Stops and deletes entry `entry_id` after `e` (a failed submit); `e`'s hint says how that went.
    fn undo(&self, id: &str, entry_id: &str, e: Error) -> Error {
        let what = match self.stop_entry(id, entry_id).and_then(|_| self.delete_entry(id, entry_id)) {
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

/// The records of a file of predictions, in order: JSON Lines of `BenchPrediction` records (what
/// `examples/bench_external.py` sends, `{"image", "condition"?, "severity"?, "boxes": [...]}`), or JSON documents
/// that hold them (an array of records, or a `{"predictions": [...]}` body of the HTTP API), one after the other.
/// The file is read as the records are taken, so a large file is never held whole.
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
    let values = serde_json::Deserializer::from_reader(BufReader::new(file)).into_iter::<Value>();
    let mut n = 0usize;
    let bad = move |what: String| invalid_arguments(format!("{shown}: {what}"));
    Ok(values.flat_map(move |v| -> Vec<Result<BenchPrediction, Error>> {
        let v = match v {
            Ok(v) => v,
            Err(e) => {
                return vec![Err(bad(format!("not JSON at line {}, column {}: {e}", e.line(), e.column())))];
            }
        };
        let records = match v {
            Value::Array(a) => a,
            Value::Object(o) if o.contains_key("predictions") => {
                match serde_json::from_value::<BenchPredictions>(Value::Object(o)) {
                    Ok(p) => {
                        return p
                            .predictions
                            .into_iter()
                            .map(|r| {
                                n += 1;
                                Ok(r)
                            })
                            .collect();
                    }
                    Err(e) => return vec![Err(bad(format!("a `predictions` document: {e}")))],
                }
            }
            other => vec![other],
        };
        records
            .into_iter()
            .map(|r| {
                n += 1;
                serde_json::from_value::<BenchPrediction>(r).map_err(|e| bad(format!("record {n}: {e}")))
            })
            .collect()
    }))
}

/// The next part of `records`: up to [`MAX_PART_BOXES`] boxes or [`MAX_PART_RECORDS`] records (empty at the end).
fn next_part(
    records: &mut impl Iterator<Item = Result<BenchPrediction, Error>>,
) -> Result<Vec<BenchPrediction>, Error> {
    let (mut part, mut boxes) = (Vec::new(), 0usize);
    while part.len() < MAX_PART_RECORDS && boxes < MAX_PART_BOXES {
        let Some(r) = records.next() else { break };
        let r = r?;
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
    fn a_file_of_predictions_is_records_or_documents_of_them() {
        let r = |image: &str| {
            format!(r#"{{"image":"{image}","boxes":[{{"class":0,"x1":1,"y1":2,"x2":3,"y2":4}}]}}"#)
        };
        // JSON Lines, an array, an API body, and all three in one file, in order.
        let lines = format!("{}\n{}\n", r("1"), r("2"));
        assert_eq!(records(&lines).unwrap().len(), 2);
        let mixed = format!("{}\n[{}, {}]\n{{\"predictions\": [{}]}}\n", r("1"), r("2"), r("3"), r("4"));
        let got: Vec<String> = records(&mixed).unwrap().into_iter().map(|p| p.image).collect();
        assert_eq!(got, ["1", "2", "3", "4"]);
        assert_eq!(records(&format!("{{\"predictions\": [{}]}}", r("9"))).unwrap()[0].boxes[0].score, 1.0);
        assert!(records("").unwrap().is_empty());
        // Bad JSON says where; a bad record says which.
        let e = records(&format!("{}\n{{\"image\": ", r("1"))).unwrap_err();
        assert!(e.body.code == "invalid_arguments" && e.body.message.contains("line 2"), "{e:?}");
        let e = records(&format!("{}\n{{\"image\":\"2\"}}\n", r("1"))).unwrap_err();
        assert!(e.body.message.contains("record 2") && e.body.message.contains("boxes"), "{e:?}");
        let missing = read_predictions(Path::new("no/such/predictions.jsonl")).err().unwrap();
        assert_eq!(missing.body.code, "path_not_found");
    }

    #[test]
    fn parts_hold_whole_records_up_to_the_box_budget() {
        let record = |boxes: usize| BenchPrediction {
            image: "1".into(),
            condition: None,
            severity: None,
            boxes: vec![
                folo_api::BenchDetection { class: 0, score: 1.0, x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
                boxes
            ],
        };
        let mut it = [1000, MAX_PART_BOXES - 1000, 1, 5].map(|b| Ok(record(b))).into_iter();
        let sizes = |p: Vec<BenchPrediction>| p.iter().map(|r| r.boxes.len()).collect::<Vec<_>>();
        assert_eq!(sizes(next_part(&mut it).unwrap()), [1000, MAX_PART_BOXES - 1000]);
        assert_eq!(sizes(next_part(&mut it).unwrap()), [1, 5]);
        assert!(next_part(&mut it).unwrap().is_empty());
    }
}
