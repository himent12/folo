//! External entries: predictions made anywhere else, checked strictly and gathered into what an entry is graded
//! on. Every error names the first record that is wrong, in plain words.
//!
//! An entry arrives in one request or in parts ([`Filling`]: an open entry). Each part is checked whole before
//! any of it is taken, so a part that is refused can be fixed and sent again. A (condition, severity) is graded
//! (its mAP pair, all the result needs of it) as soon as its last image has arrived and its boxes are dropped,
//! so a client that sends pass by pass costs the server one pass of memory; the unchanged images' detections stay
//! until the entry is finished (the evaluation and the per-image outcomes need them).

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use axum::http::StatusCode;
use folo_api::{BenchDetection, BenchPrediction};
use folo_bench::score::{self, Map, Sample};
use folo_bench::{Condition, conditions};

use super::state::BenchFile;
use crate::error::{Error, Result};

/// Most boxes one record carries.
pub const MAX_BOXES: usize = 1000;

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

/// How a record is shown in a message: `prediction 3 (image '17', haze 3)`.
fn record(i: usize, p: &BenchPrediction) -> String {
    match (&p.condition, p.severity) {
        (Some(c), Some(s)) => format!("prediction {} (image '{}', {c} {s})", i + 1, p.image),
        (Some(c), None) => format!("prediction {} (image '{}', {c})", i + 1, p.image),
        _ => format!("prediction {} (image '{}', unchanged)", i + 1, p.image),
    }
}

/// The answers of one pass (the unchanged images, or one planned variant) so far.
struct PassFill {
    /// Per image: answered.
    seen: Vec<bool>,
    /// Per image: its detections, until the pass is graded.
    boxes: Vec<Vec<BenchDetection>>,
    count: usize,
    /// The variant's mAP pair, once every image has arrived (its boxes are then dropped).
    map: Option<Map>,
}

impl PassFill {
    fn new(n: usize) -> PassFill {
        PassFill { seen: vec![false; n], boxes: vec![Vec::new(); n], count: 0, map: None }
    }
}

/// What a finished external entry is graded on.
pub struct Finished {
    /// The unchanged images, in the benchmark's order.
    pub clean: Vec<Sample>,
    /// The mAP pair of every (condition, severity) of the conditions that were answered, in plan order.
    pub runs: Vec<(Condition, u8, Map)>,
    /// Every condition of the plan was answered.
    pub complete: bool,
}

/// An external entry being filled: what has arrived, checked against the benchmark. See the module docs.
pub struct Filling {
    file: BenchFile,
    scored: bool,
    variants: Vec<(Condition, u8)>,
    index: HashMap<String, usize>,
    clean: PassFill,
    passes: Vec<PassFill>,
    /// Records received.
    pub received: u64,
    /// The entry was finished or stopped: nothing more is taken.
    pub closed: bool,
    /// When its progress was last sent.
    pub last_event: Option<Instant>,
}

impl Filling {
    /// An empty entry of the benchmark `file` (its images, classes and plan).
    pub fn new(file: BenchFile, scored: bool) -> Filling {
        let variants = folo_bench::variants(&file.plan);
        let n = file.images.len();
        let index = file.images.iter().enumerate().map(|(i, im)| (im.id.clone(), i)).collect();
        Filling {
            passes: variants.iter().map(|_| PassFill::new(n)).collect(),
            clean: PassFill::new(n),
            variants,
            index,
            file,
            scored,
            received: 0,
            closed: false,
            last_event: None,
        }
    }

    /// Whether the contestant gives confidences.
    pub fn scored(&self) -> bool {
        self.scored
    }

    /// The benchmark the entry is of.
    pub fn file(&self) -> &BenchFile {
        &self.file
    }

    /// The most records the entry can have: every image unchanged and under every planned variant.
    pub fn capacity(&self) -> u64 {
        (self.file.images.len() * (1 + self.variants.len())) as u64
    }

    /// Takes a part: every record is checked first (`400 invalid_body` for the first that is wrong, and then none
    /// of the part is taken): an image the benchmark does not have, a condition that is not in the catalog or a
    /// (condition, severity) the plan does not run, a severity without its condition or the other way round, more
    /// than [`MAX_BOXES`] boxes, a class outside the benchmark's, a score outside `0..=1` (when scored), a record
    /// that repeats one of this part or one received before. Variants whose last image arrived are graded.
    pub fn add(&mut self, predictions: Vec<BenchPrediction>) -> Result<()> {
        let classes = self.file.classes.len();
        // The pass (`None`: unchanged) and image of each record.
        let mut targets: Vec<(Option<usize>, usize)> = Vec::with_capacity(predictions.len());
        let mut here: HashMap<(Option<usize>, usize), usize> = HashMap::new();
        for (i, p) in predictions.iter().enumerate() {
            let what = record(i, p);
            let Some(&image) = self.index.get(p.image.as_str()) else {
                return Err(invalid(format!("{what}: the benchmark has no image '{}'", p.image))
                    .hint("GET /api/bench/benchmarks/{id}/images lists its image ids"));
            };
            let pass = match (&p.condition, p.severity) {
                (None, None) => None,
                (None, Some(_)) => return Err(invalid(format!("{what}: a severity needs its condition"))),
                (Some(c), None) => {
                    return Err(invalid(format!("{what}: condition '{c}' needs its severity")));
                }
                (Some(c), Some(s)) => {
                    let Some(cond) = Condition::parse(c) else {
                        return Err(invalid(format!("{what}: '{c}' is not a condition of the catalog"))
                            .hint("GET /api/bench/catalog lists the conditions"));
                    };
                    let Some(v) = self.variants.iter().position(|&(vc, vs)| vc == cond && vs == s) else {
                        let planned: Vec<String> = self
                            .file
                            .plan
                            .conditions
                            .iter()
                            .map(|pc| format!("{} {:?}", pc.condition, pc.severities))
                            .collect();
                        let runs = if planned.is_empty() {
                            "the unchanged images only".to_string()
                        } else {
                            planned.join(", ")
                        };
                        return Err(invalid(format!(
                            "{what}: the benchmark's plan does not run {c} at severity {s} (it runs {runs})"
                        )));
                    };
                    Some(v)
                }
            };
            if let Some(first) = here.insert((pass, image), i) {
                return Err(invalid(format!(
                    "{what} repeats prediction {}: answer each image and condition once",
                    first + 1
                )));
            }
            let seen = match pass {
                None => self.clean.seen[image],
                Some(v) => self.passes[v].seen[image],
            };
            if seen {
                return Err(invalid(format!(
                    "{what} was received in an earlier part: answer each image and condition once"
                ))
                .hint("send only the records not sent yet; this part was not taken"));
            }
            if p.boxes.len() > MAX_BOXES {
                return Err(invalid(format!(
                    "{what} has {} boxes; a record carries at most {MAX_BOXES}",
                    p.boxes.len()
                )));
            }
            for (b, d) in p.boxes.iter().enumerate() {
                if d.class as usize >= classes {
                    return Err(invalid(format!(
                        "{what}, box {}: class {} is not one of the benchmark's {classes} classes (0 to {})",
                        b + 1,
                        d.class,
                        classes.saturating_sub(1)
                    )));
                }
                if self.scored && !(0.0..=1.0).contains(&d.score) {
                    return Err(invalid(format!(
                        "{what}, box {}: score {} is outside 0..=1",
                        b + 1,
                        d.score
                    )));
                }
            }
            targets.push((pass, image));
        }
        // Every record is good: take them all.
        let mut touched = HashSet::new();
        for (p, (pass, image)) in predictions.into_iter().zip(targets) {
            let fill = match pass {
                None => &mut self.clean,
                Some(v) => {
                    touched.insert(v);
                    &mut self.passes[v]
                }
            };
            fill.seen[image] = true;
            fill.boxes[image] = p.boxes;
            fill.count += 1;
            self.received += 1;
        }
        for v in touched {
            if self.passes[v].count == self.file.images.len() {
                self.grade(v);
            }
        }
        Ok(())
    }

    /// Grades variant `v` (every image has arrived) and drops its boxes.
    fn grade(&mut self, v: usize) {
        let (c, s) = self.variants[v];
        let boxes = std::mem::take(&mut self.passes[v].boxes);
        let samples: Vec<Sample> = boxes
            .into_iter()
            .zip(&self.file.images)
            .map(|(detections, im)| {
                // The truth under the condition, as the image was served.
                let truths = conditions::moved_boxes(c, s, im.width, im.height, &im.boxes)
                    .expect("planned severities are valid");
                Sample { truths, detections }
            })
            .collect();
        self.passes[v].map = Some(score::map(&samples, self.file.classes.len(), self.scored));
    }

    /// Checks that the entry is whole and hands out what it is graded on: `400 invalid_body` for an image without
    /// its unchanged record, and for a condition answered for some images or severities but not all (the entry
    /// then stays as it is, to be completed).
    pub fn finish(&mut self) -> Result<Finished> {
        if let Some(missing) = self.clean.seen.iter().position(|s| !s) {
            return Err(invalid(format!(
                "no prediction for image '{}' (unchanged): every image of the benchmark must be answered for, an \
                 image with nothing found with an empty boxes list",
                self.file.images[missing].id
            )));
        }
        // A condition counts when every image is answered at every one of its severities; one that is partly
        // answered is an error, so a robustness number is never made of a part of a condition.
        let mut complete = true;
        let mut runs = Vec::new();
        for pc in &self.file.plan.conditions {
            let Some(cond) = Condition::parse(&pc.condition) else { continue };
            let of: Vec<usize> =
                self.variants.iter().enumerate().filter(|(_, v)| v.0 == cond).map(|(i, _)| i).collect();
            if !of.iter().any(|&v| self.passes[v].count > 0) {
                complete = false;
                continue;
            }
            for &v in &of {
                if let Some(missing) = self.passes[v].seen.iter().position(|s| !s) {
                    return Err(invalid(format!(
                        "condition '{}' is answered, but not for image '{}' at severity {}: answer a condition \
                         for every image at every planned severity ({:?}), or leave it out",
                        pc.condition, self.file.images[missing].id, self.variants[v].1, pc.severities
                    )));
                }
            }
            for &v in &of {
                let (c, s) = self.variants[v];
                runs.push((c, s, self.passes[v].map.expect("a whole variant is graded")));
            }
        }
        let boxes = std::mem::take(&mut self.clean.boxes);
        let clean = boxes
            .into_iter()
            .zip(&self.file.images)
            .map(|(detections, im)| Sample { truths: im.boxes.clone(), detections })
            .collect();
        self.closed = true;
        Ok(Finished { clean, runs, complete })
    }
}

#[cfg(test)]
mod tests {
    use folo_api::{BBox, BenchPlan, BenchPlanCondition, BenchSource};

    use super::super::state::{ImageRecord, VERSION};
    use super::*;

    fn file() -> BenchFile {
        let image = |id: &str| ImageRecord {
            id: id.into(),
            file: format!("{id}.jpg"),
            width: 100,
            height: 100,
            bytes: 1,
            sha256: id.into(),
            boxes: vec![BBox { class: 0, x1: 10.0, y1: 10.0, x2: 50.0, y2: 50.0 }],
        };
        BenchFile {
            version: VERSION,
            id: "bn_1".into(),
            name: "b".into(),
            description: String::new(),
            classes: vec!["a".into(), "b".into()],
            created: String::new(),
            source: BenchSource::LabelJob { job_id: "lb_1".into() },
            source_name: String::new(),
            seed: 0,
            plan: BenchPlan {
                conditions: vec![
                    BenchPlanCondition {
                        condition: "haze".into(),
                        severities: vec![1, 3],
                        weight: 1.0,
                        why: None,
                    },
                    BenchPlanCondition {
                        condition: "distance".into(),
                        severities: vec![2],
                        weight: 1.0,
                        why: None,
                    },
                ],
                rationale: None,
                designed_by: None,
            },
            images: vec![image("1"), image("2")],
        }
    }

    fn p(image: &str, cond: Option<(&str, u8)>) -> BenchPrediction {
        BenchPrediction {
            image: image.into(),
            condition: cond.map(|c| c.0.into()),
            severity: cond.map(|c| c.1),
            boxes: vec![BenchDetection { class: 0, score: 0.5, x1: 10.0, y1: 10.0, x2: 50.0, y2: 50.0 }],
        }
    }

    /// Every record of [`file`]'s benchmark.
    fn all() -> Vec<BenchPrediction> {
        let mut all = vec![p("2", None), p("1", None)];
        for im in ["1", "2"] {
            all.extend([p(im, Some(("haze", 1))), p(im, Some(("haze", 3))), p(im, Some(("distance", 2)))]);
        }
        all
    }

    fn check(f: &BenchFile, scored: bool, preds: &[BenchPrediction]) -> Result<Finished> {
        let mut fill = Filling::new(f.clone(), scored);
        fill.add(preds.to_vec())?;
        fill.finish()
    }

    #[test]
    fn complete_partial_and_moved_truth() {
        let f = file();
        let c = check(&f, true, &all()).unwrap();
        assert!(c.complete);
        assert_eq!(
            c.runs.iter().map(|r| (r.0.id(), r.1)).collect::<Vec<_>>(),
            [("haze", 1), ("haze", 3), ("distance", 2)]
        );
        assert_eq!(c.clean.len(), 2);
        // Haze keeps the truth (a perfect answer), distance moves it (the same boxes are wrong there).
        assert_eq!(c.runs[0].2.map50, 1.0);
        assert_eq!(c.runs[2].2.map50, 0.0);
        let only_clean = check(&f, true, &all()[..2]).unwrap();
        assert!(!only_clean.complete && only_clean.runs.is_empty());
    }

    #[test]
    fn parts_in_any_order_come_to_what_one_request_does() {
        let f = file();
        let whole = check(&f, true, &all()).unwrap();
        let mut fill = Filling::new(f.clone(), true);
        let mut records = all();
        records.reverse();
        // Variants first (each graded as its last image arrives), the unchanged images last.
        for part in records.chunks(2) {
            fill.add(part.to_vec()).unwrap();
        }
        assert!(fill.passes.iter().all(|v| v.map.is_some() && v.boxes.is_empty()), "graded and dropped");
        assert_eq!(fill.received, 8);
        let parts = fill.finish().unwrap();
        assert_eq!(parts.clean, whole.clean);
        assert_eq!(parts.runs, whole.runs);
        assert!(fill.closed);
    }

    #[test]
    fn a_part_is_all_or_nothing_and_repeats_are_refused_across_parts() {
        let f = file();
        let mut fill = Filling::new(f.clone(), true);
        fill.add(vec![p("1", None)]).unwrap();
        // A part with a bad record takes none of its records.
        let e = fill.add(vec![p("2", None), p("9", None)]).unwrap_err();
        assert!(e.body.message.contains("no image '9'"), "{}", e.body.message);
        assert_eq!(fill.received, 1);
        let e = fill.add(vec![p("1", None)]).unwrap_err();
        assert!(e.body.message.contains("was received in an earlier part"), "{}", e.body.message);
        // Unfinished: it says what is missing and stays open.
        let e = fill.finish().err().unwrap();
        assert!(e.body.message.contains("no prediction for image '2'"), "{}", e.body.message);
        assert!(!fill.closed);
        fill.add(vec![p("2", None)]).unwrap();
        assert!(!fill.finish().unwrap().complete);
    }

    #[test]
    fn every_wrong_record_is_named() {
        let f = file();
        let base = vec![p("1", None), p("2", None)];
        let cases: Vec<(Vec<BenchPrediction>, &str)> = vec![
            (
                vec![p("1", None), p("9", None)],
                "prediction 2 (image '9', unchanged): the benchmark has no image '9'",
            ),
            (vec![p("1", Some(("fog", 1)))], "'fog' is not a condition of the catalog"),
            (vec![p("1", Some(("jpeg", 1)))], "the benchmark's plan does not run jpeg at severity 1"),
            (vec![p("1", Some(("haze", 2)))], "does not run haze at severity 2"),
            (vec![p("1", None), p("1", None)], "prediction 2 (image '1', unchanged) repeats prediction 1"),
            (vec![BenchPrediction { severity: Some(1), ..p("1", None) }], "a severity needs its condition"),
            (
                vec![BenchPrediction { severity: None, ..p("1", Some(("haze", 1))) }],
                "condition 'haze' needs its severity",
            ),
            (vec![p("1", None)], "no prediction for image '2' (unchanged)"),
            (
                [
                    base.clone(),
                    vec![p("1", Some(("haze", 1))), p("2", Some(("haze", 1))), p("1", Some(("haze", 3)))],
                ]
                .concat(),
                "condition 'haze' is answered, but not for image '2' at severity 3",
            ),
        ];
        for (preds, want) in cases {
            let e = check(&f, true, &preds).err().expect(want);
            assert_eq!(e.body.code, "invalid_body");
            assert!(e.body.message.contains(want), "{} / {want}", e.body.message);
        }
        let mut bad_class = base.clone();
        bad_class[1].boxes[0].class = 2;
        let e = check(&f, true, &bad_class).err().unwrap();
        assert!(
            e.body.message.contains("prediction 2 (image '2', unchanged), box 1: class 2"),
            "{}",
            e.body.message
        );
        let mut bad_score = base.clone();
        bad_score[0].boxes[0].score = 1.5;
        assert!(check(&f, true, &bad_score).err().unwrap().body.message.contains("score 1.5"));
        // Without confidences the scores are not read.
        assert!(check(&f, false, &bad_score).is_ok());
        let mut many = base.clone();
        many[0].boxes = vec![many[0].boxes[0]; MAX_BOXES + 1];
        let e = check(&f, true, &many).err().unwrap();
        assert!(
            e.body.message.contains("has 1001 boxes; a record carries at most 1000"),
            "{}",
            e.body.message
        );
        many[0].boxes.truncate(MAX_BOXES);
        assert!(check(&f, true, &many).is_ok());
    }
}
