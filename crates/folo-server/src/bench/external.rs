//! External entries: predictions made anywhere else, checked strictly and turned into the passes an entry is
//! graded on. Every error names the first record that is wrong, in plain words.

use std::collections::HashMap;

use axum::http::StatusCode;
use folo_api::{BenchPlan, BenchPrediction};
use folo_bench::score::Sample;
use folo_bench::{Condition, conditions};

use super::results::Pass;
use super::state::BenchFile;
use crate::error::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", message)
}

/// What the predictions of an external entry come to.
pub struct Checked {
    /// The unchanged images.
    pub clean: Vec<Sample>,
    /// One pass per planned (condition, severity) that was answered, in plan order.
    pub variants: Vec<Pass>,
    /// Every condition of the plan was answered.
    pub complete: bool,
}

/// How a record is shown in a message: `prediction 3 (image '17', haze 3)`.
fn record(i: usize, p: &BenchPrediction) -> String {
    match (&p.condition, p.severity) {
        (Some(c), Some(s)) => format!("prediction {} (image '{}', {c} {s})", i + 1, p.image),
        (Some(c), None) => format!("prediction {} (image '{}', {c})", i + 1, p.image),
        _ => format!("prediction {} (image '{}', unchanged)", i + 1, p.image),
    }
}

/// Checks `predictions` against the benchmark `file` (its images, classes and plan) and sorts them into passes.
/// `400 invalid_body` for the first record that is wrong: an image the benchmark does not have, a condition that
/// is not one of the plan's (or a severity it does not plan), a severity without a condition, a class outside the
/// benchmark's, a score outside `0..=1`, a record repeating another; then for an image without its unchanged
/// record, and for a condition answered for some images or severities but not all.
pub fn check(file: &BenchFile, scored: bool, predictions: &[BenchPrediction]) -> Result<Checked> {
    let plan: &BenchPlan = &file.plan;
    let index: HashMap<&str, usize> =
        file.images.iter().enumerate().map(|(i, im)| (im.id.as_str(), i)).collect();
    let variants = folo_bench::variants(plan);
    let classes = file.classes.len();
    // Per image: the record index of its unchanged answer and of each planned variant's.
    let n = file.images.len();
    let mut clean: Vec<Option<usize>> = vec![None; n];
    let mut answered: Vec<Vec<Option<usize>>> = vec![vec![None; n]; variants.len()];
    for (i, p) in predictions.iter().enumerate() {
        let what = record(i, p);
        let Some(&image) = index.get(p.image.as_str()) else {
            return Err(invalid(format!("{what}: the benchmark has no image '{}'", p.image))
                .hint("GET /api/bench/benchmarks/{id}/images lists its image ids"));
        };
        let slot = match (&p.condition, p.severity) {
            (None, None) => &mut clean[image],
            (None, Some(_)) => {
                return Err(invalid(format!("{what}: a severity needs its condition")));
            }
            (Some(c), None) => {
                return Err(invalid(format!("{what}: condition '{c}' needs its severity")));
            }
            (Some(c), Some(s)) => {
                let Some(cond) = Condition::parse(c) else {
                    return Err(invalid(format!("{what}: '{c}' is not a condition of the catalog"))
                        .hint("GET /api/bench/catalog lists the conditions"));
                };
                let Some(v) = variants.iter().position(|&(vc, vs)| vc == cond && vs == s) else {
                    let planned: Vec<String> = plan
                        .conditions
                        .iter()
                        .map(|pc| format!("{} {:?}", pc.condition, pc.severities))
                        .collect();
                    return Err(invalid(format!(
                        "{what}: the benchmark's plan does not run {c} at severity {s} (it runs {})",
                        if planned.is_empty() {
                            "the unchanged images only".to_string()
                        } else {
                            planned.join(", ")
                        }
                    )));
                };
                &mut answered[v][image]
            }
        };
        if let Some(first) = *slot {
            return Err(invalid(format!(
                "{what} repeats prediction {}: answer each image and condition once",
                first + 1
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
            if scored && !(0.0..=1.0).contains(&d.score) {
                return Err(invalid(format!("{what}, box {}: score {} is outside 0..=1", b + 1, d.score)));
            }
        }
        *slot = Some(i);
    }
    if let Some(missing) = clean.iter().position(Option::is_none) {
        return Err(invalid(format!(
            "no prediction for image '{}' (unchanged): every image of the benchmark must be answered for, an \
             image with nothing found with an empty boxes list",
            file.images[missing].id
        )));
    }
    // A condition counts when every image is answered at every one of its severities; one that is partly
    // answered is an error, so a robustness number is never made of a part of a condition.
    let mut complete = true;
    let mut passes = Vec::new();
    for pc in &plan.conditions {
        let Some(cond) = Condition::parse(&pc.condition) else { continue };
        let of: Vec<usize> =
            variants.iter().enumerate().filter(|(_, v)| v.0 == cond).map(|(i, _)| i).collect();
        let any = of.iter().any(|&v| answered[v].iter().any(Option::is_some));
        if !any {
            complete = false;
            continue;
        }
        for &v in &of {
            if let Some(missing) = answered[v].iter().position(Option::is_none) {
                return Err(invalid(format!(
                    "condition '{}' is answered, but not for image '{}' at severity {}: answer a condition for \
                     every image at every planned severity ({:?}), or leave it out",
                    pc.condition, file.images[missing].id, variants[v].1, pc.severities
                )));
            }
        }
        for &v in &of {
            let (c, s) = variants[v];
            let samples = answered[v]
                .iter()
                .zip(&file.images)
                .map(|(r, im)| {
                    let p = &predictions[r.expect("checked above")];
                    // The truth under the condition, as the image was served.
                    let truths = conditions::moved_boxes(c, s, im.width, im.height, &im.boxes)
                        .expect("planned severities are valid");
                    Sample { truths, detections: p.boxes.clone() }
                })
                .collect();
            passes.push(Pass { variant: Some((c, s)), samples });
        }
    }
    let clean = clean
        .iter()
        .zip(&file.images)
        .map(|(r, im)| Sample {
            truths: im.boxes.clone(),
            detections: predictions[r.expect("checked above")].boxes.clone(),
        })
        .collect();
    Ok(Checked { clean, variants: passes, complete })
}

#[cfg(test)]
mod tests {
    use folo_api::{BBox, BenchDetection, BenchPlanCondition, BenchSource};

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

    #[test]
    fn complete_partial_and_moved_truth() {
        let f = file();
        let mut all = vec![p("2", None), p("1", None)];
        for im in ["1", "2"] {
            all.extend([p(im, Some(("haze", 1))), p(im, Some(("haze", 3))), p(im, Some(("distance", 2)))]);
        }
        let c = check(&f, true, &all).unwrap();
        assert!(c.complete);
        assert_eq!(c.variants.len(), 3);
        // In the benchmark's order, whatever the order of the records.
        assert_eq!(c.clean.len(), 2);
        // Distance moves the truth; haze does not.
        assert_eq!(c.variants[0].samples[0].truths, f.images[0].boxes);
        assert_ne!(c.variants[2].samples[0].truths, f.images[0].boxes);
        let only_clean = check(&f, true, &all[..2]).unwrap();
        assert!(!only_clean.complete && only_clean.variants.is_empty());
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
    }
}
