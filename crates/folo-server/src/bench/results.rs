//! An entry's grade from its detections: every contestant (a model, an agent, external predictions) ends here,
//! so they are all scored by the same code ([`folo_bench::score`]).

use folo_api::{BenchPlan, BenchScore};
use folo_bench::Condition;
use folo_bench::score::{self, Sample};

use super::state::{ImageOutcome, ResultParts};

/// The detections of one pass over the entry's images: the unchanged images (`None`) or one variant.
pub struct Pass {
    /// The variant, `None` for the unchanged images.
    pub variant: Option<(Condition, u8)>,
    /// One sample per image of the entry, in the benchmark's order.
    pub samples: Vec<Sample>,
}

/// What grading an entry gives.
pub struct Graded {
    pub parts: ResultParts,
    /// Per image of the unchanged pass, in order.
    pub images: Vec<ImageOutcome>,
    pub score: BenchScore,
    pub map50_95: f32,
    /// Whether the entry was graded with contestant confidences.
    pub scored: bool,
}

impl Graded {
    /// The plan's conditions that ran completely (`BenchEntry.conditions_run`): those the robustness axis counts.
    pub fn conditions_run(&self) -> u32 {
        self.parts.robustness.as_ref().map_or(0, |r| r.conditions.len() as u32)
    }
}

/// How an entry is graded.
pub struct Spec<'a> {
    /// The benchmark's class count.
    pub classes: usize,
    /// The contestant gives confidences.
    pub scored: bool,
    /// The benchmark's plan.
    pub plan: &'a BenchPlan,
    /// The benchmark's seed (the bootstrap draws from it).
    pub seed: u64,
    /// The entry ran every image of the benchmark and every condition of its plan (docs/BENCH.md, *The score*).
    pub complete: bool,
}

/// Grades an entry: `clean` are the unchanged images (`ids` their image ids, in step), `variants` the passes of
/// the plan's conditions that ran.
pub fn grade(clean: &[Sample], ids: &[String], variants: &[Pass], spec: &Spec<'_>) -> Graded {
    let runs: Vec<(Condition, u8, score::Map)> = variants
        .iter()
        .filter_map(|p| p.variant.map(|(c, s)| (c, s, score::map(&p.samples, spec.classes, spec.scored))))
        .collect();
    grade_maps(clean, ids, &runs, spec)
}

/// [`grade`] with the variants already reduced to their mAP pairs (`(condition, severity, map)`), which is all
/// the result needs of them: an external entry grades each variant as soon as it is whole.
pub fn grade_maps(
    clean: &[Sample],
    ids: &[String],
    runs: &[(Condition, u8, score::Map)],
    spec: &Spec<'_>,
) -> Graded {
    let Spec { classes, scored, plan, seed, complete } = *spec;
    let evaluation = score::evaluate(clean, classes, scored, seed);
    let robustness = score::robustness(evaluation.accuracy.map50_95, plan, runs);
    let total = score::score(&evaluation, robustness.as_ref(), complete);
    // The images at its best-F1 confidence; without one (F1 0 at every confidence, `best_conf` is then not a best F1),
    // every detection. The confidence is kept with them (`images_conf`), so readers say what they hold.
    let conf = if evaluation.accuracy.f1 > 0.0 { evaluation.accuracy.best_conf } else { 0.0 };
    let images = clean
        .iter()
        .zip(ids)
        .map(|(s, id)| {
            let o = score::outcome(s, classes, scored, conf);
            ImageOutcome { image: id.clone(), detections: o.detections, missed: o.missed, wrong: o.wrong }
        })
        .collect();
    Graded {
        map50_95: evaluation.accuracy.map50_95,
        score: total,
        scored,
        images,
        parts: ResultParts {
            accuracy: evaluation.accuracy,
            robustness,
            calibration: evaluation.calibration,
            localization: evaluation.localization,
            self_labeling: evaluation.self_labeling,
            images_conf: Some(conf),
        },
    }
}

#[cfg(test)]
mod tests {
    use folo_api::{BBox, BenchDetection};
    use folo_bench::plan::default_plan;

    use super::*;

    fn det(score: f32, x1: f32) -> BenchDetection {
        BenchDetection { class: 0, score, x1, y1: 0.0, x2: x1 + 10.0, y2: 10.0 }
    }

    fn graded(samples: &[Sample]) -> Graded {
        let ids: Vec<String> = (0..samples.len()).map(|i| i.to_string()).collect();
        let plan = default_plan();
        let spec = Spec { classes: 1, scored: true, plan: &plan, seed: 0, complete: true };
        grade_maps(samples, &ids, &[], &spec)
    }

    fn outcome(g: &Graded) -> (usize, u32, u32) {
        (g.images[0].detections.len(), g.images[0].missed, g.images[0].wrong)
    }

    #[test]
    fn images_hold_every_detection_without_a_best_f1_and_keep_where_they_were_taken() {
        // Two true boxes, three detections far from them: F1 is 0 at every confidence, so there is no best-F1
        // confidence to take the images at (`best_conf` is the grid's top, 1): they hold all three, taken at 0.
        let truths = vec![BBox { class: 0, x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0 }; 2];
        let wrong = Sample {
            truths: truths.clone(),
            detections: vec![det(0.3, 50.0), det(0.2, 70.0), det(0.1, 90.0)],
        };
        let g = graded(std::slice::from_ref(&wrong));
        assert_eq!((g.parts.accuracy.f1, g.parts.accuracy.best_conf), (0.0, 1.0));
        assert_eq!((outcome(&g), g.parts.images_conf), ((3, 2, 3), Some(0.0)));

        // 100 wrong ones above a right one: the scorer counts an image's top 100 per class, so F1 is 0 at every
        // confidence too; the images hold all 101, the right one found.
        let mut dets = vec![det(0.9, 50.0); 100];
        dets.push(det(0.3, 0.0));
        let g = graded(&[Sample { truths: truths[..1].to_vec(), detections: dets }]);
        assert_eq!(g.parts.accuracy.f1, 0.0);
        assert_eq!((outcome(&g), g.parts.images_conf), ((101, 0, 100), Some(0.0)));

        // One right detection at 0.6 and a wrong one at 0.3: the best F1 is at 0.6, and the images keep to it.
        let right = Sample { truths: truths[..1].to_vec(), detections: vec![det(0.6, 0.0), det(0.3, 50.0)] };
        let g = graded(&[right]);
        assert_eq!((outcome(&g), g.parts.images_conf), ((1, 0, 0), Some(g.parts.accuracy.best_conf)));
        assert!(g.parts.accuracy.best_conf > 0.0);
    }

    #[test]
    fn an_entry_graded_before_images_conf_reads_as_taken_at_best_conf() {
        // A file written before the field: the images were taken at `best_conf`, whatever F1 is.
        let g = graded(&[Sample { truths: Vec::new(), detections: vec![det(0.3, 0.0)] }]);
        let mut json = serde_json::to_value(&g.parts).unwrap();
        assert_eq!(json["images_conf"], serde_json::json!(0.0));
        json.as_object_mut().unwrap().remove("images_conf");
        json["accuracy"]["best_conf"] = serde_json::json!(1.0);
        let old: ResultParts = serde_json::from_value(json).unwrap();
        assert_eq!((old.images_conf, old.images_conf()), (None, 1.0));
    }
}
