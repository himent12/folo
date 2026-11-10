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
    let conf = evaluation.accuracy.best_conf;
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
        images,
        parts: ResultParts {
            accuracy: evaluation.accuracy,
            robustness,
            calibration: evaluation.calibration,
            localization: evaluation.localization,
            self_labeling: evaluation.self_labeling,
        },
    }
}
