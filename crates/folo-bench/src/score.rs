//! The score of an entry (docs/BENCH.md, *The score* and *Definitions*, which this module implements to the
//! letter).
//!
//! Input is plain data: per image a [`Sample`] with its true boxes (in that variant's coordinates) and the
//! contestant's detections, plus the class count and whether the contestant gives confidences (`scored`).
//!
//! - [`evaluate`] grades the unchanged images: [`Evaluation`] holds accuracy (with the bootstrap interval),
//!   calibration, localization and self-labeling, as the [`folo_api`] wire types.
//! - [`map`] is the mAP pair of one variant's images; [`robustness`] turns those into the robustness axis.
//! - [`score`] makes the total of the parts.
//! - [`outcome`] is one image's line of the failure gallery, [`labels`] what a self-improvement loop keeps.
//!
//! **Bad input** never panics; it is dropped before anything is computed: a true box or a detection with a
//! non-finite coordinate, no area (`x2 <= x1` or `y2 <= y1`, which includes inverted boxes) or a class outside
//! `0..num_classes`, and a detection with a non-finite score. Finite scores are clamped to `[0, 1]`. For an
//! unscored contestant every score is 1, whatever was sent.
//!
//! **Matching** ("right" in the definitions): per image and class, detections in descending confidence (ties in
//! input order) each take the still unmatched true box of their class with the highest IoU, if that IoU is at least
//! 0.5 (ties: the later box), exactly as AP's matching at IoU 0.5 in [`folo_vision::metrics`] does. Unlike AP,
//! which keeps the best 100 detections per image and class (COCO's `maxDets`), the other axes count every
//! detection.
//!
//! **Determinism.** Every number is a function of the input alone: images are evaluated in their order (in
//! parallel by fixed chunks that are merged in order), and bootstrap resample `r` draws from a generator seeded
//! by the seed and `r` only.

use folo_api::{
    BBox, BenchAccuracy, BenchAxis, BenchAxisScore, BenchCalibration, BenchCalibrationBin,
    BenchClassAccuracy, BenchConditionResult, BenchDetection, BenchLabelPoint, BenchLocalization, BenchPlan,
    BenchReadiness, BenchRobustness, BenchScore, BenchSelfLabeling, BenchSeverityPoint,
};
use folo_core::rng::Rng;
use folo_vision::cocoeval::{CocoEval, CocoTruth};
use folo_vision::{Detection, EvalReport, Evaluator, Truth, Xyxy};
use rayon::prelude::*;

use crate::conditions::Condition;
use crate::pixels::mix;

/// The precision a contestant's labels must reach on the self-labeling axis.
pub const TARGET_PRECISION: f32 = 0.9;
/// The fewest labels a self-labeling threshold must keep: a precision measured on fewer is no support for using
/// the threshold (a weak model can be right on its single most confident detection).
pub const MIN_LABELS: u64 = 10;
/// Self-labeling recall from which the verdict is [`BenchReadiness::Ready`].
pub const READY_RECALL: f32 = 0.5;
/// Self-labeling recall from which the verdict is [`BenchReadiness::Partial`].
pub const PARTIAL_RECALL: f32 = 0.25;
/// Detections below this confidence do not enter calibration.
pub const CALIBRATION_FLOOR: f32 = 0.1;
/// The IoU from which a detection matches a true box.
pub const MATCH_IOU: f64 = 0.5;
/// Resamples of the bootstrap interval.
pub const BOOTSTRAP_RESAMPLES: usize = 200;

/// Points of the self-labeling grid `0.00, 0.01, ..., 1.00`.
const GRID: usize = 101;
/// Calibration bins `[0, 0.1)` to `[0.9, 1]`, and localization bins `[0.5, 0.55)` to `[0.95, 1]`.
const BINS: usize = 10;
/// Images per evaluator when the images are evaluated in parallel (fixed, so the result is thread-count free).
const CHUNK: usize = 64;
/// The stream of the benchmark's seed the bootstrap draws from.
const BOOTSTRAP_STREAM: u64 = 0xB007_5742;

/// One image of a run: its true boxes and what the contestant found, in the same pixel coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    /// The true boxes (under a condition: as [`crate::conditions::apply`] returns them).
    pub truths: Vec<BBox>,
    /// The contestant's detections.
    pub detections: Vec<BenchDetection>,
}

/// The mAP pair of a set of images.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Map {
    /// mAP@0.5.
    pub map50: f32,
    /// mAP@0.5:0.95.
    pub map50_95: f32,
}

/// Everything measured on the unchanged images.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// mAP and the rest, with the bootstrap interval.
    pub accuracy: BenchAccuracy,
    /// `None` for an unscored contestant and when no detection reaches [`CALIBRATION_FLOOR`].
    pub calibration: Option<BenchCalibration>,
    /// Box quality of the right detections.
    pub localization: BenchLocalization,
    /// What its own labels would be worth.
    pub self_labeling: BenchSelfLabeling,
}

/// One image of the failure gallery ([`folo_api::BenchEntryImage`] without the image).
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// The detections at or above the confidence, highest first (cleaned: see the module docs).
    pub detections: Vec<BenchDetection>,
    /// True boxes none of them matches.
    pub missed: u32,
    /// Detections that match no true box.
    pub wrong: u32,
}

// ---------------------------------------------------------------------------------------------------------------
// Cleaning and matching
// ---------------------------------------------------------------------------------------------------------------

/// A box with finite coordinates and a positive area.
fn valid_box(x1: f32, y1: f32, x2: f32, y2: f32) -> bool {
    x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite() && x2 > x1 && y2 > y1
}

/// The usable detections of a list, scores clamped (all 1 when unscored), in input order.
fn clean_detections(dets: &[BenchDetection], num_classes: usize, scored: bool) -> Vec<BenchDetection> {
    dets.iter()
        .filter(|d| valid_box(d.x1, d.y1, d.x2, d.y2) && (d.class as usize) < num_classes)
        .filter(|d| !scored || d.score.is_finite())
        .map(|d| BenchDetection { score: if scored { d.score.clamp(0.0, 1.0) } else { 1.0 }, ..*d })
        .collect()
}

fn to_detection(d: &BenchDetection) -> Detection {
    Detection { bbox: Xyxy::new(d.x1, d.y1, d.x2, d.y2), score: d.score, class: d.class }
}

/// An image ready for evaluation: cleaned truths and detections (input order) as folo-vision takes them.
struct Prepared {
    truths: Vec<Truth>,
    dets: Vec<Detection>,
}

/// The usable true boxes of a list, in input order.
fn clean_truths(truths: &[BBox], num_classes: usize) -> Vec<Truth> {
    truths
        .iter()
        .filter(|b| valid_box(b.x1, b.y1, b.x2, b.y2) && (b.class as usize) < num_classes)
        .map(|b| Truth { bbox: Xyxy::new(b.x1, b.y1, b.x2, b.y2), class: b.class })
        .collect()
}

fn prepare(s: &Sample, num_classes: usize, scored: bool) -> Prepared {
    let dets = clean_detections(&s.detections, num_classes, scored).iter().map(to_detection).collect();
    Prepared { truths: clean_truths(&s.truths, num_classes), dets }
}

fn prepare_all(samples: &[Sample], num_classes: usize, scored: bool) -> Vec<Prepared> {
    samples.par_iter().map(|s| prepare(s, num_classes, scored)).collect()
}

/// IoU in f64, as AP's matching computes it.
fn iou(a: &Xyxy, b: &Xyxy) -> f64 {
    let w = (a.x2 as f64).min(b.x2 as f64) - (a.x1 as f64).max(b.x1 as f64);
    let h = (a.y2 as f64).min(b.y2 as f64) - (a.y1 as f64).max(b.y1 as f64);
    if w <= 0.0 || h <= 0.0 {
        return 0.0;
    }
    let inter = w * h;
    let area = |b: &Xyxy| (b.x2 as f64 - b.x1 as f64) * (b.y2 as f64 - b.y1 as f64);
    inter / (area(a) + area(b) - inter)
}

/// The matching of one image: for each detection (input order) the IoU of the true box it matches, if any; and
/// how many true boxes were matched.
fn match_image(dets: &[Detection], truths: &[Truth]) -> (Vec<Option<f64>>, usize) {
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&a, &b| dets[a].class.cmp(&dets[b].class).then(dets[b].score.total_cmp(&dets[a].score)));
    let mut taken = vec![false; truths.len()];
    let mut out = vec![None; dets.len()];
    let mut matched = 0;
    for &d in &order {
        let (mut best, mut m) = (MATCH_IOU, None);
        for (g, t) in truths.iter().enumerate() {
            if taken[g] || t.class != dets[d].class {
                continue;
            }
            let v = iou(&dets[d].bbox, &t.bbox);
            if v >= best {
                best = v;
                m = Some(g);
            }
        }
        if let Some(g) = m {
            taken[g] = true;
            matched += 1;
            out[d] = Some(best);
        }
    }
    (out, matched)
}

/// A detection of the unchanged images after matching.
#[derive(Clone, Copy)]
struct Rec {
    score: f32,
    /// IoU with its true box when it is right.
    iou: Option<f64>,
}

/// The AP report of the images, in their order (parallel by fixed chunks, merged in order).
fn report<'a>(images: impl IndexedParallelIterator<Item = &'a Prepared>, num_classes: usize) -> EvalReport {
    let parts: Vec<Evaluator> = images
        .chunks(CHUNK)
        .map(|chunk| {
            let mut ev = Evaluator::new(num_classes);
            for p in chunk {
                ev.add_image(&p.dets, &p.truths);
            }
            ev
        })
        .collect();
    let mut ev = Evaluator::new(num_classes);
    for p in parts {
        ev.merge(p);
    }
    ev.compute()
}

// ---------------------------------------------------------------------------------------------------------------
// The unchanged images
// ---------------------------------------------------------------------------------------------------------------

/// Grades the unchanged images: accuracy (`seed` drives the bootstrap, use the benchmark's), calibration,
/// localization and self-labeling. Never fails: no images, no truths or no detections give zeros (and `None`
/// where docs/BENCH.md says a number is not measured).
pub fn evaluate(samples: &[Sample], num_classes: usize, scored: bool, seed: u64) -> Evaluation {
    let images = prepare_all(samples, num_classes, scored);
    let rep = report(images.par_iter(), num_classes);
    let ci95 = bootstrap(&images, num_classes, seed, rep.map50_95);

    let mut coco = CocoEval::new(num_classes);
    for (i, p) in images.iter().enumerate() {
        let truths: Vec<CocoTruth> = p.truths.iter().map(|t| CocoTruth::from_box(t.bbox, t.class)).collect();
        coco.add_image(i as u64, &p.dets, &truths);
    }
    let coco = coco.compute();
    // pycocotools' -1 (nothing qualifies) is `None`.
    let measured = |v: f64| (v >= 0.0).then_some(v as f32);

    let accuracy = BenchAccuracy {
        map50_95: rep.map50_95,
        ci95,
        map50: rep.map50,
        map75: measured(coco.ap75).unwrap_or(0.0),
        precision: rep.precision,
        recall: rep.recall,
        f1: rep.f1,
        best_conf: rep.best_conf,
        map_small: measured(coco.ap_small),
        map_medium: measured(coco.ap_medium),
        map_large: measured(coco.ap_large),
        per_class: rep
            .per_class
            .iter()
            .map(|c| BenchClassAccuracy {
                class: c.class,
                truths: c.truths,
                map50: c.map50,
                map50_95: c.map50_95,
            })
            .collect(),
    };

    let matches: Vec<Vec<Option<f64>>> =
        images.par_iter().map(|p| match_image(&p.dets, &p.truths).0).collect();
    let recs: Vec<Rec> = images
        .iter()
        .zip(&matches)
        .flat_map(|(p, m)| p.dets.iter().zip(m).map(|(d, &iou)| Rec { score: d.score, iou }))
        .collect();
    let truths: u64 = images.iter().map(|p| p.truths.len() as u64).sum();

    Evaluation {
        calibration: if scored { calibration(&recs) } else { None },
        // Without confidences, every right detection counts.
        localization: localization(&recs, if scored { accuracy.best_conf } else { 0.0 }),
        self_labeling: self_labeling(&recs, truths, images.len() as u64, scored),
        accuracy,
    }
}

/// mAP@0.5:0.95's 95% interval around `measured`: [`BOOTSTRAP_RESAMPLES`] resamples of the images with
/// replacement, their 2.5th and 97.5th percentile (linear interpolation between order statistics, numpy's
/// default), shifted by the difference between `measured` and the resamples' mean. The shift matters when classes
/// are rare: a resample that happens to hold none of a hard class leaves it out of the class mean, so resamples
/// run higher than the measured value and their plain percentiles would sit beside it instead of around it. The
/// interval is clamped to `[0, 1]` and always holds `measured`. `[0, 0]` without images.
fn bootstrap(images: &[Prepared], num_classes: usize, seed: u64, measured: f32) -> [f32; 2] {
    let n = images.len();
    if n == 0 {
        return [0.0; 2];
    }
    let mut maps: Vec<f32> = (0..BOOTSTRAP_RESAMPLES)
        .into_par_iter()
        .map(|r| {
            let mut rng = Rng::seed(mix(seed ^ BOOTSTRAP_STREAM, r as u64));
            let mut ev = Evaluator::new(num_classes);
            for _ in 0..n {
                let p = &images[rng.below(n as u64) as usize];
                ev.add_image(&p.dets, &p.truths);
            }
            ev.compute().map50_95
        })
        .collect();
    maps.sort_by(f32::total_cmp);
    // Summed in sorted order, in f64: the same for every thread count.
    let mean = maps.iter().map(|&m| m as f64).sum::<f64>() / maps.len() as f64;
    let shift = measured as f64 - mean;
    let at = |q: f64| (percentile(&maps, q) as f64 + shift).clamp(0.0, 1.0) as f32;
    [at(0.025).min(measured), at(0.975).max(measured)]
}

/// The `q` quantile of sorted values, interpolating linearly between neighbours (numpy's `linear`).
fn percentile(sorted: &[f32], q: f64) -> f32 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, f) = (pos.floor() as usize, pos - pos.floor());
    let hi = (lo + 1).min(sorted.len() - 1);
    (sorted[lo] as f64 + (sorted[hi] as f64 - sorted[lo] as f64) * f) as f32
}

/// The bin of a confidence: `k` for `k / 10 <= score < (k + 1) / 10` (edges as f32), 9 up to 1.
fn conf_bin(score: f32) -> usize {
    (1..BINS).filter(|&k| score >= k as f32 / BINS as f32).count()
}

/// ECE over the detections from [`CALIBRATION_FLOOR`] up; `None` when there are none.
fn calibration(recs: &[Rec]) -> Option<BenchCalibration> {
    let mut count = [0u64; BINS];
    let mut conf = [0.0f64; BINS];
    let mut right = [0u64; BINS];
    for r in recs.iter().filter(|r| r.score >= CALIBRATION_FLOOR) {
        let b = conf_bin(r.score);
        count[b] += 1;
        conf[b] += r.score as f64;
        right[b] += r.iou.is_some() as u64;
    }
    let n: u64 = count.iter().sum();
    if n == 0 {
        return None;
    }
    let mut ece = 0.0f64;
    let bins = (0..BINS)
        .map(|b| {
            let (c, k) = (count[b], count[b] as f64);
            let (mean, prec) = (conf[b] / k, right[b] as f64 / k);
            if c > 0 {
                ece += k / n as f64 * (mean - prec).abs();
            }
            BenchCalibrationBin {
                lo: b as f32 / BINS as f32,
                hi: (b + 1) as f32 / BINS as f32,
                count: c,
                confidence: (c > 0).then_some(mean as f32),
                precision: (c > 0).then_some(prec as f32),
            }
        })
        .collect();
    Some(BenchCalibration { ece: ece as f32, detections: n, bins })
}

/// The IoU bin of a right detection: `k` for `0.5 + 0.05 k <= iou < 0.5 + 0.05 (k + 1)`, 9 up to 1.
fn iou_bin(iou: f64) -> usize {
    (1..BINS).filter(|&k| iou >= (50 + 5 * k) as f64 / 100.0).count()
}

/// Mean IoU and histogram of the right detections with a confidence of at least `conf`.
fn localization(recs: &[Rec], conf: f32) -> BenchLocalization {
    let mut histogram = vec![0u64; BINS];
    let (mut sum, mut n) = (0.0f64, 0u64);
    for r in recs.iter().filter(|r| r.score >= conf) {
        if let Some(v) = r.iou {
            sum += v;
            n += 1;
            histogram[iou_bin(v)] += 1;
        }
    }
    BenchLocalization { mean_iou: (n > 0).then(|| (sum / n as f64) as f32), matches: n, histogram }
}

/// Confidence grid point `i` of the self-labeling curve (`i / 100` in f32, the value a score is compared with).
fn grid(i: usize) -> f32 {
    i as f32 / (GRID - 1) as f32
}

/// The highest grid index whose point is at most `score` (in `[0, 1]`).
fn grid_index(score: f32) -> usize {
    let mut i = ((score * (GRID - 1) as f32) as usize).min(GRID - 1);
    while i + 1 < GRID && grid(i + 1) <= score {
        i += 1;
    }
    while i > 0 && grid(i) > score {
        i -= 1;
    }
    i
}

/// Labels kept at one operating point.
#[derive(Clone, Copy, Default)]
struct Kept {
    labels: u64,
    right: u64,
    iou_sum: f64,
}

impl Kept {
    /// Precision at least [`TARGET_PRECISION`] (exactly, in integers: `10 · right >= 9 · labels`) on at least
    /// [`MIN_LABELS`] labels.
    fn precise(&self) -> bool {
        self.labels >= MIN_LABELS && 10 * self.right >= 9 * self.labels
    }

    fn precision(&self) -> Option<f32> {
        (self.labels > 0).then(|| (self.right as f64 / self.labels as f64) as f32)
    }
}

/// The self-labeling axis and its curve (micro-averaged over every true box of every class).
fn self_labeling(recs: &[Rec], truths: u64, images: u64, scored: bool) -> BenchSelfLabeling {
    let recall = |k: &Kept| if truths == 0 { 0.0 } else { (k.right as f64 / truths as f64) as f32 };
    let point = |confidence: f32, k: &Kept| BenchLabelPoint {
        confidence,
        precision: k.precision(),
        recall: recall(k),
        labels: k.labels,
    };
    // Kept from each grid point up (one point, everything, without confidences).
    let kept: Vec<Kept> = if scored {
        let mut at = vec![Kept::default(); GRID];
        for r in recs {
            let k = &mut at[grid_index(r.score)];
            k.labels += 1;
            if let Some(v) = r.iou {
                k.right += 1;
                k.iou_sum += v;
            }
        }
        for i in (0..GRID - 1).rev() {
            let next = at[i + 1];
            let k = &mut at[i];
            (k.labels, k.right, k.iou_sum) =
                (k.labels + next.labels, k.right + next.right, k.iou_sum + next.iou_sum);
        }
        at
    } else {
        let mut k = Kept::default();
        for r in recs {
            k.labels += 1;
            if let Some(v) = r.iou {
                k.right += 1;
                k.iou_sum += v;
            }
        }
        vec![k]
    };
    let curve: Vec<BenchLabelPoint> = if scored {
        kept.iter().enumerate().map(|(i, k)| point(grid(i), k)).collect()
    } else {
        vec![point(0.0, &kept[0])]
    };
    // The lowest threshold with the target precision; without one, the numbers describe every detection.
    let t_star = if scored { kept.iter().position(Kept::precise) } else { None };
    let reached = match t_star {
        Some(_) => true,
        None => !scored && kept[0].precise(),
    };
    let at = kept[t_star.unwrap_or(0)];
    let axis = if reached { recall(&at) } else { 0.0 };
    let readiness = if !reached || truths == 0 {
        BenchReadiness::NotReady
    } else if 2 * at.right >= truths {
        BenchReadiness::Ready
    } else if 4 * at.right >= truths {
        BenchReadiness::Partial
    } else {
        BenchReadiness::NotReady
    };
    BenchSelfLabeling {
        target_precision: TARGET_PRECISION,
        threshold: t_star.map(grid),
        precision: at.precision().unwrap_or(0.0),
        recall: axis,
        mean_iou: (at.right > 0).then(|| (at.iou_sum / at.right as f64) as f32),
        labels_per_image: if images == 0 { 0.0 } else { (at.labels as f64 / images as f64) as f32 },
        labels: at.labels,
        readiness,
        curve,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Conditions and the total
// ---------------------------------------------------------------------------------------------------------------

/// mAP@0.5 and mAP@0.5:0.95 of a set of images (a variant's), as [`evaluate`] computes them for the unchanged
/// ones. Zeros without images or truths.
pub fn map(samples: &[Sample], num_classes: usize, scored: bool) -> Map {
    let images = prepare_all(samples, num_classes, scored);
    let rep = report(images.par_iter(), num_classes);
    Map { map50: rep.map50, map50_95: rep.map50_95 }
}

/// `min(1, map / clean)`, 0 when the unchanged mAP is 0 (or anything is not a number).
fn kept_share(map: f32, clean: f32) -> f32 {
    if clean > 0.0 && map.is_finite() { (map / clean).clamp(0.0, 1.0) } else { 0.0 }
}

/// The robustness axis from the unchanged images' mAP@0.5:0.95 and the mAPs of the variants that ran
/// (`(condition, severity, map)`, any order). A condition of the plan counts only when every one of its planned
/// severities ran; runs the plan does not name are ignored (the first one counts when a pair is given twice).
/// Conditions come out in plan order, severities ascending. `None` when no condition counts. Pass a plan that
/// passed [`crate::check_plan`] (a weight that is not positive counts as 0).
pub fn robustness(
    clean_map50_95: f32,
    plan: &BenchPlan,
    runs: &[(Condition, u8, Map)],
) -> Option<BenchRobustness> {
    let mut conditions = Vec::new();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for pc in &plan.conditions {
        let Some(c) = Condition::parse(&pc.condition) else { continue };
        let points: Option<Vec<BenchSeverityPoint>> = pc
            .severities
            .iter()
            .map(|&s| {
                let (_, _, m) = runs.iter().find(|r| r.0 == c && r.1 == s)?;
                Some(BenchSeverityPoint {
                    severity: s,
                    map50_95: m.map50_95,
                    map50: m.map50,
                    retention: kept_share(m.map50_95, clean_map50_95),
                })
            })
            .collect();
        let Some(mut points) = points.filter(|p| !p.is_empty()) else { continue };
        points.sort_by_key(|p| p.severity);
        let retention = points.iter().map(|p| p.retention as f64).sum::<f64>() / points.len() as f64;
        let w = if pc.weight > 0.0 && pc.weight.is_finite() { pc.weight as f64 } else { 0.0 };
        num += w * retention;
        den += w;
        conditions.push(BenchConditionResult {
            condition: c.id().to_string(),
            weight: pc.weight,
            retention: retention as f32,
            points,
        });
    }
    if conditions.is_empty() {
        return None;
    }
    Some(BenchRobustness { retention: if den > 0.0 { (num / den) as f32 } else { 0.0 }, conditions })
}

/// The localization axis: `(mean IoU − 0.5) / 0.5`, 0 when nothing is right.
pub fn localization_axis(l: &BenchLocalization) -> f32 {
    l.mean_iou.map_or(0.0, |m| ((m - 0.5) / 0.5).clamp(0.0, 1.0))
}

/// The score: `100 · Σ weight · axis / Σ weight` over the measured axes (calibration is `None` without
/// confidences, robustness when no condition counted). `partial` when calibration is not measured or `complete`
/// is false: the caller says whether the entry ran every image and every condition of the plan, so a missing
/// robustness axis in a complete entry means the plan has no conditions, which is not partial. Axes in
/// [`BenchAxis::ALL`] order with [`BenchAxis::weight`].
pub fn score(evaluation: &Evaluation, robustness: Option<&BenchRobustness>, complete: bool) -> BenchScore {
    let value = |axis: BenchAxis| -> Option<f32> {
        let v = match axis {
            BenchAxis::Accuracy => Some(evaluation.accuracy.map50_95),
            BenchAxis::Robustness => robustness.map(|r| r.retention),
            BenchAxis::SelfLabeling => Some(evaluation.self_labeling.recall),
            BenchAxis::Calibration => evaluation.calibration.as_ref().map(|c| 1.0 - c.ece),
            BenchAxis::Localization => Some(localization_axis(&evaluation.localization)),
        };
        v.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 })
    };
    let axes: Vec<BenchAxisScore> = BenchAxis::ALL
        .iter()
        .map(|&axis| BenchAxisScore { axis, value: value(axis), weight: axis.weight() })
        .collect();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for a in &axes {
        if let Some(v) = a.value {
            num += a.weight as f64 * v as f64;
            den += a.weight as f64;
        }
    }
    BenchScore {
        total: if den > 0.0 { (100.0 * num / den) as f32 } else { 0.0 },
        partial: !complete || evaluation.calibration.is_none(),
        axes,
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Per image
// ---------------------------------------------------------------------------------------------------------------

/// One image for the failure gallery: its detections at or above `confidence` (pass the entry's
/// [`BenchAccuracy::best_conf`]), highest first, how many true boxes they miss and how many of them match nothing.
pub fn outcome(sample: &Sample, num_classes: usize, scored: bool, confidence: f32) -> Outcome {
    let truths = clean_truths(&sample.truths, num_classes);
    let mut detections: Vec<BenchDetection> = clean_detections(&sample.detections, num_classes, scored)
        .into_iter()
        .filter(|d| d.score >= confidence)
        .collect();
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let dets: Vec<Detection> = detections.iter().map(to_detection).collect();
    let (m, matched) = match_image(&dets, &truths);
    Outcome {
        missed: (truths.len() - matched) as u32,
        wrong: m.iter().filter(|v| v.is_none()).count() as u32,
        detections,
    }
}

/// The labels a contestant gives an image: its detections at or above `threshold` as boxes (cleaned: see the
/// module docs), highest confidence first.
pub fn labels(detections: &[BenchDetection], num_classes: usize, scored: bool, threshold: f32) -> Vec<BBox> {
    let mut kept: Vec<BenchDetection> = clean_detections(detections, num_classes, scored)
        .into_iter()
        .filter(|d| d.score >= threshold)
        .collect();
    kept.sort_by(|a, b| b.score.total_cmp(&a.score));
    kept.iter().map(|d| BBox { class: d.class, x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 }).collect()
}

#[cfg(test)]
mod tests;
