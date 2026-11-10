//! Hand-worked cases for every number of docs/BENCH.md's *Definitions*. The expected values are derived in the
//! comments; boxes are chosen so IoUs are exact (a detection `[0, 0, 10, h]` on the truth `[0, 0, 10, 10]` has
//! IoU `h / 10`).

use super::*;
use folo_api::BenchPlanCondition;

fn t(class: u32, x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
    BBox { class, x1, y1, x2, y2 }
}

fn d(class: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> BenchDetection {
    BenchDetection { class, score, x1, y1, x2, y2 }
}

fn close(a: f32, b: f64) -> bool {
    (a as f64 - b).abs() < 1e-6
}

/// Three images, one class, five truths, seven detections:
///
/// | det | score | image | box (IoU)                   | right?                       |
/// |-----|-------|-------|-----------------------------|------------------------------|
/// | d1  | 0.95  | 1     | T1 at IoU 0.9               | yes                          |
/// | d2  | 0.85  | 1     | T2 at IoU 0.7               | yes                          |
/// | d3  | 0.55  | 1     | T1 again (IoU 1)            | no: T1 is taken, T2 IoU 0    |
/// | d4  | 0.75  | 2     | nowhere                     | no                           |
/// | d5  | 0.65  | 2     | T3 at IoU 0.6               | yes                          |
/// | d6  | 0.35  | 3     | T4 at IoU 0.8               | yes                          |
/// | d7  | 0.05  | 3     | T5 at IoU 1                 | yes                          |
fn three_images() -> Vec<Sample> {
    vec![
        Sample {
            truths: vec![t(0, 0.0, 0.0, 10.0, 10.0), t(0, 20.0, 0.0, 30.0, 10.0)],
            detections: vec![
                d(0, 0.95, 0.0, 0.0, 10.0, 9.0),
                d(0, 0.85, 20.0, 0.0, 30.0, 7.0),
                d(0, 0.55, 0.0, 0.0, 10.0, 10.0),
            ],
        },
        Sample {
            truths: vec![t(0, 0.0, 0.0, 10.0, 10.0)],
            detections: vec![d(0, 0.75, 50.0, 50.0, 60.0, 60.0), d(0, 0.65, 0.0, 0.0, 10.0, 6.0)],
        },
        Sample {
            truths: vec![t(0, 0.0, 0.0, 10.0, 10.0), t(0, 20.0, 0.0, 30.0, 10.0)],
            detections: vec![d(0, 0.35, 0.0, 0.0, 10.0, 8.0), d(0, 0.05, 20.0, 0.0, 30.0, 10.0)],
        },
    ]
}

#[test]
fn self_labeling_needs_support_and_its_curve() {
    // In confidence order: .95 R, .85 R, .75 W, .65 R, .55 W, .35 R, .05 R. Precision of what is kept from t up:
    // t <= .05: 5/7; <= .35: 4/6; <= .55: 3/5; <= .65: 3/4; <= .75: 2/3; <= .85: 2/2 = 1.
    // From 0.76 up the precision is 1, but on two labels: fewer than MIN_LABELS (10), so no threshold counts.
    // Axis 0, `not_ready`; the other numbers describe every detection: precision 5/7, seven labels over three
    // images, the right ones' IoUs .9, .7, .6, .8, 1 → mean .8.
    let e = evaluate(&three_images(), 1, true, 0);
    let s = &e.self_labeling;
    assert_eq!((s.threshold, s.recall, s.readiness, s.labels), (None, 0.0, BenchReadiness::NotReady, 7));
    assert!(close(s.precision, 5.0 / 7.0), "{s:?}");
    assert!(close(s.mean_iou.unwrap(), 0.8));
    assert!(close(s.labels_per_image, 7.0 / 3.0));
    assert_eq!(s.curve.len(), 101);
    let at = |c: usize| s.curve[c];
    assert_eq!((at(0).labels, at(5).labels, at(6).labels, at(75).labels, at(76).labels), (7, 7, 6, 3, 2));
    assert!(close(at(0).precision.unwrap(), 5.0 / 7.0) && close(at(0).recall, 1.0));
    assert!(close(at(36).precision.unwrap(), 0.6) && close(at(36).recall, 0.6));
    assert_eq!((at(100).labels, at(100).precision, at(100).recall), (0, None, 0.0));
    assert_eq!(at(76).confidence, 0.76);
}

#[test]
fn calibration_ece() {
    // From 0.1 up: six detections, one per bin. |confidence − right| per bin:
    // [0.9, 1] .95 R → .05; [0.8, .9) .85 R → .15; [.7, .8) .75 W → .75; [.6, .7) .65 R → .35;
    // [.5, .6) .55 W → .55; [.3, .4) .35 R → .65. ECE = 2.5 / 6; d7 (0.05) is below the floor.
    let c = evaluate(&three_images(), 1, true, 0).calibration.unwrap();
    assert!(close(c.ece, 2.5 / 6.0), "{}", c.ece);
    assert_eq!(c.detections, 6);
    let counts: Vec<u64> = c.bins.iter().map(|b| b.count).collect();
    assert_eq!(counts, [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]);
    assert_eq!((c.bins[0].lo, c.bins[0].hi, c.bins[9].lo, c.bins[9].hi), (0.0, 0.1, 0.9, 1.0));
    assert_eq!((c.bins[0].confidence, c.bins[0].precision), (None, None));
    assert!(close(c.bins[7].confidence.unwrap(), 0.75) && c.bins[7].precision == Some(0.0));
}

#[test]
fn localization_at_the_best_f1_confidence() {
    // F1 of what is kept from c up: c <= .05: P 5/7, R 1 → 10/12 = .833; c <= .35: 4/6, .8 → .727; then .6, .667,
    // .5, .571, .333. The best is at c <= 0.05, the highest such grid point 0.05: every right detection counts.
    // IoUs .9, .7, .6, .8, 1 → mean .8 → axis (.8 − .5) / .5 = .6.
    let e = evaluate(&three_images(), 1, true, 0);
    assert!(close(e.accuracy.best_conf, 0.05), "{}", e.accuracy.best_conf);
    assert!(close(e.accuracy.f1, 10.0 / 12.0));
    let l = &e.localization;
    assert!(close(l.mean_iou.unwrap(), 0.8));
    assert_eq!(l.matches, 5);
    // .6 → [.6, .65), .7 → [.7, .75), .8 → [.8, .85), .9 → [.9, .95), 1 → [.95, 1].
    assert_eq!(l.histogram, [0, 0, 1, 0, 1, 0, 1, 0, 1, 1]);
    assert!(close(localization_axis(l), 0.6));
}

#[test]
fn accuracy_map_and_sizes() {
    // One truth of 10 x 10 found at IoU 0.77: a TP at 0.50 ... 0.75 (6 of 10 thresholds), so AP50 = AP75 = 1 and
    // AP50-95 = 0.6. One small object, nothing medium or large: those are `None` (pycocotools' -1).
    let s = vec![Sample {
        truths: vec![t(0, 0.0, 0.0, 10.0, 10.0)],
        detections: vec![d(0, 0.9, 0.0, 0.0, 10.0, 7.7)],
    }];
    let a = evaluate(&s, 2, true, 0).accuracy;
    assert!(close(a.map50, 1.0) && close(a.map75, 1.0) && close(a.map50_95, 0.6), "{a:?}");
    assert!(close(a.map_small.unwrap(), 0.6));
    assert_eq!((a.map_medium, a.map_large), (None, None));
    assert_eq!(a.per_class.len(), 1);
    assert_eq!((a.per_class[0].class, a.per_class[0].truths), (0, 1));
    // Every resample is this one image again: the interval is the point.
    assert!(close(a.ci95[0], 0.6) && close(a.ci95[1], 0.6));
    assert_eq!(map(&s, 2, true), Map { map50: a.map50, map50_95: a.map50_95 });
}

#[test]
fn bootstrap_interval_is_seeded_and_brackets_the_estimate() {
    // Twenty images, the first ten found at IoUs from 1 down to 0.55, the others missed: resamples vary in their
    // share of each, and in which ones they draw.
    let s: Vec<Sample> = (0..20)
        .map(|i| Sample {
            truths: vec![t(0, 0.0, 0.0, 10.0, 10.0)],
            detections: if i < 10 {
                vec![d(0, 0.5 + i as f32 * 0.01, 0.0, 0.0, 10.0, 10.0 - i as f32 * 0.5)]
            } else {
                vec![]
            },
        })
        .collect();
    let a = evaluate(&s, 1, true, 3).accuracy;
    let [lo, hi] = a.ci95;
    assert!(lo < a.map50_95 && a.map50_95 < hi, "{lo} {} {hi}", a.map50_95);
    assert!(lo > 0.1 && hi < 0.9);
    assert_eq!(evaluate(&s, 1, true, 3).accuracy.ci95, a.ci95);
    assert_ne!(evaluate(&s, 1, true, 4).accuracy.ci95, a.ci95);
    // A rare, hard class: one missed box of class 1 among twenty images. Resamples without that image score
    // class 0 alone, so they run higher than the measured class mean; the interval still sits around it.
    let mut rare = s.clone();
    rare[19].truths.push(t(1, 20.0, 20.0, 30.0, 30.0));
    let a = evaluate(&rare, 2, true, 3).accuracy;
    let [lo, hi] = a.ci95;
    assert!(
        lo <= a.map50_95 && a.map50_95 <= hi && hi - a.map50_95 < 2.0 * (a.map50_95 - lo) + 0.05,
        "{lo} {} {hi}",
        a.map50_95
    );
    assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
    // numpy's linear percentile: 200 values 0..199, q 0.025 → position 4.975.
    let v: Vec<f32> = (0..200).map(|i| i as f32).collect();
    assert!((percentile(&v, 0.025) - 4.975).abs() < 1e-4 && (percentile(&v, 0.975) - 194.025).abs() < 1e-3);
}

#[test]
fn target_precision_exactly_reached() {
    // Ten detections of score 0.8, nine right and one wrong: precision exactly 0.9 from t = 0 up to 0.8, so
    // t* = 0.00 and recall 9/10: `ready`.
    let mut dets: Vec<BenchDetection> =
        (0..9).map(|i| d(0, 0.8, i as f32 * 20.0, 0.0, i as f32 * 20.0 + 10.0, 10.0)).collect();
    dets.push(d(0, 0.8, 500.0, 500.0, 510.0, 510.0));
    let truths = (0..10).map(|i| t(0, i as f32 * 20.0, 0.0, i as f32 * 20.0 + 10.0, 10.0)).collect();
    let s = evaluate(&[Sample { truths, detections: dets }], 1, true, 0).self_labeling;
    assert_eq!((s.threshold, s.labels), (Some(0.0), 10));
    assert!(close(s.precision, 0.9) && close(s.recall, 0.9));
    assert_eq!(s.readiness, BenchReadiness::Ready);
    assert_eq!(s.curve[81].labels, 0);
}

/// `n` right detections (score 0.8) on `n` truths, in one image.
fn all_right(n: usize) -> Vec<Sample> {
    let truths = (0..n).map(|i| t(0, i as f32 * 20.0, 0.0, i as f32 * 20.0 + 10.0, 10.0)).collect();
    let detections = (0..n).map(|i| d(0, 0.8, i as f32 * 20.0, 0.0, i as f32 * 20.0 + 10.0, 10.0)).collect();
    vec![Sample { truths, detections }]
}

#[test]
fn a_threshold_needs_ten_labels_scored_or_not() {
    // Nine right labels: precision 1 everywhere they are kept, but nine is fewer than MIN_LABELS. No threshold,
    // axis 0, `not_ready`, and the numbers describe all nine.
    for scored in [true, false] {
        let s = evaluate(&all_right(9), 1, scored, 0).self_labeling;
        assert_eq!((s.threshold, s.recall, s.readiness, s.labels), (None, 0.0, BenchReadiness::NotReady, 9));
        assert!(close(s.precision, 1.0) && close(s.labels_per_image, 9.0));
    }
    // Ten: the lowest grid point keeping all ten (scored: 0.00; unscored: its one operating point, no threshold)
    // reaches the target; recall 10/10, `ready`.
    let s = evaluate(&all_right(10), 1, true, 0).self_labeling;
    assert_eq!((s.threshold, s.recall, s.readiness, s.labels), (Some(0.0), 1.0, BenchReadiness::Ready, 10));
    // From 0.81 up nothing is kept; from 0.01 to 0.80 all ten: the lowest is 0.00.
    assert_eq!((s.curve[80].labels, s.curve[81].labels), (10, 0));
    let s = evaluate(&all_right(10), 1, false, 0).self_labeling;
    assert_eq!((s.threshold, s.recall, s.readiness, s.labels), (None, 1.0, BenchReadiness::Ready, 10));
}

#[test]
fn target_precision_never_reached() {
    // One right and one wrong at every confidence: precision 0.5 everywhere. No threshold, axis 0, and the other
    // numbers describe every detection.
    let s = vec![Sample {
        truths: vec![t(0, 0.0, 0.0, 10.0, 10.0), t(0, 20.0, 0.0, 30.0, 10.0)],
        detections: vec![d(0, 0.9, 0.0, 0.0, 10.0, 10.0), d(0, 0.9, 50.0, 0.0, 60.0, 10.0)],
    }];
    let e = evaluate(&s, 1, true, 0);
    let sl = &e.self_labeling;
    assert_eq!((sl.threshold, sl.recall, sl.readiness, sl.labels), (None, 0.0, BenchReadiness::NotReady, 2));
    assert!(close(sl.precision, 0.5) && close(sl.labels_per_image, 2.0));
    assert!(close(sl.curve[0].recall, 0.5));
}

#[test]
fn unscored_contestants() {
    // Scores are all taken as 1. Self-labeling has one point (everything): 5/7 right is below 0.9 → axis 0.
    // Calibration is not measured. Localization counts every right detection (mean IoU 0.8).
    let e = evaluate(&three_images(), 1, false, 0);
    assert_eq!(e.calibration, None);
    let s = &e.self_labeling;
    assert_eq!(s.curve.len(), 1);
    assert_eq!((s.curve[0].confidence, s.curve[0].labels), (0.0, 7));
    assert_eq!((s.threshold, s.recall, s.readiness), (None, 0.0, BenchReadiness::NotReady));
    assert!(close(s.precision, 5.0 / 7.0));
    assert!(close(e.localization.mean_iou.unwrap(), 0.8) && e.localization.matches == 5);
    // Without the two wrong ones: 5/5 right, but five labels are fewer than MIN_LABELS: still axis 0.
    let mut clean = three_images();
    clean[0].detections.remove(2);
    clean[1].detections.remove(0);
    let s = evaluate(&clean, 1, false, 0).self_labeling;
    assert_eq!((s.threshold, s.recall, s.readiness, s.labels), (None, 0.0, BenchReadiness::NotReady, 5));
    assert!(close(s.precision, 1.0));
    // An unscored contestant's own scores are ignored, even nonsense ones.
    let mut odd = clean.clone();
    odd[0].detections[0].score = f32::NAN;
    assert_eq!(evaluate(&odd, 1, false, 0), evaluate(&clean, 1, false, 0));
}

#[test]
fn empty_input() {
    for samples in [vec![], vec![Sample::default(); 3]] {
        let e = evaluate(&samples, 4, true, 0);
        assert_eq!(e.accuracy.map50_95, 0.0);
        assert_eq!(e.accuracy.ci95, [0.0, 0.0]);
        assert_eq!((e.accuracy.map_small, e.accuracy.map_medium, e.accuracy.map_large), (None, None, None));
        assert!(e.accuracy.per_class.is_empty());
        assert_eq!(e.calibration, None);
        assert_eq!((e.localization.mean_iou, e.localization.matches), (None, 0));
        let s = &e.self_labeling;
        assert_eq!(
            (s.threshold, s.recall, s.readiness, s.labels_per_image),
            (None, 0.0, BenchReadiness::NotReady, 0.0)
        );
        assert!(s.curve.iter().all(|p| p.labels == 0 && p.precision.is_none()));
        let sc = score(&e, None, true);
        assert_eq!(sc.total, 0.0);
        assert!(sc.partial);
    }
    // Truths and no detections: every true box missed; detections and no truths: every one wrong.
    let s = Sample { truths: vec![t(0, 0.0, 0.0, 5.0, 5.0)], detections: vec![] };
    assert_eq!(outcome(&s, 1, true, 0.0), Outcome { detections: vec![], missed: 1, wrong: 0 });
    let s = Sample { truths: vec![], detections: vec![d(0, 0.5, 0.0, 0.0, 5.0, 5.0)] };
    assert_eq!(outcome(&s, 1, true, 0.0).wrong, 1);
    assert_eq!(evaluate(&[s], 1, true, 0).self_labeling.precision, 0.0);
}

#[test]
fn bad_input_is_dropped() {
    let good = three_images();
    let mut bad = good.clone();
    bad[0].truths.extend([
        t(0, f32::NAN, 0.0, 10.0, 10.0),
        t(0, 0.0, 0.0, f32::INFINITY, 10.0),
        t(0, 10.0, 10.0, 0.0, 0.0), // inverted
        t(0, 5.0, 5.0, 5.0, 9.0),   // no area
        t(7, 0.0, 0.0, 10.0, 10.0), // no such class
    ]);
    bad[1].detections.extend([
        d(0, f32::NAN, 0.0, 0.0, 10.0, 10.0),
        d(0, f32::INFINITY, 0.0, 0.0, 10.0, 10.0),
        d(0, 0.9, f32::NEG_INFINITY, 0.0, 10.0, 10.0),
        d(0, 0.9, 10.0, 0.0, 0.0, 10.0),
        d(0, 0.9, 0.0, 0.0, 0.0, 0.0),
        d(3, 0.9, 0.0, 0.0, 10.0, 10.0),
    ]);
    assert_eq!(evaluate(&bad, 1, true, 0), evaluate(&good, 1, true, 0));
    assert_eq!(map(&bad, 1, true), map(&good, 1, true));
    assert_eq!(outcome(&bad[1], 1, true, 0.0), outcome(&good[1], 1, true, 0.0));
    // Scores outside [0, 1] are clamped.
    let s = Sample {
        truths: vec![],
        detections: vec![d(0, 7.0, 0.0, 0.0, 1.0, 1.0), d(0, -2.0, 0.0, 0.0, 1.0, 1.0)],
    };
    let o = outcome(&s, 1, true, 0.0);
    assert_eq!(o.detections.iter().map(|d| d.score).collect::<Vec<_>>(), [1.0, 0.0]);
    // No classes at all: nothing is usable, nothing panics.
    let e = evaluate(&good, 0, true, 0);
    assert_eq!((e.accuracy.map50_95, e.self_labeling.curve[0].labels), (0.0, 0));
}

#[test]
fn matching_is_per_class() {
    // A detection of the wrong class on a truth is wrong, and the truth is missed.
    let s = Sample {
        truths: vec![t(0, 0.0, 0.0, 10.0, 10.0), t(1, 20.0, 0.0, 30.0, 10.0)],
        detections: vec![d(1, 0.9, 0.0, 0.0, 10.0, 10.0), d(1, 0.8, 20.0, 0.0, 30.0, 10.0)],
    };
    let o = outcome(&s, 2, true, 0.0);
    assert_eq!((o.missed, o.wrong), (1, 1));
    // An IoU of exactly 0.5 matches ([0, 0, 10, 5] on [0, 0, 10, 10]); 0.4 does not.
    let at = |h: f32| {
        outcome(
            &Sample {
                truths: vec![t(0, 0.0, 0.0, 10.0, 10.0)],
                detections: vec![d(0, 1.0, 0.0, 0.0, 10.0, h)],
            },
            1,
            true,
            0.0,
        )
    };
    assert_eq!((at(5.0).wrong, at(4.0).wrong), (0, 1));
}

#[test]
fn outcome_and_labels() {
    let s = &three_images()[0];
    // From 0.5 up: d1, d2 and the duplicate d3, which is wrong; both truths found.
    let o = outcome(s, 1, true, 0.5);
    assert_eq!(o.detections.iter().map(|d| d.score).collect::<Vec<_>>(), [0.95, 0.85, 0.55]);
    assert_eq!((o.missed, o.wrong), (0, 1));
    // From 0.9 up: d1 only; T2 missed.
    let o = outcome(s, 1, true, 0.9);
    assert_eq!((o.detections.len(), o.missed, o.wrong), (1, 1, 0));
    // Unscored: every score is 1, so any confidence keeps everything, in input order.
    let o = outcome(s, 1, false, 1.0);
    assert_eq!(o.detections.len(), 3);
    assert!(o.detections.iter().all(|d| d.score == 1.0));
    let l = labels(&s.detections, 1, true, 0.8);
    assert_eq!(l, vec![t(0, 0.0, 0.0, 10.0, 9.0), t(0, 20.0, 0.0, 30.0, 7.0)]);
    assert_eq!(labels(&s.detections, 1, false, 0.99).len(), 3);
}

fn plan(conditions: &[(&str, &[u8], f32)]) -> BenchPlan {
    BenchPlan {
        conditions: conditions
            .iter()
            .map(|&(c, s, w)| BenchPlanCondition {
                condition: c.into(),
                severities: s.to_vec(),
                weight: w,
                why: None,
            })
            .collect(),
        rationale: None,
        designed_by: None,
    }
}

fn m(map50_95: f32) -> Map {
    Map { map50: map50_95 + 0.1, map50_95 }
}

#[test]
fn robustness_retention() {
    // Unchanged mAP 0.5. haze (weight 2): 0.45 → 0.9, 0.30 → 0.6, mean 0.75. jpeg (weight 1): 0.6 → 1 (capped),
    // 0.25 → 0.5, 0.1 → 0.2, mean 1.7/3. distance did not run: left out. defocus is not in the plan: ignored.
    // Axis = (2 · 0.75 + 1.7/3) / 3 = 0.688889.
    let p = plan(&[("haze", &[1, 3], 2.0), ("jpeg", &[1, 3, 5], 1.0), ("distance", &[1], 1.0)]);
    let runs = [
        (Condition::Jpeg, 5, m(0.1)),
        (Condition::Haze, 3, m(0.3)),
        (Condition::Haze, 1, m(0.45)),
        (Condition::Jpeg, 1, m(0.6)),
        (Condition::Jpeg, 3, m(0.25)),
        (Condition::Defocus, 1, m(0.2)),
    ];
    let r = robustness(0.5, &p, &runs).unwrap();
    assert!(close(r.retention, (1.5 + 1.7 / 3.0) / 3.0), "{}", r.retention);
    assert_eq!(r.conditions.iter().map(|c| c.condition.as_str()).collect::<Vec<_>>(), ["haze", "jpeg"]);
    let haze = &r.conditions[0];
    assert!(close(haze.retention, 0.75) && haze.weight == 2.0);
    assert_eq!(haze.points.iter().map(|p| p.severity).collect::<Vec<_>>(), [1, 3]);
    assert!(close(haze.points[0].retention, 0.9) && close(haze.points[0].map50, 0.55));
    assert!(close(r.conditions[1].points[0].retention, 1.0));

    // A condition with one planned severity missing does not count.
    let partial = robustness(0.5, &p, &runs[..4]).unwrap();
    assert_eq!(partial.conditions.len(), 1);
    assert!(close(partial.retention, 0.75));
    // Nothing counts: no axis.
    assert_eq!(robustness(0.5, &p, &runs[5..]), None);
    assert_eq!(robustness(0.5, &plan(&[]), &runs), None);
    // Unchanged mAP 0: the axis is 0.
    let zero = robustness(0.0, &p, &runs).unwrap();
    assert_eq!(zero.retention, 0.0);
    assert!(zero.conditions.iter().all(|c| c.retention == 0.0));
}

#[test]
fn total_score() {
    // The three images: accuracy a = mAP50-95, self-labeling 0 (no threshold keeps ten labels), calibration
    // 1 − 2.5/6, localization 0.6, with robustness 0.688889: total = 100 · (.35 a + .25 · .688889 + .2 · 0 +
    // .1 · (1 − 2.5/6) + .1 · .6).
    let e = evaluate(&three_images(), 1, true, 0);
    let a = e.accuracy.map50_95 as f64;
    let rob = BenchRobustness { retention: 0.688_889, conditions: vec![] };
    let s = score(&e, Some(&rob), true);
    let want = 100.0 * (0.35 * a + 0.25 * 0.688_889 + 0.2 * 0.0 + 0.1 * (1.0 - 2.5 / 6.0) + 0.1 * 0.6);
    assert!((s.total as f64 - want).abs() < 1e-4, "{} {want}", s.total);
    assert!(!s.partial);
    assert_eq!(s.axes.iter().map(|a| a.axis).collect::<Vec<_>>(), BenchAxis::ALL);
    assert_eq!(s.axes.iter().map(|a| a.weight).collect::<Vec<_>>(), [0.35, 0.25, 0.2, 0.1, 0.1]);
    assert!(score(&e, Some(&rob), false).partial);

    // Unscored and no conditions: calibration and robustness are left out of both sums.
    let e = evaluate(&three_images(), 1, false, 0);
    let s = score(&e, None, true);
    let a = e.accuracy.map50_95 as f64;
    let want = 100.0 * (0.35 * a + 0.2 * 0.0 + 0.1 * 0.6) / 0.65;
    assert!((s.total as f64 - want).abs() < 1e-4, "{} {want}", s.total);
    assert!(s.partial);
    assert_eq!((s.axes[1].value, s.axes[3].value), (None, None));

    // A complete, scored entry of a benchmark whose plan has no conditions: no robustness axis for anyone, so
    // nothing of it is missing.
    let e = evaluate(&three_images(), 1, true, 0);
    let s = score(&e, None, true);
    assert!(!s.partial && s.axes[1].value.is_none());
    assert!(score(&e, None, false).partial);
}

#[test]
fn accuracy_by_hand() {
    // The three images' AP50 (one class, five truths), in confidence order R R W R W R R:
    // recall .2 .4 .4 .6 .6 .8 1, precision 1 1 2/3 3/4 3/5 2/3 5/7, envelope 1 1 3/4 3/4 5/7 5/7 5/7.
    // 101 recall points: 0 .. .20 → 1 (21), .21 .. .40 → 1 (20), .41 .. .60 → 3/4 (20), .61 .. 1 → 5/7 (40).
    // AP50 = (41 + 15 + 200/7) / 101 = 0.837341.
    let e = evaluate(&three_images(), 1, true, 0);
    assert!((e.accuracy.map50 as f64 - (56.0 + 200.0 / 7.0) / 101.0).abs() < 1e-6, "{}", e.accuracy.map50);
    // Precision / recall at best_conf (0.05): 5/7 and 1.
    assert!(close(e.accuracy.precision, 5.0 / 7.0) && close(e.accuracy.recall, 1.0));
}
