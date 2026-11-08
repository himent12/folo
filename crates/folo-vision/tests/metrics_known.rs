//! Evaluator against hand-computed answers. See `docs/METRICS.md` for the method.

mod common;

use folo_vision::metrics::{EvalConfig, Evaluator, Truth};
use folo_vision::{Detection, Xyxy};

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn gt(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

fn close(a: f32, b: f64) -> bool {
    (a as f64 - b).abs() < 1e-6
}

/// TP, FP, TP on two ground truths: recall [.5, .5, 1], precision [1, .5, 2/3], envelope [1, 2/3, 2/3].
/// Recall points 0.00..=0.50 (51 of them) sample precision 1, 0.51..=1.00 (50) sample 2/3.
const AP_TP_FP_TP: f64 = (51.0 + 50.0 * 2.0 / 3.0) / 101.0;

#[test]
fn tp_fp_tp_single_image() {
    let mut ev = Evaluator::new(1);
    ev.add_image(
        &[det(0., 0., 10., 10., 0.9, 0), det(50., 50., 60., 60., 0.8, 0), det(20., 20., 30., 30., 0.7, 0)],
        &[gt(0., 0., 10., 10., 0), gt(20., 20., 30., 30., 0)],
    );
    let r = ev.compute();
    assert!(close(r.map50, AP_TP_FP_TP), "map50 {}", r.map50);
    assert!(close(r.map50_95, AP_TP_FP_TP), "every threshold matches (IoU 1)");
    // Best F1: at conf <= 0.7 all three count: P 2/3, R 1, F1 0.8. The highest such threshold is reported.
    assert!(close(r.best_conf, 0.7));
    assert!(close(r.precision, 2.0 / 3.0));
    assert!(close(r.recall, 1.0));
    assert!(close(r.f1, 0.8));
    assert_eq!((r.num_images, r.num_truths, r.num_preds), (1, 2, 3));
    assert_eq!(r.per_class.len(), 1);
    assert_eq!(r.per_class[0].truths, 2);
    // PR curve export: 101 points, 1 up to recall 0.5 then 2/3.
    let pr = &r.curves.per_class[0].pr;
    assert_eq!(pr.len(), 101);
    assert!(close(pr[50], 1.0) && close(pr[51], 2.0 / 3.0) && close(pr[100], 2.0 / 3.0));
    assert_eq!(r.curves.recall.len(), 101);
    assert_eq!(r.curves.confidence.len(), 101);
    // Confidence curves at 0.80 (index 80): dets with score >= 0.8 are TP, FP.
    let c = &r.curves.per_class[0];
    assert!(close(c.precision[80], 0.5) && close(c.recall[80], 0.5));
    assert!(close(c.precision[100], 0.0) && close(c.recall[100], 0.0));
}

#[test]
fn iou_thresholds_are_inclusive() {
    // IoU exactly 0.6: TP at 0.50, 0.55 and 0.60, FP above. mAP50-95 = 3/10 (AP 1 at each matched threshold).
    let mut ev = Evaluator::new(1);
    ev.add_image(&[det(0., 0., 6., 10., 0.5, 0)], &[gt(0., 0., 10., 10., 0)]);
    let r = ev.compute();
    assert!(close(r.map50, 1.0));
    assert!(close(r.map50_95, 0.3), "{}", r.map50_95);
    let ap = &r.per_class[0].ap;
    assert!(close(ap[2], 1.0) && close(ap[3], 0.0));
    assert!(close(r.map_per_iou[2], 1.0) && close(r.map_per_iou[3], 0.0));
}

#[test]
fn no_predictions() {
    let mut ev = Evaluator::new(2);
    ev.add_image(&[], &[gt(0., 0., 10., 10., 0), gt(5., 5., 10., 10., 1)]);
    let r = ev.compute();
    assert_eq!((r.map50, r.map50_95, r.precision, r.recall, r.f1), (0.0, 0.0, 0.0, 0.0, 0.0));
    assert_eq!(r.per_class.len(), 2);
    // Both truths missed: background row.
    assert_eq!(r.confusion.matrix[2], vec![1, 1, 0]);
}

#[test]
fn no_ground_truth() {
    let mut ev = Evaluator::new(2);
    ev.add_image(&[det(0., 0., 10., 10., 0.9, 1)], &[]);
    ev.add_image(&[], &[]);
    let r = ev.compute();
    // No class has ground truth: nothing to average; every summary is 0 and per_class is empty.
    assert_eq!((r.map50, r.map50_95, r.precision, r.recall), (0.0, 0.0, 0.0, 0.0));
    assert!(r.per_class.is_empty() && r.curves.per_class.is_empty());
    assert_eq!(r.num_images, 2);
    // The prediction is a false positive: background column.
    assert_eq!(r.confusion.matrix[1], vec![0, 0, 1]);
    // Empty evaluator.
    let r = Evaluator::new(3).compute();
    assert_eq!(r.map50, 0.0);
    assert_eq!(r.confusion.matrix.len(), 4);
    let r = Evaluator::new(0).compute();
    assert_eq!(r.confusion.matrix, vec![vec![0]]);
}

#[test]
fn all_false_positives() {
    let mut ev = Evaluator::new(1);
    ev.add_image(
        &[det(100., 100., 110., 110., 0.9, 0), det(200., 200., 210., 210., 0.3, 0)],
        &[gt(0., 0., 10., 10., 0)],
    );
    let r = ev.compute();
    assert_eq!((r.map50, r.map50_95, r.precision, r.recall), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn wrong_class_is_false_positive() {
    let mut ev = Evaluator::new(2);
    ev.add_image(&[det(0., 0., 10., 10., 0.9, 1)], &[gt(0., 0., 10., 10., 0)]);
    let r = ev.compute();
    assert_eq!(r.map50, 0.0);
    assert_eq!(r.per_class.iter().map(|c| c.class).collect::<Vec<_>>(), vec![0]);
    // Class-agnostic confusion matching: predicted 1, true 0.
    assert_eq!(r.confusion.matrix[1][0], 1);
}

#[test]
fn duplicate_predictions() {
    // Two exact duplicates on one gt: TP then FP. Recall reaches 1 at the first detection, so AP = 1.
    let mut ev = Evaluator::new(1);
    ev.add_image(&[det(0., 0., 10., 10., 0.9, 0), det(0., 0., 10., 10., 0.8, 0)], &[gt(0., 0., 10., 10., 0)]);
    assert!(close(ev.compute().map50, 1.0));

    // A loose higher-scored duplicate (IoU 0.52) and an exact lower-scored one. Greedy by score:
    // at IoU 0.50 the loose box takes the gt (AP 1); at 0.55..0.95 it is a FP ranked first and the exact box is the
    // TP: recall [0, 1], precision [0, .5] → envelope .5 everywhere → AP .5. mAP50-95 = (1 + 9 * .5) / 10.
    let mut ev = Evaluator::new(1);
    ev.add_image(
        &[det(0., 0., 52., 100., 0.9, 0), det(0., 0., 100., 100., 0.8, 0)],
        &[gt(0., 0., 100., 100., 0)],
    );
    let r = ev.compute();
    assert!(close(r.map50, 1.0));
    assert!(close(r.map50_95, 0.55), "{}", r.map50_95);
}

#[test]
fn greedy_match_prefers_best_unmatched_iou() {
    // One prediction overlapping two gts: it takes the higher-IoU one; the second prediction gets the other.
    let mut ev = Evaluator::new(1);
    ev.add_image(
        &[det(0., 0., 10., 10., 0.9, 0), det(4., 0., 14., 10., 0.8, 0)],
        &[gt(5., 0., 15., 10., 0), gt(0., 0., 10., 10., 0)],
    );
    // pred0 → gt1 (IoU 1), pred1 → gt0 (IoU 9/11 ≈ 0.818): two TPs at IoU <= 0.80, one at 0.85..0.95.
    let r = ev.compute();
    assert!(close(r.map50, 1.0));
    // At t >= 0.85: TP, FP → recall [.5, .5], precision [1, .5] → points 0..=.5 get 1, rest 0 → 51/101.
    let ap = &r.per_class[0].ap;
    for (t, &v) in ap.iter().enumerate() {
        let want = if t <= 6 { 1.0 } else { 51.0 / 101.0 };
        assert!(close(v, want), "t={t}: {v} vs {want}");
    }
}

#[test]
fn two_classes_and_images_average() {
    let mut ev = Evaluator::new(2);
    // Class 0: the TP/FP/TP scenario split across two images (order matters only on score ties).
    ev.add_image(
        &[det(0., 0., 10., 10., 0.9, 0), det(50., 50., 60., 60., 0.8, 0)],
        &[gt(0., 0., 10., 10., 0)],
    );
    ev.add_image(&[det(20., 20., 30., 30., 0.7, 0)], &[gt(20., 20., 30., 30., 0), gt(0., 0., 5., 5., 1)]);
    // Class 1 has one gt and no prediction: AP 0.
    let r = ev.compute();
    assert!(close(r.map50, AP_TP_FP_TP / 2.0), "{}", r.map50);
    assert_eq!(r.per_class.len(), 2);
    assert!(close(r.per_class[0].map50, AP_TP_FP_TP));
    assert_eq!(r.per_class[1].map50, 0.0);

    // Splitting the same images across two evaluators and merging gives the same report.
    let mut a = Evaluator::new(2);
    a.add_image(
        &[det(0., 0., 10., 10., 0.9, 0), det(50., 50., 60., 60., 0.8, 0)],
        &[gt(0., 0., 10., 10., 0)],
    );
    let mut b = Evaluator::new(2);
    b.add_image(&[det(20., 20., 30., 30., 0.7, 0)], &[gt(20., 20., 30., 30., 0), gt(0., 0., 5., 5., 1)]);
    a.merge(b);
    assert_eq!(a.compute(), r);
}

#[test]
fn score_ties_follow_image_order() {
    // Equal scores across images: pycocotools' stable sort keeps image order. FP (image 1) before TP (image 2):
    // recall [0, 1], precision [0, .5] → AP .5. Reversed image order: TP first → AP 1.
    let fp = (vec![det(50., 50., 60., 60., 0.5, 0)], vec![]);
    let tp = (vec![det(0., 0., 10., 10., 0.5, 0)], vec![gt(0., 0., 10., 10., 0)]);
    let mut ev = Evaluator::new(1);
    ev.add_image(&fp.0, &fp.1);
    ev.add_image(&tp.0, &tp.1);
    assert!(close(ev.compute().map50, 0.5));
    let mut ev = Evaluator::new(1);
    ev.add_image(&tp.0, &tp.1);
    ev.add_image(&fp.0, &fp.1);
    assert!(close(ev.compute().map50, 1.0));
}

#[test]
fn max_dets_per_class_cap() {
    // 100 high-scoring false positives, then one low-scoring TP. COCO's cap of 100 drops the TP.
    let mut preds: Vec<Detection> =
        (0..100).map(|i| det(100. + i as f32, 100., 105. + i as f32, 105., 0.9, 0)).collect();
    preds.push(det(0., 0., 10., 10., 0.1, 0));
    let truths = [gt(0., 0., 10., 10., 0)];
    let mut ev = Evaluator::new(1);
    ev.add_image(&preds, &truths);
    let r = ev.compute();
    assert_eq!(r.map50, 0.0);
    assert_eq!(r.num_preds, 100);
    // Without the cap the TP counts at rank 101: precision 1/101 at recall 1 → AP = 1/101.
    let mut ev = Evaluator::with_config(1, EvalConfig { max_dets_per_class: None, ..EvalConfig::default() });
    ev.add_image(&preds, &truths);
    assert!(close(ev.compute().map50, 1.0 / 101.0));
}

#[test]
fn confusion_matrix() {
    let mut ev = Evaluator::new(2);
    ev.add_image(
        &[
            det(0., 0., 10., 10., 0.9, 0),     // matches gt0 (class 0): [0][0]
            det(20., 0., 30., 10., 0.8, 0),    // matches gt1 (class 1): [0][1]
            det(40., 0., 50., 10., 0.5, 1),    // no gt: [1][bg]
            det(0., 0., 10., 10., 0.1, 1),     // below conf 0.25: ignored
            det(0.5, 0.5, 10.5, 10.5, 0.4, 0), // gt0 already taken: [0][bg]
        ],
        &[
            gt(0., 0., 10., 10., 0),
            gt(20., 0., 30., 10., 1),
            gt(60., 0., 70., 10., 1), // missed: [bg][1]
        ],
    );
    let m = ev.compute().confusion.matrix;
    assert_eq!(m, vec![vec![1, 1, 1], vec![0, 0, 1], vec![0, 1, 0]]);
}

#[test]
fn report_serialises_with_api_field_names() {
    let mut ev = Evaluator::new(1);
    ev.add_image(&[det(0., 0., 10., 10., 0.9, 0)], &[gt(0., 0., 10., 10., 0)]);
    let v = serde_json::to_value(ev.compute()).unwrap();
    for k in ["map50", "map50_95", "precision", "recall", "per_class"] {
        assert!(v.get(k).is_some(), "missing {k}");
    }
    for k in ["class", "map50", "map50_95", "precision", "recall"] {
        assert!(v["per_class"][0].get(k).is_some(), "missing per_class.{k}");
    }
    let back: folo_vision::EvalReport = serde_json::from_value(v).unwrap();
    assert_eq!(back.per_class[0].class, 0);
}

#[test]
fn random_perfect_and_shuffled_predictions() {
    // Predictions equal to the ground truth score AP 1 regardless of order; every metric lies in [0, 1].
    let mut rng = common::Rng::new(3);
    let mut perfect = Evaluator::new(5);
    let mut noisy = Evaluator::new(5);
    for _ in 0..50 {
        let truths: Vec<Truth> = (0..rng.below(8))
            .map(|_| Truth { bbox: rng.bbox(640.0, 10.0, 200.0), class: rng.below(5) as u32 })
            .collect();
        let preds: Vec<Detection> = truths
            .iter()
            .rev()
            .map(|t| Detection { bbox: t.bbox, score: 0.5 + 0.5 * rng.f32(), class: t.class })
            .collect();
        perfect.add_image(&preds, &truths);
        let mut preds: Vec<Detection> = truths
            .iter()
            .map(|t| Detection { bbox: rng.jitter(&t.bbox, 0.15), score: rng.f32(), class: t.class })
            .collect();
        for _ in 0..rng.below(5) {
            preds.push(Detection {
                bbox: rng.bbox(640.0, 10.0, 200.0),
                score: rng.f32(),
                class: rng.below(5) as u32,
            });
        }
        noisy.add_image(&preds, &truths);
    }
    let r = perfect.compute();
    assert!(close(r.map50, 1.0) && close(r.map50_95, 1.0) && close(r.precision, 1.0) && close(r.recall, 1.0));
    let r = noisy.compute();
    for v in [r.map50, r.map50_95, r.precision, r.recall, r.f1] {
        assert!((0.0..=1.0).contains(&v));
    }
    assert!(r.map50 >= r.map50_95);
    assert!(r.map50 > 0.3, "jittered boxes should mostly match at 0.5: {}", r.map50);
}
