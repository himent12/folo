//! Unit tests for detection evaluation diagnostics (`metrics/diagnostics.rs`):
//! - `DetectionConfusionMatrix` (background row/column, deterministic greedy IoU matching, per-class
//!   TP/FP/FN/P/R/F1, row/col normalization, merge, validation),
//! - `compute_pr_curves` (monotonic precision envelope, continuous AP == area under envelope, 101-point
//!   AP parity with `Evaluator`, exact optimal F1 threshold across tied & distinct scores),
//! - `compute_detection_calibration` & `reliability_from_scores` (ECE, MCE, Brier score, per-bin counts
//!   and calibration gaps, perfect & maximally miscalibrated cases, validation).

use super::diagnostics::{
    DetectionConfusionMatrix, compute_detection_calibration, compute_pr_curves, reliability_from_scores,
};
use super::{EvalConfig, Evaluator, Truth};
use crate::boxes::Xyxy;
use crate::nms::Detection;

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn truth(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

#[test]
fn confusion_matrix_matching_background_normalization_and_merge() {
    // 2 foreground classes (0, 1); index 2 = background.
    // Image 0:
    //   GT0: [0,0,10,10] cls 0 -> matched by P0 ([0,0,10,10], score 0.9, cls 0) => TP[0][0] += 1
    //   GT1: [20,0,30,10] cls 1 -> matched by P1 ([20,0,30,10], score 0.8, cls 0) => misclass[0][1] += 1
    //   GT2: [40,0,50,10] cls 1 -> unmatched (P2 has score 0.1 < 0.25) => FN[bg][1] += 1
    //   P3: [70,0,80,10] score 0.7 cls 1 -> unmatched background FP => FP[1][bg] += 1
    let img0_preds = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.9, 0),
        det(20.0, 0.0, 30.0, 10.0, 0.8, 0),
        det(40.0, 0.0, 50.0, 10.0, 0.1, 1), // below conf_thresh 0.25
        det(70.0, 0.0, 80.0, 10.0, 0.7, 1),
    ];
    let img0_truths = vec![
        truth(0.0, 0.0, 10.0, 10.0, 0),
        truth(20.0, 0.0, 30.0, 10.0, 1),
        truth(40.0, 0.0, 50.0, 10.0, 1),
    ];

    // Image 1: competing predictions on one GT (tie-break by highest IoU, then higher score).
    let img1_preds = vec![
        det(0.0, 0.0, 9.0, 10.0, 0.95, 1),  // IoU = 0.9
        det(0.0, 0.0, 10.0, 10.0, 0.60, 0), // IoU = 1.0 -> wins greedy IoU match!
    ];
    let img1_truths = vec![truth(0.0, 0.0, 10.0, 10.0, 0)];

    let mut cm = DetectionConfusionMatrix::new(2, 0.5, 0.25).unwrap();
    cm.add_image(&img0_preds, &img0_truths).unwrap();
    let mut cm2 = DetectionConfusionMatrix::new(2, 0.5, 0.25).unwrap();
    cm2.add_image(&img1_preds, &img1_truths).unwrap();
    cm.merge(&cm2).unwrap();

    let bg = cm.background_index();
    assert_eq!(bg, 2);
    assert_eq!(cm.get(0, 0), 2); // 1 from img0 + 1 from img1
    assert_eq!(cm.get(0, 1), 1); // cls 1 predicted as cls 0
    assert_eq!(cm.get(bg, 1), 1); // missed cls 1 GT
    assert_eq!(cm.get(1, bg), 2); // P3 from img0 + loser P0 from img1
    assert_eq!(cm.get(bg, bg), 0);

    assert_eq!(cm.true_positives(0), 2);
    assert_eq!(cm.false_positives(0), 1);
    assert_eq!(cm.false_negatives(0), 0);
    assert!((cm.precision(0) - 2.0 / 3.0).abs() < 1e-12);
    assert!((cm.recall(0) - 1.0).abs() < 1e-12);
    assert!((cm.f1(0) - 0.8).abs() < 1e-12);
    assert_eq!(cm.total_truths(), 4);
    assert_eq!(cm.total_predictions(), 5);

    // Row & column normalization sums to 1.0 on every non-empty row/column.
    let rn = cm.row_normalized();
    for (r, row) in rn.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        if cm.counts()[r].iter().sum::<u64>() > 0 {
            assert!((sum - 1.0).abs() < 1e-12);
        }
    }
    let cn = cm.col_normalized();
    for c in 0..=2 {
        let col_sum: u64 = cm.counts().iter().map(|row| row[c]).sum();
        let norm_sum: f64 = cn.iter().map(|row| row[c]).sum();
        if col_sum > 0 {
            assert!((norm_sum - 1.0).abs() < 1e-12);
        }
    }

    // Serde round-trip succeeds on valid matrix and rejects malformed shapes/thresholds.
    assert_eq!((cm.num_classes(), cm.iou_thresh(), cm.conf_thresh()), (2, 0.5, 0.25));
    let json = serde_json::to_string(&cm).unwrap();
    assert_eq!(serde_json::from_str::<DetectionConfusionMatrix>(&json).unwrap(), cm);
    let malformed = r#"{"num_classes":1,"conf_thresh":0.25,"iou_thresh":0.5,"counts":[]}"#;
    assert!(serde_json::from_str::<DetectionConfusionMatrix>(malformed).is_err());
    assert!(DetectionConfusionMatrix::new(usize::MAX, 0.5, 0.25).is_err());
    // Large u64::MAX deserialized counts must normalize without overflow and reject overflowing merge.
    let huge_json = format!(
        r#"{{"num_classes":1,"conf_thresh":0.25,"iou_thresh":0.5,"counts":[[{},1],[0,0]]}}"#,
        u64::MAX
    );
    let mut huge_cm: DetectionConfusionMatrix = serde_json::from_str(&huge_json).unwrap();
    let huge_rn = huge_cm.row_normalized();
    assert!((huge_rn[0][0] - 1.0).abs() < 1e-12);
    assert!(huge_cm.merge(&huge_cm.clone()).is_err());

    // Validation checks.
    assert!(DetectionConfusionMatrix::new(2, -0.1, 0.25).is_err());
    assert!(DetectionConfusionMatrix::new(2, 1.1, 0.25).is_err());
    assert!(DetectionConfusionMatrix::new(2, 0.5, f32::NAN).is_err());
    assert!(cm.add_image(&[det(0.0, 0.0, 1.0, 1.0, 0.5, 2)], &[]).is_err());
    assert!(cm.merge(&DetectionConfusionMatrix::new(2, 0.75, 0.25).unwrap()).is_err());
}

#[test]
fn pr_curve_monotonic_envelope_ap_consistency_and_optimal_f1() {
    // Construct a 1-class dataset with 4 GTs across 2 images and 6 detections whose sorted TP/FP
    // sequence is [TP(0.95), FP(0.85), TP(0.75), TP(0.65), FP(0.40), TP(0.20)].
    // Raw (recall, precision):
    //   1: (0.25, 1.00)
    //   2: (0.25, 0.50)
    //   3: (0.50, 2/3)
    //   4: (0.75, 3/4)
    //   5: (0.75, 3/5)
    //   6: (1.00, 4/6 = 2/3)
    // Monotonic precision envelope (suffix max):
    //   [1.00, 0.75, 0.75, 0.75, 2/3, 2/3]
    // Continuous AP = 0.25 * 1.0 + 0.25 * 0.75 + 0.25 * 0.75 + 0.25 * (2/3)
    //               = 0.25 * (1.0 + 0.75 + 0.75 + 2/3) = 0.25 * (3.166666...) = 19/24.
    // F1 at each unique threshold:
    //   0.95: P=1.0, R=0.25 -> F1 = 0.40
    //   0.85: P=0.5, R=0.25 -> F1 = 1/3
    //   0.75: P=2/3, R=0.50 -> F1 = 4/7 ≈ 0.5714
    //   0.65: P=3/4, R=0.75 -> F1 = 0.75
    //   0.40: P=3/5, R=0.75 -> F1 = 2/3
    //   0.20: P=2/3, R=1.00 -> F1 = 0.80  <-- global optimum!
    let img0 = (
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.95, 0),   // TP
            det(50.0, 50.0, 60.0, 60.0, 0.85, 0), // FP
            det(20.0, 0.0, 30.0, 10.0, 0.75, 0),  // TP
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 0.0, 30.0, 10.0, 0)],
    );
    let img1 = (
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.65, 0),   // TP
            det(80.0, 80.0, 90.0, 90.0, 0.40, 0), // FP
            det(20.0, 0.0, 30.0, 10.0, 0.20, 0),  // TP
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 0.0, 30.0, 10.0, 0)],
    );
    let images = [img0, img1];

    let diag = compute_pr_curves(1, 0.5, &images).unwrap();
    let c0 = &diag.per_class[0];
    assert_eq!(c0.num_truths, 4);
    assert_eq!(c0.num_preds, 6);

    // Check monotonic recall (non-decreasing) and precision envelope (non-increasing).
    for w in c0.points.windows(2) {
        assert!(w[1].recall >= w[0].recall, "recall must be non-decreasing");
        assert!(w[1].precision_envelope <= w[0].precision_envelope, "envelope must be non-increasing");
        assert!(w[0].precision_envelope >= w[0].precision - 1e-15);
    }

    let want_ap_cont = 19.0 / 24.0;
    assert!(
        (c0.ap_continuous - want_ap_cont).abs() < 1e-12,
        "ap_continuous {} vs {want_ap_cont}",
        c0.ap_continuous
    );

    // Cross-check ap_101 directly against Evaluator::compute().map50!
    let mut ev = Evaluator::with_config(1, EvalConfig { max_dets_per_class: None, ..EvalConfig::default() });
    for (p, t) in &images {
        ev.add_image(p, t);
    }
    let report = ev.compute();
    assert!(
        (c0.ap_101 as f32 - report.map50).abs() < 1e-6,
        "ap_101 {} vs Evaluator map50 {}",
        c0.ap_101,
        report.map50
    );

    // Exact optimal F1 operating point.
    assert_eq!(c0.best_f1.conf_thresh, 0.20);
    assert!((c0.best_f1.f1 - 0.80).abs() < 1e-12);
    assert!((c0.best_f1.precision - 2.0 / 3.0).abs() < 1e-12);
    assert!((c0.best_f1.recall - 1.0).abs() < 1e-12);
    assert_eq!((c0.best_f1.tp, c0.best_f1.fp, c0.best_f1.fn_count), (4, 2, 0));

    // Tied scores must be thresholded together (cannot split equal-score TP and FP).
    let tied_img = [(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.5, 0),   // TP
            det(50.0, 50.0, 60.0, 60.0, 0.5, 0), // FP with exact same score 0.5
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0)],
    )];
    let tied_diag = compute_pr_curves(1, 0.5, &tied_img).unwrap();
    // At threshold 0.5 both detections pass -> TP=1, FP=1 -> P=0.5, R=1.0 -> F1 = 2/3.
    assert!((tied_diag.per_class[0].best_f1.f1 - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(tied_diag.per_class[0].best_f1.tp, 1);
    assert_eq!(tied_diag.per_class[0].best_f1.fp, 1);

    // Exact integer F1 tie-break: 2 GTs; at 0.9: 1 TP + 3 FP -> F1 = 2/6 = 1/3; at 0.5: 2 TP + 8 FP -> F1 = 4/12 = 1/3.
    // Must prefer the HIGHER threshold 0.9 despite floating-point rounding!
    let f1_tie_img = [(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),    // TP 1
            det(100.0, 0.0, 110.0, 10.0, 0.9, 0), // FP 1
            det(120.0, 0.0, 130.0, 10.0, 0.9, 0), // FP 2
            det(140.0, 0.0, 150.0, 10.0, 0.9, 0), // FP 3
            det(20.0, 0.0, 30.0, 10.0, 0.5, 0),   // TP 2
            det(160.0, 0.0, 170.0, 10.0, 0.5, 0), // FP 4
            det(180.0, 0.0, 190.0, 10.0, 0.5, 0), // FP 5
            det(200.0, 0.0, 210.0, 10.0, 0.5, 0), // FP 6
            det(220.0, 0.0, 230.0, 10.0, 0.5, 0), // FP 7
            det(240.0, 0.0, 250.0, 10.0, 0.5, 0), // FP 8
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 0.0, 30.0, 10.0, 0)],
    )];
    let f1_tie_diag = compute_pr_curves(1, 0.5, &f1_tie_img).unwrap();
    assert_eq!(f1_tie_diag.per_class[0].best_f1.conf_thresh, 0.9);
    assert!((f1_tie_diag.per_class[0].best_f1.f1 - 1.0 / 3.0).abs() < 1e-12);

    // Zero-TP and negative-score cases: counts agree with returned threshold and no-GT classes retain points.
    let fp_only = [(
        vec![det(50.0, 50.0, 60.0, 60.0, 0.9, 0), det(70.0, 70.0, 80.0, 80.0, -0.2, 1)],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0)],
    )];
    let fp_diag = compute_pr_curves(2, 0.5, &fp_only).unwrap();
    assert_eq!(fp_diag.per_class[0].best_f1.conf_thresh, 0.0);
    assert_eq!(fp_diag.per_class[0].best_f1.fp, 1);
    assert_eq!(fp_diag.per_class[0].best_f1.fn_count, 1);
    assert_eq!(fp_diag.per_class[1].num_preds, 1);
    assert_eq!(fp_diag.per_class[1].points.len(), 1);
    assert_eq!(fp_diag.per_class[1].points[0].confidence, -0.2);
}

#[test]
fn detection_calibration_reliability_ece_mce_brier() {
    // 1. Perfectly calibrated scores: bin [0.2, 0.4) has 1 TP + 3 FP at conf 0.25 -> acc = 0.25, gap = 0;
    //    bin [0.6, 0.8) has 3 TP + 1 FP at conf 0.75 -> acc = 0.75, gap = 0.
    let scores = [0.25f32, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75];
    let flags = [true, false, false, false, true, true, true, false];
    let cal = reliability_from_scores(&scores, &flags, 5).unwrap();
    assert_eq!(cal.count, 8);
    assert!(cal.ece < 1e-12, "perfectly calibrated ECE must be 0, got {}", cal.ece);
    assert!(cal.mce < 1e-12, "perfectly calibrated MCE must be 0, got {}", cal.mce);
    // Brier = 0.5 * (0.75² * 1 + 0.25² * 3)/4 + 0.5 * (0.25² * 3 + 0.75² * 1)/4 = 3/16 = 0.1875.
    assert!((cal.brier - 0.1875).abs() < 1e-12);
    assert_eq!(cal.bins.len(), 5);
    assert_eq!(cal.bins[1].count, 4);
    assert_eq!((cal.bins[1].tp, cal.bins[1].fp), (1, 3));
    assert_eq!(cal.bins[3].count, 4);
    assert_eq!((cal.bins[3].tp, cal.bins[3].fp), (3, 1));

    // 2. End-to-end `compute_detection_calibration` on 2 classes:
    //    Class 0: 2 detections at score 0.9 -> 1 TP, 1 FP -> bin [0.8, 1.0]: conf=0.9, acc=0.5, gap=0.4.
    //    Class 1: 1 detection at score 0.8 -> 1 TP -> bin [0.8, 1.0]: conf=0.8, acc=1.0, gap=0.2.
    let images = [(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9, 0),   // TP cls 0
            det(50.0, 50.0, 60.0, 60.0, 0.9, 0), // FP cls 0
            det(20.0, 0.0, 30.0, 10.0, 0.8, 1),  // TP cls 1
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 0.0, 30.0, 10.0, 1)],
    )];
    let dc = compute_detection_calibration(2, 0.5, 0.0, 5, &images).unwrap();
    assert_eq!(dc.overall.count, 3);
    // Pooled in bin [0.8, 1.0]: scores [0.9, 0.9, 0.8] -> avg_conf = 2.6/3, avg_acc = 2/3, gap = 0.6/3 = 0.2.
    assert!((dc.overall.ece - 0.2).abs() < 1e-6);
    assert!((dc.overall.mce - 0.2).abs() < 1e-6);
    // Brier = ((0.9-1)² + (0.9-0)² + (0.8-1)²) / 3 = (0.01 + 0.81 + 0.04) / 3 = 0.86 / 3.
    assert!((dc.overall.brier - 0.86 / 3.0).abs() < 1e-6);
    assert!((dc.per_class[0].ece - 0.4).abs() < 1e-6);
    assert!((dc.per_class[1].ece - 0.2).abs() < 1e-6);

    // 3. Empty input & validation checks.
    let empty = reliability_from_scores(&[], &[], 10).unwrap();
    assert_eq!(empty.count, 0);
    assert_eq!(empty.ece, 0.0);
    assert_eq!(empty.mce, 0.0);
    assert_eq!(empty.brier, 0.0);
    assert!(reliability_from_scores(&[0.5], &[true], 0).is_err());
    assert!(reliability_from_scores(&[1.2], &[true], 5).is_err());
    assert!(reliability_from_scores(&[-0.1], &[false], 5).is_err());
    assert!(reliability_from_scores(&[0.5], &[], 5).is_err());
    // Negative and >1 detection scores must be symmetrically rejected by `compute_detection_calibration`.
    let neg_img = [(vec![det(0.0, 0.0, 10.0, 10.0, -0.1, 0)], vec![truth(0.0, 0.0, 10.0, 10.0, 0)])];
    assert!(compute_detection_calibration(1, 0.5, 0.0, 5, &neg_img).is_err());
    let over_img = [(vec![det(0.0, 0.0, 10.0, 10.0, 1.2, 0)], vec![truth(0.0, 0.0, 10.0, 10.0, 0)])];
    assert!(compute_detection_calibration(1, 0.5, 0.0, 5, &over_img).is_err());
}
