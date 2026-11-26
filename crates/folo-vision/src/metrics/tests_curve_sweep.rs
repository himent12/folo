//! Unit tests for confidence-threshold sweeps ([`compute_confidence_sweep`]) and extended reliability-bin
//! calibration diagnostics ([`reliability_diagnostics_from_scores`], [`compute_calibration_report`]).

use super::*;
use crate::boxes::Xyxy;
use crate::nms::Detection;

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn truth(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

#[test]
fn confidence_sweep_optimal_micro_and_macro_f1() {
    // 2 classes, 2 truths per class:
    // - class 0: 2 TPs at 0.90, 0.70; 2 FPs at 0.40, 0.20 -> peak F1 = 1.0 at conf 0.70.
    // - class 1: 2 TPs at 0.85, 0.60; 1 FP at 0.30        -> peak F1 = 1.0 at conf 0.60.
    // Micro optimal: conf = 0.60 achieves 4 TP, 0 FP, 0 FN -> F1 = 1.0!
    // Macro optimal on 0.00..=1.00 (101 points): conf = 0.60 achieves mean F1 = 1.0!
    let images = vec![(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.90, 0),
            det(20.0, 0.0, 30.0, 10.0, 0.70, 0),
            det(40.0, 0.0, 50.0, 10.0, 0.40, 0),
            det(60.0, 0.0, 70.0, 10.0, 0.20, 0),
            det(0.0, 20.0, 10.0, 30.0, 0.85, 1),
            det(20.0, 20.0, 30.0, 30.0, 0.60, 1),
            det(40.0, 20.0, 50.0, 30.0, 0.30, 1),
        ],
        vec![
            truth(0.0, 0.0, 10.0, 10.0, 0),
            truth(20.0, 0.0, 30.0, 10.0, 0),
            truth(0.0, 20.0, 10.0, 30.0, 1),
            truth(20.0, 20.0, 30.0, 30.0, 1),
        ],
    )];

    let cfg = ConfidenceSweepConfig::default();
    let sweep = compute_confidence_sweep(2, &cfg, &images).unwrap();

    assert_eq!(sweep.thresholds.len(), 101);
    assert_eq!(sweep.micro_curve.len(), 101);
    assert_eq!(sweep.macro_curve.len(), 101);
    assert_eq!(sweep.per_class.len(), 2);

    // Exact per-class optimal F1 thresholds.
    assert!((sweep.per_class[0].best_f1.conf_thresh - 0.70).abs() < 1e-6);
    assert!((sweep.per_class[0].best_f1.f1 - 1.0).abs() < 1e-12);
    assert!((sweep.per_class[1].best_f1.conf_thresh - 0.60).abs() < 1e-6);
    assert!((sweep.per_class[1].best_f1.f1 - 1.0).abs() < 1e-12);

    // Exact micro and macro peak F1.
    assert!((sweep.best_micro_f1.conf_thresh - 0.60).abs() < 1e-6);
    assert!((sweep.best_micro_f1.f1 - 1.0).abs() < 1e-12);
    assert_eq!((sweep.best_micro_f1.tp, sweep.best_micro_f1.fp, sweep.best_micro_f1.fn_count), (4, 0, 0));

    assert!((sweep.best_macro_f1.conf_thresh - 0.60).abs() < 1e-6);
    assert!((sweep.best_macro_f1.f1 - 1.0).abs() < 1e-12);

    // Recall must be monotonically non-increasing as confidence threshold increases.
    for w in sweep.micro_curve.windows(2) {
        assert!(w[0].recall >= w[1].recall - 1e-12);
    }

    // Matches `compute_pr_curves` per-class best F1.
    let pr = compute_pr_curves(2, 0.5, &images).unwrap();
    for c in 0..2 {
        assert_eq!(sweep.per_class[c].best_f1, pr.per_class[c].best_f1);
    }
}

#[test]
fn calibration_report_equal_width_and_equal_mass_bins() {
    // 4 samples in 2 equal-width bins [0, 0.5) and [0.5, 1.0]:
    // Bin 0: [0.20 (FP), 0.40 (FP)] -> avg_conf = 0.30, avg_acc = 0.00, gap = 0.30, over = 0.30
    // Bin 1: [0.80 (TP), 0.90 (TP)] -> avg_conf = 0.85, avg_acc = 1.00, gap = 0.15, over = 0.00
    // ECE = 0.5 * 0.30 + 0.5 * 0.15 = 0.225; MCE = 0.30;
    // signed_bias = ((0.2-0) + (0.4-0) + (0.8-1) + (0.9-1)) / 4 = 0.30 / 4 = 0.075;
    // overconfidence_error = 0.5 * 0.30 * 0.30 + 0.5 * 0.85 * 0.0 = 0.045.
    let scores = [0.20f32, 0.40, 0.80, 0.90];
    let is_tp = [false, false, true, true];

    let ew = reliability_diagnostics_from_scores(&scores, &is_tp, 2, BinningStrategy::EqualWidth).unwrap();
    assert_eq!(ew.count, 4);
    assert_eq!(ew.bins.len(), 2);
    assert!((ew.ece - 0.225).abs() < 1e-6, "got ece {}", ew.ece);
    assert!((ew.mce - 0.30).abs() < 1e-6, "got mce {}", ew.mce);
    assert!((ew.signed_bias - 0.075).abs() < 1e-6, "got bias {}", ew.signed_bias);
    assert!((ew.overconfidence_error - 0.045).abs() < 1e-6, "got oe {}", ew.overconfidence_error);

    // Equal-mass (adaptive quantile) binning splits clustered scores evenly across bins.
    let clustered = [0.81f32, 0.82, 0.85, 0.88, 0.92, 0.96];
    let flags = [false, true, false, true, true, true];
    let em = reliability_diagnostics_from_scores(&clustered, &flags, 3, BinningStrategy::EqualMass).unwrap();
    assert_eq!(em.bins.len(), 3);
    assert!(em.bins.iter().all(|b| b.count == 2), "equal-mass bins must each contain 2 samples");
    assert!(em.ece.is_finite() && em.mce >= em.ece - 1e-12);

    // End-to-end image calibration report computes overall + per-class + SCE cleanly.
    let images = vec![(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.90, 0),  // TP
            det(40.0, 0.0, 50.0, 10.0, 0.80, 0), // FP
            det(0.0, 20.0, 10.0, 30.0, 0.70, 1), // TP
        ],
        vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(0.0, 20.0, 10.0, 30.0, 1)],
    )];
    let rep = compute_calibration_report(
        2,
        &CalibrationReportConfig {
            iou_thresh: 0.5,
            conf_thresh: 0.0,
            num_bins: 5,
            binning: BinningStrategy::EqualWidth,
        },
        &images,
    )
    .unwrap();
    assert_eq!(rep.overall.count, 3);
    assert_eq!(rep.per_class.len(), 2);
    let want_sce = 0.5 * (rep.per_class[0].ece + rep.per_class[1].ece);
    assert!((rep.static_calibration_error - want_sce).abs() < 1e-12);
    assert!((rep.max_class_ece - rep.per_class[0].ece.max(rep.per_class[1].ece)).abs() < 1e-12);

    // Empty inputs & validation errors.
    let empty = reliability_diagnostics_from_scores(&[], &[], 5, BinningStrategy::EqualMass).unwrap();
    assert_eq!(empty.count, 0);
    assert_eq!(empty.ece, 0.0);
    assert!(reliability_diagnostics_from_scores(&[1.2], &[true], 5, BinningStrategy::EqualWidth).is_err());
    assert!(
        compute_confidence_sweep(
            1,
            &ConfidenceSweepConfig { num_thresholds: 1, ..ConfidenceSweepConfig::default() },
            &images
        )
        .is_err()
    );
}
