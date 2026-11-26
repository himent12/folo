//! Unit tests for `NormalizedConfusionView` (normalized precision/recall view of
//! `DetectionConfusionMatrix`) and seeded bootstrap confidence-interval mAP (`compute_bootstrap_map`):
//! exact point-estimate parity with `Evaluator` (including tied scores), bitwise determinism across
//! repeated runs, statistical CI ordering and width properties, single-class/empty edge cases, and
//! validation.

use super::bootstrap::{BootstrapConfig, compute_bootstrap_map};
use super::diagnostics::DetectionConfusionMatrix;
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
fn confusion_matrix_normalized_precision_recall_view() {
    // 2 foreground classes (0, 1); background = 2.
    // GT0 (cls 0) -> P0 (cls 0) => TP[0][0] = 1
    // GT1 (cls 0) -> P1 (cls 0) => TP[0][0] = 2
    // GT2 (cls 1) -> P2 (cls 0) => misclass[0][1] = 1 (class_fp for cls 0, class_fn for cls 1)
    // GT3 (cls 1) -> missed     => bg_fn[2][1] = 1
    // P3 (cls 1)  -> unmatched  => bg_fp[1][2] = 1
    // P4 (cls 1)  -> unmatched  => bg_fp[1][2] = 2
    let preds = vec![
        det(0.0, 0.0, 10.0, 10.0, 0.95, 0),
        det(20.0, 0.0, 30.0, 10.0, 0.90, 0),
        det(40.0, 0.0, 50.0, 10.0, 0.85, 0),
        det(70.0, 0.0, 80.0, 10.0, 0.80, 1),
        det(90.0, 0.0, 100.0, 10.0, 0.75, 1),
    ];
    let truths = vec![
        truth(0.0, 0.0, 10.0, 10.0, 0),
        truth(20.0, 0.0, 30.0, 10.0, 0),
        truth(40.0, 0.0, 50.0, 10.0, 1),
        truth(120.0, 0.0, 130.0, 10.0, 1),
    ];

    let cm = DetectionConfusionMatrix::from_images(2, 0.5, 0.25, &[(preds, truths)]).unwrap();
    assert_eq!(cm.background_fp(0), 0);
    assert_eq!(cm.background_fp(1), 2);
    assert_eq!(cm.background_fn(0), 0);
    assert_eq!(cm.background_fn(1), 1);

    let view = cm.normalized_view();
    assert_eq!(view.num_classes, 2);
    assert_eq!(view.per_class.len(), 2);

    // Class 0: TP=2, class_fp=1, bg_fp=0, fn=0, support=2, P=2/3, R=1.0, F1=0.8.
    let c0 = view.per_class[0];
    assert_eq!((c0.tp, c0.fp, c0.class_fp, c0.background_fp, c0.fn_count, c0.support), (2, 1, 1, 0, 0, 2));
    assert!((c0.precision - 2.0 / 3.0).abs() < 1e-12);
    assert!((c0.recall - 1.0).abs() < 1e-12);
    assert!((view.precision_matrix[0][0] - c0.precision).abs() < 1e-12);
    assert!((view.recall_matrix[0][0] - c0.recall).abs() < 1e-12);

    // Class 1: TP=0, class_fp=0, bg_fp=2, class_fn=1, bg_fn=1, support=2, P=0, R=0, F1=0.
    let c1 = view.per_class[1];
    assert_eq!(
        (c1.tp, c1.fp, c1.class_fp, c1.background_fp, c1.class_fn, c1.background_fn, c1.support),
        (0, 2, 0, 2, 1, 1, 2)
    );
    assert!((view.precision_matrix[1][2] - 1.0).abs() < 1e-12);
    assert!((view.recall_matrix[2][1] - 0.5).abs() < 1e-12);
    assert!((view.recall_matrix[0][1] - 0.5).abs() < 1e-12);

    // Macro & micro checks.
    assert!((view.macro_precision - (1.0 / 3.0)).abs() < 1e-12);
    assert!((view.macro_recall - 0.5).abs() < 1e-12);
    assert!((view.macro_f1 - 0.4).abs() < 1e-12);
    assert!((view.micro_precision - 2.0 / 5.0).abs() < 1e-12);
    assert!((view.micro_recall - 2.0 / 4.0).abs() < 1e-12);

    // Top confusions sorted by descending count: (1, 2) count 2 first, then (0, 1) and (2, 1) count 1.
    assert_eq!(view.top_confusions.len(), 3);
    assert_eq!(
        (view.top_confusions[0].pred_class, view.top_confusions[0].truth_class, view.top_confusions[0].count),
        (1, 2, 2)
    );
    assert_eq!(
        (view.top_confusions[1].pred_class, view.top_confusions[1].truth_class, view.top_confusions[1].count),
        (0, 1, 1)
    );
    assert_eq!(
        (view.top_confusions[2].pred_class, view.top_confusions[2].truth_class, view.top_confusions[2].count),
        (2, 1, 1)
    );
}

#[test]
fn bootstrap_map_parity_determinism_and_ci_properties() {
    // 6 varied images across 2 classes (including tied scores within an image).
    let images: Vec<(Vec<Detection>, Vec<Truth>)> = vec![
        // Img 0: perfect cls 0, perfect cls 1
        (
            vec![det(0.0, 0.0, 10.0, 10.0, 0.95, 0), det(20.0, 0.0, 30.0, 10.0, 0.90, 1)],
            vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 0.0, 30.0, 10.0, 1)],
        ),
        // Img 1: tied score TP + FP on cls 0
        (
            vec![
                det(0.0, 0.0, 10.0, 10.0, 0.80, 0),
                det(50.0, 50.0, 60.0, 60.0, 0.80, 0), // FP at identical score
            ],
            vec![truth(0.0, 0.0, 10.0, 10.0, 0)],
        ),
        // Img 2: missed cls 0, good cls 1 (IoU ~ 0.64 -> matches 0.50..0.60 only)
        (
            vec![det(2.0, 0.0, 12.0, 10.0, 0.85, 1)],
            vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(0.0, 0.0, 10.0, 10.0, 1)],
        ),
        // Img 3: perfect cls 0 + high-score FP on cls 1
        (
            vec![det(0.0, 0.0, 10.0, 10.0, 0.92, 0), det(40.0, 40.0, 50.0, 50.0, 0.98, 1)],
            vec![truth(0.0, 0.0, 10.0, 10.0, 0), truth(20.0, 20.0, 30.0, 30.0, 1)],
        ),
        // Img 4: perfect cls 1 only
        (vec![det(10.0, 10.0, 20.0, 20.0, 0.99, 1)], vec![truth(10.0, 10.0, 20.0, 20.0, 1)]),
        // Img 5: partial overlap cls 0 (IoU = 9/11 ≈ 0.818)
        (vec![det(1.0, 0.0, 11.0, 10.0, 0.70, 0)], vec![truth(0.0, 0.0, 10.0, 10.0, 0)]),
    ];

    // Exact point-estimate parity with `Evaluator::compute`.
    let mut ev = Evaluator::with_config(2, EvalConfig::default());
    for (p, t) in &images {
        ev.add_image(p, t);
    }
    let eval_rep = ev.compute();

    let cfg95 =
        BootstrapConfig { num_samples: 400, confidence_level: 0.95, seed: 42, eval: EvalConfig::default() };
    let b1 = compute_bootstrap_map(2, &images, &cfg95).unwrap();
    let b2 = compute_bootstrap_map(2, &images, &cfg95).unwrap();

    // Bitwise deterministic across repeated calls.
    assert_eq!(b1, b2);
    assert!((b1.map50.point as f32 - eval_rep.map50).abs() < 1e-6);
    assert!((b1.map50_95.point as f32 - eval_rep.map50_95).abs() < 1e-6);
    for (t, ci) in b1.map_per_iou.iter().enumerate() {
        assert!((ci.point as f32 - eval_rep.map_per_iou[t]).abs() < 1e-6);
        assert!(0.0 <= ci.lower && ci.lower <= ci.upper && ci.upper <= 1.0);
    }

    // Valid CI ordering and positive spread across heterogeneous images.
    assert!(0.0 <= b1.map50.lower && b1.map50.lower < b1.map50.upper && b1.map50.upper <= 1.0);
    assert!(0.0 <= b1.map50_95.lower && b1.map50_95.lower < b1.map50_95.upper && b1.map50_95.upper <= 1.0);
    assert!(b1.map50.std > 0.0 && b1.map50_95.std > 0.0);

    // Narrower confidence level (80% CI) sits inside 95% CI for identical seed & samples.
    let cfg80 = BootstrapConfig { confidence_level: 0.80, ..cfg95.clone() };
    let b80 = compute_bootstrap_map(2, &images, &cfg80).unwrap();
    assert!(b80.map50.lower >= b1.map50.lower - 1e-12);
    assert!(b80.map50.upper <= b1.map50.upper + 1e-12);
    assert!(b80.map50_95.lower >= b1.map50_95.lower - 1e-12);
    assert!(b80.map50_95.upper <= b1.map50_95.upper + 1e-12);

    // Zero-variance identical perfect images -> lower == point == upper == 1.0, std == 0.0.
    let perfect_imgs = vec![
        (vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0)], vec![truth(0.0, 0.0, 10.0, 10.0, 0)]),
        (vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0)], vec![truth(0.0, 0.0, 10.0, 10.0, 0)]),
    ];
    let bp = compute_bootstrap_map(1, &perfect_imgs, &BootstrapConfig { num_samples: 50, ..cfg95.clone() })
        .unwrap();
    assert!((bp.map50.point - 1.0).abs() < 1e-12);
    assert!((bp.map50.lower - 1.0).abs() < 1e-12);
    assert!((bp.map50.upper - 1.0).abs() < 1e-12);
    assert!(bp.map50.std < 1e-12);
    assert!((bp.map50_95.lower - 1.0).abs() < 1e-12);

    // Empty dataset and validation checks.
    let empty: Vec<(Vec<Detection>, Vec<Truth>)> = Vec::new();
    let be = compute_bootstrap_map(2, &empty, &cfg95).unwrap();
    assert_eq!(be.num_images, 0);
    assert_eq!(be.map50.point, 0.0);
    assert_eq!(be.per_class.len(), 2);

    assert!(compute_bootstrap_map(2, &images, &BootstrapConfig { num_samples: 0, ..cfg95.clone() }).is_err());
    assert!(
        compute_bootstrap_map(2, &images, &BootstrapConfig { confidence_level: 1.0, ..cfg95.clone() })
            .is_err()
    );
    assert!(
        compute_bootstrap_map(2, &images, &BootstrapConfig { confidence_level: 0.0, ..cfg95.clone() })
            .is_err()
    );
    assert!(compute_bootstrap_map(1, &images, &cfg95).is_err()); // class 1 out of bounds for num_classes = 1
}
