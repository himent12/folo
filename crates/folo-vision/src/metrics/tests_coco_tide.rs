//! Unit tests for COCO scale-bucketed AP/AR diagnostics (`compute_coco_scale_metrics`) and
//! TIDE-style detection error attribution (`compute_tide_diagnostics`).

use super::coco_scale::{ScaleAreaBounds, compute_coco_scale_metrics};
use super::tide::{TideConfig, TideDiagnostics, compute_tide_diagnostics};
use crate::boxes::Xyxy;
use crate::cocoeval::{CocoEval, CocoTruth};
use crate::metrics::Truth;
use crate::nms::Detection;

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn gt(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

/// `compute_coco_scale_metrics` with default COCO bounds must match `CocoEval` across all 12
/// summary metrics (`AP`, `AP50`, `AP75`, `AP_S`, `AP_M`, `AP_L`, `AR@1`, `AR@10`, `AR@100`,
/// `AR_S`, `AR_M`, `AR_L`) on a multi-image, multi-class, multi-scale dataset.
#[test]
fn coco_scale_metrics_matches_cocoeval_all_12_numbers() {
    // Small: 20x20 = 400 (< 1024); Medium: 50x50 = 2500 (1024..9216); Large: 120x120 = 14400 (> 9216).
    let img0_gt = vec![
        gt(10.0, 10.0, 30.0, 30.0, 0),     // small, class 0
        gt(100.0, 100.0, 150.0, 150.0, 0), // medium, class 0
        gt(200.0, 200.0, 320.0, 320.0, 1), // large, class 1
    ];
    let img0_pred = vec![
        det(10.0, 10.0, 30.0, 30.0, 0.95, 0),     // exact match small
        det(105.0, 100.0, 155.0, 150.0, 0.85, 0), // shifted medium (IoU = 45*50 / (2500+2500-2250) ≈ 0.818)
        det(400.0, 400.0, 420.0, 420.0, 0.60, 0), // small FP
        det(210.0, 200.0, 330.0, 320.0, 0.90, 1), // shifted large (IoU = 110*120 / (14400*2-13200) ≈ 0.846)
    ];

    let img1_gt = vec![
        gt(5.0, 5.0, 25.0, 25.0, 1),      // small, class 1 (missed)
        gt(60.0, 60.0, 120.0, 120.0, 1),  // medium (3600), class 1
        gt(300.0, 10.0, 420.0, 130.0, 0), // large (14400), class 0
    ];
    let img1_pred = vec![
        det(60.0, 60.0, 120.0, 120.0, 0.92, 1),   // exact medium
        det(300.0, 10.0, 420.0, 130.0, 0.88, 0),  // exact large
        det(500.0, 500.0, 620.0, 620.0, 0.70, 1), // large FP
    ];

    let images = vec![(img0_pred.clone(), img0_gt.clone()), (img1_pred.clone(), img1_gt.clone())];
    let diag = compute_coco_scale_metrics(2, &images, None).unwrap();

    // Reference CocoEval.
    let mut ce = CocoEval::new(2);
    for (img_id, (ps, ts)) in images.iter().enumerate() {
        let cts: Vec<CocoTruth> = ts.iter().map(|t| CocoTruth::from_box(t.bbox, t.class)).collect();
        ce.add_image(img_id as u64, ps, &cts);
    }
    let want = ce.compute();

    let close = |got: Option<f64>, ref_val: f64, name: &str| {
        let g = got.unwrap_or(-1.0);
        assert!((g - ref_val).abs() < 1e-12, "{name}: got {g} vs CocoEval {ref_val}");
    };
    close(diag.ap, want.ap, "AP");
    close(diag.ap50, want.ap50, "AP50");
    close(diag.ap75, want.ap75, "AP75");
    close(diag.ap_small, want.ap_small, "AP_S");
    close(diag.ap_medium, want.ap_medium, "AP_M");
    close(diag.ap_large, want.ap_large, "AP_L");
    close(diag.ar_1, want.ar1, "AR@1");
    close(diag.ar_10, want.ar10, "AR@10");
    close(diag.ar_100, want.ar100, "AR@100");
    close(diag.ar_small, want.ar_small, "AR_S");
    close(diag.ar_medium, want.ar_medium, "AR_M");
    close(diag.ar_large, want.ar_large, "AR_L");

    assert!(diag.ar_1.unwrap() <= diag.ar_10.unwrap() + 1e-12);
    assert!(diag.ar_10.unwrap() <= diag.ar_100.unwrap() + 1e-12);
    assert_eq!((diag.truths_all, diag.truths_small, diag.truths_medium, diag.truths_large), (6, 2, 2, 2));
}

/// Empty scale buckets return `None`, custom `ScaleAreaBounds` work, and invalid configs error.
#[test]
fn coco_scale_empty_buckets_custom_bounds_and_validation() {
    // Only a small box (10x10 = 100); medium and large buckets have zero truths -> None.
    let images = vec![(vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0)], vec![gt(0.0, 0.0, 10.0, 10.0, 0)])];
    let d = compute_coco_scale_metrics(1, &images, None).unwrap();
    assert!((d.ap.unwrap() - 1.0).abs() < 1e-12);
    assert!((d.ap_small.unwrap() - 1.0).abs() < 1e-12);
    assert!(d.ap_medium.is_none());
    assert!(d.ap_large.is_none());
    assert!(d.ar_medium.is_none());
    assert!(d.ar_large.is_none());

    // Custom bounds moving area 100 into medium bucket (small_max = 50, medium_max = 200).
    let custom = ScaleAreaBounds { small_max: 50.0, medium_max: 200.0, max_area: 1e6 };
    let dc = compute_coco_scale_metrics(1, &images, Some(custom)).unwrap();
    assert!(dc.ap_small.is_none());
    assert!((dc.ap_medium.unwrap() - 1.0).abs() < 1e-12);

    // Invalid bounds & out-of-range class id rejected.
    assert!(
        compute_coco_scale_metrics(
            1,
            &images,
            Some(ScaleAreaBounds { small_max: 200.0, medium_max: 100.0, max_area: 1e6 }),
        )
        .is_err()
    );
    assert!(compute_coco_scale_metrics(0, &images, None).is_err());
}

/// Isolate every single TIDE error category (`Cls`, `Loc`, `Both`, `Dup`, `Bkg`, `Miss`) and
/// verify exact mutually exclusive counts and `dAP` oracle impacts.
#[test]
fn tide_diagnostics_isolates_all_six_error_types() {
    let cfg = TideConfig::default(); // pos_iou = 0.50, bg_iou = 0.10

    // 1. Perfect predictions -> mAP = 1.0, zero errors, zero dAP.
    let perfect = vec![(vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 0)], vec![gt(0.0, 0.0, 10.0, 10.0, 0)])];
    let r = compute_tide_diagnostics(2, &perfect, &cfg).unwrap();
    assert!((r.map_base - 1.0).abs() < 1e-12);
    assert_eq!(r.counts.tp, 1);
    assert_eq!(r.counts.fp, 0);
    assert_eq!(r.counts.fn_count, 0);
    assert_eq!(r.counts.miss, 0);
    assert_eq!(r.impacts.cls, 0.0);
    assert_eq!(r.impacts.loc, 0.0);
    assert_eq!(r.impacts.both, 0.0);
    assert_eq!(r.impacts.dup, 0.0);
    assert_eq!(r.impacts.bkg, 0.0);
    assert_eq!(r.impacts.miss, 0.0);

    // 2. Pure Miss (no predictions): dAP_miss == 1.0, dAP_fn == 1.0.
    let missed = vec![(Vec::<Detection>::new(), vec![gt(0.0, 0.0, 10.0, 10.0, 0)])];
    let r = compute_tide_diagnostics(2, &missed, &cfg).unwrap();
    assert_eq!(r.map_base, 0.0);
    assert_eq!(r.counts.miss, 1);
    assert_eq!(r.counts.fn_count, 1);
    assert_eq!(r.counts.fp, 0);
    assert!((r.impacts.miss - 1.0).abs() < 1e-12);
    assert!((r.impacts.fn_impact - 1.0).abs() < 1e-12);
    assert_eq!(r.impacts.fp, 0.0);

    // 3. Pure Cls (exact box, predicted class 1 instead of class 0):
    // Class 0 has 1 GT, 0 preds; class 1 has 0 GTs, 1 Cls FP targeting class 0's GT.
    // Fixing Cls promotes the detection to a TP in class 0 -> AP_0 goes from 0.0 to 1.0!
    let cls_err = vec![(vec![det(0.0, 0.0, 10.0, 10.0, 0.9, 1)], vec![gt(0.0, 0.0, 10.0, 10.0, 0)])];
    let r = compute_tide_diagnostics(2, &cls_err, &cfg).unwrap();
    assert_eq!(r.map_base, 0.0);
    assert_eq!(r.counts.cls, 1);
    assert_eq!(r.counts.loc, 0);
    assert_eq!(r.counts.both, 0);
    assert_eq!(r.counts.dup, 0);
    assert_eq!(r.counts.bkg, 0);
    assert_eq!(r.counts.miss, 0, "GT explained by Cls error is not counted as unexplained Miss");
    assert_eq!(r.counts.fn_count, 1);
    assert!((r.impacts.cls - 1.0).abs() < 1e-12, "fixing Cls must recover full 1.0 AP");

    // 4. Pure Loc (same class, IoU = 40/160 = 0.25 in [0.10, 0.50)):
    let loc_err = vec![(vec![det(6.0, 0.0, 16.0, 10.0, 0.9, 0)], vec![gt(0.0, 0.0, 10.0, 10.0, 0)])];
    let r = compute_tide_diagnostics(2, &loc_err, &cfg).unwrap();
    assert_eq!(r.counts.loc, 1);
    assert_eq!(r.counts.miss, 0, "GT explained by Loc error is not unexplained Miss");
    assert!((r.impacts.loc - 1.0).abs() < 1e-12, "fixing Loc must recover full 1.0 AP");

    // 5. Pure Both (wrong class AND IoU = 0.25 in [0.10, 0.50)) ranked before a TP of class 1:
    let both_err = vec![(
        vec![
            det(6.0, 0.0, 16.0, 10.0, 0.95, 1), // Both error (class 1 on class 0 GT with IoU 0.25)
            det(50.0, 50.0, 60.0, 60.0, 0.80, 1), // TP for class 1
            det(0.0, 0.0, 10.0, 10.0, 0.90, 0), // TP for class 0
        ],
        vec![gt(0.0, 0.0, 10.0, 10.0, 0), gt(50.0, 50.0, 60.0, 60.0, 1)],
    )];
    let r = compute_tide_diagnostics(2, &both_err, &cfg).unwrap();
    assert_eq!(r.counts.both, 1);
    assert_eq!(r.counts.fp, 1);
    assert!(r.impacts.both > 0.2, "removing high-scoring Both FP must raise class 1 AP");

    // 6. Pure Dup (duplicate detection on gt0 ranked above gt1's TP):
    let dup_err = vec![(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.95, 0),   // TP on gt0
            det(0.0, 0.0, 10.0, 10.0, 0.90, 0),   // Dup on gt0
            det(50.0, 50.0, 60.0, 60.0, 0.80, 0), // TP on gt1
        ],
        vec![gt(0.0, 0.0, 10.0, 10.0, 0), gt(50.0, 50.0, 60.0, 60.0, 0)],
    )];
    let r = compute_tide_diagnostics(1, &dup_err, &cfg).unwrap();
    assert_eq!(r.counts.tp, 2);
    assert_eq!(r.counts.dup, 1);
    assert_eq!(r.counts.fp, 1);
    assert!(r.impacts.dup > 0.1, "removing mid-ranked Dup must raise AP to 1.0");
    assert!((r.map_base + r.impacts.dup - 1.0).abs() < 1e-12);

    // 7. Pure Bkg (disjoint high-scoring FP before a TP):
    let bkg_err = vec![(
        vec![
            det(200.0, 200.0, 210.0, 210.0, 0.99, 0), // Bkg FP
            det(0.0, 0.0, 10.0, 10.0, 0.80, 0),       // TP
        ],
        vec![gt(0.0, 0.0, 10.0, 10.0, 0)],
    )];
    let r = compute_tide_diagnostics(1, &bkg_err, &cfg).unwrap();
    assert_eq!(r.counts.tp, 1);
    assert_eq!(r.counts.bkg, 1);
    assert!((r.map_base - 0.5).abs() < 1e-12);
    assert!((r.impacts.bkg - 0.5).abs() < 1e-12);
    assert!((r.impacts.fp - 0.5).abs() < 1e-12);
}

/// Multi-class mixed TIDE invariants, config validation, and serde round-trip.
#[test]
fn tide_invariants_validation_and_serde() {
    let cfg = TideConfig::default();
    let images = vec![(
        vec![
            det(0.0, 0.0, 10.0, 10.0, 0.95, 0),       // TP
            det(0.0, 0.0, 10.0, 10.0, 0.92, 0),       // Dup
            det(26.0, 0.0, 36.0, 10.0, 0.88, 0),      // Loc on gt(20..30)
            det(50.0, 50.0, 60.0, 60.0, 0.85, 0),     // Cls on class 1 gt(50..60)
            det(106.0, 100.0, 116.0, 110.0, 0.80, 0), // Both on class 1 gt(100..110)
            det(300.0, 300.0, 310.0, 310.0, 0.75, 1), // Bkg
            det(0.0, 0.0, 10.0, 10.0, f32::NAN, 0),   // Ignored NaN score
        ],
        vec![
            gt(0.0, 0.0, 10.0, 10.0, 0),
            gt(20.0, 0.0, 30.0, 10.0, 0),
            gt(50.0, 50.0, 60.0, 60.0, 1),
            gt(100.0, 100.0, 110.0, 110.0, 1),
            gt(150.0, 150.0, 160.0, 160.0, 1), // Miss
        ],
    )];
    let diag = compute_tide_diagnostics(2, &images, &cfg).unwrap();
    let c = diag.counts;
    assert_eq!(c.fp, c.cls + c.loc + c.both + c.dup + c.bkg);
    assert!(c.miss <= c.fn_count);
    assert_eq!((c.tp, c.cls, c.loc, c.both, c.dup, c.bkg), (1, 1, 1, 1, 1, 1));
    assert_eq!(c.miss, 2, "gt(100..110) with only Both FP + gt(150..160) with no pred");
    assert_eq!(c.fn_count, 4);

    // Every dAP impact is finite and non-negative.
    let im = diag.impacts;
    for v in [im.cls, im.loc, im.both, im.dup, im.bkg, im.miss, im.fp, im.fn_impact] {
        assert!(v.is_finite() && v >= 0.0, "dAP must be finite and >= 0, got {v}");
    }

    // Serde round-trip.
    let json = serde_json::to_string(&diag).unwrap();
    assert_eq!(serde_json::from_str::<TideDiagnostics>(&json).unwrap(), diag);

    // Validation rejects invalid thresholds.
    assert!(compute_tide_diagnostics(2, &images, &TideConfig { pos_iou: 0.1, bg_iou: 0.5, ..cfg }).is_err());
    assert!(compute_tide_diagnostics(2, &images, &TideConfig { pos_iou: 0.5, bg_iou: 0.0, ..cfg }).is_err());
    assert!(compute_tide_diagnostics(2, &images, &TideConfig { pos_iou: 1.1, bg_iou: 0.1, ..cfg }).is_err());
    assert!(compute_tide_diagnostics(1, &images, &cfg).is_err());
}
