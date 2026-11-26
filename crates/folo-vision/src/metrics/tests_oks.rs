//! Unit tests for COCO-style Object Keypoint Similarity (`oks`, `oks_f64`, `oks_matrix`,
//! `compute_oks_ap`): visibility `0/1/2` semantics, `COCO_KEYPOINT_SIGMAS`, hand-computed OKS,
//! scale invariance, keypoint AP@50/AP@75/AP@50:95, and validation.

use super::oks::{
    COCO_KEYPOINT_SIGMAS, Keypoint, KeypointVisibility, OksPred, OksTruth, compute_oks_ap, oks, oks_f64,
    oks_matrix,
};

#[test]
fn oks_visibility_exact_values_and_scale_invariance() {
    assert_eq!(COCO_KEYPOINT_SIGMAS.len(), 17);
    assert!(!KeypointVisibility::Unlabeled.is_labeled());
    assert!(KeypointVisibility::Occluded.is_labeled());
    assert!(KeypointVisibility::Visible.is_labeled());
    assert_eq!(KeypointVisibility::from_u8(2).unwrap(), KeypointVisibility::Visible);
    assert!(KeypointVisibility::from_u8(3).is_err());

    // 3 keypoints: v=2 exact match, v=1 offset by dx=10, v=0 huge error (must be ignored!).
    let sigmas = [0.05f32, 0.1, 0.08];
    let area = 10_000.0f64;
    let pred = [[10.0, 20.0], [40.0, 30.0], [9999.0, -9999.0]];
    let gt = [Keypoint::visible(10.0, 20.0), Keypoint::occluded(30.0, 30.0), Keypoint::unlabeled()];

    // kpt 0: d=0 -> exp(0) = 1.0
    // kpt 1: d^2=100, denom = 8 * 10000 * sigma1^2 (sigma1 stored as f32)
    // kpt 2: v=0 -> ignored
    let s1 = sigmas[1] as f64;
    let want = 0.5 * (1.0 + (-100.0 / (8.0 * area * s1 * s1)).exp());
    let got = oks_f64(&pred, &gt, area, &sigmas).unwrap();
    assert!((got - want).abs() < 1e-15, "got={got} vs want={want}");
    assert!((oks(&pred, &gt, area as f32, &sigmas).unwrap() as f64 - want).abs() < 1e-6);

    // Scale invariance: multiply coords by 4 and area by 16 -> exact same OKS.
    let pred_s = pred.map(|[x, y]| [x * 4.0, y * 4.0]);
    let gt_s = gt.map(|kp| Keypoint::new(kp.x * 4.0, kp.y * 4.0, kp.visibility));
    let got_s = oks_f64(&pred_s, &gt_s, area * 16.0, &sigmas).unwrap();
    assert!((got_s - want).abs() < 1e-14);

    // All unlabeled or zero area returns 0.0 (no NaN).
    let all_unlabeled = [Keypoint::unlabeled(); 3];
    assert_eq!(oks_f64(&pred, &all_unlabeled, area, &sigmas).unwrap(), 0.0);
    assert_eq!(oks_f64(&pred, &gt, 0.0, &sigmas).unwrap(), 0.0);

    // Exact match remains 1.0 at extreme upper (1e308) and lower (1e-300) finite areas without overflow.
    let exact_pred = [[10.0, 20.0], [30.0, 30.0], [0.0, 0.0]];
    assert_eq!(oks_f64(&exact_pred, &gt, 1e308, &[0.026, 0.026, 0.026]).unwrap(), 1.0);
    assert_eq!(oks_f64(&exact_pred, &gt, 1e-300, &[0.026, 0.026, 0.026]).unwrap(), 1.0);

    // Pairwise matrix matches scalar oks and validates even when one side is empty.
    let m = oks_matrix(&[&pred[..]], &[(&gt[..], area as f32), (&all_unlabeled[..], area as f32)], &sigmas)
        .unwrap();
    assert_eq!(m.len(), 2);
    assert!((m[0] as f64 - want).abs() < 1e-6);
    assert_eq!(m[1], 0.0);
    assert!(oks_matrix(&[&pred[..2]], &[], &sigmas).is_err());
    assert!(oks_matrix(&[], &[(&gt[..], -1.0f32)], &sigmas).is_err());

    // Validation catches mismatched lengths, bad sigmas, negative area, invalid visibility.
    assert!(oks_f64(&pred[..2], &gt, area, &sigmas).is_err());
    assert!(oks_f64(&pred, &gt, -1.0, &sigmas).is_err());
    assert!(oks_f64(&pred, &gt, area, &[0.05, 0.0, 0.08]).is_err());
    assert!(oks_f64(&pred, &[Keypoint::new(0.0, 0.0, 5), gt[1], gt[2]], area, &sigmas).is_err());
}

#[test]
fn compute_oks_ap_perfect_partial_and_ignored_truths() {
    let sigmas = [0.05f32, 0.05];
    let area = 10_000.0f32;
    // denom = 8 * 10000 * 0.0025 = 200.
    // d^2 = 0 -> OKS = 1.0 (matches all 10 thresholds 0.50..0.95).
    // d^2 = 80 (dx = sqrt(80) ≈ 8.94427) -> exp(-0.4) ≈ 0.67032 (matches 0.50, 0.55, 0.60, 0.65 -> 4 thresholds; misses 0.75!).
    let dx_mid = 80.0f32.sqrt();

    let img0_preds = vec![OksPred { keypoints: vec![[10.0, 10.0], [20.0, 20.0]], score: 0.95 }];
    let img0_truths = vec![
        OksTruth::new(vec![Keypoint::visible(10.0, 10.0), Keypoint::occluded(20.0, 20.0)], area),
        // Unlabeled truth without bbox is skipped and not counted in num_truths.
        OksTruth::new(vec![Keypoint::unlabeled(), Keypoint::unlabeled()], area),
    ];

    let img1_preds =
        vec![OksPred { keypoints: vec![[10.0 + dx_mid, 10.0], [20.0 + dx_mid, 20.0]], score: 0.85 }];
    let img1_truths =
        vec![OksTruth::new(vec![Keypoint::visible(10.0, 10.0), Keypoint::visible(20.0, 20.0)], area)];

    // Perfect single-image check.
    let rep_perf =
        compute_oks_ap(&[(img0_preds.as_slice(), img0_truths.as_slice())], &sigmas, Some(20)).unwrap();
    assert_eq!(rep_perf.num_truths, 1);
    assert_eq!(rep_perf.num_preds, 1);
    assert!((rep_perf.ap50 - 1.0).abs() < 1e-6);
    assert!((rep_perf.ap75 - 1.0).abs() < 1e-6);
    assert!((rep_perf.ap50_95 - 1.0).abs() < 1e-6);

    // COCO unlabeled-person ignore behavior: high-scoring pose inside an unlabeled person's bbox
    // followed by a lower-scoring perfect match to a labeled person must be ignored (AP == 1.0, not 0.5).
    let ign_preds = vec![
        OksPred { keypoints: vec![[210.0, 210.0], [220.0, 220.0]], score: 0.99 },
        OksPred { keypoints: vec![[10.0, 10.0], [20.0, 20.0]], score: 0.80 },
    ];
    let ign_truths = vec![
        OksTruth::new(vec![Keypoint::unlabeled(), Keypoint::unlabeled()], area)
            .with_bbox(crate::boxes::Xyxy::new(200.0, 200.0, 240.0, 240.0)),
        OksTruth::new(vec![Keypoint::visible(10.0, 10.0), Keypoint::visible(20.0, 20.0)], area),
    ];
    let rep_ign =
        compute_oks_ap(&[(ign_preds.as_slice(), ign_truths.as_slice())], &sigmas, Some(20)).unwrap();
    assert_eq!(rep_ign.num_truths, 1);
    assert!((rep_ign.ap50 - 1.0).abs() < 1e-6, "ignored unlabeled-bbox match must not count as FP");
    assert!((rep_ign.ap50_95 - 1.0).abs() < 1e-6);

    // Crowd truth absorbs multiple overlapping detections without turning the 2nd into an FP (AP == 1.0).
    let crowd_preds = vec![
        OksPred { keypoints: vec![[210.0, 210.0], [220.0, 220.0]], score: 0.99 },
        OksPred { keypoints: vec![[212.0, 212.0], [222.0, 222.0]], score: 0.95 },
        OksPred { keypoints: vec![[10.0, 10.0], [20.0, 20.0]], score: 0.80 },
    ];
    let crowd_truths = vec![
        OksTruth::new(vec![Keypoint::unlabeled(), Keypoint::unlabeled()], area)
            .with_bbox(crate::boxes::Xyxy::new(200.0, 200.0, 240.0, 240.0))
            .with_crowd(true),
        OksTruth::new(vec![Keypoint::visible(10.0, 10.0), Keypoint::visible(20.0, 20.0)], area),
    ];
    let rep_crowd =
        compute_oks_ap(&[(crowd_preds.as_slice(), crowd_truths.as_slice())], &sigmas, Some(20)).unwrap();
    assert_eq!(rep_crowd.num_truths, 1);
    assert!((rep_crowd.ap50 - 1.0).abs() < 1e-6, "crowd region must absorb multiple detections");
    assert!((rep_crowd.ap50_95 - 1.0).abs() < 1e-6);

    // Two images: both match at 0.50..0.65 (AP = 1.0), only img0 matches at 0.70..0.95 (recall = 0.5 -> 51/101).
    let rep = compute_oks_ap(
        &[(img0_preds.as_slice(), img0_truths.as_slice()), (img1_preds.as_slice(), img1_truths.as_slice())],
        &sigmas,
        Some(20),
    )
    .unwrap();
    assert_eq!(rep.num_truths, 2);
    assert_eq!(rep.num_preds, 2);
    assert!((rep.ap50 - 1.0).abs() < 1e-6);
    let want_half = 51.0 / 101.0;
    assert!((rep.ap75 as f64 - want_half).abs() < 1e-6);
    let want_mean = (4.0 * 1.0 + 6.0 * want_half) / 10.0;
    assert!((rep.ap50_95 as f64 - want_mean).abs() < 1e-6);
}
