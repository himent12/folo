//! Unit tests for exact Oriented Bounding Box (OBB) Sutherland–Hodgman polygon IoU and rotated NMS:
//! analytical rotated overlaps (45° octagon, 90° swap, perpendicular cross), axis-aligned parity
//! with `boxes::box_iou`, general polygon clipper cross-check, degenerate/collinear/non-finite/scale
//! extremes, bit-for-bit symmetry, and class-aware / class-agnostic rotated NMS.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use super::{
    Obb, ObbDetection, RotatedNmsConfig, batched_rotated_nms, clip_convex_polygon_f64,
    convex_polygon_area_f64, obb_intersection_area_f64, obb_iou, obb_iou_f64, obb_iou_matrix, rotated_nms,
    rotated_nms_detections,
};
use crate::boxes::{Xyxy, box_iou};

#[test]
fn obb_analytical_exact_overlaps_and_rotations() {
    // 1. Unit square vs 45°-rotated unit square at same center:
    //    Exact intersection = 2 * (sqrt(2) - 1), exact IoU = 1 / sqrt(2).
    let sq0 = Obb::new(10.0, 20.0, 4.0, 4.0, 0.0);
    let sq45 = Obb::new(10.0, 20.0, 4.0, 4.0, FRAC_PI_4);
    let want_inter = 16.0 * 2.0 * (2.0f64.sqrt() - 1.0);
    let want_iou = std::f64::consts::FRAC_1_SQRT_2;
    assert!((obb_intersection_area_f64(&sq0, &sq45) - want_inter).abs() < 1e-6);
    assert!((obb_iou_f64(&sq0, &sq45) - want_iou).abs() < 1e-6);
    assert_eq!(obb_iou_f64(&sq0, &sq45), obb_iou_f64(&sq45, &sq0), "must be bit-for-bit symmetric");

    // 2. Rectangle (w, h, 0) vs (h, w, pi/2) at same center -> identical up to f32 FRAC_PI_2 rounding (~4.4e-8 rad).
    let rect0 = Obb::new(5.0, -3.0, 12.0, 4.0, 0.0);
    let rect90 = Obb::new(5.0, -3.0, 4.0, 12.0, FRAC_PI_2);
    assert!((obb_iou_f64(&rect0, &rect90) - 1.0).abs() < 1e-6);
    assert!((obb_iou_f64(&rect0, &Obb::new(5.0, -3.0, 4.0, 12.0, -FRAC_PI_2)) - 1.0).abs() < 1e-6);

    // 3. Rectangle rotated by pi is identical.
    let tilted = Obb::new(1.5, 2.5, 8.0, 3.0, 0.73);
    let tilted_pi = Obb::new(1.5, 2.5, 8.0, 3.0, 0.73 + PI);
    assert!((obb_iou_f64(&tilted, &tilted_pi) - 1.0).abs() < 1e-6);
    assert_eq!(obb_iou_f64(&tilted, &tilted), 1.0);

    // 4. Perpendicular cross: (10 x 2, theta=0) vs (10 x 2, theta=pi/2) at origin.
    //    Intersection = 2 x 2 = 4, union = 20 + 20 - 4 = 36, IoU = 1/9.
    let horiz = Obb::new(0.0, 0.0, 10.0, 2.0, 0.0);
    let vert = Obb::new(0.0, 0.0, 10.0, 2.0, FRAC_PI_2);
    assert!((obb_intersection_area_f64(&horiz, &vert) - 4.0).abs() < 1e-6);
    assert!((obb_iou_f64(&horiz, &vert) - 1.0 / 9.0).abs() < 1e-6);

    // 5. Cross-check against general Sutherland-Hodgman clipper on arbitrary rotated pairs.
    let pairs = [
        (Obb::new(12.0, 7.0, 9.0, 4.0, 0.35), Obb::new(14.0, 8.0, 7.0, 5.0, 1.15)),
        (Obb::new(-5.0, 3.0, 6.0, 6.0, 0.9), Obb::new(-3.0, 4.0, 8.0, 2.0, -0.4)),
        (Obb::new(0.0, 0.0, 10.0, 3.0, 0.0), Obb::new(0.0, 0.0, 10.0, 3.0, 0.01)),
    ];
    for (a, b) in pairs {
        let poly = clip_convex_polygon_f64(&b.corners_f64(), &a.corners_f64());
        let ref_area = convex_polygon_area_f64(&poly);
        let fast_area = obb_intersection_area_f64(&a, &b);
        assert!((fast_area - ref_area).abs() < 1e-11, "fast={fast_area} vs ref={ref_area}");
        assert_eq!(obb_iou_f64(&a, &b), obb_iou_f64(&b, &a));
    }
}

#[test]
fn obb_axis_aligned_parity_collinear_degenerate_and_extreme_scales() {
    // Axis-aligned OBBs match `boxes::box_iou`.
    let xy_boxes = [
        Xyxy::new(0.0, 0.0, 10.0, 10.0),
        Xyxy::new(5.0, 0.0, 15.0, 10.0),
        Xyxy::new(10.0, 0.0, 20.0, 10.0), // touching collinear edge
        Xyxy::new(2.0, 2.0, 8.0, 8.0),    // strictly contained
        Xyxy::new(30.0, 30.0, 40.0, 40.0),
    ];
    let obbs: Vec<Obb> = xy_boxes.iter().copied().map(Obb::from_xyxy).collect();
    let mat = obb_iou_matrix(&obbs, &obbs);
    for (i, a) in xy_boxes.iter().enumerate() {
        for (j, b) in xy_boxes.iter().enumerate() {
            let want = box_iou(a, b);
            let got = mat[i * obbs.len() + j];
            assert!((got - want).abs() < 1e-6, "({i},{j}): obb={got} vs box_iou={want}");
        }
    }

    // Collinear touching rotated boxes (shared rotated edge -> 0 overlap).
    let theta = 0.6f32;
    let (s, c) = theta.sin_cos();
    let touch_a = Obb::new(0.0, 0.0, 4.0, 2.0, theta);
    let touch_b = Obb::new(4.0 * c, 4.0 * s, 4.0, 2.0, theta);
    assert!(obb_iou_f64(&touch_a, &touch_b) < 1e-12);

    // Degenerate zero-area, negative-extent, and non-finite boxes never produce NaN.
    let normal = Obb::new(0.0, 0.0, 4.0, 4.0, 0.5);
    let degenerates = [
        Obb::new(0.0, 0.0, 0.0, 4.0, 0.5),
        Obb::new(0.0, 0.0, 4.0, 0.0, 0.5),
        Obb::new(0.0, 0.0, 0.0, 0.0, 0.0),
        Obb::new(0.0, 0.0, -4.0, 4.0, 0.2),
        Obb::new(f32::NAN, 0.0, 4.0, 4.0, 0.0),
        Obb::new(0.0, 0.0, 4.0, 4.0, f32::INFINITY),
    ];
    for d in degenerates {
        assert_eq!(obb_iou(&d, &normal), 0.0);
        assert_eq!(obb_iou(&normal, &d), 0.0);
        assert_eq!(obb_iou(&d, &d), 0.0);
    }

    // Extreme scales (1e-25 and 1e25) and huge translation offsets stay accurate and finite.
    let tiny_a = Obb::new(0.0, 0.0, 2e-25, 2e-25, 0.0);
    let tiny_b = Obb::new(0.0, 0.0, 2e-25, 2e-25, FRAC_PI_4);
    assert!((obb_iou_f64(&tiny_a, &tiny_b) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);

    let huge_a = Obb::new(1e7, -1e7, 1e25, 1e25, 0.3);
    let huge_b = Obb::new(1e7, -1e7, 1e25, 1e25, 0.3 + FRAC_PI_4);
    assert!((obb_iou_f64(&huge_a, &huge_b) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn rotated_nms_class_aware_class_agnostic_and_cross_separation() {
    // Two long thin perpendicular boxes at the same center:
    // Their AABBs are identical [-5, -5, 5, 5] (AABB IoU = 1.0!), yet their exact rotated IoU is 1/9 = 0.111.
    // Rotated NMS at iou_thr = 0.5 MUST keep both!
    let b0 = Obb::new(0.0, 0.0, 10.0, 2.0, 0.0);
    let b1 = Obb::new(0.0, 0.0, 10.0, 2.0, FRAC_PI_2);
    // Nearly duplicate of b0 (high IoU > 0.8) in same class 0 -> suppressed.
    let b2 = Obb::new(0.2, 0.0, 10.0, 2.0, 0.02);
    // Exact duplicate of b0 in different class 1 -> kept under class-aware, suppressed under agnostic.
    let b3 = b0;
    // Non-finite score & below-threshold score -> dropped.
    let b4 = Obb::new(50.0, 50.0, 4.0, 4.0, 0.0);
    let b5 = Obb::new(60.0, 60.0, 4.0, 4.0, 0.0);

    let boxes = [b0, b1, b2, b3, b4, b5];
    let scores = [0.95f32, 0.90, 0.85, 0.80, f32::NAN, 0.10];
    let classes = [0u32, 0, 0, 1, 0, 0];

    // Class-aware: keeps b0 (0), b1 (1), b3 (3); drops b2 (suppressed by b0), b4 (NaN), b5 (<= conf).
    let cfg_aware = RotatedNmsConfig { conf: 0.25, iou: 0.5, max_det: 10, agnostic: false };
    assert_eq!(batched_rotated_nms(&boxes, &scores, &classes, &cfg_aware), vec![0, 1, 3]);

    // Class-agnostic: b3 is suppressed by b0 -> keeps only [0, 1].
    let cfg_agn = RotatedNmsConfig { agnostic: true, ..cfg_aware };
    assert_eq!(batched_rotated_nms(&boxes, &scores, &classes, &cfg_agn), vec![0, 1]);
    assert_eq!(rotated_nms(&boxes, &scores, 0.5, 0.25, 10), vec![0, 1]);

    // max_det cap respected, deterministic tie-breaking on equal scores.
    let tied_boxes = [
        Obb::new(0.0, 0.0, 4.0, 4.0, 0.0),
        Obb::new(20.0, 0.0, 4.0, 4.0, 0.0),
        Obb::new(40.0, 0.0, 4.0, 4.0, 0.0),
    ];
    let tied_scores = [0.8f32, 0.8, 0.8];
    assert_eq!(rotated_nms(&tied_boxes, &tied_scores, 0.5, 0.2, 2), vec![0, 1]);

    // `rotated_nms_detections` wrapper preserves order and fields.
    let dets: Vec<ObbDetection> = boxes
        .iter()
        .zip(scores)
        .zip(classes)
        .map(|((&obb, score), class)| ObbDetection { obb, score, class })
        .collect();
    let kept = rotated_nms_detections(&dets, &cfg_aware);
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[0].class, 0);
    assert_eq!(kept[1].class, 0);
    assert_eq!(kept[2].class, 1);
}
