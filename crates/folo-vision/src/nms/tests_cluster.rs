//! Unit tests for Fast-NMS and Cluster-NMS (with optional DIoU, Linear/Gaussian score decay, and
//! weighted-cluster box refinement): exact equivalence of hard Cluster-NMS with greedy NMS/DIoU-NMS,
//! classic Fast-NMS chain over-suppression contrast, score-decay formulas, weighted box refinement,
//! and edge-case resilience (empty, ties, NaN/Inf scores, non-finite/degenerate boxes, multi-class).

use super::cluster::{
    ClusterNmsConfig, ClusterScoreDecay, batched_cluster_nms, batched_fast_nms, cluster_nms,
    cluster_nms_detections, fast_nms, fast_nms_detections,
};
use super::{Detection, batched_nms, diou_nms_sorted, nms_sorted};
use crate::boxes::Xyxy;
use folo_core::rng::Rng;

/// Classic 3-box chain `A -> B -> C` where `IoU(A, B) > thr`, `IoU(B, C) > thr`, `IoU(A, C) == 0`:
/// - Greedy NMS keeps `[A, C]` because `A` suppresses `B`, so `B` cannot suppress `C`.
/// - Single-pass Fast-NMS keeps only `[A]` (classic Fast-NMS over-suppression).
/// - Converged hard Cluster-NMS recovers `[A, C]` identically to greedy NMS.
/// - Cluster-NMS with `max_iters: Some(1)` matches Fast-NMS (`[A]`).
#[test]
fn chain_of_three_contrasts_fast_nms_and_cluster_nms() {
    let a = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let b = Xyxy::new(4.0, 0.0, 14.0, 10.0); // IoU(a, b) = 60/140 ≈ 0.4286
    let c = Xyxy::new(8.0, 0.0, 18.0, 10.0); // IoU(b, c) ≈ 0.4286, IoU(a, c) = 20/180 ≈ 0.111
    let boxes = [a, b, c];
    let scores = [0.9f32, 0.8, 0.7];
    let thr = 0.4f32;

    let greedy = nms_sorted(&boxes, thr, 10);
    assert_eq!(greedy, vec![0, 2]);

    let fast = fast_nms(&boxes, &scores, thr, 0.0, 10);
    assert_eq!(fast, vec![0], "Fast-NMS suppresses C via already-suppressed B");

    let cluster =
        cluster_nms(&boxes, &scores, &ClusterNmsConfig { iou_thr: thr, ..ClusterNmsConfig::default() });
    let cluster_idx: Vec<usize> = cluster.iter().map(|e| e.index).collect();
    assert_eq!(cluster_idx, greedy, "converged Cluster-NMS must match greedy NMS");

    let cluster_1iter = cluster_nms(
        &boxes,
        &scores,
        &ClusterNmsConfig { iou_thr: thr, max_iters: Some(1), ..ClusterNmsConfig::default() },
    );
    let c1_idx: Vec<usize> = cluster_1iter.iter().map(|e| e.index).collect();
    assert_eq!(c1_idx, fast, "1-iteration Cluster-NMS must match Fast-NMS");
}

/// Hard Cluster-NMS matches greedy `batched_nms` and `diou_nms_sorted` bit-for-bit across
/// deterministic seeded random multi-class boxes, thresholds, and agnostic/class-aware modes.
#[test]
fn cluster_nms_exact_parity_with_greedy_and_diou_nms() {
    let mut rng = Rng::seed(2026);
    let n = 80;
    let mut boxes = Vec::with_capacity(n);
    let mut scores = Vec::with_capacity(n);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = rng.next_f32() * 40.0;
        let y1 = rng.next_f32() * 40.0;
        let w = 4.0 + rng.next_f32() * 18.0;
        let h = 4.0 + rng.next_f32() * 18.0;
        boxes.push(Xyxy::new(x1, y1, x1 + w, y1 + h));
        // Include occasional exact score ties.
        scores.push((rng.next_f32() * 20.0).round() / 20.0 + 0.05);
        classes.push((rng.next_u64() % 4) as u32);
    }

    for &thr in &[0.3f32, 0.5, 0.75] {
        for &agnostic in &[false, true] {
            let want = batched_nms(&boxes, &scores, &classes, thr, 50, agnostic);
            let cfg = ClusterNmsConfig {
                iou_thr: thr,
                score_thr: 0.0,
                max_det: 50,
                agnostic,
                ..ClusterNmsConfig::default()
            };
            let got: Vec<usize> =
                batched_cluster_nms(&boxes, &scores, &classes, &cfg).into_iter().map(|e| e.index).collect();
            assert_eq!(got, want, "Cluster-NMS vs greedy batched_nms (thr={thr}, agnostic={agnostic})");
        }
    }

    // DIoU Cluster-NMS parity against `diou_nms_sorted` on pre-sorted boxes.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    let s_boxes: Vec<Xyxy> = order.iter().map(|&i| boxes[i]).collect();
    let s_scores: Vec<f32> = order.iter().map(|&i| scores[i]).collect();
    let want_diou = diou_nms_sorted(&s_boxes, 0.45, 50);
    let got_diou: Vec<usize> = cluster_nms(
        &s_boxes,
        &s_scores,
        &ClusterNmsConfig { iou_thr: 0.45, max_det: 50, use_diou: true, ..ClusterNmsConfig::default() },
    )
    .into_iter()
    .map(|e| e.index)
    .collect();
    assert_eq!(got_diou, want_diou, "DIoU Cluster-NMS must match greedy DIoU-NMS");
}

/// Verify Cluster-NMS Linear & Gaussian score-decay formulas and weighted-cluster box refinement.
#[test]
fn cluster_nms_score_decay_and_weighted_box_refinement() {
    let b0 = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let b1 = Xyxy::new(2.0, 0.0, 12.0, 10.0); // IoU(b0, b1) = 80 / 120 = 2/3
    let b2 = Xyxy::new(30.0, 30.0, 40.0, 40.0); // disjoint
    let boxes = [b0, b1, b2];
    let scores = [0.9f32, 0.6, 0.5];

    // Linear score decay: b1 decayed by (1 - 2/3) = 1/3 -> 0.2.
    let lin = cluster_nms(
        &boxes,
        &scores,
        &ClusterNmsConfig {
            iou_thr: 0.5,
            score_thr: 0.1,
            score_decay: ClusterScoreDecay::Linear,
            ..ClusterNmsConfig::default()
        },
    );
    assert_eq!(lin.len(), 3);
    assert_eq!(lin[0].index, 0);
    assert!((lin[0].score - 0.9).abs() < 1e-6);
    assert_eq!(lin[1].index, 2, "disjoint b2 (0.5) must outrank decayed b1 (0.2)");
    assert!((lin[1].score - 0.5).abs() < 1e-6);
    assert_eq!(lin[2].index, 1);
    assert!((lin[2].score - 0.2).abs() < 1e-6);

    // Gaussian SPM score decay: b1 decayed by exp(-(2/3)² / 0.5).
    let sigma = 0.5f32;
    let gau = cluster_nms(
        &boxes,
        &scores,
        &ClusterNmsConfig {
            iou_thr: 0.5,
            score_thr: 0.0,
            score_decay: ClusterScoreDecay::Gaussian { sigma },
            ..ClusterNmsConfig::default()
        },
    );
    let want_b1 = (0.6f64 * (-(4.0f64 / 9.0) / 0.5).exp()) as f32;
    let b1_entry = gau.iter().find(|e| e.index == 1).unwrap();
    assert!((b1_entry.score - want_b1).abs() < 1e-6, "got {} vs {want_b1}", b1_entry.score);

    // Weighted-cluster box refinement:
    // b0 has weight 0.9 * 1.0 = 0.9; b1 (suppressed by b0) has weight 0.6 * (2/3) = 0.4.
    // Refined x1 = (0.9 * 0 + 0.4 * 2) / 1.3 = 0.8 / 1.3.
    // Disjoint b2 is alone in its cluster and stays bit-identical.
    let ref_out = cluster_nms(
        &boxes,
        &scores,
        &ClusterNmsConfig { iou_thr: 0.5, weighted_box: true, ..ClusterNmsConfig::default() },
    );
    assert_eq!(ref_out.len(), 2);
    assert_eq!(ref_out[0].index, 0);
    let want_x1 = (0.8f64 / 1.3) as f32;
    let want_x2 = ((0.9 * 10.0 + 0.4 * 12.0) / 1.3) as f32;
    assert!((ref_out[0].bbox.x1 - want_x1).abs() < 1e-5);
    assert!((ref_out[0].bbox.x2 - want_x2).abs() < 1e-5);
    assert_eq!(ref_out[0].bbox.y1, 0.0);
    assert_eq!(ref_out[0].bbox.y2, 10.0);
    assert_eq!(ref_out[1].index, 2);
    assert_eq!(ref_out[1].bbox, b2);
}

/// Edge cases: empty inputs, max_det=0, exact score ties, NaN/Inf scores, non-finite/inverted
/// boxes, strict multi-class isolation vs agnostic, and serde round-trip.
#[test]
fn fast_and_cluster_nms_edge_cases_and_multiclass() {
    let cfg = ClusterNmsConfig::default();

    // Empty inputs and max_det == 0.
    assert!(fast_nms(&[], &[], 0.5, 0.0, 10).is_empty());
    assert!(batched_fast_nms(&[], &[], &[], 0.5, 0.0, 10, false).is_empty());
    assert!(cluster_nms(&[], &[], &cfg).is_empty());
    assert!(batched_cluster_nms(&[], &[], &[], &cfg).is_empty());

    let b = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    assert!(fast_nms(&[b], &[0.9], 0.5, 0.0, 0).is_empty());
    assert!(cluster_nms(&[b], &[0.9], &ClusterNmsConfig { max_det: 0, ..cfg }).is_empty());

    // Exact score ties keep the lower input index.
    assert_eq!(fast_nms(&[b, b, b], &[0.8, 0.8, 0.8], 0.5, 0.0, 10), vec![0]);
    let tie_c = cluster_nms(&[b, b, b], &[0.8, 0.8, 0.8], &cfg);
    assert_eq!(tie_c.len(), 1);
    assert_eq!(tie_c[0].index, 0);

    // NaN and ±Inf scores are dropped cleanly and never suppress valid detections.
    let messy_scores = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.75];
    let messy_boxes = [b, b, b, b];
    assert_eq!(fast_nms(&messy_boxes, &messy_scores, 0.5, 0.0, 10), vec![3]);
    let c_messy = cluster_nms(&messy_boxes, &messy_scores, &cfg);
    assert_eq!(c_messy.len(), 1);
    assert_eq!(c_messy[0].index, 3);

    // Non-finite boxes are filtered during candidate selection and never consume max_det.
    let nan_box = Xyxy::new(f32::NAN, 0.0, 10.0, 10.0);
    let inv_box = Xyxy::new(10.0, 10.0, 2.0, 2.0);
    let out_nf = cluster_nms(
        &[b, nan_box, inv_box],
        &[0.9, 0.8, 0.7],
        &ClusterNmsConfig { iou_thr: 0.5, weighted_box: true, ..cfg },
    );
    assert_eq!(out_nf[0].bbox, b);

    let nan_first =
        vec![Detection { bbox: nan_box, score: 0.99, class: 0 }, Detection { bbox: b, score: 0.8, class: 0 }];
    let f_cap1 = fast_nms_detections(&nan_first, 0.5, 0.0, 1, false);
    assert_eq!(f_cap1.len(), 1);
    assert_eq!(f_cap1[0].bbox, b, "NaN box must not displace valid box at max_det=1 in Fast-NMS");
    let c_cap1 = cluster_nms_detections(&nan_first, &ClusterNmsConfig { iou_thr: 0.5, max_det: 1, ..cfg });
    assert_eq!(c_cap1.len(), 1);
    assert_eq!(c_cap1[0].bbox, b, "NaN box must not displace valid box at max_det=1 in Cluster-NMS");

    // Negative iou_thr: rank-0 candidate still survives in fast_nms (matching 1-iter cluster_nms).
    assert_eq!(fast_nms(&[b, Xyxy::new(50.0, 50.0, 60.0, 60.0)], &[0.9, 0.8], -0.1, 0.0, 10), vec![0]);

    // DIoU-weighted refinement must only assign a suppressed box to a center that actually suppressed it.
    // b0=[0,0,10,1] (0.9), b1=[1.3,0,4.2,1] (0.8, alive), b2=[0,0,5.5,1] (0.7):
    // IoU(b0,b2)=0.55 but DIoU(b0,b2)≈0.4999 <= 0.50 (b0 does NOT suppress b2),
    // while DIoU(b1,b2)≈0.5273 > 0.50 (b1 DOES suppress b2). Thus b0 stays untouched and b1 refines with b2.
    let d0 = Xyxy::new(0.0, 0.0, 10.0, 1.0);
    let d1 = Xyxy::new(1.3, 0.0, 4.2, 1.0);
    let d2 = Xyxy::new(0.0, 0.0, 5.5, 1.0);
    let diou_ref = cluster_nms(
        &[d0, d1, d2],
        &[0.9, 0.8, 0.7],
        &ClusterNmsConfig { iou_thr: 0.5, use_diou: true, weighted_box: true, ..cfg },
    );
    assert_eq!(diou_ref.len(), 2);
    assert_eq!(diou_ref[0].index, 0);
    assert_eq!(diou_ref[0].bbox, d0, "b0 did not suppress b2 under DIoU and must stay unchanged");
    assert_eq!(diou_ref[1].index, 1);
    assert_ne!(diou_ref[1].bbox, d1, "b1 suppressed b2 under DIoU and must refine toward b2");

    // Multi-class isolation vs class-agnostic via detection helpers.
    let dets = vec![
        Detection { bbox: b, score: 0.9, class: 0 },
        Detection { bbox: b, score: 0.85, class: 1 },
        Detection { bbox: b, score: 0.7, class: 0 },
    ];
    let f_cls = fast_nms_detections(&dets, 0.5, 0.0, 10, false);
    assert_eq!(f_cls.len(), 2);
    assert_eq!((f_cls[0].class, f_cls[1].class), (0, 1));
    let f_ag = fast_nms_detections(&dets, 0.5, 0.0, 10, true);
    assert_eq!(f_ag.len(), 1);
    assert_eq!(f_ag[0].class, 0);

    let c_cls = cluster_nms_detections(&dets, &ClusterNmsConfig { iou_thr: 0.5, ..cfg });
    assert_eq!(c_cls.len(), 2);
    assert_eq!((c_cls[0].class, c_cls[1].class), (0, 1));
    let c_ag = cluster_nms_detections(&dets, &ClusterNmsConfig { iou_thr: 0.5, agnostic: true, ..cfg });
    assert_eq!(c_ag.len(), 1);
    assert_eq!(c_ag[0].class, 0);

    // Serde round-trip for ClusterNmsConfig.
    let rich = ClusterNmsConfig {
        iou_thr: 0.6,
        score_thr: 0.05,
        max_det: 100,
        agnostic: true,
        use_diou: true,
        score_decay: ClusterScoreDecay::Gaussian { sigma: 0.4 },
        weighted_box: true,
        max_iters: Some(5),
    };
    let json = serde_json::to_string(&rich).unwrap();
    assert_eq!(serde_json::from_str::<ClusterNmsConfig>(&json).unwrap(), rich);
}
