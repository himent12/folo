//! Unit tests for SimOTA dynamic-`k` label assignment:
//! hand-computed dynamic-`k` calculation, clamping to `[1, num_candidates]`, in-box / center-prior
//! gating and `non_prior_cost` preference, multi-GT conflict resolution with deterministic tie-breaking,
//! empty GTs / empty anchors, degenerate GT & predicted boxes, non-finite safety, config/shape validation,
//! and end-to-end integration with standalone VFL / QFL.

use folo_core::Tensor;

use super::{SimOtaConfig, assign_simota};
use crate::boxes::Xyxy;
use crate::decode::make_anchors;
use crate::loss::{LossReduction, QflConfig, VflConfig, quality_focal_loss, varifocal_loss};

/// Dynamic `k` equals `floor(sum of top-q candidate IoUs)` clamped to `[1, num_candidates]`, and
/// selects the `k` lowest-cost anchors for the GT.
#[test]
fn dynamic_k_computation_and_clamping() {
    // 4 anchors all strictly inside GT [0, 0, 10, 10] (area = 100).
    let gt = [Xyxy::new(0.0, 0.0, 10.0, 10.0)];
    let gt_labels = [0u32];
    let pts = [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]];
    let strides = [4.0f32; 4];

    // Predicted boxes with exact known IoUs against [0, 0, 10, 10]:
    // a0: [0,0,10,10] -> IoU = 1.00
    // a1: [0,0,9,10]  -> IoU = 0.90
    // a2: [0,0,8,10]  -> IoU = 0.80
    // a3: [0,0,2,10]  -> IoU = 0.20
    // Sum of top-4 IoUs = 2.90 -> floor = 2; with top_q = 2: 1.00 + 0.90 = 1.90 -> floor = 1.
    let preds = [
        Xyxy::new(0.0, 0.0, 10.0, 10.0),
        Xyxy::new(0.0, 0.0, 9.0, 10.0),
        Xyxy::new(0.0, 0.0, 8.0, 10.0),
        Xyxy::new(0.0, 0.0, 2.0, 10.0),
    ];
    // Equal cls logits so cost order matches descending IoU order (a0 < a1 < a2 < a3).
    let logits = [2.0f32, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0];

    let cfg = SimOtaConfig { top_q: 4, ..SimOtaConfig::default() };
    let asg = assign_simota(&logits, 2, &preds, &pts, &strides, &gt, &gt_labels, &cfg).unwrap();
    assert!(asg.inputs_finite);
    assert_eq!(asg.dynamic_k, vec![2]);
    assert_eq!(asg.num_fg, 2);
    assert_eq!(asg.fg_mask, vec![true, true, false, false]);
    assert!((asg.target_ious[0] - 1.0).abs() < 1e-6);
    assert!((asg.target_ious[1] - 0.9).abs() < 1e-6);

    // With top_q = 2: top-2 sum = 1.90 -> dynamic_k = 1 (only a0).
    let cfg_q2 = SimOtaConfig { top_q: 2, ..SimOtaConfig::default() };
    let asg_q2 = assign_simota(&logits, 2, &preds, &pts, &strides, &gt, &gt_labels, &cfg_q2).unwrap();
    assert_eq!(asg_q2.dynamic_k, vec![1]);
    assert_eq!(asg_q2.fg_mask, vec![true, false, false, false]);

    // All disjoint predictions -> sum of IoUs = 0.0, clamped cleanly up to k = 1.
    let disjoint = [Xyxy::new(50.0, 50.0, 60.0, 60.0); 4];
    let asg_min = assign_simota(&logits, 2, &disjoint, &pts, &strides, &gt, &gt_labels, &cfg).unwrap();
    assert_eq!(asg_min.dynamic_k, vec![1]);
    assert_eq!(asg_min.num_fg, 1);
    // All costs equal -> deterministic tie-break selects anchor 0.
    assert_eq!(asg_min.fg_mask, vec![true, false, false, false]);

    // High IoUs (all 1.0 -> sum = 4.0) with max_k = Some(3) clamps cleanly to 3.
    let all_perfect = [Xyxy::new(0.0, 0.0, 10.0, 10.0); 4];
    let cfg_cap = SimOtaConfig { top_q: 10, max_k: Some(3), ..SimOtaConfig::default() };
    let asg_cap = assign_simota(&logits, 2, &all_perfect, &pts, &strides, &gt, &gt_labels, &cfg_cap).unwrap();
    assert_eq!(asg_cap.dynamic_k, vec![3]);
    assert_eq!(asg_cap.num_fg, 3);
    assert_eq!(asg_cap.fg_mask, vec![true, true, true, false]);
}

/// Center-prior and in-box gating: anchors inside both `in_box && in_center` are strictly preferred
/// over anchors in only one of them (via `non_prior_cost`), and anchors outside both are excluded.
#[test]
fn center_prior_and_in_box_masking() {
    // Large GT box [0, 0, 100, 100], center = (50, 50), stride = 4, center_radius = 2.5 -> half = 10 px -> [40, 60]^2.
    let gt = [Xyxy::new(0.0, 0.0, 100.0, 100.0)];
    let gt_labels = [0u32];
    let pts = [
        [50.0, 50.0],   // a0: in box AND in center (preferred)
        [10.0, 10.0],   // a1: in box ONLY (penalised by non_prior_cost)
        [150.0, 150.0], // a2: outside both (not a candidate)
    ];
    let strides = [4.0f32; 3];
    // Give a1 a better predicted box (IoU = 1.0) than a0 (IoU = 0.5), and give a2 IoU = 1.0.
    // Because non_prior_cost = 1e5 dominates the small IoU gap, a0 (in both) must win k = 1!
    let preds = [
        Xyxy::new(0.0, 0.0, 50.0, 100.0),  // IoU = 0.5
        Xyxy::new(0.0, 0.0, 100.0, 100.0), // IoU = 1.0
        Xyxy::new(0.0, 0.0, 100.0, 100.0), // IoU = 1.0 (outside candidate mask)
    ];
    let logits = [1.0f32, 1.0, 1.0];
    let cfg = SimOtaConfig { top_q: 10, max_k: Some(1), ..SimOtaConfig::default() };
    let asg = assign_simota(&logits, 1, &preds, &pts, &strides, &gt, &gt_labels, &cfg).unwrap();
    assert_eq!(asg.fg_mask, vec![true, false, false]);
    assert_eq!(asg.target_gt[0], 0);

    // Without center_radius (in_box only): a0 and a1 have no prior penalty, so higher-IoU a1 wins k = 1.
    let cfg_box_only =
        SimOtaConfig { center_radius: None, in_box: true, max_k: Some(1), ..SimOtaConfig::default() };
    let asg_box = assign_simota(&logits, 1, &preds, &pts, &strides, &gt, &gt_labels, &cfg_box_only).unwrap();
    assert_eq!(asg_box.fg_mask, vec![false, true, false]);

    // With neither prior (global candidates): a1 and a2 tie at IoU = 1.0 -> smaller anchor index a1 wins.
    let cfg_global =
        SimOtaConfig { center_radius: None, in_box: false, max_k: Some(1), ..SimOtaConfig::default() };
    let asg_glob = assign_simota(&logits, 1, &preds, &pts, &strides, &gt, &gt_labels, &cfg_global).unwrap();
    assert_eq!(asg_glob.fg_mask, vec![false, true, false]);

    // Tiny GT box [49, 49, 51, 51] containing NO anchor strictly inside it, but center-prior catches [52, 52].
    let tiny_gt = [Xyxy::new(49.0, 49.0, 51.0, 51.0)];
    let tiny_pts = [[52.0, 52.0]];
    let tiny_preds = [Xyxy::new(48.0, 48.0, 52.0, 52.0)];
    let tiny_asg =
        assign_simota(&[1.0], 1, &tiny_preds, &tiny_pts, &[4.0], &tiny_gt, &[0], &SimOtaConfig::default())
            .unwrap();
    assert_eq!(tiny_asg.num_fg, 1);
    // Disabling center_radius leaves the tiny GT with 0 in-box candidates -> 0 matches.
    let tiny_no_center = assign_simota(
        &[1.0],
        1,
        &tiny_preds,
        &tiny_pts,
        &[4.0],
        &tiny_gt,
        &[0],
        &SimOtaConfig { center_radius: None, ..SimOtaConfig::default() },
    )
    .unwrap();
    assert_eq!(tiny_no_center.num_fg, 0);
    assert_eq!(tiny_no_center.dynamic_k, vec![0]);
}

/// Conflict resolution assigns contested anchors to the lowest-cost GT, breaking exact ties
/// deterministically in favor of the smaller GT index.
#[test]
fn conflict_resolution_lowest_cost_and_deterministic_tie_break() {
    // Two overlapping GTs of different classes sharing anchor 0 at (15, 15).
    let gts = [Xyxy::new(0.0, 0.0, 30.0, 30.0), Xyxy::new(5.0, 5.0, 35.0, 35.0)];
    let labels = [0u32, 1u32];
    let pts = [[15.0, 15.0], [28.0, 28.0]];
    let strides = [8.0f32; 2];
    let preds = [Xyxy::new(5.0, 5.0, 35.0, 35.0), Xyxy::new(5.0, 5.0, 35.0, 35.0)];
    // Anchor 0 strongly predicts class 1 (logit [-3, +4]) and matches GT 1 with IoU = 1.0,
    // so GT 1 has strictly lower cost at anchor 0 and must win the conflict!
    let logits = [-3.0f32, 4.0, -3.0, 4.0];
    let asg =
        assign_simota(&logits, 2, &preds, &pts, &strides, &gts, &labels, &SimOtaConfig::default()).unwrap();
    assert!(asg.fg_mask[0]);
    assert_eq!(asg.target_gt[0], 1);
    assert_eq!(asg.target_labels[0], 1);
    assert!((asg.target_ious[0] - 1.0).abs() < 1e-6);

    // Identical duplicate GTs (exact cost tie) must deterministically pick GT 0.
    let dup_gts = [Xyxy::new(0.0, 0.0, 30.0, 30.0), Xyxy::new(0.0, 0.0, 30.0, 30.0)];
    let dup_labels = [0u32, 0u32];
    let tie_asg = assign_simota(
        &[1.0, -1.0],
        2,
        &[Xyxy::new(0.0, 0.0, 30.0, 30.0)],
        &[[15.0, 15.0]],
        &[8.0],
        &dup_gts,
        &dup_labels,
        &SimOtaConfig::default(),
    )
    .unwrap();
    assert_eq!(tie_asg.num_fg, 1);
    assert_eq!(tie_asg.target_gt[0], 0);

    // Three-GT contesting-only conflict resolution regression: GT 0 and GT 1 both select anchor 0 (k = 1 each)
    // while GT 2 selects anchor 1 (k = 1). Even though GT 2 has a slightly lower cost at anchor 0 than GT 0/1,
    // GT 2 did not select anchor 0, so conflict resolution stays strictly among contesting GTs (GT 0 wins tie).
    let gts3 =
        [Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(2.0, 0.0, 12.0, 10.0)];
    let preds3 = [Xyxy::new(1.5, 0.0, 11.5, 10.0), Xyxy::new(2.0, 0.0, 12.0, 10.0)];
    let pts3 = [[5.0, 5.0], [6.0, 5.0]];
    let asg3 = assign_simota(
        &[1.0, 1.0],
        1,
        &preds3,
        &pts3,
        &[4.0, 4.0],
        &gts3,
        &[0, 0, 0],
        &SimOtaConfig { max_k: Some(1), ..SimOtaConfig::default() },
    )
    .unwrap();
    assert_eq!(asg3.fg_mask, vec![true, true]);
    assert_eq!(asg3.target_gt, vec![0, 2]);
}

/// Edge cases: 0 GTs, 0 anchors, degenerate GT & predicted boxes, non-finite inputs, extreme logits,
/// config/shape validation, and end-to-end feeding of SimOTA quality targets into VFL and QFL.
#[test]
fn edge_cases_degenerate_non_finite_and_vfl_qfl_pipeline() {
    let cfg = SimOtaConfig::default();

    // 1. 0 GTs.
    let asg0g = assign_simota(
        &[0.5, -0.5],
        2,
        &[Xyxy::new(0.0, 0.0, 8.0, 8.0)],
        &[[4.0, 4.0]],
        &[8.0],
        &[],
        &[],
        &cfg,
    )
    .unwrap();
    assert!(asg0g.inputs_finite && asg0g.num_fg == 0 && asg0g.fg_mask == vec![false]);
    assert!(asg0g.dynamic_k.is_empty());

    // 2. 0 anchors.
    let asg0a = assign_simota(&[], 2, &[], &[], &[], &[Xyxy::new(0.0, 0.0, 8.0, 8.0)], &[0], &cfg).unwrap();
    assert!(asg0a.inputs_finite && asg0a.num_fg == 0 && asg0a.fg_mask.is_empty());
    assert_eq!(asg0a.dynamic_k, vec![0]);

    // 3. Degenerate GT boxes (zero-area and inverted) are skipped cleanly while valid GTs still assign;
    //    degenerate predicted boxes get IoU = 0.0 without NaN.
    let gts = [
        Xyxy::new(10.0, 10.0, 10.0, 20.0), // zero width
        Xyxy::new(20.0, 20.0, 10.0, 10.0), // inverted
        Xyxy::new(0.0, 0.0, 20.0, 20.0),   // valid
    ];
    let preds = [Xyxy::new(15.0, 15.0, 5.0, 5.0)]; // inverted predicted box -> IoU = 0
    let asg_deg =
        assign_simota(&[0.0, 1.0], 2, &preds, &[[10.0, 10.0]], &[8.0], &gts, &[0, 1, 1], &cfg).unwrap();
    assert!(asg_deg.inputs_finite);
    assert_eq!(asg_deg.dynamic_k, vec![0, 0, 1]);
    assert_eq!(asg_deg.num_fg, 1);
    assert_eq!(asg_deg.target_gt[0], 2);
    assert_eq!(asg_deg.target_ious[0], 0.0);
    assert!(asg_deg.matched_costs[0].is_finite());

    // 4. Extreme logits ±50 and ±f32::MAX stay finite in f64 matched_costs and match properly.
    let asg_ext = assign_simota(
        &[-50.0, 50.0],
        2,
        &[Xyxy::new(0.0, 0.0, 20.0, 20.0)],
        &[[10.0, 10.0]],
        &[8.0],
        &[Xyxy::new(0.0, 0.0, 20.0, 20.0)],
        &[1],
        &cfg,
    )
    .unwrap();
    assert!(asg_ext.inputs_finite && asg_ext.num_fg == 1 && asg_ext.matched_costs[0].is_finite());

    let asg_max = assign_simota(
        &[-f32::MAX, f32::MAX],
        2,
        &[Xyxy::new(0.0, 0.0, 20.0, 20.0)],
        &[[10.0, 10.0]],
        &[8.0],
        &[Xyxy::new(0.0, 0.0, 20.0, 20.0)],
        &[0],
        &cfg,
    )
    .unwrap();
    assert!(asg_max.inputs_finite && asg_max.num_fg == 1 && asg_max.matched_costs[0].is_finite());
    assert!(asg_max.matched_costs[0] > 6.8e38);

    // 5. Non-finite logits, pred boxes, anchor points, or GT boxes return `inputs_finite = false` and 0 fg.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let a_logit = assign_simota(
            &[bad, 0.0],
            2,
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[[5.0, 5.0]],
            &[8.0],
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[0],
            &cfg,
        )
        .unwrap();
        assert!(!a_logit.inputs_finite && a_logit.num_fg == 0);

        let a_pred = assign_simota(
            &[0.0, 0.0],
            2,
            &[Xyxy::new(bad, 0.0, 10.0, 10.0)],
            &[[5.0, 5.0]],
            &[8.0],
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[0],
            &cfg,
        )
        .unwrap();
        assert!(!a_pred.inputs_finite && a_pred.num_fg == 0);

        let a_pt = assign_simota(
            &[0.0, 0.0],
            2,
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[[bad, 5.0]],
            &[8.0],
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[0],
            &cfg,
        )
        .unwrap();
        assert!(!a_pt.inputs_finite && a_pt.num_fg == 0);

        let a_gt = assign_simota(
            &[0.0, 0.0],
            2,
            &[Xyxy::new(0.0, 0.0, 10.0, 10.0)],
            &[[5.0, 5.0]],
            &[8.0],
            &[Xyxy::new(0.0, bad, 10.0, 10.0)],
            &[0],
            &cfg,
        )
        .unwrap();
        assert!(!a_gt.inputs_finite && a_gt.num_fg == 0);
    }

    // 6. Config & structural validation errors (never panics).
    for bad_cfg in [
        SimOtaConfig { cls_weight: -1.0, ..cfg },
        SimOtaConfig { iou_weight: -0.1, ..cfg },
        SimOtaConfig { top_q: 0, ..cfg },
        SimOtaConfig { center_radius: Some(0.0), ..cfg },
        SimOtaConfig { center_radius: Some(-1.0), ..cfg },
        SimOtaConfig { non_prior_cost: -1.0, ..cfg },
        SimOtaConfig { eps: -0.01, ..cfg },
        SimOtaConfig { iou_eps: 0.0, ..cfg },
        SimOtaConfig { max_k: Some(0), ..cfg },
        SimOtaConfig { cls_weight: f32::NAN, ..cfg },
    ] {
        assert!(bad_cfg.validate().is_err());
        assert!(assign_simota(&[], 1, &[], &[], &[], &[], &[], &bad_cfg).is_err());
    }
    assert!(assign_simota(&[0.0], 2, &[Xyxy::default()], &[[0.0, 0.0]], &[8.0], &[], &[], &cfg).is_err());
    assert!(assign_simota(&[0.0, 0.0], 2, &[Xyxy::default()], &[], &[8.0], &[], &[], &cfg).is_err());
    assert!(
        assign_simota(&[0.0, 0.0], 2, &[Xyxy::default()], &[[0.0, 0.0]], &[0.0], &[], &[], &cfg).is_err()
    );
    assert!(
        assign_simota(
            &[0.0, 0.0],
            2,
            &[Xyxy::default()],
            &[[0.0, 0.0]],
            &[8.0],
            &[Xyxy::default()],
            &[2],
            &cfg
        )
        .is_err()
    );
    assert!(assign_simota(&[], 0, &[], &[], &[], &[Xyxy::default()], &[0], &cfg).is_err());

    // 7. End-to-end multi-level FPN anchors -> SimOTA -> dense quality targets -> VFL & QFL backward.
    let anchors = make_anchors(64, 64, &[8, 16, 32], 0.5);
    let a = anchors.len();
    let nc = 3usize;
    let pts = anchors.points_px();
    let mut rng = folo_core::rng::Rng::seed(55);
    let cls_vec: Vec<f32> = (0..a * nc).map(|_| rng.uniform(-3.0, 2.0)).collect();
    let preds: Vec<Xyxy> = pts
        .iter()
        .zip(&anchors.strides)
        .map(|(p, &s)| Xyxy::new(p[0] - s, p[1] - s, p[0] + s, p[1] + s))
        .collect();
    let gts = [Xyxy::new(8.0, 8.0, 40.0, 44.0), Xyxy::new(24.0, 20.0, 56.0, 56.0)];
    let labels = [0u32, 2u32];
    let asg = assign_simota(&cls_vec, nc, &preds, &pts, &anchors.strides, &gts, &labels, &cfg).unwrap();
    assert!(asg.inputs_finite && asg.num_fg >= 2);
    assert_eq!(asg.to_assignment().num_fg, asg.num_fg);

    let q_dense = asg.dense_quality_targets(nc);
    let logits_t = Tensor::from_vec(cls_vec, [a, nc]).unwrap().requires_grad_();
    let targets_t = Tensor::from_vec(q_dense, [a, nc]).unwrap();
    let lv = varifocal_loss(
        &logits_t,
        &targets_t,
        &VflConfig { reduction: LossReduction::Mean, ..VflConfig::default() },
    )
    .unwrap();
    let lq = quality_focal_loss(
        &logits_t,
        &targets_t,
        &QflConfig { reduction: LossReduction::Mean, ..QflConfig::default() },
    )
    .unwrap();
    assert!(lv.item::<f32>().unwrap().is_finite());
    assert!(lq.item::<f32>().unwrap().is_finite());
    assert!(
        lv.backward().unwrap().get(&logits_t).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite())
    );
    assert!(
        lq.backward().unwrap().get(&logits_t).unwrap().to_vec::<f32>().unwrap().iter().all(|v| v.is_finite())
    );

    // Serde round-trip of SimOtaConfig.
    let json = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<SimOtaConfig>(&json).unwrap(), cfg);
}
