//! Unit tests for ATSS ([`assign_atss`]) and PAA ([`assign_paa`], [`fit_gmm_1d_two_component`]):
//! multi-scale top-k selection, exact hand-computed `mean + std` dynamic thresholds (population and
//! Bessel-corrected), in-GT center filtering, highest-IoU / lowest-loss conflict resolution, 2-component 1D
//! GMM EM bimodal separation, unequal-variance tail safety, empty/degenerate/non-finite handling,
//! determinism, and config/input validation.

use super::*;
use crate::boxes::Xyxy;

fn make_grid_level(grid_n: usize, stride: f32, scale: f32) -> (Vec<[f32; 2]>, Vec<f32>, Vec<Xyxy>) {
    let mut pts = Vec::with_capacity(grid_n * grid_n);
    let mut strides = Vec::with_capacity(grid_n * grid_n);
    for r in 0..grid_n {
        for c in 0..grid_n {
            pts.push([(c as f32 + 0.5) * stride, (r as f32 + 0.5) * stride]);
            strides.push(stride);
        }
    }
    let boxes = anchor_boxes_from_points(&pts, &strides, scale).unwrap();
    (pts, strides, boxes)
}

#[test]
fn atss_multiscale_threshold_and_conflict_resolution() {
    // 2 pyramid levels: 4x4 stride 8 (scale 2 -> 16x16 boxes) + 2x2 stride 16 (scale 2 -> 32x32 boxes).
    let (_, _, lvl0) = make_grid_level(4, 8.0, 2.0);
    let (_, _, lvl1) = make_grid_level(2, 16.0, 2.0);
    let level_sizes = [lvl0.len(), lvl1.len()];
    let mut anchors = Vec::new();
    anchors.extend_from_slice(&lvl0);
    anchors.extend_from_slice(&lvl1);

    // Single GT centered around (12, 12) with size 16x16 -> matches lvl0 anchor (12, 12) perfectly (IoU = 1).
    let gts = [Xyxy::new(4.0, 4.0, 20.0, 20.0)];
    let labels = [1u32];
    let cfg = AtssConfig { topk: 4, in_gt_box: true, unbiased_std: false, min_iou_thresh: 0.0, eps: 0.0 };
    let asg = assign_atss(&anchors, &level_sizes, &gts, &labels, 3, &cfg).unwrap();

    assert!(asg.inputs_finite);
    assert_eq!(asg.candidate_counts, vec![8]); // 4 from lvl0 + 4 from lvl1
    assert!(asg.iou_thresholds[0] > 0.0 && asg.iou_thresholds[0] <= 1.0);
    assert!(asg.num_fg >= 1);
    // Anchor at row=1, col=1 in lvl0 has center (12, 12) and box [4, 4, 20, 20] -> exact IoU = 1.0.
    let exact_idx = 4 + 1;
    assert!(asg.fg_mask[exact_idx], "exact-match anchor must be foreground");
    assert_eq!(asg.target_gt[exact_idx], 0);
    assert_eq!(asg.target_labels[exact_idx], 1);
    assert!((asg.target_ious[exact_idx] - 1.0).abs() < 1e-6);

    // Unbiased std uses N - 1 denominator -> strictly larger threshold than population std when variance > 0.
    let cfg_unbiased = AtssConfig { unbiased_std: true, ..cfg };
    let asg_unbiased = assign_atss(&anchors, &level_sizes, &gts, &labels, 3, &cfg_unbiased).unwrap();
    assert!(asg_unbiased.iou_thresholds[0] > asg.iou_thresholds[0]);

    // Verify conflict resolution: two overlapping GTs competing for the same anchor; higher IoU wins.
    let conflict_anchors = [Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(20.0, 20.0, 30.0, 30.0)];
    let gts_conflict = [
        Xyxy::new(0.0, 0.0, 14.0, 14.0), // IoU with anchor 0 = 100 / 196 ≈ 0.510
        Xyxy::new(0.0, 0.0, 10.0, 10.0), // IoU with anchor 0 = 1.0 (wins!)
    ];
    let asg_c = assign_atss(
        &conflict_anchors,
        &[2],
        &gts_conflict,
        &[0, 2],
        3,
        &AtssConfig { topk: 2, in_gt_box: true, ..AtssConfig::default() },
    )
    .unwrap();
    assert!(asg_c.fg_mask[0]);
    assert_eq!(asg_c.target_gt[0], 1, "higher-IoU GT 1 must win contested anchor 0");
    assert_eq!(asg_c.target_labels[0], 2);
    assert!(!asg_c.fg_mask[1]);

    // Conversion helpers match.
    let generic = asg_c.to_assignment();
    assert_eq!(generic.num_fg, 1);
    let dense = asg_c.dense_quality_targets(3);
    assert_eq!(dense.len(), 6);
    assert!((dense[2] - 1.0).abs() < 1e-6);
    assert_eq!(dense[0], 0.0);
}

#[test]
fn atss_handles_empty_degenerate_disjoint_and_nonfinite() {
    let anchors = [Xyxy::new(0.0, 0.0, 10.0, 10.0), Xyxy::new(10.0, 0.0, 20.0, 10.0)];
    let cfg = AtssConfig::default();

    // Empty GTs.
    let empty_gt = assign_atss(&anchors, &[2], &[], &[], 2, &cfg).unwrap();
    assert!(empty_gt.inputs_finite);
    assert_eq!(empty_gt.num_fg, 0);
    assert_eq!(empty_gt.target_labels, vec![2, 2]);

    // Degenerate GT + completely disjoint valid GT -> 0 positives (zero-IoU candidates must NOT become fg).
    let gts = [
        Xyxy::new(5.0, 5.0, 5.0, 10.0),    // zero width
        Xyxy::new(9.0, 9.0, 2.0, 2.0),     // inverted
        Xyxy::new(50.0, 50.0, 60.0, 60.0), // disjoint from all anchors
    ];
    let res = assign_atss(&anchors, &[2], &gts, &[0, 1, 0], 2, &cfg).unwrap();
    assert!(res.inputs_finite);
    assert_eq!(res.num_fg, 0);
    assert_eq!(res.candidate_counts, vec![0, 0, 2]);
    assert_eq!(res.iou_thresholds, vec![0.0, 0.0, 0.0]);

    // Non-finite input sets inputs_finite = false cleanly.
    let bad_gt = [Xyxy::new(f32::NAN, 0.0, 10.0, 10.0)];
    let nf = assign_atss(&anchors, &[2], &bad_gt, &[0], 2, &cfg).unwrap();
    assert!(!nf.inputs_finite);
    assert_eq!(nf.num_fg, 0);

    // Invalid configs and mismatched level_sizes error cleanly.
    assert!(assign_atss(&anchors, &[1], &[], &[], 2, &cfg).is_err());
    assert!(assign_atss(&anchors, &[2], &[], &[], 2, &AtssConfig { topk: 0, ..cfg }).is_err());
    assert!(
        assign_atss(&anchors, &[2], &[Xyxy::new(0.0, 0.0, 5.0, 5.0)], &[2], 2, &cfg).is_err(),
        "out-of-range label must error"
    );

    // Exact constant-IoU ties (3 anchors with IoU 0.1, and 9 anchors with IoU 4/9) must not lose
    // candidates to 1-ULP mean/std rounding.
    let tie3_anchors =
        [Xyxy::new(1.0, 0.0, 2.0, 10.0), Xyxy::new(4.0, 0.0, 5.0, 10.0), Xyxy::new(7.0, 0.0, 8.0, 10.0)];
    let tie3 = assign_atss(&tie3_anchors, &[3], &[Xyxy::new(0.0, 0.0, 10.0, 10.0)], &[0], 1, &cfg).unwrap();
    assert_eq!(tie3.num_fg, 3, "exact 0.1 IoU tie must keep all 3 candidates");
    assert_eq!(tie3.iou_thresholds[0], 0.1);

    let tie9_anchors: Vec<Xyxy> =
        (0..9).map(|i| Xyxy::new(i as f32 * 0.5, 0.0, i as f32 * 0.5 + 4.0, 9.0)).collect();
    let tie9 = assign_atss(&tie9_anchors, &[9], &[Xyxy::new(0.0, 0.0, 9.0, 9.0)], &[0], 1, &cfg).unwrap();
    assert_eq!(tie9.num_fg, 9, "exact 4/9 IoU tie across 9 candidates must keep all 9");
}

#[test]
fn gmm_1d_bimodal_separation_and_tail_safety() {
    // Clean bimodal distribution: 4 low-loss positives around 0.25, 5 high-loss negatives around 4.5.
    let scores = [0.15, 0.22, 0.30, 0.35, 4.1, 4.4, 4.6, 4.8, 5.2];
    let gmm = fit_gmm_1d_two_component(&scores, 50, 1e-8, 1e-6).unwrap();
    assert!(gmm.converged);
    assert!(gmm.pos_mean < 0.5 && gmm.neg_mean > 4.0);
    assert_eq!(gmm.positive_mask, vec![true, true, true, true, false, false, false, false, false]);
    assert!(gmm.pos_posteriors[..4].iter().all(|&p| p > 0.99));
    assert!(gmm.pos_posteriors[4..].iter().all(|&p| p < 0.01));

    // Unequal-variance tail safety: positive cluster has wider spread than tight negative cluster, yet
    // points >= neg_mean must NEVER flip back to positive.
    let tail_scores = [0.1, 1.0, 2.0, 9.98, 9.99, 10.0, 10.01, 10.02, 12.0];
    let gmm_tail = fit_gmm_1d_two_component(&tail_scores, 100, 1e-8, 1e-6).unwrap();
    assert!(gmm_tail.positive_mask[0], "lowest-loss sample must be positive");
    assert!(!gmm_tail.positive_mask[8], "highest-loss sample >= neg_mean must remain negative");

    // Constant & single-element edge cases stay finite and mark positives deterministically.
    let single = fit_gmm_1d_two_component(&[1.25], 20, 1e-6, 1e-6).unwrap();
    assert_eq!(single.positive_mask, vec![true]);
    let constant = fit_gmm_1d_two_component(&[2.0, 2.0, 2.0], 20, 1e-6, 1e-6).unwrap();
    assert_eq!(constant.positive_mask, vec![true, true, true]);
    let empty = fit_gmm_1d_two_component(&[], 20, 1e-6, 1e-6).unwrap();
    assert!(empty.positive_mask.is_empty());

    // Rejects non-finite scores and invalid solver parameters.
    assert!(fit_gmm_1d_two_component(&[0.5, f64::NAN], 20, 1e-6, 1e-6).is_err());
    assert!(fit_gmm_1d_two_component(&[0.5, 1.0], 0, 1e-6, 1e-6).is_err());
    assert!(fit_gmm_1d_two_component(&[0.5, 1.0], 20, 1e-6, 0.0).is_err());

    // Large identical finite values (`[1e308, 1e308]`) stay finite without sum overflow;
    // unrepresentable variance spreads (`[0.0, 1e160]`) return Error::Config instead of NaN/Inf.
    let huge_const = fit_gmm_1d_two_component(&[1e308, 1e308], 20, 1e-6, 1e-6).unwrap();
    assert!(huge_const.pos_mean == 1e308 && huge_const.log_likelihood.is_finite());
    assert!(fit_gmm_1d_two_component(&[0.0, 1e160], 20, 1e-6, 1e-6).is_err());
}

#[test]
fn paa_separates_good_and_bad_predictions_and_resolves_conflicts() {
    let (_, _, anchors) = make_grid_level(3, 10.0, 1.5); // 9 anchors
    let n_a = anchors.len();
    let num_classes = 2;
    let gt = Xyxy::new(7.5, 7.5, 22.5, 22.5); // centered on anchor 4 (15, 15)

    // Anchors 3, 4, 5 have strong class-0 logits and tight predicted boxes; others have poor logits & far boxes.
    let mut logits = vec![-4.0f32; n_a * num_classes];
    let mut pd_bboxes = vec![Xyxy::new(40.0, 40.0, 50.0, 50.0); n_a];
    for a in [3usize, 4, 5] {
        logits[a * num_classes] = 4.0;
        logits[a * num_classes + 1] = -4.0;
        pd_bboxes[a] = Xyxy::new(8.0, 8.0, 22.0, 22.0);
    }

    let cfg = PaaConfig { topk: 9, ..PaaConfig::default() };
    let asg1 = assign_paa(&logits, num_classes, &pd_bboxes, &anchors, &[n_a], &[gt], &[0], &cfg).unwrap();
    let asg2 = assign_paa(&logits, num_classes, &pd_bboxes, &anchors, &[n_a], &[gt], &[0], &cfg).unwrap();
    assert_eq!(asg1, asg2, "PAA must be bitwise deterministic");
    assert!(asg1.inputs_finite);
    assert_eq!(asg1.num_fg, 3);
    for (a, &fg) in asg1.fg_mask.iter().enumerate() {
        assert_eq!(fg, matches!(a, 3..=5), "anchor {a} mismatch");
    }
    let stats = asg1.gmm_stats[0].as_ref().unwrap();
    assert!(stats.pos_mean < stats.neg_mean);

    // Conflict resolution between two GTs: GT 1 has better classification & box match for anchor 4 -> GT 1 wins.
    let gts_2 = [Xyxy::new(7.5, 7.5, 22.5, 22.5), Xyxy::new(8.0, 8.0, 22.0, 22.0)];
    let mut logits_c = vec![-3.0f32; n_a * num_classes];
    logits_c[4 * num_classes] = 0.0;
    logits_c[4 * num_classes + 1] = 5.0; // strongly prefers class 1 (GT 1)
    let asg_c =
        assign_paa(&logits_c, num_classes, &pd_bboxes, &anchors, &[n_a], &gts_2, &[0, 1], &cfg).unwrap();
    assert!(asg_c.fg_mask[4]);
    assert_eq!(asg_c.target_gt[4], 1, "lower-loss GT 1 must win contested anchor 4");
    assert_eq!(asg_c.target_labels[4], 1);

    // Empty GTs, degenerate GT, and non-finite logits handled cleanly.
    let empty = assign_paa(&logits, num_classes, &pd_bboxes, &anchors, &[n_a], &[], &[], &cfg).unwrap();
    assert!(empty.inputs_finite && empty.num_fg == 0);

    let degen = assign_paa(
        &logits,
        num_classes,
        &pd_bboxes,
        &anchors,
        &[n_a],
        &[Xyxy::new(5.0, 5.0, 5.0, 5.0)],
        &[0],
        &cfg,
    )
    .unwrap();
    assert!(degen.inputs_finite && degen.num_fg == 0 && degen.gmm_stats[0].is_none());

    let mut bad_logits = logits.clone();
    bad_logits[0] = f32::INFINITY;
    let nf = assign_paa(&bad_logits, num_classes, &pd_bboxes, &anchors, &[n_a], &[gt], &[0], &cfg).unwrap();
    assert!(!nf.inputs_finite && nf.num_fg == 0);
}
