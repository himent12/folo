use super::*;
use crate::decode::{side_precision, side_precisions, softmax_moments};

fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> Xyxy {
    Xyxy::new(x1, y1, x2, y2)
}

/// A lone candidate (nothing else of its class reaches the cluster IoU) keeps its box exactly, voting or not.
#[test]
fn lone_candidate_is_unchanged() {
    let boxes = [b(10.0, 10.0, 50.0, 60.0), b(12.0, 10.0, 50.0, 60.0), b(200.0, 0.0, 240.0, 40.0)];
    let classes = [0, 1, 0];
    let scores = [0.9, 0.8, 0.7];
    let prec = |j: usize| [1.0 + j as f32, 2.0, 3.0, 4.0];
    for p in [None, Some(&prec as &dyn Fn(usize) -> [f32; 4])] {
        assert_eq!(fuse_boxes(&[0, 2], &boxes, &scores, &classes, false, 0.75, p), [boxes[0], boxes[2]]);
        assert_eq!(
            fuse_boxes_sebf(&[0, 2], &boxes, &scores, &classes, false, 0.75, None, 0.75, p),
            [boxes[0], boxes[2]]
        );
    }
    let got = fuse_boxes(&[0], &boxes, &scores, &classes, true, 0.75, None);
    assert!((got[0].x1 - (0.9 * 10.0 + 0.8 * 12.0) / 1.7).abs() < 1e-5, "{got:?}");
}

/// Two members, one sharp and one vague on each side: each side moves to its precision-and-score-weighted
/// mean, so the result follows the sharp member side by side; voting takes the score-weighted mean.
#[test]
fn two_members_weigh_their_certainty_per_side() {
    let boxes = [b(10.0, 20.0, 110.0, 120.0), b(12.0, 22.0, 108.0, 118.0)];
    let (scores, classes) = ([0.8f32, 0.4], [3u32, 3]);
    let prec = |j: usize| if j == 0 { [9.0, 9.0, 1.0, 1.0] } else { [1.0, 1.0, 9.0, 9.0] };
    let got = fuse_boxes(&[0], &boxes, &scores, &classes, false, 0.75, Some(&prec))[0];
    let mean = |a: f32, wa: f32, c: f32, wc: f32| (a * wa + c * wc) / (wa + wc);
    let want = [
        mean(10.0, 0.8 * 9.0, 12.0, 0.4),
        mean(20.0, 0.8 * 9.0, 22.0, 0.4),
        mean(110.0, 0.8, 108.0, 0.4 * 9.0),
        mean(120.0, 0.8, 118.0, 0.4 * 9.0),
    ];
    for (g, w) in got.to_array().iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "{got:?} vs {want:?}");
    }
    let vote = fuse_boxes(&[0], &boxes, &scores, &classes, false, 0.75, None)[0];
    assert!((vote.x2 - mean(110.0, 0.8, 108.0, 0.4)).abs() < 1e-4, "{vote:?}");
}

/// The cluster IoU is inclusive, members below it are left out, and a side with no weight (zero scores)
/// keeps the kept box's edge. The result does not depend on where other classes' candidates sit.
#[test]
fn cluster_rule_and_zero_weights() {
    let boxes = [b(0.0, 0.0, 10.0, 10.0), b(0.0, 0.0, 10.0, 7.5), b(0.0, 0.0, 10.0, 7.0)];
    let classes = [0, 0, 0];
    let got = fuse_boxes(&[0], &boxes, &[0.5, 0.5, 0.5], &classes, false, 0.75, None)[0];
    assert_eq!(got.y2, 8.75, "member 1 (IoU 0.75) is in, member 2 (0.7) is out");
    let got = fuse_boxes(&[0], &boxes, &[0.0, 0.0, 0.0], &classes, false, 0.75, None)[0];
    assert_eq!(got, boxes[0]);
    let boxes2 = [boxes[0], b(0.0, 0.0, 10.0, 9.0), boxes[1], boxes[2]];
    let got2 = fuse_boxes(&[0], &boxes2, &[0.5, 0.9, 0.5, 0.5], &[0, 1, 0, 0], false, 0.75, None)[0];
    assert_eq!(got2.y2, 8.75);
    let boxes3 = [b(0.0, 0.0, 10.0, 10.0), b(f32::NAN, 0.0, 5.0, 10.0)];
    let got3 = fuse_boxes(&[0], &boxes3, &[0.9, 0.8], &[0, 0], false, 0.75, None)[0];
    assert_eq!(got3.x1, 0.0);
    assert!(got3.to_array().iter().all(|v| v.is_finite()), "{got3:?}");
}

/// `SEBF` uses exclusive suppressor ownership (never pulling crowded adjacent kept boxes toward each other's
/// cluster members) and down-weights single-side outliers via 1-step Student-t IRLS.
#[test]
fn sebf_exclusive_ownership_and_outlier_rejection() {
    // Two crowded kept boxes (0 and 1) and a candidate 2 owned exclusively by box 0 (owner = [0, 1, 0]).
    // Even if candidate 2 overlaps box 1 above 0.75, SEBF must not let candidate 2 pull box 1.
    let boxes = [b(0.0, 0.0, 100.0, 100.0), b(24.0, 0.0, 124.0, 100.0), b(12.0, 0.0, 112.0, 100.0)];
    let scores = [0.9f32, 0.85, 0.8];
    let classes = [0u32; 3];
    let owners = [0u32, 1, 0];
    let sebf = fuse_boxes_sebf(&[0, 1], &boxes, &scores, &classes, false, 0.75, Some(&owners), 0.78, None);
    assert_eq!(sebf[1], boxes[1], "box 1 has no exclusively owned members -> unchanged");
    assert!(sebf[0].x1 > boxes[0].x1 && sebf[0].x1 < 12.0, "box 0 fuses its owned member: {:?}", sebf[0]);

    // Student-t IRLS outlier rejection: 3 tight members agree on x2=100 while 1 outlier predicts x2=120
    // (still IoU >= 0.75). SEBF pulls less toward the outlier than plain DFL fusion.
    let boxes_out = [
        b(0.0, 0.0, 100.0, 100.0),
        b(0.0, 0.0, 100.0, 100.0),
        b(0.0, 0.0, 100.0, 100.0),
        b(0.0, 0.0, 120.0, 100.0),
    ];
    let sc_out = [0.9f32, 0.85, 0.8, 0.8];
    let cl_out = [0u32; 4];
    let own_out = [0u32; 4];
    let dfl = fuse_boxes(&[0], &boxes_out, &sc_out, &cl_out, false, 0.75, None)[0];
    let sebf_o =
        fuse_boxes_sebf(&[0], &boxes_out, &sc_out, &cl_out, false, 0.75, Some(&own_out), 0.78, None)[0];
    assert!(sebf_o.x2 < dfl.x2, "SEBF should reject x2 outlier harder: sebf={} dfl={}", sebf_o.x2, dfl.x2);

    // NaN candidate does not poison SEBF.
    let nan_boxes = [b(0.0, 0.0, 10.0, 10.0), b(f32::NAN, 0.0, 5.0, 10.0)];
    let got_nan =
        fuse_boxes_sebf(&[0], &nan_boxes, &[0.9, 0.8], &[0, 0], false, 0.75, Some(&[0, 0]), 0.78, None);
    assert_eq!(got_nan[0], nan_boxes[0]);
}

/// `dfl_cgr_scores` gates repulsion by suppressor confidence `p_m^{0.25}`: a weak suppressor (`s=0.05`) repels
/// an overlapping neighbor strictly less than a strong suppressor (`s=0.90`) at the same IoU, and both stay
/// finite and strictly in `(0, 1)`.
#[test]
fn dfl_cgr_confidence_gates_repulsion() {
    let boxes = [b(0.0, 0.0, 100.0, 100.0), b(25.0, 0.0, 125.0, 100.0)];
    let classes = [0u32, 0];
    let owners = [0u32, 1];
    // Strong suppressor (0.9) vs neighbor (0.4)
    let strong_cgr =
        dfl_cgr_scores(&[0, 1], &boxes, &boxes, &[0.9, 0.4], &classes, Some(&owners), 0.78, None, None);
    let strong_rep =
        dfl_repulse_scores(&[0, 1], &boxes, &boxes, &[0.9, 0.4], &classes, Some(&owners), 0.78, None, None);
    // Weak suppressor (0.05) vs neighbor (0.04)
    let weak_cgr =
        dfl_cgr_scores(&[0, 1], &boxes, &boxes, &[0.05, 0.04], &classes, Some(&owners), 0.78, None, None);
    let weak_rep =
        dfl_repulse_scores(&[0, 1], &boxes, &boxes, &[0.05, 0.04], &classes, Some(&owners), 0.78, None, None);
    assert!(
        weak_cgr[1] > weak_rep[1],
        "weak suppressor repels less under CGR: {} vs {}",
        weak_cgr[1],
        weak_rep[1]
    );
    assert!(
        strong_cgr[1] < strong_rep[1],
        "strong suppressor repels slightly harder (γ=3 vs 2.5): {} vs {}",
        strong_cgr[1],
        strong_rep[1]
    );
    assert!(strong_cgr.iter().chain(&weak_cgr).all(|s| s.is_finite() && *s > 0.0 && *s < 1.0));
}

/// The side precision: a perfectly certain edge between two bins (DFL's {3: 0.7, 4: 0.3} for 3.3) has no
/// excess variance, like one on a bin; a spread side has less precision; at stride 32 a side with the same
/// distribution has 1/16 of its precision at stride 8 (a cluster spanning two levels trusts the finer one).
#[test]
fn side_precision_values() {
    let ln = |p: f32| p.ln();
    let mut row = vec![-30.0f32; 16];
    (row[3], row[4]) = (ln(0.7), ln(0.3));
    let (m, v) = softmax_moments(&row);
    assert!((m - 3.3).abs() < 1e-5 && (v - 0.21).abs() < 1e-5, "{m} {v}");
    let certain = 1.0 / (64.0 * DFL_FUSION_EPS);
    assert!((side_precision(m, v, 8.0) / certain - 1.0).abs() < 1e-3);
    assert!((side_precision(3.0, 0.0, 8.0) / certain - 1.0).abs() < 1e-6);
    let (m, v) = softmax_moments(&[0.0; 16]);
    assert!((m - 7.5).abs() < 1e-5 && (v - 21.25).abs() < 1e-4);
    assert!((side_precision(m, v, 8.0) - 1.0 / (64.0 * (21.0 + DFL_FUSION_EPS))).abs() < 1e-9);
    let mut four = row.clone();
    four.extend(row.iter().chain(&row).chain(&row));
    let (p8, p32) = (side_precisions(&four, 16, 8.0), side_precisions(&four, 16, 32.0));
    for s in 0..4 {
        assert!((p8[s] / p32[s] - 16.0).abs() < 1e-3);
    }
}

/// A cluster spanning two levels: two members with the same distributions and scores, one at stride 8 and one at
/// stride 32, fuse 16 : 1 towards the stride-8 member on every side.
#[test]
fn cluster_spanning_two_levels() {
    let mut row = vec![0f32; 64];
    for (s, v) in row.iter_mut().enumerate() {
        *v = -((s % 16) as f32 - 4.0).powi(2) / 2.0;
    }
    let (p8, p32) = (side_precisions(&row, 16, 8.0), side_precisions(&row, 16, 32.0));
    let boxes = [b(100.0, 100.0, 200.0, 200.0), b(104.0, 96.0, 204.0, 196.0)];
    let prec = |j: usize| if j == 0 { p8 } else { p32 };
    let got = fuse_boxes(&[1], &boxes, &[0.5, 0.5], &[0, 0], false, 0.75, Some(&prec))[0];
    let want = [(16.0 * 100.0 + 104.0) / 17.0, (16.0 * 100.0 + 96.0) / 17.0];
    assert!((got.x1 - want[0]).abs() < 1e-3 && (got.y1 - want[1]).abs() < 1e-3, "{got:?}");
    assert!((got.x2 - (16.0 * 200.0 + 204.0) / 17.0).abs() < 1e-3, "{got:?}");
}

/// A censored side (all of its mass on the last bin) has variance 0 and so the highest precision.
#[test]
fn censored_side_is_trusted_like_a_certain_one() {
    let mut censored = vec![-30.0f32; 16];
    censored[15] = 10.0;
    let (m, v) = softmax_moments(&censored);
    assert!((m - 15.0).abs() < 1e-4 && v < 1e-4, "{m} {v}");
    let p = side_precision(m, v, 8.0);
    assert!((p * 64.0 * DFL_FUSION_EPS - 1.0).abs() < 1e-3, "{p}");
}

/// Hand-computed consensus scores.
#[test]
fn consensus_scores_by_hand() {
    let boxes = [
        b(0.0, 0.0, 10.0, 10.0),
        b(0.0, 0.0, 10.0, 10.0),
        b(100.0, 0.0, 110.0, 10.0),
        b(100.0, 0.0, 110.0, 7.5),
        b(100.0, 0.0, 110.0, 7.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
    ];
    let scores = [0.8f32, 0.9, 0.4, 0.6, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5];
    let classes = [0u32, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let got = consensus_scores(&[0, 2, 5], &boxes, &scores, &classes);
    assert_eq!(got[0], (0.8f32 as f64 / 10f64.sqrt()) as f32, "lonely: s / sqrt(10)");
    let mass = 0.4f32 as f64 + 0.6f32 as f64;
    let want = 0.4f32 as f64 * (mass / 0.4f32 as f64).sqrt() / 10f64.sqrt();
    assert_eq!(got[1], want as f32, "between: M/s = 2.5");
    assert!((got[1] - 0.2).abs() < 1e-6, "0.4 * sqrt(2.5 / 10) = 0.2, got {}", got[1]);
    assert_eq!(got[2].to_bits(), 0.1f32.to_bits(), "at the cap the score stays");
}

/// Consensus edge cases: zero area, NaN boxes, non-positive scores.
#[test]
fn consensus_edge_cases() {
    let flat = [b(5.0, 5.0, 5.0, 9.0)];
    let got = consensus_scores(&[0], &flat, &[0.5], &[0])[0];
    assert_eq!(got, (0.5f64 / 10f64.sqrt()) as f32);
    let boxes = [b(0.0, 0.0, 10.0, 10.0), b(f32::NAN, 0.0, 5.0, 10.0), b(0.0, 0.0, 10.0, 10.0)];
    assert_eq!(overlap(&boxes[1], boxes[1].area(), &boxes[0]), 1.0, "the pair the guard is for");
    assert_eq!(overlap(&boxes[0], boxes[0].area(), &boxes[1]), 1.0, "the pair the guard is for");
    let got = consensus_scores(&[0], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0])[0];
    assert_eq!(got, (0.5f64 / 10f64.sqrt()) as f32, "the NaN box and the zero score add nothing");
    let got = consensus_scores(&[1], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0])[0];
    assert_eq!(got, (0.9f32 as f64 / 10f64.sqrt()) as f32, "a kept NaN box has no members but itself");
    assert_eq!(consensus_scores(&[2], &boxes, &[0.5, 0.9, 0.0], &[0, 0, 0]), [0.0]);
}

/// `fuse_boxes` reads a class only for a kept candidate under `agnostic`: a shorter `classes` still works.
#[test]
fn agnostic_fusion_reads_only_the_kept_class() {
    let boxes = [b(0.0, 0.0, 10.0, 10.0), b(1.0, 0.0, 10.0, 10.0)];
    let got = fuse_boxes(&[0], &boxes, &[0.5, 0.5], &[0], true, 0.75, None)[0];
    assert_eq!(got.x1, 0.5, "both boxes vote");
}

/// The same inputs give the same boxes, bit for bit.
#[test]
fn deterministic() {
    let boxes: Vec<Xyxy> =
        (0..40).map(|i| b(i as f32 * 0.37, 5.0 - i as f32 * 0.11, 60.0 + i as f32 * 0.2, 70.0)).collect();
    let scores: Vec<f32> = (0..40).map(|i| 0.9 - i as f32 * 0.01).collect();
    let classes = vec![0u32; 40];
    let prec = |j: usize| [1.0 / (1.0 + j as f32), 2.0, 0.5, 1.0 + (j % 3) as f32];
    let kept = [0usize, 7, 21];
    let a = fuse_boxes(&kept, &boxes, &scores, &classes, false, 0.75, Some(&prec));
    assert_eq!(a, fuse_boxes(&kept, &boxes, &scores, &classes, false, 0.75, Some(&prec)));
    assert_ne!(a[0], boxes[0], "the cluster moved the box");
    let s = fuse_boxes_sebf(&kept, &boxes, &scores, &classes, false, 0.75, None, 0.78, Some(&prec));
    assert_eq!(s, fuse_boxes_sebf(&kept, &boxes, &scores, &classes, false, 0.75, None, 0.78, Some(&prec)));
}

/// Regression: if asymmetric per-side precisions at a low `fusion_iou` invert the first weighted fit `b1`, SEBF
/// must return the original kept box `b` instead of an inverted box.
#[test]
fn sebf_inverted_first_fit_falls_back_to_kept_box() {
    let boxes = [b(0.0, 0.0, 10.0, 10.0), b(-6.0, 0.0, 4.0, 10.0), b(6.0, 0.0, 16.0, 10.0)];
    let scores = [0.9f32, 0.8, 0.8];
    let classes = [0u32, 0, 0];
    // Kept box has zero horizontal precision; left neighbor pins x2 = 4, right neighbor pins x1 = 6 -> b1 = [6, 4].
    let prec = |j: usize| match j {
        0 => [0.0, 1.0, 0.0, 1.0],
        1 => [0.0, 1.0, 100.0, 1.0],
        _ => [100.0, 1.0, 0.0, 1.0],
    };
    let got = fuse_boxes_sebf(&[0], &boxes, &scores, &classes, false, 0.2, None, 0.2, Some(&prec))[0];
    assert_eq!(got, boxes[0], "inverted b1 must fall back to kept box");
}

/// Regression: under `agnostic: true` NMS with a non-finite highest-scoring kept box that suppresses a candidate,
/// SEBF receives actual agnostic NMS ownership and must not let the next kept box steal that suppressed candidate.
#[test]
fn sebf_agnostic_respects_actual_nms_ownership_with_nan_box() {
    use crate::nms::{Detection, NmsConfig, nms_detections};
    let boxes = [b(f32::NAN, 0.0, 5.5, 10.0), b(1.2, 0.0, 11.2, 10.0), b(0.0, 0.0, 10.0, 10.0)];
    let input = [
        Detection { bbox: boxes[0], score: 0.9, class: 0 },
        Detection { bbox: boxes[1], score: 0.8, class: 1 },
        Detection { bbox: boxes[2], score: 0.7, class: 2 },
    ];
    let cfg = NmsConfig {
        conf: 0.0,
        iou: 0.78,
        fusion: BoxFusion::Sebf,
        fusion_iou: 0.78,
        score: ScoreRule::None,
        agnostic: true,
        ..NmsConfig::default()
    };
    let dets = nms_detections(&input, &cfg);
    assert_eq!(dets.len(), 2, "box 0 (NaN) suppresses box 2, keeping boxes 0 and 1");
    assert_eq!(dets[1].bbox, boxes[1], "box 1 must not fuse box 2 owned by box 0");
    let scores = [0.9f32, 0.8, 0.7];
    let classes = [0u32, 1, 2];
    let direct = fuse_boxes_sebf(&[0, 1], &boxes, &scores, &classes, true, 0.78, None, 0.78, None);
    assert_eq!(direct[1], boxes[1], "owner = None fallback must match NMS ownership on NaN box");

    // Same-class agnostic NMS with `ScoreRule::Elc` must preserve class-local NaN skipping so finite survivor 1
    // still receives support from candidate 2 during calibrated scoring.
    let same_cls_input = [
        Detection { bbox: boxes[0], score: 0.9, class: 0 },
        Detection { bbox: boxes[1], score: 0.8, class: 0 },
        Detection { bbox: boxes[2], score: 0.7, class: 0 },
    ];
    let elc_cfg = NmsConfig {
        conf: 0.0,
        iou: 0.78,
        fusion: BoxFusion::None,
        score: ScoreRule::Elc,
        agnostic: true,
        ..NmsConfig::default()
    };
    let elc_dets = nms_detections(&same_cls_input, &elc_cfg);
    let solo_elc = elc_scores(&[0], &[boxes[1]], &[0.8], &[0], None, 0.78)[0];
    let finite_det = elc_dets.iter().find(|d| d.bbox == boxes[1]).expect("finite survivor kept");
    assert!(
        finite_det.score > solo_elc,
        "class-local ELC must retain candidate-2 support on finite survivor: {} vs solo {solo_elc}",
        finite_det.score
    );
}

/// Regression: calling `fuse_boxes_sebf` directly with `agnostic = true` and `owner = None` reconstructs
/// cross-class ownership identically to precomputed agnostic NMS owners, so an earlier cross-class survivor A
/// keeps ownership of suppressed box B instead of letting a later same-class survivor C steal B.
#[test]
fn sebf_agnostic_owner_none_matches_precomputed_owner() {
    // A = [0,0,10,10] (class 0, kept #0), C = [2,0,12,10] (class 1, kept #1; IoU(A,C)=8/12=0.667 <= 0.75),
    // B = [1,0,11,10] (class 1, suppressed by A with IoU=9/11=0.818 > 0.75; also overlaps C with IoU=9/11).
    let boxes = [b(0.0, 0.0, 10.0, 10.0), b(2.0, 0.0, 12.0, 10.0), b(1.0, 0.0, 11.0, 10.0)];
    let scores = [0.9f32, 0.8, 0.7];
    let classes = [0u32, 1, 1];
    let kept = [0usize, 1];
    let precomputed = [0u32, 1, 0];
    let with_owner =
        fuse_boxes_sebf(&kept, &boxes, &scores, &classes, true, 0.75, Some(&precomputed), 0.75, None);
    let without_owner = fuse_boxes_sebf(&kept, &boxes, &scores, &classes, true, 0.75, None, 0.75, None);
    assert_eq!(
        without_owner, with_owner,
        "agnostic owner=None fallback must match precomputed agnostic owners"
    );
    assert_eq!(without_owner[1], boxes[1], "later survivor C must not fuse B owned by earlier survivor A");
}

#[test]
fn fuse_boxes_ignores_infinite_score_or_precision_without_nan() {
    let b0 = Xyxy::new(10.0, 10.0, 50.0, 50.0);
    let b1 = Xyxy::new(14.0, 14.0, 54.0, 54.0);
    let boxes = [b0, b1];
    let classes = [0, 0];

    // +Inf score on a cluster member must not turn acc / sum into Inf / Inf = NaN.
    let out_inf_score = fuse_boxes(&[0], &boxes, &[0.8, f32::INFINITY], &classes, false, 0.5, None);
    assert_eq!(out_inf_score, vec![b0]);

    // +Inf score on the sole contributing member falls back cleanly to the kept box.
    let out_only_inf = fuse_boxes(&[0], &[b0], &[f32::INFINITY], &[0], false, 0.5, None);
    assert_eq!(out_only_inf, vec![b0]);

    // +Inf DFL precision on one side leaves finite sides fused and falls back on the non-finite side.
    let prec = |j: usize| if j == 0 { [f32::INFINITY, 1.0, 1.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
    let out_inf_prec = fuse_boxes(&[0], &boxes, &[0.8, 0.4], &classes, false, 0.5, Some(&prec));
    let f = out_inf_prec[0];
    assert!(f.x1.is_finite() && f.y1.is_finite() && f.x2.is_finite() && f.y2.is_finite(), "{f:?}");
    // Side 0 only receives b1's finite contribution; sides 1..4 fuse both.
    assert!((f.x1 - 14.0).abs() < 1e-5, "{f:?}");
    assert!((f.y1 - (0.8 * 10.0 + 0.4 * 14.0) / 1.2).abs() < 1e-4, "{f:?}");
}
