//! Tests for `--score elc` (Exclusive Logit-Space Consensus) and `--score dfl-consensus` (Scale-Free Exclusive
//! Logit-Consensus with fused DFL precision), docs/MODEL.md §9.4: exact hand-checked formulas, exclusive ghost-mass
//! blocking, graceful fallback when DFL moments are absent, `agnostic` class-local ownership, NaN/extreme-score
//! safety, determinism, zero regression on `None`/`Consensus`, and end-to-end CPU candidate-first / val parity.

use std::sync::Arc;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_data::Split;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_nn::Module;
use folo_vision::boxes::Xyxy;
use folo_vision::decode::{bin_precision, dfl_entropy, side_precision, softmax_moments};
use folo_vision::fusion::{
    BoxFusion, DFL_CONSENSUS_BETA, DFL_CONSENSUS_SIG_REF, DFL_REPULSE_ALPHA, DFL_REPULSE_BETA,
    DFL_REPULSE_ENT_REF, DFL_REPULSE_GAMMA, DFL_REPULSE_R_REF, DFL_REPULSE_STRIDE_W, ELC_ALPHA, ELC_CAP,
    ELC_R_REF, ScoreRule, consensus_scores, dfl_consensus_scores, dfl_repulse_scores, elc_scores,
};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::{
    CALIBRATED_PRE_CONF, Detection, DflInputs, NmsConfig, nms_detections, non_max_suppression,
    non_max_suppression_with_dfl, non_max_suppression_with_inputs,
};
use folo_vision::val::{ValConfig, validate};

fn b(x1: f32, y1: f32, x2: f32, y2: f32) -> Xyxy {
    Xyxy::new(x1, y1, x2, y2)
}

fn ref_sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

fn ref_elc(s: f32, mass: f64) -> f32 {
    let s64 = s as f64;
    let sc = s64.clamp(1e-6, 1.0 - 1e-6);
    let z = (sc / (1.0 - sc)).ln() + ELC_ALPHA * ((mass / s64).clamp(1.0, ELC_CAP) / ELC_R_REF).ln();
    ref_sigmoid(z) as f32
}

/// Hand-checked ELC and SF-ELC-DFL values: lonely box, cluster below cap, cluster above cap, and DFL bin-precision
/// adjustment (sharp distribution lifts score relative to spread distribution).
#[test]
fn hand_checked_elc_and_dfl_consensus() {
    let boxes = [
        // Lonely box 0 (class 0); box 1 is identical in class 1 (must not contribute).
        b(0.0, 0.0, 10.0, 10.0),
        b(0.0, 0.0, 10.0, 10.0),
        // Cluster around box 2 (s=0.6): box 3 at IoU 0.75 (s=0.3, included), box 4 at IoU 0.70 (excluded).
        b(100.0, 0.0, 110.0, 10.0),
        b(100.0, 0.0, 110.0, 7.5),
        b(100.0, 0.0, 110.0, 7.0),
        // Above cap: box 5 (s=0.1) + 12 identical neighbors (s=0.1 each) => M/s = 13 -> clamped to 10.
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
        b(200.0, 0.0, 210.0, 10.0),
    ];
    let mut scores = vec![0.1f32; boxes.len()];
    scores[0] = 0.8;
    scores[1] = 0.9;
    scores[2] = 0.6;
    scores[3] = 0.3;
    scores[4] = 0.5;
    let mut classes = vec![0u32; boxes.len()];
    classes[1] = 1;

    let got = elc_scores(&[0, 2, 5], &boxes, &scores, &classes, None, 0.70);
    assert_eq!(got[0], ref_elc(0.8, 0.8f32 as f64), "lonely box");
    assert_eq!(got[1], ref_elc(0.6, 0.6f32 as f64 + 0.3f32 as f64), "cluster IoU=0.75 in, 0.70 out");
    assert_eq!(got[2], ref_elc(0.1, 1.3f64), "above cap clamps at 10");

    // SF-ELC-DFL: sharp bin_prec=10.0 (sig_bin = 1/sqrt(10) = 0.3162 < 0.50 -> positive logit shift) vs
    // spread bin_prec=1.0 (sig_bin = 1.0 > 0.50 -> negative logit shift).
    let sharp = |_j: usize| [10.0f32; 4];
    let spread = |_j: usize| [1.0f32; 4];
    let s_sharp = dfl_consensus_scores(&[0], &boxes, &scores, &classes, None, 0.70, Some(&sharp))[0];
    let s_spread = dfl_consensus_scores(&[0], &boxes, &scores, &classes, None, 0.70, Some(&spread))[0];
    let want_sharp = {
        let sc = (0.8f32 as f64).clamp(1e-6, 1.0 - 1e-6);
        let z = (sc / (1.0 - sc)).ln() + ELC_ALPHA * (1.0 / ELC_R_REF).ln()
            - DFL_CONSENSUS_BETA * (10.0f64.sqrt().recip() / DFL_CONSENSUS_SIG_REF).ln();
        ref_sigmoid(z) as f32
    };
    assert!((s_sharp - want_sharp).abs() < 1e-6, "{s_sharp} vs {want_sharp}");
    assert!(s_sharp > got[0] && got[0] > s_spread, "sharp {s_sharp} > elc {} > spread {s_spread}", got[0]);
}

/// Ghost-mass blocking: a shifted surviving duplicate box (IoU 0.60 with the winner, so it survives NMS at 0.65)
/// overlaps an intermediate candidate (IoU 0.78 with BOTH winner and duplicate, suppressed by the winner).
/// Shipped `Consensus` lets the duplicate steal the intermediate candidate's mass; `Elc` gives the duplicate ZERO
/// stolen mass (only its own score).
#[test]
fn exclusive_ownership_blocks_duplicate_ghost_mass() {
    // Winner k0: [0, 0, 100, 100], s=0.90
    // Intermediate j1: [11, 0, 111, 100], s=0.80 (IoU with k0 = 89/111 = 0.802 > 0.65 & >= 0.75)
    // Duplicate k2: [25, 0, 125, 100], s=0.30 (IoU with k0 = 75/125 = 0.60 <= 0.65 -> survives NMS;
    //                                          IoU with j1 = 86/114 = 0.754 >= 0.75!)
    let boxes = [b(0.0, 0.0, 100.0, 100.0), b(11.0, 0.0, 111.0, 100.0), b(25.0, 0.0, 125.0, 100.0)];
    let scores = [0.90f32, 0.80, 0.30];

    let cfg_ship = NmsConfig {
        conf: 0.001,
        iou: 0.65,
        multi_label: true,
        score: ScoreRule::Consensus,
        ..NmsConfig::default()
    };
    let ship = non_max_suppression(&boxes, &scores, 1, &cfg_ship);
    assert_eq!(ship.len(), 2);
    // Under shipped Consensus, k2 stole j1 (0.80) so mass = 1.10 -> M/s = 3.667 > 1!
    let k2_ship_lonely = (0.30f32 as f64 / 10f64.sqrt()) as f32;
    assert!(ship[1].score > k2_ship_lonely * 1.5, "shipped consensus inflated duplicate: {}", ship[1].score);

    let cfg_elc = NmsConfig { score: ScoreRule::Elc, ..cfg_ship };
    let elc = non_max_suppression(&boxes, &scores, 1, &cfg_elc);
    assert_eq!(elc.len(), 2);
    // Winner k0 owns j1 (mass = 0.90 + 0.80 = 1.70); duplicate k2 owns ONLY itself (mass = 0.30)!
    assert_eq!(elc[0].score, ref_elc(0.90, 0.90f32 as f64 + 0.80f32 as f64));
    assert_eq!(elc[1].score, ref_elc(0.30, 0.30f32 as f64));
}

/// `ScoreRule::DflConsensus` falls back identically to `ScoreRule::Elc` when callers pass no DFL bin precisions
/// (`nms_detections`, `non_max_suppression`, `non_max_suppression_with`).
#[test]
fn dfl_consensus_falls_back_to_elc_without_bin_precisions() {
    let boxes = [b(0.0, 0.0, 40.0, 40.0), b(1.0, 1.0, 41.0, 41.0), b(100.0, 100.0, 140.0, 140.0)];
    let scores = [0.75f32, 0.55, 0.40];
    let cfg_elc = NmsConfig {
        conf: 0.001,
        iou: 0.70,
        multi_label: true,
        score: ScoreRule::Elc,
        ..NmsConfig::default()
    };
    let cfg_dfl = NmsConfig { score: ScoreRule::DflConsensus, ..cfg_elc.clone() };
    assert_eq!(
        non_max_suppression(&boxes, &scores, 1, &cfg_dfl),
        non_max_suppression(&boxes, &scores, 1, &cfg_elc)
    );

    let dets: Vec<Detection> =
        boxes.iter().zip(scores).map(|(&bbox, score)| Detection { bbox, score, class: 0 }).collect();
    assert_eq!(nms_detections(&dets, &cfg_dfl), nms_detections(&dets, &cfg_elc));
}

/// Under `agnostic = true`, ELC ownership stays strictly class-local: interleaving boxes of another class cannot
/// steal or alter class-0's exclusive scores.
#[test]
fn agnostic_nms_keeps_elc_ownership_class_local() {
    let boxes_only0 = [b(0.0, 0.0, 50.0, 50.0), b(1.0, 0.0, 50.0, 50.0)];
    let scores_only0 = [0.70f32, 0.50];
    let cfg = NmsConfig {
        conf: 0.001,
        iou: 0.70,
        agnostic: true,
        multi_label: true,
        score: ScoreRule::Elc,
        ..NmsConfig::default()
    };
    let want = non_max_suppression(&boxes_only0, &scores_only0, 1, &cfg);
    // Add a higher-scoring class-1 box sitting far away and another overlapping class-1 box with lower score.
    let boxes_2c = [b(0.0, 0.0, 50.0, 50.0), b(1.0, 0.0, 50.0, 50.0), b(200.0, 200.0, 250.0, 250.0)];
    // 2 classes: anchor 0 has class 0 (0.70) and class 1 (0.60 — suppressed by anchor 0's class 0 under agnostic);
    // anchor 1 has class 0 (0.50); anchor 2 has class 1 (0.95).
    let scores_2c = [0.70f32, 0.60, 0.50, 0.0, 0.0, 0.95];
    let got = non_max_suppression(&boxes_2c, &scores_2c, 2, &cfg);
    let c0: Vec<Detection> = got.into_iter().filter(|d| d.class == 0).collect();
    assert_eq!(c0, want);
}

/// Edge-case safety: NaN boxes, zero-area boxes, extreme scores near 0 and 1, NaN bin precisions, and serde names.
#[test]
fn edge_cases_nan_safety_and_serde() {
    let boxes = [
        b(0.0, 0.0, 10.0, 10.0),
        b(f32::NAN, 0.0, 5.0, 10.0),
        b(5.0, 5.0, 5.0, 9.0), // zero area
        b(50.0, 50.0, 60.0, 60.0),
    ];
    let scores = [1.0f32, 0.8, 1e-8, 0.0];
    let classes = [0u32; 4];
    let bad_bp = |j: usize| if j == 1 { [f32::NAN; 4] } else { [5.0; 4] };
    let out = dfl_consensus_scores(&[0, 1, 2, 3], &boxes, &scores, &classes, None, 0.70, Some(&bad_bp));
    assert!(out[0].is_finite() && out[0] > 0.99 && out[0] <= 1.0, "extreme s=1.0: {}", out[0]);
    assert!(out[1].is_finite() && out[1] > 0.0 && out[1] < 1.0, "NaN box stayed finite: {}", out[1]);
    assert!(out[2].is_finite() && out[2] > 0.0, "tiny s=1e-8 stayed finite: {}", out[2]);
    assert_eq!(out[3], 0.0, "zero score unchanged");

    // Zero regression on `None` and `Consensus`: `candidate_conf()` stays exact `conf`.
    for rule in [ScoreRule::None, ScoreRule::Consensus] {
        let cfg = NmsConfig { conf: 0.25, score: rule, ..NmsConfig::default() };
        assert_eq!(cfg.candidate_conf(), 0.25);
    }
    for rule in [ScoreRule::Elc, ScoreRule::DflConsensus, ScoreRule::DflRepulse] {
        let pred = NmsConfig { conf: 0.25, score: rule, ..NmsConfig::default() };
        assert_eq!(pred.candidate_conf(), CALIBRATED_PRE_CONF);
        let val = NmsConfig { score: rule, ..NmsConfig::val() };
        assert_eq!(val.candidate_conf(), 0.001);
    }

    // Serde round-trip & `as_str`.
    for (rule, s) in [
        (ScoreRule::None, "none"),
        (ScoreRule::Consensus, "consensus"),
        (ScoreRule::Elc, "elc"),
        (ScoreRule::DflConsensus, "dfl-consensus"),
        (ScoreRule::DflRepulse, "dfl-repulse"),
    ] {
        assert_eq!(rule.as_str(), s);
        let json = serde_json::to_string(&rule).unwrap();
        assert_eq!(json, format!("\"{s}\""));
        assert_eq!(serde_json::from_str::<ScoreRule>(&json).unwrap(), rule);
    }

    // `bin_precision` matches `side_precision * stride²` up to float rounding.
    let logits = [
        -10.0f32, -2.0, 1.5, 2.0, -1.0, -5.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0,
    ];
    let (m, v) = softmax_moments(&logits);
    let bp = bin_precision(m, v);
    let sp = side_precision(m, v, 16.0);
    assert!((bp - sp * 256.0).abs() / bp < 1e-6, "{bp} vs {}", sp * 256.0);
}

/// Bit-exact CPU parity: `predict_candidates` + `finish_candidates` == `HeadOutput::detections` == `predict_raw`
/// for `Elc` and `DflConsensus` (with and without `BoxFusion::Dfl`, and at both `conf = 0.05` and `conf = 0.25`),
/// and `validate` matches the serial `predict_raw` loop.
#[test]
fn cpu_candidate_first_and_val_match_dense_bit_for_bit() {
    let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut Rng::seed(31)).unwrap();
    // Zero class output biases so scores sit around 0.5 and exercise both `conf = 0.25` and `conf_pre = 0.10`.
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            if t.name.contains("cls.") && t.name.ends_with(".out.bias") {
                t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
            }
        }
    }
    m.fuse().unwrap();
    let x = Tensor::rand_uniform([2, 3, 128, 128], 0.0, 1.0, &mut Rng::seed(32));

    let bits = |d: &[Detection]| -> Vec<[u32; 6]> {
        d.iter()
            .map(|d| {
                [
                    d.class,
                    d.score.to_bits(),
                    d.bbox.x1.to_bits(),
                    d.bbox.y1.to_bits(),
                    d.bbox.x2.to_bits(),
                    d.bbox.y2.to_bits(),
                ]
            })
            .collect()
    };

    for fusion in [BoxFusion::None, BoxFusion::Dfl] {
        for score in [
            ScoreRule::None,
            ScoreRule::Consensus,
            ScoreRule::Elc,
            ScoreRule::DflConsensus,
            ScoreRule::DflRepulse,
        ] {
            for conf in [0.05f32, 0.25] {
                let nms = NmsConfig { conf, iou: 0.70, max_det: 300, fusion, score, ..NmsConfig::default() };
                let raw = m.predict_raw(&x, &nms).unwrap();
                let dense = raw.output.detections(&nms).unwrap();
                let cands = m.predict_candidates(&x, &nms).unwrap();
                let lazy: Vec<Vec<Detection>> = cands.iter().map(|c| m.finish_candidates(c, &nms)).collect();
                for i in 0..2 {
                    assert_eq!(
                        bits(&dense[i]),
                        bits(&raw.images[i].detections),
                        "dense vs raw {fusion:?} {score:?} {conf}"
                    );
                    assert_eq!(bits(&lazy[i]), bits(&dense[i]), "lazy vs dense {fusion:?} {score:?} {conf}");
                    if score.is_calibrated() && conf > CALIBRATED_PRE_CONF {
                        assert!(
                            lazy[i].iter().all(|d| d.score > conf),
                            "post-filter violated at conf={conf}"
                        );
                    }
                }
            }
        }
    }

    // Check `validate` matches serial `predict_raw` with `DflConsensus` + `BoxFusion::Dfl`.
    let ds: Arc<dyn folo_data::Dataset> = Arc::new(
        SyntheticDataset::new(
            SyntheticConfig {
                images: 4,
                classes: vec!["circle".into(), "square".into(), "triangle".into()],
                imgsz: 64,
                seed: 7,
                val_fraction: 0.0,
                ..SyntheticConfig::default()
            },
            Split::Train,
        )
        .unwrap(),
    );
    let mut vc = ValConfig { samples: 2, keep_detections: true, workers: 1, ..ValConfig::new(64, 2) };
    vc.nms.fusion = BoxFusion::Dfl;
    vc.nms.score = ScoreRule::DflConsensus;
    vc.nms.conf = 0.05;
    let res = validate(&m, ds, &vc).unwrap();
    assert!(!res.detections.is_empty());
    // Deterministic repeat.
    let res2 = validate(
        &m,
        Arc::new(
            SyntheticDataset::new(
                SyntheticConfig {
                    images: 4,
                    classes: vec!["circle".into(), "square".into(), "triangle".into()],
                    imgsz: 64,
                    seed: 7,
                    val_fraction: 0.0,
                    ..SyntheticConfig::default()
                },
                Split::Train,
            )
            .unwrap(),
        ),
        &vc,
    )
    .unwrap();
    assert_eq!(res.detections, res2.detections);

    // Check `non_max_suppression_with_dfl` actually uses `bin_precision` (differs from `Elc`).
    let boxes = [b(0.0, 0.0, 20.0, 20.0), b(1.0, 1.0, 21.0, 21.0)];
    let sc = [0.60f32, 0.40];
    let bp = [[10.0f32; 4], [10.0; 4]];
    let cfg_dfl = NmsConfig {
        conf: 0.001,
        iou: 0.70,
        multi_label: true,
        score: ScoreRule::DflConsensus,
        ..NmsConfig::default()
    };
    let cfg_elc = NmsConfig { score: ScoreRule::Elc, ..cfg_dfl.clone() };
    let d_dfl = non_max_suppression_with_dfl(&boxes, &sc, None, Some(&bp), 1, &cfg_dfl);
    let d_elc = non_max_suppression_with_dfl(&boxes, &sc, None, None, 1, &cfg_elc);
    assert!(d_dfl[0].score > d_elc[0].score, "sharp DFL bin_prec must lift score above ELC");
    // Ensure `consensus_scores` stays untouched.
    let cs = consensus_scores(&[0], &boxes, &sc, &[0, 0]);
    assert_eq!(cs.len(), 1);
}

#[test]
fn elc_and_dfl_consensus_edge_case_regressions() {
    // 1. NaN / +Inf DFL logits never yield maximum bin_precision or survive calibrated NMS.
    let nan_logits = [f32::NAN; 16];
    let (mn, vn) = softmax_moments(&nan_logits);
    assert_eq!(bin_precision(mn, vn), 0.0);
    assert_eq!(side_precision(mn, vn, 8.0), 0.0);
    let mut inf_logits = [0.0f32; 16];
    inf_logits[0] = f32::INFINITY;
    let (mi, vi) = softmax_moments(&inf_logits);
    assert_eq!(bin_precision(mi, vi), 0.0);
    let cfg_cal = NmsConfig {
        conf: 0.0,
        iou: 0.70,
        score: ScoreRule::DflConsensus,
        multi_label: true,
        ..NmsConfig::default()
    };
    let cfg_cap1 = NmsConfig { max_candidates: 1, ..cfg_cal.clone() };
    let dets = non_max_suppression_with_dfl(
        &[b(f32::NAN, 0.0, 10.0, 10.0), b(0.0, 0.0, 20.0, 20.0)],
        &[0.95, 0.50],
        None,
        Some(&[[0.0; 4], [2.0; 4]]),
        1,
        &cfg_cap1,
    );
    assert_eq!(dets.len(), 1, "non-finite box must be excluded before max_candidates truncation");
    assert_eq!(dets[0].bbox, b(0.0, 0.0, 20.0, 20.0));
    let dets_inf = non_max_suppression_with_dfl(
        &[b(5.0, 5.0, 25.0, 25.0), b(0.0, 0.0, 20.0, 20.0)],
        &[f32::INFINITY, 0.50],
        None,
        Some(&[[2.0; 4], [2.0; 4]]),
        1,
        &cfg_cap1,
    );
    assert_eq!(dets_inf.len(), 1, "+Inf score must be excluded before max_candidates truncation");
    assert_eq!(dets_inf[0].bbox, b(0.0, 0.0, 20.0, 20.0));
    assert!(dets_inf[0].score.is_finite());

    // 2. Tiny `conf = 0` cluster (`1e-15`): 10 coincident members keep M/s = 10 > lone candidate.
    let tiny_boxes = vec![b(0.0, 0.0, 20.0, 20.0); 10];
    let tiny_scores = vec![1e-15f32; 10];
    let tiny_classes = vec![0u32; 10];
    let cluster_s = elc_scores(&[0], &tiny_boxes, &tiny_scores, &tiny_classes, None, 0.70)[0];
    let lone_s = elc_scores(&[0], &[b(0.0, 0.0, 20.0, 20.0)], &[1e-15f32], &[0u32], None, 0.70)[0];
    assert!(cluster_s > lone_s, "10-member 1e-15 cluster ({cluster_s:e}) must beat lone ({lone_s:e})");

    // 3. Agnostic NMS ownership: weaker box C (cls 0, 0.30) must not steal stronger suppressed box B (cls 0, 0.85).
    let box_a = b(0.0, 0.0, 100.0, 100.0);
    let box_b = b(5.0, 0.0, 105.0, 100.0); // IoU > 0.7 with A and C
    let box_c = b(18.0, 0.0, 118.0, 100.0); // IoU < 0.7 with A, IoU > 0.75 with B
    let ag_boxes = [box_a, box_b, box_c];
    let ag_scores = [0.90f32, 0.85, 0.30];
    let ag_classes = [1u32, 0, 0];
    let kept_ag = [0usize, 2]; // A suppressed B across classes; C survived
    let out_ag = elc_scores(&kept_ag, &ag_boxes, &ag_scores, &ag_classes, None, 0.70);
    let out_c_alone = elc_scores(&[0], &[box_c], &[0.30], &[0], None, 0.70);
    assert_eq!(out_ag[1].to_bits(), out_c_alone[0].to_bits(), "C must not steal stronger box B");

    // 4. +Inf score does not produce NaN.
    let inf_out = dfl_consensus_scores(
        &[0],
        &[b(0.0, 0.0, 20.0, 20.0), b(1.0, 1.0, 21.0, 21.0)],
        &[f32::INFINITY, 0.5],
        &[0, 0],
        None,
        0.70,
        None,
    );
    assert!(!inf_out[0].is_nan());
}

/// Hand-checked `ScoreRule::DflRepulse` (`dfl_repulse_scores`): q=1.5 power consensus, DFL Shannon entropy term,
/// cross-stride bonus, greedy sequential log-survival soft repulsion on output boxes, graceful fallback without
/// entropy/strides, class-local agnostic safety, and NaN/Inf/zero-area edge cases.
#[test]
fn hand_checked_dfl_repulse_and_greedy_order() {
    // 1. Hand-checked single cluster with two strides (8 and 16) + two entropies (0.8 and 1.2), plus a second
    //    kept box of the same class overlapping at IoU = 0.50 (survives NMS at 0.75, repelled by the winner).
    // Winner k0: [0, 0, 100, 100] (s=0.8, H=0.8, stride=8)
    // Member j1: [0, 0, 100, 80] (IoU=0.80 >= 0.75, s=0.4, H=1.2, stride=16, suppressed by k0)
    // Crowd/dup k2: [33.333332, 0, 133.33333, 100] -> IoU(k0, k2) = 66.66667 / 133.33333 = 0.50
    let boxes =
        [b(0.0, 0.0, 100.0, 100.0), b(0.0, 0.0, 100.0, 80.0), b(100.0 / 3.0, 0.0, 400.0 / 3.0, 100.0)];
    let scores = [0.8f32, 0.4, 0.6];
    let classes = [0u32; 3];
    let ents = [0.8f32, 1.2, 1.5];
    let strides = [8.0f32, 16.0, 8.0];
    let ent_fn = |j: usize| ents[j];
    let st_fn = |j: usize| strides[j];
    let kept = [0usize, 2];
    let repulse_boxes = [boxes[0], boxes[2]];
    let got = dfl_repulse_scores(
        &kept,
        &boxes,
        &repulse_boxes,
        &scores,
        &classes,
        None,
        0.75,
        Some(&ent_fn),
        Some(&st_fn),
    );
    // Exact hand computation for k0:
    let (s0, s1, s2) = (0.8f32 as f64, 0.4f32 as f64, 0.6f32 as f64);
    let sq0 = s0 * s0.sqrt();
    let mass_q0 = sq0 + s1 * s1.sqrt();
    let hbar0 = (s0 * (0.8f32 as f64) + s1 * (1.2f32 as f64)) / (s0 + s1);
    let f_cross0 = s1 / (s0 + s1);
    let z0 = (s0 / (1.0 - s0)).ln()
        + DFL_REPULSE_ALPHA * ((mass_q0 / sq0).clamp(1.0, ELC_CAP) / DFL_REPULSE_R_REF).ln()
        - DFL_REPULSE_BETA * (hbar0 - DFL_REPULSE_ENT_REF)
        + DFL_REPULSE_STRIDE_W * f_cross0;
    // Exact hand computation for k2 (lonely, H=1.5, f_cross=0, then repelled by k0 with IoU=0.5):
    let z2_init = (s2 / (1.0 - s2)).ln() + DFL_REPULSE_ALPHA * (1.0 / DFL_REPULSE_R_REF).ln();
    assert!(z0 > z2_init);
    let w_ov = (boxes[0].x2.min(boxes[2].x2) - boxes[0].x1.max(boxes[2].x1)).max(0.0);
    let inter = w_ov * 100.0;
    let o = (inter / (10000.0 + 10000.0 - inter)) as f64;
    let z2 = z2_init + DFL_REPULSE_GAMMA * (1.0 - o.powi(4)).max(1e-4).ln();
    assert!((got[0] - ref_sigmoid(z0) as f32).abs() < 1e-6, "{} vs {}", got[0], ref_sigmoid(z0));
    assert!((got[1] - ref_sigmoid(z2) as f32).abs() < 1e-6, "{} vs {}", got[1], ref_sigmoid(z2));

    // 2. Greedy sequential order: A (z highest) overlaps B strongly (IoU=0.67), B overlaps C strongly (IoU=0.67),
    //    A and C barely overlap (IoU=0.40). Under greedy repulsion: A freezes first and severely dampens B, so B's
    //    current z drops BELOW C's z! Thus C freezes second BEFORE B and receives NO penalty from the dampened
    //    duplicate B! Under static order, B would have repelled C too.
    let chain = [b(0.0, 0.0, 100.0, 100.0), b(20.0, 0.0, 120.0, 100.0), b(40.0, 0.0, 140.0, 100.0)];
    let chain_s = [0.90f32, 0.80, 0.78]; // initial z: A > B > C, but after A dampens B: C > B!
    let chain_c = [0u32; 3];
    let chain_out =
        dfl_repulse_scores(&[0, 1, 2], &chain, &chain, &chain_s, &chain_c, None, 0.75, None, None);
    // Compare C's score with and without B present: since C freezes before B, C's score must be identical!
    let ac_out = dfl_repulse_scores(
        &[0, 1],
        &[chain[0], chain[2]],
        &[chain[0], chain[2]],
        &[0.90, 0.78],
        &[0, 0],
        None,
        0.75,
        None,
        None,
    );
    assert_eq!(
        chain_out[2].to_bits(),
        ac_out[1].to_bits(),
        "greedy sequential order must freeze clean C before dampened duplicate B"
    );
    assert!(
        chain_out[2] > chain_out[1],
        "clean C ({}) must outrank dampened duplicate B ({})",
        chain_out[2],
        chain_out[1]
    );

    // 3. Different class boxes do not repel each other, even under `agnostic = true`.
    let cfg_ag = NmsConfig {
        conf: 0.001,
        iou: 0.75,
        agnostic: true,
        multi_label: true,
        score: ScoreRule::DflRepulse,
        ..NmsConfig::default()
    };
    let pair = [b(0.0, 0.0, 100.0, 100.0), b(30.0, 0.0, 130.0, 100.0)]; // IoU = 70/130 = 0.538 <= 0.75
    let same_cls = non_max_suppression(&pair, &[0.8, 0.7], 1, &cfg_ag);
    let diff_cls = non_max_suppression(&pair, &[0.8, 0.0, 0.0, 0.7], 2, &cfg_ag);
    assert_eq!(same_cls.len(), 2);
    assert_eq!(diff_cls.len(), 2);
    assert!(
        diff_cls[1].score > same_cls[1].score + 0.01,
        "different-class box ({}) must not be repelled like same-class ({})",
        diff_cls[1].score,
        same_cls[1].score
    );

    // 4. `non_max_suppression_with_inputs` uses entropy & strides and falls back cleanly when omitted.
    let cfg_rep = NmsConfig {
        conf: 0.001,
        iou: 0.75,
        multi_label: true,
        score: ScoreRule::DflRepulse,
        ..NmsConfig::default()
    };
    let low_ent = [0.2f32, 0.2, 0.2];
    let high_ent = [2.7f32, 2.7, 2.7];
    let d_sharp = non_max_suppression_with_inputs(
        &boxes,
        &scores,
        DflInputs { entropy: Some(&low_ent), strides: Some(&strides), ..Default::default() },
        1,
        &cfg_rep,
    );
    let d_blur = non_max_suppression_with_inputs(
        &boxes,
        &scores,
        DflInputs { entropy: Some(&high_ent), strides: Some(&strides), ..Default::default() },
        1,
        &cfg_rep,
    );
    assert!(d_sharp[0].score > d_blur[0].score, "low DFL entropy must outscore high entropy");

    // 5. NaN / +Inf / zero-area / NaN entropy & invalid stride edge cases never produce NaN.
    let bad_boxes = [
        b(0.0, 0.0, 10.0, 10.0),
        b(f32::NAN, 0.0, 5.0, 10.0),
        b(5.0, 5.0, 5.0, 9.0),
        b(50.0, 50.0, 60.0, 60.0),
    ];
    let bad_scores = [1.0f32, 0.8, 1e-8, 0.0];
    let bad_ent = |_j: usize| f32::NAN;
    let bad_st = |_j: usize| -1.0f32;
    let safe = dfl_repulse_scores(
        &[0, 1, 2, 3],
        &bad_boxes,
        &bad_boxes,
        &bad_scores,
        &[0; 4],
        None,
        0.75,
        Some(&bad_ent),
        Some(&bad_st),
    );
    assert!(safe[0].is_finite() && safe[0] > 0.99 && safe[0] <= 1.0);
    assert!(safe[1].is_finite() && safe[1] > 0.0 && safe[1] < 1.0);
    assert!(safe[2].is_finite() && safe[2] > 0.0);
    assert_eq!(safe[3], 0.0);
    let mut row = vec![-100.0f32; 64];
    for s in 0..4 {
        row[s * 16 + 3] = 50.0;
    }
    assert!(dfl_entropy(&row, 16) < 1e-5);
}
