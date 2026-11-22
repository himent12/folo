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
use folo_vision::decode::{bin_precision, side_precision, softmax_moments};
use folo_vision::fusion::{
    BoxFusion, DFL_CONSENSUS_BETA, DFL_CONSENSUS_SIG_REF, ELC_ALPHA, ELC_CAP, ELC_R_REF, ScoreRule,
    consensus_scores, dfl_consensus_scores, elc_scores,
};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::{
    CALIBRATED_PRE_CONF, Detection, NmsConfig, nms_detections, non_max_suppression,
    non_max_suppression_with_dfl,
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
    for rule in [ScoreRule::Elc, ScoreRule::DflConsensus] {
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
        for score in [ScoreRule::None, ScoreRule::Consensus, ScoreRule::Elc, ScoreRule::DflConsensus] {
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
