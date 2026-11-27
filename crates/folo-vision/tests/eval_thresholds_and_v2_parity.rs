//! Integration tests for:
//! 1. Global & per-class F1-optimal confidence threshold calibration in [`Evaluator`] / [`ValResult`].
//! 2. Per-class confidence threshold (`per_class_conf`) filtering across NMS, Soft-NMS, Matrix-NMS, calibrated
//!    `ScoreRule`, `o2o` top-k, dense vs candidate-first (`FoloModel` & `FoloV2Model`), and [`Predictor`].
//! 3. Complete `folo2` (`f32`, `int8`, `o2m`, `o2o`, `iou_head`) inference parity across `predict`,
//!    `predict_batch`, SAHI (`predict_tiled`), TTA (`predict_tta`), ByteTrack (`predict_sequence`), and calibrated
//!    per-class thresholds.

use std::collections::BTreeMap;
use std::sync::Arc;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{Image, Split};
use folo_nn::Module;
use folo_vision::boxes::Xyxy;
use folo_vision::fusion::ScoreRule;
use folo_vision::metrics::{EvalReport, Evaluator, Truth};
use folo_vision::model::HeadMode;
use folo_vision::model::quant::calib::{Method, calibrate_v2};
use folo_vision::model::v2::DEFAULT_IOU_FUSION_ALPHA;
use folo_vision::nms::{
    Candidate, Detection, NmsConfig, NmsKind, nms_candidates, nms_detections, non_max_suppression,
    non_max_suppression_pairs, non_max_suppression_rows, topk_candidates, topk_detections,
};
use folo_vision::tile::{TileConfig, TileMergeMetric};
use folo_vision::track::TrackConfig;
use folo_vision::tta::{PredictOptions, TtaConfig};
use folo_vision::val::{ValConfig, check_config_detector, validate_detector, validate_v2};
use folo_vision::{
    Detector, FoloModel, FoloV2Config, FoloV2Model, ModelConfig, OperatingPoint, PredictConfig, Predictor,
    ThresholdCalibration,
};

fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class: u32) -> Detection {
    Detection { bbox: Xyxy::new(x1, y1, x2, y2), score, class }
}

fn truth(x1: f32, y1: f32, x2: f32, y2: f32, class: u32) -> Truth {
    Truth { bbox: Xyxy::new(x1, y1, x2, y2), class }
}

fn pattern_image(w: u32, h: u32, phase: u32) -> Image {
    let mut data = vec![0u8; w as usize * h as usize * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y as usize * w as usize + x as usize) * 3;
            data[i] =
                ((x.wrapping_mul(7).wrapping_add(y.wrapping_mul(3)).wrapping_add(phase * 19)) & 255) as u8;
            data[i + 1] =
                ((x.wrapping_mul(5).wrapping_add(y.wrapping_mul(11)).wrapping_add(phase * 31)) & 255) as u8;
            data[i + 2] =
                ((x.wrapping_mul(13).wrapping_add(y.wrapping_mul(2)).wrapping_add(phase * 17)) & 255) as u8;
        }
    }
    Image { width: w, height: h, data }
}

/// Set per-level class biases on a v1 model so different levels activate different classes above and below
/// per-class thresholds.
fn boost_v1_cls_biases(model: &mut FoloModel, per_level: &[[f32; 3]; 3]) {
    let _g = no_grad();
    let apply = |bias_opt: &Option<Tensor>, vals: &[f32; 3]| {
        if let Some(b) = bias_opt {
            let src = Tensor::from_vec(vals.to_vec(), [3]).unwrap();
            b.copy_from(&src).unwrap();
        }
    };
    for (lv, vals) in model.head.levels.iter().zip(per_level) {
        apply(&lv.cls_branch.out.bias, vals);
    }
    if let Some(o2o) = &model.head.o2o {
        for (lv, vals) in o2o.iter().zip(per_level) {
            apply(&lv.cls_branch.out.bias, vals);
        }
    }
}

/// Set distinct class bias offsets on a v2 model so multiple classes emit scores spanning `[0.15, 0.85]`.
fn boost_v2_cls_biases(model: &mut FoloV2Model, biases: &[f32]) {
    let _g = no_grad();
    let apply = |bias_opt: &Option<Tensor>| {
        if let Some(b) = bias_opt {
            let src = Tensor::from_vec(biases.to_vec(), [biases.len()]).unwrap();
            b.copy_from(&src).unwrap();
        }
    };
    for lv in &model.head.levels {
        apply(&lv.cls_branch.out.bias);
    }
    if let Some(o2o) = &model.head.o2o_cls {
        for lv in o2o {
            apply(&lv.cls_branch.out.bias);
        }
    }
}

fn build_v2_model(seed: u64, nc: usize, head: HeadMode, iou_head: bool) -> FoloV2Model {
    let mut cfg = FoloV2Config::new("n", nc).unwrap();
    cfg.head = head;
    cfg.iou_head = iou_head;
    let mut m = FoloV2Model::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    m.eval();
    m.fuse().unwrap();
    // Spread class logits across a wide range so per-class thresholding exercises both kept & dropped classes.
    let biases: Vec<f32> = (0..nc).map(|c| -0.6 + 0.55 * c as f32).collect();
    boost_v2_cls_biases(&mut m, &biases);
    m
}

fn det_bits(d: &Detection) -> (u32, u32, [u32; 4]) {
    (d.class, d.score.to_bits(), [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2].map(f32::to_bits))
}

#[test]
fn evaluator_per_class_and_global_f1_threshold_calibration_exact() {
    // 4-class problem with intentionally divergent per-class score distributions:
    // - Class 0 (2 truths): TPs at 0.30 and 0.25, FP at 0.15 -> optimal F1 = 1.0 on [0.151..=0.250],
    //   highest-threshold tie-break picks best_conf = 0.250.
    // - Class 1 (2 truths): TPs at 0.90 and 0.75, FPs at 0.65, 0.55, 0.45 -> optimal F1 = 1.0 on
    //   [0.651..=0.750], highest-threshold tie-break picks best_conf = 0.750.
    // - Class 2 (1 truth): only an FP at 0.60, zero TPs -> F1 = 0 everywhere, highest-threshold tie-break
    //   picks best_conf = 1.000.
    // - Class 3 (0 truths): omitted from per_class; threshold_for(3) falls back to global.best_conf.
    let img0_truths = vec![
        truth(10.0, 10.0, 40.0, 40.0, 0),
        truth(50.0, 10.0, 80.0, 40.0, 1),
        truth(90.0, 10.0, 120.0, 40.0, 2),
    ];
    let img0_preds = vec![
        det(10.0, 10.0, 40.0, 40.0, 0.30, 0),
        det(200.0, 200.0, 220.0, 220.0, 0.15, 0),
        det(50.0, 10.0, 80.0, 40.0, 0.90, 1),
        det(230.0, 200.0, 250.0, 220.0, 0.65, 1),
        det(260.0, 200.0, 280.0, 220.0, 0.55, 1),
        det(290.0, 200.0, 310.0, 220.0, 0.60, 2),
    ];
    let img1_truths = vec![truth(10.0, 50.0, 40.0, 80.0, 0), truth(50.0, 50.0, 80.0, 80.0, 1)];
    let img1_preds = vec![
        det(10.0, 50.0, 40.0, 80.0, 0.25, 0),
        det(50.0, 50.0, 80.0, 80.0, 0.75, 1),
        det(200.0, 240.0, 220.0, 260.0, 0.45, 1),
    ];

    let mut ev = Evaluator::new(4);
    ev.add_image(&img0_preds, &img0_truths);
    ev.add_image(&img1_preds, &img1_truths);
    let report = ev.compute();

    // Parallel merge parity: splitting images across two evaluators must yield bit-exact calibration.
    let mut ev_a = Evaluator::new(4);
    ev_a.add_image(&img0_preds, &img0_truths);
    let mut ev_b = Evaluator::new(4);
    ev_b.add_image(&img1_preds, &img1_truths);
    ev_a.merge(ev_b);
    assert_eq!(ev_a.compute().thresholds, report.thresholds);

    let cal = report.calibrated_thresholds();
    assert_eq!(cal, &report.thresholds);
    assert!(cal.validate().is_ok());

    // Global operating point matches EvalReport top-level fields exactly.
    assert_eq!(cal.global.best_conf, report.best_conf);
    assert_eq!(cal.global.best_f1, report.f1);
    assert_eq!(cal.global.precision, report.precision);
    assert_eq!(cal.global.recall, report.recall);

    // Three classes had truths (0, 1, 2); class 3 had none.
    assert_eq!(cal.per_class.len(), 3);
    assert_eq!(cal.per_class[0].class, 0);
    assert!((cal.per_class[0].best_conf - 0.25).abs() < 1e-6, "got {}", cal.per_class[0].best_conf);
    assert!((cal.per_class[0].best_f1 - 1.0).abs() < 1e-6);
    assert!((cal.per_class[0].precision - 1.0).abs() < 1e-6);
    assert!((cal.per_class[0].recall - 1.0).abs() < 1e-6);
    assert_eq!((cal.per_class[0].truths, cal.per_class[0].preds), (2, 3));

    assert_eq!(cal.per_class[1].class, 1);
    assert!((cal.per_class[1].best_conf - 0.75).abs() < 1e-6, "got {}", cal.per_class[1].best_conf);
    assert!((cal.per_class[1].best_f1 - 1.0).abs() < 1e-6);
    assert!((cal.per_class[1].precision - 1.0).abs() < 1e-6);
    assert!((cal.per_class[1].recall - 1.0).abs() < 1e-6);
    assert_eq!((cal.per_class[1].truths, cal.per_class[1].preds), (2, 5));

    assert_eq!(cal.per_class[2].class, 2);
    assert!((cal.per_class[2].best_conf - 1.0).abs() < 1e-6, "zero-TP class ties at 1.0");
    assert_eq!(cal.per_class[2].best_f1, 0.0);
    assert_eq!(cal.per_class[2].precision, 0.0);
    assert_eq!(cal.per_class[2].recall, 0.0);
    assert_eq!((cal.per_class[2].truths, cal.per_class[2].preds), (1, 1));

    // Per-class macro F1 = (1.0 + 1.0 + 0.0) / 3 = 2/3, strictly exceeding single-threshold global F1!
    assert!((cal.macro_f1 - (2.0 / 3.0)).abs() < 1e-6, "macro_f1={}", cal.macro_f1);
    assert!(
        cal.macro_f1 > cal.global.best_f1 + 0.05,
        "per-class thresholds must beat single global threshold: macro={} vs global={}",
        cal.macro_f1,
        cal.global.best_f1
    );

    // Lookup & fallback behavior.
    assert!((cal.threshold_for(0) - 0.25).abs() < 1e-6);
    assert!((cal.threshold_for(1) - 0.75).abs() < 1e-6);
    assert!((cal.threshold_for(2) - 1.0).abs() < 1e-6);
    assert_eq!(cal.threshold_for(3), cal.global.best_conf);
    let map = cal.per_class_map();
    assert_eq!(map.len(), 3);
    assert!(!map.contains_key(&3));

    // Empty evaluator yields default empty calibration.
    let empty_report = Evaluator::new(4).compute();
    assert!(empty_report.thresholds.is_empty());
    assert_eq!(empty_report.thresholds.global, OperatingPoint::default());
    assert_eq!(empty_report.thresholds.macro_f1, 0.0);

    // Serde JSON round-trip & backward compatibility when `thresholds` is missing.
    let json = serde_json::to_string(&report).unwrap();
    let back: EvalReport = serde_json::from_str(&json).unwrap();
    assert_eq!(back, report);
    let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
    val.as_object_mut().unwrap().remove("thresholds");
    let legacy: EvalReport = serde_json::from_value(val).unwrap();
    assert_eq!(legacy.thresholds, ThresholdCalibration::default());
}

#[test]
fn nms_and_topk_per_class_conf_all_modes_and_variants() {
    let mut map = BTreeMap::new();
    map.insert(0, 0.70);
    map.insert(1, 0.20);
    // Class 2 falls back to cfg.conf = 0.45.

    let boxes = vec![
        Xyxy::new(0.0, 0.0, 20.0, 20.0), // anchor 0: class 0 = 0.65 (dropped), class 1 = 0.25 (kept in multi-label)
        Xyxy::new(30.0, 0.0, 50.0, 20.0), // anchor 1: class 0 = 0.75 (kept)
        Xyxy::new(60.0, 0.0, 80.0, 20.0), // anchor 2: class 1 = 0.22 (kept)
        Xyxy::new(90.0, 0.0, 110.0, 20.0), // anchor 3: class 2 = 0.40 (dropped by fallback 0.45)
        Xyxy::new(120.0, 0.0, 140.0, 20.0), // anchor 4: class 2 = 0.50 (kept by fallback 0.45)
    ];
    let scores = vec![
        0.65, 0.25, 0.05, // a0
        0.75, 0.05, 0.05, // a1
        0.05, 0.22, 0.05, // a2
        0.05, 0.05, 0.40, // a3
        0.05, 0.05, 0.50, // a4
    ];
    let mut rows = Vec::with_capacity(boxes.len() * 7);
    for (i, b) in boxes.iter().enumerate() {
        rows.extend_from_slice(&[b.x1, b.y1, b.x2, b.y2]);
        rows.extend_from_slice(&scores[i * 3..(i + 1) * 3]);
    }

    // 1. Single-label standard NMS: a0 best is (cls 0, 0.65 <= 0.70) -> dropped; keeps a1(0), a4(2), a2(1).
    let cfg = NmsConfig { conf: 0.45, per_class_conf: Some(map.clone()), ..NmsConfig::default() };
    let out = non_max_suppression(&boxes, &scores, 3, &cfg);
    assert_eq!(out, non_max_suppression_rows(&rows, 3, &cfg));
    let cls_scores: Vec<(u32, f32)> = out.iter().map(|d| (d.class, d.score)).collect();
    assert_eq!(cls_scores, vec![(0, 0.75), (2, 0.50), (1, 0.22)]);

    // `nms_detections` & `nms_candidates` agree identically on single-label candidates extracted at min_conf.
    let single_cands: Vec<Candidate> = vec![
        Candidate { anchor: 0, class: 0, score: 0.65, bbox: boxes[0] },
        Candidate { anchor: 1, class: 0, score: 0.75, bbox: boxes[1] },
        Candidate { anchor: 2, class: 1, score: 0.22, bbox: boxes[2] },
        Candidate { anchor: 3, class: 2, score: 0.40, bbox: boxes[3] },
        Candidate { anchor: 4, class: 2, score: 0.50, bbox: boxes[4] },
    ];
    assert_eq!(nms_candidates(&single_cands, &cfg), out);
    let raw_dets: Vec<Detection> =
        single_cands.iter().map(|c| Detection { bbox: c.bbox, score: c.score, class: c.class }).collect();
    assert_eq!(nms_detections(&raw_dets, &cfg), out);

    // 2. Multi-label NMS + pairs path: a0 contributes (cls 1, 0.25) even though cls 0 (0.65) failed 0.70!
    let cfg_ml = NmsConfig { multi_label: true, ..cfg.clone() };
    let out_ml = non_max_suppression(&boxes, &scores, 3, &cfg_ml);
    let ml_floor = cfg_ml.candidate_conf();
    let pairs: Vec<(u32, u32, f32)> = (0..boxes.len())
        .flat_map(|a| {
            let s = &scores;
            (0..3).filter_map(move |c| {
                let v = s[a * 3 + c];
                (v > ml_floor).then_some((a as u32, c as u32, v))
            })
        })
        .collect();
    assert_eq!(non_max_suppression_pairs(pairs, |a| boxes[a as usize], &cfg_ml), out_ml);
    let ml_cls: Vec<(u32, f32)> = out_ml.iter().map(|d| (d.class, d.score)).collect();
    assert_eq!(ml_cls, vec![(0, 0.75), (2, 0.50), (1, 0.25), (1, 0.22)]);

    // 3. O2o topk_detections (single-label & multi-label) and topk_candidates.
    // Under MODEL.md §6.6, multi-label top-k selects `k = min(max_det, A) = 5` raw (anchor, class) pairs first
    // (0.75, 0.65, 0.50, 0.40, 0.25 — excluding the 6th pair 0.22) and then applies per-class thresholds.
    assert_eq!(topk_detections(&boxes, &scores, 3, &cfg), out);
    assert_eq!(topk_candidates(&single_cands, &cfg), out);
    let topk_ml: Vec<(u32, f32)> =
        topk_detections(&boxes, &scores, 3, &cfg_ml).iter().map(|d| (d.class, d.score)).collect();
    assert_eq!(topk_ml, vec![(0, 0.75), (2, 0.50), (1, 0.25)]);

    // 4. Soft-NMS (Linear/Gaussian) & Matrix-NMS (Linear/Gaussian): class-aware & agnostic decayed threshold.
    // Two heavily overlapping boxes (IoU = 9/11 ≈ 0.818) per class:
    // - Class 0 (thr 0.50): 0.90 and 0.80 -> decayed score of second is ~0.80 * (2/11) ≈ 0.145 <= 0.50 -> dropped.
    // - Class 1 (thr 0.10): 0.90 and 0.80 -> decayed score of second is ~0.145 > 0.10 -> kept!
    let mut soft_map = BTreeMap::new();
    soft_map.insert(0, 0.50);
    soft_map.insert(1, 0.10);
    let b_hi = Xyxy::new(0.0, 0.0, 10.0, 10.0);
    let b_lo = Xyxy::new(1.0, 0.0, 11.0, 10.0);
    let soft_dets = vec![
        Detection { bbox: b_hi, score: 0.90, class: 0 },
        Detection { bbox: b_lo, score: 0.80, class: 0 },
        Detection { bbox: b_hi, score: 0.90, class: 1 },
        Detection { bbox: b_lo, score: 0.80, class: 1 },
    ];
    for kind in [
        NmsKind::SoftLinear,
        NmsKind::SoftGaussian { sigma: 0.5 },
        NmsKind::MatrixLinear,
        NmsKind::MatrixGaussian { sigma: 0.5 },
    ] {
        let scfg = NmsConfig {
            conf: 0.25,
            iou: 0.5,
            nms_kind: kind,
            per_class_conf: Some(soft_map.clone()),
            ..NmsConfig::default()
        };
        let kept = nms_detections(&soft_dets, &scfg);
        let c0_cnt = kept.iter().filter(|d| d.class == 0).count();
        let c1_cnt = kept.iter().filter(|d| d.class == 1).count();
        assert_eq!(c0_cnt, 1, "{kind:?}: class 0 decayed duplicate must be dropped at thr 0.50");
        assert_eq!(c1_cnt, 2, "{kind:?}: class 1 decayed duplicate must survive at thr 0.10");

        // Agnostic: class 0 box `b_hi` (0.95) suppresses overlapping class 0 `b_lo` (0.85, thr 0.50) AND
        // class 1 `b_lo2` (0.80, thr 0.10). Because `IoU(b_lo, b_lo2) = 81/119 < 9/11 = IoU(b_hi, b_lo)`, both
        // Soft-NMS and Matrix-NMS decay `b_lo2` by `2/11` (~0.145 > 0.10 -> kept) while dropping `b_lo`
        // (~0.155 <= 0.50).
        let b_lo2 = Xyxy::new(0.0, 1.0, 10.0, 11.0);
        let ag_dets = vec![
            Detection { bbox: b_hi, score: 0.95, class: 0 },
            Detection { bbox: b_lo, score: 0.85, class: 0 },
            Detection { bbox: b_lo2, score: 0.80, class: 1 },
        ];
        let ag_cfg = NmsConfig { agnostic: true, ..scfg.clone() };
        let ag_kept = nms_detections(&ag_dets, &ag_cfg);
        assert!(
            ag_kept.iter().all(|d| d.score > ag_cfg.conf_for_class(d.class)),
            "{kind:?} agnostic must respect every class's own threshold: {ag_kept:?}"
        );
        assert!(
            ag_kept.iter().any(|d| d.class == 1),
            "{kind:?} agnostic must keep class 1 above its 0.10 threshold: {ag_kept:?}"
        );
        assert_eq!(
            ag_kept.iter().filter(|d| d.class == 0).count(),
            1,
            "{kind:?} agnostic must drop decayed class 0 below 0.50: {ag_kept:?}"
        );
    }

    // 5. Calibrated ScoreRule::Elc with mixed thresholds across CALIBRATED_PRE_CONF (0.10):
    // Class 0 has conf = 0.05 (<= 0.10 -> no post-filter; calibrated score below 0.05 still kept),
    // Class 1 has conf = 0.50 (> 0.10 -> pre-NMS floor 0.10, post-filtered at > 0.50).
    let mut elc_map = BTreeMap::new();
    elc_map.insert(0, 0.05);
    elc_map.insert(1, 0.50);
    let elc_cfg = NmsConfig {
        conf: 0.25,
        score: ScoreRule::Elc,
        per_class_conf: Some(elc_map),
        ..NmsConfig::default()
    };
    assert_eq!(elc_cfg.candidate_conf_for_class(0), 0.05);
    assert_eq!(elc_cfg.candidate_conf_for_class(1), 0.10);
    assert_eq!(elc_cfg.candidate_conf(), 0.05);
    let elc_dets = vec![
        Detection { bbox: Xyxy::new(0.0, 0.0, 20.0, 20.0), score: 0.08, class: 0 }, // isolated -> ELC < 0.08, still kept
        Detection { bbox: Xyxy::new(50.0, 0.0, 70.0, 20.0), score: 0.40, class: 1 }, // isolated -> ELC < 0.40 <= 0.50, dropped
        Detection { bbox: Xyxy::new(100.0, 0.0, 120.0, 20.0), score: 0.90, class: 1 }, // high -> survives > 0.50
    ];
    let elc_kept = nms_detections(&elc_dets, &elc_cfg);
    let elc_classes: Vec<u32> = elc_kept.iter().map(|d| d.class).collect();
    assert_eq!(elc_classes, vec![1, 0], "ELC must keep class 0 (no post-filter) and high class 1 only");

    // 6. NmsConfig serde JSON round-trip with and without per_class_conf.
    let j_default = serde_json::to_string(&NmsConfig::default()).unwrap();
    assert!(!j_default.contains("per_class_conf"), "None must be skipped in JSON");
    let j_map = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<NmsConfig>(&j_map).unwrap(), cfg);
}

#[test]
fn dense_vs_candidate_first_per_class_conf_bit_exact_v1_and_v2() {
    let mut per_class = BTreeMap::new();
    per_class.insert(0, 0.55);
    per_class.insert(1, 0.25);
    let nms = NmsConfig {
        conf: 0.40,
        iou: 0.65,
        max_det: 20,
        per_class_conf: Some(per_class),
        ..NmsConfig::default()
    };

    let x = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(99));

    // FOLO v1: o2m & o2o
    for mode in [HeadMode::O2m, HeadMode::O2o] {
        let mut cfg = ModelConfig::new("n", 3).unwrap();
        cfg.head = mode.as_str().into();
        let mut m = FoloModel::with_rng(cfg, &mut Rng::seed(101)).unwrap();
        m.eval();
        m.fuse().unwrap();
        // Level 0 (P3): class 0 ~ 0.45 (dropped at 0.55); Level 1 (P4): class 1 ~ 0.45 (kept at 0.25);
        // Level 2 (P5): class 2 ~ 0.62 (kept at fallback 0.40) and class 0 ~ 0.65 (kept at 0.55).
        boost_v1_cls_biases(&mut m, &[[-0.2, -1.5, -1.5], [-1.5, -0.2, -1.5], [0.6, -1.5, 0.5]]);
        let _g = no_grad();
        let dense = m.forward(&x).unwrap().detections(&nms).unwrap();
        let lazy_cands = m.predict_candidates(&x, &nms).unwrap();
        let lazy: Vec<Vec<Detection>> = lazy_cands.iter().map(|c| m.finish_candidates(c, &nms)).collect();
        for (i, (d, l)) in dense.iter().zip(&lazy).enumerate() {
            let db: Vec<_> = d.iter().map(det_bits).collect();
            let lb: Vec<_> = l.iter().map(det_bits).collect();
            assert_eq!(db, lb, "v1 {mode:?} image {i}: dense vs candidate-first mismatch");
            assert!(
                d.iter().all(|det| det.score > nms.conf_for_class(det.class)),
                "v1 {mode:?} image {i}: detection violated per-class threshold"
            );
        }
        assert!(dense.iter().any(|v| !v.is_empty()), "v1 {mode:?} test should produce non-empty detections");
    }

    // FOLO v2: o2m & o2o
    for mode in [HeadMode::O2m, HeadMode::O2o] {
        let m = build_v2_model(202, 3, mode, false);
        let _g = no_grad();
        let dense = m.forward(&x).unwrap().detections(&nms, DEFAULT_IOU_FUSION_ALPHA).unwrap();
        let lazy_cands = m.predict_candidates(&x, &nms).unwrap();
        let lazy: Vec<Vec<Detection>> = lazy_cands.iter().map(|c| m.finish_candidates(c, &nms)).collect();
        for (i, (d, l)) in dense.iter().zip(&lazy).enumerate() {
            let db: Vec<_> = d.iter().map(det_bits).collect();
            let lb: Vec<_> = l.iter().map(det_bits).collect();
            assert_eq!(db, lb, "v2 {mode:?} image {i}: dense vs candidate-first mismatch");
            assert!(
                d.iter().all(|det| det.score > nms.conf_for_class(det.class)),
                "v2 {mode:?} image {i}: detection violated per-class threshold"
            );
        }
        assert!(dense.iter().any(|v| !v.is_empty()), "v2 {mode:?} test should produce non-empty detections");
    }
}

#[test]
fn val_calibration_feeds_into_predictor_and_rejects_invalid_thresholds() {
    let ds_cfg = SyntheticConfig {
        images: 4,
        classes: vec!["circle".into(), "square".into(), "triangle".into()],
        imgsz: 64,
        seed: 11,
        min_objects: 1,
        max_objects: 3,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 0.5,
        ..SyntheticConfig::default()
    };
    let ds = Arc::new(SyntheticDataset::new(ds_cfg, Split::Val).unwrap());
    let model = build_v2_model(303, 3, HeadMode::O2m, true);

    let mut val_cfg = ValConfig::new(64, 2);
    val_cfg.workers = 1;
    let res = validate_v2(&model, ds.clone(), &val_cfg).unwrap();
    let det_res =
        validate_detector(&Detector::from(model.fused_eval_clone().unwrap()), ds.clone(), &val_cfg).unwrap();
    assert_eq!(res.thresholds(), det_res.thresholds());
    assert_eq!(res.thresholds(), &res.report.thresholds);
    assert_eq!(res.per_class_conf(), res.thresholds().per_class_map());
    assert!(!res.thresholds().is_empty());
    assert!(res.thresholds().validate().is_ok());

    // Invalid per_class_conf rejected by ValConfig check_config_detector and Predictor APIs.
    let detector = Detector::from(model.fused_eval_clone().unwrap());
    for bad in [f32::NAN, -0.05, 1.05, f32::INFINITY] {
        let mut bad_map = BTreeMap::new();
        bad_map.insert(0, bad);
        let mut bad_val = val_cfg.clone();
        bad_val.nms.per_class_conf = Some(bad_map.clone());
        assert!(check_config_detector(&detector, 3, &bad_val).is_err());

        let mut pred = Predictor::new_v2(
            model.fused_eval_clone().unwrap(),
            PredictConfig { imgsz: 64, ..Default::default() },
        )
        .unwrap();
        assert!(pred.set_per_class_conf(Some(bad_map.clone())).is_err());
        assert!(
            Predictor::new_v2(
                model.fused_eval_clone().unwrap(),
                PredictConfig { imgsz: 64, ..Default::default() },
            )
            .unwrap()
            .with_per_class_conf(bad_map)
            .is_err()
        );
    }

    // Feed ValResult calibration straight into Predictor via builder & mutator.
    let mut predictor =
        Predictor::new_v2(model, PredictConfig { imgsz: 64, conf: 0.30, ..Default::default() })
            .unwrap()
            .with_calibrated_thresholds(res.thresholds())
            .unwrap();
    assert_eq!(predictor.config().conf, res.thresholds().global.best_conf);
    assert_eq!(predictor.per_class_conf(), Some(&res.per_class_conf()));

    // Empty calibration leaves `cfg.conf` untouched and clears per-class overrides.
    let saved_conf = predictor.config().conf;
    predictor.apply_calibrated_thresholds(&ThresholdCalibration::default()).unwrap();
    assert_eq!(predictor.config().conf, saved_conf);
    assert!(predictor.per_class_conf().is_none());
}

#[test]
fn folo2_complete_inference_parity_f32_int8_o2m_o2o_iou_head_sahi_tta_bytetrack_and_per_class_conf() {
    let img_small = pattern_image(64, 64, 1);
    let img_large = pattern_image(96, 80, 2);
    let seq_frames = vec![pattern_image(64, 64, 1), pattern_image(64, 64, 2), pattern_image(64, 64, 3)];

    let mut per_class = BTreeMap::new();
    per_class.insert(0, 0.72);
    per_class.insert(1, 0.20);
    per_class.insert(2, 0.35);

    let cases: [(&str, HeadMode, bool, bool); 5] = [
        ("f32_o2m", HeadMode::O2m, false, false),
        ("f32_o2o", HeadMode::O2o, false, false),
        ("f32_o2m_iou_head", HeadMode::O2m, true, false),
        ("int8_o2m", HeadMode::O2m, false, true),
        ("int8_o2o", HeadMode::O2o, false, true),
    ];

    for (label, head_mode, iou_head, use_int8) in cases {
        let mut model = build_v2_model(505, 3, head_mode, iou_head);
        if use_int8 {
            let calib_input = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(606));
            let cal =
                calibrate_v2(&model, || std::iter::once(Ok(calib_input.clone())), Method::MinMax).unwrap();
            model.set_int8(Some(cal)).unwrap();
        }
        assert_eq!(model.int8().is_some(), use_int8, "{label}");

        let base_cfg = PredictConfig {
            imgsz: 64,
            conf: 0.40,
            iou: 0.65,
            max_det: 50,
            rect: true,
            ..PredictConfig::default()
        };
        let pred = Predictor::new_v2(model.fused_eval_clone().unwrap(), base_cfg.clone())
            .unwrap()
            .with_per_class_conf(per_class.clone())
            .unwrap();
        assert_eq!(pred.device(), Device::Cpu);
        assert_eq!(pred.detector().int8().is_some(), use_int8, "{label}");

        let check_thresholds = |dets: &[Detection], ctx: &str| {
            for d in dets {
                let thr = per_class.get(&d.class).copied().unwrap_or(base_cfg.conf);
                assert!(
                    d.score > thr,
                    "{label} ({ctx}): class {} score {} <= per-class threshold {thr}",
                    d.class,
                    d.score
                );
            }
        };

        // 1. Single & batch parity + manual prepare/forward/finish parity.
        let p_single = pred.predict(&img_small).unwrap();
        check_thresholds(&p_single.detections, "predict");
        let p_batch = pred.predict_batch(&[img_small.clone(), img_small.clone()]).unwrap();
        assert_eq!(p_batch[0].detections, p_single.detections, "{label}: batch vs single");
        assert_eq!(p_batch[1].detections, p_single.detections, "{label}: batch[1] vs single");

        // Manual prepare -> forward_detector -> decode_images -> finish matches `predict` (same per-class thresholds).
        if !use_int8 {
            let batch = pred.prepare(std::slice::from_ref(&img_small), true).unwrap();
            let out = pred.forward_detector(&batch).unwrap();
            let decoded = out.decode_images(DEFAULT_IOU_FUSION_ALPHA).unwrap();
            let finished = pred.finish(&batch, decoded, 3);
            assert_eq!(finished[0], p_single.detections, "{label}: manual prepare/finish parity");
        }

        // 2. SAHI tiled inference: single-tile parity, tile_batch invariance, IoS/IoU merging, per-class filter.
        let single_tile_cfg = TileConfig {
            tile_width: 64,
            tile_height: 64,
            overlap_x: 0.2,
            overlap_y: 0.2,
            full_image: false,
            max_det: 50,
            ..TileConfig::default()
        };
        let p_one_tile = pred.predict_tiled(&img_small, &single_tile_cfg).unwrap();
        assert_eq!(p_one_tile.detections, p_single.detections, "{label}: SAHI single-tile parity");

        let tiled_b1 = TileConfig {
            tile_width: 64,
            tile_height: 64,
            overlap_x: 0.25,
            overlap_y: 0.25,
            tile_batch: 1,
            full_image: true,
            merge_metric: TileMergeMetric::Ios,
            max_det: 50,
            ..TileConfig::default()
        };
        let tiled_b4 = TileConfig { tile_batch: 4, ..tiled_b1.clone() };
        let p_t1 = pred.predict_tiled(&img_large, &tiled_b1).unwrap();
        let p_t4 = pred.predict_tiled(&img_large, &tiled_b4).unwrap();
        assert_eq!(p_t1.detections, p_t4.detections, "{label}: SAHI tile_batch invariance");
        check_thresholds(&p_t1.detections, "SAHI");
        let p_opt_tile = pred.predict_with_options(&img_large, &PredictOptions::with_tile(tiled_b4)).unwrap();
        assert_eq!(p_opt_tile.detections, p_t1.detections, "{label}: PredictOptions::with_tile");

        // 3. TTA inference: single-view parity, scale-order invariance, hflip + batch parity, per-class filter.
        let tta_single = TtaConfig { scales: vec![1.0], hflip: false, ..TtaConfig::default() };
        let p_tta1 = pred.predict_tta(&img_small, &tta_single).unwrap();
        assert_eq!(p_tta1.detections, p_single.detections, "{label}: TTA single-view parity");

        let tta_ab = TtaConfig { scales: vec![0.5, 1.0], hflip: true, ..TtaConfig::default() };
        let tta_ba = TtaConfig { scales: vec![1.0, 0.5], hflip: true, ..TtaConfig::default() };
        let p_ab = pred.predict_tta(&img_small, &tta_ab).unwrap();
        let p_ba = pred.predict_tta(&img_small, &tta_ba).unwrap();
        assert_eq!(p_ab.detections, p_ba.detections, "{label}: TTA scale-order invariance");
        check_thresholds(&p_ab.detections, "TTA");
        let p_tta_batch = pred.predict_batch_tta(&[img_small.clone(), img_small.clone()], &tta_ab).unwrap();
        assert_eq!(p_tta_batch[0].detections, p_ab.detections, "{label}: TTA batch parity");
        let p_opt_tta = pred.predict_with_options(&img_small, &PredictOptions::with_tta(tta_ab)).unwrap();
        assert_eq!(p_opt_tta.detections, p_ab.detections, "{label}: PredictOptions::with_tta");

        // 4. ByteTrack `predict_sequence`: executes cleanly on all folo2 variants and lowers high per-class
        //    thresholds internally down to `sequence_det_conf` so stage-2 recovery is never starved!
        let trk_cfg =
            TrackConfig { high_thresh: 0.45, low_thresh: 0.15, new_track_thresh: 0.45, ..Default::default() };
        let seq_with_per_class = pred.predict_sequence(&seq_frames, &trk_cfg).unwrap();
        assert_eq!(seq_with_per_class.len(), seq_frames.len(), "{label}: sequence length");
        for (idx, frame) in seq_with_per_class.iter().enumerate() {
            assert_eq!(frame.frame_index, idx);
            assert_eq!((frame.width, frame.height), (64, 64));
            assert!(frame.timing.total_ms >= 0.0 && frame.timing.total_ms.is_finite());
        }

        // Compare against a predictor with very high per-class thresholds (0.99 for all classes): because
        // `predict_sequence` clamps per-class thresholds down to `sequence_det_conf` (< low_thresh = 0.15),
        // tracking output must be identical to the same predictor without per-class overrides!
        let mut high_map = BTreeMap::new();
        high_map.insert(0, 0.99);
        high_map.insert(1, 0.99);
        high_map.insert(2, 0.99);
        let pred_high = Predictor::new_v2(model.fused_eval_clone().unwrap(), base_cfg.clone())
            .unwrap()
            .with_per_class_conf(high_map)
            .unwrap();
        let pred_plain = Predictor::new_v2(model, base_cfg).unwrap();
        let seq_high = pred_high.predict_sequence(&seq_frames, &trk_cfg).unwrap();
        let seq_plain = pred_plain.predict_sequence(&seq_frames, &trk_cfg).unwrap();
        for (a, b) in seq_high.iter().zip(&seq_plain) {
            assert_eq!(
                a.tracks, b.tracks,
                "{label}: predict_sequence must clamp high per-class thresholds to ByteTrack low_thresh floor"
            );
        }
    }
}

#[test]
fn calibrated_handoff_preserves_inclusive_boundaries_and_final_score_filtering() {
    // 1. Inclusive boundary handoff at 0.0, interior grid point (0.25), and 1.0:
    // Class 0 has a TP at exactly 0.0 -> best_conf = 0.0, best_f1 = 1.0.
    // Class 1 has a TP at exactly 0.25 and FP at 0.15 -> best_conf = 0.25, best_f1 = 1.0.
    // Class 2 has a TP at exactly 1.0 and FP at 0.80 -> best_conf = 1.0, best_f1 = 1.0.
    let truths = vec![
        truth(10.0, 10.0, 30.0, 30.0, 0),
        truth(50.0, 10.0, 70.0, 30.0, 1),
        truth(90.0, 10.0, 110.0, 30.0, 2),
    ];
    let preds = vec![
        det(10.0, 10.0, 30.0, 30.0, 0.0, 0),
        det(50.0, 10.0, 70.0, 30.0, 0.25, 1),
        det(150.0, 10.0, 170.0, 30.0, 0.15, 1),
        det(90.0, 10.0, 110.0, 30.0, 1.0, 2),
        det(190.0, 10.0, 210.0, 30.0, 0.80, 2),
    ];
    let mut ev = Evaluator::new(3);
    ev.add_image(&preds, &truths);
    let report = ev.compute();
    let cal = report.calibrated_thresholds();
    assert_eq!(cal.per_class[0].best_conf, 0.0);
    assert!((cal.per_class[1].best_conf - 0.25).abs() < 1e-6);
    assert_eq!(cal.per_class[2].best_conf, 1.0);

    // Apply calibration to a Predictor -> sets conf_inclusive = true and preserves all 3 boundary TPs!
    let model = build_v2_model(707, 3, HeadMode::O2m, false);
    let pred = Predictor::new_v2(model, PredictConfig { imgsz: 64, ..Default::default() })
        .unwrap()
        .with_calibrated_thresholds(cal)
        .unwrap();
    assert!(pred.conf_inclusive());
    let img = pattern_image(64, 64, 1);
    let batch = pred.prepare(std::slice::from_ref(&img), true).unwrap();
    let boxes: Vec<Xyxy> = preds.iter().map(|d| d.bbox).collect();
    let mut scores = vec![-1.0f32; preds.len() * 3];
    for (i, d) in preds.iter().enumerate() {
        scores[i * 3 + d.class as usize] = d.score;
    }
    let finished = pred.finish(&batch, vec![(boxes.clone(), scores.clone())], 3);
    let kept_pairs: Vec<(u32, f32)> = finished[0].iter().map(|d| (d.class, d.score)).collect();
    assert_eq!(
        kept_pairs,
        vec![(2, 1.0), (1, 0.25), (0, 0.0)],
        "inclusive calibrated handoff must keep exact boundary detections at 0.0, 0.25, and 1.0 while dropping FPs"
    );

    // Same boundary preservation on Soft-NMS, Matrix-NMS, and O2o top-k with conf_inclusive = true.
    let inc_cfg = NmsConfig {
        conf: cal.global.best_conf,
        per_class_conf: Some(cal.per_class_map()),
        conf_inclusive: true,
        ..NmsConfig::default()
    };
    assert_eq!(topk_detections(&boxes, &scores, 3, &inc_cfg).len(), 3);
    for kind in [
        NmsKind::SoftLinear,
        NmsKind::SoftGaussian { sigma: 0.5 },
        NmsKind::MatrixLinear,
        NmsKind::MatrixGaussian { sigma: 0.5 },
    ] {
        let scfg = NmsConfig { nms_kind: kind, ..inc_cfg.clone() };
        let out = nms_detections(&preds, &scfg);
        let pairs: Vec<(u32, f32)> = out.iter().map(|d| (d.class, d.score)).collect();
        assert_eq!(pairs, vec![(2, 1.0), (1, 0.25), (0, 0.0)], "{kind:?} inclusive boundaries");
    }

    // 2. Final post-rescoring score filter under ScoreRule::Elc (<= 0.10) and ScoreRule::Consensus when
    //    conf_inclusive == true (evaluator-derived calibration): no detection below applied threshold leaks!
    let mut elc_map = BTreeMap::new();
    elc_map.insert(0, 0.05);
    elc_map.insert(1, 0.50);
    let elc_inc = NmsConfig {
        conf: 0.25,
        score: ScoreRule::Elc,
        per_class_conf: Some(elc_map),
        conf_inclusive: true,
        ..NmsConfig::default()
    };
    let elc_dets = vec![
        det(0.0, 0.0, 20.0, 20.0, 0.06, 0), // isolated raw 0.06 -> ELC rescored < 0.05 -> must be DROPPED!
        det(40.0, 0.0, 60.0, 20.0, 0.40, 0), // raw 0.40 -> ELC rescored >= 0.05 -> kept
        det(80.0, 0.0, 100.0, 20.0, 0.40, 1), // raw 0.40 -> ELC rescored < 0.50 -> dropped
        det(120.0, 0.0, 140.0, 20.0, 0.90, 1), // raw 0.90 -> ELC rescored >= 0.50 -> kept
    ];
    let elc_out = nms_detections(&elc_dets, &elc_inc);
    assert_eq!(
        elc_out.len(),
        2,
        "ELC inclusive must filter final rescored detections <= 0.10 too: {elc_out:?}"
    );
    assert!(elc_out.iter().all(|d| d.score >= elc_inc.conf_for_class(d.class)));

    let cons_inc =
        NmsConfig { conf: 0.30, score: ScoreRule::Consensus, conf_inclusive: true, ..NmsConfig::default() };
    // Isolated raw 0.80 under Consensus rescales to 0.80 / sqrt(10) ≈ 0.253 < 0.30 -> must be DROPPED!
    // Cluster of 4 identical boxes at 0.80 rescales to 0.80 * sqrt(4/10) ≈ 0.506 >= 0.30 -> KEPT!
    let cons_dets = vec![
        det(0.0, 0.0, 20.0, 20.0, 0.80, 0),
        det(60.0, 0.0, 80.0, 20.0, 0.80, 0),
        det(60.0, 0.0, 80.0, 20.0, 0.80, 0),
        det(60.0, 0.0, 80.0, 20.0, 0.80, 0),
        det(60.0, 0.0, 80.0, 20.0, 0.80, 0),
    ];
    let cons_out = nms_detections(&cons_dets, &cons_inc);
    assert_eq!(cons_out.len(), 1, "Consensus inclusive must drop isolated box whose final score < 0.30");
    assert!(cons_out[0].score >= 0.30 && cons_out[0].bbox.x1 == 60.0);

    // 3. Symmetric NaN folding in NmsConfig::min_conf / candidate_conf.
    let mut nan_map = BTreeMap::new();
    nan_map.insert(1, 0.20);
    let nan_cfg = NmsConfig { conf: f32::NAN, per_class_conf: Some(nan_map), ..NmsConfig::default() };
    assert_eq!(nan_cfg.min_conf(), 0.20);
    assert_eq!(nan_cfg.candidate_conf(), 0.20);
}

#[test]
fn o2o_topk_ranks_raw_anchors_before_per_class_filtering_and_is_monotone() {
    // MODEL.md §6.6 regression:
    // max_det = 1, per_class_conf = {0: 0.98, 1: 0.20}:
    // Anchor 0 has [0.95, 0.05] (max 0.95, fails class-0 cutoff 0.98).
    // Anchor 1 has [0.05, 0.85] (max 0.85, passes class-1 cutoff 0.20).
    // Top-1 raw anchor is Anchor 0 -> filtered out by class 0's cutoff -> result MUST be empty!
    let mut map = BTreeMap::new();
    map.insert(0, 0.98);
    map.insert(1, 0.20);
    let boxes = vec![Xyxy::new(0.0, 0.0, 20.0, 20.0), Xyxy::new(30.0, 0.0, 50.0, 20.0)];
    let scores = vec![0.95, 0.05, 0.05, 0.85];
    let cfg = NmsConfig { conf: 0.50, max_det: 1, per_class_conf: Some(map.clone()), ..NmsConfig::default() };
    assert!(topk_detections(&boxes, &scores, 2, &cfg).is_empty());
    let cands = vec![
        Candidate { anchor: 0, class: 0, score: 0.95, bbox: boxes[0] },
        Candidate { anchor: 1, class: 1, score: 0.85, bbox: boxes[1] },
    ];
    assert!(topk_candidates(&cands, &cfg).is_empty());

    // Multi-label monotonicity regression:
    // Anchor 0 has [0.85, 0.25] vs [0.85, 0.15]; Anchor 1 has [0.10, 0.70]; per_class_conf = {0: 0.90, 1: 0.20}.
    // At max_det = 1: Anchor 0 wins raw top-1 (0.85) AND raw top-1 pair (cls 0, 0.85) in BOTH cases ->
    // filtered out by 0.90 -> empty in BOTH cases (never causes Anchor 1's 0.70 to pop in/out!).
    // At max_det = 2: raw top-2 pairs are (a0, cls 0, 0.85) and (a1, cls 1, 0.70) -> filtering drops cls 0
    // and keeps (a1, cls 1, 0.70) identically in BOTH cases!
    let mut ml_map = BTreeMap::new();
    ml_map.insert(0, 0.90);
    ml_map.insert(1, 0.20);
    let cfg_ml1 = NmsConfig {
        conf: 0.50,
        max_det: 1,
        multi_label: true,
        per_class_conf: Some(ml_map),
        ..NmsConfig::default()
    };
    let cfg_ml2 = NmsConfig { max_det: 2, ..cfg_ml1.clone() };
    let s_hi = vec![0.85, 0.25, 0.10, 0.70];
    let s_lo = vec![0.85, 0.15, 0.10, 0.70];
    assert!(topk_detections(&boxes, &s_hi, 2, &cfg_ml1).is_empty());
    assert!(topk_detections(&boxes, &s_lo, 2, &cfg_ml1).is_empty());
    let out_hi = topk_detections(&boxes, &s_hi, 2, &cfg_ml2);
    let out_lo = topk_detections(&boxes, &s_lo, 2, &cfg_ml2);
    assert_eq!(out_hi.len(), 1);
    assert_eq!((out_hi[0].class, out_hi[0].score), (1, 0.70));
    assert_eq!(out_hi, out_lo);

    // Dense vs CPU candidate-first parity under tight max_det = 1 on O2o (FoloModel & FoloV2Model)!
    let v2 = build_v2_model(808, 3, HeadMode::O2o, false);
    // Level 0 (P3) gets highest class 0 score (~0.73), blocked by threshold 0.95; Level 1 (P4) gets class 1 (~0.50).
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(809));
    {
        let _g = no_grad();
        let o2o = v2.head.o2o_cls.as_ref().unwrap();
        o2o[0]
            .cls_branch
            .out
            .bias
            .as_ref()
            .unwrap()
            .copy_from(&Tensor::from_vec(vec![1.0, -2.0, -2.0], [3]).unwrap())
            .unwrap();
        o2o[1]
            .cls_branch
            .out
            .bias
            .as_ref()
            .unwrap()
            .copy_from(&Tensor::from_vec(vec![-2.0, 0.0, -2.0], [3]).unwrap())
            .unwrap();
        o2o[2]
            .cls_branch
            .out
            .bias
            .as_ref()
            .unwrap()
            .copy_from(&Tensor::from_vec(vec![-2.0, -2.0, -2.0], [3]).unwrap())
            .unwrap();
    }
    let mut tight_map = BTreeMap::new();
    tight_map.insert(0, 0.95);
    tight_map.insert(1, 0.20);
    let tight_nms =
        NmsConfig { conf: 0.50, max_det: 1, per_class_conf: Some(tight_map), ..NmsConfig::default() };
    let _g = no_grad();
    let dense = v2.forward(&x).unwrap().detections(&tight_nms, DEFAULT_IOU_FUSION_ALPHA).unwrap();
    let lazy_cands = v2.predict_candidates(&x, &tight_nms).unwrap();
    let lazy: Vec<Vec<Detection>> = lazy_cands.iter().map(|c| v2.finish_candidates(c, &tight_nms)).collect();
    assert!(dense[0].is_empty(), "P3 class-0 top-1 winner rejected by 0.95 must leave empty output");
    assert_eq!(dense, lazy, "candidate-first O2o must match dense bit for bit under tight max_det");
}

#[test]
fn bytetrack_sequence_preserves_independent_per_class_candidate_floors() {
    use folo_vision::fusion::BoxFusion;

    // Fallback conf = 0.40, class-0 override = 0.05, tracker low_thresh = 0.15, BoxFusion::Vote.
    // Adding class-0 override = 0.05 must NOT lower class 1's pre-NMS floor below sequence_det_conf(0.40, 0.15)
    // (so class-1 tracks and boxes under Vote fusion stay bit-identical to having no class-0 override).
    let model = build_v2_model(909, 3, HeadMode::O2m, false);
    let frames = vec![pattern_image(64, 64, 1), pattern_image(64, 64, 2)];
    let cfg = PredictConfig {
        imgsz: 64,
        conf: 0.40,
        iou: 0.65,
        max_det: 50,
        fusion: BoxFusion::Vote,
        ..PredictConfig::default()
    };
    let trk_cfg =
        TrackConfig { high_thresh: 0.45, low_thresh: 0.15, new_track_thresh: 0.45, ..Default::default() };

    let pred_base = Predictor::new_v2(model.fused_eval_clone().unwrap(), cfg.clone()).unwrap();
    let mut map = BTreeMap::new();
    map.insert(0, 0.05);
    let pred_c0_low = Predictor::new_v2(model, cfg).unwrap().with_per_class_conf(map).unwrap();

    let seq_base = pred_base.predict_sequence(&frames, &trk_cfg).unwrap();
    let seq_c0_low = pred_c0_low.predict_sequence(&frames, &trk_cfg).unwrap();
    for (a, b) in seq_base.iter().zip(&seq_c0_low) {
        let c1_a: Vec<_> = a.tracks.iter().filter(|t| t.class_id != 0).collect();
        let c1_b: Vec<_> = b.tracks.iter().filter(|t| t.class_id != 0).collect();
        assert_eq!(
            c1_a, c1_b,
            "lowering class 0 override to 0.05 must never alter other classes' tracks/boxes"
        );
    }
}
