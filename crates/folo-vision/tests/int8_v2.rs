//! End-to-end tests for FOLO v2 (`folo2-*`) int8 PTQ, selective-`f32` calibration, `.folo` round-trip IO,
//! CPU candidate-first parity, `Predictor`, and `Validator`.

use std::sync::Arc;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{Dataset, Image, Split};
use folo_nn::io::ModelFile;
use folo_test_support::TestDir;
use folo_vision::model::quant::{
    self, Calibration, FakeQuant, Int8Plan, Method, SensitivityPolicy, VERSION_V2,
};
use folo_vision::model::v2::{DEFAULT_IOU_FUSION_ALPHA, FoloV2Config, FoloV2Model};
use folo_vision::model::{Detector, HeadMode};
use folo_vision::nms::NmsConfig;
use folo_vision::predict::{PredictConfig, Predictor};
use folo_vision::val::{ValConfig, validate_detector, validate_v2};

fn tiny_v2(cfg: FoloV2Config, seed: u64, x_cal: &Tensor) -> FoloV2Model {
    let mut cfg = cfg;
    cfg.bn_momentum = 0.999;
    let mut m = FoloV2Model::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    {
        let _g = no_grad();
        m.forward(x_cal).unwrap();
    }
    m.fuse().unwrap();
    m
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let av = a.to_vec::<f32>().unwrap();
    let bv = b.to_vec::<f32>().unwrap();
    assert_eq!(av.len(), bv.len());
    av.iter().zip(&bv).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn synth_image(w: u32, h: u32, seed: u64) -> Image {
    let mut rng = Rng::seed(seed);
    let data = (0..w * h * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
    Image::from_raw(w, h, data).unwrap()
}

/// Exact executor vs [`FakeQuant::features_v2`] parity across `spd_stem: true`, `spd_stem: false`,
/// `plus: true` (`m >= 3` `mid_tap`), and `fullpad: false`.
#[test]
fn folo2_int8_features_match_fake_quant_across_variants() {
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(11));
    let x_test = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(12));

    let variants = [
        ("v2-iso-spd", FoloV2Config::new("n", 4).unwrap()),
        ("v2-conv-stem", FoloV2Config::new("n", 4).unwrap().with_spd_stem(false)),
        ("v2-plus-midtap", FoloV2Config::new("n", 4).unwrap().with_plus(true)),
        ("v2-no-fullpad", FoloV2Config::new("n", 4).unwrap().with_fullpad(false)),
    ];

    for (label, cfg) in variants {
        let m = tiny_v2(cfg, 101, &x_cal);
        let cal = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
        assert_eq!(cal.version, VERSION_V2, "{label}: expected v2 calibration");
        assert!(cal.u8_stem, "{label}: expected u8_stem enabled");

        let plan = Int8Plan::new(cal.clone());
        let mut fq = FakeQuant::new(&cal);
        let _g = no_grad();
        let q8_feats = quant::exec::features_v2(&m, &plan, &x_test).unwrap();
        let fq_feats = fq.features_v2(&m, &x_test).unwrap();

        for l in 0..3 {
            let diff = max_abs_diff(&q8_feats[l], &fq_feats[l]);
            assert!(
                diff < 1e-2,
                "{label} level {l}: int8 executor diverged from FakeQuant (max abs diff {diff})"
            );
        }
    }
}

/// Direct `rgb0_u8` stem packing matches planar `f32` tensor input bit-for-bit for both `SPD(2×2)` and
/// `Conv3×3 s=2` stems.
#[test]
fn folo2_int8_rgb0_u8_matches_f32_tensor_bit_for_bit() {
    let img = synth_image(64, 64, 21);
    let mut rgb0 = vec![0u32; 64 * 64 + 64];
    let mut planar = vec![0.0f32; 3 * 64 * 64];
    for p in 0..64 * 64 {
        let r = img.data[3 * p];
        let g = img.data[3 * p + 1];
        let b = img.data[3 * p + 2];
        rgb0[p] = u32::from_le_bytes([r, g, b, 0]);
        planar[p] = f32::from(r) / 255.0;
        planar[64 * 64 + p] = f32::from(g) / 255.0;
        planar[2 * 64 * 64 + p] = f32::from(b) / 255.0;
    }
    let x = Tensor::from_vec(planar, [1, 3, 64, 64]).unwrap();

    for spd_stem in [true, false] {
        let cfg = FoloV2Config::new("n", 4).unwrap().with_spd_stem(spd_stem);
        let mut m = tiny_v2(cfg, 22, &x);
        let cal = quant::calibrate_v2(&m, || std::iter::once(Ok(x.clone())), Method::MinMax).unwrap();
        let plan = Int8Plan::new(cal.clone());
        m.set_int8(Some(cal)).unwrap();

        let _g = no_grad();
        let f_tensor = quant::exec::features_v2(&m, &plan, &x).unwrap();
        let f_rgb0 = quant::exec::features_v2_rgb0_u8(&m, &plan, &rgb0, 1, 64, 64).unwrap();
        for l in 0..3 {
            assert_eq!(
                f_tensor[l].to_vec::<f32>().unwrap(),
                f_rgb0[l].to_vec::<f32>().unwrap(),
                "spd_stem={spd_stem} level {l}: rgb0_u8 must match f32 tensor bit-for-bit"
            );
        }
    }
}

/// Selective-`f32` sensitivity analysis, `with_sensitivity_f32_v2`, `with_p5_tail_f32_v2`, and
/// `calibrate_detector` / `with_sensitivity_f32_detector` preserve `f32_convs` and match
/// [`FakeQuant::features_v2`].
#[test]
fn folo2_selective_f32_calibration_and_sensitivity_end_to_end() {
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(31));
    let x_test = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(32));
    let m = tiny_v2(FoloV2Config::new("n", 4).unwrap(), 33, &x_cal);

    let cal_raw = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
    let report = quant::analyze_sensitivity_v2(&m, &cal_raw).unwrap();
    assert!(!report.is_empty(), "sensitivity report must cover v2 convs");
    assert!(
        report.iter().any(|s| s.name == "backbone.hyper.out_proj"),
        "sensitivity report should include HyperContext out_proj"
    );

    let policy = SensitivityPolicy {
        max_f32_mac_fraction: 0.15,
        sqnr_threshold_db: 40.0,
        include_p5_head_out: true,
        pair_residual_bottlenecks: true,
    };
    let picked = quant::select_f32_convs_v2(&m, &cal_raw, policy).unwrap();
    assert!(!picked.is_empty());

    let cal_sens = cal_raw.clone().with_sensitivity_f32_v2(&m, policy).unwrap();
    assert_eq!(cal_sens.f32_convs, picked);

    let cal_p5 = cal_raw.with_p5_tail_f32_v2(&m);
    assert!(cal_p5.f32_convs.iter().any(|n| n == "neck.out5.cv1"));
    assert!(cal_p5.f32_convs.iter().any(|n| n == "neck.out5.cv2"));

    let _g = no_grad();
    for (label, cal) in [("sensitivity", &cal_sens), ("p5_tail", &cal_p5)] {
        let plan = Int8Plan::new(cal.clone());
        let mut fq = FakeQuant::new(cal);
        let q8_feats = quant::exec::features_v2(&m, &plan, &x_test).unwrap();
        let fq_feats = fq.features_v2(&m, &x_test).unwrap();
        for l in 0..3 {
            let diff = max_abs_diff(&q8_feats[l], &fq_feats[l]);
            assert!(diff < 1e-2, "{label} level {l}: max diff {diff}");
        }
    }

    // Detector-level calibration helper works identically on Detector::V2.
    let det = Detector::V2(Box::new(m.fused_eval_clone().unwrap()));
    let cal_det =
        quant::calibrate_detector(&det, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
    let cal_det = cal_det.with_sensitivity_f32_detector(&det, policy).unwrap();
    assert_eq!(cal_det.f32_convs, picked);
}

/// FP32 vs int8 tolerance, thread-count bit-determinism, and `ScratchPool` reuse on `folo2-n`.
#[test]
fn folo2_int8_fp32_parity_determinism_and_scratch_reuse() {
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(41));
    let x_test = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(42));
    // Default `bn_momentum` keeps well-conditioned running stats on a 64×64 input (P5 is 2×2).
    let mut m_f32 = FoloV2Model::with_rng(FoloV2Config::new("n", 4).unwrap(), &mut Rng::seed(43)).unwrap();
    m_f32.fuse().unwrap();
    let cal = quant::calibrate_v2(&m_f32, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();

    let mut m_q8 = m_f32.fused_eval_clone().unwrap();
    m_q8.set_int8(Some(cal.clone())).unwrap();

    let _g = no_grad();
    let feats_f32 = m_f32.forward_features(&x_test).unwrap();
    let feats_q8 = m_q8.forward_features(&x_test).unwrap();
    for l in 0..3 {
        let fv = feats_f32[l].to_vec::<f32>().unwrap();
        let qv = feats_q8[l].to_vec::<f32>().unwrap();
        let dot: f64 = fv.iter().zip(&qv).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let na: f64 = fv.iter().map(|&a| f64::from(a) * f64::from(a)).sum::<f64>().sqrt();
        let nb: f64 = qv.iter().map(|&b| f64::from(b) * f64::from(b)).sum::<f64>().sqrt();
        let cos = dot / (na * nb).max(1e-12);
        assert!(cos > 0.98, "level {l}: int8 vs fp32 feature cosine similarity too low: {cos}");
    }

    let out_f32 = m_f32.forward(&x_test).unwrap();
    let out_q8 = m_q8.forward(&x_test).unwrap();
    let out_q8_2 = m_q8.forward(&x_test).unwrap();

    assert_eq!(out_q8.cls_raw.to_vec::<f32>().unwrap(), out_q8_2.cls_raw.to_vec::<f32>().unwrap());

    // ScratchPool reuses buffers across layers and repeated calls.
    let plan = Int8Plan::new(cal.clone());
    let _ = quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap();
    let _ = quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap();
    let stats = plan.scratch_stats();
    assert!(stats.reuses > 0, "expected ScratchPool buffer reuse, got {stats:?}");

    // Box/score outputs stay close to FP32 reference.
    let (boxes_f32, scores_f32) = out_f32.decode_images(DEFAULT_IOU_FUSION_ALPHA).unwrap().pop().unwrap();
    let (boxes_q8, scores_q8) = out_q8.decode_images(DEFAULT_IOU_FUSION_ALPHA).unwrap().pop().unwrap();
    assert_eq!(boxes_f32.len(), boxes_q8.len());
    assert!(scores_q8.iter().all(|v| v.is_finite()));
    let max_score_diff = scores_f32.iter().zip(&scores_q8).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_box_diff = boxes_f32
        .iter()
        .zip(&boxes_q8)
        .map(|(a, b)| {
            (a.x1 - b.x1).abs().max((a.y1 - b.y1).abs()).max((a.x2 - b.x2).abs()).max((a.y2 - b.y2).abs())
        })
        .fold(0.0f32, f32::max);
    assert!(max_score_diff < 0.05, "int8 vs fp32 max score diff too large: {max_score_diff}");
    assert!(max_box_diff < 2.0, "int8 vs fp32 max box diff too large: {max_box_diff}");

    // Thread-count bit-determinism: 1 thread vs 4 threads produce identical outputs.
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let p4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let y1: Vec<Vec<f32>> = p1
        .install(|| quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap())
        .into_iter()
        .map(|t| t.to_vec::<f32>().unwrap())
        .collect();
    let y4: Vec<Vec<f32>> = p4
        .install(|| quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap())
        .into_iter()
        .map(|t| t.to_vec::<f32>().unwrap())
        .collect();
    assert_eq!(y1, y4, "folo2 int8 features must be bit-identical across thread counts");
}

/// `.folo` serialization round-trip preserves `folo.quant` on `save` / `export(F32)`, strips it on
/// `export(F16)`, loads cleanly via `open_int8` / `Detector::open_int8` / `Predictor::open_int8`, and
/// rejects tampered weights via fingerprint check.
#[test]
fn folo2_int8_dot_folo_save_load_round_trip() {
    let dir = TestDir::new("folo2-int8-io");
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(51));
    let mut m = tiny_v2(FoloV2Config::new("n", 4).unwrap(), 52, &x_cal);
    let cal = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax)
        .unwrap()
        .with_p5_tail_f32_v2(&m);
    let expected_f32_convs = cal.f32_convs.clone();
    m.set_int8(Some(cal)).unwrap();

    let path_q8 = dir.path().join("folo2-int8.folo");
    m.save(&path_q8).unwrap();

    // Regular F32 load does not silently turn int8 on; explicit `open_int8` restores calibration and f32_convs.
    let f_plain = FoloV2Model::open(&path_q8).unwrap();
    assert!(f_plain.int8().is_none());

    let loaded = FoloV2Model::open_int8(&path_q8).unwrap();
    let loaded_cal = loaded.int8().expect("int8 plan must be attached");
    assert_eq!(loaded_cal.f32_convs, expected_f32_convs);

    let det_loaded = Detector::open_int8(&path_q8).unwrap();
    assert!(det_loaded.is_v2() && det_loaded.int8().is_some());

    let pred_loaded = Predictor::open_int8(&path_q8, PredictConfig::default()).unwrap();
    assert!(pred_loaded.detector().int8().is_some());

    let _g = no_grad();
    let y_orig = m.forward(&x_cal).unwrap().cls_raw.to_vec::<f32>().unwrap();
    let y_loaded = loaded.forward(&x_cal).unwrap().cls_raw.to_vec::<f32>().unwrap();
    assert_eq!(y_orig, y_loaded, "save/open_int8 round-trip must be bit-identical");

    // Exporting as F16 strips `folo.quant`, so `open_int8` reports missing calibration.
    let path_f16 = dir.path().join("folo2-f16.folo");
    m.export(&path_f16, folo_nn::io::FloatDType::F16).unwrap();
    let f16_file = ModelFile::open(&path_f16).unwrap();
    assert!(Calibration::from_metadata(f16_file.metadata()).unwrap().is_none());
    assert!(FoloV2Model::open_int8(&path_f16).is_err());

    // Tampering with weights invalidates the embedded calibration fingerprint.
    let mut tampered = m.fused_eval_clone().unwrap();
    let bad_cal = loaded_cal.clone();
    let shifted = tampered.backbone.stem.conv.conv.weight.add_scalar(0.5).unwrap();
    tampered.backbone.stem.conv.conv.weight.copy_from(&shifted).unwrap();
    let err = tampered.set_int8(Some(bad_cal)).unwrap_err();
    assert!(err.to_string().contains("other weights"), "expected fingerprint mismatch error, got: {err}");
}

/// CPU candidate-first prediction on `FoloV2Model` matches dense `HeadOutputV2::detections` bit-for-bit for
/// both `O2m` and `O2o` (including both sparse and dense candidate levels), while `iou_head: true` stays on
/// dense `forward_rgb0_u8`.
#[test]
fn folo2_candidate_first_matches_dense_bit_for_bit_o2m_and_o2o() {
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(61));
    let x_test = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(62));

    for mode in [HeadMode::O2m, HeadMode::O2o] {
        let cfg = FoloV2Config::new("n", 4).unwrap().with_head(mode);
        let mut m = tiny_v2(cfg, 63, &x_cal);
        let cal = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
        m.set_int8(Some(cal)).unwrap();
        m.set_head_mode(mode).unwrap();

        // Test both sparse (`conf = 0.25`) and dense (`conf = 0.001`) candidate regimes.
        for conf in [0.25f32, 0.001f32] {
            let nms = NmsConfig { conf, ..NmsConfig::default() };
            assert!(m.candidates_first_applies(Device::Cpu, &nms));
            let _g = no_grad();
            let dense = m.forward(&x_test).unwrap().detections(&nms, DEFAULT_IOU_FUSION_ALPHA).unwrap();
            let lazy: Vec<_> = m
                .predict_candidates(&x_test, &nms)
                .unwrap()
                .iter()
                .map(|c| m.finish_candidates(c, &nms))
                .collect();
            assert_eq!(
                dense, lazy,
                "mode={mode:?} conf={conf}: candidate-first must match dense bit-for-bit"
            );
        }
    }

    // `iou_head: true` disables candidate-first cleanly so score fusion stays exact.
    let cfg_iou = FoloV2Config::new("n", 4).unwrap().with_iou_head(true);
    let m_iou = tiny_v2(cfg_iou, 64, &x_cal);
    assert!(!m_iou.candidates_first_applies(Device::Cpu, &NmsConfig::default()));
}

/// End-to-end `Predictor` (single, batch, `rect`, `o2m`, `o2o`, `iou_head`) and `Validator` (`validate_v2`,
/// `validate_detector`) run cleanly with int8 `folo2`.
#[test]
fn folo2_int8_predictor_and_validator_end_to_end() {
    let x_cal = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(71));
    let imgs = [synth_image(96, 64, 72), synth_image(96, 64, 73), synth_image(64, 64, 74)];

    for (label, cfg) in [
        ("o2m", FoloV2Config::new("n", 3).unwrap()),
        ("o2o", FoloV2Config::new("n", 3).unwrap().with_head(HeadMode::O2o)),
        ("iou", FoloV2Config::new("n", 3).unwrap().with_iou_head(true)),
    ] {
        let mut m = tiny_v2(cfg, 75, &x_cal);
        let cal = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
        m.set_int8(Some(cal)).unwrap();

        let p = Predictor::new_v2(
            m.fused_eval_clone().unwrap(),
            PredictConfig { imgsz: 64, conf: 0.25, rect: true, ..PredictConfig::default() },
        )
        .unwrap();
        let single = p.predict(&imgs[0]).unwrap();
        assert_eq!((single.width, single.height), (96, 64), "{label}: single predict size");
        let batch = p.predict_batch(&imgs).unwrap();
        assert_eq!(batch.len(), imgs.len(), "{label}: batch predict count");
    }

    // Validator end-to-end with int8 folo2 via `validate_v2` and `validate_detector`.
    let synth_cfg = SyntheticConfig {
        images: 6,
        classes: vec!["circle".into(), "square".into(), "triangle".into()],
        imgsz: 64,
        seed: 76,
        val_fraction: 0.5,
        ..SyntheticConfig::default()
    };
    let ds: Arc<dyn Dataset> = Arc::new(SyntheticDataset::new(synth_cfg, Split::Val).unwrap());
    let mut m_val = tiny_v2(FoloV2Config::new("n", 3).unwrap(), 77, &x_cal);
    let cal = quant::calibrate_v2(&m_val, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
    m_val.set_int8(Some(cal)).unwrap();

    let mut vcfg = ValConfig::new(64, 2);
    vcfg.nms.conf = 0.25;
    let r1 = validate_v2(&m_val, Arc::clone(&ds), &vcfg).unwrap();
    let det = Detector::V2(Box::new(m_val));
    let r2 = validate_detector(&det, ds, &vcfg).unwrap();

    assert_eq!(r1.report.per_class.len(), 3);
    assert!(r1.report.map50.is_finite());
    assert_eq!(r1.report.map50, r2.report.map50);
}
