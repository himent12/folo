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

    // Thread-count bit-determinism: 1 thread vs 4 threads produce identical outputs (with `no_grad` created
    // inside each worker closure because `no_grad` is thread-local).
    let p1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let p4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let y1: Vec<Vec<f32>> = p1
        .install(|| {
            let _g = no_grad();
            quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap()
        })
        .into_iter()
        .map(|t| t.to_vec::<f32>().unwrap())
        .collect();
    let y4: Vec<Vec<f32>> = p4
        .install(|| {
            let _g = no_grad();
            quant::exec::features_v2(&m_f32, &plan, &x_test).unwrap()
        })
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

    // Exporting as F32 preserves `folo.quant` so `open_int8` round-trips cleanly.
    let path_f32_exp = dir.path().join("folo2-f32-export.folo");
    m.export(&path_f32_exp, folo_nn::io::FloatDType::F32).unwrap();
    let exp_loaded = FoloV2Model::open_int8(&path_f32_exp).unwrap();
    assert_eq!(exp_loaded.int8().unwrap().f32_convs, expected_f32_convs);

    // Exporting as F16 strips `folo.quant`, so `open_int8` reports missing calibration.
    let path_f16 = dir.path().join("folo2-f16.folo");
    m.export(&path_f16, folo_nn::io::FloatDType::F16).unwrap();
    let f16_file = ModelFile::open(&path_f16).unwrap();
    assert!(Calibration::from_metadata(f16_file.metadata()).unwrap().is_none());
    assert!(FoloV2Model::open_int8(&path_f16).is_err());

    // Replacing weights via `load()` invalidates a fingerprinted int8 plan so neither inference nor `save()`
    // reuses the old activation ranges / fingerprint.
    let other_v2 = tiny_v2(FoloV2Config::new("n", 4).unwrap(), 59, &x_cal);
    let path_other = dir.path().join("folo2-other.folo");
    other_v2.save(&path_other).unwrap();
    let other_file = ModelFile::open(&path_other).unwrap();
    let reloaded = FoloV2Model::open_int8(&path_q8).unwrap();
    assert!(reloaded.int8().is_some());
    reloaded.load(&other_file, folo_nn::io::LoadOptions::strict(folo_nn::io::Purpose::Inference)).unwrap();
    assert!(reloaded.int8().is_none(), "load() with different weights must invalidate fingerprinted int8");
    let path_after_load = dir.path().join("folo2-after-load.folo");
    reloaded.save(&path_after_load).unwrap();
    assert!(
        Calibration::from_metadata(ModelFile::open(&path_after_load).unwrap().metadata()).unwrap().is_none(),
        "save() after weight replacement must not persist stale folo.quant"
    );

    // Lossy `to_dtype(F16) -> to_dtype(F32)` invalidates int8, while a rejected `to_dtype(U8)` preserves it.
    let mut cast_v2 = FoloV2Model::open_int8(&path_q8).unwrap();
    assert!(cast_v2.to_dtype(folo_core::DType::U8).is_err());
    assert!(cast_v2.int8().is_some(), "rejected to_dtype(U8) must preserve active int8 plan");
    cast_v2.to_dtype(folo_core::DType::F16).unwrap();
    assert!(cast_v2.int8().is_none());
    cast_v2.to_dtype(folo_core::DType::F32).unwrap();
    assert!(cast_v2.int8().is_none());

    // Even trait-level `Module::to_dtype` F32 -> F16 -> F32 or direct `ModelFile::load(&model, ...)` detects
    // weight drift via tensor storage stamps and invalidates a fingerprinted plan automatically.
    {
        use folo_nn::Module;
        let mut trait_v2 = FoloV2Model::open_int8(&path_q8).unwrap();
        Module::to_dtype(&mut trait_v2, folo_core::DType::F16).unwrap();
        assert!(trait_v2.int8().is_none());
        Module::to_dtype(&mut trait_v2, folo_core::DType::F32).unwrap();
        assert!(
            trait_v2.int8().is_none(),
            "trait Module::to_dtype F32->F16->F32 must invalidate fingerprinted int8"
        );

        let direct_v2 = FoloV2Model::open_int8(&path_q8).unwrap();
        other_file
            .load(&direct_v2, &folo_nn::io::LoadOptions::strict(folo_nn::io::Purpose::Inference))
            .unwrap();
        assert!(
            direct_v2.int8().is_none(),
            "direct ModelFile::load must invalidate fingerprinted int8 via weight stamp"
        );
    }

    let mut v1 = folo_vision::model::FoloModel::with_rng(
        folo_vision::model::ModelConfig::new("n", 4).unwrap(),
        &mut Rng::seed(53),
    )
    .unwrap();
    v1.fuse().unwrap();
    let v1_cal = quant::calibrate(&v1, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
    v1.set_int8(Some(v1_cal)).unwrap();
    assert!(v1.to_dtype(folo_core::DType::U8).is_err());
    assert!(v1.int8().is_some(), "v1 rejected to_dtype(U8) must preserve active int8 plan");
    // Detached `fused_eval_clone` must keep its own independent `Int8Plan` when `v1.load()` invalidates `v1`.
    let v1_clone = v1.fused_eval_clone().unwrap();
    let mut v1_other = folo_vision::model::FoloModel::with_rng(
        folo_vision::model::ModelConfig::new("n", 4).unwrap(),
        &mut Rng::seed(54),
    )
    .unwrap();
    v1_other.fuse().unwrap();
    let path_v1_other = dir.path().join("folo-v1-other.folo");
    v1_other.save(&path_v1_other).unwrap();
    v1.load(
        &ModelFile::open(&path_v1_other).unwrap(),
        folo_nn::io::LoadOptions::strict(folo_nn::io::Purpose::Inference),
    )
    .unwrap();
    assert!(v1.int8().is_none(), "v1 load() with different weights must invalidate fingerprinted int8");
    assert!(
        v1_clone.int8().is_some(),
        "invalidating v1 must not disable int8 on its detached fused_eval_clone"
    );
    let path_v1_after = dir.path().join("folo-v1-after.folo");
    v1.save(&path_v1_after).unwrap();
    assert!(
        Calibration::from_metadata(ModelFile::open(&path_v1_after).unwrap().metadata()).unwrap().is_none()
    );

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

/// Regression: isolated interior candidate on a `folo2-m` P5 level (`20×20`, `64→64` `conv1` eligible for
/// Winograd F(2×2)) with poisoned uninitialized CPU buffers must match dense box logits bit-for-bit.
#[test]
fn folo2_candidate_first_f2x2_tile_patch_matches_dense_with_poison() {
    folo_core::cpu::poison_uninit(true);
    let mut m = FoloV2Model::with_rng(FoloV2Config::new("m", 2).unwrap(), &mut Rng::seed(81)).unwrap();
    m.fuse().unwrap();

    // Silence P3/P4 completely and make P5 (`20×20`, `< 24` -> Winograd F(2×2)) fire only on its top peak
    // site(s), staying strictly sparse (well below the 2/3 dense fallback threshold).
    {
        let _g = no_grad();
        for lv in &m.head.levels {
            let zeros_w = Tensor::zeros(lv.cls_branch.out.weight.dims().to_vec());
            lv.cls_branch.out.weight.copy_from(&zeros_w).unwrap();
            if let Some(b) = &lv.cls_branch.out.bias {
                b.copy_from(&Tensor::full(b.dims().to_vec(), -20.0)).unwrap();
            }
        }
    }
    let p5 = &m.head.levels[2];
    {
        let _g = no_grad();
        let mut w_cls = vec![0.0f32; p5.cls_branch.out.weight.numel()];
        w_cls[0] = 10.0;
        p5.cls_branch
            .out
            .weight
            .copy_from(&Tensor::from_vec(w_cls, p5.cls_branch.out.weight.dims().to_vec()).unwrap())
            .unwrap();
        if let Some(b) = &p5.cls_branch.out.bias {
            b.copy_from(&Tensor::from_vec(vec![0.0, -20.0], [2]).unwrap()).unwrap();
        }
    }

    // Place a localized bright interior patch around `(320, 320)` so P5 (`20×20`) has a clear unique peak.
    let mut img_buf = vec![0.0f32; 3 * 640 * 640];
    let mut rng = Rng::seed(82);
    for c in 0..3 {
        for y in 272..368 {
            for x0 in 272..368 {
                img_buf[c * 640 * 640 + y * 640 + x0] = rng.uniform(0.5, 2.0);
            }
        }
    }
    let x = Tensor::from_vec(img_buf, [1, 3, 640, 640]).unwrap();
    let _g = no_grad();
    // Pick a threshold midway between the 1st and 2nd highest P5 class-0 scores so only 1 isolated P5 site fires.
    let dense_out = m.forward(&x).unwrap();
    let (_, scores) = dense_out.decode_images(DEFAULT_IOU_FUSION_ALPHA).unwrap().pop().unwrap();
    let p5_offset = 80 * 80 + 40 * 40;
    let mut p5_cls0: Vec<f32> = scores[p5_offset * 2..].as_chunks::<2>().0.iter().map(|c| c[0]).collect();
    p5_cls0.sort_by(|a, b| b.total_cmp(a));
    assert!(p5_cls0[0] > p5_cls0[1], "expected unique P5 peak, got top-2 {:?}", &p5_cls0[..2]);
    let conf = 0.5 * (p5_cls0[0] + p5_cls0[1]);
    let nms = NmsConfig { conf, ..NmsConfig::default() };
    let dense = dense_out.detections(&nms, DEFAULT_IOU_FUSION_ALPHA).unwrap();
    let cands = m.predict_candidates(&x, &nms).unwrap();
    assert_eq!(
        cands[0].cands.len(),
        1,
        "expected exactly 1 isolated P5 candidate (strictly sparse F(2×2)), got {}",
        cands[0].cands.len()
    );
    assert!(
        cands[0].cands.iter().all(|c| (c.anchor as usize) >= p5_offset),
        "expected all surviving candidates on P5"
    );
    let lazy: Vec<_> = cands.iter().map(|c| m.finish_candidates(c, &nms)).collect();
    assert_eq!(
        dense, lazy,
        "F(2×2) isolated P5 candidate must match dense bit-for-bit even with poisoned buffers"
    );
}

/// Regression: fused `HyperContext::attend_and_scene_cpu_fast` must add `(ffn2_gemm + bias)` before adding
/// `e_t`, matching the reference `e_t + (ffn2_gemm + bias)` bit-for-bit on cancelling large values.
#[test]
fn hypercontext_ffn_residual_bias_cancellation_matches_reference() {
    use folo_vision::model::blocks::{Fuse, Init};
    use folo_vision::model::v2::HyperContext;

    let mut rng = Rng::seed(91);
    let mut init = Init::new(&mut rng);
    let mut hc = HyperContext::new(4, 2, &mut init);
    hc.fuse_().unwrap();
    let cg = hc.ffn1.weight.dims()[0];
    let m_edges = hc.m_edges;
    let c_av = hc.av_proj.conv.config().out_channels;
    assert_eq!(c_av, m_edges + cg);

    {
        let _g = no_grad();
        // Make `av_proj` output `A = 0` (uniform attention -> `e_t = 1.0` everywhere) and `V = 1.0`.
        hc.av_proj.conv.weight.copy_from(&Tensor::zeros([c_av, 4, 1, 1])).unwrap();
        let mut av_b = vec![0.0f32; c_av];
        for v in &mut av_b[m_edges..] {
            *v = 1.0;
        }
        hc.av_proj.conv.bias.as_ref().unwrap().copy_from(&Tensor::from_vec(av_b, [c_av]).unwrap()).unwrap();

        // Make `ffn1(e_t)` output `1.0`, so `silu(1.0) = s1`, and set `ffn2.bias = -(s1 * w2)` with
        // `s1 * w2 ≈ +2^24` so `(ffn2_gemm + bias) == 0.0` exactly while `(1.0 + ffn2_gemm) + bias == 0.0`.
        hc.ffn1.weight.copy_from(&Tensor::zeros([cg, cg])).unwrap();
        let s1 = folo_core::cpu::silu(1.0f32);
        let w_val = 16_777_216.0f32 / s1;
        let prod = s1 * w_val;
        hc.ffn1.bias.as_ref().unwrap().copy_from(&Tensor::full([cg], 1.0)).unwrap();
        let mut w2 = vec![0.0f32; cg * cg];
        for c in 0..cg {
            w2[c * cg] = w_val;
        }
        hc.ffn2.weight.copy_from(&Tensor::from_vec(w2, [cg, cg]).unwrap()).unwrap();
        hc.ffn2.bias.as_ref().unwrap().copy_from(&Tensor::full([cg], -prod)).unwrap();
    }

    let x = Tensor::zeros([1, 4, 2, 2]);
    // Grad-enabled runs the tensor reference path (`e_t + (ffn2_gemm + bias)`).
    let (_, g_ref) = hc.forward_with_scene(&x).unwrap();
    // `no_grad` runs `attend_and_scene_cpu_fast`.
    let (_, g_fast) = {
        let _g = no_grad();
        hc.forward_with_scene(&x).unwrap()
    };
    let rv = g_ref.to_vec::<f32>().unwrap();
    let fv = g_fast.to_vec::<f32>().unwrap();
    assert_eq!(rv, fv, "HyperContext FFN cancellation diverged between fast and reference paths");
    assert!(
        fv.iter().all(|&v| (v - 1.0).abs() < 1e-3),
        "expected residual e_t = 1.0 preserved after cancelling +2^24 / -2^24, got {fv:?}"
    );
}

/// Fused `FullPadGate` direct CPU path, fused `dequantize_fullpad_gate` (scaling per-channel grids in one
/// dequantise pass), and fused `HyperContext::attend_and_scene_into_c4` match `FakeQuant` and tensor
/// reference with non-zero gate weights on multi-batch rectangular inputs, and are thread-deterministic.
#[test]
fn fullpad_gate_fused_dequant_and_hyper_c4_parity() {
    use folo_vision::model::blocks::Init;
    use folo_vision::model::v2::FullPadGate;

    let x_cal = Tensor::rand_uniform([1, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(93));
    let x_test = Tensor::rand_uniform([2, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(94));

    let mut cfg = FoloV2Config::new("n", 4).unwrap();
    cfg.fullpad = true;
    let m = tiny_v2(cfg, 95, &x_cal);
    // Give both FullPad gates non-zero weights and biases so `1 + tanh(proj(g_scene))` varies per channel.
    {
        let _g = no_grad();
        let mut wrng = Rng::seed(96);
        for gate in [m.neck.pad_p3.as_ref(), m.neck.pad_p4.as_ref()].into_iter().flatten() {
            let w_shape = gate.proj.weight.dims().to_vec();
            gate.proj.weight.copy_from(&Tensor::rand_uniform(w_shape, -0.6, 0.6, &mut wrng)).unwrap();
            if let Some(bias) = &gate.proj.bias {
                let b_shape = bias.dims().to_vec();
                bias.copy_from(&Tensor::rand_uniform(b_shape, -0.4, 0.4, &mut wrng)).unwrap();
            }
        }
    }

    // 1. Direct `FullPadGate::forward` CPU fast path vs grad-enabled reference (including batch=2).
    {
        let gate = m.neck.pad_p3.as_ref().expect("fullpad pad_p3");
        let c_feat = gate.proj.weight.dims()[0];
        let c_scene = gate.proj.weight.dims()[1];
        let feat = Tensor::rand_uniform([2, c_feat, 8, 12], -2.0, 2.0, &mut Rng::seed(97));
        let g_scene = Tensor::rand_uniform([2, c_scene], -1.5, 1.5, &mut Rng::seed(98));
        let ref_out = gate.forward(&feat, &g_scene).unwrap().to_vec::<f32>().unwrap();
        let fast_out = {
            let _g = no_grad();
            gate.forward(&feat, &g_scene).unwrap().to_vec::<f32>().unwrap()
        };
        let worst = ref_out.iter().zip(&fast_out).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "FullPadGate fast path diverged from reference: max abs diff {worst}");

        // Aliased gate weight/bias storage must not deadlock.
        let mut arng = Rng::seed(101);
        let mut aliased = FullPadGate::new(4, 4, &mut Init::new(&mut arng)).unwrap();
        aliased.proj.bias = Some(aliased.proj.weight.reshape([16]).unwrap().narrow(0, 0, 4).unwrap());
        let af = Tensor::rand_uniform([1, 4, 2, 2], -1.0, 1.0, &mut Rng::seed(99));
        let ag = Tensor::rand_uniform([1, 4], -1.0, 1.0, &mut Rng::seed(100));
        let a_ref = aliased.forward(&af, &ag).unwrap().to_vec::<f32>().unwrap();
        let a_fast = {
            let _g = no_grad();
            aliased.forward(&af, &ag).unwrap().to_vec::<f32>().unwrap()
        };
        let a_worst = a_ref.iter().zip(&a_fast).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(a_worst < 1e-5, "aliased FullPadGate diverged: {a_worst}");
    }

    // 2. End-to-end `features_v2` with non-zero FullPad gates (both pure int8 and mixed with `out_proj` in f32).
    let cal_pure = quant::calibrate_v2(&m, || std::iter::once(Ok(x_cal.clone())), Method::MinMax).unwrap();
    let mut cal_mixed = cal_pure.clone();
    cal_mixed.f32_convs.push("backbone.hyper.out_proj".into());

    let _g = no_grad();
    for (label, cal) in [("pure_int8", &cal_pure), ("hyper_out_proj_f32", &cal_mixed)] {
        let plan = Int8Plan::new(cal.clone());
        let mut fq = FakeQuant::new(cal);
        let fq_feats = fq.features_v2(&m, &x_test).unwrap();
        let pool1 = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let pool4 = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let q1 = pool1.install(|| {
            let _g = no_grad();
            quant::exec::features_v2(&m, &plan, &x_test).unwrap()
        });
        let q4 = pool4.install(|| {
            let _g = no_grad();
            quant::exec::features_v2(&m, &plan, &x_test).unwrap()
        });
        for l in 0..3 {
            let v_fq = fq_feats[l].to_vec::<f32>().unwrap();
            let v1 = q1[l].to_vec::<f32>().unwrap();
            let v4 = q4[l].to_vec::<f32>().unwrap();
            assert!(
                v1.iter().zip(&v4).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{label} level {l}: not bit-deterministic across 1 vs 4 threads"
            );
            let max_abs = v_fq.iter().map(|v| v.abs()).fold(1e-3f32, f32::max);
            let max_diff = v_fq.iter().zip(&v4).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(
                max_diff < max_abs * 0.08 + 1e-2,
                "{label} level {l}: max_diff={max_diff:.5} vs max_abs={max_abs:.5}"
            );
        }
    }
}
