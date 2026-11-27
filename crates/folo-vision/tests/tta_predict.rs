//! End-to-end integration tests for Test-Time Augmentation ([`folo_vision::tta`] wired into [`Predictor`]):
//! single-view identity parity, multi-scale + horizontal-flip consistency across `f32`/`int8`/`o2o`,
//! `BoxFusion::{Vote, Sebf, None}`, `predict_batch_tta`, `predict_file_tta`, `PredictOptions` (including `tile`
//! mutual exclusion), determinism, and error handling.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_data::Image;
use folo_test_support::TestDir;
use folo_vision::fusion::BoxFusion;
use folo_vision::model::quant::{self, Method};
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::{PredictConfig, PredictOptions, Predictor, TileConfig, TtaConfig};

fn fused_model(nc: usize, head: HeadMode, seed: u64) -> FoloModel {
    let mut cfg = ModelConfig::new("n", nc).unwrap();
    cfg.head = head.as_str().to_string();
    let mut m = FoloModel::with_rng(cfg, &mut Rng::seed(seed)).unwrap();
    m.set_classes((0..nc).map(|i| format!("cls{i}")).collect()).unwrap();
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(seed + 11));
    {
        let _g = no_grad();
        m.forward(&x).unwrap();
    }
    m.fuse().unwrap();
    m
}

fn patterned_image(w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h, [22, 30, 44]);
    img.fill_rect(10, 12, 34, 38, [235, 45, 45]);
    img.fill_rect(w as i64 - 40, h as i64 - 38, w as i64 - 12, h as i64 - 10, [45, 225, 85]);
    img.fill_rect(w as i64 / 2 - 14, h as i64 / 2 - 14, w as i64 / 2 + 14, h as i64 / 2 + 14, [240, 215, 40]);
    img
}

/// Single unflipped view at scale 1.0 degrades cleanly to `Predictor::predict` with exact bit-for-bit parity
/// (and enforces `TtaConfig::max_det`).
#[test]
fn single_view_tta_matches_direct_predict_bit_for_bit() {
    let m = fused_model(2, HeadMode::O2m, 7);
    let pred =
        Predictor::new(m, PredictConfig { imgsz: 64, conf: 0.0, max_det: 50, ..Default::default() }).unwrap();
    let img = patterned_image(96, 64);

    let direct = pred.predict(&img).unwrap();
    assert!(!direct.detections.is_empty());

    let identity = TtaConfig { hflip: false, scales: vec![1.0], ..Default::default() };
    let tta = pred.predict_tta(&img, &identity).unwrap();
    assert_eq!((tta.width, tta.height), (direct.width, direct.height));
    assert_eq!(tta.detections, direct.detections);

    let via_opts = pred.predict_with_options(&img, &PredictOptions::with_tta(identity.clone())).unwrap();
    assert_eq!(via_opts.detections, direct.detections);

    let capped = pred.predict_tta(&img, &TtaConfig { max_det: Some(4), ..identity }).unwrap();
    assert_eq!(capped.detections.len(), direct.detections.len().min(4));
    assert_eq!(capped.detections, direct.detections[..capped.detections.len()]);
}

/// Multi-scale + horizontal-flip TTA runs deterministically across `f32`, `int8` (`u8_stem`), and `o2o` models,
/// produces finite in-bounds boxes across all fusion modes, and stays independent of scale listing order.
#[test]
fn tta_is_deterministic_order_invariant_and_works_across_f32_int8_and_o2o() {
    let img = patterned_image(128, 96);

    // 1. f32 o2m: scale-order invariance and fusion modes.
    let m_f32 = fused_model(2, HeadMode::O2m, 13);
    let pred_f32 =
        Predictor::new(m_f32, PredictConfig { imgsz: 64, conf: 0.0, max_det: 30, ..Default::default() })
            .unwrap();

    for fusion in [BoxFusion::Vote, BoxFusion::Sebf, BoxFusion::None] {
        let cfg_a = TtaConfig {
            hflip: true,
            scales: vec![0.5, 1.0, 1.5], // resolves to [32, 64, 96]
            merge_iou: Some(0.5),
            merge_fusion: fusion,
            class_agnostic: false,
            max_det: Some(25),
        };
        let cfg_b = TtaConfig { scales: vec![1.5, 0.5, 1.0], ..cfg_a.clone() };
        let out_a = pred_f32.predict_tta(&img, &cfg_a).unwrap();
        let out_b = pred_f32.predict_tta(&img, &cfg_b).unwrap();
        assert_eq!(out_a.detections, out_b.detections, "TTA must not depend on scale order ({fusion:?})");
        assert!(!out_a.detections.is_empty());
        assert!(out_a.detections.len() <= 25);
        for d in &out_a.detections {
            assert!(d.score.is_finite() && (0.0..=1.0).contains(&d.score));
            assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0 && d.bbox.x2 <= 128.0 && d.bbox.y2 <= 96.0);
            assert!(d.bbox.x2 > d.bbox.x1 && d.bbox.y2 > d.bbox.y1);
        }
        // Scores remain sorted descending.
        assert!(out_a.detections.windows(2).all(|w| w[0].score >= w[1].score));
    }

    // Horizontal-flip equivariance: on an image with spatially varied pixels (no flat constant-receptive-field
    // score ties), TTA(img) with flip_only produces the exact mirrored detections of TTA(flip(img)).
    let mut grad_img = Image::new(128, 96, [0, 0, 0]);
    for y in 0..96u32 {
        for x in 0..128u32 {
            let r = ((x * 2 + y) % 256) as u8;
            let g = ((x + y * 3) % 256) as u8;
            let b = (((x * 5) ^ (y * 7)) % 256) as u8;
            grad_img.set_pixel(x, y, [r, g, b]);
        }
    }
    grad_img.fill_rect(12, 16, 46, 54, [245, 30, 30]);
    let flip_cfg = TtaConfig::flip_only();
    let orig_tta = pred_f32.predict_tta(&grad_img, &flip_cfg).unwrap();
    let mir_img = grad_img.oriented(2);
    let mir_tta = pred_f32.predict_tta(&mir_img, &flip_cfg).unwrap();
    assert_eq!(orig_tta.detections.len(), mir_tta.detections.len());
    assert!(!orig_tta.detections.is_empty());
    for (a, b) in orig_tta.detections.iter().zip(&mir_tta.detections) {
        assert_eq!(a.class, b.class);
        assert!((a.score - b.score).abs() < 1e-6, "score mismatch: {} vs {}", a.score, b.score);
        let unflipped = folo_vision::tta::unflip_xyxy_h(b.bbox, grad_img.width);
        assert!((a.bbox.x1 - unflipped.x1).abs() < 1e-3, "{:?} vs {:?}", a.bbox, unflipped);
        assert!((a.bbox.y1 - unflipped.y1).abs() < 1e-3, "{:?} vs {:?}", a.bbox, unflipped);
        assert!((a.bbox.x2 - unflipped.x2).abs() < 1e-3, "{:?} vs {:?}", a.bbox, unflipped);
        assert!((a.bbox.y2 - unflipped.y2).abs() < 1e-3, "{:?} vs {:?}", a.bbox, unflipped);
    }

    // 2. int8 (direct u8 stem) model.
    let mut m_q8 = fused_model(2, HeadMode::O2m, 13);
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(21));
    let cal = quant::calibrate(&m_q8, || std::iter::once(Ok(x.clone())), Method::MinMax).unwrap();
    assert!(cal.u8_stem);
    m_q8.set_int8(Some(cal)).unwrap();
    let pred_q8 =
        Predictor::new(m_q8, PredictConfig { imgsz: 64, conf: 0.0, max_det: 30, ..Default::default() })
            .unwrap();
    let q8_cfg = TtaConfig::multi_scale(vec![1.0, 1.5]);
    let q8_a = pred_q8.predict_tta(&img, &q8_cfg).unwrap();
    let q8_b = pred_q8.predict_tta(&img, &q8_cfg).unwrap();
    assert!(!q8_a.detections.is_empty());
    assert_eq!(q8_a.detections, q8_b.detections);

    // 3. o2o (NMS-free top-k) head + timing sanity.
    let m_o2o = fused_model(3, HeadMode::O2o, 31);
    let pred_o2o =
        Predictor::new(m_o2o, PredictConfig { imgsz: 64, conf: 0.0, max_det: 20, ..Default::default() })
            .unwrap();
    let t_wall = std::time::Instant::now();
    let o2o_out = pred_o2o.predict_tta(&img, &TtaConfig::multi_scale(vec![0.5, 1.0])).unwrap();
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1e3;
    assert!(!o2o_out.detections.is_empty());
    assert!(
        o2o_out.timing.total_ms <= wall_ms + 2.0,
        "TTA total_ms ({:.2}) must not double-count parallel CPU wall time ({:.2})",
        o2o_out.timing.total_ms,
        wall_ms
    );
    for d in &o2o_out.detections {
        assert!(d.class < 3);
        assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0 && d.bbox.x2 <= 128.0 && d.bbox.y2 <= 96.0);
    }
}

/// `predict_batch_tta`, `predict_file_tta`, and `predict_*_with_options` match single-image `predict_tta`
/// on uniform-canvas batches and reject invalid configs, combined `tta + tile`, and empty/corrupt images.
#[test]
fn batch_file_options_and_error_paths() {
    let dir = TestDir::new("tta-predict");
    let m = fused_model(2, HeadMode::O2m, 41);
    let pred =
        Predictor::new(m, PredictConfig { imgsz: 64, conf: 0.0, max_det: 25, ..Default::default() }).unwrap();
    let tta_cfg = TtaConfig::multi_scale(vec![0.5, 1.0]);

    // Empty batch succeeds cleanly.
    assert!(pred.predict_batch_tta(&[], &tta_cfg).unwrap().is_empty());
    assert!(
        pred.predict_batch_with_options(&[], &PredictOptions::with_tta(tta_cfg.clone())).unwrap().is_empty()
    );

    // Uniform-size batch has exact detection parity with single-image predict_tta.
    let mut img_b = patterned_image(96, 64);
    img_b.fill_rect(20, 20, 60, 50, [90, 120, 240]);
    let uniform = vec![patterned_image(96, 64), img_b];
    let batch = pred.predict_batch_tta(&uniform, &tta_cfg).unwrap();
    assert_eq!(batch.len(), 2);
    for (got, img) in batch.iter().zip(&uniform) {
        let single = pred.predict_tta(img, &tta_cfg).unwrap();
        assert_eq!(got.detections, single.detections);
        assert_eq!((got.width, got.height), (img.width, img.height));
        assert!(
            (got.timing.total_ms
                - (got.timing.preprocess_ms + got.timing.inference_ms + got.timing.postprocess_ms))
                .abs()
                < 1e-6
        );
    }

    // Mixed-size batch succeeds and keeps boxes bounded to each image's own dimensions.
    let mixed = vec![patterned_image(96, 64), patterned_image(64, 96)];
    let mixed_out =
        pred.predict_batch_with_options(&mixed, &PredictOptions::with_tta(tta_cfg.clone())).unwrap();
    assert_eq!(mixed_out.len(), 2);
    for (p, img) in mixed_out.iter().zip(&mixed) {
        assert_eq!((p.width, p.height), (img.width, img.height));
        for d in &p.detections {
            assert!(
                d.bbox.x1 >= 0.0
                    && d.bbox.y1 >= 0.0
                    && d.bbox.x2 <= img.width as f32
                    && d.bbox.y2 <= img.height as f32
            );
        }
    }

    // `predict_file_tta` and `predict_file_with_options` round-trip through PNG on disk.
    let img_path = dir.path().join("tta_scene.png");
    folo_data::write_image(&uniform[0], &img_path, folo_data::ImageFormat::Png).unwrap();
    let from_file = pred.predict_file_tta(&img_path, &tta_cfg).unwrap();
    assert_eq!(from_file.detections, batch[0].detections);
    let from_opts =
        pred.predict_file_with_options(&img_path, &PredictOptions::with_tta(tta_cfg.clone())).unwrap();
    assert_eq!(from_opts.detections, batch[0].detections);

    // `PredictOptions::default()` matches plain `predict`.
    let plain = pred.predict(&uniform[0]).unwrap();
    let opts_plain = pred.predict_with_options(&uniform[0], &PredictOptions::default()).unwrap();
    assert_eq!(opts_plain.detections, plain.detections);

    // Error paths: invalid TTA config, combined TTA + tile, empty image, truncated buffer.
    let bad_tta = TtaConfig { scales: vec![-0.5], ..Default::default() };
    assert!(pred.predict_tta(&uniform[0], &bad_tta).is_err());
    assert!(pred.predict_batch_tta(&[], &bad_tta).is_err());
    assert!(pred.predict_file_tta(&img_path, &bad_tta).is_err());

    let conflict = PredictOptions { tta: Some(tta_cfg.clone()), tile: Some(TileConfig::default()) };
    assert!(pred.predict_with_options(&uniform[0], &conflict).is_err());
    assert!(pred.predict_batch_with_options(&[], &conflict).is_err());
    assert!(pred.predict_file_with_options(&img_path, &conflict).is_err());

    let empty = Image { width: 0, height: 64, data: Vec::new() };
    assert!(pred.predict_tta(&empty, &tta_cfg).is_err());
    let truncated = Image { width: 64, height: 64, data: vec![0u8; 10] };
    assert!(pred.predict_tta(&truncated, &tta_cfg).is_err());
}
