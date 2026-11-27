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

    // 3. o2o (NMS-free top-k) head.
    let m_o2o = fused_model(3, HeadMode::O2o, 31);
    let pred_o2o =
        Predictor::new(m_o2o, PredictConfig { imgsz: 64, conf: 0.0, max_det: 20, ..Default::default() })
            .unwrap();
    let o2o_out = pred_o2o.predict_tta(&img, &TtaConfig::multi_scale(vec![0.5, 1.0])).unwrap();
    assert!(!o2o_out.detections.is_empty());
    for d in &o2o_out.detections {
        assert!(d.class < 3);
        assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0 && d.bbox.x2 <= 128.0 && d.bbox.y2 <= 96.0);
    }

    // Wall-time sanity on a >2-view batch (3 images × 2 flip views = 6 views > CPU_BATCH_CHUNK), verifying
    // parallel CPU chunks do not double-count wall time.
    let batch3 = vec![patterned_image(96, 64), patterned_image(96, 64), patterned_image(96, 64)];
    let t_wall = std::time::Instant::now();
    let b3_out = pred_o2o.predict_batch_tta(&batch3, &TtaConfig::flip_only()).unwrap();
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1e3;
    let sum_total_ms: f64 = b3_out.iter().map(|p| p.timing.total_ms).sum();
    assert!(
        sum_total_ms <= wall_ms + 3.0,
        "sum of batch TTA total_ms ({:.2}) must not double-count parallel CPU wall time ({:.2})",
        sum_total_ms,
        wall_ms
    );
}

/// Multi-scale + horizontal-flip `predict_tta` matches manually running separate single-scale `Predictor`s at
/// each resolved stride-32 `imgsz` (proving every scale actually runs at its resolved size, not base `imgsz`),
/// and covers `hflip: false` multi-scale, single non-1.0 scale, and `predict_batch_tta` fast paths.
#[test]
fn multiscale_reference_parity_and_noflip_and_tile_option_paths() {
    let dir = TestDir::new("tta-ref-parity");
    let img = patterned_image(128, 96);
    let m64 = fused_model(2, HeadMode::O2m, 53);
    let m32 = fused_model(2, HeadMode::O2m, 53);
    let m96 = fused_model(2, HeadMode::O2m, 53);
    let pred64 = Predictor::new(
        m64,
        PredictConfig { imgsz: 64, conf: 0.0, iou: 0.6, max_det: 20, ..Default::default() },
    )
    .unwrap();
    let pred32 = Predictor::new(
        m32,
        PredictConfig { imgsz: 32, conf: 0.0, iou: 0.6, max_det: 20, ..Default::default() },
    )
    .unwrap();
    let pred96 = Predictor::new(
        m96,
        PredictConfig { imgsz: 96, conf: 0.0, iou: 0.6, max_det: 20, ..Default::default() },
    )
    .unwrap();

    // 1. Reference multi-scale + flip parity against independent 32px and 96px Predictors.
    let ms_flip = TtaConfig {
        hflip: true,
        scales: vec![0.5, 1.5], // resolves to [32, 96]
        merge_iou: Some(0.55),
        merge_fusion: BoxFusion::Vote,
        class_agnostic: false,
        max_det: Some(15),
    };
    let flipped = img.oriented(2);
    let mut manual_dets = Vec::new();
    for p in [&pred32, &pred96] {
        manual_dets.extend(p.predict(&img).unwrap().detections);
        manual_dets.extend(p.predict(&flipped).unwrap().detections.into_iter().map(|d| {
            folo_vision::Detection { bbox: folo_vision::tta::unflip_xyxy_h(d.bbox, img.width), ..d }
        }));
    }
    let expected =
        folo_vision::tta::merge_tta_detections(manual_dets, img.width, img.height, &ms_flip, 0.6, 20)
            .unwrap();
    let got = pred64.predict_tta(&img, &ms_flip).unwrap();
    assert_eq!(got.detections, expected);

    // 2. Single non-1.0 scale (`hflip: false, scales: [1.5]`) matches a native `imgsz = 96` Predictor bit-for-bit.
    let single_96 = TtaConfig { hflip: false, scales: vec![1.5], max_det: Some(12), ..Default::default() };
    let mut want_96 = pred96.predict(&img).unwrap().detections;
    want_96.truncate(12);
    assert_eq!(pred64.predict_tta(&img, &single_96).unwrap().detections, want_96);

    // 3. Multi-scale with `hflip: false` for both single and uniform batch (exact parity).
    let noflip_ms = TtaConfig { hflip: false, scales: vec![0.5, 1.5], ..Default::default() };
    let mut img2 = patterned_image(128, 96);
    img2.fill_rect(30, 10, 70, 50, [60, 180, 240]);
    let batch2 = vec![img.clone(), img2];
    let b_noflip = pred64.predict_batch_tta(&batch2, &noflip_ms).unwrap();
    for (p, im) in b_noflip.iter().zip(&batch2) {
        assert_eq!(p.detections, pred64.predict_tta(im, &noflip_ms).unwrap().detections);
    }

    // 4. `predict_batch_tta` len==1 delegation and len==2 single-view fast path with `max_det` cap.
    let b1 = pred64.predict_batch_tta(&batch2[..1], &ms_flip).unwrap();
    assert_eq!(b1[0].detections, got.detections);

    let single_view = TtaConfig { hflip: false, scales: vec![1.0], max_det: Some(4), ..Default::default() };
    let b_fast = pred64.predict_batch_tta(&batch2, &single_view).unwrap();
    let direct_batch = pred64.predict_batch(&batch2).unwrap();
    for (fast, direct) in b_fast.iter().zip(&direct_batch) {
        let want_len = direct.detections.len().min(4);
        assert_eq!(fast.detections, direct.detections[..want_len]);
    }

    // 5. `PredictOptions::with_tile` dispatch across single, batch, and file entry points.
    let tile_cfg =
        TileConfig { tile_width: 80, tile_height: 80, overlap_x: 0.2, overlap_y: 0.2, ..Default::default() };
    let tile_opts = PredictOptions::with_tile(tile_cfg.clone());
    assert_eq!(
        pred64.predict_with_options(&img, &tile_opts).unwrap().detections,
        pred64.predict_tiled(&img, &tile_cfg).unwrap().detections
    );
    let b_tile_opts = pred64.predict_batch_with_options(&batch2, &tile_opts).unwrap();
    let b_tile_ref = pred64.predict_batch_tiled(&batch2, &tile_cfg).unwrap();
    for (a, b) in b_tile_opts.iter().zip(&b_tile_ref) {
        assert_eq!(a.detections, b.detections);
    }
    let path = dir.path().join("tile_opt.png");
    folo_data::write_image(&img, &path, folo_data::ImageFormat::Png).unwrap();
    assert_eq!(
        pred64.predict_file_with_options(&path, &tile_opts).unwrap().detections,
        pred64.predict_file_tiled(&path, &tile_cfg).unwrap().detections
    );
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

fn gpu() -> Option<folo_core::Device> {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU TTA test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU TTA test on software adapter {}", ctx.summary().name);
        return None;
    }
    folo_gpu::init().ok()
}

/// GPU multi-scale + horizontal-flip TTA (uploading source images once across scales with deduplicated resize
/// taps) matches independent single-scale GPU `Predictor` runs for both `predict_tta` and `predict_batch_tta`.
#[test]
fn gpu_multiscale_tta_zero_reupload_matches_independent_scales() {
    let Some(dev) = gpu() else { return };
    let img1 = patterned_image(128, 96);
    let mut img2 = patterned_image(128, 96);
    img2.fill_rect(24, 18, 72, 62, [70, 195, 240]);

    let base = fused_model(2, HeadMode::O2m, 71);
    let mut m64 = base.fused_eval_clone().unwrap();
    let mut m32 = base.fused_eval_clone().unwrap();
    let mut m96 = base.fused_eval_clone().unwrap();
    m64.to_device(dev).unwrap();
    m32.to_device(dev).unwrap();
    m96.to_device(dev).unwrap();

    let pred64 = Predictor::new(
        m64,
        PredictConfig { imgsz: 64, conf: 0.0, iou: 0.6, max_det: 15, ..Default::default() },
    )
    .unwrap();
    let pred32 = Predictor::new(
        m32,
        PredictConfig { imgsz: 32, conf: 0.0, iou: 0.6, max_det: 15, ..Default::default() },
    )
    .unwrap();
    let pred96 = Predictor::new(
        m96,
        PredictConfig { imgsz: 96, conf: 0.0, iou: 0.6, max_det: 15, ..Default::default() },
    )
    .unwrap();

    let tta_cfg = TtaConfig {
        hflip: true,
        scales: vec![0.5, 1.5], // resolves to [32, 96] -> triggers multi-scale zero-reupload GPU path
        merge_iou: Some(0.55),
        merge_fusion: BoxFusion::Vote,
        class_agnostic: false,
        max_det: Some(12),
    };

    // Reference: manual per-scale GPU prediction + unflip + merge on `img1`.
    let fl1 = img1.oriented(2);
    let mut manual = Vec::new();
    for p in [&pred32, &pred96] {
        let pair = p.predict_batch(&[img1.clone(), fl1.clone()]).unwrap();
        manual.extend(pair[0].detections.iter().copied());
        manual.extend(pair[1].detections.iter().map(|d| folo_vision::Detection {
            bbox: folo_vision::tta::unflip_xyxy_h(d.bbox, img1.width),
            ..*d
        }));
    }
    let expected =
        folo_vision::tta::merge_tta_detections(manual, img1.width, img1.height, &tta_cfg, 0.6, 15).unwrap();

    let single = pred64.predict_tta(&img1, &tta_cfg).unwrap();
    assert_eq!(
        single.detections, expected,
        "GPU zero-reupload predict_tta must match manual scales bit-for-bit"
    );

    let batch = pred64.predict_batch_tta(&[img1.clone(), img2.clone()], &tta_cfg).unwrap();
    assert_eq!(batch.len(), 2);
    let single2 = pred64.predict_tta(&img2, &tta_cfg).unwrap();
    for (got, want) in [
        (batch[0].detections.as_slice(), single.detections.as_slice()),
        (batch[1].detections.as_slice(), single2.detections.as_slice()),
    ] {
        assert_eq!(got.len(), want.len());
        for (a, b) in got.iter().zip(want) {
            assert_eq!(a.class, b.class);
            assert!(
                (a.score - b.score).abs() < 1e-3,
                "GPU batch TTA score mismatch: {} vs {}",
                a.score,
                b.score
            );
            assert!((a.bbox.x1 - b.bbox.x1).abs() < 0.25 && (a.bbox.y1 - b.bbox.y1).abs() < 0.25);
            assert!((a.bbox.x2 - b.bbox.x2).abs() < 0.25 && (a.bbox.y2 - b.bbox.y2).abs() < 0.25);
        }
    }
}
