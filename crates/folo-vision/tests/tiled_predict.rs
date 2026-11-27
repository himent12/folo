//! End-to-end integration tests for SAHI tiled inference ([`folo_vision::tile`] wired into [`Predictor`]):
//! single-tile fast-path parity, tile-batch invariance across `f32`/`int8`/`o2o`, cross-tile merging modes,
//! `predict_batch_tiled` & `predict_file_tiled`, and error handling.

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_data::Image;
use folo_test_support::TestDir;
use folo_vision::fusion::BoxFusion;
use folo_vision::model::quant::{self, Method};
use folo_vision::model::{FoloModel, HeadMode, ModelConfig};
use folo_vision::{PredictConfig, Predictor, TileConfig, TileMergeMetric};

fn fused_model(nc: usize, head: HeadMode, seed: u64) -> FoloModel {
    let mut cfg = ModelConfig::new("n", nc).unwrap();
    cfg.head = head.as_str().to_string();
    cfg.bn_momentum = 0.999;
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
    let mut img = Image::new(w, h, [24, 32, 48]);
    // Place crisp contrasting squares at several scales/locations across the image.
    img.fill_rect(12, 14, 38, 40, [240, 40, 40]);
    img.fill_rect(w as i64 - 44, h as i64 - 42, w as i64 - 16, h as i64 - 14, [40, 230, 80]);
    img.fill_rect(w as i64 / 2 - 18, h as i64 / 2 - 18, w as i64 / 2 + 18, h as i64 / 2 + 18, [235, 220, 40]);
    img
}

/// When an image fits inside one tile (`width <= tile_width && height <= tile_height`), `predict_tiled`
/// degrades to `predict` with exact bit-for-bit parity (and enforces `tile_cfg.max_det`).
#[test]
fn small_image_degrades_to_exact_single_pass() {
    let m = fused_model(2, HeadMode::O2m, 5);
    let pred =
        Predictor::new(m, PredictConfig { imgsz: 64, conf: 0.0, max_det: 50, ..Default::default() }).unwrap();
    let img = patterned_image(64, 48);

    let direct = pred.predict(&img).unwrap();
    assert!(!direct.detections.is_empty());

    let tiled = pred.predict_tiled(&img, &TileConfig { max_det: 50, ..TileConfig::new(96) }).unwrap();
    assert_eq!(tiled.width, direct.width);
    assert_eq!(tiled.height, direct.height);
    assert_eq!(tiled.detections, direct.detections);

    let capped = pred.predict_tiled(&img, &TileConfig { max_det: 3, ..TileConfig::new(96) }).unwrap();
    assert_eq!(capped.detections.len(), direct.detections.len().min(3));
    assert_eq!(capped.detections, direct.detections[..capped.detections.len()]);
}

/// Tiled inference produces finite, bounded detections, is bit-identical regardless of `tile_batch` chunking,
/// and works across `f32`, `int8` (`u8_stem`), and `o2o` models as well as all merge metrics and fusion modes.
#[test]
fn tiled_prediction_is_batch_invariant_across_f32_int8_o2o_and_merge_modes() {
    let img = patterned_image(192, 128);

    // 1. f32 o2m: tile_batch invariance across 1, 2, 4, 16 and merge mode coverage.
    let m_f32 = fused_model(2, HeadMode::O2m, 19);
    let pred_f32 =
        Predictor::new(m_f32, PredictConfig { imgsz: 64, conf: 0.0, max_det: 30, ..Default::default() })
            .unwrap();

    let base_cfg = TileConfig {
        tile_width: 80,
        tile_height: 80,
        overlap_x: 0.25,
        overlap_y: 0.25,
        full_image: true,
        tile_batch: 1,
        merge_metric: TileMergeMetric::Ios,
        merge_threshold: 0.5,
        merge_fusion: BoxFusion::Vote,
        class_agnostic: false,
        max_det: 40,
    };
    let ref_out = pred_f32.predict_tiled(&img, &base_cfg).unwrap();
    assert_eq!((ref_out.width, ref_out.height), (192, 128));
    assert!(!ref_out.detections.is_empty());
    assert!(ref_out.detections.len() <= 40);
    for d in &ref_out.detections {
        assert!(d.score.is_finite() && (0.0..=1.0).contains(&d.score));
        assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0);
        assert!(d.bbox.x2 <= 192.0 && d.bbox.y2 <= 128.0);
        assert!(d.bbox.x2 > d.bbox.x1 && d.bbox.y2 > d.bbox.y1);
    }

    for tb in [2usize, 4, 16] {
        let out = pred_f32.predict_tiled(&img, &TileConfig { tile_batch: tb, ..base_cfg.clone() }).unwrap();
        assert_eq!(out.detections, ref_out.detections, "tile_batch={tb} diverged from tile_batch=1");
    }

    for (metric, fusion, full_image) in [
        (TileMergeMetric::Iou, BoxFusion::None, false),
        (TileMergeMetric::Ios, BoxFusion::Sebf, true),
        (TileMergeMetric::Iou, BoxFusion::Sebf, false),
    ] {
        let cfg = TileConfig { merge_metric: metric, merge_fusion: fusion, full_image, ..base_cfg.clone() };
        let out = pred_f32.predict_tiled(&img, &cfg).unwrap();
        assert!(!out.detections.is_empty());
        for d in &out.detections {
            assert!(d.score.is_finite());
            assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0 && d.bbox.x2 <= 192.0 && d.bbox.y2 <= 128.0);
        }
    }

    // 2. int8 (direct u8 stem) model.
    let mut m_q8 = fused_model(2, HeadMode::O2m, 19);
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(20));
    let cal = quant::calibrate(&m_q8, || std::iter::once(Ok(x.clone())), Method::MinMax).unwrap();
    assert!(cal.u8_stem);
    m_q8.set_int8(Some(cal)).unwrap();
    let pred_q8 =
        Predictor::new(m_q8, PredictConfig { imgsz: 64, conf: 0.0, max_det: 30, ..Default::default() })
            .unwrap();
    let q8_a = pred_q8.predict_tiled(&img, &TileConfig { tile_batch: 1, ..base_cfg.clone() }).unwrap();
    let q8_b = pred_q8.predict_tiled(&img, &TileConfig { tile_batch: 8, ..base_cfg.clone() }).unwrap();
    assert!(!q8_a.detections.is_empty());
    assert_eq!(q8_a.detections, q8_b.detections, "int8 tiled predictions must be tile_batch-invariant");

    // 3. o2o (NMS-free top-k) head.
    let m_o2o = fused_model(3, HeadMode::O2o, 29);
    let pred_o2o =
        Predictor::new(m_o2o, PredictConfig { imgsz: 64, conf: 0.0, max_det: 20, ..Default::default() })
            .unwrap();
    let o2o_a = pred_o2o.predict_tiled(&img, &TileConfig { tile_batch: 1, ..base_cfg.clone() }).unwrap();
    let t_wall = std::time::Instant::now();
    let o2o_b = pred_o2o.predict_tiled(&img, &TileConfig { tile_batch: 8, ..base_cfg.clone() }).unwrap();
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1e3;
    assert!(!o2o_a.detections.is_empty());
    assert_eq!(o2o_a.detections, o2o_b.detections, "o2o tiled predictions must be tile_batch-invariant");
    assert!(
        o2o_b.timing.total_ms <= wall_ms + 1.0,
        "tiled total_ms ({:.2}) must not double-count parallel CPU wall time ({:.2})",
        o2o_b.timing.total_ms,
        wall_ms
    );
    for d in &o2o_a.detections {
        assert!(d.class < 3);
        assert!(d.bbox.x1 >= 0.0 && d.bbox.y1 >= 0.0 && d.bbox.x2 <= 192.0 && d.bbox.y2 <= 128.0);
    }
}

/// `predict_batch_tiled` and `predict_file_tiled` preserve exact parity with `predict_tiled` and report valid
/// timings and errors.
#[test]
fn predict_batch_tiled_file_tiled_and_error_handling() {
    let dir = TestDir::new("tiled-predict");
    let m = fused_model(2, HeadMode::O2m, 37);
    let pred =
        Predictor::new(m, PredictConfig { imgsz: 64, conf: 0.0, max_det: 25, ..Default::default() }).unwrap();
    let tile_cfg =
        TileConfig { tile_width: 80, tile_height: 80, overlap_x: 0.2, overlap_y: 0.2, ..Default::default() };

    // Empty batch succeeds cleanly.
    assert!(pred.predict_batch_tiled(&[], &tile_cfg).unwrap().is_empty());

    // Mixed-size batch (small single-tile + wide tiled + tall tiled).
    let imgs = vec![patterned_image(64, 64), patterned_image(180, 96), patterned_image(96, 180)];
    let batch = pred.predict_batch_tiled(&imgs, &tile_cfg).unwrap();
    assert_eq!(batch.len(), imgs.len());
    for (got, img) in batch.iter().zip(&imgs) {
        let single = pred.predict_tiled(img, &tile_cfg).unwrap();
        assert_eq!(got.detections, single.detections);
        assert_eq!((got.width, got.height), (img.width, img.height));
        assert!(
            got.timing.preprocess_ms >= 0.0
                && got.timing.inference_ms >= 0.0
                && got.timing.postprocess_ms >= 0.0
        );
        assert!(
            (got.timing.total_ms
                - (got.timing.preprocess_ms + got.timing.inference_ms + got.timing.postprocess_ms))
                .abs()
                < 1e-6
        );
    }

    // `predict_file_tiled` round-trips through a PNG on disk with exact detection parity.
    let img_path = dir.path().join("scene.png");
    folo_data::write_image(&imgs[1], &img_path, folo_data::ImageFormat::Png).unwrap();
    let from_file = pred.predict_file_tiled(&img_path, &tile_cfg).unwrap();
    assert_eq!(from_file.detections, batch[1].detections);
    assert_eq!((from_file.width, from_file.height), (imgs[1].width, imgs[1].height));

    // Error paths: invalid config, empty image, truncated image buffer.
    let bad_cfg = TileConfig { overlap_x: 1.0, ..tile_cfg.clone() };
    assert!(pred.predict_tiled(&imgs[0], &bad_cfg).is_err());
    assert!(pred.predict_batch_tiled(&[], &bad_cfg).is_err());
    assert!(pred.predict_file_tiled(&img_path, &bad_cfg).is_err());
    let empty_img = Image { width: 0, height: 64, data: Vec::new() };
    assert!(pred.predict_tiled(&empty_img, &tile_cfg).is_err());
    let bad_buf = Image { width: 160, height: 160, data: vec![0u8; 10] };
    assert!(pred.predict_tiled(&bad_buf, &tile_cfg).is_err());
    let bad_small_buf = Image { width: 64, height: 64, data: vec![0u8; 10] };
    assert!(pred.predict_tiled(&bad_small_buf, &tile_cfg).is_err());
}
