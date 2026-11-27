//! Per-class confidence thresholds (`perClassConf`) and inclusive threshold parity (`confInclusive`) tests
//! across `detect`, `detectTiled`, and `detectTta` on CPU and WebGPU for `folo-*` and `folo2-*`.

use std::collections::BTreeMap;

use folo_nn::io::{ModelFile, Purpose};
use folo_vision::{Detector as VisionDetector, PredictConfig, Predictor};
use folo_wasm::detector::rgba_to_image;
use folo_wasm::{BackendChoice, DetectOptions, Precision, TileDetectOptions, TtaDetectOptions};

use super::{folo2_bytes, folo2_iou_bytes, folo2_o2o_bytes, has_gpu, load_with_imgsz, model_bytes, rgba};

fn predictor_on(bytes: Vec<u8>, gpu: bool, imgsz: u32, opts: &DetectOptions) -> Predictor {
    let file = ModelFile::from_bytes(bytes).unwrap();
    let (mut m, _) = VisionDetector::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    if gpu {
        let dev = folo_gpu::init().unwrap();
        m.to_device(dev).unwrap();
    }
    let cfg = PredictConfig {
        imgsz,
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        rect: true,
        head: None,
        ..PredictConfig::default()
    };
    let mut p = Predictor::from_detector(m, cfg).unwrap();
    p.set_per_class_conf(opts.per_class_conf.clone()).unwrap();
    p.set_conf_inclusive(opts.conf_inclusive);
    p
}

fn check_variant_thresholds(label: &str, bytes: Vec<u8>, backend: BackendChoice) {
    let is_gpu = backend == BackendChoice::Webgpu;
    let imgsz = 128;
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();

    let det = load_with_imgsz(bytes.clone(), backend, Some(Precision::F32), imgsz).unwrap();

    // Probe detections at a low threshold to pick an exact boundary score and exercise per-class filtering.
    let probe = pollster::block_on(det.detect(
        &img,
        w,
        h,
        DetectOptions { conf: 0.1, iou: 0.7, max_det: 60, ..Default::default() },
    ))
    .unwrap();
    assert!(!probe.response.boxes.is_empty(), "{label}: probe needs detections");

    // Pick a real box to sit right on the exact per-class boundary (`score == threshold`).
    let boundary = &probe.response.boxes[probe.response.boxes.len() / 2];
    let other_class = (boundary.class + 1) % 3;
    let third_class = (boundary.class + 2) % 3;
    let mut per_class = BTreeMap::new();
    // Exact boundary score for `boundary.class`: excluded when `conf_inclusive = false`, included when `true`.
    per_class.insert(boundary.class, boundary.score);
    // High threshold suppressing `other_class`.
    per_class.insert(other_class, 0.9999);
    // Low threshold admitting `third_class` below fallback `conf = 0.55`.
    per_class.insert(third_class, 0.15);

    let fallback_conf = 0.55;
    let iou = 0.7;
    let max_det = 40;

    let mut strict_boxes_single = None;
    let mut inclusive_boxes_single = None;

    for conf_inclusive in [false, true] {
        // Single-pass parity.
        let opts = DetectOptions {
            conf: fallback_conf,
            iou,
            max_det,
            per_class_conf: Some(per_class.clone()),
            conf_inclusive,
        };
        let p = predictor_on(bytes.clone(), is_gpu, imgsz, &opts);
        let got = pollster::block_on(det.detect(&img, w, h, opts)).unwrap();
        let want = p.predict(&rgb).unwrap();
        assert_eq!(
            got.response.boxes,
            want.scored_boxes(p.classes()),
            "{label} ({backend:?}, inclusive={conf_inclusive}): single detect mismatch"
        );
        if conf_inclusive {
            inclusive_boxes_single = Some(got.response.boxes.clone());
        } else {
            strict_boxes_single = Some(got.response.boxes.clone());
        }

        // SAHI tiled parity (multi-tile + full-image pass).
        let tile_opts = TileDetectOptions {
            conf: fallback_conf,
            iou,
            max_det,
            per_class_conf: Some(per_class.clone()),
            conf_inclusive,
            tile_size: Some(112),
            overlap: Some(0.25),
            full_image: true,
            tile_batch: 4,
            ..TileDetectOptions::default()
        };
        let got_t = pollster::block_on(det.detect_tiled(&img, w, h, tile_opts.clone())).unwrap();
        let want_t = p.predict_tiled(&rgb, &tile_opts.to_tile_config().unwrap()).unwrap();
        assert_eq!(
            got_t.response.boxes,
            want_t.scored_boxes(p.classes()),
            "{label} ({backend:?}, inclusive={conf_inclusive}): tiled detect mismatch"
        );

        // TTA parity (multi-scale + hflip).
        let tta_opts = TtaDetectOptions {
            conf: fallback_conf,
            iou,
            max_det,
            per_class_conf: Some(per_class.clone()),
            conf_inclusive,
            hflip: true,
            scales: vec![0.75, 1.0],
            ..TtaDetectOptions::default()
        };
        let got_a = pollster::block_on(det.detect_tta(&img, w, h, tta_opts.clone())).unwrap();
        let want_a = p.predict_tta(&rgb, &tta_opts.to_tta_config().unwrap()).unwrap();
        assert_eq!(
            got_a.response.boxes,
            want_a.scored_boxes(p.classes()),
            "{label} ({backend:?}, inclusive={conf_inclusive}): TTA detect mismatch"
        );
    }

    // Verify `conf_inclusive: true` strictly includes the exact-boundary box excluded by `conf_inclusive: false`.
    let strict_b = strict_boxes_single.unwrap();
    let incl_b = inclusive_boxes_single.unwrap();
    let has_boundary = |boxes: &[folo_api::ScoredBox]| {
        boxes.iter().any(|b| b.class == boundary.class && b.score.to_bits() == boundary.score.to_bits())
    };
    assert!(
        !has_boundary(&strict_b),
        "{label} ({backend:?}): strict `>` must exclude exact boundary score {}",
        boundary.score
    );
    assert!(
        has_boundary(&incl_b),
        "{label} ({backend:?}): inclusive `>=` must keep exact boundary score {}",
        boundary.score
    );

    // Boundary `conf = 0.0` with `conf_inclusive = true` exercises `strict_floor(0.0) = -f32::MIN_POSITIVE`.
    let zero_opts = DetectOptions { conf: 0.0, iou, max_det: 15, per_class_conf: None, conf_inclusive: true };
    let p_zero = predictor_on(bytes, is_gpu, imgsz, &zero_opts);
    let got_zero = pollster::block_on(det.detect(&img, w, h, zero_opts)).unwrap();
    let want_zero = p_zero.predict(&rgb).unwrap();
    assert_eq!(
        got_zero.response.boxes,
        want_zero.scored_boxes(p_zero.classes()),
        "{label} ({backend:?}): conf=0.0 inclusive mismatch"
    );

    // Empty `per_class_conf: Some({})` matches `per_class_conf: None`.
    let empty_map = pollster::block_on(det.detect(
        &img,
        w,
        h,
        DetectOptions {
            conf: fallback_conf,
            iou,
            max_det,
            per_class_conf: Some(BTreeMap::new()),
            conf_inclusive: false,
        },
    ))
    .unwrap();
    let none_map = pollster::block_on(det.detect(
        &img,
        w,
        h,
        DetectOptions { conf: fallback_conf, iou, max_det, per_class_conf: None, conf_inclusive: false },
    ))
    .unwrap();
    assert_eq!(empty_map.response.boxes, none_map.response.boxes);
}

#[test]
fn per_class_conf_and_conf_inclusive_match_predictor_cpu() {
    for (label, bytes) in [
        ("folo1-o2m", model_bytes()),
        ("folo2-o2m", folo2_bytes()),
        ("folo2-iou_head", folo2_iou_bytes()),
        ("folo2-o2o", folo2_o2o_bytes()),
    ] {
        check_variant_thresholds(label, bytes, BackendChoice::Cpu);
    }
}

#[test]
fn per_class_conf_and_conf_inclusive_match_predictor_gpu() {
    if !has_gpu() {
        return;
    }
    for (label, bytes) in [
        ("folo1-o2m", model_bytes()),
        ("folo2-o2m", folo2_bytes()),
        ("folo2-iou_head", folo2_iou_bytes()),
        ("folo2-o2o", folo2_o2o_bytes()),
    ] {
        check_variant_thresholds(label, bytes, BackendChoice::Webgpu);
    }
}

#[test]
fn per_class_conf_json_parsing_and_invalid_rejection() {
    let d: DetectOptions =
        serde_json::from_str(r#"{"conf":0.4,"perClassConf":{"0":0.25,"2":0.8},"confInclusive":true}"#)
            .unwrap();
    assert!(d.conf_inclusive);
    assert_eq!(d.per_class_conf, Some(BTreeMap::from([(0, 0.25), (2, 0.8)])));

    let t: TileDetectOptions =
        serde_json::from_str(r#"{"tileSize":320,"perClassConf":{"1":0.6},"confInclusive":true}"#).unwrap();
    assert!(t.conf_inclusive);
    assert_eq!(t.per_class_conf, Some(BTreeMap::from([(1, 0.6)])));

    let a: TtaDetectOptions =
        serde_json::from_str(r#"{"hflip":false,"perClassConf":{"0":0.0,"3":1.0},"confInclusive":true}"#)
            .unwrap();
    assert!(a.conf_inclusive);
    assert_eq!(a.per_class_conf, Some(BTreeMap::from([(0, 0.0), (3, 1.0)])));

    // Null and empty object parse cleanly.
    let d_null: DetectOptions = serde_json::from_str(r#"{"perClassConf":null}"#).unwrap();
    assert_eq!(d_null.per_class_conf, None);
    let d_empty: DetectOptions = serde_json::from_str(r#"{"perClassConf":{}}"#).unwrap();
    assert_eq!(d_empty.per_class_conf, Some(BTreeMap::new()));

    // Invalid keys, non-object shapes, and out-of-range f64 values reject cleanly.
    for bad in [
        r#"{"perClassConf":[]}"#,
        r#"{"perClassConf":{"-1":0.5}}"#,
        r#"{"perClassConf":{"+0":0.5}}"#,
        r#"{"perClassConf":{"00":0.5}}"#,
        r#"{"perClassConf":{"01":0.5}}"#,
        r#"{"perClassConf":{"1.5":0.5}}"#,
        r#"{"perClassConf":{"abc":0.5}}"#,
        r#"{"perClassConf":{"4294967296":0.5}}"#,
        r#"{"perClassConf":{"0":1.00000001}}"#,
        r#"{"perClassConf":{"0":-1e-12}}"#,
        r#"{"perClassConf":{"0":"0.5"}}"#,
        r#"{"perClassConf":{"0":null}}"#,
        r#"{"confInclusive":"true"}"#,
    ] {
        assert!(serde_json::from_str::<DetectOptions>(bad).is_err(), "expected DetectOptions err for {bad}");
        assert!(
            serde_json::from_str::<TileDetectOptions>(bad).is_err(),
            "expected TileDetectOptions err for {bad}"
        );
        assert!(
            serde_json::from_str::<TtaDetectOptions>(bad).is_err(),
            "expected TtaDetectOptions err for {bad}"
        );
    }

    // Programmatically constructed invalid `per_class_conf` values reject at detect/tiled/tta time.
    let det = load_with_imgsz(model_bytes(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
    let img = [0u8; 4 * 4 * 4];
    for bad_val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.01, 1.01] {
        let bad_map = Some(BTreeMap::from([(0, bad_val)]));
        assert!(
            pollster::block_on(det.detect(
                &img,
                4,
                4,
                DetectOptions { per_class_conf: bad_map.clone(), ..Default::default() }
            ))
            .is_err()
        );
        assert!(
            pollster::block_on(det.detect_tiled(
                &img,
                4,
                4,
                TileDetectOptions { per_class_conf: bad_map.clone(), ..Default::default() }
            ))
            .is_err()
        );
        assert!(
            pollster::block_on(det.detect_tta(
                &img,
                4,
                4,
                TtaDetectOptions { per_class_conf: bad_map, ..Default::default() }
            ))
            .is_err()
        );
    }
}
