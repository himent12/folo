//! Quantized int8 `.folo` loading, `DetectorInfo` reporting/fallback behavior, and CPU inference parity
//! against `folo_vision::Predictor` for both `folo-*` (v1) and `folo2-*` (v2-Iso).

use std::collections::BTreeMap;

use folo_core::Tensor;
use folo_data::{Letterbox, LetterboxOptions, PAD_VALUE};
use folo_nn::io::{ModelFile, Purpose, save_weights};
use folo_test_support::TestDir;
use folo_vision::model::quant::{META_KEY, Method, calibrate, calibrate_v2};
use folo_vision::{Detector as VisionDetector, FoloModel, FoloV2Model, PredictConfig, Predictor};
use folo_wasm::detector::rgba_to_image;
use folo_wasm::{
    Backend, BackendChoice, DetectOptions, Detector, LoadOptions, Precision, TileDetectOptions,
    TtaDetectOptions,
};

use super::{folo2_bytes, has_gpu, load_with_imgsz, model_bytes, rgba};

fn calib_inputs() -> Vec<Tensor> {
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();
    let lb = Letterbox::compute(w, h, 128, 128, &LetterboxOptions { stride: Some(32), ..Default::default() });
    let [ow, oh] = lb.out.map(|v| v as usize);
    let mut buf = vec![0.0f32; 3 * ow * oh];
    lb.apply_planar_f32(&rgb, PAD_VALUE, 1.0 / 255.0, &mut buf);
    vec![Tensor::from_vec(buf, [1, 3, oh, ow]).unwrap()]
}

fn make_quantized_folo1() -> Vec<u8> {
    let file = ModelFile::from_bytes(model_bytes()).unwrap();
    let (mut m, _) = FoloModel::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let inputs = calib_inputs();
    let cal = calibrate(&m, || inputs.iter().cloned().map(Ok), Method::MinMax).unwrap();
    m.set_int8(Some(cal)).unwrap();
    let dir = TestDir::new("wasm-int8-v1");
    let path = dir.join("m1-int8.folo");
    m.save(&path).unwrap();
    std::fs::read(&path).unwrap()
}

fn make_quantized_folo2() -> Vec<u8> {
    let file = ModelFile::from_bytes(folo2_bytes()).unwrap();
    let (mut m, _) = FoloV2Model::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let inputs = calib_inputs();
    let cal = calibrate_v2(&m, || inputs.iter().cloned().map(Ok), Method::MinMax).unwrap();
    m.set_int8(Some(cal)).unwrap();
    let dir = TestDir::new("wasm-int8-v2");
    let path = dir.join("m2-int8.folo");
    m.save(&path).unwrap();
    std::fs::read(&path).unwrap()
}

fn folo1_int8_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(make_quantized_folo1).clone()
}

fn folo2_int8_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(make_quantized_folo2).clone()
}

/// Write a valid `.folo` file whose `folo.quant` metadata JSON is malformed, to test error vs fallback paths.
fn corrupt_quant_bytes() -> Vec<u8> {
    let file = ModelFile::from_bytes(model_bytes()).unwrap();
    let (mut m, _) = FoloModel::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let mut meta = m.metadata().unwrap();
    meta.extra.insert(META_KEY.to_string(), "{not-valid-calibration-json".to_string());
    let dir = TestDir::new("wasm-int8-corrupt");
    let path = dir.join("corrupt-int8.folo");
    save_weights(&path, &meta, &m).unwrap();
    std::fs::read(&path).unwrap()
}

fn int8_predictor(bytes: Vec<u8>, imgsz: u32, conf: f32, iou: f32, max_det: usize) -> Predictor {
    let file = ModelFile::from_bytes(bytes).unwrap();
    let model = VisionDetector::from_file_int8(&file).unwrap();
    let cfg = PredictConfig { imgsz, conf, iou, max_det, rect: true, head: None, ..PredictConfig::default() };
    Predictor::from_detector(model, cfg).unwrap()
}

#[test]
fn int8_folo1_and_folo2_match_predictor_across_single_tiled_tta() {
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();
    let (conf, iou, max_det) = (0.3, 0.7, 40);

    for (label, bytes, want_family, want_arch) in [
        ("folo1-int8", folo1_int8_bytes(), "folo", "folo-n"),
        ("folo2-int8", folo2_int8_bytes(), "folo2", "folo2-n"),
    ] {
        // Explicit `precision: "int8"` on CPU.
        let det_explicit =
            load_with_imgsz(bytes.clone(), BackendChoice::Cpu, Some(Precision::Int8), 128).unwrap();
        let info = det_explicit.info();
        assert_eq!(info.backend, Backend::Cpu);
        assert_eq!(info.precision, Precision::Int8, "{label}: expected int8 precision");
        assert!(info.precision_fallback.is_none(), "{label}: unexpected precision_fallback {info:?}");
        assert!(info.fallback.is_none(), "{label}: unexpected fallback {info:?}");
        assert_eq!((info.family.as_str(), info.arch.as_str()), (want_family, want_arch));

        // Omitted `precision` on CPU auto-activates `int8` when `folo.quant` is present.
        let det_auto_cpu = load_with_imgsz(bytes.clone(), BackendChoice::Cpu, None, 128).unwrap();
        assert_eq!(det_auto_cpu.info().precision, Precision::Int8);
        assert!(det_auto_cpu.info().precision_fallback.is_none());

        // Explicit `precision: "int8"` with `backend: "auto"` targets CPU directly without WebGPU.
        let det_auto_backend =
            load_with_imgsz(bytes.clone(), BackendChoice::Auto, Some(Precision::Int8), 128).unwrap();
        let auto_info = det_auto_backend.info();
        assert_eq!(auto_info.backend, Backend::Cpu);
        assert_eq!(auto_info.precision, Precision::Int8);
        assert!(auto_info.precision_fallback.is_none());
        assert!(
            auto_info.fallback.as_deref().is_some_and(|w| w.contains("CPU")),
            "{label}: auto+int8 should report why auto picked CPU: {auto_info:?}"
        );

        let p = int8_predictor(bytes.clone(), 128, conf, iou, max_det);
        let opts = DetectOptions { conf, iou, max_det, ..Default::default() };

        // Single-pass bit-for-bit parity against `Predictor` (`from_file_int8`).
        let got = pollster::block_on(det_explicit.detect(&img, w, h, opts.clone())).unwrap();
        let want = p.predict(&rgb).unwrap();
        assert!(!want.detections.is_empty(), "{label}: int8 test needs detections");
        assert_eq!(got.response.boxes, want.scored_boxes(p.classes()), "{label}: single detect mismatch");

        let got_auto_cpu = pollster::block_on(det_auto_cpu.detect(&img, w, h, opts.clone())).unwrap();
        assert_eq!(got_auto_cpu.response.boxes, got.response.boxes);
        let got_auto_b = pollster::block_on(det_auto_backend.detect(&img, w, h, opts)).unwrap();
        assert_eq!(got_auto_b.response.boxes, got.response.boxes);

        // SAHI tiled parity.
        let tile_opts = TileDetectOptions {
            conf,
            iou,
            max_det,
            tile_size: Some(112),
            overlap: Some(0.25),
            full_image: true,
            tile_batch: 4,
            ..Default::default()
        };
        let got_t = pollster::block_on(det_explicit.detect_tiled(&img, w, h, tile_opts.clone())).unwrap();
        let want_t = p.predict_tiled(&rgb, &tile_opts.to_tile_config().unwrap()).unwrap();
        assert_eq!(got_t.response.boxes, want_t.scored_boxes(p.classes()), "{label}: tiled mismatch");

        // TTA parity.
        let tta_opts = TtaDetectOptions {
            conf,
            iou,
            max_det,
            hflip: true,
            scales: vec![0.75, 1.0],
            ..Default::default()
        };
        let got_a = pollster::block_on(det_explicit.detect_tta(&img, w, h, tta_opts.clone())).unwrap();
        let want_a = p.predict_tta(&rgb, &tta_opts.to_tta_config().unwrap()).unwrap();
        assert_eq!(got_a.response.boxes, want_a.scored_boxes(p.classes()), "{label}: TTA mismatch");

        // Combined int8 + perClassConf + confInclusive parity.
        let boundary = &got.response.boxes[got.response.boxes.len() / 2];
        let per_class =
            BTreeMap::from([(boundary.class, boundary.score), ((boundary.class + 1) % 3, 0.9999)]);
        let mut p_cal = int8_predictor(bytes, 128, 0.5, iou, max_det);
        p_cal.set_per_class_conf(Some(per_class.clone())).unwrap();
        p_cal.set_conf_inclusive(true);
        let got_cal = pollster::block_on(det_explicit.detect(
            &img,
            w,
            h,
            DetectOptions { conf: 0.5, iou, max_det, per_class_conf: Some(per_class), conf_inclusive: true },
        ))
        .unwrap();
        let want_cal = p_cal.predict(&rgb).unwrap();
        assert_eq!(
            got_cal.response.boxes,
            want_cal.scored_boxes(p_cal.classes()),
            "{label}: int8 + perClassConf + confInclusive mismatch"
        );
    }
}

#[test]
fn int8_precision_overrides_fallbacks_and_errors() {
    let q_bytes = folo1_int8_bytes();
    let plain_bytes = model_bytes();
    let corrupt_bytes = corrupt_quant_bytes();

    // Explicit `precision: "f32"` on a quantized checkpoint forces F32 (does not activate int8).
    let f32_det = load_with_imgsz(q_bytes.clone(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
    assert_eq!(f32_det.info().precision, Precision::F32);
    assert!(f32_det.info().precision_fallback.is_none());

    // Explicit `precision: "int8"` on an unquantized checkpoint rejects cleanly.
    for backend in [BackendChoice::Cpu, BackendChoice::Auto] {
        let err = load_with_imgsz(plain_bytes.clone(), backend, Some(Precision::Int8), 128)
            .err()
            .expect("unquantized file must fail explicit int8");
        assert!(err.contains(META_KEY), "expected `{META_KEY}` in error, got: {err}");
    }

    // Corrupt `folo.quant` rejects on explicit `precision: "int8"`, and falls back to f32 when `precision` is omitted.
    assert!(load_with_imgsz(corrupt_bytes.clone(), BackendChoice::Cpu, Some(Precision::Int8), 128).is_err());
    let fallback_det = load_with_imgsz(corrupt_bytes, BackendChoice::Cpu, None, 128).unwrap();
    let fb_info = fallback_det.info();
    assert_eq!(fb_info.precision, Precision::F32);
    assert!(
        fb_info.precision_fallback.as_deref().is_some_and(|w| w.contains("int8")),
        "expected int8 fallback reason: {fb_info:?}"
    );

    // Explicit `backend: "webgpu"` + `precision: "int8"` runs in f32 on WebGPU with `precisionFallback`.
    if has_gpu() {
        let gpu_int8 =
            load_with_imgsz(q_bytes.clone(), BackendChoice::Webgpu, Some(Precision::Int8), 128).unwrap();
        let ginfo = gpu_int8.info();
        assert_eq!(ginfo.backend, Backend::Webgpu);
        assert_eq!(ginfo.precision, Precision::F32);
        assert!(
            ginfo.precision_fallback.as_deref().is_some_and(|w| w.contains("CPU") && w.contains("WebGPU")),
            "expected CPU/WebGPU fallback reason: {ginfo:?}"
        );

        // `backend: "auto"` with omitted precision on a WebGPU host uses WebGPU f32 cleanly.
        let auto_gpu = pollster::block_on(Detector::load(
            q_bytes,
            LoadOptions {
                backend: BackendChoice::Auto,
                model_id: Some("m_q".into()),
                imgsz: Some(128),
                precision: None,
            },
        ))
        .unwrap();
        assert_eq!(auto_gpu.backend(), Backend::Webgpu);
        assert_eq!(auto_gpu.info().precision, Precision::F32);
        assert!(auto_gpu.info().precision_fallback.is_none());
    }
}
