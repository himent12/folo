//! folo-wasm's detector, run natively: single-pass, SAHI tiled inference, and TTA on CPU against folo-vision's
//! `Predictor` (identical results), GPU paths (`download_async` readback) against the CPU paths, invalid
//! parameter rejection, repeated-frame scratch reuse, and JS options JSON parsing. The GPU tests skip without a
//! hardware adapter, like folo-vision's.

mod int8;
mod thresholds;

use folo_core::no_grad;
use folo_core::rng::Rng;
use folo_data::Image;
use folo_nn::Module;
use folo_nn::io::{ModelFile, Purpose};
use folo_test_support::TestDir;
use folo_vision::fusion::BoxFusion;
use folo_vision::model::HeadMode;
use folo_vision::{
    Detector as VisionDetector, FoloModel, FoloV2Config, FoloV2Model, ModelConfig, PredictConfig, Predictor,
    TileConfig, TileMergeMetric, TtaConfig,
};
use folo_wasm::detector::rgba_to_image;
use folo_wasm::{
    Backend, BackendChoice, DetectOptions, Detector, LoadOptions, Precision, TileDetectOptions,
    TtaDetectOptions,
};

/// A random folo-n with its class-output biases zeroed: the trained priors (MODEL.md §6.5) put every score near
/// 1e-4, and the tests need detections. Made once.
fn model_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(make_model).clone()
}

fn make_model() -> Vec<u8> {
    let mut m = FoloModel::with_rng(ModelConfig::new("n", 3).unwrap(), &mut Rng::seed(7)).unwrap();
    m.set_classes(vec!["circle".into(), "square".into(), "triangle".into()]).unwrap();
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            if t.name.starts_with("head.cls.") && t.name.ends_with(".out.bias") {
                t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
            }
        }
    }
    let dir = TestDir::new("wasm-test");
    let path = dir.join("m.folo");
    m.save(&path).unwrap();
    std::fs::read(&path).unwrap()
}

/// A smooth RGBA test pattern (alpha varies too: it must be ignored).
fn rgba(w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&[
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) % 256) as u8,
                (x % 7) as u8,
            ]);
        }
    }
    v
}

fn default_opts() -> DetectOptions {
    DetectOptions { conf: 0.3, iou: 0.7, max_det: 40, per_class_conf: None, conf_inclusive: false }
}

fn load(bytes: Vec<u8>, backend: BackendChoice) -> Result<Detector, String> {
    load_at(bytes, backend, Some(Precision::F32))
}

fn load_at(bytes: Vec<u8>, backend: BackendChoice, precision: Option<Precision>) -> Result<Detector, String> {
    load_with_imgsz(bytes, backend, precision, 256)
}

fn load_with_imgsz(
    bytes: Vec<u8>,
    backend: BackendChoice,
    precision: Option<Precision>,
    imgsz: u32,
) -> Result<Detector, String> {
    pollster::block_on(Detector::load(
        bytes,
        LoadOptions { backend, model_id: Some("m_test".into()), imgsz: Some(imgsz), precision },
    ))
}

fn cpu_predictor(bytes: Vec<u8>, imgsz: u32, conf: f32, iou: f32, max_det: usize) -> Predictor {
    let file = ModelFile::from_bytes(bytes).unwrap();
    let (mut m, _) = FoloModel::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let cfg = PredictConfig { imgsz, conf, iou, max_det, rect: true, head: None, ..PredictConfig::default() };
    Predictor::new(m, cfg).unwrap()
}

fn has_gpu() -> bool {
    match folo_gpu::GpuContext::shared() {
        Ok(ctx) if ctx.summary().device_type != "Cpu" || std::env::var("FOLO_GPU_TEST_SOFTWARE").is_ok() => {
            true
        }
        Ok(ctx) => {
            eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
            false
        }
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            false
        }
    }
}

fn assert_gpu_matches_cpu_boxes(
    gpu_boxes: &[folo_api::ScoredBox],
    cpu_boxes: &[folo_api::ScoredBox],
    score_tol: f32,
    coord_tol: f32,
    label: &str,
) {
    assert!(!cpu_boxes.is_empty(), "{label}: test needs CPU boxes");
    assert_eq!(gpu_boxes.len(), cpu_boxes.len(), "{label}: box count mismatch");
    for want in cpu_boxes {
        let close = gpu_boxes.iter().any(|b| {
            b.class == want.class
                && (b.score - want.score).abs() < score_tol
                && [b.x1 - want.x1, b.y1 - want.y1, b.x2 - want.x2, b.y2 - want.y2]
                    .iter()
                    .all(|d| d.abs() < coord_tol)
        });
        assert!(close, "{label}: no GPU box matches CPU box {want:?}");
    }
}

#[test]
fn cpu_matches_predictor() {
    let bytes = model_bytes();
    let det = load(bytes.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(det.backend(), Backend::Cpu);
    let (w, h) = (200, 120);
    let img = rgba(w, h);
    let res = pollster::block_on(det.detect(&img, w, h, default_opts())).unwrap();

    let p = cpu_predictor(bytes, 256, default_opts().conf, default_opts().iou, default_opts().max_det);
    let want = p.predict(&rgba_to_image(&img, w, h).unwrap()).unwrap();
    assert!(!want.detections.is_empty(), "the test needs some detections");
    assert_eq!(res.response.boxes, want.scored_boxes(p.classes()));
    assert_eq!(res.response.model_id, "m_test");
    assert_eq!(res.response.image, folo_api::ImageSize { width: w, height: h });
    assert_eq!(res.response.device, "cpu");

    // docs/API.md's /api/predict shape plus `backend`.
    let v = serde_json::to_value(&res).unwrap();
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["backend", "boxes", "device", "image", "model_id", "timing"]);
    assert_eq!(v["backend"], "cpu");
    let t = &v["timing"];
    assert!(t["total_ms"].as_f64().unwrap() >= t["inference_ms"].as_f64().unwrap());

    let info = det.info();
    assert_eq!(
        (
            info.family.as_str(),
            info.arch.as_str(),
            info.head_mode.as_str(),
            info.has_o2o,
            info.has_iou_head,
            info.imgsz,
            info.classes.len(),
        ),
        ("folo", "folo-n", "o2m", false, false, 256, 3)
    );
    assert!(info.fallback.is_none() && info.adapter.is_none());

    // Repeated frames reuse internal scratch buffers cleanly and return bit-identical boxes.
    let res2 = pollster::block_on(det.detect(&img, w, h, default_opts())).unwrap();
    assert_eq!(res2.response.boxes, res.response.boxes);
}

#[test]
fn cpu_tiled_matches_predictor() {
    let bytes = model_bytes();
    let det = load_with_imgsz(bytes.clone(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
    let opts = default_opts();
    let p = cpu_predictor(bytes, 128, opts.conf, opts.iou, opts.max_det);

    let (w, h) = (260, 180);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();

    // Multi-tile with full-image pass and default IoS + Vote fusion.
    let tile_opts = TileDetectOptions {
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        tile_size: Some(140),
        overlap: Some(0.25),
        full_image: true,
        tile_batch: 4,
        merge_metric: TileMergeMetric::Ios,
        merge_threshold: 0.5,
        merge_fusion: BoxFusion::Vote,
        class_agnostic: false,
        ..TileDetectOptions::default()
    };
    let tile_cfg = tile_opts.to_tile_config().unwrap();
    let got = pollster::block_on(det.detect_tiled(&img, w, h, tile_opts.clone())).unwrap();
    let want = p.predict_tiled(&rgb, &tile_cfg).unwrap();
    assert!(!want.detections.is_empty(), "tiled test needs detections");
    assert_eq!(got.response.boxes, want.scored_boxes(p.classes()));
    assert_eq!(got.response.image, folo_api::ImageSize { width: w, height: h });
    assert_eq!(got.backend, Backend::Cpu);

    // Multi-tile without full-image pass, using IoU merge and Sebf fusion.
    let tile_opts_no_full = TileDetectOptions {
        full_image: false,
        merge_metric: TileMergeMetric::Iou,
        merge_fusion: BoxFusion::Sebf,
        ..tile_opts
    };
    let tile_cfg_no_full = tile_opts_no_full.to_tile_config().unwrap();
    let got_nf = pollster::block_on(det.detect_tiled(&img, w, h, tile_opts_no_full)).unwrap();
    let want_nf = p.predict_tiled(&rgb, &tile_cfg_no_full).unwrap();
    assert_eq!(got_nf.response.boxes, want_nf.scored_boxes(p.classes()));

    // Single-tile degradation (tile larger than image) matches single-pass predict bit for bit.
    let single_opts = TileDetectOptions {
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        tile_size: Some(512),
        ..TileDetectOptions::default()
    };
    let single_cfg: TileConfig = single_opts.to_tile_config().unwrap();
    let got_single = pollster::block_on(det.detect_tiled(&img, w, h, single_opts)).unwrap();
    let want_single = p.predict_tiled(&rgb, &single_cfg).unwrap();
    assert_eq!(got_single.response.boxes, want_single.scored_boxes(p.classes()));
}

#[test]
fn cpu_tta_matches_predictor() {
    let bytes = model_bytes();
    let det = load_with_imgsz(bytes.clone(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
    let opts = default_opts();
    let p = cpu_predictor(bytes, 128, opts.conf, opts.iou, opts.max_det);

    let (w, h) = (180, 120);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();

    // Multi-scale + horizontal flip TTA (resolves 128 * [0.75, 1.0, 1.25] -> [96, 128, 160]).
    let tta_opts = TtaDetectOptions {
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        hflip: true,
        scales: vec![0.75, 1.0, 1.25],
        merge_iou: Some(0.55),
        merge_fusion: BoxFusion::Vote,
        class_agnostic: false,
        ..TtaDetectOptions::default()
    };
    let tta_cfg: TtaConfig = tta_opts.to_tta_config().unwrap();
    let got = pollster::block_on(det.detect_tta(&img, w, h, tta_opts.clone())).unwrap();
    let want = p.predict_tta(&rgb, &tta_cfg).unwrap();
    assert!(!want.detections.is_empty(), "TTA test needs detections");
    assert_eq!(got.response.boxes, want.scored_boxes(p.classes()));
    assert_eq!(det.info().imgsz, 128, "TTA must never corrupt base detector imgsz");

    // Flip-only TTA with Sebf fusion.
    let flip_opts = TtaDetectOptions {
        scales: vec![1.0],
        hflip: true,
        merge_fusion: BoxFusion::Sebf,
        ..tta_opts.clone()
    };
    let got_flip = pollster::block_on(det.detect_tta(&img, w, h, flip_opts.clone())).unwrap();
    let want_flip = p.predict_tta(&rgb, &flip_opts.to_tta_config().unwrap()).unwrap();
    assert_eq!(got_flip.response.boxes, want_flip.scored_boxes(p.classes()));

    // Single unflipped view degrades cleanly to single-pass inference at the scaled imgsz.
    let single_opts = TtaDetectOptions { scales: vec![1.25], hflip: false, ..tta_opts };
    let got_single = pollster::block_on(det.detect_tta(&img, w, h, single_opts.clone())).unwrap();
    let want_single = p.predict_tta(&rgb, &single_opts.to_tta_config().unwrap()).unwrap();
    assert_eq!(got_single.response.boxes, want_single.scored_boxes(p.classes()));
    assert_eq!(det.info().imgsz, 128);
}

#[test]
fn gpu_matches_cpu() {
    if !has_gpu() {
        return;
    }
    let bytes = model_bytes();
    let gpu = load(bytes.clone(), BackendChoice::Webgpu).unwrap();
    assert_eq!(gpu.backend(), Backend::Webgpu);
    let cpu = load(bytes, BackendChoice::Cpu).unwrap();
    let (w, h) = (160, 224);
    let img = rgba(w, h);
    let g = pollster::block_on(gpu.detect(&img, w, h, default_opts())).unwrap();
    let c = pollster::block_on(cpu.detect(&img, w, h, default_opts())).unwrap();
    assert_eq!(g.response.device, "gpu:0");
    assert_eq!(g.backend, Backend::Webgpu);
    assert_gpu_matches_cpu_boxes(&g.response.boxes, &c.response.boxes, 1e-3, 0.05, "single");

    // Exercise the >CANDIDATE_READBACK overflow readback path: a 256x256 canvas has 1,344 anchors
    // (32^2 + 16^2 + 8^2 > 1,024), so conf = 0.0 overflows the 1,024-record first download.
    let (ow, oh) = (256, 256);
    let sq = rgba(ow, oh);
    let overflow = DetectOptions { conf: 0.0, iou: 0.7, max_det: 50, ..Default::default() };
    let go = pollster::block_on(gpu.detect(&sq, ow, oh, overflow.clone())).unwrap();
    let co = pollster::block_on(cpu.detect(&sq, ow, oh, overflow)).unwrap();
    assert_gpu_matches_cpu_boxes(&go.response.boxes, &co.response.boxes, 1e-3, 0.05, "overflow");
}

#[test]
fn gpu_tiled_and_tta_match_predictor() {
    if !has_gpu() {
        return;
    }
    let bytes = model_bytes();
    let gpu = load_with_imgsz(bytes.clone(), BackendChoice::Webgpu, Some(Precision::F32), 128).unwrap();
    let opts = default_opts();

    // Compare async batched WebGPU `detect_tiled` and `detect_tta` bit-for-bit against native GPU `Predictor`.
    let gpu_dev = folo_gpu::init().unwrap();
    let file = ModelFile::from_bytes(bytes).unwrap();
    let (mut gm, _) = FoloModel::from_file(&file, Purpose::Inference).unwrap();
    gm.fuse().unwrap();
    gm.to_device(gpu_dev).unwrap();
    let gpu_pred = Predictor::new(
        gm,
        PredictConfig {
            imgsz: 128,
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            ..Default::default()
        },
    )
    .unwrap();

    // Batched WebGPU SAHI tiled inference (with and without full-image pass).
    let (tw, th) = (240, 180);
    let timg = rgba(tw, th);
    let trgb = rgba_to_image(&timg, tw, th).unwrap();
    for full_image in [true, false] {
        let tile_opts = TileDetectOptions {
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            tile_size: Some(140),
            overlap: Some(0.25),
            full_image,
            tile_batch: 4,
            ..TileDetectOptions::default()
        };
        let tile_cfg = tile_opts.to_tile_config().unwrap();
        let gt = pollster::block_on(gpu.detect_tiled(&timg, tw, th, tile_opts)).unwrap();
        let want = gpu_pred.predict_tiled(&trgb, &tile_cfg).unwrap();
        assert_eq!(gt.backend, Backend::Webgpu);
        assert!(!want.detections.is_empty(), "GPU tiled test needs detections");
        assert_eq!(
            gt.response.boxes,
            want.scored_boxes(gpu_pred.classes()),
            "async WebGPU tiled (full_image={full_image}) must match native GPU Predictor bit for bit"
        );
    }

    // Batched WebGPU TTA (multi-scale + hflip) and single-view degradation.
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();
    let tta_opts = TtaDetectOptions {
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        hflip: true,
        scales: vec![0.75, 1.0, 1.25],
        ..TtaDetectOptions::default()
    };
    let tta_cfg = tta_opts.to_tta_config().unwrap();
    let ga = pollster::block_on(gpu.detect_tta(&img, w, h, tta_opts)).unwrap();
    let want_tta = gpu_pred.predict_tta(&rgb, &tta_cfg).unwrap();
    assert_eq!(ga.backend, Backend::Webgpu);
    assert_eq!(gpu.info().imgsz, 128);
    assert!(!want_tta.detections.is_empty(), "GPU TTA test needs detections");
    assert_eq!(
        ga.response.boxes,
        want_tta.scored_boxes(gpu_pred.classes()),
        "async WebGPU TTA must match native GPU Predictor bit for bit"
    );
}

/// `precision: "f16"` on WebGPU runs the model in f16 (with shader-f16) and still matches the f32 CPU boxes, to
/// f16's rounding; on the CPU it runs in f32 and says why.
#[test]
fn f16_precision_option() {
    let bytes = model_bytes();
    let cpu = load_at(bytes.clone(), BackendChoice::Cpu, Some(Precision::F16)).unwrap();
    let info = cpu.info();
    assert_eq!(info.precision, Precision::F32);
    assert!(info.precision_fallback.as_deref().is_some_and(|w| w.contains("GPU")), "{info:?}");
    if !has_gpu() {
        return;
    }
    let gpu = load_at(bytes, BackendChoice::Webgpu, Some(Precision::F16)).unwrap();
    let info = gpu.info();
    if let Some(why) = &info.precision_fallback {
        return eprintln!("note: {why}");
    }
    assert_eq!(info.precision, Precision::F16);
    let (w, h) = (160, 224);
    let img = rgba(w, h);
    let opts = default_opts();
    let g = pollster::block_on(gpu.detect(&img, w, h, opts.clone())).unwrap();
    let c = pollster::block_on(cpu.detect(&img, w, h, opts.clone())).unwrap();
    assert!(!c.response.boxes.is_empty());
    // f16 activations: scores agree to ~1e-2, boxes to a fraction of a pixel. Near the threshold a box can
    // appear on one side only, so match the CPU boxes clearly above it.
    for want in c.response.boxes.iter().filter(|b| b.score > opts.conf + 0.02) {
        let close = g.response.boxes.iter().any(|b| {
            b.class == want.class
                && (b.score - want.score).abs() < 2e-2
                && [b.x1 - want.x1, b.y1 - want.y1, b.x2 - want.x2, b.y2 - want.y2]
                    .iter()
                    .all(|d| d.abs() < 1.0)
        });
        assert!(close, "no f16 GPU box matches CPU box {want:?}");
    }
}

#[test]
fn rejects_bad_images_and_invalid_parameters() {
    assert!(rgba_to_image(&[0; 16], 2, 2).is_ok());
    assert!(rgba_to_image(&[0; 15], 2, 2).unwrap_err().contains("expected 16 RGBA bytes"));
    assert!(rgba_to_image(&[], 0, 2).unwrap_err().contains("empty"));
    assert!(rgba_to_image(&[], u32::MAX, u32::MAX).unwrap_err().contains("too large"));
    let rgb = rgba_to_image(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1).unwrap();
    assert_eq!(rgb, Image { width: 2, height: 1, data: vec![1, 2, 3, 5, 6, 7] });

    // Invalid load inputs.
    assert!(load(b"not a folo file".to_vec(), BackendChoice::Cpu).is_err());
    assert!(load_with_imgsz(model_bytes(), BackendChoice::Cpu, Some(Precision::F32), 0).is_err());
    assert!(load_with_imgsz(model_bytes(), BackendChoice::Cpu, Some(Precision::F32), 50).is_err());
    assert!(load_with_imgsz(model_bytes(), BackendChoice::Cpu, Some(Precision::F32), 8224).is_err());

    let det = load_with_imgsz(model_bytes(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
    let img = [0u8; 4 * 4 * 4];

    // Single-pass detect parameter validation (NaN, Inf, negative, out-of-range, zero max_det).
    for bad in [
        DetectOptions { conf: 1.5, ..Default::default() },
        DetectOptions { conf: -0.1, ..Default::default() },
        DetectOptions { conf: f32::NAN, ..Default::default() },
        DetectOptions { conf: f32::INFINITY, ..Default::default() },
        DetectOptions { iou: f32::NAN, ..Default::default() },
        DetectOptions { iou: 1.1, ..Default::default() },
        DetectOptions { max_det: 0, ..Default::default() },
    ] {
        assert!(
            pollster::block_on(det.detect(&img, 4, 4, bad.clone())).is_err(),
            "expected error for {bad:?}"
        );
    }

    // SAHI tiled parameter validation.
    for bad in [
        TileDetectOptions { tile_size: Some(0), ..Default::default() },
        TileDetectOptions { tile_size: Some(0), tile_width: Some(64), ..Default::default() },
        TileDetectOptions { tile_width: Some(0), ..Default::default() },
        TileDetectOptions { tile_height: Some(0), ..Default::default() },
        TileDetectOptions { overlap: Some(-0.1), ..Default::default() },
        TileDetectOptions { overlap: Some(1.0), ..Default::default() },
        TileDetectOptions { overlap: Some(f32::NAN), ..Default::default() },
        TileDetectOptions { overlap: Some(f32::INFINITY), ..Default::default() },
        TileDetectOptions { overlap_x: Some(1.0), ..Default::default() },
        TileDetectOptions { overlap_y: Some(f32::NEG_INFINITY), ..Default::default() },
        TileDetectOptions { tile_batch: 0, ..Default::default() },
        TileDetectOptions { merge_threshold: -0.01, ..Default::default() },
        TileDetectOptions { merge_threshold: 1.01, ..Default::default() },
        TileDetectOptions { merge_threshold: f32::NAN, ..Default::default() },
        TileDetectOptions { conf: f32::NAN, ..Default::default() },
        TileDetectOptions { iou: f32::INFINITY, ..Default::default() },
        TileDetectOptions { max_det: 0, ..Default::default() },
    ] {
        assert!(
            pollster::block_on(det.detect_tiled(&img, 4, 4, bad.clone())).is_err(),
            "expected tiled error for {bad:?}"
        );
    }
    assert!(pollster::block_on(det.detect_tiled(&[], 0, 4, TileDetectOptions::default())).is_err());
    // Exceeding MAX_BROWSER_TILES (100x100 with tile_size = 1 -> 10,000 tiles > 4,096) rejects cleanly.
    let img_100 = vec![0u8; 100 * 100 * 4];
    let too_many_tiles = TileDetectOptions { tile_size: Some(1), overlap: Some(0.0), ..Default::default() };
    let err = pollster::block_on(det.detect_tiled(&img_100, 100, 100, too_many_tiles)).unwrap_err();
    assert!(err.contains("browser limit"), "{err}");

    // TTA parameter validation.
    for bad in [
        TtaDetectOptions { scales: vec![], ..Default::default() },
        TtaDetectOptions { scales: vec![0.0], ..Default::default() },
        TtaDetectOptions { scales: vec![-0.5], ..Default::default() },
        TtaDetectOptions { scales: vec![f32::NAN], ..Default::default() },
        TtaDetectOptions { scales: vec![f32::INFINITY], ..Default::default() },
        TtaDetectOptions { scales: vec![0.01], ..Default::default() }, // resolves < 32
        TtaDetectOptions { scales: vec![100.0], ..Default::default() }, // resolves > 8192
        TtaDetectOptions { merge_iou: Some(-0.1), ..Default::default() },
        TtaDetectOptions { merge_iou: Some(1.1), ..Default::default() },
        TtaDetectOptions { merge_iou: Some(f32::NAN), ..Default::default() },
        TtaDetectOptions { conf: f32::NAN, ..Default::default() },
        TtaDetectOptions { iou: 2.0, ..Default::default() },
        TtaDetectOptions { max_det: 0, ..Default::default() },
    ] {
        assert!(
            pollster::block_on(det.detect_tta(&img, 4, 4, bad.clone())).is_err(),
            "expected TTA error for {bad:?}"
        );
    }
    assert!(pollster::block_on(det.detect_tta(&[0; 12], 4, 4, TtaDetectOptions::default())).is_err());
}

#[test]
fn options_parse_from_js_json() {
    let d: DetectOptions = serde_json::from_str(r#"{"conf":0.5,"maxDet":10,"unknown":1}"#).unwrap();
    assert_eq!((d.conf, d.iou, d.max_det), (0.5, 0.7, 10));
    let l: LoadOptions = serde_json::from_str(r#"{"backend":"webgpu","modelId":"m_1","imgsz":320}"#).unwrap();
    assert_eq!(l.backend, BackendChoice::Webgpu);
    assert_eq!((l.model_id.as_deref(), l.imgsz), (Some("m_1"), Some(320)));
    let l: LoadOptions = serde_json::from_str("{}").unwrap();
    assert_eq!((l.backend, l.precision), (BackendChoice::Auto, None));
    let l: LoadOptions = serde_json::from_str(r#"{"precision":"f16"}"#).unwrap();
    assert_eq!(l.precision, Some(Precision::F16));
    let l: LoadOptions = serde_json::from_str(r#"{"precision":"int8"}"#).unwrap();
    assert_eq!(l.precision, Some(Precision::Int8));
    let l: LoadOptions = serde_json::from_str(r#"{"precision":"f32"}"#).unwrap();
    assert_eq!(l.precision, Some(Precision::F32));
    assert!(serde_json::from_str::<LoadOptions>(r#"{"precision":"f8"}"#).is_err());
    assert!(serde_json::from_str::<LoadOptions>(r#"{"backend":"cuda"}"#).is_err());

    // Tiled options JSON (camelCase + aliases `mergeIou` / `fullImagePass`).
    let t: TileDetectOptions = serde_json::from_str(
        r#"{"tileSize":320,"overlap":0.25,"mergeIou":0.45,"fullImagePass":false,"mergeMetric":"iou","mergeFusion":"sebf","classAgnostic":true,"tileBatch":4,"conf":0.4,"maxDet":50}"#,
    )
    .unwrap();
    let tc = t.to_tile_config().unwrap();
    assert_eq!((tc.tile_width, tc.tile_height), (320, 320));
    assert_eq!((tc.overlap_x, tc.overlap_y), (0.25, 0.25));
    assert!(!tc.full_image);
    assert_eq!(tc.merge_metric, TileMergeMetric::Iou);
    assert_eq!(tc.merge_threshold, 0.45);
    assert_eq!(tc.merge_fusion, BoxFusion::Sebf);
    assert!(tc.class_agnostic);
    assert_eq!((tc.tile_batch, tc.max_det), (4, 50));

    // TTA options JSON.
    let a: TtaDetectOptions = serde_json::from_str(
        r#"{"hflip":false,"scales":[0.75,1.0],"mergeIou":0.6,"mergeFusion":"none","classAgnostic":true,"conf":0.35,"maxDet":20}"#,
    )
    .unwrap();
    let ac = a.to_tta_config().unwrap();
    assert!(!ac.hflip);
    assert_eq!(ac.scales, vec![0.75, 1.0]);
    assert_eq!(ac.merge_iou, Some(0.6));
    assert_eq!(ac.merge_fusion, BoxFusion::None);
    assert!(ac.class_agnostic);
    assert_eq!(ac.max_det, Some(20));

    // Invalid enum values reject cleanly.
    assert!(serde_json::from_str::<TileDetectOptions>(r#"{"mergeMetric":"giou"}"#).is_err());
    assert!(serde_json::from_str::<TtaDetectOptions>(r#"{"mergeFusion":"invalid"}"#).is_err());
}

#[test]
fn concurrent_detect_and_tta_futures_do_not_conflict() {
    use std::future::Future;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    let bytes = model_bytes();
    let backend = if has_gpu() { BackendChoice::Webgpu } else { BackendChoice::Cpu };
    let det = load_with_imgsz(bytes, backend, Some(Precision::F32), 128).unwrap();
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let opts = default_opts();
    let tta_opts = TtaDetectOptions {
        conf: opts.conf,
        iou: opts.iou,
        max_det: opts.max_det,
        hflip: true,
        scales: vec![0.75, 1.25],
        ..Default::default()
    };

    let mut f_tta = pin!(det.detect_tta(&img, w, h, tta_opts));
    let mut f_single = pin!(det.detect(&img, w, h, opts.clone()));
    let mut cx = Context::from_waker(Waker::noop());
    let mut r_tta = None;
    let mut r_single = None;
    for _ in 0..10_000 {
        if r_tta.is_none()
            && let Poll::Ready(res) = f_tta.as_mut().poll(&mut cx)
        {
            r_tta = Some(res.unwrap());
        }
        if r_single.is_none()
            && let Poll::Ready(res) = f_single.as_mut().poll(&mut cx)
        {
            r_single = Some(res.unwrap());
        }
        if r_tta.is_some() && r_single.is_some() {
            break;
        }
        std::thread::yield_now();
    }
    let single = r_single.expect("single detect future completed");
    let tta = r_tta.expect("TTA future completed");
    let baseline = pollster::block_on(det.detect(&img, w, h, opts)).unwrap();
    assert_eq!(single.response.boxes, baseline.response.boxes);
    assert!(!tta.response.boxes.is_empty());
    assert_eq!(det.info().imgsz, 128);
}

fn make_folo2_model(head: HeadMode, iou_head: bool) -> Vec<u8> {
    let cfg = FoloV2Config::new("n", 3).unwrap().with_head(head).with_iou_head(iou_head);
    let mut m = FoloV2Model::with_rng(cfg, &mut Rng::seed(17)).unwrap();
    m.set_classes(vec!["circle".into(), "square".into(), "triangle".into()]).unwrap();
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            if (t.name.starts_with("head.cls.") || t.name.starts_with("head.o2o."))
                && t.name.ends_with(".out.bias")
                || t.name.starts_with("head.o2o.") && t.name.ends_with(".cls.bias")
            {
                t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
            }
        }
    }
    let dir = TestDir::new("wasm-v2-test");
    let path = dir.join("m2.folo");
    m.save(&path).unwrap();
    std::fs::read(&path).unwrap()
}

fn folo2_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| make_folo2_model(HeadMode::O2m, false)).clone()
}

fn folo2_iou_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| make_folo2_model(HeadMode::O2m, true)).clone()
}

fn folo2_o2o_bytes() -> Vec<u8> {
    static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    BYTES.get_or_init(|| make_folo2_model(HeadMode::O2o, false)).clone()
}

fn cpu_vision_predictor(bytes: Vec<u8>, imgsz: u32, conf: f32, iou: f32, max_det: usize) -> Predictor {
    let file = ModelFile::from_bytes(bytes).unwrap();
    let (mut m, _) = VisionDetector::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let cfg = PredictConfig { imgsz, conf, iou, max_det, rect: true, head: None, ..PredictConfig::default() };
    Predictor::from_detector(m, cfg).unwrap()
}

#[test]
fn folo2_cpu_matches_predictor_across_single_tiled_tta_and_variants() {
    let (w, h) = (180, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();

    let opts = default_opts();
    for (label, bytes, want_head, want_o2o, want_iou) in [
        ("o2m", folo2_bytes(), "o2m", false, false),
        ("o2m+iou_head", folo2_iou_bytes(), "o2m", false, true),
        ("o2o", folo2_o2o_bytes(), "o2o", true, false),
    ] {
        let det = load_with_imgsz(bytes.clone(), BackendChoice::Cpu, Some(Precision::F32), 128).unwrap();
        let p = cpu_vision_predictor(bytes, 128, opts.conf, opts.iou, opts.max_det);

        let info = det.info();
        assert_eq!(
            (
                info.family.as_str(),
                info.arch.as_str(),
                info.head_mode.as_str(),
                info.has_o2o,
                info.has_iou_head,
                info.imgsz,
                info.classes.len(),
            ),
            ("folo2", "folo2-n", want_head, want_o2o, want_iou, 128, 3),
            "{label}: DetectorInfo mismatch"
        );

        // Single-pass parity.
        let got = pollster::block_on(det.detect(&img, w, h, opts.clone())).unwrap();
        let want = p.predict(&rgb).unwrap();
        assert!(!want.detections.is_empty(), "{label}: test needs detections");
        assert_eq!(got.response.boxes, want.scored_boxes(p.classes()), "{label}: single detect mismatch");

        // SAHI tiled parity.
        let tile_opts = TileDetectOptions {
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            tile_size: Some(112),
            overlap: Some(0.25),
            full_image: true,
            tile_batch: 4,
            ..TileDetectOptions::default()
        };
        let got_t = pollster::block_on(det.detect_tiled(&img, w, h, tile_opts.clone())).unwrap();
        let want_t = p.predict_tiled(&rgb, &tile_opts.to_tile_config().unwrap()).unwrap();
        assert_eq!(got_t.response.boxes, want_t.scored_boxes(p.classes()), "{label}: tiled mismatch");

        // Multi-scale + hflip TTA parity.
        let tta_opts = TtaDetectOptions {
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            hflip: true,
            scales: vec![0.75, 1.0],
            ..TtaDetectOptions::default()
        };
        let got_a = pollster::block_on(det.detect_tta(&img, w, h, tta_opts.clone())).unwrap();
        let want_a = p.predict_tta(&rgb, &tta_opts.to_tta_config().unwrap()).unwrap();
        assert_eq!(got_a.response.boxes, want_a.scored_boxes(p.classes()), "{label}: TTA mismatch");
        assert_eq!(det.info().imgsz, 128);
    }
}

#[test]
fn folo2_gpu_matches_predictor_across_single_tiled_tta_and_iou_head() {
    if !has_gpu() {
        return;
    }
    let gpu_dev = folo_gpu::init().unwrap();
    let (w, h) = (160, 128);
    let img = rgba(w, h);
    let rgb = rgba_to_image(&img, w, h).unwrap();
    let opts = default_opts();

    for (label, bytes, want_iou) in [
        ("o2m", folo2_bytes(), false),
        ("o2m+iou_head", folo2_iou_bytes(), true),
        ("o2o", folo2_o2o_bytes(), false),
    ] {
        let gpu = load_with_imgsz(bytes.clone(), BackendChoice::Webgpu, Some(Precision::F32), 128).unwrap();
        assert_eq!(gpu.backend(), Backend::Webgpu);
        assert_eq!(gpu.info().family, "folo2");
        assert_eq!(gpu.info().arch, "folo2-n");
        assert_eq!(gpu.info().has_iou_head, want_iou);

        let file = ModelFile::from_bytes(bytes).unwrap();
        let (mut gm, _) = VisionDetector::from_file(&file, Purpose::Inference).unwrap();
        gm.fuse().unwrap();
        gm.to_device(gpu_dev).unwrap();
        let gpu_pred = Predictor::from_detector(
            gm,
            PredictConfig {
                imgsz: 128,
                conf: opts.conf,
                iou: opts.iou,
                max_det: opts.max_det,
                ..Default::default()
            },
        )
        .unwrap();

        // Single-pass bit-for-bit parity against native GPU Predictor.
        let g = pollster::block_on(gpu.detect(&img, w, h, opts.clone())).unwrap();
        let want = gpu_pred.predict(&rgb).unwrap();
        assert!(!want.detections.is_empty(), "{label}: GPU single needs detections");
        assert_eq!(g.response.boxes, want.scored_boxes(gpu_pred.classes()), "{label}: GPU single mismatch");

        // SAHI tiled bit-for-bit parity against native GPU Predictor.
        let tile_opts = TileDetectOptions {
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            tile_size: Some(112),
            overlap: Some(0.25),
            full_image: true,
            tile_batch: 4,
            ..TileDetectOptions::default()
        };
        let gt = pollster::block_on(gpu.detect_tiled(&img, w, h, tile_opts.clone())).unwrap();
        let want_t = gpu_pred.predict_tiled(&rgb, &tile_opts.to_tile_config().unwrap()).unwrap();
        assert_eq!(gt.response.boxes, want_t.scored_boxes(gpu_pred.classes()), "{label}: GPU tiled mismatch");

        // TTA bit-for-bit parity against native GPU Predictor.
        let tta_opts = TtaDetectOptions {
            conf: opts.conf,
            iou: opts.iou,
            max_det: opts.max_det,
            hflip: true,
            scales: vec![0.75, 1.0],
            ..TtaDetectOptions::default()
        };
        let ga = pollster::block_on(gpu.detect_tta(&img, w, h, tta_opts.clone())).unwrap();
        let want_a = gpu_pred.predict_tta(&rgb, &tta_opts.to_tta_config().unwrap()).unwrap();
        assert_eq!(ga.response.boxes, want_a.scored_boxes(gpu_pred.classes()), "{label}: GPU TTA mismatch");
    }
}
