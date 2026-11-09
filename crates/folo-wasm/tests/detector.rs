//! folo-wasm's detector, run natively: the CPU path against folo-vision's `Predictor` (identical results), the
//! GPU path (`download_async` readback) against the CPU path, the result's JSON shape and the options parsing.
//! The GPU test skips without a hardware adapter, like folo-vision's.

use std::sync::atomic::{AtomicUsize, Ordering};

use folo_core::no_grad;
use folo_core::rng::Rng;
use folo_data::Image;
use folo_nn::Module;
use folo_nn::io::{ModelFile, Purpose};
use folo_vision::{FoloModel, ModelConfig, Precision, PredictConfig, Predictor};
use folo_wasm::detector::rgba_to_image;
use folo_wasm::{Backend, BackendChoice, DetectOptions, Detector, LoadOptions};

/// A random folo-n with its class-output biases zeroed: the trained priors (MODEL.md §6.5) put every score near
/// 1e-4, and the tests need detections.
fn model_bytes() -> Vec<u8> {
    static N: AtomicUsize = AtomicUsize::new(0);
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
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("folo-wasm-test-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.folo");
    m.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
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

const OPTS: DetectOptions = DetectOptions { conf: 0.3, iou: 0.7, max_det: 40 };

fn load(bytes: Vec<u8>, backend: BackendChoice) -> Result<Detector, String> {
    load_at(bytes, backend, Precision::F32)
}

fn load_at(bytes: Vec<u8>, backend: BackendChoice, precision: Precision) -> Result<Detector, String> {
    pollster::block_on(Detector::load(
        bytes,
        LoadOptions { backend, model_id: Some("m_test".into()), imgsz: Some(256), precision },
    ))
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

#[test]
fn cpu_matches_predictor() {
    let bytes = model_bytes();
    let det = load(bytes.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(det.backend(), Backend::Cpu);
    let (w, h) = (200, 120);
    let img = rgba(w, h);
    let res = pollster::block_on(det.detect(&img, w, h, OPTS)).unwrap();

    let file = ModelFile::from_bytes(bytes).unwrap();
    let (mut m, _) = FoloModel::from_file(&file, Purpose::Inference).unwrap();
    m.fuse().unwrap();
    let cfg = PredictConfig {
        imgsz: 256,
        conf: OPTS.conf,
        iou: OPTS.iou,
        max_det: OPTS.max_det,
        rect: true,
        head: None,
    };
    let p = Predictor::new(m, cfg).unwrap();
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
    assert_eq!((info.arch.as_str(), info.imgsz, info.classes.len()), ("folo-n", 256, 3));
    assert!(info.fallback.is_none() && info.adapter.is_none());
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
    let g = pollster::block_on(gpu.detect(&img, w, h, OPTS)).unwrap();
    let c = pollster::block_on(cpu.detect(&img, w, h, OPTS)).unwrap();
    assert_eq!(g.response.device, "gpu:0");
    assert_eq!(g.backend, Backend::Webgpu);
    assert!(!c.response.boxes.is_empty());
    // f32 sums in another order: scores agree to ~1e-4, boxes to a few thousandths of a pixel. Boxes whose
    // scores are that close can swap places, so match each CPU box among the GPU boxes.
    assert_eq!(g.response.boxes.len(), c.response.boxes.len());
    for want in &c.response.boxes {
        let close = g.response.boxes.iter().any(|b| {
            b.class == want.class
                && (b.score - want.score).abs() < 1e-3
                && [b.x1 - want.x1, b.y1 - want.y1, b.x2 - want.x2, b.y2 - want.y2]
                    .iter()
                    .all(|d| d.abs() < 0.05)
        });
        assert!(close, "no GPU box matches CPU box {want:?}");
    }
}

/// `precision: "f16"` on WebGPU runs the model in f16 (with shader-f16) and still matches the f32 CPU boxes, to
/// f16's rounding; on the CPU it runs in f32 and says why.
#[test]
fn f16_precision_option() {
    let bytes = model_bytes();
    let cpu = load_at(bytes.clone(), BackendChoice::Cpu, Precision::F16).unwrap();
    let info = cpu.info();
    assert_eq!(info.precision, Precision::F32);
    assert!(info.precision_fallback.as_deref().is_some_and(|w| w.contains("GPU")), "{info:?}");
    if !has_gpu() {
        return;
    }
    let gpu = load_at(bytes, BackendChoice::Webgpu, Precision::F16).unwrap();
    let info = gpu.info();
    if let Some(why) = &info.precision_fallback {
        return eprintln!("note: {why}");
    }
    assert_eq!(info.precision, Precision::F16);
    let (w, h) = (160, 224);
    let img = rgba(w, h);
    let g = pollster::block_on(gpu.detect(&img, w, h, OPTS)).unwrap();
    let c = pollster::block_on(cpu.detect(&img, w, h, OPTS)).unwrap();
    assert!(!c.response.boxes.is_empty());
    // f16 activations: scores agree to ~1e-2, boxes to a fraction of a pixel. Near the threshold a box can
    // appear on one side only, so match the CPU boxes clearly above it.
    for want in c.response.boxes.iter().filter(|b| b.score > OPTS.conf + 0.02) {
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
fn rejects_bad_images() {
    assert!(rgba_to_image(&[0; 16], 2, 2).is_ok());
    assert!(rgba_to_image(&[0; 15], 2, 2).unwrap_err().contains("expected 16 RGBA bytes"));
    assert!(rgba_to_image(&[], 0, 2).unwrap_err().contains("empty"));
    assert!(rgba_to_image(&[], u32::MAX, u32::MAX).unwrap_err().contains("too large"));
    let rgb = rgba_to_image(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1).unwrap();
    assert_eq!(rgb, Image { width: 2, height: 1, data: vec![1, 2, 3, 5, 6, 7] });

    let det = load(model_bytes(), BackendChoice::Cpu).unwrap();
    let bad = DetectOptions { conf: 1.5, ..DetectOptions::default() };
    assert!(pollster::block_on(det.detect(&[0; 64], 4, 4, bad)).unwrap_err().contains("conf"));
    assert!(load(b"not a folo file".to_vec(), BackendChoice::Cpu).is_err());
}

#[test]
fn options_parse_from_js_json() {
    let d: DetectOptions = serde_json::from_str(r#"{"conf":0.5,"maxDet":10,"unknown":1}"#).unwrap();
    assert_eq!((d.conf, d.iou, d.max_det), (0.5, 0.7, 10));
    let l: LoadOptions = serde_json::from_str(r#"{"backend":"webgpu","modelId":"m_1","imgsz":320}"#).unwrap();
    assert_eq!(l.backend, BackendChoice::Webgpu);
    assert_eq!((l.model_id.as_deref(), l.imgsz), (Some("m_1"), Some(320)));
    let l: LoadOptions = serde_json::from_str("{}").unwrap();
    assert_eq!((l.backend, l.precision), (BackendChoice::Auto, Precision::F32));
    let l: LoadOptions = serde_json::from_str(r#"{"precision":"f16"}"#).unwrap();
    assert_eq!(l.precision, Precision::F16);
    assert!(serde_json::from_str::<LoadOptions>(r#"{"precision":"f8"}"#).is_err());
    assert!(serde_json::from_str::<LoadOptions>(r#"{"backend":"cuda"}"#).is_err());
}
