//! Half-precision inference (f16 weights and activations, f32 accumulation) against f32 on the GPU: the decoded
//! head of a small detector, the end-to-end predictor, loading an f16 export straight as f16, and the fallback to
//! f32 where f16 compute is not available.
//!
//! Tolerances (box / score parity): the decoded scores (post-sigmoid) agree to `2e-2` absolute, the boxes to
//! `1.5 px + 1%` of the coordinate at 64–96 px inputs. A random folo-n is ~25 convs deep; f16 rounds every
//! activation once (2^-11 relative), which compounds to about 1e-2 on the logits. Trained models agree much more
//! closely in mAP (docs/BENCHMARKS.md: within 0.002).
//!
//! Skips (with a note) without a hardware adapter or without `shader-f16`.

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::{FloatDType, ModelFile};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::NmsConfig;
use folo_vision::predict::f16_unavailable;
use folo_vision::{Precision, PredictConfig, Predictor};

/// The GPU device when it computes in f16, else `None` with a note.
fn gpu16() -> Option<Device> {
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    if let Some(why) = f16_unavailable(dev) {
        eprintln!("note: skipping f16 test: {why}");
        return None;
    }
    Some(dev)
}

macro_rules! gpu16 {
    () => {
        match gpu16() {
            Some(d) => d,
            None => return,
        }
    };
}

/// A model whose BatchNorms have non-trivial running statistics and whose class biases are zero, so scores spread
/// around 0.5 instead of sitting at the 1e-4 prior.
fn model(nc: usize) -> FoloModel {
    let m = FoloModel::new(ModelConfig::new("n", nc).unwrap()).unwrap();
    let rng = &mut Rng::seed(3);
    let _g = no_grad();
    for t in m.named_tensors() {
        let n = t.tensor.numel();
        if t.name.ends_with("running_mean") {
            t.tensor.copy_from(&Tensor::rand_uniform([n], -0.2, 0.2, rng)).unwrap();
        } else if t.name.ends_with("running_var") {
            t.tensor.copy_from(&Tensor::rand_uniform([n], 0.5, 1.5, rng)).unwrap();
        } else if t.name.starts_with("head.cls.") && t.name.ends_with(".out.bias") {
            t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
        }
    }
    m
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("folo-f16-{}-{name}.folo", std::process::id()))
}

/// The worst `(score error, box error beyond 1%)` between two full decodes of `x` (in each model's dtype).
fn decode_gap(a: &FoloModel, b: &FoloModel, x: &Tensor) -> (f32, f32) {
    let nms = NmsConfig { conf: 0.0, ..NmsConfig::default() };
    let run = |m: &FoloModel| m.predict_raw(&x.to_dtype(m.dtype()).unwrap(), &nms).unwrap();
    let (want, got) = (run(a), run(b));
    let (mut ds, mut db) = (0f32, 0f32);
    for (w, g) in want.images.iter().zip(&got.images) {
        assert_eq!((w.scores.len(), w.boxes.len()), (g.scores.len(), g.boxes.len()));
        for (p, q) in w.scores.iter().zip(&g.scores) {
            assert!(q.is_finite(), "non-finite f16 score");
            ds = ds.max((p - q).abs());
        }
        for (p, q) in w.boxes.iter().zip(&g.boxes) {
            for (u, v) in [(p.x1, q.x1), (p.y1, q.y1), (p.x2, q.x2), (p.y2, q.y2)] {
                db = db.max((u - v).abs() - 0.01 * u.abs());
            }
        }
    }
    (ds, db)
}

#[test]
fn f16_decode_matches_f32() {
    let dev = gpu16!();
    let path = temp("decode");
    model(4).save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let m32 = FoloModel::from_file_for_inference(&file, dev, DType::F32).unwrap();
    let m16 = FoloModel::from_file_for_inference(&file, dev, DType::F16).unwrap();
    assert_eq!((m32.dtype(), m16.dtype()), (DType::F32, DType::F16));
    assert!(m16.is_fused() && !m16.is_training());
    let x = Tensor::rand_uniform([2, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(5)).to_device(dev).unwrap();
    let out = {
        let _g = no_grad();
        m16.forward(&x.to_dtype(DType::F16).unwrap()).unwrap()
    };
    assert_eq!((out.box_raw.dtype(), out.cls_raw.dtype()), (DType::F16, DType::F16));
    let (ds, db) = decode_gap(&m32, &m16, &x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap());
    eprintln!("f16 vs f32: worst score gap {ds:.2e}, worst box gap beyond 1% {db:.3} px");
    assert!(ds <= 2e-2, "scores differ by {ds}");
    assert!(db <= 1.5, "boxes differ by {db} px beyond 1%");
}

#[test]
fn f16_predictor_matches_f32_end_to_end() {
    let dev = gpu16!();
    let path = temp("predict");
    model(3).save(&path).unwrap();
    // A random model scores hundreds of overlapping boxes near 0.50; NMS over near-ties is chaotic in any
    // precision, so compare the top few, whose scores are well apart.
    let cfg = PredictConfig { imgsz: 96, conf: 0.3, max_det: 7, ..PredictConfig::default() };
    let p32 = Predictor::open_on(&path, cfg.clone(), dev, Precision::F32).unwrap();
    let p16 = Predictor::open_on(&path, cfg, dev, Precision::F16).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!((p32.precision(), p16.precision()), (Precision::F32, Precision::F16));
    assert!(p16.fallback().is_none());
    let data = (0..120u32 * 80 * 3).map(|i| ((i * 37 + i / 360 * 11) % 256) as u8).collect();
    let img = folo_data::Image::from_raw(120, 80, data).unwrap();
    let (a, b) = (p32.predict(&img).unwrap(), p16.predict(&img).unwrap());
    assert!(!a.detections.is_empty(), "the test model should detect something");
    assert_eq!(a.detections.len(), b.detections.len());
    for want in &a.detections {
        let hit = b.detections.iter().any(|d| {
            let gaps = [
                d.bbox.x1 - want.bbox.x1,
                d.bbox.y1 - want.bbox.y1,
                d.bbox.x2 - want.bbox.x2,
                d.bbox.y2 - want.bbox.y2,
            ];
            d.class == want.class && (d.score - want.score).abs() < 2e-2 && gaps.iter().all(|e| e.abs() < 2.0)
        });
        assert!(hit, "no f16 detection matches {want:?}");
    }
    let batch = p16.predict_batch(&[img.clone(), img]).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].detections, batch[1].detections);
}

/// An f16 export is loaded as f16 without an f32 copy and holds exactly the weights `--half` makes from the f32
/// file (both fuse in f32 and round once), so both predict identically.
#[test]
fn f16_export_loads_straight_as_f16() {
    let dev = gpu16!();
    let (src, exp) = (temp("src"), temp("export"));
    let m = model(2);
    m.save(&src).unwrap();
    m.export(&exp, FloatDType::F16).unwrap();
    let from_export =
        FoloModel::from_file_for_inference(&ModelFile::open(&exp).unwrap(), dev, DType::F16).unwrap();
    let from_f32 =
        FoloModel::from_file_for_inference(&ModelFile::open(&src).unwrap(), dev, DType::F16).unwrap();
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&exp).ok();
    for ((n, a), (_, b)) in from_export.state().iter().zip(from_f32.state()) {
        assert_eq!((a.dtype(), a.device()), (DType::F16, dev), "{n}");
        let (a, b) = (a.to_vec::<folo_core::f16>().unwrap(), b.to_vec::<folo_core::f16>().unwrap());
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{n} differs");
    }
    let x = Tensor::rand_uniform([1, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(9))
        .to_device(dev)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let nms = NmsConfig::default();
    assert_eq!(
        from_export.predict_detections(&x, &nms).unwrap(),
        from_f32.predict_detections(&x, &nms).unwrap()
    );
}

#[test]
fn f16_falls_back_to_f32_on_the_cpu() {
    let p = Predictor::new_f16(model(2), PredictConfig::default()).unwrap();
    assert_eq!(p.precision(), Precision::F32);
    assert!(p.fallback().is_some_and(|w| w.contains("GPU")), "{:?}", p.fallback());
    assert!(Predictor::new(model(2), PredictConfig::default()).unwrap().fallback().is_none());
    assert_eq!("fp16".parse::<Precision>().unwrap(), Precision::F16);
    assert_eq!("F32".parse::<Precision>().unwrap(), Precision::F32);
    assert!("int8".parse::<Precision>().is_err());
    assert_eq!(serde_json::to_string(&Precision::F16).unwrap(), "\"f16\"");
}

#[test]
fn f16_predictor_converts_an_f32_model() {
    let dev = gpu16!();
    let mut m = model(2);
    m.to_device(dev).unwrap();
    let p = Predictor::new_f16(m, PredictConfig { imgsz: 64, ..PredictConfig::default() }).unwrap();
    assert_eq!(p.precision(), Precision::F16);
    assert!(p.model().is_fused());
    let img = folo_data::Image::from_raw(64, 64, vec![90; 64 * 64 * 3]).unwrap();
    p.predict(&img).unwrap();
}
