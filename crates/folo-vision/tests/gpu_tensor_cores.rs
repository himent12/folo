//! f16 inference on tensor cores (folo-gpu's cooperative-matrix GEMM and conv) against f32 and against the regular
//! f16 kernels: the decoded head of a small detector and the end-to-end predictor.
//!
//! This test binary asks for tensor cores before the shared GPU context exists (`request_tensor_cores`), and skips
//! (with a note) when the adapter has none. Tolerances are the f16 ones of `gpu_f16.rs` (scores `2e-2` absolute,
//! boxes `1.5 px + 1%`): tensor cores change only the f32 summation order inside each GEMM, far below the f16
//! rounding of every activation that those bounds already cover.

use std::sync::{Mutex, MutexGuard};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::ModelFile;
use folo_test_support::TestFile;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::NmsConfig;
use folo_vision::{Precision, PredictConfig, Predictor};

/// The GPU device and its context with tensor cores on (and a lock, since the tests flip the switch), or `None`.
fn gpu_tc() -> Option<(Device, folo_gpu::GpuContext, MutexGuard<'static, ()>)> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    folo_gpu::request_tensor_cores();
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: skipping GPU test, no adapter: {e}");
            return None;
        }
    };
    if !ctx.caps().coop_f16 {
        // A qualifying adapter must end up with tensor cores; anything else is a regression, not a skip.
        let why = folo_gpu::kernels::coop::adapter_unavailable_reason(ctx.adapter());
        let no_f16 = std::env::var("FOLO_GPU_NO_F16").is_ok_and(|v| v != "0" && !v.is_empty());
        assert!(why.is_some() || no_f16, "the adapter qualifies but: {}", ctx.tensor_cores_status());
        eprintln!("note: skipping tensor-core test: {}", ctx.tensor_cores_status());
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    ctx.set_tensor_cores(true);
    Some((dev, ctx, guard))
}

macro_rules! gpu_tc {
    () => {
        match gpu_tc() {
            Some(d) => d,
            None => return,
        }
    };
}

/// As in `gpu_f16.rs`: non-trivial BatchNorm statistics and zero class biases, so scores spread around 0.5.
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

fn temp(name: &str) -> TestFile {
    TestFile::new(&format!("tc-{name}"), "m.folo")
}

/// The worst `(score error, box error beyond 1%)` between the full decodes of `x` by `a` and by `b` (each in its
/// own dtype), with tensor cores set to `tc_a` / `tc_b` for each.
fn decode_gap(
    ctx: &folo_gpu::GpuContext,
    a: (&FoloModel, bool),
    b: (&FoloModel, bool),
    x: &Tensor,
) -> (f32, f32) {
    let nms = NmsConfig { conf: 0.0, ..NmsConfig::default() };
    let run = |(m, tc): (&FoloModel, bool)| {
        ctx.set_tensor_cores(tc);
        m.predict_raw(&x.to_dtype(m.dtype()).unwrap(), &nms).unwrap()
    };
    let (want, got) = (run(a), run(b));
    ctx.set_tensor_cores(true);
    let (mut ds, mut db) = (0f32, 0f32);
    for (w, g) in want.images.iter().zip(&got.images) {
        assert_eq!((w.scores.len(), w.boxes.len()), (g.scores.len(), g.boxes.len()));
        for (p, q) in w.scores.iter().zip(&g.scores) {
            assert!(q.is_finite(), "non-finite score");
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
fn tensor_core_decode_matches_f32_and_regular_f16() {
    let (dev, ctx, _g) = gpu_tc!();
    let path = temp("decode");
    model(4).save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let m32 = FoloModel::from_file_for_inference(&file, dev, DType::F32).unwrap();
    let m16 = FoloModel::from_file_for_inference(&file, dev, DType::F16).unwrap();
    // 160 × 224 makes the early convs big enough for the large tensor-core tiles too.
    let x = Tensor::rand_uniform([2, 3, 160, 224], 0.0, 1.0, &mut Rng::seed(5)).to_device(dev).unwrap();
    let x = x.to_dtype(DType::F16).unwrap().to_dtype(DType::F32).unwrap();
    let (ds, db) = decode_gap(&ctx, (&m32, false), (&m16, true), &x);
    eprintln!("f16 on tensor cores vs f32: worst score gap {ds:.2e}, worst box gap beyond 1% {db:.3} px");
    assert!(ds <= 2e-2, "scores differ by {ds}");
    assert!(db <= 1.5, "boxes differ by {db} px beyond 1%");
    let (ds, db) = decode_gap(&ctx, (&m16, false), (&m16, true), &x);
    eprintln!(
        "f16 on tensor cores vs regular f16: worst score gap {ds:.2e}, worst box gap beyond 1% {db:.3} px"
    );
    assert!(ds <= 2e-2, "scores differ by {ds}");
    assert!(db <= 1.5, "boxes differ by {db} px beyond 1%");
    assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
}

#[test]
fn tensor_core_predictor_matches_f32_end_to_end() {
    let (dev, ctx, _g) = gpu_tc!();
    let path = temp("predict");
    model(3).save(&path).unwrap();
    // As in gpu_f16.rs: compare the top few boxes, whose scores are well apart.
    let cfg = PredictConfig { imgsz: 96, conf: 0.3, max_det: 7, ..PredictConfig::default() };
    let p32 = Predictor::open_on(&path, cfg.clone(), dev, Precision::F32).unwrap();
    let p16 = Predictor::open_on(&path, cfg, dev, Precision::F16).unwrap();
    std::fs::remove_file(&path).ok();
    let data = (0..120u32 * 80 * 3).map(|i| ((i * 37 + i / 360 * 11) % 256) as u8).collect();
    let img = folo_data::Image::from_raw(120, 80, data).unwrap();
    ctx.set_tensor_cores(false);
    let a = p32.predict(&img).unwrap();
    ctx.set_tensor_cores(true);
    let b = p16.predict(&img).unwrap();
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
        assert!(hit, "no tensor-core detection matches {want:?}");
    }
    assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
}
