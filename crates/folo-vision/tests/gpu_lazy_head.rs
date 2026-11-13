//! Candidate-first prediction on the GPU (docs/MODEL.md §9.4) against the dense GPU path, bit for bit: every
//! detection of `predict_candidates` + `finish_candidates` equals the dense forward's `detections` (with NMS at
//! IoU 1 and no cap, so every candidate is compared), in f32, in f16 on the regular kernels and on tensor cores,
//! for `o2m` and `o2o` heads, at thresholds from none passing to every anchor passing (more candidates than the
//! site lists hold, so the dense head serves those images).
//!
//! This binary asks for tensor cores before the shared GPU context exists and runs the f16 cases with them on
//! and off (on only where the adapter has them).

use std::sync::{Mutex, MutexGuard};

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::Module;
use folo_nn::io::ModelFile;
use folo_test_support::TestFile;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::{Detection, NmsConfig};

/// The GPU device and its context (tensor cores requested; a lock, since the tests flip the switch), or `None`.
fn gpu() -> Option<(Device, folo_gpu::GpuContext, MutexGuard<'static, ()>)> {
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
    if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
        eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        return None;
    }
    let dev = folo_gpu::init().ok()?;
    Some((dev, ctx, guard))
}

/// A folo-n detector with `nc` classes and `head` (`o2m` or `o2o`), with non-trivial BatchNorm statistics and zero
/// class biases (scores spread around 0.5), loaded fused for inference on `dev` in `dtype`.
fn model(nc: usize, head: &str, dev: Device, dtype: DType) -> FoloModel {
    let mut c = ModelConfig::new("n", nc).unwrap();
    c.head = head.into();
    let m = FoloModel::new(c).unwrap();
    let rng = &mut Rng::seed(3);
    {
        let _g = no_grad();
        for t in m.named_tensors() {
            let n = t.tensor.numel();
            if t.name.ends_with("running_mean") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], -0.2, 0.2, rng)).unwrap();
            } else if t.name.ends_with("running_var") {
                t.tensor.copy_from(&Tensor::rand_uniform([n], 0.5, 1.5, rng)).unwrap();
            } else if t.name.contains("cls.") && t.name.ends_with(".out.bias") {
                t.tensor.copy_from(&t.tensor.zeros_like().unwrap()).unwrap();
            }
        }
    }
    let path = TestFile::new(&format!("lazy-gpu-{head}-{nc}"), "m.folo");
    m.save(&path).unwrap();
    let file = ModelFile::open(&path).unwrap();
    FoloModel::from_file_for_inference(&file, dev, dtype).unwrap()
}

fn bits(d: &[Detection]) -> Vec<[u32; 6]> {
    d.iter()
        .map(|d| {
            let b = d.bbox;
            [d.class, d.score.to_bits(), b.x1.to_bits(), b.y1.to_bits(), b.x2.to_bits(), b.y2.to_bits()]
        })
        .collect()
}

/// Candidate-first against dense on `x`, at thresholds picked from the dense scores: none passing, the top 1%,
/// 5% and 30% of all anchors, and every anchor. Returns how many detections were compared.
fn check(m: &FoloModel, x: &Tensor, what: &str) -> usize {
    let _g = no_grad();
    let dense = m.forward(x).unwrap();
    let (b, nc, a) = (dense.batch(), dense.num_classes, dense.num_anchors());
    let cls = dense.cls_raw.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap();
    let mut best: Vec<f32> = (0..b * a)
        .map(|i| {
            let (img, j) = (i / a, i % a);
            (0..nc).map(|c| cls[(img * nc + c) * a + j]).fold(f32::NEG_INFINITY, f32::max)
        })
        .map(|l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    best.sort_by(f32::total_cmp);
    let at = |q: f64| best[((best.len() as f64 * q) as usize).min(best.len() - 1)];
    let mut compared = 0;
    for conf in [best[best.len() - 1] + 1e-3, at(0.99), at(0.95), at(0.7), best[0] - 1e-3] {
        let nms = NmsConfig {
            conf,
            iou: 1.0,
            max_det: usize::MAX,
            max_candidates: usize::MAX,
            ..NmsConfig::default()
        };
        assert!(m.candidates_first_applies(x.device(), &nms), "{what}: candidate-first applies");
        let want = dense.detections(&nms).unwrap();
        let got: Vec<Vec<Detection>> =
            m.predict_candidates(x, &nms).unwrap().iter().map(|c| m.finish_candidates(c, &nms)).collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(bits(g), bits(w), "{what}: image {i} at conf {conf}");
            compared += w.len();
        }
    }
    compared
}

fn input(dims: [usize; 4], seed: u64, dev: Device, dtype: DType) -> Tensor {
    let x = Tensor::rand_uniform(dims, 0.0, 1.0, &mut Rng::seed(seed)).to_device(dev).unwrap();
    x.to_dtype(dtype).unwrap()
}

#[test]
fn gpu_candidates_first_matches_dense_bit_for_bit() {
    let Some((dev, ctx, _g)) = gpu() else { return };
    let tc = ctx.caps().coop_f16;
    let mut dtypes = vec![(DType::F32, false)];
    if ctx.caps().shader_f16 {
        dtypes.push((DType::F16, false));
        if tc {
            dtypes.push((DType::F16, true));
        }
    }
    for head in ["o2m", "o2o"] {
        for &(dtype, on) in &dtypes {
            let m = model(3, head, dev, dtype);
            ctx.set_tensor_cores(on);
            // 320²: P3 has more pixels than its site list, so "every anchor" overflows; 384×640 is rectangular.
            for (dims, seed) in [([2, 3, 320, 320], 1), ([1, 3, 384, 640], 2), ([3, 3, 128, 96], 3)] {
                let what = format!("{head} {dtype:?} tensor cores {on} {dims:?}");
                let n = check(&m, &input(dims, seed, dev, dtype), &what);
                assert!(n > 0, "{what}: nothing compared");
            }
            assert_eq!(ctx.tensor_cores(), on, "{}", ctx.tensor_cores_status());
        }
    }
    ctx.set_tensor_cores(true);
}
