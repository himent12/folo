//! The FOLO detector on the GPU (folo-gpu's backend) against the CPU: eval forward (BatchNorm folded on the
//! fly and fused), a train-mode forward + loss + backward, and `.folo` loading straight onto the device.
//!
//! Skips (with a note) without a hardware adapter; set `FOLO_GPU_TEST_SOFTWARE=1` to run on WARP / llvmpipe.
//! Tolerances are per check: ~25 conv layers deep, f32 sums in a different order compound to ~1e-4 relative.

use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_nn::Module;
use folo_vision::boxes::Xyxy;
use folo_vision::loss::{ImageTargets, LossConfig, detection_loss};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::NmsConfig;

fn gpu() -> Option<Device> {
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
    folo_gpu::init().ok()
}

macro_rules! gpu {
    () => {
        match gpu() {
            Some(d) => d,
            None => return,
        }
    };
}

#[track_caller]
fn close(got: &Tensor, want: &Tensor, atol: f32, rtol: f32, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let (g, w) = (got.to_vec::<f32>().unwrap(), want.to_vec::<f32>().unwrap());
    let mut worst = (0, 0.0f32);
    for (i, (&a, &b)) in g.iter().zip(&w).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "{what}[{i}]: got {a}, want {b} (non-finite)");
        let e = (a - b).abs() - rtol * b.abs();
        if e > worst.1 {
            worst = (i, e);
        }
    }
    let i = worst.0;
    assert!(worst.1 <= atol, "{what}[{i}]: got {}, want {} (atol {atol}, rtol {rtol})", g[i], w[i]);
}

/// A model whose BatchNorms have non-trivial running statistics (fresh ones are 0 / 1).
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
        }
    }
    m
}

#[test]
fn eval_forward_matches_cpu() {
    let dev = gpu!();
    let x = Tensor::rand_uniform([2, 3, 96, 64], 0.0, 1.0, &mut Rng::seed(1));
    let mut m = model(4);
    m.eval();
    let _g = no_grad();
    let want = m.forward(&x).unwrap();
    let ids: Vec<_> = m.parameters().iter().map(Tensor::id).collect();
    m.to_device(dev).unwrap();
    assert_eq!(m.device(), dev);
    assert_eq!(m.parameters().iter().map(Tensor::id).collect::<Vec<_>>(), ids);
    let xg = x.to_device(dev).unwrap();
    let got = m.forward(&xg).unwrap();
    assert_eq!(got.box_raw.device(), dev);
    close(&got.box_raw, &want.box_raw, 2e-3, 2e-3, "box_raw (folded on the fly)");
    close(&got.cls_raw, &want.cls_raw, 2e-3, 2e-3, "cls_raw (folded on the fly)");
    m.fuse().unwrap();
    let got = m.forward(&xg).unwrap();
    close(&got.box_raw, &want.box_raw, 2e-3, 2e-3, "box_raw (fused)");
    close(&got.cls_raw, &want.cls_raw, 2e-3, 2e-3, "cls_raw (fused)");
}

/// One training step's forward (train-mode BatchNorm: fused kernels on the GPU, the fused CPU kernel on the
/// CPU), detection loss and backward: same loss, gradients and updated running statistics.
#[test]
fn train_step_matches_cpu() {
    let dev = gpu!();
    let x = Tensor::rand_uniform([2, 3, 64, 64], 0.0, 1.0, &mut Rng::seed(2));
    let targets = vec![
        ImageTargets {
            boxes: vec![Xyxy::new(4.0, 6.0, 30.0, 40.0), Xyxy::new(20.0, 20.0, 60.0, 50.0)],
            labels: vec![0, 2],
        },
        ImageTargets { boxes: vec![Xyxy::new(10.0, 12.0, 44.0, 58.0)], labels: vec![1] },
    ];
    let run = |m: &FoloModel, x: &Tensor| {
        let out = m.forward(x).unwrap();
        let loss = detection_loss(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &LossConfig::default())
            .unwrap();
        assert!(loss.is_finite());
        let grads = loss.total.backward().unwrap();
        let gs: Vec<Tensor> =
            m.parameters().iter().map(|p| grads.get(p).expect("every parameter").clone()).collect();
        (loss.total_value, gs)
    };
    let cpu = model(3);
    let mut gm = model(3);
    let (cl, cg) = run(&cpu, &x);
    gm.to_device(dev).unwrap();
    let (gl, gg) = run(&gm, &x.to_device(dev).unwrap());
    assert!((cl - gl).abs() <= 1e-3 * cl.abs(), "loss {gl} vs {cl}");
    let names: Vec<String> = cpu.named_parameters().into_iter().map(|(n, _)| n).collect();
    for ((c, g), n) in cg.iter().zip(&gg).zip(&names) {
        assert_eq!(g.device(), dev);
        // Gradients are as large as ~1e1 deep in the backbone; compare against the tensor's scale.
        let scale = c.abs().unwrap().max_all().unwrap().item::<f32>().unwrap().max(1e-3);
        close(g, c, 5e-3 * scale, 5e-3, &format!("grad {n}"));
    }
    for ((n, c), (_, g)) in cpu.named_buffers().iter().zip(gm.named_buffers()) {
        close(&g, c, 1e-4, 1e-4, n);
    }
}

#[test]
fn open_on_device_matches_cpu_predictions() {
    let dev = gpu!();
    let path = std::env::temp_dir().join(format!("folo-gpu-open-{}.folo", std::process::id()));
    model(2).save(&path).unwrap();
    let cpu = FoloModel::open(&path).unwrap();
    let gm = FoloModel::open_on(&path, dev).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(gm.device(), dev);
    assert!(!gm.is_training());
    let x = Tensor::rand_uniform([1, 3, 64, 96], 0.0, 1.0, &mut Rng::seed(5));
    let nms = NmsConfig { conf: 0.0, ..NmsConfig::default() };
    let want = cpu.predict_raw(&x, &nms).unwrap();
    let got = gm.predict_raw(&x.to_device(dev).unwrap(), &nms).unwrap();
    let (w, g) = (&want.images[0], &got.images[0]);
    // The GPU decodes on the device (softmax expectation, boxes, sigmoid) before downloading.
    assert_eq!(w.scores.len(), g.scores.len());
    assert!(w.scores.iter().chain(&g.scores).all(|v| v.is_finite()), "non-finite scores");
    let worst = w.scores.iter().zip(&g.scores).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst <= 1e-3, "scores differ by {worst}");
    assert_eq!(w.boxes.len(), g.boxes.len());
    for (i, (a, b)) in w.boxes.iter().zip(&g.boxes).enumerate() {
        let d = [a.x1 - b.x1, a.y1 - b.y1, a.x2 - b.x2, a.y2 - b.y2].map(f32::abs);
        assert!(d.iter().all(|&e| e <= 0.05), "box {i}: {b:?} vs {a:?}");
    }
    assert!(!w.detections.is_empty());
    assert_eq!(g.detections.len(), w.detections.len());
}
