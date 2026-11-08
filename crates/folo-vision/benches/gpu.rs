//! The FOLO detector on the GPU, the numbers in docs/BENCHMARKS.md's FOLO columns. Run with
//! `cargo bench -p folo-vision --bench gpu` (`FOLO_BENCH_SCALES=n,s` picks scales, default `n,s`).
//!
//! Per scale, 80 classes, 640×640, f32, untrained (seed-0) weights:
//! - **forward**: the fused model (`fuse()`: Conv+BN folded, Conv + bias + SiLU in one kernel), eval mode,
//!   `no_grad`, input already on the GPU. Latency: every call synchronised, 25 warm-up calls, median / p90 of
//!   100 (batch 1) or 30 (batch 16). Throughput: the same calls back to back with one synchronise at the end.
//! - **predict (end to end)**: decoded `u8` RGB 640×640 images in, letterbox on the CPU (folo-data), upload of
//!   the `u8` pixels, cast / HWC→CHW / `÷255` on the GPU, forward, download of the raw head, decode and
//!   class-aware NMS on the CPU (Ultralytics predict defaults), boxes mapped back through the letterbox. The
//!   timing ends with the boxes on the host. Untrained weights give almost no candidates above `conf`, so
//!   NMS is nearly free here; with a trained model it adds about 0.1–0.3 ms per image (`benches/post.rs`).
//! - **train step**: one fixed batch of 16 random images with 8 random boxes each on the GPU; train-mode
//!   forward (fused BatchNorm + SiLU kernels), the detection loss (task-aligned assigner on host copies),
//!   backward, SGD (momentum 0.937, Nesterov, weight decay on the decay group). 3 warm-up steps, median of 10.
//! - **peak memory**: the buffer pool's peak live bytes over the measured calls (PyTorch's
//!   `max_memory_allocated` counterpart, in pool size classes, which round up by at most 25%).
//!
//! Before every section the bench prints the GPU utilisation `nvidia-smi` reports, because the GPU is shared
//! with other jobs on the dev machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_data::{Image, LetterboxOptions, letterbox};
use folo_gpu::GpuContext;
use folo_nn::Module;
use folo_nn::optim::{Optimizer, Sgd, StandardGroups};
use folo_vision::boxes::Xyxy;
use folo_vision::loss::{ImageTargets, LossConfig, detection_loss};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::NmsConfig;
use rayon::prelude::*;

struct Stats {
    median: f64,
    p90: f64,
    min: f64,
}

fn stats(mut t: Vec<Duration>) -> Stats {
    t.sort();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    Stats { median: ms(t[t.len() / 2]), p90: ms(t[(t.len() * 9 / 10).min(t.len() - 1)]), min: ms(t[0]) }
}

/// Every call synchronised.
fn latency(dev: Device, warm: usize, iters: usize, mut f: impl FnMut()) -> Stats {
    for _ in 0..warm {
        f();
        synchronize(dev).expect("sync");
    }
    stats(
        (0..iters)
            .map(|_| {
                let t = Instant::now();
                f();
                synchronize(dev).expect("sync");
                t.elapsed()
            })
            .collect(),
    )
}

/// Back to back, one synchronise at the end: ms per call.
fn throughput(dev: Device, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    synchronize(dev).expect("sync");
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    synchronize(dev).expect("sync");
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

fn gpu_load() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "n/a".into())
}

fn row(name: &str, s: &Stats, batch: usize) {
    println!(
        "  {name:<34} median {:>8.2} ms  p90 {:>8.2}  min {:>8.2}  ({:>6.0} img/s at the median)",
        s.median,
        s.p90,
        s.min,
        batch as f64 * 1e3 / s.median
    );
}

fn mib(b: u64) -> f64 {
    b as f64 / (1 << 20) as f64
}

/// Letterbox on the CPU, then `u8` NHWC → `f32` NCHW in `0..1` on the device.
fn preprocess(images: &[Image], dev: Device) -> (Tensor, Vec<folo_data::Letterbox>) {
    let opts = LetterboxOptions::default();
    let boxed: Vec<_> = images.par_iter().map(|im| letterbox(im, 640, 640, &opts)).collect();
    let mut pixels = Vec::with_capacity(images.len() * 640 * 640 * 3);
    for (im, _) in &boxed {
        pixels.extend_from_slice(&im.data);
    }
    let x = Tensor::from_vec(pixels, [images.len(), 640, 640, 3])
        .and_then(|t| t.to_device(dev))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.permute([0, 3, 1, 2]))
        .and_then(|t| t.mul_scalar(1.0 / 255.0))
        .expect("preprocess");
    (x, boxed.into_iter().map(|(_, lb)| lb).collect())
}

fn bench_scale(scale: &str, dev: Device, ctx: &GpuContext) {
    let cfg = ModelConfig::new(scale, 80).expect("config");
    let mut model = FoloModel::new(cfg.clone()).expect("model");
    println!("\nfolo-{scale}: {} parameters (unfused)", model.num_parameters());
    model.to_device(dev).expect("to_device");
    model.fuse().expect("fuse");
    println!("  fused: {} parameters", model.num_parameters());
    let rng = &mut Rng::seed(0);
    let _g = no_grad();

    // Forward.
    println!("  [gpu load before forward: {}]", gpu_load());
    for batch in [1usize, 16] {
        let x = Tensor::rand_uniform([batch, 3, 640, 640], 0.0, 1.0, rng).to_device(dev).expect("upload");
        let w0 = ctx.work_stats();
        black_box(model.forward(&x).expect("forward"));
        synchronize(dev).expect("sync");
        let w1 = ctx.work_stats();
        ctx.reset_peak_memory();
        let base = ctx.pool_stats().live_bytes;
        let iters = if batch == 1 { 100 } else { 30 };
        let s = latency(dev, 25, iters, || drop(black_box(model.forward(&x).expect("forward"))));
        let peak = ctx.pool_stats().peak_live_bytes;
        row(&format!("forward b{batch} (fused, f32)"), &s, batch);
        let t = throughput(dev, iters, || drop(black_box(model.forward(&x).expect("forward"))));
        println!(
            "    pipelined {:.2} ms/call = {:.0} img/s; {} dispatches, {} submissions per call; peak {:.0} MiB \
             (activations {:.0} MiB above weights + input)",
            t,
            batch as f64 * 1e3 / t,
            w1.dispatches - w0.dispatches,
            w1.submissions - w0.submissions,
            mib(peak),
            mib(peak.saturating_sub(base)),
        );
    }

    // End-to-end predict.
    println!("  [gpu load before predict: {}]", gpu_load());
    let nms = NmsConfig::default();
    for batch in [1usize, 16] {
        let images: Vec<Image> = (0..batch)
            .map(|_| {
                let data =
                    (0..640 * 640 * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
                Image::from_raw(640, 640, data).expect("image")
            })
            .collect();
        let mut dets = 0;
        let run = |dets: &mut usize| {
            let (x, lbs) = preprocess(&images, dev);
            let pred = model.predict_raw(&x, &nms).expect("predict");
            let mapped: Vec<Vec<[f32; 4]>> = pred
                .images
                .iter()
                .zip(&lbs)
                .map(|(im, lb)| {
                    im.detections
                        .iter()
                        .map(|d| lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]))
                        .collect()
                })
                .collect();
            *dets = mapped.iter().map(Vec::len).sum();
            black_box(mapped);
        };
        let iters = if batch == 1 { 50 } else { 15 };
        let s = latency(dev, 10, iters, || run(&mut dets));
        row(&format!("predict b{batch} (e2e, f32)"), &s, batch);
        // Where the time goes: each phase synchronised on its own.
        let (mut pre, mut fwd, mut post) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..iters {
            let t0 = Instant::now();
            let (x, _) = preprocess(&images, dev);
            synchronize(dev).expect("sync");
            let t1 = Instant::now();
            let out = model.forward(&x).expect("forward");
            synchronize(dev).expect("sync");
            let t2 = Instant::now();
            let decoded = out.decode_images().expect("decode");
            let n: usize = decoded
                .into_par_iter()
                .map(|(b, sc)| folo_vision::nms::non_max_suppression(&b, &sc, 80, &nms).len())
                .sum();
            black_box(n);
            pre.push(t1 - t0);
            fwd.push(t2 - t1);
            post.push(t2.elapsed());
        }
        let (a, b, c) = (stats(pre), stats(fwd), stats(post));
        println!(
            "    split (medians): preprocess {:.2} ms, forward {:.2} ms, download + decode + NMS {:.2} ms;              {dets} detections in the last call",
            a.median, b.median, c.median
        );
    }
    drop(model);
    drop(_g);

    // Training step.
    println!("  [gpu load before train: {}]", gpu_load());
    let mut model = FoloModel::new(cfg).expect("model");
    model.to_device(dev).expect("to_device");
    let groups = StandardGroups::from_module(&model).into_groups(0.01, 0.937, 5e-4);
    let mut opt = Sgd::new(groups).expect("sgd");
    let x = Tensor::rand_uniform([16, 3, 640, 640], 0.0, 1.0, rng).to_device(dev).expect("upload");
    let targets: Vec<ImageTargets> = (0..16)
        .map(|_| {
            let boxes = (0..8)
                .map(|_| {
                    let (cx, cy) = (rng.uniform(64.0, 576.0), rng.uniform(64.0, 576.0));
                    let (w, h) = (rng.uniform(16.0, 128.0), rng.uniform(16.0, 128.0));
                    Xyxy::new(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                })
                .collect();
            let labels = (0..8).map(|_| (rng.uniform(0.0, 80.0) as u32).min(79)).collect();
            ImageTargets { boxes, labels }
        })
        .collect();
    let lcfg = LossConfig::default();
    let (mut t_fwd, mut t_loss, mut t_bwd, mut t_opt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut step = |record: bool| {
        let t0 = Instant::now();
        let out = model.forward(&x).expect("forward");
        synchronize(dev).expect("sync");
        let t1 = Instant::now();
        let loss = detection_loss(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &lcfg).expect("loss");
        synchronize(dev).expect("sync");
        let t2 = Instant::now();
        let grads = loss.total.backward().expect("backward");
        drop(out);
        drop(loss);
        synchronize(dev).expect("sync");
        let t3 = Instant::now();
        opt.step(&grads).expect("step");
        drop(grads);
        synchronize(dev).expect("sync");
        if record {
            t_fwd.push(t1 - t0);
            t_loss.push(t2 - t1);
            t_bwd.push(t3 - t2);
            t_opt.push(t3.elapsed());
        }
        t0.elapsed()
    };
    let w0 = ctx.work_stats();
    for _ in 0..3 {
        step(false);
    }
    let w1 = ctx.work_stats();
    ctx.reset_peak_memory();
    let s = stats((0..10).map(|_| step(true)).collect());
    let peak = ctx.pool_stats().peak_live_bytes;
    row("train step b16 (f32, SGD)", &s, 16);
    let (f, l, b, o) = (stats(t_fwd), stats(t_loss), stats(t_bwd), stats(t_opt));
    println!(
        "    split (medians, each phase synchronised): forward {:.1} ms, loss {:.1} ms, backward {:.1} ms, \
         SGD {:.1} ms; {} dispatches and {} submissions per step; peak {:.0} MiB",
        f.median,
        l.median,
        b.median,
        o.median,
        (w1.dispatches - w0.dispatches) / 3,
        (w1.submissions - w0.submissions) / 3,
        mib(peak)
    );
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            return;
        }
    };
    let ctx = GpuContext::shared().expect("context");
    let s = ctx.summary();
    println!("folo-vision GPU bench on {} ({}, driver {})", s.name, s.backend, s.driver);
    let scales = std::env::var("FOLO_BENCH_SCALES").unwrap_or_else(|_| "n,s".into());
    for scale in scales.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        bench_scale(scale, dev, &ctx);
    }
}
