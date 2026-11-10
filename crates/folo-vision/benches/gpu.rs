//! The FOLO detector on the GPU, the numbers in docs/BENCHMARKS.md's FOLO columns. Run with
//! `cargo bench -p folo-vision --bench gpu` (`FOLO_BENCH_SCALES=n,s` picks scales, default `n,s`;
//! `FOLO_BENCH_PRECISION=f32,f16,f16tc` the inference precisions, default all three, f16 only where the GPU
//! computes in it and `f16tc` (f16 on tensor cores) only with `FOLO_GPU_TENSOR_CORES=1` on an adapter that has them;
//! `FOLO_BENCH_SECTIONS=forward,predict,train` the sections, default all; `FOLO_BENCH_TRAIN=f32,amp` the
//! training precisions, default both, `amp` only on a context with tensor cores; `FOLO_BENCH_CLIP=1` adds the
//! trainer's global-norm gradient clipping (`clip_grad_norm` 10, through a `GradAccumulator` as in training) to
//! the training step, default off so the documented rows stay comparable; `FOLO_BENCH_OPTIM=musgd` trains with
//! MuSGD instead of SGD).
//!
//! Per scale, 80 classes, 640×640, untrained (seed-0) weights, inference in f32 and f16 (half precision: f16
//! weights and activations, f32 accumulation; the input is f16 too, and the predictor letterboxes into f16):
//! - **forward**: the fused model (`fuse()`: Conv+BN folded, Conv + bias + SiLU in one kernel), eval mode,
//!   `no_grad`, input already on the GPU. Latency: every call synchronised, 25 warm-up calls, median / p90 of
//!   100 (batch 1) or 30 (batch 16). Throughput: the same calls back to back with one synchronise at the end.
//! - **predict (end to end)**: `folo_vision::predict::Predictor` (`predict` for batch 1, `predict_batch` for
//!   16) on decoded `u8` RGB 640×640 images: upload of the raw `u8` pixels, letterbox + `÷255` on the GPU,
//!   forward, decode on the GPU into the candidates above `conf`, one small download, class-aware NMS on the
//!   CPU (Ultralytics predict defaults), boxes mapped back through the letterbox. The timing ends with the
//!   boxes on the host. Untrained weights give almost no candidates above `conf`, so NMS is nearly free here;
//!   with a trained model it adds about 0.1–0.3 ms per image (`benches/post.rs`).
//! - **train step**: one fixed batch of 16 random images with 8 random boxes each on the GPU; train-mode
//!   forward (fused BatchNorm + SiLU kernels), the detection loss (assigner and terms as GPU kernels, one
//!   readback of the reported items),
//!   backward, SGD (momentum 0.937, Nesterov, weight decay on the decay group). 3 warm-up steps, median of 10.
//!   **amp** (mixed precision, docs/MODEL.md §8.8) is the incumbent's `autocast` + `GradScaler` step: the forward
//!   under `folo_core::autocast` (f16 on tensor cores), the loss in f32, backward of the loss times the loss scale,
//!   the gradients unscaled and checked for overflow on the GPU (one flag readback), SGD, the scaler update.
//! - **peak memory**: the buffer pool's peak live bytes over the measured calls (PyTorch's
//!   `max_memory_allocated` counterpart, in pool size classes, which round up by at most 25%).
//!
//! Before every section the bench idles for a second and prints the GPU utilisation `nvidia-smi` reports,
//! because the GPU is shared with other jobs on the dev machine.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions};
use folo_gpu::GpuContext;
use folo_nn::optim::{GradAccumulator, MuSgd, Optimizer, Sgd, StandardGroups};
use folo_nn::{GradScaler, Module};
use folo_vision::boxes::Xyxy;
use folo_vision::loss::{ImageTargets, LossConfig, detection_loss, detection_loss_host};
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, PredictConfig, Predictor, f16_unavailable, letterbox_on_device_as};

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

/// `nvidia-smi`'s utilisation after a second of idling, so its sampling window no longer covers this
/// process's previous section: what remains is other processes' load.
fn gpu_load() -> String {
    std::thread::sleep(Duration::from_secs(1));
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

/// The inference sections (forward, end-to-end predict) of folo-`scale` at `prec`.
/// `tc`: run the f16 GEMMs and convs on tensor cores (the output names it in the precision label).
fn bench_infer(scale: &str, dev: Device, ctx: &GpuContext, prec: Precision, tc: bool, sections: &str) {
    ctx.set_tensor_cores(tc);
    let label = if tc { format!("{prec}, tensor cores") } else { prec.to_string() };
    let cfg = ModelConfig::new(scale, 80).expect("config");
    let mut model = FoloModel::new(cfg).expect("model");
    println!("\nfolo-{scale} {label}: {} parameters (unfused)", model.num_parameters());
    model.to_device(dev).expect("to_device");
    model.fuse().expect("fuse");
    model.to_dtype(prec.dtype()).expect("to_dtype");
    println!("  fused: {} parameters", model.num_parameters());
    let rng = &mut Rng::seed(0);
    let _g = no_grad();

    // Forward.
    if sections.contains("forward") {
        println!("  [gpu load before forward: {}]", gpu_load());
        for batch in [1usize, 16] {
            let x = Tensor::rand_uniform([batch, 3, 640, 640], 0.0, 1.0, rng)
                .to_device(dev)
                .expect("upload")
                .to_dtype(prec.dtype())
                .expect("dtype");
            let w0 = ctx.work_stats();
            black_box(model.forward(&x).expect("forward"));
            synchronize(dev).expect("sync");
            let w1 = ctx.work_stats();
            ctx.reset_peak_memory();
            let base = ctx.pool_stats().live_bytes;
            let iters = if batch == 1 { 100 } else { 30 };
            let s = latency(dev, 25, iters, || drop(black_box(model.forward(&x).expect("forward"))));
            let peak = ctx.pool_stats().peak_live_bytes;
            row(&format!("forward b{batch} (fused, {label})"), &s, batch);
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
    }
    if !sections.contains("predict") {
        return;
    }

    // End-to-end predict, through the predictor (rect letterbox for one image, square for a batch; a
    // 640×640 image is the identity either way).
    println!("  [gpu load before predict: {}]", gpu_load());
    let pred = Predictor::with_precision(model, PredictConfig::default(), prec).expect("predictor");
    assert_eq!(pred.precision(), prec);
    let model = pred.model();
    let nms = folo_vision::nms::NmsConfig::default();
    for batch in [1usize, 16] {
        let images: Vec<Image> = (0..batch)
            .map(|_| {
                let data =
                    (0..640 * 640 * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
                Image::from_raw(640, 640, data).expect("image")
            })
            .collect();
        let mut dets = 0;
        let mut run = || {
            let p = if batch == 1 {
                vec![pred.predict(&images[0]).expect("predict")]
            } else {
                pred.predict_batch(&images).expect("predict")
            };
            dets = p.iter().map(|p| p.detections.len()).sum();
            black_box(p);
        };
        let iters = if batch == 1 { 50 } else { 15 };
        let s = latency(dev, 10, iters, &mut run);
        row(&format!("predict b{batch} (e2e, {label})"), &s, batch);
        let t = throughput(dev, iters, &mut run);
        println!("    back to back {:.2} ms/call = {:.0} img/s", t, batch as f64 * 1e3 / t);
        // Where the time goes: each phase synchronised on its own.
        let opts = LetterboxOptions { stride: (batch == 1).then_some(32), ..LetterboxOptions::default() };
        let boxes: Vec<Letterbox> =
            images.iter().map(|i| Letterbox::compute(i.width, i.height, 640, 640, &opts)).collect();
        let (mut pre, mut fwd, mut post) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..iters {
            let t0 = Instant::now();
            let x = letterbox_on_device_as(&images, &boxes, dev, prec.dtype())
                .expect("letterbox")
                .expect("GPU letterbox");
            synchronize(dev).expect("sync");
            let t1 = Instant::now();
            let out = model.forward(&x).expect("forward");
            synchronize(dev).expect("sync");
            let t2 = Instant::now();
            black_box(out.detections(&nms).expect("detections"));
            pre.push(t1 - t0);
            fwd.push(t2 - t1);
            post.push(t2.elapsed());
        }
        let (a, b, c) = (stats(pre), stats(fwd), stats(post));
        println!(
            "    split (medians): upload + letterbox {:.2} ms, forward {:.2} ms, decode + download + NMS {:.2} ms; \
             {dets} detections in the last call",
            a.median, b.median, c.median
        );
    }
}

/// The training step of folo-`scale`, f32 or mixed precision (`amp`).
fn bench_train(scale: &str, dev: Device, ctx: &GpuContext, amp: bool) {
    let cfg = ModelConfig::new(scale, 80).expect("config");
    let rng = &mut Rng::seed(1);
    let label = if amp { "amp" } else { "f32" };
    // The trainer's default clipping (docs/MODEL.md §8.3), off unless `FOLO_BENCH_CLIP=1`.
    let clip = if std::env::var("FOLO_BENCH_CLIP").is_ok_and(|v| v == "1") { 10.0 } else { 0.0 };
    let label = if clip > 0.0 { format!("{label}, clip {clip}") } else { label.to_string() };
    println!("\nfolo-{scale} training ({label})");
    println!("  [gpu load before train: {}]", gpu_load());
    let mut model = FoloModel::new(cfg).expect("model");
    model.to_device(dev).expect("to_device");
    let groups = StandardGroups::from_module(&model).into_groups(0.01, 0.937, 5e-4);
    let musgd = std::env::var("FOLO_BENCH_OPTIM").is_ok_and(|v| v == "musgd");
    let optim = if musgd { "MuSGD" } else { "SGD" };
    let mut opt: Box<dyn Optimizer> = if musgd {
        Box::new(MuSgd::new(groups).expect("musgd"))
    } else {
        Box::new(Sgd::new(groups).expect("sgd"))
    };
    let mut acc = GradAccumulator::new(&model.parameters());
    let mut scaler = GradScaler::new();
    let mut skipped = 0;
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
    // `split` synchronises after every phase (forward, loss, backward, SGD) to time them; otherwise the step runs
    // as training does, with the loss's one readback as its only wait, and is synchronised at the end. `host`
    // takes the host assigner (the previous loss, `detection_loss_host`) for comparison. Returns the step time
    // and the four phase times.
    // `FOLO_GPU_PROFILE=1`: GPU time per kernel label, per phase (the context's timestamp queries).
    let profile: std::cell::RefCell<[Profile; 4]> = Default::default();
    let mut step = |split: bool, host: bool| {
        let mut phase = 0;
        // Host time spent encoding each phase (until its synchronisation starts).
        let mut enc = [Duration::ZERO; 4];
        let mut mark = Instant::now();
        let mut sync = |on: bool| {
            enc[phase] = mark.elapsed();
            if on {
                synchronize(dev).expect("sync");
            }
            for r in ctx.profile_take() {
                let mut p = profile.borrow_mut();
                let e = p[phase].entry(r.label).or_default();
                e.0 += r.count;
                e.1 += r.ms;
            }
            phase += 1;
            mark = Instant::now();
        };
        let t0 = Instant::now();
        let out = {
            let _a = amp.then(folo_core::autocast);
            model.forward(&x).expect("forward")
        };
        let (box_raw, cls_raw) =
            (out.box_raw.to_dtype(DType::F32).unwrap(), out.cls_raw.to_dtype(DType::F32).unwrap());
        sync(split);
        let t1 = Instant::now();
        let loss = if host {
            detection_loss_host(&box_raw, &cls_raw, &out.anchors, &targets, &lcfg)
        } else {
            detection_loss(&box_raw, &cls_raw, &out.anchors, &targets, &lcfg)
        }
        .expect("loss");
        sync(split);
        let t2 = Instant::now();
        let total =
            if amp { loss.total.mul_scalar(scaler.scale() as f32).unwrap() } else { loss.total.clone() };
        let grads = total.backward().expect("backward");
        drop((out, box_raw, cls_raw, total, loss));
        sync(split);
        let t3 = Instant::now();
        if amp {
            // `scaler.step(opt)`: unscale, the overflow check (one readback), the step unless it overflowed.
            acc.zero();
            acc.add(&grads).expect("accumulate");
            drop(grads);
            let finite = acc.unscale_clip(scaler.inv_scale(), clip).expect("unscale");
            if finite {
                opt.step(&acc).expect("step");
            } else {
                skipped += 1;
            }
            scaler.update(!finite);
        } else if clip > 0.0 {
            // The trainer's f32 window end: the gradients in the accumulator, clipped on the device, then the step.
            acc.zero();
            acc.add(&grads).expect("accumulate");
            drop(grads);
            acc.clip_grad_norm(clip).expect("clip");
            opt.step(&acc).expect("step");
        } else {
            opt.step(&grads).expect("step");
            drop(grads);
        }
        sync(true);
        (t0.elapsed(), [t1 - t0, t2 - t1, t3 - t2, t3.elapsed()], enc)
    };
    let w0 = ctx.work_stats();
    for _ in 0..3 {
        step(false, false);
    }
    let w1 = ctx.work_stats();
    ctx.reset_peak_memory();
    let s = stats((0..10).map(|_| step(false, false).0).collect());
    let peak = ctx.pool_stats().peak_live_bytes;
    row(&format!("train step b16 ({label}, {optim})"), &s, 16);
    let mut split = |host: bool| {
        step(true, host);
        let runs: Vec<([Duration; 4], [Duration; 4])> =
            (0..10).map(|_| step(true, host)).map(|r| (r.1, r.2)).collect();
        let m = |i: usize| stats(runs.iter().map(|r| r.0[i]).collect()).median;
        let e = |i: usize| stats(runs.iter().map(|r| r.1[i]).collect()).median;
        ((m(0), m(1), m(2), m(3)), [e(0), e(1), e(2), e(3)])
    };
    let ((f, l, b, o), enc) = split(false);
    println!(
        "    split (medians, each phase synchronised): forward {f:.1} ms, loss {l:.1} ms, backward {b:.1} ms, \
         {} {o:.1} ms; {} dispatches and {} submissions per step; peak {:.0} MiB",
        if amp { format!("unscale + check + {optim}") } else { optim.to_string() },
        (w1.dispatches - w0.dispatches) / 3,
        (w1.submissions - w0.submissions) / 3,
        mib(peak)
    );
    println!(
        "    host encode time of each phase (medians): forward {:.1} ms, loss {:.1} ms, backward {:.1} ms, step {:.1} ms",
        enc[0], enc[1], enc[2], enc[3]
    );
    if !amp && !ctx.profiling() {
        let ((_, l, b, _), _) = split(true);
        println!("    host assigner instead (detection_loss_host): loss {l:.1} ms, backward {b:.1} ms");
    }
    if ctx.profiling() {
        // Every step above was timed (profiling waits at each submission, so the wall-clock rows above are not
        // the real ones); count these steps only.
        profile.take();
        const STEPS: usize = 5;
        for _ in 0..STEPS {
            step(true, false);
        }
        print_profile(&profile.borrow(), STEPS);
    }
    if amp {
        println!("    loss scale {} after {skipped} skipped steps", scaler.scale());
    }
}

/// GPU milliseconds and dispatch counts per kernel label.
type Profile = std::collections::HashMap<String, (u64, f64)>;

/// The per-phase kernel timings of `steps` profiled training steps, per step: totals by label group (the label's
/// first word), then the largest single labels.
fn print_profile(profile: &[Profile; 4], steps: usize) {
    let per = 1.0 / steps as f64;
    for (name, rows) in ["forward", "loss", "backward", "optimizer"].iter().zip(profile) {
        let total: f64 = rows.values().map(|r| r.1).sum();
        let count: u64 = rows.values().map(|r| r.0).sum();
        println!(
            "    profile {name}: {:.2} ms GPU time in {} dispatches per step",
            total * per,
            count / steps as u64
        );
        let mut groups: Vec<(String, u64, f64)> = Vec::new();
        for (label, &(n, ms)) in rows {
            // `tag args/kernel` -> `tag/kernel`.
            let tag = label.split(' ').next().unwrap_or(label);
            let kernel = label.rsplit('/').next().unwrap_or("");
            let key = if label.contains('/') { format!("{tag}/{kernel}") } else { tag.to_string() };
            match groups.iter_mut().find(|g| g.0 == key) {
                Some(g) => {
                    g.1 += n;
                    g.2 += ms;
                }
                None => groups.push((key, n, ms)),
            }
        }
        groups.sort_by(|a, b| b.2.total_cmp(&a.2));
        for (key, n, ms) in groups.iter().filter(|g| g.2 * per >= 0.02) {
            println!("      {:>7.2} ms {:>5} x  {key}", ms * per, n / steps as u64);
        }
        let mut top: Vec<(&String, &(u64, f64))> = rows.iter().collect();
        top.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        println!("      largest labels:");
        for (label, (n, ms)) in top.iter().take(45).filter(|r| r.1.1 * per >= 0.05) {
            println!("        {:>7.2} ms {:>4} x  {label}", ms * per, n / steps as u64);
        }
    }
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
    let sections = std::env::var("FOLO_BENCH_SECTIONS").unwrap_or_else(|_| "forward,predict,train".into());
    let train = std::env::var("FOLO_BENCH_TRAIN").unwrap_or_else(|_| "f32,amp".into());
    // `(precision, on tensor cores)` per entry.
    let precisions: Vec<(Precision, bool)> = std::env::var("FOLO_BENCH_PRECISION")
        .unwrap_or_else(|_| "f32,f16,f16tc".into())
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| match s {
            "f16tc" => (Precision::F16, true),
            s => (s.parse().expect("FOLO_BENCH_PRECISION: f32, f16 and / or f16tc"), false),
        })
        .collect();
    let tc = ctx.tensor_cores();
    for scale in scales.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for &(prec, on) in &precisions {
            if prec == Precision::F16
                && let Some(why) = f16_unavailable(dev)
            {
                println!("\nfolo-{scale} f16: skipped ({why})");
                continue;
            }
            if on && !ctx.caps().coop_f16 {
                println!("\nfolo-{scale} f16 on tensor cores: skipped ({})", ctx.tensor_cores_status());
                continue;
            }
            bench_infer(scale, dev, &ctx, prec, on, &sections);
        }
        ctx.set_tensor_cores(tc);
        if !sections.contains("train") {
            continue;
        }
        for p in train.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            match p {
                "f32" => bench_train(scale, dev, &ctx, false),
                "amp" if ctx.caps().coop_f16 && ctx.set_tensor_cores(true) => {
                    bench_train(scale, dev, &ctx, true);
                    ctx.set_tensor_cores(tc);
                }
                "amp" => println!("\nfolo-{scale} training (amp): skipped ({})", ctx.tensor_cores_status()),
                p => panic!("FOLO_BENCH_TRAIN: f32 and / or amp, got `{p}`"),
            }
        }
    }
}
