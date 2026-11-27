//! Where the GPU time of a fused inference forward goes, dispatch by dispatch. Run with
//! `cargo bench -p folo-vision --bench gpu_profile` (`FOLO_BENCH_SCALES=n,s`, `FOLO_BENCH_PRECISION=f32,f16,f16tc`
//! and `FOLO_BENCH_BATCH=1,16` pick the configurations, defaults `n`, `f32,f16tc` and `1,16`; `f16tc` needs
//! `FOLO_GPU_TENSOR_CORES=1`; `FOLO_BENCH_OPS=1` also lists every dispatch in encoding order).
//!
//! Per configuration: the host-side split of one synchronised forward (encode, submit, wait), the pipelined time
//! per call, then one forward is recorded ([`GpuContext::trace_begin`]) and every recorded dispatch is replayed
//! on its own, many times back to back with one wait at the end. The table sums those times per kernel: the sum
//! is what the forward would cost if nothing overlapped, and its split is the profile.

use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, f16_unavailable};

fn median(mut t: Vec<Duration>) -> f64 {
    t.sort();
    t[t.len() / 2].as_secs_f64() * 1e3
}

fn profile(scale: &str, dev: Device, ctx: &GpuContext, prec: Precision, tc: bool, batch: usize, list: bool) {
    ctx.set_tensor_cores(tc);
    let label = if tc { format!("{prec}, tensor cores") } else { prec.to_string() };
    let mut model = FoloModel::new(ModelConfig::new(scale, 80).expect("config")).expect("model");
    model.to_device(dev).expect("to_device");
    model.fuse().expect("fuse");
    model.to_dtype(prec.dtype()).expect("to_dtype");
    let rng = &mut Rng::seed(0);
    let _g = no_grad();
    let x = Tensor::rand_uniform([batch, 3, 640, 640], 0.0, 1.0, rng)
        .to_device(dev)
        .expect("upload")
        .to_dtype(prec.dtype())
        .expect("dtype");
    let forward = || drop(black_box(model.forward(&x).expect("forward")));
    for _ in 0..20 {
        forward();
        synchronize(dev).expect("sync");
    }
    // Host-side split of a synchronised call.
    let iters = if batch == 1 { 100 } else { 30 };
    let (mut enc, mut sub, mut wait) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..iters {
        let t0 = Instant::now();
        forward();
        let t1 = Instant::now();
        ctx.flush();
        let t2 = Instant::now();
        synchronize(dev).expect("sync");
        enc.push(t1 - t0);
        sub.push(t2 - t1);
        wait.push(t2.elapsed());
    }
    let (enc, sub, wait) = (median(enc), median(sub), median(wait));
    // Pipelined.
    let t = Instant::now();
    for _ in 0..iters {
        forward();
    }
    synchronize(dev).expect("sync");
    let piped = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "\nfolo-{scale} {label} b{batch}: encode {enc:.2} ms, submit {sub:.2} ms, wait {wait:.2} ms (sum {:.2}); \
         pipelined {piped:.2} ms/call = {:.0} img/s",
        enc + sub + wait,
        batch as f64 * 1e3 / piped
    );

    ctx.trace_begin();
    forward();
    let ops = ctx.trace_end();
    synchronize(dev).expect("sync");
    let reps = if batch == 1 { 200 } else { 40 };
    let mut times = Vec::with_capacity(ops.len());
    for op in &ops {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            for _ in 0..reps {
                ctx.replay(op).expect("replay");
            }
            let t = Instant::now();
            synchronize(dev).expect("sync");
            best = best.min(t.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
        times.push(best);
    }
    let total: f64 = times.iter().sum();
    println!("  {} dispatches, {total:.3} ms when each runs alone (best of 3 x {reps})", ops.len());
    if list {
        for (i, (op, t)) in ops.iter().zip(&times).enumerate() {
            println!("  {i:>4} {:>8.1} us  {:<22} {:?}  {}", t * 1e3, op.name, op.groups, op.spec);
        }
    }
    // Per kernel, and per kernel + spec.
    let mut by_name: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    let mut by_spec: BTreeMap<(String, String), (usize, f64)> = BTreeMap::new();
    for (op, t) in ops.iter().zip(&times) {
        let e = by_name.entry(op.name).or_default();
        *e = (e.0 + 1, e.1 + t);
        let e = by_spec.entry((op.name.to_string(), op.spec.clone())).or_default();
        *e = (e.0 + 1, e.1 + t);
    }
    let mut names: Vec<_> = by_name.into_iter().collect();
    names.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    for (name, (n, t)) in names {
        println!("  {name:<24} {n:>4} x  {t:>8.3} ms  {:>5.1}%", 100.0 * t / total);
    }
    let mut specs: Vec<_> = by_spec.into_iter().collect();
    specs.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    println!("  top kernel variants:");
    for ((name, spec), (n, t)) in specs.into_iter().take(14) {
        println!("    {t:>8.3} ms {:>5.1}%  {n:>3} x {name} [{spec}]", 100.0 * t / total);
    }
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to profile");
            return;
        }
    };
    let ctx = GpuContext::shared().expect("context");
    let s = ctx.summary();
    println!("folo-vision GPU profile on {} ({}, driver {})", s.name, s.backend, s.driver);
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let scales = var("FOLO_BENCH_SCALES", "n");
    let precisions = var("FOLO_BENCH_PRECISION", "f32,f16tc");
    let batches = var("FOLO_BENCH_BATCH", "1,16");
    let list = std::env::var("FOLO_BENCH_OPS").is_ok_and(|v| v != "0" && !v.is_empty());
    let tc = ctx.tensor_cores();
    for scale in scales.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for p in precisions.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (prec, on) = match p {
                "f16tc" => (Precision::F16, true),
                p => (p.parse().expect("FOLO_BENCH_PRECISION: f32, f16 and / or f16tc"), false),
            };
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
            for b in batches.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let batch: usize = b.parse().expect("FOLO_BENCH_BATCH: batch sizes");
                profile(scale, dev, &ctx, prec, on, batch, list);
            }
        }
    }
    ctx.set_tensor_cores(tc);
}
