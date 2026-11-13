//! End-to-end GPU prediction with trained weights on real images (docs/BENCHMARKS.md's GPU predict rows, measured
//! where the model finds objects: untrained weights pass no candidate, so the detect head's box branch would cost
//! nothing in candidate-first prediction, docs/MODEL.md §9.4). Run with
//! `cargo bench -p folo-vision --bench gpu_trained`.
//!
//! `FOLO_BENCH_WEIGHTS` (required): a `.folo` path in which `{scale}` stands for the scale; `FOLO_BENCH_SCALES`
//! (default `n,s`); `FOLO_BENCH_IMAGES` (required): an image folder (e.g. COCO val2017), of which the first
//! `FOLO_BENCH_IMAGES_N` (default 16) by name are used; `FOLO_BENCH_PRECISION=f32,f16,f16tc` as in the `gpu` bench
//! (`f16tc` only with tensor cores); `FOLO_BENCH_ITERS` (default 100) batch-1 calls, a third as many batch-16 calls.
//!
//! Rows, through the public `Predictor` with its defaults (conf 0.25): **b1** `predict` on each image in turn
//! (rect letterbox, so the input shape changes with the image), **b16** `predict_batch` on all of them (square
//! 640). Every call synchronised, after a second of warm-up calls; median, p90 and min, and the detections per
//! image of the last pass.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::Device;
use folo_core::backend::synchronize;
use folo_data::Image;
use folo_gpu::GpuContext;
use folo_vision::predict::{Precision, PredictConfig, Predictor, f16_unavailable};

fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
    std::env::var(k).ok().and_then(|s| s.parse().ok())
}

/// `(median, p90, min)` in ms of `iters` synchronised calls after at least a second of warm-up calls.
fn latency(dev: Device, iters: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(1) {
        f();
        synchronize(dev).expect("sync");
    }
    let mut v: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            synchronize(dev).expect("sync");
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2], v[(v.len() * 9 / 10).min(v.len() - 1)], v[0])
}

fn main() {
    let (Some(weights), Some(dir)) =
        (env::<String>("FOLO_BENCH_WEIGHTS"), env::<String>("FOLO_BENCH_IMAGES"))
    else {
        println!("gpu_trained bench skipped: set FOLO_BENCH_WEIGHTS (with {{scale}}) and FOLO_BENCH_IMAGES");
        return;
    };
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            println!("no GPU adapter ({e}); nothing to benchmark");
            return;
        }
    };
    let ctx = GpuContext::shared().expect("context");
    let s = ctx.summary();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("FOLO_BENCH_IMAGES")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png"))
        })
        .collect();
    paths.sort();
    paths.truncate(env("FOLO_BENCH_IMAGES_N").unwrap_or(16));
    let images: Vec<Image> = paths.iter().map(|p| folo_data::read_image(p).expect("image")).collect();
    let iters: usize = env("FOLO_BENCH_ITERS").unwrap_or(100);
    println!(
        "gpu_trained bench on {} ({}, driver {}): {} images from {dir}",
        s.name,
        s.backend,
        s.driver,
        images.len()
    );
    let tc = ctx.tensor_cores();
    let scales = std::env::var("FOLO_BENCH_SCALES").unwrap_or_else(|_| "n,s".into());
    let precisions = std::env::var("FOLO_BENCH_PRECISION").unwrap_or_else(|_| "f32,f16,f16tc".into());
    for scale in scales.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let path = weights.replace("{scale}", scale);
        for p in precisions.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (prec, on) = match p {
                "f16tc" => (Precision::F16, true),
                s => (s.parse::<Precision>().expect("FOLO_BENCH_PRECISION: f32, f16 and / or f16tc"), false),
            };
            if prec == Precision::F16
                && let Some(why) = f16_unavailable(dev)
            {
                println!("folo-{scale} {p}: skipped ({why})");
                continue;
            }
            if on && !ctx.caps().coop_f16 {
                println!("folo-{scale} {p}: skipped ({})", ctx.tensor_cores_status());
                continue;
            }
            ctx.set_tensor_cores(on);
            let pred = Predictor::open_on(&path, PredictConfig::default(), dev, prec).expect("predictor");
            let mut next = 0;
            let mut dets = 0;
            let b1 = latency(dev, iters, || {
                let r = pred.predict(&images[next % images.len()]).expect("predict");
                next += 1;
                dets += r.detections.len();
                black_box(r);
            });
            let b1_dets = dets as f64 / next as f64;
            let mut last = 0;
            let b16 = latency(dev, iters.div_ceil(3), || {
                let r = pred.predict_batch(&images).expect("predict");
                last = r.iter().map(|r| r.detections.len()).sum();
                black_box(r);
            });
            for (name, (med, p90, min), batch, d) in
                [("b1", b1, 1, b1_dets), ("b16", b16, images.len(), last as f64 / images.len() as f64)]
            {
                println!(
                    "trained folo-{scale} {p:<6} predict {name:<4} median {med:>8.2} ms  p90 {p90:>8.2}  min {min:>8.2}  \
                     ({:>6.0} img/s, {d:.1} det/img)",
                    batch as f64 * 1e3 / med
                );
            }
        }
    }
    ctx.set_tensor_cores(tc);
}
