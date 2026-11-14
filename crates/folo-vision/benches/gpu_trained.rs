//! End-to-end GPU prediction with trained weights on real images (docs/BENCHMARKS.md's GPU predict rows, measured
//! where the model finds objects: untrained weights pass no candidate, so the detect head's box branch would cost
//! nothing in candidate-first prediction, docs/MODEL.md §9.5). Run with
//! `cargo bench -p folo-vision --bench gpu_trained`.
//!
//! `FOLO_BENCH_WEIGHTS` (required): a `.folo` path in which `{scale}` stands for the scale; `FOLO_BENCH_SCALES`
//! (default `n,s`); `FOLO_BENCH_IMAGES` (required): an image folder (e.g. COCO val2017), of which the first
//! `FOLO_BENCH_IMAGES_N` (default 16) by name are used; `FOLO_BENCH_PRECISION=f32,f16,f16tc` as in the `gpu` bench
//! (`f16tc` only with tensor cores); `FOLO_BENCH_ITERS` (default 100) batch-1 calls, a third as many batch-16 calls.
//!
//! Rows, through the public `Predictor` with its defaults (conf 0.25): **b1** `predict` on each image in turn
//! (rect letterbox, so the input shape changes with the image), **b1 one** `predict` on the first image only (one
//! input shape), **b1 640** `predict` on each image resized to 640×640 in turn (the 640×640 decoded images of the
//! targets table's end-to-end rows, docs/BENCHMARKS.md), **b16** `predict_batch` on all of them (square 640),
//! **b16 640** `predict_batch` on all of them resized to 640×640 (the targets table's batch-16 row), **b16 hd**
//! `predict_batch` on all of them resized to 1280×720 (one `rect` canvas, 640×384). Every call synchronised, after a
//! second of warm-up calls; median, p90 and min, and the detections per image of the last pass. A row asked to run
//! on tensor cores fails if they were off at its end (a cooperative kernel that fails to build turns them off).
//!
//! `FOLO_BENCH_PLANS=toggle` instead compares forward plans on (a `Predictor`'s default) and off, in one process
//! with two predictors of the same weights, alternated for `FOLO_BENCH_ROUNDS` (default 6) rounds of
//! `FOLO_BENCH_ITERS` calls per row and side: **b1** and **b1 one** as above, and **step**, the model step alone
//! (`predict_candidates` on the first image's prepared input, which ends in its readback). Per row and side: p50,
//! p90 and p99 of every call, and the median and range of the round medians.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::Device;
use folo_core::backend::synchronize;
use folo_data::Image;
use folo_gpu::GpuContext;
use folo_vision::nms::NmsConfig;
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

/// `iters` synchronised calls of `f` in ms, after `warm` unmeasured ones.
fn calls(dev: Device, warm: usize, iters: usize, mut f: impl FnMut()) -> Vec<f64> {
    for _ in 0..warm {
        f();
        synchronize(dev).expect("sync");
    }
    (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            synchronize(dev).expect("sync");
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect()
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() as f64 * q) as usize).min(s.len() - 1)]
}

/// `FOLO_BENCH_PLANS=toggle`: plans on against off, alternated (see the module docs).
fn toggle(dev: Device, path: &str, prec: Precision, label: &str, images: &[Image], iters: usize) {
    let rounds: usize = env("FOLO_BENCH_ROUNDS").unwrap_or(6);
    let on = Predictor::open_on(path, PredictConfig::default(), dev, prec).expect("predictor");
    let off = Predictor::open_on(path, PredictConfig::default(), dev, prec).expect("predictor");
    off.model().set_forward_plans(false);
    let nms = NmsConfig { conf: on.config().conf, ..NmsConfig::default() };
    let x = on.prepare(std::slice::from_ref(&images[0]), true).expect("prepare").input;
    const ROWS: [&str; 3] = ["b1", "b1 one", "step"];
    // [row][side: 0 on, 1 off] -> every call, and each round's median.
    let mut all: Vec<[Vec<f64>; 2]> = (0..3).map(|_| Default::default()).collect();
    let mut med: Vec<[Vec<f64>; 2]> = (0..3).map(|_| Default::default()).collect();
    for r in 0..rounds {
        for k in 0..2 {
            let side = (r + k) % 2;
            let pred = if side == 0 { &on } else { &off };
            let mut next = 0;
            let v = [
                calls(dev, 20, iters, || {
                    black_box(pred.predict(&images[next % images.len()]).expect("predict"));
                    next += 1;
                }),
                calls(dev, 20, iters, || drop(black_box(pred.predict(&images[0]).expect("predict")))),
                calls(dev, 20, iters, || {
                    drop(black_box(pred.model().predict_candidates(&x, &nms).expect("candidates")))
                }),
            ];
            for (row, v) in v.into_iter().enumerate() {
                med[row][side].push(pct(&v, 0.5));
                all[row][side].extend(v);
            }
        }
    }
    println!(
        "trained {label}: plans answered {} calls (last refusal {:?}); the plans-off side {}",
        on.model().plan_replays(),
        on.model().plan_refusal(),
        off.model().plan_replays()
    );
    for (row, name) in ROWS.iter().enumerate() {
        let side = |s: usize| {
            let (a, m) = (&all[row][s], &med[row][s]);
            let lo = m.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = m.iter().copied().fold(0.0, f64::max);
            format!(
                "p50 {:>6.3} p90 {:>6.3} p99 {:>6.3} rounds {:.3} ({lo:.3}-{hi:.3})",
                pct(a, 0.5),
                pct(a, 0.9),
                pct(a, 0.99),
                pct(m, 0.5)
            )
        };
        let (a, b) = (pct(&all[row][0], 0.5), pct(&all[row][1], 0.5));
        println!("  {name:<7} on  {}", side(0));
        println!("  {name:<7} off {}   on vs off p50 {:+.1}%", side(1), (a / b - 1.0) * 100.0);
    }
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
    let hd: Vec<Image> = images.iter().map(|i| folo_data::resize::resize(i, 1280, 720)).collect();
    let sq: Vec<Image> = images.iter().map(|i| folo_data::resize::resize(i, 640, 640)).collect();
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
            if std::env::var("FOLO_BENCH_PLANS").is_ok_and(|v| v == "toggle") {
                toggle(dev, &path, prec, &format!("folo-{scale} {p}"), &images, iters);
                assert_eq!(ctx.tensor_cores(), on, "folo-{scale} {p}: {}", ctx.tensor_cores_status());
                continue;
            }
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
            let mut one = 0;
            let b1_one = latency(dev, iters, || {
                let r = pred.predict(&images[0]).expect("predict");
                one = r.detections.len();
                black_box(r);
            });
            let (mut next_sq, mut dets_sq) = (0, 0);
            let b1_sq = latency(dev, iters, || {
                let r = pred.predict(&sq[next_sq % sq.len()]).expect("predict");
                next_sq += 1;
                dets_sq += r.detections.len();
                black_box(r);
            });
            let mut last = 0;
            let b16 = latency(dev, iters.div_ceil(3), || {
                let r = pred.predict_batch(&images).expect("predict");
                last = r.iter().map(|r| r.detections.len()).sum();
                black_box(r);
            });
            let mut last_sq = 0;
            let b16_sq = latency(dev, iters.div_ceil(3), || {
                let r = pred.predict_batch(&sq).expect("predict");
                last_sq = r.iter().map(|r| r.detections.len()).sum();
                black_box(r);
            });
            let mut last_hd = 0;
            let b16_hd = latency(dev, iters.div_ceil(3), || {
                let r = pred.predict_batch(&hd).expect("predict");
                last_hd = r.iter().map(|r| r.detections.len()).sum();
                black_box(r);
            });
            assert_eq!(ctx.tensor_cores(), on, "folo-{scale} {p}: {}", ctx.tensor_cores_status());
            let rows = [
                ("b1", b1, 1, b1_dets),
                ("b1 one", b1_one, 1, one as f64),
                ("b1 640", b1_sq, 1, dets_sq as f64 / next_sq as f64),
                ("b16", b16, images.len(), last as f64 / images.len() as f64),
                ("b16 640", b16_sq, sq.len(), last_sq as f64 / sq.len() as f64),
                ("b16 hd", b16_hd, hd.len(), last_hd as f64 / hd.len() as f64),
            ];
            for (name, (med, p90, min), batch, d) in rows {
                println!(
                    "trained folo-{scale} {p:<6} predict {name:<6} median {med:>8.2} ms  p90 {p90:>8.2}  min {min:>8.2}  \
                     ({:>6.0} img/s, {d:.1} det/img)",
                    batch as f64 * 1e3 / med
                );
            }
        }
    }
    ctx.set_tensor_cores(tc);
}
