//! Real end-to-end GPU [`Trainer`] throughput and per-window cycle benchmark.
//!
//! Measures `folo-n` and `folo2-n` (both `f32` and `amp`, `batch = 16`, `nbs = 64` -> `accumulate = 4`)
//! on a RAM-cached synthetic dataset with augmentation off so timing isolates GPU training + host readback
//! overhead rather than disk I/O. Reports epoch-2 throughput (`img/s`), per-window 4-batch cycle time (`ms`),
//! and per-position step medians (`p1..p4`).
//!
//! Run with:
//! `FOLO_GPU_TENSOR_CORES=1 cargo bench -p folo-vision --bench gpu_trainer`
//! Optional env vars: `FOLO_BENCH_ARCHS=folo-n,folo2-n`, `FOLO_BENCH_MODES=f32,amp,o2o-amp`,
//! `FOLO_BENCH_IMGSZ=640`, `FOLO_BENCH_IMAGES=256`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use folo_api::RunEvent;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{AugmentConfig, CacheMode, Split};
use folo_vision::model::HeadMode;
use folo_vision::train::OptimizerChoice;
use folo_vision::{TrainConfig, TrainData, TrainOptions, Trainer};

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn make_data(images: usize, imgsz: u32) -> TrainData {
    let cfg = SyntheticConfig {
        images: images + 16,
        classes: vec!["circle".into(), "square".into(), "triangle".into(), "star".into()],
        imgsz,
        seed: 7,
        min_objects: 2,
        max_objects: 6,
        empty_fraction: 0.0,
        small_fraction: 0.0,
        vary_aspect: false,
        val_fraction: 16.0 / (images + 16) as f32,
        ..SyntheticConfig::default()
    };
    let train = SyntheticDataset::new(cfg.clone(), Split::Train).expect("train dataset");
    let val = SyntheticDataset::new(cfg.clone(), Split::Val).expect("val dataset");
    TrainData { train: Arc::new(train), val: Arc::new(val), classes: cfg.classes }
}

fn run_case(arch: &str, mode: &str, imgsz: u32, data: TrainData) {
    let (amp, head) = match mode {
        "f32" => (false, HeadMode::O2m),
        "amp" => (true, HeadMode::O2m),
        "o2o-f32" => (false, HeadMode::O2o),
        "o2o-amp" => (true, HeadMode::O2o),
        other => panic!("unknown FOLO_BENCH_MODES entry `{other}`"),
    };
    let cfg = TrainConfig {
        arch: arch.into(),
        epochs: 2,
        batch: 16,
        nbs: 64,
        imgsz,
        optimizer: OptimizerChoice::Sgd,
        lr0: 0.005,
        momentum: 0.937,
        warmup_epochs: 0.0,
        workers: 2,
        augment: AugmentConfig::none(),
        device: "gpu:0".into(),
        amp,
        head,
        ..TrainConfig::default()
    };
    type StepStamp = (u32, u64, Instant, f32);
    let times: Arc<Mutex<Vec<StepStamp>>> = Arc::default();
    let sink_times = times.clone();
    let opts = TrainOptions {
        step_interval: Duration::ZERO,
        cache: CacheMode::Ram,
        events: Some(Box::new(move |ev| {
            if let RunEvent::Step(s) = ev {
                sink_times.lock().expect("lock").push((s.epoch, s.step, Instant::now(), s.imgs_per_s));
            }
        })),
        ..TrainOptions::default()
    };
    let n_train = data.train.len();
    let mut trainer = match Trainer::new(cfg, data, opts) {
        Ok(t) => t,
        Err(e) => {
            println!("  {arch:<8} {mode:<8} skipped ({e})");
            return;
        }
    };
    let acc = trainer.accumulate();
    let nb = trainer.batches_per_epoch();
    let batch = trainer.config().batch;
    let summary = trainer.run().expect("train run");
    assert_eq!(summary.epochs_completed, 2);

    // Measure epoch 2 (steady-state warmed pipelines and cached RAM images).
    let recs = times.lock().expect("lock");
    let ep2: Vec<_> = recs.iter().filter(|r| r.0 == 2).collect();
    if ep2.len() < acc + 2 {
        println!("  {arch:<8} {mode:<8} too few finite steps in epoch 2 ({}/{nb})", ep2.len());
        return;
    }
    let mut by_pos: Vec<Vec<f64>> = vec![Vec::new(); acc];
    let mut total_dt_s = 0.0f64;
    let mut steady_images = 0usize;
    let batch_images = |step_in_epoch: usize| -> usize {
        let start = step_in_epoch * batch;
        n_train.saturating_sub(start).min(batch)
    };
    // Only include adjacent, gap-free finite step pairs so skipped non-finite batches never inflate one slot.
    for w in ep2.windows(2) {
        if w[1].1 != w[0].1 + 1 {
            continue;
        }
        let dt_ms = (w[1].2 - w[0].2).as_secs_f64() * 1e3;
        let step_in_epoch = ((w[1].1 - 1) as usize) % nb;
        total_dt_s += dt_ms * 1e-3;
        steady_images += batch_images(step_in_epoch);
    }
    // Collect per-position step times strictly from complete, gap-free windows of length `acc`.
    for i in 0..ep2.len() {
        let step0 = ((ep2[i].1 - 1) as usize) % nb;
        if !step0.is_multiple_of(acc) || i == 0 || ep2[i].1 != ep2[i - 1].1 + 1 || i + acc > ep2.len() {
            continue;
        }
        let complete = (1..acc).all(|j| ep2[i + j].1 == ep2[i + j - 1].1 + 1);
        if !complete {
            continue;
        }
        for j in 0..acc {
            let dt_ms = (ep2[i + j].2 - ep2[i + j - 1].2).as_secs_f64() * 1e3;
            by_pos[j].push(dt_ms);
        }
    }
    let pos_meds: Vec<f64> = by_pos.into_iter().map(median).collect();
    let cycle_ms: f64 = pos_meds.iter().sum();
    let steady_imgs_per_s = steady_images as f64 / total_dt_s.max(1e-9);
    let reported_imgs_per_s = ep2.last().map_or(0.0, |r| r.3 as f64);
    let skips = trainer.loss_scaler().map_or(0, |s| s.skipped_steps());
    println!(
        "  {arch:<8} {mode:<8} steady {:>6.1} img/s  (ep2 {:>6.1})  cycle {:>6.1} ms  pos {:?}  skips {skips}",
        steady_imgs_per_s,
        reported_imgs_per_s,
        cycle_ms,
        pos_meds.iter().map(|v| (v * 10.0).round() / 10.0).collect::<Vec<_>>()
    );
}

fn main() {
    folo_gpu::request_tensor_cores();
    let ctx = match folo_gpu::GpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            println!("no GPU adapter ({e}); skipping");
            return;
        }
    };
    if folo_gpu::init().is_err() {
        println!("GPU init failed; skipping");
        return;
    }
    let s = ctx.summary();
    let imgsz: u32 = std::env::var("FOLO_BENCH_IMGSZ").ok().and_then(|v| v.parse().ok()).unwrap_or(640);
    let images: usize = std::env::var("FOLO_BENCH_IMAGES").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    let archs = std::env::var("FOLO_BENCH_ARCHS").unwrap_or_else(|_| "folo-n,folo2-n".into());
    let modes = std::env::var("FOLO_BENCH_MODES").unwrap_or_else(|_| "f32,amp,o2o-amp".into());
    println!(
        "folo-vision GPU Trainer bench on {} ({}, tc={}) — {}px, {} images/epoch, batch 16, acc 4",
        s.name,
        s.backend,
        ctx.tensor_cores(),
        imgsz,
        images
    );
    let data = make_data(images, imgsz);
    for arch in archs.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for mode in modes.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            run_case(arch, mode, imgsz, data.clone());
        }
    }
}
