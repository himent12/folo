//! `POST /api/system/benchmark`: a few seconds of training and inference on synthetic data, recorded in the
//! throughput table next to what real runs and predictions measure (docs/API.md, *Throughput and estimates*).
//!
//! Training: folo-vision's trainer on a procedural shapes split (rendered on demand, so the loader renders instead
//! of decoding files), stopped once the training share of the budget has passed (or at the epoch's last batch, so it
//! never validates); the first 30% of that window is warm-up and left out. Inference: a randomly initialised model
//! through the predictor on one JPEG of the same generator, decoded on every call like an upload to
//! `/api/predict`, after 3 warm-up calls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{BenchmarkReport, BenchmarkRequest, InferPrecision, PhaseKind, RunEvent};
use folo_core::Device;
use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
use folo_data::{ImageFormat, Split};
use folo_vision::train::place_model;
use folo_vision::{
    FoloModel, ModelConfig, Precision, PredictConfig, Predictor, TrainData, TrainOptions, Trainer,
};

use crate::engine::{self, engine_error, nn_error};
use crate::error::{Error, Result};
use crate::throughput::{
    InferKey, InferSample, Throughput, TrainKey, TrainSample, WINDOW, infer_row, train_row,
};

/// Shortest and longest measuring budget, seconds.
pub const BUDGET_S: std::ops::RangeInclusive<f64> = 1.0..=10.0;
/// Share of the budget that trains (the rest predicts).
const TRAIN_SHARE: f64 = 0.6;
/// Share of the training window left out as warm-up.
const WARMUP_SHARE: f64 = 0.3;
/// Images of the synthetic training split (the probe stops at its last batch at the latest).
const IMAGES: usize = 8192;
/// Most predictions measured.
const MAX_PREDICTIONS: usize = 500;

/// `400 invalid_body` for settings outside the probe's limits.
pub fn check(req: &BenchmarkRequest) -> Result<()> {
    let bad = |m: &str| Err(Error::new(StatusCode::BAD_REQUEST, "invalid_body", m.to_string()));
    if !(64..=1280).contains(&req.imgsz) || !req.imgsz.is_multiple_of(32) {
        return bad("imgsz: must be a multiple of 32 in 64..=1280");
    }
    if !(1..=64).contains(&req.batch) {
        return bad("batch: must be in 1..=64");
    }
    if !BUDGET_S.contains(&req.budget_s) {
        return bad("budget_s: must be in 1..=10");
    }
    Ok(())
}

fn classes() -> Vec<String> {
    ["circle", "square", "triangle"].map(String::from).to_vec()
}

/// Training: `(img/s, batches measured)`.
fn train(req: &BenchmarkRequest, device: Device, budget: Duration) -> Result<(f64, u64)> {
    let data_err = |e: folo_data::DataError| Error::internal(format!("synthetic data: {e}"));
    let syn = SyntheticConfig {
        images: IMAGES,
        classes: classes(),
        imgsz: req.imgsz,
        val_fraction: 0.0,
        ..SyntheticConfig::default()
    };
    let data = TrainData {
        train: Arc::new(SyntheticDataset::new(syn.clone(), Split::Train).map_err(data_err)?),
        val: Arc::new(SyntheticDataset::new(syn, Split::Val).map_err(data_err)?),
        classes: classes(),
    };
    let cfg = folo_vision::TrainConfig {
        name: "benchmark".into(),
        arch: req.arch.as_str().into(),
        epochs: 1,
        batch: req.batch as usize,
        imgsz: req.imgsz,
        device: device.to_string(),
        amp: req.amp,
        ..folo_vision::TrainConfig::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    // (arrival, global step) of every step event.
    let marks: Arc<Mutex<Vec<(Instant, u64, f32)>>> = Arc::default();
    let (sink_stop, sink_marks) = (stop.clone(), marks.clone());
    let events = move |ev: &RunEvent| {
        if let RunEvent::Step(s) = ev {
            let now = Instant::now();
            let mut m = sink_marks.lock().unwrap();
            m.push((now, s.step, s.imgs_per_s));
            if now.duration_since(m[0].0) >= budget || s.step + 1 >= s.steps_per_epoch {
                sink_stop.store(true, Ordering::Relaxed);
            }
        }
    };
    let opts = TrainOptions {
        events: Some(Box::new(events)),
        step_interval: Duration::ZERO,
        stop,
        ..TrainOptions::default()
    };
    let mut trainer = Trainer::new(cfg, data, opts).map_err(engine_error)?;
    trainer.run().map_err(engine_error)?;
    drop(trainer);
    let marks = marks.lock().unwrap();
    let (Some(first), Some(last)) = (marks.first(), marks.last()) else {
        return Err(Error::internal("the probe trained no batch"));
    };
    let warm = first.0 + last.0.duration_since(first.0).mul_f64(WARMUP_SHARE);
    let from = marks.iter().find(|m| m.0 >= warm).filter(|m| m.1 < last.1).unwrap_or(first);
    if from.1 >= last.1 {
        // A single batch: its own rate (warm-up included).
        return Ok((last.2 as f64, 1));
    }
    let images = (last.1 - from.1) as f64 * req.batch as f64;
    Ok((images / last.0.duration_since(from.0).as_secs_f64().max(1e-9), last.1 - from.1))
}

/// Inference: `(total ms, forward ms)` per measured call, and the precision it ran in.
fn infer(
    req: &BenchmarkRequest,
    device: Device,
    precision: Precision,
    budget: Duration,
) -> Result<(Vec<(f64, f64)>, InferPrecision)> {
    let scale = req.arch.as_str().trim_start_matches("folo-");
    let mut model =
        FoloModel::new(ModelConfig::new(scale, classes().len()).map_err(nn_error)?).map_err(nn_error)?;
    model.set_imgsz(req.imgsz);
    let model = place_model(model, device).map_err(engine_error)?;
    let cfg = PredictConfig { imgsz: req.imgsz, rect: true, ..PredictConfig::default() };
    let p = Predictor::with_precision(model, cfg, precision).map_err(engine_error)?;
    let syn =
        SyntheticConfig { images: 1, classes: classes(), imgsz: req.imgsz, ..SyntheticConfig::default() };
    let img = folo_data::synthetic::generate(&syn, 0)
        .map_err(|e| Error::internal(format!("synthetic data: {e}")))?;
    let jpeg = folo_data::encode(&img.image, ImageFormat::Jpeg(90))
        .map_err(|e| Error::internal(format!("encoding the probe image: {e}")))?;
    let call = || -> Result<(f64, f64)> {
        let t0 = Instant::now();
        let img = folo_data::decode(&jpeg)
            .map_err(|e| Error::internal(format!("decoding the probe image: {e}")))?;
        let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t = p.predict(&img).map_err(engine_error)?.timing.to_api();
        Ok((t.total_ms + decode_ms, t.inference_ms))
    };
    for _ in 0..3 {
        call()?;
    }
    let t0 = Instant::now();
    let mut out = Vec::new();
    while (t0.elapsed() < budget || out.len() < 5) && out.len() < MAX_PREDICTIONS {
        out.push(call()?);
    }
    let precision = if p.precision() == Precision::F16 { InferPrecision::F16 } else { InferPrecision::F32 };
    Ok((out, precision))
}

/// Runs the probe on `device` (training, then inference at `precision`) and records it in `throughput`. The caller
/// holds the runs' probe gate.
pub fn run(
    req: &BenchmarkRequest,
    device: Device,
    precision: Precision,
    throughput: &Throughput,
) -> Result<BenchmarkReport> {
    check(req)?;
    let t0 = Instant::now();
    let budget = Duration::from_secs_f64(req.budget_s);
    let gpu = folo_gpu::context_for(device);
    if let Some(c) = &gpu {
        c.reset_peak_memory();
    }
    let (ips, batches) = train(req, device, budget.mul_f64(TRAIN_SHARE))?;
    let peak = gpu.as_ref().map(|c| c.pool_stats().peak_live_bytes);
    let id = engine::device_id(device);
    let amp = req.amp && !device.is_cpu();
    let trained = throughput.record_train(TrainSample {
        key: TrainKey { device: id.clone(), arch: req.arch, imgsz: req.imgsz, batch: req.batch, amp },
        phase: PhaseKind::Mosaic,
        train_imgs_per_s: Some(ips),
        val_imgs_per_s: None,
        startup_s: None,
        peak_vram_bytes: peak,
        probe: true,
    });
    let (calls, precision) = infer(req, device, precision, budget.mul_f64(1.0 - TRAIN_SHARE))?;
    let key = InferKey {
        device: id.clone(),
        arch: req.arch,
        imgsz: req.imgsz,
        precision,
        tensor_cores: engine::tensor_cores_on(device),
    };
    let mut inferred = None;
    for &(total_ms, inference_ms) in calls.iter().rev().take(WINDOW).rev() {
        inferred = Some(throughput.record_infer(InferSample { key: key.clone(), total_ms, inference_ms }));
    }
    throughput.flush();
    let inferred = inferred.expect("at least 5 predictions");
    let mut totals: Vec<f64> = calls.iter().map(|c| c.0).collect();
    totals.sort_by(f64::total_cmp);
    Ok(BenchmarkReport {
        device: id.device,
        device_name: id.device_name,
        train_imgs_per_s: (ips * 10.0).round() / 10.0,
        train_batches: batches,
        latency_ms: (totals[totals.len() / 2] * 100.0).round() / 100.0,
        predictions: calls.len() as u64,
        elapsed_s: (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
        training: train_row(&trained),
        inference: infer_row(&inferred),
    })
}
