//! SAHI tiled inference and Test-Time Augmentation (TTA) benchmark on CPU (int8) and GPU (f16).
//!
//! Run with:
//! `cargo bench -p folo-vision --bench sahi_tta`
//!
//! Environment overrides:
//! - `FOLO_BENCH_ITERS` (default `12` for GPU, `6` for CPU)
//! - `FOLO_BENCH_MODE` (`all` default, `gpu`, or `cpu`)

use std::hint::black_box;
use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_data::Image;
use folo_gpu::GpuContext;
use folo_vision::model::quant::{self, Method, SensitivityPolicy};
use folo_vision::model::v2::{FoloV2Config, FoloV2Model};
use folo_vision::model::{Detector, FoloModel, ModelConfig};
use folo_vision::predict::{Precision, PredictConfig, Predictor};
use folo_vision::{TileConfig, TtaConfig};

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn synth_image(w: u32, h: u32, seed: u64) -> Image {
    let mut rng = Rng::seed(seed);
    let data = (0..w * h * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
    Image::from_raw(w, h, data).expect("synth image")
}

fn time_ms<T>(warmup: usize, iters: usize, mut f: impl FnMut() -> T) -> (f64, f64, T) {
    for _ in 0..warmup {
        black_box(f());
    }
    let mut times = Vec::with_capacity(iters);
    let mut last = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = f();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        last = Some(r);
    }
    let med = median(&mut times);
    let min = times[0];
    (med, min, last.expect("iters >= 1"))
}

fn build_v1_cpu_int8() -> Predictor {
    let mut c = ModelConfig::new("n", 80).expect("v1 cfg");
    c.bn_momentum = 0.999;
    let mut m = FoloModel::with_rng(c, &mut Rng::seed(1)).expect("v1 model");
    let x_cal = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(2));
    {
        let _g = no_grad();
        m.forward(&x_cal).expect("warmup");
    }
    m.fuse().expect("fuse");
    let cal = quant::calibrate(&m, || std::iter::once(Ok(x_cal.clone())), Method::Percentile { p: 1e-4 })
        .expect("cal")
        .with_sensitivity_f32(&m, SensitivityPolicy::default())
        .expect("sens");
    m.set_int8(Some(cal)).expect("set_int8");
    Predictor::new(m, PredictConfig::default()).expect("predictor")
}

fn build_v1_gpu(dev: Device) -> Predictor {
    let mut c = ModelConfig::new("n", 80).expect("v1 cfg");
    c.bn_momentum = 0.999;
    let mut m = FoloModel::with_rng(c, &mut Rng::seed(1)).expect("v1 model");
    let x_cal = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(2));
    {
        let _g = no_grad();
        m.forward(&x_cal).expect("warmup");
    }
    m.fuse().expect("fuse");
    m.to_device(dev).expect("to_device");
    Predictor::with_precision(m, PredictConfig::default(), Precision::F16).expect("predictor")
}

fn build_v2_gpu(dev: Device) -> Predictor {
    let mut c = FoloV2Config::new("n", 80).expect("v2 cfg");
    c.bn_momentum = 0.999;
    let mut m = FoloV2Model::with_rng(c, &mut Rng::seed(1)).expect("v2 model");
    let x_cal = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(2));
    {
        let _g = no_grad();
        m.forward(&x_cal).expect("warmup");
    }
    m.fuse().expect("fuse");
    m.to_device(dev).expect("to_device");
    Predictor::from_detector_with_precision(Detector::from(m), PredictConfig::default(), Precision::F16)
        .expect("predictor")
}

fn bench_predictor(
    label: &str,
    pred: &Predictor,
    img_hd: &Image,
    batch_sq: &[Image],
    batch_hd: &[Image],
    iters: usize,
) {
    let tile_cfg = TileConfig {
        tile_width: 640,
        tile_height: 640,
        overlap_x: 0.2,
        overlap_y: 0.2,
        full_image: true,
        tile_batch: 8,
        ..Default::default()
    };
    let tile_cfg_no_full = TileConfig { full_image: false, ..tile_cfg.clone() };
    let tta_cfg = TtaConfig::default();

    let (med, min, p) = time_ms(2, iters, || pred.predict_tiled(img_hd, &tile_cfg).expect("tiled"));
    println!(
        "{label:<18} sahi 1x1920x1080 (full)  median {med:>7.2} ms  min {min:>7.2}  (pre {:.2} inf {:.2} post {:.2})",
        p.timing.preprocess_ms, p.timing.inference_ms, p.timing.postprocess_ms
    );

    let (med, min, ps) =
        time_ms(2, iters, || pred.predict_batch_tiled(batch_sq, &tile_cfg).expect("batch tiled sq"));
    println!(
        "{label:<18} sahi 4x1280x1280 (full)  median {med:>7.2} ms  min {min:>7.2}  (per-img pre {:.2} inf {:.2} post {:.2})",
        ps[0].timing.preprocess_ms, ps[0].timing.inference_ms, ps[0].timing.postprocess_ms
    );

    let (med, min, ps) =
        time_ms(2, iters, || pred.predict_batch_tiled(batch_sq, &tile_cfg_no_full).expect("batch tiled nf"));
    println!(
        "{label:<18} sahi 4x1280x1280 (tiles) median {med:>7.2} ms  min {min:>7.2}  (per-img pre {:.2} inf {:.2} post {:.2})",
        ps[0].timing.preprocess_ms, ps[0].timing.inference_ms, ps[0].timing.postprocess_ms
    );

    let (med, min, p) = time_ms(2, iters, || pred.predict_tta(img_hd, &tta_cfg).expect("tta"));
    println!(
        "{label:<18} tta  1x1920x1080 (3s+hf) median {med:>7.2} ms  min {min:>7.2}  (pre {:.2} inf {:.2} post {:.2})",
        p.timing.preprocess_ms, p.timing.inference_ms, p.timing.postprocess_ms
    );

    let (med, min, ps) = time_ms(2, iters, || pred.predict_batch_tta(batch_hd, &tta_cfg).expect("batch tta"));
    println!(
        "{label:<18} tta  4x1920x1080 (3s+hf) median {med:>7.2} ms  min {min:>7.2}  (per-img pre {:.2} inf {:.2} post {:.2})",
        ps[0].timing.preprocess_ms, ps[0].timing.inference_ms, ps[0].timing.postprocess_ms
    );
}

fn main() {
    let mode = std::env::var("FOLO_BENCH_MODE").unwrap_or_else(|_| "all".into());
    let img_hd = synth_image(1920, 1080, 10);
    let batch_sq: Vec<Image> = (0..4).map(|i| synth_image(1280, 1280, 20 + i)).collect();
    let batch_hd: Vec<Image> = (0..4).map(|i| synth_image(1920, 1080, 30 + i)).collect();

    if mode == "all" || mode == "cpu" {
        let iters = env("FOLO_BENCH_ITERS", 5usize);
        println!("sahi_tta CPU int8 ({} threads, {iters} iters):", rayon::current_num_threads());
        let p_cpu = build_v1_cpu_int8();
        bench_predictor("folo-n cpu-int8", &p_cpu, &img_hd, &batch_sq, &batch_hd, iters);
    }

    if mode == "all" || mode == "gpu" {
        let iters = env("FOLO_BENCH_ITERS", 10usize);
        if let Ok(ctx) = GpuContext::new() {
            let dev = Device::Gpu(0);
            folo_gpu::register(ctx, dev).expect("register gpu");
            println!("sahi_tta GPU f16 ({iters} iters):");
            let p_v1 = build_v1_gpu(dev);
            bench_predictor("folo-n gpu-f16", &p_v1, &img_hd, &batch_sq, &batch_hd, iters);
            let p_v2 = build_v2_gpu(dev);
            bench_predictor("folo2-n gpu-f16", &p_v2, &img_hd, &batch_sq, &batch_hd, iters);
        } else {
            println!("sahi_tta GPU: skipped (no GPU adapter)");
        }
    }
}
