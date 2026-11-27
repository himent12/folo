//! int8 against `f32` CPU prediction end to end at batch 1 (docs/BENCHMARKS.md rows 7 and 8 with `--precision
//! int8`, docs/MODEL.md §9.6). Run with
//! `FOLO_BENCH_WEIGHTS=<folo quantize output> FOLO_BENCH_IMAGES=<image folder> cargo bench -p folo-vision --bench
//! int8_e2e`.
//!
//! Two predictors of the same weights, one `f32` and one with the file's int8 calibration, run in the same process
//! on the first `FOLO_BENCH_IMAGES_N` (default 16) images by name, cycling through them: each letterboxed to
//! 640×640 (row 8), then each resized to 1280×720 (rect, 640×384; row 7). Per row, `FOLO_BENCH_ROUNDS` (default 4)
//! rounds alternate the two sides, `FOLO_BENCH_ITERS` (default 100) calls per side and round: each side prints the
//! median of all its calls, the min–max of its per-round medians, and detections per image. `RAYON_NUM_THREADS`
//! pins the pool. `FOLO_BENCH_SIDE=int8` (or `f32`) runs only that side, for A/B runs against another binary.

use std::hint::black_box;
use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor, no_grad};
use folo_nn::io::ModelFile;
use folo_vision::model::quant::{self, Calibration, Int8Plan, Method, SensitivityPolicy};
use folo_vision::model::v2::{FoloV2Config, FoloV2Model};
use folo_vision::model::{Detector, FoloModel, ModelConfig};
use folo_vision::predict::{PredictConfig, Predictor};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn synth_image(w: u32, h: u32, seed: u64) -> folo_data::Image {
    let mut rng = Rng::seed(seed);
    let data = (0..w * h * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
    folo_data::Image::from_raw(w, h, data).expect("image")
}

fn time_ms<T>(iters: usize, mut f: impl FnMut() -> T) -> (f64, f64, f64) {
    for _ in 0..8 {
        black_box(f());
    }
    let mut t: Vec<f64> = (0..iters)
        .map(|_| {
            let s = Instant::now();
            black_box(f());
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    let med = median(&mut t);
    let p90 = t[(t.len() * 9 / 10).min(t.len() - 1)];
    (med, t[0], p90)
}

fn run_synthetic(iters: usize) {
    println!(
        "int8_e2e (synthetic, uses_vnni={}, {} threads, {iters} iters):",
        quant::uses_vnni(),
        rayon::current_num_threads()
    );
    let mut c = ModelConfig::new("n", 80).expect("config");
    c.bn_momentum = 0.999;
    let mut m_f32 = FoloModel::with_rng(c.clone(), &mut Rng::seed(1)).expect("model");
    let x_cal = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(2));
    {
        let _g = no_grad();
        m_f32.forward(&x_cal).expect("bn warmup");
    }
    m_f32.fuse().expect("fuse");
    let cal_raw =
        quant::calibrate(&m_f32, || std::iter::once(Ok(x_cal.clone())), Method::Percentile { p: 1e-4 })
            .expect("cal");
    let cal_sens = cal_raw.clone().with_sensitivity_f32(&m_f32, SensitivityPolicy::default()).expect("sens");

    let mut m_q8 = m_f32.fused_eval_clone().expect("clone");
    m_q8.set_int8(Some(cal_sens.clone())).expect("set_int8");

    let x1 = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(3));
    let plan = Int8Plan::new(cal_sens.clone());
    {
        let _g = no_grad();
        let (med, min, p90) = time_ms(iters, || m_f32.forward(&x1).expect("fwd"));
        println!("folo-n f32  forward 640 b1       median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let (med, min, p90) = time_ms(iters, || m_q8.forward(&x1).expect("fwd"));
        println!("folo-n int8 forward 640 b1       median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| quant::exec::features(&m_f32, &plan, &x1).expect("feats"))
        });
        println!("folo-n int8 features(f32 in) 640 median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let rgb0 = vec![0x0080_8080u32; 640 * 640 + 64];
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| {
                quant::exec::features_rgb0_u8(&m_f32, &plan, &rgb0, 1, 640, 640).expect("feats")
            })
        });
        println!("folo-n int8 features(u8  in) 640 median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let feats = folo_core::cpu::in_pool(|| quant::exec::features(&m_f32, &plan, &x1).expect("feats"));
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| m_f32.head.forward([&feats[0], &feats[1], &feats[2]]).expect("head"))
        });
        println!("folo-n f32  head(dense) 640      median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
    }

    let img640 = synth_image(640, 640, 10);
    let img720 = synth_image(1280, 720, 11);
    let p_f32 = Predictor::new(m_f32, PredictConfig::default()).expect("pred");
    let p_q8 = Predictor::new(m_q8, PredictConfig::default()).expect("pred");
    for (label, img) in [("640x640", &img640), ("1280x720", &img720)] {
        let s_f32 = p_f32.predict(img).expect("predict");
        let (med, min, p90) = time_ms(iters, || p_f32.predict(img).expect("predict"));
        println!(
            "folo-n f32  predict {label:<12} median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}  (pre {:.2} inf {:.2} post {:.2} det {})",
            s_f32.timing.preprocess_ms,
            s_f32.timing.inference_ms,
            s_f32.timing.postprocess_ms,
            s_f32.detections.len()
        );
        let s_q8 = p_q8.predict(img).expect("predict");
        let (med, min, p90) = time_ms(iters, || p_q8.predict(img).expect("predict"));
        println!(
            "folo-n int8 predict {label:<12} median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}  (pre {:.2} inf {:.2} post {:.2} det {})",
            s_q8.timing.preprocess_ms,
            s_q8.timing.inference_ms,
            s_q8.timing.postprocess_ms,
            s_q8.detections.len()
        );
    }

    let mut c2 = FoloV2Config::new("n", 80).expect("v2 cfg");
    c2.bn_momentum = 0.999;
    let mut v2_f32 = FoloV2Model::with_rng(c2, &mut Rng::seed(1)).expect("v2");
    {
        let _g = no_grad();
        v2_f32.forward(&x_cal).expect("v2 warmup");
    }
    v2_f32.fuse().expect("v2 fuse");
    let v2_cal_raw =
        quant::calibrate_v2(&v2_f32, || std::iter::once(Ok(x_cal.clone())), Method::Percentile { p: 1e-4 })
            .expect("v2 cal");
    let v2_cal_sens =
        v2_cal_raw.clone().with_sensitivity_f32_v2(&v2_f32, SensitivityPolicy::default()).expect("v2 sens");
    let mut v2_q8 = v2_f32.fused_eval_clone().expect("v2 clone");
    v2_q8.set_int8(Some(v2_cal_sens.clone())).expect("v2 set_int8");
    let v2_plan = Int8Plan::new(v2_cal_sens);
    {
        let _g = no_grad();
        let (med, min, p90) = time_ms(iters, || v2_f32.forward(&x1).expect("v2 fwd"));
        println!("folo2-n f32  forward 640 b1      median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let (med, min, p90) = time_ms(iters, || v2_q8.forward(&x1).expect("v2 q8 fwd"));
        println!("folo2-n int8 forward 640 b1      median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| quant::exec::features_v2(&v2_f32, &v2_plan, &x1).expect("v2 feats"))
        });
        println!("folo2-n int8 features(f32 in)640 median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let rgb0 = vec![0x0080_8080u32; 640 * 640 + 64];
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| {
                quant::exec::features_v2_rgb0_u8(&v2_f32, &v2_plan, &rgb0, 1, 640, 640).expect("v2 feats")
            })
        });
        println!("folo2-n int8 features(u8  in)640 median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
        let v2_feats =
            folo_core::cpu::in_pool(|| quant::exec::features_v2(&v2_f32, &v2_plan, &x1).expect("v2 feats"));
        let (med, min, p90) = time_ms(iters, || {
            folo_core::cpu::in_pool(|| {
                v2_f32.head.forward([&v2_feats[0], &v2_feats[1], &v2_feats[2]]).expect("v2 head")
            })
        });
        println!("folo2-n f32  head(dense) 640     median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}");
    }
    let p_v2_f32 = Predictor::new_v2(v2_f32, PredictConfig::default()).expect("pred v2");
    let p_v2_q8 = Predictor::new_v2(v2_q8, PredictConfig::default()).expect("pred v2 q8");
    for (label, img) in [("640x640", &img640), ("1280x720", &img720)] {
        let s_f32 = p_v2_f32.predict(img).expect("predict");
        let (med, min, p90) = time_ms(iters, || p_v2_f32.predict(img).expect("predict"));
        println!(
            "folo2-n f32  predict {label:<12}median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}  (pre {:.2} inf {:.2} post {:.2} det {})",
            s_f32.timing.preprocess_ms,
            s_f32.timing.inference_ms,
            s_f32.timing.postprocess_ms,
            s_f32.detections.len()
        );
        let s_q8 = p_v2_q8.predict(img).expect("predict");
        let (med, min, p90) = time_ms(iters, || p_v2_q8.predict(img).expect("predict"));
        println!(
            "folo2-n int8 predict {label:<12}median {med:>6.2} ms  min {min:>6.2}  p90 {p90:>6.2}  (pre {:.2} inf {:.2} post {:.2} det {})",
            s_q8.timing.preprocess_ms,
            s_q8.timing.inference_ms,
            s_q8.timing.postprocess_ms,
            s_q8.detections.len()
        );
    }
}

fn main() {
    let (Ok(weights), Ok(dir)) = (std::env::var("FOLO_BENCH_WEIGHTS"), std::env::var("FOLO_BENCH_IMAGES"))
    else {
        run_synthetic(env("FOLO_BENCH_ITERS", 80));
        return;
    };
    let (n, iters, rounds): (usize, usize, usize) =
        (env("FOLO_BENCH_IMAGES_N", 16), env("FOLO_BENCH_ITERS", 100), env("FOLO_BENCH_ROUNDS", 4));
    let only = std::env::var("FOLO_BENCH_SIDE").ok();
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("FOLO_BENCH_IMAGES")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("png"))
        })
        .collect();
    paths.sort();
    paths.truncate(n);
    let imgs: Vec<folo_data::Image> =
        paths.iter().map(|p| folo_data::read_image(p).expect("image")).collect();
    assert!(!imgs.is_empty(), "no images in {dir}");
    let hd: Vec<folo_data::Image> = imgs.iter().map(|i| folo_data::resize::resize(i, 1280, 720)).collect();

    let file = ModelFile::open(&weights).expect("weights");
    let cal = Calibration::from_metadata(file.metadata()).expect("folo.quant").expect("no int8 calibration");
    let open = |int8: bool| {
        let mut m = Detector::from_file_for_inference(&file, Device::Cpu, DType::F32).expect("weights");
        if int8 {
            m.set_int8(Some(cal.clone())).expect("int8");
        }
        Predictor::from_detector(m, PredictConfig::default()).expect("predictor")
    };
    let mut sides: Vec<(&str, Predictor)> = [("f32", false), ("int8", true)]
        .into_iter()
        .filter(|(name, _)| only.as_deref().is_none_or(|o| o == *name))
        .map(|(name, int8)| (name, open(int8)))
        .collect();
    println!(
        "int8_e2e: {weights}, {} images from {dir}, {} threads, {rounds} rounds x {iters} calls per side",
        imgs.len(),
        rayon::current_num_threads()
    );
    for (row, set, rect) in
        [("row 8: 640x640 (square)", &imgs, false), ("row 7: 1280x720 (rect 640x384)", &hd, true)]
    {
        let mut calls: Vec<Vec<f64>> = vec![Vec::new(); sides.len()];
        let mut rounds_med: Vec<Vec<f64>> = vec![Vec::new(); sides.len()];
        let mut dets = vec![0usize; sides.len()];
        for (_, p) in sides.iter_mut() {
            p.set_config(PredictConfig { rect, ..PredictConfig::default() }).expect("config");
            for img in set.iter() {
                black_box(p.predict(img).expect("predict"));
            }
        }
        for _ in 0..rounds {
            for (s, (_, p)) in sides.iter().enumerate() {
                let mut t = Vec::with_capacity(iters);
                for i in 0..iters {
                    let start = Instant::now();
                    let r = black_box(p.predict(&set[i % set.len()]).expect("predict"));
                    t.push(start.elapsed().as_secs_f64() * 1e3);
                    dets[s] += r.detections.len();
                }
                calls[s].extend_from_slice(&t);
                rounds_med[s].push(median(&mut t));
            }
        }
        for (s, (name, _)) in sides.iter().enumerate() {
            let lo = rounds_med[s].iter().copied().fold(f64::INFINITY, f64::min);
            let hi = rounds_med[s].iter().copied().fold(0.0, f64::max);
            println!(
                "{row:<32} {name:<4} median {:>7.2} ms   rounds {lo:>6.2}-{hi:<6.2}   {:.1} det/img",
                median(&mut calls[s]),
                dets[s] as f64 / (rounds * iters) as f64
            );
        }
    }
}
