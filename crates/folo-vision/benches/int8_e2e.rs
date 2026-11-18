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

use folo_core::{DType, Device};
use folo_nn::io::ModelFile;
use folo_vision::model::FoloModel;
use folo_vision::model::quant::Calibration;
use folo_vision::predict::{PredictConfig, Predictor};

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let (Ok(weights), Ok(dir)) = (std::env::var("FOLO_BENCH_WEIGHTS"), std::env::var("FOLO_BENCH_IMAGES"))
    else {
        println!("int8_e2e skipped: set FOLO_BENCH_WEIGHTS (a `folo quantize` output) and FOLO_BENCH_IMAGES");
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
        let mut m = FoloModel::from_file_for_inference(&file, Device::Cpu, DType::F32).expect("weights");
        if int8 {
            m.set_int8(Some(cal.clone())).expect("int8");
        }
        Predictor::new(m, PredictConfig::default()).expect("predictor")
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
