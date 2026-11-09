//! CPU predictor end to end at batch 1 (docs/BENCHMARKS.md row 8). Run with
//! `cargo bench -p folo-vision --bench cpu_e2e`.
//!
//! folo-n (80 classes, untrained, seed 0), f32, predict defaults: `Predictor::predict` on a decoded `u8` RGB image
//! (letterbox, forward, decode, NMS, boxes back in image pixels), like Ultralytics' CPU `e2e` rows, which feed
//! 640×640 images. A 1280×720 frame (resize + padding to 640×384) is timed too, and the 640×640 image again with
//! 20 ms of idle time before each call (a 30 fps camera: the rayon workers have gone to sleep). Each row prints the
//! median of the predictor's own preprocess / inference / postprocess split next to the wall time, and the fused
//! forward alone on a ready input for reference.
//!
//! `FOLO_BENCH_ITERS` (default 100) sets the calls per row; `RAYON_NUM_THREADS` pins the pool.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{PredictConfig, Predictor};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn image(w: u32, h: u32, seed: u64) -> folo_data::Image {
    let mut rng = Rng::seed(seed);
    let data = (0..w * h * 3).map(|_| (rng.uniform(0.0, 256.0) as u32).min(255) as u8).collect();
    folo_data::Image::from_raw(w, h, data).expect("image")
}

fn main() {
    let iters: usize = std::env::var("FOLO_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    println!("folo-vision cpu_e2e bench, {} rayon threads, {iters} iters", rayon::current_num_threads());
    let model = || FoloModel::new(ModelConfig::new("n", 80).expect("config")).expect("model");

    let mut fwd = model();
    fwd.eval();
    fwd.fuse().expect("fuse");
    let x = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(0));
    {
        let _g = no_grad();
        for _ in 0..5 {
            black_box(fwd.forward(&x).expect("forward"));
        }
        let mut t: Vec<f64> = (0..iters)
            .map(|_| {
                let s = Instant::now();
                black_box(fwd.forward(&x).expect("forward"));
                ms(s.elapsed())
            })
            .collect();
        println!("{:<38} median {:>7.2} ms", "forward 640 b1 (fused)", median(&mut t));
    }
    drop(fwd);

    let pred = Predictor::new(model(), PredictConfig::default()).expect("predictor");
    let rows = [
        ("predict 640x640 u8", image(640, 640, 1), false),
        ("predict 1280x720 u8", image(1280, 720, 2), false),
        ("predict 640x640 u8, 20 ms idle first", image(640, 640, 1), true),
    ];
    for (name, img, idle) in rows {
        for _ in 0..5 {
            black_box(pred.predict(&img).expect("predict"));
        }
        let (mut wall, mut pre, mut inf, mut post) = (vec![], vec![], vec![], vec![]);
        for _ in 0..iters {
            if idle {
                std::thread::sleep(Duration::from_millis(20));
            }
            let s = Instant::now();
            let p = black_box(pred.predict(&img).expect("predict"));
            wall.push(ms(s.elapsed()));
            pre.push(p.timing.preprocess_ms);
            inf.push(p.timing.inference_ms);
            post.push(p.timing.postprocess_ms);
        }
        let mut sorted = wall.clone();
        sorted.sort_by(f64::total_cmp);
        let p90 = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
        println!(
            "{name:<38} median {:>7.2} ms   p90 {p90:>7.2}   min {:>7.2}   pre {:>5.2} / inf {:>6.2} / post {:>5.2}",
            median(&mut wall),
            sorted[0],
            median(&mut pre),
            median(&mut inf),
            median(&mut post),
        );
    }
}
