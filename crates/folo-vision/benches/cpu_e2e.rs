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
//! `FOLO_BENCH_SCALE` (default `n`) picks the model (`n`, `s`, `m`, ...), `FOLO_BENCH_BATCH` (default 0: off)
//! adds `predict_batch` rows of that many images (640×640, 1280×720 frames, and mixed shapes: 1280×720, 720×1280,
//! 640×480 and 640×640 in turn), and `FOLO_BENCH_LAYERS=1` prints the fastest and
//! median time of every layer of docs/MODEL.md §4 (the head is the forward minus the traced layers).
//!
//! Trained weights: with `FOLO_BENCH_WEIGHTS` (a `.folo` file, e.g. `folo-n-coco`) and `FOLO_BENCH_IMAGES` (a folder
//! of images, e.g. COCO val2017; the first `FOLO_BENCH_IMAGES_N`, default 16, by name) the same rows run on real
//! images with the trained model, cycling through the images: each letterboxed to 640×640, each resized to
//! 1280×720 (rect, 640×384), and `predict_batch` of `FOLO_BENCH_BATCH` of them, as they are (padded square) and
//! resized to 1280×720 (one rect canvas, 640×384). Untrained weights pass no candidate at `conf = 0.25`, so only
//! these rows show what candidate-first prediction (MODEL.md §9.5) does.
//! `FOLO_BENCH_UNTRAINED=0` skips the untrained rows. The 1280×720 row runs a second time after
//! `folo_core::cpu::clear_buffer_cache()`: the two read the same unless what the earlier rows left in the CPU
//! buffer cache slows the trained model (it did before the cache evicted stale lengths: docs/notes/hill/r3-cpufix.md).
//! Only the predictor's public API (and `folo_core::cpu`) is used, so the file builds against older trees for A/B
//! runs.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_vision::model::{FoloModel, LAYER_NAMES, ModelConfig};
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
    let scale = std::env::var("FOLO_BENCH_SCALE").unwrap_or_else(|_| "n".into());
    let batch: usize = std::env::var("FOLO_BENCH_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let layers = std::env::var("FOLO_BENCH_LAYERS").is_ok_and(|v| v == "1");
    println!(
        "folo-vision cpu_e2e bench, folo-{scale}, {} rayon threads, {iters} iters",
        rayon::current_num_threads()
    );
    let model = || FoloModel::new(ModelConfig::new(&scale, 80).expect("config")).expect("model");
    if std::env::var("FOLO_BENCH_UNTRAINED").is_ok_and(|v| v == "0") {
        trained(iters, batch);
        return;
    }

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
        let med = median(&mut t);
        println!(
            "{:<38} median {med:>7.2} ms   p10 {:>7.2}   min {:>7.2}",
            "forward 640 b1 (fused)",
            t[t.len() / 10],
            t[0]
        );
        if layers {
            layer_times(&fwd, &x, iters);
        }
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
    type Shape = fn(usize) -> (u32, u32);
    let sets: [(&str, Shape); 3] = [
        ("640x640", |_| (640, 640)),
        ("1280x720", |_| (1280, 720)),
        ("mixed", |i| [(1280, 720), (720, 1280), (640, 480), (640, 640)][i % 4]),
    ];
    for (shape, wh) in sets.into_iter().filter(|_| batch > 0) {
        let imgs: Vec<folo_data::Image> = (0..batch)
            .map(|i| {
                let (w, h) = wh(i);
                image(w, h, 10 + i as u64)
            })
            .collect();
        for _ in 0..3 {
            black_box(pred.predict_batch(&imgs).expect("predict"));
        }
        let mut wall: Vec<f64> = (0..iters.div_ceil(4))
            .map(|_| {
                let s = Instant::now();
                black_box(pred.predict_batch(&imgs).expect("predict"));
                ms(s.elapsed())
            })
            .collect();
        let med = median(&mut wall);
        println!(
            "{:<38} median {med:>7.2} ms   min {:>7.2}   {:>6.1} img/s (best {:.1})",
            format!("predict_batch {batch} x {shape} u8"),
            wall[0],
            batch as f64 * 1e3 / med,
            batch as f64 * 1e3 / wall[0],
        );
    }
    trained(iters, batch);
}

/// The rows on real images with trained weights (see the module docs).
fn trained(iters: usize, batch: usize) {
    let (Ok(weights), Ok(dir)) = (std::env::var("FOLO_BENCH_WEIGHTS"), std::env::var("FOLO_BENCH_IMAGES"))
    else {
        println!(
            "trained rows skipped: set FOLO_BENCH_WEIGHTS (a .folo file) and FOLO_BENCH_IMAGES (an image folder)"
        );
        return;
    };
    let n: usize = std::env::var("FOLO_BENCH_IMAGES_N").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
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
    let mut pred = Predictor::open(&weights, PredictConfig::default()).expect("weights");
    println!("trained: {weights}, {} images from {dir}", imgs.len());
    let rows = [
        ("trained 640x640 (square)", &imgs, false, false),
        ("trained 1280x720 (rect 640x384)", &hd, true, false),
        ("trained 1280x720, buffer cache cleared", &hd, true, true),
    ];
    for (name, set, rect, clear) in rows {
        pred.set_config(PredictConfig { rect, ..PredictConfig::default() }).expect("config");
        if clear {
            folo_core::cpu::clear_buffer_cache();
        }
        for img in set.iter() {
            black_box(pred.predict(img).expect("predict"));
        }
        let (mut wall, mut pre, mut inf, mut post, mut dets) = (vec![], vec![], vec![], vec![], 0usize);
        for i in 0..iters {
            let s = Instant::now();
            let p = black_box(pred.predict(&set[i % set.len()]).expect("predict"));
            wall.push(ms(s.elapsed()));
            pre.push(p.timing.preprocess_ms);
            inf.push(p.timing.inference_ms);
            post.push(p.timing.postprocess_ms);
            dets += p.detections.len();
        }
        let mut sorted = wall.clone();
        sorted.sort_by(f64::total_cmp);
        let p90 = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
        println!(
            "{name:<38} median {:>7.2} ms   p90 {p90:>7.2}   min {:>7.2}   pre {:>5.2} / inf {:>6.2} / post {:>5.2}   {:.1} det/img",
            median(&mut wall),
            sorted[0],
            median(&mut pre),
            median(&mut inf),
            median(&mut post),
            dets as f64 / iters as f64,
        );
    }
    for (shape, src) in [("640x640", &imgs), ("1280x720", &hd)].into_iter().filter(|_| batch > 0) {
        let set: Vec<folo_data::Image> = (0..batch).map(|i| src[i % src.len()].clone()).collect();
        for _ in 0..3 {
            black_box(pred.predict_batch(&set).expect("predict"));
        }
        let mut wall: Vec<f64> = (0..iters.div_ceil(4))
            .map(|_| {
                let s = Instant::now();
                black_box(pred.predict_batch(&set).expect("predict"));
                ms(s.elapsed())
            })
            .collect();
        let med = median(&mut wall);
        println!(
            "{:<38} median {med:>7.2} ms   min {:>7.2}   {:>6.1} img/s (best {:.1})",
            format!("trained predict_batch {batch} x {shape}"),
            wall[0],
            batch as f64 * 1e3 / med,
            batch as f64 * 1e3 / wall[0],
        );
    }
}

/// Fastest and median time of every traced layer over `iters` forwards (a layer's time runs from the previous
/// layer's output to its own), and of the head (from the last traced layer to the end).
fn layer_times(model: &FoloModel, x: &Tensor, iters: usize) {
    let n = LAYER_NAMES.len();
    let mut t: Vec<Vec<f64>> = vec![Vec::with_capacity(iters); n];
    for _ in 0..iters {
        let mut last = Instant::now();
        let mut seen = 0;
        let out = model.forward_traced(x, &mut |i, _| {
            let now = Instant::now();
            t[i].push(ms(now - last));
            last = now;
            seen = i;
        });
        black_box(out.expect("forward"));
        t[n - 1].push(ms(last.elapsed()));
        assert!(seen < n - 1);
    }
    let (mut sum_min, mut sum_med) = (0.0, 0.0);
    for (i, v) in t.iter_mut().enumerate() {
        if v.is_empty() {
            continue;
        }
        let med = median(v);
        sum_min += v[0];
        sum_med += med;
        let name = if LAYER_NAMES[i].is_empty() { "(upsample / concat)" } else { LAYER_NAMES[i] };
        println!("  layer {i:>2} {name:<22} min {:>6.3} ms   median {med:>6.3} ms", v[0]);
    }
    println!("  layers total                    min {sum_min:>6.3} ms   median {sum_med:>6.3} ms");
}
