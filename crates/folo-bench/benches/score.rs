//! Scoring timings on 500 images, 10 classes, about 10 true boxes and 30 detections per image. Run with
//! `cargo bench -p folo-bench --bench score`.
//!
//! - `evaluate`: the whole grading of the unchanged images (AP, CocoEval sizes, the 200-resample bootstrap,
//!   calibration, localization, self-labeling);
//! - `map`: one variant's mAP pair;
//! - `outcome` of every image (the failure gallery).

use folo_api::{BBox, BenchDetection};
use folo_bench::score::{self, Sample};
use folo_core::rng::Rng;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn bench<T>(name: &str, iters: usize, mut f: impl FnMut() -> T) {
    black_box(f()); // warm-up
    let mut times: Vec<Duration> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed()
        })
        .collect();
    times.sort();
    let (med, min) = (times[times.len() / 2], times[0]);
    println!("{name:<28} median {:>9.2} ms   min {:>9.2} ms   ({iters} iters)", ms(med), ms(min));
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// A detector-like run: each truth found with probability 0.8 (jittered box, high-ish score), plus false
/// positives with low-ish scores, about 30 detections per image.
fn samples(n: usize, classes: u32, rng: &mut Rng) -> Vec<Sample> {
    (0..n)
        .map(|_| {
            let truths: Vec<BBox> = (0..10)
                .map(|_| {
                    let (x, y) = (rng.uniform(0.0, 560.0), rng.uniform(0.0, 400.0));
                    let (w, h) = (rng.uniform(8.0, 80.0), rng.uniform(8.0, 80.0));
                    BBox { class: rng.below(classes as u64) as u32, x1: x, y1: y, x2: x + w, y2: y + h }
                })
                .collect();
            let mut detections = Vec::new();
            for t in &truths {
                for k in 0..2 {
                    if rng.bernoulli(0.8) {
                        let score = rng.uniform(0.3, 1.0) / (1 + k) as f32;
                        let (w, h) = (t.x2 - t.x1, t.y2 - t.y1);
                        let mut j = |v: f32, s: f32| v + s * rng.uniform(-0.15, 0.15);
                        detections.push(BenchDetection {
                            class: t.class,
                            score,
                            x1: j(t.x1, w),
                            y1: j(t.y1, h),
                            x2: j(t.x2, w),
                            y2: j(t.y2, h),
                        });
                    }
                }
            }
            while detections.len() < 30 {
                let (x, y) = (rng.uniform(0.0, 560.0), rng.uniform(0.0, 400.0));
                detections.push(BenchDetection {
                    class: rng.below(classes as u64) as u32,
                    score: rng.uniform(0.0, 0.6),
                    x1: x,
                    y1: y,
                    x2: x + rng.uniform(8.0, 80.0),
                    y2: y + rng.uniform(8.0, 80.0),
                });
            }
            Sample { truths, detections }
        })
        .collect()
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "folo-bench score bench, 500 images x 10 truths x 30 detections, 10 classes, {threads} threads available"
    );
    let s = samples(500, 10, &mut Rng::seed(5));
    let e = score::evaluate(&s, 10, true, 0);
    println!(
        "  map50-95 {:.3} [{:.3}, {:.3}]  self-labeling {:.3} at {:?}",
        e.accuracy.map50_95,
        e.accuracy.ci95[0],
        e.accuracy.ci95[1],
        e.self_labeling.recall,
        e.self_labeling.threshold
    );
    bench("evaluate (scored)", 7, || score::evaluate(&s, 10, true, 0));
    bench("evaluate (unscored)", 7, || score::evaluate(&s, 10, false, 0));
    bench("map (one variant)", 21, || score::map(&s, 10, true));
    let conf = e.accuracy.best_conf;
    bench("outcome x 500", 21, || s.iter().map(|x| score::outcome(x, 10, true, conf).wrong).sum::<u32>());
    let big = samples(5000, 80, &mut Rng::seed(6));
    bench("evaluate 5000 img, 80 cls", 3, || score::evaluate(&big, 80, true, 0));
}
