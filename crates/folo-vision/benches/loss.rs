//! Detection-loss benchmark: B = 16 at 640x640 (8400 anchors), 80 classes, 8 gts per image, on the CPU.
//! Run with `cargo bench -p folo-vision --bench loss`.
//!
//! - `assign_batch` alone (host decode + TAL for the whole batch);
//! - `detection_loss` forward (assign + the differentiable terms);
//! - forward + `backward()`.

#[path = "../tests/common/mod.rs"]
mod common;

use common::Rng;
use folo_core::Tensor;
use folo_vision::decode::make_anchors;
use folo_vision::loss::{ImageTargets, LossConfig, assign_batch, detection_loss};
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
    println!("{name:<52} median {:>9.3} ms   min {:>9.3} ms   ({iters} iters)", ms(med), ms(min));
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("folo-vision loss bench, {threads} threads available");
    let mut rng = Rng::new(0x1055);
    let (b, nc, reg_max, gts) = (16usize, 80usize, 16usize, 8usize);
    let anchors = make_anchors(640, 640, &[8, 16, 32], 0.5);
    let a = anchors.len();

    // Head outputs at their initial priors (§6.5) plus noise, like the first steps of training.
    let reg: Vec<f32> = (0..b * 4 * reg_max * a).map(|_| 2.0 + rng.range(-1.0, 1.0)).collect();
    let cls: Vec<f32> = (0..b * nc * a).map(|_| rng.range(-12.0, -6.0)).collect();
    let box_raw = Tensor::from_vec(reg, [b, 4 * reg_max, a]).unwrap().requires_grad_();
    let cls_raw = Tensor::from_vec(cls, [b, nc, a]).unwrap().requires_grad_();
    let targets: Vec<ImageTargets> = (0..b)
        .map(|_| ImageTargets {
            boxes: (0..gts).map(|_| rng.bbox(640.0, 12.0, 320.0)).collect(),
            labels: (0..gts).map(|_| rng.below(nc) as u32).collect(),
        })
        .collect();
    let cfg = LossConfig::default();

    let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
    println!("B {b}, A {a}, nc {nc}: {} foreground anchors, items {:?}", out.num_fg, out.items);

    bench("assign_batch (decode + TAL, 16 images)", 20, || {
        assign_batch(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap()
    });
    bench("detection_loss forward", 20, || {
        detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap()
    });
    bench("detection_loss forward + backward", 20, || {
        let out = detection_loss(&box_raw, &cls_raw, &anchors, &targets, &cfg).unwrap();
        out.total.backward().unwrap()
    });
}
