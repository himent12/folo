//! Training-step benchmark: folo-n (4 classes) at 320×320, batch 16, on the CPU — the MODEL.md §10.6 smoke-test
//! configuration. One step is a train-mode forward, the detection loss, backward, and an AdamW update.
//! Run with `cargo bench -p folo-vision --bench train_step` (`FOLO_BENCH_SIZE`, default 320, sets the image size;
//! `FOLO_BENCH_BATCH`, default 16, the batch).

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::{Duration, Instant};

use common::Rng;
use folo_core::Tensor;
use folo_nn::Module;
use folo_nn::optim::{AdamW, GradAccumulator, Optimizer, StandardGroups};
use folo_vision::loss::{ImageTargets, LossConfig, detection_loss};
use folo_vision::model::{FoloModel, ModelConfig};

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let threads = rayon::current_num_threads();
    let s: usize = std::env::var("FOLO_BENCH_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(320);
    let b: usize = std::env::var("FOLO_BENCH_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let nc = 4usize;
    println!("folo-vision train-step bench, {threads} rayon threads, folo-n nc {nc}, {s}x{s}, batch {b}");
    let model = FoloModel::new(ModelConfig::new("n", nc).expect("config")).expect("model");
    let mut opt =
        AdamW::new(StandardGroups::from_module(&model).into_groups(0.00125, 0.9, 5e-4)).expect("opt");
    let mut acc = GradAccumulator::new(&model.parameters());
    let mut rng = Rng::new(7);
    let x: Vec<f32> = (0..b * 3 * s * s).map(|_| rng.f32()).collect();
    let x = Tensor::from_vec(x, [b, 3, s, s]).expect("input");
    let targets: Vec<ImageTargets> = (0..b)
        .map(|_| ImageTargets {
            boxes: (0..4).map(|_| rng.bbox(s as f32, 20.0, 110.0)).collect(),
            labels: (0..4).map(|_| rng.below(nc) as u32).collect(),
        })
        .collect();
    let cfg = LossConfig::default();
    let (mut fwd, mut bwd, mut stp) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..8 {
        let t0 = Instant::now();
        let out = model.forward(&x).expect("forward");
        let loss = detection_loss(&out.box_raw, &out.cls_raw, &out.anchors, &targets, &cfg).expect("loss");
        let t1 = Instant::now();
        let grads = loss.total.backward().expect("backward");
        acc.add(&grads).expect("acc");
        drop(grads);
        drop(loss);
        drop(out);
        let t2 = Instant::now();
        opt.step(&acc).expect("step");
        acc.zero();
        let t3 = Instant::now();
        if i >= 2 {
            fwd.push(t1 - t0);
            bwd.push(t2 - t1);
            stp.push(t3 - t2);
        }
    }
    let med = |v: &mut Vec<Duration>| {
        v.sort();
        v[v.len() / 2]
    };
    let (f, bw, st) = (med(&mut fwd), med(&mut bwd), med(&mut stp));
    let total = f + bw + st;
    println!("forward + loss {:>8.1} ms", ms(f));
    println!("backward       {:>8.1} ms", ms(bw));
    println!("AdamW step     {:>8.1} ms", ms(st));
    println!("total          {:>8.1} ms   = {:.1} imgs/s", ms(total), b as f64 / total.as_secs_f64());
}
