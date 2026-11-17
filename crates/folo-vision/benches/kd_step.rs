//! The distillation term of one COCO training batch on the GPU (docs/MODEL.md §8.9): the tensor path
//! (`TeacherView` + `DistillTargets` + `distill_loss`, as the trainer ran it before the fused kernel) against the
//! fused kernel (`distill_loss_fused`), forward, readback and backward, alternated round by round in one process.
//! Run with `cargo bench -p folo-vision --bench kd_step` (`FOLO_BENCH_ROUNDS`, default 5, rounds of 30 calls per
//! path). With `FOLO_GPU_PROFILE=1` it also prints the GPU
//! time per kernel of one call of each path (slower end to end: every dispatch is timed).
//!
//! Inputs: folo-n's one-to-many head at 640 px, batch 16, 80 classes, 8400 anchors, `reg_max` 16, random logits;
//! the teacher's outputs in f16 (the f16 teacher of a mixed-precision run), the student's f32 (its f32 head copies).

use std::time::{Duration, Instant};

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{DType, Device, Tensor};
use folo_vision::loss::distill::distill_loss_fused;
use folo_vision::loss::{DistillConfig, DistillLoss, DistillTargets, TeacherView, distill_loss};

const B: usize = 16;
const NC: usize = 80;
const A: usize = 8400;
const R: usize = 16;

fn median(mut t: Vec<Duration>) -> f64 {
    t.sort();
    t[t.len() / 2].as_secs_f64() * 1e3
}

/// One call: the term, its readback (inside both paths) and the backward, synchronised.
fn call(dev: Device, fused: bool, s: &[Tensor; 2], t: &[Tensor; 2], cfg: &DistillConfig) -> f32 {
    let (z, x) = (s[0].detach().requires_grad_(), s[1].detach().requires_grad_());
    let out: DistillLoss = if fused {
        distill_loss_fused(&z, &x, &t[0], &t[1], 120.0, cfg).expect("fused")
    } else {
        let view = TeacherView::new(&t[0], &t[1], R, cfg.temperature).expect("view");
        let targets = DistillTargets::new(&view).expect("targets");
        distill_loss(&z, &x, &targets, 120.0, cfg).expect("tensor path")
    };
    let g = out.total.backward().expect("backward");
    drop(g);
    synchronize(dev).expect("sync");
    out.cls + out.dfl
}

fn main() {
    let dev = folo_gpu::init().expect("a GPU");
    let ctx = folo_gpu::context_for(dev).expect("context");
    let rounds: usize = std::env::var("FOLO_BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let cfg = DistillConfig::new("t");
    println!("kd_step: {} ({}), B {B}, nc {NC}, A {A}, reg_max {R}", ctx.summary().name, dev);
    let rng = &mut Rng::seed(0);
    let on = |t: Tensor, dt: DType| t.to_device(dev).expect("upload").to_dtype(dt).expect("dtype");
    let s = [
        on(Tensor::randn([B, 4 * R, A], 0.0, 2.0, rng), DType::F32),
        on(Tensor::rand_uniform([B, NC, A], -9.0, 2.0, rng), DType::F32),
    ];
    let t = [
        on(Tensor::randn([B, 4 * R, A], 0.0, 2.0, rng), DType::F16),
        on(Tensor::rand_uniform([B, NC, A], -9.0, 2.0, rng), DType::F16),
    ];
    for fused in [false, true] {
        for _ in 0..5 {
            call(dev, fused, &s, &t, &cfg);
        }
    }
    let (a, b) = (call(dev, false, &s, &t, &cfg), call(dev, true, &s, &t, &cfg));
    println!("reported terms: tensor path {a}, fused {b}");
    if ctx.profiling() {
        for fused in [false, true] {
            ctx.profile_take();
            call(dev, fused, &s, &t, &cfg);
            let rows = ctx.profile_take();
            let total: f64 = rows.iter().map(|r| r.ms).sum();
            println!(
                "{}: GPU {total:.3} ms in {} commands",
                if fused { "fused" } else { "tensor path" },
                rows.iter().map(|r| r.count).sum::<u64>()
            );
            for r in rows.iter().take(12) {
                println!("  {:>8.3} ms {:>4}x {}", r.ms, r.count, r.label);
            }
        }
        return;
    }
    let (mut tp, mut fu) = (Vec::new(), Vec::new());
    for round in 0..rounds {
        for fused in [false, true] {
            let mut times = Vec::new();
            for _ in 0..30 {
                let t0 = Instant::now();
                call(dev, fused, &s, &t, &cfg);
                times.push(t0.elapsed());
            }
            let m = median(times);
            println!(
                "round {round} {}: median {m:.3} ms per call",
                if fused { "fused" } else { "tensor path" }
            );
            if fused { fu.push(m) } else { tp.push(m) }
        }
    }
    let stats = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(f64::total_cmp);
        (s[s.len() / 2], s[0], s[s.len() - 1])
    };
    let ((m0, lo0, hi0), (m1, lo1, hi1)) = (stats(&tp), stats(&fu));
    println!("tensor path: median of round medians {m0:.3} ms (min {lo0:.3}, max {hi0:.3})");
    println!("fused:       median of round medians {m1:.3} ms (min {lo1:.3}, max {hi1:.3}); {:.2}x", m0 / m1);
}
