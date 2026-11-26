//! Micro-benchmarks for `q8::quantize_weights`, `argmax`/`argmin`, column-tiled `softmax`/`log_softmax`
//! forward+backward, and `layer_norm`/`rms_norm` backward:
//! `cargo bench -p folo-core --bench norm_softmax_q8`.
//!
//! `FOLO_BENCH_BUDGET_MS` (default 80) sets the time budget per case.

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::cpu::q8::quantize_weights;
use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};

fn bench_median(budget: Duration, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < 7 || start.elapsed() < budget {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2] * 1e3
}

fn main() {
    folo_core::cpu::in_pool(run);
}

fn run() {
    let budget_ms = std::env::var("FOLO_BENCH_BUDGET_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let budget = Duration::from_millis(budget_ms);
    let threads = rayon::current_num_threads();
    println!("norm_softmax_q8 bench ({threads} threads, median ms, budget {budget_ms} ms):");
    let mut rng = Rng::seed(42);

    // 1. q8::quantize_weights (symmetric per-output-channel s8 quantization)
    for &(name, o, k) in &[
        ("q8::quantize_weights [256, 2304]", 256usize, 2304usize),
        ("q8::quantize_weights [512, 4608]", 512, 4608),
    ] {
        let w: Vec<f32> = (0..o * k).map(|_| rng.normal() * 0.25).collect();
        let ms = bench_median(budget, || {
            black_box(quantize_weights(&w, o));
        });
        println!("  {name:<44}: {ms:8.3} ms");
    }

    // 2. argmax / argmin (row inner==1 and column inner>1, including outer==1)
    for &(name, shape, dim) in &[
        ("argmax col [1, 80, 8400] dim=1", &[1usize, 80, 8400][..], 1isize),
        ("argmax col [4, 80, 8400] dim=1", &[4, 80, 8400][..], 1),
        ("argmax row [1024, 1024] dim=1", &[1024, 1024][..], 1),
        ("argmin col [4, 80, 8400] dim=1", &[4, 80, 8400][..], 1),
    ] {
        let x = Tensor::randn(shape, 0.0, 1.0, &mut rng);
        let is_min = name.starts_with("argmin");
        let ms = {
            let _g = no_grad();
            bench_median(budget, || {
                if is_min {
                    black_box(x.argmin(dim, false).unwrap());
                } else {
                    black_box(x.argmax(dim, false).unwrap());
                }
            })
        };
        println!("  {name:<44}: {ms:8.3} ms");
    }

    // 3. Column-tiled softmax & log_softmax forward + backward
    for &(name, shape, dim, log) in &[
        ("softmax col [4, 16, 8400] dim=1", &[4usize, 16, 8400][..], 1isize, false),
        ("log_softmax col [4, 16, 8400] dim=1", &[4, 16, 8400][..], 1, true),
        ("softmax col [2, 80, 8400] dim=1", &[2, 80, 8400][..], 1, false),
        ("log_softmax col [2, 80, 8400] dim=1", &[2, 80, 8400][..], 1, true),
    ] {
        let x = Tensor::randn(shape, 0.0, 2.0, &mut rng).requires_grad_();
        let y = if log { x.log_softmax(dim).unwrap() } else { x.softmax(dim).unwrap() };
        let g = Tensor::randn(y.dims(), 0.0, 1.0, &mut rng);
        let fwd_ms = {
            let _g = no_grad();
            bench_median(budget, || {
                if log {
                    black_box(x.log_softmax(dim).unwrap());
                } else {
                    black_box(x.softmax(dim).unwrap());
                }
            })
        };
        let bwd_ms = bench_median(budget, || {
            black_box(y.backward_with(&g).unwrap());
        });
        println!("  {name:<44}: {fwd_ms:8.3} ms (fwd)  {bwd_ms:8.3} ms (bwd)");
    }

    // 4. LayerNorm & RmsNorm backward (with weight & bias grads)
    for &(name, outer, d, is_rms) in &[
        ("layer_norm bwd [256, 768]", 256usize, 768usize, false),
        ("layer_norm bwd [1024, 512]", 1024, 512, false),
        ("rms_norm bwd [256, 768]", 256, 768, true),
        ("rms_norm bwd [1024, 512]", 1024, 512, true),
    ] {
        let x = Tensor::randn([outer, d], 0.0, 1.0, &mut rng).requires_grad_();
        let w = Tensor::randn([d], 1.0, 0.2, &mut rng).requires_grad_();
        let b = Tensor::randn([d], 0.0, 0.2, &mut rng).requires_grad_();
        let y = if is_rms {
            x.rms_norm(Some(&w), 1e-5).unwrap()
        } else {
            x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap()
        };
        let g = Tensor::randn([outer, d], 0.0, 1.0, &mut rng);
        let bwd_ms = bench_median(budget, || {
            black_box(y.backward_with(&g).unwrap());
        });
        println!("  {name:<44}: {bwd_ms:8.3} ms");
    }

    // 5. Top-k selection (detection & classification shapes)
    for &(name, rows, d, k, largest) in &[
        ("topk [64, 8400] k=100 max", 64usize, 8400usize, 100usize, true),
        ("topk [64, 8400] k=300 max", 64, 8400, 300, true),
        ("topk [1024, 1024] k=1 max", 1024, 1024, 1, true),
        ("topk [1024, 1024] k=32 max", 1024, 1024, 32, true),
        ("topk [64, 8400] k=100 min", 64, 8400, 100, false),
    ] {
        let x = Tensor::randn([rows, d], 0.0, 1.0, &mut rng);
        let ms = {
            let _g = no_grad();
            bench_median(budget, || {
                black_box(x.topk(k, largest).unwrap());
            })
        };
        println!("  {name:<44}: {ms:8.3} ms");
    }
}
