//! Pooling micro-benchmarks (`max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`):
//! `cargo bench -p folo-core --bench pool`.
//!
//! `FOLO_BENCH_BUDGET_MS` (default 80) sets the time budget per case.

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Pool2dParams, Tensor, no_grad};

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
    println!("pool bench ({threads} threads, median ms, budget {budget_ms} ms):");
    let mut rng = Rng::seed(42);

    // 1. max_pool2d_values (no_grad / inference): YOLO SPPF 5x5 s1, 2x2 s2, 3x3 s1
    for &(name, shape, k, s, pad) in &[
        ("max_val SPPF 5x5 s1 [1,256,20,20]", [1, 256, 20, 20], 5, 1, 2),
        ("max_val SPPF 5x5 s1 [8,256,20,20]", [8, 256, 20, 20], 5, 1, 2),
        ("max_val 2x2 s2 [4,64,80,80]", [4, 64, 80, 80], 2, 2, 0),
        ("max_val 3x3 s1 [4,64,80,80]", [4, 64, 80, 80], 3, 1, 1),
    ] {
        let x = Tensor::randn(shape, 0.0, 1.0, &mut rng);
        let p = Pool2dParams { kernel: [k, k], stride: [s, s], padding: [pad, pad] };
        let ms = {
            let _g = no_grad();
            bench_median(budget, || {
                std::hint::black_box(x.max_pool2d_with(&p).unwrap());
            })
        };
        println!("  {name:<42}: {ms:8.3} ms");
    }

    // 2. max_pool2d (with grad / argmax indices): training forward + backward
    for &(name, shape, k, s, pad) in &[
        ("max_idx SPPF 5x5 s1 [8,256,20,20]", [8, 256, 20, 20], 5, 1, 2),
        ("max_idx 2x2 s2 [4,64,80,80]", [4, 64, 80, 80], 2, 2, 0),
        ("max_idx 3x3 s1 [4,64,80,80]", [4, 64, 80, 80], 3, 1, 1),
    ] {
        let x = Tensor::randn(shape, 0.0, 1.0, &mut rng).requires_grad_();
        let p = Pool2dParams { kernel: [k, k], stride: [s, s], padding: [pad, pad] };
        let fwd_ms = bench_median(budget, || {
            std::hint::black_box(x.max_pool2d_with(&p).unwrap());
        });
        println!("  {name:<42}: {fwd_ms:8.3} ms");
    }

    // 3. avg_pool2d backward (3x3 s1)
    {
        let x = Tensor::randn([4, 64, 80, 80], 0.0, 1.0, &mut rng).requires_grad_();
        let y = x.avg_pool2d(3, 1, 1).unwrap();
        let g = Tensor::randn(y.dims(), 0.0, 1.0, &mut rng);
        let bwd_ms = bench_median(budget, || {
            std::hint::black_box(y.backward_with(&g).unwrap());
        });
        println!("  {:<42}: {bwd_ms:8.3} ms", "avg_bwd 3x3 s1 [4,64,80,80]");
    }

    // 4. adaptive_avg_pool2d (GAP 1x1 + 7x7, forward + backward)
    for &(name, shape, out_hw) in &[
        ("adap_avg GAP 1x1 [8,512,20,20]", [8, 512, 20, 20], [1, 1]),
        ("adap_avg 7x7 [8,256,40,40]", [8, 256, 40, 40], [7, 7]),
    ] {
        let x = Tensor::randn(shape, 0.0, 1.0, &mut rng).requires_grad_();
        let y = x.adaptive_avg_pool2d(out_hw).unwrap();
        let g = Tensor::randn(y.dims(), 0.0, 1.0, &mut rng);
        let fwd_ms = {
            let _g = no_grad();
            bench_median(budget, || {
                std::hint::black_box(x.adaptive_avg_pool2d(out_hw).unwrap());
            })
        };
        let bwd_ms = bench_median(budget, || {
            std::hint::black_box(y.backward_with(&g).unwrap());
        });
        println!("  {name:<42}: {fwd_ms:8.3} ms (fwd)  {bwd_ms:8.3} ms (bwd)");
    }
}
