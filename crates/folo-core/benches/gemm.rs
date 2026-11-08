//! SGEMM throughput: `cargo bench -p folo-core --bench gemm`.

use std::time::{Duration, Instant};

use folo_core::cpu::gemm::{sgemm, uses_avx2};
use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};

/// Run `f` repeatedly for about `budget` and return the best time per call.
fn bench(budget: Duration, mut f: impl FnMut()) -> Duration {
    f(); // warm up (thread pool, pack buffers, page faults)
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("sgemm, {threads} threads, avx2+fma kernel: {}", uses_avx2());
    println!("{:>6} {:>12} {:>12} {:>14}", "size", "best ms", "GFLOPS", "Tensor GFLOPS");
    let mut rng = Rng::seed(1);
    let _g = no_grad();
    for n in [256usize, 512, 1024, 2048] {
        let a = Tensor::randn([n, n], 0.0, 1.0, &mut rng);
        let b = Tensor::randn([n, n], 0.0, 1.0, &mut rng);
        let (av, bv) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        let mut c = vec![0f32; n * n];
        let flops = 2.0 * (n as f64).powi(3);
        let budget = Duration::from_millis(1500);
        let t = bench(budget, || sgemm(n, n, n, &av, n, 1, &bv, n, 1, &mut c, n));
        let tt = bench(budget, || {
            std::hint::black_box(a.matmul(&b).unwrap());
        });
        println!(
            "{n:>6} {:>12.3} {:>12.1} {:>14.1}",
            t.as_secs_f64() * 1e3,
            flops / t.as_secs_f64() / 1e9,
            flops / tt.as_secs_f64() / 1e9
        );
    }
}
