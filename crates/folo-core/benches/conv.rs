//! conv2d throughput at YOLO-typical shapes: `cargo bench -p folo-core --bench conv`.

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Result, Tensor, no_grad};

fn bench(budget: Duration, mut f: impl FnMut()) -> Duration {
    f();
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

struct Case {
    name: &'static str,
    n: usize,
    c: usize,
    o: usize,
    hw: usize,
    k: usize,
    stride: usize,
    groups: usize,
}

fn main() -> Result<()> {
    let cases = [
        Case { name: "3x3 s1", n: 8, c: 64, o: 128, hw: 80, k: 3, stride: 1, groups: 1 },
        Case { name: "3x3 s2", n: 8, c: 64, o: 128, hw: 80, k: 3, stride: 2, groups: 1 },
        Case { name: "1x1", n: 8, c: 64, o: 128, hw: 80, k: 1, stride: 1, groups: 1 },
        Case { name: "3x3 s1 deep", n: 8, c: 256, o: 256, hw: 20, k: 3, stride: 1, groups: 1 },
        Case { name: "dw 3x3", n: 8, c: 128, o: 128, hw: 80, k: 3, stride: 1, groups: 128 },
    ];
    println!("conv2d, {} threads (NCHW f32, padding k/2)", rayon::current_num_threads());
    println!(
        "{:<12} {:>26} {:>10} {:>9} {:>14} {:>9}",
        "case", "shape", "fwd ms", "GFLOPS", "fwd+bwd ms", "GFLOPS"
    );
    let mut rng = Rng::seed(2);
    for c in &cases {
        let x = Tensor::randn([c.n, c.c, c.hw, c.hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([c.o, c.c / c.groups, c.k, c.k], 0.0, 0.1, &mut rng);
        let p = c.k / 2;
        let oh = (c.hw + 2 * p - c.k) / c.stride + 1;
        let flops = 2.0 * (c.n * c.o * oh * oh * (c.c / c.groups) * c.k * c.k) as f64;
        let budget = Duration::from_millis(1500);
        let fwd = {
            let _g = no_grad();
            bench(budget, || {
                std::hint::black_box(x.conv2d(&w, None, c.stride, p, 1, c.groups).unwrap());
            })
        };
        let xr = x.detach().requires_grad_();
        let wr = w.detach().requires_grad_();
        let both = bench(budget, || {
            let y = xr.conv2d(&wr, None, c.stride, p, 1, c.groups).unwrap();
            std::hint::black_box(y.sum_all().unwrap().backward().unwrap());
        });
        let shape = format!("{}x{}x{}x{}->{}", c.n, c.c, c.hw, c.hw, c.o);
        println!(
            "{:<12} {:>26} {:>10.2} {:>9.1} {:>14.2} {:>9.1}",
            c.name,
            shape,
            fwd.as_secs_f64() * 1e3,
            flops / fwd.as_secs_f64() / 1e9,
            both.as_secs_f64() * 1e3,
            3.0 * flops / both.as_secs_f64() / 1e9
        );
    }
    Ok(())
}
