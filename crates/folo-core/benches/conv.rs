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
    let args: Vec<String> = std::env::args().collect();
    let filter = args.iter().skip(1).find(|a| !a.starts_with('-')).cloned();
    #[rustfmt::skip]
    let cases = [
        Case { name: "3x3 s1", n: 8, c: 64, o: 128, hw: 80, k: 3, stride: 1, groups: 1 },
        Case { name: "3x3 s2", n: 8, c: 64, o: 128, hw: 80, k: 3, stride: 2, groups: 1 },
        Case { name: "1x1", n: 8, c: 64, o: 128, hw: 80, k: 1, stride: 1, groups: 1 },
        Case { name: "3x3 s1 deep", n: 8, c: 256, o: 256, hw: 20, k: 3, stride: 1, groups: 1 },
        Case { name: "dw 3x3", n: 8, c: 128, o: 128, hw: 80, k: 3, stride: 1, groups: 128 },
        // folo-n training shapes at 320 px, batch 8.
        Case { name: "n stem", n: 8, c: 3, o: 16, hw: 320, k: 3, stride: 2, groups: 1 },
        Case { name: "n down2", n: 8, c: 16, o: 32, hw: 160, k: 3, stride: 2, groups: 1 },
        Case { name: "n b3x3 80", n: 8, c: 16, o: 8, hw: 80, k: 3, stride: 1, groups: 1 },
        Case { name: "n b3x3 40", n: 8, c: 64, o: 64, hw: 40, k: 3, stride: 1, groups: 1 },
        Case { name: "n b3x3 10", n: 8, c: 64, o: 64, hw: 10, k: 3, stride: 1, groups: 1 },
        Case { name: "n 1x1 80", n: 8, c: 48, o: 64, hw: 80, k: 1, stride: 1, groups: 1 },
        Case { name: "n 1x1 10", n: 8, c: 384, o: 256, hw: 10, k: 1, stride: 1, groups: 1 },
        Case { name: "n dw 40", n: 8, c: 64, o: 64, hw: 40, k: 3, stride: 1, groups: 64 },
        // folo-n inference shapes at 640 px, batch 1.
        Case { name: "i stem", n: 1, c: 3, o: 16, hw: 640, k: 3, stride: 2, groups: 1 },
        Case { name: "i b3x3 80", n: 1, c: 64, o: 64, hw: 80, k: 3, stride: 1, groups: 1 },
        Case { name: "i b3x3 20", n: 1, c: 128, o: 128, hw: 20, k: 3, stride: 1, groups: 1 },
        Case { name: "i 1x1 160", n: 1, c: 32, o: 64, hw: 160, k: 1, stride: 1, groups: 1 },
        Case { name: "i 1x1 20", n: 1, c: 384, o: 256, hw: 20, k: 1, stride: 1, groups: 1 },
    ];
    println!(
        "conv2d, {} threads (NCHW f32, padding k/2), best times in ms (GFLOP/s)",
        rayon::current_num_threads()
    );
    println!("{:<12} {:>26} {:>16} {:>16} {:>16}", "case", "shape", "forward", "grad input", "grad weight");
    let mut rng = Rng::seed(2);
    for c in cases.iter().filter(|c| filter.as_ref().is_none_or(|f| c.name.contains(f.as_str()))) {
        let x = Tensor::randn([c.n, c.c, c.hw, c.hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([c.o, c.c / c.groups, c.k, c.k], 0.0, 0.1, &mut rng);
        let p = c.k / 2;
        let oh = (c.hw + 2 * p - c.k) / c.stride + 1;
        let flops = 2.0 * (c.n * c.o * oh * oh * (c.c / c.groups) * c.k * c.k) as f64;
        let budget = Duration::from_millis(1000);
        let fwd = {
            let _g = no_grad();
            bench(budget, || {
                std::hint::black_box(x.conv2d(&w, None, c.stride, p, 1, c.groups).unwrap());
            })
        };
        let gy = Tensor::randn([c.n, c.o, oh, oh], 0.0, 1.0, &mut rng);
        // Backward of one input at a time, on a retained graph (backward can run repeatedly).
        let back = |xr: &Tensor, wr: &Tensor| {
            let y = xr.conv2d(wr, None, c.stride, p, 1, c.groups).unwrap();
            bench(budget, || {
                std::hint::black_box(y.backward_with(&gy).unwrap());
            })
        };
        let dx = back(&x.detach().requires_grad_(), &w);
        let dw = back(&x, &w.detach().requires_grad_());
        let shape = format!("{}x{}x{}x{}->{} k{}", c.n, c.c, c.hw, c.hw, c.o, c.k);
        let cell =
            |d: Duration| format!("{:.2} ({:.0})", d.as_secs_f64() * 1e3, flops / d.as_secs_f64() / 1e9);
        println!("{:<12} {:>26} {:>16} {:>16} {:>16}", c.name, shape, cell(fwd), cell(dx), cell(dw));
    }
    Ok(())
}
