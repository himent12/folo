//! Per-core conv2d throughput on the folo-n inference convs (640×640, batch 1, fused bias + SiLU, as
//! `FoloModel::fuse` runs them): `cargo bench -p folo-core --bench conv_infer`.
//!
//! Every distinct conv of the folo-n forward is timed on its own (median of repeated calls), weighted by how many
//! times the forward runs it, and grouped by kernel path (stem, k×k stride 1, k×k stride 2, 1×1, depthwise). The
//! GF/s/thread column is the throughput divided by the rayon pool size, so a 1-thread run is per-core efficiency;
//! the min column (the fastest call) is the steadier figure on a busy machine.
//!
//! `RAYON_NUM_THREADS` sets the pool; `FOLO_BENCH_BUDGET_MS` (default 150) the time per shape; an argument filters
//! shapes by group name (`stem`, `k3s1`, `k3s2`, `1x1`, `dw`).

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Conv2dParams, Tensor, UnaryOp, no_grad};

/// `(group, c_in, c_out, k, stride, input h = w, groups, calls per forward)`.
type Shape = (&'static str, usize, usize, usize, usize, usize, usize, usize);

#[rustfmt::skip]
const SHAPES: &[Shape] = &[
    ("stem", 3, 16, 3, 2, 640, 1, 1),
    ("k3s2", 16, 32, 3, 2, 320, 1, 1),
    ("k3s2", 64, 64, 3, 2, 160, 1, 1),
    ("k3s2", 128, 128, 3, 2, 80, 1, 1),
    ("k3s2", 64, 64, 3, 2, 80, 1, 1),
    ("k3s2", 128, 256, 3, 2, 40, 1, 1),
    ("k3s2", 128, 128, 3, 2, 40, 1, 1),
    ("k3s1", 8, 16, 3, 1, 160, 1, 1),
    ("k3s1", 16, 8, 3, 1, 160, 1, 1),
    ("k3s1", 16, 32, 3, 1, 80, 1, 2),
    ("k3s1", 32, 16, 3, 1, 80, 1, 2),
    ("k3s1", 64, 64, 3, 1, 80, 1, 2),
    ("k3s1", 32, 32, 3, 1, 40, 1, 4),
    ("k3s1", 32, 64, 3, 1, 40, 1, 2),
    ("k3s1", 64, 32, 3, 1, 40, 1, 2),
    ("k3s1", 64, 64, 3, 1, 40, 1, 1),
    ("k3s1", 128, 64, 3, 1, 40, 1, 1),
    ("k3s1", 64, 64, 3, 1, 20, 1, 9),
    ("k3s1", 256, 64, 3, 1, 20, 1, 1),
    ("1x1", 32, 32, 1, 1, 160, 1, 1),
    ("1x1", 48, 64, 1, 1, 160, 1, 1),
    ("1x1", 64, 64, 1, 1, 80, 1, 2),
    ("1x1", 64, 80, 1, 1, 80, 1, 1),
    ("1x1", 80, 80, 1, 1, 80, 1, 2),
    ("1x1", 96, 64, 1, 1, 80, 1, 1),
    ("1x1", 96, 128, 1, 1, 80, 1, 1),
    ("1x1", 256, 64, 1, 1, 80, 1, 1),
    ("1x1", 64, 32, 1, 1, 40, 1, 2),
    ("1x1", 64, 64, 1, 1, 40, 1, 2),
    ("1x1", 80, 80, 1, 1, 40, 1, 2),
    ("1x1", 128, 80, 1, 1, 40, 1, 1),
    ("1x1", 128, 128, 1, 1, 40, 1, 1),
    ("1x1", 192, 128, 1, 1, 40, 1, 4),
    ("1x1", 384, 128, 1, 1, 40, 1, 1),
    ("1x1", 64, 64, 1, 1, 20, 1, 1),
    ("1x1", 80, 80, 1, 1, 20, 1, 2),
    ("1x1", 128, 64, 1, 1, 20, 1, 4),
    ("1x1", 128, 128, 1, 1, 20, 1, 3),
    ("1x1", 128, 256, 1, 1, 20, 1, 2),
    ("1x1", 256, 80, 1, 1, 20, 1, 1),
    ("1x1", 256, 128, 1, 1, 20, 1, 2),
    ("1x1", 256, 256, 1, 1, 20, 1, 3),
    ("1x1", 384, 256, 1, 1, 20, 1, 3),
    ("1x1", 512, 256, 1, 1, 20, 1, 1),
    ("dw", 64, 64, 3, 1, 80, 64, 1),
    ("dw", 80, 80, 3, 1, 80, 80, 1),
    ("dw", 80, 80, 3, 1, 40, 80, 1),
    ("dw", 128, 128, 3, 1, 40, 128, 1),
    ("dw", 80, 80, 3, 1, 20, 80, 1),
    ("dw", 128, 128, 3, 1, 20, 128, 1),
    ("dw", 256, 256, 3, 1, 20, 256, 1),
];

/// `(median, min)` seconds per call.
fn time(budget: Duration, mut f: impl FnMut()) -> (f64, f64) {
    for _ in 0..3 {
        f();
    }
    let mut t = Vec::new();
    let start = Instant::now();
    while t.len() < 7 || start.elapsed() < budget {
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    (t[t.len() / 2], t[0])
}

fn main() {
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
    let budget_ms = std::env::var("FOLO_BENCH_BUDGET_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(150);
    let budget = Duration::from_millis(budget_ms);
    let threads = rayon::current_num_threads();
    println!("conv_infer: folo-n inference convs, {threads} threads, fused bias + SiLU, medians");
    println!(
        "{:<6} {:>28} {:>6} {:>9} {:>9} {:>10} {:>9}",
        "group", "shape", "calls", "ms", "GF/s", "GF/s/thr", "min ms"
    );
    let _g = no_grad();
    let mut rng = Rng::seed(3);
    let mut groups: Vec<(&str, f64, f64)> = Vec::new();
    for &(group, c, o, k, s, hw, gr, calls) in SHAPES {
        if filter.as_ref().is_some_and(|f| f != group) {
            continue;
        }
        let x = Tensor::randn([1, c, hw, hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([o, c / gr, k, k], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let p = Conv2dParams { stride: [s, s], padding: [k / 2, k / 2], dilation: [1, 1], groups: gr };
        let oh = (hw + 2 * (k / 2) - k) / s + 1;
        let flops = 2.0 * (o * oh * oh * (c / gr) * k * k) as f64;
        let (t, tmin) = time(budget, || {
            std::hint::black_box(x.conv2d_act(&w, Some(&b), &p, Some(UnaryOp::Silu)).unwrap());
        });
        let gfs = flops / t / 1e9;
        let shape = format!("{c}->{o} k{k} s{s} {hw}x{hw}{}", if gr > 1 { " dw" } else { "" });
        println!(
            "{group:<6} {shape:>28} {calls:>6} {:>9.3} {gfs:>9.1} {:>10.1} {:>9.3}",
            t * 1e3,
            gfs / threads as f64,
            tmin * 1e3
        );
        match groups.iter_mut().find(|g| g.0 == group) {
            Some(g) => {
                g.1 += t * calls as f64;
                g.2 += flops * calls as f64;
            }
            None => groups.push((group, t * calls as f64, flops * calls as f64)),
        }
    }
    println!("\nper forward (weighted by calls):");
    let (mut tt, mut tf) = (0.0, 0.0);
    for (g, t, f) in &groups {
        println!(
            "{g:<6} {:>9.3} ms {:>9.1} GF/s {:>8.1} GF/s/thr",
            t * 1e3,
            f / t / 1e9,
            f / t / 1e9 / threads as f64
        );
        tt += t;
        tf += f;
    }
    println!(
        "{:<6} {:>9.3} ms {:>9.1} GF/s {:>8.1} GF/s/thr ({:.2} GFLOP)",
        "all",
        tt * 1e3,
        tf / tt / 1e9,
        tf / tt / 1e9 / threads as f64,
        tf / 1e9
    );
}
