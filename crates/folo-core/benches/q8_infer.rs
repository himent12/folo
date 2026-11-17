//! int8 against `f32` on the folo-n inference convs (640×640, batch 1, SiLU): `cargo bench -p folo-core --bench
//! q8_infer`. The int8 conv reads u8 C4 activations and writes u8 C4 codes (dequantise, bias, SiLU, requantise in
//! its epilogue); the `f32` conv is the fused `conv2d_act` a forward runs. Each shape alternates the two, `ROUNDS`
//! times, and reports the median of the per-round medians and the ratio. Calls run inside folo-core's CPU pool.
//!
//! `RAYON_NUM_THREADS` sets the pool; `FOLO_BENCH_BUDGET_MS` (default 100) the time per shape, side and round; an
//! argument filters shapes by group name (`k3s1`, `k3s2`, `1x1`) or by a shape string such as `48->64`.

use std::time::{Duration, Instant};

use folo_core::cpu::q8::{self, C4, Grid, QConv, QOut};
use folo_core::rng::Rng;
use folo_core::{Conv2dParams, Tensor, UnaryOp, no_grad};

/// `(group, c_in, c_out, k, stride, input h = w, calls per forward)`.
type Shape = (&'static str, usize, usize, usize, usize, usize, usize);

#[rustfmt::skip]
const SHAPES: &[Shape] = &[
    ("k3s2", 16, 32, 3, 2, 320, 1),
    ("k3s2", 64, 64, 3, 2, 160, 1),
    ("k3s2", 128, 128, 3, 2, 80, 1),
    ("k3s2", 64, 64, 3, 2, 80, 1),
    ("k3s2", 128, 256, 3, 2, 40, 1),
    ("k3s2", 128, 128, 3, 2, 40, 1),
    ("k3s1", 8, 16, 3, 1, 160, 1),
    ("k3s1", 16, 8, 3, 1, 160, 1),
    ("k3s1", 16, 32, 3, 1, 80, 2),
    ("k3s1", 32, 16, 3, 1, 80, 2),
    ("k3s1", 64, 64, 3, 1, 80, 2),
    ("k3s1", 32, 32, 3, 1, 40, 4),
    ("k3s1", 32, 64, 3, 1, 40, 2),
    ("k3s1", 64, 32, 3, 1, 40, 2),
    ("k3s1", 64, 64, 3, 1, 40, 1),
    ("k3s1", 128, 64, 3, 1, 40, 1),
    ("k3s1", 64, 64, 3, 1, 20, 9),
    ("k3s1", 256, 64, 3, 1, 20, 1),
    ("1x1", 32, 32, 1, 1, 160, 1),
    ("1x1", 48, 64, 1, 1, 160, 1),
    ("1x1", 64, 64, 1, 1, 80, 2),
    ("1x1", 96, 64, 1, 1, 80, 1),
    ("1x1", 96, 128, 1, 1, 80, 1),
    ("1x1", 64, 32, 1, 1, 40, 2),
    ("1x1", 64, 64, 1, 1, 40, 2),
    ("1x1", 128, 128, 1, 1, 40, 1),
    ("1x1", 192, 128, 1, 1, 40, 4),
    ("1x1", 384, 128, 1, 1, 40, 1),
    ("1x1", 64, 64, 1, 1, 20, 1),
    ("1x1", 128, 64, 1, 1, 20, 4),
    ("1x1", 128, 128, 1, 1, 20, 3),
    ("1x1", 128, 256, 1, 1, 20, 2),
    ("1x1", 256, 128, 1, 1, 20, 2),
    ("1x1", 256, 256, 1, 1, 20, 3),
    ("1x1", 384, 256, 1, 1, 20, 3),
    ("1x1", 512, 256, 1, 1, 20, 1),
];

const ROUNDS: usize = 3;

/// Median seconds per call.
fn time(budget: Duration, mut f: impl FnMut()) -> f64 {
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
    t[t.len() / 2]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    folo_core::cpu::in_pool(run);
}

fn run() {
    let filter = std::env::args().skip(1).find(|a| !a.starts_with('-'));
    let budget_ms = std::env::var("FOLO_BENCH_BUDGET_MS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let budget = Duration::from_millis(budget_ms);
    let threads = rayon::current_num_threads();
    // `Q8_BENCH_ACT=none`: identity on both sides (the epilogue without SiLU).
    let silu = std::env::var("Q8_BENCH_ACT").map_or(true, |v| v != "none");
    let act = silu.then_some(UnaryOp::Silu);
    println!(
        "q8_infer: folo-n convs, {threads} threads, {}, int8 {} vs f32, medians of {ROUNDS} alternated rounds",
        if silu { "SiLU" } else { "identity" },
        if q8::uses_vnni() { "AVX-VNNI" } else { "portable" }
    );
    println!("{:<6} {:>24} {:>6} {:>9} {:>9} {:>7}", "group", "shape", "calls", "f32 ms", "int8 ms", "ratio");
    let _g = no_grad();
    let mut rng = Rng::seed(3);
    let (mut tf, mut tq) = (0.0, 0.0);
    for &(group, c, o, k, s, hw, calls) in SHAPES {
        let shape = format!("{c}->{o} k{k} s{s} {hw}x{hw}");
        if filter.as_ref().is_some_and(|f| f != group && !shape.starts_with(f.as_str())) {
            continue;
        }
        let x = Tensor::randn([1, c, hw, hw], 0.0, 1.0, &mut rng);
        let w = Tensor::randn([o, c, k, k], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let p = Conv2dParams { stride: [s, s], padding: [k / 2, k / 2], dilation: [1, 1], groups: 1 };
        let gin = Grid::from_range(-0.3, 4.0);
        let gout = Grid::from_range(-0.3, 6.0);
        let mut xq = q8::alloc(1, c, hw * hw);
        q8::quantize(&x.to_vec::<f32>().unwrap(), 1, c, hw * hw, gin, &mut xq);
        let cv = QConv::new(
            &w.to_vec::<f32>().unwrap(),
            &b.to_vec::<f32>().unwrap(),
            (o, c, k),
            s,
            k / 2,
            silu,
            &vec![gin; c],
        );
        let (oh, ow) = cv.out_hw(hw, hw);
        let mut out = q8::alloc(1, o, oh * ow);
        let view = C4 { data: &xq, offset: 0, img_stride: c / 4 * hw * hw, c, zero: gin.zero };
        let (mut f, mut q) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            f.push(time(budget, || {
                std::hint::black_box(x.conv2d_act(&w, Some(&b), &p, act).unwrap());
            }));
            q.push(time(budget, || {
                q8::conv(&[view], 1, hw, hw, &cv, None, QOut::U8 { data: &mut out, grid: gout });
                std::hint::black_box(&out);
            }));
        }
        let (f, q) = (median(f), median(q));
        tf += f * calls as f64;
        tq += q * calls as f64;
        println!("{group:<6} {shape:>24} {calls:>6} {:>9.3} {:>9.3} {:>7.2}", f * 1e3, q * 1e3, f / q);
    }
    println!("all, weighted by calls: f32 {:.3} ms, int8 {:.3} ms, ratio {:.2}", tf * 1e3, tq * 1e3, tf / tq);
}
