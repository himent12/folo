//! Micro-benchmark for elementwise activations, math ops, broadcasting binary ops, and `where_cond`.

use std::hint::black_box;
use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{DType, Tensor, no_grad};

fn bench(name: &str, elems: usize, iters: usize, mut f: impl FnMut() -> Tensor) {
    for _ in 0..5 {
        black_box(f());
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        black_box(f());
    }
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let g_elems = elems as f64 / dt / 1e9;
    println!("{name:<30} {:8.2} us  ({:6.2} Gelem/s)", dt * 1e6, g_elems);
}

fn main() {
    let _g = no_grad();
    let mut rng = Rng::seed(42);
    // Fits L2/L3 (262_144 elements = 1 MiB) and a larger 1M-element tensor.
    for &(label, rows, cols, iters) in &[("16k (serial)", 64, 256, 500), ("262k (parallel)", 256, 1024, 200)]
    {
        let n = rows * cols;
        println!("\n=== {label} [{rows} x {cols}] ===");
        let x = Tensor::randn([rows, cols], 0.0, 2.0, &mut rng);
        let xp = Tensor::rand_uniform([rows, cols], 0.1, 10.0, &mut rng);
        let y = Tensor::randn([rows, cols], 0.0, 2.0, &mut rng);
        let b_row = Tensor::randn([1, cols], 0.0, 2.0, &mut rng);
        let b_col = Tensor::randn([rows, 1], 0.0, 2.0, &mut rng);
        let cond_f = x.gt_scalar(0.0).unwrap();
        let cond_u = cond_f.to_dtype(DType::U8).unwrap();

        bench("relu", n, iters, || x.relu().unwrap());
        bench("leaky_relu", n, iters, || x.leaky_relu(0.1).unwrap());
        bench("hardswish", n, iters, || x.hardswish().unwrap());
        bench("clamp(-1, 1)", n, iters, || x.clamp(-1.0, 1.0).unwrap());
        bench("sqrt", n, iters, || xp.sqrt().unwrap());
        bench("rsqrt", n, iters, || xp.rsqrt().unwrap());
        bench("recip", n, iters, || xp.recip().unwrap());
        bench("floor", n, iters, || x.floor().unwrap());
        bench("ceil", n, iters, || x.ceil().unwrap());
        bench("round", n, iters, || x.round().unwrap());
        bench("sign", n, iters, || x.sign().unwrap());
        bench("add (contig)", n, iters, || x.add(&y).unwrap());
        bench("add (bc row)", n, iters, || x.add(&b_row).unwrap());
        bench("add (bc col)", n, iters, || x.add(&b_col).unwrap());
        bench("maximum (contig)", n, iters, || x.maximum(&y).unwrap());
        bench("where_cond (f32)", n, iters, || cond_f.where_cond(&x, &y).unwrap());
        bench("where_cond (u8)", n, iters, || cond_u.where_cond(&x, &y).unwrap());
    }
}
