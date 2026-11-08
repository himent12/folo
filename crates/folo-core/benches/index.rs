//! `index_select` / `gather` forward + backward: `cargo bench -p folo-core --bench index`.
//!
//! The backward passes scatter-add into a zeroed buffer of the input's size; the cases cover a few rows of a
//! big table, many rows with duplicates, and the loss's gathers along a short last dimension.

use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Result, Tensor};

fn bench(budget: Duration, mut f: impl FnMut()) -> Duration {
    f();
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut runs = 0;
    while runs < 5 || start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
        runs += 1;
    }
    best
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() -> Result<()> {
    let budget = Duration::from_millis(1000);
    let mut rng = Rng::seed(5);
    println!("index ops, {} threads, best times (forward, backward)", rayon::current_num_threads());
    // index_select along dim 0: (rows, cols, selected).
    for (rows, cols, sel) in
        [(200_000, 64, 256), (50_000, 64, 200_000), (8400, 144, 8400), (64, 8400 * 8, 64)]
    {
        let x = Tensor::randn([rows, cols], 0.0, 1.0, &mut rng).requires_grad_();
        let idx: Vec<i64> = (0..sel).map(|_| rng.below(rows as u64) as i64).collect();
        let idx = Tensor::from_vec(idx, [sel])?;
        let y = x.index_select(0, &idx)?;
        let g = Tensor::randn(y.dims(), 0.0, 1.0, &mut rng);
        let fwd = bench(budget, || {
            std::hint::black_box(x.detach().index_select(0, &idx).unwrap());
        });
        let bwd = bench(budget, || {
            std::hint::black_box(y.backward_with(&g).unwrap());
        });
        println!("  index_select [{rows}, {cols}] pick {sel:>6}: {:7.3} ms  {:7.3} ms", ms(fwd), ms(bwd));
    }
    // gather: (shape, dim, picked along dim).
    for (shape, dim, k) in
        [(vec![8, 8400, 80], 2, 1), (vec![8, 8400, 4, 16], 3, 2), (vec![4096, 2048], 0, 512)]
    {
        let x = Tensor::randn(shape.as_slice(), 0.0, 1.0, &mut rng).requires_grad_();
        let mut ishape = shape.clone();
        ishape[dim] = k;
        let n: usize = ishape.iter().product();
        let idx: Vec<i64> = (0..n).map(|_| rng.below(shape[dim] as u64) as i64).collect();
        let idx = Tensor::from_vec(idx, ishape.as_slice())?;
        let y = x.gather(dim, &idx)?;
        let g = Tensor::randn(y.dims(), 0.0, 1.0, &mut rng);
        let fwd = bench(budget, || {
            std::hint::black_box(x.detach().gather(dim, &idx).unwrap());
        });
        let bwd = bench(budget, || {
            std::hint::black_box(y.backward_with(&g).unwrap());
        });
        println!("  gather {shape:?} dim {dim} k {k}: {:7.3} ms  {:7.3} ms", ms(fwd), ms(bwd));
    }
    Ok(())
}
