//! Conv weight-gradient benchmark: 3x3, stride 1, padding 1 convs of folo-n's training shapes at batch 16 and 64,
//! the gradient of the weight alone (the input does not require grad). Run with
//! `cargo bench -p folo-core --bench conv_wgrad` (`FOLO_BENCH_ITERS`, default 10, timed backward passes per shape;
//! `RAYON_NUM_THREADS` pins the pool). Prints the median and the fastest backward pass per shape.

use std::time::Instant;

use folo_core::Tensor;
use folo_core::rng::Rng;

fn main() {
    let iters: usize = std::env::var("FOLO_BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    println!("folo-core conv weight-gradient bench, {} rayon threads", rayon::current_num_threads());
    // (batch, channels in = out, size)
    for (n, c, hw) in [(16usize, 32usize, 160usize), (64, 32, 160), (64, 64, 80)] {
        let rng = &mut Rng::seed(1);
        let x = Tensor::randn([n, c, hw, hw], 0.0, 1.0, rng);
        let w = Tensor::randn([c, c, 3, 3], 0.0, 0.1, rng).requires_grad_();
        let r = Tensor::randn([n, c, hw, hw], 0.0, 1.0, rng);
        let mut times = Vec::with_capacity(iters);
        for i in 0..iters + 2 {
            let loss = x.conv2d(&w, None, 1, 1, 1, 1).unwrap().mul(&r).unwrap().sum_all().unwrap();
            let t = Instant::now();
            let g = loss.backward().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            drop(g);
            if i >= 2 {
                times.push(ms);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "wgrad 3x3 c{c} {hw}x{hw} b{n}: median {:.1} ms, min {:.1} ms",
            times[times.len() / 2],
            times[0]
        );
    }
}
