//! Detector forward-pass benchmarks. Run with `cargo bench -p folo-vision --bench model`.
//!
//! folo-n (80 classes) on the CPU in f32, eval mode under `no_grad`, random input:
//! - forward at 640×640, batch 1 and 8, unfused (BatchNorm on running statistics) and fused (Conv+BN folded);
//! - `predict_raw` (fused forward + decode + NMS at predict settings), batch 1.
//!
//! Set `RAYON_NUM_THREADS` to pin the thread count (docs/BENCHMARKS.md compares against Ultralytics' best).

use std::hint::black_box;
use std::time::{Duration, Instant};

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::nms::NmsConfig;

fn bench<T>(name: &str, iters: usize, mut f: impl FnMut() -> T) {
    for _ in 0..3 {
        black_box(f()); // warm-up (buffer cache, page faults)
    }
    let mut times: Vec<Duration> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            black_box(f());
            t.elapsed()
        })
        .collect();
    times.sort();
    let med = times[times.len() / 2];
    let min = times[0];
    let p90 = times[(times.len() * 9 / 10).min(times.len() - 1)];
    println!(
        "{name:<44} median {:>8.2} ms   p90 {:>8.2} ms   min {:>8.2} ms   ({iters} iters)",
        ms(med),
        ms(p90),
        ms(min)
    );
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("folo-vision model bench, {threads} rayon threads");
    let mut model = FoloModel::new(ModelConfig::new("n", 80).expect("config")).expect("model");
    model.eval();
    let _g = no_grad();
    let mut rng = Rng::seed(0);
    let x1 = Tensor::rand_uniform([1, 3, 640, 640], 0.0, 1.0, &mut rng);
    let x8 = Tensor::rand_uniform([8, 3, 640, 640], 0.0, 1.0, &mut rng);

    bench("folo-n forward 640 b1 (unfused)", 30, || model.forward(&x1).expect("forward"));
    bench("folo-n forward 640 b8 (unfused)", 10, || model.forward(&x8).expect("forward"));
    model.fuse().expect("fuse");
    assert!(!model.is_training());
    bench("folo-n forward 640 b1 (fused)", 50, || model.forward(&x1).expect("forward"));
    bench("folo-n forward 640 b8 (fused)", 15, || model.forward(&x8).expect("forward"));
    let cfg = NmsConfig::default();
    bench("folo-n predict_raw 640 b1 (fused, decode+NMS)", 50, || {
        model.predict_raw(&x1, &cfg).expect("predict")
    });
    layer_breakdown(&model, &x1);
}

/// Median wall time per layer of MODEL.md §4 (fused, batch 1): the time between consecutive layer outputs, so
/// each row includes that layer's Upsample / Concat glue. The head row is the rest of the forward pass.
fn layer_breakdown(model: &FoloModel, x: &Tensor) {
    const RUNS: usize = 30;
    let mut rows: Vec<Vec<Duration>> = (0..24).map(|_| Vec::with_capacity(RUNS)).collect();
    for _ in 0..RUNS + 3 {
        let mut last = Instant::now();
        let mut t = vec![Duration::ZERO; 24];
        black_box(
            model
                .forward_traced(x, &mut |l, _| {
                    let now = Instant::now();
                    t[l] = now - last;
                    last = now;
                })
                .expect("forward"),
        );
        t[23] = last.elapsed();
        for (r, d) in rows.iter_mut().zip(t) {
            r.push(d);
        }
    }
    println!("per-layer median (fused, 640, b1):");
    let mut total = 0.0;
    for (l, r) in rows.iter_mut().enumerate() {
        let r = &mut r[3..];
        r.sort();
        let m = ms(r[r.len() / 2]);
        total += m;
        let name = folo_vision::model::LAYER_NAMES[l];
        println!("  {l:>2} {:<18} {m:>7.2} ms", if name.is_empty() { "(upsample/concat)" } else { name });
    }
    println!("  sum of medians       {total:>7.2} ms");
}
