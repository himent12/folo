//! Host cost of a fused GPU forward with the bind-group cache on and off, interleaved in one process (the order
//! alternating by round) to balance call order and load drift on a shared machine between the two. Run with `cargo bench -p folo-vision --bench dispatch_ab`
//! (`FOLO_BENCH_SCALES=n,s`, `FOLO_BENCH_PRECISION=f32,f16tc` and `FOLO_BENCH_BATCH=1,16` pick the configurations,
//! defaults `n`, `f32,f16tc` and `1`; `f16tc` needs `FOLO_GPU_TENSOR_CORES=1`; `FOLO_BENCH_ROUNDS` rounds, default 12).
//!
//! Per configuration and round, each mode (cache on / off) runs 5 warm-up calls, then 30 synchronised forwards
//! timing the encode and the whole call, then 30 calls back to back with one wait (pipelined). "Encode" is the host
//! time of the whole `forward()` call: allocation, bind-cache work, the automatic submissions of a full batch and
//! the command recording, not the recording alone. Each round prints its per-mode medians (one line per round), and
//! the table gives the median over rounds of each round's median, and its range.

use std::hint::black_box;
use std::time::Instant;

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, f16_unavailable};

/// The middle value; for an even count, the upper of the two middle ones (no averaging).
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn summary(v: &[f64]) -> String {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(0.0, f64::max);
    format!("{:.3} ({lo:.3}-{hi:.3})", median(v.to_vec()))
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => return println!("no GPU adapter ({e}); nothing to measure"),
    };
    let ctx = GpuContext::shared().expect("context");
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let rounds: usize = var("FOLO_BENCH_ROUNDS", "12").parse().expect("FOLO_BENCH_ROUNDS");
    println!(
        "folo-vision dispatch A/B on {} (bind-group cache on vs off, {rounds} rounds)",
        ctx.summary().name
    );
    for scale in var("FOLO_BENCH_SCALES", "n").split(',').map(str::trim).filter(|s| !s.is_empty()) {
        for prec in var("FOLO_BENCH_PRECISION", "f32,f16tc").split(',').map(str::trim) {
            let (prec, want_tc) = match prec {
                "f16tc" => (Precision::F16, true),
                p => (p.parse::<Precision>().expect("precision"), false),
            };
            if want_tc && !ctx.caps().coop_f16 {
                println!("skipping {prec} on tensor cores: {}", ctx.tensor_cores_status());
                continue;
            }
            if let Some(why) = f16_unavailable(dev).filter(|_| prec == Precision::F16) {
                println!("skipping f16: {why}");
                continue;
            }
            ctx.set_tensor_cores(want_tc);
            let mut model = FoloModel::new(ModelConfig::new(scale, 80).expect("config")).expect("model");
            model.to_device(dev).expect("to_device");
            model.fuse().expect("fuse");
            model.to_dtype(prec.dtype()).expect("to_dtype");
            let _g = no_grad();
            for batch in
                var("FOLO_BENCH_BATCH", "1").split(',').map(|b| b.trim().parse::<usize>().expect("batch"))
            {
                let x = Tensor::rand_uniform([batch, 3, 640, 640], 0.0, 1.0, &mut Rng::seed(0))
                    .to_device(dev)
                    .expect("upload")
                    .to_dtype(prec.dtype())
                    .expect("dtype");
                let forward = || drop(black_box(model.forward(&x).expect("forward")));
                // [mode][metric] -> per-round medians; mode 0 = cache on.
                let mut res = [[Vec::new(), Vec::new(), Vec::new()], [Vec::new(), Vec::new(), Vec::new()]];
                for r in 0..rounds {
                    for m in if r % 2 == 0 { [0, 1] } else { [1, 0] } {
                        ctx.set_bind_cache(m == 0);
                        for _ in 0..5 {
                            forward();
                            synchronize(dev).expect("sync");
                        }
                        let (mut enc, mut call) = (Vec::new(), Vec::new());
                        for _ in 0..30 {
                            let t = Instant::now();
                            forward();
                            enc.push(t.elapsed().as_secs_f64() * 1e3);
                            synchronize(dev).expect("sync");
                            call.push(t.elapsed().as_secs_f64() * 1e3);
                        }
                        let t = Instant::now();
                        for _ in 0..30 {
                            forward();
                        }
                        synchronize(dev).expect("sync");
                        res[m][0].push(median(enc));
                        res[m][1].push(median(call));
                        res[m][2].push(t.elapsed().as_secs_f64() * 1e3 / 30.0);
                    }
                    let mode =
                        |m: usize| format!("{:.3} {:.3} {:.3}", res[m][0][r], res[m][1][r], res[m][2][r]);
                    println!(
                        "  folo-{scale} {prec}{} b{batch} round {r:>2} (encode, sync call, pipelined ms): on {}  off {}",
                        if want_tc { " tc" } else { "" },
                        mode(0),
                        mode(1)
                    );
                }
                ctx.set_bind_cache(true);
                let label = if want_tc { format!("{prec}, tensor cores") } else { prec.to_string() };
                println!("\nfolo-{scale} {label} b{batch} (ms: median over rounds, range)");
                for (i, metric) in ["encode", "synchronised call", "pipelined"].iter().enumerate() {
                    let (on, off) = (median(res[0][i].clone()), median(res[1][i].clone()));
                    println!(
                        "  {metric:<18} cache on {}  off {}  {:+.0}%",
                        summary(&res[0][i]),
                        summary(&res[1][i]),
                        (on / off - 1.0) * 100.0
                    );
                }
            }
        }
    }
}
