//! The ceiling of a frozen forward plan: a fused GPU forward against two replays of its own trace, interleaved in
//! one process (the order rotating by round). Run with `cargo bench -p folo-vision --bench plan_gate`
//! (`FOLO_BENCH_SCALES=n,s`, `FOLO_BENCH_PRECISION=f32,f16tc` and `FOLO_BENCH_BATCH=1,16` pick the configurations,
//! defaults `n`, `f32,f16tc` and `1`; `f16tc` needs `FOLO_GPU_TENSOR_CORES=1`; `FOLO_BENCH_ROUNDS` rounds, default
//! 12; `FOLO_BENCH_CALLS` synchronised calls per mode and round, default 60).
//!
//! Modes: `forward` is `model.forward(x)` with forward plans off; `planned` the same forward of an identical model
//! with plans on (recorded on its second call, replayed after: input copy, plan, result copies); `replay` encodes the traced operations again one by one through
//! `GpuContext::replay` (the kernel dispatch path with its uniform arena and bind-group cache, no tensor ops or pool);
//! `frozen` encodes them from `GpuContext::freeze` (bind groups made once, parameters written once: only the wgpu
//! calls). Per call: host time encoding (`enc`), submitting the rest (`sub`) and waiting (`wait`). Printed per round:
//! each mode's median call; at the end, per mode, p50 / p90 / p99 of every call and the median and range of the round
//! medians.

use std::hint::black_box;
use std::time::Instant;

use folo_core::backend::synchronize;
use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};
use folo_gpu::GpuContext;
use folo_nn::Module;
use folo_vision::model::{FoloModel, ModelConfig};
use folo_vision::predict::{Precision, f16_unavailable};

const MODES: [&str; 4] = ["forward", "replay", "frozen", "planned"];

/// The `p` quantile of `v` (sorted in place).
fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
}

fn range(v: &[f64]) -> String {
    let mut s = v.to_vec();
    let lo = s.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = s.iter().copied().fold(0.0, f64::max);
    format!("{:.3} ({lo:.3}-{hi:.3})", pct(&mut s, 0.5))
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => return println!("no GPU adapter ({e}); nothing to measure"),
    };
    let ctx = GpuContext::shared().expect("context");
    let var = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let rounds: usize = var("FOLO_BENCH_ROUNDS", "12").parse().expect("FOLO_BENCH_ROUNDS");
    let calls: usize = var("FOLO_BENCH_CALLS", "60").parse().expect("FOLO_BENCH_CALLS");
    println!("folo-vision plan gate on {} (forward vs trace replays, {rounds} rounds)", ctx.summary().name);
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
            let load = || {
                let mut model = FoloModel::new(ModelConfig::new(scale, 80).expect("config")).expect("model");
                model.to_device(dev).expect("to_device");
                model.fuse().expect("fuse");
                model.to_dtype(prec.dtype()).expect("to_dtype");
                model
            };
            let (model, planned) = (load(), load());
            model.set_forward_plans(false);
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
                let plan = || drop(black_box(planned.forward(&x).expect("forward")));
                for _ in 0..3 {
                    plan();
                }
                let replays = planned.plan_replays();
                plan();
                if planned.plan_replays() == replays {
                    println!("  no plan: {:?}", planned.plan_refusal());
                }
                for _ in 0..10 {
                    forward();
                }
                synchronize(dev).expect("sync");
                ctx.trace_begin();
                forward();
                let ops = ctx.trace_end();
                synchronize(dev).expect("sync");
                let frozen = ctx.freeze(&ops).expect("freeze");
                let names = |k: &str| ops.iter().filter(|o| o.name == k).count();
                println!(
                    "\nfolo-{scale} {prec}{} b{batch}: {} traced operations ({} copies, {} clears)",
                    if want_tc { " tc" } else { "" },
                    ops.len(),
                    names("copy"),
                    names("clear")
                );
                let (plans, bytes) = ctx.plan_stats();
                println!("  plans held by the context: {plans}, {:.1} MiB", bytes as f64 / (1 << 20) as f64);
                let run = |m: usize| match m {
                    0 => forward(),
                    1 => ops.iter().for_each(|op| ctx.replay(op).expect("replay")),
                    2 => ctx.replay_frozen(&frozen),
                    _ => plan(),
                };
                // [mode] -> every call's (enc, sub, wait, total) and each round's median total and pipelined time.
                let mut all: Vec<[Vec<f64>; 4]> = (0..4).map(|_| Default::default()).collect();
                let mut per_round: Vec<[Vec<f64>; 2]> = (0..4).map(|_| Default::default()).collect();
                let mut dispatches = [0u64; 4];
                for r in 0..rounds {
                    let mut line = String::new();
                    for k in 0..4 {
                        let m = (r + k) % 4;
                        for _ in 0..5 {
                            run(m);
                            synchronize(dev).expect("sync");
                        }
                        let d0 = ctx.work_stats().dispatches;
                        let mut tot = Vec::with_capacity(calls);
                        for _ in 0..calls {
                            let t = Instant::now();
                            run(m);
                            let e = t.elapsed().as_secs_f64() * 1e3;
                            ctx.flush();
                            let s = t.elapsed().as_secs_f64() * 1e3;
                            synchronize(dev).expect("sync");
                            let w = t.elapsed().as_secs_f64() * 1e3;
                            all[m][0].push(e);
                            all[m][1].push(s - e);
                            all[m][2].push(w - s);
                            all[m][3].push(w);
                            tot.push(w);
                        }
                        dispatches[m] = (ctx.work_stats().dispatches - d0) / calls as u64;
                        let t = Instant::now();
                        for _ in 0..30 {
                            run(m);
                        }
                        synchronize(dev).expect("sync");
                        per_round[m][1].push(t.elapsed().as_secs_f64() * 1e3 / 30.0);
                        per_round[m][0].push(pct(&mut tot, 0.5));
                    }
                    for m in 0..4 {
                        line +=
                            &format!("  {} {:.3} / {:.3}", MODES[m], per_round[m][0][r], per_round[m][1][r]);
                    }
                    println!("  round {r:>2} (sync call / pipelined ms):{line}");
                }
                println!("  dispatches per call: {dispatches:?}");
                for m in 0..4 {
                    let [e, s, w, t] = &mut all[m];
                    println!(
                        "  {:<8} call p50 {:.3} p90 {:.3} p99 {:.3} | enc p50 {:.3} sub {:.3} wait {:.3} | \
                         round medians {} | pipelined {}",
                        MODES[m],
                        pct(t, 0.5),
                        pct(t, 0.9),
                        pct(t, 0.99),
                        pct(e, 0.5),
                        pct(s, 0.5),
                        pct(w, 0.5),
                        range(&per_round[m][0]),
                        range(&per_round[m][1]),
                    );
                }
                let (f, z, p) =
                    (pct(&mut all[0][3], 0.5), pct(&mut all[2][3], 0.5), pct(&mut all[3][3], 0.5));
                println!(
                    "  p50 against forward: frozen (the ceiling) {:+.1}%, planned {:+.1}%",
                    (z / f - 1.0) * 100.0,
                    (p / f - 1.0) * 100.0
                );
            }
        }
    }
}
