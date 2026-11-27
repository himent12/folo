//! Benchmarks GPU host readback optimizations:
//! 1. Multi-tensor single-submission batched readback (`download_many` / `read_batch`) vs sequential
//!    per-tensor readbacks (`download_async` / `read`), including FOLO v2 `iou_head` (`[1, 84, 8400]` +
//!    `[1, 1, 8400]`) and 4-tensor / 8-tensor head-and-stats readbacks.
//! 2. Pipelined 2-stage deferred readback (`download_deferred` overlapped with the next step's GPU compute,
//!    using submission-targeted waits) vs synchronous readback (`download` per step).
//!
//! Run with `cargo bench -p folo-gpu --bench readback`.

use std::hint::black_box;
use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{Tensor, no_grad};

const ROUNDS: usize = 3;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn time_ms(warmup: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    let _g = no_grad();
    let mut rng = Rng::seed(7);

    println!("readback bench on {}", ctx.summary());
    println!();
    println!(
        "=== 1. Multi-tensor readback: sequential vs single-submission batched (medians of {ROUNDS} rounds) ==="
    );
    println!("{:<34} {:>10} {:>10} {:>8} {:>12}", "case", "seq ms", "batch ms", "speedup", "subs (s->b)");

    let cases: &[(&str, &[&[usize]])] = &[
        ("folo2 iou_head (2: dec + iou)", &[&[1, 84, 8400], &[1, 1, 8400]]),
        ("4 small stats (4 x 1 KiB)", &[&[256], &[256], &[256], &[256]]),
        (
            "8 candidate overflows (8 x 16 KiB)",
            &[&[4096], &[4096], &[4096], &[4096], &[4096], &[4096], &[4096], &[4096]],
        ),
        ("4 medium tensors (4 x 256 KiB)", &[&[65536], &[65536], &[65536], &[65536]]),
    ];

    for &(name, shapes) in cases {
        let tensors: Vec<Tensor> =
            shapes.iter().map(|&s| Tensor::randn(s, 0.0, 1.0, &mut rng).to_device(dev).unwrap()).collect();
        let refs: Vec<&Tensor> = tensors.iter().collect();
        ctx.sync().unwrap();

        let (mut seq_rounds, mut bat_rounds) = (Vec::new(), Vec::new());
        let (mut seq_subs, mut bat_subs) = (0u64, 0u64);
        for _ in 0..ROUNDS {
            let b0 = ctx.work_stats().submissions;
            seq_rounds.push(time_ms(5, 30, || {
                for t in &refs {
                    black_box(pollster::block_on(folo_gpu::download_async(&ctx, t)).unwrap());
                }
            }));
            seq_subs = (ctx.work_stats().submissions - b0) / 35;

            let b1 = ctx.work_stats().submissions;
            bat_rounds.push(time_ms(5, 30, || {
                black_box(folo_gpu::download_many(&ctx, &refs).unwrap());
            }));
            bat_subs = (ctx.work_stats().submissions - b1) / 35;
        }
        let (seq_ms, bat_ms) = (median(seq_rounds), median(bat_rounds));
        println!(
            "{name:<34} {:>10.3} {:>10.3} {:>7.2}x {:>5} -> {:<4}",
            seq_ms,
            bat_ms,
            seq_ms / bat_ms,
            seq_subs,
            bat_subs
        );
    }

    println!();
    println!("=== 2. Pipelined compute + deferred readback (submission-targeted) vs sync readback ===");
    println!("{:<34} {:>10} {:>10} {:>8}", "workload (8 steps)", "sync ms", "pipe ms", "speedup");

    // A lightweight compute chain (4 matmuls of 512x512 + 256 KiB readback per step) where synchronous
    // readback drains the GPU idle between steps, while 1-step-ahead deferred readback overlaps step k's
    // DMA readback with step k+1's GPU compute and waits only on step k's SubmissionIndex.
    for &(label, m, out_elems) in &[
        ("matmul 512x512 + 256 KiB read", 512usize, 65536usize),
        ("matmul 1024x1024 + 1 MiB read", 1024usize, 262144usize),
    ] {
        let a = Tensor::randn([m, m], 0.0, 0.02, &mut rng).to_device(dev).unwrap();
        let b = Tensor::randn([m, m], 0.0, 0.02, &mut rng).to_device(dev).unwrap();
        let steps = 8usize;
        let run_step = || -> Tensor {
            let mut x = a.matmul(&b).unwrap();
            x = x.matmul(&b).unwrap();
            x.flatten(0, 1).unwrap().narrow(0, 0, out_elems).unwrap()
        };
        ctx.sync().unwrap();

        let (mut sync_rounds, mut pipe_rounds) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            sync_rounds.push(time_ms(2, 10, || {
                for _ in 0..steps {
                    let y = run_step();
                    let host = pollster::block_on(folo_gpu::download_async(&ctx, &y)).unwrap();
                    black_box(host);
                }
            }));
            pipe_rounds.push(time_ms(2, 10, || {
                let mut prev = folo_gpu::download_deferred(&ctx, &run_step()).unwrap();
                for _ in 1..steps {
                    let cur = run_step();
                    let next = folo_gpu::download_deferred(&ctx, &cur).unwrap();
                    // Collect previous step while `cur`'s compute + readback execute on the GPU.
                    black_box(prev.finish().unwrap());
                    prev = next;
                }
                black_box(prev.finish().unwrap());
            }));
        }
        let (s_ms, p_ms) = (median(sync_rounds), median(pipe_rounds));
        println!("{label:<34} {:>10.3} {:>10.3} {:>7.2}x", s_ms, p_ms, s_ms / p_ms);
    }
}
