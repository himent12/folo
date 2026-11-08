//! conv2d throughput at YOLO-typical shapes (N = 8, 64 -> 128 channels, 80x80).
//!
//! `cargo bench -p folo-gpu --bench conv`

use std::time::Instant;

use folo_gpu::kernels::conv::{self, Conv2dDesc};
use folo_gpu::kernels::pooling::Window2d;
use folo_gpu::{DType, GpuContext, UnaryOp};

fn bench(ctx: &GpuContext, d: &Conv2dDesc) -> (f64, f64) {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let es = d.dtype.size() as u64;
    let x = ctx.alloc(es * (d.n * d.c_in * d.h * d.w) as u64).unwrap();
    let w = ctx.alloc(es * (d.c_out * d.c_in / d.groups * kh * kw) as u64).unwrap();
    let b = ctx.alloc(es * d.c_out as u64).unwrap();
    let y = ctx.alloc(es * (d.n * d.c_out * oh * ow) as u64).unwrap();
    for buf in [&x, &w, &b] {
        ctx.clear(buf.raw(), buf.size());
    }
    let run = |n: usize| {
        for i in 0..n {
            // Submit in small groups so the GPU runs while the next dispatches are encoded.
            if i % 8 == 7 {
                ctx.flush();
            }
            conv::conv2d(ctx, d, x.raw(), w.raw(), Some(b.raw()), y.raw()).unwrap();
        }
        ctx.sync().unwrap();
    };
    run(3);
    let t = Instant::now();
    run(1);
    let iters = ((0.15 / t.elapsed().as_secs_f64().max(1e-6)) as usize).clamp(3, 200);
    // Best of three timed runs: the GPU is shared and boost clocks vary.
    let secs = (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min);
    (secs * 1e3, d.flops().unwrap() / secs / 1e9)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping conv bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    println!("{:<44} {:>9} {:>9}", "case", "ms", "GFLOPS");
    let base = Conv2dDesc {
        n: 8,
        c_in: 64,
        h: 80,
        w: 80,
        c_out: 128,
        groups: 1,
        win: Window2d::square(3, 1, 1),
        act: UnaryOp::Silu,
        dtype: DType::F32,
    };
    let cases = [
        ("3x3 s1 p1  8x64x80x80 -> 128", base.clone()),
        ("3x3 s2 p1  8x64x80x80 -> 128", Conv2dDesc { win: Window2d::square(3, 2, 1), ..base.clone() }),
        ("1x1 s1     8x64x80x80 -> 128", Conv2dDesc { win: Window2d::square(1, 1, 0), ..base.clone() }),
        ("dw 3x3 s1  8x128x80x80", Conv2dDesc { c_in: 128, groups: 128, ..base.clone() }),
    ];
    for (name, d) in &cases {
        let (ms, gf) = bench(&ctx, d);
        println!("{name:<44} {ms:>9.3} {gf:>9.0}");
    }
    if ctx.caps().shader_f16 {
        for (name, d) in &cases {
            let d = Conv2dDesc { dtype: DType::F16, ..d.clone() };
            let (ms, gf) = bench(&ctx, &d);
            println!("{:<44} {ms:>9.3} {gf:>9.0}", format!("f16 {name}"));
        }
    }
}
