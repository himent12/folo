//! Memory-bound kernels: effective bandwidth in GB/s (bytes read + written per second).
//!
//! `cargo bench -p folo-gpu --bench elementwise`

use std::time::Instant;

use folo_gpu::kernels::{elementwise, layout, reduce};
use folo_gpu::{BinaryOp, DType, GpuContext, UnaryOp};

fn time(ctx: &GpuContext, mut f: impl FnMut()) -> f64 {
    let mut run = |n: usize| {
        for i in 0..n {
            if i % 8 == 7 {
                ctx.flush();
            }
            f();
        }
        ctx.sync().unwrap();
    };
    run(3);
    let t = Instant::now();
    run(1);
    let iters = ((0.15 / t.elapsed().as_secs_f64().max(1e-6)) as usize).clamp(3, 500);
    // Best of three timed runs: the GPU is shared and boost clocks vary.
    (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping elementwise bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    let n = 1 << 26; // 64 Mi floats = 256 MiB per tensor
    let bytes = (n * 4) as f64;
    let a = ctx.alloc(DType::F32.bytes(n)).unwrap();
    let b = ctx.alloc(DType::F32.bytes(n)).unwrap();
    let c = ctx.alloc(DType::F32.bytes(n)).unwrap();
    for buf in [&a, &b] {
        ctx.clear(buf.raw(), buf.size());
    }
    let bias = ctx.alloc(4 * 256).unwrap();
    ctx.clear(bias.raw(), bias.size());
    println!("{:<44} {:>9} {:>9}", "case (64 Mi f32)", "ms", "GB/s");
    let report = |name: &str, secs: f64, traffic: f64| {
        println!("{name:<44} {:>9.3} {:>9.0}", secs * 1e3, traffic / secs / 1e9);
    };
    let s = time(&ctx, || ctx.copy(a.raw(), 0, c.raw(), 0, bytes as u64));
    report("buffer copy (reference)", s, 2.0 * bytes);
    for op in [UnaryOp::Relu, UnaryOp::Silu, UnaryOp::Exp] {
        let s = time(&ctx, || elementwise::unary(&ctx, op, a.raw(), c.raw(), n).unwrap());
        report(&format!("unary {op:?}"), s, 2.0 * bytes);
    }
    let s = time(&ctx, || elementwise::unary(&ctx, UnaryOp::Relu, a.raw(), c.raw(), n - 1).unwrap());
    report("unary Relu, odd length (scalar path)", s, 2.0 * bytes);
    let shape = [n];
    let s = time(&ctx, || {
        elementwise::binary(&ctx, BinaryOp::Add, a.raw(), &shape, b.raw(), &shape, c.raw()).unwrap();
    });
    report("binary add, same shape", s, 3.0 * bytes);
    let x_shape = [8, 256, 128, 256];
    let b_shape = [1, 256, 1, 1];
    let s = time(&ctx, || {
        elementwise::binary(&ctx, BinaryOp::Mul, a.raw(), &x_shape, bias.raw(), &b_shape, c.raw()).unwrap();
    });
    report("binary mul, [8,256,128,256] * [1,256,1,1]", s, 2.0 * bytes);
    let s = time(&ctx, || {
        elementwise::bias_act(&ctx, a.raw(), Some(bias.raw()), c.raw(), n, 256, 128 * 256, UnaryOp::Silu)
            .unwrap();
    });
    report("bias + SiLU (NCHW)", s, 2.0 * bytes);
    let s = time(&ctx, || {
        reduce::reduce(&ctx, reduce::ReduceOp::Sum, a.raw(), c.raw(), 1 << 16, 1024, 1).unwrap()
    });
    report("sum over last dim (65536 x 1024)", s, bytes);
    let s =
        time(&ctx, || reduce::reduce(&ctx, reduce::ReduceOp::Sum, a.raw(), c.raw(), 64, 1024, 1024).unwrap());
    report("sum over middle dim (64 x 1024 x 1024)", s, bytes);
    let s = time(&ctx, || reduce::softmax(&ctx, a.raw(), c.raw(), 1 << 16, 1024, false).unwrap());
    report("softmax rows (65536 x 1024)", s, 2.0 * bytes);
    let s =
        time(&ctx, || layout::permute(&ctx, a.raw(), c.raw(), &x_shape, &[0, 2, 3, 1], DType::F32).unwrap());
    report("permute NCHW -> NHWC [8,256,128,256]", s, 2.0 * bytes);
    let s =
        time(&ctx, || layout::permute(&ctx, a.raw(), c.raw(), &x_shape, &[1, 0, 3, 2], DType::F32).unwrap());
    report("permute generic [1,0,3,2]", s, 2.0 * bytes);
    let s = time(&ctx, || {
        layout::copy_rows(&ctx, a.raw(), c.raw(), 8, n / 16, n / 8, n / 16, DType::F32).unwrap()
    });
    report("concat half of 8 rows", s, bytes);
    let s = time(&ctx, || {
        layout::upsample_nearest2d(&ctx, a.raw(), c.raw(), 1024, 128, 128, 256, 256, DType::F32).unwrap()
    });
    report("upsample 2x (1024 planes 128 -> 256)", s, bytes / 4.0 + bytes);
}
