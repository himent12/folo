//! The forward convs of folo-n and folo-s at batch 16, 640 px, that run furthest from the GEMM peak: the
//! layers at 40 and 20 px (few output tiles, large `K`) and the stride-2 downsamples.
//!
//! `cargo bench -p folo-gpu --bench conv_model` (`FOLO_BENCH_PRECISION=f32,f16tc`, default `f32`; `f16tc` needs
//! `FOLO_GPU_TENSOR_CORES=1`). The `FOLO_CONV_*` switches are tested for presence, so any value (even `0`) turns
//! them on. `FOLO_CONV_TUNE` also times every regular tile config with 1–8 split-K chunks per case; on tensor
//! cores it times every tensor-core tile config instead (these explicit-config rows keep the automatic split-K
//! choice), then the picked config with 1–8 chunks. `FOLO_CONV_GEMM` times the plain GEMM of the same size
//! (skipped on tensor cores), and `FOLO_CONV_BIG` adds folo-s's layers at 80 px and above. Each row is the best of
//! three timed runs of back-to-back dispatches, and `x` is how often the layer occurs in the forward, so the last
//! line is roughly what these layers cost per forward.

use std::time::Instant;

use folo_gpu::kernels::conv::{self, Conv2dDesc, ConvPlan};
use folo_gpu::kernels::coop::{self, CoopConfig};
use folo_gpu::kernels::gemm::{self, TileConfig};
use folo_gpu::kernels::pooling::Window2d;
use folo_gpu::{DType, GpuContext, UnaryOp};

fn bench(ctx: &GpuContext, d: &Conv2dDesc, plan: ConvPlan) -> f64 {
    bench_with(ctx, d, |x, w, b, y| {
        conv::conv2d_with_plan(ctx, d, x, w, Some(b), y, plan).unwrap();
    })
}

/// `d` on tensor cores with an explicit tile config.
fn bench_coop(ctx: &GpuContext, d: &Conv2dDesc, cfg: CoopConfig) -> f64 {
    bench_with(ctx, d, |x, w, b, y| {
        coop::conv2d(ctx, d, d.dtype, x, w, Some(b), y, Some(cfg)).unwrap();
    })
}

/// Seconds per call of `f(x, weight, bias, y)` on buffers sized for `d`: best of three timed runs.
fn bench_with(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    f: impl Fn(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
) -> f64 {
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
            if i % 8 == 7 {
                ctx.flush();
            }
            f(x.raw(), w.raw(), b.raw(), y.raw());
        }
        ctx.sync().unwrap();
    };
    run(3);
    let t = Instant::now();
    run(1);
    let iters = ((0.1 / t.elapsed().as_secs_f64().max(1e-6)) as usize).clamp(5, 400);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// The plain batched GEMM of the conv's GEMM view (`B` an already gathered `K × N` matrix per image): what
/// the conv would cost without the gather.
fn bench_gemm(ctx: &GpuContext, d: &Conv2dDesc) -> f64 {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (m, n, k) = (d.c_out, oh * ow, d.c_in * kh * kw);
    let mut g = gemm::GemmDesc::new(m, n, k, false, false, d.n);
    g.a.outer = 0;
    g.dtype = d.dtype;
    g.act = d.act;
    let es = d.dtype.size() as u64;
    let a = ctx.alloc(es * (m * k) as u64).unwrap();
    let b = ctx.alloc(es * (d.n * k * n) as u64).unwrap();
    let c = ctx.alloc(es * (d.n * m * n) as u64).unwrap();
    for buf in [&a, &b] {
        ctx.clear(buf.raw(), buf.size());
    }
    let run = |iters: usize| {
        for i in 0..iters {
            if i % 8 == 7 {
                ctx.flush();
            }
            gemm::gemm(ctx, &g, a.raw(), b.raw(), None, c.raw()).unwrap();
        }
        ctx.sync().unwrap();
    };
    run(3);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            run(50);
            t.elapsed().as_secs_f64() / 50.0
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping conv_model bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    let tune = std::env::var("FOLO_CONV_TUNE").is_ok();
    let precisions = std::env::var("FOLO_BENCH_PRECISION").unwrap_or_else(|_| "f32".into());
    // (name, c_in, size, c_out, stride, occurrences per forward)
    let mut cases: Vec<(&str, usize, usize, usize, usize, usize)> = vec![
        ("n c3k 3x3 64->64 @20", 64, 20, 64, 1, 9),
        ("n head 3x3 256->64 @20", 256, 20, 64, 1, 1),
        ("n down5 3x3 s2 128->256 @40", 128, 40, 256, 2, 1),
        ("n neck down 3x3 s2 128->128 @40", 128, 40, 128, 2, 1),
        ("n c3k 3x3 32->32 @40", 32, 40, 32, 1, 4),
        ("n bottleneck 3x3 64->32 @40", 64, 40, 32, 1, 2),
        ("n head 3x3 128->64 @40", 128, 40, 64, 1, 1),
        ("n head 3x3 64->64 @40", 64, 40, 64, 1, 1),
        ("n down4 3x3 s2 64->128 @80", 64, 80, 128, 2, 1),
        ("s c3k 3x3 128->128 @20", 128, 20, 128, 1, 9),
        ("s head 3x3 512->64 @20", 512, 20, 64, 1, 1),
        ("s down5 3x3 s2 256->512 @40", 256, 40, 512, 2, 1),
        ("s neck down 3x3 s2 256->256 @40", 256, 40, 256, 2, 1),
        ("s c3k 3x3 64->64 @40", 64, 40, 64, 1, 4),
        ("s head 3x3 256->64 @40", 256, 40, 64, 1, 1),
        ("s down4 3x3 s2 128->256 @80", 128, 80, 256, 2, 1),
        ("s 1x1 512->512 @20", 512, 20, 512, 1, 4),
    ];
    if std::env::var("FOLO_CONV_BIG").is_ok() {
        // folo-s's layers at 80 px and above, where most of a batch-16 forward's GEMM time goes.
        cases.extend([
            ("s down2 3x3 s2 32->64 @320", 32, 320, 64, 2, 1),
            ("s down3 3x3 s2 128->128 @160", 128, 160, 128, 2, 1),
            ("s head 3x3 128->64 @80", 128, 80, 64, 1, 1),
            ("s head 3x3 64->64 @80", 64, 80, 64, 1, 1),
            ("s c3k 3x3 64->64 @80", 64, 80, 64, 1, 4),
            ("s 1x1 256->256 @80", 256, 80, 256, 1, 2),
            ("s 1x1 128->128 @160", 128, 160, 128, 1, 2),
        ]);
    }
    for prec in precisions.split(',') {
        let (dtype, tc) = match prec.trim() {
            "f32" => (DType::F32, false),
            "f16" => (DType::F16, false),
            "f16tc" => (DType::F16, true),
            p => panic!("FOLO_BENCH_PRECISION: f32, f16 or f16tc, not {p}"),
        };
        if dtype == DType::F16 && !ctx.caps().shader_f16 || tc && !ctx.caps().coop_f16 {
            println!("{prec}: not available on this adapter");
            continue;
        }
        ctx.set_tensor_cores(tc);
        println!("\n{prec}, batch 16\n{:<36} {:>3} {:>9} {:>8}", "case", "x", "us", "TFLOPS");
        let mut total = 0.0;
        for &(name, c_in, size, c_out, stride, times) in &cases {
            let k = if name.contains("1x1") { 1 } else { 3 };
            let d = Conv2dDesc {
                n: 16,
                c_in,
                h: size,
                w: size,
                c_out,
                groups: 1,
                win: Window2d::square(k, stride, k / 2),
                act: UnaryOp::Silu,
                dtype,
            };
            let secs = bench(&ctx, &d, ConvPlan::default());
            total += secs * times as f64;
            let tf = d.flops().unwrap() / secs / 1e12;
            println!("{name:<36} {times:>3} {:>9.1} {tf:>8.1}", secs * 1e6);
            if std::env::var("FOLO_CONV_GEMM").is_ok() && !tc {
                let secs = bench_gemm(&ctx, &d);
                let tf = d.flops().unwrap() / secs / 1e12;
                println!("{:<40} {:>9.1} {tf:>8.1}", "    same-size plain GEMM", secs * 1e6);
            }
            if tune && !tc {
                for c in [
                    TileConfig::LARGE,
                    TileConfig::MEDIUM,
                    TileConfig::SMALL,
                    TileConfig::ROWS32,
                    TileConfig::ROWS16,
                ] {
                    for splits in [1, 2, 3, 4, 6, 8] {
                        let secs = bench(&ctx, &d, ConvPlan { tiles: Some(c), splits: Some(splits) });
                        let tf = d.flops().unwrap() / secs / 1e12;
                        let row = format!("    {}x{}x{} split {splits}", c.bm, c.bn, c.bk);
                        println!("{row:<40} {:>9.1} {tf:>8.1}", secs * 1e6);
                    }
                }
            }
            if tune && tc {
                for c in [
                    CoopConfig::C128X128,
                    CoopConfig::C64X128,
                    CoopConfig::C64X64,
                    CoopConfig::C32X128,
                    CoopConfig::C16X256,
                ] {
                    let secs = bench_coop(&ctx, &d, c);
                    let tf = d.flops().unwrap() / secs / 1e12;
                    let row = format!("    tc {}x{}x{}", c.bm, c.bn, c.bk);
                    println!("{row:<40} {:>9.1} {tf:>8.1}", secs * 1e6);
                }
                // The picked tensor-core tile config with 1–8 chunks.
                for splits in [1, 2, 3, 4, 6, 8] {
                    let secs = bench(&ctx, &d, ConvPlan { tiles: None, splits: Some(splits) });
                    let tf = d.flops().unwrap() / secs / 1e12;
                    println!("{:<40} {:>9.1} {tf:>8.1}", format!("    split {splits}"), secs * 1e6);
                }
            }
        }
        println!("{:<36} {:>3} {:>9.1}", "sum (x occurrences)", "", total * 1e6);
    }
}
