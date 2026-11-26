//! LayerNorm, RMSNorm, Softmax backward, and depthwise conv2d throughput on the GPU backend.
//!
//! `cargo bench -p folo-gpu --bench norm_softmax`

use std::time::Instant;

use folo_core::rng::Rng;
use folo_core::{Device, Tensor, no_grad};
use folo_gpu::kernels::conv::{self, Conv2dDesc};
use folo_gpu::kernels::norm::{self, BnAct, BnDesc, BnRunning};
use folo_gpu::kernels::optim::{self, ManyItem, OptimUpdate};
use folo_gpu::kernels::pooling::Window2d;
use folo_gpu::kernels::reduce;
use folo_gpu::{DType, GpuContext, UnaryOp};

fn time_dev(dev: Device, mut f: impl FnMut()) -> f64 {
    let mut run = |n: usize| {
        for _ in 0..n {
            f();
        }
        folo_core::backend::synchronize(dev).unwrap();
    };
    run(3);
    let t = Instant::now();
    run(1);
    let iters = ((0.15 / t.elapsed().as_secs_f64().max(1e-6)) as usize).clamp(3, 300);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn time_ctx(ctx: &GpuContext, mut f: impl FnMut()) -> f64 {
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
    let iters = ((0.15 / t.elapsed().as_secs_f64().max(1e-6)) as usize).clamp(3, 300);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            run(iters);
            t.elapsed().as_secs_f64() / iters as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let dev = match folo_gpu::init() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping norm_softmax bench: {e}");
            return;
        }
    };
    let ctx = folo_gpu::GpuContext::shared().unwrap();
    println!("adapter: {}", ctx.summary());
    println!("{:<48} {:>9} {:>10}", "case", "ms", "GB/s");

    let mut rng = Rng::seed(42);
    for &(rows, d) in &[(4096usize, 768usize), (16384, 512), (2048, 2048)] {
        let bytes = (rows * d * 4) as f64;
        let x = Tensor::randn([rows, d], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
        let g = Tensor::randn([rows, d], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
        let w = Tensor::ones([d]).to_device(dev).unwrap();
        let b = Tensor::zeros([d]).to_device(dev).unwrap();

        let s = time_dev(dev, || {
            let _ng = no_grad();
            drop(x.layer_norm(Some(&w), Some(&b), 1e-5).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("layer_norm fwd [{rows}x{d}]"),
            s * 1e3,
            2.0 * bytes / s / 1e9
        );

        let xl = x.detach().requires_grad_();
        let wl = w.detach().requires_grad_();
        let bl = b.detach().requires_grad_();
        let s = time_dev(dev, || {
            let y = xl.layer_norm(Some(&wl), Some(&bl), 1e-5).unwrap();
            drop(y.backward_with(&g).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("layer_norm fwd+bwd [{rows}x{d}]"),
            s * 1e3,
            5.0 * bytes / s / 1e9
        );

        let s = time_dev(dev, || {
            let _ng = no_grad();
            drop(x.rms_norm(Some(&w), 1e-5).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("rms_norm fwd [{rows}x{d}]"),
            s * 1e3,
            2.0 * bytes / s / 1e9
        );

        let s = time_dev(dev, || {
            let y = xl.rms_norm(Some(&wl), 1e-5).unwrap();
            drop(y.backward_with(&g).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("rms_norm fwd+bwd [{rows}x{d}]"),
            s * 1e3,
            5.0 * bytes / s / 1e9
        );

        let s = time_dev(dev, || {
            let y = xl.softmax(1).unwrap();
            drop(y.backward_with(&g).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("softmax fwd+bwd [{rows}x{d}]"),
            s * 1e3,
            5.0 * bytes / s / 1e9
        );

        let s = time_dev(dev, || {
            let y = xl.log_softmax(1).unwrap();
            drop(y.backward_with(&g).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("log_softmax fwd+bwd [{rows}x{d}]"),
            s * 1e3,
            5.0 * bytes / s / 1e9
        );
    }

    // Column softmax backward (DFL-like [8, 16, 8400] dim=1)
    {
        let (o, d, inn) = (8usize, 16usize, 8400usize);
        let bytes = (o * d * inn * 4) as f64;
        let x =
            Tensor::randn([o, d, inn], 0.0, 1.0, &mut rng).to_device(dev).unwrap().detach().requires_grad_();
        let g = Tensor::randn([o, d, inn], 0.0, 1.0, &mut rng).to_device(dev).unwrap();
        let s = time_dev(dev, || {
            let y = x.softmax(1).unwrap();
            drop(y.backward_with(&g).unwrap());
        });
        println!(
            "{:<48} {:>9.3} {:>10.1}",
            format!("softmax cols fwd+bwd [{o}x{d}x{inn}]"),
            s * 1e3,
            5.0 * bytes / s / 1e9
        );
    }

    // Depthwise conv2d forward (3x3, 5x5, 7x7 f32 & f16)
    println!();
    println!("{:<48} {:>9} {:>10}", "depthwise conv2d", "ms", "GFLOPS");
    let dw_cases = [
        ("dw 3x3 s1 p1  8x128x80x80", 8usize, 128usize, 80usize, 80usize, 3usize, 1usize, 1usize),
        ("dw 5x5 s1 p2  8x128x80x80", 8, 128, 80, 80, 5, 1, 2),
        ("dw 7x7 s1 p3  8x128x80x80", 8, 128, 80, 80, 7, 1, 3),
        ("dw 7x7 s1 p3  8x256x40x40", 8, 256, 40, 40, 7, 1, 3),
        ("dw 3x3 s2 p1  8x128x80x80", 8, 128, 80, 80, 3, 2, 1),
    ];
    let dtypes: &[DType] = if ctx.caps().shader_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };
    for &dt in dtypes {
        let pfx = if dt == DType::F16 { "f16 " } else { "" };
        for &(name, n, c, h, w, k, st, pad) in &dw_cases {
            let d = Conv2dDesc {
                n,
                c_in: c,
                h,
                w,
                c_out: c,
                groups: c,
                win: Window2d::square(k, st, pad),
                act: UnaryOp::Silu,
                dtype: dt,
            };
            let (oh, ow) = d.output().unwrap();
            let es = dt.size() as u64;
            let x = ctx.alloc(es * (n * c * h * w) as u64).unwrap();
            let wt = ctx.alloc(es * (c * k * k) as u64).unwrap();
            let b = ctx.alloc(es * c as u64).unwrap();
            let y = ctx.alloc(es * (n * c * oh * ow) as u64).unwrap();
            for buf in [&x, &wt, &b] {
                ctx.clear(buf.raw(), buf.size());
            }
            let s = time_ctx(&ctx, || {
                conv::conv2d(&ctx, &d, x.raw(), wt.raw(), Some(b.raw()), y.raw()).unwrap();
            });
            println!(
                "{:<48} {:>9.3} {:>10.0}",
                format!("{pfx}{name}"),
                s * 1e3,
                d.flops().unwrap() / s / 1e9
            );
            let gx = ctx.alloc(es * (n * c * h * w) as u64).unwrap();
            let sb = time_ctx(&ctx, || {
                conv::conv2d_backward_input(&ctx, &d, y.raw(), wt.raw(), gx.raw()).unwrap();
            });
            println!(
                "{:<48} {:>9.3} {:>10.0}",
                format!("{pfx}bwd_in {name}"),
                sb * 1e3,
                d.flops().unwrap() / sb / 1e9
            );
        }
    }

    // Train-mode BatchNorm2d + SiLU forward & backward across YOLO shapes (fused & multi-chunk)
    println!();
    println!("{:<48} {:>9} {:>10}", "batch_norm + silu", "ms", "GB/s");
    for &dt in dtypes {
        let pfx = if dt == DType::F16 { "f16 " } else { "" };
        let eb = dt.size() as f64;
        for &(label, n, c, h, w) in &[
            ("bn [16,512,20,20]", 16usize, 512usize, 20usize, 20usize),
            ("bn [16,256,20,20]", 16, 256, 20, 20),
            ("bn [8,256,20,20]", 8, 256, 20, 20),
            ("bn [4,256,40,40]", 4, 256, 40, 40),
            ("bn [16,128,40,40]", 16, 128, 40, 40),
            ("bn [16,64,80,80]", 16, 64, 80, 80),
            ("bn [4,64,21,21] odd", 4, 64, 21, 21),
        ] {
            let elems = n * c * h * w;
            let bytes = elems as f64 * eb;
            let x = ctx.alloc(dt.bytes(elems)).unwrap();
            let y = ctx.alloc(dt.bytes(elems)).unwrap();
            let g = ctx.alloc(dt.bytes(elems)).unwrap();
            let dx = ctx.alloc(dt.bytes(elems)).unwrap();
            let wt = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let b = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let mean = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let inv = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let rm = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let rv = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let dw = ctx.alloc(DType::F32.bytes(c)).unwrap();
            let db = ctx.alloc(DType::F32.bytes(c)).unwrap();
            for buf in [&x, &g, &wt, &b, &rm, &rv] {
                ctx.clear(buf.raw(), buf.size());
            }
            let desc = BnDesc { n, c, hw: h * w, act: BnAct::Silu, dtype: dt };
            let sf = time_ctx(&ctx, || {
                let run = BnRunning { mean: rm.raw(), var: rv.raw(), momentum: 0.03 };
                norm::batch_norm_forward(
                    &ctx,
                    &desc,
                    x.raw(),
                    wt.raw(),
                    b.raw(),
                    1e-3,
                    Some(run),
                    y.raw(),
                    mean.raw(),
                    inv.raw(),
                )
                .unwrap();
            });
            println!(
                "{:<48} {:>9.4} {:>10.1}",
                format!("{pfx}{label} fwd"),
                sf * 1e3,
                2.0 * bytes / sf / 1e9
            );
            let sb = time_ctx(&ctx, || {
                norm::batch_norm_backward(
                    &ctx,
                    &desc,
                    x.raw(),
                    g.raw(),
                    wt.raw(),
                    b.raw(),
                    mean.raw(),
                    inv.raw(),
                    dx.raw(),
                    dw.raw(),
                    db.raw(),
                )
                .unwrap();
            });
            println!(
                "{:<48} {:>9.4} {:>10.1}",
                format!("{pfx}{label} bwd"),
                sb * 1e3,
                3.0 * bytes / sb / 1e9
            );
        }
    }

    // Arg-reduce rows (inner == 1) and mixed-alignment AdamW many
    println!();
    println!("{:<48} {:>9} {:>10}", "arg_reduce / mixed optim", "ms", "GB/s");
    for &(label, rows, r) in &[
        ("argmax [8400, 80]", 8400usize, 80usize),
        ("argmax [4096, 1024]", 4096, 1024),
        ("argmax [64, 8400]", 64, 8400),
        ("argmax [1, 65536]", 1, 65536),
    ] {
        let x = ctx.alloc(DType::F32.bytes(rows * r)).unwrap();
        let y = ctx.alloc(DType::F32.bytes(rows * 2)).unwrap();
        ctx.clear(x.raw(), x.size());
        let s = time_ctx(&ctx, || {
            reduce::arg_reduce(&ctx, true, x.raw(), y.raw(), rows, r, 1).unwrap();
        });
        let bytes = (rows * r * 4 + rows * 8) as f64;
        println!("{:<48} {:>9.4} {:>10.1}", label, s * 1e3, bytes / s / 1e9);
    }

    {
        // Mixed-alignment 16-tensor AdamW chunk: 15 x 1Mi f32 weights + 1 x 3-element bias
        let lens: Vec<usize> = (0..15).map(|_| 1usize << 20).chain([3]).collect();
        let total: usize = lens.iter().sum();
        let ps: Vec<_> = lens.iter().map(|&n| ctx.alloc(DType::F32.bytes(n)).unwrap()).collect();
        let gs: Vec<_> = lens.iter().map(|&n| ctx.alloc(DType::F32.bytes(n)).unwrap()).collect();
        let s0s: Vec<_> = lens.iter().map(|&n| ctx.alloc(DType::F32.bytes(n)).unwrap()).collect();
        let s1s: Vec<_> = lens.iter().map(|&n| ctx.alloc(DType::F32.bytes(n)).unwrap()).collect();
        for b in ps.iter().chain(&gs).chain(&s0s).chain(&s1s) {
            ctx.clear(b.raw(), b.size());
        }
        let items: Vec<_> = (0..lens.len())
            .map(|i| ManyItem {
                p: ps[i].raw(),
                g: gs[i].raw(),
                s0: Some(s0s[i].raw()),
                s1: Some(s1s[i].raw()),
                n: lens[i],
            })
            .collect();
        let adamw = OptimUpdate::AdamW {
            decay: Some(0.9999),
            beta1: 0.9,
            one_minus_beta1: 0.1,
            beta2: 0.999,
            one_minus_beta2: 0.001,
            step_size: 1e-3,
            bc2_sqrt: 0.99,
            eps: 1e-8,
        };
        let s = time_ctx(&ctx, || {
            optim::optimizer_step_many(&ctx, &adamw, &items).unwrap();
        });
        println!(
            "{:<48} {:>9.4} {:>10.1}",
            "adamw_many mixed (15x1Mi + 3)",
            s * 1e3,
            7.0 * (total * 4) as f64 / s / 1e9
        );
    }
}
