//! Micro-benchmark for GPU pooling, global average pooling, and 2x upsample backward kernels.
//!
//! `cargo bench -p folo-gpu --bench pooling`

use std::time::Instant;

use folo_gpu::kernels::pool_grad::{self, Pool2d};
use folo_gpu::kernels::pooling::{self, Window2d};
use folo_gpu::{DType, GpuContext};

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
    run(5);
    let t = Instant::now();
    run(5);
    let per = (t.elapsed().as_secs_f64() / 5.0).max(1e-6);
    let iters = ((0.06 / per) as usize).clamp(5, 300);
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
            eprintln!("skipping pooling bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    let has_f16 = ctx.caps().shader_f16;
    let max_elems = 16 * 128 * 160 * 160; // 52.4M elems
    let a = ctx.alloc(DType::F32.bytes(max_elems)).unwrap();
    let b = ctx.alloc(DType::F32.bytes(max_elems)).unwrap();
    let idx = ctx.alloc(DType::F32.bytes(max_elems * 2)).unwrap();
    ctx.clear(a.raw(), a.size());
    ctx.clear(b.raw(), b.size());
    ctx.clear(idx.raw(), idx.size());

    println!("{:<52} {:>10} {:>9}", "case", "ms", "GB/s");
    let report = |name: &str, secs: f64, traffic: f64| {
        println!("{name:<52} {:>10.4} {:>9.0}", secs * 1e3, traffic / secs / 1e9);
    };

    let dtypes: &[DType] = if has_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // 1. max_pool2d (values only)
    for &dt in dtypes {
        let eb = dt.size() as f64;
        for (label, planes, h, w, win) in [
            ("max_val SPPF 5x5 s1 [16,256,20,20]", 16 * 256, 20, 20, Window2d::square(5, 1, 2)),
            ("max_val SPPF 5x5 s1 [16,256,40,40]", 16 * 256, 40, 40, Window2d::square(5, 1, 2)),
            ("max_val 3x3 s1 [16,128,80,80]", 16 * 128, 80, 80, Window2d::square(3, 1, 1)),
            ("max_val 2x2 s2 [16,128,80,80]", 16 * 128, 80, 80, Window2d::square(2, 2, 0)),
        ] {
            let (oh, ow) = win.output(h, w).unwrap();
            let s = time(&ctx, || {
                pooling::max_pool2d(&ctx, a.raw(), b.raw(), planes, h, w, &win, dt).unwrap();
            });
            let traffic = ((planes * h * w) + (planes * oh * ow)) as f64 * eb;
            report(&format!("{label} {dt:?}"), s, traffic);
        }
    }

    // 2. max_pool2d_indices (training forward)
    for &dt in dtypes {
        let eb = dt.size() as f64;
        for (label, planes, h, w, win) in [
            (
                "max_idx SPPF 5x5 s1 [16,256,20,20]",
                16 * 256,
                20,
                20,
                Pool2d { kernel: (5, 5), stride: (1, 1), padding: (2, 2) },
            ),
            (
                "max_idx SPPF 5x5 s1 [16,256,40,40]",
                16 * 256,
                40,
                40,
                Pool2d { kernel: (5, 5), stride: (1, 1), padding: (2, 2) },
            ),
            (
                "max_idx 3x3 s1 [16,128,80,80]",
                16 * 128,
                80,
                80,
                Pool2d { kernel: (3, 3), stride: (1, 1), padding: (1, 1) },
            ),
        ] {
            let (oh, ow) = win.output(h, w).unwrap();
            let s = time(&ctx, || {
                pool_grad::max_pool2d_indices(&ctx, a.raw(), b.raw(), idx.raw(), (planes, h, w), &win, dt)
                    .unwrap();
            });
            let traffic = (planes * h * w) as f64 * eb + (planes * oh * ow) as f64 * (eb + 8.0);
            report(&format!("{label} {dt:?}"), s, traffic);
        }
    }

    // 3. upsample_nearest2d_backward 2x2
    for &dt in dtypes {
        let eb = dt.size() as f64;
        for (label, planes, h, w) in [
            ("upsample_bwd 2x2 [16,128,40->20]", 16 * 128, 20, 20),
            ("upsample_bwd 2x2 [16,64,80->40]", 16 * 64, 40, 40),
            ("upsample_bwd 2x2 [16,128,160->80]", 16 * 128, 80, 80),
        ] {
            let s = time(&ctx, || {
                pool_grad::upsample_nearest2d_backward(&ctx, a.raw(), b.raw(), (planes, h, w), (2, 2), dt)
                    .unwrap();
            });
            let traffic = (planes * (4 * h * w + h * w)) as f64 * eb;
            report(&format!("{label} {dt:?}"), s, traffic);
        }
    }

    // 4. avg_pool2d 2x2 s2 p0 fwd + bwd
    let avg2 = Pool2d { kernel: (2, 2), stride: (2, 2), padding: (0, 0) };
    for &dt in dtypes {
        let eb = dt.size() as f64;
        for (label, planes, h, w) in
            [("[16,128,80,80]", 16 * 128, 80, 80), ("[16,128,160,160]", 16 * 128, 160, 160)]
        {
            let (oh, ow) = (h / 2, w / 2);
            let traffic = (planes * (h * w + oh * ow)) as f64 * eb;
            let sf = time(&ctx, || {
                pool_grad::avg_pool2d(&ctx, a.raw(), b.raw(), (planes, h, w), &avg2, dt).unwrap();
            });
            report(&format!("avg_pool 2x2 s2 fwd {label} {dt:?}"), sf, traffic);
            let sb = time(&ctx, || {
                pool_grad::avg_pool2d_backward(&ctx, b.raw(), a.raw(), (planes, h, w), &avg2, dt).unwrap();
            });
            report(&format!("avg_pool 2x2 s2 bwd {label} {dt:?}"), sb, traffic);
        }
    }

    // 5. adaptive_avg_pool2d GAP 1x1 fwd + bwd
    for (label, planes, h, w) in [
        ("[16,512,20,20]", 16 * 512, 20, 20),
        ("[16,256,40,40]", 16 * 256, 40, 40),
        ("[1,64,160,160]", 64, 160, 160),
    ] {
        let traffic = (planes * (h * w + 1)) as f64 * 4.0;
        let sf = time(&ctx, || {
            pool_grad::adaptive_avg_pool2d(&ctx, a.raw(), b.raw(), (planes, h, w), (1, 1)).unwrap();
        });
        report(&format!("adaptive_avg GAP 1x1 fwd {label}"), sf, traffic);
        let sb = time(&ctx, || {
            pool_grad::adaptive_avg_pool2d_backward(&ctx, b.raw(), a.raw(), (planes, h, w), (1, 1)).unwrap();
        });
        report(&format!("adaptive_avg GAP 1x1 bwd {label}"), sb, traffic);
    }
}
