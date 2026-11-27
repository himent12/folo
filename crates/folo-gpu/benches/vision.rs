//! Micro-benchmark for GPU vision pre-processing (`letterbox_rgb8` across common camera/video resolutions).
//!
//! `cargo bench -p folo-gpu --bench vision`

use std::time::Instant;

use folo_gpu::kernels::vision::{self, LetterboxDesc};
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

fn build_taps(pw: usize, ph: usize, rw: usize, rh: usize) -> Vec<u32> {
    let axis = |src_n: usize, dst_n: usize| -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut a = Vec::with_capacity(dst_n);
        let mut b = Vec::with_capacity(dst_n);
        let mut w = Vec::with_capacity(dst_n);
        for d in 0..dst_n {
            let pos = ((d as f64 + 0.5) * src_n as f64 / dst_n as f64 - 0.5).clamp(0.0, (src_n - 1) as f64);
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(src_n - 1);
            let frac = ((pos - i0 as f64) * 2048.0).round() as u32;
            a.push(i0 as u32);
            b.push(i1 as u32);
            w.push(frac.min(2048));
        }
        (a, b, w)
    };
    let (x0, x1, wx) = axis(pw, rw);
    let (y0, y1, wy) = axis(ph, rh);
    let mut taps = Vec::with_capacity(3 * (rw + rh));
    taps.extend(x0);
    taps.extend(x1);
    taps.extend(wx);
    taps.extend(y0);
    taps.extend(y1);
    taps.extend(wy);
    taps
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skipping vision bench: {e}");
            return;
        }
    };
    println!("adapter: {}", ctx.summary());
    let has_f16 = ctx.caps().shader_f16;
    let dtypes: &[DType] = if has_f16 { &[DType::F32, DType::F16] } else { &[DType::F32] };

    // (label, sw, sh, ow, oh, rw, rh, px, py, fx, fy)
    let cases = [
        ("640x640 -> 640x640 (no prefilter)", 640usize, 640usize, 640, 640, 640, 640, 0, 0, 1, 1),
        ("1280x720 -> 640x384 (fx=2, fy=1)", 1280, 720, 640, 384, 640, 360, 0, 12, 2, 1),
        ("1920x1080 -> 640x384 (fx=3, fy=2)", 1920, 1080, 640, 384, 640, 360, 0, 12, 3, 2),
        ("3840x2160 -> 640x384 (fx=6, fy=5)", 3840, 2160, 640, 384, 640, 360, 0, 12, 6, 5),
        ("3840x2160 -> 1280x736 (fx=3, fy=2)", 3840, 2160, 1280, 736, 1280, 720, 0, 8, 3, 2),
        ("1000x800 -> 640x640 (fx=1, fy=1)", 1000, 800, 640, 640, 640, 512, 0, 64, 1, 1),
    ];

    println!("{:<46} {:>10} {:>10}", "case", "ms", "GB/s");
    for &dt in dtypes {
        for &(label, sw, sh, ow, oh, rw, rh, px, py, fx, fy) in &cases {
            let src_bytes = sw * sh * 3;
            let src = ctx.alloc((src_bytes.div_ceil(4) * 4) as u64).unwrap();
            ctx.clear(src.raw(), src.size());
            let pw = sw.div_ceil(fx);
            let ph = sh.div_ceil(fy);
            let taps_vec = build_taps(pw, ph, rw, rh);
            let taps = ctx.upload(bytemuck::cast_slice(&taps_vec)).unwrap();
            let out_elems = 3 * ow * oh;
            let out = ctx.alloc(dt.bytes(out_elems)).unwrap();
            let desc = LetterboxDesc {
                out: [ow, oh],
                src: [sw, sh],
                src_offset: 0,
                pad: [px, py],
                resized: [rw, rh],
                prefilter: [fx, fy],
                pad_value: 114,
                scale: 1.0 / 255.0,
                out_offset: 0,
                taps: 0,
            };
            let secs = time(&ctx, || {
                vision::letterbox_rgb8(&ctx, src.raw(), taps.raw(), out.raw(), &desc, dt).unwrap();
            });
            let traffic = (src_bytes as u64 + dt.bytes(out_elems)) as f64;
            println!("{:<46} {:>10.4} {:>10.1}", format!("{label} {dt:?}"), secs * 1e3, traffic / secs / 1e9);
        }
    }
}
