//! Tensor-core backward conv2d tests: input/weight gradients, split-K, sub-pixel dX, pointwise dW, and AMP scale.

use super::{cdesc, coop, run_conv, traced_tile};
use crate::kernels::conv::{self, Conv2dDesc};
use crate::kernels::coop::{self, CoopConfig};
use crate::kernels::pooling::Window2d;
use crate::tests::coop::{assert_within, bound, upload};
use crate::tests::{Rng, f16_round};
use crate::{DType, GpuContext, GpuTensor, UnaryOp};

/// f64 reference gradients of the conv `d` for the output gradient `g` and their absolute-term sums (for the error
/// bounds), in one pass: `(gx, gw, sum_x, sum_w)`. Output channels are split over threads (each owns its slice of
/// `gw` and a partial `gx`, summed in a fixed order), so the largest cases take a fraction of a second.
type ConvGrads = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

fn cpu_conv_grads(d: &Conv2dDesc, x: &[f32], w: &[f32], g: &[f32]) -> ConvGrads {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let per_co = cin_g * kh * kw;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(d.c_out);
    let cos = d.c_out.div_ceil(threads);
    let (mut gw, mut sw) = (vec![0f64; w.len()], vec![0f64; w.len()]);
    let partial_x: Vec<(Vec<f64>, Vec<f64>)> = std::thread::scope(|scope| {
        let workers: Vec<_> = gw
            .chunks_mut(cos * per_co)
            .zip(sw.chunks_mut(cos * per_co))
            .enumerate()
            .map(|(t, (gw, sw))| {
                scope.spawn(move || {
                    let (mut gx, mut sx) = (vec![0f64; x.len()], vec![0f64; x.len()]);
                    for n in 0..d.n {
                        for co in t * cos..((t + 1) * cos).min(d.c_out) {
                            let grp = co / cout_g;
                            for oy in 0..oh {
                                for ox in 0..ow {
                                    let gv = g[((n * d.c_out + co) * oh + oy) * ow + ox] as f64;
                                    for ci in 0..cin_g {
                                        for ky in 0..kh {
                                            for kx in 0..kw {
                                                let iy = (oy * d.win.stride.0 + ky * d.win.dilation.0)
                                                    as isize
                                                    - d.win.padding.0 as isize;
                                                let ix = (ox * d.win.stride.1 + kx * d.win.dilation.1)
                                                    as isize
                                                    - d.win.padding.1 as isize;
                                                if iy < 0
                                                    || ix < 0
                                                    || iy >= d.h as isize
                                                    || ix >= d.w as isize
                                                {
                                                    continue;
                                                }
                                                let xi = ((n * d.c_in + grp * cin_g + ci) * d.h
                                                    + iy as usize)
                                                    * d.w
                                                    + ix as usize;
                                                let wi = ((co * cin_g + ci) * kh + ky) * kw + kx;
                                                let (px, pw) = (gv * w[wi] as f64, gv * x[xi] as f64);
                                                gx[xi] += px;
                                                sx[xi] += px.abs();
                                                gw[wi - t * cos * per_co] += pw;
                                                sw[wi - t * cos * per_co] += pw.abs();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    (gx, sx)
                })
            })
            .collect();
        workers.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut gx, mut sx) = (vec![0f64; x.len()], vec![0f64; x.len()]);
    for (px, ps) in &partial_x {
        for i in 0..x.len() {
            gx[i] += px[i];
            sx[i] += ps[i];
        }
    }
    let cast = |v: Vec<f64>| v.into_iter().map(|s| s as f32).collect();
    (cast(gx), cast(gw), cast(sx), cast(sw))
}

/// Both gradients of the f16 conv `d` (input gradient f16; weight gradient as `w_out`) against the f64 reference,
/// on tensor cores and on the regular kernels, each with an f32 and an f16 weight gradient.
fn run_conv_grads(ctx: &GpuContext, d: &Conv2dDesc, seed: u64) {
    conv_grads(ctx, d, seed, false);
}

/// Both gradients of the f16 conv `d` with tensor cores on, then off, and the weight gradient as f32 and as f16,
/// against [`cpu_conv_grads`]; returns the weight gradients' routes ([`conv::LAST_ROUTE`]) in that order. `exact`: small integers
/// (`-2..=2`) instead of `[-1, 1)`, so every sum is exact in f32 and the bounds drop to the output rounding
/// (`k = 1`); a dropped split-K chunk, off by its partial sum, then fails however long the reduction.
fn conv_grads(ctx: &GpuContext, d: &Conv2dDesc, seed: u64, exact: bool) -> Vec<conv::WgradRoute> {
    let mut rng = Rng::new(seed);
    let mut draw = |n: usize| {
        let v = rng.vec(n, -1.0, 1.0);
        if exact { v.iter().map(|v| (v * 2.49).round()).collect() } else { f16_round(&v) }
    };
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let x = draw(d.n * d.c_in * d.h * d.w);
    let w = draw(d.c_out * (d.c_in / d.groups) * kh * kw);
    let g = draw(d.n * d.c_out * oh * ow);
    let (want_x, want_w, sum_x, sum_w) = cpu_conv_grads(d, &x, &w, &g);
    let (gx, gw, gg) =
        (upload(ctx, &x, DType::F16), upload(ctx, &w, DType::F16), upload(ctx, &g, DType::F16));
    let ident = |d: &Conv2dDesc| Conv2dDesc { act: UnaryOp::Identity, ..d.clone() };
    let kx = if exact { 1 } else { d.c_out / d.groups * kh * kw };
    let tol_x: Vec<f64> =
        want_x.iter().zip(&sum_x).map(|(&y, &s)| bound(y, s as f64, kx, 0.0, DType::F16)).collect();
    let kw_ = if exact { 1 } else { d.n * oh * ow };
    let mut routes = Vec::new();
    for tc in [true, false] {
        ctx.set_tensor_cores(tc);
        for w_out in [DType::F32, DType::F16] {
            let dx = GpuTensor::empty(ctx, &[x.len()], DType::F16).unwrap();
            let dw = GpuTensor::empty(ctx, &[w.len()], w_out).unwrap();
            conv::conv2d_backward_input(ctx, d, gg.raw(), gw.raw(), dx.raw()).unwrap();
            conv::LAST_ROUTE.set(None);
            conv::conv2d_backward_weight_as(ctx, d, gx.raw(), gg.raw(), dw.raw(), w_out).unwrap();
            routes.push(conv::LAST_ROUTE.take());
            assert_eq!(ctx.tensor_cores(), tc, "{}", ctx.tensor_cores_status());
            let tol_w: Vec<f64> =
                want_w.iter().zip(&sum_w).map(|(&y, &s)| bound(y, s as f64, kw_, 0.0, w_out)).collect();
            let what = format!("{:?} tensor cores {tc}", ident(d));
            assert_within(&dx.to_vec().unwrap(), &want_x, &tol_x, &format!("input gradient of {what}"));
            assert_within(
                &dw.to_vec().unwrap(),
                &want_w,
                &tol_w,
                &format!("weight gradient ({w_out:?}) of {what}"),
            );
        }
    }
    routes
}

#[test]
fn coop_conv_gradients() {
    let (ctx, _g) = coop!();
    let s1 = Window2d::square(3, 1, 1);
    let s2 = Window2d::square(3, 2, 1);
    let p1 = Window2d::square(1, 1, 0);
    let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
    let rect = Window2d { kernel: (1, 3), stride: (2, 1), padding: (0, 1), dilation: (1, 1) };
    let cases = [
        // folo-n / s: 3×3 s1 p1 and s2 p1, the 3-channel stem, 1×1 (input gradient as a GEMM), ragged channels.
        cdesc(2, 16, 20, 20, 32, 1, s1),
        cdesc(2, 16, 13, 17, 32, 1, s2),
        cdesc(1, 3, 33, 31, 16, 1, s2),
        cdesc(2, 24, 7, 9, 40, 1, p1),
        cdesc(3, 20, 11, 10, 70, 1, s1),
        cdesc(1, 130, 6, 6, 33, 1, p1),
        // Dilation, rectangular kernels, a strided 1×1, groups, depthwise (direct kernels).
        cdesc(1, 8, 12, 9, 24, 1, dil),
        cdesc(2, 5, 7, 8, 12, 1, rect),
        cdesc(2, 6, 9, 9, 4, 1, Window2d::square(1, 2, 0)),
        cdesc(2, 16, 10, 10, 32, 2, s1),
        cdesc(2, 32, 10, 10, 32, 32, s1),
        // A long weight-gradient reduction (split-K: N·OH·OW = 6400) with a wide M.
        cdesc(4, 16, 40, 40, 128, 1, s1),
        // Stride phases of the input gradient: groups, no padding, a kernel smaller than the stride (phases
        // without taps), a kernel of 4 at stride 2, a phase GEMM wide enough for the 128×128 tiles.
        cdesc(2, 16, 11, 9, 24, 2, s2),
        cdesc(2, 12, 13, 12, 20, 1, Window2d::square(3, 2, 0)),
        cdesc(1, 8, 14, 13, 12, 1, Window2d::square(2, 3, 1)),
        cdesc(2, 8, 12, 15, 16, 1, Window2d::square(4, 2, 1)),
        cdesc(2, 160, 34, 34, 136, 1, s2),
        // Strides beyond the kernel: most phases have no taps (only the ones with taps are packed and run).
        cdesc(2, 8, 17, 19, 12, 1, Window2d::square(1, 4, 0)),
        cdesc(1, 8, 23, 21, 12, 1, Window2d::square(3, 5, 1)),
        cdesc(1, 2, 4, 4, 3, 1, Window2d::square(1, 65536, 0)),
    ];
    for (i, d) in cases.iter().enumerate() {
        run_conv_grads(&ctx, d, 100 + i as u64);
    }
    ctx.set_tensor_cores(true);
}

/// The stem's weight gradient (3 → 16, 3×3 s2): `C16X32` on tensor cores and the regular kernels' 16×32 `STEM`
/// tile when tensor cores are off; `N·OH·OW = 4·80·80 = 25600`, dozens of split-K chunks, exact sums so a dropped
/// chunk fails.
#[test]
fn coop_conv_weight_gradient_stem_split_k() {
    let (ctx, _g) = coop!();
    let d = cdesc(4, 3, 160, 160, 16, 1, Window2d::square(3, 2, 1));
    let routes = conv_grads(&ctx, &d, 200, true);
    for route in &routes[..2] {
        assert!(matches!(route, Some((true, (16, 32), s)) if *s >= 40), "tensor-core stem route {route:?}");
    }
    for route in &routes[2..] {
        assert!(matches!(route, Some((false, (16, 32), s)) if *s >= 40), "regular stem route {route:?}");
    }
    ctx.set_tensor_cores(true);
}

/// A narrow weight (4 → 8, 3×3) over a long reduction (`N·OH·OW = 6·160·160 = 153600`) on tensor cores: hundreds
/// of split-K chunks, exact sums so a dropped chunk fails.
#[test]
fn coop_conv_weight_gradient_long_split_k() {
    let (ctx, _g) = coop!();
    let d = cdesc(6, 4, 160, 160, 8, 1, Window2d::square(3, 1, 1));
    let routes = conv_grads(&ctx, &d, 201, true);
    for route in &routes[..2] {
        assert!(matches!(route, Some((true, _, s)) if *s >= 200), "tensor-core route {route:?}");
    }
    for route in &routes[2..] {
        assert!(matches!(route, Some((false, _, s)) if *s >= 200), "regular route {route:?}");
    }
    ctx.set_tensor_cores(true);
}

/// The weight gradient's f32 output takes f16 gradients whose products sum past the f16 range (an f16 result
/// would overflow) and keeps them finite.
#[test]
fn coop_weight_gradient_f32_output_does_not_overflow() {
    let (ctx, _g) = coop!();
    let d = cdesc(4, 8, 32, 32, 16, 1, Window2d::square(3, 1, 1));
    let x = vec![60000f32; d.n * d.c_in * d.h * d.w];
    let g = vec![60000f32; d.n * d.c_out * d.h * d.w];
    let (gx, gg) = (upload(&ctx, &x, DType::F16), upload(&ctx, &g, DType::F16));
    let dw = GpuTensor::empty(&ctx, &[d.c_out * d.c_in * 9], DType::F32).unwrap();
    conv::conv2d_backward_weight_as(&ctx, &d, gx.raw(), gg.raw(), dw.raw(), DType::F32).unwrap();
    let got = dw.to_vec().unwrap();
    // The centre tap sees every pixel: 4 · 32 · 32 · 60000² = 1.47e13.
    let centre = got[4];
    assert!((centre / 1.47456e13 - 1.0).abs() < 1e-3, "{centre}");
    assert!(got.iter().all(|v| v.is_finite()));
    // An f32 conv's weight gradient cannot be written as f16.
    let f32_conv = Conv2dDesc { dtype: DType::F32, ..d.clone() };
    let x32 = upload(&ctx, &x, DType::F32);
    let g32 = upload(&ctx, &g, DType::F32);
    assert!(
        conv::conv2d_backward_weight_as(&ctx, &f32_conv, x32.raw(), g32.raw(), dw.raw(), DType::F16).is_err()
    );
}

/// B1 verification: stride-1 `CONV_MODE = 2` tap_mask (3x3, 5x5, 4x8 exact 32-tap limit, 7x7 >32 fallback,
/// dilated 3x3), backward dW never picks C128X128, and C1 batch-1 20x20 picks C32X32 (4x more tiles).
#[test]
fn coop_conv_backward_tap_mask_and_c32x32() {
    let (ctx, _g) = coop!();
    let k4x8 = Window2d { kernel: (4, 8), stride: (1, 1), padding: (1, 3), dilation: (1, 1) };
    let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
    for (i, d) in [
        cdesc(2, 16, 11, 13, 24, 1, Window2d::square(3, 1, 1)),
        cdesc(1, 8, 9, 10, 16, 1, Window2d::square(5, 1, 2)),
        cdesc(1, 4, 10, 11, 12, 1, k4x8),
        cdesc(1, 4, 13, 13, 8, 1, Window2d::square(7, 1, 3)),
        cdesc(2, 8, 12, 10, 16, 1, dil),
    ]
    .iter()
    .enumerate()
    {
        run_conv_grads(&ctx, d, 300 + i as u64);
    }
    // Wide M=128, N=16*9=144 weight gradient must pick C64X128 (never C128X128).
    let wide = cdesc(2, 16, 16, 16, 128, 1, Window2d::square(3, 1, 1));
    let routes = conv_grads(&ctx, &wide, 310, false);
    for r in &routes[..2] {
        assert!(matches!(r, Some((true, (64, 128), _))), "expected C64X128 for wide dW, got {r:?}");
    }
    // Batch-1 20x20 pointwise conv (plus ragged M/N/K) picks C32X32 and matches CPU reference.
    ctx.set_tensor_cores(true);
    let b1 = cdesc(1, 64, 20, 20, 128, 1, Window2d::square(1, 1, 0));
    assert_eq!(coop::pick_config_conv_batched(128, 400, 1), CoopConfig::C32X32);
    assert_eq!(coop::pick_config_conv_batched(128, 400, 16), CoopConfig::C64X64);
    for (seed, d) in [(311u64, b1), (312, cdesc(1, 48, 19, 17, 50, 1, Window2d::square(1, 1, 0)))] {
        run_conv(&ctx, &d, false, DType::F16, Some(CoopConfig::C32X32), seed);
        let (oh, ow) = d.output().unwrap();
        let mut rng = Rng::new(seed);
        let gx = upload(&ctx, &f16_round(&rng.vec(d.c_in * d.h * d.w, -1.0, 1.0)), DType::F16);
        let gw = upload(&ctx, &f16_round(&rng.vec(d.c_out * d.c_in, -1.0, 1.0)), DType::F16);
        let gy = GpuTensor::empty(&ctx, &[d.c_out * oh * ow], DType::F16).unwrap();
        assert_eq!(
            traced_tile(&ctx, || {
                conv::conv2d(&ctx, &d, gx.raw(), gw.raw(), None, gy.raw()).unwrap();
            }),
            (32, 32)
        );
    }
}

/// Formula 1 verification: fused 2×2 sub-pixel stride-2 3×3 input gradient across odd/even `H`/`W`, `W % 4 == 0`
/// (vec4 epilogue, including `CHECK == false` with odd `H = 15` and `UW = 10` row-wrap) and odd `W` (scalar
/// epilogue), paddings `0/1/2/3` and asymmetric padding, groups, and tiny `1×1`/`2×2`/`3×3` inputs.
#[test]
fn coop_conv_subpixel_stride2_input_grad() {
    let (ctx, _g) = coop!();
    let s2 = |ph, pw| Window2d { kernel: (3, 3), stride: (2, 2), padding: (ph, pw), dilation: (1, 1) };
    let cases = [
        // Standard YOLO shapes: vec4 epilogue, UW even (10 wraps every 2 cells, 16 aligned, 8 tile-aligned).
        cdesc(2, 16, 16, 16, 32, 1, s2(1, 1)),
        cdesc(2, 16, 20, 20, 32, 1, s2(1, 1)),
        cdesc(2, 32, 16, 16, 64, 1, s2(1, 1)),
        cdesc(1, 64, 20, 20, 64, 1, s2(1, 1)),
        // Odd H=15 with W=16: UH=8, UW=8 -> N=64, M=64, K=128 (CHECK == false yet iy=15 must stay masked).
        cdesc(2, 16, 15, 16, 32, 1, s2(1, 1)),
        // Odd W / scalar epilogue and tiny spatial sizes.
        cdesc(2, 8, 13, 17, 16, 1, s2(1, 1)),
        cdesc(1, 16, 16, 17, 24, 1, s2(1, 1)),
        cdesc(2, 4, 1, 1, 8, 1, s2(1, 1)),
        cdesc(2, 4, 2, 2, 8, 1, s2(1, 1)),
        cdesc(2, 4, 3, 3, 8, 1, s2(0, 0)),
        // Paddings 0, 2, 3 and asymmetric (0, 1), (2, 1).
        cdesc(2, 16, 16, 16, 16, 1, s2(0, 0)),
        cdesc(2, 16, 15, 17, 16, 1, s2(0, 0)),
        cdesc(2, 16, 16, 16, 16, 1, s2(2, 2)),
        cdesc(2, 12, 13, 15, 16, 1, s2(2, 2)),
        cdesc(1, 8, 12, 12, 12, 1, s2(3, 3)),
        cdesc(2, 16, 15, 16, 24, 1, s2(0, 1)),
        cdesc(2, 12, 16, 13, 20, 1, s2(2, 1)),
        // Groups > 1, wide cin_g = 256, and C64X128 sub-pixel (N=1600 CHECK=true, N=2048 CHECK=false).
        cdesc(2, 16, 16, 20, 32, 2, s2(1, 1)),
        cdesc(2, 24, 15, 13, 36, 4, s2(0, 1)),
        cdesc(1, 256, 8, 8, 256, 1, s2(1, 1)),
        cdesc(1, 32, 80, 80, 64, 1, s2(1, 1)),
        cdesc(1, 32, 64, 128, 64, 1, s2(1, 1)),
    ];
    for (i, d) in cases.iter().enumerate() {
        run_conv_grads(&ctx, d, 400 + i as u64);
    }
    // Verify the fused path dispatches exactly 1 pack + 1 GEMM (no 4-phase loop, no pixel-shuffle pass).
    ctx.set_tensor_cores(true);
    let d = cdesc(2, 16, 20, 20, 32, 1, s2(1, 1));
    let (oh, ow) = d.output().unwrap();
    let gg = GpuTensor::zeros(&ctx, &[2 * 32 * oh * ow], DType::F16).unwrap();
    let gw = GpuTensor::zeros(&ctx, &[32 * 16 * 9], DType::F16).unwrap();
    let dx = GpuTensor::empty(&ctx, &[2 * 16 * 20 * 20], DType::F16).unwrap();
    ctx.trace_begin();
    conv::conv2d_backward_input(&ctx, &d, gg.raw(), gw.raw(), dx.raw()).unwrap();
    let names: Vec<_> = ctx.trace_end().into_iter().map(|op| op.name).collect();
    assert_eq!(names, ["conv2d_subpixel_pack", "coop_gemm"], "expected fused 2-dispatch sub-pixel dX");
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

/// Formula 2 & 3 verification: division-free `vec4` per-image split-K 1×1 weight gradient (`CONV_MODE` 0,
/// `TRANS_B`) and `conv2d_wgrad_reduce` across narrow/wide channels, single/multi-image, `sp == 1` and `sp > 1`,
/// groups, odd `H·W`, odd `wn` (fallback to `reduce_cols`), plus exact integer split-K sums.
#[test]
fn coop_conv_pointwise_split_k_weight_grad() {
    let (ctx, _g) = coop!();
    let p1 = Window2d::square(1, 1, 0);
    let cases = [
        // Single image (direct write, zero reduction) and single image with spatial splits.
        cdesc(1, 32, 16, 16, 32, 1, p1),
        cdesc(1, 64, 40, 40, 64, 1, p1),
        // Multi-image YOLO 1×1 shapes: C16X32, C16X64, C32X32, C32X64, C64X64, C64X128, plus both WGRAD_REDUCE paths.
        cdesc(4, 16, 20, 20, 16, 1, p1),
        cdesc(4, 48, 20, 20, 16, 1, p1),
        cdesc(4, 16, 20, 20, 32, 1, p1),
        cdesc(4, 32, 20, 20, 32, 1, p1),
        cdesc(4, 32, 20, 20, 64, 1, p1),
        cdesc(4, 64, 20, 20, 32, 1, p1),
        cdesc(4, 96, 20, 20, 32, 1, p1),
        cdesc(4, 48, 40, 40, 64, 1, p1),
        cdesc(4, 64, 16, 16, 128, 1, p1),
        cdesc(4, 96, 16, 16, 128, 1, p1),
        cdesc(4, 128, 20, 20, 128, 1, p1),
        cdesc(2, 256, 8, 8, 256, 1, p1),
        // Large weight (`wn4 = 9216`, `splits = 17` odd tail) exercising `conv2d_wgrad_reduce` FLAT=true 4x-unrolled path.
        cdesc(5, 64, 16, 16, 128, 1, p1),
        cdesc(3, 64, 16, 16, 64, 1, Window2d::square(3, 1, 1)),
        // folo-s stem (3→32 3×3 s2: C32X32 + PER_IMG) and grouped/odd 1×1 (reduce_cols fallback).
        cdesc(4, 3, 32, 32, 32, 1, Window2d::square(3, 2, 1)),
        cdesc(4, 32, 32, 32, 64, 2, p1),
        cdesc(3, 24, 13, 11, 40, 4, p1),
        cdesc(4, 5, 16, 16, 7, 1, p1),
    ];
    for (i, d) in cases.iter().enumerate() {
        run_conv_grads(&ctx, d, 500 + i as u64);
    }
    // Check C64X64 / C64X128 on wide 1×1 dW (never C128X128) and C32X32 on folo-s 3→32 stem dW.
    assert!(matches!(
        conv_grads(&ctx, &cdesc(2, 96, 16, 16, 128, 1, p1), 540, false)[0],
        Some((true, (64, 64), _))
    ));
    assert!(matches!(
        conv_grads(&ctx, &cdesc(2, 128, 16, 16, 128, 1, p1), 541, false)[0],
        Some((true, (64, 128), _))
    ));
    let stem_r = conv_grads(&ctx, &cdesc(4, 3, 64, 64, 32, 1, Window2d::square(3, 2, 1)), 542, true)[0];
    assert!(
        matches!(stem_r, Some((true, (32, 32), s)) if s >= 4),
        "expected C32X32 PER_IMG stem, got {stem_r:?}"
    );
    // Exact integer split-K check on 48->64 @ 40x40 (N=4, many slabs -> any dropped slab fails immediately).
    let d = cdesc(4, 48, 40, 40, 64, 1, p1);
    let routes = conv_grads(&ctx, &d, 550, true);
    for r in &routes[..2] {
        assert!(
            matches!(r, Some((true, (64, 64), s)) if *s >= 16),
            "expected split-K coop 1x1 dW, got {r:?}"
        );
    }
    for r in &routes[2..] {
        assert!(matches!(r, Some((false, _, s)) if *s >= 16), "expected split-K regular 1x1 dW, got {r:?}");
    }
    // Verify 1x1 split-K dispatches coop_gemm + conv2d_wgrad_reduce (no reduce_cols when wn % 4 == 0).
    ctx.set_tensor_cores(true);
    let gx = GpuTensor::zeros(&ctx, &[4 * 48 * 1600], DType::F16).unwrap();
    let gg = GpuTensor::zeros(&ctx, &[4 * 64 * 1600], DType::F16).unwrap();
    let dw = GpuTensor::empty(&ctx, &[64 * 48], DType::F32).unwrap();
    ctx.trace_begin();
    conv::conv2d_backward_weight_as(&ctx, &d, gx.raw(), gg.raw(), dw.raw(), DType::F32).unwrap();
    let names: Vec<_> = ctx.trace_end().into_iter().map(|op| op.name).collect();
    assert_eq!(names, ["coop_gemm", "conv2d_wgrad_reduce"], "expected 1x1 coop_gemm + wgrad_reduce");
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}

/// High-magnitude (AMP `loss_scale = 65536`) finiteness check on the new backward paths: `subpixel` `C64X128`
/// `dX`, `1×1` per-image split-K `dW`, and `PER_IMG` `C32X32` stem `dW` stay finite and accurate.
#[test]
fn coop_conv_backward_large_scale_stays_finite() {
    let (ctx, _g) = coop!();
    ctx.set_tensor_cores(true);
    let d_sp = cdesc(1, 32, 80, 80, 64, 1, Window2d::square(3, 2, 1));
    let gg = upload(&ctx, &vec![60000f32; 64 * 40 * 40], DType::F16);
    let gw = upload(&ctx, &vec![0.001f32; 64 * 32 * 9], DType::F16);
    let dx = GpuTensor::empty(&ctx, &[32 * 80 * 80], DType::F16).unwrap();
    assert_eq!(
        traced_tile(&ctx, || conv::conv2d_backward_input(&ctx, &d_sp, gg.raw(), gw.raw(), dx.raw()).unwrap()),
        (64, 128)
    );
    let got_dx = dx.to_vec().unwrap();
    assert!(got_dx.iter().all(|v| v.is_finite()), "non-finite subpixel dX");
    for (d, expect) in [
        (cdesc(16, 64, 40, 40, 128, 1, Window2d::square(1, 1, 0)), 16.0 * 1600.0 * 3.6e9),
        (cdesc(16, 3, 64, 64, 32, 1, Window2d::square(3, 2, 1)), 16.0 * 1024.0 * 3.6e9),
    ] {
        let (oh, ow) = d.output().unwrap();
        let (kh, kw) = d.win.kernel;
        let gx = upload(&ctx, &vec![60000f32; d.n * d.c_in * d.h * d.w], DType::F16);
        let gg = upload(&ctx, &vec![60000f32; d.n * d.c_out * oh * ow], DType::F16);
        let dw = GpuTensor::empty(&ctx, &[d.c_out * d.c_in * kh * kw], DType::F32).unwrap();
        conv::conv2d_backward_weight_as(&ctx, &d, gx.raw(), gg.raw(), dw.raw(), DType::F32).unwrap();
        let got = dw.to_vec().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "non-finite dW for {d:?}");
        let centre = got[kh * kw / 2] as f64;
        assert!((centre / expect - 1.0).abs() < 1e-2, "{d:?}: got {centre}, expected {expect}");
    }
}

/// 1×1 `conv2d_backward_input` on tensor cores uses `TRANS_A` directly (single `coop_gemm` dispatch, no
/// `permute` / weight-transpose allocation) across ungrouped, grouped, and odd-channel shapes.
#[test]
fn coop_conv_pointwise_dx_uses_trans_a_directly() {
    let (ctx, _g) = coop!();
    let p1 = Window2d::square(1, 1, 0);
    for (i, d) in [
        cdesc(1, 64, 20, 20, 64, 1, p1),
        cdesc(2, 128, 80, 80, 64, 1, p1),
        cdesc(2, 12, 15, 13, 24, 1, p1),
        cdesc(2, 5, 11, 13, 7, 1, p1),
        cdesc(3, 15, 9, 11, 21, 3, p1),
        cdesc(2, 32, 16, 16, 48, 4, p1),
    ]
    .iter()
    .enumerate()
    {
        run_conv_grads(&ctx, d, 600 + i as u64);
    }
    ctx.set_tensor_cores(true);
    let d = cdesc(2, 64, 20, 20, 128, 2, p1);
    let gg = GpuTensor::zeros(&ctx, &[2 * 128 * 400], DType::F16).unwrap();
    let gw = GpuTensor::zeros(&ctx, &[128 * 32], DType::F16).unwrap();
    let dx = GpuTensor::empty(&ctx, &[2 * 64 * 400], DType::F16).unwrap();
    ctx.trace_begin();
    conv::conv2d_backward_input(&ctx, &d, gg.raw(), gw.raw(), dx.raw()).unwrap();
    let trace = ctx.trace_end();
    assert_eq!(trace.iter().map(|op| op.name).collect::<Vec<_>>(), ["coop_gemm"]);
    assert!(trace[0].spec.contains("TRANS_A=1"), "expected TRANS_A=1 in {}", trace[0].spec);
}
