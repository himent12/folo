//! conv2d (implicit GEMM, pointwise GEMM and depthwise paths, f32 and f16) against a direct CPU loop.

use super::{Rng, assert_close, f16_round, gpu, to_f16, traced_splits};
use crate::kernels::conv::{self, Conv2dDesc, ConvPlan, split};
use crate::kernels::gemm::{self, TileConfig};
use crate::kernels::pooling::Window2d;
use crate::{DType, GpuContext, GpuOptions, GpuTensor, UnaryOp};

fn cpu_conv(d: &Conv2dDesc, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let cin_g = d.c_in / d.groups;
    let cout_g = d.c_out / d.groups;
    let mut y = vec![0f32; d.n * d.c_out * oh * ow];
    for n in 0..d.n {
        for co in 0..d.c_out {
            let g = co / cout_g;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = 0f64;
                    for ci in 0..cin_g {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * d.win.stride.0 + ky * d.win.dilation.0) as isize
                                    - d.win.padding.0 as isize;
                                let ix = (ox * d.win.stride.1 + kx * d.win.dilation.1) as isize
                                    - d.win.padding.1 as isize;
                                if iy < 0 || ix < 0 || iy >= d.h as isize || ix >= d.w as isize {
                                    continue;
                                }
                                let xi =
                                    ((n * d.c_in + g * cin_g + ci) * d.h + iy as usize) * d.w + ix as usize;
                                let wi = ((co * cin_g + ci) * kh + ky) * kw + kx;
                                s += x[xi] as f64 * w[wi] as f64;
                            }
                        }
                    }
                    let v = s as f32 + bias.map_or(0.0, |b| b[co]);
                    y[((n * d.c_out + co) * oh + oy) * ow + ox] = d.act.apply(v);
                }
            }
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn desc(n: usize, c_in: usize, h: usize, w: usize, c_out: usize, groups: usize, win: Window2d) -> Conv2dDesc {
    Conv2dDesc { n, c_in, h, w, c_out, groups, win, act: UnaryOp::Identity, dtype: DType::F32 }
}

fn run(d: &Conv2dDesc, with_bias: bool, cfg: Option<TileConfig>, seed: u64) {
    let ctx = gpu!();
    run_plan(&ctx, d, with_bias, ConvPlan { tiles: cfg, splits: None }, seed, false);
}

/// Runs `d` with `plan` against the CPU loop. With `trace` (only on a context no other test uses), returns the `K`
/// chunks it ran as ([`traced_splits`]) after checking them against [`split::plan`]; otherwise, or when skipped, 0.
fn run_plan(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    plan: ConvPlan,
    seed: u64,
    trace: bool,
) -> usize {
    if d.dtype == DType::F16 && !ctx.caps().shader_f16 {
        eprintln!("note: skipping f16 conv test, adapter lacks shader-f16");
        return 0;
    }
    let mut rng = Rng::new(seed);
    let (kh, kw) = d.win.kernel;
    let mut x = rng.vec(d.n * d.c_in * d.h * d.w, -1.0, 1.0);
    let mut w = rng.vec(d.c_out * (d.c_in / d.groups) * kh * kw, -1.0, 1.0);
    let mut b = rng.vec(d.c_out, -1.0, 1.0);
    if d.dtype == DType::F16 {
        x = f16_round(&x);
        w = f16_round(&w);
        b = f16_round(&b);
    }
    let want = cpu_conv(d, &x, &w, with_bias.then_some(b.as_slice()));
    let up = |v: &[f32]| match d.dtype {
        DType::F32 => GpuTensor::from_slice(ctx, v, &[v.len()]).unwrap(),
        DType::F16 => GpuTensor::from_f16(ctx, &to_f16(v), &[v.len()]).unwrap(),
    };
    let (gx, gw, gb) = (up(&x), up(&w), up(&b));
    let gy = GpuTensor::empty(ctx, &[want.len()], d.dtype).unwrap();
    let bias = with_bias.then(|| gb.raw());
    let encode = || {
        conv::conv2d_with_plan(ctx, d, gx.raw(), gw.raw(), bias, gy.raw(), plan).unwrap();
    };
    let splits = if trace {
        traced_splits(ctx, encode)
    } else {
        encode();
        0
    };
    let got = gy.to_vec().unwrap();
    let k = (d.c_in / d.groups * kh * kw) as f32;
    let (atol, rtol) = if d.dtype == DType::F16 { (3e-3, 2e-3) } else { (2e-5 * k.sqrt(), 1e-5) };
    assert_close(&got, &want, atol, rtol, &format!("conv {d:?} bias={with_bias} {plan:?}"));
    if trace {
        assert_eq!(splits, planned_splits(ctx, d, plan), "chunks of conv {d:?} {plan:?}");
    }
    splits
}

/// The `K` chunks [`split::plan`] gives the implicit GEMM of `d` on the regular kernels (1 for the other paths).
fn planned_splits(ctx: &GpuContext, d: &Conv2dDesc, plan: ConvPlan) -> usize {
    if d.is_depthwise() || d.is_pointwise() {
        return 1;
    }
    let (oh, ow) = d.output().unwrap();
    let g = d.gemm_view(oh, ow, false);
    let cfg = plan.tiles.unwrap_or_else(|| gemm::pick_config_conv(ctx, &g));
    let (bm, bn, bk) = (cfg.bm as usize, cfg.bn as usize, cfg.bk as usize);
    let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn);
    split::plan(bk, g.k, tiles, d.n * d.groups, d.n * d.c_out * oh * ow, plan.splits).0
}

#[test]
fn conv_implicit_gemm() {
    let mut d = desc(2, 5, 9, 9, 7, 1, Window2d::square(3, 1, 1));
    d.act = UnaryOp::Silu;
    run(&d, true, None, 1);
    run(&desc(2, 8, 10, 11, 16, 1, Window2d::square(3, 2, 1)), true, None, 2);
    let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
    run(&desc(1, 4, 12, 9, 6, 1, dil), false, None, 3);
    let rect = Window2d { kernel: (1, 3), stride: (2, 1), padding: (0, 1), dilation: (1, 1) };
    run(&desc(2, 3, 7, 8, 5, 1, rect), true, None, 4);
    // 1x1 with stride 2 goes through implicit GEMM, not the pointwise path.
    run(&desc(2, 6, 9, 9, 4, 1, Window2d::square(1, 2, 0)), true, None, 5);
    // 7x7 stem-like, stride 2, padding 3.
    run(&desc(1, 3, 17, 17, 8, 1, Window2d::square(7, 2, 3)), true, None, 6);
}

/// The tap mask's edge windows (`CONV_MODE` 1, one bit per tap) on the regular generator, f32 with the picked tiles
/// and f16 with explicit ones (the regular kernels also with tensor cores on): exactly 32 taps (4×8: bit 31 is the
/// last tap, inside the image for most pixels) and 5×5 (25 taps), not depthwise.
#[test]
fn conv_tap_mask_windows() {
    let k4x8 = Window2d { kernel: (4, 8), stride: (1, 1), padding: (1, 3), dilation: (1, 1) };
    let k4x8_s2 = Window2d { kernel: (4, 8), stride: (2, 2), padding: (2, 4), dilation: (1, 1) };
    for (d, seed) in [
        (desc(2, 3, 9, 13, 6, 1, k4x8), 80),
        (desc(1, 4, 11, 12, 8, 1, k4x8_s2), 81),
        (desc(2, 4, 11, 11, 6, 1, Window2d::square(5, 1, 2)), 82),
        (desc(1, 5, 12, 10, 8, 1, Window2d::square(5, 2, 2)), 83),
    ] {
        run(&d, true, None, seed);
        run(&Conv2dDesc { dtype: DType::F16, ..d }, true, Some(TileConfig::MEDIUM), seed + 10);
    }
}

/// A forward conv whose `K` (45) is no multiple of any `BK` on a channel slice read in place, NaN in the channels
/// right after the slice, finite weights, no activation, on the regular generator: the gather must stop at `K`
/// (reading on would take the next channel, and 0 × NaN is NaN).
#[test]
fn conv_ragged_k_slice_before_nan() {
    let ctx = gpu!();
    // Channels 3..8 of 12: 3 finite noise channels before the slice, 4 NaN after it.
    let (n, c, c_all, c0, h, w) = (2, 5, 12, 3, 7, 9);
    let hw = h * w;
    for (dtype, cfg, seed) in [(DType::F32, None, 84), (DType::F16, Some(TileConfig::MEDIUM), 85)] {
        if dtype == DType::F16 && !ctx.caps().shader_f16 {
            continue;
        }
        let d = Conv2dDesc { dtype, ..desc(n, c, h, w, 24, 1, Window2d::square(3, 1, 1)) };
        let mut rng = Rng::new(seed);
        let round = |v: Vec<f32>| if dtype == DType::F16 { f16_round(&v) } else { v };
        let mut full = round(rng.vec(n * c_all * hw, -1.0, 1.0));
        let (wt, b) = (round(rng.vec(24 * c * 9, -1.0, 1.0)), round(rng.vec(24, -1.0, 1.0)));
        let slice: Vec<f32> =
            (0..n).flat_map(|i| full[(i * c_all + c0) * hw..(i * c_all + c0 + c) * hw].to_vec()).collect();
        for i in 0..n {
            full[(i * c_all + c0 + c) * hw..(i + 1) * c_all * hw].fill(f32::NAN);
        }
        let want = cpu_conv(&d, &slice, &wt, Some(&b));
        let up = |v: &[f32]| match dtype {
            DType::F32 => GpuTensor::from_slice(&ctx, v, &[v.len()]).unwrap(),
            DType::F16 => GpuTensor::from_f16(&ctx, &to_f16(v), &[v.len()]).unwrap(),
        };
        let (gx, gw, gb) = (up(&full), up(&wt), up(&b));
        let gy = GpuTensor::empty(&ctx, &[want.len()], dtype).unwrap();
        let at = conv::ConvInput { offset: c0 * hw, n_stride: c_all * hw };
        conv::conv2d_input(&ctx, &d, (gx.raw(), at), gw.raw(), Some(gb.raw()), gy.raw(), cfg).unwrap();
        let got = gy.to_vec().unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "{dtype:?}: NaN from past the slice in a ragged-K conv");
        let (atol, rtol) = if dtype == DType::F16 { (3e-3, 2e-3) } else { (2e-5 * 45f32.sqrt(), 1e-5) };
        assert_close(
            &got,
            &want,
            atol,
            rtol,
            &format!("ragged-K {dtype:?} conv on a slice before NaN channels"),
        );
    }
}

#[test]
fn conv_both_tile_configs() {
    for cfg in
        [TileConfig::SMALL, TileConfig::MEDIUM, TileConfig::LARGE, TileConfig::ROWS32, TileConfig::ROWS16]
    {
        let mut d = desc(2, 16, 12, 12, 20, 1, Window2d::square(3, 1, 1));
        d.act = UnaryOp::Relu;
        run(&d, true, Some(cfg), 7);
        run(&desc(1, 8, 16, 16, 128, 1, Window2d::square(3, 1, 1)), false, Some(cfg), 8);
        run(&desc(2, 12, 6, 6, 9, 1, Window2d::square(1, 1, 0)), true, Some(cfg), 9);
        // folo-n's first two convs: the 3-channel stem (K = 27) and 16 -> 32, both 3x3 stride 2.
        run(&desc(2, 3, 18, 18, 16, 1, Window2d::square(3, 2, 1)), true, Some(cfg), 10);
        run(&desc(2, 16, 14, 14, 32, 1, Window2d::square(3, 2, 1)), true, Some(cfg), 11);
    }
}

#[test]
fn conv_split_k() {
    // Its own context: the chunk counts come from a trace, which other tests' dispatches would pollute.
    if super::ctx().is_none() {
        return;
    }
    let ctx = GpuContext::with_options(&GpuOptions::from_env()).unwrap();
    let plan = |tiles, splits| ConvPlan { tiles, splits: Some(splits) };
    let run = |d: &Conv2dDesc, bias, plan, seed| run_plan(&ctx, d, bias, plan, seed, true);
    // Chunks per request (also checked against `split::plan`; pinned so a planner change shows up here): BK
    // rounding gives fewer than asked when `K` is short.
    for (i, (splits, chunks)) in [(2, [2; 6]), (3, [3; 6]), (8, [3, 3, 6, 7, 7, 5])].into_iter().enumerate() {
        let seed = 30 + i as u64 * 10;
        let mut got = Vec::new();
        // An odd plane (scalar finish), K = 45 (no multiple of any BK), stride 2, groups.
        let mut d = desc(2, 5, 13, 13, 7, 1, Window2d::square(3, 2, 1));
        d.act = UnaryOp::Silu;
        got.push(run(&d, true, plan(None, splits), seed));
        let d = desc(2, 8, 9, 9, 12, 2, Window2d::square(3, 1, 1));
        got.push(run(&d, true, plan(None, splits), seed + 1));
        // A plane of 8x8 (vec4 finish), M and N no tile multiples, no bias.
        let d = desc(3, 32, 8, 8, 40, 1, Window2d::square(3, 1, 1));
        got.push(run(&d, false, plan(None, splits), seed + 2));
        for cfg in [TileConfig::LARGE, TileConfig::ROWS16] {
            let d = desc(1, 24, 10, 10, 16, 1, Window2d::square(3, 2, 1));
            got.push(run(&d, true, plan(Some(cfg), splits), seed + 3));
        }
        // f16 operands with f32 partial slabs (the regular kernels: an explicit tile config).
        let mut d = desc(2, 16, 7, 7, 24, 1, Window2d::square(3, 1, 1));
        d.dtype = DType::F16;
        d.act = UnaryOp::Silu;
        got.push(run(&d, true, plan(Some(TileConfig::MEDIUM), splits), seed + 4));
        // 0: skipped (no shader-f16).
        let same = got.iter().zip(&chunks).all(|(&g, &c)| g == c || g == 0);
        assert!(same, "chunks for {splits} requested: {got:?}, pinned {chunks:?}");
    }
    // Requests of 8 that do give 8 chunks: K = 576 is 72 BK-8 tiles (8 chunks of 9) or 36 BK-16 tiles (8 of 5,
    // the last one short).
    for (i, cfg) in [TileConfig::LARGE, TileConfig::SMALL].into_iter().enumerate() {
        let d = desc(1, 64, 6, 6, 24, 1, Window2d::square(3, 1, 1));
        assert_eq!(run(&d, true, plan(Some(cfg), 8), 60 + i as u64), 8);
    }
    // K = 1 (a 1x1 stride-2 conv of one channel) cannot split: the request is clamped.
    let d = desc(2, 1, 9, 9, 4, 1, Window2d::square(1, 2, 0));
    assert_eq!(run(&d, true, plan(None, 4), 70), 1);
    // A 20 px layer with few tiles splits on its own: SMALL tiles, 7 of them, K = 576 in 4 chunks of 9 tiles.
    let mut d = desc(1, 64, 20, 20, 64, 1, Window2d::square(3, 1, 1));
    d.act = UnaryOp::Silu;
    assert_eq!(run(&d, true, ConvPlan::default(), 71), 4);
    // The same layer on a channel slice read in place (channels 16..80 of 96, two images): split-K reads `B` at
    // the slice's offset and image stride.
    let mut rng = Rng::new(72);
    let (n, c_all, c0, hw) = (2, 96, 16, 400);
    let full = rng.vec(n * c_all * hw, -1.0, 1.0);
    let (w, b) = (rng.vec(64 * 64 * 9, -1.0, 1.0), rng.vec(64, -1.0, 1.0));
    let slice: Vec<f32> =
        (0..n).flat_map(|i| full[(i * c_all + c0) * hw..(i * c_all + c0 + 64) * hw].to_vec()).collect();
    let mut d = desc(n, 64, 20, 20, 64, 1, Window2d::square(3, 1, 1));
    d.act = UnaryOp::Silu;
    let want = cpu_conv(&d, &slice, &w, Some(&b));
    let up = |v: &[f32]| GpuTensor::from_slice(&ctx, v, &[v.len()]).unwrap();
    let (gx, gw, gb) = (up(&full), up(&w), up(&b));
    let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F32).unwrap();
    let at = conv::ConvInput { offset: c0 * hw, n_stride: c_all * hw };
    let chunks = traced_splits(&ctx, || {
        conv::conv2d_input(&ctx, &d, (gx.raw(), at), gw.raw(), Some(gb.raw()), gy.raw(), None).unwrap();
    });
    assert_eq!(chunks, 4);
    let atol = 2e-5 * (64.0f32 * 9.0).sqrt();
    assert_close(&gy.to_vec().unwrap(), &want, atol, 1e-5, "split-K conv on a channel slice");
}

#[test]
fn conv_groups_and_pointwise() {
    run(&desc(2, 6, 7, 7, 4, 2, Window2d::square(3, 1, 1)), true, None, 10);
    run(&desc(1, 8, 5, 6, 12, 4, Window2d::square(3, 2, 0)), false, None, 11);
    // Pointwise: odd plane (scalar GEMM loads) and a multiple-of-4 plane (vec4), with groups.
    run(&desc(2, 16, 7, 7, 24, 1, Window2d::square(1, 1, 0)), true, None, 12);
    run(&desc(2, 16, 8, 8, 24, 1, Window2d::square(1, 1, 0)), true, None, 13);
    run(&desc(2, 16, 8, 8, 24, 2, Window2d::square(1, 1, 0)), true, None, 14);
}

#[test]
fn conv_depthwise() {
    let mut d = desc(2, 8, 9, 10, 8, 8, Window2d::square(3, 1, 1));
    d.act = UnaryOp::Hardswish;
    run(&d, true, None, 15);
    run(&desc(2, 8, 9, 10, 16, 8, Window2d::square(3, 2, 1)), true, None, 16);
    run(&desc(1, 4, 11, 11, 4, 4, Window2d::square(5, 1, 2)), false, None, 17);
    let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
    run(&desc(1, 3, 8, 8, 3, 3, dil), true, None, 18);

    // Tiled shared-memory + vec4 depthwise forward coverage: 2x2, 3x3, 5x5, 7x7, asymmetric 3x5 / 5x3,
    // exact & ragged tile boundaries (16x32 tiles), vec4-aligned and odd widths, prime channels, MULT=2,
    // tiny non-tiled fallback (3x3 plane), stride-2 vec4 output, and f16 parity.
    let tiled_cases = [
        (desc(2, 7, 17, 36, 7, 7, Window2d::square(3, 1, 1)), true, UnaryOp::Silu),
        (desc(2, 5, 17, 33, 5, 5, Window2d::square(3, 1, 1)), true, UnaryOp::Relu),
        (desc(2, 4, 20, 32, 8, 4, Window2d::square(5, 1, 2)), true, UnaryOp::Silu),
        (desc(2, 3, 18, 35, 3, 3, Window2d::square(7, 1, 3)), false, UnaryOp::Identity),
        (desc(2, 6, 32, 64, 6, 6, Window2d::square(7, 1, 3)), true, UnaryOp::Silu),
        (desc(1, 4, 9, 8, 4, 4, Window2d::square(2, 1, 0)), true, UnaryOp::Hardswish),
        (
            desc(
                2,
                5,
                19,
                36,
                5,
                5,
                Window2d { kernel: (3, 5), stride: (1, 1), padding: (1, 2), dilation: (1, 1) },
            ),
            true,
            UnaryOp::Silu,
        ),
        (
            desc(
                1,
                4,
                17,
                33,
                4,
                4,
                Window2d { kernel: (5, 3), stride: (1, 1), padding: (2, 1), dilation: (1, 1) },
            ),
            false,
            UnaryOp::Identity,
        ),
        // Non-tiled fallback: tiny 3x3 plane, stride-2 with ow % 4 == 0 (vec4 store), and 9x9 kernel.
        (desc(2, 4, 3, 3, 4, 4, Window2d::square(3, 1, 1)), true, UnaryOp::Silu),
        (desc(2, 4, 16, 16, 4, 4, Window2d::square(3, 2, 1)), true, UnaryOp::Silu),
        (desc(1, 3, 11, 12, 3, 3, Window2d::square(9, 1, 4)), false, UnaryOp::Identity),
    ];
    for (i, (mut c, bias, act)) in tiled_cases.into_iter().enumerate() {
        c.act = act;
        run(&c, bias, None, 110 + i as u64);
        let mut cf16 = c;
        cf16.dtype = DType::F16;
        run(&cf16, bias, None, 130 + i as u64);
    }

    // Padded border taps must be skipped even when weights contain +Inf (0 * +Inf must not contaminate
    // padded outputs with NaN). Use c_in = c_out = groups = 2 so `is_depthwise()` routes to the tiled kernel.
    let ctx = gpu!();
    for (h, w) in [(4usize, 4usize), (4usize, 5usize)] {
        let d = desc(1, 2, h, w, 2, 2, Window2d::square(3, 1, 1));
        assert!(d.is_depthwise());
        let x = vec![1.0f32; 2 * h * w];
        let gx = GpuTensor::from_slice(&ctx, &x, &[x.len()]).unwrap();
        for (tap, label) in [(1usize, "top-center"), (3usize, "left-center")] {
            let mut wt = vec![0.0f32; 2 * 9];
            wt[tap] = f32::INFINITY;
            wt[9 + tap] = f32::INFINITY;
            let want = cpu_conv(&d, &x, &wt, None);
            let gw = GpuTensor::from_slice(&ctx, &wt, &[wt.len()]).unwrap();
            let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F32).unwrap();
            conv::conv2d(&ctx, &d, gx.raw(), gw.raw(), None, gy.raw()).unwrap();
            let got = gy.to_vec().unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g.is_infinite() && w.is_infinite() && g.is_sign_positive() == w.is_sign_positive())
                        || (g == 0.0 && w == 0.0),
                    "depthwise tiled {h}x{w} non-finite {label} tap mismatch at {i}: got={g}, want={w}"
                );
            }
        }
    }
}

#[test]
fn conv_f16() {
    for (d, bias) in [
        (desc(2, 8, 10, 10, 16, 1, Window2d::square(3, 1, 1)), true),
        (desc(1, 5, 9, 9, 7, 1, Window2d::square(3, 2, 1)), false),
        (desc(2, 16, 8, 8, 24, 1, Window2d::square(1, 1, 0)), true),
        (desc(2, 8, 9, 9, 8, 8, Window2d::square(3, 1, 1)), true),
        // Depthwise: stride 2, a width that is no multiple of the 4 outputs per invocation, channel multiplier 2.
        (desc(2, 8, 13, 11, 8, 8, Window2d::square(3, 2, 1)), false),
        (desc(1, 6, 7, 18, 12, 6, Window2d::square(5, 1, 2)), true),
    ] {
        let mut d = d;
        d.dtype = DType::F16;
        d.act = UnaryOp::Silu;
        run(&d, bias, None, 19);
    }
}

#[test]
fn conv_tensor_api() {
    let ctx = gpu!();
    let mut rng = Rng::new(20);
    let x = rng.vec(2 * 4 * 6 * 6, -1.0, 1.0);
    let w = rng.vec(8 * 4 * 3 * 3, -1.0, 1.0);
    let b = rng.vec(8, -1.0, 1.0);
    let gx = GpuTensor::from_slice(&ctx, &x, &[2, 4, 6, 6]).unwrap();
    let gw = GpuTensor::from_slice(&ctx, &w, &[8, 4, 3, 3]).unwrap();
    let gb = GpuTensor::from_slice(&ctx, &b, &[8]).unwrap();
    let y = gx.conv2d(&gw, Some(&gb), (2, 2), (1, 1), (1, 1), 1, UnaryOp::Silu).unwrap();
    assert_eq!(y.shape(), &[2, 8, 3, 3]);
    let mut d = desc(2, 4, 6, 6, 8, 1, Window2d::square(3, 2, 1));
    d.act = UnaryOp::Silu;
    assert_close(&y.to_vec().unwrap(), &cpu_conv(&d, &x, &w, Some(&b)), 1e-4, 1e-5, "conv2d tensor");
    // Mismatched groups are rejected.
    assert!(gx.conv2d(&gw, None, (1, 1), (0, 0), (1, 1), 2, UnaryOp::Identity).is_err());
}

#[test]
fn conv_rejects_bad_configs() {
    let ctx = gpu!();
    let buf = GpuTensor::zeros(&ctx, &[4096], DType::F32).unwrap();
    let d = desc(1, 4, 8, 8, 8, 1, Window2d::square(3, 1, 1));
    let bad = TileConfig { bm: 36, bn: 144, bk: 16, tm: 8, tn: 4 };
    assert!(conv::conv2d_with_config(&ctx, &d, buf.raw(), buf.raw(), None, buf.raw(), Some(bad)).is_err());
    let d = desc(1, 1 << 20, 1 << 20, 1 << 20, 8, 1, Window2d::square(3, 1, 1));
    assert!(conv::conv2d(&ctx, &d, buf.raw(), buf.raw(), None, buf.raw()).is_err());
}

fn cpu_conv_grads_f32(d: &Conv2dDesc, x: &[f32], w: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
    let (mut gx, mut gw) = (vec![0f64; x.len()], vec![0f64; w.len()]);
    for n in 0..d.n {
        for co in 0..d.c_out {
            let grp = co / cout_g;
            for oy in 0..oh {
                for ox in 0..ow {
                    let gv = g[((n * d.c_out + co) * oh + oy) * ow + ox] as f64;
                    for ci in 0..cin_g {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * d.win.stride.0 + ky * d.win.dilation.0) as isize
                                    - d.win.padding.0 as isize;
                                let ix = (ox * d.win.stride.1 + kx * d.win.dilation.1) as isize
                                    - d.win.padding.1 as isize;
                                if iy < 0 || ix < 0 || iy >= d.h as isize || ix >= d.w as isize {
                                    continue;
                                }
                                let xi =
                                    ((n * d.c_in + grp * cin_g + ci) * d.h + iy as usize) * d.w + ix as usize;
                                let wi = ((co * cin_g + ci) * kh + ky) * kw + kx;
                                gx[xi] += gv * w[wi] as f64;
                                gw[wi] += gv * x[xi] as f64;
                            }
                        }
                    }
                }
            }
        }
    }
    (gx.into_iter().map(|v| v as f32).collect(), gw.into_iter().map(|v| v as f32).collect())
}

/// Regular f32 backward formulas: fused 2×2 sub-pixel stride-2 3×3 `dX`, stride-1 `dx_tap_mask` + `ROWS16`/`ROWS32`,
/// division-free `vec4` per-image split-K 1×1 `dW`, and `conv2d_wgrad_reduce`.
#[test]
fn conv_f32_backward_formulas() {
    let ctx = gpu!();
    let s2 = |ph, pw| Window2d { kernel: (3, 3), stride: (2, 2), padding: (ph, pw), dilation: (1, 1) };
    let k4x8 = Window2d { kernel: (4, 8), stride: (1, 1), padding: (1, 3), dilation: (1, 1) };
    let cases = [
        // Sub-pixel stride-2 3×3 (vec4 aligned, UW=10 wrap, odd H with CHECK=false, odd W scalar, paddings, groups).
        desc(2, 16, 20, 20, 32, 1, s2(1, 1)),
        desc(2, 16, 15, 16, 32, 1, s2(1, 1)),
        desc(2, 12, 13, 17, 20, 1, s2(1, 1)),
        desc(2, 8, 12, 12, 16, 1, s2(0, 0)),
        desc(2, 8, 13, 15, 12, 1, s2(2, 1)),
        desc(2, 16, 16, 20, 24, 2, s2(1, 1)),
        // Stride-1 dx_tap_mask + ROWS16/ROWS32 (3x3, 5x5, 4x8 exact 32-tap limit).
        desc(2, 16, 16, 16, 32, 1, Window2d::square(3, 1, 1)),
        desc(2, 32, 13, 15, 32, 1, Window2d::square(3, 1, 1)),
        desc(1, 8, 11, 12, 16, 1, Window2d::square(5, 1, 2)),
        desc(1, 4, 10, 11, 12, 1, k4x8),
        // Pointwise 1×1 division-free split-K dW (narrow PW32/PW32X64/PW64X32, wide, single-img, grouped, odd).
        desc(1, 32, 16, 16, 32, 1, Window2d::square(1, 1, 0)),
        desc(4, 32, 20, 20, 64, 1, Window2d::square(1, 1, 0)),
        desc(4, 64, 20, 20, 32, 1, Window2d::square(1, 1, 0)),
        desc(4, 48, 32, 32, 64, 1, Window2d::square(1, 1, 0)),
        desc(4, 32, 24, 24, 64, 2, Window2d::square(1, 1, 0)),
        desc(3, 5, 11, 13, 7, 1, Window2d::square(1, 1, 0)),
    ];
    for (i, d) in cases.iter().enumerate() {
        let mut rng = Rng::new(700 + i as u64);
        let (oh, ow) = d.output().unwrap();
        let (kh, kw) = d.win.kernel;
        let x = rng.vec(d.n * d.c_in * d.h * d.w, -1.0, 1.0);
        let w = rng.vec(d.c_out * (d.c_in / d.groups) * kh * kw, -1.0, 1.0);
        let g = rng.vec(d.n * d.c_out * oh * ow, -1.0, 1.0);
        let (want_x, want_w) = cpu_conv_grads_f32(d, &x, &w, &g);
        let gx = GpuTensor::from_slice(&ctx, &x, &[x.len()]).unwrap();
        let gw = GpuTensor::from_slice(&ctx, &w, &[w.len()]).unwrap();
        let gg = GpuTensor::from_slice(&ctx, &g, &[g.len()]).unwrap();
        let dx = GpuTensor::empty(&ctx, &[x.len()], DType::F32).unwrap();
        let dw = GpuTensor::empty(&ctx, &[w.len()], DType::F32).unwrap();
        conv::conv2d_backward_input(&ctx, d, gg.raw(), gw.raw(), dx.raw()).unwrap();
        conv::conv2d_backward_weight(&ctx, d, gx.raw(), gg.raw(), dw.raw()).unwrap();
        let kx = (d.c_out / d.groups * kh * kw) as f32;
        let kw_ = (d.n * oh * ow) as f32;
        assert_close(&dx.to_vec().unwrap(), &want_x, 2e-5 * kx.sqrt(), 1e-5, &format!("f32 dX {d:?}"));
        assert_close(&dw.to_vec().unwrap(), &want_w, 2e-5 * kw_.sqrt(), 1e-5, &format!("f32 dW {d:?}"));
    }
    assert!(ctx.kernels_without_registry().is_empty(), "{:?}", ctx.kernels_without_registry());
}
