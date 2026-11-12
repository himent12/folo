//! Tensor-core conv2d (forward, gradients, parts and slices read in place) against the CPU reference, with the
//! error bound of [`super::coop`].

use super::coop::{CONFIGS, assert_within, bound, coop_ctx, upload};
use super::{Rng, f16_round, traced_splits};
use crate::kernels::conv::{self, Conv2dDesc, split};
use crate::kernels::coop::{self, CoopConfig};
use crate::kernels::pooling::Window2d;
use crate::{DType, GpuContext, GpuTensor, UnaryOp};

/// `let (ctx, _guard) = coop!();` — the tensor-core context (as [`coop_ctx`]) or an early return.
macro_rules! coop {
    () => {
        match coop_ctx() {
            Some(c) => c,
            None => return,
        }
    };
}

fn cpu_conv(d: &Conv2dDesc, x: &[f32], w: &[f32], bias: Option<&[f32]>, abs: bool) -> Vec<f32> {
    let (oh, ow) = d.output().unwrap();
    let (kh, kw) = d.win.kernel;
    let (cin_g, cout_g) = (d.c_in / d.groups, d.c_out / d.groups);
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
                                let p = x[xi] as f64 * w[((co * cin_g + ci) * kh + ky) * kw + kx] as f64;
                                s += if abs { p.abs() } else { p };
                            }
                        }
                    }
                    let i = ((n * d.c_out + co) * oh + oy) * ow + ox;
                    y[i] = if abs { s as f32 } else { d.act.apply(s as f32 + bias.map_or(0.0, |b| b[co])) };
                }
            }
        }
    }
    y
}

#[allow(clippy::too_many_arguments)]
fn cdesc(
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    groups: usize,
    win: Window2d,
) -> Conv2dDesc {
    Conv2dDesc { n, c_in, h, w, c_out, groups, win, act: UnaryOp::Silu, dtype: DType::F16 }
}

/// Runs `d` on tensor cores: through `conv::conv2d`'s routing (`out == F16`, `cfg == None`) and explicitly.
fn run_conv(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    out: DType,
    cfg: Option<CoopConfig>,
    seed: u64,
) {
    run_conv_split(ctx, d, with_bias, out, cfg, seed, None);
}

/// [`run_conv`] with the routed conv split into `splits` K chunks (`None`: picked from the shape). Returns the
/// chunks the routed and the explicit conv ran as (0: not run), after checking them against [`split::plan`].
fn run_conv_split(
    ctx: &GpuContext,
    d: &Conv2dDesc,
    with_bias: bool,
    out: DType,
    cfg: Option<CoopConfig>,
    seed: u64,
    splits: Option<usize>,
) -> [usize; 2] {
    let mut rng = Rng::new(seed);
    let (kh, kw) = d.win.kernel;
    let x = f16_round(&rng.vec(d.n * d.c_in * d.h * d.w, -1.0, 1.0));
    let w = f16_round(&rng.vec(d.c_out * (d.c_in / d.groups) * kh * kw, -1.0, 1.0));
    let mut b = rng.vec(d.c_out, -1.0, 1.0);
    if out == DType::F16 {
        b = f16_round(&b);
    }
    let want = cpu_conv(d, &x, &w, with_bias.then_some(b.as_slice()), false);
    let sums = cpu_conv(d, &x, &w, None, true);
    let k = d.c_in / d.groups * kh * kw;
    let bmax = if with_bias { b.iter().fold(0f32, |m, v| m.max(v.abs())) } else { 0.0 };
    let tol: Vec<f64> = want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, k, bmax, out)).collect();
    let (gx, gw, gb) = (upload(ctx, &x, DType::F16), upload(ctx, &w, DType::F16), upload(ctx, &b, out));
    let gy = GpuTensor::empty(ctx, &[want.len()], out).unwrap();
    let bias = with_bias.then(|| gb.raw());
    let mut chunks = [0; 2];
    if out == DType::F16 && cfg.is_none() {
        let plan = conv::ConvPlan { tiles: None, splits };
        chunks[0] = traced_splits(ctx, || {
            conv::conv2d_with_plan(ctx, d, gx.raw(), gw.raw(), bias, gy.raw(), plan).unwrap();
        });
        assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
        assert_eq!(
            chunks[0],
            planned_splits(d, None, splits),
            "chunks of routed conv {d:?} splits {splits:?}"
        );
        let routed = gy.to_vec().unwrap();
        assert_within(&routed, &want, &tol, &format!("routed conv {d:?} bias={with_bias} splits {splits:?}"));
    }
    if !d.is_depthwise() {
        chunks[1] = traced_splits(ctx, || {
            coop::conv2d(ctx, d, out, gx.raw(), gw.raw(), bias, gy.raw(), cfg).unwrap();
        });
        assert_eq!(chunks[1], planned_splits(d, cfg, None), "chunks of coop conv {d:?} cfg {cfg:?}");
        let got = gy.to_vec().unwrap();
        assert_within(
            &got,
            &want,
            &tol,
            &format!("coop conv {d:?} bias={with_bias} out {out:?} cfg {cfg:?}"),
        );
    }
    chunks
}

/// The `K` chunks [`split::plan`] gives the tensor-core implicit GEMM of `d` with `cfg` (picked when `None`) and
/// `want` chunks requested (1 for the other paths).
fn planned_splits(d: &Conv2dDesc, cfg: Option<CoopConfig>, want: Option<usize>) -> usize {
    if d.is_depthwise() || d.is_pointwise() {
        return 1;
    }
    let (oh, ow) = d.output().unwrap();
    let (m, n, k, batch) = gemm_dims(d);
    let cfg = cfg.unwrap_or_else(|| picked_config(d));
    let tiles = m.div_ceil(cfg.bm as usize) * n.div_ceil(cfg.bn as usize);
    split::plan(cfg.bk as usize, k, tiles, batch, d.n * d.c_out * oh * ow, want).0
}

/// `(M, N, K, batch)` of the implicit GEMM of `d`.
fn gemm_dims(d: &Conv2dDesc) -> (usize, usize, usize, usize) {
    let (oh, ow) = d.output().unwrap();
    let k = d.c_in / d.groups * d.win.kernel.0 * d.win.kernel.1;
    (d.c_out / d.groups, oh * ow, k, d.n * d.groups)
}

/// The tile config the tensor-core forward conv picks for `d`.
fn picked_config(d: &Conv2dDesc) -> CoopConfig {
    let (m, n, _, batch) = gemm_dims(d);
    coop::pick_config(m, n, batch)
}

#[test]
fn coop_conv_variants() {
    let (ctx, _g) = coop!();
    let s1 = Window2d::square(3, 1, 1);
    let s2 = Window2d::square(3, 2, 1);
    for out in [DType::F16, DType::F32] {
        // folo-n / s: 3×3 s1 p1 and 3×3 s2 p1 (C3k2 / downsampling), the 3-channel stem, 1×1 (pointwise GEMM).
        run_conv(&ctx, &cdesc(2, 16, 20, 20, 32, 1, s1), true, out, None, 1);
        run_conv(&ctx, &cdesc(2, 16, 13, 17, 32, 1, s2), true, out, None, 2);
        run_conv(&ctx, &cdesc(1, 3, 33, 31, 16, 1, s2), true, out, None, 3);
        run_conv(&ctx, &cdesc(2, 24, 7, 9, 40, 1, Window2d::square(1, 1, 0)), true, out, None, 4);
        run_conv(&ctx, &cdesc(1, 64, 20, 20, 64, 1, Window2d::square(1, 1, 0)), true, out, None, 5);
        // Ragged channel counts (partial M tiles), batches, no bias.
        run_conv(&ctx, &cdesc(3, 20, 11, 10, 70, 1, s1), false, out, None, 6);
        run_conv(&ctx, &cdesc(1, 32, 9, 9, 130, 1, s1), true, out, None, 7);
        // Everything else the implicit GEMM covers: 1×1 stride 2, dilation, rectangular kernels, groups.
        run_conv(&ctx, &cdesc(2, 6, 9, 9, 4, 1, Window2d::square(1, 2, 0)), true, out, None, 8);
        let dil = Window2d { kernel: (3, 3), stride: (1, 1), padding: (2, 2), dilation: (2, 2) };
        run_conv(&ctx, &cdesc(1, 8, 12, 9, 24, 1, dil), true, out, None, 9);
        let rect = Window2d { kernel: (1, 3), stride: (2, 1), padding: (0, 1), dilation: (1, 1) };
        run_conv(&ctx, &cdesc(2, 5, 7, 8, 12, 1, rect), true, out, None, 10);
        run_conv(&ctx, &cdesc(2, 16, 10, 10, 32, 2, s1), true, out, None, 11);
        // A 7x7 stride-2 stem: 49 taps, more than the 32-bit tap mask holds (the bounds-checked gather).
        run_conv(&ctx, &cdesc(1, 3, 17, 17, 8, 1, Window2d::square(7, 2, 3)), true, out, None, 13);
    }
    // Depthwise (the head's and the attention's DWConv) keeps its direct kernel with tensor cores on.
    run_conv(&ctx, &cdesc(2, 32, 10, 10, 32, 32, s1), true, DType::F16, None, 12);
}

#[test]
fn coop_conv_split_k() {
    let (ctx, _g) = coop!();
    let s1 = Window2d::square(3, 1, 1);
    let run =
        |d: &Conv2dDesc, bias, out, seed, splits| run_conv_split(&ctx, d, bias, out, None, seed, splits);
    // [routed, explicit] chunks per request (also checked against `split::plan`; pinned so a planner change shows
    // up here). The explicit conv picks its own count; BK-32 rounding gives fewer than asked when `K` is short.
    for (splits, chunks) in
        [(2, [[2, 1], [2, 1], [2, 2]]), (3, [[2, 1], [3, 1], [3, 2]]), (8, [[2, 1], [5, 1], [6, 2]])]
    {
        let seed = 30 + splits as u64;
        // An odd plane (scalar finish), K = 45 (no multiple of BK), stride 2, ragged M; groups; K = 576.
        let got = [
            run(&cdesc(2, 5, 13, 13, 7, 1, Window2d::square(3, 2, 1)), true, DType::F16, seed, Some(splits)),
            run(&cdesc(2, 32, 8, 8, 40, 2, s1), false, DType::F16, seed + 10, Some(splits)),
            run(&cdesc(1, 64, 10, 10, 130, 1, s1), true, DType::F16, seed + 20, Some(splits)),
        ];
        assert_eq!(got, chunks, "chunks for {splits} requested");
    }
    // A request of 8 that does give 8 chunks: K = 1152 is 36 BK-32 tiles, 8 chunks of 5 (the last one short); the
    // explicit conv (one 64x64 tile) picks 4.
    assert_eq!(run(&cdesc(1, 128, 6, 6, 40, 1, s1), true, DType::F16, 39, Some(8)), [8, 4]);
    // K = 1 cannot split (the request is clamped).
    assert_eq!(
        run(&cdesc(2, 1, 9, 9, 4, 1, Window2d::square(1, 2, 0)), true, DType::F16, 40, Some(4)),
        [1, 1]
    );
    // A 20 px layer with few tiles splits on its own (7 64x64 tiles, K = 576 in 2 chunks), also with an f32
    // output (explicit conv only).
    for (out, chunks) in [(DType::F16, [2, 2]), (DType::F32, [0, 2])] {
        assert_eq!(run(&cdesc(1, 64, 20, 20, 64, 1, s1), true, out, 41, None), chunks);
    }
}

/// The tap mask's edge windows (`CONV_MODE` 1, one bit per tap), routed and explicit, f16 and f32 output: exactly 32
/// taps (4×8: bit 31 is the last tap, inside the image for most pixels) and 5×5 (25 taps), not depthwise.
#[test]
fn coop_conv_tap_mask_windows() {
    let (ctx, _g) = coop!();
    let k4x8 = Window2d { kernel: (4, 8), stride: (1, 1), padding: (1, 3), dilation: (1, 1) };
    let k4x8_s2 = Window2d { kernel: (4, 8), stride: (2, 2), padding: (2, 4), dilation: (1, 1) };
    for out in [DType::F16, DType::F32] {
        run_conv(&ctx, &cdesc(2, 3, 9, 13, 6, 1, k4x8), true, out, None, 90);
        run_conv(&ctx, &cdesc(1, 4, 11, 12, 40, 1, k4x8_s2), true, out, None, 91);
        run_conv(&ctx, &cdesc(2, 4, 11, 11, 24, 1, Window2d::square(5, 1, 2)), true, out, None, 92);
        run_conv(&ctx, &cdesc(1, 5, 12, 10, 8, 1, Window2d::square(5, 2, 2)), false, out, None, 93);
    }
}

/// A forward conv whose `M`, `N` and `K` are all tile multiples, so its kernel has no bounds checks (`CHECK` false:
/// no `k < K` test in the gather), whole and in 2 `K` chunks: 32 → 128 channels at 16×16, 3×3.
#[test]
fn coop_conv_tile_multiples() {
    let (ctx, _g) = coop!();
    let d = cdesc(1, 32, 16, 16, 128, 1, Window2d::square(3, 1, 1));
    let (m, n, k, batch) = gemm_dims(&d);
    let cfg = picked_config(&d);
    let (bm, bn, bk) = (cfg.bm as usize, cfg.bn as usize, cfg.bk as usize);
    assert!(m % bm == 0 && n % bn == 0 && k % bk == 0, "{cfg:?} leaves a ragged tile of {m}x{n}x{k}");
    // 2 chunks of 5 and 4 BK tiles: the chunk is a tile multiple too.
    let tiles = (m / bm) * (n / bn);
    assert_eq!(split::plan(bk, k, tiles, batch, m * n, Some(2)), (2, 5 * bk));
    for out in [DType::F16, DType::F32] {
        let routed = if out == DType::F16 { 1 } else { 0 };
        assert_eq!(run_conv_split(&ctx, &d, true, out, None, 94, None), [routed, 1]);
    }
    assert_eq!(run_conv_split(&ctx, &d, true, DType::F16, None, 95, Some(2)), [2, 1]);
}

/// A forward conv whose `K` is no tile multiple (45, `BK` 32) on a channel slice read in place, NaN in the channels
/// right after the slice, finite weights, no activation: the gather must stop at `K` (reading on would take the next
/// channel, and 0 × NaN is NaN).
#[test]
fn coop_conv_ragged_k_slice_before_nan() {
    let (ctx, _g) = coop!();
    let mut rng = Rng::new(96);
    // Channels 3..8 of 12: 3 finite noise channels before the slice, 4 NaN after it.
    let (n, c, c_all, c0, h, w) = (2, 5, 12, 3, 7, 9);
    let hw = h * w;
    let mut d = cdesc(n, c, h, w, 24, 1, Window2d::square(3, 1, 1));
    d.act = UnaryOp::Identity;
    let (_, _, k, _) = gemm_dims(&d);
    assert!(!k.is_multiple_of(picked_config(&d).bk as usize));
    let x = f16_round(&rng.vec(n * c * hw, -1.0, 1.0));
    let wt = f16_round(&rng.vec(24 * c * 9, -1.0, 1.0));
    let b = f16_round(&rng.vec(24, -1.0, 1.0));
    let want = cpu_conv(&d, &x, &wt, Some(&b), false);
    let sums = cpu_conv(&d, &x, &wt, None, true);
    let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    let tol: Vec<f64> =
        want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, k, bmax, DType::F16)).collect();
    let mut full = embed(&x, n, c, hw, c_all, c0, &mut rng);
    for i in 0..n {
        full[(i * c_all + c0 + c) * hw..(i + 1) * c_all * hw].fill(f32::NAN);
    }
    let gx = upload(&ctx, &full, DType::F16);
    let (gw, gb) = (upload(&ctx, &wt, DType::F16), upload(&ctx, &b, DType::F16));
    let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F16).unwrap();
    let at = conv::ConvInput { offset: c0 * hw, n_stride: c_all * hw };
    conv::conv2d_input(&ctx, &d, (gx.raw(), at), gw.raw(), Some(gb.raw()), gy.raw(), None).unwrap();
    assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
    let got = gy.to_vec().unwrap();
    assert!(got.iter().all(|v| v.is_finite()), "NaN from past the slice in a ragged-K tensor-core conv");
    assert_within(&got, &want, &tol, "ragged-K tensor-core conv on a slice before NaN channels");
}

#[test]
fn coop_conv_every_config() {
    let (ctx, _g) = coop!();
    for cfg in CONFIGS {
        run_conv(
            &ctx,
            &cdesc(2, 16, 12, 12, 40, 1, Window2d::square(3, 1, 1)),
            true,
            DType::F16,
            Some(cfg),
            20,
        );
        run_conv(
            &ctx,
            &cdesc(1, 8, 16, 16, 128, 1, Window2d::square(3, 2, 1)),
            false,
            DType::F32,
            Some(cfg),
            21,
        );
        run_conv(&ctx, &cdesc(2, 12, 6, 6, 9, 1, Window2d::square(1, 1, 0)), true, DType::F16, Some(cfg), 22);
    }
}

/// The routed conv (Conv + bias + SiLU in one kernel, as folo-nn's fused `ConvBnAct` runs it) agrees with the
/// regular f16 kernel, and switching tensor cores off gives the regular kernel's result again.
#[test]
fn conv_routing_matches_regular_kernel() {
    let (ctx, _g) = coop!();
    let d = cdesc(2, 32, 20, 20, 64, 1, Window2d::square(3, 1, 1));
    let mut rng = Rng::new(50);
    let x = upload(&ctx, &f16_round(&rng.vec(2 * 32 * 400, -1.0, 1.0)), DType::F16);
    let w = upload(&ctx, &f16_round(&rng.vec(64 * 32 * 9, -0.2, 0.2)), DType::F16);
    let b = upload(&ctx, &f16_round(&rng.vec(64, -0.5, 0.5)), DType::F16);
    let run = |on: bool| {
        ctx.set_tensor_cores(on);
        let y = GpuTensor::empty(&ctx, &[2 * 64 * 400], DType::F16).unwrap();
        conv::conv2d(&ctx, &d, x.raw(), w.raw(), Some(b.raw()), y.raw()).unwrap();
        y.to_vec().unwrap()
    };
    let (tc, plain, plain2) = (run(true), run(false), run(false));
    ctx.set_tensor_cores(true);
    assert_eq!(plain, plain2);
    // Both are f16 roundings of nearly the same f32 sum: at most a couple of f16 ulps apart.
    for (i, (&p, &t)) in plain.iter().zip(&tc).enumerate() {
        assert!((p - t).abs() <= 2e-3 * (1.0 + p.abs()), "[{i}] regular {p} vs tensor cores {t}");
    }
}

// ----- conv gradients (mixed-precision training) ----------------------------------------------------------------

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

/// The stem's weight gradient (3 → 16, 3×3 s2) on the regular kernels' 16×32 tile, with tensor cores on too:
/// `N·OH·OW = 4·80·80 = 25600`, dozens of split-K chunks, exact sums so a dropped chunk fails.
#[test]
fn coop_conv_weight_gradient_stem_split_k() {
    let (ctx, _g) = coop!();
    let d = cdesc(4, 3, 160, 160, 16, 1, Window2d::square(3, 2, 1));
    for route in conv_grads(&ctx, &d, 200, true) {
        assert!(matches!(route, Some((false, (16, 32), s)) if s >= 40), "stem route {route:?}");
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

/// `x` (`[n, c, hw]`) placed at channel `at` of a larger `[n, big, hw]` buffer of noise: a channel slice.
fn embed(x: &[f32], n: usize, c: usize, hw: usize, big: usize, at: usize, rng: &mut Rng) -> Vec<f32> {
    let mut out = rng.vec(n * big * hw, -1.0, 1.0);
    for i in 0..n {
        let (src, dst) = (i * c * hw, (i * big + at) * hw);
        out[dst..dst + c * hw].copy_from_slice(&x[src..src + c * hw]);
    }
    out
}

/// The 1×1 conv over a concatenation read in place (`conv::conv2d_cat`) and convs on a channel slice
/// (`conv::conv2d_input`) on tensor cores: 2 to 4 parts, slices and whole tensors, batches, odd sizes.
#[test]
fn coop_conv_reads_parts_and_slices_in_place() {
    let (ctx, _g) = coop!();
    let mut rng = Rng::new(77);
    for (n, h, w) in [(1, 20, 20), (2, 5, 7), (2, 4, 4)] {
        let hw = h * w;
        for chans in [&[16usize, 16][..], &[8, 5, 12], &[16, 32, 16, 16]] {
            let c: usize = chans.iter().sum();
            let d = cdesc(n, c, h, w, 24, 1, Window2d::square(1, 1, 0));
            // Each part on its own; even parts are channel slices of a larger buffer (3 noise channels first).
            let xs: Vec<Vec<f32>> =
                chans.iter().map(|&ci| f16_round(&rng.vec(n * ci * hw, -1.0, 1.0))).collect();
            let mut cat = vec![0f32; n * c * hw];
            for i in 0..n {
                let mut ch = 0;
                for (x, &ci) in xs.iter().zip(chans.iter()) {
                    cat[(i * c + ch) * hw..(i * c + ch + ci) * hw]
                        .copy_from_slice(&x[i * ci * hw..(i + 1) * ci * hw]);
                    ch += ci;
                }
            }
            let wt = f16_round(&rng.vec(24 * c, -1.0, 1.0));
            let b = f16_round(&rng.vec(24, -1.0, 1.0));
            let want = cpu_conv(&d, &cat, &wt, Some(&b), false);
            let sums = cpu_conv(&d, &cat, &wt, None, true);
            let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
            let tol: Vec<f64> =
                want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, c, bmax, DType::F16)).collect();
            let mut bufs = Vec::new();
            let mut ats = Vec::new();
            for (j, (x, &ci)) in xs.iter().zip(chans.iter()).enumerate() {
                if j % 2 == 0 {
                    let big = ci + 5;
                    bufs.push(upload(&ctx, &embed(x, n, ci, hw, big, 3, &mut rng), DType::F16));
                    ats.push(conv::ConvInput { offset: 3 * hw, n_stride: big * hw });
                } else {
                    bufs.push(upload(&ctx, x, DType::F16));
                    ats.push(conv::ConvInput { offset: 0, n_stride: ci * hw });
                }
            }
            let parts: Vec<conv::ConvPart<'_>> = bufs
                .iter()
                .zip(&ats)
                .zip(chans.iter())
                .map(|((g, &at), &channels)| conv::ConvPart { buf: g.raw(), channels, at })
                .collect();
            let (gw, gb) = (upload(&ctx, &wt, DType::F16), upload(&ctx, &b, DType::F16));
            let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F16).unwrap();
            conv::conv2d_cat(&ctx, &d, &parts, gw.raw(), Some(gb.raw()), gy.raw()).unwrap();
            assert!(ctx.tensor_cores(), "{}", ctx.tensor_cores_status());
            assert_within(
                &gy.to_vec().unwrap(),
                &want,
                &tol,
                &format!("coop conv_cat n{n} {h}x{w} {chans:?}"),
            );
        }
    }
    // A channel slice as the input of a 3×3 and a 1×1 conv.
    for (k, pad) in [(3, 1), (1, 0)] {
        let d = cdesc(2, 16, 9, 10, 32, 1, Window2d::square(k, 1, pad));
        let hw = 90;
        let x = f16_round(&rng.vec(2 * 16 * hw, -1.0, 1.0));
        let wt = f16_round(&rng.vec(32 * 16 * k * k, -1.0, 1.0));
        let b = f16_round(&rng.vec(32, -1.0, 1.0));
        let want = cpu_conv(&d, &x, &wt, Some(&b), false);
        let sums = cpu_conv(&d, &x, &wt, None, true);
        let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tol: Vec<f64> =
            want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, 16 * k * k, bmax, DType::F16)).collect();
        let gx = upload(&ctx, &embed(&x, 2, 16, hw, 24, 4, &mut rng), DType::F16);
        let (gw, gb) = (upload(&ctx, &wt, DType::F16), upload(&ctx, &b, DType::F16));
        let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F16).unwrap();
        let at = conv::ConvInput { offset: 4 * hw, n_stride: 24 * hw };
        conv::conv2d_input(&ctx, &d, (gx.raw(), at), gw.raw(), Some(gb.raw()), gy.raw(), None).unwrap();
        assert_within(&gy.to_vec().unwrap(), &want, &tol, &format!("coop conv on a slice k{k}"));
    }
    // A channel slice into a conv that splits K (64 -> 40 channels at 10 px, K = 576: 2 chunks), no bias: the
    // partial slabs read B at the slice's offset and image stride.
    let d = cdesc(2, 64, 10, 10, 40, 1, Window2d::square(3, 1, 1));
    let hw = 100;
    let x = f16_round(&rng.vec(2 * 64 * hw, -1.0, 1.0));
    let wt = f16_round(&rng.vec(40 * 64 * 9, -1.0, 1.0));
    let want = cpu_conv(&d, &x, &wt, None, false);
    let sums = cpu_conv(&d, &x, &wt, None, true);
    let tol: Vec<f64> =
        want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, 64 * 9, 0.0, DType::F16)).collect();
    let gx = upload(&ctx, &embed(&x, 2, 64, hw, 96, 16, &mut rng), DType::F16);
    let gw = upload(&ctx, &wt, DType::F16);
    let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F16).unwrap();
    let at = conv::ConvInput { offset: 16 * hw, n_stride: 96 * hw };
    let chunks = traced_splits(&ctx, || {
        conv::conv2d_input(&ctx, &d, (gx.raw(), at), gw.raw(), None, gy.raw(), None).unwrap();
    });
    assert_eq!(chunks, planned_splits(&d, None, None));
    assert!(chunks > 1, "the slice case must split K, ran {chunks} chunk(s)");
    assert_within(&gy.to_vec().unwrap(), &want, &tol, "coop split-K conv on a slice");
}
