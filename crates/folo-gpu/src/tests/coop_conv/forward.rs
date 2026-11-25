//! Tensor-core forward conv2d tests: variants, split-K, tap masks, slices/concatenation, and tile selection.

use super::{
    cdesc, coop, cpu_conv, embed, gemm_dims, picked_config, planned_splits, run_conv, run_conv_split,
    traced_tile,
};
use crate::kernels::conv::{self, Conv2dDesc, split};
use crate::kernels::coop::{self, CoopConfig};
use crate::kernels::pooling::Window2d;
use crate::tests::coop::{CONFIGS, assert_within, bound, upload};
use crate::tests::{Rng, f16_round, traced_splits};
use crate::{DType, GpuTensor, UnaryOp};

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

/// The forward 1×1 conv (routed and explicit without a config) and the 1×1 conv over a concatenation read in place
/// run the forward pick's tiles, against the reference: 64×64 for 64 channels at 64×64 px (batch 4, where the GEMM
/// pick takes 64×128) and 64×128 for 128 channels at 80×80 px (6400 pixels; the GEMM pick takes 64×64).
#[test]
fn coop_pointwise_convs_take_the_forward_tiles() {
    let (ctx, _g) = coop!();
    let mut rng = Rng::new(97);
    let (c0, c1) = (4, 8);
    for (n, size, c_out, cfg) in [(4, 64, 64, CoopConfig::C64X64), (1, 80, 128, CoopConfig::C64X128)] {
        let hw = size * size;
        assert_eq!(coop::pick_config_conv(c_out, hw), cfg);
        assert_ne!(coop::pick_config(c_out, hw, n), cfg, "a shape where the GEMM pick differs");
        let d = cdesc(n, c0 + c1, size, size, c_out, 1, Window2d::square(1, 1, 0));
        let x = f16_round(&rng.vec(n * (c0 + c1) * hw, -1.0, 1.0));
        let wt = f16_round(&rng.vec(c_out * (c0 + c1), -1.0, 1.0));
        let b = f16_round(&rng.vec(c_out, -1.0, 1.0));
        let want = cpu_conv(&d, &x, &wt, Some(&b), false);
        let sums = cpu_conv(&d, &x, &wt, None, true);
        let bmax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tol: Vec<f64> =
            want.iter().zip(&sums).map(|(&y, &s)| bound(y, s as f64, c0 + c1, bmax, DType::F16)).collect();
        let (gx, gw, gb) =
            (upload(&ctx, &x, DType::F16), upload(&ctx, &wt, DType::F16), upload(&ctx, &b, DType::F16));
        let check = |what: &str, encode: &dyn Fn(&wgpu::Buffer)| {
            let gy = GpuTensor::empty(&ctx, &[want.len()], DType::F16).unwrap();
            assert_eq!(traced_tile(&ctx, || encode(gy.raw())), (cfg.bm, cfg.bn), "tile of the {what} {d:?}");
            assert_within(&gy.to_vec().unwrap(), &want, &tol, &format!("{what} {d:?}"));
        };
        check("routed 1x1 conv", &|y| {
            conv::conv2d(&ctx, &d, gx.raw(), gw.raw(), Some(gb.raw()), y).unwrap();
        });
        check("tensor-core 1x1 conv", &|y| {
            coop::conv2d(&ctx, &d, DType::F16, gx.raw(), gw.raw(), Some(gb.raw()), y, None).unwrap();
        });
        // The same input as two parts: its first channels as a slice of a larger buffer, the rest whole.
        let part = |from: usize, c: usize| -> Vec<f32> {
            (0..n)
                .flat_map(|i| x[(i * (c0 + c1) + from) * hw..(i * (c0 + c1) + from + c) * hw].to_vec())
                .collect()
        };
        let g0 = upload(&ctx, &embed(&part(0, c0), n, c0, hw, c0 + 5, 3, &mut rng), DType::F16);
        let g1 = upload(&ctx, &part(c0, c1), DType::F16);
        let parts = [
            conv::ConvPart {
                buf: g0.raw(),
                channels: c0,
                at: conv::ConvInput { offset: 3 * hw, n_stride: (c0 + 5) * hw },
            },
            conv::ConvPart {
                buf: g1.raw(),
                channels: c1,
                at: conv::ConvInput { offset: 0, n_stride: c1 * hw },
            },
        ];
        check("1x1 conv over a concatenation", &|y| {
            conv::conv2d_cat(&ctx, &d, &parts, gw.raw(), Some(gb.raw()), y).unwrap();
        });
    }
}
