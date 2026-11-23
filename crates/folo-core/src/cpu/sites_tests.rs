//! `conv2d_act_at` on the CPU: at every site, bit for bit what the dense inference conv (`conv2d_act`) computes,
//! for the Winograd tile list, the pointwise gathered GEMM and the dense fallback; and the tile list writes its
//! own tiles only.

use super::conv::{
    ConvGeom, ConvInput, ConvPart, transform_weights, transform_weights4, winograd_calls, winograd_forward,
    winograd_forward_at, winograd4_calls, winograd4_forward, winograd4_forward_at,
};
use crate::rng::Rng;
use crate::{Conv2dParams, Tensor, UnaryOp, no_grad};

const P3: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [1, 1], dilation: [1, 1], groups: 1 };
const P1: Conv2dParams = Conv2dParams { stride: [1, 1], padding: [0, 0], dilation: [1, 1], groups: 1 };

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// Sites of an `[n, *, oh, ow]` output: every corner and border midpoint of every image, plus `extra` random ones.
fn sites(n: usize, oh: usize, ow: usize, extra: usize, rng: &mut Rng) -> Vec<u32> {
    let mut v = Vec::new();
    for img in 0..n {
        for (y, x) in
            [(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1), (oh / 2, 0), (0, ow / 2), (oh - 1, ow / 2)]
        {
            v.push((img * oh * ow + y * ow + x) as u32);
        }
    }
    v.extend(
        (0..extra).map(|_| (rng.uniform(0.0, (n * oh * ow) as f32) as usize).min(n * oh * ow - 1) as u32),
    );
    v
}

/// `conv2d_act_at` (when `!tiled`) or `conv2d_act_at_tiles` (when `tiled`) against `conv2d_act` at `at`,
/// bit for bit, and the Winograd forwards it ran.
fn check_mode(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    p: &Conv2dParams,
    act: Option<UnaryOp>,
    at: &[u32],
    tiled: bool,
) -> usize {
    let _g = no_grad();
    // Held until the sites are checked: dropped, its buffer could come back from the pool as the sites' output,
    // already holding the dense values where the sites conv failed to write.
    let dense_out = x.conv2d_act(w, b, p, act).unwrap();
    let dense = bits(&dense_out);
    let before = winograd_calls()[1];
    let t = Tensor::from_vec(at.to_vec(), [at.len()]).unwrap();
    let y = if tiled {
        x.conv2d_act_at_tiles(w, b, p, act, &t).unwrap()
    } else {
        x.conv2d_act_at(w, b, p, act, &t).unwrap()
    };
    let ran = winograd_calls()[1] - before;
    let (_, o, oh, ow) = y.dims4().unwrap();
    assert_eq!(y.dims(), dense_out.dims());
    let y = bits(&y);
    for &s in at {
        let (img, pix) = (s as usize / (oh * ow), s as usize % (oh * ow));
        for oc in 0..o {
            let i = (img * o + oc) * oh * ow + pix;
            assert_eq!(y[i], dense[i], "tiled={tiled}, site {s}, channel {oc}");
        }
    }
    ran
}

/// `conv2d_act_at` against `conv2d_act` at `at`, bit for bit, and the Winograd forwards it ran.
fn check(
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    p: &Conv2dParams,
    act: Option<UnaryOp>,
    at: &[u32],
) -> usize {
    check_mode(x, w, b, p, act, at, false)
}

/// Winograd convs (both F(2x2) and F(4x4), odd sides, two images, single-tile-wide map, every site):
/// - `conv2d_act_at_tiles` gives the exact dense bits on both F(2x2) and F(4x4) shapes.
/// - `conv2d_act_at` stays on F(2x2) (never F(4x4)), matching dense bit-for-bit on F(2x2) shapes and
///   within Winograd tolerance on F(4x4) shapes.
#[test]
fn winograd_sites_are_the_dense_values() {
    let mut rng = Rng::seed(3);
    // `(n, c, o, h, w, expects_winograd4)`
    for (n, c, o, h, w, expect_f4) in [
        (2, 64, 64, 13, 11, false),
        (1, 96, 64, 20, 20, false),
        (2, 32, 32, 35, 37, false),
        (2, 64, 32, 35, 37, true),
        (1, 16, 32, 80, 44, true),
        (3, 64, 64, 1, 7, false),
        (1, 64, 72, 11, 11, false),
    ] {
        let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng);
        let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let at = sites(n, h, w, 9, &mut rng);
        let before4 = winograd4_calls()[1];
        assert_eq!(
            check_mode(&x, &wt, Some(&b), &P3, Some(UnaryOp::Silu), &at, true),
            1,
            "{n}x{c}x{h}x{w}: conv2d_act_at_tiles not Winograd"
        );
        let ran4 = winograd4_calls()[1] - before4;
        assert_eq!(ran4, usize::from(expect_f4) * 2, "{n}x{c}->{o}@{h}x{w}: unexpected F(4x4) dispatch");
        let all: Vec<u32> = (0..(n * h * w) as u32).rev().collect();
        assert_eq!(check_mode(&x, &wt, None, &P3, None, &all, true), 1);
        assert_eq!(check_mode(&x, &wt, Some(&b), &P3, None, &[], true), 1);

        // `conv2d_act_at` must NEVER run F(4x4): bit-exact on F(2x2) shapes, close on F(4x4) shapes.
        if !expect_f4 {
            let before4 = winograd4_calls()[1];
            assert_eq!(check(&x, &wt, Some(&b), &P3, Some(UnaryOp::Silu), &at), 1);
            assert_eq!(winograd4_calls()[1] - before4, 0);
        } else {
            let _g = no_grad();
            let dense =
                x.conv2d_act(&wt, Some(&b), &P3, Some(UnaryOp::Silu)).unwrap().to_vec::<f32>().unwrap();
            let at_t = Tensor::from_vec(at.clone(), [at.len()]).unwrap();
            let before = winograd_calls()[1];
            let before4 = winograd4_calls()[1];
            let y = x
                .conv2d_act_at(&wt, Some(&b), &P3, Some(UnaryOp::Silu), &at_t)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(winograd_calls()[1] - before, 1, "{n}x{c}->{o}@{h}x{w}: expected F(2x2)");
            assert_eq!(winograd4_calls()[1] - before4, 0, "conv2d_act_at must not run F(4x4)");
            for &s in &at {
                let (img, pix) = (s as usize / (h * w), s as usize % (h * w));
                for oc in 0..o {
                    let i = (img * o + oc) * h * w + pix;
                    assert!(y[i].is_finite() && (y[i] - dense[i]).abs() < 1e-4);
                }
            }
        }
    }
}

/// Receptive-field / patch isolation:
/// - `conv2d_act_at` preserves strict 3×3 receptive-field isolation on ALL shapes (including F(4x4) shapes like
///   `64->64 40x40`): changing an element outside `(4, 4)`'s 3×3 window (e.g., `(6, 6)` -> `1e20` or `NaN`)
///   leaves `(4, 4)` bit-identical and never runs F(4x4).
/// - `conv2d_act_at_tiles` isolates everything outside each requested 4×4 tile's 6×6 patch on F(4x4) shapes
///   and matches dense `conv2d_act` bit for bit.
#[test]
fn winograd_sites_isolate_outside_patch() {
    let mut rng = Rng::seed(11);
    let _g = no_grad();

    // 1. Strict 3×3 receptive-field isolation of `conv2d_act_at` on `64->64 40x40` (Sol R4 exact repro):
    // mutating input `(6, 6)` from `0.0` to `1e20` or `NaN` must leave output `(4, 4)` bit-identical.
    {
        let (c, o, h, w) = (64usize, 64usize, 40usize, 40usize);
        let mut x0 = vec![0.0f32; c * h * w];
        x0[4 * w + 4] = 1.0;
        let mut wt_vec = vec![0.0f32; o * c * 9];
        wt_vec[4] = 1.0 / 7.0; // center tap (1, 1) -> output (4, 4) = 1/7
        wt_vec[8] = 0.3; // corner tap (2, 2) so U[3, 3] != 0 (reads (5, 5) = 0 for site (4, 4))
        let wt = Tensor::from_vec(wt_vec, [o, c, 3, 3]).unwrap();
        let site = (4 * w + 4) as u32;
        let at_t = Tensor::from_vec(vec![site], [1]).unwrap();

        let run_at = |x_vec: Vec<f32>| {
            let x = Tensor::from_vec(x_vec, [1, c, h, w]).unwrap();
            let before4 = winograd4_calls()[1];
            let out = bits(&x.conv2d_act_at(&wt, None, &P3, None, &at_t).unwrap());
            assert_eq!(winograd4_calls()[1] - before4, 0, "conv2d_act_at must never take F(4x4)");
            (0..o).map(|oc| out[oc * h * w + site as usize]).collect::<Vec<_>>()
        };
        let base = run_at(x0.clone());
        assert!((f32::from_bits(base[0]) - 1.0f32 / 7.0).abs() < 1e-6);
        for poison in [1e20_f32, f32::NAN] {
            let mut xp = x0.clone();
            xp[6 * w + 6] = poison;
            assert_eq!(run_at(xp), base, "conv2d_act_at leaked input (6,6)={poison} into site (4,4)");
        }
    }

    // 2. Full NaN-isolation outside the 3×3 window for `conv2d_act_at` (both F(2x2) and F(4x4) shapes) and
    // outside the 6×6 tile patch for `conv2d_act_at_tiles` (F(4x4) shape).
    for (c, o, h, w, tiled_f4) in
        [(64usize, 64usize, 20usize, 20usize, false), (64, 32, 35, 37, false), (64, 32, 35, 37, true)]
    {
        let x_vec = Tensor::randn([1, c, h, w], 0.0, 1.0, &mut rng).to_vec::<f32>().unwrap();
        let x_clean = Tensor::from_vec(x_vec.clone(), [1, c, h, w]).unwrap();
        let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let chosen = [(0usize, 0usize), (4, 4), (5, 7), (h - 1, w - 1)];
        let at_vec: Vec<u32> = chosen.iter().map(|&(y, x)| (y * w + x) as u32).collect();
        let at_t = Tensor::from_vec(at_vec.clone(), [at_vec.len()]).unwrap();
        let want = if tiled_f4 {
            bits(&x_clean.conv2d_act(&wt, Some(&b), &P3, Some(UnaryOp::Silu)).unwrap())
        } else {
            bits(&x_clean.conv2d_act_at(&wt, Some(&b), &P3, Some(UnaryOp::Silu), &at_t).unwrap())
        };
        let mut keep = vec![false; h * w];
        for &(y0, x0) in &chosen {
            let (y_lo, y_hi, x_lo, x_hi) = if tiled_f4 {
                let (ty, tx) = (y0 / 4, x0 / 4);
                (
                    (4 * ty).saturating_sub(1),
                    (4 * ty + 5).min(h),
                    (4 * tx).saturating_sub(1),
                    (4 * tx + 5).min(w),
                )
            } else {
                (y0.saturating_sub(1), (y0 + 2).min(h), x0.saturating_sub(1), (x0 + 2).min(w))
            };
            for y in y_lo..y_hi {
                keep[y * w + x_lo..y * w + x_hi].fill(true);
            }
        }
        let mut poisoned = vec![f32::NAN; c * h * w];
        for ci in 0..c {
            for p in 0..h * w {
                if keep[p] {
                    poisoned[ci * h * w + p] = x_vec[ci * h * w + p];
                }
            }
        }
        let x_nan = Tensor::from_vec(poisoned, [1, c, h, w]).unwrap();
        let before4 = winograd4_calls()[1];
        let y = if tiled_f4 {
            bits(&x_nan.conv2d_act_at_tiles(&wt, Some(&b), &P3, Some(UnaryOp::Silu), &at_t).unwrap())
        } else {
            bits(&x_nan.conv2d_act_at(&wt, Some(&b), &P3, Some(UnaryOp::Silu), &at_t).unwrap())
        };
        assert_eq!(winograd4_calls()[1] - before4, usize::from(tiled_f4));
        for &s in &at_vec {
            for oc in 0..o {
                let i = oc * h * w + s as usize;
                assert_eq!(y[i], want[i], "tiled_f4={tiled_f4} site={s} oc={oc}");
            }
        }
    }
}

/// Pointwise convs, with and without bias and SiLU, and with two images, give the dense bits at their sites.
#[test]
fn pointwise_sites_are_the_dense_values() {
    let mut rng = Rng::seed(4);
    for (n, c, o, h, w) in [(1, 64, 64, 20, 20), (2, 37, 13, 9, 7), (1, 300, 70, 5, 6)] {
        let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng);
        let wt = Tensor::randn([o, c, 1, 1], 0.0, 0.1, &mut rng);
        let b = Tensor::randn([o], 0.0, 0.1, &mut rng);
        let at = sites(n, h, w, 11, &mut rng);
        assert_eq!(check(&x, &wt, Some(&b), &P1, None, &at), 0);
        assert_eq!(check(&x, &wt, Some(&b), &P1, Some(UnaryOp::Silu), &at), 0);
        assert_eq!(check(&x, &wt, None, &P1, Some(UnaryOp::Relu), &at), 0);
    }
}

/// Convs neither path takes (stride 2, depthwise, a 3x3 conv too small for Winograd) run densely: same values.
#[test]
fn other_convs_fall_back_to_dense() {
    let mut rng = Rng::seed(5);
    let x = Tensor::randn([2, 16, 10, 9], 0.0, 1.0, &mut rng);
    let at = sites(2, 5, 5, 4, &mut rng);
    let s2 = Conv2dParams { stride: [2, 2], ..P3 };
    let w = Tensor::randn([8, 16, 3, 3], 0.0, 0.1, &mut rng);
    assert_eq!(check(&x, &w, None, &s2, Some(UnaryOp::Silu), &at), 0);
    let dw = Conv2dParams { groups: 16, ..P3 };
    let wd = Tensor::randn([16, 1, 3, 3], 0.0, 0.1, &mut rng);
    assert_eq!(check(&x, &wd, None, &dw, Some(UnaryOp::Silu), &sites(2, 10, 9, 4, &mut rng)), 0);
    assert_eq!(check(&x, &w, None, &P3, None, &sites(2, 10, 9, 4, &mut rng)), 0, "8x16 took Winograd");
}

/// Sites outside the output, and inputs that do not fit, are errors.
#[test]
fn bad_sites_are_errors() {
    let _g = no_grad();
    let x = Tensor::zeros([1, 64, 4, 4]);
    let w = Tensor::zeros([64, 64, 3, 3]);
    let at = Tensor::from_vec(vec![16u32], [1]).unwrap();
    assert!(x.conv2d_act_at(&w, None, &P3, None, &at).is_err(), "site 16 of a 4x4 output");
    let f = Tensor::zeros([1]);
    assert!(x.conv2d_act_at(&w, None, &P3, None, &f).is_err(), "F32 sites");
}

/// The tile list writes exactly its tiles, with the dense values, at every thread count (one block on one
/// thread, the two-pass split on three, many blocks on the full pool); every other element keeps its value.
#[test]
fn tile_list_writes_its_tiles_only() {
    let mut rng = Rng::seed(6);
    let (n, c, o, h, w) = (2, 64, 48, 15, 13);
    let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng).to_vec::<f32>().unwrap();
    let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng).to_vec::<f32>().unwrap();
    let g = ConvGeom { n, c, h, w, o, kh: 3, kw: 3, oh: h, ow: w, p: P3 };
    let u = transform_weights(&wt, o, c);
    let part = [ConvPart::contiguous(&x, c, h * w)];
    let input = ConvInput { parts: &part };
    let (tw, all) = (w.div_ceil(2), h.div_ceil(2) * w.div_ceil(2));
    let mut dense = vec![0f32; n * o * h * w];
    winograd_forward(input, &u, &mut dense, &g, None);
    // Image 0: a scattered list with runs, the last column and the last row; image 1: nothing.
    let list: Vec<u32> =
        vec![0, 1, 2, 5, (tw + 3) as u32, (2 * tw - 1) as u32, (all - 2) as u32, (all - 1) as u32];
    let lists = vec![list.clone(), Vec::new()];
    for threads in [1, 3, rayon::current_num_threads()] {
        let mut out = vec![f32::NAN; n * o * h * w];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| winograd_forward_at(input, &u, &mut out, &g, None, &lists));
        for (i, (&got, &want)) in out.iter().zip(&dense).enumerate() {
            let (img, pix) = (i / (o * h * w), i % (h * w));
            let tile = ((pix / w) / 2 * tw + (pix % w) / 2) as u32;
            if img == 0 && list.contains(&tile) {
                assert_eq!(got.to_bits(), want.to_bits(), "{threads} threads: element {i}");
            } else {
                assert!(got.is_nan(), "{threads} threads: element {i} outside the list was written");
            }
        }
    }
}

#[test]
fn tile_list4_writes_its_tiles_only() {
    let mut rng = Rng::seed(7);
    let (n, c, o, h, w) = (2, 32, 32, 29, 27);
    let x = Tensor::randn([n, c, h, w], 0.0, 1.0, &mut rng).to_vec::<f32>().unwrap();
    let wt = Tensor::randn([o, c, 3, 3], 0.0, 0.05, &mut rng).to_vec::<f32>().unwrap();
    let g = ConvGeom { n, c, h, w, o, kh: 3, kw: 3, oh: h, ow: w, p: P3 };
    let u = transform_weights4(&wt, o, c);
    let part = [ConvPart::contiguous(&x, c, h * w)];
    let input = ConvInput { parts: &part };
    let (tw, all) = (w.div_ceil(4), h.div_ceil(4) * w.div_ceil(4));
    let mut dense = vec![0f32; n * o * h * w];
    winograd4_forward(input, &wt, &u, &mut dense, &g, None);
    let list: Vec<u32> =
        vec![0, 1, 2, 4, (tw + 2) as u32, (2 * tw - 1) as u32, (all - 2) as u32, (all - 1) as u32];
    let lists = vec![list.clone(), Vec::new()];
    for threads in [1, 3, rayon::current_num_threads()] {
        let mut out = vec![f32::NAN; n * o * h * w];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| winograd4_forward_at(input, &wt, &u, &mut out, &g, None, &lists));
        for (i, (&got, &want)) in out.iter().zip(&dense).enumerate() {
            let (img, pix) = (i / (o * h * w), i % (h * w));
            let tile = ((pix / w) / 4 * tw + (pix % w) / 4) as u32;
            if img == 0 && list.contains(&tile) {
                assert_eq!(got.to_bits(), want.to_bits(), "{threads} threads: element {i}");
            } else {
                assert!(got.is_nan(), "{threads} threads: element {i} outside the F(4x4) list was written");
            }
        }
    }
}
