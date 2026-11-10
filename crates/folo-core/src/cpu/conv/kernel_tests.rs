use super::im2col::Im2Col;
use super::shifted::shifted_scratch;
use super::small_c::uses_small_c;
use super::*;
use crate::backend::UnaryOp;

fn data(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn geom(n: usize, c: usize, hw: [usize; 2], o: usize, k: [usize; 2], p: Conv2dParams) -> ConvGeom {
    let out = |i: usize, k: usize| (hw[i] + 2 * p.padding[i] - p.dilation[i] * (k - 1) - 1) / p.stride[i] + 1;
    let (oh, ow) = (out(0, k[0]), out(1, k[1]));
    ConvGeom { n, c, h: hw[0], w: hw[1], o, kh: k[0], kw: k[1], oh, ow, p }
}

fn params(s: [usize; 2], pad: [usize; 2], d: [usize; 2], groups: usize) -> Conv2dParams {
    Conv2dParams { stride: s, padding: pad, dilation: d, groups }
}

/// `(n, c, [h, w], o, [kh, kw], params)`.
type Case = (usize, usize, [usize; 2], usize, [usize; 2], Conv2dParams);

/// The conv in f64, per output element: `(sum, sum of |terms|)`.
fn reference(x: &[f32], wt: &[f32], g: &ConvGeom) -> Vec<(f64, f64)> {
    let (cg, og) = (g.cg(), g.og());
    let mut out = Vec::with_capacity(g.n * g.o * g.ohw());
    for img in 0..g.n {
        for oc in 0..g.o {
            let grp = oc / og;
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let (mut s, mut a) = (0f64, 0f64);
                    for ci in 0..cg {
                        let ic = grp * cg + ci;
                        for ky in 0..g.kh {
                            for kx in 0..g.kw {
                                let iy = (oy * g.p.stride[0] + ky * g.p.dilation[0]) as isize
                                    - g.p.padding[0] as isize;
                                let ix = (ox * g.p.stride[1] + kx * g.p.dilation[1]) as isize
                                    - g.p.padding[1] as isize;
                                if iy < 0 || ix < 0 || iy >= g.h as isize || ix >= g.w as isize {
                                    continue;
                                }
                                let xv = x[((img * g.c + ic) * g.h + iy as usize) * g.w + ix as usize] as f64;
                                let wv = wt[((oc * cg + ci) * g.kh + ky) * g.kw + kx] as f64;
                                s += xv * wv;
                                a += (xv * wv).abs();
                            }
                        }
                    }
                    out.push((s, a));
                }
            }
        }
    }
    out
}

/// Runs `conv2d_forward` on every case with every epilogue (and accumulating into a prior output), against
/// the f64 reference. Tolerance: the f32 summation bound `K · 2^-23 · Σ|terms|` for a K-term dot product in
/// any order, carried through the activation (slope at most 1.1), plus SiLU's own relative error.
fn check(cases: &[Case]) {
    let bias_of = |o: usize| (0..o).map(|c| c as f32 * 0.25 - 1.0).collect::<Vec<f32>>();
    let silu = |v: f64| v / (1.0 + (-v).exp());
    for (i, &(n, c, hw, o, k, prm)) in cases.iter().enumerate() {
        let g = geom(n, c, hw, o, k, prm);
        let x = data(n * c * hw[0] * hw[1], 7 + i as u32);
        let wt = data(o * g.kdim(), 300 + i as u32);
        let want = reference(&x, &wt, &g);
        let bias = bias_of(o);
        let ohw = g.ohw();
        let kterms = g.kdim() as f64;
        for (e, act) in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)].into_iter().enumerate() {
            for with_bias in [false, true] {
                let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                let epi = (with_bias || act.is_some()).then_some(epi);
                let mut got = vec![f32::NAN; n * o * ohw];
                conv2d_forward(&x, &wt, &mut got, &g, false, epi);
                for (q, (&y, &(s, a))) in got.iter().zip(&want).enumerate() {
                    let b = if with_bias { bias[(q / ohw) % o] as f64 } else { 0.0 };
                    let t = s + b;
                    let (want, slope) = match act {
                        None => (t, 1.0),
                        Some(UnaryOp::Silu) => (silu(t), 1.1),
                        _ => (t.max(0.0), 1.0),
                    };
                    let tol = slope * kterms * 1.2e-7 * (a + b.abs()) + 5e-7 * want.abs() + 1e-7;
                    assert!(
                        (y as f64 - want).abs() <= tol,
                        "case {i} ({g:?}) epi {e}/{with_bias} at {q}: {y} vs {want} (tol {tol})"
                    );
                }
            }
        }
        // Accumulating into a prior output (no epilogue).
        let prior = data(n * o * ohw, 900 + i as u32);
        let mut got = prior.clone();
        conv2d_forward(&x, &wt, &mut got, &g, true, None);
        for (q, ((&y, &(s, a)), &p0)) in got.iter().zip(&want).zip(&prior).enumerate() {
            let want = s + p0 as f64;
            let tol = (kterms + 1.0) * 1.2e-7 * (a + (p0 as f64).abs()) + 1e-7;
            assert!((y as f64 - want).abs() <= tol, "case {i} accumulate at {q}: {y} vs {want}");
        }
    }
}

#[test]
fn depthwise_matches_reference() {
    // (n, c, [h, w], o, [kh, kw], params): the folo-n shapes, then rows narrower than one vector, ragged
    // tails (ow % 8 != 0, more than four vectors), taps that see only padding, dilation, a channel
    // multiplier, non-square kernels and the strided (non-fast) path.
    let cases = [
        (1, 64, [80, 80], 64, [3, 3], params([1, 1], [1, 1], [1, 1], 64)),
        (1, 256, [20, 20], 256, [3, 3], params([1, 1], [1, 1], [1, 1], 256)),
        (2, 3, [1, 1], 3, [3, 3], params([1, 1], [1, 1], [1, 1], 3)),
        (1, 2, [5, 7], 2, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
        (1, 3, [9, 37], 3, [3, 3], params([1, 1], [1, 1], [1, 1], 3)),
        (1, 2, [6, 41], 2, [5, 5], params([1, 1], [2, 2], [1, 1], 2)),
        (1, 2, [4, 3], 2, [5, 7], params([1, 1], [3, 4], [1, 1], 2)),
        (1, 3, [11, 19], 3, [3, 3], params([1, 1], [2, 2], [2, 2], 3)),
        (2, 2, [8, 9], 4, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
        (1, 4, [7, 13], 4, [1, 5], params([1, 1], [0, 2], [1, 1], 4)),
        (1, 3, [10, 10], 3, [3, 3], params([1, 1], [0, 0], [1, 1], 3)),
        (1, 3, [12, 17], 3, [3, 3], params([2, 2], [1, 1], [1, 1], 3)),
        (1, 2, [9, 8], 4, [3, 3], params([2, 1], [1, 1], [1, 1], 2)),
        // Zero-width inputs: every output is padding (then bias and activation).
        (1, 2, [3, 0], 2, [1, 1], params([1, 1], [0, 1], [1, 1], 2)),
        (2, 2, [3, 0], 4, [3, 3], params([1, 1], [1, 2], [1, 1], 2)),
    ];
    check(&cases);
}

#[test]
fn gemm_paths_match_reference() {
    let cases = [
        // 1x1 (packed B, rows read once into panels): ragged M and N, K over one K block (the epilogue
        // runs in the micro-kernel on the last one).
        (1, 20, [7, 9], 13, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
        (2, 300, [5, 7], 7, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
        // Shifted rows, stride 1: ragged tiles, K = 360 over one block, two groups.
        (2, 8, [11, 13], 10, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (1, 40, [6, 7], 8, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (1, 8, [9, 9], 6, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
        // Stride-2 phases (both column phases dealt out together), odd sizes, no / wide padding.
        (1, 6, [13, 15], 9, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
        (1, 5, [12, 9], 7, [3, 3], params([2, 2], [0, 0], [1, 1], 1)),
        (1, 6, [10, 11], 5, [5, 5], params([2, 2], [2, 2], [1, 1], 1)),
        // One column phase only (dilation 2 at stride 2), stride 3, and mixed strides.
        (1, 5, [15, 14], 4, [3, 3], params([2, 2], [2, 2], [2, 2], 1)),
        (1, 5, [16, 17], 6, [3, 3], params([3, 3], [1, 1], [1, 1], 1)),
        (1, 6, [12, 13], 5, [3, 3], params([2, 1], [1, 1], [1, 1], 1)),
        (1, 6, [9, 14], 5, [3, 3], params([1, 2], [1, 1], [1, 1], 1)),
        // The few-channel direct path.
        (1, 3, [17, 19], 11, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
    ];
    check(&cases);
}

/// The fused paths (shifted rows or a 1x1's planes packed row by row, bias and SiLU applied in the
/// micro-kernel) are bit for bit an unfused GEMM over an im2col B (packed panel by panel from
/// `im2col_segment`, the elementwise formula) followed by the row epilogue: the same K blocks, the same FMAs,
/// the same epilogue lanes. Covers ragged M and N, K over one `KC` block, every phase split and zero biases.
#[test]
fn fused_conv_is_bitwise_im2col_gemm_plus_row_epilogue() {
    let cases = [
        (1, 20, [7, 9], 13, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
        (2, 300, [5, 7], 7, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
        (2, 8, [11, 13], 10, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (1, 40, [6, 7], 8, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (1, 8, [9, 9], 6, [3, 3], params([1, 1], [1, 1], [1, 1], 2)),
        (1, 6, [13, 15], 9, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
        (1, 5, [12, 9], 7, [3, 3], params([2, 2], [0, 0], [1, 1], 1)),
        (1, 6, [10, 11], 5, [5, 5], params([2, 2], [2, 2], [1, 1], 1)),
        (1, 5, [15, 14], 4, [3, 3], params([2, 2], [2, 2], [2, 2], 1)),
        (1, 5, [16, 17], 6, [3, 3], params([3, 3], [1, 1], [1, 1], 1)),
        (1, 6, [12, 13], 5, [3, 3], params([2, 1], [1, 1], [1, 1], 1)),
        (1, 6, [9, 14], 5, [3, 3], params([1, 2], [1, 1], [1, 1], 1)),
    ];
    for (i, &(n, c, hw, o, k, prm)) in cases.iter().enumerate() {
        let g = geom(n, c, hw, o, k, prm);
        assert!(!g.is_depthwise() && !uses_small_c(&g));
        let x = data(n * c * hw[0] * hw[1], 40 + i as u32);
        let wt = data(o * g.kdim(), 500 + i as u32);
        // Every third bias is exactly zero.
        let bias: Vec<f32> = (0..o).map(|c| if c % 3 == 0 { 0.0 } else { c as f32 * 0.3 - 1.0 }).collect();
        let (cg, og, kdim, ohw, hwi) = (g.cg(), g.og(), g.kdim(), g.ohw(), g.h * g.w);
        for act in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)] {
            for with_bias in [false, true] {
                let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                let mut got = vec![f32::NAN; n * o * ohw];
                conv2d_forward(&x, &wt, &mut got, &g, false, Some(epi));
                let mut want = vec![f32::NAN; n * o * ohw];
                let part = [ConvPart::contiguous(&x, c, hwi)];
                let input = ConvInput { parts: &part };
                for (img, out_img) in want.chunks_mut(o * ohw).enumerate() {
                    let planes = input.planes(img, hwi);
                    for grp in 0..g.p.groups {
                        let a = Strided::lhs(&wt, grp * og * kdim, kdim, 1);
                        let b = Im2Col { planes: &planes[grp * cg..(grp + 1) * cg], g };
                        let cblk = &mut out_img[grp * og * ohw..(grp + 1) * og * ohw];
                        gemm_ex(og, ohw, kdim, &a, &b, cblk, ohw, false, 1, None);
                        for r in 0..og {
                            epi.rows_from(grp * og).apply(r, &mut cblk[r * ohw..(r + 1) * ohw]);
                        }
                    }
                }
                let same = got.iter().zip(&want).all(|(p, q)| p.to_bits() == q.to_bits());
                assert!(same, "case {i} ({g:?}) act {act:?} bias {with_bias}: not bit-identical");
            }
        }
    }
}

/// The forward, input gradient and weight gradient of `g` as bits, under the current scratch budget.
fn conv_bits(g: &ConvGeom) -> Vec<Vec<u32>> {
    let (x, wt) = (data(g.n * g.c * g.h * g.w, 1), data(g.o * g.kdim(), 2));
    let gy = data(g.n * g.o * g.ohw(), 3);
    let mut gw = vec![0f32; wt.len()];
    conv2d_backward_weight(&x, &gy, &mut gw, g);
    let mut y = vec![0f32; gy.len()];
    conv2d_forward(&x, &wt, &mut y, g, false, None);
    let mut gx = vec![0f32; x.len()];
    conv2d_backward_input(&gy, &wt, &mut gx, g);
    [gw, y, gx].into_iter().map(|v| v.iter().map(|f| f.to_bits()).collect()).collect()
}

/// `f` under scratch budget `floats` (restored after).
fn with_budget<T>(floats: usize, f: impl FnOnce() -> T) -> T {
    set_image_scratch_budget(floats);
    let r = f();
    set_image_scratch_budget(IMAGE_SCRATCH_BUDGET);
    r
}

/// The image scratch budget only changes how many images run at once, never a bit of the result: forward,
/// input gradient (shifted rows and col2im) and weight gradient (waves of images whose K parts cross wave
/// boundaries, each wave tiling its rows on its own) with no limit, the default, one image at a time and a
/// few at a time. The batch-64 cases have 128 K parts spread over many waves.
#[test]
fn image_scratch_budget_does_not_change_results() {
    let cases: [Case; 7] = [
        (5, 16, [21, 19], 24, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (5, 16, [21, 19], 24, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
        (4, 12, [17, 15], 8, [5, 5], params([1, 1], [4, 4], [2, 2], 2)),
        (3, 64, [9, 11], 24, [1, 1], params([1, 1], [0, 0], [1, 1], 1)),
        (6, 8, [30, 26], 16, [3, 3], params([3, 3], [2, 2], [1, 1], 1)),
        (64, 32, [40, 40], 32, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
        (64, 64, [20, 20], 64, [3, 3], params([1, 1], [1, 1], [1, 1], 1)),
    ];
    for (i, (n, c, hw, o, k, p)) in cases.into_iter().enumerate() {
        let g = geom(n, c, hw, o, k, p);
        let all = with_budget(usize::MAX, || conv_bits(&g));
        // One image's scratch is at least a few thousand floats in every case: 1 float runs them one at a
        // time, 3 copies' worth two or three at a time (a dozen for the batch-64 weight gradients).
        let one = shifted_scratch(&g).max(g.kdim() * g.ohw()).max(1);
        for budget in [IMAGE_SCRATCH_BUDGET, 1, 3 * one] {
            let got = with_budget(budget, || conv_bits(&g));
            assert!(got == all, "case {i}, budget {budget}: results differ");
        }
    }
}
