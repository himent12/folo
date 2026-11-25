//! Unit tests of the conv kernels: the depthwise, GEMM and Winograd paths against an `f64` reference, the fused
//! epilogue bit for bit against GEMM then epilogue, results independent of the image scratch budget, and
//! Winograd results the same for every thread count and batch.

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

/// `2^-23`, the unit of the tolerances (twice f32's unit roundoff).
const EPS: f64 = 1.0 / (1u64 << 23) as f64;

/// The case's pseudo-random inputs and weights.
fn case_data(i: usize, g: &ConvGeom) -> (Vec<f32>, Vec<f32>) {
    (data(g.n * g.c * g.h * g.w, 7 + i as u32), data(g.o * g.kdim(), 300 + i as u32))
}

/// The summation part of a forward's tolerance, from the inputs, the weights and the f64 reference
/// `(sum, Σ|terms|)`: per output element, the allowance before bias, and the factor of `|bias|` to add.
type Bound = fn(&[f32], &[f32], &ConvGeom, &[(f64, f64)]) -> (Vec<f64>, f64);

/// The direct conv: the f32 summation bound `K · 2^-23 · (Σ|terms| + |bias|)` of a K-term dot product in any order.
fn direct_bound(_: &[f32], _: &[f32], g: &ConvGeom, want: &[(f64, f64)]) -> (Vec<f64>, f64) {
    let k = g.kdim() as f64 * EPS;
    (want.iter().map(|&(_, a)| k * a).collect(), k)
}

/// Runs `conv2d_forward` on every case with every epilogue (and accumulating into a prior output), against
/// the f64 reference. Tolerance: [`direct_bound`], carried through the activation (slope at most 1.1), plus,
/// when an activation is applied, its own error (SiLU's relative error and an absolute floor).
fn check(cases: &[Case]) {
    check_with(
        cases,
        case_data,
        direct_bound,
        |x, wt, out, g, epi| conv2d_forward(x, wt, out, g, false, epi),
        true,
    );
}

/// [`check`] of `forward` (overwriting) on the inputs and weights `inputs` makes for each case, with the
/// summation tolerance `bound`; the accumulating case runs only if `accumulate`.
fn check_with(
    cases: &[Case],
    inputs: impl Fn(usize, &ConvGeom) -> (Vec<f32>, Vec<f32>),
    bound: Bound,
    forward: impl Fn(&[f32], &[f32], &mut [f32], &ConvGeom, Option<ChannelEpilogue<'_>>),
    accumulate: bool,
) {
    let bias_of = |o: usize| (0..o).map(|c| c as f32 * 0.25 - 1.0).collect::<Vec<f32>>();
    let silu = |v: f64| v / (1.0 + (-v).exp());
    for (i, &(n, c, hw, o, k, prm)) in cases.iter().enumerate() {
        let g = geom(n, c, hw, o, k, prm);
        let (x, wt) = inputs(i, &g);
        let want = reference(&x, &wt, &g);
        let (allow, bias_factor) = bound(&x, &wt, &g, &want);
        let bias = bias_of(o);
        let ohw = g.ohw();
        for (e, act) in [None, Some(UnaryOp::Silu), Some(UnaryOp::Relu)].into_iter().enumerate() {
            for with_bias in [false, true] {
                let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                let epi = (with_bias || act.is_some()).then_some(epi);
                let mut got = vec![f32::NAN; n * o * ohw];
                forward(&x, &wt, &mut got, &g, epi);
                for (q, ((&y, &(s, _)), &allow)) in got.iter().zip(&want).zip(&allow).enumerate() {
                    let b = if with_bias { bias[(q / ohw) % o] as f64 } else { 0.0 };
                    let t = s + b;
                    let (want, slope) = match act {
                        None => (t, 1.0),
                        Some(UnaryOp::Silu) => (silu(t), 1.1),
                        _ => (t.max(0.0), 1.0),
                    };
                    let act_err = if act.is_some() { 5e-7 * want.abs() + 1e-7 } else { 0.0 };
                    let tol = slope * (allow + bias_factor * b.abs()) + act_err;
                    assert!(
                        (y as f64 - want).abs() <= tol,
                        "case {i} ({g:?}) epi {e}/{with_bias} at {q}: {y} vs {want} (tol {tol})"
                    );
                }
            }
        }
        if !accumulate {
            continue;
        }
        // Accumulating into a prior output (no epilogue).
        let prior = data(n * o * ohw, 900 + i as u32);
        let mut got = prior.clone();
        conv2d_forward(&x, &wt, &mut got, &g, true, None);
        for (q, ((&y, &(s, a)), &p0)) in got.iter().zip(&want).zip(&prior).enumerate() {
            let want = s + p0 as f64;
            let tol = (g.kdim() as f64 + 1.0) * 1.2e-7 * (a + (p0 as f64).abs()) + 1e-7;
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

/// The Winograd cases: odd and even output sizes (a last tile row or column half outside the image), 1x1 and 1x2
/// images (one tile, mostly padding), no padding and wide padding (tiles that see only zeros), tiny and odd
/// channel counts, more input channels than one `KC` block, rows of more than one 64-tile segment, blocks that
/// end mid tile row, a batch, and folo-n shapes.
fn winograd_cases() -> Vec<Case> {
    let p = |pad: usize| params([1, 1], [pad, pad], [1, 1], 1);
    vec![
        (1, 1, [1, 1], 1, [3, 3], p(1)),
        (1, 2, [1, 2], 3, [3, 3], p(1)),
        (1, 3, [2, 2], 1, [3, 3], p(1)),
        (2, 5, [7, 9], 7, [3, 3], p(1)),
        (1, 4, [3, 3], 5, [3, 3], p(0)),
        (1, 4, [6, 5], 3, [3, 3], p(0)),
        (1, 3, [4, 7], 2, [3, 3], p(3)),
        (1, 7, [33, 35], 9, [3, 3], p(1)),
        (1, 2, [5, 300], 3, [3, 3], p(1)),
        (1, 300, [6, 7], 13, [3, 3], p(1)),
        (3, 32, [20, 20], 64, [3, 3], p(1)),
        (1, 64, [40, 40], 64, [3, 3], p(1)),
        (1, 16, [80, 80], 32, [3, 3], p(1)),
    ]
}

/// [`winograd_forward`] with freshly transformed weights.
fn winograd(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
    assert!(super::winograd::winograd_shape(g));
    let u = transform_weights(wt, g.o, g.c);
    let part = [ConvPart::contiguous(x, g.c, g.h * g.w)];
    winograd_forward(ConvInput { parts: &part }, &u, out, g, epi);
}

/// The Winograd F(2x2, 3x3) error bound per output element (module doc of `winograd.rs`):
/// `(C + 8) · 2^-23 · 9 · Σ_c D_c · G_c` before bias, with `D_c` the sum of `|x|` over the output's tile's 4x4
/// input patch in channel `c` (padding counts 0) and `G_c` the sum of `|w|` over its 3x3 kernel from channel `c`,
/// and `2^-23 · |bias|` for the bias add.
fn winograd_bound(x: &[f32], wt: &[f32], g: &ConvGeom, _: &[(f64, f64)]) -> (Vec<f64>, f64) {
    let (c, (th, tw)) = (g.c, (g.oh.div_ceil(2), g.ow.div_ceil(2)));
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let gsum: Vec<f64> = wt.chunks(9).map(|k| k.iter().map(|v| v.abs() as f64).sum()).collect();
    let mut out = Vec::with_capacity(g.n * g.o * g.ohw());
    for img in 0..g.n {
        // dsum[ci][tile]: Σ|x| over the tile's input patch.
        let mut dsum = vec![0f64; c * th * tw];
        for (q, d) in dsum.iter_mut().enumerate() {
            let (ci, ty, tx) = (q / (th * tw), q / tw % th, q % tw);
            for r in 0..4 {
                for s in 0..4 {
                    let (iy, ix) = ((2 * ty + r) as isize - ph, (2 * tx + s) as isize - pw);
                    if (0..g.h as isize).contains(&iy) && (0..g.w as isize).contains(&ix) {
                        *d += x[((img * c + ci) * g.h + iy as usize) * g.w + ix as usize].abs() as f64;
                    }
                }
            }
        }
        for oc in 0..g.o {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let t = oy / 2 * tw + ox / 2;
                    let s: f64 = (0..c).map(|ci| dsum[ci * th * tw + t] * gsum[oc * c + ci]).sum();
                    out.push((c as f64 + 8.0) * EPS * 9.0 * s);
                }
            }
        }
    }
    (out, EPS)
}

/// Winograd F(2x2, 3x3) against the f64 reference within [`winograd_bound`] (plus the activation's error when
/// one is applied), with every epilogue: on [`winograd_cases`], and with all-constant inputs (1.0) and weights
/// (0.1) over 512 channels, where every product has one sign and the error grows with the channel count (about
/// 20 · 2^-23 · Σ|terms|: a bound without a factor of C does not hold).
#[test]
fn winograd_matches_reference() {
    check_with(&winograd_cases(), case_data, winograd_bound, winograd, false);
    let constant = [(1, 512, [4, 4], 8, [3, 3], params([1, 1], [0, 0], [1, 1], 1))];
    let ones = |_: usize, g: &ConvGeom| (vec![1.0; g.n * g.c * g.h * g.w], vec![0.1; g.o * g.kdim()]);
    check_with(&constant, ones, winograd_bound, winograd, false);
}

/// Winograd results do not depend on the thread count (the blocks a pool size makes) or on the batch: one
/// thread, 3 and 28 threads, and each image of a batch alone, all bit for bit the same.
#[test]
fn winograd_is_deterministic() {
    for (i, (n, c, hw, o, k, prm)) in winograd_cases().into_iter().enumerate() {
        let g = geom(n, c, hw, o, k, prm);
        let x = data(n * c * hw[0] * hw[1], 70 + i as u32);
        let wt = data(o * g.kdim(), 700 + i as u32);
        let bias: Vec<f32> = (0..o).map(|c| c as f32 * 0.1 - 0.5).collect();
        let epi = Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Silu) });
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut y = vec![f32::NAN; n * o * g.ohw()];
            pool.install(|| winograd(&x, &wt, &mut y, &g, epi));
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let one = run(1);
        assert!(run(3) == one && run(28) == one, "case {i} ({g:?}): results depend on the thread count");
        let per = c * hw[0] * hw[1];
        let g1 = ConvGeom { n: 1, ..g };
        for img in 0..n {
            let mut y = vec![f32::NAN; o * g.ohw()];
            winograd(&x[img * per..(img + 1) * per], &wt, &mut y, &g1, epi);
            let bits: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
            assert!(
                bits[..] == one[img * o * g.ohw()..(img + 1) * o * g.ohw()],
                "case {i}: image {img} alone differs"
            );
        }
    }
}

/// [`winograd4_forward`] with freshly transformed weights.
fn winograd4(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
    assert!(super::winograd::winograd_shape(g));
    let u = transform_weights4(wt, g.o, g.c);
    let part = [ConvPart::contiguous(x, g.c, g.h * g.w)];
    winograd4_forward(ConvInput { parts: &part }, wt, &u, out, g, epi);
}

/// Well-conditioned Winograd F(4x4, 3x3) error bound (points `0, ±1/2, ±1, ∞`):
/// `(C + 16) · 2^-23 · 140 · Σ_c D_c · G_c` before bias, with `D_c` the sum of `|x|` over the output's 6x6
/// input patch in channel `c` and `G_c` the sum of `|w|` over its 3x3 kernel from channel `c`.
fn winograd4_bound(x: &[f32], wt: &[f32], g: &ConvGeom, _: &[(f64, f64)]) -> (Vec<f64>, f64) {
    let (c, (th, tw)) = (g.c, (g.oh.div_ceil(4), g.ow.div_ceil(4)));
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let gsum: Vec<f64> = wt.chunks(9).map(|k| k.iter().map(|v| v.abs() as f64).sum()).collect();
    let mut out = Vec::with_capacity(g.n * g.o * g.ohw());
    for img in 0..g.n {
        let mut dsum = vec![0f64; c * th * tw];
        for (q, d) in dsum.iter_mut().enumerate() {
            let (ci, ty, tx) = (q / (th * tw), q / tw % th, q % tw);
            for r in 0..6 {
                for s in 0..6 {
                    let (iy, ix) = ((4 * ty + r) as isize - ph, (4 * tx + s) as isize - pw);
                    if (0..g.h as isize).contains(&iy) && (0..g.w as isize).contains(&ix) {
                        *d += x[((img * c + ci) * g.h + iy as usize) * g.w + ix as usize].abs() as f64;
                    }
                }
            }
        }
        for oc in 0..g.o {
            for oy in 0..g.oh {
                for ox in 0..g.ow {
                    let t = oy / 4 * tw + ox / 4;
                    let s: f64 = (0..c).map(|ci| dsum[ci * th * tw + t] * gsum[oc * c + ci]).sum();
                    out.push((c as f64 + 16.0) * EPS * 140.0 * s);
                }
            }
        }
    }
    (out, EPS)
}

#[test]
fn winograd4_matches_reference() {
    check_with(&winograd_cases(), case_data, winograd4_bound, winograd4, false);
    let constant = [(1, 512, [6, 6], 8, [3, 3], params([1, 1], [0, 0], [1, 1], 1))];
    let ones = |_: usize, g: &ConvGeom| (vec![1.0; g.n * g.c * g.h * g.w], vec![0.1; g.o * g.kdim()]);
    check_with(&constant, ones, winograd4_bound, winograd4, false);
}

#[test]
fn winograd4_is_deterministic() {
    for (i, (n, c, hw, o, k, prm)) in winograd_cases().into_iter().enumerate() {
        let g = geom(n, c, hw, o, k, prm);
        let x = data(n * c * hw[0] * hw[1], 80 + i as u32);
        let wt = data(o * g.kdim(), 800 + i as u32);
        let bias: Vec<f32> = (0..o).map(|c| c as f32 * 0.1 - 0.5).collect();
        let epi = Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Silu) });
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut y = vec![f32::NAN; n * o * g.ohw()];
            pool.install(|| winograd4(&x, &wt, &mut y, &g, epi));
            y.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let one = run(1);
        assert!(
            run(3) == one && run(8) == one && run(28) == one,
            "case {i} ({g:?}): F(4x4) results depend on thread count"
        );
        let per = c * hw[0] * hw[1];
        let g1 = ConvGeom { n: 1, ..g };
        for img in 0..n {
            let mut y = vec![f32::NAN; o * g.ohw()];
            winograd4(&x[img * per..(img + 1) * per], &wt, &mut y, &g1, epi);
            let bits: Vec<u32> = y.iter().map(|v| v.to_bits()).collect();
            assert!(
                bits[..] == one[img * o * g.ohw()..(img + 1) * o * g.ohw()],
                "case {i}: F(4x4) image {img} alone differs"
            );
        }
    }
}

/// Direct AVX2/SSE == scalar bit-parity for Winograd F(4x4, 3x3) across ragged tile-segment lengths
/// (`1..=3`, `4..=7`, `8..=15`, `>= 16`), clipped right/bottom edges, and every epilogue combination.
#[test]
fn winograd4_avx2_matches_scalar_bit_for_bit() {
    let p = params([1, 1], [1, 1], [1, 1], 1);
    // Widths exercising 1..3 tiles, 4..7 tiles (SSE tail), 8..15 tiles (AVX2 + SSE/scalar), >= 16 tiles,
    // plus clipped right/bottom edges (`w % 4 != 0`, `h % 4 != 0`).
    let shapes = [
        (1, 4, [5, 5], 4),
        (1, 4, [7, 11], 6),
        (1, 8, [9, 15], 8),
        (1, 8, [13, 21], 8),
        (1, 8, [17, 27], 12),
        (1, 8, [19, 31], 8),
        (1, 8, [21, 35], 12),
        (1, 8, [25, 46], 8),
        (1, 8, [25, 50], 8),
        (1, 8, [11, 67], 8),
        (2, 16, [24, 80], 16),
    ];
    for (idx, (n, c, hw, o)) in shapes.into_iter().enumerate() {
        let g = geom(n, c, hw, o, [3, 3], p);
        let x = data(n * c * hw[0] * hw[1], 91 + idx as u32);
        let wt = data(o * g.kdim(), 910 + idx as u32);
        let u = transform_weights4(&wt, o, c);
        let part = [ConvPart::contiguous(&x, c, hw[0] * hw[1])];
        let input = ConvInput { parts: &part };
        let zeros = vec![0.0f32; o];
        let mut bias: Vec<f32> = (0..o).map(|k| k as f32 * 0.17 - 0.5).collect();
        if o > 1 {
            bias[1] = 0.0;
        }
        let epis = [
            None,
            Some(BiasAct { bias: Some(&zeros), act: None }),
            Some(BiasAct { bias: Some(&bias), act: None }),
            Some(BiasAct { bias: None, act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: Some(&zeros), act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: None, act: Some(UnaryOp::Relu) }),
            Some(BiasAct { bias: Some(&zeros), act: Some(UnaryOp::Relu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Relu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Sigmoid) }),
        ];
        for (ei, epi) in epis.into_iter().enumerate() {
            let mut got = vec![f32::NAN; n * o * g.ohw()];
            let mut want = vec![f32::NAN; n * o * g.ohw()];
            winograd4_forward(input, &wt, &u, &mut got, &g, epi);
            super::winograd4::winograd4_forward_scalar(input, &wt, &u, &mut want, &g, epi);
            let bad = got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(bad, None, "shape {idx} ({n}x{c}->{o}@{hw:?}) epi {ei}: AVX2 vs scalar bit mismatch");
        }
    }
}

/// Winograd F(2x2, 3x3) explicit AVX2 input/output transforms and fused register epilogues match the scalar
/// transform bit-for-bit across even/odd widths, segment boundaries, and every epilogue kind.
#[test]
fn winograd2_avx2_matches_scalar_bitwise_across_epilogues() {
    let p = params([1, 1], [1, 1], [1, 1], 1);
    let shapes = [
        (1, 16, [15, 15], 8),
        (1, 16, [16, 17], 12),
        (1, 16, [19, 31], 8),
        (1, 16, [20, 32], 12),
        (1, 16, [13, 35], 8),
        (2, 16, [25, 67], 16),
    ];
    for (idx, (n, c, hw, o)) in shapes.into_iter().enumerate() {
        let g = geom(n, c, hw, o, [3, 3], p);
        let x = data(n * c * hw[0] * hw[1], 101 + idx as u32);
        let wt = data(o * g.kdim(), 1010 + idx as u32);
        let u = transform_weights(&wt, o, c);
        let part = [ConvPart::contiguous(&x, c, hw[0] * hw[1])];
        let input = ConvInput { parts: &part };
        let zeros = vec![0.0f32; o];
        let mut bias: Vec<f32> = (0..o).map(|k| k as f32 * 0.17 - 0.5).collect();
        if o > 1 {
            bias[1] = 0.0;
        }
        let epis = [
            None,
            Some(BiasAct { bias: Some(&zeros), act: None }),
            Some(BiasAct { bias: Some(&bias), act: None }),
            Some(BiasAct { bias: None, act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Relu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::LeakyRelu(0.1)) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Hardswish) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Sigmoid) }),
        ];
        for (ei, epi) in epis.into_iter().enumerate() {
            let mut got = vec![f32::NAN; n * o * g.ohw()];
            let mut want = vec![f32::NAN; n * o * g.ohw()];
            winograd_forward(input, &u, &mut got, &g, epi);
            super::winograd::winograd_forward_scalar(input, &u, &mut want, &g, epi);
            let bad = got.iter().zip(&want).position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                bad, None,
                "winograd2 shape {idx} ({n}x{c}->{o}@{hw:?}) epi {ei}: AVX2 vs scalar mismatch"
            );
        }
    }
}

/// Depthwise backward input/weight and strided col2im backward input match naive f64 references across stride 1,
/// stride 2, dilation 2, multiplier > 1, and wide/ragged widths exercising AVX2 16-wide, 8-wide, and tail paths.
#[test]
fn depthwise_backward_and_col2im_match_naive() {
    let cases = [
        // Depthwise: stride 1 wide (AVX2 16+8+tail), stride 2, dilation 2, depth multiplier 2.
        (2, 8, [17, 37], 8, [3, 3], params([1, 1], [1, 1], [1, 1], 8)),
        (2, 6, [15, 29], 12, [3, 3], params([1, 1], [1, 1], [1, 1], 6)),
        (2, 8, [17, 35], 8, [3, 3], params([2, 2], [1, 1], [1, 1], 8)),
        (1, 4, [13, 25], 4, [3, 3], params([1, 1], [2, 2], [2, 2], 4)),
        // Non-depthwise col2im paths: stride 2 (standard), stride 1 with huge padding (p > d*(k-1) hits col2im!).
        (2, 8, [17, 35], 12, [3, 3], params([2, 2], [1, 1], [1, 1], 1)),
        (1, 6, [11, 29], 8, [3, 3], params([1, 1], [3, 3], [1, 1], 1)),
    ];
    for (ci, (n, c, hw, o, k, p)) in cases.into_iter().enumerate() {
        let g = geom(n, c, hw, o, k, p);
        let x = data(n * c * hw[0] * hw[1], 201 + ci as u32);
        let wt = data(o * g.kdim(), 2010 + ci as u32);
        let gy = data(n * o * g.ohw(), 3010 + ci as u32);

        let mut gx = vec![0f32; x.len()];
        conv2d_backward_input(&gy, &wt, &mut gx, &g);
        let mut gw = vec![0f32; wt.len()];
        conv2d_backward_weight(&x, &gy, &mut gw, &g);

        // Naive f64 reference for gx and gw.
        let (cg, og) = (g.cg(), g.og());
        let (sh, sw) = (p.stride[0] as isize, p.stride[1] as isize);
        let (ph, pw) = (p.padding[0] as isize, p.padding[1] as isize);
        let (dh, dw) = (p.dilation[0] as isize, p.dilation[1] as isize);
        let mut ref_gx = vec![0f64; x.len()];
        let mut ref_gw = vec![0f64; wt.len()];
        for img in 0..n {
            for grp in 0..p.groups {
                for oi in 0..og {
                    let oc = grp * og + oi;
                    for ci_l in 0..cg {
                        let ic = grp * cg + ci_l;
                        for oy in 0..g.oh {
                            for ox in 0..g.ow {
                                let gval = gy[((img * o + oc) * g.oh + oy) * g.ow + ox] as f64;
                                for ky in 0..k[0] {
                                    let iy = oy as isize * sh - ph + ky as isize * dh;
                                    if iy < 0 || iy >= hw[0] as isize {
                                        continue;
                                    }
                                    for kx in 0..k[1] {
                                        let ix = ox as isize * sw - pw + kx as isize * dw;
                                        if ix < 0 || ix >= hw[1] as isize {
                                            continue;
                                        }
                                        let xi = ((img * c + ic) * hw[0] + iy as usize) * hw[1] + ix as usize;
                                        let wi = ((oc * cg + ci_l) * k[0] + ky) * k[1] + kx;
                                        ref_gx[xi] += gval * wt[wi] as f64;
                                        ref_gw[wi] += gval * x[xi] as f64;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for (i, (&got, &want)) in gx.iter().zip(&ref_gx).enumerate() {
            assert!(
                (got as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "case {ci} gx[{i}]: {got} vs {want}"
            );
        }
        for (i, (&got, &want)) in gw.iter().zip(&ref_gw).enumerate() {
            assert!(
                (got as f64 - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "case {ci} gw[{i}]: {got} vs {want}"
            );
        }
    }
}

/// Depthwise forward SIMD path matches the scalar depthwise path across ragged widths, zero-width padding,
/// accumulate mode, and fused epilogues (`None`, `Bias`, `Silu`, `Relu`, `Sigmoid`).
#[test]
fn depthwise_simd_matches_scalar_parity() {
    let cases = [
        (1, 4, [5, 1], 4, [3, 3], params([1, 1], [1, 1], [1, 1], 4)),
        (1, 4, [5, 7], 4, [3, 3], params([1, 1], [1, 1], [1, 1], 4)),
        (1, 4, [7, 19], 4, [3, 3], params([1, 1], [1, 1], [1, 1], 4)),
        (2, 6, [9, 37], 12, [3, 3], params([1, 1], [1, 1], [1, 1], 6)),
        (1, 4, [8, 21], 4, [5, 5], params([1, 1], [2, 2], [1, 1], 4)),
        (1, 4, [9, 17], 4, [3, 3], params([1, 1], [2, 2], [2, 2], 4)),
        (1, 2, [3, 0], 2, [3, 3], params([1, 1], [1, 2], [1, 1], 2)),
    ];
    for (ci, &(n, c, hw, o, k, p)) in cases.iter().enumerate() {
        let g = geom(n, c, hw, o, k, p);
        let x = data(n * c * hw[0] * hw[1], 401 + ci as u32);
        let wt = data(o * g.kdim(), 4100 + ci as u32);
        let bias: Vec<f32> = (0..o).map(|j| if j % 2 == 0 { 0.0 } else { j as f32 * 0.2 - 0.5 }).collect();
        let part = [ConvPart::contiguous(&x, c, hw[0] * hw[1])];
        let input = ConvInput { parts: &part };
        let epis = [
            None,
            Some(BiasAct { bias: Some(&bias), act: None }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Silu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Relu) }),
            Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Sigmoid) }),
        ];
        for (ei, epi) in epis.into_iter().enumerate() {
            let mut got = vec![f32::NAN; n * o * g.ohw()];
            let mut want = vec![f32::NAN; n * o * g.ohw()];
            depthwise_forward(input, &wt, &mut got, &g, false, epi);
            super::depthwise::depthwise_forward_scalar(input, &wt, &mut want, &g, false, epi);
            for (q, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                assert_eq!(gv.to_bits(), wv.to_bits(), "case {ci} epi {ei} at {q}: {gv} vs {wv}");
                #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
                assert!(
                    (gv - wv).abs() <= 1e-5 * (1.0 + wv.abs()),
                    "case {ci} epi {ei} at {q}: {gv} vs {wv}"
                );
            }
        }
        // Accumulate into non-zero prior output.
        let prior: Vec<f32> = data(n * o * g.ohw(), 4200 + ci as u32)
            .into_iter()
            .map(|v| if v == 0.0 { 0.25 } else { v })
            .collect();
        let mut got = prior.clone();
        let mut want = prior;
        depthwise_forward(input, &wt, &mut got, &g, true, None);
        super::depthwise::depthwise_forward_scalar(input, &wt, &mut want, &g, true, None);
        for (q, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            assert_eq!(gv.to_bits(), wv.to_bits(), "case {ci} acc at {q}: {gv} vs {wv}");
            #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
            assert!((gv - wv).abs() <= 1e-5 * (1.0 + wv.abs()), "case {ci} acc at {q}: {gv} vs {wv}");
        }
    }

    // Wasm SIMD128 edge cases: non-finite weights on padded taps, accumulate over -0.0, and huge
    // padding/dilation that would overflow 32-bit padded scratch sizing.
    #[cfg(target_arch = "wasm32")]
    {
        let g = geom(1, 2, [1, 1], 2, [1, 3], params([1, 1], [0, 1], [1, 1], 2));
        let x = [1.0f32, 1.0];
        let wt = [f32::INFINITY, 2.0, 1.0, 1.0, -1.0, 1.0];
        let part = [ConvPart::contiguous(&x, 2, 1)];
        let input = ConvInput { parts: &part };
        let mut got = vec![f32::NAN; 2];
        let mut want = vec![f32::NAN; 2];
        depthwise_forward(input, &wt, &mut got, &g, false, None);
        super::depthwise::depthwise_forward_scalar(input, &wt, &mut want, &g, false, None);
        assert_eq!(got[0].to_bits(), want[0].to_bits(), "inf padded tap must not yield NaN");
        assert_eq!(got[1].to_bits(), want[1].to_bits());

        let x_zero = [0.0f32, 0.0];
        let wt_fin = [1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0];
        let part_z = [ConvPart::contiguous(&x_zero, 2, 1)];
        let input_z = ConvInput { parts: &part_z };
        let mut got_acc = vec![-0.0f32; 2];
        let mut want_acc = vec![-0.0f32; 2];
        depthwise_forward(input_z, &wt_fin, &mut got_acc, &g, true, None);
        super::depthwise::depthwise_forward_scalar(input_z, &wt_fin, &mut want_acc, &g, true, None);
        assert_eq!(
            got_acc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_acc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );

        let g_big = geom(1, 2, [1024, 1], 2, [1, 3], params([1, 1], [0, 1 << 21], [1, 1 << 21], 2));
        let x_big = vec![1.0f32; 2 * 1024];
        let wt_big = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
        let part_b = [ConvPart::contiguous(&x_big, 2, 1024)];
        let input_b = ConvInput { parts: &part_b };
        let mut got_b = vec![f32::NAN; 2 * 1024];
        let mut want_b = vec![f32::NAN; 2 * 1024];
        depthwise_forward(input_b, &wt_big, &mut got_b, &g_big, false, None);
        super::depthwise::depthwise_forward_scalar(input_b, &wt_big, &mut want_b, &g_big, false, None);
        assert_eq!(
            got_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }

    // Fused ReLU and Hardswish with NaN and -0.0 bias across SIMD prefix and scalar tail lanes.
    {
        let g_wide = geom(1, 2, [2, 10], 2, [1, 1], params([1, 1], [0, 0], [1, 1], 2));
        let x_neg0 = vec![-0.0f32; 2 * 20];
        let wt1 = [1.0f32, 1.0];
        let bias = [f32::NAN, -0.0];
        let epi = Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Relu) });
        let part_w = [ConvPart::contiguous(&x_neg0, 2, 20)];
        let input_w = ConvInput { parts: &part_w };
        let mut got_r = vec![f32::NAN; 40];
        let mut want_r = vec![f32::NAN; 40];
        depthwise_forward(input_w, &wt1, &mut got_r, &g_wide, false, epi);
        super::depthwise::depthwise_forward_scalar(input_w, &wt1, &mut want_r, &g_wide, false, epi);
        assert_eq!(
            got_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );

        let p = params([1, 1], [1, 1], [1, 1], 1);
        // F(2x2): 5 tiles wide (ow = 10 -> 4 SIMD tiles + 1 scalar tail tile).
        let g2 = geom(1, 4, [4, 10], 2, [3, 3], p);
        let x2 = vec![0.0f32; 4 * 40];
        let w2 = vec![0.0f32; 2 * 4 * 9];
        let u2 = transform_weights(&w2, 2, 4);
        let part2 = [ConvPart::contiguous(&x2, 4, 40)];
        let in2 = ConvInput { parts: &part2 };
        for act in [UnaryOp::Relu, UnaryOp::Hardswish] {
            let epi = Some(BiasAct { bias: Some(&bias), act: Some(act) });
            let mut got = vec![0.0f32; 2 * 40];
            let mut want = vec![0.0f32; 2 * 40];
            winograd_forward(in2, &u2, &mut got, &g2, epi);
            super::winograd::winograd_forward_scalar(in2, &u2, &mut want, &g2, epi);
            for (q, (&gv, &wv)) in got.iter().zip(&want).enumerate() {
                if wv.is_nan() {
                    assert!(gv.is_nan(), "wino2 {act:?} at {q}: expected NaN, got {gv}");
                } else {
                    assert_eq!(gv.to_bits(), wv.to_bits(), "wino2 {act:?} at {q}: {gv} vs {wv}");
                }
            }
        }
        // F(4x4): 5 tiles wide (ow = 20 -> 4 SIMD tiles + 1 scalar tail tile).
        let g4 = geom(1, 4, [4, 20], 2, [3, 3], p);
        let x4 = vec![0.0f32; 4 * 80];
        let w4 = vec![0.0f32; 2 * 4 * 9];
        let u4 = transform_weights4(&w4, 2, 4);
        let part4 = [ConvPart::contiguous(&x4, 4, 80)];
        let in4 = ConvInput { parts: &part4 };
        let epi = Some(BiasAct { bias: Some(&bias), act: Some(UnaryOp::Relu) });
        let mut got = vec![0.0f32; 2 * 80];
        let mut want = vec![0.0f32; 2 * 80];
        winograd4_forward(in4, &w4, &u4, &mut got, &g4, epi);
        super::winograd4::winograd4_forward_scalar(in4, &w4, &u4, &mut want, &g4, epi);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "wino4 ReLU NaN/-0.0 bias mismatch"
        );
    }
}
