//! The direct forward conv for few input channels (the 3-channel stem): per output row the taps are gathered
//! into L1 and every output channel is that many FMAs per pixel, bit for bit the GEMM path's result.

use super::{ChannelEpilogue, ConvGeom, ConvInput, valid_cols};
use crate::cpu::elementwise::SendPtr;
use crate::cpu::par_for;

/// Input channels up to which a (non-pointwise, ungrouped) forward conv runs [`small_c_forward`] instead of
/// the shifted-rows GEMM: the stem's 3-channel image input, whose GEMM has a K of only 27 and whose
/// shifted-rows copy is filled by one task per input channel (three).
const SMALL_C: usize = 4;
/// Output columns per register tile of [`small_c_forward`] (four AVX2 vectors).
const SMALL_C_TILE: usize = 32;
/// The deepest K [`small_c_forward`] takes: within one K block of the GEMM (256), so both accumulate every
/// output in the same order and agree bit for bit.
const SMALL_C_MAX_K: usize = 256;
/// Below this many flops [`small_c_forward`] runs on the calling thread (the GEMM's threshold).
const SMALL_C_PAR_FLOPS: usize = 1 << 21;

/// Whether [`super::conv2d_forward_parts`] (overwriting) takes [`small_c_forward`].
pub(super) fn uses_small_c(g: &ConvGeom) -> bool {
    g.p.groups == 1 && g.c <= SMALL_C && !g.is_pointwise() && g.ohw() > 0 && g.kdim() <= SMALL_C_MAX_K
}

/// Direct forward conv for few input channels (see [`SMALL_C`]), overwriting `out`, in parallel over bands of
/// output rows (of all images). Per output row it gathers the `K = c·kh·kw` input rows the taps read (strided
/// and zero-padded, `K x ow` floats, in L1), then every output channel is `K` multiply-adds per pixel from zero
/// in `(ci, ky, kx)` order, in register tiles of [`SMALL_C_TILE`] columns. That is the order and the
/// arithmetic (fused multiply-add with AVX2 and FMA, else a multiply then an add) of the GEMM micro-kernels on
/// one K block, so the result is bit for bit the GEMM path's. `epi` sees each finished output row.
pub(super) fn small_c_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let (hw, ohw, k) = (g.h * g.w, g.ohw(), g.kdim());
    let owp = g.ow.div_ceil(SMALL_C_TILE) * SMALL_C_TILE;
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let rows = g.n * g.oh;
    // Up to four output rows (of every channel) per task, and about four tasks per thread when rows allow.
    let band = (rows / (4 * rayon::current_num_threads())).clamp(1, 4);
    let optr = SendPtr(out.as_mut_ptr());
    let task = |t: usize| {
        let optr = &optr;
        let mut gat = vec![0f32; k * owp];
        let mut orow = vec![0f32; owp];
        for r in t * band..((t + 1) * band).min(rows) {
            let (img, oy) = (r / g.oh, r % g.oh);
            small_c_gather(&planes[img], g, oy, owp, &mut gat);
            for oc in 0..g.o {
                let w = &wt[oc * k..(oc + 1) * k];
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 and FMA were detected above.
                    unsafe { small_c_row_avx2(w, &gat, owp, &mut orow) };
                } else {
                    small_c_row(w, &gat, owp, &mut orow);
                }
                #[cfg(not(target_arch = "x86_64"))]
                small_c_row(w, &gat, owp, &mut orow);
                // SAFETY: row `oy` of channel `oc` of image `img` is inside `out` (`[n, o, oh, ow]`), and only
                // this task (which owns output row `r` of every channel) touches it.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(optr.0.add((img * g.o + oc) * ohw + oy * g.ow), g.ow)
                };
                dst.copy_from_slice(&orow[..g.ow]);
                if let Some(e) = epi {
                    e.apply(oc, dst);
                }
            }
        }
    };
    let tasks = rows.div_ceil(band);
    if 2 * g.n * g.o * ohw * k >= SMALL_C_PAR_FLOPS {
        par_for(tasks, &task);
    } else {
        (0..tasks).for_each(task);
    }
}

/// The `K x owp` taps of output row `oy`: `gat[kk * owp + ox]` is the input element that tap
/// `kk = (ci, ky, kx)` of output pixel `(oy, ox)` reads, zero where it falls in the padding and for `ox >= ow`.
fn small_c_gather(planes: &[&[f32]], g: &ConvGeom, oy: usize, owp: usize, gat: &mut [f32]) {
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    let mut rows = gat.chunks_exact_mut(owp);
    for plane in planes {
        for ky in 0..g.kh {
            let iy = oy as isize * sh - ph + ky as isize * dh;
            let src =
                (0..g.h as isize).contains(&iy).then(|| &plane[iy as usize * g.w..(iy as usize + 1) * g.w]);
            for kx in 0..g.kw {
                let dst = rows.next().expect("one gathered row per tap");
                let Some(src) = src else {
                    dst.fill(0.0);
                    continue;
                };
                let d = kx as isize * dw - pw;
                let r = valid_cols(g.ow, sw, d, g.w as isize);
                dst[..r.start].fill(0.0);
                dst[r.end..].fill(0.0);
                if r.is_empty() {
                    // The tap sees only padding (it starts right of the row, or ends left of it).
                } else if sw == 1 {
                    let s0 = (r.start as isize + d) as usize;
                    dst[r.clone()].copy_from_slice(&src[s0..s0 + r.len()]);
                } else {
                    for ox in r {
                        dst[ox] = src[(ox as isize * sw + d) as usize];
                    }
                }
            }
        }
    }
}

/// One output row of one channel from the gathered taps: `orow[ox] = Σ_kk w[kk] · gat[kk][ox]`, a multiply
/// then an add from zero in `kk` order (the portable GEMM micro-kernel's arithmetic).
fn small_c_row(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    for (t, o) in orow.as_chunks_mut::<SMALL_C_TILE>().0.iter_mut().enumerate() {
        let mut acc = [0f32; SMALL_C_TILE];
        for (kk, &wv) in w.iter().enumerate() {
            let src = &gat[kk * owp + t * SMALL_C_TILE..][..SMALL_C_TILE];
            for (a, &v) in acc.iter_mut().zip(src) {
                *a += wv * v;
            }
        }
        *o = acc;
    }
}

/// [`small_c_row`] with AVX2 fused multiply-adds (the AVX2 GEMM micro-kernels' arithmetic): four 8-wide
/// accumulators per tile.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn small_c_row_avx2(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(gat.len() >= w.len() * owp && orow.len() == owp && owp.is_multiple_of(SMALL_C_TILE));
    for x0 in (0..owp).step_by(SMALL_C_TILE) {
        let (mut c0, mut c1, mut c2, mut c3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        for (kk, wv) in w.iter().enumerate() {
            // SAFETY: row `kk` of `gat` holds `owp` floats and `x0 + SMALL_C_TILE <= owp` (asserted above).
            unsafe {
                let p = gat.as_ptr().add(kk * owp + x0);
                let a = _mm256_broadcast_ss(wv);
                c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p), c0);
                c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(8)), c1);
                c2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(16)), c2);
                c3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(24)), c3);
            }
        }
        // SAFETY: `orow` holds `owp >= x0 + SMALL_C_TILE` floats.
        unsafe {
            let o = orow.as_mut_ptr().add(x0);
            _mm256_storeu_ps(o, c0);
            _mm256_storeu_ps(o.add(8), c1);
            _mm256_storeu_ps(o.add(16), c2);
            _mm256_storeu_ps(o.add(24), c3);
        }
    }
}

#[cfg(test)]
mod small_c_tests {
    use super::*;
    use crate::backend::{Conv2dParams, UnaryOp};
    use crate::cpu::conv::shifted::ShiftedRows;
    use crate::cpu::conv::{ConvPart, conv2d_forward};
    use crate::cpu::gemm::{BiasAct, RowPtrs, Strided, gemm_rows};

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    fn geom(n: usize, c: usize, hw: [usize; 2], o: usize, k: [usize; 2], p: Conv2dParams) -> ConvGeom {
        let out =
            |i: usize, k: usize| (hw[i] + 2 * p.padding[i] - p.dilation[i] * (k - 1) - 1) / p.stride[i] + 1;
        let (oh, ow) = (out(0, k[0]), out(1, k[1]));
        ConvGeom { n, c, h: hw[0], w: hw[1], o, kh: k[0], kw: k[1], oh, ow, p }
    }

    /// The shifted-rows GEMM path [`small_c_forward`] replaces (what [`conv2d_forward_parts`] ran before).
    fn gemm_path(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
        let (kdim, ohw, hw) = (g.kdim(), g.ohw(), g.h * g.w);
        let part = [ConvPart::contiguous(x, g.c, hw)];
        let x = ConvInput { parts: &part };
        for (img, out_img) in out.chunks_mut(g.o * ohw).enumerate() {
            let planes = x.planes(img, hw);
            let a = Strided::lhs(wt, 0, kdim, 1);
            let sr = ShiftedRows::new(&planes, g, true);
            let b = RowPtrs { rows: &sr.rows, n: sr.n };
            gemm_rows(g.o, kdim, &a, &b, out_img, sr.map, rayon::current_num_threads(), epi);
        }
    }

    #[test]
    fn small_c_matches_the_gemm_path_bit_for_bit() {
        let p = |s: [usize; 2], pad: [usize; 2], d: [usize; 2]| Conv2dParams {
            stride: s,
            padding: pad,
            dilation: d,
            groups: 1,
        };
        // (n, c, [h, w], o, [kh, kw], params): the folo stem at 640 and 320, then odd shapes, strides,
        // paddings, dilations and channel counts 1..=4.
        let cases = [
            (1, 3, [640, 640], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (2, 3, [320, 384], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (1, 1, [37, 29], 5, [3, 3], p([1, 1], [1, 1], [1, 1])),
            (3, 2, [19, 23], 7, [5, 3], p([2, 1], [2, 0], [1, 2])),
            (1, 4, [33, 65], 9, [3, 3], p([3, 2], [0, 1], [1, 1])),
            (2, 3, [8, 8], 4, [3, 3], p([1, 1], [3, 3], [1, 1])),
            (1, 3, [70, 70], 33, [7, 7], p([2, 2], [3, 3], [1, 1])),
            (1, 3, [5, 200], 2, [1, 3], p([1, 4], [0, 1], [1, 1])),
            // Taps that see only padding on the left or the right of a row (and above / below it).
            (1, 2, [1, 1], 2, [5, 5], p([1, 1], [2, 2], [1, 1])),
            (2, 1, [4, 3], 3, [5, 7], p([1, 1], [2, 3], [1, 1])),
            (1, 3, [3, 2], 2, [3, 5], p([2, 2], [1, 3], [1, 1])),
        ];
        for (i, (n, c, hw, o, k, prm)) in cases.into_iter().enumerate() {
            let g = geom(n, c, hw, o, k, prm);
            assert!(uses_small_c(&g), "case {i}");
            let x = data(n * c * hw[0] * hw[1], 1 + i as u32);
            let wt = data(o * g.kdim(), 100 + i as u32);
            // Channel 4's bias is 0.0.
            let bias: Vec<f32> = (0..o).map(|c| c as f32 * 0.25 - 1.0).collect();
            let b = Some(&bias[..]);
            // The GEMM applies the identity and SiLU in its micro-kernels, ReLU as a pass over finished rows.
            let epis = [
                None,
                Some(BiasAct { bias: None, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: None }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Relu) }),
            ];
            for epi in epis {
                let mut want = vec![f32::NAN; n * o * g.ohw()];
                gemm_path(&x, &wt, &mut want, &g, epi);
                let mut got = vec![f32::NAN; want.len()];
                conv2d_forward(&x, &wt, &mut got, &g, false, epi);
                let same = got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "case {i} ({g:?}): not bit-identical with the GEMM path");
            }
        }
    }
}
