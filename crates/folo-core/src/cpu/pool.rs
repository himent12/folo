//! Pooling and nearest upsampling on contiguous NCHW planes.

use rayon::prelude::*;

use super::{alloc_uninit, alloc_zeroed};
use crate::backend::Pool2dParams;

/// `(planes, h, w)` of an NCHW tensor.
pub(crate) type Planes = (usize, usize, usize);

/// `a` beats the running best `b` (with index `bi`, negative = none yet): strictly greater, or the first NaN.
/// Scanning a window in row-major order with this rule picks its first maximum (or first NaN).
#[inline(always)]
fn beats(a: f32, b: f32, bi: i64) -> bool {
    a > b || bi < 0 || a.is_nan() && !b.is_nan()
}

/// Max pooling with the argmax of each window (flat index into its input plane). Separable: a row pass takes
/// the best of each `kw`-wide window of every input row, then a column pass takes the best of `kh` row results.
/// Both passes scan in ascending order with [`beats`], so the result is the first maximum of the window in
/// row-major order, exactly as a direct scan of the window.
pub(crate) fn max_pool2d(
    x: &[f32],
    (np, h, w): Planes,
    p: &Pool2dParams,
    [oh, ow]: [usize; 2],
) -> (Vec<f32>, Vec<i64>) {
    let mut out = alloc_uninit(np * oh * ow);
    let mut idx = vec![0i64; np * oh * ow];
    if out.is_empty() || h * w == 0 {
        return (out, idx);
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    // Window ranges along each axis (clipped to the input; padding never wins).
    let range = |o: usize, s: usize, pad: usize, k: usize, n: usize| {
        let a = (o * s) as isize - pad as isize;
        a.max(0) as usize..((a + k as isize).min(n as isize)).max(0) as usize
    };
    let xs: Vec<_> = (0..ow).map(|ox| range(ox, sw, pw, kw, w)).collect();
    let ys: Vec<_> = (0..oh).map(|oy| range(oy, sh, ph, kh, h)).collect();
    let work = |(rv, ri): &mut (Vec<f32>, Vec<i64>), (pl, (o, ix)): (usize, (&mut [f32], &mut [i64]))| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        rv.resize(h * ow, 0.0);
        ri.resize(h * ow, 0);
        for yy in 0..h {
            let row = &xp[yy * w..(yy + 1) * w];
            for (ox, r) in xs.iter().enumerate() {
                let (mut best, mut bi) = (f32::NEG_INFINITY, -1i64);
                for xx in r.clone() {
                    if beats(row[xx], best, bi) {
                        best = row[xx];
                        bi = xx as i64;
                    }
                }
                rv[yy * ow + ox] = best;
                ri[yy * ow + ox] = bi;
            }
        }
        for (oy, r) in ys.iter().enumerate() {
            for ox in 0..ow {
                let (mut best, mut bi) = (f32::NEG_INFINITY, -1i64);
                for yy in r.clone() {
                    let (v, i) = (rv[yy * ow + ox], ri[yy * ow + ox]);
                    if i >= 0 && beats(v, best, bi) {
                        best = v;
                        bi = yy as i64 * w as i64 + i;
                    }
                }
                o[oy * ow + ox] = best;
                ix[oy * ow + ox] = bi.max(0);
            }
        }
    };
    let planes = out.par_chunks_mut(oh * ow).zip(idx.par_chunks_mut(oh * ow)).enumerate();
    planes.for_each_init(|| (Vec::new(), Vec::new()), work);
    (out, idx)
}

/// [`max_pool2d`]'s values alone (inference: nobody differentiates the result), bit for bit.
///
/// The same two passes, over rows padded with `-inf`: a window's first element (padding or not) starts its
/// running best, and each next one replaces it under [`beats`]. Padding then never changes a value: `-inf`
/// never beats anything, and a `-inf` start is replaced by the first real element unless every real element
/// is `-inf` too, the same bits. With every window the same width, each pass is a select over whole rows that
/// the compiler vectorises. Planes are grouped into tasks of about [`POOL_TASK`] outputs.
pub(crate) fn max_pool2d_values(
    x: &[f32],
    (np, h, w): Planes,
    p: &Pool2dParams,
    [oh, ow]: [usize; 2],
) -> Vec<f32> {
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() {
        return out;
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    // Padded extents covering every window ((o - 1) * s + k <= n + 2 * pad).
    let (hp, wp) = ((oh - 1) * sh + kh, (ow - 1) * sw + kw);
    // One step of the scan: `a` replaces `b` if it beats it (`b` is never the empty start here).
    #[inline(always)]
    fn step(b: &mut f32, a: f32) {
        if a > *b || a.is_nan() && !b.is_nan() {
            *b = a;
        }
    }
    let work = |(row, rv): &mut (Vec<f32>, Vec<f32>), (pl, o): (usize, &mut [f32])| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        // The padding columns stay `-inf`; every row overwrites the same real columns.
        row.clear();
        row.resize(wp, f32::NEG_INFINITY);
        // Row results of every padded row (padding rows are all `-inf`).
        rv.clear();
        rv.resize(hp * ow, f32::NEG_INFINITY);
        for yy in 0..h.min(hp.saturating_sub(ph)) {
            let n = w.min(wp.saturating_sub(pw));
            row[pw..pw + n].copy_from_slice(&xp[yy * w..yy * w + n]);
            let r = &mut rv[(yy + ph) * ow..(yy + ph + 1) * ow];
            for (ox, b) in r.iter_mut().enumerate() {
                *b = row[ox * sw];
            }
            for t in 1..kw {
                for (ox, b) in r.iter_mut().enumerate() {
                    step(b, row[ox * sw + t]);
                }
            }
        }
        for (oy, orow) in o.chunks_exact_mut(ow).enumerate() {
            orow.copy_from_slice(&rv[oy * sh * ow..(oy * sh + 1) * ow]);
            for t in 1..kh {
                let src = &rv[(oy * sh + t) * ow..(oy * sh + t + 1) * ow];
                for (b, &a) in orow.iter_mut().zip(src) {
                    step(b, a);
                }
            }
        }
    };
    let per_task = (POOL_TASK / (oh * ow)).max(1);
    out.par_chunks_mut(per_task * oh * ow).enumerate().for_each_init(
        || (Vec::new(), Vec::new()),
        |scratch, (t, chunk)| {
            for (c, o) in chunk.chunks_exact_mut(oh * ow).enumerate() {
                work(scratch, (t * per_task + c, o));
            }
        },
    );
    out
}

/// Outputs per task of [`max_pool2d_values`] (a few microseconds of work).
const POOL_TASK: usize = 1 << 12;

pub(crate) fn max_pool2d_backward(g: &[f32], idx: &[i64], (np, h, w): Planes, ohw: usize) -> Vec<f32> {
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() {
        return gx;
    }
    gx.par_chunks_mut(h * w).enumerate().for_each(|(pl, gp)| {
        let gs = &g[pl * ohw..(pl + 1) * ohw];
        let is = &idx[pl * ohw..(pl + 1) * ohw];
        for (&v, &i) in gs.iter().zip(is) {
            gp[i as usize] += v;
        }
    });
    gx
}

pub(crate) fn avg_pool2d(x: &[f32], (np, h, w): Planes, p: &Pool2dParams, [oh, ow]: [usize; 2]) -> Vec<f32> {
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() || h * w == 0 {
        return out;
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    let inv = 1.0 / (kh * kw) as f32;
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(pl, o)| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        for oy in 0..oh {
            let y0 = (oy * sh) as isize - ph as isize;
            let ys = y0.max(0) as usize..((y0 + kh as isize).min(h as isize)).max(0) as usize;
            for ox in 0..ow {
                let x0 = (ox * sw) as isize - pw as isize;
                let xs = x0.max(0) as usize..((x0 + kw as isize).min(w as isize)).max(0) as usize;
                let mut s = 0f32;
                for yy in ys.clone() {
                    for xx in xs.clone() {
                        s += xp[yy * w + xx];
                    }
                }
                o[oy * ow + ox] = s * inv;
            }
        }
    });
    out
}

pub(crate) fn avg_pool2d_backward(
    g: &[f32],
    (np, h, w): Planes,
    p: &Pool2dParams,
    [oh, ow]: [usize; 2],
) -> Vec<f32> {
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() {
        return gx;
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    let inv = 1.0 / (kh * kw) as f32;
    gx.par_chunks_mut(h * w).enumerate().for_each(|(pl, gp)| {
        let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
        for oy in 0..oh {
            let y0 = (oy * sh) as isize - ph as isize;
            let ys = y0.max(0) as usize..((y0 + kh as isize).min(h as isize)).max(0) as usize;
            for ox in 0..ow {
                let x0 = (ox * sw) as isize - pw as isize;
                let xs = x0.max(0) as usize..((x0 + kw as isize).min(w as isize)).max(0) as usize;
                let v = gs[oy * ow + ox] * inv;
                for yy in ys.clone() {
                    for xx in xs.clone() {
                        gp[yy * w + xx] += v;
                    }
                }
            }
        }
    });
    gx
}

#[inline]
fn adaptive_range(i: usize, input: usize, output: usize) -> std::ops::Range<usize> {
    (i * input / output)..((i + 1) * input).div_ceil(output)
}

pub(crate) fn adaptive_avg_pool2d(x: &[f32], (np, h, w): Planes, [oh, ow]: [usize; 2]) -> Vec<f32> {
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() || h * w == 0 {
        return out;
    }
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(pl, o)| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        for oy in 0..oh {
            let ys = adaptive_range(oy, h, oh);
            for ox in 0..ow {
                let xs = adaptive_range(ox, w, ow);
                let mut s = 0f32;
                for yy in ys.clone() {
                    s += xp[yy * w + xs.start..yy * w + xs.end].iter().sum::<f32>();
                }
                o[oy * ow + ox] = s / (ys.len() * xs.len()) as f32;
            }
        }
    });
    out
}

pub(crate) fn adaptive_avg_pool2d_backward(g: &[f32], (np, h, w): Planes, [oh, ow]: [usize; 2]) -> Vec<f32> {
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() {
        return gx;
    }
    gx.par_chunks_mut(h * w).enumerate().for_each(|(pl, gp)| {
        let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
        for oy in 0..oh {
            let ys = adaptive_range(oy, h, oh);
            for ox in 0..ow {
                let xs = adaptive_range(ox, w, ow);
                let v = gs[oy * ow + ox] / (ys.len() * xs.len()) as f32;
                for yy in ys.clone() {
                    for d in &mut gp[yy * w + xs.start..yy * w + xs.end] {
                        *d += v;
                    }
                }
            }
        }
    });
    gx
}

pub(crate) fn upsample_nearest2d(x: &[f32], (np, h, w): Planes, [sh, sw]: [usize; 2]) -> Vec<f32> {
    let (oh, ow) = (h * sh, w * sw);
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() || h * w == 0 {
        return out;
    }
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(pl, o)| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        for y in 0..h {
            let src = &xp[y * w..(y + 1) * w];
            let first = &mut o[y * sh * ow..(y * sh + 1) * ow];
            for (xx, &v) in src.iter().enumerate() {
                first[xx * sw..(xx + 1) * sw].fill(v);
            }
            for r in 1..sh {
                let (head, tail) = o.split_at_mut((y * sh + r) * ow);
                tail[..ow].copy_from_slice(&head[y * sh * ow..(y * sh + 1) * ow]);
            }
        }
    });
    out
}

pub(crate) fn upsample_nearest2d_backward(g: &[f32], (np, h, w): Planes, [sh, sw]: [usize; 2]) -> Vec<f32> {
    let (oh, ow) = (h * sh, w * sw);
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() {
        return gx;
    }
    gx.par_chunks_mut(h * w).enumerate().for_each(|(pl, gp)| {
        let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
        for oy in 0..oh {
            let row = &gs[oy * ow..(oy + 1) * ow];
            let dst = &mut gp[(oy / sh) * w..(oy / sh + 1) * w];
            for (xx, d) in dst.iter_mut().enumerate() {
                *d += row[xx * sw..(xx + 1) * sw].iter().sum::<f32>();
            }
        }
    });
    gx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direct window scan the separable kernel must reproduce, argmax included.
    fn direct(x: &[f32], (np, h, w): Planes, p: &Pool2dParams, [oh, ow]: [usize; 2]) -> (Vec<f32>, Vec<i64>) {
        let (mut v, mut ix) = (Vec::new(), Vec::new());
        for pl in 0..np {
            for oy in 0..oh {
                for ox in 0..ow {
                    let (mut best, mut bi) = (f32::NEG_INFINITY, -1i64);
                    for ky in 0..p.kernel[0] {
                        for kx in 0..p.kernel[1] {
                            let yy = (oy * p.stride[0] + ky) as isize - p.padding[0] as isize;
                            let xx = (ox * p.stride[1] + kx) as isize - p.padding[1] as isize;
                            if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                                continue;
                            }
                            let i = yy as usize * w + xx as usize;
                            if beats(x[pl * h * w + i], best, bi) {
                                best = x[pl * h * w + i];
                                bi = i as i64;
                            }
                        }
                    }
                    v.push(best);
                    ix.push(bi.max(0));
                }
            }
        }
        (v, ix)
    }

    #[test]
    fn separable_max_pool_matches_direct_scan_with_ties_and_nans() {
        // Few distinct values, so most windows have tied maxima; a sprinkling of NaNs.
        let (np, h, w) = (3, 11, 9);
        let x: Vec<f32> =
            (0..np * h * w).map(|i| if i % 37 == 5 { f32::NAN } else { ((i * 7919) % 5) as f32 }).collect();
        for (k, s, pad) in [([5, 5], [1, 1], [2, 2]), ([3, 2], [2, 1], [1, 1]), ([2, 3], [2, 3], [0, 1])] {
            let p = Pool2dParams { kernel: k, stride: s, padding: pad };
            let ohw = p.output_hw(h, w).unwrap();
            let (v, ix) = max_pool2d(&x, (np, h, w), &p, ohw);
            let (wv, wix) = direct(&x, (np, h, w), &p, ohw);
            assert_eq!(ix, wix, "{p:?}");
            for (a, b) in v.iter().zip(&wv) {
                assert!(a == b || a.is_nan() && b.is_nan(), "{p:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn max_pool_values_are_the_indexed_values_bit_for_bit() {
        // Ties between -0.0 and +0.0, NaNs with distinct payloads, -inf (whole windows of it), and planes
        // narrower or shorter than the window.
        let special =
            [0.0f32, -0.0, f32::NEG_INFINITY, f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc0_0002)];
        for (np, h, w) in [(3, 11, 9), (2, 2, 3), (5, 1, 1), (17, 20, 20)] {
            let x: Vec<f32> = (0..np * h * w)
                .map(|i| match (i * 2654435761) % 11 {
                    k @ 0..=4 => special[k],
                    k => (k as f32 - 7.0) * 0.5,
                })
                .collect();
            for (k, s, pad) in [
                ([5, 5], [1, 1], [2, 2]),
                ([3, 2], [2, 1], [1, 1]),
                ([2, 3], [2, 3], [0, 1]),
                ([1, 1], [1, 1], [0, 0]),
                ([3, 3], [3, 2], [1, 0]),
            ] {
                let p = Pool2dParams { kernel: k, stride: s, padding: pad };
                let Some(ohw) = p.output_hw(h, w) else { continue };
                let (v, _) = max_pool2d(&x, (np, h, w), &p, ohw);
                let fast = max_pool2d_values(&x, (np, h, w), &p, ohw);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&fast), bits(&v), "{np}x{h}x{w} {p:?}");
            }
        }
    }
}
