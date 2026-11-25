//! Pooling and nearest upsampling on contiguous NCHW planes.

#[cfg(test)]
mod tests;
mod upsample;

use upsample::duplicate_row2;
pub(crate) use upsample::{upsample_nearest2d, upsample_nearest2d_backward};

use rayon::prelude::*;

use super::{PAR_MIN, alloc_uninit, alloc_zeroed};
use crate::backend::Pool2dParams;
use crate::cpu::par_chunks;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// `dst[i] += src[i]` over equal-length contiguous slices.
#[inline]
fn row_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { row_add_avx2(dst, src) };
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn row_add_avx2(dst: &mut [f32], src: &[f32]) {
    use std::arch::x86_64::*;
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let (sc, sr) = src.as_chunks::<8>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `d` and `s` are 8 floats.
        unsafe {
            let v = _mm256_add_ps(_mm256_loadu_ps(d.as_ptr()), _mm256_loadu_ps(s.as_ptr()));
            _mm256_storeu_ps(d.as_mut_ptr(), v);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d += s;
    }
}

/// `dst[i] += src[i] * scale` (non-FMA, bit-exact with scalar).
#[inline]
fn row_add_scaled(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { row_add_scaled_avx2(dst, src, scale) };
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += s * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn row_add_scaled_avx2(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(scale);
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let (sc, sr) = src.as_chunks::<8>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `d` and `s` are 8 floats.
        unsafe {
            let v =
                _mm256_add_ps(_mm256_loadu_ps(d.as_ptr()), _mm256_mul_ps(_mm256_loadu_ps(s.as_ptr()), sv));
            _mm256_storeu_ps(d.as_mut_ptr(), v);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d += s * scale;
    }
}

/// `dst[i] *= scale` in place.
#[inline]
fn row_scale(dst: &mut [f32], scale: f32) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { row_scale_avx2(dst, scale) };
    }
    for d in dst.iter_mut() {
        *d *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn row_scale_avx2(dst: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(scale);
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for d in dc {
        // SAFETY: `d` is 8 floats.
        unsafe {
            _mm256_storeu_ps(d.as_mut_ptr(), _mm256_mul_ps(_mm256_loadu_ps(d.as_ptr()), sv));
        }
    }
    for d in dr {
        *d *= scale;
    }
}

/// `dst[i] = src[2*i] + src[2*i + 1]` for `i < dst.len()` (`src.len() >= 2 * dst.len()`).
#[inline]
fn sum_pairs(src: &[f32], dst: &mut [f32]) {
    assert!(src.len() >= 2 * dst.len(), "sum_pairs: source slice too short");
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { sum_pairs_avx2(src, dst) };
    }
    for (i, d) in dst.iter_mut().enumerate() {
        *d = src[2 * i] + src[2 * i + 1];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sum_pairs_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let (sc, sr) = src[..2 * n].as_chunks::<16>();
    for (d, s) in dc.iter_mut().zip(sc) {
        // SAFETY: `s` is 16 floats and `d` is 8 floats.
        unsafe {
            let a = _mm256_loadu_ps(s.as_ptr());
            let b = _mm256_loadu_ps(s.as_ptr().add(8));
            // `vhaddps(a, b)` gives `[a0+a1, a2+a3, b0+b1, b2+b3, a4+a5, a6+a7, b4+b5, b6+b7]`.
            // `vpermq(0xD8)` puts the 64-bit lanes in order `[0, 2, 1, 3]` -> `[a0..a7 pairs, b0..b7 pairs]`.
            let h = _mm256_hadd_ps(a, b);
            let ordered = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(h)));
            _mm256_storeu_ps(d.as_mut_ptr(), ordered);
        }
    }
    for (d, pair) in dr.iter_mut().zip(sr.as_chunks::<2>().0) {
        *d = pair[0] + pair[1];
    }
}

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
    if h * w == 0 {
        // Every window is padding alone.
        out.fill(f32::NEG_INFINITY);
        return (out, idx);
    }
    if out.is_empty() {
        return (out, idx);
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    // Window ranges along each axis (clipped to the input; padding never wins). `o * s + k <= n + 2 * pad`
    // (the output size), so nothing overflows, also for a window wider than `isize::MAX`.
    let range = |o: usize, s: usize, pad: usize, k: usize, n: usize| {
        (o * s).saturating_sub(pad)..(o * s + k).saturating_sub(pad).min(n)
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
///
/// A padded extent more than [`PAD_SLACK`] past twice the input's (a window far larger than the plane), or an
/// empty plane, runs [`max_pool2d`] instead and drops the indices: its clipped windows never visit the padding.
pub(crate) fn max_pool2d_values(
    x: &[f32],
    (np, h, w): Planes,
    p: &Pool2dParams,
    [oh, ow]: [usize; 2],
) -> Vec<f32> {
    if np * oh * ow == 0 {
        return alloc_uninit(0);
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    // Padded extents covering every window ((o - 1) * s + k <= n + 2 * pad; `oh, ow >= 1` here).
    let extent = |o: usize, s: usize, k: usize| (o - 1).checked_mul(s)?.checked_add(k);
    let bound = |n: usize| n.saturating_mul(2).saturating_add(PAD_SLACK);
    let (hp, wp) = match (extent(oh, sh, kh), extent(ow, sw, kw)) {
        (Some(hp), Some(wp)) if h != 0 && w != 0 && hp <= bound(h) && wp <= bound(w) => (hp, wp),
        _ => return max_pool2d(x, (np, h, w), p, [oh, ow]).0,
    };
    let mut out = alloc_uninit(np * oh * ow);
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
pub(super) const POOL_TASK: usize = 1 << 12;

/// How far [`max_pool2d_values`]'s padded rows and columns may exceed twice the input's before it falls back to
/// the clipped scan (every detector pool pads by at most a few elements).
const PAD_SLACK: usize = 16;

pub(crate) fn max_pool2d_backward(g: &[f32], idx: &[i64], (np, h, w): Planes, ohw: usize) -> Vec<f32> {
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() {
        return gx;
    }
    par_chunks(&mut gx, h * w, &|pl, gp| {
        let gs = &g[pl * ohw..(pl + 1) * ohw];
        let is = &idx[pl * ohw..(pl + 1) * ohw];
        for (&v, &i) in gs.iter().zip(is) {
            gp[i as usize] += v;
        }
    });
    gx
}

/// Separable horizontal + vertical average pooling (padding zeros count toward the `kh * kw` divisor).
///
/// Follows the padded two-pass strategy of [`max_pool2d_values`]: each input row is copied once into a
/// zero-padded scratch row and horizontally summed into `rv [hp, ow]` (with AVX2 fast paths for `sw == 1`
/// and `sw == 2, kw == 2`), then `kh` rows of `rv` are vertically summed and scaled by `1 / (kh * kw)`.
/// Windows far larger than the plane fall back to clipped separable ranges without materialising padding.
pub(crate) fn avg_pool2d(x: &[f32], (np, h, w): Planes, p: &Pool2dParams, [oh, ow]: [usize; 2]) -> Vec<f32> {
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() {
        return out;
    }
    if h * w == 0 {
        out.fill(0.0);
        return out;
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    let inv = 1.0 / (kh * kw) as f32;
    let extent = |o: usize, s: usize, k: usize| (o - 1).checked_mul(s)?.checked_add(k);
    let bound = |n: usize| n.saturating_mul(2).saturating_add(PAD_SLACK);
    if let (Some(hp), Some(wp)) = (extent(oh, sh, kh), extent(ow, sw, kw))
        && hp <= bound(h)
        && wp <= bound(w)
    {
        let h_eff = h.min(hp.saturating_sub(ph));
        let w_eff = w.min(wp.saturating_sub(pw));
        let work = |(row, rv): &mut (Vec<f32>, Vec<f32>), (pl, o): (usize, &mut [f32])| {
            let xp = &x[pl * h * w..(pl + 1) * h * w];
            row.clear();
            row.resize(wp, 0.0);
            rv.clear();
            rv.resize(hp * ow, 0.0);
            for yy in 0..h_eff {
                row[pw..pw + w_eff].copy_from_slice(&xp[yy * w..yy * w + w_eff]);
                let r = &mut rv[(yy + ph) * ow..(yy + ph + 1) * ow];
                if sw == 1 {
                    r.copy_from_slice(&row[..ow]);
                    for t in 1..kw {
                        row_add(r, &row[t..t + ow]);
                    }
                } else if sw == 2 && kw == 2 {
                    sum_pairs(&row[..2 * ow], r);
                } else {
                    for (ox, b) in r.iter_mut().enumerate() {
                        *b = row[ox * sw];
                    }
                    for t in 1..kw {
                        for (ox, b) in r.iter_mut().enumerate() {
                            *b += row[ox * sw + t];
                        }
                    }
                }
            }
            for (oy, orow) in o.chunks_exact_mut(ow).enumerate() {
                orow.copy_from_slice(&rv[oy * sh * ow..(oy * sh + 1) * ow]);
                for t in 1..kh {
                    row_add(orow, &rv[(oy * sh + t) * ow..(oy * sh + t + 1) * ow]);
                }
                row_scale(orow, inv);
            }
        };
        if out.len() >= PAR_MIN && np > 1 {
            let per_task = (POOL_TASK / (oh * ow)).max(1);
            out.par_chunks_mut(per_task * oh * ow).enumerate().for_each_init(
                || (Vec::new(), Vec::new()),
                |scratch, (t, chunk)| {
                    for (c, o) in chunk.chunks_exact_mut(oh * ow).enumerate() {
                        work(scratch, (t * per_task + c, o));
                    }
                },
            );
        } else {
            let mut scratch = (Vec::new(), Vec::new());
            for (pl, o) in out.chunks_exact_mut(oh * ow).enumerate() {
                work(&mut scratch, (pl, o));
            }
        }
        return out;
    }
    // Huge window / padding fallback: clipped separable ranges without materialising padding.
    let range = |o: usize, s: usize, pad: usize, k: usize, n: usize| {
        (o * s).saturating_sub(pad)..(o * s + k).saturating_sub(pad).min(n)
    };
    let xs: Vec<_> = (0..ow).map(|ox| range(ox, sw, pw, kw, w)).collect();
    let ys: Vec<_> = (0..oh).map(|oy| range(oy, sh, ph, kh, h)).collect();
    let work = |rv: &mut Vec<f32>, (pl, o): (usize, &mut [f32])| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        rv.resize(h * ow, 0.0);
        for yy in 0..h {
            let row = &xp[yy * w..(yy + 1) * w];
            for (ox, xr) in xs.iter().enumerate() {
                rv[yy * ow + ox] = row[xr.clone()].iter().copied().sum();
            }
        }
        for (oy, yr) in ys.iter().enumerate() {
            let orow = &mut o[oy * ow..(oy + 1) * ow];
            orow.fill(0.0);
            for yy in yr.clone() {
                row_add(orow, &rv[yy * ow..(yy + 1) * ow]);
            }
            row_scale(orow, inv);
        }
    };
    if out.len() >= PAR_MIN && np > 1 {
        out.par_chunks_mut(oh * ow).enumerate().for_each_init(Vec::new, work);
    } else {
        let mut rv = Vec::new();
        for (pl, o) in out.chunks_exact_mut(oh * ow).enumerate() {
            work(&mut rv, (pl, o));
        }
    }
    out
}

/// Separable vertical + horizontal backward of [`avg_pool2d`].
pub(crate) fn avg_pool2d_backward(
    g: &[f32],
    (np, h, w): Planes,
    p: &Pool2dParams,
    [oh, ow]: [usize; 2],
) -> Vec<f32> {
    let mut gx = alloc_zeroed(np * h * w);
    if gx.is_empty() || oh * ow == 0 {
        return gx;
    }
    let [kh, kw] = p.kernel;
    let [sh, sw] = p.stride;
    let [ph, pw] = p.padding;
    let inv = 1.0 / (kh * kw) as f32;
    let range = |o: usize, s: usize, pad: usize, k: usize, n: usize| {
        (o * s).saturating_sub(pad)..(o * s + k).saturating_sub(pad).min(n)
    };
    let xs: Vec<_> = (0..ow).map(|ox| range(ox, sw, pw, kw, w)).collect();
    let ys: Vec<_> = (0..oh).map(|oy| range(oy, sh, ph, kh, h)).collect();
    let work = |rv: &mut Vec<f32>, (pl, gp): (usize, &mut [f32])| {
        let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
        rv.clear();
        rv.resize(h * ow, 0.0);
        for (oy, yr) in ys.iter().enumerate() {
            let grow = &gs[oy * ow..(oy + 1) * ow];
            for yy in yr.clone() {
                row_add_scaled(&mut rv[yy * ow..(yy + 1) * ow], grow, inv);
            }
        }
        for yy in 0..h {
            let rrow = &rv[yy * ow..(yy + 1) * ow];
            let gprow = &mut gp[yy * w..(yy + 1) * w];
            if sw == 2 && kw == 2 && pw == 0 && 2 * ow == w {
                duplicate_row2(rrow, gprow);
            } else {
                for (ox, xr) in xs.iter().enumerate() {
                    let v = rrow[ox];
                    for d in &mut gprow[xr.clone()] {
                        *d += v;
                    }
                }
            }
        }
    };
    if gx.len() >= PAR_MIN && np > 1 {
        let per_task = (POOL_TASK / (h * w)).max(1);
        gx.par_chunks_mut(per_task * h * w).enumerate().for_each_init(Vec::new, |rv, (t, chunk)| {
            for (c, gp) in chunk.chunks_exact_mut(h * w).enumerate() {
                work(rv, (t * per_task + c, gp));
            }
        });
    } else {
        let mut rv = Vec::new();
        for (pl, gp) in gx.chunks_exact_mut(h * w).enumerate() {
            work(&mut rv, (pl, gp));
        }
    }
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
    par_chunks(&mut out, oh * ow, &|pl, o| {
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
    par_chunks(&mut gx, h * w, &|pl, gp| {
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
