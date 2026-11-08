//! Pooling and nearest upsampling on contiguous NCHW planes.

use rayon::prelude::*;

use super::{alloc_uninit, alloc_zeroed};
use crate::backend::Pool2dParams;

/// `(planes, h, w)` of an NCHW tensor.
pub(crate) type Planes = (usize, usize, usize);

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
    out.par_chunks_mut(oh * ow).zip(idx.par_chunks_mut(oh * ow)).enumerate().for_each(|(pl, (o, ix))| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        for oy in 0..oh {
            let y0 = (oy * sh) as isize - ph as isize;
            let ys = y0.max(0) as usize..((y0 + kh as isize).min(h as isize)).max(0) as usize;
            for ox in 0..ow {
                let x0 = (ox * sw) as isize - pw as isize;
                let xs = x0.max(0) as usize..((x0 + kw as isize).min(w as isize)).max(0) as usize;
                let mut best = f32::NEG_INFINITY;
                let mut bi = -1i64;
                for yy in ys.clone() {
                    for xx in xs.clone() {
                        let v = xp[yy * w + xx];
                        if v > best || bi < 0 || v.is_nan() && !best.is_nan() {
                            best = v;
                            bi = (yy * w + xx) as i64;
                        }
                    }
                }
                o[oy * ow + ox] = best;
                ix[oy * ow + ox] = bi.max(0);
            }
        }
    });
    (out, idx)
}

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
