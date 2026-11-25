//! Nearest-neighbour 2D upsampling and backward, with specialized AVX2 2x2 fast paths.

#[cfg(target_arch = "x86_64")]
use super::has_avx2;
use super::{POOL_TASK, Planes};
use crate::cpu::{PAR_MIN, alloc_uninit, alloc_zeroed, par_chunks};

/// Write `dst[2*i] = dst[2*i + 1] = src[i]` for `i < src.len()` (`dst.len() == 2 * src.len()`).
#[inline]
pub(super) fn duplicate_row2(src: &[f32], dst: &mut [f32]) {
    assert_eq!(dst.len(), 2 * src.len(), "duplicate_row2: length mismatch");
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { duplicate_row2_avx2(src, dst) };
    }
    for (d, &v) in dst.as_chunks_mut::<2>().0.iter_mut().zip(src) {
        d[0] = v;
        d[1] = v;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn duplicate_row2_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let (sc, sr) = src.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` is 8 floats and `d` is 16 floats.
        unsafe {
            let v = _mm256_loadu_ps(s.as_ptr());
            let lo = _mm256_unpacklo_ps(v, v);
            let hi = _mm256_unpackhi_ps(v, v);
            let o0 = _mm256_permute2f128_ps::<0x20>(lo, hi);
            let o1 = _mm256_permute2f128_ps::<0x31>(lo, hi);
            _mm256_storeu_ps(d.as_mut_ptr(), o0);
            _mm256_storeu_ps(d.as_mut_ptr().add(8), o1);
        }
    }
    for (d, &v) in dr.as_chunks_mut::<2>().0.iter_mut().zip(sr) {
        d[0] = v;
        d[1] = v;
    }
}

/// Contiguous 2x2 nearest-neighbour upsample of one `h x w` plane into `2h x 2w`, writing both output rows
/// per input row in one pass (AVX2 `unpacklo/hi + permute2f128` 8->16 floats per step).
fn upsample2x2_plane(xp: &[f32], h: usize, w: usize, o: &mut [f32]) {
    let ow = 2 * w;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { upsample2x2_plane_avx2(xp, h, w, o) };
    }
    for y in 0..h {
        let src = &xp[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        for ((p0, p1), &v) in d0.as_chunks_mut::<2>().0.iter_mut().zip(d1.as_chunks_mut::<2>().0).zip(src) {
            p0[0] = v;
            p0[1] = v;
            p1[0] = v;
            p1[1] = v;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn upsample2x2_plane_avx2(xp: &[f32], h: usize, w: usize, o: &mut [f32]) {
    use std::arch::x86_64::*;
    let ow = 2 * w;
    for y in 0..h {
        let src = &xp[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        let (sc, sr) = src.as_chunks::<8>();
        let (d0c, d0r) = d0.as_chunks_mut::<16>();
        let (d1c, d1r) = d1.as_chunks_mut::<16>();
        for ((s, p0), p1) in sc.iter().zip(d0c.iter_mut()).zip(d1c.iter_mut()) {
            // SAFETY: `s` is 8 floats, `p0` and `p1` are 16 floats.
            unsafe {
                let v = _mm256_loadu_ps(s.as_ptr());
                let lo = _mm256_unpacklo_ps(v, v);
                let hi = _mm256_unpackhi_ps(v, v);
                let o0 = _mm256_permute2f128_ps::<0x20>(lo, hi);
                let o1 = _mm256_permute2f128_ps::<0x31>(lo, hi);
                _mm256_storeu_ps(p0.as_mut_ptr(), o0);
                _mm256_storeu_ps(p0.as_mut_ptr().add(8), o1);
                _mm256_storeu_ps(p1.as_mut_ptr(), o0);
                _mm256_storeu_ps(p1.as_mut_ptr().add(8), o1);
            }
        }
        for ((p0, p1), &v) in d0r.as_chunks_mut::<2>().0.iter_mut().zip(d1r.as_chunks_mut::<2>().0).zip(sr) {
            p0[0] = v;
            p0[1] = v;
            p1[0] = v;
            p1[1] = v;
        }
    }
}

/// Contiguous 2x2 nearest-neighbour upsample backward of one `2h x 2w` plane into `h x w`, overwriting every
/// output element in one pass (`(r0[2x] + r0[2x+1]) + (r1[2x] + r1[2x+1])`, AVX2 `hadd_ps + permute4x64`).
fn upsample2x2_bwd_plane(gs: &[f32], h: usize, w: usize, gp: &mut [f32]) {
    let ow = 2 * w;
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { upsample2x2_bwd_plane_avx2(gs, h, w, gp) };
    }
    for y in 0..h {
        let r0 = &gs[2 * y * ow..(2 * y + 1) * ow];
        let r1 = &gs[(2 * y + 1) * ow..(2 * y + 2) * ow];
        let dst = &mut gp[y * w..(y + 1) * w];
        for (x, d) in dst.iter_mut().enumerate() {
            *d = (r0[2 * x] + r0[2 * x + 1]) + (r1[2 * x] + r1[2 * x + 1]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn upsample2x2_bwd_plane_avx2(gs: &[f32], h: usize, w: usize, gp: &mut [f32]) {
    use std::arch::x86_64::*;
    let ow = 2 * w;
    for y in 0..h {
        let r0 = &gs[2 * y * ow..(2 * y + 1) * ow];
        let r1 = &gs[(2 * y + 1) * ow..(2 * y + 2) * ow];
        let dst = &mut gp[y * w..(y + 1) * w];
        let (dc, dr) = dst.as_chunks_mut::<8>();
        let (r0c, r0r) = r0.as_chunks::<16>();
        let (r1c, r1r) = r1.as_chunks::<16>();
        for ((d, p0), p1) in dc.iter_mut().zip(r0c).zip(r1c) {
            // SAFETY: `p0` and `p1` are 16 floats and `d` is 8 floats.
            unsafe {
                let s0 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                    _mm256_hadd_ps(_mm256_loadu_ps(p0.as_ptr()), _mm256_loadu_ps(p0.as_ptr().add(8))),
                )));
                let s1 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                    _mm256_hadd_ps(_mm256_loadu_ps(p1.as_ptr()), _mm256_loadu_ps(p1.as_ptr().add(8))),
                )));
                _mm256_storeu_ps(d.as_mut_ptr(), _mm256_add_ps(s0, s1));
            }
        }
        for ((d, p0), p1) in dr.iter_mut().zip(r0r.as_chunks::<2>().0).zip(r1r.as_chunks::<2>().0) {
            *d = (p0[0] + p0[1]) + (p1[0] + p1[1]);
        }
    }
}

pub(crate) fn upsample_nearest2d(x: &[f32], (np, h, w): Planes, [sh, sw]: [usize; 2]) -> Vec<f32> {
    let (oh, ow) = (h * sh, w * sw);
    let mut out = alloc_uninit(np * oh * ow);
    if out.is_empty() || h * w == 0 {
        return out;
    }
    let plane_fn = |pl: usize, o: &mut [f32]| {
        let xp = &x[pl * h * w..(pl + 1) * h * w];
        if sh == 2 && sw == 2 {
            upsample2x2_plane(xp, h, w, o);
            return;
        }
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
    };
    if out.len() >= PAR_MIN && np > 1 {
        let per_task = (POOL_TASK / (oh * ow)).max(1);
        par_chunks(&mut out, per_task * oh * ow, &|t, chunk| {
            for (c, o) in chunk.chunks_exact_mut(oh * ow).enumerate() {
                plane_fn(t * per_task + c, o);
            }
        });
    } else {
        for (pl, o) in out.chunks_exact_mut(oh * ow).enumerate() {
            plane_fn(pl, o);
        }
    }
    out
}

pub(crate) fn upsample_nearest2d_backward(g: &[f32], (np, h, w): Planes, [sh, sw]: [usize; 2]) -> Vec<f32> {
    let (oh, ow) = (h * sh, w * sw);
    if np * h * w == 0 {
        return alloc_uninit(0);
    }
    if sh == 2 && sw == 2 {
        let mut gx = alloc_uninit(np * h * w);
        let plane_fn = |pl: usize, gp: &mut [f32]| {
            let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
            upsample2x2_bwd_plane(gs, h, w, gp);
        };
        if g.len() >= PAR_MIN && np > 1 {
            let per_task = (POOL_TASK / (h * w)).max(1);
            par_chunks(&mut gx, per_task * h * w, &|t, chunk| {
                for (c, gp) in chunk.chunks_exact_mut(h * w).enumerate() {
                    plane_fn(t * per_task + c, gp);
                }
            });
        } else {
            for (pl, gp) in gx.chunks_exact_mut(h * w).enumerate() {
                plane_fn(pl, gp);
            }
        }
        return gx;
    }
    let mut gx = alloc_zeroed(np * h * w);
    let plane_fn = |pl: usize, gp: &mut [f32]| {
        let gs = &g[pl * oh * ow..(pl + 1) * oh * ow];
        for oy in 0..oh {
            let row = &gs[oy * ow..(oy + 1) * ow];
            let dst = &mut gp[(oy / sh) * w..(oy / sh + 1) * w];
            for (xx, d) in dst.iter_mut().enumerate() {
                *d += row[xx * sw..(xx + 1) * sw].iter().sum::<f32>();
            }
        }
    };
    if g.len() >= PAR_MIN && np > 1 {
        let per_task = (POOL_TASK / (h * w)).max(1);
        par_chunks(&mut gx, per_task * h * w, &|t, chunk| {
            for (c, gp) in chunk.chunks_exact_mut(h * w).enumerate() {
                plane_fn(t * per_task + c, gp);
            }
        });
    } else {
        for (pl, gp) in gx.chunks_exact_mut(h * w).enumerate() {
            plane_fn(pl, gp);
        }
    }
    gx
}
