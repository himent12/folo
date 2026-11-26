//! L1-fused `RGB / warp → optional HSV → optional flip_lr → planar (u8 / f32)` AVX2/FMA and scalar kernels.

use super::HsvTables;
#[cfg(target_arch = "x86_64")]
use super::{hsv_8_i32_avx2, pack_16_i32_to_u8_avx2};
use crate::simd::bilinear::{ROUND, SHIFT, WONE};
#[cfg(target_arch = "x86_64")]
use crate::simd::bilinear::{interp_pair_avx2, pack_8_i32_to_u64_avx2, transpose_8_rgb0_i32_avx2};
#[cfg(target_arch = "x86_64")]
use crate::simd::planar::deinterleave_rgb16_avx2;
use crate::simd::planar::{rgb_to_planar_f32_div255, rgb_to_planar_u8};
use crate::simd::{DIV255_LUT, PlanarRow};
#[cfg(target_arch = "x86_64")]
use crate::simd::{INV255_HI, INV255_LO, has_avx2, has_avx2_fma};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Single-pass L1-fused `RGB row → optional HSV → optional flip_lr → planar (u8 or f32/255)`.
#[inline]
pub(crate) fn rgb_row_to_planar_fused(
    src: &[u8],
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let w = src.len() / 3;
    assert_eq!(src.len(), w * 3);
    match dst {
        PlanarRow::U8 { r, g, b } => assert!(r.len() == w && g.len() == w && b.len() == w),
        PlanarRow::F32Div255 { r, g, b } => assert!(r.len() == w && g.len() == w && b.len() == w),
    }
    match hsv {
        None => match dst {
            PlanarRow::U8 { r, g, b } => rgb_to_planar_u8(src, r, g, b, flip_lr),
            PlanarRow::F32Div255 { r, g, b } => rgb_to_planar_f32_div255(src, r, g, b, flip_lr),
        },
        Some(t) => {
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma()
                && w >= 16
                && let PlanarRow::F32Div255 { r, g, b } = dst
            {
                // SAFETY: CPU supports AVX2+FMA and row lengths are asserted to match `w`.
                unsafe { rgb_row_hsv_to_planar_f32_avx2_fma(src, t, flip_lr, r, g, b) };
                return;
            }
            #[cfg(target_arch = "x86_64")]
            if has_avx2() && w >= 16 {
                // SAFETY: CPU supports AVX2 and row lengths are asserted to match `w`.
                unsafe { rgb_row_hsv_to_planar_avx2(src, t, flip_lr, dst) };
                return;
            }
            rgb_row_hsv_to_planar_scalar(src, t, flip_lr, dst);
        }
    }
}

#[inline]
pub(crate) fn rgb_row_hsv_to_planar_scalar(
    src: &[u8],
    t: &HsvTables,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let w = src.len() / 3;
    let pixels = src.as_chunks::<3>().0;
    match dst {
        PlanarRow::U8 { r, g, b } => {
            if flip_lr {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    let d = w - 1 - x;
                    r[d] = o[0];
                    g[d] = o[1];
                    b[d] = o[2];
                }
            } else {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    r[x] = o[0];
                    g[x] = o[1];
                    b[x] = o[2];
                }
            }
        }
        PlanarRow::F32Div255 { r, g, b } => {
            if flip_lr {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    let d = w - 1 - x;
                    r[d] = DIV255_LUT[o[0] as usize];
                    g[d] = DIV255_LUT[o[1] as usize];
                    b[d] = DIV255_LUT[o[2] as usize];
                }
            } else {
                for (x, px) in pixels.iter().enumerate() {
                    let o = t.apply_pixel(*px);
                    r[x] = DIV255_LUT[o[0] as usize];
                    g[x] = DIV255_LUT[o[1] as usize];
                    b[x] = DIV255_LUT[o[2] as usize];
                }
            }
        }
    }
}

/// Single-pass L1-fused `warp_rows interior horizontal interpolation → optional HSV → optional flip_lr → planar`.
///
/// Writes source columns `x in x_start .. x_end` (where `rel_fu0` is the relative fixed-point position at `x_start`,
/// guaranteed in-bounds for every `x < x_end`) into destination row `dst` of total `width` pixels. Border ranges
/// outside `[x_start, x_end)` are handled by the caller.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn warp_interior_to_planar_fused(
    v: &[u32],
    rel_fu0: i64,
    du: i64,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    assert!(x_start <= x_end && x_end <= width);
    match dst {
        PlanarRow::U8 { r, g, b } => assert!(r.len() == width && g.len() == width && b.len() == width),
        PlanarRow::F32Div255 { r, g, b } => assert!(r.len() == width && g.len() == width && b.len() == width),
    }
    let count = x_end - x_start;
    if count == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cols = v.len() / 4;
        let hi_excl = (cols as i128 - 1) << 16;
        let last_rel = rel_fu0 as i128 + (count as i128 - 1) * du as i128;
        if has_avx2()
            && count >= 8
            && (2..=32000).contains(&cols)
            && du.abs() <= 4 << 16
            && (0..hi_excl).contains(&(rel_fu0 as i128))
            && (0..hi_excl).contains(&last_rel)
        {
            if has_avx2_fma()
                && let PlanarRow::F32Div255 { r, g, b } = dst
            {
                // SAFETY: CPU supports AVX2+FMA; `dst` lengths match `width`, `x_end <= width`, and endpoints prove bounds.
                unsafe {
                    warp_interior_to_planar_f32_fused_avx2_fma(
                        v,
                        rel_fu0 as i32,
                        du as i32,
                        width,
                        x_start,
                        x_end,
                        hsv,
                        flip_lr,
                        r,
                        g,
                        b,
                    )
                };
                return;
            }
            // SAFETY: CPU supports AVX2; `dst` lengths match `width`, `x_end <= width`, and by linearity both
            // endpoints of `rel_fu` lie in `0..((cols-1)<<16)` with `cols <= 32000` and `|du| <= 4<<16`.
            unsafe {
                warp_interior_to_planar_fused_avx2(
                    v,
                    rel_fu0 as i32,
                    du as i32,
                    width,
                    x_start,
                    x_end,
                    hsv,
                    flip_lr,
                    dst,
                )
            };
            return;
        }
    }
    warp_interior_to_planar_fused_scalar(v, rel_fu0, du, width, x_start, x_end, hsv, flip_lr, dst);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn warp_interior_to_planar_fused_scalar(
    v: &[u32],
    rel_fu0: i64,
    du: i64,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let mut rfu = rel_fu0;
    for x in x_start..x_end {
        let k = (rfu >> 16) as usize * 4;
        let wx = ((rfu >> 5) & (WONE as i64 - 1)) as u32;
        let iwx = WONE - wx;
        rfu += du;
        let mut px = [
            ((v[k] * iwx + v[k + 4] * wx + ROUND) >> SHIFT) as u8,
            ((v[k + 1] * iwx + v[k + 5] * wx + ROUND) >> SHIFT) as u8,
            ((v[k + 2] * iwx + v[k + 6] * wx + ROUND) >> SHIFT) as u8,
        ];
        if let Some(t) = hsv {
            px = t.apply_pixel(px);
        }
        let d = if flip_lr { width - 1 - x } else { x };
        match dst {
            PlanarRow::U8 { r, g, b } => {
                r[d] = px[0];
                g[d] = px[1];
                b[d] = px[2];
            }
            PlanarRow::F32Div255 { r, g, b } => {
                r[d] = DIV255_LUT[px[0] as usize];
                g[d] = DIV255_LUT[px[1] as usize];
                b[d] = DIV255_LUT[px[2] as usize];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_row_hsv_to_planar_avx2(src: &[u8], t: &HsvTables, flip_lr: bool, dst: &mut PlanarRow<'_>) {
    let w = src.len() / 3;
    let sp = src.as_ptr();
    let rev16 = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let v255 = _mm256_set1_ps(255.0);
    let mut x = 0usize;
    match dst {
        PlanarRow::U8 { r, g, b } => {
            let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
            while x + 16 <= w {
                // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 bytes writable at `dst_x`.
                unsafe {
                    let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
                    let (r_lo, g_lo, b_lo) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(vr),
                        _mm256_cvtepu8_epi32(vg),
                        _mm256_cvtepu8_epi32(vb),
                        t,
                    );
                    let (r_hi, g_hi, b_hi) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                        t,
                    );
                    let mut pr = pack_16_i32_to_u8_avx2(r_lo, r_hi);
                    let mut pg = pack_16_i32_to_u8_avx2(g_lo, g_hi);
                    let mut pb = pack_16_i32_to_u8_avx2(b_lo, b_hi);
                    let dst_x = if flip_lr {
                        pr = _mm_shuffle_epi8(pr, rev16);
                        pg = _mm_shuffle_epi8(pg, rev16);
                        pb = _mm_shuffle_epi8(pb, rev16);
                        w - x - 16
                    } else {
                        x
                    };
                    _mm_storeu_si128(rp.add(dst_x) as *mut __m128i, pr);
                    _mm_storeu_si128(gp.add(dst_x) as *mut __m128i, pg);
                    _mm_storeu_si128(bp.add(dst_x) as *mut __m128i, pb);
                }
                x += 16;
            }
            while x < w {
                let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
                let d = if flip_lr { w - 1 - x } else { x };
                r[d] = o[0];
                g[d] = o[1];
                b[d] = o[2];
                x += 1;
            }
        }
        PlanarRow::F32Div255 { r, g, b } => {
            let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
            while x + 16 <= w {
                // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s writable at `dst_x`.
                unsafe {
                    let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
                    let (mut r_lo, mut g_lo, mut b_lo) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(vr),
                        _mm256_cvtepu8_epi32(vg),
                        _mm256_cvtepu8_epi32(vb),
                        t,
                    );
                    let (mut r_hi, mut g_hi, mut b_hi) = hsv_8_i32_avx2(
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                        _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                        t,
                    );
                    if flip_lr {
                        r_lo = _mm256_permutevar8x32_epi32(r_lo, rev8);
                        g_lo = _mm256_permutevar8x32_epi32(g_lo, rev8);
                        b_lo = _mm256_permutevar8x32_epi32(b_lo, rev8);
                        r_hi = _mm256_permutevar8x32_epi32(r_hi, rev8);
                        g_hi = _mm256_permutevar8x32_epi32(g_hi, rev8);
                        b_hi = _mm256_permutevar8x32_epi32(b_hi, rev8);
                        let dst_x = w - x - 16;
                        _mm256_storeu_ps(rp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(r_hi), v255));
                        _mm256_storeu_ps(rp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(r_lo), v255));
                        _mm256_storeu_ps(gp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(g_hi), v255));
                        _mm256_storeu_ps(gp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(g_lo), v255));
                        _mm256_storeu_ps(bp.add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(b_hi), v255));
                        _mm256_storeu_ps(bp.add(dst_x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(b_lo), v255));
                    } else {
                        _mm256_storeu_ps(rp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(r_lo), v255));
                        _mm256_storeu_ps(rp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(r_hi), v255));
                        _mm256_storeu_ps(gp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(g_lo), v255));
                        _mm256_storeu_ps(gp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(g_hi), v255));
                        _mm256_storeu_ps(bp.add(x), _mm256_div_ps(_mm256_cvtepi32_ps(b_lo), v255));
                        _mm256_storeu_ps(bp.add(x + 8), _mm256_div_ps(_mm256_cvtepi32_ps(b_hi), v255));
                    }
                }
                x += 16;
            }
            while x < w {
                let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
                let d = if flip_lr { w - 1 - x } else { x };
                r[d] = DIV255_LUT[o[0] as usize];
                g[d] = DIV255_LUT[o[1] as usize];
                b[d] = DIV255_LUT[o[2] as usize];
                x += 1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn warp_interior_to_planar_fused_avx2(
    v: &[u32],
    rel_fu0: i32,
    du: i32,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    dst: &mut PlanarRow<'_>,
) {
    let vp = v.as_ptr();
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let v255 = _mm256_set1_ps(255.0);
    let du4 = du * 4;
    let mut rfu = rel_fu0;
    let mut x = x_start;

    while x + 8 <= x_end {
        let r0 = rfu;
        let r1 = r0 + du;
        let r2 = r1 + du;
        let r3 = r2 + du;
        let r4 = r0 + du4;
        let r5 = r1 + du4;
        let r6 = r2 + du4;
        let r7 = r3 + du4;
        rfu = r4 + du4;

        // SAFETY: caller guarantees every tap in `x_start..x_end` has `k + 1 < v.len() / 4`, and `x + 8 <= width`.
        unsafe {
            let p04 = interp_pair_avx2(
                vp,
                (r0 >> 16) as usize,
                (r0 >> 5) & 2047,
                (r4 >> 16) as usize,
                (r4 >> 5) & 2047,
                vround,
            );
            let p15 = interp_pair_avx2(
                vp,
                (r1 >> 16) as usize,
                (r1 >> 5) & 2047,
                (r5 >> 16) as usize,
                (r5 >> 5) & 2047,
                vround,
            );
            let p26 = interp_pair_avx2(
                vp,
                (r2 >> 16) as usize,
                (r2 >> 5) & 2047,
                (r6 >> 16) as usize,
                (r6 >> 5) & 2047,
                vround,
            );
            let p37 = interp_pair_avx2(
                vp,
                (r3 >> 16) as usize,
                (r3 >> 5) & 2047,
                (r7 >> 16) as usize,
                (r7 >> 5) & 2047,
                vround,
            );
            let (mut vr, mut vg, mut vb) = transpose_8_rgb0_i32_avx2(p04, p15, p26, p37);
            if let Some(t) = hsv {
                (vr, vg, vb) = hsv_8_i32_avx2(vr, vg, vb, t);
            }
            let dst_x = if flip_lr { width - x - 8 } else { x };
            match dst {
                PlanarRow::U8 { r, g, b } => {
                    (r.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vr, flip_lr));
                    (g.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vg, flip_lr));
                    (b.as_mut_ptr().add(dst_x) as *mut u64)
                        .write_unaligned(pack_8_i32_to_u64_avx2(vb, flip_lr));
                }
                PlanarRow::F32Div255 { r, g, b } => {
                    if flip_lr {
                        vr = _mm256_permutevar8x32_epi32(vr, rev8);
                        vg = _mm256_permutevar8x32_epi32(vg, rev8);
                        vb = _mm256_permutevar8x32_epi32(vb, rev8);
                    }
                    _mm256_storeu_ps(r.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vr), v255));
                    _mm256_storeu_ps(g.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vg), v255));
                    _mm256_storeu_ps(b.as_mut_ptr().add(dst_x), _mm256_div_ps(_mm256_cvtepi32_ps(vb), v255));
                }
            }
        }
        x += 8;
    }

    if x < x_end {
        warp_interior_to_planar_fused_scalar(v, rfu as i64, du as i64, width, x, x_end, hsv, flip_lr, dst);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn rgb_row_hsv_to_planar_f32_avx2_fma(
    src: &[u8],
    t: &HsvTables,
    flip_lr: bool,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let w = src.len() / 3;
    let sp = src.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let mut x = 0usize;
    while x + 16 <= w {
        // SAFETY: `x + 16 <= w`, so 48 bytes are readable from `sp` and 16 `f32`s writable at `dst_x`.
        unsafe {
            let (vr, vg, vb) = deinterleave_rgb16_avx2(sp.add(x * 3));
            let (mut r_lo, mut g_lo, mut b_lo) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(vr),
                _mm256_cvtepu8_epi32(vg),
                _mm256_cvtepu8_epi32(vb),
                t,
            );
            let (mut r_hi, mut g_hi, mut b_hi) = hsv_8_i32_avx2(
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vr)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vg)),
                _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(vb)),
                t,
            );
            let (dst_x, first, second) = if flip_lr {
                r_lo = _mm256_permutevar8x32_epi32(r_lo, rev8);
                g_lo = _mm256_permutevar8x32_epi32(g_lo, rev8);
                b_lo = _mm256_permutevar8x32_epi32(b_lo, rev8);
                r_hi = _mm256_permutevar8x32_epi32(r_hi, rev8);
                g_hi = _mm256_permutevar8x32_epi32(g_hi, rev8);
                b_hi = _mm256_permutevar8x32_epi32(b_hi, rev8);
                (w - x - 16, [(r_hi, rp), (g_hi, gp), (b_hi, bp)], [(r_lo, rp), (g_lo, gp), (b_lo, bp)])
            } else {
                (x, [(r_lo, rp), (g_lo, gp), (b_lo, bp)], [(r_hi, rp), (g_hi, gp), (b_hi, bp)])
            };
            for (v, dp) in first {
                let f = _mm256_cvtepi32_ps(v);
                _mm256_storeu_ps(dp.add(dst_x), _mm256_fmadd_ps(f, vhi, _mm256_mul_ps(f, vlo)));
            }
            for (v, dp) in second {
                let f = _mm256_cvtepi32_ps(v);
                _mm256_storeu_ps(dp.add(dst_x + 8), _mm256_fmadd_ps(f, vhi, _mm256_mul_ps(f, vlo)));
            }
        }
        x += 16;
    }
    while x < w {
        let o = t.apply_pixel([src[x * 3], src[x * 3 + 1], src[x * 3 + 2]]);
        let d = if flip_lr { w - 1 - x } else { x };
        r[d] = DIV255_LUT[o[0] as usize];
        g[d] = DIV255_LUT[o[1] as usize];
        b[d] = DIV255_LUT[o[2] as usize];
        x += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn warp_interior_to_planar_f32_fused_avx2_fma(
    v: &[u32],
    rel_fu0: i32,
    du: i32,
    width: usize,
    x_start: usize,
    x_end: usize,
    hsv: Option<&HsvTables>,
    flip_lr: bool,
    r: &mut [f32],
    g: &mut [f32],
    b: &mut [f32],
) {
    let vp = v.as_ptr();
    let (rp, gp, bp) = (r.as_mut_ptr(), g.as_mut_ptr(), b.as_mut_ptr());
    let vround = _mm256_set1_epi32(ROUND as i32);
    let rev8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    let vhi = _mm256_set1_ps(INV255_HI);
    let vlo = _mm256_set1_ps(INV255_LO);
    let du4 = du * 4;
    let mut rfu = rel_fu0;
    let mut x = x_start;

    while x + 8 <= x_end {
        let r0 = rfu;
        let r1 = r0 + du;
        let r2 = r1 + du;
        let r3 = r2 + du;
        let r4 = r0 + du4;
        let r5 = r1 + du4;
        let r6 = r2 + du4;
        let r7 = r3 + du4;
        rfu = r4 + du4;

        // SAFETY: caller guarantees every tap in `x_start..x_end` has `k + 1 < v.len() / 4`, and `x + 8 <= width`.
        unsafe {
            let p04 = interp_pair_avx2(
                vp,
                (r0 >> 16) as usize,
                (r0 >> 5) & 2047,
                (r4 >> 16) as usize,
                (r4 >> 5) & 2047,
                vround,
            );
            let p15 = interp_pair_avx2(
                vp,
                (r1 >> 16) as usize,
                (r1 >> 5) & 2047,
                (r5 >> 16) as usize,
                (r5 >> 5) & 2047,
                vround,
            );
            let p26 = interp_pair_avx2(
                vp,
                (r2 >> 16) as usize,
                (r2 >> 5) & 2047,
                (r6 >> 16) as usize,
                (r6 >> 5) & 2047,
                vround,
            );
            let p37 = interp_pair_avx2(
                vp,
                (r3 >> 16) as usize,
                (r3 >> 5) & 2047,
                (r7 >> 16) as usize,
                (r7 >> 5) & 2047,
                vround,
            );
            let (mut vr, mut vg, mut vb) = transpose_8_rgb0_i32_avx2(p04, p15, p26, p37);
            if let Some(t) = hsv {
                (vr, vg, vb) = hsv_8_i32_avx2(vr, vg, vb, t);
            }
            let dst_x = if flip_lr {
                vr = _mm256_permutevar8x32_epi32(vr, rev8);
                vg = _mm256_permutevar8x32_epi32(vg, rev8);
                vb = _mm256_permutevar8x32_epi32(vb, rev8);
                width - x - 8
            } else {
                x
            };
            let fr = _mm256_cvtepi32_ps(vr);
            let fg = _mm256_cvtepi32_ps(vg);
            let fb = _mm256_cvtepi32_ps(vb);
            _mm256_storeu_ps(rp.add(dst_x), _mm256_fmadd_ps(fr, vhi, _mm256_mul_ps(fr, vlo)));
            _mm256_storeu_ps(gp.add(dst_x), _mm256_fmadd_ps(fg, vhi, _mm256_mul_ps(fg, vlo)));
            _mm256_storeu_ps(bp.add(dst_x), _mm256_fmadd_ps(fb, vhi, _mm256_mul_ps(fb, vlo)));
        }
        x += 8;
    }

    if x < x_end {
        warp_interior_to_planar_fused_scalar(
            v,
            rfu as i64,
            du as i64,
            width,
            x,
            x_end,
            hsv,
            flip_lr,
            &mut PlanarRow::F32Div255 { r, g, b },
        );
    }
}
