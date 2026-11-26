//! Wasm SIMD128 (`f32x4`) and portable scalar 2D NCHW nearest-neighbour and bilinear upsampling kernels.

/// 2D nearest-neighbour upsampling of contiguous NCHW `src` (`[n, c, h, w]`) into `dst` (`[n, c, oh, ow]`).
///
/// Uses PyTorch-compatible floor mapping `sy = (oy * h) / oh`, `sx = (ox * w) / ow`, with a dedicated
/// Wasm SIMD128 2×2 doubling fast path (`u32x4_shuffle` duplicating 4 floats -> 8 floats into two output rows).
///
/// # Panics
/// If slice lengths do not match `shape` and `[n, c, oh, ow]`, or if `oh > 0 && h == 0` / `ow > 0 && w == 0`.
pub fn upsample_nearest2d_nchw(src: &[f32], shape: [usize; 4], out_hw: [usize; 2], dst: &mut [f32]) {
    let [n, c, h, w] = shape;
    let [oh, ow] = out_hw;
    let planes = n.checked_mul(c).expect("upsample_nearest2d_nchw: n * c overflow");
    let in_plane = h.checked_mul(w).expect("upsample_nearest2d_nchw: h * w overflow");
    let out_plane = oh.checked_mul(ow).expect("upsample_nearest2d_nchw: oh * ow overflow");
    assert_eq!(src.len(), planes.checked_mul(in_plane).expect("upsample_nearest2d_nchw: src overflow"));
    assert_eq!(dst.len(), planes.checked_mul(out_plane).expect("upsample_nearest2d_nchw: dst overflow"));
    if planes == 0 || out_plane == 0 {
        return;
    }
    assert!(h > 0 && w > 0, "upsample_nearest2d_nchw: non-empty output requires non-empty input");

    if oh == h * 2 && ow == w * 2 {
        for p in 0..planes {
            let s = &src[p * in_plane..(p + 1) * in_plane];
            let d = &mut dst[p * out_plane..(p + 1) * out_plane];
            upsample_nearest2x2_plane(s, h, w, d);
        }
        return;
    }

    let x_map: Vec<usize> = (0..ow).map(|ox| ((ox as u128 * w as u128) / ow as u128) as usize).collect();
    let y_map: Vec<usize> = (0..oh).map(|oy| ((oy as u128 * h as u128) / oh as u128) as usize).collect();
    for p in 0..planes {
        let s = &src[p * in_plane..(p + 1) * in_plane];
        let d = &mut dst[p * out_plane..(p + 1) * out_plane];
        for (oy, &sy) in y_map.iter().enumerate() {
            let srow = &s[sy * w..(sy + 1) * w];
            let drow = &mut d[oy * ow..(oy + 1) * ow];
            if ow == w {
                drow.copy_from_slice(srow);
            } else {
                for (slot, &sx) in drow.iter_mut().zip(&x_map) {
                    *slot = srow[sx];
                }
            }
        }
    }
}

fn upsample_nearest2x2_plane(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd::upsample_nearest2x2_plane(src, h, w, dst);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let ow = 2 * w;
        for y in 0..h {
            let srow = &src[y * w..(y + 1) * w];
            let (d0, d1) = dst[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
            for (x, &v) in srow.iter().enumerate() {
                d0[2 * x] = v;
                d0[2 * x + 1] = v;
                d1[2 * x] = v;
                d1[2 * x + 1] = v;
            }
        }
    }
}

/// Precomputed 1D bilinear sampling coordinate table.
struct Bilinear1d {
    i0: Vec<usize>,
    i1: Vec<usize>,
    w: Vec<f32>,
}

fn bilinear_axis(in_len: usize, out_len: usize, align_corners: bool) -> Bilinear1d {
    let mut i0 = Vec::with_capacity(out_len);
    let mut i1 = Vec::with_capacity(out_len);
    let mut w = Vec::with_capacity(out_len);
    let last = in_len - 1;
    let scale = if align_corners {
        if out_len > 1 { (in_len - 1) as f64 / (out_len - 1) as f64 } else { 0.0 }
    } else {
        in_len as f64 / out_len as f64
    };
    for o in 0..out_len {
        let real = if align_corners { o as f64 * scale } else { (o as f64 + 0.5) * scale - 0.5 };
        let clamped = real.clamp(0.0, last as f64);
        let lo = clamped.floor() as usize;
        let hi = (lo + 1).min(last);
        let frac = (clamped - lo as f64) as f32;
        i0.push(lo);
        i1.push(hi);
        w.push(frac);
    }
    Bilinear1d { i0, i1, w }
}

/// 2D bilinear upsampling of contiguous NCHW `src` (`[n, c, h, w]`) into `dst` (`[n, c, oh, ow]`).
///
/// Matches PyTorch `interpolate(..., mode="bilinear", align_corners=align_corners)`. Uses Wasm SIMD128
/// (`f32x4`) for vertical row blending and 4-wide horizontal interpolation when compiled with `+simd128`.
///
/// # Panics
/// If slice lengths do not match `shape` and `[n, c, oh, ow]`, or if `oh > 0 && h == 0` / `ow > 0 && w == 0`.
pub fn upsample_bilinear2d_nchw(
    src: &[f32],
    shape: [usize; 4],
    out_hw: [usize; 2],
    align_corners: bool,
    dst: &mut [f32],
) {
    let [n, c, h, w] = shape;
    let [oh, ow] = out_hw;
    let planes = n.checked_mul(c).expect("upsample_bilinear2d_nchw: n * c overflow");
    let in_plane = h.checked_mul(w).expect("upsample_bilinear2d_nchw: h * w overflow");
    let out_plane = oh.checked_mul(ow).expect("upsample_bilinear2d_nchw: oh * ow overflow");
    assert_eq!(src.len(), planes.checked_mul(in_plane).expect("upsample_bilinear2d_nchw: src overflow"));
    assert_eq!(dst.len(), planes.checked_mul(out_plane).expect("upsample_bilinear2d_nchw: dst overflow"));
    if planes == 0 || out_plane == 0 {
        return;
    }
    assert!(h > 0 && w > 0, "upsample_bilinear2d_nchw: non-empty output requires non-empty input");

    let x_tab = bilinear_axis(w, ow, align_corners);
    let y_tab = bilinear_axis(h, oh, align_corners);
    let mut v_row = vec![0.0f32; w];

    for p in 0..planes {
        let s = &src[p * in_plane..(p + 1) * in_plane];
        let d = &mut dst[p * out_plane..(p + 1) * out_plane];
        for oy in 0..oh {
            let (y0, y1, wy) = (y_tab.i0[oy], y_tab.i1[oy], y_tab.w[oy]);
            let r0 = &s[y0 * w..(y0 + 1) * w];
            let blended: &[f32] = if y0 == y1 || wy == 0.0 {
                r0
            } else {
                let r1 = &s[y1 * w..(y1 + 1) * w];
                vblend_row(r0, r1, wy, &mut v_row);
                &v_row
            };
            let drow = &mut d[oy * ow..(oy + 1) * ow];
            hinterp_row(blended, &x_tab.i0, &x_tab.i1, &x_tab.w, drow);
        }
    }
}

fn vblend_row(r0: &[f32], r1: &[f32], wy: f32, dst: &mut [f32]) {
    debug_assert_eq!(r0.len(), r1.len());
    debug_assert_eq!(r0.len(), dst.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd::vblend_row(r0, r1, wy, dst);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for ((d, &a), &b) in dst.iter_mut().zip(r0).zip(r1) {
        *d = a + (b - a) * wy;
    }
}

fn hinterp_row(v: &[f32], x0: &[usize], x1: &[usize], wx: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), x0.len());
    debug_assert_eq!(dst.len(), x1.len());
    debug_assert_eq!(dst.len(), wx.len());
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    wasm_simd::hinterp_row(v, x0, x1, wx, dst);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for i in 0..dst.len() {
        let a = v[x0[i]];
        let b = v[x1[i]];
        dst[i] = a + (b - a) * wx[i];
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use core::arch::wasm32::*;

    pub(super) fn upsample_nearest2x2_plane(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
        let ow = 2 * w;
        for y in 0..h {
            let srow = &src[y * w..(y + 1) * w];
            let (d0, d1) = dst[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
            let (sc, sr) = srow.as_chunks::<4>();
            let (d0c, d0r) = d0.as_chunks_mut::<8>();
            let (d1c, d1r) = d1.as_chunks_mut::<8>();
            for ((s, p0), p1) in sc.iter().zip(d0c.iter_mut()).zip(d1c.iter_mut()) {
                // SAFETY: `s` is 4 floats, `p0` and `p1` are 8 floats.
                unsafe {
                    let v = v128_load(s.as_ptr() as *const v128);
                    let o0 = u32x4_shuffle::<0, 0, 1, 1>(v, v);
                    let o1 = u32x4_shuffle::<2, 2, 3, 3>(v, v);
                    v128_store(p0.as_mut_ptr() as *mut v128, o0);
                    v128_store(p0.as_mut_ptr().add(4) as *mut v128, o1);
                    v128_store(p1.as_mut_ptr() as *mut v128, o0);
                    v128_store(p1.as_mut_ptr().add(4) as *mut v128, o1);
                }
            }
            for ((p0, p1), &v) in
                d0r.as_chunks_mut::<2>().0.iter_mut().zip(d1r.as_chunks_mut::<2>().0).zip(sr)
            {
                p0[0] = v;
                p0[1] = v;
                p1[0] = v;
                p1[1] = v;
            }
        }
    }

    pub(super) fn vblend_row(r0: &[f32], r1: &[f32], wy: f32, dst: &mut [f32]) {
        let vwy = f32x4_splat(wy);
        let (c0, t0) = r0.as_chunks::<4>();
        let (c1, t1) = r1.as_chunks::<4>();
        let (cd, td) = dst.as_chunks_mut::<4>();
        for ((a, b), d) in c0.iter().zip(c1).zip(cd.iter_mut()) {
            // SAFETY: each chunk is 4 floats.
            unsafe {
                let va = v128_load(a.as_ptr() as *const v128);
                let vb = v128_load(b.as_ptr() as *const v128);
                let out = f32x4_add(va, f32x4_mul(f32x4_sub(vb, va), vwy));
                v128_store(d.as_mut_ptr() as *mut v128, out);
            }
        }
        for ((d, &a), &b) in td.iter_mut().zip(t0).zip(t1) {
            *d = a + (b - a) * wy;
        }
    }

    pub(super) fn hinterp_row(v: &[f32], x0: &[usize], x1: &[usize], wx: &[f32], dst: &mut [f32]) {
        let n = dst.len();
        let chunks = n / 4;
        for c in 0..chunks {
            let k = c * 4;
            let va = f32x4(v[x0[k]], v[x0[k + 1]], v[x0[k + 2]], v[x0[k + 3]]);
            let vb = f32x4(v[x1[k]], v[x1[k + 1]], v[x1[k + 2]], v[x1[k + 3]]);
            // SAFETY: `k + 4 <= n == wx.len() == dst.len()`.
            unsafe {
                let vw = v128_load(wx.as_ptr().add(k) as *const v128);
                let out = f32x4_add(va, f32x4_mul(f32x4_sub(vb, va), vw));
                v128_store(dst.as_mut_ptr().add(k) as *mut v128, out);
            }
        }
        for i in chunks * 4..n {
            let a = v[x0[i]];
            let b = v[x1[i]];
            dst[i] = a + (b - a) * wx[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_upsample_covers_2x2_and_arbitrary_sizes() {
        // Exact 2x2 doubling (odd & > 4 width to cover SIMD + tail).
        let src: Vec<f32> = (0..2 * 3 * 5).map(|i| i as f32 * 0.5 - 3.0).collect();
        let mut got = vec![0.0f32; 2 * 6 * 10];
        upsample_nearest2d_nchw(&src, [1, 2, 3, 5], [6, 10], &mut got);
        for c in 0..2 {
            for y in 0..6 {
                for x in 0..10 {
                    assert_eq!(got[(c * 6 + y) * 10 + x], src[(c * 3 + y / 2) * 5 + x / 2]);
                }
            }
        }

        // 1x1 -> 4x5 constant fill.
        let one = [7.25f32];
        let mut out = vec![0.0f32; 20];
        upsample_nearest2d_nchw(&one, [1, 1, 1, 1], [4, 5], &mut out);
        assert!(out.iter().all(|&v| v == 7.25));

        // Arbitrary downscale & upscale.
        let mut out2 = vec![0.0f32; 2 * 5 * 7];
        upsample_nearest2d_nchw(&src, [1, 2, 3, 5], [5, 7], &mut out2);
        for c in 0..2 {
            for oy in 0..5 {
                for ox in 0..7 {
                    let sy = oy * 3 / 5;
                    let sx = ox * 5 / 7;
                    assert_eq!(out2[(c * 5 + oy) * 7 + ox], src[(c * 3 + sy) * 5 + sx]);
                }
            }
        }
    }

    #[test]
    fn bilinear_upsample_preserves_constants_and_linear_ramps() {
        // Constant plane stays bit-exact for both align_corners settings.
        let solid = vec![3.5f32; 2 * 3 * 4];
        for align in [false, true] {
            let mut out = vec![0.0f32; 2 * 7 * 9];
            upsample_bilinear2d_nchw(&solid, [1, 2, 3, 4], [7, 9], align, &mut out);
            assert!(out.iter().all(|&v| v == 3.5));
        }

        // Bilinear with align_corners=true preserves an exact affine ramp `f(y, x) = 2*y + 3*x`.
        let (h, w, oh, ow) = (4usize, 5usize, 7usize, 9usize);
        let mut ramp = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                ramp[y * w + x] = 2.0 * y as f32 + 3.0 * x as f32;
            }
        }
        let mut out = vec![0.0f32; oh * ow];
        upsample_bilinear2d_nchw(&ramp, [1, 1, h, w], [oh, ow], true, &mut out);
        for oy in 0..oh {
            for ox in 0..ow {
                let fy = oy as f32 * (h - 1) as f32 / (oh - 1) as f32;
                let fx = ox as f32 * (w - 1) as f32 / (ow - 1) as f32;
                let want = 2.0 * fy + 3.0 * fx;
                assert!(
                    (out[oy * ow + ox] - want).abs() <= 1e-5,
                    "({oy},{ox}): got={} want={want}",
                    out[oy * ow + ox]
                );
            }
        }

        // Exact 2x2 with align_corners=false matches known PyTorch quarter-pixel weights.
        let grid = [0.0f32, 4.0, 8.0, 12.0];
        let mut up = vec![0.0f32; 16];
        upsample_bilinear2d_nchw(&grid, [1, 1, 2, 2], [4, 4], false, &mut up);
        let expected_row0 = [0.0f32, 1.0, 3.0, 4.0];
        for x in 0..4 {
            assert!((up[x] - expected_row0[x]).abs() < 1e-6);
            assert!((up[4 + x] - (expected_row0[x] + 2.0)).abs() < 1e-6);
            assert!((up[8 + x] - (expected_row0[x] + 6.0)).abs() < 1e-6);
            assert!((up[12 + x] - (expected_row0[x] + 8.0)).abs() < 1e-6);
        }

        // 1x1 -> 3x3 and empty tensors.
        let mut up1 = vec![0.0f32; 9];
        upsample_bilinear2d_nchw(&[-1.25], [1, 1, 1, 1], [3, 3], true, &mut up1);
        assert!(up1.iter().all(|&v| v == -1.25));
        let mut empty = Vec::new();
        upsample_bilinear2d_nchw(&[], [0, 2, 4, 4], [8, 8], false, &mut empty);
    }
}
