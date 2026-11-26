//! Wasm SIMD128 (`f32x4`) and portable scalar 2D NCHW max and average pooling kernels.

/// Parameters for 2D spatial pooling over an NCHW tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool2dSpec {
    /// Window size `[kh, kw]` (both must be >= 1).
    pub kernel: [usize; 2],
    /// Stride `[sh, sw]` (both must be >= 1).
    pub stride: [usize; 2],
    /// Zero-padding `[ph, pw]` applied symmetrically on top/bottom and left/right.
    pub padding: [usize; 2],
    /// For average pooling: when `true`, divide by `kh * kw`; when `false`, divide by the number of in-bounds
    /// pixels inside each window.
    pub count_include_pad: bool,
}

impl Pool2dSpec {
    /// Creates a pooling spec with `count_include_pad = false`.
    pub const fn new(kernel: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        Self { kernel, stride, padding, count_include_pad: false }
    }

    /// Computes output spatial dimensions `[oh, ow]` for input `[h, w]`, or `None` if invalid.
    pub fn output_hw(&self, h: usize, w: usize) -> Option<[usize; 2]> {
        let ([kh, kw], [sh, sw], [ph, pw]) = (self.kernel, self.stride, self.padding);
        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return None;
        }
        let eh = h.checked_add(ph.checked_mul(2)?)?;
        let ew = w.checked_add(pw.checked_mul(2)?)?;
        if eh < kh || ew < kw {
            return None;
        }
        Some([(eh - kh) / sh + 1, (ew - kw) / sw + 1])
    }
}

/// Scalar NaN-propagating max step matching IEEE first-maximum / first-NaN order.
#[inline(always)]
fn step_max(acc: &mut f32, v: f32) {
    if v > *acc || (v.is_nan() && !acc.is_nan()) {
        *acc = v;
    }
}

/// 2D max pooling over contiguous NCHW `src` (`[n, c, h, w]`) into `dst` (`[n, c, oh, ow]`).
///
/// Uses Wasm SIMD128 (`f32x4`) fast paths for 2×2 stride-2 (no padding) and stride-1 interior windows when
/// compiled with `+simd128`, with a portable scalar fallback producing identical results.
///
/// # Panics
/// If `spec` is invalid for `[h, w]` or slice lengths do not match `shape` and `[n, c, oh, ow]`.
pub fn max_pool2d_nchw(src: &[f32], shape: [usize; 4], spec: &Pool2dSpec, dst: &mut [f32]) {
    let [n, c, h, w] = shape;
    let [oh, ow] = spec.output_hw(h, w).expect("max_pool2d_nchw: invalid pooling spec for input size");
    let planes = n.checked_mul(c).expect("max_pool2d_nchw: n * c overflow");
    let in_plane = h.checked_mul(w).expect("max_pool2d_nchw: h * w overflow");
    let out_plane = oh.checked_mul(ow).expect("max_pool2d_nchw: oh * ow overflow");
    assert_eq!(src.len(), planes.checked_mul(in_plane).expect("max_pool2d_nchw: src size overflow"));
    assert_eq!(dst.len(), planes.checked_mul(out_plane).expect("max_pool2d_nchw: dst size overflow"));
    if planes == 0 || out_plane == 0 {
        return;
    }
    for p in 0..planes {
        let s = &src[p * in_plane..(p + 1) * in_plane];
        let d = &mut dst[p * out_plane..(p + 1) * out_plane];
        max_pool2d_plane(s, h, w, spec, oh, ow, d);
    }
}

fn max_pool2d_plane(
    src: &[f32],
    h: usize,
    w: usize,
    spec: &Pool2dSpec,
    oh: usize,
    ow: usize,
    dst: &mut [f32],
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if spec.kernel == [2, 2] && spec.stride == [2, 2] && spec.padding == [0, 0] {
            wasm_simd::max_pool2x2_s2_plane(src, w, oh, ow, dst);
            return;
        }
        if spec.stride[1] == 1 {
            wasm_simd::max_pool_sw1_plane(src, h, w, spec, oh, ow, dst);
            return;
        }
    }
    max_pool2d_plane_scalar(src, h, w, spec, oh, ow, dst);
}

fn max_pool2d_plane_scalar(
    src: &[f32],
    h: usize,
    w: usize,
    spec: &Pool2dSpec,
    oh: usize,
    ow: usize,
    dst: &mut [f32],
) {
    let ([kh, kw], [sh, sw], [ph, pw]) = (spec.kernel, spec.stride, spec.padding);
    let (hi, wi) = (h as isize, w as isize);
    for oy in 0..oh {
        let y0 = (oy * sh) as isize - ph as isize;
        let iy0 = y0.clamp(0, hi) as usize;
        let iy1 = (y0 + kh as isize).clamp(0, hi) as usize;
        let drow = &mut dst[oy * ow..(oy + 1) * ow];
        for (ox, slot) in drow.iter_mut().enumerate() {
            let x0 = (ox * sw) as isize - pw as isize;
            let ix0 = x0.clamp(0, wi) as usize;
            let ix1 = (x0 + kw as isize).clamp(0, wi) as usize;
            let mut m = f32::NEG_INFINITY;
            let mut init = false;
            for iy in iy0..iy1 {
                let srow = &src[iy * w..(iy + 1) * w];
                for &v in &srow[ix0..ix1] {
                    if !init {
                        m = v;
                        init = true;
                    } else {
                        step_max(&mut m, v);
                    }
                }
            }
            *slot = m;
        }
    }
}

/// 2D average pooling over contiguous NCHW `src` (`[n, c, h, w]`) into `dst` (`[n, c, oh, ow]`).
///
/// Uses Wasm SIMD128 (`f32x4`) fast paths for 2×2 stride-2 (no padding) and stride-1 interior windows when
/// compiled with `+simd128`, with a portable scalar fallback producing identical results.
///
/// # Panics
/// If `spec` is invalid for `[h, w]` or slice lengths do not match `shape` and `[n, c, oh, ow]`.
pub fn avg_pool2d_nchw(src: &[f32], shape: [usize; 4], spec: &Pool2dSpec, dst: &mut [f32]) {
    let [n, c, h, w] = shape;
    let [oh, ow] = spec.output_hw(h, w).expect("avg_pool2d_nchw: invalid pooling spec for input size");
    let planes = n.checked_mul(c).expect("avg_pool2d_nchw: n * c overflow");
    let in_plane = h.checked_mul(w).expect("avg_pool2d_nchw: h * w overflow");
    let out_plane = oh.checked_mul(ow).expect("avg_pool2d_nchw: oh * ow overflow");
    assert_eq!(src.len(), planes.checked_mul(in_plane).expect("avg_pool2d_nchw: src size overflow"));
    assert_eq!(dst.len(), planes.checked_mul(out_plane).expect("avg_pool2d_nchw: dst size overflow"));
    if planes == 0 || out_plane == 0 {
        return;
    }
    for p in 0..planes {
        let s = &src[p * in_plane..(p + 1) * in_plane];
        let d = &mut dst[p * out_plane..(p + 1) * out_plane];
        avg_pool2d_plane(s, h, w, spec, oh, ow, d);
    }
}

fn avg_pool2d_plane(
    src: &[f32],
    h: usize,
    w: usize,
    spec: &Pool2dSpec,
    oh: usize,
    ow: usize,
    dst: &mut [f32],
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if spec.kernel == [2, 2] && spec.stride == [2, 2] && spec.padding == [0, 0] {
            wasm_simd::avg_pool2x2_s2_plane(src, w, oh, ow, dst);
            return;
        }
        if spec.stride[1] == 1 {
            wasm_simd::avg_pool_sw1_plane(src, h, w, spec, oh, ow, dst);
            return;
        }
    }
    avg_pool2d_plane_scalar(src, h, w, spec, oh, ow, dst);
}

fn avg_pool2d_plane_scalar(
    src: &[f32],
    h: usize,
    w: usize,
    spec: &Pool2dSpec,
    oh: usize,
    ow: usize,
    dst: &mut [f32],
) {
    let ([kh, kw], [sh, sw], [ph, pw]) = (spec.kernel, spec.stride, spec.padding);
    let full_denom = (kh as f32) * (kw as f32);
    let (hi, wi) = (h as isize, w as isize);
    for oy in 0..oh {
        let y0 = (oy * sh) as isize - ph as isize;
        let iy0 = y0.clamp(0, hi) as usize;
        let iy1 = (y0 + kh as isize).clamp(0, hi) as usize;
        let valid_h = (iy1 - iy0) as f32;
        let drow = &mut dst[oy * ow..(oy + 1) * ow];
        for (ox, slot) in drow.iter_mut().enumerate() {
            let x0 = (ox * sw) as isize - pw as isize;
            let ix0 = x0.clamp(0, wi) as usize;
            let ix1 = (x0 + kw as isize).clamp(0, wi) as usize;
            let mut sum = 0.0f32;
            for iy in iy0..iy1 {
                let srow = &src[iy * w..(iy + 1) * w];
                for &v in &srow[ix0..ix1] {
                    sum += v;
                }
            }
            let denom = if spec.count_include_pad { full_denom } else { valid_h * ((ix1 - ix0) as f32) };
            *slot = if denom > 0.0 { sum / denom } else { 0.0 };
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd {
    use core::arch::wasm32::*;

    use super::{Pool2dSpec, step_max};

    #[inline(always)]
    fn max_step4(acc: v128, v: v128) -> v128 {
        let nle = v128_not(f32x4_le(v, acc));
        let acc_ord = f32x4_eq(acc, acc);
        v128_bitselect(v, acc, v128_and(nle, acc_ord))
    }

    pub(super) fn max_pool2x2_s2_plane(src: &[f32], w: usize, oh: usize, ow: usize, dst: &mut [f32]) {
        for oy in 0..oh {
            let r0 = &src[(2 * oy) * w..(2 * oy + 1) * w];
            let r1 = &src[(2 * oy + 1) * w..(2 * oy + 2) * w];
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            let mut ox = 0;
            // SAFETY: `ox + 4 <= ow` implies `2 * ox + 8 <= 2 * ow <= w`.
            unsafe {
                while ox + 4 <= ow {
                    let p0 = r0.as_ptr().add(2 * ox);
                    let p1 = r1.as_ptr().add(2 * ox);
                    let a0 = v128_load(p0 as *const v128);
                    let a1 = v128_load(p0.add(4) as *const v128);
                    let b0 = v128_load(p1 as *const v128);
                    let b1 = v128_load(p1.add(4) as *const v128);
                    let m0 =
                        max_step4(u32x4_shuffle::<0, 2, 4, 6>(a0, a1), u32x4_shuffle::<1, 3, 5, 7>(a0, a1));
                    let m1 =
                        max_step4(u32x4_shuffle::<0, 2, 4, 6>(b0, b1), u32x4_shuffle::<1, 3, 5, 7>(b0, b1));
                    v128_store(drow.as_mut_ptr().add(ox) as *mut v128, max_step4(m0, m1));
                    ox += 4;
                }
            }
            for x in ox..ow {
                let mut m = r0[2 * x];
                step_max(&mut m, r0[2 * x + 1]);
                step_max(&mut m, r1[2 * x]);
                step_max(&mut m, r1[2 * x + 1]);
                drow[x] = m;
            }
        }
    }

    pub(super) fn max_pool_sw1_plane(
        src: &[f32],
        h: usize,
        w: usize,
        spec: &Pool2dSpec,
        oh: usize,
        ow: usize,
        dst: &mut [f32],
    ) {
        let ([kh, kw], [sh, _], [ph, pw]) = (spec.kernel, spec.stride, spec.padding);
        let (hi, wi) = (h as isize, w as isize);
        let x_interior_lo = pw.min(ow);
        let x_interior_hi = w.saturating_add(pw).checked_sub(kw).map_or(0, |v| (v + 1).min(ow));
        for oy in 0..oh {
            let y0 = (oy * sh) as isize - ph as isize;
            let iy0 = y0.clamp(0, hi) as usize;
            let iy1 = (y0 + kh as isize).clamp(0, hi) as usize;
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            let scalar_at = |ox: usize| -> f32 {
                let x0 = ox as isize - pw as isize;
                let ix0 = x0.clamp(0, wi) as usize;
                let ix1 = (x0 + kw as isize).clamp(0, wi) as usize;
                let mut m = f32::NEG_INFINITY;
                let mut init = false;
                for iy in iy0..iy1 {
                    let srow = &src[iy * w..(iy + 1) * w];
                    for &v in &srow[ix0..ix1] {
                        if !init {
                            m = v;
                            init = true;
                        } else {
                            step_max(&mut m, v);
                        }
                    }
                }
                m
            };
            for (ox, slot) in drow.iter_mut().enumerate().take(x_interior_lo) {
                *slot = scalar_at(ox);
            }
            let mut ox = x_interior_lo;
            if iy0 < iy1 {
                // SAFETY: for `ox in x_interior_lo..x_interior_hi` with `ox + 4 <= x_interior_hi`,
                // `ox >= pw` and `(ox - pw) + (kw - 1) + 4 <= w`, so every `v128_load` is inside `srow[0..w]`.
                unsafe {
                    while ox + 4 <= x_interior_hi {
                        let ix_base = ox - pw;
                        let mut m = v128_load(src.as_ptr().add(iy0 * w + ix_base) as *const v128);
                        for t in 1..kw {
                            m = max_step4(
                                m,
                                v128_load(src.as_ptr().add(iy0 * w + ix_base + t) as *const v128),
                            );
                        }
                        for iy in (iy0 + 1)..iy1 {
                            let row_ptr = src.as_ptr().add(iy * w + ix_base);
                            for t in 0..kw {
                                m = max_step4(m, v128_load(row_ptr.add(t) as *const v128));
                            }
                        }
                        v128_store(drow.as_mut_ptr().add(ox) as *mut v128, m);
                        ox += 4;
                    }
                }
            }
            for (i, slot) in drow[ox..].iter_mut().enumerate() {
                *slot = scalar_at(ox + i);
            }
        }
    }

    pub(super) fn avg_pool2x2_s2_plane(src: &[f32], w: usize, oh: usize, ow: usize, dst: &mut [f32]) {
        let q = f32x4_splat(0.25);
        for oy in 0..oh {
            let r0 = &src[(2 * oy) * w..(2 * oy + 1) * w];
            let r1 = &src[(2 * oy + 1) * w..(2 * oy + 2) * w];
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            let mut ox = 0;
            // SAFETY: `ox + 4 <= ow` implies `2 * ox + 8 <= 2 * ow <= w`.
            unsafe {
                while ox + 4 <= ow {
                    let p0 = r0.as_ptr().add(2 * ox);
                    let p1 = r1.as_ptr().add(2 * ox);
                    let a0 = v128_load(p0 as *const v128);
                    let a1 = v128_load(p0.add(4) as *const v128);
                    let b0 = v128_load(p1 as *const v128);
                    let b1 = v128_load(p1.add(4) as *const v128);
                    let e0 = u32x4_shuffle::<0, 2, 4, 6>(a0, a1);
                    let o0 = u32x4_shuffle::<1, 3, 5, 7>(a0, a1);
                    let e1 = u32x4_shuffle::<0, 2, 4, 6>(b0, b1);
                    let o1 = u32x4_shuffle::<1, 3, 5, 7>(b0, b1);
                    // Accumulate in exact row-major scalar order `(((e0 + o0) + e1) + o1)` for bit-exact parity.
                    let sum = f32x4_add(f32x4_add(f32x4_add(e0, o0), e1), o1);
                    v128_store(drow.as_mut_ptr().add(ox) as *mut v128, f32x4_mul(sum, q));
                    ox += 4;
                }
            }
            for x in ox..ow {
                drow[x] = (((r0[2 * x] + r0[2 * x + 1]) + r1[2 * x]) + r1[2 * x + 1]) * 0.25;
            }
        }
    }

    pub(super) fn avg_pool_sw1_plane(
        src: &[f32],
        h: usize,
        w: usize,
        spec: &Pool2dSpec,
        oh: usize,
        ow: usize,
        dst: &mut [f32],
    ) {
        let ([kh, kw], [sh, _], [ph, pw]) = (spec.kernel, spec.stride, spec.padding);
        let full_denom = (kh as f32) * (kw as f32);
        let (hi, wi) = (h as isize, w as isize);
        let x_interior_lo = pw.min(ow);
        let x_interior_hi = w.saturating_add(pw).checked_sub(kw).map_or(0, |v| (v + 1).min(ow));
        for oy in 0..oh {
            let y0 = (oy * sh) as isize - ph as isize;
            let iy0 = y0.clamp(0, hi) as usize;
            let iy1 = (y0 + kh as isize).clamp(0, hi) as usize;
            let valid_h = (iy1 - iy0) as f32;
            let drow = &mut dst[oy * ow..(oy + 1) * ow];
            let scalar_at = |ox: usize| -> f32 {
                let x0 = ox as isize - pw as isize;
                let ix0 = x0.clamp(0, wi) as usize;
                let ix1 = (x0 + kw as isize).clamp(0, wi) as usize;
                let mut sum = 0.0f32;
                for iy in iy0..iy1 {
                    let srow = &src[iy * w..(iy + 1) * w];
                    for &v in &srow[ix0..ix1] {
                        sum += v;
                    }
                }
                let denom = if spec.count_include_pad { full_denom } else { valid_h * ((ix1 - ix0) as f32) };
                if denom > 0.0 { sum / denom } else { 0.0 }
            };
            for (ox, slot) in drow.iter_mut().enumerate().take(x_interior_lo) {
                *slot = scalar_at(ox);
            }
            let mut ox = x_interior_lo;
            if iy0 < iy1 {
                let denom = if spec.count_include_pad { full_denom } else { valid_h * (kw as f32) };
                let vdenom = f32x4_splat(denom);
                // SAFETY: `ox >= pw` and `(ox - pw) + kw - 1 + 4 <= w`.
                unsafe {
                    while ox + 4 <= x_interior_hi {
                        let ix_base = ox - pw;
                        let mut sum = f32x4_splat(0.0);
                        for iy in iy0..iy1 {
                            let row_ptr = src.as_ptr().add(iy * w + ix_base);
                            for t in 0..kw {
                                sum = f32x4_add(sum, v128_load(row_ptr.add(t) as *const v128));
                            }
                        }
                        v128_store(drow.as_mut_ptr().add(ox) as *mut v128, f32x4_div(sum, vdenom));
                        ox += 4;
                    }
                }
            }
            for (i, slot) in drow[ox..].iter_mut().enumerate() {
                *slot = scalar_at(ox + i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(len: usize) -> Vec<f32> {
        (0..len).map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.25).collect()
    }

    #[test]
    fn max_and_avg_pool_match_reference_across_shapes_and_edges() {
        let specs = [
            Pool2dSpec::new([2, 2], [2, 2], [0, 0]),
            Pool2dSpec::new([3, 3], [1, 1], [1, 1]),
            Pool2dSpec::new([5, 5], [1, 1], [2, 2]),
            Pool2dSpec { kernel: [2, 3], stride: [1, 2], padding: [1, 1], count_include_pad: true },
            Pool2dSpec { kernel: [3, 2], stride: [2, 1], padding: [1, 0], count_include_pad: false },
        ];
        for &[n, c, h, w] in &[[1usize, 1, 1, 1], [1, 2, 5, 7], [2, 3, 10, 13]] {
            let src = pattern(n * c * h * w);
            for spec in &specs {
                let Some([oh, ow]) = spec.output_hw(h, w) else { continue };
                let mut got_max = vec![0.0f32; n * c * oh * ow];
                let mut ref_max = vec![0.0f32; n * c * oh * ow];
                max_pool2d_nchw(&src, [n, c, h, w], spec, &mut got_max);
                for p in 0..n * c {
                    max_pool2d_plane_scalar(
                        &src[p * h * w..(p + 1) * h * w],
                        h,
                        w,
                        spec,
                        oh,
                        ow,
                        &mut ref_max[p * oh * ow..(p + 1) * oh * ow],
                    );
                }
                assert_eq!(got_max, ref_max, "max_pool mismatch shape={:?} spec={:?}", [n, c, h, w], spec);

                let mut got_avg = vec![0.0f32; n * c * oh * ow];
                let mut ref_avg = vec![0.0f32; n * c * oh * ow];
                avg_pool2d_nchw(&src, [n, c, h, w], spec, &mut got_avg);
                for p in 0..n * c {
                    avg_pool2d_plane_scalar(
                        &src[p * h * w..(p + 1) * h * w],
                        h,
                        w,
                        spec,
                        oh,
                        ow,
                        &mut ref_avg[p * oh * ow..(p + 1) * oh * ow],
                    );
                }
                for (g, r) in got_avg.iter().zip(&ref_avg) {
                    assert!(
                        (g - r).abs() <= 1e-5,
                        "avg_pool mismatch shape={:?} spec={:?}: {g} vs {r}",
                        [n, c, h, w],
                        spec
                    );
                }
            }
        }

        // Verify NaN propagation and negative values (padding must never beat negatives in max_pool).
        let neg = [-5.0f32, -2.0, f32::NAN, -9.0];
        let mut out = [0.0f32; 1];
        max_pool2d_nchw(&neg, [1, 1, 2, 2], &Pool2dSpec::new([2, 2], [2, 2], [0, 0]), &mut out);
        assert!(out[0].is_nan());

        let all_neg = [-5.0f32, -3.0, -7.0, -4.0];
        let mut out_pad = [0.0f32; 4];
        max_pool2d_nchw(&all_neg, [1, 1, 2, 2], &Pool2dSpec::new([3, 3], [1, 1], [1, 1]), &mut out_pad);
        assert_eq!(out_pad, [-3.0, -3.0, -3.0, -3.0]);

        // count_include_pad differentiates border averages on a constant plane of 4.0.
        let ones = [4.0f32; 4];
        let mut excl = [0.0f32; 4];
        let mut incl = [0.0f32; 4];
        let mut s_pad = Pool2dSpec::new([3, 3], [1, 1], [1, 1]);
        avg_pool2d_nchw(&ones, [1, 1, 2, 2], &s_pad, &mut excl);
        s_pad.count_include_pad = true;
        avg_pool2d_nchw(&ones, [1, 1, 2, 2], &s_pad, &mut incl);
        assert!(excl.iter().all(|&v| (v - 4.0).abs() < 1e-6));
        assert!(incl.iter().all(|&v| (v - 16.0 / 9.0).abs() < 1e-5));

        // Fully padded windows (`padding > kernel` on 1x1) must never trap or underflow.
        let one_px = [3.0f32];
        let mut full_pad_spec = Pool2dSpec::new([1, 1], [1, 1], [2, 2]);
        let [foh, fow] = full_pad_spec.output_hw(1, 1).unwrap();
        assert_eq!([foh, fow], [5, 5]);
        let mut max_fp = vec![0.0f32; foh * fow];
        max_pool2d_nchw(&one_px, [1, 1, 1, 1], &full_pad_spec, &mut max_fp);
        assert_eq!(max_fp[12], 3.0);
        assert!(max_fp.iter().enumerate().all(|(i, &v)| i == 12 || v == f32::NEG_INFINITY));
        let mut avg_fp = vec![-1.0f32; foh * fow];
        avg_pool2d_nchw(&one_px, [1, 1, 1, 1], &full_pad_spec, &mut avg_fp);
        assert_eq!(avg_fp[12], 3.0);
        assert!(avg_fp.iter().enumerate().all(|(i, &v)| i == 12 || v == 0.0));
        full_pad_spec.count_include_pad = true;
        avg_pool2d_nchw(&one_px, [1, 1, 1, 1], &full_pad_spec, &mut avg_fp);
        assert_eq!(avg_fp[12], 3.0);
        assert!(avg_fp.iter().enumerate().all(|(i, &v)| i == 12 || v == 0.0));

        // 2x2 stride-2 avg_pool preserves exact left-to-right row-major addition order (even under cancellation).
        let cancel = [
            1e20f32, 1.0, 1e20, 1.0, 1e20, 1.0, 1e20, 1.0, // row 0 (4 SIMD lanes)
            -1e20, 1.0, -1e20, 1.0, -1e20, 1.0, -1e20, 1.0, // row 1
        ];
        let mut cancel_out = [0.0f32; 4];
        avg_pool2d_nchw(&cancel, [1, 1, 2, 8], &Pool2dSpec::new([2, 2], [2, 2], [0, 0]), &mut cancel_out);
        assert_eq!(cancel_out, [0.25, 0.25, 0.25, 0.25]);

        // Huge padded kernel (`65536 × 65536`) must not overflow 32-bit `kh * kw` denominator on wasm32.
        let huge_spec = Pool2dSpec {
            kernel: [65_536, 65_536],
            stride: [2, 2],
            padding: [32_768, 32_768],
            count_include_pad: true,
        };
        let mut huge_out = [0.0f32; 1];
        avg_pool2d_nchw(&[3.0], [1, 1, 1, 1], &huge_spec, &mut huge_out);
        assert!((huge_out[0] - (3.0 / 4_294_967_296.0)).abs() < 1e-15, "got {}", huge_out[0]);
    }
}
