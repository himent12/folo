//! The forward's packed B operands read from the input planes: the implicit im2col of a strided or accumulating
//! conv ([`Im2Col`], never materialised: each block is unfolded as it is packed) and a pointwise conv's planes.

use super::ConvGeom;
use crate::cpu::gemm::{Operand, pack_rows_into_panels};

/// Copy one output row segment of the unfolded input: input row `iy` of `plane` (one `h x w` input plane),
/// output columns `ox0..ox0 + len`, kernel column offset `dx` (already includes `-pad`). Writes
/// `dst[t * stride]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn im2col_segment(
    plane: &[f32],
    g: &ConvGeom,
    iy: isize,
    dx: isize,
    ox0: usize,
    len: usize,
    dst: &mut [f32],
    stride: usize,
) {
    let sw = g.p.stride[1] as isize;
    if iy < 0 || iy >= g.h as isize {
        for t in 0..len {
            dst[t * stride] = 0.0;
        }
        return;
    }
    let row = &plane[iy as usize * g.w..(iy as usize + 1) * g.w];
    let wi = g.w as isize;
    // Valid output columns: 0 <= ox * sw + dx < w.
    let ix0 = ox0 as isize * sw + dx;
    let lo = if ix0 >= 0 { 0 } else { ((-ix0) + sw - 1) / sw } as usize;
    let hi = if ix0 >= wi { 0 } else { (((wi - ix0) + sw - 1) / sw) as usize };
    let (lo, hi) = (lo.min(len), hi.min(len).max(lo.min(len)));
    for t in 0..lo {
        dst[t * stride] = 0.0;
    }
    if stride == 1 && sw == 1 && hi > lo {
        let s = (ix0 + lo as isize) as usize;
        dst[lo..hi].copy_from_slice(&row[s..s + (hi - lo)]);
    } else if stride == 1 && sw == 2 && hi > lo {
        let s = (ix0 + lo as isize * 2) as usize;
        copy_stride2(&row[s..], &mut dst[lo..hi]);
    } else {
        for t in lo..hi {
            dst[t * stride] = row[(ix0 + t as isize * sw) as usize];
        }
    }
    for t in hi..len {
        dst[t * stride] = 0.0;
    }
}

/// Copy `dst[t] = src[2 * t]` for `t < dst.len()` using AVX2 `vshufps` + `vpermq` or Wasm SIMD128
/// `i32x4_shuffle` when available.
#[inline]
fn copy_stride2(src: &[f32], dst: &mut [f32]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    debug_assert!(src.len() >= 2 * n - 1);
    #[cfg(target_arch = "x86_64")]
    if n >= 8 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected above; `n >= 8`.
        unsafe { copy_stride2_avx2(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if n >= 4 {
        copy_stride2_wasm(src, dst);
        return;
    }
    for (t, d) in dst.iter_mut().enumerate() {
        *d = src[2 * t];
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn copy_stride2_wasm(src: &[f32], dst: &mut [f32]) {
    use std::arch::wasm32::*;
    let n = dst.len();
    let vec_end = (src.len() / 8).min(n / 4) * 4;
    let mut t = 0;
    // SAFETY: `2 * t + 8 <= src.len()` and `t + 4 <= n`.
    unsafe {
        while t < vec_end {
            let p = src.as_ptr().add(2 * t);
            let lo = v128_load(p as *const v128);
            let hi = v128_load(p.add(4) as *const v128);
            v128_store(dst.as_mut_ptr().add(t) as *mut v128, i32x4_shuffle::<0, 2, 4, 6>(lo, hi));
            t += 4;
        }
    }
    for tt in t..n {
        dst[tt] = src[2 * tt];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_stride2_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = dst.len();
    // Each vector step reads 16 floats (`src[2*t..2*t + 16]`), so stop while `2 * t + 16 <= src.len()`.
    let vec_end = (src.len() / 16).min(n / 8) * 8;
    let mut t = 0;
    // SAFETY: `2 * t + 16 <= src.len()` and `t + 8 <= n`.
    unsafe {
        while t < vec_end {
            let p = src.as_ptr().add(2 * t);
            let lo = _mm256_loadu_ps(p);
            let hi = _mm256_loadu_ps(p.add(8));
            let even = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0x88>(lo, hi),
            )));
            _mm256_storeu_ps(dst.as_mut_ptr().add(t), even);
            t += 8;
        }
    }
    for tt in t..n {
        dst[tt] = src[2 * tt];
    }
}

/// B operand of the forward GEMM: element `(j, kk)` is the unfolded input `col[kk][j]` of one image and group,
/// whose input channel `ci` is the plane `planes[ci]`.
pub(super) struct Im2Col<'a> {
    pub(super) planes: &'a [&'a [f32]],
    pub(super) g: ConvGeom,
}

impl Operand for Im2Col<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        let g = &self.g;
        let khw = g.kh * g.kw;
        let (sh, ph, pw) = (g.p.stride[0] as isize, g.p.padding[0] as isize, g.p.padding[1] as isize);
        let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
        for kk in 0..kc {
            let kidx = k0 + kk;
            let (ci, r) = (kidx / khw, kidx % khw);
            let (ky, kx) = ((r / g.kw) as isize, (r % g.kw) as isize);
            let plane = self.planes[ci];
            let d = &mut dst[kk * width..kk * width + width];
            d[w..].fill(0.0);
            let (mut oy, mut ox) = (o0 / g.ow, o0 % g.ow);
            let mut t = 0;
            while t < w {
                let len = (g.ow - ox).min(w - t);
                let iy = oy as isize * sh - ph + ky * dh;
                im2col_segment(plane, g, iy, kx * dw - pw, ox, len, &mut d[t..], 1);
                t += len;
                ox = 0;
                oy += 1;
            }
        }
    }
}

/// B operand of a pointwise forward GEMM: element `(j, kk)` is pixel `j` of input plane `planes[kk]`.
pub(super) struct PlaneRows<'a> {
    pub(super) planes: &'a [&'a [f32]],
}

impl Operand for PlaneRows<'_> {
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        for kk in 0..kc {
            let d = &mut dst[kk * width..kk * width + width];
            d[..w].copy_from_slice(&self.planes[k0 + kk][o0..o0 + w]);
            d[w..].fill(0.0);
        }
    }

    fn pack_panels(&self, o0: usize, w: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        pack_rows_into_panels(w, kc, dst, |kk| &self.planes[k0 + kk][o0..o0 + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_stride2_matches_scalar_across_lengths_and_tight_src() {
        for n in 0..=40usize {
            let min_len = if n == 0 { 0 } else { 2 * n - 1 };
            for extra in [0usize, 1, 5] {
                let src: Vec<f32> = (0..min_len + extra).map(|i| i as f32 * 0.25 - 7.5).collect();
                let mut got = vec![-999.0f32; n];
                copy_stride2(&src, &mut got);
                for (t, &v) in got.iter().enumerate() {
                    assert_eq!(v.to_bits(), src[2 * t].to_bits(), "n={n} extra={extra} t={t}");
                }
            }
        }
    }
}
