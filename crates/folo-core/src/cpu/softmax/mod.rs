//! Cache-tiled AVX2 online `softmax` / `log_softmax` and fused `softmax_backward`.
//!
//! - `inner == 1`, `d <= ROW_TILE`: stays in L1; `log_softmax` accumulates `sum(fast_exp(v - m))` directly in
//!   registers without writing `dst` twice; `softmax` fuses `exp` + `sum` into one pass before scaling. Both
//!   use 8-wide AVX2 `fast_exp8` (bit-exact with [`super::elementwise::fast_exp`]) and the exact 16-lane sum
//!   reduction order of [`super::reduce::sum_slice`].
//! - `inner == 1`, `d > ROW_TILE`: cache-tiled online `(m, s)` pass over 2048-float L1 tiles eliminates the
//!   extra full-row DRAM pass, followed by one AVX2 write pass.
//! - `inner > 1`: column-tiled in stack-allocated `COL_TILE`-wide strips (zero heap allocations), keeping each
//!   column strip hot in L1, precomputing `ln(s)` / `1 / s` once per column instead of `d` times, and fanning
//!   out across `(outer, col_tile)` when large (even when `outer == 1`).
//! - `softmax_backward`: fused single-allocation kernel eliminating 3 intermediate tensors and 2 full-tensor
//!   memory passes compared to the composite autograd formula, using `elementwise::exp8` for bit-exact parity.

mod backward;
#[cfg(test)]
mod tests;

pub(crate) use backward::softmax_backward;

use super::elementwise::{SendPtr, fast_exp};
use super::{PAR_MIN, alloc_uninit, par_chunks, par_for};

/// Rows up to this length fit in L1 (16 KiB) and take the exact two-pass kernel.
const ROW_TILE: usize = 4096;
/// Tile size (in floats, 8 KiB) for the online pass on rows longer than [`ROW_TILE`].
const ONLINE_TILE: usize = 2048;
/// Column strip width for `inner > 1` (1 KiB per stack array `m` / `s`, zero heap allocs).
pub(super) const COL_TILE: usize = 256;

#[cfg(target_arch = "x86_64")]
const LOG2E: f32 = std::f32::consts::LOG2_E;
#[cfg(target_arch = "x86_64")]
const FAST_EXP_MAGIC: f32 = 12_582_912.0;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// 8-lane AVX2 twin of [`fast_exp`] (non-FMA, bitwise identical for every `f32` including NaN payloads).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn fast_exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-104.0)), _mm256_set1_ps(89.0));
    let magic = _mm256_set1_ps(FAST_EXP_MAGIC);
    let n = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(xc, _mm256_set1_ps(LOG2E)), magic), magic);
    let r = _mm256_sub_ps(
        _mm256_sub_ps(xc, _mm256_mul_ps(n, _mm256_set1_ps(0.693_359_4))),
        _mm256_mul_ps(n, _mm256_set1_ps(-2.121_944_4e-4)),
    );
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(8.333_452e-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(0.5));
    let y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, r), r), r), _mm256_set1_ps(1.0));
    let ni = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32::<1>(ni);
    let n2 = _mm256_sub_epi32(ni, n1);
    let bias = _mm256_set1_epi32(127);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n1, bias)));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n2, bias)));
    let v = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);
    _mm256_blendv_ps(v, x, _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}

/// Collapse two 8-lane accumulators in exact `0..16` lane order (matches `reduce::sum_slice`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn hsum16(a0: std::arch::x86_64::__m256, a1: std::arch::x86_64::__m256, rest_vals: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = [0f32; 16];
    // SAFETY: `acc` holds 16 floats.
    unsafe {
        _mm256_storeu_ps(acc.as_mut_ptr(), a0);
        _mm256_storeu_ps(acc.as_mut_ptr().add(8), a1);
    }
    for (i, &v) in rest_vals.iter().enumerate() {
        acc[i] += v;
    }
    let mut s = 0f32;
    for a in acc {
        s += a;
    }
    s
}

/// `(row_max_ignoring_nan, any_nan)` over `src`.
fn row_max_and_nan(src: &[f32]) -> (f32, bool) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { row_max_and_nan_avx2(src) };
    }
    let mut m = f32::NEG_INFINITY;
    let mut any_nan = false;
    for &v in src {
        if v.is_nan() {
            any_nan = true;
        } else {
            m = m.max(v);
        }
    }
    (m, any_nan)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn row_max_and_nan_avx2(src: &[f32]) -> (f32, bool) {
    use std::arch::x86_64::*;
    let mut m0 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut m1 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut nan0 = _mm256_setzero_ps();
    let mut nan1 = _mm256_setzero_ps();
    let (chunks, rest) = src.as_chunks::<16>();
    for c in chunks {
        // SAFETY: `c` is 16 floats.
        unsafe {
            let v0 = _mm256_loadu_ps(c.as_ptr());
            let v1 = _mm256_loadu_ps(c.as_ptr().add(8));
            // `vmaxps(v, acc)` returns `acc` when `v` is NaN, matching `acc.max(v)`.
            m0 = _mm256_max_ps(v0, m0);
            m1 = _mm256_max_ps(v1, m1);
            nan0 = _mm256_or_ps(nan0, _mm256_cmp_ps::<_CMP_UNORD_Q>(v0, v0));
            nan1 = _mm256_or_ps(nan1, _mm256_cmp_ps::<_CMP_UNORD_Q>(v1, v1));
        }
    }
    let (c8, tail) = rest.as_chunks::<8>();
    for c in c8 {
        // SAFETY: `c` is 8 floats.
        unsafe {
            let v = _mm256_loadu_ps(c.as_ptr());
            m0 = _mm256_max_ps(v, m0);
            nan0 = _mm256_or_ps(nan0, _mm256_cmp_ps::<_CMP_UNORD_Q>(v, v));
        }
    }
    let m_vec = _mm256_max_ps(m0, m1);
    let mut lanes = [0f32; 8];
    // SAFETY: `lanes` holds 8 floats.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m_vec) };
    let mut m = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut any_nan = _mm256_movemask_ps(_mm256_or_ps(nan0, nan1)) != 0;
    for &v in tail {
        if v.is_nan() {
            any_nan = true;
        } else {
            m = m.max(v);
        }
    }
    (m, any_nan)
}

/// Sum `fast_exp(v - m)` over `src` without writing an output buffer, matching `sum_slice` exact 16-lane order.
fn sum_exp_shifted(src: &[f32], m: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { sum_exp_shifted_avx2(src, m) };
    }
    let mut acc = [0f32; 16];
    let (chunks, rest) = src.as_chunks::<16>();
    for c in chunks {
        for i in 0..16 {
            acc[i] += fast_exp(c[i] - m);
        }
    }
    for (i, &v) in rest.iter().enumerate() {
        acc[i] += fast_exp(v - m);
    }
    acc.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn sum_exp_shifted_avx2(src: &[f32], m: f32) -> f32 {
    use std::arch::x86_64::*;
    let mv = _mm256_set1_ps(m);
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let (chunks, rest) = src.as_chunks::<16>();
    for c in chunks {
        // SAFETY: `c` is 16 floats.
        unsafe {
            let e0 = fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), mv));
            let e1 = fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(c.as_ptr().add(8)), mv));
            a0 = _mm256_add_ps(a0, e0);
            a1 = _mm256_add_ps(a1, e1);
        }
    }
    let mut rbuf = [0f32; 16];
    for (i, &v) in rest.iter().enumerate() {
        rbuf[i] = fast_exp(v - m);
    }
    hsum16(a0, a1, &rbuf[..rest.len()])
}

/// Write `dst[i] = fast_exp(src[i] - m)` and return `sum_slice(dst)` in a single pass.
fn exp_shifted_and_sum(src: &[f32], dst: &mut [f32], m: f32) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { exp_shifted_and_sum_avx2(src, dst, m) };
    }
    let mut acc = [0f32; 16];
    let (sc, sr) = src.as_chunks::<16>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        for i in 0..16 {
            let e = fast_exp(s[i] - m);
            d[i] = e;
            acc[i] += e;
        }
    }
    for (i, (d, &s)) in dr.iter_mut().zip(sr).enumerate() {
        let e = fast_exp(s - m);
        *d = e;
        acc[i] += e;
    }
    acc.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn exp_shifted_and_sum_avx2(src: &[f32], dst: &mut [f32], m: f32) -> f32 {
    use std::arch::x86_64::*;
    let mv = _mm256_set1_ps(m);
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let (sc, sr) = src.as_chunks::<16>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 16 floats.
        unsafe {
            let e0 = fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr()), mv));
            let e1 = fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(8)), mv));
            _mm256_storeu_ps(d.as_mut_ptr(), e0);
            _mm256_storeu_ps(d.as_mut_ptr().add(8), e1);
            a0 = _mm256_add_ps(a0, e0);
            a1 = _mm256_add_ps(a1, e1);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d = fast_exp(s - m);
    }
    hsum16(a0, a1, dr)
}

/// Write `dst[i] = (src[i] - m) - ls`.
fn write_log_softmax_row(src: &[f32], dst: &mut [f32], m: f32, ls: f32) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { write_log_softmax_row_avx2(src, dst, m, ls) };
    }
    for (y, &v) in dst.iter_mut().zip(src) {
        *y = (v - m) - ls;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn write_log_softmax_row_avx2(src: &[f32], dst: &mut [f32], m: f32, ls: f32) {
    use std::arch::x86_64::*;
    let mv = _mm256_set1_ps(m);
    let lsv = _mm256_set1_ps(ls);
    let (sc, sr) = src.as_chunks::<16>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 16 floats.
        unsafe {
            let y0 = _mm256_sub_ps(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr()), mv), lsv);
            let y1 = _mm256_sub_ps(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(8)), mv), lsv);
            _mm256_storeu_ps(d.as_mut_ptr(), y0);
            _mm256_storeu_ps(d.as_mut_ptr().add(8), y1);
        }
    }
    for (y, &v) in dr.iter_mut().zip(sr) {
        *y = (v - m) - ls;
    }
}

/// Scale `dst[i] *= inv` in place.
fn scale_inplace(dst: &mut [f32], inv: f32) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { scale_inplace_avx2(dst, inv) };
    }
    for y in dst.iter_mut() {
        *y *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn scale_inplace_avx2(dst: &mut [f32], inv: f32) {
    use std::arch::x86_64::*;
    let iv = _mm256_set1_ps(inv);
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for d in dc {
        // SAFETY: `d` is 16 floats.
        unsafe {
            let p = d.as_mut_ptr();
            _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), iv));
            _mm256_storeu_ps(p.add(8), _mm256_mul_ps(_mm256_loadu_ps(p.add(8)), iv));
        }
    }
    for y in dr {
        *y *= inv;
    }
}

/// Write `dst[i] = fast_exp(src[i] - m) * inv` in a single pass (used for long online rows).
fn write_exp_scaled_row(src: &[f32], dst: &mut [f32], m: f32, inv: f32) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected above.
        return unsafe { write_exp_scaled_row_avx2(src, dst, m, inv) };
    }
    for (y, &v) in dst.iter_mut().zip(src) {
        *y = fast_exp(v - m) * inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn write_exp_scaled_row_avx2(src: &[f32], dst: &mut [f32], m: f32, inv: f32) {
    use std::arch::x86_64::*;
    let mv = _mm256_set1_ps(m);
    let iv = _mm256_set1_ps(inv);
    let (sc, sr) = src.as_chunks::<16>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 16 floats.
        unsafe {
            let y0 = _mm256_mul_ps(fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr()), mv)), iv);
            let y1 = _mm256_mul_ps(fast_exp8(_mm256_sub_ps(_mm256_loadu_ps(s.as_ptr().add(8)), mv)), iv);
            _mm256_storeu_ps(d.as_mut_ptr(), y0);
            _mm256_storeu_ps(d.as_mut_ptr().add(8), y1);
        }
    }
    for (y, &v) in dr.iter_mut().zip(sr) {
        *y = fast_exp(v - m) * inv;
    }
}

/// One contiguous row (`inner == 1`) of `softmax` / `log_softmax`.
fn softmax_row(src: &[f32], dst: &mut [f32], log: bool) {
    if src.len() <= ROW_TILE {
        let (m_raw, _any_nan) = row_max_and_nan(src);
        let m = if m_raw.is_finite() { m_raw } else { 0.0 };
        if log {
            let s = sum_exp_shifted(src, m);
            write_log_softmax_row(src, dst, m, s.ln());
        } else {
            let s = exp_shifted_and_sum(src, dst, m);
            scale_inplace(dst, 1.0 / s);
        }
        return;
    }
    // Online cache-tiled pass over `ONLINE_TILE` chunks: computes `(m, s)` in one streaming read of `src`.
    let mut m_cur = f32::NEG_INFINITY;
    let mut s_cur = 0.0f64;
    let mut has_finite = false;
    let mut saw_pos_inf = false;
    for tile in src.chunks(ONLINE_TILE) {
        let (mt, any_nan) = row_max_and_nan(tile);
        if any_nan {
            dst.fill(f32::NAN);
            return;
        }
        if mt == f32::INFINITY {
            saw_pos_inf = true;
            continue;
        }
        if !mt.is_finite() {
            continue;
        }
        let st = sum_exp_shifted(tile, mt) as f64;
        if !has_finite {
            m_cur = mt;
            s_cur = st;
            has_finite = true;
        } else if mt > m_cur {
            s_cur = s_cur * ((m_cur as f64) - (mt as f64)).exp() + st;
            m_cur = mt;
        } else {
            s_cur += st * ((mt as f64) - (m_cur as f64)).exp();
        }
    }
    if saw_pos_inf || !has_finite {
        let s = sum_exp_shifted(src, 0.0);
        if log {
            write_log_softmax_row(src, dst, 0.0, s.ln());
        } else {
            write_exp_scaled_row(src, dst, 0.0, 1.0 / s);
        }
        return;
    }
    let s = s_cur as f32;
    if log {
        write_log_softmax_row(src, dst, m_cur, s.ln());
    } else {
        write_exp_scaled_row(src, dst, m_cur, 1.0 / s);
    }
}

/// One column strip `c0..c0 + tw` (`tw <= COL_TILE`) of an `[d, inner]` plane, kept hot in L1.
///
/// # Safety
/// `dst_base` points to `d * inner` floats and no other task writes columns `c0..c0 + tw` of this plane.
unsafe fn softmax_col_tile(
    src: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 was detected; caller guarantees disjoint column strip.
        return unsafe { softmax_col_tile_avx2(src, dst_base, d, inner, c0, tw, log) };
    }
    let mut m = [f32::NEG_INFINITY; COL_TILE];
    for r in 0..d {
        let row = &src[r * inner + c0..r * inner + c0 + tw];
        for (a, &v) in m[..tw].iter_mut().zip(row) {
            *a = a.max(v);
        }
    }
    for a in &mut m[..tw] {
        if !a.is_finite() {
            *a = 0.0;
        }
    }
    let mut s = [0f32; COL_TILE];
    if log {
        for r in 0..d {
            let row = &src[r * inner + c0..r * inner + c0 + tw];
            for ((acc, &v), &mx) in s[..tw].iter_mut().zip(row).zip(&m[..tw]) {
                *acc += fast_exp(v - mx);
            }
        }
        let mut ls = [0f32; COL_TILE];
        for j in 0..tw {
            ls[j] = s[j].ln();
        }
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: disjoint column strip `c0..c0 + tw` of row `r`.
            let drow = unsafe { std::slice::from_raw_parts_mut(dst_base.add(r * inner + c0), tw) };
            for (((y, &v), &mx), &lsv) in drow.iter_mut().zip(srow).zip(&m[..tw]).zip(&ls[..tw]) {
                *y = (v - mx) - lsv;
            }
        }
    } else {
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: disjoint column strip `c0..c0 + tw` of row `r`.
            let drow = unsafe { std::slice::from_raw_parts_mut(dst_base.add(r * inner + c0), tw) };
            for (((y, &v), &mx), acc) in drow.iter_mut().zip(srow).zip(&m[..tw]).zip(&mut s[..tw]) {
                let e = fast_exp(v - mx);
                *y = e;
                *acc += e;
            }
        }
        let mut inv = [0f32; COL_TILE];
        for j in 0..tw {
            inv[j] = 1.0 / s[j];
        }
        for r in 0..d {
            // SAFETY: disjoint column strip `c0..c0 + tw` of row `r`.
            let drow = unsafe { std::slice::from_raw_parts_mut(dst_base.add(r * inner + c0), tw) };
            for (y, &iv) in drow.iter_mut().zip(&inv[..tw]) {
                *y *= iv;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_col_tile_avx2(
    src: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    use std::arch::x86_64::*;
    let mut m = [f32::NEG_INFINITY; COL_TILE];
    let v_end = tw / 8 * 8;
    for r in 0..d {
        let srow = &src[r * inner + c0..r * inner + c0 + tw];
        let mut j = 0;
        while j < v_end {
            // SAFETY: `j + 8 <= tw <= COL_TILE`.
            unsafe {
                let v = _mm256_loadu_ps(srow.as_ptr().add(j));
                let cur = _mm256_loadu_ps(m.as_ptr().add(j));
                _mm256_storeu_ps(m.as_mut_ptr().add(j), _mm256_max_ps(v, cur));
            }
            j += 8;
        }
        for jj in v_end..tw {
            m[jj] = m[jj].max(srow[jj]);
        }
    }
    for a in &mut m[..tw] {
        if !a.is_finite() {
            *a = 0.0;
        }
    }
    let mut s = [0f32; COL_TILE];
    if log {
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            while j < v_end {
                // SAFETY: `j + 8 <= tw <= COL_TILE`.
                unsafe {
                    let e = fast_exp8(_mm256_sub_ps(
                        _mm256_loadu_ps(srow.as_ptr().add(j)),
                        _mm256_loadu_ps(m.as_ptr().add(j)),
                    ));
                    let acc = _mm256_add_ps(_mm256_loadu_ps(s.as_ptr().add(j)), e);
                    _mm256_storeu_ps(s.as_mut_ptr().add(j), acc);
                }
                j += 8;
            }
            for jj in v_end..tw {
                s[jj] += fast_exp(srow[jj] - m[jj]);
            }
        }
        let mut ls = [0f32; COL_TILE];
        for j in 0..tw {
            ls[j] = s[j].ln();
        }
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw` in row `r`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            while j < v_end {
                // SAFETY: `j + 8 <= tw <= COL_TILE`.
                unsafe {
                    let y = _mm256_sub_ps(
                        _mm256_sub_ps(
                            _mm256_loadu_ps(srow.as_ptr().add(j)),
                            _mm256_loadu_ps(m.as_ptr().add(j)),
                        ),
                        _mm256_loadu_ps(ls.as_ptr().add(j)),
                    );
                    _mm256_storeu_ps(dptr.add(j), y);
                }
                j += 8;
            }
            for jj in v_end..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = (srow[jj] - m[jj]) - ls[jj] };
            }
        }
    } else {
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw` in row `r`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            while j < v_end {
                // SAFETY: `j + 8 <= tw <= COL_TILE`.
                unsafe {
                    let e = fast_exp8(_mm256_sub_ps(
                        _mm256_loadu_ps(srow.as_ptr().add(j)),
                        _mm256_loadu_ps(m.as_ptr().add(j)),
                    ));
                    _mm256_storeu_ps(dptr.add(j), e);
                    let acc = _mm256_add_ps(_mm256_loadu_ps(s.as_ptr().add(j)), e);
                    _mm256_storeu_ps(s.as_mut_ptr().add(j), acc);
                }
                j += 8;
            }
            for jj in v_end..tw {
                let e = fast_exp(srow[jj] - m[jj]);
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = e };
                s[jj] += e;
            }
        }
        let mut inv = [0f32; COL_TILE];
        for j in 0..tw {
            inv[j] = 1.0 / s[j];
        }
        for r in 0..d {
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw` in row `r`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            while j < v_end {
                // SAFETY: `j + 8 <= tw <= COL_TILE`.
                unsafe {
                    let y = _mm256_mul_ps(_mm256_loadu_ps(dptr.add(j)), _mm256_loadu_ps(inv.as_ptr().add(j)));
                    _mm256_storeu_ps(dptr.add(j), y);
                }
                j += 8;
            }
            for (jj, &iv) in inv.iter().enumerate().take(tw).skip(v_end) {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) *= iv };
            }
        }
    }
}

/// Softmax (`!log`) or log-softmax (`log`) along the middle dimension of `[outer, d, inner]`.
pub(crate) fn softmax(x: &[f32], (outer, d, inner): (usize, usize, usize), log: bool) -> Vec<f32> {
    let mut out = alloc_uninit(x.len());
    if out.is_empty() {
        return out;
    }
    if inner == 1 {
        let row = |o: usize, dst: &mut [f32]| softmax_row(&x[o * d..(o + 1) * d], dst, log);
        if x.len() >= PAR_MIN && outer > 1 {
            par_chunks(&mut out, d, &row);
        } else {
            out.chunks_exact_mut(d).enumerate().for_each(|(o, dst)| row(o, dst));
        }
        return out;
    }
    let tiles_per = inner.div_ceil(COL_TILE);
    let total_tasks = outer * tiles_per;
    let optr = SendPtr(out.as_mut_ptr());
    let run = |t: usize| {
        let optr = &optr;
        let (o, ct) = (t / tiles_per, t % tiles_per);
        let c0 = ct * COL_TILE;
        let tw = COL_TILE.min(inner - c0);
        let src = &x[o * d * inner..(o + 1) * d * inner];
        // SAFETY: `(o, ct)` owns disjoint columns `c0..c0 + tw` of outer plane `o`.
        unsafe { softmax_col_tile(src, optr.0.add(o * d * inner), d, inner, c0, tw, log) };
    };
    if x.len() >= PAR_MIN && total_tasks > 1 {
        par_for(total_tasks, &run);
    } else {
        (0..total_tasks).for_each(run);
    }
    out
}
