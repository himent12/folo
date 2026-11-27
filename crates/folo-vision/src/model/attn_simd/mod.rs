//! SIMD-accelerated CPU helpers for attention (`Attention`, `HyperContext`), `FullPadGate`, and fused
//! 3-stage `SPPF` C4 max pooling.
//!
//! Every AVX2+FMA path has a bit-deterministic scalar fallback and preserves IEEE non-finite semantics on
//! softmax inputs. All helpers are pure slice functions with zero heap allocations.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

mod fused;
pub(crate) use fused::fused_attn_o_bnhd;

/// Runtime check for AVX2 + FMA on x86-64 (cached once).
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static DETECT: OnceLock<bool> = OnceLock::new();
        *DETECT.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

const EXP_LO: f32 = -87.336_54;
const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const C2: f32 = 0.5;
const C3: f32 = 1.0 / 6.0;
const C4: f32 = 1.0 / 24.0;
const C5: f32 = 1.0 / 120.0;
const C6: f32 = 1.0 / 720.0;

/// Fast degree-6 `exp(z)` for `z <= 0` (`< 2e-7` relative error, exact `0.0` for `z < EXP_LO`, `1.0` at `0.0`).
#[inline(always)]
fn fast_exp_neg_scalar(z: f32) -> f32 {
    if z < EXP_LO {
        return 0.0;
    }
    let kf = (z * LOG2E).round_ties_even();
    let r = (-kf).mul_add(LN2_HI, z);
    let r = (-kf).mul_add(LN2_LO, r);
    let p = C6.mul_add(r, C5);
    let p = p.mul_add(r, C4);
    let p = p.mul_add(r, C3);
    let p = p.mul_add(r, C2);
    let p = p.mul_add(r, 1.0);
    let p = p.mul_add(r, 1.0);
    let pow2k = f32::from_bits(((kf as i32 + 127) as u32) << 23);
    p * pow2k
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn fast_exp_neg_avx2(z: __m256) -> __m256 {
    let v_lo = _mm256_set1_ps(EXP_LO);
    let v_log2e = _mm256_set1_ps(LOG2E);
    let v_ln2_hi = _mm256_set1_ps(LN2_HI);
    let v_ln2_lo = _mm256_set1_ps(LN2_LO);
    let v_one = _mm256_set1_ps(1.0);
    let v_c2 = _mm256_set1_ps(C2);
    let v_c3 = _mm256_set1_ps(C3);
    let v_c4 = _mm256_set1_ps(C4);
    let v_c5 = _mm256_set1_ps(C5);
    let v_c6 = _mm256_set1_ps(C6);
    let v_127 = _mm256_set1_epi32(127);

    let v = _mm256_max_ps(z, v_lo);
    let kf = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(v, v_log2e));
    let r = _mm256_fnmadd_ps(kf, v_ln2_hi, v);
    let r = _mm256_fnmadd_ps(kf, v_ln2_lo, r);
    let mut p = _mm256_fmadd_ps(v_c6, r, v_c5);
    p = _mm256_fmadd_ps(p, r, v_c4);
    p = _mm256_fmadd_ps(p, r, v_c3);
    p = _mm256_fmadd_ps(p, r, v_c2);
    p = _mm256_fmadd_ps(p, r, v_one);
    p = _mm256_fmadd_ps(p, r, v_one);
    let ki = _mm256_cvtps_epi32(kf);
    let pow2k = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(ki, v_127)));
    let res = _mm256_mul_ps(p, pow2k);
    let underflow = _mm256_cmp_ps::<_CMP_LT_OQ>(z, v_lo);
    _mm256_andnot_ps(underflow, res)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let m1 = _mm_max_ps(lo, hi);
    let m2 = _mm_max_ps(m1, _mm_movehl_ps(m1, m1));
    let m3 = _mm_max_ss(m2, _mm_shuffle_ps::<1>(m2, m2));
    _mm_cvtss_f32(m3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let s1 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
    let s3 = _mm_add_ss(s2, _mm_shuffle_ps::<1>(s2, s2));
    _mm_cvtss_f32(s3)
}

fn softmax_row_non_finite(row: &mut [f32]) {
    let mut sum = 0.0f32;
    let m = row
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) });
    for x in row.iter_mut() {
        let e = (*x - m).exp();
        *x = e;
        sum += e;
    }
    for x in row.iter_mut() {
        *x /= sum;
    }
}

/// In-place scaled softmax over `row`: `row[i] = softmax(row[i] * scale)`.
///
/// Uses 8-wide AVX2+FMA when available and falls back to exact IEEE `exp` if any scaled entry is non-finite.
#[inline]
pub(crate) fn softmax_row_scaled(row: &mut [f32], scale: f32) {
    if row.is_empty() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA verified by `has_avx2_fma()`.
        unsafe { softmax_row_scaled_avx2(row, scale) };
        return;
    }
    softmax_row_scaled_scalar(row, scale);
}

pub(super) fn softmax_row_scaled_scalar(row: &mut [f32], scale: f32) {
    let mut max = f32::NEG_INFINITY;
    let mut non_finite = false;
    for x in row.iter_mut() {
        let v = *x * scale;
        *x = v;
        if !v.is_finite() {
            non_finite = true;
        }
        if v > max {
            max = v;
        }
    }
    if non_finite {
        softmax_row_non_finite(row);
        return;
    }
    let mut sum = 0.0f32;
    for x in row.iter_mut() {
        let e = fast_exp_neg_scalar(*x - max);
        *x = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn softmax_row_scaled_avx2(row: &mut [f32], scale: f32) {
    let len = row.len();
    let ptr = row.as_mut_ptr();
    let v_scale = _mm256_set1_ps(scale);
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut non_fin = _mm256_setzero_si256();
    let mut i = 0;

    // SAFETY: `i + 8 <= len` inside the loop; scalar tail handles `i..len`.
    unsafe {
        while i + 8 <= len {
            let v = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
            _mm256_storeu_ps(ptr.add(i), v);
            vmax = _mm256_max_ps(vmax, v);
            let is_nf = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v), exp_mask), exp_mask);
            non_fin = _mm256_or_si256(non_fin, is_nf);
            i += 8;
        }
        let mut max = hmax_avx2(vmax);
        let mut nf = _mm256_movemask_epi8(non_fin) != 0;
        for x in &mut row[i..] {
            let v = *x * scale;
            *x = v;
            if !v.is_finite() {
                nf = true;
            }
            if v > max {
                max = v;
            }
        }
        if nf {
            softmax_row_non_finite(row);
            return;
        }

        let v_max = _mm256_set1_ps(max);
        let mut vsum = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= len {
            let v = _mm256_loadu_ps(ptr.add(j));
            let e = fast_exp_neg_avx2(_mm256_sub_ps(v, v_max));
            _mm256_storeu_ps(ptr.add(j), e);
            vsum = _mm256_add_ps(vsum, e);
            j += 8;
        }
        let mut sum = hsum_avx2(vsum);
        for x in &mut row[j..] {
            let e = fast_exp_neg_scalar(*x - max);
            *x = e;
            sum += e;
        }

        let inv = 1.0 / sum;
        let v_inv = _mm256_set1_ps(inv);
        let mut k = 0;
        while k + 8 <= len {
            let v = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(k)), v_inv);
            _mm256_storeu_ps(ptr.add(k), v);
            k += 8;
        }
        for x in &mut row[k..] {
            *x *= inv;
        }
    }
}

/// Out-of-place unit-scale row softmax: `dst[i] = softmax(src[i])` for `src.len() == dst.len()`.
#[inline]
pub(crate) fn softmax_row_copy(src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    dst.copy_from_slice(src);
    softmax_row_scaled(dst, 1.0);
}

/// Row-major column softmax over an `[m, n]` matrix: for each column `j in 0..n`,
/// `dst[r * n + j] = softmax_r(src[r * n + j])`.
///
/// Traverses rows contiguously along `j` (L1-friendly and 8-wide AVX2-vectorised), using caller-supplied
/// scratch slices `mx[..n]` and `sum[..n]` (zero allocations).
pub(crate) fn softmax_cols_row_major(
    src: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    mx: &mut [f32],
    sum: &mut [f32],
) {
    debug_assert!(src.len() >= m * n && dst.len() >= m * n && mx.len() >= n && sum.len() >= n);
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA verified by `has_avx2_fma()`.
        unsafe { softmax_cols_row_major_avx2(src, dst, m, n, mx, sum) };
        return;
    }
    softmax_cols_row_major_scalar(src, dst, m, n, mx, sum);
}

fn softmax_cols_non_finite(src: &[f32], dst: &mut [f32], m: usize, n: usize) {
    for j in 0..n {
        let mut mx = f32::NEG_INFINITY;
        for r in 0..m {
            let v = src[r * n + j];
            mx = if mx.is_nan() || v.is_nan() { f32::NAN } else { mx.max(v) };
        }
        let mut s = 0.0f32;
        for r in 0..m {
            let e = (src[r * n + j] - mx).exp();
            dst[r * n + j] = e;
            s += e;
        }
        for r in 0..m {
            dst[r * n + j] /= s;
        }
    }
}

fn softmax_cols_row_major_scalar(
    src: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    mx: &mut [f32],
    sum: &mut [f32],
) {
    let mut non_finite = false;
    mx[..n].fill(f32::NEG_INFINITY);
    for r in 0..m {
        let srow = &src[r * n..(r + 1) * n];
        for (mval, &v) in mx[..n].iter_mut().zip(srow) {
            if !v.is_finite() {
                non_finite = true;
            }
            if v > *mval {
                *mval = v;
            }
        }
    }
    if non_finite {
        softmax_cols_non_finite(src, dst, m, n);
        return;
    }
    sum[..n].fill(0.0);
    for r in 0..m {
        let srow = &src[r * n..(r + 1) * n];
        let drow = &mut dst[r * n..(r + 1) * n];
        for j in 0..n {
            let e = fast_exp_neg_scalar(srow[j] - mx[j]);
            drow[j] = e;
            sum[j] += e;
        }
    }
    for s in &mut sum[..n] {
        *s = 1.0 / *s;
    }
    for r in 0..m {
        let drow = &mut dst[r * n..(r + 1) * n];
        for j in 0..n {
            drow[j] *= sum[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_cols_row_major_avx2(
    src: &[f32],
    dst: &mut [f32],
    m: usize,
    n: usize,
    mx: &mut [f32],
    sum: &mut [f32],
) {
    let exp_mask = _mm256_set1_epi32(0x7f80_0000u32 as i32);
    let mut non_fin = _mm256_setzero_si256();
    let mut tail_nf = false;

    // SAFETY: `j + 8 <= n` in all SIMD loops; `src`/`dst` have length `>= m * n`; `mx`/`sum` have `>= n`.
    unsafe {
        // Pass 1: column maxima + non-finite check (row-major contiguous).
        let mp = mx.as_mut_ptr();
        {
            let sp0 = src.as_ptr();
            let mut j = 0;
            while j + 8 <= n {
                let v = _mm256_loadu_ps(sp0.add(j));
                _mm256_storeu_ps(mp.add(j), v);
                let is_nf = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v), exp_mask), exp_mask);
                non_fin = _mm256_or_si256(non_fin, is_nf);
                j += 8;
            }
            for jj in j..n {
                let v = src[jj];
                mx[jj] = v;
                if !v.is_finite() {
                    tail_nf = true;
                }
            }
        }
        for r in 1..m {
            let sp = src.as_ptr().add(r * n);
            let mut j = 0;
            while j + 8 <= n {
                let v = _mm256_loadu_ps(sp.add(j));
                let cur = _mm256_loadu_ps(mp.add(j));
                _mm256_storeu_ps(mp.add(j), _mm256_max_ps(cur, v));
                let is_nf = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(v), exp_mask), exp_mask);
                non_fin = _mm256_or_si256(non_fin, is_nf);
                j += 8;
            }
            for jj in j..n {
                let v = src[r * n + jj];
                if !v.is_finite() {
                    tail_nf = true;
                }
                if v > mx[jj] {
                    mx[jj] = v;
                }
            }
        }
        if tail_nf || _mm256_movemask_epi8(non_fin) != 0 {
            softmax_cols_non_finite(src, dst, m, n);
            return;
        }

        // Pass 2: exp(src - mx) and column sums (row-major contiguous).
        sum[..n].fill(0.0);
        let sp_sum = sum.as_mut_ptr();
        for r in 0..m {
            let sp = src.as_ptr().add(r * n);
            let dp = dst.as_mut_ptr().add(r * n);
            let mut j = 0;
            while j + 8 <= n {
                let v = _mm256_loadu_ps(sp.add(j));
                let m_v = _mm256_loadu_ps(mp.add(j));
                let e = fast_exp_neg_avx2(_mm256_sub_ps(v, m_v));
                _mm256_storeu_ps(dp.add(j), e);
                let s = _mm256_add_ps(_mm256_loadu_ps(sp_sum.add(j)), e);
                _mm256_storeu_ps(sp_sum.add(j), s);
                j += 8;
            }
            for jj in j..n {
                let e = fast_exp_neg_scalar(src[r * n + jj] - mx[jj]);
                dst[r * n + jj] = e;
                sum[jj] += e;
            }
        }

        // Pass 3: reciprocal + normalize columns (row-major contiguous).
        let v_one = _mm256_set1_ps(1.0);
        let mut j = 0;
        while j + 8 <= n {
            let inv = _mm256_div_ps(v_one, _mm256_loadu_ps(sp_sum.add(j)));
            _mm256_storeu_ps(sp_sum.add(j), inv);
            j += 8;
        }
        for s in &mut sum[j..n] {
            *s = 1.0 / *s;
        }
        for r in 0..m {
            let dp = dst.as_mut_ptr().add(r * n);
            let mut j = 0;
            while j + 8 <= n {
                let v = _mm256_mul_ps(_mm256_loadu_ps(dp.add(j)), _mm256_loadu_ps(sp_sum.add(j)));
                _mm256_storeu_ps(dp.add(j), v);
                j += 8;
            }
            for jj in j..n {
                dst[r * n + jj] *= sum[jj];
            }
        }
    }
}

/// Compute `out[r] = 1.0 + (dot(w[r, ..], x) + bias[r]).tanh()` for `r in 0..rows`, zero heap allocations.
pub(crate) fn fullpad_gate_values_row(
    w: &[f32],
    b: Option<&[f32]>,
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    debug_assert!(w.len() >= rows * cols && x.len() >= cols && out.len() >= rows);
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 and FMA verified by `has_avx2_fma()`.
        unsafe { fullpad_gate_values_row_avx2(w, b, x, rows, cols, out) };
        return;
    }
    for r in 0..rows {
        let wr = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (&wv, &xv) in wr.iter().zip(&x[..cols]) {
            acc = wv.mul_add(xv, acc);
        }
        let v = acc + b.map_or(0.0, |bs| bs[r]);
        out[r] = 1.0 + v.tanh();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fullpad_gate_values_row_avx2(
    w: &[f32],
    b: Option<&[f32]>,
    x: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    let xp = x.as_ptr();
    // SAFETY: `j + 8 <= cols` inside the SIMD loop; `w` has `>= rows * cols`, `x` has `>= cols`.
    unsafe {
        for r in 0..rows {
            let wp = w.as_ptr().add(r * cols);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut j = 0;
            while j + 16 <= cols {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(j)), _mm256_loadu_ps(xp.add(j)), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(j + 8)), _mm256_loadu_ps(xp.add(j + 8)), acc1);
                j += 16;
            }
            while j + 8 <= cols {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(j)), _mm256_loadu_ps(xp.add(j)), acc0);
                j += 8;
            }
            let mut dot = hsum_avx2(_mm256_add_ps(acc0, acc1));
            for jj in j..cols {
                dot = (*wp.add(jj)).mul_add(*xp.add(jj), dot);
            }
            let v = dot + b.map_or(0.0, |bs| bs[r]);
            out[r] = 1.0 + v.tanh();
        }
    }
}

/// Byte-wise maximum of two C4 words (bit-exact with `folo_core::cpu::q8::max_pool`).
#[inline(always)]
fn max4(a: u32, b: u32) -> u32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u32::from_le_bytes([0, 1, 2, 3].map(|t| a[t].max(b[t])))
}

/// Single C4 plane `k Ã— k` max pool (stride 1, padding `k / 2`), bit-exact with `q8::max_pool`.
pub(crate) fn max_pool_c4_plane(src: &[u32], h: usize, w: usize, r: usize, tmp: &mut [u32], dst: &mut [u32]) {
    let hw = h * w;
    debug_assert!(src.len() >= hw && tmp.len() >= hw && dst.len() >= hw);
    if hw == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: AVX2 verified by `has_avx2_fma()`; lengths checked above.
        unsafe { max_pool_c4_plane_avx2(src, h, w, r, tmp, dst) };
        return;
    }
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for xo in 0..w {
            let (lo, hi) = (xo.saturating_sub(r), xo.saturating_add(r).min(w - 1));
            tmp[y * w + xo] = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
    }
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), y.saturating_add(r).min(h - 1));
        for xo in 0..w {
            dst[y * w + xo] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xo]));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_pool_c4_plane_avx2(
    src: &[u32],
    h: usize,
    w: usize,
    r: usize,
    tmp: &mut [u32],
    dst: &mut [u32],
) {
    let span = r.saturating_mul(2).saturating_add(1);
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let out_row = &mut tmp[y * w..(y + 1) * w];
        let left_end = r.min(w);
        for (xo, d) in out_row.iter_mut().enumerate().take(left_end) {
            let hi = xo.saturating_add(r).min(w - 1);
            *d = row[0..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
        let mut xo = left_end;
        let inner_end = w.saturating_sub(r);
        // SAFETY: `xo >= r` and `xo + 8 <= w - r` implies `xo - r + d + 8 <= w` for all `d in 0..span`.
        unsafe {
            while xo + 8 <= inner_end {
                let base = row.as_ptr().add(xo - r);
                let mut m = _mm256_loadu_si256(base as *const __m256i);
                for d in 1..span {
                    m = _mm256_max_epu8(m, _mm256_loadu_si256(base.add(d) as *const __m256i));
                }
                _mm256_storeu_si256(out_row.as_mut_ptr().add(xo) as *mut __m256i, m);
                xo += 8;
            }
        }
        for (xx, d) in out_row.iter_mut().enumerate().skip(xo) {
            let (lo, hi) = (xx.saturating_sub(r), xx.saturating_add(r).min(w - 1));
            *d = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
    }
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = y.saturating_add(r).min(h - 1);
        let out_row = &mut dst[y * w..(y + 1) * w];
        let mut xo = 0;
        // SAFETY: `yy * w + xo + 8 <= h * w`.
        unsafe {
            while xo + 8 <= w {
                let mut m = _mm256_loadu_si256(tmp.as_ptr().add(lo * w + xo) as *const __m256i);
                for yy in (lo + 1)..=hi {
                    m = _mm256_max_epu8(
                        m,
                        _mm256_loadu_si256(tmp.as_ptr().add(yy * w + xo) as *const __m256i),
                    );
                }
                _mm256_storeu_si256(out_row.as_mut_ptr().add(xo) as *mut __m256i, m);
                xo += 8;
            }
        }
        for xx in xo..w {
            out_row[xx] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xx]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_row_and_cols_match_ieee_reference() {
        let mut vals = vec![0.0f32; 403]; // odd length to exercise SIMD + tail
        for (i, v) in vals.iter_mut().enumerate() {
            *v = ((i as f32 * 0.37).sin() * 6.0) - 2.0;
        }
        let scale = 32.0f32.powf(-0.5);
        let mut got = vals.clone();
        softmax_row_scaled(&mut got, scale);

        let scaled: Vec<f32> = vals.iter().map(|&v| v * scale).collect();
        let mx = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&v| (v - mx).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let want: Vec<f32> = exps.into_iter().map(|e| e / sum).collect();
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-6, "row softmax mismatch: got {a}, want {b}");
        }

        // Extreme underflow and non-finite fallback.
        let mut extreme = vec![-200.0, 0.0, -50.0, -88.0, -87.3, 1.0, -1000.0, 0.5, -90.0];
        softmax_row_scaled(&mut extreme, 1.0);
        let sum_e: f32 = extreme.iter().sum();
        assert!((sum_e - 1.0).abs() < 1e-6 && extreme.iter().all(|v| v.is_finite() && *v >= 0.0));

        let mut nf = vec![1.0, f32::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        softmax_row_scaled(&mut nf, 1.0);
        assert!(nf.iter().all(|v| v.is_nan()));

        // Column softmax over [16, 403].
        let (m, n) = (16, 403);
        let src: Vec<f32> = (0..m * n).map(|i| ((i as f32 * 0.13).cos()) * 4.0).collect();
        let mut dst = vec![0.0f32; m * n];
        let mut mx_buf = vec![0.0f32; n];
        let mut sum_buf = vec![0.0f32; n];
        softmax_cols_row_major(&src, &mut dst, m, n, &mut mx_buf, &mut sum_buf);
        for j in 0..n {
            let mut cmx = f32::NEG_INFINITY;
            for r in 0..m {
                cmx = cmx.max(src[r * n + j]);
            }
            let csum: f32 = (0..m).map(|r| (src[r * n + j] - cmx).exp()).sum();
            for r in 0..m {
                let want_v = (src[r * n + j] - cmx).exp() / csum;
                let got_v = dst[r * n + j];
                assert!((got_v - want_v).abs() < 1e-6, "col softmax ({r},{j}): got {got_v}, want {want_v}");
            }
        }
    }
}
