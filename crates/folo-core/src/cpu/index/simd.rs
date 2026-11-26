//! AVX2 and WebAssembly SIMD128 helpers for [`super`] (`index_select`, `gather`, `scatter_add`, and index
//! validation).

/// Validate and copy `src[..]` (`i64`) into `dst[..]` (`usize`) using AVX2 4-wide signed range checks.
/// Returns `Some(bad_offset)` with the smallest out-of-range offset inside `src` on error.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn to_usize_i64_chunk_avx2(
    src: &[i64],
    dst: &mut [usize],
    max_valid: i64,
) -> Option<usize> {
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let zero = _mm256_setzero_si256();
    let vmax = _mm256_set1_epi64x(max_valid);
    let (sc, sr) = src.as_chunks::<4>();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[i64; 4]` (32 bytes), `d` is `[usize; 4]` (32 bytes on 64-bit), AVX2 is enabled.
        unsafe {
            let v = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
            let neg = _mm256_cmpgt_epi64(zero, v);
            let hi = _mm256_cmpgt_epi64(v, vmax);
            let bad = _mm256_or_si256(neg, hi);
            if _mm256_testz_si256(bad, bad) == 0 {
                let base = ci * 4;
                for (k, &val) in s.iter().enumerate() {
                    if val < 0 || val > max_valid {
                        return Some(base + k);
                    }
                }
            }
            _mm256_storeu_si256(d.as_mut_ptr() as *mut __m256i, v);
        }
    }
    let tail_base = sc.len() * 4;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val < 0 || val > max_valid {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
}

/// Validate and copy `src[..]` (`u32`) into `dst[..]` (`usize`) using AVX2 8-wide unsigned range checks.
/// Returns `Some(bad_offset)` with the smallest out-of-range offset inside `src` on error.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn to_usize_u32_chunk_avx2(src: &[u32], dst: &mut [usize], size: usize) -> Option<usize> {
    use std::arch::x86_64::*;
    debug_assert_eq!(src.len(), dst.len());
    let bounded = size <= u32::MAX as usize;
    let max_u32 = if bounded { (size - 1) as u32 } else { u32::MAX };
    let vmax = _mm256_set1_epi32(max_u32 as i32);
    let (sc, sr) = src.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[u32; 8]` (32 bytes), `d` is `[usize; 8]` (64 bytes on 64-bit), AVX2 is enabled.
        unsafe {
            let v = _mm256_loadu_si256(s.as_ptr() as *const __m256i);
            if bounded {
                let ok = _mm256_cmpeq_epi32(_mm256_max_epu32(v, vmax), vmax);
                if _mm256_movemask_epi8(ok) != -1 {
                    let base = ci * 8;
                    for (k, &val) in s.iter().enumerate() {
                        if val > max_u32 {
                            return Some(base + k);
                        }
                    }
                }
            }
            let lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v));
            let hi = _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(v));
            _mm256_storeu_si256(d.as_mut_ptr() as *mut __m256i, lo);
            _mm256_storeu_si256(d.as_mut_ptr().add(4) as *mut __m256i, hi);
        }
    }
    let tail_base = sc.len() * 8;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val as u64 >= size as u64 {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
}

/// Validate and copy `src[..]` (`i64`) into `dst[..]` (`usize` = `u32` on `wasm32`) using Wasm SIMD128
/// 2-wide signed range checks. Returns `Some(bad_offset)` with the smallest out-of-range offset on error.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) fn to_usize_i64_chunk_wasm(src: &[i64], dst: &mut [usize], max_valid: i64) -> Option<usize> {
    use std::arch::wasm32::*;
    debug_assert_eq!(src.len(), dst.len());
    let zero = i64x2_splat(0);
    let vmax = i64x2_splat(max_valid);
    let (sc, sr) = src.as_chunks::<4>();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[i64; 4]` (32 bytes) and `d` is `[usize; 4]` (16 bytes on `wasm32`).
        unsafe {
            let p = s.as_ptr() as *const v128;
            let v0 = v128_load(p);
            let v1 = v128_load(p.add(1));
            let bad0 = v128_or(i64x2_lt(v0, zero), i64x2_gt(v0, vmax));
            let bad1 = v128_or(i64x2_lt(v1, zero), i64x2_gt(v1, vmax));
            if v128_any_true(v128_or(bad0, bad1)) {
                let base = ci * 4;
                for (k, &val) in s.iter().enumerate() {
                    if val < 0 || val > max_valid {
                        return Some(base + k);
                    }
                }
            }
            // Low 32 bits of each `i64` lane pack into 4 `u32` (`usize` on `wasm32`) lanes.
            let packed = u32x4_shuffle::<0, 2, 4, 6>(v0, v1);
            v128_store(d.as_mut_ptr() as *mut v128, packed);
        }
    }
    let tail_base = sc.len() * 4;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val < 0 || val > max_valid {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
}

/// Validate and copy `src[..]` (`u32`) into `dst[..]` (`usize` = `u32` on `wasm32`) using Wasm SIMD128
/// 4-wide unsigned range checks. Returns `Some(bad_offset)` with the smallest out-of-range offset on error.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) fn to_usize_u32_chunk_wasm(src: &[u32], dst: &mut [usize], size: usize) -> Option<usize> {
    use std::arch::wasm32::*;
    debug_assert_eq!(src.len(), dst.len());
    let max_u32 = (size - 1) as u32;
    let vmax = u32x4_splat(max_u32);
    let (sc, sr) = src.as_chunks::<4>();
    let (dc, dr) = dst.as_chunks_mut::<4>();
    for (ci, (s, d)) in sc.iter().zip(dc.iter_mut()).enumerate() {
        // SAFETY: `s` is `[u32; 4]` (16 bytes) and `d` is `[usize; 4]` (16 bytes on `wasm32`).
        unsafe {
            let v = v128_load(s.as_ptr() as *const v128);
            if v128_any_true(u32x4_gt(v, vmax)) {
                let base = ci * 4;
                for (k, &val) in s.iter().enumerate() {
                    if val > max_u32 {
                        return Some(base + k);
                    }
                }
            }
            v128_store(d.as_mut_ptr() as *mut v128, v);
        }
    }
    let tail_base = sc.len() * 4;
    for (k, (&val, out)) in sr.iter().zip(dr.iter_mut()).enumerate() {
        if val > max_u32 {
            return Some(tail_base + k);
        }
        *out = val as usize;
    }
    None
}

/// Pack 8 validated `usize` values (each `<= i32::MAX`) into 8 `i32` lanes in one `__m256i`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn pack_8_usize_to_i32(ix: &[usize; 8]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // SAFETY: `ix` is `[usize; 8]` (64 bytes on 64-bit) and AVX2 is enabled.
    unsafe {
        let p = ix.as_ptr() as *const f32;
        let lo = _mm256_loadu_ps(p);
        let hi = _mm256_loadu_ps(p.add(8));
        _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0x88>(lo, hi)))
    }
}

/// Gather `dst[t] = src[ix[t]]` for `t < dst.len()` using AVX2 `_mm256_i32gather_ps` when offsets fit `i32`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gather_1d_f32_avx2(src: &[f32], ix: &[usize], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ix.len(), dst.len());
    debug_assert!(src.len() <= i32::MAX as usize);
    let (ic, ir) = ix.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let base = src.as_ptr();
    for (i8, d8) in ic.iter().zip(dc.iter_mut()) {
        // SAFETY: every validated index is `< src.len() <= i32::MAX`, `d8` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let idx_v = pack_8_usize_to_i32(i8);
            let vals = _mm256_i32gather_ps::<4>(base, idx_v);
            _mm256_storeu_ps(d8.as_mut_ptr(), vals);
        }
    }
    for (d, &j) in dr.iter_mut().zip(ir) {
        *d = src[j];
    }
}

/// Gather one row `dst[k] = src_block[ix[k] * inner + k]` for `k in k0..k0 + dst.len()` using AVX2 gather.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn gather_row_f32_avx2(
    src_block: &[f32],
    inner: usize,
    k0: usize,
    ix: &[usize],
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ix.len(), dst.len());
    debug_assert!(src_block.len() <= i32::MAX as usize);
    let v_inner = _mm256_set1_epi32(inner as i32);
    let ramp = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut vk = _mm256_add_epi32(_mm256_set1_epi32(k0 as i32), ramp);
    let v8 = _mm256_set1_epi32(8);
    let (ic, ir) = ix.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    let base = src_block.as_ptr();
    for (i8, d8) in ic.iter().zip(dc.iter_mut()) {
        // SAFETY: every `ix[t] * inner + k < src_block.len() <= i32::MAX`, `d8` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let idx_v = pack_8_usize_to_i32(i8);
            let off = _mm256_add_epi32(_mm256_mullo_epi32(idx_v, v_inner), vk);
            let vals = _mm256_i32gather_ps::<4>(base, off);
            _mm256_storeu_ps(d8.as_mut_ptr(), vals);
            vk = _mm256_add_epi32(vk, v8);
        }
    }
    let tail_k0 = k0 + ic.len() * 8;
    for (t, (d, &j)) in dr.iter_mut().zip(ir).enumerate() {
        *d = src_block[j * inner + tail_k0 + t];
    }
}

/// Vectorised AVX2 `scatter_add` across one row's column slice `k in k0..k0 + src.len()`:
/// `dst_ptr[ix[t] * inner + k0 + t] += src[t]`.
///
/// Within any 8-column chunk, all 8 destination offsets have distinct remainders `k0 + t` modulo `inner`, so
/// the 8 destinations are strictly disjoint regardless of duplicate values in `ix`, making this bitwise
/// identical to the serial loop (including `-0.0`, `NaN`, and `±∞`). Uniform-index chunks take a direct
/// contiguous vector load/add/store; mixed-index chunks use `_mm256_i32gather_ps` + `_mm256_add_ps`.
///
/// # Safety
/// - The CPU supports AVX2.
/// - `ix.len() == src.len()`, `k0 + src.len() <= inner`, and `d * inner <= i32::MAX`.
/// - Every `j in ix` satisfies `j < d`.
/// - `dst_ptr` is valid for exclusive reads/writes at `j * inner + k` for all `j < d` and `k in k0..k0 + src.len()`.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn scatter_add_cols_avx2(
    dst_ptr: *mut f32,
    inner: usize,
    k0: usize,
    ix: &[usize],
    src: &[f32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ix.len(), src.len());
    let v_inner = _mm256_set1_epi32(inner as i32);
    let ramp = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut vk = _mm256_add_epi32(_mm256_set1_epi32(k0 as i32), ramp);
    let v8 = _mm256_set1_epi32(8);
    let (ic, ir) = ix.as_chunks::<8>();
    let (sc, sr) = src.as_chunks::<8>();
    let mut k = k0;
    for (i8, s8) in ic.iter().zip(sc.iter()) {
        // SAFETY: `i8` has 8 validated indices `< d`, `k + 8 <= inner`, all 8 destination slots are disjoint
        // and owned exclusively by the caller, and `d * inner <= i32::MAX`.
        unsafe {
            let idx_v = pack_8_usize_to_i32(i8);
            let sv = _mm256_loadu_ps(s8.as_ptr());
            let j0 = i8[0];
            let eq = _mm256_cmpeq_epi32(idx_v, _mm256_set1_epi32(j0 as i32));
            if _mm256_movemask_epi8(eq) == -1 {
                let p = dst_ptr.add(j0 * inner + k);
                let cur = _mm256_loadu_ps(p);
                _mm256_storeu_ps(p, _mm256_add_ps(cur, sv));
            } else {
                let off = _mm256_add_epi32(_mm256_mullo_epi32(idx_v, v_inner), vk);
                let cur = _mm256_i32gather_ps::<4>(dst_ptr as *const f32, off);
                let sum = _mm256_add_ps(cur, sv);
                let mut lanes = [0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
                for t in 0..8 {
                    *dst_ptr.add(i8[t] * inner + k + t) = lanes[t];
                }
            }
            vk = _mm256_add_epi32(vk, v8);
            k += 8;
        }
    }
    for (t, (&j, &v)) in ir.iter().zip(sr.iter()).enumerate() {
        // SAFETY: `j < d` and `k + t < inner`, exclusively owned by caller.
        unsafe { *dst_ptr.add(j * inner + k + t) += v };
    }
}
