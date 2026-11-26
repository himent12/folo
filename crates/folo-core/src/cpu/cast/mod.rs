//! Type conversions on the CPU, with zero-copy borrowing for contiguous inputs, rayon parallelism above
//! [`PAR_MIN`], 32-wide direct-pointer hardware F16C `f32 <-> f16` conversion on `x86_64`, and branchless
//! Wasm SIMD128 `f32 <-> f16` / `u8` / `u32` conversions.

#[cfg(test)]
mod tests;

use std::borrow::Cow;

use half::f16;

use super::{CHUNK, PAR_MIN, alloc_uninit, contiguous, elementwise, par_chunks, recycle};
use crate::{CpuStorage, DType, Element, Layout, with_cpu_storage};

fn cast_vec<S: Element, D: Element>(src: &[S]) -> Vec<D> {
    src.iter().map(|&x| D::from_f64(x.to_f64())).collect()
}

fn cast_storage<S: Element>(src: &[S], dtype: DType) -> CpuStorage {
    match dtype {
        DType::F32 => CpuStorage::F32(cast_vec(src)),
        DType::F16 => CpuStorage::F16(cast_vec(src)),
        DType::U8 => CpuStorage::U8(cast_vec(src)),
        DType::U32 => CpuStorage::U32(cast_vec(src)),
        DType::I64 => CpuStorage::I64(cast_vec(src)),
    }
}

fn empty_storage(dtype: DType) -> CpuStorage {
    match dtype {
        DType::F32 => CpuStorage::F32(Vec::new()),
        DType::F16 => CpuStorage::F16(Vec::new()),
        DType::U8 => CpuStorage::U8(Vec::new()),
        DType::U32 => CpuStorage::U32(Vec::new()),
        DType::I64 => CpuStorage::I64(Vec::new()),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_f16c() -> bool {
    std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx")
}

/// Convert `src` (`u8`) into `dst` (`f32`) 32 elements at a time with AVX2 (`vpmovzxbd` + `vcvtdq2ps`).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u8_to_f32_avx2(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let (s32, s_rem) = src.as_chunks::<32>();
    let (d32, d_rem) = dst.as_chunks_mut::<32>();
    for (sc, dc) in s32.iter().zip(d32.iter_mut()) {
        // SAFETY: `sc` is `[u8; 32]`, `dc` is `[f32; 32]`, AVX2 is enabled.
        unsafe {
            let p = sc.as_ptr() as *const __m128i;
            let o = dc.as_mut_ptr();
            let b0 = _mm_loadu_si128(p);
            let b1 = _mm_loadu_si128(p.add(1));
            _mm256_storeu_ps(o, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b0)));
            _mm256_storeu_ps(o.add(8), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(b0))));
            _mm256_storeu_ps(o.add(16), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b1)));
            _mm256_storeu_ps(o.add(24), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(b1))));
        }
    }
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[u8; 8]`, `dc` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let b = _mm_loadl_epi64(sc.as_ptr() as *const __m128i);
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)));
        }
    }
    for (d, &x) in d_tail.iter_mut().zip(s_tail) {
        *d = x as f32;
    }
}

/// Convert `src` (`f32`) into `dst` (`u8`) 32 elements at a time with AVX2, bitwise identical to Rust's
/// saturating `x as u8` (truncate toward zero, clamp to `[0, 255]`, `NaN -> 0`).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn f32_to_u8_avx2(src: &[f32], dst: &mut [u8]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let zero = _mm256_setzero_ps();
    let max255 = _mm256_set1_ps(255.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let zero_i = _mm256_setzero_si256();
    // `vmaxps(v, 0.0)` returns `0.0` for `NaN`, `-Inf`, and negatives (second operand wins on NaN);
    // `vminps(_, 255.0)` then saturates `+Inf` and `> 255.0` to `255.0`, and `cvttps` truncates toward zero.
    macro_rules! cvt8 {
        ($ptr:expr) => {
            _mm256_cvttps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps($ptr), zero), max255))
        };
    }
    let (s32, s_rem) = src.as_chunks::<32>();
    let (d32, d_rem) = dst.as_chunks_mut::<32>();
    for (sc, dc) in s32.iter().zip(d32.iter_mut()) {
        // SAFETY: `sc` is `[f32; 32]`, `dc` is `[u8; 32]`, AVX2 is enabled.
        unsafe {
            let p = sc.as_ptr();
            let q0 = cvt8!(p);
            let q1 = cvt8!(p.add(8));
            let q2 = cvt8!(p.add(16));
            let q3 = cvt8!(p.add(24));
            let p01 = _mm256_packs_epi32(q0, q1);
            let p23 = _mm256_packs_epi32(q2, q3);
            let packed = _mm256_packus_epi16(p01, p23);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm256_storeu_si256(dc.as_mut_ptr() as *mut __m256i, ordered);
        }
    }
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[f32; 8]`, `dc` is `[u8; 8]`, AVX2 is enabled.
        unsafe {
            let q0 = cvt8!(sc.as_ptr());
            let p0 = _mm256_packs_epi32(q0, zero_i);
            let packed = _mm256_packus_epi16(p0, zero_i);
            let ordered = _mm256_permutevar8x32_epi32(packed, perm);
            _mm_storel_epi64(dc.as_mut_ptr() as *mut __m128i, _mm256_castsi256_si128(ordered));
        }
    }
    for (d, &x) in d_tail.iter_mut().zip(s_tail) {
        *d = x as u8;
    }
}

/// Convert `src` (`u32`) into `dst` (`f32`) 16 elements at a time with AVX2, bitwise identical to `x as f32`
/// across all `2^32` unsigned values (`(u >> 16) * 65536.0 + (u & 0xFFFF)` has a single IEEE-754 rounding).
///
/// # Safety
/// The CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u32_to_f32_avx2(src: &[u32], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let mask_lo = _mm256_set1_epi32(0xFFFF);
    let scale_hi = _mm256_set1_ps(65536.0);
    let (s16, s_rem) = src.as_chunks::<16>();
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    for (sc, dc) in s16.iter().zip(d16.iter_mut()) {
        // SAFETY: `sc` is `[u32; 16]`, `dc` is `[f32; 16]`, AVX2 is enabled.
        unsafe {
            let p = sc.as_ptr() as *const __m256i;
            let u0 = _mm256_loadu_si256(p);
            let u1 = _mm256_loadu_si256(p.add(1));
            let lo0 = _mm256_cvtepi32_ps(_mm256_and_si256(u0, mask_lo));
            let hi0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(u0)), scale_hi);
            let lo1 = _mm256_cvtepi32_ps(_mm256_and_si256(u1, mask_lo));
            let hi1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(u1)), scale_hi);
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(hi0, lo0));
            _mm256_storeu_ps(dc.as_mut_ptr().add(8), _mm256_add_ps(hi1, lo1));
        }
    }
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[u32; 8]`, `dc` is `[f32; 8]`, AVX2 is enabled.
        unsafe {
            let u0 = _mm256_loadu_si256(sc.as_ptr() as *const __m256i);
            let lo0 = _mm256_cvtepi32_ps(_mm256_and_si256(u0, mask_lo));
            let hi0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32::<16>(u0)), scale_hi);
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_add_ps(hi0, lo0));
        }
    }
    for (d, &x) in d_tail.iter_mut().zip(s_tail) {
        *d = x as f32;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn u8_to_f32_wasm(src: &[u8], dst: &mut [f32]) {
    use std::arch::wasm32::*;
    assert_eq!(src.len(), dst.len());
    let (s16, s_rem) = src.as_chunks::<16>();
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    for (sc, dc) in s16.iter().zip(d16.iter_mut()) {
        // SAFETY: `sc` is `[u8; 16]` and `dc` is `[f32; 16]`.
        unsafe {
            let b = v128_load(sc.as_ptr() as *const v128);
            let lo16 = u16x8_extend_low_u8x16(b);
            let hi16 = u16x8_extend_high_u8x16(b);
            let o = dc.as_mut_ptr();
            v128_store(o as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(lo16)));
            v128_store(o.add(4) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(lo16)));
            v128_store(o.add(8) as *mut v128, f32x4_convert_u32x4(u32x4_extend_low_u16x8(hi16)));
            v128_store(o.add(12) as *mut v128, f32x4_convert_u32x4(u32x4_extend_high_u16x8(hi16)));
        }
    }
    for (d, &x) in d_rem.iter_mut().zip(s_rem) {
        *d = x as f32;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn f32_to_u8_wasm(src: &[f32], dst: &mut [u8]) {
    use std::arch::wasm32::*;
    assert_eq!(src.len(), dst.len());
    let zero = f32x4_splat(0.0);
    let max255 = f32x4_splat(255.0);
    let (s16, s_rem) = src.as_chunks::<16>();
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    for (sc, dc) in s16.iter().zip(d16.iter_mut()) {
        // SAFETY: `sc` is `[f32; 16]` and `dc` is `[u8; 16]`.
        unsafe {
            let p = sc.as_ptr();
            let q0 = i32x4_trunc_sat_f32x4(f32x4_pmin(max255, f32x4_pmax(zero, v128_load(p as *const v128))));
            let q1 = i32x4_trunc_sat_f32x4(f32x4_pmin(
                max255,
                f32x4_pmax(zero, v128_load(p.add(4) as *const v128)),
            ));
            let q2 = i32x4_trunc_sat_f32x4(f32x4_pmin(
                max255,
                f32x4_pmax(zero, v128_load(p.add(8) as *const v128)),
            ));
            let q3 = i32x4_trunc_sat_f32x4(f32x4_pmin(
                max255,
                f32x4_pmax(zero, v128_load(p.add(12) as *const v128)),
            ));
            let p01 = i16x8_narrow_i32x4(q0, q1);
            let p23 = i16x8_narrow_i32x4(q2, q3);
            v128_store(dc.as_mut_ptr() as *mut v128, u8x16_narrow_i16x8(p01, p23));
        }
    }
    for (d, &x) in d_rem.iter_mut().zip(s_rem) {
        *d = x as u8;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn u32_to_f32_wasm(src: &[u32], dst: &mut [f32]) {
    use std::arch::wasm32::*;
    assert_eq!(src.len(), dst.len());
    let (s8, s_rem) = src.as_chunks::<8>();
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[u32; 8]` and `dc` is `[f32; 8]`.
        unsafe {
            let p = sc.as_ptr();
            let o = dc.as_mut_ptr();
            v128_store(o as *mut v128, f32x4_convert_u32x4(v128_load(p as *const v128)));
            v128_store(o.add(4) as *mut v128, f32x4_convert_u32x4(v128_load(p.add(4) as *const v128)));
        }
    }
    let (s4, s_tail) = s_rem.as_chunks::<4>();
    let (d4, d_tail) = d_rem.as_chunks_mut::<4>();
    for (sc, dc) in s4.iter().zip(d4.iter_mut()) {
        // SAFETY: `sc` is `[u32; 4]` and `dc` is `[f32; 4]`.
        unsafe {
            v128_store(
                dc.as_mut_ptr() as *mut v128,
                f32x4_convert_u32x4(v128_load(sc.as_ptr() as *const v128)),
            );
        }
    }
    for (d, &x) in d_tail.iter_mut().zip(s_tail) {
        *d = x as f32;
    }
}

fn u8_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { u8_to_f32_avx2(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if src.len() >= 16 {
        u8_to_f32_wasm(src, dst);
        return;
    }
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x as f32;
    }
}

fn f32_to_u8_slice(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { f32_to_u8_avx2(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if src.len() >= 16 {
        f32_to_u8_wasm(src, dst);
        return;
    }
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x as u8;
    }
}

fn u32_to_f32_slice(src: &[u32], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { u32_to_f32_avx2(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if src.len() >= 4 {
        u32_to_f32_wasm(src, dst);
        return;
    }
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x as f32;
    }
}

// `half::f16` is `#[repr(transparent)]` over `u16` (2 bytes, alignment 2), allowing direct unaligned SIMD
// loads/stores without intermediate stack array copies.
const _: () = assert!(std::mem::size_of::<f16>() == 2 && std::mem::align_of::<f16>() == 2);

/// Convert `src` (`f32`) into `dst` (`f16`) 32 elements at a time using direct-pointer IEEE-754
/// round-to-nearest-even (`VCVTPS2PH`).
///
/// # Safety
/// The CPU supports AVX and F16C.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_to_f16_avx(src: &[f32], dst: &mut [f16]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let (s32, s_rem) = src.as_chunks::<32>();
    let (d32, d_rem) = dst.as_chunks_mut::<32>();
    for (sc, dc) in s32.iter().zip(d32.iter_mut()) {
        // SAFETY: `sc` is `[f32; 32]` (128 bytes) and `dc` is `[f16; 32]` (64 bytes, `#[repr(transparent)]`
        // over `u16`); unaligned AVX/F16C loads and stores are valid for any alignment.
        unsafe {
            let sp = sc.as_ptr();
            let dp = dc.as_mut_ptr() as *mut __m128i;
            let h0 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(sp));
            let h1 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(sp.add(8)));
            let h2 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(sp.add(16)));
            let h3 = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_loadu_ps(sp.add(24)));
            _mm_storeu_si128(dp, h0);
            _mm_storeu_si128(dp.add(1), h1);
            _mm_storeu_si128(dp.add(2), h2);
            _mm_storeu_si128(dp.add(3), h3);
        }
    }
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[f32; 8]` (32 bytes) and `dc` is `[f16; 8]` (16 bytes); AVX+F16C enabled.
        unsafe {
            let v = _mm256_loadu_ps(sc.as_ptr());
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dc.as_mut_ptr() as *mut __m128i, h);
        }
    }
    for (d, &s) in d_tail.iter_mut().zip(s_tail) {
        *d = f16::from_f32(s);
    }
}

/// Convert `src` (`f16`) into `dst` (`f32`) 32 elements at a time using direct-pointer hardware `VCVTPH2PS`.
///
/// # Safety
/// The CPU supports AVX and F16C.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16_to_f32_avx(src: &[f16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let (s32, s_rem) = src.as_chunks::<32>();
    let (d32, d_rem) = dst.as_chunks_mut::<32>();
    for (sc, dc) in s32.iter().zip(d32.iter_mut()) {
        // SAFETY: `sc` is `[f16; 32]` (64 bytes) and `dc` is `[f32; 32]` (128 bytes); AVX+F16C enabled.
        unsafe {
            let sp = sc.as_ptr() as *const __m128i;
            let dp = dc.as_mut_ptr();
            let v0 = _mm256_cvtph_ps(_mm_loadu_si128(sp));
            let v1 = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(1)));
            let v2 = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(2)));
            let v3 = _mm256_cvtph_ps(_mm_loadu_si128(sp.add(3)));
            _mm256_storeu_ps(dp, v0);
            _mm256_storeu_ps(dp.add(8), v1);
            _mm256_storeu_ps(dp.add(16), v2);
            _mm256_storeu_ps(dp.add(24), v3);
        }
    }
    let (s8, s_tail) = s_rem.as_chunks::<8>();
    let (d8, d_tail) = d_rem.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[f16; 8]` (16 bytes) and `dc` is `[f32; 8]` (32 bytes); AVX+F16C enabled.
        unsafe {
            let h = _mm_loadu_si128(sc.as_ptr() as *const __m128i);
            _mm256_storeu_ps(dc.as_mut_ptr(), _mm256_cvtph_ps(h));
        }
    }
    for (d, &s) in d_tail.iter_mut().zip(s_tail) {
        *d = s.to_f32();
    }
}

/// Branchless 4-lane Wasm SIMD128 `f16 -> f32` (exact across normals, subnormals, ±0, ±∞, and quiet NaNs).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn f16x4_to_f32x4_wasm(h32: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    let sign = u32x4_shl(v128_and(h32, u32x4_splat(0x8000)), 16);
    let exp = v128_and(u32x4_shr(h32, 10), u32x4_splat(0x1F));
    let mant = v128_and(h32, u32x4_splat(0x03FF));
    let norm = v128_or(sign, v128_or(u32x4_shl(u32x4_add(exp, u32x4_splat(112)), 23), u32x4_shl(mant, 13)));
    // Every 10-bit subnormal mantissa `0..=1023` is exact in `f32`, and `2^-24` (`0x3380_0000`) is an exact power of 2.
    let sub_mag = f32x4_mul(f32x4_convert_u32x4(mant), f32x4_splat(f32::from_bits(0x3380_0000)));
    let sub = v128_or(sign, sub_mag);
    let is_nan = u32x4_ne(mant, u32x4_splat(0));
    let inf_nan = v128_or(
        v128_or(sign, u32x4_splat(0x7F80_0000)),
        v128_or(u32x4_shl(mant, 13), v128_and(is_nan, u32x4_splat(0x0040_0000))),
    );
    let is_zero_or_sub = u32x4_eq(exp, u32x4_splat(0));
    let is_inf_or_nan = u32x4_eq(exp, u32x4_splat(31));
    let res = v128_bitselect(sub, norm, is_zero_or_sub);
    v128_bitselect(inf_nan, res, is_inf_or_nan)
}

/// Branchless 4-lane Wasm SIMD128 `f32 -> f16` (IEEE-754 round-to-nearest-even across normals, subnormals,
/// underflow, overflow to ±∞, and quiet NaNs), returned in the low 16 bits of each `u32` lane.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn f32x4_to_f16x4_wasm(u: std::arch::wasm32::v128) -> std::arch::wasm32::v128 {
    use std::arch::wasm32::*;
    let sign = v128_and(u32x4_shr(u, 16), u32x4_splat(0x8000));
    let abs_u = v128_and(u, u32x4_splat(0x7FFF_FFFF));
    // Subnormal + zero + subnormal-to-normal carry (`|x| < 2^-14`): adding `0.5` (`0x3F00_0000`, ULP = `2^-24`)
    // performs one exact IEEE-754 RNE rounding onto the `2^-24` grid; subtracting `0x3F00_0000` yields the f16 bits.
    let sub_f = f32x4_add(abs_u, f32x4_splat(0.5));
    let sub_h = u32x4_sub(sub_f, u32x4_splat(0x3F00_0000));
    // Normal (`0x3880_0000 <= abs_u < 0x4780_0000`): RNE bias `0x0FFF + lsb` carries `65520.0` cleanly into `0x7C00`.
    let lsb = v128_and(u32x4_shr(abs_u, 13), u32x4_splat(1));
    let norm_h = u32x4_shr(
        u32x4_add(u32x4_add(u32x4_sub(abs_u, u32x4_splat(0x3800_0000)), u32x4_splat(0x0FFF)), lsb),
        13,
    );
    let inf_h = u32x4_splat(0x7C00);
    let nan_h = v128_or(u32x4_splat(0x7E00), v128_and(u32x4_shr(abs_u, 13), u32x4_splat(0x03FF)));
    let is_sub = u32x4_lt(abs_u, u32x4_splat(0x3880_0000));
    let is_ovf = u32x4_ge(abs_u, u32x4_splat(0x4780_0000));
    let is_nan = u32x4_gt(abs_u, u32x4_splat(0x7F80_0000));
    let mag = v128_bitselect(sub_h, norm_h, is_sub);
    let mag = v128_bitselect(inf_h, mag, is_ovf);
    let mag = v128_bitselect(nan_h, mag, is_nan);
    v128_or(sign, mag)
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn f16_to_f32_wasm(src: &[f16], dst: &mut [f32]) {
    use std::arch::wasm32::*;
    assert_eq!(src.len(), dst.len());
    let (s8, s_rem) = src.as_chunks::<8>();
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[f16; 8]` (16 bytes) and `dc` is `[f32; 8]` (32 bytes).
        unsafe {
            let h = v128_load(sc.as_ptr() as *const v128);
            let lo = f16x4_to_f32x4_wasm(u32x4_extend_low_u16x8(h));
            let hi = f16x4_to_f32x4_wasm(u32x4_extend_high_u16x8(h));
            let dp = dc.as_mut_ptr();
            v128_store(dp as *mut v128, lo);
            v128_store(dp.add(4) as *mut v128, hi);
        }
    }
    for (d, &s) in d_rem.iter_mut().zip(s_rem) {
        *d = s.to_f32();
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn f32_to_f16_wasm(src: &[f32], dst: &mut [f16]) {
    use std::arch::wasm32::*;
    assert_eq!(src.len(), dst.len());
    let (s8, s_rem) = src.as_chunks::<8>();
    let (d8, d_rem) = dst.as_chunks_mut::<8>();
    for (sc, dc) in s8.iter().zip(d8.iter_mut()) {
        // SAFETY: `sc` is `[f32; 8]` (32 bytes) and `dc` is `[f16; 8]` (16 bytes).
        unsafe {
            let sp = sc.as_ptr();
            let lo = f32x4_to_f16x4_wasm(v128_load(sp as *const v128));
            let hi = f32x4_to_f16x4_wasm(v128_load(sp.add(4) as *const v128));
            // Pack the low 16 bits of each `u32` lane from `lo` and `hi` into 8 contiguous `u16` lanes.
            let packed = u16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(lo, hi);
            v128_store(dc.as_mut_ptr() as *mut v128, packed);
        }
    }
    for (d, &s) in d_rem.iter_mut().zip(s_rem) {
        *d = f16::from_f32(s);
    }
}

fn f32_to_f16_slice(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_f16c() {
        // SAFETY: AVX and F16C were detected above.
        unsafe { f32_to_f16_avx(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if src.len() >= 8 {
        f32_to_f16_wasm(src, dst);
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f16::from_f32(s);
    }
}

fn f16_to_f32_slice(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_f16c() {
        // SAFETY: AVX and F16C were detected above.
        unsafe { f16_to_f32_avx(src, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if src.len() >= 8 {
        f16_to_f32_wasm(src, dst);
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.to_f32();
    }
}

fn map_to_f32<S: Copy + Send + Sync>(src: &[S], f: impl Fn(&[S], &mut [f32]) + Sync) -> Vec<f32> {
    let n = src.len();
    let mut out = alloc_uninit(n);
    if n >= PAR_MIN {
        par_chunks(&mut out, CHUNK, &|c, chunk| {
            let start = c * CHUNK;
            f(&src[start..start + chunk.len()], chunk);
        });
    } else {
        f(src, &mut out);
    }
    out
}

/// Cast `src` (viewed through `layout`) to `dtype`, avoiding intermediate copies when `src` is contiguous.
pub(crate) fn cast(src: &CpuStorage, layout: &Layout, dtype: DType) -> CpuStorage {
    let n = layout.numel();
    if n == 0 {
        return empty_storage(dtype);
    }
    match src {
        CpuStorage::F32(v) if dtype == DType::F16 => {
            let data: Cow<'_, [f32]> = match layout.contiguous_range() {
                Some(r) => Cow::Borrowed(&v[r]),
                None => Cow::Owned(elementwise::copy_strided_f32(v, layout)),
            };
            let mut out = vec![f16::ZERO; n];
            if n >= PAR_MIN {
                par_chunks(&mut out, CHUNK, &|c, chunk| {
                    let start = c * CHUNK;
                    f32_to_f16_slice(&data[start..start + chunk.len()], chunk);
                });
            } else {
                f32_to_f16_slice(&data, &mut out);
            }
            if let Cow::Owned(buf) = data {
                recycle(buf);
            }
            CpuStorage::F16(out)
        }
        CpuStorage::F32(v) if dtype == DType::U8 => {
            let data: Cow<'_, [f32]> = match layout.contiguous_range() {
                Some(r) => Cow::Borrowed(&v[r]),
                None => Cow::Owned(elementwise::copy_strided_f32(v, layout)),
            };
            let mut out = vec![0u8; n];
            if n >= PAR_MIN {
                par_chunks(&mut out, CHUNK, &|c, chunk| {
                    let start = c * CHUNK;
                    f32_to_u8_slice(&data[start..start + chunk.len()], chunk);
                });
            } else {
                f32_to_u8_slice(&data, &mut out);
            }
            if let Cow::Owned(buf) = data {
                recycle(buf);
            }
            CpuStorage::U8(out)
        }
        CpuStorage::F16(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, f16_to_f32_slice))
        }
        CpuStorage::U8(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, u8_to_f32_slice))
        }
        CpuStorage::U32(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, u32_to_f32_slice))
        }
        // Directly, not through f64: above 2^53 that would round twice.
        CpuStorage::I64(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, |s, d| {
                for (dst, &x) in d.iter_mut().zip(s) {
                    *dst = x as f32;
                }
            }))
        }
        s => with_cpu_storage!(s, v => cast_storage(&contiguous(v, layout), dtype)),
    }
}
