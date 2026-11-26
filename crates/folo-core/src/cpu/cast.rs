//! Type conversions on the CPU, with zero-copy borrowing for contiguous inputs, rayon parallelism above
//! [`PAR_MIN`], and hardware F16C `f32 <-> f16` conversion on `x86_64`.

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

fn u8_to_f32_slice(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= 8 && has_avx2() {
        // SAFETY: AVX2 was detected above.
        unsafe { u8_to_f32_avx2(src, dst) };
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
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = x as f32;
    }
}

/// Convert `src` (`f32`) into `dst` (`f16`) using IEEE-754 round-to-nearest-even (`VCVTPS2PH`).
///
/// # Safety
/// The CPU supports AVX and F16C.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f32_to_f16_avx(src: &[f32], dst: &mut [f16]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let (s_chunks, s_rem) = src.as_chunks::<8>();
    let (d_chunks, d_rem) = dst.as_chunks_mut::<8>();
    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        // SAFETY: `sc` is `[f32; 8]` (32 bytes) and `out` is `[u16; 8]` (16 bytes); AVX+F16C enabled.
        let bits = unsafe {
            let v = _mm256_loadu_ps(sc.as_ptr());
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            let mut out = [0u16; 8];
            _mm_storeu_si128(out.as_mut_ptr().cast(), h);
            out
        };
        for (d, b) in dc.iter_mut().zip(bits) {
            *d = f16::from_bits(b);
        }
    }
    for (d, &s) in d_rem.iter_mut().zip(s_rem) {
        *d = f16::from_f32(s);
    }
}

/// Convert `src` (`f16`) into `dst` (`f32`) using hardware `VCVTPH2PS`.
///
/// # Safety
/// The CPU supports AVX and F16C.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16_to_f32_avx(src: &[f16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    assert_eq!(src.len(), dst.len());
    let (s_chunks, s_rem) = src.as_chunks::<8>();
    let (d_chunks, d_rem) = dst.as_chunks_mut::<8>();
    for (sc, dc) in s_chunks.iter().zip(d_chunks.iter_mut()) {
        let bits: [u16; 8] = std::array::from_fn(|i| sc[i].to_bits());
        // SAFETY: `bits` is `[u16; 8]` (16 bytes) and `dc` is `[f32; 8]` (32 bytes); AVX+F16C enabled.
        unsafe {
            let h = _mm_loadu_si128(bits.as_ptr().cast());
            let v = _mm256_cvtph_ps(h);
            _mm256_storeu_ps(dc.as_mut_ptr(), v);
        }
    }
    for (d, &s) in d_rem.iter_mut().zip(s_rem) {
        *d = s.to_f32();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn f16_to_f32_exhaustive_matches_half() {
        let all_f16: Vec<f16> = (0u16..=u16::MAX).map(f16::from_bits).collect();
        let mut got = vec![0f32; all_f16.len()];
        f16_to_f32_slice(&all_f16, &mut got);
        for (&h, &v) in all_f16.iter().zip(&got) {
            let want = h.to_f32();
            if want.is_nan() {
                assert!(v.is_nan(), "f16 {:#06x}: expected NaN, got {v}", h.to_bits());
            } else {
                assert_eq!(
                    v.to_bits(),
                    want.to_bits(),
                    "f16 {:#06x}: got {:#010x}, want {:#010x}",
                    h.to_bits(),
                    v.to_bits(),
                    want.to_bits()
                );
            }
        }
        // Also test ragged tails (lengths 0..17) and parallel cast (> PAR_MIN).
        for len in 0..17 {
            let mut tail = vec![0f32; len];
            f16_to_f32_slice(&all_f16[..len], &mut tail);
            for i in 0..len {
                assert_eq!(tail[i].to_bits(), all_f16[i].to_f32().to_bits());
            }
        }
        let storage = CpuStorage::F16(all_f16.clone());
        let l = Layout::contiguous(Shape::new(&[all_f16.len()]));
        let CpuStorage::F32(par_out) = cast(&storage, &l, DType::F32) else {
            panic!("expected F32");
        };
        assert_eq!(par_out.len(), got.len());
        for (a, b) in par_out.iter().zip(&got) {
            if b.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn f32_to_f16_matches_half_across_ties_subnormals_and_extremes() {
        let mut xs: Vec<f32> = (0u16..=u16::MAX)
            .filter_map(|b| {
                let h = f16::from_bits(b);
                (!h.is_nan()).then(|| h.to_f32())
            })
            .collect();
        // Exact halfway ties-to-even (normal and subnormal), underflow, overflow, signed zeros, NaNs.
        xs.extend([
            0.0,
            -0.0,
            f32::from_bits(0x3300_0000), // half of min f16 subnormal (2^-25 -> rounds to even 0)
            f32::from_bits(0x3300_0001), // just above half of min f16 subnormal -> min subnormal
            f32::from_bits(0x3380_0000), // min f16 subnormal (2^-24)
            f32::from_bits(0x33c0_0000), // 1.5 * 2^-24 -> rounds to even 2 * 2^-24
            f32::from_bits(0x3880_0000), // min f16 normal (2^-14)
            1.0 + f32::exp2(-11.0),      // exact halfway above even mantissa -> 1.0
            1.0 + f32::exp2(-10.0) + f32::exp2(-11.0), // exact halfway above odd mantissa -> rounds up
            65504.0,                     // max finite f16
            65519.996,                   // rounds down to 65504.0
            65520.0,                     // exact tie -> overflows to +inf
            -65520.0,                    // overflows to -inf
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0x7f80_0001), // sNaN
        ]);
        // Non-multiple of 8 length exercising tail path + parallel threshold.
        while xs.len() % 8 != 5 {
            xs.push(0.125);
        }
        let storage = CpuStorage::F32(xs.clone());
        let l = Layout::contiguous(Shape::new(&[xs.len()]));
        let CpuStorage::F16(got) = cast(&storage, &l, DType::F16) else {
            panic!("expected F16");
        };
        for (&x, &h) in xs.iter().zip(&got) {
            let want = f16::from_f32(x);
            if want.is_nan() {
                assert!(
                    h.is_nan(),
                    "f32 {x} ({:#010x}): expected NaN, got {:#06x}",
                    x.to_bits(),
                    h.to_bits()
                );
            } else {
                assert_eq!(
                    h.to_bits(),
                    want.to_bits(),
                    "f32 {x} ({:#010x}): got {:#06x}, want {:#06x}",
                    x.to_bits(),
                    h.to_bits(),
                    want.to_bits()
                );
            }
        }
    }

    #[test]
    fn cast_handles_empty_offset_and_non_contiguous_views() {
        let empty = Layout::new(Shape::new(&[0, 4]), vec![4, 1].into(), 10);
        let CpuStorage::F16(v) = cast(&CpuStorage::F32(vec![]), &empty, DType::F16) else {
            panic!("expected F16");
        };
        assert!(v.is_empty());

        // Transposed F32 -> F16 and I64 -> F32 (single-rounding above 2^53).
        let src: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let lt = Layout::contiguous(Shape::new(&[4, 6])).transpose(0, 1);
        let CpuStorage::F16(h) = cast(&CpuStorage::F32(src.clone()), &lt, DType::F16) else {
            panic!("expected F16");
        };
        for c in 0..6 {
            for r in 0..4 {
                assert_eq!(h[c * 4 + r].to_bits(), f16::from_f32(src[r * 6 + c]).to_bits());
            }
        }

        let big: i64 = (1i64 << 53) + 1;
        let l1 = Layout::contiguous(Shape::new(&[1]));
        let CpuStorage::F32(out) = cast(&CpuStorage::I64(vec![big]), &l1, DType::F32) else {
            panic!("expected F32");
        };
        assert_eq!(out[0].to_bits(), (big as f32).to_bits());
    }

    #[test]
    fn u8_f32_u32_avx2_casts_match_scalar_bitwise() {
        // 1. U8 -> F32: all 256 byte values, odd lengths 0..=65, and parallel > PAR_MIN.
        let bytes: Vec<u8> = (0u8..=255).collect();
        for len in [0usize, 1, 7, 8, 9, 15, 31, 32, 33, 65, 256] {
            let mut got = vec![0f32; len];
            u8_to_f32_slice(&bytes[..len], &mut got);
            for (i, (&g, &b)) in got.iter().zip(&bytes[..len]).enumerate() {
                assert_eq!(g.to_bits(), (b as f32).to_bits(), "u8->f32 len={len} at {i}");
            }
        }
        let big_u8: Vec<u8> = (0..PAR_MIN + 37).map(|i| (i * 31) as u8).collect();
        let CpuStorage::F32(got_big) = cast(
            &CpuStorage::U8(big_u8.clone()),
            &Layout::contiguous(Shape::new(&[big_u8.len()])),
            DType::F32,
        ) else {
            panic!("expected F32");
        };
        for (&g, &b) in got_big.iter().zip(&big_u8) {
            assert_eq!(g.to_bits(), (b as f32).to_bits());
        }

        // 2. F32 -> U8: NaNs, ±Inf, ±0.0, negatives, fractional truncation, saturation, tails, strided, parallel.
        let mut f_vals = vec![
            0.0f32,
            -0.0,
            -0.1,
            -1.0,
            -1000.0,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::from_bits(0x7f80_0001),
            0.0001,
            0.5,
            0.9999,
            1.0,
            1.9999,
            127.5,
            128.5,
            254.0,
            254.9999,
            255.0,
            255.0001,
            255.9,
            256.0,
            1e20,
            f32::MAX,
            f32::INFINITY,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
        ];
        for i in 0..45 {
            f_vals.push(i as f32 * 6.25 - 12.3);
        }
        for len in [0usize, 1, 7, 8, 9, 15, 31, 32, 33, 41, f_vals.len()] {
            let mut got = vec![0u8; len];
            f32_to_u8_slice(&f_vals[..len], &mut got);
            for (i, (&g, &x)) in got.iter().zip(&f_vals[..len]).enumerate() {
                assert_eq!(g, x as u8, "f32->u8 len={len} at {i} x={x}");
            }
        }
        let big_f: Vec<f32> = (0..PAR_MIN + 41).map(|i| (i % 300) as f32 - 20.5).collect();
        let CpuStorage::U8(got_u8) =
            cast(&CpuStorage::F32(big_f.clone()), &Layout::contiguous(Shape::new(&[big_f.len()])), DType::U8)
        else {
            panic!("expected U8");
        };
        for (&g, &x) in got_u8.iter().zip(&big_f) {
            assert_eq!(g, x as u8);
        }
        let lt = Layout::contiguous(Shape::new(&[4, 9])).transpose(0, 1);
        let CpuStorage::U8(got_t) = cast(&CpuStorage::F32(f_vals[..36].to_vec()), &lt, DType::U8) else {
            panic!("expected U8");
        };
        for c in 0..9 {
            for r in 0..4 {
                assert_eq!(got_t[c * 4 + r], f_vals[r * 9 + c] as u8);
            }
        }

        // 3. U32 -> F32: full 32-bit range boundaries including exact halfway ties-to-even, tails, and parallel.
        let mut u_vals = vec![
            0u32,
            1,
            0xFFFF,
            0x1_0000,
            0x1_0001,
            (1u32 << 24) - 1,
            1u32 << 24,
            (1u32 << 24) + 1,
            (1u32 << 24) + 2,
            (1u32 << 24) + 3,
            0x7FFF_FF7F,
            0x7FFF_FF80,
            0x7FFF_FFBF,
            0x7FFF_FFC0,
            0x7FFF_FFFF,
            0x8000_0000,
            0x8000_0001,
            0x8000_0080,
            0x8000_0081,
            0x8000_0180,
            0xFFFF_FF7F,
            0xFFFF_FF80,
            0xFFFF_FF81,
            u32::MAX,
        ];
        let mut rng = crate::rng::Rng::seed(12345);
        for _ in 0..50 {
            u_vals.push((rng.next_u64() >> 17) as u32);
        }
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 33, u_vals.len()] {
            let mut got = vec![0f32; len];
            u32_to_f32_slice(&u_vals[..len], &mut got);
            for (i, (&g, &u)) in got.iter().zip(&u_vals[..len]).enumerate() {
                assert_eq!(g.to_bits(), (u as f32).to_bits(), "u32->f32 len={len} at {i} u={u:#010x}");
            }
        }
        let big_u32: Vec<u32> = (0..PAR_MIN + 29).map(|i| (i as u32).wrapping_mul(2_654_435_761)).collect();
        let CpuStorage::F32(got_uf) = cast(
            &CpuStorage::U32(big_u32.clone()),
            &Layout::contiguous(Shape::new(&[big_u32.len()])),
            DType::F32,
        ) else {
            panic!("expected F32");
        };
        for (&g, &u) in got_uf.iter().zip(&big_u32) {
            assert_eq!(g.to_bits(), (u as f32).to_bits());
        }
    }
}
