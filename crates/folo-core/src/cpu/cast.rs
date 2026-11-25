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
fn has_f16c() -> bool {
    std::is_x86_feature_detected!("f16c") && std::is_x86_feature_detected!("avx")
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
        CpuStorage::F16(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, f16_to_f32_slice))
        }
        CpuStorage::U8(v) if dtype == DType::F32 => {
            let data = contiguous(v, layout);
            CpuStorage::F32(map_to_f32(&data, |s, d| {
                for (dst, &x) in d.iter_mut().zip(s) {
                    *dst = x as f32;
                }
            }))
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
}
