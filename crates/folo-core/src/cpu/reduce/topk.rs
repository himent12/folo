//! Top-k selection along the last dimension: `k == 1` zero-alloc scan, bounded-`k` max-heap, and quickselect,
//! with AVX2 (16-wide) and WebAssembly SIMD128 (8-wide) threshold pre-filters that skip non-qualifying blocks
//! while visiting every candidate/NaN lane in exact ascending index order.

use std::cmp::Ordering;

use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use super::has_avx2;
use crate::cpu::{PAR_MIN, alloc_uninit};

/// Canonicalize `-0.0` to `+0.0` so `-0.0` and `+0.0` compare equal and tie-break by index.
#[inline(always)]
fn key(v: f32) -> f32 {
    if v == 0.0 { 0.0 } else { v }
}

/// Exact total ordering matching scalar top-k (values first, ties broken by ascending index).
#[inline(always)]
fn cmp_item(largest: bool, a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    let o = if largest { key(b.0).total_cmp(&key(a.0)) } else { key(a.0).total_cmp(&key(b.0)) };
    o.then(a.1.cmp(&b.1))
}

/// Restore max-heap property under `cmp_item(largest, ..)` (so `heap[0]` is the worst of the kept `k` elements).
#[inline]
fn sift_down(heap: &mut [(f32, usize)], mut i: usize, largest: bool) {
    let n = heap.len();
    loop {
        let left = 2 * i + 1;
        if left >= n {
            break;
        }
        let right = left + 1;
        let mut worst = left;
        if right < n && cmp_item(largest, &heap[right], &heap[left]) == Ordering::Greater {
            worst = right;
        }
        if cmp_item(largest, &heap[worst], &heap[i]) == Ordering::Greater {
            heap.swap(i, worst);
            i = worst;
        } else {
            break;
        }
    }
}

#[inline]
fn scan_top1_scalar(src: &[f32], start: usize, largest: bool, best: &mut (f32, usize)) {
    for (i, &val) in src.iter().enumerate().skip(start) {
        if largest {
            if val <= best.0 {
                continue;
            }
        } else if val >= best.0 {
            continue;
        }
        if cmp_item(largest, &(val, i), best) == Ordering::Less {
            *best = (val, i);
        }
    }
}

#[inline]
fn scan_heap_scalar(src: &[f32], start: usize, largest: bool, buf: &mut [(f32, usize)]) {
    for (i, &val) in src.iter().enumerate().skip(start) {
        if largest {
            if val <= buf[0].0 {
                continue;
            }
        } else if val >= buf[0].0 {
            continue;
        }
        if cmp_item(largest, &(val, i), &buf[0]) == Ordering::Less {
            buf[0] = (val, i);
            sift_down(buf, 0, largest);
        }
    }
}

/// 16-wide AVX2 threshold pre-filter for `k == 1`, bitwise identical to [`scan_top1_scalar`].
///
/// # Safety
/// The CPU supports AVX2 and `!src.is_empty()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_top1_avx2(src: &[f32], largest: bool) -> (f32, usize) {
    use std::arch::x86_64::*;
    let mut best = (src[0], 0usize);
    let mut thresh = _mm256_set1_ps(best.0);
    let d = src.len();
    let mut i = 1usize;
    // SAFETY: `i + 16 <= d` guarantees both 8-float unaligned loads stay in `src`.
    unsafe {
        while i + 16 <= d {
            let p = src.as_ptr().add(i);
            let v0 = _mm256_loadu_ps(p);
            let v1 = _mm256_loadu_ps(p.add(8));
            let r0 = if largest {
                _mm256_cmp_ps::<_CMP_LE_OQ>(v0, thresh)
            } else {
                _mm256_cmp_ps::<_CMP_GE_OQ>(v0, thresh)
            };
            let r1 = if largest {
                _mm256_cmp_ps::<_CMP_LE_OQ>(v1, thresh)
            } else {
                _mm256_cmp_ps::<_CMP_GE_OQ>(v1, thresh)
            };
            if _mm256_movemask_ps(_mm256_and_ps(r0, r1)) == 0xFF {
                i += 16;
                continue;
            }
            let mut k0 = (!_mm256_movemask_ps(r0)) & 0xFF;
            while k0 != 0 {
                let lane = k0.trailing_zeros() as usize;
                k0 &= k0 - 1;
                let idx = i + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &best) == Ordering::Less {
                    best = (val, idx);
                    thresh = _mm256_set1_ps(best.0);
                }
            }
            let mut k1 = (!_mm256_movemask_ps(r1)) & 0xFF;
            while k1 != 0 {
                let lane = k1.trailing_zeros() as usize;
                k1 &= k1 - 1;
                let idx = i + 8 + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &best) == Ordering::Less {
                    best = (val, idx);
                    thresh = _mm256_set1_ps(best.0);
                }
            }
            i += 16;
        }
    }
    scan_top1_scalar(src, i, largest, &mut best);
    best
}

/// 16-wide AVX2 threshold pre-filter for the bounded-`k` max-heap, bitwise identical to [`scan_heap_scalar`].
///
/// # Safety
/// The CPU supports AVX2 and `!buf.is_empty()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_heap_avx2(src: &[f32], start: usize, largest: bool, buf: &mut [(f32, usize)]) {
    use std::arch::x86_64::*;
    let mut thresh = _mm256_set1_ps(buf[0].0);
    let d = src.len();
    let mut i = start;
    // SAFETY: `i + 16 <= d` guarantees both 8-float unaligned loads stay in `src`.
    unsafe {
        while i + 16 <= d {
            let p = src.as_ptr().add(i);
            let v0 = _mm256_loadu_ps(p);
            let v1 = _mm256_loadu_ps(p.add(8));
            let r0 = if largest {
                _mm256_cmp_ps::<_CMP_LE_OQ>(v0, thresh)
            } else {
                _mm256_cmp_ps::<_CMP_GE_OQ>(v0, thresh)
            };
            let r1 = if largest {
                _mm256_cmp_ps::<_CMP_LE_OQ>(v1, thresh)
            } else {
                _mm256_cmp_ps::<_CMP_GE_OQ>(v1, thresh)
            };
            if _mm256_movemask_ps(_mm256_and_ps(r0, r1)) == 0xFF {
                i += 16;
                continue;
            }
            let mut k0 = (!_mm256_movemask_ps(r0)) & 0xFF;
            while k0 != 0 {
                let lane = k0.trailing_zeros() as usize;
                k0 &= k0 - 1;
                let idx = i + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &buf[0]) == Ordering::Less {
                    buf[0] = (val, idx);
                    sift_down(buf, 0, largest);
                    thresh = _mm256_set1_ps(buf[0].0);
                }
            }
            let mut k1 = (!_mm256_movemask_ps(r1)) & 0xFF;
            while k1 != 0 {
                let lane = k1.trailing_zeros() as usize;
                k1 &= k1 - 1;
                let idx = i + 8 + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &buf[0]) == Ordering::Less {
                    buf[0] = (val, idx);
                    sift_down(buf, 0, largest);
                    thresh = _mm256_set1_ps(buf[0].0);
                }
            }
            i += 16;
        }
    }
    scan_heap_scalar(src, i, largest, buf);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn scan_top1_wasm(src: &[f32], largest: bool) -> (f32, usize) {
    use std::arch::wasm32::*;
    let mut best = (src[0], 0usize);
    let mut thresh = f32x4_splat(best.0);
    let d = src.len();
    let mut i = 1usize;
    // SAFETY: `i + 8 <= d` guarantees both 4-float loads stay in `src`.
    unsafe {
        while i + 8 <= d {
            let p = src.as_ptr().add(i);
            let v0 = v128_load(p as *const v128);
            let v1 = v128_load(p.add(4) as *const v128);
            let r0 = if largest { f32x4_le(v0, thresh) } else { f32x4_ge(v0, thresh) };
            let r1 = if largest { f32x4_le(v1, thresh) } else { f32x4_ge(v1, thresh) };
            if i32x4_all_true(v128_and(r0, r1)) {
                i += 8;
                continue;
            }
            let mut k0 = (!i32x4_bitmask(r0)) & 0xF;
            while k0 != 0 {
                let lane = k0.trailing_zeros() as usize;
                k0 &= k0 - 1;
                let idx = i + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &best) == Ordering::Less {
                    best = (val, idx);
                    thresh = f32x4_splat(best.0);
                }
            }
            let mut k1 = (!i32x4_bitmask(r1)) & 0xF;
            while k1 != 0 {
                let lane = k1.trailing_zeros() as usize;
                k1 &= k1 - 1;
                let idx = i + 4 + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &best) == Ordering::Less {
                    best = (val, idx);
                    thresh = f32x4_splat(best.0);
                }
            }
            i += 8;
        }
    }
    scan_top1_scalar(src, i, largest, &mut best);
    best
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn scan_heap_wasm(src: &[f32], start: usize, largest: bool, buf: &mut [(f32, usize)]) {
    use std::arch::wasm32::*;
    let mut thresh = f32x4_splat(buf[0].0);
    let d = src.len();
    let mut i = start;
    // SAFETY: `i + 8 <= d` guarantees both 4-float loads stay in `src`.
    unsafe {
        while i + 8 <= d {
            let p = src.as_ptr().add(i);
            let v0 = v128_load(p as *const v128);
            let v1 = v128_load(p.add(4) as *const v128);
            let r0 = if largest { f32x4_le(v0, thresh) } else { f32x4_ge(v0, thresh) };
            let r1 = if largest { f32x4_le(v1, thresh) } else { f32x4_ge(v1, thresh) };
            if i32x4_all_true(v128_and(r0, r1)) {
                i += 8;
                continue;
            }
            let mut k0 = (!i32x4_bitmask(r0)) & 0xF;
            while k0 != 0 {
                let lane = k0.trailing_zeros() as usize;
                k0 &= k0 - 1;
                let idx = i + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &buf[0]) == Ordering::Less {
                    buf[0] = (val, idx);
                    sift_down(buf, 0, largest);
                    thresh = f32x4_splat(buf[0].0);
                }
            }
            let mut k1 = (!i32x4_bitmask(r1)) & 0xF;
            while k1 != 0 {
                let lane = k1.trailing_zeros() as usize;
                k1 &= k1 - 1;
                let idx = i + 4 + lane;
                let val = *src.get_unchecked(idx);
                if cmp_item(largest, &(val, idx), &buf[0]) == Ordering::Less {
                    buf[0] = (val, idx);
                    sift_down(buf, 0, largest);
                    thresh = f32x4_splat(buf[0].0);
                }
            }
            i += 8;
        }
    }
    scan_heap_scalar(src, i, largest, buf);
}

pub(crate) fn topk(x: &[f32], rows: usize, d: usize, k: usize, largest: bool) -> (Vec<f32>, Vec<i64>) {
    if k == 0 || rows == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut vals = alloc_uninit(rows * k);
    let mut idx = vec![0i64; rows * k];
    #[cfg(target_arch = "x86_64")]
    let avx2 = has_avx2();
    let use_heap = k > 1 && k <= 1024 && k.saturating_mul(4) < d;
    let init_cap = if k == 1 {
        0
    } else if use_heap {
        k
    } else {
        d
    };
    let cmp = |a: &(f32, usize), b: &(f32, usize)| cmp_item(largest, a, b);
    let row = |buf: &mut Vec<(f32, usize)>, r: usize, v: &mut [f32], ix: &mut [i64]| {
        let src = &x[r * d..(r + 1) * d];
        if k == 1 {
            let best = {
                #[cfg(target_arch = "x86_64")]
                if avx2 && d >= 17 {
                    // SAFETY: AVX2 was detected above and `d >= 17`.
                    unsafe { scan_top1_avx2(src, largest) }
                } else {
                    let mut b = (src[0], 0usize);
                    scan_top1_scalar(src, 1, largest, &mut b);
                    b
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
                {
                    if d >= 9 {
                        scan_top1_wasm(src, largest)
                    } else {
                        let mut b = (src[0], 0usize);
                        scan_top1_scalar(src, 1, largest, &mut b);
                        b
                    }
                }
                #[cfg(not(any(
                    target_arch = "x86_64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                )))]
                {
                    let mut b = (src[0], 0usize);
                    scan_top1_scalar(src, 1, largest, &mut b);
                    b
                }
            };
            v[0] = best.0;
            ix[0] = best.1 as i64;
            return;
        }
        buf.clear();
        if use_heap {
            buf.extend(src[..k].iter().copied().zip(0..k));
            for start in (0..k / 2).rev() {
                sift_down(buf, start, largest);
            }
            #[cfg(target_arch = "x86_64")]
            if avx2 && d - k >= 16 {
                // SAFETY: AVX2 was detected above and `buf` has length `k >= 2`.
                unsafe { scan_heap_avx2(src, k, largest, buf) };
            } else {
                scan_heap_scalar(src, k, largest, buf);
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            {
                if d - k >= 8 {
                    scan_heap_wasm(src, k, largest, buf);
                } else {
                    scan_heap_scalar(src, k, largest, buf);
                }
            }
            #[cfg(not(any(
                target_arch = "x86_64",
                all(target_arch = "wasm32", target_feature = "simd128")
            )))]
            {
                scan_heap_scalar(src, k, largest, buf);
            }
            buf.sort_unstable_by(cmp);
        } else {
            buf.extend(src.iter().copied().zip(0..));
            if k < d {
                buf.select_nth_unstable_by(k - 1, cmp);
                buf.truncate(k);
            }
            buf.sort_unstable_by(cmp);
        }
        for (j, &(val, i)) in buf.iter().enumerate() {
            v[j] = val;
            ix[j] = i as i64;
        }
    };
    if rows * d >= PAR_MIN && rows > 1 {
        vals.par_chunks_mut(k)
            .zip(idx.par_chunks_mut(k))
            .enumerate()
            .for_each_init(|| Vec::with_capacity(init_cap), |buf, (r, (v, ix))| row(buf, r, v, ix));
    } else {
        let mut buf = Vec::with_capacity(init_cap);
        for (r, (v, ix)) in vals.chunks_mut(k).zip(idx.chunks_mut(k)).enumerate() {
            row(&mut buf, r, v, ix);
        }
    }
    (vals, idx)
}
