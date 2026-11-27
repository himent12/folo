//! The direct forward conv for few input channels (the 3-channel stem): per output row the taps are gathered
//! into L1 and every output channel is that many FMAs per pixel, bit for bit the GEMM path's result.

use std::cell::RefCell;

use super::{ChannelEpilogue, ConvGeom, ConvInput, valid_cols};
#[cfg(target_arch = "x86_64")]
use crate::backend::UnaryOp;
use crate::cpu::elementwise::SendPtr;
use crate::cpu::par_for;

thread_local! {
    /// Reusable per-worker scratch for [`small_c_forward`] (eliminates per-task heap allocations + zero-fills).
    static SMALL_C_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a reusable per-thread scratch slice of `len` floats (falls back to a local `Vec` on re-entry).
#[inline]
fn with_small_c_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    let mut loaned = SMALL_C_SCRATCH
        .try_with(|cell| {
            let Ok(mut v) = cell.try_borrow_mut() else { return None };
            if v.len() < len {
                v.resize(len, 0.0);
            }
            Some(std::mem::take(&mut *v))
        })
        .ok()
        .flatten();
    if let Some(mut v) = loaned.take() {
        let r = f(&mut v[..len]);
        let _ = SMALL_C_SCRATCH.try_with(|cell| {
            if let Ok(mut slot) = cell.try_borrow_mut()
                && v.capacity() >= slot.capacity()
            {
                *slot = v;
            }
        });
        return r;
    }
    let mut v = vec![0f32; len];
    f(&mut v[..len])
}

/// Input channels up to which a (non-pointwise, ungrouped) forward conv runs [`small_c_forward`] instead of
/// the shifted-rows GEMM: the stem's 3-channel image input, whose GEMM has a K of only 27 and whose
/// shifted-rows copy is filled by one task per input channel (three).
const SMALL_C: usize = 4;
/// Output columns per register tile of [`small_c_forward`] (four AVX2 vectors).
const SMALL_C_TILE: usize = 32;
/// The deepest K [`small_c_forward`] takes: within one K block of the GEMM (256), so both accumulate every
/// output in the same order and agree bit for bit.
const SMALL_C_MAX_K: usize = 256;
/// Below this many flops [`small_c_forward`] runs on the calling thread (the GEMM's threshold).
const SMALL_C_PAR_FLOPS: usize = 1 << 21;

/// Whether [`super::conv2d_forward_parts`] (overwriting) takes [`small_c_forward`].
pub(super) fn uses_small_c(g: &ConvGeom) -> bool {
    g.p.groups == 1 && g.c <= SMALL_C && !g.is_pointwise() && g.ohw() > 0 && g.kdim() <= SMALL_C_MAX_K
}

/// Direct forward conv for few input channels (see [`SMALL_C`]), overwriting `out`, in parallel over bands of
/// output rows (of all images). Per output row it gathers the `K = c·kh·kw` input rows the taps read (strided
/// and zero-padded, `K x ow` floats, in L1), then every output channel is `K` multiply-adds per pixel from zero
/// in `(ci, ky, kx)` order, in register tiles of [`SMALL_C_TILE`] columns. That is the order and the
/// arithmetic (fused multiply-add with AVX2 and FMA, else a multiply then an add) of the GEMM micro-kernels on
/// one K block, so the result is bit for bit the GEMM path's. `epi` sees each finished output row.
pub(super) fn small_c_forward(
    x: ConvInput<'_>,
    wt: &[f32],
    out: &mut [f32],
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
) {
    let (hw, ohw, k) = (g.h * g.w, g.ohw(), g.kdim());
    assert!(out.len() >= g.n * g.o * ohw, "small_c: output buffer too short");
    assert!(wt.len() >= g.o * k, "small_c: weight buffer too short");
    let owp = g.ow.div_ceil(SMALL_C_TILE) * SMALL_C_TILE;
    let planes: Vec<Vec<&[f32]>> = (0..g.n).map(|img| x.planes(img, hw)).collect();
    #[cfg(target_arch = "x86_64")]
    let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let rows = g.n * g.oh;
    let threads = rayon::current_num_threads().max(1);
    #[cfg(target_arch = "x86_64")]
    let fast_s2k3p1_avx2 = avx2
        && g.p.stride == [2, 2]
        && g.kh == 3
        && g.kw == 3
        && g.p.padding == [1, 1]
        && g.p.dilation == [1, 1]
        && match epi {
            None => true,
            Some(e) => matches!(e.act, None | Some(UnaryOp::Silu)),
        };
    // Target ~2 tasks per thread with >= 4 rows per task so rotating row slots reuse `ky=2 -> ky=0`.
    let band = if rows >= 8 { (rows / (2 * threads)).clamp(4, 16) } else { (rows / threads).max(1) };
    let optr = SendPtr(out.as_mut_ptr());
    let task = |t: usize| {
        let optr = &optr;
        let r0 = t * band;
        let r1 = ((t + 1) * band).min(rows);
        #[cfg(target_arch = "x86_64")]
        if fast_s2k3p1_avx2 {
            // 3 rotating row slots, each holding `c` channels of `(ev[owp16], od[owp16 + 1])`.
            let owp16 = g.ow.next_multiple_of(16);
            let ch_stride = owp16 + (owp16 + 1);
            let slot_len = g.c * ch_stride;
            let silu = epi.is_some_and(|e| e.act == Some(UnaryOp::Silu));
            let bias_slice = epi.and_then(|e| e.bias);
            with_small_c_scratch(3 * slot_len, |buf| {
                let mut ord = [0usize, 1, 2];
                let mut prev_img = usize::MAX;
                let mut prev_oy = usize::MAX;
                let fill_slot = |buf: &mut [f32], slot: usize, img: usize, iy: isize| {
                    let sbase = slot * slot_len;
                    for (ci, plane) in planes[img].iter().enumerate().take(g.c) {
                        let base = sbase + ci * ch_stride;
                        let (ev, od) = buf[base..base + ch_stride].split_at_mut(owp16);
                        od[0] = 0.0;
                        if iy < 0 || iy >= g.h as isize {
                            ev.fill(0.0);
                            od[1..].fill(0.0);
                        } else {
                            let srow = &plane[iy as usize * g.w..(iy as usize + 1) * g.w];
                            // SAFETY: AVX2 and FMA were detected above.
                            unsafe { deinterleave_f32_avx2(srow, &mut ev[..g.ow], &mut od[1..1 + g.ow]) };
                            ev[g.ow..].fill(0.0);
                            od[1 + g.ow..].fill(0.0);
                        }
                    }
                };
                for r in r0..r1 {
                    let (img, oy) = (r / g.oh, r % g.oh);
                    if img == prev_img && oy == prev_oy + 1 {
                        ord.rotate_left(2);
                        fill_slot(buf, ord[1], img, (2 * oy) as isize);
                        fill_slot(buf, ord[2], img, (2 * oy + 1) as isize);
                    } else {
                        ord = [0, 1, 2];
                        fill_slot(buf, 0, img, (2 * oy) as isize - 1);
                        fill_slot(buf, 1, img, (2 * oy) as isize);
                        fill_slot(buf, 2, img, (2 * oy + 1) as isize);
                    }
                    prev_img = img;
                    prev_oy = oy;
                    let mut oc = 0;
                    while oc + 4 <= g.o {
                        let b4 = match bias_slice {
                            Some(b) => [b[oc], b[oc + 1], b[oc + 2], b[oc + 3]],
                            None => [0.0; 4],
                        };
                        let add_bias = silu || b4.iter().any(|&v| v != 0.0);
                        // SAFETY: AVX2+FMA detected; 4 output rows for `oc..oc+4` at `(img, oy)` are disjointly owned.
                        unsafe {
                            let d0 = optr.0.add((img * g.o + oc) * ohw + oy * g.ow);
                            let d1 = optr.0.add((img * g.o + oc + 1) * ohw + oy * g.ow);
                            let d2 = optr.0.add((img * g.o + oc + 2) * ohw + oy * g.ow);
                            let d3 = optr.0.add((img * g.o + oc + 3) * ohw + oy * g.ow);
                            small_c_s2k3p1_row4_avx2(
                                &wt[oc * k..(oc + 4) * k],
                                buf,
                                ord,
                                slot_len,
                                ch_stride,
                                g.c,
                                g.ow,
                                owp16,
                                [d0, d1, d2, d3],
                                add_bias,
                                b4,
                                silu,
                            );
                        }
                        oc += 4;
                    }
                    while oc < g.o {
                        let bv = bias_slice.map_or(0.0, |b| b[oc]);
                        let add_bias = silu || bv != 0.0;
                        // SAFETY: AVX2+FMA detected; output row for `(img, oc, oy)` is disjointly owned.
                        unsafe {
                            let d = optr.0.add((img * g.o + oc) * ohw + oy * g.ow);
                            small_c_s2k3p1_row1_avx2(
                                &wt[oc * k..(oc + 1) * k],
                                buf,
                                ord,
                                slot_len,
                                ch_stride,
                                g.c,
                                g.ow,
                                owp16,
                                d,
                                add_bias,
                                bv,
                                silu,
                            );
                        }
                        oc += 1;
                    }
                }
            });
            return;
        }
        with_small_c_scratch(k * owp + owp, |scratch| {
            let (gat, orow) = scratch.split_at_mut(k * owp);
            for r in r0..r1 {
                let (img, oy) = (r / g.oh, r % g.oh);
                small_c_gather(&planes[img], g, oy, owp, gat);
                for oc in 0..g.o {
                    let w = &wt[oc * k..(oc + 1) * k];
                    // SAFETY: row `oy` of channel `oc` of image `img` is inside `out` (`[n, o, oh, ow]`), and only
                    // this task (which owns output row `r` of every channel) touches it.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(optr.0.add((img * g.o + oc) * ohw + oy * g.ow), g.ow)
                    };
                    #[cfg(target_arch = "x86_64")]
                    if avx2 {
                        let fused_ok = match epi {
                            None => true,
                            Some(e) => matches!(e.act, None | Some(UnaryOp::Silu)),
                        };
                        if fused_ok {
                            let bias_val = epi.and_then(|e| e.bias).map_or(0.0, |b| b[oc]);
                            let silu = epi.is_some_and(|e| e.act == Some(UnaryOp::Silu));
                            let add_bias = silu || bias_val != 0.0;
                            // SAFETY: AVX2 and FMA were detected above.
                            unsafe {
                                small_c_row_avx2_fused(w, gat, owp, dst, orow, add_bias, bias_val, silu);
                            }
                            continue;
                        }
                        // SAFETY: AVX2 and FMA were detected above.
                        unsafe { small_c_row_avx2(w, gat, owp, orow) };
                    } else {
                        small_c_row(w, gat, owp, orow);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    small_c_row(w, gat, owp, orow);
                    dst.copy_from_slice(&orow[..g.ow]);
                    if let Some(e) = epi {
                        e.apply(oc, dst);
                    }
                }
            }
        });
    };
    let tasks = rows.div_ceil(band);
    if 2 * g.n * g.o * ohw * k >= SMALL_C_PAR_FLOPS {
        par_for(tasks, &task);
    } else {
        (0..tasks).for_each(task);
    }
}

/// De-interleave `srow[0..w]` into `even[ox] = srow[2*ox]` and `odd[ox] = srow[2*ox + 1]` (zero outside `w`),
/// for `0 <= ox < ow`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn deinterleave_f32_avx2(srow: &[f32], even: &mut [f32], odd: &mut [f32]) {
    use std::arch::x86_64::*;
    let w = srow.len();
    let ow = even.len();
    assert!(odd.len() >= ow, "deinterleave_f32_avx2: odd buffer too short");
    let full = (w / 16).min(ow / 8);
    let mut ox = 0;
    // SAFETY: `2 * (ox + 8) <= w` and `ox + 8 <= ow`.
    unsafe {
        for _ in 0..full {
            let sp = srow.as_ptr().add(2 * ox);
            let lo = _mm256_loadu_ps(sp);
            let hi = _mm256_loadu_ps(sp.add(8));
            let ev = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0x88>(lo, hi),
            )));
            let od = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(
                _mm256_shuffle_ps::<0xDD>(lo, hi),
            )));
            _mm256_storeu_ps(even.as_mut_ptr().add(ox), ev);
            _mm256_storeu_ps(odd.as_mut_ptr().add(ox), od);
            ox += 8;
        }
    }
    for (off, (e, o)) in even[ox..].iter_mut().zip(&mut odd[ox..]).enumerate() {
        let x = 2 * (ox + off);
        *e = if x < w { srow[x] } else { 0.0 };
        *o = if x + 1 < w { srow[x + 1] } else { 0.0 };
    }
}

/// The `K x owp` taps of output row `oy`: `gat[kk * owp + ox]` is the input element that tap
/// `kk = (ci, ky, kx)` of output pixel `(oy, ox)` reads, zero where it falls in the padding and for `ox >= ow`.
fn small_c_gather(planes: &[&[f32]], g: &ConvGeom, oy: usize, owp: usize, gat: &mut [f32]) {
    let (sh, sw) = (g.p.stride[0] as isize, g.p.stride[1] as isize);
    let (ph, pw) = (g.p.padding[0] as isize, g.p.padding[1] as isize);
    let (dh, dw) = (g.p.dilation[0] as isize, g.p.dilation[1] as isize);
    let fast_s2k3p1 = sw == 2 && g.kw == 3 && pw == 1 && dw == 1;
    #[cfg(target_arch = "x86_64")]
    let avx2 = fast_s2k3p1 && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
    let mut rows = gat.chunks_exact_mut(owp);
    for plane in planes {
        for ky in 0..g.kh {
            let iy = oy as isize * sh - ph + ky as isize * dh;
            let src =
                (0..g.h as isize).contains(&iy).then(|| &plane[iy as usize * g.w..(iy as usize + 1) * g.w]);
            if fast_s2k3p1 {
                let d0 = rows.next().expect("kx=0");
                let d1 = rows.next().expect("kx=1");
                let d2 = rows.next().expect("kx=2");
                let Some(src) = src else {
                    d0.fill(0.0);
                    d1.fill(0.0);
                    d2.fill(0.0);
                    continue;
                };
                let ow = g.ow;
                #[cfg(target_arch = "x86_64")]
                if avx2 {
                    // SAFETY: AVX2 and FMA were detected above.
                    unsafe { deinterleave_f32_avx2(src, &mut d1[..ow], &mut d2[..ow]) };
                } else {
                    for (ox, (e, o)) in d1[..ow].iter_mut().zip(&mut d2[..ow]).enumerate() {
                        let x = 2 * ox;
                        *e = if x < src.len() { src[x] } else { 0.0 };
                        *o = if x + 1 < src.len() { src[x + 1] } else { 0.0 };
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                for (ox, (e, o)) in d1[..ow].iter_mut().zip(&mut d2[..ow]).enumerate() {
                    let x = 2 * ox;
                    *e = if x < src.len() { src[x] } else { 0.0 };
                    *o = if x + 1 < src.len() { src[x + 1] } else { 0.0 };
                }
                d0[0] = 0.0;
                if ow > 1 {
                    d0[1..ow].copy_from_slice(&d2[..ow - 1]);
                }
                d0[ow..].fill(0.0);
                d1[ow..].fill(0.0);
                d2[ow..].fill(0.0);
                continue;
            }
            for kx in 0..g.kw {
                let dst = rows.next().expect("one gathered row per tap");
                let Some(src) = src else {
                    dst.fill(0.0);
                    continue;
                };
                let d = kx as isize * dw - pw;
                let r = valid_cols(g.ow, sw, d, g.w as isize);
                dst[..r.start].fill(0.0);
                dst[r.end..].fill(0.0);
                if r.is_empty() {
                    // The tap sees only padding (it starts right of the row, or ends left of it).
                } else if sw == 1 {
                    let s0 = (r.start as isize + d) as usize;
                    dst[r.clone()].copy_from_slice(&src[s0..s0 + r.len()]);
                } else {
                    for ox in r {
                        dst[ox] = src[(ox as isize * sw + d) as usize];
                    }
                }
            }
        }
    }
}

/// One output row of one channel from the gathered taps: `orow[ox] = Σ_kk w[kk] · gat[kk][ox]`, a multiply
/// then an add from zero in `kk` order (the portable GEMM micro-kernel's arithmetic).
fn small_c_row(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    for (t, o) in orow.as_chunks_mut::<SMALL_C_TILE>().0.iter_mut().enumerate() {
        let mut acc = [0f32; SMALL_C_TILE];
        for (kk, &wv) in w.iter().enumerate() {
            let src = &gat[kk * owp + t * SMALL_C_TILE..][..SMALL_C_TILE];
            for (a, &v) in acc.iter_mut().zip(src) {
                *a += wv * v;
            }
        }
        *o = acc;
    }
}

/// Four output channels of one `s=2, k=3, p=1` row computed simultaneously from the 3 rotating `(ev, od)`
/// slots (16 output columns per register tile, 8 YMM accumulators, 4× lower input load traffic), bit-exact
/// with [`small_c_row_avx2_fused`] and the GEMM path.
///
/// # Safety
/// The CPU supports AVX2 and FMA; `dsts[0..4]` each point to `ow` disjoint writable floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn small_c_s2k3p1_row4_avx2(
    wt4: &[f32],
    buf: &[f32],
    ord: [usize; 3],
    slot_len: usize,
    ch_stride: usize,
    c: usize,
    ow: usize,
    owp16: usize,
    dsts: [*mut f32; 4],
    add_bias: bool,
    bias4: [f32; 4],
    silu: bool,
) {
    use std::arch::x86_64::*;
    let k = c * 9;
    assert!(
        wt4.len() >= 4 * k
            && buf.len() >= 3 * slot_len
            && owp16.is_multiple_of(16)
            && ow <= owp16
            && ch_stride > 2 * owp16
            && slot_len >= c * ch_stride
    );
    let w0 = wt4.as_ptr();
    // SAFETY: `wt4.len() >= 4 * k` asserted above.
    let (w1, w2, w3) = unsafe { (w0.add(k), w0.add(2 * k), w0.add(3 * k)) };
    let vb0 = _mm256_set1_ps(bias4[0]);
    let vb1 = _mm256_set1_ps(bias4[1]);
    let vb2 = _mm256_set1_ps(bias4[2]);
    let vb3 = _mm256_set1_ps(bias4[3]);
    let bptr = buf.as_ptr();
    let mut tmp = [0f32; 16];
    for x0 in (0..owp16).step_by(16) {
        let z = _mm256_setzero_ps();
        let (mut c00, mut c01) = (z, z);
        let (mut c10, mut c11) = (z, z);
        let (mut c20, mut c21) = (z, z);
        let (mut c30, mut c31) = (z, z);
        // SAFETY: `buf`, `wt4`, and `ord[0..3]` extents verified above; `x0 + 16 <= owp16` and `od` has `owp16 + 1`.
        unsafe {
            let mut kk = 0usize;
            for ci in 0..c {
                for &sl in &ord {
                    let base = bptr.add(sl * slot_len + ci * ch_stride);
                    let ev = base.add(x0);
                    let od = base.add(owp16 + x0);
                    macro_rules! tap {
                        ($src:expr, $koff:expr) => {
                            let g0 = _mm256_loadu_ps($src);
                            let g1 = _mm256_loadu_ps($src.add(8));
                            let a0 = _mm256_broadcast_ss(&*w0.add($koff));
                            c00 = _mm256_fmadd_ps(a0, g0, c00);
                            c01 = _mm256_fmadd_ps(a0, g1, c01);
                            let a1 = _mm256_broadcast_ss(&*w1.add($koff));
                            c10 = _mm256_fmadd_ps(a1, g0, c10);
                            c11 = _mm256_fmadd_ps(a1, g1, c11);
                            let a2 = _mm256_broadcast_ss(&*w2.add($koff));
                            c20 = _mm256_fmadd_ps(a2, g0, c20);
                            c21 = _mm256_fmadd_ps(a2, g1, c21);
                            let a3 = _mm256_broadcast_ss(&*w3.add($koff));
                            c30 = _mm256_fmadd_ps(a3, g0, c30);
                            c31 = _mm256_fmadd_ps(a3, g1, c31);
                        };
                    }
                    tap!(od, kk);
                    tap!(ev, kk + 1);
                    tap!(od.add(1), kk + 2);
                    kk += 3;
                }
            }
            if add_bias {
                c00 = _mm256_add_ps(c00, vb0);
                c01 = _mm256_add_ps(c01, vb0);
                c10 = _mm256_add_ps(c10, vb1);
                c11 = _mm256_add_ps(c11, vb1);
                c20 = _mm256_add_ps(c20, vb2);
                c21 = _mm256_add_ps(c21, vb2);
                c30 = _mm256_add_ps(c30, vb3);
                c31 = _mm256_add_ps(c31, vb3);
            }
            if silu {
                c00 = crate::cpu::elementwise::silu8(c00);
                c01 = crate::cpu::elementwise::silu8(c01);
                c10 = crate::cpu::elementwise::silu8(c10);
                c11 = crate::cpu::elementwise::silu8(c11);
                c20 = crate::cpu::elementwise::silu8(c20);
                c21 = crate::cpu::elementwise::silu8(c21);
                c30 = crate::cpu::elementwise::silu8(c30);
                c31 = crate::cpu::elementwise::silu8(c31);
            }
            let pairs = [(dsts[0], c00, c01), (dsts[1], c10, c11), (dsts[2], c20, c21), (dsts[3], c30, c31)];
            if x0 + 16 <= ow {
                for (d, v0, v1) in pairs {
                    let o = d.add(x0);
                    _mm256_storeu_ps(o, v0);
                    _mm256_storeu_ps(o.add(8), v1);
                }
            } else {
                let take = ow - x0;
                for (d, v0, v1) in pairs {
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v0);
                    _mm256_storeu_ps(tmp.as_mut_ptr().add(8), v1);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), d.add(x0), take);
                }
            }
        }
    }
}

/// Remainder single output channel of [`small_c_s2k3p1_row4_avx2`] when `o` is not a multiple of 4.
///
/// # Safety
/// The CPU supports AVX2 and FMA; `dst` points to `ow` disjoint writable floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn small_c_s2k3p1_row1_avx2(
    w: &[f32],
    buf: &[f32],
    ord: [usize; 3],
    slot_len: usize,
    ch_stride: usize,
    c: usize,
    ow: usize,
    owp16: usize,
    dst: *mut f32,
    add_bias: bool,
    bias_val: f32,
    silu: bool,
) {
    use std::arch::x86_64::*;
    let k = c * 9;
    assert!(
        w.len() >= k
            && buf.len() >= 3 * slot_len
            && owp16.is_multiple_of(16)
            && ow <= owp16
            && ch_stride > 2 * owp16
            && slot_len >= c * ch_stride
    );
    let wp = w.as_ptr();
    let vb = _mm256_set1_ps(bias_val);
    let bptr = buf.as_ptr();
    let mut tmp = [0f32; 16];
    for x0 in (0..owp16).step_by(16) {
        let (mut c0, mut c1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        // SAFETY: extents asserted above.
        unsafe {
            let mut kk = 0usize;
            for ci in 0..c {
                for &sl in &ord {
                    let base = bptr.add(sl * slot_len + ci * ch_stride);
                    let ev = base.add(x0);
                    let od = base.add(owp16 + x0);
                    let a0 = _mm256_broadcast_ss(&*wp.add(kk));
                    c0 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(od), c0);
                    c1 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(od.add(8)), c1);
                    let a1 = _mm256_broadcast_ss(&*wp.add(kk + 1));
                    c0 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(ev), c0);
                    c1 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(ev.add(8)), c1);
                    let a2 = _mm256_broadcast_ss(&*wp.add(kk + 2));
                    c0 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(od.add(1)), c0);
                    c1 = _mm256_fmadd_ps(a2, _mm256_loadu_ps(od.add(9)), c1);
                    kk += 3;
                }
            }
            if add_bias {
                c0 = _mm256_add_ps(c0, vb);
                c1 = _mm256_add_ps(c1, vb);
            }
            if silu {
                c0 = crate::cpu::elementwise::silu8(c0);
                c1 = crate::cpu::elementwise::silu8(c1);
            }
            if x0 + 16 <= ow {
                _mm256_storeu_ps(dst.add(x0), c0);
                _mm256_storeu_ps(dst.add(x0 + 8), c1);
            } else {
                _mm256_storeu_ps(tmp.as_mut_ptr(), c0);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), c1);
                std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst.add(x0), ow - x0);
            }
        }
    }
}

/// [`small_c_row_avx2`] with bias + optional SiLU fused in registers and stored directly to `dst` (bit-exact
/// with [`crate::cpu::gemm`] + [`crate::cpu::elementwise::silu_bias_inplace`]).
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn small_c_row_avx2_fused(
    w: &[f32],
    gat: &[f32],
    owp: usize,
    dst: &mut [f32],
    orow: &mut [f32],
    add_bias: bool,
    bias_val: f32,
    silu: bool,
) {
    use std::arch::x86_64::*;
    assert!(
        gat.len() >= w.len() * owp
            && orow.len() == owp
            && dst.len() <= owp
            && owp.is_multiple_of(SMALL_C_TILE)
    );
    let ow = dst.len();
    let vb = _mm256_set1_ps(bias_val);
    for x0 in (0..owp).step_by(SMALL_C_TILE) {
        let (mut c0, mut c1, mut c2, mut c3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        for (kk, wv) in w.iter().enumerate() {
            // SAFETY: row `kk` of `gat` holds `owp` floats and `x0 + SMALL_C_TILE <= owp`.
            unsafe {
                let p = gat.as_ptr().add(kk * owp + x0);
                let a = _mm256_broadcast_ss(wv);
                c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p), c0);
                c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(8)), c1);
                c2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(16)), c2);
                c3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(24)), c3);
            }
        }
        if add_bias {
            c0 = _mm256_add_ps(c0, vb);
            c1 = _mm256_add_ps(c1, vb);
            c2 = _mm256_add_ps(c2, vb);
            c3 = _mm256_add_ps(c3, vb);
        }
        if silu {
            c0 = crate::cpu::elementwise::silu8(c0);
            c1 = crate::cpu::elementwise::silu8(c1);
            c2 = crate::cpu::elementwise::silu8(c2);
            c3 = crate::cpu::elementwise::silu8(c3);
        }
        if x0 + SMALL_C_TILE <= ow {
            // SAFETY: `x0 + SMALL_C_TILE <= dst.len()`.
            unsafe {
                let o = dst.as_mut_ptr().add(x0);
                _mm256_storeu_ps(o, c0);
                _mm256_storeu_ps(o.add(8), c1);
                _mm256_storeu_ps(o.add(16), c2);
                _mm256_storeu_ps(o.add(24), c3);
            }
        } else {
            // SAFETY: `orow` holds `owp >= x0 + SMALL_C_TILE` floats.
            unsafe {
                let o = orow.as_mut_ptr();
                _mm256_storeu_ps(o, c0);
                _mm256_storeu_ps(o.add(8), c1);
                _mm256_storeu_ps(o.add(16), c2);
                _mm256_storeu_ps(o.add(24), c3);
            }
            dst[x0..ow].copy_from_slice(&orow[..ow - x0]);
        }
    }
}

/// [`small_c_row`] with AVX2 fused multiply-adds (the AVX2 GEMM micro-kernels' arithmetic): four 8-wide
/// accumulators per tile.
///
/// # Safety
/// The CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn small_c_row_avx2(w: &[f32], gat: &[f32], owp: usize, orow: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(gat.len() >= w.len() * owp && orow.len() == owp && owp.is_multiple_of(SMALL_C_TILE));
    for x0 in (0..owp).step_by(SMALL_C_TILE) {
        let (mut c0, mut c1, mut c2, mut c3) =
            (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        for (kk, wv) in w.iter().enumerate() {
            // SAFETY: row `kk` of `gat` holds `owp` floats and `x0 + SMALL_C_TILE <= owp` (asserted above).
            unsafe {
                let p = gat.as_ptr().add(kk * owp + x0);
                let a = _mm256_broadcast_ss(wv);
                c0 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p), c0);
                c1 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(8)), c1);
                c2 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(16)), c2);
                c3 = _mm256_fmadd_ps(a, _mm256_loadu_ps(p.add(24)), c3);
            }
        }
        // SAFETY: `orow` holds `owp >= x0 + SMALL_C_TILE` floats.
        unsafe {
            let o = orow.as_mut_ptr().add(x0);
            _mm256_storeu_ps(o, c0);
            _mm256_storeu_ps(o.add(8), c1);
            _mm256_storeu_ps(o.add(16), c2);
            _mm256_storeu_ps(o.add(24), c3);
        }
    }
}

#[cfg(test)]
mod small_c_tests {
    use super::*;
    use crate::backend::{Conv2dParams, UnaryOp};
    use crate::cpu::conv::shifted::ShiftedRows;
    use crate::cpu::conv::{ConvPart, conv2d_forward};
    use crate::cpu::gemm::{BiasAct, RowPtrs, Strided, gemm_rows};

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    fn geom(n: usize, c: usize, hw: [usize; 2], o: usize, k: [usize; 2], p: Conv2dParams) -> ConvGeom {
        let out =
            |i: usize, k: usize| (hw[i] + 2 * p.padding[i] - p.dilation[i] * (k - 1) - 1) / p.stride[i] + 1;
        let (oh, ow) = (out(0, k[0]), out(1, k[1]));
        ConvGeom { n, c, h: hw[0], w: hw[1], o, kh: k[0], kw: k[1], oh, ow, p }
    }

    /// The shifted-rows GEMM path [`small_c_forward`] replaces (what [`conv2d_forward_parts`] ran before).
    fn gemm_path(x: &[f32], wt: &[f32], out: &mut [f32], g: &ConvGeom, epi: Option<ChannelEpilogue<'_>>) {
        let (kdim, ohw, hw) = (g.kdim(), g.ohw(), g.h * g.w);
        let part = [ConvPart::contiguous(x, g.c, hw)];
        let x = ConvInput { parts: &part };
        for (img, out_img) in out.chunks_mut(g.o * ohw).enumerate() {
            let planes = x.planes(img, hw);
            let a = Strided::lhs(wt, 0, kdim, 1);
            let sr = ShiftedRows::new(&planes, g, true);
            let b = RowPtrs { rows: &sr.rows, n: sr.n };
            gemm_rows(g.o, kdim, &a, &b, out_img, sr.map, rayon::current_num_threads(), epi);
        }
    }

    #[test]
    fn small_c_matches_the_gemm_path_bit_for_bit() {
        let p = |s: [usize; 2], pad: [usize; 2], d: [usize; 2]| Conv2dParams {
            stride: s,
            padding: pad,
            dilation: d,
            groups: 1,
        };
        // (n, c, [h, w], o, [kh, kw], params): the folo stem at 640 and 320, then odd shapes, strides,
        // paddings, dilations and channel counts 1..=4.
        let cases = [
            (1, 3, [640, 640], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (2, 3, [320, 384], 16, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (2, 3, [33, 35], 7, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (1, 3, [17, 19], 1, [3, 3], p([2, 2], [1, 1], [1, 1])),
            (1, 1, [37, 29], 5, [3, 3], p([1, 1], [1, 1], [1, 1])),
            (3, 2, [19, 23], 7, [5, 3], p([2, 1], [2, 0], [1, 2])),
            (1, 4, [33, 65], 9, [3, 3], p([3, 2], [0, 1], [1, 1])),
            (2, 3, [8, 8], 4, [3, 3], p([1, 1], [3, 3], [1, 1])),
            (1, 3, [70, 70], 33, [7, 7], p([2, 2], [3, 3], [1, 1])),
            (1, 3, [5, 200], 2, [1, 3], p([1, 4], [0, 1], [1, 1])),
            // Taps that see only padding on the left or the right of a row (and above / below it).
            (1, 2, [1, 1], 2, [5, 5], p([1, 1], [2, 2], [1, 1])),
            (2, 1, [4, 3], 3, [5, 7], p([1, 1], [2, 3], [1, 1])),
            (1, 3, [3, 2], 2, [3, 5], p([2, 2], [1, 3], [1, 1])),
        ];
        for (i, (n, c, hw, o, k, prm)) in cases.into_iter().enumerate() {
            let g = geom(n, c, hw, o, k, prm);
            assert!(uses_small_c(&g), "case {i}");
            let x = data(n * c * hw[0] * hw[1], 1 + i as u32);
            let wt = data(o * g.kdim(), 100 + i as u32);
            // Channel 4's bias is 0.0.
            let bias: Vec<f32> = (0..o).map(|c| c as f32 * 0.25 - 1.0).collect();
            let b = Some(&bias[..]);
            // The GEMM applies the identity and SiLU in its micro-kernels, ReLU as a pass over finished rows.
            let epis = [
                None,
                Some(BiasAct { bias: None, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: None }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Silu) }),
                Some(BiasAct { bias: b, act: Some(UnaryOp::Relu) }),
            ];
            for epi in epis {
                let mut want = vec![f32::NAN; n * o * g.ohw()];
                gemm_path(&x, &wt, &mut want, &g, epi);
                let mut got = vec![f32::NAN; want.len()];
                conv2d_forward(&x, &wt, &mut got, &g, false, epi);
                let same = got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "case {i} ({g:?}): not bit-identical with the GEMM path");
            }
        }
    }
}
