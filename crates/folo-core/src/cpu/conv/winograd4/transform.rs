//! Winograd F(4x4, 3x3) input (`Bᵀ d B`) and output (`Aᵀ M A`) transforms for interpolation points
//! `(0, 1/2, -1/2, 1, -1, ∞)`: 8-tile AVX2/FMA + 4-tile SSE/FMA paths bit-exact with the scalar path.

#[cfg(target_arch = "x86_64")]
use crate::backend::UnaryOp;
use crate::cpu::conv::{ChannelEpilogue, ConvGeom};

use super::{SEG, Tiles};

/// 1D input transform `Bᵀ · d` (exact IEEE-754 FMA twin of `bt_1d_avx2`).
#[inline(always)]
fn bt_1d_scalar(d0: f32, d1: f32, d2: f32, d3: f32, d4: f32, d5: f32) -> [f32; 6] {
    let m_even = d2 - d4;
    let m_odd = d1 - d3;
    let t0 = d0 - d2;
    let b0 = 0.25f32.mul_add(t0, -m_even);
    let b1 = 0.5f32.mul_add(m_odd, m_even);
    let b2 = (-0.5f32).mul_add(m_odd, m_even);
    let s_even = (-0.25f32).mul_add(d2, d4);
    let s_odd = (-0.25f32).mul_add(d1, d3);
    let b3 = s_even + s_odd;
    let b4 = s_even - s_odd;
    let b5 = 0.25f32.mul_add(m_odd, d5 - d3);
    [b0, b1, b2, b3, b4, b5]
}

/// 1D output transform `Aᵀ · m` (exact IEEE-754 FMA twin of `at_1d_avx2`).
#[inline(always)]
fn at_1d_scalar(m0: f32, m1: f32, m2: f32, m3: f32, m4: f32, m5: f32) -> [f32; 4] {
    let s12 = m1 + m2;
    let d12 = m1 - m2;
    let s34 = m3 + m4;
    let d34 = m3 - m4;
    let y0 = (m0 + s12) + s34;
    let y1 = 0.5f32.mul_add(d12, d34);
    let y2 = 0.25f32.mul_add(s12, s34);
    let y3 = 0.125f32.mul_add(d12, d34 + m5);
    [y0, y1, y2, y3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bt_1d_avx2(
    d0: std::arch::x86_64::__m256,
    d1: std::arch::x86_64::__m256,
    d2: std::arch::x86_64::__m256,
    d3: std::arch::x86_64::__m256,
    d4: std::arch::x86_64::__m256,
    d5: std::arch::x86_64::__m256,
) -> [std::arch::x86_64::__m256; 6] {
    use std::arch::x86_64::*;
    let c25 = _mm256_set1_ps(0.25);
    let c50 = _mm256_set1_ps(0.5);
    let m_even = _mm256_sub_ps(d2, d4);
    let m_odd = _mm256_sub_ps(d1, d3);
    let t0 = _mm256_sub_ps(d0, d2);
    let b0 = _mm256_fmsub_ps(c25, t0, m_even);
    let b1 = _mm256_fmadd_ps(c50, m_odd, m_even);
    let b2 = _mm256_fnmadd_ps(c50, m_odd, m_even);
    let s_even = _mm256_fnmadd_ps(c25, d2, d4);
    let s_odd = _mm256_fnmadd_ps(c25, d1, d3);
    let b3 = _mm256_add_ps(s_even, s_odd);
    let b4 = _mm256_sub_ps(s_even, s_odd);
    let b5 = _mm256_fmadd_ps(c25, m_odd, _mm256_sub_ps(d5, d3));
    [b0, b1, b2, b3, b4, b5]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn at_1d_avx2(
    m0: std::arch::x86_64::__m256,
    m1: std::arch::x86_64::__m256,
    m2: std::arch::x86_64::__m256,
    m3: std::arch::x86_64::__m256,
    m4: std::arch::x86_64::__m256,
    m5: std::arch::x86_64::__m256,
) -> [std::arch::x86_64::__m256; 4] {
    use std::arch::x86_64::*;
    let s12 = _mm256_add_ps(m1, m2);
    let d12 = _mm256_sub_ps(m1, m2);
    let s34 = _mm256_add_ps(m3, m4);
    let d34 = _mm256_sub_ps(m3, m4);
    let y0 = _mm256_add_ps(_mm256_add_ps(m0, s12), s34);
    let y1 = _mm256_fmadd_ps(_mm256_set1_ps(0.5), d12, d34);
    let y2 = _mm256_fmadd_ps(_mm256_set1_ps(0.25), s12, s34);
    let y3 = _mm256_fmadd_ps(_mm256_set1_ps(0.125), d12, _mm256_add_ps(d34, m5));
    [y0, y1, y2, y3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn bt_1d_sse(
    d0: std::arch::x86_64::__m128,
    d1: std::arch::x86_64::__m128,
    d2: std::arch::x86_64::__m128,
    d3: std::arch::x86_64::__m128,
    d4: std::arch::x86_64::__m128,
    d5: std::arch::x86_64::__m128,
) -> [std::arch::x86_64::__m128; 6] {
    use std::arch::x86_64::*;
    let c25 = _mm_set1_ps(0.25);
    let c50 = _mm_set1_ps(0.5);
    let m_even = _mm_sub_ps(d2, d4);
    let m_odd = _mm_sub_ps(d1, d3);
    let t0 = _mm_sub_ps(d0, d2);
    let b0 = _mm_fmsub_ps(c25, t0, m_even);
    let b1 = _mm_fmadd_ps(c50, m_odd, m_even);
    let b2 = _mm_fnmadd_ps(c50, m_odd, m_even);
    let s_even = _mm_fnmadd_ps(c25, d2, d4);
    let s_odd = _mm_fnmadd_ps(c25, d1, d3);
    let b3 = _mm_add_ps(s_even, s_odd);
    let b4 = _mm_sub_ps(s_even, s_odd);
    let b5 = _mm_fmadd_ps(c25, m_odd, _mm_sub_ps(d5, d3));
    [b0, b1, b2, b3, b4, b5]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn at_1d_sse(
    m0: std::arch::x86_64::__m128,
    m1: std::arch::x86_64::__m128,
    m2: std::arch::x86_64::__m128,
    m3: std::arch::x86_64::__m128,
    m4: std::arch::x86_64::__m128,
    m5: std::arch::x86_64::__m128,
) -> [std::arch::x86_64::__m128; 4] {
    use std::arch::x86_64::*;
    let s12 = _mm_add_ps(m1, m2);
    let d12 = _mm_sub_ps(m1, m2);
    let s34 = _mm_add_ps(m3, m4);
    let d34 = _mm_sub_ps(m3, m4);
    let y0 = _mm_add_ps(_mm_add_ps(m0, s12), s34);
    let y1 = _mm_fmadd_ps(_mm_set1_ps(0.5), d12, d34);
    let y2 = _mm_fmadd_ps(_mm_set1_ps(0.25), s12, s34);
    let y3 = _mm_fmadd_ps(_mm_set1_ps(0.125), d12, _mm_add_ps(d34, m5));
    [y0, y1, y2, y3]
}

/// Fill `seg[..4 * len + 2]` with input row `y` from `x0`, zero-padded outside `[0, h) x [0, w)`.
#[inline(always)]
fn load_row_seg(
    plane: &[f32],
    y: isize,
    (h, w): (isize, isize),
    x0: isize,
    (lo, hi): (isize, isize),
    seg: &mut [f32],
) {
    if (0..h).contains(&y) && lo < hi {
        let row = &plane[(y * w) as usize..((y + 1) * w) as usize];
        let (a, b) = ((lo - x0) as usize, (hi - x0) as usize);
        seg[..a].fill(0.0);
        seg[a..b].copy_from_slice(&row[lo as usize..hi as usize]);
        seg[b..].fill(0.0);
    } else {
        seg.fill(0.0);
    }
}

/// Input transform `V[i * 6 + j][ci][pos] = (Bᵀ d B)[i][j]` for channels `ci0..` of `planes`, tiles `t0..t1`.
///
/// # Safety
/// `v` points to `36 · C · tb` floats and no other task accesses rows `(xi, ci0..ci0 + planes.len())`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn input_transform(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 and FMA were detected; pointer contract forwarded.
        return unsafe { input_transform_avx2(planes, ci0, g, ts, t0, t1, tb, v) };
    }
    // SAFETY: forwarded.
    unsafe { input_transform_scalar(planes, ci0, g, ts, t0, t1, tb, v) }
}

/// Portable scalar input transform (also used by tests to verify bit-exact AVX2 parity).
///
/// # Safety
/// Same contract as [`input_transform`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn input_transform_scalar(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    let (c, h, w) = (g.c, g.h as isize, g.w as isize);
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let tw = g.ow.div_ceil(4);
    let n = t1 - t0;
    let mut tr = [[[0f32; SEG]; 6]; 6];
    let mut seg = [0f32; 4 * SEG + 2];
    for (cl, plane) in planes.iter().enumerate() {
        let ci = ci0 + cl;
        // SAFETY: row `(xi, ci)` belongs exclusively to this task.
        let row = |xi: usize| unsafe { std::slice::from_raw_parts_mut(v.add((xi * c + ci) * tb), tb) };
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            let x0 = 4 * tx0 as isize - pw;
            let (lo, hi) = (x0.max(0), (x0 + 4 * len as isize + 2).min(w));
            let s = &mut seg[..4 * len + 2];
            for (r, tr_r) in tr.iter_mut().enumerate() {
                load_row_seg(plane, 4 * ty as isize + r as isize - ph, (h, w), x0, (lo, hi), s);
                for k in 0..len {
                    let b = 4 * k;
                    let cols = bt_1d_scalar(s[b], s[b + 1], s[b + 2], s[b + 3], s[b + 4], s[b + 5]);
                    for (dst_j, &val) in tr_r.iter_mut().zip(&cols) {
                        dst_j[k] = val;
                    }
                }
            }
            #[allow(clippy::needless_range_loop)]
            for j in 0..6 {
                let [r0, r1, r2, r3, r4, r5] = std::array::from_fn(|i| &tr[i][j][..len]);
                for k in 0..len {
                    let ys = bt_1d_scalar(r0[k], r1[k], r2[k], r3[k], r4[k], r5[k]);
                    for (i, &val) in ys.iter().enumerate() {
                        row(i * 6 + j)[pos + k] = val;
                    }
                }
            }
        }
        if n < tb {
            for xi in 0..36 {
                row(xi)[n..].fill(0.0);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn input_transform_avx2(
    planes: &[&[f32]],
    ci0: usize,
    g: &ConvGeom,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    v: *mut f32,
) {
    use std::arch::x86_64::*;
    let (c, h, w) = (g.c, g.h as isize, g.w as isize);
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let tw = g.ow.div_ceil(4);
    let n = t1 - t0;
    let rot1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    let mut tr = [[[0f32; SEG]; 6]; 6];
    let mut seg = [0f32; 4 * SEG + 2];
    for (cl, plane) in planes.iter().enumerate() {
        let ci = ci0 + cl;
        // SAFETY: row `(xi, ci)` belongs exclusively to this task.
        let row = |xi: usize| unsafe { std::slice::from_raw_parts_mut(v.add((xi * c + ci) * tb), tb) };
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            let x0 = 4 * tx0 as isize - pw;
            let (lo, hi) = (x0.max(0), (x0 + 4 * len as isize + 2).min(w));
            let s = &mut seg[..4 * len + 2];
            for (r, tr_r) in tr.iter_mut().enumerate() {
                load_row_seg(plane, 4 * ty as isize + r as isize - ph, (h, w), x0, (lo, hi), s);
                let mut k = 0;
                // SAFETY: `4 * (k + 8) + 2 <= s.len()` and `k + 8 <= SEG`.
                unsafe {
                    while k + 8 <= len {
                        let sp = s.as_ptr().add(4 * k);
                        let v0 = _mm256_loadu_ps(sp);
                        let v1 = _mm256_loadu_ps(sp.add(8));
                        let v2 = _mm256_loadu_ps(sp.add(16));
                        let v3 = _mm256_loadu_ps(sp.add(24));
                        let p0 = _mm256_permute2f128_ps::<0x20>(v0, v2);
                        let p1 = _mm256_permute2f128_ps::<0x31>(v0, v2);
                        let p2 = _mm256_permute2f128_ps::<0x20>(v1, v3);
                        let p3 = _mm256_permute2f128_ps::<0x31>(v1, v3);
                        let q0 = _mm256_unpacklo_ps(p0, p1);
                        let q1 = _mm256_unpackhi_ps(p0, p1);
                        let q2 = _mm256_unpacklo_ps(p2, p3);
                        let q3 = _mm256_unpackhi_ps(p2, p3);
                        let d0 = _mm256_shuffle_ps::<0x44>(q0, q2);
                        let d1 = _mm256_shuffle_ps::<0xEE>(q0, q2);
                        let d2 = _mm256_shuffle_ps::<0x44>(q1, q3);
                        let d3 = _mm256_shuffle_ps::<0xEE>(q1, q3);
                        // Tile k's d4, d5 are tile (k+1)'s d0, d1; lane 7 loads s[4*k + 32..34].
                        let d4 = _mm256_blend_ps::<0x80>(
                            _mm256_permutevar8x32_ps(d0, rot1),
                            _mm256_broadcast_ss(&*sp.add(32)),
                        );
                        let d5 = _mm256_blend_ps::<0x80>(
                            _mm256_permutevar8x32_ps(d1, rot1),
                            _mm256_broadcast_ss(&*sp.add(33)),
                        );
                        let cols = bt_1d_avx2(d0, d1, d2, d3, d4, d5);
                        for (dst_j, &col_v) in tr_r.iter_mut().zip(&cols) {
                            _mm256_storeu_ps(dst_j.as_mut_ptr().add(k), col_v);
                        }
                        k += 8;
                    }
                    if k + 4 <= len {
                        let sp = s.as_ptr().add(4 * k);
                        let v0 = _mm_loadu_ps(sp);
                        let v1 = _mm_loadu_ps(sp.add(4));
                        let v2 = _mm_loadu_ps(sp.add(8));
                        let v3 = _mm_loadu_ps(sp.add(12));
                        let t0 = _mm_unpacklo_ps(v0, v1);
                        let t1 = _mm_unpacklo_ps(v2, v3);
                        let t2 = _mm_unpackhi_ps(v0, v1);
                        let t3 = _mm_unpackhi_ps(v2, v3);
                        let d0 = _mm_movelh_ps(t0, t1);
                        let d1 = _mm_movehl_ps(t1, t0);
                        let d2 = _mm_movelh_ps(t2, t3);
                        let d3 = _mm_movehl_ps(t3, t2);
                        let tail = _mm_setr_ps(*sp.add(16), *sp.add(17), 0.0, 0.0);
                        let d4 = _mm_shuffle_ps::<0x89>(d0, _mm_shuffle_ps::<0x03>(d0, tail));
                        let d5 = _mm_shuffle_ps::<0x89>(d1, _mm_shuffle_ps::<0x53>(d1, tail));
                        let cols = bt_1d_sse(d0, d1, d2, d3, d4, d5);
                        for (dst_j, &col_v) in tr_r.iter_mut().zip(&cols) {
                            _mm_storeu_ps(dst_j.as_mut_ptr().add(k), col_v);
                        }
                        k += 4;
                    }
                }
                for kk in k..len {
                    let b = 4 * kk;
                    let cols = bt_1d_scalar(s[b], s[b + 1], s[b + 2], s[b + 3], s[b + 4], s[b + 5]);
                    for (dst_j, &val) in tr_r.iter_mut().zip(&cols) {
                        dst_j[kk] = val;
                    }
                }
            }
            for j in 0..6 {
                let dsts: [*mut f32; 6] = std::array::from_fn(|i| row(i * 6 + j)[pos..].as_mut_ptr());
                let mut k = 0;
                // SAFETY: `k + 8 <= len <= SEG`, and `pos + k + 8 <= tb`.
                unsafe {
                    while k + 8 <= len {
                        let d0 = _mm256_loadu_ps(tr[0][j].as_ptr().add(k));
                        let d1 = _mm256_loadu_ps(tr[1][j].as_ptr().add(k));
                        let d2 = _mm256_loadu_ps(tr[2][j].as_ptr().add(k));
                        let d3 = _mm256_loadu_ps(tr[3][j].as_ptr().add(k));
                        let d4 = _mm256_loadu_ps(tr[4][j].as_ptr().add(k));
                        let d5 = _mm256_loadu_ps(tr[5][j].as_ptr().add(k));
                        let ys = bt_1d_avx2(d0, d1, d2, d3, d4, d5);
                        for (&dp, &yv) in dsts.iter().zip(&ys) {
                            _mm256_storeu_ps(dp.add(k), yv);
                        }
                        k += 8;
                    }
                    if k + 4 <= len {
                        let ys = bt_1d_sse(
                            _mm_loadu_ps(tr[0][j].as_ptr().add(k)),
                            _mm_loadu_ps(tr[1][j].as_ptr().add(k)),
                            _mm_loadu_ps(tr[2][j].as_ptr().add(k)),
                            _mm_loadu_ps(tr[3][j].as_ptr().add(k)),
                            _mm_loadu_ps(tr[4][j].as_ptr().add(k)),
                            _mm_loadu_ps(tr[5][j].as_ptr().add(k)),
                        );
                        for (&dp, &yv) in dsts.iter().zip(&ys) {
                            _mm_storeu_ps(dp.add(k), yv);
                        }
                        k += 4;
                    }
                    for kk in k..len {
                        let ys = bt_1d_scalar(
                            tr[0][j][kk],
                            tr[1][j][kk],
                            tr[2][j][kk],
                            tr[3][j][kk],
                            tr[4][j][kk],
                            tr[5][j][kk],
                        );
                        for (&dp, &yv) in dsts.iter().zip(&ys) {
                            *dp.add(kk) = yv;
                        }
                    }
                }
            }
        }
        if n < tb {
            for xi in 0..36 {
                row(xi)[n..].fill(0.0);
            }
        }
    }
}

const _: () = assert!(SEG <= 64);

/// Recompute any 4x4 tile in segment `(ty, tx0..)` marked in `bad` via the exact direct 3x3 sum plus epilogue,
/// recovering cleanly when transform-domain arithmetic (`Bᵀ d B`, `U · V`, or `Aᵀ M A`) overflows `f32::MAX`
/// while the direct conv stays finite. Checked before bias and activation so `ReLU`/`Sigmoid` never mask `±inf`.
///
/// # Safety
/// Every set bit `k` in `bad` must identify a tile `(ty, tx0 + k)` of channel `oc` owned exclusively by this task.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn direct_bad_tiles(
    planes: &[&[f32]],
    wt: &[f32],
    oc: usize,
    ty: usize,
    tx0: usize,
    mut bad: u64,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    out: *mut f32,
) {
    let [ph, pw] = g.p.padding.map(|p| p as isize);
    let (h, w, oh, ow) = (g.h as isize, g.w as isize, g.oh, g.ow);
    let w_oc = &wt[oc * g.c * 9..(oc + 1) * g.c * 9];
    while bad != 0 {
        let k = bad.trailing_zeros() as usize;
        bad &= bad - 1;
        let tx = tx0 + k;
        let ox0 = 4 * tx;
        let cols = 4.min(ow - ox0);
        let rows = 4.min(oh - 4 * ty);
        for r in 0..rows {
            let oy = 4 * ty + r;
            let mut row_out = [0.0f32; 4];
            for (col, slot) in row_out[..cols].iter_mut().enumerate() {
                let ox = ox0 + col;
                let mut sum = 0.0f32;
                for (ci, plane) in planes.iter().enumerate() {
                    let kern = &w_oc[ci * 9..ci * 9 + 9];
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - ph;
                        if !(0..h).contains(&iy) {
                            continue;
                        }
                        let row = &plane[(iy * w) as usize..((iy + 1) * w) as usize];
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - pw;
                            if (0..w).contains(&ix) {
                                sum += row[ix as usize] * kern[ky * 3 + kx];
                            }
                        }
                    }
                }
                *slot = sum;
            }
            if let Some(e) = epi {
                e.apply(oc, &mut row_out[..cols]);
            }
            // SAFETY: `(oc, oy, ox0..ox0 + cols)` belongs exclusively to this task.
            let dst = unsafe { std::slice::from_raw_parts_mut(out.add((oc * oh + oy) * ow + ox0), cols) };
            dst.copy_from_slice(&row_out[..cols]);
        }
    }
}

/// Output transform `Y = Aᵀ M A` of `m` (`36 x m_stride x tb`, reading channels `os`) into `out`, plus epilogue.
///
/// # Safety
/// `out` is the image's `[O, OH, OW]` output and no other task accesses channels `os` in tiles `t0..t1`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn output_transform(
    m: &[f32],
    m_stride: usize,
    os: std::ops::Range<usize>,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    planes: &[&[f32]],
    wt: &[f32],
    out: *mut f32,
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 and FMA were detected; pointer contract forwarded.
        return unsafe { output_transform_avx2(m, m_stride, os, g, epi, ts, t0, t1, tb, planes, wt, out) };
    }
    // SAFETY: forwarded.
    unsafe { output_transform_scalar(m, m_stride, os, g, epi, ts, t0, t1, tb, planes, wt, out) }
}

/// Portable scalar output transform (also used by tests to verify bit-exact AVX2 parity).
///
/// # Safety
/// Same contract as [`output_transform`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn output_transform_scalar(
    m: &[f32],
    m_stride: usize,
    os: std::ops::Range<usize>,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    planes: &[&[f32]],
    wt: &[f32],
    out: *mut f32,
) {
    let (oh, ow) = (g.oh, g.ow);
    let tw = ow.div_ceil(4);
    let oc0 = os.start;
    let mut sm = [[[0f32; SEG]; 6]; 4];
    let mut y = [0f32; 4 * SEG];
    for oc in os {
        let mrow = |xi: usize| &m[(xi * m_stride + (oc - oc0)) * tb..][..tb];
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            for j in 0..6 {
                let r = |i: usize| &mrow(i * 6 + j)[pos..pos + len];
                let (m0, m1, m2, m3, m4, m5) = (r(0), r(1), r(2), r(3), r(4), r(5));
                for k in 0..len {
                    let s = at_1d_scalar(m0[k], m1[k], m2[k], m3[k], m4[k], m5[k]);
                    for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                        sm_i[j][k] = sv;
                    }
                }
            }
            let ox0 = 4 * tx0;
            let ox1 = (ox0 + 4 * len).min(ow);
            let mut bad = 0u64;
            for (r, sm_r) in sm.iter().enumerate() {
                let oy = 4 * ty + r;
                if oy >= oh {
                    break;
                }
                for (k, yy) in y.as_chunks_mut::<4>().0[..len].iter_mut().enumerate() {
                    *yy =
                        at_1d_scalar(sm_r[0][k], sm_r[1][k], sm_r[2][k], sm_r[3][k], sm_r[4][k], sm_r[5][k]);
                    if yy.iter().any(|v| !v.is_finite()) {
                        bad |= 1u64 << k;
                    }
                }
                // SAFETY: row `oy`, columns `ox0..ox1` of channel `oc` are inside `out` and owned by this task.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(out.add((oc * oh + oy) * ow + ox0), ox1 - ox0) };
                dst.copy_from_slice(&y[..ox1 - ox0]);
                if let Some(e) = epi {
                    e.apply(oc, dst);
                }
            }
            if bad != 0 {
                // SAFETY: every set bit in `bad` is in `0..len` of this task's owned segment.
                unsafe { direct_bad_tiles(planes, wt, oc, ty, tx0, bad, g, epi, out) };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform_avx2(
    m: &[f32],
    m_stride: usize,
    os: std::ops::Range<usize>,
    g: &ConvGeom,
    epi: Option<ChannelEpilogue<'_>>,
    ts: Tiles<'_>,
    t0: usize,
    t1: usize,
    tb: usize,
    planes: &[&[f32]],
    wt: &[f32],
    out: *mut f32,
) {
    use std::arch::x86_64::*;
    let (oh, ow) = (g.oh, g.ow);
    let tw = ow.div_ceil(4);
    let oc0 = os.start;
    let mut sm = [[[0f32; SEG]; 6]; 4];
    let mut y = [0f32; 4 * SEG];
    // Fast register epilogue for None, Bias-only, Silu, Relu (other activations use `e.apply` on the finished row).
    let fast_epi = match epi {
        None => true,
        Some(e) => matches!(e.act, None | Some(UnaryOp::Silu) | Some(UnaryOp::Relu)),
    };
    let zero = _mm256_setzero_ps();
    let zero128 = _mm_setzero_ps();
    for oc in os {
        let mrow = |xi: usize| &m[(xi * m_stride + (oc - oc0)) * tb..][..tb];
        let bias_val = epi.and_then(|e| e.bias).map_or(0.0, |b| b[oc]);
        let act = epi.and_then(|e| e.act);
        let add_bias = act == Some(UnaryOp::Silu) || bias_val != 0.0;
        let vb = _mm256_set1_ps(bias_val);
        for (ty, tx0, len, pos) in ts.segments(t0, t1, tw) {
            for j in 0..6 {
                let ptr = |i: usize| mrow(i * 6 + j)[pos..].as_ptr();
                let (p0, p1, p2, p3, p4, p5) = (ptr(0), ptr(1), ptr(2), ptr(3), ptr(4), ptr(5));
                let mut k = 0;
                // SAFETY: `k + 8 <= len <= SEG` and `pos + k + 8 <= tb`.
                unsafe {
                    while k + 8 <= len {
                        let s = at_1d_avx2(
                            _mm256_loadu_ps(p0.add(k)),
                            _mm256_loadu_ps(p1.add(k)),
                            _mm256_loadu_ps(p2.add(k)),
                            _mm256_loadu_ps(p3.add(k)),
                            _mm256_loadu_ps(p4.add(k)),
                            _mm256_loadu_ps(p5.add(k)),
                        );
                        for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                            _mm256_storeu_ps(sm_i[j].as_mut_ptr().add(k), sv);
                        }
                        k += 8;
                    }
                    if k + 4 <= len {
                        let s = at_1d_sse(
                            _mm_loadu_ps(p0.add(k)),
                            _mm_loadu_ps(p1.add(k)),
                            _mm_loadu_ps(p2.add(k)),
                            _mm_loadu_ps(p3.add(k)),
                            _mm_loadu_ps(p4.add(k)),
                            _mm_loadu_ps(p5.add(k)),
                        );
                        for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                            _mm_storeu_ps(sm_i[j].as_mut_ptr().add(k), sv);
                        }
                        k += 4;
                    }
                    for kk in k..len {
                        let s = at_1d_scalar(
                            *p0.add(kk),
                            *p1.add(kk),
                            *p2.add(kk),
                            *p3.add(kk),
                            *p4.add(kk),
                            *p5.add(kk),
                        );
                        for (sm_i, &sv) in sm.iter_mut().zip(&s) {
                            sm_i[j][kk] = sv;
                        }
                    }
                }
            }
            let ox0 = 4 * tx0;
            let ox1 = (ox0 + 4 * len).min(ow);
            let out_len = ox1 - ox0;
            let mut bad = 0u64;
            for (r, sm_r) in sm.iter().enumerate() {
                let oy = 4 * ty + r;
                if oy >= oh {
                    break;
                }
                // SAFETY: `row_dst` owns `out_len` floats of row `oy` in channel `oc`.
                let row_dst = unsafe { out.add((oc * oh + oy) * ow + ox0) };
                let mut k = 0;
                let mut direct_done = 0usize;
                // SAFETY: `k + 8 <= len <= SEG` and `4 * (k + 8) <= 4 * SEG`.
                unsafe {
                    while k + 8 <= len {
                        let [y0, y1, y2, y3] = at_1d_avx2(
                            _mm256_loadu_ps(sm_r[0].as_ptr().add(k)),
                            _mm256_loadu_ps(sm_r[1].as_ptr().add(k)),
                            _mm256_loadu_ps(sm_r[2].as_ptr().add(k)),
                            _mm256_loadu_ps(sm_r[3].as_ptr().add(k)),
                            _mm256_loadu_ps(sm_r[4].as_ptr().add(k)),
                            _mm256_loadu_ps(sm_r[5].as_ptr().add(k)),
                        );
                        // Pre-epilogue finiteness check per tile: `finite * 0 == 0`, `±inf/NaN * 0 == NaN`.
                        let nf = _mm256_fmadd_ps(
                            y3,
                            zero,
                            _mm256_fmadd_ps(y2, zero, _mm256_fmadd_ps(y1, zero, _mm256_mul_ps(y0, zero))),
                        );
                        bad |= (_mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(nf, nf)) as u64) << k;
                        let u0 = _mm256_unpacklo_ps(y0, y1);
                        let u1 = _mm256_unpackhi_ps(y0, y1);
                        let u2 = _mm256_unpacklo_ps(y2, y3);
                        let u3 = _mm256_unpackhi_ps(y2, y3);
                        let r0 = _mm256_shuffle_ps::<0x44>(u0, u2);
                        let r1 = _mm256_shuffle_ps::<0xEE>(u0, u2);
                        let r2 = _mm256_shuffle_ps::<0x44>(u1, u3);
                        let r3 = _mm256_shuffle_ps::<0xEE>(u1, u3);
                        let mut o0 = _mm256_permute2f128_ps::<0x20>(r0, r1);
                        let mut o1 = _mm256_permute2f128_ps::<0x20>(r2, r3);
                        let mut o2 = _mm256_permute2f128_ps::<0x31>(r0, r1);
                        let mut o3 = _mm256_permute2f128_ps::<0x31>(r2, r3);
                        if fast_epi && 4 * k + 32 <= out_len {
                            if add_bias {
                                o0 = _mm256_add_ps(o0, vb);
                                o1 = _mm256_add_ps(o1, vb);
                                o2 = _mm256_add_ps(o2, vb);
                                o3 = _mm256_add_ps(o3, vb);
                            }
                            match act {
                                Some(UnaryOp::Silu) => {
                                    o0 = crate::cpu::elementwise::silu8(o0);
                                    o1 = crate::cpu::elementwise::silu8(o1);
                                    o2 = crate::cpu::elementwise::silu8(o2);
                                    o3 = crate::cpu::elementwise::silu8(o3);
                                }
                                Some(UnaryOp::Relu) => {
                                    o0 = _mm256_max_ps(o0, zero);
                                    o1 = _mm256_max_ps(o1, zero);
                                    o2 = _mm256_max_ps(o2, zero);
                                    o3 = _mm256_max_ps(o3, zero);
                                }
                                _ => {}
                            }
                            let dp = row_dst.add(4 * k);
                            _mm256_storeu_ps(dp, o0);
                            _mm256_storeu_ps(dp.add(8), o1);
                            _mm256_storeu_ps(dp.add(16), o2);
                            _mm256_storeu_ps(dp.add(24), o3);
                            direct_done = 4 * k + 32;
                        } else {
                            let yp = y.as_mut_ptr().add(4 * k);
                            _mm256_storeu_ps(yp, o0);
                            _mm256_storeu_ps(yp.add(8), o1);
                            _mm256_storeu_ps(yp.add(16), o2);
                            _mm256_storeu_ps(yp.add(24), o3);
                        }
                        k += 8;
                    }
                    if k + 4 <= len {
                        let [y0, y1, y2, y3] = at_1d_sse(
                            _mm_loadu_ps(sm_r[0].as_ptr().add(k)),
                            _mm_loadu_ps(sm_r[1].as_ptr().add(k)),
                            _mm_loadu_ps(sm_r[2].as_ptr().add(k)),
                            _mm_loadu_ps(sm_r[3].as_ptr().add(k)),
                            _mm_loadu_ps(sm_r[4].as_ptr().add(k)),
                            _mm_loadu_ps(sm_r[5].as_ptr().add(k)),
                        );
                        let nf = _mm_fmadd_ps(
                            y3,
                            zero128,
                            _mm_fmadd_ps(y2, zero128, _mm_fmadd_ps(y1, zero128, _mm_mul_ps(y0, zero128))),
                        );
                        bad |= (_mm_movemask_ps(_mm_cmpunord_ps(nf, nf)) as u64) << k;
                        let t0 = _mm_unpacklo_ps(y0, y1);
                        let t1 = _mm_unpacklo_ps(y2, y3);
                        let t2 = _mm_unpackhi_ps(y0, y1);
                        let t3 = _mm_unpackhi_ps(y2, y3);
                        let q0 = _mm_movelh_ps(t0, t1);
                        let q1 = _mm_movehl_ps(t1, t0);
                        let q2 = _mm_movelh_ps(t2, t3);
                        let q3 = _mm_movehl_ps(t3, t2);
                        let mut o0 = _mm256_set_m128(q1, q0);
                        let mut o1 = _mm256_set_m128(q3, q2);
                        if fast_epi && 4 * k + 16 <= out_len {
                            if add_bias {
                                o0 = _mm256_add_ps(o0, vb);
                                o1 = _mm256_add_ps(o1, vb);
                            }
                            match act {
                                Some(UnaryOp::Silu) => {
                                    o0 = crate::cpu::elementwise::silu8(o0);
                                    o1 = crate::cpu::elementwise::silu8(o1);
                                }
                                Some(UnaryOp::Relu) => {
                                    o0 = _mm256_max_ps(o0, zero);
                                    o1 = _mm256_max_ps(o1, zero);
                                }
                                _ => {}
                            }
                            let dp = row_dst.add(4 * k);
                            _mm256_storeu_ps(dp, o0);
                            _mm256_storeu_ps(dp.add(8), o1);
                            direct_done = 4 * k + 16;
                        } else {
                            let yp = y.as_mut_ptr().add(4 * k);
                            _mm256_storeu_ps(yp, o0);
                            _mm256_storeu_ps(yp.add(8), o1);
                        }
                        k += 4;
                    }
                }
                for (kk, yy) in y.as_chunks_mut::<4>().0[k..len].iter_mut().enumerate() {
                    let kk = k + kk;
                    *yy = at_1d_scalar(
                        sm_r[0][kk],
                        sm_r[1][kk],
                        sm_r[2][kk],
                        sm_r[3][kk],
                        sm_r[4][kk],
                        sm_r[5][kk],
                    );
                    if yy.iter().any(|v| !v.is_finite()) {
                        bad |= 1u64 << kk;
                    }
                }
                if direct_done < out_len {
                    // SAFETY: remaining `out_len - direct_done` elements of row `oy` belong to this task.
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(row_dst.add(direct_done), out_len - direct_done)
                    };
                    dst.copy_from_slice(&y[direct_done..out_len]);
                    if let Some(e) = epi {
                        e.apply(oc, dst);
                    }
                }
            }
            if bad != 0 {
                // SAFETY: every set bit in `bad` is in `0..len` of this task's owned segment.
                unsafe { direct_bad_tiles(planes, wt, oc, ty, tx0, bad, g, epi, out) };
            }
        }
    }
}
