//! Fused L1-tiled AVX2+FMA multi-head attention (`Q^T K` -> scaled `softmax` -> `V @ A^T`) for CPU inference.
//!
//! Eliminates the `O(n²)` global `scores` buffer allocation and collapses all heads and sub-steps into a
//! single parallel pass over disjoint query tiles while staying bit-deterministic across thread counts.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

use super::softmax_row_scaled_scalar;
#[cfg(target_arch = "x86_64")]
use super::{has_avx2_fma, hsum_avx2, softmax_row_scaled_avx2};

/// Compute multi-head scaled dot-product attention `O[b * heads, n, hd]` directly from packed
/// `qkv_s` (`[b, heads, 2*kd + hd, n]`) using L1-resident per-tile `scores` buffers.
///
/// Output layout is `[b * heads, n, hd]` (`o_bnhd.len() >= b * heads * n * hd`), so every query tile
/// `(bi, hi, q0..q1)` owns a disjoint mutable slice and executes in a single parallel dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_attn_o_bnhd(
    qkv_s: &[f32],
    b: usize,
    heads: usize,
    kd: usize,
    hd: usize,
    n: usize,
    scale: f32,
    o_bnhd: &mut [f32],
) {
    let total_heads = b * heads;
    if total_heads == 0 || n == 0 || hd == 0 {
        return;
    }
    let head_stride = (2 * kd + hd) * n;
    let batch_stride = heads * head_stride;
    let tile_q = if n <= 512 { 8 } else { 4 };

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2_fma();

    let run_tile = |scores_buf: &mut Vec<f32>, hi_global: usize, q0: usize, dst_tile: &mut [f32]| {
        let bi = hi_global / heads;
        let hi = hi_global % heads;
        let base = bi * batch_stride + hi * head_stride;
        let q = &qkv_s[base..base + kd * n];
        let k = &qkv_s[base + kd * n..base + 2 * kd * n];
        let v = &qkv_s[base + 2 * kd * n..base + (2 * kd + hd) * n];
        let qb = dst_tile.len() / hd;
        let need = qb * n;
        if scores_buf.len() < need {
            scores_buf.resize(need, 0.0);
        }
        let scores = &mut scores_buf[..need];
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            // SAFETY: `avx2` and `fma` detected above; slice bounds verified by construction.
            unsafe {
                fused_attn_tile_avx2(q, k, v, kd, hd, n, q0, qb, scale, scores, dst_tile);
            }
            return;
        }
        fused_attn_tile_scalar(q, k, v, kd, hd, n, q0, qb, scale, scores, dst_tile);
    };

    let out_slice = &mut o_bnhd[..total_heads * n * hd];
    let work = total_heads.saturating_mul(n).saturating_mul(n).saturating_mul(kd + hd);
    if work >= 500_000 {
        if n.is_multiple_of(tile_q) {
            let tiles_per_head = n / tile_q;
            out_slice.par_chunks_exact_mut(tile_q * hd).enumerate().for_each_init(
                || Vec::with_capacity(tile_q * n),
                |scores_buf, (idx, dst_tile)| {
                    let hi_global = idx / tiles_per_head;
                    let q0 = (idx % tiles_per_head) * tile_q;
                    run_tile(scores_buf, hi_global, q0, dst_tile);
                },
            );
        } else {
            let mut tiles: Vec<(usize, usize, &mut [f32])> =
                Vec::with_capacity(total_heads * n.div_ceil(tile_q));
            for (hi_global, head_out) in out_slice.chunks_exact_mut(n * hd).enumerate() {
                for (ti, dst_tile) in head_out.chunks_mut(tile_q * hd).enumerate() {
                    tiles.push((hi_global, ti * tile_q, dst_tile));
                }
            }
            tiles.into_par_iter().for_each_init(
                || Vec::with_capacity(tile_q * n),
                |scores_buf, (hi_global, q0, dst_tile)| run_tile(scores_buf, hi_global, q0, dst_tile),
            );
        }
    } else {
        let mut scores_buf = Vec::with_capacity(tile_q * n);
        for (hi_global, head_out) in out_slice.chunks_exact_mut(n * hd).enumerate() {
            for (ti, dst_tile) in head_out.chunks_mut(tile_q * hd).enumerate() {
                run_tile(&mut scores_buf, hi_global, ti * tile_q, dst_tile);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_attn_tile_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kd: usize,
    hd: usize,
    n: usize,
    q0: usize,
    qb: usize,
    scale: f32,
    scores: &mut [f32],
    dst_tile: &mut [f32],
) {
    scores.fill(0.0);
    for d in 0..kd {
        let q_row = &q[d * n..(d + 1) * n];
        let k_row = &k[d * n..(d + 1) * n];
        for r in 0..qb {
            let qv = q_row[q0 + r];
            let s_row = &mut scores[r * n..(r + 1) * n];
            for j in 0..n {
                s_row[j] += qv * k_row[j];
            }
        }
    }
    for r in 0..qb {
        softmax_row_scaled_scalar(&mut scores[r * n..(r + 1) * n], scale);
    }
    for r in 0..qb {
        let s_row = &scores[r * n..(r + 1) * n];
        let dst_row = &mut dst_tile[r * hd..(r + 1) * hd];
        for d in 0..hd {
            let v_row = &v[d * n..(d + 1) * n];
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += s_row[j] * v_row[j];
            }
            dst_row[d] = acc;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_attn_tile_avx2(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kd: usize,
    hd: usize,
    n: usize,
    q0: usize,
    qb: usize,
    scale: f32,
    scores: &mut [f32],
    dst_tile: &mut [f32],
) {
    let qp = q.as_ptr();
    let kp = k.as_ptr();
    let vp = v.as_ptr();
    let sp = scores.as_mut_ptr();
    let dp = dst_tile.as_mut_ptr();

    // 1. scores[qb, n] = Q[:, q0..q0+qb]^T @ K[:, 0..n]
    // 4-row × 16-column register-blocked outer-product reduction (8 YMM accumulators, 4× reuse of K).
    let mut r0 = 0usize;
    while r0 + 4 <= qb {
        let (i0, i1, i2, i3) = (q0 + r0, q0 + r0 + 1, q0 + r0 + 2, q0 + r0 + 3);
        let mut j = 0usize;
        // SAFETY: `j + 16 <= n`, `d < kd`, `i0..i3 < n`, `(r0 + 3) * n + j + 16 <= qb * n`.
        unsafe {
            while j + 16 <= n {
                let mut a00 = _mm256_setzero_ps();
                let mut a01 = _mm256_setzero_ps();
                let mut a10 = _mm256_setzero_ps();
                let mut a11 = _mm256_setzero_ps();
                let mut a20 = _mm256_setzero_ps();
                let mut a21 = _mm256_setzero_ps();
                let mut a30 = _mm256_setzero_ps();
                let mut a31 = _mm256_setzero_ps();
                for d in 0..kd {
                    let dn = d * n;
                    let k_ptr = kp.add(dn + j);
                    let k0 = _mm256_loadu_ps(k_ptr);
                    let k1 = _mm256_loadu_ps(k_ptr.add(8));
                    let q_row = qp.add(dn);
                    let qv0 = _mm256_set1_ps(*q_row.add(i0));
                    a00 = _mm256_fmadd_ps(qv0, k0, a00);
                    a01 = _mm256_fmadd_ps(qv0, k1, a01);
                    let qv1 = _mm256_set1_ps(*q_row.add(i1));
                    a10 = _mm256_fmadd_ps(qv1, k0, a10);
                    a11 = _mm256_fmadd_ps(qv1, k1, a11);
                    let qv2 = _mm256_set1_ps(*q_row.add(i2));
                    a20 = _mm256_fmadd_ps(qv2, k0, a20);
                    a21 = _mm256_fmadd_ps(qv2, k1, a21);
                    let qv3 = _mm256_set1_ps(*q_row.add(i3));
                    a30 = _mm256_fmadd_ps(qv3, k0, a30);
                    a31 = _mm256_fmadd_ps(qv3, k1, a31);
                }
                _mm256_storeu_ps(sp.add(r0 * n + j), a00);
                _mm256_storeu_ps(sp.add(r0 * n + j + 8), a01);
                _mm256_storeu_ps(sp.add((r0 + 1) * n + j), a10);
                _mm256_storeu_ps(sp.add((r0 + 1) * n + j + 8), a11);
                _mm256_storeu_ps(sp.add((r0 + 2) * n + j), a20);
                _mm256_storeu_ps(sp.add((r0 + 2) * n + j + 8), a21);
                _mm256_storeu_ps(sp.add((r0 + 3) * n + j), a30);
                _mm256_storeu_ps(sp.add((r0 + 3) * n + j + 8), a31);
                j += 16;
            }
            while j + 8 <= n {
                let mut a0 = _mm256_setzero_ps();
                let mut a1 = _mm256_setzero_ps();
                let mut a2 = _mm256_setzero_ps();
                let mut a3 = _mm256_setzero_ps();
                for d in 0..kd {
                    let dn = d * n;
                    let k0 = _mm256_loadu_ps(kp.add(dn + j));
                    let q_row = qp.add(dn);
                    a0 = _mm256_fmadd_ps(_mm256_set1_ps(*q_row.add(i0)), k0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_set1_ps(*q_row.add(i1)), k0, a1);
                    a2 = _mm256_fmadd_ps(_mm256_set1_ps(*q_row.add(i2)), k0, a2);
                    a3 = _mm256_fmadd_ps(_mm256_set1_ps(*q_row.add(i3)), k0, a3);
                }
                _mm256_storeu_ps(sp.add(r0 * n + j), a0);
                _mm256_storeu_ps(sp.add((r0 + 1) * n + j), a1);
                _mm256_storeu_ps(sp.add((r0 + 2) * n + j), a2);
                _mm256_storeu_ps(sp.add((r0 + 3) * n + j), a3);
                j += 8;
            }
            for jj in j..n {
                let (mut t0, mut t1, mut t2, mut t3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for d in 0..kd {
                    let dn = d * n;
                    let kv = *kp.add(dn + jj);
                    let q_row = qp.add(dn);
                    t0 = (*q_row.add(i0)).mul_add(kv, t0);
                    t1 = (*q_row.add(i1)).mul_add(kv, t1);
                    t2 = (*q_row.add(i2)).mul_add(kv, t2);
                    t3 = (*q_row.add(i3)).mul_add(kv, t3);
                }
                *sp.add(r0 * n + jj) = t0;
                *sp.add((r0 + 1) * n + jj) = t1;
                *sp.add((r0 + 2) * n + jj) = t2;
                *sp.add((r0 + 3) * n + jj) = t3;
            }
        }
        r0 += 4;
    }
    for r in r0..qb {
        let ir = q0 + r;
        let mut j = 0usize;
        // SAFETY: `j + 16 <= n`, `r < qb`, `ir < n`.
        unsafe {
            while j + 16 <= n {
                let mut a0 = _mm256_setzero_ps();
                let mut a1 = _mm256_setzero_ps();
                for d in 0..kd {
                    let dn = d * n;
                    let qv = _mm256_set1_ps(*qp.add(dn + ir));
                    let k_ptr = kp.add(dn + j);
                    a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k_ptr), a0);
                    a1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(k_ptr.add(8)), a1);
                }
                _mm256_storeu_ps(sp.add(r * n + j), a0);
                _mm256_storeu_ps(sp.add(r * n + j + 8), a1);
                j += 16;
            }
            while j + 8 <= n {
                let mut a0 = _mm256_setzero_ps();
                for d in 0..kd {
                    let dn = d * n;
                    let qv = _mm256_set1_ps(*qp.add(dn + ir));
                    a0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(kp.add(dn + j)), a0);
                }
                _mm256_storeu_ps(sp.add(r * n + j), a0);
                j += 8;
            }
            for jj in j..n {
                let mut acc = 0.0f32;
                for d in 0..kd {
                    let dn = d * n;
                    acc = (*qp.add(dn + ir)).mul_add(*kp.add(dn + jj), acc);
                }
                *sp.add(r * n + jj) = acc;
            }
        }
    }

    // 2. In-L1 scaled softmax over the `qb` score rows.
    for r in 0..qb {
        // SAFETY: `avx2` + `fma` enabled on this function.
        unsafe {
            softmax_row_scaled_avx2(&mut scores[r * n..(r + 1) * n], scale);
        }
    }

    // 3. dst_tile[qb, hd] = scores[qb, n] @ V[hd, n]^T
    // 4-row × 2-channel register-blocked dot-product kernel (8 YMM accumulators, 4× reuse of L2-resident V).
    let mut r0 = 0usize;
    while r0 + 4 <= qb {
        // SAFETY: `(r0 + 3) * n + n <= qb * n`, `(d + 1) * n + n <= hd * n`.
        unsafe {
            let sp0 = sp.add(r0 * n);
            let sp1 = sp.add((r0 + 1) * n);
            let sp2 = sp.add((r0 + 2) * n);
            let sp3 = sp.add((r0 + 3) * n);
            let mut d = 0usize;
            while d + 2 <= hd {
                let vp0 = vp.add(d * n);
                let vp1 = vp.add((d + 1) * n);
                let mut c00 = _mm256_setzero_ps();
                let mut c01 = _mm256_setzero_ps();
                let mut c10 = _mm256_setzero_ps();
                let mut c11 = _mm256_setzero_ps();
                let mut c20 = _mm256_setzero_ps();
                let mut c21 = _mm256_setzero_ps();
                let mut c30 = _mm256_setzero_ps();
                let mut c31 = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= n {
                    let v0 = _mm256_loadu_ps(vp0.add(j));
                    let v1 = _mm256_loadu_ps(vp1.add(j));
                    let s0 = _mm256_loadu_ps(sp0.add(j));
                    c00 = _mm256_fmadd_ps(s0, v0, c00);
                    c01 = _mm256_fmadd_ps(s0, v1, c01);
                    let s1 = _mm256_loadu_ps(sp1.add(j));
                    c10 = _mm256_fmadd_ps(s1, v0, c10);
                    c11 = _mm256_fmadd_ps(s1, v1, c11);
                    let s2 = _mm256_loadu_ps(sp2.add(j));
                    c20 = _mm256_fmadd_ps(s2, v0, c20);
                    c21 = _mm256_fmadd_ps(s2, v1, c21);
                    let s3 = _mm256_loadu_ps(sp3.add(j));
                    c30 = _mm256_fmadd_ps(s3, v0, c30);
                    c31 = _mm256_fmadd_ps(s3, v1, c31);
                    j += 8;
                }
                let (mut t00, mut t01) = (hsum_avx2(c00), hsum_avx2(c01));
                let (mut t10, mut t11) = (hsum_avx2(c10), hsum_avx2(c11));
                let (mut t20, mut t21) = (hsum_avx2(c20), hsum_avx2(c21));
                let (mut t30, mut t31) = (hsum_avx2(c30), hsum_avx2(c31));
                for jj in j..n {
                    let (v0, v1) = (*vp0.add(jj), *vp1.add(jj));
                    let s0 = *sp0.add(jj);
                    t00 = s0.mul_add(v0, t00);
                    t01 = s0.mul_add(v1, t01);
                    let s1 = *sp1.add(jj);
                    t10 = s1.mul_add(v0, t10);
                    t11 = s1.mul_add(v1, t11);
                    let s2 = *sp2.add(jj);
                    t20 = s2.mul_add(v0, t20);
                    t21 = s2.mul_add(v1, t21);
                    let s3 = *sp3.add(jj);
                    t30 = s3.mul_add(v0, t30);
                    t31 = s3.mul_add(v1, t31);
                }
                *dp.add(r0 * hd + d) = t00;
                *dp.add(r0 * hd + d + 1) = t01;
                *dp.add((r0 + 1) * hd + d) = t10;
                *dp.add((r0 + 1) * hd + d + 1) = t11;
                *dp.add((r0 + 2) * hd + d) = t20;
                *dp.add((r0 + 2) * hd + d + 1) = t21;
                *dp.add((r0 + 3) * hd + d) = t30;
                *dp.add((r0 + 3) * hd + d + 1) = t31;
                d += 2;
            }
            if d < hd {
                let vp0 = vp.add(d * n);
                let mut c0 = _mm256_setzero_ps();
                let mut c1 = _mm256_setzero_ps();
                let mut c2 = _mm256_setzero_ps();
                let mut c3 = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= n {
                    let v0 = _mm256_loadu_ps(vp0.add(j));
                    c0 = _mm256_fmadd_ps(_mm256_loadu_ps(sp0.add(j)), v0, c0);
                    c1 = _mm256_fmadd_ps(_mm256_loadu_ps(sp1.add(j)), v0, c1);
                    c2 = _mm256_fmadd_ps(_mm256_loadu_ps(sp2.add(j)), v0, c2);
                    c3 = _mm256_fmadd_ps(_mm256_loadu_ps(sp3.add(j)), v0, c3);
                    j += 8;
                }
                let (mut t0, mut t1, mut t2, mut t3) =
                    (hsum_avx2(c0), hsum_avx2(c1), hsum_avx2(c2), hsum_avx2(c3));
                for jj in j..n {
                    let v0 = *vp0.add(jj);
                    t0 = (*sp0.add(jj)).mul_add(v0, t0);
                    t1 = (*sp1.add(jj)).mul_add(v0, t1);
                    t2 = (*sp2.add(jj)).mul_add(v0, t2);
                    t3 = (*sp3.add(jj)).mul_add(v0, t3);
                }
                *dp.add(r0 * hd + d) = t0;
                *dp.add((r0 + 1) * hd + d) = t1;
                *dp.add((r0 + 2) * hd + d) = t2;
                *dp.add((r0 + 3) * hd + d) = t3;
            }
        }
        r0 += 4;
    }
    for r in r0..qb {
        // SAFETY: `r < qb`, `d < hd`, `j + 8 <= n`.
        unsafe {
            let spr = sp.add(r * n);
            let dpr = dp.add(r * hd);
            let mut d = 0usize;
            while d + 4 <= hd {
                let vp0 = vp.add(d * n);
                let vp1 = vp.add((d + 1) * n);
                let vp2 = vp.add((d + 2) * n);
                let vp3 = vp.add((d + 3) * n);
                let mut c0 = _mm256_setzero_ps();
                let mut c1 = _mm256_setzero_ps();
                let mut c2 = _mm256_setzero_ps();
                let mut c3 = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= n {
                    let sr = _mm256_loadu_ps(spr.add(j));
                    c0 = _mm256_fmadd_ps(sr, _mm256_loadu_ps(vp0.add(j)), c0);
                    c1 = _mm256_fmadd_ps(sr, _mm256_loadu_ps(vp1.add(j)), c1);
                    c2 = _mm256_fmadd_ps(sr, _mm256_loadu_ps(vp2.add(j)), c2);
                    c3 = _mm256_fmadd_ps(sr, _mm256_loadu_ps(vp3.add(j)), c3);
                    j += 8;
                }
                let (mut t0, mut t1, mut t2, mut t3) =
                    (hsum_avx2(c0), hsum_avx2(c1), hsum_avx2(c2), hsum_avx2(c3));
                for jj in j..n {
                    let sr = *spr.add(jj);
                    t0 = sr.mul_add(*vp0.add(jj), t0);
                    t1 = sr.mul_add(*vp1.add(jj), t1);
                    t2 = sr.mul_add(*vp2.add(jj), t2);
                    t3 = sr.mul_add(*vp3.add(jj), t3);
                }
                *dpr.add(d) = t0;
                *dpr.add(d + 1) = t1;
                *dpr.add(d + 2) = t2;
                *dpr.add(d + 3) = t3;
                d += 4;
            }
            while d < hd {
                let vp0 = vp.add(d * n);
                let mut c0 = _mm256_setzero_ps();
                let mut j = 0usize;
                while j + 8 <= n {
                    c0 = _mm256_fmadd_ps(_mm256_loadu_ps(spr.add(j)), _mm256_loadu_ps(vp0.add(j)), c0);
                    j += 8;
                }
                let mut t0 = hsum_avx2(c0);
                for jj in j..n {
                    t0 = (*spr.add(jj)).mul_add(*vp0.add(jj), t0);
                }
                *dpr.add(d) = t0;
                d += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_attn_matches_ieee_reference_even_and_odd_shapes() {
        for &(b, heads, kd, hd, n) in &[(1, 2, 32, 64, 400), (2, 3, 13, 19, 37)] {
            let head_stride = (2 * kd + hd) * n;
            let qkv: Vec<f32> = (0..b * heads * head_stride)
                .map(|i| ((i as f32 * 0.17).sin() * 0.8) + ((i as f32 * 0.03).cos() * 0.3))
                .collect();
            let scale = (kd as f64).powf(-0.5) as f32;
            let mut got = vec![0.0f32; b * heads * n * hd];
            fused_attn_o_bnhd(&qkv, b, heads, kd, hd, n, scale, &mut got);

            // Reference in f64 per (bi, hi).
            for bi in 0..b {
                for hi in 0..heads {
                    let base = (bi * heads + hi) * head_stride;
                    let q = &qkv[base..base + kd * n];
                    let k = &qkv[base + kd * n..base + 2 * kd * n];
                    let v = &qkv[base + 2 * kd * n..base + (2 * kd + hd) * n];
                    let out_head = &got[(bi * heads + hi) * n * hd..(bi * heads + hi + 1) * n * hd];
                    for i in 0..n {
                        let mut row = vec![0.0f64; n];
                        let mut mx = f64::NEG_INFINITY;
                        for j in 0..n {
                            let mut s = 0.0f64;
                            for d in 0..kd {
                                s += f64::from(q[d * n + i]) * f64::from(k[d * n + j]);
                            }
                            let sv = s * f64::from(scale);
                            row[j] = sv;
                            mx = mx.max(sv);
                        }
                        let mut sum = 0.0f64;
                        for r in &mut row {
                            *r = (*r - mx).exp();
                            sum += *r;
                        }
                        for d in 0..hd {
                            let mut acc = 0.0f64;
                            for j in 0..n {
                                acc += (row[j] / sum) * f64::from(v[d * n + j]);
                            }
                            let g = out_head[i * hd + d];
                            assert!(
                                (f64::from(g) - acc).abs() < 1e-4,
                                "shape ({b},{heads},{kd},{hd},{n}) bi={bi} hi={hi} i={i} d={d}: got {g} want {acc}"
                            );
                        }
                    }
                }
            }
        }
    }
}
