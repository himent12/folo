//! WebAssembly SIMD128 (`v128`, 4-lane `f32x4`) `softmax` / `log_softmax` forward and backward kernels,
//! bitwise identical to their scalar and AVX2 counterparts on all non-NaN inputs (including `±0.0`,
//! subnormals, and `±∞`) while preserving NaN propagation semantics (WebAssembly arithmetic leaves payload
//! selection on `NaN ∘ NaN` implementation-defined).

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::COL_TILE;
use crate::cpu::elementwise::{exp_scalar, exp4_wasm, fast_exp, ln_scalar, ln4_wasm};

const LOG2E: f32 = std::f32::consts::LOG2_E;
const FAST_EXP_MAGIC: f32 = 12_582_912.0;

/// 4-lane Wasm SIMD128 twin of [`fast_exp`] (non-FMA, bitwise identical for every `f32` including NaN payloads).
#[inline(always)]
pub(super) fn fast_exp4(x: v128) -> v128 {
    let xc = f32x4_pmin(f32x4_splat(89.0), f32x4_pmax(f32x4_splat(-104.0), x));
    let magic = f32x4_splat(FAST_EXP_MAGIC);
    let n = f32x4_sub(f32x4_add(f32x4_mul(xc, f32x4_splat(LOG2E)), magic), magic);
    let r = f32x4_sub(
        f32x4_sub(xc, f32x4_mul(n, f32x4_splat(0.693_359_4))),
        f32x4_mul(n, f32x4_splat(-2.121_944_4e-4)),
    );
    let mut p = f32x4_splat(1.987_569_1e-4);
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.398_199_9e-3));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(8.333_452e-3));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(4.166_579_6e-2));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(1.666_666_5e-1));
    p = f32x4_add(f32x4_mul(p, r), f32x4_splat(0.5));
    let y = f32x4_add(f32x4_add(f32x4_mul(f32x4_mul(p, r), r), r), f32x4_splat(1.0));
    let ni = i32x4_trunc_sat_f32x4(n);
    let n1 = i32x4_shr(ni, 1);
    let n2 = i32x4_sub(ni, n1);
    let bias = i32x4_splat(127);
    let s1 = i32x4_shl(i32x4_add(n1, bias), 23);
    let s2 = i32x4_shl(i32x4_add(n2, bias), 23);
    let v = f32x4_mul(f32x4_mul(y, s1), s2);
    v128_bitselect(x, v, f32x4_ne(x, x))
}

#[inline]
pub(super) fn hsum16_wasm(a0: v128, a1: v128, a2: v128, a3: v128, rest_vals: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    // SAFETY: `acc` holds 16 floats.
    unsafe {
        v128_store(acc.as_mut_ptr() as *mut v128, a0);
        v128_store(acc.as_mut_ptr().add(4) as *mut v128, a1);
        v128_store(acc.as_mut_ptr().add(8) as *mut v128, a2);
        v128_store(acc.as_mut_ptr().add(12) as *mut v128, a3);
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
pub(super) fn row_max_and_nan_wasm(src: &[f32]) -> (f32, bool) {
    let mut m0 = f32x4_splat(f32::NEG_INFINITY);
    let mut m1 = f32x4_splat(f32::NEG_INFINITY);
    let mut nan0 = i32x4_splat(0);
    let mut nan1 = i32x4_splat(0);
    let (chunks, rest) = src.as_chunks::<8>();
    for c in chunks {
        // SAFETY: `c` is 8 floats.
        unsafe {
            let v0 = v128_load(c.as_ptr() as *const v128);
            let v1 = v128_load(c.as_ptr().add(4) as *const v128);
            // `f32x4_pmax(m, v)` returns `m` when `v` is NaN, matching `m.max(v)`.
            m0 = f32x4_pmax(m0, v0);
            m1 = f32x4_pmax(m1, v1);
            nan0 = v128_or(nan0, f32x4_ne(v0, v0));
            nan1 = v128_or(nan1, f32x4_ne(v1, v1));
        }
    }
    let (c4, tail) = rest.as_chunks::<4>();
    for c in c4 {
        // SAFETY: `c` is 4 floats.
        unsafe {
            let v = v128_load(c.as_ptr() as *const v128);
            m0 = f32x4_pmax(m0, v);
            nan0 = v128_or(nan0, f32x4_ne(v, v));
        }
    }
    let m_vec = f32x4_pmax(m0, m1);
    let mut lanes = [0f32; 4];
    // SAFETY: `lanes` holds 4 floats.
    unsafe { v128_store(lanes.as_mut_ptr() as *mut v128, m_vec) };
    let mut m = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut any_nan = v128_any_true(v128_or(nan0, nan1));
    for &v in tail {
        if v.is_nan() {
            any_nan = true;
        } else {
            m = m.max(v);
        }
    }
    (m, any_nan)
}

/// 16-lane Wasm SIMD128 `sum(fast_exp(src[i] - m))`.
pub(super) fn sum_exp_shifted_wasm(src: &[f32], m: f32) -> f32 {
    let mv = f32x4_splat(m);
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let (chunks, rest) = src.as_chunks::<16>();
    for c in chunks {
        // SAFETY: `c` is 16 floats.
        unsafe {
            let p = c.as_ptr();
            a0 = f32x4_add(a0, fast_exp4(f32x4_sub(v128_load(p as *const v128), mv)));
            a1 = f32x4_add(a1, fast_exp4(f32x4_sub(v128_load(p.add(4) as *const v128), mv)));
            a2 = f32x4_add(a2, fast_exp4(f32x4_sub(v128_load(p.add(8) as *const v128), mv)));
            a3 = f32x4_add(a3, fast_exp4(f32x4_sub(v128_load(p.add(12) as *const v128), mv)));
        }
    }
    let mut rbuf = [0f32; 16];
    for (i, &v) in rest.iter().enumerate() {
        rbuf[i] = fast_exp(v - m);
    }
    hsum16_wasm(a0, a1, a2, a3, &rbuf[..rest.len()])
}

/// Write `dst[i] = fast_exp(src[i] - m)` and return exact 16-lane `sum_slice(dst)`.
pub(super) fn exp_shifted_and_sum_wasm(src: &[f32], dst: &mut [f32], m: f32) -> f32 {
    let mv = f32x4_splat(m);
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    let (sc, sr) = src.as_chunks::<16>();
    let (dc, dr) = dst.as_chunks_mut::<16>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 16 floats.
        unsafe {
            let sp = s.as_ptr();
            let dp = d.as_mut_ptr();
            let e0 = fast_exp4(f32x4_sub(v128_load(sp as *const v128), mv));
            let e1 = fast_exp4(f32x4_sub(v128_load(sp.add(4) as *const v128), mv));
            let e2 = fast_exp4(f32x4_sub(v128_load(sp.add(8) as *const v128), mv));
            let e3 = fast_exp4(f32x4_sub(v128_load(sp.add(12) as *const v128), mv));
            v128_store(dp as *mut v128, e0);
            v128_store(dp.add(4) as *mut v128, e1);
            v128_store(dp.add(8) as *mut v128, e2);
            v128_store(dp.add(12) as *mut v128, e3);
            a0 = f32x4_add(a0, e0);
            a1 = f32x4_add(a1, e1);
            a2 = f32x4_add(a2, e2);
            a3 = f32x4_add(a3, e3);
        }
    }
    for (d, &s) in dr.iter_mut().zip(sr) {
        *d = fast_exp(s - m);
    }
    hsum16_wasm(a0, a1, a2, a3, dr)
}

/// Write `dst[i] = (src[i] - m) - ls`.
pub(super) fn write_log_softmax_row_wasm(src: &[f32], dst: &mut [f32], m: f32, ls: f32) {
    let mv = f32x4_splat(m);
    let lsv = f32x4_splat(ls);
    let (sc, sr) = src.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 8 floats.
        unsafe {
            let sp = s.as_ptr();
            let dp = d.as_mut_ptr();
            v128_store(dp as *mut v128, f32x4_sub(f32x4_sub(v128_load(sp as *const v128), mv), lsv));
            v128_store(
                dp.add(4) as *mut v128,
                f32x4_sub(f32x4_sub(v128_load(sp.add(4) as *const v128), mv), lsv),
            );
        }
    }
    for (y, &v) in dr.iter_mut().zip(sr) {
        *y = (v - m) - ls;
    }
}

/// Scale `dst[i] *= inv` in place.
pub(super) fn scale_inplace_wasm(dst: &mut [f32], inv: f32) {
    let iv = f32x4_splat(inv);
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for d in dc {
        // SAFETY: `d` is 8 floats.
        unsafe {
            let p = d.as_mut_ptr();
            v128_store(p as *mut v128, f32x4_mul(v128_load(p as *const v128), iv));
            v128_store(p.add(4) as *mut v128, f32x4_mul(v128_load(p.add(4) as *const v128), iv));
        }
    }
    for y in dr {
        *y *= inv;
    }
}

/// Write `dst[i] = fast_exp(src[i] - m) * inv`.
pub(super) fn write_exp_scaled_row_wasm(src: &[f32], dst: &mut [f32], m: f32, inv: f32) {
    let mv = f32x4_splat(m);
    let iv = f32x4_splat(inv);
    let (sc, sr) = src.as_chunks::<8>();
    let (dc, dr) = dst.as_chunks_mut::<8>();
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        // SAFETY: `s` and `d` are 8 floats.
        unsafe {
            let sp = s.as_ptr();
            let dp = d.as_mut_ptr();
            v128_store(
                dp as *mut v128,
                f32x4_mul(fast_exp4(f32x4_sub(v128_load(sp as *const v128), mv)), iv),
            );
            v128_store(
                dp.add(4) as *mut v128,
                f32x4_mul(fast_exp4(f32x4_sub(v128_load(sp.add(4) as *const v128), mv)), iv),
            );
        }
    }
    for (y, &v) in dr.iter_mut().zip(sr) {
        *y = fast_exp(v - m) * inv;
    }
}

/// Column-tiled Wasm SIMD128 `softmax` / `log_softmax` over columns `c0..c0 + tw` (`tw <= COL_TILE`).
///
/// # Safety
/// `dst_base` points to `d * inner` floats and no other task writes columns `c0..c0 + tw`.
pub(super) unsafe fn softmax_col_tile_wasm(
    src: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    let mut m = [f32::NEG_INFINITY; COL_TILE];
    let v8 = tw / 8 * 8;
    let v4 = tw / 4 * 4;
    for r in 0..d {
        let srow = &src[r * inner + c0..r * inner + c0 + tw];
        let mut j = 0;
        // SAFETY: `j + 8 <= v8 <= tw <= COL_TILE` and `j + 4 <= v4 <= tw`.
        unsafe {
            while j < v8 {
                let sp = srow.as_ptr().add(j);
                let mp = m.as_mut_ptr().add(j);
                v128_store(
                    mp as *mut v128,
                    f32x4_pmax(v128_load(mp as *const v128), v128_load(sp as *const v128)),
                );
                v128_store(
                    mp.add(4) as *mut v128,
                    f32x4_pmax(v128_load(mp.add(4) as *const v128), v128_load(sp.add(4) as *const v128)),
                );
                j += 8;
            }
            if j < v4 {
                let mp = m.as_mut_ptr().add(j);
                v128_store(
                    mp as *mut v128,
                    f32x4_pmax(v128_load(mp as *const v128), v128_load(srow.as_ptr().add(j) as *const v128)),
                );
            }
        }
        for jj in v4..tw {
            m[jj] = m[jj].max(srow[jj]);
        }
    }
    let pos_inf = f32x4_splat(f32::INFINITY);
    let mut j = 0;
    // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
    unsafe {
        while j < v4 {
            let mp = m.as_mut_ptr().add(j);
            let mv = v128_load(mp as *const v128);
            let finite = f32x4_lt(f32x4_abs(mv), pos_inf);
            v128_store(mp as *mut v128, v128_and(mv, finite));
            j += 4;
        }
    }
    for a in &mut m[v4..tw] {
        if !a.is_finite() {
            *a = 0.0;
        }
    }
    let mut s = [0f32; COL_TILE];
    if log {
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let ap = s.as_mut_ptr().add(j);
                    let e = fast_exp4(f32x4_sub(
                        v128_load(srow.as_ptr().add(j) as *const v128),
                        v128_load(m.as_ptr().add(j) as *const v128),
                    ));
                    v128_store(ap as *mut v128, f32x4_add(v128_load(ap as *const v128), e));
                    j += 4;
                }
            }
            for jj in v4..tw {
                s[jj] += fast_exp(srow[jj] - m[jj]);
            }
        }
        let mut ls = [0f32; COL_TILE];
        let mut j = 0;
        // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
        unsafe {
            while j < v4 {
                v128_store(
                    ls.as_mut_ptr().add(j) as *mut v128,
                    ln4_wasm(v128_load(s.as_ptr().add(j) as *const v128)),
                );
                j += 4;
            }
        }
        for jj in v4..tw {
            ls[jj] = ln_scalar(s[jj]);
        }
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let y = f32x4_sub(
                        f32x4_sub(
                            v128_load(srow.as_ptr().add(j) as *const v128),
                            v128_load(m.as_ptr().add(j) as *const v128),
                        ),
                        v128_load(ls.as_ptr().add(j) as *const v128),
                    );
                    v128_store(dptr.add(j) as *mut v128, y);
                    j += 4;
                }
            }
            for jj in v4..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = (srow[jj] - m[jj]) - ls[jj] };
            }
        }
    } else {
        for r in 0..d {
            let srow = &src[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let ap = s.as_mut_ptr().add(j);
                    let e = fast_exp4(f32x4_sub(
                        v128_load(srow.as_ptr().add(j) as *const v128),
                        v128_load(m.as_ptr().add(j) as *const v128),
                    ));
                    v128_store(dptr.add(j) as *mut v128, e);
                    v128_store(ap as *mut v128, f32x4_add(v128_load(ap as *const v128), e));
                    j += 4;
                }
            }
            for jj in v4..tw {
                let e = fast_exp(srow[jj] - m[jj]);
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = e };
                s[jj] += e;
            }
        }
        let mut inv = [0f32; COL_TILE];
        let one = f32x4_splat(1.0);
        let mut j = 0;
        // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
        unsafe {
            while j < v4 {
                v128_store(
                    inv.as_mut_ptr().add(j) as *mut v128,
                    f32x4_div(one, v128_load(s.as_ptr().add(j) as *const v128)),
                );
                j += 4;
            }
        }
        for jj in v4..tw {
            inv[jj] = 1.0 / s[jj];
        }
        for r in 0..d {
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let dp = dptr.add(j);
                    v128_store(
                        dp as *mut v128,
                        f32x4_mul(
                            v128_load(dp as *const v128),
                            v128_load(inv.as_ptr().add(j) as *const v128),
                        ),
                    );
                    j += 4;
                }
            }
            for (jj, &iv) in inv.iter().enumerate().take(tw).skip(v4) {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) *= iv };
            }
        }
    }
}

/// 16-lane Wasm SIMD128 `sum(g)` matching `sum_block` bit-for-bit.
pub(super) fn sum_block_wasm(g: &[f32]) -> f32 {
    let (gc, gr) = g.as_chunks::<16>();
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    for gs in gc {
        // SAFETY: `gs` is 16 floats.
        unsafe {
            let p = gs.as_ptr();
            a0 = f32x4_add(a0, v128_load(p as *const v128));
            a1 = f32x4_add(a1, v128_load(p.add(4) as *const v128));
            a2 = f32x4_add(a2, v128_load(p.add(8) as *const v128));
            a3 = f32x4_add(a3, v128_load(p.add(12) as *const v128));
        }
    }
    hsum16_wasm(a0, a1, a2, a3, gr)
}

/// 16-lane Wasm SIMD128 `sum(g * y)` matching `dot_block` bit-for-bit.
pub(super) fn dot_block_wasm(g: &[f32], y: &[f32]) -> f32 {
    let (gc, gr) = g.as_chunks::<16>();
    let (yc, yr) = y.as_chunks::<16>();
    let z = f32x4_splat(0.0);
    let (mut a0, mut a1, mut a2, mut a3) = (z, z, z, z);
    for (gs, ys) in gc.iter().zip(yc) {
        // SAFETY: `gs` and `ys` are 16 floats.
        unsafe {
            let gp = gs.as_ptr();
            let yp = ys.as_ptr();
            a0 = f32x4_add(a0, f32x4_mul(v128_load(gp as *const v128), v128_load(yp as *const v128)));
            a1 = f32x4_add(
                a1,
                f32x4_mul(v128_load(gp.add(4) as *const v128), v128_load(yp.add(4) as *const v128)),
            );
            a2 = f32x4_add(
                a2,
                f32x4_mul(v128_load(gp.add(8) as *const v128), v128_load(yp.add(8) as *const v128)),
            );
            a3 = f32x4_add(
                a3,
                f32x4_mul(v128_load(gp.add(12) as *const v128), v128_load(yp.add(12) as *const v128)),
            );
        }
    }
    let mut rprod = [0f32; 16];
    for (i, (&gv, &yv)) in gr.iter().zip(yr).enumerate() {
        rprod[i] = gv * yv;
    }
    hsum16_wasm(a0, a1, a2, a3, &rprod[..gr.len()])
}

/// Wasm SIMD128 row backward apply for `softmax` (`y * (g - red)`) and `log_softmax` (`g - exp(y) * red`).
pub(super) fn softmax_bwd_row_apply_wasm(y: &[f32], g: &[f32], out: &mut [f32], red: f32, log: bool) {
    let (yc, yr) = y.as_chunks::<8>();
    let (gc, gr) = g.as_chunks::<8>();
    let (oc, or) = out.as_chunks_mut::<8>();
    let rv = f32x4_splat(red);
    if log {
        for ((ys, gs), os) in yc.iter().zip(gc).zip(oc.iter_mut()) {
            // SAFETY: `ys`, `gs`, `os` are 8 floats.
            unsafe {
                let yp = ys.as_ptr();
                let gp = gs.as_ptr();
                let op = os.as_mut_ptr();
                let r0 = f32x4_sub(
                    v128_load(gp as *const v128),
                    f32x4_mul(exp4_wasm(v128_load(yp as *const v128)), rv),
                );
                let r1 = f32x4_sub(
                    v128_load(gp.add(4) as *const v128),
                    f32x4_mul(exp4_wasm(v128_load(yp.add(4) as *const v128)), rv),
                );
                v128_store(op as *mut v128, r0);
                v128_store(op.add(4) as *mut v128, r1);
            }
        }
        for ((o, &yv), &gv) in or.iter_mut().zip(yr).zip(gr) {
            *o = gv - exp_scalar(yv) * red;
        }
    } else {
        for ((ys, gs), os) in yc.iter().zip(gc).zip(oc.iter_mut()) {
            // SAFETY: `ys`, `gs`, `os` are 8 floats.
            unsafe {
                let yp = ys.as_ptr();
                let gp = gs.as_ptr();
                let op = os.as_mut_ptr();
                let r0 = f32x4_mul(v128_load(yp as *const v128), f32x4_sub(v128_load(gp as *const v128), rv));
                let r1 = f32x4_mul(
                    v128_load(yp.add(4) as *const v128),
                    f32x4_sub(v128_load(gp.add(4) as *const v128), rv),
                );
                v128_store(op as *mut v128, r0);
                v128_store(op.add(4) as *mut v128, r1);
            }
        }
        for ((o, &yv), &gv) in or.iter_mut().zip(yr).zip(gr) {
            *o = yv * (gv - red);
        }
    }
}

/// Column-tiled Wasm SIMD128 backward over columns `c0..c0 + tw` (`tw <= COL_TILE`).
///
/// # Safety
/// `dst_base` points to `d * inner` floats and no other task writes columns `c0..c0 + tw`.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn softmax_bwd_col_tile_wasm(
    y: &[f32],
    g: &[f32],
    dst_base: *mut f32,
    d: usize,
    inner: usize,
    c0: usize,
    tw: usize,
    log: bool,
) {
    let mut red = [0f32; COL_TILE];
    let v4 = tw / 4 * 4;
    if log {
        for r in 0..d {
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let rp = red.as_mut_ptr().add(j);
                    v128_store(
                        rp as *mut v128,
                        f32x4_add(
                            v128_load(rp as *const v128),
                            v128_load(grow.as_ptr().add(j) as *const v128),
                        ),
                    );
                    j += 4;
                }
            }
            for jj in v4..tw {
                red[jj] += grow[jj];
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let e = exp4_wasm(v128_load(yrow.as_ptr().add(j) as *const v128));
                    let res = f32x4_sub(
                        v128_load(grow.as_ptr().add(j) as *const v128),
                        f32x4_mul(e, v128_load(red.as_ptr().add(j) as *const v128)),
                    );
                    v128_store(dptr.add(j) as *mut v128, res);
                    j += 4;
                }
            }
            for jj in v4..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = grow[jj] - exp_scalar(yrow[jj]) * red[jj] };
            }
        }
    } else {
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let rp = red.as_mut_ptr().add(j);
                    let p = f32x4_mul(
                        v128_load(grow.as_ptr().add(j) as *const v128),
                        v128_load(yrow.as_ptr().add(j) as *const v128),
                    );
                    v128_store(rp as *mut v128, f32x4_add(v128_load(rp as *const v128), p));
                    j += 4;
                }
            }
            for jj in v4..tw {
                red[jj] += grow[jj] * yrow[jj];
            }
        }
        for r in 0..d {
            let yrow = &y[r * inner + c0..r * inner + c0 + tw];
            let grow = &g[r * inner + c0..r * inner + c0 + tw];
            // SAFETY: caller guarantees disjoint ownership of `c0..c0 + tw`.
            let dptr = unsafe { dst_base.add(r * inner + c0) };
            let mut j = 0;
            // SAFETY: `j + 4 <= v4 <= tw <= COL_TILE`.
            unsafe {
                while j < v4 {
                    let res = f32x4_mul(
                        v128_load(yrow.as_ptr().add(j) as *const v128),
                        f32x4_sub(
                            v128_load(grow.as_ptr().add(j) as *const v128),
                            v128_load(red.as_ptr().add(j) as *const v128),
                        ),
                    );
                    v128_store(dptr.add(j) as *mut v128, res);
                    j += 4;
                }
            }
            for jj in v4..tw {
                // SAFETY: `jj < tw`.
                unsafe { *dptr.add(jj) = yrow[jj] * (grow[jj] - red[jj]) };
            }
        }
    }
}
