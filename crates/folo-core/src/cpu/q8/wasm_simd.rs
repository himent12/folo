//! WebAssembly SIMD128 (`v128`) int8 micro-kernel (`i32x4_dot_i16x8`) and C4 packing/pooling/upsampling
//! helpers, bitwise identical to their scalar and AVX2 counterparts.

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use std::arch::wasm32::*;

use super::Grid;
use super::kernel::{Epi, MR, NR, Out, Tile, TileF32};
use super::silu::{C0, C1, C2, C3, C4, C5, MAGIC_EXP, NEG_LOG2_E, RECIP_MAGIC};

/// Exact Wasm SIMD128 int8 micro-kernel: splits each 4-byte word into even (`[b0, b2]`) and odd (`[b1, b3]`)
/// 16-bit lanes and accumulates via `i32x4_dot_i16x8`. Zero `i16` saturation (`max |u8 * i8 + u8 * i8| =
/// 65280 << i32::MAX`) and bitwise identical to [`super::kernel::kernel_scalar`].
///
/// Processes `NR = 24` columns in two 12-column (`3 × v128`) halves so the 12 `v128` accumulators stay
/// register-resident on 16-register hosts without spills.
///
/// # Safety
/// `a` points to `ks * MR` words; each `rows[p]` (`p < ks`) to at least `col + NR` words.
pub(super) unsafe fn kernel_wasm_simd128(
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut Tile,
) {
    #[inline(always)]
    fn split_w(w: i32) -> (i32, i32) {
        let b = w.to_le_bytes();
        let ev = (b[0] as i8 as i16 as u16 as u32) | ((b[2] as i8 as i16 as u16 as u32) << 16);
        let od = (b[1] as i8 as i16 as u16 as u32) | ((b[3] as i8 as i16 as u16 as u32) << 16);
        (ev as i32, od as i32)
    }
    let z = i32x4_splat(0);
    let m_ff = u16x8_splat(0x00FF);
    // SAFETY: caller guarantees `ks * MR` words at `a` and `col + NR` words at every `rows[p]`.
    unsafe {
        for half in [0usize, 12] {
            let (mut c00, mut c01, mut c02) = (z, z, z);
            let (mut c10, mut c11, mut c12) = (z, z, z);
            let (mut c20, mut c21, mut c22) = (z, z, z);
            let (mut c30, mut c31, mut c32) = (z, z, z);
            for p in 0..ks {
                let ap = a.add(p * MR);
                let (w0e, w0o) = split_w(*ap);
                let (w1e, w1o) = split_w(*ap.add(1));
                let (w2e, w2o) = split_w(*ap.add(2));
                let (w3e, w3o) = split_w(*ap.add(3));
                let (vw0e, vw0o) = (i32x4_splat(w0e), i32x4_splat(w0o));
                let (vw1e, vw1o) = (i32x4_splat(w1e), i32x4_splat(w1o));
                let (vw2e, vw2o) = (i32x4_splat(w2e), i32x4_splat(w2o));
                let (vw3e, vw3o) = (i32x4_splat(w3e), i32x4_splat(w3o));
                let bp = (*rows.add(p)).add(col + half) as *const v128;
                macro_rules! col_step {
                    ($b:expr, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {
                        let b_ev = v128_and($b, m_ff);
                        let b_od = u16x8_shr($b, 8);
                        $c0 = i32x4_add($c0, i32x4_dot_i16x8(b_ev, vw0e));
                        $c0 = i32x4_add($c0, i32x4_dot_i16x8(b_od, vw0o));
                        $c1 = i32x4_add($c1, i32x4_dot_i16x8(b_ev, vw1e));
                        $c1 = i32x4_add($c1, i32x4_dot_i16x8(b_od, vw1o));
                        $c2 = i32x4_add($c2, i32x4_dot_i16x8(b_ev, vw2e));
                        $c2 = i32x4_add($c2, i32x4_dot_i16x8(b_od, vw2o));
                        $c3 = i32x4_add($c3, i32x4_dot_i16x8(b_ev, vw3e));
                        $c3 = i32x4_add($c3, i32x4_dot_i16x8(b_od, vw3o));
                    };
                }
                let b0 = v128_load(bp);
                col_step!(b0, c00, c10, c20, c30);
                let b1 = v128_load(bp.add(1));
                col_step!(b1, c01, c11, c21, c31);
                let b2 = v128_load(bp.add(2));
                col_step!(b2, c02, c12, c22, c32);
            }
            for (r, vs) in
                [[c00, c01, c02], [c10, c11, c12], [c20, c21, c22], [c30, c31, c32]].into_iter().enumerate()
            {
                let dst = acc[r].as_mut_ptr().add(half) as *mut v128;
                v128_store(dst, vs[0]);
                v128_store(dst.add(1), vs[1]);
                v128_store(dst.add(2), vs[2]);
            }
        }
    }
}

/// Quantise one 4-channel slice `p0..p3[0..len]` into C4 words `out[0..len]` using Wasm SIMD128, bit-exact
/// with [`Grid::quantize`].
pub(super) fn quantize_group_wasm(
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    gs: [Grid; 4],
    out: &mut [u32],
) {
    let len = out.len();
    debug_assert!(p0.len() >= len && p1.len() >= len && p2.len() >= len && p3.len() >= len);
    let lo = f32x4_splat(-512.0);
    let hi = f32x4_splat(512.0);
    let inv0 = f32x4_splat(gs[0].inv());
    let inv1 = f32x4_splat(gs[1].inv());
    let inv2 = f32x4_splat(gs[2].inv());
    let inv3 = f32x4_splat(gs[3].inv());
    let z0 = i32x4_splat(gs[0].zero as i32);
    let z1 = i32x4_splat(gs[1].zero as i32);
    let z2 = i32x4_splat(gs[2].zero as i32);
    let z3 = i32x4_splat(gs[3].zero as i32);
    let mut i = 0;
    // SAFETY: `i + 4 <= len` and all five slices have at least `len` elements.
    unsafe {
        while i + 4 <= len {
            let q0 = i32x4_add(
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                    hi,
                    f32x4_pmax(lo, f32x4_mul(v128_load(p0.as_ptr().add(i) as *const v128), inv0)),
                ))),
                z0,
            );
            let q1 = i32x4_add(
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                    hi,
                    f32x4_pmax(lo, f32x4_mul(v128_load(p1.as_ptr().add(i) as *const v128), inv1)),
                ))),
                z1,
            );
            let q2 = i32x4_add(
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                    hi,
                    f32x4_pmax(lo, f32x4_mul(v128_load(p2.as_ptr().add(i) as *const v128), inv2)),
                ))),
                z2,
            );
            let q3 = i32x4_add(
                i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                    hi,
                    f32x4_pmax(lo, f32x4_mul(v128_load(p3.as_ptr().add(i) as *const v128), inv3)),
                ))),
                z3,
            );
            let p01 = i16x8_narrow_i32x4(q0, q1);
            let p23 = i16x8_narrow_i32x4(q2, q3);
            let bytes = u8x16_narrow_i16x8(p01, p23);
            let c4 = u8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(bytes, bytes);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, c4);
            i += 4;
        }
    }
    for j in i..len {
        out[j] = u32::from_le_bytes([
            gs[0].quantize(p0[j]),
            gs[1].quantize(p1[j]),
            gs[2].quantize(p2[j]),
            gs[3].quantize(p3[j]),
        ]);
    }
}

/// Dequantise one C4 group from `src[0..len]` into four `f32` planes `p0..p3[0..len]`, bit-exact with scalar.
pub(super) fn dequantize_group_wasm(
    src: &[u32],
    gs: [Grid; 4],
    p0: &mut [f32],
    p1: &mut [f32],
    p2: &mut [f32],
    p3: &mut [f32],
) {
    let len = src.len();
    debug_assert!(p0.len() >= len && p1.len() >= len && p2.len() >= len && p3.len() >= len);
    let mask = i32x4_splat(0xFF);
    let vz0 = i32x4_splat(gs[0].zero as i32);
    let vz1 = i32x4_splat(gs[1].zero as i32);
    let vz2 = i32x4_splat(gs[2].zero as i32);
    let vz3 = i32x4_splat(gs[3].zero as i32);
    let vs0 = f32x4_splat(gs[0].scale);
    let vs1 = f32x4_splat(gs[1].scale);
    let vs2 = f32x4_splat(gs[2].scale);
    let vs3 = f32x4_splat(gs[3].scale);
    let mut i = 0;
    // SAFETY: `i + 4 <= len` and all five slices have at least `len` elements.
    unsafe {
        while i + 4 <= len {
            let w = v128_load(src.as_ptr().add(i) as *const v128);
            let q0 = v128_and(w, mask);
            let q1 = v128_and(u32x4_shr(w, 8), mask);
            let q2 = v128_and(u32x4_shr(w, 16), mask);
            let q3 = u32x4_shr(w, 24);
            v128_store(
                p0.as_mut_ptr().add(i) as *mut v128,
                f32x4_mul(vs0, f32x4_convert_i32x4(i32x4_sub(q0, vz0))),
            );
            v128_store(
                p1.as_mut_ptr().add(i) as *mut v128,
                f32x4_mul(vs1, f32x4_convert_i32x4(i32x4_sub(q1, vz1))),
            );
            v128_store(
                p2.as_mut_ptr().add(i) as *mut v128,
                f32x4_mul(vs2, f32x4_convert_i32x4(i32x4_sub(q2, vz2))),
            );
            v128_store(
                p3.as_mut_ptr().add(i) as *mut v128,
                f32x4_mul(vs3, f32x4_convert_i32x4(i32x4_sub(q3, vz3))),
            );
            i += 4;
        }
    }
    for j in i..len {
        let w = src[j];
        p0[j] = gs[0].dequantize(w as u8);
        p1[j] = gs[1].dequantize((w >> 8) as u8);
        p2[j] = gs[2].dequantize((w >> 16) as u8);
        p3[j] = gs[3].dequantize((w >> 24) as u8);
    }
}

/// Split `src[0..w]` into `even[j] = src[2*j]` and `odd[j] = src[2*j + 1]` (padded with `pad`).
pub(super) fn deinterleave_even_odd_wasm(src: &[u32], even: &mut [u32], odd: &mut [u32], pad: u32) {
    let w = src.len();
    let pairs = even.len();
    debug_assert_eq!(odd.len(), pairs);
    let full = (w / 8).min(pairs / 4);
    let mut j = 0;
    // SAFETY: `2 * (j + 4) <= w` and `j + 4 <= pairs`.
    unsafe {
        for _ in 0..full {
            let sp = src.as_ptr().add(2 * j) as *const v128;
            let lo = v128_load(sp);
            let hi = v128_load(sp.add(1));
            let ev = u32x4_shuffle::<0, 2, 4, 6>(lo, hi);
            let od = u32x4_shuffle::<1, 3, 5, 7>(lo, hi);
            v128_store(even.as_mut_ptr().add(j) as *mut v128, ev);
            v128_store(odd.as_mut_ptr().add(j) as *mut v128, od);
            j += 4;
        }
    }
    for jj in j..pairs {
        let x0 = 2 * jj;
        even[jj] = if x0 < w { src[x0] } else { pad };
        odd[jj] = if x0 + 1 < w { src[x0 + 1] } else { pad };
    }
}

#[inline(always)]
fn max4(a: u32, b: u32) -> u32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u32::from_le_bytes([0, 1, 2, 3].map(|t| a[t].max(b[t])))
}

/// Separable byte-wise max pool on one C4 plane (`4` C4 words = `16` bytes per `u8x16_max`).
pub(super) fn max_pool_plane_wasm(src: &[u32], h: usize, w: usize, r: usize, tmp: &mut [u32], o: &mut [u32]) {
    let span = r.saturating_mul(2).saturating_add(1);
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let dst = &mut tmp[y * w..(y + 1) * w];
        let left_end = r.min(w);
        for (xo, d) in dst.iter_mut().enumerate().take(left_end) {
            let hi = xo.saturating_add(r).min(w - 1);
            *d = row[0..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
        let mut xo = left_end;
        let inner_end = w.saturating_sub(r);
        // SAFETY: `xo >= r` and `xo + 4 <= w - r` implies `xo - r + d + 4 <= w` for all `d in 0..span`.
        unsafe {
            while xo + 4 <= inner_end {
                let base = row.as_ptr().add(xo - r);
                let mut m = v128_load(base as *const v128);
                for d in 1..span {
                    m = u8x16_max(m, v128_load(base.add(d) as *const v128));
                }
                v128_store(dst.as_mut_ptr().add(xo) as *mut v128, m);
                xo += 4;
            }
        }
        for (xx, d) in dst.iter_mut().enumerate().skip(xo) {
            let (lo, hi) = (xx.saturating_sub(r), xx.saturating_add(r).min(w - 1));
            *d = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
    }
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = y.saturating_add(r).min(h - 1);
        let dst = &mut o[y * w..(y + 1) * w];
        let mut xo = 0;
        // SAFETY: `yy * w + xo + 4 <= h * w`.
        unsafe {
            while xo + 4 <= w {
                let mut m = v128_load(tmp.as_ptr().add(lo * w + xo) as *const v128);
                for yy in (lo + 1)..=hi {
                    m = u8x16_max(m, v128_load(tmp.as_ptr().add(yy * w + xo) as *const v128));
                }
                v128_store(dst.as_mut_ptr().add(xo) as *mut v128, m);
                xo += 4;
            }
        }
        for xx in xo..w {
            dst[xx] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xx]));
        }
    }
}

/// 2x2 nearest-neighbour upsample of one C4 plane (`4 -> 8` words per step).
pub(super) fn upsample2_plane_wasm(src: &[u32], h: usize, w: usize, o: &mut [u32]) {
    let ow = 2 * w;
    for y in 0..h {
        let srow = &src[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        let mut x = 0;
        // SAFETY: `x + 4 <= w` and `2 * x + 8 <= ow`.
        unsafe {
            while x + 4 <= w {
                let v = v128_load(srow.as_ptr().add(x) as *const v128);
                let out0 = u32x4_shuffle::<0, 0, 1, 1>(v, v);
                let out1 = u32x4_shuffle::<2, 2, 3, 3>(v, v);
                v128_store(d0.as_mut_ptr().add(2 * x) as *mut v128, out0);
                v128_store(d0.as_mut_ptr().add(2 * x + 4) as *mut v128, out1);
                v128_store(d1.as_mut_ptr().add(2 * x) as *mut v128, out0);
                v128_store(d1.as_mut_ptr().add(2 * x + 4) as *mut v128, out1);
                x += 4;
            }
        }
        for xx in x..w {
            let v = srow[xx];
            d0[2 * xx] = v;
            d0[2 * xx + 1] = v;
            d1[2 * xx] = v;
            d1[2 * xx + 1] = v;
        }
    }
}

/// Pack planar RGB `f32` pixels in `[0, 1]` into `[R, G, B, 0]` `u32` words, bit-exact with scalar.
pub(super) fn pack_rgb0_slice_wasm(r: &[f32], g: &[f32], b: &[f32], out: &mut [u32]) {
    let len = out.len();
    debug_assert!(r.len() >= len && g.len() >= len && b.len() >= len);
    let scale = f32x4_splat(255.0);
    let zero = f32x4_splat(0.0);
    let z_i = i32x4_splat(0);
    let mut i = 0;
    // SAFETY: `i + 4 <= len` and all four slices have at least `len` elements.
    unsafe {
        while i + 4 <= len {
            let vr = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                scale,
                f32x4_pmax(zero, f32x4_mul(v128_load(r.as_ptr().add(i) as *const v128), scale)),
            )));
            let vg = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                scale,
                f32x4_pmax(zero, f32x4_mul(v128_load(g.as_ptr().add(i) as *const v128), scale)),
            )));
            let vb = i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                scale,
                f32x4_pmax(zero, f32x4_mul(v128_load(b.as_ptr().add(i) as *const v128), scale)),
            )));
            let p01 = i16x8_narrow_i32x4(vr, vg);
            let p23 = i16x8_narrow_i32x4(vb, z_i);
            let bytes = u8x16_narrow_i16x8(p01, p23);
            let rgb0 = u8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(bytes, bytes);
            v128_store(out.as_mut_ptr().add(i) as *mut v128, rgb0);
            i += 4;
        }
    }
    for j in i..len {
        let to_u8 = |x: f32| {
            let v = super::silu::minps(super::silu::maxps(x * 255.0, 0.0), 255.0);
            v.round_ties_even() as i32 as u8
        };
        out[j] = u32::from_le_bytes([to_u8(r[j]), to_u8(g[j]), to_u8(b[j]), 0]);
    }
}

#[inline(always)]
fn quantize4_wasm(v: v128, vs: v128, sign_bit: v128, half: v128, one: v128, max127: v128) -> v128 {
    let x = f32x4_div(v, vs);
    let sign = v128_and(x, sign_bit);
    let abs_x = f32x4_abs(x);
    let fl = f32x4_floor(abs_x);
    let frac = f32x4_sub(abs_x, fl);
    let up = v128_and(f32x4_ge(frac, half), one);
    let r = f32x4_add(fl, up);
    let clamped = f32x4_pmin(max127, r);
    let signed = v128_or(clamped, sign);
    let clean = v128_andnot(signed, f32x4_ne(x, x));
    i32x4_trunc_sat_f32x4(clean)
}

/// Wasm SIMD128 symmetric s8 weight row quantisation, bitwise identical to [`super::quant::quantize_row_scalar`].
pub(super) fn quantize_row_wasm(row: &[f32], dst: &mut [i8]) -> f32 {
    assert_eq!(row.len(), dst.len());
    let mut m0 = f32x4_splat(0.0);
    let mut m1 = f32x4_splat(0.0);
    let (r8, r_rem) = row.as_chunks::<8>();
    for c in r8 {
        // SAFETY: `c` is `[f32; 8]`.
        unsafe {
            let p = c.as_ptr();
            m0 = f32x4_pmax(m0, f32x4_abs(v128_load(p as *const v128)));
            m1 = f32x4_pmax(m1, f32x4_abs(v128_load(p.add(4) as *const v128)));
        }
    }
    let (r4, r_tail) = r_rem.as_chunks::<4>();
    for c in r4 {
        // SAFETY: `c` is `[f32; 4]`.
        unsafe {
            m0 = f32x4_pmax(m0, f32x4_abs(v128_load(c.as_ptr() as *const v128)));
        }
    }
    let m_vec = f32x4_pmax(m0, m1);
    let mut lanes = [0f32; 4];
    // SAFETY: `lanes` holds 4 floats.
    unsafe { v128_store(lanes.as_mut_ptr() as *mut v128, m_vec) };
    let mut m = lanes.iter().copied().fold(0f32, f32::max);
    for &v in r_tail {
        m = m.max(v.abs());
    }

    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    let vs = f32x4_splat(s);
    let sign_bit = f32x4_splat(-0.0);
    let half = f32x4_splat(0.5);
    let one = f32x4_splat(1.0);
    let max127 = f32x4_splat(127.0);

    let (w16, w_rem) = row.as_chunks::<16>();
    let (d16, d_rem) = dst.as_chunks_mut::<16>();
    for (wc, dc) in w16.iter().zip(d16.iter_mut()) {
        // SAFETY: `wc` is `[f32; 16]` and `dc` is `[i8; 16]`.
        unsafe {
            let p = wc.as_ptr();
            let q0 = quantize4_wasm(v128_load(p as *const v128), vs, sign_bit, half, one, max127);
            let q1 = quantize4_wasm(v128_load(p.add(4) as *const v128), vs, sign_bit, half, one, max127);
            let q2 = quantize4_wasm(v128_load(p.add(8) as *const v128), vs, sign_bit, half, one, max127);
            let q3 = quantize4_wasm(v128_load(p.add(12) as *const v128), vs, sign_bit, half, one, max127);
            let p01 = i16x8_narrow_i32x4(q0, q1);
            let p23 = i16x8_narrow_i32x4(q2, q3);
            v128_store(dc.as_mut_ptr() as *mut v128, i8x16_narrow_i16x8(p01, p23));
        }
    }
    for (d, &v) in d_rem.iter_mut().zip(w_rem) {
        *d = (v / s).round().clamp(-127.0, 127.0) as i8;
    }
    s
}

/// Division-free 4-lane Wasm SIMD128 [`super::silu::silu_q8`], bit-exact with scalar `silu_q8` on `wasm32`.
///
/// Note `f32x4_pmax(lhs, rhs)` evaluates `lhs < rhs ? rhs : lhs`, so `f32x4_pmax(b, a)` matches `maxps(a, b)`
/// on every IEEE-754 value including `NaN` and `±0.0`.
#[inline(always)]
pub(super) fn silu_q8_wasm4(x: v128) -> v128 {
    let x = f32x4_pmax(x, f32x4_splat(-85.0));
    let u = f32x4_pmax(f32x4_splat(-124.0), f32x4_mul(x, f32x4_splat(NEG_LOG2_E)));
    let magic = f32x4_splat(MAGIC_EXP);
    let z = f32x4_add(u, magic);
    let n = f32x4_sub(z, magic);
    let f = f32x4_sub(u, n);
    let f2 = f32x4_mul(f, f);
    let p01 = f32x4_add(f32x4_mul(f32x4_splat(C1), f), f32x4_splat(C0));
    let p23 = f32x4_add(f32x4_mul(f32x4_splat(C3), f), f32x4_splat(C2));
    let p45 = f32x4_add(f32x4_mul(f32x4_splat(C5), f), f32x4_splat(C4));
    let p2345 = f32x4_add(f32x4_mul(p45, f2), p23);
    let p = f32x4_add(f32x4_mul(p2345, f2), p01);
    let e = f32x4_mul(p, i32x4_shl(z, 23));
    let one = f32x4_splat(1.0);
    let d = f32x4_add(one, e);
    let r0 = i32x4_sub(i32x4_splat(RECIP_MAGIC as i32), d);
    let e0 = f32x4_add(f32x4_mul(f32x4_neg(d), r0), one);
    let u2 = f32x4_add(f32x4_mul(r0, e0), r0);
    let e2 = f32x4_mul(e0, e0);
    let t2 = f32x4_add(f32x4_mul(e2, e2), e2);
    let r = f32x4_add(f32x4_mul(u2, t2), u2);
    f32x4_mul(x, r)
}

/// Wasm SIMD128 [`super::kernel::accumulate_part_f32`], 4 columns per `f32x4`, bit-exact with scalar on `wasm32`.
pub(super) fn accumulate_part_f32_wasm(acc: &Tile, corr: &[i32; MR], scale: &[f32; MR], dst: &mut TileF32) {
    // SAFETY: fixed-size `[MR][NR]` arrays with `NR = 24 = 6 * 4`.
    unsafe {
        for r in 0..MR {
            let vc = i32x4_splat(corr[r]);
            let vs = f32x4_splat(scale[r]);
            let ap = acc[r].as_ptr();
            let dp = dst[r].as_mut_ptr();
            for k in 0..NR / 4 {
                let a = v128_load(ap.add(4 * k) as *const v128);
                let t = f32x4_convert_i32x4(i32x4_sub(a, vc));
                let cur = v128_load(dp.add(4 * k) as *const v128);
                let next = f32x4_add(f32x4_mul(t, vs), cur);
                v128_store(dp.add(4 * k) as *mut v128, next);
            }
        }
    }
}

/// Shared 4-column (`MR = 4` channels) Wasm SIMD128 epilogue tail: optional SiLU, optional residual FMA, and
/// either C4 byte-packed `u8` requantisation or `f32` tile store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn finish_epi_chunk_wasm(
    epi: &Epi<'_>,
    mut y0: v128,
    mut y1: v128,
    mut y2: v128,
    mut y3: v128,
    res_ptr: *const u32,
    q_ptr: *mut u32,
    f: &mut [[f32; NR]; MR],
    col: usize,
) {
    if epi.silu {
        y0 = silu_q8_wasm4(y0);
        y1 = silu_q8_wasm4(y1);
        y2 = silu_q8_wasm4(y2);
        y3 = silu_q8_wasm4(y3);
    }
    // SAFETY: caller guarantees `col + 4 <= NR` for `res_ptr`, `q_ptr`, and `f[0..4]`.
    unsafe {
        if let Some((rs, rz)) = epi.res {
            let mask = i32x4_splat(0xFF);
            let rw = v128_load(res_ptr.add(col) as *const v128);
            let r0 = f32x4_convert_i32x4(i32x4_sub(v128_and(rw, mask), i32x4_splat(rz[0])));
            let r1 = f32x4_convert_i32x4(i32x4_sub(v128_and(u32x4_shr(rw, 8), mask), i32x4_splat(rz[1])));
            let r2 = f32x4_convert_i32x4(i32x4_sub(v128_and(u32x4_shr(rw, 16), mask), i32x4_splat(rz[2])));
            let r3 = f32x4_convert_i32x4(i32x4_sub(u32x4_shr(rw, 24), i32x4_splat(rz[3])));
            y0 = f32x4_add(f32x4_mul(f32x4_splat(rs[0]), r0), y0);
            y1 = f32x4_add(f32x4_mul(f32x4_splat(rs[1]), r1), y1);
            y2 = f32x4_add(f32x4_mul(f32x4_splat(rs[2]), r2), y2);
            y3 = f32x4_add(f32x4_mul(f32x4_splat(rs[3]), r3), y3);
        }
        match epi.out {
            Out::F32 => {
                v128_store(f[0].as_mut_ptr().add(col) as *mut v128, y0);
                v128_store(f[1].as_mut_ptr().add(col) as *mut v128, y1);
                v128_store(f[2].as_mut_ptr().add(col) as *mut v128, y2);
                v128_store(f[3].as_mut_ptr().add(col) as *mut v128, y3);
            }
            Out::U8 { inv, zero } => {
                let lo = f32x4_splat(-512.0);
                let hi = f32x4_splat(512.0);
                let q0 = i32x4_add(
                    i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                        hi,
                        f32x4_pmax(lo, f32x4_mul(y0, f32x4_splat(inv[0]))),
                    ))),
                    i32x4_splat(zero[0]),
                );
                let q1 = i32x4_add(
                    i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                        hi,
                        f32x4_pmax(lo, f32x4_mul(y1, f32x4_splat(inv[1]))),
                    ))),
                    i32x4_splat(zero[1]),
                );
                let q2 = i32x4_add(
                    i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                        hi,
                        f32x4_pmax(lo, f32x4_mul(y2, f32x4_splat(inv[2]))),
                    ))),
                    i32x4_splat(zero[2]),
                );
                let q3 = i32x4_add(
                    i32x4_trunc_sat_f32x4(f32x4_nearest(f32x4_pmin(
                        hi,
                        f32x4_pmax(lo, f32x4_mul(y3, f32x4_splat(inv[3]))),
                    ))),
                    i32x4_splat(zero[3]),
                );
                let p01 = i16x8_narrow_i32x4(q0, q1);
                let p23 = i16x8_narrow_i32x4(q2, q3);
                let bytes = u8x16_narrow_i16x8(p01, p23);
                let c4 = u8x16_shuffle::<0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15>(bytes, bytes);
                v128_store(q_ptr.add(col) as *mut v128, c4);
            }
        }
    }
}

/// Wasm SIMD128 [`Epi::apply_full`] / [`Epi::apply_full_u8_direct`] over raw pointers (`NR = 24 = 6 × 4` lanes),
/// bit-exact with scalar `Epi::apply`.
///
/// # Safety
/// `q_ptr` is writable for `NR` words when `epi.out` is `Out::U8`; `res_ptr` is readable for `NR` words when
/// `epi.res.is_some()`.
pub(super) unsafe fn apply_epi_wasm_ptr(
    epi: &Epi<'_>,
    acc: &Tile,
    res_ptr: *const u32,
    q_ptr: *mut u32,
    f: &mut [[f32; NR]; MR],
) {
    let (vc0, vc1, vc2, vc3) = (
        i32x4_splat(epi.corr[0]),
        i32x4_splat(epi.corr[1]),
        i32x4_splat(epi.corr[2]),
        i32x4_splat(epi.corr[3]),
    );
    let (vs0, vs1, vs2, vs3) = (
        f32x4_splat(epi.scale[0]),
        f32x4_splat(epi.scale[1]),
        f32x4_splat(epi.scale[2]),
        f32x4_splat(epi.scale[3]),
    );
    let (vb0, vb1, vb2, vb3) = (
        f32x4_splat(epi.bias[0]),
        f32x4_splat(epi.bias[1]),
        f32x4_splat(epi.bias[2]),
        f32x4_splat(epi.bias[3]),
    );
    // SAFETY: fixed-size `[MR][NR]` array (`NR = 24 = 6 * 4`) and caller pointer contract.
    unsafe {
        for k in 0..NR / 4 {
            let col = 4 * k;
            let y0 = f32x4_add(
                f32x4_mul(
                    f32x4_convert_i32x4(i32x4_sub(v128_load(acc[0].as_ptr().add(col) as *const v128), vc0)),
                    vs0,
                ),
                vb0,
            );
            let y1 = f32x4_add(
                f32x4_mul(
                    f32x4_convert_i32x4(i32x4_sub(v128_load(acc[1].as_ptr().add(col) as *const v128), vc1)),
                    vs1,
                ),
                vb1,
            );
            let y2 = f32x4_add(
                f32x4_mul(
                    f32x4_convert_i32x4(i32x4_sub(v128_load(acc[2].as_ptr().add(col) as *const v128), vc2)),
                    vs2,
                ),
                vb2,
            );
            let y3 = f32x4_add(
                f32x4_mul(
                    f32x4_convert_i32x4(i32x4_sub(v128_load(acc[3].as_ptr().add(col) as *const v128), vc3)),
                    vs3,
                ),
                vb3,
            );
            finish_epi_chunk_wasm(epi, y0, y1, y2, y3, res_ptr, q_ptr, f, col);
        }
    }
}

/// Safe wrapper over [`apply_epi_wasm_ptr`].
pub(super) fn apply_epi_wasm(
    epi: &Epi<'_>,
    acc: &Tile,
    res: &[u32; NR],
    q: &mut [u32; NR],
    f: &mut [[f32; NR]; MR],
) {
    // SAFETY: `res` and `q` have length `NR`.
    unsafe { apply_epi_wasm_ptr(epi, acc, res.as_ptr(), q.as_mut_ptr(), f) };
}

/// Wasm SIMD128 [`Epi::apply_f32_full`] / [`Epi::apply_f32_full_u8_direct`] over raw pointers (`NR = 24 = 6 × 4`
/// lanes), bit-exact with scalar `Epi::apply_f32`.
///
/// # Safety
/// `q_ptr` is writable for `NR` words when `epi.out` is `Out::U8`; `res_ptr` is readable for `NR` words when
/// `epi.res.is_some()`.
pub(super) unsafe fn apply_epi_f32_wasm_ptr(
    epi: &Epi<'_>,
    tile: &TileF32,
    res_ptr: *const u32,
    q_ptr: *mut u32,
    f: &mut [[f32; NR]; MR],
) {
    let (vb0, vb1, vb2, vb3) = (
        f32x4_splat(epi.bias[0]),
        f32x4_splat(epi.bias[1]),
        f32x4_splat(epi.bias[2]),
        f32x4_splat(epi.bias[3]),
    );
    // SAFETY: fixed-size `[MR][NR]` array (`NR = 24 = 6 * 4`) and caller pointer contract.
    unsafe {
        for k in 0..NR / 4 {
            let col = 4 * k;
            let y0 = f32x4_add(v128_load(tile[0].as_ptr().add(col) as *const v128), vb0);
            let y1 = f32x4_add(v128_load(tile[1].as_ptr().add(col) as *const v128), vb1);
            let y2 = f32x4_add(v128_load(tile[2].as_ptr().add(col) as *const v128), vb2);
            let y3 = f32x4_add(v128_load(tile[3].as_ptr().add(col) as *const v128), vb3);
            finish_epi_chunk_wasm(epi, y0, y1, y2, y3, res_ptr, q_ptr, f, col);
        }
    }
}

/// Safe wrapper over [`apply_epi_f32_wasm_ptr`].
pub(super) fn apply_epi_f32_wasm(
    epi: &Epi<'_>,
    tile: &TileF32,
    res: &[u32; NR],
    q: &mut [u32; NR],
    f: &mut [[f32; NR]; MR],
) {
    // SAFETY: `res` and `q` have length `NR`.
    unsafe { apply_epi_f32_wasm_ptr(epi, tile, res.as_ptr(), q.as_mut_ptr(), f) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::q8::kernel::{NR, kernel_name, kernel_scalar};
    use crate::cpu::q8::{QStem, alloc, pack_f32_to_rgb0_u8, stem_u8};
    use crate::rng::Rng;

    #[test]
    fn wasm_simd128_q8_kernel_and_helpers_match_scalar_exact() {
        let expected_name =
            if std::env::var_os("FOLO_Q8_SCALAR").is_some() { "portable" } else { "Wasm SIMD128" };
        assert_eq!(kernel_name(), expected_name);
        let mut rng = Rng::seed(0x9E37_79B9);
        for ks in [0usize, 1, 2, 7, 16, 37] {
            let mut a = vec![0i32; ks * MR];
            let mut b_storage = vec![0u32; ks * (NR + 4)];
            for (i, w) in a.iter_mut().enumerate() {
                *w = match i % 7 {
                    0 => i32::from_le_bytes([0x80, 0x80, 0x80, 0x80]),
                    1 => i32::from_le_bytes([0x7F, 0x7F, 0x7F, 0x7F]),
                    2 => i32::from_le_bytes([0x80, 0x7F, 0x80, 0x7F]),
                    _ => (rng.next_u64() >> 11) as i32,
                };
            }
            for (i, v) in b_storage.iter_mut().enumerate() {
                *v = match i % 5 {
                    0 => 0xFFFF_FFFF,
                    1 => 0x0000_0000,
                    2 => 0xFF00_FF00,
                    _ => (rng.next_u64() >> 9) as u32,
                };
            }
            let rows: Vec<*const u32> = (0..ks).map(|p| b_storage[p * (NR + 4)..].as_ptr()).collect();
            for col in [0usize, 1, 3] {
                let mut want: Tile = [[0; NR]; MR];
                let mut got: Tile = [[0; NR]; MR];
                // SAFETY: `a` has `ks * MR` words and each row has `NR + 4 >= col + NR` words.
                unsafe {
                    kernel_scalar(ks, a.as_ptr(), rows.as_ptr(), col, &mut want);
                    kernel_wasm_simd128(ks, a.as_ptr(), rows.as_ptr(), col, &mut got);
                }
                assert_eq!(got, want, "kernel_wasm_simd128 mismatch ks={ks} col={col}");
            }
        }

        // Deinterleave even/odd across SIMD (8 -> 4+4) and padded odd tails.
        for (w, pairs) in [(0usize, 0usize), (1, 1), (7, 4), (15, 9), (16, 8), (27, 15)] {
            let src: Vec<u32> = (0..w).map(|i| (i as u32).wrapping_mul(0x0102_0304) ^ 0x55AA_33CC).collect();
            let (mut ev, mut od) = (vec![0u32; pairs], vec![0u32; pairs]);
            let pad = 0xDEAD_BEEF;
            deinterleave_even_odd_wasm(&src, &mut ev, &mut od, pad);
            for j in 0..pairs {
                assert_eq!(ev[j], if 2 * j < w { src[2 * j] } else { pad });
                assert_eq!(od[j], if 2 * j + 1 < w { src[2 * j + 1] } else { pad });
            }
        }

        // Wide `pack_rgb0_slice_wasm` + `stem_u8` (hits `deinterleave_even_odd_wasm` + `kernel_wasm_simd128`).
        let (h, w, o) = (14usize, 54usize, 8usize);
        let mut rgb: Vec<f32> = (0..3 * h * w).map(|_| rng.uniform(-0.1, 1.1)).collect();
        rgb[0] = f32::NAN;
        rgb[5] = 0.5 / 255.0; // exact tie to even -> 0
        rgb[6] = 1.5 / 255.0; // exact tie to even -> 2
        let mut rgb0 = alloc(1, 4, h * w);
        pack_f32_to_rgb0_u8(&rgb, 1, h, w, &mut rgb0);
        for p in 0..h * w {
            let to_u8 = |x: f32| {
                let v = crate::cpu::q8::silu::minps(crate::cpu::q8::silu::maxps(x * 255.0, 0.0), 255.0);
                v.round_ties_even() as i32 as u8
            };
            let want =
                u32::from_le_bytes([to_u8(rgb[p]), to_u8(rgb[h * w + p]), to_u8(rgb[2 * h * w + p]), 0]);
            assert_eq!(rgb0[p], want, "pack_rgb0 mismatch at {p}");
        }
        let wt: Vec<f32> = (0..o * 3 * 9).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let bias: Vec<f32> = (0..o).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let qs = QStem::new(&wt, &bias, (o, 3), 2, 1, true);
        let (oh, ow) = qs.out_hw(h, w);
        let mut out_f = vec![0f32; o * oh * ow];
        stem_u8(&rgb0, 1, h, w, &qs, crate::cpu::q8::QOut::F32 { data: &mut out_f });
        assert!(out_f.iter().all(|v| v.is_finite()));

        // Verify `silu_q8_wasm4`, `accumulate_part_f32_wasm`, `apply_epi_wasm`, and `apply_epi_f32_wasm`
        // match scalar bit-for-bit across normal, extreme, signed-zero, and NaN inputs.
        let special = [
            -200.0f32,
            -85.0,
            -1.5,
            -0.0,
            0.0,
            0.5,
            3.25,
            85.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -0.125,
        ];
        for chunk in special.as_chunks::<4>().0 {
            // SAFETY: `chunk` has 4 `f32` lanes.
            let got = unsafe {
                let mut out = [0f32; 4];
                v128_store(
                    out.as_mut_ptr() as *mut v128,
                    silu_q8_wasm4(v128_load(chunk.as_ptr() as *const v128)),
                );
                out
            };
            for (g, &x) in got.iter().zip(chunk) {
                let w = crate::cpu::q8::silu_q8(x);
                if x.is_nan() {
                    assert!(g.is_nan(), "silu_q8_wasm4(NaN) must stay NaN");
                } else {
                    assert_eq!(g.to_bits(), w.to_bits(), "silu_q8_wasm4({x}) mismatch: {g} vs {w}");
                }
            }
        }

        // Verify non-FMA `silu_q8` / `silu_q8_wasm4` accuracy over `[-86, 86]` on `wasm32`.
        let mut max_rel = 0.0f64;
        for _ in 0..4_000 {
            let x = rng.uniform(-86.0, 86.0);
            let want = x as f64 / (1.0 + (-(x as f64)).exp());
            let got = crate::cpu::q8::silu_q8(x) as f64;
            max_rel = max_rel.max((got - want).abs() / want.abs().max(1.0));
        }
        assert!(max_rel <= 3e-7, "wasm32 silu_q8 max_rel {max_rel:e} exceeded 3e-7");

        let mut acc: Tile = [[0; NR]; MR];
        let mut init_tile: TileF32 = [[0.0; NR]; MR];
        for r in 0..MR {
            for j in 0..NR {
                acc[r][j] = (rng.next_u64() as i32) >> 12;
                init_tile[r][j] = rng.uniform(-3.0, 3.0);
            }
        }
        // Extreme values and exact half-integer tie-to-even values (`inv[1] = 8.0`, `bias[1] = 0.0`, `zero[1] = 120`).
        init_tile[0][0] = f32::NAN;
        init_tile[0][1] = f32::INFINITY;
        init_tile[0][2] = f32::NEG_INFINITY;
        init_tile[0][3] = 1.0e6;
        init_tile[0][4] = -1.0e6;
        for (j, target) in [0.5f32, 1.5, 2.5, 120.5, 121.5, 254.5].into_iter().enumerate() {
            init_tile[1][j] = (target - 120.0) / 8.0;
        }
        let corr = [17i32, -31, 0, 128];
        let scale = [0.004f32, 0.012, 0.007, 0.019];
        let bias = [-0.25f32, 0.0, 0.75, -1.1];
        let rs = [0.02f32, 0.03, 0.015, 0.025];
        let rz = [10i32, 128, 0, 200];
        let inv = [12.5f32, 8.0, 15.0, 6.25];
        let zero = [32i32, 120, 0, 210];
        let mut res = [0u32; NR];
        for w in &mut res {
            *w = rng.next_u64() as u32;
        }
        let mut dst_w = init_tile;
        let mut dst_s = init_tile;
        accumulate_part_f32_wasm(&acc, &corr, &scale, &mut dst_w);
        for r in 0..MR {
            for j in 0..NR {
                dst_s[r][j] += (acc[r][j].wrapping_sub(corr[r]) as f32) * scale[r];
                if dst_s[r][j].is_nan() {
                    assert!(dst_w[r][j].is_nan());
                } else {
                    assert_eq!(dst_w[r][j].to_bits(), dst_s[r][j].to_bits());
                }
            }
        }
        for silu in [false, true] {
            for has_res in [false, true] {
                for is_u8 in [false, true] {
                    let epi = Epi {
                        corr: &corr,
                        scale: &scale,
                        bias: &bias,
                        silu,
                        res: has_res.then_some((&rs, &rz)),
                        out: if is_u8 { Out::U8 { inv: &inv, zero: &zero } } else { Out::F32 },
                    };
                    let (mut qw, mut qs) = ([0u32; NR], [0u32; NR]);
                    let (mut fw, mut fs) = ([[0f32; NR]; MR], [[0f32; NR]; MR]);
                    apply_epi_wasm(&epi, &acc, &res, &mut qw, &mut fw);
                    epi.apply(&acc, NR, &res, &mut qs, &mut fs);
                    if is_u8 {
                        assert_eq!(qw, qs, "apply_epi_wasm u8 mismatch silu={silu} res={has_res}");
                        let mut qd = [0u32; NR];
                        let res_ptr = if has_res { res.as_ptr() } else { std::ptr::null() };
                        // SAFETY: `qd` holds `NR` words and `res_ptr` is valid when `has_res` is true.
                        unsafe { epi.apply_full_u8_direct(&acc, res_ptr, qd.as_mut_ptr()) };
                        assert_eq!(qd, qs, "apply_full_u8_direct mismatch silu={silu} res={has_res}");
                    } else {
                        for r in 0..MR {
                            for j in 0..NR {
                                assert_eq!(fw[r][j].to_bits(), fs[r][j].to_bits());
                            }
                        }
                    }
                    let (mut qfw, mut qfs) = ([0u32; NR], [0u32; NR]);
                    let (mut ffw, mut ffs) = ([[0f32; NR]; MR], [[0f32; NR]; MR]);
                    apply_epi_f32_wasm(&epi, &dst_w, &res, &mut qfw, &mut ffw);
                    epi.apply_f32(&dst_s, NR, &res, &mut qfs, &mut ffs);
                    if is_u8 {
                        assert_eq!(qfw, qfs, "apply_epi_f32_wasm u8 mismatch silu={silu} res={has_res}");
                        let mut qfd = [0u32; NR];
                        let res_ptr = if has_res { res.as_ptr() } else { std::ptr::null() };
                        // SAFETY: `qfd` holds `NR` words and `res_ptr` is valid when `has_res` is true.
                        unsafe { epi.apply_f32_full_u8_direct(&dst_w, res_ptr, qfd.as_mut_ptr()) };
                        assert_eq!(qfd, qfs, "apply_f32_full_u8_direct mismatch silu={silu} res={has_res}");
                    } else {
                        for r in 0..MR {
                            for j in 0..NR {
                                if ffs[r][j].is_nan() {
                                    assert!(ffw[r][j].is_nan());
                                } else {
                                    assert_eq!(ffw[r][j].to_bits(), ffs[r][j].to_bits());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
