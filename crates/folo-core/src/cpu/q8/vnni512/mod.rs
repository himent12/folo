//! 512-bit AVX-512 VNNI (`avx512f + avx512bw + avx512vl + avx512vnni`) int8 microkernels, bit-exact 16-lane
//! SiLU and epilogues, and fused `4 × 48` (`NR512 = 48`) tile-block / rotating row-strip drivers.
//!
//! Uses 12 ZMM accumulators (4 output channels × 3 ZMMs = 48 pixels per tile) out of the 32-register AVX-512
//! register file, halving `vpdpbusd` instruction count vs 256-bit YMM tiles on Zen 4/5 and Ice Lake / Sapphire
//! Rapids CPUs while staying bit-for-bit identical to [`super::kernel::kernel_scalar`] +
//! [`super::kernel::Epi::apply`].

#![cfg(target_arch = "x86_64")]

mod fused;
mod strip_3x3;

use std::arch::x86_64::*;

pub(super) use fused::{run_multi_part, run_single_part, run_stem_row};
pub(super) use strip_3x3::run_3x3_row;

use super::fused_epi::{AlignedTile, BMode};
use super::kernel::{MR, Tile};
use super::silu::{C0, C1, C2, C3, C4, C5, MAGIC_EXP, NEG_LOG2_E, RECIP_MAGIC};

/// Columns (output pixels) per 512-bit ZMM tile: three 16-lane ZMM registers (`2 * NR`).
pub(super) const NR512: usize = 48;

/// 64-byte aligned local `4 × 48` `i32` accumulator tile (768 bytes) for clean L1 store-forwarding.
#[repr(align(64))]
struct AlignedTile512([[i32; NR512]; MR]);

/// 64-byte aligned local `4 × 48` `f32` multi-part accumulator tile (768 bytes).
#[repr(align(64))]
struct AlignedTileF32_512([[f32; NR512]; MR]);

/// Standalone `4 × 24` AVX-512 VNNI kernel using hybrid 1×ZMM (cols `0..16`) + 1×YMM (cols `16..24`) with
/// 2× K-unrolling (8 `vpdpbusd` and 2 B loads per K-step instead of 12 and 3).
///
/// # Safety
/// `a` points to `ks * MR` words; each `rows[p]` (`p < ks`) to at least `col + NR` words; the CPU supports
/// `avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni`.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn kernel_avx512vnni_zmm(
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut Tile,
) {
    let zz = _mm512_setzero_si512();
    let zy = _mm256_setzero_si256();
    let (mut c00, mut c10, mut c20, mut c30) = (zz, zz, zz, zz);
    let (mut c01, mut c11, mut c21, mut c31) = (zy, zy, zy, zy);
    macro_rules! step24 {
        ($p:expr) => {
            let bp = (*rows.add($p)).add(col);
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = _mm256_loadu_si256(bp.add(16) as *const __m256i);
            let ap = a.add(($p) * MR);
            let w0 = _mm512_set1_epi32(*ap);
            c00 = _mm512_dpbusd_epi32(c00, b0, w0);
            c01 = _mm256_dpbusd_epi32(c01, b1, _mm512_castsi512_si256(w0));
            let w1 = _mm512_set1_epi32(*ap.add(1));
            c10 = _mm512_dpbusd_epi32(c10, b0, w1);
            c11 = _mm256_dpbusd_epi32(c11, b1, _mm512_castsi512_si256(w1));
            let w2 = _mm512_set1_epi32(*ap.add(2));
            c20 = _mm512_dpbusd_epi32(c20, b0, w2);
            c21 = _mm256_dpbusd_epi32(c21, b1, _mm512_castsi512_si256(w2));
            let w3 = _mm512_set1_epi32(*ap.add(3));
            c30 = _mm512_dpbusd_epi32(c30, b0, w3);
            c31 = _mm256_dpbusd_epi32(c31, b1, _mm512_castsi512_si256(w3));
        };
    }
    // SAFETY: caller guarantees `ks * MR` weights at `a` and `col + NR` readable words per row.
    unsafe {
        let mut p = 0;
        while p + 2 <= ks {
            step24!(p);
            step24!(p + 1);
            p += 2;
        }
        if p < ks {
            step24!(p);
        }
        for (r, (vz, vy)) in [(c00, c01), (c10, c11), (c20, c21), (c30, c31)].into_iter().enumerate() {
            let dst = acc[r].as_mut_ptr();
            _mm512_storeu_si512(dst as *mut __m512i, vz);
            _mm256_storeu_si256(dst.add(16) as *mut __m256i, vy);
        }
    }
}

/// [`super::silu::silu_q8`] on 16 lanes (`__m512`), bit-identical to scalar [`super::silu::silu_q8`].
#[cfg(test)]
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn silu_q8_avx512(x: __m512) -> __m512 {
    let x = _mm512_max_ps(_mm512_set1_ps(-85.0), x);
    let u = _mm512_max_ps(_mm512_mul_ps(x, _mm512_set1_ps(NEG_LOG2_E)), _mm512_set1_ps(-124.0));
    let magic = _mm512_set1_ps(MAGIC_EXP);
    let z = _mm512_add_ps(u, magic);
    let n = _mm512_sub_ps(z, magic);
    let f = _mm512_sub_ps(u, n);
    let f2 = _mm512_mul_ps(f, f);
    let p01 = _mm512_fmadd_ps(_mm512_set1_ps(C1), f, _mm512_set1_ps(C0));
    let p23 = _mm512_fmadd_ps(_mm512_set1_ps(C3), f, _mm512_set1_ps(C2));
    let p45 = _mm512_fmadd_ps(_mm512_set1_ps(C5), f, _mm512_set1_ps(C4));
    let p2345 = _mm512_fmadd_ps(p45, f2, p23);
    let p = _mm512_fmadd_ps(p2345, f2, p01);
    let s = _mm512_slli_epi32::<23>(_mm512_castps_si512(z));
    let e = _mm512_mul_ps(p, _mm512_castsi512_ps(s));
    let one = _mm512_set1_ps(1.0);
    let d = _mm512_add_ps(one, e);
    let r0 =
        _mm512_castsi512_ps(_mm512_sub_epi32(_mm512_set1_epi32(RECIP_MAGIC as i32), _mm512_castps_si512(d)));
    let e0 = _mm512_fnmadd_ps(d, r0, one);
    let u2 = _mm512_fmadd_ps(r0, e0, r0);
    let e2 = _mm512_mul_ps(e0, e0);
    let t2 = _mm512_fmadd_ps(e2, e2, e2);
    let r = _mm512_fmadd_ps(u2, t2, u2);
    _mm512_mul_ps(x, r)
}

/// [`super::silu::silu_q8`] on 2 independent 16-lane ZMM vectors (32 floats), bit-identical to scalar.
#[target_feature(enable = "avx512f")]
#[inline]
fn silu_q8_avx512_x2(x: [__m512; 2]) -> [__m512; 2] {
    let min_x = _mm512_set1_ps(-85.0);
    let x0 = _mm512_max_ps(min_x, x[0]);
    let x1 = _mm512_max_ps(min_x, x[1]);
    let nlog2e = _mm512_set1_ps(NEG_LOG2_E);
    let u_lo = _mm512_set1_ps(-124.0);
    let u0 = _mm512_max_ps(_mm512_mul_ps(x0, nlog2e), u_lo);
    let u1 = _mm512_max_ps(_mm512_mul_ps(x1, nlog2e), u_lo);
    let magic = _mm512_set1_ps(MAGIC_EXP);
    let z0 = _mm512_add_ps(u0, magic);
    let z1 = _mm512_add_ps(u1, magic);
    let f0 = _mm512_sub_ps(u0, _mm512_sub_ps(z0, magic));
    let f1 = _mm512_sub_ps(u1, _mm512_sub_ps(z1, magic));
    let f2_0 = _mm512_mul_ps(f0, f0);
    let f2_1 = _mm512_mul_ps(f1, f1);
    let c5 = _mm512_set1_ps(C5);
    let c4 = _mm512_set1_ps(C4);
    let p45_0 = _mm512_fmadd_ps(c5, f0, c4);
    let p45_1 = _mm512_fmadd_ps(c5, f1, c4);
    let c3 = _mm512_set1_ps(C3);
    let c2 = _mm512_set1_ps(C2);
    let p23_0 = _mm512_fmadd_ps(c3, f0, c2);
    let p23_1 = _mm512_fmadd_ps(c3, f1, c2);
    let c1 = _mm512_set1_ps(C1);
    let c0 = _mm512_set1_ps(C0);
    let p01_0 = _mm512_fmadd_ps(c1, f0, c0);
    let p01_1 = _mm512_fmadd_ps(c1, f1, c0);
    let p2345_0 = _mm512_fmadd_ps(p45_0, f2_0, p23_0);
    let p2345_1 = _mm512_fmadd_ps(p45_1, f2_1, p23_1);
    let p0 = _mm512_fmadd_ps(p2345_0, f2_0, p01_0);
    let p1 = _mm512_fmadd_ps(p2345_1, f2_1, p01_1);
    let e0 = _mm512_mul_ps(p0, _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_castps_si512(z0))));
    let e1 = _mm512_mul_ps(p1, _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_castps_si512(z1))));
    let one = _mm512_set1_ps(1.0);
    let d0 = _mm512_add_ps(one, e0);
    let d1 = _mm512_add_ps(one, e1);
    let rmagic = _mm512_set1_epi32(RECIP_MAGIC as i32);
    let r0_0 = _mm512_castsi512_ps(_mm512_sub_epi32(rmagic, _mm512_castps_si512(d0)));
    let r0_1 = _mm512_castsi512_ps(_mm512_sub_epi32(rmagic, _mm512_castps_si512(d1)));
    let err0 = _mm512_fnmadd_ps(d0, r0_0, one);
    let err1 = _mm512_fnmadd_ps(d1, r0_1, one);
    let u2_0 = _mm512_fmadd_ps(r0_0, err0, r0_0);
    let u2_1 = _mm512_fmadd_ps(r0_1, err1, r0_1);
    let e2_0 = _mm512_mul_ps(err0, err0);
    let e2_1 = _mm512_mul_ps(err1, err1);
    let t2_0 = _mm512_fmadd_ps(e2_0, e2_0, e2_0);
    let t2_1 = _mm512_fmadd_ps(e2_1, e2_1, e2_1);
    let r0 = _mm512_fmadd_ps(u2_0, t2_0, u2_0);
    let r1 = _mm512_fmadd_ps(u2_1, t2_1, u2_1);
    [_mm512_mul_ps(x0, r0), _mm512_mul_ps(x1, r1)]
}

/// Shared 16-pixel AVX-512 epilogue core from four `f32` vectors `y0..y3`.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn finish_col16<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool>(
    mut y0: __m512,
    mut y1: __m512,
    mut y2: __m512,
    mut y3: __m512,
    res_ptr: *const u32,
    rs: &[f32; MR],
    rz: &[i32; MR],
    inv: &[f32; MR],
    zero: &[i32; MR],
    q_dst: *mut u32,
    f_dst: &mut [[f32; NR512]; MR],
    off: usize,
) {
    if SILU {
        [y0, y1] = silu_q8_avx512_x2([y0, y1]);
        [y2, y3] = silu_q8_avx512_x2([y2, y3]);
    }
    // SAFETY: caller guarantees `res_ptr` readable for 16 words when `HAS_RES`, `q_dst` writable for 16 words
    // when `OUT_U8`, and `off + 16 <= NR512`.
    unsafe {
        if HAS_RES {
            let mask_ff = _mm512_set1_epi32(0xFF);
            let w = _mm512_loadu_si512(res_ptr as *const __m512i);
            let b0 = _mm512_and_si512(w, mask_ff);
            let b1 = _mm512_and_si512(_mm512_srli_epi32::<8>(w), mask_ff);
            let b2 = _mm512_and_si512(_mm512_srli_epi32::<16>(w), mask_ff);
            let b3 = _mm512_srli_epi32::<24>(w);
            y0 = _mm512_fmadd_ps(
                _mm512_set1_ps(rs[0]),
                _mm512_cvtepi32_ps(_mm512_sub_epi32(b0, _mm512_set1_epi32(rz[0]))),
                y0,
            );
            y1 = _mm512_fmadd_ps(
                _mm512_set1_ps(rs[1]),
                _mm512_cvtepi32_ps(_mm512_sub_epi32(b1, _mm512_set1_epi32(rz[1]))),
                y1,
            );
            y2 = _mm512_fmadd_ps(
                _mm512_set1_ps(rs[2]),
                _mm512_cvtepi32_ps(_mm512_sub_epi32(b2, _mm512_set1_epi32(rz[2]))),
                y2,
            );
            y3 = _mm512_fmadd_ps(
                _mm512_set1_ps(rs[3]),
                _mm512_cvtepi32_ps(_mm512_sub_epi32(b3, _mm512_set1_epi32(rz[3]))),
                y3,
            );
        }
        if OUT_U8 {
            let lo = _mm512_set1_ps(-512.0);
            let hi = _mm512_set1_ps(512.0);
            let transpose =
                _mm512_broadcast_i32x4(_mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
            let (z0, z1, z2, z3) = (zero[0] as i16, zero[1] as i16, zero[2] as i16, zero[3] as i16);
            let z01 = _mm512_broadcast_i32x4(_mm_setr_epi16(z0, z0, z0, z0, z1, z1, z1, z1));
            let z23 = _mm512_broadcast_i32x4(_mm_setr_epi16(z2, z2, z2, z2, z3, z3, z3, z3));
            let v0 = _mm512_cvtps_epi32(_mm512_min_ps(
                _mm512_max_ps(_mm512_mul_ps(y0, _mm512_set1_ps(inv[0])), lo),
                hi,
            ));
            let v1 = _mm512_cvtps_epi32(_mm512_min_ps(
                _mm512_max_ps(_mm512_mul_ps(y1, _mm512_set1_ps(inv[1])), lo),
                hi,
            ));
            let v2 = _mm512_cvtps_epi32(_mm512_min_ps(
                _mm512_max_ps(_mm512_mul_ps(y2, _mm512_set1_ps(inv[2])), lo),
                hi,
            ));
            let v3 = _mm512_cvtps_epi32(_mm512_min_ps(
                _mm512_max_ps(_mm512_mul_ps(y3, _mm512_set1_ps(inv[3])), lo),
                hi,
            ));
            let p01 = _mm512_adds_epi16(_mm512_packs_epi32(v0, v1), z01);
            let p23 = _mm512_adds_epi16(_mm512_packs_epi32(v2, v3), z23);
            let bytes = _mm512_shuffle_epi8(_mm512_packus_epi16(p01, p23), transpose);
            _mm512_storeu_si512(q_dst as *mut __m512i, bytes);
        } else {
            _mm512_storeu_ps(f_dst[0].as_mut_ptr().add(off), y0);
            _mm512_storeu_ps(f_dst[1].as_mut_ptr().add(off), y1);
            _mm512_storeu_ps(f_dst[2].as_mut_ptr().add(off), y2);
            _mm512_storeu_ps(f_dst[3].as_mut_ptr().add(off), y3);
        }
    }
}

/// Apply the 512-bit epilogue across `ZVECS` 16-pixel chunks (`16`, `32`, or `48` columns) from `acc`.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn apply_epi_512<const SILU: bool, const HAS_RES: bool, const OUT_U8: bool, const ZVECS: usize>(
    acc: &AlignedTile512,
    corr: &[i32; MR],
    scale: &[f32; MR],
    bias: &[f32; MR],
    res_ptr: *const u32,
    rs: &[f32; MR],
    rz: &[i32; MR],
    inv: &[f32; MR],
    zero: &[i32; MR],
    q_dst: *mut u32,
    f_dst: &mut [[f32; NR512]; MR],
) {
    // SAFETY: `acc.0` is 64-byte aligned (`MR x NR512`); `ZVECS <= 3`; pointer bounds guaranteed by caller.
    unsafe {
        let vc0 = _mm512_set1_epi32(corr[0]);
        let vc1 = _mm512_set1_epi32(corr[1]);
        let vc2 = _mm512_set1_epi32(corr[2]);
        let vc3 = _mm512_set1_epi32(corr[3]);
        let vs0 = _mm512_set1_ps(scale[0]);
        let vs1 = _mm512_set1_ps(scale[1]);
        let vs2 = _mm512_set1_ps(scale[2]);
        let vs3 = _mm512_set1_ps(scale[3]);
        let vb0 = _mm512_set1_ps(bias[0]);
        let vb1 = _mm512_set1_ps(bias[1]);
        let vb2 = _mm512_set1_ps(bias[2]);
        let vb3 = _mm512_set1_ps(bias[3]);
        for k in 0..ZVECS {
            let off = 16 * k;
            let y0 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_load_si512(acc.0[0].as_ptr().add(off) as *const __m512i),
                    vc0,
                )),
                vs0,
                vb0,
            );
            let y1 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_load_si512(acc.0[1].as_ptr().add(off) as *const __m512i),
                    vc1,
                )),
                vs1,
                vb1,
            );
            let y2 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_load_si512(acc.0[2].as_ptr().add(off) as *const __m512i),
                    vc2,
                )),
                vs2,
                vb2,
            );
            let y3 = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_load_si512(acc.0[3].as_ptr().add(off) as *const __m512i),
                    vc3,
                )),
                vs3,
                vb3,
            );
            finish_col16::<SILU, HAS_RES, OUT_U8>(
                y0,
                y1,
                y2,
                y3,
                if HAS_RES { res_ptr.add(off) } else { res_ptr },
                rs,
                rz,
                inv,
                zero,
                if OUT_U8 { q_dst.add(off) } else { q_dst },
                f_dst,
                off,
            );
        }
    }
}

macro_rules! dp512 {
    ($zvecs:expr, $b0:ident, $b1:ident, $b2:ident, $ap:ident,
     $c00:ident, $c01:ident, $c02:ident,
     $c10:ident, $c11:ident, $c12:ident,
     $c20:ident, $c21:ident, $c22:ident,
     $c30:ident, $c31:ident, $c32:ident) => {
        let w0 = _mm512_set1_epi32(*$ap);
        $c00 = _mm512_dpbusd_epi32($c00, $b0, w0);
        if $zvecs >= 2 {
            $c01 = _mm512_dpbusd_epi32($c01, $b1, w0);
        }
        if $zvecs >= 3 {
            $c02 = _mm512_dpbusd_epi32($c02, $b2, w0);
        }
        let w1 = _mm512_set1_epi32(*$ap.add(1));
        $c10 = _mm512_dpbusd_epi32($c10, $b0, w1);
        if $zvecs >= 2 {
            $c11 = _mm512_dpbusd_epi32($c11, $b1, w1);
        }
        if $zvecs >= 3 {
            $c12 = _mm512_dpbusd_epi32($c12, $b2, w1);
        }
        let w2 = _mm512_set1_epi32(*$ap.add(2));
        $c20 = _mm512_dpbusd_epi32($c20, $b0, w2);
        if $zvecs >= 2 {
            $c21 = _mm512_dpbusd_epi32($c21, $b1, w2);
        }
        if $zvecs >= 3 {
            $c22 = _mm512_dpbusd_epi32($c22, $b2, w2);
        }
        let w3 = _mm512_set1_epi32(*$ap.add(3));
        $c30 = _mm512_dpbusd_epi32($c30, $b0, w3);
        if $zvecs >= 2 {
            $c31 = _mm512_dpbusd_epi32($c31, $b1, w3);
        }
        if $zvecs >= 3 {
            $c32 = _mm512_dpbusd_epi32($c32, $b2, w3);
        }
    };
}
use dp512;

macro_rules! store_tile512 {
    ($zvecs:expr, $acc:ident,
     $c00:ident, $c01:ident, $c02:ident,
     $c10:ident, $c11:ident, $c12:ident,
     $c20:ident, $c21:ident, $c22:ident,
     $c30:ident, $c31:ident, $c32:ident) => {
        let out = $acc.0.as_mut_ptr() as *mut __m512i;
        _mm512_store_si512(out, $c00);
        if $zvecs >= 2 {
            _mm512_store_si512(out.add(1), $c01);
        }
        if $zvecs >= 3 {
            _mm512_store_si512(out.add(2), $c02);
        }
        _mm512_store_si512(out.add(3), $c10);
        if $zvecs >= 2 {
            _mm512_store_si512(out.add(4), $c11);
        }
        if $zvecs >= 3 {
            _mm512_store_si512(out.add(5), $c12);
        }
        _mm512_store_si512(out.add(6), $c20);
        if $zvecs >= 2 {
            _mm512_store_si512(out.add(7), $c21);
        }
        if $zvecs >= 3 {
            _mm512_store_si512(out.add(8), $c22);
        }
        _mm512_store_si512(out.add(9), $c30);
        if $zvecs >= 2 {
            _mm512_store_si512(out.add(10), $c31);
        }
        if $zvecs >= 3 {
            _mm512_store_si512(out.add(11), $c32);
        }
    };
}
use store_tile512;

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_1x1_512<const ZVECS: usize>(
    ks: usize,
    a: *const i32,
    base: *const u32,
    hw: usize,
    col: usize,
    acc: &mut AlignedTile512,
) {
    let z = _mm512_setzero_si512();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: caller guarantees `ks` planes of stride `hw` with `col + ZVECS * 16` readable words.
    unsafe {
        let mut bp = base.add(col);
        let mut ap = a;
        let hw2 = hw.wrapping_mul(2);
        let mut p = 0;
        while p + 2 <= ks {
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
            let bp1 = bp.add(hw);
            let ap1 = ap.add(MR);
            let b0 = _mm512_loadu_si512(bp1 as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp1.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp1.add(32) as *const __m512i) } else { z };
            dp512!(ZVECS, b0, b1, b2, ap1, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
            bp = bp.wrapping_add(hw2);
            ap = ap.wrapping_add(2 * MR);
            p += 2;
        }
        if p < ks {
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
        }
        store_tile512!(ZVECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_1x1_parts_512<const ZVECS: usize>(
    parts: &[(*const u32, usize)],
    hw: usize,
    a: *const i32,
    col: usize,
    acc: &mut AlignedTile512,
) {
    let z = _mm512_setzero_si512();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: caller guarantees each part `(base, g_cnt)` has `g_cnt` planes of stride `hw` with
    // `col + ZVECS * 16` readable words, and `a` holds `sum(g_cnt) * MR` words.
    unsafe {
        let hw2 = hw.wrapping_mul(2);
        let mut ap = a;
        for &(base, g_cnt) in parts {
            if g_cnt == 0 {
                continue;
            }
            let mut bp = base.add(col);
            let mut p = 0;
            while p + 2 <= g_cnt {
                let b0 = _mm512_loadu_si512(bp as *const __m512i);
                let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
                let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
                dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                let bp1 = bp.add(hw);
                let ap1 = ap.add(MR);
                let b0 = _mm512_loadu_si512(bp1 as *const __m512i);
                let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp1.add(16) as *const __m512i) } else { z };
                let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp1.add(32) as *const __m512i) } else { z };
                dp512!(ZVECS, b0, b1, b2, ap1, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                bp = bp.wrapping_add(hw2);
                ap = ap.wrapping_add(2 * MR);
                p += 2;
            }
            if p < g_cnt {
                let b0 = _mm512_loadu_si512(bp as *const __m512i);
                let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
                let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
                dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
                ap = ap.wrapping_add(MR);
            }
        }
        store_tile512!(ZVECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_rows_512<const ZVECS: usize>(
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut AlignedTile512,
) {
    let z = _mm512_setzero_si512();
    let (mut c00, mut c01, mut c02) = (z, z, z);
    let (mut c10, mut c11, mut c12) = (z, z, z);
    let (mut c20, mut c21, mut c22) = (z, z, z);
    let (mut c30, mut c31, mut c32) = (z, z, z);
    // SAFETY: caller guarantees `ks * MR` weights and `col + ZVECS * 16` readable words per row.
    unsafe {
        let mut p = 0;
        while p + 2 <= ks {
            let bp = (*rows.add(p)).add(col);
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            let ap = a.add(p * MR);
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
            let bp = (*rows.add(p + 1)).add(col);
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            let ap = a.add((p + 1) * MR);
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
            p += 2;
        }
        if p < ks {
            let bp = (*rows.add(p)).add(col);
            let b0 = _mm512_loadu_si512(bp as *const __m512i);
            let b1 = if ZVECS >= 2 { _mm512_loadu_si512(bp.add(16) as *const __m512i) } else { z };
            let b2 = if ZVECS >= 3 { _mm512_loadu_si512(bp.add(32) as *const __m512i) } else { z };
            let ap = a.add(p * MR);
            dp512!(ZVECS, b0, b1, b2, ap, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
        }
        store_tile512!(ZVECS, acc, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32);
    }
}

/// 8-column (1 YMM) EVEX `vpdpbusd` K-loop for narrow `1..=8` row/plane tails without reading 16 words.
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_bmode_8(
    b_mode: BMode<'_>,
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut AlignedTile,
) {
    let z = _mm256_setzero_si256();
    let (mut c0, mut c1, mut c2, mut c3) = (z, z, z, z);
    macro_rules! step8 {
        ($bp:expr, $ap:expr) => {
            let b0 = _mm256_loadu_si256(($bp) as *const __m256i);
            c0 = _mm256_dpbusd_epi32(c0, b0, _mm256_set1_epi32(*$ap));
            c1 = _mm256_dpbusd_epi32(c1, b0, _mm256_set1_epi32(*$ap.add(1)));
            c2 = _mm256_dpbusd_epi32(c2, b0, _mm256_set1_epi32(*$ap.add(2)));
            c3 = _mm256_dpbusd_epi32(c3, b0, _mm256_set1_epi32(*$ap.add(3)));
        };
    }
    // SAFETY: caller guarantees `col + 8` readable words in B and `ks * MR` weights at `a`.
    unsafe {
        match b_mode {
            BMode::Pointwise1x1 { base, hw } => {
                let mut bp = base.add(col);
                let mut ap = a;
                for _ in 0..ks {
                    step8!(bp, ap);
                    bp = bp.wrapping_add(hw);
                    ap = ap.wrapping_add(MR);
                }
            }
            BMode::Pointwise1x1Parts { parts, hw } => {
                let mut ap = a;
                for &(base, g_cnt) in parts {
                    let mut bp = base.add(col);
                    for _ in 0..g_cnt {
                        step8!(bp, ap);
                        bp = bp.wrapping_add(hw);
                        ap = ap.wrapping_add(MR);
                    }
                }
            }
            BMode::Rows => {
                for p in 0..ks {
                    step8!((*rows.add(p)).add(col), a.add(p * MR));
                }
            }
        }
        _mm256_store_si256(acc.0[0].as_mut_ptr() as *mut __m256i, c0);
        _mm256_store_si256(acc.0[1].as_mut_ptr() as *mut __m256i, c1);
        _mm256_store_si256(acc.0[2].as_mut_ptr() as *mut __m256i, c2);
        _mm256_store_si256(acc.0[3].as_mut_ptr() as *mut __m256i, c3);
    }
}

#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline]
unsafe fn k_loop_bmode_512<const ZVECS: usize>(
    b_mode: BMode<'_>,
    ks: usize,
    a: *const i32,
    rows: *const *const u32,
    col: usize,
    acc: &mut AlignedTile512,
) {
    // SAFETY: forwarded from caller.
    unsafe {
        match b_mode {
            BMode::Pointwise1x1 { base, hw } => k_loop_1x1_512::<ZVECS>(ks, a, base, hw, col, acc),
            BMode::Pointwise1x1Parts { parts, hw } => k_loop_1x1_parts_512::<ZVECS>(parts, hw, a, col, acc),
            BMode::Rows => k_loop_rows_512::<ZVECS>(ks, a, rows, col, acc),
        }
    }
}
