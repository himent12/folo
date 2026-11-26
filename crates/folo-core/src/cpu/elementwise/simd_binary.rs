//! AVX2 vectorised binary ops (arithmetic, exact IEEE-754 min/max, comparisons) and `where_cond` kernels.
//!
//! Every kernel handles contiguous, unit-stride `(1, 1)`, broadcast `(1, 0)` / `(0, 1)` / `(0, 0)`, and
//! general strided layouts without intermediate allocations, matching scalar IEEE-754 semantics bit for bit.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use super::simd::has_avx2;
use super::{fill_rows, max_num, min_num, where3};
use crate::Layout;
use crate::backend::BinaryOp;
use crate::cpu::pool_alloc::alloc_uninit;
use crate::cpu::{CHUNK, PAR_MIN, par_chunks};
use crate::layout::RowPlan;

/// Bitwise-exact `f32::max(a, b)` matching `if a > b || b.is_nan() { a } else { b }` (`maxNum`, same as
/// Rust's `f32::max` on x86_64 and `wasm_simd::max4_exact`).
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn max8_exact(a: __m256, b: __m256) -> __m256 {
    let m = _mm256_max_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
}

/// Bitwise-exact `f32::min(a, b)` matching `if a < b || b.is_nan() { a } else { b }` (`minNum`, same as
/// Rust's `f32::min` on x86_64 and `wasm_simd::min4_exact`).
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn min8_exact(a: __m256, b: __m256) -> __m256 {
    let m = _mm256_min_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
}

#[inline(always)]
fn mask_f32(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// Generate `(vv, vs, sv)` 2x8-unrolled + 8-wide + exact scalar-tail AVX2 kernels for a two-operand op,
/// optionally accepting one scalar parameter `$p: f32` broadcast into `$vp: __m256`.
/// Note: `$va/$vb` and `$sa/$sb` MUST be distinct identifiers so vector bindings do not shadow scalar tails.
macro_rules! avx2_bin_loops {
    ($va:ident, $vb:ident => $vexpr:expr; $sa:ident, $sb:ident => $sexpr:expr $(; param $p:ident, $vp:ident)?) => {{
        #[target_feature(enable = "avx2")]
        unsafe fn vv(a: &[f32], b: &[f32], out: &mut [f32] $(, $p: f32)?) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());
            $(let $vp = _mm256_set1_ps($p);)?
            let (ap, ar) = a.as_chunks::<16>();
            let (bp, br) = b.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for ((ac, bc), oc) in ap.iter().zip(bp.iter()).zip(op.iter_mut()) {
                // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let (ap, bp, dp) = (ac.as_ptr(), bc.as_ptr(), oc.as_mut_ptr());
                    let ($va, $vb) = (_mm256_loadu_ps(ap), _mm256_loadu_ps(bp));
                    let r0 = $vexpr;
                    let ($va, $vb) = (_mm256_loadu_ps(ap.add(8)), _mm256_loadu_ps(bp.add(8)));
                    let r1 = $vexpr;
                    _mm256_storeu_ps(dp, r0);
                    _mm256_storeu_ps(dp.add(8), r1);
                }
            }
            let (a8, at) = ar.as_chunks::<8>();
            let (b8, bt) = br.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for ((ac, bc), oc) in a8.iter().zip(b8.iter()).zip(o8.iter_mut()) {
                // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let ($va, $vb) = (_mm256_loadu_ps(ac.as_ptr()), _mm256_loadu_ps(bc.as_ptr()));
                    _mm256_storeu_ps(oc.as_mut_ptr(), $vexpr);
                }
            }
            for ((o, &$sa), &$sb) in ot.iter_mut().zip(at).zip(bt) {
                *o = $sexpr;
            }
        }
        #[target_feature(enable = "avx2")]
        unsafe fn vs(a: &[f32], $sb: f32, out: &mut [f32] $(, $p: f32)?) {
            debug_assert_eq!(a.len(), out.len());
            $(let $vp = _mm256_set1_ps($p);)?
            let $vb = _mm256_set1_ps($sb);
            let (ap, ar) = a.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (ac, oc) in ap.iter().zip(op.iter_mut()) {
                // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let (ap, dp) = (ac.as_ptr(), oc.as_mut_ptr());
                    let $va = _mm256_loadu_ps(ap);
                    let r0 = $vexpr;
                    let $va = _mm256_loadu_ps(ap.add(8));
                    let r1 = $vexpr;
                    _mm256_storeu_ps(dp, r0);
                    _mm256_storeu_ps(dp.add(8), r1);
                }
            }
            let (a8, at) = ar.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (ac, oc) in a8.iter().zip(o8.iter_mut()) {
                // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let $va = _mm256_loadu_ps(ac.as_ptr());
                    _mm256_storeu_ps(oc.as_mut_ptr(), $vexpr);
                }
            }
            for (o, &$sa) in ot.iter_mut().zip(at) {
                *o = $sexpr;
            }
        }
        #[target_feature(enable = "avx2")]
        unsafe fn sv($sa: f32, b: &[f32], out: &mut [f32] $(, $p: f32)?) {
            debug_assert_eq!(b.len(), out.len());
            $(let $vp = _mm256_set1_ps($p);)?
            let $va = _mm256_set1_ps($sa);
            let (bp, br) = b.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (bc, oc) in bp.iter().zip(op.iter_mut()) {
                // SAFETY: 16-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let (bp, dp) = (bc.as_ptr(), oc.as_mut_ptr());
                    let $vb = _mm256_loadu_ps(bp);
                    let r0 = $vexpr;
                    let $vb = _mm256_loadu_ps(bp.add(8));
                    let r1 = $vexpr;
                    _mm256_storeu_ps(dp, r0);
                    _mm256_storeu_ps(dp.add(8), r1);
                }
            }
            let (b8, bt) = br.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (bc, oc) in b8.iter().zip(o8.iter_mut()) {
                // SAFETY: 8-float chunks; unaligned AVX2 loads/stores.
                unsafe {
                    let $vb = _mm256_loadu_ps(bc.as_ptr());
                    _mm256_storeu_ps(oc.as_mut_ptr(), $vexpr);
                }
            }
            for (o, &$sb) in ot.iter_mut().zip(bt) {
                *o = $sexpr;
            }
        }
        (
            vv as unsafe fn(&[f32], &[f32], &mut [f32] $(, $p: f32)?),
            vs as unsafe fn(&[f32], f32, &mut [f32] $(, $p: f32)?),
            sv as unsafe fn(f32, &[f32], &mut [f32] $(, $p: f32)?),
        )
    }};
}
pub(super) use avx2_bin_loops;

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn map2_avx2(
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    vv: impl Fn(&[f32], &[f32], &mut [f32]) + Sync,
    vs: impl Fn(&[f32], f32, &mut [f32]) + Sync,
    sv: impl Fn(f32, &[f32], &mut [f32]) + Sync,
    scalar: impl Fn(f32, f32) -> f32 + Sync,
) -> Vec<f32> {
    let n = la.numel();
    let mut out = alloc_uninit(n);
    if n == 0 {
        return out;
    }
    if let (Some(ra), Some(rb)) = (la.contiguous_range(), lb.contiguous_range()) {
        let (a, b) = (&a[ra], &b[rb]);
        let piece = |c: usize, o: &mut [f32]| {
            let r = c * CHUNK..c * CHUNK + o.len();
            vv(&a[r.clone()], &b[r], o);
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan = RowPlan::<2>::new(la.dims(), [la.strides(), lb.strides()], [la.offset(), lb.offset()]);
    let [sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oa, ob]| {
        let len = row.len();
        match (sa, sb) {
            (1, 1) => vv(&a[oa..oa + len], &b[ob..ob + len], row),
            (1, 0) => vs(&a[oa..oa + len], b[ob], row),
            (0, 1) => sv(a[oa], &b[ob..ob + len], row),
            (0, 0) => row.fill(scalar(a[oa], b[ob])),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = scalar(a[oa + i * sa], b[ob + i * sb]);
                }
            }
        }
    });
    out
}

/// AVX2 binary op dispatch (`Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`, `Eq`, `Ne`, `Gt`, `Ge`, `Lt`, `Le`).
pub(super) fn try_binary_avx2(
    op: BinaryOp,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
) -> Option<Vec<f32>> {
    if !has_avx2() {
        return None;
    }
    macro_rules! go2 {
        ($va:ident, $vb:ident => $vexpr:expr; $sa:ident, $sb:ident => $sexpr:expr) => {{
            let (vv, vs, sv) = avx2_bin_loops!($va, $vb => $vexpr; $sa, $sb => $sexpr);
            // SAFETY: `has_avx2()` checked above.
            Some(map2_avx2(
                a,
                la,
                b,
                lb,
                |x, y, o| unsafe { vv(x, y, o) },
                |x, y, o| unsafe { vs(x, y, o) },
                |x, y, o| unsafe { sv(x, y, o) },
                |$sa, $sb| $sexpr,
            ))
        }};
    }
    use BinaryOp::*;
    match op {
        Add => go2!(vx, vy => _mm256_add_ps(vx, vy); sx, sy => sx + sy),
        Sub => go2!(vx, vy => _mm256_sub_ps(vx, vy); sx, sy => sx - sy),
        Mul => go2!(vx, vy => _mm256_mul_ps(vx, vy); sx, sy => sx * sy),
        Div => go2!(vx, vy => _mm256_div_ps(vx, vy); sx, sy => sx / sy),
        Maximum => go2!(vx, vy => max8_exact(vx, vy); sx, sy => max_num(sx, sy)),
        Minimum => go2!(vx, vy => min8_exact(vx, vy); sx, sy => min_num(sx, sy)),
        Eq => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx == sy)
        ),
        Ne => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_NEQ_UQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx != sy)
        ),
        Gt => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx > sy)
        ),
        Ge => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx >= sy)
        ),
        Lt => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx < sy)
        ),
        Le => go2!(
            vx, vy => _mm256_and_ps(_mm256_cmp_ps::<_CMP_LE_OQ>(vx, vy), _mm256_set1_ps(1.0));
            sx, sy => mask_f32(sx <= sy)
        ),
        Pow | Atan2 => None,
    }
}

// ---------------------------------------------------------------------------
// AVX2 `where3` (`cond != 0 ? a : b`) for F32 and U8 conditions
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_f32_mask8(p: *const f32) -> __m256 {
    // SAFETY: caller guarantees `p` is valid for 8 `f32` reads.
    // `_CMP_NEQ_UQ` matches Rust `c != 0.0` for all values including `-0.0` (false) and `NaN` (true).
    unsafe { _mm256_cmp_ps::<_CMP_NEQ_UQ>(_mm256_loadu_ps(p), _mm256_setzero_ps()) }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_u8_mask8(p: *const u8) -> __m256 {
    // SAFETY: caller guarantees `p` is valid for 8 `u8` reads; `_mm_loadl_epi64` reads exactly 8 bytes.
    let bytes = unsafe { _mm_loadl_epi64(p as *const __m128i) };
    let wide = _mm256_cvtepu8_epi32(bytes);
    _mm256_castsi256_ps(_mm256_cmpgt_epi32(wide, _mm256_setzero_si256()))
}

macro_rules! define_where_kernels {
    ($c_ty:ty, $mask8:ident, $truthy:expr, $k111:ident, $k110:ident, $k101:ident, $k100:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $k111(c: &[$c_ty], a: &[f32], b: &[f32], out: &mut [f32]) {
            let (cp, cr) = c.as_chunks::<16>();
            let (ap, ar) = a.as_chunks::<16>();
            let (bp, br) = b.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (((cc, ac), bc), oc) in cp.iter().zip(ap.iter()).zip(bp.iter()).zip(op.iter_mut()) {
                // SAFETY: 16-element chunks.
                unsafe {
                    let (cp, ap, bp, dp) = (cc.as_ptr(), ac.as_ptr(), bc.as_ptr(), oc.as_mut_ptr());
                    let m0 = $mask8(cp);
                    let m1 = $mask8(cp.add(8));
                    _mm256_storeu_ps(dp, _mm256_blendv_ps(_mm256_loadu_ps(bp), _mm256_loadu_ps(ap), m0));
                    _mm256_storeu_ps(
                        dp.add(8),
                        _mm256_blendv_ps(_mm256_loadu_ps(bp.add(8)), _mm256_loadu_ps(ap.add(8)), m1),
                    );
                }
            }
            let (c8, ct) = cr.as_chunks::<8>();
            let (a8, at) = ar.as_chunks::<8>();
            let (b8, bt) = br.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (((cc, ac), bc), oc) in c8.iter().zip(a8.iter()).zip(b8.iter()).zip(o8.iter_mut()) {
                // SAFETY: 8-element chunks.
                unsafe {
                    let m = $mask8(cc.as_ptr());
                    _mm256_storeu_ps(
                        oc.as_mut_ptr(),
                        _mm256_blendv_ps(_mm256_loadu_ps(bc.as_ptr()), _mm256_loadu_ps(ac.as_ptr()), m),
                    );
                }
            }
            for (((o, &cv), &av), &bv) in ot.iter_mut().zip(ct).zip(at).zip(bt) {
                *o = if ($truthy)(cv) { av } else { bv };
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $k110(c: &[$c_ty], a: &[f32], b: f32, out: &mut [f32]) {
            let bv = _mm256_set1_ps(b);
            let (cp, cr) = c.as_chunks::<16>();
            let (ap, ar) = a.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for ((cc, ac), oc) in cp.iter().zip(ap.iter()).zip(op.iter_mut()) {
                // SAFETY: 16-element chunks.
                unsafe {
                    let (cp, ap, dp) = (cc.as_ptr(), ac.as_ptr(), oc.as_mut_ptr());
                    let m0 = $mask8(cp);
                    let m1 = $mask8(cp.add(8));
                    _mm256_storeu_ps(dp, _mm256_blendv_ps(bv, _mm256_loadu_ps(ap), m0));
                    _mm256_storeu_ps(dp.add(8), _mm256_blendv_ps(bv, _mm256_loadu_ps(ap.add(8)), m1));
                }
            }
            let (c8, ct) = cr.as_chunks::<8>();
            let (a8, at) = ar.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for ((cc, ac), oc) in c8.iter().zip(a8.iter()).zip(o8.iter_mut()) {
                // SAFETY: 8-element chunks.
                unsafe {
                    let m = $mask8(cc.as_ptr());
                    _mm256_storeu_ps(oc.as_mut_ptr(), _mm256_blendv_ps(bv, _mm256_loadu_ps(ac.as_ptr()), m));
                }
            }
            for ((o, &cv), &av) in ot.iter_mut().zip(ct).zip(at) {
                *o = if ($truthy)(cv) { av } else { b };
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $k101(c: &[$c_ty], a: f32, b: &[f32], out: &mut [f32]) {
            let av = _mm256_set1_ps(a);
            let (cp, cr) = c.as_chunks::<16>();
            let (bp, br) = b.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for ((cc, bc), oc) in cp.iter().zip(bp.iter()).zip(op.iter_mut()) {
                // SAFETY: 16-element chunks.
                unsafe {
                    let (cp, bp, dp) = (cc.as_ptr(), bc.as_ptr(), oc.as_mut_ptr());
                    let m0 = $mask8(cp);
                    let m1 = $mask8(cp.add(8));
                    _mm256_storeu_ps(dp, _mm256_blendv_ps(_mm256_loadu_ps(bp), av, m0));
                    _mm256_storeu_ps(dp.add(8), _mm256_blendv_ps(_mm256_loadu_ps(bp.add(8)), av, m1));
                }
            }
            let (c8, ct) = cr.as_chunks::<8>();
            let (b8, bt) = br.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for ((cc, bc), oc) in c8.iter().zip(b8.iter()).zip(o8.iter_mut()) {
                // SAFETY: 8-element chunks.
                unsafe {
                    let m = $mask8(cc.as_ptr());
                    _mm256_storeu_ps(oc.as_mut_ptr(), _mm256_blendv_ps(_mm256_loadu_ps(bc.as_ptr()), av, m));
                }
            }
            for ((o, &cv), &bv) in ot.iter_mut().zip(ct).zip(bt) {
                *o = if ($truthy)(cv) { a } else { bv };
            }
        }

        #[target_feature(enable = "avx2")]
        unsafe fn $k100(c: &[$c_ty], a: f32, b: f32, out: &mut [f32]) {
            let (av, bv) = (_mm256_set1_ps(a), _mm256_set1_ps(b));
            let (cp, cr) = c.as_chunks::<16>();
            let (op, or) = out.as_chunks_mut::<16>();
            for (cc, oc) in cp.iter().zip(op.iter_mut()) {
                // SAFETY: 16-element chunks.
                unsafe {
                    let (cp, dp) = (cc.as_ptr(), oc.as_mut_ptr());
                    let m0 = $mask8(cp);
                    let m1 = $mask8(cp.add(8));
                    _mm256_storeu_ps(dp, _mm256_blendv_ps(bv, av, m0));
                    _mm256_storeu_ps(dp.add(8), _mm256_blendv_ps(bv, av, m1));
                }
            }
            let (c8, ct) = cr.as_chunks::<8>();
            let (o8, ot) = or.as_chunks_mut::<8>();
            for (cc, oc) in c8.iter().zip(o8.iter_mut()) {
                // SAFETY: 8-element chunks.
                unsafe {
                    let m = $mask8(cc.as_ptr());
                    _mm256_storeu_ps(oc.as_mut_ptr(), _mm256_blendv_ps(bv, av, m));
                }
            }
            for (o, &cv) in ot.iter_mut().zip(ct) {
                *o = if ($truthy)(cv) { a } else { b };
            }
        }
    };
}

define_where_kernels!(
    f32,
    load_f32_mask8,
    |v: f32| v != 0.0,
    where_f32_111,
    where_f32_110,
    where_f32_101,
    where_f32_100
);
define_where_kernels!(
    u8,
    load_u8_mask8,
    |v: u8| v != 0,
    where_u8_111,
    where_u8_110,
    where_u8_101,
    where_u8_100
);

type Where111<C> = unsafe fn(&[C], &[f32], &[f32], &mut [f32]);
type Where110<C> = unsafe fn(&[C], &[f32], f32, &mut [f32]);
type Where101<C> = unsafe fn(&[C], f32, &[f32], &mut [f32]);
type Where100<C> = unsafe fn(&[C], f32, f32, &mut [f32]);

struct WhereKernels<C> {
    truthy: fn(C) -> bool,
    k111: Where111<C>,
    k110: Where110<C>,
    k101: Where101<C>,
    k100: Where100<C>,
}

#[inline]
fn where3_avx2_dispatch<C: Copy + Send + Sync>(
    c: &[C],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
    k: WhereKernels<C>,
) -> Vec<f32> {
    if !has_avx2() {
        return where3(c, lc, a, la, b, lb, k.truthy);
    }
    let n = la.numel();
    if n == 0 {
        return Vec::new();
    }
    let mut out = alloc_uninit(n);
    if let (Some(rc), Some(ra), Some(rb)) =
        (lc.contiguous_range(), la.contiguous_range(), lb.contiguous_range())
    {
        let (c, a, b) = (&c[rc], &a[ra], &b[rb]);
        let piece = |chunk: usize, o: &mut [f32]| {
            let r = chunk * CHUNK..chunk * CHUNK + o.len();
            // SAFETY: `has_avx2()` checked above; all slices have length `o.len()`.
            unsafe { (k.k111)(&c[r.clone()], &a[r.clone()], &b[r], o) };
        };
        if n >= PAR_MIN {
            par_chunks(&mut out, CHUNK, &piece);
        } else {
            piece(0, &mut out);
        }
        return out;
    }
    let plan = RowPlan::<3>::new(
        la.dims(),
        [lc.strides(), la.strides(), lb.strides()],
        [lc.offset(), la.offset(), lb.offset()],
    );
    let [sc, sa, sb] = plan.inner;
    fill_rows(&mut out, &plan, |row, [oc, oa, ob]| {
        let len = row.len();
        match (sc, sa, sb) {
            // SAFETY: `has_avx2()` checked above; every slice has length `len`.
            (1, 1, 1) => unsafe { (k.k111)(&c[oc..oc + len], &a[oa..oa + len], &b[ob..ob + len], row) },
            (1, 1, 0) => unsafe { (k.k110)(&c[oc..oc + len], &a[oa..oa + len], b[ob], row) },
            (1, 0, 1) => unsafe { (k.k101)(&c[oc..oc + len], a[oa], &b[ob..ob + len], row) },
            (1, 0, 0) => unsafe { (k.k100)(&c[oc..oc + len], a[oa], b[ob], row) },
            (0, 1, 1) => {
                if (k.truthy)(c[oc]) {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 1, 0) => {
                if (k.truthy)(c[oc]) {
                    row.copy_from_slice(&a[oa..oa + len]);
                } else {
                    row.fill(b[ob]);
                }
            }
            (0, 0, 1) => {
                if (k.truthy)(c[oc]) {
                    row.fill(a[oa]);
                } else {
                    row.copy_from_slice(&b[ob..ob + len]);
                }
            }
            (0, 0, 0) => row.fill(if (k.truthy)(c[oc]) { a[oa] } else { b[ob] }),
            _ => {
                for (i, r) in row.iter_mut().enumerate() {
                    *r = if (k.truthy)(c[oc + i * sc]) { a[oa + i * sa] } else { b[ob + i * sb] };
                }
            }
        }
    });
    out
}

/// AVX2-accelerated `where_cond` for `F32` condition (`c != 0.0`).
pub(super) fn where3_f32_avx2(
    c: &[f32],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
) -> Vec<f32> {
    where3_avx2_dispatch(
        c,
        lc,
        a,
        la,
        b,
        lb,
        WhereKernels {
            truthy: |v| v != 0.0,
            k111: where_f32_111,
            k110: where_f32_110,
            k101: where_f32_101,
            k100: where_f32_100,
        },
    )
}

/// AVX2-accelerated `where_cond` for `U8` condition (`c != 0`).
pub(super) fn where3_u8_avx2(
    c: &[u8],
    lc: &Layout,
    a: &[f32],
    la: &Layout,
    b: &[f32],
    lb: &Layout,
) -> Vec<f32> {
    where3_avx2_dispatch(
        c,
        lc,
        a,
        la,
        b,
        lb,
        WhereKernels {
            truthy: |v| v != 0,
            k111: where_u8_111,
            k110: where_u8_110,
            k101: where_u8_101,
            k100: where_u8_100,
        },
    )
}
