//! The packed-panel micro-kernel, `C (+)= A_panel · B_panel` on one `MR x NR` block: a portable version and an
//! AVX2/FMA one chosen at run time, and the block stores (with the [`RegEpi`] epilogue) every kernel shares.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use super::RegAct;
use super::{MR, NR, RegEpi};

pub(super) type Kernel =
    unsafe fn(usize, *const f32, *const f32, *mut f32, usize, usize, usize, bool, Option<&RegEpi>);

pub(super) fn kernel() -> Kernel {
    static K: OnceLock<Kernel> = OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return kernel_avx2_entry as Kernel;
            }
        }
        kernel_generic as Kernel
    })
}

/// True in a wasm build with SIMD128 (`+simd128`, as scripts/build-wasm.sh builds): LLVM vectorises
/// the portable micro-kernel to it. A hand-written SIMD128 kernel measured the same (~20 GFLOP/s on one thread), so
/// there is none.
pub fn uses_simd128() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

/// True if the AVX2/FMA micro-kernel is in use.
pub fn uses_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        kernel() as usize == kernel_avx2_entry as Kernel as usize
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Portable micro-kernel: `C[0..mr, 0..nr] (+)= A_panel * B_panel` (accumulate if `acc_c`, else overwrite),
/// then `epi` on the result.
///
/// # Safety
/// `a` points to `kc * MR` floats, `b` to `kc * NR` floats, and `c` to an `mr x nr` block with row stride
/// `ldc` that nobody else accesses concurrently.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn kernel_generic(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc_c: bool,
    epi: Option<&RegEpi>,
) {
    // SAFETY: guaranteed by the caller.
    let (a, b) = unsafe { (std::slice::from_raw_parts(a, kc * MR), std::slice::from_raw_parts(b, kc * NR)) };
    let mut acc = [[0f32; NR]; MR];
    for p in 0..kc {
        let bv: &[f32; NR] = b[p * NR..p * NR + NR].try_into().unwrap();
        let av: &[f32; MR] = a[p * MR..p * MR + MR].try_into().unwrap();
        for i in 0..MR {
            for j in 0..NR {
                acc[i][j] += av[i] * bv[j];
            }
        }
    }
    // SAFETY: the caller owns the mr x nr block at `c`.
    unsafe { store_block_generic(&mut acc, c, ldc, mr, nr, acc_c, epi) }
}

/// Store a finished `MR x NR` register block into the `mr x nr` block at `c`: `C (+)= acc`, then `epi`.
///
/// # Safety
/// As [`kernel_generic`] for `c`.
#[inline]
pub(super) unsafe fn store_block_generic(
    acc: &mut [[f32; NR]; MR],
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc_c: bool,
    epi: Option<&RegEpi>,
) {
    for (i, row) in acc.iter_mut().enumerate().take(mr) {
        let row = &mut row[..nr];
        if acc_c {
            for (j, v) in row.iter_mut().enumerate() {
                // SAFETY: (i, j) is inside the block the caller owns.
                *v += unsafe { *c.add(i * ldc + j) };
            }
        }
        if let Some(e) = epi {
            e.apply_row(i, row);
        }
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping(row.as_ptr(), c.add(i * ldc), nr) };
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_avx2_entry(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    // SAFETY: only selected after runtime detection of avx2 and fma; pointer contract forwarded.
    unsafe { kernel_avx2(kc, a, b, c, ldc, mr, nr, acc, epi) }
}

/// AVX2/FMA micro-kernel with 4x K-unrolling (preserving exact sequential K accumulation order) and
/// specialized 2x16 (`mr <= 2`), 4x16 (`mr <= 4`), and 6x16 (`mr <= 6`) register tiles.
///
/// # Safety
/// As [`kernel_generic`], and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_avx2(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    use std::arch::x86_64::*;
    if mr == 0 || nr == 0 {
        return;
    }
    let mut ap = a;
    let mut bp = b;
    let k4 = kc / 4;
    let kr = kc % 4;

    // SAFETY: the caller guarantees `kc * MR` floats at `a`, `kc * NR` floats at `b`, and owns `mr x nr` at `c`.
    unsafe {
        if mr <= 2 {
            let (mut c00, mut c01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            macro_rules! step2 {
                ($k:expr) => {{
                    let b0 = _mm256_loadu_ps(bp.add($k * NR));
                    let b1 = _mm256_loadu_ps(bp.add($k * NR + 8));
                    let a0 = _mm256_broadcast_ss(&*ap.add($k * MR));
                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c01 = _mm256_fmadd_ps(a0, b1, c01);
                    let a1 = _mm256_broadcast_ss(&*ap.add($k * MR + 1));
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c11 = _mm256_fmadd_ps(a1, b1, c11);
                }};
            }
            for _ in 0..k4 {
                step2!(0);
                step2!(1);
                step2!(2);
                step2!(3);
                ap = ap.add(4 * MR);
                bp = bp.add(4 * NR);
            }
            for _ in 0..kr {
                step2!(0);
                ap = ap.add(MR);
                bp = bp.add(NR);
            }
            let rows = [[c00, c01], [c10, c11]];
            store_block_avx2(&rows, c, ldc, mr, nr, acc, epi);
            return;
        }
        if mr <= 4 {
            let (mut c00, mut c01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut c20, mut c21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let (mut c30, mut c31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            macro_rules! step4 {
                ($k:expr) => {{
                    let b0 = _mm256_loadu_ps(bp.add($k * NR));
                    let b1 = _mm256_loadu_ps(bp.add($k * NR + 8));
                    let a0 = _mm256_broadcast_ss(&*ap.add($k * MR));
                    c00 = _mm256_fmadd_ps(a0, b0, c00);
                    c01 = _mm256_fmadd_ps(a0, b1, c01);
                    let a1 = _mm256_broadcast_ss(&*ap.add($k * MR + 1));
                    c10 = _mm256_fmadd_ps(a1, b0, c10);
                    c11 = _mm256_fmadd_ps(a1, b1, c11);
                    let a2 = _mm256_broadcast_ss(&*ap.add($k * MR + 2));
                    c20 = _mm256_fmadd_ps(a2, b0, c20);
                    c21 = _mm256_fmadd_ps(a2, b1, c21);
                    let a3 = _mm256_broadcast_ss(&*ap.add($k * MR + 3));
                    c30 = _mm256_fmadd_ps(a3, b0, c30);
                    c31 = _mm256_fmadd_ps(a3, b1, c31);
                }};
            }
            for _ in 0..k4 {
                step4!(0);
                step4!(1);
                step4!(2);
                step4!(3);
                ap = ap.add(4 * MR);
                bp = bp.add(4 * NR);
            }
            for _ in 0..kr {
                step4!(0);
                ap = ap.add(MR);
                bp = bp.add(NR);
            }
            let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31]];
            store_block_avx2(&rows, c, ldc, mr, nr, acc, epi);
            return;
        }
        let (mut c00, mut c01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c10, mut c11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c20, mut c21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c30, mut c31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c40, mut c41) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let (mut c50, mut c51) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        macro_rules! step6 {
            ($k:expr) => {{
                let b0 = _mm256_loadu_ps(bp.add($k * NR));
                let b1 = _mm256_loadu_ps(bp.add($k * NR + 8));
                let a0 = _mm256_broadcast_ss(&*ap.add($k * MR));
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                let a1 = _mm256_broadcast_ss(&*ap.add($k * MR + 1));
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                let a2 = _mm256_broadcast_ss(&*ap.add($k * MR + 2));
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
                let a3 = _mm256_broadcast_ss(&*ap.add($k * MR + 3));
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
                let a4 = _mm256_broadcast_ss(&*ap.add($k * MR + 4));
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);
                let a5 = _mm256_broadcast_ss(&*ap.add($k * MR + 5));
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
            }};
        }
        for _ in 0..k4 {
            step6!(0);
            step6!(1);
            step6!(2);
            step6!(3);
            ap = ap.add(4 * MR);
            bp = bp.add(4 * NR);
        }
        for _ in 0..kr {
            step6!(0);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]];
        store_block_avx2(&rows, c, ldc, mr, nr, acc, epi);
    }
}

/// `epi` on row `i` of a finished block, in registers: `+ bias[i]`, then the fused activation matching
/// [`crate::cpu::elementwise::bias_act_inplace`] bit for bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn finish_avx2(v: std::arch::x86_64::__m256, epi: Option<&RegEpi>, i: usize) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    match epi {
        None => v,
        Some(e) => {
            let x = _mm256_add_ps(v, _mm256_set1_ps(e.bias[i]));
            match e.act {
                RegAct::None => x,
                RegAct::Silu => crate::cpu::elementwise::silu8(x),
                // `vmaxps(x, +0.0)` returns `x > +0.0 ? x : +0.0`, matching `f32::max(x, 0.0)` on all values
                // including `-0.0` and `NaN`.
                RegAct::Relu => _mm256_max_ps(x, _mm256_setzero_ps()),
                RegAct::LeakyRelu(a) => {
                    let pos = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
                    let neg = _mm256_mul_ps(_mm256_set1_ps(a), x);
                    _mm256_blendv_ps(neg, x, pos)
                }
                // `vminps(6.0, vmaxps(0.0, y))` preserves NaN and matches `y.clamp(0.0, 6.0)` bit for bit.
                RegAct::Hardswish => {
                    let y = _mm256_add_ps(x, _mm256_set1_ps(3.0));
                    let c = _mm256_min_ps(_mm256_set1_ps(6.0), _mm256_max_ps(_mm256_setzero_ps(), y));
                    _mm256_mul_ps(_mm256_mul_ps(x, c), _mm256_set1_ps(1.0 / 6.0))
                }
            }
        }
    }
}

/// Store a finished `rows.len() x 16` register block into the `mr x nr` block at `c`: `C (+)= rows`, then `epi`.
///
/// # Safety
/// As [`kernel_generic`] for `c`, `mr <= rows.len()`, `nr <= NR`, and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub(super) unsafe fn store_block_avx2(
    rows: &[[std::arch::x86_64::__m256; 2]],
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    use std::arch::x86_64::*;
    assert!(mr <= rows.len() && mr <= MR && nr <= NR);
    // SAFETY: every access is inside the mr x nr block at `c` (the caller's) or the local `tmp`.
    unsafe {
        if nr == NR {
            for (i, r) in rows.iter().enumerate().take(mr) {
                let cp = c.add(i * ldc);
                let (mut v0, mut v1) = (r[0], r[1]);
                if acc {
                    v0 = _mm256_add_ps(_mm256_loadu_ps(cp), v0);
                    v1 = _mm256_add_ps(_mm256_loadu_ps(cp.add(8)), v1);
                }
                _mm256_storeu_ps(cp, finish_avx2(v0, epi, i));
                _mm256_storeu_ps(cp.add(8), finish_avx2(v1, epi, i));
            }
        } else {
            // A ragged block goes through a full-width row buffer, so its values see the same lanes.
            let mut tmp = [0f32; NR];
            let t = tmp.as_mut_ptr();
            for (i, r) in rows.iter().enumerate().take(mr) {
                let (mut v0, mut v1) = (r[0], r[1]);
                if acc {
                    std::ptr::copy_nonoverlapping(c.add(i * ldc), t, nr);
                    v0 = _mm256_add_ps(_mm256_loadu_ps(t), v0);
                    v1 = _mm256_add_ps(_mm256_loadu_ps(t.add(8)), v1);
                }
                _mm256_storeu_ps(t, finish_avx2(v0, epi, i));
                _mm256_storeu_ps(t.add(8), finish_avx2(v1, epi, i));
                std::ptr::copy_nonoverlapping(t, c.add(i * ldc), nr);
            }
        }
    }
}
