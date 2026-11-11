//! GEMM with an A read in place through row pointers ([`gemm_arows`]), the shape of a conv weight gradient: A's
//! rows are the input's shifted rows and B, the packed output gradient, is small and reused by every row.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use super::kernel::store_block_avx2;
use super::kernel::store_block_generic;
use super::{KC, MR, NR};

/// Micro-kernel over an A read in place: `C[0..mr, 0..nr] (+)= A * B_panel` where row `i` of A is the `kc`
/// floats at `a[i]` (rows past `mr` repeat row 0 and are not stored).
type ARowsKernel = unsafe fn(usize, &[*const f32; MR], *const f32, *mut f32, usize, usize, usize, bool);

fn arows_kernel() -> ARowsKernel {
    static K: OnceLock<ARowsKernel> = OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return arows_kernel_avx2_entry as ARowsKernel;
            }
        }
        arows_kernel_generic as ARowsKernel
    })
}

/// Portable [`ARowsKernel`].
///
/// # Safety
/// Every `a[i]` points to `kc` floats, `b` to `kc * NR` floats, and `c` to an `mr x nr` block with row stride
/// `ldc` that nobody else accesses concurrently.
#[allow(clippy::too_many_arguments)]
unsafe fn arows_kernel_generic(
    kc: usize,
    a: &[*const f32; MR],
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc_c: bool,
) {
    // SAFETY: guaranteed by the caller.
    let b = unsafe { std::slice::from_raw_parts(b, kc * NR) };
    let rows: [&[f32]; MR] = a.map(|p| unsafe { std::slice::from_raw_parts(p, kc) });
    let mut acc = [[0f32; NR]; MR];
    for p in 0..kc {
        let bv: &[f32; NR] = b[p * NR..p * NR + NR].try_into().unwrap();
        for i in 0..MR {
            let av = rows[i][p];
            for j in 0..NR {
                acc[i][j] += av * bv[j];
            }
        }
    }
    // SAFETY: the caller owns the mr x nr block at `c`.
    unsafe { store_block_generic(&mut acc, c, ldc, mr, nr, acc_c, None) }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn arows_kernel_avx2_entry(
    kc: usize,
    a: &[*const f32; MR],
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
) {
    // SAFETY: only selected after runtime detection of avx2 and fma; pointer contract forwarded.
    unsafe { arows_kernel_avx2(kc, a, b, c, ldc, mr, nr, acc) }
}

/// AVX2/FMA [`ARowsKernel`]: `kernel_avx2` with the six A values of each K step broadcast from six rows.
///
/// # Safety
/// As [`arows_kernel_generic`], and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn arows_kernel_avx2(
    kc: usize,
    a: &[*const f32; MR],
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
) {
    use std::arch::x86_64::*;
    let mut c00 = _mm256_setzero_ps();
    let mut c01 = _mm256_setzero_ps();
    let mut c10 = _mm256_setzero_ps();
    let mut c11 = _mm256_setzero_ps();
    let mut c20 = _mm256_setzero_ps();
    let mut c21 = _mm256_setzero_ps();
    let mut c30 = _mm256_setzero_ps();
    let mut c31 = _mm256_setzero_ps();
    let mut c40 = _mm256_setzero_ps();
    let mut c41 = _mm256_setzero_ps();
    let mut c50 = _mm256_setzero_ps();
    let mut c51 = _mm256_setzero_ps();
    let [a0p, a1p, a2p, a3p, a4p, a5p] = *a;
    let mut bp = b;
    // SAFETY: the caller guarantees kc floats at every `a[i]` and kc * NR at `b`, and owns the block at `c`.
    unsafe {
        for p in 0..kc {
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            let a0 = _mm256_broadcast_ss(&*a0p.add(p));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            let a1 = _mm256_broadcast_ss(&*a1p.add(p));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            let a2 = _mm256_broadcast_ss(&*a2p.add(p));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            let a3 = _mm256_broadcast_ss(&*a3p.add(p));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
            let a4 = _mm256_broadcast_ss(&*a4p.add(p));
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);
            let a5 = _mm256_broadcast_ss(&*a5p.add(p));
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
            bp = bp.add(NR);
        }
        let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]];
        store_block_avx2(&rows, c, ldc, mr, nr, acc, None);
    }
}

/// `C (+)= A * B` on the calling thread, for an `m x k` A given by row pointers (row `i` is the `k` contiguous
/// floats at `a[i] + a_off`, read in place, never packed) and a `k x n` B already in packed-panel form (panel
/// `q`, columns `q·NR..q·NR + NR`, at `b[q * b_stride..]`, K row `kk` at `+ kk * NR`, zeros past column `n`),
/// with `C` row-major (row stride `ldc`). Every element of C is summed over `KC` blocks in order (on top of C's
/// old value if `accumulate`).
///
/// This is the shape of a conv weight gradient (`dWᵀ = X_unfolded · gyᵀ`): A's rows are the shifted rows of the
/// input, K runs over output pixels and B, the output gradient, is small and reused by every row of A.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_arows(
    m: usize,
    n: usize,
    k: usize,
    a: &[*const f32],
    a_off: usize,
    b: &[f32],
    b_stride: usize,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
) {
    let panels = n.div_ceil(NR);
    assert!(a.len() >= m && ldc >= n, "gemm_arows: bad geometry");
    assert!(m == 0 || (m - 1) * ldc + n <= c.len(), "gemm_arows: C buffer too small");
    assert!(n == 0 || k == 0 || (panels - 1) * b_stride + k * NR <= b.len(), "gemm_arows: B too small");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        return;
    }
    let kern = arows_kernel();
    let mut kb = 0;
    while kb < k {
        let kc = KC.min(k - kb);
        let acc = accumulate || kb != 0;
        // Each B panel stays in L1 while the A rows (in L2) sweep past it.
        for q in 0..panels {
            let j = q * NR;
            let bp = &b[q * b_stride + kb * NR..][..kc * NR];
            let mut i = 0;
            while i < m {
                let mr = MR.min(m - i);
                // SAFETY: the caller guarantees `k` floats from `a[r] + a_off`; `kb + kc <= k`.
                let rows: [*const f32; MR] =
                    std::array::from_fn(|r| unsafe { a[i + if r < mr { r } else { 0 }].add(a_off + kb) });
                // SAFETY: `bp` holds kc * NR floats, every row `kc` floats from `rows[r]`, and the C block
                // [i..i+mr, j..j+nr] is inside `c` (asserted above), which this call borrows mutably.
                unsafe {
                    kern(kc, &rows, bp.as_ptr(), c.as_mut_ptr().add(i * ldc + j), ldc, mr, NR.min(n - j), acc)
                };
                i += MR;
            }
        }
        kb += KC;
    }
}
