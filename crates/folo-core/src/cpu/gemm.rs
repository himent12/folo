//! Single-precision GEMM: packed, cache-blocked, register-tiled and rayon-parallel.
//!
//! The structure follows BLIS: operands are packed into micro-panels (`MR` rows of A, `NR` columns of B, `KC`
//! deep), and a register-tiled micro-kernel computes an `MR x NR` block of C from one A and one B micro-panel.
//! On x86-64 with AVX2+FMA (detected at runtime) the micro-kernel uses explicit intrinsics; elsewhere a
//! portable kernel the compiler vectorises.
//!
//! Operands are abstract ([`Operand`]): anything that can write its micro-panels. Plain strided matrices cover
//! transposed and non-contiguous views without copies; conv2d supplies an implicit im2col operand so the
//! unfolded matrix is never materialised.
//!
//! Parallelism: C is split into tiles processed by rayon tasks after the panels of a K block are packed in
//! parallel. When C is too small to give every thread a tile but K is long, K is split across threads with
//! private accumulators that are summed at the end.

use std::cell::Cell;
use std::sync::OnceLock;

use rayon::prelude::*;

use super::elementwise::SendPtr;

/// Rows of a micro-tile.
pub const MR: usize = 6;
/// Columns of a micro-tile.
pub const NR: usize = 16;
/// Depth of a packed K block (an `NR x KC` B micro-panel is 16 KiB and stays in L1).
const KC: usize = 256;
/// Packed-buffer budget per K super-block, in floats (16 MiB).
const PACK_BUDGET: usize = 1 << 22;
/// Below this many flops, run on the calling thread.
const PAR_FLOPS: usize = 1 << 21;

/// One GEMM operand seen as an `outer x K` matrix: for A, `outer` is the row index `i` and the element is
/// `A[i, kk]`; for B, `outer` is the column index `j` and the element is `B[kk, j]`.
pub trait Operand: Sync {
    /// Write a micro-panel: `dst[kk * width + ii] = self(o0 + ii, k0 + kk)` for `ii < w`, `kk < kc`, and zeros
    /// for `w <= ii < width`. `dst.len() >= kc * width`.
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]);
}

/// A strided matrix operand: element `(o, kk)` is `data[offset + o * so + kk * sk]`.
#[derive(Clone, Copy, Debug)]
pub struct Strided<'a> {
    /// The buffer.
    pub data: &'a [f32],
    /// Element `(0, 0)`.
    pub offset: usize,
    /// Stride along the outer index.
    pub so: usize,
    /// Stride along K.
    pub sk: usize,
}

impl<'a> Strided<'a> {
    /// The A operand of `C = A B` for a matrix with row stride `rs` and column stride `cs`.
    pub fn lhs(data: &'a [f32], offset: usize, rs: usize, cs: usize) -> Self {
        Strided { data, offset, so: rs, sk: cs }
    }

    /// The B operand of `C = A B` for a matrix with row stride `rs` and column stride `cs`.
    pub fn rhs(data: &'a [f32], offset: usize, rs: usize, cs: usize) -> Self {
        Strided { data, offset, so: cs, sk: rs }
    }
}

impl Operand for Strided<'_> {
    #[inline]
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]) {
        let dst = &mut dst[..kc * width];
        if w < width {
            dst.fill(0.0);
        }
        let base = self.offset + o0 * self.so + k0 * self.sk;
        if self.so == 1 {
            for kk in 0..kc {
                let s = base + kk * self.sk;
                dst[kk * width..kk * width + w].copy_from_slice(&self.data[s..s + w]);
            }
        } else if self.sk == 1 {
            for ii in 0..w {
                let row = &self.data[base + ii * self.so..base + ii * self.so + kc];
                for (kk, &v) in row.iter().enumerate() {
                    dst[kk * width + ii] = v;
                }
            }
        } else {
            for kk in 0..kc {
                for ii in 0..w {
                    dst[kk * width + ii] = self.data[base + ii * self.so + kk * self.sk];
                }
            }
        }
    }
}

type Kernel = unsafe fn(usize, *const f32, *const f32, *mut f32, usize, usize, usize, bool);

fn kernel() -> Kernel {
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

/// Portable micro-kernel: `C[0..mr, 0..nr] (+)= A_panel * B_panel` (accumulate if `acc_c`, else overwrite).
///
/// # Safety
/// `a` points to `kc * MR` floats, `b` to `kc * NR` floats, and `c` to an `mr x nr` block with row stride
/// `ldc` that nobody else accesses concurrently.
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_generic(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc_c: bool,
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
    for (i, row) in acc.iter().enumerate().take(mr) {
        for (j, &v) in row.iter().enumerate().take(nr) {
            // SAFETY: (i, j) is inside the block the caller owns.
            unsafe {
                let p = c.add(i * ldc + j);
                *p = if acc_c { *p + v } else { v };
            }
        }
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
) {
    // SAFETY: only selected after runtime detection of avx2 and fma; pointer contract forwarded.
    unsafe { kernel_avx2(kc, a, b, c, ldc, mr, nr, acc) }
}

/// AVX2/FMA 6x16 micro-kernel: 12 ymm accumulators, 2 for B, 1 broadcast of A.
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
    let mut ap = a;
    let mut bp = b;
    // SAFETY: the caller guarantees kc * MR floats at `a` and kc * NR at `b`.
    unsafe {
        for _ in 0..kc {
            let b0 = _mm256_loadu_ps(bp);
            let b1 = _mm256_loadu_ps(bp.add(8));
            let a0 = _mm256_broadcast_ss(&*ap);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            let a1 = _mm256_broadcast_ss(&*ap.add(1));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            let a2 = _mm256_broadcast_ss(&*ap.add(2));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            let a3 = _mm256_broadcast_ss(&*ap.add(3));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
            let a4 = _mm256_broadcast_ss(&*ap.add(4));
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            c41 = _mm256_fmadd_ps(a4, b1, c41);
            let a5 = _mm256_broadcast_ss(&*ap.add(5));
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            c51 = _mm256_fmadd_ps(a5, b1, c51);
            ap = ap.add(MR);
            bp = bp.add(NR);
        }
        let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]];
        if mr == MR && nr == NR {
            for (i, r) in rows.iter().enumerate() {
                let cp = c.add(i * ldc);
                if acc {
                    _mm256_storeu_ps(cp, _mm256_add_ps(_mm256_loadu_ps(cp), r[0]));
                    _mm256_storeu_ps(cp.add(8), _mm256_add_ps(_mm256_loadu_ps(cp.add(8)), r[1]));
                } else {
                    _mm256_storeu_ps(cp, r[0]);
                    _mm256_storeu_ps(cp.add(8), r[1]);
                }
            }
        } else {
            let mut tmp = [0f32; MR * NR];
            for (i, r) in rows.iter().enumerate() {
                _mm256_storeu_ps(tmp.as_mut_ptr().add(i * NR), r[0]);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(i * NR + 8), r[1]);
            }
            for i in 0..mr {
                for j in 0..nr {
                    let p = c.add(i * ldc + j);
                    *p = if acc { *p + tmp[i * NR + j] } else { tmp[i * NR + j] };
                }
            }
        }
    }
}

thread_local! {
    static PACK_BUF: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// Run `f` with a scratch buffer of at least `len` floats, reused across calls on the same thread. Re-entrant
/// calls (rayon can run another task while this one waits) simply get a fresh buffer.
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    let mut buf = PACK_BUF.with(|b| b.take());
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    let r = f(&mut buf[..len]);
    PACK_BUF.with(|b| {
        let old = b.take();
        b.set(if old.len() > buf.len() { old } else { buf });
    });
    r
}

#[derive(Clone, Copy)]
struct Geometry {
    m: usize,
    n: usize,
    mc: usize,
    nc: usize,
}

impl Geometry {
    fn tiles(&self) -> (usize, usize) {
        (self.m.div_ceil(self.mc), self.n.div_ceil(self.nc))
    }
}

fn choose_tiles(m: usize, n: usize, threads: usize) -> Geometry {
    let mut g = Geometry { m, n, mc: 144, nc: 512 };
    let want = 3 * threads;
    loop {
        let (tm, tn) = g.tiles();
        if tm * tn >= want || (g.mc <= 18 && g.nc <= 64) {
            return g;
        }
        let can_n = g.nc > 64 && g.n > g.nc / 2;
        let can_m = g.mc > 18 && g.m > g.mc / 2;
        if can_n && (g.nc >= 2 * g.mc || !can_m) {
            g.nc /= 2;
        } else if can_m {
            g.mc /= 2;
        } else {
            return g;
        }
    }
}

/// `C = A * B` (or `C += A * B` if `accumulate`) where `A` is `m x k`, `B` is `k x n` and `C` is row-major
/// with row stride `ldc`. Without `accumulate`, the previous contents of C are never read.
///
/// Set `parallel` to false when the caller already runs many GEMMs in parallel.
///
/// # Panics
/// If `c` is too short for an `m x n` block with row stride `ldc`.
#[allow(clippy::too_many_arguments)]
pub fn gemm<A: Operand, B: Operand>(
    m: usize,
    n: usize,
    k: usize,
    a: &A,
    b: &B,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
    parallel: bool,
) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(ldc >= n && c.len() >= (m - 1) * ldc + n, "gemm: C buffer too small");
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        return;
    }
    let threads = if parallel { rayon::current_num_threads() } else { 1 };
    if threads <= 1 || 2 * m * n * k < PAR_FLOPS {
        gemm_blocked(m, n, 0..k, a, b, c, ldc, accumulate, false);
        return;
    }
    let g = choose_tiles(m, n, threads);
    let (tm, tn) = g.tiles();
    if tm * tn < threads && k >= 8 * KC {
        // Split K: private accumulators, summed at the end.
        let parts = threads.min(k / (2 * KC)).max(2);
        let step = k.div_ceil(parts).div_ceil(KC) * KC;
        let partials: Vec<Vec<f32>> = (0..parts)
            .into_par_iter()
            .filter_map(|p| {
                let k0 = p * step;
                if k0 >= k {
                    return None;
                }
                let mut buf = vec![0f32; m * n];
                gemm_blocked(m, n, k0..(k0 + step).min(k), a, b, &mut buf, n, false, false);
                Some(buf)
            })
            .collect();
        for i in 0..m {
            let row = &mut c[i * ldc..i * ldc + n];
            if !accumulate {
                row.fill(0.0);
            }
            for p in &partials {
                for (r, &v) in row.iter_mut().zip(&p[i * n..i * n + n]) {
                    *r += v;
                }
            }
        }
        return;
    }
    gemm_blocked(m, n, 0..k, a, b, c, ldc, accumulate, true);
}

#[allow(clippy::too_many_arguments)]
fn gemm_blocked<A: Operand, B: Operand>(
    m: usize,
    n: usize,
    krange: std::ops::Range<usize>,
    a: &A,
    b: &B,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
    parallel: bool,
) {
    let kern = kernel();
    let k = krange.len();
    let pm = m.div_ceil(MR);
    let pn = n.div_ceil(NR);
    // K super-block: as many KC blocks as fit the pack budget.
    let blocks_fit = (PACK_BUDGET / ((pm * MR + pn * NR) * KC)).max(1);
    let ks = (blocks_fit * KC).min(k.div_ceil(KC) * KC);
    let g = if parallel {
        choose_tiles(m, n, rayon::current_num_threads())
    } else {
        Geometry { m, n, mc: 144, nc: 512 }
    };
    let (tm, tn) = g.tiles();
    let cptr = SendPtr(c.as_mut_ptr());
    let a_len = pm * MR * ks;
    with_scratch(a_len + pn * NR * ks, |buf| {
        let (pa, pb) = buf.split_at_mut(a_len);
        let mut s0 = krange.start;
        while s0 < krange.end {
            let ksz = ks.min(krange.end - s0);
            let pa = &mut pa[..pm * MR * ksz];
            let pb = &mut pb[..pn * NR * ksz];
            let pack_a = |(p, dst): (usize, &mut [f32])| {
                a.pack(p * MR, MR.min(m - p * MR), s0, ksz, MR, dst);
            };
            let pack_b = |(p, dst): (usize, &mut [f32])| {
                b.pack(p * NR, NR.min(n - p * NR), s0, ksz, NR, dst);
            };
            if parallel {
                pa.par_chunks_mut(MR * ksz).enumerate().for_each(pack_a);
                pb.par_chunks_mut(NR * ksz).enumerate().for_each(pack_b);
            } else {
                pa.chunks_mut(MR * ksz).enumerate().for_each(pack_a);
                pb.chunks_mut(NR * ksz).enumerate().for_each(pack_b);
            }
            let (pa, pb) = (&*pa, &*pb);
            let tile = |t: usize| {
                let (ti, tj) = (t / tn, t % tn);
                let (i0, i1) = (ti * g.mc, ((ti + 1) * g.mc).min(m));
                let (j0, j1) = (tj * g.nc, ((tj + 1) * g.nc).min(n));
                let cp = &cptr;
                let mut kb = 0;
                while kb < ksz {
                    let kc = KC.min(ksz - kb);
                    let mut j = j0;
                    while j < j1 {
                        let nr = NR.min(j1 - j);
                        let bpanel = &pb[(j / NR) * NR * ksz + kb * NR..];
                        let mut i = i0;
                        while i < i1 {
                            let mr = MR.min(i1 - i);
                            let apanel = &pa[(i / MR) * MR * ksz + kb * MR..];
                            debug_assert!(apanel.len() >= kc * MR && bpanel.len() >= kc * NR);
                            // SAFETY: panels hold kc*MR / kc*NR floats (packed above); the C block
                            // [i..i+mr, j..j+nr] lies inside `c` (asserted by `gemm`) and belongs to this tile
                            // only, so no other task touches it.
                            let acc = accumulate || s0 != krange.start || kb != 0;
                            unsafe {
                                kern(
                                    kc,
                                    apanel.as_ptr(),
                                    bpanel.as_ptr(),
                                    cp.0.add(i * ldc + j),
                                    ldc,
                                    mr,
                                    nr,
                                    acc,
                                )
                            };
                            i += MR;
                        }
                        j += NR;
                    }
                    kb += KC;
                }
            };
            if parallel {
                (0..tm * tn).into_par_iter().for_each(tile);
            } else {
                (0..tm * tn).for_each(tile);
            }
            s0 += ksz;
        }
    });
}

/// `C = A * B` (overwriting `C`) for strided matrices: `A` is `m x k` with strides `(rsa, csa)`, `B` is `k x n`
/// with strides `(rsb, csb)`, `C` is row-major `m x n` with row stride `ldc`.
#[allow(clippy::too_many_arguments)]
pub fn sgemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    gemm(m, n, k, &Strided::lhs(a, 0, rsa, csa), &Strided::rhs(b, 0, rsb, csb), c, ldc, false, true);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f64;
                for p in 0..k {
                    s += a[i * k + p] as f64 * b[p * n + j] as f64;
                }
                c[i * n + j] = s as f32;
            }
        }
        c
    }

    fn data(len: usize, seed: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn matches_naive_on_odd_sizes() {
        for &(m, n, k) in
            &[(1, 1, 1), (7, 17, 5), (13, 33, 300), (64, 64, 64), (100, 37, 520), (5, 300, 3000)]
        {
            let a = data(m * k, 1);
            let b = data(k * n, 2);
            let want = naive(m, n, k, &a, &b);
            let mut c = vec![0f32; m * n];
            sgemm(m, n, k, &a, k, 1, &b, n, 1, &mut c, n);
            for (x, y) in c.iter().zip(&want) {
                assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()), "{m}x{n}x{k}: {x} vs {y}");
            }
            // Transposed storage of both operands.
            let at: Vec<f32> = (0..k * m).map(|idx| a[(idx % m) * k + idx / m]).collect();
            let bt: Vec<f32> = (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect();
            let mut c2 = vec![0f32; m * n];
            sgemm(m, n, k, &at, 1, m, &bt, 1, k, &mut c2, n);
            for (x, y) in c2.iter().zip(&want) {
                assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()));
            }
        }
    }

    #[test]
    fn generic_kernel_matches() {
        let kc = 9;
        let a = data(kc * MR, 3);
        let b = data(kc * NR, 4);
        let mut c1 = vec![1f32; MR * NR];
        let mut c2 = c1.clone();
        // SAFETY: buffers sized as the kernels require.
        unsafe {
            kernel_generic(kc, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr(), NR, MR, NR, true);
            kernel()(kc, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr(), NR, MR, NR, true);
        }
        for (x, y) in c1.iter().zip(&c2) {
            assert!((x - y).abs() < 1e-4);
        }
    }
}
