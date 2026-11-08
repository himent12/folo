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

use std::cell::RefCell;
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
        // wasm has no runtime feature detection: a module built with simd128 (scripts/build-wasm.sh) uses
        // it unconditionally, as every current browser supports it.
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            return kernel_simd128 as Kernel;
        }
        #[allow(unreachable_code)]
        {
            kernel_generic as Kernel
        }
    })
}

/// True if the wasm SIMD128 micro-kernel is in use (a wasm build with `+simd128`).
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

/// wasm SIMD128 6x16 micro-kernel. Browsers run wasm on 16 vector registers, so the 6x16 block is done as
/// two 6x8 halves (12 `f32x4` accumulators, 2 for B, 1 broadcast of A each) over the same L1-resident A
/// panel instead of 24 accumulators that would spill. Base SIMD128 has no FMA: multiply, then add.
///
/// # Safety
/// As [`kernel_generic`].
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn kernel_simd128(
    kc: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
) {
    use std::arch::wasm32::*;
    let mut tmp = [0f32; MR * NR];
    let full = mr == MR && nr == NR;
    for half in 0..2 {
        let col = half * 8;
        if col >= nr {
            break;
        }
        let z = f32x4_splat(0.0);
        let mut r = [[z; 2]; MR];
        // SAFETY: the caller guarantees kc * MR floats at `a` and kc * NR at `b`; `col + 8 <= NR`.
        unsafe {
            let mut ap = a;
            let mut bp = b.add(col);
            for _ in 0..kc {
                let b0 = v128_load(bp as *const v128);
                let b1 = v128_load(bp.add(4) as *const v128);
                for (i, ri) in r.iter_mut().enumerate() {
                    let ai = f32x4_splat(*ap.add(i));
                    ri[0] = f32x4_add(ri[0], f32x4_mul(ai, b0));
                    ri[1] = f32x4_add(ri[1], f32x4_mul(ai, b1));
                }
                ap = ap.add(MR);
                bp = bp.add(NR);
            }
            if full {
                for (i, ri) in r.iter().enumerate() {
                    let cp = c.add(i * ldc + col) as *mut v128;
                    let cq = c.add(i * ldc + col + 4) as *mut v128;
                    if acc {
                        v128_store(cp, f32x4_add(v128_load(cp), ri[0]));
                        v128_store(cq, f32x4_add(v128_load(cq), ri[1]));
                    } else {
                        v128_store(cp, ri[0]);
                        v128_store(cq, ri[1]);
                    }
                }
            } else {
                for (i, ri) in r.iter().enumerate() {
                    v128_store(tmp.as_mut_ptr().add(i * NR + col) as *mut v128, ri[0]);
                    v128_store(tmp.as_mut_ptr().add(i * NR + col + 4) as *mut v128, ri[1]);
                }
            }
        }
    }
    if !full {
        for i in 0..mr {
            for j in 0..nr {
                // SAFETY: (i, j) is inside the block the caller owns.
                unsafe {
                    let p = c.add(i * ldc + j);
                    *p = if acc { *p + tmp[i * NR + j] } else { tmp[i * NR + j] };
                }
            }
        }
    }
}

thread_local! {
    static PACK_BUFS: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// Scratch buffers kept per thread. Re-entrant calls (rayon runs other tasks, including other GEMMs, on a thread
/// while one of its tasks waits) take further buffers from the same stack instead of allocating.
const PACK_BUFS_KEPT: usize = 4;

/// Run `f` with a scratch buffer of at least `len` floats, reused across calls on the same thread.
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    // The smallest kept buffer that is large enough, else the largest (grown below).
    let mut buf = PACK_BUFS
        .with(|b| {
            let mut b = b.borrow_mut();
            let fit = (0..b.len()).filter(|&i| b[i].len() >= len).min_by_key(|&i| b[i].len());
            let pick = fit.or_else(|| (0..b.len()).max_by_key(|&i| b[i].len()));
            pick.map(|i| b.swap_remove(i))
        })
        .unwrap_or_default();
    if buf.len() < len {
        buf.resize(len, 0.0);
    }
    let r = f(&mut buf[..len]);
    PACK_BUFS.with(|b| {
        let mut b = b.borrow_mut();
        b.push(buf);
        if b.len() > PACK_BUFS_KEPT {
            // Drop the smallest.
            let i = (0..b.len()).min_by_key(|&i| b[i].len()).unwrap_or(0);
            b.swap_remove(i);
        }
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
    let threads = if parallel { rayon::current_num_threads() } else { 1 };
    gemm_ex(m, n, k, a, b, c, ldc, accumulate, threads, None);
}

/// Applied to every finished row segment of C, while it is still in cache: `f(i, j0, &mut C[i, j0..j1])`.
/// Each element of C is passed exactly once, after its last K block.
pub(crate) type Epilogue<'a> = &'a (dyn Fn(usize, usize, &mut [f32]) + Sync);

/// [`gemm`] with a thread budget (`1` = the calling thread only; the tiling is sized for `threads` tasks, so
/// GEMMs that already run side by side can share the pool) and an optional [`Epilogue`].
///
/// The result does not depend on `threads`: every element of C is computed by one task in a fixed K order, and
/// the K split (see [`k_split_parts`]) is a function of the shape only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_ex<A: Operand, B: Operand>(
    m: usize,
    n: usize,
    k: usize,
    a: &A,
    b: &B,
    c: &mut [f32],
    ldc: usize,
    accumulate: bool,
    threads: usize,
    epi: Option<Epilogue<'_>>,
) {
    if m == 0 || n == 0 {
        return;
    }
    let needed = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n));
    assert!(ldc >= n && needed.is_some_and(|e| e <= c.len()), "gemm: C buffer too small");
    let finish_rows = |c: &mut [f32]| {
        if let Some(f) = epi {
            for i in 0..m {
                f(i, 0, &mut c[i * ldc..i * ldc + n]);
            }
        }
    };
    if k == 0 {
        if !accumulate {
            for i in 0..m {
                c[i * ldc..i * ldc + n].fill(0.0);
            }
        }
        finish_rows(c);
        return;
    }
    let parts = k_split_parts(m, n, k);
    if parts > 1 {
        // Split K: private accumulators, summed in part order. The split depends only on (m, n, k), never on
        // the thread budget, so the result is bitwise the same however it is scheduled.
        let step = k.div_ceil(parts).div_ceil(KC) * KC;
        let part = |p: usize| {
            let k0 = p * step;
            let mut buf = super::alloc_uninit(m * n);
            gemm_blocked(m, n, k0..(k0 + step).min(k), a, b, &mut buf, n, false, 1, None);
            buf
        };
        let parts = k.div_ceil(step);
        let partials: Vec<Vec<f32>> = if threads > 1 {
            (0..parts).into_par_iter().map(part).collect()
        } else {
            (0..parts).map(part).collect()
        };
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
        partials.into_iter().for_each(super::recycle);
        finish_rows(c);
        return;
    }
    let threads = if 2 * m * n * k < PAR_FLOPS { 1 } else { threads };
    gemm_blocked(m, n, 0..k, a, b, c, ldc, accumulate, threads, epi);
}

/// Number of K parts for a GEMM whose C is too small to keep every thread busy with tiles but whose K is long
/// (conv weight gradients: K = N·OH·OW), or 1. A function of the shape only, for reproducibility.
fn k_split_parts(m: usize, n: usize, k: usize) -> usize {
    if k < 8 * KC || 2 * m * n * k < PAR_FLOPS {
        return 1;
    }
    let (tm, tn) = choose_tiles(m, n, K_SPLIT_THREADS).tiles();
    if tm * tn >= K_SPLIT_THREADS {
        return 1;
    }
    (k / (2 * KC)).clamp(2, K_SPLIT_THREADS)
}

/// The nominal thread count the K-split decision is made for (see [`k_split_parts`]).
const K_SPLIT_THREADS: usize = 32;

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
    threads: usize,
    epi: Option<Epilogue<'_>>,
) {
    let kern = kernel();
    let k = krange.len();
    let pm = m.div_ceil(MR);
    let pn = n.div_ceil(NR);
    let parallel = threads > 1;
    // K super-block: as many KC blocks as fit the pack budget.
    let blocks_fit = (PACK_BUDGET / ((pm * MR + pn * NR) * KC)).max(1);
    let ks = (blocks_fit * KC).min(k.div_ceil(KC) * KC);
    let g = if parallel { choose_tiles(m, n, threads) } else { Geometry { m, n, mc: 144, nc: 512 } };
    let (tm, tn) = g.tiles();
    let cptr = SendPtr(c.as_mut_ptr());
    let a_len = pm * MR * ks;
    with_scratch(a_len + pn * NR * ks, |buf| {
        let (pa, pb) = buf.split_at_mut(a_len);
        let mut s0 = krange.start;
        while s0 < krange.end {
            let ksz = ks.min(krange.end - s0);
            let last = s0 + ksz == krange.end;
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
                            // [i..i+mr, j..j+nr] lies inside `c` (asserted by `gemm_ex`) and belongs to this
                            // tile only, so no other task touches it.
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
                if let (true, Some(f)) = (last, epi) {
                    for i in i0..i1 {
                        // SAFETY: row segment [i, j0..j1] is inside `c` and owned by this tile (see above); no
                        // other reference to it exists while `f` runs.
                        let row = unsafe { std::slice::from_raw_parts_mut(cp.0.add(i * ldc + j0), j1 - j0) };
                        f(i, j0, row);
                    }
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
    #[should_panic(expected = "C buffer too small")]
    fn rejects_overflowing_ldc() {
        let (a, b) = ([1f32], [1f32]);
        let mut c = [0f32];
        gemm(3, 1, 1, &Strided::lhs(&a, 0, 0, 0), &Strided::rhs(&b, 0, 0, 0), &mut c, 1 << 63, false, false);
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
