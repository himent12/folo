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
use crate::backend::UnaryOp;

/// Rows of a micro-tile.
pub const MR: usize = 6;
/// Columns of a micro-tile.
pub const NR: usize = 16;
/// Depth of a packed K block (an `NR x KC` B micro-panel is 16 KiB and stays in L1).
pub(crate) const KC: usize = 256;
/// Packed-buffer budget per K super-block, in floats (16 MiB).
const PACK_BUDGET: usize = 1 << 22;
/// Below this many flops, run on the calling thread.
pub(crate) const PAR_FLOPS: usize = 1 << 21;

/// One GEMM operand seen as an `outer x K` matrix: for A, `outer` is the row index `i` and the element is
/// `A[i, kk]`; for B, `outer` is the column index `j` and the element is `B[kk, j]`.
pub trait Operand: Sync {
    /// Write a micro-panel: `dst[kk * width + ii] = self(o0 + ii, k0 + kk)` for `ii < w`, `kk < kc`, and zeros
    /// for `w <= ii < width`. `dst.len() >= kc * width`.
    fn pack(&self, o0: usize, w: usize, k0: usize, kc: usize, width: usize, dst: &mut [f32]);

    /// Write the `w.div_ceil(NR)` B micro-panels of outer indices `o0..o0 + w`, panel `q` at
    /// `dst[q * NR * kc..]`, as [`Operand::pack`] with `width = NR` would one panel at a time. Operands whose
    /// K rows are contiguous along the outer index override it to read each row once, front to back (a stream
    /// the hardware prefetcher follows), instead of one cache line from each of `kc` rows per panel.
    fn pack_panels(&self, o0: usize, w: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        for q in 0..w.div_ceil(NR) {
            let j = q * NR;
            self.pack(o0 + j, NR.min(w - j), k0, kc, NR, &mut dst[q * NR * kc..(q + 1) * NR * kc]);
        }
    }
}

/// [`Operand::pack_panels`] for an operand whose K row `kk` is the contiguous slice `row(kk)` of its `w` outer
/// elements: each row is read once, front to back, and scattered over the panels.
#[inline]
pub(crate) fn pack_rows_into_panels<'r>(
    w: usize,
    kc: usize,
    dst: &mut [f32],
    row: impl Fn(usize) -> &'r [f32],
) {
    let panels = w.div_ceil(NR);
    let dst = &mut dst[..panels * NR * kc];
    let full = w / NR;
    for kk in 0..kc {
        let src = &row(kk)[..w];
        for q in 0..full {
            dst[q * NR * kc + kk * NR..][..NR].copy_from_slice(&src[q * NR..q * NR + NR]);
        }
        if full < panels {
            let tail = w - full * NR;
            let d = &mut dst[full * NR * kc + kk * NR..][..NR];
            d[..tail].copy_from_slice(&src[full * NR..]);
            d[tail..].fill(0.0);
        }
    }
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

    fn pack_panels(&self, o0: usize, w: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        if self.so != 1 {
            for q in 0..w.div_ceil(NR) {
                let j = q * NR;
                self.pack(o0 + j, NR.min(w - j), k0, kc, NR, &mut dst[q * NR * kc..(q + 1) * NR * kc]);
            }
            return;
        }
        let base = self.offset + o0 + k0 * self.sk;
        pack_rows_into_panels(w, kc, dst, |kk| &self.data[base + kk * self.sk..base + kk * self.sk + w]);
    }
}

/// What a micro-kernel applies to a finished block before storing it (see [`BiasAct`]): row `i` gets
/// `+ bias[i]`, then SiLU if `silu`.
#[derive(Clone, Copy, Debug)]
struct RegEpi {
    bias: [f32; MR],
    silu: bool,
}

impl RegEpi {
    /// Row `i` of a finished block, on the scalar path: [`super::elementwise::bias_act_inplace`]'s arithmetic.
    #[inline]
    fn apply_row(&self, i: usize, v: &mut [f32]) {
        if self.silu {
            super::elementwise::silu_bias_inplace(v, self.bias[i]);
        } else {
            v.iter_mut().for_each(|x| *x += self.bias[i]);
        }
    }
}

type Kernel = unsafe fn(usize, *const f32, *const f32, *mut f32, usize, usize, usize, bool, Option<&RegEpi>);

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
unsafe fn kernel_generic(
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
unsafe fn store_block_generic(
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
    epi: Option<&RegEpi>,
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
    // SAFETY: the caller guarantees kc * MR floats at `a` and kc * NR at `b`, and owns the block at `c`.
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
        store_block_avx2(&rows, c, ldc, mr, nr, acc, epi);
    }
}

/// `epi` on row `i` of a finished block, in registers: `+ bias[i]`, then SiLU through the lanes
/// [`super::elementwise::silu_bias_inplace`] uses, so the result is bit for bit the row pass's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn finish_avx2(v: std::arch::x86_64::__m256, epi: Option<&RegEpi>, i: usize) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    match epi {
        None => v,
        Some(e) => {
            let x = _mm256_add_ps(v, _mm256_set1_ps(e.bias[i]));
            if e.silu { super::elementwise::silu8(x) } else { x }
        }
    }
}

/// Store a finished 6x16 register block into the `mr x nr` block at `c`: `C (+)= rows`, then `epi`.
///
/// # Safety
/// As [`kernel_generic`] for `c`, and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_block_avx2(
    rows: &[[std::arch::x86_64::__m256; 2]; MR],
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    use std::arch::x86_64::*;
    // SAFETY: every access is inside the mr x nr block at `c` (the caller's) or the local `tmp`.
    unsafe {
        if mr == MR && nr == NR {
            for (i, r) in rows.iter().enumerate() {
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

#[derive(Clone, Copy, Debug)]
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

/// Widest column tile: its packed `KC x NC` B block (512 KiB) stays in L2 while the A panels stream past it.
const NC_MAX: usize = 32 * NR;
/// Smallest amount of work (flops) worth a task of its own; below `threads` of these the pool is not filled.
const TASK_FLOPS: usize = 1 << 20;

/// Tile sizes for [`gemm_blocked`]: column tiles first (every task packs the B block of its own columns, so
/// splitting rows packs the same B more than once), rows only when there are too few columns to give every
/// task a tile. Aims at about three tasks per thread so work stealing can even out fast and slow cores, but
/// never at tasks smaller than [`TASK_FLOPS`].
fn plan_tiles(m: usize, n: usize, k: usize, threads: usize) -> Geometry {
    let flops = 2 * m * n * k;
    let want = if threads > 1 { (3 * threads).min(flops / TASK_FLOPS).max(1) } else { 1 };
    let panels = n.div_ceil(NR);
    let nc = (panels.div_ceil(want) * NR).min(NC_MAX);
    let tn = n.div_ceil(nc);
    let mc = if tn >= want { m } else { (m.div_ceil(want.div_ceil(tn))).div_ceil(MR) * MR };
    Geometry { m, n, mc: mc.clamp(MR, m.max(MR)), nc }
}

/// Floats of packed A from which the panels are packed in parallel, and about how many each task packs.
const PACK_A_PAR_MIN: usize = 1 << 17;
const PACK_A_TASK: usize = 1 << 15;

/// Pack the A panels (`panel` floats each) of `pa` with `pack(index, panel)`: in parallel, several panels a
/// task, when there is enough of it.
fn pack_lhs_panels(pa: &mut [f32], panel: usize, parallel: bool, pack: impl Fn((usize, &mut [f32])) + Sync) {
    if parallel && pa.len() >= PACK_A_PAR_MIN {
        let per_task = PACK_A_TASK.div_ceil(panel);
        pa.par_chunks_mut(per_task * panel).enumerate().for_each(|(t, chunk)| {
            chunk.chunks_mut(panel).enumerate().for_each(|(p, dst)| pack((t * per_task + p, dst)))
        });
    } else {
        pa.chunks_mut(panel).enumerate().for_each(pack);
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

/// Bias and activation applied to every finished element of C, once, after its last K block: row `i` becomes
/// `act(C[i, j] + bias[i])`, exactly as [`super::elementwise::bias_act_inplace`] computes it (no bias: 0).
///
/// The identity and SiLU are applied by the micro-kernel to its accumulators before they are stored (through
/// the same lanes as the row pass, so bit for bit its result); other activations run as a pass over each
/// finished row segment while it is in cache.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BiasAct<'a> {
    /// One value per row of C.
    pub bias: Option<&'a [f32]>,
    pub act: Option<UnaryOp>,
}

impl BiasAct<'_> {
    /// Apply to `v`, a run of row `i`.
    pub(crate) fn apply(&self, i: usize, v: &mut [f32]) {
        super::elementwise::bias_act_inplace(self.act, self.bias.map_or(0.0, |b| b[i]), v);
    }

    /// The same epilogue for the rows from `i0` on (row `i0` becomes row 0).
    pub(crate) fn rows_from(self, i0: usize) -> Self {
        BiasAct { bias: self.bias.map(|b| &b[i0..]), act: self.act }
    }

    /// True if the micro-kernels apply it (else it is a pass over the finished rows).
    fn in_kernel(&self) -> bool {
        matches!(self.act, None | Some(UnaryOp::Silu))
    }

    /// The micro-kernel form for the `mr` rows from `i` (only if [`BiasAct::in_kernel`]).
    fn kernel_rows(&self, i: usize, mr: usize) -> RegEpi {
        let silu = self.act == Some(UnaryOp::Silu);
        let mut bias = [0f32; MR];
        if let Some(b) = self.bias {
            bias[..mr].copy_from_slice(&b[i..i + mr]);
        }
        if !silu {
            // `bias_act_inplace` skips a zero bias when there is no activation, and `x + -0.0` is `x` bit for
            // bit (`-0.0 + 0.0` would be `+0.0`).
            bias.iter_mut().filter(|v| **v == 0.0).for_each(|v| *v = -0.0);
        }
        RegEpi { bias, silu }
    }
}

/// [`gemm`] with a thread budget (`1` = the calling thread only; the tiling is sized for `threads` tasks, so
/// GEMMs that already run side by side can share the pool) and an optional [`BiasAct`] epilogue.
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
    epi: Option<BiasAct<'_>>,
) {
    if m == 0 || n == 0 {
        return;
    }
    let needed = (m - 1).checked_mul(ldc).and_then(|v| v.checked_add(n));
    assert!(ldc >= n && needed.is_some_and(|e| e <= c.len()), "gemm: C buffer too small");
    let finish_rows = |c: &mut [f32]| {
        if let Some(e) = epi {
            for i in 0..m {
                e.apply(i, &mut c[i * ldc..i * ldc + n]);
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

/// The 64-byte-aligned start of a scratch buffer of at least `len + 16` floats, as a `len`-float slice.
fn aligned(buf: &mut [f32], len: usize) -> &mut [f32] {
    let off = buf.as_ptr().align_offset(64).min(16);
    &mut buf[off..off + len]
}

/// The blocked GEMM over `krange`. A is packed once per K super-block (as much K as fits [`PACK_BUDGET`]) and
/// shared by all tasks. Each task owns a tile of C and, for every `KC` block, packs the B panels of its own
/// columns into a thread-local, L2-sized buffer, then sweeps the A panels past each B panel (which stays in
/// L1). For conv2d this expands the implicit im2col operand one small block at a time, while it is in cache,
/// instead of materialising the whole unfolded matrix first.
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
    epi: Option<BiasAct<'_>>,
) {
    let kern = kernel();
    let k = krange.len();
    let pm = m.div_ceil(MR);
    let reg = epi.filter(BiasAct::in_kernel);
    // K super-block: as many KC blocks of packed A as fit the pack budget.
    let blocks_fit = (PACK_BUDGET / (pm * MR * KC)).max(1);
    let ks = (blocks_fit * KC).min(k.div_ceil(KC) * KC);
    let g = plan_tiles(m, n, k, threads);
    let (tm, tn) = g.tiles();
    let parallel = threads > 1 && tm * tn > 1;
    let cptr = SendPtr(c.as_mut_ptr());
    let a_len = pm * MR * ks;
    with_scratch(a_len + 16, |abuf| {
        let pa_all = aligned(abuf, a_len);
        let mut s0 = krange.start;
        while s0 < krange.end {
            let ksz = ks.min(krange.end - s0);
            let last = s0 + ksz == krange.end;
            let first = s0 == krange.start;
            let pa = &mut pa_all[..pm * MR * ksz];
            let pack_a = |(p, dst): (usize, &mut [f32])| {
                a.pack(p * MR, MR.min(m - p * MR), s0, ksz, MR, dst);
            };
            pack_lhs_panels(pa, MR * ksz, parallel, pack_a);
            let pa = &*pa;
            let tile = |t: usize| {
                let (ti, tj) = (t / tn, t % tn);
                let (i0, i1) = (ti * g.mc, ((ti + 1) * g.mc).min(m));
                let (j0, j1) = (tj * g.nc, ((tj + 1) * g.nc).min(n));
                let cp = &cptr;
                let panels = (j1 - j0).div_ceil(NR);
                let b_len = panels * NR * KC.min(ksz);
                with_scratch(b_len + 16, |bbuf| {
                    let pb = aligned(bbuf, b_len);
                    let mut kb = 0;
                    while kb < ksz {
                        let kc = KC.min(ksz - kb);
                        b.pack_panels(j0, j1 - j0, s0 + kb, kc, pb);
                        let acc = accumulate || !first || kb != 0;
                        let finished = last && kb + kc == ksz;
                        for q in 0..panels {
                            let j = j0 + q * NR;
                            let nr = NR.min(j1 - j);
                            let bpanel = &pb[q * NR * kc..(q + 1) * NR * kc];
                            let mut i = i0;
                            while i < i1 {
                                let mr = MR.min(i1 - i);
                                let apanel = &pa[(i / MR) * MR * ksz + kb * MR..];
                                debug_assert!(apanel.len() >= kc * MR);
                                let re = reg.filter(|_| finished).map(|e| e.kernel_rows(i, mr));
                                // SAFETY: the panels hold kc*MR / kc*NR floats (packed above); the C block
                                // [i..i+mr, j..j+nr] lies inside `c` (asserted by `gemm_ex`) and belongs to
                                // this tile only, so no other task touches it.
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
                                        re.as_ref(),
                                    )
                                };
                                i += MR;
                            }
                        }
                        kb += KC;
                    }
                });
                if let (true, Some(e), None) = (last, epi, reg) {
                    for i in i0..i1 {
                        // SAFETY: row segment [i, j0..j1] is inside `c` and owned by this tile (see above); no
                        // other reference to it exists while `e` runs.
                        let row = unsafe { std::slice::from_raw_parts_mut(cp.0.add(i * ldc + j0), j1 - j0) };
                        e.apply(i, row);
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

/// Micro-kernel over a B read in place: `C[0..mr, 0..NR] (+)= A_panel * B` where row `p` of B is the `NR`
/// floats at `rows[p] + j`, then the epilogue.
type RowsKernel =
    unsafe fn(usize, *const f32, *const *const f32, usize, *mut f32, usize, usize, bool, Option<&RegEpi>);

fn rows_kernel() -> RowsKernel {
    static K: OnceLock<RowsKernel> = OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return rows_kernel_avx2_entry as RowsKernel;
            }
        }
        rows_kernel_generic as RowsKernel
    })
}

/// Portable [`RowsKernel`].
///
/// # Safety
/// `a` points to `kc * MR` floats; `rows` to `kc` pointers each readable for `j + NR` floats; `c` to an
/// `mr x NR` block with row stride `ldc` that nobody else accesses concurrently.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_kernel_generic(
    kc: usize,
    a: *const f32,
    rows: *const *const f32,
    j: usize,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    acc_c: bool,
    epi: Option<&RegEpi>,
) {
    let mut acc = [[0f32; NR]; MR];
    for p in 0..kc {
        // SAFETY: guaranteed by the caller.
        let (av, bv) = unsafe {
            (
                std::slice::from_raw_parts(a.add(p * MR), MR),
                std::slice::from_raw_parts((*rows.add(p)).add(j), NR),
            )
        };
        for i in 0..MR {
            for jj in 0..NR {
                acc[i][jj] += av[i] * bv[jj];
            }
        }
    }
    // SAFETY: the caller owns the mr x NR block at `c`.
    unsafe { store_block_generic(&mut acc, c, ldc, mr, NR, acc_c, epi) }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows_kernel_avx2_entry(
    kc: usize,
    a: *const f32,
    rows: *const *const f32,
    j: usize,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    // SAFETY: only selected after runtime detection of avx2 and fma; pointer contract forwarded.
    unsafe { rows_kernel_avx2(kc, a, rows, j, c, ldc, mr, acc, epi) }
}

/// AVX2/FMA 6x16 [`RowsKernel`]: the 6x16 register tile of [`kernel_avx2`], with each B row loaded from its
/// own pointer instead of a packed panel.
///
/// # Safety
/// As [`rows_kernel_generic`], and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows_kernel_avx2(
    kc: usize,
    a: *const f32,
    rows: *const *const f32,
    j: usize,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
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
    // SAFETY: the caller guarantees kc * MR floats at `a`, kc row pointers each readable for j + NR floats, and
    // owns the block at `c`.
    unsafe {
        for p in 0..kc {
            let bp = (*rows.add(p)).add(j);
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
        }
        let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]];
        store_block_avx2(&rows, c, ldc, mr, NR, acc, epi);
    }
}

/// The B operand of [`gemm_rows`]: `B[kk, j] = *rows[kk].add(j)` for `j < n`, each row readable for `n`
/// floats.
pub(crate) struct RowPtrs<'a> {
    pub rows: &'a [*const f32],
    pub n: usize,
}

// SAFETY: the pointers are only read, and the caller of `gemm_rows` keeps the data alive and unmodified.
unsafe impl Sync for RowPtrs<'_> {}

/// How the columns of [`gemm_rows`]' C land in the output: column `j = y * wp + x` is output column
/// `y * ow + x` when `x < ow`, and dropped otherwise (the padding columns of a shifted-rows convolution).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnMap {
    pub wp: usize,
    pub ow: usize,
}

/// `out = A * B` for an `m x k` A and a `k x n` B given by row pointers (read in place, never packed), with
/// C's columns mapped into `out` (row stride `map.ow * (n / map.wp)`) by `map`. `epi` applies to every
/// finished output element once, as in [`gemm_ex`]. Tiles, K order and therefore results are those of
/// [`gemm_blocked`]: each element is summed over `KC` blocks in order by one task, whatever `threads` is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_rows<A: Operand>(
    m: usize,
    k: usize,
    a: &A,
    b: &RowPtrs<'_>,
    out: &mut [f32],
    map: ColumnMap,
    threads: usize,
    epi: Option<BiasAct<'_>>,
) {
    let n = b.n;
    assert!(b.rows.len() == k && map.ow <= map.wp && n.is_multiple_of(map.wp), "gemm_rows: bad geometry");
    let ldo = map.ow * (n / map.wp);
    assert!(m == 0 || (m - 1) * ldo + ldo <= out.len(), "gemm_rows: output too small");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            let row = &mut out[i * ldo..(i + 1) * ldo];
            row.fill(0.0);
            if let Some(e) = epi {
                e.apply(i, row);
            }
        }
        return;
    }
    let threads = if 2 * m * n * k < PAR_FLOPS { 1 } else { threads };
    let kern = kernel();
    let rkern = rows_kernel();
    let pm = m.div_ceil(MR);
    let reg = epi.filter(BiasAct::in_kernel);
    let blocks_fit = (PACK_BUDGET / (pm * MR * KC)).max(1);
    let ks = (blocks_fit * KC).min(k.div_ceil(KC) * KC);
    let g = plan_tiles(m, n, k, threads);
    let (tm, tn) = g.tiles();
    let parallel = threads > 1 && tm * tn > 1;
    let optr = SendPtr(out.as_mut_ptr());
    let a_len = pm * MR * ks;
    // The C tiles of one K super-block live in task-local buffers between super-blocks only when K needs
    // more than one; then they are kept in `cbuf` (one per tile).
    let multi = k > ks;
    let mut cbuf = if multi { super::alloc_uninit(m * n) } else { Vec::new() };
    let cbase = SendPtr(cbuf.as_mut_ptr());
    with_scratch(a_len + 16, |abuf| {
        let pa_all = aligned(abuf, a_len);
        let mut s0 = 0;
        while s0 < k {
            let ksz = ks.min(k - s0);
            let (first, last) = (s0 == 0, s0 + ksz == k);
            let pa = &mut pa_all[..pm * MR * ksz];
            let pack_a = |(p, dst): (usize, &mut [f32])| {
                a.pack(p * MR, MR.min(m - p * MR), s0, ksz, MR, dst);
            };
            pack_lhs_panels(pa, MR * ksz, parallel, pack_a);
            let pa = &*pa;
            let tile = |t: usize| {
                let (ti, tj) = (t / tn, t % tn);
                let (i0, i1) = (ti * g.mc, ((ti + 1) * g.mc).min(m));
                let (j0, j1) = (tj * g.nc, ((tj + 1) * g.nc).min(n));
                let (th, tw) = (i1 - i0, j1 - j0);
                // Whole-value uses, so the closure captures the `Sync` wrappers rather than their raw fields.
                let (b, optr, cbase) = (b, &optr, &cbase);
                // `ct` is this tile's C block (row stride `ldc`); only this task touches it.
                let run = |ct: *mut f32, ldc: usize| {
                    let mut edge = [0f32; KC * NR];
                    let mut kb = 0;
                    while kb < ksz {
                        let kc = KC.min(ksz - kb);
                        let acc = !first || kb != 0;
                        let finished = last && kb + kc == ksz;
                        let rows = &b.rows[s0 + kb..s0 + kb + kc];
                        let mut j = j0;
                        while j < j1 {
                            let nr = NR.min(j1 - j);
                            if nr < NR {
                                // The ragged last panel: copy its rows so no load runs past them.
                                for (p, r) in rows.iter().enumerate() {
                                    // SAFETY: row `r` is readable for `n` floats and j + nr <= n.
                                    let src = unsafe { std::slice::from_raw_parts(r.add(j), nr) };
                                    edge[p * NR..p * NR + nr].copy_from_slice(src);
                                    edge[p * NR + nr..(p + 1) * NR].fill(0.0);
                                }
                            }
                            let mut i = i0;
                            while i < i1 {
                                let mr = MR.min(i1 - i);
                                let apanel = &pa[(i / MR) * MR * ksz + kb * MR..];
                                debug_assert!(apanel.len() >= kc * MR);
                                let re = reg.filter(|_| finished).map(|e| e.kernel_rows(i, mr));
                                let re = re.as_ref();
                                // SAFETY: `apanel` holds kc * MR floats; full panels read rows[p] + j..j + NR,
                                // inside each row (j + NR <= n); the ragged one reads `edge`; the C block
                                // [i..i + mr, j..j + NR] is inside this task's own tile.
                                unsafe {
                                    let cp = ct.add((i - i0) * ldc + (j - j0));
                                    if nr == NR {
                                        rkern(kc, apanel.as_ptr(), rows.as_ptr(), j, cp, ldc, mr, acc, re);
                                    } else {
                                        kern(kc, apanel.as_ptr(), edge.as_ptr(), cp, ldc, mr, nr, acc, re);
                                    }
                                }
                                i += MR;
                            }
                            j += NR;
                        }
                        kb += KC;
                    }
                };
                let emit = |ct: *const f32, ldc: usize| {
                    for i in i0..i1 {
                        // SAFETY: row i, columns j0..j1 of the tile, written by `run` above.
                        let crow = unsafe { std::slice::from_raw_parts(ct.add((i - i0) * ldc), tw) };
                        let mut y = j0 / map.wp;
                        while y * map.wp < j1 {
                            let (s, e) = ((y * map.wp).max(j0), (y * map.wp + map.ow).min(j1));
                            if s < e {
                                let o0 = y * map.ow + (s - y * map.wp);
                                // SAFETY: output columns o0..o0 + (e - s) of row i belong to this tile only
                                // (the column map is injective), and lie inside `out` (asserted above).
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(optr.0.add(i * ldo + o0), e - s)
                                };
                                dst.copy_from_slice(&crow[s - j0..e - j0]);
                                if let (Some(e), None) = (epi, reg) {
                                    e.apply(i, dst);
                                }
                            }
                            y += 1;
                        }
                    }
                };
                if multi {
                    // Tile (ti, tj) owns rows i0..i1, columns j0..j1 of `cbuf` (row stride n).
                    // SAFETY: i0 * n + j0 is inside `cbuf` (m x n).
                    let ct = unsafe { cbase.0.add(i0 * n + j0) };
                    run(ct, n);
                    if last {
                        emit(ct, n);
                    }
                } else {
                    with_scratch(th * tw + 16, |buf| {
                        let ct = aligned(buf, th * tw).as_mut_ptr();
                        run(ct, tw);
                        emit(ct, tw);
                    });
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
    super::recycle(cbuf);
}

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

/// AVX2/FMA [`ARowsKernel`]: [`kernel_avx2`] with the six A values of each K step broadcast from six rows.
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
            kernel_generic(kc, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr(), NR, MR, NR, true, None);
            kernel()(kc, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr(), NR, MR, NR, true, None);
        }
        for (x, y) in c1.iter().zip(&c2) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    /// The epilogue a micro-kernel applies to its registers is bit for bit the row pass over the stored block
    /// ([`BiasAct::apply`]), on the portable and the dispatched kernel, for full and ragged blocks, overwriting and
    /// accumulating, with zero, negative and positive biases.
    #[test]
    fn kernel_epilogue_is_bitwise_the_row_pass() {
        let kc = 37;
        let a = data(kc * MR, 5);
        let b = data(kc * NR, 6);
        let bias = [0.0f32, -0.75, 1.5, 0.0, -2.0, 0.25];
        let kernels: [Kernel; 2] = [kernel_generic, kernel()];
        for kern in kernels {
            for (mr, nr) in [(MR, NR), (4, NR), (MR, 11), (1, 3)] {
                for acc in [false, true] {
                    for act in [None, Some(UnaryOp::Silu)] {
                        for with_bias in [false, true] {
                            let epi = BiasAct { bias: with_bias.then_some(&bias[..]), act };
                            let prior = data(MR * NR, 7);
                            let (mut fused, mut want) = (prior.clone(), prior.clone());
                            let re = epi.kernel_rows(0, mr);
                            // SAFETY: the panels and blocks are sized as the kernels require.
                            unsafe {
                                kern(
                                    kc,
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    fused.as_mut_ptr(),
                                    NR,
                                    mr,
                                    nr,
                                    acc,
                                    Some(&re),
                                );
                                kern(kc, a.as_ptr(), b.as_ptr(), want.as_mut_ptr(), NR, mr, nr, acc, None);
                            }
                            for i in 0..mr {
                                epi.apply(i, &mut want[i * NR..i * NR + nr]);
                            }
                            let same = fused.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits());
                            assert!(same, "mr {mr} nr {nr} acc {acc} act {act:?} bias {with_bias}");
                        }
                    }
                }
            }
        }
    }
}
