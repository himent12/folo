//! GEMM with a B read in place through row pointers ([`gemm_rows`], the shifted-rows convolution's B, never
//! packed): the 6x16 micro-kernel with each B row loaded from its own pointer, and its tiled driver.

use std::sync::OnceLock;

use super::arows::arows_kernel;
use super::blocked::{aligned, pack_lhs_panels, plan_tiles, with_scratch};
#[cfg(target_arch = "x86_64")]
use super::kernel::store_block_avx2;
use super::kernel::{kernel, store_block_generic};
use super::{ARows, BiasAct, KC, MR, NR, Operand, PACK_BUDGET, PAR_FLOPS, RegEpi};
use crate::cpu::elementwise::SendPtr;
use crate::cpu::par_for;

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

/// [`RowsKernel`] with A read in place too: row `i` of A is the `kc` floats at `a[i]` (rows past `mr` repeat
/// row 0 and are not stored). The same FMAs in the same order as [`RowsKernel`], so the same results.
pub(super) type RowsAKernel = unsafe fn(
    usize,
    &[*const f32; MR],
    *const *const f32,
    usize,
    *mut f32,
    usize,
    usize,
    bool,
    Option<&RegEpi>,
);

pub(super) fn rows_a_kernel() -> RowsAKernel {
    static K: OnceLock<RowsAKernel> = OnceLock::new();
    *K.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return rows_a_kernel_avx2_entry as RowsAKernel;
            }
        }
        rows_a_kernel_generic as RowsAKernel
    })
}

/// Portable [`RowsAKernel`].
///
/// # Safety
/// Every `a[i]` points to `kc` floats; otherwise as [`rows_kernel_generic`].
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn rows_a_kernel_generic(
    kc: usize,
    a: &[*const f32; MR],
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
        let bv = unsafe { std::slice::from_raw_parts((*rows.add(p)).add(j), NR) };
        for i in 0..MR {
            // SAFETY: guaranteed by the caller.
            let av = unsafe { *a[i].add(p) };
            for jj in 0..NR {
                acc[i][jj] += av * bv[jj];
            }
        }
    }
    // SAFETY: the caller owns the mr x NR block at `c`.
    unsafe { store_block_generic(&mut acc, c, ldc, mr, NR, acc_c, epi) }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows_a_kernel_avx2_entry(
    kc: usize,
    a: &[*const f32; MR],
    rows: *const *const f32,
    j: usize,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    acc: bool,
    epi: Option<&RegEpi>,
) {
    // SAFETY: only selected after runtime detection of avx2 and fma; pointer contract forwarded.
    unsafe { rows_a_kernel_avx2(kc, a, rows, j, c, ldc, mr, acc, epi) }
}

/// AVX2/FMA 6x16 [`RowsAKernel`]: [`rows_kernel_avx2`] with the six A values of each K step broadcast from six
/// rows instead of a packed panel.
///
/// # Safety
/// As [`rows_a_kernel_generic`], and the CPU supports AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn rows_a_kernel_avx2(
    kc: usize,
    a: &[*const f32; MR],
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
    let [a0p, a1p, a2p, a3p, a4p, a5p] = *a;
    // SAFETY: the caller guarantees kc floats at every `a[i]`, kc row pointers each readable for j + NR floats,
    // and owns the block at `c`.
    unsafe {
        for p in 0..kc {
            let bp = (*rows.add(p)).add(j);
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
        }
        let rows = [[c00, c01], [c10, c11], [c20, c21], [c30, c31], [c40, c41], [c50, c51]];
        store_block_avx2(&rows, c, ldc, mr, NR, acc, epi);
    }
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

/// AVX2/FMA 6x16 [`RowsKernel`]: the 6x16 register tile of `kernel_avx2`, with each B row loaded from its
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
/// `gemm_blocked`: each element is summed over `KC` blocks in order by one task, whatever `threads` is.
///
/// [`gemm_ex`]: super::gemm_ex
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
    let (kern, rkern) = (kernel(), rows_kernel());
    let (akern, rakern) = (arows_kernel(), rows_a_kernel());
    // A read in place (a conv's weights) is never packed: one K super-block.
    let ain = ARows::of(a, m, k);
    let pm = m.div_ceil(MR);
    let reg = epi.filter(BiasAct::in_kernel);
    let blocks_fit = (PACK_BUDGET / (pm * MR * KC)).max(1);
    let ks = if ain.is_some() { k } else { (blocks_fit * KC).min(k.div_ceil(KC) * KC) };
    let g = plan_tiles(m, n, k, threads);
    let (tm, tn) = g.tiles();
    let parallel = threads > 1 && tm * tn > 1;
    let optr = SendPtr(out.as_mut_ptr());
    let a_len = if ain.is_some() { 0 } else { pm * MR * ks };
    // The C tiles of one K super-block live in task-local buffers between super-blocks only when K needs
    // more than one; then they are kept in `cbuf` (one per tile).
    let multi = k > ks;
    let mut cbuf = if multi { crate::cpu::alloc_uninit(m * n) } else { Vec::new() };
    let cbase = SendPtr(cbuf.as_mut_ptr());
    with_scratch(a_len + 16, |abuf| {
        let pa_all = aligned(abuf, a_len);
        let mut s0 = 0;
        while s0 < k {
            let ksz = ks.min(k - s0);
            let (first, last) = (s0 == 0, s0 + ksz == k);
            let pa = &mut pa_all[..a_len.min(pm * MR * ksz)];
            if ain.is_none() {
                let pack_a = |(p, dst): (usize, &mut [f32])| {
                    a.pack(p * MR, MR.min(m - p * MR), s0, ksz, MR, dst);
                };
                pack_lhs_panels(pa, MR * ksz, parallel, pack_a);
            }
            let pa = &*pa;
            let tile = |t: usize| {
                let (ti, tj) = (t / tn, t % tn);
                let (i0, i1) = (ti * g.mc, ((ti + 1) * g.mc).min(m));
                let (j0, j1) = (tj * g.nc, ((tj + 1) * g.nc).min(n));
                let (th, tw) = (i1 - i0, j1 - j0);
                // Whole-value uses, so the closure captures the `Sync` wrappers rather than their raw fields.
                let (b, optr, cbase, ain) = (b, &optr, &cbase, &ain);
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
                                let re = reg.filter(|_| finished).map(|e| e.kernel_rows(i, mr));
                                let re = re.as_ref();
                                // SAFETY: full panels read rows[p] + j..j + NR, inside each row (j + NR <= n);
                                // the ragged one reads `edge`; the C block [i..i + mr, j..j + NR] is inside
                                // this task's own tile.
                                unsafe {
                                    let cp = ct.add((i - i0) * ldc + (j - j0));
                                    if let Some(ar) = ain {
                                        // The rows i..i + mr of A hold K elements s0 + kb..s0 + kb + kc.
                                        let ap = ar.panel(i, mr, s0 + kb);
                                        if nr == NR {
                                            rakern(kc, &ap, rows.as_ptr(), j, cp, ldc, mr, acc, re);
                                        } else {
                                            akern(kc, &ap, edge.as_ptr(), cp, ldc, mr, nr, acc, re);
                                        }
                                    } else {
                                        // `apanel` holds kc * MR floats.
                                        let apanel = &pa[(i / MR) * MR * ksz + kb * MR..];
                                        debug_assert!(apanel.len() >= kc * MR);
                                        if nr == NR {
                                            rkern(
                                                kc,
                                                apanel.as_ptr(),
                                                rows.as_ptr(),
                                                j,
                                                cp,
                                                ldc,
                                                mr,
                                                acc,
                                                re,
                                            );
                                        } else {
                                            kern(
                                                kc,
                                                apanel.as_ptr(),
                                                edge.as_ptr(),
                                                cp,
                                                ldc,
                                                mr,
                                                nr,
                                                acc,
                                                re,
                                            );
                                        }
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
                par_for(tm * tn, &tile);
            } else {
                (0..tm * tn).for_each(tile);
            }
            s0 += ksz;
        }
    });
    crate::cpu::recycle(cbuf);
}

/// `C = A * B` on the calling thread, for the small GEMMs of a Winograd tile block: A is row-major `m x k` (row
/// `i` is the `k` floats from `a[i * lda]`), B's `k` rows are read in place through `rows` (each `n` floats, `n` a
/// multiple of `NR`), and C is row-major `m x n` with row stride `ldc`, overwritten. No epilogue. K runs in `KC`
/// blocks; each B micro-panel (`kc x NR`) stays in L1 while every row panel of A passes it.
///
/// # Safety
/// Every pointer in `rows` is readable for `n` floats.
///
/// # Panics
/// If `n` is not a multiple of `NR`, `rows` is not `k` long, or `a` / `c` are too short.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_rows_serial(
    m: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    rows: &[*const f32],
    n: usize,
    c: &mut [f32],
    ldc: usize,
) {
    assert!(n.is_multiple_of(NR) && rows.len() == k && ldc >= n, "gemm_rows_serial: bad geometry");
    assert!(
        m == 0 || ((m - 1) * lda + k <= a.len() && (m - 1) * ldc + n <= c.len()),
        "gemm_rows_serial: short"
    );
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        (0..m).for_each(|i| c[i * ldc..i * ldc + n].fill(0.0));
        return;
    }
    let rakern = rows_a_kernel();
    let (ap, cp) = (a.as_ptr(), c.as_mut_ptr());
    let mut kb = 0;
    while kb < k {
        let kc = KC.min(k - kb);
        let brows = &rows[kb..kb + kc];
        let mut j = 0;
        while j < n {
            let mut i = 0;
            while i < m {
                let mr = MR.min(m - i);
                // SAFETY: rows i..i + mr of A hold k floats from i * lda (asserted), rows past mr repeat row i;
                // each B row is readable for n floats (the caller's contract), j + NR <= n; the C block
                // [i..i + mr, j..j + NR] is inside `c` (asserted).
                unsafe {
                    let apan: [*const f32; MR] =
                        std::array::from_fn(|r| ap.add((i + if r < mr { r } else { 0 }) * lda + kb));
                    rakern(kc, &apan, brows.as_ptr(), j, cp.add(i * ldc + j), ldc, mr, kb != 0, None);
                }
                i += MR;
            }
            j += NR;
        }
        kb += KC;
    }
}
