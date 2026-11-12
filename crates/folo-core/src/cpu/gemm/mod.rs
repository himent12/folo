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

mod arows;
mod blocked;
mod kernel;
mod rows;
#[cfg(test)]
mod tests;

use crate::backend::UnaryOp;

pub(crate) use arows::gemm_arows;
pub(crate) use blocked::gemm_ex;
pub use kernel::{uses_avx2, uses_simd128};
pub(crate) use rows::{ColumnMap, RowPtrs, gemm_rows};

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

    /// For an A operand whose K run of each row is contiguous: `Some((data, offset, so))`, where element
    /// `(o, kk)` is `data[offset + o * so + kk]`. The drivers then read A's rows in place through row pointers
    /// instead of packing it first (a serial pass before every GEMM of a conv forward, on the calling thread).
    fn rows_in_place(&self) -> Option<(&[f32], usize, usize)> {
        None
    }
}

/// A's rows read in place (see [`Operand::rows_in_place`]): element `(o, kk)` is `data[offset + o * so + kk]`.
#[derive(Clone, Copy)]
pub(super) struct ARows<'a> {
    data: &'a [f32],
    offset: usize,
    so: usize,
}

impl<'a> ARows<'a> {
    /// `a`'s rows in place, if it has them and its packed form would fit one K super-block ([`PACK_BUDGET`]
    /// floats: every conv weight), checked to hold `m x k` elements.
    ///
    /// # Panics
    /// If `data` is too short for `m` rows of `k` elements, or that extent overflows `usize` (the packing path
    /// would fail its slice bounds check there too): the kernels read these rows through raw pointers.
    pub(super) fn of<A: Operand>(a: &'a A, m: usize, k: usize) -> Option<Self> {
        let (data, offset, so) = a.rows_in_place()?;
        if (m.div_ceil(MR) * MR).checked_mul(k).is_none_or(|v| v > PACK_BUDGET) {
            // Larger than one packed super-block: packing is a small part of such a GEMM, and done in parallel.
            return None;
        }
        // One past the last element read: `offset + (m - 1) * so + k`.
        let end = (m.saturating_sub(1).checked_mul(so)).and_then(|v| v.checked_add(offset)?.checked_add(k));
        assert!(m == 0 || k == 0 || end.is_some_and(|e| e <= data.len()), "gemm: A buffer too small");
        Some(ARows { data, offset, so })
    }

    /// Pointers to K element `k0` of the `mr` rows from `i` (`mr <= MR`, rows `i..i + mr` inside A, and at
    /// least `k0 + kc <= k` read from each); rows past `mr` repeat row `i`, as the kernels compute but never
    /// store them.
    #[inline]
    pub(super) fn panel(&self, i: usize, mr: usize, k0: usize) -> [*const f32; MR] {
        let base = self.data.as_ptr();
        // SAFETY: `of` checked that every row `< m` holds `k` floats from `offset + row * so`.
        std::array::from_fn(|r| unsafe {
            base.add(self.offset + (i + if r < mr { r } else { 0 }) * self.so + k0)
        })
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

    fn rows_in_place(&self) -> Option<(&[f32], usize, usize)> {
        (self.sk == 1).then_some((self.data, self.offset, self.so))
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
