//! Symmetric block-quantized int4 (`Q4_0`-style) weights and dequantizing dot / GEMV / matmul kernels.
//!
//! Each [`Q4Block`] packs [`BLOCK_SIZE`] (`32`) signed 4-bit integers `q ∈ [-8, 7]` into [`PACKED_BLOCK_BYTES`]
//! (`16`) bytes plus one `f32` scale. Elements use the split-half layout (`qs[j]` stores element `j` in its low
//! nibble and element `j + 16` in its high nibble, biased by `+8` into `0..=15`) so AVX2 unpacks a full
//! 32-element block with one 128-bit load, two nibble masks, and one `vpsubb`.
//!
//! On `x86_64` with AVX2 + FMA, quantization, dequantization, `Q4 × f32` FMA dot / 4-row GEMV / matmul, and
//! exact `Q4 × Q8` integer dot (`vpmaddubsw` + `vpmaddwd`) dispatch to [`avx2`]; other architectures run the
//! portable scalar implementation with matching quantization semantics.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(test)]
mod tests;

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

/// Elements per symmetric quantization block.
pub const BLOCK_SIZE: usize = 32;
/// Packed nibble bytes per [`Q4Block`] (`BLOCK_SIZE / 2`).
pub const PACKED_BLOCK_BYTES: usize = BLOCK_SIZE / 2;

/// Minimum multiply-add work before `gemv` / `matmul` fans out across rayon threads.
const Q4_PAR_MIN: usize = 1 << 15;

/// One block of 32 symmetric 4-bit weights (`x[i] ≈ scale * q[i]`, `q[i] ∈ [-8, 7]`).
///
/// Byte `qs[j]` (`0 <= j < 16`) stores `(q[j] + 8)` in bits `0..4` and `(q[j + 16] + 8)` in bits `4..8`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4Block {
    /// Non-negative finite block scale (`0.0` for all-zero blocks).
    pub scale: f32,
    /// Packed biased 4-bit nibbles (`q + 8 ∈ 0..=15`).
    pub qs: [u8; PACKED_BLOCK_BYTES],
}

impl Q4Block {
    /// All-zero block (`scale = 0.0`, every quantized value `q = 0`, biased nibble `8` -> `0x88`).
    pub const ZERO: Self = Self { scale: 0.0, qs: [0x88; PACKED_BLOCK_BYTES] };
}

/// One block of 32 symmetric 8-bit activations (`x[i] ≈ scale * qs[i]`, `qs[i] ∈ [-127, 127]`), with the
/// exact integer sum of `qs` precomputed for unbiased `Q4 × Q8` dot products.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8Block {
    /// Non-negative finite block scale (`0.0` for all-zero blocks).
    pub scale: f32,
    /// Exact sum `Σ_i qs[i]`, used to cancel the `+8` nibble bias in [`dot_q4_q8`].
    pub sum: i32,
    /// Symmetric signed 8-bit codes in `[-127, 127]`.
    pub qs: [i8; BLOCK_SIZE],
}

impl Q8Block {
    /// All-zero block.
    pub const ZERO: Self = Self { scale: 0.0, sum: 0, qs: [0; BLOCK_SIZE] };
}

/// A `[rows, cols]` weight matrix quantized row-wise into [`Q4Block`]s (each row padded with zeros to a
/// multiple of [`BLOCK_SIZE`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Q4Matrix {
    /// Number of output rows (`N`).
    pub rows: usize,
    /// Logical columns per row (`K`, need not be a multiple of [`BLOCK_SIZE`]).
    pub cols: usize,
    /// Number of [`Q4Block`]s stored per row (`cols.div_ceil(BLOCK_SIZE)`).
    pub blocks_per_row: usize,
    /// Row-major blocks of length `rows * blocks_per_row`.
    pub blocks: Vec<Q4Block>,
}

impl Q4Matrix {
    /// Quantizes a row-major `[rows, cols]` `f32` matrix into symmetric 4-bit blocks.
    ///
    /// # Panics
    /// If `weights.len() < rows * cols` or `rows * cols` overflows.
    pub fn quantize(weights: &[f32], rows: usize, cols: usize) -> Self {
        let total = rows.checked_mul(cols).expect("q4::Q4Matrix::quantize: size overflow");
        assert!(weights.len() >= total, "q4::Q4Matrix::quantize: weights slice too short");
        let blocks_per_row = cols.div_ceil(BLOCK_SIZE);
        let total_blocks = rows.checked_mul(blocks_per_row).expect("q4::Q4Matrix::quantize: block overflow");
        let mut blocks = vec![Q4Block::ZERO; total_blocks];
        if total == 0 {
            return Self { rows, cols, blocks_per_row, blocks };
        }
        let run = |r: usize, dst_row: &mut [Q4Block]| {
            let src_row = &weights[r * cols..(r + 1) * cols];
            quantize_row_q4_into(src_row, dst_row);
        };
        if total < Q4_PAR_MIN {
            for (r, dst_row) in blocks.chunks_exact_mut(blocks_per_row).enumerate() {
                run(r, dst_row);
            }
        } else {
            crate::cpu::par_chunks(&mut blocks, blocks_per_row, &run);
        }
        Self { rows, cols, blocks_per_row, blocks }
    }

    /// Dequantizes the entire matrix back to row-major `[rows, cols]` `f32`.
    pub fn dequantize(&self) -> Vec<f32> {
        let total = self.rows.checked_mul(self.cols).expect("q4::Q4Matrix::dequantize: size overflow");
        let mut out = vec![0.0f32; total];
        if total == 0 {
            return out;
        }
        let bpr = self.blocks_per_row;
        for (r, dst_row) in out.chunks_exact_mut(self.cols).enumerate() {
            dequantize_q4(&self.blocks[r * bpr..(r + 1) * bpr], dst_row);
        }
        out
    }

    /// Returns the block slice for row `r`.
    #[inline]
    pub fn row_blocks(&self, r: usize) -> &[Q4Block] {
        let start = r * self.blocks_per_row;
        &self.blocks[start..start + self.blocks_per_row]
    }
}

/// Whether AVX2 + FMA `q4` kernels are active on this CPU.
pub fn uses_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Scalar quantization of one 32-element slice into a [`Q4Block`].
///
/// Non-finite inputs (`NaN`, `±∞`) are sanitized to `0.0`, so `scale` is always finite and `>= 0.0`.
/// Uses symmetric scale `amax / 7.0` with IEEE round-to-nearest-ties-to-even (`[-7, 7]`), bit-exact with AVX2.
#[inline]
pub(crate) fn quantize_block_q4_scalar(src: &[f32; BLOCK_SIZE]) -> Q4Block {
    let mut amax = 0.0f32;
    for &v in src {
        if v.is_finite() {
            amax = amax.max(v.abs());
        }
    }
    let scale = amax / 7.0;
    let inv = 1.0 / scale;
    if !(scale > 0.0 && inv.is_finite()) {
        return Q4Block::ZERO;
    }
    let mut qs = [0u8; PACKED_BLOCK_BYTES];
    for j in 0..PACKED_BLOCK_BYTES {
        let lo_v = if src[j].is_finite() { src[j] } else { 0.0 };
        let hi_v = if src[j + PACKED_BLOCK_BYTES].is_finite() { src[j + PACKED_BLOCK_BYTES] } else { 0.0 };
        let q0 = ((lo_v * inv).clamp(-7.0, 7.0).round_ties_even() as i32 + 8) as u8;
        let q1 = ((hi_v * inv).clamp(-7.0, 7.0).round_ties_even() as i32 + 8) as u8;
        qs[j] = q0 | (q1 << 4);
    }
    Q4Block { scale, qs }
}

/// Scalar dequantization of one [`Q4Block`] into 32 `f32`s.
#[inline]
pub(crate) fn dequantize_block_q4_scalar(block: &Q4Block, out: &mut [f32; BLOCK_SIZE]) {
    let s = block.scale;
    for j in 0..PACKED_BLOCK_BYTES {
        let byte = block.qs[j];
        let q0 = (byte & 0x0F) as i32 - 8;
        let q1 = (byte >> 4) as i32 - 8;
        out[j] = q0 as f32 * s;
        out[j + PACKED_BLOCK_BYTES] = q1 as f32 * s;
    }
}

/// Scalar quantization of one 32-element slice into a [`Q8Block`].
#[inline]
pub(crate) fn quantize_block_q8_scalar(src: &[f32; BLOCK_SIZE]) -> Q8Block {
    let mut amax = 0.0f32;
    for &v in src {
        if v.is_finite() {
            amax = amax.max(v.abs());
        }
    }
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    if !(scale > 0.0 && inv.is_finite()) {
        return Q8Block::ZERO;
    }
    let mut qs = [0i8; BLOCK_SIZE];
    let mut sum = 0i32;
    for (d, &raw) in qs.iter_mut().zip(src) {
        let v = if raw.is_finite() { raw } else { 0.0 };
        let q = (v * inv).clamp(-127.0, 127.0).round_ties_even() as i32;
        *d = q as i8;
        sum += q;
    }
    Q8Block { scale, sum, qs }
}

/// Quantizes `src` into `dst` (length `src.len().div_ceil(BLOCK_SIZE)`), zero-padding any partial tail block.
pub fn quantize_row_q4_into(src: &[f32], dst: &mut [Q4Block]) {
    let nblocks = src.len().div_ceil(BLOCK_SIZE);
    assert_eq!(dst.len(), nblocks, "q4::quantize_row_q4_into: destination length mismatch");
    if src.is_empty() {
        return;
    }
    let (chunks, tail) = src.as_chunks::<BLOCK_SIZE>();
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 + FMA detected.
        unsafe {
            for (c, out_blk) in chunks.iter().zip(dst.iter_mut()) {
                *out_blk = avx2::quantize_block_q4_avx2(c);
            }
            if !tail.is_empty() {
                let mut padded = [0.0f32; BLOCK_SIZE];
                padded[..tail.len()].copy_from_slice(tail);
                dst[chunks.len()] = avx2::quantize_block_q4_avx2(&padded);
            }
        }
        return;
    }
    for (c, out_blk) in chunks.iter().zip(dst.iter_mut()) {
        *out_blk = quantize_block_q4_scalar(c);
    }
    if !tail.is_empty() {
        let mut padded = [0.0f32; BLOCK_SIZE];
        padded[..tail.len()].copy_from_slice(tail);
        dst[chunks.len()] = quantize_block_q4_scalar(&padded);
    }
}

/// Quantizes a slice of `f32`s into symmetric 4-bit [`Q4Block`]s.
pub fn quantize_q4(src: &[f32]) -> Vec<Q4Block> {
    let mut dst = vec![Q4Block::ZERO; src.len().div_ceil(BLOCK_SIZE)];
    quantize_row_q4_into(src, &mut dst);
    dst
}

/// Dequantizes `blocks` into `out` (`out.len()` need not be a multiple of [`BLOCK_SIZE`]).
///
/// # Panics
/// If `blocks.len() < out.len().div_ceil(BLOCK_SIZE)`.
pub fn dequantize_q4(blocks: &[Q4Block], out: &mut [f32]) {
    let nblocks = out.len().div_ceil(BLOCK_SIZE);
    assert!(blocks.len() >= nblocks, "q4::dequantize_q4: not enough blocks for output length");
    if out.is_empty() {
        return;
    }
    let (chunks, tail) = out.as_chunks_mut::<BLOCK_SIZE>();
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 + FMA detected.
        unsafe {
            for (blk, dst_chunk) in blocks.iter().zip(chunks.iter_mut()) {
                avx2::dequantize_block_q4_avx2(blk, dst_chunk);
            }
            if !tail.is_empty() {
                let mut tmp = [0.0f32; BLOCK_SIZE];
                avx2::dequantize_block_q4_avx2(&blocks[chunks.len()], &mut tmp);
                tail.copy_from_slice(&tmp[..tail.len()]);
            }
        }
        return;
    }
    for (blk, dst_chunk) in blocks.iter().zip(chunks.iter_mut()) {
        dequantize_block_q4_scalar(blk, dst_chunk);
    }
    if !tail.is_empty() {
        let mut tmp = [0.0f32; BLOCK_SIZE];
        dequantize_block_q4_scalar(&blocks[chunks.len()], &mut tmp);
        tail.copy_from_slice(&tmp[..tail.len()]);
    }
}

/// Quantizes an `f32` activation slice into symmetric 8-bit [`Q8Block`]s (zero-padded tail).
pub fn quantize_q8_blocks(src: &[f32]) -> Vec<Q8Block> {
    let nblocks = src.len().div_ceil(BLOCK_SIZE);
    let mut dst = vec![Q8Block::ZERO; nblocks];
    if src.is_empty() {
        return dst;
    }
    let (chunks, tail) = src.as_chunks::<BLOCK_SIZE>();
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 + FMA detected.
        unsafe {
            for (c, out_blk) in chunks.iter().zip(dst.iter_mut()) {
                *out_blk = avx2::quantize_block_q8_avx2(c);
            }
            if !tail.is_empty() {
                let mut padded = [0.0f32; BLOCK_SIZE];
                padded[..tail.len()].copy_from_slice(tail);
                dst[chunks.len()] = avx2::quantize_block_q8_avx2(&padded);
            }
        }
        return dst;
    }
    for (c, out_blk) in chunks.iter().zip(dst.iter_mut()) {
        *out_blk = quantize_block_q8_scalar(c);
    }
    if !tail.is_empty() {
        let mut padded = [0.0f32; BLOCK_SIZE];
        padded[..tail.len()].copy_from_slice(tail);
        dst[chunks.len()] = quantize_block_q8_scalar(&padded);
    }
    dst
}

/// Scalar `Q4 × f32` dot product over `x.len()` elements.
#[inline]
pub(crate) fn dot_q4_f32_scalar(w: &[Q4Block], x: &[f32]) -> f32 {
    let (chunks, tail) = x.as_chunks::<BLOCK_SIZE>();
    let mut acc = 0.0f32;
    for (blk, xs) in w.iter().zip(chunks) {
        let mut s = 0.0f32;
        for j in 0..PACKED_BLOCK_BYTES {
            let byte = blk.qs[j];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = (byte >> 4) as i32 - 8;
            s += q0 as f32 * xs[j] + q1 as f32 * xs[j + PACKED_BLOCK_BYTES];
        }
        acc += s * blk.scale;
    }
    if !tail.is_empty() {
        let blk = &w[chunks.len()];
        let mut s = 0.0f32;
        for (idx, &xv) in tail.iter().enumerate() {
            let q = if idx < PACKED_BLOCK_BYTES {
                (blk.qs[idx] & 0x0F) as i32 - 8
            } else {
                (blk.qs[idx - PACKED_BLOCK_BYTES] >> 4) as i32 - 8
            };
            s += q as f32 * xv;
        }
        acc += s * blk.scale;
    }
    acc
}

/// Dequantizing dot product between `Q4Block` weights `w` and `f32` activations `x`.
///
/// Each 32-element block accumulates `Σ q[i] * x[i]` in `f32` before multiplying by `blk.scale`, so extreme
/// `x` values near `f32::MAX` may overflow the unscaled inner sum even when `blk.scale < 1`.
///
/// # Panics
/// If `w.len() < x.len().div_ceil(BLOCK_SIZE)`.
pub fn dot_q4_f32(w: &[Q4Block], x: &[f32]) -> f32 {
    let nblocks = x.len().div_ceil(BLOCK_SIZE);
    assert!(w.len() >= nblocks, "q4::dot_q4_f32: weight blocks too short");
    if x.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 + FMA detected and `w.len() >= nblocks`.
        return unsafe { avx2::dot_q4_f32_avx2(w, x) };
    }
    dot_q4_f32_scalar(w, x)
}

/// Scalar `Q4 × Q8` dot product over `w.len()` blocks.
#[inline]
pub(crate) fn dot_q4_q8_scalar(w: &[Q4Block], x: &[Q8Block]) -> f32 {
    let mut acc = 0.0f32;
    for (wb, xb) in w.iter().zip(x) {
        let mut isum = 0i32;
        for j in 0..PACKED_BLOCK_BYTES {
            let byte = wb.qs[j];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = (byte >> 4) as i32 - 8;
            isum += q0 * xb.qs[j] as i32 + q1 * xb.qs[j + PACKED_BLOCK_BYTES] as i32;
        }
        // Multiply by `wb.scale` then `xb.scale` (rather than `wb.scale * xb.scale` first) so small
        // non-zero blocks do not prematurely underflow to zero.
        acc += ((isum as f32) * wb.scale) * xb.scale;
    }
    acc
}

/// Integer-accelerated dot product between quantized `Q4Block` weights and `Q8Block` activations.
///
/// Requires each [`Q8Block::sum`] to equal `Σ_i qs[i]` (as produced by [`quantize_q8_blocks`]).
///
/// # Panics
/// If `w.len() != x.len()`.
pub fn dot_q4_q8(w: &[Q4Block], x: &[Q8Block]) -> f32 {
    assert_eq!(w.len(), x.len(), "q4::dot_q4_q8: block count mismatch");
    if w.is_empty() {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    if uses_avx2() {
        // SAFETY: AVX2 + FMA detected and `w.len() == x.len()`.
        return unsafe { avx2::dot_q4_q8_avx2(w, x) };
    }
    dot_q4_q8_scalar(w, x)
}

/// Computes `out[r] = dot(w[r, :], x[:])` for `w` of shape `[rows, cols]` and `x` of length `cols`.
///
/// # Panics
/// If `x.len() != w.cols` or `out.len() != w.rows`.
pub fn gemv_q4_f32(w: &Q4Matrix, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "q4::gemv_q4_f32: input length mismatch");
    assert_eq!(out.len(), w.rows, "q4::gemv_q4_f32: output length mismatch");
    if w.rows == 0 {
        return;
    }
    if w.cols == 0 {
        out.fill(0.0);
        return;
    }
    let work = w.rows.saturating_mul(w.cols);
    let run_chunk = |base_row: usize, dst: &mut [f32]| {
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            while i + 4 <= dst.len() {
                let r = base_row + i;
                // SAFETY: AVX2 + FMA detected and `r + 4 <= w.rows`.
                let res = unsafe {
                    avx2::dot_q4_f32_x4_avx2(
                        [w.row_blocks(r), w.row_blocks(r + 1), w.row_blocks(r + 2), w.row_blocks(r + 3)],
                        x,
                    )
                };
                dst[i..i + 4].copy_from_slice(&res);
                i += 4;
            }
        }
        while i < dst.len() {
            dst[i] = dot_q4_f32(w.row_blocks(base_row + i), x);
            i += 1;
        }
    };
    if work < Q4_PAR_MIN {
        run_chunk(0, out);
    } else {
        const ROW_CHUNK: usize = 32;
        crate::cpu::par_chunks(out, ROW_CHUNK, &|ci, dst| run_chunk(ci * ROW_CHUNK, dst));
    }
}

/// Quantizes `x` once into [`Q8Block`]s and computes `out = W_q4 @ x_q8`.
///
/// # Panics
/// If `x.len() != w.cols` or `out.len() != w.rows`.
pub fn gemv_q4_q8(w: &Q4Matrix, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "q4::gemv_q4_q8: input length mismatch");
    assert_eq!(out.len(), w.rows, "q4::gemv_q4_q8: output length mismatch");
    if w.rows == 0 {
        return;
    }
    if w.cols == 0 {
        out.fill(0.0);
        return;
    }
    let xq = quantize_q8_blocks(x);
    let work = w.rows.saturating_mul(w.cols);
    let run_chunk = |base_row: usize, dst: &mut [f32]| {
        for (i, slot) in dst.iter_mut().enumerate() {
            *slot = dot_q4_q8(w.row_blocks(base_row + i), &xq);
        }
    };
    if work < Q4_PAR_MIN {
        run_chunk(0, out);
    } else {
        const ROW_CHUNK: usize = 32;
        crate::cpu::par_chunks(out, ROW_CHUNK, &|ci, dst| run_chunk(ci * ROW_CHUNK, dst));
    }
}

/// Computes `out[i, j] = dot(a[i, :], b[j, :]) + bias[j]` for `a` `[m, k]` and `b` `[n, k]` (`Q4Matrix`).
///
/// # Panics
/// If slice dimensions do not match `[m, k]`, `b.cols == k`, `[m, n]`, or `bias.len() == n`.
pub fn matmul_q4_f32(a: &[f32], m: usize, k: usize, b: &Q4Matrix, bias: Option<&[f32]>, out: &mut [f32]) {
    let n = b.rows;
    assert_eq!(b.cols, k, "q4::matmul_q4_f32: k dimension mismatch");
    assert_eq!(a.len(), m.checked_mul(k).expect("q4::matmul_q4_f32: m * k overflow"));
    assert_eq!(out.len(), m.checked_mul(n).expect("q4::matmul_q4_f32: m * n overflow"));
    if let Some(bs) = bias {
        assert_eq!(bs.len(), n, "q4::matmul_q4_f32: bias length mismatch");
    }
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for row in out.chunks_exact_mut(n) {
            match bias {
                Some(bs) => row.copy_from_slice(bs),
                None => row.fill(0.0),
            }
        }
        return;
    }
    let compute_row = |mi: usize, dst: &mut [f32]| {
        let a_row = &a[mi * k..(mi + 1) * k];
        let mut j = 0;
        #[cfg(target_arch = "x86_64")]
        if uses_avx2() {
            while j + 4 <= n {
                // SAFETY: AVX2 + FMA detected and `j + 4 <= n`.
                let res = unsafe {
                    avx2::dot_q4_f32_x4_avx2(
                        [b.row_blocks(j), b.row_blocks(j + 1), b.row_blocks(j + 2), b.row_blocks(j + 3)],
                        a_row,
                    )
                };
                dst[j..j + 4].copy_from_slice(&res);
                j += 4;
            }
        }
        while j < n {
            dst[j] = dot_q4_f32(b.row_blocks(j), a_row);
            j += 1;
        }
        if let Some(bs) = bias {
            for (d, &bv) in dst.iter_mut().zip(bs) {
                *d += bv;
            }
        }
    };
    let work = m.saturating_mul(n).saturating_mul(k);
    if m == 1 {
        gemv_q4_f32(b, a, out);
        if let Some(bs) = bias {
            for (d, &bv) in out.iter_mut().zip(bs) {
                *d += bv;
            }
        }
    } else if work < Q4_PAR_MIN {
        for (mi, dst) in out.chunks_exact_mut(n).enumerate() {
            compute_row(mi, dst);
        }
    } else {
        crate::cpu::par_chunks(out, n, &compute_row);
    }
}
