//! int8 inference on the CPU (docs/MODEL.md §9.6, docs/CORE.md "int8 convs"): u8 activations on an affine grid,
//! s8 weights, exact i32 accumulation with AVX-VNNI (`vpdpbusd`) where the CPU has it and a portable kernel
//! with the same integers where it does not.
//!
//! Activations are channel-blocked ("C4"): `[N][C / 4][H][W]` words of four u8 codes, channels `4g..4g + 4` of a
//! pixel in one `u32` (byte `t` is channel `4g + t`), so a word is one K step of four for the kernel. Every
//! buffer an activation lives in has [`SLACK`] words after its last plane: the kernel reads whole 24-column
//! tiles and drops the columns past a plane.
//!
//! - `kernel.rs`: the 4 × 24 micro-kernel (VNNI and portable) and the epilogue (dequantise, bias, SiLU,
//!   residual, requantise), the same bits on both paths.
//! - `conv.rs`: packed weights ([`QConv`]) and the conv driver ([`conv`]).
//! - this file: grids, C4 views, packing from and to `f32`, max pools and upsamples on codes.
//! - `tests.rs`: everything against a scalar integer reference.

mod conv;
mod kernel;
#[cfg(test)]
mod tests;

use rayon::prelude::*;

pub use conv::{MAX_PAD, MAX_REDUCTION, MAX_STRIDE, QConv, QOut, conv, quantize_weights};
pub use kernel::{silu_q8, uses_vnni};

/// Words of readable slack after an activation's last plane.
pub const SLACK: usize = 64;

/// An affine u8 grid: `x ≈ scale · (q − zero)`, `q` in `0..=255`; 0 is always on the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    /// Step between codes.
    pub scale: f32,
    /// The code of 0.
    pub zero: u8,
}

impl Grid {
    /// The grid spanning `[lo, hi]`, widened to hold 0 (a degenerate range gets a tiny step).
    pub fn from_range(lo: f32, hi: f32) -> Grid {
        let (lo, hi) = (lo.min(0.0), hi.max(0.0));
        let scale = ((hi - lo) / 255.0).max(1e-8);
        let zero = (-lo / scale).round().clamp(0.0, 255.0) as u8;
        Grid { scale, zero }
    }

    /// `1 / scale`, the multiplier quantisation uses.
    #[inline]
    pub fn inv(&self) -> f32 {
        1.0 / self.scale
    }

    /// The code of `x`: `clamp(round(x · inv) + zero, 0, 255)`, ties to even, as the conv epilogue computes it.
    #[inline]
    pub fn quantize(&self, x: f32) -> u8 {
        let v = x * self.inv();
        let v = if v > -512.0 { v } else { -512.0 };
        let v = if v < 512.0 { v } else { 512.0 };
        (v.round_ties_even() as i32 + self.zero as i32).clamp(0, 255) as u8
    }

    /// The value of code `q`.
    #[inline]
    pub fn dequantize(&self, q: u8) -> f32 {
        self.scale * (q as i32 - self.zero as i32) as f32
    }
}

/// A C4 activation (or a channel slice of one) read in place: image `i`'s group `g` plane is the `h · w` words at
/// `offset + i · img_stride + g · h · w`.
#[derive(Debug, Clone, Copy)]
pub struct C4<'a> {
    /// The buffer.
    pub data: &'a [u32],
    /// Words before image 0's first plane.
    pub offset: usize,
    /// Words between images.
    pub img_stride: usize,
    /// Channels (a multiple of 4).
    pub c: usize,
    /// The grid's zero code (the conv pads with it).
    pub zero: u8,
}

impl C4<'_> {
    /// Panics unless `n` images of `hw` pixels and the slack are inside `data`.
    pub fn check(&self, n: usize, hw: usize) {
        assert!(self.c.is_multiple_of(4), "C4: {} channels", self.c);
        let end = n
            .saturating_sub(1)
            .checked_mul(self.img_stride)
            .and_then(|v| v.checked_add(self.offset))
            .and_then(|v| (self.c / 4).checked_mul(hw).and_then(|p| v.checked_add(p)))
            .and_then(|v| v.checked_add(SLACK));
        assert!(
            end.is_some_and(|e| self.data.len() >= e),
            "C4: buffer of {} words, need {end:?} (None: overflow)",
            self.data.len()
        );
    }
}

/// A C4 buffer for `n` images of `c` channels and `hw` pixels, with the slack, from folo-core's CPU buffer cache
/// (contents unspecified). Give it back with [`recycle`].
pub fn alloc(n: usize, c: usize, hw: usize) -> Vec<u32> {
    words(n * c / 4 * hw + SLACK)
}

/// `n` words from the CPU buffer cache, contents unspecified.
pub(crate) fn words(n: usize) -> Vec<u32> {
    let v = std::mem::ManuallyDrop::new(crate::cpu::alloc_uninit(n));
    // SAFETY: f32 and u32 have the same size and alignment, and every bit pattern is a valid u32; the
    // allocation is handed over whole (pointer, length, capacity).
    unsafe { Vec::from_raw_parts(v.as_ptr() as *mut u32, v.len(), v.capacity()) }
}

/// Return a buffer from [`alloc`] (or any `Vec<u32>`) to the CPU buffer cache.
pub fn recycle(v: Vec<u32>) {
    let v = std::mem::ManuallyDrop::new(v);
    // SAFETY: as in `words`, the other way round (every u32 bit pattern is a valid f32).
    crate::cpu::recycle(unsafe { Vec::from_raw_parts(v.as_ptr() as *mut f32, v.len(), v.capacity()) });
}

/// `[n][c][hw]` `f32` (`c` a multiple of 4) into C4 codes on `grid`.
pub fn quantize(x: &[f32], n: usize, c: usize, hw: usize, grid: Grid, out: &mut [u32]) {
    assert!(c.is_multiple_of(4) && x.len() >= n * c * hw && out.len() >= n * c / 4 * hw);
    // Blocks of a plane, so a few large planes (the stem's 4 groups) still spread over the pool.
    const BLOCK: usize = 4096;
    out[..n * c / 4 * hw].par_chunks_mut(hw.max(1)).enumerate().for_each(|(ig, o)| {
        let (img, g) = (ig / (c / 4), ig % (c / 4));
        let base = (img * c + 4 * g) * hw;
        o.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
            for (i, d) in o.iter_mut().enumerate() {
                let p = b * BLOCK + i;
                *d = u32::from_le_bytes([0, 1, 2, 3].map(|t| grid.quantize(x[base + t * hw + p])));
            }
        });
    });
}

/// C4 codes on `grid` into `[n][c][hw]` `f32`.
pub fn dequantize(x: C4<'_>, n: usize, hw: usize, grid: Grid, out: &mut [f32]) {
    x.check(n, hw);
    let c = x.c;
    assert!(out.len() >= n * c * hw);
    out[..n * c * hw].par_chunks_mut(hw.max(1)).enumerate().for_each(|(ic, o)| {
        let (img, ch) = (ic / c, ic % c);
        let src = &x.data[x.offset + img * x.img_stride + ch / 4 * hw..][..hw];
        let shift = 8 * (ch % 4);
        for (d, &w) in o.iter_mut().zip(src) {
            *d = grid.dequantize((w >> shift) as u8);
        }
    });
}

/// Byte-wise maximum of two C4 words.
#[inline(always)]
fn max4(a: u32, b: u32) -> u32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u32::from_le_bytes([0, 1, 2, 3].map(|t| a[t].max(b[t])))
}

/// `k × k` max pool, stride 1, padding `k / 2` (padding never wins), of C4 codes into a contiguous C4 `out`
/// (`[n][c / 4][h · w]`). Exact: max commutes with a monotone quantiser.
pub fn max_pool(x: C4<'_>, n: usize, h: usize, w: usize, k: usize, out: &mut [u32]) {
    let hw = h * w;
    x.check(n, hw);
    let groups = x.c / 4;
    assert!(out.len() >= n * groups * hw);
    let r = (k / 2) as isize;
    out[..n * groups * hw].par_chunks_mut(hw.max(1)).enumerate().for_each(|(ig, o)| {
        let (img, g) = (ig / groups, ig % groups);
        let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
        // Rows first, then columns (separable, exact for max).
        let mut tmp = vec![0u32; hw];
        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            for xo in 0..w {
                let (lo, hi) = ((xo as isize - r).max(0) as usize, ((xo as isize + r) as usize).min(w - 1));
                tmp[y * w + xo] = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
            }
        }
        for y in 0..h {
            let (lo, hi) = ((y as isize - r).max(0) as usize, ((y as isize + r) as usize).min(h - 1));
            for xo in 0..w {
                o[y * w + xo] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xo]));
            }
        }
    });
}

/// Nearest-neighbour upsample by 2 of C4 codes into a contiguous C4 `out` (`[n][c / 4][2h · 2w]`).
pub fn upsample2(x: C4<'_>, n: usize, h: usize, w: usize, out: &mut [u32]) {
    let hw = h * w;
    x.check(n, hw);
    let groups = x.c / 4;
    let ohw = 4 * hw;
    assert!(out.len() >= n * groups * ohw);
    out[..n * groups * ohw].par_chunks_mut(ohw.max(1)).enumerate().for_each(|(ig, o)| {
        let (img, g) = (ig / groups, ig % groups);
        let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
        for y in 0..2 * h {
            let srow = &src[(y / 2) * w..(y / 2 + 1) * w];
            for (xo, d) in o[y * 2 * w..(y + 1) * 2 * w].iter_mut().enumerate() {
                *d = srow[xo / 2];
            }
        }
    });
}
