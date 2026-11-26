//! int8 inference on the CPU (docs/MODEL.md §9.6, docs/CORE.md "int8 convs"): u8 activations on affine grids
//! (per-tensor or per-channel in C4), s8 weights, exact i32 accumulation with AVX-VNNI (`vpdpbusd`) where the CPU
//! has it and a portable kernel with the same integers where it does not.
//!
//! Activations are channel-blocked ("C4"): `[N][C / 4][H][W]` words of four u8 codes, channels `4g..4g + 4` of a
//! pixel in one `u32` (byte `t` is channel `4g + t`), so a word is one K step of four for the kernel. Every
//! buffer an activation lives in has [`SLACK`] words after its last plane: the kernel reads whole 24-column
//! tiles and drops the columns past a plane.
//!
//! - `silu.rs`: bit-exact division-free Estrin degree-5 fast SiLU (`silu_q8`, scalar and AVX2).
//! - `kernel.rs`: the 4 × 24 micro-kernel (VNNI, AVX-512 VNNI VL, AVX2, and portable), L1 `TileF32` part
//!   accumulator, and per-channel epilogue (dequantise, bias, SiLU, residual, requantise), the same bits on all paths.
//! - `fused.rs` + `fused_epi.rs`: fused x86-64 L1-blocked tile-block drivers and zero-spill epilogues for `conv`,
//!   `conv_per_channel`, and `stem_u8`.
//! - `rows.rs`: input row-pointer preparation and row-band parallelised padded phase copies (`image_rows`).
//! - `stem.rs`: direct `u8` RGB0-to-C4 AVX-VNNI stem ([`QStem`], [`stem_u8`], [`pack_f32_to_rgb0_u8`]).
//! - `conv.rs`: packed weights ([`QConv`]) and the conv drivers ([`conv`], [`conv_per_channel`]).
//! - this file: grids, C4 views, AVX2-vectorised packing from and to `f32`, max pools and upsamples on codes.
//! - `tests.rs`: everything against a scalar integer reference.

mod conv;
#[cfg(target_arch = "x86_64")]
mod fused;
#[cfg(target_arch = "x86_64")]
mod fused_epi;
mod kernel;
mod quant;
mod rows;
mod silu;
mod stem;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd;

pub use conv::{
    MAX_PAD, MAX_REDUCTION, MAX_STRIDE, QConv, QOut, conv, conv_per_channel, conv_size, out_size,
    should_decouple_parts,
};
pub use kernel::{kernel_name, uses_vnni};
pub use quant::quantize_weights;
pub use silu::silu_q8;
pub use stem::{PIXEL_SCALE, QStem, pack_f32_to_rgb0_u8, stem_u8};

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
    /// The grid spanning `[lo, hi]`, widened to hold 0 (a degenerate range gets a tiny step). The step is not
    /// finite when the range is not, or is wider than `f32::MAX`: callers check it (folo-vision's
    /// `Calibration::param` refuses such ranges).
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
    /// The grid's zero code (the single-grid conv pads with it; per-channel convs use the weights' zero points).
    pub zero: u8,
}

impl C4<'_> {
    /// Panics unless `n` images of `hw` pixels and the slack are inside `data`.
    pub fn check(&self, n: usize, hw: usize) {
        assert!(self.c.is_multiple_of(4), "C4: {} channels", self.c);
        if n == 0 {
            return;
        }
        let end = (n - 1)
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
    assert!(c.is_multiple_of(4), "q8::alloc: {c} channels");
    let len = n
        .checked_mul(c / 4)
        .and_then(|v| v.checked_mul(hw))
        .and_then(|v| v.checked_add(SLACK))
        .expect("q8::alloc: size overflow");
    words(len)
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

/// Quantise one 4-channel slice `p0..p3[0..len]` into C4 words `out[0..len]` using per-channel grids `gs`,
/// bit-exact between scalar and AVX2.
fn quantize_group(p0: &[f32], p1: &[f32], p2: &[f32], p3: &[f32], gs: [Grid; 4], out: &mut [u32]) {
    let len = out.len();
    assert!(p0.len() >= len && p1.len() >= len && p2.len() >= len && p3.len() >= len);
    #[cfg(target_arch = "x86_64")]
    if kernel::uses_avx2() {
        // SAFETY: AVX2 and FMA were detected; slices have at least `out.len()` elements (asserted above).
        return unsafe { quantize_group_avx2(p0, p1, p2, p3, gs, out) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if kernel::uses_wasm_simd() {
        return wasm_simd::quantize_group_wasm(p0, p1, p2, p3, gs, out);
    }
    for i in 0..out.len() {
        out[i] = u32::from_le_bytes([
            gs[0].quantize(p0[i]),
            gs[1].quantize(p1[i]),
            gs[2].quantize(p2[i]),
            gs[3].quantize(p3[i]),
        ]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_group_avx2(
    p0: &[f32],
    p1: &[f32],
    p2: &[f32],
    p3: &[f32],
    gs: [Grid; 4],
    out: &mut [u32],
) {
    use std::arch::x86_64::*;
    let len = out.len();
    let (lo, hi) = (_mm256_set1_ps(-512.0), _mm256_set1_ps(512.0));
    let inv0 = _mm256_set1_ps(gs[0].inv());
    let inv1 = _mm256_set1_ps(gs[1].inv());
    let inv2 = _mm256_set1_ps(gs[2].inv());
    let inv3 = _mm256_set1_ps(gs[3].inv());
    let (z0, z1, z2, z3) = (gs[0].zero as i16, gs[1].zero as i16, gs[2].zero as i16, gs[3].zero as i16);
    let z01 = _mm256_setr_epi16(z0, z0, z0, z0, z1, z1, z1, z1, z0, z0, z0, z0, z1, z1, z1, z1);
    let z23 = _mm256_setr_epi16(z2, z2, z2, z2, z3, z3, z3, z3, z2, z2, z2, z2, z3, z3, z3, z3);
    let transpose = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7,
        11, 15,
    );
    let mut i = 0;
    // SAFETY: `i + 8 <= len` and all five slices have length `len`.
    unsafe {
        while i + 8 <= len {
            let v0 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p0.as_ptr().add(i)), inv0), lo),
                hi,
            ));
            let v1 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p1.as_ptr().add(i)), inv1), lo),
                hi,
            ));
            let v2 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p2.as_ptr().add(i)), inv2), lo),
                hi,
            ));
            let v3 = _mm256_cvtps_epi32(_mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(p3.as_ptr().add(i)), inv3), lo),
                hi,
            ));
            let p01 = _mm256_adds_epi16(_mm256_packs_epi32(v0, v1), z01);
            let p23 = _mm256_adds_epi16(_mm256_packs_epi32(v2, v3), z23);
            let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(p01, p23), transpose);
            _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, bytes);
            i += 8;
        }
    }
    for j in i..len {
        out[j] = u32::from_le_bytes([
            gs[0].quantize(p0[j]),
            gs[1].quantize(p1[j]),
            gs[2].quantize(p2[j]),
            gs[3].quantize(p3[j]),
        ]);
    }
}

/// Split `src[0..w]` into `even[j] = src[2*j]` and `odd[j] = src[2*j + 1]` (padded with `pad` when `2*j + 1 >= w`)
/// for `0 <= j < pairs`, using AVX2 `vshufps` + `vpermq` when available.
pub(super) fn deinterleave_even_odd(src: &[u32], even: &mut [u32], odd: &mut [u32], pad: u32) {
    let pairs = even.len();
    assert_eq!(odd.len(), pairs, "q8::deinterleave_even_odd: output length mismatch");
    #[cfg(target_arch = "x86_64")]
    if kernel::uses_avx2() {
        // SAFETY: AVX2 was detected; bounds checked inside.
        return unsafe { deinterleave_even_odd_avx2(src, even, odd, pad) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if kernel::uses_wasm_simd() {
        return wasm_simd::deinterleave_even_odd_wasm(src, even, odd, pad);
    }
    deinterleave_even_odd_scalar(src, even, odd, pad);
}

#[inline]
fn deinterleave_even_odd_scalar(src: &[u32], even: &mut [u32], odd: &mut [u32], pad: u32) {
    let w = src.len();
    for j in 0..even.len() {
        let x0 = 2 * j;
        even[j] = if x0 < w { src[x0] } else { pad };
        odd[j] = if x0 + 1 < w { src[x0 + 1] } else { pad };
    }
}

/// # Safety
/// Caller must ensure AVX2 and FMA are available and `odd.len() == even.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn deinterleave_even_odd_avx2(src: &[u32], even: &mut [u32], odd: &mut [u32], pad: u32) {
    use std::arch::x86_64::*;
    let w = src.len();
    let pairs = even.len();
    debug_assert_eq!(odd.len(), pairs);
    let full = (w / 16).min(pairs / 8);
    let mut j = 0;
    // SAFETY: `2 * (j + 8) <= w` and `j + 8 <= pairs`.
    unsafe {
        for _ in 0..full {
            let sp = src.as_ptr().add(2 * j) as *const __m256i;
            let lo = _mm256_castsi256_ps(_mm256_loadu_si256(sp));
            let hi = _mm256_castsi256_ps(_mm256_loadu_si256(sp.add(1)));
            let ev = _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0x88>(lo, hi)));
            let od = _mm256_permute4x64_epi64::<0xD8>(_mm256_castps_si256(_mm256_shuffle_ps::<0xDD>(lo, hi)));
            _mm256_storeu_si256(even.as_mut_ptr().add(j) as *mut __m256i, ev);
            _mm256_storeu_si256(odd.as_mut_ptr().add(j) as *mut __m256i, od);
            j += 8;
        }
    }
    for jj in j..pairs {
        let x0 = 2 * jj;
        even[jj] = if x0 < w { src[x0] } else { pad };
        odd[jj] = if x0 + 1 < w { src[x0 + 1] } else { pad };
    }
}

/// `[n][c][hw]` `f32` (`c` a multiple of 4) into C4 codes on `grid`.
pub fn quantize(x: &[f32], n: usize, c: usize, hw: usize, grid: Grid, out: &mut [u32]) {
    assert!(c.is_multiple_of(4), "q8::quantize: {c} channels");
    let total = n.checked_mul(c).and_then(|v| v.checked_mul(hw)).expect("q8::quantize: size overflow");
    let out_words = total / 4;
    assert!(x.len() >= total && out.len() >= out_words, "q8::quantize: buffer too short");
    let groups = c / 4;
    if total == 0 {
        return;
    }
    if total < crate::cpu::PAR_MIN {
        for ig in 0..n * groups {
            let (img, g) = (ig / groups, ig % groups);
            let base = (img * c + 4 * g) * hw;
            let o = &mut out[ig * hw..(ig + 1) * hw];
            quantize_group(
                &x[base..base + hw],
                &x[base + hw..base + 2 * hw],
                &x[base + 2 * hw..base + 3 * hw],
                &x[base + 3 * hw..base + 4 * hw],
                [grid; 4],
                o,
            );
        }
        return;
    }
    const BLOCK: usize = 4096;
    let blocks = hw.div_ceil(BLOCK);
    let optr = crate::cpu::elementwise::SendPtr(out.as_mut_ptr());
    let optr = &optr;
    crate::cpu::par_for(n * groups * blocks, &|t| {
        let (ig, b) = (t / blocks, t % blocks);
        let (img, g) = (ig / groups, ig % groups);
        let p0 = b * BLOCK;
        let len = BLOCK.min(hw - p0);
        let base = (img * c + 4 * g) * hw + p0;
        // SAFETY: each `(ig, b)` owns disjoint words `ig * hw + p0..+len` inside `out` (`out_words <= out.len()`).
        let o = unsafe { std::slice::from_raw_parts_mut(optr.0.add(ig * hw + p0), len) };
        quantize_group(
            &x[base..base + len],
            &x[base + hw..base + hw + len],
            &x[base + 2 * hw..base + 2 * hw + len],
            &x[base + 3 * hw..base + 3 * hw + len],
            [grid; 4],
            o,
        );
    });
}

/// `[n][c][hw]` `f32` (`c` a multiple of 4) into C4 codes on per-channel `grids` (`grids.len() == c`).
pub fn quantize_per_channel(x: &[f32], n: usize, c: usize, hw: usize, grids: &[Grid], out: &mut [u32]) {
    assert!(c.is_multiple_of(4), "q8::quantize_per_channel: {c} channels");
    assert_eq!(grids.len(), c, "q8::quantize_per_channel: grids length");
    let total =
        n.checked_mul(c).and_then(|v| v.checked_mul(hw)).expect("q8::quantize_per_channel: size overflow");
    let out_words = total / 4;
    assert!(x.len() >= total && out.len() >= out_words, "q8::quantize_per_channel: buffer too short");
    let groups = c / 4;
    if total == 0 {
        return;
    }
    if total < crate::cpu::PAR_MIN {
        for ig in 0..n * groups {
            let (img, g) = (ig / groups, ig % groups);
            let base = (img * c + 4 * g) * hw;
            let gs = [grids[4 * g], grids[4 * g + 1], grids[4 * g + 2], grids[4 * g + 3]];
            let o = &mut out[ig * hw..(ig + 1) * hw];
            quantize_group(
                &x[base..base + hw],
                &x[base + hw..base + 2 * hw],
                &x[base + 2 * hw..base + 3 * hw],
                &x[base + 3 * hw..base + 4 * hw],
                gs,
                o,
            );
        }
        return;
    }
    const BLOCK: usize = 4096;
    let blocks = hw.div_ceil(BLOCK);
    let optr = crate::cpu::elementwise::SendPtr(out.as_mut_ptr());
    let optr = &optr;
    crate::cpu::par_for(n * groups * blocks, &|t| {
        let (ig, b) = (t / blocks, t % blocks);
        let (img, g) = (ig / groups, ig % groups);
        let p0 = b * BLOCK;
        let len = BLOCK.min(hw - p0);
        let base = (img * c + 4 * g) * hw + p0;
        let gs = [grids[4 * g], grids[4 * g + 1], grids[4 * g + 2], grids[4 * g + 3]];
        // SAFETY: each `(ig, b)` owns disjoint words `ig * hw + p0..+len` inside `out` (`out_words <= out.len()`).
        let o = unsafe { std::slice::from_raw_parts_mut(optr.0.add(ig * hw + p0), len) };
        quantize_group(
            &x[base..base + len],
            &x[base + hw..base + hw + len],
            &x[base + 2 * hw..base + 2 * hw + len],
            &x[base + 3 * hw..base + 3 * hw + len],
            gs,
            o,
        );
    });
}

/// Dequantise one C4 group (`4` channels) from `src[0..len]` into four `f32` planes `p0..p3[0..len]` in a single
/// read of `src`, bit-exact scalar == AVX2.
fn dequantize_group(
    src: &[u32],
    gs: [Grid; 4],
    p0: &mut [f32],
    p1: &mut [f32],
    p2: &mut [f32],
    p3: &mut [f32],
) {
    let len = src.len();
    assert!(p0.len() >= len && p1.len() >= len && p2.len() >= len && p3.len() >= len);
    #[cfg(target_arch = "x86_64")]
    if kernel::uses_avx2() {
        // SAFETY: AVX2 and FMA were detected; slice lengths asserted above.
        return unsafe { dequantize_group_avx2(src, gs, p0, p1, p2, p3) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if kernel::uses_wasm_simd() {
        return wasm_simd::dequantize_group_wasm(src, gs, p0, p1, p2, p3);
    }
    for i in 0..src.len() {
        let w = src[i];
        p0[i] = gs[0].dequantize(w as u8);
        p1[i] = gs[1].dequantize((w >> 8) as u8);
        p2[i] = gs[2].dequantize((w >> 16) as u8);
        p3[i] = gs[3].dequantize((w >> 24) as u8);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dequantize_group_avx2(
    src: &[u32],
    gs: [Grid; 4],
    p0: &mut [f32],
    p1: &mut [f32],
    p2: &mut [f32],
    p3: &mut [f32],
) {
    use std::arch::x86_64::*;
    let len = src.len();
    let mask = _mm256_set1_epi32(0xFF);
    let vz0 = _mm256_set1_epi32(gs[0].zero as i32);
    let vz1 = _mm256_set1_epi32(gs[1].zero as i32);
    let vz2 = _mm256_set1_epi32(gs[2].zero as i32);
    let vz3 = _mm256_set1_epi32(gs[3].zero as i32);
    let vs0 = _mm256_set1_ps(gs[0].scale);
    let vs1 = _mm256_set1_ps(gs[1].scale);
    let vs2 = _mm256_set1_ps(gs[2].scale);
    let vs3 = _mm256_set1_ps(gs[3].scale);
    let mut i = 0;
    // SAFETY: `i + 8 <= len` and all five slices have length `len`.
    unsafe {
        while i + 8 <= len {
            let w = _mm256_loadu_si256(src.as_ptr().add(i) as *const __m256i);
            let q0 = _mm256_and_si256(w, mask);
            let q1 = _mm256_and_si256(_mm256_srli_epi32::<8>(w), mask);
            let q2 = _mm256_and_si256(_mm256_srli_epi32::<16>(w), mask);
            let q3 = _mm256_srli_epi32::<24>(w);
            _mm256_storeu_ps(
                p0.as_mut_ptr().add(i),
                _mm256_mul_ps(vs0, _mm256_cvtepi32_ps(_mm256_sub_epi32(q0, vz0))),
            );
            _mm256_storeu_ps(
                p1.as_mut_ptr().add(i),
                _mm256_mul_ps(vs1, _mm256_cvtepi32_ps(_mm256_sub_epi32(q1, vz1))),
            );
            _mm256_storeu_ps(
                p2.as_mut_ptr().add(i),
                _mm256_mul_ps(vs2, _mm256_cvtepi32_ps(_mm256_sub_epi32(q2, vz2))),
            );
            _mm256_storeu_ps(
                p3.as_mut_ptr().add(i),
                _mm256_mul_ps(vs3, _mm256_cvtepi32_ps(_mm256_sub_epi32(q3, vz3))),
            );
            i += 8;
        }
    }
    for j in i..len {
        let w = src[j];
        p0[j] = gs[0].dequantize(w as u8);
        p1[j] = gs[1].dequantize((w >> 8) as u8);
        p2[j] = gs[2].dequantize((w >> 16) as u8);
        p3[j] = gs[3].dequantize((w >> 24) as u8);
    }
}

/// C4 codes on `grid` into `[n][c][hw]` `f32`.
pub fn dequantize(x: C4<'_>, n: usize, hw: usize, grid: Grid, out: &mut [f32]) {
    x.check(n, hw);
    let c = x.c;
    let total = n.checked_mul(c).and_then(|v| v.checked_mul(hw)).expect("q8::dequantize: size overflow");
    assert!(out.len() >= total, "q8::dequantize: output buffer too short");
    if total == 0 {
        return;
    }
    let groups = c / 4;
    let run = |ig: usize, o: &mut [f32]| {
        let (img, g) = (ig / groups, ig % groups);
        let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
        let (p0, rest) = o.split_at_mut(hw);
        let (p1, rest) = rest.split_at_mut(hw);
        let (p2, p3) = rest.split_at_mut(hw);
        dequantize_group(src, [grid; 4], p0, p1, p2, p3);
    };
    if total < crate::cpu::PAR_MIN {
        for (ig, o) in out[..total].chunks_exact_mut(4 * hw).enumerate() {
            run(ig, o);
        }
    } else {
        crate::cpu::par_chunks(&mut out[..total], 4 * hw, &run);
    }
}

/// C4 codes on per-channel `grids` (`grids.len() == x.c`) into `[n][c][hw]` `f32`.
pub fn dequantize_per_channel(x: C4<'_>, n: usize, hw: usize, grids: &[Grid], out: &mut [f32]) {
    x.check(n, hw);
    let c = x.c;
    assert_eq!(grids.len(), c, "q8::dequantize_per_channel: grids length");
    let total =
        n.checked_mul(c).and_then(|v| v.checked_mul(hw)).expect("q8::dequantize_per_channel: size overflow");
    assert!(out.len() >= total, "q8::dequantize_per_channel: output buffer too short");
    if total == 0 {
        return;
    }
    let groups = c / 4;
    let run = |ig: usize, o: &mut [f32]| {
        let (img, g) = (ig / groups, ig % groups);
        let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
        let gs = [grids[4 * g], grids[4 * g + 1], grids[4 * g + 2], grids[4 * g + 3]];
        let (p0, rest) = o.split_at_mut(hw);
        let (p1, rest) = rest.split_at_mut(hw);
        let (p2, p3) = rest.split_at_mut(hw);
        dequantize_group(src, gs, p0, p1, p2, p3);
    };
    if total < crate::cpu::PAR_MIN {
        for (ig, o) in out[..total].chunks_exact_mut(4 * hw).enumerate() {
            run(ig, o);
        }
    } else {
        crate::cpu::par_chunks(&mut out[..total], 4 * hw, &run);
    }
}

/// Byte-wise maximum of two C4 words.
#[inline(always)]
fn max4(a: u32, b: u32) -> u32 {
    let (a, b) = (a.to_le_bytes(), b.to_le_bytes());
    u32::from_le_bytes([0, 1, 2, 3].map(|t| a[t].max(b[t])))
}

fn max_pool_plane(src: &[u32], h: usize, w: usize, r: usize, tmp: &mut [u32], o: &mut [u32]) {
    let hw = h.checked_mul(w).expect("q8::max_pool_plane: h * w overflow");
    assert!(src.len() >= hw && tmp.len() >= hw && o.len() >= hw);
    if hw == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if kernel::uses_avx2() {
        // SAFETY: AVX2 was detected; buffer lengths asserted above.
        return unsafe { max_pool_plane_avx2(src, h, w, r, tmp, o) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if kernel::uses_wasm_simd() {
        return wasm_simd::max_pool_plane_wasm(src, h, w, r, tmp, o);
    }
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for xo in 0..w {
            let (lo, hi) = (xo.saturating_sub(r), xo.saturating_add(r).min(w - 1));
            tmp[y * w + xo] = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
    }
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), y.saturating_add(r).min(h - 1));
        for xo in 0..w {
            o[y * w + xo] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xo]));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn max_pool_plane_avx2(src: &[u32], h: usize, w: usize, r: usize, tmp: &mut [u32], o: &mut [u32]) {
    use std::arch::x86_64::*;
    let span = r.saturating_mul(2).saturating_add(1);
    // Horizontal pass: 8 C4 words (= 32 bytes) per `_mm256_max_epu8`.
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let dst = &mut tmp[y * w..(y + 1) * w];
        let left_end = r.min(w);
        for (xo, d) in dst.iter_mut().enumerate().take(left_end) {
            let hi = xo.saturating_add(r).min(w - 1);
            *d = row[0..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
        let mut xo = left_end;
        let inner_end = w.saturating_sub(r);
        // SAFETY: `xo >= r` and `xo + 8 <= w - r` implies `2 * r + 8 <= w`, so `span = 2*r + 1 < w` and
        // `xo - r + d + 8 <= w` for all `d in 0..span`.
        unsafe {
            while xo + 8 <= inner_end {
                let base = row.as_ptr().add(xo - r);
                let mut m = _mm256_loadu_si256(base as *const __m256i);
                for d in 1..span {
                    m = _mm256_max_epu8(m, _mm256_loadu_si256(base.add(d) as *const __m256i));
                }
                _mm256_storeu_si256(dst.as_mut_ptr().add(xo) as *mut __m256i, m);
                xo += 8;
            }
        }
        for (xx, d) in dst.iter_mut().enumerate().skip(xo) {
            let (lo, hi) = (xx.saturating_sub(r), xx.saturating_add(r).min(w - 1));
            *d = row[lo..=hi].iter().fold(0u32, |m, &v| max4(m, v));
        }
    }
    // Vertical pass: 8 C4 words per `_mm256_max_epu8`.
    for y in 0..h {
        let lo = y.saturating_sub(r);
        let hi = y.saturating_add(r).min(h - 1);
        let dst = &mut o[y * w..(y + 1) * w];
        let mut xo = 0;
        // SAFETY: `yy * w + xo + 8 <= h * w`.
        unsafe {
            while xo + 8 <= w {
                let mut m = _mm256_loadu_si256(tmp.as_ptr().add(lo * w + xo) as *const __m256i);
                for yy in (lo + 1)..=hi {
                    m = _mm256_max_epu8(
                        m,
                        _mm256_loadu_si256(tmp.as_ptr().add(yy * w + xo) as *const __m256i),
                    );
                }
                _mm256_storeu_si256(dst.as_mut_ptr().add(xo) as *mut __m256i, m);
                xo += 8;
            }
        }
        for xx in xo..w {
            dst[xx] = (lo..=hi).fold(0u32, |m, yy| max4(m, tmp[yy * w + xx]));
        }
    }
}

thread_local! {
    /// Per-thread scratch for small serial [`max_pool`] planes (`hw < MIN_POOLED`), avoiding a heap allocation
    /// per call on e.g. YOLO's `20×20` SPPF blocks.
    static MAX_POOL_TMP: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// RAII wrapper that obtains `n` scratch `u32` words from the CPU buffer cache and recycles them on drop.
struct ScratchWords(Vec<u32>);

impl ScratchWords {
    fn new(n: usize) -> Self {
        Self(words(n))
    }
}

impl Drop for ScratchWords {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.0);
        if !v.is_empty() {
            recycle(v);
        }
    }
}

/// `k × k` max pool, stride 1, padding `k / 2` (padding never wins), of C4 codes into a contiguous C4 `out`
/// (`[n][c / 4][h · w]`). Exact: max commutes with a monotone quantiser (per-tensor or per-channel).
pub fn max_pool(x: C4<'_>, n: usize, h: usize, w: usize, k: usize, out: &mut [u32]) {
    use rayon::prelude::*;

    let hw = h.checked_mul(w).expect("q8::max_pool: h * w overflow");
    x.check(n, hw);
    let groups = x.c / 4;
    let total = n.checked_mul(groups).and_then(|v| v.checked_mul(hw)).expect("q8::max_pool: size overflow");
    assert!(out.len() >= total, "q8::max_pool: output buffer too short");
    if total == 0 {
        return;
    }
    let r = k / 2;
    if total < crate::cpu::PAR_MIN {
        let mut run_serial = |tmp: &mut [u32]| {
            for (ig, o) in out[..total].chunks_exact_mut(hw).enumerate() {
                let (img, g) = (ig / groups, ig % groups);
                let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
                max_pool_plane(src, h, w, r, tmp, o);
            }
        };
        if hw < crate::cpu::pool_alloc::MIN_POOLED {
            let ran = MAX_POOL_TMP.try_with(|cell| {
                let Ok(mut tmp) = cell.try_borrow_mut() else { return false };
                if tmp.len() < hw {
                    tmp.resize(hw, 0);
                }
                run_serial(&mut tmp[..hw]);
                true
            });
            if !matches!(ran, Ok(true)) {
                let mut tmp = vec![0u32; hw];
                run_serial(&mut tmp);
            }
        } else {
            let mut tmp = ScratchWords::new(hw);
            run_serial(&mut tmp.0);
        }
        return;
    }
    out[..total].par_chunks_exact_mut(hw).enumerate().for_each_init(
        || ScratchWords::new(hw),
        |tmp, (ig, o)| {
            let (img, g) = (ig / groups, ig % groups);
            let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
            max_pool_plane(src, h, w, r, &mut tmp.0, o);
        },
    );
}

fn upsample2_plane(src: &[u32], h: usize, w: usize, o: &mut [u32]) {
    let hw = h.checked_mul(w).expect("q8::upsample2_plane: h * w overflow");
    let ow = w.checked_mul(2).expect("q8::upsample2_plane: width overflow");
    let ohw = hw.checked_mul(4).expect("q8::upsample2_plane: output size overflow");
    assert!(src.len() >= hw && o.len() >= ohw);
    #[cfg(target_arch = "x86_64")]
    if kernel::uses_avx2() {
        // SAFETY: AVX2 was detected; buffer lengths asserted above.
        return unsafe { upsample2_plane_avx2(src, h, w, o) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if kernel::uses_wasm_simd() {
        return wasm_simd::upsample2_plane_wasm(src, h, w, o);
    }
    for y in 0..h {
        let srow = &src[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        for (x, &v) in srow.iter().enumerate() {
            d0[2 * x] = v;
            d0[2 * x + 1] = v;
            d1[2 * x] = v;
            d1[2 * x + 1] = v;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn upsample2_plane_avx2(src: &[u32], h: usize, w: usize, o: &mut [u32]) {
    use std::arch::x86_64::*;
    let ow = 2 * w;
    for y in 0..h {
        let srow = &src[y * w..(y + 1) * w];
        let (d0, d1) = o[2 * y * ow..(2 * y + 2) * ow].split_at_mut(ow);
        let mut x = 0;
        // SAFETY: `x + 8 <= w` and `2 * x + 16 <= ow`.
        unsafe {
            while x + 8 <= w {
                let v = _mm256_loadu_si256(srow.as_ptr().add(x) as *const __m256i);
                let lo = _mm256_unpacklo_epi32(v, v);
                let hi = _mm256_unpackhi_epi32(v, v);
                let out0 = _mm256_permute2x128_si256::<0x20>(lo, hi);
                let out1 = _mm256_permute2x128_si256::<0x31>(lo, hi);
                _mm256_storeu_si256(d0.as_mut_ptr().add(2 * x) as *mut __m256i, out0);
                _mm256_storeu_si256(d0.as_mut_ptr().add(2 * x + 8) as *mut __m256i, out1);
                _mm256_storeu_si256(d1.as_mut_ptr().add(2 * x) as *mut __m256i, out0);
                _mm256_storeu_si256(d1.as_mut_ptr().add(2 * x + 8) as *mut __m256i, out1);
                x += 8;
            }
        }
        for xx in x..w {
            let v = srow[xx];
            d0[2 * xx] = v;
            d0[2 * xx + 1] = v;
            d1[2 * xx] = v;
            d1[2 * xx + 1] = v;
        }
    }
}

/// Nearest-neighbour upsample by 2 of C4 codes into a contiguous C4 `out` (`[n][c / 4][2h · 2w]`).
pub fn upsample2(x: C4<'_>, n: usize, h: usize, w: usize, out: &mut [u32]) {
    let hw = h.checked_mul(w).expect("q8::upsample2: h * w overflow");
    x.check(n, hw);
    let groups = x.c / 4;
    let ohw = hw.checked_mul(4).expect("q8::upsample2: output plane overflow");
    let total = n.checked_mul(groups).and_then(|v| v.checked_mul(ohw)).expect("q8::upsample2: size overflow");
    assert!(out.len() >= total, "q8::upsample2: output buffer too short");
    if total == 0 {
        return;
    }
    if total < 2 * crate::cpu::PAR_MIN {
        for (ig, o) in out[..total].chunks_exact_mut(ohw).enumerate() {
            let (img, g) = (ig / groups, ig % groups);
            let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
            upsample2_plane(src, h, w, o);
        }
        return;
    }
    crate::cpu::par_chunks(&mut out[..total], ohw, &|ig, o| {
        let (img, g) = (ig / groups, ig % groups);
        let src = &x.data[x.offset + img * x.img_stride + g * hw..][..hw];
        upsample2_plane(src, h, w, o);
    });
}
