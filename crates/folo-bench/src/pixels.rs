//! Pixel kernels behind the conditions: sRGB tables, seeded noise, separable and sparse convolution, area and
//! bilinear resampling, all on interleaved RGB.
//!
//! Deterministic by construction, so a variant has the same bytes on every machine and thread count:
//! - only IEEE basic operations (`+ - * /`, `sqrt`, comparisons, `floor`/`round`) touch pixels or build tables;
//!   the few transcendental values (the sRGB curve, Gaussian weights) come from [`ln`] / [`exp`] written here in
//!   those operations, never from the platform's libm, whose last bits differ between systems;
//! - every output value is computed by one thread in a fixed order (rayon splits rows, never a sum);
//! - noise is drawn from a generator seeded by the variant's seed and the row, so where it falls does not depend
//!   on how rows are scheduled.

use folo_core::rng::Rng;
use rayon::prelude::*;
use std::f64::consts::LN_2;
use std::sync::OnceLock;

/// Images with at least this many pixels are processed with rayon, a row per task.
pub(crate) const PAR_PIXELS: usize = 1 << 16;

// ---------------------------------------------------------------------------------------------------------------
// Math from basic operations
// ---------------------------------------------------------------------------------------------------------------

/// Natural logarithm of a positive, finite, normal `x`, to about 1e-15 relative error, from basic operations.
pub(crate) fn ln(x: f64) -> f64 {
    debug_assert!(x.is_normal() && x > 0.0, "ln({x})");
    let bits = x.to_bits();
    let mut e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    // The mantissa in [1, 2), then folded into [0.707, 1.414] so the series below converges fast.
    let mut m = f64::from_bits((bits & ((1u64 << 52) - 1)) | (1023u64 << 52));
    if m > std::f64::consts::SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    // ln(m) = 2 atanh(z) = 2 (z + z³/3 + z⁵/5 + ...), |z| <= 0.172.
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let (mut term, mut sum) = (z, 0.0);
    for k in 0..20 {
        sum += term / (2 * k + 1) as f64;
        term *= z2;
    }
    2.0 * sum + e as f64 * LN_2
}

/// `e^x` for `x <= 700`, to about 1e-15 relative error, from basic operations. Underflows to 0 below -700.
pub(crate) fn exp(x: f64) -> f64 {
    if x < -700.0 {
        return 0.0;
    }
    debug_assert!(x <= 700.0, "exp({x})");
    // x = k ln 2 + r with |r| <= ln 2 / 2 (ln 2 split in two, fdlibm's, so k · LN2_HI is exact); e^r by its
    // Taylor series, 2^k by building the float.
    const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
    let k = (x / LN_2).round();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let (mut term, mut sum) = (1.0, 1.0);
    for i in 1..25 {
        term *= r / i as f64;
        sum += term;
    }
    sum * f64::from_bits(((1023 + k as i64) as u64) << 52)
}

/// `x^y` for `x >= 0` (0 for `x = 0`).
pub(crate) fn pow(x: f64, y: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { exp(y * ln(x)) }
}

// ---------------------------------------------------------------------------------------------------------------
// sRGB
// ---------------------------------------------------------------------------------------------------------------

/// Entries of the encode table, indexed by `sqrt(linear)` (fine steps near black, where sRGB is steep).
const ENC_N: usize = 1 << 14;

/// Tables between 8-bit sRGB and linear light in `[0, 1]`.
pub(crate) struct Srgb {
    lin: [f32; 256],
    enc: Vec<u8>,
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { pow((v + 0.055) / 1.055, 2.4) }
}

fn linear_to_srgb(l: f64) -> f64 {
    if l <= 0.0031308 { 12.92 * l } else { 1.055 * pow(l, 1.0 / 2.4) - 0.055 }
}

/// The shared tables, built on first use.
pub(crate) fn srgb() -> &'static Srgb {
    static TABLES: OnceLock<Srgb> = OnceLock::new();
    TABLES.get_or_init(|| Srgb {
        lin: std::array::from_fn(|c| srgb_to_linear(c as f64 / 255.0) as f32),
        enc: (0..ENC_N)
            .map(|i| {
                let s = i as f64 / (ENC_N - 1) as f64;
                (linear_to_srgb(s * s) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            })
            .collect(),
    })
}

impl Srgb {
    /// Linear light of an 8-bit sRGB value.
    #[inline]
    pub(crate) fn lin(&self, v: u8) -> f32 {
        self.lin[v as usize]
    }

    /// The 8-bit sRGB value of linear light `l` (clamped to `[0, 1]`; NaN gives 0).
    #[inline]
    pub(crate) fn enc(&self, l: f32) -> u8 {
        let s = l.clamp(0.0, 1.0).sqrt();
        self.enc[(s * (ENC_N - 1) as f32 + 0.5) as usize]
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Seeds and noise
// ---------------------------------------------------------------------------------------------------------------

/// SplitMix64's finaliser.
#[inline]
pub(crate) fn fmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A well-mixed 64-bit value of the pair `(a, b)`.
#[inline]
pub(crate) fn mix(a: u64, b: u64) -> u64 {
    fmix(a ^ fmix(b.wrapping_add(0x9E37_79B9_7F4A_7C15)))
}

/// The noise generator of one row of a variant.
#[inline]
pub(crate) fn row_rng(seed: u64, row: usize) -> Rng {
    Rng::seed(mix(seed, row as u64))
}

/// An approximately standard normal sample: the sum of four uniform 16-bit integers, centred and scaled to unit
/// variance (Irwin–Hall, within ±3.46σ). Integer arithmetic and one multiply, so the same on every platform.
#[inline]
pub(crate) fn gauss(rng: &mut Rng) -> f32 {
    let r = rng.next_u64();
    let s = (r & 0xFFFF) + ((r >> 16) & 0xFFFF) + ((r >> 32) & 0xFFFF) + (r >> 48);
    // Mean 4 · 65535 / 2; variance 4 · (65536² − 1) / 12.
    (s as f32 - 131_070.0) * (1.0 / 37_837.23)
}

// ---------------------------------------------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------------------------------------------

/// Calls `f(y, row)` for every row of `buf` (rows of `row_len` values), in parallel for large images.
pub(crate) fn for_rows<T: Send>(buf: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 {
        return;
    }
    if buf.len() / 3 >= PAR_PIXELS {
        buf.par_chunks_mut(row_len).enumerate().for_each(|(y, r)| f(y, r));
    } else {
        buf.chunks_mut(row_len).enumerate().for_each(|(y, r)| f(y, r));
    }
}

/// [`for_rows`] with a scratch value per task, made by `init`.
pub(crate) fn for_rows_with<T: Send, S>(
    buf: &mut [T],
    row_len: usize,
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, &mut [T]) + Sync + Send,
) {
    if row_len == 0 {
        return;
    }
    if buf.len() / 3 >= PAR_PIXELS {
        buf.par_chunks_mut(row_len).enumerate().for_each_init(&init, |s, (y, r)| f(s, y, r));
    } else {
        let mut s = init();
        buf.chunks_mut(row_len).enumerate().for_each(|(y, r)| f(&mut s, y, r));
    }
}

/// Register tile width in floats for convolution and vertical resampling (8 RGB pixels; exact multiple of 3 so
/// per-channel affine epilogues such as haze align without `% 3` drift across tiles).
const CONV_TILE: usize = 24;

/// Round and clamp a 0..=255 float value to `u8` (NaN gives 0).
#[inline(always)]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 255.0) + 0.5) as u8
}

/// The per-channel mean of interleaved RGB bytes, rounded (exact integer sums). Black for an empty image.
pub(crate) fn mean_rgb(raw: &[u8]) -> [u8; 3] {
    let mut sum = [0u64; 3];
    for px in raw.as_chunks::<3>().0 {
        for c in 0..3 {
            sum[c] += px[c] as u64;
        }
    }
    let n = (raw.len() / 3) as u64;
    if n == 0 {
        return [0; 3];
    }
    sum.map(|s| ((s + n / 2) / n) as u8)
}

// ---------------------------------------------------------------------------------------------------------------
// Float images
// ---------------------------------------------------------------------------------------------------------------

/// An interleaved RGB image of `f32` values on the 0..=255 scale.
pub(crate) struct Plane {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) data: Vec<f32>,
}

/// A normalised Gaussian kernel of standard deviation `sigma` px, radius `ceil(3 sigma)`; `[1]` for
/// `sigma < 0.05`.
pub(crate) fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma.is_nan() || sigma < 0.05 {
        return vec![1.0];
    }
    let r = (3.0 * sigma as f64).ceil() as usize;
    let d = 2.0 * sigma as f64 * sigma as f64;
    let w: Vec<f64> = (0..=2 * r).map(|i| exp(-((i as f64 - r as f64).powi(2)) / d)).collect();
    let sum: f64 = w.iter().sum();
    w.iter().map(|&v| (v / sum) as f32).collect()
}

/// Horizontal separable pass from 8-bit interleaved RGB into a `w x h x 3` float buffer, register-tiled in
/// exact `k` order (bit-identical to `Plane::from_u8` + `axpy`).
fn convolve_h_from_u8(raw: &[u8], w: usize, h: usize, kx: &[f32]) -> Vec<f32> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let stride = w * 3;
    assert_eq!(raw.len(), h * stride);
    let rx = kx.len() / 2;
    let mut tmp = vec![0.0f32; raw.len()];
    for_rows_with(
        &mut tmp,
        stride,
        || vec![0.0f32; (w + 2 * rx) * 3],
        |pad, y, out| {
            let src = &raw[y * stride..(y + 1) * stride];
            let l = [src[0] as f32, src[1] as f32, src[2] as f32];
            let r = [src[stride - 3] as f32, src[stride - 2] as f32, src[stride - 1] as f32];
            for i in 0..rx {
                pad[i * 3..i * 3 + 3].copy_from_slice(&l);
                pad[(rx + w + i) * 3..(rx + w + i) * 3 + 3].copy_from_slice(&r);
            }
            for (dst, &v) in pad[rx * 3..(rx + w) * 3].iter_mut().zip(src) {
                *dst = v as f32;
            }
            let mut x = 0;
            while x + CONV_TILE <= stride {
                let mut acc = [0.0f32; CONV_TILE];
                for (k, &wk) in kx.iter().enumerate() {
                    let s = &pad[k * 3 + x..k * 3 + x + CONV_TILE];
                    for i in 0..CONV_TILE {
                        acc[i] += wk * s[i];
                    }
                }
                out[x..x + CONV_TILE].copy_from_slice(&acc);
                x += CONV_TILE;
            }
            if x < stride {
                let rem = stride - x;
                let mut acc = [0.0f32; CONV_TILE];
                for (k, &wk) in kx.iter().enumerate() {
                    let s = &pad[k * 3 + x..k * 3 + x + rem];
                    for i in 0..rem {
                        acc[i] += wk * s[i];
                    }
                }
                out[x..stride].copy_from_slice(&acc[..rem]);
            }
        },
    );
    tmp
}

/// Vertical separable pass from `tmp` directly to 8-bit interleaved RGB, with an optional fused per-channel
/// haze affine `v * t + veil[c]`.
fn convolve_v_to_u8(tmp: &[f32], w: usize, h: usize, ky: &[f32], haze: Option<(f32, [f32; 3])>) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    const { assert!(CONV_TILE.is_multiple_of(3)) };
    let stride = w * 3;
    assert_eq!(tmp.len(), h * stride);
    let ry = ky.len() / 2;
    let mut out = vec![0u8; tmp.len()];
    for_rows_with(
        &mut out,
        stride,
        || Vec::<(usize, f32)>::with_capacity(ky.len()),
        |rows, y, out_row| {
            rows.clear();
            for (k, &wk) in ky.iter().enumerate() {
                let sy = (y + k).saturating_sub(ry).min(h - 1);
                rows.push((sy * stride, wk));
            }
            match haze {
                None => {
                    let mut x = 0;
                    while x + CONV_TILE <= stride {
                        let mut acc = [0.0f32; CONV_TILE];
                        for &(base, wk) in rows.iter() {
                            let s = &tmp[base + x..base + x + CONV_TILE];
                            for i in 0..CONV_TILE {
                                acc[i] += wk * s[i];
                            }
                        }
                        for i in 0..CONV_TILE {
                            out_row[x + i] = to_u8(acc[i]);
                        }
                        x += CONV_TILE;
                    }
                    if x < stride {
                        let rem = stride - x;
                        let mut acc = [0.0f32; CONV_TILE];
                        for &(base, wk) in rows.iter() {
                            let s = &tmp[base + x..base + x + rem];
                            for i in 0..rem {
                                acc[i] += wk * s[i];
                            }
                        }
                        for i in 0..rem {
                            out_row[x + i] = to_u8(acc[i]);
                        }
                    }
                }
                Some((t, veil)) => {
                    let mut x = 0;
                    while x + CONV_TILE <= stride {
                        let mut acc = [0.0f32; CONV_TILE];
                        for &(base, wk) in rows.iter() {
                            let s = &tmp[base + x..base + x + CONV_TILE];
                            for i in 0..CONV_TILE {
                                acc[i] += wk * s[i];
                            }
                        }
                        for i in 0..CONV_TILE {
                            out_row[x + i] = to_u8(acc[i] * t + veil[i % 3]);
                        }
                        x += CONV_TILE;
                    }
                    if x < stride {
                        let rem = stride - x;
                        let mut acc = [0.0f32; CONV_TILE];
                        for &(base, wk) in rows.iter() {
                            let s = &tmp[base + x..base + x + rem];
                            for i in 0..rem {
                                acc[i] += wk * s[i];
                            }
                        }
                        for i in 0..rem {
                            out_row[x + i] = to_u8(acc[i] * t + veil[i % 3]);
                        }
                    }
                }
            }
        },
    );
    out
}

/// 8-bit interleaved RGB `raw` (`w x h`) convolved with `kx` along rows and `ky` along columns (odd lengths,
/// centred), edges clamped, rounded directly back to 8-bit RGB.
pub(crate) fn convolve_separable_u8(raw: &[u8], w: usize, h: usize, kx: &[f32], ky: &[f32]) -> Vec<u8> {
    let tmp = convolve_h_from_u8(raw, w, h, kx);
    convolve_v_to_u8(&tmp, w, h, ky, None)
}

/// [`convolve_separable_u8`] with a fused haze epilogue `v * t + veil[c]` before rounding to `u8`.
pub(crate) fn convolve_separable_haze_u8(
    raw: &[u8],
    w: usize,
    h: usize,
    kx: &[f32],
    ky: &[f32],
    t: f32,
    veil: [f32; 3],
) -> Vec<u8> {
    let tmp = convolve_h_from_u8(raw, w, h, kx);
    convolve_v_to_u8(&tmp, w, h, ky, Some((t, veil)))
}

/// 8-bit interleaved RGB `raw` (`w x h`) convolved with a sparse kernel of `(dx, dy, weight)` taps, edges
/// clamped: output `(x, y)` is `Σ weight · p(x + dx, y + dy)`, rounded directly back to 8-bit RGB.
pub(crate) fn convolve_taps_u8(raw: &[u8], w: usize, h: usize, taps: &[(isize, isize, f32)]) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let stride = w * 3;
    assert_eq!(raw.len(), h * stride);
    let r = taps.iter().map(|t| t.0.unsigned_abs()).max().unwrap_or(0);
    // Rows padded by `r` clamped pixels on each side as u8 (4x smaller than f32), so every tap reads a
    // contiguous slice and widens on load inside the register-tiled accumulator loop.
    let pstride = (w + 2 * r) * 3;
    let mut padded = vec![0u8; h * pstride];
    for_rows(&mut padded, pstride, |y, pad| {
        let src = &raw[y * stride..(y + 1) * stride];
        let (l, rr) = (&src[..3], &src[stride - 3..]);
        for i in 0..r {
            pad[i * 3..i * 3 + 3].copy_from_slice(l);
            pad[(r + w + i) * 3..(r + w + i) * 3 + 3].copy_from_slice(rr);
        }
        pad[r * 3..(r + w) * 3].copy_from_slice(src);
    });
    let mut out = vec![0u8; raw.len()];
    for_rows_with(
        &mut out,
        stride,
        || Vec::<(usize, f32)>::with_capacity(taps.len()),
        |tap_offs, y, out_row| {
            tap_offs.clear();
            for &(dx, dy, wk) in taps {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let x0 = r.checked_add_signed(dx).expect("tap offset within padded row") * 3;
                tap_offs.push((sy * pstride + x0, wk));
            }
            let mut x = 0;
            while x + CONV_TILE <= stride {
                let mut acc = [0.0f32; CONV_TILE];
                for &(base, wk) in tap_offs.iter() {
                    let s = &padded[base + x..base + x + CONV_TILE];
                    for i in 0..CONV_TILE {
                        acc[i] += wk * s[i] as f32;
                    }
                }
                for i in 0..CONV_TILE {
                    out_row[x + i] = to_u8(acc[i]);
                }
                x += CONV_TILE;
            }
            if x < stride {
                let rem = stride - x;
                let mut acc = [0.0f32; CONV_TILE];
                for &(base, wk) in tap_offs.iter() {
                    let s = &padded[base + x..base + x + rem];
                    for i in 0..rem {
                        acc[i] += wk * s[i] as f32;
                    }
                }
                for i in 0..rem {
                    out_row[x + i] = to_u8(acc[i]);
                }
            }
        },
    );
    out
}

/// Resampling weights along one axis: output `i` is `Σ_j w[off[i] + j] · input[start[i] + j]` for
/// `j < off[i + 1] - off[i]`.
pub(crate) struct Taps {
    start: Vec<usize>,
    off: Vec<usize>,
    w: Vec<f32>,
}

impl Taps {
    #[inline]
    fn get(&self, i: usize) -> (usize, &[f32]) {
        (self.start[i], &self.w[self.off[i]..self.off[i + 1]])
    }
}

/// Area averaging from `n_in` to `n_out` pixels: output pixel `i` is the mean of the input over
/// `[i, i + 1) · n_in / n_out`, partial pixels weighted by their coverage. Input coordinate `x` lands on output
/// coordinate `x · n_out / n_in` exactly.
pub(crate) fn area_taps(n_in: usize, n_out: usize) -> Taps {
    let scale = n_in as f64 / n_out as f64;
    let mut t = Taps { start: Vec::with_capacity(n_out), off: vec![0], w: Vec::new() };
    for i in 0..n_out {
        let (a, b) = (i as f64 * scale, ((i + 1) as f64 * scale).min(n_in as f64));
        let j0 = a.floor() as usize;
        let j1 = (b.ceil() as usize).clamp(j0 + 1, n_in);
        t.start.push(j0);
        for j in j0..j1 {
            let cover = b.min((j + 1) as f64) - a.max(j as f64);
            t.w.push((cover.max(0.0) / scale) as f32);
        }
        t.off.push(t.w.len());
    }
    t
}

/// Bilinear interpolation from `n_in` to `n_out` pixels (pixel centres aligned, edges clamped).
pub(crate) fn bilinear_taps(n_in: usize, n_out: usize) -> Taps {
    let scale = n_in as f64 / n_out as f64;
    let mut t = Taps { start: Vec::with_capacity(n_out), off: vec![0], w: Vec::new() };
    for i in 0..n_out {
        let x = ((i as f64 + 0.5) * scale - 0.5).clamp(0.0, (n_in - 1) as f64);
        let j0 = x.floor() as usize;
        let f = x - j0 as f64;
        t.start.push(j0);
        t.w.push((1.0 - f) as f32);
        if j0 + 1 < n_in {
            t.w.push(f as f32);
        }
        t.off.push(t.w.len());
    }
    t
}

/// Horizontal resampling pass from 8-bit interleaved RGB into a `w2 x h x 3` float buffer.
fn resample_h_u8(raw: &[u8], w: usize, h: usize, w2: usize, xt: &Taps) -> Vec<f32> {
    let (stride, stride2) = (w * 3, w2 * 3);
    assert_eq!(raw.len(), h * stride);
    let mut tmp = vec![0.0f32; h * stride2];
    for_rows(&mut tmp, stride2, |y, out| {
        let src = &raw[y * stride..(y + 1) * stride];
        for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (s, ws) = xt.get(x);
            let mut acc = [0.0f32; 3];
            for (j, &wj) in ws.iter().enumerate() {
                let q = (s + j) * 3;
                acc[0] += wj * src[q] as f32;
                acc[1] += wj * src[q + 1] as f32;
                acc[2] += wj * src[q + 2] as f32;
            }
            *o = acc;
        }
    });
    tmp
}

/// Horizontal resampling pass from a float [`Plane`] into a `w2 x h x 3` float buffer.
fn resample_h_f32(p: &Plane, w2: usize, xt: &Taps) -> Vec<f32> {
    let (stride, stride2) = (p.w * 3, w2 * 3);
    let mut tmp = vec![0.0f32; p.h * stride2];
    for_rows(&mut tmp, stride2, |y, out| {
        let src = &p.data[y * stride..(y + 1) * stride];
        for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
            let (s, ws) = xt.get(x);
            let mut acc = [0.0f32; 3];
            for (j, &wj) in ws.iter().enumerate() {
                let q = (s + j) * 3;
                acc[0] += wj * src[q];
                acc[1] += wj * src[q + 1];
                acc[2] += wj * src[q + 2];
            }
            *o = acc;
        }
    });
    tmp
}

/// Register-tiled vertical resampling pass into a float [`Plane`].
fn resample_v_to_plane(tmp: &[f32], w2: usize, h2: usize, yt: &Taps) -> Plane {
    let stride2 = w2 * 3;
    let mut data = vec![0.0f32; h2 * stride2];
    for_rows(&mut data, stride2, |y, out| {
        let (s, ws) = yt.get(y);
        let mut x = 0;
        while x + CONV_TILE <= stride2 {
            let mut acc = [0.0f32; CONV_TILE];
            for (j, &wj) in ws.iter().enumerate() {
                let row = &tmp[(s + j) * stride2 + x..(s + j) * stride2 + x + CONV_TILE];
                for i in 0..CONV_TILE {
                    acc[i] += wj * row[i];
                }
            }
            out[x..x + CONV_TILE].copy_from_slice(&acc);
            x += CONV_TILE;
        }
        if x < stride2 {
            let rem = stride2 - x;
            let mut acc = [0.0f32; CONV_TILE];
            for (j, &wj) in ws.iter().enumerate() {
                let row = &tmp[(s + j) * stride2 + x..(s + j) * stride2 + x + rem];
                for i in 0..rem {
                    acc[i] += wj * row[i];
                }
            }
            out[x..stride2].copy_from_slice(&acc[..rem]);
        }
    });
    Plane { w: w2, h: h2, data }
}

/// Register-tiled vertical resampling pass directly to 8-bit interleaved RGB.
fn resample_v_to_u8(tmp: &[f32], w2: usize, h2: usize, yt: &Taps) -> Vec<u8> {
    let stride2 = w2 * 3;
    let mut out = vec![0u8; h2 * stride2];
    for_rows(&mut out, stride2, |y, out_row| {
        let (s, ws) = yt.get(y);
        let mut x = 0;
        while x + CONV_TILE <= stride2 {
            let mut acc = [0.0f32; CONV_TILE];
            for (j, &wj) in ws.iter().enumerate() {
                let row = &tmp[(s + j) * stride2 + x..(s + j) * stride2 + x + CONV_TILE];
                for i in 0..CONV_TILE {
                    acc[i] += wj * row[i];
                }
            }
            for i in 0..CONV_TILE {
                out_row[x + i] = to_u8(acc[i]);
            }
            x += CONV_TILE;
        }
        if x < stride2 {
            let rem = stride2 - x;
            let mut acc = [0.0f32; CONV_TILE];
            for (j, &wj) in ws.iter().enumerate() {
                let row = &tmp[(s + j) * stride2 + x..(s + j) * stride2 + x + rem];
                for i in 0..rem {
                    acc[i] += wj * row[i];
                }
            }
            for i in 0..rem {
                out_row[x + i] = to_u8(acc[i]);
            }
        }
    });
    out
}

/// 8-bit interleaved RGB `raw` (`w x h`) resampled to a `w2 x h2` float [`Plane`].
pub(crate) fn resample_u8_to_plane(
    raw: &[u8],
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    xt: &Taps,
    yt: &Taps,
) -> Plane {
    let tmp = resample_h_u8(raw, w, h, w2, xt);
    resample_v_to_plane(&tmp, w2, h2, yt)
}

/// Float [`Plane`] `p` resampled to `w2 x h2` and rounded directly to 8-bit interleaved RGB.
pub(crate) fn resample_plane_to_u8(p: &Plane, w2: usize, h2: usize, xt: &Taps, yt: &Taps) -> Vec<u8> {
    let tmp = resample_h_f32(p, w2, xt);
    resample_v_to_u8(&tmp, w2, h2, yt)
}

/// 8-bit interleaved RGB `raw` (`w x h`) resampled to `w2 x h2` and rounded directly to 8-bit interleaved RGB.
pub(crate) fn resample_u8_to_u8(
    raw: &[u8],
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    xt: &Taps,
    yt: &Taps,
) -> Vec<u8> {
    let tmp = resample_h_u8(raw, w, h, w2, xt);
    resample_v_to_u8(&tmp, w2, h2, yt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_matches_libm_closely() {
        for &x in &[1e-9, 0.003, 0.25, 0.5, 0.999, 1.0, 1.5, 7.0, 1234.5] {
            assert!((ln(x) - x.ln()).abs() < 1e-14 * x.ln().abs().max(1.0), "ln {x}");
        }
        for &x in &[-650.0, -20.0, -1.0, -0.1, 0.0, 0.3, 2.0, 40.0] {
            assert!((exp(x) - x.exp()).abs() <= 1e-14 * x.exp(), "exp {x}");
        }
        assert_eq!(pow(0.0, 2.4), 0.0);
        assert!((pow(0.5, 2.4) - 0.5f64.powf(2.4)).abs() < 1e-15);
    }

    #[test]
    fn srgb_round_trips_every_code() {
        let t = srgb();
        for c in 0..=255u8 {
            assert_eq!(t.enc(t.lin(c)), c, "{c}");
        }
        assert_eq!((t.enc(-1.0), t.enc(2.0), t.enc(f32::NAN)), (0, 255, 0));
    }

    #[test]
    fn gauss_has_unit_variance() {
        let mut rng = Rng::seed(3);
        let n = 200_000;
        let v: Vec<f64> = (0..n).map(|_| gauss(&mut rng) as f64).collect();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.01, "{mean} {var}");
    }

    #[test]
    fn taps_sum_to_one() {
        for (a, b) in [(640, 427), (7, 3), (5, 5), (3, 7), (1, 1), (10, 1)] {
            for t in [area_taps(a, b), bilinear_taps(a, b)] {
                for i in 0..b {
                    let s: f32 = t.get(i).1.iter().sum();
                    assert!((s - 1.0).abs() < 1e-5, "{a}->{b} at {i}: {s}");
                }
            }
        }
        let k = gaussian_kernel(2.0);
        assert_eq!(k.len(), 13);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    fn ref_separable(
        raw: &[u8],
        w: usize,
        h: usize,
        kx: &[f32],
        ky: &[f32],
        haze: Option<(f32, [f32; 3])>,
    ) -> Vec<u8> {
        let stride = w * 3;
        let (rx, ry) = (kx.len() / 2, ky.len() / 2);
        let plane: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
        let mut tmp = vec![0.0f32; plane.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (k, &wk) in kx.iter().enumerate() {
                        let sx = (x + k).saturating_sub(rx).min(w - 1);
                        acc += wk * plane[(y * w + sx) * 3 + c];
                    }
                    tmp[y * stride + x * 3 + c] = acc;
                }
            }
        }
        let mut out = vec![0u8; raw.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (k, &wk) in ky.iter().enumerate() {
                        let sy = (y + k).saturating_sub(ry).min(h - 1);
                        acc += wk * tmp[sy * stride + x * 3 + c];
                    }
                    if let Some((t, veil)) = haze {
                        acc = acc * t + veil[c];
                    }
                    out[y * stride + x * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
                }
            }
        }
        out
    }

    fn ref_taps(raw: &[u8], w: usize, h: usize, taps: &[(isize, isize, f32)]) -> Vec<u8> {
        let mut out = vec![0u8; raw.len()];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for &(dx, dy, wk) in taps {
                        let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        acc += wk * raw[(sy * w + sx) * 3 + c] as f32;
                    }
                    out[(y * w + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
                }
            }
        }
        out
    }

    fn ref_resample(raw: &[u8], w: usize, h: usize, w2: usize, h2: usize, xt: &Taps, yt: &Taps) -> Vec<u8> {
        let mut tmp = vec![0.0f32; h * w2 * 3];
        for y in 0..h {
            for x in 0..w2 {
                let (s, ws) = xt.get(x);
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &wj) in ws.iter().enumerate() {
                        acc += wj * raw[(y * w + s + j) * 3 + c] as f32;
                    }
                    tmp[(y * w2 + x) * 3 + c] = acc;
                }
            }
        }
        let mut out = vec![0u8; h2 * w2 * 3];
        for y in 0..h2 {
            let (s, ws) = yt.get(y);
            for x in 0..w2 {
                for c in 0..3 {
                    let mut acc = 0.0f32;
                    for (j, &wj) in ws.iter().enumerate() {
                        acc += wj * tmp[((s + j) * w2 + x) * 3 + c];
                    }
                    out[(y * w2 + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
                }
            }
        }
        out
    }

    #[test]
    fn register_tiled_convolutions_and_resample_match_reference_bit_exact() {
        let mut rng = Rng::seed(0xB17_E4AC);
        let k = gaussian_kernel(1.7);
        let taps = vec![(-3, -2, 0.08), (-1, -1, 0.14), (0, 0, 0.36), (2, 1, 0.25), (4, 2, 0.17)];
        let veil = [220.0 * 0.35, 225.0 * 0.35, 232.0 * 0.35];
        for (w, h) in [(1, 1), (2, 3), (7, 5), (8, 4), (9, 7), (15, 6), (16, 8), (17, 9), (25, 11), (65, 13)]
        {
            let raw: Vec<u8> = (0..w * h * 3).map(|_| rng.below(256) as u8).collect();
            assert_eq!(
                convolve_separable_u8(&raw, w, h, &k, &k),
                ref_separable(&raw, w, h, &k, &k, None),
                "separable {w}x{h}"
            );
            assert_eq!(
                convolve_separable_haze_u8(&raw, w, h, &k, &k, 0.65, veil),
                ref_separable(&raw, w, h, &k, &k, Some((0.65, veil))),
                "haze {w}x{h}"
            );
            assert_eq!(convolve_taps_u8(&raw, w, h, &taps), ref_taps(&raw, w, h, &taps), "taps {w}x{h}");
            let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
            let (xt, yt) = (area_taps(w, w2), area_taps(h, h2));
            assert_eq!(
                resample_u8_to_u8(&raw, w, h, w2, h2, &xt, &yt),
                ref_resample(&raw, w, h, w2, h2, &xt, &yt),
                "resample u8->u8 {w}x{h}->{w2}x{h2}"
            );
            let small = resample_u8_to_plane(&raw, w, h, w2, h2, &xt, &yt);
            let (bx, by) = (bilinear_taps(w2, w), bilinear_taps(h2, h));
            let up = resample_plane_to_u8(&small, w, h, &bx, &by);
            // Cross-check the two-stage plane resample against reference scalar formula on `small.data`.
            let mut ref_up = vec![0u8; raw.len()];
            let mut tmp = vec![0.0f32; h2 * w * 3];
            for y in 0..h2 {
                for x in 0..w {
                    let (s, ws) = bx.get(x);
                    for c in 0..3 {
                        let mut acc = 0.0f32;
                        for (j, &wj) in ws.iter().enumerate() {
                            acc += wj * small.data[(y * w2 + s + j) * 3 + c];
                        }
                        tmp[(y * w + x) * 3 + c] = acc;
                    }
                }
            }
            for y in 0..h {
                let (s, ws) = by.get(y);
                for x in 0..w {
                    for c in 0..3 {
                        let mut acc = 0.0f32;
                        for (j, &wj) in ws.iter().enumerate() {
                            acc += wj * tmp[((s + j) * w + x) * 3 + c];
                        }
                        ref_up[(y * w + x) * 3 + c] = (acc.clamp(0.0, 255.0) + 0.5) as u8;
                    }
                }
            }
            assert_eq!(up, ref_up, "resample plane->u8 {w}x{h}");
        }
    }
}
