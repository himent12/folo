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

/// Minimum rows per rayon task so fast per-pixel passes avoid per-row scheduling overhead while still
/// load-balancing evenly across high-core-count CPUs.
const MIN_PAR_ROWS: usize = 4;

/// Calls `f(y, row)` for every row of `buf` (rows of `row_len` values), in parallel for large images.
pub(crate) fn for_rows<T: Send>(buf: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    if row_len == 0 {
        return;
    }
    if buf.len() / 3 >= PAR_PIXELS {
        buf.par_chunks_mut(row_len).with_min_len(MIN_PAR_ROWS).enumerate().for_each(|(y, r)| f(y, r));
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
        buf.par_chunks_mut(row_len)
            .with_min_len(MIN_PAR_ROWS)
            .enumerate()
            .for_each_init(&init, |s, (y, r)| f(s, y, r));
    } else {
        let mut s = init();
        buf.chunks_mut(row_len).enumerate().for_each(|(y, r)| f(&mut s, y, r));
    }
}

/// Processes `buf` (`h` rows of `row_len` elements) in contiguous row bands of up to `band_rows` rows,
/// parallelised with rayon when `parallel` is true, passing `(y_start, band_slice)` and per-worker scratch.
fn for_row_bands_with<T: Send, S>(
    buf: &mut [T],
    row_len: usize,
    band_rows: usize,
    parallel: bool,
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, &mut [T]) + Sync + Send,
) {
    if row_len == 0 || buf.is_empty() {
        return;
    }
    let total_rows = (buf.len() / row_len).max(1);
    let band_rows = band_rows.clamp(1, total_rows);
    let band_elems = row_len * band_rows;
    if parallel {
        buf.par_chunks_mut(band_elems)
            .enumerate()
            .for_each_init(&init, |s, (b, slice)| f(s, b * band_rows, slice));
    } else {
        let mut s = init();
        f(&mut s, 0, buf);
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

/// Exact integer RGB channel sum over `chunk` (a multiple of 3 bytes), using `u32` inner block accumulators.
fn sum_rgb_chunk(chunk: &[u8]) -> [u64; 3] {
    let mut total = [0u64; 3];
    // 4096 pixels * 255 = 1,044,480 << u32::MAX, so inner blocks never overflow.
    for block in chunk.chunks(4096 * 3) {
        let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
        for px in block.as_chunks::<3>().0 {
            r += px[0] as u32;
            g += px[1] as u32;
            b += px[2] as u32;
        }
        total[0] += r as u64;
        total[1] += g as u64;
        total[2] += b as u64;
    }
    total
}

/// The per-channel mean of interleaved RGB bytes, rounded (exact integer sums). Black for an empty image.
pub(crate) fn mean_rgb(raw: &[u8]) -> [u8; 3] {
    let pixels = raw.len() / 3;
    if pixels == 0 {
        return [0; 3];
    }
    let active = &raw[..pixels * 3];
    let sum = if pixels >= PAR_PIXELS {
        active
            .par_chunks(16384 * 3)
            .map(sum_rgb_chunk)
            .reduce(|| [0u64; 3], |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    } else {
        sum_rgb_chunk(active)
    };
    let n = pixels as u64;
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

/// Horizontally convolves one 8-bit RGB source row (`stride = w * 3`) with `kx` into `out` (`stride` floats),
/// using `pad` (`(w + 2 * rx) * 3` floats) as clamped scratch, register-tiled in exact `k` order.
#[inline]
fn convolve_h_row_u8(src: &[u8], w: usize, kx: &[f32], rx: usize, pad: &mut [f32], out: &mut [f32]) {
    let stride = w * 3;
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
}

/// Vertically convolves one output row from `ring` (indexed by `(base, wk)` pairs) straight into `out_row`,
/// applying an optional fused haze affine `v * t + veil[c]`.
#[inline]
fn convolve_v_row_to_u8(
    ring: &[f32],
    rows: &[(usize, f32)],
    stride: usize,
    haze: Option<(f32, [f32; 3])>,
    out_row: &mut [u8],
) {
    match haze {
        None => {
            let mut x = 0;
            while x + CONV_TILE <= stride {
                let mut acc = [0.0f32; CONV_TILE];
                for &(base, wk) in rows {
                    let s = &ring[base + x..base + x + CONV_TILE];
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
                for &(base, wk) in rows {
                    let s = &ring[base + x..base + x + rem];
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
            let veil_tile: [f32; CONV_TILE] = std::array::from_fn(|i| veil[i % 3]);
            let mut x = 0;
            while x + CONV_TILE <= stride {
                let mut acc = [0.0f32; CONV_TILE];
                for &(base, wk) in rows {
                    let s = &ring[base + x..base + x + CONV_TILE];
                    for i in 0..CONV_TILE {
                        acc[i] += wk * s[i];
                    }
                }
                for i in 0..CONV_TILE {
                    out_row[x + i] = to_u8(acc[i] * t + veil_tile[i]);
                }
                x += CONV_TILE;
            }
            if x < stride {
                let rem = stride - x;
                let mut acc = [0.0f32; CONV_TILE];
                for &(base, wk) in rows {
                    let s = &ring[base + x..base + x + rem];
                    for i in 0..rem {
                        acc[i] += wk * s[i];
                    }
                }
                for i in 0..rem {
                    out_row[x + i] = to_u8(acc[i] * t + veil_tile[i]);
                }
            }
        }
    }
}

/// Separable 2D convolution (`kx` then `ky`, odd lengths, centred): uses a band-streamed L2-resident ring buffer
/// for compact kernels (`ky.len() <= 25`, e.g. `haze` and mild `defocus`) to avoid the full-frame intermediate
/// buffer, and falls back to two-pass row-parallel convolution for wide kernels where per-band halo recomputation
/// would otherwise dominate.
fn convolve_separable_impl(
    raw: &[u8],
    w: usize,
    h: usize,
    kx: &[f32],
    ky: &[f32],
    haze: Option<(f32, [f32; 3])>,
) -> Vec<u8> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    const { assert!(CONV_TILE.is_multiple_of(3)) };
    const RING_MAX_KLEN: usize = 25;
    let stride = w * 3;
    assert_eq!(raw.len(), h * stride);
    assert!(kx.len() % 2 == 1 && ky.len() % 2 == 1, "separable kernels must have odd length");
    let (rx, ry) = (kx.len() / 2, ky.len() / 2);
    let klen = ky.len();
    let mut out = vec![0u8; raw.len()];
    if klen <= RING_MAX_KLEN {
        let band_rows = (klen * 2).max(24);
        struct Scratch {
            pad: Vec<f32>,
            ring: Vec<f32>,
            rows: Vec<(usize, f32)>,
        }
        for_row_bands_with(
            &mut out,
            stride,
            band_rows,
            raw.len() / 3 >= PAR_PIXELS,
            || Scratch {
                pad: vec![0.0f32; (w + 2 * rx) * 3],
                ring: vec![0.0f32; klen * stride],
                rows: Vec::with_capacity(klen),
            },
            |sc, y_start, band| {
                let first_lo = y_start.saturating_sub(ry);
                let first_hi = (y_start + ry).min(h - 1);
                for sy in first_lo..=first_hi {
                    let slot = (sy % klen) * stride;
                    convolve_h_row_u8(
                        &raw[sy * stride..(sy + 1) * stride],
                        w,
                        kx,
                        rx,
                        &mut sc.pad,
                        &mut sc.ring[slot..slot + stride],
                    );
                }
                let mut last_loaded = first_hi;
                for (dy, out_row) in band.chunks_exact_mut(stride).enumerate() {
                    let y = y_start + dy;
                    let need_hi = (y + ry).min(h - 1);
                    while last_loaded < need_hi {
                        last_loaded += 1;
                        let slot = (last_loaded % klen) * stride;
                        convolve_h_row_u8(
                            &raw[last_loaded * stride..(last_loaded + 1) * stride],
                            w,
                            kx,
                            rx,
                            &mut sc.pad,
                            &mut sc.ring[slot..slot + stride],
                        );
                    }
                    sc.rows.clear();
                    for (k, &wk) in ky.iter().enumerate() {
                        let sy = (y + k).saturating_sub(ry).min(h - 1);
                        sc.rows.push(((sy % klen) * stride, wk));
                    }
                    convolve_v_row_to_u8(&sc.ring, &sc.rows, stride, haze, out_row);
                }
            },
        );
    } else {
        let mut tmp = vec![0.0f32; raw.len()];
        for_rows_with(
            &mut tmp,
            stride,
            || vec![0.0f32; (w + 2 * rx) * 3],
            |pad, y, dst| {
                convolve_h_row_u8(&raw[y * stride..(y + 1) * stride], w, kx, rx, pad, dst);
            },
        );
        for_rows_with(
            &mut out,
            stride,
            || Vec::<(usize, f32)>::with_capacity(klen),
            |rows, y, out_row| {
                rows.clear();
                for (k, &wk) in ky.iter().enumerate() {
                    let sy = (y + k).saturating_sub(ry).min(h - 1);
                    rows.push((sy * stride, wk));
                }
                convolve_v_row_to_u8(&tmp, rows, stride, haze, out_row);
            },
        );
    }
    out
}

/// 8-bit interleaved RGB `raw` (`w x h`) convolved with `kx` along rows and `ky` along columns (odd lengths,
/// centred), edges clamped, rounded directly back to 8-bit RGB.
pub(crate) fn convolve_separable_u8(raw: &[u8], w: usize, h: usize, kx: &[f32], ky: &[f32]) -> Vec<u8> {
    convolve_separable_impl(raw, w, h, kx, ky, None)
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
    convolve_separable_impl(raw, w, h, kx, ky, Some((t, veil)))
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
    if n_in == 0 || n_out == 0 {
        return Taps { start: Vec::new(), off: vec![0], w: Vec::new() };
    }
    let scale = n_in as f64 / n_out as f64;
    let mut t = Taps { start: Vec::with_capacity(n_out), off: vec![0], w: Vec::new() };
    for i in 0..n_out {
        let (a, b) = (i as f64 * scale, ((i + 1) as f64 * scale).min(n_in as f64));
        let j0 = (a.floor() as usize).min(n_in - 1);
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
    if n_in == 0 || n_out == 0 {
        return Taps { start: Vec::new(), off: vec![0], w: Vec::new() };
    }
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

/// Horizontally resamples one 8-bit RGB row into `out` (`w2 * 3` floats).
#[inline]
fn resample_h_u8_row(src: &[u8], xt: &Taps, out: &mut [f32]) {
    for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let (s, ws) = xt.get(x);
        if ws.len() == 2 {
            let (w0, w1) = (ws[0], ws[1]);
            let q = s * 3;
            *o = [
                (0.0f32 + w0 * src[q] as f32) + w1 * src[q + 3] as f32,
                (0.0f32 + w0 * src[q + 1] as f32) + w1 * src[q + 4] as f32,
                (0.0f32 + w0 * src[q + 2] as f32) + w1 * src[q + 5] as f32,
            ];
        } else {
            let mut acc = [0.0f32; 3];
            for (j, &wj) in ws.iter().enumerate() {
                let q = (s + j) * 3;
                acc[0] += wj * src[q] as f32;
                acc[1] += wj * src[q + 1] as f32;
                acc[2] += wj * src[q + 2] as f32;
            }
            *o = acc;
        }
    }
}

/// Horizontally resamples one `f32` RGB row into `out` (`w2 * 3` floats).
#[inline]
fn resample_h_f32_row(src: &[f32], xt: &Taps, out: &mut [f32]) {
    for (x, o) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let (s, ws) = xt.get(x);
        if ws.len() == 2 {
            let (w0, w1) = (ws[0], ws[1]);
            let q = s * 3;
            *o = [
                (0.0f32 + w0 * src[q]) + w1 * src[q + 3],
                (0.0f32 + w0 * src[q + 1]) + w1 * src[q + 4],
                (0.0f32 + w0 * src[q + 2]) + w1 * src[q + 5],
            ];
        } else {
            let mut acc = [0.0f32; 3];
            for (j, &wj) in ws.iter().enumerate() {
                let q = (s + j) * 3;
                acc[0] += wj * src[q];
                acc[1] += wj * src[q + 1];
                acc[2] += wj * src[q + 2];
            }
            *o = acc;
        }
    }
}

/// Source row range `[lo, hi)` required by output rows `y_start..y_end`.
#[inline]
fn band_src_range(yt: &Taps, y_start: usize, y_end: usize) -> (usize, usize) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for y in y_start..y_end {
        let (s, ws) = yt.get(y);
        lo = lo.min(s);
        hi = hi.max(s + ws.len());
    }
    if lo >= hi { (0, 0) } else { (lo, hi) }
}

/// Register-tiled vertical resampling of one row into `out` (`stride2` floats) from band buffer `tmp` starting at
/// source row `src_lo`.
#[inline]
fn resample_v_row_to_f32(tmp: &[f32], src_lo: usize, stride2: usize, s: usize, ws: &[f32], out: &mut [f32]) {
    let base_s = s - src_lo;
    let mut x = 0;
    while x + CONV_TILE <= stride2 {
        let mut acc = [0.0f32; CONV_TILE];
        for (j, &wj) in ws.iter().enumerate() {
            let row = &tmp[(base_s + j) * stride2 + x..(base_s + j) * stride2 + x + CONV_TILE];
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
            let row = &tmp[(base_s + j) * stride2 + x..(base_s + j) * stride2 + x + rem];
            for i in 0..rem {
                acc[i] += wj * row[i];
            }
        }
        out[x..stride2].copy_from_slice(&acc[..rem]);
    }
}

/// Register-tiled vertical resampling of one row into `out_row` (`stride2` bytes) from band buffer `tmp` starting
/// at source row `src_lo`.
#[inline]
fn resample_v_row_to_u8(
    tmp: &[f32],
    src_lo: usize,
    stride2: usize,
    s: usize,
    ws: &[f32],
    out_row: &mut [u8],
) {
    let base_s = s - src_lo;
    if ws.len() == 2 {
        let (w0, w1) = (ws[0], ws[1]);
        let r0 = &tmp[base_s * stride2..(base_s + 1) * stride2];
        let r1 = &tmp[(base_s + 1) * stride2..(base_s + 2) * stride2];
        for i in 0..stride2 {
            out_row[i] = to_u8((0.0f32 + w0 * r0[i]) + w1 * r1[i]);
        }
        return;
    }
    let mut x = 0;
    while x + CONV_TILE <= stride2 {
        let mut acc = [0.0f32; CONV_TILE];
        for (j, &wj) in ws.iter().enumerate() {
            let row = &tmp[(base_s + j) * stride2 + x..(base_s + j) * stride2 + x + CONV_TILE];
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
            let row = &tmp[(base_s + j) * stride2 + x..(base_s + j) * stride2 + x + rem];
            for i in 0..rem {
                acc[i] += wj * row[i];
            }
        }
        for i in 0..rem {
            out_row[x + i] = to_u8(acc[i]);
        }
    }
}

/// 8-bit interleaved RGB `raw` (`w x h`) resampled to a `w2 x h2` float [`Plane`] via band-streamed passes
/// (zero full-frame intermediate buffer).
pub(crate) fn resample_u8_to_plane(
    raw: &[u8],
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    xt: &Taps,
    yt: &Taps,
) -> Plane {
    let (stride, stride2) = (w * 3, w2 * 3);
    assert_eq!(raw.len(), h * stride);
    let mut data = vec![0.0f32; h2 * stride2];
    let parallel = w * h >= PAR_PIXELS || w2 * h2 >= PAR_PIXELS;
    let band_rows = (h2 / 32).clamp(4, 16);
    for_row_bands_with(&mut data, stride2, band_rows, parallel, Vec::<f32>::new, |tmp, y_start, band| {
        let band_h = band.len() / stride2;
        let (src_lo, src_hi) = band_src_range(yt, y_start, y_start + band_h);
        let need = (src_hi - src_lo) * stride2;
        if tmp.len() < need {
            tmp.resize(need, 0.0);
        }
        for sy in src_lo..src_hi {
            let dst = &mut tmp[(sy - src_lo) * stride2..(sy - src_lo + 1) * stride2];
            resample_h_u8_row(&raw[sy * stride..(sy + 1) * stride], xt, dst);
        }
        for (dy, out_row) in band.chunks_exact_mut(stride2).enumerate() {
            let (s, ws) = yt.get(y_start + dy);
            resample_v_row_to_f32(&tmp[..need], src_lo, stride2, s, ws, out_row);
        }
    });
    Plane { w: w2, h: h2, data }
}

/// Float [`Plane`] `p` resampled to `w2 x h2` and rounded directly to 8-bit interleaved RGB via band-streamed
/// passes (zero full-frame intermediate buffer).
pub(crate) fn resample_plane_to_u8(p: &Plane, w2: usize, h2: usize, xt: &Taps, yt: &Taps) -> Vec<u8> {
    let (stride, stride2) = (p.w * 3, w2 * 3);
    assert_eq!(p.data.len(), p.h * stride);
    let mut out = vec![0u8; h2 * stride2];
    let parallel = p.w * p.h >= PAR_PIXELS || w2 * h2 >= PAR_PIXELS;
    let band_rows = (h2 / 32).clamp(4, 16);
    for_row_bands_with(&mut out, stride2, band_rows, parallel, Vec::<f32>::new, |tmp, y_start, band| {
        let band_h = band.len() / stride2;
        let (src_lo, src_hi) = band_src_range(yt, y_start, y_start + band_h);
        let need = (src_hi - src_lo) * stride2;
        if tmp.len() < need {
            tmp.resize(need, 0.0);
        }
        for sy in src_lo..src_hi {
            let dst = &mut tmp[(sy - src_lo) * stride2..(sy - src_lo + 1) * stride2];
            resample_h_f32_row(&p.data[sy * stride..(sy + 1) * stride], xt, dst);
        }
        for (dy, out_row) in band.chunks_exact_mut(stride2).enumerate() {
            let (s, ws) = yt.get(y_start + dy);
            resample_v_row_to_u8(&tmp[..need], src_lo, stride2, s, ws, out_row);
        }
    });
    out
}

/// 8-bit interleaved RGB `raw` (`w x h`) resampled to `w2 x h2` and rounded directly to 8-bit interleaved RGB via
/// band-streamed passes (zero full-frame intermediate buffer).
pub(crate) fn resample_u8_to_u8(
    raw: &[u8],
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    xt: &Taps,
    yt: &Taps,
) -> Vec<u8> {
    let (stride, stride2) = (w * 3, w2 * 3);
    assert_eq!(raw.len(), h * stride);
    let mut out = vec![0u8; h2 * stride2];
    let parallel = w * h >= PAR_PIXELS || w2 * h2 >= PAR_PIXELS;
    let band_rows = (h2 / 32).clamp(4, 16);
    for_row_bands_with(&mut out, stride2, band_rows, parallel, Vec::<f32>::new, |tmp, y_start, band| {
        let band_h = band.len() / stride2;
        let (src_lo, src_hi) = band_src_range(yt, y_start, y_start + band_h);
        let need = (src_hi - src_lo) * stride2;
        if tmp.len() < need {
            tmp.resize(need, 0.0);
        }
        for sy in src_lo..src_hi {
            let dst = &mut tmp[(sy - src_lo) * stride2..(sy - src_lo + 1) * stride2];
            resample_h_u8_row(&raw[sy * stride..(sy + 1) * stride], xt, dst);
        }
        for (dy, out_row) in band.chunks_exact_mut(stride2).enumerate() {
            let (s, ws) = yt.get(y_start + dy);
            resample_v_row_to_u8(&tmp[..need], src_lo, stride2, s, ws, out_row);
        }
    });
    out
}

#[cfg(test)]
mod tests;
