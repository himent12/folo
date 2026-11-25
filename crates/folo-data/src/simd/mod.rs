//! Bit-exact AVX2 SIMD kernels and scalar fallbacks for `folo-data`.
//!
//! Every kernel here matches the scalar reference bit for bit (all `u8` bytes and all IEEE-754 `f32` bits), verified
//! by unit tests and `tests/golden.rs`. On x86_64 CPUs with AVX2, runtime feature detection (`OnceLock`) dispatches to
//! the AVX2 path; everywhere else (ARM, WASM, older x86) the scalar fallback runs.

pub(crate) mod bilinear;
pub(crate) mod box_filter;
pub(crate) mod hsv_fused;
pub(crate) mod planar;
#[cfg(test)]
mod tests;

/// A single image's planar `[3, H, W]` destination buffer, either `u8` or `f32` (`v as f32 / 255.0`).
pub(crate) enum PlanarDst<'a> {
    U8 { r: &'a mut [u8], g: &'a mut [u8], b: &'a mut [u8] },
    F32Div255 { r: &'a mut [f32], g: &'a mut [f32], b: &'a mut [f32] },
}

impl<'a> PlanarDst<'a> {
    /// Splits a contiguous `[3 * plane]` `u8` buffer into three plane slices.
    #[inline]
    pub(crate) fn from_u8(buf: &'a mut [u8], plane: usize) -> Self {
        assert_eq!(buf.len(), plane.checked_mul(3).expect("plane size overflow"));
        let (r, rest) = buf.split_at_mut(plane);
        let (g, b) = rest.split_at_mut(plane);
        PlanarDst::U8 { r, g, b }
    }

    /// Splits a contiguous `[3 * plane]` `f32` buffer into three plane slices (`v as f32 / 255.0`).
    #[inline]
    pub(crate) fn from_f32_div255(buf: &'a mut [f32], plane: usize) -> Self {
        assert_eq!(buf.len(), plane.checked_mul(3).expect("plane size overflow"));
        let (r, rest) = buf.split_at_mut(plane);
        let (g, b) = rest.split_at_mut(plane);
        PlanarDst::F32Div255 { r, g, b }
    }

    /// Borrows a sub-range of rows `[y0 .. y0 + rows]` (`width` pixels per row) across all three planes.
    #[inline]
    pub(crate) fn row_slices(&mut self, y: usize, width: usize) -> PlanarRow<'_> {
        let off = y * width;
        match self {
            PlanarDst::U8 { r, g, b } => PlanarRow::U8 {
                r: &mut r[off..off + width],
                g: &mut g[off..off + width],
                b: &mut b[off..off + width],
            },
            PlanarDst::F32Div255 { r, g, b } => PlanarRow::F32Div255 {
                r: &mut r[off..off + width],
                g: &mut g[off..off + width],
                b: &mut b[off..off + width],
            },
        }
    }
}

/// One destination row (`width` pixels) across the three planar channels.
pub(crate) enum PlanarRow<'a> {
    U8 { r: &'a mut [u8], g: &'a mut [u8], b: &'a mut [u8] },
    F32Div255 { r: &'a mut [f32], g: &'a mut [f32], b: &'a mut [f32] },
}

impl PlanarRow<'_> {
    /// Fills columns `[lo .. hi)` with one RGB colour (`u8` or `v as f32 / 255.0`).
    #[inline]
    pub(crate) fn fill_range(&mut self, lo: usize, hi: usize, rgb: [u8; 3]) {
        if lo >= hi {
            return;
        }
        match self {
            PlanarRow::U8 { r, g, b } => {
                r[lo..hi].fill(rgb[0]);
                g[lo..hi].fill(rgb[1]);
                b[lo..hi].fill(rgb[2]);
            }
            PlanarRow::F32Div255 { r, g, b } => {
                r[lo..hi].fill(rgb[0] as f32 / 255.0);
                g[lo..hi].fill(rgb[1] as f32 / 255.0);
                b[lo..hi].fill(rgb[2] as f32 / 255.0);
            }
        }
    }
}

/// Verifies `img.data` holds at least `img.width * img.height * 3` bytes (guards raw-pointer kernels against
/// malformed public [`crate::image::Image`] structs).
#[inline]
pub(crate) fn assert_image_data(img: &crate::image::Image) {
    let need = (img.width as usize)
        .checked_mul(img.height as usize)
        .and_then(|n| n.checked_mul(3))
        .expect("image dimensions overflow");
    assert!(
        img.data.len() >= need,
        "image buffer has {} bytes, expected at least {}x{}x3 ({need})",
        img.data.len(),
        img.width,
        img.height
    );
}

impl PlanarDst<'_> {
    /// Number of rows held by this planar band for a row width of `width` pixels.
    #[inline]
    pub(crate) fn rows(&self, width: usize) -> usize {
        match self {
            PlanarDst::U8 { r, .. } => r.len() / width,
            PlanarDst::F32Div255 { r, .. } => r.len() / width,
        }
    }
}

/// Splits a [`PlanarDst`] into row bands (parallel with rayon when worthwhile and not in `par::serial`).
pub(crate) fn for_planar_bands<F>(dst: &mut PlanarDst<'_>, width: usize, height: usize, f: F)
where
    F: Fn(usize, &mut PlanarDst<'_>) + Sync,
{
    if width == 0 || height == 0 {
        return;
    }
    if crate::par::is_serial() || width * height * 3 < 64 * 1024 {
        f(0, dst);
        return;
    }
    use rayon::prelude::*;
    match dst {
        PlanarDst::U8 { r, g, b } => {
            let rows_per_band = (64 * 1024 / (3 * width)).max(1);
            let n = rows_per_band * width;
            r.par_chunks_mut(n)
                .zip(g.par_chunks_mut(n))
                .zip(b.par_chunks_mut(n))
                .enumerate()
                .for_each(|(t, ((r, g), b))| f(t * rows_per_band, &mut PlanarDst::U8 { r, g, b }));
        }
        PlanarDst::F32Div255 { r, g, b } => {
            let rows_per_band = (64 * 1024 / (12 * width)).max(1);
            let n = rows_per_band * width;
            r.par_chunks_mut(n)
                .zip(g.par_chunks_mut(n))
                .zip(b.par_chunks_mut(n))
                .enumerate()
                .for_each(|(t, ((r, g), b))| f(t * rows_per_band, &mut PlanarDst::F32Div255 { r, g, b }));
        }
    }
}

/// Whether this CPU supports AVX2 (cached once per process).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}

/// Whether this CPU supports both AVX2 and FMA (cached once per process).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    use std::sync::OnceLock;
    static AVX2_FMA: OnceLock<bool> = OnceLock::new();
    *AVX2_FMA.get_or_init(|| std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"))
}

/// Exact two-term FMA reciprocal of `255.0` (`INV255_HI + INV255_LO ≈ 1/255` to ~48 bits, both positive).
///
/// Since `8421504 * 255 = 2^31 - 128`, `INV255_HI = 8421504 / 2^31` and the exact residual is
/// `128 / (255 * 2^31) = 1 / (255 * 2^24)`. For every `x in 0..=255`,
/// `fma(x as f32, INV255_HI, (x as f32) * INV255_LO)` matches `x as f32 / 255.0` bit-for-bit in IEEE-754 `f32`.
#[cfg(target_arch = "x86_64")]
pub(crate) const INV255_HI: f32 = f32::from_bits(0x3b80_8080);
#[cfg(target_arch = "x86_64")]
pub(crate) const INV255_LO: f32 = (1.0f64 / (255.0f64 * 16_777_216.0f64)) as f32;
