//! Bit-exact AVX2 SIMD kernels and scalar fallbacks for `folo-data`.
//!
//! Every kernel here matches the scalar reference bit for bit (all `u8` bytes and all IEEE-754 `f32` bits), verified
//! by unit tests and `tests/golden.rs`. On x86_64 CPUs with AVX2, runtime feature detection (`OnceLock`) dispatches to
//! the AVX2 path; everywhere else (ARM, WASM, older x86) the scalar fallback runs.

pub(crate) mod bilinear;
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
        let (r, rest) = buf.split_at_mut(plane);
        let (g, b) = rest.split_at_mut(plane);
        PlanarDst::U8 { r, g, b }
    }

    /// Splits a contiguous `[3 * plane]` `f32` buffer into three plane slices (`v as f32 / 255.0`).
    #[inline]
    pub(crate) fn from_f32_div255(buf: &'a mut [f32], plane: usize) -> Self {
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

/// Whether this CPU supports AVX2 (cached once per process).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static AVX2: OnceLock<bool> = OnceLock::new();
    *AVX2.get_or_init(|| std::is_x86_feature_detected!("avx2"))
}
