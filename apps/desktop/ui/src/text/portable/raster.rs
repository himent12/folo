// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/linux/text_system.rs, crates/gpui/src/platform/blade/blade_renderer.rs (gamma ratios)
//! Glyph rasterization with swash (through cosmic-text), as GPUI's Linux backend does it, and the glyph
//! correction GPUI's Linux renderer applies to the masks (`platform/blade/shaders.wgsl`).

use super::fonts::State;
use crate::{Bounds, DevicePixels, GlyphBlend, RenderGlyphParams, SUBPIXEL_VARIANTS_X, Size, point, size};
use anyhow::{Context as _, Result, bail};
use cosmic_text::{CacheKey, CacheKeyFlags, SwashContent, SwashImage};
use std::collections::HashMap;

/// Rendered images by key. `glyph_raster_bounds` and `rasterize_glyph` come in pairs for the same glyph,
/// so the image rendered for the first is reused by the second; the renderer keeps its own mask cache, so
/// this one is small and simply emptied when full.
#[derive(Default)]
pub(super) struct ImageCache(HashMap<CacheKey, Option<SwashImage>>);

const IMAGE_CACHE_ENTRIES: usize = 256;

/// The correction GPUI's blade renderer (Linux) applies to glyph masks: its defaults, gamma 1.8 and
/// grayscale enhanced contrast 1.0. GPUI reads `ZED_FONTS_GAMMA` / `ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST`;
/// this backend does not, so masks blend the same everywhere.
pub fn portable_glyph_blend() -> GlyphBlend {
    GlyphBlend::new(gamma_ratios(1.8), 1.0)
}

/// GPUI's `RenderingParameters::get_gamma_ratios` (from Windows Terminal's DirectWrite renderer).
fn gamma_ratios(gamma: f32) -> [f32; 4] {
    const GAMMA_INCORRECT_TARGET_RATIOS: [[f32; 4]; 13] = [
        [0.0000 / 4.0, 0.0000 / 4.0, 0.0000 / 4.0, 0.0000 / 4.0], // gamma = 1.0
        [0.0166 / 4.0, -0.0807 / 4.0, 0.2227 / 4.0, -0.0751 / 4.0], // gamma = 1.1
        [0.0350 / 4.0, -0.1760 / 4.0, 0.4325 / 4.0, -0.1370 / 4.0], // gamma = 1.2
        [0.0543 / 4.0, -0.2821 / 4.0, 0.6302 / 4.0, -0.1876 / 4.0], // gamma = 1.3
        [0.0739 / 4.0, -0.3963 / 4.0, 0.8167 / 4.0, -0.2287 / 4.0], // gamma = 1.4
        [0.0933 / 4.0, -0.5161 / 4.0, 0.9926 / 4.0, -0.2616 / 4.0], // gamma = 1.5
        [0.1121 / 4.0, -0.6395 / 4.0, 1.1588 / 4.0, -0.2877 / 4.0], // gamma = 1.6
        [0.1300 / 4.0, -0.7649 / 4.0, 1.3159 / 4.0, -0.3080 / 4.0], // gamma = 1.7
        [0.1469 / 4.0, -0.8911 / 4.0, 1.4644 / 4.0, -0.3234 / 4.0], // gamma = 1.8
        [0.1627 / 4.0, -1.0170 / 4.0, 1.6051 / 4.0, -0.3347 / 4.0], // gamma = 1.9
        [0.1773 / 4.0, -1.1420 / 4.0, 1.7385 / 4.0, -0.3426 / 4.0], // gamma = 2.0
        [0.1908 / 4.0, -1.2652 / 4.0, 1.8650 / 4.0, -0.3476 / 4.0], // gamma = 2.1
        [0.2031 / 4.0, -1.3864 / 4.0, 1.9851 / 4.0, -0.3501 / 4.0], // gamma = 2.2
    ];
    const NORM13: f32 = ((0x10000 as f64) / (255.0 * 255.0) * 4.0) as f32;
    const NORM24: f32 = ((0x100 as f64) / (255.0) * 4.0) as f32;

    let index = ((gamma * 10.0).round() as usize).clamp(10, 22) - 10;
    let ratios = GAMMA_INCORRECT_TARGET_RATIOS[index];
    [ratios[0] * NORM13, ratios[1] * NORM24, ratios[2] * NORM13, ratios[3] * NORM24]
}

impl State {
    /// The swash cache key for `params` with its subpixel shift, and the whole device pixels (down) the y
    /// shift leaves out of the key, which the image's placement takes.
    fn cache_key(&self, params: &RenderGlyphParams) -> (CacheKey, i32) {
        let font = &self.loaded_font(params.font_id).font;
        // GPUI divides the device-pixel variant by the scale factor here (so at scale 2 the four x variants
        // bin to three shifts); kept for parity with GPUI on Linux.
        let shift_x = params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor;
        // The y variant moves the glyph down by its share of a device pixel (as GPUI's macOS backend shifts
        // it; on Windows and Linux there is one vertical variant and the shift is zero). Swash draws y up,
        // so its offset is the negated shift: a bin above and one whole pixel down.
        let shift_y = params.subpixel_variant.y as f32 / self.subpixel_variants_y.max(1) as f32;
        let (key, _, y) = CacheKey::new(
            font.id(),
            params.glyph_id.0 as u16,
            (params.font_size * params.scale_factor).into(),
            (shift_x, -shift_y),
            CacheKeyFlags::empty(),
        );
        (key, -y)
    }

    fn image(&mut self, params: &RenderGlyphParams) -> Result<&SwashImage> {
        let key = self.cache_key(params).0;
        if self.images.0.len() >= IMAGE_CACHE_ENTRIES && !self.images.0.contains_key(&key) {
            self.images.0.clear();
        }
        let image = self
            .images
            .0
            .entry(key)
            .or_insert_with(|| self.swash.get_image_uncached(&mut self.font_system, key));
        image.as_ref().with_context(|| format!("no image for {params:?}"))
    }

    pub fn raster_bounds(&mut self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let dy = self.cache_key(params).1;
        let placement = self.image(params)?.placement;
        Ok(Bounds {
            origin: point(DevicePixels(placement.left), DevicePixels(dy - placement.top)),
            size: size(DevicePixels(placement.width as i32), DevicePixels(placement.height as i32)),
        })
    }

    pub fn rasterize_glyph(
        &mut self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 <= 0 || glyph_bounds.size.height.0 <= 0 {
            bail!("glyph bounds are empty");
        }
        let image = self.image(params)?;
        let pixels = image.placement.width as usize * image.placement.height as usize;
        let data = match (image.content, params.is_emoji) {
            (SwashContent::Color, true) => rgba_to_bgra_premultiplied(&image.data),
            // A colour glyph asked for as a mask (a colour font cosmic-text fell back to that is not known
            // as an emoji font): its coverage.
            (SwashContent::Color, false) => image.data.as_chunks::<4>().0.iter().map(|p| p[3]).collect(),
            // An outline glyph asked for as an emoji: white, premultiplied, so the sprite is well formed.
            (_, true) => image.data.iter().take(pixels).flat_map(|&a| [a, a, a, a]).collect(),
            (_, false) => image.data.clone(),
        };
        Ok((glyph_bounds.size, data))
    }
}

/// Swash's colour images in BGRA, premultiplied. Colour outlines come out of swash premultiplied
/// already (it composes layers onto transparent black); colour bitmaps (PNG strikes) are straight
/// alpha, which shows as a channel above its alpha, and get premultiplied here.
fn rgba_to_bgra_premultiplied(rgba: &[u8]) -> Vec<u8> {
    let pixels = rgba.as_chunks::<4>().0;
    let straight = pixels.iter().any(|p| p[0] > p[3] || p[1] > p[3] || p[2] > p[3]);
    let mut out = Vec::with_capacity(rgba.len());
    for &[r, g, b, a] in pixels {
        let mul = |c: u8| if straight { ((c as u32 * a as u32 + 127) / 255) as u8 } else { c };
        out.extend_from_slice(&[mul(b), mul(g), mul(r), a]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_ratios_match_gpui_default() {
        let blend = portable_glyph_blend();
        assert!(blend.apply);
        assert_eq!(blend.grayscale_enhanced_contrast, 1.0);
        let norm13 = (0x10000 as f64 / (255.0 * 255.0) * 4.0) as f32;
        assert!((blend.gamma_ratios[0] - 0.1469 / 4.0 * norm13).abs() < 1e-6);
        assert_eq!(gamma_ratios(1.0), [0.0; 4]);
        assert_eq!(gamma_ratios(5.0), gamma_ratios(2.2));
    }

    /// The coverage-weighted mean of the mask's pixel-centre rows, in device pixels from the baseline (down).
    fn centroid_y(state: &mut State, params: &RenderGlyphParams) -> f32 {
        let bounds = state.raster_bounds(params).unwrap();
        let (_, mask) = state.rasterize_glyph(params, bounds).unwrap();
        let w = bounds.size.width.0 as usize;
        let (mut sum, mut weight) = (0., 0.);
        for (i, &a) in mask.iter().enumerate() {
            sum += (bounds.origin.y.0 as f32 + (i / w) as f32 + 0.5) * a as f32;
            weight += a as f32;
        }
        sum / weight
    }

    #[test]
    fn vertical_variants_move_the_glyph_down_by_their_share_of_a_pixel() {
        let geist: &'static [u8] = include_bytes!("../../../../../../design/fonts/Geist-Regular.ttf");
        let mut state = State::new();
        state.add_fonts(vec![geist.into()]);
        let font_id = state.font_id(&crate::font("Geist")).unwrap();
        let glyph_id = state.glyph_for_char(font_id, 'E').unwrap();
        let params = |y: u8, scale: f32| RenderGlyphParams {
            font_id,
            glyph_id,
            font_size: crate::px(13.),
            subpixel_variant: crate::Point { x: 0, y },
            scale_factor: scale,
            is_emoji: false,
        };
        // macOS has four vertical variants; the count is set here so this runs on every platform.
        state.subpixel_variants_y = 4;
        for scale in [1., 2.] {
            let base = centroid_y(&mut state, &params(0, scale));
            for y in 1..4u8 {
                let moved = centroid_y(&mut state, &params(y, scale)) - base;
                let want = y as f32 / 4.;
                assert!((moved - want).abs() < 0.03, "scale {scale} variant {y}: moved {moved}, want {want}");
            }
        }
        // One vertical variant (Windows, Linux): variant 0 is the unshifted glyph, as before.
        let four = state.raster_bounds(&params(0, 1.5)).unwrap();
        state.subpixel_variants_y = 1;
        assert_eq!(state.raster_bounds(&params(0, 1.5)).unwrap(), four);
    }

    #[test]
    fn colour_images_become_bgra_premultiplied() {
        // Already premultiplied: only swapped.
        assert_eq!(rgba_to_bgra_premultiplied(&[10, 20, 30, 40]), vec![30, 20, 10, 40]);
        // Straight alpha: premultiplied too.
        assert_eq!(
            rgba_to_bgra_premultiplied(&[255, 0, 128, 128, 0, 0, 0, 0]),
            vec![64, 0, 128, 128, 0, 0, 0, 0]
        );
    }
}
