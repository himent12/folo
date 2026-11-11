// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/direct_write.rs (glyph run analysis,
// raster bounds, rasterize_monochrome, rasterize_color), crates/gpui/src/platform/windows/
// color_text_raster.hlsl (the colour layer composition, here on the CPU)
//! Glyph masks: `CreateGlyphRunAnalysis` with GPUI's rendering, measuring, grid-fit and antialias modes
//! and subpixel offsets, `DWRITE_TEXTURE_ALIASED_1x1` alpha, and colour (COLR) glyphs composed layer by
//! layer into BGRA premultiplied pixels exactly as GPUI's emoji shader writes its 8-bit target.

use super::DirectWriteState;
use super::blend::apply_contrast_and_gamma_correction;
use crate::{
    Bounds, DevicePixels, GlyphBlend, RenderGlyphParams, SUBPIXEL_VARIANTS_X, SUBPIXEL_VARIANTS_Y, Size,
    point, size,
};
use anyhow::{Result, bail};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::DirectWrite::*;
use windows::core::BOOL;

/// A `windows_numerics::Vector2` (the crate is not a direct dependency; it is `#[repr(C)] { X, Y }`).
pub(super) fn vector2<T: Copy>(x: f32, y: f32) -> T {
    assert_eq!(std::mem::size_of::<T>(), 8, "Vector2 is two f32");
    // SAFETY: `T` is `windows_numerics::Vector2`, `#[repr(C)]` of two `f32`, the size checked above.
    unsafe { std::mem::transmute_copy(&[x, y]) }
}

fn scale_transform(scale_factor: f32) -> DWRITE_MATRIX {
    DWRITE_MATRIX { m11: scale_factor, m12: 0.0, m21: 0.0, m22: scale_factor, dx: 0.0, dy: 0.0 }
}

impl DirectWriteState {
    /// GPUI's `create_glyph_run_analysis`: the face's recommended rendering mode at 96 dpi with the
    /// scale as transform (outline becomes natural symmetric), natural measuring, grayscale antialiasing,
    /// and the subpixel variant as a baseline offset in logical pixels.
    fn create_glyph_run_analysis(&self, params: &RenderGlyphParams) -> Result<IDWriteGlyphRunAnalysis> {
        let font = &self.fonts[params.font_id.0];
        let glyph_id = [params.glyph_id.0 as u16];
        let advance = [0.0f32];
        let offset = [DWRITE_GLYPH_OFFSET::default()];
        let glyph_run = DWRITE_GLYPH_RUN {
            // SAFETY: a borrowed copy of the face pointer, never released (`ManuallyDrop`).
            fontFace: unsafe { std::mem::transmute_copy(&font.font_face) },
            fontEmSize: params.font_size.0,
            glyphCount: 1,
            glyphIndices: glyph_id.as_ptr(),
            glyphAdvances: advance.as_ptr(),
            glyphOffsets: offset.as_ptr(),
            isSideways: BOOL(0),
            bidiLevel: 0,
        };
        let transform = scale_transform(params.scale_factor);
        let baseline_origin_x =
            params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor;
        let baseline_origin_y =
            params.subpixel_variant.y as f32 / SUBPIXEL_VARIANTS_Y as f32 / params.scale_factor;

        let mut rendering_mode = DWRITE_RENDERING_MODE1::default();
        let mut grid_fit_mode = DWRITE_GRID_FIT_MODE::default();
        // SAFETY: out pointers are valid; the run's arrays live on this stack frame.
        unsafe {
            font.font_face.GetRecommendedRenderingMode(
                params.font_size.0,
                // 96: the scale is in the transform.
                96.0,
                96.0,
                Some(&transform),
                false,
                DWRITE_OUTLINE_THRESHOLD_ANTIALIASED,
                DWRITE_MEASURING_MODE_NATURAL,
                None::<&IDWriteRenderingParams>,
                &mut rendering_mode,
                &mut grid_fit_mode,
            )?;
        }
        let rendering_mode = match rendering_mode {
            DWRITE_RENDERING_MODE1_OUTLINE => DWRITE_RENDERING_MODE1_NATURAL_SYMMETRIC,
            m => m,
        };

        // SAFETY: as above.
        let analysis = unsafe {
            self.factory.CreateGlyphRunAnalysis(
                &glyph_run,
                Some(&transform),
                rendering_mode,
                DWRITE_MEASURING_MODE_NATURAL,
                grid_fit_mode,
                DWRITE_TEXT_ANTIALIAS_MODE_GRAYSCALE,
                baseline_origin_x,
                baseline_origin_y,
            )?
        };
        Ok(analysis)
    }

    /// The glyph's alpha texture bounds, relative to its floored device origin (empty: zero bounds).
    pub(super) fn raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let analysis = self.create_glyph_run_analysis(params)?;
        // SAFETY: a live analysis.
        let bounds = unsafe { analysis.GetAlphaTextureBounds(DWRITE_TEXTURE_ALIASED_1x1)? };
        if bounds.right < bounds.left {
            Ok(Bounds::default())
        } else {
            Ok(Bounds {
                origin: point(bounds.left.into(), bounds.top.into()),
                size: size((bounds.right - bounds.left).into(), (bounds.bottom - bounds.top).into()),
            })
        }
    }

    /// An alpha mask, or for an emoji its colour layers composed to BGRA premultiplied (falling back to
    /// the monochrome mask as black, as GPUI).
    pub(super) fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
        blend: &GlyphBlend,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 <= 0 || glyph_bounds.size.height.0 <= 0 {
            bail!("glyph bounds are empty");
        }
        let bitmap_data = if params.is_emoji {
            match self.rasterize_color(params, glyph_bounds, blend) {
                Ok(color) => color,
                Err(_) => self
                    .rasterize_monochrome(params, glyph_bounds)?
                    .into_iter()
                    .flat_map(|pixel| [0, 0, 0, pixel])
                    .collect(),
            }
        } else {
            self.rasterize_monochrome(params, glyph_bounds)?
        };
        Ok((glyph_bounds.size, bitmap_data))
    }

    fn rasterize_monochrome(
        &self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<Vec<u8>> {
        let mut bitmap_data =
            vec![0u8; glyph_bounds.size.width.0 as usize * glyph_bounds.size.height.0 as usize];
        let analysis = self.create_glyph_run_analysis(params)?;
        let rect = RECT {
            left: glyph_bounds.origin.x.0,
            top: glyph_bounds.origin.y.0,
            right: glyph_bounds.size.width.0 + glyph_bounds.origin.x.0,
            bottom: glyph_bounds.size.height.0 + glyph_bounds.origin.y.0,
        };
        // SAFETY: the buffer holds exactly the rect's pixels, one byte each (aliased 1x1).
        unsafe { analysis.CreateAlphaTexture(DWRITE_TEXTURE_ALIASED_1x1, &rect, &mut bitmap_data)? };
        Ok(bitmap_data)
    }

    /// GPUI's `rasterize_color` without the GPU: each COLR layer's grayscale coverage, contrast and
    /// gamma corrected for its colour, is drawn premultiplied with `ONE, INV_SRC_ALPHA` blending into an
    /// 8-bit BGRA target the size of the glyph bounds, layer after layer.
    fn rasterize_color(
        &self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
        blend: &GlyphBlend,
    ) -> Result<Vec<u8>> {
        let width = glyph_bounds.size.width.0 as usize;
        let height = glyph_bounds.size.height.0 as usize;
        let subpixel_shift = params.subpixel_variant.map(|v| v as f32 / SUBPIXEL_VARIANTS_X as f32);
        let baseline_origin_x = subpixel_shift.x / params.scale_factor;
        let baseline_origin_y = subpixel_shift.y / params.scale_factor;
        let transform = scale_transform(params.scale_factor);

        let font = &self.fonts[params.font_id.0];
        let glyph_id = [params.glyph_id.0 as u16];
        let advance = [glyph_bounds.size.width.0 as f32];
        // The glyph moved so its bounds start at the target's (0, 0).
        let offset = [DWRITE_GLYPH_OFFSET {
            advanceOffset: -glyph_bounds.origin.x.0 as f32 / params.scale_factor,
            ascenderOffset: glyph_bounds.origin.y.0 as f32 / params.scale_factor,
        }];
        let glyph_run = DWRITE_GLYPH_RUN {
            // SAFETY: a borrowed copy of the face pointer, never released (`ManuallyDrop`).
            fontFace: unsafe { std::mem::transmute_copy(&font.font_face) },
            fontEmSize: params.font_size.0,
            glyphCount: 1,
            glyphIndices: glyph_id.as_ptr(),
            glyphAdvances: advance.as_ptr(),
            glyphOffsets: offset.as_ptr(),
            isSideways: BOOL(0),
            bidiLevel: 0,
        };

        // SAFETY: the run's arrays live on this stack frame for the call.
        let color_enumerator = unsafe {
            self.factory.TranslateColorGlyphRun(
                vector2(baseline_origin_x, baseline_origin_y),
                &glyph_run,
                None,
                DWRITE_GLYPH_IMAGE_FORMATS_COLR,
                DWRITE_MEASURING_MODE_NATURAL,
                Some(&transform),
                0,
            )?
        };

        let mut target = vec![0u8; width * height * 4];
        loop {
            // SAFETY: the enumerator's current run stays valid until `MoveNext`.
            let color_run = unsafe { &*color_enumerator.GetCurrentRun()? };
            let image_format = color_run.glyphImageFormat & !DWRITE_GLYPH_IMAGE_FORMATS_TRUETYPE;
            if image_format == DWRITE_GLYPH_IMAGE_FORMATS_COLR {
                // SAFETY: the layer's glyph run is owned by the enumerator.
                let color_analysis = unsafe {
                    self.factory.CreateGlyphRunAnalysis(
                        &color_run.Base.glyphRun as *const _,
                        Some(&transform),
                        DWRITE_RENDERING_MODE1_NATURAL_SYMMETRIC,
                        DWRITE_MEASURING_MODE_NATURAL,
                        DWRITE_GRID_FIT_MODE_DEFAULT,
                        DWRITE_TEXT_ANTIALIAS_MODE_GRAYSCALE,
                        baseline_origin_x,
                        baseline_origin_y,
                    )?
                };
                // SAFETY: a live analysis.
                let color_bounds =
                    unsafe { color_analysis.GetAlphaTextureBounds(DWRITE_TEXTURE_ALIASED_1x1)? };
                let layer_width = color_bounds.right - color_bounds.left;
                let layer_height = color_bounds.bottom - color_bounds.top;
                if layer_width > 0 && layer_height > 0 {
                    let mut alpha = vec![0u8; (layer_width * layer_height) as usize];
                    // SAFETY: the buffer holds exactly the bounds' pixels.
                    unsafe {
                        color_analysis.CreateAlphaTexture(
                            DWRITE_TEXTURE_ALIASED_1x1,
                            &color_bounds,
                            &mut alpha,
                        )?
                    };
                    let c = color_run.Base.runColor;
                    let layer = Layer {
                        left: color_bounds.left,
                        top: color_bounds.top,
                        width: layer_width,
                        height: layer_height,
                        alpha: &alpha,
                        color: [c.r, c.g, c.b, c.a],
                    };
                    compose_layer(&mut target, width, height, &layer, blend);
                }
            }
            // SAFETY: a live enumerator.
            let has_next = unsafe { color_enumerator.MoveNext() }.map(|e| e.as_bool()).unwrap_or(false);
            if !has_next {
                break;
            }
        }
        Ok(target)
    }
}

/// One colour layer: its coverage at its bounds in the target, and its straight RGBA colour.
pub(super) struct Layer<'a> {
    pub(super) left: i32,
    pub(super) top: i32,
    pub(super) width: i32,
    pub(super) height: i32,
    pub(super) alpha: &'a [u8],
    pub(super) color: [f32; 4],
}

/// Draws `layer` into the BGRA target as GPUI's `emoji_rasterization_fragment` with a viewport on the
/// layer's bounds and `SRC = ONE, DEST = INV_SRC_ALPHA` blending: `a = corrected(coverage) * color.a`,
/// `out = (color.rgb * a, a) + dst * (1 - a)`, stored as 8-bit UNORM (round to nearest) after each layer.
pub(super) fn compose_layer(
    target: &mut [u8],
    width: usize,
    height: usize,
    layer: &Layer,
    blend: &GlyphBlend,
) {
    let [r, g, b, a] = layer.color;
    let x0 = layer.left.max(0);
    let y0 = layer.top.max(0);
    let x1 = (layer.left + layer.width).min(width as i32);
    let y1 = (layer.top + layer.height).min(height as i32);
    for y in y0..y1 {
        let row = ((y - layer.top) * layer.width) as usize;
        for x in x0..x1 {
            let sample = layer.alpha[row + (x - layer.left) as usize] as f32 / 255.0;
            let corrected = apply_contrast_and_gamma_correction(
                sample,
                [r, g, b],
                blend.grayscale_enhanced_contrast,
                blend.gamma_ratios,
            );
            let alpha = corrected * a;
            let src = [b * alpha, g * alpha, r * alpha, alpha];
            let px = (y as usize * width + x as usize) * 4;
            for (channel, src) in target[px..px + 4].iter_mut().zip(src) {
                let dst = *channel as f32 / 255.0;
                *channel = unorm8(src + dst * (1.0 - alpha));
            }
        }
    }
}

/// D3D's float to 8-bit UNORM conversion: clamp, scale, round to nearest.
fn unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// What GPUI did with the composed target before putting it in its atlas: un-premultiply each pixel
/// (`channel * 255 / a`, clamped, truncated). Its sprite shader then blends the straight colour with
/// `SRC_ALPHA, INV_SRC_ALPHA`; a renderer that wants GPUI's exact emoji pixels applies this to the
/// BGRA premultiplied mask from `rasterize_glyph` and blends straight.
pub fn unpremultiply_like_gpui(bgra: &mut [u8]) {
    for chunk in bgra.as_chunks_mut::<4>().0 {
        let a = chunk[3] as f32;
        if a > 0.0 {
            let inv_a = 255.0 / a;
            for channel in &mut chunk[..3] {
                *channel = (*channel as f32 * inv_a).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Layer, compose_layer, unorm8, unpremultiply_like_gpui};
    use crate::GlyphBlend;

    #[test]
    fn layers_blend_premultiplied_over_each_other() {
        // Identity correction (gamma ratios 0, contrast 0) keeps the coverage as is.
        let blend = GlyphBlend::new([0.; 4], 0.);
        let mut target = vec![0u8; 2 * 4];
        let full = [255u8, 255];
        let red = Layer { left: 0, top: 0, width: 2, height: 1, alpha: &full, color: [1., 0., 0., 1.] };
        compose_layer(&mut target, 2, 1, &red, &blend);
        assert_eq!(target, [0, 0, 255, 255, 0, 0, 255, 255]);
        // Half-covered blue over the right pixel only (the layer hangs off the target's right edge).
        let half = [128u8, 128];
        let blue = Layer { left: 1, top: 0, width: 2, height: 1, alpha: &half, color: [0., 0., 1., 1.] };
        compose_layer(&mut target, 2, 1, &blue, &blend);
        let a = 128.0f32 / 255.0;
        assert_eq!(&target[4..], [unorm8(a), 0, unorm8(1. - a), 255]);
        assert_eq!(&target[..4], [0, 0, 255, 255]);
    }

    #[test]
    fn unpremultiply_matches_gpui() {
        let mut px = [64u8, 32, 0, 128, 10, 10, 10, 0];
        unpremultiply_like_gpui(&mut px);
        assert_eq!(px, [127, 63, 0, 128, 10, 10, 10, 0]);
    }
}
