//! `FixedTextSystem`: a tiny deterministic text backend for tests. Every glyph advances 0.6 em, the
//! ascent is 0.8 em and the descent 0.2 em, a glyph's id is its char code, and its mask is a solid
//! block (empty for whitespace and control characters).

use crate::{
    Bounds, DevicePixels, Font, FontId, FontMetrics, FontRun, GlyphBlend, GlyphId, LineLayout, Pixels,
    PlatformTextSystem, RenderGlyphParams, ShapedGlyph, ShapedRun, Size, point, px, size,
};
use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use std::borrow::Cow;

/// Font units per em of the fixed font.
const UNITS_PER_EM: u32 = 1000;
/// The advance of every glyph, in ems.
pub const FIXED_ADVANCE_EM: f32 = 0.6;
/// The ascent, in ems.
pub const FIXED_ASCENT_EM: f32 = 0.8;
/// The descent, in ems (positive, below the baseline).
pub const FIXED_DESCENT_EM: f32 = 0.2;

/// A deterministic `PlatformTextSystem` for tests (see the module docs).
#[derive(Default)]
pub struct FixedTextSystem {
    fonts: Mutex<Vec<Font>>,
}

impl FixedTextSystem {
    /// A fixed text system with no fonts resolved yet.
    pub fn new() -> Self {
        Self::default()
    }
}

fn is_blank(id: GlyphId) -> bool {
    char::from_u32(id.0).is_none_or(|c| c.is_whitespace() || c.is_control())
}

impl PlatformTextSystem for FixedTextSystem {
    fn add_fonts(&self, _fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        Ok(())
    }

    fn all_font_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.fonts.lock().iter().map(|f| f.family.to_string()).collect();
        names.dedup();
        names
    }

    fn font_id(&self, descriptor: &Font) -> Result<FontId> {
        let mut fonts = self.fonts.lock();
        if let Some(i) = fonts.iter().position(|f| f == descriptor) {
            return Ok(FontId(i));
        }
        fonts.push(descriptor.clone());
        Ok(FontId(fonts.len() - 1))
    }

    fn font_metrics(&self, _font_id: FontId) -> FontMetrics {
        let em = UNITS_PER_EM as f32;
        FontMetrics {
            units_per_em: UNITS_PER_EM,
            ascent: FIXED_ASCENT_EM * em,
            descent: -FIXED_DESCENT_EM * em,
            line_gap: 0.,
            underline_position: -0.1 * em,
            underline_thickness: 0.05 * em,
            cap_height: 0.7 * em,
            x_height: 0.5 * em,
            bounding_box: Bounds::new(point(0., -FIXED_DESCENT_EM * em), size(FIXED_ADVANCE_EM * em, em)),
        }
    }

    fn typographic_bounds(&self, font_id: FontId, _glyph_id: GlyphId) -> Result<Bounds<f32>> {
        Ok(self.font_metrics(font_id).bounding_box)
    }

    fn advance(&self, _font_id: FontId, _glyph_id: GlyphId) -> Result<Size<f32>> {
        Ok(size(FIXED_ADVANCE_EM * UNITS_PER_EM as f32, 0.))
    }

    fn glyph_for_char(&self, _font_id: FontId, ch: char) -> Option<GlyphId> {
        Some(GlyphId(ch as u32))
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        if is_blank(params.glyph_id) {
            return Ok(Bounds::default());
        }
        let em = params.font_size.0 * params.scale_factor;
        let width = (FIXED_ADVANCE_EM * em).ceil() as i32;
        let ascent = (FIXED_ASCENT_EM * em).ceil() as i32;
        let descent = (FIXED_DESCENT_EM * em).ceil() as i32;
        Ok(Bounds::new(
            point(DevicePixels(0), DevicePixels(-ascent)),
            size(DevicePixels(width), DevicePixels(ascent + descent)),
        ))
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        let (w, h) = (raster_bounds.size.width.0, raster_bounds.size.height.0);
        if w < 0 || h < 0 {
            return Err(anyhow!("negative raster bounds {raster_bounds:?}"));
        }
        let bytes_per_pixel = if params.is_emoji { 4 } else { 1 };
        Ok((raster_bounds.size, vec![255; w as usize * h as usize * bytes_per_pixel]))
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        let advance = font_size * FIXED_ADVANCE_EM;
        let mut x = px(0.);
        let mut start = 0;
        let mut shaped = Vec::with_capacity(runs.len());
        for run in runs {
            let end = (start + run.len).min(text.len());
            let glyphs = text[start..end]
                .char_indices()
                .map(|(i, ch)| {
                    let glyph = ShapedGlyph {
                        id: GlyphId(ch as u32),
                        position: point(x, px(0.)),
                        index: start + i,
                        is_emoji: false,
                    };
                    x += advance;
                    glyph
                })
                .collect();
            shaped.push(ShapedRun { font_id: run.font_id, glyphs });
            start = end;
        }
        LineLayout {
            font_size,
            width: x,
            ascent: font_size * FIXED_ASCENT_EM,
            descent: font_size * FIXED_DESCENT_EM,
            runs: shaped,
            len: text.len(),
        }
    }

    fn glyph_blend(&self) -> GlyphBlend {
        GlyphBlend::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, font};

    #[test]
    fn lays_out_fixed_advances_across_runs() {
        let ts = FixedTextSystem::new();
        let a = ts.font_id(&font("A")).unwrap();
        let b = ts.font_id(&font("B")).unwrap();
        assert_eq!(ts.font_id(&font("A")).unwrap(), a);
        let line = ts.layout_line(
            "héllo",
            px(10.),
            &[FontRun { len: 3, font_id: a }, FontRun { len: 3, font_id: b }],
        );
        assert_eq!(line.len, 6);
        assert_eq!(line.width, px(30.));
        assert_eq!((line.ascent, line.descent), (px(8.), px(2.)));
        let indices: Vec<usize> = line.runs.iter().flat_map(|r| r.glyphs.iter().map(|g| g.index)).collect();
        assert_eq!(indices, [0, 1, 3, 4, 5]);
        assert_eq!(line.runs[1].glyphs[0].position.x, px(12.));
        assert_eq!(line.runs[1].glyphs[0].id, GlyphId('l' as u32));
    }

    #[test]
    fn masks_are_solid_blocks() {
        let ts = FixedTextSystem::new();
        let params = RenderGlyphParams {
            font_id: FontId(0),
            glyph_id: GlyphId('x' as u32),
            font_size: px(10.),
            subpixel_variant: Point::default(),
            scale_factor: 2.,
            is_emoji: false,
        };
        let bounds = ts.glyph_raster_bounds(&params).unwrap();
        assert_eq!(bounds.origin.y, DevicePixels(-16));
        assert_eq!(bounds.size, size(DevicePixels(12), DevicePixels(20)));
        let (sz, mask) = ts.rasterize_glyph(&params, bounds).unwrap();
        assert_eq!(sz, bounds.size);
        assert!(mask.len() == 240 && mask.iter().all(|&a| a == 255));
        let space = RenderGlyphParams { glyph_id: GlyphId(' ' as u32), ..params };
        assert_eq!(ts.glyph_raster_bounds(&space).unwrap(), Bounds::default());
    }
}
