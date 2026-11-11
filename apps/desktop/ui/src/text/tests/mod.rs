//! Tests of the text front against `FixedTextSystem` (every glyph 0.6 em: 9.6 px at 16 px, the same
//! widths as GPUI's `NoopTextSystem`, so GPUI's own wrapping and truncation cases carry over).

mod cache;
mod layout;
mod paint;
mod wrapper;

use crate::{
    FixedTextSystem, Font, FontFeatures, FontStyle, FontWeight, Hsla, PlatformTextSystem, TextRun,
    TextSystem, WindowTextSystem, font,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) fn platform() -> Arc<dyn PlatformTextSystem> {
    Arc::new(FixedTextSystem::new())
}

pub(super) fn text_system() -> Arc<TextSystem> {
    Arc::new(TextSystem::new(platform()))
}

pub(super) fn window_text_system() -> WindowTextSystem {
    WindowTextSystem::new(text_system())
}

pub(super) fn dummy_font() -> Font {
    Font {
        family: "Dummy".into(),
        features: FontFeatures::default(),
        fallbacks: None,
        weight: FontWeight::default(),
        style: FontStyle::Normal,
    }
}

/// Runs of the dummy font with these lengths (GPUI's `generate_test_runs`).
pub(super) fn runs(lens: &[usize]) -> Vec<TextRun> {
    lens.iter().map(|len| run(*len, Hsla::default())).collect()
}

pub(super) fn run(len: usize, color: Hsla) -> TextRun {
    TextRun { len, font: dummy_font(), color, background_color: None, underline: None, strikethrough: None }
}

pub(super) fn mono() -> Font {
    font(".ZedMono")
}

/// `FixedTextSystem` that counts `layout_line` calls (what the caches must avoid).
#[derive(Default)]
pub(super) struct CountingTextSystem {
    inner: FixedTextSystem,
    layout_calls: AtomicUsize,
}

impl CountingTextSystem {
    pub(super) fn layout_calls(&self) -> usize {
        self.layout_calls.load(Ordering::Relaxed)
    }
}

impl PlatformTextSystem for CountingTextSystem {
    fn add_fonts(&self, fonts: Vec<std::borrow::Cow<'static, [u8]>>) -> anyhow::Result<()> {
        self.inner.add_fonts(fonts)
    }
    fn all_font_names(&self) -> Vec<String> {
        self.inner.all_font_names()
    }
    fn font_id(&self, descriptor: &Font) -> anyhow::Result<crate::FontId> {
        self.inner.font_id(descriptor)
    }
    fn font_metrics(&self, font_id: crate::FontId) -> crate::FontMetrics {
        self.inner.font_metrics(font_id)
    }
    fn typographic_bounds(
        &self,
        font_id: crate::FontId,
        glyph_id: crate::GlyphId,
    ) -> anyhow::Result<crate::Bounds<f32>> {
        self.inner.typographic_bounds(font_id, glyph_id)
    }
    fn advance(&self, font_id: crate::FontId, glyph_id: crate::GlyphId) -> anyhow::Result<crate::Size<f32>> {
        self.inner.advance(font_id, glyph_id)
    }
    fn glyph_for_char(&self, font_id: crate::FontId, ch: char) -> Option<crate::GlyphId> {
        self.inner.glyph_for_char(font_id, ch)
    }
    fn glyph_raster_bounds(
        &self,
        params: &crate::RenderGlyphParams,
    ) -> anyhow::Result<crate::Bounds<crate::DevicePixels>> {
        self.inner.glyph_raster_bounds(params)
    }
    fn rasterize_glyph(
        &self,
        params: &crate::RenderGlyphParams,
        raster_bounds: crate::Bounds<crate::DevicePixels>,
    ) -> anyhow::Result<(crate::Size<crate::DevicePixels>, Vec<u8>)> {
        self.inner.rasterize_glyph(params, raster_bounds)
    }
    fn layout_line(
        &self,
        text: &str,
        font_size: crate::Pixels,
        runs: &[crate::FontRun],
    ) -> crate::LineLayout {
        self.layout_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.layout_line(text, font_size, runs)
    }
    fn glyph_blend(&self) -> crate::GlyphBlend {
        self.inner.glyph_blend()
    }
}
