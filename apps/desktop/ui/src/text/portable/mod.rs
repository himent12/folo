//! The cosmic-text `PlatformTextSystem` (Linux, macOS, and Windows tests with feature `portable-text`).
//!
//! It behaves as GPUI's Linux backend (GPUI 0.2.2's `src/platform/linux/text_system.rs`) with two
//! differences on purpose: the font database starts with only the fonts given to `add_fonts` (system
//! fonts are loaded the first time a glyph or a family is missing), and nothing depends on the locale, so
//! the same input gives the same layout and masks on every OS until a fallback is needed.

mod fonts;
mod layout;
mod matching;
mod raster;

use crate::{
    Bounds, DevicePixels, Font, FontId, FontMetrics, FontRun, GlyphBlend, GlyphId, LineLayout, Pixels,
    PlatformTextSystem, RenderGlyphParams, Size,
};
use anyhow::Result;
use parking_lot::Mutex;
use std::borrow::Cow;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use raster::portable_glyph_blend;

/// The cosmic-text text backend (see the module docs).
pub struct CosmicTextSystem {
    state: Mutex<fonts::State>,
    system_font_loads: AtomicUsize,
}

impl CosmicTextSystem {
    /// A text system that knows no fonts yet (add them with `add_fonts`); it does not scan the system.
    pub fn new() -> Self {
        Self { state: Mutex::new(fonts::State::new()), system_font_loads: AtomicUsize::new(0) }
    }

    /// How many times system fonts were loaded (0 or 1): 0 until a glyph or family the added fonts lack
    /// is needed.
    pub fn system_font_loads(&self) -> usize {
        self.system_font_loads.load(Ordering::Relaxed)
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut fonts::State) -> R) -> R {
        let mut state = self.state.lock();
        let loaded_before = state.system_fonts_loaded();
        let result = f(&mut state);
        if !loaded_before && state.system_fonts_loaded() {
            self.system_font_loads.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

impl Default for CosmicTextSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl PlatformTextSystem for CosmicTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.with_state(|state| state.add_fonts(fonts));
        Ok(())
    }

    fn all_font_names(&self) -> Vec<String> {
        self.with_state(|state| state.all_font_names())
    }

    fn font_id(&self, descriptor: &Font) -> Result<FontId> {
        self.with_state(|state| state.font_id(descriptor))
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        self.with_state(|state| state.font_metrics(font_id))
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        self.with_state(|state| state.typographic_bounds(font_id, glyph_id))
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.with_state(|state| state.advance(font_id, glyph_id))
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.with_state(|state| state.glyph_for_char(font_id, ch))
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.with_state(|state| state.raster_bounds(params))
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        self.with_state(|state| state.rasterize_glyph(params, raster_bounds))
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        self.with_state(|state| state.layout_line(text, font_size, runs))
    }

    fn glyph_blend(&self) -> GlyphBlend {
        portable_glyph_blend()
    }
}
