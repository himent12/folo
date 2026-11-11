// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system.rs
//! `TextSystem`: the app-wide text front (font ids with GPUI's fallback stack, cached metrics and raster
//! bounds, the line wrapper pool) over a `PlatformTextSystem`.
//!
//! Adding fonts can change what a family resolves to and which face a character falls back to. So
//! `add_fonts` forgets the resolutions and moves the font generation on, and everything that keeps a
//! result of resolution or fallback carries the generation it was made under: the line layout caches
//! (in their keys), the line wrappers (a wrapper from an older generation is not pooled again) and the
//! windows' layout reuse (`window/layout_tree.rs`). A `FontId` always names the same face, so ids
//! handed out stay valid (scenes already made paint as they were), and what is keyed by id alone stays
//! right: metrics, raster bounds and glyph masks. (GPUI 0.2.2 kept the old resolutions: a family drawn
//! before its font was added stayed on its fallback.)

use super::fx_hash::FxHashMap;
use crate::{
    Bounds, DevicePixels, Font, FontId, FontMetrics, GlyphBlend, LineWrapper, Pixels, PlatformTextSystem,
    RenderGlyphParams, Size, font, px,
};
use anyhow::{Context as _, Result, anyhow};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// The app-wide text system: resolves fonts, caches metrics and glyph raster bounds, and pools line
/// wrappers. Each window adds a line layout cache on top (`WindowTextSystem`).
pub struct TextSystem {
    pub(crate) platform_text_system: Arc<dyn PlatformTextSystem>,
    font_ids_by_font: RwLock<FxHashMap<Font, Result<FontId>>>,
    font_metrics: RwLock<FxHashMap<FontId, FontMetrics>>,
    raster_bounds: RwLock<FxHashMap<RenderGlyphParams, Bounds<DevicePixels>>>,
    wrapper_pool: Mutex<FxHashMap<FontIdWithSize, Vec<LineWrapper>>>,
    pub(crate) font_runs_pool: Mutex<Vec<Vec<crate::FontRun>>>,
    fallback_font_stack: SmallVec<[Font; 2]>,
    glyph_blend: OnceLock<GlyphBlend>,
    /// How many times fonts were added (see the module docs).
    font_generation: AtomicU64,
}

impl TextSystem {
    /// A text system over `platform_text_system`, with GPUI's fallback font stack.
    pub fn new(platform_text_system: Arc<dyn PlatformTextSystem>) -> Self {
        TextSystem {
            platform_text_system,
            font_metrics: RwLock::default(),
            raster_bounds: RwLock::default(),
            font_ids_by_font: RwLock::default(),
            wrapper_pool: Mutex::default(),
            font_runs_pool: Mutex::default(),
            // GPUI's stack, in GPUI's order.
            fallback_font_stack: smallvec![
                font(".ZedMono"),
                font(".ZedSans"),
                font("Helvetica"),
                font("Segoe UI"),
                font("Ubuntu"),
                font("Adwaita Sans"),
                font("Cantarell"),
                font("Noto Sans"),
                font("DejaVu Sans"),
                font("Arial"),
            ],
            glyph_blend: OnceLock::new(),
            font_generation: AtomicU64::new(0),
        }
    }

    /// All font family names: the backend's, the fallback stack's and ".SystemUIFont", sorted.
    pub fn all_font_names(&self) -> Vec<String> {
        let mut names = self.platform_text_system.all_font_names();
        names.extend(self.fallback_font_stack.iter().map(|font| font.family.to_string()));
        names.push(".SystemUIFont".to_string());
        names.sort();
        names.dedup();
        names
    }

    /// Adds font files (the app's bundled fonts). Families are resolved again from now on, and the font
    /// generation moves on (see the module docs); font ids already handed out stay valid.
    pub fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.platform_text_system.add_fonts(fonts)?;
        // Cleared and moved on under both locks (the pool's first, as `line_wrapper` takes them), so a
        // resolution or a wrapper from before the add cannot be published after it.
        let mut wrapper_pool = self.wrapper_pool.lock();
        let mut font_ids = self.font_ids_by_font.write();
        font_ids.clear();
        wrapper_pool.clear();
        self.font_generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Moves on each time fonts are added: what was shaped or measured under another generation may
    /// resolve or fall back differently now.
    pub fn font_generation(&self) -> u64 {
        self.font_generation.load(Ordering::Acquire)
    }

    /// The id of `font`, cached (also when the backend fails, as GPUI).
    pub fn font_id(&self, font: &Font) -> Result<FontId> {
        fn clone_font_id_result(font_id: &Result<FontId>) -> Result<FontId> {
            match font_id {
                Ok(font_id) => Ok(*font_id),
                Err(err) => Err(anyhow!("{err}")),
            }
        }

        let font_id = self.font_ids_by_font.read().get(font).map(clone_font_id_result);
        if let Some(font_id) = font_id {
            font_id
        } else {
            let generation = self.font_generation();
            let font_id = self.platform_text_system.font_id(font);
            // Not cached if fonts were added meanwhile (on another thread): the answer may be the old one.
            let mut font_ids = self.font_ids_by_font.write();
            if self.font_generation() == generation {
                font_ids.insert(font.clone(), clone_font_id_result(&font_id));
            }
            font_id
        }
    }

    /// The font a font id was resolved from, if any.
    pub fn get_font_for_id(&self, id: FontId) -> Option<Font> {
        let lock = self.font_ids_by_font.read();
        lock.iter()
            .filter_map(|(font, result)| match result {
                Ok(font_id) if *font_id == id => Some(font.clone()),
                _ => None,
            })
            .next()
    }

    /// Resolves `font`, falling back to the default font stack if it fails to load.
    ///
    /// # Panics
    ///
    /// Panics if the font and none of the fallbacks can be resolved (as GPUI).
    pub fn resolve_font(&self, font: &Font) -> FontId {
        if let Ok(font_id) = self.font_id(font) {
            return font_id;
        }
        for fallback in &self.fallback_font_stack {
            if let Ok(font_id) = self.font_id(fallback) {
                return font_id;
            }
        }
        let stack: Vec<&str> = self.fallback_font_stack.iter().map(|f| f.family.as_ref()).collect();
        panic!("failed to resolve font '{}' or any of the fallbacks: {}", font.family, stack.join(", "));
    }

    /// The font's bounding box (every glyph superimposed) at `font_size`.
    pub fn bounding_box(&self, font_id: FontId, font_size: Pixels) -> Bounds<Pixels> {
        self.read_metrics(font_id, |metrics| metrics.bounding_box(font_size))
    }

    /// The typographic bounds of `character` in the font at `font_size`.
    pub fn typographic_bounds(
        &self,
        font_id: FontId,
        font_size: Pixels,
        character: char,
    ) -> Result<Bounds<Pixels>> {
        let glyph_id = self
            .platform_text_system
            .glyph_for_char(font_id, character)
            .with_context(|| format!("glyph not found for character '{character}'"))?;
        let bounds = self.platform_text_system.typographic_bounds(font_id, glyph_id)?;
        Ok(self.read_metrics(font_id, |metrics| (bounds / metrics.units_per_em as f32 * font_size.0).map(px)))
    }

    /// The advance of `ch` in the font at `font_size`.
    pub fn advance(&self, font_id: FontId, font_size: Pixels, ch: char) -> Result<Size<Pixels>> {
        let glyph_id = self
            .platform_text_system
            .glyph_for_char(font_id, ch)
            .with_context(|| format!("glyph not found for character '{ch}'"))?;
        let result =
            self.platform_text_system.advance(font_id, glyph_id)? / self.units_per_em(font_id) as f32;
        Ok(result * font_size)
    }

    /// The width of an em (the typographic width of `m`).
    pub fn em_width(&self, font_id: FontId, font_size: Pixels) -> Result<Pixels> {
        Ok(self.typographic_bounds(font_id, font_size, 'm')?.size.width)
    }

    /// The advance of an em (the advance of `m`).
    pub fn em_advance(&self, font_id: FontId, font_size: Pixels) -> Result<Pixels> {
        Ok(self.advance(font_id, font_size, 'm')?.width)
    }

    /// The width of a ch (the typographic width of `0`).
    pub fn ch_width(&self, font_id: FontId, font_size: Pixels) -> Result<Pixels> {
        Ok(self.typographic_bounds(font_id, font_size, '0')?.size.width)
    }

    /// The advance of a ch (the advance of `0`).
    pub fn ch_advance(&self, font_id: FontId, font_size: Pixels) -> Result<Pixels> {
        Ok(self.advance(font_id, font_size, '0')?.width)
    }

    /// Font units per em.
    pub fn units_per_em(&self, font_id: FontId) -> u32 {
        self.read_metrics(font_id, |metrics| metrics.units_per_em)
    }

    /// The height of a capital letter at `font_size`.
    pub fn cap_height(&self, font_id: FontId, font_size: Pixels) -> Pixels {
        self.read_metrics(font_id, |metrics| metrics.cap_height(font_size))
    }

    /// The height of a lowercase x at `font_size`.
    pub fn x_height(&self, font_id: FontId, font_size: Pixels) -> Pixels {
        self.read_metrics(font_id, |metrics| metrics.x_height(font_size))
    }

    /// The distance from the baseline to the top of the glyphs at `font_size`.
    pub fn ascent(&self, font_id: FontId, font_size: Pixels) -> Pixels {
        self.read_metrics(font_id, |metrics| metrics.ascent(font_size))
    }

    /// The font's descent at `font_size` (negative: below the baseline, as GPUI).
    pub fn descent(&self, font_id: FontId, font_size: Pixels) -> Pixels {
        self.read_metrics(font_id, |metrics| metrics.descent(font_size))
    }

    /// The baseline's distance from the top of a line of `line_height` (GPUI's formula, with its signed
    /// descent).
    pub fn baseline_offset(&self, font_id: FontId, font_size: Pixels, line_height: Pixels) -> Pixels {
        let ascent = self.ascent(font_id, font_size);
        let descent = self.descent(font_id, font_size);
        let padding_top = (line_height - ascent - descent) / 2.;
        padding_top + ascent
    }

    fn read_metrics<T>(&self, font_id: FontId, read: impl FnOnce(&FontMetrics) -> T) -> T {
        let lock = self.font_metrics.upgradable_read();
        if let Some(metrics) = lock.get(&font_id) {
            read(metrics)
        } else {
            let mut lock = RwLockUpgradableReadGuard::upgrade(lock);
            let metrics =
                lock.entry(font_id).or_insert_with(|| self.platform_text_system.font_metrics(font_id));
            read(metrics)
        }
    }

    /// A line wrapper for `font` at `font_size`, from the pool (it returns to the pool when dropped).
    pub fn line_wrapper(self: &Arc<Self>, font: Font, font_size: Pixels) -> LineWrapperHandle {
        let lock = &mut self.wrapper_pool.lock();
        let font_generation = self.font_generation();
        let font_id = self.resolve_font(&font);
        let wrappers = lock.entry(FontIdWithSize { font_id, font_size }).or_default();
        let wrapper = wrappers
            .pop()
            .unwrap_or_else(|| LineWrapper::new(font_id, font_size, self.platform_text_system.clone()));
        LineWrapperHandle { wrapper: Some(wrapper), text_system: self.clone(), font_generation }
    }

    /// The device-pixel bounds of a glyph's raster, relative to its (floored) origin; cached.
    pub fn raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let raster_bounds = self.raster_bounds.upgradable_read();
        if let Some(bounds) = raster_bounds.get(params) {
            Ok(*bounds)
        } else {
            let mut raster_bounds = RwLockUpgradableReadGuard::upgrade(raster_bounds);
            let bounds = self.platform_text_system.glyph_raster_bounds(params)?;
            raster_bounds.insert(params.clone(), bounds);
            Ok(bounds)
        }
    }

    /// Rasterizes a glyph into its raster bounds: an alpha mask, or BGRA premultiplied for an emoji.
    pub fn rasterize_glyph(&self, params: &RenderGlyphParams) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        let raster_bounds = self.raster_bounds(params)?;
        self.platform_text_system.rasterize_glyph(params, raster_bounds)
    }

    /// The contrast / gamma correction the renderer applies to glyph masks (asked once).
    pub fn glyph_blend(&self) -> GlyphBlend {
        *self.glyph_blend.get_or_init(|| self.platform_text_system.glyph_blend())
    }
}

#[derive(Hash, Eq, PartialEq)]
struct FontIdWithSize {
    font_id: FontId,
    font_size: Pixels,
}

/// A line wrapper borrowed from the text system's pool; it goes back when dropped.
pub struct LineWrapperHandle {
    wrapper: Option<LineWrapper>,
    text_system: Arc<TextSystem>,
    /// The font generation the wrapper was taken under (its cached widths are of those fonts).
    font_generation: u64,
}

impl Drop for LineWrapperHandle {
    fn drop(&mut self) {
        let mut state = self.text_system.wrapper_pool.lock();
        let wrapper = self.wrapper.take().expect("a handle holds its wrapper until dropped");
        if self.font_generation != self.text_system.font_generation() {
            // Fonts were added meanwhile: its character widths may be stale.
            return;
        }
        state
            .entry(FontIdWithSize { font_id: wrapper.font_id, font_size: wrapper.font_size })
            .or_default()
            .push(wrapper);
    }
}

impl Deref for LineWrapperHandle {
    type Target = LineWrapper;

    fn deref(&self) -> &Self::Target {
        self.wrapper.as_ref().expect("a handle holds its wrapper until dropped")
    }
}

impl DerefMut for LineWrapperHandle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.wrapper.as_mut().expect("a handle holds its wrapper until dropped")
    }
}
