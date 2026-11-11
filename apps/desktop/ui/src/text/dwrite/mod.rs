// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/direct_write.rs (state, add_fonts)
//! The DirectWrite `PlatformTextSystem` (Windows), used the way GPUI 0.2.2 uses it
//! (`platform/windows/direct_write.rs`): the same calls with the same parameters, so line widths and
//! glyph masks are identical. Unlike GPUI it creates no GPU device (colour glyph layers are composed on
//! the CPU) and builds nothing system-wide at start: the system font collection is loaded the first
//! time a font or a fallback needs it.

mod blend;
mod fonts;
mod layout;
mod raster;

pub use blend::{apply_contrast_and_gamma_correction, gamma_ratios};
pub use raster::unpremultiply_like_gpui;

use crate::{
    Bounds, DevicePixels, Font, FontId, FontMetrics, FontRun, GlyphBlend, GlyphId, LineLayout, Pixels,
    PlatformTextSystem, RenderGlyphParams, SharedString, Size,
};
use anyhow::Result;
use fonts::{FontIdentifier, FontInfo};
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::OnceLock;
use windows::Win32::Globalization::GetUserDefaultLocaleName;
use windows::Win32::Graphics::DirectWrite::{
    DWRITE_FACTORY_TYPE_SHARED, DWriteCreateFactory, IDWriteFactory5, IDWriteFontCollection1,
    IDWriteFontSetBuilder1, IDWriteInMemoryFontFileLoader, IDWriteTextRenderer,
};
use windows::Win32::System::SystemServices::LOCALE_NAME_MAX_LENGTH;

/// The DirectWrite text backend.
pub struct DirectWriteTextSystem {
    state: RwLock<DirectWriteState>,
    blend: OnceLock<GlyphBlend>,
}

// SAFETY: the factory is created with `DWRITE_FACTORY_TYPE_SHARED`, whose objects (factory, font
// faces, collections, the in-memory loader) are free-threaded; every mutation of the state goes through
// the `RwLock`, and layout (which drives our text renderer) takes the write lock.
unsafe impl Send for DirectWriteTextSystem {}
// SAFETY: as above.
unsafe impl Sync for DirectWriteTextSystem {}

struct DirectWriteState {
    locale: String,
    factory: IDWriteFactory5,
    in_memory_loader: IDWriteInMemoryFontFileLoader,
    builder: IDWriteFontSetBuilder1,
    text_renderer: IDWriteTextRenderer,
    /// The system UI font family (GPUI's `get_system_ui_font_name`), read on first need.
    system_ui_font_name: Option<SharedString>,
    /// The system font collection, loaded on first need (GPUI loads it at start).
    system_font_collection: Option<IDWriteFontCollection1>,
    /// The collection of the fonts `add_fonts` added (none before the first call).
    custom_font_collection: Option<IDWriteFontCollection1>,
    fonts: Vec<FontInfo>,
    font_selections: HashMap<Font, FontId>,
    font_id_by_identifier: HashMap<FontIdentifier, FontId>,
    /// How often the system font collection was (re)loaded.
    system_collection_loads: usize,
}

impl DirectWriteTextSystem {
    /// Creates the factory, the in-memory font loader and the text renderer; nothing system-wide.
    pub fn new() -> Result<Self> {
        // SAFETY: plain DirectWrite factory calls; the loader is registered once and unregistered on
        // drop (`DirectWriteState::drop`).
        let state = unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
            // `IDWriteInMemoryFontFileLoader` needs Windows 10 1703 (as GPUI).
            let in_memory_loader = factory.CreateInMemoryFontFileLoader()?;
            factory.RegisterFontFileLoader(&in_memory_loader)?;
            let builder = factory.CreateFontSetBuilder()?;
            let mut locale_vec = vec![0u16; LOCALE_NAME_MAX_LENGTH as usize];
            GetUserDefaultLocaleName(&mut locale_vec);
            let end = locale_vec.iter().position(|&c| c == 0).unwrap_or(locale_vec.len());
            let locale = String::from_utf16_lossy(&locale_vec[..end]);
            let text_renderer = layout::text_renderer(&locale);
            DirectWriteState {
                locale,
                factory,
                in_memory_loader,
                builder,
                text_renderer,
                system_ui_font_name: None,
                system_font_collection: None,
                custom_font_collection: None,
                fonts: Vec::new(),
                font_selections: HashMap::default(),
                font_id_by_identifier: HashMap::default(),
                system_collection_loads: 0,
            }
        };
        Ok(Self { state: RwLock::new(state), blend: OnceLock::new() })
    }

    /// How many times the system font collection has been loaded (0 until a font the app did not
    /// bundle, a fallback list or the font names need it).
    pub fn system_collection_loads(&self) -> usize {
        self.state.read().system_collection_loads
    }
}

impl PlatformTextSystem for DirectWriteTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.state.write().add_fonts(fonts)
    }

    fn all_font_names(&self) -> Vec<String> {
        self.state.write().all_font_names()
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let lock = self.state.upgradable_read();
        if let Some(font_id) = lock.font_selections.get(font) {
            Ok(*font_id)
        } else {
            let mut lock = RwLockUpgradableReadGuard::upgrade(lock);
            let font_id = lock.select_font(font)?;
            lock.font_selections.insert(font.clone(), font_id);
            Ok(font_id)
        }
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        self.state.read().font_metrics(font_id)
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        self.state.read().typographic_bounds(font_id, glyph_id)
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.state.read().advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.state.read().glyph_for_char(font_id, ch)
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.state.read().raster_bounds(params)
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        let blend = self.glyph_blend();
        self.state.read().rasterize_glyph(params, raster_bounds, &blend)
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        // GPUI logs the error and returns an empty layout.
        self.state
            .write()
            .layout_line(text, font_size, runs)
            .unwrap_or(LineLayout { font_size, ..Default::default() })
    }

    fn glyph_blend(&self) -> GlyphBlend {
        *self.blend.get_or_init(|| {
            let factory = self.state.read().factory.clone();
            blend::system_glyph_blend(&factory).unwrap_or(GlyphBlend::IDENTITY)
        })
    }
}

impl DirectWriteState {
    fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        for font_data in fonts {
            // SAFETY: with no owner object the loader copies the data, so `font_data` may be dropped.
            unsafe {
                let font_file = self.in_memory_loader.CreateInMemoryFontFileReference(
                    &self.factory,
                    font_data.as_ptr() as _,
                    font_data.len() as _,
                    None,
                )?;
                self.builder.AddFontFile(&font_file)?;
            }
        }
        // SAFETY: plain DirectWrite calls on live objects.
        let collection = unsafe {
            let set = self.builder.CreateFontSet()?;
            self.factory.CreateFontCollectionFromFontSet(&set)?
        };
        self.custom_font_collection = Some(collection);
        // A family may now be in the bundled collection: select fonts again (GPUI kept its selections, so
        // a family asked for before its font was added stayed on the system UI font). Ids handed out
        // stay valid.
        self.font_selections.clear();
        Ok(())
    }
}

impl Drop for DirectWriteState {
    fn drop(&mut self) {
        // SAFETY: the loader was registered with this factory in `new`.
        unsafe {
            let _ = self.factory.UnregisterFontFileLoader(&self.in_memory_loader);
        }
    }
}
