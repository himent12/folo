// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/linux/text_system.rs
//! The backend's state: the font database (bundled fonts first, system fonts on first need), the faces
//! resolved to `FontId`s, metrics and character lookups.

use super::matching::{FaceProperties, find_best_match};
use crate::{
    Bounds, Font, FontFeatures, FontId, FontMetrics, FontStyle, FontWeight, GlyphId, SharedString, Size,
    point, size,
};
use anyhow::{Context as _, Result};
use cosmic_text::fontdb;
use cosmic_text::{FontFeatures as CosmicFontFeatures, FontSystem, ShapeBuffer, SwashCache};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// The locale cosmic-text is created with: fixed, so layout never depends on the user's settings.
const LOCALE: &str = "en-US";

/// A family and feature set, the key faces are resolved under.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontKey {
    family: SharedString,
    features: FontFeatures,
}

/// One face the backend handed out a `FontId` for.
pub(super) struct LoadedFont {
    pub font: Arc<cosmic_text::Font>,
    pub features: CosmicFontFeatures,
    pub is_known_emoji_font: bool,
}

/// Everything behind the backend's lock.
pub(super) struct State {
    pub font_system: FontSystem,
    system_fonts_loaded: bool,
    pub swash: SwashCache,
    pub scratch: ShapeBuffer,
    /// Every face handed out, indexed by `FontId`.
    pub loaded_fonts: Vec<LoadedFont>,
    font_ids_by_family: HashMap<FontKey, SmallVec<[FontId; 4]>>,
    pub images: super::raster::ImageCache,
    /// The platform's vertical subpixel positions (`SUBPIXEL_VARIANTS_Y`; tests set others).
    pub subpixel_variants_y: u8,
}

impl State {
    pub fn new() -> Self {
        Self {
            font_system: FontSystem::new_with_locale_and_db(LOCALE.into(), fontdb::Database::new()),
            system_fonts_loaded: false,
            swash: SwashCache::new(),
            scratch: ShapeBuffer::default(),
            loaded_fonts: Vec::new(),
            font_ids_by_family: HashMap::new(),
            images: Default::default(),
            subpixel_variants_y: crate::SUBPIXEL_VARIANTS_Y,
        }
    }

    pub fn system_fonts_loaded(&self) -> bool {
        self.system_fonts_loaded
    }

    /// Loads the system's fonts into the database, once. Faces already resolved keep their ids.
    pub fn load_system_fonts(&mut self) {
        if !std::mem::replace(&mut self.system_fonts_loaded, true) {
            self.font_system.db_mut().load_system_fonts();
        }
    }

    pub fn loaded_font(&self, font_id: FontId) -> &LoadedFont {
        &self.loaded_fonts[font_id.0]
    }

    pub fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) {
        let db = self.font_system.db_mut();
        for bytes in fonts {
            // Static data (the bundled fonts) is shared, not copied.
            let source = match bytes {
                Cow::Borrowed(data) => fontdb::Source::Binary(Arc::new(data)),
                Cow::Owned(data) => fontdb::Source::Binary(Arc::new(data)),
            };
            db.load_font_source(source);
        }
    }

    pub fn all_font_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .font_system
            .db()
            .faces()
            .filter_map(|face| face.families.first().map(|family| family.0.clone()))
            .collect();
        names.sort();
        names.dedup();
        names
    }

    pub fn font_id(&mut self, font: &Font) -> Result<FontId> {
        let key = FontKey { family: font.family.clone(), features: font.features.clone() };
        let candidates = match self.font_ids_by_family.get(&key) {
            Some(ids) => ids.clone(),
            None => {
                let mut ids = self.load_family(&font.family, &font.features)?;
                if ids.is_empty() && !self.system_fonts_loaded {
                    self.load_system_fonts();
                    ids = self.load_family(&font.family, &font.features)?;
                }
                self.font_ids_by_family.insert(key, ids.clone());
                ids
            }
        };
        let properties: SmallVec<[FaceProperties; 4]> = candidates
            .iter()
            .map(|&id| {
                let face = self.font_system.db().face(self.loaded_font(id).font.id());
                face.map(face_properties).unwrap_or(FaceProperties {
                    weight: FontWeight::NORMAL,
                    style: FontStyle::Normal,
                    stretch: 100.,
                })
            })
            .collect();
        let query = FaceProperties { weight: font.weight, style: font.style, stretch: 100. };
        let index = find_best_match(&properties, &query)
            .context("requested font family contains no font matching the other parameters")?;
        Ok(candidates[index])
    }

    /// The faces of `name` as `FontId`s (bundled faces only, when the family was bundled).
    fn load_family(&mut self, name: &str, features: &FontFeatures) -> Result<SmallVec<[FontId; 4]>> {
        let name = font_name_with_fallbacks(name, "IBM Plex Sans");
        let faces: SmallVec<[(fontdb::ID, String, bool); 4]> = self
            .font_system
            .db()
            .faces()
            .filter(|face| face.families.iter().any(|family| family.0 == name))
            .map(|face| {
                (face.id, face.post_script_name.clone(), matches!(face.source, fontdb::Source::Binary(_)))
            })
            .collect();
        // A family the app bundled must not change because the same family is also installed.
        let any_bundled = faces.iter().any(|(_, _, bundled)| *bundled);

        let mut ids = SmallVec::new();
        for (face_id, postscript_name, bundled) in faces {
            if any_bundled && !bundled {
                continue;
            }
            let font = self.font_system.get_font(face_id).context("Could not load font")?;
            // GPUI's rule: a face without an "m" is a symbol font, not a text face (Segoe Fluent Icons
            // excepted, so Windows caption icons can render).
            let allowed_bad_font_names = ["SegoeFluentIcons", "Segoe Fluent Icons"];
            if font.as_swash().charmap().map('m') == 0
                && !allowed_bad_font_names.contains(&postscript_name.as_str())
            {
                self.font_system.db_mut().remove_face(font.id());
                continue;
            }
            let id = FontId(self.loaded_fonts.len());
            ids.push(id);
            self.loaded_fonts.push(LoadedFont {
                font,
                features: cosmic_features(features)?,
                is_known_emoji_font: is_known_emoji_font(&postscript_name),
            });
        }
        Ok(ids)
    }

    /// The `FontId` of a face cosmic-text chose as a fallback (added on the fly, with no features: only
    /// `layout_line`'s output uses it).
    pub fn font_id_for_cosmic_id(&mut self, id: fontdb::ID) -> FontId {
        if let Some(index) = self.loaded_fonts.iter().position(|loaded| loaded.font.id() == id) {
            return FontId(index);
        }
        let font = self.font_system.get_font(id).expect("cosmic-text shaped with a face it cannot load");
        let postscript_name =
            self.font_system.db().face(id).map(|f| f.post_script_name.clone()).unwrap_or_default();
        let font_id = FontId(self.loaded_fonts.len());
        self.loaded_fonts.push(LoadedFont {
            font,
            features: CosmicFontFeatures::new(),
            is_known_emoji_font: is_known_emoji_font(&postscript_name),
        });
        font_id
    }

    pub fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        let metrics = self.loaded_font(font_id).font.as_swash().metrics(&[]);
        FontMetrics {
            units_per_em: metrics.units_per_em as u32,
            ascent: metrics.ascent,
            descent: -metrics.descent,
            line_gap: metrics.leading,
            underline_position: metrics.underline_offset,
            underline_thickness: metrics.stroke_size,
            cap_height: metrics.cap_height,
            x_height: metrics.x_height,
            // GPUI's (approximate) box: the widest advance by ascent + descent.
            bounding_box: Bounds {
                origin: point(0.0, 0.0),
                size: size(metrics.max_width, metrics.ascent + metrics.descent),
            },
        }
    }

    pub fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        // GPUI's: the advance box, not the outline's extent.
        let advance = self.advance(font_id, glyph_id)?;
        Ok(Bounds { origin: point(0.0, 0.0), size: advance })
    }

    pub fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        let metrics = self.loaded_font(font_id).font.as_swash().glyph_metrics(&[]);
        let glyph = glyph_id.0 as u16;
        Ok(size(metrics.advance_width(glyph), metrics.advance_height(glyph)))
    }

    pub fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        let glyph = self.loaded_font(font_id).font.as_swash().charmap().map(ch);
        (glyph != 0).then_some(GlyphId(glyph.into()))
    }
}

/// GPUI's family aliases (`text_system::font_name_with_fallbacks`).
fn font_name_with_fallbacks<'a>(name: &'a str, system: &'a str) -> &'a str {
    match name {
        ".SystemUIFont" => system,
        ".ZedSans" | "Zed Plex Sans" => "IBM Plex Sans",
        ".ZedMono" | "Zed Plex Mono" => "Lilex",
        _ => name,
    }
}

/// Colour emoji faces (GPUI knows Noto Color Emoji; Apple's and Segoe's are added so emoji are colour on
/// macOS and on the Windows dev machine too).
fn is_known_emoji_font(postscript_name: &str) -> bool {
    matches!(postscript_name, "NotoColorEmoji" | "AppleColorEmoji" | "SegoeUIEmoji")
}

fn cosmic_features(features: &FontFeatures) -> Result<CosmicFontFeatures> {
    let mut result = CosmicFontFeatures::new();
    for (name, value) in features.tag_value_list() {
        let tag: [u8; 4] = name.as_bytes().try_into().context("Incorrect feature flag format")?;
        result.set(cosmic_text::FeatureTag::new(&tag), *value);
    }
    Ok(result)
}

fn face_properties(face: &fontdb::FaceInfo) -> FaceProperties {
    let style = match face.style {
        fontdb::Style::Normal => FontStyle::Normal,
        fontdb::Style::Italic => FontStyle::Italic,
        fontdb::Style::Oblique => FontStyle::Oblique,
    };
    let stretch = match face.stretch {
        fontdb::Stretch::UltraCondensed => 50.,
        fontdb::Stretch::ExtraCondensed => 62.5,
        fontdb::Stretch::Condensed => 75.,
        fontdb::Stretch::SemiCondensed => 87.5,
        fontdb::Stretch::Normal => 100.,
        fontdb::Stretch::SemiExpanded => 112.5,
        fontdb::Stretch::Expanded => 125.,
        fontdb::Stretch::ExtraExpanded => 150.,
        fontdb::Stretch::UltraExpanded => 200.,
    };
    FaceProperties { weight: FontWeight(face.weight.0.into()), style, stretch }
}
