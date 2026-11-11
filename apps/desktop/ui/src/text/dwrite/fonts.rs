// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/direct_write.rs (font selection,
// features, fallbacks, names, metrics)
//! Font selection as GPUI resolves it (the bundled collection first, then the system collection, then
//! the system UI font), OpenType features per font, custom fallback chains, metrics and glyph queries.

use super::DirectWriteState;
use crate::{
    Bounds, Font, FontFallbacks, FontFeatures, FontId, FontMetrics, FontStyle, FontWeight, GlyphId, Point,
    SharedString, Size,
};
use anyhow::{Context as _, Result, anyhow};
use windows::Win32::Graphics::DirectWrite::*;
use windows::Win32::Graphics::Gdi::LOGFONTW;
use windows::Win32::UI::WindowsAndMessaging::{
    SPI_GETICONTITLELOGFONT, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SystemParametersInfoW,
};
use windows::core::{BOOL, HSTRING, PCWSTR, w};

/// A resolved font: its face, its features as a typography object, and its fallback chain.
pub(super) struct FontInfo {
    pub(super) font_family: String,
    pub(super) font_face: IDWriteFontFace3,
    pub(super) features: IDWriteTypography,
    pub(super) fallbacks: Option<IDWriteFontFallback>,
    pub(super) is_system_font: bool,
}

/// What identifies a face across layouts (fallback runs are mapped back to font ids by it).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(super) struct FontIdentifier {
    postscript_name: String,
    weight: i32,
    style: i32,
}

const DEFAULT_LOCALE_NAME: PCWSTR = w!("en-US");

impl DirectWriteState {
    /// The system font collection, loaded on first use (and counted).
    pub(super) fn system_font_collection(&mut self) -> Result<IDWriteFontCollection1> {
        if let Some(collection) = &self.system_font_collection {
            return Ok(collection.clone());
        }
        self.load_system_font_collection()
    }

    /// GPUI's `GetSystemFontCollection(false, …, true)` (also its `update_system_font_collection`).
    fn load_system_font_collection(&mut self) -> Result<IDWriteFontCollection1> {
        let mut collection = None;
        // SAFETY: `collection` receives an owned interface pointer.
        unsafe { self.factory.GetSystemFontCollection(false, &mut collection, true)? };
        let collection = collection.context("no system font collection")?;
        self.system_collection_loads += 1;
        self.system_font_collection = Some(collection.clone());
        Ok(collection)
    }

    /// The collection a font was selected from.
    pub(super) fn collection_of(&self, font_info: &FontInfo) -> Option<&IDWriteFontCollection1> {
        if font_info.is_system_font {
            self.system_font_collection.as_ref()
        } else {
            self.custom_font_collection.as_ref()
        }
    }

    fn system_ui_font_name(&mut self) -> SharedString {
        self.system_ui_font_name.get_or_insert_with(get_system_ui_font_name).clone()
    }

    /// GPUI's `select_font`: the family (with GPUI's aliases) from the bundled collection, then the
    /// system one; a family that is nowhere becomes the system UI font (GPUI's release behaviour).
    pub(super) fn select_font(&mut self, target_font: &Font) -> Result<FontId> {
        let family = match target_font.family.as_ref() {
            ".SystemUIFont" => self.system_ui_font_name().to_string(),
            ".ZedSans" | "Zed Plex Sans" => "IBM Plex Sans".to_string(),
            ".ZedMono" | "Zed Plex Mono" => "Lilex".to_string(),
            name => name.to_string(),
        };
        let found = self.find_font_id(
            &family,
            target_font.weight,
            target_font.style,
            &target_font.features,
            target_font.fallbacks.as_ref(),
        );
        if let Some(font_id) = found {
            return Ok(font_id);
        }
        let system_ui = self.system_ui_font_name();
        self.get_font_id_from_font_collection(
            &system_ui,
            target_font.weight,
            target_font.style,
            &target_font.features,
            target_font.fallbacks.as_ref(),
            true,
        )
        .ok_or_else(|| anyhow!("{} not found, nor the system UI font {system_ui}", target_font.family))
    }

    fn find_font_id(
        &mut self,
        family_name: &str,
        weight: FontWeight,
        style: FontStyle,
        features: &FontFeatures,
        fallbacks: Option<&FontFallbacks>,
    ) -> Option<FontId> {
        self.get_font_id_from_font_collection(family_name, weight, style, features, fallbacks, false)
            .or_else(|| {
                self.get_font_id_from_font_collection(family_name, weight, style, features, fallbacks, true)
            })
            .or_else(|| {
                // GPUI reloads the system collection once (a font installed while running).
                self.load_system_font_collection().ok()?;
                self.get_font_id_from_font_collection(family_name, weight, style, features, fallbacks, true)
            })
    }

    fn get_font_id_from_font_collection(
        &mut self,
        family_name: &str,
        font_weight: FontWeight,
        font_style: FontStyle,
        font_features: &FontFeatures,
        font_fallbacks: Option<&FontFallbacks>,
        is_system_font: bool,
    ) -> Option<FontId> {
        let collection = if is_system_font {
            self.system_font_collection().ok()?
        } else {
            self.custom_font_collection.clone()?
        };
        // SAFETY: plain DirectWrite queries on live objects.
        let fonts = unsafe {
            let fontset = collection.GetFontSet().ok()?;
            fontset
                .GetMatchingFonts(
                    &HSTRING::from(family_name),
                    dwrite_weight(font_weight),
                    DWRITE_FONT_STRETCH_NORMAL,
                    dwrite_style(font_style),
                )
                .ok()?
        };
        // SAFETY: as above.
        let total_number = unsafe { fonts.GetFontCount() };
        for index in 0..total_number {
            // SAFETY: `index` is below the count.
            let Ok(font_face_ref) = (unsafe { fonts.GetFontFaceReference(index) }) else {
                continue;
            };
            // SAFETY: a live face reference.
            let Ok(font_face) = (unsafe { font_face_ref.CreateFontFace() }) else {
                continue;
            };
            let Some(identifier) = get_font_identifier(&font_face, &self.locale) else {
                continue;
            };
            let Ok(features) = self.generate_font_features(font_features) else {
                continue;
            };
            let fallbacks =
                font_fallbacks.and_then(|fallbacks| self.generate_font_fallbacks(fallbacks).ok().flatten());
            let font_id = FontId(self.fonts.len());
            self.fonts.push(FontInfo {
                font_family: family_name.to_owned(),
                font_face,
                features,
                fallbacks,
                is_system_font,
            });
            self.font_id_by_identifier.insert(identifier, font_id);
            return Some(font_id);
        }
        None
    }

    /// GPUI's custom fallback chain: each listed family (from the system collection) for its Unicode
    /// ranges, then the system fallback.
    fn generate_font_fallbacks(&mut self, fallbacks: &FontFallbacks) -> Result<Option<IDWriteFontFallback>> {
        if fallbacks.fallback_list().is_empty() {
            return Ok(None);
        }
        let system = self.system_font_collection()?;
        // SAFETY: plain DirectWrite calls; the ranges buffer is sized from the first query.
        unsafe {
            let builder = self.factory.CreateFontFallbackBuilder()?;
            let font_set = system.GetFontSet()?;
            for family_name in fallbacks.fallback_list() {
                let Ok(fonts) = font_set.GetMatchingFonts(
                    &HSTRING::from(family_name),
                    DWRITE_FONT_WEIGHT_NORMAL,
                    DWRITE_FONT_STRETCH_NORMAL,
                    DWRITE_FONT_STYLE_NORMAL,
                ) else {
                    continue;
                };
                if fonts.GetFontCount() == 0 {
                    continue;
                }
                let font = fonts.GetFontFaceReference(0)?.CreateFontFace()?;
                let mut count = 0;
                font.GetUnicodeRanges(None, &mut count).ok();
                if count == 0 {
                    continue;
                }
                let mut unicode_ranges = vec![DWRITE_UNICODE_RANGE::default(); count as usize];
                if font.GetUnicodeRanges(Some(&mut unicode_ranges), &mut count).is_err() {
                    continue;
                }
                let target_family_name = HSTRING::from(family_name);
                builder.AddMapping(
                    &unicode_ranges,
                    &[target_family_name.as_ptr()],
                    None::<&IDWriteFontCollection>,
                    PCWSTR::null(),
                    PCWSTR::null(),
                    1.0,
                )?;
            }
            let system_fallbacks = self.factory.GetSystemFontFallback()?;
            builder.AddMappings(&system_fallbacks)?;
            Ok(Some(builder.CreateFontFallback()?))
        }
    }

    fn generate_font_features(&self, font_features: &FontFeatures) -> Result<IDWriteTypography> {
        // SAFETY: a new typography object, filled below.
        let typography = unsafe { self.factory.CreateTypography()? };
        apply_font_features(&typography, font_features)?;
        Ok(typography)
    }

    /// The family names of the system and bundled collections (loads the system collection).
    pub(super) fn all_font_names(&mut self) -> Vec<String> {
        let mut result = match self.system_font_collection() {
            Ok(collection) => get_font_names_from_collection(&collection, &self.locale),
            Err(_) => Vec::new(),
        };
        if let Some(custom) = &self.custom_font_collection {
            result.extend(get_font_names_from_collection(custom, &self.locale));
        }
        result
    }

    pub(super) fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        let font_info = &self.fonts[font_id.0];
        let mut metrics = DWRITE_FONT_METRICS1::default();
        // SAFETY: `metrics` is a valid out pointer.
        unsafe { font_info.font_face.GetMetrics(&mut metrics) };
        FontMetrics {
            units_per_em: metrics.Base.designUnitsPerEm as _,
            ascent: metrics.Base.ascent as _,
            descent: -(metrics.Base.descent as f32),
            line_gap: metrics.Base.lineGap as _,
            underline_position: metrics.Base.underlinePosition as _,
            underline_thickness: metrics.Base.underlineThickness as _,
            cap_height: metrics.Base.capHeight as _,
            x_height: metrics.Base.xHeight as _,
            bounding_box: Bounds {
                origin: Point { x: metrics.glyphBoxLeft as _, y: metrics.glyphBoxBottom as _ },
                size: Size {
                    width: (metrics.glyphBoxRight - metrics.glyphBoxLeft) as _,
                    height: (metrics.glyphBoxTop - metrics.glyphBoxBottom) as _,
                },
            },
        }
    }

    fn design_glyph_metrics(&self, font_id: FontId, glyph_id: GlyphId) -> Result<DWRITE_GLYPH_METRICS> {
        let font = &self.fonts[font_id.0].font_face;
        let glyph_indices = [glyph_id.0 as u16];
        let mut metrics = [DWRITE_GLYPH_METRICS::default()];
        // SAFETY: one index in, one metrics struct out.
        unsafe { font.GetDesignGlyphMetrics(glyph_indices.as_ptr(), 1, metrics.as_mut_ptr(), false)? };
        Ok(metrics[0])
    }

    pub(super) fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        let metrics = self.design_glyph_metrics(font_id, glyph_id)?;
        let advance_width = metrics.advanceWidth as i32;
        let advance_height = metrics.advanceHeight as i32;
        let left_side_bearing = metrics.leftSideBearing;
        let right_side_bearing = metrics.rightSideBearing;
        let top_side_bearing = metrics.topSideBearing;
        let bottom_side_bearing = metrics.bottomSideBearing;
        let vertical_origin_y = metrics.verticalOriginY;

        let y_offset = vertical_origin_y + bottom_side_bearing - advance_height;
        let width = advance_width - (left_side_bearing + right_side_bearing);
        let height = advance_height - (top_side_bearing + bottom_side_bearing);

        Ok(Bounds {
            origin: Point { x: left_side_bearing as f32, y: y_offset as f32 },
            size: Size { width: width as f32, height: height as f32 },
        })
    }

    pub(super) fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        let metrics = self.design_glyph_metrics(font_id, glyph_id)?;
        Ok(Size { width: metrics.advanceWidth as f32, height: 0.0 })
    }

    pub(super) fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        let font_info = &self.fonts[font_id.0];
        let codepoints = [ch as u32];
        let mut glyph_indices = [0u16; 1];
        // SAFETY: one code point in, one index out.
        unsafe { font_info.font_face.GetGlyphIndices(codepoints.as_ptr(), 1, glyph_indices.as_mut_ptr()) }
            .ok()
            .map(|_| GlyphId(glyph_indices[0] as u32))
    }
}

/// DirectWrite's weight for a `FontWeight` (truncated, as GPUI).
pub(super) fn dwrite_weight(weight: FontWeight) -> DWRITE_FONT_WEIGHT {
    DWRITE_FONT_WEIGHT(weight.0 as i32)
}

/// DirectWrite's style for a `FontStyle`.
pub(super) fn dwrite_style(style: FontStyle) -> DWRITE_FONT_STYLE {
    match style {
        FontStyle::Normal => DWRITE_FONT_STYLE_NORMAL,
        FontStyle::Italic => DWRITE_FONT_STYLE_ITALIC,
        FontStyle::Oblique => DWRITE_FONT_STYLE_OBLIQUE,
    }
}

fn font_style(style: DWRITE_FONT_STYLE) -> FontStyle {
    match style {
        DWRITE_FONT_STYLE_ITALIC => FontStyle::Italic,
        DWRITE_FONT_STYLE_OBLIQUE => FontStyle::Oblique,
        _ => FontStyle::Normal,
    }
}

fn get_font_names_from_collection(collection: &IDWriteFontCollection1, locale: &str) -> Vec<String> {
    let mut result = Vec::new();
    // SAFETY: indices stay below the family count.
    unsafe {
        let family_count = collection.GetFontFamilyCount();
        for index in 0..family_count {
            let Ok(font_family) = collection.GetFontFamily(index) else {
                continue;
            };
            let Ok(localized_family_name) = font_family.GetFamilyNames() else {
                continue;
            };
            let Ok(family_name) = get_name(localized_family_name, locale) else {
                continue;
            };
            result.push(family_name);
        }
    }
    result
}

/// The identifier, a `Font` describing the face (default features) and whether it is a colour font
/// (GPUI's `get_font_identifier_and_font_struct`).
pub(super) fn get_font_identifier_and_font_struct(
    font_face: &IDWriteFontFace3,
    locale: &str,
) -> Option<(FontIdentifier, Font, bool)> {
    let postscript_name = get_postscript_name(font_face, locale).ok()?;
    // SAFETY: plain queries on a live face.
    let (localized_family_name, weight, style, is_color) = unsafe {
        (
            font_face.GetFamilyNames().ok()?,
            font_face.GetWeight(),
            font_face.GetStyle(),
            font_face.IsColorFont(),
        )
    };
    let family_name = get_name(localized_family_name, locale).ok()?;
    let identifier = FontIdentifier { postscript_name, weight: weight.0, style: style.0 };
    let font_struct = Font {
        family: family_name.into(),
        features: FontFeatures::default(),
        weight: FontWeight(weight.0 as f32),
        style: font_style(style),
        fallbacks: None,
    };
    Some((identifier, font_struct, is_color.as_bool()))
}

fn get_font_identifier(font_face: &IDWriteFontFace3, locale: &str) -> Option<FontIdentifier> {
    // SAFETY: plain queries on a live face.
    let (weight, style) = unsafe { (font_face.GetWeight().0, font_face.GetStyle().0) };
    get_postscript_name(font_face, locale).ok().map(|postscript_name| FontIdentifier {
        postscript_name,
        weight,
        style,
    })
}

fn get_postscript_name(font_face: &IDWriteFontFace3, locale: &str) -> Result<String> {
    let mut info = None;
    let mut exists = BOOL(0);
    // SAFETY: both out pointers are valid.
    unsafe {
        font_face.GetInformationalStrings(
            DWRITE_INFORMATIONAL_STRING_POSTSCRIPT_NAME,
            &mut info,
            &mut exists,
        )?
    };
    match info {
        Some(info) if exists.as_bool() => get_name(info, locale),
        _ => Err(anyhow!("No postscript name found for font face")),
    }
}

/// The string for `locale`, else for en-US (GPUI's `get_name`).
fn get_name(string: IDWriteLocalizedStrings, locale: &str) -> Result<String> {
    let mut locale_name_index = 0u32;
    let mut exists = BOOL(0);
    // SAFETY: out pointers are valid; the buffer is sized from `GetStringLength`.
    unsafe {
        string.FindLocaleName(&HSTRING::from(locale), &mut locale_name_index, &mut exists)?;
        if !exists.as_bool() {
            string.FindLocaleName(DEFAULT_LOCALE_NAME, &mut locale_name_index, &mut exists)?;
            anyhow::ensure!(exists.as_bool(), "No localised string for {locale}");
        }
        let name_length = string.GetStringLength(locale_name_index)? as usize;
        let mut name_vec = vec![0u16; name_length + 1];
        string.GetString(locale_name_index, &mut name_vec)?;
        Ok(String::from_utf16_lossy(&name_vec[..name_length]))
    }
}

/// The icon title font's face name, else "Segoe UI" (GPUI's `get_system_ui_font_name`).
fn get_system_ui_font_name() -> SharedString {
    let mut info = LOGFONTW::default();
    // SAFETY: `info` is a LOGFONTW of the size passed.
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETICONTITLELOGFONT,
            std::mem::size_of::<LOGFONTW>() as u32,
            Some(&mut info as *mut _ as _),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
    };
    if ok.is_err() {
        "Segoe UI".into()
    } else {
        String::from_utf16_lossy(&info.lfFaceName).trim_matches(char::from(0)).to_owned().into()
    }
}

/// GPUI's feature handling: `liga`, `clig` and `calt` are always set (on unless turned off), every
/// other feature is added as given; nothing is set when the list is empty.
fn apply_font_features(direct_write_features: &IDWriteTypography, features: &FontFeatures) -> Result<()> {
    let tag_values = features.tag_value_list();
    if tag_values.is_empty() {
        return Ok(());
    }

    let mut feature_liga = make_direct_write_feature("liga", 1);
    let mut feature_clig = make_direct_write_feature("clig", 1);
    let mut feature_calt = make_direct_write_feature("calt", 1);

    for (tag, value) in tag_values {
        if tag.as_str() == "liga" && *value == 0 {
            feature_liga.parameter = 0;
            continue;
        }
        if tag.as_str() == "clig" && *value == 0 {
            feature_clig.parameter = 0;
            continue;
        }
        if tag.as_str() == "calt" && *value == 0 {
            feature_calt.parameter = 0;
            continue;
        }
        // SAFETY: a plain feature added to a live typography object.
        unsafe { direct_write_features.AddFontFeature(make_direct_write_feature(tag, *value))? };
    }
    // SAFETY: as above.
    unsafe {
        direct_write_features.AddFontFeature(feature_liga)?;
        direct_write_features.AddFontFeature(feature_clig)?;
        direct_write_features.AddFontFeature(feature_calt)?;
    }
    Ok(())
}

fn make_direct_write_feature(feature_name: &str, parameter: u32) -> DWRITE_FONT_FEATURE {
    let bytes = feature_name.as_bytes();
    let tag = match bytes {
        [a, b, c, d] => u32::from_le_bytes([*a, *b, *c, *d]),
        // GPUI asserts four bytes in debug builds; pad a shorter tag with spaces as OpenType does.
        _ => {
            let mut tag = [b' '; 4];
            for (slot, byte) in tag.iter_mut().zip(bytes) {
                *slot = *byte;
            }
            u32::from_le_bytes(tag)
        }
    };
    DWRITE_FONT_FEATURE { nameTag: DWRITE_FONT_FEATURE_TAG(tag), parameter }
}
