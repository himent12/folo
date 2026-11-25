// Adapted from GPUI (Apache-2.0): crates/gpui/src/text_system.rs, crates/gpui/src/text_system/line_layout.rs,
// crates/gpui/src/text_system/font_features.rs, crates/gpui/src/text_system/font_fallbacks.rs,
// crates/gpui/src/style.rs (UnderlineStyle, StrikethroughStyle), crates/gpui/src/platform.rs (PlatformTextSystem)
//! Text value types (fonts, runs, shaped lines, glyph raster parameters) and the `PlatformTextSystem`
//! trait every text backend implements (`ui/DESIGN.md`, Text contract).

use crate::{Bounds, DevicePixels, GlyphBlend, Hsla, Pixels, Point, SharedString, Size, px};
use anyhow::Result;
use std::borrow::Cow;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Sub};
use std::sync::{Arc, LazyLock};

/// An opaque identifier for a specific font.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
#[repr(C)]
pub struct FontId(pub usize);

/// An opaque identifier for a specific font family.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct FontFamilyId(pub usize);

/// Horizontal subpixel positions a glyph is rasterized at (GPUI's).
pub const SUBPIXEL_VARIANTS_X: u8 = 4;

/// Vertical subpixel positions a glyph is rasterized at (GPUI's: 1 on Windows and Linux).
pub const SUBPIXEL_VARIANTS_Y: u8 =
    if cfg!(target_os = "windows") || cfg!(target_os = "linux") { 1 } else { SUBPIXEL_VARIANTS_X };

/// The degree of blackness or stroke thickness of a font. This value ranges from 100.0 to 900.0,
/// with 400.0 as normal.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct FontWeight(pub f32);

impl Display for FontWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<f32> for FontWeight {
    fn from(weight: f32) -> Self {
        FontWeight(weight)
    }
}

impl Default for FontWeight {
    #[inline]
    fn default() -> FontWeight {
        FontWeight::NORMAL
    }
}

impl Hash for FontWeight {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(u32::from_be_bytes(self.0.to_be_bytes()));
    }
}

impl Eq for FontWeight {}

impl Add for FontWeight {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        FontWeight(self.0 + rhs.0)
    }
}

impl Sub for FontWeight {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        FontWeight(self.0 - rhs.0)
    }
}

impl FontWeight {
    /// Thin weight (100), the thinnest value.
    pub const THIN: FontWeight = FontWeight(100.0);
    /// Extra light weight (200).
    pub const EXTRA_LIGHT: FontWeight = FontWeight(200.0);
    /// Light weight (300).
    pub const LIGHT: FontWeight = FontWeight(300.0);
    /// Normal (400).
    pub const NORMAL: FontWeight = FontWeight(400.0);
    /// Medium weight (500, higher than normal).
    pub const MEDIUM: FontWeight = FontWeight(500.0);
    /// Semibold weight (600).
    pub const SEMIBOLD: FontWeight = FontWeight(600.0);
    /// Bold weight (700).
    pub const BOLD: FontWeight = FontWeight(700.0);
    /// Extra-bold weight (800).
    pub const EXTRA_BOLD: FontWeight = FontWeight(800.0);
    /// Black weight (900), the thickest value.
    pub const BLACK: FontWeight = FontWeight(900.0);

    /// All of the font weights, in order from thinnest to thickest.
    pub const ALL: [FontWeight; 9] = [
        Self::THIN,
        Self::EXTRA_LIGHT,
        Self::LIGHT,
        Self::NORMAL,
        Self::MEDIUM,
        Self::SEMIBOLD,
        Self::BOLD,
        Self::EXTRA_BOLD,
        Self::BLACK,
    ];
}

/// Allows italic or oblique faces to be selected.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Default)]
pub enum FontStyle {
    /// A face that is neither italic not obliqued.
    #[default]
    Normal,
    /// A form that is generally cursive in nature.
    Italic,
    /// A typically-sloped version of the regular face.
    Oblique,
}

impl Display for FontStyle {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

static EMPTY_FONT_FEATURES: LazyLock<Arc<Vec<(String, u32)>>> = LazyLock::new(|| Arc::new(Vec::new()));
static DISABLE_LIGATURES_FEATURES: LazyLock<Arc<Vec<(String, u32)>>> =
    LazyLock::new(|| Arc::new(vec![("calt".into(), 0)]));

/// The OpenType features that can be configured for a given font.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct FontFeatures(pub Arc<Vec<(String, u32)>>);

impl Default for FontFeatures {
    #[inline]
    fn default() -> Self {
        Self(Arc::clone(&EMPTY_FONT_FEATURES))
    }
}

impl FontFeatures {
    /// Disables `calt`.
    pub fn disable_ligatures() -> Self {
        Self(Arc::clone(&DISABLE_LIGATURES_FEATURES))
    }

    /// The tag name list of the font's OpenType features (only enabled or disabled ones).
    pub fn tag_value_list(&self) -> &[(String, u32)] {
        self.0.as_slice()
    }

    /// Whether the `calt` feature is enabled (`None`: not present).
    pub fn is_calt_enabled(&self) -> Option<bool> {
        self.0.iter().find(|(feature, _)| feature == "calt").map(|(_, value)| *value == 1)
    }
}

impl Debug for FontFeatures {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("FontFeatures");
        for (tag, value) in self.tag_value_list() {
            debug.field(tag, value);
        }
        debug.finish()
    }
}

/// The fallback fonts that can be configured for a given font (family names).
#[derive(Default, Clone, Eq, PartialEq, Hash, Debug)]
pub struct FontFallbacks(pub Arc<Vec<String>>);

impl FontFallbacks {
    /// The fallback fonts' family names.
    pub fn fallback_list(&self) -> &[String] {
        self.0.as_slice()
    }

    /// Create a font fallback from a list of strings.
    pub fn from_fonts(fonts: Vec<String>) -> Self {
        FontFallbacks(Arc::new(fonts))
    }
}

/// The configuration details for identifying a specific font.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Font {
    /// The font family name.
    pub family: SharedString,
    /// The font features to use.
    pub features: FontFeatures,
    /// The fallbacks fonts to use.
    pub fallbacks: Option<FontFallbacks>,
    /// The font weight.
    pub weight: FontWeight,
    /// The font style.
    pub style: FontStyle,
}

/// Get a [`Font`] for a given name.
pub fn font(family: impl Into<SharedString>) -> Font {
    Font {
        family: family.into(),
        features: FontFeatures::default(),
        weight: FontWeight::default(),
        style: FontStyle::default(),
        fallbacks: None,
    }
}

impl Font {
    /// Set this Font to be bold.
    pub fn bold(mut self) -> Self {
        self.weight = FontWeight::BOLD;
        self
    }

    /// Set this Font to be italic.
    pub fn italic(mut self) -> Self {
        self.style = FontStyle::Italic;
        self
    }
}

/// A struct for storing font metrics, in font units (`units_per_em` per em).
#[derive(Clone, Copy, Debug)]
pub struct FontMetrics {
    /// The number of font units that make up the "em square".
    pub units_per_em: u32,
    /// The vertical distance from the baseline to the top of the tallest glyphs.
    pub ascent: f32,
    /// The distance from the baseline to the bottom of the lowest glyphs (negative, font-kit's sign).
    pub descent: f32,
    /// The recommended additional space to add between lines of type.
    pub line_gap: f32,
    /// The suggested position of the underline.
    pub underline_position: f32,
    /// The suggested thickness of the underline.
    pub underline_thickness: f32,
    /// The height of a capital letter measured from the baseline.
    pub cap_height: f32,
    /// The height of a lowercase x.
    pub x_height: f32,
    /// The outer limits of the area that the font covers.
    pub bounding_box: Bounds<f32>,
}

impl FontMetrics {
    /// Returns the vertical distance from the baseline of the font to the top of the glyph.
    pub fn ascent(&self, font_size: Pixels) -> Pixels {
        Pixels((self.ascent / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the vertical distance from the baseline of the font to the bottom of the glyph.
    pub fn descent(&self, font_size: Pixels) -> Pixels {
        Pixels((self.descent / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the recommended additional space to add between lines of type.
    pub fn line_gap(&self, font_size: Pixels) -> Pixels {
        Pixels((self.line_gap / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the suggested position of the underline.
    pub fn underline_position(&self, font_size: Pixels) -> Pixels {
        Pixels((self.underline_position / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the suggested thickness of the underline.
    pub fn underline_thickness(&self, font_size: Pixels) -> Pixels {
        Pixels((self.underline_thickness / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the height of a capital letter measured from the baseline.
    pub fn cap_height(&self, font_size: Pixels) -> Pixels {
        Pixels((self.cap_height / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the height of a lowercase x.
    pub fn x_height(&self, font_size: Pixels) -> Pixels {
        Pixels((self.x_height / self.units_per_em as f32) * font_size.0)
    }

    /// Returns the outer limits of the area that the font covers.
    pub fn bounding_box(&self, font_size: Pixels) -> Bounds<Pixels> {
        (self.bounding_box / self.units_per_em as f32 * font_size.0).map(px)
    }
}

/// The properties that can be applied to an underline.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct UnderlineStyle {
    /// The thickness of the underline.
    pub thickness: Pixels,
    /// The color of the underline.
    pub color: Option<Hsla>,
    /// Whether the underline should be wavy, like in a spell checker.
    pub wavy: bool,
}

/// The properties that can be applied to a strikethrough.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct StrikethroughStyle {
    /// The thickness of the strikethrough.
    pub thickness: Pixels,
    /// The color of the strikethrough.
    pub color: Option<Hsla>,
}

/// A styled run of text: `len` bytes in one font, colour and decoration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextRun {
    /// A number of utf8 bytes.
    pub len: usize,
    /// The font to use for this run.
    pub font: Font,
    /// The color.
    pub color: Hsla,
    /// The background color (if any).
    pub background_color: Option<Hsla>,
    /// The underline style (if any).
    pub underline: Option<UnderlineStyle>,
    /// The strikethrough style (if any).
    pub strikethrough: Option<StrikethroughStyle>,
}

/// An identifier for a specific glyph, as returned by `PlatformTextSystem::layout_line`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct GlyphId(pub u32);

/// A run of `len` utf-8 bytes in one font, as the text system hands it to `layout_line`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct FontRun {
    /// A number of utf8 bytes.
    pub len: usize,
    /// The font of these bytes.
    pub font_id: FontId,
}

/// A laid out and styled line of text.
#[derive(Default, Debug)]
pub struct LineLayout {
    /// The font size for this line.
    pub font_size: Pixels,
    /// The width of the line.
    pub width: Pixels,
    /// The ascent of the line.
    pub ascent: Pixels,
    /// The descent of the line (positive, below the baseline).
    pub descent: Pixels,
    /// The shaped runs that make up this line.
    pub runs: Vec<ShapedRun>,
    /// The length of the line in utf-8 bytes.
    pub len: usize,
}

/// A run of text that has been shaped.
#[derive(Debug, Clone)]
pub struct ShapedRun {
    /// The font id for this run.
    pub font_id: FontId,
    /// The glyphs that make up this run.
    pub glyphs: Vec<ShapedGlyph>,
}

/// A single glyph, ready to paint.
#[derive(Clone, Debug)]
pub struct ShapedGlyph {
    /// The ID for this glyph, as determined by the text system.
    pub id: GlyphId,
    /// The position of this glyph in its containing line.
    pub position: Point<Pixels>,
    /// The index of this glyph in the original text (utf-8 bytes).
    pub index: usize,
    /// Whether this glyph is an emoji.
    pub is_emoji: bool,
}

/// What to rasterize: one glyph at one size, scale and subpixel offset.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderGlyphParams {
    /// The glyph's font.
    pub font_id: FontId,
    /// The glyph.
    pub glyph_id: GlyphId,
    /// The font size in logical pixels.
    pub font_size: Pixels,
    /// The subpixel variant (`0..SUBPIXEL_VARIANTS_X`, `0..SUBPIXEL_VARIANTS_Y`).
    pub subpixel_variant: Point<u8>,
    /// Device pixels per logical pixel.
    pub scale_factor: f32,
    /// A colour emoji (rasterized as BGRA, not as an alpha mask).
    pub is_emoji: bool,
}

impl Eq for RenderGlyphParams {}

impl Hash for RenderGlyphParams {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.font_id.0.hash(state);
        self.glyph_id.0.hash(state);
        self.font_size.0.to_bits().hash(state);
        self.subpixel_variant.hash(state);
        self.scale_factor.to_bits().hash(state);
        self.is_emoji.hash(state);
    }
}

/// A text backend: fonts, metrics, shaping and glyph rasterization (DirectWrite, cosmic-text, or the
/// fixed test backend).
pub trait PlatformTextSystem: Send + Sync {
    /// Adds font files (the app bundles its own).
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()>;
    /// The family names the backend knows.
    fn all_font_names(&self) -> Vec<String>;
    /// The id of the best match for `descriptor`.
    fn font_id(&self, descriptor: &Font) -> Result<FontId>;
    /// The font's metrics, in font units.
    fn font_metrics(&self, font_id: FontId) -> FontMetrics;
    /// A glyph's typographic bounds, in font units.
    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>>;
    /// A glyph's advance, in font units.
    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>>;
    /// The glyph for `ch` in this font, if it has one.
    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId>;
    /// The device-pixel bounds of the glyph's raster, relative to its origin on the baseline.
    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>>;
    /// Rasterizes the glyph into `raster_bounds`: an alpha mask (one byte per pixel), or BGRA
    /// premultiplied for an emoji.
    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)>;
    /// Shapes one line of text (no wrapping).
    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout;
    /// The contrast / gamma correction the renderer applies to this backend's glyph masks.
    fn glyph_blend(&self) -> GlyphBlend;
}
