// Adapted from GPUI (Apache-2.0): crates/gpui/src/style.rs
//! Text styles: `TextStyle` and its refinement, `HighlightStyle`, white space, overflow and alignment.

use super::refine::{full_ty, refine_full, refine_part, refineable, refinement_ty};
use crate::{
    AbsoluteLength, DefiniteLength, Font, FontFallbacks, FontFeatures, FontStyle, FontWeight, Hsla, Pixels,
    Rgba, SharedString, StrikethroughStyle, TextRun, UnderlineStyle, black, phi, rems,
};
use std::hash::{Hash, Hasher};

/// How to handle whitespace in text.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WhiteSpace {
    /// Normal line wrapping when text overflows the width of the element.
    #[default]
    Normal,
    /// No line wrapping: text overflows the width of the element.
    Nowrap,
}

/// How to truncate text that overflows the width of the element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextOverflow {
    /// Truncate the text when it doesn't fit, and show the given string in its place.
    Truncate(SharedString),
}

pub use crate::text::TextAlign;

refineable! {
    /// The properties that style text (they cascade from an element to its children).
    #[derive(Clone, Debug, PartialEq)]
    pub struct TextStyle / #[derive(Clone, Debug, Default, PartialEq)] TextStyleRefinement {
        /// The color of the text.
        plain color: Hsla,
        /// The font family.
        plain font_family: SharedString,
        /// The font features.
        plain font_features: FontFeatures,
        /// The fallback fonts.
        opt font_fallbacks: FontFallbacks,
        /// The font size, in pixels or rems.
        plain font_size: AbsoluteLength,
        /// The line height, in pixels or a fraction of the font size.
        plain line_height: DefiniteLength,
        /// The font weight.
        plain font_weight: FontWeight,
        /// The font style.
        plain font_style: FontStyle,
        /// The background color of the text.
        opt background_color: Hsla,
        /// The underline.
        opt underline: UnderlineStyle,
        /// The strikethrough.
        opt strikethrough: StrikethroughStyle,
        /// How to handle whitespace.
        plain white_space: WhiteSpace,
        /// Truncate the text if it overflows the element.
        opt text_overflow: TextOverflow,
        /// How the text is aligned within the element.
        plain text_align: TextAlign,
        /// The number of lines to show before truncating.
        opt line_clamp: usize,
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        TextStyle {
            color: black(),
            font_family: ".SystemUIFont".into(),
            font_features: FontFeatures::default(),
            font_fallbacks: None,
            font_size: rems(1.).into(),
            line_height: phi(),
            font_weight: FontWeight::default(),
            font_style: FontStyle::default(),
            background_color: None,
            underline: None,
            strikethrough: None,
            white_space: WhiteSpace::Normal,
            text_overflow: None,
            text_align: TextAlign::default(),
            line_clamp: None,
        }
    }
}

impl TextStyle {
    /// This style with `style` applied: colors blend, other fields are replaced.
    pub fn highlight(mut self, style: impl Into<HighlightStyle>) -> Self {
        let style = style.into();
        if let Some(weight) = style.font_weight {
            self.font_weight = weight;
        }
        if let Some(style) = style.font_style {
            self.font_style = style;
        }
        if let Some(color) = style.color {
            self.color = self.color.blend(color);
        }
        if let Some(factor) = style.fade_out {
            self.color.fade_out(factor);
        }
        if let Some(background_color) = style.background_color {
            self.background_color = Some(background_color);
        }
        if let Some(underline) = style.underline {
            self.underline = Some(underline);
        }
        if let Some(strikethrough) = style.strikethrough {
            self.strikethrough = Some(strikethrough);
        }
        self
    }

    /// The font this style selects.
    pub fn font(&self) -> Font {
        Font {
            family: self.font_family.clone(),
            features: self.font_features.clone(),
            fallbacks: self.font_fallbacks.clone(),
            weight: self.font_weight,
            style: self.font_style,
        }
    }

    /// Borrowed key view of the font this style selects (zero clones).
    pub(crate) fn font_key_ref(&self) -> crate::text::FontKeyRef<'_> {
        crate::text::FontKeyRef {
            family: self.font_family.as_str(),
            features: &self.font_features,
            fallbacks: self.font_fallbacks.as_ref(),
            weight: self.font_weight,
            style: self.font_style,
        }
    }

    /// The line height in pixels, rounded (GPUI rounds it to whole logical pixels).
    pub fn line_height_in_pixels(&self, rem_size: Pixels) -> Pixels {
        self.line_height.to_pixels(self.font_size, rem_size).round()
    }

    /// A [`TextRun`] of `len` bytes in this style.
    pub fn to_run(&self, len: usize) -> TextRun {
        TextRun {
            len,
            font: self.font(),
            color: self.color,
            background_color: self.background_color,
            underline: self.underline,
            strikethrough: self.strikethrough,
        }
    }
}

/// A highlight to apply over a text style: one font, uniformly sized and spaced text.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HighlightStyle {
    /// The color of the text.
    pub color: Option<Hsla>,
    /// The font weight.
    pub font_weight: Option<FontWeight>,
    /// The font style.
    pub font_style: Option<FontStyle>,
    /// The background color of the text.
    pub background_color: Option<Hsla>,
    /// The underline.
    pub underline: Option<UnderlineStyle>,
    /// The strikethrough.
    pub strikethrough: Option<StrikethroughStyle>,
    /// Makes the text less vibrant, like CSS `opacity`.
    pub fade_out: Option<f32>,
}

impl Eq for HighlightStyle {}

impl Hash for HighlightStyle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color.hash(state);
        self.font_weight.hash(state);
        self.font_style.hash(state);
        self.background_color.hash(state);
        self.underline.hash(state);
        self.strikethrough.hash(state);
        state.write_u32(u32::from_be_bytes(self.fade_out.map(|f| f.to_be_bytes()).unwrap_or_default()));
    }
}

impl HighlightStyle {
    /// A highlight that only sets the color.
    pub fn color(color: Hsla) -> Self {
        Self { color: Some(color), ..Default::default() }
    }

    /// This highlight blended with `other`: colors blend and fades compound, other fields are replaced.
    #[must_use]
    pub fn highlight(self, other: HighlightStyle) -> Self {
        Self {
            color: other
                .color
                .map(
                    |other_color| {
                        if let Some(color) = self.color { color.blend(other_color) } else { other_color }
                    },
                )
                .or(self.color),
            font_weight: other.font_weight.or(self.font_weight),
            font_style: other.font_style.or(self.font_style),
            background_color: other.background_color.or(self.background_color),
            underline: other.underline.or(self.underline),
            strikethrough: other.strikethrough.or(self.strikethrough),
            fade_out: other
                .fade_out
                .map(|source_fade| {
                    self.fade_out
                        .map(|dest_fade| (dest_fade * (1. + source_fade)).clamp(0., 1.))
                        .unwrap_or(source_fade)
                })
                .or(self.fade_out),
        }
    }
}

impl From<TextStyle> for HighlightStyle {
    fn from(other: TextStyle) -> Self {
        Self::from(&other)
    }
}

impl From<&TextStyle> for HighlightStyle {
    fn from(other: &TextStyle) -> Self {
        Self {
            color: Some(other.color),
            font_weight: Some(other.font_weight),
            font_style: Some(other.font_style),
            background_color: other.background_color,
            underline: other.underline,
            strikethrough: other.strikethrough,
            fade_out: None,
        }
    }
}

impl From<Hsla> for HighlightStyle {
    fn from(color: Hsla) -> Self {
        Self { color: Some(color), ..Default::default() }
    }
}

impl From<FontWeight> for HighlightStyle {
    fn from(font_weight: FontWeight) -> Self {
        Self { font_weight: Some(font_weight), ..Default::default() }
    }
}

impl From<FontStyle> for HighlightStyle {
    fn from(font_style: FontStyle) -> Self {
        Self { font_style: Some(font_style), ..Default::default() }
    }
}

impl From<Rgba> for HighlightStyle {
    fn from(color: Rgba) -> Self {
        Self { color: Some(color.into()), ..Default::default() }
    }
}
