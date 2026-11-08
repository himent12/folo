//! The active theme, built from `design/tokens.json` (shared with FOLO Studio).
//!
//! `build.rs` turns the token file into [`tokens`]: a `Colors` struct per theme plus the type
//! scale, spacing, radii, sizes and motion constants. This module adds the runtime pieces: the
//! current mode as a GPUI global, colour helpers and the class palette.

use gpui::{App, BoxShadow, Global, Hsla, Rgba, point, px};

#[allow(dead_code, clippy::all)]
pub mod tokens {
    use super::rgba32;
    use gpui::Hsla;
    include!(concat!(env!("OUT_DIR"), "/tokens.rs"));
}

pub use tokens::{Colors, TypeStep, radius, size, ty};

pub const SANS: &str = "Geist";
pub const MONO: &str = "Geist Mono";

/// `0xRRGGBBAA` to a colour.
pub fn rgba32(v: u32) -> Hsla {
    gpui::rgba(v).into()
}

pub fn alpha(c: Hsla, a: f32) -> Hsla {
    Hsla { a, ..c }
}

/// Composite `top` over `base` (both may be translucent), for painting token overlays such as
/// `bg_hover` where a solid colour is needed (canvas fills, rings over images).
pub fn over(base: Hsla, top: Hsla) -> Hsla {
    let (b, t): (Rgba, Rgba) = (base.into(), top.into());
    let a = t.a + b.a * (1.0 - t.a);
    if a <= 0.0 {
        return base;
    }
    let ch = |tc: f32, bc: f32| (tc * t.a + bc * b.a * (1.0 - t.a)) / a;
    Rgba { r: ch(t.r, b.r), g: ch(t.g, b.g), b: ch(t.b, b.b), a }.into()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Dark,
    Light,
}

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub mode: Mode,
    pub c: Colors,
}

impl std::ops::Deref for Theme {
    type Target = Colors;
    fn deref(&self) -> &Colors {
        &self.c
    }
}

impl Theme {
    pub fn new(mode: Mode) -> Self {
        let c = match mode {
            Mode::Dark => tokens::dark(),
            Mode::Light => tokens::light(),
        };
        Self { mode, c }
    }

    pub fn is_dark(&self) -> bool {
        self.mode == Mode::Dark
    }

    /// Near-black or white text, whichever reads better on `bg` (WCAG contrast).
    pub fn readable_on(&self, bg: Hsla) -> Hsla {
        let lum = |c: Hsla| {
            let c: Rgba = c.into();
            let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
            0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
        };
        let l = lum(bg);
        let dark = rgba32(0x0A0B0DFF);
        let light = rgba32(0xFFFFFFFF);
        if (l + 0.05) / (lum(dark) + 0.05) >= (lum(light) + 0.05) / (l + 0.05) { dark } else { light }
    }

    /// The accent for non-text fills on surfaces (progress, live dots, slider tracks). Lime on
    /// white is too pale (< 3:1), so light mode uses the olive `accent.line`.
    pub fn accent_fill(&self) -> Hsla {
        if self.is_dark() { self.accent_solid } else { self.accent_line }
    }

    /// Detection class `i` (theme independent: boxes sit on photos).
    pub fn class(&self, i: u32) -> Hsla {
        rgba32(tokens::CLASS_PALETTE[i as usize % tokens::CLASS_PALETTE.len()])
    }

    fn shadow(layers: &[(f32, f32, f32, f32, u32)]) -> Vec<BoxShadow> {
        layers
            .iter()
            .map(|&(x, y, blur, spread, c)| BoxShadow {
                color: rgba32(c),
                offset: point(px(x), px(y)),
                blur_radius: px(blur),
                spread_radius: px(spread),
            })
            .collect()
    }

    /// Elevation for menus, popovers, the palette and dialogs.
    pub fn shadow_lg(&self) -> Vec<BoxShadow> {
        Self::shadow(if self.is_dark() { tokens::shadow::DARK_LG } else { tokens::shadow::LIGHT_LG })
    }

    pub fn shadow_md(&self) -> Vec<BoxShadow> {
        Self::shadow(if self.is_dark() { tokens::shadow::DARK_MD } else { tokens::shadow::LIGHT_MD })
    }

    pub fn shadow_sm(&self) -> Vec<BoxShadow> {
        Self::shadow(if self.is_dark() { tokens::shadow::DARK_SM } else { tokens::shadow::LIGHT_SM })
    }

    /// Loss component colours (design/README.md: box = series 2, cls = 3, dfl = 4, total = 1).
    pub fn loss_colors(&self) -> [Hsla; 4] {
        let s = self.chart_series;
        [s[1], s[2], s[3], s[0]]
    }
}

impl Global for Theme {}

/// The active theme (copied out so callers don't hold a borrow of the app).
pub fn theme(cx: &App) -> Theme {
    *cx.global::<Theme>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn luminance(c: Hsla) -> f32 {
        let c: Rgba = c.into();
        let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
    }

    fn contrast(a: Hsla, b: Hsla) -> f32 {
        let (la, lb) = (luminance(a), luminance(b));
        (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
    }

    #[test]
    fn text_meets_contrast_on_every_surface() {
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
            for surface in [t.bg_app, t.bg_panel, t.bg_raised, t.bg_overlay] {
                assert!(contrast(t.text_primary, surface) >= 7.0, "{:?} primary", t.mode);
                assert!(contrast(t.text_secondary, surface) >= 4.5, "{:?} secondary", t.mode);
                assert!(contrast(t.text_tertiary, surface) >= 3.0, "{:?} tertiary", t.mode);
                assert!(contrast(t.accent_text, surface) >= 4.5, "{:?} accent text", t.mode);
            }
            assert!(contrast(t.accent_on, t.accent_solid) >= 7.0, "{:?} on-accent", t.mode);
        }
    }

    #[test]
    fn over_composites() {
        let base = rgba32(0x000000FF);
        let top = rgba32(0xFFFFFF80);
        let r: Rgba = over(base, top).into();
        assert!((r.r - 0.502).abs() < 0.01 && (r.a - 1.0).abs() < 1e-4);
    }

    #[test]
    fn tokens_are_complete() {
        assert_eq!(tokens::CLASS_PALETTE.len(), 16);
        assert_eq!(tokens::dark().chart_series.len(), 6);
    }
}
