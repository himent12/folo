//! The active theme, built from `design/tokens.json` (shared with FOLO Studio; design/THEMES.md).
//!
//! `build.rs` turns the token file into [`tokens`]: a `Colors` constructor per built-in theme, the
//! `THEMES` table, setters by token path, and the type scale, spacing, radii, sizes and motion
//! constants. This module adds the runtime pieces: the active [`Theme`] as a folo-ui global, colour
//! helpers and the class palette; [`catalog`] the themes to pick from and the mode / slot rules;
//! [`custom`] the custom theme files.

use folo_ui::{App, BoxShadow, Global, Hsla, Rgba, point, px};

pub mod catalog;
pub mod custom;

pub use catalog::{Catalog, ModePref, ThemePrefs};

#[allow(dead_code, clippy::all)]
pub mod tokens {
    use super::{Mode, rgba32};
    use folo_ui::Hsla;
    include!(concat!(env!("OUT_DIR"), "/tokens.rs"));
}

pub use tokens::{Colors, TypeStep, layout, radius, size, ty};

pub const SANS: &str = "Geist";
pub const MONO: &str = "Geist Mono";

/// `0xRRGGBBAA` to a colour.
pub fn rgba32(v: u32) -> Hsla {
    folo_ui::rgba(v).into()
}

pub fn alpha(c: Hsla, a: f32) -> Hsla {
    Hsla { a, ..c }
}

/// `c` at `f` of its own opacity (for translucent tokens such as `line.*`, where [`alpha`] would
/// replace their alpha and make them opaque).
pub fn fade(c: Hsla, f: f32) -> Hsla {
    Hsla { a: c.a * f, ..c }
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

/// A resolved mode: what the app shows now (the user's choice, with System, is a [`ModePref`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Dark,
    Light,
}

impl Mode {
    /// The mode's default theme: `dark` or `light`, the first of its mode in tokens.json (build.rs
    /// checks).
    pub fn default_theme(self) -> &'static str {
        match self {
            Mode::Dark => "dark",
            Mode::Light => "light",
        }
    }

    /// `dark` or `light`.
    pub fn as_str(self) -> &'static str {
        self.default_theme()
    }
}

/// A theme id kept inline, so [`Theme`] stays `Copy`: ids are 1 to 40 of `a-z`, `0-9` and `-`
/// (design/THEMES.md).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ThemeId {
    len: u8,
    bytes: [u8; ThemeId::MAX],
}

impl ThemeId {
    pub const MAX: usize = 40;

    /// `None` unless `id` is a well-formed theme id: 1 to 40 of a-z, 0-9 and -, starting with a letter or digit
    /// (so `--theme <id>` can never read as a flag).
    pub fn new(id: &str) -> Option<Self> {
        let ok = !id.is_empty()
            && id.len() <= Self::MAX
            && !id.starts_with('-')
            && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        ok.then(|| {
            let mut bytes = [0; Self::MAX];
            bytes[..id.len()].copy_from_slice(id.as_bytes());
            Self { len: id.len() as u8, bytes }
        })
    }

    pub fn as_str(&self) -> &str {
        // Only ASCII is ever stored (`new`).
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or_default()
    }
}

impl std::fmt::Debug for ThemeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// The active theme: its mode (shadows and the few mode-dependent rules follow it), its id and its
/// palette.
#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub mode: Mode,
    pub id: ThemeId,
    pub c: Colors,
}

impl std::ops::Deref for Theme {
    type Target = Colors;
    fn deref(&self) -> &Colors {
        &self.c
    }
}

/// Where a class colour sits: on a photo, or on a panel of the app's own chrome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Photo,
    Panel,
}

impl Theme {
    /// The mode's default theme (Dark or Light).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(mode: Mode) -> Self {
        Self::builtin(mode.default_theme()).expect("tokens.json has the default themes (build.rs checks)")
    }

    /// The built-in theme `id`.
    pub fn builtin(id: &str) -> Option<Self> {
        let b = tokens::THEMES.iter().find(|b| b.id == id)?;
        Some(Self { mode: b.mode, id: ThemeId::new(b.id)?, c: (b.colors)() })
    }

    pub fn is_dark(&self) -> bool {
        self.mode == Mode::Dark
    }

    /// Class `i`'s colour for the surface it sits on: the palette as is on a photo (boxes, chips),
    /// the stroke shade on a panel (bars, curves, legends), so it keeps 3:1 there. Every class
    /// mark outside the image renderer goes through here (a test holds the line).
    pub fn class_on(&self, i: u32, surface: Surface) -> Hsla {
        match surface {
            Surface::Photo => self.class(i),
            Surface::Panel => self.class_stroke(i),
        }
    }

    /// A class colour for thin strokes on a panel: the photo palette as is in dark mode, darkened
    /// in light mode so lines keep ≥ 3:1 against white.
    pub fn class_stroke(&self, i: u32) -> Hsla {
        let c = self.class(i);
        if self.is_dark() { c } else { over(c, alpha(rgba32(0x000000FF), 0.38)) }
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

    /// Text on [`Self::accent_fill`] (the getting-started step numbers): `accent.on`, or in light mode
    /// `text.inverse` when that reads better on the olive. High contrast light darkens `accent.line`
    /// until `accent.on` holds only 3.56:1 on it, white 5.4:1; Light and Paper keep `accent.on`.
    pub fn on_accent_fill(&self) -> Hsla {
        let fill = self.accent_fill();
        if !self.is_dark() && contrast(self.text_inverse, fill) > contrast(self.accent_on, fill) {
            self.text_inverse
        } else {
            self.accent_on
        }
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

    /// Loss components (box, cls, dfl, total), per design/README: box = series 1, cls = series 4
    /// (dashed), dfl = series 3 (teal, dotted: far from box's amber), total = text.primary (thicker).
    pub fn loss_colors(&self) -> [Hsla; 4] {
        let s = self.chart_series;
        [s[1], s[4], s[3], self.text_primary]
    }
}

/// The WCAG relative luminance of `c` (alpha ignored).
pub fn luminance(c: Hsla) -> f32 {
    let c: Rgba = c.into();
    let f = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b)
}

/// The WCAG contrast ratio of two opaque colours.
pub fn contrast(a: Hsla, b: Hsla) -> f32 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

impl Global for Theme {}

/// The active theme (copied out so callers don't hold a borrow of the app).
pub fn theme(cx: &App) -> Theme {
    *cx.global::<Theme>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every built-in theme: each holds the design's pairings (design/THEMES.md).
    fn all_builtin() -> Vec<Theme> {
        tokens::THEMES.iter().map(|b| Theme::builtin(b.id).unwrap()).collect()
    }

    #[test]
    fn text_meets_contrast_on_every_surface() {
        for t in all_builtin() {
            for surface in [t.bg_app, t.bg_panel, t.bg_raised, t.bg_overlay] {
                assert!(contrast(t.text_primary, surface) >= 7.0, "{:?} primary", t.id);
                assert!(contrast(t.text_secondary, surface) >= 4.5, "{:?} secondary", t.id);
                assert!(contrast(t.text_tertiary, surface) >= 3.0, "{:?} tertiary", t.id);
                assert!(contrast(t.accent_text, surface) >= 4.5, "{:?} accent text", t.id);
            }
            assert!(contrast(t.accent_on, t.accent_solid) >= 7.0, "{:?} on-accent", t.id);
        }
    }

    #[test]
    fn control_line_meets_3_to_1_on_every_surface() {
        // WCAG 1.4.11: a line that is the only mark of a control (switch track, slider rail,
        // input border) needs 3:1 against what it sits on.
        for t in all_builtin() {
            for (name, bg) in [
                ("app", t.bg_app),
                ("panel", t.bg_panel),
                ("raised", t.bg_raised),
                ("overlay", t.bg_overlay),
                ("inset", t.bg_inset),
            ] {
                let c = contrast(over(bg, t.line_control), bg);
                assert!(c >= 3.0, "{:?} line.control on {name}: {c:.2}:1", t.id);
            }
        }
    }

    /// Status text (success, warning, danger, info, and accent text) is ≥ 4.5:1 (WCAG AA) on every
    /// surface it sits on, in every built-in theme: each plain surface (inset included), those
    /// under a hover or selection, and its own tint (the muted fill of badges, banners and the
    /// Danger button) over the surfaces tints are drawn on. Inset only carries the danger tint
    /// (error rows of the run log). Every failing pair is reported, so a token change names what it
    /// broke. design/contrast.mjs checks the same compositions (7:1 in the high contrast themes).
    #[test]
    fn status_text_meets_aa_on_every_surface_and_tint() {
        let mut failures = Vec::new();
        for t in all_builtin() {
            let surfaces = [
                ("app", t.bg_app),
                ("panel", t.bg_panel),
                ("raised", t.bg_raised),
                ("overlay", t.bg_overlay),
                ("inset", t.bg_inset),
                ("panel+hover", over(t.bg_panel, t.bg_hover)),
                ("panel+selected", over(t.bg_panel, t.bg_selected)),
            ];
            let statuses = [
                ("success", t.success_text, t.success_muted),
                ("warning", t.warning_text, t.warning_muted),
                ("danger", t.danger_text, t.danger_muted),
                ("info", t.info_text, t.info_muted),
                ("accent", t.accent_text, t.accent_muted),
            ];
            for (name, text, tint) in statuses {
                for (where_, bg) in surfaces {
                    let mut pairs = vec![(where_.to_string(), bg)];
                    if where_ != "inset" || name == "danger" {
                        pairs.push((format!("{where_}+{name} tint"), over(bg, tint)));
                    }
                    for (on, fill) in pairs {
                        let c = contrast(text, fill);
                        if c < 4.5 {
                            failures.push(format!("{:?} {name} text on {on}: {c:.2}:1", t.id));
                        }
                    }
                }
            }
        }
        assert!(failures.is_empty(), "status text below 4.5:1:\n{}", failures.join("\n"));
    }

    /// The step numbers on the accent fill read at 4.5:1 in every theme, and the default Light keeps
    /// `accent.on` (its look unchanged).
    #[test]
    fn text_on_the_accent_fill_reads_in_every_theme() {
        for t in all_builtin() {
            let c = contrast(t.on_accent_fill(), t.accent_fill());
            assert!(c >= 4.5, "{:?}: {c:.2}:1", t.id);
        }
        let light = Theme::new(Mode::Light);
        assert_eq!(light.on_accent_fill(), light.accent_on);
        let hc = Theme::builtin("contrast-light").unwrap();
        assert_eq!(hc.on_accent_fill(), hc.text_inverse);
    }

    #[test]
    fn over_composites() {
        let base = rgba32(0x000000FF);
        let top = rgba32(0xFFFFFF80);
        let r: Rgba = over(base, top).into();
        assert!((r.r - 0.502).abs() < 0.01 && (r.a - 1.0).abs() < 1e-4);
    }
}
