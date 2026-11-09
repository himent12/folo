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

/// Where a class colour sits: on a photo, or on a panel of the app's own chrome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Photo,
    Panel,
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

impl Global for Theme {}

/// The active theme (copied out so callers don't hold a borrow of the app).
pub fn theme(cx: &App) -> Theme {
    *cx.global::<Theme>()
}

#[cfg(test)]
mod tests {
    #[test]
    fn small_text_is_annotated() {
        // design/README's text-size floor: sentences, labels, captions, legends and readouts are
        // 12 px (`sm`) or more; `xs` and `micro` are for chart ticks and image chips only. Every use
        // carries a `small:` note on its line or the one before, saying which.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src];
        let mut bad = Vec::new();
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                if p.extension().is_none_or(|x| x != "rs") || p.ends_with("theme.rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&p).unwrap();
                let lines: Vec<&str> = text.lines().collect();
                for (i, l) in lines.iter().enumerate() {
                    let noted = l.contains("small:") || i > 0 && lines[i - 1].contains("small:");
                    if (l.contains("ty::XS") || l.contains("ty::MICRO")) && !noted {
                        bad.push(format!("{}:{}", p.file_name().unwrap().to_string_lossy(), i + 1));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "ty::XS / ty::MICRO without a `small:` note at {bad:?}");
    }

    #[test]
    fn class_colours_go_through_class_on() {
        // A raw `t.class(` outside the image renderer and the swatch picks a shade by call site,
        // which is how one class came to show in two shades. Use `class_on` or `class_swatch`.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut stack = vec![src];
        let mut bad = Vec::new();
        while let Some(dir) = stack.pop() {
            for e in std::fs::read_dir(&dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                if !name.ends_with(".rs") || ["viz.rs", "theme.rs", "ui.rs"].contains(&name.as_str()) {
                    continue;
                }
                let text = std::fs::read_to_string(&p).unwrap();
                if text.contains("t.class(") {
                    bad.push(name);
                }
            }
        }
        assert!(bad.is_empty(), "raw t.class( in {bad:?}");
    }

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
    fn control_line_meets_3_to_1_on_every_surface() {
        // WCAG 1.4.11: a line that is the only mark of a control (switch track, slider rail,
        // input border) needs 3:1 against what it sits on.
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
            for (name, bg) in [
                ("app", t.bg_app),
                ("panel", t.bg_panel),
                ("raised", t.bg_raised),
                ("overlay", t.bg_overlay),
                ("inset", t.bg_inset),
            ] {
                let c = contrast(over(bg, t.line_control), bg);
                assert!(c >= 3.0, "{:?} line.control on {name}: {c:.2}:1", t.mode);
            }
        }
    }

    /// Status text (success, warning, danger, info, and accent text) is ≥ 4.5:1 (WCAG AA) on every
    /// surface it sits on, in both themes: each plain surface (inset included), those under a
    /// hover or selection, and its own tint (the muted fill of badges, banners and the Danger
    /// button) over the surfaces tints are drawn on. Inset only carries the danger tint (error
    /// rows of the run log). Every failing pair is reported, so a token change names what it broke.
    #[test]
    fn status_text_meets_aa_on_every_surface_and_tint() {
        let mut failures = Vec::new();
        for t in [Theme::new(Mode::Dark), Theme::new(Mode::Light)] {
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
                            failures.push(format!("{:?} {name} text on {on}: {c:.2}:1", t.mode));
                        }
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "status text below 4.5:1:
{}",
            failures.join(
                "
"
            )
        );
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
