//! Stateless building blocks: type, surfaces, badges, metrics, progress, placeholders.
//!
//! Interactive controls that need focus handles live on the app (`controls.rs`); everything
//! here is a pure function of the theme and its arguments.

use std::sync::Arc;
use std::time::Duration;

use gpui::prelude::FluentBuilder;
use gpui::{
    Animation, AnimationExt, Div, FontFeatures, FontWeight, Hsla, IntoElement, ParentElement,
    SharedString, Styled, Svg, div, px, relative, svg,
};

use crate::api::RunStatus;
use crate::theme::{Theme, TypeStep, alpha, radius, ty};

pub fn weight(w: u16) -> FontWeight {
    FontWeight(w as f32)
}

pub trait StyledExt: Styled + Sized {
    /// Tabular numerals, so live counters and table columns don't jitter.
    fn tnum(mut self) -> Self {
        self.text_style()
            .get_or_insert_with(Default::default)
            .font_features = Some(FontFeatures(Arc::new(vec![("tnum".into(), 1)])));
        self
    }

    fn step(self, s: TypeStep) -> Self {
        self.text_size(px(s.size)).line_height(px(s.line)).font_weight(weight(s.weight))
    }
}

impl<T: Styled> StyledExt for T {}

/// Sans text at a type step.
pub fn text(s: TypeStep, color: Hsla) -> Div {
    div().font_family(crate::theme::SANS).step(s).text_color(color)
}

/// Mono text at a type step (IDs, paths, logs, code, numbers in tables).
pub fn mono(s: TypeStep, color: Hsla) -> Div {
    div().font_family(crate::theme::MONO).step(s).text_color(color).tnum()
}

/// Uppercase overline (`micro`).
pub fn overline(t: &Theme, label: &str) -> Div {
    text(ty::MICRO, t.text_tertiary).child(label.to_uppercase())
}

pub fn icon(path: &'static str, size: f32, color: Hsla) -> Svg {
    svg().path(path).size(px(size)).flex_none().text_color(color)
}

/// Raised content surface with a hairline edge.
pub fn panel(t: &Theme) -> Div {
    div().bg(t.bg_panel).border_1().border_color(t.line_default).rounded(px(radius::LG))
}

/// Panel header row: overline title on the left, optional extras on the right.
pub fn panel_header(t: &Theme, title: &str) -> Div {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(px(8.))
        .h(px(40.))
        .px(px(16.))
        .border_b_1()
        .border_color(t.line_subtle)
        .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()))
}

pub fn hairline(t: &Theme) -> Div {
    div().h(px(1.)).w_full().flex_none().bg(t.line_subtle)
}

pub fn vline(t: &Theme) -> Div {
    div().w(px(1.)).h_full().flex_none().bg(t.line_subtle)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
    Info,
}

impl Tone {
    pub fn colors(self, t: &Theme) -> (Hsla, Hsla, Hsla) {
        // (background, text, dot)
        match self {
            Tone::Neutral => (t.bg_active, t.text_secondary, t.text_tertiary),
            Tone::Accent => (t.accent_muted, t.accent_text, t.accent_solid),
            Tone::Success => (t.success_muted, t.success_text, t.success_solid),
            Tone::Warning => (t.warning_muted, t.warning_text, t.warning_solid),
            Tone::Danger => (t.danger_muted, t.danger_text, t.danger_solid),
            Tone::Info => (t.info_muted, t.info_text, t.info_solid),
        }
    }
}

pub fn status_tone(s: RunStatus) -> Tone {
    match s {
        RunStatus::Running => Tone::Accent,
        RunStatus::Completed => Tone::Success,
        RunStatus::Failed => Tone::Danger,
        RunStatus::Stopping => Tone::Warning,
        RunStatus::Queued => Tone::Info,
        RunStatus::Stopped => Tone::Neutral,
    }
}

/// A small status dot; `live` pulses a halo around it.
pub fn dot(color: Hsla, size: f32, live: bool, id: impl Into<SharedString>) -> impl IntoElement {
    let id: SharedString = id.into();
    div()
        .relative()
        .size(px(size))
        .flex_none()
        .child(div().absolute().size_full().rounded_full().bg(color))
        .when(live, |d| {
            d.child(
                div()
                    .absolute()
                    .size_full()
                    .rounded_full()
                    .bg(color)
                    .with_animation(
                        gpui::ElementId::Name(format!("pulse-{id}").into()),
                        Animation::new(Duration::from_millis(1600)).repeat(),
                        move |d, p| {
                            let grow = size * 1.6 * p;
                            d.top(px(-grow / 2.))
                                .left(px(-grow / 2.))
                                .size(px(size + grow))
                                .opacity(0.45 * (1.0 - p))
                        },
                    ),
            )
        })
}

/// Tinted label chip, 20 px tall.
pub fn badge(t: &Theme, label: impl Into<SharedString>, tone: Tone) -> Div {
    let (bg, fg, _) = tone.colors(t);
    text(ty::XS, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .px(px(6.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(5.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(label.into())
}

/// Status chip for a run: a dot (pulsing while running) and the status word.
pub fn status_badge(t: &Theme, s: RunStatus, id: &str) -> Div {
    let tone = status_tone(s);
    let (_, _, dotc) = tone.colors(t);
    let (bg, fg, _) = tone.colors(t);
    text(ty::XS, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .pl(px(6.))
        .pr(px(7.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(6.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(dot(dotc, 6., s == RunStatus::Running, format!("badge-{id}")))
        .child(s.label())
}

/// Keycap for shortcut hints.
pub fn kbd(t: &Theme, key: impl Into<SharedString>) -> Div {
    text(ty::XS, t.text_secondary)
        .font_weight(FontWeight::MEDIUM)
        .h(px(18.))
        .min_w(px(18.))
        .px(px(4.))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .rounded(px(radius::SM))
        .bg(t.bg_active)
        .border_1()
        .border_color(t.line_subtle)
        .child(key.into())
}

/// A shortcut as keycaps ("Ctrl", "K").
pub fn shortcut(t: &Theme, keys: &[&str]) -> Div {
    div().flex().items_center().gap(px(3.)).children(keys.iter().map(|k| kbd(t, k.to_string())))
}

/// A number with its unit one step smaller in tertiary ("0.612", "ms").
pub fn metric(t: &Theme, value: impl Into<SharedString>, unit: &str, s: TypeStep, unit_step: TypeStep) -> Div {
    div()
        .flex()
        .items_baseline()
        .gap(px(3.))
        .whitespace_nowrap()
        .child(text(s, t.text_primary).tnum().child(value.into()))
        .when(!unit.is_empty(), |d| d.child(text(unit_step, t.text_tertiary).tnum().child(unit.to_string())))
}

/// Labelled metric tile body: overline label, value, optional detail line.
pub fn stat(t: &Theme, label: &str, value: impl Into<SharedString>, unit: &str, detail: Option<Div>) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(6.))
        .min_w_0()
        .child(overline(t, label))
        .child(metric(t, value, unit, ty::X2XL, ty::SM))
        .children(detail)
}

/// A thin progress track with a filled part.
pub fn progress(t: &Theme, frac: f32, color: Hsla, height: f32) -> Div {
    div()
        .w_full()
        .h(px(height))
        .rounded(px(height / 2.))
        .bg(t.bg_active)
        .overflow_hidden()
        .child(div().h_full().w(relative(frac.clamp(0.0, 1.0))).rounded(px(height / 2.)).bg(color))
}

/// A placeholder block that shimmers while data loads.
pub fn skeleton(t: &Theme, id: impl Into<SharedString>) -> impl IntoElement {
    let base = t.bg_active;
    let id: SharedString = id.into();
    div().rounded(px(radius::SM)).bg(base).with_animation(
        gpui::ElementId::Name(format!("skel-{id}").into()),
        Animation::new(Duration::from_millis(1400)).repeat(),
        move |d, p| {
            let w = (p * std::f32::consts::TAU).sin() * 0.5 + 0.5;
            d.opacity(0.55 + 0.45 * w)
        },
    )
}

/// Centered empty / error message with an icon, a title, a line of help and optional actions.
pub fn empty_state(t: &Theme, icon_path: &'static str, title: &str, body: &str, tone: Tone) -> Div {
    let (bg, _, dotc) = tone.colors(t);
    div()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap(px(12.))
        .py(px(40.))
        .px(px(24.))
        .child(
            div()
                .size(px(40.))
                .rounded(px(radius::LG))
                .bg(if tone == Tone::Neutral { t.bg_raised } else { bg })
                .border_1()
                .border_color(t.line_subtle)
                .flex()
                .items_center()
                .justify_center()
                .child(icon(icon_path, 18., if tone == Tone::Neutral { t.text_secondary } else { dotc })),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .items_center()
                .gap(px(4.))
                .child(text(ty::MD, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()))
                .child(text(ty::SM, t.text_secondary).max_w(px(360.)).text_center().child(body.to_string())),
        )
}

/// A class swatch: a small square in the class colour.
pub fn swatch(color: Hsla, size: f32) -> Div {
    div().size(px(size)).flex_none().rounded(px(2.)).bg(color)
}

/// A row of key/value pairs for detail panes.
pub fn kv(t: &Theme, key: &str, value: impl IntoElement) -> Div {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap(px(12.))
        .min_h(px(28.))
        .child(text(ty::SM, t.text_tertiary).flex_none().child(key.to_string()))
        .child(div().min_w_0().flex().justify_end().child(value))
}

/// Soft tinted backdrop for a selected/active row.
pub fn selected_bg(t: &Theme) -> Hsla {
    t.bg_selected
}

pub fn tint(c: Hsla, a: f32) -> Hsla {
    alpha(c, a)
}

// ---------- number formatting ----------

/// 4210 → "4,210".
pub fn int(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// 5_400_000 → "5.4 MB".
pub fn bytes(n: u64) -> (String, &'static str) {
    let f = n as f64;
    if f >= 1e9 {
        (format!("{:.1}", f / 1e9), "GB")
    } else if f >= 1e6 {
        (format!("{:.1}", f / 1e6), "MB")
    } else if f >= 1e3 {
        (format!("{:.0}", f / 1e3), "KB")
    } else {
        (n.to_string(), "B")
    }
}

/// 2_630_000 → "2.63 M".
pub fn count_si(n: u64) -> (String, &'static str) {
    let f = n as f64;
    if f >= 1e9 {
        (format!("{:.2}", f / 1e9), "B")
    } else if f >= 1e6 {
        (format!("{:.2}", f / 1e6), "M")
    } else if f >= 1e3 {
        (format!("{:.1}", f / 1e3), "K")
    } else {
        (n.to_string(), "")
    }
}

pub fn f3(v: f32) -> String {
    format!("{v:.3}")
}

pub fn pct(v: f32) -> String {
    format!("{:.0}", v * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(int(4210), "4,210");
        assert_eq!(int(999), "999");
        assert_eq!(int(1_000_000), "1,000,000");
        assert_eq!(bytes(5_400_000), ("5.4".into(), "MB"));
        assert_eq!(count_si(2_630_000), ("2.63".into(), "M"));
    }
}
