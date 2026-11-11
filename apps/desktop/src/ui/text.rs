//! Type: text steps, overlines, captions, rich text, icons, ellipses, inline links, meta lines and
//! keyboard keys.

use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{Div, FontFeatures, FontWeight, Hsla, ParentElement, SharedString, Styled, Svg, div, px, svg};

use crate::theme::{Theme, TypeStep, radius, ty};

pub fn weight(w: u16) -> FontWeight {
    FontWeight(w as f32)
}

pub trait StyledExt: Styled + Sized {
    /// Tabular numerals, so live counters and table columns don't jitter.
    fn tnum(mut self) -> Self {
        self.text_style().get_or_insert_with(Default::default).font_features =
            Some(FontFeatures(Arc::new(vec![("tnum".into(), 1)])));
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

/// A group heading (palette groups, form sections). Sentence case: folo-ui has no letter-spacing, and
/// untracked capitals look cramped next to sentence-case titles.
pub fn overline(t: &Theme, label: &str) -> Div {
    text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).child(label.to_string())
}

/// A metric or field label: `xs` medium, tertiary, in its own case.
pub fn caption(t: &Theme, label: &str) -> Div {
    text(ty::SM, t.text_tertiary).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label.to_string())
}

/// Centred text (`.text_left()` to align it left) where `backticked` spans render as inline code:
/// one text element, so words space and wrap as body text does, with code in Geist Mono on a
/// chip fill (a no-break space either side for its padding).
pub fn rich(t: &Theme, s: TypeStep, color: Hsla, content: &str) -> Div {
    let (sans, mono) = (folo_ui::font(crate::theme::SANS), folo_ui::font(crate::theme::MONO));
    let mut text = String::new();
    let mut runs = Vec::new();
    for (i, part) in content.split('`').enumerate() {
        if part.is_empty() {
            continue;
        }
        let code = i % 2 == 1;
        // No-break spaces keep a code span together where they can, but the line wrapper can still
        // break at one, so a command that must stay whole belongs on its own chip line.
        let seg =
            if code { format!("\u{a0}{}\u{a0}", part.replace(' ', "\u{a0}")) } else { part.to_string() };
        runs.push(folo_ui::TextRun {
            len: seg.len(),
            font: if code { mono.clone() } else { sans.clone() },
            color: if code { t.text_primary } else { color },
            background_color: code.then_some(t.bg_active),
            underline: None,
            strikethrough: None,
        });
        text.push_str(&seg);
    }
    div()
        .text_size(px(s.size))
        .line_height(px(s.line))
        .text_color(color)
        .text_center()
        .child(folo_ui::StyledText::new(text).with_runs(runs))
}

pub fn icon(path: &'static str, size: f32, color: Hsla) -> Svg {
    svg().path(path).size(px(size)).flex_none().text_color(color)
}

/// An action inside a line of text ("Show them ›", "Use 0.31 (best F1) ›"): secondary ink and a
/// trailing chevron, so every inline action reads as one at rest, not only on hover.
pub fn inline_link(t: &Theme, label: impl Into<SharedString>) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(2.))
        .child(text(ty::SM, t.text_secondary).child(label.into()))
        .child(icon("icons/chevron-right.svg", 12., t.text_secondary))
}

/// Truncation that ends in "…" wherever the element sits. folo-ui's `truncate()` measures
/// no-wrap text once at its unbounded width and keeps that, so a flex-sized text clips without
/// an ellipsis; wrapping text clamped to one line re-measures at each width and truncates there.
/// Every truncated text uses this (a test keeps `.truncate()` out).
pub trait Ellipsis: Styled + Sized {
    fn ellipsis(self) -> Self {
        self.overflow_hidden().whitespace_normal().line_clamp(1).text_ellipsis()
    }
}

impl<T: Styled> Ellipsis for T {}

/// One metadata line: `lead` (an architecture, say), an optional tag right after it, then the rest
/// as " · "-joined text. The line owns its separators (4 px either side of each dot) and only the
/// rest gives way, with a real ellipsis, so every meta line spaces and truncates alike.
pub fn meta_row(
    size: TypeStep,
    color: Hsla,
    lead: impl Into<SharedString>,
    tag: Option<folo_ui::Stateful<Div>>,
    rest: &[String],
) -> Div {
    let rest: Vec<&str> = rest.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
    // The lead and its tag hold their width; everything after them is one group that gives way
    // as a whole, ending in "…", so no part can push the row past its bounds.
    div()
        .w_full()
        .flex()
        .items_center()
        .gap(px(4.))
        .min_w_0()
        .child(text(size, color).flex_none().whitespace_nowrap().child(lead.into()))
        .children(tag)
        // One separator rule: every dot is the font's " · " inside the group, the first included
        // (a flex gap beside a lone "·" spaced it wider than the rest).
        .when(!rest.is_empty(), |d| {
            d.child(
                text(size, color)
                    .flex_1()
                    .min_w_0()
                    .ml(px(-4.))
                    .ellipsis()
                    .child(format!(" · {}", rest.join(" · "))),
            )
        })
}

/// Keycap for shortcut hints.
pub fn kbd(t: &Theme, key: impl Into<SharedString>) -> Div {
    text(ty::SM, t.text_secondary)
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
