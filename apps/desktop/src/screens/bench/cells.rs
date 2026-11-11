//! Table cells the benchmark tables share: names with badges, numbers and bars.

use folo_ui::{FontWeight, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use crate::theme::{Theme, ty};
use crate::ui::*;

/// A name and its badges on one line `width` px wide. GPUI truncates text only against a width it
/// knows, so the name gets one: its own width, or what the badges leave of the line.
pub(super) fn name_with_badges(
    t: &Theme,
    window: &mut Window,
    name: &str,
    badges: &[(&str, Tone)],
    width: f32,
) -> folo_ui::Div {
    use crate::charts::text_width;
    // Medium weight runs a little wider than the regular face text_width measures.
    let mut measure = |s: &str| f32::from(text_width(window, s, ty::SM.size)) * 1.04;
    let badge_w: f32 = badges.iter().map(|(l, _)| measure(l) + 12. + 8.).sum();
    let name_w = (measure(name) + 2.).min((width - badge_w).max(48.));
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .w(px(width))
        .child(
            text(ty::SM, t.text_primary)
                .font_weight(FontWeight::MEDIUM)
                .w(px(name_w))
                .ellipsis()
                .child(name.to_string()),
        )
        .children(badges.iter().map(|(l, tone)| badge(t, l.to_string(), *tone)))
}

pub(super) fn num_cell(w: f32, content: impl IntoElement) -> folo_ui::Div {
    div().w(px(w)).flex_none().flex().justify_end().child(content)
}

/// A labelled bar: a 0..=1 value as a filled track with its text on the right.
pub(super) fn bar_row(
    t: &Theme,
    label: impl Into<SharedString>,
    frac: f32,
    value: String,
    color: folo_ui::Hsla,
) -> folo_ui::Div {
    div()
        .flex()
        .items_center()
        .gap(px(12.))
        .h(px(28.))
        .child(text(ty::SM, t.text_secondary).w(px(140.)).flex_none().ellipsis().child(label.into()))
        .child(div().flex_1().min_w_0().child(progress(t, frac, color, 6.)))
        .child(text(ty::SM, t.text_primary).tnum().w(px(56.)).flex_none().flex().justify_end().child(value))
}
