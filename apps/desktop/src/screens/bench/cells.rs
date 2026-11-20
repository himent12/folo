//! Table cells the benchmark tables share: names with badges, numbers and bars.

use folo_ui::{
    AnyElement, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::theme::{Theme, ty};
use crate::ui::*;

/// A name, its status (the table form: glyph and plain word) and its badges on one line `width` px
/// wide. folo-ui truncates text only against a width it knows, so the name gets one: its own width,
/// or what the status and badges leave of the line. `id` keys the status mark's animation. A badge
/// with a tip says it on hover ("partial": why).
pub(super) fn name_with_badges(
    t: &Theme,
    window: &mut Window,
    name: &str,
    status: Option<(StatusKind, &str)>,
    badges: &[(&str, Tone, Option<String>)],
    width: f32,
    id: &str,
) -> folo_ui::Div {
    use crate::charts::text_width;
    // Medium weight runs a little wider than the regular face text_width measures.
    let mut measure = |s: &str| f32::from(text_width(window, s, ty::SM.size)) * 1.04;
    let status_w = status.map_or(0., |(_, w)| measure(w) + 8. + 6. + 8.);
    let badge_w: f32 = badges.iter().map(|(l, _, _)| measure(l) + 12. + 8.).sum::<f32>() + status_w;
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
        .children(status.map(|(k, w)| status_cell(t, k, w.to_string(), format!("nb-{id}")).flex_none()))
        .children(badges.iter().enumerate().map(|(i, (l, tone, tip))| {
            let b = badge(t, l.to_string(), *tone);
            match tip {
                Some(tip) => div()
                    .id(SharedString::from(format!("{id}-badge-{i}")))
                    .flex_none()
                    .tooltip(crate::controls::tip(tip.clone(), None))
                    .child(b)
                    .into_any_element(),
                None => b.into_any_element(),
            }
        }))
}

/// The "partial" badge with its reasons on hover (`partial_why`), as Studio's.
pub(super) fn partial_badge(t: &Theme, why: String, id: String) -> AnyElement {
    div()
        .id(SharedString::from(id))
        .flex_none()
        .tooltip(crate::controls::tip(why, None))
        .child(badge(t, "partial", Tone::Neutral))
        .into_any_element()
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
