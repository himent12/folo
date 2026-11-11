//! Surfaces and placeholders: panels, section headers, rules, progress bars, skeletons, empty states
//! and the selection bar.

use std::time::Duration;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Animation, AnimationExt, Div, FontWeight, Hsla, IntoElement, ParentElement, SharedString, Styled, div,
    px, relative,
};

use super::format::sci;
use super::motion::reduce_motion;
use super::status::Tone;
use super::text::{icon, rich, text};
use crate::theme::{Theme, radius, ty};

/// Raised content surface with a hairline edge.
pub fn panel(t: &Theme) -> Div {
    div().bg(t.bg_panel).border_1().border_color(t.line_default).rounded(px(radius::LG))
}

/// Panel header row: overline title on the left, optional extras on the right.
/// A chart's anatomy, everywhere one shows: a header (title, caption) on a rule, a 28 px legend
/// row whose items start under the title, then the plot inset 12 px. The run page frames it as a
/// card; the Overview hero sets it under its KPIs. `short` windows trade 8 px of header and 4 of
/// legend for plot.
pub fn chart_section(
    t: &Theme,
    title: &str,
    sub: &str,
    legend: impl IntoElement,
    plot: impl IntoElement,
    short: bool,
) -> Div {
    div()
        .w_full()
        .flex()
        .flex_col()
        .min_w_0()
        .child(
            // A caption's power of ten ("× 10⁻³") in the same raised run as every value's.
            panel_header(t, title)
                .child(sci(ty::SM, t.text_tertiary, sub.to_string()).font_weight(FontWeight::MEDIUM))
                .when(short, |d| d.h(px(32.))),
        )
        .child(
            div()
                .h(px(if short { 24. } else { 28. }))
                // Legends start under the title (16 px); the loss toggles pull their 6 px hover
                // pads back out so their swatches land there too.
                .px(px(16.))
                .pt(px(if short { 4. } else { 6. }))
                .flex()
                .items_center()
                .child(legend),
        )
        .child(div().px(px(12.)).pb(px(6.)).child(plot))
}

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

pub fn vline(t: &Theme) -> Div {
    div().w(px(1.)).h_full().flex_none().bg(t.line_subtle)
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
pub fn skeleton(t: &Theme, id: impl Into<SharedString>) -> folo_ui::AnyElement {
    let base = t.bg_active;
    let id: SharedString = id.into();
    let block = div().size_full().rounded(px(radius::SM)).bg(base);
    if reduce_motion() {
        return block.into_any_element();
    }
    block
        .with_opacity_animation(
            folo_ui::ElementId::Name(format!("skel-{id}").into()),
            Animation::new(Duration::from_millis(1400)).repeat(),
            move |p| {
                let w = (p * std::f32::consts::TAU).sin() * 0.5 + 0.5;
                0.55 + 0.45 * w
            },
        )
        .into_any_element()
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
                .child(rich(t, ty::SM, t.text_secondary, body).max_w(px(400.))),
        )
}

/// The 2 px accent bar on the selected row of a box list (design/README).
pub fn select_bar(t: &Theme) -> Div {
    div().absolute().left_0().top(px(4.)).bottom(px(4.)).w(px(2.)).rounded_full().bg(t.accent_fill())
}
