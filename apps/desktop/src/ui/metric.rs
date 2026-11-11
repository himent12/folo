//! Metrics: value pairs, KPI values and stats, durations and key/value rows.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{Div, IntoElement, ParentElement, SharedString, Styled, div, px};

use super::text::{StyledExt, caption, text};
use crate::theme::{Theme, TypeStep, ty};

/// Two values as one figure, "0.861 / 0.792": both at `size`, the slash a step down in tertiary
/// ink. The run and model pages' Precision / Recall both use it.
pub fn metric_pair(t: &Theme, size: TypeStep, a: String, b: String) -> Div {
    div()
        .flex()
        .items_baseline()
        .gap(px(4.))
        .child(text(size, t.text_primary).tnum().child(a))
        .child(text(ty::LG, t.text_tertiary).child("/"))
        .child(text(size, t.text_primary).tnum().child(b))
}

/// A number with its unit one step smaller in tertiary ("0.612", "ms").
pub fn metric(
    t: &Theme,
    value: impl Into<SharedString>,
    unit: &str,
    s: TypeStep,
    unit_step: TypeStep,
) -> Div {
    // GPUI's flex baseline doesn't align mixed sizes, so bottom-align and lift the unit by the
    // difference between the two lines' baselines. A space's width apart ("9.42 M", as text
    // writes a number and its unit everywhere).
    div()
        .flex()
        .items_end()
        .gap(px(4.))
        .whitespace_nowrap()
        .child(text(s, t.text_primary).tnum().child(value.into()))
        .when(!unit.is_empty(), |d| {
            d.child(
                text(unit_step, t.text_tertiary)
                    .tnum()
                    .pb(px(baseline_lift(s, unit_step)))
                    .child(unit.to_string()),
            )
        })
}

/// Labelled metric tile body: overline label, value, optional detail line.
/// A stat at a given value size (no unit).
pub fn stat_sized(
    t: &Theme,
    label: &str,
    value: impl Into<SharedString>,
    size: TypeStep,
    detail: Option<Div>,
) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(6.))
        .min_w_0()
        .child(caption(t, label))
        .child(metric(t, value, "", size, ty::SM))
        .children(detail)
}

/// A labelled stat with its value at `step` (X2XL; compact strips on short windows use XL).
pub fn stat_at(
    t: &Theme,
    label: &str,
    value: impl Into<SharedString>,
    unit: &str,
    detail: Option<Div>,
    step: TypeStep,
) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(px(if step.size < ty::X2XL.size { 4. } else { 6. }))
        .min_w_0()
        .child(caption(t, label))
        .child(metric(t, value, unit, step, ty::SM))
        .children(detail)
}

/// A KPI tile's first detail line, at the caption line's height, for a graphic (a progress or
/// split bar) that stands in for text, so its tile's later lines stay on the strip's grid.
pub fn detail_line(content: impl IntoElement) -> Div {
    div().h(px(ty::SM.line)).w_full().flex().items_center().child(content)
}

/// A duration as a metric: numbers at value size, units ("h", "m", "s") one step smaller.
/// How far above the bottom of its line box a smaller step's baseline must move to meet a larger
/// step's baseline (both bottom-aligned): baseline ≈ half the leading plus 0.2 em of descent.
pub fn baseline_lift(big: TypeStep, small: TypeStep) -> f32 {
    let from_bottom = |s: TypeStep| (s.line - s.size) / 2. + 0.2 * s.size;
    (from_bottom(big) - from_bottom(small)).max(0.)
}

pub fn duration_metric(t: &Theme, secs: Option<f64>, s: TypeStep, unit_step: TypeStep) -> Div {
    let Some(secs) = secs else { return text(s, t.text_primary).child("–") };
    let total = secs.max(0.0).round() as i64;
    let parts: Vec<(String, &str)> = if total < 60 {
        vec![(total.to_string(), "s")]
    } else if total < 3600 {
        vec![((total / 60).to_string(), "m"), (format!("{:02}", total % 60), "s")]
    } else {
        vec![((total / 3600).to_string(), "h"), (format!("{:02}", total / 60 % 60), "m")]
    };
    // "14m 55s": each unit hugs its number; the pair is one reading.
    div().flex().items_end().gap(px(5.)).whitespace_nowrap().children(parts.into_iter().map(|(v, u)| {
        div()
            .flex()
            .items_end()
            .gap(px(1.))
            .child(text(s, t.text_primary).tnum().child(v))
            .child(text(unit_step, t.text_tertiary).pb(px(baseline_lift(s, unit_step))).child(u))
    }))
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
