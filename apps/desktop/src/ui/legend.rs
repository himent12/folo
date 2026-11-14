//! Legend swatches: line samples, the best ring, colour squares and class colours.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{Div, Hsla, ParentElement, Styled, div, px};

use super::format::sci;
use super::text::{StyledExt, text};
use crate::theme::{Theme, ty};

/// One legend entry with its value: swatch, name, number (tabular).
/// A 10 px line sample matching a series stroke: solid, dashed (two dashes) or dotted (three dots).
pub fn line_swatch(color: Hsla, dashed: bool, dotted: bool, weight: f32) -> Div {
    if dotted {
        div()
            .w(px(10.))
            .flex()
            .gap(px(2.))
            .children((0..3).map(move |_| div().w(px(2.)).h(px(2.)).rounded_full().bg(color)))
    } else if dashed {
        div().w(px(10.)).flex().gap(px(2.)).children((0..2).map(move |_| div().w(px(4.)).h(px(2.)).bg(color)))
    } else {
        div().w(px(10.)).h(px(weight)).rounded_full().bg(color)
    }
}

pub fn legend_value(t: &Theme, name: &str, color: Hsla, dashed: bool, value: Option<String>) -> Div {
    let swatch = line_swatch(color, dashed, false, 2.);
    div()
        .flex()
        .items_center()
        .gap(px(5.))
        .child(swatch)
        .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(name.to_string()))
        .when_some(value, |d, v| d.child(sci(ty::SM, t.text_primary, v)))
}

/// The best point's key, "○ best 0.656 @ 41": the chart's hollow ink ring, then the value.
pub fn best_legend(t: &Theme, value: String) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(5.))
        .child(div().size(px(9.)).flex_none().rounded_full().border(px(1.5)).border_color(t.text_primary))
        .child(text(ty::SM, t.text_secondary).child("best"))
        .child(text(ty::SM, t.text_primary).tnum().child(value))
}

/// A class swatch: a small square in the class colour.
pub fn swatch(color: Hsla, size: f32) -> Div {
    div().size(px(size)).flex_none().rounded(px(2.)).bg(color)
}

/// A class swatch: the hue its boxes have on the image, so the key matches the photo. Its 1 px
/// edge is `chart.swatch_edge` in dark; in light, the class's own panel shade (3:1 on white), so
/// the swatch also matches that class's bars and curves beside it.
pub fn class_swatch(t: &Theme, class: u32, size: f32) -> Div {
    use crate::theme::Surface;
    let edge = if t.is_dark() { t.chart_swatch_edge } else { t.class_on(class, Surface::Panel) };
    swatch(t.class_on(class, Surface::Photo), size).border_1().border_color(edge)
}
