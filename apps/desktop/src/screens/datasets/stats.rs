//! A dataset's statistics: class balance and histograms, each histogram one Tab stop.

use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::health::{Finding, findings_text, health_line};
use crate::api::Dataset;
use crate::app::FoloApp;
use crate::charts::{HoverFn, histogram};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// The Statistics tab: the one-line health summary (`health`), the stat strip under it, the class
    /// balance and the two histograms.
    pub(super) fn dataset_stats(
        &mut self,
        ds: &Dataset,
        health: &[Finding],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let Some(s) = ds.stats.clone() else {
            return panel(&t)
                .child(empty_state(
                    &t,
                    "icons/chart-line.svg",
                    "No statistics",
                    "This dataset has not been scanned yet.",
                    Tone::Neutral,
                ))
                .into_any_element();
        };
        let max = s.class_counts.iter().copied().max().unwrap_or(1).max(1) as f32;
        let total: u64 = s.class_counts.iter().sum();
        let class_rows = div().flex().flex_col().gap(px(10.)).p(px(16.)).children(
            ds.classes.iter().enumerate().map(|(i, name)| {
                let c = s.class_counts.get(i).copied().unwrap_or(0);
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(
                        div()
                            .w(px(96.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(crate::ui::class_swatch(&t, i as u32, 8.))
                            .child(text(ty::SM, t.text_secondary).ellipsis().child(name.clone())),
                    )
                    .child(
                        div().flex_1().h(px(16.)).rounded(px(3.)).bg(t.bg_inset).child(
                            div()
                                .h_full()
                                .w(folo_ui::relative(c as f32 / max))
                                .rounded(px(3.))
                                .bg(t.class_on(i as u32, crate::theme::Surface::Panel)),
                        ),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().w(px(56.)).text_right().child(int(c)))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .w(px(40.))
                            .text_right()
                            .child(format!("{:.0}%", c as f32 / total.max(1) as f32 * 100.)),
                    )
            }),
        );
        let view = cx.entity().downgrade();
        let hover_fn = |key: &'static str| -> HoverFn {
            let view = view.clone();
            Rc::new(move |f, _, cx| {
                view.update(cx, |this, cx| {
                    match f {
                        Some(v) => this.hist_hover.insert(key, v),
                        None => this.hist_hover.remove(key),
                    };
                    cx.notify();
                })
                .ok();
            })
        };
        let size_labels: Vec<String> =
            s.box_size_hist.edges.windows(2).map(|w| format!("{}–{}%", pct(w[0]), pct(w[1]))).collect();
        let size_n = s.box_size_hist.counts.len();
        let per_n = s.boxes_per_image_hist.counts.len();
        let per_labels: Vec<String> =
            s.boxes_per_image_hist
                .edges
                .windows(2)
                .map(|w| {
                    if w[1] - w[0] <= 1.0 {
                        format!("{}", w[0])
                    } else {
                        format!("{}–{}", w[0], w[1] - 1.0)
                    }
                })
                .collect();
        let size_hist = histogram(
            &t,
            "hist-size",
            Arc::new(s.box_size_hist.counts.clone()),
            Arc::new(size_labels),
            t.chart_series[0],
            200.,
            self.hist_hover.get("size").copied(),
            Some(hover_fn("size")),
        );
        let per_hist = histogram(
            &t,
            "hist-per",
            Arc::new(s.boxes_per_image_hist.counts.clone()),
            Arc::new(per_labels),
            t.chart_series[0],
            200.,
            self.hist_hover.get("per").copied(),
            Some(hover_fn("per")),
        );
        // The health line is one Tab stop: its focus tip lists every finding with what it means
        // (the tooltips are the pointer's way to the same).
        let health = {
            let fh = self.fh("ds-health".to_string(), cx);
            let ring = self.ring(&fh, window);
            div()
                .id("ds-health")
                .track_focus(&fh)
                .relative()
                .when(ring, |d| {
                    d.child(crate::app::focus_ring(&t, radius::LG)).child(crate::controls::focus_tip(
                        &t,
                        findings_text(health),
                        40.,
                    ))
                })
                .child(health_line(&t, health, self.content_width(window)))
        };
        div()
            .id("ds-stats")
            .flex_1()
            .overflow_y_scroll()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(health)
                    .child(self.dataset_strip_stats(ds, window, cx))
                    .child(
                        panel(&t)
                            .child(panel_header(&t, "Class balance").child(
                                text(ty::SM, t.text_tertiary).tnum().child(counted(total, "box", "boxes")),
                            ))
                            .child(class_rows),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(px(16.))
                            .child(
                                panel(&t)
                                    .flex_1()
                                    .min_w_0()
                                    .child(panel_header(&t, "Box size").child(
                                        text(ty::SM, t.text_tertiary).child("√(box area ÷ image area)"),
                                    ))
                                    .child(self.hist_group(
                                        "hist-size-kb",
                                        "size",
                                        size_n,
                                        div().p(px(16.)).child(size_hist),
                                        window,
                                        cx,
                                    )),
                            )
                            .child(
                                panel(&t)
                                    .flex_1()
                                    .min_w_0()
                                    .child(
                                        panel_header(&t, "Boxes per image")
                                            .child(text(ty::SM, t.text_tertiary).child("images")),
                                    )
                                    .child(self.hist_group(
                                        "hist-per-kb",
                                        "per",
                                        per_n,
                                        div().p(px(16.)).child(per_hist),
                                        window,
                                        cx,
                                    )),
                            ),
                    ),
            )
            .into_any_element()
    }
}

/// A fraction as a short percentage: one decimal under 10 % when it matters ("2.5", "10").
fn pct(v: f64) -> String {
    let p = v * 100.0;
    if p >= 10.0 || (p - p.round()).abs() < 0.05 { format!("{}", p.round()) } else { format!("{p:.1}") }
}

impl FoloApp {
    /// Make a histogram one Tab stop: ←/→ select bins, Home/End jump, Esc clears; the hover readout
    /// follows the selection.
    fn hist_group(
        &self,
        key: &str,
        hover_key: &'static str,
        bins: usize,
        content: folo_ui::Div,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let n = bins.max(1);
        content
            .id(SharedString::from(key.to_string()))
            .track_focus(&fh)
            .relative()
            .on_action(cx.listener(move |this, _: &crate::app::Dismiss, _, cx| {
                if this.hist_hover.remove(hover_key).is_some() {
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let bin = this.hist_hover.get(hover_key).map(|f| ((f * n as f32) as usize).min(n - 1));
                let stride = if ev.keystroke.modifiers.shift { 10 } else { 1 };
                let next = match ev.keystroke.key.as_str() {
                    // Shift steps ten bins, as the brief's chart keys do (clamped to the ends).
                    "left" => Some(bin.map_or(0, |b| b.saturating_sub(stride))),
                    "right" => Some(bin.map_or(0, |b| (b + stride).min(n - 1))),
                    "home" => Some(0),
                    "end" => Some(n - 1),
                    "escape" if bin.is_some() => None,
                    _ => return,
                };
                match next {
                    Some(b) => this.hist_hover.insert(hover_key, (b as f32 + 0.5) / n as f32),
                    None => this.hist_hover.remove(hover_key),
                };
                cx.stop_propagation();
                cx.notify();
            }))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::LG)))
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pct_labels() {
        assert_eq!(super::pct(0.025), "2.5");
        assert_eq!(super::pct(0.1), "10");
        assert_eq!(super::pct(0.0), "0");
    }
}
