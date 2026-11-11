//! The precision-recall panel: per-class curves, their line patterns and the isolated class.

use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, ModelEval};
use crate::app::{FoloApp, act};
use crate::charts::{HoverFn, LineChart, Series, line_chart};
use crate::screens::th;
use crate::theme::{alpha, radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// The class isolated on the PR chart: the hovered one, else the focused one, else the pin.
    pub(super) fn pr_isolated(&self, ev: &ModelEval, window: &Window) -> Option<u32> {
        let focused = ev.pr.iter().map(|c| c.class).find(|c| {
            [format!("pr-chip-{c}"), format!("pc-row-{c}")]
                .iter()
                .any(|k| self.focus.borrow().get(k.as_str()).is_some_and(|f| f.is_focused(window)))
        });
        // In keyboard mode the focused row wins over a hover left behind by the pointer (as in
        // Playground and the inspector); moving the pointer hands it back. A pin from another
        // evaluation never applies here.
        let lead = if self.kb { focused.or(self.pr_focus) } else { self.pr_focus.or(focused) };
        lead.or(self.pr_pin).filter(|c| ev.pr.iter().any(|p| p.class == *c))
    }

    pub(super) fn pr_panel(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let iso = self.pr_isolated(ev, window);
        // End each curve at its highest recall instead of a drop to zero.
        let curves: Vec<(u32, crate::charts::Points)> = ev
            .pr
            .iter()
            .map(|c| {
                let mut pts: Vec<(f32, f32)> =
                    c.recall.iter().zip(&c.precision).map(|(r, p)| (*r, *p)).collect();
                if let Some(last) = pts.iter().rposition(|p| p.1 > 0.0) {
                    pts.truncate(last + 1);
                }
                (c.class, Arc::new(pts))
            })
            .collect();
        let hover = self.pr_hover;
        // Precision at the inspected recall, only where a class's curve reaches (the same 6 %
        // tolerance as the cursor dots); nothing while the cursor is away.
        let at_cursor: std::collections::HashMap<u32, f32> = hover.map_or_else(Default::default, |hx| {
            curves.iter().filter_map(|(c, pts)| Some((*c, crate::charts::value_near(pts, hx, 1.)?))).collect()
        });
        let at_cursor = |class: u32| at_cursor.get(&class).copied();
        let rest_precision = |class: u32| ev.per_class.iter().find(|c| c.class == class).map(|c| c.precision);
        let series: Vec<Series> = ev
            .pr
            .iter()
            .zip(&curves)
            .map(|(c, (_, pts))| {
                let pts = pts.clone();
                let name = m.classes.get(c.class as usize).cloned().unwrap_or_default();
                let (color, width) = match iso {
                    Some(f) if f == c.class => (t.class_stroke(c.class), 2.75),
                    Some(_) => (alpha(t.class_stroke(c.class), 0.25), 1.25),
                    None => (t.class_stroke(c.class), 2.),
                };
                // A line pattern per class as well as a colour (five distinct ones), matched by the
                // legend's swatch: the curves read apart without colour.
                let s = Series::new(name, color, pts).width(width);
                match pr_pattern(c.class) {
                    1 => s.dashed(),
                    2 => s.dotted(),
                    3 => s.pattern(crate::charts::LONG_DASH),
                    4 => s.pattern(crate::charts::DASH_DOT),
                    _ => s,
                }
            })
            .collect();
        let view = cx.entity().downgrade();
        let on_hover: HoverFn = Rc::new(move |f, _, cx| {
            view.update(cx, |this, cx| {
                if this.pr_hover != f {
                    this.pr_hover = f;
                    cx.notify();
                }
            })
            .ok();
        });
        // The legend is a list beside the plot, in class order (as the confusion matrix and the
        // class table beside it run); hover or focus isolates a curve.
        let mut order: Vec<&crate::api::PrCurve> = ev.pr.iter().collect();
        order.sort_by_key(|c| c.class);
        let chips: Vec<AnyElement> = order
            .into_iter()
            .map(|c| {
                let class = c.class;
                let key = format!("pr-chip-{class}");
                let on = iso == Some(class);
                let pinned = self.pr_pin == Some(class);
                let name = m.classes.get(class as usize).cloned().unwrap_or_default();
                // The full name (the 68 px slot, 54 beside the pin, can cut it short), the operating
                // point's P and R (the row has room for P only), and what a click does, on hover and
                // on keyboard focus.
                let op = ev
                    .per_class
                    .iter()
                    .find(|p| p.class == class)
                    .map(|p| format!(" · P {:.3} · R {:.3}", p.precision, p.recall))
                    .unwrap_or_default();
                let tip_text =
                    format!("{name}{op} · {}", if pinned { "click to unpin" } else { "click to pin" });
                let ring = self.ring(&self.fh(key.clone(), cx), window);
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .relative()
                    .tooltip(crate::controls::tip(tip_text.clone(), None))
                    .when(ring, |d| d.child(crate::controls::focus_tip(&t, tip_text.clone(), 28.)))
                    .h(px(28.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    .when(on && !pinned, |d| d.bg(t.bg_hover))
                    .when(pinned, |d| d.bg(t.bg_selected))
                    .hover(|s| s.bg(t.bg_hover))
                    .active(|s| s.bg(t.bg_active))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        let next = if *h { Some(class) } else { None };
                        if this.pr_focus != next && (*h || this.pr_focus == Some(class)) {
                            this.pr_focus = next;
                            cx.notify();
                        }
                    }))
                    // A real pointer move (not a row scrolled under a still pointer) ends keyboard
                    // mode, so the hovered class isolates again.
                    .on_mouse_move(cx.listener(move |this, _: &folo_ui::MouseMoveEvent, _, cx| {
                        if this.kb {
                            this.kb = false;
                            this.pr_focus = Some(class);
                            cx.notify();
                        }
                    }))
                    .child(
                        div()
                            .w(px(14.))
                            .flex_none()
                            .child(pattern_swatch(t.class_stroke(class), pr_pattern(class))),
                    )
                    .child(
                        // A known width (the legend row is 220 px: 14 swatch, 68 name, 70 P, 44 AP50
                        // and three 8 px gaps), so long class names end in "…" (text truncates against
                        // the pre-shrink width). A pinned class ends its slot with a pin, the columns
                        // after it unmoved.
                        div()
                            .w(px(68.))
                            .flex_none()
                            .flex()
                            .items_center()
                            .gap(px(2.))
                            .child(
                                text(ty::SM, if pinned { t.text_primary } else { t.text_secondary })
                                    .w(px(if pinned { 54. } else { 68. }))
                                    .flex_none()
                                    .ellipsis()
                                    .child(name),
                            )
                            .when(pinned, |d| d.child(icon("icons/pin.svg", 12., t.text_secondary))),
                    )
                    .child(
                        text(ty::SM, t.text_secondary)
                            .tnum()
                            .w(px(70.))
                            .flex_none()
                            .text_right()
                            // At rest: precision at the best-F1 threshold (as the KPI reports it);
                            // while inspecting: precision at the cursor's recall.
                            .child(
                                at_cursor(class)
                                    .or_else(|| (hover.is_none()).then(|| rest_precision(class)).flatten())
                                    .map_or("–".to_string(), |p| format!("{p:.3}")),
                            ),
                    )
                    .child(
                        text(ty::SM, t.text_primary)
                            .tnum()
                            .w(px(44.))
                            .flex_none()
                            .text_right()
                            .child(format!("{:.3}", c.ap50)),
                    );
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        this.pr_pin = if this.pr_pin == Some(class) { None } else { Some(class) };
                        cx.notify();
                    }),
                )
                .into_any_element()
            })
            .collect();
        let mut chart = LineChart::new("pr-curve", series);
        chart.x_range = Some((0., 1.));
        chart.y_range = Some((0., 1.));
        chart.x_fmt = |v| format!("{v:.2}");
        chart.x_name = "recall";
        chart.axis_titles = Some(("Recall", "Precision"));
        chart.x_ticks = Some(4);
        // Each class's operating point: recall and precision at the one threshold with the best mean
        // F1 (shared by all classes, not each class's own optimum), the values
        // the legend's "P @ best F1" column and the KPI strip report.
        for c in &ev.per_class {
            let shown = match iso {
                Some(f) => f == c.class,
                None => true,
            };
            if shown && ev.pr.iter().any(|p| p.class == c.class) {
                chart.dots.push((c.recall, c.precision, t.class_stroke(c.class)));
            }
        }
        chart.ease_key = crate::charts::ease_key(&m.id);
        // As tall as the confusion matrix beside it (its rows set the panel height).
        // Beside the matrix, as tall as it (its rows set the panel height); stacked above it, a
        // compact 240 so the matrix starts within the first screen.
        chart.height =
            if self.eval_stacked(window) { 240. } else { (32. * ev.confusion.len() as f32 + 136.).max(240.) };
        chart.hover = self.pr_hover;
        chart.on_hover = Some(on_hover);
        // Values live in the legend beside the plot; the cursor shows a recall chip on the axis.
        chart.readout = false;

        // The instruction follows the input in use: keys while the chart or a class row
        // has keyboard focus, the pointer otherwise. Class focus is either family of rows
        // (the legend's chips, the class table's rows), as `pr_isolated` reads it; pointer
        // mode skips the lookups.
        let focused = |k: &str| self.focus.borrow().get(k).is_some_and(|f| f.is_focused(window));
        let row_focused = || {
            ev.pr
                .iter()
                .any(|c| focused(&format!("pr-chip-{}", c.class)) || focused(&format!("pc-row-{}", c.class)))
        };
        if self.kb && row_focused() {
            self.hint_keys("Enter pins the class on the Precision–recall chart (again to unpin)");
        }
        let pr_hint = if self.kb && focused("pr-chart") {
            "← → inspect recall · Esc clears"
        } else if self.kb && row_focused() {
            if self.pr_pin.is_some() {
                "Enter pins the class (again to unpin)"
            } else {
                "Enter pins the class"
            }
        } else if self.pr_pin.is_some() {
            "Pinned: click the class again to unpin."
        } else {
            "Hover a class to isolate it · click to pin"
        };

        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Precision–recall").child(caption(&t, "IoU 0.50")))
            .child(
                div()
                    .flex()
                    .child(div().flex_1().min_w_0().child(self.chart_group(
                        "pr-chart",
                        0.01,
                        1.0,
                        crate::controls::Cursor::Pr,
                        div().pl(px(12.)).pt(px(12.)).pb(px(8.)).child(line_chart(&t, chart)),
                        window,
                        cx,
                    )))
                    .child(
                        div()
                            .w(px(252.))
                            .flex_none()
                            .p(px(8.))
                            .pt(px(12.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(
                                div()
                                    .h(px(20.))
                                    .px(px(8.))
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(th(&t, "Class").flex_1())
                                    // At rest precision at the best-F1 threshold (the KPI's operating
                                    // point); while inspecting, precision at the recall under the cursor.
                                    .child(div().w(px(70.)).flex_none().flex().justify_end().child(th(
                                        &t,
                                        &hover.map_or("P @ best F1".to_string(), |r| format!("P @ R {r:.2}")),
                                    )))
                                    .child(
                                        div()
                                            .w(px(44.))
                                            .flex_none()
                                            .flex()
                                            .justify_end()
                                            .child(th(&t, "AP50")),
                                    ),
                            )
                            .children(chips)
                            // Every class together, as the KPI strip reports it.
                            .when_some(m.metrics.as_ref(), |d, mm| {
                                d.child(
                                    div()
                                        .mt(px(4.))
                                        .pt(px(6.))
                                        .h(px(30.))
                                        .px(px(8.))
                                        .border_t_1()
                                        .border_color(t.line_subtle)
                                        .flex()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(div().w(px(14.)).flex_none())
                                        .child(
                                            text(ty::SM, t.text_secondary)
                                                .w(px(68.))
                                                .flex_none()
                                                .child("All classes"),
                                        )
                                        .child(
                                            text(ty::SM, t.text_secondary)
                                                .tnum()
                                                .w(px(70.))
                                                .flex_none()
                                                .text_right()
                                                // No aggregate value at the cursor: a dash while inspecting.
                                                .child(if hover.is_some() {
                                                    "–".to_string()
                                                } else {
                                                    format!("{:.3}", mm.precision)
                                                }),
                                        )
                                        .child(
                                            text(ty::SM, t.text_primary)
                                                .tnum()
                                                .font_weight(FontWeight::MEDIUM)
                                                .w(px(44.))
                                                .flex_none()
                                                .text_right()
                                                .child(format!("{:.3}", mm.map50)),
                                        ),
                                )
                            })
                            // How to use the panel, in the room under the list (beside the plot).
                            .child(
                                div()
                                    .mt(px(10.))
                                    .px(px(8.))
                                    .flex()
                                    .flex_col()
                                    .gap(px(4.))
                                    .child(
                                        text(ty::SM, t.text_tertiary)
                                            .whitespace_normal()
                                            .child("Dots: P and R at the best F1 threshold."),
                                    )
                                    .child(text(ty::SM, t.text_tertiary).whitespace_normal().child(pr_hint)),
                            ),
                    ),
            )
            .into_any_element()
    }
}

/// A PR class curve's line pattern: 0 solid, 1 dashed, 2 dotted, 3 long dash, 4 dash-dot (by
/// class, so it never changes with sort order or isolation).
fn pr_pattern(class: u32) -> u32 {
    class % 5
}

/// A 14 px legend swatch drawing `pattern` (see [`pr_pattern`]) in `color`.
fn pattern_swatch(color: folo_ui::Hsla, pattern: u32) -> folo_ui::Div {
    let bar = |w: f32| div().w(px(w)).h(px(2.)).flex_none().bg(color);
    let dot = || div().size(px(2.)).flex_none().rounded_full().bg(color);
    let row = div().w(px(14.)).h(px(2.)).flex().items_center().overflow_hidden();
    match pattern {
        1 => row.gap(px(2.)).child(bar(4.)).child(bar(4.)).child(bar(2.)),
        2 => row.gap(px(2.)).child(dot()).child(dot()).child(dot()).child(dot()),
        3 => row.gap(px(3.)).child(bar(9.)).child(bar(2.)),
        4 => row.gap(px(2.)).child(bar(6.)).child(dot()).child(bar(2.)),
        _ => row.child(bar(14.)),
    }
}
