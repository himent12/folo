//! The per-class table, the precision-recall plot's legend: Class, AP50, AP50-95, Precision and
//! Recall. Hovering a row isolates its curve, a click (Enter, Space) pins it; the rows scroll inside
//! the panel.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, ModelEval};
use crate::app::{FoloApp, act};
use crate::screens::th;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// A metric column's width: fits "Precision" and "0.000".
const NUM_W: f32 = 60.;
/// A row's height.
const ROW: f32 = 32.;

impl FoloApp {
    /// The table. `beside`: it sits beside the plot, so its rows take the plot's height and scroll;
    /// under it, they stop at 320 px.
    pub(super) fn class_table(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        lit: &[u32],
        beside: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let pin = self.pr_pin;
        let nums = |vals: [String; 4], ink: folo_ui::Hsla| {
            vals.map(|v| text(ty::SM, ink).tnum().w(px(NUM_W)).flex_none().text_right().child(v))
        };
        let head = div()
            .flex()
            .items_center()
            .gap(px(8.))
            .px(px(16.))
            .h(px(36.))
            .flex_none()
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().flex_1().min_w_0().child(th(&t, "Class")))
            .children(["AP50", "AP50-95", "Precision", "Recall"].map(|h| num_cell(NUM_W, th(&t, h))));
        // Every class together, as the KPI strip reports it, drawn as the plot's ink line.
        let all = m.metrics.as_ref().map(|mm| {
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(16.))
                .h(px(ROW))
                .flex_none()
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(div().w(px(10.)).flex_none().child(line_swatch(
                            t.text_primary,
                            false,
                            false,
                            2.,
                        )))
                        .child(
                            text(ty::SM, t.text_primary)
                                .font_weight(FontWeight::MEDIUM)
                                .min_w_0()
                                .ellipsis()
                                .child("All classes"),
                        ),
                )
                .children(nums(
                    [f3(mm.map50), f3(mm.map50_95), f3(mm.precision), f3(mm.recall)],
                    t.text_primary,
                ))
        });
        let mut per: Vec<_> = ev.per_class.iter().collect();
        // Class order, as the confusion matrix runs.
        per.sort_by_key(|c| c.class);
        let rows: Vec<AnyElement> = per
            .into_iter()
            .map(|c| {
                let class = c.class;
                let key = format!("pc-row-{class}");
                let on = lit.contains(&class);
                let pinned = pin == Some(class);
                let name = m.classes.get(class as usize).cloned().unwrap_or_default();
                let tip =
                    format!("{name} · {}", if pinned { "click to unpin" } else { "click to pin its curve" });
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .px(px(16.))
                    .h(px(ROW))
                    .flex_none()
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .cursor_pointer()
                    .tooltip(crate::controls::tip(tip, None))
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
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(div().w(px(10.)).flex_none().child(class_swatch(&t, class, 8.)))
                            .child(
                                text(ty::SM, if on || pinned { t.text_primary } else { t.text_secondary })
                                    .min_w_0()
                                    .ellipsis()
                                    .child(name),
                            )
                            .when(pinned, |d| d.child(icon("icons/pin.svg", 12., t.text_secondary))),
                    )
                    .children(nums(
                        [f3(c.map50), f3(c.map50_95), f3(c.precision), f3(c.recall)],
                        t.text_secondary,
                    ));
                self.bind(
                    row,
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
        // The instruction follows the input in use: keys while a class row has keyboard focus, the
        // pointer otherwise.
        let focused = |k: &str| self.focus.borrow().get(k).is_some_and(|f| f.is_focused(window));
        let row_focused = self.kb && ev.pr.iter().any(|c| focused(&format!("pc-row-{}", c.class)));
        if row_focused {
            self.hint_keys("Enter pins the class on the Precision–recall chart (again to unpin)");
        }
        let hint = if self.kb && focused("pr-chart") {
            "← → inspect recall · Esc clears"
        } else if row_focused {
            if pin.is_some() { "Enter pins the class (again to unpin)" } else { "Enter pins the class" }
        } else if pin.is_some() {
            "Pinned: click the class again to unpin."
        } else {
            "Hover a class to isolate its curve · click to pin"
        };
        let list = div()
            .id("pc-scroll")
            .flex()
            .flex_col()
            .overflow_y_scroll()
            .track_scroll(&self.pc_scroll)
            .when(beside, |d| d.flex_1().flex_basis(px(0.)).min_h(px(0.)))
            .when(!beside, |d| d.max_h(px(320.)))
            .children(rows);
        div()
            .flex_1()
            .min_h(px(0.))
            .flex()
            .flex_col()
            .child(head)
            .children(all)
            .child(list)
            // How to use it, and what the dots are, under the rows.
            .child(
                div()
                    .flex_none()
                    .px(px(16.))
                    .py(px(8.))
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(text(ty::SM, t.text_tertiary).whitespace_normal().child(hint))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .whitespace_normal()
                            .child("Dot: precision and recall at the best F1 threshold."),
                    ),
            )
            .into_any_element()
    }
}

/// A fixed-width table cell whose content sits flush right (headers and values share it, so they
/// line up to the pixel).
pub(super) fn num_cell(w: f32, content: impl IntoElement) -> folo_ui::Div {
    div().w(px(w)).flex_none().flex().justify_end().child(content)
}
