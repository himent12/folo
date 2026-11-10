//! The per-class metrics table.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, ModelEval};
use crate::app::{FoloApp, act};
use crate::screens::th;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn class_table(
        &self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The class column fits the longest name (swatch, gap, name, room for the pin), 120 to 240
        // px; the four metric columns share the rest, so the bars start right after the names.
        let longest = ev
            .per_class
            .iter()
            .filter_map(|c| m.classes.get(c.class as usize))
            .map(|n| f32::from(crate::charts::text_width(window, n, ty::BASE.size)))
            .fold(0f32, f32::max);
        let name_w = longest.ceil().min(CLASS_NAME_MAX);
        let class_col = (8. + 8. + name_w + 16. + 24.).clamp(120., 240.);
        // The column head doubles as the table's title row.
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(class_col)).flex_none().child(th(&t, "Class")))
            .child(metric_col().justify_end().child(th(&t, "AP50")))
            .child(metric_col().justify_end().child(th(&t, "AP50-95")))
            .child(metric_col().justify_end().child(th(&t, "Precision")))
            .child(metric_col().justify_end().child(th(&t, "Recall")));
        let focus = self.pr_isolated(ev, window);
        let pin = self.pr_pin;
        // Every metric the same way: a 0–1 bar that grows with the column, then the value.
        let num = |v: f32| {
            metric_col()
                .gap(px(8.))
                .child(div().flex_1().min_w(px(40.)).child(progress(&t, v, t.text_tertiary, 4.)))
                .child(text(ty::SM, t.text_primary).tnum().w(px(40.)).flex_none().text_right().child(f3(v)))
        };
        let mut per: Vec<_> = ev.per_class.iter().collect();
        // Class order, as the PR legend and the confusion matrix run.
        per.sort_by_key(|c| c.class);
        // Numbers right-aligned (bar to their left), like every other table's metric columns.
        let rows: Vec<AnyElement> = per
            .into_iter()
            .map(|c| {
                let class = c.class;
                let key = format!("pc-row-{class}");
                let hot = focus == Some(class);
                let pinned = pin == Some(class);
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    .h(px(36.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .cursor_pointer()
                    .when(hot && !pinned, |d| d.bg(t.bg_hover))
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
                    .on_mouse_move(cx.listener(move |this, _: &gpui::MouseMoveEvent, _, cx| {
                        if this.kb {
                            this.kb = false;
                            this.pr_focus = Some(class);
                            cx.notify();
                        }
                    }))
                    .child(
                        div()
                            .w(px(class_col))
                            .flex_none()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(crate::ui::class_swatch(&t, c.class, 8.))
                            .child(
                                text(ty::BASE, t.text_primary)
                                    .w(px(name_w + 1.))
                                    .flex_none()
                                    .ellipsis()
                                    .child(m.classes.get(c.class as usize).cloned().unwrap_or_default()),
                            )
                            // The same pin as the PR legend's pinned row.
                            .when(pinned, |d| d.child(icon("icons/pin.svg", 12., t.text_secondary))),
                    )
                    .child(num(c.map50))
                    .child(num(c.map50_95))
                    .child(num(c.precision))
                    .child(num(c.recall));
                // Keyboard: Enter/Space isolates (or releases) this class on the PR chart.
                self.bind(
                    row,
                    &key,
                    0.,
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
        panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
    }
}

/// The widest a class name gets in the per-class table before it ends in "…".
const CLASS_NAME_MAX: f32 = 168.;

/// A per-class metric column: equal shares of the width after the class column, at least 120.
fn metric_col() -> gpui::Div {
    div().flex_1().min_w(px(120.)).flex().items_center()
}

/// A fixed-width table cell whose content sits flush right (headers and values share it, so they
/// line up to the pixel).
pub(super) fn num_cell(w: f32, content: impl IntoElement) -> gpui::Div {
    div().w(px(w)).flex_none().flex().justify_end().child(content)
}
