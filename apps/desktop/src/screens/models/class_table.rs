//! The per-class table, the precision-recall plot's legend: Class, AP50, AP50-95, Precision and
//! Recall. Hovering a row isolates its curve, a click (Enter, Space) pins it; the rows scroll inside
//! the panel.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{ClassMetrics, Model, ModelEval};
use crate::app::{FoloApp, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Widths of the four metric columns (`AP50`, `AP50-95`, `Precision`, `Recall`), sized to fit each
/// header label beside its sort chevron without crowding the Class column at 1280 px.
const COL_W: [f32; 4] = [56., 66., 68., 56.];
/// A row's height.
const ROW: f32 = 32.;

/// Orders `per` according to `sort` (`None` keeps class index order). Non-finite metric values
/// always sort last in both directions; ties preserve ascending class index.
pub(super) fn sort_per_class(per: &mut [&ClassMetrics], classes: &[String], sort: Option<(&str, bool)>) {
    let cmp_opt_f32 = |a: f32, b: f32, desc: bool| match (a.is_finite(), b.is_finite()) {
        (true, true) => {
            if desc {
                b.total_cmp(&a)
            } else {
                a.total_cmp(&b)
            }
        }
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        (false, false) => std::cmp::Ordering::Equal,
    };
    match sort {
        None => per.sort_by_key(|c| c.class),
        Some(("class", desc)) => per.sort_by(|a, b| {
            let na = classes.get(a.class as usize).map_or("", String::as_str).to_lowercase();
            let nb = classes.get(b.class as usize).map_or("", String::as_str).to_lowercase();
            let ord = if desc { nb.cmp(&na) } else { na.cmp(&nb) };
            ord.then_with(|| a.class.cmp(&b.class))
        }),
        Some((col, desc)) => per.sort_by(|a, b| {
            let val = |c: &ClassMetrics| match col {
                "map50" => c.map50,
                "map50_95" => c.map50_95,
                "precision" => c.precision,
                "recall" => c.recall,
                _ => f32::NAN,
            };
            cmp_opt_f32(val(a), val(b), desc).then_with(|| a.class.cmp(&b.class))
        }),
    }
}

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
            vals.into_iter()
                .zip(COL_W)
                .map(move |(v, w)| text(ty::SM, ink).tnum().w(px(w)).flex_none().text_right().child(v))
        };
        let class_th = self.sort_th_on(
            self.class_table_sort,
            |a| &mut a.class_table_sort,
            "pc-sort",
            "class",
            "Class",
            52.,
            false,
            window,
            cx,
        );
        let metric_ths = [
            ("AP50", "map50", COL_W[0]),
            ("AP50-95", "map50_95", COL_W[1]),
            ("Precision", "precision", COL_W[2]),
            ("Recall", "recall", COL_W[3]),
        ]
        .map(|(label, key, w)| {
            self.sort_th_on(
                self.class_table_sort,
                |a| &mut a.class_table_sort,
                "pc-sort",
                key,
                label,
                w,
                true,
                window,
                cx,
            )
        });
        let head = div()
            .flex()
            .items_center()
            .gap(px(8.))
            .px(px(16.))
            .h(px(36.))
            .flex_none()
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().flex_1().min_w_0().child(class_th))
            .children(metric_ths);
        // Every class together, as the KPI strip reports it, drawn as the plot's ink line; a run's
        // model only (`run_metrics`: a file's stored numbers may be on another dataset).
        let all = super::detail::run_metrics(m).map(|mm| {
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
        sort_per_class(&mut per, &m.classes, self.class_table_sort);
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

#[cfg(test)]
mod tests {
    use super::sort_per_class;
    use crate::api::ClassMetrics;

    fn cm(class: u32, map50: f32, map50_95: f32, precision: f32, recall: f32) -> ClassMetrics {
        ClassMetrics { class, map50, map50_95, precision, recall }
    }

    #[test]
    fn sort_per_class_orders_metrics_and_keeps_non_finite_last() {
        let classes = vec!["zebra".into(), "apple".into(), "mango".into(), "berry".into()];
        let data = [
            cm(0, 0.70, 0.50, 0.80, 0.60),
            cm(1, f32::NAN, 0.40, 0.90, 0.50),
            cm(2, 0.90, 0.50, 0.70, f32::INFINITY),
            cm(3, 0.70, 0.80, 0.85, 0.75),
        ];
        let mut refs: Vec<&ClassMetrics> = data.iter().collect();

        // Default None keeps class index order.
        sort_per_class(&mut refs, &classes, None);
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [0, 1, 2, 3]);

        // AP50 desc/asc: NaN sorts LAST in both; tie (0.70) breaks by class index ascending.
        sort_per_class(&mut refs, &classes, Some(("map50", true)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [2, 0, 3, 1]);
        sort_per_class(&mut refs, &classes, Some(("map50", false)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [0, 3, 2, 1]);

        // Recall desc/asc: +inf sorts LAST in both.
        sort_per_class(&mut refs, &classes, Some(("recall", true)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [3, 0, 1, 2]);
        sort_per_class(&mut refs, &classes, Some(("recall", false)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [1, 0, 3, 2]);

        // AP50-95 and Precision.
        sort_per_class(&mut refs, &classes, Some(("map50_95", true)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [3, 0, 2, 1]);
        sort_per_class(&mut refs, &classes, Some(("precision", true)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [1, 3, 0, 2]);

        // Class name sorting (asc = A..Z, desc = Z..A).
        sort_per_class(&mut refs, &classes, Some(("class", false)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [1, 3, 2, 0]);
        sort_per_class(&mut refs, &classes, Some(("class", true)));
        assert_eq!(refs.iter().map(|c| c.class).collect::<Vec<_>>(), [0, 2, 3, 1]);
    }
}
