//! The precision-recall panel: the plot on the left seven twelfths, the per-class table on the right
//! five as its legend. Every class is a thin context line, the mean over all classes is in ink, and
//! a hovered or pinned class is drawn in its class hue (one rule at 3 classes and at 80).

use std::rc::Rc;
use std::sync::Arc;

use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use crate::api::{Model, ModelEval, PrCurve};
use crate::app::FoloApp;
use crate::charts::{HoverFn, LineChart, Series, line_chart};
use crate::theme::{theme, ty};
use crate::ui::*;

/// The plot's height beside the table (its rows scroll inside the panel).
const PLOT_H: f32 = 300.;

/// A curve without its zero tail: evaluators pad it with precision 0 up to recall 1, which no
/// detector reached.
pub(super) fn trim_tail(mut pts: Vec<(f32, f32)>) -> Vec<(f32, f32)> {
    let end = pts.iter().rposition(|p| p.1 > 0.).map_or(0, |i| i + 1);
    pts.truncate(end);
    pts
}

/// The mean precision-recall curve over every class: at each recall step of the shared grid, the
/// classes' mean precision (as Studio computes it from `/eval`).
pub(super) fn mean_curve(pr: &[PrCurve]) -> Vec<(f32, f32)> {
    let Some(first) = pr.first() else { return Vec::new() };
    let n = pr.len() as f32;
    let pts = first
        .recall
        .iter()
        .enumerate()
        .map(|(i, &x)| (x, pr.iter().map(|c| c.precision.get(i).copied().unwrap_or(0.)).sum::<f32>() / n))
        .collect();
    trim_tail(pts)
}

/// The operating points on the chart as (recall, precision, class; None for the whole model): each
/// lit class's from this evaluation, else the whole model's from its run's stored metrics. A file's
/// stored numbers (imported, published) may be on another dataset than these curves: no dot
/// (`detail::run_metrics`).
pub(super) fn operating_dots(m: &Model, ev: &ModelEval, lit: &[u32]) -> Vec<(f32, f32, Option<u32>)> {
    let whole =
        super::detail::run_metrics(m).filter(|_| lit.is_empty()).map(|mm| (mm.recall, mm.precision, None));
    let classes = ev
        .per_class
        .iter()
        .filter(|c| lit.contains(&c.class))
        .map(|c| (c.recall, c.precision, Some(c.class)));
    whole.into_iter().chain(classes).collect()
}

impl FoloApp {
    /// The class isolated on the PR chart: the hovered row, else the focused one, else the pin.
    pub(super) fn pr_isolated(&self, ev: &ModelEval, window: &Window) -> Option<u32> {
        let focused = ev.pr.iter().map(|c| c.class).find(|c| {
            self.focus.borrow().get(format!("pc-row-{c}").as_str()).is_some_and(|f| f.is_focused(window))
        });
        // In keyboard mode the focused row wins over a hover left behind by the pointer (as in
        // Playground and the inspector); moving the pointer hands it back. A pin from another
        // evaluation never applies here.
        let lead = if self.kb { focused.or(self.pr_focus) } else { self.pr_focus.or(focused) };
        lead.or(self.pr_pin).filter(|c| ev.pr.iter().any(|p| p.class == *c))
    }

    /// Plot beside the table unless the content is too narrow for both: then the table goes under.
    fn pr_stacked(&self, window: &Window) -> bool {
        self.content_width(window) < 940.
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
        // Drawn in their hue: the pin, and the class under the pointer or focus.
        let lit: Vec<u32> = [self.pr_pin.filter(|c| ev.pr.iter().any(|p| p.class == *c)), iso]
            .into_iter()
            .flatten()
            .fold(Vec::new(), |mut v, c| {
                if !v.contains(&c) {
                    v.push(c);
                }
                v
            });
        let curves: Vec<(u32, crate::charts::Points)> = ev
            .pr
            .iter()
            .map(|c| {
                (
                    c.class,
                    Arc::new(trim_tail(c.recall.iter().copied().zip(c.precision.iter().copied()).collect())),
                )
            })
            .collect();
        let mean: crate::charts::Points = Arc::new(mean_curve(&ev.pr));
        let name = |c: u32| m.classes.get(c as usize).cloned().unwrap_or_else(|| format!("class {c}"));
        // Context lines first, the mean over them, the lit classes on top.
        let mut series: Vec<Series> = curves
            .iter()
            .filter(|(c, _)| !lit.contains(c))
            .map(|(c, pts)| Series::new(name(*c), t.chart_context, pts.clone()).width(1.))
            .collect();
        series.push(Series::new("All classes", t.text_primary, mean.clone()).width(2.));
        for c in &lit {
            if let Some((_, pts)) = curves.iter().find(|(k, _)| k == c) {
                series.push(Series::new(name(*c), t.class_stroke(*c), pts.clone()).width(2.));
            }
        }
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
        let mut chart = LineChart::new("pr-curve", series);
        chart.x_range = Some((0., 1.));
        chart.y_range = Some((0., 1.));
        chart.x_fmt = |v| format!("{v:.2}");
        chart.x_name = "recall";
        chart.axis_titles = Some(("Recall", "Precision"));
        chart.x_ticks = Some(4);
        // The operating point (recall and precision at the one threshold with the best mean F1):
        // the lit classes' in their hue, else the whole model's in ink.
        chart.dots = operating_dots(m, ev, &lit)
            .into_iter()
            .map(|(r, p, c)| (r, p, c.map_or(t.text_primary, |c| t.class_stroke(c))))
            .collect();
        chart.ease_key = crate::charts::ease_key(&m.id);
        chart.height = PLOT_H;
        chart.hover = self.pr_hover;
        chart.on_hover = Some(on_hover);
        // The legend row above the plot is the readout; the cursor shows a recall chip on the axis.
        chart.readout = false;

        // The readout: the mean and each lit class, AP50 at rest, precision at the cursor's recall.
        let hover = self.pr_hover;
        let at = |pts: &crate::charts::Points| hover.and_then(|hx| crate::charts::value_near(pts, hx, 1.));
        let reading =
            |label: String, swatch: folo_ui::Div, pts: &crate::charts::Points, rest: Option<f32>| {
                let v = match hover {
                    Some(_) => at(pts).map_or("–".to_string(), |p| format!("P {p:.3}")),
                    None => rest.map_or("–".to_string(), |a| format!("AP50 {a:.3}")),
                };
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(swatch)
                    .child(text(ty::SM, t.text_secondary).child(label))
                    .child(text(ty::SM, t.text_primary).tnum().child(v))
            };
        let mut legend = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_x(px(16.))
            .gap_y(px(4.))
            .min_h(px(28.))
            .px(px(16.))
            .pt(px(8.))
            .child(reading(
                "All classes".into(),
                line_swatch(t.text_primary, false, false, 2.),
                &mean,
                super::detail::run_metrics(m).map(|x| x.map50),
            ));
        for c in &lit {
            if let Some((_, pts)) = curves.iter().find(|(k, _)| k == c) {
                let ap = ev.pr.iter().find(|p| p.class == *c).map(|p| p.ap50);
                legend = legend.child(reading(
                    name(*c),
                    line_swatch(t.class_stroke(*c), false, false, 2.),
                    pts,
                    ap,
                ));
            }
        }
        if let Some(r) = hover {
            legend = legend.child(text(ty::SM, t.text_tertiary).tnum().child(format!("at recall {r:.2}")));
        }
        let plot = div().flex().flex_col().min_w_0().child(legend).child(self.chart_group(
            "pr-chart",
            0.01,
            1.0,
            crate::controls::Cursor::Pr,
            div().pl(px(12.)).pt(px(8.)).pb(px(8.)).child(line_chart(&t, chart)),
            window,
            cx,
        ));
        let stacked = self.pr_stacked(window);
        let table = self.class_table(m, ev, &lit, !stacked, window, cx);
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Precision–recall").child(caption(&t, "IoU 0.50")))
            .child(if stacked {
                div()
                    .flex()
                    .flex_col()
                    .child(plot)
                    .child(div().border_t_1().border_color(t.line_subtle).child(table))
            } else {
                div()
                    .flex()
                    .child(div().flex_grow().flex_basis(folo_ui::relative(7. / 12.)).min_w_0().child(plot))
                    .child(
                        div()
                            .flex_grow()
                            .flex_basis(folo_ui::relative(5. / 12.))
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .border_l_1()
                            .border_color(t.line_subtle)
                            .child(table),
                    )
            })
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(class: u32, precision: &[f32]) -> PrCurve {
        let n = precision.len();
        PrCurve {
            class,
            recall: (0..n).map(|i| i as f32 / (n - 1) as f32).collect(),
            precision: precision.to_vec(),
            ap50: 0.,
        }
    }

    #[test]
    fn the_mean_curve_is_the_classes_mean_precision_at_each_recall() {
        let pr = [curve(0, &[1., 0.8, 0.6, 0.]), curve(1, &[0.6, 0.4, 0., 0.])];
        let m = mean_curve(&pr);
        assert_eq!(m.len(), 3, "the zero tail is not data");
        assert!((m[0].1 - 0.8).abs() < 1e-6 && (m[1].1 - 0.6).abs() < 1e-6 && (m[2].1 - 0.3).abs() < 1e-6);
        assert!((m[2].0 - 2. / 3.).abs() < 1e-6);
        assert!(mean_curve(&[]).is_empty());
    }

    #[test]
    fn a_curve_ends_at_its_last_recall_with_precision() {
        assert_eq!(trim_tail(vec![(0., 1.), (0.5, 0.5), (1., 0.)]), vec![(0., 1.), (0.5, 0.5)]);
        assert!(trim_tail(vec![(0., 0.), (1., 0.)]).is_empty());
    }

    #[test]
    fn a_files_stored_metrics_never_draw_the_evaluations_operating_point() {
        use crate::api::{Backend, mock::MockBackend};
        let api = MockBackend::new(4200, true);
        let run_model = api.models().unwrap().into_iter().find(|m| m.run_id.is_some()).unwrap();
        let ev = api.model_eval(&run_model.id).unwrap();
        let mm = run_model.metrics.unwrap();
        // A run's model: its own operating point in ink when no class is lit.
        assert_eq!(operating_dots(&run_model, &ev, &[]), vec![(mm.recall, mm.precision, None)]);
        // A file (imported, published) evaluated here: its stored precision and recall are not
        // this evaluation's, so no whole-model dot; a lit class still gets its own from `ev`.
        let file = Model { run_id: None, ..run_model };
        assert!(operating_dots(&file, &ev, &[]).is_empty());
        let c = &ev.per_class[0];
        assert_eq!(operating_dots(&file, &ev, &[c.class]), vec![(c.recall, c.precision, Some(c.class))]);
    }
}
