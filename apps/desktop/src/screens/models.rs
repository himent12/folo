//! Models: the model list and a model's evaluation (PR curves, confusion matrix, per-class
//! metrics) with export.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString, Styled,
    Window, div, px,
};

use super::*;
use crate::api::{Model, ModelEval, time};
use crate::app::{FoloApp, Route, act};
use crate::charts::{HoverFn, LineChart, Series, line_chart};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{alpha, radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned();
        let subtitle = models.as_ref().map_or("Loading…".to_string(), |m| {
            format!("{} models · best weights from finished runs and exports", m.len())
        });
        let open = self
            .button(
                "models-open",
                "Try in Playground",
                Btn::Secondary,
                Some("icons/scan-eye.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Playground, w, cx)),
            )
            .into_any_element();
        let header = page_header(&t, "Models", subtitle, vec![open]);
        if let Some(e) = self.store.models.error().cloned() {
            let err = self.error_panel(&e, "models-retry", window, cx);
            return self.page_frame(&t, header, err);
        }
        let now = self.store.api.now();
        let cols = |d: gpui::Stateful<gpui::Div>| d.flex().items_center().gap(px(16.)).px(px(16.));
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Model").flex_1())
            .child(th(&t, "Architecture").w(px(110.)))
            .child(th(&t, "mAP50-95").w(px(130.)))
            .child(th(&t, "mAP50").w(px(64.)).text_right())
            .child(th(&t, "Params").w(px(72.)).text_right())
            .child(th(&t, "Size").w(px(72.)).text_right())
            .child(th(&t, "Created").w(px(84.)).text_right());
        let body = match models {
            None => panel(&t).child(head).child(loading_rows(&t, "models", 4)).into_any_element(),
            Some(m) if m.is_empty() => panel(&t)
                .child(empty_state(&t, "icons/box.svg", "No models yet", "When a run finishes, its best weights appear here, ready to evaluate, export or try in the Playground.", Tone::Neutral))
                .into_any_element(),
            Some(m) => {
                let ids: Vec<String> = m.iter().map(|m| m.id.clone()).collect();
                let rows: Vec<AnyElement> = m
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let metrics = m.metrics.clone();
                        let (p, pu) = count_si(m.params);
                        let (b, bu) = bytes(m.bytes);
                        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).map_or("imported".to_string(), |r| format!("from {}", r.name));
                        let id = m.id.clone();
                        let key = format!("model-row-{}", m.id);
                        let ids_k = ids.clone();
                        let row = cols(div().id(SharedString::from(key.clone())))
                            .h(px(52.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, w, cx| {
                                let next = match ev.keystroke.key.as_str() {
                                    "down" => i + 1,
                                    "up" if i > 0 => i - 1,
                                    _ => return,
                                };
                                if let Some(id) = ids_k.get(next) {
                                    this.focus_key(&format!("model-row-{id}"), w, cx);
                                    cx.stop_propagation();
                                }
                            }))
                            .child(cell2(&t, m.name.clone(), run).flex_1())
                            .child(div().w(px(110.)).flex().child(badge(&t, format!("{} · {}", m.arch, m.imgsz), Tone::Neutral)))
                            .child(div().w(px(130.)).child(metrics.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |mm| bar_value(&t, mm.map50_95, f3(mm.map50_95), t.accent_fill()))))
                            .child(text(ty::SM, t.text_secondary).tnum().w(px(64.)).text_right().child(metrics.as_ref().map_or("–".into(), |mm| f3(mm.map50))))
                            .child(text(ty::SM, t.text_secondary).tnum().w(px(72.)).text_right().child(format!("{p} {pu}")))
                            .child(text(ty::SM, t.text_secondary).tnum().w(px(72.)).text_right().child(format!("{b} {bu}")))
                            .child(text(ty::SM, t.text_tertiary).w(px(84.)).text_right().child(time::parse(&m.created).map_or(String::new(), |c| time::ago(c, now))));
                        self.bind(row, &key, 0., window, cx, act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx))).into_any_element()
                    })
                    .collect();
                panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
            }
        };
        self.page_frame(&t, header, body)
    }

    pub fn model_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(m) = self.store.model(id).cloned() else {
            let header = page_header(&t, "Model", "", vec![]);
            let body = if self.store.models.is_loading() {
                panel(&t).child(loading_rows(&t, "model", 5)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Model not found",
                        "It may have been deleted or moved.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body);
        };
        self.ensure_eval(id, cx);
        let now = self.store.api.now();
        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).cloned();
        let created = time::parse(&m.created).map_or(String::new(), |c| time::ago(c, now));

        let mid = m.id.clone();
        let try_btn = self
            .button(
                "model-try",
                "Try in Playground",
                Btn::Primary,
                Some("icons/scan-eye.svg"),
                window,
                cx,
                act(move |this, w, cx| {
                    this.play_model = Some(mid.clone());
                    this.play_pred = Load::Idle;
                    this.navigate(Route::Playground, w, cx);
                    this.run_predict(cx);
                }),
            )
            .into_any_element();
        let export_btn = self.export_menu(&m, window, cx);
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .child(text(ty::XL, t.text_primary).child(m.name.clone()))
            .child(badge(&t, m.arch.clone(), Tone::Neutral));
        let mut subtitle =
            div().flex().items_center().gap(px(6.)).child(text(ty::BASE, t.text_secondary).child(format!(
                "{} px · {} classes · created {created}",
                m.imgsz,
                m.classes.len()
            )));
        if let Some(r) = run.clone() {
            let rid = r.id.clone();
            let link = div()
                .id("model-run-link")
                .flex()
                .items_center()
                .gap(px(4.))
                .px(px(4.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .child(text(ty::BASE, t.text_tertiary).child("· from"))
                .child(text(ty::BASE, t.accent_text).child(r.name.clone()))
                .child(icon("icons/arrow-up-right.svg", 12., t.accent_text));
            subtitle = subtitle.child(self.bind(
                link,
                "model-run-link",
                radius::SM,
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Run(rid.clone()), w, cx)),
            ));
        }
        let header = div()
            .flex()
            .items_end()
            .justify_between()
            .gap(px(16.))
            .pb(px(4.))
            .child(div().flex().flex_col().gap(px(4.)).min_w_0().child(title).child(subtitle))
            .child(div().flex().gap(px(8.)).child(export_btn).child(try_btn));

        let mm = m.metrics.clone();
        let tile = |content: gpui::Div| div().flex_1().min_w_0().px(px(16.)).py(px(14.)).child(content);
        let (p, pu) = count_si(m.params);
        let (b, bu) = bytes(m.bytes);
        let kpis = panel(&t)
            .flex()
            .child(tile(stat(
                &t,
                "mAP50-95",
                mm.as_ref().map_or("–".into(), |x| f3(x.map50_95)),
                "",
                Some(text(ty::XS, t.text_tertiary).child("COCO 101-point")),
            )))
            .child(vline(&t))
            .child(tile(stat(&t, "mAP50", mm.as_ref().map_or("–".into(), |x| f3(x.map50)), "", None)))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Precision",
                mm.as_ref().map_or("–".into(), |x| format!("{:.2}", x.precision)),
                "",
                None,
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Recall",
                mm.as_ref().map_or("–".into(), |x| format!("{:.2}", x.recall)),
                "",
                None,
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "Parameters",
                p,
                pu,
                Some(text(ty::XS, t.text_tertiary).tnum().child(format!("{:.1} GFLOPs", m.flops_g))),
            )))
            .child(vline(&t))
            .child(tile(stat(
                &t,
                "File size",
                b,
                bu,
                Some(text(ty::XS, t.text_tertiary).child(".folo · f32")),
            )));

        let body = match self.store.evals.get(id) {
            Some(Load::Ready(ev)) => {
                let ev = ev.clone();
                let pr = self.pr_panel(&m, &ev, cx);
                let cm = self.confusion_panel(&m, &ev, window, cx);
                let table = self.class_table(&m, &ev, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(
                        div()
                            .flex()
                            .gap(px(16.))
                            .child(div().flex_1().min_w_0().flex().child(pr))
                            .child(div().w(px(460.)).flex_none().flex().child(cm)),
                    )
                    .child(table)
                    .into_any_element()
            }
            Some(Load::Failed(e)) if e.code == "not_supported" => div()
                .flex()
                .flex_col()
                .gap(px(16.))
                .child(kpis)
                .child(panel(&t).child(empty_state(
                    &t,
                    "icons/chart-line.svg",
                    "No evaluation yet",
                    "Run `folo val` on this model to see PR curves and the confusion matrix.",
                    Tone::Neutral,
                )))
                .into_any_element(),
            Some(Load::Failed(e)) => {
                let e = e.clone();
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(self.error_panel(&e, "eval-retry", window, cx))
                    .into_any_element()
            }
            _ => div()
                .flex()
                .flex_col()
                .gap(px(16.))
                .child(kpis)
                .child(div().h(px(360.)).child(skeleton(&t, "eval")))
                .into_any_element(),
        };
        self.page_frame(&t, header, body)
    }

    fn export_menu(&mut self, m: &Model, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let open = self.export_open;
        let trigger = self.button(
            "model-export",
            "Export",
            Btn::Secondary,
            Some("icons/download.svg"),
            window,
            cx,
            act(|this, _, cx| {
                this.export_open = !this.export_open;
                cx.notify();
            }),
        );
        let (half, hu) = bytes(m.bytes / 2);
        let (onnx, ou) = bytes(m.bytes + 120_000);
        let options = [
            ("folo-f16", "FOLO · f16", format!("{half} {hu} · half-precision weights for GPU inference")),
            ("onnx", "ONNX", format!("{onnx} {ou} · opset 17, for other runtimes")),
        ];
        let menu = open.then(|| {
            let mid = m.id.clone();
            gpui::deferred(
                div()
                    .absolute()
                    .top(px(34.))
                    .right_0()
                    .w(px(300.))
                    .occlude()
                    .p(px(4.))
                    .flex()
                    .flex_col()
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .children(options.into_iter().map(|(fmt, label, detail)| {
                        let id = mid.clone();
                        let key = format!("export-{fmt}");
                        let el = div()
                            .id(SharedString::from(key.clone()))
                            .px(px(10.))
                            .py(px(8.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .child(
                                text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child(label),
                            )
                            .child(text(ty::XS, t.text_tertiary).tnum().child(detail));
                        self.bind(
                            el,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, _, cx| this.export_model(id.clone(), fmt, cx)),
                        )
                        .into_any_element()
                    })),
            )
            .with_priority(2)
        });
        div().relative().child(trigger).children(menu).into_any_element()
    }

    fn pr_panel(&mut self, m: &Model, ev: &ModelEval, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let series: Vec<Series> = ev
            .pr
            .iter()
            .map(|c| {
                let pts: Vec<(f32, f32)> = c.recall.iter().zip(&c.precision).map(|(r, p)| (*r, *p)).collect();
                let name = m.classes.get(c.class as usize).cloned().unwrap_or_default();
                Series::new(name, t.class(c.class), Arc::new(pts)).width(1.5)
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
        let mut chart = LineChart::new("pr-curve", series);
        chart.x_range = Some((0., 1.));
        chart.y_range = Some((0., 1.));
        chart.x_fmt = |v| format!("{v:.2}");
        chart.x_name = "recall";
        chart.axis_titles = Some(("Recall", "Precision"));
        chart.height = 300.;
        chart.hover = self.pr_hover;
        chart.on_hover = Some(on_hover);
        chart.value_fmt = Some(|v| format!("P {v:.3}"));
        let legend = div().flex().flex_wrap().gap(px(12.)).children(ev.pr.iter().map(|c| {
            div()
                .flex()
                .items_center()
                .gap(px(5.))
                .child(div().w(px(10.)).h(px(2.)).rounded_full().bg(t.class(c.class)))
                .child(
                    text(ty::XS, t.text_secondary)
                        .child(m.classes.get(c.class as usize).cloned().unwrap_or_default()),
                )
                .child(text(ty::XS, t.text_tertiary).tnum().child(format!("{:.3}", c.ap50)))
        }));
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Precision–recall")
                    .child(text(ty::XS, t.text_tertiary).child("per class · IoU 0.50")),
            )
            .child(div().px(px(12.)).pt(px(12.)).child(line_chart(&t, chart)))
            .child(div().px(px(16.)).pb(px(14.)).pt(px(4.)).child(legend))
            .into_any_element()
    }

    fn confusion_panel(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let norm = self.confusion_norm;
        let seg = self.segmented(
            "cm-norm",
            &["Normalized", "Counts"],
            if norm { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, _| this.confusion_norm = i == 0),
        );
        let n = ev.confusion.len();
        let mut labels: Vec<String> = m.classes.clone();
        labels.truncate(n.saturating_sub(1));
        labels.push("background".into());
        let row_sums: Vec<u32> = ev.confusion.iter().map(|r| r.iter().sum::<u32>().max(1)).collect();
        let max_count = ev.confusion.iter().flatten().copied().max().unwrap_or(1).max(1);
        let cell = px(((428. - 80.) / n as f32).floor().clamp(36., 64.));
        let base = t.chart_series[1];
        let header_row = div().flex().pl(px(80.)).children(labels.iter().enumerate().map(|(i, l)| {
            div().w(cell).flex().justify_center().child(
                text(ty::MICRO, if i == n - 1 { t.text_tertiary } else { t.text_secondary })
                    .truncate()
                    .child(l.clone()),
            )
        }));
        let rows = ev.confusion.iter().enumerate().map(|(r, row)| {
            div()
                .flex()
                .items_center()
                .child(
                    text(ty::XS, if r == n - 1 { t.text_tertiary } else { t.text_secondary })
                        .w(px(80.))
                        .pr(px(8.))
                        .text_right()
                        .truncate()
                        .child(labels[r].clone()),
                )
                .children(row.iter().enumerate().map(|(c, &v)| {
                    let frac = if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
                    let diag = r == c;
                    let na = r == n - 1 && c == n - 1;
                    let fill = if na {
                        gpui::transparent_black()
                    } else if v == 0 {
                        t.bg_inset
                    } else {
                        crate::theme::over(t.bg_inset, alpha(base, (0.08 + 0.85 * frac).min(0.93)))
                    };
                    let label = if na {
                        "–".to_string()
                    } else if norm {
                        if v == 0 { String::new() } else { format!("{:.2}", v as f32 / row_sums[r] as f32) }
                    } else {
                        int(v as u64)
                    };
                    let strong = frac > 0.55;
                    // Pick the label colour from the composited cell, not the theme.
                    let ink = if na || v == 0 {
                        t.text_tertiary
                    } else {
                        t.readable_on(crate::theme::over(t.bg_panel, fill))
                    };
                    div().w(cell).h(cell).p(px(1.)).child(
                        div()
                            .size_full()
                            .rounded(px(3.))
                            .bg(fill)
                            .when(diag && !na, |d| d.border_1().border_color(alpha(base, 0.9)))
                            .when(na, |d| d.border_1().border_color(t.line_subtle))
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(
                                text(ty::XS, ink)
                                    .tnum()
                                    .when(strong, |d| d.font_weight(FontWeight::MEDIUM))
                                    .child(label),
                            ),
                    )
                }))
        });
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Confusion matrix").child(seg))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .p(px(16.))
                    .gap(px(4.))
                    .child(div().flex().pl(px(80.)).child(text(ty::XS, t.text_tertiary).child("Predicted →")))
                    .child(header_row)
                    .children(rows)
                    .child(div().pt(px(8.)).child(text(ty::XS, t.text_tertiary).tnum().child(format!(
                        "Rows are ground truth · {} validation images",
                        int(ev.images as u64)
                    )))),
            )
            .into_any_element()
    }

    fn class_table(&self, m: &Model, ev: &ModelEval, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Class").flex_1())
            .child(th(&t, "AP50").w(px(140.)))
            .child(th(&t, "AP50-95").w(px(140.)))
            .child(th(&t, "Precision").w(px(140.)))
            .child(th(&t, "Recall").w(px(140.)));
        let rows = ev.per_class.iter().map(|c| {
            let color = t.class(c.class);
            div()
                .flex()
                .items_center()
                .gap(px(16.))
                .px(px(16.))
                .h(px(40.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    div().flex_1().flex().items_center().gap(px(8.)).child(swatch(color, 8.)).child(
                        text(ty::BASE, t.text_primary)
                            .child(m.classes.get(c.class as usize).cloned().unwrap_or_default()),
                    ),
                )
                .child(div().w(px(140.)).child(bar_value(&t, c.map50, f3(c.map50), color)))
                .child(div().w(px(140.)).child(bar_value(&t, c.map50_95, f3(c.map50_95), color)))
                .child(div().w(px(140.)).child(bar_value(
                    &t,
                    c.precision,
                    format!("{:.2}", c.precision),
                    t.text_tertiary,
                )))
                .child(div().w(px(140.)).child(bar_value(
                    &t,
                    c.recall,
                    format!("{:.2}", c.recall),
                    t.text_tertiary,
                )))
        });
        panel(&t)
            .overflow_hidden()
            .child(panel_header(&t, "Per class"))
            .child(head)
            .children(rows)
            .into_any_element()
    }
}
