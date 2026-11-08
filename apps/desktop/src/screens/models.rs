//! Models: the model list and a model's evaluation (PR curves, confusion matrix, per-class
//! metrics) with export.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::*;
use crate::api::{Model, ModelEval, time};
use crate::app::{FoloApp, Route, act};
use crate::charts::{HoverFn, LineChart, Series, line_chart};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{alpha, radius, theme, ty};
use crate::ui::*;

/// Row-label column of the confusion matrix (fits "False positives").
const LABEL_W: f32 = 100.;

impl FoloApp {
    pub fn models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned();
        let subtitle = models.as_ref().map_or("Loading…".to_string(), |m| {
            format!("{} · best and last checkpoints, plus exports", plural(m.len(), "model"))
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
            return self.page_frame(&t, header, err, cx);
        }
        let now = self.store.api.now();
        // Narrow windows drop whole columns (GPUI truncates against pre-shrink widths).
        // Budget: every column at full width needs about 1,030 px of content; Params goes first
        // (below 1360), then From run (below 1240).
        let vw = window.viewport_size().width;
        let wide = vw >= px(1240.);
        let params_col = vw >= px(1360.);
        let cols = |d: gpui::Stateful<gpui::Div>| d.flex().items_center().gap(px(16.)).px(px(16.));
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Model").flex_1().min_w(px(200.)))
            .child(th(&t, "Architecture").w(px(110.)).flex_none())
            .when(wide, |d| d.child(th(&t, "From run").w(px(150.)).flex_none()))
            .child(self.sort_th("map", "mAP50-95", 130., true, window, cx))
            .child(num_cell(64., th(&t, "mAP50")))
            .when(params_col, |d| d.child(num_cell(72., th(&t, "Params"))))
            .child(self.sort_th("size", "Size", 72., true, window, cx))
            .child(self.sort_th("created", "Created", 84., true, window, cx));
        // Keyed by the list's content (ids and the sorted fields), so a reload or another backend
        // with the same length never reuses a stale order.
        let src = self.store.models.ready().map_or(0, |v| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for m in v {
                (&m.id, &m.created, m.bytes, m.metrics.as_ref().map(|x| x.map50_95.to_bits())).hash(&mut h);
            }
            h.finish() as usize
        });
        let models = models.map(|m| {
            let Some((k, desc)) = self.models_sort else { return m };
            let key = (src, m.len(), k, desc);
            let mut cache = self.models_order.borrow_mut();
            if cache.as_ref().is_none_or(|(c, _)| *c != key) {
                let score = |m: &Model| -> i64 {
                    let v = match k {
                        "map" => m.metrics.as_ref().map_or(-1, |x| (x.map50_95 * 1e6) as i64),
                        "size" => m.bytes as i64,
                        _ => time::parse(&m.created).unwrap_or(0),
                    };
                    if desc { -v } else { v }
                };
                let mut idx: Vec<usize> = (0..m.len()).collect();
                idx.sort_by_cached_key(|&i| score(&m[i]));
                *cache = Some((key, idx));
            }
            cache.as_ref().map_or(m.clone(), |(_, idx)| idx.iter().map(|&i| m[i].clone()).collect())
        });
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
                        let metrics = m.metrics;
                        let (p, pu) = count_si(m.params);
                        let (b, bu) = bytes(m.bytes);
                        let run = m.run_id.as_ref().and_then(|r| self.store.run(r)).map_or("imported".to_string(), |r| format!("from {}", r.name));
                        let id = m.id.clone();
                        let key = format!("model-row-{}", m.id);
                        let ids_k = ids.clone();
                        let row = cols(div().id(SharedString::from(key.clone())))
                            .h(px(36.))
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
                            .child(div().flex_1().min_w(px(200.)).child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).w(px(200.)).truncate().child(m.name.clone())))
                            .child(text(ty::SM, t.text_secondary).w(px(110.)).flex_none().tnum().child(format!("{} · {} px", m.arch.as_str(), m.imgsz)))
                            .when(wide, |d| d.child(text(ty::SM, t.text_tertiary).w(px(150.)).flex_none().truncate().child(run.trim_start_matches("from ").to_string())))
                            .child(num_cell(130., metrics.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |mm| bar_value(&t, mm.map50_95, f3(mm.map50_95), t.text_tertiary))))
                            .child(num_cell(64., text(ty::SM, t.text_secondary).tnum().child(metrics.as_ref().map_or("–".into(), |mm| f3(mm.map50)))))
                            .when(params_col, |d| d.child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(format!("{p} {pu}")))))
                            .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(format!("{b} {bu}"))))
                            .child(num_cell(84., text(ty::SM, t.text_tertiary).child(time::parse(&m.created).map_or(String::new(), |c| time::ago(c, now)))));
                        self.bind(row, &key, 0., window, cx, act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx))).into_any_element()
                    })
                    .collect();
                panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
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
            return self.page_frame(&t, header, body, cx);
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
            .child(badge(&t, m.arch.as_str(), Tone::Neutral));
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
                .ml(px(-6.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .child(text(ty::BASE, t.text_tertiary).child("· from"))
                .child(text(ty::BASE, t.text_primary).child(r.name.clone()))
                .child(icon("icons/arrow-up-right.svg", 12., t.text_tertiary));
            subtitle = subtitle.child(self.bind(
                link,
                "model-run-link",
                radius::SM,
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Run(rid.clone()), w, cx)),
            ));
        }
        let header = detail_header(
            &t,
            title,
            subtitle,
            vec![export_btn.into_any_element(), try_btn.into_any_element()],
        );

        let mm = m.metrics;
        let tile = |content: gpui::Div| {
            div()
                .flex_grow()
                .flex_basis(gpui::relative(0.144))
                .min_w_0()
                .px(px(16.))
                .py(px(14.))
                .child(content)
        };
        let (p, pu) = count_si(m.params);
        let (b, bu) = bytes(m.bytes);
        // The headline metric gets the headline: twice the width, one size up; the rest are context.
        // Every tile: caption, a value line as tall as the headline's (values bottom-aligned on one
        // baseline), then the detail line.
        let small =
            |label: &str, value: String, unit: &str, detail: Option<gpui::Div>| {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .min_w_0()
                    .child(caption(&t, label))
                    .child(div().h(px(ty::X3XL.line)).flex().items_end().child(
                        metric(&t, value, unit, ty::XL, ty::SM).pb(px(baseline_lift(ty::X3XL, ty::XL))),
                    ))
                    .children(detail)
            };
        let kpis = panel(&t)
            .flex()
            .child(
                div().flex_grow().flex_basis(gpui::relative(0.28)).min_w_0().px(px(16.)).py(px(14.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .child(caption(&t, "mAP50-95"))
                        .child(
                            text(ty::X3XL, t.text_primary)
                                .h(px(ty::X3XL.line))
                                .tnum()
                                .child(mm.as_ref().map_or("–".into(), |x| f3(x.map50_95))),
                        )
                        .child(text(ty::XS, t.text_tertiary).child("IoU 0.50–0.95 · validation split")),
                ),
            )
            .child(vline(&t))
            .child(tile(small(
                "mAP50",
                mm.as_ref().map_or("–".into(), |x| f3(x.map50)),
                "",
                Some(text(ty::XS, t.text_tertiary).child("IoU 0.50")),
            )))
            .child(vline(&t))
            .child(tile(small(
                "Precision",
                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.precision)),
                "",
                Some(text(ty::XS, t.text_tertiary).child("conf 0.25 · IoU 0.50")),
            )))
            .child(vline(&t))
            .child(tile(small(
                "Recall",
                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.recall)),
                "",
                Some(text(ty::XS, t.text_tertiary).child("conf 0.25 · IoU 0.50")),
            )))
            .child(vline(&t))
            .child(tile(small(
                "Parameters",
                p,
                pu,
                Some(text(ty::XS, t.text_tertiary).tnum().child(format!("{:.1} GFLOPs", m.flops_g))),
            )))
            .child(vline(&t))
            .child(tile(small(
                "File size",
                b,
                bu,
                Some(
                    text(ty::XS, t.text_tertiary)
                        .child(format!(".folo · {}", weight_dtype(m.bytes, m.params))),
                ),
            )));

        let body = match self.store.evals.get(id) {
            Some(Load::Ready(ev)) => {
                let ev = ev.clone();
                let pr = self.pr_panel(&m, &ev, window, cx);
                let cm = self.confusion_panel(&m, &ev, window, cx);
                let table = self.class_table(&m, &ev, window, cx);
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
                            .child(div().flex_1().min_w(px(380.)).flex().child(cm)),
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
        self.page_frame(&t, header, body, cx)
    }

    fn export_menu(&mut self, m: &Model, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let open = self.export_open;
        let (half, hu) = bytes(m.bytes / 2);
        let (onnx, ou) = bytes(m.bytes + 120_000);
        let options = [
            ("folo-f16", "FOLO · f16", format!("{half} {hu} · half-precision weights for GPU inference")),
            ("onnx", "ONNX", format!("{onnx} {ou} · opset 17, for other runtimes")),
        ];
        let n = options.len();
        let fmts: Vec<&'static str> = options.iter().map(|o| o.0).collect();
        let hl = self.export_hl.min(n - 1);
        let mid = m.id.clone();
        // Like a select: focus stays on the trigger, arrows move a highlight, Enter commits it.
        let trigger = self
            .button(
                "model-export",
                "Export",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    this.export_open = !this.export_open;
                    this.export_hl = 0;
                    cx.notify();
                }),
            )
            // Captured before the button's own Enter/Space toggle.
            .capture_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, _, cx| {
                if this.export_open && matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    let fmt = fmts[this.export_hl.min(fmts.len() - 1)];
                    this.export_model(mid.clone(), fmt, cx);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .when(open, |d| {
                let cell = self.menu_trigger.clone();
                d.child(
                    gpui::canvas(move |b, _, _| cell.set(Some(b)), |_, _, _, _| {})
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full(),
                )
            })
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, _, cx| {
                let open = this.export_open;
                match ev.keystroke.key.as_str() {
                    "down" | "up" if !open => {
                        this.export_open = true;
                        this.export_hl = 0;
                    }
                    "down" => this.export_hl = (this.export_hl + 1).min(n - 1),
                    "up" => this.export_hl = this.export_hl.saturating_sub(1),
                    "home" if open => this.export_hl = 0,
                    "end" if open => this.export_hl = n - 1,
                    "escape" if open => this.export_open = false,
                    "tab" if open => {
                        this.export_open = false;
                        cx.notify();
                        return;
                    }
                    _ => return,
                }
                cx.stop_propagation();
                cx.notify();
            }));
        let menu = open.then(|| {
            let mid = m.id.clone();
            gpui::deferred(
                div()
                    .absolute()
                    .top(px(34.))
                    .right_0()
                    .w(px(300.))
                    .occlude()
                    .on_mouse_down_out(cx.listener(|this, ev: &gpui::MouseDownEvent, _, cx| {
                        let on_trigger = this.menu_trigger.get().is_some_and(|b| b.contains(&ev.position));
                        if this.export_open && !on_trigger {
                            this.export_open = false;
                            cx.notify();
                        }
                    }))
                    .p(px(4.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .rounded(px(radius::LG))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    .children(options.into_iter().enumerate().map(|(i, (fmt, label, detail))| {
                        let id = mid.clone();
                        div()
                            .id(SharedString::from(format!("export-{fmt}")))
                            .px(px(10.))
                            .py(px(8.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .when(i == hl, |d| d.bg(t.bg_hover))
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .on_click(
                                cx.listener(move |this, _, _, cx| this.export_model(id.clone(), fmt, cx)),
                            )
                            .child(
                                text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child(label),
                            )
                            .child(text(ty::XS, t.text_tertiary).tnum().child(detail))
                            .into_any_element()
                    })),
            )
            .with_priority(2)
        });
        div().relative().child(trigger).children(menu).into_any_element()
    }

    /// A sortable column header: click or Enter/Space sorts by it (descending first, then flips).
    fn sort_th(
        &self,
        key: &'static str,
        label: &'static str,
        w: f32,
        right: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let on = self.models_sort.filter(|s| s.0 == key);
        let chevron = |desc: bool| {
            icon(if desc { "icons/chevron-down.svg" } else { "icons/chevron-up.svg" }, 12., t.text_secondary)
        };
        let el = div()
            .id(SharedString::from(format!("sort-{key}")))
            .w(px(w))
            .flex_none()
            .h(px(24.))
            .flex()
            .items_center()
            .when(right, |d| d.justify_end())
            .gap(px(2.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            // The chevron sits on the side away from the values, so the label stays aligned with them.
            .when_some(on.filter(|_| right), |d, (_, desc)| d.child(chevron(desc)))
            .child(
                text(ty::XS, if on.is_some() { t.text_secondary } else { t.text_tertiary })
                    .font_weight(FontWeight::MEDIUM)
                    .whitespace_nowrap()
                    .child(label),
            )
            .when_some(on.filter(|_| !right), |d, (_, desc)| d.child(chevron(desc)));
        self.bind(
            el,
            &format!("sort-{key}"),
            radius::SM,
            window,
            cx,
            act(move |this, _, cx| {
                this.models_sort = match this.models_sort {
                    Some((k, desc)) if k == key => Some((key, !desc)),
                    _ => Some((key, true)),
                };
                cx.notify();
            }),
        )
        .into_any_element()
    }

    /// The class isolated on the PR chart: the hovered one, else the focused one, else the pin.
    fn pr_isolated(&self, ev: &ModelEval, window: &Window) -> Option<u32> {
        let focused = ev.pr.iter().map(|c| c.class).find(|c| {
            [format!("pr-chip-{c}"), format!("pc-row-{c}")]
                .iter()
                .any(|k| self.focus.borrow().get(k.as_str()).is_some_and(|f| f.is_focused(window)))
        });
        // A pin from another evaluation never applies here.
        self.pr_focus.or(focused).or(self.pr_pin).filter(|c| ev.pr.iter().any(|p| p.class == *c))
    }

    fn pr_panel(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let iso = self.pr_isolated(ev, window);
        let series: Vec<Series> = ev
            .pr
            .iter()
            .map(|c| {
                // End each curve at its highest recall (with a dot) instead of a drop to zero.
                let mut pts: Vec<(f32, f32)> =
                    c.recall.iter().zip(&c.precision).map(|(r, p)| (*r, *p)).collect();
                if let Some(last) = pts.iter().rposition(|p| p.1 > 0.0) {
                    pts.truncate(last + 1);
                }
                let name = m.classes.get(c.class as usize).cloned().unwrap_or_default();
                let (color, width) = match iso {
                    Some(f) if f == c.class => (t.class_stroke(c.class), 2.75),
                    Some(_) => (alpha(t.class_stroke(c.class), 0.25), 1.25),
                    None => (t.class_stroke(c.class), 2.),
                };
                Series::new(name, color, Arc::new(pts)).width(width).live(true)
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
        // The legend is a list beside the plot, best AP50 first; hover or focus isolates a curve.
        let mut order: Vec<&crate::api::PrCurve> = ev.pr.iter().collect();
        order.sort_by(|a, b| b.ap50.total_cmp(&a.ap50));
        let chips: Vec<AnyElement> = order
            .into_iter()
            .map(|c| {
                let class = c.class;
                let key = format!("pr-chip-{class}");
                let on = iso == Some(class);
                let pinned = self.pr_pin == Some(class);
                let el = div()
                    .id(SharedString::from(key.clone()))
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
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        let next = if *h { Some(class) } else { None };
                        if this.pr_focus != next && (*h || this.pr_focus == Some(class)) {
                            this.pr_focus = next;
                            cx.notify();
                        }
                    }))
                    .child(div().w(px(10.)).h(px(2.)).rounded_full().bg(t.class_stroke(class)))
                    .child(
                        text(ty::SM, t.text_secondary)
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .child(m.classes.get(class as usize).cloned().unwrap_or_default()),
                    )
                    .child(text(ty::SM, t.text_primary).tnum().child(format!("{:.3}", c.ap50)));
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
        chart.ease_key = crate::charts::ease_key(&m.id);
        // As tall as the confusion matrix beside it (its rows set the panel height).
        chart.height = (32. * ev.confusion.len() as f32 + 76.).max(220.);
        chart.hover = self.pr_hover;
        chart.on_hover = Some(on_hover);
        chart.value_fmt = Some(|v| format!("P {v:.3}"));

        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Precision–recall")
                    .child(caption(&t, "IoU 0.50 · hover a class to isolate, click to pin")),
            )
            .child(
                div()
                    .flex()
                    .child(div().flex_1().min_w_0().child(self.chart_group(
                        "pr-chart",
                        0.01,
                        crate::controls::Cursor::Pr,
                        div().pl(px(12.)).pt(px(12.)).pb(px(8.)).child(line_chart(&t, chart)),
                        window,
                        cx,
                    )))
                    .child(
                        div()
                            .w(px(156.))
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
                                    .justify_between()
                                    .child(th(&t, "Class"))
                                    .child(th(&t, "AP50")),
                            )
                            .children(chips),
                    ),
            )
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
            &["Row %", "Counts"],
            if norm { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, _| this.confusion_norm = i == 0),
        );
        let n = ev.confusion.len();
        let mut labels: Vec<String> = m.classes.clone();
        labels.truncate(n.saturating_sub(1));
        labels.push("Missed".into());
        // Rows are what was there, columns what was predicted: the last column is objects nothing
        // detected, the last row detections with no object.
        let row_sums: Vec<u32> = ev.confusion.iter().map(|r| r.iter().sum::<u32>().max(1)).collect();
        let max_count = ev.confusion.iter().flatten().copied().max().unwrap_or(1).max(1);
        let cell_h = px(32.);
        // A cool sequential ramp: amber stays reserved for warnings.
        let base = t.chart_series[0];
        let header_row = div().flex().pl(px(LABEL_W)).children(labels.iter().enumerate().map(|(i, l)| {
            div()
                .flex_1()
                .min_w_0()
                .px(px(2.))
                .when(i == n - 1, |d| d.ml(px(6.)))
                .flex()
                .justify_center()
                .child(
                    text(ty::MICRO, if i == n - 1 { t.text_tertiary } else { t.text_secondary })
                        .truncate()
                        .child(l.clone()),
                )
        }));
        let rows = ev.confusion.iter().enumerate().map(|(r, row)| {
            div()
                .flex()
                .items_center()
                .when(r == n - 1, |d| d.mt(px(6.)))
                .child(
                    text(ty::XS, if r == n - 1 { t.text_tertiary } else { t.text_secondary })
                        .w(px(LABEL_W))
                        .flex_none()
                        .pr(px(8.))
                        .text_right()
                        .truncate()
                        .child(if r == n - 1 { "False positives".to_string() } else { labels[r].clone() }),
                )
                .children(row.iter().enumerate().map(|(c, &v)| {
                    let frac = if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
                    let diag = r == c;
                    let na = r == n - 1 && c == n - 1;
                    // Under 1 % reads as noise: leave those cells blank, like zero.
                    let tiny = norm && frac < 0.01;
                    let fill = if na {
                        gpui::transparent_black()
                    } else if v == 0 || tiny {
                        crate::theme::over(t.bg_panel, t.bg_hover)
                    } else {
                        // Correct cells in the series blue; errors on a danger ramp (square root over
                        // 0–100 %), so 2 %, 10 %, 22 % and 25 % stay visibly apart.
                        if diag {
                            crate::theme::over(t.bg_inset, alpha(base, (0.35 + 0.6 * frac).min(0.95)))
                        } else {
                            crate::theme::over(t.bg_panel, alpha(t.danger_solid, err_alpha(frac)))
                        }
                    };
                    let label = if na {
                        "–".to_string()
                    } else if norm {
                        if v == 0 {
                            "·".to_string()
                        } else if tiny {
                            "<1%".to_string()
                        } else {
                            format!("{:.0}%", v as f32 / row_sums[r] as f32 * 100.)
                        }
                    } else {
                        int(v as u64)
                    };
                    let strong = frac > 0.55;
                    // Pick the label colour from the composited cell, not the theme.
                    let ink = if na || v == 0 || tiny {
                        t.text_tertiary
                    } else {
                        t.readable_on(crate::theme::over(t.bg_panel, fill))
                    };
                    div().flex_1().min_w_0().h(cell_h).p(px(1.)).when(c == n - 1, |d| d.ml(px(6.))).child(
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
        // Scale key for the error ramp: a continuous strip from 0 to 100 %.
        let key = div()
            .flex()
            .items_center()
            .gap(px(6.))
            .child(
                div()
                    .w(px(10.))
                    .h(px(10.))
                    .rounded(px(2.))
                    .bg(crate::theme::over(t.bg_inset, alpha(base, 0.8)))
                    .border_1()
                    .border_color(alpha(base, 0.9)),
            )
            .child(text(ty::XS, t.text_tertiary).mr(px(6.)).child("correct"))
            .child(text(ty::XS, t.text_tertiary).child("errors 0%"))
            .child(div().flex().h(px(8.)).w(px(96.)).rounded(px(2.)).overflow_hidden().children((0..12).map(
                |i| {
                    let f = i as f32 / 11.;
                    div()
                        .flex_1()
                        .h_full()
                        .bg(crate::theme::over(t.bg_inset, alpha(t.danger_solid, err_alpha(f))))
                },
            )))
            .child(text(ty::XS, t.text_tertiary).child("100%"));
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Confusion matrix").child(seg))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .px(px(16.))
                    .py(px(12.))
                    .gap(px(2.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .child(
                                text(ty::XS, t.text_tertiary)
                                    .w(px(LABEL_W))
                                    .pr(px(8.))
                                    .text_right()
                                    .child("Actual ↓"),
                            )
                            .child(text(ty::XS, t.text_tertiary).flex_1().child("Predicted →"))
                            .child(key),
                    )
                    .child(header_row)
                    .children(rows)
                    .child(div().pt(px(8.)).w_full().min_w_0().child(
                        text(ty::XS, t.text_tertiary).tnum().whitespace_normal().child(if norm {
                            format!(
                                "Rows: actual objects · bottom row: how all false positives split by class · {} images",
                                int(ev.images as u64)
                            )
                        } else {
                            format!(
                                "Counts of objects and unmatched detections · {} images",
                                int(ev.images as u64)
                            )
                        }),
                    )),
            )
            .into_any_element()
    }

    fn class_table(
        &self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // The column head doubles as the table's title row.
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
        let focus = self.pr_isolated(ev, window);
        let pin = self.pr_pin;
        let rows: Vec<AnyElement> = ev
            .per_class
            .iter()
            .map(|c| {
                let color = t.class(c.class);
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
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        let next = if *h { Some(class) } else { None };
                        if this.pr_focus != next && (*h || this.pr_focus == Some(class)) {
                            this.pr_focus = next;
                            cx.notify();
                        }
                    }))
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(swatch(color, 8.))
                            .child(
                                text(ty::BASE, t.text_primary)
                                    .child(m.classes.get(c.class as usize).cloned().unwrap_or_default()),
                            )
                            .when(pinned, |d| d.child(text(ty::XS, t.text_tertiary).child("pinned"))),
                    )
                    .child(div().w(px(140.)).child(bar_value(&t, c.map50, f3(c.map50), color)))
                    .child(div().w(px(140.)).child(bar_value(&t, c.map50_95, f3(c.map50_95), color)))
                    .child(div().w(px(140.)).child(bar_value(
                        &t,
                        c.precision,
                        format!("{:.3}", c.precision),
                        t.text_tertiary,
                    )))
                    .child(div().w(px(140.)).child(bar_value(
                        &t,
                        c.recall,
                        format!("{:.3}", c.recall),
                        t.text_tertiary,
                    )));
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

/// A fixed-width table cell whose content sits flush right (headers and values share it, so they
/// line up to the pixel).
fn num_cell(w: f32, content: impl IntoElement) -> gpui::Div {
    div().w(px(w)).flex_none().flex().justify_end().child(content)
}

/// Fill opacity of an error cell: a square-root ramp over the whole 0–100 % range.
fn err_alpha(frac: f32) -> f32 {
    0.1 + 0.7 * frac.clamp(0., 1.).sqrt()
}

/// Weight precision inferred from file size per parameter (the API doesn't report it).
fn weight_dtype(bytes: u64, params: u64) -> &'static str {
    match bytes as f64 / params.max(1) as f64 {
        b if b < 1.5 => "int8",
        b if b < 3.0 => "f16",
        _ => "f32",
    }
}
