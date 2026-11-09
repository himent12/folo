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

/// Row-label column of the confusion matrix (fits "hexagon" and "FP share"); the columns get the
/// rest, so class names head them in full at 1280.
const LABEL_W: f32 = 64.;

impl FoloApp {
    pub fn models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned();
        let subtitle = match (&models, self.store.models.error()) {
            (Some(m), _) => format!("{} · best and last checkpoints, plus exports", plural(m.len(), "model")),
            (None, Some(_)) => "The model list didn't load".to_string(),
            (None, None) => "Loading…".to_string(),
        };
        let open = self
            .button(
                "models-open",
                // No model in hand on the index: it opens Playground (a model's page tries that one).
                "Open Playground",
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
        // The name takes what the other columns leave (GPUI truncates against a known width, so
        // it's computed: content width less the fixed columns, their 16 px gaps, row padding).
        let name_w = {
            let content = self.content_width(window);
            // Architecture is 150 px (its "folo-s · 640 px" plus the e2e tag).
            let fixed = [150., 130., 72., 72., 84.].iter().sum::<f32>()
                + if wide { 150. } else { 0. }
                + if params_col { 72. } else { 0. };
            let columns = 6. + f32::from(u8::from(wide)) + f32::from(u8::from(params_col));
            // Row padding, panel borders, and a few px of slack for rounding.
            (content - fixed - (columns - 1.) * 16. - 32. - 2. - 8.).clamp(200., 520.)
        };
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Model").flex_1().min_w(px(200.)))
            .child(th(&t, "Architecture").w(px(150.)).flex_none())
            .when(wide, |d| d.child(th(&t, "From run").w(px(150.)).flex_none()))
            .child(self.sort_th("map", "mAP50-95", 130., true, window, cx))
            .child(self.sort_th("map50", "mAP50", 72., true, window, cx))
            .when(params_col, |d| d.child(num_cell(72., th(&t, "Params"))))
            .child(self.sort_th("size", "Size", 72., true, window, cx))
            .child(self.sort_th("created", "Created", 84., true, window, cx));
        // Keyed by the list's content (ids and the sorted fields), so a reload or another backend
        // with the same length never reuses a stale order.
        let src = self.store.models.ready().map_or(0, |v| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for m in v {
                (
                    &m.id,
                    &m.created,
                    m.bytes,
                    m.metrics.as_ref().map(|x| (x.map50_95.to_bits(), x.map50.to_bits())),
                )
                    .hash(&mut h);
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
                        "map50" => m.metrics.as_ref().map_or(-1, |x| (x.map50 * 1e6) as i64),
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
                        // Keyboard focus shows what hover does: the full name, the exact time, and
                        // what the e2e tag means.
                        let e2e = self.model_e2e(m);
                        let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
                            let base = time::parse(&m.created).map_or(m.name.clone(), |c| {
                                format!("{} · created {}", m.name, time::local_datetime(c))
                            });
                            if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
                        });
                        // What its numbers were measured on: the dataset of the run that trained it
                        // (mAPs compare only within one dataset), or that it isn't known.
                        let eval_on = m
                            .run_id
                            .as_ref()
                            .and_then(|r| self.store.run(r))
                            .map(|r| {
                                self.store
                                    .dataset_summary(&r.config.dataset_id)
                                    .map_or(r.config.dataset_id.clone(), |d| d.name.clone())
                            })
                            .map_or("Evaluation dataset unknown".to_string(), |d| format!("on {d}"))
                            // Then the weights as stored, so an f16 export reads as one by its record,
                            // not its name.
                            + &model_storage(m).map_or(String::new(), |s| format!(" · {s}"));
                        let row = cols(div().id(SharedString::from(key.clone())))
                            .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 48.)))
                            .h(px(48.))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
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
                            .child(div().id(SharedString::from(format!("m-name-{}", m.id))).flex_1().min_w(px(200.)).when(crate::ui::may_overflow(&m.name, ty::SM.size, name_w), |d| d.tooltip(crate::controls::tip(m.name.clone(), None))).child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).w(px(name_w)).ellipsis().child(m.name.clone())).child(text(ty::SM, t.text_tertiary).w(px(name_w)).ellipsis().child(eval_on)))
                            .child(div().w(px(150.)).flex_none().flex().items_center().gap(px(6.)).child(text(ty::SM, t.text_secondary).tnum().whitespace_nowrap().child(format!("{} · {} px", m.arch.as_str(), m.imgsz))).when(self.model_e2e(m), |d| d.child(e2e_tag(&t, format!("e2e-m-{}", m.id)))))
                            .when(wide, |d| d.child(text(ty::SM, t.text_tertiary).w(px(150.)).flex_none().ellipsis().child(run.trim_start_matches("from ").to_string())))
                            .child(num_cell(130., metrics.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |mm| text(ty::SM, t.text_primary).tnum().child(f3(mm.map50_95)))))
                            .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(metrics.as_ref().map_or("–".into(), |mm| f3(mm.map50)))))
                            .when(params_col, |d| d.child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(format!("{p} {pu}")))))
                            .child(num_cell(72., text(ty::SM, t.text_secondary).tnum().child(format!("{b} {bu}"))))
                            .child(num_cell(84., {
                                // Relative in the column, the exact local time on hover.
                                let at = time::parse(&m.created);
                                text(ty::SM, t.text_tertiary)
                                    .id(SharedString::from(format!("mc-{}", m.id)))
                                    .when_some(at, |d, c| d.tooltip(crate::controls::tip(time::local_datetime(c), None)))
                                    .tnum()
                                    .child(at.map_or(String::new(), |c| time::ago(c, now)))
                            }));
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
            } else if let Some(e) = self.store.models.error().cloned() {
                // The list failed to load: say that (not "not found", which suggests a deletion).
                let retry = self.button(
                    "model-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_models(cx)),
                );
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Models didn't load",
                        &e.message,
                        Tone::Danger,
                    ))
                    .child(div().pb(px(24.)).flex().justify_center().child(retry))
                    .into_any_element()
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
        let e2e = self.model_e2e(&m);
        let title = div()
            .flex()
            .items_center()
            .gap(px(10.))
            .child(text(ty::XL, t.text_primary).child(m.name.clone()))
            .child(badge(&t, m.arch.as_str(), Tone::Neutral))
            // End-to-end, with its help on hover and keyboard focus (the help icon's own stop).
            .when(e2e, |d| {
                d.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(4.))
                        // The same compact tag as every list, right after the architecture badge;
                        // its help on hover and keyboard focus.
                        .child(e2e_tag(&t, "e2e-model-head"))
                        .child(self.help_icon("model-head-help", FoloApp::E2E_HELP, window, cx)),
                )
            });
        let mut subtitle =
            div().flex().items_center().gap(px(6.)).child(text(ty::BASE, t.text_secondary).child(format!(
                "{} px · {} classes{}{} · created {created}",
                m.imgsz,
                m.classes.len(),
                // The weights as stored (precision, fused), when the record says.
                model_storage(&m).map_or(String::new(), |s| format!(" · {s} weights")),
                // How it was trained, when the record says (`trained_with`; exports have none).
                if m.trained_with.is_some_and(|w| w.amp) { " · trained in f16" } else { "" },
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
                .active(|s| s.bg(t.bg_active))
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
        // The dataset the headline numbers were measured on (the run's), named beside them.
        // The split evaluation used: validation, or train when the dataset has no validation
        // images (the server falls back to them).
        let head_ds = run
            .as_ref()
            .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
            .map(|d| (d.name.clone(), if d.splits.val == 0 { "train" } else { "val" }));
        // Where precision and recall are taken: the best-F1 confidence, named when the record has it.
        // The headline names the split; this tile's narrow caption says where on the curve.
        let op_point = pr_caption(mm.as_ref().and_then(|x| x.best_f1_conf), "", true);
        let tile = |content: gpui::Div| {
            div()
                .flex_grow()
                .flex_basis(gpui::relative(0.144))
                .min_w_0()
                .px(px(16.))
                .py(px(14.))
                .child(content)
        };
        let wide_tile = |content: gpui::Div| {
            div()
                .flex_grow()
                .flex_basis(gpui::relative(0.21))
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
                        .child(text(ty::SM, t.text_tertiary).whitespace_nowrap().ellipsis().child(
                            match &head_ds {
                                Some((d, split)) => {
                                    format!("IoU 0.50–0.95 · {}", split_label(d, split, None))
                                }
                                None => "IoU 0.50–0.95".to_string(),
                            },
                        )),
                ),
            )
            .child(vline(&t))
            .child(tile(small(
                "mAP50",
                mm.as_ref().map_or("–".into(), |x| f3(x.map50)),
                "",
                Some(text(ty::SM, t.text_tertiary).child("IoU 0.50")),
            )))
            .child(vline(&t))
            // The same pair and caption as the run page (ui::metric_pair, ui::pr_caption); a wider
            // share than the other small tiles, as it holds two values and the operating point.
            .child(wide_tile(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .min_w_0()
                    .child(caption(&t, "Precision / Recall"))
                    .child(
                        div().h(px(ty::X3XL.line)).flex().items_end().child(
                            metric_pair(
                                &t,
                                ty::XL,
                                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.precision)),
                                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.recall)),
                            )
                            .pb(px(baseline_lift(ty::X3XL, ty::XL))),
                        ),
                    )
                    // Clipped, not ellipsized: the run page's twin caption hit GPUI 0.2.2's truncation
                    // panic (a stale run length sliced inside "·"); this one fits its tile.
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .whitespace_nowrap()
                            .overflow_hidden()
                            .child(op_point.clone()),
                    ),
            ))
            .child(vline(&t))
            .child(tile(small(
                "Parameters",
                p,
                pu,
                Some(text(ty::SM, t.text_tertiary).tnum().child(format!("{:.1} GFLOPs", m.flops_g))),
            )))
            .child(vline(&t))
            .child(tile(small(
                "File size",
                b,
                bu,
                Some(
                    text(ty::SM, t.text_tertiary)
                        // The record's stored precision (as the header says), never a guess from size.
                        // The format; the stored precision is the subtitle's to say (once).
                        .child(".folo"),
                ),
            )));

        let body = match self.store.evals.get(id) {
            Some(Load::Ready(ev)) => {
                let ev = ev.clone();
                // Side by side only while the PR panel keeps 600 px beside the matrix (46 %, 440 px
                // minimum): its plot and side legend both fit. Else one above the other, each full
                // width (at the 1200 content cap that's below a ~1430 px window).
                let stacked = {
                    let row = self.content_width(window) - 16.;
                    row - (row * 0.46).max(440.) < 600.
                };
                let pr = self.pr_panel(&m, &ev, window, cx);
                let cm = self.confusion_panel(&m, &ev, window, cx);
                let table = self.class_table(&m, &ev, window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(if stacked {
                        // Narrow: one above the other, each full width.
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(16.))
                            .child(div().flex().child(pr))
                            .child(div().flex().child(cm))
                    } else {
                        // The PR panel holds a plot and its legend: it takes the larger share, so the
                        // plot keeps ~240 px at 1280; the matrix (46 %, 440 px minimum) heads every column
                        // with the class's full name.
                        div()
                            .flex()
                            .gap(px(16.))
                            .child(
                                div().flex_grow().flex_basis(gpui::relative(0.54)).min_w_0().flex().child(pr),
                            )
                            .child(
                                div()
                                    .flex_grow()
                                    .flex_basis(gpui::relative(0.46))
                                    .min_w(px(440.))
                                    .flex()
                                    .child(cm),
                            )
                    })
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
                    // The server can't evaluate here (not a model without results): say which.
                    "Evaluation isn't available on this server",
                    "Its PR curves and confusion matrix need a server that evaluates models. `folo val` prints its metrics in a terminal.",
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
        // Estimates from the parameter count: 2 bytes per weight in f16, 4 in ONNX (f32).
        let (half, hu) = bytes(m.params * 2);
        let (onnx, ou) = bytes(m.params * 4);
        let options = [
            ("folo-f16", "FOLO · f16", format!("≈ {half} {hu} · half-precision weights for GPU inference")),
            ("onnx", "ONNX", format!("≈ {onnx} {ou} · f32, opset 17, for other runtimes")),
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
                            .child(text(ty::SM, t.text_tertiary).tnum().child(detail))
                            .into_any_element()
                    })),
            )
            .with_priority(2)
        });
        div().relative().child(trigger).children(menu).into_any_element()
    }

    /// A sortable column header of the Models table.
    fn sort_th(
        &self,
        key: &'static str,
        label: &'static str,
        w: f32,
        right: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.sort_th_on(self.models_sort, |a| &mut a.models_sort, "sort", key, label, w, right, window, cx)
    }

    /// A sortable column header: click or Enter/Space sorts by it (descending first, then flips).
    /// `current` is the table's sort and `field` where it lives; `prefix` keeps element ids
    /// apart between tables.
    #[allow(clippy::too_many_arguments)]
    pub fn sort_th_on(
        &self,
        current: Option<(&'static str, bool)>,
        field: fn(&mut Self) -> &mut Option<(&'static str, bool)>,
        prefix: &str,
        key: &'static str,
        label: &'static str,
        w: f32,
        right: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let on = current.filter(|s| s.0 == key);
        let chevron = |desc: bool| {
            icon(if desc { "icons/chevron-down.svg" } else { "icons/chevron-up.svg" }, 12., t.text_primary)
        };
        // Unsorted columns: a faint up-down mark, a different shape from the active direction.
        let mark = |on: Option<(&'static str, bool)>| match on {
            Some((_, desc)) => chevron(desc),
            None => icon("icons/chevrons-up-down.svg", 12., t.text_disabled),
        };
        let id = format!("{prefix}-{key}");
        let el = div()
            .id(SharedString::from(id.clone()))
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
            .tooltip(crate::controls::tip(format!("Sort by {label}"), None))
            // The chevron sits on the side away from the values, so the label stays aligned with them.
            // Unsorted columns show a faint one, so they read as sortable before the first click.
            .when(right, |d| d.child(mark(on)))
            .child(
                text(ty::SM, if on.is_some() { t.text_primary } else { t.text_tertiary })
                    .font_weight(FontWeight::MEDIUM)
                    .whitespace_nowrap()
                    .child(label),
            )
            .when(!right, |d| d.child(mark(on)));
        self.bind(
            el,
            &id,
            radius::SM,
            window,
            cx,
            act(move |this, _, cx| {
                let sort = field(this);
                *sort = match *sort {
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
        // In keyboard mode the focused row wins over a hover left behind by the pointer (as in
        // Playground and the inspector); moving the pointer hands it back. A pin from another
        // evaluation never applies here.
        let lead = if self.kb { focused.or(self.pr_focus) } else { self.pr_focus.or(focused) };
        lead.or(self.pr_pin).filter(|c| ev.pr.iter().any(|p| p.class == *c))
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
                    .on_mouse_move(cx.listener(move |this, _: &gpui::MouseMoveEvent, _, cx| {
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
                        // and three 8 px gaps), so long class names end in "…" (GPUI truncates against
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
        chart.height = (32. * ev.confusion.len() as f32 + 136.).max(240.);
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

    fn confusion_panel(
        &mut self,
        m: &Model,
        ev: &ModelEval,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let norm = self.confusion_norm;
        // The evaluated dataset and split, as the server reports them (`ModelEval`); an older server
        // leaves them out, and then they're inferred from the model's run.
        let eval_ds = ev.dataset_name.clone().zip(ev.split.map(|s| s.as_str())).or_else(|| {
            m.run_id
                .as_ref()
                .and_then(|r| self.store.run(r))
                .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
                .map(|d| (d.name.clone(), if d.splits.val == 0 { "train" } else { "val" }))
        });
        // What the matrix counts: predictions above `confusion_conf`, matched at `confusion_iou`.
        let cm_rule = ev
            .confusion_conf
            .zip(ev.confusion_iou)
            .map(|(c, i)| format!(" · conf {} · IoU {:.2}", fmt_conf(c), i))
            .unwrap_or_default();
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
        // The error ramp spans 0 to the largest confusion between classes (the false-positive row,
        // a split of all false positives that sums to 100 %, is plain numbers), so the worst confusion is the
        // darkest cell and small ones stay apart (the key states that maximum).
        let frac_of =
            |r: usize, v: u32| if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
        let last = ev.confusion.len().saturating_sub(1);
        let max_err = ev
            .confusion
            .iter()
            .enumerate()
            .flat_map(|(r, row)| row.iter().enumerate().map(move |(c, &v)| (r, c, v)))
            .filter(|&(r, c, _)| r != c && r != last)
            .map(|(r, _, v)| frac_of(r, v))
            .fold(0f32, f32::max)
            .max(1e-6);
        let cell_h = px(32.);
        // Correct cells on the success hue: a series or class colour would read as "that class"
        // (series 1 is also the circle class's cyan).
        let base = t.success_solid;
        // Column headings get a measured width (the column area as last laid out, split evenly):
        // a name that doesn't fit shows a short form, in full on hover.
        let cols = self.cm_cols.clone();
        // A heading may use its whole cell less a 2 px gap to the next one (medium weight measures
        // ~4 % wider than the regular face `text_width` sets).
        let wide = |window: &mut Window, s: &str| crate::charts::text_width(window, s, ty::SM.size) * 1.04;
        // "Missed" is never shortened: when an even share can't hold it, it takes the width it needs
        // and the class columns share the rest.
        let missed_need = wide(window, "Missed") + px(4.);
        let missed_w =
            cols.get().filter(|w| (*w - px(6.)) / n.max(1) as f32 <= missed_need).map(|_| missed_need);
        let col_w = cols.get().map(|w| {
            let k = if missed_w.is_some() { n.saturating_sub(1) } else { n };
            (w - px(6.) - missed_w.unwrap_or_default()) / k.max(1) as f32 - px(2.)
        });
        let header_row = div().flex().pl(px(LABEL_W)).child(
            div()
                .relative()
                .flex_1()
                .min_w_0()
                .flex()
                .child(
                    gpui::canvas(
                        move |b, window, _| {
                            if cols.get() != Some(b.size.width) {
                                cols.set(Some(b.size.width));
                                window.request_animation_frame();
                            }
                        },
                        |_, _, _, _| {},
                    )
                    .absolute()
                    .top_0()
                    .left_0()
                    .size_full(),
                )
                .children(labels.iter().enumerate().map(|(i, l)| {
                    // The full name, else its short form, else its initial (each measured); a
                    // shortened heading shows the full name on hover.
                    // A heading may overhang its cell by 2 px into the 4 px gap.
                    let room =
                        |window: &mut Window, s: &str| col_w.is_none_or(|w| wide(window, s) <= w + px(4.));
                    let fits = i == n - 1 || room(window, l);
                    let shown = if fits {
                        l.clone()
                    } else if room(window, &short_name(l)) {
                        short_name(l)
                    } else {
                        l.chars().next().map(String::from).unwrap_or_default()
                    };
                    div()
                        .id(SharedString::from(format!("cm-col-{i}")))
                        .flex_1()
                        .min_w_0()
                        .px(px(2.))
                        .when(i == n - 1, |d| d.ml(px(6.)))
                        .when(i == n - 1, |d| d.when_some(missed_w, |d, w| d.flex_none().w(w)))
                        .flex()
                        .justify_center()
                        .when(!fits, |d| d.tooltip(crate::controls::tip(l.clone(), None)))
                        .child(
                            text(ty::SM, if i == n - 1 { t.text_tertiary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .whitespace_nowrap()
                                // A name that passed the fit check renders whole (a few px into the gap at most);
                                // only a shortened form keeps the hard cap.
                                .when(!fits, |d| {
                                    d.when_some(col_w, |d, w| d.max_w(w + px(4.)).overflow_hidden())
                                })
                                .child(shown),
                        )
                })),
        );
        // One cell at a time can be read out: under the pointer, or the matrix's keyboard cursor.
        let sel = self.cm_cell.filter(|&(r, c)| r < n && c < n);
        let readout = sel.map(|(r, c)| {
            let v = ev.confusion[r][c];
            let pct = |d: u32| format!("{:.0}%", v as f32 / d.max(1) as f32 * 100.);
            let name = |i: usize| labels.get(i).cloned().unwrap_or_default();
            let count = int(v as u64);
            match (r == n - 1, c == n - 1) {
                (true, true) => "No value: there is no unmatched detection of no object".to_string(),
                (true, false) => {
                    format!(
                        "False positives predicted as {}: {count} · {} of all false positives",
                        name(c),
                        pct(row_sums[r])
                    )
                }
                (false, true) => {
                    format!("{} missed: {count} · {} of {} objects", name(r), pct(row_sums[r]), name(r))
                }
                (false, false) if r == c => {
                    format!("{} found: {count} · {} of {} objects", name(r), pct(row_sums[r]), name(r))
                }
                (false, false) => format!(
                    "{} predicted as {}: {count} · {} of {} objects",
                    name(r),
                    name(c),
                    pct(row_sums[r]),
                    name(r)
                ),
            }
        });
        let rows = ev.confusion.iter().enumerate().map(|(r, row)| {
            div()
                .flex()
                .items_center()
                .when(r == n - 1, |d| d.mt(px(8.)))
                .child(
                    text(ty::SM, if r == n - 1 { t.text_tertiary } else { t.text_secondary })
                        .font_weight(FontWeight::MEDIUM)
                        .w(px(LABEL_W))
                        .flex_none()
                        .pr(px(8.))
                        .text_right()
                        .ellipsis()
                        .child(if r == n - 1 {
                            // Its denominator differs from the class rows': all false positives.
                            if norm { "FP share".to_string() } else { "FP count".to_string() }
                        } else {
                            labels[r].clone()
                        }),
                )
                .children(row.iter().enumerate().map(|(c, &v)| {
                    let frac = if norm { v as f32 / row_sums[r] as f32 } else { v as f32 / max_count as f32 };
                    let diag = r == c;
                    let na = r == n - 1 && c == n - 1;
                    // Under 1 % reads as noise: leave those cells blank, like zero.
                    let tiny = norm && frac < 0.01;
                    let fill = if na {
                        gpui::transparent_black()
                    } else if v == 0 {
                        crate::theme::over(t.bg_panel, t.bg_hover)
                    } else {
                        // Correct cells on the success ramp; errors on a danger ramp (linear from a
                        // faint floor up to the largest error), so small and large confusions stay apart.
                        if diag {
                            crate::theme::over(t.bg_inset, alpha(base, (0.35 + 0.6 * frac).min(0.95)))
                        } else if r == n - 1 {
                            // The false-positive split is read as numbers: a flat inset, no ramp (a
                            // grey ramp beside the red one read as an encoding without a key).
                            t.bg_inset
                        } else {
                            crate::theme::over(t.bg_panel, alpha(t.danger_solid, err_alpha(frac / max_err)))
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
                    let ink = if na || v == 0 {
                        t.text_tertiary
                    } else if r == n - 1 && !diag {
                        t.text_secondary
                    } else {
                        t.readable_on(crate::theme::over(t.bg_panel, fill))
                    };
                    let picked = sel == Some((r, c));
                    div()
                        .id(SharedString::from(format!("cm-{r}-{c}")))
                        .flex_1()
                        .min_w_0()
                        .h(cell_h)
                        .p(px(1.))
                        .when(c == n - 1, |d| d.ml(px(6.)).when_some(missed_w, |d, w| d.flex_none().w(w)))
                        .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                            if *h {
                                this.cm_cell = Some((r, c));
                                cx.notify();
                            } else if this.cm_cell == Some((r, c)) {
                                this.cm_cell = None;
                                cx.notify();
                            }
                        }))
                        .child(
                            div()
                                .size_full()
                                .rounded(px(3.))
                                .bg(fill)
                                .when(diag && !na, |d| d.border_1().border_color(alpha(base, 0.9)))
                                // The cell being read out.
                                .when(picked, |d| d.border_2().border_color(t.text_primary))
                                .flex()
                                .items_center()
                                .justify_center()
                                .child(
                                    text(ty::SM, ink)
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
            .child(text(ty::SM, t.text_tertiary).mr(px(6.)).child("correct"))
            .child(text(ty::SM, t.text_tertiary).ml(px(2.)).child(if norm {
                "Error share 0%"
            } else {
                "Errors 0"
            }))
            .child(div().flex().h(px(8.)).w(px(96.)).rounded(px(2.)).overflow_hidden().children((0..12).map(
                |i| {
                    let f = i as f32 / 11.;
                    div()
                        .flex_1()
                        .h_full()
                        .bg(crate::theme::over(t.bg_inset, alpha(t.danger_solid, err_alpha(f))))
                },
            )))
            .child(text(ty::SM, t.text_tertiary).tnum().child(if norm {
                format!("{:.0}% max", max_err * 100.)
            } else {
                format!("{} max", int((max_err * max_count as f32).round() as u64))
            }));
        // The cells are one Tab stop: arrows move a cursor cell (the readout follows), Esc clears.
        let rows: Vec<AnyElement> = rows.map(|r| r.into_any_element()).collect();
        let fh = self.fh("cm-grid", cx);
        let ring = self.ring(&fh, window);
        let grid = div()
            .id("cm-grid")
            .track_focus(&fh)
            .relative()
            .flex()
            .flex_col()
            .gap(px(2.))
            .when(ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .on_action(cx.listener(|this, _: &crate::app::Dismiss, _, cx| {
                if this.cm_cell.take().is_some() {
                    cx.notify();
                } else {
                    cx.propagate();
                }
            }))
            .on_key_down(cx.listener(move |this, ev: &gpui::KeyDownEvent, _, cx| {
                let key = ev.keystroke.key.as_str();
                let arrows = matches!(key, "left" | "right" | "up" | "down" | "home" | "end");
                // A cell from another model's (larger) matrix doesn't count: start over.
                let cur = this.cm_cell.filter(|&(r, c)| r < n && c < n);
                let next = match (cur, key) {
                    // The first press lands on the top-left cell.
                    (None, _) if arrows => Some((0, 0)),
                    (Some((r, c)), "left") => Some((r, c.saturating_sub(1))),
                    (Some((r, c)), "right") => Some((r, (c + 1).min(n - 1))),
                    (Some((r, c)), "up") => Some((r.saturating_sub(1), c)),
                    (Some((r, c)), "down") => Some(((r + 1).min(n - 1), c)),
                    (Some((r, _)), "home") => Some((r, 0)),
                    (Some((r, _)), "end") => Some((r, n - 1)),
                    _ => None,
                };
                if let Some(cell) = next {
                    this.cm_cell = Some(cell);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .children(rows);
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
                                text(ty::SM, t.text_tertiary)
                                    .w(px(LABEL_W))
                                    .pr(px(8.))
                                    .text_right()
                                    .child("Actual ↓"),
                            )
                            .child(text(ty::SM, t.text_tertiary).flex_1().child("Predicted →")),
                    )
                    .child(header_row)
                    .child(grid)
                    // The key sits under the matrix, away from the column labels.
                    .child(div().pt(px(10.)).child(key))
                    // The selected cell in words, else what the rows mean.
                    .child(div().pt(px(4.)).w_full().min_w_0().child(match readout {
                        Some(r) => text(ty::SM, t.text_primary).tnum().whitespace_normal().child(r),
                        None => text(ty::SM, t.text_tertiary).whitespace_normal().child(if norm {
                            "Rows are actual objects; the bottom row splits false positives by class. Hover or focus a cell to read it."
                        } else {
                            "Counts of objects and unmatched detections. Hover or focus a cell to read it."
                        }),
                    }))
                    // The grid is one Tab stop: with keyboard focus, how to move in it.
                    .when(ring, |d| {
                        d.child(
                            div()
                                .pt(px(2.))
                                .child(text(ty::SM, t.text_tertiary).child("Arrows move · Home / End row ends · Esc clears")),
                        )
                    })
                    .child(
                        div().pt(px(4.)).child(
                            text(ty::SM, t.text_tertiary)
                                .tnum()
                                .child(match &eval_ds {
                                    Some((d, split)) => {
                                        format!("{}{cm_rule}", split_label(d, split, Some(ev.images as u64)))
                                    }
                                    // Unknown dataset: the split isn't known either.
                                    None => format!("Evaluation · {} images{cm_rule}", int(ev.images as u64)),
                                }),
                        ),
                    )
                    // Evaluated on another split than validation: the server's sentence says why.
                    .when_some(ev.split_fallback.as_ref(), |d, f| {
                        d.child(text(ty::SM, t.text_secondary).child(f.message.clone()))
                    }),
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
            // Spare width is shared (Class 2 : each metric 1, to 360 / 160 px); past the caps it
            // opens before the metrics, which stay close enough to read across.
            .child(grow_col(th(&t, "Class"), 200., 2., 360.))
            .child(grow_col(th(&t, "AP50"), 120., 1., 160.).ml_auto().text_right())
            .child(grow_col(th(&t, "AP50-95"), 120., 1., 160.).text_right())
            .child(grow_col(th(&t, "Precision"), 120., 1., 160.).text_right())
            .child(grow_col(th(&t, "Recall"), 120., 1., 160.).text_right());
        let focus = self.pr_isolated(ev, window);
        let pin = self.pr_pin;
        let num =
            |v: String| grow_col(text(ty::SM, t.text_secondary).tnum(), 120., 1., 160.).text_right().child(v);
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
                        grow_col(div(), 200., 2., 360.)
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(crate::ui::class_swatch(&t, c.class, 8.))
                            .child(
                                text(ty::BASE, t.text_primary)
                                    .w(px(if pinned { 148. } else { 168. }))
                                    .flex_none()
                                    .ellipsis()
                                    .child(m.classes.get(c.class as usize).cloned().unwrap_or_default()),
                            )
                            // The same pin as the PR legend's pinned row.
                            .when(pinned, |d| d.child(icon("icons/pin.svg", 12., t.text_secondary))),
                    )
                    // Only the headline metric carries a comparison bar; the others are plain
                    // numbers, so AP50-95 leads the row.
                    .child(num(f3(c.map50)).ml_auto())
                    .child(grow_col(div(), 120., 1., 160.).flex().justify_end().child(bar_value(
                        &t,
                        c.map50_95,
                        f3(c.map50_95),
                        t.text_tertiary,
                    )))
                    .child(num(format!("{:.3}", c.precision)))
                    .child(num(format!("{:.3}", c.recall)));
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

/// A PR class curve's line pattern: 0 solid, 1 dashed, 2 dotted, 3 long dash, 4 dash-dot (by
/// class, so it never changes with sort order or isolation).
fn pr_pattern(class: u32) -> u32 {
    class % 5
}

/// A 14 px legend swatch drawing `pattern` (see [`pr_pattern`]) in `color`.
fn pattern_swatch(color: gpui::Hsla, pattern: u32) -> gpui::Div {
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

/// A column heading's short form: the first three letters and a full stop ("hexagon" → "hex.").
fn short_name(name: &str) -> String {
    let short: String = name.chars().take(3).collect();
    if short.chars().count() < name.chars().count() { format!("{short}.") } else { short }
}

/// Fill opacity of an error cell: linear over `frac`, the cell's share of the largest
/// class-to-class confusion shown, from a faint 0.1 (still apart from the empty cells) to 0.8.
/// The square-root ramp it replaces lifted 1–4 % confusions toward full red, so the whole
/// off-diagonal read as alarm.
fn err_alpha(frac: f32) -> f32 {
    0.1 + 0.7 * frac.clamp(0., 1.)
}
