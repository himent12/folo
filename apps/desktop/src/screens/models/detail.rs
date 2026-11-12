//! A model's page: its header, metrics and evaluation layout, and the export menu.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, time};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
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
                    this.play_pred = Load::Idle;
                    this.navigate(Route::Playground, w, cx);
                    this.pick_play_model(mid.clone(), cx);
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
        // Published weights (a `folo weights` download): what they are, in their own line, on hover.
        if run.is_none() && m.source == Some(crate::api::ModelSource::Published) {
            subtitle = subtitle.child(
                div()
                    .id("model-about")
                    .when_some(m.about.clone(), |d, a| d.tooltip(crate::controls::tip(a, None)))
                    .child(text(ty::BASE, t.text_tertiary).child("· published weights")),
            );
        }
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
        // The dataset and split the headline numbers were measured on, named beside them: as the
        // evaluation reports them (`ModelEval`, so a fallback to train never reads as "val"), else,
        // until it loads or from an older server, inferred from the run's dataset (train when it has
        // no validation images, as the server falls back).
        let head_ds = self
            .store
            .evals
            .get(&m.id)
            .and_then(|l| l.ready())
            .and_then(|ev| ev.dataset_name.clone().zip(ev.split.map(|s| s.as_str())))
            .or_else(|| {
                run.as_ref()
                    .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
                    .map(|d| (d.name.clone(), if d.splits.val == 0 { "train" } else { "val" }))
            });
        // Where precision and recall are taken: the best-F1 confidence, named when the record has it.
        // The headline names the split; this tile's narrow caption says where on the curve.
        let op_point = pr_caption(
            mm.as_ref().and_then(|x| operating_conf(x.best_f1_conf, x.precision, x.recall)),
            "",
            true,
        );
        let tile = |content: folo_ui::Div| {
            div()
                .flex_grow()
                .flex_basis(folo_ui::relative(0.144))
                .min_w_0()
                .px(px(16.))
                .py(px(14.))
                .child(content)
        };
        let wide_tile = |content: folo_ui::Div| {
            div()
                .flex_grow()
                .flex_basis(folo_ui::relative(0.21))
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
            |label: &str, value: String, unit: &str, detail: Option<folo_ui::Div>| {
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
                div()
                    .flex_grow()
                    .flex_basis(folo_ui::relative(0.28))
                    .min_w_0()
                    .px(px(16.))
                    .py(px(14.))
                    .child(
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
                    .child(text(ty::SM, t.text_tertiary).ellipsis().child(op_point.clone())),
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
                // width (below a ~1410 px window: content width under ~1127 px).
                let stacked = self.eval_stacked(window);
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
                                div().flex_grow().flex_basis(folo_ui::relative(0.54)).min_w_0().flex().child(pr),
                            )
                            .child(
                                div()
                                    .flex_grow()
                                    .flex_basis(folo_ui::relative(0.46))
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
            .capture_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, _, cx| {
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
                    folo_ui::canvas(move |b, _, _| cell.set(Some(b)), |_, _, _, _| {})
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full(),
                )
            })
            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, _, cx| {
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
            folo_ui::deferred(
                div()
                    .absolute()
                    .top(px(34.))
                    .right_0()
                    .w(px(300.))
                    .occlude()
                    .on_mouse_down_out(cx.listener(|this, ev: &folo_ui::MouseDownEvent, _, cx| {
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

    /// The PR panel sits above the matrix (each full width) unless it keeps 600 px beside it
    /// (46 %, 440 px minimum): its plot and side legend both fit.
    pub(super) fn eval_stacked(&self, window: &Window) -> bool {
        let row = self.content_width(window) - 16.;
        row - (row * 0.46).max(440.) < 600.
    }
}
