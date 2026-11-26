//! A model's page: its header, metrics and evaluation layout, and the export menu.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Model, ModelEval, ModelMetrics, time};
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
                let all = self.button(
                    "model-all",
                    "All models",
                    Btn::Secondary,
                    Some("icons/arrow-left.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.navigate(Route::Models, w, cx)),
                );
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Model not found",
                        "It may have been deleted or moved.",
                        Tone::Neutral,
                    ))
                    .child(div().pb(px(24.)).flex().justify_center().child(all))
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
            .w_full()
            .flex()
            .items_center()
            .gap(px(10.))
            .min_w_0()
            .child(text(ty::XL, t.text_primary).min_w_0().ellipsis().child(m.name.clone()))
            .child(div().flex_none().child(badge(&t, m.arch.as_str(), Tone::Neutral)))
            // The same compact tag as every list, right after the architecture badge, its help on
            // hover; the Details panel's Head row gives it a keyboard route.
            .when(e2e, |d| d.child(div().flex_none().child(e2e_tag(&t, "e2e-model-head"))));
        let mut subtitle =
            div().flex().items_center().gap(px(6.)).child(text(ty::BASE, t.text_secondary).child(format!(
                "{} px · {}{}{} · created {created}",
                m.imgsz,
                counted(m.classes.len() as u64, "class", "classes"),
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
        // An imported or published file's stored numbers: no dataset named (`headline_source`).
        let head_ds = headline_source(
            &m,
            self.store.evals.get(&m.id).and_then(|l| l.ready()),
            run.as_ref()
                .and_then(|r| self.store.dataset_summary(&r.config.dataset_id))
                .map(|d| (d.name.clone(), if d.splits.val == 0 { "train" } else { "val" })),
        );
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
        let small = |label: &str, value: String, unit: &str, detail: Option<folo_ui::Div>| {
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .min_w_0()
                .child(kpi_label(&t, label))
                .child(
                    div().h(px(KPI_HEADLINE.line)).flex().items_end().child(
                        metric(&t, value, unit, KPI_VALUE, KPI_UNIT)
                            .pb(px(baseline_lift(KPI_HEADLINE, KPI_VALUE))),
                    ),
                )
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
                            .child(kpi_label(&t, "mAP50-95"))
                            .child(
                                text(KPI_HEADLINE, t.text_primary)
                                    .h(px(KPI_HEADLINE.line))
                                    .tnum()
                                    .child(mm.as_ref().map_or("–".into(), |x| f3(x.map50_95))),
                            )
                            .child(kpi_note(&t).whitespace_nowrap().ellipsis().child(match &head_ds {
                                Some((d, split)) => {
                                    format!("IoU 0.50–0.95 · {}", split_label(d, split, None))
                                }
                                None => "IoU 0.50–0.95".to_string(),
                            })),
                    ),
            )
            .child(vline(&t))
            .child(tile(small(
                "mAP50",
                mm.as_ref().map_or("–".into(), |x| f3(x.map50)),
                "",
                Some(kpi_note(&t).child("IoU 0.50")),
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
                    .child(kpi_label(&t, "Precision / Recall"))
                    .child(
                        div().h(px(KPI_HEADLINE.line)).flex().items_end().child(
                            metric_pair(
                                &t,
                                KPI_VALUE,
                                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.precision)),
                                mm.as_ref().map_or("–".into(), |x| format!("{:.3}", x.recall)),
                            )
                            .pb(px(baseline_lift(KPI_HEADLINE, KPI_VALUE))),
                        ),
                    )
                    .child(kpi_note(&t).ellipsis().child(op_point.clone())),
            ))
            .child(vline(&t))
            .child(tile(small(
                "Parameters",
                p,
                pu,
                Some(kpi_note(&t).child(format!("{:.1} GFLOPs", m.flops_g))),
            )))
            .child(vline(&t))
            .child(tile(small(
                "File size",
                b,
                bu,
                Some(
                    kpi_note(&t)
                        // The record's stored precision (as the header says), never a guess from size.
                        // The format; the stored precision is the subtitle's to say (once).
                        .child(".folo"),
                ),
            )));

        let body = match self.store.evals.get(id) {
            Some(Load::Ready(ev)) => {
                let ev = ev.clone();
                // The precision-recall panel (the plot and its class table), the confusion matrix and
                // its lists, then the Details: each a full-width panel.
                let pr = self.pr_panel(&m, &ev, window, cx);
                let cm = self.confusion_panel(&m, &ev, window, cx);
                let details = self.details_panel(&m, Some(&ev), window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(div().flex().child(pr))
                    .child(div().flex().child(cm))
                    .child(details)
                    .into_any_element()
            }
            Some(Load::Failed(e)) if e.code == "not_supported" => {
                let details = self.details_panel(&m, None, window, cx);
                div()
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
                    .child(details)
                    .into_any_element()
            }
            Some(Load::Failed(e)) => {
                let e = e.clone();
                let details = self.details_panel(&m, None, window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(self.error_panel(&e, "eval-retry", window, cx))
                    .child(details)
                    .into_any_element()
            }
            _ => {
                let details = self.details_panel(&m, None, window, cx);
                div()
                    .flex()
                    .flex_col()
                    .gap(px(16.))
                    .child(kpis)
                    .child(div().h(px(360.)).child(skeleton(&t, "eval")))
                    .child(details)
                    .into_any_element()
            }
        };
        self.page_frame(&t, header, body, cx)
    }

    fn export_menu(&mut self, m: &Model, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let open = self.export_open;
        let any_busy = self.exporting.iter().any(|(id, _)| id == &m.id);
        // Estimates from the parameter count: 2 bytes per weight in f16, 4 in ONNX (f32).
        let (half, hu) = bytes(m.params * 2);
        let (onnx, ou) = bytes(m.params * 4);
        let options = [
            ("folo-f16", "FOLO · f16", format!("≈ {half} {hu} · half-precision weights for GPU inference")),
            ("onnx", "ONNX", format!("≈ {onnx} {ou} · f32, opset 17, for other runtimes")),
        ];
        let n = options.len();
        let fmts: Vec<&'static str> = options.iter().map(|o| o.0).collect();
        let usable: Vec<usize> =
            (0..n).filter(|&i| !self.exporting.contains(&(m.id.clone(), fmts[i]))).collect();
        let first_usable = usable.first().copied().unwrap_or(0);
        let last_usable = usable.last().copied().unwrap_or(0);
        let hl = if usable.contains(&self.export_hl) { self.export_hl } else { first_usable };
        let step_usable = usable.clone();
        let step = move |i: usize, down: bool| {
            let next = if down {
                step_usable.iter().find(|&&u| u > i)
            } else {
                step_usable.iter().rev().find(|&&u| u < i)
            };
            next.copied().unwrap_or(i)
        };
        let mid = m.id.clone();
        let key_usable = usable.clone();
        // Like a select: focus stays on the trigger, arrows move a highlight, Enter commits it.
        let trigger = self
            .button(
                "model-export",
                if any_busy { "Exporting…" } else { "Export" },
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    let opening = !this.export_open;
                    this.export_open = opening;
                    this.menu_trigger.set(None);
                    if opening {
                        this.menu_open = None;
                        this.menu_typeahead.clear();
                    }
                    this.export_hl = first_usable;
                    cx.notify();
                }),
            )
            // Captured before the button's own Enter/Space toggle.
            .capture_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, _, cx| {
                if this.export_open && matches!(ev.keystroke.key.as_str(), "enter" | "space") {
                    if key_usable.is_empty() {
                        this.export_open = false;
                        this.menu_trigger.set(None);
                    } else {
                        let pick =
                            if key_usable.contains(&this.export_hl) { this.export_hl } else { first_usable };
                        if let Some(&fmt) = fmts.get(pick)
                            && !this.exporting.contains(&(mid.clone(), fmt))
                        {
                            this.export_model(mid.clone(), fmt, cx);
                        }
                    }
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
                        this.menu_trigger.set(None);
                        this.menu_open = None;
                        this.menu_typeahead.clear();
                        this.export_hl = first_usable;
                    }
                    "down" => this.export_hl = step(this.export_hl.min(n - 1), true),
                    "up" => this.export_hl = step(this.export_hl.min(n - 1), false),
                    "home" if open => this.export_hl = first_usable,
                    "end" if open => this.export_hl = last_usable,
                    "escape" if open => {
                        this.export_open = false;
                        this.menu_trigger.set(None);
                    }
                    "tab" if open => {
                        this.export_open = false;
                        this.menu_trigger.set(None);
                        cx.notify();
                        return;
                    }
                    _ => return,
                }
                cx.stop_propagation();
                cx.notify();
            }));
        let export_w = (f32::from(window.viewport_size().width) - 16.).clamp(180., 300.);
        let menu = open.then(|| {
            let mid = m.id.clone();
            folo_ui::deferred(
                div()
                    .absolute()
                    .top(px(34.))
                    .right_0()
                    .w(px(export_w))
                    .occlude()
                    .on_mouse_down_out(cx.listener(|this, ev: &folo_ui::MouseDownEvent, _, cx| {
                        let on_trigger = this.menu_trigger.get().is_some_and(|b| b.contains(&ev.position));
                        if this.export_open && !on_trigger {
                            this.export_open = false;
                            this.menu_trigger.set(None);
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
                        let busy = self.exporting.contains(&(id.clone(), fmt));
                        div()
                            .id(SharedString::from(format!("export-{fmt}")))
                            .px(px(10.))
                            .py(px(8.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .rounded(px(radius::MD))
                            .when(busy, |d| d.cursor_not_allowed().opacity(0.65))
                            .when(!busy, |d| {
                                d.cursor_pointer()
                                    .when(i == hl, |d| d.bg(t.bg_hover))
                                    .hover(|s| s.bg(t.bg_hover))
                                    .active(|s| s.bg(t.bg_active))
                                    .on_mouse_move(cx.listener(
                                        move |this, _: &folo_ui::MouseMoveEvent, _, cx| {
                                            if this.export_hl != i {
                                                this.export_hl = i;
                                                cx.notify();
                                            }
                                        },
                                    ))
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        this.export_model(id.clone(), fmt, cx)
                                    }))
                            })
                            .child(
                                text(ty::BASE, if busy { t.text_tertiary } else { t.text_primary })
                                    .font_weight(FontWeight::MEDIUM)
                                    .child(if busy {
                                        format!("{label} · Exporting…")
                                    } else {
                                        label.into()
                                    }),
                            )
                            .child(text(ty::SM, t.text_tertiary).tnum().child(detail))
                            .into_any_element()
                    })),
            )
            .with_priority(2)
        });
        div().relative().child(trigger).children(menu).into_any_element()
    }
}

/// The dataset and split the headline (the model's stored metrics) was measured on: the evaluation's
/// as it reports them, else the run's dataset. None for a model no run made (an imported or published
/// file): its stored numbers do not say what they were measured on, and this server's evaluation
/// (picked by class match, maybe on another dataset) is not theirs (Studio's KpiSummary `fromFile`).
pub(super) fn headline_source(
    m: &Model,
    ev: Option<&ModelEval>,
    run_ds: Option<(String, &'static str)>,
) -> Option<(String, &'static str)> {
    m.run_id.as_ref()?;
    ev.and_then(|ev| ev.dataset_name.clone().zip(ev.split.map(|s| s.as_str()))).or(run_ds)
}

/// The stored metrics shown among this server's evaluation (the "All classes" row and legend): a
/// run's own only. A file's stored numbers may be on another dataset than the curves beside them.
pub(super) fn run_metrics(m: &Model) -> Option<&ModelMetrics> {
    m.run_id.as_ref().and(m.metrics.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, Split, mock::MockBackend};

    #[test]
    fn a_files_stored_metrics_never_take_the_evaluations_dataset() {
        let api = MockBackend::new(4200, true);
        let run_model = api.models().unwrap().into_iter().find(|m| m.run_id.is_some()).unwrap();
        let file = Model { run_id: None, ..run_model.clone() };
        let ev = ModelEval {
            dataset_name: Some("other".into()),
            split: Some(Split::Val),
            ..api.model_eval(&run_model.id).unwrap()
        };
        let run_ds = || Some(("shapes".to_string(), "val"));
        // A run's model: the evaluation's own provenance, else the run's dataset.
        assert_eq!(headline_source(&run_model, Some(&ev), run_ds()), Some(("other".into(), "val")));
        assert_eq!(headline_source(&run_model, None, run_ds()), Some(("shapes".into(), "val")));
        assert!(run_metrics(&run_model).is_some());
        // A file (imported, published): neither the evaluation's dataset nor its stored numbers
        // among the curves; the headline still shows them.
        assert_eq!(headline_source(&file, Some(&ev), None), None);
        assert!(file.metrics.is_some() && run_metrics(&file).is_none());
    }

    #[test]
    fn model_page_sorts_per_class_links_run_and_dataset_and_guards_export() {
        use crate::app::tests::{drive, open};
        use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};

        let send_enter = |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<FoloApp>| {
            let ks = Keystroke::parse("enter").unwrap();
            app.simulate_input(
                w.into(),
                PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
            );
            app.run_until_idle();
            app.frame(w.into());
            app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
            app.run_until_idle();
            app.frame(w.into());
        };

        let (mut app, window) = open();
        let (mid, rid, dsid) = drive(&mut app, window, |this, w, cx| {
            let m = this.store.models.ready().unwrap().iter().find(|m| m.run_id.is_some()).unwrap().clone();
            let rid = m.run_id.clone().unwrap();
            let dsid = this.store.run(&rid).unwrap().config.dataset_id.clone();
            this.navigate(Route::Model(m.id.clone()), w, cx);
            (m.id, rid, dsid)
        });
        for _ in 0..3 {
            app.frame(window.into());
            app.run_until_idle();
        }

        // 1. Per-class table sort header toggles desc -> asc.
        drive(&mut app, window, |this, w, cx| this.focus_key("pc-sort-map50", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.class_table_sort, Some(("map50", true)));
        });
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.class_table_sort, Some(("map50", false)));
        });

        // 2. Duplicate export guard & hold_loading guard (each step drains async tasks).
        let before = drive(&mut app, window, |this, _, cx| {
            let before = this.toasts.len();
            // While hold_loading is true, export_model must not strand an in-flight flag.
            this.hold_loading = true;
            this.export_model(mid.clone(), "onnx", cx);
            assert!(this.exporting.is_empty(), "hold_loading drops export without stranding flag");
            this.hold_loading = false;
            // With onnx already in flight, a duplicate request must not dispatch or toast.
            this.exporting.insert((mid.clone(), "onnx"));
            this.export_model(mid.clone(), "onnx", cx);
            before
        });
        drive(&mut app, window, |this, _, cx| {
            assert_eq!(this.toasts.len(), before, "in-flight export blocked duplicate async request");
            assert!(this.exporting.contains(&(mid.clone(), "onnx")), "sentinel still present");
            // Opening the menu while folo-f16 is busy selects the available format (onnx = 1).
            this.exporting.remove(&(mid.clone(), "onnx"));
            this.exporting.insert((mid.clone(), "folo-f16"));
            cx.notify();
        });
        app.frame(window.into());
        drive(&mut app, window, |this, w, cx| this.focus_key("model-export", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, cx| {
            assert!(this.export_open);
            assert_eq!(this.export_hl, 1, "menu skips busy folo-f16 and highlights available onnx");
            this.export_open = false;
            this.exporting.clear();
            this.export_model(mid.clone(), "onnx", cx);
        });
        drive(&mut app, window, |this, _, _| {
            assert!(this.exporting.is_empty(), "exporting clears after completion");
            assert_eq!(this.toasts.len(), before + 1, "completed export posts one toast");
        });

        // 3. Details panel links navigate to the source run and training dataset, resetting table sort.
        drive(&mut app, window, |this, w, cx| this.focus_key("model-details-run-link", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, w, cx| {
            assert_eq!(this.route, Route::Run(rid.clone()));
            assert_eq!(this.class_table_sort, None, "navigation resets per-class sort");
            this.navigate(Route::Model(mid.clone()), w, cx);
        });
        for _ in 0..2 {
            app.frame(window.into());
            app.run_until_idle();
        }
        drive(&mut app, window, |this, w, cx| this.focus_key("model-details-train-ds-link", w, cx));
        app.frame(window.into());
        send_enter(&mut app, window);
        drive(&mut app, window, |this, _, _| {
            assert_eq!(this.route, Route::Dataset(dsid));
        });
    }
}
