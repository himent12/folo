//! Overview: the live run, the machine, recent runs, best models and datasets at a glance.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::*;
use crate::api::time;
use crate::api::{Run, RunStatus};
use crate::app::{FoloApp, Route, act};
use crate::charts::{LineChart, Series, line_chart, sparkline_in};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Width of the right-hand column on both Overview rows, so their gutters line up.
const SIDE_W: f32 = 360.;

impl FoloApp {
    pub fn overview(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let n_ds = self.store.datasets.ready().map(|d| d.len());
        let n_models = self.store.models.ready().map(|m| m.len());
        let n_live =
            self.store.runs.ready().map(|r| r.iter().filter(|r| r.status == RunStatus::Running).count());
        let subtitle = match (n_ds, n_models, n_live) {
            (Some(d), Some(m), Some(l)) => {
                let live = match l {
                    0 => "nothing training".to_string(),
                    1 => "1 run training".to_string(),
                    n => format!("{n} runs training"),
                };
                format!("{d} datasets · {m} models · {live}")
            }
            // The error panel below says it; the subtitle doesn't repeat it.
            _ if self.store.runs.error().is_some() => String::new(),
            _ => "Loading…".to_string(),
        };
        // Training needs a server and a dataset; until then the action explains itself instead.
        let no_ds = self.store.datasets.ready().is_some_and(|d| d.is_empty());
        let blocked = if self.store.runs.error().is_some() {
            Some("Connect to folo serve first")
        } else if no_ds {
            Some("Add a dataset first")
        } else {
            None
        };
        let new_run = match blocked {
            Some(reason) => self
                .button_disabled("ov-new", "New run", Btn::Primary, Some("icons/plus.svg"), reason, cx)
                .into_any_element(),
            None => self
                .button(
                    "ov-new",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_run(None, w, cx)),
                )
                .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
                .into_any_element(),
        };
        let actions = vec![
            self.button(
                "ov-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, _, cx| this.pick_image(cx)),
            )
            .tooltip(crate::controls::tip("Run a model on an image", Some("Ctrl O")))
            .into_any_element(),
            new_run,
        ];
        let nothing_yet = no_ds
            && self.store.runs.ready().is_some_and(|r| r.is_empty())
            && self.store.models.ready().is_some_and(|m| m.is_empty());
        // First run: the getting-started panel owns the only call to action.
        let actions = if nothing_yet { Vec::new() } else { actions };
        let header = page_header(&t, "Overview", text(ty::BASE, t.text_secondary).child(subtitle), actions);

        if let Some(e) = self.store.runs.error().cloned() {
            let err = self.error_panel(&e, "ov-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }

        if nothing_yet {
            let body = self.getting_started(window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let live = self.store.live_run().cloned();
        let hero = match &live {
            Some(r) => self.live_panel(r, window, cx),
            None => self.idle_panel(window, cx),
        };
        // Below this the System column would squeeze the training charts; it becomes a strip.
        let narrow = window.viewport_size().width < px(1360.);
        let (system, strip) = if narrow {
            (div().into_any_element(), Some(self.system_strip(cx)))
        } else {
            (self.system_panel(cx), None)
        };
        let recent = self.recent_runs(window, cx);
        let models = self.top_models(window, cx);
        let datasets = self.dataset_strip(window, cx);
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(div().flex_1().min_w_0().flex().child(hero))
                    .when(!narrow, |d| d.child(div().w(px(SIDE_W)).flex_none().flex().child(system))),
            )
            .when_some(strip, |d, s| d.child(s))
            // Narrow windows stack the two lists full width rather than squeeze the runs table.
            .map(|d| {
                if window.viewport_size().width < px(1200.) {
                    d.child(div().flex().child(recent)).child(div().flex().child(models))
                } else {
                    d.child(
                        div().flex().gap(px(16.)).child(div().flex_1().min_w_0().flex().child(recent)).child(
                            div().w(px(if narrow { 380. } else { SIDE_W })).flex_none().flex().child(models),
                        ),
                    )
                }
            })
            .child(datasets);
        self.page_frame(&t, header, body, cx)
    }

    fn live_panel(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        // Short windows trade chart height for the sections below.
        let hero_h = if window.viewport_size().height < px(700.) { 100. } else { 132. };
        let t = theme(cx);
        self.ensure_run_data(&r.id, cx);
        let ds = self
            .store
            .dataset_summary(&r.config.dataset_id)
            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
        let now = self.store.api.now();
        let started = r
            .started
            .as_deref()
            .and_then(time::parse)
            .map_or(String::new(), |s| format!("started {}", time::ago(s, now)));
        let device = crate::ui::short_gpu(r.device.split(" (").next().unwrap_or(&r.device));
        let id = r.id.clone();
        let open = self.button(
            "ov-dash",
            "Open dashboard",
            Btn::Secondary,
            Some("icons/arrow-up-right.svg"),
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
        );
        let data = self
            .store
            .run_data
            .get(&r.id)
            .and_then(|l| l.ready())
            .map(|d| (d.derived(), d.last_step().cloned(), d.last_epoch().cloned()));
        let best = r.best;
        let (charts, ips, map_now) = match data {
            Some((d, step, epoch)) => {
                let hover = self.chart_hover;
                let on_hover: crate::charts::HoverFn = {
                    let view = cx.entity().downgrade();
                    Rc::new(move |f, _, cx| {
                        view.update(cx, |this, cx| {
                            if this.chart_hover != f {
                                this.chart_hover = f;
                                cx.notify();
                            }
                        })
                        .ok();
                    })
                };
                let total = r.epochs as f32;
                let mut loss = LineChart::new(
                    "ov-loss",
                    vec![
                        Series::new("total", t.loss_colors()[3], d.loss[3].clone())
                            .smooth(0.92)
                            .area()
                            .live(true),
                    ],
                );
                loss.x_range = Some((0., total));
                loss.x_int = true;
                let loss_max = d.loss[3].iter().map(|p| p.1).fold(0f32, f32::max);
                loss.y_range = Some((0., crate::charts::nice_ceil((loss_max * 1.05).max(0.1), loss.y_ticks)));
                loss.height = hero_h;
                loss.hover = hover;
                loss.on_hover = Some(on_hover.clone());
                loss.x_name = "epoch";
                loss.value_fmt = Some(|v| format!("{v:.3}"));
                let mut map = LineChart::new(
                    "ov-map",
                    vec![
                        Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().live(true),
                        Series::new("mAP50", t.chart_series[1], d.map50.clone()).width(1.25),
                    ],
                );
                map.x_range = Some((0., total));
                map.x_int = true;
                map.y_range = Some((0., 1.));
                map.height = hero_h;
                map.hover = hover;
                map.on_hover = Some(on_hover);
                map.x_name = "epoch";
                map.value_fmt = Some(|v| format!("{v:.3}"));
                let charts = div()
                    .flex()
                    .gap(px(16.))
                    .child(chart_block(&t, "Training loss", "total · smoothed", None, line_chart(&t, loss)))
                    .child(chart_block(
                        &t,
                        "Validation mAP",
                        "",
                        Some(legend(&t, &[("mAP50-95", t.chart_series[0]), ("mAP50", t.chart_series[1])])),
                        line_chart(&t, map),
                    ))
                    .into_any_element();
                (charts, step.map(|s| s.imgs_per_s), epoch.map(|e| e.val.map50_95))
            }
            None => (div().h(px(150.)).child(skeleton(&t, "ov-charts")).into_any_element(), None, None),
        };
        let eta_s = r.eta_s;
        let best_line =
            best.map(|b| format!("best {} · epoch {}", f3(b.map50_95), b.epoch)).unwrap_or_default();
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .p(px(16.))
            .gap(px(16.))
            .child(
                div()
                    .flex()
                    .items_start()
                    .justify_between()
                    .gap(px(12.))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.))
                            .min_w_0()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(10.))
                                    .child(text(ty::LG, t.text_primary).truncate().child(r.name.clone()))
                                    .child(status_badge(&t, r.status, &r.id)),
                            )
                            .child(
                                text(ty::SM, t.text_tertiary).truncate().child(format!(
                                    "{ds} · {} · {device} · {started}",
                                    r.config.arch.as_str()
                                )),
                            ),
                    )
                    .child(open),
            )
            .child(
                div()
                    .flex()
                    .child(div().flex_1().min_w_0().child(stat(
                        &t,
                        "Epoch",
                        r.epoch.to_string(),
                        &format!("/ {}", r.epochs),
                        None,
                    )))
                    .child(vline(&t))
                    .child(div().flex_1().min_w_0().pl(px(16.)).child(stat(
                        &t,
                        "mAP50-95",
                        map_now.map_or("–".into(), f3),
                        "",
                        Some(text(ty::XS, t.text_tertiary).tnum().child(best_line)),
                    )))
                    .child(vline(&t))
                    .child(div().flex_1().min_w_0().pl(px(16.)).child(stat(
                        &t,
                        "Throughput",
                        ips.map_or("–".into(), |v| format!("{v:.0}")),
                        "img/s",
                        None,
                    )))
                    .child(vline(&t))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .pl(px(16.))
                            .flex()
                            .flex_col()
                            .gap(px(6.))
                            .child(caption(&t, "Time left"))
                            .child(duration_metric(&t, eta_s, ty::X2XL, ty::SM)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(div().flex_1().child(progress(&t, r.progress(), t.accent_fill(), 4.)))
                    .child(
                        text(ty::XS, t.text_secondary).tnum().child(format!("{:.0}%", r.progress() * 100.)),
                    ),
            )
            .child(charts)
            .into_any_element()
    }

    fn idle_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        if self.store.runs.is_loading() {
            let block = |id: &str, w: f32, h: f32| {
                div().w(gpui::relative(w)).h(px(h)).child(skeleton(&t, id.to_string()))
            };
            return panel(&t)
                .flex_1()
                .p(px(16.))
                .flex()
                .flex_col()
                .gap(px(16.))
                .child(block("hero-title", 0.4, 22.))
                .child(div().flex().gap(px(16.)).children(
                    (0..4).map(|i| div().flex_1().h(px(52.)).child(skeleton(&t, format!("hero-kpi-{i}")))),
                ))
                .child(block("hero-progress", 1.0, 4.))
                .child(
                    div()
                        .flex()
                        .gap(px(16.))
                        .child(div().flex_1().h(px(150.)).child(skeleton(&t, "hero-c1")))
                        .child(div().flex_1().h(px(150.)).child(skeleton(&t, "hero-c2"))),
                )
                .into_any_element();
        }
        let has_ds = self.store.datasets.ready().is_some_and(|d| !d.is_empty());
        let (title, body) = if has_ds {
            ("Nothing is training", "Start a run to watch loss, mAP and throughput update live here.")
        } else {
            (
                "No datasets yet",
                "Register a YOLO or COCO folder, or generate a synthetic shapes dataset to try FOLO end to end.",
            )
        };
        let action = if has_ds {
            self.button(
                "ov-empty-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            )
        } else {
            self.button(
                "ov-empty-ds",
                "Add a dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Datasets, w, cx)),
            )
        };
        panel(&t)
            .flex_1()
            .flex()
            .items_center()
            .justify_center()
            .child(
                empty_state(
                    &t,
                    if has_ds { "icons/activity.svg" } else { "icons/images.svg" },
                    title,
                    body,
                    Tone::Neutral,
                )
                .child(action),
            )
            .into_any_element()
    }

    fn system_panel(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let body = match &self.store.system {
            Load::Ready(s) => {
                let gpu = s.gpus.first();
                let util = gpu.and_then(|g| g.util).unwrap_or(0.0);
                let hist: Vec<f32> = self.store.gpu_util.iter().copied().collect();
                let mem = s.memory.used_bytes as f32 / s.memory.total_bytes.max(1) as f32;
                let (used, _) = bytes(s.memory.used_bytes);
                let (total, unit) = bytes(s.memory.total_bytes);
                let row = |label: &str, value: String, frac: f32, color| {
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .child(
                            div()
                                .flex()
                                .justify_between()
                                .child(text(ty::SM, t.text_secondary).child(label.to_string()))
                                .child(text(ty::SM, t.text_primary).tnum().child(value)),
                        )
                        .child(progress(&t, frac, color, 4.))
                };
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap(px(14.))
                    .p(px(16.))
                    .when_some(gpu, |d, g| {
                        let (vram, vu) = bytes(g.memory_bytes);
                        d.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(10.))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .justify_between()
                                        .gap(px(8.))
                                        .child(
                                            text(ty::BASE, t.text_primary)
                                                .font_weight(FontWeight::MEDIUM)
                                                .truncate()
                                                .child(crate::ui::short_gpu(&g.name)),
                                        )
                                        .child(badge(&t, g.backend.label(), Tone::Neutral)),
                                )
                                .child(metric(&t, format!("{:.0}", util * 100.), "%", ty::X3XL, ty::LG))
                                .child(
                                    text(ty::XS, t.text_tertiary)
                                        .tnum()
                                        .child(format!("GPU utilisation · {vram} {vu} GPU memory")),
                                ),
                        )
                    })
                    .child(
                        // Utilisation history fills the space the hero gives this panel.
                        div()
                            .flex_1()
                            .min_h(px(56.))
                            .flex()
                            .flex_col()
                            .gap(px(6.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_h(px(40.))
                                    .rounded(px(radius::MD))
                                    .bg(t.bg_inset)
                                    .p(px(6.))
                                    .when(hist.len() <= 1, |d| {
                                        d.flex()
                                            .items_center()
                                            .justify_center()
                                            .child(text(ty::XS, t.text_tertiary).child("Collecting…"))
                                    })
                                    .when(hist.len() > 1, |d| {
                                        d.child(sparkline_in(
                                            &t,
                                            Arc::new(hist.clone()),
                                            t.text_secondary,
                                            0.,
                                            true,
                                            Some((0., 1.)),
                                        ))
                                    }),
                            )
                            .child(
                                div()
                                    .flex()
                                    .justify_between()
                                    .child(text(ty::XS, t.text_tertiary).child("Utilisation"))
                                    .child(
                                        text(ty::XS, t.text_tertiary)
                                            .tnum()
                                            .child(format!("last {} min", (hist.len() * 2).div_ceil(60))),
                                    ),
                            ),
                    )
                    .child(hairline(&t))
                    .child(row(
                        &format!("CPU · {} threads", s.cpu.threads),
                        format!("{:.0}%", s.cpu.util * 100.),
                        s.cpu.util,
                        t.text_tertiary,
                    ))
                    .child(row("Memory", format!("{used} / {total} {unit}"), mem, t.text_tertiary))
                    .into_any_element()
            }
            Load::Failed(e) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            _ => loading_rows(&t, "sys", 4).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .child(panel_header(&t, "System"))
            .child(body)
            .into_any_element()
    }

    /// The machine in one line, for narrow windows: GPU, CPU and memory with small meters.
    fn system_strip(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(s) = self.store.system.ready().cloned() else {
            return panel(&t).h(px(48.)).child(skeleton(&t, "sys-strip")).into_any_element();
        };
        let cell = |label: String, value: String, frac: f32, color: gpui::Hsla| {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(10.))
                .px(px(16.))
                .child(text(ty::SM, t.text_secondary).flex_none().whitespace_nowrap().child(label))
                .child(div().flex_1().min_w(px(80.)).child(progress(&t, frac, color, 4.)))
                .child(text(ty::SM, t.text_primary).flex_none().tnum().whitespace_nowrap().child(value))
        };
        let mut row = div().h(px(48.)).flex().items_center();
        if let Some(g) = s.gpus.first() {
            let util = g.util.unwrap_or(0.0);
            row = row
                .child(cell(
                    crate::ui::short_gpu(&g.name),
                    format!("{:.0}%", util * 100.),
                    util,
                    // A meter, not an action: neutral like CPU and memory, amber when saturated.
                    if util > 0.95 { t.warning_solid } else { t.text_tertiary },
                ))
                .child(vline(&t));
        }
        let mem = s.memory.used_bytes as f32 / s.memory.total_bytes.max(1) as f32;
        let (used, _) = bytes(s.memory.used_bytes);
        let (total, unit) = bytes(s.memory.total_bytes);
        row = row
            .child(cell("CPU".into(), format!("{:.0}%", s.cpu.util * 100.), s.cpu.util, t.text_tertiary))
            .child(vline(&t))
            .child(cell("Memory".into(), format!("{used} / {total} {unit}"), mem, t.text_tertiary));
        panel(&t).child(row).into_any_element()
    }

    fn recent_runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let now = self.store.api.now();
        let all = self.view_all("ov-runs-all", Route::Runs, window, cx);
        let header = panel_header(&t, "Recent runs")
            .child(div().flex().items_center().gap(px(8.)).child(caption(&t, "best mAP50-95")).child(all));
        let body = match &self.store.runs {
            Load::Ready(runs) if runs.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/activity.svg",
                    "No runs yet",
                    "Training runs appear here with their status and best mAP.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(runs) => {
                let runs: Vec<Run> = runs.iter().take(5).cloned().collect();
                div()
                    .flex()
                    .flex_col()
                    .py(px(4.))
                    .children(runs.into_iter().map(|r| {
                        let ds = self
                            .store
                            .dataset_summary(&r.config.dataset_id)
                            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
                        let when = time::parse(&r.created).map_or(String::new(), |c| time::ago(c, now));
                        // Two fixed columns: progress (training runs only) and best mAP50-95.
                        let training = r.status.is_active() && r.status != crate::api::RunStatus::Queued;
                        let right = div()
                            .flex()
                            .items_center()
                            .gap(px(16.))
                            .child(div().w(px(120.)).flex().items_center().gap(px(8.)).when(training, |d| {
                                d.child(div().flex_1().child(progress(&t, r.progress(), t.accent_fill(), 4.)))
                                    .child(
                                        text(ty::XS, t.text_secondary)
                                            .tnum()
                                            .w(px(30.))
                                            .text_right()
                                            .child(format!("{:.0}%", r.progress() * 100.)),
                                    )
                            }))
                            .child(
                                text(ty::SM, if r.best.is_some() { t.text_primary } else { t.text_tertiary })
                                    .tnum()
                                    .w(px(48.))
                                    .text_right()
                                    .child(r.best.as_ref().map_or("–".to_string(), |b| f3(b.map50_95))),
                            );
                        let id = r.id.clone();
                        let key = format!("ov-run-{}", r.id);
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .h(px(48.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            // Same order as the Runs table: name, then status.
                            .child(
                                cell2(&t, r.name.clone(), format!("{ds} · {}", r.config.arch.as_str()))
                                    // A fixed column (statuses line up); the spacer after them takes the rest.
                                    .w(px(170.))
                                    .flex_none(),
                            )
                            .child(div().w(px(96.)).flex_none().flex().child(status_badge(
                                &t,
                                r.status,
                                &format!("ov-{}", r.id),
                            )))
                            .child(div().flex_1())
                            .child(right)
                            .child(
                                text(ty::XS, t.text_tertiary)
                                    .w(px(80.))
                                    .flex_none()
                                    .whitespace_nowrap()
                                    .text_right()
                                    .child(when),
                            );
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-runs", 5).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(header)
            .child(body)
            .into_any_element()
    }

    fn top_models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-models-all", Route::Models, window, cx);
        let header = panel_header(&t, "Best models")
            .child(div().flex().items_center().gap(px(8.)).child(caption(&t, "by mAP50-95")).child(all));
        let body = match &self.store.models {
            Load::Ready(models) if models.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/box.svg",
                    "No models yet",
                    "Finished runs save their best weights here.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(models) => {
                let mut models = models.clone();
                models.sort_by(|a, b| {
                    b.metrics
                        .as_ref()
                        .map_or(0., |m| m.map50_95)
                        .total_cmp(&a.metrics.as_ref().map_or(0., |m| m.map50_95))
                });
                div()
                    .flex()
                    .flex_col()
                    .py(px(4.))
                    .children(models.into_iter().take(4).map(|m| {
                        let id = m.id.clone();
                        let key = format!("ov-model-{}", m.id);
                        let map = m.metrics.as_ref().map_or(0., |m| m.map50_95);
                        let (p, pu) = count_si(m.params);
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .h(px(48.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .child(
                                cell2(
                                    &t,
                                    m.name.clone(),
                                    format!("{} · {} px · {p} {pu} params", m.arch.as_str(), m.imgsz),
                                )
                                .flex_1(),
                            )
                            .child(bar_value(&t, map, f3(map), t.text_tertiary));
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-models", 4).into_any_element(),
        };
        panel(&t).flex_1().flex().flex_col().child(header).child(body).into_any_element()
    }

    fn dataset_strip(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-ds-all", Route::Datasets, window, cx);
        let datasets = self.store.datasets.ready().cloned().unwrap_or_default();
        let cards: Vec<AnyElement> =
            datasets.iter().take(3).map(|d| self.dataset_card(d, window, cx)).collect();
        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Datasets").child(all))
            .map(|d| {
                if cards.is_empty() && self.store.datasets.ready().is_some() {
                    d.child(empty_state(
                        &t,
                        "icons/images.svg",
                        "No datasets registered",
                        "Datasets you add or generate show up here with a preview of their images.",
                        Tone::Neutral,
                    ))
                } else if cards.is_empty() {
                    d.child(div().flex().gap(px(16.)).children(
                        (0..3).map(|i| div().flex_1().h(px(196.)).child(skeleton(&t, format!("ov-ds-{i}")))),
                    ))
                } else {
                    d.child(div().p(px(16.)).flex().gap(px(16.)).children(cards))
                }
            })
            .into_any_element()
    }

    /// First run: what FOLO needs, in order, with the next step as the one primary action.
    fn getting_started(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let add = self
            .button(
                "gs-add",
                "Add a dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Datasets, w, cx)),
            )
            .into_any_element();
        let step = |n: &str, title: &str, body: &str, active: bool| {
            div()
                .flex()
                .gap(px(14.))
                .p(px(16.))
                .rounded(px(radius::LG))
                .border_1()
                .border_color(if active { t.line_strong } else { t.line_subtle })
                .when(active, |d| d.bg(t.bg_raised))
                .child(
                    div()
                        .size(px(24.))
                        .flex_none()
                        .rounded_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(if active { t.accent_fill() } else { t.bg_active })
                        .child(
                            text(ty::XS, if active { t.accent_on } else { t.text_tertiary })
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(n.to_string()),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(
                            text(ty::BASE, if active { t.text_primary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .child(title.to_string()),
                        )
                        .child(rich(&t, ty::SM, t.text_tertiary, body).justify_start()),
                )
        };
        panel(&t)
            .p(px(24.))
            .flex()
            .flex_col()
            .gap(px(20.))
            .child(
                div().flex().items_center().gap(px(14.)).child(crate::app::logo(&t, 32.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(text(ty::LG, t.text_primary).child("Get started with FOLO"))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .child("Three steps from images to a model you can run."),
                        ),
                ),
            )
            .child(
                div()
                    .flex()
                    .gap(px(12.))
                    .child(div().flex_1().child(step(
                        "1",
                        "Add a dataset",
                        "Point FOLO at a YOLO folder, or generate one with `folo dataset synth`",
                        true,
                    )))
                    .child(div().flex_1().child(step(
                        "2",
                        "Train a model",
                        "Start a run; loss, mAP and throughput stream in live.",
                        false,
                    )))
                    .child(div().flex_1().child(step(
                        "3",
                        "Try it",
                        "Open any image in the Playground and tune the thresholds.",
                        false,
                    ))),
            )
            .child(div().flex().child(add))
            .into_any_element()
    }

    /// Dataset card: a 2×2 mosaic of its first images, name and counts.
    pub fn dataset_card(
        &mut self,
        d: &crate::api::Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_dataset(&d.id, cx);
        let metas: Vec<crate::api::ImageInfo> = self
            .store
            .images
            .get(&d.id)
            .and_then(|l| l.ready())
            .map(|v| v.iter().filter(|m| !m.boxes.is_empty()).take(4).cloned().collect())
            .unwrap_or_default();
        let tiles: Vec<AnyElement> = (0..4)
            .map(|i| match metas.get(i) {
                Some(m) => {
                    let tile = self.tile(&d.id, m, true, crate::viz::Fit::Cover, cx);
                    div().flex_1().h_full().overflow_hidden().child(tile).into_any_element()
                }
                None => div().flex_1().h_full().bg(t.bg_raised).into_any_element(),
            })
            .collect();
        let mut it = tiles.into_iter();
        let row1: Vec<AnyElement> = it.by_ref().take(2).collect();
        let row2: Vec<AnyElement> = it.collect();
        let id = d.id.clone();
        let key = format!("ds-card-{}", d.id);
        let card = div()
            .id(SharedString::from(key.clone()))
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .rounded(px(radius::LG))
            .bg(t.bg_panel)
            .border_1()
            .border_color(t.line_default)
            .overflow_hidden()
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_strong))
            .child(
                div()
                    .h(px(136.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .bg(t.bg_inset)
                    .child(div().flex_1().flex().gap(px(1.)).children(row1))
                    .child(div().flex_1().flex().gap(px(1.)).children(row2)),
            )
            .child(
                div()
                    .p(px(12.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(cell2(
                        &t,
                        d.name.clone(),
                        format!("{} images · {} classes", int(d.splits.total()), d.classes.len()),
                    ))
                    .child(badge(&t, d.format.label(), Tone::Neutral)),
            );
        self.bind(
            card,
            &key,
            radius::LG,
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Dataset(id.clone()), w, cx)),
        )
        .into_any_element()
    }

    pub fn view_all(&self, key: &str, route: Route, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .flex()
            .items_center()
            .gap(px(4.))
            .px(px(6.))
            .mr(px(-6.))
            .h(px(22.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .child(text(ty::XS, t.text_secondary).child("View all"))
            .child(icon("icons/chevron-right.svg", 12., t.text_tertiary));
        self.bind(
            el,
            key,
            radius::SM,
            window,
            cx,
            act(move |this, w, cx| this.navigate(route.clone(), w, cx)),
        )
        .into_any_element()
    }
}

/// A chart with a small title row.
pub fn chart_block(
    t: &crate::theme::Theme,
    title: &str,
    detail: &str,
    legend: Option<AnyElement>,
    chart: AnyElement,
) -> gpui::Div {
    div()
        .flex_1()
        .min_w_0()
        .flex()
        .flex_col()
        .gap(px(8.))
        .child(
            div()
                .flex()
                .items_baseline()
                .gap(px(6.))
                .child(
                    text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(title.to_string()),
                )
                .child(text(ty::XS, t.text_tertiary).flex_1().child(detail.to_string()))
                .children(legend),
        )
        .child(chart)
}
