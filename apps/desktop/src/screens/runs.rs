//! Runs: the run list and the live training dashboard.

use std::rc::Rc;
use std::sync::Arc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, KeyDownEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::overview::chart_block;
use super::*;
use crate::api::{LogLevel, Run, RunStatus, time};
use crate::app::{FoloApp, Route, RunFilter, SampleView, act};
use crate::charts::{HoverFn, LineChart, Marker, Series, line_chart, sparkline};
use crate::controls::Btn;
use crate::store::{ImgKind, Load};
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit};

fn duration_of(r: &Run, now: i64) -> Option<f64> {
    let start = time::parse(r.started.as_deref()?)?;
    let end = r.finished.as_deref().and_then(time::parse).unwrap_or(now);
    Some((end - start) as f64)
}

impl FoloApp {
    pub fn runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let runs = self.store.runs.ready().cloned().unwrap_or_default();
        let count = |f: RunFilter| runs.iter().filter(|r| matches_filter(r, f)).count();
        let active = count(RunFilter::Active);
        let subtitle = if self.store.runs.is_loading() {
            "Loading…".to_string()
        } else {
            format!("{} runs · {} active · {} failed", runs.len(), active, count(RunFilter::Failed))
        };
        let new = if self.store.datasets.ready().is_some_and(|d| d.is_empty()) {
            self.button_disabled(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                "Add a dataset first",
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            )
            .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
            .into_any_element()
        };
        let header = page_header(&t, "Runs", subtitle, vec![new]);
        if let Some(e) = self.store.runs.error().cloned() {
            let err = self.error_panel(&e, "runs-retry", window, cx);
            return self.page_frame(&t, header, err);
        }
        let filters = [RunFilter::All, RunFilter::Active, RunFilter::Completed, RunFilter::Failed];
        let labels: Vec<String> = filters
            .iter()
            .map(|f| {
                let name = match f {
                    RunFilter::All => "All",
                    RunFilter::Active => "Active",
                    RunFilter::Completed => "Finished",
                    RunFilter::Failed => "Failed",
                };
                format!("{name}  {}", count(*f))
            })
            .collect();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let sel = filters.iter().position(|f| *f == self.run_filter).unwrap_or(0);
        let seg = self.segmented(
            "runs-filter",
            &label_refs,
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| this.run_filter = filters[i]),
        );
        let now = self.store.api.now();
        let visible: Vec<Run> = runs.into_iter().filter(|r| matches_filter(r, self.run_filter)).collect();
        let ids: Vec<String> = visible.iter().map(|r| r.id.clone()).collect();
        let cols = |d: gpui::Div| d.flex().items_center().gap(px(16.)).px(px(16.));
        let head = cols(div())
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Run").flex_1())
            .child(th(&t, "Status").w(px(96.)))
            .child(th(&t, "Progress").w(px(150.)))
            .child(th(&t, "Best mAP50-95").w(px(110.)).text_right())
            .child(th(&t, "Duration").w(px(88.)).text_right())
            .child(th(&t, "Created").w(px(84.)).text_right());
        let rows: Vec<AnyElement> = visible
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let ds = self
                    .store
                    .dataset_summary(&r.config.dataset_id)
                    .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
                let dur = duration_of(r, now).map_or("–".into(), time::duration);
                let created = time::parse(&r.created).map_or(String::new(), |c| time::ago(c, now));
                let progress_cell = match r.status {
                    RunStatus::Queued => text(ty::SM, t.text_tertiary).child("Waiting for GPU"),
                    _ => div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(div().flex_1().child(progress(
                            &t,
                            r.progress(),
                            if r.status == RunStatus::Running {
                                t.accent_fill()
                            } else if r.status == RunStatus::Failed {
                                t.danger_solid
                            } else {
                                t.text_tertiary
                            },
                            4.,
                        )))
                        .child(
                            text(ty::XS, t.text_secondary)
                                .tnum()
                                .w(px(46.))
                                .text_right()
                                .child(format!("{} / {}", r.epoch, r.epochs)),
                        ),
                };
                let best = r.best.as_ref().map_or(text(ty::SM, t.text_tertiary).child("–"), |b| {
                    text(ty::SM, t.text_primary).tnum().child(f3(b.map50_95))
                });
                let id = r.id.clone();
                let key = format!("run-row-{}", r.id);
                let ids_up = ids.clone();
                let row = div()
                    .id(SharedString::from(key.clone()))
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    .h(px(52.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .cursor_pointer()
                    .hover(|s| s.bg(t.bg_hover))
                    .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, w, cx| {
                        let next = match ev.keystroke.key.as_str() {
                            "down" => i + 1,
                            "up" if i > 0 => i - 1,
                            _ => return,
                        };
                        if let Some(id) = ids_up.get(next) {
                            this.focus_key(&format!("run-row-{id}"), w, cx);
                            cx.stop_propagation();
                        }
                    }))
                    .child(
                        cell2(
                            &t,
                            r.name.clone(),
                            format!(
                                "{} · {ds} · {} · batch {}",
                                r.config.arch.as_str(),
                                r.config.imgsz,
                                r.config.batch
                            ),
                        )
                        .flex_1(),
                    )
                    .child(div().w(px(96.)).flex().child(status_badge(
                        &t,
                        r.status,
                        &format!("row-{}", r.id),
                    )))
                    .child(div().w(px(150.)).child(progress_cell))
                    .child(div().w(px(110.)).flex().justify_end().child(best))
                    .child(text(ty::SM, t.text_secondary).tnum().w(px(88.)).text_right().child(dur))
                    .child(text(ty::SM, t.text_tertiary).w(px(84.)).text_right().child(created));
                self.bind(
                    row,
                    &key,
                    0.,
                    window,
                    cx,
                    act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                )
                .into_any_element()
            })
            .collect();
        let table = panel(&t).overflow_hidden().child(head).map(|d| {
            if self.store.runs.is_loading() {
                d.child(loading_rows(&t, "runs", 6))
            } else if rows.is_empty() {
                d.child(empty_state(
                    &t,
                    "icons/activity.svg",
                    "No runs here",
                    "Runs matching this filter will show up here.",
                    Tone::Neutral,
                ))
            } else {
                d.children(rows)
            }
        });
        let body = div().flex().flex_col().gap(px(12.)).child(div().flex().child(seg)).child(table);
        self.page_frame(&t, header, body)
    }

    pub fn run_view(&mut self, id: &str, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(r) = self.store.run(id).cloned() else {
            let header = page_header(&t, "Run", "", vec![]);
            let body = if self.store.runs.is_loading() {
                panel(&t).child(loading_rows(&t, "run", 6)).into_any_element()
            } else if let Some(e) = self.store.runs.error().cloned() {
                self.error_panel(&e, "run-retry", window, cx)
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Run not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body);
        };
        self.ensure_run_data(id, cx);
        self.ensure_logs(id, cx);
        if r.epoch > 0 {
            self.ensure_samples(id, cx);
        }
        let now = self.store.api.now();
        let ds_name = self
            .store
            .dataset_summary(&r.config.dataset_id)
            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
        let started = r.started.as_deref().and_then(time::parse).map(|s| time::ago(s, now));
        let device = crate::ui::device_label(&r.device);
        let subtitle = text(ty::BASE, t.text_secondary).child(match started {
            Some(s) => format!("{} · {ds_name} · {device} · started {s}", r.config.arch.as_str()),
            None => format!("{} · {ds_name} · {device}", r.config.arch.as_str()),
        });

        let mut actions = Vec::new();
        match r.status {
            RunStatus::Running | RunStatus::Queued => {
                let rid = r.id.clone();
                actions.push(
                    self.button(
                        "run-stop",
                        "Stop run",
                        Btn::Danger,
                        Some("icons/stop.svg"),
                        window,
                        cx,
                        act(move |this, w, cx| {
                            this.ask_stop(rid.clone(), w, cx);
                            cx.notify();
                        }),
                    )
                    .into_any_element(),
                );
            }
            RunStatus::Completed | RunStatus::Stopped => {
                if let Some(mid) = r.model_id.clone() {
                    let m2 = mid.clone();
                    actions.push(
                        self.button(
                            "run-try",
                            "Try in Playground",
                            Btn::Secondary,
                            Some("icons/scan-eye.svg"),
                            window,
                            cx,
                            act(move |this, w, cx| {
                                this.play_model = Some(m2.clone());
                                this.play_pred = Load::Idle;
                                this.navigate(Route::Playground, w, cx);
                                this.run_predict(cx);
                            }),
                        )
                        .into_any_element(),
                    );
                    actions.push(
                        self.button(
                            "run-model",
                            "Open model",
                            Btn::Primary,
                            Some("icons/box.svg"),
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Model(mid.clone()), w, cx)),
                        )
                        .into_any_element(),
                    );
                }
            }
            RunStatus::Failed => {
                // Reuse the whole config; for out-of-memory failures, halve the batch.
                let mut cfg = r.config.clone();
                let oom = r.error.as_deref().is_some_and(|e| e.contains("out of memory"));
                let note = oom.then(|| crate::app::retry_batch(cfg.batch)).flatten().map(|to| {
                    let from = cfg.batch;
                    cfg.batch = to;
                    format!("Batch lowered from {from} to {to} so the run fits in GPU memory.")
                });
                actions.push(
                    self.button(
                        "run-retry-cfg",
                        "New run from this config",
                        Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(move |this, w, cx| this.open_new_run_from(&cfg, note.clone(), w, cx)),
                    )
                    .into_any_element(),
                );
            }
            RunStatus::Stopping => {}
        }
        let title = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .child(text(ty::XL, t.text_primary).child(r.name.clone()))
            .child(status_badge(&t, r.status, &format!("hdr-{}", r.id)));
        let header = detail_header(&t, title, subtitle, actions);

        let mut body = div().flex().flex_col().gap(px(16.));
        if r.status == RunStatus::Failed {
            body = body.child(self.failure_banner(&r, cx));
        }
        let data = self.store.run_data.get(id).map(|l| match l {
            Load::Ready(d) => Ok((
                d.derived(),
                d.recent_ips(120),
                d.last_epoch().cloned(),
                d.metrics.epochs.len().checked_sub(2).map(|i| d.metrics.epochs[i].val.map50_95),
            )),
            Load::Failed(e) => Err(Some(e.clone())),
            _ => Err(None),
        });
        match data {
            Some(Ok((d, ips, last, prev))) => {
                let narrow = window.viewport_size().width < px(1360.);
                body = body.child(self.kpi_strip(&r, &ips, last.as_ref(), prev, now, narrow, cx));
                body = body.child(self.run_charts(&r, &d, window, cx));
            }
            Some(Err(Some(e))) => body = body.child(self.error_panel(&e, "run-data-retry", window, cx)),
            _ => {
                body = body.child(panel(&t).h(px(96.)).child(loading_rows(&t, "kpi", 2)));
                body = body.child(div().h(px(560.)).child(skeleton(&t, "charts")));
            }
        }
        let samples = self.samples_panel(&r, window, cx);
        let logs = self.logs_panel(&r, cx);
        body = body.child(
            div()
                .flex()
                .gap(px(16.))
                .child(div().flex_1().min_w_0().flex().child(samples))
                // Wide windows split evenly, so log lines fit on one row; otherwise a fixed column
                // wide enough for an epoch summary line.
                .map(|d| {
                    if window.viewport_size().width >= px(1500.) {
                        d.child(div().flex_1().min_w_0().flex().child(logs))
                    } else {
                        d.child(div().w(px(500.)).flex_none().flex().child(logs))
                    }
                }),
        );
        body = body.child(self.config_panel(&r, &ds_name, cx));
        self.page_frame(&t, header, body)
    }

    fn failure_banner(&self, r: &Run, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let msg = r.error.clone().unwrap_or_else(|| "The run failed.".into());
        let hint = if msg.contains("out of memory") {
            "Lower the batch size or image size, or pick a smaller architecture. The last checkpoint was not saved."
        } else {
            "See the log below for details."
        };
        div()
            .flex()
            .gap(px(12.))
            .p(px(14.))
            .rounded(px(radius::LG))
            .bg(t.danger_muted)
            .border_1()
            .border_color(crate::theme::alpha(t.danger_solid, 0.35))
            .child(icon("icons/circle-x.svg", 18., t.danger_solid))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .child(format!("Failed at epoch {} of {}", r.epoch, r.epochs)),
                    )
                    .child(
                        div()
                            .font_family(crate::theme::MONO)
                            .text_size(px(ty::SM.size))
                            .text_color(t.danger_text)
                            .child(msg),
                    )
                    .child(text(ty::SM, t.text_secondary).child(hint)),
            )
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn kpi_strip(
        &self,
        r: &Run,
        ips: &[f32],
        last: Option<&crate::api::Epoch>,
        prev: Option<f32>,
        now: i64,
        narrow: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let live = r.status == RunStatus::Running;
        // Change since the previous epoch only means something while training.
        let delta = match (last, prev) {
            (Some(l), Some(p)) if live => {
                let d = l.val.map50_95 - p;
                let (c, s) = if d >= 0.0 {
                    (t.success_text, format!("+{d:.3}"))
                } else {
                    (t.danger_text, format!("{d:.3}"))
                };
                let best = r.best.as_ref().map_or(String::new(), |b| format!("· best {}", f3(b.map50_95)));
                Some(
                    div()
                        .flex()
                        .gap(px(4.))
                        .child(text(ty::XS, c).tnum().child(format!("{s} vs last")))
                        .child(text(ty::XS, t.text_tertiary).tnum().child(best)),
                )
            }
            _ => None,
        };
        let best50 = self
            .store
            .run_data
            .get(&r.id)
            .and_then(|l| l.ready())
            .and_then(|d| d.metrics.epochs.iter().map(|e| e.val.map50).max_by(|a, b| a.total_cmp(b)));
        let ips_now = ips.last().copied();
        let ips_avg = if ips.is_empty() { None } else { Some(ips.iter().sum::<f32>() / ips.len() as f32) };
        // Smooth the throughput trace so the sparkline shows the trend, not per-step jitter.
        let pts: Vec<(f32, f32)> = ips.iter().enumerate().map(|(i, v)| (i as f32, *v)).collect();
        let smooth: Vec<f32> = crate::charts::ema(&pts, 0.85).into_iter().map(|p| p.1).collect();
        let duration_tile = |label: &str, secs: Option<f64>, detail: gpui::Div| {
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .min_w_0()
                .child(caption(&t, label))
                .child(duration_metric(&t, secs, ty::X2XL, ty::SM))
                .child(detail)
        };
        let time_tile = if live || r.status == RunStatus::Queued {
            duration_tile(
                "Time left",
                r.eta_s,
                text(ty::XS, t.text_tertiary).tnum().child(
                    duration_of(r, now).map_or(String::new(), |d| format!("{} elapsed", time::duration(d))),
                ),
            )
        } else {
            duration_tile(
                "Duration",
                duration_of(r, now),
                text(ty::XS, t.text_tertiary).child(
                    r.finished
                        .as_deref()
                        .and_then(time::parse)
                        .map_or(String::new(), |f| format!("ended {}", time::ago(f, now))),
                ),
            )
        };
        let tiles: Vec<gpui::Div> = vec![
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(caption(&t, "Epoch"))
                .child(metric(&t, r.epoch.to_string(), &format!("/ {}", r.epochs), ty::X2XL, ty::SM))
                .child(div().pt(px(6.)).child(progress(
                    &t,
                    r.progress(),
                    if live { t.accent_fill() } else { t.text_tertiary },
                    3.,
                ))),
            stat(
                &t,
                if live { "mAP50-95 · latest" } else { "mAP50-95 · final" },
                last.map_or("–".into(), |e| f3(e.val.map50_95)),
                "",
                delta.or_else(|| {
                    r.best.as_ref().map(|b| {
                        text(ty::XS, t.text_tertiary).tnum().child(format!(
                            "best {} at epoch {}",
                            f3(b.map50_95),
                            b.epoch
                        ))
                    })
                }),
            ),
            stat(
                &t,
                "mAP50",
                last.map_or("–".into(), |e| f3(e.val.map50)),
                "",
                best50.map(|b| text(ty::XS, t.text_tertiary).tnum().child(format!("best {}", f3(b)))),
            ),
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(caption(&t, "Precision / Recall"))
                .child(
                    div()
                        .flex()
                        .items_baseline()
                        .gap(px(4.))
                        .child(
                            text(ty::X2XL, t.text_primary)
                                .tnum()
                                .child(last.map_or("–".into(), |e| format!("{:.3}", e.val.precision))),
                        )
                        .child(text(ty::LG, t.text_tertiary).child("/"))
                        .child(
                            text(ty::X2XL, t.text_primary)
                                .tnum()
                                .child(last.map_or("–".into(), |e| format!("{:.3}", e.val.recall))),
                        ),
                )
                .child(text(ty::XS, t.text_tertiary).child("on the validation split")),
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(caption(&t, "Throughput"))
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(px(10.))
                        .child(metric(
                            &t,
                            ips_now.map_or("–".into(), |v| format!("{v:.0}")),
                            "img/s",
                            ty::X2XL,
                            ty::SM,
                        ))
                        .child(
                            div().flex_1().min_w(px(40.)).h(px(28.)).pb(px(4.)).when(smooth.len() > 2, |d| {
                                d.child(sparkline(&t, Arc::new(smooth.clone()), t.text_secondary, 24., live))
                            }),
                        ),
                )
                .child(text(ty::XS, t.text_tertiary).tnum().child(
                    ips_avg.map_or(String::new(), |a| format!("avg {a:.0} · last {} steps", ips.len())),
                )),
            time_tile,
        ];
        let tile = |content: gpui::Div| {
            div().flex_1().min_w_0().px(px(16.)).py(px(if narrow { 10. } else { 14. })).child(content)
        };
        let per_row = if narrow { 3 } else { 6 };
        let mut rows = Vec::new();
        let mut it = tiles.into_iter().peekable();
        let mut first = true;
        while it.peek().is_some() {
            let mut row = div().flex();
            if !first {
                row = row.border_t_1().border_color(t.line_subtle);
            }
            first = false;
            for (i, c) in it.by_ref().take(per_row).enumerate() {
                if i > 0 {
                    row = row.child(vline(&t));
                }
                row = row.child(tile(c));
            }
            rows.push(row);
        }
        panel(&t).flex().flex_col().children(rows).into_any_element()
    }

    fn run_charts(
        &mut self,
        r: &Run,
        d: &crate::store::Derived,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let live = r.status == RunStatus::Running;
        let hover = self.chart_hover;
        let view = cx.entity().downgrade();
        let on_hover: HoverFn = Rc::new(move |f, _, cx| {
            view.update(cx, |this, cx| {
                if this.chart_hover != f {
                    this.chart_hover = f;
                    cx.notify();
                }
            })
            .ok();
        });
        // Active and completed runs show the whole schedule; cut-short runs zoom to the epochs they ran.
        let total = if r.status.is_active() || r.status == RunStatus::Completed {
            r.epochs as f32
        } else {
            r.epoch.max(1) as f32
        };
        let closing = (r.epochs.saturating_sub(r.config.augment.close_mosaic)) as f32;
        let close_marker = Marker { x: closing, label: "close mosaic".into(), color: t.text_tertiary };
        // Every chart in the grid reserves the same two label lanes, so plot tops always line up.
        let lanes = 2;
        let base = |id: &str, series: Vec<Series>| {
            let mut c = LineChart::new(id.to_string(), series);
            c.x_range = Some((0., total));
            c.hover = hover;
            c.on_hover = Some(on_hover.clone());
            c.height = 200.;
            // All four charts reserve the same marker band, so plot tops line up across a row.
            c.min_lanes = lanes;
            c.x_int = true;
            c.x_name = "epoch";
            c.x_fmt = |v| format!("{v:.0}");
            c.markers = vec![close_marker.clone()];
            c
        };
        let names = ["box", "cls", "dfl", "total"];
        let colors = t.loss_colors();
        let loss_series: Vec<Series> = (0..4)
            .filter(|&i| self.loss_parts[i])
            .map(|i| Series::new(names[i], colors[i], d.loss[i].clone()).smooth(0.94).live(live))
            .collect();
        let loss_max = (0..4).filter(|&i| self.loss_parts[i]).map(|i| d.loss_max[i]).fold(0f32, f32::max);
        let mut loss = base("run-loss", loss_series);
        loss.y_range = Some((0., crate::charts::nice_ceil((loss_max * 1.05).max(0.1), 4)));
        loss.value_fmt = Some(|v| format!("{v:.3}"));
        let mut map = base(
            "run-map",
            vec![
                // Legend order: mAP50-95 first, so the hover readout lists it first too.
                Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().live(live).width(1.75),
                Series::new("mAP50", t.chart_series[1], d.map50.clone()).width(1.25),
            ],
        );
        map.y_range = Some((0., 1.));
        map.value_fmt = Some(|v| format!("{v:.3}"));
        if let Some(b) = &r.best {
            map.markers.insert(
                0,
                Marker {
                    x: b.epoch as f32,
                    label: format!("best {} · epoch {}", f3(b.map50_95), b.epoch).into(),
                    color: t.accent_text,
                },
            );
        }
        // A failed run marks where it stopped, on the charts that tell the story.
        if r.status == RunStatus::Failed {
            let fail = Marker {
                x: r.epoch as f32,
                label: format!("failed · epoch {}", r.epoch).into(),
                color: t.danger_text,
            };
            loss.markers.insert(0, fail.clone());
            map.markers.insert(0, fail);
        }
        let mut pr = base(
            "run-pr",
            vec![
                // One hue for the pair, told apart by dash: no other chart here uses slot 5.
                Series::new("precision", t.chart_series[5], d.precision.clone()).live(live),
                Series::new("recall", t.chart_series[5], d.recall.clone()).dashed().live(live),
            ],
        );
        pr.y_range = Some((0., 1.));
        pr.value_fmt = Some(|v| format!("{v:.3}"));
        let mut lr = base("run-lr", vec![Series::new("lr", t.text_tertiary, d.lr.clone()).area().live(live)]);
        lr.value_fmt = Some(|v| format!("{v:.5}"));
        let lr_max = d.lr_max;
        lr.y_range = Some((0., (lr_max * 1.15).max(1e-4)));
        lr.markers.clear();
        lr.markers.push(Marker { x: 3., label: "warmup".into(), color: t.text_tertiary });

        // Legend toggles for the loss parts.
        let mut legend = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let on = self.loss_parts[i];
            let key = format!("loss-part-{name}");
            let color = colors[i];
            let el = div()
                .id(SharedString::from(key.clone()))
                .h(px(20.))
                .px(px(6.))
                .flex()
                .items_center()
                .gap(px(5.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .tooltip(crate::controls::tip(if on { "Hide this series" } else { "Show this series" }, None))
                .child(
                    div()
                        .w(px(10.))
                        .h(px(if on { 2. } else { 6. }))
                        .rounded(px(2.))
                        .map(|d| if on { d.bg(color) } else { d.border_1().border_color(color) }),
                )
                .child(
                    text(ty::XS, if on { t.text_secondary } else { t.text_tertiary }).child(name.to_string()),
                );
            legend.push(
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let n = this.loss_parts.iter().filter(|b| **b).count();
                        if !(this.loss_parts[i] && n == 1) {
                            this.loss_parts[i] = !this.loss_parts[i];
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        let card = |title: &str, extra: AnyElement, chart: AnyElement| {
            panel(&t)
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .child(panel_header(&t, title).child(extra))
                .child(div().px(px(12.)).pt(px(12.)).pb(px(8.)).child(chart))
        };
        let legend_row = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(
                div()
                    .id("loss-ema")
                    .mr(px(6.))
                    .tooltip(crate::controls::tip(
                        "Lines are an exponential moving average (0.94) of per-step loss; the faint trace is raw",
                        None,
                    ))
                    .child(text(ty::XS, t.text_tertiary).child("smoothed · EMA 0.94")),
            )
            .children(legend)
            .into_any_element();
        let map_legend =
            crate::ui::legend(&t, &[("mAP50-95", t.chart_series[0]), ("mAP50", t.chart_series[1])]);
        let pr_legend = crate::ui::legend_styled(
            &t,
            &[("precision", t.chart_series[5], false), ("recall", t.chart_series[5], true)],
        );
        let lr_legend = text(ty::XS, t.text_tertiary).child("cosine · warmup 3 epochs").into_any_element();
        let _ = chart_block;
        let grid = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(card("Training loss", legend_row, line_chart(&t, loss)))
                    .child(card("Validation mAP", map_legend, line_chart(&t, map))),
            )
            .child(
                div()
                    .flex()
                    .gap(px(16.))
                    .child(card("Precision and recall", pr_legend, line_chart(&t, pr)))
                    .child(card("Learning rate", lr_legend, line_chart(&t, lr))),
            );
        // One Tab stop for the linked charts: arrows walk the epochs.
        self.chart_group("run-charts", 1.0 / total.max(1.0), crate::controls::Cursor::Run, grid, window, cx)
    }

    fn samples_panel(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let views = [SampleView::Predictions, SampleView::Truth, SampleView::Both];
        let sel = views.iter().position(|v| *v == self.sample_view).unwrap_or(0);
        let seg = self.segmented(
            "samples-view",
            &["Predictions", "Ground truth", "Both"],
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, _| this.sample_view = views[i]),
        );
        let classes =
            self.store.dataset_summary(&r.config.dataset_id).map(|d| d.classes.clone()).unwrap_or_default();
        let body = match self.store.samples.get(&r.id) {
            _ if r.epoch == 0 => div()
                .child(empty_state(
                    &t,
                    "icons/image.svg",
                    "No validation yet",
                    "Predictions on 16 fixed validation images appear after the first epoch.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Some(Load::Ready(s)) => {
                let samples: Vec<crate::api::Sample> = s.samples.iter().take(8).cloned().collect();
                let epoch = s.epoch;
                let view = self.sample_view;
                let tiles: Vec<AnyElement> = samples
                    .iter()
                    .map(|smp| {
                        let img = self.image(&smp.dataset_id, &smp.image_id, ImgKind::Thumb, cx);
                        let mut boxes: Vec<DrawBox> = Vec::new();
                        if view != SampleView::Predictions {
                            boxes.extend(smp.truth.iter().map(|b| DrawBox {
                                truth: true,
                                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
                            }));
                        }
                        if view != SampleView::Truth {
                            boxes.extend(smp.pred.iter().filter(|p| p.score >= 0.3).map(|p| DrawBox {
                                label: classes.get(p.class as usize).cloned().map(Into::into),
                                score: Some(p.score),
                                ..DrawBox::new(p.class, p.x1, p.y1, p.x2, p.y2)
                            }));
                        }
                        let dims = (smp.width as f32, smp.height as f32);
                        // Square cells: every sample gets the same frame whatever its aspect. The
                        // ratio sits on an inner full-width box, whose width is definite.
                        let mut square = div()
                            .w_full()
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .child(image_canvas(&t, img, dims, boxes, Fit::Contain, false));
                        square.style().aspect_ratio = Some(1.0);
                        div().flex_1().min_w(px(0.)).child(square).into_any_element()
                    })
                    .collect();
                let mut it = tiles.into_iter();
                let row1: Vec<AnyElement> = it.by_ref().take(4).collect();
                let row2: Vec<AnyElement> = it.collect();
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .p(px(16.))
                    .child(div().flex().items_start().gap(px(8.)).children(row1))
                    .child(div().flex().items_start().gap(px(8.)).children(row2))
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!(
                        "Epoch {epoch} · 8 of 16 fixed validation images · predictions with score ≥ 0.30"
                    )))
                    .into_any_element()
            }
            Some(Load::Failed(e)) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            _ => div()
                .p(px(12.))
                .flex()
                .gap(px(8.))
                .children((0..4).map(|i| {
                    let mut sq = div().w_full().child(skeleton(&t, format!("smp{i}")));
                    sq.style().aspect_ratio = Some(1.0);
                    div().flex_1().min_w(px(0.)).child(sq)
                }))
                .into_any_element(),
        };
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .child(panel_header(&t, "Validation samples").child(seg))
            .child(body)
            .into_any_element()
    }

    fn logs_panel(&mut self, r: &Run, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let body = match self.store.logs.get(&r.id) {
            Some(Load::Ready(lines)) => {
                let start = lines.len().saturating_sub(300);
                div()
                    .relative()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("log-scroll")
                            .flex_1()
                            .h(px(372.))
                            .overflow_y_scroll()
                            .track_scroll(&self.log_scroll)
                            .bg(t.bg_inset)
                            .py(px(10.))
                            .children(lines[start..].iter().map(|l| {
                                let (tag, color) = match l.level {
                                    LogLevel::Debug => ("DBG", t.text_disabled),
                                    LogLevel::Info => ("INF", t.text_tertiary),
                                    LogLevel::Warn => ("WRN", t.warning_text),
                                    LogLevel::Error => ("ERR", t.danger_text),
                                };
                                let clock = time::parse(&l.t).map(time::clock).unwrap_or_default();
                                div()
                                    .flex()
                                    .gap(px(12.))
                                    .px(px(16.))
                                    .py(px(2.))
                                    .font_family(crate::theme::MONO)
                                    .text_size(px(ty::XS.size))
                                    .line_height(px(17.))
                                    .when(l.level == LogLevel::Error, |d| d.bg(t.danger_muted))
                                    .child(div().flex_none().text_color(t.text_disabled).child(clock))
                                    .child(div().flex_none().w(px(24.)).text_color(color).child(tag))
                                    .child(
                                        // Long entries wrap under the message column, so nothing is clipped.
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .text_color(if l.level == LogLevel::Debug {
                                                t.text_tertiary
                                            } else {
                                                t.text_secondary
                                            })
                                            .child(l.message.clone()),
                                    )
                            })),
                    )
                    .child(div().absolute().top_0().left_0().right_0().h(px(12.)).bg(gpui::linear_gradient(
                        180.,
                        gpui::linear_color_stop(t.bg_inset, 0.),
                        gpui::linear_color_stop(crate::theme::alpha(t.bg_inset, 0.), 1.),
                    )))
                    .into_any_element()
            }
            Some(Load::Failed(e)) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            _ => loading_rows(&t, "logs", 8).into_any_element(),
        };
        let follow = if r.status == RunStatus::Running {
            div()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(dot(t.accent_fill(), 6., true, "log-live"))
                .child(text(ty::XS, t.text_tertiary).child("Following"))
        } else {
            div()
        };
        panel(&t)
            .flex_1()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Log").child(follow))
            .child(body)
            .into_any_element()
    }

    fn config_panel(&self, r: &Run, ds_name: &str, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let c = &r.config;
        let a = &c.augment;
        let mono_v = |s: String| text(ty::SM, t.text_primary).tnum().child(s);
        let col1 = div()
            .flex_1()
            .flex()
            .flex_col()
            .child(kv(&t, "Dataset", mono_v(ds_name.to_string())))
            .child(kv(&t, "Architecture", mono_v(c.arch.as_str().to_string())))
            .child(kv(&t, "Pretrained weights", mono_v(c.weights.clone().unwrap_or_else(|| "none".into()))))
            .child(kv(&t, "Seed", mono_v(c.seed.to_string())));
        let col2 = div()
            .flex_1()
            .flex()
            .flex_col()
            .child(kv(&t, "Epochs", mono_v(c.epochs.to_string())))
            .child(kv(&t, "Batch", mono_v(c.batch.to_string())))
            .child(kv(&t, "Image size", mono_v(format!("{} px", c.imgsz))))
            .child(kv(&t, "Device", mono_v(c.device.clone())));
        let col3 = div()
            .flex_1()
            .flex()
            .flex_col()
            .child(kv(&t, "Optimizer", mono_v(format!("{} · lr0 {}", c.optimizer.label(), c.lr0))))
            .child(kv(
                &t,
                "Mosaic · close",
                mono_v(format!("{:.1} · last {} epochs", a.mosaic, a.close_mosaic)),
            ))
            .child(kv(&t, "Mixup · flip", mono_v(format!("{:.2} · {:.2}", a.mixup, a.fliplr))))
            .child(kv(
                &t,
                "HSV · scale",
                mono_v(format!("{:.3}/{:.1}/{:.1} · {:.1}", a.hsv[0], a.hsv[1], a.hsv[2], a.scale)),
            ));
        panel(&t)
            .flex()
            .flex_col()
            .child(
                panel_header(&t, "Configuration")
                    .child(text(ty::XS, t.text_tertiary).font_family(crate::theme::MONO).child(r.id.clone())),
            )
            .child(div().flex().gap(px(32.)).px(px(16.)).py(px(8.)).child(col1).child(col2).child(col3))
            .into_any_element()
    }
}

fn matches_filter(r: &Run, f: RunFilter) -> bool {
    match f {
        RunFilter::All => true,
        RunFilter::Active => r.status.is_active(),
        RunFilter::Completed => matches!(r.status, RunStatus::Completed | RunStatus::Stopped),
        RunFilter::Failed => r.status == RunStatus::Failed,
    }
}
