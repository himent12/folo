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
        let new = self
            .button(
                "runs-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            )
            .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
            .into_any_element();
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
                                t.accent_solid
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
                                .child(format!("{}/{}", r.epoch, r.epochs)),
                        ),
                };
                let best = r.best.as_ref().map_or(text(ty::SM, t.text_disabled).child("–"), |b| {
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
                                r.config.arch, r.config.imgsz, r.config.batch
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
        let device = r.device.replace("NVIDIA GeForce ", "");
        let subtitle = div()
            .flex()
            .items_center()
            .gap(px(8.))
            .child(
                text(ty::BASE, t.text_secondary).child(format!("{} · {ds_name} · {device}", r.config.arch)),
            )
            .when_some(started, |d, s| {
                d.child(text(ty::BASE, t.text_tertiary).child(format!("· started {s}")))
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
                        Some("icons/square.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.confirm_stop = Some(rid.clone());
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
                let ds = r.config.dataset_id.clone();
                actions.push(
                    self.button(
                        "run-retry-cfg",
                        "New run from this config",
                        Btn::Secondary,
                        Some("icons/refresh-cw.svg"),
                        window,
                        cx,
                        act(move |this, w, cx| this.open_new_run(Some(&ds), w, cx)),
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
        let header = div()
            .flex()
            .items_end()
            .justify_between()
            .gap(px(16.))
            .pb(px(4.))
            .child(div().flex().flex_col().gap(px(4.)).min_w_0().child(title).child(subtitle))
            .child(div().flex().gap(px(8.)).children(actions));

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
                .child(div().w(px(440.)).flex_none().flex().child(logs)),
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
        let delta = match (last, prev) {
            (Some(l), Some(p)) => {
                let d = l.val.map50_95 - p;
                let (c, s) = if d >= 0.0 {
                    (t.success_text, format!("+{d:.3}"))
                } else {
                    (t.danger_text, format!("{d:.3}"))
                };
                Some(text(ty::XS, c).tnum().child(format!("{s} vs last epoch")))
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
        let time_tile = if live || r.status == RunStatus::Queued {
            stat(
                &t,
                "Time left",
                r.eta_s.map_or("–".into(), time::duration),
                "",
                Some(text(ty::XS, t.text_tertiary).tnum().child(
                    duration_of(r, now).map_or(String::new(), |d| format!("{} elapsed", time::duration(d))),
                )),
            )
        } else {
            stat(
                &t,
                "Duration",
                duration_of(r, now).map_or("–".into(), time::duration),
                "",
                Some(
                    text(ty::XS, t.text_tertiary).child(
                        r.finished
                            .as_deref()
                            .and_then(time::parse)
                            .map_or(String::new(), |f| format!("ended {}", time::ago(f, now))),
                    ),
                ),
            )
        };
        let tiles: Vec<gpui::Div> = vec![
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(overline(&t, "Epoch"))
                .child(metric(&t, r.epoch.to_string(), &format!("/ {}", r.epochs), ty::X2XL, ty::SM))
                .child(div().pt(px(6.)).child(progress(
                    &t,
                    r.progress(),
                    if live { t.accent_solid } else { t.text_tertiary },
                    3.,
                ))),
            stat(
                &t,
                "mAP50-95",
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
                .child(overline(&t, "Precision / Recall"))
                .child(
                    div()
                        .flex()
                        .items_baseline()
                        .gap(px(4.))
                        .child(
                            text(ty::X2XL, t.text_primary)
                                .tnum()
                                .child(last.map_or("–".into(), |e| format!("{:.2}", e.val.precision))),
                        )
                        .child(text(ty::LG, t.text_tertiary).child("/"))
                        .child(
                            text(ty::X2XL, t.text_primary)
                                .tnum()
                                .child(last.map_or("–".into(), |e| format!("{:.2}", e.val.recall))),
                        ),
                )
                .child(text(ty::XS, t.text_tertiary).child("on the validation split")),
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(overline(&t, "Throughput"))
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
                        .child(div().flex_1().min_w(px(40.)).h(px(28.)).pb(px(4.)).when(
                            smooth.len() > 2 && live,
                            |d| {
                                d.child(sparkline(&t, Arc::new(smooth.clone()), t.chart_series[1], 24., live))
                            },
                        )),
                )
                .child(text(ty::XS, t.text_tertiary).tnum().child(
                    ips_avg.map_or(String::new(), |a| format!("avg {a:.0} · last {} steps", ips.len())),
                )),
            time_tile,
        ];
        let tile = |content: gpui::Div| div().flex_1().min_w_0().px(px(16.)).py(px(14.)).child(content);
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
            (r.epoch as f32 * 1.15).max(5.)
        };
        let closing = (r.epochs.saturating_sub(r.config.augment.close_mosaic)) as f32;
        let close_marker = Marker { x: closing, label: "close mosaic".into(), color: t.text_tertiary };
        let base = |id: &str, series: Vec<Series>| {
            let mut c = LineChart::new(id.to_string(), series);
            c.x_range = Some((0., total));
            c.hover = hover;
            c.on_hover = Some(on_hover.clone());
            c.height = 200.;
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
        let mut loss = base("run-loss", loss_series);
        loss.value_fmt = Some(|v| format!("{v:.3}"));
        let mut map = base(
            "run-map",
            vec![
                Series::new("mAP50", t.chart_series[1], d.map50.clone()).width(1.25),
                Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().live(live).width(1.75),
            ],
        );
        map.y_range = Some((0., 1.));
        map.value_fmt = Some(|v| format!("{v:.3}"));
        if let Some(b) = &r.best {
            map.markers.push(Marker {
                x: b.epoch as f32,
                label: format!("best {}", f3(b.map50_95)).into(),
                color: t.accent_text,
            });
        }
        let mut pr = base(
            "run-pr",
            vec![
                Series::new("precision", t.chart_series[2], d.precision.clone()).live(live),
                Series::new("recall", t.chart_series[3], d.recall.clone()).live(live),
            ],
        );
        pr.y_range = Some((0., 1.));
        pr.value_fmt = Some(|v| format!("{v:.3}"));
        let mut lr =
            base("run-lr", vec![Series::new("lr", t.chart_series[5], d.lr.clone()).area().live(live)]);
        lr.value_fmt = Some(|v| format!("{v:.5}"));
        let lr_max = d.lr.iter().map(|p| p.1).fold(0f32, f32::max);
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
                .child(div().w(px(10.)).h(px(2.)).rounded_full().bg(if on { color } else { t.line_strong }))
                .child(
                    text(ty::XS, if on { t.text_secondary } else { t.text_disabled }).child(name.to_string()),
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
        let legend_row = div().flex().gap(px(2.)).children(legend).into_any_element();
        let map_legend = legend_static(&t, &[("mAP50-95", t.chart_series[0]), ("mAP50", t.chart_series[1])]);
        let pr_legend = legend_static(&t, &[("precision", t.chart_series[2]), ("recall", t.chart_series[3])]);
        let lr_legend = text(ty::XS, t.text_tertiary).child("cosine · warmup 3 epochs").into_any_element();
        let _ = chart_block;
        div()
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
            )
            .into_any_element()
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
                        div()
                            .flex_1()
                            .min_w(px(0.))
                            .h(px(150.))
                            .rounded(px(radius::MD))
                            .overflow_hidden()
                            .bg(t.bg_inset)
                            .child(image_canvas(&t, img, dims, boxes, Fit::Contain, false))
                            .into_any_element()
                    })
                    .collect();
                let mut it = tiles.into_iter();
                let row1: Vec<AnyElement> = it.by_ref().take(4).collect();
                let row2: Vec<AnyElement> = it.collect();
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .p(px(12.))
                    .child(div().flex().gap(px(8.)).children(row1))
                    .child(div().flex().gap(px(8.)).children(row2))
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
                .children((0..4).map(|i| div().flex_1().h(px(150.)).child(skeleton(&t, format!("smp{i}")))))
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
                    .id("log-scroll")
                    .flex_1()
                    .h(px(372.))
                    .overflow_y_scroll()
                    .track_scroll(&self.log_scroll)
                    .bg(t.bg_inset)
                    .py(px(8.))
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
                            .gap(px(10.))
                            .px(px(12.))
                            .py(px(2.))
                            .font_family(crate::theme::MONO)
                            .text_size(px(ty::XS.size))
                            .line_height(px(17.))
                            .when(l.level == LogLevel::Error, |d| d.bg(t.danger_muted))
                            .child(div().flex_none().text_color(t.text_disabled).child(clock))
                            .child(div().flex_none().w(px(24.)).text_color(color).child(tag))
                            .child(
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
                    }))
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
                .child(dot(t.accent_solid, 6., true, "log-live"))
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
            .child(kv(&t, "Architecture", mono_v(c.arch.clone())))
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
            .child(kv(&t, "Optimizer", mono_v(format!("{} · lr0 {}", c.optimizer, c.lr0))))
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

fn legend_static(t: &crate::theme::Theme, items: &[(&str, gpui::Hsla)]) -> AnyElement {
    div()
        .flex()
        .gap(px(12.))
        .children(items.iter().map(|(name, color)| {
            div()
                .flex()
                .items_center()
                .gap(px(5.))
                .child(div().w(px(10.)).h(px(2.)).rounded_full().bg(*color))
                .child(text(ty::XS, t.text_secondary).child(name.to_string()))
        }))
        .into_any_element()
}

fn matches_filter(r: &Run, f: RunFilter) -> bool {
    match f {
        RunFilter::All => true,
        RunFilter::Active => r.status.is_active(),
        RunFilter::Completed => matches!(r.status, RunStatus::Completed | RunStatus::Stopped),
        RunFilter::Failed => r.status == RunStatus::Failed,
    }
}
