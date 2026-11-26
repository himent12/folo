//! A run's page: its header and actions, the failure banner and the KPI strip.

use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, IntoElement, ParentElement, StatefulInteractiveElement, Styled, Window,
    div, px,
};

use super::duration_of;
use crate::api::{Run, RunExt, RunStatus, time};
use crate::app::{DeleteTarget, FoloApp, Route, act};
use crate::charts::sparkline;
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Why Resume is off on demo data (the built-in demo, `folo serve --mock`).
const DEMO_NO_RESUME: &str = "Demo data can't resume runs";

impl FoloApp {
    /// Resume a stopped or failed run where it left off (its last finished epoch).
    /// The last finished (validated) epoch, from the loaded history: where a resume continues from
    /// (`weights/last.folo`). None until the history is in (the run's own fields don't say).
    fn last_finished_epoch(&self, r: &Run) -> Option<u32> {
        let d = self.store.run_data.get(&r.id)?.ready()?;
        Some(d.metrics.epochs.last().map_or(0, |e| e.epoch))
    }

    fn resume_button(&self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let rid = r.id.clone();
        // No checkpoint (no epoch finished): it starts again, as Studio says, never "Resume".
        let again = !r.can_continue() || self.last_finished_epoch(r) == Some(0);
        let from = self.last_finished_epoch(r).map(|e| e + 1);
        let label = match from {
            _ if again => "Start again".to_string(),
            Some(f) => format!("Resume from epoch {f}"),
            None => "Resume".to_string(),
        };
        let icon = if again { "icons/refresh-cw.svg" } else { "icons/play.svg" };
        // Demo data can't resume a run: the built-in demo refuses it, and `folo serve --mock` answers
        // 501. Both say `mock` in their health; while that is unknown, Resume is offered.
        if self.store.health.ready().is_some_and(|h| h.mock) {
            return self
                .button_disabled("run-resume", label, Btn::Secondary, Some(icon), DEMO_NO_RESUME, window, cx)
                .into_any_element();
        }
        self.button(
            "run-resume",
            label,
            Btn::Secondary,
            Some(icon),
            window,
            cx,
            act(move |this, _, cx| this.resume_run(rid.clone(), cx)),
        )
        .tooltip(crate::controls::tip(
            if again {
                "No epoch finished: starts over with the same config"
            } else {
                "Continues from the last saved checkpoint"
            },
            None,
        ))
        .into_any_element()
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
            return self.page_frame(&t, header, body, cx);
        };
        self.ensure_run_data(id, cx);
        self.ensure_logs(id, cx);
        if r.epoch > 0 {
            self.ensure_samples(id, cx);
        }
        let now = self.store.api.now();
        let ds_name = self.store.run_dataset_name(&r);
        // The subtitle leads with the architecture and its e2e tag, as every list does.
        let subtitle = self.run_subtitle(&t, &r, ty::BASE, t.text_secondary, "hdr");

        let mut actions = Vec::new();
        match r.status {
            RunStatus::Running | RunStatus::Queued => {
                if r.status == RunStatus::Queued
                    && r.queue_position.is_some_and(|p| p > 1)
                    && self.store.features.iter().any(|f| f == "queue_control")
                {
                    let rid = r.id.clone();
                    actions.push(
                        self.button(
                            "run-next",
                            "Start next",
                            Btn::Secondary,
                            Some("icons/play.svg"),
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.move_queued(rid.clone(), crate::api::QueueAction::Top, cx)
                            }),
                        )
                        .tooltip(crate::controls::tip("Move it to the front of the queue", None))
                        .into_any_element(),
                    );
                }
                let rid = r.id.clone();
                // A run still waiting leaves the queue; nothing of it is stopped.
                let waiting = r.status == RunStatus::Queued && r.cloud.is_none();
                actions.push(
                    self.button(
                        "run-stop",
                        if waiting { "Remove from queue" } else { "Stop run" },
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
                if r.status == RunStatus::Stopped {
                    actions.push(self.resume_button(&r, window, cx));
                }
                let cfg = r.config.clone();
                actions.push(
                    self.button(
                        "run-clone-cfg",
                        "Clone config",
                        Btn::Secondary,
                        Some("icons/copy.svg"),
                        window,
                        cx,
                        act(move |this, w, cx| this.open_new_run_from(&cfg, None, w, cx)),
                    )
                    .tooltip(crate::controls::tip(
                        "Open New run prefilled with this run's configuration",
                        None,
                    ))
                    .into_any_element(),
                );
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
                                this.play_pred = Load::Idle;
                                this.navigate(Route::Playground, w, cx);
                                this.pick_play_model(m2.clone(), cx);
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
                // Reuse the whole config; for out-of-memory failures, halve the batch. Resuming an
                // OOM failure would hit the same limit (the config can't change), so it isn't offered.
                let mut cfg = r.config.clone();
                let oom = r.error.as_deref().is_some_and(is_gpu_oom);
                // Mixed precision refused (no tensor cores): the new run trains in f32, and resuming
                // would hit the same refusal.
                let no_amp = r.error_code.as_deref() == Some("not_supported") && cfg.amp;
                if !oom && !no_amp {
                    actions.push(self.resume_button(&r, window, cx));
                }
                let note = if no_amp {
                    cfg.amp = false;
                    Some("Mixed precision turned off: it isn't available on this run's device.".to_string())
                } else {
                    oom.then(|| crate::app::retry_batch(cfg.batch)).flatten().map(|to| {
                        let from = cfg.batch;
                        cfg.batch = to;
                        format!("Batch lowered from {from} to {to} so the run fits in GPU memory.")
                    })
                };
                actions.push(
                    self.button(
                        "run-retry-cfg",
                        "New run from this config",
                        // The way forward after a failure: the page's primary action.
                        Btn::Primary,
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
        // Delete leads the actions (the way forward stays last): a run still active says why not.
        // One waiting in the queue has Remove from queue instead (which deletes it): no dead Delete.
        let waiting = r.status == RunStatus::Queued && r.cloud.is_none();
        let delete = if waiting {
            None
        } else if r.status.is_active() {
            Some(self.button_disabled(
                "run-delete",
                "Delete",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                "Stop it first: an active run can't be deleted",
                window,
                cx,
            ))
        } else {
            let target = DeleteTarget::Run(r.id.clone());
            Some(self.button(
                "run-delete",
                "Delete",
                Btn::Secondary,
                Some("icons/trash-2.svg"),
                window,
                cx,
                act(move |this, w, cx| this.ask_delete(target.clone(), w, cx)),
            ))
        };
        if let Some(delete) = delete {
            actions.insert(0, delete.into_any_element());
        }
        let title = div()
            .w_full()
            .flex()
            .items_center()
            .min_w_0()
            .gap(px(12.))
            .child(text(ty::XL, t.text_primary).min_w_0().ellipsis().child(r.name.clone()))
            .child(div().flex_none().child(status_badge(&t, &r, &format!("hdr-{}", r.id))))
            // The worker is silent: the numbers stand still for that, not because training does
            // (the Cloud panel says more).
            .when(r.cloud.as_ref().is_some_and(|c| c.unanswered_since.is_some()), |d| {
                d.child(div().flex_none().child(badge_icon(
                    &t,
                    "icons/wifi-off.svg",
                    "Worker not answering",
                    Tone::Warning,
                )))
            })
            // The subtitle's e2e tag explains itself on hover; this help gives it a keyboard route.
            .when(r.config.head == crate::api::Head::O2o, |d| {
                d.child(div().flex_none().child(self.help_icon(
                    "run-head-help",
                    FoloApp::E2E_HELP,
                    window,
                    cx,
                )))
            });
        let header = detail_header(&t, title, subtitle, actions);

        let mut body = div().flex().flex_col().gap(px(16.));
        if r.status == RunStatus::Failed {
            body = body.child(self.failure_banner(&r, cx));
        }
        // Where it runs, how far the hand-over is and what it costs, ahead of the training itself.
        if let Some(c) = r.cloud.clone() {
            body = body.child(self.cloud_run_panel(&r, &c, now, window, cx));
        } else if r.status == RunStatus::Queued {
            body = body.child(self.queue_panel(&r, cx));
        } else if r.status == RunStatus::Stopped {
            body = body.child(self.stopped_panel(&r, cx));
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
                let window_w = f32::from(window.viewport_size().width);
                let short = window.viewport_size().height < px(700.);
                body = body.child(self.kpi_strip(&r, &ips, last.as_ref(), prev, now, window_w, short, cx));
                body = body.child(self.run_charts(&r, &d, window, cx));
            }
            Some(Err(Some(e))) => body = body.child(self.error_panel(&e, "run-data-retry", window, cx)),
            _ => {
                body = body.child(panel(&t).h(px(96.)).child(loading_rows(&t, "kpi", 2)));
                body = body.child(div().h(px(560.)).child(skeleton(&t, "charts")));
            }
        }
        // Rows 4 and 5: the samples (two thirds) beside each class's AP, and the log (two thirds)
        // beside the config. Below 1120 px each row stacks, as the chart rows do.
        let stack = window.viewport_size().width < px(1120.);
        let samples = self.samples_panel(&r, window, cx);
        let per_class = self.per_class_panel(&r, !stack, window, cx);
        let logs = self.logs_panel(&r, !stack, window, cx);
        let config = self.config_panel(&r, &ds_name, window, cx);
        body = body.child(two_thirds(samples, per_class, stack)).child(two_thirds(logs, config, stack));
        self.page_frame(&t, header, body, cx)
    }

    /// A waiting run's place in the training queue and what it waits for.
    fn queue_panel(&self, r: &Run, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let runs = self.store.runs.ready().map(|r| r.as_slice()).unwrap_or_default();
        let ahead = crate::api::training_now(runs, &r.id).map(|a| (a.name.as_str(), a.eta_s));
        panel(&t)
            .flex()
            .items_center()
            .gap(px(12.))
            .px(px(16.))
            .py(px(14.))
            .child(icon("icons/timer.svg", 18., t.text_secondary))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .child(queue_text(r.queue_position)),
                    )
                    .child(text(ty::SM, t.text_secondary).child(queue_wait_text(r.queue_position, ahead))),
            )
            .into_any_element()
    }

    /// A stopped run's outcome, as Studio says it: who stopped it (the server's shutdown is never
    /// the user's own stop) and what it kept; no checkpoint is claimed before an epoch finished.
    fn stopped_panel(&self, r: &Run, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        panel(&t)
            .flex()
            .items_center()
            .gap(px(12.))
            .px(px(16.))
            .py(px(14.))
            .child(icon("icons/stop.svg", 18., t.text_secondary))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .child(r.stopped_title()),
                    )
                    .child(text(ty::SM, t.text_secondary).child(r.stopped_detail())),
            )
            .into_any_element()
    }

    fn failure_banner(&self, r: &Run, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let msg = r.error.clone().unwrap_or_else(|| "The run failed.".into());
        let hint = if r.error_code.as_deref() == Some("not_supported") && r.config.amp {
            "Mixed precision needs a GPU with tensor cores (and folo serve --tensor-cores); this run's device had none. New run from this config trains without it.".to_string()
        } else if is_gpu_oom(&msg) {
            match crate::app::retry_batch(r.config.batch) {
                Some(to) => format!(
                    "The batch didn't fit in GPU memory. New run from this config starts with batch {to}; a smaller image size or architecture also helps."
                ),
                None => "The batch didn't fit in GPU memory and can't go lower: lower the image size or pick a smaller architecture in New run from this config.".to_string(),
            }
        } else {
            "See the log below for details.".to_string()
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
                // Shrinks beside the icon, so a long backend message wraps inside the banner.
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .child(failed_title(r.failed_epoch(), r.epochs)),
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
        window_w: f32,
        short: bool,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        // One row of five; very narrow windows wrap to two rows, one type step smaller.
        let narrow = window_w < 1000.;
        // Short windows: tighter tiles (padding only; every value keeps its detail line).
        let compact = short;
        let big = if narrow { ty::XL } else { KPI_VALUE };
        let live = r.status == RunStatus::Running;
        // Waiting to train for the first time: nothing is "final" or "last" yet.
        let unstarted = r.status == RunStatus::Queued && r.epoch == 0;
        // Change since the previous epoch only means something while training.
        let delta = match (last, prev) {
            (Some(l), Some(p)) if live => {
                // Green up, red down, and neutral when it rounds to no change.
                let (s, dir) = score_delta(l.val.map50_95 - p);
                let c = match dir {
                    std::cmp::Ordering::Greater => t.success_text,
                    std::cmp::Ordering::Less => t.danger_text,
                    std::cmp::Ordering::Equal => t.text_tertiary,
                };
                let best = r.best.as_ref().map_or(String::new(), |b| format!("· best {}", f3(b.map50_95)));
                Some(
                    div()
                        .flex()
                        .gap(px(4.))
                        .child(text(ty::SM, c).tnum().child(format!("{s} vs last")))
                        .child(kpi_note(&t).child(best)),
                )
            }
            _ => None,
        };
        let ips_now = ips.last().copied();
        let ips_avg = if ips.is_empty() { None } else { Some(ips.iter().sum::<f32>() / ips.len() as f32) };
        // Smooth the throughput trace so the sparkline shows the trend, not per-step jitter.
        let pts: Vec<(f32, f32)> = ips.iter().enumerate().map(|(i, v)| (i as f32, *v)).collect();
        let smooth: Vec<f32> = crate::charts::ema(&pts, 0.85).into_iter().map(|p| p.1).collect();
        let duration_tile = |label: &str, secs: Option<f64>, detail: folo_ui::Div| {
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .min_w_0()
                .child(kpi_label(&t, label))
                .child(duration_metric(&t, secs, big, KPI_UNIT))
                .child(detail)
        };
        let time_tile = if live || r.status == RunStatus::Queued {
            duration_tile(
                "Time left",
                r.eta_s,
                kpi_note(&t).child(
                    duration_of(r, now).map_or(String::new(), |d| format!("{} elapsed", time::duration(d))),
                ),
            )
        } else {
            duration_tile(
                "Duration",
                duration_of(r, now),
                kpi_note(&t).child(
                    r.finished
                        .as_deref()
                        .and_then(time::parse)
                        .map_or(String::new(), |f| format!("ended {}", time::ago(f, now))),
                ),
            )
        };
        // A failed run's tile names the epoch it failed in, as its callout and its charts do.
        let shown_epoch = if r.status == RunStatus::Failed { r.failed_epoch() } else { r.epoch };
        let tiles: Vec<folo_ui::Div> = vec![
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(kpi_label(&t, "Epoch"))
                .child(metric(&t, shown_epoch.to_string(), &format!("/ {}", r.epochs), big, KPI_UNIT))
                .child(div().pt(px(if compact { 4. } else { 6. })).child(progress(
                    &t,
                    r.progress(),
                    if live { t.accent_fill() } else { t.text_tertiary },
                    3.,
                ))),
            stat_sized(
                &t,
                // "final" only for a run that finished; a stopped or failed one ends on its last evaluated epoch.
                if live {
                    "mAP50-95 · latest"
                } else if unstarted {
                    "mAP50-95"
                } else if r.status == RunStatus::Completed {
                    "mAP50-95 · final"
                } else {
                    "mAP50-95 · last evaluated"
                },
                last.map_or("–".into(), |e| fmt_score(e.val.map50_95)),
                big,
                delta.or_else(|| {
                    r.best
                        .as_ref()
                        .map(|b| kpi_note(&t).child(format!("best {} @ {}", fmt_score(b.map50_95), b.epoch)))
                }),
            ),
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(kpi_label(&t, "Precision / Recall"))
                .child(metric_pair(
                    &t,
                    big,
                    last.map_or("–".into(), |e| format!("{:.3}", e.val.precision)),
                    last.map_or("–".into(), |e| format!("{:.3}", e.val.recall)),
                ))
                // The epoch's own operating point once the engine reports it (`best_f1_conf`); the
                // P/R chart below states the match rule, so the tile's line fits at 1280.
                .child(kpi_note(&t).ellipsis().child(pr_caption(
                    last.and_then(|e| operating_conf(e.val.best_f1_conf, e.val.precision, e.val.recall)),
                    self.eval_split(r),
                    false,
                ))),
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(kpi_label(&t, if live || unstarted { "Throughput" } else { "Final throughput" }))
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(px(10.))
                        .child(metric(
                            &t,
                            ips_now.map_or("–".into(), |v| format!("{v:.0}")),
                            "img/s",
                            big,
                            KPI_UNIT,
                        ))
                        .child(
                            // A glance, not a chart: 160 px in the one-row strip; in the wrapped
                            // two-column tile a little wider (240), so 120 steps don't spread into
                            // a noise band.
                            div()
                                .flex_1()
                                .min_w(px(40.))
                                .max_w(px(if narrow { 240. } else { 160. }))
                                .h(px(28.))
                                .pb(px(4.))
                                .when(smooth.len() > 2, |d| {
                                    d.child(sparkline(
                                        &t,
                                        Arc::new(smooth.clone()),
                                        t.text_secondary,
                                        24.,
                                        live,
                                    ))
                                }),
                        ),
                )
                .child(kpi_note(&t).child(
                    ips_avg.map_or(String::new(), |a| format!("avg {a:.0} · last {} steps", ips.len())),
                )),
            time_tile,
        ];
        let tile = |content: folo_ui::Div| {
            div()
                .flex_1()
                .min_w_0()
                .px(px(16.))
                .py(px(if compact { 8. } else { 12. }))
                .child(content.when(compact, |d| d.gap(px(2.))))
        };
        let per_row = if window_w < 1000. { 3 } else { 5 };
        let mut rows = Vec::new();
        let mut it = tiles.into_iter().peekable();
        let mut first = true;
        while it.peek().is_some() {
            // Full width, so the two-column span below has a width to take a fraction of.
            let mut row = div().w_full().flex();
            if !first {
                row = row.border_t_1().border_color(t.line_subtle);
            }
            let wrapped = !first;
            first = false;
            for (i, c) in it.by_ref().take(per_row).enumerate() {
                if i > 0 {
                    row = row.child(vline(&t));
                }
                // On a 3-column grid: the wrapped row's first tile (Throughput, whose sparkline
                // fills it) spans two columns, so its divider lines up with the first row's second.
                // flex_1's 0 basis would win over a width, so the basis is set itself.
                row = row.child(if wrapped && per_row == 3 && i == 0 {
                    tile(c).flex_none().flex_basis(folo_ui::relative(2. / 3.))
                } else {
                    tile(c)
                });
            }
            rows.push(row);
        }
        panel(&t).flex().flex_col().children(rows).into_any_element()
    }
}

/// A row of the run page: `a` in two thirds beside `b` in one (both as tall as the taller), or one
/// above the other when `stack`.
fn two_thirds(a: AnyElement, b: AnyElement, stack: bool) -> folo_ui::Div {
    if stack {
        return div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(div().flex().child(a))
            .child(div().flex().child(b));
    }
    let cell =
        |e: AnyElement, f: f32| div().flex().min_w_0().flex_grow().flex_basis(folo_ui::relative(f)).child(e);
    div().flex().gap(px(16.)).child(cell(a, 2. / 3.)).child(cell(b, 1. / 3.))
}

/// A run that failed with `gpu_out_of_memory`. Runs carry the message, not the code (docs/API.md):
/// the server writes "the GPU ran out of memory (…)", older servers the driver's "… out of memory …".
fn is_gpu_oom(message: &str) -> bool {
    let m = message.to_ascii_lowercase();
    m.contains("out of memory") || m.contains("outofmemory")
}

/// A failed run's banner title, in Studio's words (`failedTitle`): a run that failed before its
/// first step (its dataset unreadable) is not "Failed in epoch 0", an epoch that does not exist.
fn failed_title(failed_epoch: u32, epochs: u32) -> String {
    if failed_epoch == 0 {
        "Failed before training started".into()
    } else {
        format!("Failed in epoch {failed_epoch} of {epochs}")
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_run_that_failed_before_its_first_step_says_so() {
        use super::failed_title;
        assert_eq!(failed_title(0, 2), "Failed before training started");
        assert_eq!(failed_title(2, 3), "Failed in epoch 2 of 3");
    }

    #[test]
    fn out_of_memory_failures_are_recognised() {
        use super::is_gpu_oom;
        assert!(is_gpu_oom("the GPU ran out of memory (Out of Memory: tried to allocate 2.1 GiB)"));
        assert!(is_gpu_oom("wgpu: OutOfMemory"));
        assert!(!is_gpu_oom("invalid dataset: no training images"));
    }
}
