//! The Labeling jobs list screen (`Route::LabelJobs`).

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::{job_status, run_agent_summary};
use crate::api::{LabelJob, time};
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::label::{LabelJobFilter, filter_label_jobs};
use crate::screens::{loading_rows, page_header, th};
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn label_jobs_screen(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        self.ensure_label_jobs(cx);
        if std::mem::take(&mut self.label.list_stale) && self.label.jobs.ready().is_some() {
            self.refresh_label_jobs(cx);
        }
        let list = self.label.jobs.ready().cloned();
        let agents = self.label.agents.ready().cloned();
        let list_err = self
            .label
            .jobs
            .error()
            .or_else(|| list.is_none().then(|| self.label_blocked_error()).flatten())
            .cloned();
        let subtitle = if self.label_unsupported() {
            "Not available on this server".to_string()
        } else {
            match (&list, &list_err) {
                (Some(l), _) => {
                    let active = l.iter().filter(|j| j.status.is_active()).count();
                    let images: u64 = l.iter().map(|j| j.counts.images).sum();
                    let labeled: u64 = l.iter().map(|j| j.counts.labeled).sum();
                    let mut parts = vec![counted(l.len() as u64, "job", "jobs")];
                    if active > 0 {
                        parts.push(format!("{active} labeling"));
                    }
                    parts.push(format!("{} / {} labeled", int(labeled), int(images)));
                    parts.join(" · ")
                }
                (None, Some(_)) => "Could not load labeling jobs".to_string(),
                _ => "Loading…".to_string(),
            }
        };
        let mut actions = Vec::new();
        if self.label_on() && self.label.new_job.is_none() {
            actions.push(
                self.button(
                    "label-new",
                    "New labeling job",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_label_job(w, cx)),
                )
                .into_any_element(),
            );
        }
        let header = page_header(&t, "Labeling", subtitle, actions);
        if self.label_unsupported() {
            let body = self.label_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        if let Some(e) = list_err {
            let err = self.error_panel(&e, "label-jobs-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }

        let form = self.label.new_job.is_some().then(|| self.new_label_job_form(&t, window, cx));
        let content = self.content_width(window);
        let show_agent = content >= 860.;
        let show_created = content >= 960.;
        let query = self.label_search.read(cx).text().to_string();
        let filter = self.label.list_filter;
        let filters_active = filter != LabelJobFilter::All || !query.trim().is_empty();

        let toolbar = list.as_ref().filter(|l| !l.is_empty()).map(|l| {
            let active_n = l.iter().filter(|j| j.status.is_active()).count();
            let failed_n =
                l.iter().filter(|j| !j.status.is_active() && job_status(j).2 == StatusKind::Failed).count();
            let idle_n = l.len().saturating_sub(active_n + failed_n);
            let labels = [
                format!("All\u{a0}{}", int(l.len() as u64)),
                format!("Labeling\u{a0}{}", int(active_n as u64)),
                format!("Failed\u{a0}{}", int(failed_n as u64)),
                format!("Idle\u{a0}{}", int(idle_n as u64)),
            ];
            let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            let buckets =
                [LabelJobFilter::All, LabelJobFilter::Active, LabelJobFilter::Failed, LabelJobFilter::Idle];
            let sel = buckets.iter().position(|b| *b == filter).unwrap_or(0);
            let seg = self.segmented(
                "label-filter",
                &refs,
                sel,
                window,
                cx,
                Rc::new(move |this, i, _, cx| {
                    this.label.list_filter = buckets[i.min(buckets.len() - 1)];
                    cx.notify();
                }),
            );
            let search = self.search_field(
                "label-search",
                self.label_search.clone(),
                if self.narrow { 180. } else { 220. },
                window,
                cx,
            );
            let reset_btn = filters_active.then(|| {
                self.button(
                    "label-filter-reset",
                    "Reset",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.label.list_filter = LabelJobFilter::All;
                        this.label_search.update(cx, |i, cx| i.set_text("", cx));
                        cx.notify();
                    }),
                )
                .into_any_element()
            });
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap(px(8.))
                .child(seg)
                .child(div().flex_1())
                .child(search)
                .children(reset_btn)
        });

        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Job").flex_1().min_w(px(160.)))
            .child(div().w(px(72.)).flex_none().flex().justify_end().child(th(&t, "Images")))
            .child(div().w(px(156.)).flex_none().child(th(&t, "Labeled")))
            .child(div().w(px(108.)).flex_none().child(th(&t, "Status")))
            .when(show_agent, |d| d.child(div().w(px(164.)).flex_none().child(th(&t, "Last agent"))))
            .when(show_created, |d| {
                d.child(div().w(px(84.)).flex_none().flex().justify_end().child(th(&t, "Created")))
            });

        let now = self.store.api.now();
        let table = match list {
            None => panel(&t).child(head).child(loading_rows(&t, "label-jobs", 3)).into_any_element(),
            Some(l) if l.is_empty() => {
                let cta = self.button(
                    "label-empty-new",
                    "New labeling job",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_label_job(w, cx)),
                );
                panel(&t)
                    .child(
                        empty_state(
                            &t,
                            "icons/sparkles.svg",
                            "No labeling jobs yet",
                            "Point a job at a folder of unlabeled images, name the classes, let Claude Code, Codex or Antigravity draw the draft boxes, then review and export a YOLO dataset.",
                            Tone::Neutral,
                        )
                        .child(cta),
                    )
                    .into_any_element()
            }
            Some(l) => {
                let shown: Vec<&LabelJob> = filter_label_jobs(&l, filter, &query);
                if shown.is_empty() {
                    let reset = self.button(
                        "label-empty-reset",
                        "Reset filters",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, w, cx| {
                            this.label.list_filter = LabelJobFilter::All;
                            this.label_search.update(cx, |i, cx| i.set_text("", cx));
                            w.focus(&this.label_search.read(cx).focus_handle(cx));
                            cx.notify();
                        }),
                    );
                    panel(&t)
                        .overflow_hidden()
                        .child(head)
                        .child(
                            empty_state(
                                &t,
                                "icons/list-filter.svg",
                                "No labeling jobs match",
                                "Try another status filter or search term.",
                                Tone::Neutral,
                            )
                            .child(reset),
                        )
                        .into_any_element()
                } else {
                    let first_id = shown.first().map(|j| j.id.clone());
                    let last_id = shown.last().map(|j| j.id.clone());
                    let rows: Vec<AnyElement> = (0..shown.len())
                        .map(|i| {
                            let job = shown[i];
                            let id = job.id.clone();
                            let key = format!("label-row-{id}");
                            let prev_id = i.checked_sub(1).map(|j| shown[j].id.clone());
                            let next_id = shown.get(i + 1).map(|j| j.id.clone());
                            let home_id = (i > 0).then(|| first_id.clone()).flatten();
                            let end_id = (i + 1 < shown.len()).then(|| last_id.clone()).flatten();
                            let (st_label, st_tone, st_kind) = job_status(job);
                            let ratio = if job.counts.images > 0 {
                                (job.counts.labeled as f32 / job.counts.images as f32).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let bar_color =
                                if job.status.is_active() { t.accent_fill() } else { t.text_tertiary };
                            let agent_text = job
                                .run
                                .as_ref()
                                .map(|r| run_agent_summary(r, agents.as_deref()))
                                .unwrap_or_else(|| "No run yet".to_string());
                            let at = time::parse(&job.created);
                            let classes_summary = job.classes.join(", ");
                            let row = div()
                                .id(SharedString::from(key.clone()))
                                .flex()
                                .items_center()
                                .gap(px(16.))
                                .px(px(16.))
                                .h(px(44.))
                                .border_b_1()
                                .border_color(t.line_subtle)
                                .cursor_pointer()
                                .hover(|s| s.bg(t.bg_hover))
                                .active(|s| s.bg(t.bg_active))
                                .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, w, cx| {
                                    let m = &ev.keystroke.modifiers;
                                    if m.control || m.alt || m.platform || m.shift {
                                        return;
                                    }
                                    let target = match ev.keystroke.key.as_str() {
                                        "down" => next_id.as_deref(),
                                        "up" => prev_id.as_deref(),
                                        "home" => home_id.as_deref(),
                                        "end" => end_id.as_deref(),
                                        _ => return,
                                    };
                                    if let Some(tid) = target {
                                        this.focus_key_and_reveal(&format!("label-row-{tid}"), w, cx);
                                        cx.stop_propagation();
                                    }
                                }))
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w(px(160.))
                                        .flex()
                                        .flex_col()
                                        .gap(px(1.))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap(px(6.))
                                                .child(
                                                    text(ty::SM, t.text_primary)
                                                        .font_weight(FontWeight::MEDIUM)
                                                        .ellipsis()
                                                        .child(job.name.clone()),
                                                )
                                                .when(job.status.is_active(), |d| {
                                                    d.child(status_mark(
                                                        StatusKind::Running,
                                                        StatusKind::Running.hue(&t),
                                                        format!("label-live-{}", job.id),
                                                    ))
                                                }),
                                        )
                                        .child(
                                            text(ty::XS, t.text_tertiary).ellipsis().child(classes_summary),
                                        ),
                                )
                                .child(div().w(px(72.)).flex_none().flex().justify_end().child(
                                    text(ty::SM, t.text_secondary).tnum().child(int(job.counts.images)),
                                ))
                                .child(
                                    div()
                                        .w(px(156.))
                                        .flex_none()
                                        .flex()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(
                                            text(ty::SM, t.text_secondary)
                                                .tnum()
                                                .w(px(44.))
                                                .child(int(job.counts.labeled)),
                                        )
                                        .child(div().flex_1().child(progress(&t, ratio, bar_color, 4.))),
                                )
                                .child(
                                    div()
                                        .w(px(108.))
                                        .flex_none()
                                        .flex()
                                        .items_center()
                                        .gap(px(6.))
                                        .child(status_mark(
                                            st_kind,
                                            st_kind.hue(&t),
                                            format!("label-st-{}", job.id),
                                        ))
                                        .child(badge(&t, st_label, st_tone)),
                                )
                                .when(show_agent, |d| {
                                    d.child(
                                        text(
                                            ty::SM,
                                            if job.run.is_some() {
                                                t.text_secondary
                                            } else {
                                                t.text_tertiary
                                            },
                                        )
                                        .w(px(164.))
                                        .flex_none()
                                        .ellipsis()
                                        .child(agent_text),
                                    )
                                })
                                .when(show_created, |d| {
                                    d.child(
                                        div().w(px(84.)).flex_none().flex().justify_end().child(
                                            text(ty::SM, t.text_tertiary)
                                                .tnum()
                                                .whitespace_nowrap()
                                                .child(at.map_or(String::new(), |c| time::ago(c, now))),
                                        ),
                                    )
                                });
                            self.bind(
                                row,
                                &key,
                                0.,
                                window,
                                cx,
                                act(move |this, w, cx| this.navigate(Route::LabelJob(id.clone()), w, cx)),
                            )
                            .into_any_element()
                        })
                        .collect();
                    panel(&t).overflow_hidden().child(head).children(rows).into_any_element()
                }
            }
        };

        let body = div().flex().flex_col().gap(px(16.)).children(form).children(toolbar).child(table);
        self.page_frame(&t, header, body, cx)
    }
}
