//! Right rail of the Agent Labeling job screen: agent/model/effort selection, live run progress,
//! last run outcome & token usage, job classes/instructions, and live job logs.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::{format_usage, run_agent_summary, unlabeled_count};
use crate::api::*;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// Right rail of `Route::LabelJob`: Agent runner card + Classes/Instructions + Logs.
    pub(super) fn label_agent_rail(
        &self,
        t: &Theme,
        job: &LabelJob,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let agent_card = self.label_agent_card(t, job, window, cx);
        let info_card = self.label_info_card(t, job);
        let logs_card = self.label_logs_card(t, &job.id);
        div()
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(agent_card)
            .child(info_card)
            .child(logs_card)
            .into_any_element()
    }

    fn label_agent_card(
        &self,
        t: &Theme,
        job: &LabelJob,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let agents = self.label.agents.ready().map(Vec::as_slice);
        let running = job.status.is_active();
        let mut sections: Vec<AnyElement> = Vec::new();

        if running && let Some(run) = &job.run {
            let done_all = run.done + run.failed;
            let ratio =
                if run.total > 0 { (done_all as f32 / run.total as f32).clamp(0.0, 1.0) } else { 0.0 };
            let who = run_agent_summary(run, agents);
            let stop_btn = if job.status == LabelJobStatus::Stopping || self.label.busy.contains(&job.id) {
                self.button_disabled("label-stop", "Stopping…", Btn::Danger, None, "Stopping run", window, cx)
            } else {
                let jid = job.id.clone();
                self.button(
                    "label-stop",
                    "Stop labeling",
                    Btn::Danger,
                    Some("icons/square.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.stop_label_run(&jid, cx)),
                )
            };
            sections.push(
                div()
                    .p(px(14.))
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .tnum()
                                    .child(format!("{} of {} labeled", int(run.done), int(run.total))),
                            )
                            .when(run.failed > 0, |d| {
                                d.child(
                                    text(ty::XS, t.danger_text)
                                        .tnum()
                                        .child(format!("{} failed", int(run.failed))),
                                )
                            }),
                    )
                    .child(progress(t, ratio, t.accent_fill(), 4.))
                    .child(
                        text(ty::XS, t.text_secondary)
                            .child(format!("{who} · {} effort", run.effort.as_str())),
                    )
                    .child(stop_btn)
                    .into_any_element(),
            );
        } else {
            // Idle: agent controls + start button + last run summary.
            let selected_agent = agents.and_then(|list| {
                self.label
                    .agent_id
                    .and_then(|id| list.iter().find(|a| a.id == id))
                    .or_else(|| list.iter().find(|a| a.available))
                    .or_else(|| list.first())
            });
            let available = selected_agent.is_some_and(|a| a.available);
            let to_label = unlabeled_count(job);
            let scope =
                if to_label == 0 && job.counts.images > 0 { LabelScope::All } else { self.label.scope };
            let why_disabled = if job.counts.images == 0 {
                Some("Add images to this job first")
            } else if agents.is_none() {
                Some("Checking installed agents…")
            } else if !available {
                Some("Install Claude Code, Codex, or Antigravity on the server")
            } else if self.label.starting {
                Some("Starting agent run…")
            } else {
                None
            };

            let start_label = if self.label.starting {
                "Starting…".to_string()
            } else if scope == LabelScope::All {
                format!("Label all {} again", counted(job.counts.images, "image", "images"))
            } else {
                format!("Label {}", counted(to_label, "image", "images"))
            };
            let start_btn = match why_disabled {
                Some(why) => self.button_disabled(
                    "label-start",
                    &start_label,
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    why,
                    window,
                    cx,
                ),
                None => {
                    let jid = job.id.clone();
                    self.button(
                        "label-start",
                        &start_label,
                        if scope == LabelScope::All { Btn::Secondary } else { Btn::Primary },
                        Some("icons/sparkles.svg"),
                        window,
                        cx,
                        act(move |this, _, cx| this.start_label_run(&jid, scope, cx)),
                    )
                }
            };

            let mut run_col = div().p(px(14.)).flex().flex_col().gap(px(10.));

            // Last run failure callout if applicable.
            if let Some(run) = &job.run
                && run.outcome == Some(LabelOutcome::Failed)
            {
                run_col = run_col.child(
                    div()
                        .p(px(10.))
                        .rounded(px(radius::MD))
                        .bg(t.danger_muted)
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .child(
                            text(ty::SM, t.danger_text)
                                .font_weight(FontWeight::MEDIUM)
                                .child(format!("{} stopped", run.agent.display_name())),
                        )
                        .when_some(run.error.clone(), |d, e| {
                            d.child(text(ty::XS, t.danger_text).whitespace_normal().child(e))
                        })
                        .when_some(run.hint.clone(), |d, h| {
                            d.child(text(ty::XS, t.text_secondary).whitespace_normal().child(h))
                        }),
                );
            }

            // Agent selector.
            if let Some(list) = agents {
                let opts: Vec<(folo_ui::SharedString, Option<folo_ui::SharedString>)> = list
                    .iter()
                    .map(|a| {
                        let detail = if a.available { None } else { Some("missing".into()) };
                        (a.name.clone().into(), detail)
                    })
                    .collect();
                let sel_idx =
                    selected_agent.and_then(|s| list.iter().position(|a| a.id == s.id)).unwrap_or(0);
                let current_label: folo_ui::SharedString =
                    opts.get(sel_idx).map(|(l, _)| l.clone()).unwrap_or_else(|| "Agent".into());
                let agent_ids: Vec<(LabelAgentId, LabelEffort)> =
                    list.iter().map(|a| (a.id, a.default_effort)).collect();
                let agent_sel = self.select(
                    "label-agent",
                    current_label,
                    opts,
                    sel_idx,
                    window,
                    cx,
                    Rc::new(move |this, i, _, cx| {
                        if let Some(&(id, eff)) = agent_ids.get(i) {
                            this.label.agent_id = Some(id);
                            this.label.model_idx = 0;
                            this.label.effort = eff;
                            this.label.fast = false;
                            cx.notify();
                        }
                    }),
                );
                run_col = run_col.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(4.))
                        .child(text(ty::XS, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Agent"))
                        .child(agent_sel),
                );
                if let Some(a) = selected_agent {
                    if !a.available
                        && let Some(hint) = &a.hint
                    {
                        run_col = run_col
                            .child(text(ty::XS, t.warning_text).whitespace_normal().child(hint.clone()));
                    }
                    if !a.models.is_empty() {
                        let mut m_opts: Vec<(folo_ui::SharedString, Option<folo_ui::SharedString>)> = vec![(
                            match &a.default_model {
                                Some(d) => format!("Default ({d})").into(),
                                None => "Default".into(),
                            },
                            None,
                        )];
                        m_opts.extend(a.models.iter().map(|m| (m.label.clone().into(), None)));
                        let cur_m = self.label.model_idx.min(m_opts.len() - 1);
                        let cur_m_label =
                            m_opts.get(cur_m).map(|(l, _)| l.clone()).unwrap_or_else(|| "Default".into());
                        let model_sel = self.select(
                            "label-model",
                            cur_m_label,
                            m_opts,
                            cur_m,
                            window,
                            cx,
                            Rc::new(|this, i, _, cx| {
                                this.label.model_idx = i;
                                cx.notify();
                            }),
                        );
                        run_col = run_col.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .child(
                                    text(ty::XS, t.text_secondary)
                                        .font_weight(FontWeight::MEDIUM)
                                        .child("Model"),
                                )
                                .child(model_sel),
                        );
                    }
                    let efforts = [LabelEffort::Low, LabelEffort::Medium, LabelEffort::High];
                    let eff_idx = efforts.iter().position(|e| *e == self.label.effort).unwrap_or(0);
                    let eff_seg = self.segmented(
                        "label-effort",
                        &["Low", "Medium", "High"],
                        eff_idx,
                        window,
                        cx,
                        Rc::new(move |this, i, _, cx| {
                            this.label.effort = efforts[i.min(2)];
                            cx.notify();
                        }),
                    );
                    run_col = run_col.child(
                        div()
                            .flex()
                            .flex_col()
                            .gap(px(4.))
                            .child(
                                text(ty::XS, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Reasoning effort"),
                            )
                            .child(eff_seg),
                    );
                    if to_label > 0 && job.counts.labeled > 0 {
                        let scope_idx = if self.label.scope == LabelScope::All { 1 } else { 0 };
                        let scope_seg = self.segmented(
                            "label-scope",
                            &["Unlabeled", "All"],
                            scope_idx,
                            window,
                            cx,
                            Rc::new(|this, i, _, cx| {
                                this.label.scope =
                                    if i == 1 { LabelScope::All } else { LabelScope::Unlabeled };
                                cx.notify();
                            }),
                        );
                        run_col = run_col.child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .child(
                                    text(ty::XS, t.text_secondary)
                                        .font_weight(FontWeight::MEDIUM)
                                        .child("Scope"),
                                )
                                .child(scope_seg),
                        );
                    }
                }
            } else if let Some(err) = self.label.agents.error() {
                run_col =
                    run_col.child(text(ty::XS, t.danger_text).whitespace_normal().child(err.message.clone()));
            } else {
                run_col = run_col.child(text(ty::XS, t.text_tertiary).child("Checking installed agents…"));
            }

            run_col = run_col.child(start_btn);
            if let Some(err) = &self.label.start_error {
                run_col =
                    run_col.child(
                        div().p(px(8.)).rounded(px(radius::SM)).bg(t.danger_muted).child(
                            text(ty::XS, t.danger_text).whitespace_normal().child(err.message.clone()),
                        ),
                    );
            }
            if job.usage.calls > 0 {
                run_col = run_col.child(
                    text(ty::XS, t.text_tertiary)
                        .tnum()
                        .child(format!("Usage: {}", format_usage(&job.usage))),
                );
            }
            sections.push(run_col.into_any_element());
        }

        panel(t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(t, "Agent"))
            .children(sections)
            .into_any_element()
    }

    fn label_info_card(&self, t: &Theme, job: &LabelJob) -> AnyElement {
        let chips = job.classes.iter().enumerate().map(|(i, c)| {
            div()
                .flex()
                .items_center()
                .gap(px(5.))
                .h(px(22.))
                .px(px(8.))
                .rounded(px(radius::SM))
                .bg(t.bg_active)
                .child(div().size(px(7.)).rounded(px(2.)).bg(t.class(i as u32)))
                .child(text(ty::XS, t.text_primary).child(c.clone()))
        });
        panel(t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(t, "Classes & instructions"))
            .child(
                div()
                    .p(px(14.))
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .child(div().flex().flex_wrap().gap(px(5.)).children(chips))
                    .child(
                        text(
                            ty::XS,
                            if job.instructions.is_empty() { t.text_tertiary } else { t.text_secondary },
                        )
                        .whitespace_normal()
                        .child(if job.instructions.is_empty() {
                            "No extra instructions — the agent uses class names alone.".to_string()
                        } else {
                            job.instructions.clone()
                        }),
                    ),
            )
            .into_any_element()
    }

    fn label_logs_card(&self, t: &Theme, job_id: &str) -> AnyElement {
        let lines = self.label.logs.get(job_id).and_then(|l| l.ready());
        let body: AnyElement = match lines {
            None => div()
                .p(px(12.))
                .child(text(ty::XS, t.text_tertiary).child("Loading logs…"))
                .into_any_element(),
            Some(list) if list.is_empty() => div()
                .p(px(12.))
                .child(text(ty::XS, t.text_tertiary).child("No log entries yet."))
                .into_any_element(),
            Some(list) => {
                let tail = list.iter().rev().take(40).rev();
                div()
                    .id("label-logs-scroll")
                    .max_h(px(160.))
                    .overflow_y_scroll()
                    .track_scroll(&self.label.logs_scroll)
                    .p(px(10.))
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .children(tail.map(|line| {
                        let color = match line.level {
                            LogLevel::Error => t.danger_text,
                            LogLevel::Warn => t.warning_text,
                            LogLevel::Info => t.text_secondary,
                            LogLevel::Debug => t.text_tertiary,
                        };
                        text(ty::XS, color)
                            .font_family(crate::theme::MONO)
                            .whitespace_normal()
                            .child(line.message.clone())
                    }))
                    .into_any_element()
            }
        };
        panel(t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(t, "Activity log"))
            .child(body)
            .into_any_element()
    }
}
