//! "Design with an agent" beside the plan: the request, then the proposal.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, IntoElement, ParentElement, Styled, Window, div, px,
};

use super::cells::num_cell;
use super::format::pct;
use crate::api::{Bench, LabelAgentId};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// "Design with an agent": pick the agent, add a note, ask; then the proposal, to use or discard.
    pub(super) fn design_panel(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let agents = self.bench_agents();
        let Some(f) = self.bench.design.as_ref() else { return div().into_any_element() };
        let (agent, asking, applying, error, proposal) =
            (f.agent, f.asking, f.applying, f.error.clone(), f.proposal.clone());
        let note_input = f.note.clone();
        let note_fh = note_input.read(cx).focus_handle(cx);
        let close = self.icon_button(
            "bench-design-close",
            "icons/x.svg",
            "Close",
            window,
            cx,
            act(|this, w, cx| this.close_bench_form(crate::bench::BenchForm::Design, w, cx)),
        );
        let title = if proposal.is_some() { "Proposed plan" } else { "Design the plan with an agent" };
        let header = panel_header(&t, title).child(close);
        let mut body = div().p(px(16.)).flex().flex_col().gap(px(14.));
        if let Some(d) = &proposal {
            let catalog = self.bench.catalog.ready().cloned();
            let cond_name = |id: &str| {
                catalog
                    .as_ref()
                    .and_then(|x| x.conditions.iter().find(|i| i.id == id))
                    .map_or(id.to_string(), |i| i.name.clone())
            };
            let total: f32 = d.plan.conditions.iter().map(|c| c.weight).sum::<f32>().max(1e-6);
            let current: Vec<&str> = b.plan.conditions.iter().map(|c| c.condition.as_str()).collect();
            if let Some(r) = &d.plan.rationale {
                body = body.child(text(ty::SM, t.text_secondary).child(r.clone()));
            }
            let rows = d.plan.conditions.iter().map(|c| {
                let sev = c.severities.iter().map(u8::to_string).collect::<Vec<_>>().join(", ");
                let new = !current.contains(&c.condition.as_str());
                div()
                    .flex()
                    .items_start()
                    .gap(px(12.))
                    .py(px(6.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(8.))
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(cond_name(&c.condition)),
                                    )
                                    .when(new, |d| d.child(badge(&t, "new", Tone::Accent))),
                            )
                            .children(c.why.clone().map(|w| text(ty::SM, t.text_tertiary).child(w))),
                    )
                    .child(
                        text(ty::SM, t.text_secondary)
                            .tnum()
                            .w(px(140.))
                            .flex_none()
                            .child(format!("severities {sev}")),
                    )
                    .child(num_cell(56., text(ty::SM, t.text_secondary).tnum().child(pct(c.weight / total))))
            });
            let dropped: Vec<String> = b
                .plan
                .conditions
                .iter()
                .filter(|c| !d.plan.conditions.iter().any(|x| x.condition == c.condition))
                .map(|c| cond_name(&c.condition))
                .collect();
            body = body.child(div().flex().flex_col().children(rows));
            if !dropped.is_empty() {
                body = body.child(
                    text(ty::SM, t.text_tertiary).child(format!("Leaves out: {}.", dropped.join(", "))),
                );
            }
            let cost = d.usage.cost_usd.map_or(String::new(), |c| format!(" · {}", fmt_cost(c)));
            body = body.child(
                text(ty::SM, t.text_tertiary)
                    .tnum()
                    .child(format!("{}{cost}", plural(d.usage.calls as usize, "call"))),
            );
            let use_btn = if applying {
                self.button_disabled(
                    "bench-design-use",
                    "Saving…",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    "Saving the plan",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-design-use",
                    "Use this plan",
                    Btn::Primary,
                    Some("icons/check.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.apply_design(cx)),
                )
            };
            let discard = self.button(
                "bench-design-discard",
                "Discard",
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, w, cx| this.close_bench_form(crate::bench::BenchForm::Design, w, cx)),
            );
            body = body.child(div().flex().gap(px(8.)).child(use_btn).child(discard));
        } else {
            let names: Vec<String> = agents.iter().map(|a| a.name.clone()).collect();
            let ids: Vec<LabelAgentId> = agents.iter().map(|a| a.id).collect();
            let sel = ids.iter().position(|a| *a == agent).unwrap_or(0);
            let picker = (!names.is_empty()).then(|| {
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                self.segmented(
                    "bench-design-agent",
                    &refs,
                    sel,
                    window,
                    cx,
                    std::rc::Rc::new(move |this, i, _, cx| {
                        if let (Some(f), Some(a)) = (this.bench.design.as_mut(), ids.get(i)) {
                            f.agent = *a;
                        }
                        cx.notify();
                    }),
                )
            });
            let note = div()
                .relative()
                .w(px(420.))
                .h(px(32.))
                .px(px(10.))
                .flex()
                .items_center()
                .rounded(px(radius::MD))
                .bg(t.bg_raised)
                .border_1()
                .border_color(if note_fh.is_focused(window) { t.accent_line } else { t.line_control })
                .text_size(px(ty::SM.size))
                .when(self.ring(&note_fh, window), |d| d.child(crate::app::focus_ring(&t, radius::MD)))
                .child(note_input);
            body = body
                .child(text(ty::SM, t.text_secondary).child(
                    "One request, no tools: the agent reads the description, the classes and a few of the images, and answers with a plan from the catalog. You see it here before anything changes.",
                ))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_start()
                        .gap(px(16.))
                        .when_some(picker, |d, p| {
                            d.child(div().flex().flex_col().gap(px(6.)).child(caption(&t, "Agent")).child(p))
                        })
                        .child(div().flex().flex_col().gap(px(6.)).child(caption(&t, "Note")).child(note)),
                );
            let who =
                agents.iter().find(|a| a.id == agent).map_or("the agent".to_string(), |a| a.name.clone());
            let ask = if agents.is_empty() {
                self.button_disabled(
                    "bench-design-ask",
                    "Ask",
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    "Install Claude Code or Codex on the server's machine",
                    window,
                    cx,
                )
            } else if asking {
                self.button_disabled(
                    "bench-design-ask",
                    format!("Asking {who}…"),
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    "Waiting for the plan",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-design-ask",
                    format!("Ask {who}"),
                    Btn::Primary,
                    Some("icons/sparkles.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.ask_design(cx)),
                )
            };
            body = body.child(div().flex().child(ask));
        }
        if let Some(e) = error {
            body = body.child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                    .child(text(ty::SM, t.danger_text).child(e.message)),
            );
            if let Some(h) = e.hint {
                body = body.child(text(ty::SM, t.text_tertiary).child(h));
            }
        }
        panel(&t).child(header).child(body).into_any_element()
    }
}
