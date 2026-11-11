//! A benchmark's Plan tab: its conditions, each previewed on the first image.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use super::cells::num_cell;
use super::format::{agent_name, pct};
use crate::api::{Bench, BenchEntryStatus, Label};
use crate::app::{FoloApp, act};
use crate::bench::preview_severity;
use crate::controls::Btn;
use crate::screens::{image_canvas, th};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;
use crate::viz::Fit;

impl FoloApp {
    /// The plan can still change: no entry has finished or is under way (docs/BENCH.md).
    fn plan_open(&self, b: &Bench) -> bool {
        !b.active
            && self.bench.entries.get(&b.id).and_then(|l| l.ready()).is_some_and(|l| {
                l.iter().all(|e| e.status != BenchEntryStatus::Done && !e.status.is_active())
            })
    }

    /// A 72 × 48 thumbnail of the benchmark's first image, unchanged or under a condition.
    fn condition_thumb(
        &mut self,
        t: &Theme,
        b: &Bench,
        condition: Option<(&str, u8)>,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let first =
            self.bench.previews.get(&b.id).and_then(|l| l.ready()).and_then(|p| p.images.first()).cloned();
        let frame =
            div().w(px(72.)).h(px(48.)).flex_none().rounded(px(radius::SM)).overflow_hidden().bg(t.bg_inset);
        let Some(im) = first else { return frame.into_any_element() };
        let img = self.bench_image(&b.id, &im.id, condition, cx);
        frame
            .child(image_canvas(
                t,
                img,
                (im.width as f32, im.height as f32),
                Vec::new(),
                Fit::Cover,
                false,
                false,
            ))
            .into_any_element()
    }

    pub(super) fn bench_plan(
        &mut self,
        t: &Theme,
        b: &Bench,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        self.ensure_previews(&b.id, cx);
        let open = self.plan_open(b);
        let design_here = self.bench.design.as_ref().is_some_and(|f| f.bench == b.id);
        // Design with an agent: while the plan can change, on servers that run agents.
        let design_btn = (open && self.agents_on() && !design_here).then(|| {
            let bid = b.id.clone();
            self.button(
                "bench-design",
                "Design with an agent",
                Btn::Secondary,
                Some("icons/sparkles.svg"),
                window,
                cx,
                act(move |this, w, cx| this.open_design(&bid, w, cx)),
            )
        });
        let lock = (!open).then(|| {
            div()
                .flex()
                .items_center()
                .gap(px(6.))
                .child(icon("icons/lock.svg", 12., t.text_tertiary))
                .child(text(ty::SM, t.text_tertiary).child("Fixed: entries have run on it"))
        });
        let header = panel_header(&t, "Plan")
            .child(div().flex().items_center().gap(px(8.)).children(lock).children(design_btn));
        let catalog = self.bench.catalog.ready().cloned();
        let cond =
            |id: &str| catalog.as_ref().and_then(|c| c.conditions.iter().find(|x| x.id == id).cloned());
        let total: f32 = b.plan.conditions.iter().map(|c| c.weight).sum::<f32>().max(1e-6);
        let who = b.plan.designed_by.as_ref().map(|d| {
            format!(
                "Designed by {}{}",
                agent_name(d.agent),
                d.model.as_ref().map_or(String::new(), |m| format!(" ({m})"))
            )
        });
        let mut intro = div()
            .px(px(16.))
            .py(px(12.))
            .flex()
            .flex_col()
            .gap(px(4.))
            .border_b_1()
            .border_color(t.line_subtle);
        intro = intro.child(
            text(ty::SM, t.text_secondary).child(match &who {
                Some(w) => {
                    format!("{w}. Each condition changes every image the same way for every contestant.")
                }
                None => {
                    "The default plan. Each condition changes every image the same way for every contestant."
                        .to_string()
                }
            }),
        );
        if let Some(r) = &b.plan.rationale {
            intro = intro.child(text(ty::SM, t.text_tertiary).child(r.clone()));
        }
        let design = design_here.then(|| self.design_panel(&t, b, window, cx));
        if b.plan.conditions.is_empty() {
            let plan = panel(&t).child(header).child(empty_state(
                &t,
                "icons/sliders-horizontal.svg",
                "Unchanged images only",
                "This plan runs no conditions, so no entry has a robustness axis.",
                Tone::Neutral,
            ));
            return div().flex().flex_col().gap(px(16.)).children(design).child(plan).into_any_element();
        }
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(72.)).flex_none().child(th(&t, "Preview")))
            .child(th(&t, "Condition").flex_1().min_w(px(220.)))
            .child(th(&t, "Group").w(px(84.)).flex_none())
            .child(th(&t, "Severities").w(px(150.)).flex_none())
            .child(num_cell(110., th(&t, "Share of robustness")));
        let row =
            |thumb: AnyElement, name: String, what: String, group: &str, sev: folo_ui::Div, share: String| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    .py(px(8.))
                    .min_h(px(64.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(thumb)
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(220.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(name))
                            .child(text(ty::SM, t.text_tertiary).child(what)),
                    )
                    .child(text(ty::SM, t.text_secondary).w(px(84.)).flex_none().child(group.to_string()))
                    .child(sev.w(px(150.)).flex_none())
                    .child(num_cell(110., text(ty::SM, t.text_secondary).tnum().child(share)))
                    .into_any_element()
            };
        let mut rows = vec![{
            let thumb = self.condition_thumb(&t, b, None, cx);
            row(
                thumb,
                "Unchanged".into(),
                "The images as they were frozen: accuracy is measured on these.".into(),
                "",
                div(),
                String::new(),
            )
        }];
        for c in &b.plan.conditions {
            let info = cond(&c.condition);
            let name = info.as_ref().map_or(c.condition.clone(), |i| i.name.clone());
            let what =
                c.why.clone().or_else(|| info.as_ref().map(|i| i.description.clone())).unwrap_or_default();
            let group = info.as_ref().map_or("", |i| i.group.label());
            let shown = preview_severity(&c.severities);
            let thumb = self.condition_thumb(&t, b, Some((&c.condition, shown)), cx);
            let sev = div().flex().gap(px(4.)).children((1..=crate::api::BENCH_SEVERITIES).map(|s| {
                let on = c.severities.contains(&s);
                div()
                    .size(px(20.))
                    .rounded(px(radius::XS))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(if on { t.accent_muted } else { t.bg_inset })
                    .when(s == shown && on, |d| d.border_1().border_color(t.accent_line))
                    .child(
                        text(ty::SM, if on { t.accent_text } else { t.text_disabled })
                            .tnum()
                            .child(s.to_string()),
                    )
            }));
            rows.push(row(thumb, name, what, group, sev, pct(c.weight / total)));
        }
        let plan = panel(&t).overflow_hidden().child(header).child(intro).child(head).children(rows).child(
            div().px(px(16.)).py(px(10.)).child(
                text(ty::SM, t.text_tertiary)
                    .child("Previews show the first image at the outlined severity."),
            ),
        );
        div().flex().flex_col().gap(px(16.)).children(design).child(plan).into_any_element()
    }
}
