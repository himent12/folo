//! A benchmark's Plan tab: its conditions, each previewed on the first image.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::num_cell;
use super::format::{agent_name, pct, plan_intro};
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

    /// A 72 × 48 thumbnail of the benchmark's first image, unchanged or under a condition. One that
    /// failed is marked so (not left looking as if it loads) and is itself a Retry: a click, Enter
    /// or Space asks for it again.
    fn condition_thumb(
        &mut self,
        t: &Theme,
        b: &Bench,
        condition: Option<(&str, u8)>,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let first =
            self.bench.previews.get(&b.id).and_then(|l| l.ready()).and_then(|p| p.images.first()).cloned();
        // The image clips inside; the control itself doesn't, so its focus ring shows whole.
        let frame = div().w(px(72.)).h(px(48.)).flex_none().rounded(px(radius::SM)).bg(t.bg_inset);
        let Some(im) = first else { return frame.into_any_element() };
        let img = self.bench_image(&b.id, &im.id, condition, cx);
        let failed = img.is_none() && self.bench_image_failed(&b.id, &im.id, condition);
        let painted =
            image_canvas(t, img, (im.width as f32, im.height as f32), Vec::new(), Fit::Cover, false, false);
        let painted = if failed { crate::screens::unavailable_over(t, painted, false) } else { painted };
        let frame = frame.child(div().size_full().rounded(px(radius::SM)).overflow_hidden().child(painted));
        let key = match condition {
            Some((c, s)) => format!("bench-preview-retry-{c}-{s}"),
            None => "bench-preview-retry".to_string(),
        };
        if !failed {
            return match self.passing_stop(&key, false, cx) {
                Some(fh) => frame.track_focus(&fh).into_any_element(),
                None => frame.into_any_element(),
            };
        }
        self.passing_stop(&key, true, cx);
        let (bid, iid, cond) = (b.id.clone(), im.id.clone(), condition.map(|(c, s)| (c.to_string(), s)));
        let el = frame
            .id(folo_ui::SharedString::from(key.clone()))
            .cursor_pointer()
            .tooltip(crate::controls::tip(format!("{} · Retry", crate::screens::TILE_UNAVAILABLE), None));
        self.bind(
            el,
            &key,
            radius::SM,
            window,
            cx,
            act(move |this, _, cx| {
                this.retry_bench_image(&bid, &iid, cond.as_ref().map(|(c, s)| (c.as_str(), *s)), cx)
            }),
        )
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
        // Studio's words: the default plan only when it is the catalog's.
        let default_len = catalog.as_ref().map(|c| c.default_plan.conditions.len());
        intro = intro.child(text(ty::SM, t.text_secondary).child(plan_intro(
            who.as_deref(),
            &b.plan,
            default_len,
        )));
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
            let thumb = self.condition_thumb(&t, b, None, window, cx);
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
            let shown = self
                .bench
                .plan_preview_sev
                .get(&(b.id.clone(), c.condition.clone()))
                .copied()
                .filter(|s| c.severities.contains(s))
                .unwrap_or_else(|| preview_severity(&c.severities));
            let thumb = self.condition_thumb(&t, b, Some((&c.condition, shown)), window, cx);
            let chips: Vec<AnyElement> = (1..=crate::api::BENCH_SEVERITIES)
                .map(|s| {
                    let on = c.severities.contains(&s);
                    let active = s == shown && on;
                    let key = format!("plan-sev-{}-{s}", c.condition);
                    let base = div()
                        .id(folo_ui::SharedString::from(key.clone()))
                        .size(px(22.))
                        .rounded(px(radius::XS))
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(if active {
                            t.accent_muted
                        } else if on {
                            t.bg_raised
                        } else {
                            t.bg_inset
                        })
                        .when(active, |d| d.border_1().border_color(t.accent_line))
                        .when(on && !active, |d| d.border_1().border_color(t.line_subtle))
                        .child(
                            text(
                                ty::SM,
                                if active {
                                    t.accent_text
                                } else if on {
                                    t.text_secondary
                                } else {
                                    t.text_disabled
                                },
                            )
                            .tnum()
                            .child(s.to_string()),
                        );
                    if on {
                        let bid = b.id.clone();
                        let cond_id = c.condition.clone();
                        let el = base
                            .cursor_pointer()
                            .hover(|st| st.bg(t.bg_hover))
                            .active(|st| st.bg(t.bg_active))
                            .tooltip(crate::controls::tip(format!("Preview {name} at severity {s}"), None));
                        self.bind(
                            el,
                            &key,
                            radius::XS,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.bench.plan_preview_sev.insert((bid.clone(), cond_id.clone()), s);
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    } else {
                        base.tooltip(crate::controls::tip(format!("Severity {s} is not in this plan"), None))
                            .into_any_element()
                    }
                })
                .collect();
            let sev = div().flex().items_center().gap(px(4.)).children(chips);
            rows.push(row(thumb, name, what, group, sev, pct(c.weight / total)));
        }
        let plan = panel(&t).overflow_hidden().child(header).child(intro).child(head).children(rows).child(
            div().px(px(16.)).py(px(10.)).child(
                text(ty::SM, t.text_tertiary)
                    .child("Click a severity (1–3) to preview the first image under it."),
            ),
        );
        div().flex().flex_col().gap(px(16.)).children(design).child(plan).into_any_element()
    }
}
