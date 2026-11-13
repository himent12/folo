//! A cloud run's panel: its phases, the current transfer and the machine it runs on.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Label, Run, RunExt, time};
use crate::app::{FoloApp, Route, act};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// A cloud run's panel: its phases (done, current with its transfer, to come), then the
    /// instance, the cost so far (final once released), the time limit and the last checkpoint
    /// copied here. Before training its title says what the run waits on. As Studio's
    /// (`web/src/routes/run/CloudPanel.tsx`), it says when the worker doesn't answer, and when a
    /// released run's instance isn't confirmed terminated (it may still be billed: releasing is
    /// still under way, and the cost is the instance's own, so far).
    pub(super) fn cloud_run_panel(
        &self,
        r: &Run,
        c: &crate::api::RunCloud,
        now: i64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        use crate::api::CloudPhase;
        let t = theme(cx);
        let provider = self.provider_name(&c.provider);
        let released = c.phase == CloudPhase::Released;
        let live = r.status.is_active();
        // Once released, the instance in the ledger: still alive means the provider hasn't
        // confirmed the termination.
        let ledger = self.released_instance(c);
        let unconfirmed = ledger.is_some_and(|i| i.status.is_alive());
        // Nothing from the ledger confirms the release yet (loading, or a read failed: then said).
        let unverified = self.released_instance_unverified(c);
        let unknown = self.released_instance_unknown(c);
        // Where it failed, and what it never reached, once released (not every phase done).
        let phases = crate::cloud::phase_states(c, r, unconfirmed || unverified);
        let count = phases.len();
        let steps = phases.into_iter().enumerate().map(|(i, (p, state))| {
            use crate::cloud::PhaseState;
            let done = state == PhaseState::Done;
            let current = state == PhaseState::Now;
            let failed = state == PhaseState::Failed;
            let last = i + 1 == count;
            let dot = if current {
                dot(t.accent_fill(), 8., live, format!("cloud-{}", r.id)).into_any_element()
            } else if failed {
                div().size(px(8.)).flex_none().rounded_full().bg(t.danger_text).into_any_element()
            } else if done {
                div().size(px(8.)).flex_none().rounded_full().bg(t.text_secondary).into_any_element()
            } else {
                div()
                    .size(px(8.))
                    .flex_none()
                    .rounded_full()
                    .border_1()
                    .border_color(t.line_strong)
                    .into_any_element()
            };
            div()
                .flex_1()
                .flex_basis(px(0.))
                .min_w_0()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(div().h(px(8.)).flex().items_center().gap(px(6.)).child(dot).when(!last, |d| {
                    d.child(div().flex_1().mr(px(6.)).h(px(2.)).rounded_full().bg(if done {
                        t.text_tertiary
                    } else {
                        t.line_default
                    }))
                }))
                .child(
                    text(
                        ty::SM,
                        if current {
                            t.text_primary
                        } else if failed {
                            t.danger_text
                        } else if done {
                            t.text_secondary
                        } else {
                            t.text_tertiary
                        },
                    )
                    .when(current, |d| d.font_weight(FontWeight::MEDIUM))
                    .whitespace_nowrap()
                    .child(p.label()),
                )
                // The phase's transfer: how much has moved, as text and a thin bar.
                .when_some(c.progress.filter(|_| current), |d, p| {
                    d.child(text(ty::SM, t.text_secondary).tnum().whitespace_nowrap().child(fmt_transfer(&p)))
                        .when_some(transfer_frac(&p), |d, f| {
                            d.child(div().w(px(96.)).child(progress(&t, f, t.accent_fill(), 2.)))
                        })
                })
        });
        // The facts: what it runs on (Settings › Cloud has its instance), what it costs, until when.
        // The instance's line takes the room the others leave (fixed widths for what may cut).
        let fact = |label: &str, width: Option<f32>, value: AnyElement| {
            div()
                .map(|d| match width {
                    Some(w) => d.w(px(w)).flex_none(),
                    None => d.flex_1().min_w_0(),
                })
                .flex()
                .flex_col()
                .gap(px(4.))
                .child(caption(&t, label))
                .child(value)
        };
        let value = |s: String| text(ty::SM, t.text_primary).tnum().ellipsis().child(s).into_any_element();
        let line = offer_line(&c.offer);
        let warning = crate::cloud::panel_warning(c, unconfirmed, unknown, now);
        let warn = warning.map(|(mark, words, to_settings)| {
            let line = warn_line(&t, mark, words);
            if !to_settings {
                return line;
            }
            line.child(
                self.bind(
                    div()
                        .id("cloud-warn-settings")
                        .flex_none()
                        .px(px(4.))
                        .mx(px(-4.))
                        .rounded(px(radius::SM))
                        .cursor_pointer()
                        .hover(|s| s.bg(t.bg_hover))
                        .child(text(ty::SM, t.accent_text).whitespace_nowrap().child("Settings › Cloud")),
                    "cloud-warn-settings",
                    radius::SM,
                    window,
                    cx,
                    act(|this, w, cx| this.navigate(Route::Settings, w, cx)),
                ),
            )
        });
        let instance = if c.instance_id.is_some() {
            let el = div()
                .id("cloud-instance")
                .max_w_full()
                .px(px(4.))
                .mx(px(-4.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .tooltip(crate::controls::tip("Its instance in Settings › Cloud", None))
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(4.))
                        .child(text(ty::SM, t.text_primary).min_w_0().ellipsis().child(line))
                        .child(icon("icons/chevron-right.svg", 12., t.text_secondary)),
                );
            self.bind(
                el,
                "cloud-instance",
                radius::SM,
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Settings, w, cx)),
            )
            .into_any_element()
        } else {
            value(line)
        };
        let cost = crate::cloud::panel_cost(c, ledger, unverified);
        let limit = match c.deadline.as_deref().and_then(time::parse).filter(|_| !released) {
            Some(d) => format!("{} · stops by {}", fmt_hours(c.max_hours), time::local_deadline(d, now)),
            None => fmt_hours(c.max_hours),
        };
        let synced =
            c.synced.as_deref().and_then(time::parse).map_or("none yet".to_string(), |s| time::ago(s, now));
        let title = match r.cloud_pending() {
            Some(c) => cloud_wait_title(c, &provider),
            None => "Cloud".to_string(),
        };
        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, &title))
            .child(div().flex().gap(px(8.)).px(px(16.)).pt(px(14.)).pb(px(12.)).children(steps))
            .children(warn.map(|w| w.px(px(16.)).py(px(8.)).border_t_1().border_color(t.line_subtle)))
            .child(
                div()
                    .flex()
                    .gap(px(24.))
                    .px(px(16.))
                    .py(px(12.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(fact("Instance", None, instance))
                    .child(fact("Cost", Some(120.), value(cost)))
                    .child(fact("Limit", Some(240.), value(limit)))
                    .child(fact("Last checkpoint copy", Some(140.), value(synced))),
            )
            .into_any_element()
    }
}

/// One line of trouble in the panel (a silent worker, a termination not confirmed, a ledger not
/// read): the mark in the warning hue, the words secondary, as Studio's `.warn-line`.
fn warn_line(t: &crate::theme::Theme, mark: &'static str, words: String) -> folo_ui::Div {
    div()
        .flex()
        .items_start()
        .gap(px(8.))
        .child(div().pt(px(2.)).flex_none().child(icon(mark, 14., t.warning_solid)))
        .child(text(ty::SM, t.text_secondary).flex_1().min_w_0().child(words))
}
