//! A benchmark's Loops tab and its table.

use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::cells::{name_with_badges, num_cell};
use super::format::{loop_tone, loop_trend};
use crate::api::{Bench, BenchLoop, BenchLoopStatus, BenchLoopVerdict, Label, time};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::loading_rows;
use crate::store::Load;
use crate::theme::{Theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn bench_loops(
        &mut self,
        t: &Theme,
        b: &Bench,
        loops: &Load<Vec<BenchLoop>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let form = self.bench.new_loop.is_some().then(|| self.new_loop_form(t, b, window, cx));
        let table = self.bench_loops_table(t, b, loops, window, cx);
        div().flex().flex_col().gap(px(16.)).children(form).child(table).into_any_element()
    }

    fn bench_loops_table(
        &mut self,
        t: &Theme,
        b: &Bench,
        loops: &Load<Vec<BenchLoop>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let new_btn = (self.bench.new_loop.is_none()).then(|| {
            self.button(
                "bench-loop-new",
                "New loop",
                Btn::Secondary,
                Some("icons/repeat.svg"),
                window,
                cx,
                act(|this, _, cx| this.open_new_loop(cx)),
            )
        });
        let header = panel_header(&t, "Self-improvement loops").child(
            div()
                .flex()
                .items_center()
                .gap(px(12.))
                .child(
                    text(ty::SM, t.text_tertiary)
                        .child("A model labels a pool, a student trains on those labels"),
                )
                .children(new_btn),
        );
        let list = match loops {
            Load::Ready(l) => l.clone(),
            Load::Failed(e) => {
                return panel(&t)
                    .child(header)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Loops didn't load",
                        &e.message,
                        Tone::Danger,
                    ))
                    .into_any_element();
            }
            _ => return panel(&t).child(header).child(loading_rows(&t, "loops", 3)).into_any_element(),
        };
        if list.is_empty() {
            return panel(&t)
                .child(header)
                .child(empty_state(
                    &t,
                    "icons/repeat.svg",
                    "No loops yet",
                    "A loop asks whether a model's own labels can train a better one: it labels a pool of images, a student trains on those labels alone, and this benchmark grades the student.",
                    Tone::Neutral,
                ))
                .into_any_element();
        }
        let now = self.store.api.now();
        let name_w = (self.content_width(window) - (220. + 84. + 116.) - 3. * 16. - 42.).clamp(120., 640.);
        let rows: Vec<AnyElement> = list
            .iter()
            .map(|l| {
                let key = format!("bench-loop-{}", l.id);
                let busy = self.bench.busy.contains(&l.id);
                let confirming = self
                    .bench
                    .confirm_delete
                    .as_ref()
                    .is_some_and(|(x, i)| *x == b.id && i.as_deref() == Some(l.id.as_str()));
                let total = l.config.generations.unwrap_or(1);
                let done_gens =
                    l.generations.iter().filter(|g| g.generation > 0 && g.score.is_some()).count();
                let trend = loop_trend(l);
                let state = match (l.status, l.phase) {
                    (BenchLoopStatus::Running, Some(p)) => format!(
                        "{} · generation {} of {total}",
                        p.label(),
                        (done_gens + 1).min(total as usize)
                    ),
                    (BenchLoopStatus::Failed, _) => l.error.clone().unwrap_or_else(|| "Failed".into()),
                    _ => format!("{done_gens} of {total} generations"),
                };
                let verdict = l.verdict.map(|v| {
                    let tone = match v {
                        BenchLoopVerdict::Improved => Tone::Success,
                        BenchLoopVerdict::Held => Tone::Neutral,
                        BenchLoopVerdict::Regressed => Tone::Danger,
                    };
                    (v.label(), tone)
                });
                let actions: AnyElement = if busy {
                    text(ty::SM, t.text_tertiary).child("…").into_any_element()
                } else if confirming {
                    div()
                        .flex()
                        .gap(px(4.))
                        .child(self.button(
                            &format!("{key}-keep"),
                            "Keep",
                            Btn::Secondary,
                            None,
                            window,
                            cx,
                            act(|this, _, cx| {
                                this.bench.confirm_delete = None;
                                cx.notify();
                            }),
                        ))
                        .child(self.button(
                            &format!("{key}-del-ok"),
                            "Delete",
                            Btn::DangerCommit,
                            None,
                            window,
                            cx,
                            act(|this, _, cx| this.confirm_bench_delete(cx)),
                        ))
                        .into_any_element()
                } else if l.status == BenchLoopStatus::Running {
                    let (bid, lid) = (b.id.clone(), l.id.clone());
                    self.icon_button(
                        &format!("{key}-stop"),
                        "icons/stop.svg",
                        "Stop this loop",
                        window,
                        cx,
                        act(move |this, _, cx| this.stop_bench_loop(&bid, &lid, cx)),
                    )
                    .into_any_element()
                } else if l.status.is_active() {
                    div().into_any_element()
                } else {
                    let (bid, lid) = (b.id.clone(), l.id.clone());
                    self.icon_button(
                        &format!("{key}-del"),
                        "icons/trash-2.svg",
                        "Delete this loop",
                        window,
                        cx,
                        act(move |this, _, cx| {
                            this.bench.confirm_delete = Some((bid.clone(), Some(lid.clone())));
                            cx.notify();
                        }),
                    )
                    .into_any_element()
                };
                let at = time::parse(&l.created);
                div()
                    .flex()
                    .items_center()
                    .gap(px(16.))
                    .px(px(16.))
                    // A failed loop's reason is read whole: its row grows to wrap it.
                    .min_h(px(56.))
                    .py(px(8.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(200.))
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .child({
                                let mut badges = vec![(l.status.label(), loop_tone(l.status))];
                                badges.extend(verdict);
                                name_with_badges(&t, window, &l.name, &badges, name_w)
                            })
                            .child(if l.status == BenchLoopStatus::Failed {
                                text(ty::SM, t.danger_text).child(state)
                            } else {
                                text(ty::SM, t.text_tertiary).ellipsis().child(state)
                            }),
                    )
                    .child(text(ty::SM, t.text_secondary).tnum().w(px(220.)).flex_none().child(trend))
                    .child(num_cell(
                        84.,
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .child(at.map_or(String::new(), |c| time::ago(c, now))),
                    ))
                    .child(div().w(px(116.)).flex_none().flex().justify_end().child(actions))
                    .into_any_element()
            })
            .collect();
        panel(&t).overflow_hidden().child(header).children(rows).into_any_element()
    }
}
