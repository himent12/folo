//! A benchmark's Leaderboard tab: who can run, the ranked entries and each entry's row.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::{name_with_badges, num_cell};
use super::format::{
    agent_name, axis_short, axis_text, contestant_line, entry_frac, entry_tone, pct, ranks, score_text,
};
use crate::api::{Bench, BenchAxis, BenchEntry, BenchEntryStatus, Label};
use crate::app::{FoloApp, Route, act};
use crate::bench::{Contender, model_fits};
use crate::controls::Btn;
use crate::screens::{loading_rows, th};
use crate::store::Load;
use crate::theme::{Theme, ty};
use crate::ui::*;

impl FoloApp {
    /// A contestant's name in the Run picker, and what it runs.
    fn contender_option(&self, c: &Contender) -> (SharedString, Option<SharedString>) {
        match c {
            Contender::Model(id) => self.store.model(id).map_or_else(
                || (id.clone().into(), None),
                |m| (m.name.clone().into(), Some(format!("{} · {} px", m.arch.as_str(), m.imgsz).into())),
            ),
            Contender::Agent(a) => {
                let agent = self.bench_agents().into_iter().find(|x| x.id == *a);
                let name = agent.as_ref().map_or_else(|| agent_name(*a).to_string(), |x| x.name.clone());
                let model = agent.and_then(|x| x.default_model).unwrap_or_else(|| "default model".into());
                (name.into(), Some(format!("agent · {model} · first images, unchanged").into()))
            }
        }
    }

    pub(super) fn bench_leaderboard(
        &mut self,
        t: &Theme,
        b: &Bench,
        entries: &Load<Vec<BenchEntry>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // Run a contestant: the picked one (the best model by default), or a coding agent.
        let contenders = self.bench_contenders(b);
        let has_models = contenders.iter().any(|c| matches!(c, Contender::Model(_)));
        let picked = self.bench_run_model(b);
        let picked_i = picked.as_ref().and_then(|p| contenders.iter().position(|c| c == p)).unwrap_or(0);
        // The models that can't run here are listed too, off, with the reason.
        let unfit: Vec<Contender> = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| !model_fits(&m.classes, &b.classes))
            .map(|m| Contender::Model(m.id.clone()))
            .collect();
        let mut run_controls = div().flex().items_center().gap(px(8.));
        if !contenders.is_empty() {
            let list: Vec<Contender> = contenders.iter().chain(&unfit).cloned().collect();
            let options: Vec<(SharedString, Option<SharedString>)> =
                list.iter().map(|c| self.contender_option(c)).collect();
            let off: Vec<Option<SharedString>> = (0..list.len())
                .map(|i| (i >= contenders.len()).then(|| "It knows none of this benchmark's classes".into()))
                .collect();
            let label = options.get(picked_i).map_or_else(|| "Model".into(), |o| o.0.clone());
            let select = self.select_with(
                "bench-run-model",
                label,
                options,
                picked_i,
                &off,
                window,
                cx,
                std::rc::Rc::new(move |this, i, _, cx| {
                    this.bench.run_model = list.get(i).cloned();
                    cx.notify();
                }),
            );
            let bid = b.id.clone();
            let run = if self.bench.starting {
                self.button_disabled(
                    "bench-run",
                    "Starting…",
                    Btn::Primary,
                    Some("icons/play.svg"),
                    "Starting the entry",
                    window,
                    cx,
                )
            } else {
                self.button(
                    "bench-run",
                    "Run",
                    Btn::Primary,
                    Some("icons/play.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.run_bench_model(&bid, cx)),
                )
            };
            run_controls = run_controls.child(div().w(px(220.)).child(select)).child(run);
        }
        let head_row = panel_header(&t, "Leaderboard").child(run_controls);
        let cols = |d: folo_ui::Stateful<folo_ui::Div>| d.flex().items_center().gap(px(16.)).px(px(16.));
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(num_cell(24., th(&t, "#")))
            .child(th(&t, "Entry").flex_1().min_w(px(180.)))
            .child(num_cell(64., th(&t, "Score")))
            .children(BenchAxis::ALL.iter().map(|a| num_cell(68., th(&t, axis_short(*a)))))
            .child(div().w(px(116.)).flex_none());
        let list = match entries {
            Load::Ready(l) => l.clone(),
            Load::Failed(e) => {
                return panel(&t)
                    .child(head_row)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "The leaderboard didn't load",
                        &e.message,
                        Tone::Danger,
                    ))
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(head_row)
                    .child(head)
                    .child(loading_rows(&t, "board", 4))
                    .into_any_element();
            }
        };
        if list.is_empty() {
            return panel(&t)
                .child(head_row)
                .child(empty_state(
                    &t,
                    "icons/play.svg",
                    "No entries yet",
                    if has_models {
                        "Pick a model and run it: every image, unchanged and under each condition of the plan."
                    } else {
                        "Train a model first, then run it here: every image, unchanged and under each condition of the plan."
                    },
                    Tone::Neutral,
                ))
                .into_any_element();
        }
        let rank = ranks(b, &list);
        let rows: Vec<AnyElement> = list
            .iter()
            .zip(rank)
            .map(|(e, rank)| self.leaderboard_row(&t, b, e, rank, cols, window, cx))
            .collect();
        let note = "Axes on the score's 0–100 scale. Partial entries ran a part of the benchmark or missed an axis; they rank after complete ones.";
        panel(&t)
            .overflow_hidden()
            .child(head_row)
            .child(head)
            .children(rows)
            .child(div().px(px(16.)).py(px(10.)).child(text(ty::SM, t.text_tertiary).child(note)))
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn leaderboard_row(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        rank: Option<usize>,
        cols: impl Fn(folo_ui::Stateful<folo_ui::Div>) -> folo_ui::Stateful<folo_ui::Div>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // The entry column's width: what the number columns, their gaps and the row padding leave.
        let name_w =
            (self.content_width(window) - (24. + 64. + 5. * 68. + 116.) - 8. * 16. - 42.).clamp(120., 640.);
        let key = format!("bench-entry-{}", e.id);
        let done = e.status == BenchEntryStatus::Done;
        let stale = done && e.fingerprint != b.fingerprint;
        let busy = self.bench.busy.contains(&e.id);
        let confirming = self
            .bench
            .confirm_delete
            .as_ref()
            .is_some_and(|(x, i)| *x == b.id && i.as_deref() == Some(e.id.as_str()));
        let axis = |a: BenchAxis| {
            e.score.as_ref().and_then(|s| s.axes.iter().find(|x| x.axis == a)).and_then(|x| x.value)
        };
        let mut badges: Vec<(&str, Tone)> = Vec::new();
        if !done {
            badges.push((e.status.label(), entry_tone(e.status)));
        }
        if e.score.as_ref().is_some_and(|s| s.partial) {
            badges.push(("partial", Tone::Neutral));
        }
        if stale {
            badges.push(("earlier version", Tone::Warning));
        }
        let name_line = name_with_badges(&t, window, &e.name, &badges, name_w);
        // The second line: who it is, or how far it is, or why it failed.
        let detail: AnyElement = match e.status {
            BenchEntryStatus::Running | BenchEntryStatus::Queued | BenchEntryStatus::Stopping => {
                let at = e.progress.condition.as_ref().map_or_else(
                    || "unchanged images".to_string(),
                    |c| {
                        format!(
                            "{} · severity {}",
                            self.bench_condition_name(c),
                            e.progress.severity.unwrap_or(1)
                        )
                    },
                );
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(div().w(px(120.)).flex_none().child(progress(
                        &t,
                        entry_frac(e),
                        t.accent_fill(),
                        4.,
                    )))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .ellipsis()
                            .child(format!("{} · {at}", pct(entry_frac(e)))),
                    )
                    .into_any_element()
            }
            BenchEntryStatus::Failed => text(ty::SM, t.danger_text)
                .ellipsis()
                .child(e.error.clone().unwrap_or_else(|| "Failed".to_string()))
                .into_any_element(),
            _ => text(ty::SM, t.text_tertiary)
                .ellipsis()
                .child({
                    let mut s = contestant_line(e);
                    if let Some(sp) = &e.speed {
                        s += &format!(" · {:.1} ms/image", sp.ms_per_image);
                    }
                    s
                })
                .into_any_element(),
        };
        let score = e.score.as_ref().map_or_else(
            || text(ty::SM, t.text_tertiary).child("–"),
            |s| {
                text(ty::MD, t.text_primary)
                    .tnum()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(score_text(s.total))
            },
        );
        // Actions: stop while it runs; delete (a second click confirms) once it has ended.
        let actions: AnyElement = if busy {
            text(ty::SM, t.text_tertiary).child("…").into_any_element()
        } else if confirming {
            let k = div()
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
                ));
            k.into_any_element()
        } else if e.status.is_active() && e.status != BenchEntryStatus::Stopping {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.icon_button(
                &format!("{key}-stop"),
                "icons/stop.svg",
                "Stop this entry",
                window,
                cx,
                act(move |this, _, cx| this.stop_bench_entry(&bid, &eid, cx)),
            )
            .into_any_element()
        } else if e.status.is_active() {
            div().into_any_element()
        } else {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.icon_button(
                &format!("{key}-del"),
                "icons/trash-2.svg",
                "Delete this entry",
                window,
                cx,
                act(move |this, _, cx| {
                    this.bench.confirm_delete = Some((bid.clone(), Some(eid.clone())));
                    cx.notify();
                }),
            )
            .into_any_element()
        };
        let faded = !done || stale;
        let row = cols(div().id(SharedString::from(key.clone())))
            .h(px(52.))
            .border_b_1()
            .border_color(t.line_subtle)
            .when(done, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active)))
            .child(num_cell(
                24.,
                text(ty::SM, t.text_tertiary).tnum().child(rank.map_or("–".to_string(), |r| r.to_string())),
            ))
            .child(
                div().flex_1().min_w(px(180.)).flex().flex_col().gap(px(2.)).child(name_line).child(detail),
            )
            .child(num_cell(64., score))
            .children(BenchAxis::ALL.iter().map(|a| {
                let v = axis(*a);
                num_cell(
                    68.,
                    text(ty::SM, if faded || v.is_none() { t.text_tertiary } else { t.text_secondary })
                        .tnum()
                        .child(axis_text(v)),
                )
            }))
            .child(div().w(px(116.)).flex_none().flex().justify_end().child(actions));
        if done {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.bind(
                row,
                &key,
                0.,
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::BenchEntry(bid.clone(), eid.clone()), w, cx)),
            )
            .into_any_element()
        } else {
            row.into_any_element()
        }
    }
}
