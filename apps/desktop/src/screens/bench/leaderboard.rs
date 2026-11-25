//! A benchmark's Leaderboard tab: who can run, sortable ranked entries, baseline pinning with
//! `+/-` deltas, expandable error panels, and progress tooltips.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::cells::{name_with_badges, num_cell};
use super::format::{
    agent_name, axis_short, axis_text, contestant_line, entry_frac, partial_why, pct, ranks, score_text,
};
use crate::api::{Bench, BenchAxis, BenchEntry, BenchEntryStatus, Label};
use crate::app::{FoloApp, Route, act};
use crate::bench::{Contender, model_fits};
use crate::controls::Btn;
use crate::screens::{loading_rows, th};
use crate::store::Load;
use crate::theme::{Theme, radius, size as tsize, ty};
use crate::ui::*;

const SCORE_W: f32 = 68.;
const AXIS_W: [f32; 5] = [74., 70., 76., 64., 64.];
const ACTIONS_W: f32 = 112.;
const COL_GAP: f32 = 12.;

/// Sort column key for a benchmark axis.
pub(crate) fn axis_key(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "accuracy",
        BenchAxis::Robustness => "robustness",
        BenchAxis::SelfLabeling => "self_labeling",
        BenchAxis::Calibration => "calibration",
        BenchAxis::Localization => "localization",
    }
}

/// Numeric value for leaderboard sorting (Score on 0–100, axes on 0–100).
pub(crate) fn entry_metric(e: &BenchEntry, key: &str) -> Option<f32> {
    let score = e.score.as_ref()?;
    match key {
        "score" => Some(score.total),
        k => BenchAxis::ALL
            .iter()
            .find(|a| axis_key(**a) == k)
            .and_then(|a| score.axes.iter().find(|x| x.axis == *a))
            .and_then(|x| x.value)
            .map(|v| (v * 100.).clamp(0., 100.)),
    }
}

/// Sorts leaderboard `(entry, official_rank)` pairs by `sort`. Unmeasured entries always sort
/// after measured ones in both directions; ties preserve original server rank order.
pub(crate) fn sort_leaderboard(rows: &mut [(BenchEntry, Option<usize>)], sort: Option<(&'static str, bool)>) {
    let Some((key, desc)) = sort else { return };
    rows.sort_by(|(a, _), (b, _)| match (entry_metric(a, key), entry_metric(b, key)) {
        (Some(av), Some(bv)) => {
            if desc {
                bv.total_cmp(&av)
            } else {
                av.total_cmp(&bv)
            }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
}

/// Formats `val - base` with `decimals` places and a semantic tone, never emitting signed zeros.
pub(crate) fn delta_text(val: f32, base: f32, decimals: usize) -> (String, Tone) {
    let diff = val - base;
    let mag = format!("{:.decimals$}", diff.abs());
    if mag.bytes().all(|b| b == b'0' || b == b'.') {
        (mag, Tone::Neutral)
    } else if diff > 0. {
        (format!("+{mag}"), Tone::Success)
    } else {
        (format!("−{mag}"), Tone::Danger)
    }
}

/// Empty-leaderboard guidance copy, accounting for coding agents when no models fit yet.
pub(crate) fn empty_leaderboard_copy(has_models: bool, has_agents: bool) -> &'static str {
    match (has_models, has_agents) {
        (true, true) => {
            "Pick a model (every image across every condition) or a coding agent (unchanged sample images) above and run it."
        }
        (true, false) => {
            "Pick a model and run it: every image, unchanged and under each condition of the plan."
        }
        (false, true) => {
            "No trained models match this benchmark's classes yet — pick a coding agent above to run unchanged sample images, or train a model first."
        }
        (false, false) => {
            "Train a model first, then run it here: every image, unchanged and under each condition of the plan."
        }
    }
}

/// Full tooltip text for an active entry's progress bar (percentage, counts, condition, severity).
pub(crate) fn progress_tooltip(e: &BenchEntry, cond_name: Option<&str>) -> String {
    let where_at = match cond_name {
        Some(name) => format!("{name} · severity {}", e.progress.severity.unwrap_or(1)),
        None => "unchanged images".to_string(),
    };
    if e.progress.total > 0 {
        format!("{} ({} of {}) · {where_at}", pct(entry_frac(e)), int(e.progress.done), int(e.progress.total),)
    } else {
        format!("{} · {where_at}", pct(entry_frac(e)))
    }
}

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
        let has_agents = contenders.iter().any(|c| matches!(c, Contender::Agent(_)));
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
        let cols = |d: folo_ui::Stateful<folo_ui::Div>| d.flex().items_center().gap(px(COL_GAP)).px(px(16.));
        let sort = self.bench.leaderboard_sort;
        let score_th = self.sort_th_on(
            sort,
            |a| &mut a.bench.leaderboard_sort,
            "bench-sort",
            "score",
            "Score",
            SCORE_W,
            true,
            window,
            cx,
        );
        let axis_ths: Vec<AnyElement> = BenchAxis::ALL
            .iter()
            .zip(AXIS_W)
            .map(|(a, w)| {
                self.sort_th_on(
                    sort,
                    |app| &mut app.bench.leaderboard_sort,
                    "bench-sort",
                    axis_key(*a),
                    axis_short(*a),
                    w,
                    true,
                    window,
                    cx,
                )
            })
            .collect();
        let head = div()
            .flex()
            .items_center()
            .gap(px(COL_GAP))
            .px(px(16.))
            .h(px(36.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(num_cell(24., th(&t, "#")))
            .child(th(&t, "Entry").flex_1().min_w(px(160.)))
            .child(score_th)
            .children(axis_ths)
            .child(div().w(px(ACTIONS_W)).flex_none());
        let retry_anchor = match self.passing_stop("bench-board-retry", entries.error().is_some(), cx) {
            Some(fh) if entries.error().is_none() => Some(div().track_focus(&fh)),
            _ => None,
        };
        let list = match entries {
            Load::Ready(l) => l.clone(),
            Load::Failed(e) => {
                let bid = b.id.clone();
                let retry = self.button(
                    "bench-board-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_bench_entries(&bid, cx)),
                );
                return panel(&t)
                    .child(head_row)
                    .child(
                        empty_state(
                            &t,
                            "icons/circle-alert.svg",
                            "The leaderboard didn't load",
                            &e.message,
                            Tone::Danger,
                        )
                        .child(retry),
                    )
                    .into_any_element();
            }
            _ => {
                return panel(&t)
                    .child(head_row)
                    .children(retry_anchor)
                    .child(head)
                    .child(loading_rows(&t, "board", 4))
                    .into_any_element();
            }
        };
        if list.is_empty() {
            return panel(&t)
                .child(head_row)
                .children(retry_anchor)
                .child(empty_state(
                    &t,
                    "icons/play.svg",
                    "No entries yet",
                    empty_leaderboard_copy(has_models, has_agents),
                    Tone::Neutral,
                ))
                .into_any_element();
        }
        // Compute official ranks from server order before applying any column sort.
        let rank = ranks(b, &list);
        let mut ranked: Vec<(BenchEntry, Option<usize>)> = list.iter().cloned().zip(rank).collect();
        sort_leaderboard(&mut ranked, sort);
        let pinned = self
            .bench
            .pinned_entry
            .as_ref()
            .filter(|(bid, _)| bid == &b.id)
            .and_then(|(_, eid)| {
                list.iter().find(|e| e.id == *eid && e.status == BenchEntryStatus::Done && e.score.is_some())
            })
            .cloned();
        let rows: Vec<AnyElement> = ranked
            .iter()
            .map(|(e, rank)| self.leaderboard_row(&t, b, e, *rank, pinned.as_ref(), cols, window, cx))
            .collect();
        let note = "Axes on the score's 0–100 scale. Partial entries ran a part of the benchmark or missed an axis; they rank after complete ones.";
        let footer_actions = (pinned.is_some() || sort.is_some()).then(|| {
            let clear_pin = pinned.as_ref().map(|p| {
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).child(format!("Baseline: {}", p.name)))
                    .child(self.button(
                        "bench-pin-clear",
                        "Clear baseline",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| {
                            this.bench.pinned_entry = None;
                            cx.notify();
                        }),
                    ))
            });
            let reset_sort = sort.map(|_| {
                self.button(
                    "bench-sort-reset",
                    "Reset order",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(|this, _, cx| {
                        this.bench.leaderboard_sort = None;
                        cx.notify();
                    }),
                )
            });
            div().flex().flex_wrap().items_center().gap(px(8.)).children(clear_pin).children(reset_sort)
        });
        panel(&t)
            .overflow_hidden()
            .child(head_row)
            .children(retry_anchor)
            .child(head)
            .children(rows)
            .child(
                div()
                    .px(px(16.))
                    .py(px(10.))
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(px(12.))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .flex_1()
                            .min_w(px(220.))
                            .whitespace_normal()
                            .child(note),
                    )
                    .children(footer_actions),
            )
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn leaderboard_row(
        &mut self,
        t: &Theme,
        b: &Bench,
        e: &BenchEntry,
        rank: Option<usize>,
        pinned: Option<&BenchEntry>,
        cols: impl Fn(folo_ui::Stateful<folo_ui::Div>) -> folo_ui::Stateful<folo_ui::Div>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        // The entry column's width: what the number columns, their gaps and the row padding leave.
        let fixed_w = 24. + SCORE_W + AXIS_W.iter().sum::<f32>() + ACTIONS_W;
        let name_w = (self.content_width(window) - fixed_w - 8. * COL_GAP - 42.).clamp(120., 640.);
        let key = format!("bench-entry-{}", e.id);
        let done = e.status == BenchEntryStatus::Done;
        let failed = e.status == BenchEntryStatus::Failed;
        let stale = done && e.fingerprint != b.fingerprint;
        let busy = self.bench.busy.contains(&e.id);
        let confirming = self
            .bench
            .confirm_delete
            .as_ref()
            .is_some_and(|(x, i)| *x == b.id && i.as_deref() == Some(e.id.as_str()));
        let is_pinned = pinned.is_some_and(|p| p.id == e.id);
        let compare_base = pinned.filter(|p| p.id != e.id);
        let trimmed_err = e.error.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let has_err_detail = trimmed_err.is_some();
        let err_full = failed.then(|| trimmed_err.unwrap_or("Failed").to_string());
        let err_expanded =
            failed && has_err_detail && self.bench.expanded_errors.contains(&(b.id.clone(), e.id.clone()));
        let axis_of = |entry: &BenchEntry, a: BenchAxis| {
            entry.score.as_ref().and_then(|s| s.axes.iter().find(|x| x.axis == a)).and_then(|x| x.value)
        };
        let mut badges: Vec<(&str, Tone, Option<String>)> = Vec::new();
        let status = (!done).then(|| (StatusKind::of_entry(e.status), e.status.label()));
        if is_pinned {
            badges.push(("baseline", Tone::Accent, None));
        }
        if e.score.as_ref().is_some_and(|s| s.partial) {
            badges.push(("partial", Tone::Neutral, Some(partial_why(e, b))));
        }
        if stale {
            badges.push(("earlier version", Tone::Warning, None));
        }
        let name_line = name_with_badges(&t, window, &e.name, status, &badges, name_w, &key);
        // The second line: who it is, or how far it is, or why it failed.
        let detail: AnyElement = match e.status {
            BenchEntryStatus::Running | BenchEntryStatus::Queued | BenchEntryStatus::Stopping => {
                let cond_name = e.progress.condition.as_deref().map(|c| self.bench_condition_name(c));
                let at = cond_name.as_deref().map_or_else(
                    || "unchanged images".to_string(),
                    |name| format!("{name} · severity {}", e.progress.severity.unwrap_or(1)),
                );
                let tip_text = progress_tooltip(e, cond_name.as_deref());
                let bar_w = if name_w < 240. { 60. } else { 96. };
                div()
                    .id(SharedString::from(format!("{key}-prog")))
                    .tooltip(crate::controls::tip(tip_text, None))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .min_w_0()
                    // The table's 2 px rail, lime only while the entry runs (UNIFIED.md §2 Progress).
                    .child(div().w(px(bar_w)).flex_none().child(progress(
                        &t,
                        entry_frac(e),
                        if e.status == BenchEntryStatus::Running { t.accent_fill() } else { t.text_tertiary },
                        2.,
                    )))
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .flex_1()
                            .min_w_0()
                            .ellipsis()
                            .child(format!("{} · {at}", pct(entry_frac(e)))),
                    )
                    .into_any_element()
            }
            BenchEntryStatus::Failed => {
                let msg = err_full.clone().unwrap_or_else(|| "Failed".to_string());
                div()
                    .id(SharedString::from(format!("{key}-err-line")))
                    .min_w_0()
                    .tooltip(crate::controls::tip(msg.clone(), None))
                    .child(text(ty::SM, t.danger_text).ellipsis().child(msg))
                    .into_any_element()
            }
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
        let delta_el = |val: f32, base_val: f32, decimals: usize| {
            let (label, tone) = delta_text(val, base_val, decimals);
            let (_, fg, _) = tone.colors(&t);
            let color = if tone == Tone::Neutral { t.text_tertiary } else { fg };
            text(ty::SM, color).tnum().child(label)
        };
        let score: AnyElement = match &e.score {
            None => text(ty::SM, t.text_tertiary).child("–").into_any_element(),
            Some(s) => {
                let main = text(ty::MD, t.text_primary)
                    .tnum()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(score_text(s.total));
                let delta = compare_base
                    .and_then(|base| base.score.as_ref())
                    .map(|bs| delta_el(s.total, bs.total, 1));
                match delta {
                    Some(d) => div().flex().flex_col().items_end().child(main).child(d).into_any_element(),
                    None => main.into_any_element(),
                }
            }
        };
        // Actions: stop while it runs; pin/expand + delete once it has ended.
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
            let pin_btn = (done && e.score.is_some()).then(|| {
                let (bid, eid) = (b.id.clone(), e.id.clone());
                let pin_key = format!("{key}-pin");
                let tip_label = if is_pinned { "Unpin baseline" } else { "Pin as baseline" };
                let ring = self.ring(&self.fh(pin_key.clone(), cx), window);
                let el = div()
                    .id(SharedString::from(pin_key.clone()))
                    .size(px(tsize::CONTROL_MD))
                    .flex()
                    .flex_none()
                    .items_center()
                    .justify_center()
                    .rounded(px(radius::MD))
                    .cursor_pointer()
                    .when(is_pinned, |d| d.bg(t.accent_muted))
                    .hover(move |s| s.bg(if is_pinned { t.accent_muted } else { t.bg_hover }))
                    .active(move |s| s.bg(t.bg_active))
                    .tooltip(crate::controls::tip(tip_label, None))
                    .when(ring, |d| d.child(crate::controls::focus_tip(&t, tip_label, tsize::CONTROL_MD)))
                    .child(icon(
                        "icons/pin.svg",
                        15.,
                        if is_pinned { t.accent_text } else { t.text_secondary },
                    ));
                self.bind(
                    el,
                    &pin_key,
                    radius::MD,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let target = (bid.clone(), eid.clone());
                        if this.bench.pinned_entry.as_ref() == Some(&target) {
                            this.bench.pinned_entry = None;
                        } else {
                            this.bench.pinned_entry = Some(target);
                        }
                        cx.notify();
                    }),
                )
            });
            let err_btn = (failed && has_err_detail).then(|| {
                let (bid, eid) = (b.id.clone(), e.id.clone());
                self.icon_button(
                    &format!("{key}-err-toggle"),
                    if err_expanded { "icons/chevron-up.svg" } else { "icons/chevron-down.svg" },
                    if err_expanded { "Hide error details" } else { "Show full error" },
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let pair = (bid.clone(), eid.clone());
                        if !this.bench.expanded_errors.remove(&pair) {
                            this.bench.expanded_errors.insert(pair);
                        }
                        cx.notify();
                    }),
                )
            });
            let (bid, eid) = (b.id.clone(), e.id.clone());
            let del_btn = self.icon_button(
                &format!("{key}-del"),
                "icons/trash-2.svg",
                "Delete this entry",
                window,
                cx,
                act(move |this, _, cx| {
                    this.bench.confirm_delete = Some((bid.clone(), Some(eid.clone())));
                    cx.notify();
                }),
            );
            div()
                .flex()
                .items_center()
                .gap(px(4.))
                .children(pin_btn)
                .children(err_btn)
                .child(del_btn)
                .into_any_element()
        };
        let faded = !done || stale;
        let clickable = done || (failed && has_err_detail && !confirming && !busy);
        let row = cols(div().id(SharedString::from(key.clone())))
            .h(px(52.))
            .when(!err_expanded, |d| d.border_b_1().border_color(t.line_subtle))
            .when(clickable, |d| d.cursor_pointer().hover(|s| s.bg(t.bg_hover)).active(|s| s.bg(t.bg_active)))
            .child(num_cell(
                24.,
                text(ty::SM, t.text_tertiary).tnum().child(rank.map_or("–".to_string(), |r| r.to_string())),
            ))
            .child(
                div().flex_1().min_w(px(160.)).flex().flex_col().gap(px(2.)).child(name_line).child(detail),
            )
            .child(num_cell(SCORE_W, score))
            .children(BenchAxis::ALL.iter().zip(AXIS_W).map(|(a, w)| {
                let v = axis_of(e, *a);
                let main =
                    text(ty::SM, if faded || v.is_none() { t.text_tertiary } else { t.text_secondary })
                        .tnum()
                        .child(axis_text(v));
                let delta = v.and_then(|val| {
                    let base_v = compare_base.and_then(|base| axis_of(base, *a))?;
                    Some(delta_el((val * 100.).clamp(0., 100.), (base_v * 100.).clamp(0., 100.), 0))
                });
                let cell: AnyElement = match delta {
                    Some(d) => div().flex().flex_col().items_end().child(main).child(d).into_any_element(),
                    None => main.into_any_element(),
                };
                num_cell(w, cell)
            }))
            .child(
                div()
                    .w(px(ACTIONS_W))
                    .flex_none()
                    .flex()
                    .justify_end()
                    // Keep presses on row action buttons from starting a click on the outer row.
                    .on_mouse_down(folo_ui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                    .child(actions),
            );
        let row_el: AnyElement = if done {
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
        } else if failed && has_err_detail && !confirming && !busy {
            let (bid, eid) = (b.id.clone(), e.id.clone());
            self.bind(
                row,
                &key,
                0.,
                window,
                cx,
                act(move |this, _, cx| {
                    let pair = (bid.clone(), eid.clone());
                    if !this.bench.expanded_errors.remove(&pair) {
                        this.bench.expanded_errors.insert(pair);
                    }
                    cx.notify();
                }),
            )
            .into_any_element()
        } else {
            row.into_any_element()
        };
        if !err_expanded {
            return row_el;
        }
        let msg = err_full.unwrap_or_default();
        let copy_msg = msg.clone();
        let copy_btn = self.button(
            &format!("{key}-err-copy"),
            "Copy error",
            Btn::Secondary,
            Some("icons/copy.svg"),
            window,
            cx,
            act(move |this, _, cx| {
                cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_msg.clone()));
                this.toast(Tone::Neutral, "Copied error message", None, cx);
            }),
        );
        div()
            .flex()
            .flex_col()
            .child(row_el)
            .child(
                div()
                    .id(SharedString::from(format!("{key}-err-panel")))
                    .px(px(16.))
                    .py(px(10.))
                    .bg(t.danger_muted)
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .flex()
                    .items_start()
                    .gap(px(12.))
                    .child(text(ty::SM, t.danger_text).flex_1().min_w_0().whitespace_normal().child(msg))
                    .child(copy_btn),
            )
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, BenchProgress};

    fn dummy_entry(
        id: &str,
        status: BenchEntryStatus,
        total: Option<f32>,
        acc: Option<f32>,
        rob: Option<f32>,
    ) -> BenchEntry {
        use folo_api::{BenchAxisScore, BenchScore};
        let base = crate::api::mock::MockBackend::new(4200, true).bench_entries("bn_3c90").unwrap().remove(0);
        BenchEntry {
            id: id.into(),
            name: id.into(),
            status,
            progress: BenchProgress { done: 42, total: 100, condition: None, severity: None },
            score: total.map(|t| BenchScore {
                total: t,
                partial: false,
                axes: vec![
                    BenchAxisScore { axis: BenchAxis::Accuracy, value: acc, weight: 0.35 },
                    BenchAxisScore { axis: BenchAxis::Robustness, value: rob, weight: 0.25 },
                    BenchAxisScore { axis: BenchAxis::SelfLabeling, value: Some(0.5), weight: 0.15 },
                    BenchAxisScore { axis: BenchAxis::Calibration, value: Some(0.6), weight: 0.15 },
                    BenchAxisScore { axis: BenchAxis::Localization, value: Some(0.7), weight: 0.10 },
                ],
            }),
            error: (status == BenchEntryStatus::Failed).then(|| "worker OOM".into()),
            ..base
        }
    }

    #[test]
    fn sort_leaderboard_orders_by_axis_and_keeps_unmeasured_entries_last() {
        let mut rows = vec![
            (dummy_entry("a", BenchEntryStatus::Done, Some(80.0), Some(0.90), Some(0.50)), Some(1)),
            (dummy_entry("b", BenchEntryStatus::Done, Some(70.0), Some(0.70), Some(0.85)), Some(2)),
            (dummy_entry("c", BenchEntryStatus::Running, None, None, None), None),
            (dummy_entry("d", BenchEntryStatus::Failed, None, None, None), None),
        ];
        // Sort by robustness descending: b (85) > a (50), then c, d.
        sort_leaderboard(&mut rows, Some(("robustness", true)));
        let ids: Vec<&str> = rows.iter().map(|(e, _)| e.id.as_str()).collect();
        let official_ranks: Vec<Option<usize>> = rows.iter().map(|(_, r)| *r).collect();
        assert_eq!(ids, ["b", "a", "c", "d"]);
        assert_eq!(official_ranks, [Some(2), Some(1), None, None], "official ranks stay attached");

        // Sort by accuracy ascending: b (70) < a (90), unmeasured still last!
        sort_leaderboard(&mut rows, Some(("accuracy", false)));
        let ids: Vec<&str> = rows.iter().map(|(e, _)| e.id.as_str()).collect();
        assert_eq!(ids, ["b", "a", "c", "d"]);
    }

    #[test]
    fn delta_text_formats_positive_negative_and_zero_within_rounding() {
        assert_eq!(delta_text(75.4, 70.0, 1), ("+5.4".to_string(), Tone::Success));
        assert_eq!(delta_text(68.2, 70.0, 1), ("−1.8".to_string(), Tone::Danger));
        assert_eq!(delta_text(70.02, 70.0, 1), ("0.0".to_string(), Tone::Neutral));
        assert_eq!(delta_text(82.0, 75.0, 0), ("+7".to_string(), Tone::Success));
        assert_eq!(delta_text(70.0, 75.0, 0), ("−5".to_string(), Tone::Danger));
        assert_eq!(delta_text(75.3, 75.0, 0), ("0".to_string(), Tone::Neutral));
        // Banker's rounding at exact ±0.5 rounds to 0 and must stay neutral, not "+0"/"−0".
        assert_eq!(delta_text(75.5, 75.0, 0), ("0".to_string(), Tone::Neutral));
        assert_eq!(delta_text(74.5, 75.0, 0), ("0".to_string(), Tone::Neutral));
    }

    #[test]
    fn empty_leaderboard_copy_mentions_coding_agents_when_only_agents_are_available() {
        assert!(empty_leaderboard_copy(false, true).contains("coding agent"));
        assert!(empty_leaderboard_copy(true, true).contains("coding agent"));
        assert!(empty_leaderboard_copy(true, true).contains("unchanged sample images"));
        assert!(!empty_leaderboard_copy(true, false).contains("coding agent"));
        assert!(empty_leaderboard_copy(false, false).contains("Train a model first"));
    }

    #[test]
    fn progress_tooltip_includes_counts_condition_and_severity() {
        let mut e = dummy_entry("r", BenchEntryStatus::Running, None, None, None);
        assert_eq!(progress_tooltip(&e, None), "42% (42 of 100) · unchanged images");
        e.progress.condition = Some("low_contrast".into());
        e.progress.severity = Some(2);
        assert_eq!(progress_tooltip(&e, Some("Low contrast")), "42% (42 of 100) · Low contrast · severity 2");
    }
}
