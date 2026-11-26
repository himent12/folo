//! Benchmark Compare screen (`Route::BenchCompare`): 2 to 4 finished entries side by side and
//! overlaid against a selectable baseline, with apples-to-apples caveats, 95% confidence intervals,
//! robustness by condition, calibration/self-labeling summaries, and per-class AP50-95.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, Hsla, IntoElement, ParentElement, SharedString, Styled, Window, div, px,
    relative,
};

use super::format::{axis_name, calibration_none, pct, score_text};
use super::leaderboard::delta_text;
use crate::api::{Bench, BenchAxis, BenchEntry, BenchReadiness, BenchResult, Label};
use crate::app::{FoloApp, Route, act};
use crate::bench::{
    COMPARE_CLS_SHOWN, COMPARE_MAX, COMPARE_SLOTS, best_in_row, compare_caveats, compare_class_rows,
    compare_condition_rows, intervals_domain, intervals_verdict, is_comparable,
};
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::store::Load;
use crate::theme::{Theme, alpha, radius, theme, ty};
use crate::ui::*;

fn slot_color(t: &Theme, slot: usize) -> Hsla {
    t.chart_series[COMPARE_SLOTS[slot % COMPARE_SLOTS.len()]]
}

fn slot_swatch(color: Hsla) -> folo_ui::Div {
    div().w(px(12.)).h(px(4.)).flex_none().rounded(px(2.)).bg(color)
}

fn axis_value(e: &BenchEntry, a: BenchAxis) -> Option<f32> {
    e.score
        .as_ref()
        .and_then(|s| s.axes.iter().find(|x| x.axis == a))
        .and_then(|x| x.value)
        .filter(|v| v.is_finite())
        .map(|v| (v * 100.).clamp(0., 100.))
}

fn delta_badge(t: &Theme, val: f32, base: f32, decimals: usize, higher_is_better: bool) -> AnyElement {
    let (label, raw_tone) = delta_text(val, base, decimals);
    let tone = if higher_is_better {
        raw_tone
    } else {
        match raw_tone {
            Tone::Success => Tone::Danger,
            Tone::Danger => Tone::Success,
            other => other,
        }
    };
    let (_, fg, _) = tone.colors(t);
    let color = if tone == Tone::Neutral { t.text_tertiary } else { fg };
    text(ty::SM, color).tnum().child(label).into_any_element()
}

impl FoloApp {
    pub fn bench_compare_view(
        &mut self,
        id: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_bench(id, cx);
        if self.bench_unsupported() {
            let header = page_header(&t, "Compare entries", "Not available on this server", vec![]);
            let body = self.bench_unsupported_panel(&t, window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let b = self.bench.list.ready().and_then(|l| l.iter().find(|x| x.id == id)).cloned();
        let entries_load = self.bench.entries.get(id).cloned().unwrap_or_default();
        let (Some(b), Load::Ready(all_entries)) = (b, entries_load) else {
            let header = page_header(&t, "Compare entries", "", vec![]);
            let err = self
                .bench
                .entries
                .get(id)
                .and_then(|l| l.error())
                .or_else(|| self.bench.list.error())
                .or_else(|| self.bench_blocked_error())
                .cloned();
            let body = if let Some(e) = err {
                self.error_panel(&e, "bench-cmp-retry", window, cx)
            } else if self.bench.list.is_loading()
                || self.bench_discovering()
                || self.bench.entries.get(id).is_none_or(|l| l.is_loading())
            {
                panel(&t).child(loading_rows(&t, "cmp", 6)).into_any_element()
            } else {
                panel(&t)
                    .child(empty_state(
                        &t,
                        "icons/circle-alert.svg",
                        "Benchmark not found",
                        "It may have been deleted.",
                        Tone::Neutral,
                    ))
                    .into_any_element()
            };
            return self.page_frame(&t, header, body, cx);
        };

        let shown = self.resolved_compare_entries(id, &all_entries);
        let stale: Vec<bool> = shown.iter().map(|e| e.fingerprint != b.fingerprint).collect();
        let candidates: Vec<BenchEntry> = all_entries
            .iter()
            .filter(|e| is_comparable(e) && !shown.iter().any(|s| s.id == e.id))
            .cloned()
            .collect();

        let sub = format!(
            "{} · {}{}",
            b.name,
            counted(shown.len() as u64, "entry", "entries"),
            if stale.iter().any(|&x| x) {
                " · not all on current fingerprint"
            } else {
                " · current fingerprint"
            }
        );
        let bid_back = b.id.clone();
        let back = self
            .button(
                "bench-cmp-back",
                "Leaderboard",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Bench(bid_back.clone()), w, cx)),
            )
            .into_any_element();
        let header =
            detail_header(&t, text(ty::XL, t.text_primary).child("Compare entries"), sub, vec![back]);

        let chips_bar = self.compare_chips_bar(&t, &b, &shown, &candidates, window, cx);
        if shown.len() < 2 {
            let bid_empty = b.id.clone();
            let back_btn = self.button(
                "bench-cmp-empty-back",
                "Back to leaderboard",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Bench(bid_empty.clone()), w, cx)),
            );
            let title = if shown.is_empty() {
                "Pick two to four finished entries"
            } else {
                "Add one more entry to compare"
            };
            let desc = if candidates.is_empty() {
                format!("Run at least two entries on {} to compare them side by side.", b.name)
            } else {
                format!(
                    "Choose another finished entry of {} above, or pick entries on the leaderboard.",
                    b.name
                )
            };
            let body = div().flex().flex_col().gap(px(16.)).child(chips_bar).child(panel(&t).child(
                empty_state(&t, "icons/git-compare-arrows.svg", title, &desc, Tone::Neutral).child(back_btn),
            ));
            return self.page_frame(&t, header, body, cx);
        }

        // Trigger result loads for every compared entry.
        for e in &shown {
            self.ensure_bench_result(&b.id, &e.id, cx);
        }

        let caveats = compare_caveats(&shown, &b);
        let caveats_el =
            (!caveats.is_empty()).then(|| {
                panel(&t).bg(t.warning_muted).border_color(t.warning_solid).child(
                    div()
                        .px(px(16.))
                        .py(px(12.))
                        .flex()
                        .flex_col()
                        .gap(px(6.))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .child(icon("icons/triangle-alert.svg", 15., t.warning_text))
                                .child(
                                    text(ty::SM, t.warning_text)
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .child("Not like for like"),
                                ),
                        )
                        .children(caveats.into_iter().map(|c| {
                            text(ty::SM, t.text_primary).whitespace_normal().child(format!("• {c}"))
                        })),
                )
            });

        let side_by_side = self.compare_side_by_side(&t, &b, &shown, &stale, window, cx);
        let intervals = self.compare_intervals_panel(&t, &b, &shown);
        let details = self.compare_details_section(&t, &b, &shown, &stale, window, cx);

        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(chips_bar)
            .children(caveats_el)
            .child(side_by_side)
            .child(intervals)
            .child(details);
        self.page_frame(&t, header, body, cx)
    }

    fn compare_chips_bar(
        &mut self,
        t: &Theme,
        b: &Bench,
        shown: &[BenchEntry],
        candidates: &[BenchEntry],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let mut bar = div().flex().flex_wrap().items_center().gap(px(8.));
        for (slot, e) in shown.iter().enumerate() {
            let color = slot_color(&t, slot);
            let is_base = slot == 0;
            let (bid_open, eid_open) = (b.id.clone(), e.id.clone());
            let open_btn = self.icon_button(
                &format!("bench-cmp-open-{}", e.id),
                "icons/arrow-up-right.svg",
                "Open entry result",
                window,
                cx,
                act(move |this, w, cx| {
                    this.navigate(Route::BenchEntry(bid_open.clone(), eid_open.clone()), w, cx)
                }),
            );
            let (bid_rm, eid_rm) = (b.id.clone(), e.id.clone());
            let rm_btn = self.icon_button(
                &format!("bench-cmp-rm-{}", e.id),
                "icons/x.svg",
                "Remove from comparison",
                window,
                cx,
                act(move |this, _, cx| this.compare_remove_entry(&bid_rm, &eid_rm, cx)),
            );
            let score_str = e.score.as_ref().map_or("–".into(), |s| score_text(s.total));
            let chip = div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(10.))
                .py(px(6.))
                .rounded(px(radius::MD))
                .bg(t.bg_panel)
                .border_1()
                .border_color(if is_base { t.accent_line } else { t.line_default })
                .child(slot_swatch(color))
                .child(
                    text(ty::SM, t.text_primary)
                        .font_weight(FontWeight::MEDIUM)
                        .max_w(px(180.))
                        .ellipsis()
                        .child(e.name.clone()),
                )
                .child(text(ty::SM, t.text_secondary).tnum().child(score_str))
                .when(is_base, |d| d.child(badge(&t, "baseline", Tone::Accent)))
                .when(e.score.as_ref().is_some_and(|s| s.partial), |d| {
                    d.child(badge(&t, "partial", Tone::Neutral))
                })
                .when(e.fingerprint != b.fingerprint, |d| d.child(badge(&t, "earlier", Tone::Warning)))
                .child(div().flex().items_center().gap(px(2.)).child(open_btn).child(rm_btn));
            bar = bar.child(chip);
        }
        if shown.len() < COMPARE_MAX && !candidates.is_empty() {
            let cand_ids: Vec<String> = candidates.iter().map(|c| c.id.clone()).collect();
            let options: Vec<(SharedString, Option<SharedString>)> = candidates
                .iter()
                .map(|c| {
                    let detail = c.score.as_ref().map(|s| format!("score {}", score_text(s.total)).into());
                    (c.name.clone().into(), detail)
                })
                .collect();
            let bid = b.id.clone();
            let adder = self.select(
                "bench-cmp-add",
                "Add entry…".into(),
                options,
                candidates.len(),
                window,
                cx,
                std::rc::Rc::new(move |this, i, _, cx| {
                    if let Some(eid) = cand_ids.get(i).cloned() {
                        this.compare_add_entry(&bid, eid, cx);
                    }
                }),
            );
            bar = bar.child(div().w(px(200.)).child(adder));
        }
        bar.into_any_element()
    }

    fn compare_side_by_side(
        &mut self,
        t: &Theme,
        b: &Bench,
        shown: &[BenchEntry],
        stale: &[bool],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let compact = self.content_width(window) < 880.;
        let lead_w = if compact { 128. } else { 164. };

        // Header row: Measure + each entry with Baseline / Make baseline control.
        let mut head = div()
            .flex()
            .items_end()
            .gap(px(12.))
            .px(px(16.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().w(px(lead_w)).flex_none().child(caption(&t, "Measure")));
        for (i, e) in shown.iter().enumerate() {
            let color = slot_color(&t, i);
            let role_el: AnyElement = if i == 0 {
                text(ty::SM, t.accent_text).child("Baseline").into_any_element()
            } else {
                let (bid, eid) = (b.id.clone(), e.id.clone());
                self.button(
                    &format!("bench-cmp-base-{}", e.id),
                    "Make baseline",
                    Btn::Secondary,
                    None,
                    window,
                    cx,
                    act(move |this, _, cx| this.compare_make_baseline(&bid, &eid, cx)),
                )
                .into_any_element()
            };
            head = head.child(
                div()
                    .flex_1()
                    .min_w(px(110.))
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap(px(4.))
                    .child(
                        div()
                            .flex()
                            .items_start()
                            .justify_end()
                            .gap(px(6.))
                            .w_full()
                            .child(div().pt(px(7.)).child(slot_swatch(color)))
                            .child(
                                text(ty::SM, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .flex_shrink()
                                    .min_w_0()
                                    .whitespace_normal()
                                    .child(e.name.clone()),
                            ),
                    )
                    .child(role_el),
            );
        }

        let make_row = |label: &str,
                        values: Vec<Option<f32>>,
                        decimals: usize,
                        higher_is_better: bool,
                        bold: bool,
                        fmt: fn(f32) -> String,
                        suffix: &'static str| {
            let best = best_in_row(&values, stale, decimals, higher_is_better);
            let base_val = if stale[0] { None } else { values[0] };
            let mut row = div()
                .flex()
                .items_center()
                .gap(px(12.))
                .px(px(16.))
                .py(px(10.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    text(ty::SM, if bold { t.text_primary } else { t.text_secondary })
                        .when(bold, |d| d.font_weight(FontWeight::SEMIBOLD))
                        .w(px(lead_w))
                        .flex_none()
                        .ellipsis()
                        .child(label.to_string()),
                );
            for (i, val) in values.iter().enumerate() {
                let is_best = best[i];
                let cell: AnyElement = match val {
                    None => text(ty::SM, t.text_tertiary).child("–").into_any_element(),
                    Some(v) => {
                        let main = div()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap(px(6.))
                            .when(is_best, |d| d.child(badge(&t, "best", Tone::Success)))
                            .child(
                                text(
                                    if bold { ty::MD } else { ty::SM },
                                    if stale[i] { t.text_tertiary } else { t.text_primary },
                                )
                                .tnum()
                                .when(bold || is_best, |d| d.font_weight(FontWeight::SEMIBOLD))
                                .child(format!("{}{suffix}", fmt(*v))),
                            );
                        let delta = (i > 0 && !stale[i])
                            .then(|| base_val.map(|bv| delta_badge(&t, *v, bv, decimals, higher_is_better)))
                            .flatten();
                        div()
                            .flex()
                            .flex_col()
                            .items_end()
                            .gap(px(1.))
                            .child(main)
                            .children(delta)
                            .into_any_element()
                    }
                };
                row = row.child(div().flex_1().min_w(px(110.)).flex().justify_end().child(cell));
            }
            row
        };

        let mut rows = div().flex().flex_col();
        // Score row
        let scores: Vec<Option<f32>> =
            shown.iter().map(|e| e.score.as_ref().map(|s| s.total).filter(|v| v.is_finite())).collect();
        rows = rows.child(make_row("Score", scores, 1, true, true, score_text, ""));
        // Five axes
        for a in BenchAxis::ALL {
            let vals: Vec<Option<f32>> = shown.iter().map(|e| axis_value(e, a)).collect();
            rows = rows.child(make_row(axis_name(a), vals, 0, true, false, |v| format!("{v:.0}"), ""));
        }
        // mAP50-95
        let maps: Vec<Option<f32>> = shown.iter().map(|e| e.map50_95.filter(|v| v.is_finite())).collect();
        rows = rows.child(make_row("mAP50-95", maps, 3, true, false, f3, ""));
        // Speed (ms/image — lower is faster, informational)
        let speeds: Vec<Option<f32>> = shown
            .iter()
            .map(|e| e.speed.as_ref().map(|s| s.ms_per_image).filter(|v| v.is_finite()))
            .collect();
        if speeds.iter().any(Option::is_some) {
            rows = rows.child(make_row("Speed", speeds, 1, false, false, |v| format!("{v:.1}"), " ms"));
        }

        panel(&t)
            .overflow_hidden()
            .child(
                panel_header(&t, "Side by side")
                    .child(text(ty::SM, t.text_tertiary).child("against the baseline (first column)")),
            )
            .child(head)
            .child(rows)
            .into_any_element()
    }

    fn compare_intervals_panel(&self, t: &Theme, b: &Bench, shown: &[BenchEntry]) -> AnyElement {
        let t = *t;
        let Some((lo, hi)) = intervals_domain(shown) else {
            return div().into_any_element();
        };
        let span = (hi - lo).max(1e-4);
        let frac = |v: f32| ((v - lo) / span).clamp(0.0, 1.0);
        let mut rows = div().px(px(16.)).py(px(12.)).flex().flex_col().gap(px(10.));
        for (slot, e) in shown.iter().enumerate() {
            let color = slot_color(&t, slot);
            let m = e.map50_95.filter(|v| v.is_finite());
            let ci = e.ci95.filter(|c| c[0].is_finite() && c[1].is_finite());
            let track = div()
                .relative()
                .flex_1()
                .min_w(px(140.))
                .h(px(20.))
                .flex()
                .items_center()
                .child(div().w_full().h(px(2.)).rounded(px(1.)).bg(t.line_subtle))
                .when_some(ci, |d, [a, c]| {
                    let left = frac(a.min(c));
                    let right = frac(a.max(c));
                    d.child(
                        div()
                            .absolute()
                            .top(px(7.))
                            .left(relative(left))
                            .w(relative((right - left).max(0.008)))
                            .h(px(6.))
                            .rounded(px(3.))
                            .bg(alpha(color, 0.35))
                            .border_1()
                            .border_color(color),
                    )
                })
                .when_some(m, |d, mv| {
                    d.child(
                        div()
                            .absolute()
                            .top(px(6.))
                            .left(relative(frac(mv)))
                            .ml(px(-4.))
                            .size(px(8.))
                            .rounded_full()
                            .bg(color),
                    )
                });
            let ci_text = ci.map_or_else(|| "–".into(), |[a, c]| format!("{} – {}", f3(a), f3(c)));
            rows = rows.child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(
                        div()
                            .w(px(164.))
                            .flex_none()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(slot_swatch(color))
                            .child(text(ty::SM, t.text_primary).ellipsis().child(e.name.clone())),
                    )
                    .child(
                        text(ty::SM, t.text_primary)
                            .tnum()
                            .w(px(46.))
                            .flex_none()
                            .flex()
                            .justify_end()
                            .child(m.map_or_else(|| "–".into(), f3)),
                    )
                    .child(track)
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .tnum()
                            .w(px(104.))
                            .flex_none()
                            .flex()
                            .justify_end()
                            .child(ci_text),
                    ),
            );
        }
        // Axis scale labels aligned with track.
        let mid = (lo + hi) * 0.5;
        rows = rows
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(div().w(px(222.)).flex_none())
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .justify_between()
                            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("{lo:.2}")))
                            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("{mid:.2}")))
                            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("{hi:.2}"))),
                    )
                    .child(div().w(px(104.)).flex_none()),
            )
            .child(
                text(ty::SM, t.text_secondary)
                    .whitespace_normal()
                    .pt(px(4.))
                    .child(intervals_verdict(shown, b)),
            );
        panel(&t)
            .child(
                panel_header(&t, "mAP50-95 with 95% intervals")
                    .child(text(ty::SM, t.text_tertiary).child("unchanged images")),
            )
            .child(rows)
            .into_any_element()
    }

    fn compare_details_section(
        &mut self,
        t: &Theme,
        b: &Bench,
        shown: &[BenchEntry],
        stale: &[bool],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let loads: Vec<Load<BenchResult>> = shown
            .iter()
            .map(|e| self.bench.results.get(&(b.id.clone(), e.id.clone())).cloned().unwrap_or_default())
            .collect();
        let failed: Vec<(usize, crate::api::ApiError)> =
            loads.iter().enumerate().filter_map(|(i, l)| l.error().map(|e| (i, e.clone()))).collect();
        let any_ready = loads.iter().any(|l| l.ready().is_some());
        let any_loading = loads.iter().any(|l| l.is_loading() || matches!(l, Load::Idle));
        if !any_ready {
            if let Some((idx, err)) = failed.first() {
                let (bid, eid, name) = (b.id.clone(), shown[*idx].id.clone(), shown[*idx].name.clone());
                let retry = self.button(
                    &format!("bench-cmp-res-retry-{eid}"),
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_bench_result(&bid, &eid, cx)),
                );
                return panel(&t)
                    .child(
                        empty_state(
                            &t,
                            "icons/circle-alert.svg",
                            &format!("Could not load result for {name}"),
                            &err.message,
                            Tone::Danger,
                        )
                        .child(retry),
                    )
                    .into_any_element();
            }
            if any_loading {
                return panel(&t).child(loading_rows(&t, "cmp-res", 5)).into_any_element();
            }
        }
        let failed_banner = (!failed.is_empty()).then(|| {
            let mut box_el = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(8.));
            for (idx, err) in &failed {
                let (bid, eid, name) = (b.id.clone(), shown[*idx].id.clone(), shown[*idx].name.clone());
                let retry = self.button(
                    &format!("bench-cmp-res-retry-{eid}"),
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| this.retry_bench_result(&bid, &eid, cx)),
                );
                box_el = box_el.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap(px(12.))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .min_w_0()
                                .child(icon("icons/circle-alert.svg", 14., t.danger_text))
                                .child(
                                    text(ty::SM, t.danger_text)
                                        .whitespace_normal()
                                        .child(format!("Could not load details for {name}: {}", err.message)),
                                ),
                        )
                        .child(retry),
                );
            }
            panel(&t).bg(t.danger_muted).border_color(t.danger_solid).child(box_el)
        });
        let results: Vec<Option<&BenchResult>> = loads.iter().map(|l| l.ready()).collect();

        // 1. Robustness by condition.
        let cond_rows = compare_condition_rows(b, &results);
        let no_rob: Vec<&str> = shown
            .iter()
            .enumerate()
            .filter(|(i, _)| results[*i].is_some_and(|r| r.robustness.is_none()))
            .map(|(_, e)| e.name.as_str())
            .collect();
        let robustness_panel = (!cond_rows.is_empty()).then(|| {
            let mut body = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(10.));
            for row in &cond_rows {
                let cname = self.bench_condition_name(&row.condition);
                let mut bars = div().flex_1().min_w_0().flex().flex_col().gap(px(4.));
                let base_v = if stale[0] { None } else { row.values[0] };
                for (slot, val) in row.values.iter().enumerate() {
                    let color = slot_color(&t, slot);
                    let bar_line = match val {
                        Some(v) => {
                            let delta = (slot > 0 && !stale[slot])
                                .then(|| base_v.map(|bv| delta_badge(&t, v * 100., bv * 100., 0, true)))
                                .flatten();
                            div()
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .h(px(16.))
                                .child(div().flex_1().min_w_0().child(progress(&t, *v, color, 5.)))
                                .child(
                                    text(ty::SM, t.text_primary)
                                        .tnum()
                                        .w(px(38.))
                                        .flex_none()
                                        .flex()
                                        .justify_end()
                                        .child(pct(*v)),
                                )
                                .child(div().w(px(36.)).flex_none().flex().justify_end().children(delta))
                        }
                        None => div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .h(px(16.))
                            .child(div().flex_1().min_w_0().child(progress(&t, 0.0, t.line_subtle, 5.)))
                            .child(
                                text(ty::SM, t.text_tertiary)
                                    .w(px(82.))
                                    .flex_none()
                                    .flex()
                                    .justify_end()
                                    .child("–"),
                            ),
                    };
                    bars = bars.child(bar_line);
                }
                body = body.child(
                    div()
                        .flex()
                        .items_start()
                        .gap(px(12.))
                        .child(text(ty::SM, t.text_secondary).w(px(140.)).flex_none().ellipsis().child(cname))
                        .child(bars),
                );
            }
            if !no_rob.is_empty() {
                body = body.child(
                    text(ty::SM, t.text_tertiary)
                        .whitespace_normal()
                        .child(format!("{} ran no conditions (no robustness bars).", no_rob.join(", "))),
                );
            }
            panel(&t)
                .child(
                    panel_header(&t, "Robustness by condition")
                        .child(text(ty::SM, t.text_tertiary).child("accuracy kept")),
                )
                .child(body)
        });

        // 2. Calibration & Self-labeling pair.
        let mut cal_body = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(8.));
        for (slot, e) in shown.iter().enumerate() {
            let color = slot_color(&t, slot);
            let cal = results[slot].and_then(|r| r.calibration.as_ref());
            let right_text = match (results[slot], cal) {
                (None, _) => "–".to_string(),
                (Some(_), Some(c)) => format!("ECE {} · {} dets", f3(c.ece), int(c.detections)),
                (Some(_), None) => calibration_none(e),
            };
            cal_body = cal_body.child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(slot_swatch(color))
                    .child(
                        text(ty::SM, t.text_primary).w(px(136.)).flex_none().ellipsis().child(e.name.clone()),
                    )
                    .child(div().flex_1().min_w(px(60.)).child(progress(
                        &t,
                        cal.map_or(0.0, |c| (1.0 - c.ece).clamp(0.0, 1.0)),
                        if cal.is_some() { color } else { t.line_subtle },
                        5.,
                    )))
                    .child(text(ty::SM, t.text_secondary).tnum().child(right_text)),
            );
        }
        let cal_panel = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(
                panel_header(&t, "Calibration")
                    .child(text(ty::SM, t.text_tertiary).child("lower ECE is better")),
            )
            .child(cal_body);

        let mut sl_body = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(8.));
        for (slot, e) in shown.iter().enumerate() {
            let color = slot_color(&t, slot);
            if let Some(sl) = results[slot].map(|r| &r.self_labeling) {
                let tone = match sl.readiness {
                    BenchReadiness::Ready => Tone::Success,
                    BenchReadiness::Partial => Tone::Warning,
                    BenchReadiness::NotReady => Tone::Neutral,
                };
                let thr = sl.threshold.map_or_else(|| "no t*".into(), |c| format!("t* {}", fmt_conf(c)));
                sl_body = sl_body.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(slot_swatch(color))
                        .child(
                            text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(e.name.clone()),
                        )
                        .child(badge(&t, sl.readiness.label(), tone))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .tnum()
                                .child(format!("{} recall · {thr}", pct(sl.recall))),
                        ),
                );
            } else {
                sl_body = sl_body.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .child(slot_swatch(color))
                        .child(
                            text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(e.name.clone()),
                        )
                        .child(text(ty::SM, t.text_tertiary).child("–")),
                );
            }
        }
        let sl_panel = panel(&t)
            .flex_1()
            .min_w(px(320.))
            .child(
                panel_header(&t, "Self-labeling")
                    .child(text(ty::SM, t.text_tertiary).child("objects labeled at 90% precision")),
            )
            .child(sl_body);

        let pair = div().flex().flex_wrap().gap(px(16.)).child(cal_panel).child(sl_panel);

        // 3. Per-class AP50-95 (hardest first).
        let cls_rows = compare_class_rows(b, &results);
        let total_cls = cls_rows.len();
        let show_all = self.bench.compare_all_classes;
        let visible_cls = if show_all { total_cls } else { total_cls.min(COMPARE_CLS_SHOWN) };
        let cls_meta = if total_cls > COMPARE_CLS_SHOWN && !show_all {
            format!("the {COMPARE_CLS_SHOWN} hardest of {total_cls}")
        } else {
            format!("{total_cls} classes, hardest first")
        };
        let mut cls_body = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(10.));
        for row in cls_rows.iter().take(visible_cls) {
            let mut bars = div().flex_1().min_w_0().flex().flex_col().gap(px(4.));
            for (slot, val) in row.values.iter().enumerate() {
                let color = slot_color(&t, slot);
                let v = val.unwrap_or(0.0);
                bars = bars.child(
                    div()
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .h(px(16.))
                        .child(div().flex_1().min_w_0().child(progress(
                            &t,
                            if val.is_some() { v } else { 0.0 },
                            if val.is_some() { color } else { t.line_subtle },
                            5.,
                        )))
                        .child(
                            text(ty::SM, if val.is_some() { t.text_primary } else { t.text_tertiary })
                                .tnum()
                                .w(px(46.))
                                .flex_none()
                                .flex()
                                .justify_end()
                                .child(val.map_or_else(|| "–".into(), f3)),
                        ),
                );
            }
            cls_body = cls_body.child(
                div()
                    .flex()
                    .items_start()
                    .gap(px(12.))
                    .child(
                        div()
                            .w(px(152.))
                            .flex_none()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .child(class_swatch(&t, row.class, 9.))
                            .child(text(ty::SM, t.text_secondary).ellipsis().child(format!(
                                "{} · {}",
                                row.name,
                                int(row.truths)
                            ))),
                    )
                    .child(bars),
            );
        }
        if total_cls > COMPARE_CLS_SHOWN {
            let label = if show_all {
                format!("Show {COMPARE_CLS_SHOWN} hardest")
            } else {
                format!("Show all {total_cls} classes")
            };
            cls_body = cls_body.child(div().pt(px(4.)).child(self.button(
                "bench-cmp-classes-toggle",
                label,
                Btn::Secondary,
                None,
                window,
                cx,
                act(|this, _, cx| {
                    this.bench.compare_all_classes = !this.bench.compare_all_classes;
                    cx.notify();
                }),
            )));
        }
        let cls_panel = (!cls_rows.is_empty()).then(|| {
            panel(&t)
                .child(
                    panel_header(&t, "AP50-95 per class")
                        .child(text(ty::SM, t.text_tertiary).child(cls_meta)),
                )
                .child(cls_body)
        });

        div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .children(failed_banner)
            .children(robustness_panel)
            .child(pair)
            .children(cls_panel)
            .into_any_element()
    }
}
