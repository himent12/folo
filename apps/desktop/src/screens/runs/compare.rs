//! Side-by-side Multi-Run Compare screen (`Route::RunCompare`): overlaid mAP & training loss
//! curves, selectable baseline, per-run chip isolation, and hyperparameter diff table.

use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, Hsla, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::compare_data::{
    CompareMapMetric, CompareXAxis, LossSeriesCacheKey, RUN_COMPARE_MAX, build_loss_points, build_map_points,
    default_compare_runs, reconcile_compare_runs, same_dataset, shared_x_max,
};
use crate::api::{Run, RunStatus};
use crate::app::{FoloApp, Route, act};
use crate::charts::{End, HoverFn, LineChart, Series, line_chart};
use crate::controls::Btn;
use crate::screens::{detail_header, loading_rows, page_header};
use crate::theme::{Theme, alpha, radius, theme, ty};
use crate::ui::*;

/// EMA smoothing factor matching `screens/runs/charts.rs` and Studio's default.
const COMPARE_LOSS_EMA: f32 = 0.98;

/// Colour slots for up to 4 compared runs (matching `BenchCompare` palette order).
const RUN_SLOTS: [usize; 4] = [0, 1, 2, 4];

pub(super) fn run_slot_color(t: &Theme, slot: usize) -> Hsla {
    t.chart_series[RUN_SLOTS[slot % RUN_SLOTS.len()]]
}

pub(super) fn slot_swatch(color: Hsla, slot: usize) -> AnyElement {
    crate::ui::line_swatch(color, slot == 1, slot == 2, if slot == 0 { 2.5 } else { 2.0 }).into_any_element()
}

fn log_range(lo: f32, hi: f32) -> (f32, f32) {
    if !lo.is_finite() || hi.is_nan() || hi <= 0. {
        return (0.5, 4.);
    }
    let lo = lo.max(1e-6);
    let y0 = 2f32.powi(lo.log2().floor() as i32);
    let y1 = 2f32.powi((hi.max(lo) * 1.05).log2().ceil() as i32).max(y0 * 2.);
    (y0, y1)
}

fn linear_top(hi: f32, ticks: usize) -> f32 {
    if !hi.is_finite() || hi <= 0. {
        return 1.0;
    }
    let top = hi * 1.05;
    let step = crate::charts::nice_step(top, ticks);
    if !step.is_finite() || step <= 0. {
        return top;
    }
    (top / step).ceil() * step
}

impl FoloApp {
    /// Opens `Route::RunCompare`, ensuring 2..=4 valid runs are selected when possible.
    pub fn open_run_compare(&mut self, preferred: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        let runs = self.store.runs.ready().cloned().unwrap_or_default();
        let mut ids = reconcile_compare_runs(&self.run_compare.ids, &runs);
        if let Some(pref) = preferred
            && runs.iter().any(|r| r.id == pref)
        {
            ids.retain(|id| id != pref);
            ids.insert(0, pref.to_string());
            ids.truncate(RUN_COMPARE_MAX);
        }
        if ids.len() < 2 {
            let defaults = default_compare_runs(&runs, preferred.or_else(|| ids.first().map(String::as_str)));
            for id in defaults {
                if !ids.contains(&id) && ids.len() < RUN_COMPARE_MAX {
                    ids.push(id);
                }
            }
        }
        self.run_compare.ids = ids;
        self.run_compare.seeded = true;
        self.run_compare.select_mode = false;
        self.run_compare.pinned_run = None;
        self.run_compare.hover = None;
        self.run_compare.hover_on = None;
        self.navigate(Route::RunCompare, window, cx);
    }

    /// Toggles a run in/out of the compare selection (capped at `RUN_COMPARE_MAX`).
    pub fn toggle_run_compare_pin(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(runs) = self.store.runs.ready() {
            self.run_compare.ids = reconcile_compare_runs(&self.run_compare.ids, runs);
        }
        if let Some(pos) = self.run_compare.ids.iter().position(|x| x == &id) {
            self.run_compare.ids.remove(pos);
            if self.run_compare.pinned_run.as_deref() == Some(id.as_str()) {
                self.run_compare.pinned_run = None;
            }
            cx.notify();
            return;
        }
        if self.run_compare.ids.len() >= RUN_COMPARE_MAX {
            self.toast(
                Tone::Info,
                format!("Up to {RUN_COMPARE_MAX} runs can be compared at once"),
                Some("Remove a selected run first to add another.".into()),
                cx,
            );
            return;
        }
        self.run_compare.ids.push(id);
        self.run_compare.seeded = true;
        cx.notify();
    }

    /// Adds a run to the active comparison.
    pub fn run_compare_add(&mut self, id: String, cx: &mut Context<Self>) {
        if let Some(runs) = self.store.runs.ready() {
            self.run_compare.ids = reconcile_compare_runs(&self.run_compare.ids, runs);
        }
        if !self.run_compare.ids.contains(&id) && self.run_compare.ids.len() < RUN_COMPARE_MAX {
            self.run_compare.ids.push(id);
            self.run_compare.seeded = true;
            cx.notify();
        }
    }

    /// Removes a run from the active comparison.
    pub fn run_compare_remove(&mut self, id: &str, cx: &mut Context<Self>) {
        self.run_compare.ids.retain(|x| x != id);
        self.run_compare.seeded = true;
        if self.run_compare.pinned_run.as_deref() == Some(id) {
            self.run_compare.pinned_run = None;
        }
        cx.notify();
    }

    /// Moves `id` to slot 0 (baseline) in the comparison.
    pub fn run_compare_make_baseline(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(pos) = self.run_compare.ids.iter().position(|x| x == id) {
            let item = self.run_compare.ids.remove(pos);
            self.run_compare.ids.insert(0, item);
            cx.notify();
        }
    }

    /// Main page view for `Route::RunCompare`.
    pub fn run_compare_view(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let back = self
            .button(
                "run-cmp-back",
                "All runs",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Runs, w, cx)),
            )
            .into_any_element();

        if let Some(e) = self.store.runs.error().cloned() {
            let header = page_header(&t, "Compare runs", "The run list didn't load", vec![back]);
            let body = self.error_panel(&e, "run-cmp-retry", window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let Some(all_runs) = self.store.runs.ready().cloned() else {
            let header = page_header(&t, "Compare runs", "Loading…", vec![back]);
            let body = panel(&t).child(loading_rows(&t, "run-cmp", 6)).into_any_element();
            return self.page_frame(&t, header, body, cx);
        };

        // Reconcile persisted IDs against live runs; seed defaults once on direct `--open runs-compare`
        // (never re-seed after the user intentionally removes all chips).
        let mut reconciled = reconcile_compare_runs(&self.run_compare.ids, &all_runs);
        if !self.run_compare.seeded && reconciled.is_empty() && all_runs.len() >= 2 {
            reconciled = default_compare_runs(&all_runs, None);
            self.run_compare.seeded = true;
        } else if !reconciled.is_empty() {
            self.run_compare.seeded = true;
        }
        if reconciled != self.run_compare.ids {
            self.run_compare.ids = reconciled.clone();
        }
        if self.run_compare.pinned_run.as_deref().is_some_and(|p| !reconciled.iter().any(|id| id == p)) {
            self.run_compare.pinned_run = None;
        }

        let chosen: Vec<Run> =
            reconciled.iter().filter_map(|id| all_runs.iter().find(|r| r.id == *id).cloned()).collect();
        let candidates: Vec<Run> =
            all_runs.iter().filter(|r| !reconciled.iter().any(|id| id == &r.id)).cloned().collect();
        let same_ds = same_dataset(&chosen);

        let subtitle = if chosen.len() < 2 {
            format!("{} selected · pick at least 2 runs to overlay curves", chosen.len())
        } else {
            format!(
                "{} · {}",
                counted(chosen.len() as u64, "run", "runs"),
                if same_ds { "same dataset" } else { "different datasets" }
            )
        };
        let header =
            detail_header(&t, text(ty::XL, t.text_primary).child("Compare runs"), subtitle, vec![back]);

        let chips_bar = self.run_compare_chips(&t, &chosen, &candidates, window, cx);
        if chosen.len() < 2 {
            let back_btn = self.button(
                "run-cmp-empty-back",
                "Back to Runs",
                Btn::Secondary,
                Some("icons/arrow-left.svg"),
                window,
                cx,
                act(|this, w, cx| this.navigate(Route::Runs, w, cx)),
            );
            let title = if chosen.is_empty() {
                "Pick two to four runs to compare"
            } else {
                "Add one more run to compare"
            };
            let desc = if candidates.is_empty() {
                "Train at least two runs to overlay their loss and mAP curves side by side."
            } else {
                "Choose another run from the picker above, or select runs on the Runs page."
            };
            let body = div().flex().flex_col().gap(px(16.)).child(chips_bar).child(panel(&t).child(
                empty_state(&t, "icons/git-compare-arrows.svg", title, desc, Tone::Neutral).child(back_btn),
            ));
            return self.page_frame(&t, header, body, cx);
        }

        // Kick off metric loads for every compared run.
        for r in &chosen {
            self.ensure_run_data(&r.id, cx);
        }

        let caveat = (!same_ds).then(|| {
            panel(&t).bg(t.warning_muted).border_color(t.warning_solid).child(
                div()
                    .px(px(16.))
                    .py(px(10.))
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .child(icon("icons/triangle-alert.svg", 15., t.warning_text))
                    .child(
                        text(ty::SM, t.warning_text)
                            .whitespace_normal()
                            .child("Different datasets — mAP and loss values are only directly comparable on the same dataset."),
                    ),
            )
        });

        let failed_banner = self.run_compare_metric_errors(&t, &chosen, window, cx);
        let charts = self.run_compare_charts(&t, &chosen, window, cx);
        let table = self.run_compare_table(&t, &chosen, same_ds, window, cx);

        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(chips_bar)
            .children(caveat)
            .children(failed_banner)
            .child(charts)
            .child(table);
        self.page_frame(&t, header, body, cx)
    }

    fn run_compare_chips(
        &mut self,
        t: &Theme,
        chosen: &[Run],
        candidates: &[Run],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let pinned = self.run_compare.pinned_run.clone();
        let mut bar = div().flex().flex_wrap().items_center().gap(px(8.));
        for (slot, r) in chosen.iter().enumerate() {
            let color = run_slot_color(&t, slot);
            let is_base = slot == 0;
            let is_focused = pinned.as_deref() == Some(r.id.as_str());
            let rid_open = r.id.clone();
            let open_btn = self.icon_button(
                &format!("run-cmp-open-{}", r.id),
                "icons/arrow-up-right.svg",
                "Open run",
                window,
                cx,
                act(move |this, w, cx| this.navigate(Route::Run(rid_open.clone()), w, cx)),
            );
            let rid_rm = r.id.clone();
            let rm_btn = self.icon_button(
                &format!("run-cmp-rm-{}", r.id),
                "icons/x.svg",
                "Remove from comparison",
                window,
                cx,
                act(move |this, _, cx| this.run_compare_remove(&rid_rm, cx)),
            );
            let rid_pin = r.id.clone();
            let pin_btn = self.icon_button(
                &format!("run-cmp-pin-{}", r.id),
                "icons/pin.svg",
                if is_focused { "Clear highlight" } else { "Highlight this run's curves" },
                window,
                cx,
                act(move |this, _, cx| {
                    this.run_compare.pinned_run =
                        if this.run_compare.pinned_run.as_deref() == Some(rid_pin.as_str()) {
                            None
                        } else {
                            Some(rid_pin.clone())
                        };
                    cx.notify();
                }),
            );
            let best_el = match &r.best {
                Some(b) => div()
                    .id(("cmp-chip-best", slot))
                    .tooltip(crate::controls::tip(
                        format!("Best mAP50-95: {} at epoch {}", f3(b.map50_95), b.epoch),
                        None,
                    ))
                    .child(text(ty::SM, t.text_secondary).tnum().child(format!("best {}", f3(b.map50_95)))),
                None => div().id(("cmp-chip-best", slot)).child(text(ty::SM, t.text_tertiary).child("–")),
            };
            let chip = div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(10.))
                .py(px(6.))
                .rounded(px(radius::MD))
                .bg(if is_focused { t.bg_active } else { t.bg_panel })
                .border_1()
                .border_color(if is_focused || is_base { t.accent_line } else { t.line_default })
                .child(slot_swatch(color, slot))
                .child(
                    text(ty::SM, t.text_primary)
                        .font_weight(FontWeight::MEDIUM)
                        .max_w(px(180.))
                        .ellipsis()
                        .child(r.name.clone()),
                )
                .child(best_el)
                .when(is_base, |d| d.child(badge(&t, "baseline", Tone::Accent)))
                .child(div().flex().items_center().gap(px(2.)).child(pin_btn).child(open_btn).child(rm_btn));
            bar = bar.child(chip);
        }
        if chosen.len() < RUN_COMPARE_MAX && !candidates.is_empty() {
            let cand_ids: Vec<String> = candidates.iter().map(|c| c.id.clone()).collect();
            let options: Vec<(SharedString, Option<SharedString>)> = candidates
                .iter()
                .map(|c| {
                    let detail = match &c.best {
                        Some(b) => format!("{} · mAP {}", c.config.arch.as_str(), f3(b.map50_95)),
                        None => format!("{} · {}", c.config.arch.as_str(), status_label(c).to_lowercase()),
                    };
                    (c.name.clone().into(), Some(detail.into()))
                })
                .collect();
            let adder = self.select(
                "run-cmp-add",
                "Add run…".into(),
                options,
                candidates.len(),
                window,
                cx,
                Rc::new(move |this, i, _, cx| {
                    if let Some(rid) = cand_ids.get(i).cloned() {
                        this.run_compare_add(rid, cx);
                    }
                }),
            );
            bar = bar.child(div().w(px(210.)).child(adder));
        }
        bar.into_any_element()
    }

    fn run_compare_metric_errors(
        &self,
        t: &Theme,
        chosen: &[Run],
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let failed: Vec<(&Run, crate::api::ApiError)> = chosen
            .iter()
            .filter_map(|r| self.store.run_data.get(&r.id)?.error().map(|e| (r, e.clone())))
            .collect();
        if failed.is_empty() {
            return None;
        }
        let mut box_el = div().px(px(16.)).py(px(10.)).flex().flex_col().gap(px(8.));
        for (r, err) in failed {
            let rid = r.id.clone();
            let retry = self.button(
                &format!("run-cmp-mretry-{}", r.id),
                "Retry",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| {
                    this.store.run_data.remove(&rid);
                    this.ensure_run_data(&rid, cx);
                    cx.notify();
                }),
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
                                    .child(format!("Could not load metrics for {}: {}", r.name, err.message)),
                            ),
                    )
                    .child(retry),
            );
        }
        Some(panel(t).bg(t.danger_muted).border_color(t.danger_solid).child(box_el).into_any_element())
    }

    fn run_compare_charts(
        &mut self,
        t: &Theme,
        chosen: &[Run],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = *t;
        let x_axis = self.run_compare.x_axis;
        let map_metric = self.run_compare.map_metric;
        let loss_part = self.run_compare.loss_part;
        let smooth_on = self.run_compare.loss_smooth;
        let log_on = self.run_compare.loss_log_y;
        let pinned = self.run_compare.pinned_run.clone();
        let ema = if smooth_on { COMPARE_LOSS_EMA } else { 0.0 };

        let map_pts: Vec<Vec<(f32, f32)>> = chosen
            .iter()
            .map(|r| {
                self.store
                    .run_data
                    .get(&r.id)
                    .and_then(|l| l.ready())
                    .map(|d| build_map_points(r, d, map_metric, x_axis))
                    .unwrap_or_default()
            })
            .collect();
        let loss_key = LossSeriesCacheKey {
            runs: chosen
                .iter()
                .map(|r| {
                    let (n, last_step, spe) = self
                        .store
                        .run_data
                        .get(&r.id)
                        .and_then(|l| l.ready())
                        .map(|d| (d.metrics.steps.len(), d.metrics.steps.last().map_or(0, |s| s.step), d.spe))
                        .unwrap_or((0, 0, r.steps_per_epoch));
                    (r.id.clone(), n, last_step, r.epochs, spe)
                })
                .collect(),
            loss_part,
            x_axis,
        };
        let loss_arcs: Vec<Arc<Vec<(f32, f32)>>> = match &self.run_compare.loss_cache {
            Some((k, cached)) if *k == loss_key => cached.clone(),
            _ => {
                let built: Vec<Arc<Vec<(f32, f32)>>> = chosen
                    .iter()
                    .map(|r| {
                        let pts = self
                            .store
                            .run_data
                            .get(&r.id)
                            .and_then(|l| l.ready())
                            .map(|d| build_loss_points(r, d, loss_part, x_axis))
                            .unwrap_or_default();
                        Arc::new(pts)
                    })
                    .collect();
                self.run_compare.loss_cache = Some((loss_key, built.clone()));
                built
            }
        };

        let x_max = shared_x_max(chosen, &map_pts, x_axis).max(shared_x_max(chosen, &loss_arcs, x_axis));
        let hover_on = self.run_compare.hover_on;
        let hover_val = self.run_compare.hover;
        let hover_of = |id: &str| hover_val.filter(|_| hover_on == Some(id));

        let view = cx.entity().downgrade();
        let make_hover = |id: &'static str| -> HoverFn {
            let view = view.clone();
            Rc::new(move |f, _, cx| {
                view.update(cx, |this, cx| {
                    let next = match f {
                        Some(_) => Some((f, Some(id))),
                        None if this.run_compare.hover_on.is_none_or(|o| o == id) => Some((None, None)),
                        None => None,
                    };
                    if let Some((hv, ho)) = next
                        && (this.run_compare.hover, this.run_compare.hover_on) != (hv, ho)
                    {
                        this.run_compare.hover = hv;
                        this.run_compare.hover_on = ho;
                        cx.notify();
                    }
                })
                .ok();
            })
        };

        let run_end = |r: &Run| match r.status {
            RunStatus::Running | RunStatus::Stopping => End::Live,
            RunStatus::Completed => End::Done,
            RunStatus::Stopped => End::Stopped,
            RunStatus::Failed => End::Failed,
            RunStatus::Queued => End::None,
        };
        let style_series = |mut s: Series, slot: usize, run: &Run| -> Series {
            let dimmed = pinned.as_deref().is_some_and(|p| p != run.id);
            if dimmed {
                s.color = alpha(s.color, 0.25);
            }
            s = s.width(if slot == 0 { 2.0 } else { 1.6 }).end(run_end(run));
            match slot {
                1 => s.dashed(),
                2 => s.dotted(),
                _ => s,
            }
        };

        // 1. Validation mAP overlaid chart.
        let map_series: Vec<Series> = chosen
            .iter()
            .enumerate()
            .map(|(slot, r)| {
                let color = run_slot_color(&t, slot);
                style_series(Series::new(r.name.clone(), color, Arc::new(map_pts[slot].clone())), slot, r)
            })
            .collect();
        let map_max = map_pts.iter().flat_map(|p| p.iter()).map(|p| p.1).fold(0.0f32, f32::max);
        let mut map_chart = LineChart::new("cmp-map", map_series);
        map_chart.x_range = Some((0.0, x_max));
        map_chart.unit_max = Some(map_max);
        map_chart.height = 230.;
        map_chart.x_int = x_axis == CompareXAxis::Epochs;
        map_chart.x_name = if x_axis == CompareXAxis::Progress { "progress" } else { "epoch" };
        map_chart.x_fmt = if x_axis == CompareXAxis::Progress {
            |x| format!("{}%", x.round() as u32)
        } else {
            |x| format!("{}", x.round() as u32)
        };
        map_chart.hover = hover_of("cmp-map");
        map_chart.on_hover = Some(make_hover("cmp-map"));
        map_chart.readout = false;
        map_chart.x_ticks = Some(6);

        // 2. Training loss overlaid chart (reuses memoized Arc<Vec> so chart EMA cache hits on hover).
        let loss_series: Vec<Series> = chosen
            .iter()
            .enumerate()
            .map(|(slot, r)| {
                let color = run_slot_color(&t, slot);
                style_series(Series::new(r.name.clone(), color, loss_arcs[slot].clone()).smooth(ema), slot, r)
            })
            .collect();
        let loss_max = loss_arcs.iter().flat_map(|p| p.iter()).map(|p| p.1).fold(0.0f32, f32::max);
        let loss_min = loss_arcs
            .iter()
            .flat_map(|p| p.iter())
            .map(|p| p.1)
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(f32::INFINITY, f32::min);
        let mut loss_chart = LineChart::new("cmp-loss", loss_series);
        loss_chart.x_range = Some((0.0, x_max));
        if log_on {
            loss_chart.log_y = true;
            loss_chart.y_range = Some(log_range(loss_min, loss_max));
        } else {
            loss_chart.log_y = false;
            loss_chart.y_range = Some((0.0, linear_top(loss_max, 4)));
        }
        loss_chart.height = 230.;
        loss_chart.x_int = x_axis == CompareXAxis::Epochs;
        loss_chart.x_name = map_chart.x_name;
        loss_chart.x_fmt = map_chart.x_fmt;
        loss_chart.hover = hover_of("cmp-loss");
        loss_chart.on_hover = Some(make_hover("cmp-loss"));
        loss_chart.readout = false;
        loss_chart.x_ticks = Some(6);

        // Per-run legends: when hovering, use `value_near` so runs without data at the cursor show
        // no stale endpoint value; match slot 2's dotted swatch as well as slot 1's dashed swatch.
        let legend_item = |name: &str, slot: usize, val: Option<String>| {
            let color = run_slot_color(&t, slot);
            div()
                .flex()
                .items_center()
                .gap(px(5.))
                .child(crate::ui::line_swatch(color, slot == 1, slot == 2, if slot == 0 { 2.5 } else { 2.0 }))
                .child(text(ty::SM, t.text_secondary).whitespace_nowrap().child(name.to_string()))
                .when_some(val, |d, v| d.child(crate::ui::sci(ty::SM, t.text_primary, v)))
        };
        let legend_val = |pts: &[(f32, f32)], hv: Option<f32>| -> Option<String> {
            match hv {
                Some(f) if f.is_finite() => crate::charts::value_near(pts, x_max * f, x_max).map(f3),
                Some(_) => None,
                None => crate::charts::value_at(pts, None, 0.0, x_max).map(f3),
            }
        };
        let legend_scope = |hv: Option<f32>| -> String {
            match hv {
                Some(f) if f.is_finite() => match x_axis {
                    CompareXAxis::Epochs => format!("@ ep {}", (x_max * f).round() as i64),
                    CompareXAxis::Progress => format!("@ {:.0}%", (x_max * f).clamp(0.0, 100.0)),
                },
                _ => "latest".into(),
            }
        };
        let map_hv = hover_of("cmp-map");
        let loss_hv = hover_of("cmp-loss");
        let map_legend = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(12.))
            .child(text(ty::XS, t.text_tertiary).tnum().whitespace_nowrap().child(legend_scope(map_hv)))
            .children(chosen.iter().enumerate().map(|(slot, r)| {
                let val = legend_val(&map_pts[slot], map_hv);
                legend_item(&r.name, slot, val)
            }));
        let loss_legend = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(12.))
            .child(text(ty::XS, t.text_tertiary).tnum().whitespace_nowrap().child(legend_scope(loss_hv)))
            .children(chosen.iter().enumerate().map(|(slot, r)| {
                let val = if smooth_on {
                    let sm = crate::charts::smoothed(&loss_arcs[slot], COMPARE_LOSS_EMA);
                    legend_val(&sm, loss_hv)
                } else {
                    legend_val(&loss_arcs[slot], loss_hv)
                };
                legend_item(&r.name, slot, val)
            }));

        let map_metric_seg = self.segmented(
            "run-cmp-map-metric",
            &["mAP50-95", "mAP50"],
            if map_metric == CompareMapMetric::Map50_95 { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_compare.map_metric =
                    if i == 0 { CompareMapMetric::Map50_95 } else { CompareMapMetric::Map50 };
                cx.notify();
            }),
        );
        let x_axis_seg = self.segmented(
            "run-cmp-x-axis",
            &["Epochs", "Progress %"],
            if x_axis == CompareXAxis::Epochs { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_compare.x_axis = if i == 0 { CompareXAxis::Epochs } else { CompareXAxis::Progress };
                this.run_compare.hover = None;
                cx.notify();
            }),
        );
        let loss_part_idx = match loss_part {
            3 => 0,
            0 => 1,
            1 => 2,
            _ => 3,
        };
        let loss_part_seg = self.segmented(
            "run-cmp-loss-part",
            &["Total", "Box", "Cls", "DFL"],
            loss_part_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_compare.loss_part = match i {
                    0 => 3,
                    1 => 0,
                    2 => 1,
                    _ => 2,
                };
                cx.notify();
            }),
        );
        let loss_scale_seg = self.segmented(
            "run-cmp-loss-scale",
            &["Log", "Lin"],
            if log_on { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_compare.loss_log_y = i == 0;
                cx.notify();
            }),
        );
        let loss_smooth_seg = self.segmented(
            "run-cmp-loss-smooth",
            &["Smooth", "Raw"],
            if smooth_on { 0 } else { 1 },
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.run_compare.loss_smooth = i == 0;
                cx.notify();
            }),
        );

        let any_loading = chosen.iter().any(|r| {
            self.store
                .run_data
                .get(&r.id)
                .is_none_or(|l| l.is_loading() || matches!(l, crate::store::Load::Idle))
        });

        // Wrapping chart header (`min_h` instead of fixed 40 px) so segmented controls never clip
        // or overlap the legend below at 960–1120 px.
        let chart_head = |title: &str, controls: folo_ui::Div| {
            div()
                .min_h(px(40.))
                .px(px(16.))
                .py(px(6.))
                .border_b_1()
                .border_color(t.line_subtle)
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap(px(8.))
                .child(text(ty::SM, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()))
                .child(controls)
        };

        let map_card = panel(&t).flex_1().min_w_0().child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .min_w_0()
                .child(chart_head(
                    "Validation mAP",
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(6.))
                        .child(map_metric_seg)
                        .child(x_axis_seg),
                ))
                .child(div().px(px(16.)).pt(px(8.)).child(map_legend))
                .child(div().px(px(12.)).pb(px(8.)).child(line_chart(&t, map_chart)))
                .when(any_loading, |d| {
                    d.child(
                        div()
                            .px(px(16.))
                            .pb(px(8.))
                            .child(text(ty::XS, t.text_tertiary).child("Loading run metrics…")),
                    )
                }),
        );

        let loss_card = panel(&t).flex_1().min_w_0().child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .min_w_0()
                .child(chart_head(
                    "Training loss",
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(px(6.))
                        .child(loss_part_seg)
                        .child(loss_smooth_seg)
                        .child(loss_scale_seg),
                ))
                .child(div().px(px(16.)).pt(px(8.)).child(loss_legend))
                .child(div().px(px(12.)).pb(px(8.)).child(line_chart(&t, loss_chart))),
        );

        // Stack charts whenever two side-by-side cards wouldn't comfortably fit their control rows.
        let stack = self.content_width(window) < 1040.;
        if stack {
            div().flex().flex_col().gap(px(16.)).child(map_card).child(loss_card).into_any_element()
        } else {
            div().flex().gap(px(16.)).child(map_card).child(loss_card).into_any_element()
        }
    }
}
