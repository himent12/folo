//! A run's training charts: losses, metrics, precision / recall and the learning rate.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, KeyDownEvent, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Run, RunExt, RunStatus, time};
use crate::app::{FoloApp, act};
use crate::charts::{End, HoverFn, LineChart, Marker, Series, line_chart};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// The run charts in reading order: ↑ / ↓ move the keys' cursor through them.
const CHARTS: [&str; 4] = ["run-map", "run-loss", "run-pr", "run-lr"];

/// The run chart a cursor stepped with the keys shows on until ↑ / ↓ pick another.
const FIRST_CHART: &str = CHARTS[0];

/// The loss lines' smoothing: Studio's default (`DEFAULT_SMOOTHING`, the same debiased EMA), so the
/// two front ends show the same smoothed loss.
const LOSS_EMA: f32 = 0.98;

impl FoloApp {
    pub(super) fn run_charts(
        &mut self,
        r: &Run,
        d: &crate::store::Derived,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let live = r.status == RunStatus::Running;
        // How each line ends: while it trains the chart's first series in the lime dot and the
        // others in a dot of their hue (`trail`), else the run's outcome (charts::marks).
        let end = match r.status {
            RunStatus::Running | RunStatus::Stopping => End::Live,
            RunStatus::Completed => End::Done,
            RunStatus::Stopped => End::Stopped,
            RunStatus::Failed => End::Failed,
            RunStatus::Queued => End::None,
        };
        let trail = end.trailing();
        // The cursor shows on one chart only, the one the pointer is on (a cursor stepped with the
        // keys, no pointer on a chart, on the first); the other charts and their legends stay at
        // the latest point.
        let (cursor, hover_on) = (self.chart_hover, self.chart_hover_on);
        let hover_of = move |id: &str| cursor_on(cursor, hover_on, id);
        let (split, map_title) = (self.eval_split(r), self.map_title(r));
        // The P/R chart's operating point follows the cursor: each epoch's own best-F1 confidence.
        let pr_conf = {
            // Every validated epoch, its confidence optional: the cursor picks the epoch the P/R
            // series picks (nearest x), then reads that epoch's own value, unknown included.
            let epochs: Vec<(f32, Option<f32>)> = self
                .store
                .run_data
                .get(&r.id)
                .and_then(|x| x.ready())
                .map(|x| {
                    x.metrics
                        .epochs
                        .iter()
                        .map(|e| {
                            (
                                e.epoch as f32,
                                operating_conf(e.val.best_f1_conf, e.val.precision, e.val.recall),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            let total = self.run_x_total(r);
            let xs: Vec<(f32, f32)> = epochs.iter().map(|e| (e.0, 0.)).collect();
            let at = match hover_of("run-pr") {
                Some(f) => crate::charts::nearest(&xs, total * f),
                None => epochs.len().checked_sub(1),
            };
            at.and_then(|i| epochs[i].1)
        };
        let view = cx.entity().downgrade();
        // Each chart reports for itself: the pointer on it takes the cursor there; leaving it (or a
        // move outside its plot) clears the cursor only if it was this chart's, so a late leave of
        // one chart never clears the next one's.
        let on_hover = |id: &'static str| -> HoverFn {
            let view = view.clone();
            Rc::new(move |f, _, cx| {
                view.update(cx, |this, cx| {
                    let Some(next) = hovered(this.chart_hover_on, id, f) else { return };
                    if (this.chart_hover, this.chart_hover_on) != next {
                        (this.chart_hover, this.chart_hover_on) = next;
                        cx.notify();
                    }
                })
                .ok();
            })
        };
        // The planned run while it trains (and once it completed); fitted to the epochs trained
        // for a run cut short, or when "So far" is picked.
        let total = self.run_x_total(r);
        let planned = total >= r.epochs as f32;
        // Where mosaic closes, as the engine runs it (the capped value: 7 of a 15-epoch run's
        // asked-for 10, docs/DOGFOOD-3 S2).
        let close_n = r.close_mosaic_epochs();
        let closing = r.epochs.saturating_sub(close_n) as f32;
        // One annotation band: the schedule (close mosaic) and what went wrong (failure). The best
        // epoch is a dot on its series, with the value in the legend.
        let close_marker =
            Marker { x: closing, label: "close mosaic".into(), color: t.text_tertiary, lane: Some(0) };
        // The steps per epoch the step charts' x is in (store::live): the cursor reads a step's
        // epoch back from its step.
        let spe = self.store.run_data.get(&r.id).and_then(|x| x.ready()).map_or(r.steps_per_epoch, |d| d.spe);
        let cursor_epoch: Rc<dyn Fn(f32) -> String> = Rc::new(move |x| epoch_at(x, spe));
        let base = |id: &'static str, series: Vec<Series>, height: f32| {
            let mut c = LineChart::new(id.to_string(), series);
            c.x_range = Some((0., total));
            c.hover = hover_of(id);
            c.on_hover = Some(on_hover(id));
            c.height = height;
            c.min_lanes = 1;
            c.x_int = true;
            c.x_name = "epoch";
            c.x_fmt = epoch_tick;
            c.cursor_fmt = Some(cursor_epoch.clone());
            // Named on the first chart only; the others keep the line.
            // No switch to mark when mosaic never closes, or is off throughout.
            c.markers = if close_n == 0 || close_n >= r.epochs {
                Vec::new()
            } else if id == "run-map" || id == "run-loss" {
                vec![close_marker.clone()]
            } else {
                vec![Marker { label: SharedString::default(), ..close_marker.clone() }]
            };
            // Values live in each panel's legend; the plot keeps only the cursor and dots.
            c.readout = false;
            // Same x ticks in all four charts.
            c.x_ticks = Some(6);
            c.ease_key = crate::charts::ease_key(&r.id);
            if live && planned {
                c.future = Some((r.epoch as f32, SharedString::default()));
            }
            c
        };
        // Short windows keep both primary charts on screen.
        let short = window.viewport_size().height < px(700.);
        let (primary_h, secondary_h) = if short { (140., 110.) } else { (220., 140.) };
        let names = ["box", "cls", "dfl", "total"];
        let colors = t.loss_colors();
        // Total leads the loss chart (its lime end while live); without it, the first part shown.
        let lead = if self.loss_parts[3] { 3 } else { (0..4).find(|&i| self.loss_parts[i]).unwrap_or(3) };
        // Each loss line's name at its end, in its hue's text ink (total in ink).
        let label_inks = [t.chart_text[1], t.chart_text[4], t.chart_text[3], t.text_primary];
        let loss_series: Vec<Series> = (0..4)
            .filter(|&i| self.loss_parts[i])
            .map(|i| {
                let s = Series::new(names[i], colors[i], d.loss[i].clone())
                    .smooth(LOSS_EMA)
                    .end(if i == lead { end } else { trail })
                    .end_label(label_inks[i]);
                match i {
                    1 => s.dashed(),
                    2 => s.dotted(),
                    3 => s.width(2.25),
                    _ => s,
                }
            })
            .collect();
        let loss_max = (0..4).filter(|&i| self.loss_parts[i]).map(|i| d.loss_max[i]).fold(0f32, f32::max);
        let mut loss = base("run-loss", loss_series, primary_h);
        let loss_min =
            (0..4).filter(|&i| self.loss_parts[i]).map(|i| d.loss_min[i]).fold(f32::INFINITY, f32::min);
        loss.log_y = true;
        loss.y_range = Some(log_range(loss_min, loss_max));
        loss.end_labels = true;
        let mut map = base(
            "run-map",
            vec![
                Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().end(end).width(1.75),
                // Dashed: the pair reads apart without colour (as in the legend).
                Series::new("mAP50", t.chart_series[1], d.map50.clone()).width(1.25).dashed().end(trail),
            ],
            primary_h,
        );
        // From 0 to a little over the highest value, 3 to 5 round gridlines (charts::domain).
        let map_max = series_max(&[&d.map50, &d.map50_95]);
        map.unit_max = Some(map_max);
        if live && planned {
            map.future = Some((
                r.epoch as f32,
                r.eta_s.map_or("remaining".into(), |s| format!("ETA {}", time::duration(s)).into()),
            ));
        }
        // The best epoch: a hollow ink ring on its series ("○ best" in the legend).
        map.best = r.best.as_ref().map(|b| (b.epoch as f32, b.map50_95));
        // A failed run marks where it stopped, on the charts that tell the story.
        let failed = r.status == RunStatus::Failed;
        if let Some((x, label)) = fail_mark(failed, r.failed_epoch(), r.epoch, d.loss[3].last().map(|p| p.0))
        {
            let fail = Marker { x, label: label.into(), color: t.danger_text, lane: Some(0) };
            loss.markers.insert(0, fail.clone());
            map.markers.insert(0, fail);
        }
        let mut pr = base(
            "run-pr",
            vec![
                // One hue for the pair, told apart by dash: no other chart here uses slot 5.
                Series::new("precision", t.chart_series[2], d.precision.clone()).end(end),
                Series::new("recall", t.chart_series[3], d.recall.clone()).dashed().end(trail),
            ],
            secondary_h,
        );
        // A 0–1 pair too: from 0, so a converged pair still reads as high, not as a fall.
        let pr_max = series_max(&[&d.precision, &d.recall]);
        pr.unit_max = Some(pr_max);
        let mut lr = base(
            "run-lr",
            vec![Series::new("lr", t.text_tertiary, d.lr.clone()).area().end(end)],
            secondary_h,
        );
        // The top on a nice tick above the peak, so the top gridline is labelled and the warm-up
        // peak sits under it (a padded max left an unlabelled line and a peak above the last tick).
        let top = (d.lr_max * 1.05).max(1e-4);
        let step = crate::charts::nice_step(top, 3);
        let lr_top = (top / step).ceil() * step;
        lr.y_range = Some((0., lr_top));
        // Its "× 10⁻ⁿ" goes in the panel caption, once, beside the legend that uses it.
        lr.pow_caption = false;
        lr.y_ticks = 3;

        // Legend values: at the cursor on their own chart, else the latest point.
        let v = |id: &str, pts: &[(f32, f32)], fmt: fn(f32) -> String| {
            crate::charts::value_at(pts, hover_of(id), 0., total).map(fmt)
        };
        let f3v: fn(f32) -> String = |x| format!("{x:.3}");
        // Loss parts are toggles; their values are the smoothed curve's, like the lines.
        let mut legend = Vec::new();
        let shown = self.loss_parts.iter().filter(|b| **b).count();
        for (i, name) in names.iter().enumerate() {
            let on = self.loss_parts[i];
            let key = format!("loss-part-{name}");
            let color = colors[i];
            let value =
                on.then(|| v("run-loss", &crate::charts::smoothed(&d.loss[i], LOSS_EMA), f3v)).flatten();
            let el = div()
                .id(SharedString::from(key.clone()))
                .h(px(20.))
                .px(px(6.))
                .flex()
                .items_center()
                .gap(px(5.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .active(|s| s.bg(t.bg_active))
                .tooltip(crate::controls::tip(
                    if on && shown == 1 {
                        "Keep at least one series visible"
                    } else if on {
                        "Hide this series"
                    } else {
                        "Show this series"
                    },
                    None,
                ))
                .child(if on {
                    crate::ui::line_swatch(color, i == 1, i == 2, if i == 3 { 3. } else { 2. })
                } else {
                    div()
                        .w(px(10.))
                        .h(px(if on { 2. } else { 6. }))
                        .rounded(px(2.))
                        .map(|d| if on { d.bg(color) } else { d.border_1().border_color(color) })
                })
                .child(
                    text(ty::SM, if on { t.text_secondary } else { t.text_tertiary }).child(name.to_string()),
                )
                .when_some(value, |d, val| d.child(text(ty::SM, t.text_primary).tnum().child(val)));
            legend.push(
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| {
                        let n = this.loss_parts.iter().filter(|b| **b).count();
                        if !(this.loss_parts[i] && n == 1) {
                            this.loss_parts[i] = !this.loss_parts[i];
                        }
                        cx.notify();
                    }),
                )
                .into_any_element(),
            );
        }
        // Every card has the same anatomy: header, a 28 px legend row, then the plot. Plot tops line
        // up across a row however long a legend is.
        let card = |title: &str, sub: &str, extra: AnyElement, chart: AnyElement| {
            panel(&t).flex_1().min_w_0().child(crate::ui::chart_section(&t, title, sub, extra, chart, short))
        };
        // The smoothing explanation, on hover and on keyboard focus.
        let loss_help = self.help_icon(
            "loss-ema-help",
            "Lines and their values are an exponential moving average (0.98, about the last 50 steps) of per-step loss, as in Studio; the faint trace is raw",
            window,
            cx,
        );
        let loss_legend = div()
            .id("loss-ema")
            .ml(px(-6.))
            .flex()
            .items_center()
            .gap(px(2.))
            .children(legend)
            .child(loss_help)
            .into_any_element();
        let mut map_items = vec![
            crate::ui::legend_value(&t, "mAP50-95", t.chart_series[0], false, v("run-map", &d.map50_95, f3v))
                .into_any_element(),
            crate::ui::legend_value(&t, "mAP50", t.chart_series[1], true, v("run-map", &d.map50, f3v))
                .into_any_element(),
        ];
        if let Some(b) = &r.best {
            map_items.push(
                crate::ui::best_legend(&t, format!("{} @ {}", f3(b.map50_95), b.epoch)).into_any_element(),
            );
        }
        let map_legend = div().flex().items_center().gap(px(12.)).children(map_items).into_any_element();
        let pr_legend = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .child(crate::ui::legend_value(
                &t,
                "precision",
                t.chart_series[2],
                false,
                v("run-pr", &d.precision, f3v),
            ))
            .child(crate::ui::legend_value(
                &t,
                "recall",
                t.chart_series[3],
                true,
                v("run-pr", &d.recall, f3v),
            ))
            .into_any_element();
        // The axis's power of ten ("× 10⁻³", in the caption, once); the legend value in normalised
        // notation as Studio writes it ("1.43×10⁻⁵", not "0.014×10⁻³"), so it reads on its own.
        let lr_pow = crate::charts::axis_pow(0., lr_top, 3);
        // In f64: a tiny valid rate would overflow 10^-p in f32.
        let lr_value =
            crate::charts::value_at(&d.lr, hover_of("run-lr"), 0., total).map(|x| lr_sci(x as f64));
        let lr_legend =
            crate::ui::legend_value(&t, "lr", t.text_tertiary, false, lr_value).into_any_element();
        // What the run actually trains with (Run.resolved, once the trainer is built): the
        // optimizer Auto picked and the decay, so the curve's shape has its reason beside it.
        let resolved = r.resolved.as_ref().map(|s| {
            let opt = match s.optimizer {
                crate::api::ResolvedOptimizer::Sgd => "SGD",
                crate::api::ResolvedOptimizer::Adamw => "AdamW",
                crate::api::ResolvedOptimizer::Musgd => "MuSGD",
                crate::api::ResolvedOptimizer::Unknown => "an unknown optimizer",
            };
            let auto = if r.config.optimizer == crate::api::Optimizer::Auto { " (auto)" } else { "" };
            let decay = if s.cos_lr { "cosine" } else { "linear" };
            // The end rate as a share of lr0, without rounding a 0.5% end to "0%" or "1%".
            let pct = format!("{:.2}", s.lrf * 100.);
            let pct = pct.trim_end_matches('0').trim_end_matches('.');
            format!("{opt}{auto} · {decay} to {pct}%")
        });
        let lr_caption = match (resolved, lr_pow) {
            (Some(s), Some(p)) => format!("{s} · × 10{}", crate::charts::superscript(p)),
            (Some(s), None) => s,
            (None, Some(p)) => format!("per step · by epoch · × 10{}", crate::charts::superscript(p)),
            (None, None) => "per step · by epoch".to_string(),
        };
        // The total as Studio's Loss tab says it: the step's raw reading and the smoothed line's
        // value, at the cursor or the latest step, so the two numbers never seem to disagree.
        let loss_caption = match (
            v("run-loss", &d.loss[3], f3v),
            v("run-loss", &crate::charts::smoothed(&d.loss[3], LOSS_EMA), f3v),
        ) {
            (Some(raw), Some(smooth)) => format!("raw {raw} · smoothed {smooth} · log scale"),
            _ => "log scale · EMA 0.98".to_string(),
        };
        // The mAP chart is the focal point: wider and first. Narrow windows stack each row.
        let stack = window.viewport_size().width < px(1120.);
        // Narrow and short: the primary pair stays side by side (half each), so Training loss is
        // on screen with Validation mAP instead of below the fold.
        let pair_a = stack && short;
        let row = |a: AnyElement, b: AnyElement, ga: f32, gb: f32| {
            // A negative weight marks the primary row.
            if stack && !(pair_a && ga < 0.) {
                div().flex().flex_col().gap(px(16.)).child(a).child(b)
            } else {
                div()
                    .flex()
                    .gap(px(16.))
                    .child(
                        div().flex().min_w_0().flex_grow().flex_basis(folo_ui::relative(ga.abs())).child(a),
                    )
                    .child(div().flex().min_w_0().flex_grow().flex_basis(folo_ui::relative(gb)).child(b))
            }
        };
        // "So far / All N" in the first chart's header fits every chart's x axis to the epochs
        // trained, or shows the planned run; only while those differ.
        let x_switch = (r.epoch > 0 && self.run_x_so_far(r) < r.epochs as f32).then(|| {
            let all = format!("All {}", r.epochs);
            self.segmented(
                "run-x-fit",
                &["So far", all.as_str()],
                if planned { 1 } else { 0 },
                window,
                cx,
                Rc::new(|this, i, _, _| this.run_x_fit = Some(i == 0)),
            )
            .tooltip(crate::controls::tip(
                "The x axis of every chart: the epochs trained, or the planned run",
                None,
            ))
            .into_any_element()
        });
        // A 0–1 axis cut under 0.25 says so beside its qualifier.
        let zoom = |q: &str, max: f32| {
            if crate::charts::domain::unit_zoomed(max) {
                format!("{q} · {}", crate::charts::domain::UNIT_ZOOM_NOTE)
            } else {
                q.to_string()
            }
        };
        let map_card = panel(&t).flex_1().min_w_0().child(
            div()
                .w_full()
                .flex()
                .flex_col()
                .min_w_0()
                .child(
                    panel_header(&t, map_title)
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(12.))
                                .child(
                                    text(ty::SM, t.text_tertiary)
                                        .font_weight(folo_ui::FontWeight::MEDIUM)
                                        .child(zoom("per epoch", map_max)),
                                )
                                .children(x_switch),
                        )
                        .when(short, |d| d.h(px(32.))),
                )
                .child(
                    div()
                        .h(px(if short { 24. } else { 28. }))
                        .px(px(16.))
                        .pt(px(if short { 4. } else { 6. }))
                        .flex()
                        .items_center()
                        .child(map_legend),
                )
                .child(div().px(px(12.)).pb(px(6.)).child(line_chart(&t, map))),
        );
        let grid = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(row(
                map_card.into_any_element(),
                card("Training loss", &loss_caption, loss_legend, line_chart(&t, loss)).into_any_element(),
                if pair_a { -0.5 } else { -0.6 },
                if pair_a { 0.5 } else { 0.4 },
            ))
            .child(row(
                card(
                    "Precision / Recall",
                    &zoom(&pr_caption(pr_conf, split, true), pr_max),
                    pr_legend,
                    line_chart(&t, pr),
                )
                .into_any_element(),
                card("Learning rate", &lr_caption, lr_legend, line_chart(&t, lr)).into_any_element(),
                0.6,
                0.4,
            ));
        // One Tab stop for the charts: ← / → walk the epochs on one chart, ↑ / ↓ take the cursor
        // to the chart before or after it (the first press shows it at the latest point).
        // Recorded data ends at the latest step or epoch; the rest of the x range is the schedule.
        let recorded = d.loss[3].last().map_or(0., |p| p.0).max(d.map50_95.last().map_or(0., |p| p.0));
        let end = (recorded / total.max(1.0)).clamp(0., 1.);
        let grid = grid.on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
            let by = match ev.keystroke.key.as_str() {
                "up" => -1,
                "down" => 1,
                _ => return,
            };
            this.chart_hover_on = Some(step_chart(this.chart_hover_on, by));
            this.chart_hover = Some(this.chart_hover.map_or(end, |c| c.min(end)));
            cx.stop_propagation();
            cx.notify();
        }));
        let charts = self.chart_group(
            "run-charts",
            1.0 / total.max(1.0),
            end,
            crate::controls::Cursor::Run,
            grid,
            window,
            cx,
        );
        if self.ring(&self.fh("run-charts", cx), window) {
            self.hint_keys(
                "← → inspect · ↑ ↓ another chart · Shift ×10 · Home / End · PgUp / PgDn scroll the page · Esc clears",
            );
        }
        charts
    }
}

/// The cursor chart `id` draws: the cursor (`cursor`, a fraction of the x range) when `on` is this
/// chart (the pointer's, or the one ↑ / ↓ picked), or when `on` is None (a cursor stepped with
/// ← / → only) and this is the first chart; else none.
fn cursor_on(cursor: Option<f32>, on: Option<&'static str>, id: &str) -> Option<f32> {
    cursor.filter(|_| on.unwrap_or(FIRST_CHART) == id)
}

/// The chart `by` places from `on` (None: the first) in [`CHARTS`], held at either end.
fn step_chart(on: Option<&'static str>, by: isize) -> &'static str {
    let at = on.and_then(|on| CHARTS.iter().position(|c| *c == on)).unwrap_or(0) as isize;
    CHARTS[(at + by).clamp(0, CHARTS.len() as isize - 1) as usize]
}

/// The cursor and the chart it is on after chart `id` reports the pointer at `f` (None: off its
/// plot, or gone), given the chart the cursor is on now; None to leave both as they are (another
/// chart's cursor, which a late leave of this one must not clear).
fn hovered(
    on: Option<&'static str>,
    id: &'static str,
    f: Option<f32>,
) -> Option<(Option<f32>, Option<&'static str>)> {
    match f {
        Some(_) => Some((f, Some(id))),
        None if on.is_none_or(|on| on == id) => Some((None, None)),
        None => None,
    }
}

/// A log axis over positive values `lo..=hi`, from the power of two at or below `lo` to the one at
/// or above `hi` (5 % headroom), at least one doubling tall. No positive data: 0.5 to 4.
fn log_range(lo: f32, hi: f32) -> (f32, f32) {
    if !lo.is_finite() || hi.is_nan() || hi <= 0. {
        return (0.5, 4.);
    }
    // The floor the chart's log transform uses too, so nothing lands below the plot.
    let lo = lo.max(1e-6);
    let y0 = 2f32.powi(lo.log2().floor() as i32);
    let y1 = 2f32.powi((hi.max(lo) * 1.05).log2().ceil() as i32).max(y0 * 2.);
    (y0, y1)
}

/// The highest finite value of some series (0 without data): what a fitted 0–1 axis is fitted to.
fn series_max(series: &[&[(f32, f32)]]) -> f32 {
    series.iter().flat_map(|s| s.iter()).map(|p| p.1).filter(|v| v.is_finite()).fold(0., f32::max)
}

/// Where a failed run's charts mark its failure (x, label): at its last step, else its kept epoch.
/// None for a run that failed before its first step: there is no "epoch 0" to mark (Studio's
/// failEvent).
fn fail_mark(failed: bool, failed_epoch: u32, kept: u32, last_step_x: Option<f32>) -> Option<(f32, String)> {
    (failed && failed_epoch > 0)
        .then(|| (last_step_x.unwrap_or(kept as f32), format!("failed during epoch {failed_epoch}")))
}

/// An x tick on the run charts: whole epochs (one computed as 0.1 * 30 is 3.0000002). Saturating:
/// 0 and -0.0 read 0.
fn epoch_tick(x: f32) -> String {
    format!("{}", x.round() as u32)
}

/// The epoch of the point the cursor is on: a step's x is `step / spe` (store::live), so the step
/// is read back and epoch e's steps `(e - 1)·spe + 1 ..= e·spe` name e; a validated epoch sits at
/// x = e. Not from x itself: rounding called the first half of every epoch the one before it, and
/// at 20 000 steps an epoch step 1 is x = 0.00005, too close to 0 for any tolerance.
fn epoch_at(x: f32, spe: u64) -> String {
    let spe = spe.max(1);
    let step = (f64::from(x) * spe as f64).round().max(0.) as u64;
    format!("{}", step.div_ceil(spe))
}

impl FoloApp {
    /// The run charts' x extent, in epochs: the planned run, or (So far, and by default for a run that
    /// ended short of its plan) the epochs trained.
    pub(super) fn run_x_total(&self, r: &Run) -> f32 {
        let fit = self.run_x_fit.unwrap_or(!(r.status.is_active() || r.status == RunStatus::Completed));
        if fit { self.run_x_so_far(r) } else { r.epochs.max(1) as f32 }
    }

    /// The "So far" extent: the epochs trained, steps included ([`fitted_total`]).
    fn run_x_so_far(&self, r: &Run) -> f32 {
        let last_step =
            self.store.run_data.get(&r.id).and_then(|x| x.ready()).and_then(|d| d.metrics.steps.last());
        fitted_total(r.epoch, last_step.map(|s| s.epoch))
    }
}

/// "So far": the epochs trained, the one a run stopped (or failed) in included. A run stopped
/// during epoch 3 keeps epoch 2 (`Run.epoch`), but it took epoch 3's steps: fitted to 2, the loss and
/// LR charts cut them off while their legends read the last of them.
fn fitted_total(kept: u32, last_step_epoch: Option<u32>) -> f32 {
    kept.max(last_step_epoch.unwrap_or(0)).max(1) as f32
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_cursor_shows_on_the_chart_the_pointer_is_on_only() {
        use super::{cursor_on, hovered};
        // The pointer on Training loss: only that chart draws the cursor.
        let (cursor, on) = hovered(None, "run-loss", Some(0.4)).unwrap();
        assert_eq!(cursor_on(cursor, on, "run-loss"), Some(0.4));
        for other in ["run-map", "run-pr", "run-lr"] {
            assert_eq!(cursor_on(cursor, on, other), None, "{other}");
        }
        // Moving to Validation mAP takes the cursor there; Training loss's late leave keeps it.
        let (cursor, on) = hovered(on, "run-map", Some(0.7)).unwrap();
        assert_eq!(hovered(on, "run-loss", None), None);
        assert_eq!(cursor_on(cursor, on, "run-map"), Some(0.7));
        assert_eq!(cursor_on(cursor, on, "run-loss"), None);
        // Leaving the chart it is on clears it.
        assert_eq!(hovered(on, "run-map", None), Some((None, None)));
        // A cursor stepped with the keys (no pointer on a chart) shows on the first chart.
        assert_eq!(cursor_on(Some(0.5), None, "run-map"), Some(0.5));
        assert_eq!(cursor_on(Some(0.5), None, "run-lr"), None);
    }

    #[test]
    fn up_and_down_take_the_keys_cursor_through_the_charts_in_reading_order() {
        use super::step_chart;
        assert_eq!(step_chart(None, 1), "run-loss");
        assert_eq!(step_chart(Some("run-loss"), 1), "run-pr");
        assert_eq!(step_chart(Some("run-pr"), 1), "run-lr");
        assert_eq!(step_chart(Some("run-lr"), 1), "run-lr", "held at the last");
        assert_eq!(step_chart(Some("run-lr"), -1), "run-pr");
        assert_eq!(step_chart(None, -1), "run-map", "held at the first");
    }

    #[test]
    fn a_fitted_axis_takes_the_highest_finite_value() {
        let a = [(1., 0.2), (2., f32::NAN), (3., 0.6)];
        let b = [(1., 0.4)];
        assert_eq!(super::series_max(&[&a, &b]), 0.6);
        assert_eq!(super::series_max(&[]), 0.);
    }

    #[test]
    fn so_far_holds_the_steps_of_the_epoch_a_run_stopped_in() {
        use super::fitted_total;
        // Stopped during epoch 3 of 3 (all 36 steps taken, validation cut): epoch 2 kept, 3 drawn.
        assert_eq!(fitted_total(2, Some(3)), 3.);
        // Stopped right after an epoch's validation: nothing past it.
        assert_eq!(fitted_total(2, Some(2)), 2.);
        // No history loaded yet, or none at all.
        assert_eq!(fitted_total(2, None), 2.);
        assert_eq!(fitted_total(0, None), 1.);
    }

    #[test]
    fn a_run_that_failed_before_its_first_step_has_no_failure_marker() {
        use super::fail_mark;
        // Failed before its first step (epoch 0, no step): nothing to mark (d2-runs-8).
        assert_eq!(fail_mark(true, 0, 0, None), None);
        // Failed during epoch 2 at step 21 of 12 per epoch: marked there.
        assert_eq!(fail_mark(true, 2, 1, Some(21. / 12.)), Some((21. / 12., "failed during epoch 2".into())));
        // Its step history not in yet: at the epoch it kept.
        assert_eq!(fail_mark(true, 2, 1, None), Some((1., "failed during epoch 2".into())));
        // Not failed: no marker.
        assert_eq!(fail_mark(false, 2, 1, Some(1.5)), None);
    }

    #[test]
    fn the_cursor_names_the_epoch_a_step_belongs_to() {
        use super::{epoch_at, epoch_tick};
        // A step's x as store::live computes it.
        let x = |step: u64, spe: u64| step as f32 / spe as f32;
        // 12 steps per epoch: step 4 is in epoch 1, step 17 in epoch 2, step 24 ends epoch 2.
        assert_eq!(epoch_at(x(4, 12), 12), "1");
        assert_eq!(epoch_at(x(17, 12), 12), "2");
        assert_eq!(epoch_at(x(24, 12), 12), "2");
        assert_eq!(epoch_at(x(25, 12), 12), "3");
        // 20 000 steps per epoch (d2-runs-5 at scale): step 1 is x = 0.00005, still epoch 1.
        assert_eq!(epoch_at(x(1, 20_000), 20_000), "1");
        assert_eq!(epoch_at(x(20_000, 20_000), 20_000), "1");
        assert_eq!(epoch_at(x(20_001, 20_000), 20_000), "2");
        // Every step just before, at and after an epoch's end, over a 300-epoch run.
        for spe in [1u64, 7, 12, 1_000, 20_000] {
            for e in 1..=300u64 {
                let end = e * spe;
                assert_eq!(epoch_at(x(end, spe), spe), e.to_string(), "step {end} of {spe}");
                assert_eq!(epoch_at(x(end + 1, spe), spe), (e + 1).to_string(), "step {} of {spe}", end + 1);
                if spe > 1 {
                    assert_eq!(epoch_at(x(end - 1, spe), spe), e.to_string(), "step {} of {spe}", end - 1);
                }
            }
        }
        // A per-epoch point (x = e) names its epoch; nothing trained yet reads 0.
        assert_eq!(epoch_at(3., 12), "3");
        assert_eq!(epoch_at(0., 12), "0");
        assert_eq!(epoch_at(-0., 12), "0");
        // Ticks stay whole.
        assert_eq!(epoch_tick(0.), "0");
        assert_eq!(epoch_tick(-0.), "0");
        assert_eq!(epoch_tick(3.), "3");
        assert_eq!(epoch_tick(0.1 * 30.), "3");
    }

    use super::log_range;

    #[test]
    fn the_log_loss_axis_holds_every_positive_value() {
        // A single tiny loss sits inside the range, not under it.
        let (a, b) = log_range(1e-4, 1e-4);
        assert!(a <= 1e-4 && 1e-4 < b && b >= a * 2.);
        let (a, b) = log_range(0.3, 6.);
        assert_eq!((a, b), (0.25, 8.));
        assert_eq!(log_range(f32::INFINITY, 0.), (0.5, 4.));
    }
}
