//! A run's training charts: losses, metrics, precision / recall and the learning rate.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::{Run, RunExt, RunStatus, time};
use crate::app::{FoloApp, act};
use crate::charts::{HoverFn, LineChart, Marker, Series, line_chart};
use crate::theme::{radius, theme, ty};
use crate::ui::*;

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
        let hover = self.chart_hover;
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
                .map(|x| x.metrics.epochs.iter().map(|e| (e.epoch as f32, e.val.best_f1_conf)).collect())
                .unwrap_or_default();
            let total = if r.status.is_active() || r.status == RunStatus::Completed {
                r.epochs as f32
            } else {
                r.epoch.max(1) as f32
            };
            let xs: Vec<(f32, f32)> = epochs.iter().map(|e| (e.0, 0.)).collect();
            let at = match hover {
                Some(f) => crate::charts::nearest(&xs, total * f),
                None => epochs.len().checked_sub(1),
            };
            at.and_then(|i| epochs[i].1)
        };
        let view = cx.entity().downgrade();
        let on_hover: HoverFn = Rc::new(move |f, _, cx| {
            view.update(cx, |this, cx| {
                if this.chart_hover != f {
                    this.chart_hover = f;
                    cx.notify();
                }
            })
            .ok();
        });
        // Active and completed runs show the whole schedule; cut-short runs zoom to the epochs they ran.
        let total = if r.status.is_active() || r.status == RunStatus::Completed {
            r.epochs as f32
        } else {
            r.epoch.max(1) as f32
        };
        // Where mosaic closes, as the engine runs it (the capped value: 7 of a 15-epoch run's
        // asked-for 10, docs/DOGFOOD-3 S2).
        let close_n = r.close_mosaic_epochs();
        let closing = r.epochs.saturating_sub(close_n) as f32;
        // One annotation band: the schedule (close mosaic) and what went wrong (failure). The best
        // epoch is a dot on its series, with the value in the legend.
        let close_marker =
            Marker { x: closing, label: "close mosaic".into(), color: t.text_tertiary, lane: Some(0) };
        let base = |id: &str, series: Vec<Series>, height: f32| {
            let mut c = LineChart::new(id.to_string(), series);
            c.x_range = Some((0., total));
            c.hover = hover;
            c.on_hover = Some(on_hover.clone());
            c.height = height;
            c.min_lanes = 1;
            c.x_int = true;
            c.x_name = "epoch";
            c.x_fmt = |v| format!("{v:.0}");
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
            if live {
                c.future = Some((r.epoch as f32, SharedString::default()));
            }
            c
        };
        // Short windows keep both primary charts on screen.
        let short = window.viewport_size().height < px(700.);
        let (primary_h, secondary_h) = if short { (140., 110.) } else { (220., 140.) };
        let names = ["box", "cls", "dfl", "total"];
        let colors = t.loss_colors();
        let loss_series: Vec<Series> = (0..4)
            .filter(|&i| self.loss_parts[i])
            .map(|i| {
                let s = Series::new(names[i], colors[i], d.loss[i].clone()).smooth(0.94).live(live);
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
        let mut map = base(
            "run-map",
            vec![
                Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().live(live).width(1.75),
                // Dashed: the pair reads apart without colour (as in the legend).
                Series::new("mAP50", t.chart_series[1], d.map50.clone()).width(1.25).dashed().live(live),
            ],
            primary_h,
        );
        map.y_range = Some((0., 1.));
        if live {
            map.future = Some((
                r.epoch as f32,
                r.eta_s.map_or("remaining".into(), |s| format!("ETA {}", time::duration(s)).into()),
            ));
        }
        map.y_fmt = Some(|v| format!("{v:.2}"));
        if let Some(b) = &r.best {
            map.dots.push((b.epoch as f32, b.map50_95, t.accent_fill()));
        }
        // A failed run marks where it stopped, on the charts that tell the story.
        if r.status == RunStatus::Failed {
            let stopped = d.loss[3].last().map_or(r.epoch as f32, |p| p.0);
            let fail = Marker {
                x: stopped,
                label: format!("failed during epoch {}", r.epoch).into(),
                color: t.danger_text,
                lane: Some(0),
            };
            loss.markers.insert(0, fail.clone());
            map.markers.insert(0, fail);
        }
        let mut pr = base(
            "run-pr",
            vec![
                // One hue for the pair, told apart by dash: no other chart here uses slot 5.
                Series::new("precision", t.chart_series[2], d.precision.clone()).live(live),
                Series::new("recall", t.chart_series[3], d.recall.clone()).dashed().live(live),
            ],
            secondary_h,
        );
        // From a tenth under the lowest value (0 while either is still low), so a converged pair
        // spreads over the plot instead of its top half.
        let low = d.precision.iter().chain(d.recall.iter()).map(|p| p.1).fold(f32::INFINITY, f32::min);
        let floor = pr_floor(low);
        pr.y_range = Some((floor, 1.));
        // From 0: three labels (0, 0.50, 1.00), as the lower row's plot is under 180 px, where five
        // crowd. A raised floor (0.4 or more) labels every 0.2 from it, at most four, so the axis
        // never reads as starting at 0.
        pr.y_ticks = 2;
        if floor > 0. {
            // Every 0.2 from the floor: 0.40 / 0.60 / 0.80 / 1.00 at most (the grid the other
            // charts' ticks sit on, not a computed middle).
            pr.y_tick_values =
                Some((0..).map(|k| floor + 0.2 * k as f32).take_while(|v| *v <= 1.001).collect());
        }
        pr.y_fmt = Some(|v| format!("{v:.2}"));
        let mut lr = base(
            "run-lr",
            vec![Series::new("lr", t.text_tertiary, d.lr.clone()).area().live(live)],
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

        // Legend values: at the shared cursor, else the latest point.
        let v = |pts: &[(f32, f32)], fmt: fn(f32) -> String| {
            crate::charts::value_at(pts, hover, 0., total).map(fmt)
        };
        let f3v: fn(f32) -> String = |x| format!("{x:.3}");
        // Loss parts are toggles; their values are the smoothed curve's, like the lines.
        let mut legend = Vec::new();
        let shown = self.loss_parts.iter().filter(|b| **b).count();
        for (i, name) in names.iter().enumerate() {
            let on = self.loss_parts[i];
            let key = format!("loss-part-{name}");
            let color = colors[i];
            let value = on.then(|| v(&crate::charts::smoothed(&d.loss[i], 0.94), f3v)).flatten();
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
            "Lines are an exponential moving average (0.94) of per-step loss; the faint trace is raw",
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
            crate::ui::legend_value(&t, "mAP50-95", t.chart_series[0], false, v(&d.map50_95, f3v))
                .into_any_element(),
            crate::ui::legend_value(&t, "mAP50", t.chart_series[1], true, v(&d.map50, f3v))
                .into_any_element(),
        ];
        if let Some(b) = &r.best {
            map_items.push(
                div()
                    .flex()
                    .items_center()
                    .gap(px(5.))
                    .child(div().size(px(6.)).rounded_full().bg(t.accent_fill()))
                    .child(text(ty::SM, t.text_secondary).child("best"))
                    .child(text(ty::SM, t.text_primary).tnum().child(format!(
                        "{} @ {}",
                        f3(b.map50_95),
                        b.epoch
                    )))
                    .into_any_element(),
            );
        }
        let map_legend = div().flex().items_center().gap(px(12.)).children(map_items).into_any_element();
        let pr_legend = div()
            .flex()
            .items_center()
            .gap(px(12.))
            .child(crate::ui::legend_value(&t, "precision", t.chart_series[2], false, v(&d.precision, f3v)))
            .child(crate::ui::legend_value(&t, "recall", t.chart_series[3], true, v(&d.recall, f3v)))
            .into_any_element();
        // In the axis's power of ten ("× 10⁻³" is printed once, on the axis).
        // The axis's own top, so caption, ticks and legend share one power.
        let lr_pow = crate::charts::axis_pow(0., lr_top, 3);
        // The legend value is in the axis's power, written out ("6.58×10⁻³", "0.10×10⁻³"): one
        // power per card, and the value still reads on its own (the dogfood report's ask).
        let lr_value = crate::charts::value_at(&d.lr, hover, 0., total).map(|x| match lr_pow {
            // In f64: a tiny valid rate would overflow 10^-p in f32.
            Some(p) if x.is_finite() => crate::charts::in_pow(x as f64, p),
            _ => format!("{x:.4}"),
        });
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
                    .child(div().flex().min_w_0().flex_grow().flex_basis(folo_ui::relative(ga.abs())).child(a))
                    .child(div().flex().min_w_0().flex_grow().flex_basis(folo_ui::relative(gb)).child(b))
            }
        };
        let grid = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(row(
                card(map_title, "per epoch", map_legend, line_chart(&t, map)).into_any_element(),
                card("Training loss", "log scale · EMA 0.94", loss_legend, line_chart(&t, loss))
                    .into_any_element(),
                if pair_a { -0.5 } else { -0.6 },
                if pair_a { 0.5 } else { 0.4 },
            ))
            .child(row(
                card("Precision / Recall", &pr_caption(pr_conf, split, true), pr_legend, line_chart(&t, pr))
                    .into_any_element(),
                card("Learning rate", &lr_caption, lr_legend, line_chart(&t, lr)).into_any_element(),
                0.6,
                0.4,
            ));
        // One Tab stop for the linked charts: arrows walk the epochs.
        // Recorded data ends at the latest step or epoch; the rest of the x range is the schedule.
        let recorded = d.loss[3].last().map_or(0., |p| p.0).max(d.map50_95.last().map_or(0., |p| p.0));
        let end = (recorded / total.max(1.0)).clamp(0., 1.);
        self.chart_group(
            "run-charts",
            1.0 / total.max(1.0),
            end,
            crate::controls::Cursor::Run,
            grid,
            window,
            cx,
        )
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

/// The P/R chart's lower bound: the 0.2 step at least 0.05 under the lowest value, or 0 when that
/// is under 0.4 (or there is no data yet).
fn pr_floor(low: f32) -> f32 {
    if !low.is_finite() {
        return 0.;
    }
    // Steps of 0.2 from 0.4 (a 0.2 floor bought little and read as arbitrary).
    let f = ((low - 0.05) * 5.).floor() / 5.;
    if f < 0.4 { 0. } else { f.min(0.8) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pr_floor_steps() {
        assert_eq!(super::pr_floor(f32::INFINITY), 0.);
        assert_eq!(super::pr_floor(0.12), 0.);
        assert!((super::pr_floor(0.47) - 0.4).abs() < 1e-6);
        assert_eq!(super::pr_floor(0.36), 0.);
        assert!((super::pr_floor(0.71) - 0.6).abs() < 1e-6);
        assert!((super::pr_floor(0.99) - 0.8).abs() < 1e-6);
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
