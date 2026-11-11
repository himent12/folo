//! The live run panel, with the machine's meters as its footer.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{AnyElement, Context, IntoElement, ParentElement, SharedString, Styled, Window, div, px};

use super::page::hero_height;
use crate::api::{Run, RunExt, time};
use crate::app::{FoloApp, Route, act};
use crate::charts::{LineChart, Series, line_chart};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn live_panel(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        // Short windows trade chart height for the sections below.
        // Short windows (620 px minimum): a lower plot, so two Recent runs rows show under the hero.
        let hero_h = hero_height(window);
        // The card's inset: 16 px at the sides always, tighter above and below with the short chart.
        // The chart section's rule spans it.
        let card_pad = 16.;
        let card_pad_y = if hero_h < 120. { 12. } else { 16. };
        let t = theme(cx);
        self.ensure_run_data(&r.id, cx);
        let id = r.id.clone();
        let open = self.button(
            "ov-dash",
            "Open dashboard",
            Btn::Secondary,
            Some("icons/arrow-up-right.svg"),
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
        );
        let data = self
            .store
            .run_data
            .get(&r.id)
            .and_then(|l| l.ready())
            .map(|d| (d.derived(), d.last_step().cloned(), d.last_epoch().cloned()));
        let best = r.best;
        // Every KPI gets a second line, as on the run's dashboard: throughput's recent average and
        // the time spent so far.
        let ips_avg = self.store.run_data.get(&r.id).and_then(|l| l.ready()).and_then(|d| {
            let s = &d.metrics.steps;
            let n = s.len().min(120);
            (n > 0).then(|| s[s.len() - n..].iter().map(|x| x.imgs_per_s).sum::<f32>() / n as f32)
        });
        let elapsed = crate::screens::runs::duration_of(r, self.store.api.now())
            .map(|d| format!("{} elapsed", time::duration(d)));
        let (chart, legend_el, ips, map_now) = match data {
            Some((d, step, epoch)) => {
                let hover = self.chart_hover;
                let on_hover: crate::charts::HoverFn = {
                    let view = cx.entity().downgrade();
                    Rc::new(move |f, _, cx| {
                        view.update(cx, |this, cx| {
                            if this.chart_hover != f {
                                this.chart_hover = f;
                                cx.notify();
                            }
                        })
                        .ok();
                    })
                };
                let total = r.epochs as f32;
                // One chart here: validation mAP is what a glance needs; loss lives on the dashboard.
                let mut map = LineChart::new(
                    "ov-map",
                    vec![
                        Series::new("mAP50-95", t.chart_series[0], d.map50_95.clone()).area().live(true),
                        // Dashed: the pair reads apart without colour (as in the legend).
                        Series::new("mAP50", t.chart_series[1], d.map50.clone())
                            .width(1.25)
                            .dashed()
                            .live(true),
                    ],
                );
                map.x_range = Some((0., total));
                map.x_int = true;
                map.ease_key = crate::charts::ease_key(&r.id);
                // Epochs are whole numbers ("5", not "5.0").
                map.x_fmt = |v| format!("{v:.0}");
                map.y_range = Some((0., 1.));
                // Short windows: 0 / 0.50 / 1.00, so labels keep their distance in the lower plot.
                map.y_ticks = if hero_h < 120. { 2 } else { 4 };
                map.y_fmt = Some(|v| format!("{v:.2}"));
                // One annotation lane above the plot, as on the run dashboard's charts: the ETA sits
                // there, not inside the data. The card grows by the lane; short windows take it out
                // of the plot instead, so Recent runs keeps its rows.
                map.min_lanes = 1;
                map.height = hero_h + 16. + if hero_h < 120. { 0. } else { 16. };
                map.hover = hover;
                map.on_hover = Some(on_hover);
                map.x_name = "epoch";
                map.readout = false;
                // The span shows what's left, labelled like the run dashboard's.
                map.future = Some((
                    r.epoch as f32,
                    r.eta_s.map_or(SharedString::default(), |s| format!("ETA {}", time::duration(s)).into()),
                ));
                if let Some(b) = &best {
                    map.dots.push((b.epoch as f32, b.map50_95, t.accent_fill()));
                }
                let v = |pts: &[(f32, f32)]| {
                    crate::charts::value_at(pts, hover, 0., total).map(|x| format!("{x:.3}"))
                };
                let legend_el = div()
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .child(crate::ui::legend_value(&t, "mAP50-95", t.chart_series[0], false, v(&d.map50_95)))
                    .child(crate::ui::legend_value(&t, "mAP50", t.chart_series[1], true, v(&d.map50)))
                    // The lime dot on the chart, keyed as on the run dashboard.
                    .children(best.as_ref().map(|b| {
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
                    }))
                    .into_any_element();
                // One Tab stop with the dashboard's keyboard cursor (← / →, Home / End, Esc), ending
                // at the latest validated epoch.
                let last = d.map50_95.last().map_or(0., |p| p.0);
                let chart = self.chart_group(
                    "ov-chart",
                    1. / total.max(1.),
                    (last / total.max(1.)).clamp(0., 1.),
                    crate::controls::Cursor::Run,
                    div().child(line_chart(&t, map)),
                    window,
                    cx,
                );
                (chart, legend_el, step.map(|s| s.imgs_per_s), epoch.map(|e| e.val.map50_95))
            }
            None => (
                div().h(px(hero_h + 16.)).child(skeleton(&t, "ov-charts")).into_any_element(),
                div().into_any_element(),
                None,
                None,
            ),
        };
        let eta_s = r.eta_s;
        let best_line = best.map(|b| format!("best {} @ {}", f3(b.map50_95), b.epoch)).unwrap_or_default();
        let kpi = |label: &str, value: AnyElement, detail: Option<String>| {
            div()
                .flex()
                .flex_col()
                .gap(px(4.))
                .min_w_0()
                .child(caption(&t, label))
                .child(value)
                .when_some(detail, |d, s| d.child(text(ty::SM, t.text_tertiary).tnum().child(s)))
        };
        // One row of four KPIs above a full-width chart: no column beside the chart to leave a
        // gap in, whatever the chart's height. Four equal tracks (a grid, so a tile's padding and
        // divider can't make it wider than the others) with 1 px dividers, like the run, model and
        // dataset strips.
        // Every tile insets its content by the card's padding (the strip itself has none at the
        // sides), so the four labels step evenly across the card and the first lines up with the
        // header's name.
        let tile = |first: bool, body| {
            div()
                .min_w_0()
                .pl(px(card_pad))
                .when(!first, |d| d.border_l_1().border_color(t.line_subtle))
                .child(body)
        };
        let kpis = div()
            .w_full()
            .grid()
            .grid_cols(4)
            .child(tile(
                true,
                // Progress is drawn once, as the line under the card's header; here it's a
                // number like the other tiles' detail lines.
                kpi(
                    "Epoch",
                    metric(&t, r.epoch.to_string(), &format!("/ {}", r.epochs), ty::X2XL, ty::SM)
                        .into_any_element(),
                    Some(format!("{:.0}% complete", r.progress() * 100.)),
                ),
            ))
            .child(tile(
                false,
                kpi(
                    // Named as on the run's dashboard.
                    "mAP50-95 · latest",
                    text(ty::X2XL, t.text_primary)
                        .tnum()
                        .child(map_now.map_or("–".into(), f3))
                        .into_any_element(),
                    Some(best_line),
                ),
            ))
            .child(tile(
                false,
                kpi(
                    "Throughput",
                    metric(&t, ips.map_or("–".into(), |v| format!("{v:.0}")), "img/s", ty::X2XL, ty::SM)
                        .into_any_element(),
                    ips_avg.map(|a| format!("avg {a:.0}")),
                ),
            ))
            .child(tile(
                false,
                kpi("Time left", duration_metric(&t, eta_s, ty::X2XL, ty::SM).into_any_element(), elapsed),
            ));
        let footer = self.system_footer(cx);
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            // Header: name, status, meta, and the way to the full dashboard.
            .child(
                div()
                    .h(px(48.))
                    .px(px(16.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    // A long name gives way (with "…") before it can push the meta under the
                    // action; the meta's own box clips as the last line of defence.
                    .child(
                        text(ty::LG, t.text_primary)
                            .flex_shrink()
                            .min_w(px(80.))
                            .ellipsis()
                            .child(r.name.clone()),
                    )
                    .child(status_badge(&t, r, &r.id))
                    .child(div().flex_1().min_w_0().overflow_hidden().child(self.run_subtitle(
                        &t,
                        r,
                        ty::SM,
                        t.text_tertiary,
                        "hero",
                    )))
                    .child(open),
            )
            // Progress as a hairline rail across the panel.
            .child(
                div()
                    .h(px(2.))
                    .bg(t.bg_inset)
                    .child(div().h_full().w(folo_ui::relative(r.progress().clamp(0., 1.))).bg(t.accent_fill())),
            )
            .child(div().w_full().py(px(card_pad_y)).child(kpis))
            // The chart section is the run page's chart card without its frame: a header row on a
            // full-width rule, the legend row, then the plot, so the two screens' charts read alike.
            .child(div().w_full().min_w_0().border_t_1().border_color(t.line_subtle).child(
                crate::ui::chart_section(&t, self.map_title(r), "per epoch", legend_el, chart, hero_h < 120.),
            ))
            .children(footer)
            .into_any_element()
    }

    /// The machine at a glance, as a 36 px footer row of meters (GPU, CPU, memory).
    fn system_footer(&mut self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let t = theme(cx);
        let s = self.store.system.ready()?.clone();
        let meter = |label: String, value: String, frac: f32, color: folo_ui::Hsla| {
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(text(ty::SM, t.text_tertiary).whitespace_nowrap().child(label))
                .child(div().w(px(48.)).child(progress(&t, frac, color, 3.)))
                .child(text(ty::SM, t.text_secondary).tnum().whitespace_nowrap().child(value))
        };
        let mut row = div()
            .h(px(36.))
            .px(px(16.))
            .flex()
            .items_center()
            .gap(px(24.))
            .border_t_1()
            .border_color(t.line_subtle);
        if let Some(g) = s.gpus.first() {
            // wgpu can't report memory or load on every platform: say so instead of "0 B · 0%".
            let util = g.util.unwrap_or(0.0);
            let label = if g.memory_bytes > 0 {
                let (vram, vu) = bytes(g.memory_bytes);
                // Its total memory, named (the meter beside it is load, not memory use).
                format!("{} · {vram} {vu} VRAM", crate::ui::short_gpu(&g.name))
            } else {
                format!("{} · VRAM unknown", crate::ui::short_gpu(&g.name))
            };
            row = row.child(meter(
                label,
                g.util.map_or("–".into(), |u| format!("{:.0}% load", u * 100.)),
                util,
                if util > 0.95 { t.warning_solid } else { t.text_tertiary },
            ));
        }
        let mem = s.memory.used_bytes as f32 / s.memory.total_bytes.max(1) as f32;
        let (used, _) = bytes(s.memory.used_bytes);
        let (total, unit) = bytes(s.memory.total_bytes);
        row = row
            .child(meter(
                format!("CPU · {} threads", s.cpu.threads),
                format!("{:.0}%", s.cpu.util * 100.),
                s.cpu.util,
                t.text_tertiary,
            ))
            .child(meter("Memory".into(), format!("{used} / {total} {unit}"), mem, t.text_tertiary));
        Some(row.into_any_element())
    }
}
