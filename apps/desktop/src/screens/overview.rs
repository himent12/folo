//! Overview: the live run, the machine, recent runs, best models and datasets at a glance.

use std::rc::Rc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::*;
use crate::api::time;
use crate::api::{Run, RunStatus};
use crate::app::{FoloApp, Route, act};
use crate::charts::{LineChart, Series, line_chart};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{radius, theme, ty};
use crate::ui::*;

/// Width of the right-hand column on both Overview rows, so their gutters line up.
const SIDE_W: f32 = 360.;

impl FoloApp {
    pub fn overview(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let n_ds = self.store.datasets.ready().map(|d| d.len());
        let n_models = self.store.models.ready().map(|m| m.len());
        let n_live =
            self.store.runs.ready().map(|r| r.iter().filter(|r| r.status == RunStatus::Running).count());
        let subtitle = match (n_ds, n_models, n_live) {
            (Some(d), Some(m), Some(l)) => {
                let live = match l {
                    0 => "nothing training".to_string(),
                    1 => "1 run training".to_string(),
                    n => format!("{n} runs training"),
                };
                format!("{} · {} · {live}", plural(d, "dataset"), plural(m, "model"))
            }
            // The header keeps its subtitle row: where the app tried to connect (the panel below
            // says what went wrong).
            _ if self.store.runs.error().is_some() => {
                format!("Not connected · {}", self.store.api.describe())
            }
            // Runs came back but something else failed: not "Loading…" for ever.
            _ if self.store.models.error().is_some() || self.store.datasets.error().is_some() => {
                "Some data didn't load · retry in its panel".to_string()
            }
            // The sidebar says "Connecting…" until the server answers; the page says the same.
            _ if self.store.health.is_loading() => "Connecting…".to_string(),
            _ => "Loading…".to_string(),
        };
        // Training needs a server and a dataset; until then the action explains itself instead.
        let no_ds = self.store.datasets.ready().is_some_and(|d| d.is_empty());
        let blocked = if self.store.runs.error().is_some() {
            Some("Connect to folo serve first")
        } else if no_ds {
            Some("Add a dataset first")
        } else if self.store.datasets.error().is_some() {
            // Settled, not loading: say so, and where to retry.
            Some("Datasets didn't load · retry on the Datasets page")
        } else if self.store.datasets.ready().is_none() {
            Some("Loading datasets…")
        } else {
            None
        };
        let new_run = match blocked {
            Some(reason) => self
                .button_disabled(
                    "ov-new",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    reason,
                    window,
                    cx,
                )
                .into_any_element(),
            None => self
                .button(
                    "ov-new",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_run(None, w, cx)),
                )
                .tooltip(crate::controls::tip("Start training", Some("Ctrl N")))
                .into_any_element(),
        };
        // One rule for Open image: it needs a model to run (not connected, still loading or none
        // yet, it says why, as New run beside it does).
        let open_blocked = self.open_image_blocked();
        let open = if let Some(why) = open_blocked {
            self.button_disabled(
                "ov-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                why,
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "ov-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, _, cx| this.pick_image(cx)),
            )
            .tooltip(crate::controls::tip("Run a model on an image", Some("Ctrl O")))
            .into_any_element()
        };
        let actions = vec![open, new_run];
        let nothing_yet = no_ds
            && self.store.runs.ready().is_some_and(|r| r.is_empty())
            && self.store.models.ready().is_some_and(|m| m.is_empty());
        // One rule for the header: actions show once the page has settled and has a body of its own
        // to act on. First run (the getting-started panel's "Add dataset" is the call to action),
        // not connected (the error panel's Retry is) and still loading show none, so nothing
        // flashes in disabled and then vanishes. A list that failed has settled: each action then
        // says for itself whether it can run (a failed models list doesn't hide New run).
        let pending = [
            self.store.runs.ready().is_none() && self.store.runs.error().is_none(),
            self.store.models.ready().is_none() && self.store.models.error().is_none(),
            self.store.datasets.ready().is_none() && self.store.datasets.error().is_none(),
        ]
        .contains(&true);
        let hide = nothing_yet || pending || self.store.runs.error().is_some();
        let actions = if hide { Vec::new() } else { actions };
        let header = page_header(&t, "Overview", text(ty::BASE, t.text_secondary).child(subtitle), actions);

        if let Some(e) = self.store.runs.error().cloned() {
            let err = self.error_panel(&e, "ov-retry", window, cx);
            return self.page_frame(&t, header, err, cx);
        }

        if nothing_yet {
            let body = self.getting_started(window, cx);
            return self.page_frame(&t, header, body, cx);
        }
        let live = self.store.live_run().cloned();
        let hero = match &live {
            Some(r) => self.live_panel(r, window, cx),
            None => self.idle_panel(window, cx),
        };
        // With a live run the machine sits in the hero's footer; otherwise a slim strip says it.
        // While runs load, the hero skeleton already reserves the machine footer.
        let strip = (live.is_none() && !self.store.runs.is_loading()).then(|| self.system_strip(cx));
        let recent = self.recent_runs(window, cx);
        let models = self.top_models(window, cx);
        let datasets = self.dataset_strip(window, cx);
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .child(div().flex().child(hero))
            .when_some(strip, |d, s| d.child(s))
            // Narrow windows stack the two lists full width rather than squeeze the runs table.
            .map(|d| {
                // Side by side only when Recent runs keeps a full row beside Best model (measured,
                // so a content cap change can't squeeze it); otherwise the two tables stack.
                if !self.tables_beside(window) {
                    d.child(div().flex().child(recent)).child(div().flex().child(models))
                } else {
                    d.child(
                        div()
                            .flex()
                            .gap(px(16.))
                            .child(div().flex_1().min_w_0().flex().child(recent))
                            .child(div().w(px(SIDE_W + 40.)).flex_none().flex().child(models)),
                    )
                }
            })
            .child(datasets);
        self.page_frame(&t, header, body, cx)
    }

    fn live_panel(&mut self, r: &Run, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
                    .child(div().h_full().w(gpui::relative(r.progress().clamp(0., 1.))).bg(t.accent_fill())),
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
        let meter = |label: String, value: String, frac: f32, color: gpui::Hsla| {
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

    fn idle_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        if self.store.runs.is_loading() {
            // Same anatomy and heights as the live panel: header, rail, a row of four KPIs above the
            // chart (its title row, then the plot), meter footer. Nothing moves when data lands.
            let hero_h = hero_height(window);
            let sk = |id: &str| skeleton(&t, id.to_string());
            return panel(&t)
                .flex_1()
                .flex()
                .flex_col()
                .child(
                    div()
                        .h(px(48.))
                        .px(px(16.))
                        .flex()
                        .items_center()
                        .child(div().w(px(260.)).h(px(18.)).child(sk("hero-title"))),
                )
                .child(div().h(px(2.)).bg(t.bg_inset))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(if hero_h < 120. { 8. } else { 16. }))
                        // 16 px at the sides always, as the live card's.
                        .px(px(16.))
                        .py(px(if hero_h < 120. { 12. } else { 16. }))
                        // One strip with 1 px dividers, as the loaded tiles sit.
                        .child(div().grid().grid_cols(4).children((0..4).map(|c| {
                            div()
                                .min_w_0()
                                .h(px(72.))
                                .pr(px(16.))
                                .when(c > 0, |d| d.border_l_1().border_color(t.line_subtle).pl(px(16.)))
                                .child(skeleton(&t, format!("hero-kpi-{c}")))
                        })))
                        .child(
                            // The chart section opens on the same full-width rule and inset as the
                            // live card's, so the loaded card is exactly this tall.
                            div()
                                .flex_1()
                                .flex()
                                .flex_col()
                                .gap(px(4.))
                                .mx(px(-16.))
                                .px(px(16.))
                                .pt(px(if hero_h < 120. { 12. } else { 16. }))
                                .border_t_1()
                                .border_color(t.line_subtle)
                                .child(div().h(px(20.)).w(px(120.)).child(sk("hero-chart-title")))
                                .child(div().mt(px(2.)).h(px(18.)).w(px(220.)).child(sk("hero-chart-legend")))
                                .child(div().h(px(hero_h + 10.)).child(sk("hero-chart"))),
                        ),
                )
                .child(
                    // The machine footer's meters, as placeholders.
                    div()
                        .h(px(36.))
                        .px(px(16.))
                        .flex()
                        .items_center()
                        .gap(px(24.))
                        .border_t_1()
                        .border_color(t.line_subtle)
                        .children((0..3).map(|i| {
                            div().w(px(150.)).h(px(10.)).child(skeleton(&t, format!("hero-meter-{i}")))
                        })),
                )
                .into_any_element();
        }
        let has_ds = self.store.datasets.ready().is_some_and(|d| !d.is_empty());
        let (title, body) = if has_ds {
            ("Nothing is training", "Start a run to watch loss, mAP and throughput update live here.")
        } else {
            (
                "No datasets yet",
                "Register a YOLO or COCO folder, or generate a synthetic shapes dataset to try FOLO end to end.",
            )
        };
        let action = if has_ds {
            self.button(
                "ov-empty-new",
                "New run",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| this.open_new_run(None, w, cx)),
            )
        } else {
            self.button(
                "ov-empty-ds",
                "Add dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    // The label promises the picker: land on Datasets with it open (the built-in demo
                    // can't register folders, and says so there instead).
                    this.navigate(Route::Datasets, w, cx);
                    this.add_dataset_folder(cx);
                }),
            )
        };
        panel(&t)
            .flex_1()
            .flex()
            .items_center()
            .justify_center()
            .child(
                empty_state(
                    &t,
                    if has_ds { "icons/activity.svg" } else { "icons/images.svg" },
                    title,
                    body,
                    Tone::Neutral,
                )
                .child(action),
            )
            .into_any_element()
    }

    fn system_strip(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(s) = self.store.system.ready().cloned() else {
            return panel(&t).h(px(48.)).child(skeleton(&t, "sys-strip")).into_any_element();
        };
        let cell = |label: String, value: String, frac: f32, color: gpui::Hsla| {
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .items_center()
                .gap(px(10.))
                .px(px(16.))
                .child(text(ty::SM, t.text_secondary).flex_none().whitespace_nowrap().child(label))
                .child(div().flex_1().min_w(px(80.)).child(progress(&t, frac, color, 4.)))
                .child(text(ty::SM, t.text_primary).flex_none().tnum().whitespace_nowrap().child(value))
        };
        let mut row = div().h(px(48.)).flex().items_center();
        if let Some(g) = s.gpus.first() {
            let util = g.util.unwrap_or(0.0);
            row = row
                .child(cell(
                    crate::ui::short_gpu(&g.name),
                    g.util.map_or("–".into(), |u| format!("{:.0}% load", u * 100.)),
                    util,
                    // A meter, not an action: neutral like CPU and memory, amber when saturated.
                    if util > 0.95 { t.warning_solid } else { t.text_tertiary },
                ))
                .child(vline(&t));
        }
        let mem = s.memory.used_bytes as f32 / s.memory.total_bytes.max(1) as f32;
        let (used, _) = bytes(s.memory.used_bytes);
        let (total, unit) = bytes(s.memory.total_bytes);
        row = row
            .child(cell("CPU".into(), format!("{:.0}%", s.cpu.util * 100.), s.cpu.util, t.text_tertiary))
            .child(vline(&t))
            .child(cell("Memory".into(), format!("{used} / {total} {unit}"), mem, t.text_tertiary));
        panel(&t).child(row).into_any_element()
    }

    /// Whether Recent runs sits beside Best model: from 1500 px, and only while the list keeps its
    /// narrow column set (798 px, plus 8 of slack) in what Best model leaves. A row wider than
    /// its slot makes GPUI lay the whole page out at its min-content width.
    fn tables_beside(&self, window: &Window) -> bool {
        window.viewport_size().width >= px(1500.) && self.content_width(window) - (SIDE_W + 40.) - 16. >= 806.
    }

    fn recent_runs(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        // The list is never squeezed beside Best model (`tables_beside` stacks them first), so
        // the architecture and dataset always have their column.
        let created = true;
        // The list's width: the content (stacked) or what Best model leaves beside it.
        // The wider name and dataset columns need 918 px (812 of columns, six 12 px gaps, row
        // padding, margins, borders); with less, the narrower set (798 px) keeps Created inside.
        let content = self.content_width(window);
        let beside = self.tables_beside(window);
        let list_w = if beside { content - (SIDE_W + 40.) - 16. } else { content };
        // 8 px of slack on every budget: a row exactly as wide as its slot can overflow it by a
        // rounding error, and GPUI then lays the whole page out at its min-content width (the
        // hero chart collapsed to its axis labels at 1680).
        const SLACK: f32 = 8.;
        let full = list_w >= 918. + SLACK;
        let (name_w, prog_w, best_w, when_w) =
            if full { (200., 160., 96., 80.) } else { (150., 120., 96., 80.) };
        // At least its heading, "Architecture · dataset".
        let meta_w = if full { 180. } else { 150. };
        // Stacked, width past the set's need goes to the name, arch · dataset and
        // progress (2 : 1 : 1, to their caps); only what's left opens before the numbers. Beside
        // Best model the set stays as is: widened there, GPUI 0.2.2 laid the page out at its
        // min-content width (the hero chart collapsed to its axis at 1680).
        let spare = if beside { 0. } else { list_w - SLACK - if full { 918. } else { 798. } };
        let [name_w, meta_w, prog_w] =
            crate::ui::share_spare(spare, [(name_w, 2., 280.), (meta_w, 1., 240.), (prog_w, 1., 200.)]);
        let now = self.store.api.now();
        let all = self.view_all("ov-runs-all", Route::Runs, window, cx);
        let header = panel_header(&t, "Recent runs").child(all);
        // Column heads on the rows' grid, so the metric label sits over its values.
        let heads = div()
            .h(px(28.))
            .mx(px(4.))
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(12.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "Run").w(px(name_w)).flex_none())
            .child(th(&t, "Architecture · dataset").w(px(meta_w)).flex_none())
            .child(th(&t, "Status").w(px(96.)).flex_none())
            .child(th(&t, "Progress").w(px(prog_w)).flex_none())
            .child(div().flex_1())
            .child(div().w(px(best_w)).flex_none().flex().justify_end().child(th(&t, "Best mAP50-95")))
            .when(created, |d| {
                d.child(div().w(px(when_w)).flex_none().flex().justify_end().child(th(&t, "Created")))
            });
        let body = match &self.store.runs {
            Load::Ready(runs) if runs.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/activity.svg",
                    "No runs yet",
                    "Training runs appear here with their status and best mAP.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(runs) => {
                let runs: Vec<Run> = runs.iter().take(5).cloned().collect();
                div()
                    .flex()
                    .flex_col()
                    .pb(px(4.))
                    .child(heads)
                    .children(runs.into_iter().map(|r| {
                        let created_at = time::parse(&r.created);
                        let when = created_at.map_or(String::new(), |c| time::ago(c, now));
                        // Progress (training runs only) right after Status; best mAP50-95 at the end.
                        let training = r.status.is_active() && r.status != crate::api::RunStatus::Queued;
                        let progress_el =
                            div().w(px(prog_w)).flex_none().flex().items_center().gap(px(8.)).map(|d| {
                                if r.status == crate::api::RunStatus::Queued {
                                    // The Runs table's phrase, so a queued run reads the same everywhere.
                                    return d.child(
                                        text(ty::SM, t.text_tertiary)
                                            .whitespace_nowrap()
                                            .child(waiting_text(&r)),
                                    );
                                }
                                let fill = match r.status {
                                    _ if training => t.accent_fill(),
                                    crate::api::RunStatus::Failed => t.danger_solid,
                                    _ => t.text_tertiary,
                                };
                                // A bar under 48 px reads as a stub: narrow columns keep the
                                // number only.
                                d.when(prog_w - 64. >= 48., |d| {
                                    d.child(div().flex_1().child(progress(&t, r.progress(), fill, 4.)))
                                })
                                .child(
                                    text(ty::SM, t.text_secondary)
                                        .tnum()
                                        .w(px(56.))
                                        .flex_none()
                                        .whitespace_nowrap()
                                        .text_right()
                                        .child(format!("{} / {}", r.epoch, r.epochs)),
                                )
                            });
                        let best = div().w(px(best_w)).flex_none().flex().justify_end().child(
                            text(ty::SM, if r.best.is_some() { t.text_primary } else { t.text_tertiary })
                                .tnum()
                                .child(r.best.as_ref().map_or("–".to_string(), |b| f3(b.map50_95))),
                        );
                        let id = r.id.clone();
                        let key = format!("ov-run-{}", r.id);
                        // Keyboard focus shows what hover does: the full name and the exact time.
                        let e2e = r.config.head == crate::api::Head::O2o;
                        // The focus tip says in full what the row may cut: name, architecture,
                        // dataset, created time, and the head.
                        let focus_text = self.ring(&self.fh(key.clone(), cx), window).then(|| {
                            let who = format!("{} · {}", r.name, self.run_meta(&r));
                            let base = created_at.map_or(who.clone(), |c| {
                                format!("{who} · created {}", time::local_datetime(c))
                            });
                            if e2e { base + " · end-to-end head: predicts without NMS" } else { base }
                        });
                        // Architecture first (short, so it survives a narrow cell), then the
                        // dataset, then the e2e tag for an end-to-end run.
                        // The e2e tag rides its architecture, as in the Runs and Models lists; the
                        // dataset after them gives way first.
                        let ds_name = self
                            .store
                            .dataset_summary(&r.config.dataset_id)
                            .map_or(r.config.dataset_id.clone(), |d| d.name.clone());
                        let meta = |w: f32| {
                            div().w(px(w)).flex_none().child(meta_row(
                                ty::SM,
                                t.text_tertiary,
                                r.config.arch.as_str(),
                                e2e.then(|| e2e_tag(&t, format!("e2e-ovr-{}-{w}", r.id))),
                                std::slice::from_ref(&ds_name),
                            ))
                        };
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .when_some(focus_text, |d, s| d.child(crate::controls::focus_tip(&t, s, 36.)))
                            .h(px(36.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            // Same order as the Runs table: name, then status.
                            // One line per run: name, then dataset and architecture, then status as a dot.
                            // A name that may be cut shows in full on hover (and on focus, above).
                            .child(
                                div()
                                    .id(SharedString::from(format!("ov-rn-{}", r.id)))
                                    .w(px(name_w))
                                    .flex_none()
                                    .when(crate::ui::may_overflow(&r.name, ty::SM.size, name_w), |d| {
                                        d.tooltip(crate::controls::tip(r.name.clone(), None))
                                    })
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .w_full()
                                            .ellipsis()
                                            .child(r.name.clone()),
                                    ),
                            )
                            // Wide: the dataset and architecture in their own column.
                            .child(meta(meta_w))
                            .child(div().w(px(96.)).flex_none().child(status_dot(
                                &t,
                                &r,
                                &format!("ov-{}", r.id),
                            )))
                            .child(progress_el)
                            .child(div().flex_1())
                            .child(best)
                            .when(created, |d| {
                                d.child(
                                    div().w(px(when_w)).flex_none().flex().justify_end().child(
                                        text(ty::SM, t.text_tertiary)
                                            .id(SharedString::from(format!("ov-rc-{}", r.id)))
                                            .when_some(created_at, |d, c| {
                                                d.tooltip(crate::controls::tip(time::local_datetime(c), None))
                                            })
                                            .tnum()
                                            .whitespace_nowrap()
                                            .child(when),
                                    ),
                                )
                            });
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Run(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-runs", 5).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(header)
            .child(body)
            .into_any_element()
    }

    fn top_models(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-models-all", Route::Models, window, cx);
        // mAP compares only within one dataset's validation split, so the card ranks within each
        // dataset and shows its best: one row per dataset, never one list across them.
        let header = panel_header(&t, "Best model per dataset").child(all);
        let body = match &self.store.models {
            Load::Ready(models) if models.is_empty() => div()
                .child(empty_state(
                    &t,
                    "icons/box.svg",
                    "No models yet",
                    "Finished runs save their best weights here.",
                    Tone::Neutral,
                ))
                .into_any_element(),
            Load::Ready(models) => 'card: {
                // Each evaluated model's dataset (from the run that trained it; an imported model
                // without one is left out: there's nothing to compare it with).
                let ds_of = |m: &crate::api::Model| {
                    m.run_id.as_ref().and_then(|r| self.store.run(r)).map(|r| r.config.dataset_id.clone())
                };
                let mut best: Vec<(String, crate::api::Model)> = Vec::new();
                for m in models.iter().filter(|m| m.metrics.is_some()) {
                    let Some(ds) = ds_of(m) else { continue };
                    let map = |m: &crate::api::Model| m.metrics.as_ref().map_or(0., |x| x.map50_95);
                    match best.iter_mut().find(|(d, _)| *d == ds) {
                        Some((_, b)) if map(m) > map(b) => *b = m.clone(),
                        Some(_) => {}
                        None => best.push((ds, m.clone())),
                    }
                }
                // Datasets in name order (their mAPs aren't comparable, so no rank between rows).
                let name_of =
                    |ds: &str| self.store.dataset_summary(ds).map_or(ds.to_string(), |d| d.name.clone());
                best.sort_by_key(|(ds, _)| name_of(ds));
                let models: Vec<(String, crate::api::Model)> =
                    best.into_iter().map(|(ds, m)| (name_of(&ds), m)).collect();
                // Models, but none to rank yet: unevaluated, imported, or their runs still loading.
                if models.is_empty() {
                    let body = if self.store.runs.error().is_some() {
                        "Runs didn't load, so the datasets models were trained on aren't known: retry above."
                    } else if self.store.runs.ready().is_none() {
                        "Waiting for runs to say which dataset each model was trained on."
                    } else {
                        "Evaluated models trained here show up per dataset."
                    };
                    break 'card div()
                        .child(empty_state(
                            &t,
                            "icons/box.svg",
                            "Nothing to compare yet",
                            body,
                            Tone::Neutral,
                        ))
                        .into_any_element();
                }
                // Stacked under Recent runs (full width), the table carries the Models list's
                // architecture, mAP50 and age too, rather than three columns across 1,100 px.
                let wide = !self.tables_beside(window);
                let now = self.store.api.now();
                // Column heads like Recent runs beside it, so the two lists share row lines.
                let heads = div()
                    .h(px(28.))
                    .mx(px(4.))
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .gap(px(12.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    // The dataset leads: it's what the rows are, one each, in name order.
                    .child(th(&t, "Dataset").w(px(120.)).flex_none())
                    .child(th(&t, "Best model").flex_1())
                    .when(wide, |d| d.child(th(&t, "Architecture").w(px(120.)).flex_none()))
                    .child(div().w(px(64.)).flex_none().flex().justify_end().child(th(&t, "mAP50-95")))
                    .when(wide, |d| {
                        d.child(div().w(px(64.)).flex_none().flex().justify_end().child(th(&t, "mAP50")))
                            .child(div().w(px(80.)).flex_none().flex().justify_end().child(th(&t, "Created")))
                    });
                div()
                    .flex()
                    .flex_col()
                    .pb(px(4.))
                    .child(heads)
                    .children(models.into_iter().take(5).map(|(ds_name, m)| {
                        let id = m.id.clone();
                        let key = format!("ov-model-{}", m.id);
                        let map = m.metrics.as_ref().map_or(0., |m| m.map50_95);
                        let map50 = m.metrics.as_ref().map(|m| m.map50);
                        let arch = format!("{} · {} px", m.arch.as_str(), m.imgsz);
                        let created = time::parse(&m.created).map_or(String::new(), |c| time::ago(c, now));
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .h(px(36.))
                            .mx(px(4.))
                            .px(px(12.))
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .rounded(px(radius::MD))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .w(px(120.))
                                    .flex_none()
                                    .ellipsis()
                                    .child(ds_name.clone()),
                            )
                            // Its best model, then the same compact e2e tag as the Models table.
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(
                                        text(ty::SM, t.text_primary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .min_w_0()
                                            .ellipsis()
                                            .child(m.name.clone()),
                                    )
                                    .when(self.model_e2e(&m), |d| {
                                        d.child(e2e_tag(&t, format!("e2e-ov-{}", m.id)))
                                    }),
                            )
                            .when(wide, |d| {
                                d.child(
                                    text(ty::SM, t.text_secondary)
                                        .tnum()
                                        .w(px(120.))
                                        .flex_none()
                                        .ellipsis()
                                        .child(arch),
                                )
                            })
                            // The number alone: a bar would set rows from different datasets against
                            // each other, which their mAPs can't support.
                            .child(
                                div()
                                    .w(px(64.))
                                    .flex_none()
                                    .flex()
                                    .justify_end()
                                    .child(text(ty::SM, t.text_secondary).tnum().child(f3(map))),
                            )
                            .when(wide, |d| {
                                d.child(div().w(px(64.)).flex_none().flex().justify_end().child(
                                    text(ty::SM, t.text_tertiary).tnum().child(map50.map_or("–".into(), f3)),
                                ))
                                .child(
                                    div().w(px(80.)).flex_none().flex().justify_end().child(
                                        text(ty::SM, t.text_tertiary).whitespace_nowrap().child(created),
                                    ),
                                )
                            });
                        self.bind(
                            row,
                            &key,
                            radius::MD,
                            window,
                            cx,
                            act(move |this, w, cx| this.navigate(Route::Model(id.clone()), w, cx)),
                        )
                        .into_any_element()
                    }))
                    .into_any_element()
            }
            // A failed list says so and can be fetched again (it would otherwise read as loading
            // forever: nothing else retries a failed load).
            Load::Failed(e) => {
                let retry = self.button(
                    "ov-models-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_models(cx)),
                );
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .items_start()
                    .gap(px(8.))
                    .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message.clone()))
                    .child(retry)
                    .into_any_element()
            }
            _ => loading_rows(&t, "ov-models", 5).into_any_element(),
        };
        panel(&t)
            .flex_1()
            .min_w_0()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(header)
            .child(body)
            .into_any_element()
    }

    fn dataset_strip(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let all = self.view_all("ov-ds-all", Route::Datasets, window, cx);
        let datasets = self.store.datasets.ready().cloned().unwrap_or_default();
        let cards: Vec<AnyElement> =
            datasets.iter().take(3).map(|d| self.dataset_card(d, window, cx)).collect();
        panel(&t)
            .flex()
            .flex_col()
            .child(panel_header(&t, "Datasets").child(all))
            .map(|d| {
                if cards.is_empty() && self.store.datasets.ready().is_some() {
                    d.child(empty_state(
                        &t,
                        "icons/images.svg",
                        "No datasets registered",
                        "Datasets you add or generate show up here with a preview of their images.",
                        Tone::Neutral,
                    ))
                } else if cards.is_empty() {
                    d.child(div().p(px(16.)).flex().gap(px(16.)).children(
                        (0..3).map(|i| div().flex_1().h(px(196.)).child(skeleton(&t, format!("ov-ds-{i}")))),
                    ))
                } else {
                    let n = cards.len();
                    d.child(
                        div()
                            .p(px(16.))
                            .flex()
                            .gap(px(16.))
                            .children(cards.into_iter().map(|c| {
                                div()
                                    .flex_grow()
                                    .flex_basis(px(0.))
                                    .min_w_0()
                                    .flex()
                                    .child(c)
                                    .into_any_element()
                            }))
                            .children((n..3).map(|_| div().flex_grow().flex_basis(px(0.)).min_w_0())),
                    )
                }
            })
            .into_any_element()
    }

    /// First run: what FOLO needs, in order, with the next step as the one primary action.
    fn getting_started(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let add = self
            .button(
                "gs-add",
                "Add dataset",
                Btn::Primary,
                Some("icons/plus.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    // The label promises the picker: land on Datasets with it open (the built-in demo
                    // can't register folders, and says so there instead).
                    this.navigate(Route::Datasets, w, cx);
                    this.add_dataset_folder(cx);
                }),
            )
            .into_any_element();
        // The app makes a sample itself (Datasets' generator), so a first run needn't leave for a
        // terminal to have something to train on.
        let synth = self
            .button(
                "gs-synth",
                "Generate synthetic",
                Btn::Secondary,
                Some("icons/zap.svg"),
                window,
                cx,
                act(|this, w, cx| {
                    this.navigate(Route::Datasets, w, cx);
                    this.open_synth(w, cx);
                }),
            )
            .into_any_element();
        let step = |n: &str, title: &str, body: &str, active: bool| {
            div()
                // Fills its wrapper, so the three cards end at one height whichever wraps most; may
                // shrink below its text (which then wraps).
                .flex_1()
                .min_w_0()
                .flex()
                .gap(px(14.))
                .p(px(16.))
                .rounded(px(radius::LG))
                .border_1()
                .border_color(if active { t.line_strong } else { t.line_subtle })
                .when(active, |d| d.bg(t.bg_raised))
                .child(
                    div()
                        .size(px(24.))
                        .flex_none()
                        .rounded_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(if active { t.accent_fill() } else { t.bg_active })
                        .child(
                            text(ty::SM, if active { t.accent_on } else { t.text_tertiary })
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(n.to_string()),
                        ),
                )
                // Shrinks to its third of the row, so the body (one text run) wraps inside it.
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(
                            text(ty::BASE, if active { t.text_primary } else { t.text_secondary })
                                .font_weight(FontWeight::MEDIUM)
                                .child(title.to_string()),
                        )
                        .child(rich(&t, ty::SM, t.text_tertiary, body).text_left()),
                )
        };
        let steps = panel(&t)
            .p(px(16.))
            .flex()
            .flex_col()
            .gap(px(20.))
            .child(
                div().flex().items_center().gap(px(14.)).child(crate::app::logo(&t, 32.)).child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(text(ty::LG, t.text_primary).child("Get started with FOLO"))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .child("Three steps from images to a model you can run."),
                        ),
                ),
            )
            .child(
                div()
                    // Children stretch to the row's height (the flex default).
                    .flex()
                    .gap(px(12.))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "1",
                        "Add a dataset",
                        "A YOLO or COCO folder.",
                        true,
                    )))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "2",
                        "Train a model",
                        "Watch loss and mAP live.",
                        false,
                    )))
                    .child(div().flex().flex_1().flex_basis(px(0.)).min_w_0().child(step(
                        "3",
                        "Try it",
                        "Open any image in the Playground.",
                        false,
                    ))),
            )
            .child(div().flex().gap(px(8.)).child(add).child(synth));
        // The same path from a terminal, for people who live there (docs/CLI.md).
        const CMDS: [&str; 3] =
            ["folo serve", "folo dataset synth shapes --images 1000", "folo train --data shapes --epochs 50"];
        let mut copy: Vec<Option<AnyElement>> = CMDS
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                Some(
                    self.icon_button(
                        &format!("gs-copy-{i}"),
                        "icons/copy.svg",
                        "Copy command",
                        window,
                        cx,
                        act(move |this, _, cx| {
                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(c.to_string()));
                            this.toast(Tone::Neutral, "Copied", Some(format!("`{c}`").into()), cx);
                        }),
                    )
                    .into_any_element(),
                )
            })
            .collect();
        let mut cmd = |i: usize, what: &str| {
            div()
                .flex()
                .items_center()
                .gap(px(16.))
                .child(
                    div()
                        .w(px(420.))
                        .flex_none()
                        .flex()
                        .items_center()
                        .gap(px(4.))
                        .pl(px(10.))
                        .pr(px(2.))
                        .py(px(2.))
                        .rounded(px(radius::SM))
                        .bg(t.bg_inset)
                        .child(
                            text(ty::SM, t.text_primary)
                                .font_family(crate::theme::MONO)
                                .flex_1()
                                .min_w_0()
                                .ellipsis()
                                .child(CMDS[i]),
                        )
                        .children(copy[i].take()),
                )
                .child(text(ty::SM, t.text_tertiary).child(what.to_string()))
        };
        let terminal = panel(&t)
            .p(px(16.))
            .flex()
            .flex_col()
            .gap(px(10.))
            .child(text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child("Or from a terminal"))
            // Honest about what reaches this app: the server and the datasets registered with it;
            // `folo train` runs on its own (runs/ in the working folder).
            .child(cmd(0, "start it, then Connect in Settings"))
            .child(cmd(1, "then Add dataset → choose ./shapes"))
            // Training from the terminal stands apart from the sequence above.
            .child(div().pt(px(6.)).child(caption(&t, "Independent CLI training")))
            .child(cmd(2, "trains here in the terminal; this run won't appear in Desktop"));
        // Capped like Settings, so a wide window doesn't stretch three short steps and the
        // commands' captions across 1,600 px of panel.
        div().flex().flex_col().gap(px(16.)).max_w(px(1040.)).child(steps).child(terminal).into_any_element()
    }

    /// Dataset card: a 2×2 mosaic of its first images, name and counts.
    pub fn dataset_card(
        &mut self,
        d: &crate::api::Dataset,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        self.ensure_dataset(&d.id, cx);
        let metas: Vec<crate::api::ImageInfo> = self
            .store
            .images
            .get(&d.id)
            .and_then(|l| l.ready())
            .map(|v| v.iter().filter(|m| !m.boxes.is_empty()).take(4).cloned().collect())
            .unwrap_or_default();
        let tiles: Vec<AnyElement> = (0..4)
            .map(|i| match metas.get(i) {
                Some(m) => {
                    // Outlines only: the card previews the data, its page names every box.
                    let tile = self.tile(&d.id, m, true, false, crate::viz::Fit::Cover, false, cx);
                    div().flex_1().h_full().overflow_hidden().child(tile).into_any_element()
                }
                None => div().flex_1().h_full().bg(t.bg_raised).into_any_element(),
            })
            .collect();
        let mut it = tiles.into_iter();
        let row1: Vec<AnyElement> = it.by_ref().take(2).collect();
        let row2: Vec<AnyElement> = it.collect();
        let id = d.id.clone();
        let key = format!("ds-card-{}", d.id);
        // The card in full: its name, size and every class, on hover and on keyboard focus.
        let full = format!("{} · {} images · {}", d.name, int(d.splits.total()), d.classes.join(", "));
        let focused = self.ring(&self.fh(key.clone(), cx), window);
        let card = div()
            .id(SharedString::from(key.clone()))
            .tooltip(crate::controls::tip(full.clone(), None))
            // Under the card (its mosaic and footer are about 196 px tall), kept inside the window.
            .when(focused, |d| d.child(crate::controls::focus_tip(&t, full.clone(), 196.)))
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .rounded(px(radius::LG))
            .bg(t.bg_panel)
            .border_1()
            .border_color(t.line_default)
            .overflow_hidden()
            .cursor_pointer()
            .hover(|s| s.border_color(t.line_strong))
            .active(|s| s.bg(crate::theme::over(t.bg_panel, t.bg_active)))
            .child(
                div()
                    .h(px(136.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .bg(t.bg_inset)
                    .child(div().flex_1().flex().gap(px(1.)).children(row1))
                    .child(div().flex_1().flex().gap(px(1.)).children(row2)),
            )
            .child(
                div()
                    .p(px(12.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(cell2(
                        &t,
                        d.name.clone(),
                        // Counts that never need cutting; the class names are the card's tip (hover
                        // and keyboard focus), in full.
                        format!(
                            "{} images · {}",
                            int(d.splits.total()),
                            if d.classes.len() == 1 {
                                "1 class".to_string()
                            } else {
                                format!("{} classes", d.classes.len())
                            }
                        ),
                    ))
                    .child(badge(&t, d.format.label(), Tone::Neutral)),
            );
        self.bind(
            card,
            &key,
            radius::LG,
            window,
            cx,
            act(move |this, w, cx| this.navigate(Route::Dataset(id.clone()), w, cx)),
        )
        .into_any_element()
    }

    pub fn view_all(&self, key: &str, route: Route, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .flex()
            .items_center()
            .gap(px(4.))
            .px(px(6.))
            .mr(px(-6.))
            .h(px(22.))
            .rounded(px(radius::SM))
            .cursor_pointer()
            .hover(|s| s.bg(t.bg_hover))
            .active(|s| s.bg(t.bg_active))
            .child(text(ty::SM, t.text_secondary).child("View all"))
            .child(icon("icons/chevron-right.svg", 12., t.text_tertiary));
        self.bind(
            el,
            key,
            radius::SM,
            window,
            cx,
            act(move |this, w, cx| this.navigate(route.clone(), w, cx)),
        )
        .into_any_element()
    }
}

/// The Overview hero chart's plot height: lower in short windows (620 px minimum, so two Recent
/// runs rows show under it), taller in tall, wide ones so the curve isn't a flat band across 1,100
/// px. The skeleton and the live card share it.
fn hero_height(window: &Window) -> f32 {
    let v = window.viewport_size();
    if v.height < px(700.) {
        108.
    } else if v.width >= px(1600.) && v.height >= px(960.) {
        180.
    } else {
        132.
    }
}
