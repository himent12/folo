//! The panel shown while nothing trains, and the machine at a glance.

use gpui::prelude::FluentBuilder;
use gpui::{AnyElement, Context, IntoElement, ParentElement, Styled, Window, div, px};

use super::page::hero_height;
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    pub(super) fn idle_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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

    pub(super) fn system_strip(&mut self, cx: &mut Context<Self>) -> AnyElement {
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
}
