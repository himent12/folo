//! Playground side rail extensions: Compare A/B Model B selector, export action bar, and
//! concise A vs B delta summary (detection count delta, latency A vs B, shared/unique classes,
//! and per-model detection tabs).

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::compare::{CompareSideTab, format_latency_comparison, format_signed_delta, summarize_comparison};
use crate::api::{Model, PredictResponse};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::{loading_rows, th};
use crate::store::Load;
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

impl FoloApp {
    /// Model B picker and Swap A/B button shown inside the Playground side rail when Compare mode is active.
    pub(super) fn play_model_b_controls(
        &mut self,
        t: &Theme,
        models: &[Model],
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        self.ensure_play_model_b();
        let model_b = self.play_shown_model_b().cloned();
        let sel_b = model_b.as_ref().and_then(|m| models.iter().position(|x| x.id == m.id)).unwrap_or(0);
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        let select_b = self.select(
            "play-model-b",
            model_b.as_ref().map_or("No models".into(), |m| SharedString::from(m.name.clone())),
            models
                .iter()
                .map(|m| {
                    (
                        SharedString::from(m.name.clone()),
                        Some(SharedString::from(format!(
                            "{}{}{}",
                            m.arch.as_str(),
                            if self.model_e2e(m) { " · end-to-end" } else { " · NMS" },
                            m.metrics.as_ref().map_or(String::new(), |x| format!(" · {:.3}", x.map50_95)),
                        ))),
                    )
                })
                .collect(),
            sel_b,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(id) = ids.get(i) {
                    this.pick_play_model_b(id.clone(), cx);
                }
            }),
        );
        let same_model = self.play_model.is_some() && self.play_model == self.play_compare.model_b;
        let swap = if same_model {
            self.button_disabled(
                "play-swap-ab",
                "Swap A/B",
                Btn::Secondary,
                Some("icons/git-compare-arrows.svg"),
                "Pick two different models to swap",
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "play-swap-ab",
                "Swap A/B",
                Btn::Secondary,
                Some("icons/git-compare-arrows.svg"),
                window,
                cx,
                act(|this, _, cx| this.swap_play_models(cx)),
            )
            .into_any_element()
        };

        div()
            .flex()
            .flex_col()
            .gap(px(6.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(8.))
                    .child(div().flex().items_center().gap(px(6.)).child(badge(t, "B", Tone::Info)).child(
                        text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Model B"),
                    ))
                    .child(swap),
            )
            .child(select_b)
            .when_some(model_b.as_ref(), |d, m| {
                d.child(text(ty::SM, t.text_tertiary).child(format!(
                    "{} · {} px{}",
                    m.arch.as_str(),
                    m.imgsz,
                    model_storage(m).map_or(String::new(), |s| format!(" · {s}")),
                )))
            })
            .when(same_model && models.len() <= 1, |d| {
                d.child(
                    text(ty::SM, t.text_tertiary)
                        .whitespace_normal()
                        .child("Only one model in this workspace. Train another to compare."),
                )
            })
            .into_any_element()
    }

    /// Export action bar (Export PNG, Export JSON, Copy JSON) shared between Single and Compare modes.
    pub(super) fn play_export_bar(&self, t: &Theme, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let has_image = self.play_image.is_some();
        let has_json = self.has_play_export_json();
        let png_btn = if has_image {
            self.button(
                "play-export-png",
                "Export PNG",
                Btn::Secondary,
                Some("icons/download.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_png(cx)),
            )
            .tooltip(crate::controls::tip("Save annotated image as PNG", None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-export-png",
                "Export PNG",
                Btn::Secondary,
                Some("icons/download.svg"),
                "Open an image first",
                window,
                cx,
            )
            .into_any_element()
        };
        let json_btn = if has_json {
            self.button(
                "play-export-json",
                "Export JSON",
                Btn::Secondary,
                Some("icons/file-text.svg"),
                window,
                cx,
                act(|this, _, cx| this.export_play_json(cx)),
            )
            .tooltip(crate::controls::tip("Save detections payload as formatted JSON", None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-export-json",
                "Export JSON",
                Btn::Secondary,
                Some("icons/file-text.svg"),
                "Run a prediction first",
                window,
                cx,
            )
            .into_any_element()
        };
        let copy_btn = if has_json {
            self.button(
                "play-copy-json",
                "Copy JSON",
                Btn::Secondary,
                Some("icons/copy.svg"),
                window,
                cx,
                act(|this, _, cx| this.copy_play_json(cx)),
            )
            .tooltip(crate::controls::tip("Copy visible detections as JSON", None))
            .into_any_element()
        } else {
            self.button_disabled(
                "play-copy-json",
                "Copy JSON",
                Btn::Secondary,
                Some("icons/copy.svg"),
                "Run a prediction first",
                window,
                cx,
            )
            .into_any_element()
        };

        div()
            .px(px(12.))
            .py(px(8.))
            .border_b_1()
            .border_color(t.line_subtle)
            .flex()
            .flex_wrap()
            .items_center()
            .gap(px(6.))
            .child(png_btn)
            .child(json_btn)
            .child(copy_btn)
            .into_any_element()
    }

    /// Compare A/B results section in the Playground side rail: Boxes/Labels toggles, export actions,
    /// `Summary | Model A | Model B` switcher, delta metrics, shared/unique class breakdown, and
    /// per-side detection lists.
    pub(super) fn play_compare_results(
        &mut self,
        t: &Theme,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        if self.play_image.is_none() {
            return div().into_any_element();
        }
        let boxes_chip = self.chip(
            "play-boxes",
            "Boxes",
            self.play_boxes,
            None,
            None,
            window,
            cx,
            act(|this, _, cx| {
                this.play_boxes = !this.play_boxes;
                if !this.play_boxes {
                    this.clear_play_hover();
                    this.play_compare.clear_hover();
                }
                cx.notify();
            }),
        );
        let labels_chip = self.chip(
            "play-labels",
            "Labels",
            self.play_boxes && self.play_labels,
            None,
            None,
            window,
            cx,
            act(|this, _, cx| {
                if !this.play_boxes {
                    this.play_boxes = true;
                    this.play_labels = true;
                } else {
                    this.play_labels = !this.play_labels;
                }
                cx.notify();
            }),
        );

        let pred_a = self.play_pred.ready().cloned();
        let pred_b = self.play_compare.pred_b.ready().cloned();
        let vis_a: Option<Vec<crate::api::ScoredBox>> = pred_a.as_ref().map(|p| {
            super::page::filter_detections(&p.boxes, self.play_conf, self.play_class)
                .into_iter()
                .cloned()
                .collect()
        });
        let vis_b: Option<Vec<crate::api::ScoredBox>> =
            pred_b.as_ref().map(|p| self.filter_detections_b(&p.boxes).into_iter().cloned().collect());
        let refs_a: Option<Vec<&crate::api::ScoredBox>> = vis_a.as_ref().map(|v| v.iter().collect());
        let refs_b: Option<Vec<&crate::api::ScoredBox>> = vis_b.as_ref().map(|v| v.iter().collect());
        let summary =
            summarize_comparison(refs_a.as_deref(), refs_b.as_deref(), pred_a.as_ref(), pred_b.as_ref());

        let tab_idx = match self.play_compare.side_tab {
            CompareSideTab::Summary => 0,
            CompareSideTab::ModelA => 1,
            CompareSideTab::ModelB => 2,
        };
        let tabs = self.segmented_fill(
            "play-compare-tab",
            &[
                "Delta",
                &format!("A ({})", summary.count_a.map_or("—".to_string(), |n| n.to_string())),
                &format!("B ({})", summary.count_b.map_or("—".to_string(), |n| n.to_string())),
            ],
            tab_idx,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.play_compare.side_tab = match i {
                    1 => CompareSideTab::ModelA,
                    2 => CompareSideTab::ModelB,
                    _ => CompareSideTab::Summary,
                };
                cx.notify();
            }),
        );

        let body = match self.play_compare.side_tab {
            CompareSideTab::Summary => self.compare_summary_body(t, &summary, window, cx),
            CompareSideTab::ModelA => {
                self.compare_side_list(t, false, &self.play_pred.clone(), vis_a, window, cx)
            }
            CompareSideTab::ModelB => {
                self.compare_side_list(t, true, &self.play_compare.pred_b.clone(), vis_b, window, cx)
            }
        };

        let export_bar = self.play_export_bar(t, window, cx);
        div()
            .flex()
            .flex_col()
            .child(panel_header(t, "Compare A/B").h(px(32.)).child(
                text(ty::SM, t.text_tertiary).tnum().child(
                    if self.play_pred_pending || self.play_compare.pred_b_pending {
                        "Updating…".to_string()
                    } else {
                        format!("Δ {}", format_signed_delta(summary.count_delta))
                    },
                ),
            ))
            .child(
                div()
                    .px(px(12.))
                    .py(px(6.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(boxes_chip)
                    .child(labels_chip),
            )
            .child(export_bar)
            .child(div().px(px(12.)).py(px(8.)).child(tabs))
            .child(body)
            .into_any_element()
    }

    fn compare_summary_body(
        &mut self,
        t: &Theme,
        summary: &super::compare::CompareSummary,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let count_a = summary.count_a.map_or("—".to_string(), |n| n.to_string());
        let count_b = summary.count_b.map_or("—".to_string(), |n| n.to_string());
        let delta_str = format_signed_delta(summary.count_delta);
        let delta_tone = match summary.count_delta {
            Some(d) if d > 0 => Tone::Success,
            Some(d) if d < 0 => Tone::Warning,
            _ => Tone::Neutral,
        };
        let lat_a = summary.latency_a_ms.map_or("—".to_string(), |v| {
            if summary.cold_a == Some(true) { format!("{v:.1} ms (cold)") } else { format!("{v:.1} ms") }
        });
        let lat_b = summary.latency_b_ms.map_or("—".to_string(), |v| {
            if summary.cold_b == Some(true) { format!("{v:.1} ms (cold)") } else { format!("{v:.1} ms") }
        });
        let lat_cmp = format_latency_comparison(summary);
        let lat_cmp_color = if summary.cold_a == Some(true) || summary.cold_b == Some(true) {
            t.warning_text
        } else {
            t.text_secondary
        };

        let class_rows: Vec<AnyElement> = summary
            .classes
            .iter()
            .map(|row| {
                let d_str = format_signed_delta(row.delta);
                let d_color = match row.delta {
                    Some(d) if d > 0 => t.success_text,
                    Some(d) if d < 0 => t.warning_text,
                    _ => t.text_tertiary,
                };
                div()
                    .h(px(26.))
                    .px(px(8.))
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(class_swatch(t, row.color_class, 8.))
                    .child(text(ty::SM, t.text_primary).flex_1().min_w_0().ellipsis().child(row.name.clone()))
                    .child(
                        text(ty::SM, t.text_secondary)
                            .tnum()
                            .w(px(28.))
                            .text_right()
                            .child(row.count_a.map_or("—".to_string(), |n| n.to_string())),
                    )
                    .child(
                        text(ty::SM, t.text_secondary)
                            .tnum()
                            .w(px(28.))
                            .text_right()
                            .child(row.count_b.map_or("—".to_string(), |n| n.to_string())),
                    )
                    .child(text(ty::SM, d_color).tnum().w(px(32.)).text_right().child(d_str))
                    .into_any_element()
            })
            .collect();

        let err_a = self.play_pred.error().cloned();
        let err_b = self.play_compare.pred_b.error().cloned();
        let retry_a = err_a.map(|e| {
            let btn = self.button(
                "play-cmp-retry-a",
                "Retry A",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(|this, _, cx| this.rerun_predict(cx)),
            );
            div()
                .p(px(10.))
                .rounded(px(radius::MD))
                .bg(t.danger_muted)
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(
                    text(ty::SM, t.danger_text).whitespace_normal().child(format!("Model A: {}", e.message)),
                )
                .child(btn)
        });
        let retry_b = err_b.map(|e| {
            let btn = self.button(
                "play-cmp-retry-b",
                "Retry B",
                Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(|this, _, cx| this.rerun_predict_b(cx)),
            );
            div()
                .p(px(10.))
                .rounded(px(radius::MD))
                .bg(t.danger_muted)
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(
                    text(ty::SM, t.danger_text).whitespace_normal().child(format!("Model B: {}", e.message)),
                )
                .child(btn)
        });

        div()
            .px(px(12.))
            .pb(px(12.))
            .flex()
            .flex_col()
            .gap(px(10.))
            .children(retry_a)
            .children(retry_b)
            .child(
                div()
                    .p(px(10.))
                    .rounded(px(radius::MD))
                    .bg(t.bg_inset)
                    .border_1()
                    .border_color(t.line_subtle)
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Detections"),
                            )
                            .child(badge(t, format!("Δ {delta_str}"), delta_tone)),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(text(ty::SM, t.text_primary).tnum().child(format!("A: {count_a}")))
                            .child(text(ty::SM, t.text_primary).tnum().child(format!("B: {count_b}"))),
                    )
                    .child(div().h(px(1.)).bg(t.line_subtle))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Latency"),
                            )
                            .child(text(ty::SM, lat_cmp_color).tnum().child(lat_cmp)),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("A: {lat_a}")))
                            .child(text(ty::SM, t.text_tertiary).tnum().child(format!("B: {lat_b}"))),
                    ),
            )
            .when(summary.count_a.is_some() && summary.count_b.is_some(), |d| {
                d.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap(px(6.))
                        .child(badge(t, format!("Shared {}", summary.shared_classes.len()), Tone::Neutral))
                        .child(badge(t, format!("A only {}", summary.only_a_classes.len()), Tone::Accent))
                        .child(badge(t, format!("B only {}", summary.only_b_classes.len()), Tone::Info)),
                )
            })
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .when(!class_rows.is_empty(), |d| {
                        d.child(
                            div()
                                .h(px(20.))
                                .px(px(8.))
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(th(t, "Class").flex_1())
                                .child(th(t, "A").w(px(28.)).text_right())
                                .child(th(t, "B").w(px(28.)).text_right())
                                .child(th(t, "Δ").w(px(32.)).text_right()),
                        )
                    })
                    .children(class_rows)
                    .when(summary.classes.is_empty(), |d| {
                        let msg = if self.play_pred.is_loading() || self.play_compare.pred_b.is_loading() {
                            "Running both models…"
                        } else {
                            "No detections above the confidence threshold on either model."
                        };
                        d.child(text(ty::SM, t.text_tertiary).whitespace_normal().px(px(4.)).child(msg))
                    }),
            )
            .into_any_element()
    }

    fn compare_side_list(
        &mut self,
        t: &Theme,
        is_b: bool,
        pred: &Load<PredictResponse>,
        vis: Option<Vec<crate::api::ScoredBox>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        match pred {
            Load::Loading => loading_rows(t, if is_b { "detb" } else { "deta" }, 4).into_any_element(),
            Load::Failed(e) => {
                let retry = self.button(
                    if is_b { "play-retry-b" } else { "play-retry" },
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        if is_b {
                            this.rerun_predict_b(cx);
                        } else {
                            this.rerun_predict(cx);
                        }
                    }),
                );
                div()
                    .p(px(12.))
                    .flex()
                    .flex_col()
                    .items_start()
                    .gap(px(8.))
                    .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message.clone()))
                    .when_some(e.hint.clone(), |d, h| {
                        d.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(h))
                    })
                    .child(retry)
                    .into_any_element()
            }
            Load::Idle => div().into_any_element(),
            Load::Ready(_) => {
                let vis = vis.unwrap_or_default();
                let active = if is_b { self.play_active_b(window, cx) } else { self.play_active(window, cx) };
                let prefix = if is_b { "det-b" } else { "det" };
                let n_vis = vis.len();
                let rows: Vec<AnyElement> = vis
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let hot = active == Some(i);
                        let key = format!("{prefix}-{i}");
                        let score = if b.score.is_finite() { b.score.clamp(0.0, 1.0) } else { 0.0 };
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .flex_none()
                            .h(px(28.))
                            .px(px(8.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .rounded(px(radius::SM))
                            .when(hot, |d| d.bg(t.bg_active).child(select_bar(t)))
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .on_key_down(cx.listener(move |this, ev: &folo_ui::KeyDownEvent, w, cx| {
                                let m = &ev.keystroke.modifiers;
                                if m.control || m.alt || m.platform {
                                    return;
                                }
                                let next = match ev.keystroke.key.as_str() {
                                    "down" => (i + 1 < n_vis).then_some(i + 1),
                                    "up" => i.checked_sub(1),
                                    "home" => (i > 0).then_some(0),
                                    "end" => (i + 1 < n_vis).then_some(n_vis - 1),
                                    _ => return,
                                };
                                if let Some(next) = next {
                                    this.focus_key_and_reveal(&format!("{prefix}-{next}"), w, cx);
                                }
                                cx.stop_propagation();
                            }))
                            .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                                if is_b {
                                    if *h {
                                        this.play_compare.hover_b_from_stage = false;
                                        if this.play_compare.hover_b != Some(i) {
                                            this.play_compare.hover_b = Some(i);
                                            cx.notify();
                                        }
                                    } else if !this.play_compare.hover_b_from_stage
                                        && this.play_compare.hover_b == Some(i)
                                    {
                                        this.play_compare.clear_hover();
                                        cx.notify();
                                    }
                                } else if *h {
                                    this.play_hover_from_stage = false;
                                    if this.play_hover != Some(i) {
                                        this.play_hover = Some(i);
                                        cx.notify();
                                    }
                                } else if !this.play_hover_from_stage && this.play_hover == Some(i) {
                                    this.clear_play_hover();
                                    cx.notify();
                                }
                            }))
                            .child(class_swatch(t, b.class, 8.))
                            .child(text(ty::SM, t.text_primary).w(px(68.)).ellipsis().child(b.name.clone()))
                            .child(
                                text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                                    .tnum()
                                    .w(px(64.))
                                    .flex_none()
                                    .child(format!(
                                        "{:.0}×{:.0}",
                                        (b.x2 - b.x1).max(0.0),
                                        (b.y2 - b.y1).max(0.0)
                                    )),
                            )
                            .child(
                                div().flex_1().h(px(4.)).rounded_full().bg(t.bg_inset).child(
                                    div()
                                        .h_full()
                                        .w(relative(score))
                                        .rounded_full()
                                        .bg(t.class_on(b.class, crate::theme::Surface::Panel)),
                                ),
                            )
                            .child(
                                text(ty::SM, t.text_primary)
                                    .tnum()
                                    .w(px(36.))
                                    .text_right()
                                    .child(format!("{score:.2}")),
                            );
                        self.bind(
                            row,
                            &key,
                            radius::SM,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                if is_b {
                                    this.play_compare.hover_b_from_stage = false;
                                    this.play_compare.hover_b =
                                        if this.play_compare.hover_b == Some(i) { None } else { Some(i) };
                                } else {
                                    this.play_hover_from_stage = false;
                                    this.play_hover = if this.play_hover == Some(i) { None } else { Some(i) };
                                }
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();

                div()
                    .px(px(8.))
                    .pb(px(12.))
                    .flex()
                    .flex_col()
                    .gap(px(1.))
                    .when(!vis.is_empty(), |d| {
                        d.child(
                            div()
                                .h(px(20.))
                                .px(px(8.))
                                .flex()
                                .items_center()
                                .gap(px(8.))
                                .child(th(t, "Class").w(px(84.)).flex_none())
                                .child(th(t, "Size (px)").w(px(64.)).flex_none())
                                .child(th(t, "Confidence").flex_1().text_right()),
                        )
                    })
                    .children(rows)
                    .when(vis.is_empty(), |d| {
                        d.child(
                            text(ty::SM, t.text_tertiary)
                                .whitespace_normal()
                                .p(px(8.))
                                .child("Nothing above the confidence threshold."),
                        )
                    })
                    .into_any_element()
            }
        }
    }
}
