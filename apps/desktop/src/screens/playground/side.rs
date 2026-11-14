//! The side panel: the model, thresholds and the detections list.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use super::device::device_name;
use crate::api::Split;
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::screens::{image_canvas, loading_rows, th};
use crate::store::{ImgKind, Load};
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::Fit;

impl FoloApp {
    pub(super) fn play_side(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned().unwrap_or_default();
        // The model Playground runs, and no other in its place (a deleted one is replaced, and that
        // said, as soon as the list shows it gone).
        let model = self.play_shown_model().cloned();
        let sel = model.as_ref().and_then(|m| models.iter().position(|x| x.id == m.id)).unwrap_or(0);
        // An end-to-end model (the run that trained it had the one-to-one head) predicts without NMS,
        // so its IoU setting does nothing.
        let e2e = model.as_ref().is_some_and(|m| self.model_e2e(m));
        let best_conf = model.as_ref().and_then(|m| m.metrics.as_ref()).and_then(offered_conf);
        // The built-in demo's numbers are simulated: its timing says so.
        let simulated = self.play_target.is_none() && !self.store.api.local_inference();
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        let select = self.select(
            "play-model",
            // A failed list isn't an empty one.
            model.as_ref().map_or(
                if self.store.models.error().is_some() {
                    "Models didn't load".into()
                } else {
                    "No models".into()
                },
                |m| SharedString::from(m.name.clone()),
            ),
            models
                .iter()
                .map(|m| {
                    (
                        SharedString::from(m.name.clone()),
                        // Architecture and head always (end-to-end is why the NMS row changes);
                        // mAP when the model has been evaluated.
                        Some(SharedString::from(format!(
                            "{}{}{}",
                            m.arch.as_str(),
                            // Both heads named: each says how its boxes are made.
                            if self.model_e2e(m) { " · end-to-end" } else { " · NMS" },
                            m.metrics.as_ref().map_or(String::new(), |x| format!(" · {:.3}", x.map50_95)),
                        ))),
                    )
                })
                .collect(),
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(id) = ids.get(i) {
                    this.pick_play_model(id.clone(), cx);
                }
            }),
        );
        // The range widens to an applied threshold below the usual floor (a best-F1 confidence).
        let conf = self.slider(
            "play-conf",
            self.play_conf,
            crate::app::PLAY_CONF_FLOOR.min(self.play_conf),
            0.95,
            0.01,
            window,
            cx,
        );
        let iou = self.slider("play-iou", self.play_iou, 0.3, 0.95, 0.01, window, cx);
        let conf_help = self.help_icon(
            "help-conf",
            "Hide detections scoring below this. Filtering is instant; it doesn't rerun the model.",
            window,
            cx,
        );
        let iou_help = self.help_icon("help-iou", "Non-max suppression drops a weaker box that overlaps a stronger one by more than this. Changing it reruns the model.", window, cx);
        // Short windows: a tighter control block, so more detection rows show without scrolling.
        let tight = window.viewport_size().height < px(700.);
        // Sliders on one line ("label · slider · value") in short windows, and whenever both show,
        // so the latency breakdown fits at 1280 x 800.
        let inline = true;
        let field = |label: &str, value: String, control: AnyElement, help: AnyElement| {
            let name = div()
                .flex()
                .items_center()
                .gap(px(4.))
                .child(
                    text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child(label.to_string()),
                )
                .child(help);
            let value = text(ty::SM, t.text_primary).tnum().child(value);
            if inline {
                // One row each: name, slider, value.
                div()
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .child(name.w(px(92.)).flex_none())
                    .child(div().flex_1().min_w_0().child(control))
                    .child(div().w(px(32.)).flex_none().flex().justify_end().child(value))
            } else {
                div()
                    .flex()
                    .flex_col()
                    .gap(px(6.))
                    .child(div().flex().justify_between().items_center().child(name).child(value))
                    .child(control)
            }
        };
        let no_models = models.is_empty() && self.store.models.ready().is_some();
        let run_on = self.run_on(window, cx);
        let controls = div()
            .flex()
            .flex_col()
            // 12 between the controls: with the 12 px picker row, the rail fits whole at 1280 × 800.
            .gap(px(if tight { 8. } else { 12. }))
            .px(px(16.))
            .py(px(if tight { 10. } else { 16. }))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Model"))
                    .map(|d| {
                        if no_models {
                            // Nothing to choose: a quiet, inert field instead of a dropdown.
                            d.child(
                                div()
                                    .h(px(32.))
                                    .px(px(10.))
                                    .flex()
                                    .items_center()
                                    .rounded(px(radius::MD))
                                    .bg(t.bg_inset)
                                    .border_1()
                                    .border_color(t.line_subtle)
                                    .child(text(ty::SM, t.text_disabled).child("No trained models")),
                            )
                        } else {
                            d.child(select)
                        }
                    })
                    // What's running, without opening the menu: architecture, input size and head.
                    .when_some(model.as_ref(), |d, m| {
                        // Identity and storage; the head is the NMS row's to say (once).
                        d.child(text(ty::SM, t.text_tertiary).child(format!(
                            "{} · {} px{}",
                            m.arch.as_str(),
                            m.imgsz,
                            model_storage(m).map_or(String::new(), |s| format!(" · {s}")),
                        )))
                    })
                    .when(no_models, |d| {
                        d.child(
                            text(ty::SM, t.text_tertiary)
                                .child("No models yet. Finished runs add their best weights here."),
                        )
                    }),
            )
            .when(!no_models, |d| {
                d.child(run_on)
                    .child(field("Confidence", fmt_conf(self.play_conf), conf.into_any_element(), conf_help))
                    // The model's best-F1 threshold (where its precision and recall were taken), one
                    // click away while the slider is elsewhere.
                    // A fixed row, so using it moves nothing: a secondary link while the slider is
                    // elsewhere, a quiet note once it's there (accent stays the slider's).
                    // Always there for a picked model (empty without a best-F1 threshold), so picking
                    // one doesn't push what is under it down.
                    .when(best_conf.is_none() && model.is_some(), |d| d.child(div().h(px(20.))))
                    .when_some(best_conf, |d, c| {
                        let at = (c - self.play_conf).abs() < 0.0005;
                        let row = div().h(px(20.)).flex().justify_end().items_center();
                        if at {
                            return d.child(row.child(
                                text(ty::SM, t.text_tertiary).child(format!("At best F1 ({})", fmt_conf(c))),
                            ));
                        }
                        let link = div()
                            .id("play-best-conf")
                            .px(px(4.))
                            .rounded(px(radius::SM))
                            .cursor_pointer()
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            // The shared inline-link look; -4 px so its text ends on the value above.
                            .mr(px(-4.))
                            .child(inline_link(&t, format!("Use {} (best F1)", fmt_conf(c))));
                        d.child(row.child(self.bind(
                            link,
                            "play-best-conf",
                            radius::SM,
                            window,
                            cx,
                            act(move |this, w, cx| {
                                // Below what's been fetched: set_play_conf fetches down to it.
                                this.set_play_conf(c, cx);
                                // The link becomes a note: focus moves to the slider it set.
                                this.focus_key("play-conf", w, cx);
                                cx.notify();
                            }),
                        )))
                    })
                    // "NMS IoU": the suppression overlap, not the evaluation's IoU 0.50. An end-to-end
                    // model has no suppression step: the row says so instead of offering an inert slider.
                    .map(|d| {
                        if e2e {
                            d.child(
                                div()
                                    .h(px(20.))
                                    .flex()
                                    .justify_between()
                                    .items_center()
                                    .gap(px(10.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("NMS"),
                                    )
                                    .child(
                                        text(ty::SM, t.text_tertiary).child("Not used · end-to-end model"),
                                    ),
                            )
                        } else {
                            d.child(field(
                                "NMS IoU",
                                format!("{:.2}", self.play_iou),
                                iou.into_any_element(),
                                iou_help,
                            ))
                        }
                    })
            });

        let active = self.play_active(window, cx);
        let mut timing: Option<AnyElement> = None;
        let detections = match &self.play_pred {
            Load::Ready(p) => {
                let vis: Vec<crate::api::ScoredBox> =
                    p.boxes.iter().filter(|b| b.score >= self.play_conf).cloned().collect();
                let rows: Vec<AnyElement> = vis
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let hot = active == Some(i);
                        let key = format!("det-{i}");
                        // A cut class name shows in full on hover and on keyboard focus.
                        let long = may_overflow(&b.name, ty::SM.size, 68.);
                        let row = self
                            .disclose(
                                div().id(SharedString::from(key.clone())),
                                &key,
                                b.name.clone(),
                                long,
                                28.,
                                false,
                                window,
                                cx,
                            )
                            .flex_none()
                            .h(px(28.))
                            .px(px(8.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .rounded(px(radius::SM))
                            .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                            .hover(|s| s.bg(t.bg_hover))
                            .active(|s| s.bg(t.bg_active))
                            .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                                let next = if *h { Some(i) } else { None };
                                if this.play_hover != next {
                                    this.play_hover = next;
                                    cx.notify();
                                }
                            }))
                            .child(crate::ui::class_swatch(&t, b.class, 8.))
                            .child(text(ty::SM, t.text_primary).w(px(68.)).ellipsis().child(b.name.clone()))
                            // Size tells apart two boxes of the same class.
                            .child(
                                // On the active row's fill, tertiary falls under AA: secondary there.
                                text(ty::SM, if hot { t.text_secondary } else { t.text_tertiary })
                                    .tnum()
                                    .w(px(68.))
                                    .flex_none()
                                    .child(format!("{:.0}×{:.0}", b.x2 - b.x1, b.y2 - b.y1)),
                            )
                            .child(
                                div().flex_1().h(px(4.)).rounded_full().bg(t.bg_inset).child(
                                    // The class's surface-safe stroke (the photo palette washes out on a light panel).
                                    div()
                                        .h_full()
                                        .w(relative(b.score))
                                        .rounded_full()
                                        .bg(t.class_on(b.class, crate::theme::Surface::Panel)),
                                ),
                            )
                            .child(
                                text(ty::SM, t.text_primary)
                                    .tnum()
                                    .w(px(36.))
                                    .text_right()
                                    .child(format!("{:.2}", b.score)),
                            );
                        // Focusable so the keyboard can inspect boxes too (focus = highlight).
                        self.bind(
                            row,
                            &key,
                            radius::SM,
                            window,
                            cx,
                            act(move |this, _, cx| {
                                this.play_hover = if this.play_hover == Some(i) { None } else { Some(i) };
                                cx.notify();
                            }),
                        )
                        .into_any_element()
                    })
                    .collect();
                let hidden = p.boxes.len() - vis.len();
                let total = p.timing.total_ms.max(0.001);
                let seg = |v: f64, c| div().h_full().w(relative((v / total) as f32)).bg(c);
                // One neutral ramp: Inference (the bulk) in secondary ink, Preprocess tertiary a step
                // below it (70 % dark, 80 % light: each 3:1 on its panel), and Postprocess an outline
                // (a faint fill in a tertiary edge), in the bar and its key alike.
                let (c_pre, c_model, c_nms) = (
                    crate::theme::alpha(t.text_tertiary, if t.is_dark() { 0.7 } else { 0.8 }),
                    t.text_secondary,
                    crate::theme::alpha(t.text_tertiary, 0.3),
                );
                let timing_el = div()
                    .w_full()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .px(px(16.))
                    .py(px(12.))
                    .border_t_1()
                    .border_color(t.line_subtle)
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .items_start()
                            .gap(px(8.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .gap(px(2.))
                                    // Where the number comes from: the built-in demo simulates it; a
                                    // server or this app times one image (not sustained throughput).
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(if simulated { "Simulated latency" } else { "Latency" }),
                                    )
                                    // Under the label, whole: beside it a local GPU's name left no room
                                    // ("RTX 4080 SUPER in this app · o…").
                                    .child(text(ty::SM, t.text_tertiary).tnum().whitespace_normal().child(
                                        // The demo has no device of its own to name.
                                        if simulated {
                                            "one image".to_string()
                                        } else {
                                            format!(
                                                "{} · one image",
                                                device_name(&p.device, self.store.system.ready()),
                                            )
                                        },
                                    )),
                            )
                            .child(metric(&t, format!("{:.1}", p.timing.total_ms), "ms", ty::LG, ty::SM)),
                    )
                    .child(
                        div()
                            .flex()
                            .h(px(6.))
                            .gap(px(2.))
                            .rounded_full()
                            .overflow_hidden()
                            .child(seg(p.timing.preprocess_ms, c_pre))
                            .child(seg(p.timing.inference_ms, c_model))
                            .child(
                                seg(p.timing.postprocess_ms, c_nms).border_1().border_color(t.text_tertiary),
                            ),
                    )
                    .child(
                        // One line in the 280 px rail: short stage names, the full name on hover. Whole
                        // entries still wrap if the numbers get wide (100 ms inference on a narrow rail).
                        div().flex().flex_wrap().gap_x(px(12.)).gap_y(px(4.)).children(
                            [
                                ("Pre", "Preprocess", p.timing.preprocess_ms, c_pre),
                                ("Infer", "Inference", p.timing.inference_ms, c_model),
                                // The engine's postprocess: box decoding, and NMS unless the model
                                // is end-to-end (named as Studio names it).
                                (
                                    "Post",
                                    if e2e {
                                        "Postprocess · box decoding"
                                    } else {
                                        "Postprocess · box decoding and NMS"
                                    },
                                    p.timing.postprocess_ms,
                                    c_nms,
                                ),
                            ]
                            .into_iter()
                            .map(|(n, full, v, c)| {
                                div()
                                    .id(SharedString::from(format!("lat-{n}")))
                                    .tooltip(crate::controls::tip(full, None))
                                    .flex()
                                    .flex_none()
                                    .items_center()
                                    .gap(px(5.))
                                    .whitespace_nowrap()
                                    .child(
                                        swatch(c, 8.).when(n == "Post", |d| {
                                            d.border_1().border_color(t.text_tertiary)
                                        }),
                                    )
                                    .child(text(ty::SM, t.text_tertiary).child(n))
                                    .child(text(ty::SM, t.text_secondary).tnum().child(format!("{v:.1} ms")))
                            }),
                        ),
                    );
                timing = Some(timing_el.into_any_element());
                div()
                    .flex()
                    .flex_col()
                    .flex_none()
                    // While a rerun (an IoU change) is out, the rows are the previous answer: say so.
                    .child(panel_header(&t, "Detections").h(px(32.)).child(
                        text(ty::SM, t.text_tertiary).tnum().child(if self.play_pred_pending {
                            "Updating…".to_string()
                        // The total latency reads in the toolbar at every size; the count stays a count.
                        } else if hidden > 0 {
                            format!("{} shown · {hidden} below threshold", vis.len())
                        } else {
                            vis.len().to_string()
                        }),
                    ))
                    .child(
                        div()
                            // Another device's boxes while the picked one runs: dimmed.
                            .when(self.play_pred_dim && self.play_pred_pending, |d| d.opacity(0.5))
                            .px(px(8.))
                            .py(px(4.))
                            .flex()
                            .flex_col()
                            .gap(px(1.))
                            .when(!vis.is_empty(), |d| {
                                // Column labels, on the rows' grid: swatch + class, box size, confidence.
                                d.child(
                                    div()
                                        .h(px(20.))
                                        .px(px(8.))
                                        .flex()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(th(&t, "Class").w(px(84.)).flex_none())
                                        .child(th(&t, "Size (px)").w(px(68.)).flex_none())
                                        .child(th(&t, "Confidence").flex_1().text_right()),
                                )
                            })
                            .children(rows)
                            .when(vis.is_empty(), |d| {
                                d.child(div().p(px(10.)).child(text(ty::SM, t.text_tertiary).child(
                                "Nothing above the threshold. Lower the confidence to see weaker detections.",
                            )))
                            }),
                    )
                    .into_any_element()
            }
            Load::Failed(e) => {
                let retry = self.button(
                    "play-retry",
                    "Retry",
                    Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    // Show "Detecting…" again (the failure would otherwise stay up, and its
                    // button live, while the retry runs).
                    act(|this, _, cx| this.rerun_predict(cx)),
                );
                // Out of GPU memory locally: the CPU is one click away.
                let to_cpu = (e.code == "gpu_out_of_memory"
                    && self.play_target == Some(crate::local::Target::Gpu))
                .then(|| {
                    self.button(
                        "play-to-cpu",
                        "Run on Local CPU",
                        Btn::Secondary,
                        None,
                        window,
                        cx,
                        act(|this, _, cx| this.set_play_target(Some(crate::local::Target::Cpu), cx)),
                    )
                });
                div()
                    .p(px(16.))
                    .flex()
                    .flex_col()
                    .items_start()
                    .gap(px(6.))
                    .child(text(ty::SM, t.danger_text).whitespace_normal().child(e.message.clone()))
                    .when_some(e.hint.clone(), |d, h| {
                        d.child(text(ty::SM, t.text_tertiary).whitespace_normal().child(h))
                    })
                    // Retry (same image, model and thresholds), with the CPU beside it after an OOM.
                    .child(div().pt(px(4.)).flex().gap(px(8.)).child(retry).children(to_cpu))
                    .into_any_element()
            }
            Load::Loading => div()
                .flex_none()
                .child(
                    panel_header(&t, "Detections")
                        .h(px(32.))
                        .child(text(ty::SM, t.text_tertiary).child("Detecting…")),
                )
                .child(loading_rows(&t, "det", 4))
                .into_any_element(),
            Load::Idle => div().into_any_element(),
        };
        // Sample picker: validation images of the model's dataset.
        let ds_id = self.play_sample_dataset();
        let mut thumbs: Vec<AnyElement> = Vec::new();
        if let Some(ds_id) = ds_id.clone() {
            self.ensure_dataset(&ds_id, cx);
            let samples: Vec<crate::api::ImageInfo> = self
                .store
                .images
                .get(&ds_id)
                .and_then(|l| l.ready())
                .map(|v| {
                    v.iter()
                        .filter(|m| m.split == Split::Val && !m.boxes.is_empty())
                        .take(4)
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            for m in &samples {
                let img = self.image(&ds_id, &m.id, ImgKind::Thumb, cx);
                let key = format!("play-sample-{}", m.id);
                let (d, i) = (ds_id.clone(), m.id.clone());
                // The image clips inside; the control itself doesn't, so its focus ring shows whole.
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .flex_1()
                    .h(px(48.))
                    .rounded(px(radius::SM))
                    .cursor_pointer()
                    // Over an image a fill wouldn't show: the edge answers, as on dataset tiles.
                    .border_1()
                    .border_color(folo_ui::transparent_black())
                    .hover(|s| s.border_color(t.line_strong))
                    .active(|s| s.border_color(t.text_tertiary))
                    .child(div().size_full().rounded(px(radius::SM)).overflow_hidden().bg(t.bg_inset).child(
                        image_canvas(
                            &t,
                            img,
                            (m.width as f32, m.height as f32),
                            Vec::new(),
                            Fit::Cover,
                            false,
                            false,
                        ),
                    ));
                thumbs.push(
                    self.bind(
                        el,
                        &key,
                        radius::SM,
                        window,
                        cx,
                        act(move |this, _, cx| this.play_dataset_image(&d, &i, cx)),
                    )
                    .into_any_element(),
                );
            }
        }
        // Below 900 px (the default 1280 x 800 too) the picker folds to its heading, so the
        // detections, the screen's result, start higher up.
        let fold = window.viewport_size().height < px(900.);
        let picker_open = !fold || self.play_picker_open;
        let heading =
            text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Try a validation image");
        let heading = if fold {
            let el = div()
                .id("play-picker-toggle")
                .flex()
                .items_center()
                .justify_between()
                .h(px(24.))
                // A row like the others: a hover fill, inset 4 px past the text on each side.
                .mx(px(-4.))
                .px(px(4.))
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .active(|s| s.bg(t.bg_active))
                .child(heading)
                .child(icon(
                    if picker_open { "icons/chevron-down.svg" } else { "icons/chevron-right.svg" },
                    14.,
                    t.text_tertiary,
                ));
            self.bind(
                el,
                "play-picker-toggle",
                radius::SM,
                window,
                cx,
                act(|this, _, cx| {
                    this.play_picker_open = !this.play_picker_open;
                    cx.notify();
                }),
            )
            .into_any_element()
        } else {
            heading.into_any_element()
        };
        let picker = (!thumbs.is_empty() && !no_models).then(|| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .px(px(16.))
                .py(px(if fold { 10. } else { 12. }))
                .border_t_1()
                .border_color(t.line_subtle)
                .child(heading)
                .when(picker_open, |d| d.child(div().flex().gap(px(6.)).children(thumbs)))
        });
        // One scrolling column: controls and sample images, then detections and their latency (no
        // pinned footer, so no dead band and nothing hidden on short windows).
        let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
        panel(&t)
            .relative()
            .w(px(rail))
            .flex_none()
            .overflow_hidden()
            .flex()
            .flex_col()
            .child(
                div()
                    .id("play-side")
                    .flex_1()
                    .min_h_0()
                    // Clip inside the panel's bottom border, not across it.
                    .mb(px(1.))
                    .overflow_y_scroll()
                    .track_scroll(&self.play_side_scroll)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(controls)
                            // Inputs first (the model, its settings, an image to try), then the
                            // results and their latency.
                            .children(picker)
                            // The rule above the results is the block's own top border: a separate
                            // 1 px child at a fractional device offset (1.5x) could paint nothing.
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .border_t_1()
                                    .border_color(t.line_subtle)
                                    .child(detections),
                            )
                            .children(timing)
                            // Breathing room after the last row (the fade below shows only while more
                            // lies under it); 8 px, so the rail fits without a scroll at 1280 × 800.
                            .child(div().h(px(8.))),
                    ),
            )
            // A row cut by the rail's bottom edge fades out instead of stopping mid-glyph; at the end
            // of the scroll there's nothing below, so no fade dims the last line.
            .children(self.scroll_chrome(self.play_side_scroll.clone(), t.bg_panel, 0., radius::LG - 1., cx))
            .into_any_element()
    }
}

/// The confidence Playground offers as "Use … (best F1)": the model's best-F1 threshold, unless the
/// model found nothing (precision and recall 0), where that threshold is only the end of the search
/// grid and using it would hide every detection.
fn offered_conf(m: &crate::api::ModelMetrics) -> Option<f32> {
    operating_conf(m.best_f1_conf, m.precision, m.recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_f1_conf_is_offered_only_when_the_model_found_something() {
        let m = |precision: f32, recall: f32| crate::api::ModelMetrics {
            map50: 0.,
            map50_95: 0.,
            precision,
            recall,
            best_f1_conf: Some(1.0),
        };
        // A model that learned nothing yet: its "best" threshold is 1.00, which shows no boxes.
        assert_eq!(offered_conf(&m(0., 0.)), None);
        assert_eq!(offered_conf(&m(0.4, 0.)), Some(1.0));
        assert_eq!(offered_conf(&m(0.8, 0.6)), Some(1.0));
        assert_eq!(offered_conf(&crate::api::ModelMetrics { best_f1_conf: None, ..m(0.8, 0.6) }), None);
    }
}
