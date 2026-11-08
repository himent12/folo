//! Playground: open or drop an image, run a model, inspect boxes, tune thresholds.

use std::cell::Cell;
use std::rc::Rc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Bounds, Context, FontWeight, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    Pixels, SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px, relative,
};

use super::*;
use crate::api::{PredictResponse, Split};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::store::{ImgKind, Load};
use crate::theme::{radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image};

impl FoloApp {
    /// Boxes of the current prediction that pass the confidence threshold.
    fn visible_boxes(&self, p: &PredictResponse, active: Option<usize>) -> Vec<DrawBox> {
        p.boxes
            .iter()
            .filter(|b| b.score >= self.play_conf)
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: Some(b.name.clone().into()),
                score: Some(b.score),
                highlight: active == Some(i),
                dim: active.is_some_and(|h| h != i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect()
    }

    pub fn playground(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let no_models_hdr = self.store.models.ready().is_some_and(|m| m.is_empty());
        let open = if self.store.models.ready().is_some_and(|m| m.is_empty()) {
            self.button_disabled(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                "Train a model first",
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, _, cx| this.pick_image(cx)),
            )
            .tooltip(crate::controls::tip("Open an image", Some("Ctrl O")))
            .into_any_element()
        };
        let header = page_header(
            &t,
            "Playground",
            if no_models_hdr {
                "Run a model on any image."
            } else {
                "Run a model on any image. Drop a file anywhere in the window."
            },
            vec![open],
        );
        let stage = self.stage(window, cx);
        // Without models the side panel has nothing to offer: the empty state gets the full width.
        let no_models = self.store.models.ready().is_some_and(|m| m.is_empty());
        let side = (!no_models).then(|| self.play_side(window, cx));
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(16.))
            .px(px(24.))
            .pt(px(20.))
            .pb(px(20.))
            .child(header)
            .child(div().flex_1().min_h_0().flex().gap(px(16.)).child(stage).children(side))
            .into_any_element()
    }

    fn stage(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let no_models = self.store.models.ready().is_some_and(|m| m.is_empty());
        if self.play_image.is_none() {
            let pick = if no_models {
                self.button(
                    "play-pick",
                    "New run",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_new_run(None, w, cx)),
                )
            } else {
                self.button(
                    "play-pick",
                    "Choose an image",
                    Btn::Primary,
                    Some("icons/image-up.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.pick_image(cx)),
                )
            };
            let drop_hover = self.play_drop_hover;
            return div()
                .id("drop-zone")
                .flex_1()
                .min_w_0()
                .rounded(px(radius::LG))
                .bg(t.bg_inset)
                .border_1()
                .border_dashed()
                .border_color(if drop_hover { t.accent_line } else { t.line_strong })
                .flex()
                .items_center()
                .justify_center()
                .on_drag_move(cx.listener(|this, _: &gpui::DragMoveEvent<gpui::ExternalPaths>, _, cx| {
                    if !this.play_drop_hover {
                        this.play_drop_hover = true;
                        cx.notify();
                    }
                }))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .items_center()
                        .gap(px(16.))
                        .child(crate::app::logo(&t, 40.))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap(px(4.))
                                .child(text(ty::LG, t.text_primary).child(if no_models {
                                    "Train a model to start detecting"
                                } else {
                                    "Drop an image to detect objects"
                                }))
                                .child(text(ty::SM, t.text_secondary).child(if no_models {
                                    "The Playground runs your trained models. Start a run, then come back with an image."
                                } else {
                                    "JPEG or PNG · or pick a validation image on the right"
                                })),
                        )
                        .child(div().flex().items_center().gap(px(12.)).child(pick).when(!no_models, |d| {
                            d.child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(text(ty::XS, t.text_tertiary).child("or press"))
                                    .child(shortcut(&t, &["Ctrl", "O"])),
                            )
                        })),
                )
                .into_any_element();
        }
        let active = self.play_active(window, cx);
        let img = self.play_image.as_ref().unwrap();
        let dims = (img.width as f32, img.height as f32);
        let image = img.image.clone();
        let name = img.name.clone();
        let (boxes, count, loading) = match &self.play_pred {
            Load::Ready(p) => {
                let b = self.visible_boxes(p, active);
                let n = b.len();
                (b, Some(n), false)
            }
            Load::Loading => (Vec::new(), None, true),
            _ => (Vec::new(), None, false),
        };
        let bounds_cell: Rc<Cell<Option<Bounds<Pixels>>>> = Rc::new(Cell::new(None));
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let (zoom, pan) = (self.play_zoom, self.play_pan);
        let zoom_bar = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(self.icon_button(
                "play-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out (−)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.0 / 1.4, cx)),
            ))
            .child(text(ty::XS, t.text_secondary).tnum().w(px(40.)).text_center().child(if zoom <= 1.0 {
                "Fit".to_string()
            } else {
                format!("{zoom:.1}×")
            }))
            .child(self.icon_button(
                "play-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in (+)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.4, cx)),
            ))
            .child(self.icon_button(
                "play-fit",
                "icons/maximize.svg",
                "Fit (0)",
                window,
                cx,
                act(|this, _, cx| {
                    this.play_zoom = 1.0;
                    this.play_pan = (0., 0.);
                    cx.notify();
                }),
            ));
        let hover_boxes = boxes.clone();
        let paint_boxes = boxes;
        let tt = t;
        div()
            .id("stage")
            .flex_1()
            .min_w_0()
            .flex()
            .flex_col()
            .rounded(px(radius::LG))
            .border_1()
            .border_color(t.line_default)
            .overflow_hidden()
            .child(
                div()
                    .h(px(40.))
                    .flex_none()
                    .px(px(12.))
                    .flex()
                    .items_center()
                    .gap(px(10.))
                    .bg(t.bg_panel)
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(div().flex_none().child(icon("icons/image.svg", 14., t.text_tertiary)))
                    // "dataset / file": the dataset part gives way first, the file name never.
                    .map(|d| {
                        let compact = window.viewport_size().width < px(1150.);
                        let (prefix, file) = match name.rsplit_once(" / ") {
                            Some((p, f)) if !compact => (Some(p.to_string()), f.to_string()),
                            Some((_, f)) => (None, f.to_string()),
                            None => (None, name.to_string()),
                        };
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .min_w_0()
                                .flex_shrink()
                                .font_family(crate::theme::MONO)
                                .when_some(prefix, |d, p| {
                                    d.child(
                                        text(ty::SM, t.text_tertiary)
                                            .min_w_0()
                                            .flex_shrink()
                                            .truncate()
                                            .child(p),
                                    )
                                    .child(text(ty::SM, t.text_tertiary).flex_none().child("/"))
                                })
                                .child(text(ty::SM, t.text_primary).flex_none().child(file)),
                        )
                    })
                    .when(true, |d| {
                        d.child(
                            text(ty::XS, t.text_tertiary)
                                .flex_none()
                                .tnum()
                                .child(format!("{} × {}", img.width, img.height)),
                        )
                    })
                    .child(div().flex_1().min_w(px(8.)))
                    .when(loading, |d| d.child(text(ty::XS, t.text_tertiary).flex_none().child("Detecting…")))
                    // Compact windows: the side panel's latency scrolls away, so the toolbar shows it.
                    .when_some(
                        self.play_pred
                            .ready()
                            .map(|p| p.timing.total_ms)
                            .filter(|_| window.viewport_size().width < px(1100.)),
                        |d, ms| {
                            d.child(
                                text(ty::XS, t.text_secondary)
                                    .flex_none()
                                    .tnum()
                                    .child(format!("{ms:.1} ms")),
                            )
                            .child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                        },
                    )
                    .when_some(count, |d, n| {
                        d.child(div().flex_none().flex().items_center().gap(px(6.)).child(
                            text(ty::XS, t.text_secondary).tnum().child(format!(
                                "{n} {} ≥ {:.2}",
                                if n == 1 { "object" } else { "objects" },
                                self.play_conf
                            )),
                        ))
                    })
                    .child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                    .child(zoom_bar.flex_none()),
            )
            .child(
                div()
                    .id("stage-canvas")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .bg(t.bg_inset)
                    .cursor(if zoom > 1.0 { gpui::CursorStyle::OpenHand } else { gpui::CursorStyle::Arrow })
                    .on_scroll_wheel(cx.listener(move |this, ev: &gpui::ScrollWheelEvent, w, cx| {
                        let Some(b) = wheel_cell.get() else { return };
                        let dy = f32::from(ev.delta.pixel_delta(w.line_height()).y);
                        let new_zoom =
                            (this.play_zoom * (1.0 + dy * 0.0025).clamp(0.5, 2.0)).clamp(1.0, 12.0);
                        this.play_pan = crate::viz::zoom_about(
                            b,
                            dims,
                            this.play_zoom,
                            this.play_pan,
                            ev.position,
                            new_zoom,
                        );
                        this.play_zoom = new_zoom;
                        cx.notify();
                    }))
                    .on_mouse_down(
                        gpui::MouseButton::Left,
                        cx.listener(|this, ev: &gpui::MouseDownEvent, _, _| {
                            this.play_drag = Some((ev.position, this.play_pan));
                        }),
                    )
                    .on_mouse_up(gpui::MouseButton::Left, cx.listener(|this, _, _, _| this.play_drag = None))
                    .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                        if let Some((from, pan0)) = this.play_drag
                            && ev.pressed_button == Some(gpui::MouseButton::Left)
                            && this.play_zoom > 1.0
                        {
                            this.play_pan = (
                                pan0.0 + f32::from(ev.position.x - from.x),
                                pan0.1 + f32::from(ev.position.y - from.y),
                            );
                            cx.notify();
                            return;
                        }
                        let Some(b) = bounds_cell.get() else { return };
                        let (x, y) =
                            to_image(b, dims, Fit::Contain, this.play_zoom, this.play_pan, ev.position);
                        let hit = hit_box(&hover_boxes, x, y);
                        if hit != this.play_hover {
                            this.play_hover = hit;
                            cx.notify();
                        }
                    }))
                    .on_hover(cx.listener(|this, h: &bool, _, cx| {
                        if !*h && this.play_hover.is_some() {
                            this.play_hover = None;
                            cx.notify();
                        }
                    }))
                    .child(
                        canvas(
                            |_, _, _| (),
                            move |b, _, window, cx| {
                                let inner = Bounds::new(
                                    b.origin + gpui::point(px(16.), px(16.)),
                                    b.size - gpui::size(px(32.), px(32.)),
                                );
                                paint_cell.set(Some(inner));
                                paint_image_with_boxes(
                                    &tt,
                                    window,
                                    cx,
                                    inner,
                                    Some(&image),
                                    dims,
                                    &paint_boxes,
                                    Fit::Contain,
                                    zoom,
                                    pan,
                                    true,
                                    0.,
                                );
                            },
                        )
                        .size_full(),
                    ),
            )
            .into_any_element()
    }

    fn play_side(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let models = self.store.models.ready().cloned().unwrap_or_default();
        let sel =
            self.play_model.as_ref().and_then(|id| models.iter().position(|m| &m.id == id)).unwrap_or(0);
        let model = models.get(sel).cloned();
        let ids: Vec<String> = models.iter().map(|m| m.id.clone()).collect();
        let select = self.select(
            "play-model",
            model.as_ref().map_or("No models".into(), |m| SharedString::from(m.name.clone())),
            models
                .iter()
                .map(|m| {
                    (
                        SharedString::from(m.name.clone()),
                        m.metrics
                            .as_ref()
                            .map(|x| SharedString::from(format!("{} · {:.3}", m.arch.as_str(), x.map50_95))),
                    )
                })
                .collect(),
            sel,
            window,
            cx,
            Rc::new(move |this, i, _, cx| {
                if let Some(id) = ids.get(i) {
                    this.play_model = Some(id.clone());
                    this.run_predict(cx);
                }
            }),
        );
        let conf = self.slider("play-conf", self.play_conf, 0.05, 0.95, 0.01, window, cx);
        let iou = self.slider("play-iou", self.play_iou, 0.3, 0.95, 0.01, window, cx);
        let conf_help = self.help_icon(
            "help-conf",
            "Hide detections scoring below this. Filtering is instant; it doesn't rerun the model.",
            window,
            cx,
        );
        let iou_help = self.help_icon("help-iou", "Non-max suppression drops a weaker box that overlaps a stronger one by more than this. Changing it reruns the model.", window, cx);
        let field = |label: &str, value: String, control: AnyElement, help: AnyElement| {
            div()
                .flex()
                .flex_col()
                .gap(px(6.))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .items_center()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(4.))
                                .child(
                                    text(ty::SM, t.text_secondary)
                                        .font_weight(FontWeight::MEDIUM)
                                        .child(label.to_string()),
                                )
                                .child(help),
                        )
                        .child(text(ty::SM, t.text_primary).tnum().child(value)),
                )
                .child(control)
        };
        let no_models = models.is_empty() && self.store.models.ready().is_some();
        let controls = div()
            .flex()
            .flex_col()
            .gap(px(14.))
            .p(px(16.))
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
                    .when(no_models, |d| {
                        d.child(
                            text(ty::XS, t.text_tertiary)
                                .child("No models yet. Finished runs add their best weights here."),
                        )
                    }),
            )
            .when(!no_models, |d| {
                d.child(field(
                    "Confidence",
                    format!("{:.2}", self.play_conf),
                    conf.into_any_element(),
                    conf_help,
                ))
                .child(field(
                    "IoU",
                    format!("{:.2}", self.play_iou),
                    iou.into_any_element(),
                    iou_help,
                ))
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
                        let color = t.class(b.class);
                        let key = format!("det-{i}");
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .flex_none()
                            .h(px(28.))
                            .px(px(8.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .rounded(px(radius::SM))
                            .when(hot, |d| d.bg(t.bg_active).child(crate::ui::select_bar(&t)))
                            .hover(|s| s.bg(t.bg_hover))
                            .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                                let next = if *h { Some(i) } else { None };
                                if this.play_hover != next {
                                    this.play_hover = next;
                                    cx.notify();
                                }
                            }))
                            .child(swatch(color, 8.))
                            .child(text(ty::SM, t.text_primary).w(px(68.)).truncate().child(b.name.clone()))
                            // Size tells apart two boxes of the same class.
                            .child(
                                text(ty::XS, t.text_tertiary).tnum().w(px(52.)).flex_none().child(format!(
                                    "{:.0}×{:.0}",
                                    b.x2 - b.x1,
                                    b.y2 - b.y1
                                )),
                            )
                            .child(
                                div().flex_1().h(px(4.)).rounded_full().bg(t.bg_inset).child(
                                    div()
                                        .h_full()
                                        .w(relative(b.score))
                                        .rounded_full()
                                        .bg(t.class_stroke(b.class)),
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
                // One neutral ramp: the model step darkest, the cheap steps lighter.
                // One neutral ramp; the model step strongest without shouting.
                let (c_pre, c_model, c_nms) = (
                    crate::theme::alpha(t.text_tertiary, 0.4),
                    t.text_tertiary,
                    crate::theme::alpha(t.text_tertiary, 0.65),
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
                            .items_baseline()
                            .child(
                                text(ty::SM, t.text_secondary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child("Latency"),
                            )
                            .child(metric(&t, format!("{:.1}", p.timing.total_ms), "ms", ty::LG, ty::XS)),
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
                            .child(seg(p.timing.postprocess_ms, c_nms)),
                    )
                    .child(
                        div().flex().gap(px(12.)).justify_between().children(
                            [
                                ("pre", p.timing.preprocess_ms, c_pre),
                                ("model", p.timing.inference_ms, c_model),
                                ("NMS", p.timing.postprocess_ms, c_nms),
                            ]
                            .into_iter()
                            .map(|(n, v, c)| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(5.))
                                    .child(swatch(c, 6.))
                                    .child(text(ty::XS, t.text_tertiary).child(n))
                                    .child(text(ty::XS, t.text_secondary).tnum().child(format!("{v:.1} ms")))
                            }),
                        ),
                    )
                    .child(text(ty::XS, t.text_tertiary).tnum().child(format!(
                        "{} · {:.0} fps at this size",
                        p.device,
                        1000. / total
                    )));
                timing = Some(timing_el.into_any_element());
                div()
                    .flex()
                    .flex_col()
                    .flex_none()
                    .child(panel_header(&t, "Detections").child(text(ty::XS, t.text_tertiary).tnum().child(
                        if hidden > 0 {
                            format!("{} shown · {hidden} below threshold", vis.len())
                        } else {
                            vis.len().to_string()
                        },
                    )))
                    .child(
                        div()
                            .p(px(6.))
                            .flex()
                            .flex_col()
                            .gap(px(1.))
                            .when(!vis.is_empty(), |d| {
                                // Column labels, on the rows' grid: swatch + class, box size, confidence.
                                d.child(
                                    div()
                                        .h(px(24.))
                                        .px(px(8.))
                                        .flex()
                                        .items_center()
                                        .gap(px(8.))
                                        .child(th(&t, "Class").w(px(84.)).flex_none())
                                        .child(th(&t, "Size (px)").w(px(52.)).flex_none())
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
            Load::Failed(e) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            Load::Loading => div()
                .flex_none()
                .child(
                    panel_header(&t, "Detections").child(text(ty::XS, t.text_tertiary).child("Detecting…")),
                )
                .child(loading_rows(&t, "det", 4))
                .into_any_element(),
            Load::Idle => div().into_any_element(),
        };
        // Sample picker: validation images of the model's dataset.
        let ds_id = model
            .as_ref()
            .and_then(|m| m.run_id.as_ref())
            .and_then(|r| self.store.run(r))
            .map(|r| r.config.dataset_id.clone())
            .or_else(|| self.store.datasets.ready().and_then(|d| d.first()).map(|d| d.id.clone()));
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
                        .skip(17)
                        .take(4)
                        .cloned()
                        .collect()
                })
                .unwrap_or_default();
            for m in &samples {
                let img = self.image(&ds_id, &m.id, ImgKind::Thumb, cx);
                let key = format!("play-sample-{}", m.id);
                let (d, i) = (ds_id.clone(), m.id.clone());
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .flex_1()
                    .h(px(48.))
                    .rounded(px(radius::SM))
                    .overflow_hidden()
                    .bg(t.bg_inset)
                    .cursor_pointer()
                    .hover(|s| s.opacity(0.85))
                    .child(image_canvas(
                        &t,
                        img,
                        (m.width as f32, m.height as f32),
                        Vec::new(),
                        Fit::Cover,
                        false,
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
        let picker = (!thumbs.is_empty() && !no_models).then(|| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .p(px(16.))
                .border_t_1()
                .border_color(t.line_subtle)
                .child(
                    text(ty::SM, t.text_secondary)
                        .font_weight(FontWeight::MEDIUM)
                        .child("Try a validation image"),
                )
                .child(div().flex().gap(px(6.)).children(thumbs))
        });
        // One scrolling column: controls, detections, samples, then latency right after them (no
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
                    .overflow_y_scroll()
                    .track_scroll(&self.play_side_scroll)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(controls)
                            .child(hairline(&t))
                            .child(detections)
                            .children(picker)
                            .children(timing),
                    ),
            )
            .child(self.scrollbar(self.play_side_scroll.clone(), cx))
            .into_any_element()
    }

    /// The detection being inspected: under the pointer, or a keyboard-focused row.
    pub fn play_active(&self, window: &Window, cx: &mut Context<Self>) -> Option<usize> {
        if self.play_hover.is_some() {
            return self.play_hover;
        }
        let n = self
            .play_pred
            .ready()
            .map_or(0, |p| p.boxes.iter().filter(|b| b.score >= self.play_conf).count());
        (0..n).find(|i| self.fh(format!("det-{i}"), cx).is_focused(window))
    }
}
