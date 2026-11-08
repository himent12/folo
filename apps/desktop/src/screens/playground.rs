//! Playground: open or drop an image, run a model, inspect boxes, tune thresholds.

use std::cell::Cell;
use std::rc::Rc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Bounds, Context, FontWeight, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    Pixels, SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px, relative,
};

use super::*;
use crate::api::{Prediction, Split};
use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::store::{ImgKind, Load};
use crate::theme::{alpha, radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image};

impl FoloApp {
    /// Boxes of the current prediction that pass the confidence threshold.
    fn visible_boxes(&self, p: &Prediction) -> Vec<DrawBox> {
        p.boxes
            .iter()
            .filter(|b| b.score >= self.play_conf)
            .enumerate()
            .map(|(i, b)| DrawBox {
                label: Some(b.name.clone().into()),
                score: Some(b.score),
                highlight: self.play_hover == Some(i),
                dim: self.play_hover.is_some_and(|h| h != i),
                ..DrawBox::new(b.class, b.x1, b.y1, b.x2, b.y2)
            })
            .collect()
    }

    pub fn playground(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let open = self
            .button(
                "play-open",
                "Open image",
                Btn::Secondary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, _, cx| this.pick_image(cx)),
            )
            .tooltip(crate::controls::tip("Open an image", Some("Ctrl O")))
            .into_any_element();
        let header = page_header(
            &t,
            "Playground",
            "Run a model on any image. Drop a file anywhere in the window.",
            vec![open],
        );
        let stage = self.stage(window, cx);
        let side = self.play_side(window, cx);
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(16.))
            .px(px(24.))
            .pt(px(20.))
            .pb(px(20.))
            .child(header)
            .child(div().flex_1().min_h_0().flex().gap(px(16.)).child(stage).child(side))
            .into_any_element()
    }

    fn stage(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let Some(img) = self.play_image.as_ref() else {
            let pick = self.button(
                "play-pick",
                "Choose an image",
                Btn::Primary,
                Some("icons/image-up.svg"),
                window,
                cx,
                act(|this, _, cx| this.pick_image(cx)),
            );
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
                                .child(text(ty::LG, t.text_primary).child("Drop an image to detect objects"))
                                .child(
                                    text(ty::SM, t.text_secondary)
                                        .child("JPEG or PNG · or pick a validation sample on the right"),
                                ),
                        )
                        .child(
                            div().flex().items_center().gap(px(12.)).child(pick).child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(text(ty::XS, t.text_tertiary).child("or press"))
                                    .child(shortcut(&t, &["Ctrl", "O"])),
                            ),
                        ),
                )
                .into_any_element();
        };
        let dims = (img.width as f32, img.height as f32);
        let image = img.image.clone();
        let name = img.name.clone();
        let (boxes, count, loading) = match &self.play_pred {
            Load::Ready(p) => {
                let b = self.visible_boxes(p);
                let n = b.len();
                (b, Some(n), false)
            }
            Load::Loading => (Vec::new(), None, true),
            _ => (Vec::new(), None, false),
        };
        let bounds_cell: Rc<Cell<Option<Bounds<Pixels>>>> = Rc::new(Cell::new(None));
        let paint_cell = bounds_cell.clone();
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
                    .child(icon("icons/image.svg", 14., t.text_tertiary))
                    .child(
                        text(ty::SM, t.text_primary).font_family(crate::theme::MONO).truncate().child(name),
                    )
                    .child(
                        text(ty::XS, t.text_tertiary).tnum().child(format!("{} × {}", img.width, img.height)),
                    )
                    .child(div().flex_1())
                    .when(loading, |d| d.child(text(ty::XS, t.text_tertiary).child("Detecting…")))
                    .when_some(count, |d, n| {
                        d.child(
                            div()
                                .flex()
                                .items_center()
                                .gap(px(6.))
                                .child(div().size(px(6.)).rounded_full().bg(t.accent_solid))
                                .child(text(ty::XS, t.text_secondary).tnum().child(format!(
                                    "{n} {} ≥ {:.2}",
                                    if n == 1 { "object" } else { "objects" },
                                    self.play_conf
                                ))),
                        )
                    }),
            )
            .child(
                div()
                    .id("stage-canvas")
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .bg(t.bg_inset)
                    .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                        let Some(b) = bounds_cell.get() else { return };
                        let (x, y) = to_image(b, dims, Fit::Contain, 1.0, (0., 0.), ev.position);
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
                                    1.0,
                                    (0., 0.),
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
                            .map(|x| SharedString::from(format!("{} · {:.3}", m.arch, x.map50_95))),
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
        let field = |label: &str, value: String, control: AnyElement, help: &str| {
            div()
                .flex()
                .flex_col()
                .gap(px(8.))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .items_baseline()
                        .child(
                            text(ty::SM, t.text_secondary)
                                .font_weight(FontWeight::MEDIUM)
                                .child(label.to_string()),
                        )
                        .child(text(ty::SM, t.text_primary).tnum().child(value)),
                )
                .child(control)
                .child(text(ty::XS, t.text_tertiary).child(help.to_string()))
        };
        let controls = div()
            .flex()
            .flex_col()
            .gap(px(18.))
            .p(px(16.))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(8.))
                    .child(text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Model"))
                    .child(select),
            )
            .child(field(
                "Confidence",
                format!("{:.2}", self.play_conf),
                conf.into_any_element(),
                "Hide detections scoring below this.",
            ))
            .child(field(
                "IoU",
                format!("{:.2}", self.play_iou),
                iou.into_any_element(),
                "Overlap at which NMS merges boxes.",
            ));

        let detections = match &self.play_pred {
            Load::Ready(p) => {
                let vis: Vec<crate::api::DetBox> =
                    p.boxes.iter().filter(|b| b.score >= self.play_conf).cloned().collect();
                let rows: Vec<AnyElement> = vis
                    .iter()
                    .enumerate()
                    .map(|(i, b)| {
                        let hot = self.play_hover == Some(i);
                        let color = t.class(b.class);
                        div()
                            .id(("det", i))
                            .h(px(32.))
                            .px(px(8.))
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .rounded(px(radius::SM))
                            .when(hot, |d| d.bg(t.bg_active))
                            .hover(|s| s.bg(t.bg_hover))
                            .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                                let next = if *h { Some(i) } else { None };
                                if this.play_hover != next {
                                    this.play_hover = next;
                                    cx.notify();
                                }
                            }))
                            .child(swatch(color, 8.))
                            .child(text(ty::SM, t.text_primary).w(px(72.)).truncate().child(b.name.clone()))
                            .child(div().flex_1().h(px(4.)).rounded_full().bg(t.bg_inset).child(
                                div().h_full().w(relative(b.score)).rounded_full().bg(alpha(color, 0.9)),
                            ))
                            .child(
                                text(ty::SM, t.text_primary)
                                    .tnum()
                                    .w(px(36.))
                                    .text_right()
                                    .child(format!("{:.2}", b.score)),
                            )
                            .into_any_element()
                    })
                    .collect();
                let hidden = p.boxes.len() - vis.len();
                let total = p.timing.total_ms.max(0.001);
                let seg = |v: f32, c| div().h_full().w(relative(v / total)).bg(c);
                let timing = div()
                    .flex()
                    .flex_col()
                    .gap(px(10.))
                    .p(px(16.))
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
                            .child(seg(p.timing.preprocess_ms, t.chart_series[1]))
                            .child(seg(p.timing.inference_ms, t.chart_series[0]))
                            .child(seg(p.timing.postprocess_ms, t.chart_series[2])),
                    )
                    .child(
                        div().flex().justify_between().children(
                            [
                                ("pre", p.timing.preprocess_ms, t.chart_series[1]),
                                ("model", p.timing.inference_ms, t.chart_series[0]),
                                ("NMS", p.timing.postprocess_ms, t.chart_series[2]),
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
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h_0()
                    .child(panel_header(&t, "Detections").child(text(ty::XS, t.text_tertiary).tnum().child(if hidden > 0 { format!("{} · {hidden} below threshold", vis.len()) } else { vis.len().to_string() })))
                    .child(div().id("det-list").flex_1().min_h_0().overflow_y_scroll().p(px(6.)).children(rows).when(vis.is_empty(), |d| d.child(div().p(px(10.)).child(text(ty::SM, t.text_tertiary).child("Nothing above the threshold. Lower the confidence to see weaker detections.")))))
                    .child(hairline(&t))
                    .child(timing)
                    .into_any_element()
            }
            Load::Failed(e) => div()
                .p(px(16.))
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
                .into_any_element(),
            Load::Loading => div()
                .child(panel_header(&t, "Detections"))
                .child(loading_rows(&t, "det", 4))
                .into_any_element(),
            Load::Idle => div().flex_1().into_any_element(),
        };
        // Sample picker: validation images of the model's dataset.
        let ds_id = model
            .as_ref()
            .and_then(|m| m.run_id.as_ref())
            .and_then(|r| self.store.run(r))
            .map(|r| r.config.dataset_id.clone())
            .unwrap_or_else(|| "ds_8f2c".into());
        self.ensure_dataset(&ds_id, cx);
        let samples: Vec<crate::api::ImageMeta> = self
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
        let thumbs: Vec<AnyElement> = samples
            .iter()
            .map(|m| {
                let img = self.image(&ds_id, &m.id, ImgKind::Thumb, cx);
                let key = format!("play-sample-{}", m.id);
                let (d, i) = (ds_id.clone(), m.id.clone());
                let el = div()
                    .id(SharedString::from(key.clone()))
                    .flex_1()
                    .h(px(56.))
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
                self.bind(
                    el,
                    &key,
                    radius::SM,
                    window,
                    cx,
                    act(move |this, _, cx| this.play_dataset_image(&d, &i, cx)),
                )
                .into_any_element()
            })
            .collect();
        let picker = div()
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
            .child(div().flex().gap(px(6.)).children(thumbs));
        panel(&t)
            .w(px(320.))
            .flex_none()
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(controls)
            .child(hairline(&t))
            .child(div().flex_1().min_h_0().flex().flex_col().child(detections))
            .child(picker)
            .into_any_element()
    }
}
