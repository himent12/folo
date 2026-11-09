//! Playground: open or drop an image, run a model, inspect boxes, tune thresholds.

use std::rc::Rc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Bounds, Context, FontWeight, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px, relative,
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
                window,
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
            // No models: the body's call to action is the only one (no inert button beside it).
            if no_models_hdr { Vec::new() } else { vec![open] },
        );
        let stage = self.stage(window, cx);
        self.probe_local_gpu(cx);
        // Without models the side panel has nothing to offer: the empty state gets the full width.
        let no_models = self.store.models.ready().is_some_and(|m| m.is_empty());
        let side = (!no_models).then(|| self.play_side(window, cx));
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap(px(16.))
            .px(px(self.gutter()))
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
            // Every enabled action does something: with no datasets either, the first step is one.
            let no_datasets = self.store.datasets.ready().is_some_and(|d| d.is_empty());
            // The model list failed: the step is fetching it again, not picking an image.
            let models_failed = self.store.models.error().is_some();
            let pick = if models_failed {
                self.button(
                    "play-pick",
                    "Retry",
                    Btn::Primary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.retry_models(cx)),
                )
            } else if no_models && no_datasets {
                self.button(
                    "play-pick",
                    "Add dataset",
                    Btn::Primary,
                    Some("icons/plus.svg"),
                    window,
                    cx,
                    act(|this, w, cx| {
                        // The label promises the picker: land on Datasets with it open (the built-in demo
                        // can't register folders, and says so there instead).
                        this.navigate(crate::app::Route::Datasets, w, cx);
                        this.add_dataset_folder(cx);
                    }),
                )
            } else if no_models {
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
            // Capped at 420 px and centred: an empty state, not a 970 px hole.
            let zone = div()
                .id("drop-zone")
                .flex_1()
                .min_w_0()
                .rounded(px(radius::LG))
                // The panel surface, like every other empty state.
                .bg(t.bg_panel)
                .border_1()
                // Dashed only where a drop works (a model to run it).
                .when(!no_models, |d| d.border_dashed())
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
                                // Headline, body and button name the same next step.
                                .child(text(ty::LG, t.text_primary).child(if models_failed {
                                    "The model list didn't load"
                                } else if no_models && no_datasets {
                                    "Add a dataset to get started"
                                } else if no_models {
                                    "Train a model to start detecting"
                                } else {
                                    "Drop an image to detect objects"
                                }))
                                .child(text(ty::SM, t.text_secondary).child(if models_failed {
                                    "Check the connection, then try again."
                                } else if no_models && no_datasets {
                                    "Then train a model on it and come back with an image."
                                } else if no_models {
                                    "The Playground runs your trained models. Start a run, then come back with an image."
                                } else {
                                    "JPEG or PNG · or pick a validation image on the right"
                                })),
                        )
                        .child(div().flex().items_center().gap(px(12.)).child(pick).when(!no_models && !models_failed, |d| {
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
                .max_h(px(420.));
            return div().flex_1().min_w_0().flex().flex_col().child(zone).into_any_element();
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
        let bounds_cell = self.play_view.clone();
        if let Some(b) = bounds_cell.get() {
            self.play_pan =
                crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, self.play_zoom, self.play_pan);
        }
        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let (zoom, pan) = (self.play_zoom, self.play_pan);
        // Fitted (by the readout, 0, −, or a new image), the readout is text: focus left on it
        // moves to its neighbour instead of falling back to the window.
        if zoom <= 1.0 && self.fh("play-fit", cx).is_focused(window) {
            window.focus(&self.fh("play-zoom-in", cx));
        }
        // How to move around the image, as the inspector lists it (the fitted "Fit" names no key).
        let view_help = self.help_icon(
            "play-view-help",
            "Scroll, + or − zoom · drag, or arrows when zoomed, pan · 0 fits the image",
            window,
            cx,
        );
        let zoom_bar = div()
            .flex()
            .items_center()
            .gap(px(2.))
            .child(div().pr(px(4.)).child(view_help))
            .child(self.icon_button(
                "play-zoom-out",
                "icons/zoom-out.svg",
                "Zoom out (−)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.0 / 1.4, cx)),
            ))
            .child(self.zoom_readout(
                "play-fit",
                zoom,
                window,
                cx,
                act(|this, _, cx| {
                    this.play_zoom = 1.0;
                    this.play_pan = (0., 0.);
                    cx.notify();
                }),
            ))
            .child(self.icon_button(
                "play-zoom-in",
                "icons/zoom-in.svg",
                "Zoom in (+)",
                window,
                cx,
                act(|this, _, cx| this.play_zoom_by(1.4, cx)),
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
                                .child({
                                    // The name takes what the toolbar leaves: the stage's width (window
                                    // less sidebar, gutters and rail) less its other items (~560 px:
                                    // crumb, size, latency, count, zoom). A long name ends in "…" at that
                                    // known width, as GPUI needs; the full name is on hover.
                                    let vw = f32::from(window.viewport_size().width);
                                    let side = if vw < 1120. {
                                        crate::theme::size::SIDEBAR_COLLAPSED
                                    } else {
                                        crate::theme::size::SIDEBAR
                                    };
                                    let rail = if vw < 1100. { 280. } else { 320. };
                                    let stage = vw.min(crate::theme::size::CONTENT_MAX + side)
                                        - side
                                        - 2. * self.gutter()
                                        - rail
                                        - 16.;
                                    let budget =
                                        (stage - 584.).clamp(if compact { 120. } else { 160. }, 520.);
                                    // Monospace: the measured width is the glyph count times the advance.
                                    let advance = window.text_system().advance(
                                        window.text_system().resolve_font(&gpui::font(crate::theme::MONO)),
                                        px(ty::SM.size),
                                        'M',
                                    );
                                    let width = advance.map_or(8., |a| f32::from(a.width))
                                        * file.chars().count() as f32;
                                    let long = width > budget;
                                    let full = file.clone();
                                    div()
                                        .id("play-file-name")
                                        .flex_none()
                                        .when(long, |d| d.tooltip(crate::controls::tip(full, None)))
                                        .child(
                                            text(ty::SM, t.text_primary)
                                                .w(px(width.min(budget).ceil()))
                                                .truncate()
                                                .child(file),
                                        )
                                }),
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
                    // The latency always reads here, beside the count (the rail's breakdown is
                    // below the fold at most window sizes).
                    .when_some(self.play_pred.ready().map(|p| p.timing.total_ms), |d, ms| {
                        d.child(
                            text(ty::XS, t.text_secondary).flex_none().tnum().child(format!("{ms:.1} ms")),
                        )
                        .child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
                    })
                    .when_some(count, |d, n| {
                        d.child(div().flex_none().flex().items_center().gap(px(6.)).child(
                            text(ty::XS, t.text_secondary).tnum().child(format!(
                                "{n} {} ≥ {:.2}",
                                if n == 1 { "detection" } else { "detections" },
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
                            // The pointer moved over the image: hover leads again.
                            this.kb = false;
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
        // An end-to-end model (the run that trained it had the one-to-one head) predicts without NMS,
        // so its IoU setting does nothing.
        let e2e = model.as_ref().is_some_and(|m| self.model_e2e(m));
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
                        m.metrics.as_ref().map(|x| {
                            // End-to-end named in the picker: it's why the NMS row changes.
                            let e2e = if self.model_e2e(m) { " · end-to-end" } else { "" };
                            SharedString::from(format!("{}{e2e} · {:.3}", m.arch.as_str(), x.map50_95))
                        }),
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
        // Short windows: a tighter control block, so more detection rows show without scrolling.
        let tight = window.viewport_size().height < px(700.);
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
            if tight {
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
            .gap(px(if tight { 8. } else { 14. }))
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
                    .when(no_models, |d| {
                        d.child(
                            text(ty::XS, t.text_tertiary)
                                .child("No models yet. Finished runs add their best weights here."),
                        )
                    }),
            )
            .when(!no_models, |d| {
                d.child(run_on)
                    .child(field(
                        "Confidence",
                        format!("{:.2}", self.play_conf),
                        conf.into_any_element(),
                        conf_help,
                    ))
                    // "NMS IoU": the suppression overlap, not the evaluation's IoU 0.50. An end-to-end
                    // model has no suppression step: the row says so instead of offering an inert slider.
                    .map(|d| {
                        if e2e {
                            d.child(
                                div()
                                    .flex()
                                    .justify_between()
                                    .items_baseline()
                                    .gap(px(10.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .child("NMS"),
                                    )
                                    .child(
                                        text(ty::XS, t.text_tertiary).child("Not used · end-to-end model"),
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
                        let color = t.class(b.class);
                        let key = format!("det-{i}");
                        // A cut class name shows in full on hover and on keyboard focus.
                        let long = may_overflow(&b.name, ty::SM.size, 68.);
                        let focused = long && self.ring(&self.fh(key.clone(), cx), window);
                        let full = b.name.clone();
                        let row = div()
                            .id(SharedString::from(key.clone()))
                            .when(long, |d| d.tooltip(crate::controls::tip(full.clone(), None)))
                            .when(focused, |d| d.child(crate::controls::focus_tip(&t, full.clone(), 28.)))
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
                            .child(crate::ui::class_swatch(&t, color, 8.))
                            .child(text(ty::SM, t.text_primary).w(px(68.)).truncate().child(b.name.clone()))
                            // Size tells apart two boxes of the same class.
                            .child(
                                // On the active row's fill, tertiary falls under AA: secondary there.
                                text(ty::XS, if hot { t.text_secondary } else { t.text_tertiary })
                                    .tnum()
                                    .w(px(52.))
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
                            .items_baseline()
                            .gap(px(8.))
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .items_baseline()
                                    .gap(px(6.))
                                    .child(
                                        text(ty::SM, t.text_secondary)
                                            .font_weight(FontWeight::MEDIUM)
                                            .flex_none()
                                            .child("Latency"),
                                    )
                                    .child(text(ty::XS, t.text_tertiary).tnum().min_w_0().truncate().child(
                                        format!(
                                            "{} · {:.0} fps",
                                            device_name(&p.device, self.store.system.ready()),
                                            1000. / total
                                        ),
                                    )),
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
                                    .child(text(ty::XS, t.text_tertiary).child(n))
                                    .child(text(ty::XS, t.text_secondary).tnum().child(format!("{v:.1} ms")))
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
                        text(ty::XS, t.text_tertiary).tnum().child(if self.play_pred_pending {
                            "Updating…".to_string()
                        } else if hidden > 0 {
                            format!("{} shown · {hidden} below threshold", vis.len())
                        } else {
                            vis.len().to_string()
                        }),
                    ))
                    .child(
                        div()
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
                        d.child(text(ty::XS, t.text_tertiary).whitespace_normal().child(h))
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
                        .child(text(ty::XS, t.text_tertiary).child("Detecting…")),
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
                    .border_color(gpui::transparent_black())
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
                .py(px(if fold { 10. } else { 16. }))
                .border_t_1()
                .border_color(t.line_subtle)
                .child(heading)
                .when(picker_open, |d| d.child(div().flex().gap(px(6.)).children(thumbs)))
        });
        // One scrolling column: controls and sample images, then detections and their latency (no
        // pinned footer, so no dead band and nothing hidden on short windows).
        let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
        let more_below = {
            let s = &self.play_side_scroll;
            s.max_offset().height + s.offset().y > px(1.)
        };
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
                            .child(hairline(&t))
                            .child(detections)
                            .children(timing)
                            // Breathing room after the last row (the fade below shows only while more
                            // lies under it); 8 px, so the rail fits without a scroll at 1280 × 800.
                            .child(div().h(px(8.))),
                    ),
            )
            // A row cut by the rail's bottom edge fades out instead of stopping mid-glyph; at the end
            // of the scroll there's nothing below, so no fade dims the last line.
            .when(more_below, |d| {
                d.child(
                    div()
                        .absolute()
                        .bottom(px(1.))
                        .left_0()
                        .right_0()
                        .h(px(20.))
                        .rounded_b(px(radius::LG))
                        .bg(gpui::linear_gradient(
                            0.,
                            gpui::linear_color_stop(t.bg_panel, 0.),
                            gpui::linear_color_stop(crate::theme::alpha(t.bg_panel, 0.), 1.),
                        )),
                )
            })
            .child(self.scrollbar(self.play_side_scroll.clone(), cx))
            .into_any_element()
    }

    /// Server / Local GPU / Local CPU, with what each means here.
    fn run_on(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        use crate::local::Target;
        let t = theme(cx);
        let local = self.store.api.local_inference();
        let no_local: Option<SharedString> =
            (!local).then(|| "Needs folo serve: the built-in demo has no weights to run here".into());
        let gpu_why = no_local.clone().or_else(|| match &self.local_gpu {
            Load::Ready(_) => None,
            Load::Failed(_) => Some("No GPU this app can use was found".into()),
            _ => Some("Looking for a GPU…".into()),
        });
        let selected = match self.play_target {
            None => 0,
            Some(Target::Gpu) => 1,
            Some(Target::Cpu) => 2,
        };
        let control = self.segmented_fill_with(
            "play-target",
            // The built-in demo isn't a server: say what actually runs.
            &[if local { "Server" } else { "Demo" }, "Local GPU", "Local CPU"],
            selected,
            &[None, gpu_why, no_local],
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                let target = match i {
                    1 => Some(Target::Gpu),
                    2 => Some(Target::Cpu),
                    _ => None,
                };
                this.set_play_target(target, cx);
            }),
        );
        let help = self.help_icon(
            "help-run-on",
            "Local runs download the model's weights once (f16, checked for changes before each run) and run them in this app, leaving the server's GPU to training.",
            window,
            cx,
        );
        let note = match (self.play_target, &self.local_gpu) {
            (None, _) if !local => "Built-in demo · simulated inference".into(),
            (None, _) => format!("On folo serve at {}", self.store.api.describe()),
            (Some(Target::Gpu), Load::Ready(name)) => {
                format!("In this app, on {}", crate::ui::short_gpu(name))
            }
            (Some(Target::Gpu), _) => "In this app, on this PC's GPU".into(),
            (Some(Target::Cpu), _) => "In this app, on the CPU".into(),
        };
        div()
            .flex()
            .flex_col()
            .gap(px(6.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(4.))
                    .child(text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Run on"))
                    .child(help),
            )
            .child(control)
            // Wraps rather than truncating: GPUI only ellipsizes against a known width, so a
            // w_full note would hard-clip mid-word in a narrow rail.
            .child(text(ty::XS, t.text_tertiary).w_full().whitespace_normal().child(note))
            .into_any_element()
    }

    /// The detection being inspected: under the pointer, or a keyboard-focused row.
    /// While the keyboard is in use the focused row leads (a pointer hover left on another row
    /// doesn't win); otherwise the hover does, as in the inspector.
    pub fn play_active(&self, window: &Window, cx: &mut Context<Self>) -> Option<usize> {
        let n = self
            .play_pred
            .ready()
            .map_or(0, |p| p.boxes.iter().filter(|b| b.score >= self.play_conf).count());
        let focused = (0..n).find(|i| self.fh(format!("det-{i}"), cx).is_focused(window));
        if self.kb { focused.or(self.play_hover) } else { self.play_hover.or(focused) }
    }
}

/// "gpu:0" as the adapter's name ("RTX 4080 SUPER"), "cpu" as "CPU"; anything else as given.
fn device_name(device: &str, system: Option<&crate::api::SystemInfo>) -> String {
    // Local inference names its device itself.
    if let Some(name) = device.strip_prefix("local:") {
        return format!("{name} in this app");
    }
    if device == "cpu" {
        return "CPU".into();
    }
    device
        .strip_prefix("gpu:")
        .and_then(|n| n.parse::<u32>().ok())
        .and_then(|n| system?.gpus.iter().find(|g| g.ordinal == n))
        .map_or(device.to_string(), |g| crate::ui::short_gpu(&g.name))
}
