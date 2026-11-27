//! Playground Compare A/B dual-canvas stage: Model A vs Model B side-by-side with shared zoom/pan,
//! per-pane header readouts, loading/error states, and independent detection hover.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Bounds, Context, FontWeight, InteractiveElement, IntoElement, MouseMoveEvent, ParentElement,
    Pixels, RenderImage, SharedString, StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::app::{FoloApp, act};
use crate::controls::Btn;
use crate::store::Load;
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;
use crate::viz::{DrawBox, Fit, hit_box, paint_image_with_boxes, to_image};

impl FoloApp {
    /// Side-by-side dual canvas stage for Playground Compare A/B mode with synchronized zoom and pan.
    pub(super) fn compare_stage(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let img = self.play_image.as_ref().unwrap();
        let dims = (img.width as f32, img.height as f32);
        let image = img.image.clone();
        let name = img.name.clone();
        let size_text = format!("{} × {}", img.width, img.height);
        let (_, file) = match name.rsplit_once(" / ") {
            Some((p, f)) => (Some(p.to_string()), f.to_string()),
            None => (None, name.to_string()),
        };

        let active_a = self.play_active(window, cx);
        let active_b = self.play_active_b(window, cx);
        let sim = self.play_target.is_none() && !self.store.api.local_inference();

        let model_a_name = self
            .play_shown_model()
            .map(|m| m.name.clone())
            .or_else(|| self.play_model.clone())
            .unwrap_or_else(|| "Model A".to_string());
        let model_b_name = self
            .play_shown_model_b()
            .map(|m| m.name.clone())
            .or_else(|| self.play_compare.model_b.clone())
            .unwrap_or_else(|| "Model B".to_string());

        let (boxes_a, count_a, lat_a, loading_a, err_a) = match &self.play_pred {
            Load::Ready(p) => {
                let b = self.visible_boxes(p, active_a);
                let n = b.len();
                let mut lat = latency_text(p.timing.total_ms, sim);
                if p.cold {
                    lat.push_str(" (cold)");
                }
                (b, Some(n), Some(lat), self.play_pred_pending, None)
            }
            Load::Loading => (Vec::new(), None, None, true, None),
            Load::Failed(e) => (Vec::new(), None, None, false, Some(e.message.clone())),
            Load::Idle => (Vec::new(), None, None, false, None),
        };
        let (boxes_b, count_b, lat_b, loading_b, err_b) = match &self.play_compare.pred_b {
            Load::Ready(p) => {
                let b = self.visible_boxes_b(p, active_b);
                let n = b.len();
                let mut lat = latency_text(p.timing.total_ms, sim);
                if p.cold {
                    lat.push_str(" (cold)");
                }
                (b, Some(n), Some(lat), self.play_compare.pred_b_pending, None)
            }
            Load::Loading => (Vec::new(), None, None, true, None),
            Load::Failed(e) => (Vec::new(), None, None, false, Some(e.message.clone())),
            Load::Idle => (Vec::new(), None, None, false, None),
        };

        // Clamp shared pan against Pane A's bounds (Pane A and Pane B share identical geometry).
        if let Some(b) = self.play_view.get() {
            self.play_pan =
                crate::viz::clamp_pan(b, dims.0, dims.1, Fit::Contain, self.play_zoom, self.play_pan);
        }
        let zoom = self.play_zoom;
        let pan = self.play_pan;
        if zoom <= 1.0 && self.fh("play-fit", cx).is_focused(window) {
            window.focus(&self.fh("play-zoom-in", cx));
        }

        let view_help = self.help_icon(
            "play-view-help",
            "Synchronized A/B view · scroll, + or − zoom · drag or arrows pan · 0 fits",
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
                    this.play_drag = None;
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

        let stage_w = {
            let rail = if window.viewport_size().width < px(1100.) { 280. } else { 320. };
            (self.content_width(window) - rail - 16. - 2.).max(0.)
        };
        let show_synced_badge = stage_w >= 540.;
        let show_size = stage_w >= 460.;
        let file_tip = format!("{name} · {size_text}");
        let toolbar = div()
            .h(px(40.))
            .flex_none()
            .px(px(12.))
            .flex()
            .items_center()
            .gap(px(8.))
            .bg(t.bg_panel)
            .border_b_1()
            .border_color(t.line_subtle)
            .child(div().flex_none().child(icon("icons/git-compare-arrows.svg", 14., t.accent_text)))
            .child(
                div()
                    .id("play-cmp-file-name")
                    .min_w_0()
                    .flex_shrink()
                    .tooltip(crate::controls::tip(file_tip, None))
                    .child(
                        text(ty::SM, t.text_primary)
                            .font_family(crate::theme::MONO)
                            .min_w_0()
                            .ellipsis()
                            .child(file),
                    ),
            )
            .when(show_size, |d| d.child(text(ty::SM, t.text_tertiary).flex_none().tnum().child(size_text)))
            .child(div().flex_1().min_w(px(8.)))
            .when(show_synced_badge, |d| d.child(badge(&t, "Synced zoom", Tone::Neutral).flex_none()))
            .child(div().flex_none().w(px(1.)).h(px(16.)).bg(t.line_subtle))
            .child(zoom_bar.flex_none());

        let draw_a = if self.play_boxes { boxes_a } else { Vec::new() };
        let draw_b = if self.play_boxes { boxes_b } else { Vec::new() };
        let show_labels = self.play_labels;
        let dim_a = self.play_pred_dim && self.play_pred_pending;
        let dim_b = self.play_compare.pred_b_dim && self.play_compare.pred_b_pending;

        let pane_a = self.compare_pane(
            &t,
            false,
            &model_a_name,
            count_a,
            lat_a,
            loading_a,
            err_a,
            image.clone(),
            dims,
            draw_a,
            show_labels,
            dim_a,
            zoom,
            pan,
            self.play_view.clone(),
            window,
            cx,
        );
        let pane_b = self.compare_pane(
            &t,
            true,
            &model_b_name,
            count_b,
            lat_b,
            loading_b,
            err_b,
            image,
            dims,
            draw_b,
            show_labels,
            dim_b,
            zoom,
            pan,
            self.play_compare.view_b.clone(),
            window,
            cx,
        );

        let batch_strip =
            self.play_batch.is_active().then(|| self.play_batch_filmstrip(&t, stage_w, window, cx));

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
            .child(toolbar)
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .child(pane_a)
                    .child(div().w(px(1.)).h_full().bg(t.line_default).flex_none())
                    .child(pane_b),
            )
            .children(self.play_batch_failed_banner(&t, window, cx))
            .children(batch_strip)
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn compare_pane(
        &mut self,
        t: &Theme,
        is_b: bool,
        model_name: &str,
        count: Option<usize>,
        latency: Option<String>,
        loading: bool,
        error: Option<String>,
        image: Arc<RenderImage>,
        dims: (f32, f32),
        boxes: Vec<DrawBox>,
        show_labels: bool,
        dim: bool,
        zoom: f32,
        pan: (f32, f32),
        bounds_cell: Rc<Cell<Option<Bounds<Pixels>>>>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let tt = *t;
        let side_label = if is_b { "B" } else { "A" };
        let side_tone = if is_b { Tone::Info } else { Tone::Accent };
        let canvas_id: &'static str = if is_b { "stage-canvas-b" } else { "stage-canvas" };
        let canvas_fh = self.fh(canvas_id, cx);
        let canvas_focused = canvas_fh.is_focused(window);
        let canvas_ring = self.ring(&canvas_fh, window);

        let status_text = if loading {
            "Detecting…".to_string()
        } else if error.is_some() {
            "Failed".to_string()
        } else {
            let mut parts = Vec::new();
            if let Some(n) = count {
                parts.push(if n == 1 { "1 det".to_string() } else { format!("{n} dets") });
            }
            if let Some(lat) = latency {
                parts.push(lat);
            }
            parts.join(" · ")
        };

        let header_id: &'static str = if is_b { "play-pane-b-model" } else { "play-pane-a-model" };
        let header_tip = if status_text.is_empty() {
            format!("{side_label} · {model_name}")
        } else {
            format!("{side_label} · {model_name} · {status_text}")
        };
        let header = div()
            .id(header_id)
            .h(px(30.))
            .flex_none()
            .px(px(10.))
            .bg(t.bg_raised)
            .border_b_1()
            .border_color(t.line_subtle)
            .flex()
            .items_center()
            .gap(px(6.))
            .tooltip(crate::controls::tip(header_tip, None))
            .child(badge(t, side_label, side_tone).flex_none())
            .child(
                text(ty::SM, t.text_primary)
                    .font_weight(FontWeight::MEDIUM)
                    .flex_1()
                    .min_w_0()
                    .ellipsis()
                    .child(model_name.to_string()),
            )
            .child(
                text(ty::SM, if error.is_some() { t.danger_text } else { t.text_tertiary })
                    .flex_none()
                    .tnum()
                    .child(status_text),
            );

        // Overlay error banner inside the canvas container so both panes always keep identical
        // height and fit geometry for synchronized zoom and pan.
        let retry_overlay = error.map(|msg| {
            let btn_key = if is_b { "play-pane-b-retry" } else { "play-pane-a-retry" };
            let retry = self.button(
                btn_key,
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
                .absolute()
                .top(px(8.))
                .left(px(8.))
                .right(px(8.))
                .px(px(10.))
                .py(px(6.))
                .rounded(px(radius::MD))
                .bg(t.danger_muted)
                .border_1()
                .border_color(t.line_default)
                .on_mouse_down(folo_ui::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .flex()
                .items_center()
                .justify_between()
                .gap(px(8.))
                .child(text(ty::SM, t.danger_text).flex_1().min_w_0().ellipsis().child(msg))
                .child(retry)
        });

        let paint_cell = bounds_cell.clone();
        let wheel_cell = bounds_cell.clone();
        let hover_boxes = boxes.clone();
        let paint_boxes = boxes;

        div()
            .flex_1()
            .min_w_0()
            .h_full()
            .flex()
            .flex_col()
            .child(header)
            .child(
                div()
                    .id(SharedString::from(canvas_id))
                    .track_focus(&canvas_fh)
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .bg(t.bg_inset)
                    .cursor(if zoom > 1.0 {
                        folo_ui::CursorStyle::OpenHand
                    } else {
                        folo_ui::CursorStyle::Arrow
                    })
                    .on_scroll_wheel(cx.listener(move |this, ev: &folo_ui::ScrollWheelEvent, w, cx| {
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
                        this.play_drag = None;
                        w.prevent_default();
                        cx.stop_propagation();
                        cx.notify();
                    }))
                    .on_mouse_down(
                        folo_ui::MouseButton::Left,
                        cx.listener(move |this, ev: &folo_ui::MouseDownEvent, w, _| {
                            w.focus(&canvas_fh);
                            this.play_drag = Some((ev.position, this.play_pan));
                        }),
                    )
                    .on_mouse_up(
                        folo_ui::MouseButton::Left,
                        cx.listener(|this, _, _, _| this.play_drag = None),
                    )
                    .on_mouse_up_out(
                        folo_ui::MouseButton::Left,
                        cx.listener(|this, _, _, _| this.play_drag = None),
                    )
                    .on_mouse_move(cx.listener(move |this, ev: &MouseMoveEvent, _, cx| {
                        if let Some(drag) = this.play_drag {
                            if ev.pressed_button == Some(folo_ui::MouseButton::Left) {
                                if this.play_zoom > 1.0 {
                                    let (next_pan, next_drag) = super::stage::clamp_and_rebase_drag(
                                        drag,
                                        ev.position,
                                        bounds_cell.get(),
                                        dims,
                                        this.play_zoom,
                                    );
                                    this.play_pan = next_pan;
                                    this.play_drag = Some(next_drag);
                                    cx.notify();
                                    return;
                                }
                            } else {
                                this.play_drag = None;
                            }
                        }
                        let Some(b) = bounds_cell.get() else { return };
                        let hit = if b.contains(&ev.position) {
                            let (x, y) =
                                to_image(b, dims, Fit::Contain, this.play_zoom, this.play_pan, ev.position);
                            hit_box(&hover_boxes, x, y)
                        } else {
                            None
                        };
                        if is_b {
                            if hit != this.play_compare.hover_b
                                || (hit.is_some() && !this.play_compare.hover_b_from_stage)
                            {
                                this.play_compare.hover_b = hit;
                                this.play_compare.hover_b_from_stage = hit.is_some();
                                this.kb = false;
                                cx.notify();
                            }
                        } else if hit != this.play_hover || (hit.is_some() && !this.play_hover_from_stage) {
                            this.play_hover = hit;
                            this.play_hover_from_stage = hit.is_some();
                            this.kb = false;
                            cx.notify();
                        }
                    }))
                    .on_hover(cx.listener(move |this, h: &bool, _, cx| {
                        if !*h {
                            if is_b && this.play_compare.hover_b_from_stage {
                                this.play_compare.hover_b_from_stage = false;
                                if this.play_compare.hover_b.take().is_some() {
                                    cx.notify();
                                }
                            } else if !is_b && this.play_hover_from_stage {
                                this.play_hover_from_stage = false;
                                if this.play_hover.take().is_some() {
                                    cx.notify();
                                }
                            }
                        }
                    }))
                    .child(
                        div().size_full().when(dim, |d| d.opacity(0.5)).child(
                            canvas(
                                |_, _, _| (),
                                move |b, _, window, cx| {
                                    let pad_x = (b.size.width / 2.).min(px(10.));
                                    let pad_y = (b.size.height / 2.).min(px(10.));
                                    let inner = Bounds::new(
                                        b.origin + folo_ui::point(pad_x, pad_y),
                                        b.size - folo_ui::size(pad_x * 2., pad_y * 2.),
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
                                        show_labels,
                                        false,
                                        0.,
                                    );
                                },
                            )
                            .size_full(),
                        ),
                    )
                    .when(canvas_focused, |d| {
                        d.child(
                            div()
                                .absolute()
                                .inset_0()
                                .when(canvas_ring, |d| d.border_2())
                                .when(!canvas_ring, |d| d.border_1())
                                .border_color(t.focus),
                        )
                    })
                    .when(canvas_ring, |d| {
                        self.hint_keys(if zoom > 1.0 {
                            "Synced A/B · Arrows pan · Shift ×4 · + − zoom · 0 fits"
                        } else {
                            "Synced A/B · + − zoom · arrows pan when zoomed · 0 fits"
                        });
                        d
                    })
                    .children(retry_overlay),
            )
            .into_any_element()
    }
}
