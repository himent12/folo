//! Sliders.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Context, Div, InteractiveElement, KeyDownEvent, MouseButton, MouseDownEvent, ParentElement, Stateful,
    Styled, Window, canvas, div, px, relative,
};

use crate::app::{Drag, FoloApp, focus_ring};
use crate::theme::theme;

impl FoloApp {
    /// A horizontal slider over `[min, max]`. Drag, click, or use arrows / Home / End.
    #[allow(clippy::too_many_arguments)]
    pub fn slider(
        &self,
        key: &'static str,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let t = theme(cx);
        let dragging = self.drag.as_ref().is_some_and(|d| d.id == key);
        // Mid-drag the range holds still (a widened range would otherwise shrink with the value
        // under the pointer and the thumb run away from it).
        let (min, max) = match self.slider_ranges.borrow().get(key) {
            Some(&r) if dragging => r,
            _ => (min, max),
        };
        self.slider_ranges.borrow_mut().insert(key, (min, max));
        let fh = self.fh(key, cx);
        let ring = self.ring(&fh, window);
        let span = max - min;
        let f = if span.is_finite() && span > 0.0 && value.is_finite() {
            ((value - min) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let store = self.slider_bounds.clone();
        let fill_color = t.accent_fill();
        div()
            .id(key)
            .track_focus(&fh)
            .relative()
            .w_full()
            // A 28 px target in the 20 px it takes in the layout (the margin gives 8 px back).
            .h(px(28.))
            .my(px(-4.))
            .cursor_pointer()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &MouseDownEvent, window, cx| {
                    window.focus(&this.fh(key, cx));
                    this.drag = Some(Drag { id: key });
                    let b = this.slider_bounds.borrow().get(key).copied();
                    if let Some(b) = b
                        && b.size.width > px(0.)
                    {
                        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
                        this.set_slider(key, frac, cx);
                    }
                    cx.notify();
                }),
            )
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, _, cx| {
                let m = &ev.keystroke.modifiers;
                if m.control || m.alt || m.platform {
                    return;
                }
                if !span.is_finite() || span <= 0.0 {
                    return;
                }
                let big = m.shift;
                let s = step / span * if big { 10.0 } else { 1.0 };
                let next = match ev.keystroke.key.as_str() {
                    "left" | "down" => Some(f - s),
                    "right" | "up" => Some(f + s),
                    "home" => Some(0.0),
                    "end" => Some(1.0),
                    _ => None,
                };
                if let Some(n) = next {
                    let prev_iou = this.play_iou;
                    this.set_slider(key, n.clamp(0.0, 1.0), cx);
                    if key == "play-iou" && (this.play_iou - prev_iou).abs() > f32::EPSILON {
                        this.on_play_iou_changed(cx);
                    }
                    cx.stop_propagation();
                }
            }))
            .child(
                canvas(
                    move |b, _, _| {
                        store.borrow_mut().insert(key, b);
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .left(px(7.))
                .right(px(7.))
                .top_0()
                .bottom_0(),
            )
            // The track spans the full column (aligned with the fields above); the thumb's centre
            // travels 7 px inside it, so the fill is a 7 px lead plus its share of the travel.
            .child(
                div()
                    .absolute()
                    .left_0()
                    .right_0()
                    .top(px(12.5))
                    .h(px(3.))
                    .rounded_full()
                    // The rail is the control's only mark: line.control (3:1), not a separator line.
                    .bg(t.line_control)
                    // 8 px: a px under the proportional fill, so no seam shows where the two meet.
                    .child(
                        div().absolute().left_0().top_0().h_full().w(px(8.)).rounded_l_full().bg(fill_color),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(px(7.))
                            .right(px(7.))
                            .top_0()
                            .h_full()
                            .child(div().h_full().w(relative(f)).bg(fill_color)),
                    ),
            )
            .child(
                div().absolute().left(px(7.)).right(px(7.)).top_0().bottom_0().child(
                    div()
                        .absolute()
                        .left(relative(f))
                        .ml(px(-7.))
                        .top(px(7.))
                        .size(px(14.))
                        .rounded_full()
                        .bg(if dragging { t.accent_fill() } else { folo_ui::white() })
                        .border_2()
                        .border_color(t.accent_fill())
                        .shadow(t.shadow_sm())
                        .hover(|s| {
                            s.shadow(t.shadow_md()).bg(crate::theme::over(folo_ui::white(), t.accent_muted))
                        })
                        .when(ring, |d| d.child(focus_ring(&t, 7.))),
                ),
            )
    }

    /// Map a slider fraction to its value and store it.
    pub fn set_slider(&mut self, key: &str, frac: f32, cx: &mut Context<Self>) {
        let q = |v: f32, step: f32| (v / step).round() * step;
        // The range the slider was drawn with (it may have widened to a typed value), else its
        // preset.
        let range = self.slider_ranges.borrow().get(key).copied();
        let at = |lo: f32, hi: f32| {
            let (lo, hi) = range.unwrap_or((lo, hi));
            lo + frac * (hi - lo)
        };
        // Quantised, except the ends stay exact (a typed 1 epoch or 0.0001 learning rate is an
        // end of a widened range: rounding it to the step would make it invalid).
        let snap = |v: f32, step: f32| {
            if frac <= 0. || frac >= 1. { v } else { q(v, step) }
        };
        match key {
            // Quantised but never below the slider's own low end (a widened range keeps it exact).
            "play-conf" => {
                let lo = range.map_or(0.05, |r| r.0);
                let v = q(at(0.05, 0.95), 0.01).max(lo);
                self.set_play_conf(v, cx);
            }
            "play-iou" => self.play_iou = q(at(0.3, 0.95), 0.01),
            "sample-conf" => {
                let (lo, hi, step) = crate::screens::runs::SAMPLE_CONF;
                self.sample_conf = q(at(lo, hi), step).clamp(lo, hi);
            }
            _ => {
                if let Some(f) = self.new_run.as_mut() {
                    match key {
                        "nr-epochs" => {
                            f.epochs = snap(at(10., 300.), 5.).round().clamp(1., 10_000.);
                            f.epochs_invalid = false;
                            let (input, v) = (f.epochs_input.clone(), format!("{:.0}", f.epochs));
                            input.update(cx, |i, cx| i.set_text(v, cx));
                            // Close mosaic follows while it is the default, else is rechecked.
                            if let Some(v) = f.follow_epochs() {
                                f.close_input.clone().update(cx, |i, cx| i.set_text(v, cx));
                            } else if !f.close_auto {
                                let close = f.close_input.read(cx).text();
                                f.check_close(&close);
                            }
                        }
                        "nr-lr0" => {
                            // Above 0 and below 10, the range the field accepts.
                            f.lr0 = snap(at(0.001, 0.05), 0.001).clamp(1e-6, 9.9999);
                            let (input, v) = (f.lr_input.clone(), crate::app::format_lr(f.lr0));
                            input.update(cx, |i, cx| i.set_text(v, cx));
                        }
                        "nr-mosaic" => f.mosaic = q(at(0., 1.), 0.05),
                        "nr-mixup" => f.mixup = q(at(0., 0.5), 0.05),
                        "nr-fliplr" => f.fliplr = q(at(0., 1.), 0.05),
                        "nr-close" => {
                            f.close_auto = false;
                            f.close_mosaic = snap(at(0., 30.), 1.).round().max(0.);
                            let (input, v) = (f.close_input.clone(), format!("{:.0}", f.close_mosaic));
                            input.update(cx, |i, cx| i.set_text(v.clone(), cx));
                            f.check_close(&v);
                        }
                        "nr-scale" => f.scale = q(at(0., 0.9), 0.05),
                        _ => {}
                    }
                }
            }
        }
        cx.notify();
    }
}
