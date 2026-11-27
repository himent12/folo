//! Pointer drags that continue outside the element they started on: sliders, scroll thumbs and
//! image pans.

use folo_ui::point;
use folo_ui::{
    AnyElement, Context, DispatchPhase, IntoElement, MouseButton, MouseMoveEvent, MouseUpEvent, Styled,
    Window, canvas, px,
};

use super::FoloApp;

impl FoloApp {
    /// Window-wide drag and pointer-motion listener: receives move and release events even while an
    /// `.occlude()` overlay (a dialog card, scrim, or inspector) covers `root`, so dialog sliders
    /// and scrollbars drag smoothly across and outside their tracks.
    pub(crate) fn drag_tracker(cx: &mut Context<Self>) -> AnyElement {
        let view = cx.entity().downgrade();
        canvas(
            |_, _, _| {},
            move |_, _, window, _| {
                let move_view = view.clone();
                window.on_mouse_event(move |ev: &MouseMoveEvent, phase, window, cx| {
                    if phase == DispatchPhase::Capture {
                        let _ = move_view.update(cx, |this: &mut Self, cx| {
                            this.on_drag_move(ev, window, cx);
                        });
                    }
                });
                let up_view = view.clone();
                window.on_mouse_event(move |ev: &MouseUpEvent, phase, window, cx| {
                    if phase == DispatchPhase::Capture && ev.button == MouseButton::Left {
                        let _ = up_view.update(cx, |this: &mut Self, cx| {
                            this.on_drag_end(ev, window, cx);
                        });
                    }
                });
            },
        )
        .absolute()
        .into_any_element()
    }

    // ---------- sliders ----------

    pub fn on_drag_move(&mut self, ev: &MouseMoveEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if !self.pointer_moved {
            self.pointer_moved = true;
            cx.notify();
        }
        // A scroll thumb being dragged: move the content by the thumb's travel ratio.
        if self.scroll_drag.is_some() {
            if ev.pressed_button != Some(MouseButton::Left) {
                self.scroll_drag = None;
                cx.notify();
                return;
            }
            if let Some((handle, y0, off0)) = self.scroll_drag.as_ref()
                && let Some(g) = crate::ui::ThumbGeo::of(handle)
            {
                let travel = (g.h - g.thumb).max(px(1.));
                let to = (*off0 + (ev.position.y - *y0) * (g.max / travel)).clamp(px(0.), g.max);
                handle.set_offset(point(handle.offset().x, -to));
                cx.notify();
            }
            return;
        }
        if self.label.editor.drawing.is_some() {
            if ev.pressed_button != Some(MouseButton::Left) {
                self.finish_label_box_draw(cx);
            } else {
                self.update_label_box_draw(ev.position, cx);
            }
            return;
        }
        let Some(drag) = self.drag.as_ref() else { return };
        if ev.pressed_button != Some(MouseButton::Left) {
            if let Some(d) = self.drag.take() {
                if d.id == "play-iou" {
                    self.run_predict(cx);
                    if self.play_compare.enabled {
                        self.run_predict_b(cx);
                    }
                }
                cx.notify();
            }
            return;
        }
        let Some(b) = self.slider_bounds.borrow().get(drag.id).copied() else { return };
        if b.size.width <= px(0.) {
            return;
        }
        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
        let id = drag.id;
        self.set_slider(id, frac, cx);
    }

    pub(super) fn on_drag_end(&mut self, ev: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if self.scroll_drag.take().is_some() {
            cx.notify();
        }
        // An image pan ends wherever the button comes up (also outside its canvas).
        if self.play_drag.take().is_some() {
            cx.notify();
        }
        if let Some(i) = self.inspector.as_mut()
            && i.drag_from.take().is_some()
        {
            cx.notify();
        }
        if self.label.editor.drawing.is_some() {
            self.update_label_box_draw(ev.position, cx);
            self.finish_label_box_draw(cx);
        }
        if let Some(d) = self.drag.take() {
            if d.id == "play-iou" {
                self.run_predict(cx);
                if self.play_compare.enabled {
                    self.run_predict_b(cx);
                }
            }
            cx.notify();
        }
    }
}
