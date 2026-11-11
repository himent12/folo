//! Pointer drags that continue outside the element they started on: sliders, scroll thumbs and
//! image pans.

use folo_ui::point;
use folo_ui::{Context, MouseMoveEvent, MouseUpEvent, Window, px};

use super::FoloApp;

impl FoloApp {
    // ---------- sliders ----------

    pub fn on_drag_move(&mut self, ev: &MouseMoveEvent, _window: &mut Window, cx: &mut Context<Self>) {
        // A scroll thumb being dragged: move the content by the thumb's travel ratio.
        if let Some((handle, y0, off0)) = self.scroll_drag.as_ref() {
            if let Some(g) = crate::ui::ThumbGeo::of(handle) {
                let travel = (g.h - g.thumb).max(px(1.));
                let to = (*off0 + (ev.position.y - *y0) * (g.max / travel)).clamp(px(0.), g.max);
                handle.set_offset(point(handle.offset().x, -to));
                cx.notify();
            }
            return;
        }
        let Some(drag) = self.drag.as_ref() else { return };
        let Some(b) = self.slider_bounds.borrow().get(drag.id).copied() else { return };
        let frac = ((ev.position.x - b.left()) / b.size.width).clamp(0.0, 1.0);
        let id = drag.id;
        self.set_slider(id, frac, cx);
    }

    pub(super) fn on_drag_end(&mut self, _: &MouseUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
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
        if let Some(d) = self.drag.take() {
            if d.id == "play-iou" {
                self.run_predict(cx);
            }
            cx.notify();
        }
    }
}
