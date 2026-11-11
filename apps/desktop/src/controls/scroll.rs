//! Scroll chrome: the draggable thumb and the edge fades.

use folo_ui::{
    AnyElement, Context, Hsla, InteractiveElement, IntoElement, MouseButton, ParentElement, Styled, div, px,
};

use crate::app::FoloApp;
use crate::theme::theme;

impl FoloApp {
    /// A scroll thumb for a scroll container: drawn from its handle, draggable, and a click on
    /// the track jumps there. Place it after the container inside a relative parent.
    pub fn scrollbar(&self, handle: folo_ui::ScrollHandle, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let paint = handle.clone();
        let hot = self.scroll_drag.as_ref().is_some_and(|(h, _, _)| h.bounds() == handle.bounds());
        div()
            .absolute()
            .top_0()
            .bottom_0()
            .right_0()
            .w(px(12.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, ev: &folo_ui::MouseDownEvent, _, cx| {
                    let Some(g) = crate::ui::ThumbGeo::of(&handle) else { return };
                    if ev.position.y < g.y || ev.position.y > g.y + g.thumb {
                        // Track click: centre the thumb on the pointer.
                        let travel = (g.h - g.thumb).max(px(1.));
                        let at = ((ev.position.y - g.top - g.thumb / 2.) / travel).clamp(0., 1.);
                        handle.set_offset(folo_ui::point(handle.offset().x, -(g.max * at)));
                    }
                    this.scroll_drag = Some((handle.clone(), ev.position.y, -handle.offset().y));
                    cx.stop_propagation();
                    cx.notify();
                }),
            )
            .child(
                folo_ui::canvas(
                    |_, _, _| (),
                    move |_, _, window, cx| crate::ui::paint_thumb(&t, &paint, hot, window, cx),
                )
                .size_full(),
            )
            .into_any_element()
    }

    /// A scroll area's chrome, for the `relative` parent of its scrolling element: the edge fades
    /// (`ui::scroll_edges`) and the overlay thumb. Every scroll area, the page included, uses it.
    pub fn scroll_chrome(
        &self,
        handle: folo_ui::ScrollHandle,
        bg: Hsla,
        inset: f32,
        radius: f32,
        cx: &mut Context<Self>,
    ) -> Vec<AnyElement> {
        let mut v: Vec<AnyElement> =
            crate::ui::scroll_edges(&handle, bg, inset, radius).map(IntoElement::into_any_element).collect();
        v.push(self.scrollbar(handle, cx));
        v
    }
}
