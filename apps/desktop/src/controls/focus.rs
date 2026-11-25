//! Focus and activation: making an element a Tab stop that a click, Enter and Space activate.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    ClickEvent, Context, Div, InteractiveElement, KeyDownEvent, ParentElement, Stateful,
    StatefulInteractiveElement, Styled, Window,
};

use crate::app::{Action, FoloApp, focus_ring};
use crate::theme::theme;

impl FoloApp {
    /// Make `el` focusable and activatable by click, Enter and Space.
    pub fn bind(
        &self,
        el: Stateful<Div>,
        key: &str,
        radius_px: f32,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let t = theme(cx);
        self.bind_ring(el, key, radius_px, t.focus, window, cx, action)
    }

    /// [`Self::bind`] with a custom ring colour (accent-filled controls ring in the text colour,
    /// since a lime ring around a lime button reads as a thicker button, not focus).
    #[allow(clippy::too_many_arguments)]
    pub fn bind_ring(
        &self,
        el: Stateful<Div>,
        key: &str,
        radius_px: f32,
        ring_color: folo_ui::Hsla,
        window: &Window,
        cx: &mut Context<Self>,
        action: Action,
    ) -> Stateful<Div> {
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        let t = theme(cx);
        let on_click = action.clone();
        let click_fh = fh.clone();
        el.track_focus(&fh)
            .relative()
            .on_click(cx.listener(move |this, ev: &ClickEvent, window, cx| {
                // Enter and Space act on the press (below): the release's keyboard click would act twice.
                if ev.is_keyboard() {
                    return;
                }
                // The clicked control takes focus first (keys then act on it, not on whatever had
                // focus before); the action may still move focus on, as dialogs do.
                window.focus(&click_fh);
                on_click(this, window, cx);
                cx.stop_propagation();
                cx.notify();
            }))
            .on_key_down(cx.listener(move |this, ev: &KeyDownEvent, window, cx| {
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") && !ev.keystroke.modifiers.control {
                    action(this, window, cx);
                    cx.stop_propagation();
                    cx.notify();
                }
            }))
            .when(ring, |d| d.child(focus_ring(&t, radius_px).border_color(ring_color)))
    }
}
