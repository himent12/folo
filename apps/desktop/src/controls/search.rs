//! Inline search field with a leading search icon, focus ring, and a clear button when non-empty.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Entity, Focusable, InteractiveElement, IntoElement, MouseButton, MouseDownEvent,
    ParentElement, SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, act, focus_ring};
use crate::input::TextInput;
use crate::theme::{radius, size as tsize, theme, ty};
use crate::ui::icon;

impl FoloApp {
    /// A single-line search input for list and log toolbars, with a clear (`×`) button while
    /// non-empty. `key` prefixes the clear button's focus key (`{key}-clear`).
    pub fn search_field(
        &self,
        key: &str,
        input: Entity<TextInput>,
        width: f32,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = input.read(cx).focus_handle(cx);
        let ring = self.ring(&fh, window);
        let has_text = !input.read(cx).text().is_empty();
        let clear = has_text.then(|| {
            let to_clear = input.clone();
            let clear_key = format!("{key}-clear");
            let clear_ring = self.ring(&self.fh(clear_key.clone(), cx), window);
            let el = div()
                .id(SharedString::from(clear_key.clone()))
                .w(px(20.))
                .h(px(20.))
                .flex()
                .flex_none()
                .items_center()
                .justify_center()
                .rounded(px(radius::SM))
                .cursor_pointer()
                .hover(|s| s.bg(t.bg_hover))
                .tooltip(super::tip("Clear search", None))
                .when(clear_ring, |d| d.child(super::focus_tip(&t, "Clear search", 20.)))
                .child(icon("icons/x.svg", 12., t.text_tertiary));
            self.bind(
                el,
                &clear_key,
                radius::SM,
                window,
                cx,
                act(move |_, w, cx| {
                    to_clear.update(cx, |i, cx| i.set_text("", cx));
                    w.focus(&to_clear.read(cx).focus_handle(cx));
                    cx.notify();
                }),
            )
            .into_any_element()
        });
        let focus_fh = fh.clone();
        div()
            .relative()
            .w(px(width))
            .h(px(tsize::CONTROL_MD))
            .px(px(10.))
            .flex()
            .flex_none()
            .items_center()
            .gap(px(6.))
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if fh.is_focused(window) { t.accent_line } else { t.line_control })
            .cursor_text()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |_, _: &MouseDownEvent, w, cx| {
                    w.focus(&focus_fh);
                    cx.notify();
                }),
            )
            .when(ring, |d| d.child(focus_ring(&t, radius::MD)))
            .child(icon("icons/search.svg", 14., t.text_tertiary))
            .child(div().flex_1().min_w_0().text_size(px(ty::SM.size)).child(input))
            .when_some(clear, |d, c| d.child(c))
            .into_any_element()
    }
}
