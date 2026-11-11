//! Tooltips: hover tips, focus tips, help icons and the disclosure of cut labels.

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, AnyView, AppContext, Context, Div, InteractiveElement, IntoElement, ParentElement, Render,
    SharedString, Stateful, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{FoloApp, focus_ring};
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;

/// Hover tooltip: one short line on the overlay surface.
pub struct Tip(pub SharedString, pub Option<SharedString>);

impl Render for Tip {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = theme(cx);
        div().p(px(4.)).child(
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .px(px(8.))
                .py(px(5.))
                .rounded(px(radius::MD))
                .bg(t.bg_overlay)
                .border_1()
                .border_color(t.line_default)
                .shadow(t.shadow_md())
                .child(text(ty::SM, t.text_primary).max_w(px(280.)).child(self.0.clone()))
                .when_some(self.1.clone(), |d, k| d.child(text(ty::SM, t.text_tertiary).child(k))),
        )
    }
}

pub fn tip(
    text: impl Into<SharedString>,
    keys: Option<&'static str>,
) -> impl Fn(&mut Window, &mut gpui::App) -> AnyView + 'static {
    let text: SharedString = text.into();
    move |_, cx| cx.new(|_| Tip(text.clone(), keys.map(Into::into))).into()
}

impl FoloApp {
    /// An info glyph that explains a control: tooltip on hover, and a Tab stop that shows the same
    /// text in a popover while keyboard-focused.
    pub fn help_icon(
        &self,
        key: &str,
        help: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let fh = self.fh(key.to_string(), cx);
        let shown = self.ring(&fh, window);
        let el = div()
            .id(SharedString::from(key.to_string()))
            .relative()
            // A 28 px target in the 20 px it takes in the row (the margin gives the 8 px back).
            .size(px(28.))
            .m(px(-4.))
            .flex_none()
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(radius::SM))
            .cursor_default()
            .hover(|s| s.bg(t.bg_hover))
            .tooltip(tip(help, None))
            .child(icon("icons/info.svg", 12., t.text_tertiary))
            .when(shown, |d| {
                // Anchored under the icon and kept 8 px inside the window (an icon near the right
                // edge would otherwise push it out), like `focus_tip`.
                d.child(
                    div().absolute().top(px(28.)).left(px(-4.)).child(
                        gpui::deferred(
                            gpui::anchored().snap_to_window_with_margin(px(8.)).child(
                                div()
                                    .w(px(260.))
                                    .p(px(8.))
                                    .rounded(px(radius::MD))
                                    .bg(t.bg_overlay)
                                    .border_1()
                                    .border_color(t.line_default)
                                    .shadow(t.shadow_md())
                                    .child(text(ty::SM, t.text_primary).child(help)),
                            ),
                        )
                        .with_priority(1),
                    ),
                )
            });
        self.bind(el, key, radius::SM, window, cx, std::rc::Rc::new(|_, _, _| {})).into_any_element()
    }

    /// A cut label's disclosure: when `long`, its full text shows on hover (a tooltip) and on
    /// keyboard focus (a tip `below` px down). Rows that already take focus pass
    /// `tab_stop: false`; plain text passes `true` and becomes a Tab stop with its own ring, so
    /// the full text is never pointer-only. Short text gets nothing. Every cut label uses it.
    #[allow(clippy::too_many_arguments)]
    pub fn disclose(
        &self,
        el: Stateful<Div>,
        key: &str,
        full: impl Into<SharedString>,
        long: bool,
        below: f32,
        tab_stop: bool,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        if !long {
            return el;
        }
        let t = theme(cx);
        let full: SharedString = full.into();
        let fh = self.fh(key.to_string(), cx);
        let ring = self.ring(&fh, window);
        el.relative()
            .tooltip(tip(full.clone(), None))
            .when(tab_stop, |d| d.track_focus(&fh))
            .when(ring && tab_stop, |d| d.child(focus_ring(&t, radius::SM)))
            .when(ring, |d| d.child(focus_tip(&t, full, below)))
    }
}

/// A tooltip shown while a control has keyboard focus, below it and kept inside the window.
pub fn focus_tip(t: &Theme, text_: impl Into<SharedString>, below: f32) -> Div {
    let text_: SharedString = text_.into();
    // The box the tip is laid out in: it sizes to its text up to this width, then wraps (a narrow
    // control, like a rail icon, would otherwise wrap it word by word).
    div().absolute().top(px(below + 6.)).left_0().w(px(320.)).child(
        gpui::deferred(
            gpui::anchored().snap_to_window_with_margin(px(8.)).child(
                div()
                    .px(px(8.))
                    .py(px(4.))
                    .rounded(px(radius::SM))
                    .bg(t.bg_overlay)
                    .border_1()
                    .border_color(t.line_default)
                    .shadow(t.shadow_lg())
                    // Short tips stay on one line; long ones wrap at 320 px, not across the window.
                    .max_w(px(320.))
                    .child(text(ty::SM, t.text_secondary).whitespace_normal().child(text_)),
            ),
        )
        .with_priority(3),
    )
}
