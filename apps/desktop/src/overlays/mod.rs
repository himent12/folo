//! Overlays drawn above the page: command palette, dialogs, the image inspector and toasts.

mod cloud_dialogs;
mod confirm_delete;
mod confirm_stop;
mod inspector;
mod new_run;
mod new_run_compute;
mod new_run_footer;
mod palette;
mod synth;
mod toasts;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Div, FontWeight, InteractiveElement, IntoElement, MouseButton, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{Action, FoloApp};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

pub use new_run_footer::estimate;

/// A dialog's footer: its actions on the raised band under a hairline.
fn dialog_footer(t: &Theme) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .px(px(20.))
        .py(px(14.))
        .border_t_1()
        .border_color(t.line_subtle)
        .bg(t.bg_raised)
        // The frame's 12 px corners, less its 1 px edge: a square fill would show past them.
        .rounded_b(px(radius::XL - 1.))
}

/// A refusal inside a dialog: the server's message, and its hint (what to change) under it.
fn error_note(t: &Theme, message: String, hint: Option<String>) -> Div {
    div()
        .flex()
        .items_start()
        .gap(px(8.))
        .p(px(10.))
        .rounded(px(radius::MD))
        .bg(t.danger_muted)
        .child(icon("icons/circle-alert.svg", 14., t.danger_solid))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(2.))
                .min_w_0()
                .child(text(ty::SM, t.danger_text).whitespace_normal().child(message))
                .when_some(hint, |d, h| d.child(text(ty::SM, t.text_secondary).whitespace_normal().child(h))),
        )
}

impl FoloApp {
    pub fn overlays(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Vec<AnyElement> {
        let mut out = Vec::new();
        if self.inspector.is_some() {
            out.push(self.inspector_view(window, cx));
        }
        if self.new_run.is_some() {
            out.push(self.new_run_dialog(window, cx));
        }
        if self.synth.is_some() {
            out.push(self.synth_dialog(window, cx));
        }
        if let Some(id) = self.confirm_stop.clone() {
            out.push(self.stop_dialog(&id, window, cx));
        }
        if let Some(target) = self.confirm_delete.clone() {
            out.push(self.delete_dialog(&target, window, cx));
        }
        if self.cloud.connect.is_some() {
            out.push(self.connect_dialog(window, cx));
        }
        if self.cloud.disconnect.is_some() {
            out.push(self.disconnect_dialog(window, cx));
        }
        if self.cloud.terminate.is_some() {
            out.push(self.terminate_dialog(window, cx));
        }
        if self.palette.is_some() {
            out.push(self.palette_view(window, cx));
        }
        if !self.toasts.is_empty() {
            out.push(self.toasts_view(window, cx));
        }
        out
    }

    /// The dimmed backdrop under an overlay's panel: a press on it calls `on_click` (dismisses).
    /// The window's caption areas go over it (`caption_controls`), since it hides the title bar's.
    fn scrim(
        &self,
        t: &Theme,
        id: &'static str,
        window: &Window,
        cx: &mut Context<Self>,
        on_click: Action,
    ) -> folo_ui::Stateful<Div> {
        div()
            .id(id)
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .child(div().id("fill").size_full().bg(t.bg_scrim).occlude().on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, _, w, cx| {
                    on_click(this, w, cx);
                    cx.notify();
                }),
            ))
            .child(self.caption_controls(window, cx))
    }

    fn dialog_frame(
        &self,
        t: &Theme,
        id: &'static str,
        width: f32,
        title: &str,
        subtitle: Option<String>,
        body: impl IntoElement,
        footer: impl IntoElement,
        window: &Window,
        cx: &mut Context<Self>,
        on_close: Action,
    ) -> AnyElement {
        let close_btn =
            self.icon_button("dialog-close", "icons/x.svg", "Close (Esc)", window, cx, on_close.clone());
        // From the last layout, as the run log's follow state is.
        // Scrolled: a hairline under the fixed header, so rows passing beneath it read as scrolled
        // away, not cut.
        let scrolled = self.dialog_scroll.offset().y < px(-1.);
        crate::ui::appear(
            div()
                .absolute()
                .top_0()
                .left_0()
                .size_full()
                .child(self.scrim(t, "dialog-scrim", window, cx, on_close))
                .child(
                    div()
                        .absolute()
                        .top_0()
                        .left_0()
                        .size_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .child(
                            div()
                                .id(id)
                                .w(px(width))
                                .max_h(folo_ui::relative(0.92))
                                .occlude()
                                .flex()
                                .flex_col()
                                .rounded(px(radius::XL))
                                .bg(t.bg_overlay)
                                .border_1()
                                .border_color(t.line_default)
                                .shadow(t.shadow_lg())
                                .overflow_hidden()
                                .child(
                                    div()
                                        .flex()
                                        .items_start()
                                        .justify_between()
                                        .px(px(20.))
                                        .pt(px(18.))
                                        .pb(px(14.))
                                        .child(
                                            // A long name in the title ends in "…" short of the close
                                            // button (the body names it in full), never cut mid-glyph.
                                            div()
                                                .flex_1()
                                                .min_w_0()
                                                .flex()
                                                .flex_col()
                                                .gap(px(4.))
                                                .child(
                                                    text(ty::XL, t.text_primary)
                                                        .font_weight(FontWeight::SEMIBOLD)
                                                        .ellipsis()
                                                        .child(title.to_string()),
                                                )
                                                .when_some(subtitle, |d, s| {
                                                    d.child(text(ty::SM, t.text_secondary).child(s))
                                                }),
                                        )
                                        .child(close_btn),
                                )
                                .child(
                                    div()
                                        .relative()
                                        .flex_1()
                                        .min_h_0()
                                        .flex()
                                        .flex_col()
                                        .border_t_1()
                                        .border_color(if scrolled {
                                            t.line_subtle
                                        } else {
                                            folo_ui::transparent_black()
                                        })
                                        .child(
                                            div()
                                                .id("dialog-body")
                                                .flex_1()
                                                .min_h_0()
                                                .overflow_y_scroll()
                                                .track_scroll(&self.dialog_scroll)
                                                .child(body),
                                        )
                                        // Fades only where more of the form lies (a row half under
                                        // the header reads as scrolled away; the ends show undimmed).
                                        .children(self.scroll_chrome(
                                            self.dialog_scroll.clone(),
                                            t.bg_overlay,
                                            0.,
                                            0.,
                                            cx,
                                        )),
                                )
                                .child(footer),
                        ),
                ),
            &format!("{id}-in"),
        )
    }
}
