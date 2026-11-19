//! Overlays drawn above the page: command palette, dialogs, the image inspector and toasts.

mod cloud_dialogs;
mod confirm_close;
mod confirm_delete;
mod confirm_stop;
mod inspector;
mod new_run;
mod new_run_compute;
mod new_run_footer;
mod new_run_weights;
mod palette;
mod synth;
mod toasts;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, App, Context, Div, FontWeight, InteractiveElement, IntoElement, MouseButton, MouseDownEvent,
    ParentElement, StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::app::{Action, FoloApp};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

pub use confirm_close::closing_stops;
pub use new_run_footer::estimate;

/// Whether the overlay on top has seen a press begin, so that the second press of a double-click
/// whose first press opened it is not taken as a press of its own: on the backdrop it would
/// dismiss the overlay at once, on the panel it could press what opened under the pointer (a
/// confirmation's Delete). The platform counts the clicks with the system's double-click time and
/// distance (`MouseDownEvent::click_count`), so no timer is kept here: a press that starts a new
/// click (count 1) is the overlay's own, and so is every later press of that click.
#[derive(Default)]
pub struct OverlayPresses {
    top: Option<&'static str>,
    began: bool,
}

impl OverlayPresses {
    /// The overlay now on top (each render): one that has just come up has seen no press.
    pub fn shown(&mut self, top: Option<&'static str>) {
        if self.top != top {
            self.top = top;
            self.began = false;
        }
    }

    /// Whether the overlay on top takes a press with this click count.
    pub fn admits(&mut self, click_count: usize) -> bool {
        if click_count <= 1 {
            self.began = true;
        }
        self.began
    }
}

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
        // The last one pushed is on top (toasts aside: they take no presses for it).
        let mut top = None;
        if self.inspector.is_some() {
            out.push(self.inspector_view(window, cx));
            top = Some("inspector");
        }
        if self.new_run.is_some() {
            out.push(self.new_run_dialog(window, cx));
            top = Some("new-run");
        }
        if self.synth.is_some() {
            out.push(self.synth_dialog(window, cx));
            top = Some("synth");
        }
        if let Some(id) = self.confirm_stop.clone() {
            out.push(self.stop_dialog(&id, window, cx));
            top = Some("confirm-stop");
        }
        if let Some(target) = self.confirm_delete.clone() {
            out.push(self.delete_dialog(&target, window, cx));
            top = Some("confirm-delete");
        }
        if let Some(names) = self.confirm_close.clone() {
            out.push(self.close_dialog(&names, window, cx));
            top = Some("confirm-close");
        }
        if self.cloud.connect.is_some() {
            out.push(self.connect_dialog(window, cx));
            top = Some("cloud-connect");
        }
        if self.cloud.disconnect.is_some() {
            out.push(self.disconnect_dialog(window, cx));
            top = Some("cloud-disconnect");
        }
        if self.cloud.terminate.is_some() {
            out.push(self.terminate_dialog(window, cx));
            top = Some("cloud-terminate");
        }
        if self.palette.is_some() {
            out.push(self.palette_view(window, cx));
            top = Some("palette");
        }
        self.overlay_presses.shown(top);
        if !self.toasts.is_empty() {
            out.push(self.toasts_view(window, cx));
        }
        out
    }

    /// For an overlay's backdrop and panel (the parts that take its presses): drops a press that
    /// continues a double-click begun before the overlay came up (`OverlayPresses`).
    fn guard_presses(cx: &mut Context<Self>) -> impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static {
        cx.listener(|this, ev: &MouseDownEvent, _, cx| {
            if !this.overlay_presses.admits(ev.click_count) {
                cx.stop_propagation();
            }
        })
    }

    /// The dimmed backdrop under an overlay's panel: a press on it calls `on_click` (dismisses),
    /// unless it is the second half of the double-click that opened the overlay.
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
            .child(
                div()
                    .id("fill")
                    .size_full()
                    .bg(t.bg_scrim)
                    .occlude()
                    .capture_any_mouse_down(Self::guard_presses(cx))
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, _, w, cx| {
                            on_click(this, w, cx);
                            cx.notify();
                        }),
                    ),
            )
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
                                .capture_any_mouse_down(Self::guard_presses(cx))
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

#[cfg(test)]
mod tests {
    use super::OverlayPresses;

    #[test]
    fn an_overlay_takes_the_presses_of_clicks_that_began_on_it() {
        let mut p = OverlayPresses::default();
        p.shown(Some("synth"));
        assert!(!p.admits(2), "the double-click that opened it");
        assert!(p.admits(1) && p.admits(2), "a double-click of its own");
        // A dialog it opens on a press: that press's next one is not the new dialog's.
        p.shown(Some("confirm-delete"));
        assert!(!p.admits(2));
        // Back to the first one when it closes: the second press of the click that closed it is
        // not the first one's either.
        p.admits(1);
        p.shown(Some("synth"));
        assert!(!p.admits(2));
        // Closed, then opened again by a double-click.
        p.admits(1);
        p.shown(None);
        p.shown(Some("synth"));
        assert!(!p.admits(2) && p.admits(1));
    }
}
