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
    AnyElement, App, Context, DispatchPhase, Div, FontWeight, InteractiveElement, IntoElement, MouseButton,
    MouseDownEvent, MouseUpEvent, ParentElement, StatefulInteractiveElement, Styled, Window, canvas, div, px,
};

use crate::app::{Action, FoloApp};
use crate::theme::{Theme, radius, ty};
use crate::ui::*;

pub use confirm_close::closing_stops;
pub use new_run_footer::estimate;

/// Keeps an overlay from taking the rest of the click that opened it: the second press of a
/// double-click whose first press (or its release) opened the overlay would dismiss it at once on
/// the backdrop, or press what opened under the pointer on the panel (a confirmation's Delete).
/// Only an overlay that came up while a press or release was handled is guarded, and only against
/// the later presses of that same click, those whose click count goes on from the opening one's
/// (`MouseDownEvent::click_count`, the platform's): one opened by a key, a command or a timer takes
/// every press, and a press that starts a click of its own ends the guard. No timer is kept here.
#[derive(Default)]
pub struct OverlayPresses {
    /// The overlay on top in the last frame.
    top: Option<&'static str>,
    /// The click count of a press or release handled since the last frame.
    pointer: Option<usize>,
    /// The overlay on top came up while a press or release of this click count was handled.
    opened_by: Option<usize>,
}

impl OverlayPresses {
    /// A press (`down`) or release is being handled, before any element sees it (`press_tracker`).
    /// A press that does not go on counting from the opening one starts a click of its own.
    pub fn pointer(&mut self, click_count: usize, down: bool) {
        if down && self.opened_by.is_some_and(|n| click_count <= n) {
            self.opened_by = None;
        }
        self.pointer = Some(click_count);
    }

    /// The overlay on top this frame (each render; a press draws one): one that came up since the
    /// last frame is guarded only when a press or release was handled meanwhile.
    pub fn shown(&mut self, top: Option<&'static str>) {
        if self.top != top {
            self.top = top;
            self.opened_by = top.and(self.pointer);
        }
        self.pointer = None;
    }

    /// Whether the overlay on top takes a press with this click count.
    pub fn admits(&self, click_count: usize) -> bool {
        self.opened_by.is_none_or(|n| click_count <= n)
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

    /// Tells `overlay_presses` of every press and release in the window, whatever is under the
    /// pointer, before the page or an overlay sees it (the root's first child: ahead of them in the
    /// capture phase), and draws a frame after it, in which `shown` sees whether it opened one.
    pub(crate) fn press_tracker(cx: &mut Context<Self>) -> AnyElement {
        let view = cx.entity().downgrade();
        let track = move |click_count, down, cx: &mut App| {
            let _ = view.update(cx, |this: &mut Self, cx| {
                this.overlay_presses.pointer(click_count, down);
                cx.notify();
            });
        };
        canvas(
            |_, _, _| {},
            move |_, _, window, _| {
                let up = track.clone();
                window.on_mouse_event(move |ev: &MouseDownEvent, phase, _, cx| {
                    if phase == DispatchPhase::Capture {
                        track(ev.click_count, true, cx);
                    }
                });
                window.on_mouse_event(move |ev: &MouseUpEvent, phase, _, cx| {
                    if phase == DispatchPhase::Capture {
                        up(ev.click_count, false, cx);
                    }
                });
            },
        )
        .absolute()
        .into_any_element()
    }

    /// For an overlay's backdrop and panel (the parts that take its presses): drops a press that
    /// goes on with the click that opened the overlay (`OverlayPresses`).
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

    /// A press of this click count, as the window sees it: the tracker, then the overlay's guard.
    fn press(p: &mut OverlayPresses, click_count: usize) -> bool {
        p.pointer(click_count, true);
        p.admits(click_count)
    }

    /// A whole click (press and release) of this click count.
    fn click(p: &mut OverlayPresses, click_count: usize) -> bool {
        let admitted = press(p, click_count);
        p.pointer(click_count, false);
        admitted
    }

    #[test]
    fn an_overlay_takes_no_more_presses_of_the_click_that_opened_it() {
        let mut p = OverlayPresses::default();
        click(&mut p, 1);
        p.shown(Some("synth"));
        assert!(!press(&mut p, 2), "the double-click that opened it");
        assert!(!press(&mut p, 3), "and a triple click");
        assert!(press(&mut p, 1) && press(&mut p, 2), "a double-click of its own");
        // A dialog it opens on a click: that click's next press is not the new dialog's.
        click(&mut p, 1);
        p.shown(Some("confirm-delete"));
        assert!(!press(&mut p, 2));
        // Back to the first one when it closes: the next press of the click that closed it is
        // not the first one's either.
        click(&mut p, 1);
        p.shown(Some("synth"));
        assert!(!press(&mut p, 2));
        // Closed by a click, then opened again by a double-click.
        click(&mut p, 1);
        p.shown(None);
        click(&mut p, 1);
        p.shown(Some("synth"));
        assert!(!press(&mut p, 2) && press(&mut p, 1));
    }

    #[test]
    fn an_overlay_that_no_press_opened_takes_every_press() {
        let mut p = OverlayPresses::default();
        // A click on the page that opens nothing (its frame is drawn), then a key opens the palette.
        click(&mut p, 1);
        p.shown(None);
        p.shown(Some("palette"));
        assert!(press(&mut p, 2), "the next press of that click dismisses it");
        // A press that opened it, then a frame: a later overlay on top (a timer's) is no press's.
        p.shown(None);
        click(&mut p, 1);
        p.shown(Some("synth"));
        p.shown(Some("confirm-close"));
        assert!(press(&mut p, 2));
    }
}
