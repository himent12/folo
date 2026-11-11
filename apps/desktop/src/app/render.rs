//! The root view's render: title bar, sidebar, page and overlays, the key-hint strip, and the
//! window's key and mouse handlers.

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, Render, SharedString, Styled,
    Window, div, px,
};

use crate::theme::{Theme, theme, ty};
use crate::ui::*;

use super::focus::REVEAL;
use super::{
    Back, Dismiss, FocusNext, FocusPrev, FoloApp, GoBenchmarks, GoDatasets, GoModels, GoOverview,
    GoPlayground, GoRuns, GoSettings, NewRunAction, NextRegion, OpenImage, OpenPalette, PrevRegion, Route,
    Submit, ToggleFullscreen, ToggleTheme,
};

// ---------- render ----------

impl Render for FoloApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some(r) = self.route_pending.take() {
            self.navigate(r, window, cx);
        }
        if std::mem::take(&mut self.restore_pending) {
            self.restore_focus(window);
        }
        if self.scroll_end_pending > 0 {
            self.scroll_end_pending -= 1;
            let max = self.page_scroll.max_offset().height;
            self.page_scroll.set_offset(folo_ui::point(px(0.), -max));
            window.request_animation_frame();
        }
        if self.reveal_pending > 0 {
            self.reveal_pending -= 1;
            REVEAL.with(|r| *r.borrow_mut() = Some(vec![self.page_scroll.clone()]));
            window.request_animation_frame();
        }
        if self.open_new_run_pending && self.store.datasets.ready().is_some_and(|d| !d.is_empty()) {
            self.open_new_run_pending = false;
            self.open_new_run(None, window, cx);
        }
        self.narrow = window.viewport_size().width < px(1120.);
        self.shown_mode = if window.is_fullscreen() {
            crate::window_state::WindowMode::Fullscreen
        } else if window.is_maximized() {
            crate::window_state::WindowMode::Maximized
        } else {
            crate::window_state::WindowMode::Windowed
        };
        let t = theme(cx);
        self.ensure_core(cx);
        self.ensure_cloud(cx);
        let titlebar = self.titlebar(window, cx).into_any_element();
        let sidebar = self.sidebar(window, cx).into_any_element();
        *self.key_hint.borrow_mut() = None;
        let page = self.page(window, cx);
        // A dialog or the inspector owns the keys: the page's hint stands down.
        let key_hint = self.key_hint.borrow_mut().take().filter(|_| self.kb && !self.modal_open());
        let overlays = self.overlays(window, cx);
        div()
            .id("root")
            .key_context("FoloApp")
            .track_focus(&self.root_focus)
            .size_full()
            .flex()
            .font_family(crate::theme::SANS)
            .text_size(px(ty::BASE.size))
            .line_height(px(ty::BASE.line))
            .text_color(t.text_primary)
            .bg(t.bg_app)
            .on_action(cx.listener(|this, _: &OpenPalette, w, cx| this.open_palette(w, cx)))
            .on_action(cx.listener(|this, _: &NewRunAction, w, cx| {
                if !this.modal_open() {
                    this.palette = None;
                    this.open_new_run(None, w, cx)
                }
            }))
            .on_action(cx.listener(|this, _: &GoOverview, w, cx| this.nav_shortcut(Route::Overview, w, cx)))
            .on_action(cx.listener(|this, _: &GoDatasets, w, cx| this.nav_shortcut(Route::Datasets, w, cx)))
            .on_action(cx.listener(|this, _: &GoRuns, w, cx| this.nav_shortcut(Route::Runs, w, cx)))
            .on_action(cx.listener(|this, _: &GoModels, w, cx| this.nav_shortcut(Route::Models, w, cx)))
            .on_action(cx.listener(|this, _: &GoBenchmarks, w, cx| this.nav_shortcut(Route::Benches, w, cx)))
            .on_action(
                cx.listener(|this, _: &GoPlayground, w, cx| this.nav_shortcut(Route::Playground, w, cx)),
            )
            .on_action(cx.listener(|this, _: &GoSettings, w, cx| this.nav_shortcut(Route::Settings, w, cx)))
            .on_action(cx.listener(|this, _: &ToggleTheme, _, cx| this.toggle_theme(cx)))
            .on_action(cx.listener(|_, _: &ToggleFullscreen, w, _| w.toggle_fullscreen()))
            .on_action(cx.listener(|this, _: &Back, w, cx| {
                if !this.modal_open() {
                    this.back(w, cx)
                }
            }))
            .on_action(cx.listener(|this, _: &OpenImage, _, cx| {
                if !this.modal_open() {
                    this.pick_image(cx)
                }
            }))
            .on_action(cx.listener(|this, _: &Submit, w, cx| {
                if this.cloud.connect.is_some() {
                    this.submit_connect(w, cx);
                } else if this.new_run.is_some() {
                    this.submit_new_run(w, cx);
                } else if this.synth.is_some() {
                    this.submit_synth(w, cx);
                }
            }))
            .on_action(cx.listener(|this, _: &FocusNext, w, cx| {
                this.menu_open = None;
                this.export_open = false;
                this.kb = true;
                this.cycle_focus(true, w, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &FocusPrev, w, cx| {
                this.menu_open = None;
                this.export_open = false;
                this.kb = true;
                this.cycle_focus(false, w, cx);
                cx.notify();
            }))
            .on_action(cx.listener(|this, _: &NextRegion, w, cx| this.cycle_region(true, w, cx)))
            .on_action(cx.listener(|this, _: &PrevRegion, w, cx| this.cycle_region(false, w, cx)))
            .on_action(cx.listener(|this, _: &Dismiss, w, cx| this.dismiss(w, cx)))
            // Keyboard mode turns on in the capture phase, before a focused control (a slider, a
            // segmented group) consumes the key, so its first keyboard step already shows the ring.
            .capture_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                let nav = matches!(
                    ev.keystroke.key.as_str(),
                    "up" | "down" | "left" | "right" | "home" | "end" | "pageup" | "pagedown"
                );
                if nav && !this.kb {
                    this.kb = true;
                    cx.notify();
                }
            }))
            .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                // Enter and Space only when nothing took them (typed into a field they're text).
                if matches!(ev.keystroke.key.as_str(), "enter" | "space") && !this.kb {
                    this.kb = true;
                    cx.notify();
                }
                let m = &ev.keystroke.modifiers;
                // Page Up / Down scroll the page about a screen (less 32 px, so a line of context
                // stays) whenever the focused control didn't use them: a chart group keeps its one
                // Tab stop and the page still pages.
                if matches!(ev.keystroke.key.as_str(), "pageup" | "pagedown")
                    && !this.modal_open()
                    && this.palette.is_none()
                    && !(m.control || m.alt || m.platform || m.shift)
                {
                    let s = &this.page_scroll;
                    let (y, max) = (-s.offset().y, s.max_offset().height);
                    let page = (s.bounds().size.height - px(32.)).max(px(32.));
                    let to = if ev.keystroke.key == "pageup" { y - page } else { y + page };
                    s.set_offset(folo_ui::point(s.offset().x, -to.clamp(px(0.), max)));
                    cx.stop_propagation();
                    cx.notify();
                    return;
                }
                if this.route == Route::Playground
                    && this.play_image.is_some()
                    && !this.modal_open()
                    && this.palette.is_none()
                    && !(m.control || m.alt || m.platform)
                {
                    let d = if m.shift { 128. } else { 32. };
                    let zoomed = this.play_zoom > 1.0;
                    match ev.keystroke.key.as_str() {
                        // Arrows reach here only when no focused control used them.
                        "left" if zoomed => this.play_pan.0 += d,
                        "right" if zoomed => this.play_pan.0 -= d,
                        "up" if zoomed => this.play_pan.1 += d,
                        "down" if zoomed => this.play_pan.1 -= d,
                        "+" | "=" => this.play_zoom_by(1.4, cx),
                        "-" => this.play_zoom_by(1.0 / 1.4, cx),
                        "0" => {
                            this.play_zoom = 1.0;
                            this.play_pan = (0., 0.);
                        }
                        _ => return,
                    }
                    cx.notify();
                }
            }))
            .capture_any_mouse_down(self.leave_kb(cx))
            .on_mouse_move(cx.listener(Self::on_drag_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::on_drag_end))
            .on_drop(cx.listener(|this, paths: &folo_ui::ExternalPaths, w, cx| {
                this.play_drop_hover = false;
                // A dialog or the inspector is up: don't navigate out from under it.
                if this.modal_open() {
                    this.toast(
                        Tone::Info,
                        "Close the dialog first",
                        Some("Then drop the image again.".into()),
                        cx,
                    );
                    return;
                }
                // A drop obeys Open image's rule (and says why) before it navigates anywhere.
                if let Some(why) = this.open_image_blocked() {
                    this.toast(Tone::Neutral, "Can't open an image yet", Some(why.into()), cx);
                    return;
                }
                if let Some(p) = paths.paths().first().cloned() {
                    this.navigate(Route::Playground, w, cx);
                    this.play_file(p, cx);
                }
            }))
            .child(sidebar)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(titlebar)
                    // The page's region anchor for F6 (not a Tab stop itself).
                    .child(
                        div()
                            .id("page-region")
                            .track_focus(&self.fh("page-region", cx).tab_stop(false))
                            .flex_1()
                            .min_h_0()
                            .relative()
                            .child(page)
                            .when_some(key_hint, |d, h| {
                                d.child(key_hint_strip(&t, h).absolute().left_0().right_0().bottom_0())
                            }),
                    ),
            )
            .children(overlays)
            // The top edge resizes the window, as the other three do. Windows only resizes where no
            // caption area is hit, and the title bar is one up to the top: this strip, on top of
            // it and hiding it from the hit test, is not. Maximized or full screen there is no edge,
            // and the top pixels are the title bar's.
            .when(!window.is_maximized() && !window.is_fullscreen(), |d| {
                d.child(
                    div()
                        .id("resize-top")
                        .absolute()
                        .top_0()
                        .left_0()
                        .right_0()
                        .h(px(crate::theme::layout::DESKTOP_RESIZE_EDGE))
                        .occlude()
                        .capture_any_mouse_down(self.leave_kb(cx)),
                )
            })
    }
}

/// The strip over the page's bottom edge that names the focused control's keys: 32 px, a
/// hairline above, a keyboard glyph and the keys. An overlay, so focusing a control never moves
/// the page (its last 32 px are the page's own bottom padding).
fn key_hint_strip(t: &Theme, keys: SharedString) -> folo_ui::Stateful<folo_ui::Div> {
    div()
        .id("key-hint")
        .flex_none()
        .h(px(32.))
        .px(px(16.))
        .flex()
        .items_center()
        .gap(px(8.))
        .border_t_1()
        .border_color(t.line_subtle)
        .bg(t.bg_panel)
        .child(crate::ui::icon("icons/keyboard.svg", 14., t.text_tertiary))
        .child(crate::ui::text(ty::SM, t.text_secondary).whitespace_nowrap().child(keys))
}
