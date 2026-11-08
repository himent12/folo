//! Settings: connection, appearance, keyboard shortcuts and about.

use std::rc::Rc;

use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{AnyElement, Context, FontWeight, IntoElement, ParentElement, Styled, Window, div, px};

use super::*;
use crate::app::{FoloApp, ServerMode, act};
use crate::store::Load;
use crate::theme::{Mode, radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let header = page_header(&t, "Settings", "Connection, appearance and keyboard shortcuts.", vec![]);
        let row = |title: &str, help: &str, control: AnyElement| {
            div()
                .flex()
                .items_center()
                .justify_between()
                .gap(px(24.))
                .px(px(16.))
                .py(px(14.))
                .border_b_1()
                .border_color(t.line_subtle)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .min_w_0()
                        .child(
                            text(ty::BASE, t.text_primary)
                                .font_weight(FontWeight::MEDIUM)
                                .child(title.to_string()),
                        )
                        .child(text(ty::SM, t.text_tertiary).child(help.to_string())),
                )
                .child(div().flex_none().child(control))
        };
        let mode_sel = if self.server_mode == ServerMode::Mock { 0 } else { 1 };
        let mode_seg = self.segmented(
            "set-server",
            &["Mock data", "Local server"],
            mode_sel,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                this.server_mode = if i == 0 { ServerMode::Mock } else { ServerMode::Local };
                if i == 1 {
                    this.toast(
                        Tone::Info,
                        "Local server support lands with folo serve",
                        Some("The app keeps using mock data until the engine's API is on main.".into()),
                        cx,
                    );
                }
            }),
        );
        let url_fh = self.server_url.read(cx).focus_handle(cx);
        let url_ring = self.ring(&url_fh, window);
        let url = div()
            .relative()
            .w(px(280.))
            .h(px(32.))
            .px(px(10.))
            .flex()
            .items_center()
            .rounded(px(radius::MD))
            .bg(t.bg_raised)
            .border_1()
            .border_color(if url_fh.is_focused(window) { t.accent_line } else { t.line_default })
            .font_family(crate::theme::MONO)
            .text_size(px(ty::SM.size))
            .when(url_ring, |d| d.child(crate::app::focus_ring(&t, radius::MD)))
            .map(|d| {
                // With mock data the address is unused: show it read-only (not focusable).
                if self.server_mode == ServerMode::Mock {
                    d.bg(t.bg_inset)
                        .border_color(t.line_subtle)
                        .child(text(ty::SM, t.text_disabled).child(self.server_url.read(cx).text()))
                } else {
                    d.child(self.server_url.clone())
                }
            })
            .into_any_element();
        let spawn = self.switch(
            "set-spawn",
            self.spawn_server,
            window,
            cx,
            act(|this, _, cx| {
                this.spawn_server = !this.spawn_server;
                cx.notify();
            }),
        );
        let status = match &self.store.health {
            Load::Ready(h) => div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(div().size(px(6.)).rounded_full().bg(if h.mock {
                    t.accent_fill()
                } else {
                    t.success_solid
                }))
                .child(text(ty::SM, t.text_secondary).child(format!(
                    "Connected · {} · v{}",
                    self.store.api.describe(),
                    h.version
                ))),
            Load::Failed(e) => div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(div().size(px(6.)).rounded_full().bg(t.danger_solid))
                .child(text(ty::SM, t.danger_text).child(e.message.clone())),
            _ => div().child(text(ty::SM, t.text_tertiary).child("Connecting…")),
        };
        let connection = panel(&t)
            .overflow_hidden()
            .child(panel_header(&t, "Connection").child(status))
            .child(row(
                "Data source",
                "Mock data runs entirely in the app; Local server talks to folo serve over HTTP.",
                mode_seg.into_any_element(),
            ))
            .child(row("Server address", "Where folo serve listens.", url))
            .child(row(
                "Start folo serve automatically",
                "Launch a local server when the app opens and none is running.",
                spawn.into_any_element(),
            ));

        let theme_sel = if self.mode == Mode::Dark { 0 } else { 1 };
        let theme_seg = self.segmented(
            "set-theme",
            &["Dark", "Light"],
            theme_sel,
            window,
            cx,
            Rc::new(|this, i, _, cx| this.set_mode(if i == 0 { Mode::Dark } else { Mode::Light }, cx)),
        );
        let motion = self.switch(
            "set-motion",
            self.reduce_motion,
            window,
            cx,
            act(|this, _, cx| {
                this.reduce_motion = !this.reduce_motion;
                set_reduce_motion(this.reduce_motion);
                cx.refresh_windows();
            }),
        );
        let appearance = panel(&t)
            .overflow_hidden()
            .child(panel_header(&t, "Appearance"))
            .child(row(
                "Theme",
                "Ctrl Shift L switches between dark and light.",
                theme_seg.into_any_element(),
            ))
            .child(row(
                "Reduce motion",
                "Turn off pulsing live indicators and transitions.",
                motion.into_any_element(),
            ));

        let shortcuts: [(&str, &[&str]); 12] = [
            ("Command palette", &["Ctrl", "K"]),
            ("New run", &["Ctrl", "N"]),
            ("Open image", &["Ctrl", "O"]),
            ("Overview", &["Ctrl", "1"]),
            ("Datasets", &["Ctrl", "2"]),
            ("Runs", &["Ctrl", "3"]),
            ("Models", &["Ctrl", "4"]),
            ("Playground", &["Ctrl", "5"]),
            ("Settings", &["Ctrl", ","]),
            ("Back", &["Alt", "←"]),
            ("Close / back", &["Esc"]),
            ("Start training (in the dialog)", &["Ctrl", "Enter"]),
        ];
        let half = shortcuts.len().div_ceil(2);
        let col = |items: &[(&str, &[&str])]| {
            div().flex_1().flex().flex_col().children(items.iter().map(|(name, keys)| {
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(36.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(text(ty::SM, t.text_secondary).child(name.to_string()))
                    .child(shortcut(&t, keys))
            }))
        };
        let keys = panel(&t).child(panel_header(&t, "Keyboard")).child(
            div()
                .flex()
                .gap(px(32.))
                .px(px(16.))
                .pb(px(8.))
                .child(col(&shortcuts[..half]))
                .child(col(&shortcuts[half..])),
        );

        let about = panel(&t).child(
            div().flex().items_center().gap(px(14.)).p(px(16.)).child(crate::app::logo(&t, 28.)).child(
                div()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        text(ty::BASE, t.text_primary)
                            .font_weight(FontWeight::MEDIUM)
                            .child(format!("FOLO Desktop {}", env!("CARGO_PKG_VERSION"))),
                    )
                    .child(
                        text(ty::SM, t.text_tertiary)
                            .child("FOLO Only Looks Once · GPUI · MIT OR Apache-2.0"),
                    ),
            ),
        );
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .max_w(px(880.))
            .child(connection)
            .child(appearance)
            .child(keys)
            .child(about);
        self.page_frame(&t, header, body)
    }
}
