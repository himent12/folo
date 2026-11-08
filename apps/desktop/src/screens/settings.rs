//! Settings: connection, appearance, keyboard shortcuts and about.

use std::rc::Rc;

use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, Styled, Window, div, px,
};

use super::*;
use crate::app::{FoloApp, ServerMode, act};
use crate::store::Load;
use crate::theme::{Mode, radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let header = page_header(&t, "Settings", "Connection, appearance and keyboard shortcuts.", vec![]);
        let row = |title: &str, help: AnyElement, control: AnyElement| {
            div()
                .w_full()
                .flex()
                .items_center()
                .justify_between()
                .gap(px(24.))
                .px(px(16.))
                // 56 px rows (DIRECTION); a row whose description wraps grows.
                .min_h(px(56.))
                .py(px(8.))
                .border_b_1()
                .border_color(t.line_subtle)
                // Label and description take the slack (and wrap); every control ends on the same edge.
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .min_w_0()
                        .child(
                            text(ty::BASE, t.text_primary)
                                .font_weight(FontWeight::MEDIUM)
                                .child(title.to_string()),
                        )
                        .child(help),
                )
                .child(div().flex_none().flex().justify_end().child(control))
        };
        let help = |s: &str| text(ty::SM, t.text_tertiary).child(s.to_string()).into_any_element();
        let mode_sel = if self.server_mode == ServerMode::Mock { 0 } else { 1 };
        let mode_seg = self.segmented(
            "set-server",
            &["Built-in demo", "folo serve"],
            mode_sel,
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                let mode = if i == 0 { ServerMode::Mock } else { ServerMode::Local };
                // While a connection attempt is pending the installed mode is still the old one, so
                // choosing the demo must go through even if it "already" is the demo: that cancels it.
                if mode != this.server_mode || (mode == ServerMode::Mock && this.connecting.is_some()) {
                    this.switch_source(mode, cx);
                }
            }),
        );
        let url_fh = self.server_url.read(cx).focus_handle(cx);
        let url_ring = self.ring(&url_fh, window);
        let url = div()
            .relative()
            .w(px(232.))
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
            // Always editable, even while the demo is installed: it's where the next Connect goes.
            .child(self.server_url.clone())
            .into_any_element();
        let connect = self
            .button(
                "set-connect",
                "Connect",
                crate::controls::Btn::Secondary,
                Some("icons/plug.svg"),
                window,
                cx,
                act(|this, _, cx| this.switch_source(ServerMode::Local, cx)),
            )
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
                .child(div().size(px(6.)).rounded_full().bg(t.success_solid))
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
            _ => div().child(
                text(ty::SM, t.text_tertiary)
                    .child(self.connecting.clone().unwrap_or_else(|| "Connecting…".into())),
            ),
        };
        let connection = panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Connection").child(status))
            .child(row(
                "Data source",
                help("Built-in demo data runs inside the app; folo serve is the FOLO engine over HTTP."),
                mode_seg.into_any_element(),
            ))
            .child(row(
                "Server address",
                help(if self.server_mode == ServerMode::Mock {
                    "Address of folo serve. Connect switches the data source to it."
                } else {
                    "Where folo serve listens. Press Enter or Connect to connect again."
                }),
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .on_key_down(cx.listener(|this, ev: &gpui::KeyDownEvent, _, cx| {
                        if ev.keystroke.key == "enter" {
                            this.switch_source(ServerMode::Local, cx);
                            cx.stop_propagation();
                        }
                    }))
                    .child(url)
                    .child(connect)
                    .into_any_element(),
            ))
            .child(row(
                "Start folo serve automatically",
                help("Launch a local server when the app opens and none is running."),
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
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Appearance"))
            .child(row(
                "Theme",
                div()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .child(shortcut(&t, &["Ctrl", "Shift", "L"]))
                    .child(text(ty::SM, t.text_tertiary).child("switches anywhere"))
                    .into_any_element(),
                theme_seg.into_any_element(),
            ))
            .child(row(
                "Reduce motion",
                help("Turn off pulsing live indicators and transitions."),
                motion.into_any_element(),
            ));

        let actions: [(&str, &[&str]); 7] = [
            ("Command palette", &["Ctrl", "K"]),
            ("New run", &["Ctrl", "N"]),
            ("Open image", &["Ctrl", "O"]),
            ("Switch theme", &["Ctrl", "Shift", "L"]),
            ("Back", &["Alt", "←"]),
            ("Close / back", &["Esc"]),
            ("Start training (in the dialog)", &["Ctrl", "Enter"]),
        ];
        let go_to: [(&str, &[&str]); 6] = [
            ("Overview", &["Ctrl", "1"]),
            ("Datasets", &["Ctrl", "2"]),
            ("Runs", &["Ctrl", "3"]),
            ("Models", &["Ctrl", "4"]),
            ("Playground", &["Ctrl", "5"]),
            ("Settings", &["Ctrl", "Comma"]),
        ];
        let col = |title: &str, items: &[(&str, &[&str])]| {
            div()
                .flex_1()
                .flex()
                .flex_col()
                .child(div().pt(px(4.)).pb(px(6.)).child(overline(&t, title)))
                .children(items.iter().map(|(name, keys)| {
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
        let keys = panel(&t).flex().flex_col().child(panel_header(&t, "Keyboard")).child(
            div()
                .flex()
                .gap(px(32.))
                .px(px(16.))
                .pt(px(8.))
                .pb(px(8.))
                .child(col("Actions", &actions))
                .child(col("Go to", &go_to)),
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
        // A readable column, left-aligned: labels stay near their controls at any window width.
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .w_full()
            .max_w(px(760.))
            .child(connection)
            .child(appearance)
            .child(keys)
            .child(about);
        self.page_frame(&t, header, body, cx)
    }
}
