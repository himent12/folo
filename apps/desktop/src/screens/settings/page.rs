//! The Settings page: connection, appearance, keyboard shortcuts and about.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, Focusable, FontWeight, InteractiveElement, IntoElement, ParentElement, Styled,
    Window, div, px,
};

use crate::app::{FoloApp, ServerMode, act};
use crate::screens::page_header;
use crate::store::Load;
use crate::theme::{Theme, radius, theme, ty};
use crate::ui::*;

/// The settings column's widest: wide enough not to read as a narrow form, close enough that a
/// label and its control (right edge) stay one glance apart on big windows. The column, and the
/// page header with it, is centred in the page.
pub(super) const COLUMN_MAX: f32 = 880.;

/// Settings › Keyboard's "Go to" column: every page shortcut `app::bind_keys` binds.
const GO_TO_KEYS: [(&str, &[&str]); 9] = [
    ("Overview", &["Ctrl", "1"]),
    ("Datasets", &["Ctrl", "2"]),
    ("Runs", &["Ctrl", "3"]),
    ("Models", &["Ctrl", "4"]),
    ("Benchmarks", &["Ctrl", "5"]),
    ("Playground", &["Ctrl", "6"]),
    ("Settings", &["Ctrl", "Comma"]),
    // Sidebar, search, page.
    ("Next region", &["F6"]),
    ("Previous region", &["Shift", "F6"]),
];

/// Base HTTP(S) URL of FOLO Studio served by the connected `folo serve`, or `None` in Built-in demo
/// mode, when the address is invalid, or when it embeds URL credentials.
pub(crate) fn studio_url(
    server_mode: ServerMode,
    connected_url: Option<&str>,
    input_url: &str,
    bad_address: bool,
) -> Option<String> {
    if server_mode != ServerMode::Local || bad_address {
        return None;
    }
    let raw = connected_url
        .and_then(crate::app::persistable_server_url)
        .or_else(|| crate::app::persistable_server_url(input_url))?;
    let backend = crate::api::http::HttpBackend::new(&raw);
    let base = backend.base();
    (base.starts_with("http://") || base.starts_with("https://")).then(|| base.to_string())
}

/// One Settings row (UNIFIED.md §4): the name and its one-line description on the left (they take the
/// slack and wrap), the control on the right edge; 56 px high, more when the description wraps.
pub(super) fn settings_row(t: &Theme, title: &str, help: AnyElement, control: AnyElement) -> folo_ui::Div {
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
                    text(ty::BASE, t.text_primary).font_weight(FontWeight::MEDIUM).child(title.to_string()),
                )
                .child(help),
        )
        .child(div().flex_none().flex().justify_end().child(control))
}

impl FoloApp {
    pub fn settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let header = page_header(
            &t,
            "Settings",
            // Studio's fact, for this app: where what is picked here is kept (desktop.json in
            // FOLO's home folder, `window_state`).
            "Preferences are stored on this computer.",
            vec![],
        )
        // The header sits on the column's edges, so the two centre together.
        .w_full()
        .max_w(px(COLUMN_MAX))
        .mx_auto();
        let row = |title: &str, help: AnyElement, control: AnyElement| settings_row(&t, title, help, control);
        // Command names render as code chips, as on Overview.
        let help = |s: &str| crate::ui::rich(&t, ty::SM, t.text_tertiary, s).text_left().into_any_element();
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
            .border_color(if url_fh.is_focused(window) { t.accent_line } else { t.line_control })
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
                act(|this, _, cx| this.connect_server(cx)),
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
        let fail_hint = self.store.health.error().and_then(|e| e.hint.clone());
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
        let studio_target = studio_url(
            self.server_mode,
            self.connected_url.as_deref(),
            &self.server_url.read(cx).text(),
            self.store.bad_address(),
        );
        let studio_controls = if let Some(url) = studio_target {
            let open_url = url.clone();
            let copy_url = url;
            div()
                .flex()
                .items_center()
                .gap(px(8.))
                .child(self.button(
                    "set-studio-open",
                    "Open in browser",
                    crate::controls::Btn::Secondary,
                    Some("icons/external-link.svg"),
                    window,
                    cx,
                    act(move |_, _, cx| cx.open_url(&open_url)),
                ))
                .child(self.button(
                    "set-studio-copy",
                    "Copy URL",
                    crate::controls::Btn::Secondary,
                    Some("icons/copy.svg"),
                    window,
                    cx,
                    act(move |this, _, cx| {
                        cx.write_to_clipboard(folo_ui::ClipboardItem::new_string(copy_url.clone()));
                        this.toast(Tone::Neutral, "Studio URL copied", Some(copy_url.clone().into()), cx);
                    }),
                ))
                .into_any_element()
        } else {
            let reason = if self.server_mode == ServerMode::Mock {
                "Connect to folo serve to open FOLO Studio"
            } else {
                "Enter a valid folo serve address to open FOLO Studio"
            };
            self.button_disabled(
                "set-studio-open",
                "Open in browser",
                crate::controls::Btn::Secondary,
                Some("icons/external-link.svg"),
                reason,
                window,
                cx,
            )
            .into_any_element()
        };
        let connection = panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Connection").child(status))
            // What to do about a failure ("Type it as http://host:port, …"), under its words.
            .when_some(fail_hint, |d, h| {
                d.child(
                    div()
                        .px(px(16.))
                        .py(px(8.))
                        .border_b_1()
                        .border_color(t.line_subtle)
                        .child(text(ty::SM, t.text_tertiary).child(h)),
                )
            })
            .child(row(
                "Data source",
                help("Built-in demo data runs inside the app; `folo serve` is the FOLO engine over HTTP."),
                mode_seg.into_any_element(),
            ))
            .child(row(
                "Server address",
                help(if self.server_mode == ServerMode::Mock {
                    "Address of `folo serve`. Connect switches the data source to it."
                } else {
                    "Where `folo serve` listens. Press Enter or Connect to connect again."
                }),
                div()
                    .flex()
                    .items_center()
                    .gap(px(8.))
                    .on_key_down(cx.listener(|this, ev: &folo_ui::KeyDownEvent, _, cx| {
                        if ev.keystroke.key == "enter" {
                            this.connect_server(cx);
                            cx.stop_propagation();
                        }
                    }))
                    .child(url)
                    .child(connect)
                    .into_any_element(),
            ))
            .child(row(
                "Start a local server automatically",
                help("When no server is running at the address above, launch `folo serve` on a free port and stop it on exit."),
                spawn.into_any_element(),
            ))
            .child(row(
                "FOLO Studio",
                help("Web UI served by `folo serve` for browser-only workflows like agent labeling and run comparison."),
                studio_controls,
            ));

        let appearance = self.appearance_panel(window, cx);

        let actions: [(&str, &[&str]); 10] = [
            ("Command palette", &["Ctrl", "K"]),
            ("New run", &["Ctrl", "N"]),
            ("Open image", &["Ctrl", "O"]),
            ("Toggle sidebar", &["Ctrl", "B"]),
            ("Switch theme", &["Ctrl", "Shift", "L"]),
            ("Full screen", &["F11"]),
            ("Back", &["Alt", "←"]),
            ("Forward", &["Alt", "→"]),
            ("Close / back", &["Esc"]),
            ("Submit the open dialog or form", &["Ctrl", "Enter"]),
        ];
        let go_to = GO_TO_KEYS;
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
                            .child("FOLO Only Looks Once · folo-ui · MIT OR Apache-2.0"),
                    ),
            ),
        );
        // A readable column, centred in the page: labels stay near their controls at any window
        // width, and a wide window's spare room falls evenly on both sides.
        let body = div()
            .flex()
            .flex_col()
            .gap(px(16.))
            .w_full()
            .max_w(px(COLUMN_MAX))
            .mx_auto()
            .child(connection)
            // Where work runs, both: this server, then the compute it can rent or reach.
            .when(self.cloud_on(), |d| {
                d.child(self.cloud_panel(window, cx)).children(self.instances_panel(window, cx))
            })
            .when(self.store.features.iter().any(|f| f == "throughput"), |d| {
                d.child(self.throughput_panel(window, cx))
            })
            .child(appearance)
            .child(keys)
            .child(about);
        self.page_frame(&t, header, body, cx)
    }
}

#[cfg(test)]
mod tests {
    use super::studio_url;
    use crate::app::{Route, ServerMode};

    #[test]
    fn every_page_shortcut_is_listed() {
        // Ctrl+1–6 go to the six pages (app::bind_keys) in sidebar order: 5 = Benchmarks, 6 = Playground.
        for n in ["1", "2", "3", "4", "5", "6"] {
            assert!(super::GO_TO_KEYS.iter().any(|(_, k)| *k == ["Ctrl", n]), "Ctrl {n}");
        }
        assert!(super::GO_TO_KEYS.iter().any(|(name, k)| *name == "Benchmarks" && *k == ["Ctrl", "5"]));
        assert!(super::GO_TO_KEYS.iter().any(|(name, k)| *name == "Playground" && *k == ["Ctrl", "6"]));
    }

    #[test]
    fn studio_url_requires_valid_credential_free_local_server() {
        assert_eq!(
            studio_url(ServerMode::Mock, Some("http://127.0.0.1:7878"), "http://127.0.0.1:7878", false),
            None
        );
        assert_eq!(studio_url(ServerMode::Local, None, "http://127.0.0.1:99999", true), None);
        assert_eq!(studio_url(ServerMode::Local, None, "https://user:pass@host:7878", false), None);
        assert_eq!(
            studio_url(ServerMode::Local, Some("http://127.0.0.1:54321/"), "http://127.0.0.1:7878", false),
            Some("http://127.0.0.1:54321".into())
        );
        assert_eq!(
            studio_url(ServerMode::Local, None, "10.0.0.5:7878", false),
            Some("http://10.0.0.5:7878".into())
        );
    }

    #[test]
    fn sidebar_toggle_and_ctrl_5_6_shortcuts_work() {
        use crate::app::tests::{drive, open};
        use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};

        let press =
            |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<crate::app::FoloApp>, key: &str| {
                let ks = Keystroke::parse(key).unwrap();
                app.simulate_input(
                    w.into(),
                    PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
                );
                app.run_until_idle();
                app.frame(w.into());
                app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
                app.run_until_idle();
                app.frame(w.into());
            };

        let (mut app, window) = open();
        app.update(|cx| {
            crate::app::bind_keys(cx);
            crate::input::bind_keys(cx);
        });
        app.frame(window.into());

        // Ctrl+5 -> Benchmarks, Ctrl+6 -> Playground (matching sidebar order).
        press(&mut app, window, "ctrl-5");
        assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::Benches);
        press(&mut app, window, "ctrl-6");
        assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::Playground);

        // Ctrl+B collapses the sidebar and expands content_width on a 1280 px window.
        let wide_content = drive(&mut app, window, |this, w, _| {
            assert!(!this.sidebar_collapsed);
            assert!(!this.sidebar_compact(w));
            this.content_width(w)
        });
        press(&mut app, window, "ctrl-b");
        let collapsed_content = drive(&mut app, window, |this, w, _| {
            assert!(this.sidebar_collapsed);
            assert!(this.sidebar_compact(w));
            this.content_width(w)
        });
        assert!(
            collapsed_content > wide_content,
            "collapsing sidebar widens content area ({collapsed_content} <= {wide_content})"
        );
        press(&mut app, window, "ctrl-b");
        assert!(!drive(&mut app, window, |this, _, _| this.sidebar_collapsed));

        // Settings renders FOLO Studio and Custom themes Open folder cleanly.
        drive(&mut app, window, |this, w, cx| this.navigate(Route::Settings, w, cx));
        app.frame(window.into());
    }
}
