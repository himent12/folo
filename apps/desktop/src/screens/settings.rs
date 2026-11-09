//! Settings: connection, device throughput, appearance, keyboard shortcuts and about.

use std::rc::Rc;

use gpui::Focusable;
use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use super::*;
use crate::api::ThroughputSource;
use crate::app::{FoloApp, Route, ServerMode, act};
use crate::store::Load;
use crate::theme::{Mode, radius, theme, ty};
use crate::ui::*;

/// The settings column's widest: wide enough not to read as a narrow form, close enough that a
/// label and its control (right edge) stay one glance apart on big windows.
const COLUMN_MAX: f32 = 880.;

impl FoloApp {
    pub fn settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let header = page_header(
            &t,
            "Settings",
            if self.cloud_on() {
                "Connection, cloud compute, this machine's speed, appearance and keyboard shortcuts."
            } else {
                "Connection, this machine's speed, appearance and keyboard shortcuts."
            },
            vec![],
        );
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
                // What autoconnect actually starts: a demo server (`folo serve --mock`), not
                // one that trains. Read at startup, whatever the current source.
                "Start a demo server automatically",
                help("At startup, connect to a running server; otherwise launch `folo serve --mock`. If that fails, use the built-in demo."),
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
        let go_to: [(&str, &[&str]); 8] = [
            ("Overview", &["Ctrl", "1"]),
            ("Datasets", &["Ctrl", "2"]),
            ("Runs", &["Ctrl", "3"]),
            ("Models", &["Ctrl", "4"]),
            ("Playground", &["Ctrl", "5"]),
            ("Settings", &["Ctrl", "Comma"]),
            // Sidebar, search, page.
            ("Next region", &["F6"]),
            ("Previous region", &["Shift", "F6"]),
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
            .max_w(px(COLUMN_MAX))
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

    /// The server's measured speeds, and the reference figures for everything it hasn't measured,
    /// each row saying which it is and from how many samples. "Measure now" runs the server's quick
    /// probe (servers with `benchmark`).
    fn throughput_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        if self.throughput.needs_fetch() {
            self.load_throughput(cx);
        }
        let now = self.store.api.now();
        let can_measure = self.store.features.iter().any(|f| f == "benchmark");
        let measure = if self.bench_busy {
            self.button_disabled(
                "tp-measure",
                "Measuring…",
                crate::controls::Btn::Secondary,
                Some("icons/timer.svg"),
                "A probe is running: a few seconds of training, then predictions",
                window,
                cx,
            )
            .into_any_element()
        } else {
            self.button(
                "tp-measure",
                "Measure now",
                crate::controls::Btn::Secondary,
                Some("icons/timer.svg"),
                window,
                cx,
                act(|this, _, cx| this.measure_now(cx)),
            )
            .tooltip(crate::controls::tip(
                "Trains folo-n at 320 px for about 4 s on synthetic images, then times predictions (about 8 s in all)",
                None,
            ))
            .into_any_element()
        };
        let table = self.throughput.ready().cloned();
        // The adapter or CPU each device is on this server now: a measured row from other hardware
        // under the same name (a moved data dir, a new GPU) isn't "measured here", and says where.
        let system = self.store.system.ready().cloned();
        let current = |device: &str| -> Option<String> {
            if device == "cpu" {
                return system.as_ref().map(|s| s.cpu.name.clone());
            }
            let ordinal: u32 = device.strip_prefix("gpu:")?.parse().ok()?;
            system
                .as_ref()
                .and_then(|s| s.gpus.iter().find(|g| g.ordinal == ordinal).map(|g| g.name.clone()))
                .or_else(|| table.as_ref().filter(|tb| tb.device == device).map(|tb| tb.device_name.clone()))
        };
        // Measured on this device's current hardware, confirmed: a device the server no longer
        // has (or doesn't name) keeps the row's attribution.
        let here = |device: &str, on: &str| current(device).is_some_and(|c| c == on);
        let (n_train, n_infer) = table.as_ref().map_or((0, 0), |tb| {
            let measured = |s: ThroughputSource| s == ThroughputSource::Measured;
            (
                tb.training.iter().filter(|r| measured(r.source) && here(&r.device, &r.measured_on)).count(),
                tb.inference.iter().filter(|r| measured(r.source) && here(&r.device, &r.measured_on)).count(),
            )
        });
        let view_seg = self.segmented(
            "tp-view",
            &["Training", "Inference"],
            self.tp_view,
            window,
            cx,
            Rc::new(|this, i, _, _| this.tp_view = i),
        );
        let all_seg = self.segmented(
            "tp-all",
            &["Measured", "With reference"],
            usize::from(self.tp_all),
            window,
            cx,
            Rc::new(|this, i, _, _| this.tp_all = i == 1),
        );
        let device_name = table.as_ref().map(|tb| tb.device_name.clone());
        let infer = self.tp_view == 1;
        // Fixed columns (the panel is at most 880 px): device, architecture, image size, batch
        // (training), precision, latency (inference), rate, then source and age.
        let cell = |w: f32, right: bool, el: gpui::Div| {
            el.w(px(w)).flex_none().whitespace_nowrap().when(right, |d| d.text_right())
        };
        let head = div()
            .flex()
            .items_center()
            .gap(px(16.))
            .h(px(36.))
            .px(px(16.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(cell(64., false, th(&t, "Device")))
            .child(cell(96., false, th(&t, "Architecture")))
            .child(cell(72., false, th(&t, "Image size")))
            .when(!infer, |d| d.child(cell(40., true, th(&t, "Batch"))))
            .child(cell(80., false, th(&t, "Precision")))
            .when(infer, |d| d.child(cell(72., true, th(&t, "Latency"))))
            .child(cell(72., true, th(&t, "img/s")))
            .child(th(&t, "Source").flex_1().min_w_0())
            .child(cell(72., true, th(&t, "Updated")));
        // One row: the settings, the rate, and where the number comes from, said plainly.
        let row = |key: String,
                   device: &str,
                   arch: &str,
                   imgsz: u32,
                   batch: Option<u32>,
                   precision: &str,
                   latency: Option<f64>,
                   rate: f64,
                   source: String,
                   measured: bool,
                   note: Option<String>,
                   updated: Option<&str>| {
            let ink = if measured { t.text_primary } else { t.text_secondary };
            div()
                .id(SharedString::from(key))
                .flex()
                .items_center()
                .gap(px(16.))
                .h(px(36.))
                .px(px(16.))
                .border_b_1()
                .border_color(t.line_subtle)
                // The whole source on hover (it can end in "…" at small widths), with the
                // server's note on how a reference was derived.
                .tooltip(crate::controls::tip(
                    match note {
                        Some(n) => format!("{source} · {n}"),
                        None => source.clone(),
                    },
                    None,
                ))
                .child(cell(
                    64.,
                    false,
                    text(ty::SM, t.text_secondary).child(if device == "cpu" {
                        "CPU".to_string()
                    } else {
                        device.replace("gpu:", "GPU ")
                    }),
                ))
                .child(cell(96., false, text(ty::SM, ink).child(arch.to_string())))
                .child(cell(72., false, text(ty::SM, t.text_secondary).tnum().child(format!("{imgsz} px"))))
                .when_some(batch, |d, b| {
                    d.child(cell(40., true, text(ty::SM, t.text_secondary).tnum().child(b.to_string())))
                })
                .child(cell(80., false, text(ty::SM, t.text_secondary).child(precision.to_string())))
                .when_some(latency, |d, l| {
                    d.child(cell(72., true, text(ty::SM, ink).tnum().child(format!("{l:.1} ms"))))
                })
                .child(cell(72., true, text(ty::SM, ink).tnum().child(int(rate.round() as u64))))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .child(div().size(px(6.)).flex_none().rounded_full().bg(if measured {
                            t.success_solid
                        } else {
                            t.text_disabled
                        }))
                        .child(
                            text(ty::SM, if measured { t.text_secondary } else { t.text_tertiary })
                                .ellipsis()
                                .child(source.clone()),
                        ),
                )
                .child(cell(
                    72.,
                    true,
                    text(ty::SM, t.text_tertiary).tnum().child(
                        updated
                            .and_then(crate::api::time::parse)
                            .map_or("–".to_string(), |u| crate::api::time::ago(u, now)),
                    ),
                ))
                .into_any_element()
        };
        // Measured: how many samples, and probes among them; on another adapter than this server's
        // (a moved data dir), it says so: estimates don't use those rows.
        let source = |measured: bool, samples: u64, probes: u64, on: &str, device: &str| {
            if !measured {
                return "Reference · not measured here".to_string();
            }
            let n = |k: u64, one: &str, many: &str| format!("{} {}", int(k), if k == 1 { one } else { many });
            let what = if probes > 0 && probes == samples {
                n(probes, "probe", "probes")
            } else if probes > 0 {
                format!("{} · {}", n(samples, "sample", "samples"), n(probes, "probe", "probes"))
            } else {
                n(samples, "sample", "samples")
            };
            if here(device, on) {
                format!("Measured · {what}")
            } else {
                format!("Measured on {} · {what}", crate::ui::short_gpu(on))
            }
        };
        let rows: Vec<AnyElement> = match &table {
            None => Vec::new(),
            Some(tb) if infer => tb
                .inference
                .iter()
                .enumerate()
                .filter(|(_, r)| self.tp_all || r.source == ThroughputSource::Measured)
                .map(|(i, r)| {
                    let measured = r.source == ThroughputSource::Measured;
                    row(
                        format!("tp-i-{i}"),
                        &r.device,
                        r.arch.as_str(),
                        r.imgsz,
                        None,
                        if r.precision == crate::api::InferPrecision::F16 { "f16" } else { "f32" },
                        Some(r.latency_ms),
                        r.imgs_per_s,
                        source(measured, r.samples, 0, &r.measured_on, &r.device),
                        measured,
                        r.note.clone(),
                        r.updated.as_deref(),
                    )
                })
                .collect(),
            Some(tb) => tb
                .training
                .iter()
                .enumerate()
                .filter(|(_, r)| self.tp_all || r.source == ThroughputSource::Measured)
                .map(|(i, r)| {
                    let measured = r.source == ThroughputSource::Measured;
                    row(
                        format!("tp-t-{i}"),
                        &r.device,
                        r.arch.as_str(),
                        r.imgsz,
                        Some(r.batch),
                        if r.amp { "mixed (f16)" } else { "f32" },
                        None,
                        r.imgs_per_s,
                        source(measured, r.samples, r.probes, &r.measured_on, &r.device),
                        measured,
                        r.note.clone(),
                        r.updated.as_deref(),
                    )
                })
                .collect(),
        };
        let measured_n = if infer { n_infer } else { n_train };
        let body = match &self.throughput {
            Load::Failed(e) => div()
                .p(px(16.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(text(ty::SM, t.danger_text).flex_1().min_w_0().child(format!(
                    "Couldn't load the table: {}",
                    e.message
                )))
                .child(self.button(
                    "tp-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.load_throughput(cx)),
                ))
                .into_any_element(),
            Load::Ready(_) if rows.is_empty() => div()
                .p(px(16.))
                .child(text(ty::SM, t.text_tertiary).child(if can_measure {
                    "Nothing measured on this server yet: train a run, or Measure now. With reference shows FOLO's figures."
                } else {
                    "Nothing measured on this server yet: train a run. With reference shows FOLO's figures."
                }))
                .into_any_element(),
            Load::Ready(_) => div().flex().flex_col().child(head).children(rows).into_any_element(),
            _ => div()
                .p(px(16.))
                .flex()
                .flex_col()
                .gap(px(8.))
                .children((0..3).map(|i| div().h(px(20.)).child(skeleton(&t, format!("tp-sk-{i}")))))
                .into_any_element(),
        };
        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(
                panel_header(&t, "Device throughput")
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(12.))
                            .when_some(device_name.clone(), |d, n| {
                                d.child(text(ty::SM, t.text_tertiary).child(crate::ui::short_gpu(&n)))
                            })
                            // The table as the server has it now (a run may have added rows).
                            .child(self.icon_button(
                                "tp-reload",
                                "icons/refresh-cw.svg",
                                "Reload the table",
                                window,
                                cx,
                                act(|this, _, cx| this.load_throughput(cx)),
                            ))
                            .when(can_measure, |d| d.child(measure)),
                    ),
            )
            .child(
                div()
                    .px(px(16.))
                    .py(px(10.))
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap(px(16.))
                    .border_b_1()
                    .border_color(t.line_subtle)
                    .child(
                        // What the numbers are: this machine's own, or FOLO's reference figures.
                        text(ty::SM, t.text_tertiary).flex_1().min_w_0().child(format!(
                            "{} measured here (runs, predictions and probes); the rest are FOLO's reference figures, scaled. Estimates use measured rows first.",
                            match measured_n {
                                0 => "No rows".to_string(),
                                1 => "1 row".to_string(),
                                n => format!("{n} rows"),
                            }
                        )),
                    )
                    .child(div().flex().flex_none().gap(px(8.)).child(view_seg).child(all_seg)),
            )
            .when_some(self.bench_note.clone(), |d, (tone, msg)| {
                let (_, fg, dot) = tone.colors(&t);
                d.child(
                    div()
                        .px(px(16.))
                        .py(px(8.))
                        .flex()
                        .items_center()
                        .gap(px(8.))
                        .border_b_1()
                        .border_color(t.line_subtle)
                        .child(div().size(px(6.)).flex_none().rounded_full().bg(dot))
                        .child(text(ty::SM, fg).child(msg)),
                )
            })
            .child(body)
            .into_any_element()
    }

    /// Settings › Cloud: the server's cloud connections, each with its account, balance and status,
    /// Check and Disconnect; Connect in the header (docs/CLOUD.md).
    fn cloud_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let now = self.store.api.now();
        let connect = match &self.cloud.providers {
            Load::Ready(p) if !p.is_empty() => self
                .button(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    window,
                    cx,
                    act(|this, w, cx| this.open_connect(w, cx)),
                )
                .into_any_element(),
            Load::Failed(_) => self
                .button_disabled(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    "The server didn't list its cloud providers",
                    window,
                    cx,
                )
                .into_any_element(),
            _ => self
                .button_disabled(
                    "cloud-connect",
                    "Connect",
                    crate::controls::Btn::Secondary,
                    Some("icons/plug.svg"),
                    "Loading the providers…",
                    window,
                    cx,
                )
                .into_any_element(),
        };
        // In the header, or the empty state's one action.
        let mut connect = Some(connect);
        let connections = self.cloud.connections.ready().cloned();
        let body = match &self.cloud.connections {
            Load::Failed(e) => div()
                .p(px(16.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    text(ty::SM, t.danger_text)
                        .flex_1()
                        .min_w_0()
                        .child(format!("Couldn't load the connections: {}", e.message)),
                )
                .child(self.button(
                    "cloud-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.load_connections(cx)),
                ))
                .into_any_element(),
            Load::Ready(list) if list.is_empty() => empty_state(
                &t,
                "icons/cloud.svg",
                "No cloud connected",
                "Connect RunPod to train on a rented GPU, or another FOLO server of yours.",
                Tone::Neutral,
            )
            .py(px(28.))
            .children(connect.take())
            .into_any_element(),
            Load::Ready(_) => div()
                .flex()
                .flex_col()
                .children(
                    connections.unwrap_or_default().iter().map(|c| self.connection_row(c, now, window, cx)),
                )
                .into_any_element(),
            _ => loading_rows(&t, "cloud-sk", 2).into_any_element(),
        };
        panel(&t)
            .flex()
            .flex_col()
            .overflow_hidden()
            .child(panel_header(&t, "Cloud").children(connect))
            .child(body)
            .into_any_element()
    }

    /// One connection: name (and its provider when they differ), status, who and what it is, the
    /// last check's error or a note, then Check and Disconnect.
    fn connection_row(
        &mut self,
        c: &crate::api::CloudConnection,
        now: i64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let t = theme(cx);
        let failed = c.status == crate::api::CloudConnectionStatus::Error;
        let provider = self.provider_name(&c.provider);
        let mut facts: Vec<String> = c.account.iter().cloned().collect();
        if let Some(b) = c.balance_usd {
            facts.push(format!("Balance {}", fmt_cost(b)));
        }
        if let Some(at) = c.checked.as_deref().and_then(crate::api::time::parse) {
            facts.push(format!("checked {}", crate::api::time::ago(at, now)));
        }
        let (dot_c, status, status_c) = if failed {
            (t.danger_solid, "Error", t.danger_text)
        } else {
            (t.success_solid, "Connected", t.text_secondary)
        };
        let id = c.id.clone();
        let check = if self.cloud.checking.contains(&c.id) {
            self.button_disabled(
                &format!("cloud-check-{}", c.id),
                "Checking…",
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                "Checking the credentials with the provider",
                window,
                cx,
            )
        } else {
            let id = id.clone();
            self.button(
                &format!("cloud-check-{}", c.id),
                if failed { "Check again" } else { "Check" },
                crate::controls::Btn::Secondary,
                Some("icons/refresh-cw.svg"),
                window,
                cx,
                act(move |this, _, cx| this.check_connection(id.clone(), cx)),
            )
            .tooltip(crate::controls::tip("Checks the credentials with the provider again", None))
        };
        let disconnect = self.button(
            &format!("cloud-disc-{}", c.id),
            "Disconnect",
            crate::controls::Btn::Secondary,
            None,
            window,
            cx,
            act(move |this, w, cx| this.ask_disconnect(id.clone(), w, cx)),
        );
        div()
            .flex()
            .items_center()
            .gap(px(24.))
            .px(px(16.))
            .min_h(px(56.))
            .py(px(10.))
            .border_b_1()
            .border_color(t.line_subtle)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(px(8.))
                            .child(
                                text(ty::BASE, t.text_primary)
                                    .font_weight(FontWeight::MEDIUM)
                                    .child(c.name.clone()),
                            )
                            .when(c.name != provider, |d| {
                                d.child(text(ty::SM, t.text_tertiary).child(provider))
                            })
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(div().size(px(6.)).flex_none().rounded_full().bg(dot_c))
                                    .child(text(ty::SM, status_c).child(status)),
                            ),
                    )
                    .when(!facts.is_empty(), |d| {
                        d.child(text(ty::SM, t.text_tertiary).tnum().ellipsis().child(facts.join(" · ")))
                    })
                    .when_some(c.error.clone().filter(|_| failed), |d, e| {
                        d.child(text(ty::SM, t.danger_text).whitespace_normal().child(e))
                    })
                    .when_some(c.note.clone(), |d, n| {
                        d.child(
                            div()
                                .flex()
                                .items_start()
                                .gap(px(6.))
                                .pt(px(2.))
                                .child(div().pt(px(2.)).child(icon("icons/info.svg", 12., t.text_tertiary)))
                                .child(text(ty::SM, t.text_tertiary).whitespace_normal().child(n)),
                        )
                    }),
            )
            .child(div().flex().flex_none().items_center().gap(px(8.)).child(check).child(disconnect))
            .into_any_element()
    }

    /// Settings › Cloud › Instances: every machine FOLO launched (alive first, then the last 7
    /// days'), with its run, cost so far and deadline, and Terminate for the alive ones. Hidden
    /// while there are none and no connection provisions any.
    fn instances_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Option<AnyElement> {
        let t = theme(cx);
        let provisions = |p: &str| {
            self.cloud.providers.ready().is_some_and(|ps| ps.iter().any(|x| x.id == p && x.provisions))
        };
        let any_provisioning =
            self.cloud.connections.ready().is_some_and(|l| l.iter().any(|c| provisions(&c.provider)));
        let instances = self.cloud.instances.ready().cloned();
        if instances.as_ref().is_some_and(|l| l.is_empty()) && !any_provisioning {
            return None;
        }
        // Nothing to say until there's something to show (connections still loading included).
        if self.cloud.instances.is_loading() && !any_provisioning {
            return None;
        }
        let now = self.store.api.now();
        let running = instances
            .as_ref()
            .map_or(0, |l| l.iter().filter(|i| i.status == crate::api::CloudInstanceStatus::Running).count());
        // An alive instance's deadline, "—" for the others.
        let deadline = |i: &crate::api::CloudInstance| {
            i.deadline
                .as_deref()
                .and_then(crate::api::time::parse)
                .filter(|_| i.status.is_alive())
                .map_or("—".to_string(), |d| crate::api::time::local_deadline(d, now))
        };
        // Column widths, measured: Stops by holds its longest reading ("2026-11-08 23:24" when a
        // deadline falls on another day). The GPU line takes what the others leave; when that is
        // less than its longest line, Started (the least needed) goes, as a whole column.
        let list = instances.as_deref().unwrap_or_default();
        let measure = |window: &mut Window, s: &str, size: f32| {
            f32::from(crate::charts::text_width(window, s, size)).ceil()
        };
        // Headings are medium weight: ~4 % wider than the regular face `text_width` measures.
        let heading_w = (measure(window, "Stops by", ty::SM.size) * 1.04).ceil();
        let stops_w =
            list.iter().map(|i| measure(window, &deadline(i), ty::SM.size)).fold(heading_w, f32::max);
        let gpu_need =
            list.iter().map(|i| measure(window, &offer_line(&i.offer), ty::SM.size)).fold(0., f32::max);
        const GAP: f32 = 12.;
        // The panel's border and its rows' padding; Run, Status, Cost and the action, each with a gap.
        let fixed = 2. + 32. + (128. + 100. + 64. + 88. + 4. * GAP) + stops_w + GAP;
        let table_w = self.content_width(window).min(COLUMN_MAX);
        let show_started = table_w - fixed - (72. + GAP) >= gpu_need;
        let cell =
            |w: f32, right: bool, el: gpui::Div| el.w(px(w)).flex_none().when(right, |d| d.text_right());
        let cols = |d: gpui::Div| d.flex().items_center().gap(px(GAP)).px(px(16.)).h(px(36.));
        let head = cols(div())
            .border_b_1()
            .border_color(t.line_subtle)
            .child(th(&t, "GPU").flex_1().min_w_0())
            .child(cell(128., false, th(&t, "Run")))
            .child(cell(100., false, th(&t, "Status")))
            .when(show_started, |d| d.child(cell(72., true, th(&t, "Started"))))
            .child(cell(64., true, th(&t, "Cost")))
            .child(cell(stops_w, true, th(&t, "Stops by")))
            .child(cell(88., false, div()));
        let body = match &self.cloud.instances {
            Load::Failed(e) => div()
                .p(px(16.))
                .flex()
                .items_center()
                .justify_between()
                .gap(px(16.))
                .child(
                    text(ty::SM, t.danger_text)
                        .flex_1()
                        .min_w_0()
                        .child(format!("Couldn't load the instances: {}", e.message)),
                )
                .child(self.button(
                    "cloud-inst-retry",
                    "Retry",
                    crate::controls::Btn::Secondary,
                    Some("icons/refresh-cw.svg"),
                    window,
                    cx,
                    act(|this, _, cx| this.refresh_instances(cx)),
                ))
                .into_any_element(),
            Load::Ready(list) if list.is_empty() => div()
                .p(px(16.))
                .child(text(ty::SM, t.text_tertiary).child(
                    "No instances. FOLO starts one per cloud run and terminates it when the run ends.",
                ))
                .into_any_element(),
            Load::Ready(list) => {
                let list = list.clone();
                let rows: Vec<AnyElement> = list
                    .iter()
                    .map(|i| {
                        let live = i.status.is_alive();
                        let ink = if live { t.text_primary } else { t.text_tertiary };
                        let quiet = if live { t.text_secondary } else { t.text_tertiary };
                        let (dot_c, label_c) = match i.status {
                            crate::api::CloudInstanceStatus::Running => (t.accent_fill(), t.accent_text),
                            crate::api::CloudInstanceStatus::Terminating => (t.warning_solid, t.warning_text),
                            crate::api::CloudInstanceStatus::Lost => (t.danger_solid, t.danger_text),
                            // Launching, starting, terminated, and a newer server's status (counted
                            // alive: it keeps its Terminate): no status hue.
                            _ => (t.text_tertiary, quiet),
                        };
                        let line = offer_line(&i.offer);
                        let run = i.run_id.as_ref().map(|r| {
                            (r.clone(), self.store.run(r).map_or(r.clone(), |run| run.name.clone()))
                        });
                        let run_cell = match run {
                            Some((rid, name)) => {
                                let el = div()
                                    .id(SharedString::from(format!("cloud-run-{}", i.id)))
                                    .rounded(px(radius::SM))
                                    .cursor_pointer()
                                    .hover(|s| s.bg(t.bg_hover))
                                    .child(
                                        text(ty::SM, if live { t.text_secondary } else { t.text_tertiary })
                                            .ellipsis()
                                            .child(name),
                                    );
                                self.bind(
                                    el,
                                    &format!("cloud-run-{}", i.id),
                                    radius::SM,
                                    window,
                                    cx,
                                    act(move |this, w, cx| this.navigate(Route::Run(rid.clone()), w, cx)),
                                )
                                .into_any_element()
                            }
                            None => text(ty::SM, t.text_tertiary).child("—").into_any_element(),
                        };
                        let started = crate::api::time::parse(&i.created)
                            .map_or(String::new(), |c| crate::api::time::ago(c, now));
                        // Every instance that may still cost money can be terminated (again, for
                        // one whose termination isn't confirmed: it is idempotent).
                        let action = if !live {
                            div().into_any_element()
                        } else {
                            let id = i.id.clone();
                            self.button(
                                &format!("cloud-term-{}", i.id),
                                "Terminate",
                                crate::controls::Btn::Secondary,
                                None,
                                window,
                                cx,
                                act(move |this, w, cx| this.ask_terminate(id.clone(), w, cx)),
                            )
                            .into_any_element()
                        };
                        cols(div())
                            .id(SharedString::from(format!("cloud-inst-{}", i.id)))
                            .border_b_1()
                            .border_color(t.line_subtle)
                            .tooltip(crate::controls::tip(format!("{line} · {}", i.name), None))
                            .child(text(ty::SM, ink).flex_1().min_w_0().ellipsis().child(line))
                            .child(cell(128., false, div().child(run_cell)))
                            .child(cell(
                                100.,
                                false,
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(div().size(px(6.)).flex_none().rounded_full().bg(dot_c))
                                    .child(text(ty::SM, label_c).whitespace_nowrap().child(i.status.label())),
                            ))
                            .when(show_started, |d| {
                                d.child(cell(
                                    72.,
                                    true,
                                    text(ty::SM, quiet).tnum().whitespace_nowrap().child(started),
                                ))
                            })
                            .child(cell(
                                64.,
                                true,
                                text(ty::SM, ink).tnum().child(i.cost_usd.map_or("—".to_string(), fmt_cost)),
                            ))
                            .child(cell(
                                stops_w,
                                true,
                                text(ty::SM, quiet).tnum().whitespace_nowrap().child(deadline(i)),
                            ))
                            .child(cell(88., false, div().flex().justify_end().child(action)))
                            .into_any_element()
                    })
                    .collect();
                // A termination the provider hasn't confirmed is a problem, not "done": it leads.
                let stuck: Vec<AnyElement> = list
                    .iter()
                    .filter(|i| i.status == crate::api::CloudInstanceStatus::Terminating && i.error.is_some())
                    .map(|i| {
                        let provider = self.provider_name(&i.provider);
                        // How to find it there.
                        let reference = i
                            .provider_ref
                            .as_ref()
                            .map_or(String::new(), |r| format!(" Its {provider} id is `{r}`."));
                        div()
                            .m(px(12.))
                            .mb(px(4.))
                            .p(px(10.))
                            .flex()
                            .items_start()
                            .gap(px(8.))
                            .rounded(px(radius::MD))
                            .bg(t.warning_muted)
                            .child(icon("icons/triangle-alert.svg", 14., t.warning_solid))
                            .child(
                                rich(
                                    &t,
                                    ty::SM,
                                    t.text_primary,
                                    &format!(
                                        "FOLO could not confirm that `{}` is terminated: {} It keeps trying. Check the provider's console; the instance may still be billed.{reference}",
                                        i.name,
                                        i.error.clone().unwrap_or_default().trim_end_matches('.').to_string() + ".",
                                    ),
                                )
                                .text_left()
                                .flex_1()
                                .min_w_0(),
                            )
                            .into_any_element()
                    })
                    .collect();
                div().flex().flex_col().children(stuck).child(head).children(rows).into_any_element()
            }
            _ => loading_rows(&t, "cloud-inst-sk", 2).into_any_element(),
        };
        Some(
            panel(&t)
                .flex()
                .flex_col()
                .overflow_hidden()
                .child(panel_header(&t, "Instances").when(running > 0, |d| {
                    d.child(
                        text(ty::SM, t.text_tertiary)
                            .child(format!("{running} running · refreshes every 5 s")),
                    )
                }))
                .child(body)
                .into_any_element(),
        )
    }

    pub fn load_throughput(&mut self, cx: &mut Context<Self>) {
        self.throughput = Load::Loading;
        self.fetch(cx, |api| api.throughput(), |this, r, _| this.throughput = r.into());
    }

    /// Runs the server's quick probe; the table reloads with its rows, and the panel says what it
    /// measured (or why the server declined).
    fn measure_now(&mut self, cx: &mut Context<Self>) {
        if self.bench_busy {
            return;
        }
        self.bench_busy = true;
        self.bench_note = None;
        cx.notify();
        let req = crate::api::BenchmarkRequest::default();
        self.fetch(
            cx,
            move |api| api.benchmark(&req),
            |this, r, cx| {
                this.bench_busy = false;
                this.bench_note = Some(match r {
                    Ok(b) => (
                        Tone::Success,
                        format!(
                            "Measured {} at {} px: {} img/s training · {:.1} ms per prediction",
                            b.training.arch.as_str(),
                            b.training.imgsz,
                            int(b.train_imgs_per_s.round() as u64),
                            b.latency_ms
                        ),
                    ),
                    Err(e) if e.code == "run_active" => (
                        Tone::Warning,
                        "A run is queued or training: measure when the queue is empty.".to_string(),
                    ),
                    Err(e) => (Tone::Danger, format!("Couldn't measure: {}", e.message)),
                });
                this.load_throughput(cx);
            },
        );
    }
}
