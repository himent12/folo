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
use crate::app::{FoloApp, ServerMode, act};
use crate::store::Load;
use crate::theme::{Mode, radius, theme, ty};
use crate::ui::*;

impl FoloApp {
    pub fn settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let t = theme(cx);
        let header = page_header(
            &t,
            "Settings",
            "Connection, this machine's speed, appearance and keyboard shortcuts.",
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
            // Capped at 880 px: wide enough not to read as a narrow form, close enough that a label
            // and its control (right edge) stay one glance apart on big windows.
            .w_full()
            .max_w(px(880.))
            .child(connection)
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
        let (n_train, n_infer) = table.as_ref().map_or((0, 0), |tb| {
            let measured = |s: ThroughputSource| s == ThroughputSource::Measured;
            (
                tb.training.iter().filter(|r| measured(r.source)).count(),
                tb.inference.iter().filter(|r| measured(r.source)).count(),
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
                .when_some(note, |d, n| d.tooltip(crate::controls::tip(n, None)))
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
                                .child(source),
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
        let source = |measured: bool, samples: u64, probes: u64, on: &str, this_gpu: bool| {
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
            match &device_name {
                Some(d) if this_gpu && on != d => {
                    format!("Measured on {} · {what}", crate::ui::short_gpu(on))
                }
                _ => format!("Measured · {what}"),
            }
        };
        let this_device = table.as_ref().map(|tb| tb.device.clone()).unwrap_or_default();
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
                        source(measured, r.samples, 0, &r.measured_on, r.device == this_device),
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
                        source(measured, r.samples, r.probes, &r.measured_on, r.device == this_device),
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
                .child(text(ty::SM, t.danger_text).child(e.message.clone()))
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
