//! Settings › Device throughput: the server's measured speeds and the reference figures.

use std::rc::Rc;

use gpui::prelude::FluentBuilder;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};

use crate::api::ThroughputSource;
use crate::app::{FoloApp, act};
use crate::screens::th;
use crate::store::Load;
use crate::theme::{theme, ty};
use crate::ui::*;

impl FoloApp {
    /// The server's measured speeds, and the reference figures for everything it hasn't measured,
    /// each row saying which it is and from how many samples. "Measure now" runs the server's quick
    /// probe (servers with `benchmark`).
    pub(super) fn throughput_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
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
