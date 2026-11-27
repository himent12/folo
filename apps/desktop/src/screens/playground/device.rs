//! Where the model runs: the server, the local GPU or the local CPU.

use std::rc::Rc;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    AnyElement, Context, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px, relative,
};

use crate::api::PredictResponse;
use crate::app::FoloApp;
use crate::store::Load;
use crate::theme::{Theme, theme, ty};
use crate::ui::*;

impl FoloApp {
    /// Server / Local GPU / Local CPU, with what each means here.
    pub(super) fn run_on(&mut self, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        use crate::local::Target;
        let t = theme(cx);
        let local = self.store.api.local_inference();
        let no_local: Option<SharedString> =
            (!local).then(|| "Needs folo serve: the built-in demo has no weights to run here".into());
        let gpu_why = no_local.clone().or_else(|| match &self.local_gpu {
            Load::Ready(_) => None,
            Load::Failed(_) => Some("No GPU this app can use was found".into()),
            _ => Some("Looking for a GPU…".into()),
        });
        let selected = match self.play_target {
            None => 0,
            Some(Target::Gpu) => 1,
            Some(Target::Cpu) => 2,
        };
        let control = self.segmented_fill_with(
            "play-target",
            // The built-in demo isn't a server: say what actually runs.
            &[if local { "Server" } else { "Demo" }, "Local GPU", "Local CPU"],
            selected,
            &[None, gpu_why, no_local],
            window,
            cx,
            Rc::new(|this, i, _, cx| {
                let target = match i {
                    1 => Some(Target::Gpu),
                    2 => Some(Target::Cpu),
                    _ => None,
                };
                this.set_play_target(target, cx);
            }),
        );
        let help = self.help_icon(
            "help-run-on",
            "Local runs download the model's weights once (f16, checked for changes before each run) and run them in this app, leaving the server's GPU to training.",
            window,
            cx,
        );
        let note = match (self.play_target, &self.local_gpu) {
            // The caption also says why Local GPU and CPU are off, so keyboard users (whose arrows
            // skip disabled options) learn it without the tooltip.
            (None, _) if !local => "Simulated demo · local runs need folo serve".into(),
            (None, _) => format!("On folo serve at {}", self.store.api.describe()),
            (Some(Target::Gpu), Load::Ready(name)) => {
                format!("In this app, on {}", crate::ui::short_gpu(name))
            }
            (Some(Target::Gpu), _) => "In this app, on this PC's GPU".into(),
            (Some(Target::Cpu), _) => "In this app, on the CPU".into(),
        };
        div()
            .flex()
            .flex_col()
            .gap(px(6.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(4.))
                    .child(text(ty::SM, t.text_secondary).font_weight(FontWeight::MEDIUM).child("Run on"))
                    .child(help),
            )
            .child(control)
            // Wraps rather than truncating: text only ellipsizes against a known width, so a
            // w_full note would hard-clip mid-word in a narrow rail.
            .child(text(ty::SM, t.text_tertiary).w_full().whitespace_normal().child(note))
            .into_any_element()
    }
}

/// "gpu:0" as the adapter's name ("RTX 4080 SUPER"), "cpu" as "CPU"; anything else as given.
pub(super) fn device_name(device: &str, system: Option<&crate::api::SystemInfo>) -> String {
    // Local inference names its device itself.
    if let Some(name) = device.strip_prefix("local:") {
        return format!("{name} in this app");
    }
    if device == "cpu" {
        return "CPU".into();
    }
    device
        .strip_prefix("gpu:")
        .and_then(|n| n.parse::<u32>().ok())
        .and_then(|n| system?.gpus.iter().find(|g| g.ordinal == n))
        .map_or(device.to_string(), |g| crate::ui::short_gpu(&g.name))
}

/// Latency breakdown strip at the foot of the Playground side rail.
pub(super) fn play_timing_panel(
    t: &Theme,
    p: &PredictResponse,
    system: Option<&crate::api::SystemInfo>,
    simulated: bool,
    e2e: bool,
) -> AnyElement {
    let total = p.timing.total_ms.max(0.001);
    let seg = |v: f64, c| {
        let frac = if v.is_finite() { (v / total).clamp(0.0, 1.0) as f32 } else { 0.0 };
        div().h_full().w(relative(frac)).bg(c)
    };
    let (c_pre, c_model, c_nms) = (
        crate::theme::alpha(t.text_tertiary, if t.is_dark() { 0.7 } else { 0.8 }),
        t.text_secondary,
        crate::theme::alpha(t.text_tertiary, 0.3),
    );
    div()
        .w_full()
        .flex()
        .flex_col()
        .gap(px(8.))
        .px(px(16.))
        .py(px(12.))
        .border_t_1()
        .border_color(t.line_subtle)
        .child(
            div()
                .flex()
                .justify_between()
                .items_start()
                .gap(px(8.))
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(px(2.))
                        .child(
                            text(ty::SM, t.text_secondary)
                                .font_weight(FontWeight::MEDIUM)
                                .child(if simulated { "Simulated latency" } else { "Latency" }),
                        )
                        .child(text(ty::SM, t.text_tertiary).tnum().whitespace_normal().child(
                            if simulated {
                                "one image".to_string()
                            } else {
                                format!("{} · one image", device_name(&p.device, system))
                            },
                        )),
                )
                .child(metric(t, format!("{:.1}", p.timing.total_ms), "ms", ty::LG, ty::SM)),
        )
        .child(
            div()
                .flex()
                .h(px(6.))
                .gap(px(2.))
                .rounded_full()
                .overflow_hidden()
                .child(seg(p.timing.preprocess_ms, c_pre))
                .child(seg(p.timing.inference_ms, c_model))
                .child(seg(p.timing.postprocess_ms, c_nms).border_1().border_color(t.text_tertiary)),
        )
        .child(
            div().flex().flex_wrap().gap_x(px(12.)).gap_y(px(4.)).children(
                [
                    ("Pre", "Preprocess", p.timing.preprocess_ms, c_pre),
                    ("Infer", "Inference", p.timing.inference_ms, c_model),
                    (
                        "Post",
                        if e2e {
                            "Postprocess · box decoding"
                        } else {
                            "Postprocess · box decoding and NMS"
                        },
                        p.timing.postprocess_ms,
                        c_nms,
                    ),
                ]
                .into_iter()
                .map(|(n, full, v, c)| {
                    div()
                        .id(SharedString::from(format!("lat-{n}")))
                        .tooltip(crate::controls::tip(full, None))
                        .flex()
                        .flex_none()
                        .items_center()
                        .gap(px(5.))
                        .whitespace_nowrap()
                        .child(
                            swatch(c, 8.).when(n == "Post", |d| d.border_1().border_color(t.text_tertiary)),
                        )
                        .child(text(ty::SM, t.text_tertiary).child(n))
                        .child(text(ty::SM, t.text_secondary).tnum().child(format!("{v:.1} ms")))
                }),
            ),
        )
        .into_any_element()
}
