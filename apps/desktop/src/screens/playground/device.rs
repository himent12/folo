//! Where the model runs: the server, the local GPU or the local CPU.

use std::rc::Rc;

use folo_ui::{
    AnyElement, Context, FontWeight, IntoElement, ParentElement, SharedString, Styled, Window, div, px,
};

use crate::app::FoloApp;
use crate::store::Load;
use crate::theme::{theme, ty};
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
