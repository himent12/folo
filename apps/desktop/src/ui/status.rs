//! Tones and status: the tone palette, status dots, badges and tags.

use std::time::Duration;

use folo_ui::{
    Animation, AnimationExt, Div, FontWeight, Hsla, IntoElement, ParentElement, SharedString, Styled, div, px,
};

use super::motion::reduce_motion;
use super::text::text;
use crate::api::{Label, Run, RunExt, RunStatus};
use crate::theme::{Theme, radius, ty};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
    Info,
}

impl Tone {
    pub fn colors(self, t: &Theme) -> (Hsla, Hsla, Hsla) {
        // (background, text, dot)
        match self {
            Tone::Neutral => (t.bg_active, t.text_secondary, t.text_tertiary),
            Tone::Accent => (t.accent_muted, t.accent_text, t.accent_fill()),
            Tone::Success => (t.success_muted, t.success_text, t.success_solid),
            Tone::Warning => (t.warning_muted, t.warning_text, t.warning_solid),
            Tone::Danger => (t.danger_muted, t.danger_text, t.danger_solid),
            Tone::Info => (t.info_muted, t.info_text, t.info_solid),
        }
    }
}

pub fn status_tone(s: RunStatus) -> Tone {
    match s {
        RunStatus::Running => Tone::Accent,
        RunStatus::Completed => Tone::Success,
        RunStatus::Failed => Tone::Danger,
        RunStatus::Stopping => Tone::Warning,
        // Waiting is not a status hue (design/README): neutral grey.
        RunStatus::Queued => Tone::Neutral,
        RunStatus::Stopped => Tone::Neutral,
    }
}

/// A small status dot. `live` breathes: the dot itself eases dimmer and back, with nothing drawn around it (a ring
/// growing out of every live dot read as a row of little O's).
pub fn dot(color: Hsla, size: f32, live: bool, id: impl Into<SharedString>) -> folo_ui::AnyElement {
    let dot = div().size(px(size)).flex_none().rounded_full().bg(color);
    if !live || reduce_motion() {
        return dot.into_any_element();
    }
    let id: SharedString = id.into();
    dot.with_animation(
        folo_ui::ElementId::Name(format!("breathe-{id}").into()),
        Animation::new(Duration::from_millis(2400)).repeat(),
        // One cosine period per loop, so the loop has no seam: full, down to 40 %, full.
        |d, p| d.opacity(0.7 + 0.3 * (p * std::f32::consts::TAU).cos()),
    )
    .into_any_element()
}

/// The compact end-to-end tag in tables (the title rows carry the full badge and its help).
pub fn e2e_tag(t: &Theme, id: impl Into<SharedString>) -> folo_ui::Stateful<Div> {
    use folo_ui::{InteractiveElement, StatefulInteractiveElement};
    div()
        .id(id.into())
        .tooltip(crate::controls::tip(crate::app::FoloApp::E2E_HELP, None))
        .flex_none()
        .px(px(4.))
        .rounded(px(crate::theme::radius::XS))
        .bg(t.bg_active)
        .child(text(ty::SM, t.text_secondary).child("e2e"))
}

pub fn badge(t: &Theme, label: impl Into<SharedString>, tone: Tone) -> Div {
    let (bg, fg, _) = tone.colors(t);
    text(ty::SM, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .px(px(6.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(5.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(label.into())
}

/// A run's status in one word, wherever a status is drawn (badges, rows, the palette): the status,
/// or for a cloud run handing over to its instance its phase ("Provisioning", "Booting", "Uploading
/// data"), since it waits in no queue. Such a run keeps a waiting run's look (`status_tone`).
pub fn status_label(r: &Run) -> &'static str {
    r.cloud_pending().map_or(r.status.label(), |c| c.phase.label())
}

/// Status for dense rows: a dot and a word, no tinted pill.
pub fn status_dot(t: &Theme, r: &Run, id: &str) -> Div {
    let s = r.status;
    let (_, fg, dotc) = status_tone(s).colors(t);
    div()
        .flex()
        .items_center()
        .gap(px(6.))
        .child(if s == RunStatus::Queued {
            // Waiting, not halted: a ring where a stopped run has a filled dot.
            div().size(px(6.)).flex_none().rounded_full().border_1().border_color(dotc).into_any_element()
        } else {
            dot(dotc, 6., s == RunStatus::Running, format!("sd-{id}")).into_any_element()
        })
        .child(
            text(ty::SM, if s == RunStatus::Running { t.accent_text } else { fg })
                .whitespace_nowrap()
                .child(status_label(r)),
        )
}

/// Status chip for a run: a dot (pulsing while running) and the status word.
pub fn status_badge(t: &Theme, r: &Run, id: &str) -> Div {
    let s = r.status;
    let tone = status_tone(s);
    let (_, _, dotc) = tone.colors(t);
    let (bg, fg, _) = tone.colors(t);
    text(ty::SM, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .pl(px(6.))
        .pr(px(7.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(6.))
        .rounded(px(radius::XS))
        .bg(bg)
        .whitespace_nowrap()
        .child(dot(dotc, 6., s == RunStatus::Running, format!("badge-{id}")))
        .child(status_label(r))
}
