//! Tones and status: the tone palette, status marks (in tables and in badges), badges and tags, and a
//! run's progress in a table.
//!
//! Every status the app shows (a run's, a benchmark entry's or loop's, a cloud instance's) is drawn
//! here, from one [`StatusKind`]: one glyph per state, so the shape carries the meaning and colour is
//! never the only cue (design/UNIFIED.md §2, Status; Studio's `StatusBadge`).

use std::time::Duration;

use folo_ui::prelude::FluentBuilder;
use folo_ui::{
    Animation, AnimationExt, Div, FontWeight, Hsla, IntoElement, ParentElement, SharedString, Styled, div, px,
};

use super::motion::reduce_motion;
use super::surface::progress;
use super::text::{StyledExt, icon, text};
use crate::api::{BenchEntryStatus, BenchLoopStatus, CloudInstanceStatus, Label, Run, RunExt, RunStatus};
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

/// The states a status mark tells apart, whatever has them: a run, a benchmark entry or loop, a
/// cloud instance. Each has its own glyph: queued a hollow ring, running a ringed dot, completed a
/// check, failed a cross, stopped a square; stopping (on its way to stopped) a filled dot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StatusKind {
    Queued,
    Running,
    Stopping,
    Completed,
    Failed,
    Stopped,
}

impl StatusKind {
    pub fn of_run(s: RunStatus) -> Self {
        match s {
            RunStatus::Queued => StatusKind::Queued,
            RunStatus::Running => StatusKind::Running,
            RunStatus::Stopping => StatusKind::Stopping,
            RunStatus::Completed => StatusKind::Completed,
            RunStatus::Failed => StatusKind::Failed,
            RunStatus::Stopped => StatusKind::Stopped,
        }
    }

    pub fn of_entry(s: BenchEntryStatus) -> Self {
        match s {
            BenchEntryStatus::Queued => StatusKind::Queued,
            BenchEntryStatus::Running => StatusKind::Running,
            BenchEntryStatus::Stopping => StatusKind::Stopping,
            BenchEntryStatus::Done => StatusKind::Completed,
            BenchEntryStatus::Failed => StatusKind::Failed,
            BenchEntryStatus::Stopped => StatusKind::Stopped,
        }
    }

    pub fn of_loop(s: BenchLoopStatus) -> Self {
        match s {
            BenchLoopStatus::Running => StatusKind::Running,
            BenchLoopStatus::Stopping => StatusKind::Stopping,
            BenchLoopStatus::Done => StatusKind::Completed,
            BenchLoopStatus::Failed => StatusKind::Failed,
            BenchLoopStatus::Stopped => StatusKind::Stopped,
        }
    }

    /// A cloud instance: waiting while it launches and starts (and a newer server's status, which
    /// is counted alive), running once its worker answers, stopping while it terminates, stopped
    /// once terminated, failed when the provider lost it.
    pub fn of_instance(s: CloudInstanceStatus) -> Self {
        match s {
            CloudInstanceStatus::Running => StatusKind::Running,
            CloudInstanceStatus::Terminating => StatusKind::Stopping,
            CloudInstanceStatus::Terminated => StatusKind::Stopped,
            CloudInstanceStatus::Lost => StatusKind::Failed,
            CloudInstanceStatus::Launching | CloudInstanceStatus::Starting | CloudInstanceStatus::Unknown => {
                StatusKind::Queued
            }
        }
    }

    /// The badge's tone: waiting and stopped are neutral grey, not a status hue (design/README).
    pub fn tone(self) -> Tone {
        match self {
            StatusKind::Queued | StatusKind::Stopped => Tone::Neutral,
            StatusKind::Running => Tone::Accent,
            StatusKind::Stopping => Tone::Warning,
            StatusKind::Completed => Tone::Success,
            StatusKind::Failed => Tone::Danger,
        }
    }

    /// The glyph's colour in a table: the status hue (a mark's `solid`), neutral for queued and
    /// stopped.
    pub fn hue(self, t: &Theme) -> Hsla {
        match self {
            StatusKind::Queued => t.text_secondary,
            StatusKind::Stopped => t.text_tertiary,
            k => k.tone().colors(t).2,
        }
    }

    /// The word's ink in a table: plain, quieter for the states that are not doing anything.
    pub fn ink(self, t: &Theme) -> Hsla {
        match self {
            StatusKind::Queued | StatusKind::Stopped => t.text_secondary,
            _ => t.text_primary,
        }
    }
}

/// A small status dot. `live` breathes: the dot itself eases dimmer and back, with nothing drawn around it (a ring
/// growing out of every live dot read as a row of little O's).
pub fn dot(color: Hsla, size: f32, live: bool, id: impl Into<SharedString>) -> folo_ui::AnyElement {
    breathe(div().size(px(size)).flex_none().rounded_full().bg(color), live, id)
}

/// `el` easing dimmer and back while `live` (off with reduced motion).
fn breathe(el: Div, live: bool, id: impl Into<SharedString>) -> folo_ui::AnyElement {
    if !live || reduce_motion() {
        return el.into_any_element();
    }
    let id: SharedString = id.into();
    // An opacity-only animation: the window re-fades the dot at each refresh without rebuilding the UI.
    el.with_opacity_animation(
        folo_ui::ElementId::Name(format!("breathe-{id}").into()),
        Animation::new(Duration::from_millis(2400)).repeat(),
        // One cosine period per loop, so the loop has no seam: full, down to 40 %, full.
        |p| 0.7 + 0.3 * (p * std::f32::consts::TAU).cos(),
    )
    .into_any_element()
}

/// A status's glyph in `color`, in one 8 px slot whatever the state, so words after it line up. A
/// running one's dot breathes (`id` keys its animation).
pub fn status_mark(kind: StatusKind, color: Hsla, id: impl Into<SharedString>) -> folo_ui::AnyElement {
    let slot = div().size(px(8.)).flex_none().flex().items_center().justify_center();
    let mark = match kind {
        // Waiting, not halted: a ring where a stopped one has a square.
        StatusKind::Queued => {
            div().size(px(6.)).rounded_full().border_1().border_color(color).into_any_element()
        }
        // A dot in a ring: it reads "live" even when the breathing is off for reduced motion.
        StatusKind::Running => div()
            .size(px(8.))
            .rounded_full()
            .border_1()
            .border_color(color)
            .flex()
            .items_center()
            .justify_center()
            .child(breathe(div().size(px(4.)).rounded_full().bg(color), true, id))
            .into_any_element(),
        StatusKind::Stopping => div().size(px(6.)).rounded_full().bg(color).into_any_element(),
        StatusKind::Completed => icon("icons/status-check.svg", 8., color).into_any_element(),
        StatusKind::Failed => icon("icons/status-cross.svg", 8., color).into_any_element(),
        StatusKind::Stopped => div().size(px(6.)).rounded(px(1.5)).bg(color).into_any_element(),
    };
    slot.child(mark).into_any_element()
}

/// Status in a table cell: the glyph in the status hue, the word in plain ink.
pub fn status_cell(
    t: &Theme,
    kind: StatusKind,
    word: impl Into<SharedString>,
    id: impl Into<SharedString>,
) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(6.))
        .min_w_0()
        .child(status_mark(kind, kind.hue(t), id))
        .child(text(ty::SM, kind.ink(t)).whitespace_nowrap().child(word.into()))
}

/// Status on a detail header: the glyph and the word in the status's ink on its muted fill, 20 high.
pub fn status_badge_of(
    t: &Theme,
    kind: StatusKind,
    word: impl Into<SharedString>,
    id: impl Into<SharedString>,
) -> Div {
    let (bg, fg, _) = kind.tone().colors(t);
    text(ty::SM, fg)
        .font_weight(FontWeight::MEDIUM)
        .h(px(20.))
        .pl(px(6.))
        .pr(px(8.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(6.))
        .rounded(px(radius::SM))
        .bg(bg)
        .whitespace_nowrap()
        .child(status_mark(kind, fg, id))
        .child(word.into())
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

/// [`badge`] with a mark before its word, in the word's colour ("Worker not answering").
pub fn badge_icon(t: &Theme, mark: &'static str, label: impl Into<SharedString>, tone: Tone) -> Div {
    let (bg, fg, _) = tone.colors(t);
    div()
        .h(px(20.))
        .px(px(6.))
        .flex()
        .flex_none()
        .items_center()
        .gap(px(5.))
        .rounded(px(radius::XS))
        .bg(bg)
        .child(icon(mark, 12., fg))
        .child(text(ty::SM, fg).font_weight(FontWeight::MEDIUM).whitespace_nowrap().child(label.into()))
}

/// A run's status in one word, wherever a status is drawn (badges, rows, the palette): the status,
/// or for a cloud run handing over to its instance its phase ("Provisioning", "Booting", "Uploading
/// data"), since it waits in no queue. Such a run keeps a waiting run's look (its status's mark).
pub fn status_label(r: &Run) -> &'static str {
    r.cloud_pending().map_or(r.status.label(), |c| c.phase.label())
}

/// A run's status in a table row: its glyph in the status hue and its word in plain ink.
pub fn status_dot(t: &Theme, r: &Run, id: &str) -> Div {
    status_cell(t, StatusKind::of_run(r.status), status_label(r), format!("sd-{id}"))
}

/// A run's status badge, for the header of the one run a screen is about.
pub fn status_badge(t: &Theme, r: &Run, id: &str) -> Div {
    status_badge_of(t, StatusKind::of_run(r.status), status_label(r), format!("badge-{id}"))
}

/// Progress in a table (design §2, Progress): a 2 px rail, lime only while `live`, tertiary
/// otherwise, then "42 / 100" in a `num_w` column. `rail: false` keeps the number alone where a
/// rail would be a stub.
pub fn progress_cell(t: &Theme, frac: f32, done: u32, total: u32, live: bool, rail: bool, num_w: f32) -> Div {
    div()
        .flex()
        .items_center()
        .gap(px(8.))
        .when(rail, |d| {
            d.child(div().flex_1().min_w_0().child(progress(
                t,
                frac,
                if live { t.accent_fill() } else { t.text_tertiary },
                2.,
            )))
        })
        .when(!rail, |d| d.child(div().flex_1()))
        .child(
            text(ty::SM, t.text_secondary)
                .tnum()
                .w(px(num_w))
                .flex_none()
                .whitespace_nowrap()
                .text_right()
                .child(format!("{done} / {total}")),
        )
}

/// [`progress_cell`] for a run: live only while it trains.
pub fn run_progress_cell(t: &Theme, r: &Run, rail: bool, num_w: f32) -> Div {
    progress_cell(t, r.progress(), r.epoch, r.epochs, r.status == RunStatus::Running, rail, num_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_state_has_its_own_glyph_and_waiting_is_neutral() {
        let kinds = [
            StatusKind::Queued,
            StatusKind::Running,
            StatusKind::Stopping,
            StatusKind::Completed,
            StatusKind::Failed,
            StatusKind::Stopped,
        ];
        // A run, an entry and a loop in the same state draw the same mark with the same tone.
        assert_eq!(StatusKind::of_entry(BenchEntryStatus::Done), StatusKind::of_run(RunStatus::Completed));
        assert_eq!(StatusKind::of_loop(BenchLoopStatus::Stopped), StatusKind::of_run(RunStatus::Stopped));
        assert_eq!(StatusKind::of_entry(BenchEntryStatus::Stopped).tone(), Tone::Neutral);
        assert_eq!(StatusKind::of_instance(CloudInstanceStatus::Starting), StatusKind::Queued);
        assert_eq!(StatusKind::of_instance(CloudInstanceStatus::Lost), StatusKind::Failed);
        assert_eq!(StatusKind::of_run(RunStatus::Queued).tone(), Tone::Neutral);
        let t = Theme::new(crate::theme::Mode::Dark);
        for k in kinds {
            // The word is plain ink, never the status hue.
            let ink = k.ink(&t);
            assert!(ink == t.text_primary || ink == t.text_secondary, "{k:?}");
        }
        assert_eq!(StatusKind::Running.hue(&t), t.accent_fill());
        assert_eq!(StatusKind::Failed.hue(&t), t.danger_solid);
    }
}
