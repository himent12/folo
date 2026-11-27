//! Agent labeling screens: job list (`list.rs`), job detail (`page.rs`), image review workspace
//! (`review.rs`, `notices.rs`, `filmstrip.rs`, `stage.rs`, `box_rail.rs`), agent runner rail
//! (`agent_panel.rs`), and inline forms (`forms.rs`).

mod agent_panel;
mod box_rail;
mod filmstrip;
mod forms;
mod list;
mod notices;
mod page;
mod review;
mod stage;

#[cfg(test)]
pub(crate) use filmstrip::filmstrip_max_tiles;

use folo_ui::{AnyElement, Context, IntoElement, ParentElement, Window};

use crate::api::*;
use crate::app::{FoloApp, Route, act};
use crate::controls::Btn;
use crate::theme::Theme;
use crate::ui::*;

/// Status badge `(label, tone, kind)` for a labeling job.
pub(super) fn job_status(job: &LabelJob) -> (&'static str, Tone, StatusKind) {
    match job.status {
        LabelJobStatus::Running => ("Labeling", Tone::Accent, StatusKind::Running),
        LabelJobStatus::Stopping => ("Stopping", Tone::Warning, StatusKind::Running),
        LabelJobStatus::Idle => {
            if job.run.as_ref().is_some_and(|r| r.outcome == Some(LabelOutcome::Failed)) {
                ("Failed", Tone::Danger, StatusKind::Failed)
            } else if job.counts.images > 0 && job.counts.labeled >= job.counts.images {
                ("Labeled", Tone::Success, StatusKind::Completed)
            } else {
                ("Idle", Tone::Neutral, StatusKind::Stopped)
            }
        }
    }
}

/// Formats an agent run's display summary (`"Claude Code · Sonnet"`).
pub(super) fn run_agent_summary(run: &LabelRun, agents: Option<&[LabelAgent]>) -> String {
    let agent_name = run.agent.display_name();
    let model = match (&run.model, agents) {
        (Some(id), Some(list)) => list
            .iter()
            .find(|a| a.id == run.agent)
            .and_then(|a| a.models.iter().find(|m| m.id == *id))
            .map_or_else(|| id.clone(), |m| m.label.clone()),
        (Some(id), None) => id.clone(),
        (None, _) => "default".to_string(),
    };
    format!("{agent_name} · {model}")
}

/// Formats token/call usage into a compact human-readable line.
pub(super) fn format_usage(u: &LabelUsage) -> String {
    let mut parts = vec![
        counted(u.calls, "call", "calls"),
        format!("{} in / {} out tok", int(u.input_tokens), int(u.output_tokens)),
    ];
    if let Some(c) = u.cost_usd.filter(|v| v.is_finite() && *v >= 0.0) {
        parts.push(format!("${c:.3}"));
    }
    parts.join(" · ")
}

/// Number of images still needing labels (`pending + failed`).
pub(super) fn unlabeled_count(job: &LabelJob) -> u64 {
    job.counts.pending + job.counts.failed
}

impl FoloApp {
    /// Shown in place of the Labeling screens when the connected server does not advertise `agent_labeling`.
    pub(super) fn label_unsupported_panel(
        &self,
        t: &Theme,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let settings = self.button(
            "label-settings",
            "Connection settings",
            Btn::Secondary,
            Some("icons/settings.svg"),
            window,
            cx,
            act(|this, w, cx| this.navigate(Route::Settings, w, cx)),
        );
        panel(t)
            .child(
                empty_state(
                    t,
                    "icons/lock.svg",
                    "This server has no agent labeling",
                    "Agent labeling needs a real folo serve (not folo serve --mock), or the built-in demo. Start folo serve without --mock or switch data source in Settings.",
                    Tone::Neutral,
                )
                .child(settings),
            )
            .into_any_element()
    }
}
