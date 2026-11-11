//! Benchmarks as text: axis names and scores, contestants, sources, status tones, ranks and names.

use crate::api::{
    Bench, BenchAxis, BenchContestant, BenchEntry, BenchEntryStatus, BenchLoopStatus, BenchSource, Label,
    LabelAgentId,
};
use crate::app::FoloApp;
use crate::ui::*;

/// An axis's short name, as the leaderboard heads its column.
pub fn axis_short(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "Accuracy",
        BenchAxis::Robustness => "Robust.",
        BenchAxis::SelfLabeling => "Self-label",
        BenchAxis::Calibration => "Calib.",
        BenchAxis::Localization => "Local.",
    }
}

/// An axis's full name.
pub fn axis_name(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "Accuracy",
        BenchAxis::Robustness => "Robustness",
        BenchAxis::SelfLabeling => "Self-labeling",
        BenchAxis::Calibration => "Calibration",
        BenchAxis::Localization => "Localization",
    }
}

/// An axis value (0..=1) on the score's 0–100 scale; "–" when it was not measured.
pub fn axis_text(v: Option<f32>) -> String {
    v.map_or_else(|| "–".to_string(), |v| format!("{:.0}", (v * 100.).clamp(0., 100.)))
}

/// A share as a percentage: 0.834 → "83%".
pub fn pct(v: f32) -> String {
    format!("{:.0}%", (v * 100.).clamp(0., 100.))
}

/// Who an entry is, in a few words: "folo-n · 640 px", "Claude Code · sonnet", "External model".
pub fn contestant_line(e: &BenchEntry) -> String {
    match &e.contestant {
        BenchContestant::Model { arch, imgsz, .. } => format!("{arch} · {imgsz} px"),
        BenchContestant::Agent { agent, model, effort } => {
            let who = agent_name(*agent);
            let model = model.as_deref().unwrap_or("default model");
            format!("{who} · {model} · {} effort", effort.label().to_lowercase())
        }
        BenchContestant::External => "External model".to_string(),
    }
}

/// A coding agent's product name.
pub fn agent_name(a: LabelAgentId) -> &'static str {
    match a {
        LabelAgentId::Claude => "Claude Code",
        LabelAgentId::Codex => "Codex",
    }
}

/// Where a benchmark's images came from: "shapes-1k · val".
pub fn source_line(b: &Bench) -> String {
    match &b.source {
        BenchSource::Dataset { split, .. } => {
            format!("{} · {}", b.source_name, split.map_or("val", |s| s.as_str()))
        }
        BenchSource::LabelJob { .. } => format!("{} · labeling job", b.source_name),
    }
}

/// The badge tone of an entry's status.
pub fn entry_tone(s: BenchEntryStatus) -> Tone {
    match s {
        BenchEntryStatus::Queued | BenchEntryStatus::Stopping => Tone::Neutral,
        BenchEntryStatus::Running => Tone::Accent,
        BenchEntryStatus::Done => Tone::Success,
        BenchEntryStatus::Failed => Tone::Danger,
        BenchEntryStatus::Stopped => Tone::Warning,
    }
}

/// The badge tone of a loop's status.
pub fn loop_tone(s: BenchLoopStatus) -> Tone {
    match s {
        BenchLoopStatus::Running => Tone::Accent,
        BenchLoopStatus::Stopping => Tone::Neutral,
        BenchLoopStatus::Done => Tone::Success,
        BenchLoopStatus::Failed => Tone::Danger,
        BenchLoopStatus::Stopped => Tone::Warning,
    }
}

/// How far a running entry is, 0..=1.
pub fn entry_frac(e: &BenchEntry) -> f32 {
    if e.progress.total == 0 { 0. } else { (e.progress.done as f32 / e.progress.total as f32).clamp(0., 1.) }
}

/// The leaderboard's ranks: finished entries with the benchmark's fingerprint, in the server's order
/// (complete ones first, then partial ones); the rest are not ranked.
pub fn ranks(b: &Bench, entries: &[BenchEntry]) -> Vec<Option<usize>> {
    let mut n = 0;
    entries
        .iter()
        .map(|e| {
            (e.status == BenchEntryStatus::Done && e.fingerprint == b.fingerprint).then(|| {
                n += 1;
                n
            })
        })
        .collect()
}

impl FoloApp {
    /// The display name of a benchmark (its id until the list is in).
    pub fn bench_name(&self, id: &str) -> String {
        self.bench
            .list
            .ready()
            .and_then(|l| l.iter().find(|b| b.id == id))
            .map_or(id.to_string(), |b| b.name.clone())
    }

    /// The display name of an entry (its id until the leaderboard is in).
    pub fn bench_entry_name(&self, bench: &str, entry: &str) -> String {
        self.bench
            .entries
            .get(bench)
            .and_then(|l| l.ready())
            .and_then(|l| l.iter().find(|e| e.id == entry))
            .map_or(entry.to_string(), |e| e.name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_read_on_the_score_scale() {
        assert_eq!(axis_text(Some(0.834)), "83");
        assert_eq!(axis_text(None), "–");
        assert_eq!(axis_text(Some(1.2)), "100");
        assert_eq!(pct(0.5), "50%");
    }
}
