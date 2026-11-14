//! Benchmarks as text: axis names and scores, contestants, sources, status tones, ranks and names.

use crate::api::{
    Bench, BenchAxis, BenchContestant, BenchEntry, BenchEntryStatus, BenchLoop, BenchSource, Label,
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

/// A score on its 0–100 scale as the leaderboard shows it: 53.4.
pub fn score_text(total: f32) -> String {
    format!("{total:.1}")
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

/// How far a running entry is, 0..=1.
pub fn entry_frac(e: &BenchEntry) -> f32 {
    if e.progress.total == 0 { 0. } else { (e.progress.done as f32 / e.progress.total as f32).clamp(0., 1.) }
}

/// A loop's accuracy from its teacher to its latest student that was graded: "mAP50-95 0.413 →
/// 0.000"; the teacher's alone until a student is ("teacher mAP50-95 0.413"). A generation that
/// never ran (the loop failed or stopped before it) is not the latest.
pub fn loop_trend(l: &BenchLoop) -> String {
    let first = l.generations.first().and_then(|g| g.map50_95);
    let last = l.generations.iter().rev().filter(|g| g.generation > 0).find_map(|g| g.map50_95);
    match (first, last) {
        (Some(a), Some(z)) => format!("mAP50-95 {} → {}", f3(a), f3(z)),
        (Some(a), None) => format!("teacher mAP50-95 {}", f3(a)),
        _ => "–".to_string(),
    }
}

/// The leaderboard's ranks: finished entries with the benchmark's fingerprint, in the server's order
/// (complete ones first, then partial ones); the rest are not ranked. Entries whose scores read the
/// same (to the tenth shown) and are equally complete share a rank, as in Studio: 1, 1, 3.
pub fn ranks(b: &Bench, entries: &[BenchEntry]) -> Vec<Option<usize>> {
    let mut n = 0;
    let mut prev: Option<((String, bool), usize)> = None;
    entries
        .iter()
        .map(|e| {
            (e.status == BenchEntryStatus::Done && e.fingerprint == b.fingerprint).then(|| {
                n += 1;
                let shown = e.score.as_ref().map(|s| (score_text(s.total), s.partial));
                let rank = match (&prev, &shown) {
                    (Some((p, r)), Some(s)) if p == s => *r,
                    _ => n,
                };
                prev = shown.map(|s| (s, rank));
                rank
            })
        })
        .collect()
}

impl FoloApp {
    /// A condition's display name from the catalog ("Low contrast"), its id until the catalog is in.
    pub fn bench_condition_name(&self, id: &str) -> String {
        self.bench
            .catalog
            .ready()
            .and_then(|c| c.conditions.iter().find(|x| x.id == id))
            .map_or_else(|| id.to_string(), |c| c.name.clone())
    }

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

    /// The demo's first benchmark and its finished entries, in the server's order.
    fn demo_board() -> (Bench, Vec<BenchEntry>) {
        use crate::api::Backend;
        let api = crate::api::mock::MockBackend::new(4200, true);
        let finished = |b: &Bench| -> Vec<BenchEntry> {
            let all = api.bench_entries(&b.id).unwrap().into_iter();
            all.filter(|e| e.status == BenchEntryStatus::Done && e.fingerprint == b.fingerprint).collect()
        };
        let b = api.benches().unwrap().into_iter().max_by_key(|b| finished(b).len()).unwrap();
        let done = finished(&b);
        assert!(done.len() >= 2, "the demo has finished entries to rank");
        (b, done)
    }

    #[test]
    fn equal_scores_share_a_rank_as_in_studio() {
        let (b, mut list) = demo_board();
        // The same weights twice (a run's best and last): the same score.
        let mut twin = list[0].clone();
        twin.id = "be_twin".into();
        list.insert(1, twin);
        let r = ranks(&b, &list);
        assert_eq!(r[0], Some(1));
        assert_eq!(r[1], Some(1), "a tie shares the rank");
        assert_eq!(r[2], Some(3), "and the next one counts both");
        // A partial entry with the same number doesn't tie with a complete one.
        let mut part = list[0].clone();
        part.id = "be_part".into();
        part.score.as_mut().unwrap().partial = true;
        list.insert(1, part);
        assert_eq!(ranks(&b, &list)[1], Some(2));
        // Ties are what the leaderboard shows: 53.35 and 53.30 both read 53.3 (f32), 53.4 doesn't.
        let at = |total: f32| {
            let mut e = list[0].clone();
            e.score.as_mut().unwrap().total = total;
            e
        };
        let shown = vec![at(53.35), at(53.30), at(53.26), at(53.2)];
        assert_eq!(
            shown.iter().map(|e| score_text(e.score.as_ref().unwrap().total)).collect::<Vec<_>>(),
            ["53.3", "53.3", "53.3", "53.2"]
        );
        assert_eq!(ranks(&b, &shown), [Some(1), Some(1), Some(1), Some(4)]);
    }

    #[test]
    fn a_loop_reads_from_its_teacher_to_its_last_graded_student() {
        use crate::api::Backend;
        let api = crate::api::mock::MockBackend::new(4200, true);
        let mut l = api
            .benches()
            .unwrap()
            .iter()
            .flat_map(|b| api.bench_loops(&b.id).unwrap())
            .find(|l| l.generations.first().is_some_and(|g| g.map50_95.is_some()))
            .expect("a demo loop with a graded teacher");
        let teacher = l.generations[0].clone();
        let mut student = teacher.clone();
        (student.generation, student.map50_95) = (1, Some(0.0));
        // Generation 2 never ran: the loop failed before its labels.
        let never = crate::api::BenchGeneration { generation: 2, map50_95: None, ..student.clone() };
        l.generations = vec![teacher.clone(), student, never];
        let t = teacher.map50_95.unwrap();
        assert_eq!(loop_trend(&l), format!("mAP50-95 {} → 0.000", f3(t)));
        l.generations.truncate(1);
        assert_eq!(loop_trend(&l), format!("teacher mAP50-95 {}", f3(t)));
    }
}
