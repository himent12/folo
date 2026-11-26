//! Benchmarks as text: axis names and scores, contestants, sources, status tones, ranks and names.

use crate::api::{
    Bench, BenchAccuracy, BenchAxis, BenchContestant, BenchEntry, BenchEntryImagePage, BenchEntryStatus,
    BenchLoop, BenchPlan, BenchSource, Label, LabelAgentId,
};
use crate::app::FoloApp;
use crate::ui::*;
use folo_bench::score::CALIBRATION_FLOOR;

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

/// An axis's compact label for narrow leaderboard widths so all five axes and actions stay visible.
pub fn axis_compact(a: BenchAxis) -> &'static str {
    match a {
        BenchAxis::Accuracy => "Acc.",
        BenchAxis::Robustness => "Rob.",
        BenchAxis::SelfLabeling => "Self",
        BenchAxis::Calibration => "Cal.",
        BenchAxis::Localization => "Loc.",
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
        LabelAgentId::Antigravity => "Antigravity",
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

/// Where a plan nobody designed comes from, as Studio's Plan tab says it (`PlanTab.tsx`): the
/// default one (every condition of the catalog at severities 1, 3 and 5, weight 1), none at all, or
/// one written by hand (`POST`/`PUT` with a `plan`). `default_len` is the catalog's default plan's
/// length, `None` while the catalog loads or after it failed: a plan of the default's shape is then
/// called neither (Studio waits for the catalog too), as its number of conditions alone can't tell.
pub fn plan_origin(plan: &BenchPlan, default_len: Option<usize>) -> Option<&'static str> {
    let default_shape = plan.conditions.iter().all(|c| c.weight == 1.0 && c.severities == [1, 3, 5]);
    if plan.conditions.is_empty() {
        Some("No conditions: entries run the unchanged images only, and the score has no robustness axis.")
    } else if !default_shape {
        Some("Written by hand.")
    } else {
        match default_len {
            Some(n) if n == plan.conditions.len() => {
                Some("The default plan: every condition at severities 1, 3 and 5, each counting the same.")
            }
            Some(_) => Some("Written by hand."),
            None => None,
        }
    }
}

const SAME_WAY: &str = "Each condition changes every image the same way for every contestant.";

/// The Plan tab's first line: who designed the plan, else where it comes from ([`plan_origin`]).
pub fn plan_intro(designed_by: Option<&str>, plan: &BenchPlan, default_len: Option<usize>) -> String {
    match (designed_by, plan_origin(plan, default_len)) {
        (Some(who), _) => format!("{who}. {SAME_WAY}"),
        (None, Some(origin)) if plan.conditions.is_empty() => origin.to_string(),
        (None, Some(origin)) => format!("{origin} {SAME_WAY}"),
        (None, None) => SAME_WAY.to_string(),
    }
}

/// An entry's best-F1 confidence, `None` when it has none: its F1 is 0 at every confidence (`best_conf` is then 1,
/// or 0 when no class has a true box; docs/BENCH.md, Accuracy).
pub fn best_f1_conf(a: &BenchAccuracy) -> Option<f32> {
    (a.f1 > 0.).then_some(a.best_conf)
}

/// The label of an entry's precision, recall and F1: where they were read.
pub fn prf_label(a: &BenchAccuracy) -> String {
    match best_f1_conf(a) {
        Some(c) => format!("P · R · F1 at {}", fmt_conf(c)),
        None => "P · R · F1 (F1 is 0 at every confidence)".into(),
    }
}

/// Why an entry is partial, one reason each, as Studio's `coverageFacts` gives them to `partialWhy` (without the
/// facts that don't make a score partial: the classes a model knows, an earlier version of the benchmark): "48 of
/// 360 images", "unchanged only" or "4 of 7 conditions", "no confidences", "calibration not measured (…)", "no
/// condition answered whole"; "some conditions left out" for a partial score with none of these (an external entry
/// that left conditions out does not say which).
pub fn partial_reasons(e: &BenchEntry, b: &Bench) -> Vec<String> {
    let axis = |a: BenchAxis| e.score.as_ref().and_then(|s| s.axes.iter().find(|x| x.axis == a));
    let unmeasured = |a: BenchAxis| axis(a).is_some_and(|x| x.value.is_none());
    let mut out = Vec::new();
    if e.images < b.images {
        out.push(format!("{} of {} images", int(e.images), int(b.images)));
    }
    let planned = b.plan.conditions.len();
    // An open external entry has not said yet which conditions it answers.
    let receiving =
        e.status == BenchEntryStatus::Running && matches!(e.contestant, BenchContestant::External);
    if planned > 0 && !e.conditions && !receiving {
        out.push("unchanged only".into());
    } else if planned > 0 && e.status == BenchEntryStatus::Done && (e.conditions_run as usize) < planned {
        out.push(format!("{} of {planned} conditions", e.conditions_run));
    }
    if !e.scored {
        out.push(calibration_none(e));
    }
    if e.scored && e.score.is_some() && unmeasured(BenchAxis::Calibration) {
        out.push(format!(
            "calibration not measured (no detection on the unchanged images reached confidence {CALIBRATION_FLOOR})"
        ));
    }
    if planned > 0 && e.conditions && e.score.is_some() && unmeasured(BenchAxis::Robustness) {
        out.push("no condition answered whole".into());
    }
    if out.is_empty() && e.score.as_ref().is_some_and(|s| s.partial) {
        out.push("some conditions left out".into());
    }
    out
}

/// Why an entry is partial, in a sentence (the "partial" badge's tooltip), in Studio's words (`partialWhy`).
pub fn partial_why(e: &BenchEntry, b: &Bench) -> String {
    format!("Partial: {}. It compares only loosely with complete entries.", partial_reasons(e, b).join(", "))
}

/// Why an entry has no calibration, in a few words (the Calibration tile and panel header): it gave no confidences
/// (an agent), was added as a model that gives none (an external entry: confidences in its file were not used), or
/// none of its detections reached the calibration floor, 0.1 (docs/BENCH.md, Calibration). Short enough for the
/// tile at 1280 px; [`calibration_why_not`] says it in full.
pub fn calibration_none(e: &BenchEntry) -> String {
    match (&e.contestant, e.scored) {
        (_, true) => format!("none reached {CALIBRATION_FLOOR}"),
        (BenchContestant::Agent { .. }, false) => "no confidences".into(),
        (_, false) => "confidences not used".into(),
    }
}

/// The same in a sentence, in Studio's words (`axisWhyNot`).
/// Calibration is measured on the unchanged images only, so a confident detection on a condition does not count.
pub fn calibration_why_not(e: &BenchEntry) -> String {
    match (&e.contestant, e.scored) {
        (_, true) => format!("No detection on the unchanged images reached confidence {CALIBRATION_FLOOR}."),
        (BenchContestant::Agent { .. }, false) => {
            "Agents give no confidences: an agent answers with the boxes it is sure of, so there is nothing to calibrate."
                .into()
        }
        // Added with "The model gives confidences" off: its file may have had scores, which were not used.
        (_, false) => "It was added as giving no confidences, so any scores in its predictions were not used.".into(),
    }
}

/// What the images panel shows of a scored entry's detections, from the confidence its images were taken at (the
/// page's `conf`, `best_conf` from a server before it): its best-F1 confidence as before; 0, its detections at every
/// confidence, said as the cutoff and not a total so it is true on a partial page (or that it made none, once every
/// image is loaded and none has one); above 0 with no best F1, an older FOLO's grading.
pub fn images_shown(page: &BenchEntryImagePage, a: &BenchAccuracy) -> String {
    let at = page.conf.unwrap_or(a.best_conf);
    let best = best_f1_conf(a);
    let all = page.images.len() as u64 >= page.total;
    if best == Some(at) {
        format!(" at confidence {} and up", fmt_conf(at))
    } else if at <= 0. {
        if all && page.images.iter().all(|im| im.detections.is_empty()) {
            ": it made none".into()
        } else {
            " at every confidence (there is no best-F1 confidence)".into()
        }
    } else if best.is_some() {
        format!(" at confidence {} and up", fmt_conf(at))
    } else {
        format!(
            " at confidence {} and up (an older FOLO graded it: running the benchmark again shows every one)",
            fmt_conf(at)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{BBox, BenchDetection, BenchEntryImage, BenchImage};

    fn accuracy(f1: f32, best_conf: f32) -> BenchAccuracy {
        BenchAccuracy {
            map50_95: 0.,
            ci95: [0., 0.],
            map50: 0.,
            map75: 0.,
            precision: if f1 > 0. { 0.8 } else { 0. },
            recall: if f1 > 0. { 0.5 } else { 0. },
            f1,
            best_conf,
            map_small: Some(0.),
            map_medium: None,
            map_large: None,
            per_class: Vec::new(),
        }
    }

    /// A page of `images` with one true box each and `dets` detections on each, taken at `conf`.
    fn page(conf: Option<f32>, images: usize, total: u64, dets: usize) -> BenchEntryImagePage {
        let b = BBox { class: 0, x1: 10., y1: 10., x2: 20., y2: 20. };
        let d = |score: f32| BenchDetection { class: 0, score, x1: 10., y1: 10., x2: 20., y2: 20. };
        BenchEntryImagePage {
            total,
            conf,
            images: (0..images)
                .map(|i| BenchEntryImage {
                    image: BenchImage {
                        id: i.to_string(),
                        file: format!("{i}.jpg"),
                        width: 64,
                        height: 64,
                        boxes: vec![b],
                    },
                    detections: (0..dets).map(|k| d(0.9 - k as f32 * 0.001)).collect(),
                    missed: u32::from(dets == 0),
                    wrong: dets.saturating_sub(1) as u32,
                })
                .collect(),
        }
    }

    #[test]
    fn an_entry_with_no_best_f1_says_so_and_what_its_images_hold() {
        // F1 0 at every confidence: `best_conf` is 1 (or 0 with no class evaluated), neither a best F1.
        let none = accuracy(0., 1.);
        assert_eq!(best_f1_conf(&none), None);
        assert_eq!(best_f1_conf(&accuracy(0., 0.)), None);
        assert_eq!(prf_label(&none), "P · R · F1 (F1 is 0 at every confidence)");
        // 100 wrong above one right on an image (the scorer counts the top 100, so F1 is 0): graded at 0, all 101.
        let every = " at every confidence (there is no best-F1 confidence)";
        let cap = page(Some(0.), 1, 1, 101);
        assert_eq!(images_shown(&cap, &none), every);
        // No class evaluated: graded at 0 too, and `best_conf` 0 is not named.
        assert_eq!(images_shown(&cap, &accuracy(0., 0.)), every);
        // It made no detections: said once every image is in. Before, the cutoff, not a total: 24 of 27 loaded, none
        // with a detection, may still have one on the last page.
        assert_eq!(images_shown(&page(Some(0.), 2, 2, 0), &none), ": it made none");
        assert_eq!(images_shown(&page(Some(0.), 24, 27, 0), &none), every);
        // Stored by an older FOLO (no `conf`): taken at `best_conf`, 1, so not every detection.
        assert_eq!(
            images_shown(&page(None, 2, 2, 0), &none),
            " at confidence 1.00 and up (an older FOLO graded it: running the benchmark again shows every one)"
        );
        // A normal entry, as before, with or without the page's `conf`.
        let some = accuracy(0.62, 0.31);
        assert_eq!(best_f1_conf(&some), Some(0.31));
        assert_eq!(prf_label(&some), "P · R · F1 at 0.31");
        assert_eq!(images_shown(&page(Some(0.31), 1, 1, 2), &some), " at confidence 0.31 and up");
        assert_eq!(images_shown(&page(None, 1, 1, 2), &some), " at confidence 0.31 and up");
    }

    #[test]
    fn calibration_says_why_it_was_not_measured() {
        let (_, entries) = demo_board();
        let mut e = entries[0].clone();
        // A model whose scores all stay under the floor gave confidences: it is told none reached 0.1.
        e.scored = true;
        assert_eq!(calibration_none(&e), "none reached 0.1");
        assert_eq!(calibration_why_not(&e), "No detection on the unchanged images reached confidence 0.1.");
        // A contestant that gives none keeps the explanation about missing confidences.
        e.scored = false;
        e.contestant = BenchContestant::External;
        assert_eq!(calibration_none(&e), "confidences not used");
        assert_eq!(
            calibration_why_not(&e),
            "It was added as giving no confidences, so any scores in its predictions were not used."
        );
        let effort = crate::api::LabelEffort::Medium;
        e.contestant = BenchContestant::Agent { agent: LabelAgentId::Claude, model: None, effort };
        assert!(calibration_why_not(&e).starts_with("Agents give no confidences"));
        assert_eq!(calibration_none(&e), "no confidences");
        assert_eq!(agent_name(LabelAgentId::Antigravity), "Antigravity");
        e.contestant = BenchContestant::Agent {
            agent: LabelAgentId::Antigravity,
            model: Some("gemini-4-neon".into()),
            effort: crate::api::LabelEffort::Low,
        };
        assert_eq!(contestant_line(&e), "Antigravity · gemini-4-neon · low effort");
    }

    /// #382: the "partial" badge said nothing of why; Studio's `partialWhy` lists the reasons.
    #[test]
    fn a_partial_entry_says_why() {
        let (b, entries) = demo_board();
        let mut e = entries[0].clone();
        e.score.as_mut().unwrap().partial = true;
        e.images = b.images / 2;
        e.conditions = false;
        e.scored = false;
        e.contestant = BenchContestant::External;
        let half = format!("{} of {} images", int(e.images), int(b.images));
        assert_eq!(
            partial_why(&e, &b),
            format!(
                "Partial: {half}, unchanged only, confidences not used. It compares only loosely with complete entries."
            )
        );
        // A scored model that ran every condition: calibration not measured, and robustness when none was answered.
        let mut e = entries[0].clone();
        e.images = b.images;
        e.conditions = true;
        e.conditions_run = b.plan.conditions.len() as u32;
        e.scored = true;
        for x in &mut e.score.as_mut().unwrap().axes {
            if matches!(x.axis, BenchAxis::Calibration | BenchAxis::Robustness) {
                x.value = None;
            }
        }
        assert_eq!(
            partial_reasons(&e, &b),
            [
                "calibration not measured (no detection on the unchanged images reached confidence 0.1)",
                "no condition answered whole"
            ]
        );
        // Partial with none of those: an external entry that left conditions out.
        e.score.as_mut().unwrap().axes.iter_mut().for_each(|x| x.value = Some(0.5));
        e.score.as_mut().unwrap().partial = true;
        assert_eq!(partial_reasons(&e, &b), ["some conditions left out"]);
    }

    #[test]
    fn only_the_default_plan_is_called_the_default_one() {
        use crate::api::BenchPlanCondition;
        let c = |id: &str, sev: &[u8], weight: f32| BenchPlanCondition {
            condition: id.into(),
            severities: sev.to_vec(),
            weight,
            why: None,
        };
        let plan = |conditions| BenchPlan { conditions, ..Default::default() };
        // The line the Plan tab shows (`plan.rs` builds it with `plan_intro`).
        let says = |p: &BenchPlan, catalog: Option<usize>| plan_intro(None, p, catalog);
        let full = plan(vec![c("haze", &[1, 3, 5], 1.0), c("jpeg", &[1, 3, 5], 1.0)]);
        assert!(says(&full, Some(2)).starts_with("The default plan"));
        // A benchmark made with its own plan (lane d4-bench-settings: haze 1, 5 and sensor noise 3)
        // was called "The default plan".
        let own = plan(vec![c("haze", &[1, 5], 1.0), c("sensor_noise", &[3], 1.0)]);
        assert!(says(&own, Some(12)).starts_with("Written by hand."));
        assert!(says(&own, None).starts_with("Written by hand."), "its shape alone tells");
        assert!(says(&plan(vec![c("haze", &[1, 3, 5], 2.0)]), None).starts_with("Written by hand."));
        assert!(says(&full, Some(12)).starts_with("Written by hand."), "a part of the catalog");
        // Haze alone at 1, 3 and 5 has the default's shape: while the catalog loads, or after it
        // failed, it is called neither (review round 1).
        let haze = plan(vec![c("haze", &[1, 3, 5], 1.0)]);
        assert_eq!(says(&haze, None), SAME_WAY);
        assert_eq!(says(&full, None), SAME_WAY);
        assert!(says(&haze, Some(12)).starts_with("Written by hand."));
        assert!(says(&plan(vec![]), Some(12)).starts_with("No conditions"));
        assert!(says(&plan(vec![]), None).starts_with("No conditions"));
        assert!(plan_intro(Some("Designed by Claude"), &own, None).starts_with("Designed by Claude. Each"));
        // The demo's default plan reads as one.
        use crate::api::Backend;
        let api = crate::api::mock::MockBackend::new(4200, true);
        let cat = api.bench_catalog().unwrap();
        assert!(says(&cat.default_plan, Some(cat.default_plan.conditions.len())).starts_with("The default"));
    }

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
