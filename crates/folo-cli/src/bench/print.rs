//! The human output of `folo bench` (stdout, or stderr with `--json`).

use folo_api::{BenchEntry, BenchEntryStatus, BenchReadiness, BenchSource};
use folo_mcp::types::{BenchCatalogResult, BenchDetail, BenchEntryState, BenchPlanRow, BenchSummary};

use crate::output::Out;

fn status_name(s: BenchEntryStatus) -> &'static str {
    match s {
        BenchEntryStatus::Queued => "queued",
        BenchEntryStatus::Running => "running",
        BenchEntryStatus::Stopping => "stopping",
        BenchEntryStatus::Done => "done",
        BenchEntryStatus::Failed => "failed",
        BenchEntryStatus::Stopped => "stopped",
    }
}

fn plan(rows: &[BenchPlanRow]) -> String {
    if rows.is_empty() {
        return "the unchanged images only".into();
    }
    let rows: Vec<String> = rows
        .iter()
        .map(|r| {
            let sev: Vec<String> = r.severities.iter().map(u8::to_string).collect();
            let weight = if r.weight == 1.0 { String::new() } else { format!(" ×{}", r.weight) };
            format!("{} {}{weight}", r.condition, sev.join(","))
        })
        .collect();
    rows.join("; ")
}

pub(super) fn catalog(c: &BenchCatalogResult, out: Out) {
    out.human(format!("conditions (severities 1 to {}):", c.severities));
    let width = c.conditions.iter().map(|x| x.id.len()).max().unwrap_or(0);
    for x in &c.conditions {
        out.human(format!("  {:<width$} {:<9} {}", x.id, x.group, x.description));
    }
    out.human(format!("default plan: {}", plan(&c.default_plan)));
    let axes: Vec<String> =
        c.axes.iter().map(|a| format!("{} {:.0}%", a.axis.as_str(), a.weight * 100.0)).collect();
    out.human(format!("score: {}", axes.join(", ")));
}

pub(super) fn list(list: &[BenchSummary], out: Out) {
    if list.is_empty() {
        out.human("no benchmarks; `folo bench create` makes one");
        return;
    }
    out.human(format!("{:<16} {:>6} {:>7} {:>7}  {:<24} name", "id", "images", "classes", "entries", "best"));
    for b in list {
        let best = b.best.as_ref().map_or("-".to_string(), |x| {
            format!("{:.1}{} {}", x.score, if x.partial { "*" } else { "" }, x.name)
        });
        out.human(format!(
            "{:<16} {:>6} {:>7} {:>7}  {:<24} {}{}",
            b.id,
            b.images,
            b.num_classes,
            b.entries,
            best,
            b.name,
            if b.active { " (running)" } else { "" }
        ));
    }
}

pub(super) fn detail(b: &BenchDetail, out: Out) {
    let source = match &b.source {
        BenchSource::Dataset { split, .. } => {
            format!("the {} split of '{}'", split.map_or("val", |s| s.as_str()), b.source_name)
        }
        BenchSource::LabelJob { .. } => format!("labeling job '{}'", b.source_name),
    };
    out.human(format!("{} '{}': {} images, {} boxes from {source}", b.id, b.name, b.images, b.boxes));
    out.human(format!("classes: {}", b.classes.join(", ")));
    out.human(format!("plan: {}", plan(&b.plan)));
    if b.leaderboard.is_empty() {
        out.human(format!("no entries yet; `folo bench run {} --model ID` adds one", b.id));
        return;
    }
    out.human(format!(
        "{:<4} {:<14} {:<9} {:>6} {:>8}  {:<15} name",
        "#", "entry", "status", "score", "mAP", "95% interval"
    ));
    for (i, e) in b.leaderboard.iter().enumerate() {
        let score = e.score.map_or("-".into(), |s| format!("{s:.1}{}", if e.partial { "*" } else { "" }));
        let map = e.map50_95.map_or("-".into(), |m| format!("{m:.3}"));
        let ci = e.ci95.map_or("-".into(), |[lo, hi]| format!("{lo:.3}–{hi:.3}"));
        let stale = if e.stale { " (another version of the benchmark)" } else { "" };
        out.human(format!(
            "{:<4} {:<14} {:<9} {:>6} {:>8}  {:<15} {} ({}){stale}",
            i + 1,
            e.id,
            status_name(e.status),
            score,
            map,
            ci,
            e.name,
            e.kind
        ));
    }
    if b.leaderboard.iter().any(|e| e.partial) {
        out.human("* partial: an axis was not measured or the entry ran part of the images");
    }
}

/// One line of a running entry's progress.
pub(super) fn progress(e: &BenchEntry) -> String {
    let p = &e.progress;
    let pass = match (&p.condition, p.severity) {
        (Some(c), Some(s)) => format!(" ({c} {s})"),
        _ => String::new(),
    };
    format!("{} {}/{} images{pass}", status_name(e.status), p.done, p.total)
}

pub(super) fn state(s: &BenchEntryState, out: Out) {
    out.human(format!(
        "entry {} '{}' of {}: {} {}/{}",
        s.entry_id,
        s.name,
        s.benchmark_id,
        status_name(s.status),
        s.done,
        s.total
    ));
    if let Some(e) = &s.error {
        out.human(format!("error: {e}"));
    }
    if !s.missing_classes.is_empty() {
        out.human(format!("classes the model does not know (AP 0): {}", s.missing_classes.join(", ")));
    }
    let Some(r) = &s.result else {
        if s.status == BenchEntryStatus::Done
            && let Some(h) = &s.hint
        {
            out.human(format!("hint: {h}"));
        }
        return;
    };
    out.human(format!("score {:.1}{}", r.score, if r.partial { " (partial)" } else { "" }));
    for a in &r.axes {
        let value = a.value.map_or("not measured".into(), |v| format!("{v:.3}"));
        out.human(format!("  {:<14} {value:<12} weight {:.2}", a.axis.as_str(), a.weight));
    }
    let a = &r.accuracy;
    out.human(format!(
        "mAP50-95 {:.3} [{:.3}, {:.3}]  mAP50 {:.3}  mAP75 {:.3}  P {:.3}  R {:.3} at conf {:.2}",
        a.map50_95, a.ci95[0], a.ci95[1], a.map50, a.map75, a.precision, a.recall, a.best_conf
    ));
    let width = r.conditions.iter().map(|c| c.condition.len()).max().unwrap_or(0);
    for c in &r.conditions {
        let points: Vec<String> =
            c.points.iter().map(|p| format!("{}: {:.3} ({:.0}%)", p[0], p[1], p[2] * 100.0)).collect();
        out.human(format!(
            "  {:<width$} kept {:>4.0}%  {}",
            c.condition,
            c.retention * 100.0,
            points.join("  ")
        ));
    }
    let measured = |v: Option<f32>| v.map_or("not measured".into(), |v| format!("{v:.3}"));
    out.human(format!("ECE {}  mean IoU {}", measured(r.ece), measured(r.mean_iou)));
    let sl = &r.self_labeling;
    let threshold = sl.threshold.map_or("none".into(), |t| format!("{t:.2}"));
    let readiness = match sl.readiness {
        BenchReadiness::Ready => "ready to train on",
        BenchReadiness::Partial => "partly ready",
        BenchReadiness::NotReady => "not ready",
    };
    out.human(format!(
        "self-labeling: threshold {threshold}, precision {:.3}, recall {:.3} ({readiness})",
        sl.precision, sl.recall
    ));
    if let Some(ms) = r.ms_per_image {
        out.human(format!("{ms:.1} ms per image"));
    }
}
