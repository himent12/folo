//! The human output of `folo label` (stdout, or stderr with `--json`).

use folo_api::{LabelJobStatus, LabelOutcome};
use folo_mcp::types::{
    LabelAgentsResult, LabelImageSummary, LabelImagesAddedResult, LabelJobSummary, LabelRunStarted,
    LabelStatusResult, Page, ReviewLabelImagesResult,
};

use crate::output::Out;

pub(super) fn status_name(s: LabelJobStatus) -> &'static str {
    match s {
        LabelJobStatus::Idle => "idle",
        LabelJobStatus::Running => "running",
        LabelJobStatus::Stopping => "stopping",
    }
}

pub(super) fn agents(r: &LabelAgentsResult, out: Out) {
    for a in &r.agents {
        if a.available {
            out.human(format!(
                "{:<7} {} (default {} at {} effort; models: {})",
                a.id.as_str(),
                a.version.as_deref().unwrap_or("installed"),
                a.default_model.as_deref().unwrap_or("its own"),
                a.default_effort.as_str(),
                a.models.join(", ")
            ));
        } else {
            out.human(format!(
                "{:<7} not installed{}",
                a.id.as_str(),
                a.hint.as_deref().map(|h| format!(": {h}")).unwrap_or_default()
            ));
        }
    }
}

pub(super) fn jobs(jobs: &[LabelJobSummary], out: Out) {
    if jobs.is_empty() {
        out.human("no labeling jobs; `folo label create` makes one");
        return;
    }
    out.human(format!("{:<10} {:<8} {:>7} {:>7} {:>6}  name", "id", "status", "images", "labeled", "failed"));
    for j in jobs {
        out.human(format!(
            "{:<10} {:<8} {:>7} {:>7} {:>6}  {}",
            j.id,
            status_name(j.status),
            j.counts.images,
            j.counts.labeled,
            j.counts.failed,
            j.name
        ));
    }
}

pub(super) fn added(r: &LabelImagesAddedResult, out: Out) {
    out.human(format!(
        "{}: added {} images ({} skipped); the job has {}",
        r.job_id, r.added, r.skipped, r.images
    ));
    for s in &r.skipped_first {
        out.human(format!("  skipped {}: {}", s.file, s.reason));
    }
}

pub(super) fn started(r: &LabelRunStarted, out: Out) {
    match &r.run {
        Some(run) => out.human(format!(
            "{}: {} ({}, {} effort) labels {} images",
            r.job_id,
            run.agent.display_name(),
            run.model.as_deref().unwrap_or("its default model"),
            run.effort.as_str(),
            run.total
        )),
        None => out.human(format!("{}: {}", r.job_id, status_name(r.status))),
    }
}

pub(super) fn status(r: &LabelStatusResult, out: Out) {
    let c = &r.counts;
    out.human(format!(
        "{} ({}) {}: {} images, {} labeled ({} reviewed), {} pending, {} failed, {} boxes",
        r.id,
        r.name,
        status_name(r.status),
        c.images,
        c.labeled,
        r.reviewed,
        c.pending,
        c.failed,
        c.boxes
    ));
    out.human(format!("classes: {}", r.classes.join(", ")));
    if let Some(run) = &r.run {
        let outcome = match run.outcome {
            None => "running".to_string(),
            Some(LabelOutcome::Completed) => "completed".into(),
            Some(LabelOutcome::Stopped) => "stopped".into(),
            Some(LabelOutcome::Failed) => {
                format!("failed: {}", run.error.as_deref().unwrap_or("see the log"))
            }
        };
        out.human(format!(
            "last run: {} ({}), {}/{} labeled, {} failed, {outcome}",
            run.agent.display_name(),
            run.model.as_deref().unwrap_or("default model"),
            run.done,
            run.total,
            run.failed
        ));
    }
    let u = &r.usage;
    out.human(format!(
        "tokens: {} in ({} cached), {} out over {} calls{}",
        u.input_tokens,
        u.cached_input_tokens,
        u.output_tokens,
        u.calls,
        u.cost_usd.map(|c| format!(", ${c:.3} reported")).unwrap_or_default()
    ));
    if let Some(d) = &r.dataset_id {
        out.human(format!("dataset: {d}"));
    }
    for l in &r.log {
        out.human(format!("  {l}"));
    }
}

pub(super) fn images(page: &Page<LabelImageSummary>, offset: u64, out: Out) {
    out.human(format!("{:<6} {:<9} {:>5} {:<7} {:<8} file", "id", "status", "boxes", "source", "reviewed"));
    for i in &page.items {
        let source = match i.source {
            Some(folo_api::LabelSource::Agent) => "agent",
            Some(folo_api::LabelSource::Manual) => "manual",
            None => "-",
        };
        out.human(format!(
            "{:<6} {:<9} {:>5} {:<7} {:<8} {}{}",
            i.id,
            i.status.as_str(),
            i.num_boxes,
            source,
            if i.reviewed { "yes" } else { "no" },
            i.file,
            i.error.as_deref().map(|e| format!("  ({e})")).unwrap_or_default()
        ));
        if let Some(boxes) = &i.boxes {
            for (n, b) in boxes.iter().enumerate() {
                out.human(format!("         {} class {} {},{},{},{}", n + 1, b[0], b[1], b[2], b[3], b[4]));
            }
        }
    }
    let shown = page.items.len() as u64;
    let from = if shown == 0 { offset } else { offset + 1 };
    out.human(format!("{from}-{} of {}", offset + shown, page.total));
}

pub(super) fn reviewed(r: &ReviewLabelImagesResult, reviewed: bool, out: Out) {
    out.human(format!(
        "marked {} {}; {} left to review",
        r.updated,
        if reviewed { "reviewed" } else { "not reviewed" },
        r.to_review
    ));
    for f in &r.failed {
        out.human(format!("  {}: {} ({})", f.id, f.message, f.code));
    }
}
