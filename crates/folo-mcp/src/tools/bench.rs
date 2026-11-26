//! The vision-benchmark tools: the catalog, benchmarks (list, one with its leaderboard, create, delete) and their
//! entries (start one of a model or of a file of predictions, follow it to its score, stop it). The operations are
//! [`crate::bench::Benchmarks`]'s (shared with `folo bench`); this file adds the path of a predictions file, the
//! blocking pool, the hints in MCP words, waits and the one-line summaries.

use std::time::{Duration, Instant};

use folo_api::{BenchEntryStatus, BenchSource, CreateBench};
use http::StatusCode;

use super::results::{
    MAX_RESULT_BYTES, MAX_SUMMARY_CHARS, Output, json_len, out, page_of_window, page_window, short,
};
use crate::bench::hints::{self, Front};
use crate::bench::{Benchmarks, Contestant};
use crate::errors::{Error, invalid_arguments};
use crate::server::Inner;
use crate::types::*;

/// Longest `wait` of `bench_entry`, in seconds (well under the minute some clients give a tool call).
pub const MAX_WAIT_SECONDS: u32 = 50;
/// How often a wait reads the entry: in process, and through a server.
const POLL_HERE: Duration = Duration::from_millis(100);
const POLL_SERVER: Duration = Duration::from_millis(500);
/// Leaderboard rows `get_benchmark` returns (fewer when the result would be over its size limit).
const MAX_ROWS: usize = 50;
/// The bytes `get_benchmark` fits its result in (the result limit, less room for the envelope).
const FIT: usize = MAX_RESULT_BYTES - 1024;
/// How long `bench_entry` may take beyond its `wait`: every request to a server ends by then.
const GRACE: Duration = Duration::from_secs(5);
/// Ends the summary of an entry whose text was cut: where the whole text is.
const ENTRY_CUT: &str = " (long text cut; GET /api/bench/benchmarks/{id}/entries/{entry_id} has it whole)";
/// The tools `--confine` refuses: all of them, see [`refuse_confined`].
const BENCH_TOOLS: [&str; 8] = [
    "bench_catalog",
    "list_benchmarks",
    "get_benchmark",
    "create_benchmark",
    "delete_benchmark",
    "start_bench_entry",
    "bench_entry",
    "stop_bench_entry",
];

/// With `--confine`, refuses a benchmark tool. A benchmark's files (its frozen images, its entries, the weights a
/// model entry loads) are read, written and deleted by the store without the confinement's checks, and through a
/// server they are out of its reach, so the one rule that holds in both modes is not to offer them.
pub(super) fn refuse_confined(tool: &str, confine: bool) -> Result<(), Error> {
    if !confine || !BENCH_TOOLS.contains(&tool) {
        return Ok(());
    }
    Err(Error::new(
        StatusCode::NOT_IMPLEMENTED,
        "not_supported",
        format!("{tool} is not available with --confine (benchmark files are not confined to the data dir)"),
    )
    .hint("restart `folo mcp` without --confine to use the benchmark tools"))
}

impl Inner {
    /// Runs benchmark calls on the blocking pool (they read files or make HTTP requests); errors get MCP hints.
    async fn benchmarks<T, F>(&self, f: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&Benchmarks) -> Result<T, Error> + Send + 'static,
    {
        let b = self.benchmarks.clone();
        let r = tokio::task::spawn_blocking(move || f(&b)).await.map_err(super::join)?;
        r.map_err(|e| hints::decorate(e, Front::Mcp))
    }

    pub(super) async fn bench_catalog(&self, _: NoArgs) -> Result<Output, Error> {
        let r = self.benchmarks(|b| b.catalog_result()).await?;
        let summary = format!(
            "{} conditions at severities 1 to {}; the default plan runs {}",
            r.conditions.len(),
            r.severities,
            r.default_plan.len()
        );
        out(&r, summary)
    }

    pub(super) async fn list_benchmarks(&self, a: PageArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        let all = self.benchmarks(|b| b.summaries()).await?;
        let page = page_of_window(all, offset, limit);
        let names: Vec<String> =
            page.items.iter().take(5).map(|b| format!("{} {}", b.id, short(&b.name, 40))).collect();
        let summary = match page.total {
            0 => "no benchmarks; create_benchmark makes one".to_string(),
            n => format!("{n} benchmark(s): {}", names.join(", ")),
        };
        out(&page, summary)
    }

    pub(super) async fn get_benchmark(&self, a: BenchmarkIdArgs) -> Result<Output, Error> {
        let id = a.benchmark_id;
        let mut r = self.benchmarks(move |b| b.detail(&id)).await?;
        let entries = r.leaderboard.len();
        r.leaderboard.truncate(MAX_ROWS);
        let cut = fit_detail(&mut r);
        let mut summary =
            format!("{} '{}': {} images, {} entries", r.id, short(&r.name, 40), r.images, entries);
        if entries > r.leaderboard.len() {
            summary.push_str(&format!(
                " (the first {} listed; GET /api/bench/benchmarks/{{id}}/entries has all)",
                r.leaderboard.len()
            ));
        }
        if cut {
            summary
                .push_str(" (long text cut; GET /api/bench/benchmarks/{id} and its /entries have it whole)");
        }
        if let Some(e) = r.leaderboard.first().filter(|e| e.score.is_some()) {
            summary.push_str(&format!("; first: {} {:.1}", short(&e.name, 40), e.score.unwrap_or_default()));
        }
        out(&r, summary)
    }

    pub(super) async fn create_benchmark(&self, a: CreateBenchmarkArgs) -> Result<Output, Error> {
        let source = match (a.dataset_id, a.label_job_id) {
            (Some(dataset_id), None) => BenchSource::Dataset { dataset_id, split: a.split },
            (None, Some(job_id)) if a.split.is_none() => BenchSource::LabelJob { job_id },
            (None, Some(_)) => return Err(invalid_arguments("split is for a dataset, not for label_job_id")),
            _ => return Err(invalid_arguments("give one source: dataset_id (with split) or label_job_id")),
        };
        let req = CreateBench {
            name: a.name,
            description: a.description,
            source,
            max_images: a.max_images,
            seed: a.seed,
            plan: a.plan.map(Into::into),
        };
        let r = self.benchmarks(move |b| b.create_bench(req)).await?;
        let summary = format!(
            "{}: {} images, {} boxes, {} conditions; start_bench_entry runs a model on it",
            r.id,
            r.images,
            r.boxes,
            r.plan.len()
        );
        // The summary of list_benchmarks (get_benchmark has the rest): a small schema for every session to carry.
        let made = BenchSummary {
            id: r.id,
            name: r.name,
            source_name: r.source_name,
            images: r.images,
            num_classes: r.classes.len(),
            conditions: r.plan.len(),
            entries: 0,
            best: None,
            active: false,
        };
        out(&made, summary)
    }

    pub(super) async fn delete_benchmark(&self, a: BenchmarkIdArgs) -> Result<Output, Error> {
        let id = a.benchmark_id;
        let gone = id.clone();
        self.benchmarks(move |b| b.delete(&id)).await?;
        out(&BenchDeleted { benchmark_id: gone.clone(), deleted: true }, format!("deleted {gone}"))
    }

    pub(super) async fn start_bench_entry(&self, a: StartBenchEntryArgs) -> Result<Output, Error> {
        let who = match (a.model_id, a.predictions_file) {
            (Some(model_id), None) => {
                if a.scored.is_some() {
                    return Err(invalid_arguments(
                        "scored is for predictions_file: a model gives confidences",
                    ));
                }
                Contestant::Model { model_id, max_images: a.max_images, conditions: a.conditions }
            }
            (None, Some(file)) => {
                if a.max_images.is_some() || a.conditions.is_some() {
                    return Err(invalid_arguments(
                        "max_images and conditions are for model_id: predictions say what they ran",
                    ));
                }
                Contestant::Predictions { path: self.resolve(file).await?, scored: a.scored.unwrap_or(true) }
            }
            _ => return Err(invalid_arguments("give one contestant: model_id or predictions_file")),
        };
        let (id, name) = (a.benchmark_id, a.name);
        let r = self.benchmarks(move |b| b.start(&id, who, name)).await?;
        let summary = entry_summary(&r);
        out(&BenchEntryBrief::from(r), summary)
    }

    pub(super) async fn bench_entry(&self, a: BenchEntryArgs) -> Result<Output, Error> {
        if a.wait > MAX_WAIT_SECONDS {
            return Err(invalid_arguments(format!("wait must be in 0..={MAX_WAIT_SECONDS}, got {}", a.wait)));
        }
        let poll = if self.proxy.is_some() { POLL_SERVER } else { POLL_HERE };
        let wait = Duration::from_secs(a.wait.into());
        let (id, entry) = (a.benchmark_id, a.entry_id);
        let mut r = self.benchmarks(move |b| wait_state(b, &id, &entry, poll, wait, GRACE)).await?;
        let cut = fit_state(&mut r);
        let summary = fitted_summary(&r, cut);
        out(&r, summary)
    }

    pub(super) async fn stop_bench_entry(&self, a: BenchEntryIdArgs) -> Result<Output, Error> {
        let (id, entry) = (a.benchmark_id, a.entry_id);
        let r = self.benchmarks(move |b| b.stop(&id, &entry)).await?;
        let summary = entry_summary(&r);
        out(&BenchEntryBrief::from(r), summary)
    }
}

/// An entry's state once it has ended or `wait` has passed. Through a server, every request ends by `grace` after
/// the wait, so a server that stops answering cannot hold the call (or this thread) for the client's long timeout.
pub(super) fn wait_state(
    b: &Benchmarks,
    id: &str,
    entry: &str,
    poll: Duration,
    wait: Duration,
    grace: Duration,
) -> Result<BenchEntryState, Error> {
    let deadline = Instant::now() + wait;
    let b = b.until(deadline + grace);
    let keep_going = |_: &folo_api::BenchEntry| Instant::now() + poll <= deadline;
    b.follow(id, entry, poll, keep_going).and_then(|e| b.state_of(e)).map_err(|mut e| {
        if e.body.code == "server_unreachable" {
            e.body.hint = Some(
                "the server is slow or gone; the entry stays on it: bench_entry follows it again".into(),
            );
        }
        e
    })
}

/// Cuts `s` to `max` characters; true when that changed it.
fn cut_to(s: &mut String, max: usize) -> bool {
    let t = short(s, max);
    let cut = t != *s;
    *s = t;
    cut
}

/// Fits a benchmark within the result limit whatever its names, keeping the ranking longest: a shorter description,
/// then shorter names (its own, its classes', its entries'), then fewer rows from the end (the API has them all).
/// True when text was cut.
pub(super) fn fit_detail(r: &mut BenchDetail) -> bool {
    if json_len(r) <= FIT {
        return false;
    }
    let mut cut = cut_to(&mut r.description, 200);
    if json_len(r) > FIT {
        cut |= cut_to(&mut r.name, 200);
        cut |= cut_to(&mut r.source_name, 200);
        for c in &mut r.classes {
            cut |= cut_to(c, 24);
        }
    }
    if json_len(r) > FIT {
        for e in &mut r.leaderboard {
            cut |= cut_to(&mut e.name, 40);
        }
    }
    while json_len(r) > FIT && !r.leaderboard.is_empty() {
        r.leaderboard.pop();
    }
    cut
}

/// Fits an entry's state within the result limit: at most 80 classes are missing, so with short names, a short error
/// and a short hint it always fits (the API has them whole). True when text was cut.
pub(super) fn fit_state(r: &mut BenchEntryState) -> bool {
    if json_len(r) <= FIT {
        return false;
    }
    let mut cut = cut_to(&mut r.name, 200);
    for c in &mut r.missing_classes {
        cut |= cut_to(c, 24);
    }
    for text in [&mut r.error, &mut r.hint].into_iter().flatten() {
        cut |= cut_to(text, 500);
    }
    cut
}

/// An entry's summary, keeping room for where the whole text is when `cut`.
pub(super) fn fitted_summary(r: &BenchEntryState, cut: bool) -> String {
    let summary = entry_summary(r);
    if !cut {
        return summary;
    }
    // `short` may add one character, the ellipsis.
    format!("{}{ENTRY_CUT}", short(&summary, MAX_SUMMARY_CHARS - ENTRY_CUT.len() - 1))
}

/// One line of an entry's state: its progress, or its score.
fn entry_summary(r: &BenchEntryState) -> String {
    let head = format!("{} '{}'", r.entry_id, short(&r.name, 40));
    match (&r.result, r.status) {
        (Some(s), _) => format!(
            "{head} done: score {:.1}{}, mAP50-95 {:.3} [{:.3}, {:.3}]",
            s.score,
            if s.partial { " (partial)" } else { "" },
            s.accuracy.map50_95,
            s.accuracy.ci95[0],
            s.accuracy.ci95[1]
        ),
        (None, BenchEntryStatus::Failed) => {
            format!("{head} failed: {}", short(r.error.as_deref().unwrap_or("no reason given"), 200))
        }
        (None, BenchEntryStatus::Done) => match &r.hint {
            Some(h) => format!("{head} done: {}", short(h, 200)),
            None => format!("{head} done"),
        },
        (None, status) if status.is_active() => {
            format!(
                "{head} {}: {}/{} images; bench_entry with wait follows it",
                status_name(status),
                r.done,
                r.total
            )
        }
        (None, status) => format!("{head} {}", status_name(status)),
    }
}

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
