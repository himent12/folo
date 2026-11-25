//! A run's history: event subscriptions, metrics and logs, and bringing `run.json` up to the files after a crash.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use folo_api::{BestEpoch, Epoch, LogLevel, LogLine, RunEvent, RunLogs, RunMetrics, Step};
use serde::Deserialize;

use crate::error::Result;
use crate::store::{
    LogsQuery, MetricsQuery, RunSseStream, RunSseSubscription, RunStatusResync, SseFrame, Subscription,
};
use crate::util::{fnv1a, parse_rfc3339, rfc3339, unix_now};

use super::entry::{model_metrics, status_event};
use super::{RunFile, Runs};

/// Bytes hashed at the start and end of the cached `metrics.jsonl` prefix to detect truncation or in-place edits.
const HASH_WINDOW: u64 = 512;

/// Cached incremental parse state of one run's `metrics.jsonl`.
#[derive(Default)]
pub(super) struct MetricsCache {
    /// Byte offset just past the last `\n`-terminated line folded into `parsed`.
    offset: u64,
    /// Total file length observed when `offset` was committed.
    file_len: u64,
    /// Last observed file modification time (helps catch same-size in-place rewrites).
    modified: Option<SystemTime>,
    /// FNV-1a hash of `[0..min(offset, HASH_WINDOW)]`.
    prefix_hash: u64,
    /// FNV-1a hash of `[offset.saturating_sub(HASH_WINDOW)..offset]`.
    tail_hash: u64,
    /// Metrics accumulated from all complete `\n`-terminated lines in `0..offset`.
    parsed: ParsedMetrics,
    /// Counter of incremental vs full parses (for tests verifying tail reuse).
    #[cfg(test)]
    pub(super) full_parses: u64,
}

#[derive(Clone, Default)]
struct ParsedMetrics {
    steps: Vec<Step>,
    epochs: Vec<Epoch>,
    last_status: Option<String>,
    resumed_unflushed: bool,
}

#[derive(Deserialize)]
struct EventTag<'a> {
    #[serde(borrow)]
    event: Cow<'a, str>,
}

#[derive(Deserialize)]
struct StatusTag<'a> {
    #[serde(borrow)]
    status: Cow<'a, str>,
}

/// Zero-allocation string validator for skipped `log.jsonl` lines.
struct StrCheck;

impl<'de> Deserialize<'de> for StrCheck {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = StrCheck;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, _v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrCheck)
            }
            fn visit_string<E: serde::de::Error>(self, _v: String) -> std::result::Result<Self::Value, E> {
                Ok(StrCheck)
            }
        }
        deserializer.deserialize_str(Visitor)
    }
}

/// Structural mirror of [`LogLine`] that validates skipped lines without allocating `String`s.
#[derive(Deserialize)]
struct LogLineCheck {
    #[serde(rename = "t")]
    _t: StrCheck,
    #[serde(rename = "level")]
    _level: LogLevel,
    #[serde(rename = "message")]
    _message: StrCheck,
}

impl ParsedMetrics {
    /// Folds one trimmed JSON line from `metrics.jsonl` directly into typed state (no `serde_json::Value`).
    fn apply_line(&mut self, line: &[u8]) {
        if line.is_empty() {
            return;
        }
        let Ok(tag) = serde_json::from_slice::<EventTag<'_>>(line) else { return };
        match tag.event.as_ref() {
            "step" => {
                if let Ok(s) = serde_json::from_slice::<Step>(line) {
                    while self.steps.last().is_some_and(|l| l.step >= s.step) {
                        self.steps.pop();
                    }
                    while self.epochs.last().is_some_and(|e| e.epoch >= s.epoch) {
                        self.epochs.pop();
                    }
                    self.steps.push(s);
                    self.resumed_unflushed = false;
                }
            }
            "epoch" => {
                if let Ok(e) = serde_json::from_slice::<Epoch>(line) {
                    while self.epochs.last().is_some_and(|x| x.epoch >= e.epoch) {
                        self.epochs.pop();
                    }
                    self.epochs.push(e);
                    self.resumed_unflushed = false;
                }
            }
            "status" => {
                if let Ok(st) = serde_json::from_slice::<StatusTag<'_>>(line) {
                    if st.status == "running" && !self.steps.is_empty() {
                        self.resumed_unflushed = true;
                    }
                    self.last_status = Some(st.status.into_owned());
                }
            }
            "resume" if !self.steps.is_empty() => {
                self.resumed_unflushed = true;
            }
            _ => {}
        }
    }

    /// Projects the parsed history with `epochs_done` / `waiting_resume`, merges live `tail` steps, and applies `q`.
    fn select_with_tail(
        &self,
        epochs_done: u32,
        waiting_resume: bool,
        tail: &[Step],
        q: &MetricsQuery,
    ) -> RunMetrics {
        let hide_unflushed =
            self.resumed_unflushed || (waiting_resume && self.last_status.as_deref() != Some("running"));
        let completed = epochs_done.max(self.epochs.last().map_or(0, |e| e.epoch));

        let mut last = if hide_unflushed {
            self.steps.iter().rev().find(|s| s.epoch <= completed).map_or(0, |s| s.step)
        } else {
            self.steps.last().map_or(0, |s| s.step)
        };

        let start = match q.after_step {
            Some(after) => self.steps.partition_point(|s| s.step <= after),
            None => 0,
        };
        let mut steps: Vec<Step> = if hide_unflushed {
            self.steps[start..].iter().copied().filter(|s| s.epoch <= completed).collect()
        } else {
            self.steps[start..].to_vec()
        };

        for &s in tail {
            if s.step <= last {
                continue;
            }
            if q.after_step.is_some() && s.step != last + 1 {
                break;
            }
            last = s.step;
            if q.after_step.is_none_or(|a| s.step > a) {
                steps.push(s);
            }
        }

        if q.every > 1 {
            let latest = steps.last().map(|s| s.step);
            steps.retain(|s| s.step % q.every == 0 || Some(s.step) == latest);
        }

        let ep_start = match q.after_epoch {
            Some(after) => self.epochs.partition_point(|e| e.epoch <= after),
            None => 0,
        };
        let epochs = self.epochs[ep_start..].to_vec();
        RunMetrics { steps, epochs }
    }
}

impl MetricsCache {
    fn reset(&mut self) {
        self.offset = 0;
        self.file_len = 0;
        self.modified = None;
        self.prefix_hash = 0;
        self.tail_hash = 0;
        self.parsed = ParsedMetrics::default();
    }

    /// Incrementally refreshes from `path` (re-parsing from byte 0 if truncated or rewritten) and returns the
    /// projected metrics for `q` merged with live `tail`.
    fn read_select(
        &mut self,
        path: &Path,
        epochs_done: u32,
        waiting_resume: bool,
        tail: &[Step],
        q: &MetricsQuery,
    ) -> RunMetrics {
        let Ok(mut file) = File::open(path) else {
            self.reset();
            return self.parsed.select_with_tail(epochs_done, waiting_resume, tail, q);
        };
        let Ok(meta) = file.metadata() else {
            self.reset();
            return self.parsed.select_with_tail(epochs_done, waiting_resume, tail, q);
        };
        let file_len = meta.len();
        let modified = meta.modified().ok();

        let can_reuse = self.offset > 0
            && file_len >= self.offset
            && !(file_len == self.file_len && modified != self.modified)
            && verify_windows(&mut file, self.offset, self.prefix_hash, self.tail_hash);

        if !can_reuse {
            self.reset();
            #[cfg(test)]
            {
                self.full_parses += 1;
            }
        }

        let mut trailing: Option<Vec<u8>> = None;
        let old_offset = self.offset;
        if file_len > self.offset && file.seek(SeekFrom::Start(self.offset)).is_ok() {
            let mut reader = BufReader::new(&mut file);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if buf.ends_with(b"\n") {
                            self.offset += n as u64;
                            self.parsed.apply_line(trim_line(&buf));
                        } else {
                            // Unterminated trailing line: apply only to this call's snapshot so a torn write
                            // never corrupts the committed cache offset.
                            trailing = Some(std::mem::take(&mut buf));
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }

        self.file_len = file_len;
        self.modified = modified;
        if self.offset != old_offset && self.offset > 0 {
            self.prefix_hash = hash_slice(&mut file, 0, self.offset.min(HASH_WINDOW)).unwrap_or(0);
            let tail_start = self.offset.saturating_sub(HASH_WINDOW);
            self.tail_hash = hash_slice(&mut file, tail_start, self.offset - tail_start).unwrap_or(0);
        }

        if let Some(last_line) = trailing {
            let mut snapshot = self.parsed.clone();
            snapshot.apply_line(trim_line(&last_line));
            snapshot.select_with_tail(epochs_done, waiting_resume, tail, q)
        } else {
            self.parsed.select_with_tail(epochs_done, waiting_resume, tail, q)
        }
    }
}

fn trim_line(mut line: &[u8]) -> &[u8] {
    if let Some(rest) = line.strip_suffix(b"\n") {
        line = rest;
    }
    if let Some(rest) = line.strip_suffix(b"\r") {
        line = rest;
    }
    line
}

fn hash_slice(file: &mut File, start: u64, len: u64) -> Option<u64> {
    let mut buf = [0u8; HASH_WINDOW as usize];
    let n = usize::try_from(len).ok()?.min(buf.len());
    file.seek(SeekFrom::Start(start)).ok()?;
    file.read_exact(&mut buf[..n]).ok()?;
    Some(fnv1a(&buf[..n]))
}

fn verify_windows(file: &mut File, offset: u64, prefix_hash: u64, tail_hash: u64) -> bool {
    let prefix_len = offset.min(HASH_WINDOW);
    if hash_slice(file, 0, prefix_len) != Some(prefix_hash) {
        return false;
    }
    let tail_start = offset.saturating_sub(HASH_WINDOW);
    hash_slice(file, tail_start, offset - tail_start) == Some(tail_hash)
}

impl Runs {
    /// Subscribes to a run's events.
    pub fn subscribe(&self, id: &str) -> Result<Subscription> {
        let e = self.get(id)?;
        // Under the state lock (which every status change holds while it sends), so the initial status and the
        // first event received are in order.
        let st = e.state.lock().unwrap();
        let events = e.tx.subscribe();
        Ok(Subscription { initial: status_event(&st.file.run), events })
    }

    /// Subscribes to a run's pre-serialized SSE frames with a weak resync callback for lag recovery (holding
    /// `Weak<RunEntry>` so deleting the run drops the channel and closes open streams).
    pub fn subscribe_sse(&self, id: &str) -> Result<RunSseSubscription> {
        let e = self.get(id)?;
        let (initial, frames) = {
            let st = e.state.lock().unwrap();
            (status_event(&st.file.run), e.sse_tx.subscribe())
        };
        let weak = Arc::downgrade(&e);
        let current: RunStatusResync = Arc::new(move || {
            let entry = weak.upgrade()?;
            let st = entry.state.lock().unwrap_or_else(|p| p.into_inner());
            (!st.deleted).then(|| status_event(&st.file.run))
        });
        Ok(RunSseSubscription {
            initial: SseFrame::run(&RunEvent::Status(initial)),
            stream: RunSseStream::Frames(frames),
            current: Some(current),
        })
    }

    /// The run's metric history as `q` selects it: `metrics.jsonl` (resumed epochs replace the steps and epochs
    /// they repeat) plus the steps not flushed to it yet. Those are the steps the trainer sent live, which skips some
    /// (they reach the file at the epoch's end), so a query with a step cursor takes them only as far as they follow
    /// on without a gap: a poller that continues after the last step it got never misses one.
    pub fn metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics> {
        let e = self.get(id)?;
        let path = e.dir.join("metrics.jsonl");
        loop {
            let (attempt, tail, epochs_done, waiting_resume) = {
                let st = e.state.lock().unwrap();
                (st.attempt, st.tail.clone(), st.file.epochs_done, st.file.resumed_at.is_some())
            };
            let m = {
                let mut cache = e.metrics_cache.lock().unwrap();
                cache.read_select(&path, epochs_done, waiting_resume, &tail, q)
            };
            // If the run resumed while `metrics.jsonl` was read, `tail` belongs to the previous attempt;
            // re-read so steps from two attempts are never mixed.
            if e.state.lock().unwrap().attempt == attempt {
                return Ok(m);
            }
        }
    }

    /// The log lines `q` selects, oldest first, with the length of the whole log. Skips discarded prefix lines
    /// without allocating `LogLine` strings for them.
    pub fn logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs> {
        let e = self.get(id)?;
        Ok(read_logs(&e.dir.join("log.jsonl"), q))
    }
}

/// Streams `log.jsonl` line-by-line, validating discarded lines in place without allocating `LogLine`s and only
/// deserializing the returned window.
pub(super) fn read_logs(path: &Path, q: &LogsQuery) -> RunLogs {
    let Ok(file) = File::open(path) else {
        return RunLogs { lines: Vec::new(), total: 0 };
    };
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    let mut total = 0u64;

    if let Some(after) = q.after {
        let mut lines = Vec::with_capacity(q.limit.min(1024));
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = trim_line(&buf);
                    if line.is_empty() {
                        continue;
                    }
                    if total < after || lines.len() >= q.limit {
                        if serde_json::from_slice::<LogLineCheck>(line).is_ok() {
                            total += 1;
                        }
                    } else if let Ok(parsed) = serde_json::from_slice::<LogLine>(line) {
                        lines.push(parsed);
                        total += 1;
                    }
                }
            }
        }
        return RunLogs { lines, total };
    }

    if q.limit == 0 {
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    let line = trim_line(&buf);
                    if !line.is_empty() && serde_json::from_slice::<LogLineCheck>(line).is_ok() {
                        total += 1;
                    }
                }
            }
        }
        return RunLogs { lines: Vec::new(), total };
    }

    // Tail mode (`after == None`): keep a bounded ring of raw JSON lines, reusing buffers once full, and
    // deserialize only the final `<= q.limit` lines.
    let mut ring: VecDeque<Vec<u8>> = VecDeque::with_capacity(q.limit.min(1024));
    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let line = trim_line(&buf);
                if line.is_empty() || serde_json::from_slice::<LogLineCheck>(line).is_err() {
                    continue;
                }
                total += 1;
                if ring.len() == q.limit {
                    let mut slot = ring.pop_front().expect("ring non-empty");
                    slot.clear();
                    slot.extend_from_slice(line);
                    ring.push_back(slot);
                } else {
                    ring.push_back(line.to_vec());
                }
            }
        }
    }
    let lines = ring.iter().filter_map(|raw| serde_json::from_slice::<LogLine>(raw).ok()).collect();
    RunLogs { lines, total }
}

/// Brings the progress in `run.json` of a run a previous process left active up to its files: the trainer writes
/// `weights/last.folo`, then the epoch's line in `metrics.jsonl`, then (through its event) `run.json`, so a crash in
/// between leaves `run.json` behind. Epochs done and the best epoch come from the newest of the three; the last
/// epoch's metrics only from `metrics.jsonl` (unknown when only the checkpoint has that epoch). A run that had
/// started ended when it last trained ([`last_trained`]), so the time the server was down is not training time.
pub(super) fn reconcile(file: &mut RunFile, dir: &Path) {
    if file.run.started.is_some() {
        file.run.finished = last_trained(file, dir).map(rfc3339);
    }
    let m = read_metrics(&dir.join("metrics.jsonl"));
    let ckpt = crate::engine::read_model_file(&dir.join("weights").join("last.folo"))
        .ok()
        .and_then(|f| f.checkpoint().map(|c| (c.epoch, c.best)));
    let from_metrics = m.epochs.last().map_or(0, |e| e.epoch);
    let from_ckpt = ckpt.map_or(0, |(e, _)| e.min(u32::MAX as u64) as u32);
    let done = file.epochs_done.max(from_metrics).max(from_ckpt);
    if done == file.epochs_done {
        return;
    }
    let mut best = file.run.best;
    for e in &m.epochs {
        if best.is_none_or(|b| e.val.map50_95 > b.map50_95) {
            best = Some(BestEpoch { epoch: e.epoch, map50_95: e.val.map50_95 });
        }
    }
    if let Some((_, b)) = ckpt
        && let Some(f) = b.fitness
        && best.is_none_or(|x| (f as f32) > x.map50_95)
    {
        best = Some(BestEpoch { epoch: (b.epoch + 1).max(1) as u32, map50_95: f as f32 });
    }
    file.run.best = best;
    file.last_metrics = m.epochs.last().filter(|e| e.epoch == done).map(|e| model_metrics(&e.val));
    file.epochs_done = done;
}

/// When a started run a previous process left active last trained (Unix seconds): `alive_at`, or for a run.json
/// from before it the newest of its files; a resume it has not trained since is later (it waited from then on).
/// Never before `started` nor after now; none when nothing tells.
fn last_trained(file: &RunFile, dir: &Path) -> Option<f64> {
    let started = file.run.started.as_deref().and_then(parse_rfc3339)?;
    let written = || {
        ["run.json", "metrics.jsonl", "log.jsonl", "weights/last.folo"]
            .iter()
            .filter_map(|f| std::fs::metadata(dir.join(f)).and_then(|m| m.modified()).ok())
            .filter_map(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as f64)
            .reduce(f64::max)
    };
    let at = file.alive_at.or_else(written).into_iter().chain(file.resumed_at).reduce(f64::max)?;
    Some(at.clamp(started, unix_now().floor().max(started)))
}

/// Parses `metrics.jsonl`. A resumed run appends the steps of the epoch it repeats again: a step or epoch number
/// at or below the last one seen replaces everything from it on.
pub(super) fn read_metrics(path: &Path) -> RunMetrics {
    let mut cache = MetricsCache::default();
    cache.read_select(path, 0, false, &[], &MetricsQuery::default())
}
