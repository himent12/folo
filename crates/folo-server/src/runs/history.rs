//! A run's history: event subscriptions, metrics and logs, and bringing `run.json` up to the files after a crash.

use std::path::Path;

use folo_api::{BestEpoch, Epoch, RunLogs, RunMetrics, Step};

use crate::error::Result;
use crate::store::{LogsQuery, MetricsQuery, Subscription};

use super::entry::{model_metrics, status_event};
use super::{RunFile, Runs};

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

    /// The run's metric history as `q` selects it: `metrics.jsonl` (resumed epochs replace the steps and epochs
    /// they repeat) plus the steps not flushed to it yet. Those are the steps the trainer sent live, which skips some
    /// (they reach the file at the epoch's end), so a query with a step cursor takes them only as far as they follow
    /// on without a gap: a poller that continues after the last step it got never misses one.
    pub fn metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics> {
        let e = self.get(id)?;
        let mut m = read_metrics(&e.dir.join("metrics.jsonl"));
        let tail = e.state.lock().unwrap().tail.clone();
        let mut last = m.steps.last().map_or(0, |s| s.step);
        for s in tail {
            if s.step <= last {
                continue;
            }
            if q.after_step.is_some() && s.step != last + 1 {
                break;
            }
            last = s.step;
            m.steps.push(s);
        }
        Ok(q.select(m))
    }

    /// The log lines `q` selects, oldest first, with the length of the whole log.
    pub fn logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs> {
        let e = self.get(id)?;
        let text = std::fs::read_to_string(e.dir.join("log.jsonl")).unwrap_or_default();
        Ok(q.select(text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()))
    }
}

/// Brings the progress in `run.json` of a run a previous process left active up to its files: the trainer writes
/// `weights/last.folo`, then the epoch's line in `metrics.jsonl`, then (through its event) `run.json`, so a crash in
/// between leaves `run.json` behind. Epochs done and the best epoch come from the newest of the three; the last
/// epoch's metrics only from `metrics.jsonl` (unknown when only the checkpoint has that epoch).
pub(super) fn reconcile(file: &mut RunFile, dir: &Path) {
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

/// Parses `metrics.jsonl`. A resumed run appends the steps of the epoch it repeats again: a step or epoch number
/// at or below the last one seen replaces everything from it on.
pub(super) fn read_metrics(path: &Path) -> RunMetrics {
    let mut m = RunMetrics { steps: Vec::new(), epochs: Vec::new() };
    let Ok(text) = std::fs::read_to_string(path) else { return m };
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        match v.get("event").and_then(|e| e.as_str()) {
            Some("step") => {
                if let Ok(s) = serde_json::from_value::<Step>(v) {
                    while m.steps.last().is_some_and(|l| l.step >= s.step) {
                        m.steps.pop();
                    }
                    m.epochs.retain(|e| e.epoch < s.epoch);
                    m.steps.push(s);
                }
            }
            Some("epoch") => {
                if let Ok(e) = serde_json::from_value::<Epoch>(v) {
                    m.epochs.retain(|x| x.epoch < e.epoch);
                    m.epochs.push(e);
                }
            }
            _ => {}
        }
    }
    m
}
