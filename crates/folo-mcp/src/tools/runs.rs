//! The training-run tools: train (or resume), status, list, stop, reorder and compare.

use std::collections::{BTreeMap, HashSet};

use folo_api::{LogLevel, LogLine, QueueRequest, Run, RunMetrics};
use folo_server::store::{
    AutoBatchOptions, LogsQuery, MAX_RUN_NAME_CHARS, MetricsQuery, QueueMove, check_auto_batch_options,
};
use rmcp::model::JsonObject;
use serde_json::Value;

use super::arguments::{check_name_at, parse_args};
use super::results::{
    MAX_NAME_CHARS, MAX_RESULT_BYTES, Output, json_len, out, page_of_window, page_window, short,
};
use crate::errors::{Error, invalid_arguments};
use crate::server::Inner;
use crate::types::*;

/// A run as a list item: names and the error cut so every item is small.
fn run_summary(mut r: Run) -> Run {
    r.name = short(&r.name, MAX_NAME_CHARS);
    r.config.name = short(&r.config.name, MAX_NAME_CHARS);
    r.error = r.error.map(|e| short(&e, 500));
    r
}

impl Inner {
    pub(super) async fn train(&self, mut args: JsonObject) -> Result<Output, Error> {
        // Resuming takes the arguments as given: the other fields must match the run's config or be omitted, so
        // they are not filled with TrainConfig's defaults (and `dataset_id` may be left out).
        if let Some(id) = args.get("resume_run_id").filter(|v| !v.is_null()) {
            let id =
                id.as_str().ok_or_else(|| invalid_arguments("resume_run_id must be a string"))?.to_string();
            return self.resume(id, args).await;
        }
        let mut batch_was_auto = false;
        if let Some(Value::String(s)) = args.get("batch") {
            if s == "auto" {
                if args.get("auto_batch").and_then(Value::as_bool) == Some(false) {
                    return Err(Error::new(
                        http::StatusCode::BAD_REQUEST,
                        "invalid_config",
                        "batch: \"auto\" conflicts with auto_batch: false",
                    ));
                }
                batch_was_auto = true;
                args.remove("batch");
            } else {
                return Err(Error::new(
                    http::StatusCode::BAD_REQUEST,
                    "invalid_config",
                    format!(
                        "invalid train config: batch must be an integer in 1..=1024 or \"auto\", got \"{s}\""
                    ),
                ));
            }
        }
        // An explicit `arch` next to `weights` (a fine-tune takes the weights' architecture, as `folo train --weights`
        // does) must name that architecture.
        let explicit_arch = args.get("arch").and_then(|v| v.as_str()).map(str::to_string);
        // A TrainConfig value the trainer cannot use (`arch: "folo-q"`, a negative `epochs`) is `invalid_config`,
        // as in the CLI and `POST /api/runs`' range checks; a missing `dataset_id` stays `invalid_arguments`.
        let a: TrainArgs = parse_args(args).map_err(|mut e| {
            if !e.body.message.contains("missing field") {
                e.body.code = "invalid_config".into();
                if let Some(rest) = e.body.message.strip_prefix("invalid arguments: ") {
                    e.body.message = format!("invalid train config: {rest}");
                }
            }
            e
        })?;
        let auto_batch = AutoBatchOptions {
            enabled: batch_was_auto || a.auto_batch.unwrap_or(false),
            target_utilization: a.auto_batch_target,
        };
        check_auto_batch_options(&auto_batch)?;
        // The server's limit (`POST /api/runs`), also in the inputSchema's `maxLength`.
        check_name_at("name", &a.config.name, MAX_RUN_NAME_CHARS)?;
        self.ensure_dataset(&a.config.dataset_id).await?;
        if let (Some(arch), Some(w)) = (explicit_arch, a.config.weights.clone()) {
            let model = self.blocking(move |s| s.get_model(&w)).await?;
            if model.arch.as_str() != arch {
                return Err(Error::new(
                    http::StatusCode::BAD_REQUEST,
                    "invalid_config",
                    format!(
                        "arch is {arch}, but the weights {} are {}: a fine-tune takes the weights' architecture",
                        model.id,
                        model.arch.as_str()
                    ),
                )
                .hint("leave `arch` out when `weights` is set (or set it to the weights' arch)"));
            }
        }
        let auto_note = if auto_batch.enabled { " (auto-batch)" } else { "" };
        let run = self
            .blocking(move |s| s.create_run_with_auto_batch(a.config, auto_batch))
            .await
            .map_err(|e| self.amp_hint(e, false))?;
        // Embedded runs end with this process, so they are stopped on disconnect; proxied runs belong to the
        // `folo serve` (and Studio) and outlive the session, and so do cloud runs, which train without this process.
        if self.proxy.is_none() && run.config.compute.is_none() {
            self.created_runs.lock().unwrap().push(run.id.clone());
        }
        let summary = format!(
            "started {} ({}): {} on {}, {} epochs{auto_note}; follow it with run_status or folo://runs/{}",
            run.id,
            run.name,
            run.config.arch.as_str(),
            run.config.dataset_id,
            run.epochs,
            run.id
        );
        out(&TrainResult { run_id: run.id, status: run.status }, summary)
    }

    /// A `not_supported` from `train` (an `amp` run this server cannot train: no tensor cores, or the CPU) gets a
    /// hint an MCP client can act on: the switch, or how to restart this server with tensor cores. A resumed run
    /// keeps its `amp`, so its hint is the restart, or a new f32 run from its model.
    fn amp_hint(&self, mut e: Error, resume: bool) -> Error {
        if e.body.code != "not_supported" {
            return e;
        }
        let restart = match &self.proxy {
            Some(p) => format!("restart the `folo serve` at {} with `--tensor-cores`", p.base()),
            None if self.mock => "this mock server's tensor cores are off".to_string(),
            None => {
                "restart this MCP server with tensor cores: add `--tensor-cores` to `folo mcp`'s args (or \
                     FOLO_GPU_TENSOR_CORES=1 to its `env`) in the MCP client's config"
                    .to_string()
            }
        };
        e.body.hint = Some(if resume {
            format!(
                "the run trained in mixed precision and resumes in it: {restart}; or train further in f32 with a new \
                 `train` whose `weights` is the run's model"
            )
        } else {
            format!(
                "set `amp: false` to train in f32 (or `device: \"auto\"` if the run asked for the CPU), or {restart}; \
             folo_info's `capabilities.amp` says when amp is available"
            )
        });
        e
    }

    async fn resume(&self, id: String, args: JsonObject) -> Result<Output, Error> {
        let rid = id.clone();
        let dataset_id = self.blocking(move |s| Ok(s.get_run(&rid)?.config.dataset_id)).await?;
        self.ensure_dataset(&dataset_id).await?;
        let rid = id.clone();
        let run =
            self.blocking(move |s| s.resume_run(&rid, &args)).await.map_err(|e| self.amp_hint(e, true))?;
        if self.proxy.is_none() && run.config.compute.is_none() {
            let mut created = self.created_runs.lock().unwrap();
            if !created.contains(&run.id) {
                created.push(run.id.clone());
            }
        }
        let summary = format!(
            "resuming {} ({}) after epoch {} of {}; follow it with run_status or folo://runs/{}",
            run.id, run.name, run.epoch, run.epochs, run.id
        );
        out(&TrainResult { run_id: run.id, status: run.status }, summary)
    }

    pub(super) async fn run_status(&self, a: RunStatusArgs) -> Result<Output, Error> {
        if a.log_lines > 200 {
            return Err(invalid_arguments(format!("log_lines must be in 0..=200, got {}", a.log_lines)));
        }
        let id = a.run_id.clone();
        let n = a.log_lines as usize;
        let (run, metrics, mut log) = self
            .blocking(move |s| {
                let run = s.get_run(&id)?;
                let metrics = s.run_metrics(&id, &MetricsQuery::every(u64::MAX))?;
                // The whole log (as far as /logs serves it), for the warnings; `log` keeps the last n lines.
                let log = s.run_logs(&id, &LogsQuery::last(WARNING_SCAN_LINES))?.lines;
                Ok((run, metrics, log))
            })
            .await?;
        let warnings = log_warnings(&log);
        log.drain(..log.len().saturating_sub(n));
        let mut r = RunStatusResult {
            run,
            last_step: metrics.steps.last().copied(),
            last_epoch: metrics.epochs.last().cloned(),
            warnings,
            log,
        };
        while json_len(&r) > MAX_RESULT_BYTES && !r.log.is_empty() {
            r.log.remove(0);
        }
        if json_len(&r) > MAX_RESULT_BYTES
            && let Some(e) = r.last_epoch.as_mut()
        {
            e.val.per_class.clear();
        }
        let mut line = run_line(&r.run);
        if let Some(w) = r.warnings.first() {
            line.push_str(&format!("; warning: {w}"));
            if r.warnings.len() > 1 {
                line.push_str(&format!(" (+{} more in `warnings`)", r.warnings.len() - 1));
            }
        }
        out(&r, line)
    }

    pub(super) async fn list_runs(&self, a: ListRunsArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        let mut all = self.blocking(|s| s.list_runs()).await?;
        if let Some(st) = a.status {
            all.retain(|r| r.status == st);
        }
        let all: Vec<Run> = all.into_iter().map(run_summary).collect();
        let page = page_of_window(all, offset, limit);
        let summary = format!("{} of {} runs", page.items.len(), page.total);
        out(&page, summary)
    }

    pub(super) async fn stop_run(&self, a: RunIdArgs) -> Result<Output, Error> {
        let run = self.blocking(move |s| s.stop_run(&a.run_id)).await?;
        out(&run, format!("{}: {}", run.id, run.status_str()))
    }

    pub(super) async fn reorder_run(&self, a: ReorderRunArgs) -> Result<Output, Error> {
        let mv = QueueMove::parse(QueueRequest { action: a.action, position: a.position })?;
        let run = self.blocking(move |s| s.move_run(&a.run_id, mv)).await?;
        let place = run
            .queue_position
            .map_or_else(|| run.status_str().to_string(), |p| format!("queue position {p}"));
        let summary = format!("{}: {place}", run.id);
        out(
            &ReorderResult { run_id: run.id, status: run.status, queue_position: run.queue_position },
            summary,
        )
    }

    pub(super) async fn compare_runs(&self, a: CompareArgs) -> Result<Output, Error> {
        if !(2..=8).contains(&a.run_ids.len()) {
            return Err(invalid_arguments(format!("run_ids must hold 2 to 8 ids, got {}", a.run_ids.len())));
        }
        let ids = a.run_ids.clone();
        let runs: Vec<(Run, RunMetrics)> = self
            .blocking(move |s| {
                ids.iter()
                    .map(|id| Ok((s.get_run(id)?, s.run_metrics(id, &MetricsQuery::every(u64::MAX))?)))
                    .collect()
            })
            .await?;
        let entries: Vec<CompareEntry> = runs
            .iter()
            .map(|(run, m)| {
                let best_metrics = run
                    .best
                    .and_then(|b| m.epochs.iter().find(|e| e.epoch == b.epoch))
                    .map(|e| metrics_of(&e.val));
                CompareEntry {
                    run_id: run.id.clone(),
                    name: run.name.clone(),
                    status: run.status,
                    arch: run.config.arch,
                    epochs_done: m.epochs.len() as u32,
                    epochs: run.epochs,
                    best: run.best,
                    best_metrics,
                    final_train_loss: m.epochs.last().map(|e| e.train_loss),
                    duration_s: (m.epochs.iter().map(|e| e.duration_s).sum::<f64>() * 10.0).round() / 10.0,
                }
            })
            .collect();
        let configs: Vec<Value> =
            runs.iter().map(|(r, _)| serde_json::to_value(&r.config).unwrap_or_default()).collect();
        let result = CompareResult { runs: entries, config_diff: config_diff(&configs) };
        let best = result
            .runs
            .iter()
            .filter_map(|e| e.best.map(|b| (e, b)))
            .max_by(|x, y| x.1.map50_95.total_cmp(&y.1.map50_95));
        let summary = match best {
            Some((e, b)) => format!(
                "{} runs; best {} ({}) with mAP50-95 {:.3} at epoch {}; {} config fields differ",
                result.runs.len(),
                e.run_id,
                e.name,
                b.map50_95,
                b.epoch,
                result.config_diff.len()
            ),
            None => format!("{} runs, none validated yet", result.runs.len()),
        };
        out(&result, summary)
    }
}

trait StatusStr {
    fn status_str(&self) -> String;
}

impl StatusStr for Run {
    fn status_str(&self) -> String {
        serde_json::to_value(self.status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default()
    }
}

/// One line describing a run.
/// Log lines `run_status` reads for its `warnings` (the server's `/logs` maximum).
const WARNING_SCAN_LINES: usize = 5000;

/// Distinct `warn` lines `run_status` reports.
pub(super) const MAX_WARNINGS: usize = 5;

/// The distinct `warn`-level messages of a run's log, oldest first, at most [`MAX_WARNINGS`] (each capped at 400
/// characters).
pub(super) fn log_warnings(log: &[LogLine]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for l in log.iter().filter(|l| l.level == LogLevel::Warn) {
        let m = folo_api::codes::cap(&l.message, 400);
        if !out.contains(&m) {
            out.push(m);
            if out.len() == MAX_WARNINGS {
                break;
            }
        }
    }
    out
}

fn run_line(r: &Run) -> String {
    let mut s =
        format!("{} ({}): {}, epoch {}/{}, step {}", r.id, r.name, r.status_str(), r.epoch, r.epochs, r.step);
    if let Some(b) = r.best {
        s.push_str(&format!(", best mAP50-95 {:.3} at epoch {}", b.map50_95, b.epoch));
    }
    if let Some(eta) = r.eta_s {
        s.push_str(&format!(", ETA {eta:.0} s"));
    }
    if let Some(m) = &r.model_id {
        s.push_str(&format!(", model {m}"));
    }
    if let Some(e) = &r.error {
        s.push_str(&format!(", error: {e}"));
    }
    s
}

/// Dotted-path fields whose values differ between `configs`, with one value per config.
pub(super) fn config_diff(configs: &[Value]) -> BTreeMap<String, Vec<Value>> {
    fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                    flatten(&key, v, out);
                }
            }
            other => {
                out.insert(prefix.to_string(), other.clone());
            }
        }
    }
    let flat: Vec<BTreeMap<String, Value>> = configs
        .iter()
        .map(|c| {
            let mut m = BTreeMap::new();
            flatten("", c, &mut m);
            m
        })
        .collect();
    let keys: HashSet<&String> = flat.iter().flat_map(|m| m.keys()).collect();
    let mut diff = BTreeMap::new();
    for k in keys {
        let values: Vec<Value> = flat.iter().map(|m| m.get(k).cloned().unwrap_or(Value::Null)).collect();
        if values.windows(2).any(|w| w[0] != w[1]) {
            diff.insert(k.clone(), values);
        }
    }
    diff
}
