//! A tool call from the request to its handler: the read-only and argument checks, progress notifications and
//! the error codes a tool reports.

use std::time::{Duration, Instant};

use rmcp::model::{JsonObject, ProgressNotificationParam, ProgressToken};
use rmcp::service::{Peer, RoleServer};
use serde_json::Value;

use super::ToolDef;
use super::arguments::{MAX_SUGGEST_CHARS, closest_named, parse_args, unknown_argument};
use super::results::{Output, short};
use crate::errors::{Error, read_only};
use crate::server::Inner;

/// Candidate identifier(s) supplied by a tool call or resource read for fuzzy `"did you mean ...?"`
/// enrichment on `*_not_found` errors.
pub(crate) enum NotFoundTarget {
    Dataset(String),
    Model(String),
    Run(String),
    Runs(Vec<String>),
}

/// Entity IDs captured from tool arguments before dispatch so `*_not_found` errors can suggest close IDs.
struct CallIds {
    dataset_id: Option<String>,
    model_id: Option<String>,
    run_id: Option<String>,
    run_ids: Vec<String>,
}

impl CallIds {
    fn from_args(tool: &str, args: &JsonObject) -> Self {
        let take = |key: &str| {
            args.get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && s.chars().count() <= MAX_SUGGEST_CHARS)
                .map(str::to_string)
        };
        let dataset_id = take("dataset_id");
        let model_id = take("model_id").or_else(|| (tool == "train").then(|| take("weights")).flatten());
        let run_id = take("run_id").or_else(|| (tool == "train").then(|| take("resume_run_id")).flatten());
        let run_ids = args
            .get("run_ids")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .filter(|s| !s.is_empty() && s.chars().count() <= MAX_SUGGEST_CHARS)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        Self { dataset_id, model_id, run_id, run_ids }
    }

    fn target_for(&self, code: &str) -> Option<NotFoundTarget> {
        match code {
            "dataset_not_found" => self.dataset_id.clone().map(NotFoundTarget::Dataset),
            "model_not_found" => self.model_id.clone().map(NotFoundTarget::Model),
            "run_not_found" => {
                if let Some(id) = &self.run_id {
                    Some(NotFoundTarget::Run(id.clone()))
                } else if !self.run_ids.is_empty() {
                    Some(NotFoundTarget::Runs(self.run_ids.clone()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn format_suggestion(id: &str, name: &str) -> String {
    if name.is_empty() || name == id {
        format!("did you mean `{id}`?")
    } else {
        format!("did you mean `{id}` ({})?", short(name, 48))
    }
}

fn append_hint(e: &mut Error, extra: String) {
    match &mut e.body.hint {
        Some(h) if !h.is_empty() => {
            h.push_str("; ");
            h.push_str(&extra);
        }
        slot => *slot = Some(extra),
    }
}

/// Sends `notifications/progress` for one request, if it asked for progress: monotonically increasing values, at
/// most ~4 per second (the final value is always sent).
pub struct Progress {
    target: Option<(Peer<RoleServer>, ProgressToken)>,
    last_sent: Option<Instant>,
    last_value: f64,
}

impl Progress {
    /// Progress for a request with `token` (none without one).
    pub fn new(peer: Peer<RoleServer>, token: Option<ProgressToken>) -> Progress {
        Progress { target: token.map(|t| (peer, t)), last_sent: None, last_value: f64::NEG_INFINITY }
    }

    /// No progress (tests call tools without a peer).
    #[cfg(test)]
    pub fn none() -> Progress {
        Progress { target: None, last_sent: None, last_value: f64::NEG_INFINITY }
    }

    /// Reports `done` of `total`. `last` forces the send (the final update before the result).
    pub async fn report(&mut self, done: f64, total: Option<f64>, last: bool) {
        let Some((peer, token)) = &self.target else { return };
        if done <= self.last_value {
            return;
        }
        if !last && self.last_sent.is_some_and(|t| t.elapsed() < Duration::from_millis(250)) {
            return;
        }
        let mut p = ProgressNotificationParam::new(token.clone(), done);
        if let Some(t) = total {
            p = p.with_total(t);
        }
        if peer.notify_progress(p).await.is_ok() {
            self.last_sent = Some(Instant::now());
            self.last_value = done;
        }
    }
}

impl Inner {
    /// Dispatches a tool call. `Err` is a tool error (an `isError` result), never a protocol error.
    pub async fn call(
        &self,
        def: &ToolDef,
        args: JsonObject,
        progress: &mut Progress,
    ) -> Result<Output, Error> {
        if self.opts.read_only && def.mutating {
            return Err(read_only(def.name));
        }
        if let Some(e) = unknown_argument(def.name, def.input_schema(), &args) {
            return Err(e);
        }
        super::bench::refuse_confined(def.name, self.opts.confine)?;
        let call_ids = CallIds::from_args(def.name, &args);
        let res = match def.name {
            "folo_info" => self.folo_info(parse_args(args)?).await,
            "list_datasets" => self.list_datasets(parse_args(args)?).await,
            "get_dataset" => self.get_dataset(parse_args(args)?).await,
            "list_dataset_issues" => self.list_dataset_issues(parse_args(args)?).await,
            "list_dataset_images" => self.list_dataset_images(parse_args(args)?).await,
            "register_dataset" => self.register_dataset(parse_args(args)?).await,
            "create_synthetic_dataset" => self.create_synthetic(parse_args(args)?, progress).await,
            "list_models" => self.list_models(parse_args(args)?).await,
            "train" => self.train(args).await,
            "run_status" => self.run_status(parse_args(args)?).await,
            "list_runs" => self.list_runs(parse_args(args)?).await,
            "stop_run" => self.stop_run(parse_args(args)?).await,
            "reorder_run" => self.reorder_run(parse_args(args)?).await,
            "val" => self.val(parse_args(args)?, progress).await,
            "predict" => self.predict(parse_args(args)?).await,
            "predict_batch" => self.predict_batch(parse_args(args)?, progress).await,
            "export" => self.export(parse_args(args)?).await,
            "compare_runs" => self.compare_runs(parse_args(args)?).await,
            "cloud_connections" => self.cloud_connections(parse_args(args)?).await,
            "cloud_offers" => self.cloud_offers(parse_args(args)?).await,
            "cloud_instances" => self.cloud_instances(parse_args(args)?).await,
            "cloud_terminate" => self.cloud_terminate(parse_args(args)?).await,
            "list_label_jobs" => self.list_label_jobs(parse_args(args)?).await,
            "create_label_job" => self.create_label_job(parse_args(args)?).await,
            "start_labeling" => self.start_labeling(parse_args(args)?).await,
            "stop_labeling" => self.stop_labeling(parse_args(args)?).await,
            "label_status" => self.label_status(parse_args(args)?).await,
            "list_label_images" => self.list_label_images(parse_args(args)?).await,
            "view_label_image" => self.view_label_image(parse_args(args)?).await,
            "set_label_boxes" => self.set_label_boxes(parse_args(args)?).await,
            "review_label_images" => self.review_label_images(parse_args(args)?).await,
            "export_label_dataset" => self.export_label_dataset(parse_args(args)?).await,
            "bench_catalog" => self.bench_catalog(parse_args(args)?).await,
            "list_benchmarks" => self.list_benchmarks(parse_args(args)?).await,
            "get_benchmark" => self.get_benchmark(parse_args(args)?).await,
            "create_benchmark" => self.create_benchmark(parse_args(args)?).await,
            "delete_benchmark" => self.delete_benchmark(parse_args(args)?).await,
            "start_bench_entry" => self.start_bench_entry(parse_args(args)?).await,
            "bench_entry" => self.bench_entry(parse_args(args)?).await,
            "stop_bench_entry" => self.stop_bench_entry(parse_args(args)?).await,
            other => Err(Error::internal(format!("tool `{other}` has no handler"))),
        };
        match res {
            Ok(out) => Ok(out),
            Err(e) => {
                let e = mcp_code(crate::confine::map_denied(e));
                let target = call_ids.target_for(&e.body.code);
                Err(match target {
                    Some(t) => self.enrich_not_found(e, t).await,
                    None => e,
                })
            }
        }
    }

    /// Appends a fuzzy `"did you mean ...?"` suggestion to `e.body.hint` when `e` is a matching
    /// `*_not_found` error. Dataset suggestions strictly respect `--confine` so hidden datasets are
    /// never leaked. Listing failures leave `e` unchanged.
    pub(crate) async fn enrich_not_found(&self, mut e: Error, target: NotFoundTarget) -> Error {
        let suggestion = match (e.body.code.as_str(), target) {
            ("dataset_not_found", NotFoundTarget::Dataset(id)) => {
                if id.is_empty() || id.chars().count() > MAX_SUGGEST_CHARS {
                    return e;
                }
                let c = self.confinement.clone();
                let Ok(items) = self
                    .blocking(move |s| {
                        Ok(s.list_datasets()?
                            .into_iter()
                            .filter(|d| c.dataset(s, &d.id).is_ok())
                            .map(|d| (d.id, d.name))
                            .collect::<Vec<_>>())
                    })
                    .await
                else {
                    return e;
                };
                closest_named(&id, &items).map(|(cid, name)| format_suggestion(cid, name))
            }
            ("model_not_found", NotFoundTarget::Model(id)) => {
                if id.is_empty() || id.chars().count() > MAX_SUGGEST_CHARS {
                    return e;
                }
                let Ok(items) = self
                    .blocking(|s| {
                        Ok(s.list_models()?.into_iter().map(|m| (m.id, m.name)).collect::<Vec<_>>())
                    })
                    .await
                else {
                    return e;
                };
                closest_named(&id, &items).map(|(cid, name)| format_suggestion(cid, name))
            }
            ("run_not_found", NotFoundTarget::Run(id)) => {
                if id.is_empty() || id.chars().count() > MAX_SUGGEST_CHARS {
                    return e;
                }
                let Ok(items) = self
                    .blocking(|s| Ok(s.list_runs()?.into_iter().map(|r| (r.id, r.name)).collect::<Vec<_>>()))
                    .await
                else {
                    return e;
                };
                closest_named(&id, &items).map(|(cid, name)| format_suggestion(cid, name))
            }
            ("run_not_found", NotFoundTarget::Runs(ids)) => {
                if ids.is_empty() {
                    return e;
                }
                let Ok(items) = self
                    .blocking(|s| Ok(s.list_runs()?.into_iter().map(|r| (r.id, r.name)).collect::<Vec<_>>()))
                    .await
                else {
                    return e;
                };
                ids.iter()
                    .find(|id| !items.iter().any(|(existing, _)| existing == *id))
                    .and_then(|missing| closest_named(missing, &items))
                    .map(|(cid, name)| format_suggestion(cid, name))
            }
            _ => None,
        };
        if let Some(s) = suggestion {
            append_hint(&mut e, s);
        }
        e
    }

    /// With `--confine`, fails unless dataset `id` reads only files inside the data dir.
    pub(super) async fn ensure_dataset(&self, id: &str) -> Result<(), Error> {
        if !self.confinement.active() {
            return Ok(());
        }
        let (c, id) = (self.confinement.clone(), id.to_string());
        self.blocking(move |s| c.dataset(s, &id)).await
    }

    /// Resolves an input path on the blocking pool (canonicalisation touches the disk).
    pub(super) async fn resolve(&self, input: String) -> Result<std::path::PathBuf, Error> {
        let paths = self.paths.clone();
        tokio::task::spawn_blocking(move || paths.resolve(&input)).await.map_err(join)?
    }
}

/// Store errors keep their codes, except the HTTP layer's names for bad input (`invalid_body`, `invalid_query`,
/// `invalid_path`, `bad_request`), which a tool reports as `invalid_arguments` (one vocabulary with the CLI:
/// `folo_api::codes`).
fn mcp_code(mut e: Error) -> Error {
    if matches!(e.body.code.as_str(), "invalid_body" | "invalid_query" | "invalid_path" | "bad_request") {
        e.body.code = "invalid_arguments".into();
        e.body.hint.get_or_insert_with(|| "check the tool's inputSchema (tools/list)".into());
    }
    e
}

pub(crate) fn join(e: tokio::task::JoinError) -> Error {
    Error::internal(format!("task failed: {e}"))
}
