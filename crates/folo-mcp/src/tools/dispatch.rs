//! A tool call from the request to its handler: the read-only and argument checks, progress notifications and
//! the error codes a tool reports.

use std::time::{Duration, Instant};

use rmcp::model::{JsonObject, ProgressNotificationParam, ProgressToken};
use rmcp::service::{Peer, RoleServer};

use super::ToolDef;
use super::arguments::{parse_args, unknown_argument};
use super::results::Output;
use crate::errors::{Error, read_only};
use crate::server::Inner;

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
        if let Some(e) = unknown_argument(def.name, &(def.input)(), &args) {
            return Err(e);
        }
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
            other => Err(Error::internal(format!("tool `{other}` has no handler"))),
        };
        res.map_err(|e| mcp_code(crate::confine::map_denied(e)))
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
