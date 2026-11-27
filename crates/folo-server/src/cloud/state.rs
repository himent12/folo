//! `runs/<id>/cloud.json`: what a process needs to follow a cloud run again after a restart: its instance, its
//! worker's address and token, the ids it has there and the cursors of its history. Written owner-only (the token is
//! a credential) after every change that matters, and never served.

use std::path::{Path, PathBuf};

use folo_api::RunStatus;
use serde::{Deserialize, Serialize};

use super::files;

/// How a cloud run ended on its worker, kept until its instance is released, so a restart in between still
/// publishes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Outcome {
    pub status: RunStatus,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
}

impl Outcome {
    /// `status` without an error.
    pub fn plain(status: RunStatus) -> Outcome {
        Outcome { status, error: None, error_code: None }
    }

    /// `failed` with `code` and `message`.
    pub fn failed(code: &str, message: impl Into<String>) -> Outcome {
        Outcome { status: RunStatus::Failed, error: Some(message.into()), error_code: Some(code.into()) }
    }
}

/// The private state of one cloud run.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct Private {
    /// Its ledger record (provisioning providers).
    #[serde(default)]
    pub instance_id: Option<String>,
    /// The worker's base URL, once the instance runs.
    #[serde(default)]
    pub worker_url: Option<String>,
    /// The token of a provisioned worker (generated for the run). A `remote` worker's token is its connection's
    /// credential and stays in `connections.json`.
    #[serde(default)]
    pub token: Option<String>,
    /// Unix seconds the run's time limit ends at.
    #[serde(default)]
    pub deadline: Option<f64>,
    /// The dataset's class names (for the samples files).
    #[serde(default)]
    pub classes: Vec<String>,
    /// The run on the worker.
    #[serde(default)]
    pub run_id: Option<String>,
    /// History mirrored so far: the last step and epoch (`/metrics` cursors) and the log lines.
    #[serde(default)]
    pub after_step: u64,
    #[serde(default)]
    pub after_epoch: u32,
    #[serde(default)]
    pub after_log: u64,
    /// Unix seconds `weights/last.folo` was last copied.
    #[serde(default)]
    pub synced: Option<f64>,
    /// Automatic preemption recovery attempts already started for this run.
    #[serde(default)]
    pub preemptions: u32,
    /// Cumulative estimated USD cost of earlier instances replaced after preemption.
    #[serde(default)]
    pub prior_cost_usd: f64,
    /// How the run ended on the worker, once its weights were collected.
    #[serde(default)]
    pub outcome: Option<Outcome>,
}

// Not derived: the token must not reach a log through `{:?}`.
impl std::fmt::Debug for Private {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Private")
            .field("instance_id", &self.instance_id)
            .field("worker_url", &self.worker_url)
            .field("run_id", &self.run_id)
            .field("preemptions", &self.preemptions)
            .finish_non_exhaustive()
    }
}

fn path(run_dir: &Path) -> PathBuf {
    run_dir.join("cloud.json")
}

impl Private {
    /// The state saved in `run_dir`, empty when there is none (or it does not parse: the run is then ended as one
    /// that never trained).
    pub fn load(run_dir: &Path) -> Private {
        std::fs::read(path(run_dir)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    /// Saves it in `run_dir`.
    pub fn save(&self, run_dir: &Path) {
        let bytes = serde_json::to_vec_pretty(self).expect("cloud state serializes");
        if let Err(e) = files::write_private(&path(run_dir), &bytes) {
            tracing::warn!("writing {}: {e}", path(run_dir).display());
        }
    }
}
