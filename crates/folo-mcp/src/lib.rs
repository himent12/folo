//! `folo mcp`: a Model Context Protocol server over stdio, so agents (Claude Code, Codex, any MCP client) can build
//! datasets, train, evaluate, run and export FOLO models. The contract is `docs/MCP.md`.
//!
//! The protocol layer is the official Rust SDK (`rmcp`, pinned), which implements both revisions FOLO supports
//! (2026-07-28 and 2025-11-25), the newline-delimited stdio framing with a single writer, and the JSON-RPC error
//! rules. This crate adds the FOLO tools, resources and prompts on top of a [`Store`]: the embedded
//! [`RealStore`] or [`MockStore`], or [`proxy::HttpStore`], which forwards to a running `folo serve`. Tool
//! schemas are generated with schemars from the `folo-api` types, so they match the HTTP API.
//!
//! [`build`] picks the store for a [`Backend`] and [`serve_stdio`] runs the session; `folo mcp`
//! (crates/folo-cli/src/mcp.rs) calls both. The modules:
//!
//! - `server`: the rmcp `ServerHandler` ([`FoloMcp`]): tools, resources and subscriptions, prompts, logging.
//! - `batch_cache`: bounded LRU cache of recent `predict_batch` payloads (`folo://predictions/{batch_id}`).
//! - `tools/`: the tool table, its schemas and argument checks, and one handler file per area.
//! - `resources`: `folo://` URIs, listing and templates; `render`: the sample-grid JPEG of
//!   `folo://runs/{id}/samples/{epoch}`.
//! - `prompts`: `train_on_folder` and `diagnose_run`.
//! - [`proxy`]: `--server`, a `Store` over the HTTP API through `folo-client`.
//! - `paths` and `confine`: relative input paths and `--confine`.
//! - `errors`: tool errors (the server's error type); [`types`]: the MCP-only argument and result types.
//! - [`label`]: agent labeling in process or through `folo serve`, shared with `folo label`; the view of an image
//!   with its boxes numbered.
//! - [`bench`]: vision benchmarks in process or through `folo serve`, shared with `folo bench`.

mod batch_cache;
pub mod bench;
mod confine;
mod errors;
pub mod label;
mod paths;
mod prompts;
pub mod proxy;
mod render;
mod resources;
mod server;
mod stdio;
mod tools;
pub mod types;

use std::path::PathBuf;
use std::sync::Arc;

use folo_server::{MockStore, RealStore, Store};
use rmcp::ServiceExt;
use rmcp::service::ServerInitializeError;

pub use server::{FoloMcp, VERSIONS};

/// Server options (the `folo mcp` flags).
#[derive(Clone, Debug)]
pub struct Options {
    /// Where FOLO writes, and the base of relative input paths.
    pub data_dir: PathBuf,
    /// Hide and reject mutating tools.
    pub read_only: bool,
    /// Only read input paths inside `data_dir`.
    pub confine: bool,
    /// Embedded mode: the real store's inference and validation device (`auto`, `cpu`, `gpu:<n>`).
    pub device: String,
    /// Embedded mode: the `folo weights` cache whose downloaded published weights list as models
    /// ([`RealStore::published_weights`]; `folo mcp` passes `folo_api::weights::cache_dir()`).
    pub published_weights: Option<PathBuf>,
}

/// Where tool calls go.
#[derive(Clone)]
pub enum Backend {
    /// The real store on `data_dir` (embedded mode).
    Real,
    /// Deterministic mock data (`--mock`).
    Mock,
    /// A running `folo serve` (`--server`).
    Proxy {
        /// Its base URL, `http://` or `https://`.
        url: String,
        /// The token it was started with (`--token`), if any.
        token: Option<String>,
    },
}

// Not derived: the token must not reach a log through `{:?}`.
impl std::fmt::Debug for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Real => f.write_str("Real"),
            Backend::Mock => f.write_str("Mock"),
            Backend::Proxy { url, token } => {
                f.debug_struct("Proxy").field("url", url).field("token", &token.is_some()).finish()
            }
        }
    }
}

/// The tools as `{ name, description, mutating, inputSchema, outputSchema }` (for `folo describe`).
pub fn describe_tools() -> serde_json::Value {
    tools::describe()
}

/// Opens the embedded [`RealStore`] configured by `opts`, skipping `Benches::open` when `--confine` refuses every
/// benchmark tool.
pub(crate) fn open_real_store(opts: &Options) -> RealStore {
    let store = if opts.confine {
        RealStore::open_without_benches(opts.data_dir.clone())
    } else {
        RealStore::open(opts.data_dir.clone())
    };
    let mut store = store.device(opts.device.clone());
    if let Some(dir) = &opts.published_weights {
        store = store.published_weights(dir);
    }
    // Test support: how long a dataset's index is trusted before its files are looked at again (5 s).
    if let Some(ms) = std::env::var("FOLO_DATASET_RECHECK_MS").ok().and_then(|v| v.parse().ok()) {
        store = store.recheck_every(std::time::Duration::from_millis(ms));
    }
    store
}

/// Builds the server for `backend`.
pub fn build(backend: Backend, opts: Options) -> Result<FoloMcp, String> {
    // With --confine, the dataset reader refuses files outside the data dir (embedded mode; in proxy mode
    // `folo serve` reads the datasets).
    if opts.confine && !matches!(backend, Backend::Proxy { .. }) {
        let policy = paths::PathPolicy { data_dir: opts.data_dir.clone(), confine: true };
        confine::install_read_guard(&policy).map_err(|e| e.body.message)?;
    }
    Ok(match backend {
        Backend::Real => FoloMcp::new(Arc::new(open_real_store(&opts)), None, opts),
        Backend::Mock => FoloMcp::new(Arc::new(MockStore::new()), None, opts),
        Backend::Proxy { url, token } => {
            if let Err(why) = folo_client::check_base_url(&url) {
                return Err(format!(
                    "--server (or FOLO_SERVER) must be the http:// or https:// URL of a `folo serve`, got `{url}` ({why})"
                ));
            }
            let http = Arc::new(proxy::HttpStore::new(&url, token.as_deref()));
            let store: Arc<dyn Store> = http.clone();
            FoloMcp::new(store, Some(http), opts)
        }
    })
}

/// Serves MCP on stdin / stdout until the client disconnects or `shutdown` resolves, then stops the runs this
/// session started and, in embedded mode, shuts the store down ([`Store::shutdown`]: the active run stops at its
/// next batch and keeps its last checkpoint). stdout carries protocol messages only; logs go to stderr.
pub async fn serve_stdio(
    server: FoloMcp,
    shutdown: impl std::future::Future<Output = ()> + Send,
) -> Result<(), String> {
    let transport = (tokio::io::stdin(), stdio::ErrorIdWriter::new(tokio::io::stdout()));
    let running = match server.clone().serve(transport).await {
        Ok(r) => Some(r),
        // The client went away (or never spoke) before the lifecycle started: nothing to clean up.
        Err(ServerInitializeError::ConnectionClosed(_)) => None,
        Err(e) => {
            tracing::warn!("MCP session ended during startup: {e}");
            None
        }
    };
    if let Some(running) = running {
        tokio::select! {
            r = running.waiting() => {
                if let Err(e) = r {
                    tracing::warn!("MCP service task failed: {e}");
                }
            }
            _ = shutdown => tracing::info!("shutting down"),
        }
    }
    server.stop_session_runs().await;
    server.shutdown_store().await;
    Ok(())
}
