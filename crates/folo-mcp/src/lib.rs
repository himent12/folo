//! `folo mcp`: a Model Context Protocol server over stdio, so agents (Claude Code, Codex, any MCP client) can build
//! datasets, train, evaluate, run and export FOLO models. The contract is `docs/MCP.md`.
//!
//! The protocol layer is the official Rust SDK (`rmcp`, pinned), which implements both revisions FOLO supports
//! (2026-07-28 and 2025-11-25), the newline-delimited stdio framing with a single writer, and the JSON-RPC error
//! rules. This crate adds the FOLO tools, resources and prompts on top of a [`Store`]: the embedded
//! [`RealStore`] or [`MockStore`], or [`proxy::HttpStore`], which forwards to a running `folo serve`. Tool
//! schemas are generated with schemars from the `folo-api` types, so they match the HTTP API.

mod confine;
mod errors;
mod paths;
mod prompts;
pub mod proxy;
mod render;
mod resources;
mod server;
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
}

/// Where tool calls go.
#[derive(Clone, Debug)]
pub enum Backend {
    /// The real store on `data_dir` (embedded mode).
    Real,
    /// Deterministic mock data (`--mock`).
    Mock,
    /// A running `folo serve` at this base URL (`--server`).
    Proxy(String),
}

/// The tools as `{ name, description, mutating, inputSchema, outputSchema }` (for `folo describe`).
pub fn describe_tools() -> serde_json::Value {
    tools::describe()
}

/// Builds the server for `backend`.
pub fn build(backend: Backend, opts: Options) -> Result<FoloMcp, String> {
    // With --confine, the dataset reader refuses files outside the data dir (embedded mode; in proxy mode
    // `folo serve` reads the datasets).
    if opts.confine && !matches!(backend, Backend::Proxy(_)) {
        let policy = paths::PathPolicy { data_dir: opts.data_dir.clone(), confine: true };
        confine::install_read_guard(&policy).map_err(|e| e.body.message)?;
    }
    Ok(match backend {
        Backend::Real => {
            let store = RealStore::open(opts.data_dir.clone()).device(opts.device.clone());
            FoloMcp::new(Arc::new(store), None, opts)
        }
        Backend::Mock => FoloMcp::new(Arc::new(MockStore::new()), None, opts),
        Backend::Proxy(url) => {
            if !url.starts_with("http://") {
                return Err(format!("--server must be an http:// URL of a `folo serve`, got `{url}`"));
            }
            let http = Arc::new(proxy::HttpStore::new(&url));
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
    let running = match server.clone().serve(rmcp::transport::stdio()).await {
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
