//! The MCP `ServerHandler`: tools, resources (with subscriptions), prompts and logging on top of a
//! [`Store`](folo_server::Store). The protocol itself (both revisions, framing, the `initialize` handshake,
//! `server/discover`, per-request metadata, `subscriptions/listen` acknowledgements) is rmcp's.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use base64::Engine as _;
use folo_api::{GpuBackend, GpuInfo};
use folo_server::Store;
use rmcp::ErrorData as McpError;
use rmcp::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, GetPromptRequestParams,
    GetPromptResponse, Implementation, ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult,
    ListToolsResult, PaginatedRequestParams, ProtocolVersion, ReadResourceRequestParams,
    ReadResourceResponse, ReadResourceResult, ResourceContents, ResourceUpdatedNotificationParam,
    ServerCapabilities, ServerConfig, SubscribeRequestParams, SubscriptionFilter, ToolsCapability,
    UnsubscribeRequestParams,
};
use rmcp::service::{RequestContext, RoleServer, SubscriptionContext};
use serde_json::{Value, json};
use tokio::task::AbortHandle;

use crate::confine::Confinement;
use crate::errors::Error;
use crate::paths::PathPolicy;
use crate::proxy::HttpStore;
use crate::resources::{self, FoloUri};
use crate::tools::{self, Progress};
use crate::{Options, prompts};

/// The revisions this server implements and tests (docs/MCP.md).
pub static VERSIONS: [ProtocolVersion; 2] = [ProtocolVersion::V_2026_07_28, ProtocolVersion::V_2025_11_25];

/// How often subscribed resources are checked for changes (and so the most updates per second).
pub const WATCH_INTERVAL: Duration = Duration::from_secs(1);

/// State shared by every request of the connection.
pub struct Inner {
    pub(crate) store: Arc<dyn Store>,
    pub(crate) proxy: Option<Arc<HttpStore>>,
    pub(crate) mock: bool,
    pub(crate) opts: Options,
    pub(crate) paths: PathPolicy,
    /// The dataset side of `--confine`.
    pub(crate) confinement: Confinement,
    /// Runs started by this session (stopped gracefully when the client disconnects).
    pub(crate) created_runs: Mutex<Vec<String>>,
    /// `predict_batch` results, served as `folo://predictions/{batch_id}` for the server's lifetime.
    pub(crate) batches: Mutex<HashMap<String, Arc<Value>>>,
    batch_seq: AtomicU64,
    /// `uri -> (fingerprint, revision)` of run snapshots.
    revisions: Mutex<HashMap<String, (u64, u64)>>,
    /// Legacy (`resources/subscribe`) watchers by URI.
    legacy_subs: Mutex<HashMap<String, AbortHandle>>,
    pub(crate) local_gpus: Arc<OnceLock<Vec<GpuInfo>>>,
}

impl Inner {
    /// Runs a store call on the blocking pool (stores read files, render images or make HTTP requests).
    pub async fn blocking<T, F>(&self, f: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&dyn Store) -> Result<T, Error> + Send + 'static,
    {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || f(&*store)).await.map_err(tools::join)?
    }

    pub(crate) fn next_batch_id(&self) -> String {
        format!("pb_{}", self.batch_seq.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Records the fingerprint of a resource snapshot and returns its revision (bumped on every change).
    pub(crate) fn revision(&self, uri: &str, fingerprint: u64) -> u64 {
        let mut map = self.revisions.lock().unwrap();
        let entry = map.entry(uri.to_string()).or_insert((fingerprint, 1));
        if entry.0 != fingerprint {
            *entry = (fingerprint, entry.1 + 1);
        }
        entry.1
    }

    /// The fingerprint of a subscribable resource now (`None` once it no longer exists).
    async fn fingerprint(&self, uri: &FoloUri) -> Option<u64> {
        let key = uri.to_string();
        let res = match uri.clone() {
            FoloUri::Run(id) => self
                .blocking(move |s| resources::run_snapshot(s, &id))
                .await
                .map(|snap| hash(&snap.to_string())),
            FoloUri::Metrics(id) => {
                self.blocking(move |s| {
                    let run = s.get_run(&id)?;
                    let m = s.run_metrics(&id, u64::MAX)?;
                    Ok(hash(&(m.epochs.len(), format!("{:?}", run.status))))
                })
                .await
            }
            _ => return None,
        };
        let fp = res.ok()?;
        if matches!(uri, FoloUri::Run(_)) {
            self.revision(&key, fp);
        }
        Some(fp)
    }

    /// Calls `notify` (at most once per [`WATCH_INTERVAL`]) whenever one of `uris` changes, until it returns
    /// false. A resource that disappears is reported once and then dropped.
    async fn watch<F, Fut>(self: Arc<Self>, uris: Vec<FoloUri>, mut notify: F)
    where
        F: FnMut(String) -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let mut last: Vec<Option<Option<u64>>> = vec![None; uris.len()];
        let mut tick = tokio::time::interval(WATCH_INTERVAL);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            for (i, uri) in uris.iter().enumerate() {
                if last[i] == Some(None) {
                    continue;
                }
                let fp = self.fingerprint(uri).await;
                let changed = last[i].is_some_and(|prev| prev != fp);
                last[i] = Some(fp);
                if changed && !notify(uri.to_string()).await {
                    return;
                }
            }
            if last.iter().all(|l| *l == Some(None)) {
                return;
            }
        }
    }
}

fn hash<T: Hash + ?Sized>(v: &T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// GPUs on this machine, through folo-gpu (for the real store, which does not enumerate them yet).
pub fn enumerate_gpus() -> Vec<GpuInfo> {
    let options = folo_gpu::GpuOptions::from_env();
    folo_gpu::GpuContext::enumerate(&options)
        .into_iter()
        .filter_map(|a| {
            let backend = match a.backend.to_ascii_lowercase().as_str() {
                "vulkan" => GpuBackend::Vulkan,
                "dx12" => GpuBackend::Dx12,
                "metal" => GpuBackend::Metal,
                _ => return None,
            };
            Some(GpuInfo { ordinal: a.index as u32, name: a.name, backend, memory_bytes: 0, util: None })
        })
        .collect()
}

/// The FOLO MCP server. Cheap to clone.
#[derive(Clone)]
pub struct FoloMcp {
    pub(crate) inner: Arc<Inner>,
}

impl FoloMcp {
    /// A server over `store` (`proxy` is the same store when it forwards to `folo serve`).
    pub fn new(store: Arc<dyn Store>, proxy: Option<Arc<HttpStore>>, opts: Options) -> FoloMcp {
        let mock = proxy.is_none() && store.is_mock();
        let paths = PathPolicy { data_dir: opts.data_dir.clone(), confine: opts.confine };
        FoloMcp {
            inner: Arc::new(Inner {
                store,
                proxy,
                mock,
                opts,
                confinement: Confinement::new(paths.clone()),
                paths,
                created_runs: Mutex::new(Vec::new()),
                batches: Mutex::new(HashMap::new()),
                batch_seq: AtomicU64::new(0),
                revisions: Mutex::new(HashMap::new()),
                legacy_subs: Mutex::new(HashMap::new()),
                local_gpus: Arc::new(OnceLock::new()),
            }),
        }
    }

    /// Stops runs this session started that are still active (client disconnect or shutdown signal).
    pub async fn stop_session_runs(&self) {
        let ids = std::mem::take(&mut *self.inner.created_runs.lock().unwrap());
        if ids.is_empty() {
            return;
        }
        let _ = self
            .inner
            .blocking(move |s| {
                for id in ids {
                    if s.get_run(&id).is_ok_and(|r| r.status.is_active()) {
                        match s.stop_run(&id) {
                            Ok(_) => tracing::info!(run = %id, "stopped run on disconnect"),
                            Err(e) => tracing::warn!(run = %id, "could not stop run: {e}"),
                        }
                    }
                }
                Ok(())
            })
            .await;
        for (_, h) in self.inner.legacy_subs.lock().unwrap().drain() {
            h.abort();
        }
    }
}

/// An `isError` result. Messages and hints are cut so error results stay small too (store and server messages
/// may echo arguments).
fn tool_error(e: &Error) -> CallToolResult {
    let mut body = e.body.clone();
    body.message = tools::short(&body.message, 1000);
    body.hint = body.hint.map(|h| tools::short(&h, 500));
    let text = serde_json::to_string(&body).expect("errors serialize");
    CallToolResult::error(vec![ContentBlock::text(text)])
}

/// Maps a store error on a resource read: missing things are the spec's resource-not-found error.
fn resource_error(uri: &str, e: Error) -> McpError {
    let e = crate::confine::map_denied(e);
    let data = Some(json!({ "uri": uri, "error": e.body }));
    if e.status == http::StatusCode::NOT_FOUND {
        McpError::resource_not_found(format!("resource not found: {uri} ({})", e.body.message), data)
    } else if e.status.is_client_error() {
        McpError::invalid_params(format!("reading {uri}: {}", e.body.message), data)
    } else {
        McpError::internal_error(format!("reading {uri}: {}", e.body.message), data)
    }
}

fn not_found(uri: &str) -> McpError {
    McpError::resource_not_found(format!("resource not found: {uri}"), Some(json!({ "uri": uri })))
}

impl ServerHandler for FoloMcp {
    #[allow(deprecated)] // `enable_logging`, see `set_level`
    fn get_info(&self) -> ServerConfig {
        let mut caps = ServerCapabilities::builder()
            .enable_logging()
            .enable_prompts()
            .enable_resources()
            .enable_resources_subscribe()
            .enable_tools()
            .build();
        let mut tools = ToolsCapability::default();
        tools.list_changed = Some(false);
        caps.tools = Some(tools);
        ServerConfig::new(caps)
            .with_protocol_version(ProtocolVersion::V_2025_11_25)
            .with_server_info(Implementation::new("folo", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "FOLO: a fast YOLO-style object detector. Call folo_info first, then list_datasets / \
                 create_synthetic_dataset / register_dataset, train, run_status (or subscribe to \
                 folo://runs/{run_id}), val and predict. Errors are isError results whose text is \
                 {code, message, hint}.",
            )
    }

    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(&VERSIONS)
    }

    #[allow(deprecated)] // logging is in both revisions FOLO serves; rmcp marks it for a future removal (SEP-2577)
    async fn set_level(
        &self,
        _request: rmcp::model::SetLevelRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        // Accepted for clients that set it; the server only logs to stderr and the run logs.
        Ok(())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(
            tools::listed(self.inner.opts.read_only).map(|t| t.tool()).collect(),
        ))
    }

    fn get_tool(&self, name: &str) -> Option<rmcp::model::Tool> {
        tools::listed(self.inner.opts.read_only).find(|t| t.name == name).map(|t| t.tool())
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let Some(def) = tools::find(&request.name) else {
            return Err(McpError::invalid_params(format!("unknown tool: {}", request.name), None));
        };
        let mut progress = Progress::new(context.peer.clone(), context.meta.get_progress_token());
        let args = request.arguments.unwrap_or_default();
        let result = match self.inner.call(def, args, &mut progress).await {
            Ok(o) => {
                let mut content = vec![ContentBlock::text(o.summary)];
                content.extend(o.links);
                let mut r = CallToolResult::success(content);
                r.structured_content = Some(o.value);
                r
            }
            Err(e) => {
                tracing::debug!(tool = def.name, code = %e.body.code, "tool error: {}", e.body.message);
                tool_error(&e)
            }
        };
        Ok(result.into())
    }

    async fn list_resources(
        &self,
        request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        let cursor = request.and_then(|r| r.cursor);
        let all = self
            .inner
            .blocking({
                let c = self.inner.confinement.clone();
                move |s| resources::list(s, &|id| c.dataset(s, id).is_ok())
            })
            .await
            .map_err(|e| McpError::internal_error(e.body.message, None))?;
        let offset = match cursor.as_deref() {
            None => 0,
            Some(c) => c
                .strip_prefix("o")
                .and_then(|n| n.parse::<usize>().ok())
                .ok_or_else(|| McpError::invalid_params(format!("invalid cursor `{c}`"), None))?,
        };
        let end = (offset + resources::PAGE).min(all.len());
        let mut res = ListResourcesResult::with_all_items(all.get(offset..end).unwrap_or_default().to_vec());
        res.next_cursor = (end < all.len()).then(|| format!("o{end}"));
        Ok(res)
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult::with_all_items(resources::templates()))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let uri = request.uri;
        let parsed = FoloUri::parse(&uri).ok_or_else(|| not_found(&uri))?;
        let json_contents =
            |v: &Value| ResourceContents::text(v.to_string(), uri.clone()).with_mime_type("application/json");
        let c = self.inner.confinement.clone();
        let contents = match parsed {
            FoloUri::Dataset(id) => {
                let d = self
                    .inner
                    .blocking(move |s| {
                        c.dataset(s, &id)?;
                        s.get_dataset(&id)
                    })
                    .await
                    .map_err(|e| resource_error(&uri, e))?;
                json_contents(&serde_json::to_value(d).unwrap_or_default())
            }
            FoloUri::Image(id, image_id) => {
                let img = self
                    .inner
                    .blocking(move |s| {
                        c.dataset(s, &id)?;
                        s.image(&id, &image_id, None)
                    })
                    .await
                    .map_err(|e| resource_error(&uri, e))?;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&img.bytes);
                ResourceContents::blob(b64, uri.clone()).with_mime_type(img.content_type)
            }
            FoloUri::Run(id) => {
                let snap = self
                    .inner
                    .blocking(move |s| resources::run_snapshot(s, &id))
                    .await
                    .map_err(|e| resource_error(&uri, e))?;
                let revision = self.inner.revision(&uri, hash(&snap.to_string()));
                let mut v = snap;
                v["revision"] = json!(revision);
                json_contents(&v)
            }
            FoloUri::Metrics(id) => {
                let m = self
                    .inner
                    .blocking(move |s| s.run_metrics(&id, 1))
                    .await
                    .map_err(|e| resource_error(&uri, e))?;
                json_contents(&serde_json::to_value(m).unwrap_or_default())
            }
            FoloUri::Samples(id, epoch) => {
                let jpeg = self
                    .inner
                    .blocking(move |s| {
                        c.dataset(s, &s.get_run(&id)?.config.dataset_id)?;
                        crate::render::sample_grid(s, &id, epoch)
                    })
                    .await
                    .map_err(|e| resource_error(&uri, e))?;
                let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
                ResourceContents::blob(b64, uri.clone()).with_mime_type("image/jpeg")
            }
            FoloUri::Predictions(batch) => {
                let v =
                    self.inner.batches.lock().unwrap().get(&batch).cloned().ok_or_else(|| not_found(&uri))?;
                json_contents(&v)
            }
            FoloUri::Doc(name) => {
                let text = resources::doc(&name).ok_or_else(|| not_found(&uri))?;
                ResourceContents::text(text, uri.clone()).with_mime_type("text/markdown")
            }
        };
        Ok(ReadResourceResult::new(vec![contents]).into())
    }

    // 2025-11-25: resources/subscribe + notifications/resources/updated.
    async fn subscribe(
        &self,
        request: SubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        let uri = request.uri;
        let parsed = FoloUri::parse(&uri).filter(FoloUri::subscribable).ok_or_else(|| {
            McpError::invalid_params(
                format!("{uri} is not subscribable (folo://runs/{{id}} and its /metrics are)"),
                None,
            )
        })?;
        if self.inner.fingerprint(&parsed).await.is_none() {
            return Err(not_found(&uri));
        }
        let peer = context.peer.clone();
        let inner = self.inner.clone();
        let task =
            tokio::spawn(inner.watch(vec![parsed], move |uri| {
                let peer = peer.clone();
                async move {
                    peer.notify_resource_updated(ResourceUpdatedNotificationParam::new(uri)).await.is_ok()
                }
            }));
        if let Some(old) = self.inner.legacy_subs.lock().unwrap().insert(uri, task.abort_handle()) {
            old.abort();
        }
        Ok(())
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), McpError> {
        if let Some(h) = self.inner.legacy_subs.lock().unwrap().remove(&request.uri) {
            h.abort();
        }
        Ok(())
    }

    // 2026-07-28: subscriptions/listen, acknowledged with the accepted filter, updates tagged with its id.
    fn accepted_subscription_filter(&self, requested: &SubscriptionFilter) -> Option<SubscriptionFilter> {
        let uris: Vec<String> = requested
            .resource_subscriptions
            .iter()
            .flatten()
            .filter(|u| FoloUri::parse(u).is_some_and(|p| p.subscribable()))
            .cloned()
            .collect();
        Some(SubscriptionFilter::builder().resource_subscriptions(uris).build())
    }

    async fn listen(&self, context: SubscriptionContext) -> Result<(), McpError> {
        let uris: Vec<FoloUri> = context
            .accepted()
            .resource_subscriptions
            .iter()
            .flatten()
            .filter_map(|u| FoloUri::parse(u))
            .collect();
        let sink = context.sink().clone();
        let watch = self.inner.clone().watch(uris, move |uri| {
            let sink = sink.clone();
            async move { sink.notify_resource_updated(uri).await.is_ok() }
        });
        tokio::select! {
            _ = watch => context.cancelled().await,
            _ = context.cancelled() => {}
        }
        Ok(())
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult::with_all_items(prompts::list(self.inner.opts.read_only)))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, McpError> {
        prompts::get(&self.inner, request).await.map(Into::into)
    }
}
