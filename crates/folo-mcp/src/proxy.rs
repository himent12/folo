//! Proxy mode (`folo mcp --server http://127.0.0.1:7878`): a [`Store`] that forwards to a running `folo serve`
//! over docs/API.md, so Studio and the agent share datasets and runs. Every tool goes through the same `Store`
//! methods as in embedded mode; only the transport differs: [`folo_client::Client`] (blocking, `http://` or
//! `https://`, the server's `--token` when it has one). The MCP layer runs the calls on tokio's blocking pool.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use folo_api::{
    CloudConnection, CloudConnectionList, CloudInstance, CloudInstanceList, CloudOfferList, CloudProvider,
    CloudProviderList, ConnectCloud, Dataset, DatasetList, ExportResult, ExportSpec,
    FEATURE_EXPORT_ONNX_SWITCHES, GpuInfo, Health, ImagePage, IssuePage, Model, ModelEval, ModelList,
    PredictRequest, PredictResponse, QueueAction, QueueRequest, RegisterDataset, Run, RunEstimate, RunList,
    RunLogs, RunMetrics, RunSamples, SyntheticDataset, SystemInfo, SystemResponse, ThroughputTable,
    TrainConfig, ValReport, ValRequest,
};
use folo_client::{Client, encode};
use folo_server::store::{
    DynamicAxes, FEATURE_EXPORT_DYNAMIC_AXES, FEATURE_VAL_DIAGNOSTICS, ImageData, ImageQuery, IssueQuery,
    LogsQuery, MetricsQuery, PredictInput, PredictSource, QueueMove, Store, Subscription, ValDiagConfig,
    ValDiagnostics, check_export_axes, check_val_diag_config,
};
use http::StatusCode;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::errors::{Error, not_implemented};

/// Cached remote capability probe shared across an [`HttpStore`] and its [`HttpStore::until`] clones.
#[derive(Default)]
struct ProbeState {
    /// Bumped by [`HttpStore::clear_probe_cache`] so an older in-flight probe never commits stale data.
    generation: u64,
    mock: Option<bool>,
    features: Option<Vec<String>>,
    /// Cached `data_dir` reported by `GET /api/system` (`Some(None)` means the server explicitly omitted it).
    data_dir: Option<Option<String>>,
    /// True while one caller is probing `/api/health` (and `/api/system` for legacy servers) for `features`.
    features_in_flight: bool,
    #[cfg(test)]
    waiters: usize,
}

#[derive(Default)]
struct SharedProbe {
    state: Mutex<ProbeState>,
    ready: Condvar,
}

/// Clears `features_in_flight` and wakes waiters even on error or panic.
struct InFlightGuard<'a>(&'a SharedProbe);

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        s.features_in_flight = false;
        self.0.ready.notify_all();
    }
}

/// A [`Store`] backed by a `folo serve` HTTP API.
pub struct HttpStore {
    client: Client,
    deadline: Option<Instant>,
    probe: Arc<SharedProbe>,
}

fn action(a: QueueAction) -> QueueRequest {
    QueueRequest { action: Some(a), position: None }
}

/// A transport error as a tool error: the server's code, status and hint; a server that does not answer is `502
/// server_unreachable`.
pub(crate) fn store_error(e: folo_client::Error) -> Error {
    match e.status {
        Some(status) => Error { status, body: e.error },
        None => Error::new(StatusCode::BAD_GATEWAY, "server_unreachable", e.error.message)
            .hint("is `folo serve` running there? start it, or run `folo mcp` without --server"),
    }
}

impl HttpStore {
    /// A store forwarding to `base` (e.g. `http://127.0.0.1:7878`), sending `token` to a server started with
    /// `--token`.
    pub fn new(base: &str, token: Option<&str>) -> HttpStore {
        let client = Client::new(base).token(token.unwrap_or_default());
        HttpStore { client, deadline: None, probe: Arc::new(SharedProbe::default()) }
    }

    /// [`HttpStore::new`] optionally trusting the validated CA bundle `tls_ca` (for self-signed or private-CA
    /// `https://` servers).
    pub fn with_tls_ca(base: &str, token: Option<&str>, tls_ca: Option<&folo_client::TlsCa>) -> HttpStore {
        let client = match tls_ca {
            Some(ca) => Client::with_tls_ca(base, Duration::from_secs(5), Duration::from_secs(600), ca),
            None => Client::new(base),
        };
        HttpStore {
            client: client.token(token.unwrap_or_default()),
            deadline: None,
            probe: Arc::new(SharedProbe::default()),
        }
    }

    /// The server's base URL.
    pub fn base(&self) -> &str {
        self.client.base()
    }

    /// This store with every request ending by `deadline` (`bench_entry`'s wait promises an answer by then).
    /// Shares the cached `/api/health` and feature probe with the parent store.
    pub fn until(&self, deadline: Instant) -> HttpStore {
        let effective = self.deadline.map_or(deadline, |d| d.min(deadline));
        HttpStore {
            client: self.client.until(effective),
            deadline: Some(effective),
            probe: Arc::clone(&self.probe),
        }
    }

    /// The client (the labeling tools call `/api/label` through it, `crate::label`).
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Clears the cached `/api/health` and feature probe so the next capability check queries the server again.
    /// Any probe already in flight will not repopulate the cache when it finishes.
    pub fn clear_probe_cache(&self) {
        let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
        s.generation = s.generation.wrapping_add(1);
        s.mock = None;
        s.features = None;
        s.data_dir = None;
        self.probe.ready.notify_all();
    }

    /// This store, taking `features` as the server's (tests: a server of an earlier version).
    #[cfg(test)]
    pub(crate) fn assume_features(self, features: Vec<String>) -> HttpStore {
        self.probe.state.lock().unwrap_or_else(|e| e.into_inner()).features = Some(features);
        self
    }

    /// This store, taking `data_dir` as the server's `/api/system` `data_dir` (tests: remote proxy confinement).
    #[cfg(test)]
    pub(crate) fn assume_data_dir(self, data_dir: Option<String>) -> HttpStore {
        self.probe.state.lock().unwrap_or_else(|e| e.into_inner()).data_dir = Some(data_dir);
        self
    }

    /// The server's `data_dir` from `GET /api/system`, cached once on success (never caching transient errors).
    pub(crate) fn remote_data_dir(&self) -> Result<Option<String>, Error> {
        let start_gen = {
            let s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(dir) = &s.data_dir {
                return Ok(dir.clone());
            }
            s.generation
        };
        let dir = self.system()?.data_dir;
        let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.generation == start_gen {
            s.data_dir = Some(dir.clone());
        }
        Ok(dir)
    }

    /// Calls `GET /api/health` and updates the shared cache on success if `generation` has not been cleared
    /// (never caching transient errors).
    fn probe_health(&self) -> Result<Health, Error> {
        let start_gen = self.probe.state.lock().unwrap_or_else(|e| e.into_inner()).generation;
        let health: Health = self.get("/health")?;
        let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.generation == start_gen {
            s.mock = Some(health.mock);
            if !health.features.is_empty() && s.features.is_none() {
                s.features = Some(health.features.clone());
            }
        }
        Ok(health)
    }

    /// Whether the server lists `feature` (cached once per session and shared across [`Self::until`] clones,
    /// with single-flight coordination so concurrent callers do not duplicate `/api/health` or `/api/system`
    /// probes). `/api/health` answers without a hardware probe; only a server that predates its `features` is
    /// asked `/api/system`, whose CPU sample once stalled a labeling call for 20 s on a loaded machine.
    pub fn has_feature(&self, feature: &str) -> Result<bool, Error> {
        let start_gen = {
            let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(features) = &s.features {
                    return Ok(features.iter().any(|f| f == feature));
                }
                if !s.features_in_flight {
                    s.features_in_flight = true;
                    break s.generation;
                }
                if let Some(deadline) = self.deadline {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        return Err(Error::new(
                            StatusCode::BAD_GATEWAY,
                            "server_unreachable",
                            format!("timed out waiting for capability probe of {}", self.base()),
                        )
                        .hint(
                            "is `folo serve` running there? start it, or run `folo mcp` without --server",
                        ));
                    };
                    #[cfg(test)]
                    {
                        s.waiters += 1;
                    }
                    let (guard, _) =
                        self.probe.ready.wait_timeout(s, remaining).unwrap_or_else(|e| e.into_inner());
                    s = guard;
                    #[cfg(test)]
                    {
                        s.waiters -= 1;
                    }
                } else {
                    #[cfg(test)]
                    {
                        s.waiters += 1;
                    }
                    s = self.probe.ready.wait(s).unwrap_or_else(|e| e.into_inner());
                    #[cfg(test)]
                    {
                        s.waiters -= 1;
                    }
                }
            }
        };
        let _in_flight = InFlightGuard(&self.probe);
        let health: Health = self.get("/health")?;
        if !health.features.is_empty() {
            let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.generation == start_gen {
                s.mock = Some(health.mock);
                s.features.get_or_insert_with(|| health.features.clone());
            }
            return Ok(health.features.iter().any(|f| f == feature));
        }
        {
            let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.generation == start_gen {
                s.mock = Some(health.mock);
            }
        }
        let features = self.system()?.features;
        let mut s = self.probe.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.generation == start_gen {
            s.features.get_or_insert_with(|| features.clone());
        }
        Ok(features.iter().any(|f| f == feature))
    }

    /// `GET /api/system` of the server.
    pub fn system(&self) -> Result<SystemResponse, Error> {
        self.get("/system")
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        self.client.get(path).map_err(store_error)
    }

    fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T, Error> {
        self.client.post(path, body).map_err(store_error)
    }

    fn delete(&self, path: &str) -> Result<(), Error> {
        self.client.delete(path).map_err(store_error)
    }
}

fn static_content_type(ct: &str) -> &'static str {
    match ct.split(';').next().unwrap_or("").trim() {
        "image/png" => "image/png",
        "image/bmp" => "image/bmp",
        "image/webp" => "image/webp",
        "image/jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
}

/// Picks a multipart boundary that does not appear anywhere in `fields` or `image`.
fn choose_boundary(fields: &[(&str, String)], image: &[u8]) -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let nanos =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0u128, |d| d.as_nanos());
    loop {
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let candidate = format!("folo-mcp-{nanos:x}-{seq:x}");
        let needle = candidate.as_bytes();
        let clashes = image.windows(needle.len()).any(|w| w == needle)
            || fields.iter().any(|(_, v)| v.contains(&candidate));
        if !clashes {
            return candidate;
        }
    }
}

/// A `multipart/form-data` body for `POST /api/predict`.
fn multipart(fields: &[(&str, String)], image: &[u8]) -> Result<(String, Vec<u8>), Error> {
    for (name, value) in fields {
        if value.contains(['\r', '\n', '\0']) {
            return Err(crate::errors::invalid_arguments(format!(
                "{name} must not contain line breaks or NUL bytes"
            )));
        }
    }
    let boundary = choose_boundary(fields, image);
    let mut body = Vec::with_capacity(image.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"image\"\r\nContent-Type: \
             application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(image);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok((format!("multipart/form-data; boundary={boundary}"), body))
}

/// Rejects an empty or dot-only (`"."` / `".."`) URL path segment with the same `<kind>_not_found` error that
/// an embedded store returns, instead of interpolating a malformed path that hits `404 route_not_found` or
/// resolves to the parent collection route.
pub(crate) fn check_segment(kind: &str, id: &str) -> Result<(), Error> {
    if id.is_empty() || matches!(id, "." | "..") {
        return Err(Error::not_found(kind, id));
    }
    Ok(())
}

impl Store for HttpStore {
    fn is_mock(&self) -> bool {
        if let Some(mock) = self.probe.state.lock().unwrap_or_else(|e| e.into_inner()).mock {
            return mock;
        }
        self.probe_health().map(|h| h.mock).unwrap_or(false)
    }

    fn gpus(&self) -> Vec<GpuInfo> {
        self.get::<SystemInfo>("/system").map(|s| s.gpus).unwrap_or_default()
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>, Error> {
        Ok(self.get::<DatasetList>("/datasets")?.datasets)
    }

    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset, Error> {
        self.post("/datasets", &req)
    }

    fn register_dataset_tracked(&self, req: RegisterDataset) -> Result<(Dataset, Option<String>), Error> {
        let (headers, ds) = self.client.post_with_headers("/datasets", &req).map_err(store_error)?;
        // Only servers that emit `x-folo-dataset-rollback` both confirm ownership of this exact registration and
        // support `DELETE /api/datasets/{id}?rollback=<token>`; older servers omit it so callers never send an
        // unsafe unconditional rollback delete.
        let token = headers
            .get("x-folo-dataset-rollback")
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(str::to_string);
        Ok((ds, token))
    }

    fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset, Error> {
        self.post("/datasets/synthetic", &req)
    }

    fn get_dataset(&self, id: &str) -> Result<Dataset, Error> {
        check_segment("dataset", id)?;
        self.get(&format!("/datasets/{}", encode(id)))
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage, Error> {
        check_segment("dataset", id)?;
        let mut path =
            format!("/datasets/{}/images?offset={}&limit={}", encode(id), q.offset, q.limit.min(1000));
        if let Some(s) = q.split {
            path.push_str(&format!("&split={}", s.as_str()));
        }
        if let Some(c) = q.class {
            path.push_str(&format!("&class={c}"));
        }
        self.get(&path)
    }

    fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<IssuePage, Error> {
        check_segment("dataset", id)?;
        let mut path =
            format!("/datasets/{}/issues?offset={}&limit={}", encode(id), q.offset, q.limit.min(1000));
        if let Some(k) = &q.kind {
            path.push_str(&format!("&kind={}", encode(k)));
        }
        if let Some(s) = q.severity {
            path.push_str(&format!("&severity={}", s.as_str()));
        }
        self.get(&path)
    }

    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData, Error> {
        check_segment("dataset", id)?;
        check_segment("image", image_id)?;
        let mut path = format!("/datasets/{}/images/{}", encode(id), encode(image_id));
        if let Some(t) = thumb {
            path.push_str(&format!("?thumb={t}"));
        }
        let (headers, bytes) = self.client.get_bytes(&path).map_err(store_error)?;
        let ct = headers.get(http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
        Ok(ImageData { content_type: static_content_type(ct), bytes, immutable: false })
    }

    fn delete_dataset(&self, id: &str) -> Result<(), Error> {
        check_segment("dataset", id)?;
        self.delete(&format!("/datasets/{}", encode(id)))
    }

    fn delete_dataset_if_unused(&self, id: &str, token: &str) -> Result<(), Error> {
        check_segment("dataset", id)?;
        self.delete(&format!("/datasets/{}?rollback={}", encode(id), encode(token)))
    }

    fn delete_dataset_files(&self, id: &str) -> Result<(), Error> {
        check_segment("dataset", id)?;
        self.delete(&format!("/datasets/{}?files=true", encode(id)))
    }

    fn list_models(&self) -> Result<Vec<Model>, Error> {
        Ok(self.get::<ModelList>("/models")?.models)
    }

    fn get_model(&self, id: &str) -> Result<Model, Error> {
        check_segment("model", id)?;
        self.get(&format!("/models/{}", encode(id)))
    }

    fn model_eval(&self, id: &str) -> Result<ModelEval, Error> {
        check_segment("model", id)?;
        self.get(&format!("/models/{}/eval", encode(id)))
    }

    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult, Error> {
        folo_server::store::check_export(&spec)?;
        check_segment("model", id)?;
        // An older server accepts the switches and ignores them, which would export the defaults silently.
        if spec.onnx.any() && !self.has_feature(FEATURE_EXPORT_ONNX_SWITCHES)? {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for the ONNX switches (decode, dynamic, half)",
                    self.base()
                ),
            )
            .hint("update folo serve, or export without the switches and use `folo export --format onnx`"));
        }
        self.post(&format!("/models/{}/export", encode(id)), &spec)
    }

    fn export_model_with_axes(
        &self,
        id: &str,
        mut spec: ExportSpec,
        axes: DynamicAxes,
    ) -> Result<ExportResult, Error> {
        check_export_axes(&spec, axes)?;
        check_segment("model", id)?;
        if axes.batch == axes.hw {
            spec.onnx.dynamic = axes.batch;
            return self.export_model(id, spec);
        }
        if !self.has_feature(FEATURE_EXPORT_DYNAMIC_AXES)? {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for independent ONNX dynamic axes (`dynamic_batch` / `dynamic_hw`)",
                    self.base()
                ),
            )
            .hint("update folo serve, or export with `dynamic: true` / `folo export --format onnx`"));
        }
        if (spec.onnx.decode || spec.onnx.half) && !self.has_feature(FEATURE_EXPORT_ONNX_SWITCHES)? {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for the ONNX switches (decode, dynamic, half)",
                    self.base()
                ),
            )
            .hint("update folo serve, or export without the switches and use `folo export --format onnx`"));
        }
        let mut body = serde_json::to_value(spec).map_err(|e| Error::internal(e.to_string()))?;
        body["dynamic_batch"] = serde_json::json!(axes.batch);
        body["dynamic_hw"] = serde_json::json!(axes.hw);
        self.post(&format!("/models/{}/export", encode(id)), &body)
    }

    fn list_runs(&self) -> Result<Vec<Run>, Error> {
        Ok(self.get::<RunList>("/runs")?.runs)
    }

    fn create_run(&self, config: TrainConfig) -> Result<Run, Error> {
        self.post("/runs", &config)
    }

    fn resume_run(
        &self,
        id: &str,
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Run, Error> {
        let mut body = fields.clone();
        body.insert("resume_run_id".into(), id.into());
        self.post("/runs", &serde_json::Value::Object(body))
    }

    fn get_run(&self, id: &str) -> Result<Run, Error> {
        check_segment("run", id)?;
        self.get(&format!("/runs/{}", encode(id)))
    }

    fn stop_run(&self, id: &str) -> Result<Run, Error> {
        check_segment("run", id)?;
        self.client.post_empty(&format!("/runs/{}/stop", encode(id))).map_err(store_error)
    }

    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run, Error> {
        check_segment("run", id)?;
        let body = match mv {
            QueueMove::To(p) => QueueRequest { action: None, position: Some(p) },
            QueueMove::Top => action(QueueAction::Top),
            QueueMove::Up => action(QueueAction::Up),
            QueueMove::Down => action(QueueAction::Down),
            QueueMove::Bottom => action(QueueAction::Bottom),
            QueueMove::Start => action(QueueAction::Start),
        };
        self.post(&format!("/runs/{}/queue", encode(id)), &body)
    }

    fn delete_run(&self, id: &str) -> Result<(), Error> {
        check_segment("run", id)?;
        self.delete(&format!("/runs/{}", encode(id)))
    }

    fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics, Error> {
        check_segment("run", id)?;
        let mut path = format!("/runs/{}/metrics?every={}", encode(id), q.every.max(1));
        if let Some(s) = q.after_step {
            path.push_str(&format!("&after_step={s}"));
        }
        if let Some(e) = q.after_epoch {
            path.push_str(&format!("&after_epoch={e}"));
        }
        self.get(&path)
    }

    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples, Error> {
        check_segment("run", id)?;
        let q = epoch.map_or(String::new(), |e| format!("?epoch={e}"));
        self.get(&format!("/runs/{}/samples{q}", encode(id)))
    }

    fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs, Error> {
        check_segment("run", id)?;
        let mut path = format!("/runs/{}/logs?limit={}", encode(id), q.limit.clamp(1, 5000));
        if let Some(a) = q.after {
            path.push_str(&format!("&after={a}"));
        }
        self.get(&path)
    }

    fn subscribe(&self, _id: &str) -> Result<Subscription, Error> {
        // The MCP layer polls runs for resource updates instead of following SSE.
        Err(not_implemented(
            "run event streams through the MCP proxy",
            "subscribe to the run resource instead",
        ))
    }

    fn predict(&self, input: PredictInput) -> Result<PredictResponse, Error> {
        let has_opts = input.options.tile.is_some() || input.options.tta.is_some();
        if has_opts && !self.has_feature(folo_server::store::FEATURE_PREDICT_OPTIONS)? {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for SAHI tiling (`tile`) or TTA (`tta`)",
                    self.base()
                ),
            )
            .hint("update folo serve, or run without `tile` / `tta`"));
        }
        match input.source {
            PredictSource::Dataset { dataset_id, image_id } => {
                let mut body = serde_json::to_value(&PredictRequest {
                    model_id: input.model_id,
                    dataset_id,
                    image_id,
                    conf: input.conf,
                    iou: input.iou,
                    max_det: input.max_det,
                })
                .map_err(|e| Error::internal(e.to_string()))?;
                if let Some(tile) = &input.options.tile {
                    body["tile"] = serde_json::to_value(tile).map_err(|e| Error::internal(e.to_string()))?;
                }
                if let Some(tta) = &input.options.tta {
                    body["tta"] = serde_json::to_value(tta).map_err(|e| Error::internal(e.to_string()))?;
                }
                self.post("/predict", &body)
            }
            PredictSource::Upload(bytes) => {
                let mut fields = vec![
                    ("model_id", input.model_id),
                    ("conf", input.conf.to_string()),
                    ("iou", input.iou.to_string()),
                    ("max_det", input.max_det.to_string()),
                ];
                if let Some(tile) = &input.options.tile {
                    fields.push((
                        "tile",
                        serde_json::to_string(tile).map_err(|e| Error::internal(e.to_string()))?,
                    ));
                }
                if let Some(tta) = &input.options.tta {
                    fields.push((
                        "tta",
                        serde_json::to_string(tta).map_err(|e| Error::internal(e.to_string()))?,
                    ));
                }
                let (ct, body) = multipart(&fields, &bytes)?;
                self.client.send("/predict", &ct, &body).map_err(store_error)
            }
        }
    }

    fn val(&self, req: ValRequest) -> Result<ValReport, Error> {
        self.post("/val", &req)
    }

    fn val_with_diagnostics(
        &self,
        req: ValRequest,
        diag: Option<ValDiagConfig>,
    ) -> Result<(ValReport, Option<ValDiagnostics>), Error> {
        let Some(cfg) = diag else {
            return self.val(req).map(|r| (r, None));
        };
        check_val_diag_config(&cfg)?;
        if !self.has_feature(FEATURE_VAL_DIAGNOSTICS)? {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for validation error diagnostics (`diagnostics` / `worst`)",
                    self.base()
                ),
            )
            .hint("update folo serve, or run `val` without `diagnostics` / `worst`"));
        }
        let mut body = serde_json::to_value(req).map_err(|e| Error::internal(e.to_string()))?;
        body["diagnostics"] = serde_json::json!(true);
        body["worst"] = serde_json::json!(cfg.worst);
        body["diag_conf"] = serde_json::json!(cfg.conf);
        body["diag_tp_iou"] = serde_json::json!(cfg.tp_iou);
        body["diag_bg_iou"] = serde_json::json!(cfg.bg_iou);
        #[derive(serde::Deserialize)]
        struct WireVal {
            #[serde(flatten)]
            report: ValReport,
            #[serde(default)]
            diagnostics: Option<ValDiagnostics>,
        }
        let wire: WireVal = self.post("/val", &body)?;
        Ok((wire.report, wire.diagnostics))
    }

    fn throughput(&self) -> Result<ThroughputTable, Error> {
        self.get("/system/throughput")
    }

    fn estimate_run(&self, config: TrainConfig) -> Result<RunEstimate, Error> {
        self.post("/runs/estimate", &config)
    }

    fn cloud_providers(&self) -> Result<Vec<CloudProvider>, Error> {
        Ok(self.get::<CloudProviderList>("/cloud/providers")?.providers)
    }

    fn cloud_connections(&self) -> Result<Vec<CloudConnection>, Error> {
        Ok(self.get::<CloudConnectionList>("/cloud/connections")?.connections)
    }

    fn connect_cloud(&self, req: ConnectCloud) -> Result<CloudConnection, Error> {
        self.post("/cloud/connections", &req)
    }

    fn check_cloud(&self, id: &str) -> Result<CloudConnection, Error> {
        check_segment("connection", id)?;
        self.client.post_empty(&format!("/cloud/connections/{}/check", encode(id))).map_err(store_error)
    }

    fn disconnect_cloud(&self, id: &str) -> Result<(), Error> {
        check_segment("connection", id)?;
        self.delete(&format!("/cloud/connections/{}", encode(id)))
    }

    fn cloud_offers(&self, id: &str, refresh: bool) -> Result<CloudOfferList, Error> {
        check_segment("connection", id)?;
        self.get(&format!("/cloud/connections/{}/offers?refresh={refresh}", encode(id)))
    }

    fn cloud_instances(&self) -> Result<Vec<CloudInstance>, Error> {
        Ok(self.get::<CloudInstanceList>("/cloud/instances")?.instances)
    }

    fn terminate_instance(&self, id: &str) -> Result<CloudInstance, Error> {
        check_segment("instance", id)?;
        self.client.delete_json(&format!("/cloud/instances/{}", encode(id))).map_err(store_error)
    }
}

#[cfg(test)]
#[path = "proxy_tests.rs"]
mod tests;
