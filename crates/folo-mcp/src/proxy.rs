//! Proxy mode (`folo mcp --server http://127.0.0.1:7878`): a [`Store`] that forwards to a running `folo serve`
//! over docs/API.md, so Studio and the agent share datasets and runs. Every tool goes through the same `Store`
//! methods as in embedded mode; only the transport differs: [`folo_client::Client`] (blocking, `http://` or
//! `https://`, the server's `--token` when it has one). The MCP layer runs the calls on tokio's blocking pool.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

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
    ImageData, ImageQuery, IssueQuery, LogsQuery, MetricsQuery, PredictInput, PredictSource, QueueMove,
    Store, Subscription,
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
        self.probe.ready.notify_all();
    }

    /// This store, taking `features` as the server's (tests: a server of an earlier version).
    #[cfg(test)]
    pub(crate) fn assume_features(self, features: Vec<String>) -> HttpStore {
        self.probe.state.lock().unwrap_or_else(|e| e.into_inner()).features = Some(features);
        self
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
        self.get(&format!("/datasets/{}", encode(id)))
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage, Error> {
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
        let mut path = format!("/datasets/{}/images/{}", encode(id), encode(image_id));
        if let Some(t) = thumb {
            path.push_str(&format!("?thumb={t}"));
        }
        let (headers, bytes) = self.client.get_bytes(&path).map_err(store_error)?;
        let ct = headers.get(http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
        Ok(ImageData { content_type: static_content_type(ct), bytes, immutable: false })
    }

    fn delete_dataset(&self, id: &str) -> Result<(), Error> {
        self.delete(&format!("/datasets/{}", encode(id)))
    }

    fn delete_dataset_if_unused(&self, id: &str, token: &str) -> Result<(), Error> {
        self.delete(&format!("/datasets/{}?rollback={}", encode(id), encode(token)))
    }

    fn delete_dataset_files(&self, id: &str) -> Result<(), Error> {
        self.delete(&format!("/datasets/{}?files=true", encode(id)))
    }

    fn list_models(&self) -> Result<Vec<Model>, Error> {
        Ok(self.get::<ModelList>("/models")?.models)
    }

    fn get_model(&self, id: &str) -> Result<Model, Error> {
        self.get(&format!("/models/{}", encode(id)))
    }

    fn model_eval(&self, id: &str) -> Result<ModelEval, Error> {
        self.get(&format!("/models/{}/eval", encode(id)))
    }

    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult, Error> {
        folo_server::store::check_export(&spec)?;
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
        self.get(&format!("/runs/{}", encode(id)))
    }

    fn stop_run(&self, id: &str) -> Result<Run, Error> {
        self.client.post_empty(&format!("/runs/{}/stop", encode(id))).map_err(store_error)
    }

    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run, Error> {
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
        self.delete(&format!("/runs/{}", encode(id)))
    }

    fn run_metrics(&self, id: &str, q: &MetricsQuery) -> Result<RunMetrics, Error> {
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
        let q = epoch.map_or(String::new(), |e| format!("?epoch={e}"));
        self.get(&format!("/runs/{}/samples{q}", encode(id)))
    }

    fn run_logs(&self, id: &str, q: &LogsQuery) -> Result<RunLogs, Error> {
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
        match input.source {
            PredictSource::Dataset { dataset_id, image_id } => self.post(
                "/predict",
                &PredictRequest {
                    model_id: input.model_id,
                    dataset_id,
                    image_id,
                    conf: input.conf,
                    iou: input.iou,
                    max_det: input.max_det,
                },
            ),
            PredictSource::Upload(bytes) => {
                let fields = [
                    ("model_id", input.model_id),
                    ("conf", input.conf.to_string()),
                    ("iou", input.iou.to_string()),
                    ("max_det", input.max_det.to_string()),
                ];
                let (ct, body) = multipart(&fields, &bytes)?;
                self.client.send("/predict", &ct, &body).map_err(store_error)
            }
        }
    }

    fn val(&self, req: ValRequest) -> Result<ValReport, Error> {
        self.post("/val", &req)
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
        self.client.post_empty(&format!("/cloud/connections/{}/check", encode(id))).map_err(store_error)
    }

    fn disconnect_cloud(&self, id: &str) -> Result<(), Error> {
        self.delete(&format!("/cloud/connections/{}", encode(id)))
    }

    fn cloud_offers(&self, id: &str, refresh: bool) -> Result<CloudOfferList, Error> {
        self.get(&format!("/cloud/connections/{}/offers?refresh={refresh}", encode(id)))
    }

    fn cloud_instances(&self) -> Result<Vec<CloudInstance>, Error> {
        Ok(self.get::<CloudInstanceList>("/cloud/instances")?.instances)
    }

    fn terminate_instance(&self, id: &str) -> Result<CloudInstance, Error> {
        self.client.delete_json(&format!("/cloud/instances/{}", encode(id))).map_err(store_error)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use folo_api::{ExportFormat, OnnxSwitches};

    use super::*;

    /// Loopback mock API server that drains request bodies (preventing TCP RST on close) and shuts down cleanly
    /// on drop.
    struct MockApi {
        addr: SocketAddr,
        base: String,
        log: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockApi {
        fn start(handler: impl Fn(&str, usize) -> (&'static str, String) + Send + 'static) -> MockApi {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let base = format!("http://{addr}");
            let log = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let recorded = Arc::clone(&log);
            let done = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                let mut idx = 0usize;
                for stream in listener.incoming() {
                    if done.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(mut s) = stream else { continue };
                    let Some(req) = read_request(&mut s) else { continue };
                    recorded.lock().unwrap().push(req.clone());
                    let (status, body) = handler(&req, idx);
                    idx += 1;
                    let _ = write!(
                        s,
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
                         close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = s.flush();
                }
            });
            MockApi { addr, base, log, stop, handle: Some(handle) }
        }

        fn requests(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    impl Drop for MockApi {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            let _ = TcpStream::connect(self.addr);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    /// Reads the HTTP request head, drains any `Content-Length` body so closing the socket never triggers a TCP
    /// `RST`, and returns `"METHOD /path"`.
    fn read_request(s: &mut TcpStream) -> Option<String> {
        let mut buf = Vec::new();
        let mut b = [0u8; 1];
        while !buf.ends_with(b"\r\n\r\n") {
            match s.read(&mut b) {
                Ok(1) => buf.push(b[0]),
                _ => break,
            }
        }
        if buf.is_empty() {
            return None;
        }
        let head = String::from_utf8_lossy(&buf);
        let content_length = head
            .lines()
            .skip(1)
            .find_map(|line| {
                let (k, v) = line.split_once(':')?;
                k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        if content_length > 0 {
            let mut body = vec![0u8; content_length];
            let _ = s.read_exact(&mut body);
        }
        let first = head.lines().next().unwrap_or("");
        let mut parts = first.split_whitespace();
        Some(format!("{} {}", parts.next().unwrap_or(""), parts.next().unwrap_or("")))
    }

    fn system_json(features: &[&str]) -> String {
        serde_json::to_string(&SystemResponse {
            system: SystemInfo {
                cpu: folo_api::CpuInfo { name: "test".into(), threads: 4, util: 0.1 },
                memory: folo_api::MemoryInfo { used_bytes: 512, total_bytes: 1024 },
                gpus: vec![],
            },
            data_dir: None,
            features: features.iter().map(|s| (*s).to_string()).collect(),
            capabilities: None,
            gpu_pool: None,
        })
        .unwrap()
    }

    #[test]
    fn unreachable_server_is_a_tool_error() {
        // A server that hangs up without an answer. (A closed port would do too, but on Windows a refused
        // loopback connection takes 2 s.)
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                drop(conn);
            }
        });
        let s = HttpStore::new(&url, Some("s3cret"));
        let e = s.list_datasets().unwrap_err();
        assert_eq!((e.status, e.body.code.as_str()), (StatusCode::BAD_GATEWAY, "server_unreachable"));
        assert!(e.body.hint.unwrap().contains("folo serve"));
        assert!(!e.body.message.contains("s3cret"));
    }

    #[test]
    fn feature_and_mock_probes_share_cache_across_until_and_export() {
        let server = MockApi::start(|req, _| match req {
            "GET /api/health" => (
                "200 OK",
                serde_json::json!({
                    "status": "ok",
                    "version": "0.1.0",
                    "mock": true,
                    "features": [FEATURE_EXPORT_ONNX_SWITCHES, folo_api::FEATURE_VISION_BENCH],
                })
                .to_string(),
            ),
            "POST /api/models/m1/export" => {
                ("200 OK", serde_json::json!({ "path": "/tmp/m1-f16.onnx", "bytes": 1024 }).to_string())
            }
            other => panic!("unexpected request: {other}"),
        });

        let store = HttpStore::new(&server.base, None);
        // `is_mock` probes `/api/health` once and populates both `mock` and `features`.
        assert!(store.is_mock());
        assert!(store.has_feature(FEATURE_EXPORT_ONNX_SWITCHES).unwrap());
        // `until` shares the cache with its parent store.
        let deadline = Instant::now() + Duration::from_secs(5);
        assert!(store.until(deadline).has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        // ONNX-switch export reuses the cached features without hitting `/api/system`.
        let exported = store
            .export_model(
                "m1",
                ExportSpec {
                    format: ExportFormat::Onnx,
                    onnx: OnnxSwitches { half: true, ..OnnxSwitches::default() },
                },
            )
            .unwrap();
        assert_eq!(exported.path, "/tmp/m1-f16.onnx");

        // Clearing the cache forces a fresh probe, and probing through `until` populates the parent cache too.
        store.clear_probe_cache();
        assert!(store.until(deadline).has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        assert!(store.is_mock());

        assert_eq!(
            server.requests(),
            vec!["GET /api/health", "POST /api/models/m1/export", "GET /api/health"]
        );
    }

    #[test]
    fn legacy_server_falls_back_to_system_once_and_caches_missing_export_feature() {
        let server = MockApi::start(|req, _| match req {
            "GET /api/health" => (
                "200 OK",
                serde_json::json!({ "status": "ok", "version": "0.1.0", "mock": false, "features": [] })
                    .to_string(),
            ),
            "GET /api/system" => ("200 OK", system_json(&[folo_api::FEATURE_VISION_BENCH])),
            other => panic!("unexpected request: {other}"),
        });

        let store = HttpStore::new(&server.base, None);
        assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        assert!(!store.is_mock());
        let err = store
            .export_model(
                "m1",
                ExportSpec {
                    format: ExportFormat::Onnx,
                    onnx: OnnxSwitches { decode: true, ..OnnxSwitches::default() },
                },
            )
            .unwrap_err();
        assert_eq!(err.status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/system"]);
    }

    #[test]
    fn transient_health_error_is_not_cached() {
        let server = MockApi::start(|req, idx| match (req, idx) {
            ("GET /api/health", 0) => (
                "503 Service Unavailable",
                serde_json::json!({ "error": { "code": "internal", "message": "starting up" } }).to_string(),
            ),
            ("GET /api/health", _) => (
                "200 OK",
                serde_json::json!({
                    "status": "ok",
                    "version": "0.1.0",
                    "mock": true,
                    "features": [folo_api::FEATURE_VISION_BENCH],
                })
                .to_string(),
            ),
            other => panic!("unexpected request: {other:?}"),
        });

        let store = HttpStore::new(&server.base, None);
        assert!(!store.is_mock(), "transient 503 returns false without poisoning the cache");
        assert!(store.is_mock(), "next probe succeeds once the server recovers");
        assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/health"]);
    }

    #[test]
    fn concurrent_legacy_feature_checks_probe_system_only_once() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let server = MockApi::start(move |req, _| match req {
            "GET /api/health" => (
                "200 OK",
                serde_json::json!({ "status": "ok", "version": "0.1.0", "mock": false, "features": [] })
                    .to_string(),
            ),
            "GET /api/system" => {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().unwrap().recv();
                ("200 OK", system_json(&[folo_api::FEATURE_VISION_BENCH]))
            }
            other => panic!("unexpected request: {other}"),
        });

        let store = Arc::new(HttpStore::new(&server.base, None));
        let parent = Arc::clone(&store);
        let child = store.until(Instant::now() + Duration::from_secs(5));

        let t1 = std::thread::spawn(move || parent.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        // Wait until t1 is inside `GET /api/system`, then launch a concurrent `until` caller.
        entered_rx.recv().unwrap();
        let t2 = std::thread::spawn(move || child.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        // Wait deterministically until t2 is blocked on the single-flight condvar before releasing the response.
        while store.probe.state.lock().unwrap().waiters == 0 {
            std::thread::yield_now();
        }
        release_tx.send(()).unwrap();

        assert!(t1.join().unwrap());
        assert!(t2.join().unwrap());
        assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/system"]);
    }

    #[test]
    fn in_flight_probe_does_not_repopulate_after_clear_probe_cache() {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let server = MockApi::start(move |req, idx| match (req, idx) {
            ("GET /api/health", 0) => {
                let _ = entered_tx.send(());
                let _ = release_rx.lock().unwrap().recv();
                (
                    "200 OK",
                    serde_json::json!({
                        "status": "ok",
                        "version": "0.1.0",
                        "mock": false,
                        "features": ["stale_feature"],
                    })
                    .to_string(),
                )
            }
            ("GET /api/health", _) => (
                "200 OK",
                serde_json::json!({
                    "status": "ok",
                    "version": "0.1.0",
                    "mock": true,
                    "features": [folo_api::FEATURE_VISION_BENCH],
                })
                .to_string(),
            ),
            other => panic!("unexpected request: {other:?}"),
        });

        let store = Arc::new(HttpStore::new(&server.base, None));
        let probe_store = Arc::clone(&store);
        let t = std::thread::spawn(move || probe_store.has_feature("stale_feature").unwrap());
        entered_rx.recv().unwrap();
        // Clear the cache while the first probe is in flight, then let it finish.
        store.clear_probe_cache();
        release_tx.send(()).unwrap();
        assert!(t.join().unwrap());

        // The stale probe must not have populated the cache: the next check makes a fresh request.
        assert!(store.has_feature(folo_api::FEATURE_VISION_BENCH).unwrap());
        assert!(store.is_mock());
        assert_eq!(server.requests(), vec!["GET /api/health", "GET /api/health"]);
    }

    #[test]
    fn multipart_rejects_header_injection_and_avoids_boundary_collision() {
        let bad = [("model_id", "m_1\r\n--fake".to_string())];
        assert_eq!(multipart(&bad, b"img").unwrap_err().body.code, "invalid_arguments");
        let bad_nul = [("model_id", "m_1\0x".to_string())];
        assert_eq!(multipart(&bad_nul, b"img").unwrap_err().body.code, "invalid_arguments");

        let fields = [("model_id", "m_1".to_string())];
        let (ct, body) = multipart(&fields, b"hello--folo-mcp-world").unwrap();
        let boundary = ct.strip_prefix("multipart/form-data; boundary=").unwrap();
        assert!(body.windows(boundary.len()).filter(|w| *w == boundary.as_bytes()).count() == 3);
    }
}
