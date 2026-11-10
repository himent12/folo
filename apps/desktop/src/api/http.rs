//! [`Backend`] over HTTP + SSE against `folo serve` (docs/API.md), and a launcher for a local
//! `folo serve`.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::channel;
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::*;

/// How long an event stream may take to send its response headers.
const STREAM_HANDSHAKE: Duration = Duration::from_secs(5);

pub struct HttpBackend {
    base: String,
    agent: ureq::Agent,
    /// No overall timeout (SSE streams stay open for the life of a run), but the handshake is
    /// bounded: connect and response headers.
    stream_agent: ureq::Agent,
}

fn agent(timeout: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_global(timeout)
        .build()
        .into()
}

fn stream_agent(headers: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(2)))
        .timeout_recv_response(Some(headers))
        .build()
        .into()
}

/// Percent-encode one path segment (ids are opaque strings).
fn seg(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

impl HttpBackend {
    /// `base` like `http://127.0.0.1:7878` (no trailing `/api`).
    pub fn new(base: &str) -> Self {
        Self::with_handshake(base, STREAM_HANDSHAKE)
    }

    fn with_handshake(base: &str, handshake: Duration) -> Self {
        // As typed or pasted in Settings: surrounding whitespace (a copied line's end) and a
        // missing scheme ("127.0.0.1:7878") would otherwise make every request fail.
        let base = base.trim().trim_end_matches('/');
        let base = if base.contains("://") { base.to_string() } else { format!("http://{base}") };
        Self { base, agent: agent(Some(Duration::from_secs(30))), stream_agent: stream_agent(handshake) }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api{path}", self.base)
    }

    fn unreachable(&self, e: &ureq::Error) -> ApiError {
        ApiError::new("server_unreachable", format!("Could not reach folo serve at {}", self.host()))
            .with_hint(format!(
                "Start it with `folo serve`, or switch to the built-in demo in Settings. ({e})"
            ))
    }

    fn host(&self) -> String {
        self.base.trim_start_matches("http://").trim_start_matches("https://").to_string()
    }

    /// Turn a response into `T`, or the API's error body into an [`ApiError`].
    fn decode<T: DeserializeOwned>(&self, mut resp: ureq::http::Response<ureq::Body>) -> ApiResult<T> {
        let status = resp.status().as_u16();
        if status >= 400 {
            let body: Option<folo_api::ErrorBody> = resp.body_mut().read_json().ok();
            return Err(match body {
                Some(b) => ApiError { code: b.error.code, message: b.error.message, hint: b.error.hint },
                None => ApiError::new("http_error", format!("folo serve answered HTTP {status}")),
            });
        }
        if status == 204 {
            return serde_json::from_str("null").map_err(|e| ApiError::new("bad_response", e.to_string()));
        }
        resp.body_mut()
            .with_config()
            .limit(256 << 20)
            .read_json()
            .map_err(|e| ApiError::new("bad_response", format!("Unexpected response from folo serve: {e}")))
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.get(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = self.agent.post(&self.url(path)).send_json(body).map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A POST that may legitimately take minutes (indexing a large folder, writing images).
    fn post_slow<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> ApiResult<T> {
        let resp = agent(Some(Duration::from_secs(600)))
            .post(&self.url(path))
            .send_json(body)
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    fn post_empty<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self
            .agent
            .post(&self.url(path))
            .header("Content-Type", "application/json")
            .send("{}")
            .map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }

    /// A DELETE: `204` decodes as `()`, a body as `T`.
    fn delete<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self.agent.delete(&self.url(path)).call().map_err(|e| self.unreachable(&e))?;
        self.decode(resp)
    }
}

/// Split an SSE stream into `(event, data)` pairs; comments and unknown fields are skipped.
#[cfg(test)]
pub fn parse_sse(reader: impl BufRead, on_event: impl FnMut(&str, &str) -> bool) {
    parse_sse_while(reader, || true, on_event)
}

/// [`parse_sse`] that also stops as soon as `go` turns false, checked on every line received
/// (keep-alive comments included), so a quiet stream still notices a cancelled subscriber.
pub fn parse_sse_while(
    reader: impl BufRead,
    go: impl Fn() -> bool,
    mut on_event: impl FnMut(&str, &str) -> bool,
) {
    let mut event = String::new();
    let mut data = String::new();
    for line in reader.lines() {
        let Ok(line) = line else { return };
        if !go() {
            return;
        }
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            if !data.is_empty() {
                let name = if event.is_empty() { "message" } else { event.as_str() };
                if !on_event(name, &data) {
                    return;
                }
            }
            event.clear();
            data.clear();
        } else if let Some(v) = line.strip_prefix("event:") {
            event = v.trim_start().to_string();
        } else if let Some(v) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(v.strip_prefix(' ').unwrap_or(v));
        }
    }
}

impl Backend for HttpBackend {
    fn describe(&self) -> String {
        self.host()
    }

    fn now(&self) -> i64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
    }

    fn health(&self) -> ApiResult<Health> {
        self.get("/health")
    }

    fn system(&self) -> ApiResult<SystemResponse> {
        self.get("/system")
    }

    fn datasets(&self) -> ApiResult<Vec<Dataset>> {
        self.get::<folo_api::DatasetList>("/datasets").map(|l| l.datasets)
    }

    fn dataset(&self, id: &str) -> ApiResult<Dataset> {
        self.get(&format!("/datasets/{}", seg(id)))
    }

    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        // The server caps a page at 1000; fetch pages until the requested limit is met.
        let want = if q.limit == 0 { 100 } else { q.limit } as u64;
        let mut images = Vec::new();
        let mut total;
        let mut offset = q.offset as u64;
        loop {
            let limit = (want - images.len() as u64).min(1000);
            let mut path = format!("/datasets/{}/images?offset={offset}&limit={limit}", seg(id));
            if let Some(s) = q.split {
                path.push_str(&format!("&split={}", s.as_str()));
            }
            if let Some(c) = q.class {
                path.push_str(&format!("&class={c}"));
            }
            let page: ImagePage = self.get(&path)?;
            total = page.total;
            let n = page.images.len() as u64;
            images.extend(page.images);
            offset += n;
            if n == 0 || images.len() as u64 >= want || offset >= total {
                break;
            }
        }
        Ok(ImagePage { total, images })
    }

    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        let mut path = format!("/datasets/{}/images/{}", seg(dataset_id), seg(image_id));
        if let Some(t) = thumb {
            path.push_str(&format!("?thumb={t}"));
        }
        let mut resp = self.agent.get(&self.url(&path)).call().map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "image request failed")));
        }
        let bytes = resp
            .body_mut()
            .with_config()
            .limit(64 << 20)
            .read_to_vec()
            .map_err(|e| ApiError::new("bad_response", e.to_string()))?;
        Ok(ImageData::Encoded(bytes))
    }

    fn models(&self) -> ApiResult<Vec<Model>> {
        self.get::<folo_api::ModelList>("/models").map(|l| l.models)
    }

    fn model_eval(&self, id: &str) -> ApiResult<ModelEval> {
        self.get(&format!("/models/{}/eval", seg(id)))
    }

    fn export_model(&self, id: &str, format: &str) -> ApiResult<ExportResult> {
        let format =
            if format == "onnx" { folo_api::ExportFormat::Onnx } else { folo_api::ExportFormat::FoloF16 };
        self.post(&format!("/models/{}/export", seg(id)), &folo_api::ExportRequest { format })
    }

    fn runs(&self) -> ApiResult<Vec<Run>> {
        self.get::<folo_api::RunList>("/runs").map(|l| l.runs)
    }

    fn run(&self, id: &str) -> ApiResult<Run> {
        self.get(&format!("/runs/{}", seg(id)))
    }

    fn create_run(&self, config: TrainConfig) -> ApiResult<Run> {
        self.post("/runs", &config)
    }

    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.post_empty(&format!("/runs/{}/stop", seg(id)))
    }

    fn run_cli(&self, id: &str) -> ApiResult<CliCommand> {
        self.get(&format!("/runs/{}/cli", seg(id)))
    }

    fn estimate_run(&self, config: &TrainConfig) -> ApiResult<RunEstimate> {
        self.post("/runs/estimate", config)
    }

    fn throughput(&self) -> ApiResult<ThroughputTable> {
        self.get("/system/throughput")
    }

    fn benchmark(&self, req: &BenchmarkRequest) -> ApiResult<BenchmarkReport> {
        self.post("/system/benchmark", req)
    }

    fn queue_run(&self, id: &str, req: QueueRequest) -> ApiResult<Run> {
        self.post(&format!("/runs/{}/queue", seg(id)), &req)
    }

    fn resume_run(&self, id: &str) -> ApiResult<Run> {
        #[derive(Serialize)]
        struct Resume<'a> {
            resume_run_id: &'a str,
        }
        self.post("/runs", &Resume { resume_run_id: id })
    }

    fn register_dataset(&self, req: &RegisterDataset) -> ApiResult<Dataset> {
        self.post_slow("/datasets", req)
    }

    fn synth_dataset(&self, req: &SyntheticDataset) -> ApiResult<Dataset> {
        self.post_slow("/datasets/synthetic", req)
    }

    fn dataset_issues(&self, id: &str) -> ApiResult<IssuePage> {
        self.get(&format!("/datasets/{}/issues?limit=1", seg(id)))
    }

    fn cloud_providers(&self) -> ApiResult<Vec<CloudProvider>> {
        self.get::<folo_api::CloudProviderList>("/cloud/providers").map(|l| l.providers)
    }

    fn cloud_connections(&self) -> ApiResult<Vec<CloudConnection>> {
        self.get::<folo_api::CloudConnectionList>("/cloud/connections").map(|l| l.connections)
    }

    fn connect_cloud(&self, req: &ConnectCloud) -> ApiResult<CloudConnection> {
        self.post("/cloud/connections", req)
    }

    fn check_cloud(&self, id: &str) -> ApiResult<CloudConnection> {
        self.post_empty(&format!("/cloud/connections/{}/check", seg(id)))
    }

    fn disconnect_cloud(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/cloud/connections/{}", seg(id)))
    }

    fn cloud_offers(&self, connection: &str, refresh: bool) -> ApiResult<CloudOfferList> {
        let query = if refresh { "?refresh=true" } else { "" };
        self.get(&format!("/cloud/connections/{}/offers{query}", seg(connection)))
    }

    fn cloud_instances(&self) -> ApiResult<Vec<CloudInstance>> {
        self.get::<folo_api::CloudInstanceList>("/cloud/instances").map(|l| l.instances)
    }

    fn terminate_instance(&self, id: &str) -> ApiResult<CloudInstance> {
        self.delete(&format!("/cloud/instances/{}", seg(id)))
    }

    fn local_inference(&self) -> bool {
        true
    }

    fn model_weights(&self, id: &str, etag: Option<&str>) -> ApiResult<Option<Weights>> {
        let url = self.url(&format!("/models/{}/weights?format=f16", seg(id)));
        // The first request of a checkpoint makes its f16 export on the server: allow for it. A
        // revalidation (a copy is already loaded) gives up on a silent server after 15 s, so the
        // copy runs instead of every local prediction waiting on it.
        let headers = if etag.is_some() { 15 } else { 300 };
        let mut req = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(2)))
            .timeout_recv_response(Some(Duration::from_secs(headers)))
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .new_agent()
            .get(&url);
        if let Some(e) = etag {
            req = req.header("If-None-Match", e);
        }
        let mut resp = req.call().map_err(|e| self.unreachable(&e))?;
        match resp.status().as_u16() {
            304 => return Ok(None),
            s if s >= 400 => {
                return Err(self.decode::<serde_json::Value>(resp).err().unwrap_or_else(|| {
                    ApiError::new("http_error", format!("folo serve answered HTTP {s}"))
                }));
            }
            _ => {}
        }
        let etag = resp.headers().get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
        let bytes = resp.body_mut().with_config().limit(2 << 30).read_to_vec().map_err(|e| match e {
            // The connection dropped or stalled mid-download: the server is out of reach (callers
            // holding older weights fall back to them), not a malformed answer.
            ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed => {
                self.unreachable(&e)
            }
            _ => ApiError::new("bad_response", e.to_string()),
        })?;
        Ok(Some(Weights { bytes, etag }))
    }

    fn metrics(&self, id: &str) -> ApiResult<RunMetrics> {
        self.get(&format!("/runs/{}/metrics", seg(id)))
    }

    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<RunSamples> {
        match epoch {
            Some(e) => self.get(&format!("/runs/{}/samples?epoch={e}", seg(id))),
            None => self.get(&format!("/runs/{}/samples", seg(id))),
        }
    }

    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>> {
        self.get::<folo_api::RunLogs>(&format!("/runs/{}/logs?limit=1000", seg(id))).map(|l| l.lines)
    }

    fn events(&self, id: &str) -> ApiResult<Subscription> {
        let url = self.url(&format!("/runs/{}/events", seg(id)));
        let resp = self
            .stream_agent
            .get(&url)
            .header("Accept", "text/event-stream")
            .call()
            .map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "event stream failed")));
        }
        let (tx, rx) = channel();
        let (sub, cancel) = Subscription::new(rx);
        let reader = BufReader::new(resp.into_body().into_reader());
        std::thread::spawn(move || {
            // Ends when the server closes the stream or the subscription is dropped; the reader and
            // its socket go with it.
            parse_sse_while(
                reader,
                || !cancel.load(std::sync::atomic::Ordering::Relaxed),
                |name, data| match RunEvent::from_sse(name, data) {
                    // A finished run's stream stays open on the server; once its final status is
                    // delivered there's nothing more to read.
                    Some(Ok(RunEvent::Status(s))) if !s.status.is_active() => {
                        let _ = tx.send(RunEvent::Status(s));
                        false
                    }
                    Some(Ok(ev)) => tx.send(ev).is_ok(),
                    // Unknown event types are skipped; malformed data too (the stream goes on).
                    _ => true,
                },
            );
        });
        Ok(sub)
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        match &req.input {
            PredictInput::Dataset { dataset_id, image_id } => self.post(
                "/predict",
                &folo_api::PredictRequest {
                    model_id: req.model_id.clone(),
                    dataset_id: dataset_id.clone(),
                    image_id: image_id.clone(),
                    conf: req.conf,
                    iou: req.iou,
                    max_det: req.max_det,
                },
            ),
            PredictInput::File { name, bytes } => {
                let boundary = format!("folo-{:016x}", crate::synth::hash(name) ^ bytes.len() as u64);
                let mut body = Vec::with_capacity(bytes.len() + 1024);
                let field = |body: &mut Vec<u8>, k: &str, v: &str| {
                    body.extend_from_slice(
                        format!(
                            "--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                        )
                        .as_bytes(),
                    );
                };
                field(&mut body, "model_id", &req.model_id);
                field(&mut body, "conf", &req.conf.to_string());
                field(&mut body, "iou", &req.iou.to_string());
                field(&mut body, "max_det", &req.max_det.to_string());
                let safe_name = name.replace(['"', '\r', '\n'], "_");
                body.extend_from_slice(
                    format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{safe_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
                let resp = self
                    .agent
                    .post(&self.url("/predict"))
                    .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
                    .send(&body[..])
                    .map_err(|e| self.unreachable(&e))?;
                self.decode(resp)
            }
        }
    }
}

/// A local `folo serve` started by the app; stopped when dropped.
pub struct ServerProcess {
    child: Child,
    pub url: String,
}

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Arguments passed to `folo` when starting a local server: the real engine (never `--mock`), on a
/// free port, emitting JSON Lines so we read its `"listening"` record.
pub const SERVE_ARGS: &[&str] = &["--json", "serve", "--port", "0"];

/// Where to find the `folo` binary: `$FOLO_BIN`, next to this executable, or on `PATH`.
pub fn find_folo() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("FOLO_BIN").map(std::path::PathBuf::from)
        && p.is_file()
    {
        return Some(p);
    }
    let exe = if cfg!(windows) { "folo.exe" } else { "folo" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        let p = dir.join(exe);
        if p.is_file() {
            return Some(p);
        }
    }
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(exe)).find(|p| p.is_file()))
}

/// Make the OS stop `child` when this process exits, even if it is killed (no Drop runs then).
#[cfg(windows)]
fn bind_to_app(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
    };
    // SAFETY: plain Win32 calls on handles we own. The job handle is deliberately never closed:
    // the OS closes it when this process exits, and KILL_ON_JOB_CLOSE then ends the child.
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok != 0 {
            AssignProcessToJobObject(job, child.as_raw_handle() as _);
        }
    }
}

#[cfg(not(windows))]
fn bind_to_app(_child: &Child) {}

/// What starting a local `folo serve` gave.
pub enum Spawned {
    /// A server this app started (and stops when the [`ServerProcess`] drops).
    Started(ServerProcess),
    /// The data dir already has a `folo serve` (another app's, or one started by hand), listening here: use it,
    /// don't start a second one on the same data.
    Running(String),
}

/// What the starting server's JSON Lines told us.
enum Startup {
    Listening(String),
    InUse(Option<String>),
    Failed(String),
}

/// Reads one JSON Lines record of `folo --json serve`.
fn startup_record(line: &str) -> Option<Startup> {
    let v = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let url = v["url"].as_str().map(str::to_string);
    match v["event"].as_str() {
        Some("listening") => url.map(Startup::Listening),
        Some("data_dir_in_use") => Some(Startup::InUse(url)),
        _ => v["error"]["message"].as_str().map(|m| Startup::Failed(m.to_string())),
    }
}

/// Start `folo serve` (the real engine, bare until datasets/runs exist) on a free port and wait
/// (up to `timeout`) for its address.
pub fn spawn_server(timeout: Duration) -> Result<Spawned, String> {
    let bin = find_folo().ok_or("folo was not found (set FOLO_BIN or put folo on PATH)")?;
    spawn_server_bin(&bin, timeout)
}

fn spawn_server_bin(bin: &std::path::Path, timeout: Duration) -> Result<Spawned, String> {
    let mut cmd = Command::new(bin);
    cmd.args(SERVE_ARGS).stdout(Stdio::piped()).stderr(Stdio::null()).stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", bin.display()))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(rec) = startup_record(&line)
                && tx.send(rec).is_err()
            {
                break;
            }
        }
    });
    bind_to_app(&child);
    let outcome = rx.recv_timeout(timeout);
    if !matches!(outcome, Ok(Startup::Listening(_))) {
        let _ = child.kill();
        let _ = child.wait();
    }
    match outcome {
        Ok(Startup::Listening(url)) => Ok(Spawned::Started(ServerProcess { child, url })),
        Ok(Startup::InUse(Some(url))) => Ok(Spawned::Running(url)),
        Ok(Startup::InUse(None)) => {
            Err("another folo serve is using the data dir and has not said where".into())
        }
        Ok(Startup::Failed(message)) => Err(message),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err("folo serve did not start in time".into()),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            Err("folo serve exited before it was listening".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server that accepts the event stream request, sends `prelude`, then only keep-alives.
    fn keepalive_server(prelude: &'static [u8], write_failed: std::sync::mpsc::Sender<()>) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n",
            );
            if !prelude.is_empty() {
                let frame = [format!("{:x}\r\n", prelude.len()).as_bytes(), prelude, b"\r\n"].concat();
                let _ = sock.write_all(&frame);
            }
            loop {
                std::thread::sleep(Duration::from_millis(10));
                let chunk = b": keep-alive\n\n";
                let frame = [format!("{:x}\r\n", chunk.len()).as_bytes(), chunk, b"\r\n"].concat();
                if sock.write_all(&frame).and_then(|_| sock.flush()).is_err() {
                    let _ = write_failed.send(());
                    return;
                }
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn dropping_a_subscription_closes_a_quiet_stream() {
        let (tx, failed) = std::sync::mpsc::channel();
        let base = keepalive_server(b"", tx);
        let sub = HttpBackend::new(&base).events("run_x").expect("stream opens");
        // Still open while subscribed, through several keep-alives.
        assert!(failed.recv_timeout(Duration::from_millis(60)).is_err());
        drop(sub);
        // The reader sees the next keep-alive, stops, and closes the socket: the server's writes fail.
        assert!(
            failed.recv_timeout(Duration::from_secs(5)).is_ok(),
            "reader thread outlived its subscription"
        );
    }

    #[test]
    fn a_finished_run_closes_its_stream() {
        let (tx, failed) = std::sync::mpsc::channel();
        let base =
            keepalive_server(b"event: status\ndata: {\"status\":\"completed\",\"error\":null}\n\n", tx);
        let sub = HttpBackend::new(&base).events("run_x").expect("stream opens");
        let first = sub.recv_timeout(Duration::from_secs(2)).expect("status delivered");
        assert!(matches!(first, RunEvent::Status(s) if s.status == RunStatus::Completed));
        // Still subscribed, yet the reader is done: the socket closes and the server's writes fail.
        assert!(failed.recv_timeout(Duration::from_secs(5)).is_ok(), "reader kept a finished stream open");
        drop(sub);
    }

    /// A server answering each connection with the next canned response (and recording requests).
    fn canned_server(responses: Vec<Vec<u8>>) -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for resp in responses {
                let Ok((mut sock, _)) = listener.accept() else { return };
                // The whole request, body included (answering before a POST's body is read would
                // reset the connection under the client).
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = sock.read(&mut buf).unwrap_or(0);
                    req.extend_from_slice(&buf[..n]);
                    let Some(end) = req.windows(4).position(|w| w == b"\r\n\r\n") else {
                        if n == 0 { break } else { continue }
                    };
                    let head = String::from_utf8_lossy(&req[..end]).to_ascii_lowercase();
                    let body = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if n == 0 || req.len() >= end + 4 + body {
                        break;
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&req).into_owned());
                let _ = sock.write_all(&resp);
                let _ = sock.flush();
                // Dropping the socket closes the connection (cutting a short body).
            }
        });
        (format!("http://{addr}"), rx)
    }

    #[test]
    fn weights_revalidate_by_etag_and_a_cut_download_is_unreachable() {
        let (url, requests) = canned_server(vec![
            b"HTTP/1.1 200 OK
etag: \"w1\"
content-length: 4
connection: close

FOLO"
                .to_vec(),
            b"HTTP/1.1 304 Not Modified
etag: \"w1\"
connection: close

"
            .to_vec(),
            // Announces 100 bytes, sends 4, hangs up.
            b"HTTP/1.1 200 OK
etag: \"w2\"
content-length: 100
connection: close

FOLO"
                .to_vec(),
        ]);
        let api = HttpBackend::new(&url);
        let w = api.model_weights("m_1", None).unwrap().expect("weights");
        assert_eq!((w.bytes.as_slice(), w.etag.as_str()), (&b"FOLO"[..], "\"w1\""));
        let first = requests.recv().unwrap();
        assert!(first.starts_with("GET /api/models/m_1/weights?format=f16 "), "{first}");
        assert!(!first.to_ascii_lowercase().contains("if-none-match"));
        assert!(api.model_weights("m_1", Some("\"w1\"")).unwrap().is_none(), "304 is unchanged");
        assert!(requests.recv().unwrap().to_ascii_lowercase().contains("if-none-match: \"w1\""));
        let cut = api.model_weights("m_1", Some("\"w1\"")).err().expect("a cut body fails");
        assert_eq!(cut.code, "server_unreachable");
    }

    /// A canned JSON response with its status line.
    fn json(status: &str, body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn cloud_calls_hit_their_paths_and_decode() {
        let offer = r#"{ "id": "NVIDIA GeForce RTX 4090@secure", "gpu": "RTX 4090", "vram_bytes": 25769803776,
            "price_per_hour": 0.69, "region": null, "detail": "Secure Cloud", "availability": "high" }"#;
        let conn = r#"{ "id": "cc_1", "provider": "runpod", "name": "RunPod", "created": "2026-11-08T06:00:00Z",
            "status": "ok", "account": "you@example.com", "balance_usd": 42.17, "error": null, "note": null,
            "checked": "2026-11-08T07:00:00Z", "values": { "api_key": "••••a1b2" } }"#;
        let instance = format!(
            r#"{{ "id": "ci_1", "connection_id": "cc_1", "provider": "runpod", "provider_ref": "abc123",
            "name": "folo-run_1-4c1a", "run_id": "run_1", "offer": {offer}, "status": "terminating",
            "created": "2026-11-08T06:00:00Z", "ended": null, "deadline": "2026-11-08T18:00:00Z",
            "cost_usd": 0.42, "error": null }}"#
        );
        let (url, requests) = canned_server(vec![
            json(
                "200 OK",
                r#"{ "providers": [{ "id": "runpod", "name": "RunPod", "about": "GPU pods rented by the hour.",
                    "docs_url": "https://www.runpod.io/console/user/settings",
                    "fields": [{ "key": "api_key", "label": "API key", "secret": true, "required": true }],
                    "provisions": true, "self_terminates": true }] }"#,
            ),
            json("200 OK", &format!(r#"{{ "connections": [{conn}] }}"#)),
            json("201 Created", conn),
            json("200 OK", conn),
            b"HTTP/1.1 204 No Content\r\ncontent-length: 0\r\nconnection: close\r\n\r\n".to_vec(),
            json("200 OK", &format!(r#"{{ "offers": [{offer}], "fetched": "2026-11-08T07:00:00Z" }}"#)),
            json("200 OK", &format!(r#"{{ "instances": [{instance}] }}"#)),
            json("200 OK", &instance),
            json(
                "409 Conflict",
                r#"{ "error": { "code": "connection_in_use", "message": "RunPod has a live instance", "hint": null } }"#,
            ),
        ]);
        let api = HttpBackend::new(&url);
        let line = || requests.recv().unwrap().lines().next().unwrap_or_default().to_string();

        let providers = api.cloud_providers().unwrap();
        assert!(providers[0].fields[0].secret && providers[0].provisions);
        assert_eq!(line(), "GET /api/cloud/providers HTTP/1.1");
        assert_eq!(api.cloud_connections().unwrap()[0].balance_usd, Some(42.17));
        assert_eq!(line(), "GET /api/cloud/connections HTTP/1.1");
        let req = ConnectCloud {
            provider: "runpod".into(),
            name: String::new(),
            values: [("api_key".to_string(), "rpa_x".to_string())].into(),
        };
        assert_eq!(api.connect_cloud(&req).unwrap().id, "cc_1");
        assert_eq!(line(), "POST /api/cloud/connections HTTP/1.1");
        assert_eq!(api.check_cloud("cc_1").unwrap().status, CloudConnectionStatus::Ok);
        assert_eq!(line(), "POST /api/cloud/connections/cc_1/check HTTP/1.1");
        api.disconnect_cloud("cc_1").unwrap();
        assert_eq!(line(), "DELETE /api/cloud/connections/cc_1 HTTP/1.1");
        let offers = api.cloud_offers("cc_1", true).unwrap();
        assert_eq!((offers.offers[0].gpu_count, offers.offers[0].availability), (1, CloudAvailability::High));
        assert_eq!(line(), "GET /api/cloud/connections/cc_1/offers?refresh=true HTTP/1.1");
        assert_eq!(api.cloud_instances().unwrap()[0].status, CloudInstanceStatus::Terminating);
        assert_eq!(line(), "GET /api/cloud/instances HTTP/1.1");
        assert_eq!(api.terminate_instance("ci_1").unwrap().cost_usd, Some(0.42));
        assert_eq!(line(), "DELETE /api/cloud/instances/ci_1 HTTP/1.1");
        let busy = api.disconnect_cloud("cc_1").unwrap_err();
        assert_eq!(
            (busy.code.as_str(), busy.message.as_str()),
            ("connection_in_use", "RunPod has a live instance")
        );
    }

    #[test]
    fn a_silent_server_cannot_hang_the_stream_handshake() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept and never answer.
        std::thread::spawn(move || {
            let _held = listener.accept();
            std::thread::sleep(Duration::from_secs(10));
        });
        let t0 = std::time::Instant::now();
        let r =
            HttpBackend::with_handshake(&format!("http://{addr}"), Duration::from_millis(50)).events("run_x");
        assert!(r.is_err());
        assert!(t0.elapsed() < Duration::from_secs(3), "handshake took {:?}", t0.elapsed());
    }

    #[test]
    fn sse_frames_split_into_events() {
        let stream = "event: status\ndata: {\"status\":\"running\",\"error\":null}\n\n: keep-alive\n\nevent: log\ndata: {\"t\":\"2026-11-08T07:00:00Z\",\"level\":\"info\",\"message\":\"hi\"}\n\nevent: step\ndata: {\"epoch\":12,\"step\":340,\"lr\":0.0098,\"loss\":{\"box\":1.21,\"cls\":0.83,\"dfl\":1.02,\"total\":3.06},\"imgs_per_s\":412.5,\"eta_s\":1840}\n\nevent: future\ndata: {}\n\n";
        let mut seen = Vec::new();
        parse_sse(std::io::Cursor::new(stream), |name, data| {
            seen.push((name.to_string(), RunEvent::from_sse(name, data)));
            true
        });
        let names: Vec<&str> = seen.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["status", "log", "step", "future"]);
        assert!(matches!(seen[0].1, Some(Ok(RunEvent::Status(_)))));
        assert!(matches!(seen[1].1, Some(Ok(RunEvent::Log(_)))));
        // The loss's `box` field (a Rust keyword) decodes.
        assert!(matches!(&seen[2].1, Some(Ok(RunEvent::Step(s))) if s.loss.box_ == 1.21));
        assert!(seen[3].1.is_none(), "an unknown event is skipped");
    }

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(seg("ds_8f2c"), "ds_8f2c");
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn a_pasted_or_bare_server_address_still_connects() {
        // As Settings › Connection hands it over: pasted with the line's end, or typed without
        // the scheme.
        let (url, requests) =
            canned_server(vec![json("200 OK", r#"{"providers":[]}"#), json("200 OK", r#"{"providers":[]}"#)]);
        let pasted = format!(" {url}/\r\n");
        assert!(HttpBackend::new(&pasted).cloud_providers().is_ok(), "{pasted:?}");
        assert!(requests.recv().unwrap().starts_with("GET /api/cloud/providers "));
        let bare = url.trim_start_matches("http://").to_string();
        assert!(HttpBackend::new(&bare).cloud_providers().is_ok(), "{bare:?}");
        assert_eq!(HttpBackend::new(&bare).describe(), bare);
    }

    #[test]
    fn unreachable_server_is_a_clean_error() {
        // A server that hangs up without answering (a refused connection takes Windows about
        // 2 s of retries to report, and goes the same way).
        let (url, _requests) = canned_server(vec![Vec::new()]);
        let api = HttpBackend::new(&url);
        let e = api.health().unwrap_err();
        assert_eq!(e.code, "server_unreachable");
        assert!(e.hint.is_some());
    }

    #[test]
    fn startup_records_are_read() {
        let url = |r: Option<Startup>| match r {
            Some(Startup::Listening(u)) => format!("listening {u}"),
            Some(Startup::InUse(u)) => format!("in use {u:?}"),
            Some(Startup::Failed(m)) => format!("failed {m}"),
            None => "none".into(),
        };
        assert_eq!(
            url(startup_record(r#"{"event":"listening","url":"http://127.0.0.1:5000","mock":false}"#)),
            "listening http://127.0.0.1:5000"
        );
        assert_eq!(
            url(startup_record(
                r#"{"event":"data_dir_in_use","data_dir":"C:/x","url":"http://127.0.0.1:7"}"#
            )),
            r#"in use Some("http://127.0.0.1:7")"#
        );
        assert_eq!(url(startup_record(r#"{"event":"data_dir_in_use","url":null}"#)), "in use None");
        assert_eq!(
            url(startup_record(r#"{"error":{"code":"bind_failed","message":"could not bind","hint":null}}"#)),
            "failed could not bind"
        );
        assert_eq!(url(startup_record("FOLO serving")), "none");
    }

    #[test]
    fn spawned_server_uses_the_real_store_not_mock() {
        assert_eq!(SERVE_ARGS, ["--json", "serve", "--port", "0"]);
        assert!(!SERVE_ARGS.contains(&"--mock"));
    }
}
