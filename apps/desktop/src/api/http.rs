//! [`Backend`] over HTTP + SSE against `folo serve` (docs/API.md), and a launcher for a local
//! `folo serve --mock`.

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
        Self {
            base: base.trim_end_matches('/').to_string(),
            agent: agent(Some(Duration::from_secs(30))),
            stream_agent: stream_agent(handshake),
        }
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

    fn post_empty<T: DeserializeOwned>(&self, path: &str) -> ApiResult<T> {
        let resp = self
            .agent
            .post(&self.url(path))
            .header("Content-Type", "application/json")
            .send("{}")
            .map_err(|e| self.unreachable(&e))?;
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

    fn system(&self) -> ApiResult<SystemInfo> {
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

/// A `folo serve --mock` started by the app; stopped when dropped.
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

/// Start `folo serve --mock` on a free port and wait (up to `timeout`) for its address.
pub fn spawn_mock_server(timeout: Duration) -> Result<ServerProcess, String> {
    let bin = find_folo().ok_or("folo was not found (set FOLO_BIN or put folo on PATH)")?;
    let mut child = Command::new(&bin)
        .args(["--json", "serve", "--mock", "--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", bin.display()))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
                && v["event"] == "listening"
                && let Some(url) = v["url"].as_str()
            {
                let _ = tx.send(url.to_string());
            }
        }
    });
    bind_to_app(&child);
    match rx.recv_timeout(timeout) {
        Ok(url) => Ok(ServerProcess { child, url }),
        Err(_) => {
            let _ = child.kill();
            Err("folo serve did not start in time".into())
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
                std::thread::sleep(Duration::from_millis(30));
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
        // Still open while subscribed.
        assert!(failed.recv_timeout(Duration::from_millis(300)).is_err());
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
        let r = HttpBackend::with_handshake(&format!("http://{addr}"), Duration::from_millis(300))
            .events("run_x");
        assert!(r.is_err());
        assert!(t0.elapsed() < Duration::from_secs(3), "handshake took {:?}", t0.elapsed());
    }

    #[test]
    fn sse_frames_split_into_events() {
        let stream = "event: status\ndata: {\"status\":\"running\",\"error\":null}\n\n: keep-alive\n\nevent: log\ndata: {\"t\":\"2026-11-08T07:00:00Z\",\"level\":\"info\",\"message\":\"hi\"}\n\nevent: future\ndata: {}\n\n";
        let mut seen = Vec::new();
        parse_sse(std::io::Cursor::new(stream), |name, data| {
            seen.push((name.to_string(), RunEvent::from_sse(name, data).map(|r| r.is_ok())));
            true
        });
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0], ("status".into(), Some(true)));
        assert_eq!(seen[1], ("log".into(), Some(true)));
        assert_eq!(seen[2], ("future".into(), None));
    }

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(seg("ds_8f2c"), "ds_8f2c");
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
    }

    #[test]
    fn unreachable_server_is_a_clean_error() {
        let api = HttpBackend::new("http://127.0.0.1:9");
        let e = api.health().unwrap_err();
        assert_eq!(e.code, "server_unreachable");
        assert!(e.hint.is_some());
    }
}
