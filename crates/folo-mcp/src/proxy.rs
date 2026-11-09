//! Proxy mode (`folo mcp --server http://127.0.0.1:7878`): a [`Store`] that forwards to a running `folo serve`
//! over docs/API.md, so Studio and the agent share datasets and runs. Every tool goes through the same `Store`
//! methods as in embedded mode; only the transport differs. Calls are blocking (`ureq`); the MCP layer runs them
//! on tokio's blocking pool.

use std::time::Duration;

use folo_api::{
    Dataset, DatasetList, ErrorBody, ExportResult, ExportSpec, FEATURE_EXPORT_ONNX_SWITCHES, GpuInfo, Health,
    ImagePage, IssuePage, LogLine, Model, ModelEval, ModelList, PredictRequest, PredictResponse, QueueAction,
    QueueRequest, RegisterDataset, Run, RunList, RunLogs, RunMetrics, RunSamples, SyntheticDataset,
    SystemInfo, SystemResponse, TrainConfig, ValReport, ValRequest,
};
use folo_server::store::{
    ImageData, ImageQuery, IssueQuery, PredictInput, PredictSource, QueueMove, Store, Subscription,
};
use http::StatusCode;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::errors::{Error, not_implemented};

/// Largest response body read from the server (images, metric histories).
const MAX_BODY: u64 = 256 * 1024 * 1024;

/// A [`Store`] backed by a `folo serve` HTTP API.
pub struct HttpStore {
    base: String,
    agent: ureq::Agent,
    mock: std::sync::OnceLock<bool>,
}

fn action(a: QueueAction) -> QueueRequest {
    QueueRequest { action: Some(a), position: None }
}

/// Percent-encodes one path segment or query value.
pub fn encode(s: &str) -> String {
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

impl HttpStore {
    /// A store forwarding to `base` (e.g. `http://127.0.0.1:7878`).
    pub fn new(base: &str) -> HttpStore {
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into();
        HttpStore { base: base.trim_end_matches('/').to_string(), agent, mock: std::sync::OnceLock::new() }
    }

    /// The server's base URL.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// `GET /api/system` of the server.
    pub fn system(&self) -> Result<SystemResponse, Error> {
        self.get("/system")
    }

    fn url(&self, path: &str) -> String {
        format!("{}/api{path}", self.base)
    }

    fn unreachable(&self, e: ureq::Error) -> Error {
        Error::new(StatusCode::BAD_GATEWAY, "server_unreachable", format!("{}: {e}", self.base))
            .hint("is `folo serve` running there? start it, or run `folo mcp` without --server")
    }

    /// Reads a response: the body on 2xx, the API error envelope otherwise.
    fn finish(
        &self,
        res: Result<http::Response<ureq::Body>, ureq::Error>,
    ) -> Result<(http::HeaderMap, Vec<u8>), Error> {
        let res = res.map_err(|e| self.unreachable(e))?;
        let status = res.status();
        let headers = res.headers().clone();
        let bytes =
            res.into_body().with_config().limit(MAX_BODY).read_to_vec().map_err(|e| self.unreachable(e))?;
        if status.is_success() {
            return Ok((headers, bytes));
        }
        let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        Err(match serde_json::from_slice::<ErrorBody>(&bytes) {
            Ok(b) => Error { status, body: b.error },
            Err(_) => Error::new(status, "internal", String::from_utf8_lossy(&bytes).into_owned()),
        })
    }

    fn parse<T: DeserializeOwned>(&self, bytes: &[u8]) -> Result<T, Error> {
        serde_json::from_slice(bytes)
            .map_err(|e| Error::internal(format!("unexpected response from {}: {e}", self.base)))
    }

    fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, Error> {
        let (_, b) = self.finish(self.agent.get(&self.url(path)).call())?;
        self.parse(&b)
    }

    fn post<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T, Error> {
        let json = serde_json::to_vec(body).expect("request bodies serialize");
        let req = self.agent.post(&self.url(path)).header("content-type", "application/json");
        let (_, b) = self.finish(req.send(&json[..]))?;
        self.parse(&b)
    }

    fn delete(&self, path: &str) -> Result<(), Error> {
        self.finish(self.agent.delete(&self.url(path)).call()).map(|_| ())
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

/// A `multipart/form-data` body for `POST /api/predict`.
fn multipart(fields: &[(&str, String)], image: &[u8]) -> (String, Vec<u8>) {
    let boundary = format!(
        "folo-mcp-{:x}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
    );
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
    (format!("multipart/form-data; boundary={boundary}"), body)
}

impl Store for HttpStore {
    fn is_mock(&self) -> bool {
        *self.mock.get_or_init(|| self.get::<Health>("/health").map(|h| h.mock).unwrap_or(false))
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
        let (headers, bytes) = self.finish(self.agent.get(&self.url(&path)).call())?;
        let ct = headers.get(http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
        Ok(ImageData { content_type: static_content_type(ct), bytes, immutable: false })
    }

    fn delete_dataset(&self, id: &str) -> Result<(), Error> {
        self.delete(&format!("/datasets/{}", encode(id)))
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
        if spec.onnx.any() && !self.system()?.features.iter().any(|f| f == FEATURE_EXPORT_ONNX_SWITCHES) {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
                format!(
                    "the folo serve at {} is too old for the ONNX switches (decode, dynamic, half)",
                    self.base
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
        let (_, b) =
            self.finish(self.agent.post(&self.url(&format!("/runs/{}/stop", encode(id)))).send_empty())?;
        self.parse(&b)
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

    fn run_metrics(&self, id: &str, every: u64) -> Result<RunMetrics, Error> {
        self.get(&format!("/runs/{}/metrics?every={}", encode(id), every.max(1)))
    }

    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples, Error> {
        let q = epoch.map_or(String::new(), |e| format!("?epoch={e}"));
        self.get(&format!("/runs/{}/samples{q}", encode(id)))
    }

    fn run_logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>, Error> {
        Ok(self.get::<RunLogs>(&format!("/runs/{}/logs?limit={}", encode(id), limit.clamp(1, 5000)))?.lines)
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
                let (ct, body) = multipart(&fields, &bytes);
                let req = self.agent.post(&self.url("/predict")).header("content-type", &ct);
                let (_, b) = self.finish(req.send(&body[..]))?;
                self.parse(&b)
            }
        }
    }

    fn val(&self, req: ValRequest) -> Result<ValReport, Error> {
        self.post("/val", &req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_segments() {
        assert_eq!(encode("ds_1"), "ds_1");
        assert_eq!(encode("a/b c"), "a%2Fb%20c");
    }

    #[test]
    fn unreachable_server_is_a_tool_error() {
        // Port 9 (discard) is essentially never served on loopback.
        let s = HttpStore::new("http://127.0.0.1:9");
        let e = s.list_datasets().unwrap_err();
        assert_eq!(e.body.code, "server_unreachable");
    }
}
