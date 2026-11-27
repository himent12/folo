//! [`Backend`] over HTTP + SSE against `folo serve` (docs/API.md), and a launcher for a local
//! `folo serve`.

mod images;
mod inference;
mod label;
mod refused;
mod server;
mod streams;
mod transport;

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;

use super::*;

pub use server::{Probe, ServerProcess, Spawned, is_loopback, probe, runs_now, spawn_server};
pub(crate) use transport::address_ok;
#[cfg(test)]
pub(crate) use transport::needs_token;
use transport::{BODY_LIMIT, agent, json, seg, server_error, status_error, stream_agent};

/// How long an event stream may take to send its response headers.
const STREAM_HANDSHAKE: Duration = Duration::from_secs(5);
/// How long an open event stream may say nothing (not even the server's 15 s keep-alive) before it is
/// taken for dead: three keep-alives missed.
const STREAM_SILENCE: Duration = Duration::from_secs(45);

pub struct HttpBackend {
    base: String,
    agent: ureq::Agent,
    /// No overall timeout (SSE streams stay open for the life of a run), but the handshake is
    /// bounded (connect and response headers), and so is a silence on an open stream.
    stream_agent: ureq::Agent,
    /// The address as typed when it names no HTTP server ("abc def", "http://", "foo://x"):
    /// nothing is sent, and errors say so.
    not_an_address: Option<String>,
    /// Canonical architecture strings (`"folo2-*"`) learned from `create_run_with_arch`, `run_cli`,
    /// or `GET /api/models/{id}`'s `x-folo-arch` header.
    run_archs: Mutex<HashMap<String, String>>,
    model_archs: Mutex<HashMap<String, String>>,
}

impl HttpBackend {
    /// `base` like `http://127.0.0.1:7878` (no trailing `/api`).
    pub fn new(base: &str) -> Self {
        Self::with_stream_limits(base, STREAM_HANDSHAKE, STREAM_SILENCE)
    }

    fn with_stream_limits(base: &str, handshake: Duration, silence: Duration) -> Self {
        // As typed or pasted in Settings: surrounding whitespace (a copied line's end) and a
        // missing scheme ("127.0.0.1:7878") would otherwise make every request fail.
        let typed = base.trim();
        // Judged as typed: trimming "http://" to "http:" and adding a scheme would make it a host.
        let not_an_address = (!transport::address_ok(typed)).then(|| typed.to_string());
        // Left as typed then, so ureq turns it down before a lookup ("http:///api" has no host).
        let base = match &not_an_address {
            Some(typed) => typed.clone(),
            None if typed.contains("://") => typed.trim_end_matches('/').to_string(),
            None => format!("http://{}", typed.trim_end_matches('/')),
        };
        Self {
            base,
            agent: agent(Some(Duration::from_secs(30))),
            stream_agent: stream_agent(handshake, silence),
            not_an_address,
            run_archs: Mutex::new(HashMap::new()),
            model_archs: Mutex::new(HashMap::new()),
        }
    }

    /// The server's address as requests use it (scheme added, no trailing `/`).
    pub fn base(&self) -> &str {
        &self.base
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
        let mut resp = self.agent.get(&self.url("/health")).call().map_err(|e| self.unreachable(&e))?;
        let status = resp.status().as_u16();
        // An answer is folo serve's exactly when its body has FOLO's shape: on a 2xx a healthy health
        // (`status: "ok"`, as Studio's `isHealth`), on any other status its error JSON. Such an answer
        // reads as every call's; anything else came from something else at the address (#349).
        let body = match resp.status().is_success() {
            true => json::<Health>(resp.body_mut(), BODY_LIMIT).map(Ok),
            false => json::<folo_api::ErrorBody>(resp.body_mut(), BODY_LIMIT).map(Err),
        };
        match body {
            Ok(Ok(h)) if h.status == "ok" => Ok(h),
            Ok(Ok(_)) => Err(self.not_folo(status)),
            Ok(Err(b)) => Err(server_error(b, status)),
            // A body cut off midway was never read whole, so it has no shape to judge: as on any call.
            Err(ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed)
                if status >= 400 =>
            {
                Err(status_error(status))
            }
            Err(e @ (ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed)) => {
                Err(self.body_failed(e, ""))
            }
            Err(_) => Err(self.not_folo(status)),
        }
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
        self.images_for(id, q)
    }

    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        self.image_for(dataset_id, image_id, thumb)
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

    fn export_model_with_options(&self, id: &str, opts: ModelExportOptions) -> ApiResult<ExportResult> {
        if opts.format != "onnx" {
            return self.export_model(id, opts.format);
        }
        #[derive(Serialize)]
        struct OnnxExportBody {
            format: folo_api::ExportFormat,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            decode: bool,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            dynamic: bool,
            #[serde(skip_serializing_if = "Option::is_none")]
            dynamic_batch: Option<bool>,
            #[serde(skip_serializing_if = "Option::is_none")]
            dynamic_hw: Option<bool>,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            half: bool,
        }
        let both_dynamic = opts.dynamic_batch && opts.dynamic_hw;
        let body = OnnxExportBody {
            format: folo_api::ExportFormat::Onnx,
            decode: opts.decode,
            dynamic: both_dynamic,
            dynamic_batch: (!both_dynamic && opts.dynamic_batch).then_some(true),
            dynamic_hw: (!both_dynamic && opts.dynamic_hw).then_some(true),
            half: opts.half,
        };
        self.post(&format!("/models/{}/export", seg(id)), &body)
    }

    fn is_remote(&self) -> bool {
        self.not_an_address.is_none() && !is_loopback(&self.base)
    }

    fn export_download_url(&self, id: &str, format: &str) -> Option<String> {
        if self.not_an_address.is_some() {
            return None;
        }
        let q = if format == "onnx" { "onnx" } else { "f16" };
        Some(self.url(&format!("/models/{}/weights?format={q}", seg(id))))
    }

    fn export_download_url_with_options(&self, id: &str, opts: ModelExportOptions) -> Option<String> {
        if self.not_an_address.is_some() {
            return None;
        }
        if opts.format != "onnx" {
            return self.export_download_url(id, opts.format);
        }
        // `GET /api/models/{id}/weights?format=onnx` only supports uniform `dynamic=true` (both axes).
        if opts.dynamic_batch != opts.dynamic_hw {
            return None;
        }
        let mut url = self.url(&format!("/models/{}/weights?format=onnx", seg(id)));
        if opts.decode {
            url.push_str("&decode=true");
        }
        if opts.dynamic_batch && opts.dynamic_hw {
            url.push_str("&dynamic=true");
        }
        if opts.half {
            url.push_str("&half=true");
        }
        Some(url)
    }

    fn runs(&self) -> ApiResult<Vec<Run>> {
        self.get::<folo_api::RunList>("/runs").map(|l| l.runs)
    }

    fn run(&self, id: &str) -> ApiResult<Run> {
        self.get(&format!("/runs/{}", seg(id)))
    }

    fn model_arch(&self, id: &str) -> ApiResult<Option<String>> {
        if let Some(cached) = self.model_archs.lock().unwrap().get(id).cloned() {
            return Ok(Some(cached));
        }
        let (model, header) =
            self.get_with_header::<Model>(&format!("/models/{}", seg(id)), "x-folo-arch")?;
        if let Some(arch) = header.as_deref().and_then(crate::arch::ModelArch::parse) {
            let s = arch.as_str().to_string();
            self.model_archs.lock().unwrap().insert(id.to_string(), s.clone());
            if let Some(run_id) = model.run_id {
                self.run_archs.lock().unwrap().insert(run_id, s.clone());
            }
            return Ok(Some(s));
        }
        Ok(None)
    }

    fn known_archs(&self) -> (HashMap<String, String>, HashMap<String, String>) {
        (self.run_archs.lock().unwrap().clone(), self.model_archs.lock().unwrap().clone())
    }

    fn clear_arch_cache(&self) {
        self.run_archs.lock().unwrap().clear();
        self.model_archs.lock().unwrap().clear();
    }

    fn create_run(&self, config: TrainConfig) -> ApiResult<Run> {
        self.post("/runs", &config)
    }

    fn create_run_with_arch(&self, config: TrainConfig, arch_override: Option<&str>) -> ApiResult<Run> {
        let Some(arch) = arch_override else {
            return self.create_run(config);
        };
        let mut body = serde_json::to_value(&config)
            .map_err(|e| ApiError::new("bad_request", format!("Could not encode run config: {e}")))?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("arch".into(), serde_json::Value::String(arch.into()));
        }
        let run: Run = self.post("/runs", &body)?;
        self.run_archs.lock().unwrap().insert(run.id.clone(), arch.into());
        if let Some(mid) = &run.model_id {
            self.model_archs.lock().unwrap().insert(mid.clone(), arch.into());
        }
        Ok(run)
    }

    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.post_empty(&format!("/runs/{}/stop", seg(id)))
    }

    fn delete_run(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/runs/{}", seg(id)))
    }

    fn delete_dataset(&self, id: &str, files: bool) -> ApiResult<()> {
        self.delete(&format!("/datasets/{}{}", seg(id), if files { "?files=true" } else { "" }))
    }

    fn run_cli(&self, id: &str) -> ApiResult<CliCommand> {
        let cmd: CliCommand = self.get(&format!("/runs/{}/cli", seg(id)))?;
        if let Some(arch) = crate::arch::ModelArch::from_cli_argv(&cmd.argv) {
            self.run_archs.lock().unwrap().insert(id.to_string(), arch.as_str().to_string());
        }
        Ok(cmd)
    }

    fn estimate_run(&self, config: &TrainConfig) -> ApiResult<RunEstimate> {
        self.post("/runs/estimate", config)
    }

    fn estimate_run_with_arch(
        &self,
        config: &TrainConfig,
        arch_override: Option<&str>,
    ) -> ApiResult<RunEstimate> {
        let Some(arch) = arch_override else {
            return self.estimate_run(config);
        };
        let mut body = serde_json::to_value(config)
            .map_err(|e| ApiError::new("bad_request", format!("Could not encode run config: {e}")))?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("arch".into(), serde_json::Value::String(arch.into()));
        }
        self.post("/runs/estimate", &body)
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

    fn bench_catalog(&self) -> ApiResult<BenchCatalog> {
        self.get("/bench/catalog")
    }

    fn benches(&self) -> ApiResult<Vec<Bench>> {
        self.get::<BenchList>("/bench/benchmarks").map(|l| l.benchmarks)
    }

    fn create_bench(&self, req: &CreateBench) -> ApiResult<Bench> {
        self.post_slow("/bench/benchmarks", req)
    }

    fn create_generated_bench(&self, req: &CreateGeneratedBench) -> ApiResult<Bench> {
        self.post_slow("/bench/benchmarks", req)
    }

    fn update_bench(&self, id: &str, req: &UpdateBench) -> ApiResult<Bench> {
        self.put(&format!("/bench/benchmarks/{}", seg(id)), req)
    }

    fn delete_bench(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/bench/benchmarks/{}", seg(id)))
    }

    fn design_bench(&self, id: &str, req: &DesignBench) -> ApiResult<BenchDesign> {
        self.post_slow(&format!("/bench/benchmarks/{}/design", seg(id)), req)
    }

    fn bench_images(
        &self,
        id: &str,
        offset: u32,
        limit: u32,
        condition: Option<(&str, u8)>,
    ) -> ApiResult<BenchImagePage> {
        let mut path = format!("/bench/benchmarks/{}/images?offset={offset}&limit={limit}", seg(id));
        if let Some((c, s)) = condition {
            path.push_str(&format!("&condition={}&severity={s}", seg(c)));
        }
        self.get(&path)
    }

    fn bench_image(
        &self,
        id: &str,
        image_id: &str,
        condition: Option<(&str, u8)>,
        thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        self.bench_image_for(id, image_id, condition, thumb)
    }

    fn bench_entries(&self, id: &str) -> ApiResult<Vec<BenchEntry>> {
        self.get::<BenchEntryList>(&format!("/bench/benchmarks/{}/entries", seg(id))).map(|l| l.entries)
    }

    fn bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<BenchEntry> {
        self.get(&format!("/bench/benchmarks/{}/entries/{}", seg(id), seg(entry_id)))
    }

    fn start_bench_entry(&self, id: &str, req: &StartBenchEntry) -> ApiResult<BenchEntry> {
        self.post_slow(&format!("/bench/benchmarks/{}/entries", seg(id)), req)
    }

    fn send_bench_predictions(
        &self,
        id: &str,
        entry_id: &str,
        req: &BenchPredictions,
    ) -> ApiResult<BenchEntry> {
        self.post_slow(&format!("/bench/benchmarks/{}/entries/{}/predictions", seg(id), seg(entry_id)), req)
    }

    fn finish_bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<BenchEntry> {
        let empty = serde_json::json!({});
        self.post_slow(&format!("/bench/benchmarks/{}/entries/{}/finish", seg(id), seg(entry_id)), &empty)
    }

    fn bench_result(&self, id: &str, entry_id: &str) -> ApiResult<BenchResult> {
        self.get(&format!("/bench/benchmarks/{}/entries/{}/result", seg(id), seg(entry_id)))
    }

    fn bench_entry_images(
        &self,
        id: &str,
        entry_id: &str,
        sort: &str,
        offset: u32,
        limit: u32,
    ) -> ApiResult<BenchEntryImagePage> {
        self.get(&format!(
            "/bench/benchmarks/{}/entries/{}/images?sort={}&offset={offset}&limit={limit}",
            seg(id),
            seg(entry_id),
            seg(sort)
        ))
    }

    fn stop_bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<BenchEntry> {
        self.post_empty(&format!("/bench/benchmarks/{}/entries/{}/stop", seg(id), seg(entry_id)))
    }

    fn delete_bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<()> {
        self.delete(&format!("/bench/benchmarks/{}/entries/{}", seg(id), seg(entry_id)))
    }

    fn bench_loops(&self, id: &str) -> ApiResult<Vec<BenchLoop>> {
        self.get::<BenchLoopList>(&format!("/bench/benchmarks/{}/loops", seg(id))).map(|l| l.loops)
    }

    fn start_bench_loop(&self, id: &str, req: &StartBenchLoop) -> ApiResult<BenchLoop> {
        self.post(&format!("/bench/benchmarks/{}/loops", seg(id)), req)
    }

    fn stop_bench_loop(&self, id: &str, loop_id: &str) -> ApiResult<BenchLoop> {
        self.post_empty(&format!("/bench/benchmarks/{}/loops/{}/stop", seg(id), seg(loop_id)))
    }

    fn delete_bench_loop(&self, id: &str, loop_id: &str) -> ApiResult<()> {
        self.delete(&format!("/bench/benchmarks/{}/loops/{}", seg(id), seg(loop_id)))
    }

    fn bench_events(&self, id: &str) -> ApiResult<BenchSubscription> {
        self.bench_events_for(id)
    }

    fn label_agents(&self) -> ApiResult<Vec<LabelAgent>> {
        self.get::<LabelAgentList>("/label/agents").map(|l| l.agents)
    }

    fn label_jobs(&self) -> ApiResult<Vec<LabelJob>> {
        self.get::<LabelJobList>("/label/jobs").map(|l| l.jobs)
    }

    fn create_label_job(&self, req: &CreateLabelJob) -> ApiResult<LabelJob> {
        self.create_label_job_for(req)
    }

    fn delete_label_job(&self, id: &str) -> ApiResult<()> {
        self.delete_label_job_for(id)
    }

    fn import_label_images(&self, id: &str, req: &ImportLabelImages) -> ApiResult<LabelImagesAdded> {
        self.import_label_images_for(id, req)
    }

    fn label_images(&self, id: &str, offset: u32, limit: u32) -> ApiResult<LabelImagePage> {
        self.label_images_for(id, offset, limit)
    }

    fn label_image(&self, job_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        self.label_image_for(job_id, image_id, thumb)
    }

    fn update_label_image(
        &self,
        job_id: &str,
        image_id: &str,
        req: &UpdateLabelImage,
    ) -> ApiResult<LabelImage> {
        self.update_label_image_for(job_id, image_id, req)
    }

    fn delete_label_image(&self, job_id: &str, image_id: &str) -> ApiResult<()> {
        self.delete_label_image_for(job_id, image_id)
    }

    fn start_labeling(&self, id: &str, req: &StartLabeling) -> ApiResult<LabelJob> {
        self.start_labeling_for(id, req)
    }

    fn stop_labeling(&self, id: &str) -> ApiResult<LabelJob> {
        self.stop_labeling_for(id)
    }

    fn label_logs(&self, id: &str) -> ApiResult<RunLogs> {
        self.label_logs_for(id)
    }

    fn label_events(&self, id: &str) -> ApiResult<LabelSubscription> {
        self.label_events_for(id)
    }

    fn create_label_dataset(&self, id: &str, req: &LabelDatasetRequest) -> ApiResult<Dataset> {
        self.create_label_dataset_for(id, req)
    }

    fn local_inference(&self) -> bool {
        true
    }

    fn model_weights(&self, id: &str, etag: Option<&str>) -> ApiResult<Option<Weights>> {
        self.model_weights_for(id, etag)
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

    fn logs(&self, id: &str) -> ApiResult<RunLogs> {
        self.get(&format!("/runs/{}/logs?limit=1000", seg(id)))
    }

    fn events(&self, id: &str) -> ApiResult<Subscription> {
        self.events_for(id)
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        self.predict_for(req)
    }
}

#[cfg(test)]
mod tests;
