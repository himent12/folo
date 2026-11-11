//! [`Backend`] over HTTP + SSE against `folo serve` (docs/API.md), and a launcher for a local
//! `folo serve`.

mod images;
mod inference;
mod server;
mod streams;
mod transport;

use std::time::Duration;

use serde::Serialize;

use super::*;

pub use server::{ServerProcess, Spawned, spawn_server};
use transport::{agent, seg, stream_agent};

/// How long an event stream may take to send its response headers.
const STREAM_HANDSHAKE: Duration = Duration::from_secs(5);

pub struct HttpBackend {
    base: String,
    agent: ureq::Agent,
    /// No overall timeout (SSE streams stay open for the life of a run), but the handshake is
    /// bounded: connect and response headers.
    stream_agent: ureq::Agent,
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

    fn delete_run(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/runs/{}", seg(id)))
    }

    fn delete_dataset(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/datasets/{}", seg(id)))
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

    fn bench_catalog(&self) -> ApiResult<BenchCatalog> {
        self.get("/bench/catalog")
    }

    fn benches(&self) -> ApiResult<Vec<Bench>> {
        self.get::<BenchList>("/bench/benchmarks").map(|l| l.benchmarks)
    }

    fn create_bench(&self, req: &CreateBench) -> ApiResult<Bench> {
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

    fn start_bench_entry(&self, id: &str, req: &StartBenchEntry) -> ApiResult<BenchEntry> {
        self.post_slow(&format!("/bench/benchmarks/{}/entries", seg(id)), req)
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

    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>> {
        self.get::<folo_api::RunLogs>(&format!("/runs/{}/logs?limit=1000", seg(id))).map(|l| l.lines)
    }

    fn events(&self, id: &str) -> ApiResult<Subscription> {
        self.events_for(id)
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        self.predict_for(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Every benchmark call the app makes, against a real `folo serve` with a benchmark that has a
    /// finished entry and a model: `FOLO_SERVER=… cargo test real_server_bench -- --ignored
    /// --nocapture`. It makes a benchmark, starts and stops an entry and deletes both again.
    #[test]
    #[ignore]
    fn real_server_bench_calls() {
        let api = HttpBackend::new(&std::env::var("FOLO_SERVER").unwrap_or("http://127.0.0.1:7878".into()));
        let features = api.system().unwrap().features;
        assert!(features.iter().any(|f| f == FEATURE_VISION_BENCH), "{features:?}");
        let catalog = api.bench_catalog().unwrap();
        assert_eq!(catalog.conditions.len(), 12);
        let benches = api.benches().unwrap();
        let (b, done) = benches
            .iter()
            .find_map(|b| {
                let entries = api.bench_entries(&b.id).unwrap();
                entries.into_iter().find(|e| e.status == BenchEntryStatus::Done).map(|e| (b.clone(), e))
            })
            .expect("a benchmark with a finished entry");
        let result = api.bench_result(&b.id, &done.id).unwrap();
        println!("{} / {}: score {:?}", b.name, done.name, result.entry.score.map(|s| s.total));
        let page = api.bench_entry_images(&b.id, &done.id, "worst", 0, 24).unwrap();
        assert!(!page.images.is_empty() && page.total >= page.images.len() as u64);
        let bad: Vec<u32> = page.images.iter().map(|i| i.missed + i.wrong).collect();
        assert!(bad.windows(2).all(|w| w[0] >= w[1]), "{bad:?}");
        let first = api.bench_images(&b.id, 0, 1, None).unwrap().images.remove(0);
        let decode = |d: ImageData| match d {
            ImageData::Encoded(bytes) => image::load_from_memory(&bytes).unwrap().to_rgb8(),
            ImageData::Rgba { .. } => panic!("the server sends files"),
        };
        let plain = decode(api.bench_image(&b.id, &first.id, None, Some(320)).unwrap());
        let cond = b.plan.conditions.first().map(|c| (c.condition.clone(), c.severities[0]));
        if let Some((c, s)) = &cond {
            let changed = decode(api.bench_image(&b.id, &first.id, Some((c, *s)), Some(320)).unwrap());
            assert_eq!(plain.dimensions(), changed.dimensions());
            assert_ne!(plain.as_raw(), changed.as_raw(), "{c} {s} changes the pixels");
        }
        assert!(plain.width().max(plain.height()) <= 320);
        let _ = api.bench_loops(&b.id).unwrap();
        let _ = api.label_agents().unwrap();
        let _ = api.label_jobs().unwrap();
        drop(api.bench_events(&b.id).unwrap());
        // The write path, on a benchmark of its own.
        let ds = api
            .datasets()
            .unwrap()
            .into_iter()
            .find(|d| d.splits.val > 0)
            .expect("a dataset with val images");
        let created = api
            .create_bench(&CreateBench {
                name: "desktop real-server check".into(),
                description: String::new(),
                source: BenchSource::Dataset { dataset_id: ds.id.clone(), split: Some(Split::Val) },
                max_images: Some(8),
                seed: 0,
                plan: None,
            })
            .unwrap();
        // Whatever happens next, the benchmark goes again.
        let run = || -> ApiResult<()> {
            let model = api
                .models()?
                .into_iter()
                .find(|m| crate::bench::model_fits(&m.classes, &created.classes))
                .expect("a model that knows the dataset's classes");
            let req = StartBenchEntry {
                contestant: BenchContestantSpec::Model { model_id: model.id },
                name: None,
                max_images: None,
                conditions: Some(false),
            };
            let events = api.bench_events(&created.id)?;
            let e = api.start_bench_entry(&created.id, &req)?;
            // The entry comes over the stream too (its progress, or its end).
            let streamed = std::iter::from_fn(|| events.recv_timeout(Duration::from_secs(20)).ok())
                .any(|ev| matches!(ev, BenchEvent::Entry(x) if x.id == e.id));
            assert!(streamed, "the entry is streamed");
            if e.status.is_active() {
                let _ = api.stop_bench_entry(&created.id, &e.id);
            }
            for _ in 0..100 {
                if api.bench_entries(&created.id)?.iter().all(|x| !x.status.is_active()) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            api.delete_bench_entry(&created.id, &e.id)
        };
        let ran = run();
        api.delete_bench(&created.id).unwrap();
        ran.unwrap();
        assert!(!api.benches().unwrap().iter().any(|x| x.id == created.id));
    }
}
