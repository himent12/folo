//! [`Backend`] for the mock: every call answered from the fixtures and the simulated clock.

use std::sync::Arc;
use std::time::Instant;

use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::cloud::{
    CloudSim, Ended, cloud_providers, launch_state, no_connection, no_instance, offers_of, secret_hint,
    unencrypted,
};
use super::predictions::fake_predictions;
use super::throughput::with_reference;
use super::training::{Curve, RunFixture, epoch_line, initial_logs, resolve};
use super::{CPU_DEVICE, GPU_DEVICE, MOCK_NOW, MockBackend, STEP_DT};

impl Backend for MockBackend {
    fn describe(&self) -> String {
        // The failing variant stands in for an unreachable folo serve (the error scenario): it
        // names the address the app tried, as the HTTP backend would.
        if self.fail { "127.0.0.1:7878".into() } else { "Built-in demo".into() }
    }

    fn now(&self) -> i64 {
        self.sim_now()
    }

    fn health(&self) -> ApiResult<Health> {
        self.check()?;
        Ok(Health {
            status: "ok".into(),
            version: "0.1.0".into(),
            api: folo_api::API_VERSION.into(),
            mock: true,
        })
    }

    fn system(&self) -> ApiResult<SystemResponse> {
        self.check()?;
        let t = self.started.elapsed().as_secs_f32();
        let wobble = if self.frozen { 0.0 } else { (t * 0.7).sin() * 0.02 };
        // A run on this machine's GPU (a cloud run trains elsewhere).
        let live = !self.empty
            && self
                .runs
                .lock()
                .unwrap()
                .iter()
                .any(|f| f.cloud.is_none() && Self::snapshot(f).status == RunStatus::Running);
        let system = SystemInfo {
            cpu: CpuInfo {
                name: "AMD Ryzen 9 7950X".into(),
                threads: 32,
                util: if live { 0.31 + wobble } else { 0.04 },
            },
            memory: MemoryInfo { used_bytes: 23_100_000_000, total_bytes: 68_719_476_736 },
            gpus: vec![GpuInfo {
                ordinal: 0,
                name: "NVIDIA GeForce RTX 4080 SUPER".into(),
                backend: GpuBackend::Vulkan,
                memory_bytes: 17_171_480_576,
                util: Some(if live { 0.94 + wobble } else { 0.02 }),
            }],
        };
        // The demo's simulated GPU has tensor cores, so Mixed precision reads as this server's (a
        // run trains the same simulated curve either way).
        let capabilities = Capabilities {
            device: "gpu:0".into(),
            shader_f16: true,
            tensor_cores: TensorCoreSupport { available: true, enabled: true, reason: None },
            amp: AmpSupport { available: true, reason: None },
        };
        let mut features = vec![
            "queue_control".into(),
            "weights_onnx".into(),
            "run_estimate".into(),
            "run_cli".into(),
            "throughput".into(),
            "benchmark".into(),
            folo_api::FEATURE_CLOUD.into(),
        ];
        if self.benches.enabled {
            features.push(folo_api::FEATURE_VISION_BENCH.into());
            features.push(folo_api::FEATURE_AGENT_LABELING.into());
        }
        Ok(SystemResponse {
            system,
            data_dir: None,
            features,
            capabilities: Some(capabilities),
            gpu_pool: None,
        })
    }

    fn datasets(&self) -> ApiResult<Vec<Dataset>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        let list = self
            .datasets
            .iter()
            .filter(|d| !self.is_removed(&d.ds.id))
            .map(|d| Dataset { stats: None, ..d.ds.clone() })
            .collect();
        self.hold_point("datasets");
        Ok(list)
    }

    fn dataset(&self, id: &str) -> ApiResult<Dataset> {
        self.check()?;
        Ok(self.fixture(id)?.ds.clone())
    }

    fn dataset_issues(&self, id: &str) -> ApiResult<IssuePage> {
        self.check()?;
        // The broken labels split as folo dataset check would: most are errors (lines the loader
        // skips), a third warnings (boxes clipped to the image).
        let broken = self.fixture(id)?.ds.stats.as_ref().map_or(0, |s| s.broken_labels);
        let warning = broken / 3;
        let error = broken - warning;
        let mut counts = IssueCounts { error, warning, info: 0, by_kind: Default::default() };
        if error > 0 {
            counts.by_kind.insert("class_out_of_range".into(), error);
        }
        if warning > 0 {
            counts.by_kind.insert("box_out_of_bounds".into(), warning);
        }
        Ok(IssuePage { total: broken, issues: Vec::new(), counts })
    }

    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        self.check()?;
        {
            let mut fail = self.fail_images_from.lock().unwrap();
            if fail.is_some_and(|from| q.offset >= from) {
                *fail = None;
                return Err(ApiError::new(
                    "server_unreachable",
                    "Could not reach folo serve at 127.0.0.1:7878",
                ));
            }
        }
        let f = self.fixture(id)?;
        let matching: Vec<&ImageInfo> = f
            .images
            .iter()
            .filter(|im| q.split.is_none_or(|s| im.split == s))
            .filter(|im| q.class.is_none_or(|c| im.boxes.iter().any(|b| b.class == c)))
            .collect();
        let limit = if q.limit == 0 { 100 } else { q.limit };
        let page = ImagePage {
            total: matching.len() as u64,
            images: matching.into_iter().skip(q.offset as usize).take(limit as usize).cloned().collect(),
        };
        self.hold_point("images");
        Ok(page)
    }

    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        let f = self.fixture(dataset_id)?;
        let index: u32 = image_id
            .strip_prefix("img_")
            .and_then(|n| n.parse::<u32>().ok())
            .and_then(|n| n.checked_sub(1))
            .filter(|&i| (i as usize) < f.images.len())
            .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
        let meta = &f.images[index as usize];
        let (mut w, mut h) = (meta.width, meta.height);
        if let Some(t) = thumb {
            let s = t as f32 / w.max(h) as f32;
            if s < 1.0 {
                w = ((w as f32 * s).round() as u32).max(1);
                h = ((h as f32 * s).round() as u32).max(1);
            }
        }
        let key = (dataset_id.to_string(), image_id.to_string(), w);
        if let Some(px) = self.image_cache.lock().unwrap().get(&key) {
            return Ok(ImageData::Rgba { width: w, height: h, pixels: px.to_vec() });
        }
        let pixels = f.scene(index).render(w, h);
        let mut cache = self.image_cache.lock().unwrap();
        if cache.len() > 600 {
            cache.clear();
        }
        cache.insert(key, Arc::new(pixels.clone()));
        Ok(ImageData::Rgba { width: w, height: h, pixels })
    }

    fn models(&self) -> ApiResult<Vec<Model>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        let list = self
            .models
            .iter()
            .filter(|m| !m.run_id.as_ref().is_some_and(|r| self.is_removed(r)))
            .cloned()
            .collect();
        self.hold_point("models");
        Ok(list)
    }

    fn model_eval(&self, id: &str) -> ApiResult<ModelEval> {
        self.eval_for(id)
    }

    fn export_model(&self, id: &str, format: &str) -> ApiResult<ExportResult> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {id}")))?;
        let (ext, bytes) = match format {
            "onnx" => ("onnx", m.bytes + 120_000),
            _ => ("f16.folo", m.bytes / 2),
        };
        Ok(ExportResult { path: format!("D:/folo/models/{}.{ext}", m.name), bytes })
    }

    fn runs(&self) -> ApiResult<Vec<Run>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        let list = self.runs.lock().unwrap().iter().map(Self::snapshot).collect();
        self.hold_point("runs");
        Ok(list)
    }

    fn run(&self, id: &str) -> ApiResult<Run> {
        self.check()?;
        self.runs
            .lock()
            .unwrap()
            .iter()
            .find(|f| f.run.id == id)
            .map(Self::snapshot)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))
    }

    fn create_run(&self, config: TrainConfig) -> ApiResult<Run> {
        self.check()?;
        let ds = self.fixture(&config.dataset_id)?;
        // A cloud run is checked like any run plus its compute (docs/API.md, Cloud runs).
        let target = match &config.compute {
            Some(c) => {
                let (conn, offer) = self.compute_offer(c, &config)?;
                if offer.availability == CloudAvailability::None {
                    return Err(ApiError::new(
                        "cloud_refused",
                        format!("{} has no {} available right now", conn.name, crate::ui::offer_line(&offer)),
                    )
                    .with_hint("Pick another GPU, or try again in a few minutes."));
                }
                Some((c.clone(), conn, offer))
            }
            None => None,
        };
        let mut runs = self.runs.lock().unwrap();
        let id = format!("run_{:04x}", hash(&format!("{}{}", config.name, runs.len())) & 0xFFFF);
        let spe = (ds.ds.splits.train / config.batch.max(1) as u64).max(1);
        let curve = Curve {
            seed: hash(&id),
            epochs: config.epochs,
            spe,
            lr0: resolve(&config, ds.ds.splits.train, ds.ds.classes.len()).lr0 as f32,
            batch: config.batch,
            map_max: 0.66,
            ips: 380.0,
            classes: ds.ds.classes.len() as u32,
            close_mosaic: config.augment.close_mosaic,
        };
        let now = self.sim_now();
        let mut f = RunFixture {
            run: Run {
                id: id.clone(),
                name: config.name.clone(),
                status: RunStatus::Running,
                created: time::format(now),
                started: Some(time::format(now)),
                finished: None,
                paused_s: None,
                epoch: 0,
                epochs: config.epochs,
                step: 0,
                steps_per_epoch: spe,
                eta_s: None,
                best: None,
                device: if config.device == "cpu" { CPU_DEVICE.into() } else { GPU_DEVICE.into() },
                model_id: None,
                error: None,
                error_code: None,
                resolved: None,
                queue_position: None,
                cloud: None,
                dataset: Some(RunDataset { name: ds.ds.name.clone(), classes: ds.ds.classes.clone() }),
                config,
            },
            curve,
            live: (!self.frozen && target.is_none()).then(|| (0, Instant::now())),
            step: 0,
            logs: Vec::new(),
            cloud: None,
        };
        if let Some((compute, conn, offer)) = target {
            let base = launch_state(&id, &compute, &conn.provider, offer, conn.provider == "runpod");
            let mut sim = CloudSim::new(base, self.clock, self.clock.now(), f.run.epochs as u64 * spe);
            // The dataset as FOLO sends it: about 180 kB an image.
            sim.upload_bytes = ds.ds.splits.total() * 180_000;
            f.cloud = Some(sim);
        }
        // A local run's lines follow its start; a cloud run's its launch, now.
        f.logs = initial_logs(&f, if f.cloud.is_some() { now } else { now - 5 });
        let run = Self::snapshot(&f);
        runs.insert(0, f);
        Ok(run)
    }

    fn run_cli(&self, id: &str) -> ApiResult<CliCommand> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        Ok(self.cli_for(&f.run.config))
    }

    fn throughput(&self) -> ApiResult<ThroughputTable> {
        self.check()?;
        let (training, inference) = self.measured.lock().unwrap().clone();
        Ok(with_reference(training, inference))
    }

    fn benchmark(&self, req: &BenchmarkRequest) -> ApiResult<BenchmarkReport> {
        self.benchmark_for(req)
    }

    fn estimate_run(&self, config: &TrainConfig) -> ApiResult<RunEstimate> {
        self.estimate_for(config)
    }

    fn queue_run(&self, id: &str, _req: QueueRequest) -> ApiResult<Run> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        if f.run.status != RunStatus::Queued {
            return Err(ApiError::new("run_not_queued", "This run is no longer waiting in the queue"));
        }
        // The demo queues one run at a time: every move leaves it first.
        f.run.queue_position = Some(1);
        Ok(f.run.clone())
    }

    fn delete_run(&self, id: &str) -> ApiResult<()> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let i = runs
            .iter()
            .position(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        // A queued run that never started leaves the queue and is deleted, as on the server.
        let snap = Self::snapshot(&runs[i]);
        if snap.status.is_active() && !snap.never_started() {
            return Err(ApiError::new("run_active", format!("run '{id}' is still active"))
                .with_hint(format!("stop it first: POST /api/runs/{id}/stop")));
        }
        runs.remove(i);
        self.removed.lock().unwrap().insert(id.to_string());
        Ok(())
    }

    fn delete_dataset(&self, id: &str, files: bool) -> ApiResult<()> {
        self.check()?;
        self.fixture(id)?;
        // The demo's datasets are folders of its own, none generated into a data dir (it has none).
        if files {
            return Err(ApiError::new(
                "invalid_query",
                "Only a dataset FOLO generated can go with its files",
            ));
        }
        self.removed.lock().unwrap().insert(id.to_string());
        Ok(())
    }

    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        if !Self::snapshot(f).status.is_active() {
            return Err(ApiError::new("run_not_active", "This run is not running"));
        }
        // A cloud run stops on its instance: its checkpoint is collected and the instance released
        // (at once before training).
        if let Some(sim) = f.cloud.as_mut() {
            sim.ended = Some((sim.elapsed(), Ended::Stopped));
            f.logs.push(LogLine {
                t: time::format(self.sim_now()),
                level: LogLevel::Warn,
                message: "stop requested; collecting the checkpoint and releasing the instance".into(),
            });
            return Ok(Self::snapshot(f));
        }
        f.step = Self::current_step(f);
        f.live = None;
        f.run.status = RunStatus::Stopped;
        f.run.finished = Some(time::format(self.sim_now()));
        let epoch = f.step / f.run.steps_per_epoch;
        f.logs.push(LogLine {
            t: time::format(self.sim_now()),
            level: LogLevel::Warn,
            message: "stop requested; finishing current step".into(),
        });
        f.logs.push(LogLine {
            t: time::format(self.sim_now()),
            level: LogLevel::Info,
            message: format!("saved last checkpoint (epoch {epoch})"),
        });
        Ok(Self::snapshot(f))
    }

    fn metrics(&self, id: &str) -> ApiResult<RunMetrics> {
        self.check()?;
        if self.fail_metrics.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878"));
        }
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        let step = Self::current_step(f);
        let metrics = RunMetrics {
            steps: (1..=step).map(|i| f.curve.step(i)).collect(),
            epochs: (1..=(step / f.run.steps_per_epoch) as u32).map(|e| f.curve.epoch(e)).collect(),
        };
        drop(runs);
        // One run's history ("metrics run_52e8"): other runs' (a live one's) are not held.
        self.hold_point(&format!("metrics {id}"));
        Ok(metrics)
    }

    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<RunSamples> {
        self.check()?;
        self.hold_point(&format!("samples {id}"));
        let (curve, dataset_id, done) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (
                f.curve.clone(),
                f.run.config.dataset_id.clone(),
                (Self::current_step(f) / f.run.steps_per_epoch) as u32,
            )
        };
        let epoch = epoch.unwrap_or(done).min(done);
        let q = (curve.quality(epoch) * curve.map_max / 0.75).clamp(0.05, 1.0);
        let class_names = self.fixture(&dataset_id).map(|f| f.ds.classes.clone()).unwrap_or_default();
        let samples = self
            .val_images(&dataset_id)
            .into_iter()
            .take(16)
            .map(|(i, im)| Sample {
                image_id: im.id.clone(),
                dataset_id: dataset_id.clone(),
                width: im.width,
                height: im.height,
                truth: im.boxes.clone(),
                pred: fake_predictions(
                    &im.boxes,
                    im.width,
                    im.height,
                    q,
                    &class_names,
                    hash(id) ^ i as u64 ^ (epoch as u64) << 20,
                ),
            })
            .collect();
        Ok(RunSamples { epoch, samples })
    }

    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        // Epoch summaries up to now, merged with the fixture's lines in time order (a cloud run's
        // from when it started training on its instance).
        let created = time::parse(&f.run.created).unwrap_or(MOCK_NOW);
        let trains_at = created + f.cloud.as_ref().map_or(10, |s| s.train_at() as i64 + 10);
        let done = (Self::current_step(f) / f.run.steps_per_epoch) as u32;
        let per_epoch = (f.run.steps_per_epoch as f64 * STEP_DT) as i64 + 1;
        let mut lines = f.logs.clone();
        let from = done.saturating_sub(60) + 1;
        for e in from..=done {
            lines.push(LogLine {
                t: time::format(trains_at + e as i64 * per_epoch),
                level: LogLevel::Info,
                message: epoch_line(e, f.run.epochs, &f.curve.epoch(e)),
            });
        }
        lines.sort_by(|a, b| a.t.cmp(&b.t));
        Ok(lines)
    }

    fn events(&self, id: &str) -> ApiResult<Subscription> {
        if self.fail_events.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878"));
        }
        self.events_for(id)
    }

    fn cloud_providers(&self) -> ApiResult<Vec<CloudProvider>> {
        self.check()?;
        Ok(cloud_providers())
    }

    fn cloud_connections(&self) -> ApiResult<Vec<CloudConnection>> {
        self.check()?;
        Ok(self.connections.lock().unwrap().clone())
    }

    fn connect_cloud(&self, req: &ConnectCloud) -> ApiResult<CloudConnection> {
        self.check()?;
        let provider = cloud_providers().into_iter().find(|p| p.id == req.provider).ok_or_else(|| {
            ApiError::new("provider_not_found", format!("No cloud provider {}", req.provider))
                .with_hint("GET /api/cloud/providers lists them.")
        })?;
        let value = |key: &str| req.values.get(key).map(|v| v.trim()).filter(|v| !v.is_empty());
        if let Some(f) = provider.fields.iter().find(|f| f.required && value(&f.key).is_none()) {
            return Err(ApiError::new("invalid_body", format!("{} is missing", f.label)));
        }
        // The demo's provider rejects any secret containing "bad".
        if provider.fields.iter().any(|f| f.secret && value(&f.key).is_some_and(|v| v.contains("bad"))) {
            return Err(ApiError::new(
                "cloud_auth_failed",
                format!("{} rejected these credentials", provider.name),
            )
            .with_hint("Check that the key is current and allowed to manage pods."));
        }
        let now = time::format(self.sim_now());
        let url = value("url").map(str::to_string);
        let mut conns = self.connections.lock().unwrap();
        let base =
            if req.name.trim().is_empty() { provider.name.clone() } else { req.name.trim().to_string() };
        let name = (1..)
            .map(|n| if n == 1 { base.clone() } else { format!("{base} {n}") })
            .find(|n| !conns.iter().any(|c| &c.name == n))
            .unwrap_or(base);
        let conn = CloudConnection {
            id: format!("cc_{:04x}", hash(&format!("{name}{}", conns.len())) & 0xFFFF),
            provider: provider.id.clone(),
            name,
            created: now.clone(),
            status: CloudConnectionStatus::Ok,
            account: match &url {
                Some(u) => Some(u.split("://").nth(1).unwrap_or(u).trim_end_matches('/').to_string()),
                None => Some("you@example.com".into()),
            },
            balance_usd: provider.provisions.then_some(42.17),
            error: None,
            note: url.as_deref().filter(|u| unencrypted(u)).map(|_| {
                "Unencrypted http://: the token and the data cross the network in the clear. Use https:// \
                 (a TLS proxy or a tunnel)."
                    .to_string()
            }),
            checked: Some(now),
            values: provider
                .fields
                .iter()
                .filter_map(|f| {
                    let v = value(&f.key)?;
                    Some((f.key.clone(), if f.secret { secret_hint(v) } else { v.to_string() }))
                })
                .collect(),
        };
        conns.push(conn.clone());
        Ok(conn)
    }

    fn check_cloud(&self, id: &str) -> ApiResult<CloudConnection> {
        self.check()?;
        let mut conns = self.connections.lock().unwrap();
        let c = conns.iter_mut().find(|c| c.id == id).ok_or_else(|| no_connection(id))?;
        c.checked = Some(time::format(self.sim_now()));
        Ok(c.clone())
    }

    fn disconnect_cloud(&self, id: &str) -> ApiResult<()> {
        self.check()?;
        let name = self.connection(id)?.name;
        let in_use = self.runs.lock().unwrap().iter().any(|f| {
            f.cloud.as_ref().is_some_and(|s| s.base.connection_id == id)
                && (Self::snapshot(f).status.is_active()
                    || f.cloud
                        .as_ref()
                        .and_then(|s| s.instance(&f.run.id))
                        .is_some_and(|i| i.status.is_alive()))
        });
        if in_use {
            return Err(ApiError::new(
                "connection_in_use",
                format!("{name} has a live instance or an active run"),
            )
            .with_hint("Terminate its instances in Settings › Cloud, or wait for its runs to end."));
        }
        self.connections.lock().unwrap().retain(|c| c.id != id);
        Ok(())
    }

    fn cloud_offers(&self, connection: &str, _refresh: bool) -> ApiResult<CloudOfferList> {
        self.check()?;
        let conn = self.connection(connection)?;
        Ok(CloudOfferList { offers: offers_of(&conn)?, fetched: time::format(self.sim_now()) })
    }

    fn cloud_instances(&self) -> ApiResult<Vec<CloudInstance>> {
        self.check()?;
        if self.fail_instances.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878"));
        }
        if self.empty {
            return Ok(Vec::new());
        }
        let week_ago = time::format(self.sim_now() - 7 * 86_400);
        let mut list: Vec<CloudInstance> = self
            .runs
            .lock()
            .unwrap()
            .iter()
            .filter_map(|f| f.cloud.as_ref()?.instance(&f.run.id))
            .filter(|i| i.ended.as_ref().is_none_or(|e| *e >= week_ago))
            .collect();
        // Alive first, then the ended ones; newest first within each.
        list.sort_by(|a, b| b.status.is_alive().cmp(&a.status.is_alive()).then(b.created.cmp(&a.created)));
        self.hold_point("instances");
        Ok(list)
    }

    fn terminate_instance(&self, id: &str) -> ApiResult<CloudInstance> {
        self.check()?;
        let now = time::format(self.sim_now());
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.cloud.as_ref().is_some_and(|s| s.base.instance_id.as_deref() == Some(id)))
            .ok_or_else(|| no_instance(id))?;
        let run_id = f.run.id.clone();
        if let Some(sim) =
            f.cloud.as_mut().filter(|s| s.instance(&run_id).is_some_and(|i| i.status.is_alive()))
        {
            let at = sim.elapsed();
            // The provider confirms at once; a run still using it fails.
            sim.release_error = None;
            if sim.released_at().is_none() {
                sim.ended = Some((at, Ended::InstanceLost));
                f.logs.push(LogLine {
                    t: now,
                    level: LogLevel::Error,
                    message: format!("cloud: instance {id} was terminated by hand"),
                });
            }
        }
        f.cloud.as_ref().and_then(|s| s.instance(&run_id)).ok_or_else(|| no_instance(id))
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        let r = self.predict_for(req);
        self.hold_point("predict");
        r
    }

    // ---------- vision benchmarks (deterministic fixtures, `mock_bench`) ----------

    fn bench_catalog(&self) -> ApiResult<BenchCatalog> {
        self.check()?;
        self.benches.catalog()
    }

    fn benches(&self) -> ApiResult<Vec<Bench>> {
        self.check()?;
        if self.empty && self.benches.enabled {
            return Ok(Vec::new());
        }
        self.benches.list(&self.datasets, &self.models)
    }

    fn create_bench(&self, req: &CreateBench) -> ApiResult<Bench> {
        self.check()?;
        self.benches.create(req, &self.datasets, &self.models)
    }

    fn update_bench(&self, id: &str, req: &UpdateBench) -> ApiResult<Bench> {
        self.check()?;
        self.benches.update(id, req, &self.datasets, &self.models)
    }

    fn delete_bench(&self, id: &str) -> ApiResult<()> {
        self.check()?;
        self.benches.delete(id, &self.datasets, &self.models)
    }

    fn design_bench(&self, id: &str, req: &DesignBench) -> ApiResult<BenchDesign> {
        self.check()?;
        self.benches.design(id, req, &self.datasets, &self.models)
    }

    fn bench_images(
        &self,
        id: &str,
        offset: u32,
        limit: u32,
        condition: Option<(&str, u8)>,
    ) -> ApiResult<BenchImagePage> {
        self.check()?;
        self.benches.images(id, offset, limit, condition, &self.datasets, &self.models)
    }

    fn bench_image(
        &self,
        id: &str,
        image_id: &str,
        condition: Option<(&str, u8)>,
        thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        self.check()?;
        self.benches.image_pixels(id, image_id, condition, thumb, &self.datasets, &self.models)
    }

    fn bench_entries(&self, id: &str) -> ApiResult<Vec<BenchEntry>> {
        self.check()?;
        self.benches.entries(id, &self.datasets, &self.models)
    }

    fn start_bench_entry(&self, id: &str, req: &StartBenchEntry) -> ApiResult<BenchEntry> {
        self.check()?;
        self.benches.start_entry(id, req, &self.datasets, &self.models)
    }

    fn bench_result(&self, id: &str, entry_id: &str) -> ApiResult<BenchResult> {
        self.check()?;
        self.benches.result(id, entry_id, &self.datasets, &self.models)
    }

    fn bench_entry_images(
        &self,
        id: &str,
        entry_id: &str,
        sort: &str,
        offset: u32,
        limit: u32,
    ) -> ApiResult<BenchEntryImagePage> {
        self.check()?;
        self.benches.entry_images(id, entry_id, sort, offset, limit, &self.datasets, &self.models)
    }

    fn stop_bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<BenchEntry> {
        self.check()?;
        self.benches.stop_entry(id, entry_id, &self.datasets, &self.models)
    }

    fn delete_bench_entry(&self, id: &str, entry_id: &str) -> ApiResult<()> {
        self.check()?;
        self.benches.delete_entry(id, entry_id, &self.datasets, &self.models)
    }

    fn bench_loops(&self, id: &str) -> ApiResult<Vec<BenchLoop>> {
        self.check()?;
        self.benches.loops(id, &self.datasets, &self.models)
    }

    fn start_bench_loop(&self, id: &str, req: &StartBenchLoop) -> ApiResult<BenchLoop> {
        self.check()?;
        self.benches.start_loop(id, req, &self.datasets, &self.models)
    }

    fn stop_bench_loop(&self, id: &str, loop_id: &str) -> ApiResult<BenchLoop> {
        self.check()?;
        self.benches.stop_loop(id, loop_id, &self.datasets, &self.models)
    }

    fn delete_bench_loop(&self, id: &str, loop_id: &str) -> ApiResult<()> {
        self.check()?;
        self.benches.delete_loop(id, loop_id, &self.datasets, &self.models)
    }

    fn bench_events(&self, id: &str) -> ApiResult<BenchSubscription> {
        self.check()?;
        crate::api::mock_bench::MockBench::events(
            self.benches.clone(),
            id,
            self.datasets.clone(),
            self.models.clone(),
        )
    }

    fn label_agents(&self) -> ApiResult<Vec<LabelAgent>> {
        self.check()?;
        self.benches.agents(&self.datasets, &self.models)
    }

    fn label_jobs(&self) -> ApiResult<Vec<LabelJob>> {
        self.check()?;
        self.benches.jobs(&self.datasets, &self.models)
    }
}
