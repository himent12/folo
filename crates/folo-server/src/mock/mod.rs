//! `MockStore`: deterministic, realistic data for UI work before (or without) the engine.
//!
//! Three procedural datasets, six models, six finished runs (one stopped, one failed) with full metric histories,
//! and one live run that advances in real time (100 epochs in ~10 minutes, then it loops). Runs created through the
//! API are simulated the same way. Datasets registered from disk are scanned for real.

mod catalog;
mod cloud;
mod datasets;
mod models;
mod runs;
mod scene;
mod store;
mod throughput;
mod val;
mod weights;

use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use folo_api::{
    AmpSupport, Capabilities, CloudConnection, CloudInstance, CloudOffer, CloudProvider, Compute, GpuBackend,
    Model, ModelMetrics, ModelPrecision, ModelSource, TensorCoreSupport, TrainConfig, TrainedWith,
};

use crate::error::{Error, Result};
use crate::store::Store;
use crate::util::rfc3339;
use datasets::DatasetEntry;
use models::StaticModel;
use runs::{SimRun, arch_spec};

pub use cloud::{ENV_MOCK_CLOUD, MockCloud, PHASE_S};
pub use runs::TICK;

const DEVICE: &str = "NVIDIA GeForce RTX 4080 SUPER (vulkan)";

/// The mock store. See the module docs.
pub struct MockStore {
    datasets: RwLock<Vec<Arc<DatasetEntry>>>,
    /// Unregistered datasets (canonical path for one from disk, format, id), as real mode's tombstones: the same
    /// folder registered again gets its id back, and no other dataset ever gets it.
    retired: RwLock<Vec<(Option<std::path::PathBuf>, folo_api::DatasetFormat, String)>>,
    models: RwLock<Vec<StaticModel>>,
    runs: RwLock<Vec<(Arc<SimRun>, Arc<DatasetEntry>)>>,
    /// The runs created through the API, in queue order: they share one training slot ([`runs::reschedule`]).
    queue: std::sync::Mutex<Vec<Arc<SimRun>>>,
    weights: weights::MockWeights,
    tensor_cores: MockTensorCores,
    throughput: crate::throughput::Throughput,
    /// How much of the cloud it serves, and its connections.
    cloud: MockCloud,
    connections: RwLock<Vec<CloudConnection>>,
}

/// Environment variable that picks the mock GPU's tensor-core state ([`MockTensorCores`]), so front ends can test
/// every state of the Mixed precision switch: `on` (the default), `off` or `none` (also `0` / `false`).
pub const ENV_MOCK_TENSOR_CORES: &str = "FOLO_MOCK_TENSOR_CORES";

/// The mock GPU's tensor cores, as `/api/system`'s `capabilities` reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MockTensorCores {
    /// The RTX 4080 SUPER through Vulkan with `--tensor-cores`: tensor cores and `amp` available.
    #[default]
    On,
    /// The same adapter without `--tensor-cores`: available but not enabled, so no `amp`.
    Off,
    /// The adapter through DX12, where wgpu has no cooperative matrices: neither tensor cores nor `amp`.
    None,
}

impl MockTensorCores {
    /// [`ENV_MOCK_TENSOR_CORES`], or [`MockTensorCores::On`] when it is unset or unrecognised.
    pub fn from_env() -> MockTensorCores {
        Self::parse(&std::env::var(ENV_MOCK_TENSOR_CORES).unwrap_or_default())
    }

    /// An [`ENV_MOCK_TENSOR_CORES`] value (case-insensitive); anything unrecognised is [`MockTensorCores::On`].
    pub fn parse(s: &str) -> MockTensorCores {
        match s.trim().to_ascii_lowercase().as_str() {
            "off" => MockTensorCores::Off,
            "none" | "0" | "false" | "no" => MockTensorCores::None,
            _ => MockTensorCores::On,
        }
    }

    fn backend(self) -> GpuBackend {
        if self == MockTensorCores::None { GpuBackend::Dx12 } else { GpuBackend::Vulkan }
    }

    /// The capabilities the real server reports in this state.
    fn capabilities(self) -> Capabilities {
        let status = match self {
            MockTensorCores::On => None,
            MockTensorCores::Off => Some(folo_gpu::context::TENSOR_CORES_NOT_REQUESTED.to_string()),
            MockTensorCores::None => {
                Some(format!("off: {MOCK_GPU} (Dx12) has no cooperative matrices in wgpu"))
            }
        };
        let amp = match &status {
            None => AmpSupport { available: true, reason: None },
            Some(s) => AmpSupport {
                available: false,
                reason: Some(
                    folo_vision::train::amp_not_supported(&format!("gpu:0 has tensor cores {s}")).to_string(),
                ),
            },
        };
        Capabilities {
            device: "gpu:0".into(),
            shader_f16: true,
            tensor_cores: TensorCoreSupport {
                available: self != MockTensorCores::None,
                enabled: status.is_none(),
                reason: status,
            },
            amp,
        }
    }
}

/// What `folo train --weights` would take for mock model `id`: a published name as it is, else a made-up file under
/// the mock data dir (mock models have no files).
fn mock_weights(id: &str) -> String {
    if folo_api::weights::find(id).is_some() { id.to_string() } else { format!("~/.folo/models/{id}.folo") }
}

/// The mock GPU.
const MOCK_GPU: &str = "NVIDIA GeForce RTX 4080 SUPER";

/// The scripted live run (it loops forever, so it does not block benchmark probes).
const LIVE_RUN: &str = "run_3c02";

/// The seeded cloud run (finished, released).
const CLOUD_RUN: &str = "run_3c01";

impl Default for MockStore {
    fn default() -> Self {
        MockStore::new()
    }
}

fn steps_per_epoch(train: u64, batch: u32) -> u64 {
    train.div_ceil(batch.max(1) as u64).max(1)
}

fn model_bytes(params: u64, f16: bool) -> u64 {
    params * if f16 { 2 } else { 4 } + 18_432
}

impl MockStore {
    /// The providers, unless the mock serves no cloud.
    fn providers(&self) -> Result<Vec<CloudProvider>> {
        if self.cloud == MockCloud::Off {
            return Err(cloud::off());
        }
        Ok(folo_cloud::providers().iter().map(|p| p.info()).collect())
    }

    /// Connection `id`.
    fn connection(&self, id: &str) -> Result<CloudConnection> {
        self.providers()?;
        let list = self.connections.read().unwrap();
        list.iter().find(|c| c.id == id).cloned().ok_or_else(|| crate::cloud::connection_not_found(id))
    }

    /// What connection `c` can run on: RunPod's list, or a remote server's own GPU.
    fn offers_of(c: &CloudConnection) -> Vec<CloudOffer> {
        match c.provider.as_str() {
            "remote" => vec![CloudOffer {
                id: "gpu".into(),
                gpu: "NVIDIA GeForce RTX 3090".into(),
                gpu_count: 1,
                vram_bytes: 24 << 30,
                price_per_hour: None,
                region: None,
                detail: None,
                availability: folo_api::CloudAvailability::High,
            }],
            _ => cloud::offers(),
        }
    }

    /// The offer a run with `compute` rents, checked as the real store checks it.
    fn cloud_offer(&self, config: &TrainConfig, compute: &Compute) -> Result<CloudOffer> {
        crate::cloud::check_device(config)?;
        crate::cloud::check_limits(compute)?;
        let c = self.connection(&compute.connection_id)?;
        crate::cloud::find_offer(MockStore::offers_of(&c), compute)
    }

    /// Every cloud run's instance.
    fn instances(&self) -> Vec<CloudInstance> {
        let mut all: Vec<CloudInstance> =
            self.runs.read().unwrap().iter().filter_map(|(r, _)| r.instance()).collect();
        all.sort_by(|a, b| {
            (b.status.is_alive(), &b.created, &b.id).cmp(&(a.status.is_alive(), &a.created, &a.id))
        });
        all
    }

    /// The tensor-core state `/api/system`'s `capabilities` reports (default: [`MockTensorCores::from_env`]).
    pub fn with_tensor_cores(mut self, tensor_cores: MockTensorCores) -> MockStore {
        self.tensor_cores = tensor_cores;
        self
    }

    fn dataset(&self, id: &str) -> Result<Arc<DatasetEntry>> {
        self.datasets
            .read()
            .unwrap()
            .iter()
            .find(|d| d.meta().id == id)
            .cloned()
            .ok_or_else(|| Error::not_found("dataset", id))
    }

    fn run(&self, id: &str) -> Result<(Arc<SimRun>, Arc<DatasetEntry>)> {
        self.runs
            .read()
            .unwrap()
            .iter()
            .find(|r| r.0.id == id)
            .cloned()
            .ok_or_else(|| Error::not_found("run", id))
    }

    fn all_models(&self) -> Vec<Model> {
        let mut out: Vec<Model> = self.models.read().unwrap().iter().map(|m| m.model.clone()).collect();
        for (run, ds) in self.runs.read().unwrap().iter().filter(|r| r.0.user) {
            let snap = run.snap();
            let Some(id) = run.current_model(&snap) else { continue };
            let best = run.curve.best(run.epochs_done(&snap)).map_or(1, |b| b.epoch);
            let val = run.curve.val(best);
            let (params, flops_g, ..) = arch_spec(run.config.arch);
            out.push(Model {
                id,
                name: format!("{}-best", run.config.name),
                arch: run.config.arch,
                imgsz: run.config.imgsz,
                classes: ds.meta().classes.clone(),
                params,
                flops_g,
                bytes: model_bytes(params, false),
                created: rfc3339(snap.finished.unwrap_or(snap.created)),
                run_id: Some(run.id.clone()),
                metrics: Some(ModelMetrics {
                    map50: val.map50,
                    map50_95: val.map50_95,
                    precision: val.precision,
                    recall: val.recall,
                    best_f1_conf: val.best_f1_conf,
                }),
                head: run.config.head,
                trained_with: Some(TrainedWith {
                    amp: run.config.amp,
                    imgsz: run.config.imgsz,
                    epochs: run.config.epochs,
                }),
                precision: Some(ModelPrecision::F32),
                fused: Some(false),
                source: Some(ModelSource::Run),
                about: None,
                path: None,
            });
        }
        out.sort_by(|a, b| b.created.cmp(&a.created).then(a.id.cmp(&b.id)));
        out
    }

    fn validate(&self, c: &TrainConfig) -> Result<()> {
        let bad = |field: &str, why: &str| {
            Err(Error::new(StatusCode::BAD_REQUEST, "invalid_config", format!("{field}: {why}")))
        };
        if !(1..=10_000).contains(&c.epochs) {
            return bad("epochs", "must be in 1..=10000");
        }
        if !(1..=1024).contains(&c.batch) {
            return bad("batch", "must be in 1..=1024");
        }
        if c.imgsz < 32 || c.imgsz > 4096 || !c.imgsz.is_multiple_of(32) {
            return bad("imgsz", "must be a multiple of 32 in 32..=4096");
        }
        if !(c.lr0.is_finite() && c.lr0 > 0.0 && c.lr0 < 10.0) {
            return bad("lr0", "must be in (0, 10)");
        }
        if c.optimizer == folo_api::Optimizer::Unknown {
            return bad("optimizer", "unknown value (this build knows auto, sgd, adamw and musgd)");
        }
        let gpu_ok = c.device.strip_prefix("gpu:").is_some_and(|n| n.parse::<u32>().is_ok());
        if !(c.device == "auto" || c.device == "cpu" || gpu_ok) {
            return bad("device", "must be auto, cpu or gpu:<n>");
        }
        if let Some(w) = &c.weights {
            self.get_model(w)?;
        }
        Ok(())
    }
}

fn quality_of(model: &Model) -> f32 {
    model.metrics.map_or(0.5, |m| m.map50)
}

impl MockStore {
    /// `(subscribers, ticker running)` of a run's event hub. For tests that check streams do not leak.
    #[doc(hidden)]
    pub fn hub_state(&self, id: &str) -> (usize, bool) {
        let (run, _) = self.run(id).unwrap();
        (run.subscribers(), run.ticking())
    }

    /// Lets `d` pass at once for every live run, as if the tests waited for it: their clocks move ahead (queue
    /// waits, stops, cloud phases and epochs), and streams report the change on their next tick.
    #[doc(hidden)]
    pub fn pass_time(&self, d: std::time::Duration) {
        for (run, _) in self.runs.read().unwrap().iter() {
            run.pass_time(d);
        }
    }
}
