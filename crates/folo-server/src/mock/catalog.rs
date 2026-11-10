//! The seeded catalogue: three datasets, the finished runs and their models, and the live run.

use std::sync::{Arc, RwLock};

use folo_api::{
    Arch, Compute, Head, Model, ModelMetrics, ModelPrecision, ModelSource, Splits, TrainConfig, TrainedWith,
};

use crate::util::{fnv1a, parse_rfc3339, rfc3339, unix_now};

use super::cloud::{self, CONNECTION, SEEDED_OFFER, SimCloud};
use super::datasets::{DatasetEntry, ProcDataset};
use super::models::{EvalBasis, StaticModel};
use super::runs::{Curve, LiveClock, SimRun, Timeline, arch_spec};
use super::scene::Theme;
use super::{
    CLOUD_RUN, DEVICE, MockCloud, MockStore, MockTensorCores, model_bytes, steps_per_epoch, throughput,
    weights,
};

struct RunDef {
    id: &'static str,
    name: &'static str,
    ds: usize,
    arch: Arch,
    epochs: u32,
    batch: u32,
    quality: f32,
    created: &'static str,
    end: End,
    model: Option<(&'static str, &'static str)>,
    /// `TrainConfig.amp`.
    amp: bool,
    /// `TrainConfig.head`.
    head: Head,
}

enum End {
    Completed,
    Stopped(u64),
    Failed(u64, &'static str),
}

impl MockStore {
    /// Builds the mock catalogue. The live run starts at epoch ~12 so charts have data right away. The cloud is
    /// [`ENV_MOCK_CLOUD`]'s.
    pub fn new() -> MockStore {
        MockStore::with_cloud(MockCloud::from_env())
    }

    /// [`MockStore::new`] serving `cloud`.
    pub fn with_cloud(cloud: MockCloud) -> MockStore {
        let shapes = ProcDataset::new(
            "ds_5a1e",
            "shapes-1k",
            Theme::Shapes,
            &["circle", "square", "triangle"],
            640,
            640,
            Splits { train: 800, val: 200, test: 0 },
            1,
            "2026-10-12T09:14:00Z",
            0,
        );
        let traffic = ProcDataset::new(
            "ds_7c2f",
            "traffic-cams",
            Theme::Traffic,
            &["car", "truck", "bus", "person", "bicycle"],
            1280,
            720,
            Splits { train: 1900, val: 400, test: 100 },
            2,
            "2026-10-19T14:02:00Z",
            3,
        );
        let r6 = ProcDataset::new(
            "ds_3b9d",
            "r6-operators",
            Theme::Operators,
            &["body", "Head"],
            1920,
            1080,
            Splits { train: 700, val: 160, test: 0 },
            3,
            "2026-10-27T20:41:00Z",
            0,
        );
        let datasets: Vec<Arc<DatasetEntry>> =
            [shapes, traffic, r6].into_iter().map(|d| Arc::new(DatasetEntry::Proc(d))).collect();

        let oom = "gpu:0 out of memory: tried to allocate 2.25 GiB (15.21 GiB of 15.99 GiB in use); \
                   lower --batch or --imgsz";
        let defs = [
            RunDef {
                id: "run_1a01",
                name: "shapes-n",
                ds: 0,
                arch: Arch::N,
                epochs: 100,
                batch: 16,
                quality: 0.78,
                created: "2026-10-13T08:05:00Z",
                end: End::Completed,
                model: Some(("m_31aa", "shapes-n-best")),
                amp: false,
                head: Head::O2m,
            },
            RunDef {
                id: "run_1a02",
                name: "shapes-s",
                ds: 0,
                arch: Arch::S,
                epochs: 100,
                batch: 16,
                quality: 0.84,
                created: "2026-10-14T10:30:00Z",
                end: End::Completed,
                model: Some(("m_31ab", "shapes-s-best")),
                amp: false,
                head: Head::O2o,
            },
            RunDef {
                id: "run_2b01",
                name: "traffic-s",
                ds: 1,
                arch: Arch::S,
                epochs: 120,
                batch: 16,
                quality: 0.52,
                created: "2026-10-20T07:45:00Z",
                end: End::Completed,
                model: Some(("m_42c1", "traffic-s-best")),
                amp: false,
                head: Head::O2m,
            },
            RunDef {
                id: "run_2b02",
                name: "traffic-m",
                ds: 1,
                arch: Arch::M,
                epochs: 150,
                batch: 16,
                quality: 0.6,
                created: "2026-10-22T18:10:00Z",
                end: End::Stopped(87 * 119 + 31),
                model: Some(("m_42c2", "traffic-m-last")),
                amp: false,
                head: Head::O2m,
            },
            RunDef {
                id: "run_2b03",
                name: "traffic-l",
                ds: 1,
                arch: Arch::L,
                epochs: 150,
                batch: 32,
                quality: 0.62,
                created: "2026-10-24T09:00:00Z",
                end: End::Failed(2 * 60 + 17, oom),
                model: None,
                amp: false,
                head: Head::O2m,
            },
            RunDef {
                id: "run_3c01",
                name: "r6-n",
                ds: 2,
                arch: Arch::N,
                epochs: 80,
                batch: 16,
                quality: 0.61,
                created: "2026-10-28T11:20:00Z",
                end: End::Completed,
                model: Some(("m_53d1", "r6-n-best")),
                amp: true,
                head: Head::O2m,
            },
        ];
        let mut runs = Vec::new();
        let mut models = Vec::new();
        for d in &defs {
            let entry = &datasets[d.ds];
            let meta = entry.meta();
            // The seeded cloud run (a server without the `cloud` feature has none).
            let compute = (d.id == CLOUD_RUN && cloud != MockCloud::Off).then(|| Compute {
                connection_id: CONNECTION.into(),
                offer_id: SEEDED_OFFER.into(),
                max_hours: folo_api::DEFAULT_MAX_HOURS,
                disk_gb: None,
            });
            let config = TrainConfig {
                name: d.name.into(),
                dataset_id: meta.id.clone(),
                arch: d.arch,
                epochs: d.epochs,
                batch: d.batch,
                amp: d.amp,
                head: d.head,
                compute: compute.clone(),
                ..serde_json::from_str(r#"{"dataset_id":""}"#).expect("defaults")
            };
            let spe = steps_per_epoch(meta.splits.train, d.batch);
            let curve = Curve::new(fnv1a(d.id.as_bytes()), d.quality, meta.classes.len(), &config, spe);
            let (status, last_step, error) = match d.end {
                End::Completed => (folo_api::RunStatus::Completed, curve.total_steps(), None),
                End::Stopped(s) => (folo_api::RunStatus::Stopped, s, None),
                End::Failed(s, e) => (folo_api::RunStatus::Failed, s, Some(e.to_string())),
            };
            let created = parse_rfc3339(d.created).expect("valid date");
            let offer = cloud::offers().into_iter().find(|o| o.id == SEEDED_OFFER).expect("the seeded offer");
            let device = if compute.is_some() {
                format!("NVIDIA GeForce {} (vulkan)", offer.gpu)
            } else {
                DEVICE.into()
            };
            let mut run = SimRun::new(
                d.id.into(),
                config,
                meta.classes.clone(),
                spe,
                curve,
                device,
                d.model.map(|m| m.0.to_string()),
                false,
                Timeline::Static { created, status, last_step, error },
            )
            .resolved_for(meta.splits.train);
            if let Some(c) = &compute {
                run = run.with_cloud(SimCloud::new(d.id, CONNECTION, offer, c.max_hours));
            }
            if let Some((mid, mname)) = d.model {
                let snap = run.snap();
                let best = run.curve.best(run.epochs_done(&snap)).map(|b| b.epoch).unwrap_or(1);
                let val = run.curve.val(best);
                let metrics = ModelMetrics {
                    map50: val.map50,
                    map50_95: val.map50_95,
                    precision: val.precision,
                    recall: val.recall,
                    best_f1_conf: val.best_f1_conf,
                };
                let (params, flops_g, ..) = arch_spec(d.arch);
                let finished = snap.finished.unwrap_or(created);
                let trained_with = TrainedWith { amp: d.amp, imgsz: run.config.imgsz, epochs: d.epochs };
                let base = Model {
                    id: mid.into(),
                    name: mname.into(),
                    arch: d.arch,
                    imgsz: 640,
                    classes: meta.classes.clone(),
                    params,
                    flops_g,
                    bytes: model_bytes(params, false),
                    created: rfc3339(finished),
                    run_id: Some(d.id.into()),
                    metrics: Some(metrics),
                    head: d.head,
                    trained_with: Some(trained_with),
                    // A run's checkpoint.
                    precision: Some(ModelPrecision::F32),
                    fused: Some(false),
                    source: Some(ModelSource::Run),
                    about: None,
                    path: None,
                };
                let basis = EvalBasis::new(&run, entry);
                if mid == "m_53d1" {
                    models.push(StaticModel {
                        basis: basis.clone(),
                        model: Model {
                            id: "m_53d2".into(),
                            name: "r6-n-best-f16".into(),
                            bytes: model_bytes(params, true),
                            created: rfc3339(finished + 3600.0),
                            // A `folo-f16` export: fused F16 weights, and no `folo.train` (a checkpoint key).
                            trained_with: None,
                            precision: Some(ModelPrecision::F16),
                            fused: Some(true),
                            source: Some(ModelSource::File),
                            ..base.clone()
                        },
                    });
                }
                models.push(StaticModel { model: base, basis });
            }
            runs.push((Arc::new(run), Arc::clone(entry)));
        }

        // The live run: r6-operators, folo-s, 100 epochs at 6 s each, looping.
        let r6 = &datasets[2];
        let config = TrainConfig {
            name: "r6-s".into(),
            dataset_id: r6.meta().id.clone(),
            arch: Arch::S,
            ..serde_json::from_str(r#"{"dataset_id":""}"#).expect("defaults")
        };
        let spe = steps_per_epoch(r6.meta().splits.train, config.batch);
        let curve = Curve::new(0x3c02, 0.68, 2, &config, spe);
        let (epoch_s, queue_s) = (6.0, 3.0);
        let clock = LiveClock::new(epoch_s, queue_s, 12.0, true, queue_s + 12.3 * epoch_s);
        let live = SimRun::new(
            "run_3c02".into(),
            config,
            r6.meta().classes.clone(),
            spe,
            curve,
            DEVICE.into(),
            None,
            false,
            Timeline::Live(std::sync::Mutex::new(clock)),
        )
        .resolved_for(r6.meta().splits.train);
        runs.push((Arc::new(live), Arc::clone(r6)));

        MockStore {
            datasets: RwLock::new(datasets),
            models: RwLock::new(models),
            runs: RwLock::new(runs),
            queue: std::sync::Mutex::default(),
            weights: weights::MockWeights::new(),
            tensor_cores: MockTensorCores::from_env(),
            throughput: crate::throughput::Throughput::in_memory(throughput::table()),
            connections: RwLock::new(match cloud {
                MockCloud::On => vec![cloud::seeded(unix_now())],
                _ => Vec::new(),
            }),
            cloud,
        }
    }
}
