//! The demo's contents: datasets, models, and finished, failed, stopped, queued, live and cloud runs.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::cloud::{
    A100_OFFER, COLLECT_S, CloudSim, L40S_OFFER, RELEASE_S, RTX4090_OFFER, RUNPOD_CONNECTION, launch_state,
    runpod_connection, runpod_offers,
};
use super::datasets::DatasetFixture;
use super::throughput::mock_measured;
use super::training::{Clock, Curve, RunFixture, initial_logs, resolve};
use super::{CLOUD_SYNCING_ID, CLOUD_TRAINING_ID, CPU_DEVICE, GPU_DEVICE, LIVE_ID, MOCK_NOW, MockBackend};

impl MockBackend {
    pub(super) fn build(live_step: u32, frozen: bool, cloud_runs: bool) -> Self {
        let h = |hours: f64| MOCK_NOW - (hours * 3600.0) as i64;
        let datasets = vec![
            DatasetFixture::new(
                "ds_8f2c",
                "shapes",
                "D:/data/shapes",
                "yolo",
                &["circle", "square", "triangle"],
                (800, 200, 0),
                6,
                false,
                &[(640, 640)],
                h(70.0),
            ),
            DatasetFixture::new(
                "ds_3b91",
                "shapes-hard",
                "D:/data/shapes-hard",
                "coco",
                &["circle", "square", "triangle", "star", "hexagon"],
                (3200, 800, 400),
                12,
                true,
                &[(640, 480), (800, 600), (480, 640), (640, 640)],
                h(30.0),
            ),
            DatasetFixture::new(
                "ds_c07a",
                "shapes-mini",
                "D:/data/shapes-mini",
                "yolo",
                &["circle", "square", "triangle"],
                (240, 60, 0),
                4,
                false,
                &[(320, 320)],
                h(4.0),
            ),
            DatasetFixture::new(
                "ds_5e21",
                "shapes-crowded",
                "D:/data/shapes-crowded",
                "yolo",
                &["circle", "square", "triangle", "star", "hexagon"],
                (1600, 400, 200),
                24,
                false,
                &[(640, 640), (800, 600)],
                h(200.0),
            ),
            DatasetFixture::new(
                "ds_a3d0",
                "shapes-tiny",
                "D:/data/shapes-tiny",
                "yolo",
                &["circle", "square", "triangle"],
                (1200, 300, 0),
                10,
                true,
                &[(512, 512)],
                h(120.0),
            ),
        ];
        let classes = |i: usize| datasets[i].ds.classes.clone();
        let model = |id: &str,
                     name: &str,
                     arch: &str,
                     imgsz,
                     ds: usize,
                     params,
                     flops_g,
                     bytes,
                     hours,
                     run: &str,
                     m: [f32; 4]| Model {
            id: id.into(),
            name: name.into(),
            arch: Arch::parse(arch).unwrap_or_default(),
            imgsz,
            classes: classes(ds),
            params,
            flops_g,
            bytes,
            created: time::format(h(hours)),
            run_id: Some(run.into()),
            metrics: Some(ModelMetrics {
                map50: m[0],
                map50_95: m[1],
                precision: m[2],
                recall: m[3],
                // Where F1 peaks: lower for the stronger models (their scores sit higher).
                best_f1_conf: Some(((0.42 - 0.18 * m[1]) * 100.).round() / 100.),
            }),
            head: folo_api::Head::O2m,
            trained_with: None,
            precision: Some(folo_api::ModelPrecision::F32),
            fused: Some(false),
            source: Some(folo_api::ModelSource::Run),
            about: None,
            path: None,
        };
        let mut models = vec![
            model(
                "m_44c0",
                "shapes-hard-s-best",
                "folo-s",
                640,
                1,
                9_420_000,
                21.5,
                18_900_000,
                6.5,
                "run_52e8",
                [0.874, 0.624, 0.861, 0.792],
            ),
            model(
                "m_31aa",
                "shapes-n-best",
                "folo-n",
                640,
                0,
                2_630_000,
                6.6,
                5_400_000,
                52.0,
                "run_7d1e",
                [0.952, 0.781, 0.928, 0.903],
            ),
            model(
                "m_31ab",
                "shapes-n-f16",
                "folo-n",
                640,
                0,
                2_630_000,
                6.6,
                2_700_000,
                51.5,
                "run_7d1e",
                [0.951, 0.780, 0.927, 0.903],
            ),
            model(
                "m_90d2",
                "shapes-mini-n-last",
                "folo-n",
                320,
                2,
                2_630_000,
                1.7,
                5_400_000,
                3.2,
                "run_1a2b",
                [0.803, 0.544, 0.817, 0.716],
            ),
            model(
                "m_7c11",
                "shapes-crowded-s-best",
                "folo-s",
                640,
                3,
                9_420_000,
                21.5,
                18_900_000,
                128.0,
                "run_c9f1",
                [0.801, 0.553, 0.790, 0.742],
            ),
            model(
                "m_7c12",
                "shapes-tiny-n-best",
                "folo-n",
                512,
                4,
                2_630_000,
                4.3,
                5_400_000,
                98.0,
                "run_d204",
                [0.661, 0.402, 0.712, 0.584],
            ),
        ];
        // shapes-hard-s-best comes from the end-to-end, mixed-precision run below: its record says so.
        if let Some(m) = models.iter_mut().find(|m| m.id == "m_44c0") {
            m.head = folo_api::Head::O2o;
            m.trained_with = Some(folo_api::TrainedWith { amp: true, imgsz: 640, epochs: 120 });
        }
        // shapes-n-f16 is the fused f16 export of shapes-n-best.
        if let Some(m) = models.iter_mut().find(|m| m.id == "m_31ab") {
            m.precision = Some(folo_api::ModelPrecision::F16);
            m.fused = Some(true);
        }

        let cfg = |name: &str, ds: &str, arch: &str, epochs, batch, imgsz, device: &str, lr0| TrainConfig {
            name: name.into(),
            dataset_id: ds.into(),
            arch: Arch::parse(arch).unwrap_or_default(),
            epochs,
            batch,
            imgsz,
            device: device.into(),
            optimizer: Optimizer::Auto,
            lr0,
            weights: None,
            augment: Augment::default(),
            seed: 0,
            amp: false,
            head: Default::default(),
            compute: None,
        };
        // Trains on a cloud connection's offer (the cloud fixtures below).
        let on = |offer: &str, max_hours: f32, c: TrainConfig| TrainConfig {
            compute: Some(Compute {
                connection_id: RUNPOD_CONNECTION.into(),
                offer_id: offer.into(),
                max_hours,
                disk_gb: None,
            }),
            device: "auto".into(),
            ..c
        };
        let gpu = GPU_DEVICE.to_string();
        struct Spec {
            id: &'static str,
            config: TrainConfig,
            status: RunStatus,
            step: u64,
            spe: u64,
            created_h: f64,
            map_max: f32,
            ips: f32,
            model: Option<&'static str>,
            error: Option<&'static str>,
        }
        let mut specs = vec![
            Spec {
                id: LIVE_ID,
                config: cfg("shapes-hard-m", "ds_3b91", "folo-m", 100, 32, 640, "gpu:0", 0.01),
                status: RunStatus::Running,
                step: live_step as u64,
                spe: 100,
                created_h: 0.25,
                map_max: 0.71,
                ips: 268.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_b7e2",
                config: cfg("shapes-mini-s", "ds_c07a", "folo-s", 60, 16, 320, "gpu:0", 0.01),
                status: RunStatus::Queued,
                step: 0,
                spe: 19,
                created_h: 0.1,
                map_max: 0.6,
                ips: 900.0,
                model: None,
                error: None,
            },
            Spec {
                id: "run_52e8",
                // The end-to-end head (no NMS) in mixed precision, so its model shows both.
                config: TrainConfig {
                    head: Head::O2o,
                    amp: true,
                    ..cfg("shapes-hard-s", "ds_3b91", "folo-s", 120, 32, 640, "gpu:0", 0.01)
                },
                status: RunStatus::Completed,
                step: 120 * 100,
                spe: 100,
                created_h: 9.0,
                map_max: 0.612,
                ips: 412.0,
                model: Some("m_44c0"),
                error: None,
            },
            Spec {
                id: "run_e4f0",
                config: cfg("shapes-hard-l", "ds_3b91", "folo-l", 100, 64, 1024, "gpu:0", 0.01),
                status: RunStatus::Failed,
                step: 112,
                spe: 50,
                created_h: 10.5,
                map_max: 0.7,
                ips: 96.0,
                model: None,
                error: Some(
                    "GPU out of memory: tried to allocate 2.1 GiB with 0.6 GiB free (batch 64, imgsz 1024)",
                ),
            },
            Spec {
                id: "run_1a2b",
                config: cfg("shapes-mini-n", "ds_c07a", "folo-n", 100, 16, 320, "cpu", 0.01),
                status: RunStatus::Stopped,
                step: 37 * 15,
                spe: 15,
                created_h: 4.0,
                map_max: 0.58,
                ips: 140.0,
                model: Some("m_90d2"),
                error: None,
            },
            Spec {
                id: "run_7d1e",
                config: cfg("shapes-n", "ds_8f2c", "folo-n", 100, 16, 640, "gpu:0", 0.01),
                status: RunStatus::Completed,
                step: 100 * 50,
                spe: 50,
                created_h: 55.0,
                map_max: 0.781,
                ips: 610.0,
                model: Some("m_31aa"),
                error: None,
            },
            // The demo's one cloud run (docs/CLOUD.md), released long ago with its final cost.
            Spec {
                id: "run_c9f1",
                config: on(
                    A100_OFFER,
                    12.,
                    cfg("shapes-crowded-s", "ds_5e21", "folo-s", 80, 32, 640, "auto", 0.01),
                ),
                status: RunStatus::Completed,
                step: 80 * 50,
                spe: 50,
                created_h: 131.0,
                map_max: 0.553,
                ips: 398.0,
                model: Some("m_7c11"),
                error: None,
            },
            Spec {
                id: "run_d204",
                config: cfg("shapes-tiny-n", "ds_a3d0", "folo-n", 60, 32, 512, "gpu:0", 0.01),
                status: RunStatus::Completed,
                step: 60 * 38,
                spe: 38,
                created_h: 101.0,
                map_max: 0.402,
                ips: 720.0,
                model: Some("m_7c12"),
                error: None,
            },
            Spec {
                id: "run_0b5e",
                config: cfg("shapes-crowded-m", "ds_5e21", "folo-m", 120, 32, 800, "gpu:0", 0.01),
                status: RunStatus::Stopped,
                step: 23 * 50,
                spe: 50,
                created_h: 30.0,
                map_max: 0.6,
                ips: 241.0,
                model: None,
                error: None,
            },
        ];
        if cloud_runs {
            // Among the newest, after the live and the queued run.
            specs.splice(
                2..2,
                [
                    Spec {
                        id: CLOUD_SYNCING_ID,
                        config: on(
                            L40S_OFFER,
                            24.,
                            cfg("shapes-crowded-l", "ds_5e21", "folo-l", 120, 32, 800, "auto", 0.01),
                        ),
                        status: RunStatus::Queued,
                        step: 0,
                        spe: 50,
                        created_h: 0.1,
                        map_max: 0.62,
                        ips: 120.0,
                        model: None,
                        error: None,
                    },
                    Spec {
                        id: CLOUD_TRAINING_ID,
                        // The out-of-memory run above, again on a 24 GB GPU.
                        config: on(
                            RTX4090_OFFER,
                            12.,
                            cfg("shapes-hard-l-2", "ds_3b91", "folo-l", 100, 32, 1024, "auto", 0.01),
                        ),
                        status: RunStatus::Running,
                        step: 4400,
                        spe: 100,
                        created_h: 2.4,
                        map_max: 0.7,
                        ips: 96.0,
                        model: None,
                        error: None,
                    },
                ],
            );
        }
        let now = Instant::now();
        // Each run resolved against its own dataset's class count (Auto's AdamW rate depends on it).
        let nc = |ds: &str| datasets.iter().find(|d| d.ds.id == ds).map_or(5, |d| d.ds.classes.len());
        let mut runs = specs
            .into_iter()
            .map(|s| {
                let curve = Curve {
                    seed: hash(s.id),
                    epochs: s.config.epochs,
                    spe: s.spe,
                    // The rate the run resolves to (Auto's pick), not the form's.
                    lr0: resolve(&s.config, s.spe * s.config.batch as u64, nc(&s.config.dataset_id)).lr0
                        as f32,
                    batch: s.config.batch,
                    map_max: s.map_max,
                    ips: s.ips,
                    classes: datasets
                        .iter()
                        .find(|d| d.ds.id == s.config.dataset_id)
                        .map_or(3, |d| d.ds.classes.len() as u32),
                    close_mosaic: s.config.augment.close_mosaic,
                };
                let created = h(s.created_h);
                let total = s.config.epochs as u64 * s.spe;
                let run_secs = |steps: u64| (steps as f64 * (s.config.batch as f64 / s.ips as f64)) as i64;
                let finished =
                    matches!(s.status, RunStatus::Completed | RunStatus::Stopped | RunStatus::Failed).then(
                        || time::format(created + 5 + run_secs(s.step) + s.step as i64 / s.spe as i64 * 3),
                    );
                let started = (s.status != RunStatus::Queued).then(|| time::format(created + 5));
                let mut fixture = RunFixture {
                    run: Run {
                        id: s.id.into(),
                        name: s.config.name.clone(),
                        status: s.status,
                        created: time::format(created),
                        started,
                        finished,
                        epoch: 0,
                        epochs: s.config.epochs,
                        step: s.step,
                        steps_per_epoch: s.spe,
                        eta_s: None,
                        best: None,
                        device: if s.config.device == "cpu" { CPU_DEVICE.into() } else { gpu.clone() },
                        model_id: s.model.map(Into::into),
                        error: s.error.map(Into::into),
                        error_code: s.error.map(|_| "training_failed".into()),
                        resolved: (s.status != RunStatus::Queued).then(|| {
                            resolve(&s.config, s.spe * s.config.batch as u64, nc(&s.config.dataset_id))
                        }),
                        queue_position: (s.status == RunStatus::Queued).then_some(1),
                        cloud: None,
                        dataset: None,
                        config: s.config,
                    },
                    curve,
                    live: (s.status == RunStatus::Running && !frozen).then_some((s.step, now)),
                    step: s.step.min(total),
                    logs: Vec::new(),
                    cloud: None,
                };
                fixture.logs = initial_logs(&fixture, created);
                fixture
            })
            .collect::<Vec<RunFixture>>();
        // The cloud fixtures live on their instances: released 5 days ago after training as long as
        // it would have here (its Duration reads the same), 6 minutes into uploading 4.8 GiB, and 2.4
        // hours into training (at step 4,400).
        let clock = Clock { frozen, started: now };
        let gib = 1u64 << 30;
        for f in &mut runs {
            let Some(compute) = f.run.config.compute.clone() else { continue };
            let t0 = time::parse(&f.run.created).unwrap_or(MOCK_NOW);
            let total = f.run.epochs as u64 * f.run.steps_per_epoch;
            let offer = runpod_offers().into_iter().find(|o| o.id == compute.offer_id);
            let Some(offer) = offer else { continue };
            let mut sim = CloudSim::new(
                launch_state(&f.run.id, &compute, "runpod", offer, true),
                clock,
                t0 as f64,
                total,
            );
            match f.run.id.as_str() {
                CLOUD_SYNCING_ID => (sim.pre, sim.upload_bytes) = ([40., 150., 680.], gib * 24 / 5),
                CLOUD_TRAINING_ID => {
                    (sim.pre, sim.upload_bytes) = ([45., 230., 290.], gib * 21 / 10);
                    sim.anchor = ((MOCK_NOW - t0) as f64, f.step);
                }
                _ => {
                    let at = |t: &Option<String>| t.as_deref().and_then(time::parse).unwrap_or(t0);
                    let trained = (at(&f.run.finished) - at(&f.run.started)) as f64 - COLLECT_S - RELEASE_S;
                    (sim.pre, sim.upload_bytes) = ([45., 240., 360.], gib * 13 / 10);
                    sim.step_dt = trained / total as f64;
                }
            }
            // Unless anchored mid-run, training starts from step 0 when the upload ends.
            if sim.anchor.1 == 0 {
                sim.anchor.0 = sim.train_at();
            }
            f.cloud = Some(sim);
            f.live = None;
            f.run.queue_position = None;
            f.logs = initial_logs(f, t0);
        }
        // A model's operating point is its run's best epoch's (as `folo serve --mock`'s), so the run and model
        // pages name the same confidence.
        for model in &mut models {
            let Some(f) = runs.iter().find(|f| Some(&*f.run.id) == model.run_id.as_deref()) else { continue };
            let (Some(best), Some(metrics)) = (Self::snapshot(f).best, model.metrics.as_mut()) else {
                continue;
            };
            metrics.best_f1_conf = f.curve.epoch(best.epoch).val.best_f1_conf;
        }
        let fixture_runs = runs.iter().map(|f| f.run.id.clone()).collect();
        Self {
            datasets,
            models,
            fixture_runs,
            measured: Mutex::new(mock_measured()),
            runs: Arc::new(Mutex::new(runs)),
            frozen,
            started: now,
            image_cache: Mutex::new(HashMap::new()),
            fail: false,
            empty: false,
            connections: Mutex::new(vec![runpod_connection()]),
            benches: Arc::new(crate::api::mock_bench::MockBench::new(frozen, now)),
            clock,
        }
    }
}
