//! In-process fixtures behind [`Backend`]: three synthetic datasets, a handful of models, finished,
//! failed, stopped and queued runs, and one live run whose metrics advance in real time.
//!
//! Everything is a pure function of seeds and the simulated clock, so a frozen clock gives
//! pixel-identical screenshots.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::time;
use super::*;
use crate::synth::{Rng, Scene, hash};

/// The fixtures' "now": 2026-11-08T07:30:00Z.
pub const MOCK_NOW: i64 = 1_794_123_000;

/// Seconds per training step of a live mock run.
const STEP_DT: f64 = 0.12;

struct DatasetFixture {
    ds: Dataset,
    seed: u64,
    max_objects: u32,
    small: bool,
    sizes: &'static [(u32, u32)],
    /// Per-image boxes, computed once from the scenes.
    images: Vec<ImageMeta>,
}

impl DatasetFixture {
    fn new(
        id: &str,
        name: &str,
        path: &str,
        format: &str,
        classes: &[&str],
        splits: (u32, u32, u32),
        max_objects: u32,
        small: bool,
        sizes: &'static [(u32, u32)],
        created: i64,
    ) -> Self {
        let seed = hash(id);
        let total = splits.0 + splits.1 + splits.2;
        let mut images = Vec::with_capacity(total as usize);
        for i in 0..total {
            let split = if i < splits.0 {
                Split::Train
            } else if i < splits.0 + splits.1 {
                Split::Val
            } else {
                Split::Test
            };
            let (w, h) = sizes[(i as usize * 7 + 3) % sizes.len()];
            let scene = Scene::generate(seed ^ (i as u64).wrapping_mul(0x9E37), classes.len() as u32, max_objects, small);
            images.push(ImageMeta {
                id: format!("img_{i:05}"),
                file: format!("images/{}/{i:06}.jpg", split.label()),
                split,
                width: w,
                height: h,
                boxes: scene.boxes(w, h),
            });
        }
        let mut class_counts = vec![0u32; classes.len()];
        let size_edges = vec![0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
        let mut size_counts = vec![0u32; size_edges.len() - 1];
        let per_edges = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut per_counts = vec![0u32; per_edges.len() - 1];
        let mut boxes = 0;
        let mut empty = 0;
        for im in &images {
            boxes += im.boxes.len() as u32;
            if im.boxes.is_empty() {
                empty += 1;
            }
            let n = im.boxes.len() as f32;
            let slot = per_edges.windows(2).position(|w| n >= w[0] && n < w[1]).unwrap_or(per_counts.len() - 1);
            per_counts[slot] += 1;
            for b in &im.boxes {
                class_counts[b.class as usize] += 1;
                let area = (b.x2 - b.x1) * (b.y2 - b.y1) / (im.width * im.height) as f32;
                let slot = size_edges.windows(2).position(|w| area >= w[0] && area < w[1]).unwrap_or(size_counts.len() - 1);
                size_counts[slot] += 1;
            }
        }
        let ds = Dataset {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            format: format.into(),
            classes: classes.iter().map(|c| c.to_string()).collect(),
            created: time::format(created),
            splits: Splits { train: splits.0, val: splits.1, test: splits.2 },
            stats: Some(DatasetStats {
                boxes,
                empty_images: empty,
                broken_labels: if small { 3 } else { 0 },
                class_counts,
                box_size_hist: Hist { edges: size_edges, counts: size_counts },
                boxes_per_image_hist: Hist { edges: per_edges, counts: per_counts },
            }),
        };
        Self { ds, seed, max_objects, small, sizes, images }
    }

    fn scene(&self, index: u32) -> Scene {
        Scene::generate(
            self.seed ^ (index as u64).wrapping_mul(0x9E37),
            self.ds.classes.len() as u32,
            self.max_objects,
            self.small,
        )
    }
}

/// Shape of a run's simulated learning curves.
#[derive(Clone)]
struct Curve {
    seed: u64,
    epochs: u32,
    spe: u32,
    lr0: f32,
    batch: u32,
    map_max: f32,
    ips: f32,
    classes: u32,
    close_mosaic: u32,
}

impl Curve {
    fn noise(&self, i: u32, k: u64) -> f32 {
        Rng::new(self.seed ^ (i as u64).wrapping_mul(0x2545_F491) ^ k).normal()
    }

    fn lr(&self, frac: f32) -> f32 {
        let warm = 3.0;
        if frac < warm {
            self.lr0 * (0.1 + 0.9 * frac / warm)
        } else {
            let t = ((frac - warm) / (self.epochs as f32 - warm)).clamp(0.0, 1.0);
            self.lr0 * (0.01 + 0.99 * 0.5 * (1.0 + (std::f32::consts::PI * t).cos()))
        }
    }

    fn loss_at(&self, frac: f32) -> Loss {
        let e = self.epochs as f32;
        let closing = (frac - (e - self.close_mosaic as f32)).clamp(0.0, 1.0);
        let k = 1.0 - 0.1 * closing;
        let box_ = (0.92 + 1.38 * (-frac / (e * 0.2)).exp()) * k;
        let cls = (0.40 + 2.7 * (-frac / (e * 0.1)).exp()) * k;
        let dfl = (0.86 + 0.58 * (-frac / (e * 0.24)).exp()) * k;
        Loss { box_, cls, dfl, total: box_ + cls + dfl }
    }

    fn step(&self, i: u32) -> Step {
        let frac = i as f32 / self.spe as f32;
        let base = self.loss_at(frac);
        let jitter = |k: u64| 1.0 + 0.045 * self.noise(i, k);
        let box_ = base.box_ * jitter(1);
        let cls = base.cls * jitter(2);
        let dfl = base.dfl * jitter(3);
        let ips = self.ips * (1.0 + 0.025 * self.noise(i, 4));
        let remaining = (self.epochs * self.spe).saturating_sub(i) as f64;
        Step {
            epoch: (i.max(1) - 1) / self.spe + 1,
            step: i,
            lr: self.lr(frac),
            loss: Loss { box_, cls, dfl, total: box_ + cls + dfl },
            imgs_per_s: ips,
            eta_s: Some(remaining * STEP_DT),
        }
    }

    fn quality(&self, epoch: u32) -> f32 {
        let e = epoch as f32;
        let n = self.epochs as f32;
        let closing = (e - (n - self.close_mosaic as f32)).clamp(0.0, self.close_mosaic as f32) / self.close_mosaic as f32;
        ((1.0 - (-e / (n * 0.17)).exp()) * (1.0 + 0.012 * self.noise(epoch, 9)) + 0.025 * closing).min(1.02)
    }

    fn epoch(&self, epoch: u32) -> Epoch {
        let q = self.quality(epoch);
        let m = (self.map_max * q).clamp(0.0, 0.99);
        let map50 = (m * 1.22 + 0.05 * (1.0 - m)).min(0.985);
        let per_class = (0..self.classes)
            .map(|c| {
                let d = 1.0 + 0.09 * Rng::new(self.seed ^ c as u64).normal() - 0.03 * c as f32;
                let cm = (m * d).clamp(0.0, 0.99);
                ClassMetrics {
                    class: c,
                    map50: (cm * 1.22 + 0.05 * (1.0 - cm)).min(0.99),
                    map50_95: cm,
                    precision: (0.42 + 0.56 * cm / self.map_max.max(0.01) * 0.93).min(0.99),
                    recall: (0.35 + 0.6 * cm / self.map_max.max(0.01) * 0.9).min(0.99),
                }
            })
            .collect();
        Epoch {
            epoch,
            train_loss: self.loss_at(epoch as f32 - 0.5),
            val: ValMetrics {
                map50,
                map50_95: m,
                precision: (0.42 + 0.52 * q).min(0.98),
                recall: (0.33 + 0.56 * q).min(0.97),
                per_class,
            },
            duration_s: (self.spe * self.batch) as f32 / self.ips + 2.4,
        }
    }
}

struct RunFixture {
    run: Run,
    curve: Curve,
    /// Live runs: steps done when the clock started, and when that was.
    live: Option<(u32, Instant)>,
    /// Frozen at this step (finished, stopped, failed, or a frozen clock).
    step: u32,
    logs: Vec<LogLine>,
}

pub struct MockBackend {
    datasets: Vec<DatasetFixture>,
    models: Vec<Model>,
    runs: Mutex<Vec<RunFixture>>,
    frozen: bool,
    started: Instant,
    image_cache: Mutex<HashMap<(String, String, u32), Arc<Vec<u8>>>>,
    /// Inject an error into every list call (error-state screenshots).
    fail: bool,
    /// Pretend there is nothing on disk yet (empty-state screenshots).
    empty: bool,
}

const LIVE_ID: &str = "run_a19f";

impl MockBackend {
    /// `live_step`: where the live run starts. `frozen`: keep the clock still (screenshots).
    pub fn new(live_step: u32, frozen: bool) -> Self {
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
        ];
        let classes = |i: usize| datasets[i].ds.classes.clone();
        let model = |id: &str, name: &str, arch: &str, imgsz, ds: usize, params, flops_g, bytes, hours, run: &str, m: [f32; 4]| Model {
            id: id.into(),
            name: name.into(),
            arch: arch.into(),
            imgsz,
            classes: classes(ds),
            params,
            flops_g,
            bytes,
            created: time::format(h(hours)),
            run_id: Some(run.into()),
            metrics: Some(ModelMetrics { map50: m[0], map50_95: m[1], precision: m[2], recall: m[3] }),
        };
        let models = vec![
            model("m_44c0", "shapes-hard-s-best", "folo-s", 640, 1, 9_420_000, 21.5, 18_900_000, 6.5, "run_52e8", [0.874, 0.612, 0.861, 0.792]),
            model("m_31aa", "shapes-n-best", "folo-n", 640, 0, 2_630_000, 6.6, 5_400_000, 52.0, "run_7d1e", [0.952, 0.781, 0.928, 0.903]),
            model("m_31ab", "shapes-n-f16", "folo-n", 640, 0, 2_630_000, 6.6, 2_700_000, 51.5, "run_7d1e", [0.951, 0.780, 0.927, 0.903]),
            model("m_90d2", "shapes-mini-n-last", "folo-n", 320, 2, 2_630_000, 1.7, 5_400_000, 3.2, "run_1a2b", [0.803, 0.544, 0.817, 0.716]),
        ];

        let cfg = |name: &str, ds: &str, arch: &str, epochs, batch, imgsz, device: &str, lr0| TrainConfig {
            name: name.into(),
            dataset_id: ds.into(),
            arch: arch.into(),
            epochs,
            batch,
            imgsz,
            device: device.into(),
            optimizer: "auto".into(),
            lr0,
            weights: None,
            augment: Augment::default(),
            seed: 0,
        };
        let gpu = "NVIDIA GeForce RTX 4080 SUPER (vulkan)".to_string();
        struct Spec {
            id: &'static str,
            config: TrainConfig,
            status: RunStatus,
            step: u32,
            spe: u32,
            created_h: f64,
            map_max: f32,
            ips: f32,
            model: Option<&'static str>,
            error: Option<&'static str>,
        }
        let specs = vec![
            Spec { id: LIVE_ID, config: cfg("shapes-hard-m", "ds_3b91", "folo-m", 100, 32, 640, "gpu:0", 0.01), status: RunStatus::Running, step: live_step, spe: 100, created_h: 0.25, map_max: 0.71, ips: 268.0, model: None, error: None },
            Spec { id: "run_b7e2", config: cfg("shapes-mini-s", "ds_c07a", "folo-s", 60, 16, 320, "gpu:0", 0.01), status: RunStatus::Queued, step: 0, spe: 19, created_h: 0.1, map_max: 0.6, ips: 900.0, model: None, error: None },
            Spec { id: "run_52e8", config: cfg("shapes-hard-s", "ds_3b91", "folo-s", 120, 32, 640, "gpu:0", 0.01), status: RunStatus::Completed, step: 120 * 100, spe: 100, created_h: 9.0, map_max: 0.612, ips: 412.0, model: Some("m_44c0"), error: None },
            Spec { id: "run_e4f0", config: cfg("shapes-hard-l", "ds_3b91", "folo-l", 100, 64, 1024, "gpu:0", 0.01), status: RunStatus::Failed, step: 112, spe: 50, created_h: 10.5, map_max: 0.7, ips: 96.0, model: None, error: Some("GPU out of memory: tried to allocate 2.1 GiB with 0.6 GiB free (batch 64, imgsz 1024)") },
            Spec { id: "run_1a2b", config: cfg("shapes-mini-n", "ds_c07a", "folo-n", 100, 16, 320, "cpu", 0.01), status: RunStatus::Stopped, step: 37 * 15, spe: 15, created_h: 4.0, map_max: 0.58, ips: 140.0, model: Some("m_90d2"), error: None },
            Spec { id: "run_7d1e", config: cfg("shapes-n", "ds_8f2c", "folo-n", 100, 16, 640, "gpu:0", 0.01), status: RunStatus::Completed, step: 100 * 50, spe: 50, created_h: 55.0, map_max: 0.781, ips: 610.0, model: Some("m_31aa"), error: None },
        ];
        let now = Instant::now();
        let runs = specs
            .into_iter()
            .map(|s| {
                let curve = Curve {
                    seed: hash(s.id),
                    epochs: s.config.epochs,
                    spe: s.spe,
                    lr0: s.config.lr0,
                    batch: s.config.batch,
                    map_max: s.map_max,
                    ips: s.ips,
                    classes: datasets.iter().find(|d| d.ds.id == s.config.dataset_id).map_or(3, |d| d.ds.classes.len() as u32),
                    close_mosaic: s.config.augment.close_mosaic,
                };
                let created = h(s.created_h);
                let total = s.config.epochs * s.spe;
                let run_secs = |steps: u32| (steps as f64 * (s.config.batch as f64 / s.ips as f64)) as i64;
                let finished = matches!(s.status, RunStatus::Completed | RunStatus::Stopped | RunStatus::Failed)
                    .then(|| time::format(created + 5 + run_secs(s.step) + s.step as i64 / s.spe as i64 * 3));
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
                        device: if s.config.device == "cpu" { "AMD Ryzen 9 7950X (32 threads)".into() } else { gpu.clone() },
                        model_id: s.model.map(Into::into),
                        error: s.error.map(Into::into),
                        config: s.config,
                    },
                    curve,
                    live: (s.status == RunStatus::Running && !frozen).then_some((s.step, now)),
                    step: s.step.min(total),
                    logs: Vec::new(),
                };
                fixture.logs = initial_logs(&fixture, created);
                fixture
            })
            .collect();
        Self {
            datasets,
            models,
            runs: Mutex::new(runs),
            frozen,
            started: now,
            image_cache: Mutex::new(HashMap::new()),
            fail: false,
            empty: false,
        }
    }

    pub fn failing(mut self) -> Self {
        self.fail = true;
        self
    }

    pub fn empty(mut self) -> Self {
        self.empty = true;
        self
    }

    fn check(&self) -> ApiResult<()> {
        if self.fail {
            Err(ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878")
                .with_hint("Start it with `folo serve`, or switch to mock data in Settings."))
        } else {
            Ok(())
        }
    }

    fn fixture(&self, id: &str) -> ApiResult<&DatasetFixture> {
        self.datasets
            .iter()
            .find(|d| d.ds.id == id)
            .ok_or_else(|| ApiError::new("dataset_not_found", format!("No dataset with id {id}")))
    }

    /// Current step of a run fixture, advancing live runs with the clock.
    fn current_step(f: &RunFixture) -> u32 {
        let total = f.run.epochs * f.run.steps_per_epoch;
        match f.live {
            Some((base, at)) => (base + (at.elapsed().as_secs_f64() / STEP_DT) as u32).min(total),
            None => f.step.min(total),
        }
    }

    fn snapshot(f: &RunFixture) -> Run {
        let mut run = f.run.clone();
        let step = Self::current_step(f);
        let total = run.epochs * run.steps_per_epoch;
        run.step = step;
        run.epoch = if step == 0 { 0 } else { ((step - 1) / run.steps_per_epoch + 1).min(run.epochs) };
        if f.live.is_some() && step >= total {
            run.status = RunStatus::Completed;
        }
        if run.status == RunStatus::Running {
            run.eta_s = Some((total - step) as f64 * STEP_DT + (run.epochs - run.epoch) as f64 * 0.4);
        }
        let done_epochs = step / run.steps_per_epoch;
        run.best = (1..=done_epochs)
            .map(|e| (e, f.curve.epoch(e).val.map50_95))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(epoch, map50_95)| Best { epoch, map50_95 });
        run
    }

    fn sim_now(&self) -> i64 {
        if self.frozen { MOCK_NOW } else { MOCK_NOW + self.started.elapsed().as_secs() as i64 }
    }

    fn val_images(&self, dataset_id: &str) -> Vec<(u32, &ImageMeta)> {
        let Ok(f) = self.fixture(dataset_id) else { return Vec::new() };
        f.images
            .iter()
            .enumerate()
            .filter(|(_, im)| im.split == Split::Val && !im.boxes.is_empty())
            .map(|(i, im)| (i as u32, im))
            .collect()
    }
}

fn initial_logs(f: &RunFixture, created: i64) -> Vec<LogLine> {
    let c = &f.run.config;
    let mut logs = vec![
        (0, LogLevel::Info, format!("folo train {} on {} ({} epochs, batch {}, imgsz {})", c.arch, c.dataset_id, c.epochs, c.batch, c.imgsz)),
        (1, LogLevel::Info, format!("device: {}", f.run.device)),
        (2, LogLevel::Debug, "dataloader: 8 workers, prefetch 4 batches, mosaic 1.0".to_string()),
        (3, LogLevel::Info, format!("model: {} · {} params · fused BN off (training)", c.arch, match c.arch.as_str() { "folo-n" => "2.63 M", "folo-s" => "9.42 M", "folo-m" => "20.1 M", _ => "43.7 M" })),
        (4, LogLevel::Info, "optimizer: AdamW(lr=0.01, wd=5e-4), cosine schedule, warmup 3 epochs".to_string()),
    ];
    if f.run.status == RunStatus::Failed {
        logs.push((40, LogLevel::Warn, "GPU memory at 97% after epoch 2 (15.5 / 16.0 GiB)".into()));
        logs.push((48, LogLevel::Error, f.run.error.clone().unwrap_or_default()));
    }
    if f.run.status == RunStatus::Stopped {
        logs.push((900, LogLevel::Warn, "stop requested; finishing current step".into()));
        logs.push((901, LogLevel::Info, "saved last checkpoint (epoch 37)".into()));
    }
    if f.run.id == LIVE_ID {
        logs.push((120, LogLevel::Warn, "label 'img_03417' has a box outside the image; clipped".into()));
    }
    logs.into_iter()
        .map(|(dt, level, message)| LogLine { t: time::format(created + 5 + dt), level, message })
        .collect()
}

impl Backend for MockBackend {
    fn describe(&self) -> String {
        "Mock data".into()
    }

    fn now(&self) -> i64 {
        self.sim_now()
    }

    fn health(&self) -> ApiResult<Health> {
        self.check()?;
        Ok(Health { status: "ok".into(), version: "0.1.0".into(), mock: true })
    }

    fn system(&self) -> ApiResult<SystemInfo> {
        self.check()?;
        let t = self.started.elapsed().as_secs_f32();
        let wobble = if self.frozen { 0.0 } else { (t * 0.7).sin() * 0.02 };
        let live = self.runs.lock().unwrap().iter().any(|f| Self::snapshot(f).status == RunStatus::Running);
        Ok(SystemInfo {
            cpu: Cpu { name: "AMD Ryzen 9 7950X".into(), threads: 32, util: if live { 0.31 + wobble } else { 0.04 } },
            memory: Memory { used_bytes: 23_100_000_000, total_bytes: 68_719_476_736 },
            gpus: vec![Gpu {
                ordinal: 0,
                name: "NVIDIA GeForce RTX 4080 SUPER".into(),
                backend: "vulkan".into(),
                memory_bytes: 17_171_480_576,
                util: Some(if live { 0.94 + wobble } else { 0.02 }),
            }],
        })
    }

    fn datasets(&self) -> ApiResult<Vec<Dataset>> {
        self.check()?;
        if self.empty {
            return Ok(Vec::new());
        }
        Ok(self.datasets.iter().map(|d| Dataset { stats: None, ..d.ds.clone() }).collect())
    }

    fn dataset(&self, id: &str) -> ApiResult<Dataset> {
        self.check()?;
        Ok(self.fixture(id)?.ds.clone())
    }

    fn images(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        self.check()?;
        let f = self.fixture(id)?;
        let matching: Vec<&ImageMeta> = f
            .images
            .iter()
            .filter(|im| q.split.is_none_or(|s| im.split == s))
            .filter(|im| q.class.is_none_or(|c| im.boxes.iter().any(|b| b.class == c)))
            .collect();
        let limit = if q.limit == 0 { 100 } else { q.limit };
        Ok(ImagePage {
            total: matching.len() as u32,
            images: matching.into_iter().skip(q.offset as usize).take(limit as usize).cloned().collect(),
        })
    }

    fn image(&self, dataset_id: &str, image_id: &str, thumb: Option<u32>) -> ApiResult<ImageData> {
        let f = self.fixture(dataset_id)?;
        let index: u32 = image_id
            .strip_prefix("img_")
            .and_then(|n| n.parse().ok())
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
        Ok(self.models.clone())
    }

    fn model_eval(&self, id: &str) -> ApiResult<ModelEval> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {id}")))?;
        let metrics = m.metrics.clone().unwrap_or(ModelMetrics { map50: 0.5, map50_95: 0.3, precision: 0.5, recall: 0.5 });
        let n = m.classes.len();
        let mut rng = Rng::new(hash(id));
        let per_class: Vec<ClassMetrics> = (0..n as u32)
            .map(|c| {
                let d = 1.0 + 0.07 * rng.normal() - 0.025 * c as f32;
                ClassMetrics {
                    class: c,
                    map50: (metrics.map50 * d).min(0.995),
                    map50_95: (metrics.map50_95 * d).min(0.99),
                    precision: (metrics.precision * (1.0 + 0.04 * rng.normal())).min(0.995),
                    recall: (metrics.recall * (1.0 + 0.05 * rng.normal())).min(0.995),
                }
            })
            .collect();
        let pr = per_class
            .iter()
            .map(|cm| {
                // A smooth monotone PR curve whose area matches the class's AP50.
                let k = (cm.map50 / (1.0 - cm.map50).max(0.02)).powf(0.9) * 1.6;
                let recall: Vec<f32> = (0..=50).map(|i| i as f32 / 50.0).collect();
                let precision: Vec<f32> = recall.iter().map(|&r| (1.0 - r.powf(k.max(0.3))).max(0.0).powf(0.35)).collect();
                PrCurve { class: cm.class, recall, precision, ap50: cm.map50 }
            })
            .collect();
        let images = 800;
        let mut confusion = vec![vec![0u32; n + 1]; n + 1];
        for t in 0..n {
            let count = 900 + rng.below(500);
            let recall = per_class[t].recall;
            let hit = (count as f32 * recall) as u32;
            confusion[t][t] = hit;
            let mut left = count - hit;
            for (p, row_cell) in confusion[t].iter_mut().enumerate().take(n) {
                if p != t {
                    let share = (left as f32 * rng.range(0.05, 0.25)) as u32;
                    *row_cell += share;
                    left -= share;
                }
            }
            confusion[t][n] = left;
            confusion[n][t] = (count as f32 * (1.0 - per_class[t].precision) * 0.6) as u32;
        }
        Ok(ModelEval { pr, confusion, per_class, images })
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
        Ok(self.runs.lock().unwrap().iter().map(Self::snapshot).collect())
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
        let mut runs = self.runs.lock().unwrap();
        let id = format!("run_{:04x}", hash(&format!("{}{}", config.name, runs.len())) & 0xFFFF);
        let spe = (ds.ds.splits.train / config.batch.max(1)).max(1);
        let curve = Curve {
            seed: hash(&id),
            epochs: config.epochs,
            spe,
            lr0: config.lr0,
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
                epoch: 0,
                epochs: config.epochs,
                step: 0,
                steps_per_epoch: spe,
                eta_s: None,
                best: None,
                device: if config.device == "cpu" { "AMD Ryzen 9 7950X (32 threads)".into() } else { "NVIDIA GeForce RTX 4080 SUPER (vulkan)".into() },
                model_id: None,
                error: None,
                config,
            },
            curve,
            live: (!self.frozen).then(|| (0, Instant::now())),
            step: 0,
            logs: Vec::new(),
        };
        f.logs = initial_logs(&f, now - 5);
        let run = Self::snapshot(&f);
        runs.insert(0, f);
        Ok(run)
    }

    fn stop_run(&self, id: &str) -> ApiResult<Run> {
        self.check()?;
        let mut runs = self.runs.lock().unwrap();
        let f = runs
            .iter_mut()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        if !f.run.status.is_active() {
            return Err(ApiError::new("run_not_active", "This run is not running"));
        }
        f.step = Self::current_step(f);
        f.live = None;
        f.run.status = RunStatus::Stopped;
        f.run.finished = Some(time::format(self.sim_now()));
        let epoch = f.step / f.run.steps_per_epoch;
        f.logs.push(LogLine { t: time::format(self.sim_now()), level: LogLevel::Warn, message: "stop requested; finishing current step".into() });
        f.logs.push(LogLine { t: time::format(self.sim_now()), level: LogLevel::Info, message: format!("saved last checkpoint (epoch {epoch})") });
        Ok(Self::snapshot(f))
    }

    fn metrics(&self, id: &str) -> ApiResult<Metrics> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        let step = Self::current_step(f);
        Ok(Metrics {
            steps: (1..=step).map(|i| f.curve.step(i)).collect(),
            epochs: (1..=step / f.run.steps_per_epoch).map(|e| f.curve.epoch(e)).collect(),
        })
    }

    fn samples(&self, id: &str, epoch: Option<u32>) -> ApiResult<Samples> {
        self.check()?;
        let (curve, dataset_id, done) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (f.curve.clone(), f.run.config.dataset_id.clone(), Self::current_step(f) / f.run.steps_per_epoch)
        };
        let epoch = epoch.unwrap_or(done).min(done);
        let q = (curve.quality(epoch) * curve.map_max / 0.75).clamp(0.05, 1.0);
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
                pred: fake_predictions(&im.boxes, im.width, im.height, q, curve.classes, hash(id) ^ i as u64 ^ (epoch as u64) << 20),
            })
            .collect();
        Ok(Samples { epoch, samples })
    }

    fn logs(&self, id: &str) -> ApiResult<Vec<LogLine>> {
        self.check()?;
        let runs = self.runs.lock().unwrap();
        let f = runs
            .iter()
            .find(|f| f.run.id == id)
            .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
        // Epoch summaries up to now, merged with the fixture's lines in time order.
        let created = time::parse(&f.run.created).unwrap_or(MOCK_NOW);
        let done = Self::current_step(f) / f.run.steps_per_epoch;
        let per_epoch = (f.run.steps_per_epoch as f64 * STEP_DT) as i64 + 1;
        let mut lines = f.logs.clone();
        let from = done.saturating_sub(60) + 1;
        for e in from..=done {
            let ep = f.curve.epoch(e);
            lines.push(LogLine {
                t: time::format(created + 10 + e as i64 * per_epoch),
                level: LogLevel::Info,
                message: format!(
                    "epoch {e}/{} · mAP50-95 {:.3} · mAP50 {:.3} · {:.1} s",
                    f.run.epochs, ep.val.map50_95, ep.val.map50, ep.duration_s
                ),
            });
        }
        lines.sort_by(|a, b| a.t.cmp(&b.t));
        Ok(lines)
    }

    fn events(&self, id: &str) -> ApiResult<Receiver<RunEvent>> {
        self.check()?;
        let (tx, rx) = channel();
        let (live, curve, spe, epochs, status) = {
            let runs = self.runs.lock().unwrap();
            let f = runs
                .iter()
                .find(|f| f.run.id == id)
                .ok_or_else(|| ApiError::new("run_not_found", format!("No run {id}")))?;
            (f.live, f.curve.clone(), f.run.steps_per_epoch, f.run.epochs, Self::snapshot(f).status)
        };
        let _ = tx.send(RunEvent::Status(StatusEvent { status, error: None }));
        let Some((base, at)) = live else { return Ok(rx) };
        let total = epochs * spe;
        let start_secs = self.sim_now();
        std::thread::spawn(move || {
            let mut sent = (base + (at.elapsed().as_secs_f64() / STEP_DT) as u32).min(total);
            loop {
                std::thread::sleep(Duration::from_millis(100));
                let now = (base + (at.elapsed().as_secs_f64() / STEP_DT) as u32).min(total);
                while sent < now {
                    sent += 1;
                    if tx.send(RunEvent::Step(curve.step(sent))).is_err() {
                        return;
                    }
                    if sent % spe == 0 {
                        let e = sent / spe;
                        let ep = curve.epoch(e);
                        let t = time::format(start_secs + at.elapsed().as_secs() as i64);
                        let msg = format!(
                            "epoch {e}/{epochs} · mAP50-95 {:.3} · mAP50 {:.3} · {:.1} s",
                            ep.val.map50_95, ep.val.map50, ep.duration_s
                        );
                        if tx.send(RunEvent::Epoch(ep)).is_err()
                            || tx.send(RunEvent::Log(LogLine { t, level: LogLevel::Info, message: msg })).is_err()
                        {
                            return;
                        }
                    }
                }
                if sent >= total {
                    let _ = tx.send(RunEvent::Status(StatusEvent { status: RunStatus::Completed, error: None }));
                    return;
                }
            }
        });
        Ok(rx)
    }

    fn predict(&self, req: &PredictRequest) -> ApiResult<Prediction> {
        self.check()?;
        let m = self
            .models
            .iter()
            .find(|m| m.id == req.model_id)
            .ok_or_else(|| ApiError::new("model_not_found", format!("No model {}", req.model_id)))?;
        let q = m.metrics.as_ref().map_or(0.5, |m| m.map50_95 / 0.8).clamp(0.1, 1.0);
        let (w, h, raw, seed) = match &req.input {
            PredictInput::Dataset { dataset_id, image_id } => {
                let f = self.fixture(dataset_id)?;
                let im = f
                    .images
                    .iter()
                    .find(|im| &im.id == image_id)
                    .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
                let p = fake_predictions(&im.boxes, im.width, im.height, q, m.classes.len() as u32, hash(image_id) ^ hash(&m.id));
                (im.width, im.height, p, hash(image_id))
            }
            PredictInput::File { name, bytes } => {
                let (w, h) = image::ImageReader::new(std::io::Cursor::new(bytes.as_slice()))
                    .with_guessed_format()
                    .ok()
                    .and_then(|r| r.into_dimensions().ok())
                    .ok_or_else(|| {
                        ApiError::new("unsupported_image", format!("Could not decode {name}"))
                            .with_hint("FOLO reads JPEG and PNG images.")
                    })?;
                let seed = bytes.iter().step_by(97).fold(hash(name), |h, &b| h.rotate_left(5) ^ b as u64);
                let mut rng = Rng::new(seed);
                let n = 2 + rng.below(4);
                let s = w.min(h) as f32;
                let p = (0..n)
                    .map(|_| {
                        let bw = rng.range(0.12, 0.32) * s;
                        let bh = bw * rng.range(0.7, 1.4);
                        let x1 = rng.range(0.0, w as f32 - bw);
                        let y1 = rng.range(0.0, (h as f32 - bh).max(1.0));
                        ScoredBox { class: rng.below(m.classes.len() as u32), score: rng.range(0.3, 0.92), x1, y1, x2: x1 + bw, y2: (y1 + bh).min(h as f32) }
                    })
                    .collect();
                (w, h, p, seed)
            }
        };
        let mut boxes: Vec<DetBox> = raw
            .into_iter()
            .filter(|b| b.score >= req.conf)
            .map(|b| DetBox {
                class: b.class,
                name: m.classes.get(b.class as usize).cloned().unwrap_or_default(),
                score: b.score,
                x1: b.x1,
                y1: b.y1,
                x2: b.x2,
                y2: b.y2,
            })
            .collect();
        boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
        boxes.truncate(req.max_det as usize);
        let scale = match m.arch.as_str() {
            "folo-n" => 1.0,
            "folo-s" => 1.7,
            "folo-m" => 3.1,
            _ => 5.4,
        } * (m.imgsz as f32 / 640.0).powi(2);
        let mut rng = Rng::new(seed ^ 0x71);
        let pre = 0.62 + 0.1 * rng.f32();
        let inf = 2.9 * scale * (1.0 + 0.05 * rng.normal());
        let post = 0.14 + 0.02 * boxes.len() as f32;
        Ok(Prediction {
            model_id: m.id.clone(),
            image: ImageSize { width: w, height: h },
            boxes,
            timing: Timing { preprocess_ms: pre, inference_ms: inf, postprocess_ms: post, total_ms: pre + inf + post },
            device: "gpu:0".into(),
        })
    }
}

/// Plausible detections for known ground truth at model quality `q` (0..1): jittered hits,
/// occasional class confusions and misses, and a few low-score false positives.
fn fake_predictions(truth: &[BBoxT], w: u32, h: u32, q: f32, classes: u32, seed: u64) -> Vec<ScoredBox> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for b in truth {
        if rng.f32() > 0.55 + 0.44 * q {
            continue;
        }
        let (bw, bh) = (b.x2 - b.x1, b.y2 - b.y1);
        let j = (1.0 - q) * 0.12 + 0.015;
        let x1 = (b.x1 + rng.normal() * j * bw).clamp(0.0, w as f32);
        let y1 = (b.y1 + rng.normal() * j * bh).clamp(0.0, h as f32);
        let x2 = (b.x2 + rng.normal() * j * bw).clamp(x1 + 1.0, w as f32);
        let y2 = (b.y2 + rng.normal() * j * bh).clamp(y1 + 1.0, h as f32);
        let class = if rng.f32() < (1.0 - q) * 0.18 { rng.below(classes) } else { b.class };
        out.push(ScoredBox { class, score: (0.45 + 0.5 * q * rng.range(0.75, 1.0) + 0.04 * rng.normal()).clamp(0.26, 0.99), x1, y1, x2, y2 });
    }
    let fps = if rng.f32() < (1.0 - q) * 0.9 { 1 + rng.below(2) } else { 0 };
    for _ in 0..fps {
        let s = w.min(h) as f32 * rng.range(0.05, 0.15);
        let x1 = rng.range(0.0, w as f32 - s);
        let y1 = rng.range(0.0, h as f32 - s);
        out.push(ScoredBox { class: rng.below(classes), score: rng.range(0.26, 0.48), x1, y1, x2: x1 + s, y2: y1 + s });
    }
    out
}

type BBoxT = super::types::LabelBox;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_mock_is_deterministic() {
        let a = MockBackend::new(4200, true);
        let b = MockBackend::new(4200, true);
        assert_eq!(a.runs().unwrap(), b.runs().unwrap());
        assert_eq!(a.metrics(LIVE_ID).unwrap(), b.metrics(LIVE_ID).unwrap());
        let live = a.run(LIVE_ID).unwrap();
        assert_eq!(live.status, RunStatus::Running);
        assert_eq!(live.epoch, 42);
        assert_eq!(a.metrics(LIVE_ID).unwrap().epochs.len(), 42);
    }

    #[test]
    fn learning_curves_improve() {
        let m = MockBackend::new(0, true);
        let metrics = m.metrics("run_7d1e").unwrap();
        let first = &metrics.epochs[0];
        let last = metrics.epochs.last().unwrap();
        assert!(last.val.map50_95 > first.val.map50_95 + 0.3);
        assert!(last.train_loss.total < first.train_loss.total);
        assert!(metrics.epochs.iter().all(|e| e.val.map50 >= e.val.map50_95));
    }

    #[test]
    fn image_pages_and_filters() {
        let m = MockBackend::new(0, true);
        let all = m.images("ds_8f2c", &ImageQuery { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(all.total, 1000);
        assert_eq!(all.images.len(), 10);
        let val = m.images("ds_8f2c", &ImageQuery { split: Some(Split::Val), limit: 500, ..Default::default() }).unwrap();
        assert_eq!(val.total, 200);
        let squares = m.images("ds_8f2c", &ImageQuery { class: Some(1), limit: 5, ..Default::default() }).unwrap();
        assert!(squares.images.iter().all(|im| im.boxes.iter().any(|b| b.class == 1)));
        match m.image("ds_8f2c", "img_00003", Some(128)).unwrap() {
            ImageData::Rgba { width, height, pixels } => {
                assert_eq!((width, height), (128, 128));
                assert_eq!(pixels.len(), 128 * 128 * 4);
            }
            ImageData::Encoded(_) => panic!("mock returns raw pixels"),
        }
        assert!(m.image("ds_8f2c", "img_99999", None).is_err());
    }

    #[test]
    fn stats_match_images() {
        let m = MockBackend::new(0, true);
        let ds = m.dataset("ds_3b91").unwrap();
        let stats = ds.stats.unwrap();
        assert_eq!(stats.class_counts.iter().sum::<u32>(), stats.boxes);
        assert_eq!(stats.boxes_per_image_hist.counts.iter().sum::<u32>(), ds.splits.total());
    }

    #[test]
    fn stop_and_errors() {
        let m = MockBackend::new(4200, true);
        let stopped = m.stop_run(LIVE_ID).unwrap();
        assert_eq!(stopped.status, RunStatus::Stopped);
        assert!(m.stop_run(LIVE_ID).is_err());
        assert!(MockBackend::new(0, true).failing().runs().is_err());
        assert!(MockBackend::new(0, true).empty().datasets().unwrap().is_empty());
    }

    #[test]
    fn predictions_respect_threshold() {
        let m = MockBackend::new(0, true);
        let req = PredictRequest {
            model_id: "m_31aa".into(),
            input: PredictInput::Dataset { dataset_id: "ds_8f2c".into(), image_id: "img_00810".into() },
            conf: 0.5,
            iou: 0.7,
            max_det: 300,
        };
        let p = m.predict(&req).unwrap();
        assert!(p.boxes.iter().all(|b| b.score >= 0.5));
        assert!(p.timing.total_ms > p.timing.inference_ms);
    }
}
