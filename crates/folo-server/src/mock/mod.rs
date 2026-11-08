//! `MockStore`: deterministic, realistic data for UI work before (or without) the engine.
//!
//! Three procedural datasets, six models, six finished runs (one stopped, one failed) with full metric histories,
//! and one live run that advances in real time (100 epochs in ~10 minutes, then it loops). Runs created through the
//! API are simulated the same way. Datasets registered from disk are scanned for real.

mod runs;
mod scene;

use std::io::Cursor;
use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use folo_api::{
    Arch, BBox, Dataset, DatasetFormat, ExportFormat, ExportResult, GpuBackend, GpuInfo, ImageInfo,
    ImagePage, Model, ModelMetrics, PredictResponse, RegisterDataset, Run, RunSamples, Sample, ScoredBox,
    Split, Splits, SyntheticDataset, TrainConfig,
};

use crate::disk::DiskDataset;
use crate::error::{Error, Result};
use crate::stats::StatsBuilder;
use crate::store::{ImageData, ImageQuery, PredictInput, PredictSource, Store, Subscription};
use crate::util::{Rng, fnv1a, mix64, noise, parse_rfc3339, rfc3339, unique_id, unix_now};
use runs::{Curve, LiveClock, SimRun, Timeline, arch_spec};
use scene::{Scene, Theme};

pub use runs::TICK;

const DEVICE: &str = "NVIDIA GeForce RTX 4080 SUPER (vulkan)";

/// A procedural dataset.
struct ProcDataset {
    meta: Dataset,
    theme: Theme,
    w: u32,
    h: u32,
    seed: u64,
    labels: Vec<Vec<BBox>>,
    stats: folo_api::DatasetStats,
}

impl ProcDataset {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: &str,
        name: &str,
        theme: Theme,
        classes: &[&str],
        w: u32,
        h: u32,
        splits: Splits,
        seed: u64,
        created: &str,
        broken: u64,
    ) -> Self {
        let classes: Vec<String> = classes.iter().map(|s| s.to_string()).collect();
        ProcDataset::build(id.into(), name.into(), theme, classes, w, h, splits, seed, created.into(), broken)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        id: String,
        name: String,
        theme: Theme,
        classes: Vec<String>,
        w: u32,
        h: u32,
        splits: Splits,
        seed: u64,
        created: String,
        broken: u64,
    ) -> Self {
        let n = splits.train + splits.val + splits.test;
        let labels: Vec<Vec<BBox>> =
            (0..n).map(|i| Scene::generate(theme, &classes, w, h, seed, i).labels()).collect();
        let mut sb = StatsBuilder::new(classes.len());
        let (wf, hf) = (w as f32, h as f32);
        for l in &labels {
            sb.add_image(l.iter().map(|b| (b.class, (b.x2 - b.x1) / wf, (b.y2 - b.y1) / hf)));
        }
        sb.add_broken(broken);
        let meta = Dataset {
            id,
            path: format!("mock://{name}"),
            name,
            format: DatasetFormat::Yolo,
            classes,
            created,
            splits,
            stats: None,
        };
        ProcDataset { meta, theme, w, h, seed, labels, stats: sb.finish() }
    }

    fn split_of(&self, i: u64) -> Split {
        let s = self.meta.splits;
        if i < s.train {
            Split::Train
        } else if i < s.train + s.val {
            Split::Val
        } else {
            Split::Test
        }
    }

    fn index(&self, image_id: &str) -> Result<u64> {
        image_id
            .parse::<u64>()
            .ok()
            .filter(|&i| (i as usize) < self.labels.len() && format!("{i:06}") == image_id)
            .ok_or_else(|| Error::not_found("image", image_id))
    }

    fn info(&self, i: u64) -> ImageInfo {
        let split = self.split_of(i);
        ImageInfo {
            id: format!("{i:06}"),
            file: format!("images/{}/{i:06}.jpg", split.as_str()),
            split,
            width: self.w,
            height: self.h,
            boxes: self.labels[i as usize].clone(),
        }
    }
}

enum DatasetEntry {
    Proc(ProcDataset),
    Disk(DiskDataset),
}

impl DatasetEntry {
    fn meta(&self) -> &Dataset {
        match self {
            DatasetEntry::Proc(d) => &d.meta,
            DatasetEntry::Disk(d) => &d.meta,
        }
    }

    fn with_stats(&self) -> Dataset {
        let (meta, stats) = match self {
            DatasetEntry::Proc(d) => (&d.meta, &d.stats),
            DatasetEntry::Disk(d) => (&d.meta, &d.stats),
        };
        Dataset { stats: Some(stats.clone()), ..meta.clone() }
    }

    fn list_images(&self, q: &ImageQuery) -> ImagePage {
        match self {
            DatasetEntry::Disk(d) => d.list_images(q),
            DatasetEntry::Proc(d) => {
                let matches = |i: u64| {
                    q.split.is_none_or(|s| s == d.split_of(i))
                        && q.class.is_none_or(|c| d.labels[i as usize].iter().any(|b| b.class == c))
                };
                let n = d.labels.len() as u64;
                let total = (0..n).filter(|&i| matches(i)).count() as u64;
                let images = (0..n).filter(|&i| matches(i)).skip(q.offset as usize).take(q.limit as usize);
                ImagePage { total, images: images.map(|i| d.info(i)).collect() }
            }
        }
    }

    fn image(&self, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        match self {
            DatasetEntry::Disk(d) => d.image(image_id, thumb),
            DatasetEntry::Proc(d) => {
                let i = d.index(image_id)?;
                let scene = Scene::generate(d.theme, &d.meta.classes, d.w, d.h, d.seed, i);
                let quality = if thumb.is_some() { 80 } else { 90 };
                let bytes = scene.render(thumb).to_jpeg(quality);
                Ok(ImageData { content_type: "image/jpeg", bytes, immutable: true })
            }
        }
    }

    /// `(width, height, truth)` of an image.
    fn truth(&self, image_id: &str) -> Result<(u32, u32, Vec<BBox>)> {
        match self {
            DatasetEntry::Disk(d) => d.truth(image_id),
            DatasetEntry::Proc(d) => {
                let i = d.index(image_id)?;
                Ok((d.w, d.h, d.labels[i as usize].clone()))
            }
        }
    }

    /// Ids of up to `n` validation images spread over the split (falls back to train when there is no val split).
    fn sample_ids(&self, n: usize) -> Vec<String> {
        let split = if self.meta().splits.val > 0 { Split::Val } else { Split::Train };
        let all =
            self.list_images(&ImageQuery { split: Some(split), offset: 0, limit: u64::MAX, class: None });
        let step = (all.images.len() / n.max(1)).max(1);
        all.images.into_iter().step_by(step).take(n).map(|im| im.id).collect()
    }
}

/// Fake detections: the truth, jittered and occasionally missed or mislabelled, plus a few false positives. Better
/// `quality` (0..1) gives tighter boxes, higher scores and fewer mistakes.
fn fake_detections(
    truth: &[BBox],
    w: u32,
    h: u32,
    classes: &[String],
    quality: f32,
    seed: u64,
) -> Vec<ScoredBox> {
    if classes.is_empty() {
        return Vec::new();
    }
    let mut r = Rng::new(seed);
    let (wf, hf) = (w as f32, h as f32);
    let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| format!("class{c}"));
    let mut out = Vec::new();
    for t in truth {
        if !r.chance(0.35 + 0.63 * quality) {
            continue;
        }
        let (bw, bh) = (t.x2 - t.x1, t.y2 - t.y1);
        let j = (1.0 - quality) * 0.22 + 0.015;
        let mut class = t.class;
        if classes.len() > 1 && r.chance((1.0 - quality) * 0.25) {
            class = r.below(classes.len() as u64) as u32;
        }
        let score = (0.35 + 0.6 * quality + 0.25 * (r.f32() - 0.5)).clamp(0.05, 0.99);
        out.push(ScoredBox {
            class,
            name: name(class),
            score,
            x1: (t.x1 + bw * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, wf),
            y1: (t.y1 + bh * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, hf),
            x2: (t.x2 + bw * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, wf),
            y2: (t.y2 + bh * j * (r.f32() * 2.0 - 1.0)).clamp(0.0, hf),
        });
    }
    let fp = (r.f32() * 3.5 * (1.0 - quality)).floor() as u32;
    for _ in 0..fp {
        let (bw, bh) = (r.range(0.04, 0.25) * wf, r.range(0.04, 0.25) * hf);
        let (x, y) = (r.range(0.0, wf - bw), r.range(0.0, hf - bh));
        let class = r.below(classes.len().max(1) as u64) as u32;
        out.push(ScoredBox {
            class,
            name: name(class),
            score: r.range(0.08, 0.45),
            x1: x,
            y1: y,
            x2: x + bw,
            y2: y + bh,
        });
    }
    for b in &mut out {
        if b.x2 < b.x1 {
            std::mem::swap(&mut b.x1, &mut b.x2);
        }
        if b.y2 < b.y1 {
            std::mem::swap(&mut b.y1, &mut b.y2);
        }
        b.score = (b.score * 1000.0).round() / 1000.0;
        for v in [&mut b.x1, &mut b.y1, &mut b.x2, &mut b.y2] {
            *v = (*v * 10.0).round() / 10.0;
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

struct StaticModel {
    model: Model,
}

/// The mock store. See the module docs.
pub struct MockStore {
    datasets: RwLock<Vec<Arc<DatasetEntry>>>,
    models: RwLock<Vec<StaticModel>>,
    runs: RwLock<Vec<(Arc<SimRun>, Arc<DatasetEntry>)>>,
}

impl Default for MockStore {
    fn default() -> Self {
        MockStore::new()
    }
}

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
}

enum End {
    Completed,
    Stopped(u64),
    Failed(u64, &'static str),
}

fn steps_per_epoch(train: u64, batch: u32) -> u64 {
    train.div_ceil(batch.max(1) as u64).max(1)
}

fn model_bytes(params: u64, f16: bool) -> u64 {
    params * if f16 { 2 } else { 4 } + 18_432
}

impl MockStore {
    /// Builds the mock catalogue. The live run starts at epoch ~12 so charts have data right away.
    pub fn new() -> MockStore {
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
            },
        ];
        let mut runs = Vec::new();
        let mut models = Vec::new();
        for d in &defs {
            let entry = &datasets[d.ds];
            let meta = entry.meta();
            let config = TrainConfig {
                name: d.name.into(),
                dataset_id: meta.id.clone(),
                arch: d.arch,
                epochs: d.epochs,
                batch: d.batch,
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
            let run = SimRun::new(
                d.id.into(),
                config,
                meta.classes.clone(),
                spe,
                curve,
                DEVICE.into(),
                d.model.map(|m| m.0.to_string()),
                false,
                Timeline::Static { created, status, last_step, error },
            );
            if let Some((mid, mname)) = d.model {
                let snap = run.snap();
                let best = run.curve.best(run.epochs_done(&snap)).map(|b| b.epoch).unwrap_or(1);
                let val = run.curve.val(best);
                let metrics = ModelMetrics {
                    map50: val.map50,
                    map50_95: val.map50_95,
                    precision: val.precision,
                    recall: val.recall,
                };
                let (params, flops_g, ..) = arch_spec(d.arch);
                let finished = snap.finished.unwrap_or(created);
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
                };
                if mid == "m_53d1" {
                    models.push(StaticModel {
                        model: Model {
                            id: "m_53d2".into(),
                            name: "r6-n-best-f16".into(),
                            bytes: model_bytes(params, true),
                            created: rfc3339(finished + 3600.0),
                            ..base.clone()
                        },
                    });
                }
                models.push(StaticModel { model: base });
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
        );
        runs.push((Arc::new(live), Arc::clone(r6)));

        MockStore { datasets: RwLock::new(datasets), models: RwLock::new(models), runs: RwLock::new(runs) }
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
                }),
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

impl Store for MockStore {
    fn is_mock(&self) -> bool {
        true
    }

    fn gpus(&self) -> Vec<GpuInfo> {
        let busy =
            self.runs.read().unwrap().iter().any(|r| r.0.snap().status == folo_api::RunStatus::Running);
        let t = unix_now() as u64;
        let util = if busy { 0.9 + 0.06 * noise(&[t, 1]) } else { 0.03 + 0.02 * noise(&[t, 2]).abs() };
        vec![GpuInfo {
            ordinal: 0,
            name: "NVIDIA GeForce RTX 4080 SUPER".into(),
            backend: GpuBackend::Vulkan,
            memory_bytes: 17_171_480_576,
            util: Some((util * 100.0).round() / 100.0),
        }]
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>> {
        Ok(self.datasets.read().unwrap().iter().map(|d| d.meta().clone()).collect())
    }

    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset> {
        let mut ds = DiskDataset::scan(String::new(), req.name, &req.path, req.format, rfc3339(unix_now()))?;
        let mut list = self.datasets.write().unwrap();
        ds.meta.id = unique_id("ds", fnv1a(req.path.as_bytes()) ^ unix_now().to_bits(), |id| {
            list.iter().any(|d| d.meta().id == id)
        });
        let out = Dataset { stats: Some(ds.stats.clone()), ..ds.meta.clone() };
        list.push(Arc::new(DatasetEntry::Disk(ds)));
        Ok(out)
    }

    fn create_synthetic(&self, req: SyntheticDataset) -> Result<Dataset> {
        let bad = |m: &str| Err(Error::new(StatusCode::BAD_REQUEST, "invalid_body", m.to_string()));
        if req.name.trim().is_empty() {
            return bad("name must not be empty");
        }
        if !(1..=100_000).contains(&req.images) {
            return bad("images must be in 1..=100000");
        }
        if req.classes.is_empty() || req.classes.len() > 80 || req.classes.iter().any(|c| c.trim().is_empty())
        {
            return bad("classes must hold 1..=80 non-empty names");
        }
        if !(32..=4096).contains(&req.imgsz) {
            return bad("imgsz must be in 32..=4096");
        }
        let seed = fnv1a(req.name.as_bytes()) ^ req.seed ^ unix_now().to_bits();
        let train = (req.images as f64 * 0.8).round() as u64;
        let splits = Splits { train, val: req.images - train, test: 0 };
        let mut ds = ProcDataset::build(
            String::new(),
            req.name,
            Theme::Shapes,
            req.classes,
            req.imgsz,
            req.imgsz,
            splits,
            mix64(req.seed ^ 0x5717),
            rfc3339(unix_now()),
            0,
        );
        let mut list = self.datasets.write().unwrap();
        ds.meta.id = unique_id("ds", seed, |id| list.iter().any(|d| d.meta().id == id));
        let entry = Arc::new(DatasetEntry::Proc(ds));
        let out = entry.with_stats();
        list.push(entry);
        Ok(out)
    }

    fn get_dataset(&self, id: &str) -> Result<Dataset> {
        Ok(self.dataset(id)?.with_stats())
    }

    fn list_images(&self, id: &str, q: &ImageQuery) -> Result<ImagePage> {
        Ok(self.dataset(id)?.list_images(q))
    }

    fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        self.dataset(id)?.image(image_id, thumb)
    }

    fn delete_dataset(&self, id: &str) -> Result<()> {
        let mut ds = self.datasets.write().unwrap();
        let before = ds.len();
        ds.retain(|d| d.meta().id != id);
        if ds.len() == before { Err(Error::not_found("dataset", id)) } else { Ok(()) }
    }

    fn list_models(&self) -> Result<Vec<Model>> {
        Ok(self.all_models())
    }

    fn get_model(&self, id: &str) -> Result<Model> {
        self.all_models().into_iter().find(|m| m.id == id).ok_or_else(|| Error::not_found("model", id))
    }

    fn export_model(&self, id: &str, format: ExportFormat) -> Result<ExportResult> {
        let m = self.get_model(id)?;
        let (ext, bytes) = match format {
            ExportFormat::FoloF16 => ("f16.folo", model_bytes(m.params, true)),
            ExportFormat::Onnx => ("onnx", m.params * 4 + 61_440),
        };
        Ok(ExportResult { path: format!("mock://exports/{}.{ext}", m.name), bytes })
    }

    fn list_runs(&self) -> Result<Vec<Run>> {
        let mut runs: Vec<Run> = self.runs.read().unwrap().iter().map(|r| r.0.to_api()).collect();
        runs.sort_by(|a, b| b.created.cmp(&a.created).then(b.id.cmp(&a.id)));
        Ok(runs)
    }

    fn create_run(&self, mut config: TrainConfig) -> Result<Run> {
        let ds = self.dataset(&config.dataset_id)?;
        self.validate(&config)?;
        let meta = ds.meta();
        if meta.classes.is_empty() || meta.splits.train == 0 {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("dataset_id: '{}' has no classes or no training images", meta.id),
            ));
        }
        if config.name.trim().is_empty() {
            config.name = format!("{}-{}", meta.name, &config.arch.as_str()[5..]);
        }
        // Ids are allocated under the runs write lock so concurrent creates cannot pick the same one, and model
        // ids are checked against every model that exists or will exist when a run completes.
        let mut runs = self.runs.write().unwrap();
        let id = unique_id("run", fnv1a(config.name.as_bytes()) ^ unix_now().to_bits(), |id| {
            runs.iter().any(|r| r.0.id == id)
        });
        let models_taken: std::collections::HashSet<String> = self
            .models
            .read()
            .unwrap()
            .iter()
            .map(|m| m.model.id.clone())
            .chain(runs.iter().filter_map(|r| r.0.model_id.clone()))
            .collect();
        let model_id = unique_id("m", fnv1a(id.as_bytes()), |m| models_taken.contains(m));
        let spe = steps_per_epoch(meta.splits.train, config.batch);
        let quality = 0.45 + 0.4 * (fnv1a(meta.id.as_bytes()) % 100) as f32 / 100.0;
        let curve = Curve::new(fnv1a(id.as_bytes()), quality, meta.classes.len(), &config, spe);
        let device = match config.device.as_str() {
            "cpu" => "CPU".to_string(),
            _ => DEVICE.to_string(),
        };
        let clock = LiveClock::new(6.0, 2.0, 0.0, false, 0.0);
        let run = Arc::new(SimRun::new(
            id,
            config,
            meta.classes.clone(),
            spe,
            curve,
            device,
            Some(model_id),
            true,
            Timeline::Live(std::sync::Mutex::new(clock)),
        ));
        let api = run.to_api();
        runs.push((run, Arc::clone(&ds)));
        Ok(api)
    }

    fn get_run(&self, id: &str) -> Result<Run> {
        Ok(self.run(id)?.0.to_api())
    }

    fn stop_run(&self, id: &str) -> Result<Run> {
        let (run, _) = self.run(id)?;
        if !run.stop() {
            return Err(Error::conflict("run_not_active", format!("run '{id}' is not running")));
        }
        Ok(run.to_api())
    }

    fn delete_run(&self, id: &str) -> Result<()> {
        let (run, _) = self.run(id)?;
        if run.is_active() {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                .hint(format!("stop it first: POST /api/runs/{id}/stop")));
        }
        run.delete();
        self.runs.write().unwrap().retain(|r| r.0.id != id);
        Ok(())
    }

    fn run_metrics(&self, id: &str, every: u64) -> Result<folo_api::RunMetrics> {
        Ok(self.run(id)?.0.metrics(every))
    }

    fn run_samples(&self, id: &str, epoch: Option<u32>) -> Result<RunSamples> {
        let (run, ds) = self.run(id)?;
        let done = run.epochs_done(&run.snap());
        let epoch = epoch.unwrap_or(done);
        if epoch == 0 || epoch > done {
            return Err(Error::new(
                StatusCode::NOT_FOUND,
                "epoch_not_available",
                format!("run '{id}' has no validation results for epoch {epoch} (finished epochs: {done})"),
            ));
        }
        let quality = run.curve.val(epoch).map50;
        let samples = ds
            .sample_ids(16)
            .into_iter()
            .filter_map(|image_id| {
                let (w, h, truth) = ds.truth(&image_id).ok()?;
                let seed = mix64(fnv1a(run.id.as_bytes()) ^ epoch as u64) ^ fnv1a(image_id.as_bytes());
                let pred: Vec<ScoredBox> = fake_detections(&truth, w, h, &run.classes, quality, seed)
                    .into_iter()
                    .filter(|b| b.score >= 0.25)
                    .collect();
                Some(Sample { image_id, dataset_id: ds.meta().id.clone(), width: w, height: h, truth, pred })
            })
            .collect();
        Ok(RunSamples { epoch, samples })
    }

    fn subscribe(&self, id: &str) -> Result<Subscription> {
        Ok(self.run(id)?.0.subscribe())
    }

    fn predict(&self, input: PredictInput) -> Result<PredictResponse> {
        let model = self.get_model(&input.model_id)?;
        let (w, h, truth, seed) = match &input.source {
            PredictSource::Dataset { dataset_id, image_id } => {
                let ds = self.dataset(dataset_id)?;
                let (w, h, truth) = ds.truth(image_id)?;
                // The model only knows its own classes: map the truth by class name and drop the rest.
                let names = &ds.meta().classes;
                let truth: Vec<BBox> = truth
                    .into_iter()
                    .filter_map(|b| {
                        let name = names.get(b.class as usize)?;
                        let k = model.classes.iter().position(|c| c == name)?;
                        Some(BBox { class: k as u32, ..b })
                    })
                    .collect();
                (w, h, truth, fnv1a(format!("{dataset_id}/{image_id}").as_bytes()))
            }
            PredictSource::Upload(bytes) => {
                let (w, h) = image::ImageReader::new(Cursor::new(bytes))
                    .with_guessed_format()
                    .map_err(|e| Error::internal(e.to_string()))?
                    .into_dimensions()
                    .map_err(|e| {
                        Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string())
                            .hint("send a JPEG, PNG, BMP or WebP file in the `image` field")
                    })?;
                let seed = fnv1a(bytes);
                let mut r = Rng::new(seed);
                let n = 1 + r.below(6);
                let nc = model.classes.len().max(1) as u64;
                let truth = (0..n)
                    .map(|_| {
                        let (bw, bh) = (r.range(0.08, 0.4) * w as f32, r.range(0.08, 0.4) * h as f32);
                        let (x, y) = (r.range(0.0, w as f32 - bw), r.range(0.0, h as f32 - bh));
                        BBox { class: r.below(nc) as u32, x1: x, y1: y, x2: x + bw, y2: y + bh }
                    })
                    .collect();
                (w, h, truth, seed)
            }
        };
        let seed = seed ^ fnv1a(model.id.as_bytes());
        let mut boxes: Vec<ScoredBox> =
            fake_detections(&truth, w, h, &model.classes, quality_of(&model), seed)
                .into_iter()
                .filter(|b| b.score >= input.conf)
                .collect();
        boxes.truncate(input.max_det as usize);
        let mut r = Rng::new(seed ^ 0x7131);
        let (.., inf_ms) = arch_spec(model.arch);
        let round = |v: f64| (v * 100.0).round() / 100.0;
        let mp = w as f64 * h as f64 / 1e6;
        let preprocess_ms = round(0.35 + 0.28 * mp + 0.1 * r.f32() as f64);
        let inference_ms =
            round(inf_ms * (model.imgsz as f64 / 640.0).powi(2) * (0.94 + 0.12 * r.f32() as f64));
        let postprocess_ms = round(0.12 + 0.015 * boxes.len() as f64 + 0.05 * r.f32() as f64);
        Ok(PredictResponse {
            model_id: model.id,
            image: folo_api::ImageSize { width: w, height: h },
            boxes,
            timing: folo_api::Timing {
                preprocess_ms,
                inference_ms,
                postprocess_ms,
                total_ms: round(preprocess_ms + inference_ms + postprocess_ms),
            },
            device: "gpu:0".into(),
        })
    }
}

impl MockStore {
    /// `(subscribers, ticker running)` of a run's event hub. For tests that check streams do not leak.
    #[doc(hidden)]
    pub fn hub_state(&self, id: &str) -> (usize, bool) {
        let (run, _) = self.run(id).unwrap();
        (run.subscribers(), run.ticking())
    }
}
