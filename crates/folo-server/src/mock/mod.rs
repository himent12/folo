//! `MockStore`: deterministic, realistic data for UI work before (or without) the engine.
//!
//! Three procedural datasets, six models, six finished runs (one stopped, one failed) with full metric histories,
//! and one live run that advances in real time (100 epochs in ~10 minutes, then it loops). Runs created through the
//! API are simulated the same way. Datasets registered from disk are scanned for real.

mod runs;
mod scene;
mod throughput;
mod val;
mod weights;

use std::io::Cursor;
use std::sync::{Arc, RwLock};

use axum::http::StatusCode;
use folo_api::{
    AmpSupport, Arch, BBox, BenchmarkReport, BenchmarkRequest, Capabilities, ClassMetrics, Dataset,
    DatasetFormat, DatasetIssue, ExportFormat, ExportResult, ExportSpec, GpuBackend, GpuInfo, Head,
    ImageInfo, ImagePage, IssuePage, IssueSeverity, LogLine, Model, ModelEval, ModelMetrics, ModelPrecision,
    PhaseKind, PrCurve, PredictResponse, RegisterDataset, Run, RunEstimate, RunSamples, Sample, ScoredBox,
    Split, Splits, SyntheticDataset, TensorCoreSupport, ThroughputTable, TrainConfig, TrainedWith,
};

use crate::error::{Error, Result};
use crate::index::{Index, read_image};
use crate::stats::StatsBuilder;
use crate::store::{
    ImageData, ImageQuery, IssueQuery, PredictInput, PredictSource, QueueMove, Store, Subscription, Weights,
    WeightsFormat,
};
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

    /// Deterministic `folo dataset check` findings consistent with the stats: one `broken_label` per broken label
    /// file (those images are not listed, as the loader skips them), an `empty_image` per image without boxes, and
    /// a sprinkle of warnings (boxes clipped at the frame, a duplicated image) chosen by the seed.
    fn issues(&self) -> Vec<DatasetIssue> {
        let issue = |kind: &str, severity, file: String, line: Option<u32>, message: String| {
            // The fixes `folo dataset check --fix-dry-run` suggests for these kinds.
            let fix = match (kind, line) {
                ("empty_image", _) => None,
                ("clipped_box", Some(l)) => Some(format!("clip the box on line {l} to the image")),
                (_, Some(l)) => Some(format!("delete line {l}")),
                _ => Some(match message.strip_prefix("same bytes as ") {
                    Some(first) => format!("remove this copy (and its label file); keep {first}"),
                    None => "remove the image and its label file".into(),
                }),
            };
            DatasetIssue { kind: kind.into(), severity, file, line, message, fix }
        };
        let mut out = Vec::new();
        let n = self.labels.len() as u64;
        for k in 0..self.stats.broken_labels {
            let message = match k % 3 {
                0 => "expected 5 values (class cx cy w h) or a polygon, got 4 (the loader skips this image)"
                    .into(),
                1 => format!(
                    "class id {} out of range for {} classes (the loader skips this image)",
                    self.meta.classes.len() + k as usize,
                    self.meta.classes.len()
                ),
                _ => "coordinates out of [0, 1]: labels must be normalised (the loader skips this image)"
                    .into(),
            };
            let kind = ["broken_label", "class_overflow", "out_of_range_box"][(k % 3) as usize];
            let file = format!("labels/train/{:06}.txt", n + k);
            out.push(issue(kind, IssueSeverity::Error, file, Some(1 + (k % 4) as u32), message));
        }
        for i in 0..n {
            let split = self.split_of(i).as_str();
            let labels = &self.labels[i as usize];
            if labels.is_empty() {
                let file = format!("images/{split}/{i:06}.jpg");
                out.push(issue(
                    "empty_image",
                    IssueSeverity::Info,
                    file,
                    None,
                    "no label file (background image)".into(),
                ));
                continue;
            }
            let h = mix64(self.seed ^ 0x155e ^ i);
            if h.is_multiple_of(97) {
                let line = 1 + (h >> 8) % labels.len() as u64;
                let file = format!("labels/{split}/{i:06}.txt");
                let message = "box extends slightly past the image edge".into();
                out.push(issue("clipped_box", IssueSeverity::Warning, file, Some(line as u32), message));
            }
            if h % 389 == 1 && i > 0 {
                let j = (h >> 16) % i;
                let file = format!("images/{split}/{i:06}.jpg");
                let message = format!("same bytes as images/{}/{j:06}.jpg", self.split_of(j).as_str());
                out.push(issue("duplicate_image", IssueSeverity::Warning, file, None, message));
            }
        }
        out
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
    Disk(Dataset, Box<Index>),
}

impl DatasetEntry {
    fn meta(&self) -> &Dataset {
        match self {
            DatasetEntry::Proc(d) => &d.meta,
            DatasetEntry::Disk(meta, _) => meta,
        }
    }

    fn with_stats(&self) -> Dataset {
        let (meta, stats) = match self {
            DatasetEntry::Proc(d) => (&d.meta, &d.stats),
            DatasetEntry::Disk(meta, index) => (meta, &index.stats),
        };
        Dataset { stats: Some(stats.clone()), ..meta.clone() }
    }

    fn list_images(&self, q: &ImageQuery) -> ImagePage {
        match self {
            DatasetEntry::Disk(_, index) => index.list_images(q),
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
            DatasetEntry::Disk(_, index) => read_image(&index.image_path(image_id)?, thumb, None),
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
            DatasetEntry::Disk(_, index) => index.truth(image_id),
            DatasetEntry::Proc(d) => {
                let i = d.index(image_id)?;
                Ok((d.w, d.h, d.labels[i as usize].clone()))
            }
        }
    }

    fn list_issues(&self, q: &IssueQuery) -> IssuePage {
        match self {
            DatasetEntry::Disk(_, index) => index.list_issues(q),
            DatasetEntry::Proc(d) => q.page(d.issues().iter()),
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
    basis: EvalBasis,
}

/// What a model's mock evaluation is derived from.
#[derive(Clone)]
struct EvalBasis {
    per_class: Vec<ClassMetrics>,
    /// The confidence the per-class precision and recall are taken at.
    best_f1_conf: Option<f32>,
    /// Validation instances per class.
    counts: Vec<u64>,
    images: u32,
    seed: u64,
    /// The evaluated dataset's id and name, the split evaluated and why, when it is not `val`.
    dataset: (String, String),
    split: folo_api::Split,
    split_fallback: Option<folo_api::SplitFallback>,
}

impl EvalBasis {
    /// From a run's best epoch and its dataset's validation split (train when there is none).
    fn new(run: &SimRun, ds: &DatasetEntry) -> EvalBasis {
        let best = run.curve.best(run.epochs_done(&run.snap())).map_or(1, |b| b.epoch);
        let split = if ds.meta().splits.val > 0 { Split::Val } else { Split::Train };
        let page =
            ds.list_images(&ImageQuery { split: Some(split), offset: 0, limit: u64::MAX, class: None });
        let mut counts = vec![0u64; run.classes.len()];
        for b in page.images.iter().flat_map(|im| &im.boxes) {
            if let Some(c) = counts.get_mut(b.class as usize) {
                *c += 1;
            }
        }
        let val = run.curve.val(best);
        let meta = ds.meta();
        let used = if split == Split::Val { folo_api::Split::Val } else { folo_api::Split::Train };
        EvalBasis {
            best_f1_conf: val.best_f1_conf,
            per_class: val.per_class,
            counts,
            images: page.total as u32,
            seed: fnv1a(run.id.as_bytes()),
            split: used,
            split_fallback: (used != folo_api::Split::Val).then(|| {
                folo_api::SplitFallback::empty_split(folo_api::Split::Val, used, &meta.name, page.total)
            }),
            dataset: (meta.id.clone(), meta.name.clone()),
        }
    }

    /// PR curves, confusion matrix and per-class metrics, consistent with each other.
    fn eval(&self) -> ModelEval {
        let n = self.per_class.len();
        let recall: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let mut per_class = self.per_class.clone();
        let pr = per_class
            .iter_mut()
            .map(|m| {
                // A wiggled `1 - r^k` envelope; its mean grows with k, so bisect k until the curve's AP is the class
                // mAP50 the run reported for this epoch (the model page and the run page then agree).
                let curve = |k: f32| {
                    let mut p: Vec<f32> = recall
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let wiggle = 1.0 + 0.04 * noise(&[self.seed, m.class as u64, i as u64]);
                            ((1.0 - r.powf(k)) * wiggle).clamp(0.0, 1.0)
                        })
                        .collect();
                    for i in (0..p.len() - 1).rev() {
                        p[i] = p[i].max(p[i + 1]);
                    }
                    p
                };
                let mean = |p: &[f32]| p.iter().sum::<f32>() / p.len() as f32;
                let (mut lo, mut hi) = (1e-3f32, 1e3f32);
                for _ in 0..60 {
                    let mid = (lo * hi).sqrt();
                    if mean(&curve(mid)) < m.map50 { lo = mid } else { hi = mid }
                }
                let precision: Vec<f32> = curve((lo * hi).sqrt()).into_iter().map(runs::r4).collect();
                let ap50 = runs::r4(precision.iter().sum::<f32>() / precision.len() as f32);
                m.map50 = ap50;
                PrCurve { class: m.class, recall: recall.clone(), precision, ap50 }
            })
            .collect();
        // Rows: truth (+ background), columns: prediction (+ background).
        let mut confusion = vec![vec![0u32; n + 1]; n + 1];
        for (c, m) in self.per_class.iter().enumerate() {
            let total = self.counts.get(c).copied().unwrap_or(0) as f32;
            let tp = (total * m.recall).round();
            let wrong = ((total - tp) * 0.3).round();
            confusion[c][c] = tp as u32;
            if n > 1 && wrong > 0.0 {
                // Spread the confusions over the other classes with deterministic weights.
                let others: Vec<usize> = (0..n).filter(|&j| j != c).collect();
                let w: Vec<f32> =
                    others.iter().map(|&j| 1.2 + noise(&[self.seed, c as u64, j as u64, 7])).collect();
                let sum: f32 = w.iter().sum();
                let mut left = wrong as u32;
                for (idx, &j) in others.iter().enumerate() {
                    let share = if idx + 1 == others.len() {
                        left
                    } else {
                        ((wrong * w[idx] / sum).round() as u32).min(left)
                    };
                    confusion[c][j] += share;
                    left -= share;
                }
            }
            let assigned: u32 = confusion[c][..n].iter().sum();
            confusion[c][n] = (total as u32).saturating_sub(assigned);
        }
        for (j, m) in self.per_class.iter().enumerate() {
            // precision = tp / column total, so the column holds tp / p predictions: tp, the confusions from other
            // classes, and false positives on background. Confusions beyond that budget move back to "missed".
            let tp = confusion[j][j];
            let p = m.precision.clamp(0.05, 1.0);
            let budget = ((tp as f32 / p).round() as u32).saturating_sub(tp);
            let from_others: u32 = (0..n).filter(|&c| c != j).map(|c| confusion[c][j]).sum();
            if from_others > budget {
                let mut excess = from_others - budget;
                for c in (0..n).filter(|&c| c != j) {
                    let take = confusion[c][j].min(excess);
                    confusion[c][j] -= take;
                    confusion[c][n] += take;
                    excess -= take;
                }
            }
            let from_others: u32 = (0..n).filter(|&c| c != j).map(|c| confusion[c][j]).sum();
            confusion[n][j] = budget - from_others;
        }
        // Report the precision / recall the matrix actually implies, so every view of the eval agrees.
        for (c, m) in per_class.iter_mut().enumerate() {
            let row: u32 = confusion[c].iter().sum();
            let col: u32 = (0..=n).map(|r| confusion[r][c]).sum();
            if row > 0 {
                m.recall = runs::r4(confusion[c][c] as f32 / row as f32);
            }
            if col > 0 {
                m.precision = runs::r4(confusion[c][c] as f32 / col as f32);
            }
        }
        // The real evaluator's confusion thresholds (docs/METRICS.md §7).
        let thresholds = folo_vision::EvalConfig::default();
        ModelEval {
            pr,
            confusion,
            per_class,
            images: self.images,
            best_f1_conf: self.best_f1_conf,
            dataset_id: Some(self.dataset.0.clone()),
            dataset_name: Some(self.dataset.1.clone()),
            split: Some(self.split),
            split_fallback: self.split_fallback.clone(),
            confusion_conf: Some(thresholds.confusion_conf),
            confusion_iou: Some(thresholds.confusion_iou),
        }
    }
}

/// The mock store. See the module docs.
pub struct MockStore {
    datasets: RwLock<Vec<Arc<DatasetEntry>>>,
    models: RwLock<Vec<StaticModel>>,
    runs: RwLock<Vec<(Arc<SimRun>, Arc<DatasetEntry>)>>,
    /// The runs created through the API, in queue order: they share one training slot ([`runs::reschedule`]).
    queue: std::sync::Mutex<Vec<Arc<SimRun>>>,
    weights: weights::MockWeights,
    tensor_cores: MockTensorCores,
    throughput: crate::throughput::Throughput,
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

/// The mock GPU.
const MOCK_GPU: &str = "NVIDIA GeForce RTX 4080 SUPER";

/// The scripted live run (it loops forever, so it does not block benchmark probes).
const LIVE_RUN: &str = "run_3c02";

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
            let config = TrainConfig {
                name: d.name.into(),
                dataset_id: meta.id.clone(),
                arch: d.arch,
                epochs: d.epochs,
                batch: d.batch,
                amp: d.amp,
                head: d.head,
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
            )
            .resolved_for(meta.splits.train);
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
        }
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
            name: MOCK_GPU.into(),
            backend: self.tensor_cores.backend(),
            memory_bytes: 17_171_480_576,
            util: Some((util * 100.0).round() / 100.0),
        }]
    }

    fn capabilities(&self) -> Option<Capabilities> {
        Some(self.tensor_cores.capabilities())
    }

    fn throughput(&self) -> Result<ThroughputTable> {
        Ok(self.throughput.table(&throughput::gpu(), &[throughput::gpu(), throughput::cpu()]))
    }

    fn estimate_run(&self, config: TrainConfig) -> Result<RunEstimate> {
        let ds = self.dataset(&config.dataset_id)?;
        self.validate(&config)?;
        let splits = ds.meta().splits;
        if ds.meta().classes.is_empty() || splits.train == 0 {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("dataset_id: '{}' has no classes or no training images", ds.meta().id),
            ));
        }
        let augment = folo_data::AugmentConfig {
            close_mosaic: config.augment.close_mosaic,
            ..folo_data::AugmentConfig::default()
        };
        let device = if config.device == "cpu" { throughput::cpu() } else { throughput::gpu() };
        Ok(self.throughput.estimate(&crate::throughput::EstimateInput {
            device,
            arch: config.arch,
            imgsz: config.imgsz,
            amp: config.amp,
            batch: config.batch,
            epochs: config.epochs,
            mosaic: config.augment.mosaic > 0.0,
            close_mosaic_epochs: augment.close_mosaic_epochs(config.epochs),
            train_images: splits.train,
            val_images: if splits.val == 0 { splits.train } else { splits.val },
        }))
    }

    /// Answers at once with numbers near the table's (or the reference's): no device is used. Refused while a run
    /// created through the API is queued or training (the scripted live run does not count).
    fn benchmark(&self, req: BenchmarkRequest) -> Result<BenchmarkReport> {
        crate::probe::check(&req)?;
        let busy =
            self.runs.read().unwrap().iter().any(|r| r.0.id != LIVE_RUN && r.0.snap().status.is_active());
        if busy {
            return Err(Error::conflict(
                "run_active",
                "a training run is queued or running; a probe now would measure both slower",
            )
            .hint("wait until it finishes (or stop it), then probe again"));
        }
        let amp = req.amp && self.tensor_cores == MockTensorCores::On;
        if req.amp && !amp {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                self.tensor_cores.capabilities().amp.reason.unwrap_or_default(),
            ));
        }
        let gpu = throughput::gpu();
        let reference = crate::throughput::reference::train(true, req.arch, req.imgsz, amp);
        let seed = fnv1a(format!("{:?}{}{}{}", req.arch, req.imgsz, req.batch, amp).as_bytes());
        let ips = (reference.mosaic * (1.0 + 0.04 * noise(&[seed, 1]) as f64) * 10.0).round() / 10.0;
        let trained = self.throughput.record_train(crate::throughput::TrainSample {
            key: crate::throughput::TrainKey {
                device: gpu.clone(),
                arch: req.arch,
                imgsz: req.imgsz,
                batch: req.batch,
                amp,
            },
            phase: PhaseKind::Mosaic,
            train_imgs_per_s: Some(ips),
            val_imgs_per_s: None,
            startup_s: None,
            peak_vram_bytes: Some(crate::throughput::reference::memory(req.arch, req.imgsz, req.batch, amp)),
            probe: true,
        });
        let (latency, forward, _) =
            crate::throughput::reference::infer(true, req.arch, req.imgsz, folo_api::InferPrecision::F32);
        let calls = (req.budget_s * 0.4 * 1000.0 / latency).clamp(5.0, 500.0) as u64;
        let mut inferred = None;
        for i in 0..calls.min(crate::throughput::WINDOW as u64) {
            let k = 1.0 + 0.08 * noise(&[seed, 100 + i]) as f64;
            inferred = Some(self.throughput.record_infer(crate::throughput::InferSample {
                key: crate::throughput::InferKey {
                    device: gpu.clone(),
                    arch: req.arch,
                    imgsz: req.imgsz,
                    precision: folo_api::InferPrecision::F32,
                    tensor_cores: false,
                },
                total_ms: (latency + 0.9) * k,
                inference_ms: forward * k,
            }));
        }
        let inference = crate::throughput::infer_row(&inferred.expect("at least 5 predictions"));
        Ok(BenchmarkReport {
            device: gpu.device,
            device_name: gpu.device_name,
            train_imgs_per_s: ips,
            train_batches: ((req.budget_s * 0.6 * 0.7 * ips) / req.batch as f64).max(1.0) as u64,
            latency_ms: inference.latency_ms,
            predictions: calls,
            elapsed_s: ((req.budget_s + 1.8) * 10.0).round() / 10.0,
            training: crate::throughput::train_row(&trained),
            inference,
        })
    }

    fn list_datasets(&self) -> Result<Vec<Dataset>> {
        Ok(self.datasets.read().unwrap().iter().map(|d| d.meta().clone()).collect())
    }

    fn register_dataset(&self, req: RegisterDataset) -> Result<Dataset> {
        // Registering a folder again (by its canonical path) returns the dataset it already is: checked before
        // indexing and again under the write lock.
        let want = std::fs::canonicalize(&req.path).ok();
        let same = |list: &[Arc<DatasetEntry>]| {
            let want = want.as_ref()?;
            list.iter().find_map(|d| match &**d {
                DatasetEntry::Disk(meta, _)
                    if std::fs::canonicalize(&meta.path).ok().as_ref() == Some(want)
                        && (req.format == DatasetFormat::Auto || req.format == meta.format) =>
                {
                    Some(d.with_stats())
                }
                _ => None,
            })
        };
        if let Some(d) = same(&self.datasets.read().unwrap()) {
            return Ok(d);
        }
        let index = Index::build(&req.path, req.format, &[])?;
        let mut list = self.datasets.write().unwrap();
        if let Some(d) = same(&list) {
            return Ok(d);
        }
        let id = unique_id("ds", fnv1a(req.path.as_bytes()) ^ unix_now().to_bits(), |id| {
            list.iter().any(|d| d.meta().id == id)
        });
        let name = req.name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
            index
                .base
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "dataset".into())
        });
        let meta = Dataset {
            id,
            name,
            path: req.path,
            format: index.format,
            classes: index.classes.clone(),
            created: rfc3339(unix_now()),
            splits: index.splits,
            stats: None,
        };
        let entry = DatasetEntry::Disk(meta, Box::new(index));
        let out = entry.with_stats();
        list.push(Arc::new(entry));
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

    fn list_issues(&self, id: &str, q: &IssueQuery) -> Result<IssuePage> {
        Ok(self.dataset(id)?.list_issues(q))
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

    fn model_eval(&self, id: &str) -> Result<ModelEval> {
        if let Some(m) = self.models.read().unwrap().iter().find(|m| m.model.id == id) {
            return Ok(m.basis.eval());
        }
        let runs = self.runs.read().unwrap();
        let (run, ds) = runs
            .iter()
            .find(|(r, _)| r.user && r.current_model(&r.snap()).as_deref() == Some(id))
            .ok_or_else(|| Error::not_found("model", id))?;
        Ok(EvalBasis::new(run, ds).eval())
    }

    fn export_model(&self, id: &str, spec: ExportSpec) -> Result<ExportResult> {
        crate::store::check_export(&spec)?;
        let m = self.get_model(id)?;
        let (ext, bytes) = match spec.format {
            ExportFormat::FoloF16 => ("f16.folo", model_bytes(m.params, true)),
            ExportFormat::Onnx if spec.onnx.half => ("f16.onnx", m.params * 2 + 61_440),
            ExportFormat::Onnx => ("onnx", m.params * 4 + 61_440),
        };
        Ok(ExportResult { path: format!("mock://exports/{}.{ext}", m.name), bytes })
    }

    fn model_weights(&self, id: &str, format: WeightsFormat) -> Result<Weights> {
        self.weights.get(&self.get_model(id)?, format)
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
        let run = Arc::new(
            SimRun::new(
                id,
                config,
                meta.classes.clone(),
                spe,
                curve,
                device,
                Some(model_id),
                true,
                Timeline::Live(std::sync::Mutex::new(clock)),
            )
            .resolved_for(meta.splits.train),
        );
        runs.push((Arc::clone(&run), Arc::clone(&ds)));
        drop(runs);
        {
            let mut queue = self.queue.lock().unwrap();
            queue.push(Arc::clone(&run));
            runs::reschedule(&queue, std::time::Instant::now());
        }
        Ok(run.to_api())
    }

    fn get_run(&self, id: &str) -> Result<Run> {
        Ok(self.run(id)?.0.to_api())
    }

    fn stop_run(&self, id: &str) -> Result<Run> {
        let (run, _) = self.run(id)?;
        if !run.stop() {
            return Err(Error::run_not_active(
                id,
                serde_json::to_value(run.to_api().status).unwrap().as_str().unwrap_or(""),
            ));
        }
        // The runs behind it start earlier now.
        runs::reschedule(&self.queue.lock().unwrap(), std::time::Instant::now());
        Ok(run.to_api())
    }

    fn move_run(&self, id: &str, mv: QueueMove) -> Result<Run> {
        let (run, _) = self.run(id)?;
        let moved = runs::move_waiting(&mut self.queue.lock().unwrap(), id, mv);
        if let Some((from, to, waiting)) = moved {
            let after = run.to_api();
            let place = format!("{} of {waiting}", to + 1);
            let message = match mv {
                QueueMove::Start if after.status == folo_api::RunStatus::Running => {
                    "start requested: starting now".to_string()
                }
                QueueMove::Start => {
                    format!(
                        "start requested: moved to the front ({place}); it starts when the active run ends"
                    )
                }
                _ if from == to => format!("queue position unchanged ({place})"),
                _ => format!("moved in the queue to position {place}"),
            };
            run.note(folo_api::LogLevel::Info, message);
            return Ok(after);
        }
        let what = match run.to_api().status {
            folo_api::RunStatus::Queued => "is not in the training queue".to_string(),
            s => format!("is {}", serde_json::to_value(s).unwrap().as_str().unwrap_or("finished")),
        };
        Err(Error::conflict("run_not_queued", format!("run '{id}' {what}, not waiting in the queue"))
            .hint("only runs waiting in the queue (with a queue_position) can be moved"))
    }

    fn delete_run(&self, id: &str) -> Result<()> {
        let (run, _) = self.run(id)?;
        if run.is_active() {
            return Err(Error::conflict("run_active", format!("run '{id}' is still active"))
                .hint(format!("stop it first: POST /api/runs/{id}/stop")));
        }
        run.delete();
        self.runs.write().unwrap().retain(|r| r.0.id != id);
        self.queue.lock().unwrap().retain(|r| r.id != id);
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

    fn run_logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>> {
        let mut lines = self.run(id)?.0.logs();
        let skip = lines.len().saturating_sub(limit);
        lines.drain(..skip);
        Ok(lines)
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

    fn val(&self, req: folo_api::ValRequest) -> Result<folo_api::ValReport> {
        val::val(self, req)
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

#[cfg(test)]
mod eval_tests {
    use super::*;

    fn basis(per_class: &[(f32, f32)], counts: &[u64]) -> EvalBasis {
        let per_class = per_class
            .iter()
            .enumerate()
            .map(|(c, &(precision, recall))| ClassMetrics {
                class: c as u32,
                map50: 0.7,
                map50_95: 0.5,
                precision,
                recall,
            })
            .collect();
        EvalBasis {
            per_class,
            best_f1_conf: Some(0.3),
            counts: counts.to_vec(),
            images: 10,
            seed: 1,
            dataset: ("ds_t".into(), "t".into()),
            split: folo_api::Split::Val,
            split_fallback: None,
        }
    }

    /// Row sums equal instance counts and `per_class` precision / recall match the matrix.
    fn check(b: &EvalBasis) {
        let ev = b.eval();
        let n = b.per_class.len();
        for c in 0..n {
            assert_eq!(ev.confusion[c].iter().map(|&v| v as u64).sum::<u64>(), b.counts[c], "row {c}");
            let row: u32 = ev.confusion[c].iter().sum();
            let col: u32 = (0..=n).map(|r| ev.confusion[r][c]).sum();
            let tp = ev.confusion[c][c] as f32;
            if row > 0 {
                assert!((ev.per_class[c].recall - tp / row as f32).abs() < 1e-3);
            }
            if col > 0 {
                assert!((ev.per_class[c].precision - tp / col as f32).abs() < 1e-3);
            }
        }
        assert_eq!(ev.confusion[n][n], 0);
    }

    #[test]
    fn confusion_is_consistent() {
        // Balanced, imbalanced (a rare class that receives many confusions), zero-count and single-class cases.
        check(&basis(&[(0.9, 0.85), (0.8, 0.7), (0.7, 0.6)], &[400, 380, 390]));
        check(&basis(&[(0.6, 0.4), (0.95, 0.9), (0.5, 0.3)], &[5000, 3, 1200]));
        check(&basis(&[(0.9, 0.8), (0.9, 0.8)], &[100, 0]));
        check(&basis(&[(0.05, 0.02)], &[50]));
        check(&basis(&[(1.0, 1.0), (0.01, 0.01)], &[1, 1]));
    }
}
