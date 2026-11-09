//! Validation: a model on a dataset split, scored with [`Evaluator`] (docs/MODEL.md §8.7, docs/METRICS.md).
//!
//! Images are letterboxed to a square `imgsz` by folo-data's validation transform; ground-truth boxes go through
//! the same resize and pad, so predictions are compared in network-input pixels with no inverse map. The first
//! [`ValConfig::samples`] images of the split are also returned as [`ValSample`]s (truth and predictions mapped back
//! to the original image), the fixed sample set of docs/API.md's `/samples`.

use std::sync::Arc;
use std::time::Instant;

use folo_core::{DType, Device, Tensor};
use folo_data::{Batch, DataLoader, Dataset, LoaderConfig};
use folo_nn::Module;
use serde::Serialize;

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{EvalReport, Evaluator, Truth};
use crate::model::{FoloModel, HeadMode};
use crate::nms::{Detection, NmsConfig};

/// Validation settings. [`ValConfig::new`] gives MODEL.md §8.7's.
#[derive(Clone, Debug, PartialEq)]
pub struct ValConfig {
    /// Square input size (a multiple of 32).
    pub imgsz: u32,
    /// Images per forward pass.
    pub batch: usize,
    /// Loader worker threads.
    pub workers: usize,
    /// Post-processing: [`NmsConfig::val`] (`conf 0.001`, `iou 0.7`, multi-label, `max_det 300`).
    pub nms: NmsConfig,
    /// How many leading images of the split to return as [`ValSample`]s.
    pub samples: usize,
    /// Score threshold of the sample predictions (docs/API.md: 0.25).
    pub sample_conf: f32,
    /// Also return every image's detections in original-image pixels ([`ValResult::detections`]), e.g. for
    /// [`crate::cocoeval::CocoEval`] or a COCO results file.
    pub keep_detections: bool,
    /// The head branch to evaluate (docs/MODEL.md §6.6): `None` uses the model's ([`FoloModel::head_mode`]),
    /// `o2m` the one-to-many branch with NMS, `o2o` the one-to-one branch with its top-k (no NMS; `iou` unused).
    pub head: Option<HeadMode>,
}

impl ValConfig {
    /// MODEL.md §8.7 at `imgsz` with `batch` images per forward pass and 16 samples.
    pub fn new(imgsz: u32, batch: usize) -> Self {
        ValConfig {
            imgsz,
            batch,
            workers: crate::train::default_workers(),
            nms: NmsConfig::val(),
            samples: 16,
            sample_conf: 0.25,
            keep_detections: false,
            head: None,
        }
    }
}

/// One validation image with its ground truth and predictions, in original-image pixels.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ValSample {
    /// Index of the image in the split.
    pub index: usize,
    /// Image file, for file-backed datasets.
    pub file: Option<String>,
    /// Original width.
    pub width: u32,
    /// Original height.
    pub height: u32,
    /// Ground truth.
    pub truth: Vec<Truth>,
    /// Predictions with `score >= sample_conf`, highest first.
    pub pred: Vec<Detection>,
}

/// What [`validate`] returns.
#[derive(Clone, Debug)]
pub struct ValResult {
    /// The metrics.
    pub report: EvalReport,
    /// The sample images.
    pub samples: Vec<ValSample>,
    /// With [`ValConfig::keep_detections`]: per image, in loader order, its dataset index and its detections (all
    /// that NMS kept) in original-image pixels, clipped to the image.
    pub detections: Vec<(usize, Vec<Detection>)>,
    /// Wall time in seconds.
    pub seconds: f64,
}

/// Wrap a loader batch as an `[n, 3, h, w]` tensor of `dtype` (`F32`, or `F16` for a half-precision model) in
/// `0..1` on `device`. A `u8` batch ([`LoaderConfig::u8_images`]) is uploaded as bytes and converted there
/// (`value / 255`, computed in f32).
pub(crate) fn batch_tensor(batch: &mut Batch, device: Device, dtype: DType) -> Result<Tensor> {
    if !batch.images_u8.is_empty() {
        let t = Tensor::from_vec(std::mem::take(&mut batch.images_u8), batch.shape)?;
        let t = if device.is_cpu() { t } else { t.to_device(device)? };
        return Ok(t.to_dtype(dtype)?.div_scalar(255.0)?);
    }
    let images = std::mem::take(&mut batch.images);
    let t = Tensor::from_vec(images, batch.shape)?;
    let t = if device.is_cpu() { t } else { t.to_device(device)? };
    Ok(t.to_dtype(dtype)?)
}

/// The ground truth of image `i` of a batch, in network-input pixels.
pub(crate) fn batch_truths(batch: &Batch) -> Vec<Vec<Truth>> {
    let [n, _, h, w] = batch.shape;
    let (fw, fh) = (w as f32, h as f32);
    let mut out = vec![Vec::new(); n];
    for &[bi, cls, cx, cy, bw, bh] in &batch.targets {
        let (cx, cy, hw, hh) = (cx * fw, cy * fh, bw * fw * 0.5, bh * fh * 0.5);
        out[bi as usize]
            .push(Truth { bbox: Xyxy::new(cx - hw, cy - hh, cx + hw, cy + hh), class: cls as u32 });
    }
    out
}

/// Evaluate `model` (in eval mode, fused or not; `F32`, or `F16` after [`Module::to_dtype`] on a GPU that computes
/// in it) on `ds`. Every class id of the dataset must be below the model's class count.
pub fn validate(model: &FoloModel, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<ValResult> {
    let t0 = Instant::now();
    if model.is_training() {
        return Err(Error::Config("validate: the model is in training mode; call eval() first".into()));
    }
    let n = &cfg.nms;
    let unit = |x: f32| (0.0..=1.0).contains(&x); // false for NaN
    if !unit(n.conf) || !unit(n.iou) || !unit(cfg.sample_conf) || n.max_det == 0 || n.max_candidates == 0 {
        return Err(Error::Config(format!(
            "validation needs conf, iou and sample_conf in 0..=1 and max_det, max_candidates >= 1; got conf {} iou \
             {} sample_conf {} max_det {} max_candidates {}",
            n.conf, n.iou, cfg.sample_conf, n.max_det, n.max_candidates
        )));
    }
    let head = cfg.head.unwrap_or(model.head_mode());
    if head == HeadMode::O2o && !model.has_o2o() {
        return Err(Error::Config(format!(
            "validation with head o2o needs a model with the one-to-one head; {} has head `{}`",
            model.arch(),
            model.config().head
        )));
    }
    let nc = model.num_classes();
    if ds.class_names().len() > nc {
        return Err(Error::Dataset(format!(
            "the dataset has {} classes but the model only {nc}",
            ds.class_names().len()
        )));
    }
    if !cfg.imgsz.is_multiple_of(32) || cfg.imgsz == 0 || cfg.batch == 0 {
        return Err(Error::Config(format!(
            "validation needs imgsz a positive multiple of 32 and batch >= 1, got imgsz {} batch {}",
            cfg.imgsz, cfg.batch
        )));
    }
    let mut ev = Evaluator::new(nc);
    let mut samples = Vec::new();
    if ds.is_empty() {
        return Ok(ValResult {
            report: ev.compute(),
            samples,
            detections: Vec::new(),
            seconds: t0.elapsed().as_secs_f64(),
        });
    }
    let device = model.device();
    // A device gets the pixels as bytes (a quarter of the upload) and converts them itself.
    let loader = DataLoader::new(
        ds.clone(),
        LoaderConfig {
            workers: cfg.workers,
            u8_images: !device.is_cpu(),
            ..LoaderConfig::val(cfg.batch, cfg.imgsz)
        },
    )?;
    let mut index = 0usize;
    let mut kept = Vec::new();
    for batch in loader.epoch(0) {
        let mut batch = batch?;
        let truths = batch_truths(&batch);
        let x = batch_tensor(&mut batch, device, model.dtype())?;
        let pred = model.predict_raw_as(&x, &cfg.nms, head)?;
        for (i, img) in pred.images.iter().enumerate() {
            ev.add_image(&img.detections, &truths[i]);
            let meta = &batch.meta[i];
            let unmap = |b: &Xyxy| {
                let [x1, y1, x2, y2] = meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
                Xyxy::new(x1, y1, x2, y2)
            };
            if cfg.keep_detections {
                kept.push((
                    meta.index,
                    img.detections.iter().map(|d| Detection { bbox: unmap(&d.bbox), ..*d }).collect(),
                ));
            }
            if index < cfg.samples {
                samples.push(ValSample {
                    index: meta.index,
                    file: ds.image_path(meta.index).map(|p| p.display().to_string()),
                    width: meta.orig_size[0],
                    height: meta.orig_size[1],
                    truth: truths[i].iter().map(|t| Truth { bbox: unmap(&t.bbox), class: t.class }).collect(),
                    pred: img
                        .detections
                        .iter()
                        .filter(|d| d.score >= cfg.sample_conf)
                        .map(|d| Detection { bbox: unmap(&d.bbox), ..*d })
                        .collect(),
                });
            }
            index += 1;
        }
    }
    Ok(ValResult { report: ev.compute(), samples, detections: kept, seconds: t0.elapsed().as_secs_f64() })
}

/// docs/API.md's `ValMetrics` of a report. `best_f1_conf` is the report's `best_conf`, `None` when no class has
/// ground truth (there is no operating point then).
pub fn val_metrics(r: &EvalReport) -> folo_api::ValMetrics {
    folo_api::ValMetrics {
        best_f1_conf: (!r.per_class.is_empty()).then_some(r.best_conf),
        map50: r.map50,
        map50_95: r.map50_95,
        precision: r.precision,
        recall: r.recall,
        per_class: r
            .per_class
            .iter()
            .map(|c| folo_api::ClassMetrics {
                class: c.class,
                map50: c.map50,
                map50_95: c.map50_95,
                precision: c.precision,
                recall: c.recall,
            })
            .collect(),
    }
}
