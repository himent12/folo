//! Validation: a model on a dataset split, scored with [`Evaluator`] (docs/MODEL.md §8.7, docs/METRICS.md).
//!
//! Images are letterboxed to a square `imgsz` by folo-data's validation transform; ground-truth boxes go through
//! the same resize and pad, so predictions are compared in network-input pixels with no inverse map. The first
//! [`ValConfig::samples`] images of the split are also returned as [`ValSample`]s (truth and predictions mapped back
//! to the original image), the fixed sample set of docs/API.md's `/samples`.

use std::sync::Arc;
use std::time::Instant;

use folo_core::{Device, Tensor};
use folo_data::{Batch, DataLoader, Dataset, LoaderConfig};
use folo_nn::Module;
use serde::Serialize;

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::metrics::{EvalReport, Evaluator, Truth};
use crate::model::FoloModel;
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
    /// Wall time in seconds.
    pub seconds: f64,
}

/// Wrap a loader batch as an `[n, 3, h, w]` tensor on `device`.
pub(crate) fn batch_tensor(batch: &mut Batch, device: Device) -> Result<Tensor> {
    let images = std::mem::take(&mut batch.images);
    let t = Tensor::from_vec(images, batch.shape)?;
    Ok(if device.is_cpu() { t } else { t.to_device(device)? })
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

/// Evaluate `model` (in eval mode, fused or not) on `ds`. Every class id of the dataset must be below the model's
/// class count.
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
        return Ok(ValResult { report: ev.compute(), samples, seconds: t0.elapsed().as_secs_f64() });
    }
    let loader = DataLoader::new(
        ds.clone(),
        LoaderConfig { workers: cfg.workers, ..LoaderConfig::val(cfg.batch, cfg.imgsz) },
    )?;
    let device = model.device();
    let mut index = 0usize;
    for batch in loader.epoch(0) {
        let mut batch = batch?;
        let truths = batch_truths(&batch);
        let x = batch_tensor(&mut batch, device)?;
        let pred = model.predict_raw(&x, &cfg.nms)?;
        for (i, img) in pred.images.iter().enumerate() {
            ev.add_image(&img.detections, &truths[i]);
            if index < cfg.samples {
                let meta = &batch.meta[i];
                let unmap = |b: &Xyxy| {
                    let [x1, y1, x2, y2] = meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
                    Xyxy::new(x1, y1, x2, y2)
                };
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
    Ok(ValResult { report: ev.compute(), samples, seconds: t0.elapsed().as_secs_f64() })
}

/// docs/API.md's `ValMetrics` of a report.
pub fn val_metrics(r: &EvalReport) -> folo_api::ValMetrics {
    folo_api::ValMetrics {
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
