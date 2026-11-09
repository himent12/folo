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
    let ev = Evaluator::new(nc);
    let samples = Vec::new();
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
    // The download of a batch's decoded outputs and the host post-processing (NMS per image, metrics, samples)
    // run on a second thread while this one encodes the next batches' forward passes, so the GPU does not wait
    // for the host. Batches reach that thread in order, so the result is the serial one.
    let (tx, rx) = std::sync::mpsc::sync_channel::<Decoded>(1);
    let (ev, samples, kept) = std::thread::scope(|scope| -> Result<_> {
        // Owned here, so an early error return closes the channel and the scope can join the thread.
        let tx = tx;
        let post = scope.spawn(|| -> Result<Post> {
            let mut post = Post { ev, samples, kept: Vec::new(), index: 0 };
            for d in rx {
                post.add(d, &*ds, cfg)?;
            }
            Ok(post)
        });
        for batch in loader.epoch(0) {
            let mut batch = batch?;
            let truths = batch_truths(&batch);
            let x = batch_tensor(&mut batch, device, model.dtype())?;
            let output = {
                let _g = folo_core::no_grad();
                model.forward_head(&x, head)?
            };
            drop(x);
            let (nc, mode) = (output.num_classes, output.mode);
            let images = if output.box_raw.device().is_cpu() {
                Outputs::Host(output.decode_images()?)
            } else {
                // Encoded here, downloaded by the other thread (HeadOutput::decode_images' device path).
                Outputs::Device(output.decoded_on_device()?, output.num_anchors())
            };
            drop(output);
            if tx.send(Decoded { images, nc, mode, truths, meta: batch.meta }).is_err() {
                break; // the post-processing thread failed; joining it below reports why
            }
        }
        drop(tx);
        let post = post.join().unwrap_or_else(|p| std::panic::resume_unwind(p))?;
        Ok((post.ev, post.samples, post.kept))
    })?;
    Ok(ValResult { report: ev.compute(), samples, detections: kept, seconds: t0.elapsed().as_secs_f64() })
}

/// One batch's decoded outputs and ground truth, on their way to [`Post`].
struct Decoded {
    images: Outputs,
    nc: usize,
    mode: HeadMode,
    truths: Vec<Vec<Truth>>,
    meta: Vec<folo_data::ImageMeta>,
}

/// A batch's decoded outputs ([`crate::model::HeadOutput::decode_images`]).
enum Outputs {
    /// Per image: decoded boxes and class scores.
    Host(Vec<(Vec<Xyxy>, Vec<f32>)>),
    /// [`crate::model::HeadOutput::decoded_on_device`] (not read back yet) and the anchor count.
    Device(Tensor, usize),
}

/// The host half of [`validate`]: post-processing, metrics, samples and kept detections, batch by batch in order.
struct Post {
    ev: Evaluator,
    samples: Vec<ValSample>,
    kept: Vec<(usize, Vec<Detection>)>,
    /// Images seen so far.
    index: usize,
}

impl Post {
    fn add(&mut self, d: Decoded, ds: &dyn Dataset, cfg: &ValConfig) -> Result<()> {
        use rayon::prelude::*;
        let Decoded { images, nc, mode, truths, meta } = d;
        let images = match images {
            Outputs::Host(images) => images,
            Outputs::Device(t, a) => crate::model::HeadOutput::split_decoded(&t.to_vec::<f32>()?, a, nc),
        };
        let dets: Vec<Vec<Detection>> = images
            .into_par_iter()
            .map(|(b, s)| crate::nms::postprocess(&b, &s, nc, &cfg.nms, mode))
            .collect();
        for (i, dets) in dets.into_iter().enumerate() {
            self.ev.add_image(&dets, &truths[i]);
            let meta = &meta[i];
            let unmap = |b: &Xyxy| {
                let [x1, y1, x2, y2] = meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
                Xyxy::new(x1, y1, x2, y2)
            };
            if self.index < cfg.samples {
                self.samples.push(ValSample {
                    index: meta.index,
                    file: ds.image_path(meta.index).map(|p| p.display().to_string()),
                    width: meta.orig_size[0],
                    height: meta.orig_size[1],
                    truth: truths[i].iter().map(|t| Truth { bbox: unmap(&t.bbox), class: t.class }).collect(),
                    pred: dets
                        .iter()
                        .filter(|d| d.score >= cfg.sample_conf)
                        .map(|d| Detection { bbox: unmap(&d.bbox), ..*d })
                        .collect(),
                });
            }
            if cfg.keep_detections {
                self.kept.push((
                    meta.index,
                    dets.iter().map(|d| Detection { bbox: unmap(&d.bbox), ..*d }).collect(),
                ));
            }
            self.index += 1;
        }
        Ok(())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use folo_data::Split;
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    use folo_nn::io::ModelConfig;

    /// The pipelined [`validate`] (post-processing on a second thread) gives exactly what the serial loop gives:
    /// predict, then score each image in order. With conf 0 an untrained model keeps a box at every anchor, so NMS,
    /// the metrics, the samples and the kept detections all have work to do.
    #[test]
    fn validate_matches_the_serial_loop() {
        check_against_serial(Device::Cpu);
    }

    /// The same on a GPU, where the other thread also downloads the decoded outputs.
    #[test]
    fn validate_matches_the_serial_loop_on_gpu() {
        let ctx = match folo_gpu::GpuContext::shared() {
            Ok(c) => c,
            Err(e) => return eprintln!("note: skipping GPU test, no adapter: {e}"),
        };
        if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
            return eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
        }
        check_against_serial(folo_gpu::init().unwrap());
    }

    fn check_against_serial(device: Device) {
        let cfg = SyntheticConfig {
            images: 10,
            classes: vec!["circle".into(), "square".into()],
            imgsz: 64,
            seed: 5,
            val_fraction: 0.0,
            ..SyntheticConfig::default()
        };
        let ds: Arc<dyn Dataset> = Arc::new(SyntheticDataset::new(cfg, Split::Train).unwrap());
        let model = FoloModel::new(ModelConfig::new("n", 2).unwrap()).unwrap();
        let model = crate::train::place_model(model, device).unwrap();
        model.eval();
        let mut vc = ValConfig { samples: 5, keep_detections: true, workers: 2, ..ValConfig::new(64, 4) };
        // Untrained scores sit near the class prior, so keep everything: NMS then has every anchor to sort out.
        (vc.nms.conf, vc.sample_conf) = (0.0, 0.0);
        let got = validate(&model, ds.clone(), &vc).unwrap();

        let lc = LoaderConfig { workers: 0, u8_images: !device.is_cpu(), ..LoaderConfig::val(4, 64) };
        let loader = DataLoader::new(ds.clone(), lc).unwrap();
        let mut ev = Evaluator::new(2);
        let (mut samples, mut kept, mut index) = (Vec::new(), Vec::new(), 0);
        for batch in loader.epoch(0) {
            let mut batch = batch.unwrap();
            let truths = batch_truths(&batch);
            let x = batch_tensor(&mut batch, device, DType::F32).unwrap();
            let pred = model.predict_raw(&x, &vc.nms).unwrap();
            for (i, img) in pred.images.iter().enumerate() {
                ev.add_image(&img.detections, &truths[i]);
                let meta = &batch.meta[i];
                let unmap = |b: &Xyxy| {
                    let [x1, y1, x2, y2] = meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
                    Xyxy::new(x1, y1, x2, y2)
                };
                if index < vc.samples {
                    samples.push(ValSample {
                        index: meta.index,
                        file: None,
                        width: meta.orig_size[0],
                        height: meta.orig_size[1],
                        truth: truths[i]
                            .iter()
                            .map(|t| Truth { bbox: unmap(&t.bbox), class: t.class })
                            .collect(),
                        pred: img
                            .detections
                            .iter()
                            .filter(|d| d.score >= vc.sample_conf)
                            .map(|d| Detection { bbox: unmap(&d.bbox), ..*d })
                            .collect(),
                    });
                }
                kept.push((
                    meta.index,
                    img.detections.iter().map(|d| Detection { bbox: unmap(&d.bbox), ..*d }).collect(),
                ));
                index += 1;
            }
        }
        assert!(
            kept.iter().map(|k: &(usize, Vec<Detection>)| k.1.len()).sum::<usize>() > 100,
            "too few boxes to test"
        );
        assert_eq!(got.report, ev.compute());
        assert_eq!(got.samples, samples);
        assert_eq!(got.detections, kept);
    }
}
