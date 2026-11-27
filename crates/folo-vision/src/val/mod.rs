//! Validation: a model on a dataset split, scored with [`Evaluator`] (docs/MODEL.md §8.7, docs/METRICS.md).
//!
//! Images are letterboxed to a square `imgsz` by folo-data's validation transform; ground-truth boxes go through
//! the same resize and pad, so predictions are compared in network-input pixels with no inverse map. The first
//! [`ValConfig::samples`] images of the split are also returned as [`ValSample`]s (truth and predictions mapped back
//! to the original image), the fixed sample set of docs/API.md's `/samples`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use folo_core::{DType, Device, Tensor};
use folo_data::{Batch, DataLoader, Dataset, LoaderConfig};
use folo_nn::Module;
use serde::Serialize;

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::eval::{ThresholdCalibration, validate_per_class_conf};
#[cfg(test)]
use crate::fusion::BoxFusion;
use crate::metrics::{EvalConfig, EvalReport, Evaluator, Truth};
use crate::model::v2::{DEFAULT_IOU_FUSION_ALPHA, FoloV2Model};
use crate::model::{Detector, DetectorOutput, DflAux, DflAuxOutputs, FoloModel, HeadMode};
use crate::nms::{Detection, DflInputs, NmsConfig};

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
    /// Evaluation settings forwarded to [`Evaluator::with_config`] (including [`EvalConfig::record_diagnostics`]
    /// for per-image TP/FP/FN error diagnosis).
    pub eval: EvalConfig,
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
            eval: EvalConfig::default(),
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

impl ValResult {
    /// Global and per-class F1-optimal confidence thresholds computed during validation (zero extra inference
    /// passes).
    pub fn thresholds(&self) -> &ThresholdCalibration {
        &self.report.thresholds
    }

    /// Per-class F1-optimal confidence threshold map ready for [`crate::Predictor::with_per_class_conf`] or
    /// [`crate::NmsConfig::per_class_conf`].
    pub fn per_class_conf(&self) -> BTreeMap<u32, f32> {
        self.report.thresholds.per_class_map()
    }
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

/// `dets` (network-input pixels) clipped to the letterboxed image inside the input, `[pad, pad + size · ratio]`
/// per axis: the boxes [`Evaluator`] scores, as the COCO path scores them clipped to the original image
/// ([`folo_data::ImageMeta::unmap_xyxy`]). The ground truth lies inside the image, so a box reaching into the
/// padding would otherwise lose IoU for area no truth can cover.
pub(crate) fn clipped_to_image(dets: &[Detection], meta: &folo_data::ImageMeta) -> Vec<Detection> {
    let ([rx, ry], [px, py]) = (meta.ratio, meta.pad);
    let (x2, y2) = (px + meta.orig_size[0] as f32 * rx, py + meta.orig_size[1] as f32 * ry);
    dets.iter()
        .map(|d| Detection {
            bbox: Xyxy::new(
                d.bbox.x1.clamp(px, x2),
                d.bbox.y1.clamp(py, y2),
                d.bbox.x2.clamp(px, x2),
                d.bbox.y2.clamp(py, y2),
            ),
            ..*d
        })
        .collect()
}

enum ValModel<'a> {
    V1(&'a FoloModel),
    V2(&'a FoloV2Model),
}

impl<'a> ValModel<'a> {
    fn from_detector(d: &'a Detector) -> Self {
        match d {
            Detector::V1(m) => ValModel::V1(m),
            Detector::V2(m) => ValModel::V2(m),
        }
    }
    fn is_training(&self) -> bool {
        match self {
            ValModel::V1(m) => m.is_training(),
            ValModel::V2(m) => m.is_training(),
        }
    }
    fn head_mode(&self) -> HeadMode {
        match self {
            ValModel::V1(m) => m.head_mode(),
            ValModel::V2(m) => m.head_mode(),
        }
    }
    fn has_o2o(&self) -> bool {
        match self {
            ValModel::V1(m) => m.has_o2o(),
            ValModel::V2(m) => m.has_o2o(),
        }
    }
    fn arch(&self) -> String {
        match self {
            ValModel::V1(m) => m.arch(),
            ValModel::V2(m) => m.arch(),
        }
    }
    fn head_name(&self) -> &str {
        match self {
            ValModel::V1(m) => m.config().head.as_str(),
            ValModel::V2(m) => m.config().head.as_str(),
        }
    }
    fn num_classes(&self) -> usize {
        match self {
            ValModel::V1(m) => m.num_classes(),
            ValModel::V2(m) => m.num_classes(),
        }
    }
    fn device(&self) -> Device {
        match self {
            ValModel::V1(m) => m.device(),
            ValModel::V2(m) => m.device(),
        }
    }
    fn dtype(&self) -> DType {
        match self {
            ValModel::V1(m) => m.dtype(),
            ValModel::V2(m) => m.dtype(),
        }
    }
    fn forward_head(&self, x: &Tensor, mode: HeadMode) -> Result<DetectorOutput> {
        Ok(match self {
            ValModel::V1(m) => DetectorOutput::from_v1(m.forward_head(x, mode)?),
            ValModel::V2(m) => DetectorOutput::from_v2(m.forward_head(x, mode)?),
        })
    }
}

/// Evaluate `model` (in eval mode, fused or not; `F32`, or `F16` after [`Module::to_dtype`] on a GPU that computes
/// in it) on `ds`. Every class id of the dataset must be below the model's class count.
pub fn validate(model: &FoloModel, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<ValResult> {
    validate_counted_inner(ValModel::V1(model), ds, cfg).map(|(r, _)| r)
}

/// Evaluate a [`FoloV2Model`] (`folo2-*`) on `ds` (scoring with [`Evaluator`], fusing `iou_raw` when present).
pub fn validate_v2(model: &FoloV2Model, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<ValResult> {
    validate_counted_inner(ValModel::V2(model), ds, cfg).map(|(r, _)| r)
}

/// Evaluate a unified [`Detector`] (`folo-*` or `folo2-*`) on `ds`.
pub fn validate_detector(model: &Detector, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<ValResult> {
    validate_counted_inner(ValModel::from_detector(model), ds, cfg).map(|(r, _)| r)
}

#[cfg(test)]
fn validate_counted(model: &FoloModel, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<(ValResult, usize)> {
    validate_counted_inner(ValModel::V1(model), ds, cfg)
}

/// [`validate`], and how many images were post-processed from device candidate pairs ([`Outputs::Pairs`] within
/// their capacity), so the tests can tell that path ran.
fn validate_counted_inner(
    model: ValModel<'_>,
    ds: Arc<dyn Dataset>,
    cfg: &ValConfig,
) -> Result<(ValResult, usize)> {
    let t0 = Instant::now();
    check_config_inner(&model, ds.class_names().len(), cfg)?;
    let head = cfg.head.unwrap_or(model.head_mode());
    let nc = model.num_classes();
    let ev = Evaluator::with_config(nc, cfg.eval.clone());
    let samples = Vec::new();
    if ds.is_empty() {
        let seconds = t0.elapsed().as_secs_f64();
        return Ok((ValResult { report: ev.compute(), samples, detections: Vec::new(), seconds }, 0));
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
    // The most candidate pairs an image had so far, for the capacity of the next batches' candidate buffers.
    let most = AtomicUsize::new(0);
    let (ev, samples, kept, from_pairs) = std::thread::scope(|scope| -> Result<_> {
        // Owned here, so an early error return closes the channel and the scope can join the thread.
        let tx = tx;
        let post = scope.spawn(|| -> Result<Post> {
            let mut post = Post { ev, samples, kept: Vec::new(), index: 0, most: &most, from_pairs: 0 };
            for d in rx {
                post.add(d, &*ds, cfg)?;
            }
            Ok(post)
        });
        for batch in loader.epoch(0) {
            let mut batch = batch?;
            let truths = batch_truths(&batch);
            let x = batch_tensor(&mut batch, device, model.dtype())?;
            let det_out = {
                let _g = folo_core::no_grad();
                model.forward_head(&x, head)?
            };
            drop(x);
            let has_iou = det_out.iou_raw.is_some();
            let output = &det_out.head;
            let (nc, mode) = (output.num_classes, output.mode);
            let a = output.num_anchors();
            let cap = pair_capacity(most.load(Ordering::Relaxed), a, nc);
            // DFL fusion / scoring metadata: on a GPU this encodes the device stats without syncing; the post
            // thread downloads them alongside the decoded outputs so the forward thread never stalls.
            let aux = output.dfl_aux_outputs(&cfg.nms)?;
            let images = if has_iou {
                Outputs::Host(det_out.decode_images(DEFAULT_IOU_FUSION_ALPHA)?)
            } else if output.box_raw.device().is_cpu() {
                Outputs::Host(output.decode_images()?)
            } else if let Some(cap) = cap
                && mode == HeadMode::O2m
                && pairs_apply(&cfg.nms)
                && folo_gpu::detect::supported(device)
            {
                // `[B, 4 + nc, A]`: the transpose of the `[B, A, 4 + nc]` view is the dense decode itself.
                let decoded = output.decoded_on_device()?.transpose(1, 2)?;
                let pairs =
                    folo_gpu::detect::multi_label_candidates(&decoded, cfg.nms.candidate_conf(), cap)?;
                let boxes = decoded.narrow(1, 0, 4)?;
                Outputs::Pairs { pairs, boxes, decoded, cap, anchors: a }
            } else {
                // Encoded here, downloaded by the other thread (HeadOutput::decode_images' device path).
                Outputs::Device(output.decoded_on_device()?, a)
            };
            drop(det_out);
            if tx.send(Decoded { images, aux, nc, mode, truths, meta: batch.meta }).is_err() {
                break; // the post-processing thread failed; joining it below reports why
            }
        }
        drop(tx);
        let post = post.join().unwrap_or_else(|p| std::panic::resume_unwind(p))?;
        Ok((post.ev, post.samples, post.kept, post.from_pairs))
    })?;
    let seconds = t0.elapsed().as_secs_f64();
    Ok((ValResult { report: ev.compute(), samples, detections: kept, seconds }, from_pairs))
}

/// What [`validate`] refuses before it reads an image, for a dataset of `classes` classes: a caller with something
/// to prepare for the results (a folder) checks this first, so a refused request leaves nothing behind.
pub fn check_config(model: &FoloModel, classes: usize, cfg: &ValConfig) -> Result<()> {
    check_config_inner(&ValModel::V1(model), classes, cfg)
}

/// [`check_config`] for a unified [`Detector`] (`folo-*` or `folo2-*`).
pub fn check_config_detector(model: &Detector, classes: usize, cfg: &ValConfig) -> Result<()> {
    check_config_inner(&ValModel::from_detector(model), classes, cfg)
}

fn check_config_inner(model: &ValModel<'_>, classes: usize, cfg: &ValConfig) -> Result<()> {
    if model.is_training() {
        return Err(Error::Config("validate: the model is in training mode; call eval() first".into()));
    }
    let n = &cfg.nms;
    let unit = |x: f32| (0.0..=1.0).contains(&x); // false for NaN
    if !unit(n.conf)
        || !unit(n.iou)
        || !unit(n.fusion_iou)
        || !unit(cfg.sample_conf)
        || n.max_det == 0
        || n.max_candidates == 0
    {
        return Err(Error::Config(format!(
            "validation needs conf, iou, fusion_iou and sample_conf in 0..=1 and max_det, max_candidates >= 1; got \
             conf {} iou {} fusion_iou {} sample_conf {} max_det {} max_candidates {}",
            n.conf, n.iou, n.fusion_iou, cfg.sample_conf, n.max_det, n.max_candidates
        )));
    }
    if let Some(map) = &n.per_class_conf {
        validate_per_class_conf(map)?;
    }
    if let crate::nms::NmsKind::SoftGaussian { sigma } | crate::nms::NmsKind::MatrixGaussian { sigma } =
        n.nms_kind
        && (!sigma.is_finite() || sigma <= 0.0)
    {
        return Err(Error::Config(format!(
            "validation Gaussian NMS sigma must be finite and > 0; got {sigma}"
        )));
    }
    if n.nms_kind.is_soft()
        && (n.score != crate::fusion::ScoreRule::None || n.fusion == crate::fusion::BoxFusion::Sebf)
    {
        return Err(Error::Config(
            "validation cannot combine Soft-NMS or Matrix-NMS with ScoreRule rescoring or BoxFusion::Sebf \
             (score-decaying NMS has no hard suppression clusters)"
                .into(),
        ));
    }
    if cfg.head.unwrap_or(model.head_mode()) == HeadMode::O2o && !model.has_o2o() {
        return Err(Error::Config(format!(
            "validation with head o2o needs a model with the one-to-one head; {} has head `{}`",
            model.arch(),
            model.head_name()
        )));
    }
    let nc = model.num_classes();
    if classes > nc {
        return Err(Error::Dataset(format!("the dataset has {classes} classes but the model only {nc}")));
    }
    if !cfg.imgsz.is_multiple_of(32) || cfg.imgsz == 0 || cfg.batch == 0 {
        return Err(Error::Config(format!(
            "validation needs imgsz a positive multiple of 32 and batch >= 1, got imgsz {} batch {}",
            cfg.imgsz, cfg.batch
        )));
    }
    Ok(())
}

/// One batch's decoded outputs and ground truth, on their way to [`Post`].
struct Decoded {
    images: Outputs,
    /// Optional DFL precisions, entropies and anchor strides (downloaded on [`Post`] when on a GPU).
    aux: DflAuxOutputs,
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
    /// The multi-label candidate pairs of the one-to-many branch, found on the device
    /// (`folo_gpu::detect::multi_label_candidates`, `[B, 1 + 3 · cap]`), the decoded boxes `[B, 4, A]`, and the
    /// whole decode `[B, 4 + nc, A]` for an image with more than `cap` pairs. Not read back yet.
    Pairs { pairs: Tensor, boxes: Tensor, decoded: Tensor, cap: usize, anchors: usize },
}

/// Whether validation's NMS can start from the device's candidate pairs: multi-label, and a threshold that is a
/// normal float (a GPU may flush subnormal scores to zero, where the host compares them exactly).
fn pairs_apply(nms: &NmsConfig) -> bool {
    nms.multi_label && nms.candidate_conf() >= f32::MIN_POSITIVE
}

/// Candidate pairs per image the next batch makes room for, with `a` anchors and `nc` classes: a quarter more than
/// the most seen so far (`most`), at least [`MIN_PAIRS`], whole KiB-multiples so the buffer sizes repeat. An image
/// with more falls back to downloading its whole decode, so this only sets the readback size, never the result.
///
/// The pairs are worth it only while they are clearly smaller than the decode (`(4 + nc) · A` floats per image):
/// a capacity never makes an image's region (`1 + 3 · cap` words) larger than half of it, and once the room wanted
/// is past that (`3 · want > (4 + nc) · A / 2`: many classes at a low `conf`) this is `None` and the batch downloads
/// its whole decode ([`Outputs::Device`]). So the pairs buffer is never larger than half the decode, which the
/// device has room for whenever it has room for the decode, and the detections are the same either way.
fn pair_capacity(most: usize, a: usize, nc: usize) -> Option<usize> {
    let half = (4 + nc).saturating_mul(a) / 2;
    let want = most + most / 4;
    // The most `cap` with `1 + 3 · cap <= half`.
    let limit = half.saturating_sub(1) / 3;
    if want.saturating_mul(3) > half || limit == 0 {
        return None;
    }
    Some(want.max(MIN_PAIRS).next_multiple_of(1024).min(limit))
}

/// Smallest [`pair_capacity`]: what COCO val2017 needs for most images with `folo-n-coco` at conf 0.001.
const MIN_PAIRS: usize = 16 * 1024;

/// The host half of [`validate`]: post-processing, metrics, samples and kept detections, batch by batch in order.
struct Post<'a> {
    ev: Evaluator,
    samples: Vec<ValSample>,
    kept: Vec<(usize, Vec<Detection>)>,
    /// Images seen so far.
    index: usize,
    /// The most candidate pairs an image had ([`pair_capacity`]).
    most: &'a AtomicUsize,
    /// Images post-processed from their device candidate pairs (not from a fallback decode).
    from_pairs: usize,
}

impl Post<'_> {
    fn add(&mut self, d: Decoded, ds: &dyn Dataset, cfg: &ValConfig) -> Result<()> {
        use rayon::prelude::*;
        let Decoded { images, aux, nc, mode, truths, meta } = d;
        let aux = aux.download()?;
        let post = |(i, (b, s)): (usize, (Vec<Xyxy>, Vec<f32>))| {
            crate::nms::postprocess_with_inputs(&b, &s, aux.inputs(i), nc, &cfg.nms, mode)
        };
        let dets: Vec<Vec<Detection>> = match images {
            Outputs::Host(images) => images.into_par_iter().enumerate().map(post).collect(),
            Outputs::Device(t, a) => {
                let all = t.to_vec::<f32>()?;
                if mode == HeadMode::O2m {
                    // NMS straight from the downloaded rows, without splitting them into boxes and scores.
                    all.par_chunks_exact(a * (4 + nc))
                        .enumerate()
                        .map(|(i, rows)| {
                            crate::nms::non_max_suppression_rows_with_inputs(
                                rows,
                                aux.inputs(i),
                                nc,
                                &cfg.nms,
                            )
                        })
                        .collect()
                } else {
                    let images = crate::model::HeadOutput::split_decoded(&all, a, nc);
                    images.into_par_iter().enumerate().map(post).collect()
                }
            }
            Outputs::Pairs { pairs, boxes, decoded, cap, anchors: a } => {
                self.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &cfg.nms, &aux)?
            }
        };
        for (i, dets) in dets.into_iter().enumerate() {
            let meta = &meta[i];
            let clipped = clipped_to_image(&dets, meta);
            if cfg.eval.record_diagnostics {
                let id = ds
                    .image_path(meta.index)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| format!("image_{}", meta.index));
                self.ev.add_image_with_id(id, &clipped, &truths[i]);
            } else {
                self.ev.add_image(&clipped, &truths[i]);
            }
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

impl Post<'_> {
    /// The detections of a batch from its device candidate pairs ([`Outputs::Pairs`]): exactly
    /// [`crate::nms::non_max_suppression_rows`] of the decode, from a fraction of the readback.
    #[allow(clippy::too_many_arguments)]
    fn pairs_nms(
        &mut self,
        pairs: &Tensor,
        boxes: &Tensor,
        decoded: &Tensor,
        cap: usize,
        a: usize,
        nc: usize,
        nms: &NmsConfig,
        aux: &DflAux,
    ) -> Result<Vec<Vec<Detection>>> {
        use rayon::prelude::*;
        let words = pairs.to_vec::<u32>()?;
        let boxes = boxes.to_vec::<f32>()?;
        let region = folo_gpu::detect::candidate_region_words(cap);
        let b = words.len() / region;
        let counts: Vec<usize> = (0..b).map(|i| words[i * region] as usize).collect();
        self.most.fetch_max(counts.iter().copied().max().unwrap_or(0), Ordering::Relaxed);
        self.from_pairs += counts.iter().filter(|&&n| n <= cap).count();
        // Images with more pairs than room: their whole decode, as the full path downloads it.
        let mut full: Vec<Option<Vec<f32>>> = vec![None; b];
        for (i, &n) in counts.iter().enumerate() {
            if n > a * nc {
                return Err(folo_nn::Error::Invalid(format!(
                    "candidates: {n} pairs for {a} anchors x {nc} classes"
                ))
                .into());
            }
            if n > cap {
                full[i] = Some(decoded.narrow(0, i, 1)?.transpose(1, 2)?.to_vec::<f32>()?);
            }
        }
        (0..b)
            .into_par_iter()
            .map(|i| {
                let dfl: DflInputs<'_> = aux.inputs(i);
                if let Some(rows) = &full[i] {
                    return Ok(crate::nms::non_max_suppression_rows_with_inputs(rows, dfl, nc, nms));
                }
                let n = counts[i];
                let recs = &words[i * region + 1..][..folo_gpu::detect::CANDIDATE_RECORD_WORDS * n];
                let mut list = Vec::with_capacity(n);
                for w in recs.as_chunks::<3>().0 {
                    if w[0] as usize >= a || w[1] as usize >= nc {
                        return Err(folo_nn::Error::Invalid(format!(
                            "candidates: pair ({}, {}) out of range",
                            w[0], w[1]
                        ))
                        .into());
                    }
                    list.push((w[0], w[1], f32::from_bits(w[2])));
                }
                let bx = &boxes[i * 4 * a..(i + 1) * 4 * a];
                let bbox = |j: u32| {
                    let j = j as usize;
                    Xyxy::new(bx[j], bx[a + j], bx[2 * a + j], bx[3 * a + j])
                };
                Ok(crate::nms::non_max_suppression_pairs_with_inputs(list, bbox, dfl, nms))
            })
            .collect()
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
mod tests;
