//! Validation: a model on a dataset split, scored with [`Evaluator`] (docs/MODEL.md §8.7, docs/METRICS.md).
//!
//! Images are letterboxed to a square `imgsz` by folo-data's validation transform; ground-truth boxes go through
//! the same resize and pad, so predictions are compared in network-input pixels with no inverse map. The first
//! [`ValConfig::samples`] images of the split are also returned as [`ValSample`]s (truth and predictions mapped back
//! to the original image), the fixed sample set of docs/API.md's `/samples`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use folo_core::{DType, Device, Tensor};
use folo_data::{Batch, DataLoader, Dataset, LoaderConfig};
use folo_nn::Module;
use serde::Serialize;

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::fusion::BoxFusion;
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

/// Evaluate `model` (in eval mode, fused or not; `F32`, or `F16` after [`Module::to_dtype`] on a GPU that computes
/// in it) on `ds`. Every class id of the dataset must be below the model's class count.
pub fn validate(model: &FoloModel, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<ValResult> {
    validate_counted(model, ds, cfg).map(|(r, _)| r)
}

/// [`validate`], and how many images were post-processed from device candidate pairs ([`Outputs::Pairs`] within
/// their capacity), so the tests can tell that path ran.
fn validate_counted(model: &FoloModel, ds: Arc<dyn Dataset>, cfg: &ValConfig) -> Result<(ValResult, usize)> {
    let t0 = Instant::now();
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
            let output = {
                let _g = folo_core::no_grad();
                model.forward_head(&x, head)?
            };
            drop(x);
            let (nc, mode) = (output.num_classes, output.mode);
            let a = output.num_anchors();
            let cap = pair_capacity(most.load(Ordering::Relaxed), a, nc);
            // DFL fusion's side precisions, read before the decode is handed over (`None` unless asked for).
            let prec = if cfg.nms.fusion == BoxFusion::Dfl && mode == HeadMode::O2m {
                Some(output.side_precisions()?)
            } else {
                None
            };
            let images = if output.box_raw.device().is_cpu() {
                Outputs::Host(output.decode_images()?)
            } else if let Some(cap) = cap
                && mode == HeadMode::O2m
                && pairs_apply(&cfg.nms)
                && folo_gpu::detect::supported(device)
            {
                // `[B, 4 + nc, A]`: the transpose of the `[B, A, 4 + nc]` view is the dense decode itself.
                let decoded = output.decoded_on_device()?.transpose(1, 2)?;
                let pairs = folo_gpu::detect::multi_label_candidates(&decoded, cfg.nms.conf, cap)?;
                let boxes = decoded.narrow(1, 0, 4)?;
                Outputs::Pairs { pairs, boxes, decoded, cap, anchors: a }
            } else {
                // Encoded here, downloaded by the other thread (HeadOutput::decode_images' device path).
                Outputs::Device(output.decoded_on_device()?, a)
            };
            drop(output);
            if tx.send(Decoded { images, prec, nc, mode, truths, meta: batch.meta }).is_err() {
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

/// One batch's decoded outputs and ground truth, on their way to [`Post`].
struct Decoded {
    images: Outputs,
    /// Per image, every anchor's side precisions for [`BoxFusion::Dfl`].
    prec: Option<Vec<Vec<[f32; 4]>>>,
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
    nms.multi_label && nms.conf >= f32::MIN_POSITIVE
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
        let Decoded { images, prec, nc, mode, truths, meta } = d;
        let prec_of = |i: usize| prec.as_ref().map(|p| p[i].as_slice());
        let post = |(i, (b, s)): (usize, (Vec<Xyxy>, Vec<f32>))| {
            crate::nms::postprocess_with(&b, &s, prec_of(i), nc, &cfg.nms, mode)
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
                            crate::nms::non_max_suppression_rows_with(rows, prec_of(i), nc, &cfg.nms)
                        })
                        .collect()
                } else {
                    let images = crate::model::HeadOutput::split_decoded(&all, a, nc);
                    images.into_par_iter().enumerate().map(post).collect()
                }
            }
            Outputs::Pairs { pairs, boxes, decoded, cap, anchors: a } => {
                self.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &cfg.nms, prec.as_deref())?
            }
        };
        for (i, dets) in dets.into_iter().enumerate() {
            let meta = &meta[i];
            self.ev.add_image(&clipped_to_image(&dets, meta), &truths[i]);
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
        prec: Option<&[Vec<[f32; 4]>]>,
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
                let p = prec.map(|p| p[i].as_slice());
                if let Some(rows) = &full[i] {
                    return Ok(crate::nms::non_max_suppression_rows_with(rows, p, nc, nms));
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
                Ok(crate::nms::non_max_suppression_pairs_with(list, bbox, p, nms))
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
mod tests {
    use super::*;
    use folo_data::Split;
    use folo_data::synthetic::{SyntheticConfig, SyntheticDataset};
    use folo_nn::io::ModelConfig;

    /// The evaluator's boxes are clipped to the letterboxed image, not to the input: a 640×480 image at 640 has 80
    /// pixels of padding above and below, and a box reaching into it ends at the image's edge.
    #[test]
    fn evaluator_boxes_are_clipped_to_the_image() {
        let meta =
            folo_data::ImageMeta { index: 0, orig_size: [320, 240], ratio: [2.0, 2.0], pad: [0.0, 80.0] };
        let d = |x1, y1, x2, y2| Detection { bbox: Xyxy::new(x1, y1, x2, y2), score: 0.5, class: 1 };
        let got = clipped_to_image(
            &[d(-5.0, 70.0, 100.0, 200.0), d(10.0, 90.0, 650.0, 600.0), d(1.0, 81.0, 2.0, 82.0)],
            &meta,
        );
        assert_eq!(got, [d(0.0, 80.0, 100.0, 200.0), d(10.0, 90.0, 640.0, 560.0), d(1.0, 81.0, 2.0, 82.0)]);
        // As the COCO path clips: the clipped box maps back to the clipped original-image box.
        for (g, o) in got.iter().zip([d(-5.0, 70.0, 100.0, 200.0), d(10.0, 90.0, 650.0, 600.0)]) {
            let back = |b: &Xyxy| meta.unmap_xyxy([b.x1, b.y1, b.x2, b.y2]);
            assert_eq!(back(&g.bbox), back(&o.bbox));
        }
    }

    /// Validation's metrics score the clipped box: a prediction that matches its truth inside the image but reaches
    /// 20 pixels into the padding is a perfect detection (unclipped, its IoU would be 0.86 and mAP50-95 0.8).
    #[test]
    fn validation_scores_the_clipped_box() {
        let (_, ds) = untrained(Device::Cpu);
        let most = AtomicUsize::new(0);
        let mut post = Post {
            ev: Evaluator::new(1),
            samples: Vec::new(),
            kept: Vec::new(),
            index: 0,
            most: &most,
            from_pairs: 0,
        };
        let meta =
            folo_data::ImageMeta { index: 0, orig_size: [320, 240], ratio: [2.0, 2.0], pad: [0.0, 80.0] };
        let truth = Truth { bbox: Xyxy::new(0.0, 80.0, 100.0, 200.0), class: 0 };
        let images = Outputs::Host(vec![(vec![Xyxy::new(0.0, 60.0, 100.0, 200.0)], vec![0.9])]);
        let d = Decoded {
            images,
            prec: None,
            nc: 1,
            mode: HeadMode::O2m,
            truths: vec![vec![truth]],
            meta: vec![meta],
        };
        let cfg = ValConfig { samples: 0, ..ValConfig::new(640, 1) };
        post.add(d, &*ds, &cfg).unwrap();
        assert_eq!(post.ev.compute().map50_95, 1.0);
    }

    /// The pipelined [`validate`] (post-processing on a second thread) gives exactly what the serial loop gives:
    /// predict, then score each image in order. With conf 0 an untrained model keeps a box at every anchor, so NMS,
    /// the metrics, the samples and the kept detections all have work to do.
    #[test]
    fn validate_matches_the_serial_loop() {
        let (model, ds) = untrained(Device::Cpu);
        check_against_serial(&model, &ds, 0.0, BoxFusion::None);
        // DFL fusion: the precisions read from the logits on the host, as `predict_raw`'s full decode reads them.
        check_against_serial(&model, &ds, 0.0, BoxFusion::Dfl);
    }

    /// The same on a GPU, where the other thread also downloads the decoded outputs: at conf 0 the whole decode
    /// ([`Outputs::Device`]); at validation's conf 0.001, which lies between the untrained class priors of the
    /// stride-8 and the stride-16/32 anchors, the device candidate pairs ([`Outputs::Pairs`]), in f32 and, on a
    /// GPU that computes in it, in f16. Both paths with DFL fusion too (the side moments computed on the device).
    #[test]
    fn validate_matches_the_serial_loop_on_gpu() {
        let Some(device) = gpu() else { return };
        let (mut model, ds) = untrained(device);
        let conf = NmsConfig::val().conf;
        for fusion in [BoxFusion::None, BoxFusion::Dfl] {
            assert_eq!(
                check_against_serial(&model, &ds, 0.0, fusion),
                0,
                "conf 0 downloads the whole decode"
            );
            assert_eq!(
                check_against_serial(&model, &ds, conf, fusion),
                10,
                "every image from its device pairs"
            );
        }
        if folo_core::backend::computes_f16(device) {
            model.to_dtype(DType::F16).unwrap();
            assert_eq!(
                check_against_serial(&model, &ds, conf, BoxFusion::None),
                10,
                "f16: every image from its pairs"
            );
            assert_eq!(check_against_serial(&model, &ds, conf, BoxFusion::Dfl), 10, "f16, DFL fusion");
        } else {
            eprintln!("note: skipping the f16 case, the GPU does not compute in f16");
        }
    }

    /// A threshold outside 0..=1 (NaN included) is a config error before any image is read, the fusion cluster IoU
    /// too: NaN would put every same-class candidate in one cluster, a value above 1 would switch fusion off.
    #[test]
    fn validate_rejects_thresholds_outside_the_unit_range() {
        let (model, ds) = untrained(Device::Cpu);
        // (conf, iou, fusion_iou)
        for (conf, iou, fusion_iou) in
            [(f32::NAN, 0.7, 0.75), (0.001, 1.5, 0.75), (0.001, 0.7, f32::NAN), (0.001, 0.7, 1.5)]
        {
            let mut vc = ValConfig::new(64, 4);
            vc.nms.fusion = BoxFusion::Dfl;
            (vc.nms.conf, vc.nms.iou, vc.nms.fusion_iou) = (conf, iou, fusion_iou);
            let got = validate(&model, ds.clone(), &vc);
            assert!(matches!(got, Err(Error::Config(_))), "conf {conf} iou {iou} fusion_iou {fusion_iou}");
        }
    }

    /// The pairs path's capacity: none once the pairs would not be clearly smaller than the decode, else at least
    /// [`MIN_PAIRS`], KiB-rounded, and never a region larger than half the decode.
    #[test]
    fn pair_capacity_is_bounded_by_half_the_decode() {
        // COCO at 640: 8400 anchors, 80 classes, half the decode is 352_800 words.
        assert_eq!(pair_capacity(0, 8400, 80), Some(MIN_PAIRS));
        assert_eq!(pair_capacity(20_000, 8400, 80), Some(25 * 1024));
        assert_eq!(pair_capacity(94_000, 8400, 80), Some(117_599)); // 1 + 3 · 117_599 = 352_798: the limit
        assert_eq!(pair_capacity(94_100, 8400, 80), None);
        // 400 classes at a low conf: a capacity for every pair would be three times the decode.
        assert_eq!(pair_capacity(8400 * 400, 8400, 400), None);
        // Tiny heads: the limit, and none when not even one record fits.
        assert_eq!(pair_capacity(0, 84, 2), Some(83));
        assert_eq!(pair_capacity(0, 1, 1), None);
        for (most, a, nc) in [(0, 84, 2), (60, 84, 2), (1000, 8400, 400), (300_000, 8400, 80)] {
            if let Some(cap) = pair_capacity(most, a, nc) {
                assert!(cap >= 1 && 2 * (1 + 3 * cap) <= (4 + nc) * a, "{most} {a} {nc}: {cap}");
            }
        }
    }

    /// The device candidate pairs give exactly the full-row NMS: random decodes with scores at and around `conf`,
    /// NaNs and many overlapping boxes, with room for every pair and with too little room (the full fallback).
    #[test]
    fn device_pairs_match_full_rows_nms() {
        let Some(device) = gpu() else { return };
        let (b, nc, a) = (3usize, 5usize, 700usize);
        let conf = 0.25f32;
        let mut rng = folo_data::Rng::new(11);
        let mut host = vec![0f32; b * (4 + nc) * a];
        for i in 0..b {
            for j in 0..a {
                let at = |r: usize| (i * (4 + nc) + r) * a + j;
                let (x, y) = (rng.uniform(0.0, 60.0), rng.uniform(0.0, 60.0));
                let (w, h) = (rng.uniform(1.0, 30.0), rng.uniform(1.0, 30.0));
                host[at(0)] = x;
                host[at(1)] = y;
                host[at(2)] = x + w;
                host[at(3)] = y + h;
                for c in 0..nc {
                    host[at(4 + c)] = match (j + c) % 23 {
                        0 => conf,
                        1 => f32::NAN,
                        2 => f32::from_bits(conf.to_bits() + 1),
                        _ => rng.uniform(0.0, 1.0),
                    };
                }
            }
        }
        // Image 1 has few pairs, so a capacity between its count and the others' overflows images 0 and 2 only.
        for j in 0..a {
            for c in 0..nc {
                let v = &mut host[(4 + nc + 4 + c) * a + j];
                if j % 9 != 0 {
                    *v = 0.0;
                }
            }
        }
        let decoded = Tensor::from_vec(host.clone(), [b, 4 + nc, a]).unwrap().to_device(device).unwrap();
        let nms = NmsConfig { conf, max_det: 40, max_candidates: 900, ..NmsConfig::val() };
        let want: Vec<Vec<Detection>> = (0..b)
            .map(|i| {
                let mut rows = Vec::with_capacity(a * (4 + nc));
                for j in 0..a {
                    rows.extend((0..4 + nc).map(|r| host[(i * (4 + nc) + r) * a + j]));
                }
                crate::nms::non_max_suppression_rows(&rows, nc, &nms)
            })
            .collect();
        assert!(want.iter().all(|d| !d.is_empty()));
        let counts: Vec<usize> = (0..b)
            .map(|i| host[(i * (4 + nc) + 4) * a..][..nc * a].iter().filter(|&&s| s > conf).count())
            .collect();
        assert!(counts[1] < 1024 && counts[0] > 1024 && counts[2] > 1024, "{counts:?}");
        let most = AtomicUsize::new(0);
        let boxes = decoded.narrow(1, 0, 4).unwrap();
        // DFL fusion with arbitrary side precisions: the pairs path still gives the full-row result.
        let dfl = NmsConfig { fusion: BoxFusion::Dfl, fusion_iou: 0.5, ..nms.clone() };
        let prec: Vec<Vec<[f32; 4]>> = (0..b)
            .map(|_| (0..a).map(|_| std::array::from_fn(|_| rng.uniform(0.01, 4.0))).collect())
            .collect();
        let want_dfl: Vec<Vec<Detection>> = (0..b)
            .map(|i| {
                let mut rows = Vec::with_capacity(a * (4 + nc));
                for j in 0..a {
                    rows.extend((0..4 + nc).map(|r| host[(i * (4 + nc) + r) * a + j]));
                }
                crate::nms::non_max_suppression_rows_with(&rows, Some(&prec[i]), nc, &dfl)
            })
            .collect();
        assert!(want_dfl.iter().zip(&want).any(|(f, w)| f.iter().zip(w).any(|(f, w)| f.bbox != w.bbox)));
        // Room for every pair, for image 1's only, for none: how many images are served from their pairs.
        for (cap, served) in [(a * nc, 3), (1024, 1), (200, 0)] {
            let mut post = Post {
                ev: Evaluator::new(nc),
                samples: Vec::new(),
                kept: Vec::new(),
                index: 0,
                most: &most,
                from_pairs: 0,
            };
            let pairs = folo_gpu::detect::multi_label_candidates(&decoded, conf, cap).unwrap();
            let words = pairs.to_vec::<u32>().unwrap();
            let region = folo_gpu::detect::candidate_region_words(cap);
            assert_eq!((0..b).map(|i| words[i * region] as usize).collect::<Vec<_>>(), counts, "cap {cap}");
            let got = post.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &nms, None).unwrap();
            assert_eq!(got, want, "cap {cap}");
            assert_eq!(post.from_pairs, served, "cap {cap}");
            let got = post.pairs_nms(&pairs, &boxes, &decoded, cap, a, nc, &dfl, Some(&prec)).unwrap();
            assert_eq!(got, want_dfl, "cap {cap}, DFL fusion");
        }
        assert_eq!(most.load(Ordering::Relaxed), *counts.iter().max().unwrap());
    }

    /// The registered GPU, or `None` (with a note) when there is none or only a software adapter.
    fn gpu() -> Option<Device> {
        let ctx = match folo_gpu::GpuContext::shared() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("note: skipping GPU test, no adapter: {e}");
                return None;
            }
        };
        if ctx.summary().device_type == "Cpu" && std::env::var("FOLO_GPU_TEST_SOFTWARE").is_err() {
            eprintln!("note: skipping GPU test on software adapter {}", ctx.summary().name);
            return None;
        }
        Some(folo_gpu::init().unwrap())
    }

    /// An untrained folo-n in eval mode on `device` and 10 synthetic images of its 2 classes.
    fn untrained(device: Device) -> (FoloModel, Arc<dyn Dataset>) {
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
        (model, ds)
    }

    /// [`validate`] of `model` on `ds` at `conf` (the sample conf too) with `fusion` against the serial loop; returns
    /// how many of the images it post-processed from device candidate pairs.
    fn check_against_serial(model: &FoloModel, ds: &Arc<dyn Dataset>, conf: f32, fusion: BoxFusion) -> usize {
        let (device, dtype) = (model.device(), model.dtype());
        let mut vc = ValConfig { samples: 5, keep_detections: true, workers: 2, ..ValConfig::new(64, 4) };
        vc.nms.fusion = fusion;
        // Untrained scores sit near the class prior: conf 0 keeps everything, so NMS has every anchor to sort out.
        (vc.nms.conf, vc.sample_conf) = (conf, conf);
        let (got, from_pairs) = validate_counted(model, ds.clone(), &vc).unwrap();

        let lc = LoaderConfig { workers: 0, u8_images: !device.is_cpu(), ..LoaderConfig::val(4, 64) };
        let loader = DataLoader::new(ds.clone(), lc).unwrap();
        let mut ev = Evaluator::new(2);
        let (mut samples, mut kept, mut index) = (Vec::new(), Vec::new(), 0);
        for batch in loader.epoch(0) {
            let mut batch = batch.unwrap();
            let truths = batch_truths(&batch);
            let x = batch_tensor(&mut batch, device, dtype).unwrap();
            let pred = model.predict_raw(&x, &vc.nms).unwrap();
            for (i, img) in pred.images.iter().enumerate() {
                let meta = &batch.meta[i];
                ev.add_image(&clipped_to_image(&img.detections, meta), &truths[i]);
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
        let boxes = kept.iter().map(|k: &(usize, Vec<Detection>)| k.1.len()).sum::<usize>();
        assert!(boxes > if conf == 0.0 { 100 } else { 20 }, "too few boxes to test: {boxes}");
        assert_eq!(got.report, ev.compute());
        assert_eq!(got.samples, samples);
        assert_eq!(got.detections, kept);
        from_pairs
    }
}
