//! Decoding and the raw prediction path (docs/MODEL.md §6.4, §9.3): forward, decode (on the CPU with
//! [`crate::decode`], or as tensor ops on the outputs' device), then NMS on the host.

use folo_core::backend::DETECT_RECORD_WORDS;
use folo_core::{DType, DetectDecode, Device, Tensor, no_grad};
use folo_nn::{Error, Module, Result};
use rayon::prelude::*;

use super::{FoloModel, HeadMode, HeadOutput};
use crate::boxes::Xyxy;
use crate::decode::{decode_head, dist2bbox, sigmoid, softmax_expectation};
use crate::nms::{Candidate, Detection, NmsConfig, nms_candidates, postprocess, topk_candidates};

/// Records per image in the first download of [`HeadOutput::detections`]' device path: the counts and up to
/// this many candidates of every image come back in one readback; an image with more takes a second one.
pub const CANDIDATE_READBACK: usize = 1024;

/// Anchors per task of the CPU candidate extraction ([`HeadOutput::detections`] on the CPU): 8400 anchors at
/// 640 px make nine blocks, and a block's running class maximum (4 KiB) stays in L1.
const CPU_DECODE_BLOCK: usize = 1024;

/// One image's decoded output, in network-input pixels (no letterbox inverse).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePrediction {
    /// `[A]` decoded boxes, `xyxy`.
    pub boxes: Vec<Xyxy>,
    /// `[A, nc]` row-major class scores (post-sigmoid).
    pub scores: Vec<f32>,
    /// The post-processing output (NMS, or the top-k of an `o2o` head), sorted by descending score.
    pub detections: Vec<Detection>,
}

/// What [`FoloModel::predict_raw`] returns.
#[derive(Clone)]
pub struct RawPrediction {
    /// The raw head outputs.
    pub output: HeadOutput,
    /// Per image: decoded boxes and scores, and the detections after post-processing.
    pub images: Vec<ImagePrediction>,
}

/// `[B, C, A]` → host `[B, A, C]` (anchor-major, the layout [`crate::decode`] takes). `to_vec` walks the
/// transposed view in logical order, so this is one copy.
fn anchor_major(t: &Tensor) -> Result<Vec<f32>> {
    let t = if t.device() == Device::Cpu { t.clone() } else { t.to_device(Device::Cpu)? };
    Ok(t.to_dtype(DType::F32)?.transpose(1, 2)?.to_vec::<f32>()?)
}

impl HeadOutput {
    /// Decode every image (MODEL.md §6.4): `(boxes [A], scores [A, nc])` in input pixels. CPU outputs are
    /// decoded on the CPU; outputs on another device are decoded there ([`HeadOutput::decode_on_device`])
    /// and downloaded once.
    pub fn decode_images(&self) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        let _g = no_grad();
        if !self.box_raw.device().is_cpu() {
            return self.decode_on_device();
        }
        let (a, nc, rm) = (self.num_anchors(), self.num_classes, self.reg_max);
        let reg = anchor_major(&self.box_raw)?;
        let cls = anchor_major(&self.cls_raw)?;
        if reg.len() != self.batch() * a * 4 * rm || cls.len() != self.batch() * a * nc {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} / cls_raw {:?} do not match {a} anchors",
                self.box_raw.dims(),
                self.cls_raw.dims()
            )));
        }
        Ok(reg
            .chunks_exact(a * 4 * rm)
            .zip(cls.chunks_exact(a * nc))
            .map(|(r, c)| decode_head(r, c, &self.anchors, rm, nc))
            .collect())
    }

    /// [`HeadOutput::decode_images`] as device tensor ops: the DFL softmax expectation, `ltrb` → `xyxy`
    /// around the anchor points times the stride, and the class sigmoid, assembled anchor-major as
    /// `[B, A, 4 + nc]` on the device so that only the decoded values are downloaded (about 60% of the raw
    /// head) and the host does no transpose. Same formulas as the CPU path (max-subtracted softmax). `F16`
    /// outputs (half-precision inference) are decoded in `F32`.
    pub fn decode_on_device(&self) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        let all = self.decoded_on_device()?.to_vec::<f32>()?;
        Ok(Self::split_decoded(&all, self.num_anchors(), self.num_classes))
    }

    /// The device half of [`HeadOutput::decode_on_device`]: the decoded `[B, A, 4 + nc]` tensor (a transposed
    /// view) on the outputs' device, not read back yet. Readers that cannot block (the browser) download it
    /// asynchronously and hand the values to [`HeadOutput::split_decoded`].
    pub fn decoded_on_device(&self) -> Result<Tensor> {
        let _g = no_grad();
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] {
            return Err(Error::Invalid(format!(
                "HeadOutput: box_raw {:?} / cls_raw {:?} do not match {a} anchors",
                self.box_raw.dims(),
                self.cls_raw.dims()
            )));
        }
        let dev = self.box_raw.device();
        let (box_raw, cls_raw) = (self.box_raw.to_dtype(DType::F32)?, self.cls_raw.to_dtype(DType::F32)?);
        // One upload: anchor x, anchor y (grid units), stride, then the bin indices 0..reg_max.
        let mut consts = Vec::with_capacity(3 * a + rm);
        consts.extend(self.anchors.points.iter().map(|p| p[0]));
        consts.extend(self.anchors.points.iter().map(|p| p[1]));
        consts.extend_from_slice(&self.anchors.strides);
        consts.extend((0..rm).map(|i| i as f32));
        let consts = Tensor::from_vec(consts, [3 * a + rm])?.to_device(dev)?;
        let points = consts.narrow(0, 0, 2 * a)?.reshape([1, 2, a])?;
        let stride = consts.narrow(0, 2 * a, a)?.reshape([1, 1, a])?;
        let bins = consts.narrow(0, 3 * a, rm)?.reshape([1, 1, rm, 1])?;
        let dist = box_raw.reshape([b, 4, rm, a])?.softmax(2)?.mul(&bins)?.sum(2, false)?; // [B, 4, A]
        let xy1 = points.sub(&dist.narrow(1, 0, 2)?)?;
        let xy2 = points.add(&dist.narrow(1, 2, 2)?)?;
        let boxes = Tensor::cat(&[&xy1, &xy2], 1)?.mul(&stride)?;
        let scores = cls_raw.sigmoid()?;
        Ok(Tensor::cat(&[&boxes, &scores], 1)?.transpose(1, 2)?) // [B, A, 4 + nc]
    }

    /// Per image `(boxes [A], scores [A, nc])` from the host values of [`HeadOutput::decoded_on_device`]
    /// (`[B, A, 4 + nc]`, row-major).
    pub fn split_decoded(all: &[f32], a: usize, nc: usize) -> Vec<(Vec<Xyxy>, Vec<f32>)> {
        let row = 4 + nc;
        all.par_chunks_exact(a * row)
            .map(|img| {
                let boxes = img.chunks_exact(row).map(|r| Xyxy::new(r[0], r[1], r[2], r[3])).collect();
                let mut scores = Vec::with_capacity(a * nc);
                for r in img.chunks_exact(row) {
                    scores.extend_from_slice(&r[4..]);
                }
                (boxes, scores)
            })
            .collect()
    }

    /// The detections of every image (MODEL.md §6.4 decode, then §9.3 NMS with `nms`, or for one-to-one
    /// outputs, [`HeadOutput::mode`] `o2o`, §6.6's top-k with `nms.conf` and `nms.max_det` and no NMS), in
    /// network-input pixels, highest score first.
    ///
    /// On a device with the candidate kernel (the GPU) and without `multi_label`, the decode runs there in
    /// one kernel ([`Tensor::detect_candidates`], which reads `F16` outputs as they are and decodes in `F32`)
    /// and only the anchors whose best class scores `> conf` are
    /// downloaded: one readback of the counts plus up to [`CANDIDATE_READBACK`] records per image, a second
    /// one only for an image with more. On the CPU (single-label) the anchors are prefiltered by their largest
    /// logit and only the candidates are decoded, with the same functions as [`HeadOutput::decode_images`], so
    /// the result is bit for bit that of the full decode. Everything else decodes every anchor and runs
    /// [`crate::nms::non_max_suppression`] (or [`crate::nms::topk_detections`]). All paths give the same
    /// detections up to the device's float rounding.
    pub fn detections(&self, nms: &NmsConfig) -> Result<Vec<Vec<Detection>>> {
        let _g = no_grad();
        if !nms.multi_label {
            let fast = if self.box_raw.device().is_cpu() {
                self.detections_on_cpu(nms)?
            } else {
                self.detections_on_device(nms)?
            };
            if let Some(d) = fast {
                return Ok(d);
            }
        }
        let (nc, mode) = (self.num_classes, self.mode);
        Ok(self.decode_images()?.into_par_iter().map(|(b, s)| postprocess(&b, &s, nc, nms, mode)).collect())
    }

    /// The single-label post-process of one image's candidates (sorted by anchor) for this output's branch.
    fn finish_candidates(&self, cands: &[Candidate], nms: &NmsConfig) -> Vec<Detection> {
        match self.mode {
            HeadMode::O2m => nms_candidates(cands, nms),
            HeadMode::O2o => topk_candidates(cands, nms),
        }
    }

    /// The CPU half of [`HeadOutput::detections`]: straight from the channel-major outputs (borrowed, not
    /// copied), a running maximum of the class logits per anchor, then the full single-label extraction
    /// (sigmoid of every class, first maximum, `> conf`) and the DFL decode only for anchors whose largest logit
    /// could pass. The sigmoid is monotonic, so a skipped anchor has no score above `conf` (a NaN first score
    /// never passes either). Like the GPU candidate kernel, anchors are processed in independent blocks
    /// ([`CPU_DECODE_BLOCK`]) on the rayon pool, and the blocks' candidates are joined in anchor order, so the
    /// result does not depend on the split. `None` for shapes (or dtypes) the full path should report.
    fn detections_on_cpu(&self, nms: &NmsConfig) -> Result<Option<Vec<Vec<Detection>>>> {
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] || a == 0 || nc == 0 {
            return Ok(None);
        }
        if self.box_raw.dtype() != DType::F32 || self.cls_raw.dtype() != DType::F32 {
            return Ok(None);
        }
        let anchors = &self.anchors;
        // The candidates among anchors `j0..j1` of one image.
        let block = |cl: &[f32], rg: &[f32], j0: usize, j1: usize| -> Vec<Candidate> {
            let mut best = [0f32; CPU_DECODE_BLOCK];
            let best = &mut best[..j1 - j0];
            best.copy_from_slice(&cl[j0..j1]);
            for row in cl.chunks_exact(a).skip(1) {
                for (m, &v) in best.iter_mut().zip(&row[j0..j1]) {
                    *m = m.max(v);
                }
            }
            let mut cands = Vec::new();
            let mut bins = vec![0f32; rm];
            for (j, &m) in (j0..j1).zip(best.iter()) {
                // A margin of a few ulp, so even a libm `exp` that is not monotonic cannot skip a candidate;
                // the exact test follows.
                let may_pass = sigmoid(m) * (1.0 + 1e-6) + 1e-30 > nms.conf;
                if !may_pass {
                    continue;
                }
                // Exactly the extraction of `non_max_suppression`.
                let (mut bc, mut bs) = (0usize, sigmoid(cl[j]));
                for c in 1..nc {
                    let s = sigmoid(cl[c * a + j]);
                    if s > bs {
                        (bc, bs) = (c, s);
                    }
                }
                let passes = bs > nms.conf; // false for NaN
                if !passes {
                    continue;
                }
                let mut d = [0f32; 4];
                for (side, d) in d.iter_mut().enumerate() {
                    for (k, v) in bins.iter_mut().enumerate() {
                        *v = rg[(side * rm + k) * a + j];
                    }
                    *d = softmax_expectation(&bins);
                }
                let st = anchors.strides[j];
                let bbox = dist2bbox(d, anchors.points[j]).scale(st, st);
                cands.push(Candidate { anchor: j as u32, class: bc as u32, score: bs, bbox });
            }
            cands
        };
        let blocks = a.div_ceil(CPU_DECODE_BLOCK);
        let images = |reg: &[f32], cls: &[f32]| -> Vec<Vec<Detection>> {
            (0..b)
                .into_par_iter()
                .map(|i| {
                    let cl = &cls[i * nc * a..(i + 1) * nc * a];
                    let rg = &reg[i * 4 * rm * a..(i + 1) * 4 * rm * a];
                    let parts: Vec<Vec<Candidate>> = (0..blocks)
                        .into_par_iter()
                        .map(|t| block(cl, rg, t * CPU_DECODE_BLOCK, ((t + 1) * CPU_DECODE_BLOCK).min(a)))
                        .collect();
                    self.finish_candidates(&parts.concat(), nms)
                })
                .collect()
        };
        // Views of one storage (e.g. channel slices of one `[B, 4·reg_max + nc, A]` tensor) are not borrowed
        // twice: a second read lock on a storage while a writer queues for it could deadlock. Copy one instead.
        let dets = if self.box_raw.same_storage(&self.cls_raw) {
            let cls = self.cls_raw.to_vec::<f32>()?;
            self.box_raw.with_f32_slice(|reg| images(reg, &cls))?
        } else {
            self.box_raw.with_f32_slice(|reg| self.cls_raw.with_f32_slice(|cls| images(reg, cls)))??
        };
        Ok(Some(dets))
    }

    /// The anchors as `[3, A]` on `dev`: point x, point y (grid units), stride.
    fn anchors_tensor(&self, dev: Device) -> Result<Tensor> {
        let a = self.num_anchors();
        let mut v = Vec::with_capacity(3 * a);
        v.extend(self.anchors.points.iter().map(|p| p[0]));
        v.extend(self.anchors.points.iter().map(|p| p[1]));
        v.extend_from_slice(&self.anchors.strides);
        Ok(Tensor::from_vec(v, [3, a])?.to_device(dev)?)
    }

    /// The device half of [`HeadOutput::detections`]; `None` when the device has no candidate kernel.
    fn detections_on_device(&self, nms: &NmsConfig) -> Result<Option<Vec<Vec<Detection>>>> {
        let (b, a, nc, rm) = (self.batch(), self.num_anchors(), self.num_classes, self.reg_max);
        if self.box_raw.dims() != [b, 4 * rm, a] || self.cls_raw.dims() != [b, nc, a] || a == 0 || nc == 0 {
            return Ok(None);
        }
        let dev = self.box_raw.device();
        let p = DetectDecode { reg_max: rm, num_classes: nc, conf: nms.conf };
        let anchors = self.anchors_tensor(dev)?;
        let Some(cands) = Tensor::detect_candidates(&self.box_raw, &self.cls_raw, &anchors, &p)? else {
            return Ok(None);
        };
        const W: usize = DETECT_RECORD_WORDS;
        let cap = CANDIDATE_READBACK.min(a);
        let head = cands.narrow(1, 0, 1 + W * cap)?.to_vec::<u32>()?;
        let mut records: Vec<Vec<u32>> = Vec::with_capacity(b);
        for (i, row) in head.chunks_exact(1 + W * cap).enumerate() {
            let n = row[0] as usize;
            if n > a {
                return Err(Error::Invalid(format!("detect_candidates: {n} candidates for {a} anchors")));
            }
            records.push(if n <= cap {
                row[1..1 + W * n].to_vec()
            } else {
                cands.narrow(0, i, 1)?.narrow(1, 1, W * n)?.to_vec::<u32>()?
            });
        }
        Ok(Some(
            records
                .into_par_iter()
                .map(|r| {
                    let mut c: Vec<Candidate> = r
                        .as_chunks::<W>()
                        .0
                        .iter()
                        .map(|w| Candidate {
                            anchor: w[0],
                            class: w[1],
                            score: f32::from_bits(w[2]),
                            bbox: Xyxy::new(
                                f32::from_bits(w[3]),
                                f32::from_bits(w[4]),
                                f32::from_bits(w[5]),
                                f32::from_bits(w[6]),
                            ),
                        })
                        .collect();
                    // The kernel appends in arrival order; NMS tie-breaks by anchor.
                    c.sort_unstable_by_key(|c| c.anchor);
                    self.finish_candidates(&c, nms)
                })
                .collect(),
        ))
    }

    /// The decoded output `[B, A, 4 + nc]` (MODEL.md §6.4): per anchor `x1, y1, x2, y2` in input pixels, then
    /// the `nc` class scores. A CPU tensor without gradient.
    pub fn decode(&self) -> Result<Tensor> {
        let (a, nc) = (self.num_anchors(), self.num_classes);
        let images = self.decode_images()?;
        let mut out = Vec::with_capacity(images.len() * a * (4 + nc));
        for (boxes, scores) in &images {
            for (b, s) in boxes.iter().zip(scores.chunks_exact(nc.max(1))) {
                out.extend_from_slice(&[b.x1, b.y1, b.x2, b.y2]);
                out.extend_from_slice(s);
            }
        }
        Ok(Tensor::from_vec(out, [images.len(), a, 4 + nc])?)
    }
}

impl FoloModel {
    /// The predictor's model step: an eval-mode forward under `no_grad` (of the [`FoloModel::head_mode`]
    /// branch), then decode and class-aware NMS (`nms`, MODEL.md §9.3; for an `o2o` head the top-k of §6.6) on
    /// the host, images in parallel; outputs on a GPU are decoded there and downloaded
    /// once ([`HeadOutput::decode_on_device`]). Boxes are in network-input
    /// pixels; mapping back through the letterbox is the caller's job. Fails in training mode, where the
    /// forward would update the BatchNorm statistics.
    pub fn predict_raw(&self, x: &Tensor, nms: &NmsConfig) -> Result<RawPrediction> {
        self.predict_raw_as(x, nms, self.head_mode())
    }

    /// [`FoloModel::predict_raw`] with the `mode` branch and its post-processing ([`FoloModel::forward_head`]).
    pub fn predict_raw_as(&self, x: &Tensor, nms: &NmsConfig, mode: HeadMode) -> Result<RawPrediction> {
        if self.is_training() {
            return Err(Error::Invalid(
                "FoloModel::predict_raw: the model is in training mode; call eval() or fuse() first".into(),
            ));
        }
        let _g = no_grad();
        let output = self.forward_head(x, mode)?;
        let (nc, mode) = (output.num_classes, output.mode);
        let images = output
            .decode_images()?
            .into_par_iter()
            .map(|(boxes, scores)| {
                let detections = postprocess(&boxes, &scores, nc, nms, mode);
                ImagePrediction { boxes, scores, detections }
            })
            .collect();
        Ok(RawPrediction { output, images })
    }

    /// The predictor's model step without the per-anchor outputs: an eval-mode forward under `no_grad`,
    /// then [`HeadOutput::detections`] (decoded on the outputs' device where it can be, so a GPU prediction
    /// downloads only the candidates). Boxes are in network-input pixels. Fails in training mode.
    pub fn predict_detections(&self, x: &Tensor, nms: &NmsConfig) -> Result<Vec<Vec<Detection>>> {
        if self.is_training() {
            return Err(Error::Invalid(
                "FoloModel::predict_detections: the model is in training mode; call eval() or fuse() first"
                    .into(),
            ));
        }
        let _g = no_grad();
        self.forward(x)?.detections(nms)
    }
}
