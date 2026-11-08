//! Decoding and the raw prediction path (docs/MODEL.md §6.4, §9.3): forward, decode (on the CPU with
//! [`crate::decode`], or as tensor ops on the outputs' device), then NMS on the host.

use folo_core::{Device, Tensor, no_grad};
use folo_nn::{Error, Module, Result};
use rayon::prelude::*;

use super::{FoloModel, HeadOutput};
use crate::boxes::Xyxy;
use crate::decode::decode_head;
use crate::nms::{Detection, NmsConfig, non_max_suppression};

/// One image's decoded output, in network-input pixels (no letterbox inverse).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePrediction {
    /// `[A]` decoded boxes, `xyxy`.
    pub boxes: Vec<Xyxy>,
    /// `[A, nc]` row-major class scores (post-sigmoid).
    pub scores: Vec<f32>,
    /// The NMS output, sorted by descending score.
    pub detections: Vec<Detection>,
}

/// What [`FoloModel::predict_raw`] returns.
#[derive(Clone)]
pub struct RawPrediction {
    /// The raw head outputs.
    pub output: HeadOutput,
    /// Per image: decoded boxes and scores, and the detections after NMS.
    pub images: Vec<ImagePrediction>,
}

/// `[B, C, A]` → host `[B, A, C]` (anchor-major, the layout [`crate::decode`] takes). `to_vec` walks the
/// transposed view in logical order, so this is one copy.
fn anchor_major(t: &Tensor) -> Result<Vec<f32>> {
    let t = if t.device() == Device::Cpu { t.clone() } else { t.to_device(Device::Cpu)? };
    Ok(t.transpose(1, 2)?.to_vec::<f32>()?)
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
    /// head) and the host does no transpose. Same formulas as the CPU path (max-subtracted softmax).
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
        let dist = self.box_raw.reshape([b, 4, rm, a])?.softmax(2)?.mul(&bins)?.sum(2, false)?; // [B, 4, A]
        let xy1 = points.sub(&dist.narrow(1, 0, 2)?)?;
        let xy2 = points.add(&dist.narrow(1, 2, 2)?)?;
        let boxes = Tensor::cat(&[&xy1, &xy2], 1)?.mul(&stride)?;
        let scores = self.cls_raw.sigmoid()?;
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
    /// The predictor's model step: an eval-mode forward under `no_grad`, then decode and class-aware NMS
    /// (`nms`, MODEL.md §9.3) on the host, images in parallel; outputs on a GPU are decoded there and downloaded
    /// once ([`HeadOutput::decode_on_device`]). Boxes are in network-input
    /// pixels; mapping back through the letterbox is the caller's job. Fails in training mode, where the
    /// forward would update the BatchNorm statistics.
    pub fn predict_raw(&self, x: &Tensor, nms: &NmsConfig) -> Result<RawPrediction> {
        if self.is_training() {
            return Err(Error::Invalid(
                "FoloModel::predict_raw: the model is in training mode; call eval() or fuse() first".into(),
            ));
        }
        let _g = no_grad();
        let output = self.forward(x)?;
        let nc = output.num_classes;
        let images = output
            .decode_images()?
            .into_par_iter()
            .map(|(boxes, scores)| {
                let detections = non_max_suppression(&boxes, &scores, nc, nms);
                ImagePrediction { boxes, scores, detections }
            })
            .collect();
        Ok(RawPrediction { output, images })
    }
}
