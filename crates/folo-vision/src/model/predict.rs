//! Decoding and the raw prediction path (docs/MODEL.md §6.4, §9.3): forward, then decode + NMS on CPU copies.

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

/// `[B, C, A]` → host `[B, A, C]` (anchor-major, the layout [`crate::decode`] takes).
fn anchor_major(t: &Tensor) -> Result<Vec<f32>> {
    let t = if t.device() == Device::Cpu { t.clone() } else { t.to_device(Device::Cpu)? };
    Ok(t.transpose(1, 2)?.contiguous()?.to_vec::<f32>()?)
}

impl HeadOutput {
    /// Decode every image (MODEL.md §6.4) on the CPU: `(boxes [A], scores [A, nc])` in input pixels.
    pub fn decode_images(&self) -> Result<Vec<(Vec<Xyxy>, Vec<f32>)>> {
        let _g = no_grad();
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
    /// (`nms`, MODEL.md §9.3) of CPU copies of the outputs, images in parallel. Boxes are in network-input
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
