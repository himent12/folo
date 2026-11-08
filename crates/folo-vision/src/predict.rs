//! The predictor: `.folo` weights → detections in original-image pixels (docs/MODEL.md §9).
//!
//! [`Predictor::open`] loads the EMA weights when the file has them (a checkpoint or an export), fuses Conv+BN
//! (§9.1) and keeps the model in eval mode. Each call letterboxes (§9.2: `rect` padding to multiples of 32 for
//! single images, a square `imgsz` canvas for batches), runs the forward pass, decodes and NMSes (§9.3) and maps
//! the boxes back through the letterbox, clipped to the image. [`Timing`] splits the wall time into
//! preprocess / inference / postprocess like docs/API.md's `/api/predict`.

use std::path::Path;
use std::time::Instant;

use folo_core::{Device, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::model::FoloModel;
use crate::nms::{Detection, NmsConfig, non_max_suppression};

/// Prediction settings; the defaults are MODEL.md §9.3's `predict` column at `imgsz = 640`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictConfig {
    /// Target size `S` of the letterbox (a multiple of 32).
    pub imgsz: u32,
    /// Score threshold.
    pub conf: f32,
    /// NMS IoU threshold.
    pub iou: f32,
    /// Detections kept per image.
    pub max_det: usize,
    /// Single images: pad only to the next multiple of 32 (`rect`, the default) instead of `S × S`.
    pub rect: bool,
}

impl Default for PredictConfig {
    fn default() -> Self {
        PredictConfig { imgsz: 640, conf: 0.25, iou: 0.7, max_det: 300, rect: true }
    }
}

impl PredictConfig {
    fn validate(&self) -> Result<()> {
        if self.imgsz == 0 || !self.imgsz.is_multiple_of(32) || self.imgsz > 8192 {
            return Err(Error::Config(format!(
                "imgsz must be a multiple of 32 in 32..=8192, got {}",
                self.imgsz
            )));
        }
        if !(0.0..=1.0).contains(&self.conf) || !(0.0..=1.0).contains(&self.iou) {
            return Err(Error::Config(format!(
                "conf and iou must be in 0..=1, got {} and {}",
                self.conf, self.iou
            )));
        }
        if self.max_det == 0 {
            return Err(Error::Config("max_det must be at least 1".into()));
        }
        Ok(())
    }

    fn nms(&self) -> NmsConfig {
        NmsConfig { conf: self.conf, iou: self.iou, max_det: self.max_det, ..NmsConfig::default() }
    }
}

/// Wall time of a prediction in milliseconds (docs/API.md `timing`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Timing {
    /// Letterbox and conversion to the input tensor.
    pub preprocess_ms: f64,
    /// Forward pass.
    pub inference_ms: f64,
    /// Decode, NMS and rescale.
    pub postprocess_ms: f64,
    /// Sum of the above.
    pub total_ms: f64,
}

impl Timing {
    fn new(pre: f64, inf: f64, post: f64) -> Self {
        Timing { preprocess_ms: pre, inference_ms: inf, postprocess_ms: post, total_ms: pre + inf + post }
    }

    /// docs/API.md's `Timing`.
    pub fn to_api(self) -> folo_api::Timing {
        folo_api::Timing {
            preprocess_ms: self.preprocess_ms,
            inference_ms: self.inference_ms,
            postprocess_ms: self.postprocess_ms,
            total_ms: self.total_ms,
        }
    }
}

/// The detections of one image.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Prediction {
    /// Original width.
    pub width: u32,
    /// Original height.
    pub height: u32,
    /// Detections in original-image pixels, highest score first.
    pub detections: Vec<Detection>,
    /// Wall time (for a batch: the batch's time divided by its size).
    pub timing: Timing,
}

impl Prediction {
    /// The detections as docs/API.md `ScoredBox`es, naming each class from `classes`.
    pub fn scored_boxes(&self, classes: &[String]) -> Vec<folo_api::ScoredBox> {
        self.detections
            .iter()
            .map(|d| folo_api::ScoredBox {
                class: d.class,
                name: classes.get(d.class as usize).cloned().unwrap_or_else(|| format!("class{}", d.class)),
                score: d.score,
                x1: d.bbox.x1,
                y1: d.bbox.y1,
                x2: d.bbox.x2,
                y2: d.bbox.y2,
            })
            .collect()
    }
}

/// A fused, eval-mode detector ready for inference. See the module docs.
pub struct Predictor {
    model: FoloModel,
    cfg: PredictConfig,
}

impl Predictor {
    /// Load `.folo` weights (EMA when present; weights, exports and checkpoints all work) and fuse them.
    pub fn open(path: impl AsRef<Path>, cfg: PredictConfig) -> Result<Self> {
        Self::new(FoloModel::open(path)?, cfg)
    }

    /// Wrap a model: it is fused (a no-op if it already is) and stays in eval mode.
    pub fn new(mut model: FoloModel, cfg: PredictConfig) -> Result<Self> {
        cfg.validate()?;
        model.fuse()?;
        Ok(Predictor { model, cfg })
    }

    /// The (fused) model.
    pub fn model(&self) -> &FoloModel {
        &self.model
    }

    /// Class names.
    pub fn classes(&self) -> &[String] {
        self.model.classes()
    }

    /// The settings.
    pub fn config(&self) -> &PredictConfig {
        &self.cfg
    }

    /// Change the settings.
    pub fn set_config(&mut self, cfg: PredictConfig) -> Result<()> {
        cfg.validate()?;
        self.cfg = cfg;
        Ok(())
    }

    /// The device the model runs on.
    pub fn device(&self) -> Device {
        self.model.device()
    }

    /// Decode an image file and predict it.
    pub fn predict_file(&self, path: impl AsRef<Path>) -> Result<Prediction> {
        let t = Instant::now();
        let img = folo_data::read_image(path.as_ref())?;
        let decode_ms = ms(t);
        let mut p = self.predict(&img)?;
        p.timing =
            Timing::new(p.timing.preprocess_ms + decode_ms, p.timing.inference_ms, p.timing.postprocess_ms);
        Ok(p)
    }

    /// One image, letterboxed `rect` (or square when [`PredictConfig::rect`] is off).
    pub fn predict(&self, img: &Image) -> Result<Prediction> {
        Ok(self.run(std::slice::from_ref(img), self.cfg.rect)?.pop().expect("one image in, one out"))
    }

    /// A batch in one forward pass: every image is letterboxed to the square `imgsz × imgsz` canvas. Each
    /// prediction's timing is the batch's divided by its size.
    pub fn predict_batch(&self, imgs: &[Image]) -> Result<Vec<Prediction>> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }
        self.run(imgs, false)
    }

    fn run(&self, imgs: &[Image], rect: bool) -> Result<Vec<Prediction>> {
        let t0 = Instant::now();
        let s = self.cfg.imgsz;
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        for img in imgs {
            if img.width == 0 || img.height == 0 {
                return Err(Error::Config("cannot predict an empty image".into()));
            }
        }
        let boxes: Vec<Letterbox> =
            imgs.iter().map(|i| Letterbox::compute(i.width, i.height, s, s, &opts)).collect();
        // `rect` is only used for one image, so every canvas of a call has the same size.
        let [w, h] = boxes[0].out;
        let plane = (w * h) as usize;
        let mut input = vec![0f32; imgs.len() * 3 * plane];
        input.par_chunks_mut(3 * plane).zip(imgs.par_iter().zip(&boxes)).for_each(|(dst, (img, lb))| {
            let canvas = lb.apply(img, folo_data::PAD_VALUE);
            to_chw(&canvas.data, dst, plane);
        });
        let x = Tensor::from_vec(input, [imgs.len(), 3, h as usize, w as usize])?;
        let device = self.model.device();
        let x = if device.is_cpu() { x } else { x.to_device(device)? };
        let pre = ms(t0);

        let t1 = Instant::now();
        let _g = no_grad();
        let out = self.model.forward(&x)?;
        drop(x);
        let inf = ms(t1);

        // docs/API.md counts the decode as post-processing.
        let t2 = Instant::now();
        let decoded = out.decode_images()?;
        let nc = out.num_classes;
        let nms = self.cfg.nms();
        let dets: Vec<Vec<Detection>> = decoded
            .into_par_iter()
            .zip(boxes.par_iter())
            .map(|((b, sc), lb)| {
                non_max_suppression(&b, &sc, nc, &nms)
                    .into_iter()
                    .map(|d| {
                        let [x1, y1, x2, y2] = lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]);
                        Detection { bbox: Xyxy::new(x1, y1, x2, y2), ..d }
                    })
                    .collect()
            })
            .collect();
        let post = ms(t2);
        let n = imgs.len() as f64;
        let timing = Timing::new(pre / n, inf / n, post / n);
        Ok(imgs
            .iter()
            .zip(dets)
            .map(|(img, detections)| Prediction { width: img.width, height: img.height, detections, timing })
            .collect())
    }
}

/// Interleaved RGB `u8` → planar `f32` in `0..1`.
fn to_chw(rgb: &[u8], dst: &mut [f32], plane: usize) {
    let (r, rest) = dst.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        r[i] = px[0] as f32 / 255.0;
        g[i] = px[1] as f32 / 255.0;
        b[i] = px[2] as f32 / 255.0;
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
