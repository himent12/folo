//! The predictor: `.folo` weights → detections in original-image pixels (docs/MODEL.md §9).
//!
//! [`Predictor::open`] loads the EMA weights when the file has them (a checkpoint or an export), fuses Conv+BN
//! (§9.1) and keeps the model in eval mode. Each call letterboxes (§9.2: `rect` padding to multiples of 32 for
//! single images, a square `imgsz` canvas for batches), runs the forward pass, decodes and NMSes (§9.3) and maps
//! the boxes back through the letterbox, clipped to the image. [`Timing`] splits the wall time into
//! preprocess / inference / postprocess like docs/API.md's `/api/predict`.
//!
//! On a GPU the whole pipeline stays on the device: the decoded `u8` images are uploaded as they are and
//! letterboxed there (`Tensor::letterbox_rgb8`, bit-exact with the CPU letterbox), and the head is decoded
//! there into above-threshold candidates, so only those come back ([`crate::model::HeadOutput::detections`]).
//! GPU work is queued asynchronously, so on a GPU the time of the forward pass shows up under postprocess,
//! where the first download waits for it.

use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_core::{Device, LetterboxImage, Tensor, no_grad};
use folo_data::{Image, Letterbox, LetterboxOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::model::{FoloModel, HeadOutput};
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
    /// A timing from its three parts (`total_ms` is their sum).
    pub fn new(pre: f64, inf: f64, post: f64) -> Self {
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
        let batch = self.prepare(imgs, rect)?;
        let pre = ms(t0);

        let t1 = Instant::now();
        let out = self.forward(&batch)?;
        let inf = ms(t1);

        // docs/API.md counts the decode as post-processing.
        let t2 = Instant::now();
        let dets = self.finish_detections(&batch, out.detections(&self.cfg.nms())?);
        let post = ms(t2);
        let n = imgs.len() as f64;
        Ok(batch.predictions(dets, Timing::new(pre / n, inf / n, post / n)))
    }

    /// The first step of a prediction, for callers that drive the steps themselves (the browser, which
    /// must await the GPU readback between [`Predictor::forward`] and [`Predictor::finish`]): letterbox the
    /// images (`rect` padding for one image, else the square canvas) into the input tensor, on the model's
    /// device (on a GPU from the raw `u8` images, [`letterbox_on_device`]). [`Predictor::predict`] is
    /// `prepare` → `forward` → [`HeadOutput::detections`] → [`Predictor::finish_detections`]; [`Predictor::finish`]
    /// takes decoded outputs instead.
    pub fn prepare(&self, imgs: &[Image], rect: bool) -> Result<PreparedBatch> {
        if imgs.is_empty() {
            return Err(Error::Config("cannot predict an empty batch".into()));
        }
        let rect = rect && imgs.len() == 1;
        let s = self.cfg.imgsz;
        let opts = LetterboxOptions { stride: rect.then_some(32), ..LetterboxOptions::default() };
        for img in imgs {
            if img.width == 0 || img.height == 0 {
                return Err(Error::Config("cannot predict an empty image".into()));
            }
        }
        let letterboxes: Vec<Letterbox> =
            imgs.iter().map(|i| Letterbox::compute(i.width, i.height, s, s, &opts)).collect();
        // `rect` is only used for one image, so every canvas of a call has the same size.
        let device = self.model.device();
        let on_device = if device.is_cpu() { None } else { letterbox_on_device(imgs, &letterboxes, device)? };
        let input = match on_device {
            Some(x) => x,
            None => {
                let x = letterbox_on_host(imgs, &letterboxes)?;
                if device.is_cpu() { x } else { x.to_device(device)? }
            }
        };
        let sizes = imgs.iter().map(|i| [i.width, i.height]).collect();
        Ok(PreparedBatch { input, letterboxes, sizes })
    }

    /// The forward pass of a [`PreparedBatch`] under `no_grad`. On a GPU the result is queued work: nothing
    /// waits until its outputs are read back.
    pub fn forward(&self, batch: &PreparedBatch) -> Result<HeadOutput> {
        let _g = no_grad();
        Ok(self.model.forward(&batch.input)?)
    }

    /// The last step: class-aware NMS (with the current [`PredictConfig`]) on each image's decoded
    /// `(boxes, scores)` (from [`HeadOutput::decode_images`] or [`HeadOutput::split_decoded`]), then the boxes
    /// mapped back through the letterbox into original-image pixels. `nc` is the head's class count.
    pub fn finish(
        &self,
        batch: &PreparedBatch,
        decoded: Vec<(Vec<Xyxy>, Vec<f32>)>,
        nc: usize,
    ) -> Vec<Vec<Detection>> {
        let nms = self.cfg.nms();
        let dets = decoded.into_par_iter().map(|(b, sc)| non_max_suppression(&b, &sc, nc, &nms)).collect();
        self.finish_detections(batch, dets)
    }

    /// The last step for detections in network-input pixels (from [`HeadOutput::detections`] with this predictor's
    /// NMS settings): each image's boxes mapped back through its letterbox.
    pub fn finish_detections(&self, batch: &PreparedBatch, dets: Vec<Vec<Detection>>) -> Vec<Vec<Detection>> {
        dets.into_par_iter()
            .zip(batch.letterboxes.par_iter())
            .map(|(d, lb)| {
                d.into_iter()
                    .map(|d| {
                        let [x1, y1, x2, y2] = lb.unmap_xyxy([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]);
                        Detection { bbox: Xyxy::new(x1, y1, x2, y2), ..d }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Letterboxed images and their input tensor ([`Predictor::prepare`]).
pub struct PreparedBatch {
    /// `[B, 3, H, W]` in `0..1`, on the model's device.
    pub input: Tensor,
    letterboxes: Vec<Letterbox>,
    sizes: Vec<[u32; 2]>,
}

impl PreparedBatch {
    /// Images in the batch.
    pub fn len(&self) -> usize {
        self.sizes.len()
    }

    /// Whether the batch is empty (never, for a batch from [`Predictor::prepare`]).
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }

    /// One [`Prediction`] per image from [`Predictor::finish`]'s detections and a per-image timing.
    pub fn predictions(&self, dets: Vec<Vec<Detection>>, timing: Timing) -> Vec<Prediction> {
        self.sizes
            .iter()
            .zip(dets)
            .map(|(&[width, height], detections)| Prediction { width, height, detections, timing })
            .collect()
    }
}

/// `u8` pixel → network input. A product (not `/ 255`), so the device letterbox can match it bit for bit.
const INV_255: f32 = 1.0 / 255.0;

/// The `[B, 3, H, W]` network input (`0..1`) of `imgs` letterboxed with `boxes`, built on the host. Every
/// geometry must have the same output size (the predictor uses `rect` only for single images).
pub fn letterbox_on_host(imgs: &[Image], boxes: &[Letterbox]) -> Result<Tensor> {
    let [w, h] = check_letterbox(imgs, boxes)?;
    let plane = (w * h) as usize;
    let mut input = vec![0f32; imgs.len() * 3 * plane];
    input.par_chunks_mut(3 * plane).zip(imgs.par_iter().zip(boxes)).for_each(|(dst, (img, lb))| {
        let canvas = lb.apply(img, folo_data::PAD_VALUE);
        to_chw(&canvas.data, dst, plane);
    });
    Ok(Tensor::from_vec(input, [imgs.len(), 3, h as usize, w as usize])?)
}

/// [`letterbox_on_host`] on `device` from the raw `u8` images, bit-exact with it (`None` when the device has
/// no letterbox kernel): the images go up as they are, with folo-data's resize taps, and the canvas is
/// built there.
pub fn letterbox_on_device(imgs: &[Image], boxes: &[Letterbox], device: Device) -> Result<Option<Tensor>> {
    let [w, h] = check_letterbox(imgs, boxes)?;
    let mut taps: Vec<u32> = Vec::new();
    let mut geoms = Vec::with_capacity(imgs.len());
    for (img, lb) in imgs.iter().zip(boxes) {
        let [rw, rh] = lb.resized;
        let plan = folo_data::resize::resize_plan(img.width, img.height, rw, rh);
        geoms.push(LetterboxImage {
            src: [img.width as usize, img.height as usize],
            resized: [rw as usize, rh as usize],
            pad: [lb.pad[0] as usize, lb.pad[1] as usize],
            prefilter: [plan.prefilter[0] as usize, plan.prefilter[1] as usize],
            taps: taps.len(),
        });
        for t in [&plan.x0, &plan.x1, &plan.wx, &plan.y0, &plan.y1, &plan.wy] {
            taps.extend_from_slice(t);
        }
    }
    let n = taps.len();
    let taps = Tensor::from_vec(taps, [n])?.to_device(device)?;
    // Host copies in parallel; the uploads (one per image) are queued in order.
    let host: Vec<Tensor> = imgs
        .par_iter()
        .map(|im| Tensor::from_vec(im.data.clone(), [im.height as usize, im.width as usize, 3]))
        .collect::<folo_core::Result<_>>()?;
    let dev_imgs = host.iter().map(|t| t.to_device(device)).collect::<folo_core::Result<Vec<_>>>()?;
    let refs: Vec<&Tensor> = dev_imgs.iter().collect();
    let out = [h as usize, w as usize];
    Ok(Tensor::letterbox_rgb8(&refs, &taps, &geoms, out, folo_data::PAD_VALUE, INV_255)?)
}

/// The common canvas `[width, height]` of a letterbox batch: one geometry per image, computed for its size,
/// all with the same output size.
fn check_letterbox(imgs: &[Image], boxes: &[Letterbox]) -> Result<[u32; 2]> {
    let first = boxes.first().ok_or_else(|| Error::Config("letterbox: no images".into()))?;
    if imgs.len() != boxes.len() {
        return Err(Error::Config(format!(
            "letterbox: {} images but {} geometries",
            imgs.len(),
            boxes.len()
        )));
    }
    for (img, lb) in imgs.iter().zip(boxes) {
        if lb.orig != [img.width, img.height] || lb.out != first.out {
            return Err(Error::Config(format!(
                "letterbox: geometry {lb:?} does not fit a {}x{} image on a {:?} canvas",
                img.width, img.height, first.out
            )));
        }
    }
    Ok(first.out)
}

/// Interleaved RGB `u8` → planar `f32` in `0..1`.
fn to_chw(rgb: &[u8], dst: &mut [f32], plane: usize) {
    let (r, rest) = dst.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
        r[i] = px[0] as f32 * INV_255;
        g[i] = px[1] as f32 * INV_255;
        b[i] = px[2] as f32 * INV_255;
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
