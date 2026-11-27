//! Test-Time Augmentation (TTA) over [`crate::Predictor`].
//!
//! Multi-scale and horizontal-flip ensembling improves recall and box localization without retraining: each
//! input image is evaluated across the stride-32 inference sizes resolved from [`TtaConfig::scales`] (and
//! optionally its horizontally flipped mirror via [`TtaConfig::hflip`]), flipped detections are mapped back
//! into original-image coordinates (`x1' = width - x2`, `x2' = width - x1`), and all view detections are
//! merged deterministically through FOLO's shared class-aware NMS and box-fusion pipeline
//! ([`BoxFusion::Vote`], [`BoxFusion::Sebf`], or [`BoxFusion::None`]).
//!
//! Within each scale, original and horizontally flipped views share identical pixel dimensions and therefore
//! the same `rect` letterbox canvas, so they execute together in one batched forward pass — inheriting every
//! CPU, int8 (`u8_stem`), GPU, `f16`, `o2o`, and candidate-first fast path of [`Predictor`].

use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_core::Tensor;
use folo_data::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::boxes::Xyxy;
use crate::error::{Error, Result};
use crate::fusion::BoxFusion;
use crate::nms::Detection;
use crate::predict::{
    Prediction, Predictor, Timing, add_scaled_timing, ms, one_canvas, upload_images_on_device,
};
use crate::tile::{MergeOptions, TileConfig, TileMergeMetric, check_image, merge_detections};

/// Default multi-scale factors for Test-Time Augmentation (`[0.83, 1.0, 1.17]`).
pub const DEFAULT_TTA_SCALES: &[f32] = &[0.83, 1.0, 1.17];

/// Minimum stride-aligned inference size accepted by [`Predictor`].
const MIN_IMGSZ: u32 = 32;
/// Maximum stride-aligned inference size accepted by [`Predictor`].
const MAX_IMGSZ: u32 = 8192;

/// Settings for Test-Time Augmentation ([`Predictor::predict_tta`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtaConfig {
    /// Also evaluate a horizontally flipped copy of every image at each scale (doubles the view count).
    pub hflip: bool,
    /// Multi-scale multipliers applied to [`crate::PredictConfig::imgsz`]. Each scale must be finite and
    /// strictly positive, and its rounded multiple-of-32 inference size must lie in `32..=8192`. Distinct scales
    /// that round to the same stride-32 size are deduplicated in first-seen order so no view is evaluated twice.
    pub scales: Vec<f32>,
    /// IoU threshold in `[0.0, 1.0]` used to cluster and deduplicate detections across augmented views.
    /// When `None`, inherits [`crate::PredictConfig::iou`].
    pub merge_iou: Option<f32>,
    /// Box refinement across agreeing detections from different views ([`BoxFusion::Vote`] by default).
    /// Cross-view detections carry no DFL logits, so [`BoxFusion::Dfl`] refines as [`BoxFusion::Vote`].
    pub merge_fusion: BoxFusion,
    /// Merge across different classes (`false`: class-aware merging, the default).
    pub class_agnostic: bool,
    /// Maximum merged detections kept per image. When `None`, inherits [`crate::PredictConfig::max_det`].
    pub max_det: Option<usize>,
}

impl Default for TtaConfig {
    fn default() -> Self {
        TtaConfig {
            hflip: true,
            scales: DEFAULT_TTA_SCALES.to_vec(),
            merge_iou: None,
            merge_fusion: BoxFusion::Vote,
            class_agnostic: false,
            max_det: None,
        }
    }
}

impl TtaConfig {
    /// Fast 2-view horizontal-flip TTA at native `imgsz` (`scales = [1.0]`, `hflip = true`).
    pub fn flip_only() -> Self {
        TtaConfig { hflip: true, scales: vec![1.0], ..Self::default() }
    }

    /// Multi-scale + horizontal-flip TTA with custom `scales`.
    pub fn multi_scale(scales: Vec<f32>) -> Self {
        TtaConfig { hflip: true, scales, ..Self::default() }
    }

    /// Validate all fields independent of a specific `imgsz`.
    pub fn validate(&self) -> Result<()> {
        if self.scales.is_empty() {
            return Err(Error::Config("tta scales must not be empty".into()));
        }
        for &s in &self.scales {
            if !s.is_finite() || s <= 0.0 {
                return Err(Error::Config(format!("tta scales must be finite and positive, got {s}")));
            }
        }
        if let Some(iou) = self.merge_iou
            && (!iou.is_finite() || !(0.0..=1.0).contains(&iou))
        {
            return Err(Error::Config(format!("tta merge_iou must be finite and in [0, 1], got {iou}")));
        }
        if let Some(0) = self.max_det {
            return Err(Error::Config("tta max_det must be at least 1".into()));
        }
        Ok(())
    }

    /// Resolve `self.scales` against `base_imgsz` into unique stride-32 inference sizes in `32..=8192`,
    /// preserving first-seen order. Fails if `self` is invalid, `base_imgsz` is not a valid multiple of 32 in
    /// `32..=8192`, or any scaled size falls outside `32..=8192`.
    pub fn resolve_imgszs(&self, base_imgsz: u32) -> Result<Vec<u32>> {
        self.validate()?;
        if !(MIN_IMGSZ..=MAX_IMGSZ).contains(&base_imgsz) || !base_imgsz.is_multiple_of(32) {
            return Err(Error::Config(format!(
                "base imgsz must be a multiple of 32 in {MIN_IMGSZ}..={MAX_IMGSZ}, got {base_imgsz}"
            )));
        }
        let mut sizes = Vec::with_capacity(self.scales.len());
        for &scale in &self.scales {
            let scaled = (base_imgsz as f64) * (scale as f64);
            let rounded = (scaled / 32.0).round() * 32.0;
            if !rounded.is_finite() || rounded < MIN_IMGSZ as f64 || rounded > MAX_IMGSZ as f64 {
                return Err(Error::Config(format!(
                    "tta scale {scale} on imgsz {base_imgsz} resolves outside {MIN_IMGSZ}..={MAX_IMGSZ} ({rounded})"
                )));
            }
            let sz = rounded as u32;
            debug_assert!((MIN_IMGSZ..=MAX_IMGSZ).contains(&sz) && sz.is_multiple_of(32));
            if !sizes.contains(&sz) {
                sizes.push(sz);
            }
        }
        Ok(sizes)
    }
}

/// Optional inference-time extensions for [`Predictor::predict_with_options`],
/// [`Predictor::predict_batch_with_options`], and [`Predictor::predict_file_with_options`].
///
/// `tta` and `tile` are mutually exclusive: enabling both in one [`PredictOptions`] returns
/// [`Error::Config`] so callers choose an explicit strategy without compounding double-merge distortion.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PredictOptions {
    /// Test-Time Augmentation (multi-scale and/or horizontal-flip ensembling).
    pub tta: Option<TtaConfig>,
    /// SAHI tiled inference for small objects in large images.
    pub tile: Option<TileConfig>,
}

impl PredictOptions {
    /// Options with Test-Time Augmentation enabled.
    pub fn with_tta(tta: TtaConfig) -> Self {
        PredictOptions { tta: Some(tta), tile: None }
    }

    /// Options with SAHI tiled inference enabled.
    pub fn with_tile(tile: TileConfig) -> Self {
        PredictOptions { tta: None, tile: Some(tile) }
    }

    /// Validate all configured options against `base_imgsz`.
    pub fn validate(&self, base_imgsz: u32) -> Result<()> {
        if self.tta.is_some() && self.tile.is_some() {
            return Err(Error::Config(
                "PredictOptions cannot combine `tta` and `tile`; enable either TTA or tiled inference".into(),
            ));
        }
        if let Some(tta) = &self.tta {
            tta.resolve_imgszs(base_imgsz)?;
        }
        if let Some(tile) = &self.tile {
            tile.validate()?;
        }
        Ok(())
    }
}

/// Map a bounding box from a horizontally flipped `img_w`-wide image back to original pixel coordinates:
/// `x1' = width - x2`, `x2' = width - x1`, clamped cleanly to `[0, img_w]`.
///
/// Non-finite coordinates are preserved as non-finite so downstream validation/merging drops them rather than
/// clamping them silently to image edges.
#[inline]
pub fn unflip_xyxy_h(bbox: Xyxy, img_w: u32) -> Xyxy {
    let fw = img_w as f32;
    let x1 = if bbox.x2.is_finite() { (fw - bbox.x2).clamp(0.0, fw) } else { f32::NAN };
    let x2 = if bbox.x1.is_finite() { (fw - bbox.x1).clamp(0.0, fw) } else { f32::NAN };
    Xyxy::new(x1, bbox.y1, x2, bbox.y2)
}

/// Merge detections gathered across augmented TTA views in original-image pixel coordinates.
///
/// Detections are clamped to `[0, img_w] × [0, img_h]`, sorted deterministically by descending score with
/// coordinate tie-breakers (so results never depend on view order), clustered greedily by IoU using
/// `cfg.merge_iou.unwrap_or(default_iou)`, refined via [`TtaConfig::merge_fusion`], and capped to
/// `cfg.max_det.unwrap_or(default_max_det)`.
pub fn merge_tta_detections(
    dets: Vec<Detection>,
    img_w: u32,
    img_h: u32,
    cfg: &TtaConfig,
    default_iou: f32,
    default_max_det: usize,
) -> Result<Vec<Detection>> {
    cfg.validate()?;
    let threshold = cfg.merge_iou.unwrap_or(default_iou);
    let max_det = cfg.max_det.unwrap_or(default_max_det);
    merge_detections(
        dets,
        img_w,
        img_h,
        MergeOptions {
            metric: TileMergeMetric::Iou,
            threshold,
            fusion: cfg.merge_fusion,
            class_agnostic: cfg.class_agnostic,
            max_det,
        },
    )
}

/// Execute `views`, recursively splitting in half on GPU out-of-memory / buffer-limit errors so batch doubling
/// never fails on tight device limits while preserving the exact canvas chosen for the scale.
pub(crate) fn run_views_with_oom_fallback<T>(
    views: &[&Image],
    run_chunk: &mut impl FnMut(&[&Image]) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    match run_chunk(views) {
        Ok(out) => Ok(out),
        Err(e) if views.len() > 1 && e.is_gpu_out_of_memory() => {
            let mid = views.len() / 2;
            let mut left = run_views_with_oom_fallback(&views[..mid], run_chunk)?;
            let right = run_views_with_oom_fallback(&views[mid..], run_chunk)?;
            left.extend(right);
            Ok(left)
        }
        Err(e) => Err(e),
    }
}

/// Like [`run_views_with_oom_fallback`] for views already uploaded to a GPU (`dev_imgs` parallel to `views`),
/// splitting both slices in half on GPU OOM / buffer-limit errors without re-uploading pixels.
fn run_uploaded_views_with_oom_fallback<T>(
    dev_imgs: &[Tensor],
    views: &[&Image],
    run_chunk: &mut impl FnMut(&[Tensor], &[&Image]) -> Result<Vec<T>>,
) -> Result<Vec<T>> {
    match run_chunk(dev_imgs, views) {
        Ok(out) => Ok(out),
        Err(e) if views.len() > 1 && e.is_gpu_out_of_memory() => {
            let mid = views.len() / 2;
            let mut left = run_uploaded_views_with_oom_fallback(&dev_imgs[..mid], &views[..mid], run_chunk)?;
            let right = run_uploaded_views_with_oom_fallback(&dev_imgs[mid..], &views[mid..], run_chunk)?;
            left.extend(right);
            Ok(left)
        }
        Err(e) => Err(e),
    }
}

/// Execute one scale's views, using pre-uploaded GPU tensors `uploaded` when present. If the resident-upload
/// path still hits GPU OOM even after chunking down to single views, drops `uploaded` to reclaim VRAM, calls
/// `release_memory()`, and retries via per-chunk host-to-device uploads (`run_host`), keeping `uploaded` cleared
/// for all remaining scales.
fn run_scale_with_resident_fallback<T, U>(
    uploaded: &mut Option<U>,
    run_uploaded: impl FnOnce(&U) -> Result<T>,
    mut release_memory: impl FnMut(),
    run_host: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if let Some(dev) = uploaded.as_ref() {
        match run_uploaded(dev) {
            Ok(out) => return Ok(out),
            Err(e) if e.is_gpu_out_of_memory() => {
                *uploaded = None;
                release_memory();
            }
            Err(e) => return Err(e),
        }
    }
    run_host()
}

impl Predictor {
    fn run_tta_views(&self, views: &[&Image], rect: bool, imgsz: u32) -> Result<Vec<Prediction>> {
        run_views_with_oom_fallback(views, &mut |chunk| self.run_with_imgsz(chunk, rect, imgsz))
    }

    fn run_uploaded_tta_views(
        &self,
        dev_imgs: &[Tensor],
        views: &[&Image],
        rect: bool,
        imgsz: u32,
    ) -> Result<Vec<Prediction>> {
        run_uploaded_views_with_oom_fallback(dev_imgs, views, &mut |dev_chunk, img_chunk| {
            self.run_uploaded_with_imgsz(dev_chunk, img_chunk, rect, imgsz)
        })
    }

    /// Upload `views` once to a GPU when multi-scale TTA will evaluate more than one inference size, returning
    /// `(None, elapsed_ms)` on CPU, single-scale TTA, or GPU OOM (reclaiming cached GPU memory and preserving
    /// the elapsed upload time so callers fall back cleanly to chunked per-scale uploads without losing timing).
    fn upload_tta_views_once(
        &self,
        views: &[&Image],
        num_scales: usize,
    ) -> Result<(Option<Vec<Tensor>>, f64)> {
        if self.device().is_cpu() || num_scales <= 1 {
            return Ok((None, 0.0));
        }
        let t_up = Instant::now();
        match upload_images_on_device(views, self.device()) {
            Ok(dev_imgs) => Ok((Some(dev_imgs), ms(t_up))),
            Err(e) if e.is_gpu_out_of_memory() => {
                if let Some(ctx) = folo_gpu::context_for(self.device()) {
                    let _ = ctx.release_memory();
                }
                Ok((None, ms(t_up)))
            }
            Err(e) => Err(e),
        }
    }

    /// Predict one image with Test-Time Augmentation ([`TtaConfig`]): runs each resolved stride-32 scale (and
    /// its horizontally flipped mirror when [`TtaConfig::hflip`] is enabled, batched together per scale without
    /// cloning the original RGB buffer), un-flips mirrored boxes back to original-image pixels, and merges all
    /// view detections via [`merge_tta_detections`].
    ///
    /// On a GPU with multiple scales, source views are uploaded once and letterboxed directly on device at every
    /// scale rather than repeating PCIe transfers per scale.
    ///
    /// When TTA resolves to a single unflipped view (`!tta_cfg.hflip && resolved_imgszs.len() == 1`), this
    /// degrades cleanly to one pass at that size (bit-identical to [`Predictor::predict`] when the size equals
    /// [`crate::PredictConfig::imgsz`], with `max_det` applied) and skips cross-view merging.
    pub fn predict_tta(&self, img: &Image, tta_cfg: &TtaConfig) -> Result<Prediction> {
        let imgszs = tta_cfg.resolve_imgszs(self.config().imgsz)?;
        check_image(img)?;
        let max_det = tta_cfg.max_det.unwrap_or(self.config().max_det);

        if !tta_cfg.hflip && imgszs.len() == 1 {
            let mut p = self
                .run_with_imgsz(&[img], self.config().rect, imgszs[0])?
                .pop()
                .expect("one image in, one out");
            p.detections.truncate(max_det);
            return Ok(p);
        }

        let mut pre_ms = 0.0f64;
        let mut inf_ms = 0.0f64;
        let mut post_ms = 0.0f64;

        let flipped = if tta_cfg.hflip {
            let t_flip = Instant::now();
            let fl = img.oriented(2);
            pre_ms += ms(t_flip);
            Some(fl)
        } else {
            None
        };
        let pair: [&Image; 2];
        let single: [&Image; 1];
        let views: &[&Image] = match &flipped {
            Some(fl) => {
                pair = [img, fl];
                &pair
            }
            None => {
                single = [img];
                &single
            }
        };

        let (mut uploaded, up_ms) = self.upload_tta_views_once(views, imgszs.len())?;
        pre_ms += up_ms;

        let mut all_dets: Vec<Detection> = Vec::new();
        for &imgsz in &imgszs {
            let t_scale = Instant::now();
            let mut preds = run_scale_with_resident_fallback(
                &mut uploaded,
                |dev| self.run_uploaded_tta_views(dev, views, self.config().rect, imgsz),
                || {
                    if let Some(ctx) = folo_gpu::context_for(self.device()) {
                        let _ = ctx.release_memory();
                    }
                },
                || self.run_tta_views(views, self.config().rect, imgsz),
            )?;
            let scale_wall = ms(t_scale);
            let (mut b_pre, mut b_inf, mut b_post) = (0.0f64, 0.0f64, 0.0f64);
            for p in &preds {
                b_pre += p.timing.preprocess_ms;
                b_inf += p.timing.inference_ms;
                b_post += p.timing.postprocess_ms;
            }
            add_scaled_timing(&mut pre_ms, &mut inf_ms, &mut post_ms, b_pre, b_inf, b_post, scale_wall);

            let t_collect = Instant::now();
            if tta_cfg.hflip {
                let flip_pred = preds.pop().expect("two views in, two out");
                let orig_pred = preds.pop().expect("two views in, two out");
                all_dets.extend(orig_pred.detections);
                all_dets.extend(
                    flip_pred
                        .detections
                        .into_iter()
                        .map(|d| Detection { bbox: unflip_xyxy_h(d.bbox, img.width), ..d }),
                );
            } else {
                let orig_pred = preds.pop().expect("one view in, one out");
                all_dets.extend(orig_pred.detections);
            }
            post_ms += ms(t_collect);
        }
        drop(uploaded);

        let t_merge = Instant::now();
        let detections = merge_tta_detections(
            all_dets,
            img.width,
            img.height,
            tta_cfg,
            self.config().iou,
            self.config().max_det,
        )?;
        post_ms += ms(t_merge);

        Ok(Prediction {
            width: img.width,
            height: img.height,
            detections,
            timing: Timing::new(pre_ms, inf_ms, post_ms),
        })
    }

    /// Predict a batch of images with Test-Time Augmentation ([`TtaConfig`]), in input order.
    ///
    /// Original (and optional flipped) views are batched per resolved scale without cloning the input RGB buffers
    /// so same-canvas batches share one `rect` letterbox pass (matching [`Predictor::predict_batch`]'s canvas
    /// rules), uploading source images once on GPU across multi-scale passes and falling back to smaller view
    /// chunks on GPU memory / buffer-limit errors.
    pub fn predict_batch_tta(&self, imgs: &[Image], tta_cfg: &TtaConfig) -> Result<Vec<Prediction>> {
        let imgszs = tta_cfg.resolve_imgszs(self.config().imgsz)?;
        if imgs.is_empty() {
            return Ok(Vec::new());
        }
        for img in imgs {
            check_image(img)?;
        }
        if imgs.len() == 1 {
            return self.predict_tta(&imgs[0], tta_cfg).map(|p| vec![p]);
        }
        let max_det = tta_cfg.max_det.unwrap_or(self.config().max_det);
        if !tta_cfg.hflip && imgszs.len() == 1 {
            let rect = self.config().rect && one_canvas(imgs, imgszs[0]);
            let refs: Vec<&Image> = imgs.iter().collect();
            let mut preds = self.run_with_imgsz(&refs, rect, imgszs[0])?;
            for p in &mut preds {
                p.detections.truncate(max_det);
            }
            return Ok(preds);
        }

        let n = imgs.len();
        let inv_n = 1.0 / n as f64;
        let mut pre_ms = 0.0f64;
        let mut inf_ms = 0.0f64;
        let mut post_ms = 0.0f64;

        let flipped: Vec<Image> = if tta_cfg.hflip {
            let t_flip = Instant::now();
            let fl: Vec<Image> = imgs.par_iter().map(|img| img.oriented(2)).collect();
            pre_ms += ms(t_flip) * inv_n;
            fl
        } else {
            Vec::new()
        };
        let mut views: Vec<&Image> = Vec::with_capacity(n + flipped.len());
        views.extend(imgs.iter());
        views.extend(flipped.iter());

        let (mut uploaded, up_ms) = self.upload_tta_views_once(&views, imgszs.len())?;
        pre_ms += up_ms * inv_n;

        let mut per_img_dets: Vec<Vec<Detection>> = vec![Vec::new(); n];
        for &imgsz in &imgszs {
            // `flipped[i]` has the exact same dimensions as `imgs[i]`, so scanning `imgs` alone suffices.
            let rect = self.config().rect && one_canvas(imgs, imgsz);
            let t_scale = Instant::now();
            let preds = run_scale_with_resident_fallback(
                &mut uploaded,
                |dev| self.run_uploaded_tta_views(dev, &views, rect, imgsz),
                || {
                    if let Some(ctx) = folo_gpu::context_for(self.device()) {
                        let _ = ctx.release_memory();
                    }
                },
                || self.run_tta_views(&views, rect, imgsz),
            )?;
            let scale_wall = ms(t_scale);
            let (mut b_pre, mut b_inf, mut b_post) = (0.0f64, 0.0f64, 0.0f64);
            for p in &preds {
                b_pre += p.timing.preprocess_ms;
                b_inf += p.timing.inference_ms;
                b_post += p.timing.postprocess_ms;
            }
            add_scaled_timing(
                &mut pre_ms,
                &mut inf_ms,
                &mut post_ms,
                b_pre,
                b_inf,
                b_post,
                scale_wall * inv_n,
            );

            let t_collect = Instant::now();
            let mut iter = preds.into_iter();
            for dets in &mut per_img_dets {
                dets.extend(iter.next().expect("batch size matches").detections);
            }
            if tta_cfg.hflip {
                for (i, dets) in per_img_dets.iter_mut().enumerate() {
                    let w = imgs[i].width;
                    let flip_pred = iter.next().expect("flipped batch matches");
                    dets.extend(
                        flip_pred
                            .detections
                            .into_iter()
                            .map(|d| Detection { bbox: unflip_xyxy_h(d.bbox, w), ..d }),
                    );
                }
            }
            post_ms += ms(t_collect) * inv_n;
        }
        drop(uploaded);

        let t_merge = Instant::now();
        let default_iou = self.config().iou;
        let default_max_det = self.config().max_det;
        let merged: Vec<Vec<Detection>> = per_img_dets
            .into_par_iter()
            .zip(imgs.par_iter())
            .map(|(dets, img)| {
                merge_tta_detections(dets, img.width, img.height, tta_cfg, default_iou, default_max_det)
            })
            .collect::<Result<_>>()?;
        post_ms += ms(t_merge) * inv_n;

        let timing = Timing::new(pre_ms, inf_ms, post_ms);
        Ok(imgs
            .iter()
            .zip(merged)
            .map(|(img, detections)| Prediction { width: img.width, height: img.height, detections, timing })
            .collect())
    }

    /// Decode an image file and predict it with Test-Time Augmentation ([`Predictor::predict_tta`]).
    pub fn predict_file_tta(&self, path: impl AsRef<Path>, tta_cfg: &TtaConfig) -> Result<Prediction> {
        tta_cfg.resolve_imgszs(self.config().imgsz)?;
        let t = Instant::now();
        let img = folo_data::read_image(path.as_ref())?;
        let decode_ms = ms(t);
        let mut p = self.predict_tta(&img, tta_cfg)?;
        p.timing =
            Timing::new(p.timing.preprocess_ms + decode_ms, p.timing.inference_ms, p.timing.postprocess_ms);
        Ok(p)
    }

    /// Predict one image using optional inference extensions ([`PredictOptions`]: TTA, SAHI tiling, or plain
    /// single-pass inference when neither is set).
    pub fn predict_with_options(&self, img: &Image, opts: &PredictOptions) -> Result<Prediction> {
        opts.validate(self.config().imgsz)?;
        match (&opts.tta, &opts.tile) {
            (Some(tta), None) => self.predict_tta(img, tta),
            (None, Some(tile)) => self.predict_tiled(img, tile),
            (None, None) => self.predict(img),
            (Some(_), Some(_)) => unreachable!("validated above"),
        }
    }

    /// Predict a batch of images using [`PredictOptions`], in input order.
    pub fn predict_batch_with_options(
        &self,
        imgs: &[Image],
        opts: &PredictOptions,
    ) -> Result<Vec<Prediction>> {
        opts.validate(self.config().imgsz)?;
        match (&opts.tta, &opts.tile) {
            (Some(tta), None) => self.predict_batch_tta(imgs, tta),
            (None, Some(tile)) => self.predict_batch_tiled(imgs, tile),
            (None, None) => self.predict_batch(imgs),
            (Some(_), Some(_)) => unreachable!("validated above"),
        }
    }

    /// Decode an image file and predict it using [`PredictOptions`].
    pub fn predict_file_with_options(
        &self,
        path: impl AsRef<Path>,
        opts: &PredictOptions,
    ) -> Result<Prediction> {
        opts.validate(self.config().imgsz)?;
        match (&opts.tta, &opts.tile) {
            (Some(tta), None) => self.predict_file_tta(path, tta),
            (None, Some(tile)) => self.predict_file_tiled(path, tile),
            (None, None) => self.predict_file(path),
            (Some(_), Some(_)) => unreachable!("validated above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unflip_xyxy_h_is_exact_involution_and_clamps_cleanly() {
        let w = 640u32;
        let orig = Xyxy::new(100.25, 42.5, 260.75, 198.0);
        let flipped = unflip_xyxy_h(orig, w);
        assert_eq!(flipped, Xyxy::new(640.0 - 260.75, 42.5, 640.0 - 100.25, 198.0));
        assert_eq!(unflip_xyxy_h(flipped, w), orig);

        // Edge-touching box maps to the opposite edge cleanly.
        let left = Xyxy::new(0.0, 10.0, 50.0, 40.0);
        let right = unflip_xyxy_h(left, 200);
        assert_eq!(right, Xyxy::new(150.0, 10.0, 200.0, 40.0));
        assert_eq!(unflip_xyxy_h(right, 200), left);

        // Slightly out-of-bounds box clamps into [0, width].
        let oob = Xyxy::new(-5.0, 10.0, 210.0, 40.0);
        assert_eq!(unflip_xyxy_h(oob, 200), Xyxy::new(0.0, 10.0, 200.0, 40.0));

        // Non-finite inputs stay non-finite rather than silently becoming valid edge coordinates.
        let nan_box = Xyxy::new(f32::NAN, 10.0, 50.0, 40.0);
        assert!(unflip_xyxy_h(nan_box, 200).x2.is_nan());
    }

    #[test]
    fn scale_validation_and_stride_resolution() {
        let def = TtaConfig::default();
        assert_eq!(def.resolve_imgszs(640).unwrap(), vec![544, 640, 736]);
        assert_eq!(def.resolve_imgszs(320).unwrap(), vec![256, 320, 384]);
        // Duplicate resolved stride-32 sizes are deduplicated in first-seen order.
        assert_eq!(def.resolve_imgszs(64).unwrap(), vec![64]);
        let ordered = TtaConfig::multi_scale(vec![1.5, 1.0, 0.5, 1.0]);
        assert_eq!(ordered.resolve_imgszs(64).unwrap(), vec![96, 64, 32]);

        // Reject empty, non-positive, non-finite, and out-of-range scales.
        assert!(TtaConfig { scales: vec![], ..Default::default() }.resolve_imgszs(640).is_err());
        assert!(TtaConfig { scales: vec![0.0], ..Default::default() }.resolve_imgszs(640).is_err());
        assert!(TtaConfig { scales: vec![-1.0], ..Default::default() }.resolve_imgszs(640).is_err());
        assert!(TtaConfig { scales: vec![f32::NAN], ..Default::default() }.resolve_imgszs(640).is_err());
        assert!(TtaConfig { scales: vec![f32::INFINITY], ..Default::default() }.resolve_imgszs(640).is_err());
        // Too small (< 32 after rounding) or too large (> 8192 after rounding).
        assert!(TtaConfig { scales: vec![0.01], ..Default::default() }.resolve_imgszs(640).is_err());
        assert!(TtaConfig { scales: vec![20.0], ..Default::default() }.resolve_imgszs(640).is_err());
        // Invalid base imgsz.
        assert!(def.resolve_imgszs(0).is_err());
        assert!(def.resolve_imgszs(50).is_err());
        assert!(def.resolve_imgszs(8224).is_err());
        // Invalid merge_iou / max_det.
        assert!(TtaConfig { merge_iou: Some(-0.1), ..Default::default() }.validate().is_err());
        assert!(TtaConfig { merge_iou: Some(1.1), ..Default::default() }.validate().is_err());
        assert!(TtaConfig { merge_iou: Some(f32::NAN), ..Default::default() }.validate().is_err());
        assert!(TtaConfig { max_det: Some(0), ..Default::default() }.validate().is_err());
    }

    #[test]
    fn predict_options_rejects_simultaneous_tta_and_tile() {
        let both = PredictOptions { tta: Some(TtaConfig::default()), tile: Some(TileConfig::default()) };
        assert!(both.validate(640).is_err());
        assert!(PredictOptions::with_tta(TtaConfig::default()).validate(640).is_ok());
        assert!(PredictOptions::with_tile(TileConfig::default()).validate(640).is_ok());
        assert!(PredictOptions::default().validate(640).is_ok());
    }

    #[test]
    fn merge_tta_detections_deduplicates_and_fuses_order_independently() {
        let orig = Detection { bbox: Xyxy::new(40.0, 20.0, 100.0, 80.0), score: 0.90, class: 0 };
        let flipped_unmapped = Detection {
            bbox: unflip_xyxy_h(Xyxy::new(96.0, 22.0, 156.0, 82.0), 200), // -> (44, 22, 104, 82)
            score: 0.70,
            class: 0,
        };
        let other_cls = Detection { bbox: Xyxy::new(42.0, 21.0, 102.0, 81.0), score: 0.80, class: 1 };

        for fusion in [BoxFusion::Vote, BoxFusion::Sebf, BoxFusion::None] {
            let cfg = TtaConfig { merge_iou: Some(0.5), merge_fusion: fusion, ..Default::default() };
            let a = merge_tta_detections(vec![orig, flipped_unmapped, other_cls], 200, 120, &cfg, 0.7, 300)
                .unwrap();
            let b = merge_tta_detections(vec![other_cls, flipped_unmapped, orig], 200, 120, &cfg, 0.7, 300)
                .unwrap();
            assert_eq!(a, b, "TTA merge must be independent of view order ({fusion:?})");
            assert_eq!(a.len(), 2, "class-aware merge keeps class 0 and class 1 separate");
            if fusion == BoxFusion::None {
                assert_eq!(a[0].bbox, orig.bbox);
            } else {
                // Fused class-0 box lies strictly between orig (40..100) and flipped_unmapped (44..104).
                assert!(a[0].bbox.x1 > 40.0 && a[0].bbox.x1 < 44.0, "{:?}", a[0].bbox);
                assert!(a[0].bbox.y1 > 20.0 && a[0].bbox.y1 < 22.0, "{:?}", a[0].bbox);
            }
        }

        // max_det cap and class-agnostic merging.
        let capped = TtaConfig { max_det: Some(1), ..Default::default() };
        let out = merge_tta_detections(vec![orig, other_cls], 200, 120, &capped, 0.5, 300).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, 0);

        let agnostic = TtaConfig { class_agnostic: true, merge_iou: Some(0.5), ..Default::default() };
        let out = merge_tta_detections(vec![orig, other_cls], 200, 120, &agnostic, 0.7, 300).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class, 0);

        // Non-finite and empty-image guards, plus invalid fallback thresholds/max_det.
        let bad = Detection { bbox: Xyxy::new(f32::NAN, 0.0, 10.0, 10.0), score: 0.9, class: 0 };
        assert!(
            merge_tta_detections(vec![bad], 200, 120, &TtaConfig::default(), 0.7, 300).unwrap().is_empty()
        );
        assert!(merge_tta_detections(vec![orig], 0, 120, &TtaConfig::default(), 0.7, 300).is_err());
        assert!(merge_tta_detections(vec![orig], 200, 120, &TtaConfig::default(), f32::NAN, 300).is_err());
        assert!(merge_tta_detections(vec![orig], 200, 120, &TtaConfig::default(), 1.5, 300).is_err());
        assert!(merge_tta_detections(vec![orig], 200, 120, &TtaConfig::default(), 0.7, 0).is_err());
    }

    #[test]
    fn merge_max_det_cap_retains_fusion_refinement_and_matches_uncapped_prefix() {
        // Two high-scoring disjoint clusters with lower-scoring overlapping refinements arriving AFTER many
        // disjoint lower-scoring candidates that exceed `max_det = 2`.
        let mut dets = vec![
            Detection { bbox: Xyxy::new(10.0, 10.0, 50.0, 50.0), score: 0.95, class: 0 },
            Detection { bbox: Xyxy::new(100.0, 10.0, 140.0, 50.0), score: 0.90, class: 0 },
        ];
        // Disjoint lower-scoring candidates past `max_det = 2`.
        for k in 0..8 {
            let x = 200.0 + (k as f32) * 50.0;
            dets.push(Detection {
                bbox: Xyxy::new(x, 10.0, x + 30.0, 40.0),
                score: 0.80 - (k as f32) * 0.02,
                class: 0,
            });
        }
        // Lower-scoring refinements overlapping the top-2 survivors.
        dets.push(Detection { bbox: Xyxy::new(14.0, 12.0, 54.0, 52.0), score: 0.50, class: 0 });
        dets.push(Detection { bbox: Xyxy::new(104.0, 12.0, 144.0, 52.0), score: 0.45, class: 0 });

        for fusion in [BoxFusion::None, BoxFusion::Vote, BoxFusion::Sebf] {
            let uncapped = TtaConfig {
                merge_iou: Some(0.5),
                merge_fusion: fusion,
                max_det: Some(100),
                ..Default::default()
            };
            let capped = TtaConfig { max_det: Some(2), ..uncapped.clone() };
            let full = merge_tta_detections(dets.clone(), 1000, 200, &uncapped, 0.5, 100).unwrap();
            let got = merge_tta_detections(dets.clone(), 1000, 200, &capped, 0.5, 100).unwrap();
            assert_eq!(got, full[..2], "capped merge diverged from uncapped prefix for {fusion:?}");
            if fusion != BoxFusion::None {
                assert!(got[0].bbox.x1 > 10.0 && got[0].bbox.x1 < 14.0, "{:?}", got[0].bbox);
                assert!(got[1].bbox.x1 > 100.0 && got[1].bbox.x1 < 104.0, "{:?}", got[1].bbox);
            }
        }
    }

    #[test]
    fn oom_fallback_splits_view_batches_in_order_and_propagates_other_errors() {
        let imgs: Vec<Image> = (0..6).map(|i| Image::new(8 + i, 8, [i as u8; 3])).collect();
        let refs: Vec<&Image> = imgs.iter().collect();
        let mut seen_sizes = Vec::new();
        let out = run_views_with_oom_fallback(&refs, &mut |chunk| {
            seen_sizes.push(chunk.len());
            if chunk.len() > 2 {
                return Err(Error::Core(folo_core::Error::Backend {
                    backend: "wgpu",
                    msg: "device limit exceeded: buffer of 200 bytes exceeds the device limit (128 bytes)"
                        .into(),
                }));
            }
            Ok(chunk.iter().map(|im| im.width).collect())
        })
        .unwrap();
        assert_eq!(out, vec![8, 9, 10, 11, 12, 13]);
        assert!(seen_sizes.contains(&6) && seen_sizes.iter().all(|&s| s <= 6));

        // Non-OOM errors are NOT retried.
        let mut calls = 0usize;
        let err = run_views_with_oom_fallback::<u32>(&refs, &mut |_| {
            calls += 1;
            Err(Error::Config("bad input".into()))
        });
        assert!(err.is_err());
        assert_eq!(calls, 1);

        // Single-view OOM cannot split further and returns the error.
        let err = run_views_with_oom_fallback::<u32>(&refs[..1], &mut |_| {
            Err(Error::Core(folo_core::Error::Backend { backend: "wgpu", msg: "Out of Memory".into() }))
        });
        assert!(err.is_err());

        // Uploaded-view OOM fallback keeps paired device tensors aligned across splits and propagates non-OOM errors.
        let dev_imgs: Vec<Tensor> = (0..6).map(|i| Tensor::full([1], i as f32)).collect();
        let out_up = run_uploaded_views_with_oom_fallback(&dev_imgs, &refs, &mut |devs, chunk| {
            assert_eq!(devs.len(), chunk.len());
            if chunk.len() > 2 {
                return Err(Error::Core(folo_core::Error::Backend {
                    backend: "wgpu",
                    msg: "Out of Memory".into(),
                }));
            }
            Ok(devs
                .iter()
                .zip(chunk)
                .map(|(t, im)| (t.to_vec::<f32>().unwrap()[0] as u32, im.width))
                .collect())
        })
        .unwrap();
        assert_eq!(out_up, vec![(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13)]);
    }

    #[test]
    fn resident_fallback_evicts_uploaded_buffers_releases_memory_and_retries_host() {
        use std::sync::Arc;

        // 1. On GPU OOM during the uploaded path, resident tensors are dropped BEFORE `release_memory()` and
        // `run_host()`, and `uploaded` remains `None` for subsequent scales.
        let sentinel = Arc::new(());
        let weak = Arc::downgrade(&sentinel);
        let mut uploaded = Some(sentinel);
        let mut released = 0usize;
        let mut host_calls = 0usize;
        let out = run_scale_with_resident_fallback(
            &mut uploaded,
            |_| Err(Error::Core(folo_core::Error::Backend { backend: "wgpu", msg: "Out of Memory".into() })),
            || {
                assert_eq!(weak.strong_count(), 0, "resident buffers must be dropped before release_memory");
                released += 1;
            },
            || {
                assert_eq!(weak.strong_count(), 0, "resident buffers must be dropped before host retry");
                host_calls += 1;
                Ok(42usize)
            },
        )
        .unwrap();
        assert_eq!(out, 42);
        assert!(uploaded.is_none());
        assert_eq!(released, 1);
        assert_eq!(host_calls, 1);

        // Next scale runs directly on the host fallback without attempting uploaded execution again.
        let out2 = run_scale_with_resident_fallback(
            &mut uploaded,
            |_| unreachable!("uploaded is cleared"),
            || released += 1,
            || {
                host_calls += 1;
                Ok(99usize)
            },
        )
        .unwrap();
        assert_eq!(out2, 99);
        assert_eq!(released, 1);
        assert_eq!(host_calls, 2);

        // 2. Non-OOM errors propagate without dropping uploaded buffers or retrying.
        let mut uploaded_ok = Some(7u32);
        let mut released_non_oom = 0usize;
        let err = run_scale_with_resident_fallback::<usize, u32>(
            &mut uploaded_ok,
            |_| Err(Error::Config("bad shape".into())),
            || released_non_oom += 1,
            || unreachable!("non-OOM must not retry host path"),
        );
        assert!(err.is_err());
        assert_eq!(uploaded_ok, Some(7));
        assert_eq!(released_non_oom, 0);
    }

    #[test]
    fn serde_json_round_trip_and_partial_defaults() {
        let def: TtaConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(def, TtaConfig::default());
        let json = serde_json::to_string(&def).unwrap();
        assert_eq!(serde_json::from_str::<TtaConfig>(&json).unwrap(), def);

        let opts_empty: PredictOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(opts_empty, PredictOptions::default());
        let opts_tta: PredictOptions = serde_json::from_str(r#"{"tta": {}}"#).unwrap();
        assert_eq!(opts_tta, PredictOptions::with_tta(TtaConfig::default()));
    }
}
