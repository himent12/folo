//! Test-Time Augmentation ([`Detector::detect_tta`]) for `folo-wasm`.
//!
//! Mirrors [`folo_vision::Predictor::predict_tta`] over browser RGBA inputs:
//! - resolves `scales` against the detector's base `imgsz` into unique stride-32 sizes (`32..=8192`);
//! - on the CPU, horizontally mirrors RGBA via SIMD128 `flip_rgba_h` into a reusable scratch buffer (zero
//!   intermediate RGB allocation) and runs each scale through [`Detector::detect_cpu_raw`], matching
//!   `Predictor::predict_tta` bit for bit;
//! - on WebGPU, batches `[orig, flipped]` together per scale through [`Detector::detect_gpu_batch`] without
//!   cloning the pooled original RGB image (compact async candidate readback with OOM split fallback);
//! - un-flips mirrored detections (`unflip_xyxy_h`) and merges all views deterministically via
//!   [`folo_vision::tta::merge_tta_detections`].

use folo_data::Image;
use folo_vision::Timing;
use folo_vision::nms::Detection;
use folo_vision::tta::{merge_tta_detections, unflip_xyxy_h};

use crate::simd::flip_rgba_h;

use super::{DetectResult, Detector, Instant, PooledRgb, TtaDetectOptions, err, ms, validate_rgba};

/// Owns the WebGPU TTA view slice (`[orig]` or `[orig, flipped]`) without cloning `orig`, returning `orig`'s RGB
/// buffer to `Detector`'s scratch pool on drop.
struct PooledTtaViews<'a> {
    detector: &'a Detector,
    views: Vec<Image>,
}

impl<'a> PooledTtaViews<'a> {
    fn new(pooled: PooledRgb<'a>, detector: &'a Detector, hflip: bool) -> Self {
        let orig = pooled.into_image();
        let views = if hflip {
            let flipped = orig.oriented(2);
            vec![orig, flipped]
        } else {
            vec![orig]
        };
        PooledTtaViews { detector, views }
    }

    fn as_slice(&self) -> &[Image] {
        &self.views
    }
}

impl Drop for PooledTtaViews<'_> {
    fn drop(&mut self) {
        if let Some(orig) = self.views.drain(..).next() {
            self.detector.return_rgb_buffer(orig.data);
        }
    }
}

impl Detector {
    /// Detect objects using Test-Time Augmentation (multi-scale and/or horizontal-flip ensembling).
    pub async fn detect_tta(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        opts: TtaDetectOptions,
    ) -> Result<DetectResult, String> {
        let t0 = Instant::now();
        validate_rgba(rgba, width, height)?;
        let tta_cfg = opts.to_tta_config()?;
        let (cfg, nms) = self.build_predict_and_nms(
            opts.conf,
            opts.iou,
            opts.max_det,
            opts.per_class_conf,
            opts.conf_inclusive,
            self.base_cfg.imgsz,
        )?;
        let imgszs = tta_cfg.resolve_imgszs(self.base_cfg.imgsz).map_err(err)?;

        // Single unflipped view: degrade cleanly to one pass at that size, skipping cross-view merge.
        if !tta_cfg.hflip && imgszs.len() == 1 {
            let single_cfg = self.predict_config(opts.conf, opts.iou, opts.max_det, imgszs[0])?;
            return self.detect_single_with_cfg(rgba, width, height, &single_cfg, &nms, t0).await;
        }

        let mut pre_ms = 0.0f64;
        let mut inf_ms = 0.0f64;
        let mut post_ms = 0.0f64;
        let mut all_dets: Vec<Detection> = Vec::new();

        if let Some(ctx) = &self.gpu {
            let t_rgb = Instant::now();
            let pooled = PooledRgb::new(self, rgba, width, height)?;
            let views = PooledTtaViews::new(pooled, self, tta_cfg.hflip);
            pre_ms += ms(t_rgb);

            for &imgsz in &imgszs {
                let scale_cfg = self.predict_config(opts.conf, opts.iou, opts.max_det, imgsz)?;
                let (mut preds, timing) =
                    self.detect_gpu_batch(ctx, views.as_slice(), &scale_cfg, &nms).await?;
                pre_ms += timing.preprocess_ms;
                inf_ms += timing.inference_ms;
                post_ms += timing.postprocess_ms;

                let t_collect = Instant::now();
                if tta_cfg.hflip {
                    let flip_dets = preds.pop().expect("two views in, two out");
                    let orig_dets = preds.pop().expect("two views in, two out");
                    all_dets.extend(orig_dets);
                    all_dets.extend(
                        flip_dets.into_iter().map(|d| Detection { bbox: unflip_xyxy_h(d.bbox, width), ..d }),
                    );
                } else {
                    all_dets.extend(preds.pop().expect("one view in, one out"));
                }
                post_ms += ms(t_collect);
            }
        } else {
            // Direct CPU RGBA path: reuse `scratch.rgba_work` for the horizontally flipped RGBA image.
            let mut flipped_rgba = if tta_cfg.hflip {
                let t_flip = Instant::now();
                let mut buf = std::mem::take(&mut self.scratch.borrow_mut().rgba_work);
                flip_rgba_h(rgba, width, height, &mut buf);
                pre_ms += ms(t_flip);
                Some(buf)
            } else {
                None
            };

            let run = (|| -> Result<(), String> {
                for &imgsz in &imgszs {
                    let t_orig = Instant::now();
                    let (orig_dets, timing) =
                        self.detect_cpu_raw(rgba, width, height, imgsz, cfg.rect, &nms, t_orig)?;
                    pre_ms += timing.preprocess_ms;
                    inf_ms += timing.inference_ms;
                    post_ms += timing.postprocess_ms;
                    all_dets.extend(orig_dets);

                    if let Some(fl) = flipped_rgba.as_deref() {
                        let t_fl = Instant::now();
                        let (flip_dets, timing) =
                            self.detect_cpu_raw(fl, width, height, imgsz, cfg.rect, &nms, t_fl)?;
                        pre_ms += timing.preprocess_ms;
                        inf_ms += timing.inference_ms;
                        post_ms += timing.postprocess_ms;

                        let t_unflip = Instant::now();
                        all_dets.extend(
                            flip_dets
                                .into_iter()
                                .map(|d| Detection { bbox: unflip_xyxy_h(d.bbox, width), ..d }),
                        );
                        post_ms += ms(t_unflip);
                    }
                }
                Ok(())
            })();

            if let Some(buf) = flipped_rgba.take() {
                let mut scratch = self.scratch.borrow_mut();
                if buf.capacity() > scratch.rgba_work.capacity() {
                    scratch.rgba_work = buf;
                }
            }
            run?;
        }

        let t_merge = Instant::now();
        let detections =
            merge_tta_detections(all_dets, width, height, &tta_cfg, cfg.iou, cfg.max_det).map_err(err)?;
        post_ms += ms(t_merge);

        Ok(self.build_result(width, height, detections, Timing::new(pre_ms, inf_ms, post_ms)))
    }
}
