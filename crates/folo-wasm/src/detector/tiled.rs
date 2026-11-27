//! SAHI tiled inference ([`Detector::detect_tiled`]) for `folo-wasm`.
//!
//! Mirrors [`folo_vision::Predictor::predict_tiled`] over browser RGBA inputs:
//! - on the CPU, crops tiles directly from RGBA into a reusable scratch buffer (zero full-image RGB copy) and
//!   runs each tile through [`Detector::detect_cpu_raw`], matching `Predictor::predict_tiled` bit for bit;
//! - on WebGPU, strips RGBA→RGB once into a reusable scratch buffer, crops tile batches of `tile_batch` tiles,
//!   and predicts each batch via [`Detector::detect_gpu_batch`] (with compact async candidate readback and OOM
//!   chunk-splitting);
//! - when the image fits in a single tile, degrades straight to single-pass inference with zero slicing overhead.

use folo_data::Image;
use folo_vision::Timing;
use folo_vision::boxes::Xyxy;
use folo_vision::nms::Detection;
use folo_vision::tile::{compute_tiles, crop_tile, merge_tiled_detections};

use crate::simd::crop_rgba_tile;

use super::{
    DetectResult, Detector, Instant, PooledRgb, TileDetectOptions, err, ms, nms_config, validate_rgba,
};

impl Detector {
    /// Detect objects using Slicing Aided Hyper Inference (SAHI): slices an RGBA image into uniform overlapping
    /// tiles, runs the tiles (batched on WebGPU; direct RGBA-cropped on CPU), optionally runs a full-image pass,
    /// and merges all detections in original-image coordinates.
    pub async fn detect_tiled(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        opts: TileDetectOptions,
    ) -> Result<DetectResult, String> {
        let t0 = Instant::now();
        validate_rgba(rgba, width, height)?;
        let cfg = self.predict_config(opts.conf, opts.iou, opts.max_det, self.base_cfg.imgsz)?;
        let tile_cfg = opts.to_tile_config()?;
        let tiles = compute_tiles(width, height, &tile_cfg).map_err(err)?;
        let nms = nms_config(&cfg);

        // Single-tile fast path: degrade cleanly to single-pass inference with zero cropping overhead.
        if tiles.len() == 1 {
            return self.detect_single_with_cfg(rgba, width, height, &cfg, &nms, t0).await;
        }

        let mut pre_ms = 0.0f64;
        let mut inf_ms = 0.0f64;
        let mut post_ms = 0.0f64;
        let mut all_dets: Vec<Detection> = Vec::new();

        if let Some(ctx) = &self.gpu {
            let t_rgb = Instant::now();
            let pooled = PooledRgb::new(self, rgba, width, height)?;
            let full_img = pooled.image();
            pre_ms += ms(t_rgb);

            for chunk in tiles.chunks(tile_cfg.tile_batch) {
                let t_crop = Instant::now();
                let crops: Vec<Image> = chunk
                    .iter()
                    .map(|&r| crop_tile(full_img, r))
                    .collect::<folo_vision::Result<_>>()
                    .map_err(err)?;
                pre_ms += ms(t_crop);

                let (batch_dets, timing) = self.detect_gpu_batch(ctx, &crops, &cfg, &nms).await?;
                pre_ms += timing.preprocess_ms;
                inf_ms += timing.inference_ms;
                post_ms += timing.postprocess_ms;

                let t_shift = Instant::now();
                for (dets, tile) in batch_dets.into_iter().zip(chunk) {
                    let (ox, oy) = (tile.x as f32, tile.y as f32);
                    all_dets.extend(dets.into_iter().map(|d| Detection {
                        bbox: Xyxy::new(d.bbox.x1 + ox, d.bbox.y1 + oy, d.bbox.x2 + ox, d.bbox.y2 + oy),
                        ..d
                    }));
                }
                post_ms += ms(t_shift);
            }

            if tile_cfg.full_image {
                let (mut full_dets, timing) =
                    self.detect_gpu_batch(ctx, std::slice::from_ref(full_img), &cfg, &nms).await?;
                pre_ms += timing.preprocess_ms;
                inf_ms += timing.inference_ms;
                post_ms += timing.postprocess_ms;
                all_dets.extend(full_dets.pop().expect("one image in, one out"));
            }
        } else {
            // Direct CPU RGBA path: reuse a single tile RGBA scratch buffer across all tiles (no full RGB copy).
            let mut tile_rgba = std::mem::take(&mut self.scratch.borrow_mut().rgba_work);
            let run = (|| -> Result<(), String> {
                for &tile in &tiles {
                    let t_tile = Instant::now();
                    crop_rgba_tile(rgba, width, height, tile, &mut tile_rgba);
                    let (dets, timing) = self.detect_cpu_raw(
                        &tile_rgba,
                        tile.width,
                        tile.height,
                        cfg.imgsz,
                        cfg.rect,
                        &nms,
                        t_tile,
                    )?;
                    pre_ms += timing.preprocess_ms;
                    inf_ms += timing.inference_ms;
                    post_ms += timing.postprocess_ms;

                    let t_shift = Instant::now();
                    let (ox, oy) = (tile.x as f32, tile.y as f32);
                    all_dets.extend(dets.into_iter().map(|d| Detection {
                        bbox: Xyxy::new(d.bbox.x1 + ox, d.bbox.y1 + oy, d.bbox.x2 + ox, d.bbox.y2 + oy),
                        ..d
                    }));
                    post_ms += ms(t_shift);
                }
                Ok(())
            })();
            {
                let mut scratch = self.scratch.borrow_mut();
                if tile_rgba.capacity() > scratch.rgba_work.capacity() {
                    scratch.rgba_work = tile_rgba;
                }
            }
            run?;

            if tile_cfg.full_image {
                let t_full = Instant::now();
                let (dets, timing) =
                    self.detect_cpu_raw(rgba, width, height, cfg.imgsz, cfg.rect, &nms, t_full)?;
                pre_ms += timing.preprocess_ms;
                inf_ms += timing.inference_ms;
                post_ms += timing.postprocess_ms;
                all_dets.extend(dets);
            }
        }

        let t_merge = Instant::now();
        let detections = merge_tiled_detections(all_dets, width, height, &tile_cfg).map_err(err)?;
        post_ms += ms(t_merge);

        Ok(self.build_result(width, height, detections, Timing::new(pre_ms, inf_ms, post_ms)))
    }
}
