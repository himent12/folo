//! SAHI tiled inference ([`Detector::detect_tiled`]) for `folo-wasm`.
//!
//! Mirrors [`folo_vision::Predictor::predict_tiled`] over browser RGBA inputs:
//! - enforces a browser-safe tile count cap ([`MAX_BROWSER_TILES`]) in `O(1)` before allocating the tile grid,
//!   so adversarial inputs like `tileSize: 1` on a large image reject cleanly instead of exhausting wasm32 memory;
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
use folo_vision::tile::{TileConfig, compute_tiles, crop_tile, merge_tiled_detections};

use crate::simd::crop_rgba_tile;

use super::{
    DetectResult, Detector, Instant, PooledRgb, TileDetectOptions, err, ms, nms_config, validate_rgba,
};

/// Maximum number of tiles allowed in one browser [`Detector::detect_tiled`] call (64×64 grid). Checked in `O(1)`
/// before allocating the tile grid so adversarial `tileSize` / `overlap` combinations reject cleanly instead of
/// hitting wasm32's allocation limits.
pub const MAX_BROWSER_TILES: usize = 4096;

/// Compute the exact number of tiles along one 1D axis in `O(1)` (matching `folo_vision::tile::axis_starts`).
fn axis_tile_count(img_len: u32, tile_len: u32, overlap: f32) -> u64 {
    let tw = tile_len.min(img_len);
    if tw >= img_len {
        return 1;
    }
    let overlap_px = ((tw as f64) * (overlap as f64)).round() as u32;
    let stride = tw.saturating_sub(overlap_px.min(tw - 1)).max(1) as u64;
    let span = (img_len - tw) as u64;
    1 + span.div_ceil(stride)
}

/// Validate in `O(1)` that the tile grid will not exceed [`MAX_BROWSER_TILES`] before calling `compute_tiles`.
pub(crate) fn check_tile_grid_limit(img_w: u32, img_h: u32, cfg: &TileConfig) -> Result<usize, String> {
    let nx = axis_tile_count(img_w, cfg.tile_width, cfg.overlap_x);
    let ny = axis_tile_count(img_h, cfg.tile_height, cfg.overlap_y);
    let total = nx.checked_mul(ny);
    match total {
        Some(n) if n <= MAX_BROWSER_TILES as u64 => Ok(n as usize),
        Some(n) => Err(format!(
            "tile grid has {n} tiles ({nx}x{ny}) for a {img_w}x{img_h} image, exceeding the browser limit of {MAX_BROWSER_TILES}"
        )),
        None => Err(format!(
            "tile grid ({nx}x{ny}) overflows capacity for a {img_w}x{img_h} image (browser limit is {MAX_BROWSER_TILES})"
        )),
    }
}

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
        check_tile_grid_limit(width, height, &tile_cfg)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_grid_count_matches_compute_tiles_and_rejects_huge_grids_in_o1() {
        for (w, h, tw, th, ox, oy) in [
            (64u32, 64u32, 128u32, 128u32, 0.2f32, 0.2f32),
            (800, 500, 300, 200, 0.2, 0.25),
            (513, 319, 128, 96, 0.0, 0.5),
            (50, 50, 10, 10, 0.9, 0.9),
        ] {
            let cfg = TileConfig {
                tile_width: tw,
                tile_height: th,
                overlap_x: ox,
                overlap_y: oy,
                ..Default::default()
            };
            let got = check_tile_grid_limit(w, h, &cfg).unwrap();
            let want = compute_tiles(w, h, &cfg).unwrap().len();
            assert_eq!(got, want, "{w}x{h} tile {tw}x{th} overlap {ox}/{oy}");
        }

        // Adversarial 12000x12000 with tile_size = 1 (144M tiles) is rejected in O(1) with zero allocation.
        let huge = TileConfig {
            tile_width: 1,
            tile_height: 1,
            overlap_x: 0.0,
            overlap_y: 0.0,
            ..Default::default()
        };
        let err = check_tile_grid_limit(12_000, 12_000, &huge).unwrap_err();
        assert!(err.contains(&MAX_BROWSER_TILES.to_string()), "{err}");
    }
}
