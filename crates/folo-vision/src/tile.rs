//! Slicing Aided Hyper Inference (SAHI) tiled detection over [`crate::Predictor`].
//!
//! Large high-resolution images lose small objects when downscaled straight to `imgsz`. Tiled inference slices an
//! image into uniform overlapping [`TileRect`] windows ([`compute_tiles`]), crops each window ([`crop_tile`]),
//! predicts them in bounded batches through [`crate::Predictor::predict_batch`] (preserving every CPU, int8, GPU,
//! f16, o2o, and candidate-first fast path), optionally runs one full-image pass to keep large boundary-spanning
//! objects intact, and merges all detections in original-image coordinates ([`merge_tiled_detections`]).

use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use folo_data::Image;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::boxes::{IouKind, Xyxy, iou as box_iou};
use crate::error::{Error, Result};
use crate::fusion::{BoxFusion, fuse_boxes_sebf};
use crate::nms::Detection;
use crate::predict::{Prediction, Predictor, Timing, add_scaled_timing, ms, rect_canvas};
use crate::tta::run_views_with_oom_fallback;

/// One axis-aligned tile crop inside an image, in original-image pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileRect {
    /// Left pixel offset (`0..img_w`).
    pub x: u32,
    /// Top pixel offset (`0..img_h`).
    pub y: u32,
    /// Crop width in pixels (`1..=img_w - x`).
    pub width: u32,
    /// Crop height in pixels (`1..=img_h - y`).
    pub height: u32,
}

/// Overlap metric used when merging detections across overlapping tiles and the optional full-image pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TileMergeMetric {
    /// Intersection-over-Smaller (`inter / min(area_a, area_b)`): suppresses both duplicate detections across
    /// overlapping tiles and truncated partial fragments clipped at a tile seam when a full box covers them.
    #[default]
    Ios,
    /// Standard Intersection-over-Union (`inter / union`).
    Iou,
}

/// Settings for tiled inference ([`crate::Predictor::predict_tiled`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TileConfig {
    /// Target tile width in pixels (clamped to the image width when the image is narrower).
    pub tile_width: u32,
    /// Target tile height in pixels (clamped to the image height when the image is shorter).
    pub tile_height: u32,
    /// Horizontal overlap ratio between adjacent tiles in `[0.0, 1.0)`.
    pub overlap_x: f32,
    /// Vertical overlap ratio between adjacent tiles in `[0.0, 1.0)`.
    pub overlap_y: f32,
    /// Also run a full-image pass alongside the tiles so large objects spanning tile boundaries are kept whole.
    pub full_image: bool,
    /// Maximum number of cropped tiles predicted in one [`crate::Predictor::predict_batch`] call.
    pub tile_batch: usize,
    /// Overlap metric used to cluster and suppress cross-tile duplicates ([`TileMergeMetric::Ios`] by default).
    pub merge_metric: TileMergeMetric,
    /// Merge overlap threshold in `[0.0, 1.0]`: candidates with `metric > merge_threshold` are assigned to the
    /// higher-scoring survivor.
    pub merge_threshold: f32,
    /// Box refinement across agreeing cross-tile detections ([`BoxFusion::Vote`] by default). When
    /// [`TileMergeMetric::Ios`] suppresses a truncated fragment, only cluster members whose IoU also exceeds
    /// `merge_threshold` contribute to fused coordinates so partial fragments do not shrink a full box.
    /// Cross-tile merging has no DFL logits, so [`BoxFusion::Dfl`] refines as [`BoxFusion::Vote`].
    pub merge_fusion: BoxFusion,
    /// Merge across different classes (`false`: class-aware merging, the default).
    pub class_agnostic: bool,
    /// Maximum merged detections kept per image.
    pub max_det: usize,
}

impl Default for TileConfig {
    fn default() -> Self {
        TileConfig {
            tile_width: 640,
            tile_height: 640,
            overlap_x: 0.2,
            overlap_y: 0.2,
            full_image: true,
            tile_batch: 8,
            merge_metric: TileMergeMetric::Ios,
            merge_threshold: 0.5,
            merge_fusion: BoxFusion::Vote,
            class_agnostic: false,
            max_det: 300,
        }
    }
}

impl TileConfig {
    /// Square `tile_size × tile_size` tiled config with default overlap (`0.2`), full-image pass, and IoS merge.
    pub fn new(tile_size: u32) -> Self {
        TileConfig { tile_width: tile_size, tile_height: tile_size, ..Self::default() }
    }

    /// Validate all fields before slicing or merging.
    pub fn validate(&self) -> Result<()> {
        if self.tile_width == 0 || self.tile_height == 0 {
            return Err(Error::Config(format!(
                "tile dimensions must be positive, got {}x{}",
                self.tile_width, self.tile_height
            )));
        }
        if !self.overlap_x.is_finite() || !(0.0..1.0).contains(&self.overlap_x) {
            return Err(Error::Config(format!(
                "tile overlap_x must be finite and in [0, 1), got {}",
                self.overlap_x
            )));
        }
        if !self.overlap_y.is_finite() || !(0.0..1.0).contains(&self.overlap_y) {
            return Err(Error::Config(format!(
                "tile overlap_y must be finite and in [0, 1), got {}",
                self.overlap_y
            )));
        }
        if self.tile_batch == 0 {
            return Err(Error::Config("tile_batch must be at least 1".into()));
        }
        if !self.merge_threshold.is_finite() || !(0.0..=1.0).contains(&self.merge_threshold) {
            return Err(Error::Config(format!(
                "merge_threshold must be finite and in [0, 1], got {}",
                self.merge_threshold
            )));
        }
        if self.max_det == 0 {
            return Err(Error::Config("tile max_det must be at least 1".into()));
        }
        Ok(())
    }
}

/// Axis offsets for tiling a 1D span of `img_len` pixels with tiles of size `tile_len` and `overlap` ratio.
/// Every returned tile has the same effective width `tw = tile_len.min(img_len)` (the last tile clamps back to
/// `img_len - tw`), guaranteeing 100% pixel coverage and uniform canvas dimensions for batched prediction.
fn axis_starts(img_len: u32, tile_len: u32, overlap: f32) -> (u32, Vec<u32>) {
    let tw = tile_len.min(img_len);
    if tw == img_len {
        return (tw, vec![0]);
    }
    let overlap_px = ((tw as f64) * (overlap as f64)).round() as u32;
    let stride = tw.saturating_sub(overlap_px.min(tw - 1)).max(1);
    let mut starts = Vec::new();
    let mut pos = 0u32;
    loop {
        starts.push(pos);
        if pos + tw >= img_len {
            break;
        }
        let next = pos.saturating_add(stride).min(img_len - tw);
        if next == pos {
            break;
        }
        pos = next;
    }
    (tw, starts)
}

/// Compute the uniform overlapping tile grid covering an `img_w × img_h` image.
///
/// All tiles share identical `(width, height)` (`tile_width.min(img_w) × tile_height.min(img_h)`), so a whole
/// batch of crops shares one `rect` letterbox canvas inside [`crate::Predictor::predict_batch`].
pub fn compute_tiles(img_w: u32, img_h: u32, cfg: &TileConfig) -> Result<Vec<TileRect>> {
    cfg.validate()?;
    if img_w == 0 || img_h == 0 {
        return Err(Error::Config(format!("cannot tile an empty image ({img_w}x{img_h})")));
    }
    let (tw, xs) = axis_starts(img_w, cfg.tile_width, cfg.overlap_x);
    let (th, ys) = axis_starts(img_h, cfg.tile_height, cfg.overlap_y);
    let count = xs
        .len()
        .checked_mul(ys.len())
        .ok_or_else(|| Error::Config(format!("tile grid overflows capacity for {img_w}x{img_h}")))?;
    let mut out = Vec::with_capacity(count);
    for &y in &ys {
        for &x in &xs {
            out.push(TileRect { x, y, width: tw, height: th });
        }
    }
    Ok(out)
}

/// Verify that `img` has non-zero dimensions and a matching `width * height * 3` RGB byte buffer.
pub(crate) fn check_image(img: &Image) -> Result<()> {
    if img.width == 0 || img.height == 0 {
        return Err(Error::Config(format!("cannot predict an empty image ({}x{})", img.width, img.height)));
    }
    let expected = (img.width as usize)
        .checked_mul(img.height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Config("image dimensions overflow".into()))?;
    if img.data.len() != expected {
        return Err(Error::Config(format!(
            "image buffer has {} bytes, expected {}x{}x3 ({expected})",
            img.data.len(),
            img.width,
            img.height
        )));
    }
    Ok(())
}

/// Copy `rect` rows from an already-validated `img` into `dst`, reusing `dst.data`'s allocation when sized.
fn copy_tile_rows(img: &Image, rect: TileRect, dst: &mut Image) {
    let dst_row = rect.width as usize * 3;
    let need = dst_row * rect.height as usize;
    dst.width = rect.width;
    dst.height = rect.height;
    if dst.data.len() != need {
        dst.data.resize(need, 0);
    }
    let src_stride = img.stride();
    let x_off = rect.x as usize * 3;
    for row in 0..rect.height as usize {
        let src_start = (rect.y as usize + row) * src_stride + x_off;
        let dst_start = row * dst_row;
        dst.data[dst_start..dst_start + dst_row].copy_from_slice(&img.data[src_start..src_start + dst_row]);
    }
}

/// Crop `rect` from `img` into `dst`, reusing `dst`'s backing pixel buffer when already sized to avoid
/// repeated multi-megabyte allocations across SAHI tile chunks.
pub fn crop_tile_into(img: &Image, rect: TileRect, dst: &mut Image) -> Result<()> {
    check_image(img)?;
    let x_end = rect.x.checked_add(rect.width);
    let y_end = rect.y.checked_add(rect.height);
    if rect.width == 0
        || rect.height == 0
        || x_end.is_none_or(|e| e > img.width)
        || y_end.is_none_or(|e| e > img.height)
    {
        return Err(Error::Config(format!(
            "tile {rect:?} is out of bounds for {}x{} image",
            img.width, img.height
        )));
    }
    copy_tile_rows(img, rect, dst);
    Ok(())
}

/// Crop `rect` from `img` using contiguous row-slice copies.
pub fn crop_tile(img: &Image, rect: TileRect) -> Result<Image> {
    let mut dst = Image { width: 0, height: 0, data: Vec::new() };
    crop_tile_into(img, rect, &mut dst)?;
    Ok(dst)
}

#[inline]
fn box_overlap(a: &Xyxy, b: &Xyxy, metric: TileMergeMetric) -> f32 {
    match metric {
        TileMergeMetric::Iou => box_iou(a, b, IouKind::Iou),
        TileMergeMetric::Ios => {
            let iw = (a.x2.min(b.x2) as f64 - a.x1.max(b.x1) as f64).max(0.0);
            let ih = (a.y2.min(b.y2) as f64 - a.y1.max(b.y1) as f64).max(0.0);
            let inter = iw * ih;
            if inter <= 0.0 {
                return 0.0;
            }
            let area_a = (a.x2 as f64 - a.x1 as f64).max(0.0) * (a.y2 as f64 - a.y1 as f64).max(0.0);
            let area_b = (b.x2 as f64 - b.x1 as f64).max(0.0) * (b.y2 as f64 - b.y1 as f64).max(0.0);
            let min_area = area_a.min(area_b);
            if min_area <= 0.0 { 0.0 } else { (inter / min_area).min(1.0) as f32 }
        }
    }
}

#[inline]
fn clip_box(b: Xyxy, fw: f32, fh: f32) -> Option<Xyxy> {
    if !(b.x1.is_finite() && b.y1.is_finite() && b.x2.is_finite() && b.y2.is_finite()) {
        return None;
    }
    let x1 = b.x1.clamp(0.0, fw);
    let y1 = b.y1.clamp(0.0, fh);
    let x2 = b.x2.clamp(0.0, fw);
    let y2 = b.y2.clamp(0.0, fh);
    (x2 > x1 && y2 > y1).then(|| Xyxy::new(x1, y1, x2, y2))
}

/// Shared clustering and box-fusion parameters for merging multi-view detections (SAHI tiles and TTA views).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MergeOptions {
    pub metric: TileMergeMetric,
    pub threshold: f32,
    pub fusion: BoxFusion,
    pub class_agnostic: bool,
    pub max_det: usize,
}

/// Merge detections gathered from overlapping tiles (and optional full-image pass) in original-image pixels.
///
/// Detections are clamped to `[0, img_w] × [0, img_h]`, sorted deterministically by descending score (with
/// coordinate tie-breakers so results never depend on tile order), clustered greedily under
/// [`TileConfig::merge_metric`] and [`TileConfig::merge_threshold`], optionally refined with
/// [`TileConfig::merge_fusion`], and capped to [`TileConfig::max_det`].
pub fn merge_tiled_detections(
    dets: Vec<Detection>,
    img_w: u32,
    img_h: u32,
    cfg: &TileConfig,
) -> Result<Vec<Detection>> {
    cfg.validate()?;
    merge_detections(
        dets,
        img_w,
        img_h,
        MergeOptions {
            metric: cfg.merge_metric,
            threshold: cfg.merge_threshold,
            fusion: cfg.merge_fusion,
            class_agnostic: cfg.class_agnostic,
            max_det: cfg.max_det,
        },
    )
}

/// Shared deterministic multi-view detection merge (clip, canonical sort, greedy exclusive clustering, optional
/// `Vote`/`Sebf` box fusion, and `max_det` cap). Used by both SAHI tiled inference and TTA.
pub(crate) fn merge_detections(
    dets: Vec<Detection>,
    img_w: u32,
    img_h: u32,
    opts: MergeOptions,
) -> Result<Vec<Detection>> {
    if img_w == 0 || img_h == 0 {
        return Err(Error::Config(format!("cannot merge detections for empty image ({img_w}x{img_h})")));
    }
    if !opts.threshold.is_finite() || !(0.0..=1.0).contains(&opts.threshold) {
        return Err(Error::Config(format!(
            "merge threshold must be finite and in [0, 1], got {}",
            opts.threshold
        )));
    }
    if opts.max_det == 0 {
        return Err(Error::Config("merge max_det must be at least 1".into()));
    }
    let (fw, fh) = (img_w as f32, img_h as f32);
    let mut clean: Vec<Detection> = dets
        .into_iter()
        .filter_map(|d| {
            if !d.score.is_finite() {
                return None;
            }
            let bbox = clip_box(d.bbox, fw, fh)?;
            Some(Detection { bbox, score: d.score, class: d.class })
        })
        .collect();
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    // Canonical sort so cross-view tie-breaking is independent of view ordering, batching, or thread scheduling.
    clean.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.class.cmp(&b.class))
            .then_with(|| a.bbox.x1.total_cmp(&b.bbox.x1))
            .then_with(|| a.bbox.y1.total_cmp(&b.bbox.y1))
            .then_with(|| a.bbox.x2.total_cmp(&b.bbox.x2))
            .then_with(|| a.bbox.y2.total_cmp(&b.bbox.y2))
    });

    // Greedy suppression with exclusive ownership: `owners[i]` records the position in `kept` (`0..kept.len()`)
    // of the survivor that kept or absorbed candidate `i`, matching `fuse_boxes_sebf`'s owner convention.
    // Once `kept` reaches `opts.max_det`, stop adding new survivors (`O(N * max_det)` instead of `O(N^2)`);
    // for `Vote`/`Dfl`/`Sebf`, keep scanning remaining candidates so they still refine the retained survivors,
    // which is bit-identical to keeping all survivors and truncating at the end because `kept` is scanned in
    // ascending slot order. `BoxFusion::None` stops immediately at the cap.
    let n = clean.len();
    let mut kept: Vec<usize> = Vec::with_capacity(n.min(opts.max_det));
    let mut owners: Vec<u32> = vec![u32::MAX; n];
    for i in 0..n {
        let cand = &clean[i];
        let mut matched: Option<usize> = None;
        for (slot, &k) in kept.iter().enumerate() {
            if !opts.class_agnostic && clean[k].class != cand.class {
                continue;
            }
            if box_overlap(&clean[k].bbox, &cand.bbox, opts.metric) > opts.threshold {
                matched = Some(slot);
                break;
            }
        }
        if let Some(slot) = matched {
            owners[i] = slot as u32;
        } else if kept.len() < opts.max_det {
            owners[i] = kept.len() as u32;
            kept.push(i);
            if kept.len() == opts.max_det && opts.fusion == BoxFusion::None {
                break;
            }
        }
    }

    let fused_boxes: Vec<Xyxy> = match opts.fusion {
        BoxFusion::None => kept.iter().map(|&k| clean[k].bbox).collect(),
        BoxFusion::Vote | BoxFusion::Dfl => {
            // Group owned members per survivor; fuse only members whose IoU also meets `threshold`
            // (plus the survivor itself) so IoS-suppressed truncated tile fragments never shrink a full box.
            let mut acc = vec![[0.0f64; 4]; kept.len()];
            let mut sum = vec![0.0f64; kept.len()];
            for (i, d) in clean.iter().enumerate() {
                if owners[i] == u32::MAX {
                    continue;
                }
                let slot = owners[i] as usize;
                let owner = kept[slot];
                if i != owner {
                    let iou = box_iou(&clean[owner].bbox, &d.bbox, IouKind::Iou);
                    if iou <= 0.0 || iou < opts.threshold {
                        continue;
                    }
                }
                let wt = d.score as f64;
                if wt.is_finite() && wt > 0.0 {
                    let edges = d.bbox.to_array();
                    for e in 0..4 {
                        acc[slot][e] += wt * edges[e] as f64;
                    }
                    sum[slot] += wt;
                }
            }
            kept.iter()
                .enumerate()
                .map(|(slot, &k)| {
                    let old = clean[k].bbox;
                    let s = sum[slot];
                    let raw = if s > 0.0 && s.is_finite() && acc[slot].iter().all(|v| v.is_finite()) {
                        Xyxy::new(
                            (acc[slot][0] / s) as f32,
                            (acc[slot][1] / s) as f32,
                            (acc[slot][2] / s) as f32,
                            (acc[slot][3] / s) as f32,
                        )
                    } else {
                        old
                    };
                    clip_box(raw, fw, fh).unwrap_or(old)
                })
                .collect()
        }
        BoxFusion::Sebf => {
            let boxes: Vec<Xyxy> = clean.iter().map(|d| d.bbox).collect();
            let scores: Vec<f32> = clean.iter().map(|d| d.score).collect();
            let classes: Vec<u32> = clean.iter().map(|d| d.class).collect();
            fuse_boxes_sebf(
                &kept,
                &boxes,
                &scores,
                &classes,
                opts.class_agnostic,
                opts.threshold,
                Some(&owners),
                opts.threshold,
                None,
            )
            .into_iter()
            .zip(&kept)
            .map(|(b, &k)| clip_box(b, fw, fh).unwrap_or(clean[k].bbox))
            .collect()
        }
    };

    let mut out: Vec<Detection> = kept
        .into_iter()
        .zip(fused_boxes)
        .map(|(k, bbox)| Detection { bbox, score: clean[k].score, class: clean[k].class })
        .collect();
    out.truncate(opts.max_det);
    Ok(out)
}

/// Push `item` into the group matching `key` (preserving first-seen group order for deterministic execution).
fn push_grouped<K: PartialEq, V>(groups: &mut Vec<(K, Vec<V>)>, key: K, item: V) {
    if let Some((_, items)) = groups.iter_mut().find(|(k, _)| *k == key) {
        items.push(item);
    } else {
        groups.push((key, vec![item]));
    }
}

/// Distribute one batch's scaled `(pre, inf, post)` wall time evenly across the owning images of each view.
fn distribute_chunk_timing(
    per_img_timing: &mut [[f64; 3]],
    owners: impl Iterator<Item = usize>,
    preds: &[Prediction],
    batch_wall: f64,
) {
    let count = preds.len();
    if count == 0 {
        return;
    }
    let (mut b_pre, mut b_inf, mut b_post) = (0.0f64, 0.0f64, 0.0f64);
    for p in preds {
        b_pre += p.timing.preprocess_ms;
        b_inf += p.timing.inference_ms;
        b_post += p.timing.postprocess_ms;
    }
    let (mut s_pre, mut s_inf, mut s_post) = (0.0f64, 0.0f64, 0.0f64);
    add_scaled_timing(&mut s_pre, &mut s_inf, &mut s_post, b_pre, b_inf, b_post, batch_wall);
    let inv = 1.0 / count as f64;
    let (d_pre, d_inf, d_post) = (s_pre * inv, s_inf * inv, s_post * inv);
    for img_idx in owners {
        per_img_timing[img_idx][0] += d_pre;
        per_img_timing[img_idx][1] += d_inf;
        per_img_timing[img_idx][2] += d_post;
    }
}

impl Predictor {
    fn run_tile_views(&self, views: &[&Image], rect: bool) -> Result<Vec<Prediction>> {
        let imgsz = self.config().imgsz;
        run_views_with_oom_fallback(views, &mut |chunk| self.run_with_imgsz(chunk, rect, imgsz))
    }

    /// Decode an image file and predict it with SAHI tiled inference ([`Predictor::predict_tiled`]).
    pub fn predict_file_tiled(&self, path: impl AsRef<Path>, tile_cfg: &TileConfig) -> Result<Prediction> {
        tile_cfg.validate()?;
        let t = Instant::now();
        let img = folo_data::read_image(path.as_ref())?;
        let decode_ms = ms(t);
        let mut p = self.predict_tiled(&img, tile_cfg)?;
        p.timing =
            Timing::new(p.timing.preprocess_ms + decode_ms, p.timing.inference_ms, p.timing.postprocess_ms);
        Ok(p)
    }

    /// Predict one image with Slicing Aided Hyper Inference (SAHI): slices the image into uniform overlapping
    /// tiles ([`compute_tiles`]), reuses a bounded scratch pool of [`TileConfig::tile_batch`] crop buffers across
    /// chunks (with automatic GPU OOM chunk splitting), optionally runs one full-image pass
    /// ([`TileConfig::full_image`]), and merges all detections in original-image coordinates.
    ///
    /// When the image fits inside a single tile (`width <= tile_width && height <= tile_height`), this degrades
    /// cleanly to [`Predictor::predict`] (with `tile_cfg.max_det` applied) so small images incur zero slicing
    /// overhead.
    pub fn predict_tiled(&self, img: &Image, tile_cfg: &TileConfig) -> Result<Prediction> {
        tile_cfg.validate()?;
        check_image(img)?;
        let tiles = compute_tiles(img.width, img.height, tile_cfg)?;
        if tiles.len() == 1 {
            let mut p = self.predict(img)?;
            p.detections.truncate(tile_cfg.max_det);
            return Ok(p);
        }

        let rect = self.config().rect;
        let mut pre_ms = 0.0f64;
        let mut inf_ms = 0.0f64;
        let mut post_ms = 0.0f64;
        let mut all_dets: Vec<Detection> = Vec::new();
        let cap = tile_cfg.tile_batch.min(tiles.len());
        let mut scratch: Vec<Image> = Vec::with_capacity(cap);

        for chunk in tiles.chunks(tile_cfg.tile_batch) {
            let t_crop = Instant::now();
            if scratch.len() < chunk.len() {
                scratch.resize_with(chunk.len(), || Image { width: 0, height: 0, data: Vec::new() });
            }
            scratch[..chunk.len()]
                .par_iter_mut()
                .zip(chunk.par_iter())
                .for_each(|(dst, &r)| copy_tile_rows(img, r, dst));
            pre_ms += ms(t_crop);

            let refs: Vec<&Image> = scratch[..chunk.len()].iter().collect();
            let t_batch = Instant::now();
            let preds = self.run_tile_views(&refs, rect)?;
            let batch_wall = ms(t_batch);
            let (mut b_pre, mut b_inf, mut b_post) = (0.0f64, 0.0f64, 0.0f64);
            for (pred, tile) in preds.into_iter().zip(chunk) {
                b_pre += pred.timing.preprocess_ms;
                b_inf += pred.timing.inference_ms;
                b_post += pred.timing.postprocess_ms;
                let (ox, oy) = (tile.x as f32, tile.y as f32);
                all_dets.extend(pred.detections.into_iter().map(|d| Detection {
                    bbox: Xyxy::new(d.bbox.x1 + ox, d.bbox.y1 + oy, d.bbox.x2 + ox, d.bbox.y2 + oy),
                    ..d
                }));
            }
            add_scaled_timing(&mut pre_ms, &mut inf_ms, &mut post_ms, b_pre, b_inf, b_post, batch_wall);
        }

        if tile_cfg.full_image {
            let full = self.predict(img)?;
            pre_ms += full.timing.preprocess_ms;
            inf_ms += full.timing.inference_ms;
            post_ms += full.timing.postprocess_ms;
            all_dets.extend(full.detections);
        }

        let t_merge = Instant::now();
        let detections = merge_tiled_detections(all_dets, img.width, img.height, tile_cfg)?;
        post_ms += ms(t_merge);

        Ok(Prediction {
            width: img.width,
            height: img.height,
            detections,
            timing: Timing::new(pre_ms, inf_ms, post_ms),
        })
    }

    /// Predict a batch of images with SAHI tiled inference ([`Predictor::predict_tiled`]), in input order.
    ///
    /// Tiles and optional full-image passes are batched across images that share the same letterbox canvas (in
    /// bounded chunks of [`TileConfig::tile_batch`] with reusable crop scratch buffers and GPU OOM chunk
    /// splitting), and per-image cross-tile merging runs in parallel — eliminating per-image remainder batches
    /// and sequential batch-1 full-image passes while preserving exact CPU parity with [`Predictor::predict_tiled`].
    pub fn predict_batch_tiled(&self, imgs: &[Image], tile_cfg: &TileConfig) -> Result<Vec<Prediction>> {
        tile_cfg.validate()?;
        if imgs.is_empty() {
            return Ok(Vec::new());
        }
        for img in imgs {
            check_image(img)?;
        }
        if imgs.len() == 1 {
            return self.predict_tiled(&imgs[0], tile_cfg).map(|p| vec![p]);
        }

        let n = imgs.len();
        let imgsz = self.config().imgsz;
        let rect = self.config().rect;
        type TileGroup = ([u32; 2], Vec<(usize, TileRect)>);
        let mut is_single_tile = vec![false; n];
        let mut tile_groups: Vec<TileGroup> = Vec::new();
        let mut full_groups: Vec<([u32; 2], Vec<usize>)> = Vec::new();

        for (i, img) in imgs.iter().enumerate() {
            let tiles = compute_tiles(img.width, img.height, tile_cfg)?;
            if tiles.len() == 1 {
                is_single_tile[i] = true;
                let canvas = rect_canvas(img.width, img.height, imgsz, rect);
                push_grouped(&mut full_groups, canvas, i);
            } else {
                let first = tiles[0];
                let tile_canvas = rect_canvas(first.width, first.height, imgsz, rect);
                let bucket = match tile_groups.iter_mut().find(|(k, _)| *k == tile_canvas) {
                    Some((_, v)) => v,
                    None => {
                        tile_groups.push((tile_canvas, Vec::with_capacity(tiles.len())));
                        &mut tile_groups.last_mut().expect("just pushed").1
                    }
                };
                bucket.extend(tiles.into_iter().map(|r| (i, r)));
                if tile_cfg.full_image {
                    let canvas = rect_canvas(img.width, img.height, imgsz, rect);
                    push_grouped(&mut full_groups, canvas, i);
                }
            }
        }

        let mut per_img_dets: Vec<Vec<Detection>> = vec![Vec::new(); n];
        let mut per_img_timing: Vec<[f64; 3]> = vec![[0.0; 3]; n];

        // 1. Batched cropped tiles grouped by shared letterbox canvas, reusing one crop buffer pool.
        let max_tile_chunk =
            tile_groups.iter().map(|(_, jobs)| jobs.len().min(tile_cfg.tile_batch)).max().unwrap_or(0);
        let mut scratch: Vec<Image> = Vec::with_capacity(max_tile_chunk);

        for (_, jobs) in &tile_groups {
            for chunk in jobs.chunks(tile_cfg.tile_batch) {
                let t_crop = Instant::now();
                if scratch.len() < chunk.len() {
                    scratch.resize_with(chunk.len(), || Image { width: 0, height: 0, data: Vec::new() });
                }
                scratch[..chunk.len()]
                    .par_iter_mut()
                    .zip(chunk.par_iter())
                    .for_each(|(dst, &(img_idx, r))| copy_tile_rows(&imgs[img_idx], r, dst));
                let crop_share = ms(t_crop) / chunk.len() as f64;
                for &(img_idx, _) in chunk {
                    per_img_timing[img_idx][0] += crop_share;
                }

                let refs: Vec<&Image> = scratch[..chunk.len()].iter().collect();
                let t_batch = Instant::now();
                let preds = self.run_tile_views(&refs, rect)?;
                let batch_wall = ms(t_batch);
                distribute_chunk_timing(
                    &mut per_img_timing,
                    chunk.iter().map(|&(idx, _)| idx),
                    &preds,
                    batch_wall,
                );
                for (pred, &(img_idx, tile)) in preds.into_iter().zip(chunk) {
                    let (ox, oy) = (tile.x as f32, tile.y as f32);
                    per_img_dets[img_idx].extend(pred.detections.into_iter().map(|d| Detection {
                        bbox: Xyxy::new(d.bbox.x1 + ox, d.bbox.y1 + oy, d.bbox.x2 + ox, d.bbox.y2 + oy),
                        ..d
                    }));
                }
            }
        }
        drop(scratch);

        // 2. Batched full-image passes (both single-tile images and `full_image: true` multi-tile images).
        for (_, idxs) in &full_groups {
            for chunk in idxs.chunks(tile_cfg.tile_batch) {
                let refs: Vec<&Image> = chunk.iter().map(|&idx| &imgs[idx]).collect();
                let t_batch = Instant::now();
                let preds = self.run_tile_views(&refs, rect)?;
                let batch_wall = ms(t_batch);
                distribute_chunk_timing(&mut per_img_timing, chunk.iter().copied(), &preds, batch_wall);
                for (mut pred, &img_idx) in preds.into_iter().zip(chunk) {
                    if is_single_tile[img_idx] {
                        pred.detections.truncate(tile_cfg.max_det);
                        per_img_dets[img_idx] = pred.detections;
                    } else {
                        per_img_dets[img_idx].extend(pred.detections);
                    }
                }
            }
        }

        // 3. Parallel cross-tile merging for multi-tile images; single-tile images pass straight through.
        let t_merge_all = Instant::now();
        let merged: Vec<(Vec<Detection>, f64)> = per_img_dets
            .into_par_iter()
            .enumerate()
            .map(|(i, dets)| -> Result<(Vec<Detection>, f64)> {
                if is_single_tile[i] {
                    return Ok((dets, 0.0));
                }
                let t0 = Instant::now();
                let out = merge_tiled_detections(dets, imgs[i].width, imgs[i].height, tile_cfg)?;
                Ok((out, ms(t0)))
            })
            .collect::<Result<_>>()?;
        let merge_wall = ms(t_merge_all);
        let merge_sum: f64 = merged.iter().map(|(_, m)| *m).sum();
        let merge_scale = if merge_sum > 0.0 { merge_wall / merge_sum } else { 0.0 };

        Ok(imgs
            .iter()
            .zip(merged)
            .zip(per_img_timing)
            .map(|((img, (detections, m_ms)), [pre, inf, post])| Prediction {
                width: img.width,
                height: img.height,
                detections,
                timing: Timing::new(pre, inf, post + m_ms * merge_scale),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_tiles_covers_every_pixel_with_uniform_tiles() {
        let cfg = TileConfig {
            tile_width: 300,
            tile_height: 200,
            overlap_x: 0.2,
            overlap_y: 0.25,
            ..Default::default()
        };
        let tiles = compute_tiles(800, 500, &cfg).unwrap();
        assert!(tiles.len() > 1);
        for t in &tiles {
            assert_eq!((t.width, t.height), (300, 200));
            assert!(t.x + t.width <= 800 && t.y + t.height <= 500);
        }
        // Check 100% pixel coverage on every column and row.
        for x in 0..800u32 {
            assert!(tiles.iter().any(|t| x >= t.x && x < t.x + t.width), "uncovered column {x}");
        }
        for y in 0..500u32 {
            assert!(tiles.iter().any(|t| y >= t.y && y < t.y + t.height), "uncovered row {y}");
        }
        // Small image collapses to a single tile covering the whole image.
        let one = compute_tiles(120, 80, &cfg).unwrap();
        assert_eq!(one, vec![TileRect { x: 0, y: 0, width: 120, height: 80 }]);
    }

    #[test]
    fn crop_tile_copies_exact_pixels_and_rejects_bad_rects() {
        let mut img = Image::new(10, 8, [0, 0, 0]);
        for y in 0..8 {
            for x in 0..10 {
                img.set_pixel(x, y, [x as u8 * 20, y as u8 * 30, (x + y) as u8 * 10]);
            }
        }
        let rect = TileRect { x: 3, y: 2, width: 5, height: 4 };
        let crop = crop_tile(&img, rect).unwrap();
        assert_eq!((crop.width, crop.height), (5, 4));
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(crop.pixel(x, y), img.pixel(rect.x + x, rect.y + y));
            }
        }
        assert!(crop_tile(&img, TileRect { x: 8, y: 0, width: 3, height: 4 }).is_err());
        assert!(crop_tile(&img, TileRect { x: 0, y: 0, width: 0, height: 4 }).is_err());

        // `crop_tile_into` matches `crop_tile` and reuses backing storage across equal-sized tiles.
        let mut scratch = Image { width: 0, height: 0, data: Vec::new() };
        crop_tile_into(&img, rect, &mut scratch).unwrap();
        assert_eq!(scratch, crop);
        let ptr_before = scratch.data.as_ptr();
        let rect2 = TileRect { x: 1, y: 3, width: 5, height: 4 };
        crop_tile_into(&img, rect2, &mut scratch).unwrap();
        assert_eq!(scratch.data.as_ptr(), ptr_before, "equal-size crop_tile_into must reuse buffer");
        assert_eq!(scratch, crop_tile(&img, rect2).unwrap());
        let rect_small = TileRect { x: 0, y: 0, width: 2, height: 2 };
        crop_tile_into(&img, rect_small, &mut scratch).unwrap();
        assert_eq!(scratch, crop_tile(&img, rect_small).unwrap());
        assert!(crop_tile_into(&img, TileRect { x: 8, y: 0, width: 3, height: 4 }, &mut scratch).is_err());
    }

    #[test]
    fn ios_suppresses_truncated_tile_fragment_without_shrinking_full_box() {
        let full = Detection { bbox: Xyxy::new(100.0, 100.0, 300.0, 200.0), score: 0.9, class: 0 };
        // Truncated seam fragment inside the left half of `full` (IoS = 1.0, IoU = 0.4).
        let frag = Detection { bbox: Xyxy::new(100.0, 100.0, 180.0, 200.0), score: 0.8, class: 0 };
        let cfg = TileConfig {
            merge_metric: TileMergeMetric::Ios,
            merge_threshold: 0.5,
            merge_fusion: BoxFusion::Vote,
            ..Default::default()
        };
        let merged = merge_tiled_detections(vec![frag, full], 500, 500, &cfg).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].bbox, full.bbox, "truncated fragment must not pull full box inward");

        // Under IoU with threshold 0.5, IoU=0.4 does not suppress the fragment.
        let iou_cfg = TileConfig { merge_metric: TileMergeMetric::Iou, ..cfg };
        let kept_both = merge_tiled_detections(vec![frag, full], 500, 500, &iou_cfg).unwrap();
        assert_eq!(kept_both.len(), 2);
    }

    #[test]
    fn vote_and_sebf_fuse_agreeing_overlapping_tile_detections_deterministically() {
        let d0 = Detection { bbox: Xyxy::new(50.0, 50.0, 100.0, 100.0), score: 0.9, class: 1 };
        let d1 = Detection { bbox: Xyxy::new(54.0, 52.0, 104.0, 102.0), score: 0.6, class: 1 };
        let other_cls = Detection { bbox: Xyxy::new(51.0, 51.0, 101.0, 101.0), score: 0.85, class: 2 };

        for fusion in [BoxFusion::Vote, BoxFusion::Sebf] {
            let cfg = TileConfig { merge_fusion: fusion, merge_threshold: 0.5, ..Default::default() };
            let a = merge_tiled_detections(vec![d0, d1, other_cls], 200, 200, &cfg).unwrap();
            let b = merge_tiled_detections(vec![other_cls, d1, d0], 200, 200, &cfg).unwrap();
            assert_eq!(a, b, "merge must not depend on input order ({fusion:?})");
            assert_eq!(a.len(), 2, "class-aware merge keeps class 1 and class 2 separate");
            // Fused class-1 box lies strictly between d0 and d1.
            assert!(a[0].bbox.x1 > 50.0 && a[0].bbox.x1 < 54.0, "{:?}", a[0].bbox);
        }

        // Class-agnostic merge collapses class 2 into the higher-scoring class 1 survivor.
        let agnostic =
            TileConfig { class_agnostic: true, merge_fusion: BoxFusion::None, ..Default::default() };
        let res = merge_tiled_detections(vec![d0, other_cls], 200, 200, &agnostic).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].class, 1);
    }

    #[test]
    fn sebf_fuses_later_clusters_after_suppressed_candidates_without_contamination() {
        // Cluster 0: indices 0 (survivor, kept[0]=0) and 1 (suppressed).
        // Cluster 1: indices 2 (survivor, kept[1]=2 != 1) and 3 (suppressed).
        // Cluster 2: indices 4 (survivor, kept[2]=4) and 5 (suppressed); slot 2 collides numerically with
        // cluster 1's candidate index 2, verifying zero cross-cluster contamination.
        let dets = vec![
            Detection { bbox: Xyxy::new(10.0, 10.0, 50.0, 50.0), score: 0.95, class: 0 },
            Detection { bbox: Xyxy::new(14.0, 12.0, 54.0, 52.0), score: 0.85, class: 0 },
            Detection { bbox: Xyxy::new(100.0, 10.0, 140.0, 50.0), score: 0.80, class: 0 },
            Detection { bbox: Xyxy::new(104.0, 12.0, 144.0, 52.0), score: 0.70, class: 0 },
            Detection { bbox: Xyxy::new(100.0, 12.0, 140.0, 52.0), score: 0.65, class: 1 },
            Detection { bbox: Xyxy::new(104.0, 14.0, 144.0, 54.0), score: 0.60, class: 1 },
        ];
        let cfg = TileConfig {
            merge_metric: TileMergeMetric::Iou,
            merge_threshold: 0.5,
            merge_fusion: BoxFusion::Sebf,
            class_agnostic: false,
            ..Default::default()
        };
        let merged = merge_tiled_detections(dets, 200, 200, &cfg).unwrap();
        assert_eq!(merged.len(), 3);
        assert!(merged[0].bbox.x1 > 10.0 && merged[0].bbox.x1 < 14.0, "cluster 0: {:?}", merged[0].bbox);
        assert!(merged[1].bbox.x1 > 100.0 && merged[1].bbox.x1 < 104.0, "cluster 1: {:?}", merged[1].bbox);
        assert!(merged[2].bbox.x1 > 100.0 && merged[2].bbox.x1 < 104.0, "cluster 2: {:?}", merged[2].bbox);
        assert!(merged[2].bbox.y1 > 12.0 && merged[2].bbox.y1 < 14.0, "cluster 2 y1: {:?}", merged[2].bbox);
    }

    #[test]
    fn config_validation_rejects_invalid_tile_parameters() {
        assert!(TileConfig { tile_width: 0, ..Default::default() }.validate().is_err());
        assert!(TileConfig { tile_height: 0, ..Default::default() }.validate().is_err());
        assert!(TileConfig { overlap_x: -0.1, ..Default::default() }.validate().is_err());
        assert!(TileConfig { overlap_x: 1.0, ..Default::default() }.validate().is_err());
        assert!(TileConfig { overlap_y: f32::NAN, ..Default::default() }.validate().is_err());
        assert!(TileConfig { tile_batch: 0, ..Default::default() }.validate().is_err());
        assert!(TileConfig { merge_threshold: 1.1, ..Default::default() }.validate().is_err());
        assert!(TileConfig { merge_threshold: f32::NAN, ..Default::default() }.validate().is_err());
        assert!(TileConfig { max_det: 0, ..Default::default() }.validate().is_err());
    }
}
