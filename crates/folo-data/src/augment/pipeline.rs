//! The training transform: mosaic → affine → mixup → HSV → flips, ending at `imgsz x imgsz`.

use super::geometry::{long_side_dims, random_affine_matrix, transform_boxes_in_place};
use super::hsv::{HsvTables, random_hsv_tables};
use super::mosaic::{mixup_weight, mosaic_plane};
use super::{AugmentConfig, flip_lr, flip_ud, mixup, mosaic_affine, random_hsv};
use crate::bbox::BBox;
use crate::buffers::{self, scratch_u8};
use crate::dataset::Dataset;
use crate::error::Result;
use crate::image::{Image, Sample};
use crate::letterbox::{Letterbox, LetterboxOptions, PAD_VALUE, letterbox_sample};
use crate::ram_cache::Entry as RamEntry;
use crate::rng::Rng;
use crate::simd::hsv::rgb_row_to_planar_fused;
use crate::simd::planar::mixup_blend_u8_into;
use crate::simd::{PlanarDst, for_planar_bands};
use crate::warp::{Plane, PlaneTiles, Tile, warp_plane, warp_plane_fused};

/// A sample as the transform's `fetch` delivers it.
#[derive(Clone, Debug)]
pub struct Fetched {
    /// Dataset index of the sample actually loaded.
    pub index: usize,
    /// The sample: as decoded, or already resized so its long side is the transform's `imgsz` (what
    /// [`resize_long_side`] makes of it, e.g. from the loader's RAM cache).
    pub sample: Sample,
    /// Size `[width, height]` of the image as decoded (before any resize).
    pub orig_size: [u32; 2],
}

impl Fetched {
    /// A sample as decoded.
    pub fn decoded(index: usize, sample: Sample) -> Self {
        let orig_size = [sample.image.width, sample.image.height];
        Fetched { index, sample, orig_size }
    }
}

/// Internal fetched sample: either owned (freshly decoded, pixel buffer recycled after use) or a shared
/// read-only entry from [`crate::ram_cache::RamCache`] (zero-copy, never recycled).
pub(crate) enum FetchedSample {
    Owned(Fetched),
    Shared { index: usize, entry: RamEntry },
}

impl FetchedSample {
    pub(crate) fn decoded(index: usize, sample: Sample) -> Self {
        FetchedSample::Owned(Fetched::decoded(index, sample))
    }

    fn index(&self) -> usize {
        match self {
            FetchedSample::Owned(f) => f.index,
            FetchedSample::Shared { index, .. } => *index,
        }
    }

    fn sample(&self) -> &Sample {
        match self {
            FetchedSample::Owned(f) => &f.sample,
            FetchedSample::Shared { entry, .. } => &entry.0,
        }
    }

    fn orig_size(&self) -> [u32; 2] {
        match self {
            FetchedSample::Owned(f) => f.orig_size,
            FetchedSample::Shared { entry, .. } => entry.1,
        }
    }

    /// Resizes `self` so its long side is `long` pixels without copying when already that size (the common
    /// `--cache ram` case).
    fn into_resized_long_side(self, long: u32) -> Self {
        let s = self.sample();
        let (w, h) = (s.image.width, s.image.height);
        let (nw, nh) = long_side_dims(w, h, long);
        if (nw, nh) == (w, h) {
            return self;
        }
        match self {
            FetchedSample::Owned(f) => {
                let image = crate::resize::resize(&f.sample.image, nw, nh);
                buffers::recycle(f.sample.image.data);
                FetchedSample::Owned(Fetched {
                    index: f.index,
                    sample: Sample { image, boxes: f.sample.boxes },
                    orig_size: f.orig_size,
                })
            }
            FetchedSample::Shared { index, entry } => {
                let image = crate::resize::resize(&entry.0.image, nw, nh);
                FetchedSample::Owned(Fetched {
                    index,
                    sample: Sample { image, boxes: entry.0.boxes.clone() },
                    orig_size: entry.1,
                })
            }
        }
    }

    /// Returns an owned pixel buffer to the thread-local scratch pool; a shared cache entry is only unreferenced.
    fn recycle(self) {
        if let FetchedSample::Owned(f) = self {
            buffers::recycle(f.sample.image.data);
        }
    }
}

/// Letterboxes `f` to `s x s` exactly as its original image would be: the pixels of a pre-resized sample are those
/// of its original (the resize to the same size is a copy), and its boxes (normalised, so unchanged by the resize)
/// are mapped with the geometry of the original size, as the letterbox of the original maps them.
fn letterbox_fetched(f: &FetchedSample, s: u32) -> (Sample, Letterbox) {
    let opts = LetterboxOptions::default();
    let src = f.sample();
    let (mut sample, lb) = letterbox_sample(src, s, s, &opts);
    if [src.image.width, src.image.height] == f.orig_size() {
        return (sample, lb);
    }
    let [w, h] = f.orig_size();
    let orig = Letterbox::compute(w, h, s, s, &opts);
    debug_assert_eq!((orig.resized, orig.out, orig.pad), (lb.resized, lb.out, lb.pad));
    sample.boxes = src.boxes.iter().map(|b| orig.map_box(b)).collect();
    (sample, orig)
}

/// Metadata and transformed boxes of a sample written straight into a planar batch buffer by
/// [`TrainTransform::apply_into_planar`].
pub(crate) struct ProducedMeta {
    pub index: usize,
    pub orig_size: [u32; 2],
    pub letterbox: Letterbox,
    pub boxes: Vec<BBox>,
}

/// Fuses `mixup + HSV + flip_ud + flip_lr + interleaved RGB → planar` row by row in L1 scratch, eliminating the
/// intermediate full-image blend write and re-read.
#[allow(clippy::too_many_arguments)]
fn mixup_to_planar_fused(
    a: &Image,
    b: &Image,
    wa: u32,
    hsv: Option<&HsvTables>,
    flip_ud: bool,
    flip_lr: bool,
    dst: &mut PlanarDst<'_>,
) {
    assert_eq!((a.width, a.height), (b.width, b.height), "mixup needs images of the same size");
    let (w, h) = (a.width as usize, a.height as usize);
    let stride = a.stride();
    for_planar_bands(dst, w, h, |dy0, band| {
        let mut row_buf = if wa == 0 || wa == 256 { buffers::ScratchU8::empty() } else { scratch_u8(stride) };
        let rows = band.rows(w);
        for bi in 0..rows {
            let src_y = if flip_ud { h - 1 - (dy0 + bi) } else { dy0 + bi };
            let off = src_y * stride;
            let row = if wa == 256 {
                &a.data[off..off + stride]
            } else if wa == 0 {
                &b.data[off..off + stride]
            } else {
                mixup_blend_u8_into(&a.data[off..off + stride], &b.data[off..off + stride], wa, &mut row_buf);
                &row_buf
            };
            let mut drow = band.row_slices(bi, w);
            rgb_row_to_planar_fused(row, hsv, flip_lr, &mut drow);
        }
    });
}

fn flip_boxes(boxes: &mut [BBox], do_ud: bool, do_lr: bool) {
    if do_ud {
        for b in boxes.iter_mut() {
            b.cy = 1.0 - b.cy;
        }
    }
    if do_lr {
        for b in boxes.iter_mut() {
            b.cx = 1.0 - b.cx;
        }
    }
}

/// A transformed sample plus the letterbox geometry of its primary image.
#[derive(Clone, Debug)]
pub struct Transformed {
    /// Dataset index of the primary image actually used (differs from the requested one when the loader
    /// substituted an unreadable image).
    pub index: usize,
    /// The `imgsz x imgsz` result.
    pub sample: Sample,
    /// Size `[width, height]` of the primary source image as decoded (also when it came pre-resized).
    pub orig_size: [u32; 2],
    /// The letterbox step applied to the primary image. Geometric augmentation applied afterwards (mosaic,
    /// affine, flips) is not part of it, so it maps predictions back only for unaugmented (validation) samples.
    pub letterbox: Letterbox,
}

/// YOLO-style training (or validation) transform producing square `imgsz` images.
#[derive(Clone, Copy, Debug)]
pub struct TrainTransform {
    /// Output side in pixels.
    pub imgsz: u32,
    /// Augmentation parameters; `None` means validation: letterbox only.
    pub augment: Option<AugmentConfig>,
    /// Whether mosaic (and mixup) may run; see [`AugmentConfig::mosaic_active`].
    pub mosaic: bool,
}

impl TrainTransform {
    /// A training transform for `epoch` of `epochs` (0 = unknown length), honouring `close_mosaic`.
    pub fn train(imgsz: u32, augment: AugmentConfig, epoch: u32, epochs: u32) -> Self {
        TrainTransform { imgsz, augment: Some(augment), mosaic: augment.mosaic_active(epoch, epochs) }
    }

    /// A validation transform: letterbox to `imgsz`, nothing random.
    pub fn val(imgsz: u32) -> Self {
        TrainTransform { imgsz, augment: None, mosaic: false }
    }

    /// Produces the transformed sample for `index` of `ds`.
    pub fn apply_dataset(&self, ds: &dyn Dataset, index: usize, rng: &mut Rng) -> Result<Transformed> {
        self.apply(index, ds.len(), &|i| ds.get(i).map(|s| Fetched::decoded(i, s)), rng)
    }

    /// Produces the transformed sample for dataset `index`. `fetch(i)` loads sample `i` of a dataset with `len`
    /// samples, with the index it actually loaded (the loader passes a fetch that substitutes unreadable images
    /// deterministically) and its decoded size; the sample may come resized to long side `imgsz` (the loader's
    /// RAM cache), which gives the same result.
    pub fn apply<F>(&self, index: usize, len: usize, fetch: &F, rng: &mut Rng) -> Result<Transformed>
    where
        F: Fn(usize) -> Result<Fetched>,
    {
        let fetch_owned = |i| fetch(i).map(FetchedSample::Owned);
        let Some(cfg) = &self.augment else {
            let f = fetch_owned(index)?;
            let (sample, letterbox) = letterbox_fetched(&f, self.imgsz);
            let (out_index, orig_size) = (f.index(), f.orig_size());
            f.recycle();
            return Ok(Transformed { index: out_index, sample, orig_size, letterbox });
        };
        let mut out = self.one(cfg, index, len, &fetch_owned, rng)?;
        if self.mosaic && rng.chance(cfg.mixup) {
            let other = self.one(cfg, rng.below(len), len, &fetch_owned, rng)?;
            mixup(&mut out.sample, &other.sample, rng);
            buffers::recycle(other.sample.image.data);
        }
        random_hsv(&mut out.sample.image, cfg.hsv, rng);
        if rng.chance(cfg.flipud) {
            flip_ud(&mut out.sample);
        }
        if rng.chance(cfg.fliplr) {
            flip_lr(&mut out.sample);
        }
        Ok(out)
    }

    /// Produces the transformed sample straight into planar `dst` (`u8` or `f32 / 255.0`), fusing warp (or letterbox),
    /// HSV, flips and planar conversion in L1 cache without allocating an intermediate `imgsz x imgsz` HWC image.
    /// Bit for bit identical to [`TrainTransform::apply`] followed by planar conversion.
    pub(crate) fn apply_into_planar<F>(
        &self,
        index: usize,
        len: usize,
        fetch: &F,
        rng: &mut Rng,
        dst: &mut PlanarDst<'_>,
    ) -> Result<ProducedMeta>
    where
        F: Fn(usize) -> Result<FetchedSample>,
    {
        let s = self.imgsz;
        let Some(cfg) = &self.augment else {
            let f = fetch(index)?;
            let opts = LetterboxOptions::default();
            let src = f.sample();
            let lb_img = Letterbox::compute(src.image.width, src.image.height, s, s, &opts);
            lb_img.apply_planar_dst(&src.image, PAD_VALUE, dst);
            let orig_size = f.orig_size();
            let (boxes, letterbox) = if [src.image.width, src.image.height] == orig_size {
                (src.boxes.iter().map(|b| lb_img.map_box(b)).collect(), lb_img)
            } else {
                let orig = Letterbox::compute(orig_size[0], orig_size[1], s, s, &opts);
                debug_assert_eq!(
                    (orig.resized, orig.out, orig.pad),
                    (lb_img.resized, lb_img.out, lb_img.pad)
                );
                (src.boxes.iter().map(|b| orig.map_box(b)).collect(), orig)
            };
            let out_index = f.index();
            f.recycle();
            return Ok(ProducedMeta { index: out_index, orig_size, letterbox, boxes });
        };

        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, rng.below(len), rng.below(len), rng.below(len)];
            let p0 = fetch(ids[0])?;
            let (primary, orig_size) = (p0.index(), p0.orig_size());
            let parts = [
                p0.into_resized_long_side(s),
                fetch(ids[1])?.into_resized_long_side(s),
                fetch(ids[2])?.into_resized_long_side(s),
                fetch(ids[3])?.into_resized_long_side(s),
            ];
            let (plane, mut boxes) = mosaic_plane(
                [parts[0].sample(), parts[1].sample(), parts[2].sample(), parts[3].sample()],
                s,
                rng,
            );
            let m = random_affine_matrix(plane.width, plane.height, s, s, cfg, rng);
            let is_id = m.is_identity() && (plane.width, plane.height) == (s, s);
            if !is_id {
                transform_boxes_in_place(&mut boxes, plane.width, plane.height, &m, s, s, &cfg.box_filter());
            }
            let letterbox = Letterbox::identity(s, s);
            if self.mosaic && rng.chance(cfg.mixup) {
                let img = if is_id { plane.render() } else { warp_plane(&plane, &m, s, s) };
                for p in parts {
                    p.recycle();
                }
                return self.finish_mixup_into_planar(
                    cfg, primary, orig_size, letterbox, img, boxes, len, fetch, rng, dst,
                );
            }
            let hsv = random_hsv_tables(cfg.hsv, rng);
            let do_ud = rng.chance(cfg.flipud);
            let do_lr = rng.chance(cfg.fliplr);
            flip_boxes(&mut boxes, do_ud, do_lr);
            warp_plane_fused(&plane, &m, s, s, hsv.as_ref(), do_ud, do_lr, dst);
            for p in parts {
                p.recycle();
            }
            Ok(ProducedMeta { index: primary, orig_size, letterbox, boxes })
        } else {
            let f = fetch(index)?;
            let opts = LetterboxOptions::default();
            let src = f.sample();
            let orig_size = f.orig_size();
            let out_index = f.index();
            let lb_img = Letterbox::compute(src.image.width, src.image.height, s, s, &opts);
            let (mut boxes, letterbox): (Vec<BBox>, _) = if [src.image.width, src.image.height] == orig_size {
                (src.boxes.iter().map(|b| lb_img.map_box(b)).collect(), lb_img)
            } else {
                let orig = Letterbox::compute(orig_size[0], orig_size[1], s, s, &opts);
                debug_assert_eq!(
                    (orig.resized, orig.out, orig.pad),
                    (lb_img.resized, lb_img.out, lb_img.pad)
                );
                (src.boxes.iter().map(|b| orig.map_box(b)).collect(), orig)
            };
            let [rw, rh] = lb_img.resized;
            let [left, top] = lb_img.pad.map(|v| v as i64);
            let resized_opt = if (src.image.width, src.image.height) == (rw, rh) {
                None
            } else {
                Some(crate::resize::resize(&src.image, rw, rh))
            };
            let tile_img = resized_opt.as_ref().unwrap_or(&src.image);
            let tile =
                Tile { img: tile_img, dst: [left, top, left + rw as i64, top + rh as i64], src: [0, 0] };
            let plane = Plane { width: s, height: s, fill: [PAD_VALUE; 3], tiles: PlaneTiles::single(tile) };
            let m = random_affine_matrix(s, s, s, s, cfg, rng);
            let is_id = m.is_identity();
            if !is_id {
                transform_boxes_in_place(&mut boxes, s, s, &m, s, s, &cfg.box_filter());
            }
            if self.mosaic && rng.chance(cfg.mixup) {
                let img = if is_id { plane.render() } else { warp_plane(&plane, &m, s, s) };
                if let Some(r) = resized_opt {
                    buffers::recycle(r.data);
                }
                f.recycle();
                return self.finish_mixup_into_planar(
                    cfg, out_index, orig_size, letterbox, img, boxes, len, fetch, rng, dst,
                );
            }
            let hsv = random_hsv_tables(cfg.hsv, rng);
            let do_ud = rng.chance(cfg.flipud);
            let do_lr = rng.chance(cfg.fliplr);
            flip_boxes(&mut boxes, do_ud, do_lr);
            warp_plane_fused(&plane, &m, s, s, hsv.as_ref(), do_ud, do_lr, dst);
            if let Some(r) = resized_opt {
                buffers::recycle(r.data);
            }
            f.recycle();
            Ok(ProducedMeta { index: out_index, orig_size, letterbox, boxes })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_mixup_into_planar<F>(
        &self,
        cfg: &AugmentConfig,
        index: usize,
        orig_size: [u32; 2],
        letterbox: Letterbox,
        img: Image,
        mut boxes: Vec<BBox>,
        len: usize,
        fetch: &F,
        rng: &mut Rng,
        dst: &mut PlanarDst<'_>,
    ) -> Result<ProducedMeta>
    where
        F: Fn(usize) -> Result<FetchedSample>,
    {
        let other = match self.one(cfg, rng.below(len), len, fetch, rng) {
            Ok(o) => o,
            Err(e) => {
                buffers::recycle(img.data);
                return Err(e);
            }
        };
        let wa = mixup_weight(rng);
        boxes.extend_from_slice(&other.sample.boxes);
        let hsv = random_hsv_tables(cfg.hsv, rng);
        let do_ud = rng.chance(cfg.flipud);
        let do_lr = rng.chance(cfg.fliplr);
        flip_boxes(&mut boxes, do_ud, do_lr);
        mixup_to_planar_fused(&img, &other.sample.image, wa, hsv.as_ref(), do_ud, do_lr, dst);
        buffers::recycle(other.sample.image.data);
        buffers::recycle(img.data);
        Ok(ProducedMeta { index, orig_size, letterbox, boxes })
    }

    fn one<F>(
        &self,
        cfg: &AugmentConfig,
        index: usize,
        len: usize,
        fetch: &F,
        rng: &mut Rng,
    ) -> Result<Transformed>
    where
        F: Fn(usize) -> Result<FetchedSample>,
    {
        let s = self.imgsz;
        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, rng.below(len), rng.below(len), rng.below(len)];
            let p0 = fetch(ids[0])?;
            let (primary, orig_size) = (p0.index(), p0.orig_size());
            // The identity for a pre-resized sample.
            let parts = [
                p0.into_resized_long_side(s),
                fetch(ids[1])?.into_resized_long_side(s),
                fetch(ids[2])?.into_resized_long_side(s),
                fetch(ids[3])?.into_resized_long_side(s),
            ];
            let sample = mosaic_affine(
                [parts[0].sample(), parts[1].sample(), parts[2].sample(), parts[3].sample()],
                s,
                cfg,
                rng,
            );
            for p in parts {
                p.recycle();
            }
            Ok(Transformed { index: primary, sample, orig_size, letterbox: Letterbox::identity(s, s) })
        } else {
            let f = fetch(index)?;
            let opts = LetterboxOptions::default();
            let src = f.sample();
            let orig_size = f.orig_size();
            let out_index = f.index();
            let lb_img = Letterbox::compute(src.image.width, src.image.height, s, s, &opts);
            let (mut boxes, letterbox): (Vec<BBox>, _) = if [src.image.width, src.image.height] == orig_size {
                (src.boxes.iter().map(|b| lb_img.map_box(b)).collect(), lb_img)
            } else {
                let orig = Letterbox::compute(orig_size[0], orig_size[1], s, s, &opts);
                debug_assert_eq!(
                    (orig.resized, orig.out, orig.pad),
                    (lb_img.resized, lb_img.out, lb_img.pad)
                );
                (src.boxes.iter().map(|b| orig.map_box(b)).collect(), orig)
            };
            let [rw, rh] = lb_img.resized;
            let [left, top] = lb_img.pad.map(|v| v as i64);
            let resized_opt = if (src.image.width, src.image.height) == (rw, rh) {
                None
            } else {
                Some(crate::resize::resize(&src.image, rw, rh))
            };
            let tile_img = resized_opt.as_ref().unwrap_or(&src.image);
            let tile =
                Tile { img: tile_img, dst: [left, top, left + rw as i64, top + rh as i64], src: [0, 0] };
            let plane = Plane { width: s, height: s, fill: [PAD_VALUE; 3], tiles: PlaneTiles::single(tile) };
            let m = random_affine_matrix(s, s, s, s, cfg, rng);
            let image = if m.is_identity() {
                plane.render()
            } else {
                transform_boxes_in_place(&mut boxes, s, s, &m, s, s, &cfg.box_filter());
                warp_plane(&plane, &m, s, s)
            };
            if let Some(r) = resized_opt {
                buffers::recycle(r.data);
            }
            f.recycle();
            Ok(Transformed { index: out_index, sample: Sample { image, boxes }, orig_size, letterbox })
        }
    }
}
