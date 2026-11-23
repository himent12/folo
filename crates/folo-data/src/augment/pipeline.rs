//! The training transform: mosaic → affine → mixup → HSV → flips, ending at `imgsz x imgsz`.

use super::geometry::{random_affine_matrix, transform_boxes};
use super::hsv::{HsvTables, random_hsv_tables};
use super::mosaic::mosaic_plane;
use super::{
    AugmentConfig, flip_lr, flip_ud, mixup, mosaic_affine, random_affine, random_hsv, resize_long_side,
};
use crate::bbox::BBox;
use crate::buffers;
use crate::dataset::Dataset;
use crate::error::Result;
use crate::image::{Image, Sample};
use crate::letterbox::{Letterbox, LetterboxOptions, PAD_VALUE, letterbox_sample};
use crate::rng::Rng;
use crate::simd::hsv_fused::rgb_row_to_planar_fused;
use crate::simd::{PlanarDst, for_planar_bands};
use crate::warp::{Plane, Tile, warp_plane, warp_plane_fused};

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

/// Letterboxes `f` to `s x s` exactly as its original image would be: the pixels of a pre-resized sample are those
/// of its original (the resize to the same size is a copy), and its boxes (normalised, so unchanged by the resize)
/// are mapped with the geometry of the original size, as the letterbox of the original maps them.
fn letterbox_fetched(f: &Fetched, s: u32) -> (Sample, Letterbox) {
    let opts = LetterboxOptions::default();
    let (mut sample, lb) = letterbox_sample(&f.sample, s, s, &opts);
    if [f.sample.image.width, f.sample.image.height] == f.orig_size {
        return (sample, lb);
    }
    let [w, h] = f.orig_size;
    let orig = Letterbox::compute(w, h, s, s, &opts);
    debug_assert_eq!((orig.resized, orig.out, orig.pad), (lb.resized, lb.out, lb.pad));
    sample.boxes = f.sample.boxes.iter().map(|b| orig.map_box(b)).collect();
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

/// Fuses `HSV + flip_ud + flip_lr + interleaved RGB → planar` over an existing HWC image (the mixup tail path).
fn image_to_planar_fused(
    img: &Image,
    hsv: Option<&HsvTables>,
    flip_ud: bool,
    flip_lr: bool,
    dst: &mut PlanarDst<'_>,
) {
    let (w, h) = (img.width as usize, img.height as usize);
    let stride = img.stride();
    for_planar_bands(dst, w, h, |dy0, band| {
        let rows = band.rows(w);
        for bi in 0..rows {
            let src_y = if flip_ud { h - 1 - (dy0 + bi) } else { dy0 + bi };
            let mut drow = band.row_slices(bi, w);
            rgb_row_to_planar_fused(&img.data[src_y * stride..(src_y + 1) * stride], hsv, flip_lr, &mut drow);
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
        let Some(cfg) = &self.augment else {
            let f = fetch(index)?;
            let (sample, letterbox) = letterbox_fetched(&f, self.imgsz);
            buffers::recycle(f.sample.image.data);
            return Ok(Transformed { index: f.index, sample, orig_size: f.orig_size, letterbox });
        };
        let mut out = self.one(cfg, index, len, fetch, rng)?;
        if self.mosaic && rng.chance(cfg.mixup) {
            let other = self.one(cfg, rng.below(len), len, fetch, rng)?;
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
        F: Fn(usize) -> Result<Fetched>,
    {
        let s = self.imgsz;
        let Some(cfg) = &self.augment else {
            let f = fetch(index)?;
            let opts = LetterboxOptions::default();
            let lb_img = Letterbox::compute(f.sample.image.width, f.sample.image.height, s, s, &opts);
            lb_img.apply_planar_dst(&f.sample.image, PAD_VALUE, dst);
            let (boxes, letterbox) = if [f.sample.image.width, f.sample.image.height] == f.orig_size {
                (f.sample.boxes.iter().map(|b| lb_img.map_box(b)).collect(), lb_img)
            } else {
                let orig = Letterbox::compute(f.orig_size[0], f.orig_size[1], s, s, &opts);
                debug_assert_eq!(
                    (orig.resized, orig.out, orig.pad),
                    (lb_img.resized, lb_img.out, lb_img.pad)
                );
                (f.sample.boxes.iter().map(|b| orig.map_box(b)).collect(), orig)
            };
            buffers::recycle(f.sample.image.data);
            return Ok(ProducedMeta { index: f.index, orig_size: f.orig_size, letterbox, boxes });
        };

        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, rng.below(len), rng.below(len), rng.below(len)];
            let mut parts = Vec::with_capacity(4);
            let (mut primary, mut orig_size) = (index, [0, 0]);
            for (k, &i) in ids.iter().enumerate() {
                let f = fetch(i)?;
                if k == 0 {
                    (primary, orig_size) = (f.index, f.orig_size);
                }
                parts.push(resize_long_side(f.sample, s));
            }
            let (plane, raw_boxes) = mosaic_plane([&parts[0], &parts[1], &parts[2], &parts[3]], s, rng);
            let m = random_affine_matrix(plane.width, plane.height, s, s, cfg, rng);
            let mut boxes = if m.is_identity() && (plane.width, plane.height) == (s, s) {
                raw_boxes
            } else {
                transform_boxes(&raw_boxes, plane.width, plane.height, &m, s, s, &cfg.box_filter())
            };
            let letterbox = Letterbox::identity(s, s);
            if self.mosaic && rng.chance(cfg.mixup) {
                let img = if m.is_identity() && (plane.width, plane.height) == (s, s) {
                    plane.render()
                } else {
                    warp_plane(&plane, &m, s, s)
                };
                drop(plane);
                for p in parts {
                    buffers::recycle(p.image.data);
                }
                let mut sample = Sample { image: img, boxes };
                let other = self.one(cfg, rng.below(len), len, fetch, rng)?;
                mixup(&mut sample, &other.sample, rng);
                buffers::recycle(other.sample.image.data);
                let hsv = random_hsv_tables(cfg.hsv, rng);
                let do_ud = rng.chance(cfg.flipud);
                let do_lr = rng.chance(cfg.fliplr);
                flip_boxes(&mut sample.boxes, do_ud, do_lr);
                image_to_planar_fused(&sample.image, hsv.as_ref(), do_ud, do_lr, dst);
                buffers::recycle(sample.image.data);
                return Ok(ProducedMeta { index: primary, orig_size, letterbox, boxes: sample.boxes });
            }
            let hsv = random_hsv_tables(cfg.hsv, rng);
            let do_ud = rng.chance(cfg.flipud);
            let do_lr = rng.chance(cfg.fliplr);
            flip_boxes(&mut boxes, do_ud, do_lr);
            warp_plane_fused(&plane, &m, s, s, hsv.as_ref(), do_ud, do_lr, dst);
            drop(plane);
            for p in parts {
                buffers::recycle(p.image.data);
            }
            Ok(ProducedMeta { index: primary, orig_size, letterbox, boxes })
        } else {
            let f = fetch(index)?;
            let opts = LetterboxOptions::default();
            let lb_img = Letterbox::compute(f.sample.image.width, f.sample.image.height, s, s, &opts);
            let (lb_boxes, letterbox): (Vec<BBox>, _) =
                if [f.sample.image.width, f.sample.image.height] == f.orig_size {
                    (f.sample.boxes.iter().map(|b| lb_img.map_box(b)).collect(), lb_img)
                } else {
                    let orig = Letterbox::compute(f.orig_size[0], f.orig_size[1], s, s, &opts);
                    debug_assert_eq!(
                        (orig.resized, orig.out, orig.pad),
                        (lb_img.resized, lb_img.out, lb_img.pad)
                    );
                    (f.sample.boxes.iter().map(|b| orig.map_box(b)).collect(), orig)
                };
            let [rw, rh] = lb_img.resized;
            let [left, top] = lb_img.pad.map(|v| v as i64);
            let resized_opt = if (f.sample.image.width, f.sample.image.height) == (rw, rh) {
                None
            } else {
                Some(crate::resize::resize(&f.sample.image, rw, rh))
            };
            let tile_img = resized_opt.as_ref().unwrap_or(&f.sample.image);
            let tile =
                Tile { img: tile_img, dst: [left, top, left + rw as i64, top + rh as i64], src: [0, 0] };
            let plane = Plane { width: s, height: s, fill: [PAD_VALUE; 3], tiles: vec![tile] };
            let m = random_affine_matrix(s, s, s, s, cfg, rng);
            let mut boxes = if m.is_identity() {
                lb_boxes
            } else {
                transform_boxes(&lb_boxes, s, s, &m, s, s, &cfg.box_filter())
            };
            if self.mosaic && rng.chance(cfg.mixup) {
                let img = if m.is_identity() { plane.render() } else { warp_plane(&plane, &m, s, s) };
                drop(plane);
                if let Some(r) = resized_opt {
                    buffers::recycle(r.data);
                }
                buffers::recycle(f.sample.image.data);
                let mut sample = Sample { image: img, boxes };
                let other = self.one(cfg, rng.below(len), len, fetch, rng)?;
                mixup(&mut sample, &other.sample, rng);
                buffers::recycle(other.sample.image.data);
                let hsv = random_hsv_tables(cfg.hsv, rng);
                let do_ud = rng.chance(cfg.flipud);
                let do_lr = rng.chance(cfg.fliplr);
                flip_boxes(&mut sample.boxes, do_ud, do_lr);
                image_to_planar_fused(&sample.image, hsv.as_ref(), do_ud, do_lr, dst);
                buffers::recycle(sample.image.data);
                return Ok(ProducedMeta {
                    index: f.index,
                    orig_size: f.orig_size,
                    letterbox,
                    boxes: sample.boxes,
                });
            }
            let hsv = random_hsv_tables(cfg.hsv, rng);
            let do_ud = rng.chance(cfg.flipud);
            let do_lr = rng.chance(cfg.fliplr);
            flip_boxes(&mut boxes, do_ud, do_lr);
            warp_plane_fused(&plane, &m, s, s, hsv.as_ref(), do_ud, do_lr, dst);
            drop(plane);
            if let Some(r) = resized_opt {
                buffers::recycle(r.data);
            }
            buffers::recycle(f.sample.image.data);
            Ok(ProducedMeta { index: f.index, orig_size: f.orig_size, letterbox, boxes })
        }
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
        F: Fn(usize) -> Result<Fetched>,
    {
        let s = self.imgsz;
        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, rng.below(len), rng.below(len), rng.below(len)];
            let mut parts = Vec::with_capacity(4);
            let (mut primary, mut orig_size) = (index, [0, 0]);
            for (k, &i) in ids.iter().enumerate() {
                let f = fetch(i)?;
                if k == 0 {
                    (primary, orig_size) = (f.index, f.orig_size);
                }
                // The identity for a pre-resized sample.
                parts.push(resize_long_side(f.sample, s));
            }
            let sample = mosaic_affine([&parts[0], &parts[1], &parts[2], &parts[3]], s, cfg, rng);
            for p in parts {
                buffers::recycle(p.image.data);
            }
            Ok(Transformed { index: primary, sample, orig_size, letterbox: Letterbox::identity(s, s) })
        } else {
            let f = fetch(index)?;
            let (lb_sample, letterbox) = letterbox_fetched(&f, s);
            let sample = random_affine(&lb_sample, s, s, cfg, rng);
            buffers::recycle(lb_sample.image.data);
            buffers::recycle(f.sample.image.data);
            Ok(Transformed { index: f.index, sample, orig_size: f.orig_size, letterbox })
        }
    }
}
