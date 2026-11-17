//! The training transform: mosaic → affine → mixup → HSV → flips, ending at `imgsz x imgsz`.

use super::{AugmentConfig, flip_lr, flip_ud, mixup, mosaic4, random_affine, random_hsv, resize_long_side};
use crate::dataset::Dataset;
use crate::error::Result;
use crate::image::Sample;
use crate::letterbox::{Letterbox, LetterboxOptions, letterbox_sample};
use crate::rng::Rng;

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

    /// Produces the transformed sample for dataset `index`, its mosaic and mixup partners drawn uniformly from the
    /// dataset. `fetch(i)` loads sample `i` of a dataset with `len` samples, with the index it actually loaded (the
    /// loader passes a fetch that substitutes unreadable images deterministically) and its decoded size; the sample
    /// may come resized to long side `imgsz` (the loader's RAM cache), which gives the same result.
    pub fn apply<F>(&self, index: usize, len: usize, fetch: &F, rng: &mut Rng) -> Result<Transformed>
    where
        F: Fn(usize) -> Result<Fetched>,
    {
        self.apply_with(index, &|r: &mut Rng| r.below(len), fetch, rng)
    }

    /// [`TrainTransform::apply`] with the partners chosen by `pick`. Samples are named by ids the transform only
    /// passes on: the primary is `primary`, each mosaic or mixup partner is `pick(rng)` (one draw per partner, in
    /// the order of [`TrainTransform::apply`]), and `fetch(id)` loads one. [`TrainTransform::apply`] uses dataset
    /// indices and a uniform draw; the loader's mosaic window ([`AugmentConfig::mosaic_window`]) uses positions in
    /// the epoch's order and a draw from the previous ones.
    pub fn apply_with<P, F>(&self, primary: usize, pick: &P, fetch: &F, rng: &mut Rng) -> Result<Transformed>
    where
        P: Fn(&mut Rng) -> usize,
        F: Fn(usize) -> Result<Fetched>,
    {
        let Some(cfg) = &self.augment else {
            let f = fetch(primary)?;
            let (sample, letterbox) = letterbox_fetched(&f, self.imgsz);
            return Ok(Transformed { index: f.index, sample, orig_size: f.orig_size, letterbox });
        };
        let mut out = self.one(cfg, primary, pick, fetch, rng)?;
        if self.mosaic && rng.chance(cfg.mixup) {
            let other = self.one(cfg, pick(rng), pick, fetch, rng)?;
            mixup(&mut out.sample, &other.sample, rng);
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

    fn one<P, F>(&self, cfg: &AugmentConfig, index: usize, pick: &P, fetch: &F, rng: &mut Rng) -> Result<Transformed>
    where
        P: Fn(&mut Rng) -> usize,
        F: Fn(usize) -> Result<Fetched>,
    {
        let s = self.imgsz;
        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, pick(rng), pick(rng), pick(rng)];
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
            let canvas = mosaic4([&parts[0], &parts[1], &parts[2], &parts[3]], s, rng);
            let sample = random_affine(&canvas, s, s, cfg, rng);
            Ok(Transformed { index: primary, sample, orig_size, letterbox: Letterbox::identity(s, s) })
        } else {
            let f = fetch(index)?;
            let (lb_sample, letterbox) = letterbox_fetched(&f, s);
            let sample = random_affine(&lb_sample, s, s, cfg, rng);
            Ok(Transformed { index: f.index, sample, orig_size: f.orig_size, letterbox })
        }
    }
}
