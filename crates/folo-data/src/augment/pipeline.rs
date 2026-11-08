//! The training transform: mosaic → affine → mixup → HSV → flips, ending at `imgsz x imgsz`.

use super::{AugmentConfig, flip_lr, flip_ud, mixup, mosaic4, random_affine, random_hsv, resize_long_side};
use crate::error::Result;
use crate::image::Sample;
use crate::letterbox::{Letterbox, LetterboxOptions, letterbox_sample};
use crate::rng::Rng;

/// A transformed sample plus the letterbox geometry of its primary image.
#[derive(Clone, Debug)]
pub struct Transformed {
    /// The `imgsz x imgsz` result.
    pub sample: Sample,
    /// Size `[width, height]` of the primary source image as decoded.
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

    /// Produces the transformed sample for dataset `index`. `fetch(i)` loads sample `i` of a dataset with `len`
    /// samples (the loader passes a fetch that substitutes unreadable images deterministically).
    pub fn apply<F>(&self, index: usize, len: usize, fetch: &F, rng: &mut Rng) -> Result<Transformed>
    where
        F: Fn(usize) -> Result<Sample>,
    {
        let Some(cfg) = &self.augment else {
            let s = fetch(index)?;
            let orig_size = [s.image.width, s.image.height];
            let (sample, letterbox) =
                letterbox_sample(&s, self.imgsz, self.imgsz, &LetterboxOptions::default());
            return Ok(Transformed { sample, orig_size, letterbox });
        };
        let mut out = self.one(cfg, index, len, fetch, rng)?;
        if self.mosaic && rng.chance(cfg.mixup) {
            let other = self.one(cfg, rng.below(len), len, fetch, rng)?;
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

    fn one<F>(
        &self,
        cfg: &AugmentConfig,
        index: usize,
        len: usize,
        fetch: &F,
        rng: &mut Rng,
    ) -> Result<Transformed>
    where
        F: Fn(usize) -> Result<Sample>,
    {
        let s = self.imgsz;
        if self.mosaic && rng.chance(cfg.mosaic) {
            let ids = [index, rng.below(len), rng.below(len), rng.below(len)];
            let mut parts = Vec::with_capacity(4);
            let mut orig_size = [0, 0];
            for (k, &i) in ids.iter().enumerate() {
                let p = fetch(i)?;
                if k == 0 {
                    orig_size = [p.image.width, p.image.height];
                }
                parts.push(resize_long_side(p, s));
            }
            let canvas = mosaic4([&parts[0], &parts[1], &parts[2], &parts[3]], s, rng);
            let sample = random_affine(&canvas, s, s, cfg, rng);
            Ok(Transformed { sample, orig_size, letterbox: Letterbox::identity(s, s) })
        } else {
            let p = fetch(index)?;
            let orig_size = [p.image.width, p.image.height];
            let (lb_sample, letterbox) = letterbox_sample(&p, s, s, &LetterboxOptions::default());
            let sample = random_affine(&lb_sample, s, s, cfg, rng);
            Ok(Transformed { sample, orig_size, letterbox })
        }
    }
}
