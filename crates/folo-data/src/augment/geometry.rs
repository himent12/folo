//! Box-aware geometric augmentations: random affine, flips.

use super::AugmentConfig;
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::warp::{Affine, warp_affine};

/// Rules for dropping boxes that an augmentation made useless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxFilter {
    /// Minimum width and height in output pixels.
    pub min_px: f32,
    /// Minimum ratio of the final area to the area the box would have had if it were not clipped.
    pub min_area_ratio: f32,
    /// Maximum aspect ratio (long side / short side).
    pub max_aspect: f32,
}

impl Default for BoxFilter {
    fn default() -> Self {
        BoxFilter { min_px: 2.0, min_area_ratio: 0.1, max_aspect: 100.0 }
    }
}

impl BoxFilter {
    /// Whether a box of `w x h` pixels survives, given the area it would have had unclipped.
    pub fn keep(&self, w: f32, h: f32, unclipped_area: f32) -> bool {
        w > self.min_px
            && h > self.min_px
            && w * h >= self.min_area_ratio * unclipped_area
            && (w / h).max(h / w) < self.max_aspect
    }
}

/// Warps a sample by `m` (source pixels → output pixels) into `width x height`, transforming every box by its four
/// corners, clipping to the output and dropping boxes that fail `filter`.
/// The identity at the same size returns the sample unchanged (no filtering: nothing was transformed).
pub fn affine_sample(s: &Sample, m: &Affine, width: u32, height: u32, filter: &BoxFilter) -> Sample {
    if m.is_identity() && s.image.width == width && s.image.height == height {
        return s.clone();
    }
    let image = warp_affine(&s.image, m, width, height, [PAD_VALUE; 3]);
    Sample {
        image,
        boxes: transform_boxes(&s.boxes, s.image.width, s.image.height, m, width, height, filter),
    }
}

/// Transforms normalised boxes of a `src_w x src_h` image through `m` into normalised boxes of `width x height`.
pub fn transform_boxes(
    boxes: &[BBox],
    src_w: u32,
    src_h: u32,
    m: &Affine,
    width: u32,
    height: u32,
    filter: &BoxFilter,
) -> Vec<BBox> {
    let area_scale = m.det().abs() as f32;
    let (fw, fh) = (width as f32, height as f32);
    let mut out = Vec::with_capacity(boxes.len());
    for b in boxes {
        let [x1, y1, x2, y2] = b.pixel_xyxy(src_w, src_h);
        let corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)];
        let (mut lx, mut ly, mut hx, mut hy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in corners {
            let (u, v) = m.apply(x as f64, y as f64);
            let (u, v) = (u as f32, v as f32);
            lx = lx.min(u);
            ly = ly.min(v);
            hx = hx.max(u);
            hy = hy.max(v);
        }
        let (cx1, cy1, cx2, cy2) =
            (lx.clamp(0.0, fw), ly.clamp(0.0, fh), hx.clamp(0.0, fw), hy.clamp(0.0, fh));
        let unclipped = (x2 - x1) * (y2 - y1) * area_scale;
        if filter.keep(cx2 - cx1, cy2 - cy1, unclipped) {
            out.push(BBox::from_pixel_xyxy(b.class, [cx1, cy1, cx2, cy2], width, height));
        }
    }
    out
}

/// Builds the YOLO-style random affine for a `src_w x src_h` input rendered into `out_w x out_h`: centre, rotate and
/// scale, shear, then translate so the input centre lands at `(0.5 ± translate) * out`.
pub fn random_affine_matrix(
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
    cfg: &AugmentConfig,
    rng: &mut Rng,
) -> Affine {
    // Always draw the same numbers so the stream does not depend on which parameters are enabled.
    let angle = rng.uniform(-cfg.degrees, cfg.degrees) as f64;
    let scale = rng.uniform(1.0 - cfg.scale, 1.0 + cfg.scale) as f64;
    let shx = rng.uniform(-cfg.shear, cfg.shear) as f64;
    let shy = rng.uniform(-cfg.shear, cfg.shear) as f64;
    let tx = rng.uniform(0.5 - cfg.translate, 0.5 + cfg.translate) as f64;
    let ty = rng.uniform(0.5 - cfg.translate, 0.5 + cfg.translate) as f64;
    Affine::translate(-(src_w as f64) / 2.0, -(src_h as f64) / 2.0)
        .then(&Affine::rotate(angle))
        .then(&Affine::scale(scale.max(1e-3), scale.max(1e-3)))
        .then(&Affine::shear(shx, shy))
        .then(&Affine::translate(tx * out_w as f64, ty * out_h as f64))
}

/// Random affine (scale, translate, shear, rotation) of a sample into `out_w x out_h`.
pub fn random_affine(s: &Sample, out_w: u32, out_h: u32, cfg: &AugmentConfig, rng: &mut Rng) -> Sample {
    let m = random_affine_matrix(s.image.width, s.image.height, out_w, out_h, cfg, rng);
    affine_sample(s, &m, out_w, out_h, &cfg.box_filter())
}

/// Mirrors a sample left to right in place.
pub fn flip_lr(s: &mut Sample) {
    let stride = s.image.stride();
    let w = s.image.width as usize;
    for row in s.image.data.chunks_exact_mut(stride) {
        for x in 0..w / 2 {
            let (a, b) = (x * 3, (w - 1 - x) * 3);
            for c in 0..3 {
                row.swap(a + c, b + c);
            }
        }
    }
    for b in &mut s.boxes {
        b.cx = 1.0 - b.cx;
    }
}

/// Mirrors a sample top to bottom in place.
pub fn flip_ud(s: &mut Sample) {
    let stride = s.image.stride();
    let h = s.image.height as usize;
    let data = &mut s.image.data;
    for y in 0..h / 2 {
        let (top, bottom) = data.split_at_mut((h - 1 - y) * stride);
        top[y * stride..(y + 1) * stride].swap_with_slice(&mut bottom[..stride]);
    }
    for b in &mut s.boxes {
        b.cy = 1.0 - b.cy;
    }
}

/// Resizes a sample so its long side is `long` pixels (keeping the aspect ratio). Boxes are normalised, so they do
/// not change.
pub fn resize_long_side(s: Sample, long: u32) -> Sample {
    let (w, h) = (s.image.width, s.image.height);
    let r = long as f64 / w.max(h) as f64;
    let nw = ((w as f64 * r).round() as u32).max(1);
    let nh = ((h as f64 * r).round() as u32).max(1);
    if (nw, nh) == (w, h) {
        return s;
    }
    Sample { image: crate::resize::resize(&s.image, nw, nh), boxes: s.boxes }
}
