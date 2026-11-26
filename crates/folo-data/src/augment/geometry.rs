//! Box-aware geometric augmentations: random affine, flips.

use super::AugmentConfig;
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::warp::{Affine, Plane, warp_affine, warp_plane};

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
        w > 0.0
            && h > 0.0
            && w > self.min_px
            && h > self.min_px
            && unclipped_area.is_finite()
            && unclipped_area >= 0.0
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
    if boxes.is_empty() || width == 0 || height == 0 || m.inverse().is_none() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(boxes.len());
    transform_boxes_into(boxes, src_w, src_h, m, width, height, filter, &mut out);
    out
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn transform_box_axis(
    bx: &BBox,
    src_w: u32,
    src_h: u32,
    a: f64,
    c: f64,
    e: f64,
    f: f64,
    area_scale: f32,
    width: u32,
    height: u32,
    fw: f32,
    fh: f32,
    filter: &BoxFilter,
) -> Option<BBox> {
    let [x1, y1, x2, y2] = bx.pixel_xyxy(src_w, src_h);
    let u1 = (a * x1 as f64 + c) as f32;
    let u2 = (a * x2 as f64 + c) as f32;
    let v1 = (e * y1 as f64 + f) as f32;
    let v2 = (e * y2 as f64 + f) as f32;
    let (lx, hx) = (u1.min(u2), u1.max(u2));
    let (ly, hy) = (v1.min(v2), v1.max(v2));
    let (cx1, cy1, cx2, cy2) = (lx.clamp(0.0, fw), ly.clamp(0.0, fh), hx.clamp(0.0, fw), hy.clamp(0.0, fh));
    let unclipped = (x2 - x1) * (y2 - y1) * area_scale;
    filter
        .keep(cx2 - cx1, cy2 - cy1, unclipped)
        .then(|| BBox::from_pixel_xyxy(bx.class, [cx1, cy1, cx2, cy2], width, height))
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn transform_box_general(
    bx: &BBox,
    src_w: u32,
    src_h: u32,
    [a, b, c, d, e, f]: [f64; 6],
    area_scale: f32,
    width: u32,
    height: u32,
    fw: f32,
    fh: f32,
    filter: &BoxFilter,
) -> Option<BBox> {
    let [x1, y1, x2, y2] = bx.pixel_xyxy(src_w, src_h);
    let (x1d, y1d, x2d, y2d) = (x1 as f64, y1 as f64, x2 as f64, y2 as f64);
    let (ax1, ax2, by1, by2) = (a * x1d, a * x2d, b * y1d, b * y2d);
    let (dx1, dx2, ey1, ey2) = (d * x1d, d * x2d, e * y1d, e * y2d);
    let u0 = (ax1 + by1 + c) as f32;
    let u1 = (ax2 + by1 + c) as f32;
    let u2 = (ax1 + by2 + c) as f32;
    let u3 = (ax2 + by2 + c) as f32;
    let v0 = (dx1 + ey1 + f) as f32;
    let v1 = (dx2 + ey1 + f) as f32;
    let v2 = (dx1 + ey2 + f) as f32;
    let v3 = (dx2 + ey2 + f) as f32;
    let lx = u0.min(u1).min(u2).min(u3);
    let hx = u0.max(u1).max(u2).max(u3);
    let ly = v0.min(v1).min(v2).min(v3);
    let hy = v0.max(v1).max(v2).max(v3);
    let (cx1, cy1, cx2, cy2) = (lx.clamp(0.0, fw), ly.clamp(0.0, fh), hx.clamp(0.0, fw), hy.clamp(0.0, fh));
    let unclipped = (x2 - x1) * (y2 - y1) * area_scale;
    filter
        .keep(cx2 - cx1, cy2 - cy1, unclipped)
        .then(|| BBox::from_pixel_xyxy(bx.class, [cx1, cy1, cx2, cy2], width, height))
}

#[allow(clippy::too_many_arguments)]
fn transform_boxes_into(
    boxes: &[BBox],
    src_w: u32,
    src_h: u32,
    m: &Affine,
    width: u32,
    height: u32,
    filter: &BoxFilter,
    out: &mut Vec<BBox>,
) {
    let [a, b, c, d, e, f] = m.0;
    let area_scale = m.det().abs() as f32;
    let (fw, fh) = (width as f32, height as f32);
    if b == 0.0 && d == 0.0 {
        for bx in boxes {
            if let Some(t) =
                transform_box_axis(bx, src_w, src_h, a, c, e, f, area_scale, width, height, fw, fh, filter)
            {
                out.push(t);
            }
        }
        return;
    }
    for bx in boxes {
        if let Some(t) =
            transform_box_general(bx, src_w, src_h, m.0, area_scale, width, height, fw, fh, filter)
        {
            out.push(t);
        }
    }
}

/// Transforms `boxes` in place without allocating a second `Vec<BBox>`.
pub(crate) fn transform_boxes_in_place(
    boxes: &mut Vec<BBox>,
    src_w: u32,
    src_h: u32,
    m: &Affine,
    width: u32,
    height: u32,
    filter: &BoxFilter,
) {
    if boxes.is_empty() {
        return;
    }
    if width == 0 || height == 0 || m.inverse().is_none() {
        boxes.clear();
        return;
    }
    let [a, b, c, d, e, f] = m.0;
    let area_scale = m.det().abs() as f32;
    let (fw, fh) = (width as f32, height as f32);
    let mut write = 0usize;
    if b == 0.0 && d == 0.0 {
        for read in 0..boxes.len() {
            if let Some(t) = transform_box_axis(
                &boxes[read],
                src_w,
                src_h,
                a,
                c,
                e,
                f,
                area_scale,
                width,
                height,
                fw,
                fh,
                filter,
            ) {
                boxes[write] = t;
                write += 1;
            }
        }
    } else {
        for read in 0..boxes.len() {
            if let Some(t) = transform_box_general(
                &boxes[read],
                src_w,
                src_h,
                m.0,
                area_scale,
                width,
                height,
                fw,
                fh,
                filter,
            ) {
                boxes[write] = t;
                write += 1;
            }
        }
    }
    boxes.truncate(write);
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

/// [`random_affine`] of the sample whose image is `plane`'s source image ([`Plane::render`]) and whose boxes are
/// `boxes`, without rendering it (mosaic tiles are warped where they lie).
pub(crate) fn random_affine_plane(
    plane: &Plane,
    boxes: &[BBox],
    out_w: u32,
    out_h: u32,
    cfg: &AugmentConfig,
    rng: &mut Rng,
) -> Sample {
    let (w, h) = (plane.width, plane.height);
    let m = random_affine_matrix(w, h, out_w, out_h, cfg, rng);
    if m.is_identity() && (w, h) == (out_w, out_h) {
        return Sample { image: plane.render(), boxes: boxes.to_vec() };
    }
    Sample {
        image: warp_plane(plane, &m, out_w, out_h),
        boxes: transform_boxes(boxes, w, h, &m, out_w, out_h, &cfg.box_filter()),
    }
}

/// Mirrors a sample left to right in place.
pub fn flip_lr(s: &mut Sample) {
    let stride = s.image.stride();
    let h = s.image.height as usize;
    if stride > 0 && h > 0 {
        crate::simd::assert_image_data(&s.image);
        crate::par::for_rows(&mut s.image.data[..h * stride], stride, |_, row| {
            crate::simd::planar::flip_lr_row(row);
        });
    }
    for b in &mut s.boxes {
        b.cx = 1.0 - b.cx;
    }
}

/// Mirrors a sample top to bottom in place.
pub fn flip_ud(s: &mut Sample) {
    let stride = s.image.stride();
    let h = s.image.height as usize;
    let half = h / 2;
    if stride > 0 && half > 0 {
        let total = h.checked_mul(stride).expect("flip_ud image size overflow");
        let data = &mut s.image.data[..total];
        let (top, bot) = data.split_at_mut((h - half) * stride);
        for (t, b) in top[..half * stride].chunks_exact_mut(stride).zip(bot.chunks_exact_mut(stride).rev()) {
            t.swap_with_slice(b);
        }
    }
    for b in &mut s.boxes {
        b.cy = 1.0 - b.cy;
    }
}

/// Target `(width, height)` after scaling `(w, h)` so its long side is `long` pixels.
pub(crate) fn long_side_dims(w: u32, h: u32, long: u32) -> (u32, u32) {
    let r = long as f64 / w.max(h) as f64;
    let nw = ((w as f64 * r).round() as u32).max(1);
    let nh = ((h as f64 * r).round() as u32).max(1);
    (nw, nh)
}

/// Resizes a sample so its long side is `long` pixels (keeping the aspect ratio). Boxes are normalised, so they do
/// not change.
pub fn resize_long_side(s: Sample, long: u32) -> Sample {
    let (w, h) = (s.image.width, s.image.height);
    let (nw, nh) = long_side_dims(w, h, long);
    if (nw, nh) == (w, h) {
        return s;
    }
    let image = crate::resize::resize(&s.image, nw, nh);
    crate::buffers::recycle(s.image.data);
    Sample { image, boxes: s.boxes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    fn transform_boxes_ref(
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
        let mut out = Vec::new();
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

    #[test]
    fn transform_boxes_fast_and_general_match_reference_bit_exact() {
        let boxes = vec![
            BBox::new(0, 0.25, 0.35, 0.2, 0.3),
            BBox::new(1, 0.92, 0.1, 0.25, 0.2),
            BBox::new(2, 0.5, 0.5, 0.01, 0.01),
            BBox::new(3, 0.1, 0.85, 0.18, 0.4),
        ];
        let filter = BoxFilter::default();
        let matrices = [
            Affine([1.25, 0.0, -30.5, 0.0, 0.85, 14.25]),
            Affine([-0.9, 0.0, 640.0, 0.0, -1.1, 480.0]),
            Affine::translate(-320.0, -240.0)
                .then(&Affine::rotate(13.5))
                .then(&Affine::scale(0.92, 1.07))
                .then(&Affine::shear(3.0, -2.5))
                .then(&Affine::translate(310.0, 260.0)),
        ];
        for m in &matrices {
            let got = transform_boxes(&boxes, 640, 480, m, 640, 640, &filter);
            let want = transform_boxes_ref(&boxes, 640, 480, m, 640, 640, &filter);
            assert_eq!(got.len(), want.len(), "{m:?}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.class, w.class);
                assert_eq!(
                    [g.cx.to_bits(), g.cy.to_bits(), g.w.to_bits(), g.h.to_bits()],
                    [w.cx.to_bits(), w.cy.to_bits(), w.w.to_bits(), w.h.to_bits()],
                    "{m:?}"
                );
            }
        }
    }

    #[test]
    fn flips_match_reference_across_widths_and_heights() {
        for (w, h) in [(1, 1), (3, 2), (16, 3), (31, 4), (32, 5), (33, 6), (48, 7), (67, 8)] {
            let mut data = vec![0u8; (w * h * 3) as usize];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i * 37 + 13) % 251) as u8;
            }
            let orig = Sample {
                image: Image { width: w, height: h, data },
                boxes: vec![BBox::new(0, 0.25, 0.3, 0.2, 0.4)],
            };
            let mut lr = orig.clone();
            flip_lr(&mut lr);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(lr.image.pixel(x, y), orig.image.pixel(w - 1 - x, y), "lr {w}x{h} ({x},{y})");
                }
            }
            assert!((lr.boxes[0].cx - 0.75).abs() < 1e-6);

            let mut ud = orig.clone();
            flip_ud(&mut ud);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(ud.image.pixel(x, y), orig.image.pixel(x, h - 1 - y), "ud {w}x{h} ({x},{y})");
                }
            }
            assert!((ud.boxes[0].cy - 0.7).abs() < 1e-6);
        }
        // Oversized backing buffer: only the declared `height` rows are flipped; trailing bytes stay untouched.
        let mut oversized = Sample {
            image: Image { width: 1, height: 2, data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9] },
            boxes: vec![BBox::new(0, 0.5, 0.25, 0.2, 0.2)],
        };
        flip_ud(&mut oversized);
        assert_eq!(oversized.image.data, vec![4, 5, 6, 1, 2, 3, 7, 8, 9]);
        // Truncated backing buffer panics before swapping any rows.
        let truncated = std::panic::catch_unwind(|| {
            let mut s = Sample {
                image: Image { width: 1, height: 4, data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9] },
                boxes: vec![],
            };
            flip_ud(&mut s);
        });
        assert!(truncated.is_err());
    }

    #[test]
    fn flip_lr_and_box_filter_edge_cases() {
        // Zero-width/zero-height images do not panic in `flip_lr`.
        for (w, h) in [(0u32, 0u32), (0, 5), (5, 0)] {
            let mut s = Sample {
                image: Image { width: w, height: h, data: Vec::new() },
                boxes: vec![BBox::new(0, 0.25, 0.5, 0.2, 0.2)],
            };
            flip_lr(&mut s);
            assert!((s.boxes[0].cx - 0.75).abs() < 1e-6);
        }
        // Oversized backing buffer leaves trailing bytes untouched; truncated panics.
        let mut oversized = Sample {
            image: Image { width: 2, height: 1, data: vec![1, 2, 3, 4, 5, 6, 9, 9, 9] },
            boxes: vec![],
        };
        flip_lr(&mut oversized);
        assert_eq!(oversized.image.data, vec![4, 5, 6, 1, 2, 3, 9, 9, 9]);
        let truncated = std::panic::catch_unwind(|| {
            let mut s =
                Sample { image: Image { width: 4, height: 2, data: vec![1, 2, 3, 4, 5, 6] }, boxes: vec![] };
            flip_lr(&mut s);
        });
        assert!(truncated.is_err());
        // Permissive BoxFilter (`min_px == 0.0`) still rejects zero-area, negative, and NaN inputs.
        let perm = BoxFilter { min_px: 0.0, min_area_ratio: 0.0, max_aspect: 1e9 };
        assert!(!perm.keep(0.0, 10.0, 100.0));
        assert!(!perm.keep(10.0, 0.0, 100.0));
        assert!(!perm.keep(10.0, 10.0, f32::NAN));
        assert!(!perm.keep(10.0, 10.0, -1.0));
        assert!(perm.keep(1.0, 1.0, 1.0));
        // `transform_boxes_in_place` matches `transform_boxes` bit for bit and handles zero output size.
        let boxes = vec![BBox::new(0, 0.5, 0.5, 0.3, 0.4), BBox::new(1, 0.99, 0.99, 0.01, 0.01)];
        let m = Affine::rotate(11.0).then(&Affine::translate(20.0, 10.0));
        let want = transform_boxes(&boxes, 100, 100, &m, 64, 64, &BoxFilter::default());
        let mut got = boxes.clone();
        transform_boxes_in_place(&mut got, 100, 100, &m, 64, 64, &BoxFilter::default());
        assert_eq!(got, want);
        let mut zero_out = boxes.clone();
        transform_boxes_in_place(&mut zero_out, 100, 100, &m, 0, 64, &BoxFilter::default());
        assert!(zero_out.is_empty());
        // Singular affine collapses the image to border padding and must drop all boxes, even when its
        // collapsed diagonal bounding box would otherwise have positive width and height.
        let singular = Affine([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(singular.inverse().is_none());
        assert!(transform_boxes(&boxes, 100, 100, &singular, 64, 64, &perm).is_empty());
        let mut sing_in_place = boxes.clone();
        transform_boxes_in_place(&mut sing_in_place, 100, 100, &singular, 64, 64, &perm);
        assert!(sing_in_place.is_empty());
        let sample = Sample { image: Image::new(32, 32, [200, 10, 20]), boxes };
        let warped = affine_sample(&sample, &singular, 32, 32, &perm);
        assert!(warped.boxes.is_empty());
        assert!(warped.image.data.iter().all(|&b| b == PAD_VALUE));
    }
}
