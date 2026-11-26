//! 4-image mosaic and mixup.

use super::AugmentConfig;
use super::BoxFilter;
use super::color::clamp_and_filter_boxes;
use super::geometry::random_affine_plane;
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::simd::assert_image_data;
use crate::warp::{Plane, PlaneTiles, Tile};

/// Tiles four samples around a random centre on a `2s x 2s` grey canvas (YOLO's mosaic).
///
/// The centre is drawn from `[s/2, 3s/2)` in both axes; image 0 goes top-left of it, 1 top-right, 2 bottom-left and
/// 3 bottom-right, each cropped by the canvas edge. Inputs are normally pre-resized so their long side is `s`. Boxes
/// are offset and clipped to each image's visible tile on the canvas; follow with an affine (as
/// [`crate::TrainTransform`] does) to get back to `s x s` and to drop boxes that became too small.
pub fn mosaic4(parts: [&Sample; 4], s: u32, rng: &mut Rng) -> Sample {
    let (plane, boxes) = mosaic_plane(parts, s, rng);
    Sample { image: plane.render(), boxes }
}

/// [`random_affine`](super::random_affine) of [`mosaic4`] into `s x s` (the training transform's mosaic), bit for
/// bit, drawing the same numbers from `rng`, without painting the `2s x 2s` canvas: the warp reads the four tiles
/// where they lie on it.
pub fn mosaic_affine(parts: [&Sample; 4], s: u32, cfg: &AugmentConfig, rng: &mut Rng) -> Sample {
    let (plane, boxes) = mosaic_plane(parts, s, rng);
    random_affine_plane(&plane, &boxes, s, s, cfg, rng)
}

/// [`mosaic4`] as its canvas's tiles on a grey plane (the canvas is [`Plane::render`]) and its boxes, drawing the
/// same numbers from `rng`: the training transform warps the tiles without painting the canvas.
pub(crate) fn mosaic_plane<'a>(parts: [&'a Sample; 4], s: u32, rng: &mut Rng) -> (Plane<'a>, Vec<BBox>) {
    let cs = 2 * s as i64;
    let xc = rng.uniform(s as f32 * 0.5, s as f32 * 1.5) as i64;
    let yc = rng.uniform(s as f32 * 0.5, s as f32 * 1.5) as i64;
    let mut tiles = PlaneTiles::new();
    let total_boxes: usize = parts.iter().map(|p| p.boxes.len()).sum();
    let mut boxes = Vec::with_capacity(total_boxes);
    for (i, part) in parts.into_iter().enumerate() {
        let (w, h) = (part.image.width as i64, part.image.height as i64);
        // Destination rectangle on the canvas (a) and source rectangle in the image (b).
        let (x1a, y1a, x2a, y2a) = match i {
            0 => ((xc - w).max(0), (yc - h).max(0), xc, yc),
            1 => (xc, (yc - h).max(0), (xc + w).min(cs), yc),
            2 => ((xc - w).max(0), yc, xc, (yc + h).min(cs)),
            _ => (xc, yc, (xc + w).min(cs), (yc + h).min(cs)),
        };
        let (x1b, y1b) = match i {
            0 => (w - (x2a - x1a), h - (y2a - y1a)),
            1 => (0, h - (y2a - y1a)),
            2 => (w - (x2a - x1a), 0),
            _ => (0, 0),
        };
        if x2a > x1a && y2a > y1a {
            tiles.push(Tile { img: &part.image, dst: [x1a, y1a, x2a, y2a], src: [x1b, y1b] });
            let (padw, padh) = ((x1a - x1b) as f32, (y1a - y1b) as f32);
            let (tx0, ty0, tx1, ty1) = (x1a as f32, y1a as f32, x2a as f32, y2a as f32);
            for b in &part.boxes {
                let [x1, y1, x2, y2] = b.pixel_xyxy(part.image.width, part.image.height);
                let c = [
                    (x1 + padw).clamp(tx0, tx1),
                    (y1 + padh).clamp(ty0, ty1),
                    (x2 + padw).clamp(tx0, tx1),
                    (y2 + padh).clamp(ty0, ty1),
                ];
                if c[2] > c[0] && c[3] > c[1] {
                    boxes.push(BBox::from_pixel_xyxy(b.class, c, cs as u32, cs as u32));
                }
            }
        }
    }
    (Plane { width: cs as u32, height: cs as u32, fill: [PAD_VALUE; 3], tiles }, boxes)
}

/// Draws the 8-bit fixed-point blend weight `wa ∈ 0..=256` for [`mixup`] (shared with the fused planar path).
#[inline]
pub(crate) fn mixup_weight(rng: &mut Rng) -> u32 {
    // Beta(32, 32) has mean 0.5 and sd ~0.062; a clamped normal is indistinguishable for this purpose.
    let r = (0.5 + 0.0620 * rng.normal()).clamp(0.0, 1.0);
    (r * 256.0).round() as u32
}

/// Blends `other` into `s` (YOLO mixup): `s = r * s + (1 - r) * other` with `r ≈ Beta(32, 32)`, and concatenates
/// the boxes. Both images must have the same size.
pub fn mixup(s: &mut Sample, other: &Sample, rng: &mut Rng) {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "mixup needs images of the same size"
    );
    let wa = mixup_weight(rng);
    let n = s.image.data.len().min(other.image.data.len());
    crate::simd::planar::mixup_blend_u8(&mut s.image.data[..n], &other.image.data[..n], wa);
    s.boxes.extend_from_slice(&other.boxes);
}

/// Parameters for [`mixup_beta`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixUpConfig {
    /// Symmetric Beta shape parameter `alpha > 0` (`r ~ Beta(alpha, alpha)`, default `32.0`).
    /// Smaller values (e.g. `0.4`–`1.0`) explore the full `[0, 1]` range; larger values concentrate around `0.5`.
    pub alpha: f32,
    /// Optional [`BoxFilter`] applied after clamping both host and donor boxes to `[0, 1]`.
    pub filter: Option<BoxFilter>,
}

impl Default for MixUpConfig {
    fn default() -> Self {
        Self { alpha: 32.0, filter: None }
    }
}

impl MixUpConfig {
    /// Validates parameter ranges; returns an error message on invalid values.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return Err(format!("mixup.alpha must be finite and > 0, got {}", self.alpha));
        }
        Ok(())
    }
}

/// Samples `ln(Gamma(shape, 1))` deterministically using Marsaglia–Tsang's method so tiny `shape` values
/// never underflow to zero before Beta normalization.
fn sample_log_gamma(shape: f64, rng: &mut Rng) -> f64 {
    if shape < 1.0 {
        let log_g = sample_log_gamma(shape + 1.0, rng);
        let u = (1.0 - rng.f64()).max(f64::MIN_POSITIVE);
        return log_g + u.ln() / shape;
    }
    let d = shape - 1.0 / 3.0;
    let c = (1.0 / (9.0 * d)).sqrt();
    loop {
        let x = rng.normal() as f64;
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = 1.0 - rng.f64();
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 || u.ln() < 0.5 * x2 + d * (1.0 - v3 + v3.ln()) {
            return (d * v3).ln();
        }
    }
}

/// Samples a symmetric `Beta(alpha, alpha)` weight in `[0.0, 1.0]` deterministically in log-space.
fn sample_beta_sym(alpha: f32, rng: &mut Rng) -> f32 {
    let a = alpha as f64;
    let lx = sample_log_gamma(a, rng);
    let ly = sample_log_gamma(a, rng);
    let diff = lx - ly;
    if diff.is_nan() {
        return 0.5;
    }
    let r = if diff >= 0.0 {
        1.0 / (1.0 + (-diff).exp())
    } else {
        let e = diff.exp();
        e / (1.0 + e)
    };
    r.clamp(0.0, 1.0) as f32
}

/// Blends `other` into `s` with an explicit weight `ratio ∈ [0, 1]` (`s = ratio * s + (1 - ratio) * other`),
/// preserves boxes from both samples, clamps out-of-bounds boxes to `[0, 1]`, and drops degenerate,
/// non-finite, or `filter`-rejected boxes.
///
/// Pixel blending and endpoint box selection both use the rounded 8-bit fixed-point weight
/// `wa = (ratio.clamp(0, 1) * 256).round()`: when `wa == 256` (`ratio >= 255.5 / 256`), only host pixels and
/// clamped host boxes are kept; when `wa == 0` (`ratio < 0.5 / 256`), only donor pixels and clamped donor
/// boxes are kept. Both images must have the same dimensions.
///
/// # Panics
/// If image dimensions do not match or image buffers are shorter than `width * height * 3`.
pub fn mixup_with_ratio(s: &mut Sample, other: &Sample, ratio: f32, filter: Option<&BoxFilter>) {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "mixup_with_ratio needs images of the same size"
    );
    assert_image_data(&s.image);
    assert_image_data(&other.image);
    let r = if ratio.is_nan() { 1.0 } else { ratio.clamp(0.0, 1.0) };
    let wa = (r * 256.0).round() as u32;
    let total = s.image.height as usize * s.image.stride();
    if total > 0 {
        crate::simd::planar::mixup_blend_u8(&mut s.image.data[..total], &other.image.data[..total], wa);
    }
    if wa == 0 {
        s.boxes.clear();
        s.boxes.extend_from_slice(&other.boxes);
    } else if wa < 256 {
        s.boxes.extend_from_slice(&other.boxes);
    }
    clamp_and_filter_boxes(&mut s.boxes, s.image.width, s.image.height, filter);
}

/// Deterministic `Beta(cfg.alpha, cfg.alpha)` MixUp blending `other` into `s` with box clamping and optional
/// [`BoxFilter`] filtering. Returns the unrounded sampled blend weight `r ∈ [0, 1]` (`1.0` no-op if `cfg` is
/// invalid); pixel blending and endpoint box selection use the rounded 8-bit weight described in
/// [`mixup_with_ratio`].
///
/// Both images must have the same dimensions.
pub fn mixup_beta(s: &mut Sample, other: &Sample, cfg: &MixUpConfig, rng: &mut Rng) -> f32 {
    assert_eq!(
        (s.image.width, s.image.height),
        (other.image.width, other.image.height),
        "mixup_beta needs images of the same size"
    );
    if cfg.validate().is_err() {
        return 1.0;
    }
    let r = sample_beta_sym(cfg.alpha, rng);
    mixup_with_ratio(s, other, r, cfg.filter.as_ref());
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    #[test]
    fn mixup_with_ratio_blends_pixels_and_clamps_or_filters_boxes() {
        let host = Sample {
            image: Image::new(32, 32, [200, 100, 0]),
            boxes: vec![
                // Valid host box -> preserved.
                BBox::from_pixel_xyxy(0, [4.0, 4.0, 20.0, 20.0], 32, 32),
                // Partially OOB host box -> clamped to [0, 32].
                BBox::from_xyxy(1, [-0.25, 0.25, 0.5, 0.75]),
                // Degenerate host box -> dropped.
                BBox::new(2, 0.5, 0.5, 0.0, 0.2),
            ],
        };
        let donor = Sample {
            image: Image::new(32, 32, [0, 100, 200]),
            boxes: vec![
                // Valid donor box -> preserved.
                BBox::from_pixel_xyxy(3, [10.0, 10.0, 28.0, 28.0], 32, 32),
                // Tiny 1x1 donor box -> dropped when BoxFilter(min_px=2.0) is active.
                BBox::from_pixel_xyxy(4, [2.0, 2.0, 3.0, 3.0], 32, 32),
                // Non-finite donor box -> dropped.
                BBox::new(5, 0.5, f32::NAN, 0.2, 0.2),
            ],
        };

        // ratio = 0.5 blends pixels halfway and keeps clamped valid boxes from both images.
        let mut mid = host.clone();
        let filter = BoxFilter::default();
        mixup_with_ratio(&mut mid, &donor, 0.5, Some(&filter));
        assert_eq!(mid.image.pixel(16, 16), [100, 100, 100]);
        assert_eq!(mid.boxes.len(), 3);
        assert_eq!(mid.boxes[0].class, 0);
        assert_eq!(mid.boxes[1].class, 1);
        assert!((mid.boxes[1].xyxy()[0] - 0.0).abs() < 1e-6);
        assert_eq!(mid.boxes[2].class, 3);

        // ratio = 1.0 keeps only host pixels and clamped host boxes; ratio = 0.0 keeps only donor.
        let mut only_host = host.clone();
        mixup_with_ratio(&mut only_host, &donor, 1.0, Some(&filter));
        assert_eq!(only_host.image.pixel(0, 0), [200, 100, 0]);
        assert_eq!(only_host.boxes.len(), 2);
        assert!(only_host.boxes.iter().all(|b| b.class <= 1));

        let mut only_donor = host.clone();
        mixup_with_ratio(&mut only_donor, &donor, 0.0, Some(&filter));
        assert_eq!(only_donor.image.pixel(0, 0), [0, 100, 200]);
        assert_eq!(only_donor.boxes.len(), 1);
        assert_eq!(only_donor.boxes[0].class, 3);

        // 1x1 and 0x0 images with empty boxes never panic.
        let mut one = Sample { image: Image::new(1, 1, [20, 40, 60]), boxes: vec![] };
        let one_other = Sample { image: Image::new(1, 1, [100, 120, 140]), boxes: vec![] };
        mixup_with_ratio(&mut one, &one_other, 0.5, None);
        assert_eq!(one.image.pixel(0, 0), [60, 80, 100]);
        assert!(one.boxes.is_empty());

        let mut empty = Sample { image: Image::new(0, 0, [0, 0, 0]), boxes: vec![] };
        let empty_other = empty.clone();
        mixup_with_ratio(&mut empty, &empty_other, 0.5, None);
        assert!(empty.boxes.is_empty());
    }

    #[test]
    fn mixup_beta_is_deterministic_symmetric_and_validates_config() {
        let host =
            Sample { image: Image::new(8, 8, [200, 50, 10]), boxes: vec![BBox::new(0, 0.5, 0.5, 0.4, 0.4)] };
        let donor =
            Sample { image: Image::new(8, 8, [20, 180, 240]), boxes: vec![BBox::new(1, 0.3, 0.7, 0.2, 0.2)] };
        for &alpha in &[0.4f32, 1.0, 32.0] {
            let cfg = MixUpConfig { alpha, filter: Some(BoxFilter::default()) };
            assert!(cfg.validate().is_ok());
            let (mut a, mut b) = (host.clone(), host.clone());
            let ra = mixup_beta(&mut a, &donor, &cfg, &mut Rng::new(77));
            let rb = mixup_beta(&mut b, &donor, &cfg, &mut Rng::new(77));
            assert_eq!(ra.to_bits(), rb.to_bits());
            assert!((0.0..=1.0).contains(&ra));
            assert_eq!(a, b);
        }

        // Distribution mean for symmetric Beta(0.5, 0.5) is 0.5.
        let mut rng = Rng::new(123);
        let mean: f64 = (0..2000).map(|_| sample_beta_sym(0.5, &mut rng) as f64).sum::<f64>() / 2000.0;
        assert!((mean - 0.5).abs() < 0.05, "unexpected Beta(0.5, 0.5) mean {mean}");

        // Tiny alpha concentrates near endpoints (0 and 1) with no artificial underflow spike at 0.5.
        let mut rng_tiny = Rng::new(123);
        let tiny_samples: Vec<f32> = (0..2000).map(|_| sample_beta_sym(0.0005, &mut rng_tiny)).collect();
        let mid_count = tiny_samples.iter().filter(|&&v| (0.1..0.9).contains(&v)).count();
        assert!(
            mid_count < 40,
            "expected Beta(0.0005, 0.0005) near endpoints, got {mid_count}/2000 in (0.1, 0.9)"
        );

        // Invalid alpha is a no-op returning 1.0.
        for bad_alpha in [0.0f32, -2.0, f32::NAN, f32::INFINITY] {
            let cfg = MixUpConfig { alpha: bad_alpha, filter: None };
            assert!(cfg.validate().is_err());
            let mut probe = host.clone();
            assert_eq!(mixup_beta(&mut probe, &donor, &cfg, &mut Rng::new(1)), 1.0);
            assert_eq!(probe, host);
        }
    }
}
