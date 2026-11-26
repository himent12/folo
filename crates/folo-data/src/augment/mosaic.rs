//! 4-image mosaic and mixup.

use super::AugmentConfig;
use super::geometry::random_affine_plane;
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::warp::{Plane, PlaneTiles, Tile};

/// Tiles four samples around a random centre on a `2s x 2s` grey canvas (YOLO's mosaic).
///
/// The centre is drawn from `[s/2, 3s/2)` in both axes; image 0 goes top-left of it, 1 top-right, 2 bottom-left and
/// 3 bottom-right, each cropped by the canvas edge. Inputs are normally pre-resized so their long side is `s`. Boxes
/// are offset and clipped to the canvas; follow with an affine (as [`crate::TrainTransform`] does) to get back to
/// `s x s` and to drop boxes that became too small.
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
