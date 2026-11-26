//! Box-aware GridMask augmentation ([`grid_mask`], [`random_grid_mask`]).
//!
//! Masks a regular periodic grid of square cells across the image while computing exact visible area and tight
//! visible bounding boxes via analytical 1D×1D Cartesian-product interval overlap (no per-pixel mask allocation,
//! exact agreement with 2D pixel occlusion).

use super::{BoxFilter, ErasingFill};
use crate::bbox::BBox;
use crate::image::Sample;
use crate::letterbox::PAD_VALUE;
use crate::rng::Rng;
use crate::simd::assert_image_data;

/// Parameters for [`random_grid_mask`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridMaskConfig {
    /// Probability of applying GridMask (default `0.5`).
    pub prob: f32,
    /// Masked fraction of each grid period along each axis, in `(0, 1]` (default `0.5`, which masks
    /// `ratio² = 25%` of the image area).
    pub ratio: f32,
    /// Minimum grid spacing period `d` in pixels (default `32`).
    pub d_min: u32,
    /// Maximum grid spacing period `d` in pixels, inclusive (default `96`).
    pub d_max: u32,
    /// How masked grid cells are filled (default constant `[PAD_VALUE; 3]`).
    pub fill: ErasingFill,
}

impl Default for GridMaskConfig {
    fn default() -> Self {
        GridMaskConfig {
            prob: 0.5,
            ratio: 0.5,
            d_min: 32,
            d_max: 96,
            fill: ErasingFill::Constant([PAD_VALUE; 3]),
        }
    }
}

impl GridMaskConfig {
    /// Checks parameter ranges; returns a message for the first invalid field.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.prob) {
            return Err(format!("grid_mask.prob must be in [0, 1], got {}", self.prob));
        }
        if !(self.ratio.is_finite() && self.ratio > 0.0 && self.ratio <= 1.0) {
            return Err(format!("grid_mask.ratio must be in (0, 1], got {}", self.ratio));
        }
        if self.d_min == 0 || self.d_min > self.d_max {
            return Err(format!(
                "grid_mask.d_min ({}) must be >= 1 and <= d_max ({})",
                self.d_min, self.d_max
            ));
        }
        Ok(())
    }
}

/// Enumerates all non-empty masked 1D intervals `[a, b) ⊆ [0, len)` for a periodic grid of period `d`,
/// masked width `hole <= d`, and phase `offset < d`: a coordinate `p` is masked iff `(p - offset) mod d < hole`.
fn masked_intervals_1d(len: u32, d: u32, hole: u32, offset: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if len == 0 || d == 0 || hole == 0 {
        return out;
    }
    let hole = hole.min(d);
    if hole == d {
        out.push((0, len));
        return out;
    }
    let offset = offset % d;
    let len_i = len as i64;
    let d_i = d as i64;
    let hole_i = hole as i64;
    // Period `k = -1` starts at `offset - d < 0` and may wrap past 0 into `[0, offset - d + hole)`.
    let mut start = offset as i64 - d_i;
    while start < len_i {
        let a = start.max(0);
        let b = (start + hole_i).min(len_i);
        if b > a {
            out.push((a as u32, b as u32));
        }
        start += d_i;
    }
    out
}

/// Total 1D overlap of `[lo, hi]` with sorted disjoint intervals `intervals`, plus tight unmasked endpoints
/// `(unmasked_lo, unmasked_hi)` if any portion of `[lo, hi]` lies outside `intervals`.
fn interval_overlap_and_tight(lo: f32, hi: f32, intervals: &[(u32, u32)]) -> (f64, Option<(f32, f32)>) {
    let mut occ = 0.0f64;
    let mut tight_lo = f32::INFINITY;
    let mut tight_hi = f32::NEG_INFINITY;
    let mut cursor = lo;
    for &(ia, ib) in intervals {
        let a = ia as f32;
        let b = ib as f32;
        if b <= lo {
            continue;
        }
        if a >= hi {
            break;
        }
        let ca = a.max(lo);
        let cb = b.min(hi);
        if ca > cursor {
            tight_lo = tight_lo.min(cursor);
            tight_hi = tight_hi.max(ca);
        }
        if cb > ca {
            occ += (cb - ca) as f64;
        }
        cursor = cursor.max(cb);
    }
    if cursor < hi {
        tight_lo = tight_lo.min(cursor);
        tight_hi = tight_hi.max(hi);
    }
    let tight = (tight_hi > tight_lo).then_some((tight_lo, tight_hi));
    (occ, tight)
}

/// Filters and tightens a single box against a regular Cartesian product `xs × ys` of masked intervals.
fn filter_grid_occluded_box(
    bx: &BBox,
    xs: &[(u32, u32)],
    ys: &[(u32, u32)],
    width: u32,
    height: u32,
    filter: &BoxFilter,
) -> Option<BBox> {
    if width == 0 || height == 0 {
        return None;
    }
    let [raw_x1, raw_y1, raw_x2, raw_y2] = bx.pixel_xyxy(width, height);
    if !(raw_x1.is_finite() && raw_y1.is_finite() && raw_x2.is_finite() && raw_y2.is_finite()) {
        return None;
    }
    if !(raw_x2 > raw_x1 && raw_y2 > raw_y1) {
        return None;
    }
    let (fw, fh) = (width as f32, height as f32);
    let [x1, y1, x2, y2] =
        [raw_x1.clamp(0.0, fw), raw_y1.clamp(0.0, fh), raw_x2.clamp(0.0, fw), raw_y2.clamp(0.0, fh)];
    let (bw, bh) = (x2 - x1, y2 - y1);
    if !(bw > 0.0 && bh > 0.0) {
        return None;
    }
    let orig_area = bw as f64 * bh as f64;
    let (occ_x, tight_x) = interval_overlap_and_tight(x1, x2, xs);
    let (occ_y, tight_y) = interval_overlap_and_tight(y1, y2, ys);
    let occ_area = occ_x * occ_y;
    if occ_area == 0.0 {
        return Some(*bx);
    }
    let visible_area = (orig_area - occ_area).max(0.0);
    if visible_area <= 0.0 {
        return None;
    }
    // If every row of the box is masked (`tight_y.is_none()`), visible X is restricted to `tight_x`;
    // otherwise at least one unmasked row spans the full width `[x1, x2]`. Symmetrically for Y.
    let (vx1, vx2) = match tight_y {
        Some(_) => (x1, x2),
        None => tight_x?,
    };
    let (vy1, vy2) = match tight_x {
        Some(_) => (y1, y2),
        None => tight_y?,
    };
    if visible_area < filter.min_area_ratio as f64 * orig_area {
        return None;
    }
    filter
        .keep(vx2 - vx1, vy2 - vy1, orig_area as f32)
        .then(|| BBox::from_pixel_xyxy(bx.class, [vx1, vy1, vx2, vy2], width, height))
}

/// Applies a deterministic periodic GridMask with period `d`, masked square side `hole`, and phase
/// `(offset_x, offset_y)` to `sample`, updating/dropping boxes via exact Cartesian-product occlusion filtering.
///
/// Returns the number of masked cells written to the image (`0` for empty images, `d == 0`, or `hole == 0`).
#[allow(clippy::too_many_arguments)]
pub fn grid_mask(
    sample: &mut Sample,
    d: u32,
    hole: u32,
    offset_x: u32,
    offset_y: u32,
    fill: ErasingFill,
    filter: &BoxFilter,
    rng: &mut Rng,
) -> usize {
    assert_image_data(&sample.image);
    let (w, h) = (sample.image.width, sample.image.height);
    if w == 0 || h == 0 || d == 0 || hole == 0 {
        return 0;
    }
    let xs = masked_intervals_1d(w, d, hole, offset_x);
    let ys = masked_intervals_1d(h, d, hole, offset_y);
    if xs.is_empty() || ys.is_empty() {
        return 0;
    }
    let stride = sample.image.stride();
    for &(y1, y2) in &ys {
        for &(x1, x2) in &xs {
            match fill {
                ErasingFill::Constant(rgb) => {
                    for y in y1 as usize..y2 as usize {
                        let row = &mut sample.image.data
                            [y * stride + x1 as usize * 3..y * stride + x2 as usize * 3];
                        if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
                            row.fill(rgb[0]);
                        } else {
                            for px in row.as_chunks_mut::<3>().0.iter_mut() {
                                *px = rgb;
                            }
                        }
                    }
                }
                ErasingFill::RandomNoise => {
                    for y in y1 as usize..y2 as usize {
                        let row = &mut sample.image.data
                            [y * stride + x1 as usize * 3..y * stride + x2 as usize * 3];
                        for byte in row.iter_mut() {
                            *byte = (rng.next_u64() >> 56) as u8;
                        }
                    }
                }
            }
        }
    }
    sample.boxes =
        sample.boxes.iter().filter_map(|b| filter_grid_occluded_box(b, &xs, &ys, w, h, filter)).collect();
    xs.len() * ys.len()
}

/// Randomly applies [`grid_mask`] according to `cfg`, returning `true` when applied.
pub fn random_grid_mask(
    sample: &mut Sample,
    cfg: &GridMaskConfig,
    filter: &BoxFilter,
    rng: &mut Rng,
) -> bool {
    if cfg.validate().is_err() || !rng.chance(cfg.prob) {
        return false;
    }
    if sample.image.width == 0 || sample.image.height == 0 {
        return false;
    }
    let span = (cfg.d_max - cfg.d_min) as usize + 1;
    let d = cfg.d_min + rng.below(span) as u32;
    let hole = ((d as f64 * cfg.ratio as f64).round() as u32).clamp(1, d);
    let offset_x = rng.below(d as usize) as u32;
    let offset_y = rng.below(d as usize) as u32;
    grid_mask(sample, d, hole, offset_x, offset_y, cfg.fill, filter, rng) > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;

    /// Brute-force pixel-level occlusion oracle for verifying exact Cartesian-product box filtering.
    fn pixel_occlusion_oracle(
        bx: &BBox,
        w: u32,
        h: u32,
        d: u32,
        hole: u32,
        ox: u32,
        oy: u32,
    ) -> (u32, Option<[u32; 4]>) {
        let [x1, y1, x2, y2] = bx.pixel_xyxy(w, h);
        let ix1 = x1.round().clamp(0.0, w as f32) as u32;
        let iy1 = y1.round().clamp(0.0, h as f32) as u32;
        let ix2 = x2.round().clamp(0.0, w as f32) as u32;
        let iy2 = y2.round().clamp(0.0, h as f32) as u32;
        let is_masked = |p: u32, off: u32| -> bool {
            let r = p % d;
            let o = off % d;
            let dist = if r >= o { r - o } else { r + d - o };
            dist < hole.min(d)
        };
        let mut vis = 0u32;
        let (mut vx1, mut vy1, mut vx2, mut vy2) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in iy1..iy2 {
            let my = is_masked(y, oy);
            for x in ix1..ix2 {
                if !(my && is_masked(x, ox)) {
                    vis += 1;
                    vx1 = vx1.min(x);
                    vy1 = vy1.min(y);
                    vx2 = vx2.max(x + 1);
                    vy2 = vy2.max(y + 1);
                }
            }
        }
        (vis, (vis > 0).then_some([vx1, vy1, vx2, vy2]))
    }

    #[test]
    fn grid_mask_matches_pixel_oracle_for_visible_area_and_tight_boxes() {
        let (w, h) = (64u32, 48u32);
        let filter = BoxFilter { min_px: 1.0, min_area_ratio: 0.25, max_aspect: 100.0 };
        // Integer-aligned boxes covering disjoint, partially occluded, strip-trimmed, and fully occluded cases.
        let boxes = vec![
            BBox::from_pixel_xyxy(0, [2.0, 2.0, 10.0, 10.0], w, h),
            BBox::from_pixel_xyxy(1, [16.0, 16.0, 28.0, 28.0], w, h),
            BBox::from_pixel_xyxy(2, [4.0, 16.0, 40.0, 24.0], w, h),
            BBox::from_pixel_xyxy(3, [16.0, 4.0, 24.0, 40.0], w, h),
            BBox::from_pixel_xyxy(4, [0.0, 0.0, 64.0, 48.0], w, h),
        ];
        for &(d, hole, ox, oy) in
            &[(16u32, 8u32, 0u32, 0u32), (16, 10, 12, 14), (20, 12, 5, 17), (16, 16, 3, 7)]
        {
            let mut s = Sample { image: Image::new(w, h, [200, 100, 50]), boxes: boxes.clone() };
            grid_mask(&mut s, d, hole, ox, oy, ErasingFill::Constant([0, 0, 0]), &filter, &mut Rng::new(7));
            let expected: Vec<BBox> = boxes
                .iter()
                .filter_map(|b| {
                    let [x1, y1, x2, y2] = b.pixel_xyxy(w, h);
                    let orig_area = (x2 - x1) * (y2 - y1);
                    let (vis, tight) = pixel_occlusion_oracle(b, w, h, d, hole, ox, oy);
                    let [vx1, vy1, vx2, vy2] = tight?;
                    if (vis as f32) < filter.min_area_ratio * orig_area {
                        return None;
                    }
                    let tw = (vx2 - vx1) as f32;
                    let th = (vy2 - vy1) as f32;
                    filter.keep(tw, th, orig_area).then(|| {
                        BBox::from_pixel_xyxy(b.class, [vx1 as f32, vy1 as f32, vx2 as f32, vy2 as f32], w, h)
                    })
                })
                .collect();
            assert_eq!(s.boxes.len(), expected.len(), "d={d} hole={hole} ox={ox} oy={oy}");
            for (g, e) in s.boxes.iter().zip(&expected) {
                assert_eq!(g.class, e.class);
                assert!((g.cx - e.cx).abs() < 1e-6 && (g.cy - e.cy).abs() < 1e-6);
                assert!((g.w - e.w).abs() < 1e-6 && (g.h - e.h).abs() < 1e-6);
            }
            // Verify actual painted pixels match the periodic formula.
            for y in 0..h {
                for x in 0..w {
                    let rx = x % d;
                    let ry = y % d;
                    let mx = (if rx >= ox % d { rx - ox % d } else { rx + d - ox % d }) < hole.min(d);
                    let my = (if ry >= oy % d { ry - oy % d } else { ry + d - oy % d }) < hole.min(d);
                    let want = if mx && my { [0, 0, 0] } else { [200, 100, 50] };
                    assert_eq!(s.image.pixel(x, y), want, "pixel ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn grid_mask_determinism_noise_and_degenerate_inputs() {
        let base = Sample {
            image: Image::new(40, 32, [100, 150, 200]),
            boxes: vec![
                BBox::new(0, 0.5, 0.5, 0.5, 0.5),
                BBox::new(1, f32::NAN, 0.5, 0.2, 0.2),
                BBox::new(2, 0.5, 0.5, 0.0, 0.2),
            ],
        };
        let cfg =
            GridMaskConfig { prob: 1.0, ratio: 0.5, d_min: 10, d_max: 20, fill: ErasingFill::RandomNoise };
        assert!(cfg.validate().is_ok());
        let filter = BoxFilter::default();
        let (mut a, mut b) = (base.clone(), base.clone());
        assert!(random_grid_mask(&mut a, &cfg, &filter, &mut Rng::new(42)));
        assert!(random_grid_mask(&mut b, &cfg, &filter, &mut Rng::new(42)));
        assert_eq!(a.image.data, b.image.data);
        assert_eq!(a.boxes, b.boxes);
        // NaN and zero-width boxes must be dropped.
        assert!(a.boxes.iter().all(|bx| bx.class == 0));

        // Empty image and invalid configs are safe no-ops.
        let mut empty = Sample { image: Image { width: 0, height: 0, data: Vec::new() }, boxes: vec![] };
        assert!(!random_grid_mask(&mut empty, &cfg, &filter, &mut Rng::new(1)));
        assert!(GridMaskConfig { ratio: 0.0, ..Default::default() }.validate().is_err());
        assert!(GridMaskConfig { ratio: 1.2, ..Default::default() }.validate().is_err());
        assert!(GridMaskConfig { d_min: 0, ..Default::default() }.validate().is_err());
        assert!(GridMaskConfig { d_min: 40, d_max: 20, ..Default::default() }.validate().is_err());
    }
}
