//! Integer pixel rectangles and the GPU's pixel-coverage rules for edges and clips.

use crate::{Bounds, ScaledPixels};

/// A rectangle of whole device pixels, `x0..x1` by `y0..y1` (exclusive ends).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct IRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl IRect {
    pub const EMPTY: IRect = IRect { x0: 0, y0: 0, x1: 0, y1: 0 };

    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    pub fn intersect(&self, o: &IRect) -> IRect {
        IRect { x0: self.x0.max(o.x0), y0: self.y0.max(o.y0), x1: self.x1.min(o.x1), y1: self.y1.min(o.y1) }
    }

    /// Whether `o` lies inside `self` (an empty `self` contains nothing).
    pub fn contains(&self, o: &IRect) -> bool {
        !self.is_empty() && self.x0 <= o.x0 && self.y0 <= o.y0 && self.x1 >= o.x1 && self.y1 >= o.y1
    }

    pub fn width(&self) -> i32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> i32 {
        self.y1 - self.y0
    }
}

/// `v.ceil() as i32` without a libm call (SSE2 has no rounding instruction); saturates like `as`.
#[inline(always)]
pub(crate) fn ceil_i32(v: f32) -> i32 {
    let i = v as i32;
    if (i as f32) < v { i.saturating_add(1) } else { i }
}

/// `v.floor() as i32`, likewise.
#[inline(always)]
pub(crate) fn floor_i32(v: f32) -> i32 {
    let i = v as i32;
    if (i as f32) > v { i.saturating_sub(1) } else { i }
}

/// The first pixel whose centre `i + 0.5` is at or right of the edge `v`, with `v` first snapped to the
/// rasterizer's 1/256 px grid (D3D11 converts vertex positions, clipped ones included, to 16.8 fixed
/// point before the fill rule; nearest, ties up).
#[inline(always)]
fn first_centre_at_or_after(v: f32) -> i32 {
    // Far outside any frame: clamp so the fixed point fits i32 (NaN stays NaN and becomes 0).
    const LIMIT: f32 = (1 << 22) as f32;
    let k = floor_i32(v.clamp(-LIMIT, LIMIT) * 256. + 0.5);
    // ceil((k - 128) / 256): `>>` floors, also for negative values.
    (k + 127) >> 8
}

/// The pixels a GPU rasterizes for an edge from `lo` to `hi`: those whose centre `i + 0.5` lies in
/// `[lo, hi)` (the top-left fill rule), the edges snapped to 1/256 px as the rasterizer does.
#[inline]
pub(crate) fn raster_span(lo: f32, hi: f32) -> (i32, i32) {
    (first_centre_at_or_after(lo), first_centre_at_or_after(hi))
}

/// The pixels a clip rectangle keeps. GPUI clips with `SV_ClipDistance`, which clips the triangles
/// themselves, so the clipped edges are rasterized like any other: a centre exactly on the left or
/// top edge is kept, one exactly on the right or bottom edge is not (the top-left fill rule). A 1.5 px
/// border side clipped to `[384, 385.5)` covers pixel 384 only, as in the GPUI captures.
#[inline]
pub(crate) fn clip_span(lo: f32, hi: f32) -> (i32, i32) {
    raster_span(lo, hi)
}

/// The pixels a primitive drawn over `bounds` and clipped by `clip` can touch (what `items::Clips`
/// computes, keeping the clip's part).
#[cfg(test)]
pub(crate) fn pixel_rect(bounds: &Bounds<ScaledPixels>, clip: &Bounds<ScaledPixels>) -> IRect {
    let (bx0, bx1) = raster_span(bounds.origin.x.0, bounds.origin.x.0 + bounds.size.width.0);
    let (by0, by1) = raster_span(bounds.origin.y.0, bounds.origin.y.0 + bounds.size.height.0);
    let (cx0, cx1) = clip_span(clip.origin.x.0, clip.origin.x.0 + clip.size.width.0);
    let (cy0, cy1) = clip_span(clip.origin.y.0, clip.origin.y.0 + clip.size.height.0);
    IRect::new(bx0.max(cx0), by0.max(cy0), bx1.min(cx1), by1.min(cy1))
}

/// The pixel rectangle of a clip alone.
pub(crate) fn clip_rect(clip: &Bounds<ScaledPixels>) -> IRect {
    let (cx0, cx1) = clip_span(clip.origin.x.0, clip.origin.x.0 + clip.size.width.0);
    let (cy0, cy1) = clip_span(clip.origin.y.0, clip.origin.y.0 + clip.size.height.0);
    IRect::new(cx0, cy0, cx1, cy1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_follow_pixel_centres() {
        assert_eq!(raster_span(10., 20.), (10, 20));
        // Centre 10.5 is on the left edge (kept), centre 20.5 on the right edge (dropped).
        assert_eq!(raster_span(10.5, 20.5), (10, 20));
        assert_eq!(raster_span(10.6, 20.4), (11, 20));
        assert_eq!(clip_span(10., 20.), (10, 20));
        // A clip keeps a centre exactly on its left edge and drops one on its right edge.
        assert_eq!(clip_span(10.5, 20.5), (10, 20));
        assert_eq!(clip_span(384., 385.5), (384, 385));
        assert_eq!(clip_span(1702.5, 1704.), (1702, 1704));
        // Edges snap to 1/256 px first: a clip computed as 1553.5001 (float error in the layout
        // arithmetic) is 1553.5 to the rasterizer and keeps pixel 1553, as the GPUI captures show.
        assert_eq!(clip_span(1553.5001, 1555.), (1553, 1555));
        assert_eq!(raster_span(10.5 + 1. / 512. + 1e-4, 20.5 - 1. / 600.), (11, 20));
        assert_eq!(raster_span(284.999_97, 318.), (285, 318));
        assert_eq!(raster_span(-0.5, -10.5), (-1, -11));
        assert_eq!(raster_span(f32::NAN, 1e12), (0, 1 << 22));
        assert!(IRect::new(0, 0, 4, 4).contains(&IRect::new(1, 1, 4, 4)));
        assert!(!IRect::EMPTY.contains(&IRect::EMPTY));
        for v in [0., 0.5, -0.5, 1.0, -1.0, 2.25, -2.25, 1e9, -1e9, 3e10, f32::MIN_POSITIVE, -0.0] {
            assert_eq!(ceil_i32(v), v.ceil() as i32, "{v}");
            assert_eq!(floor_i32(v), v.floor() as i32, "{v}");
        }
    }
}
