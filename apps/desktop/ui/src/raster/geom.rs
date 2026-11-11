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

/// The pixels a GPU rasterizes for an edge from `lo` to `hi`: those whose centre `i + 0.5` lies in
/// `[lo, hi)` (the top-left fill rule).
#[inline]
pub(crate) fn raster_span(lo: f32, hi: f32) -> (i32, i32) {
    (ceil_i32(lo - 0.5), ceil_i32(hi - 0.5))
}

/// The pixels a clip rectangle keeps: those whose centre has a clip distance `>= 0` to both edges.
#[inline]
pub(crate) fn clip_span(lo: f32, hi: f32) -> (i32, i32) {
    (ceil_i32(lo - 0.5), floor_i32(hi - 0.5).saturating_add(1))
}

/// The pixels a primitive drawn over `bounds` and clipped by `clip` can touch.
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
        // A clip keeps a centre exactly on either edge.
        assert_eq!(clip_span(10.5, 20.5), (10, 21));
        assert!(IRect::new(0, 0, 4, 4).contains(&IRect::new(1, 1, 4, 4)));
        assert!(!IRect::EMPTY.contains(&IRect::EMPTY));
        for v in [0., 0.5, -0.5, 1.0, -1.0, 2.25, -2.25, 1e9, -1e9, 3e10, f32::MIN_POSITIVE, -0.0] {
            assert_eq!(ceil_i32(v), v.ceil() as i32, "{v}");
            assert_eq!(floor_i32(v), v.floor() as i32, "{v}");
        }
    }
}
