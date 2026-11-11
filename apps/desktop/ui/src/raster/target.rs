//! Where a tile draws: a window of the framebuffer that one thread owns while it rasterizes.

use super::geom::IRect;
use std::marker::PhantomData;

/// The framebuffer's pixels as a raw base pointer, shared by the threads that draw disjoint tiles.
#[derive(Clone, Copy)]
pub(crate) struct FrameMut {
    base: *mut u32,
    stride: usize,
    len: usize,
}

// SAFETY: `FrameMut` is only turned into `Target`s for disjoint tile rectangles (`Renderer::render`
// hands each dirty tile to exactly one thread), so no two threads touch the same pixel.
unsafe impl Send for FrameMut {}
// SAFETY: as above; sharing the pointer is sound because every write goes to a tile one thread owns.
unsafe impl Sync for FrameMut {}

impl FrameMut {
    pub fn new(pixels: &mut [u32], width: u32) -> Self {
        FrameMut { base: pixels.as_mut_ptr(), stride: width as usize, len: pixels.len() }
    }

    /// A target for `rect`.
    ///
    /// # Safety
    /// No other live `Target` overlaps `rect`. (That `rect` lies inside the frame is checked: every
    /// write through the target relies on it.)
    pub unsafe fn target(&self, rect: IRect) -> Target<'_> {
        assert!(
            rect.x0 >= 0
                && rect.y0 >= 0
                && rect.x1 as usize <= self.stride
                && (rect.y1.max(0) as usize).saturating_mul(self.stride) <= self.len,
            "tile {rect:?} outside the frame"
        );
        Target { frame: *self, rect, _owner: PhantomData }
    }
}

/// One tile's pixels.
pub(crate) struct Target<'a> {
    frame: FrameMut,
    /// The tile (device px); only pixels inside it may be written.
    pub rect: IRect,
    _owner: PhantomData<&'a mut [u32]>,
}

impl Target<'_> {
    /// Pixels `x0..x1` of row `y`, which must lie inside `rect`.
    #[inline(always)]
    pub fn row(&mut self, y: i32, x0: i32, x1: i32) -> &mut [u32] {
        assert!(
            y >= self.rect.y0 && y < self.rect.y1 && x0 >= self.rect.x0 && x1 <= self.rect.x1 && x0 <= x1,
            "row {y} [{x0}, {x1}) outside tile {:?}",
            self.rect
        );
        let start = y as usize * self.frame.stride + x0 as usize;
        // SAFETY: the span lies inside `rect` (checked above), which lies inside the frame and is owned
        // by this target alone (`FrameMut::target`); `&mut self` keeps spans of one target from aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.frame.base.add(start), (x1 - x0) as usize) }
    }

    /// Fills the whole tile with `pixel`.
    pub fn fill(&mut self, pixel: u32) {
        let r = self.rect;
        for y in r.y0..r.y1 {
            self.row(y, r.x0, r.x1).fill(pixel);
        }
    }
}
