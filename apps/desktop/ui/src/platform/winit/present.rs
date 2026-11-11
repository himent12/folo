//! Presenting: copying a framebuffer's damage rectangles into softbuffer's buffer (pure, so it is
//! testable without a window), and the `FOLO_UI_TRACE` first-present line.

use crate::{Bounds, DevicePixels};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};

/// A damage rectangle clamped to the buffer, in device pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// softbuffer's rectangle (`None` when empty).
    pub fn to_softbuffer(self) -> Option<softbuffer::Rect> {
        Some(softbuffer::Rect {
            x: self.x,
            y: self.y,
            width: NonZeroU32::new(self.width)?,
            height: NonZeroU32::new(self.height)?,
        })
    }
}

/// `damage` clamped to a `width` x `height` buffer, empty rectangles dropped.
pub(crate) fn clamp_damage(damage: &[Bounds<DevicePixels>], width: u32, height: u32) -> Vec<Rect> {
    damage
        .iter()
        .filter_map(|bounds| {
            let x0 = bounds.origin.x.0.clamp(0, width as i32) as u32;
            let y0 = bounds.origin.y.0.clamp(0, height as i32) as u32;
            let x1 = (bounds.origin.x.0.saturating_add(bounds.size.width.0)).clamp(0, width as i32) as u32;
            let y1 = (bounds.origin.y.0.saturating_add(bounds.size.height.0)).clamp(0, height as i32) as u32;
            (x1 > x0 && y1 > y0).then_some(Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 })
        })
        .collect()
}

/// Copies `rects` of `src` into `dst` (both `width` pixels wide, `0x00RRGGBB`, the layout softbuffer
/// uses).
pub(crate) fn copy_rects(src: &[u32], dst: &mut [u32], width: u32, rects: &[Rect]) {
    let w = width as usize;
    for rect in rects {
        let (x0, x1) = (rect.x as usize, (rect.x + rect.width) as usize);
        for y in rect.y as usize..(rect.y + rect.height) as usize {
            dst[y * w + x0..y * w + x1].copy_from_slice(&src[y * w + x0..y * w + x1]);
        }
    }
}

/// Prints `first-present <ms since process creation>` to stderr at the first present of the process,
/// when `FOLO_UI_TRACE=1`.
pub(crate) fn trace_first_present() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::Relaxed) || std::env::var_os("FOLO_UI_TRACE").is_none_or(|v| v != "1") {
        return;
    }
    match super::process::age() {
        Some(age) => eprintln!("first-present {:.1}", age.as_secs_f64() * 1e3),
        None => eprintln!("first-present ?"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn bounds(x: i32, y: i32, w: i32, h: i32) -> Bounds<DevicePixels> {
        Bounds::new(point(DevicePixels(x), DevicePixels(y)), size(DevicePixels(w), DevicePixels(h)))
    }

    #[test]
    fn damage_is_clamped_and_copied_exactly() {
        let rects = clamp_damage(&[bounds(-2, 1, 4, 2), bounds(3, 3, 10, 10), bounds(1, 1, 0, 3)], 4, 4);
        assert_eq!(
            rects,
            [Rect { x: 0, y: 1, width: 2, height: 2 }, Rect { x: 3, y: 3, width: 1, height: 1 }]
        );
        let src: Vec<u32> = (0..16).collect();
        let mut dst = vec![99; 16];
        copy_rects(&src, &mut dst, 4, &rects);
        #[rustfmt::skip]
        assert_eq!(dst, [
            99, 99, 99, 99,
             4,  5, 99, 99,
             8,  9, 99, 99,
            99, 99, 99, 15,
        ]);
        assert!(Rect { x: 0, y: 0, width: 0, height: 1 }.to_softbuffer().is_none());
        assert_eq!(rects[0].to_softbuffer().unwrap().width.get(), 2);
    }
}
