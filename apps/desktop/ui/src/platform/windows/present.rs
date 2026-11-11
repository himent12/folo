//! Presenting with GDI. The window keeps one 32-bit top-down DIB section (`Surface`, the framebuffer's
//! `0x00RRGGBB` layout): `present` copies the damage rectangles of the framebuffer into it and
//! `BitBlt`s them to the window. `WM_PAINT` (an expose, or `PrintWindow`, which for a window wholly
//! off screen is the only way its pixels are ever seen) and `WM_PRINTCLIENT` repaint from it without
//! asking the core to draw again. No GPU.

use crate::{Bounds, DevicePixels, Framebuffer};
use ::windows::Win32::Foundation::RECT;
use ::windows::Win32::Graphics::Gdi::*;

/// The retained copy of the last presented frame, selected into a memory DC.
pub(crate) struct Surface {
    dc: HDC,
    bitmap: HBITMAP,
    previous: HGDIOBJ,
    bits: *mut u32,
    pub(crate) width: i32,
    pub(crate) height: i32,
}

impl Surface {
    /// An empty surface (no bitmap until the first present).
    pub(crate) fn new() -> Self {
        // SAFETY: a memory DC compatible with the screen, deleted in Drop.
        let dc = unsafe { CreateCompatibleDC(None) };
        Self { dc, bitmap: HBITMAP::default(), previous: HGDIOBJ::default(), bits: std::ptr::null_mut(), width: 0, height: 0 }
    }

    /// Makes the bitmap `width` x `height`; returns false if it had to be recreated (content lost).
    fn ensure(&mut self, width: i32, height: i32) -> bool {
        if width == self.width && height == self.height && !self.bits.is_null() {
            return true;
        }
        self.release_bitmap();
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width,
                biHeight: -height,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits = std::ptr::null_mut();
        // SAFETY: a top-down 32-bit DIB section owned by this surface and selected into its DC.
        if let Ok(bitmap) = unsafe { CreateDIBSection(Some(self.dc), &info, DIB_RGB_COLORS, &mut bits, None, 0) } {
            self.bitmap = bitmap;
            self.bits = bits as *mut u32;
            // SAFETY: selecting our bitmap into our DC; the previous object is put back on release.
            self.previous = unsafe { SelectObject(self.dc, bitmap.into()) };
            self.width = width;
            self.height = height;
        }
        false
    }

    fn release_bitmap(&mut self) {
        if !self.bitmap.is_invalid() {
            // SAFETY: deselecting and deleting our own bitmap.
            unsafe {
                SelectObject(self.dc, self.previous);
                let _ = DeleteObject(self.bitmap.into());
            }
        }
        self.bitmap = HBITMAP::default();
        self.bits = std::ptr::null_mut();
        self.width = 0;
        self.height = 0;
    }

    /// Copies `rects` of `frame` into the surface (all of it if the surface was just recreated) and
    /// then to `target` at the same place. Returns how many rectangles were shown.
    pub(crate) fn present(&mut self, target: HDC, frame: &Framebuffer, rects: &[Bounds<DevicePixels>]) -> usize {
        let (width, height) = (frame.width as i32, frame.height as i32);
        if frame.is_empty() || frame.pixels.len() < width as usize * height as usize {
            return 0;
        }
        let all = [Bounds::new(Default::default(), crate::size(DevicePixels(width), DevicePixels(height)))];
        let rects = if self.ensure(width, height) { rects } else { &all };
        if self.bits.is_null() {
            return 0;
        }
        // GDI may still be reading the bitmap from an earlier batched call.
        // SAFETY: no arguments.
        let _ = unsafe { GdiFlush() };
        let mut shown = 0;
        for rect in rects {
            let Some((x, y, w, h)) = clip(rect, width, height) else { continue };
            for row in y..y + h {
                let start = (row * width + x) as usize;
                let src = &frame.pixels[start..start + w as usize];
                // SAFETY: the DIB holds `width * height` pixels and `start..start + w` is inside a row.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.bits.add(start), w as usize) };
            }
            // SAFETY: both DCs are valid; the copy is clipped by the target's own region.
            if unsafe { BitBlt(target, x, y, w, h, Some(self.dc), x, y, SRCCOPY) }.is_ok() {
                shown += 1;
            }
        }
        shown
    }

    /// Repaints `area` of `target` from the surface. False if there is nothing of that size to show.
    pub(crate) fn paint(&self, target: HDC, area: RECT, width: i32, height: i32) -> bool {
        if self.bits.is_null() || self.width != width || self.height != height {
            return false;
        }
        let (x, y) = (area.left.max(0), area.top.max(0));
        let (w, h) = (area.right.min(width) - x, area.bottom.min(height) - y);
        if w > 0 && h > 0 {
            // SAFETY: both DCs are valid.
            let _ = unsafe { BitBlt(target, x, y, w, h, Some(self.dc), x, y, SRCCOPY) };
        }
        true
    }

    /// The surface's pixels (`0x00RRGGBB`, top-down), for tests.
    #[cfg(test)]
    fn pixels(&self) -> &[u32] {
        if self.bits.is_null() {
            return &[];
        }
        // SAFETY: the DIB holds `width * height` pixels while the bitmap lives.
        unsafe { std::slice::from_raw_parts(self.bits, (self.width * self.height) as usize) }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        self.release_bitmap();
        // SAFETY: our memory DC.
        let _ = unsafe { DeleteDC(self.dc) };
    }
}

/// The part of `rect` inside a `width` x `height` frame, as (x, y, w, h), or `None` if empty.
pub(crate) fn clip(rect: &Bounds<DevicePixels>, width: i32, height: i32) -> Option<(i32, i32, i32, i32)> {
    let x0 = rect.origin.x.0.clamp(0, width);
    let y0 = rect.origin.y.0.clamp(0, height);
    let x1 = (rect.origin.x.0 + rect.size.width.0).clamp(0, width);
    let y1 = (rect.origin.y.0 + rect.size.height.0).clamp(0, height);
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1 - x0, y1 - y0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn r(x: i32, y: i32, w: i32, h: i32) -> Bounds<DevicePixels> {
        Bounds::new(point(DevicePixels(x), DevicePixels(y)), size(DevicePixels(w), DevicePixels(h)))
    }

    #[test]
    fn rects_are_clipped_to_the_frame() {
        assert_eq!(clip(&r(-5, -5, 10, 10), 100, 50), Some((0, 0, 5, 5)));
        assert_eq!(clip(&r(90, 40, 64, 64), 100, 50), Some((90, 40, 10, 10)));
        assert_eq!(clip(&r(100, 0, 5, 5), 100, 50), None);
        assert_eq!(clip(&r(10, 10, 0, 5), 100, 50), None);
    }

    fn numbered(w: u32, h: u32, salt: u32) -> Framebuffer {
        let mut frame = Framebuffer::new(w, h);
        for (i, p) in frame.pixels.iter_mut().enumerate() {
            *p = (i as u32 * 0x0001_0203 + salt) & 0x00ff_ffff;
        }
        frame
    }

    /// A memory DC with its own top-down DIB, standing in for a window.
    struct Target {
        dc: HDC,
        bitmap: HBITMAP,
        bits: *mut u32,
        len: usize,
    }

    impl Target {
        fn new(w: i32, h: i32) -> Self {
            let mut s = Surface::new();
            s.ensure(w, h);
            let t = Target { dc: s.dc, bitmap: s.bitmap, bits: s.bits, len: (w * h) as usize };
            std::mem::forget(s);
            // SAFETY: filling our own DIB.
            unsafe { std::ptr::write_bytes(t.bits, 0xff, t.len) };
            t
        }
        fn pixels(&self) -> Vec<u32> {
            // SAFETY: flushing GDI, then reading our own DIB.
            unsafe {
                let _ = GdiFlush();
                std::slice::from_raw_parts(self.bits, self.len).to_vec()
            }
        }
    }

    impl Drop for Target {
        fn drop(&mut self) {
            // SAFETY: our own objects.
            unsafe {
                let _ = DeleteObject(self.bitmap.into());
                let _ = DeleteDC(self.dc);
            }
        }
    }

    #[test]
    fn the_first_present_is_whole_then_only_damage_is_copied() {
        let target = Target::new(4, 3);
        let mut surface = Surface::new();
        let first = numbered(4, 3, 0);
        // A new surface shows everything, whatever the damage says.
        assert_eq!(surface.present(target.dc, &first, &[r(0, 0, 1, 1)]), 1);
        assert_eq!(target.pixels(), first.pixels);
        assert_eq!(surface.pixels(), first.pixels.as_slice());
        // Then only the damage moves; the rest of the target keeps the first frame.
        let second = numbered(4, 3, 7);
        assert_eq!(surface.present(target.dc, &second, &[r(1, 1, 2, 2), r(9, 9, 1, 1)]), 1);
        let mut want = first.pixels.clone();
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            want[y * 4 + x] = second.pixels[y * 4 + x];
        }
        assert_eq!(target.pixels(), want);
        // An expose repaints from the surface alone.
        let other = Target::new(4, 3);
        assert!(surface.paint(other.dc, RECT { left: 0, top: 0, right: 4, bottom: 3 }, 4, 3));
        assert_eq!(other.pixels(), want);
        assert!(!surface.paint(other.dc, RECT { left: 0, top: 0, right: 4, bottom: 3 }, 5, 3));
    }
}
