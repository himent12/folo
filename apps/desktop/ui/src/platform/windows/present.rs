//! Presenting: the framebuffer's damage rectangles go to a DC with `SetDIBitsToDevice` (a top-down
//! 32-bit `BI_RGB` DIB is exactly the framebuffer's `0x00RRGGBB` layout). No GPU, and no copy of the
//! frame is kept here: the core keeps its framebuffer, and an expose asks it to present again.
//!
//! Measured on the dev machine (3840x2160 display), this beats keeping a DIB section and `BitBlt`ing
//! from it: 0.69 ms against 7.8 ms for a whole 1920x1200 frame, 2.6 ms against 6.2 ms at 3840x2160.

use crate::{Bounds, DevicePixels, Framebuffer};
use ::windows::Win32::Graphics::Gdi::*;

/// Copies `rects` of `frame` (device pixels, clipped to the frame) to `hdc` at the same place.
/// Returns how many rectangles were copied.
pub(crate) fn blit(hdc: HDC, frame: &Framebuffer, rects: &[Bounds<DevicePixels>]) -> usize {
    if frame.is_empty() || frame.pixels.len() < frame.width as usize * frame.height as usize {
        return 0;
    }
    let mut copied = 0;
    for rect in rects {
        let Some((x, y, w, h)) = clip(rect, frame.width as i32, frame.height as i32) else { continue };
        // A band of `h` whole rows starting at row `y`: the DIB is `width` x `h`, top-down, and the copy
        // takes columns `x..x + w` of all its rows, so no scan-line origin rules come into play.
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: frame.width as i32,
                biHeight: -h,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let band = &frame.pixels[y as usize * frame.width as usize..];
        // SAFETY: `band` holds at least `h` rows of `width` pixels (checked by `clip`), the header
        // describes exactly that, and the DC belongs to a live window or bitmap.
        let lines = unsafe {
            SetDIBitsToDevice(
                hdc,
                x,
                y,
                w as u32,
                h as u32,
                x,
                0,
                0,
                h as u32,
                band.as_ptr() as *const _,
                &info,
                DIB_RGB_COLORS,
            )
        };
        if lines > 0 {
            copied += 1;
        }
    }
    // Nothing stays in GDI's batch: a capture or the compositor reads the window right after.
    // SAFETY: no arguments.
    let _ = unsafe { GdiFlush() };
    copied
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

    #[test]
    fn blit_copies_only_the_damage() {
        // A 4x3 frame with a distinct value per pixel, copied in two rectangles (one partly outside)
        // to a memory DC's top-down DIB section.
        let mut frame = Framebuffer::new(4, 3);
        for (i, p) in frame.pixels.iter_mut().enumerate() {
            *p = (0x0010_0000 * i as u32 + i as u32) & 0x00ff_ffff;
        }
        // SAFETY: a memory DC and a DIB section of the frame's size, read back and freed here.
        unsafe {
            let dc = CreateCompatibleDC(None);
            let info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: 4,
                    biHeight: -3,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut bits = std::ptr::null_mut();
            let bitmap = CreateDIBSection(Some(dc), &info, DIB_RGB_COLORS, &mut bits, None, 0).unwrap();
            let old = SelectObject(dc, bitmap.into());
            std::ptr::write_bytes(bits as *mut u32, 0xff, 12);
            assert_eq!(blit(dc, &frame, &[r(1, 1, 2, 2), r(3, 2, 5, 5), r(9, 9, 1, 1)]), 2);
            let _ = GdiFlush();
            let got: Vec<u32> =
                std::slice::from_raw_parts(bits as *const u32, 12).iter().map(|p| p & 0x00ff_ffff).collect();
            SelectObject(dc, old);
            let _ = DeleteObject(bitmap.into());
            let _ = DeleteDC(dc);
            let mut want = vec![0x00ff_ffffu32; 12];
            for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)] {
                want[y * 4 + x] = frame.pixels[y * 4 + x];
            }
            assert_eq!(got, want);
        }
    }
}
