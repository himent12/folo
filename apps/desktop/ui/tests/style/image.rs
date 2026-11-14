//! Images (`image.rs`): a `RenderImage` holds BGRA frames with unique ids, and each `ObjectFit` places an image
//! as GPUI does.

use folo_ui::{Bounds, DevicePixels, ObjectFit, Pixels, RenderImage, point, px, size};
use std::time::Duration;

fn frame(w: u32, h: u32, ms: u32) -> image::Frame {
    let mut buf = image::RgbaImage::new(w, h);
    buf.put_pixel(0, 0, image::Rgba([1, 2, 3, 4]));
    image::Frame::from_parts(buf, 0, 0, image::Delay::from_numer_denom_ms(ms, 1))
}

#[test]
fn render_images_hold_bgra_frames_with_unique_ids() {
    let a = RenderImage::new(vec![frame(3, 2, 40), frame(3, 2, 60)]);
    let b = RenderImage::new(vec![frame(1, 1, 0)]);
    assert_ne!(a.id, b.id);
    assert_eq!((a.frame_count(), a.size(1)), (2, size(DevicePixels(3), DevicePixels(2))));
    assert_eq!(a.render_size(0), size(px(3.), px(2.)));
    assert_eq!(&a.as_bytes(0).unwrap()[..4], &[1, 2, 3, 4], "bytes as given (the app swaps to BGRA)");
    assert!(a.as_bytes(2).is_none());
    assert_eq!(Duration::from(a.delay(1)), Duration::from_millis(60));
}

#[test]
fn object_fit_places_an_image_as_gpui_does() {
    let bounds: Bounds<Pixels> = Bounds::new(point(px(0.), px(0.)), size(px(200.), px(100.)));
    let image = size(DevicePixels(50), DevicePixels(50));
    let at = |x: f32, y: f32, w: f32, h: f32| Bounds::new(point(px(x), px(y)), size(px(w), px(h)));
    assert_eq!(ObjectFit::Fill.get_bounds(bounds, image), bounds);
    assert_eq!(ObjectFit::Contain.get_bounds(bounds, image), at(50., 0., 100., 100.));
    assert_eq!(ObjectFit::Cover.get_bounds(bounds, image), at(0., -50., 200., 200.));
    assert_eq!(ObjectFit::ScaleDown.get_bounds(bounds, image), at(75., 25., 50., 50.));
    assert_eq!(ObjectFit::None.get_bounds(bounds, image), at(0., 0., 50., 50.));
}
