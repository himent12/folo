//! Tests of the geometry other code trusts blindly.

use super::*;

fn b(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
    Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
}

#[test]
fn intersect_union_contains() {
    let a = b(0., 0., 10., 10.);
    let c = b(5., 5., 10., 10.);
    assert_eq!(a.intersect(&c), b(5., 5., 5., 5.));
    assert_eq!(a.union(&c), b(0., 0., 15., 15.));
    assert!(a.intersects(&c));
    // Disjoint bounds intersect to an empty box.
    let far = b(20., 20., 5., 5.);
    assert!(a.intersect(&far).is_empty());
    assert!(!a.intersects(&far));
    // `contains` includes every edge, as GPUI's does.
    assert!(a.contains(&point(px(0.), px(0.))));
    assert!(a.contains(&point(px(10.), px(10.))));
    assert!(!a.contains(&point(px(10.1), px(5.))));
    assert!(!a.contains(&point(px(5.), px(-0.1))));
}

#[test]
fn dilate_inset_center_corners() {
    let a = b(10., 10., 20., 10.);
    assert_eq!(a.dilate(px(2.)), b(8., 8., 24., 14.));
    assert_eq!(a.inset(px(2.)), b(12., 12., 16., 6.));
    assert_eq!(a.center(), point(px(20.), px(15.)));
    assert_eq!(a.bottom_right(), point(px(30.), px(20.)));
    assert_eq!(Bounds::centered_at(point(px(20.), px(15.)), a.size), a);
    assert_eq!(Bounds::from_corners(a.origin, a.bottom_right()), a);
}

#[test]
fn scale_and_device_rounding() {
    let a = b(1., 2., 3., 4.);
    let s = a.scale(1.5);
    assert_eq!(s.origin, point(ScaledPixels(1.5), ScaledPixels(3.)));
    assert_eq!(s.size, size(ScaledPixels(4.5), ScaledPixels(6.)));
    assert_eq!(px(2.5).round(), px(3.));
    assert_eq!(DevicePixels::from(ScaledPixels(2.6)), DevicePixels(3));
    assert_eq!(relative(0.5), DefiniteLength::Fraction(0.5));
    assert_eq!(rems(2.).to_pixels(px(16.)), px(32.));
}
