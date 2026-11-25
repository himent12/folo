//! Paths (`path_builder.rs`): lyon's strokes and fills, dashes, curve flattening, and `paint_path` scaling,
//! clipping and fading.

use folo_ui::{
    Bounds, BuiltPath, ContentMask, PathBuilder, PathKind, Pixels, Point, Primitive, Scene, StylePaint,
    linear_color_stop, linear_gradient, paint_path, point, px, red, size,
};

fn p(x: f32, y: f32) -> Point<Pixels> {
    point(px(x), px(y))
}

fn polyline(points: &[(f32, f32)], width: f32, dash: &[f32]) -> BuiltPath {
    let mut b = PathBuilder::stroke(px(width));
    if !dash.is_empty() {
        b = b.dash_array(&dash.iter().map(|d| px(*d)).collect::<Vec<_>>());
    }
    b.move_to(p(points[0].0, points[0].1));
    for &(x, y) in &points[1..] {
        b.line_to(p(x, y));
    }
    b.build().unwrap()
}

/// The area the triangles cover, counting overlaps (lyon's fills do not overlap).
fn area(path: &BuiltPath) -> f32 {
    path.triangles
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let (a, b, c) = (t[0], t[1], t[2]);
            let (abx, aby) = (f32::from(b.x - a.x), f32::from(b.y - a.y));
            let (acx, acy) = (f32::from(c.x - a.x), f32::from(c.y - a.y));
            (abx * acy - aby * acx).abs() / 2.
        })
        .sum()
}

fn xs(path: &BuiltPath) -> Vec<f32> {
    let mut v: Vec<f32> = path.triangles.iter().map(|q| f32::from(q.x)).collect();
    v.sort_by(f32::total_cmp);
    v.dedup();
    v
}

#[test]
fn a_stroked_polyline_is_lyons_tessellation_with_its_defaults() {
    let path = polyline(&[(0., 0.), (10., 0.), (10., 10.)], 2., &[]);
    assert_eq!(path.triangles.len() % 3, 0);
    // Butt caps add nothing past the ends; the right-angle miter (lyon's default join) reaches
    // (11, -1); the bounds are the vertices' box, as GPUI's `Path::bounds`.
    assert!(path.triangles.contains(&p(11., -1.)));
    assert_eq!(path.bounds, Bounds::from_corners(p(0., -1.), p(11., 10.)));
    // Two 10 x 2 bands (overlapping in a 1 x 1 square) plus the miter's 1 x 1 corner, no overlap.
    assert!((area(&path) - 40.).abs() < 1e-3, "{}", area(&path));
}

#[test]
fn fills_are_lyons_with_its_even_odd_rule() {
    // Two overlapping squares: even-odd leaves their overlap out.
    let mut b = PathBuilder::fill();
    for (x, y) in [(0., 0.), (2., 2.)] {
        b.move_to(p(x, y));
        for q in [p(x + 4., y), p(x + 4., y + 4.), p(x, y + 4.)] {
            b.line_to(q);
        }
        b.close();
    }
    let path = b.build().unwrap();
    assert!((area(&path) - (16. + 16. - 2. * 4.)).abs() < 1e-3, "{}", area(&path));
    assert_eq!(path.bounds, Bounds::new(p(0., 0.), size(px(6.), px(6.))));

    // After `close` the current point is the contour's start; a lone segment fills nothing.
    let mut b = PathBuilder::fill();
    b.move_to(p(0., 0.));
    for q in [p(4., 0.), p(4., 4.), p(0., 4.)] {
        b.line_to(q);
    }
    b.close();
    b.line_to(p(2., 8.));
    let path = b.build().unwrap();
    assert!((area(&path) - 16.).abs() < 1e-3);
    assert_eq!(path.bounds, Bounds::new(p(0., 0.), size(px(4.), px(4.))));
}

#[test]
fn dashes_cut_the_whole_path_as_gpui_does() {
    let path = polyline(&[(0., 0.), (10., 0.)], 1., &[3., 2.]);
    assert_eq!(xs(&path), vec![0., 3., 5., 8.]);
    assert!((area(&path) - 6.).abs() < 1e-3);

    // An odd list repeats ([6] is [6, 6]); the second dash runs from (4, 8) to (4, 10).
    let path = polyline(&[(0., 0.), (4., 0.), (4., 10.)], 1., &[6.]);
    let second: Vec<_> = path.triangles.iter().filter(|q| f32::from(q.y) > 7.).collect();
    assert!(!second.is_empty());
    let ys: Vec<f32> = second.iter().map(|q| f32::from(q.y)).collect();
    assert!(ys.iter().all(|&y| (y - 8.).abs() < 1e-3 || (y - 10.).abs() < 1e-3), "{ys:?}");
}

#[test]
fn curves_flatten_within_tolerance() {
    let mut b = PathBuilder::stroke(px(1.));
    b.move_to(p(0., 0.));
    b.curve_to(p(20., 0.), p(10., 10.));
    let path = b.build().unwrap();
    assert!(path.triangles.len() > 12, "several segments");
    // The curve's apex is at (10, 5); the stroke reaches half a width past it, the tolerance more.
    let max_y = path.triangles.iter().map(|q| f32::from(q.y)).fold(f32::MIN, f32::max);
    assert!((5.4..5.6).contains(&max_y), "{max_y}");
}

#[test]
fn paint_path_scales_clips_and_fades() {
    let path = polyline(&[(0., 0.), (10., 0.)], 2., &[]);
    let mut scene = Scene::new();
    let cx = StylePaint {
        rem_size: px(16.),
        scale_factor: 1.5,
        opacity: 0.5,
        content_mask: ContentMask { bounds: Bounds::new(p(0., 0.), size(px(100.), px(100.))) },
    };
    paint_path(&mut scene, &cx, &path, red());
    let Primitive::Path(prim) = &scene.primitives[0] else { panic!("a path") };
    assert_eq!(prim.bounds, path.bounds.scale(1.5));
    assert_eq!(prim.paint, red().opacity(0.5).into());
    assert_eq!(*prim.path.kind(), PathKind::Triangles);
    let scaled: Vec<_> = path.triangles.iter().map(|q| q.scale(1.5)).collect();
    assert_eq!(prim.path.triangles(), scaled.as_slice());

    // Gradients pass through; a path outside the mask is skipped; an empty path paints nothing.
    let gradient = linear_gradient(180., linear_color_stop(red(), 0.), linear_color_stop(red(), 1.));
    paint_path(&mut scene, &cx, &path, gradient);
    assert_eq!(scene.len(), 2);
    let far = polyline(&[(200., 200.), (210., 200.)], 2., &[]);
    paint_path(&mut scene, &cx, &far, red());
    paint_path(&mut scene, &cx, &PathBuilder::fill().build().unwrap(), red());
    assert_eq!(scene.len(), 2);
}

#[test]
fn degenerate_dash_arrays_end() {
    // All zero: GPUI's loop never ends; zero-length dashes draw nothing.
    let path = polyline(&[(0., 0.), (10., 0.)], 1., &[0., 0.]);
    assert!(path.triangles.is_empty(), "{path:?}");
    let path = polyline(&[(0., 0.), (10., 0.)], 1., &[0.]);
    assert!(path.triangles.is_empty(), "{path:?}");
    // Zero "on" dashes with a gap draw nothing; zero gaps draw the whole line.
    assert!(polyline(&[(0., 0.), (10., 0.)], 1., &[0., 2.]).triangles.is_empty());
    assert!((area(&polyline(&[(0., 0.), (10., 0.)], 1., &[2., 0.])) - 10.).abs() < 1e-3);
    // Negative and non-finite lengths are rejected.
    for bad in [[-1., 2.], [f32::NAN, 2.], [f32::INFINITY, 1.], [1., f32::NEG_INFINITY]] {
        let mut b = PathBuilder::stroke(px(1.)).dash_array(&bad.map(px));
        b.move_to(p(0., 0.));
        b.line_to(p(10., 0.));
        assert!(b.build().is_err(), "{bad:?}");
    }
}
