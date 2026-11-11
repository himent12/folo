use folo_ui::{
    Bounds, BuiltPath, ContentMask, FillRule, LineCap, LineJoin, PathBuilder, PathContour, PathKind,
    PathMode, Pixels, Point, Primitive, Scene, StylePaint, linear_color_stop, linear_gradient, paint_path,
    point, px, red, size,
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

#[test]
fn a_stroked_polyline_keeps_its_points_and_lyons_defaults() {
    let path = polyline(&[(0., 0.), (10., 0.), (10., 10.)], 2., &[]);
    assert_eq!(
        path.contours,
        vec![PathContour { points: vec![p(0., 0.), p(10., 0.), p(10., 10.)], closed: false }]
    );
    assert_eq!(
        path.mode,
        PathMode::Stroke { width: px(2.), cap: LineCap::Butt, join: LineJoin::Miter, miter_limit: 4. }
    );
    // Butt caps add nothing past the ends; the right-angle miter reaches (11, -1).
    assert_eq!(path.bounds, Bounds::from_corners(p(0., -1.), p(11., 10.)));
}

#[test]
fn a_closed_fill_is_one_closed_contour_with_lyons_even_odd_rule() {
    let mut b = PathBuilder::fill();
    b.move_to(p(0., 0.));
    for q in [p(4., 0.), p(4., 4.), p(0., 4.)] {
        b.line_to(q);
    }
    b.close();
    // After `close` the current point is the contour's start: a `line_to` starts a new contour there.
    b.line_to(p(2., 8.));
    let path = b.build().unwrap();
    assert_eq!(path.mode, PathMode::Fill(FillRule::EvenOdd));
    assert_eq!(path.contours.len(), 2);
    assert_eq!(
        path.contours[0],
        PathContour { points: vec![p(0., 0.), p(4., 0.), p(4., 4.), p(0., 4.)], closed: true }
    );
    assert_eq!(path.contours[1].points, vec![p(0., 0.), p(2., 8.)]);
    assert_eq!(path.bounds, Bounds::new(p(0., 0.), size(px(4.), px(8.))));
}

#[test]
fn dashes_cut_the_whole_path_into_open_contours() {
    let path = polyline(&[(0., 0.), (10., 0.)], 1., &[3., 2.]);
    let dashes: Vec<_> = path.contours.iter().map(|c| (c.points.clone(), c.closed)).collect();
    assert_eq!(dashes, vec![(vec![p(0., 0.), p(3., 0.)], false), (vec![p(5., 0.), p(8., 0.)], false)]);

    // An odd list repeats ([3] is [3, 3]); a dash crossing a corner keeps the corner point.
    let path = polyline(&[(0., 0.), (4., 0.), (4., 10.)], 1., &[6.]);
    let points: Vec<_> = path.contours.iter().map(|c| c.points.clone()).collect();
    assert_eq!(points, vec![vec![p(0., 0.), p(4., 0.), p(4., 2.)], vec![p(4., 8.), p(4., 10.)]]);
}

#[test]
fn curves_flatten_within_tolerance_and_end_on_their_end_point() {
    let mut b = PathBuilder::stroke(px(1.));
    b.move_to(p(0., 0.));
    b.curve_to(p(20., 0.), p(10., 10.));
    let path = b.build().unwrap();
    let pts = &path.contours[0].points;
    assert!(pts.len() > 4, "flattened into several segments");
    assert_eq!(*pts.last().unwrap(), p(20., 0.));
    // The curve's apex is at (10, 5); every flattened point lies on or under it.
    assert!(pts.iter().all(|q| f32::from(q.y) <= 5.0 + 1e-4));
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
    let PathKind::Stroke(stroke) = &prim.path.kind else { panic!("a stroke") };
    assert_eq!((stroke.width, stroke.dash.is_none()), (px(2.).scale(1.5), true));
    assert_eq!(prim.path.contours[0].points, vec![p(0., 0.).scale(1.5), p(10., 0.).scale(1.5)]);

    // Gradients pass through; a path outside the mask is skipped; an empty path paints nothing.
    let gradient = linear_gradient(180., linear_color_stop(red(), 0.), linear_color_stop(red(), 1.));
    paint_path(&mut scene, &cx, &path, gradient);
    assert_eq!(scene.len(), 2);
    let far = polyline(&[(200., 200.), (210., 200.)], 2., &[]);
    paint_path(&mut scene, &cx, &far, red());
    paint_path(&mut scene, &cx, &PathBuilder::fill().build().unwrap(), red());
    assert_eq!(scene.len(), 2);
}
