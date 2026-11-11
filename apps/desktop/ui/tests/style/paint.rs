use folo_ui::{
    BorderStyle, Bounds, BoxShadow, ContentMask, Corners, DevicePixels, Edges, Hsla, Overflow, Pixels,
    Primitive, Refineable, ScaledPixels, Scene, Style, StylePaint, StyleRefinement, Styled, black, blue,
    hsla, point, px, red, size, svg_sprite_bounds, transparent_black,
};

fn sp(v: f32) -> ScaledPixels {
    v.into()
}

fn sbounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
    Bounds::new(point(sp(x), sp(y)), size(sp(w), sp(h)))
}

fn lbounds(x: f32, y: f32, w: f32, h: f32) -> Bounds<Pixels> {
    Bounds::new(point(px(x), px(y)), size(px(w), px(h)))
}

fn cx(scale_factor: f32, opacity: f32) -> StylePaint {
    StylePaint {
        rem_size: px(16.),
        scale_factor,
        opacity,
        content_mask: ContentMask { bounds: lbounds(0., 0., 1000., 1000.) },
    }
}

fn shadow_color() -> Hsla {
    hsla(0., 0., 0., 0.25)
}

/// Background, 1 px border, 8 px radius and one shadow layer (offset 2 down, blur 4, spread -1).
fn card() -> Style {
    Style::default().refined(
        StyleRefinement::default().bg(red()).border_1().border_color(blue()).rounded(px(8.)).shadow(vec![
            BoxShadow {
                color: shadow_color(),
                offset: point(px(0.), px(2.)),
                blur_radius: px(4.),
                spread_radius: px(-1.),
            },
        ]),
    )
}

fn paint(style: &Style, bounds: Bounds<Pixels>, cx: &StylePaint) -> Scene {
    let mut scene = Scene::new();
    style.paint_before_children(bounds, cx, &mut scene);
    style.paint_after_children(bounds, cx, &mut scene);
    scene
}

/// The primitives GPUI's `Style::paint` makes for `card()` at (10, 20) 100x50, computed by hand at scale `s`.
fn check_card(s: f32) {
    let scene = paint(&card(), lbounds(10., 20., 100., 50.), &cx(s, 1.));
    assert_eq!(scene.len(), 6, "shadow, background, four border bands");
    let mask = sbounds(0., 0., 1000. * s, 1000. * s);

    let Primitive::Shadow(shadow) = &scene.primitives[0] else { panic!("shadow first") };
    // (10, 22) 100x50 dilated by -1.
    assert_eq!(shadow.bounds, sbounds(11. * s, 23. * s, 98. * s, 48. * s));
    assert_eq!(shadow.corner_radii, Corners::all(sp(8. * s)));
    assert_eq!((shadow.blur_radius, shadow.color, shadow.clip), (sp(4. * s), shadow_color(), mask));

    let Primitive::Quad(bg) = &scene.primitives[1] else { panic!("background second") };
    let outer = sbounds(10. * s, 20. * s, 100. * s, 50. * s);
    assert_eq!((bg.bounds, bg.clip), (outer, mask));
    assert_eq!(bg.background, red().into());
    assert_eq!(bg.border_widths, Edges::default());
    assert_eq!(bg.border_color, Hsla { a: 0., ..red() });
    assert_eq!(bg.corner_radii, Corners::all(sp(8. * s)));

    // Bands 8 deep (max of border 1 and radius 8): top, right, bottom, left.
    let bands = [
        sbounds(10. * s, 20. * s, 100. * s, 8. * s),
        sbounds(109. * s, 28. * s, 1. * s, 34. * s),
        sbounds(10. * s, 62. * s, 100. * s, 8. * s),
        sbounds(10. * s, 28. * s, 1. * s, 34. * s),
    ];
    for (primitive, band) in scene.primitives[2..].iter().zip(bands) {
        let Primitive::Quad(q) = primitive else { panic!("border quads last") };
        assert_eq!((q.bounds, q.clip), (outer, band));
        assert_eq!(q.border_widths, Edges::all(sp(s)));
        assert_eq!((q.border_color, q.background), (blue(), Hsla { a: 0., ..blue() }.into()));
        assert_eq!((q.corner_radii, q.border_style), (Corners::all(sp(8. * s)), BorderStyle::Solid));
    }
}

#[test]
fn card_paints_gpuis_primitives_at_scale_1() {
    check_card(1.0);
}

#[test]
fn card_paints_gpuis_primitives_at_scale_1_5() {
    check_card(1.5);
}

#[test]
fn radii_clamp_opacity_fades_and_masks_clip() {
    let style = Style::default().refined(StyleRefinement::default().bg(red()).rounded_full().shadow_sm());
    let scene = paint(&style, lbounds(0., 0., 40., 10.), &cx(1., 0.5));
    // Two shadow layers, then the background; no border (none set).
    assert_eq!(scene.len(), 3);
    let Primitive::Quad(bg) = &scene.primitives[2] else { panic!() };
    assert_eq!(bg.corner_radii, Corners::all(sp(5.)), "9999 px clamps to half the short side");
    assert_eq!(bg.background, red().opacity(0.5).into());
    let Primitive::Shadow(first) = &scene.primitives[0] else { panic!() };
    assert_eq!(first.color, hsla(0., 0., 0., 0.05));

    // A content mask that misses the element: nothing is pushed.
    let mut away = cx(1., 1.);
    away.content_mask = ContentMask { bounds: lbounds(500., 500., 10., 10.) };
    assert!(paint(&card(), lbounds(10., 20., 100., 50.), &away).is_empty());

    // The clip is the intersected content mask; a transparent background paints no quad.
    let mut narrow = cx(2., 1.);
    narrow.content_mask = ContentMask { bounds: lbounds(0., 0., 30., 30.) };
    let scene = paint(&card(), lbounds(10., 20., 100., 50.), &narrow);
    let Primitive::Quad(top) = &scene.primitives[2] else { panic!() };
    assert_eq!(top.clip, sbounds(20., 40., 40., 16.));
    let clear = Style::default().refined(StyleRefinement::default().bg(transparent_black()));
    assert!(paint(&clear, lbounds(0., 0., 10., 10.), &cx(1., 1.)).is_empty());
}

#[test]
fn overflow_mask_insets_by_visible_borders_on_hidden_axes() {
    let b = lbounds(10., 10., 100., 50.);
    assert_eq!(Style::default().overflow_mask(b, px(16.)), None);

    let mut s = card();
    s.overflow.y = Overflow::Hidden;
    // GPUI's arithmetic, kept: with x visible and y hidden the mask takes the border-inset x range and
    // the full y range.
    assert_eq!(s.overflow_mask(b, px(16.)).unwrap().bounds, lbounds(11., 10., 98., 50.));
    s.overflow.x = Overflow::Hidden;
    assert_eq!(s.overflow_mask(b, px(16.)).unwrap().bounds, lbounds(11., 11., 98., 48.));
    // A transparent border does not inset.
    s.border_color = Some(Hsla { a: 0., ..black() });
    assert_eq!(s.overflow_mask(b, px(16.)).unwrap().bounds, b);
}

#[test]
fn svg_sprites_center_on_their_bounds() {
    // A 16 px icon at (2, 3) on a 1.5x display: 24 device px, a 48 px mask, origin rounded (4.5 -> 5).
    let bounds = lbounds(2., 3., 16., 16.).scale(1.5);
    let sprite = svg_sprite_bounds(bounds, size(DevicePixels(48), DevicePixels(48)));
    assert_eq!(sprite, sbounds(3., 5., 24., 24.));
}
