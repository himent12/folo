//! Layout: nested flex boxes land where GPUI's rules put them, at scale 1.0 and 1.5 (taffy in device
//! pixels with its rounding on: relative locations rounded, sizes as rounded right edge minus rounded
//! left edge), and a scroll container moves its children.

use super::*;
use crate::{InteractiveElement, ParentElement, StatefulInteractiveElement, Styled, div, relative};

/// The tree: a 400x300 root (padding 10, border 2, gap 5, column) holding a 21 px row, a 30 px row of
/// [50 px | flex_1 + min_w_0 | 25%] with gap 6, an absolute 10x10 box 4 px from the top right, and a
/// scroll container (flex_1) with three 100 px children.
fn tree(_window: &mut Window, _cx: &mut Context<TestView>) -> AnyElement {
    div()
        .size_full()
        .flex()
        .flex_col()
        .p(px(10.))
        .border_2()
        .border_color(color(20))
        .gap(px(5.))
        .bg(color(0))
        .child(div().h(px(21.)).bg(color(1)))
        .child(
            div()
                .flex()
                .h(px(30.))
                .gap(px(6.))
                .child(div().w(px(50.)).h_full().bg(color(2)))
                .child(div().flex_1().min_w_0().h_full().bg(color(3)))
                .child(div().w(relative(0.25)).h_full().bg(color(4))),
        )
        .child(div().absolute().top(px(4.)).right(px(4.)).w(px(10.)).h(px(10.)).bg(color(5)))
        .child(
            div()
                .id("scroller")
                .flex_1()
                .overflow_y_scroll()
                .bg(color(6))
                .children((0..3).map(|i| div().h(px(100.)).flex_none().bg(color(7 + i)))),
        )
        .into_any_element()
}

#[test]
fn nested_flex_at_scale_1() {
    let (mut app, window) = open(size(px(400.), px(300.)), 1.0, tree);
    let s = 1.0;
    assert_bounds(quad_of(&mut app, window, color(0)), device(0., 0., 400., 300., s));
    // Content box: 2 border + 10 padding on each side.
    assert_bounds(quad_of(&mut app, window, color(1)), device(12., 12., 376., 21., s));
    // Row at 12 + 21 + 5; 25% of 376 = 94; flex_1 takes 376 - 50 - 94 - 2 * 6 = 220.
    assert_bounds(quad_of(&mut app, window, color(2)), device(12., 38., 50., 30., s));
    assert_bounds(quad_of(&mut app, window, color(3)), device(68., 38., 220., 30., s));
    assert_bounds(quad_of(&mut app, window, color(4)), device(294., 38., 94., 30., s));
    // Absolute: from the padding box (inside the border): 2 + 4 down, 398 - 4 - 10 across.
    assert_bounds(quad_of(&mut app, window, color(5)), device(384., 6., 10., 10., s));
    // The scroller gets the rest: 276 - 21 - 30 - 2 gaps (the absolute child takes no gap) = 215.
    assert_bounds(quad_of(&mut app, window, color(6)), device(12., 73., 376., 215., s));
    assert_bounds(quad_of(&mut app, window, color(7)), device(12., 73., 376., 100., s));
    assert_bounds(quad_of(&mut app, window, color(9)), device(12., 273., 376., 100., s));
}

#[test]
fn nested_flex_at_scale_1_5_rounds_to_device_pixels() {
    let (mut app, window) = open(size(px(400.), px(300.)), 1.5, tree);
    let dev = |x: f32, y: f32, w: f32, h: f32| {
        Bounds::new(point(ScaledPixels(x), ScaledPixels(y)), size(ScaledPixels(w), ScaledPixels(h)))
    };
    assert_bounds(quad_of(&mut app, window, color(0)), dev(0., 0., 600., 450.));
    // 21 px is 31.5 device px: the bottom edge 18 + 31.5 rounds to 50, so the height is 32.
    assert_bounds(quad_of(&mut app, window, color(1)), dev(18., 18., 564., 32.));
    // The row starts at 18 + 31.5 + 7.5 = 57 (unrounded positions accumulate).
    assert_bounds(quad_of(&mut app, window, color(2)), dev(18., 57., 75., 45.));
    assert_bounds(quad_of(&mut app, window, color(3)), dev(102., 57., 330., 45.));
    assert_bounds(quad_of(&mut app, window, color(4)), dev(441., 57., 141., 45.));
    assert_bounds(quad_of(&mut app, window, color(5)), dev(576., 9., 15., 15.));
    // The scroller's location 109.5 rounds to 110; its bottom 109.5 + 322.5 = 432 stays.
    assert_bounds(quad_of(&mut app, window, color(6)), dev(18., 110., 564., 322.));
    assert_bounds(quad_of(&mut app, window, color(7)), dev(18., 110., 564., 150.));
}

#[test]
fn wheel_scrolls_clamps_and_moves_children() {
    let (mut app, window) = open(size(px(400.), px(300.)), 1.0, tree);
    // Already at the top: wheel up is clamped and does not redraw.
    wheel(&mut app, window, 100., 150., point(px(0.), px(20.)));
    assert!(!app.is_dirty(window), "clamped wheel up at top does not redraw");
    // Over the scroller; 300 px of content in 215: the most it scrolls is 85.
    wheel(&mut app, window, 100., 150., point(px(0.), px(-40.)));
    assert!(app.is_dirty(window), "a wheel that moved the content redraws");
    app.frame(window);
    assert_bounds(quad_of(&mut app, window, color(7)), device(12., 33., 376., 100., 1.));
    wheel(&mut app, window, 100., 150., point(px(0.), px(-1000.)));
    assert!(app.is_dirty(window));
    app.frame(window);
    assert_bounds(quad_of(&mut app, window, color(7)), device(12., -12., 376., 100., 1.));
    // Clamped: a further wheel down does not move it, and does not redraw.
    wheel(&mut app, window, 100., 150., point(px(0.), px(-10.)));
    assert!(!app.is_dirty(window), "clamped wheel down at bottom does not redraw");
    assert_bounds(quad_of(&mut app, window, color(7)), device(12., -12., 376., 100., 1.));
    // Children are clipped by the scroller (overflow hidden on y).
    let clip = app.scene(window).primitives.iter().find_map(|p| match p {
        Primitive::Quad(q) if q.background.solid == color(7) => Some(q.clip),
        _ => None,
    });
    assert_bounds(clip.unwrap(), device(12., 73., 376., 215., 1.));
    // A wheel outside the scroller does nothing.
    wheel(&mut app, window, 100., 20., point(px(0.), px(100.)));
    assert!(!app.is_dirty(window));
}

#[test]
fn recomputed_layout_invalidates_cached_bounds_for_entire_subtree() {
    use crate::{AvailableSpace, Edges, Length, Style};

    let (mut app, window) = open(size(px(400.), px(300.)), 1.0, |_, _| div().size_full().into_any_element());
    with_window(&mut app, window, |window, cx| {
        let mut child_style = Style::default();
        child_style.size.width = Length::Definite(relative(0.5));
        child_style.size.height = Length::Definite(px(20.).into());
        let child = window.request_layout(child_style, [], cx);

        let mut parent_style = Style::default();
        parent_style.size.width = Length::Definite(relative(1.0));
        parent_style.size.height = Length::Definite(px(50.).into());
        parent_style.padding = Edges::all(px(10.).into());
        let parent = window.request_layout(parent_style, [child], cx);

        window.compute_layout(
            parent,
            size(AvailableSpace::Definite(px(200.)), AvailableSpace::Definite(px(100.))),
            cx,
        );
        // Query child before parent to populate parent via recursion first.
        let child_before = window.layout_bounds(child);
        let parent_before = window.layout_bounds(parent);
        assert_eq!(parent_before, Bounds::new(point(px(0.), px(0.)), size(px(200.), px(50.))));
        assert_eq!(child_before, Bounds::new(point(px(10.), px(10.)), size(px(90.), px(20.))));

        // Recomputing the same root at a new width must invalidate cached bounds for the subtree.
        window.compute_layout(
            parent,
            size(AvailableSpace::Definite(px(400.)), AvailableSpace::Definite(px(100.))),
            cx,
        );
        let child_after = window.layout_bounds(child);
        let parent_after = window.layout_bounds(parent);
        assert_eq!(parent_after, Bounds::new(point(px(0.), px(0.)), size(px(400.), px(50.))));
        assert_eq!(child_after, Bounds::new(point(px(10.), px(10.)), size(px(190.), px(20.))));
    });
}
