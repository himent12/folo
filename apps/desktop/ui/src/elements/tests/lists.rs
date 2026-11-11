//! `uniform_list`: measuring one row, rendering only the visible range, `scroll_to_item`.

use std::{cell::RefCell, ops::Range, rc::Rc};

use super::{draw, open, quads, sb};
use crate::{
    IntoElement, ParentElement, Pixels, ScrollStrategy, Styled, UniformListScrollHandle, div, px, rgb,
    uniform_list,
};

/// A 100-row list (20 px rows) in a 100 px tall box; returns the ranges rendered.
fn list_app(
    handle: UniformListScrollHandle,
) -> (crate::HeadlessApp, crate::AnyWindowHandle, Rc<RefCell<Vec<Range<usize>>>>) {
    let ranges = Rc::new(RefCell::new(Vec::new()));
    let ranges2 = ranges.clone();
    let (app, window) = open(move |_, _| {
        let ranges = ranges2.clone();
        div()
            .h(px(100.))
            .w(px(50.))
            .child(
                uniform_list("rows", 100, move |range, _, _| {
                    ranges.borrow_mut().push(range.clone());
                    range.map(|ix| div().h(px(20.)).w_full().bg(rgb(ix as u32 + 1))).collect::<Vec<_>>()
                })
                .h_full()
                .track_scroll(handle.clone()),
            )
            .into_any_element()
    });
    (app, window, ranges)
}

fn painted_rows(scene: &crate::Scene) -> Vec<(u32, f32)> {
    quads(scene)
        .iter()
        .map(|q| {
            let rgba: crate::Rgba = q.background.solid.into();
            (
                ((rgba.r * 255.).round() as u32) << 16
                    | ((rgba.g * 255.).round() as u32) << 8
                    | (rgba.b * 255.).round() as u32,
                q.bounds.origin.y.0,
            )
        })
        .collect()
}

#[test]
fn renders_only_the_visible_rows() {
    let handle = UniformListScrollHandle::new();
    let (mut app, window, ranges) = list_app(handle.clone());
    let scene = draw(&mut app, window);
    // Row 0 is measured (twice: layout and prepaint), then the visible range 0..5 is rendered.
    assert_eq!(ranges.borrow().last().cloned(), Some(0..5));
    let rows = painted_rows(&scene);
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], (1, 0.));
    assert_eq!(rows[4], (5, 80.));
    assert_eq!(quads(&scene)[0].bounds, sb(0., 0., 50., 20.));
    assert_eq!(handle.0.borrow().last_item_height, Some(px(20.)));
}

#[test]
fn scroll_to_item_top_and_bottom() {
    let handle = UniformListScrollHandle::new();
    let (mut app, window, ranges) = list_app(handle.clone());
    draw(&mut app, window);

    handle.scroll_to_item(50, ScrollStrategy::Top);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(ranges.borrow().last().cloned(), Some(50..55));
    assert_eq!(painted_rows(&scene)[0], (51, 0.), "row 50 at the top");
    assert_eq!(*handle.0.borrow().base_handle.0.borrow().offset.borrow(), crate::point(px(0.), px(-1000.)));

    // Already visible: nothing moves.
    handle.scroll_to_item(52, ScrollStrategy::Bottom);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(painted_rows(&scene)[0], (51, 0.));

    // Above the view, placed at the bottom.
    handle.scroll_to_item(10, ScrollStrategy::Bottom);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(ranges.borrow().last().cloned(), Some(6..11));
    let rows = painted_rows(&scene);
    assert_eq!(rows.last().cloned(), Some((11, 80.)), "row 10 ends at the list's bottom");
}

#[test]
fn scroll_to_item_center_and_clamping() {
    use crate::elements::uniform_list::ListGeometry;
    let geometry = ListGeometry {
        item_height: px(20.),
        padding: Default::default(),
        list_height: px(100.),
        content_height: px(2000.),
    };
    let at = |ix, strategy, y: f32| geometry.scroll_to_item(ix, strategy, px(y));
    assert_eq!(at(50, ScrollStrategy::Center, 0.), px(-960.), "row centre at the list centre");
    assert_eq!(at(99, ScrollStrategy::Top, 0.), px(-1900.), "clamped to the content end");
    assert_eq!(at(0, ScrollStrategy::Bottom, -500.), px(0.), "clamped to the start");
    assert_eq!(at(2, ScrollStrategy::Top, 0.), px(0.), "visible rows do not move the list");
    let _: Pixels = at(1, ScrollStrategy::Top, 0.);
}
