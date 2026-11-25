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
    // Row 0 is measured once in request_layout (reused in prepaint), then the visible range 0..5 is rendered.
    assert_eq!(*ranges.borrow(), vec![0..1, 0..5]);
    let rows = painted_rows(&scene);
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0], (1, 0.));
    assert_eq!(rows[4], (5, 80.));
    assert_eq!(quads(&scene)[0].bounds, sb(0., 0., 50., 20.));
    assert_eq!(handle.0.borrow().last_item_height, Some(px(20.)));
}

#[test]
fn steady_frame_reuses_layout_and_updates_when_items_or_height_change() {
    use std::cell::Cell;

    let handle = UniformListScrollHandle::new();
    let count = Rc::new(Cell::new(100usize));
    let row_h = Rc::new(Cell::new(20.0f32));
    let row_w = Rc::new(Cell::new(50.0f32));
    let count_render = count.clone();
    let row_h_render = row_h.clone();
    let row_w_render = row_w.clone();
    let handle_render = handle.clone();
    // Auto-sized row container (no fixed width on the list) so taffy queries intrinsic width
    // (`MaxContent`) and height changes both directly affect the list's own laid-out bounds.
    let (mut app, window) = open(move |_, _| {
        let n = count_render.get();
        let h = row_h_render.get();
        let w = row_w_render.get();
        div()
            .flex()
            .flex_row()
            .items_start()
            .child(
                uniform_list("rows", n, move |range, _, _| {
                    range.map(|ix| div().h(px(h)).w(px(w)).bg(rgb(ix as u32 + 1))).collect::<Vec<_>>()
                })
                .max_h(px(100.))
                .track_scroll(handle_render.clone()),
            )
            .into_any_element()
    });
    let list_bounds = || handle.0.borrow().base_handle.0.borrow().bounds;

    // Frame 1: initial layout (100 * 20 capped by max_h 100 -> 50 x 100).
    draw(&mut app, window);
    assert_eq!(list_bounds().size, crate::size(px(50.), px(100.)));

    // Frame 2: unchanged list reuses the root/list layout without recomputing any taffy node.
    app.update(|cx| cx.refresh_windows());
    draw(&mut app, window);
    let (answered, computed) = app.read(|cx| match cx.windows.get(&window.id) {
        Some(Some(w)) => {
            let tree = &w.layout_engine.as_ref().expect("engine").tree;
            (tree.answered_last, tree.computed_last)
        }
        _ => (0, 0),
    });
    assert!(answered > 0 && computed == 0, "expected full reuse, got {answered} / {computed}");
    assert_eq!(list_bounds().size, crate::size(px(50.), px(100.)));

    // Frame 3: shrinking item_count invalidates the MeasureKey and shrinks list height to 2 * 20 = 40 px.
    count.set(2);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(list_bounds().size, crate::size(px(50.), px(40.)));
    assert_eq!(painted_rows(&scene), vec![(1, 0.), (2, 20.)]);

    // Frame 4: changing row height invalidates the MeasureKey and grows list height to 2 * 30 = 60 px.
    row_h.set(30.);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(list_bounds().size, crate::size(px(50.), px(60.)));
    assert_eq!(painted_rows(&scene), vec![(1, 0.), (2, 30.)]);
    assert_eq!(quads(&scene)[0].bounds, sb(0., 0., 50., 30.));

    // Frame 5: changing only intrinsic row width invalidates the MeasureKey and widens list to 80 px.
    row_w.set(80.);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(list_bounds().size, crate::size(px(80.), px(60.)));
    assert_eq!(quads(&scene)[0].bounds, sb(0., 0., 80., 30.));

    // Frame 6: emptying the list shrinks its bounds to 0 x 0.
    count.set(0);
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(list_bounds().size, crate::size(px(0.), px(0.)));
    assert!(painted_rows(&scene).is_empty());
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
