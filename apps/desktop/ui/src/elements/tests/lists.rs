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

#[test]
fn uniform_list_reuses_scratch_pool_and_skips_redraw_on_clamped_wheel() {
    use crate::{PlatformInput, ScrollDelta, ScrollWheelEvent, point};

    let handle = UniformListScrollHandle::new();
    let (mut app, window, _) = list_app(handle.clone());
    draw(&mut app, window);

    // Scratch buffer was returned to the window's pool after painting frame 1.
    let pool_state = app.read(|cx| match cx.windows.get(&window.id) {
        Some(Some(w)) => (w.element_scratch_pool.len(), w.element_scratch_pool.first().map(Vec::capacity)),
        _ => (0, None),
    });
    assert_eq!(pool_state.0, 1);
    assert!(pool_state.1.unwrap_or(0) >= 5);

    let wheel = |app: &mut crate::HeadlessApp, dy: f32| {
        app.simulate_input(
            window,
            PlatformInput::ScrollWheel(ScrollWheelEvent {
                position: point(px(25.), px(50.)),
                delta: ScrollDelta::Pixels(point(px(0.), px(dy))),
                ..Default::default()
            }),
        );
    };

    // At top (offset 0): wheel up is a no-op and leaves the window clean.
    wheel(&mut app, 40.);
    assert!(!app.is_dirty(window), "wheel up at top of uniform_list must not redraw");

    // Scroll to the very bottom (100 * 20 - 100 = 1900).
    wheel(&mut app, -5000.);
    assert!(app.is_dirty(window));
    draw(&mut app, window);
    assert_eq!(*handle.0.borrow().base_handle.0.borrow().offset.borrow(), point(px(0.), px(-1900.)));

    // At bottom: further wheel down is clamped and leaves the window clean.
    wheel(&mut app, -40.);
    assert!(!app.is_dirty(window), "wheel down at bottom of uniform_list must not redraw");

    // Pool depth stays 1 across repeated redraws (no leak or unbounded growth).
    let pool_len = app.read(|cx| match cx.windows.get(&window.id) {
        Some(Some(w)) => w.element_scratch_pool.len(),
        _ => 0,
    });
    assert_eq!(pool_len, 1);
}

#[test]
fn uniform_list_truncates_excess_rendered_items_without_panicking() {
    let (mut app, window) = open(move |_, _| {
        div()
            .h(px(40.))
            .w(px(50.))
            .child(
                uniform_list("excess", 10, move |range, _, _| {
                    // Deliberately return extra items beyond `range.len()`.
                    (0..range.len() + 3).map(|ix| div().h(px(20.)).w_full().bg(rgb(ix as u32 + 1))).collect()
                })
                .h_full(),
            )
            .into_any_element()
    });
    let scene = draw(&mut app, window);
    assert_eq!(painted_rows(&scene).len(), 2, "only visible_range.len() rows are laid out and painted");
}

#[test]
fn div_scroll_handle_scrolls_on_first_frame_tracks_container_move_and_clears_empty_children() {
    use std::cell::Cell;

    use crate::{InteractiveElement, ScrollHandle, StatefulInteractiveElement};

    let handle = ScrollHandle::new();
    let count = Rc::new(Cell::new(10usize));
    let top_margin = Rc::new(Cell::new(0.0f32));

    // Request scroll-to-item BEFORE the scrollable container has ever been laid out.
    handle.scroll_to_item(8);

    let (mut app, window) = open({
        let (handle, count, top_margin) = (handle.clone(), count.clone(), top_margin.clone());
        move |_, _| {
            let n = count.get();
            div()
                .size_full()
                .child(
                    div()
                        .id("scroller")
                        .mt(px(top_margin.get()))
                        .w(px(100.))
                        .h(px(60.))
                        .overflow_y_scroll()
                        .track_scroll(&handle)
                        .children((0..n).map(|ix| div().h(px(20.)).w_full().bg(rgb(ix as u32 + 1)))),
                )
                .into_any_element()
        }
    });

    // Frame 1: child 8 (y = 160..180) is scrolled into the 60 px viewport on the very first draw.
    draw(&mut app, window);
    assert_eq!(handle.children_count(), 10);
    assert_eq!(handle.offset().y, px(-120.), "item 8 bottom (180) aligns with viewport bottom (60)");

    // Frame 2: shifting the container down by 50 px updates `bounds` BEFORE resolving scroll_to_item(0).
    top_margin.set(50.);
    handle.scroll_to_item(0);
    app.update(|cx| cx.refresh_windows());
    draw(&mut app, window);
    assert_eq!(handle.offset().y, px(0.), "item 0 scrolls to top using the moved container's bounds");

    // Frame 3: emptying all children clears stale child_bounds instead of retaining the old 10 entries.
    count.set(0);
    app.update(|cx| cx.refresh_windows());
    draw(&mut app, window);
    assert_eq!(handle.children_count(), 0);
    assert!(handle.bounds_for_item(0).is_none());
    assert_eq!(handle.offset().y, px(0.));
}

#[test]
fn uniform_list_with_vertical_padding_and_border_aligns_rows_and_scroll_max() {
    let handle = UniformListScrollHandle::new();
    let (mut app, window) = open({
        let handle = handle.clone();
        move |_, _| {
            div()
                .h(px(100.))
                .w(px(80.))
                .child(
                    uniform_list("padded", 10, |range, _, _| {
                        range.map(|ix| div().h(px(20.)).w_full().bg(rgb(ix as u32 + 1))).collect::<Vec<_>>()
                    })
                    .h_full()
                    .p(px(10.))
                    .border_2()
                    .track_scroll(handle.clone()),
                )
                .into_any_element()
        }
    });

    // Inner border-box height is 100 - 4 = 96, content height is 10 * 20 + 20 = 220 => max scroll = 124.
    let scene = draw(&mut app, window);
    let base = handle.0.borrow().base_handle.clone();
    assert_eq!(base.max_offset().height, px(124.));
    // Row 0 starts at border.top (2) + padding.top (10) = 12, not double-padded at 22.
    let rows: Vec<_> = painted_rows(&scene).into_iter().filter(|&(c, _)| c > 0 && c <= 10).collect();
    assert_eq!(rows[0], (1, 12.));
    assert_eq!(rows[1], (2, 32.));

    // Scroll to the very bottom: last row (id 10) ends at y = 88 (leaving 10 px bottom padding + 2 px border).
    base.set_offset(crate::point(px(0.), px(-200.)));
    app.update(|cx| cx.refresh_windows());
    let scene = draw(&mut app, window);
    assert_eq!(base.offset().y, px(-124.));
    let rows: Vec<_> = painted_rows(&scene).into_iter().filter(|&(c, _)| c > 0 && c <= 10).collect();
    assert_eq!(rows.last().copied(), Some((10, 68.)), "row 9 sits at 68..88 above 10px padding + 2px border");
}

#[test]
fn nested_scroll_containers_prevent_outer_scroll_until_inner_reaches_boundary() {
    use std::cell::Cell;

    use crate::{
        InteractiveElement, Modifiers, PlatformInput, ScrollDelta, ScrollHandle, ScrollWheelEvent,
        StatefulInteractiveElement, TouchPhase, point,
    };

    let outer = ScrollHandle::new();
    let inner = ScrollHandle::new();
    let inner_wheel_hits = Rc::new(Cell::new(0usize));

    let (mut app, window) = open({
        let (outer, inner, inner_wheel_hits) = (outer.clone(), inner.clone(), inner_wheel_hits.clone());
        move |_, _| {
            let hits = inner_wheel_hits.clone();
            div()
                .id("outer")
                .w(px(200.))
                .h(px(120.))
                .overflow_y_scroll()
                .track_scroll(&outer)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .id("inner")
                                .w_full()
                                .h(px(60.))
                                .overflow_y_scroll()
                                .track_scroll(&inner)
                                .on_scroll_wheel(move |_, _, _| {
                                    hits.set(hits.get() + 1);
                                })
                                .child(div().h(px(120.)).w_full()),
                        )
                        .child(div().h(px(200.)).w_full()),
                )
                .into_any_element()
        }
    });

    draw(&mut app, window);
    assert_eq!(inner.max_offset().height, px(60.));
    assert_eq!(outer.max_offset().height, px(140.));

    let wheel = |app: &mut crate::HeadlessApp, dy: f32| {
        app.simulate_input(
            window,
            PlatformInput::ScrollWheel(ScrollWheelEvent {
                position: point(px(20.), px(20.)),
                delta: ScrollDelta::Pixels(point(px(0.), px(dy))),
                modifiers: Modifiers::none(),
                touch_phase: TouchPhase::Moved,
            }),
        );
        draw(app, window);
    };

    // Scrolling inside the inner container consumes the wheel tick and leaves outer untouched,
    // while still running the inner container's own `on_scroll_wheel` listener.
    wheel(&mut app, -30.);
    assert_eq!(inner.offset().y, px(-30.));
    assert_eq!(outer.offset().y, px(0.));
    assert_eq!(inner_wheel_hits.get(), 1);

    // Reaching inner's bottom still consumes that tick.
    wheel(&mut app, -50.);
    assert_eq!(inner.offset().y, px(-60.));
    assert_eq!(outer.offset().y, px(0.));

    // Once inner is already at its boundary, the next wheel tick chains to the outer container.
    wheel(&mut app, -40.);
    assert_eq!(inner.offset().y, px(-60.));
    assert_eq!(outer.offset().y, px(-40.));
}
