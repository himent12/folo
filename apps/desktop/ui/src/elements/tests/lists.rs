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

#[test]
fn empty_padded_scroll_div_has_zero_max_offset_and_bordered_scroll_reaches_bottom() {
    use crate::{
        InteractiveElement, Modifiers, PlatformInput, ScrollDelta, ScrollHandle, ScrollWheelEvent,
        StatefulInteractiveElement, TouchPhase, point,
    };

    let outer = ScrollHandle::new();
    let empty_inner = ScrollHandle::new();
    let bordered = ScrollHandle::new();

    let (mut app, window) = open({
        let (outer, empty_inner, bordered) = (outer.clone(), empty_inner.clone(), bordered.clone());
        move |_, _| {
            div()
                .flex()
                .flex_row()
                .child(
                    div()
                        .id("outer-empty")
                        .w(px(100.))
                        .h(px(100.))
                        .overflow_y_scroll()
                        .track_scroll(&outer)
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .child(
                                    div()
                                        .id("empty-padded")
                                        .w_full()
                                        .h(px(60.))
                                        .p(px(16.))
                                        .overflow_y_scroll()
                                        .track_scroll(&empty_inner),
                                )
                                .child(div().h(px(200.)).w_full()),
                        ),
                )
                .child(
                    div()
                        .id("bordered-scroll")
                        .w(px(100.))
                        .h(px(100.))
                        .border_4()
                        .border_color(rgb(0x444444))
                        .overflow_y_scroll()
                        .track_scroll(&bordered)
                        .child(div().h(px(200.)).w_full().bg(rgb(0x11aa22))),
                )
                .into_any_element()
        }
    });

    draw(&mut app, window);
    assert_eq!(empty_inner.max_offset().height, px(0.), "empty padded div must not scroll");
    // 200px child inside 100px box with 4px top+bottom borders (inner 92px) => max_offset = 108px.
    assert_eq!(bordered.max_offset().height, px(108.));

    // Wheel over the empty padded inner container chains directly to outer without getting swallowed.
    app.simulate_input(
        window,
        PlatformInput::ScrollWheel(ScrollWheelEvent {
            position: point(px(20.), px(20.)),
            delta: ScrollDelta::Pixels(point(px(0.), px(-30.))),
            modifiers: Modifiers::none(),
            touch_phase: TouchPhase::Moved,
        }),
    );
    draw(&mut app, window);
    assert_eq!(empty_inner.offset().y, px(0.));
    assert_eq!(outer.offset().y, px(-30.));

    // Non-finite wheel deltas are ignored and never poison scroll_offset.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        app.simulate_input(
            window,
            PlatformInput::ScrollWheel(ScrollWheelEvent {
                position: point(px(150.), px(50.)),
                delta: ScrollDelta::Pixels(point(px(0.), px(bad))),
                modifiers: Modifiers::none(),
                touch_phase: TouchPhase::Moved,
            }),
        );
        draw(&mut app, window);
        assert_eq!(bordered.offset().y, px(0.), "non-finite delta {bad} must not move offset");
    }
}

#[test]
fn negative_scrolled_child_bounds_do_not_inflate_inner_scroll_or_anchored_size() {
    use crate::{InteractiveElement, ScrollHandle, StatefulInteractiveElement, anchored, point};

    let outer = ScrollHandle::new();
    let inner = ScrollHandle::new();
    let anchored_bounds: Rc<RefCell<Option<crate::Bounds<Pixels>>>> = Rc::new(RefCell::new(None));
    let (mut app, window) = open({
        let (outer, inner, anchored_bounds) = (outer.clone(), inner.clone(), anchored_bounds.clone());
        move |_, _| {
            let ab = anchored_bounds.clone();
            div()
                .id("outer-neg")
                .w(px(200.))
                .h(px(100.))
                .overflow_y_scroll()
                .track_scroll(&outer)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .id("inner-neg")
                                .w_full()
                                .h(px(100.))
                                .overflow_y_scroll()
                                .track_scroll(&inner)
                                .child(div().h(px(150.)).w_full()),
                        )
                        .child(
                            div().relative().h(px(40.)).w_full().child(
                                anchored()
                                    .position(point(px(10.), px(10.)))
                                    .snap_to_window_with_margin(px(0.))
                                    .child(
                                        crate::canvas(
                                            move |b, _, _| *ab.borrow_mut() = Some(b),
                                            |_, _, _, _| {},
                                        )
                                        .w(px(60.))
                                        .h(px(30.)),
                                    ),
                            ),
                        )
                        .child(div().h(px(400.)).w_full()),
                )
                .into_any_element()
        }
    });

    draw(&mut app, window);
    assert_eq!(inner.max_offset().height, px(50.));

    // Scroll outer far down so inner's children sit entirely at negative y (< 0).
    outer.set_offset(point(px(0.), px(-300.)));
    app.update(|cx| cx.refresh_windows());
    draw(&mut app, window);
    assert_eq!(
        inner.max_offset().height,
        px(50.),
        "negative child y coordinates must not inflate content height to 0 - child_min"
    );
    let b = anchored_bounds.borrow().expect("anchored child prepainted");
    assert_eq!(b.origin, point(px(10.), px(10.)), "anchored size stays 30px instead of inflating");
}

#[test]
fn uniform_list_clamps_inverted_visible_range_and_out_of_bounds_deferred_scroll() {
    use std::cell::Cell;

    let handle = UniformListScrollHandle::new();
    let count = Rc::new(Cell::new(5usize));
    let height = Rc::new(Cell::new(0.0f32));
    let ranges = Rc::new(RefCell::new(Vec::<Range<usize>>::new()));

    let (mut app, window) = open({
        let (handle, count, height, ranges) = (handle.clone(), count.clone(), height.clone(), ranges.clone());
        move |_, _| {
            let ranges = ranges.clone();
            let n = count.get();
            div()
                .h(px(height.get()))
                .w(px(80.))
                .child(
                    uniform_list("edge", n, move |range, _, _| {
                        assert!(
                            range.start <= range.end && range.end <= n,
                            "visible_range {range:?} must never invert or exceed item_count"
                        );
                        ranges.borrow_mut().push(range.clone());
                        range.map(|ix| div().h(px(20.)).w_full().bg(rgb(ix as u32 + 1))).collect::<Vec<_>>()
                    })
                    .h_full()
                    .pb(px(24.))
                    .track_scroll(handle.clone()),
                )
                .into_any_element()
        }
    });

    // Collapsed 0px viewport scrolled past the last item into bottom padding: must not invert range.
    let base = handle.0.borrow().base_handle.clone();
    base.set_offset(crate::point(px(0.), px(-124.)));
    draw(&mut app, window);

    // Shrink list to 1 item inside a 100px viewport with 50px bottom padding while requesting an
    // out-of-bounds deferred scroll index: offset must stay clamped to 0.
    height.set(100.);
    count.set(1);
    handle.scroll_to_item(5, ScrollStrategy::Top);
    app.update(|cx| cx.refresh_windows());
    draw(&mut app, window);
    assert_eq!(base.offset().y, px(0.), "deferred scroll clamps to min_scroll_offset..=0");
}

#[test]
fn tooltip_does_not_appear_while_mouse_button_held_or_file_drag_active() {
    use std::cell::Cell;
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::{
        AppContext, ExternalPaths, FileDropEvent, InteractiveElement, Modifiers, MouseButton, MouseDownEvent,
        MouseMoveEvent, MouseUpEvent, PlatformInput, Render, StatefulInteractiveElement, point,
    };

    struct TipView;
    impl Render for TipView {
        fn render(&mut self, _: &mut crate::Window, _: &mut crate::Context<Self>) -> impl IntoElement {
            div().w(px(40.)).h(px(16.)).bg(rgb(0xabcdef))
        }
    }

    let built = Rc::new(Cell::new(0usize));
    let (mut app, window) = open({
        let built = built.clone();
        move |_, _| {
            let built = built.clone();
            div()
                .size_full()
                .child(div().id("tip-target").w(px(80.)).h(px(40.)).bg(rgb(0x222222)).tooltip(
                    move |_, cx| {
                        built.set(built.get() + 1);
                        cx.new(|_| TipView).into()
                    },
                ))
                .into_any_element()
        }
    });
    draw(&mut app, window);

    // 1. Holding left mouse button from outside and dragging into the tooltipped element must not
    // schedule or build a tooltip.
    app.simulate_input(
        window,
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(px(150.), px(150.)),
            modifiers: Modifiers::none(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.simulate_input(
        window,
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(20.), px(20.)),
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::none(),
        }),
    );
    app.advance_clock(Duration::from_millis(700));
    draw(&mut app, window);
    assert_eq!(built.get(), 0, "dragging with a held mouse button must not show tooltips");
    app.simulate_input(
        window,
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: point(px(150.), px(150.)),
            modifiers: Modifiers::none(),
            click_count: 1,
        }),
    );
    draw(&mut app, window);

    // 2. Hovering during an active file drag must not schedule or build a tooltip either.
    app.simulate_input(
        window,
        PlatformInput::FileDrop(FileDropEvent::Entered {
            position: point(px(20.), px(20.)),
            paths: ExternalPaths::new(vec![PathBuf::from("test.png")]),
        }),
    );
    app.advance_clock(Duration::from_millis(700));
    draw(&mut app, window);
    assert_eq!(built.get(), 0, "file drag hover must not schedule or build tooltips");
}
