//! Frames: an idle window draws nothing and asks for nothing, a resize repaints at the new size, animation
//! frames keep the window dirty, scroll handles report and obey, and (manually) what a frame costs.

use std::time::Instant;

use super::*;
use crate::{
    InteractiveElement, ParentElement, RequestFrameOptions, ScrollHandle, StatefulInteractiveElement, Styled,
    div,
};

fn frame_count(app: &mut HeadlessApp, window: AnyWindowHandle) -> u64 {
    with_window(app, window, |window, _| window.frame_times.frame)
}

fn frame_requested(app: &HeadlessApp) -> bool {
    app.platform().windows.borrow()[0].frame_requested.get()
}

#[test]
fn an_idle_window_draws_no_frame_and_requests_none() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    assert!(!app.is_dirty(window));
    app.platform().windows.borrow()[0].frame_requested.set(false);
    let drawn = frame_count(&mut app, window);
    app.run_until_idle();
    with_window(&mut app, window, |window, cx| window.on_request_frame(RequestFrameOptions::default(), cx));
    assert_eq!(frame_count(&mut app, window), drawn, "a frame request on a clean window draws nothing");
    assert!(!frame_requested(&app), "and nothing asks for another");
    // A mouse move over an element with no hover behaviour costs no frame either.
    mouse_move(&mut app, window, 50., 50.);
    assert!(!app.is_dirty(window) && !frame_requested(&app));
    // A refresh asks the platform exactly once.
    with_window(&mut app, window, |window, _| {
        window.refresh();
        window.refresh();
    });
    assert!(app.is_dirty(window) && frame_requested(&app));
}

#[test]
fn resize_repaints_at_the_new_size() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let drawn = frame_count(&mut app, window);
    // What both backends do inside a resize: the callback, then at once a frame that must be presented.
    with_window(&mut app, window, |window, cx| {
        window.on_resize(size(px(300.), px(120.)), 1.5, cx);
        let options = RequestFrameOptions { require_presentation: true, force_render: false };
        window.on_request_frame(options, cx);
    });
    assert_eq!(frame_count(&mut app, window), drawn + 1, "drawn once, synchronously");
    assert_bounds(quad_of(&mut app, window, color(1)), device(0., 0., 300., 120., 1.5));
    let framebuffer = app.frame(window);
    assert_eq!((framebuffer.width, framebuffer.height), (450, 180));
}

#[test]
fn a_window_resized_to_nothing_draws_and_comes_back() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    let present = |app: &mut HeadlessApp, to: Size<Pixels>| {
        with_window(app, window, |window, cx| {
            window.on_resize(to, 1.0, cx);
            let options = RequestFrameOptions { require_presentation: true, force_render: false };
            window.on_request_frame(options, cx);
        });
    };
    // Minimized (both sides zero), then collapsed to one side.
    for to in [size(px(0.), px(0.)), size(px(200.), px(0.)), size(px(0.), px(100.))] {
        present(&mut app, to);
        let framebuffer = app.frame(window);
        assert_eq!(framebuffer.width * framebuffer.height, 0);
    }
    present(&mut app, size(px(200.), px(100.)));
    let framebuffer = app.frame(window);
    assert_eq!((framebuffer.width, framebuffer.height), (200, 100));
    let want = framebuffer.pixel(100, 50);
    assert!(framebuffer.pixels.iter().all(|&p| p == want), "repainted whole");
}

#[test]
fn a_required_presentation_presents_the_whole_frame_even_when_nothing_changed() {
    let (mut app, window) =
        open(size(px(200.), px(100.)), 1.0, |_, _| div().size_full().bg(color(1)).into_any_element());
    app.platform().windows.borrow()[0].last_damage.borrow_mut().clear();
    let drawn = frame_count(&mut app, window);
    with_window(&mut app, window, |window, cx| {
        let options = RequestFrameOptions { require_presentation: true, force_render: false };
        window.on_request_frame(options, cx);
    });
    assert_eq!(frame_count(&mut app, window), drawn, "nothing was dirty: no draw");
    let damage = app.platform().windows.borrow()[0].last_damage.borrow().clone();
    assert_eq!(damage.len(), 1);
    assert_eq!((damage[0].size.width.0, damage[0].size.height.0), (200, 100));
}

#[test]
fn animation_frames_keep_the_window_dirty_until_they_stop() {
    let frames_left = Rc::new(std::cell::Cell::new(2));
    let (mut app, window) = open(size(px(100.), px(100.)), 1.0, {
        let frames_left = frames_left.clone();
        move |window, _| {
            if frames_left.get() > 0 {
                frames_left.set(frames_left.get() - 1);
                window.request_animation_frame();
            }
            div().size_full().into_any_element()
        }
    });
    assert!(app.is_dirty(window), "the first frame asked for another");
    app.frame(window);
    assert!(app.is_dirty(window));
    app.frame(window);
    assert!(!app.is_dirty(window), "no request in the last frame: idle");
}

#[test]
fn scroll_handle_reports_bounds_and_scrolls_to_items() {
    let handle = ScrollHandle::new();
    let (mut app, window) = open(size(px(200.), px(100.)), 1.0, {
        let handle = handle.clone();
        move |_, _| {
            div()
                .id("list")
                .size_full()
                .overflow_y_scroll()
                .track_scroll(&handle)
                .children((0..10).map(|i| div().h(px(30.)).flex_none().bg(color(i))))
                .into_any_element()
        }
    });
    assert_eq!(handle.max_offset(), size(px(0.), px(200.)));
    assert_eq!(handle.bounds(), Bounds::new(point(px(0.), px(0.)), size(px(200.), px(100.))));
    assert_eq!(handle.bounds_for_item(3).unwrap().origin.y, px(90.));
    // Item 5 (150..180) into a 100 px view: its bottom aligns with the bottom.
    handle.scroll_to_item(5);
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(-80.));
    assert_bounds(quad_of(&mut app, window, color(5)), device(0., 70., 200., 30., 1.));
    handle.scroll_to_bottom();
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(-200.));
    handle.set_offset(point(px(0.), px(50.)));
    with_window(&mut app, window, |window, _| window.refresh());
    app.frame(window);
    assert_eq!(handle.offset().y, px(0.), "clamped at the next prepaint");
}

/// Measures the pipeline on a mid-sized tree: 40 rows of 12 cells (about 530 divs, 480 with hover styles
/// and listeners). Run with `cargo test -p folo-ui --release --features test-support frame_cost -- --ignored
/// --nocapture`.
#[test]
#[ignore = "measurement: run by hand in release"]
fn frame_cost_of_a_mid_sized_tree() {
    let (mut app, window) = open(size(px(1600.), px(1000.)), 1.5, |_, _| {
        div()
            .size_full()
            .flex()
            .flex_col()
            .p(px(8.))
            .gap(px(2.))
            .children((0..40).map(|row| {
                div().id(("row", row)).flex().gap(px(4.)).h(px(20.)).children((0..12).map(move |col| {
                    div()
                        .id(("cell", col))
                        .flex_1()
                        .h_full()
                        .rounded(px(3.))
                        .border_1()
                        .border_color(color(1))
                        .bg(color((row + col) % 7))
                        .hover(|s| s.bg(color(9)))
                        .on_click(|_, _, _| {})
                }))
            }))
            .into_any_element()
    });
    for _ in 0..20 {
        with_window(&mut app, window, |window, _| window.refresh());
        app.frame(window);
    }
    let runs = 200;
    let mut totals = crate::FrameTimes::default();
    let start = Instant::now();
    for _ in 0..runs {
        with_window(&mut app, window, |window, _| window.refresh());
        app.frame(window);
        let t = with_window(&mut app, window, |window, _| window.last_frame_times());
        totals.build_us += t.build_us;
        totals.layout_us += t.layout_us;
        totals.prepaint_us += t.prepaint_us;
        totals.paint_us += t.paint_us;
        totals.finish_us += t.finish_us;
        totals.raster_us += t.raster_us;
        totals.present_us += t.present_us;
        totals.primitives = t.primitives;
        totals.hitboxes = t.hitboxes;
    }
    let wall = start.elapsed().as_micros() as u32 / runs;
    let r = runs;
    eprintln!(
        "mid-sized tree ({} prims, {} hitboxes), mean over {r} frames: build {} layout {} prepaint {} paint {} finish {} raster {} present {} us; wall {} us/frame (harness included)",
        totals.primitives,
        totals.hitboxes,
        totals.build_us / r,
        totals.layout_us / r,
        totals.prepaint_us / r,
        totals.paint_us / r,
        totals.finish_us / r,
        totals.raster_us / r,
        totals.present_us / r,
        wall,
    );
}
